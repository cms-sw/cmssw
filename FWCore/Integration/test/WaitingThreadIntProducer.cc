#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"


#include <memory>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <thread>

namespace {
  /* Holds the data the WaitingServer needs to remember for each stream
   */
  struct StreamData {
    StreamData() :
    in_(nullptr), out_(nullptr), task_(nullptr) {}
    //The members are atomic so that a framework thread can
    // be putting new values in them as the server thread is running
    std::atomic<std::vector<int> const*> in_;
    std::atomic<std::vector<int>*>* out_;
    std::atomic<edm::WaitingTask*> task_;
  };
  
  /*
   This class controls a waiting thread. The classes waits until the 
   required number of streams have passed it data (or until a time limit is reached)
   and then does work on all those streams.
   Synchronization between the module on a stream and this server is done
   via a call to requestValuesAsync.
  */
  class WaitingServer {
  public:
    WaitingServer(unsigned int iNumberOfStreams,
                  unsigned int iMinNumberOfStreamsBeforeDoingWork,
                  unsigned int iSecondsToWait):
    m_perStream(iNumberOfStreams),
    m_minNumStreamsBeforeDoingWork(iMinNumberOfStreamsBeforeDoingWork),
    m_secondsToWait(iSecondsToWait),
    m_shouldStop(false),
    m_drainQueue(false) {}
    void start();
    void stop();
    
    ///iIn and iOut must have a lifetime longer than the asynchronous call
    void requestValuesAsync(edm::StreamID, std::vector<int> const* iIn, std::atomic<std::vector<int>*>* iOut, edm::WaitingTask* iWaitingTask);
    
    //These are not used in this example at the moment since there is no
    // way at present to determine that no more events will be processed.
    // In the future a call to endStreamLuminosityBlock could be used
    // as a signal to start draining.
    void drainQueue() {m_drainQueue.store(true);}
    void stopDrainingQueue() {m_drainQueue.store(false);}
    
  private:
    void serverDoWork();
    
    bool readyForWork() const;
    
    edm::WaitingTaskList m_taskList;
    std::mutex m_mutex; //needed by m_cond
    std::condition_variable m_cond;
    std::unique_ptr<std::thread> m_thread;
    std::vector<StreamData> m_perStream;
    std::vector<unsigned int> m_waitingStreams;
    const unsigned int m_minNumStreamsBeforeDoingWork;
    const unsigned int m_secondsToWait;
    std::atomic<bool> m_shouldStop;
    std::atomic<bool> m_drainQueue;
  };
  
  void WaitingServer::requestValuesAsync(edm::StreamID iID,
                                    std::vector<int> const* iIn,
                                    std::atomic<std::vector<int>*>* iOut,
                                    edm::WaitingTask* iWaitingTask) {
    auto& streamData = m_perStream[iID.value()];
    assert(streamData.in_.load() == nullptr);
    
    streamData.in_.store(iIn);
    
    //increment to keep it from running immediately
    iWaitingTask->increment_ref_count();
    streamData.task_.store(iWaitingTask);
    
    std::lock_guard<std::mutex> guard(m_mutex);
    m_waitingStreams.push_back(iID.value());
    streamData.out_=iOut;
    m_cond.notify_one(); //wakes up the server thread
  }
  
  void WaitingServer::stop() {
    m_shouldStop = true;
    m_thread->join();
    m_thread.reset();
  }
  
  void WaitingServer::start() {
    m_thread = std::make_unique<std::thread>([this]() { serverDoWork(); } );
  }
  
  /* Used to determine if there is something for the server thread to do*/
  bool WaitingServer::readyForWork() const {
    if(m_shouldStop) {
      return true;
    }
    if(m_drainQueue or (m_minNumStreamsBeforeDoingWork<= m_waitingStreams.size())) {
      return true;
    }
    return false;
  }
  
  void WaitingServer::serverDoWork() {
    while( not m_shouldStop) {
      std::vector<unsigned int> streamsToUse;
      {
        std::unique_lock<std::mutex> lk(m_mutex);

        //Other threads could have provided work to do
        // before we started wait_for
        if(not readyForWork()) {
          
          //We use wait for to handle the cases where
          // no more events will be sent to the server
          // or where we have a synchronization point
          // where all events must stop processing for a time.
          // In both cases we need to drain the system
          m_cond.wait_for(lk,
                          std::chrono::seconds(m_secondsToWait),
                          [this] ()->bool
          {
            return readyForWork();
          });
        }
        
        if(m_shouldStop) {
          lk.unlock();
          break;
        }
        //Once we know which streams have given us data
        // we can release the lock and let other streams
        // set their data
        streamsToUse.swap(m_waitingStreams);
        lk.unlock();
      }
      
      //Here is the work that the server does for the modules
      // it will just add 1 to each value it has been given
      for(auto index: streamsToUse){
        auto & streamData = m_perStream[index];
        auto task = streamData.task_.exchange(nullptr);
        //release the waiting task for this stream when we are done
        m_taskList.add( task );
        task->decrement_ref_count();
        
        auto out = streamData.out_->load();
        out->clear();
        auto in = streamData.in_.load();
        
        for( auto v: *in) {
          out->push_back(v+1);
        }
        //to be sure memory in other threads will see these changes
        // we will just put the pointer back into the atomic
        streamData.out_->store(out);
        streamData.in_.store(nullptr);
      }
      
      //now inform all waiting tasks that we have done the work
      m_taskList.doneWaiting(std::exception_ptr());
      // reset so that next call to add will wait
      m_taskList.reset();
    }
  }
}

namespace edmtest {

  class WaitingThreadIntProducer : public edm::global::EDProducer<> {
  public:

    explicit WaitingThreadIntProducer(edm::ParameterSet const& iConfig);
    ~WaitingThreadIntProducer() override;
    
    virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    virtual void endJob() override;
    
  private:
    virtual void preallocStreams(unsigned int) override;

    std::vector<edm::EDGetTokenT<IntProduct>> m_tokens;
    std::unique_ptr<WaitingServer> m_server;
    const unsigned int m_numberOfStreamsToAccumulate;
    const unsigned int m_secondsToWaitForWork;
  };

  
  WaitingThreadIntProducer::WaitingThreadIntProducer(edm::ParameterSet const& iConfig):
  m_numberOfStreamsToAccumulate(iConfig.getUntrackedParameter<unsigned int>("streamsToAccumulate")),
  m_secondsToWaitForWork(iConfig.getUntrackedParameter<unsigned int>("secondsToWaitForWork",10U))
  {
    for( auto const& tag: iConfig.getParameter<std::vector<edm::InputTag>>("tags")) {
      m_tokens.emplace_back(consumes<IntProduct>(tag));
    }
    produces<IntProduct>();
  }
  
  WaitingThreadIntProducer::~WaitingThreadIntProducer() {
    if(m_server) {
      m_server->stop();
    }
  }

  void WaitingThreadIntProducer::preallocStreams(unsigned int iNStreams) {
    m_server = std::make_unique<WaitingServer>(iNStreams,
                                               m_numberOfStreamsToAccumulate <=iNStreams? m_numberOfStreamsToAccumulate : iNStreams,
                                               m_secondsToWaitForWork);
    m_server->start();
  }
  
  void WaitingThreadIntProducer::endJob() {
    if(m_server) {
      m_server->stop();
    }
    m_server.reset();
  }


  // Functions that gets called by framework every event
  void WaitingThreadIntProducer::produce(edm::StreamID iID, edm::Event& e, edm::EventSetup const&) const {
    std::vector<int> retrieved;
    
    for(auto const& token: m_tokens) {
      edm::Handle<IntProduct> handle;
      e.getByToken(token, handle);
      retrieved.push_back(handle->value);
    }
    
    std::vector<int> values;
    
    auto taskToWait = edm::make_empty_waiting_task();
    taskToWait->set_ref_count(2);
    std::atomic<std::vector<int>*> sync{&values};
    
    m_server->requestValuesAsync(iID, &retrieved, &sync, taskToWait.get());
    taskToWait->decrement_ref_count();
    
    taskToWait->wait_for_all();
    
    //make sure the memory is actually synced
    auto& tValues = *sync.load();
    
    int sum = 0;
    for(auto v : tValues) {
      sum += v;
    }
    e.put(std::make_unique<IntProduct>(sum));
  }
}

using edmtest::WaitingThreadIntProducer;
DEFINE_FWK_MODULE(WaitingThreadIntProducer);
