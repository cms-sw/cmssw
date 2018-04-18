// -*- C++ -*-
//
// Package:     FWCore/Modules
// Class  :     TimeStudyModules
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 22 Mar 2018 16:23:48 GMT
//

// system include files
#include <unistd.h>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


namespace timestudy {
  namespace {
    struct Sleeper {
      Sleeper(edm::ParameterSet const& p, edm::ConsumesCollector&& iCol ) {
        auto const& cv = p.getParameter<std::vector<edm::InputTag>>("consumes");
        tokens_.reserve(cv.size());
        for(auto const& c: cv) {
          tokens_.emplace_back( iCol.consumes<int>(c));
        }
        
        auto const& tv = p.getParameter<std::vector<double>>("eventTimes");
        eventTimes_.reserve(tv.size());
        for(auto t: tv) {
          eventTimes_.push_back( static_cast<useconds_t>(t*1E6));
        }
      }

      void
      getAndSleep(edm::Event const& e) const {
        edm::Handle<int> h;
        for(auto const&t: tokens_) {
          e.getByToken(t,h);
        }
        //Event number minimum value is 1
        usleep( eventTimes_[ (e.id().event()-1) % eventTimes_.size()]);
      }
      
      static void fillDescription(edm::ParameterSetDescription& desc) {
        desc.add<std::vector<edm::InputTag>>("consumes", {})->setComment("What event int data products to consume");
        desc.add<std::vector<double>>("eventTimes")->setComment("The time, in seconds, for how long the module should sleep each event. The index to use is based on a modulo of size of the list applied to the Event ID number.");
      }
      
    private:
      std::vector<edm::EDGetTokenT<int>> tokens_;
      std::vector<useconds_t> eventTimes_;

    };
  }
//--------------------------------------------------------------------
//
// Produces an IntProduct instance.
//
class SleepingProducer : public edm::global::EDProducer<> {
public:
  explicit SleepingProducer(edm::ParameterSet const& p) :
  value_(p.getParameter<int>("ivalue")),
  sleeper_(p, consumesCollector())
  {
    produces<int>();
  }
  void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  int value_;
  Sleeper sleeper_;
};

void
SleepingProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
  // EventSetup is not used.
  sleeper_.getAndSleep(e);
  
  e.put(std::make_unique<int>(value_));
}
  
void
SleepingProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<int>("ivalue")->setComment("Value to put into Event");
  Sleeper::fillDescription(desc);
  
  descriptions.addDefault(desc);
}

  class OneSleepingProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  public:
    explicit OneSleepingProducer(edm::ParameterSet const& p) :
    value_(p.getParameter<int>("ivalue")),
    sleeper_(p, consumesCollector())
    {
      produces<int>();
      usesResource(p.getParameter<std::string>("resource"));
    }
    void produce( edm::Event& e, edm::EventSetup const& c) override;
    
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    
  private:
    int value_;
    Sleeper sleeper_;
  };
  
  void
  OneSleepingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // EventSetup is not used.
    sleeper_.getAndSleep(e);
    
    e.put(std::make_unique<int>(value_));
  }
  
  void
  OneSleepingProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    
    desc.add<int>("ivalue")->setComment("Value to put into Event");
    desc.add<std::string>("resource",std::string())->setComment("The name of the resource that is being shared");
    Sleeper::fillDescription(desc);
    
    descriptions.addDefault(desc);
  }

  class OneSleepingAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit OneSleepingAnalyzer(edm::ParameterSet const& p) :
    sleeper_(p, consumesCollector())
    {
    }
    void analyze( edm::Event const& e, edm::EventSetup const& c) override;
    
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    
  private:
    int value_;
    Sleeper sleeper_;
  };
  
  void
  OneSleepingAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    // EventSetup is not used.
    sleeper_.getAndSleep(e);
  }
  
  void
  OneSleepingAnalyzer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    
    Sleeper::fillDescription(desc);
    
    descriptions.addDefault(desc);
  }

  /*
   The SleepingServer is configured to wait to accumulate X events before starting to run.
   On a call to asyncWork
    -the data will be added to the streams' slot then the waiting thread will be informed
    -if the server is waiting on threads
        - it wakes up and sleeps for 'initTime'
        - it then checks to see if another event was pushed and if it does it continues to do the sleep loop
        - once all sleep are done, it checks to see if enough events have contacted it and if so it sleeps for the longest 'workTime' duration given
           - when done, it sleeps for each event 'finishTime' and when it wakes it sends the callback
           - when all calledback, it goes back to check if there are waiting events
        - if there are not enough waiting events, it goes back to waiting on a condition variable
   
   The SleepingServer keeps track of the number of active Streams by counting the number of streamBeginLumi and streamEndLumi calls have taken place. If there are insufficient active Lumis compared to the number of events it wants to wait for, the Server thread is told to start processing without further waiting.
   
   */
  class SleepingServer {
  public:
    SleepingServer(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR):
    nWaitingEvents_(iPS.getUntrackedParameter<unsigned int>("nWaitingEvents"))
    {
      iAR.watchPreallocate([this](edm::service::SystemBounds const& iBounds) {
        auto const nStreams =iBounds.maxNumberOfStreams();
        waitingStreams_.reserve(nStreams);
        waitTimesPerStream_.resize(nStreams);
        waitingTaskPerStream_.resize(nStreams);
      });

      iAR.watchPreEndJob([this]() {
        stopProcessing_ = true;
        condition_.notify_one();
        serverThread_->join();
      });
      iAR.watchPreStreamBeginLumi([this](edm::StreamContext const&) {
        ++activeStreams_;
      });
      iAR.watchPreStreamEndLumi([this](edm::StreamContext const&) {
        --activeStreams_;
        condition_.notify_one();
      });

      serverThread_ = std::make_unique<std::thread>([this]() { threadWork(); } );
    }
    
    void asyncWork(edm::StreamID id, edm::WaitingTaskWithArenaHolder iTask, long initTime, long workTime, long finishTime) {
      waitTimesPerStream_[id.value()]={{initTime,workTime,finishTime}};
      waitingTaskPerStream_[id.value()]=std::move(iTask);
      {
        std::lock_guard<std::mutex> lk{mutex_};
        waitingStreams_.push_back(id.value());
      }
      condition_.notify_one();
    }
    
  private:
    bool readyToDoSomething() {
      if(stopProcessing_) {
        return true;
      }
      if(waitingStreams_.size() >= nWaitingEvents_) {
        return true;
      }
      //every running stream is now waiting
      return waitingStreams_.size() == activeStreams_;
    }
    
    void threadWork() {
      while(not stopProcessing_.load()) {
        std::vector<int> streamsToProcess;
        {
          std::unique_lock<std::mutex> lk(mutex_);
          condition_.wait(lk, [this]() {
            return readyToDoSomething();
          });
          swap(streamsToProcess,waitingStreams_);
        }
        if(stopProcessing_) {
          break;
        }
        long longestTime = 0;
        //simulate filling the external device
        for(auto i: streamsToProcess) {
          auto const& v=waitTimesPerStream_[i];
          if(v[1]>longestTime) {
            longestTime = v[1];
          }
          usleep(v[0]);
        }
        //simulate running external device
        usleep(longestTime);

        //simulate copying data back
        for(auto i: streamsToProcess) {
          auto const& v=waitTimesPerStream_[i];
          usleep(v[2]);
          waitingTaskPerStream_[i].doneWaiting(std::exception_ptr());
        }
      }
      waitingTaskPerStream_.clear();
    }
    const unsigned int nWaitingEvents_;
    std::unique_ptr<std::thread> serverThread_;
    std::vector<int> waitingStreams_;
    std::vector<std::array<long,3>> waitTimesPerStream_;
    std::vector<edm::WaitingTaskWithArenaHolder> waitingTaskPerStream_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<unsigned int> activeStreams_{0};
    std::atomic<bool> stopProcessing_{false};
  };

  class ExternalWorkSleepingProducer : public edm::global::EDProducer<edm::ExternalWork> {
  public:
    explicit ExternalWorkSleepingProducer(edm::ParameterSet const& p) :
    value_(p.getParameter<int>("ivalue")),
    sleeper_(p, consumesCollector())
    {
      {
        auto const& tv = p.getParameter<std::vector<double>>("serviceInitTimes");
        initTimes_.reserve(tv.size());
        for(auto t: tv) {
          initTimes_.push_back( static_cast<useconds_t>(t*1E6));
        }
      }
      {
        auto const& tv = p.getParameter<std::vector<double>>("serviceWorkTimes");
        workTimes_.reserve(tv.size());
        for(auto t: tv) {
          workTimes_.push_back( static_cast<useconds_t>(t*1E6));
        }
      }
      {
        auto const& tv = p.getParameter<std::vector<double>>("serviceFinishTimes");
        finishTimes_.reserve(tv.size());
        for(auto t: tv) {
          finishTimes_.push_back( static_cast<useconds_t>(t*1E6));
        }
      }
      assert(finishTimes_.size() == initTimes_.size());
      assert(workTimes_.size() == initTimes_.size());

      produces<int>();
    }
    void acquire(edm::StreamID, edm::Event const & e, edm::EventSetup const& c, edm::WaitingTaskWithArenaHolder holder) const override;

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
    
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    
  private:
    std::vector<long> initTimes_;
    std::vector<long> workTimes_;
    std::vector<long> finishTimes_;
    int value_;
    Sleeper sleeper_;
  };

  void
  ExternalWorkSleepingProducer::acquire(edm::StreamID id, edm::Event const& e, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) const {
    // EventSetup is not used.
    sleeper_.getAndSleep(e);
    edm::Service<SleepingServer> server;
    auto index = (e.id().event()-1) % initTimes_.size();
    server->asyncWork(id, std::move(holder), initTimes_[index], workTimes_[index], finishTimes_[index]);
  }

  void
  ExternalWorkSleepingProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    e.put(std::make_unique<int>(value_));
  }
  
  void
  ExternalWorkSleepingProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    
    desc.add<int>("ivalue")->setComment("Value to put into Event");
    desc.add<std::vector<double>>("serviceInitTimes");
    desc.add<std::vector<double>>("serviceWorkTimes");
    desc.add<std::vector<double>>("serviceFinishTimes");
    Sleeper::fillDescription(desc);
    
    descriptions.addDefault(desc);
  }

}
DEFINE_FWK_SERVICE(timestudy::SleepingServer);
DEFINE_FWK_MODULE(timestudy::SleepingProducer);
DEFINE_FWK_MODULE(timestudy::OneSleepingProducer);
DEFINE_FWK_MODULE(timestudy::ExternalWorkSleepingProducer);
DEFINE_FWK_MODULE(timestudy::OneSleepingAnalyzer);

