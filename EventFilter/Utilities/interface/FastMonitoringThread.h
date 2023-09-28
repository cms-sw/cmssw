#ifndef EVF_FASTMONITORINGTHREAD
#define EVF_FASTMONITORINGTHREAD

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"  //state enums?

#include <iostream>
#include <memory>

#include <vector>
#include <thread>
#include <mutex>

namespace evf {

  constexpr int nReservedModules = 128;
  constexpr int nSpecialModules = 10;
  constexpr int nReservedPaths = 1;

  namespace FastMonState {
    enum Macrostate;
  }

  class FastMonitoringService;

  template <typename T>
  struct ContainableAtomic {
    ContainableAtomic() : m_value{} {}
    ContainableAtomic(T iValue) : m_value(iValue) {}
    ContainableAtomic(ContainableAtomic<T> const& iOther) : m_value(iOther.m_value.load()) {}
    ContainableAtomic<T>& operator=(T iValue) {
      m_value.store(iValue, std::memory_order_relaxed);
      return *this;
    }
    operator T() { return m_value.load(std::memory_order_relaxed); }

    std::atomic<T> m_value;
  };

  struct FastMonEncoding {
    FastMonEncoding(unsigned int res) : reserved_(res), current_(reserved_), currentReserved_(0) {
      if (reserved_)
        dummiesForReserved_ = new edm::ModuleDescription[reserved_];
      //	  completeReservedWithDummies();
    }
    ~FastMonEncoding() {
      if (reserved_)
        delete[] dummiesForReserved_;
    }
    //trick: only encode state when sending it over (i.e. every sec)
    int encode(const void* add) const {
      std::unordered_map<const void*, int>::const_iterator it = quickReference_.find(add);
      return (it != quickReference_.end()) ? (*it).second : 0;
    }

    //this allows to init path list in beginJob, but strings used later are not in the same memory
    //position. Therefore path address lookup will be updated when snapshot (encode) is called
    //with this we can remove ugly path legend update in preEventPath, but will still need a check
    //that any event has been processed (any path will do)
    int encodeString(const std::string* add) {
      std::unordered_map<const void*, int>::const_iterator it = quickReference_.find((void*)add);
      if (it == quickReference_.end()) {
        //try to match by string content (encode only used
        auto it = quickReferencePreinit_.find(*add);
        if (it == quickReferencePreinit_.end())
          return 0;
        else {
          //overwrite pointer in decoder and add to reference
          decoder_[(*it).second] = (void*)add;
          quickReference_[(void*)add] = (*it).second;
          quickReferencePreinit_.erase(it);
          return encode((void*)add);
        }
      }
      return (*it).second;
    }

    const void* decode(unsigned int index) { return decoder_[index]; }
    void fillReserved(const void* add, unsigned int i) {
      //	  translation_[*name]=current_;
      quickReference_[add] = i;
      if (decoder_.size() <= i)
        decoder_.push_back(add);
      else
        decoder_[currentReserved_] = add;
    }
    void updateReserved(const void* add) {
      fillReserved(add, currentReserved_);
      currentReserved_++;
    }
    void completeReservedWithDummies() {
      for (unsigned int i = currentReserved_; i < reserved_; i++)
        fillReserved(dummiesForReserved_ + i, i);
    }
    void update(const void* add) {
      //	  translation_[*name]=current_;
      quickReference_[add] = current_;
      decoder_.push_back(add);
      current_++;
    }

    void updatePreinit(std::string const& add) {
      //	  translation_[*name]=current_;
      quickReferencePreinit_[add] = current_;
      decoder_.push_back((void*)&add);
      current_++;
    }

    unsigned int vecsize() { return decoder_.size(); }
    std::unordered_map<const void*, int> quickReference_;
    std::unordered_map<std::string, int> quickReferencePreinit_;
    std::vector<const void*> decoder_;
    unsigned int reserved_;
    int current_;
    int currentReserved_;
    edm::ModuleDescription* dummiesForReserved_;
  };

  class FastMonitoringThread {
  public:
    struct MonitorData {
      //fastpath global monitorables
      jsoncollector::IntJ fastMacrostateJ_;
      jsoncollector::DoubleJ fastThroughputJ_;
      jsoncollector::DoubleJ fastAvgLeadTimeJ_;
      jsoncollector::IntJ fastFilesProcessedJ_;
      jsoncollector::DoubleJ fastLockWaitJ_;
      jsoncollector::IntJ fastLockCountJ_;
      jsoncollector::IntJ fastEventsProcessedJ_;

      unsigned int varIndexThrougput_;

      //per stream
      std::vector<unsigned int> microstateEncoded_;
      std::vector<unsigned int> ministateEncoded_;
      std::vector<jsoncollector::AtomicMonUInt*> processed_;
      jsoncollector::IntJ fastPathProcessedJ_;
      std::vector<unsigned int> threadMicrostateEncoded_;
      std::vector<unsigned int> inputState_;

      //tracking luminosity of a stream
      std::vector<unsigned int> streamLumi_;

      //N bins for histograms
      unsigned int macrostateBins_;
      unsigned int ministateBins_;
      unsigned int microstateBins_;
      unsigned int inputstateBins_;

      //global state
      std::atomic<FastMonState::Macrostate> macrostate_;

      //per stream
      std::vector<ContainableAtomic<const std::string*>> ministate_;
      std::vector<ContainableAtomic<const void*>> microstate_;
      std::vector<ContainableAtomic<unsigned char>> microstateAcqFlag_;
      std::vector<ContainableAtomic<const void*>> threadMicrostate_;

      FastMonEncoding encModule_;
      std::vector<FastMonEncoding> encPath_;

      //unsigned int prescaleindex_; // ditto

      MonitorData() : encModule_(nReservedModules) {
        fastMacrostateJ_ = FastMonState::sInit;
        fastThroughputJ_ = 0;
        fastAvgLeadTimeJ_ = 0;
        fastFilesProcessedJ_ = 0;
        fastLockWaitJ_ = 0;
        fastLockCountJ_ = 0;
        fastMacrostateJ_.setName("Macrostate");
        fastThroughputJ_.setName("Throughput");
        fastAvgLeadTimeJ_.setName("AverageLeadTime");
        fastFilesProcessedJ_.setName("FilesProcessed");
        fastLockWaitJ_.setName("LockWaitUs");
        fastLockCountJ_.setName("LockCount");

        fastPathProcessedJ_ = 0;
        fastPathProcessedJ_.setName("Processed");
      }

      //to be called after fast monitor is constructed
      void registerVariables(jsoncollector::FastMonitor* fm, unsigned int nStreams, unsigned int nThreads) {
        //tell FM to track these global variables(for fast and slow monitoring)
        fm->registerGlobalMonitorable(&fastMacrostateJ_, true, &macrostateBins_);
        fm->registerGlobalMonitorable(&fastThroughputJ_, false);
        fm->registerGlobalMonitorable(&fastAvgLeadTimeJ_, false);
        fm->registerGlobalMonitorable(&fastFilesProcessedJ_, false);
        fm->registerGlobalMonitorable(&fastLockWaitJ_, false);
        fm->registerGlobalMonitorable(&fastLockCountJ_, false);

        for (unsigned int i = 0; i < nStreams; i++) {
          jsoncollector::AtomicMonUInt* p = new jsoncollector::AtomicMonUInt;
          *p = 0;
          processed_.push_back(p);
          streamLumi_.push_back(0);
        }

        microstateEncoded_.resize(nStreams);
        ministateEncoded_.resize(nStreams);
        threadMicrostateEncoded_.resize(nThreads);
        inputState_.resize(nStreams);
        for (unsigned int j = 0; j < inputState_.size(); j++)
          inputState_[j] = 0;

        //tell FM to track these int vectors
        fm->registerStreamMonitorableUIntVec("Ministate", &ministateEncoded_, true, &ministateBins_);

        if (nThreads <= nStreams)  //no overlapping in module execution per stream
          fm->registerStreamMonitorableUIntVec("Microstate", &microstateEncoded_, true, &microstateBins_);
        else
          fm->registerStreamMonitorableUIntVec("Microstate", &threadMicrostateEncoded_, true, &microstateBins_);

        fm->registerStreamMonitorableUIntVecAtomic("Processed", &processed_, false, nullptr);

        //input source state tracking (not stream, but other than first item in vector is set to Ignore state)
        fm->registerStreamMonitorableUIntVec("Inputstate", &inputState_, true, &inputstateBins_);

        //global cumulative event counter is used for fast path
        fm->registerFastGlobalMonitorable(&fastPathProcessedJ_);

        //provide vector with updated per stream lumis and let it finish initialization
        fm->commit(&streamLumi_);
      }
    };

    //constructor
    FastMonitoringThread() : m_stoprequest(false) {}

    void resetFastMonitor(std::string const& microStateDefPath, std::string const& fastMicroStateDefPath) {
      std::string defGroup = "data";
      jsonMonitor_ = std::make_unique<jsoncollector::FastMonitor>(microStateDefPath, defGroup, false);
      if (!fastMicroStateDefPath.empty())
        jsonMonitor_->addFastPathDefinition(fastMicroStateDefPath, defGroup, false);
    }

    void start(void (FastMonitoringService::*fp)(), FastMonitoringService* cp) {
      assert(!m_thread);
      m_thread = std::make_shared<std::thread>(fp, cp);
    }
    void stop() {
      if (m_thread.get()) {
        m_stoprequest = true;
        m_thread->join();
        m_thread.reset();
      }
    }

    ~FastMonitoringThread() { stop(); }

  private:
    std::atomic<bool> m_stoprequest;
    std::shared_ptr<std::thread> m_thread;
    MonitorData m_data;
    std::mutex monlock_;

    std::unique_ptr<jsoncollector::FastMonitor> jsonMonitor_;

    friend class FastMonitoringService;
  };
}  //end namespace evf
#endif
