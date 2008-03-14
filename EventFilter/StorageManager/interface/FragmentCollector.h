#ifndef HLT_FRAG_COLL_HPP
#define HLT_FRAG_COLL_HPP

/*
  All buffers passed in on queues are owned by the fragment collector.

  JBK - I think the frag_q is going to need to be a pair of pointers.
  The first is the address of the object that needs to be deleted 
  using the Deleter function.  The second is the address of the buffer
  of information that we manipulate (payload of the received object).

  The code should allow for deleters to be functors or simple functions.
 */

#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "EventFilter/StorageManager/interface/EvtMsgRingBuffer.h"
#include "EventFilter/StorageManager/interface/EventServer.h"
#include "EventFilter/StorageManager/interface/DQMEventServer.h"
#include "EventFilter/StorageManager/interface/ServiceManager.h"
#include "EventFilter/StorageManager/interface/DQMServiceManager.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <vector>
#include <map>
#include <string>

namespace stor
{

  class FragmentCollector
  {
  public:
    typedef void (*Deleter)(void*);
    typedef std::vector<unsigned char> Buffer;

    // This is not the most efficient way to store and manipulate this
    // type of data.  It is like this because there is not much time
    // available to create the prototype.
    typedef std::vector<FragEntry> Fragments;
    typedef std::map<stor::FragKey, Fragments> Collection;

    FragmentCollector(HLTInfo& h, Deleter d,
                      const std::string& config_str="");
    FragmentCollector(std::auto_ptr<HLTInfo>, Deleter d,
                      const std::string& config_str="");
    ~FragmentCollector();

    void start();
    void join();
    void stop();

    edm::EventBuffer& getFragmentQueue() { return *frag_q_; }
    edm::EventBuffer& getCommandQueue() { return *cmd_q_; }
    
    void setEventServer(boost::shared_ptr<EventServer>& es) {
      eventServer_ = es;
      if (eventServer_.get() != NULL && writer_.get() != NULL) {
        eventServer_->setStreamSelectionTable(writer_->getStreamSelectionTable());
      }
    }
    void setInitMsgCollection(boost::shared_ptr<InitMsgCollection>& imColl) { initMsgCollection_ = imColl; }

  private:
    static void run(FragmentCollector*);
    void processFragments();
    void processEvent(FragEntry* msg);
    void processHeader(FragEntry* msg);
    void processDQMEvent(FragEntry* msg);

    edm::EventBuffer* cmd_q_;
    edm::EventBuffer* evtbuf_q_;
    edm::EventBuffer* frag_q_;

    Deleter buffer_deleter_;
    Buffer event_area_;
    Collection fragment_area_;
    boost::shared_ptr<boost::thread> me_;
    const edm::ProductRegistry* prods_; // change to shared_ptr ? 
    stor::HLTInfo* info_;  // cannot be const when using EP_Runner?

  public:

    void setNumberOfFileSystems(int disks)   { disks_        = disks; }
    void setFileCatalog(std::string catalog) { catalog_      = catalog; }
    void setSourceId(std::string sourceId)   { sourceId_     = sourceId; }

    void setCollateDQM(bool collateDQM)
    { dqmServiceManager_->setCollateDQM(collateDQM); }

    void setArchiveDQM(bool archiveDQM)
    { dqmServiceManager_->setArchiveDQM(archiveDQM); }

    void setPurgeTimeDQM(int purgeTimeDQM)
    { dqmServiceManager_->setPurgeTime(purgeTimeDQM);}

    void setReadyTimeDQM(int readyTimeDQM)
    { dqmServiceManager_->setReadyTime(readyTimeDQM);}

    void setFilePrefixDQM(std::string filePrefixDQM)
    { dqmServiceManager_->setFilePrefix(filePrefixDQM);}

    void setUseCompressionDQM(bool useCompressionDQM)
    { dqmServiceManager_->setUseCompression(useCompressionDQM);}

    void setCompressionLevelDQM(int compressionLevelDQM)
    { dqmServiceManager_->setCompressionLevel(compressionLevelDQM);}

    void setDQMEventServer(boost::shared_ptr<DQMEventServer>& es)
    {
      // The auto_ptr still owns the memory after this get()
      if (dqmServiceManager_.get() != NULL) dqmServiceManager_->setDQMEventServer(es);
      DQMeventServer_ = es;
    }
    boost::shared_ptr<DQMEventServer>& getDQMEventServer() { return DQMeventServer_; }

    std::list<std::string>& get_filelist() { return writer_->get_filelist();  }
    std::list<std::string>& get_currfiles() { return writer_->get_currfiles(); }
  private:
    uint32 runNumber_;
    uint32 disks_;
    std::string catalog_;
    std::string sourceId_;

    std::auto_ptr<edm::ServiceManager> writer_;
    std::auto_ptr<stor::DQMServiceManager> dqmServiceManager_;

    boost::shared_ptr<EventServer> eventServer_;
    boost::shared_ptr<DQMEventServer> DQMeventServer_;
    boost::shared_ptr<InitMsgCollection> initMsgCollection_;
  };
}

#endif
