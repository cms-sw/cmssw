#ifndef DQMHttpSource_H
#define DQMHttpSource_H

/** 
 *  An input source for DQM consumers using cmsRun that connect to
 *  the StorageManager or SMProxyServer to get DQM data.
 *
 *  $Id: DQMHttpSource.h,v 1.5 2007/07/30 04:50:42 wmtan Exp $
 */
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Sources/interface/RawInputSource.h"

#include <vector>
#include <memory>
#include <string>

namespace edm
{
  class DQMHttpSource : public edm::RawInputSource {

   public:
    typedef std::vector<char> Buf;
    explicit DQMHttpSource(const edm::ParameterSet& pset, 
    const edm::InputSourceDescription& desc);
  
    virtual ~DQMHttpSource() {};


   private:

    std::auto_ptr<edm::Event> getOneDQMEvent();
    virtual std::auto_ptr<edm::Event> readOneEvent();
    virtual void registerWithDQMEventServer();

    void addMonitorable(std::string name, std::string dir_path);
    void setIsDesired(MonitorElementRootFolder *folder, std::string ME_name, bool flag);

    unsigned int updatesCounter_;

    std::string sourceurl_;
    char DQMeventurl_[256];
    char DQMsubscriptionurl_[256];
    Buf buf_;
    unsigned int events_read_;
    std::string DQMconsumerName_;
    std::string DQMconsumerPriority_;
    std::string consumerTopFolderName_;
    int headerRetryInterval_;
    double minDQMEventRequestInterval_;
    unsigned int DQMconsumerId_;
    struct timeval lastDQMRequestTime_;

    bool alreadySaidHalted_;

    protected:
      DaqMonitorROOTBackEnd *bei_;

  };

}

#endif
