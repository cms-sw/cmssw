#ifndef DQMHttpSource_H
#define DQMHttpSource_H

/** 
 *  An input source for DQM consumers using cmsRun that connect to
 *  the StorageManager or SMProxyServer to get DQM data.
 *
 *  $Id: DQMHttpSource.h,v 1.9 2009/06/10 08:15:25 dshpakov Exp $
 */
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "DQMServices/Core/interface/DQMStore.h"

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

    unsigned int updatesCounter_;

    std::string sourceurl_;
    char DQMeventurl_[256];
    char DQMsubscriptionurl_[256];
    Buf buf_;
    unsigned int events_read_;
    std::string DQMconsumerName_;
    std::string consumerTopFolderName_;
    int headerRetryInterval_;
    double minDQMEventRequestInterval_;
    unsigned int DQMconsumerId_;
    struct timeval lastDQMRequestTime_;

    bool alreadySaidHalted_;

    Strings firstHistoExtractDone_;

    protected:
      DQMStore *bei_;

  };

}

#endif
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
