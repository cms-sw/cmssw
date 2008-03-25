#ifndef STREAMER_EVENTSTREAMHTTPREADER_H
#define STREAMER_EVENTSTREAMHTTPREADER_H

// $Id: EventStreamHttpReader.h,v 1.13 2007/05/16 22:55:03 hcheung Exp $

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace edm
{
  struct ReadData;

  class EventStreamHttpReader : public edm::StreamerInputSource
  {
  public:
    typedef std::vector<char> Buf;

    EventStreamHttpReader(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~EventStreamHttpReader();

    virtual std::auto_ptr<edm::EventPrincipal> read();
    virtual std::auto_ptr<edm::SendJobHeader> readHeader();
    virtual void registerWithEventServer();

  private:  
    std::auto_ptr<edm::EventPrincipal> getOneEvent();

    std::string sourceurl_;
    char eventurl_[256];
    char headerurl_[256];
    char subscriptionurl_[256];
    Buf buf_;
    int events_read_;      // use this until we inherent from the right input source
    int hltBitCount;
    int l1BitCount;
    std::string consumerName_;
    std::string consumerPriority_;
    std::string consumerPSetString_;
    int headerRetryInterval_;
    double minEventRequestInterval_;
    unsigned int consumerId_;
    struct timeval lastRequestTime_;
  };

}
#endif

