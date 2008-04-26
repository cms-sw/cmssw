#ifndef ONLINEHTTPREADER_H
#define ONLINEHTTPREADER_H

// Online version of EventStreamHttpReader that works with
// the FUEventProcessor XDAQ application
// TODO: create a common source for this and EventStreamHttpReader
//       so we do not duplicate code, and make maintenance easier
// $Id$

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

  class OnlineHttpReader : public edm::StreamerInputSource
  {
  public:
    typedef std::vector<char> Buf;

    OnlineHttpReader(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~OnlineHttpReader();

    virtual std::auto_ptr<edm::EventPrincipal> read();
    virtual std::auto_ptr<edm::SendJobHeader> readHeader();
    virtual void registerWithEventServer();

  private:  
    std::auto_ptr<edm::EventPrincipal> getOneEvent();

    virtual void setRun(RunNumber_t r);

    std::string sourceurl_;
    char eventurl_[256];
    char headerurl_[256];
    char subscriptionurl_[256];
    Buf buf_;
    int hltBitCount;
    int l1BitCount;
    std::string consumerName_;
    std::string consumerPriority_;
    std::string consumerPSetString_;
    int headerRetryInterval_;
    double minEventRequestInterval_;
    unsigned int consumerId_;
    struct timeval lastRequestTime_;
    bool endRunAlreadyNotified_;
    bool runEnded_;
    bool alreadySaidHalted_;
    bool alreadyRegistered_;
    bool alreadyGotHeader_;
    enum
    {
      DEFAULT_MAX_CONNECT_TRIES = 360,
      DEFAULT_CONNECT_TRY_SLEEP_TIME = 10
    };
    int maxConnectTries_;
    int connectTrySleepTime_;
  };

}
#endif

