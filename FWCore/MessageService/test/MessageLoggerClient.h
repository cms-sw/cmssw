#ifndef Integration_MessageLoggerClient_h
#define Integration_MessageLoggerClient_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


namespace edm {
  class ParameterSet;
}


namespace edmtest
{

class MessageLoggerClient
  : public edm::EDAnalyzer
{
public:
  explicit
    MessageLoggerClient( edm::ParameterSet const & )
  { }

  
    ~MessageLoggerClient() override
  { }

  
    void analyze( edm::Event      const & e
                , edm::EventSetup const & c
                ) override;

private:
};


}  // namespace edmtest


#endif  // Integration_MessageLoggerClient_h
