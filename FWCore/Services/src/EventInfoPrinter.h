#ifndef EVENTINFOPRINTER_H
#define EVENTINFOPRINTER_H
#include "FWCore/CoreFramework/interface/EDAnalyzer.h"

namespace edm {
  
  class EventInfoPrinter : public edm::EDAnalyzer {
  public:
    EventInfoPrinter( const edm::ParameterSet & );
    ~EventInfoPrinter();
    void analyze( const edm::Event& , const edm::EventSetup& );
  private:
    unsigned long counter;
  };

}

#endif
