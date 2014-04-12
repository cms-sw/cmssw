#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFRECORDINSERTER
#define EVENTFILTER_UTILTIES_PLUGINS_EVFRECORDINSERTER

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include "EvffedFillerEP.h"

namespace evf{
  class EvFRecordInserter : public edm::EDAnalyzer
    {
    public:
      
      explicit EvFRecordInserter( const edm::ParameterSet& );
      ~EvFRecordInserter(){};
      
      void analyze(const edm::Event & e, const edm::EventSetup& c);
      
    private:
      EvffedFillerEP ef_; 
      uint32_t evc_;
      uint64_t ehi_;
      uint32_t last_;
      edm::InputTag label_;
    };
}

#endif
