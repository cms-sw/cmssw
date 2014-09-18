#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

namespace Phase2Tracker {
  
  class Phase2TrackerHeaderProducer : public edm::EDProducer {
     public:
        explicit Phase2TrackerHeaderProducer(const edm::ParameterSet&);
        ~Phase2TrackerHeaderProducer();
        void produce( edm::Event& event, const edm::EventSetup& es );
  
     private:
        edm::EDGetTokenT<FEDRawDataCollection> token_;
  };

} // end Phase2tracker Namespace  
