#ifndef EgammaIsolationProducers_EgammaHcalIsolation_h
#define EgammaIsolationProducers_EgammaHcalIsolation_h
//*****************************************************************************
// File:      EgammaHcalExtractor.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"




namespace egammaisolation {

   class EgammaHcalExtractor  : public reco::isodeposit::IsoDepositExtractor {
      public:
         EgammaHcalExtractor ( const edm::ParameterSet& par, edm::ConsumesCollector && iC ) :
           EgammaHcalExtractor(par, iC) {}
         EgammaHcalExtractor ( const edm::ParameterSet& par, edm::ConsumesCollector & iC );

         virtual ~EgammaHcalExtractor() ;

         virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup,
                                 const reco::TrackCollection & tracks) { }
         virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup,
                                             const reco::Track & track) const {
            throw cms::Exception("Configuration Error") <<
                     "This extractor " << (typeid(this).name()) << " is not made for tracks";
         }
         virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup,
                                              const reco::Candidate & c) const ;

      private:
         double extRadius_ ;
         double intRadius_ ;
         double etLow_ ;

         edm::EDGetTokenT<HBHERecHitCollection> hcalRecHitProducerToken_;
   };
}
#endif
