#ifndef EgammaIsolationProducers_EgammaRecHitIsolation_h
#define EgammaIsolationProducers_EgammaRecHitIsolation_h
//*****************************************************************************
// File:      EgammaRecHitExtractor.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalExtractor by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

namespace egammaisolation {

   class EgammaRecHitExtractor : public reco::isodeposit::IsoDepositExtractor {
      public:
         EgammaRecHitExtractor(const edm::ParameterSet& par); 
         virtual ~EgammaRecHitExtractor() ; 
         virtual void fillVetos(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) { }
         virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const {
            throw cms::Exception("Configuration Error") << "This extractor " << (typeid(this).name()) << " is not made for tracks";
         }
         virtual reco::IsoDeposit deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & c) const ;

      private:
         void collect(reco::IsoDeposit &deposit, 
                  const GlobalPoint &caloPosition, CaloDualConeSelector &cone, //cone.select is not const. why?
                  const CaloGeometry* caloGeom,
                  const CaloRecHitMetaCollectionV &hits) const;

         double etMin_ ;
         double extRadius_ ;
         double intRadius_ ;
         edm::InputTag barrelRecHitsTag_;
         edm::InputTag endcapRecHitsTag_;
         bool fakeNegativeDeposit_;
         bool  tryBoth_;
         bool  sameTag_;
         bool  useEt_;
         DetId::Detector detector_;

         //edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
         //CaloRecHitMetaCollectionV* caloHits_ ;
   };
}
#endif
