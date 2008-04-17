#ifndef EgammaTowerIsolation_h
#define EgammaTowerIsolation_h

//*****************************************************************************
// File:      EgammaTowerExtractor.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"


namespace egammaisolation {

   class EgammaTowerExtractor  : public reco::isodeposit::IsoDepositExtractor {
      public:
         EgammaTowerExtractor ( const edm::ParameterSet& par ) :
            extRadius2_(par.getParameter<double>("extRadius")),
            intRadius_(par.getParameter<double>("intRadius")),
            etLow_(par.getParameter<double>("etMin")),
            caloTowerTag_(par.getParameter<edm::InputTag>("caloTowers")) 
         { 
            extRadius2_ *= extRadius2_;
         }

         virtual ~EgammaTowerExtractor() ;

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
         double extRadius2_ ;
         double intRadius_ ;
         double etLow_ ;

         edm::InputTag caloTowerTag_;
         //const CaloTowerCollection *towercollection_ ;
   };
}
#endif
