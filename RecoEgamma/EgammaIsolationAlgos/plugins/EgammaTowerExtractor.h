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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"


namespace egammaisolation {

  class EgammaTowerExtractor  : public reco::isodeposit::IsoDepositExtractor {

      public:
         enum HcalDepth{AllDepths=-1,Undefined=0,Depth1=1,Depth2=2};


      public:
         EgammaTowerExtractor ( const edm::ParameterSet& par, edm::ConsumesCollector && iC ) :
           EgammaTowerExtractor(par, iC) {}
         EgammaTowerExtractor ( const edm::ParameterSet& par, edm::ConsumesCollector & iC ) :
            extRadius2_(par.getParameter<double>("extRadius")),
            intRadius_(par.getParameter<double>("intRadius")),
            etLow_(par.getParameter<double>("etMin")),
            caloTowerToken(iC.consumes<CaloTowerCollection>(par.getParameter<edm::InputTag>("caloTowers"))),
	    depth_(par.getParameter<int>("hcalDepth"))
         {
            extRadius2_ *= extRadius2_;
	    //lets just check we have a valid depth
	    //should we throw an exception or just warn and then fail gracefully later?
	    if(depth_!=AllDepths && depth_!=Depth1 && depth_!=Depth2){
	      throw cms::Exception("Configuration Error") << "hcalDepth passed to EgammaTowerExtractor is invalid "<<std::endl;
	    }
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


         edm::EDGetTokenT<CaloTowerCollection> caloTowerToken;
         int depth_;
         //const CaloTowerCollection *towercollection_ ;
   };
}
#endif
