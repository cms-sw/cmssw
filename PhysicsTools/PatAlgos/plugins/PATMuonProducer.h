//
// $Id: PATMuonProducer.h,v 1.1.2.1 2008/04/03 13:34:22 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATMuonProducer_h
#define PhysicsTools_PatAlgos_PATMuonProducer_h

/**
  \class    pat::PATMuonProducer PATMuonProducer.h "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
  \brief    Produces pat::Muon's

   The PATMuonProducer produces analysis-level pat::Muon's starting from
   a collection of objects of MuonType.

  \author   Steven Lowette, Roger Wolf
  \version  $Id: PATMuonProducer.h,v 1.1.2.1 2008/04/03 13:34:22 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;
  class LeptonLRCalc;


  class PATMuonProducer : public edm::EDProducer {

    public:

      explicit PATMuonProducer(const edm::ParameterSet & iConfig);
      ~PATMuonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag muonSrc_;
      bool          addGenMatch_;
      edm::InputTag genPartSrc_;
      float         maxDeltaR_;
      float         minRecoOnGenEt_;
      float         maxRecoOnGenEt_;
      bool          addResolutions_;
      bool          useNNReso_;
      std::string   muonResoFile_;
      bool          addMuonID_;
      bool          addLRValues_;
      edm::InputTag tracksSrc_;
      std::string   muonLRFile_;
      // tools
      ObjectResolutionCalc * theResoCalc_;
      LeptonLRCalc         * theLeptonLRCalc_;
      GreaterByPt<Muon>      pTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

  };


}

#endif
