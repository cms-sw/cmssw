//
// $Id: PATTauProducer.h,v 1.2 2008/04/03 13:34:22 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauProducer_h
#define PhysicsTools_PatAlgos_PATTauProducer_h

/**
  \class    pat::PATTauProducer PATTauProducer.h "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
  \brief    Produces pat::Tau's

   The PATTauProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauProducer.h,v 1.2 2008/04/03 13:34:22 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;

  class PATTauProducer : public edm::EDProducer {

    public:

      explicit PATTauProducer(const edm::ParameterSet & iConfig);
      ~PATTauProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag tauSrc_;
      bool          addGenMatch_;
      edm::InputTag genPartSrc_;
      bool          addResolutions_;
      bool          useNNReso_;
      std::string   tauResoFile_;
      // tools
      ObjectResolutionCalc * theResoCalc_;
      GreaterByPt<Tau>       pTTauComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;
  };
}

#endif
