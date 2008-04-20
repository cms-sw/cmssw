//
// $Id: PATTauProducer.h,v 1.5 2008/03/05 14:56:50 fronga Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauProducer_h
#define PhysicsTools_PatAlgos_PATTauProducer_h

/**
  \class    pat::PATTauProducer PATTauProducer.h "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
  \brief    Produces pat::Tau's

   The PATTauProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauProducer.h,v 1.5 2008/03/05 14:56:50 fronga Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

#include <string>


namespace pat {


  class ObjectResolutionCalc;
  class LeptonLRCalc;


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
      bool          addLRValues_;
      std::string   tauLRFile_;
      // tools
      ObjectResolutionCalc * theResoCalc_;
      LeptonLRCalc *         theLeptonLRCalc_;
      GreaterByPt<Tau>       pTTauComparator_;

  };


}

#endif
