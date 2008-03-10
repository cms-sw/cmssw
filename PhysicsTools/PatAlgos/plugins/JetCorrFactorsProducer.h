//
// $Id: JetCorrFactorsProducer.h,v 1.1 2008/03/07 18:52:56 lowette Exp $
//

#ifndef PhysicsTools_PatAlgos_JetCorrFactorsProducer_h
#define PhysicsTools_PatAlgos_JetCorrFactorsProducer_h

/**
  \class    pat::JetCorrFactorsProducer JetCorrFactorsProducer.h "PhysicsTools/PatAlgos/interface/JetCorrFactorsProducer.h"
  \brief    Produces JetCorrFactors and a ValueMap to the originating
            reco jets

   The JetCorrFactorsProducer produces a set of correction factors,
   defined in the class pat::JetCorrFactors. The vector of these
   factors is linked to the originating reco jets through a ValueMap. This
   production of associated correction factors is to be done in the PAT Layer-0.
   This ValueMap is then again collapsed inside the pat::Jet when it is
   created in the PAT Layer-1.

  \author   Steven Lowette
  \version  $Id: JetCorrFactorsProducer.h,v 1.1 2008/03/07 18:52:56 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>


namespace pat {


  class JetCorrFactorsProducer : public edm::EDProducer {

    typedef edm::ValueMap<pat::JetCorrFactors> JetCorrFactorsMap;

    public:

      explicit JetCorrFactorsProducer(const edm::ParameterSet & iConfig);
      ~JetCorrFactorsProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag jetsSrc_;
      std::string defaultJetCorrService_;
      std::string udsJetCorrService_;
      std::string gluJetCorrService_;
      std::string cJetCorrService_;
      std::string bJetCorrService_;

  };


}

#endif
