//
// $Id: JetCorrFactorsProducer.h,v 1.3 2008/11/04 14:12:58 auterman Exp $
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
  \version  $Id: JetCorrFactorsProducer.h,v 1.3 2008/11/04 14:12:58 auterman Exp $
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "CondFormats/JetMETObjects/interface/CombinedJetCorrector.h"

#include <string>


namespace pat {


  class JetCorrFactorsProducer : public edm::EDProducer {

    typedef edm::ValueMap<pat::JetCorrFactors> JetCorrFactorsMap;

    public:

      explicit JetCorrFactorsProducer(const edm::ParameterSet & iConfig);
      ~JetCorrFactorsProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      /// configure the constructor strings for the CombinedJetCorrector
      void configure(std::string level, std::string tag);
      /// evaluate the jet correction foactor according to level and corrector type
      double evaluate(edm::View<reco::Jet>::const_iterator& jet, CombinedJetCorrector* corrector, int& idx);

    private:

      /// configurables
      bool useEMF_;
      edm::InputTag jetsSrc_;

      /// constructor strings for 
      /// the CombinedJetCorrector
      std::string tags_;
      std::string levels_;

      /// module label name 
      std::string moduleLabel_;      


      /// CombinedJetCorrector: common
      CombinedJetCorrector* jetCorrector_;
      /// CombinedJetCorrector: glu
      CombinedJetCorrector* jetCorrectorGlu_;
      /// CombinedJetCorrector: uds
      CombinedJetCorrector* jetCorrectorUds_;
      /// CombinedJetCorrector: c
      CombinedJetCorrector* jetCorrectorC_;
      /// CombinedJetCorrector: b
      CombinedJetCorrector* jetCorrectorB_;
  };
}

#endif
