//
// $Id: JetCorrFactorsProducer.h,v 1.10 2010/02/20 21:00:17 wmtan Exp $
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
  \version  $Id: JetCorrFactorsProducer.h,v 1.10 2010/02/20 21:00:17 wmtan Exp $
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESWatcher.h"


#include <string>


namespace pat {


  class JetCorrFactorsProducer : public edm::EDProducer {

    /// sample type for flavor dependend correction
    enum SampleType {kNone, kDijet, kTtbar};
    /// correction type for flavor dependend corrections
    enum CorrType   {kPlain, kFlavor, kParton, kCombined};
    /// correction type for flavor dependend corrections
    enum FlavorType {kMixed, kGluon, kQuark, kCharm, kBeauty};
    /// typedef for jetCorrFactors map
    typedef edm::ValueMap<pat::JetCorrFactors> JetCorrFactorsMap;

    public:

      explicit JetCorrFactorsProducer(const edm::ParameterSet & iConfig);
      ~JetCorrFactorsProducer();
      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      /// configure the constructor strings for the CombinedJetCorrector
      void configure(std::string level, std::string tag);
      /// evaluate the jet correction factor according to level and corrector type
      double evaluate(edm::View<reco::Jet>::const_iterator& jet, FactorizedJetCorrector* corrector, int& idx);
      // create strings for parton and flavor dependend corrections
      std::string flavorTag(CorrType correction, SampleType sample, FlavorType flavor);

    private:

      /// configurables
      bool useEMF_;
      edm::InputTag jetsSrc_;
      std::string jetCorrSet_;

      /// constructor strings for 
      /// the CombinedJetCorrector
      std::string tags_;
      std::string levels_;

      /// module label name 
      std::string moduleLabel_;      


      /// CombinedJetCorrector: common
      FactorizedJetCorrector* jetCorrector_;
      /// CombinedJetCorrector: glu
      FactorizedJetCorrector* jetCorrectorGlu_;
      /// CombinedJetCorrector: uds
      FactorizedJetCorrector* jetCorrectorUds_;
      /// CombinedJetCorrector: c
      FactorizedJetCorrector* jetCorrectorC_;
      /// CombinedJetCorrector: b
      FactorizedJetCorrector* jetCorrectorB_;
      
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl1_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl2_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl3_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl4_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl5_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl6_;
      /// JetCorrectionUncertainty: L1
      JetCorrectionUncertainty * jtuncrtl7_;

      /// Watcher for JetCorrections
      edm::ESWatcher<JetCorrectionsRecord> watchJetCorrectionsRecord_;

  };
}

#endif
