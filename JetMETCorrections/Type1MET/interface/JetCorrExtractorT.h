#ifndef JetMETCorrections_Type1MET_JetCorrExtractorT_h
#define JetMETCorrections_Type1MET_JetCorrExtractorT_h

/** \class JetCorrExtractorT
 *
 * Retrieve jet energy correction factor for
 *  o reco::CaloJets
 *  o reco::PFJets
 *  o pat::Jets (of either PF-type or Calo-type)
 *
 * NOTE: this "general" implementation is to be used for reco::CaloJets and reco::PFJets, **not** for pat::Jets
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: JetCorrExtractorT.h,v 1.1 2011/09/13 14:35:34 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

template <typename T>
class JetCorrExtractorT
{
 public:

  reco::Candidate::LorentzVector operator()(const T& rawJet, const std::string& jetCorrLabel, 
					    const edm::Event* evt = 0, const edm::EventSetup* es = 0, 
					    const edm::RefToBase<reco::Jet>* rawJetRef = 0, 
					    double jetCorrEtaMax = 9.9,
					    const reco::Candidate::LorentzVector* rawJetP4_specified = 0)
  {
    // "general" implementation requires access to edm::Event and edm::EventSetup,
    // only specialization for pat::Jets doesn't
    assert(evt && es && rawJetRef);

    // allow to specify four-vector to be used as "raw" (uncorrected) jet momentum,
    // call 'rawJet.p4()' in case four-vector not specified explicitely
    reco::Candidate::LorentzVector rawJetP4 = ( rawJetP4_specified ) ?
      (*rawJetP4_specified) : rawJet.p4();
    
    const JetCorrector* jetCorrector = JetCorrector::getJetCorrector(jetCorrLabel, *es);
    if ( !jetCorrector )  
      throw cms::Exception("JetCorrExtractor")
	<< "Failed to access Jet corrections for = " << jetCorrLabel << " !!\n";

    reco::Candidate::LorentzVector corrJetP4 = rawJetP4;

    // restrict computation of JEC factors to region |eta| < 4.7,
    // to work around problem with CMSSW_4_2_x JEC factors at high eta,
    // reported in
    //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
    //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
    if ( fabs(rawJetP4.eta()) < jetCorrEtaMax ) {      
      //corrJetP4 *= jetCorrector->correction(rawJet, *rawJetRef, *evt, *es);
      corrJetP4 *= jetCorrector->correction(rawJet, *evt, *es);
    } else {
      reco::Candidate::PolarLorentzVector modJetPolarP4(rawJetP4);
      modJetPolarP4.SetEta(jetCorrEtaMax);
      
      reco::Candidate::LorentzVector modJetP4(modJetPolarP4);
      
      T modJet(rawJet);
      modJet.setP4(modJetP4);
      
      //corrJetP4 *= jetCorrector->correction(modJet, *rawJetRef, *evt, *es);
      corrJetP4 *= jetCorrector->correction(modJet, *evt, *es);
    }

    return corrJetP4;
  }
};

#endif
