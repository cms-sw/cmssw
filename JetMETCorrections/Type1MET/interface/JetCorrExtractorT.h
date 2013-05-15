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
 * \version $Revision: 1.5 $
 *
 * $Id: JetCorrExtractorT.h,v 1.5 2012/04/19 17:55:52 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace
{
  template <typename T>
  double getCorrection(const T& rawJet, const std::string& jetCorrLabel, 
		       const edm::Event& evt, const edm::EventSetup& es)
  {
    const JetCorrector* jetCorrector = JetCorrector::getJetCorrector(jetCorrLabel, es);
    if ( !jetCorrector )  
      throw cms::Exception("JetCorrExtractor")
	<< "Failed to access Jet corrections for = " << jetCorrLabel << " !!\n";
    return jetCorrector->correction(rawJet, evt, es);
  }

  double sign(double x)
  {
    if      ( x > 0. ) return +1.;
    else if ( x < 0. ) return -1.;
    else               return  0.;
  }
}

template <typename T>
class JetCorrExtractorT
{
 public:

  reco::Candidate::LorentzVector operator()(const T& rawJet, const std::string& jetCorrLabel, 
					    const edm::Event* evt = 0, const edm::EventSetup* es = 0, 
					    double jetCorrEtaMax = 9.9, 
					    const reco::Candidate::LorentzVector* rawJetP4_specified = 0)
  {
    //std::cout << "<JetCorrExtractor::operator()>:" << std::endl;
    //std::cout << " jetCorrLabel = " << jetCorrLabel << std::endl;

    // "general" implementation requires access to edm::Event and edm::EventSetup,
    // only specialization for pat::Jets doesn't
    assert(evt && es);

    // allow to specify four-vector to be used as "raw" (uncorrected) jet momentum,
    // call 'rawJet.p4()' in case four-vector not specified explicitely
    reco::Candidate::LorentzVector rawJetP4 = ( rawJetP4_specified ) ?
      (*rawJetP4_specified) : rawJet.p4();

    double jetCorrFactor = 1.;
    if ( fabs(rawJetP4.eta()) < jetCorrEtaMax ) {      
      if ( fabs(rawJetP4.E() - rawJet.energy()) > (1.e-3*rawJet.energy()) ) {
	T modJet(rawJet);
	modJet.setP4(rawJetP4);
	//std::cout << "modJet: Pt = " << modJet.pt() << ", eta = " << modJet.eta() << ", phi = " << modJet.phi() 
	//	    << " (Px = " << modJet.px() << ", Py = " << modJet.py() << ")" << std::endl;
	
	jetCorrFactor = getCorrection(modJet, jetCorrLabel, *evt, *es);
	//std::cout << "--> jetCorrFactor(case 3) = " << jetCorrFactor << std::endl;
      } else {
	//std::cout << "rawJet: Pt = " << rawJet.pt() << ", eta = " << rawJet.eta() << ", phi = " << rawJet.phi() 
	//	    << " (Px = " << rawJet.px() << ", Py = " << rawJet.py() << ")" << std::endl;
	
	jetCorrFactor = getCorrection(rawJet, jetCorrLabel, *evt, *es);
	//std::cout << "--> jetCorrFactor(case 2) = " << jetCorrFactor << std::endl;
      }
    } else {
      reco::Candidate::PolarLorentzVector modJetPolarP4(rawJetP4);
      modJetPolarP4.SetEta(sign(rawJetP4.eta())*jetCorrEtaMax);
      
      reco::Candidate::LorentzVector modJetP4(modJetPolarP4);
      
      T modJet(rawJet);
      modJet.setP4(modJetP4);
      //std::cout << "modJet: Pt = " << modJet.pt() << ", eta = " << modJet.eta() << ", phi = " << modJet.phi() 
      //          << " (Px = " << modJet.px() << ", Py = " << modJet.py() << ")" << std::endl;
      
      jetCorrFactor = getCorrection(modJet, jetCorrLabel, *evt, *es);
      //std::cout << "--> jetCorrFactor(case 3) = " << jetCorrFactor << std::endl;
    }

    reco::Candidate::LorentzVector corrJetP4 = rawJetP4;
    corrJetP4 *= jetCorrFactor;

    return corrJetP4;
  }
};

#endif
