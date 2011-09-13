#ifndef JetMETCorrections_Type1MET_metCorrAuxFunctions_h
#define JetMETCorrections_Type1MET_metCorrAuxFunctions_h

/** \file metCorrAuxFunctions
 *
 * Auxiliary functions for computing Type 1 + 2 MET corrections 
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.00 $
 *
 * $Id: metCorrAuxFunctions.h,v 1.18 2011/05/30 15:19:41 veelken Exp $
 *
 */

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

namespace metCorr_namespace
{
  // restrict computation of JEC factors to region |eta| < 4.7,
  // to work around problem with CMSSW_4_2_x JEC factors at high eta,
  // reported in
  //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
  //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
  template <typename T>
  double getJetCorrFactor(const JetCorrector* corrector, const T& jet, const edm::RefToBase<reco::Jet>& jetRef, 
			  const reco::Candidate::LorentzVector& jetP4, edm::Event& evt, const edm::EventSetup& es, double jetCorrEtaMax)
  {
    if ( fabs(jetP4.eta()) < jetCorrEtaMax ) {
      return corrector->correction(jet, jetRef, evt, es);
    } else {
      reco::Candidate::PolarLorentzVector modJetPolarP4(jetP4);
      modJetPolarP4.SetEta(jetCorrEtaMax);

      reco::Candidate::LorentzVector modJetP4(modJetPolarP4);

      T modJet(jet);
      modJet.setP4(modJetP4);

      return corrector->correction(modJet, jetRef, evt, es);
    }
  }
}

#endif
