#ifndef PhysicsTools_PatUtils_SmearedPFCandidateProducerForPFNoPUMEt_h
#define PhysicsTools_PatUtils_SmearedPFCandidateProducerForPFNoPUMEt_h

/** \class SmearedPFCandidateProducerForPFNoPUMEt
 *
 * Produce collection of "smeared" PFCandidates for PFCandidates that are within jets of Pt > 10 GeV.
 * The aim of this correction is to account for the difference in jet energy resolution
 * between Monte Carlo simulation and Data.
 * Smearing of PFCandidate constituents of the jets is necessary to improve the data/MC agreement:
 * As the no-PU MET algorithm considers only jets of Pt > 30 GeV as "jets",
 * differences in jet energy resolution for jets between 10 and 30 GeV need to be accounted for by smearing
 * the PFCandidate constituents.
 * The jet energy resolutions have been measured in QCD di-jet and gamma + jets events selected in 2010 data,
 * as documented in the PAS JME-10-014.
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty
 *       on PFMET reconstructed by no-PU MET reconstruction algorithm
 *      (implemented in JetMETCorrections/Type1MET/src/PFNoPUMETProducer.cc)
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SmearedPFCandidateProducerForPFNoPUMEt.h,v 1.1 2012/08/31 10:06:13 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include <string>
#include <vector>

template <typename T, typename Textractor>
  class SmearedPFCandidateProducerForPFNoPUMEtT : public edm::stream::EDProducer<>
{
  typedef std::vector<T> JetCollection;

 public:

  explicit SmearedPFCandidateProducerForPFNoPUMEtT(const edm::ParameterSet&);
  ~SmearedPFCandidateProducerForPFNoPUMEtT();

 private:

  void produce(edm::Event&, const edm::EventSetup&);

  SmearedJetProducer_namespace::GenJetMatcherT<T> genJetMatcher_;

//--- configuration parameters

  // collection of pat::Jets (with L2L3/L2L3Residual corrections applied)
  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;
  edm::EDGetTokenT<JetCollection> srcJets_;

  TFile* inputFile_;
  TH2* lut_;

  SmearedJetProducer_namespace::JetResolutionExtractorT<T> jetResolutionExtractor_;
  TRandom3 rnd_;

  edm::InputTag jetCorrLabel_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrToken_;    // e.g. 'ak5CaloJetL1FastL2L3' (MC) / 'ak5CaloJetL1FastL2L3Residual' (Data)
  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
  Textractor jetCorrExtractor_;

  double sigmaMaxGenJetMatch_; // maximum difference between energy of reconstructed jet and matched generator level jet
                               // (if the difference between reconstructed and generated jet energy exceeds this threshold,
                               //  the jet is considered to have substantial pile-up contributions are is considered to be unmatched)

  double smearBy_; // option to "smear" jet energy by N standard-deviations, useful for template morphing

  double shiftBy_; // option to increase/decrease within uncertainties the jet energy resolution used for smearing

  StringCutObjectSelector<T>* skipJetSelection_; // jets passing this cut are **not** smeared
  double skipRawJetPtThreshold_;  // jets with transverse momenta below this value (either on "raw" or "corrected" level)
  double skipCorrJetPtThreshold_; // are **not** smeared

  int verbosity_; // flag to enabled/disable debug output

  bool skipJetSel_;

};

#endif




