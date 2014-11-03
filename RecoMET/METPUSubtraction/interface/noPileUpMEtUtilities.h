#ifndef RecoMET_METPUSubtraction_noPileUpMEtUtilities_h
#define RecoMET_METPUSubtraction_noPileUpMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/MVAMEtData.h"
#include "DataFormats/METReco/interface/MVAMEtDataFwd.h"

#include <vector>

namespace noPileUpMEtUtilities
{
  //-------------------------------------------------------------------------------
  // general auxiliary functions
  void finalizeMEtData(CommonMETData&);
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions for jets
  reco::MVAMEtJetInfoCollection cleanJets(const reco::MVAMEtJetInfoCollection&, 
					  const std::vector<reco::Candidate::LorentzVector>&, 
					  double, bool);
  reco::MVAMEtJetInfoCollection selectJets(const reco::MVAMEtJetInfoCollection&,
					   double, double, int);
  CommonMETData computeJetSum(const reco::MVAMEtJetInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeJetSum_neutral(const reco::MVAMEtJetInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  reco::MVAMEtJetInfo jet(const reco::MVAMEtJetInfoCollection&, unsigned);
  reco::MVAMEtJetInfo leadJet(const reco::MVAMEtJetInfoCollection&);
  reco::MVAMEtJetInfo subleadJet(const reco::MVAMEtJetInfoCollection&);
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions for PFCandidates
  reco::MVAMEtPFCandInfoCollection cleanPFCandidates(const reco::MVAMEtPFCandInfoCollection&,
						     const std::vector<reco::Candidate::LorentzVector>&,
						     double, bool);
  reco::MVAMEtPFCandInfoCollection selectPFCandidates(const reco::MVAMEtPFCandInfoCollection&,
						      double, double, int, int);
  CommonMETData computePFCandidateSum(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions to compute different types of MEt/hadronic recoils
  //
  // NOTE: all pfCandidates and jets passed as function arguments
  //       need to be cleaned wrt. leptons
  //
  CommonMETData computeTrackRecoil(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeTrackRecoilNoPU(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeTrackRecoilPU(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeNeutralRecoil_unclustered(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeHadRecoilNoPU(const reco::MVAMEtJetInfoCollection&,
				     const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  CommonMETData computeHadRecoilPU(const reco::MVAMEtJetInfoCollection&,
				   const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0);
  //-------------------------------------------------------------------------------
}

#endif
