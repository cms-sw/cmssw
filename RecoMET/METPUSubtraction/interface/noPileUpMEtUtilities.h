#ifndef RecoMET_METPUSubtraction_noPileUpMEtUtilities_h
#define RecoMET_METPUSubtraction_noPileUpMEtUtilities_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/PUSubMETData.h"
#include "DataFormats/METReco/interface/PUSubMETDataFwd.h"

#include <vector>

class noPileUpMEtUtilities
{

 public:

  enum {kOutsideJet=0,kWithin, kAll};
  enum {kChHSMET=0, kChPUMET, kNeutralUncMET, kHadronicHSMET, kHadronicPUMET};

  noPileUpMEtUtilities();
  ~noPileUpMEtUtilities();

 
  // general auxiliary functions
  void finalizeMEtData(CommonMETData&);
  
  //-------------------------------------------------------------------------------
  // auxiliary functions for jets
  reco::PUSubMETCandInfoCollection cleanJets(const reco::PUSubMETCandInfoCollection&, 
					     const std::vector<reco::Candidate::LorentzVector>&, 
					     double, bool);
  /* reco::MVAMEtJetInfoCollection selectJets(const reco::MVAMEtJetInfoCollection&, */
  /* 					   double, double, int); */

  reco::PUSubMETCandInfo jet(const reco::PUSubMETCandInfoCollection&, unsigned);
  reco::PUSubMETCandInfo leadJet(const reco::PUSubMETCandInfoCollection&);
  reco::PUSubMETCandInfo subleadJet(const reco::PUSubMETCandInfoCollection&);

  // auxiliary functions for PFCandidates
  reco::PUSubMETCandInfoCollection cleanPFCandidates(const reco::PUSubMETCandInfoCollection&,
						     const std::vector<reco::Candidate::LorentzVector>&,
						     double, bool);

 
  // common internal functions for jets and pfCandidates
  void computeAllSums( const reco::PUSubMETCandInfoCollection& jets,
		       const reco::PUSubMETCandInfoCollection& pfCandidates);

  CommonMETData computeRecoil(int metType, double& sumAbsPx, double& sumAbsPy);

  /* CommonMETData computeJetSum(const reco::MVAMEtJetInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeJetSum_neutral(const reco::MVAMEtJetInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */

  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
 
  /* reco::MVAMEtPFCandInfoCollection selectPFCandidates(const reco::MVAMEtPFCandInfoCollection&, */
  /* 						      double, double, int, int); */
  /* CommonMETData computePFCandidateSum(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions to compute different types of MEt/hadronic recoils
  //
  // NOTE: all pfCandidates and jets passed as function arguments
  //       need to be cleaned wrt. leptons
  //
  /* CommonMETData computeTrackRecoil(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeTrackRecoilNoPU(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeTrackRecoilPU(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeNeutralRecoil_unclustered(const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeHadRecoilNoPU(const reco::MVAMEtJetInfoCollection&, */
  /* 				     const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  /* CommonMETData computeHadRecoilPU(const reco::MVAMEtJetInfoCollection&, */
  /* 				   const reco::MVAMEtPFCandInfoCollection&, double* sumAbsPx = 0, double* sumAbsPy = 0); */
  //-------------------------------------------------------------------------------

 private:
 
  // common internal functions for jets and pfCandidates
  CommonMETData computeCandidateSum(const reco::PUSubMETCandInfoCollection& cands,
				    bool neutralFracOnly, double& sumAbsPx, double& sumAbsPy);
  
  reco::PUSubMETCandInfoCollection selectCandidates(const reco::PUSubMETCandInfoCollection& cands, 
						    double minPt, double maxPt, int type, 
						    bool isCharged, int isWithinJet);


 private:


  double _minPtDef;
  double _maxPtDef;

  CommonMETData _chHSPfcSum;
  CommonMETData _chPUPfcSum;
  CommonMETData _nUncPfcSum;
  CommonMETData _nHSJetSum;
  CommonMETData _nPUJetSum;

  double _chHSPfcSumAbsPx;
  double _chPUPfcSumAbsPx;
  double _nUncPfcSumAbsPx;
  double _nHSJetSumAbsPx;
  double _nPUJetSumAbsPx;

  double _chHSPfcSumAbsPy;
  double _chPUPfcSumAbsPy;
  double _nUncPfcSumAbsPy;
  double _nHSJetSumAbsPy;
  double _nPUJetSumAbsPy;

};

#endif
