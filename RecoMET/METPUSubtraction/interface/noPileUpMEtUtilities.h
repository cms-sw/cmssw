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
  //-------------------------------------------------------------------------------

 private:
 
  // common internal functions for jets and pfCandidates, to compute the different object sums
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
