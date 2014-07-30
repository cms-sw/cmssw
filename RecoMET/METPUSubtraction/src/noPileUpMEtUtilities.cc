#include "RecoMET/METPUSubtraction/interface/noPileUpMEtUtilities.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>
#include <math.h>

namespace noPileUpMEtUtilities
{
  //-------------------------------------------------------------------------------
  // general auxiliary functions
  void finalizeMEtData(CommonMETData& metData)
  {
    metData.met = sqrt(metData.mex*metData.mex + metData.mey*metData.mey);
    metData.mez = 0.;
    metData.phi = atan2(metData.mey, metData.mex);
  }
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions for jets
  reco::MVAMEtJetInfoCollection cleanJets(const reco::MVAMEtJetInfoCollection& jets,
					  const std::vector<reco::Candidate::LorentzVector>& leptons,
					  double dRoverlap, bool invert)
  {
    reco::MVAMEtJetInfoCollection retVal;
    for ( reco::MVAMEtJetInfoCollection::const_iterator jet = jets.begin();
	  jet != jets.end(); ++jet ) {
      bool isOverlap = false;
      for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	    lepton != leptons.end(); ++lepton ) {
	if ( deltaR(jet->p4_, *lepton) < dRoverlap ) {
	  isOverlap = true;
	  break;
	}
      }
      if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*jet);
    }
    return retVal;
  }

  reco::MVAMEtJetInfoCollection selectJets(const reco::MVAMEtJetInfoCollection& jets,
					   double minJetPt, double maxJetPt, int type)
  {
    reco::MVAMEtJetInfoCollection retVal;
    for ( reco::MVAMEtJetInfoCollection::const_iterator jet = jets.begin();
	  jet != jets.end(); ++jet ) {
      double jetPt = jet->p4_.pt();      
      if (  jetPt > minJetPt && 
	    jetPt < maxJetPt && 
	   (type == reco::MVAMEtJetInfo::kUndefined || jet->type_ == type) ) retVal.push_back(*jet);
    }
    return retVal;
  }

  CommonMETData computeJetSum(const reco::MVAMEtJetInfoCollection& jets, double* sumAbsPx, double* sumAbsPy)
  {
    CommonMETData retVal;
    retVal.mex   = 0.;
    retVal.mey   = 0.;
    retVal.sumet = 0.;
    double retVal_sumAbsPx = 0.;
    double retVal_sumAbsPy = 0.;
    for ( reco::MVAMEtJetInfoCollection::const_iterator jet = jets.begin();
	  jet != jets.end(); ++jet ) {
      retVal.mex   += jet->p4_.px();
      retVal.mey   += jet->p4_.py();
      retVal.sumet += jet->p4_.pt();
      retVal_sumAbsPx += fabs(jet->p4_.px());
      retVal_sumAbsPy += fabs(jet->p4_.py());
    }
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
    if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
    return retVal;
  }

  CommonMETData computeJetSum_neutral(const reco::MVAMEtJetInfoCollection& jets, double* sumAbsPx, double* sumAbsPy)
  {
    CommonMETData retVal;
    retVal.mex   = 0.;
    retVal.mey   = 0.;
    retVal.sumet = 0.;
    double retVal_sumAbsPx = 0.;
    double retVal_sumAbsPy = 0.;
    for ( reco::MVAMEtJetInfoCollection::const_iterator jet = jets.begin();
	  jet != jets.end(); ++jet ) {
      retVal.mex   += (jet->p4_.px()*jet->neutralEnFrac_);
      retVal.mey   += (jet->p4_.py()*jet->neutralEnFrac_);
      retVal.sumet += (jet->p4_.pt()*jet->neutralEnFrac_);
      retVal_sumAbsPx += fabs(jet->p4_.px());
      retVal_sumAbsPy += fabs(jet->p4_.py());
    }
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
    if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
    return retVal;
  }

  reco::MVAMEtJetInfo jet(const reco::MVAMEtJetInfoCollection& jets, unsigned idx)
  {
    reco::MVAMEtJetInfo retVal;
    if ( idx < jets.size() ) {
      reco::MVAMEtJetInfoCollection jets_sorted = jets;
      std::sort(jets_sorted.begin(), jets_sorted.end());
      retVal = jets_sorted[idx];
    }
    return retVal;
  }

  reco::MVAMEtJetInfo leadJet(const reco::MVAMEtJetInfoCollection& jets)
  {
    return jet(jets, 0);
  }

  reco::MVAMEtJetInfo subleadJet(const reco::MVAMEtJetInfoCollection& jets)
  {
    return jet(jets, 1);
  }
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions for PFCandidates
  reco::MVAMEtPFCandInfoCollection cleanPFCandidates(const reco::MVAMEtPFCandInfoCollection& pfCandidates,
						     const std::vector<reco::Candidate::LorentzVector>& leptons,
						     double dRoverlap, bool invert)
  // invert: false = PFCandidates are required not to overlap with leptons
  //         true  = PFCandidates are required to overlap with leptons
  {
    reco::MVAMEtPFCandInfoCollection retVal;
    for ( reco::MVAMEtPFCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
	  pfCandidate != pfCandidates.end(); ++pfCandidate ) {
      bool isOverlap = false;
      for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	    lepton != leptons.end(); ++lepton ) {
	if ( deltaR(pfCandidate->p4_, *lepton) < dRoverlap ) {
	  isOverlap = true;
	  break;
	}
      }
      if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*pfCandidate);
    }
    return retVal;
  }

  reco::MVAMEtPFCandInfoCollection selectPFCandidates(const reco::MVAMEtPFCandInfoCollection& pfCandidates,
						      double minCharge, double maxCharge, int type, int isWithinJet)
  // isWithinJet: -1 = no selection applied
  //               0 = PFCandidates are required not to be within jets 
  //              +1 = PFCandidates are required to be within jets 
  {
    reco::MVAMEtPFCandInfoCollection retVal;
    for ( reco::MVAMEtPFCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
	  pfCandidate != pfCandidates.end(); ++pfCandidate ) {
      int charge = abs(pfCandidate->charge_);
      if (  charge > minCharge &&
	    charge < maxCharge &&
	   (type == reco::MVAMEtJetInfo::kUndefined || pfCandidate->type_ == type) &&
	   (isWithinJet == -1 || (isWithinJet == 1 && pfCandidate->isWithinJet_) || (isWithinJet == 0 && !pfCandidate->isWithinJet_)) )
	retVal.push_back(*pfCandidate);
    }
    return retVal;
  }

  CommonMETData computePFCandidateSum(const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    CommonMETData retVal;
    retVal.mex   = 0.;
    retVal.mey   = 0.;
    retVal.sumet = 0.;
    double retVal_sumAbsPx = 0.;
    double retVal_sumAbsPy = 0.;
    for ( reco::MVAMEtPFCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
	  pfCandidate != pfCandidates.end(); ++pfCandidate ) {
      retVal.mex   += pfCandidate->p4_.px();
      retVal.mey   += pfCandidate->p4_.py();
      retVal.sumet += pfCandidate->p4_.pt();
      retVal_sumAbsPx += fabs(pfCandidate->p4_.px());
      retVal_sumAbsPy += fabs(pfCandidate->p4_.py());
    }
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
    if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
    return retVal;
  }
  //-------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------
  // auxiliary functions to compute different types of MEt/hadronic recoils
  //
  // NOTE: all pfCandidates and jets passed as function arguments
  //       need to be cleaned wrt. leptons
  //
  CommonMETData computeTrackRecoil(const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    reco::MVAMEtPFCandInfoCollection chargedPFCandidates = selectPFCandidates(
      pfCandidates, 0.5, 1.e+3, reco::MVAMEtPFCandInfo::kUndefined, -1);
    double trackSumAbsPx = 0.;
    double trackSumAbsPy = 0.;
    CommonMETData trackSum = computePFCandidateSum(chargedPFCandidates, &trackSumAbsPx, &trackSumAbsPy);
    CommonMETData retVal;
    retVal.mex   = -trackSum.mex;
    retVal.mey   = -trackSum.mey;
    retVal.sumet =  trackSum.sumet;
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = trackSumAbsPx;
    if ( sumAbsPy ) (*sumAbsPy) = trackSumAbsPy;
    return retVal;
  }

  CommonMETData computeTrackRecoilNoPU(const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    reco::MVAMEtPFCandInfoCollection chargedPFCandidatesNoPU = selectPFCandidates(
      pfCandidates, 0.5, 1.e+3, reco::MVAMEtPFCandInfo::kNoPileUpCharged, -1);
    return computePFCandidateSum(chargedPFCandidatesNoPU, sumAbsPx, sumAbsPy);
  }

  CommonMETData computeTrackRecoilPU(const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    reco::MVAMEtPFCandInfoCollection chargedPFCandidatesPU = selectPFCandidates(
      pfCandidates, 0.5, 1.e+3, reco::MVAMEtPFCandInfo::kPileUpCharged, -1);
    return computePFCandidateSum(chargedPFCandidatesPU, sumAbsPx, sumAbsPy);
  }

  CommonMETData computeNeutralRecoil_unclustered(const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    reco::MVAMEtPFCandInfoCollection neutralPFCandidates_unclustered = selectPFCandidates(
      pfCandidates, -0.5, +0.5, reco::MVAMEtPFCandInfo::kNeutral, 0);
    return computePFCandidateSum(neutralPFCandidates_unclustered, sumAbsPx, sumAbsPy);
  }

  CommonMETData computeHadRecoilNoPU(const reco::MVAMEtJetInfoCollection& jets,
				     const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    double trackSumAbsPx = 0.;
    double trackSumAbsPy = 0.;
    CommonMETData trackSumNoPU = computeTrackRecoilNoPU(pfCandidates, &trackSumAbsPx, &trackSumAbsPy);
    reco::MVAMEtJetInfoCollection jetsNoPU = selectJets(jets, 10.0, 1.e+6, reco::MVAMEtJetInfo::kNoPileUp);
    double jetSumNoPUabsPx_neutral = 0.;
    double jetSumNoPUabsPy_neutral = 0.;
    CommonMETData jetSumNoPU_neutral = computeJetSum_neutral(jetsNoPU, &jetSumNoPUabsPx_neutral, &jetSumNoPUabsPy_neutral);
    CommonMETData retVal;
    retVal.mex   = trackSumNoPU.mex   + jetSumNoPU_neutral.mex;
    retVal.mey   = trackSumNoPU.mey   + jetSumNoPU_neutral.mey;
    retVal.sumet = trackSumNoPU.sumet + jetSumNoPU_neutral.sumet;
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = trackSumAbsPx + jetSumNoPUabsPx_neutral;
    if ( sumAbsPy ) (*sumAbsPy) = trackSumAbsPy + jetSumNoPUabsPy_neutral;
    return retVal;
  }

  CommonMETData computeHadRecoilPU(const reco::MVAMEtJetInfoCollection& jets,
				   const reco::MVAMEtPFCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
  {
    double trackSumPUabsPx = 0.;
    double trackSumPUabsPy = 0.;
    CommonMETData trackSumPU = computeTrackRecoilPU(pfCandidates, &trackSumPUabsPx, &trackSumPUabsPy);
    reco::MVAMEtJetInfoCollection jetsPU = selectJets(jets, 10.0, 1.e+6, reco::MVAMEtJetInfo::kPileUp);
    double jetSumPUabsPx_neutral = 0.;
    double jetSumPUabsPy_neutral = 0.;
    CommonMETData jetSumPU_neutral = computeJetSum_neutral(jetsPU, &jetSumPUabsPx_neutral, &jetSumPUabsPy_neutral);
    CommonMETData retVal;
    retVal.mex   = trackSumPU.mex   + jetSumPU_neutral.mex;
    retVal.mey   = trackSumPU.mey   + jetSumPU_neutral.mey;
    retVal.sumet = trackSumPU.sumet + jetSumPU_neutral.sumet;
    finalizeMEtData(retVal);
    if ( sumAbsPx ) (*sumAbsPx) = trackSumPUabsPx + jetSumPUabsPx_neutral;
    if ( sumAbsPy ) (*sumAbsPy) = trackSumPUabsPy + jetSumPUabsPy_neutral;
    return retVal;
  }
  //-------------------------------------------------------------------------------
}

