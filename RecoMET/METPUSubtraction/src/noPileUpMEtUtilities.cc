#include "RecoMET/METPUSubtraction/interface/noPileUpMEtUtilities.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>
#include <math.h>


noPileUpMEtUtilities::noPileUpMEtUtilities() {
  _minPtDef=-1;
  _maxPtDef=1000000;
}

noPileUpMEtUtilities:: ~noPileUpMEtUtilities() {
}

// namespace noPileUpMEtUtilities
// {
//-------------------------------------------------------------------------------
// general auxiliary functions
void 
noPileUpMEtUtilities::finalizeMEtData(CommonMETData& metData) {
  metData.met = sqrt(metData.mex*metData.mex + metData.mey*metData.mey);
  metData.mez = 0.;
  metData.phi = atan2(metData.mey, metData.mex);
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions for jets
reco::PUSubMETCandInfoCollection 
noPileUpMEtUtilities::cleanJets(const reco::PUSubMETCandInfoCollection& jets,
				const std::vector<reco::Candidate::LorentzVector>& leptons,
				double dRoverlap, bool invert) {
  reco::PUSubMETCandInfoCollection retVal;
  for ( reco::PUSubMETCandInfoCollection::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    bool isOverlap = false;
    for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR2(jet->p4_, *lepton) < dRoverlap*dRoverlap ) {
	isOverlap = true;
	break;
      }
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*jet);
  }
  return retVal;
}


CommonMETData 
noPileUpMEtUtilities::computeCandidateSum(const reco::PUSubMETCandInfoCollection& cands,
					  bool neutralFracOnly, double& sumAbsPx, double& sumAbsPy) {
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  double retVal_sumAbsPx = 0.;
  double retVal_sumAbsPy = 0.;
  double pFrac=1;
  for ( reco::PUSubMETCandInfoCollection::const_iterator cand = cands.begin();
	cand != cands.end(); ++cand ) {
 
    pFrac=1;
    if(neutralFracOnly) pFrac = (1-cand->chargedEnFrac_);
      
    retVal.mex   += cand->p4_.px()*pFrac;
    retVal.mey   += cand->p4_.py()*pFrac;
    retVal.sumet += cand->p4_.pt()*pFrac;
    retVal_sumAbsPx += std::abs(cand->p4_.px());
    retVal_sumAbsPy += std::abs(cand->p4_.py());
  }
  finalizeMEtData(retVal);
  sumAbsPx = retVal_sumAbsPx;
  sumAbsPy = retVal_sumAbsPy;
  return retVal;
}
  

// CommonMETData computeJetSum(const reco::PUSubMETCandInfoCollection& jets, double* sumAbsPx, double* sumAbsPy)
// {
//   CommonMETData retVal;
//   retVal.mex   = 0.;
//   retVal.mey   = 0.;
//   retVal.sumet = 0.;
//   double retVal_sumAbsPx = 0.;
//   double retVal_sumAbsPy = 0.;
//   for ( reco::PUSubMETCandInfoCollection::const_iterator jet = jets.begin();
// 	  jet != jets.end(); ++jet ) {
//     retVal.mex   += jet->p4_.px();
//     retVal.mey   += jet->p4_.py();
//     retVal.sumet += jet->p4_.pt();
//     retVal_sumAbsPx += std::abs(jet->p4_.px());
//     retVal_sumAbsPy += std::abs(jet->p4_.py());
//   }
//   finalizeMEtData(retVal);
//   if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
//   if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
//   return retVal;
// }

// CommonMETData computeJetSum_neutral(const reco::PUSubMETCandInfoCollection& jets, double* sumAbsPx, double* sumAbsPy)
// {
//   CommonMETData retVal;
//   retVal.mex   = 0.;
//   retVal.mey   = 0.;
//   retVal.sumet = 0.;
//   double retVal_sumAbsPx = 0.;
//   double retVal_sumAbsPy = 0.;
//   for ( reco::PUSubMETCandInfoCollection::const_iterator jet = jets.begin();
// 	  jet != jets.end(); ++jet ) {
//     retVal.mex   += (jet->p4_.px()*jet->neutralEnFrac_);
//     retVal.mey   += (jet->p4_.py()*jet->neutralEnFrac_);
//     retVal.sumet += (jet->p4_.pt()*jet->neutralEnFrac_);
//     retVal_sumAbsPx += std::abs(jet->p4_.px());
//     retVal_sumAbsPy += std::abs(jet->p4_.py());
//   }
//   finalizeMEtData(retVal);
//   if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
//   if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
//   return retVal;
// }

reco::PUSubMETCandInfo
noPileUpMEtUtilities::jet(const reco::PUSubMETCandInfoCollection& jets, unsigned idx) {
  reco::PUSubMETCandInfo retVal;
  if ( idx < jets.size() ) {
    reco::PUSubMETCandInfoCollection jets_sorted = jets;
    std::sort(jets_sorted.begin(), jets_sorted.end());
    retVal = jets_sorted[idx];
  }
  return retVal;
}

reco::PUSubMETCandInfo
noPileUpMEtUtilities::leadJet(const reco::PUSubMETCandInfoCollection& jets) {
  return jet(jets, 0);
}

reco::PUSubMETCandInfo
noPileUpMEtUtilities::subleadJet(const reco::PUSubMETCandInfoCollection& jets) {
  return jet(jets, 1);
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions for PFCandidates
reco::PUSubMETCandInfoCollection 
noPileUpMEtUtilities::cleanPFCandidates(const reco::PUSubMETCandInfoCollection& pfCandidates,
					const std::vector<reco::Candidate::LorentzVector>& leptons,
					double dRoverlap, bool invert) {
// invert: false = PFCandidates are required not to overlap with leptons
//         true  = PFCandidates are required to overlap with leptons
  
  reco::PUSubMETCandInfoCollection retVal;
  for ( reco::PUSubMETCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    bool isOverlap = false;
    for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR2(pfCandidate->p4_, *lepton) < dRoverlap*dRoverlap ) {
	isOverlap = true;
	break;
      }
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*pfCandidate);
  }
  return retVal;
}


reco::PUSubMETCandInfoCollection 
noPileUpMEtUtilities::selectCandidates(const reco::PUSubMETCandInfoCollection& cands, 
				       double minPt, double maxPt, int type, 
				       bool isCharged, int isWithinJet) {
  reco::PUSubMETCandInfoCollection retVal;
  for ( reco::PUSubMETCandInfoCollection::const_iterator cand = cands.begin();
	cand != cands.end(); ++cand ) {
      
    if( isCharged && cand->charge_==0) continue;
    double jetPt = cand->p4_.pt();      
    if(  jetPt < minPt || jetPt > maxPt ) continue;
    if(type != reco::PUSubMETCandInfo::kUndefined && cand->type_ != type) continue; 
      
    //for pf candidates
    if( isWithinJet!=noPileUpMEtUtilities::kAll && ( cand->isWithinJet_!=isWithinJet ) ) continue;
      
    retVal.push_back(*cand);
  }
  return retVal;
}



// reco::PUSubMETCandInfoCollection selectJets(const reco::PUSubMETCandInfoCollection& jets,
// 					   double minJetPt, double maxJetPt, int type)
// {
//   reco::PUSubMETCandInfoCollection retVal;
//   for ( reco::PUSubMETCandInfoCollection::const_iterator jet = jets.begin();
// 	  jet != jets.end(); ++jet ) {
//     double jetPt = jet->p4_.pt();      
//     if (  jetPt > minJetPt && 
// 	    jetPt < maxJetPt && 
// 	   (type == reco::PUSubMETCandInfo::kUndefined || jet->type_ == type) ) retVal.push_back(*jet);
//   }
//   return retVal;
// }

// reco::PUSubMETCandInfoCollection selectPFCandidates(const reco::PUSubMETCandInfoCollection& pfCandidates,
// 						      double minCharge, double maxCharge, int type, int isWithinJet)
// // isWithinJet: -1 = no selection applied
// //               0 = PFCandidates are required not to be within jets 
// //              +1 = PFCandidates are required to be within jets 
// {
//   reco::PUSubMETCandInfoCollection retVal;
//   for ( reco::PUSubMETCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
// 	  pfCandidate != pfCandidates.end(); ++pfCandidate ) {
//     int charge = abs(pfCandidate->charge_);
//     if (  charge > minCharge &&
// 	    charge < maxCharge &&
// 	   (type == reco::PUSubMETCandInfo::kUndefined || pfCandidate->type_ == type) &&
// 	   (isWithinJet == -1 || (isWithinJet == 1 && pfCandidate->isWithinJet_) || (isWithinJet == 0 && !pfCandidate->isWithinJet_)) )
// 	retVal.push_back(*pfCandidate);
//   }
//   return retVal;
// }

// CommonMETData computePFCandidateSum(const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
// {
//   CommonMETData retVal;
//   retVal.mex   = 0.;
//   retVal.mey   = 0.;
//   retVal.sumet = 0.;
//   double retVal_sumAbsPx = 0.;
//   double retVal_sumAbsPy = 0.;
//   for ( reco::PUSubMETCandInfoCollection::const_iterator pfCandidate = pfCandidates.begin();
// 	  pfCandidate != pfCandidates.end(); ++pfCandidate ) {
//     retVal.mex   += pfCandidate->p4_.px();
//     retVal.mey   += pfCandidate->p4_.py();
//     retVal.sumet += pfCandidate->p4_.pt();
//     retVal_sumAbsPx += std::abs(pfCandidate->p4_.px());
//     retVal_sumAbsPy += std::abs(pfCandidate->p4_.py());
//   }
//   finalizeMEtData(retVal);
//   if ( sumAbsPx ) (*sumAbsPx) = retVal_sumAbsPx;
//   if ( sumAbsPy ) (*sumAbsPy) = retVal_sumAbsPy;
//   return retVal;
// }
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions to compute different types of MEt/hadronic recoils
//
// NOTE: all pfCandidates and jets passed as function arguments
//       need to be cleaned wrt. leptons
//

void 
noPileUpMEtUtilities::computeAllSums( const reco::PUSubMETCandInfoCollection& jets,
				      const reco::PUSubMETCandInfoCollection& pfCandidates) {
    
  reco::PUSubMETCandInfoCollection pfcsCh = selectCandidates( pfCandidates, _minPtDef, _maxPtDef,
							      reco::PUSubMETCandInfo::kUndefined, false,
							      noPileUpMEtUtilities::kAll);

  reco::PUSubMETCandInfoCollection pfcsChHS = selectCandidates( pfCandidates, _minPtDef, _maxPtDef, 
								reco::PUSubMETCandInfo::kChHS, true,
								noPileUpMEtUtilities::kAll);

  reco::PUSubMETCandInfoCollection pfcsChPU = selectCandidates( pfCandidates, _minPtDef, _maxPtDef,
								reco::PUSubMETCandInfo::kChPU, true, 
								noPileUpMEtUtilities::kAll);
  
  reco::PUSubMETCandInfoCollection pfcsNUnclustered = selectCandidates( pfCandidates, _minPtDef, _maxPtDef, 
									reco::PUSubMETCandInfo::kNeutral, false,
									noPileUpMEtUtilities::kOutsideJet);
  
  reco::PUSubMETCandInfoCollection jetsHS = selectCandidates(jets, 10.0, _maxPtDef, reco::PUSubMETCandInfo::kHS, false,
							     noPileUpMEtUtilities::kAll);
  reco::PUSubMETCandInfoCollection jetsPU = selectCandidates(jets, 10.0, _maxPtDef, reco::PUSubMETCandInfo::kPU, false,
							     noPileUpMEtUtilities::kAll);

  //not used so far
  //_chPfcSum = computeCandidateSum(pfcsCh, false, &_chPfcSumAbsPx, &_chPfcSumAbsPy);
  _chHSPfcSum = computeCandidateSum(pfcsChHS, false, _chHSPfcSumAbsPx, _chHSPfcSumAbsPy);
  _chPUPfcSum = computeCandidateSum(pfcsChPU, false, _chPUPfcSumAbsPx, _chPUPfcSumAbsPy);
  _nUncPfcSum = computeCandidateSum(pfcsNUnclustered, false, _nUncPfcSumAbsPx, _nUncPfcSumAbsPy);

  _nHSJetSum = computeCandidateSum(jetsHS, true, _nHSJetSumAbsPx, _nHSJetSumAbsPy);
  _nPUJetSum = computeCandidateSum(jetsPU, true, _nPUJetSumAbsPx, _nPUJetSumAbsPy);
  
}
  
CommonMETData 
noPileUpMEtUtilities::computeRecoil(int metType, double& sumAbsPx, double& sumAbsPy) {
  CommonMETData retVal;
  double retSumAbsPx = 0.;
  double retSumAbsPy = 0.;

  //never used....
  // if(metType==noPileUpMEtUtilities::kChMET) {
  //   CommonMETData chPfcSum = computeCandidateSum(_pfcsCh, false, &trackSumAbsPx, &trackSumAbsPy);
  //   retVal.mex   = -chPfcSum.mex;
  //   retVal.mey   = -chPfcSum.mey;
  //   retVal.sumet = chPfcSum.sumet;
  //   retSumAbsPx = ; 
  //   retSumAbsPy = ; 
  // }
  if(metType==noPileUpMEtUtilities::kChHSMET) {
    retVal.mex   = _chHSPfcSum.mex;
    retVal.mey   = _chHSPfcSum.mey;
    retVal.sumet = _chHSPfcSum.sumet;
    retSumAbsPx = _chHSPfcSumAbsPx; 
    retSumAbsPy = _chHSPfcSumAbsPy; 
  }
  if(metType==noPileUpMEtUtilities::kChPUMET) {
    retVal.mex   = _chPUPfcSum.mex;
    retVal.mey   = _chPUPfcSum.mey;
    retVal.sumet = _chPUPfcSum.sumet;
    retSumAbsPx = _chPUPfcSumAbsPx; 
    retSumAbsPy = _chPUPfcSumAbsPy; 
  }
  if(metType==noPileUpMEtUtilities::kNeutralUncMET) {
    retVal.mex   = _nUncPfcSum.mex;
    retVal.mey   = _nUncPfcSum.mey;
    retVal.sumet = _nUncPfcSum.sumet;
    retSumAbsPx = _nUncPfcSumAbsPx; 
    retSumAbsPy = _nUncPfcSumAbsPy; 
  }
  if(metType==noPileUpMEtUtilities::kHadronicHSMET) {
    retVal.mex   = _chHSPfcSum.mex + _nHSJetSum.mex;
    retVal.mey   = _chHSPfcSum.mey + _nHSJetSum.mey;
    retVal.sumet = _chHSPfcSum.sumet + _nHSJetSum.sumet;
    retSumAbsPx = _chHSPfcSumAbsPx + _nHSJetSumAbsPx; 
    retSumAbsPy = _chHSPfcSumAbsPy + _nHSJetSumAbsPy; 
  }
  if(metType==noPileUpMEtUtilities::kHadronicPUMET) {
    retVal.mex   = _chPUPfcSum.mex + _nHSJetSum.mex;
    retVal.mey   = _chPUPfcSum.mey + _nHSJetSum.mey;
    retVal.sumet = _chPUPfcSum.sumet + _nHSJetSum.sumet;
    retSumAbsPx = _chPUPfcSumAbsPx + _nHSJetSumAbsPx; 
    retSumAbsPy = _chPUPfcSumAbsPy + _nHSJetSumAbsPy; 
  }

 sumAbsPx = retSumAbsPx;
 sumAbsPy = retSumAbsPy;
 
 return retVal;
}


//   CommonMETData computeTrackRecoil(const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     reco::PUSubMETCandInfoCollection chargedPFCandidates = selectPFCandidates(
//       pfCandidates, 0.5, 1.e+3, reco::PUSubMETCandInfo::kUndefined, -1);
//     double trackSumAbsPx = 0.;
//     double trackSumAbsPy = 0.;
//     CommonMETData trackSum = computePFCandidateSum(chargedPFCandidates, &trackSumAbsPx, &trackSumAbsPy);
//     CommonMETData retVal;
//     retVal.mex   = -trackSum.mex;
//     retVal.mey   = -trackSum.mey;
//     retVal.sumet =  trackSum.sumet;
//     finalizeMEtData(retVal);
//     if ( sumAbsPx ) (*sumAbsPx) = trackSumAbsPx;
//     if ( sumAbsPy ) (*sumAbsPy) = trackSumAbsPy;
//     return retVal;
//   }

//   CommonMETData computeTrackRecoilNoPU(const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     reco::PUSubMETCandInfoCollection chargedPFCandidatesNoPU = selectPFCandidates(
//       pfCandidates, 0.5, 1.e+3, reco::PUSubMETCandInfo::kNoPileUpCharged, -1);
//     return computePFCandidateSum(chargedPFCandidatesNoPU, sumAbsPx, sumAbsPy);
//   }

//   CommonMETData computeTrackRecoilPU(const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     reco::PUSubMETCandInfoCollection chargedPFCandidatesPU = selectPFCandidates(
//       pfCandidates, 0.5, 1.e+3, reco::PUSubMETCandInfo::kPileUpCharged, -1);
//     return computePFCandidateSum(chargedPFCandidatesPU, sumAbsPx, sumAbsPy);
//   }

//   CommonMETData computeNeutralRecoil_unclustered(const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     reco::PUSubMETCandInfoCollection neutralPFCandidates_unclustered = selectPFCandidates(
//       pfCandidates, -0.5, +0.5, reco::PUSubMETCandInfo::kNeutral, 0);
//     return computePFCandidateSum(neutralPFCandidates_unclustered, sumAbsPx, sumAbsPy);
//   }

//   CommonMETData computeHadRecoilNoPU(const reco::PUSubMETCandInfoCollection& jets,
// 				     const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     double trackSumAbsPx = 0.;
//     double trackSumAbsPy = 0.;
//     CommonMETData trackSumNoPU = computeTrackRecoilNoPU(pfCandidates, &trackSumAbsPx, &trackSumAbsPy);
//     reco::PUSubMETCandInfoCollection jetsNoPU = selectJets(jets, 10.0, 1.e+6, reco::PUSubMETCandInfo::kNoPileUp);
//     double jetSumNoPUabsPx_neutral = 0.;
//     double jetSumNoPUabsPy_neutral = 0.;
//     CommonMETData jetSumNoPU_neutral = computeJetSum_neutral(jetsNoPU, &jetSumNoPUabsPx_neutral, &jetSumNoPUabsPy_neutral);
//     CommonMETData retVal;
//     retVal.mex   = trackSumNoPU.mex   + jetSumNoPU_neutral.mex;
//     retVal.mey   = trackSumNoPU.mey   + jetSumNoPU_neutral.mey;
//     retVal.sumet = trackSumNoPU.sumet + jetSumNoPU_neutral.sumet;
//     finalizeMEtData(retVal);
//     if ( sumAbsPx ) (*sumAbsPx) = trackSumAbsPx + jetSumNoPUabsPx_neutral;
//     if ( sumAbsPy ) (*sumAbsPy) = trackSumAbsPy + jetSumNoPUabsPy_neutral;
//     return retVal;
//   }

//   CommonMETData computeHadRecoilPU(const reco::PUSubMETCandInfoCollection& jets,
// 				   const reco::PUSubMETCandInfoCollection& pfCandidates, double* sumAbsPx, double* sumAbsPy)
//   {
//     double trackSumPUabsPx = 0.;
//     double trackSumPUabsPy = 0.;
//     CommonMETData trackSumPU = computeTrackRecoilPU(pfCandidates, &trackSumPUabsPx, &trackSumPUabsPy);
//     reco::PUSubMETCandInfoCollection jetsPU = selectJets(jets, 10.0, 1.e+6, reco::PUSubMETCandInfo::kPileUp);
//     double jetSumPUabsPx_neutral = 0.;
//     double jetSumPUabsPy_neutral = 0.;
//     CommonMETData jetSumPU_neutral = computeJetSum_neutral(jetsPU, &jetSumPUabsPx_neutral, &jetSumPUabsPy_neutral);
//     CommonMETData retVal;
//     retVal.mex   = trackSumPU.mex   + jetSumPU_neutral.mex;
//     retVal.mey   = trackSumPU.mey   + jetSumPU_neutral.mey;
//     retVal.sumet = trackSumPU.sumet + jetSumPU_neutral.sumet;
//     finalizeMEtData(retVal);
//     if ( sumAbsPx ) (*sumAbsPx) = trackSumPUabsPx + jetSumPUabsPx_neutral;
//     if ( sumAbsPy ) (*sumAbsPy) = trackSumPUabsPy + jetSumPUabsPy_neutral;
//     return retVal;
//   }
//   //-------------------------------------------------------------------------------
// }

