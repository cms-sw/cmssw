#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtUtilities.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>
#include <cmath>


NoPileUpMEtUtilities::NoPileUpMEtUtilities() {
  minPtDef_=-1;
  maxPtDef_=1000000;
}

NoPileUpMEtUtilities:: ~NoPileUpMEtUtilities() {
}

// namespace NoPileUpMEtUtilities
// {
//-------------------------------------------------------------------------------
// general auxiliary functions
void 
NoPileUpMEtUtilities::finalizeMEtData(CommonMETData& metData) {
  metData.met = sqrt(metData.mex*metData.mex + metData.mey*metData.mey);
  metData.mez = 0.;
  metData.phi = atan2(metData.mey, metData.mex);
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions for jets
reco::PUSubMETCandInfoCollection 
NoPileUpMEtUtilities::cleanJets(const reco::PUSubMETCandInfoCollection& jets,
				const std::vector<reco::Candidate::LorentzVector>& leptons,
				double dRoverlap, bool invert) {
  reco::PUSubMETCandInfoCollection retVal;
  for ( reco::PUSubMETCandInfoCollection::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    bool isOverlap = false;
    for ( std::vector<reco::Candidate::LorentzVector>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR2(jet->p4(), *lepton) < dRoverlap*dRoverlap ) {
	isOverlap = true;
	break;
      }
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*jet);
  }
  return retVal;
}


CommonMETData 
NoPileUpMEtUtilities::computeCandidateSum(const reco::PUSubMETCandInfoCollection& cands,
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
    if(neutralFracOnly) pFrac = (1-cand->chargedEnFrac() );
      
    retVal.mex   += cand->p4().px()*pFrac;
    retVal.mey   += cand->p4().py()*pFrac;
    retVal.sumet += cand->p4().pt()*pFrac;
    retVal_sumAbsPx += std::abs(cand->p4().px());
    retVal_sumAbsPy += std::abs(cand->p4().py());
  }
  finalizeMEtData(retVal);
  sumAbsPx = retVal_sumAbsPx;
  sumAbsPy = retVal_sumAbsPy;
  return retVal;
}
  
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions for PFCandidates
reco::PUSubMETCandInfoCollection 
NoPileUpMEtUtilities::cleanPFCandidates(const reco::PUSubMETCandInfoCollection& pfCandidates,
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
      if ( deltaR2(pfCandidate->p4(), *lepton) < dRoverlap*dRoverlap ) {
	isOverlap = true;
	break;
      }
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*pfCandidate);
  }
  return retVal;
}


reco::PUSubMETCandInfoCollection 
NoPileUpMEtUtilities::selectCandidates(const reco::PUSubMETCandInfoCollection& cands, 
				       double minPt, double maxPt, int type, 
				       bool isCharged, int isWithinJet) {
  reco::PUSubMETCandInfoCollection retVal;
  for ( reco::PUSubMETCandInfoCollection::const_iterator cand = cands.begin();
	cand != cands.end(); ++cand ) {
      
    if( isCharged && cand->charge()==0) continue;
    double jetPt = cand->p4().pt();      
    if(  jetPt < minPt || jetPt > maxPt ) continue;
    if(type != reco::PUSubMETCandInfo::kUndefined && cand->type() != type) continue; 
      
    //for pf candidates
    if( isWithinJet!=NoPileUpMEtUtilities::kAll && ( cand->isWithinJet()!=isWithinJet ) ) continue;
      
    retVal.push_back(*cand);
  }
  return retVal;
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// auxiliary functions to compute different types of MEt/hadronic recoils
//
// NOTE: all pfCandidates and jets passed as function arguments
//       need to be cleaned wrt. leptons
//

void 
NoPileUpMEtUtilities::computeAllSums( const reco::PUSubMETCandInfoCollection& jets,
				      const reco::PUSubMETCandInfoCollection& pfCandidates) {
  
  reco::PUSubMETCandInfoCollection pfcsCh = selectCandidates( pfCandidates, minPtDef_, maxPtDef_,
							      reco::PUSubMETCandInfo::kUndefined, false,
							      NoPileUpMEtUtilities::kAll);

  reco::PUSubMETCandInfoCollection pfcsChHS = selectCandidates( pfCandidates, minPtDef_, maxPtDef_, 
								reco::PUSubMETCandInfo::kChHS, true,
								NoPileUpMEtUtilities::kAll);

  reco::PUSubMETCandInfoCollection pfcsChPU = selectCandidates( pfCandidates, minPtDef_, maxPtDef_,
								reco::PUSubMETCandInfo::kChPU, true, 
								NoPileUpMEtUtilities::kAll);
  
  reco::PUSubMETCandInfoCollection pfcsNUnclustered = selectCandidates( pfCandidates, minPtDef_, maxPtDef_, 
									reco::PUSubMETCandInfo::kNeutral, false,
									NoPileUpMEtUtilities::kOutsideJet);
  
  reco::PUSubMETCandInfoCollection jetsHS = selectCandidates(jets, 10.0, maxPtDef_, reco::PUSubMETCandInfo::kHS, false,
							     NoPileUpMEtUtilities::kAll);
  reco::PUSubMETCandInfoCollection jetsPU = selectCandidates(jets, 10.0, maxPtDef_, reco::PUSubMETCandInfo::kPU, false,
							     NoPileUpMEtUtilities::kAll);

  //not used so far
  //_chPfcSum = computeCandidateSum(pfcsCh, false, &_chPfcSumAbsPx, &_chPfcSumAbsPy);
  chHSPfcSum_ = computeCandidateSum(pfcsChHS, false, chHSPfcSumAbsPx_, chHSPfcSumAbsPy_);
  chPUPfcSum_ = computeCandidateSum(pfcsChPU, false, chPUPfcSumAbsPx_, chPUPfcSumAbsPy_);
  nUncPfcSum_ = computeCandidateSum(pfcsNUnclustered, false, nUncPfcSumAbsPx_, nUncPfcSumAbsPy_);

  nHSJetSum_ = computeCandidateSum(jetsHS, true, nHSJetSumAbsPx_, nHSJetSumAbsPy_);
  nPUJetSum_ = computeCandidateSum(jetsPU, true, nPUJetSumAbsPx_, nPUJetSumAbsPy_);
  
}
  
CommonMETData 
NoPileUpMEtUtilities::computeRecoil(int metType, double& sumAbsPx, double& sumAbsPy) {
  CommonMETData retVal;
  double retSumAbsPx = 0.;
  double retSumAbsPy = 0.;

  //never used....
  // if(metType==NoPileUpMEtUtilities::kChMET) {
  //   CommonMETData chPfcSum = computeCandidateSum(_pfcsCh, false, &trackSumAbsPx, &trackSumAbsPy);
  //   retVal.mex   = -chPfcSum.mex;
  //   retVal.mey   = -chPfcSum.mey;
  //   retVal.sumet = chPfcSum.sumet;
  //   retSumAbsPx = ; 
  //   retSumAbsPy = ; 
  // }
  if(metType==NoPileUpMEtUtilities::kChHSMET) {
    retVal.mex   = chHSPfcSum_.mex;
    retVal.mey   = chHSPfcSum_.mey;
    retVal.sumet = chHSPfcSum_.sumet;
    retSumAbsPx = chHSPfcSumAbsPx_; 
    retSumAbsPy = chHSPfcSumAbsPy_; 
  }
  if(metType==NoPileUpMEtUtilities::kChPUMET) {
    retVal.mex   = chPUPfcSum_.mex;
    retVal.mey   = chPUPfcSum_.mey;
    retVal.sumet = chPUPfcSum_.sumet;
    retSumAbsPx = chPUPfcSumAbsPx_; 
    retSumAbsPy = chPUPfcSumAbsPy_; 
  }
  if(metType==NoPileUpMEtUtilities::kNeutralUncMET) {
    retVal.mex   = nUncPfcSum_.mex;
    retVal.mey   = nUncPfcSum_.mey;
    retVal.sumet = nUncPfcSum_.sumet;
    retSumAbsPx = nUncPfcSumAbsPx_; 
    retSumAbsPy = nUncPfcSumAbsPy_; 
  }
  if(metType==NoPileUpMEtUtilities::kHadronicHSMET) {
    retVal.mex   = chHSPfcSum_.mex + nHSJetSum_.mex;
    retVal.mey   = chHSPfcSum_.mey + nHSJetSum_.mey;
    retVal.sumet = chHSPfcSum_.sumet + nHSJetSum_.sumet;
    retSumAbsPx = chHSPfcSumAbsPx_ + nHSJetSumAbsPx_; 
    retSumAbsPy = chHSPfcSumAbsPy_ + nHSJetSumAbsPy_; 
  }
  if(metType==NoPileUpMEtUtilities::kHadronicPUMET) {
    retVal.mex   = chPUPfcSum_.mex + nHSJetSum_.mex;
    retVal.mey   = chPUPfcSum_.mey + nHSJetSum_.mey;
    retVal.sumet = chPUPfcSum_.sumet + nHSJetSum_.sumet;
    retSumAbsPx = chPUPfcSumAbsPx_ + nHSJetSumAbsPx_; 
    retSumAbsPy = chPUPfcSumAbsPy_ + nHSJetSumAbsPy_; 
  }

 sumAbsPx = retSumAbsPx;
 sumAbsPy = retSumAbsPy;
 
 return retVal;
}
