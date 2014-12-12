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
