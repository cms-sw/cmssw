#include "RecoMET/METAlgorithms/interface/mvaMEtUtilities.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>
#include <math.h>

mvaMEtUtilities::mvaMEtUtilities(const edm::ParameterSet& cfg) 
{
  // jet id
  /* ===> this code parses an xml it uses parameter set 
  edm::ParameterSet jetConfig = iConfig.getParameter<edm::ParameterSet>("JetIdParams");
  for(int i0 = 0; i0 < 3; i0++) { 
    std::string lCutType                            = "Tight";
    if(i0 == PileupJetIdentifier::kMedium) lCutType = "Medium";
    if(i0 == PileupJetIdentifier::kLoose)  lCutType = "Loose";
    std::vector<double> pt010  = jetConfig.getParameter<std::vector<double> >(("Pt010_" +lCutType).c_str());
    std::vector<double> pt1020 = jetConfig.getParameter<std::vector<double> >(("Pt1020_"+lCutType).c_str());
    std::vector<double> pt2030 = jetConfig.getParameter<std::vector<double> >(("Pt2030_"+lCutType).c_str());
    std::vector<double> pt3050 = jetConfig.getParameter<std::vector<double> >(("Pt3050_"+lCutType).c_str());
    for(int i1 = 0; i1 < 4; i1++) mvacut_[i0][0][i1] = pt010 [i1];
    for(int i1 = 0; i1 < 4; i1++) mvacut_[i0][1][i1] = pt1020[i1];
    for(int i1 = 0; i1 < 4; i1++) mvacut_[i0][2][i1] = pt2030[i1];
    for(int i1 = 0; i1 < 4; i1++) mvacut_[i0][3][i1] = pt3050[i1];
    }
  */
  //Tight Id
  mvaCut_[0][0][0] =  0.5; mvaCut_[0][0][1] = 0.6; mvaCut_[0][0][2] = 0.6; mvaCut_[0][0][3] = 0.9;
  mvaCut_[0][1][0] = -0.2; mvaCut_[0][1][1] = 0.2; mvaCut_[0][1][2] = 0.2; mvaCut_[0][1][3] = 0.6;
  mvaCut_[0][2][0] =  0.3; mvaCut_[0][2][1] = 0.4; mvaCut_[0][2][2] = 0.7; mvaCut_[0][2][3] = 0.8;
  mvaCut_[0][3][0] =  0.5; mvaCut_[0][3][1] = 0.4; mvaCut_[0][3][2] = 0.8; mvaCut_[0][3][3] = 0.9;
  //Medium id
  mvaCut_[1][0][0] =  0.2; mvaCut_[1][0][1] = 0.4; mvaCut_[1][0][2] = 0.2; mvaCut_[1][0][3] = 0.6;
  mvaCut_[1][1][0] = -0.3; mvaCut_[1][1][1] = 0. ; mvaCut_[1][1][2] = 0. ; mvaCut_[1][1][3] = 0.5;
  mvaCut_[1][2][0] =  0.2; mvaCut_[1][2][1] = 0.2; mvaCut_[1][2][2] = 0.5; mvaCut_[1][2][3] = 0.7;
  mvaCut_[1][3][0] =  0.3; mvaCut_[1][3][1] = 0.2; mvaCut_[1][3][2] = 0.7; mvaCut_[1][3][3] = 0.8;
  //Loose Id 
  mvaCut_[2][0][0] = -0.2; mvaCut_[2][0][1] =  0. ; mvaCut_[2][0][2] =  0.2; mvaCut_[2][0][3] =  0.5;
  mvaCut_[2][1][0] =  0.2; mvaCut_[2][1][1] = -0.6; mvaCut_[2][1][2] = -0.6; mvaCut_[2][1][3] = -0.4;
  mvaCut_[2][2][0] =  0.2; mvaCut_[2][2][1] = -0.6; mvaCut_[2][2][2] = -0.6; mvaCut_[2][2][3] = -0.4;
  mvaCut_[2][3][0] =  0.2; mvaCut_[2][3][1] = -0.8; mvaCut_[2][3][2] = -0.8; mvaCut_[2][3][3] = -0.4;
}

mvaMEtUtilities::~mvaMEtUtilities() 
{
// nothing to be done yet...
}

bool mvaMEtUtilities::passesMVA(const reco::Candidate::LorentzVector& jetP4, double mvaJetId) 
{ 
  int ptBin = 0; 
  if ( jetP4.pt() > 10. && jetP4.pt() < 20. ) ptBin = 1;
  if ( jetP4.pt() > 20. && jetP4.pt() < 30. ) ptBin = 2;
  if ( jetP4.pt() > 30.                     ) ptBin = 3;
  
  int etaBin = 0;
  if ( fabs(jetP4.eta()) > 2.5  && fabs(jetP4.eta()) < 2.75) etaBin = 1; 
  if ( fabs(jetP4.eta()) > 2.75 && fabs(jetP4.eta()) < 3.0 ) etaBin = 2; 
  if ( fabs(jetP4.eta()) > 3.0  && fabs(jetP4.eta()) < 5.0 ) etaBin = 3; 

  return ( mvaJetId > mvaCut_[2][ptBin][etaBin] );
}

reco::Candidate::LorentzVector mvaMEtUtilities::leadJetP4(const std::vector<JetInfo>& jets) 
{
  return jetP4(jets, 0);
}

reco::Candidate::LorentzVector mvaMEtUtilities::subleadJetP4(const std::vector<JetInfo>& jets) 
{
  return jetP4(jets, 1);
}

bool operator<(const mvaMEtUtilities::JetInfo& jet1, const mvaMEtUtilities::JetInfo& jet2)
{
  return jet1.p4_.pt() > jet2.p4_.pt();
} 

reco::Candidate::LorentzVector mvaMEtUtilities::jetP4(const std::vector<JetInfo>& jets, unsigned idx) 
{
  reco::Candidate::LorentzVector retVal(0.,0.,0.,0.);
  if ( idx < jets.size() ) {
    std::vector<JetInfo> jets_sorted = jets;
    std::sort(jets_sorted.begin(), jets_sorted.end()); 
    retVal = jets_sorted[idx].p4_;
  }
  return retVal;
}
unsigned mvaMEtUtilities::numJetsAboveThreshold(const std::vector<JetInfo>& jets, double ptThreshold) 
{
  unsigned retVal = 0;
  for ( std::vector<JetInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    if ( jet->p4_.pt() > ptThreshold ) ++retVal;
  }
  return retVal;
}
std::vector<mvaMEtUtilities::JetInfo> mvaMEtUtilities::cleanJets(const std::vector<JetInfo>&    jets, 
								 const std::vector<leptonInfo>& leptons,
								 double ptThreshold, double dRmatch)
{
  std::vector<JetInfo> retVal;
  for ( std::vector<JetInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    bool isOverlap = false;
    for ( std::vector<leptonInfo>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR(jet->p4_, lepton->p4_) < dRmatch ) isOverlap = true;	
    }
    if ( jet->p4_.pt() > ptThreshold && !isOverlap ) retVal.push_back(*jet);
  }
  return retVal;
}

std::vector<mvaMEtUtilities::pfCandInfo> mvaMEtUtilities::cleanPFCands(const std::vector<pfCandInfo>& pfCandidates, 
								       const std::vector<leptonInfo>& leptons,
								       double dRmatch, bool invert)
{
  std::vector<pfCandInfo> retVal;
  for ( std::vector<pfCandInfo>::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    bool isOverlap = false;
    for ( std::vector<leptonInfo>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR(pfCandidate->p4_, lepton->p4_) < dRmatch ) isOverlap = true;
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*pfCandidate);
  }
  return retVal;
}
void mvaMEtUtilities::finalize(CommonMETData& metData)
{
  metData.met = sqrt(metData.mex*metData.mex + metData.mey*metData.mey);
  metData.mez = 0.;
  metData.phi = atan2(metData.mey, metData.mex);
}

CommonMETData mvaMEtUtilities::computePFCandSum(const std::vector<pfCandInfo>& pfCandidates, double dZmax, int dZflag)
{
  // dZcut
  //   maximum distance within which tracks are considered to be associated to hard scatter vertex
  // dZflag 
  //   0 : select charged PFCandidates originating from hard scatter vertex
  //   1 : select charged PFCandidates originating from pile-up vertices
  //   2 : select all PFCandidates
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  for ( std::vector<pfCandInfo>::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    if ( pfCandidate->dZ_ < 0.    && dZflag != 2 ) continue;
    if ( pfCandidate->dZ_ > dZmax && dZflag == 0 ) continue;
    if ( pfCandidate->dZ_ < dZmax && dZflag == 1 ) continue;
    retVal.mex   += pfCandidate->p4_.px();
    retVal.mey   += pfCandidate->p4_.py();
    retVal.sumet += pfCandidate->p4_.pt();
  }
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeSumLeptons(const std::vector<mvaMEtUtilities::leptonInfo>& leptons, bool iCharged)
{
  //Computes vectorial sum of charged(iCharged == true) or all(iCharged == false) components
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  for ( std::vector<leptonInfo>::const_iterator lepton = leptons.begin();
	lepton != leptons.end(); ++lepton ) {
    double pChargedFrac = 1;
    if(iCharged) pChargedFrac = lepton->chargedFrac_;
    retVal.mex   += lepton->p4_.px()*pChargedFrac;
    retVal.mey   += lepton->p4_.py()*pChargedFrac;
    retVal.sumet += lepton->p4_.pt()*pChargedFrac;
  }
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeJetSum_neutral(const std::vector<JetInfo>& jets, bool mvaPassFlag)
{
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  for ( std::vector<JetInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    bool passesMVAjetId = passesMVA(jet->p4_, jet->mva_);
    if (  passesMVAjetId && !mvaPassFlag ) continue;
    if ( !passesMVAjetId &&  mvaPassFlag ) continue;
    retVal.mex   += jet->p4_.px()*jet->neutralEnFrac_;
    retVal.mey   += jet->p4_.py()*jet->neutralEnFrac_;
    retVal.sumet += jet->p4_.pt()*jet->neutralEnFrac_;
  }
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computePUMEt(const std::vector<pfCandInfo>& pfCandidates, 
					    const std::vector<JetInfo>& jets, double dZcut)
{
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  CommonMETData trackSumPU = computePFCandSum(pfCandidates, dZcut, 1);
  CommonMETData jetSumPU_neutral = computeJetSum_neutral(jets, false);
  retVal.mex   = -(trackSumPU.mex + jetSumPU_neutral.mex);
  retVal.mey   = -(trackSumPU.mey + jetSumPU_neutral.mey);
  retVal.sumet = trackSumPU.sumet + jetSumPU_neutral.sumet;
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeNegPFRecoil(const CommonMETData& leptons, 
						  const std::vector<pfCandInfo>& pfCandidates, double dZcut)
{
  CommonMETData retVal;
  CommonMETData pfCandSum = computePFCandSum(pfCandidates, dZcut, 2);
  retVal.mex   = -pfCandSum.mex + leptons.mex; 
  retVal.mey   = -pfCandSum.mey + leptons.mey;
  retVal.sumet = pfCandSum.sumet - leptons.sumet;
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeNegTrackRecoil(const CommonMETData& leptons, 
						     const std::vector<pfCandInfo>& pfCandidates, double dZcut)
{
  CommonMETData retVal;
  CommonMETData trackSum = computePFCandSum(pfCandidates, dZcut, 0);
  retVal.mex   = -trackSum.mex  + leptons.mex; 
  retVal.mey   = -trackSum.mey  + leptons.mey;
  retVal.sumet = trackSum.sumet - leptons.sumet;
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeNegNoPURecoil(const CommonMETData& leptons,
						    const std::vector<pfCandInfo>& pfCandidates, 
						    const std::vector<JetInfo>& jets, double dZcut)
{
  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  CommonMETData trackSumNoPU = computePFCandSum(pfCandidates, dZcut, 0);
  CommonMETData jetSumNoPU_neutral = computeJetSum_neutral(jets, true);
  retVal.mex   = -(trackSumNoPU.mex + jetSumNoPU_neutral.mex)  + leptons.mex;
  retVal.mey   = -(trackSumNoPU.mey + jetSumNoPU_neutral.mey)  + leptons.mey;
  retVal.sumet = trackSumNoPU.sumet + jetSumNoPU_neutral.sumet - leptons.sumet;
  finalize(retVal);
  return retVal;
}

CommonMETData mvaMEtUtilities::computeNegPUCRecoil(const CommonMETData& leptons, 
						   const std::vector<pfCandInfo>& pfCandidates, 
						   const std::vector<JetInfo>& jets, double dZcut)
{
   CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;
  CommonMETData pfCandSum = computePFCandSum(pfCandidates, dZcut, 2);
  CommonMETData trackSumNoPU = computePFCandSum(pfCandidates, dZcut, 1);
  CommonMETData jetSumPU_neutral = computeJetSum_neutral(jets, false);
  retVal.mex   = -(pfCandSum.mex - (trackSumNoPU.mex + jetSumPU_neutral.mex))    + leptons.mex;
  retVal.mey   = -(pfCandSum.mey - (trackSumNoPU.mey + jetSumPU_neutral.mey))    + leptons.mey;
  retVal.sumet = pfCandSum.sumet - (trackSumNoPU.sumet + jetSumPU_neutral.sumet) - leptons.sumet;
  finalize(retVal);
  return retVal;
}
  
