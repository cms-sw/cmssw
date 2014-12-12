#include "RecoMET/METPUSubtraction/interface/mvaMEtUtilities.h"

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
  //Tight Id => not used
  mvaCut_[0][0][0] =  0.5; mvaCut_[0][0][1] = 0.6; mvaCut_[0][0][2] = 0.6; mvaCut_[0][0][3] = 0.9;
  mvaCut_[0][1][0] = -0.2; mvaCut_[0][1][1] = 0.2; mvaCut_[0][1][2] = 0.2; mvaCut_[0][1][3] = 0.6;
  mvaCut_[0][2][0] =  0.3; mvaCut_[0][2][1] = 0.4; mvaCut_[0][2][2] = 0.7; mvaCut_[0][2][3] = 0.8;
  mvaCut_[0][3][0] =  0.5; mvaCut_[0][3][1] = 0.4; mvaCut_[0][3][2] = 0.8; mvaCut_[0][3][3] = 0.9;
  //Medium id => not used
  mvaCut_[1][0][0] =  0.2; mvaCut_[1][0][1] = 0.4; mvaCut_[1][0][2] = 0.2; mvaCut_[1][0][3] = 0.6;
  mvaCut_[1][1][0] = -0.3; mvaCut_[1][1][1] = 0. ; mvaCut_[1][1][2] = 0. ; mvaCut_[1][1][3] = 0.5;
  mvaCut_[1][2][0] =  0.2; mvaCut_[1][2][1] = 0.2; mvaCut_[1][2][2] = 0.5; mvaCut_[1][2][3] = 0.7;
  mvaCut_[1][3][0] =  0.3; mvaCut_[1][3][1] = 0.2; mvaCut_[1][3][2] = 0.7; mvaCut_[1][3][3] = 0.8;
  //Met Id => used
  mvaCut_[2][0][0] = -0.2; mvaCut_[2][0][1] = -0.3; mvaCut_[2][0][2] = -0.5; mvaCut_[2][0][3] = -0.5;
  mvaCut_[2][1][0] = -0.2; mvaCut_[2][1][1] = -0.2; mvaCut_[2][1][2] = -0.5; mvaCut_[2][1][3] = -0.3;
  mvaCut_[2][2][0] = -0.2; mvaCut_[2][2][1] = -0.2; mvaCut_[2][2][2] = -0.2; mvaCut_[2][2][3] =  0.1;
  mvaCut_[2][3][0] = -0.2; mvaCut_[2][3][1] = -0.2; mvaCut_[2][3][2] =  0. ; mvaCut_[2][3][3] =  0.2;

  _dzCut = cfg.getParameter<double>("dZcut");
  _ptThreshold = ( cfg.exists("ptThreshold") ) ?
    cfg.getParameter<int>("ptThreshold") : -1000;
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
  if ( std::abs(jetP4.eta()) > 2.5  && std::abs(jetP4.eta()) < 2.75) etaBin = 1; 
  if ( std::abs(jetP4.eta()) > 2.75 && std::abs(jetP4.eta()) < 3.0 ) etaBin = 2; 
  if ( std::abs(jetP4.eta()) > 3.0  && std::abs(jetP4.eta()) < 5.0 ) etaBin = 3; 

  return ( mvaJetId > mvaCut_[2][ptBin][etaBin] );
}

reco::Candidate::LorentzVector mvaMEtUtilities::leadJetP4(const std::vector<reco::PUSubMETCandInfo>& jets) 
{
  return jetP4(jets, 0);
}

reco::Candidate::LorentzVector mvaMEtUtilities::subleadJetP4(const std::vector<reco::PUSubMETCandInfo>& jets) 
{
  return jetP4(jets, 1);
}

reco::Candidate::LorentzVector mvaMEtUtilities::jetP4(const std::vector<reco::PUSubMETCandInfo>& jets, unsigned idx) 
{
  reco::Candidate::LorentzVector retVal(0.,0.,0.,0.);
  if ( idx < jets.size() ) {
    std::vector<reco::PUSubMETCandInfo> jets_sorted = jets;
    std::sort(jets_sorted.begin(), jets_sorted.end()); 
    retVal = jets_sorted[idx].p4_;
  }
  return retVal;
}
unsigned mvaMEtUtilities::numJetsAboveThreshold(const std::vector<reco::PUSubMETCandInfo>& jets, double ptThreshold) 
{
  unsigned retVal = 0;
  for ( std::vector<reco::PUSubMETCandInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    if ( jet->p4_.pt() > ptThreshold ) ++retVal;
  }
  return retVal;
}
std::vector<reco::PUSubMETCandInfo> mvaMEtUtilities::cleanJets(const std::vector<reco::PUSubMETCandInfo>&    jets, 
								 const std::vector<reco::PUSubMETCandInfo>& leptons,
								 double ptThreshold, double dRmatch)
{

  double dR2match = dRmatch*dRmatch;
  std::vector<reco::PUSubMETCandInfo> retVal;
  for ( std::vector<reco::PUSubMETCandInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    bool isOverlap = false;
    for ( std::vector<reco::PUSubMETCandInfo>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR2(jet->p4_, lepton->p4_) < dR2match ) isOverlap = true;	
    }
    if ( jet->p4_.pt() > ptThreshold && !isOverlap ) retVal.push_back(*jet);
  }
  return retVal;
}

std::vector<reco::PUSubMETCandInfo> mvaMEtUtilities::cleanPFCands(const std::vector<reco::PUSubMETCandInfo>& pfCandidates, 
								     const std::vector<reco::PUSubMETCandInfo>& leptons,
								     double dRmatch, bool invert)
{

  double dR2match = dRmatch*dRmatch;
  std::vector<reco::PUSubMETCandInfo> retVal;
  for ( std::vector<reco::PUSubMETCandInfo>::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    bool isOverlap = false;
    for ( std::vector<reco::PUSubMETCandInfo>::const_iterator lepton = leptons.begin();
	  lepton != leptons.end(); ++lepton ) {
      if ( deltaR2(pfCandidate->p4_, lepton->p4_) < dR2match ) isOverlap = true;
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

CommonMETData 
mvaMEtUtilities::computeCandSum( int compKey, double dZmax, int dZflag,
				 bool iCharged,  bool mvaPassFlag,
				 const std::vector<reco::PUSubMETCandInfo>& objects ) {

  CommonMETData retVal;
  retVal.mex   = 0.;
  retVal.mey   = 0.;
  retVal.sumet = 0.;

  for ( std::vector<reco::PUSubMETCandInfo>::const_iterator object = objects.begin();
	object != objects.end(); ++object ) {

    double pFrac = 1;

    //pf candidates
    // dZcut
    //   maximum distance within which tracks are 
    //considered to be associated to hard scatter vertex
    // dZflag 
    //   0 : select charged PFCandidates originating from hard scatter vertex
    //   1 : select charged PFCandidates originating from pile-up vertices
    //   2 : select all PFCandidates
    if( compKey==mvaMEtUtilities::kPFCands ) {
      if ( object->dZ_ < 0.    && dZflag != 2 ) continue;
      if ( object->dZ_ > dZmax && dZflag == 0 ) continue;
      if ( object->dZ_ < dZmax && dZflag == 1 ) continue;
    }

    //leptons
    if( compKey==mvaMEtUtilities::kLeptons) {
      if(iCharged) pFrac = object->chargedEnFrac_;
    }

    //jets
    if( compKey==mvaMEtUtilities::kJets) {
      bool passesMVAjetId = passesMVA(object->p4_, object->mva_);
      
      if (  passesMVAjetId && !mvaPassFlag ) continue;
      if ( !passesMVAjetId &&  mvaPassFlag ) continue;
   
      pFrac = 1-object->chargedEnFrac_;//neutral energy fraction
    }

    retVal.mex   += object->p4_.px()*pFrac;
    retVal.mey   += object->p4_.py()*pFrac;
    retVal.sumet += object->p4_.pt()*pFrac;
  }

  finalize(retVal);
  return retVal;
}


CommonMETData
mvaMEtUtilities::computeRecoil(int metType) {

  CommonMETData retVal;

  if(metType == mvaMEtUtilities::kPF ) {
    //MET = pfMET = - all candidates 
    // MET (1) in JME-13-003
    retVal.mex  = _leptonsSum.mex - _pfCandSum.mex;
    retVal.mey  = _leptonsSum.mey - _pfCandSum.mey;
    retVal.sumet = _pfCandSum.sumet - _leptonsSum.sumet;
  }
  if(metType == mvaMEtUtilities::kChHS ) {
    //MET = - charged HS
    // MET (2) in JME-13-003
    retVal.mex  = _leptonsChSum.mex - _pfCandChHSSum.mex;
    retVal.mey  = _leptonsChSum.mey - _pfCandChHSSum.mey;
    retVal.sumet = _pfCandChHSSum.sumet - _leptonsChSum.sumet;
  }
  if(metType == mvaMEtUtilities::kHS ) { 
    //MET = - charged HS - neutral HS in jets
    // MET (3) in JME-13-003
    retVal.mex  = _leptonsChSum.mex - (_pfCandChHSSum.mex + _neutralJetHSSum.mex);
    retVal.mey  = _leptonsChSum.mey - (_pfCandChHSSum.mey + _neutralJetHSSum.mey);
    retVal.sumet = _pfCandChHSSum.sumet + _neutralJetHSSum.sumet - _leptonsChSum.sumet;
  }
  if(metType == mvaMEtUtilities::kPU ) {
    //MET = - charged PU - neutral PU in jets  
    //MET = -recoil in that particular case, - sign not useful for the MVA and then discarded
    //motivated as PU IS its own recoil
    // MET (4) in JME-13-003
    retVal.mex   = -(_pfCandChPUSum.mex + _neutralJetPUSum.mex);
    retVal.mey   = -(_pfCandChPUSum.mey + _neutralJetPUSum.mey);
    retVal.sumet = _pfCandChPUSum.sumet + _neutralJetPUSum.sumet;
  }
  if(metType == mvaMEtUtilities::kHSMinusNeutralPU ) {
    //MET = all candidates - charged PU - neutral PU in jets
    // = all charged HS + all neutrals - neutral PU in jets
    // MET (5) in JME-13-003
    retVal.mex  = _leptonsSum.mex - (_pfCandSum.mex - _pfCandChPUSum.mex - _neutralJetPUSum.mex);
    retVal.mey  = _leptonsSum.mey - (_pfCandSum.mey - _pfCandChPUSum.mey - _neutralJetPUSum.mey);
    retVal.sumet = (_pfCandSum.sumet - _pfCandChPUSum.sumet - _neutralJetPUSum.sumet) -_leptonsSum.sumet;
  }

  finalize(retVal);
  return retVal;
}

void
mvaMEtUtilities::computeAllSums(const std::vector<reco::PUSubMETCandInfo>& jets, 
			       const std::vector<reco::PUSubMETCandInfo>& leptons,
			       const std::vector<reco::PUSubMETCandInfo>& pfCandidates ) {

  _cleanedJets = cleanJets(jets, leptons, _ptThreshold, 0.5);

  _leptonsSum = computeCandSum( kLeptons, 0., 0, false , false, leptons );
  _leptonsChSum = computeCandSum( kLeptons, 0., 0, true , false, leptons);
  _pfCandSum = computeCandSum( kPFCands, _dzCut, 2, false , false, pfCandidates);
  _pfCandChHSSum = computeCandSum( kPFCands, _dzCut, 0, false , false, pfCandidates);
  _pfCandChPUSum = computeCandSum( kPFCands, _dzCut, 1, false , false, pfCandidates);
  _neutralJetHSSum = computeCandSum( kJets, 0., 0, false , true, jets );
  _neutralJetPUSum = computeCandSum( kJets, 0., 0, false , false, jets );

}

double
mvaMEtUtilities::getLeptonsSumMEX() {
  return _leptonsSum.mex;
}

double
mvaMEtUtilities::getLeptonsSumMEY() {
  return _leptonsSum.mey;
}

double
mvaMEtUtilities::getLeptonsChSumMEX() {
  return _leptonsChSum.mex;
}

double
mvaMEtUtilities::getLeptonsChSumMEY() {
  return _leptonsChSum.mey;
} 


std::vector<reco::PUSubMETCandInfo> 
mvaMEtUtilities::getCleanedJets() {
  return _cleanedJets;
}
