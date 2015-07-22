#include "RecoMET/METPUSubtraction/interface/MvaMEtUtilities.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>
#include <cmath>

MvaMEtUtilities::MvaMEtUtilities(const edm::ParameterSet& cfg) 
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

  dzCut_ = cfg.getParameter<double>("dZcut");
  ptThreshold_ = ( cfg.exists("ptThreshold") ) ?
    cfg.getParameter<int>("ptThreshold") : -1000;
}

MvaMEtUtilities::~MvaMEtUtilities() 
{
// nothing to be done yet...
}

bool MvaMEtUtilities::passesMVA(const reco::Candidate::LorentzVector& jetP4, double mvaJetId) 
{ 
  int ptBin = 0; 
  if ( jetP4.pt() >= 10. && jetP4.pt() < 20. ) ptBin = 1;
  if ( jetP4.pt() >= 20. && jetP4.pt() < 30. ) ptBin = 2;
  if ( jetP4.pt() >= 30.                     ) ptBin = 3;
  
  int etaBin = 0;
  if ( std::abs(jetP4.eta()) >= 2.5  && std::abs(jetP4.eta()) < 2.75) etaBin = 1; 
  if ( std::abs(jetP4.eta()) >= 2.75 && std::abs(jetP4.eta()) < 3.0 ) etaBin = 2; 
  if ( std::abs(jetP4.eta()) >= 3.0  && std::abs(jetP4.eta()) < 5.0 ) etaBin = 3; 

  return ( mvaJetId > mvaCut_[2][ptBin][etaBin] );
}

reco::Candidate::LorentzVector MvaMEtUtilities::leadJetP4(const std::vector<reco::PUSubMETCandInfo>& jets) 
{
  return jetP4(jets, 0);
}

reco::Candidate::LorentzVector MvaMEtUtilities::subleadJetP4(const std::vector<reco::PUSubMETCandInfo>& jets) 
{
  return jetP4(jets, 1);
}

reco::Candidate::LorentzVector MvaMEtUtilities::jetP4(const std::vector<reco::PUSubMETCandInfo>& jets, unsigned idx) 
{
  reco::Candidate::LorentzVector retVal(0.,0.,0.,0.);
  if ( idx < jets.size() ) {
    std::vector<reco::PUSubMETCandInfo> jets_sorted = jets;
    std::sort(jets_sorted.rbegin(), jets_sorted.rend());
    retVal = jets_sorted[idx].p4();
  }
  return retVal;
}
unsigned MvaMEtUtilities::numJetsAboveThreshold(const std::vector<reco::PUSubMETCandInfo>& jets, double ptThreshold) 
{
  unsigned retVal = 0;
  for ( std::vector<reco::PUSubMETCandInfo>::const_iterator jet = jets.begin();
	jet != jets.end(); ++jet ) {
    if ( jet->p4().pt() > ptThreshold ) ++retVal;
  }
  return retVal;
}
std::vector<reco::PUSubMETCandInfo> MvaMEtUtilities::cleanJets(const std::vector<reco::PUSubMETCandInfo>&    jets, 
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
      if ( deltaR2(jet->p4(), lepton->p4()) < dR2match ) isOverlap = true;	
    }
    if ( jet->p4().pt() > ptThreshold && !isOverlap ) retVal.push_back(*jet);
  }
  return retVal;
}

std::vector<reco::PUSubMETCandInfo> MvaMEtUtilities::cleanPFCands(const std::vector<reco::PUSubMETCandInfo>& pfCandidates, 
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
      if ( deltaR2(pfCandidate->p4(), lepton->p4()) < dR2match ) isOverlap = true;
    }
    if ( (!isOverlap && !invert) || (isOverlap && invert) ) retVal.push_back(*pfCandidate);
  }
  return retVal;
}
void MvaMEtUtilities::finalize(CommonMETData& metData)
{
  metData.met = sqrt(metData.mex*metData.mex + metData.mey*metData.mey);
  metData.mez = 0.;
  metData.phi = atan2(metData.mey, metData.mex);
}

CommonMETData 
MvaMEtUtilities::computeCandSum( int compKey, double dZmax, int dZflag,
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
    if( compKey==MvaMEtUtilities::kPFCands ) {
      if ( object->dZ() < 0.    && dZflag != 2 ) continue;
      if ( object->dZ() > dZmax && dZflag == 0 ) continue;
      if ( object->dZ() < dZmax && dZflag == 1 ) continue;
    }

    //leptons
    if( compKey==MvaMEtUtilities::kLeptons) {
      if(iCharged) pFrac = object->chargedEnFrac();
    }

    //jets
    if( compKey==MvaMEtUtilities::kJets) {
      bool passesMVAjetId = passesMVA(object->p4(), object->mva() );
      
      if (  passesMVAjetId && !mvaPassFlag ) continue;
      if ( !passesMVAjetId &&  mvaPassFlag ) continue;
   
      pFrac = 1-object->chargedEnFrac();//neutral energy fraction
    }

    retVal.mex   += object->p4().px()*pFrac;
    retVal.mey   += object->p4().py()*pFrac;
    retVal.sumet += object->p4().pt()*pFrac;
  }

  finalize(retVal);
  return retVal;
}


CommonMETData
MvaMEtUtilities::computeRecoil(int metType) {

  CommonMETData retVal;

  if(metType == MvaMEtUtilities::kPF ) {
    //MET = pfMET = - all candidates 
    // MET (1) in JME-13-003
    retVal.mex  = leptonsSum_.mex - pfCandSum_.mex;
    retVal.mey  = leptonsSum_.mey - pfCandSum_.mey;
    retVal.sumet = pfCandSum_.sumet - leptonsSum_.sumet;
  }
  if(metType == MvaMEtUtilities::kChHS ) {
    //MET = - charged HS
    // MET (2) in JME-13-003
    retVal.mex  = leptonsChSum_.mex - pfCandChHSSum_.mex;
    retVal.mey  = leptonsChSum_.mey - pfCandChHSSum_.mey;
    retVal.sumet = pfCandChHSSum_.sumet - leptonsChSum_.sumet;
  }
  if(metType == MvaMEtUtilities::kHS ) { 
    //MET = - charged HS - neutral HS in jets
    // MET (3) in JME-13-003
    retVal.mex  = leptonsChSum_.mex - (pfCandChHSSum_.mex + neutralJetHSSum_.mex);
    retVal.mey  = leptonsChSum_.mey - (pfCandChHSSum_.mey + neutralJetHSSum_.mey);
    retVal.sumet = pfCandChHSSum_.sumet + neutralJetHSSum_.sumet - leptonsChSum_.sumet;
  }
  if(metType == MvaMEtUtilities::kPU ) {
    //MET = - charged PU - neutral PU in jets  
    //MET = -recoil in that particular case, - sign not useful for the MVA and then discarded
    //motivated as PU IS its own recoil
    // MET (4) in JME-13-003
    retVal.mex   = -(pfCandChPUSum_.mex + neutralJetPUSum_.mex);
    retVal.mey   = -(pfCandChPUSum_.mey + neutralJetPUSum_.mey);
    retVal.sumet = pfCandChPUSum_.sumet + neutralJetPUSum_.sumet;
  }
  if(metType == MvaMEtUtilities::kHSMinusNeutralPU ) {
    //MET = all candidates - charged PU - neutral PU in jets
    // = all charged HS + all neutrals - neutral PU in jets
    // MET (5) in JME-13-003
    retVal.mex  = leptonsSum_.mex - (pfCandSum_.mex - pfCandChPUSum_.mex - neutralJetPUSum_.mex);
    retVal.mey  = leptonsSum_.mey - (pfCandSum_.mey - pfCandChPUSum_.mey - neutralJetPUSum_.mey);
    retVal.sumet = (pfCandSum_.sumet - pfCandChPUSum_.sumet - neutralJetPUSum_.sumet) -leptonsSum_.sumet;
  }

  finalize(retVal);
  return retVal;
}

void
MvaMEtUtilities::computeAllSums(const std::vector<reco::PUSubMETCandInfo>& jets, 
			       const std::vector<reco::PUSubMETCandInfo>& leptons,
			       const std::vector<reco::PUSubMETCandInfo>& pfCandidates ) {

  cleanedJets_ = cleanJets(jets, leptons, ptThreshold_, 0.5);

  leptonsSum_ = computeCandSum( kLeptons, 0., 0, false , false, leptons );
  leptonsChSum_ = computeCandSum( kLeptons, 0., 0, true , false, leptons);
  pfCandSum_ = computeCandSum( kPFCands, dzCut_, 2, false , false, pfCandidates);
  pfCandChHSSum_ = computeCandSum( kPFCands, dzCut_, 0, false , false, pfCandidates);
  pfCandChPUSum_ = computeCandSum( kPFCands, dzCut_, 1, false , false, pfCandidates);
  neutralJetHSSum_ = computeCandSum( kJets, 0., 0, false , true, jets );
  neutralJetPUSum_ = computeCandSum( kJets, 0., 0, false , false, jets );

}

double
MvaMEtUtilities::getLeptonsSumMEX() const {
  return leptonsSum_.mex;
}

double
MvaMEtUtilities::getLeptonsSumMEY() const {
  return leptonsSum_.mey;
}

double
MvaMEtUtilities::getLeptonsChSumMEX() const {
  return leptonsChSum_.mex;
}

double
MvaMEtUtilities::getLeptonsChSumMEY() const {
  return leptonsChSum_.mey;
} 


const std::vector<reco::PUSubMETCandInfo>& 
MvaMEtUtilities::getCleanedJets() const {
  return cleanedJets_;
}
