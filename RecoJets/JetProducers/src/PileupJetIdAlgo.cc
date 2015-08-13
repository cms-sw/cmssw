#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/Utils/interface/TMVAZipReader.h"

#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"

// ------------------------------------------------------------------------------------------
const float large_val = std::numeric_limits<float>::max();

// ------------------------------------------------------------------------------------------
PileupJetIdAlgo::PileupJetIdAlgo(const edm::ParameterSet & ps, bool runMvas) 
{
	impactParTkThreshod_ = 1.;/// ps.getParameter<double>("impactParTkThreshod");
	cutBased_ = false;
	etaBinnedWeights_ = false;
	runMvas_=runMvas;
	//std::string label    = ps.getParameter<std::string>("label");
	cutBased_ =  ps.getParameter<bool>("cutBased");
	if(!cutBased_) 
	  {
	    etaBinnedWeights_ = ps.getParameter<bool>("etaBinnedWeights");
	    if(etaBinnedWeights_){
	      tmvaWeights_jteta_0_2_        = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights_jteta_0_2")).fullPath();
	      tmvaWeights_jteta_2_2p5_      = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights_jteta_2_2p5")).fullPath();
	      tmvaWeights_jteta_2p5_3_      = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights_jteta_2p5_3")).fullPath();
	      tmvaWeights_jteta_3_5_        = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights_jteta_3_5")).fullPath();
	    }
	    else{
	      tmvaWeights_                  = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights")).fullPath();  
	    }
	    tmvaMethod_          = ps.getParameter<std::string>("tmvaMethod");
	    if(etaBinnedWeights_){
	    	tmvaVariables_jteta_0_3_       = ps.getParameter<std::vector<std::string> >("tmvaVariables_jteta_0_3");
	    	tmvaVariables_jteta_3_5_       = ps.getParameter<std::vector<std::string> >("tmvaVariables_jteta_3_5");
	    } else {
		tmvaVariables_       = ps.getParameter<std::vector<std::string> >("tmvaVariables");
	    }
	    tmvaSpectators_      = ps.getParameter<std::vector<std::string> >("tmvaSpectators");
	    version_             = ps.getParameter<int>("version");
	  }
        else version_ = USER;
	edm::ParameterSet jetConfig = ps.getParameter<edm::ParameterSet>("JetIdParams");
	for(int i0 = 0; i0 < 3; i0++) { 
	  std::string lCutType                            = "Tight";
	  if(i0 == PileupJetIdentifier::kMedium) lCutType = "Medium";
	  if(i0 == PileupJetIdentifier::kLoose)  lCutType = "Loose";
	  int nCut = 1;
	  if(cutBased_) nCut++;
	  for(int i1 = 0; i1 < nCut; i1++) {
	    std::string lFullCutType = lCutType;
	    if(cutBased_ && i1 == 0) lFullCutType = "BetaStar"+ lCutType; 
	    if(cutBased_ && i1 == 1) lFullCutType = "RMS"     + lCutType; 
	    std::vector<double> pt010  = jetConfig.getParameter<std::vector<double> >(("Pt010_" +lFullCutType).c_str());
	    std::vector<double> pt1020 = jetConfig.getParameter<std::vector<double> >(("Pt1020_"+lFullCutType).c_str());
	    std::vector<double> pt2030 = jetConfig.getParameter<std::vector<double> >(("Pt2030_"+lFullCutType).c_str());
	    std::vector<double> pt3050 = jetConfig.getParameter<std::vector<double> >(("Pt3050_"+lFullCutType).c_str());
	    if(!cutBased_) { 
	      for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][0][i2] = pt010 [i2];
	      for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][1][i2] = pt1020[i2];
	      for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][2][i2] = pt2030[i2];
	      for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][3][i2] = pt3050[i2];
	    }
	    if(cutBased_ && i1 == 0) { 
	      for(int i2 = 0; i2 < 4; i2++) betaStarCut_[i0][0][i2] = pt010 [i2];
	      for(int i2 = 0; i2 < 4; i2++) betaStarCut_[i0][1][i2] = pt1020[i2];
	      for(int i2 = 0; i2 < 4; i2++) betaStarCut_[i0][2][i2] = pt2030[i2];
	      for(int i2 = 0; i2 < 4; i2++) betaStarCut_[i0][3][i2] = pt3050[i2];
	    }
	    if(cutBased_ && i1 == 1) { 
	      for(int i2 = 0; i2 < 4; i2++) rmsCut_[i0][0][i2] = pt010 [i2];
	      for(int i2 = 0; i2 < 4; i2++) rmsCut_[i0][1][i2] = pt1020[i2];
	      for(int i2 = 0; i2 < 4; i2++) rmsCut_[i0][2][i2] = pt2030[i2];
	      for(int i2 = 0; i2 < 4; i2++) rmsCut_[i0][3][i2] = pt3050[i2];
	    }
	  }
	}
	setup();
}


// ------------------------------------------------------------------------------------------
PileupJetIdAlgo::PileupJetIdAlgo(int version,
				 const std::string & tmvaWeights, 
				 const std::string & tmvaMethod, 
				 Float_t impactParTkThreshod,
				 const std::vector<std::string> & tmvaVariables,
				 bool runMvas
	) 
{
	impactParTkThreshod_ = impactParTkThreshod;
	tmvaWeights_         = tmvaWeights;
	tmvaMethod_          = tmvaMethod;
	tmvaVariables_       = tmvaVariables;
	version_             = version;
	
	runMvas_=runMvas;
	
	setup();
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::setup()
{
	initVariables();
	
	if(etaBinnedWeights_){
	  tmvaNames_["DRweighted"] = "dR2Mean";
	  tmvaNames_["rho"] = "nvtx";
	  tmvaNames_["nTot"] = "nParticles";
	  tmvaNames_["nCh"] = "nCharged";
	  tmvaNames_["p4.fCoordinates.fPt"] ="jetPt";
	  tmvaNames_["p4.fCoordinates.fEta"] ="jetEta";
	  tmvaNames_["axisMajor"] = "majW";
	  tmvaNames_["axisMinor"] = "minW";
	  tmvaNames_["fRing0"] = "frac01";
	  tmvaNames_["fRing1"] = "frac02";
	  tmvaNames_["fRing2"] = "frac03";
	  tmvaNames_["fRing3"] = "frac04";
	  tmvaNames_["min(pull,0.1)"] = "dRMean";
	  tmvaNames_["dRMatch"] = "chFrac02";
	}
	

	if( version_ == PHILv0 ) {
		tmvaVariables_.clear();
		tmvaVariables_.push_back( "jspt_1"  );
		tmvaVariables_.push_back( "jseta_1" );
		tmvaVariables_.push_back( "jsphi_1" );
		tmvaVariables_.push_back( "jd0_1"   );
		tmvaVariables_.push_back( "jdZ_1"   );
		tmvaVariables_.push_back( "jm_1"    );
		tmvaVariables_.push_back( "npart_1" );
		tmvaVariables_.push_back( "lpt_1"   );
		tmvaVariables_.push_back( "leta_1"  );
		tmvaVariables_.push_back( "lphi_1"  );
		tmvaVariables_.push_back( "spt_1"   );
		tmvaVariables_.push_back( "seta_1"  );
		tmvaVariables_.push_back( "sphi_1"  );
		tmvaVariables_.push_back( "lnept_1" );
		tmvaVariables_.push_back( "lneeta_1");
		tmvaVariables_.push_back( "lnephi_1");
		tmvaVariables_.push_back( "lempt_1" );
		tmvaVariables_.push_back( "lemeta_1");
		tmvaVariables_.push_back( "lemphi_1");
		tmvaVariables_.push_back( "lchpt_1" );
		// tmvaVariables_.push_back( "lcheta_1"); FIXME missing in weights file
		tmvaVariables_.push_back( "lchphi_1");
		tmvaVariables_.push_back( "lLfr_1"  );
		tmvaVariables_.push_back( "drlc_1"  );
		tmvaVariables_.push_back( "drls_1"  );
		tmvaVariables_.push_back( "drm_1"   );
		tmvaVariables_.push_back( "drmne_1" );
		tmvaVariables_.push_back( "drem_1"  );
		tmvaVariables_.push_back( "drch_1"  );
		
		tmvaNames_["jspt_1"] = "jetPt"; 
		tmvaNames_["jseta_1"] = "jetEta";
		tmvaNames_["jsphi_1"] = "jetPhi";
		tmvaNames_["jm_1"] = "jetM";
		tmvaNames_["jd0_1"] = "d0";
		tmvaNames_["jdZ_1"] = "dZ";
		tmvaNames_["npart_1"] = "nParticles";

		tmvaNames_["lpt_1"] = "leadPt";
		tmvaNames_["leta_1"] = "leadEta";
		tmvaNames_["lphi_1"] = "leadPhi";
		tmvaNames_["spt_1"] = "secondPt";
		tmvaNames_["seta_1"] = "secondEta";
		tmvaNames_["sphi_1"] = "secondPhi";
		tmvaNames_["lnept_1"] = "leadNeutPt";
		tmvaNames_["lneeta_1"] = "leadNeutEta";  
		tmvaNames_["lnephi_1"] = "leadNeutPhi";  
		tmvaNames_["lempt_1"] = "leadEmPt";
		tmvaNames_["lemeta_1"] = "leadEmEta";
		tmvaNames_["lemphi_1"] = "leadEmPhi";
		tmvaNames_["lchpt_1"] = "leadChPt";
		tmvaNames_["lcheta_1"] = "leadChEta";
		tmvaNames_["lchphi_1"] = "leadChPhi";
		tmvaNames_["lLfr_1"] = "leadFrac";

		tmvaNames_["drlc_1"] = "dRLeadCent";
		tmvaNames_["drls_1"] = "dRLead2nd";
		tmvaNames_["drm_1"] = "dRMean";
		tmvaNames_["drmne_1"] = "dRMeanNeut";
		tmvaNames_["drem_1"] = "dRMeanEm";
		tmvaNames_["drch_1"] = "dRMeanCh";

	} else if( ! cutBased_ ){
		assert( tmvaMethod_.empty() || ((! tmvaVariables_.empty() || ( !tmvaVariables_jteta_0_3_.empty() && !tmvaVariables_jteta_3_5_.empty() )) && version_ == USER) );
	}
	if(( ! cutBased_ ) && (runMvas_)) { bookReader();}
}

// ------------------------------------------------------------------------------------------
PileupJetIdAlgo::~PileupJetIdAlgo() 
{
}

// ------------------------------------------------------------------------------------------
void assign(const std::vector<float> & vec, float & a, float & b, float & c, float & d )
{
	size_t sz = vec.size();
	a = ( sz > 0 ? vec[0] : 0. );
	b = ( sz > 1 ? vec[1] : 0. );
	c = ( sz > 2 ? vec[2] : 0. );
	d = ( sz > 3 ? vec[3] : 0. );
}
// ------------------------------------------------------------------------------------------
void setPtEtaPhi(const reco::Candidate & p, float & pt, float & eta, float &phi )
{
	pt  = p.pt();
	eta = p.eta();
	phi = p.phi();
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::bookReader()
{
	if(etaBinnedWeights_){
		reader_jteta_0_2_ = std::unique_ptr<TMVA::Reader>(new TMVA::Reader("!Color:Silent"));
		reader_jteta_2_2p5_ = std::unique_ptr<TMVA::Reader>(new TMVA::Reader("!Color:Silent"));
		reader_jteta_2p5_3_ = std::unique_ptr<TMVA::Reader>(new TMVA::Reader("!Color:Silent"));
		reader_jteta_3_5_ = std::unique_ptr<TMVA::Reader>(new TMVA::Reader("!Color:Silent"));
	} else {
		reader_ = std::unique_ptr<TMVA::Reader>(new TMVA::Reader("!Color:Silent"));
	}
	if(etaBinnedWeights_){
	  for(std::vector<std::string>::iterator it=tmvaVariables_jteta_0_3_.begin(); it!=tmvaVariables_jteta_0_3_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		reader_jteta_0_2_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
		reader_jteta_2_2p5_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
		reader_jteta_2p5_3_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
	  }
	  for(std::vector<std::string>::iterator it=tmvaVariables_jteta_3_5_.begin(); it!=tmvaVariables_jteta_3_5_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		reader_jteta_3_5_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
	  }
	} else {
	  for(std::vector<std::string>::iterator it=tmvaVariables_.begin(); it!=tmvaVariables_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		reader_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
	  }
	}
	for(std::vector<std::string>::iterator it=tmvaSpectators_.begin(); it!=tmvaSpectators_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		if(etaBinnedWeights_){
			reader_jteta_0_2_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
			reader_jteta_2_2p5_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
			reader_jteta_2p5_3_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
			reader_jteta_3_5_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
		} else {
			reader_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
		}
	}
	if(etaBinnedWeights_){
		reco::details::loadTMVAWeights(reader_jteta_0_2_.get(),  tmvaMethod_.c_str(), tmvaWeights_jteta_0_2_.c_str() ); 
		reco::details::loadTMVAWeights(reader_jteta_2_2p5_.get(),  tmvaMethod_.c_str(), tmvaWeights_jteta_2_2p5_.c_str() ); 
		reco::details::loadTMVAWeights(reader_jteta_2p5_3_.get(),  tmvaMethod_.c_str(), tmvaWeights_jteta_2p5_3_.c_str() ); 
		reco::details::loadTMVAWeights(reader_jteta_3_5_.get(),  tmvaMethod_.c_str(), tmvaWeights_jteta_3_5_.c_str() ); 
	} else {
		reco::details::loadTMVAWeights(reader_.get(),  tmvaMethod_.c_str(), tmvaWeights_.c_str() ); 
	}
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::set(const PileupJetIdentifier & id)
{
	internalId_ = id;
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::runMva()
{
  	if( cutBased_ ) {
		internalId_.idFlag_ = computeCutIDflag(internalId_.betaStarClassic_,internalId_.dR2Mean_,internalId_.nvtx_,internalId_.jetPt_,internalId_.jetEta_);
	} else {
		if(std::abs(internalId_.jetEta_) >= 5.0) {
			internalId_.mva_ = -2.;
		} else {
			if(etaBinnedWeights_){
			  if(std::abs(internalId_.jetEta_)<=2.) internalId_.mva_ = reader_jteta_0_2_->EvaluateMVA( tmvaMethod_.c_str() );
			  else if(std::abs(internalId_.jetEta_)<=2.5) internalId_.mva_ = reader_jteta_2_2p5_->EvaluateMVA( tmvaMethod_.c_str() );
			  else if(std::abs(internalId_.jetEta_)<=3.) internalId_.mva_ = reader_jteta_2p5_3_->EvaluateMVA( tmvaMethod_.c_str() );
			  else internalId_.mva_ = reader_jteta_3_5_->EvaluateMVA( tmvaMethod_.c_str() );
			} else {
			  internalId_.mva_ = reader_->EvaluateMVA( tmvaMethod_.c_str() );
			}
		}
		internalId_.idFlag_ = computeIDflag(internalId_.mva_,internalId_.jetPt_,internalId_.jetEta_);
	}
}

// ------------------------------------------------------------------------------------------
std::pair<int,int> PileupJetIdAlgo::getJetIdKey(float jetPt, float jetEta)
{
  int ptId = 0;
  if(jetPt >= 10 && jetPt < 20) ptId = 1;                                                                                 
  if(jetPt >= 20 && jetPt < 30) ptId = 2;                                                                                 
  if(jetPt >= 30              ) ptId = 3;                                                                                          
  
  int etaId = 0;
  if(std::abs(jetEta) >= 2.5  && std::abs(jetEta) < 2.75) etaId = 1;                                                              
  if(std::abs(jetEta) >= 2.75 && std::abs(jetEta) < 3.0 ) etaId = 2;                                                              
  if(std::abs(jetEta) >= 3.0  && std::abs(jetEta) < 5.0 ) etaId = 3;                                 

  return std::pair<int,int>(ptId,etaId);
}
// ------------------------------------------------------------------------------------------
int PileupJetIdAlgo::computeCutIDflag(float betaStarClassic,float dR2Mean,float nvtx, float jetPt, float jetEta)
{
  std::pair<int,int> jetIdKey = getJetIdKey(jetPt,jetEta);
  float betaStarModified = betaStarClassic/log(nvtx-0.64);
  int idFlag(0);
  if(betaStarModified < betaStarCut_[PileupJetIdentifier::kTight ][jetIdKey.first][jetIdKey.second] && 
     dR2Mean          < rmsCut_     [PileupJetIdentifier::kTight ][jetIdKey.first][jetIdKey.second] 
     ) idFlag += 1 <<  PileupJetIdentifier::kTight;

  if(betaStarModified < betaStarCut_[PileupJetIdentifier::kMedium ][jetIdKey.first][jetIdKey.second] && 
     dR2Mean          < rmsCut_     [PileupJetIdentifier::kMedium ][jetIdKey.first][jetIdKey.second] 
     ) idFlag += 1 <<  PileupJetIdentifier::kMedium;
  
  if(betaStarModified < betaStarCut_[PileupJetIdentifier::kLoose  ][jetIdKey.first][jetIdKey.second] && 
     dR2Mean          < rmsCut_     [PileupJetIdentifier::kLoose  ][jetIdKey.first][jetIdKey.second] 
     ) idFlag += 1 <<  PileupJetIdentifier::kLoose;
  return idFlag;
}
// ------------------------------------------------------------------------------------------
int PileupJetIdAlgo::computeIDflag(float mva, float jetPt, float jetEta)
{
  std::pair<int,int> jetIdKey = getJetIdKey(jetPt,jetEta);
  return computeIDflag(mva,jetIdKey.first,jetIdKey.second);
}

// ------------------------------------------------------------------------------------------
int PileupJetIdAlgo::computeIDflag(float mva,int ptId,int etaId)
{
  int idFlag(0);
  if(mva > mvacut_[PileupJetIdentifier::kTight ][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kTight;
  if(mva > mvacut_[PileupJetIdentifier::kMedium][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kMedium;
  if(mva > mvacut_[PileupJetIdentifier::kLoose ][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kLoose;
  return idFlag;
}


// ------------------------------------------------------------------------------------------
PileupJetIdentifier PileupJetIdAlgo::computeMva()
{
	runMva();
	return PileupJetIdentifier(internalId_);
}

// ------------------------------------------------------------------------------------------
PileupJetIdentifier PileupJetIdAlgo::computeIdVariables(const reco::Jet * jet, float jec, const reco::Vertex * vtx,
							const reco::VertexCollection & allvtx, double rho) 
{

	static std::atomic<int> printWarning{10};
	
	// initialize all variables to 0
	resetVariables();
	
	// loop over constituents, accumulate sums and find leading candidates
	const pat::Jet * patjet = dynamic_cast<const pat::Jet *>(jet);
	const reco::PFJet * pfjet = dynamic_cast<const reco::PFJet *>(jet);
	assert( patjet != nullptr || pfjet != nullptr );
	if( patjet != nullptr && jec == 0. ) { // if this is a pat jet and no jec has been passed take the jec from the object
	  jec = patjet->pt()/patjet->correctedJet(0).pt();
	}
	if( jec <= 0. ) {
	  jec = 1.;
	}
	
	const reco::Candidate* lLead = nullptr, *lSecond = nullptr, *lLeadNeut = nullptr, *lLeadEm = nullptr, *lLeadCh = nullptr, *lTrail = nullptr;
	std::vector<float> frac, fracCh, fracEm, fracNeut;
	float cones[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
	size_t ncones = sizeof(cones)/sizeof(float);
	float * coneFracs[]     = { &internalId_.frac01_, &internalId_.frac02_, &internalId_.frac03_, &internalId_.frac04_, 
				    &internalId_.frac05_,  &internalId_.frac06_,  &internalId_.frac07_ };
	float * coneEmFracs[]   = { &internalId_.emFrac01_, &internalId_.emFrac02_, &internalId_.emFrac03_, &internalId_.emFrac04_, 
				    &internalId_.emFrac05_, &internalId_.emFrac06_, &internalId_.emFrac07_ }; 
	float * coneNeutFracs[] = { &internalId_.neutFrac01_, &internalId_.neutFrac02_, &internalId_.neutFrac03_, &internalId_.neutFrac04_, 
				    &internalId_.neutFrac05_, &internalId_.neutFrac06_, &internalId_.neutFrac07_ }; 
	float * coneChFracs[]   = { &internalId_.chFrac01_, &internalId_.chFrac02_, &internalId_.chFrac03_, &internalId_.chFrac04_, 
				    &internalId_.chFrac05_, &internalId_.chFrac06_, &internalId_.chFrac07_ }; 
	TMatrixDSym covMatrix(2); covMatrix = 0.;
	float jetPt = jet->pt() / jec; // use uncorrected pt for shape variables
	float sumPt = 0., sumPt2 = 0., sumTkPt = 0.,sumPtCh=0,sumPtNe = 0;
	setPtEtaPhi(*jet,internalId_.jetPt_,internalId_.jetEta_,internalId_.jetPhi_); // use corrected pt for jet kinematics
	internalId_.jetM_ = jet->mass(); 
	internalId_.nvtx_ = allvtx.size();
	internalId_.rho_ = rho;

	float dRmin(1000);
	
	for ( unsigned i = 0; i < jet->numberOfSourceCandidatePtrs(); ++i ) {
	  reco::CandidatePtr pfJetConstituent = jet->sourceCandidatePtr(i);
  
	  const reco::Candidate* icand = pfJetConstituent.get();
	  const pat::PackedCandidate* lPack = dynamic_cast<const pat::PackedCandidate *>( icand );
	  const reco::PFCandidate *lPF=dynamic_cast<const reco::PFCandidate*>( icand );
	  bool isPacked = true;
	  if (lPack == nullptr){
	    isPacked = false;
	  }

	    float candPt = icand->pt();
	    float candPtFrac = candPt/jetPt;
	    float candDr   = reco::deltaR(*icand,*jet);
	    float candDeta = std::abs( icand->eta() - jet->eta() );
	    float candDphi = reco::deltaPhi(*icand,*jet);
	    float candPtDr = candPt * candDr;
	    size_t icone = std::lower_bound(&cones[0],&cones[ncones],candDr) - &cones[0];

	    if(candDr < dRmin) dRmin = candDr;

		// // all particles
		if( lLead == nullptr || candPt > lLead->pt() ) {
			lSecond = lLead;
			lLead = icand; 
		} else if( (lSecond == nullptr || candPt > lSecond->pt()) && (candPt < lLead->pt()) ) {
			lSecond = icand;
		}

		// // average shapes
		internalId_.dRMean_     += candPtDr;
		internalId_.dR2Mean_    += candPtDr*candPtDr;
		covMatrix(0,0) += candPt*candPt*candDeta*candDeta;
		covMatrix(0,1) += candPt*candPt*candDeta*candDphi;
		covMatrix(1,1) += candPt*candPt*candDphi*candDphi;
		internalId_.ptD_ += candPt*candPt;
		sumPt += candPt;
		sumPt2 += candPt*candPt;
		
		// single most energetic candiates and jet shape profiles
		frac.push_back(candPtFrac);
		
		if( icone < ncones ) { *coneFracs[icone] += candPt; }
	
		// neutrals
		if( abs(icand->pdgId()) == 130) {
			if (lLeadNeut == nullptr || candPt > lLeadNeut->pt()) { lLeadNeut = icand; }
			internalId_.dRMeanNeut_ += candPtDr;
			fracNeut.push_back(candPtFrac);
			if( icone < ncones ) { *coneNeutFracs[icone] += candPt; }
			internalId_.ptDNe_    += candPt*candPt;
			sumPtNe               += candPt;
		}
		// EM candidated
		if( icand->pdgId() == 22 ) {
			if(lLeadEm == nullptr || candPt > lLeadEm->pt())  { lLeadEm = icand; }
			internalId_.dRMeanEm_ += candPtDr;
			fracEm.push_back(candPtFrac);
			if( icone < ncones ) { *coneEmFracs[icone] += candPt; }
			internalId_.ptDNe_    += candPt*candPt;
			sumPtNe               += candPt;
		}
		// Charged  particles
		if(  icand->charge() != 0 ) {
		        if (lLeadCh == nullptr || candPt > lLeadCh->pt()) { 
			  lLeadCh = icand;
			
			  const reco::Track* pfTrk = icand->bestTrack();
			  if (lPF && std::abs(icand->pdgId()) == 13 && pfTrk == nullptr){
			    reco::MuonRef lmuRef = lPF->muonRef();
			    if (lmuRef.isNonnull()){
			      const reco::Muon& lmu = *lmuRef.get();
			      pfTrk = lmu.bestTrack();
			      edm::LogWarning("BadMuon")<<"Found a PFCandidate muon without a trackRef: falling back to Muon::bestTrack ";
			    }
			  }
			  if(pfTrk==nullptr) { //protection against empty pointers for the miniAOD case
			    //To handle the electron case
			    if(lPF!=nullptr) {
			      pfTrk=(lPF->trackRef().get()==nullptr)?lPF->gsfTrackRef().get():lPF->trackRef().get();
			    }
			    const reco::Track& impactTrack = (lPack==nullptr)?(*pfTrk):(lPack->pseudoTrack());
			    internalId_.d0_ = std::abs(impactTrack.dxy(vtx->position()));
			    internalId_.dZ_ = std::abs(impactTrack.dz(vtx->position()));
			  }
			  else {
			    internalId_.d0_ = std::abs(pfTrk->dxy(vtx->position()));
			    internalId_.dZ_ = std::abs(pfTrk->dz(vtx->position()));
			  }
			}
			internalId_.dRMeanCh_  += candPtDr;
			internalId_.ptDCh_     += candPt*candPt;
			fracCh.push_back(candPtFrac);
			if( icone < ncones ) { *coneChFracs[icone] += candPt; }
			sumPtCh                += candPt;
		}
		// // beta and betastar		
		if( icand->charge() != 0 ) {
		      if (!isPacked){
			   if(lPF->trackRef().isNonnull() ) { 
	      			float tkpt = candPt;
				sumTkPt += tkpt;
				// 'classic' beta definition based on track-vertex association
				bool inVtx0 = vtx->trackWeight ( lPF->trackRef()) > 0 ;
					
				bool inAnyOther = false;
				// alternative beta definition based on track-vertex distance of closest approach
				double dZ0 = std::abs(lPF->trackRef()->dz(vtx->position()));
				double dZ = dZ0;
				for(reco::VertexCollection::const_iterator  vi=allvtx.begin(); vi!=allvtx.end(); ++vi ) {
				    const reco::Vertex & iv = *vi;
				    if( iv.isFake() || iv.ndof() < 4 ) { continue; }
				        // the primary vertex may have been copied by the user: check identity by position
				        bool isVtx0  = (iv.position() - vtx->position()).r() < 0.02;
					// 'classic' beta definition: check if the track is associated with
					// any vertex other than the primary one
					if( ! isVtx0 && ! inAnyOther ) {
					  inAnyOther =  vtx->trackWeight ( lPF->trackRef()) <= 0 ;
					}
					// alternative beta: find closest vertex to the track
					dZ = std::min(dZ,std::abs(lPF->trackRef()->dz(iv.position())));
				}
				// classic beta/betaStar
				if( inVtx0 && ! inAnyOther ) {
				    internalId_.betaClassic_ += tkpt;
				} else if( ! inVtx0 && inAnyOther ) {
				    internalId_.betaStarClassic_ += tkpt;
				}
				// alternative beta/betaStar
				if( dZ0 < 0.2 ) {
				    internalId_.beta_ += tkpt;
				} else if( dZ < 0.2 ) {
				    internalId_.betaStar_ += tkpt;
				}
			   } 
			}
			else{
				// setting classic and alternative to be the same for now
			        float tkpt = candPt;
			        sumTkPt += tkpt;
				bool inVtx0 = false; 
				bool inVtxOther = false; 
				if (lPack->fromPV() == pat::PackedCandidate::PVUsedInFit) inVtx0 = true;
				if (lPack->fromPV() == 0) inVtxOther = true;
				if (inVtx0){
					internalId_.betaClassic_ += tkpt;
					internalId_.beta_ += tkpt;
				}
				if (inVtxOther){
					internalId_.betaStarClassic_ += tkpt;
					internalId_.betaStar_ += tkpt;
				}
			}
		}
		// trailing candidate
		if( lTrail == nullptr || candPt < lTrail->pt() ) {
			lTrail = icand; 
		}
	
	}

	// // Finalize all variables
	assert( !(lLead == nullptr) );

	if ( lSecond == nullptr )   { lSecond   = lTrail; }
	if ( lLeadNeut == nullptr ) { lLeadNeut = lTrail; }
	if ( lLeadEm == nullptr )   { lLeadEm   = lTrail; }
	if ( lLeadCh == nullptr )   { lLeadCh   = lTrail; }
	
	internalId_.nCharged_    = pfjet->chargedMultiplicity();
	internalId_.nNeutrals_   = pfjet->neutralMultiplicity();
	internalId_.chgEMfrac_   = pfjet->chargedEmEnergy()    /jet->energy();
	internalId_.neuEMfrac_   = pfjet->neutralEmEnergy()    /jet->energy();
	internalId_.chgHadrfrac_ = pfjet->chargedHadronEnergy()/jet->energy();
	internalId_.neuHadrfrac_ = pfjet->neutralHadronEnergy()/jet->energy();
	internalId_.nParticles_ = jet->numberOfDaughters(); 

	setPtEtaPhi(*lLead,internalId_.leadPt_,internalId_.leadEta_,internalId_.leadPhi_);                 
	setPtEtaPhi(*lSecond,internalId_.secondPt_,internalId_.secondEta_,internalId_.secondPhi_);	      
	setPtEtaPhi(*lLeadNeut,internalId_.leadNeutPt_,internalId_.leadNeutEta_,internalId_.leadNeutPhi_); 
	setPtEtaPhi(*lLeadEm,internalId_.leadEmPt_,internalId_.leadEmEta_,internalId_.leadEmPhi_);	      
	setPtEtaPhi(*lLeadCh,internalId_.leadChPt_,internalId_.leadChEta_,internalId_.leadChPhi_);         

	std::sort(frac.begin(),frac.end(),std::greater<float>());
	std::sort(fracCh.begin(),fracCh.end(),std::greater<float>());
	std::sort(fracEm.begin(),fracEm.end(),std::greater<float>());
	std::sort(fracNeut.begin(),fracNeut.end(),std::greater<float>());
	assign(frac,    internalId_.leadFrac_,    internalId_.secondFrac_,    internalId_.thirdFrac_,    internalId_.fourthFrac_);
	assign(fracCh,  internalId_.leadChFrac_,  internalId_.secondChFrac_,  internalId_.thirdChFrac_,  internalId_.fourthChFrac_);
	assign(fracEm,  internalId_.leadEmFrac_,  internalId_.secondEmFrac_,  internalId_.thirdEmFrac_,  internalId_.fourthEmFrac_);
	assign(fracNeut,internalId_.leadNeutFrac_,internalId_.secondNeutFrac_,internalId_.thirdNeutFrac_,internalId_.fourthNeutFrac_);
	
	covMatrix(0,0) /= sumPt2;
	covMatrix(0,1) /= sumPt2;
	covMatrix(1,1) /= sumPt2;
	covMatrix(1,0)  = covMatrix(0,1);
	internalId_.etaW_ = sqrt(covMatrix(0,0));
	internalId_.phiW_ = sqrt(covMatrix(1,1));
	internalId_.jetW_ = 0.5*(internalId_.etaW_+internalId_.phiW_);
	TVectorD eigVals(2); eigVals = TMatrixDSymEigen(covMatrix).GetEigenValues();
	internalId_.majW_ = sqrt(std::abs(eigVals(0)));
	internalId_.minW_ = sqrt(std::abs(eigVals(1)));
	if( internalId_.majW_ < internalId_.minW_ ) { std::swap(internalId_.majW_,internalId_.minW_); }
	
	internalId_.dRLeadCent_ = reco::deltaR(*jet,*lLead);
	if( lSecond == nullptr ) { internalId_.dRLead2nd_  = reco::deltaR(*jet,*lSecond); }
	internalId_.dRMean_     /= jetPt;
	internalId_.dRMeanNeut_ /= jetPt;
	internalId_.dRMeanEm_   /= jetPt;
	internalId_.dRMeanCh_   /= jetPt;
	internalId_.dR2Mean_    /= sumPt2;

	for(size_t ic=0; ic<ncones; ++ic){
		*coneFracs[ic]     /= jetPt;
		*coneEmFracs[ic]   /= jetPt;
		*coneNeutFracs[ic] /= jetPt;
		*coneChFracs[ic]   /= jetPt;
	}
	//http://jets.physics.harvard.edu/qvg/
	double ptMean = sumPt/internalId_.nParticles_;
	double ptRMS  = 0;
	for(unsigned int i0 = 0; i0 < frac.size(); i0++) {ptRMS+=(frac[i0]-ptMean)*(frac[i0]-ptMean);}
	ptRMS/=internalId_.nParticles_;
	ptRMS=sqrt(ptRMS);
	
	internalId_.ptMean_  = ptMean;
	internalId_.ptRMS_   = ptRMS/jetPt;
	internalId_.pt2A_    = sqrt( internalId_.ptD_     /internalId_.nParticles_)/jetPt;
	internalId_.ptD_     = sqrt( internalId_.ptD_)    / sumPt;
	internalId_.ptDCh_   = sqrt( internalId_.ptDCh_)  / sumPtCh;
	internalId_.ptDNe_   = sqrt( internalId_.ptDNe_)  / sumPtNe;
	internalId_.sumPt_   = sumPt;
	internalId_.sumChPt_ = sumPtCh;
	internalId_.sumNePt_ = sumPtNe;

	internalId_.jetR_    = lLead->pt()/sumPt;
	internalId_.jetRchg_ = lLeadEm->pt()/sumPt;
	internalId_.dRMatch_ = dRmin;

	if( sumTkPt != 0. ) {
		internalId_.beta_     /= sumTkPt;
		internalId_.betaStar_ /= sumTkPt;
		internalId_.betaClassic_ /= sumTkPt;
		internalId_.betaStarClassic_ /= sumTkPt;
	} else {
		assert( internalId_.beta_ == 0. && internalId_.betaStar_ == 0.&& internalId_.betaClassic_ == 0. && internalId_.betaStarClassic_ == 0. );
	}

	if( runMvas_ ) {
		runMva();
	}
	
	return PileupJetIdentifier(internalId_);
}



// ------------------------------------------------------------------------------------------
std::string PileupJetIdAlgo::dumpVariables() const
{
	std::stringstream out;
	for(variables_list_t::const_iterator it=variables_.begin(); 
	    it!=variables_.end(); ++it ) {
		out << std::setw(15) << it->first << std::setw(3) << "=" 
		    << std::setw(5) << *it->second.first 
		    << " (" << std::setw(5) << it->second.second << ")" << std::endl;
	}
	return out.str();
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::resetVariables()
{
	internalId_.idFlag_    = 0;
	for(variables_list_t::iterator it=variables_.begin(); 
	    it!=variables_.end(); ++it ) {
		*it->second.first = it->second.second;
	}
}

// ------------------------------------------------------------------------------------------
#define INIT_VARIABLE(NAME,TMVANAME,VAL)	\
	internalId_.NAME ## _ = VAL; \
	variables_[ # NAME   ] = std::make_pair(& internalId_.NAME ## _, VAL);

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::initVariables()
{
	internalId_.idFlag_    = 0;
  	INIT_VARIABLE(mva        , "", -100.);
	//INIT_VARIABLE(jetPt      , "jspt_1", 0.);
	//INIT_VARIABLE(jetEta     , "jseta_1", large_val);
	INIT_VARIABLE(jetPt      , "p4.fCoordinates.fPt", 0.);
	INIT_VARIABLE(jetEta     , "p4.fCoordinates.fEta", large_val);
	INIT_VARIABLE(jetPhi     , "jsphi_1", large_val);
	INIT_VARIABLE(jetM       , "jm_1", 0.);

	//INIT_VARIABLE(nCharged   , "", 0.);
	INIT_VARIABLE(nCharged , "nCh"  , 0.);  
	INIT_VARIABLE(nNeutrals  , "", 0.);
	
	INIT_VARIABLE(chgEMfrac  , "", 0.);
	INIT_VARIABLE(neuEMfrac  , "", 0.);
	INIT_VARIABLE(chgHadrfrac, "", 0.);
	INIT_VARIABLE(neuHadrfrac, "", 0.);
	
	INIT_VARIABLE(d0         , "jd0_1"    , -1000.);   
	INIT_VARIABLE(dZ         , "jdZ_1"    , -1000.);  
	//INIT_VARIABLE(nParticles , "npart_1"  , 0.);  
	INIT_VARIABLE(nParticles , "nTot"  , 0.);  
	
	INIT_VARIABLE(leadPt     , "lpt_1"    , 0.);  
	INIT_VARIABLE(leadEta    , "leta_1"   , large_val);  
	INIT_VARIABLE(leadPhi    , "lphi_1"   , large_val);  
	INIT_VARIABLE(secondPt   , "spt_1"    , 0.);  
	INIT_VARIABLE(secondEta  , "seta_1"   , large_val);  
	INIT_VARIABLE(secondPhi  , "sphi_1"   , large_val);  
	INIT_VARIABLE(leadNeutPt , "lnept_1"    , 0.);  
	INIT_VARIABLE(leadNeutEta, "lneeta_1"   , large_val);  
	INIT_VARIABLE(leadNeutPhi, "lnephi_1"   , large_val);  
	INIT_VARIABLE(leadEmPt   , "lempt_1"  , 0.);  
	INIT_VARIABLE(leadEmEta  , "lemeta_1" , large_val);  
	INIT_VARIABLE(leadEmPhi  , "lemphi_1" , large_val);  
	INIT_VARIABLE(leadChPt   , "lchpt_1"  , 0.);  
	INIT_VARIABLE(leadChEta  , "lcheta_1" , large_val);  
	INIT_VARIABLE(leadChPhi  , "lchphi_1" , large_val);  
	INIT_VARIABLE(leadFrac   , "lLfr_1"   , 0.);  
	
	INIT_VARIABLE(dRLeadCent , "drlc_1"   , 0.);  
	INIT_VARIABLE(dRLead2nd  , "drls_1"   , 0.);  
	//INIT_VARIABLE(dRMean     , "drm_1"    , 0.);  
	INIT_VARIABLE(dRMean     , "min(pull,0.1)"    , 0.);  
	INIT_VARIABLE(dRMeanNeut , "drmne_1"  , 0.);  
	INIT_VARIABLE(dRMeanEm   , "drem_1"   , 0.);  
	INIT_VARIABLE(dRMeanCh   , "drch_1"   , 0.);  
	//INIT_VARIABLE(dR2Mean    , ""         , 0.);  
	INIT_VARIABLE(dR2Mean     , "DRweighted"    , 0.);  
	
	INIT_VARIABLE(ptD        , "", 0.);
	INIT_VARIABLE(ptMean     , "", 0.);
	INIT_VARIABLE(ptRMS      , "", 0.);
	INIT_VARIABLE(pt2A       , "", 0.);
	INIT_VARIABLE(ptDCh      , "", 0.);
	INIT_VARIABLE(ptDNe      , "", 0.);
	INIT_VARIABLE(sumPt      , "", 0.);
	INIT_VARIABLE(sumChPt    , "", 0.);
	INIT_VARIABLE(sumNePt    , "", 0.);

	INIT_VARIABLE(secondFrac  ,"" ,0.);  
	INIT_VARIABLE(thirdFrac   ,"" ,0.);  
	INIT_VARIABLE(fourthFrac  ,"" ,0.);  

	INIT_VARIABLE(leadChFrac    ,"" ,0.);  
	INIT_VARIABLE(secondChFrac  ,"" ,0.);  
	INIT_VARIABLE(thirdChFrac   ,"" ,0.);  
	INIT_VARIABLE(fourthChFrac  ,"" ,0.);  

	INIT_VARIABLE(leadNeutFrac    ,"" ,0.);  
	INIT_VARIABLE(secondNeutFrac  ,"" ,0.);  
	INIT_VARIABLE(thirdNeutFrac   ,"" ,0.);  
	INIT_VARIABLE(fourthNeutFrac  ,"" ,0.);  

	INIT_VARIABLE(leadEmFrac    ,"" ,0.);  
	INIT_VARIABLE(secondEmFrac  ,"" ,0.);  
	INIT_VARIABLE(thirdEmFrac   ,"" ,0.);  
	INIT_VARIABLE(fourthEmFrac  ,"" ,0.);  

	INIT_VARIABLE(jetW  ,"" ,1.);  
	INIT_VARIABLE(etaW  ,"" ,1.);  
	INIT_VARIABLE(phiW  ,"" ,1.);  

	//INIT_VARIABLE(majW  ,"" ,1.);  
	//INIT_VARIABLE(minW  ,"" ,1.);  
	INIT_VARIABLE(majW  ,"axisMajor" ,1.);  
	INIT_VARIABLE(minW  ,"axisMinor" ,1.);  

	//INIT_VARIABLE(frac01    ,"" ,0.);  
	//INIT_VARIABLE(frac02    ,"" ,0.);  
	//INIT_VARIABLE(frac03    ,"" ,0.);  
	//INIT_VARIABLE(frac04    ,"" ,0.);
	INIT_VARIABLE(frac05   ,"" ,0.);  
	INIT_VARIABLE(frac06   ,"" ,0.);  
	INIT_VARIABLE(frac07   ,"" ,0.);
	INIT_VARIABLE(frac01    ,"fRing0" ,0.);  
	INIT_VARIABLE(frac02    ,"fRing1" ,0.);
	INIT_VARIABLE(frac03    ,"fRing2" ,0.);  
	INIT_VARIABLE(frac04    ,"fRing3" ,0.); 
		
	INIT_VARIABLE(chFrac01    ,"" ,0.);  
	INIT_VARIABLE(chFrac02    ,"" ,0.);
	INIT_VARIABLE(chFrac03    ,"" ,0.);  
	INIT_VARIABLE(chFrac04    ,"" ,0.);  
	INIT_VARIABLE(chFrac05   ,"" ,0.);  
	INIT_VARIABLE(chFrac06   ,"" ,0.);  
	INIT_VARIABLE(chFrac07   ,"" ,0.);  

	INIT_VARIABLE(neutFrac01    ,"" ,0.);  
	INIT_VARIABLE(neutFrac02    ,"" ,0.);  
	INIT_VARIABLE(neutFrac03    ,"" ,0.);  
	INIT_VARIABLE(neutFrac04    ,"" ,0.);  
	INIT_VARIABLE(neutFrac05   ,"" ,0.);  
	INIT_VARIABLE(neutFrac06   ,"" ,0.);  
	INIT_VARIABLE(neutFrac07   ,"" ,0.);  

	INIT_VARIABLE(emFrac01    ,"" ,0.);  
	INIT_VARIABLE(emFrac02    ,"" ,0.);  
	INIT_VARIABLE(emFrac03    ,"" ,0.);  
	INIT_VARIABLE(emFrac04    ,"" ,0.);  
	INIT_VARIABLE(emFrac05   ,"" ,0.);  
	INIT_VARIABLE(emFrac06   ,"" ,0.);  
	INIT_VARIABLE(emFrac07   ,"" ,0.);  

	INIT_VARIABLE(beta   ,"" ,0.);  
	INIT_VARIABLE(betaStar   ,"" ,0.);  
	INIT_VARIABLE(betaClassic   ,"" ,0.);  
	INIT_VARIABLE(betaStarClassic   ,"" ,0.);  

	INIT_VARIABLE(nvtx   ,"" ,0.);  
	INIT_VARIABLE(rho   ,"" ,0.);  
	INIT_VARIABLE(nTrueInt   ,"" ,0.);

	INIT_VARIABLE(jetR       , "", 0.);
	INIT_VARIABLE(jetRchg    , "", 0.);
	INIT_VARIABLE(dRMatch    , "", 0.);
	
}

#undef INIT_VARIABLE
