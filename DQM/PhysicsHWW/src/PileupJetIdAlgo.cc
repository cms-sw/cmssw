#include "DataFormats/PatCandidates/interface/Jet.h" 
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"

#include "DQM/PhysicsHWW/interface/PileupJetIdAlgo.h"

// ------------------------------------------------------------------------------------------
const float large_val = std::numeric_limits<float>::max();

// ------------------------------------------------------------------------------------------
PileupJetIdAlgo::PileupJetIdAlgo(const edm::ParameterSet & ps) 
{
	impactParTkThreshod_ = 1.;/// ps.getParameter<double>("impactParTkThreshod");
	cutBased_ = false;
	//std::string label    = ps.getParameter<std::string>("label");
	cutBased_ =  ps.getParameter<bool>("cutBased");
	if(!cutBased_) 
	  {
	    tmvaWeights_         = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights")).fullPath(); 
	    tmvaMethod_          = ps.getParameter<std::string>("tmvaMethod");
	    tmvaVariables_       = ps.getParameter<std::vector<std::string> >("tmvaVariables");
	    tmvaSpectators_      = ps.getParameter<std::vector<std::string> >("tmvaSpectators");
	    version_             = ps.getParameter<int>("version");
	  }
	reader_              = 0;
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
				 const std::vector<std::string> & tmvaVariables
	) 
{
	impactParTkThreshod_ = impactParTkThreshod;
	tmvaWeights_         = tmvaWeights;
	tmvaMethod_          = tmvaMethod;
	tmvaVariables_       = tmvaVariables;
	version_             = version;
	
	reader_              = 0;
	
	setup();
}

// ------------------------------------------------------------------------------------------
void PileupJetIdAlgo::setup()
{
	initVariables();

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
		assert( tmvaMethod_.empty() || (! tmvaVariables_.empty() && version_ == USER) );
	}
}

// ------------------------------------------------------------------------------------------
PileupJetIdAlgo::~PileupJetIdAlgo() 
{
	if( reader_ ) {
		delete reader_;
	}
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
	reader_ = new TMVA::Reader("!Color:!Silent");
	assert( ! tmvaMethod_.empty() && !  tmvaWeights_.empty() );
	for(std::vector<std::string>::iterator it=tmvaVariables_.begin(); it!=tmvaVariables_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		reader_->AddVariable( *it, variables_[ tmvaNames_[*it] ].first );
	}
	for(std::vector<std::string>::iterator it=tmvaSpectators_.begin(); it!=tmvaSpectators_.end(); ++it) {
		if(  tmvaNames_[*it].empty() ) { 
			tmvaNames_[*it] = *it;
		}
		reader_->AddSpectator( *it, variables_[ tmvaNames_[*it] ].first );
	}
	reader_->BookMVA(tmvaMethod_.c_str(), tmvaWeights_.c_str());
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
		if( ! reader_ ) { bookReader(); /*std::cerr << "Reader booked" << std::endl;*/ }
		if(fabs(internalId_.jetEta_) <  5.0) internalId_.mva_ = reader_->EvaluateMVA( tmvaMethod_.c_str() );
		if(fabs(internalId_.jetEta_) >= 5.0) internalId_.mva_ = -2.;
		internalId_.idFlag_ = computeIDflag(internalId_.mva_,internalId_.jetPt_,internalId_.jetEta_);
	}
}

// ------------------------------------------------------------------------------------------
std::pair<int,int> PileupJetIdAlgo::getJetIdKey(float jetPt, float jetEta)
{
  int ptId = 0;                                                                                                                                    
  if(jetPt > 10 && jetPt < 20) ptId = 1;                                                                                 
  if(jetPt > 20 && jetPt < 30) ptId = 2;                                                                                 
  if(jetPt > 30              ) ptId = 3;                                                                                          
  
  int etaId = 0;                                                                                                                                   
  if(fabs(jetEta) > 2.5  && fabs(jetEta) < 2.75) etaId = 1;                                                              
  if(fabs(jetEta) > 2.75 && fabs(jetEta) < 3.0 ) etaId = 2;                                                              
  if(fabs(jetEta) > 3.0  && fabs(jetEta) < 5.0 ) etaId = 3;                                 

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
							const reco::VertexCollection & allvtx,
							bool calculateMva) 
{
	//static int printWarning = 10; 
	typedef std::vector <reco::PFCandidatePtr> constituents_type;
	typedef std::vector <reco::PFCandidatePtr>::iterator constituents_iterator;

	// initialize all variables to 0
	resetVariables();
	
	// loop over constituents, accumulate sums and find leading candidates
	const pat::Jet * patjet = dynamic_cast<const pat::Jet *>(jet);
	const reco::PFJet * pfjet = dynamic_cast<const reco::PFJet *>(jet);
	assert( patjet != 0 || pfjet != 0 );
	if( patjet != 0 && jec == 0. ) { // if this is a pat jet and no jec has been passed take the jec from the object
		jec = patjet->pt()/patjet->correctedJet(0).pt();
	}
	if( jec < 0. ) {
		jec = 1.;
	}
	constituents_type constituents = pfjet ? pfjet->getPFConstituents() : patjet->getPFConstituents();
	
	reco::PFCandidatePtr lLead, lSecond, lLeadNeut, lLeadEm, lLeadCh, lTrail;
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
	
	reco::TrackRef impactTrack;
	float jetPt = jet->pt() / jec; // use uncorrected pt for shape variables
	float sumPt = 0., sumPt2 = 0., sumTkPt = 0.,sumPtCh=0,sumPtNe = 0;
	setPtEtaPhi(*jet,internalId_.jetPt_,internalId_.jetEta_,internalId_.jetPhi_); // use corrected pt for jet kinematics
	internalId_.jetM_ = jet->mass(); 
	internalId_.nvtx_ = allvtx.size();
	
	for(constituents_iterator it=constituents.begin(); it!=constituents.end(); ++it) {
		reco::PFCandidatePtr & icand = *it;
		float candPt = icand->pt();
		float candPtFrac = candPt/jetPt;
		float candDr   = reco::deltaR(**it,*jet);
		float candDeta = fabs( (*it)->eta() - jet->eta() );
		float candDphi = reco::deltaPhi(**it,*jet);
		float candPtDr = candPt * candDr;
		size_t icone = std::lower_bound(&cones[0],&cones[ncones],candDr) - &cones[0];
		
		// all particles
		if( lLead.isNull() || candPt > lLead->pt() ) {
			lSecond = lLead;
			lLead = icand; 
		} else if( (lSecond.isNull() || candPt > lSecond->pt()) && (candPt < lLead->pt()) ) {
			lSecond = icand;
		}

		// average shapes
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
		if( icand->particleId() == reco::PFCandidate::h0 ) {
			if (lLeadNeut.isNull() || candPt > lLeadNeut->pt()) { lLeadNeut = icand; }
			internalId_.dRMeanNeut_ += candPtDr;
			fracNeut.push_back(candPtFrac);
			if( icone < ncones ) { *coneNeutFracs[icone] += candPt; }
			internalId_.ptDNe_    += candPt*candPt;
			sumPtNe               += candPt;
		}
		// EM candidated
		if( icand->particleId() == reco::PFCandidate::gamma ) {
			if(lLeadEm.isNull() || candPt > lLeadEm->pt())  { lLeadEm = icand; }
			internalId_.dRMeanEm_ += candPtDr;
			fracEm.push_back(candPtFrac);
			if( icone < ncones ) { *coneEmFracs[icone] += candPt; }
			internalId_.ptDNe_    += candPt*candPt;
			sumPtNe               += candPt;
		}
		// Charged  particles
		if(  icand->trackRef().isNonnull() && icand->trackRef().isAvailable() ) {
			if (lLeadCh.isNull() || candPt > lLeadCh->pt()) { lLeadCh = icand; }
			internalId_.dRMeanCh_  += candPtDr;
			internalId_.ptDCh_     += candPt*candPt;
			fracCh.push_back(candPtFrac);
			if( icone < ncones ) { *coneChFracs[icone] += candPt; }
			sumPtCh                += candPt;
		}
		// beta and betastar		
		if(  icand->trackRef().isNonnull() && icand->trackRef().isAvailable() ) {
			try { 
				float tkpt = icand->trackRef()->pt(); 
				sumTkPt += tkpt;
				// 'classic' beta definition based on track-vertex association
				bool inVtx0 = find( vtx->tracks_begin(), vtx->tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != vtx->tracks_end();
				bool inAnyOther = false;
				// alternative beta definition based on track-vertex distance of closest approach
				double dZ0 = fabs(icand->trackRef()->dz(vtx->position()));
				double dZ = dZ0;
				for(reco::VertexCollection::const_iterator  vi=allvtx.begin(); vi!=allvtx.end(); ++vi ) {
					const reco::Vertex & iv = *vi;
					if( iv.isFake() || iv.ndof() < 4 ) { continue; }
					// the primary vertex may have been copied by the user: check identity by position
					bool isVtx0  = (iv.position() - vtx->position()).r() < 0.02;
					// 'classic' beta definition: check if the track is associated with any vertex other than the primary one
					if( ! isVtx0 && ! inAnyOther ) {
						inAnyOther = find( iv.tracks_begin(), iv.tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != iv.tracks_end();
					}
					// alternative beta: find closest vertex to the track
					dZ = std::min(dZ,fabs(icand->trackRef()->dz(iv.position())));
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
			} catch (cms::Exception &e) {
				//if(printWarning-- > 0) { std::cerr << e << std::endl; }
			}
		}
		// trailing candidate
		if( lTrail.isNull() || candPt < lTrail->pt() ) {
			lTrail = icand; 
		}
	}
	
	// Finalize all variables
	assert( lLead.isNonnull() );

	if ( lSecond.isNull() )   { lSecond   = lTrail; }
	if ( lLeadNeut.isNull() ) { lLeadNeut = lTrail; }
	if ( lLeadEm.isNull() )   { lLeadEm   = lTrail; }
	if ( lLeadCh.isNull() )   { lLeadCh   = lTrail; }
	impactTrack = lLeadCh->trackRef();
	
	internalId_.nCharged_  = pfjet ? pfjet->chargedMultiplicity() : patjet->chargedMultiplicity();
	internalId_.nNeutrals_ = pfjet ? pfjet->neutralMultiplicity() : patjet->neutralMultiplicity();

	internalId_.chgEMfrac_   = ( pfjet ? pfjet->chargedEmEnergy()     : patjet->chargedEmEnergy()      ) /jet->energy();
	internalId_.neuEMfrac_   = ( pfjet ? pfjet->neutralEmEnergy()     : patjet->neutralEmEnergy()      ) /jet->energy();
	internalId_.chgHadrfrac_ = ( pfjet ? pfjet->chargedHadronEnergy() : patjet->chargedHadronEnergy()  ) /jet->energy();
	internalId_.neuHadrfrac_ = ( pfjet ? pfjet->neutralHadronEnergy() : patjet->neutralHadronEnergy()  ) /jet->energy();
	
	if( impactTrack.isNonnull() && impactTrack.isAvailable() ) {
		internalId_.d0_ = fabs(impactTrack->dxy(vtx->position()));
		internalId_.dZ_ = fabs(impactTrack->dz(vtx->position()));
	} else {
		//if(printWarning-- > 0) { std::cerr << "WARNING : did not find any valid track reference attached to the jet " << std::endl; }
	}
	internalId_.nParticles_ = constituents.size(); 

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
	internalId_.majW_ = sqrt(fabs(eigVals(0)));
	internalId_.minW_ = sqrt(fabs(eigVals(1)));
	if( internalId_.majW_ < internalId_.minW_ ) { std::swap(internalId_.majW_,internalId_.minW_); }
	
	internalId_.dRLeadCent_ = reco::deltaR(*jet,*lLead);
	if( lSecond.isNonnull() ) { internalId_.dRLead2nd_  = reco::deltaR(*jet,*lSecond); }
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

	if( sumTkPt != 0. ) {
		internalId_.beta_     /= sumTkPt;
		internalId_.betaStar_ /= sumTkPt;
		internalId_.betaClassic_ /= sumTkPt;
		internalId_.betaStarClassic_ /= sumTkPt;
	} else {
		assert( internalId_.beta_ == 0. && internalId_.betaStar_ == 0.&& internalId_.betaClassic_ == 0. && internalId_.betaStarClassic_ == 0. );
	}

	if( calculateMva ) {
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
	INIT_VARIABLE(jetPt      , "jspt_1", 0.);
	INIT_VARIABLE(jetEta     , "jseta_1", large_val);
	INIT_VARIABLE(jetPhi     , "jsphi_1", large_val);
	INIT_VARIABLE(jetM       , "jm_1", 0.);

	INIT_VARIABLE(nCharged   , "", 0.);
	INIT_VARIABLE(nNeutrals  , "", 0.);
	
	INIT_VARIABLE(chgEMfrac  , "", 0.);
	INIT_VARIABLE(neuEMfrac  , "", 0.);
	INIT_VARIABLE(chgHadrfrac, "", 0.);
	INIT_VARIABLE(neuHadrfrac, "", 0.);
	
	INIT_VARIABLE(d0         , "jd0_1"    , -1000.);   
	INIT_VARIABLE(dZ         , "jdZ_1"    , -1000.);  
	INIT_VARIABLE(nParticles , "npart_1"  , 0.);  
	
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
	INIT_VARIABLE(dRMean     , "drm_1"    , 0.);  
	INIT_VARIABLE(dRMeanNeut , "drmne_1"  , 0.);  
	INIT_VARIABLE(dRMeanEm   , "drem_1"   , 0.);  
	INIT_VARIABLE(dRMeanCh   , "drch_1"   , 0.);  
	INIT_VARIABLE(dR2Mean    , ""         , 0.);  
	
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

	INIT_VARIABLE(majW  ,"" ,1.);  
	INIT_VARIABLE(minW  ,"" ,1.);  

	INIT_VARIABLE(frac01    ,"" ,0.);  
	INIT_VARIABLE(frac02    ,"" ,0.);  
	INIT_VARIABLE(frac03    ,"" ,0.);  
	INIT_VARIABLE(frac04    ,"" ,0.);  
	INIT_VARIABLE(frac05   ,"" ,0.);  
	INIT_VARIABLE(frac06   ,"" ,0.);  
	INIT_VARIABLE(frac07   ,"" ,0.);  
	
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
	
}

#undef INIT_VARIABLE
