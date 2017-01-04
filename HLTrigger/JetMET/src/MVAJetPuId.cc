#include "HLTrigger/JetMET/interface/MVAJetPuId.h"
//#include "RecoJets/JetProducers/interface/MVAJetPuId.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/Utils/interface/TMVAZipReader.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
//http://cmslxr.fnal.gov/source/RecoJets/JetProducers/python/puJetIDAlgo_cff.py
const float large_val = std::numeric_limits<float>::max();


MVAJetPuId::MVAJetPuId(const edm::ParameterSet & ps) 
{
	impactParTkThreshod_ = 1.;

	tmvaWeights_         = edm::FileInPath(ps.getParameter<std::string>("tmvaWeights")).fullPath(); 
	tmvaMethod_          = ps.getParameter<std::string>("tmvaMethod");
	tmvaVariables_       = ps.getParameter<std::vector<std::string> >("tmvaVariables");
	tmvaSpectators_      = ps.getParameter<std::vector<std::string> >("tmvaSpectators");
	version_             = ps.getParameter<int>("version");
	reader_              = 0;
	edm::ParameterSet jetConfig = ps.getParameter<edm::ParameterSet>("JetIdParams");
	for(int i0 = 0; i0 < 3; i0++) { 
		std::string lCutType                            = "Tight";
		if(i0 == PileupJetIdentifier::kMedium) lCutType = "Medium";
		if(i0 == PileupJetIdentifier::kLoose)  lCutType = "Loose";
		for(int i1 = 0; i1 < 1; i1++) {
			std::vector<double> pt010  = jetConfig.getParameter<std::vector<double> >(("Pt010_" +lCutType).c_str());
			std::vector<double> pt1020 = jetConfig.getParameter<std::vector<double> >(("Pt1020_"+lCutType).c_str());
			std::vector<double> pt2030 = jetConfig.getParameter<std::vector<double> >(("Pt2030_"+lCutType).c_str());
			std::vector<double> pt3050 = jetConfig.getParameter<std::vector<double> >(("Pt3050_"+lCutType).c_str());
			for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][0][i2] = pt010 [i2];
			for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][1][i2] = pt1020[i2];
			for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][2][i2] = pt2030[i2];
			for(int i2 = 0; i2 < 4; i2++) mvacut_[i0][3][i2] = pt3050[i2];

		}
	}
	setup();
}





MVAJetPuId::MVAJetPuId(int version,
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



void MVAJetPuId::setup()
{
	initVariables();


		tmvaVariables_.clear();
		tmvaVariables_.push_back( "nvtx"  );
		tmvaVariables_.push_back( "nParticles" );
		tmvaVariables_.push_back( "nCharged" );
		tmvaVariables_.push_back( "majW"   );
		tmvaVariables_.push_back( "minW"   );
		tmvaVariables_.push_back( "frac01"    );
		tmvaVariables_.push_back( "frac02" );
		tmvaVariables_.push_back( "frac03"   );
		tmvaVariables_.push_back( "frac04"  );
		tmvaVariables_.push_back( "ptD"  );
		tmvaVariables_.push_back( "beta"   );
		tmvaVariables_.push_back( "betaStar"  );
		tmvaVariables_.push_back( "dR2Mean"  );
		tmvaVariables_.push_back( "dRMean" );
		tmvaVariables_.push_back( "dRLeadCent");
		tmvaVariables_.push_back( "dRMeanCh");
		tmvaVariables_.push_back("jspt_1");
		tmvaVariables_.push_back("jseta_1");

		tmvaNames_["nvtx"] = "rho";
		tmvaNames_["jspt_1"] = "pt"; 
		tmvaNames_["jseta_1"] = "eta";
		tmvaNames_["nParticles"] = "nTot";
		tmvaNames_["nCharged"] = "nCh";
		tmvaNames_["majW"] = "axisMajor";
		tmvaNames_["minW"] = "axisMinor";
		tmvaNames_["frac01"] = "fRing0";
		tmvaNames_["frac02"] = "fRing1";
		tmvaNames_["frac03"] = "fRing2";
		tmvaNames_["frac04"] = "fRing3";
		tmvaNames_["ptD"] = "ptD";
		tmvaNames_["beta"] = "beta";
		tmvaNames_["betaStar"] = "betaStar";
		tmvaNames_["dR2Mean"] = "DR_weighted";
		tmvaNames_["dRMean"] = "pull:=(pull, 0.1)";  
		tmvaNames_["dRLeadCent"] = "jetR";  
		tmvaNames_["dRMeanCh"] = "jetRchg";
}



MVAJetPuId::~MVAJetPuId() 
{
	if( reader_ ) {
		delete reader_;
	}
}


void assign(const std::vector<float> & vec, float & a, float & b, float & c, float & d )
{
	size_t sz = vec.size();
	a = ( sz > 0 ? vec[0] : 0. );
	b = ( sz > 1 ? vec[1] : 0. );
	c = ( sz > 2 ? vec[2] : 0. );
	d = ( sz > 3 ? vec[3] : 0. );
}


void setPtEtaPhi(const reco::Candidate & p, float & pt, float & eta, float &phi )
{
	pt  = p.pt();
	eta = p.eta();
	phi = p.phi();
}


void MVAJetPuId::bookReader()
{
	reader_ = new TMVA::Reader("!Color:Silent");
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
	reco::details::loadTMVAWeights(reader_,  tmvaMethod_.c_str(), tmvaWeights_.c_str() ); 
}


void MVAJetPuId::set(const PileupJetIdentifier & id)
{
	internalId_ = id;
}


void MVAJetPuId::runMva()
{
	if( ! reader_ ) { bookReader();}
	if(fabs(internalId_.jetEta_) <  5.0) internalId_.mva_ = reader_->EvaluateMVA( tmvaMethod_.c_str() );
	if(fabs(internalId_.jetEta_) >= 5.0) internalId_.mva_ = -2.;
	internalId_.idFlag_ = computeIDflag(internalId_.mva_,internalId_.jetPt_,internalId_.jetEta_);
}

std::pair<int,int> MVAJetPuId::getJetIdKey(float jetPt, float jetEta)
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


int MVAJetPuId::computeIDflag(float mva, float jetPt, float jetEta)
{
	std::pair<int,int> jetIdKey = getJetIdKey(jetPt,jetEta);
	return computeIDflag(mva,jetIdKey.first,jetIdKey.second);
}

int MVAJetPuId::computeIDflag(float mva,int ptId,int etaId)
{
	int idFlag(0);
	if(mva > mvacut_[PileupJetIdentifier::kTight ][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kTight;
	if(mva > mvacut_[PileupJetIdentifier::kMedium][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kMedium;
	if(mva > mvacut_[PileupJetIdentifier::kLoose ][ptId][etaId]) idFlag += 1 << PileupJetIdentifier::kLoose;
	return idFlag;
}

PileupJetIdentifier MVAJetPuId::computeMva()
{
	runMva();
	return PileupJetIdentifier(internalId_);
}

PileupJetIdentifier MVAJetPuId::computeIdVariables(const reco::Jet * jet, float jec, const reco::Vertex * vtx,
		const reco::VertexCollection & allvtx, double rho,
		bool calculateMva) 
{
	static std::atomic<int> printWarning{10};
	typedef std::vector <reco::PFCandidatePtr> constituents_type;
	typedef std::vector <reco::PFCandidatePtr>::iterator constituents_iterator;

	resetVariables();

	const reco::PFJet * pfjet = dynamic_cast<const reco::PFJet *>(jet);

	if( jec < 0. ) {
		jec = 1.;
	}

	constituents_type constituents = pfjet->getPFConstituents();

	reco::PFCandidatePtr lLead, lSecond, lLeadNeut, lLeadEm, lLeadCh, lTrail;
	std::vector<float> frac, fracCh, fracEm, fracNeut;
	float cones[] = { 0.1, 0.2, 0.3, 0.4 };
	size_t ncones = sizeof(cones)/sizeof(float);
	float * coneFracs[]     = { &internalId_.frac01_, &internalId_.frac02_, &internalId_.frac03_, &internalId_.frac04_};
	TMatrixDSym covMatrix(2); covMatrix = 0.;

	reco::TrackRef impactTrack;
	float jetPt = jet->pt() / jec; // use uncorrected pt for shape variables
	float sumPt = 0., sumPt2 = 0., sumTkPt = 0.,sumPtCh=0,sumPtNe = 0; float sum_deta =0 ; float sum_dphi =0 ; float sum_deta2 =0 ; float sum_detadphi =0 ; float sum_dphi2=0;
	setPtEtaPhi(*jet,internalId_.jetPt_,internalId_.jetEta_,internalId_.jetPhi_); // use corrected pt for jet kinematics
	internalId_.jetM_ = jet->mass(); 
	internalId_.nvtx_ = rho;//allvtx.size();
	for(constituents_iterator it=constituents.begin(); it!=constituents.end(); ++it) {
		reco::PFCandidatePtr & icand = *it;
		float candPt = icand->pt();
		float candPtFrac = candPt/jetPt;
		float candDr   = reco::deltaR(**it,*jet);
		float candDeta = fabs( (*it)->eta() - jet->eta() );
		float candDphi = reco::deltaPhi(**it,*jet);
		float candPtDr = candPt * candDr;
		size_t icone = std::lower_bound(&cones[0],&cones[ncones],candDr) - &cones[0];
		float weight2 = candPt * candPt;


		if( lLead.isNull() || candPt > lLead->pt() ) {
			lSecond = lLead;
			lLead = icand; 
		} else if( (lSecond.isNull() || candPt > lSecond->pt()) && (candPt < lLead->pt()) ) {
			lSecond = icand;
		}

		//internalId_.dRMean_     += candPtDr;
		internalId_.dR2Mean_    += candPtDr*candPtDr;

		internalId_.ptD_ += candPt*candPt;
		sumPt += candPt;
		sumPt2 += candPt*candPt;
		sum_deta     += candDeta*weight2;
		sum_dphi     += candDphi*weight2;
		sum_deta2    += candDeta*candDeta*weight2;
		sum_detadphi += candDeta*candDphi*weight2;
		sum_dphi2    += candDphi*candDphi*weight2;
		//Teta         += candPt * candDR * candDeta;
		//Tphi         += candPt * candDR * candDphi;



		frac.push_back(candPtFrac);
		if( icone < ncones ) { *coneFracs[icone] += candPt; }


		if( icand->particleId() == reco::PFCandidate::h0 ) {
			if (lLeadNeut.isNull() || candPt > lLeadNeut->pt()) { lLeadNeut = icand; }
			internalId_.dRMeanNeut_ += candPtDr;
			fracNeut.push_back(candPtFrac);
			sumPtNe               += candPt;
		}

		if( icand->particleId() == reco::PFCandidate::gamma ) {
			if(lLeadEm.isNull() || candPt > lLeadEm->pt())  { lLeadEm = icand; }
			internalId_.dRMeanEm_ += candPtDr;
			fracEm.push_back(candPtFrac);
			sumPtNe               += candPt;
		}

		if(  icand->trackRef().isNonnull() && icand->trackRef().isAvailable() ) {
			if (lLeadCh.isNull() || candPt > lLeadCh->pt()) { lLeadCh = icand; }
			//internalId_.dRMeanCh_  += candPtDr;
			fracCh.push_back(candPtFrac);
			sumPtCh                += candPt;
		}

		if(  icand->trackRef().isNonnull() && icand->trackRef().isAvailable() ) {
			try { 
				float tkpt = icand->trackRef()->pt(); 
				sumTkPt += tkpt;
				bool inVtx0 = find( vtx->tracks_begin(), vtx->tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != vtx->tracks_end();
				bool inAnyOther = false;

				double dZ0 = fabs(icand->trackRef()->dz(vtx->position()));
				double dZ = dZ0;
				for(reco::VertexCollection::const_iterator  vi=allvtx.begin(); vi!=allvtx.end(); ++vi ) {
					const reco::Vertex & iv = *vi;
					if( iv.isFake() || iv.ndof() < 4 ) { continue; }

					bool isVtx0  = (iv.position() - vtx->position()).r() < 0.02;

					if( ! isVtx0 && ! inAnyOther ) {
						inAnyOther = find( iv.tracks_begin(), iv.tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != iv.tracks_end();
					}

					dZ = std::min(dZ,fabs(icand->trackRef()->dz(iv.position())));
				}
				if( inVtx0 && ! inAnyOther ) {
					internalId_.betaClassic_ += tkpt;
				} else if( ! inVtx0 && inAnyOther ) {
					internalId_.betaStarClassic_ += tkpt;
				}

				if( dZ0 < 0.2 ) {
					internalId_.beta_ += tkpt;
				} else if( dZ < 0.2 ) {
					internalId_.betaStar_ += tkpt;
				}
			} catch (cms::Exception &e) {
				if(printWarning-- > 0) { std::cerr << e << std::endl; }
			}
		}

		if( lTrail.isNull() || candPt < lTrail->pt() ) {
			lTrail = icand; 
		}
	}

	assert( lLead.isNonnull() );
	if ( lSecond.isNull() )   { lSecond   = lTrail; }
	if ( lLeadNeut.isNull() ) { lLeadNeut = lTrail; }
	if ( lLeadEm.isNull() )   { lLeadEm   = lTrail; }
	if ( lLeadCh.isNull() )   { lLeadCh   = lTrail; }
	impactTrack = lLeadCh->trackRef();

	internalId_.nCharged_    = pfjet->chargedMultiplicity();
	internalId_.nNeutrals_   = pfjet->neutralMultiplicity();
	internalId_.chgEMfrac_   = pfjet->chargedEmEnergy()    /jet->energy();
	internalId_.neuEMfrac_   = pfjet->neutralEmEnergy()    /jet->energy();
	internalId_.chgHadrfrac_ = pfjet->chargedHadronEnergy()/jet->energy();
	internalId_.neuHadrfrac_ = pfjet->neutralHadronEnergy()/jet->energy();

	if( impactTrack.isNonnull() && impactTrack.isAvailable() ) {
		internalId_.d0_ = fabs(impactTrack->dxy(vtx->position()));
		internalId_.dZ_ = fabs(impactTrack->dz(vtx->position()));
	} else {
		if(printWarning-- > 0) { std::cerr << "WARNING : did not find any valid track reference attached to the jet " << std::endl; 
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

		//covMatrix(0,0) /= sumPt2;
		//covMatrix(0,1) /= sumPt2;
		//covMatrix(1,1) /= sumPt2;
		//covMatrix(1,0)  = covMatrix(0,1);
		//internalId_.etaW_ = sqrt(covMatrix(0,0));
		//internalId_.phiW_ = sqrt(covMatrix(1,1));
		//internalId_.jetW_ = 0.5*(internalId_.etaW_+internalId_.phiW_);
		//TVectorD eigVals(2); eigVals = TMatrixDSymEigen(covMatrix).GetEigenValues();
		//	
		if( internalId_.majW_ < internalId_.minW_ ) { std::swap(internalId_.majW_,internalId_.minW_); }

		//internalId_.dRLeadCent_ = reco::deltaR(*jet,*lLead);
		if( lSecond.isNonnull() ) { internalId_.dRLead2nd_  = reco::deltaR(*jet,*lSecond); }
		internalId_.dRMeanNeut_ /= jetPt;
		internalId_.dRMeanEm_   /= jetPt;
		//internalId_.dRMeanCh_   /= jetPt;
		internalId_.dR2Mean_    /= sumPt2;
		for(size_t ic=0; ic<ncones; ++ic){
			*coneFracs[ic]     /= jetPt;
		}

		double ptMean = sumPt/internalId_.nParticles_;
		double ptRMS  = 0;
		for(unsigned int i0 = 0; i0 < frac.size(); i0++) {ptRMS+=(frac[i0]-ptMean)*(frac[i0]-ptMean);}
		ptRMS/=internalId_.nParticles_;
		ptRMS=sqrt(ptRMS);
		internalId_.dRMeanCh_ = internalId_.leadChPt_/sumPt;
		internalId_.dRLeadCent_ = internalId_.leadPt_/sumPt;

		internalId_.ptMean_  = ptMean;
		internalId_.ptRMS_   = ptRMS/jetPt;
		internalId_.pt2A_    = sqrt( internalId_.ptD_     /internalId_.nParticles_)/jetPt;
		internalId_.ptD_     = sqrt( internalId_.ptD_)    / sumPt;
		internalId_.sumPt_   = sumPt;
		internalId_.sumChPt_ = sumPtCh;
		internalId_.sumNePt_ = sumPtNe;
		if (sumPt > 0) {
			internalId_.beta_     /= sumPt;
			internalId_.betaStar_ /= sumPt;
		} else {
			assert( internalId_.beta_ == 0. && internalId_.betaStar_ == 0.);
		}
		float ave_deta = sum_deta/sumPt2;
		float ave_dphi = sum_dphi/sumPt2;
		float ave_deta2 = sum_deta2/sumPt2;
		float ave_dphi2 = sum_dphi2/sumPt2;
		float a = ave_deta2-ave_deta*ave_deta;
		float b = ave_dphi2-ave_dphi*ave_dphi;
		float c = -(sum_detadphi/sumPt2-ave_deta*ave_dphi);
		float delta = sqrt(fabs((a-b)*(a-b)+4*c*c));
		float axis1=0; float axis2=0;
		if (a+b+delta > 0) {
			axis1 = sqrt(0.5*(a+b+delta));
		}
		if (a+b-delta > 0) {
			axis2 = sqrt(0.5*(a+b-delta));
		}
		internalId_.majW_ = axis1; //sqrt(fabs(eigVals(0)));
		internalId_.minW_ = axis2;//sqrt(fabs(eigVals(1)));
		//compute Pull

		float ddetaR_sum(0.0), ddphiR_sum(0.0);
		for(int i=0; i<internalId_.nParticles_; ++i) {
			reco::PFCandidatePtr part = pfjet->getPFConstituent(i);
			float weight =part->pt()*part->pt() ;
			float deta = part->eta() - jet->eta();
			float dphi = 2*atan(tan((part->phi()-jet->phi())/2));
			float ddeta, ddphi, ddR;
			ddeta = deta - ave_deta ;
			ddphi = 2*atan(tan((dphi - ave_dphi)/2.)) ;
			ddR = sqrt(ddeta*ddeta + ddphi*ddphi);
			ddetaR_sum += ddR*ddeta*weight;
			ddphiR_sum += ddR*ddphi*weight;
		}if (sumPt2 > 0) {
			float ddetaR_ave = ddetaR_sum/sumPt2;
			float ddphiR_ave = ddphiR_sum/sumPt2;
			internalId_.dRMean_ = sqrt(ddetaR_ave*ddetaR_ave+ddphiR_ave*ddphiR_ave);
		}



	}

	if( calculateMva ) {
		runMva();
	}

	return PileupJetIdentifier(internalId_);
}

std::string MVAJetPuId::dumpVariables() const
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

void MVAJetPuId::resetVariables()
{
	internalId_.idFlag_    = 0;
	for(variables_list_t::iterator it=variables_.begin(); 
			it!=variables_.end(); ++it ) {
		*it->second.first = it->second.second;
	}
}

#define INIT_VARIABLE(NAME,TMVANAME,VAL)    \
	internalId_.NAME ## _ = VAL; \
variables_[ # NAME   ] = std::make_pair(& internalId_.NAME ## _, VAL);

void MVAJetPuId::initVariables()
{
	internalId_.idFlag_    = 0;
	INIT_VARIABLE(mva        , "", -100.);
	INIT_VARIABLE(jetPt      , "pt", 0.);
	INIT_VARIABLE(jetEta     , "eta", large_val);
	INIT_VARIABLE(jetPhi     , "", large_val);
	INIT_VARIABLE(jetM       , "", 0.);
	INIT_VARIABLE(nCharged   , "", 0.);
	INIT_VARIABLE(nNeutrals  , "", 0.);

	INIT_VARIABLE(chgEMfrac  , "", 0.);
	INIT_VARIABLE(neuEMfrac  , "", 0.);
	INIT_VARIABLE(chgHadrfrac, "", 0.);
	INIT_VARIABLE(neuHadrfrac, "", 0.);

	INIT_VARIABLE(d0         , ""    , -1000.);   
	INIT_VARIABLE(dZ         , ""    , -1000.);  
	INIT_VARIABLE(nParticles , ""  , 0.);  

	INIT_VARIABLE(leadPt     , ""    , 0.);  
	INIT_VARIABLE(leadEta    , ""   , large_val);  
	INIT_VARIABLE(leadPhi    , ""   , large_val);  
	INIT_VARIABLE(secondPt   , ""    , 0.);  
	INIT_VARIABLE(secondEta  , ""   , large_val);  
	INIT_VARIABLE(secondPhi  , ""   , large_val);  
	INIT_VARIABLE(leadNeutPt , ""    , 0.);  
	INIT_VARIABLE(leadNeutEta, ""   , large_val);  

	INIT_VARIABLE(dRLeadCent , ""   , 0.);  
	INIT_VARIABLE(dRMean     , ""    , 0.);  
	INIT_VARIABLE(dRMeanCh   , ""   , 0.);  
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

	INIT_VARIABLE(beta   ,"" ,0.);  
	INIT_VARIABLE(betaStar   ,"" ,0.);  
	INIT_VARIABLE(betaClassic   ,"" ,0.);  
	INIT_VARIABLE(betaStarClassic   ,"" ,0.);  
	INIT_VARIABLE(nvtx   ,"rho" ,0.);  

}
#undef INIT_VARIABLE

