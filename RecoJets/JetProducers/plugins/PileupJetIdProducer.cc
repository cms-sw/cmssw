// -*- C++ -*-
//
// Package:    PileupJetIdProducer
// Class:      PileupJetIdProducer
// 
/**\class PileupJetIdProducer PileupJetIdProducer.cc CMGTools/PileupJetIdProducer/src/PileupJetIdProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Pasquale Musella,40 2-A12,+41227671706,
//         Created:  Wed Apr 18 15:48:47 CEST 2012
//
//

#include <memory>
#include "RecoJets/JetProducers/plugins/PileupJetIdProducer.h"


// ------------------------------------------------------------------------------------------
PileupJetIdProducer::PileupJetIdProducer(const edm::ParameterSet& iConfig)
{
	runMvas_ = iConfig.getParameter<bool>("runMvas");
	produceJetIds_ = iConfig.getParameter<bool>("produceJetIds");
	jets_ = iConfig.getParameter<edm::InputTag>("jets");
	vertexes_ = iConfig.getParameter<edm::InputTag>("vertexes");
	jetids_  = iConfig.getParameter<edm::InputTag>("jetids");
	inputIsCorrected_ = iConfig.getParameter<bool>("inputIsCorrected");
	applyJec_ = iConfig.getParameter<bool>("applyJec");
	jec_ =  iConfig.getParameter<std::string>("jec");
	rho_ = iConfig.getParameter<edm::InputTag>("rho");
	residualsFromTxt_ = iConfig.getParameter<bool>("residualsFromTxt");
	if(residualsFromTxt_) residualsTxt_ = iConfig.getParameter<edm::FileInPath>("residualsTxt");
	std::vector<edm::ParameterSet> algos = iConfig.getParameter<std::vector<edm::ParameterSet> >("algos");
	
	jecCor_ = 0;

	if( ! runMvas_ ) assert( algos.size() == 1 );
	
	if( produceJetIds_ ) {
		produces<edm::ValueMap<StoredPileupJetIdentifier> > ("");
	}
	for(std::vector<edm::ParameterSet>::iterator it=algos.begin(); it!=algos.end(); ++it) {
		std::string label = it->getParameter<std::string>("label");
		algos_.push_back( std::make_pair(label,new PileupJetIdAlgo(*it, runMvas_)) );
		if( runMvas_ ) {
			produces<edm::ValueMap<float> > (label+"Discriminant");
			produces<edm::ValueMap<int> > (label+"Id");
		}
	}

	input_jet_token_ = consumes<edm::View<reco::Jet> >(jets_);
	input_vertex_token_ = consumes<reco::VertexCollection>(vertexes_);
        input_vm_pujetid_token_ = consumes<edm::ValueMap<StoredPileupJetIdentifier> >(jetids_);
        input_rho_token_ = consumes<double>(rho_); 

}



// ------------------------------------------------------------------------------------------
PileupJetIdProducer::~PileupJetIdProducer()
{
}


// ------------------------------------------------------------------------------------------
void
PileupJetIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	using namespace std;
	using namespace reco;
	
	// Input jets
	Handle<View<Jet> > jetHandle;
	iEvent.getByToken(input_jet_token_,jetHandle);
	const View<Jet> & jets = *jetHandle;
	// vertexes 
	Handle<VertexCollection> vertexHandle;
	if(  produceJetIds_ ) {
	        iEvent.getByToken(input_vertex_token_, vertexHandle);
	}
	const VertexCollection & vertexes = *(vertexHandle.product());
	// input variables
	Handle<ValueMap<StoredPileupJetIdentifier> > vmap;
	if( ! produceJetIds_ ) {
		iEvent.getByToken(input_vm_pujetid_token_, vmap);
	}
	// rho
	edm::Handle< double > rhoH;
	double rho = 0.;
	
	// products
	vector<StoredPileupJetIdentifier> ids; 
	map<string, vector<float> > mvas;
	map<string, vector<int> > idflags;

	VertexCollection::const_iterator vtx;
	if( produceJetIds_ ) {
		// require basic quality cuts on the vertexes
		vtx = vertexes.begin();
		while( vtx != vertexes.end() && ( vtx->isFake() || vtx->ndof() < 4 ) ) {
			++vtx;
		}
		if( vtx == vertexes.end() ) { vtx = vertexes.begin(); }
	}
	
	// Loop over input jets
	for ( unsigned int i=0; i<jets.size(); ++i ) {
		// Pick the first algo to compute the input variables
		vector<pair<string,PileupJetIdAlgo *> >::iterator algoi = algos_.begin();
		PileupJetIdAlgo * ialgo = algoi->second;
		
		const Jet & jet = jets.at(i);
		//const pat::Jet * patjet =  dynamic_cast<const pat::Jet *>(&jet);
		//bool ispat = patjet != 0;
		
		// Get jet energy correction
		float jec = 0.;
		if( applyJec_ ) {
			// If haven't done it get rho from the event
			if( rho == 0. ) {
				iEvent.getByToken(input_rho_token_,rhoH);
				rho = *rhoH;
			}
			// jet corrector
			if( jecCor_ == 0 ) {
				initJetEnergyCorrector( iSetup, iEvent.isRealData() );
			}
			//if( ispat ) {
			//	jecCor_->setJetPt(patjet->correctedJet(0).pt());
			//} else {
			jecCor_->setJetPt(jet.pt());
			//}
			jecCor_->setJetEta(jet.eta());
			jecCor_->setJetA(jet.jetArea());
			jecCor_->setRho(rho);
			jec = jecCor_->getCorrection();
		}
		// If it was requested AND the input is an uncorrected jet apply the JEC
		bool applyJec = applyJec_ && !inputIsCorrected_;  //( ! ispat && ! inputIsCorrected_ );
		reco::Jet * corrJet = 0;
		
		if( applyJec ) {
			float scale = jec;
			//if( ispat ) {
			//	corrJet = new pat::Jet(patjet->correctedJet(0)) ;
			//} else {
			corrJet = dynamic_cast<reco::Jet *>( jet.clone() );
			//}
			corrJet->scaleEnergy(scale);
		}
		const reco::Jet * theJet = ( applyJec ? corrJet : &jet );
	
		PileupJetIdentifier puIdentifier;
		if( produceJetIds_ ) {
		        // Compute the input variables
		        puIdentifier = ialgo->computeIdVariables(theJet, jec,  &(*vtx), vertexes, rho);
			ids.push_back( puIdentifier );
		} else {
		        // Or read it from the value map
			puIdentifier = (*vmap)[jets.refAt(i)]; 
			puIdentifier.jetPt(theJet->pt());    // make sure JEC is applied when computing the MVA
			puIdentifier.jetEta(theJet->eta());
			puIdentifier.jetPhi(theJet->phi());
			ialgo->set(puIdentifier); 
			puIdentifier = ialgo->computeMva();
		}
	
		if( runMvas_ ) {
		        // Compute the MVA and WP
			mvas[algoi->first].push_back( puIdentifier.mva() );
			idflags[algoi->first].push_back( puIdentifier.idFlag() );
			for( ++algoi; algoi!=algos_.end(); ++algoi) {
				ialgo = algoi->second;
				ialgo->set(puIdentifier);
				PileupJetIdentifier id = ialgo->computeMva();
				mvas[algoi->first].push_back( id.mva() );
				idflags[algoi->first].push_back( id.idFlag() );
			}
		}
		
		// cleanup
		if( corrJet ) { delete corrJet; }
	}
	
	// Produce the output value maps
	if( runMvas_ ) {
		for(vector<pair<string,PileupJetIdAlgo *> >::iterator ialgo = algos_.begin(); ialgo!=algos_.end(); ++ialgo) {
			// MVA
			vector<float> & mva = mvas[ialgo->first];
			auto_ptr<ValueMap<float> > mvaout(new ValueMap<float>());
			ValueMap<float>::Filler mvafiller(*mvaout);
			mvafiller.insert(jetHandle,mva.begin(),mva.end());
			mvafiller.fill();
			iEvent.put(mvaout,ialgo->first+"Discriminant");
			
			// WP
			vector<int> & idflag = idflags[ialgo->first];
			auto_ptr<ValueMap<int> > idflagout(new ValueMap<int>());
			ValueMap<int>::Filler idflagfiller(*idflagout);
			idflagfiller.insert(jetHandle,idflag.begin(),idflag.end());
			idflagfiller.fill();
			iEvent.put(idflagout,ialgo->first+"Id");
		}
	}
	// input variables
	if( produceJetIds_ ) {
		assert( jetHandle->size() == ids.size() );
		auto_ptr<ValueMap<StoredPileupJetIdentifier> > idsout(new ValueMap<StoredPileupJetIdentifier>());
		ValueMap<StoredPileupJetIdentifier>::Filler idsfiller(*idsout);
		idsfiller.insert(jetHandle,ids.begin(),ids.end());
		idsfiller.fill();
		iEvent.put(idsout);
	}
}



// ------------------------------------------------------------------------------------------
void
PileupJetIdProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	//The following says we do not know what parameters are allowed so do no validation
	// Please change this to state exactly what you do use, even if it is no parameters
	edm::ParameterSetDescription desc;
	desc.setUnknown();
	descriptions.addDefault(desc);
}


// ------------------------------------------------------------------------------------------
void 
PileupJetIdProducer::initJetEnergyCorrector(const edm::EventSetup &iSetup, bool isData)
{
	//jet energy correction levels to apply on raw jet
	std::vector<std::string> jecLevels;
	jecLevels.push_back("L1FastJet");
	jecLevels.push_back("L2Relative");
	jecLevels.push_back("L3Absolute");
	if(isData && ! residualsFromTxt_ ) jecLevels.push_back("L2L3Residual");

	//check the corrector parameters needed according to the correction levels
	edm::ESHandle<JetCorrectorParametersCollection> parameters;
	iSetup.get<JetCorrectionsRecord>().get(jec_,parameters);
	for(std::vector<std::string>::const_iterator ll = jecLevels.begin(); ll != jecLevels.end(); ++ll)
	{ 
		const JetCorrectorParameters& ip = (*parameters)[*ll];
		jetCorPars_.push_back(ip); 
	} 
	if( isData && residualsFromTxt_ ) {
		jetCorPars_.push_back(JetCorrectorParameters(residualsTxt_.fullPath())); 
	}
	
	//instantiate the jet corrector
	jecCor_ = new FactorizedJetCorrector(jetCorPars_);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PileupJetIdProducer);
