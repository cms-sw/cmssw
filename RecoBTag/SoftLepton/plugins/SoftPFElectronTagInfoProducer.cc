#include "RecoBTag/SoftLepton/plugins/SoftPFElectronTagInfoProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

// Transient Track and IP
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <cmath>
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/PatCandidates/interface/Electron.h"


SoftPFElectronTagInfoProducer::SoftPFElectronTagInfoProducer (const edm::ParameterSet& conf)
{
	token_jets          = consumes<edm::View<reco::Jet> >(conf.getParameter<edm::InputTag>("jets"));
	token_elec          = consumes<edm::View<reco::GsfElectron> >(conf.getParameter<edm::InputTag>("electrons"));
	token_primaryVertex = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("primaryVertex"));
	token_BeamSpot      = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
	token_allConversions= consumes<reco::ConversionCollection>(edm::InputTag("allConversions"));
	DeltaRElectronJet   = conf.getParameter<double>("DeltaRElectronJet");
	MaxSip3Dsig            = conf.getParameter<double>("MaxSip3Dsig");
        produces<reco::CandSoftLeptonTagInfoCollection>();
}

SoftPFElectronTagInfoProducer::~SoftPFElectronTagInfoProducer()
{

}

void SoftPFElectronTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	std::auto_ptr<reco::CandSoftLeptonTagInfoCollection> theElecTagInfo(new reco::CandSoftLeptonTagInfoCollection);
	edm::ESHandle<TransientTrackBuilder> builder;
 	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
 	transientTrackBuilder=builder.product();
 
 	edm::Handle<reco::VertexCollection> PVCollection;
	iEvent.getByToken(token_primaryVertex, PVCollection);
 	if(!PVCollection.isValid()) return;
 	if(!PVCollection->empty()){
		goodvertex = true;
		vertex=&PVCollection->front();
	}else goodvertex = false; 
  	edm::Handle<reco::ConversionCollection> hConversions;
  	iEvent.getByToken(token_allConversions, hConversions);
 
  	edm::Handle<edm::View<reco::Jet> > theJetCollection;
  	iEvent.getByToken(token_jets, theJetCollection);

	edm::Handle<edm::View<reco::GsfElectron > > theGEDGsfElectronCollection;
  	iEvent.getByToken(token_elec, theGEDGsfElectronCollection);

	edm::Handle<reco::BeamSpot> bsHandle;
  	iEvent.getByToken(token_BeamSpot, bsHandle);
  	const reco::BeamSpot &beamspot = *bsHandle.product();

	 for (unsigned int i = 0; i < theJetCollection->size(); i++){
    		edm::RefToBase<reco::Jet> jetRef = theJetCollection->refAt(i);
		reco::CandSoftLeptonTagInfo tagInfo;
                tagInfo.setJetRef(jetRef);
    		std::vector<const reco::GsfElectron *> Elec;
		for(unsigned int ie=0, ne=theGEDGsfElectronCollection->size(); ie<ne; ++ie){
			//Get the edm::Ptr and the GsfElectron
			edm::Ptr<reco::Candidate> lepPtr=theGEDGsfElectronCollection->ptrAt(ie);
			const reco::GsfElectron* recoelectron=theGEDGsfElectronCollection->refAt(ie).get();
			const pat::Electron* patelec=dynamic_cast<const pat::Electron*>(recoelectron);
			if(patelec){
				if(!patelec->passConversionVeto()) continue;
			}
			else{
				if(ConversionTools::hasMatchedConversion(*(recoelectron),hConversions,beamspot.position())) continue;
			}
			//Make sure that the electron is inside the jet
			if(reco::deltaR2((*recoelectron),(*jetRef))>DeltaRElectronJet*DeltaRElectronJet) continue;
			// Need a gsfTrack
			if(recoelectron->gsfTrack().get()==NULL) continue;
			reco::SoftLeptonProperties properties;
			// reject if it has issues with the track
			if(!isElecClean(iEvent,recoelectron) ) continue;
			//Compute the TagInfos members
			math::XYZVector pel=recoelectron->p4().Vect();
  			math::XYZVector pjet=jetRef->p4().Vect();
  			reco::TransientTrack transientTrack=transientTrackBuilder->build(recoelectron->gsfTrack());
			Measurement1D ip2d = IPTools::signedTransverseImpactParameter(transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), *vertex).second;
			Measurement1D ip3d    = IPTools::signedImpactParameter3D(transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), *vertex).second;
  			properties.sip2dsig    = ip2d.significance();
  			properties.sip3dsig    = ip3d.significance();
			properties.sip2d    = ip2d.value();
  			properties.sip3d    = ip3d.value();
  			properties.deltaR   = reco::deltaR((*jetRef), (*recoelectron));
  			properties.ptRel    = ( (pjet-pel).Cross(pel) ).R() / pjet.R();
  			float mag = pel.R()*pjet.R();
  			float dot = recoelectron->p4().Dot(jetRef->p4());
  			properties.etaRel   = -log((mag - dot)/(mag + dot)) / 2.;
  			properties.ratio    = recoelectron->pt() / jetRef->pt();
  			properties.ratioRel = recoelectron->p4().Dot(jetRef->p4()) / pjet.Mag2();
  			properties.p0Par    = boostedPPar(recoelectron->momentum(), jetRef->momentum());
			properties.elec_mva    = recoelectron->mva_e_pi();
			if(std::abs(properties.sip3dsig)>MaxSip3Dsig) continue;
			// Fill the TagInfos
			tagInfo.insert(lepPtr, properties );
		}
		theElecTagInfo->push_back(tagInfo);
  	}
  	iEvent.put(theElecTagInfo);
}

bool SoftPFElectronTagInfoProducer::isElecClean(edm::Event& iEvent,const reco::GsfElectron*candidate)
{
    using namespace reco;
    const HitPattern &hitPattern = candidate->gsfTrack().get()->hitPattern();
    uint32_t hit = hitPattern.getHitPattern(HitPattern::TRACK_HITS, 0);
    bool hitCondition = !(HitPattern::validHitFilter(hit) 
            && ((HitPattern::pixelBarrelHitFilter(hit) 
                        && HitPattern::getLayer(hit) < 3) 
                    || HitPattern::pixelEndcapHitFilter(hit))); 
    if(hitCondition) return false;

    return true;
}

float SoftPFElectronTagInfoProducer::boostedPPar(const math::XYZVector& vector, const math::XYZVector& axis) {
	static const double lepton_mass = 0.00; 
	static const double jet_mass    = 5.279; 
  	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > lepton(vector.Dot(axis) / axis.r(), ROOT::Math::VectorUtil::Perp(vector, axis), 0., lepton_mass);
  	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > jet( axis.r(), 0., 0., jet_mass );
  	ROOT::Math::BoostX boost( -jet.Beta() );
  	return boost(lepton).x();
}
