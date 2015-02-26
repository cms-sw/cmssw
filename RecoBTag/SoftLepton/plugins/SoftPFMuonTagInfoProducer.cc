// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#include "RecoBTag/SoftLepton/plugins/SoftPFMuonTagInfoProducer.h"
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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
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

SoftPFMuonTagInfoProducer::SoftPFMuonTagInfoProducer(const edm::ParameterSet& conf) {
  jetToken    = consumes<edm::View<reco::Jet> >(conf.getParameter<edm::InputTag>("jets"));
  muonToken   = consumes<edm::View<reco::Muon> >(conf.getParameter<edm::InputTag>("muons"));
  vertexToken = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("primaryVertex"));
  pTcut       = conf.getParameter<double>("muonPt");
  SIPcut      = conf.getParameter<double>("muonSIP");
  IPcut       = conf.getParameter<double>("filterIp");
  ratio1cut   = conf.getParameter<double>("filterRatio1");
  ratio2cut   = conf.getParameter<double>("filterRatio2");
  useFilter   = conf.getParameter<bool>("filterPromptMuons");
  produces<reco::CandSoftLeptonTagInfoCollection>();
}

SoftPFMuonTagInfoProducer::~SoftPFMuonTagInfoProducer() {}

void SoftPFMuonTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Declare produced collection
  std::auto_ptr<reco::CandSoftLeptonTagInfoCollection> theMuonTagInfo(new reco::CandSoftLeptonTagInfoCollection);
  
  // Declare and open Jet collection
  edm::Handle<edm::View<reco::Jet> > theJetCollection;
  iEvent.getByToken(jetToken, theJetCollection);
  
  // Declare Muon collection
  edm::Handle<edm::View<reco::Muon> > theMuonCollection;
  iEvent.getByToken(muonToken, theMuonCollection);
  
  // Declare and open Vertex collection
  edm::Handle<reco::VertexCollection> theVertexCollection;
  iEvent.getByToken(vertexToken, theVertexCollection);
  if(!theVertexCollection.isValid() || theVertexCollection->empty()) return;
  const reco::Vertex* vertex=&theVertexCollection->front();
  
  // Biult TransientTrackBuilder
  edm::ESHandle<TransientTrackBuilder> theTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theTrackBuilder);
  const TransientTrackBuilder* transientTrackBuilder=theTrackBuilder.product();
  
  // Loop on jets
  for(unsigned int ij=0, nj=theJetCollection->size(); ij<nj; ij++) {
    edm::RefToBase<reco::Jet> jetRef = theJetCollection->refAt(ij);
    // Build TagInfo object
    reco::CandSoftLeptonTagInfo tagInfo;
    tagInfo.setJetRef(jetRef);
    // Loop on jet daughters
    for(unsigned int id=0, nd=jetRef->numberOfDaughters(); id<nd; ++id) {
      edm::Ptr<reco::Candidate> lepPtr = jetRef->daughterPtr(id);
      if(std::abs(lepPtr->pdgId())!=13) continue;
      
      const reco::Muon* muon(NULL);
      // Step 1: try to access the muon from reco::PFCandidate
      const reco::PFCandidate* pfcand=dynamic_cast<const reco::PFCandidate*>(lepPtr.get());
      if(pfcand) {
        muon=pfcand->muonRef().get();
      }
      // If not PFCandidate is available, find a match looping on the muon collection
      else {
        for(unsigned int im=0, nm=theMuonCollection->size(); im<nm; ++im) { // --- Begin loop on muons
          const reco::Muon* recomuon=&theMuonCollection->at(im);
          const pat::Muon* patmuon=dynamic_cast<const pat::Muon*>(recomuon);
          // Step 2: try a match between reco::Candidate
          if(patmuon) {
            if(patmuon->originalObjectRef()==lepPtr) {
              muon=theMuonCollection->refAt(im).get();
              break;
            }
          }
          // Step 3: try a match with dR and dpT if pat::Muon casting fails
          else {
            if(reco::deltaR(*recomuon, *lepPtr)<0.01 && std::abs(recomuon->pt()-lepPtr->pt())/lepPtr->pt()<0.1) {
              muon=theMuonCollection->refAt(im).get();
              break;
            }
          }
        } // --- End loop on muons
      }
      if(!muon || !muon::isLooseMuon(*muon) || muon->pt()<pTcut) continue;
      reco::TrackRef trkRef( muon->innerTrack() );
      reco::TrackBaseRef trkBaseRef( trkRef );
      // Build Transient Track
      reco::TransientTrack transientTrack=transientTrackBuilder->build(trkRef);
      // Define jet and muon vectors
      reco::Candidate::Vector jetvect(jetRef->p4().Vect()), muonvect(muon->p4().Vect());
      // Calculate variables
      reco::SoftLeptonProperties properties;
      properties.sip2d    = IPTools::signedTransverseImpactParameter(transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), *vertex).second.significance();
      properties.sip3d    = IPTools::signedImpactParameter3D(transientTrack, GlobalVector(jetRef->px(), jetRef->py(), jetRef->pz()), *vertex).second.significance();
      properties.deltaR   = reco::deltaR(*jetRef, *muon);
      properties.ptRel    = ( (jetvect-muonvect).Cross(muonvect) ).R() / jetvect.R(); // | (Pj-Pu) X Pu | / | Pj |
      float mag = muonvect.R()*jetvect.R();
      float dot = muon->p4().Dot(jetRef->p4());
      properties.etaRel   = -log((mag - dot)/(mag + dot)) / 2.;
      properties.ratio    = muon->pt() / jetRef->pt();
      properties.ratioRel = muon->p4().Dot(jetRef->p4()) / jetvect.Mag2();
      properties.p0Par    = boostedPPar(muon->momentum(), jetRef->momentum());
      
      if(std::abs(properties.sip3d)>SIPcut) continue;
      
      // Filter leptons from W, Z decays
      if(useFilter && ((std::abs(properties.sip3d)<IPcut && properties.ratio>ratio1cut) || properties.ratio>ratio2cut)) continue;
      
      // Insert lepton properties
      tagInfo.insert(lepPtr, properties);
      
    } // --- End loop on daughters
    
    // Fill the TagInfo collection
    theMuonTagInfo->push_back(tagInfo);
  } // --- End loop on jets
  
  // Put the TagInfo collection in the event
  iEvent.put(theMuonTagInfo);
}


// compute the lepton momentum along the jet axis, in the jet rest frame
float SoftPFMuonTagInfoProducer::boostedPPar(const math::XYZVector& vector, const math::XYZVector& axis) {
  static const double lepton_mass = 0.00;       // assume a massless (ultrarelativistic) lepton
  static const double jet_mass    = 5.279;      // use B±/B0 mass as the jet rest mass [PDG 2007 updates]
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > lepton(vector.Dot(axis) / axis.r(), ROOT::Math::VectorUtil::Perp(vector, axis), 0., lepton_mass);
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > jet( axis.r(), 0., 0., jet_mass );
  ROOT::Math::BoostX boost( -jet.Beta() );
  return boost(lepton).x();
}

