// -*- C++ -*-
//
// Package:    HLTMuTree
// Class:      HLTMuTree
//
/**\class HLTMuTree HLTMuTree.cc UserCode/HLTMuTree/src/HLTMuTree.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Mihee Jo,588 R-012,+41227673278,
//         Created:  Thu Jul  7 11:47:28 CEST 2011
// $Id: HLTMuTree.cc,v 1.7 2013/02/15 08:56:33 azsigmon Exp $
//
//

#include "HeavyIonsAnalysis/MuonAnalysis/interface/HLTMuTree.h"
#include <iostream>
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace HepMC;

//
// constructors and destructor
//
HLTMuTree::HLTMuTree(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  tagRecoMu = iConfig.getParameter<edm::InputTag>("muons");
  tagVtx = iConfig.getParameter<edm::InputTag>("vertices");
  doReco = iConfig.getUntrackedParameter<bool>("doReco");
  doGen = iConfig.getUntrackedParameter<bool>("doGen");
  tagGenPtl = iConfig.getParameter<edm::InputTag>("genparticle");
  tagSimTrk = iConfig.getParameter<edm::InputTag>("simtrack");
  // tagCompVtx = iConfig.getParameter<edm::InputTag>("
  //higherPuritySelection_(iConfig.getParameter<std::string>("higherPuritySelection")),
  //lowerPuritySelection_(iConfig.getParameter<std::string>("lowerPuritySelection")),

}


HLTMuTree::~HLTMuTree()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTMuTree::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  //Initialization
  GenMu.nptl = GlbMu.nptl = StaMu.nptl = DiMu.npair = 0;
  for (int i=0; i<nmax; i++) {
    GenMu.pid[i] = 10;
    GenMu.status[i] = 0;
    GenMu.mom[i] = 10;
    GenMu.pt[i] = 0;
    GenMu.p[i] = 0;
    GenMu.eta[i] = 0;
    GenMu.phi[i] = 0;
    GlbMu.charge[i] = 0;
    GlbMu.pt[i] = 0;
    GlbMu.p[i] = 0;
    GlbMu.eta[i] = 0;
    GlbMu.phi[i] = 0;
    GlbMu.dxy[i] = 0;
    GlbMu.dz[i] = 0;
    GlbMu.nValMuHits[i] = 0;
    GlbMu.nValTrkHits[i] = 0;
    GlbMu.nTrkFound[i] = 0;
    GlbMu.glbChi2_ndof[i] = 0;
    GlbMu.trkChi2_ndof[i] = 0;
    GlbMu.pixLayerWMeas[i] = 0;
    GlbMu.trkDxy[i] = 0;
    GlbMu.trkDz[i] = 0;
    StaMu.charge[i] = 0;
    StaMu.pt[i] = 0;
    StaMu.p[i] = 0;
    StaMu.eta[i] = 0;
    StaMu.phi[i] = 0;
    StaMu.dxy[i] = 0;
    StaMu.dz[i] = 0;
    GlbMu.isArbitrated[i] = -1;
    DiMu.vProb[i] = -1;
    DiMu.mass[i] = -1;
    DiMu.pt[i] = -1;
    DiMu.pt1[i] = 0;
    DiMu.pt2[i] = 0;
    DiMu.eta1[i] = 0;
    DiMu.eta2[i] = 0;
    DiMu.phi1[i] = 0;
    DiMu.phi2[i] = 0;
    DiMu.charge1[i] = 0;
    DiMu.charge2[i] = 0;
    DiMu.isArb1[i] = -1;
    DiMu.isArb2[i] = -1;
    DiMu.nTrkHit1[i] = 0;
    DiMu.nTrkHit2[i] = 0;
    DiMu.trkChi2_1[i] = 0;
    DiMu.trkChi2_2[i] = 0;
    DiMu.glbChi2_1[i] = 0;
    DiMu.glbChi2_2[i] = 0;
    DiMu.dxy1[i] = 0;
    DiMu.dxy2[i] = 0;
    DiMu.dz1[i] = 0;
    DiMu.dz2[i] = 0;
  }

  //Get run, event, centrality
  event = iEvent.id().event();
  run = iEvent.id().run();
  lumi = iEvent.id().luminosityBlock();

  //Loop over GenParticles, g4SimHits
  if (doGen) {
    int nGen = 0;

    edm::Handle<reco::GenParticleCollection> genColl;
    iEvent.getByLabel(tagGenPtl,genColl);
    if (genColl.isValid()) {
      for (reco::GenParticleCollection::size_type i=0; i+1<genColl.product()->size(); i++) {
        const GenParticleRef genPtl(genColl,i);
        if (abs(genPtl->pdgId()) == 13 && genPtl->status() == 1) {
          GenMu.pt[nGen] = genPtl->pt();
          GenMu.p[nGen] = genPtl->p();
          GenMu.eta[nGen] = genPtl->eta();
          GenMu.phi[nGen] = genPtl->phi();
          GenMu.status[nGen] = genPtl->status();
          GenMu.pid[nGen] = genPtl->pdgId();

          GenMu.mom[nGen] = 10;
          if (genPtl->numberOfMothers() > 0 ) {
            vector<int> momid;
            vector<int>::iterator it_jpsi, it_ups;
            for (unsigned int mom = 0; mom < genPtl->numberOfMothers(); mom++) {
              //cout << "mom pid: " << genPtl->mother(mom)->pdgId() << endl;
              momid.push_back(genPtl->mother(mom)->pdgId());
            }

            if (!momid.empty()) {
              it_jpsi = find(momid.begin(),momid.end(),443);
              it_ups = find(momid.begin(),momid.end(),553);
              if (it_jpsi != momid.end()) GenMu.mom[nGen] = 443;
              if (it_ups != momid.end()) GenMu.mom[nGen] = 553;

              //No J/psi, Y mother -> Should check grandmother
              if (it_jpsi == momid.end() && it_ups == momid.end()) {
                const Candidate *mother = genPtl->mother(0);
                momid.clear();
                for (unsigned int mom = 0; mom < mother->numberOfMothers(); mom++) {
                  //cout << "grand mom pid: " << mother->mother(mom)->pdgId() << endl;
                  momid.push_back(mother->mother(mom)->pdgId());
                }

                if (!momid.empty()) {
                  it_jpsi = find(momid.begin(),momid.end(),443);
                  it_ups = find(momid.begin(),momid.end(),553);
                  if (it_jpsi != momid.end()) GenMu.mom[nGen] = 443;
                  if (it_ups != momid.end()) GenMu.mom[nGen] = 553;
                  if (it_jpsi == momid.end() && it_ups == momid.end()) GenMu.mom[nGen] = momid[0];
                }
              } //End of no J/psi, Y mother -> Should check grandmother
            }

          }
          nGen++;

/*          if (genPtl->numberOfMothers() > 0 ) {
            GenMu.mom[nGen] = genPtl->mother(0)->pdgId();
            cout << "mom pid: " << genPtl->mother(0)->pdgId() << endl;
	    } else {
            GenMu.mom[nGen] = 10;
	    }*/

        }
      }
    } //End of gen collection

/*
    edm::Handle<TrackingParticleCollection> simColl;
    iEvent.getByLabel(tagSimTrk,simColl);
    if (simColl.isValid()) {
    for (TrackingParticleCollection::size_type i=0; i+1<simColl.product()->size(); i++) {
    const TrackingParticleRef simTrk(simColl,i);
    if (simTrk.isNull()) continue;
    if (abs(simTrk->pdgId()) == 13 && simTrk->status() == -99) {

    GenMu.pid[nGen] = simTrk->pdgId();
    GenMu.mom[nGen] = 10;
    for (TrackingParticle::genp_iterator it=simTrk->genParticle_begin();
    it!=simTrk->genParticle_end(); ++it) {
    if ((*it)->status() == 1) GenMu.mom[nGen] = (*it)->pdg_id();
    cout << "sim track mom pid: " << (*it)->pdg_id() <<"\t"  << (*it)->status() << endl;
    }

    GenMu.pt[nGen] = simTrk->pt();
    GenMu.p[nGen] = simTrk->p();
    GenMu.eta[nGen] = simTrk->eta();
    GenMu.phi[nGen] = simTrk->phi();
    GenMu.status[nGen] = simTrk->status();
    nGen++;
    }
    }
    } //End of sim tracks
*/
    GenMu.nptl = nGen;
    //cout << "gen_nptl: " << GenMu.nptl << endl;
    if (nGen >= nmax) {
      cout << "Gen muons in a event exceeded maximum. \n";
      return ;
    }

  } //End of doGen

  //Loop over reco::muon
  if (doReco) {
    //Get vertex position
    edm::Handle< vector<reco::Vertex> > vertex;
    iEvent.getByLabel(tagVtx,vertex);
    if(vertex->size() > 0){
      vx = vertex->begin()->x();
      vy = vertex->begin()->y();
      vz = vertex->begin()->z();
    } else {
      vx = -1;
      vy = -1;
      vz = -1;
    }

    // edm::Handle <reco::VertexCompositeCandidate> compvertex;
    // iEvent.getByLabel(

    edm::Handle< edm::View<reco::Muon> > muons;
    iEvent.getByLabel(tagRecoMu,muons);

    int nGlb = 0;
    int nSta = 0;

    for (unsigned int i=0; i<muons->size(); i++) {
      edm::RefToBase<reco::Muon> muCand(muons,i);
      if (muCand.isNull()) continue;
      if (muCand->globalTrack().isNonnull() && muCand->innerTrack().isNonnull()) {
        if (muCand->isGlobalMuon() && muCand->isTrackerMuon() && fabs(muCand->combinedMuon()->eta()) < 2.4) {
          edm::RefToBase<reco::Track> trk = edm::RefToBase<reco::Track>(muCand->innerTrack());
          edm::RefToBase<reco::Track> glb = edm::RefToBase<reco::Track>(muCand->combinedMuon());
          const reco::HitPattern& p = trk->hitPattern();

          GlbMu.nValMuHits[nGlb] = muCand->combinedMuon().get()->hitPattern().numberOfValidMuonHits();
          GlbMu.nValTrkHits[nGlb] = muCand->innerTrack().get()->hitPattern().numberOfValidTrackerHits();

          GlbMu.nTrkFound[nGlb] = trk->found();
          GlbMu.glbChi2_ndof[nGlb] = glb->chi2()/glb->ndof();
          GlbMu.trkChi2_ndof[nGlb] = trk->chi2()/trk->ndof();
          GlbMu.pixLayerWMeas[nGlb] = p.pixelLayersWithMeasurement();
          GlbMu.trkDxy[nGlb] = fabs(trk->dxy(vertex->begin()->position()));
          GlbMu.trkDz[nGlb] = fabs(trk->dz(vertex->begin()->position()));

          muon::SelectionType st = muon::selectionTypeFromString("TrackerMuonArbitrated");
          GlbMu.isArbitrated[nGlb] = muon::isGoodMuon(*muCand.get(), st);

          GlbMu.charge[nGlb] = glb->charge();
          GlbMu.pt[nGlb] = glb->pt();
          GlbMu.p[nGlb] = glb->p();
          GlbMu.eta[nGlb] = glb->eta();
          GlbMu.phi[nGlb] = glb->phi();
          GlbMu.dxy[nGlb] = glb->dxy(vertex->begin()->position());
          GlbMu.dz[nGlb] = glb->dz(vertex->begin()->position());

          GlbMu.trkLayerWMeas[nGlb] = muCand->globalTrack()->hitPattern().trackerLayersWithMeasurement();
          GlbMu.nValPixHits[nGlb] = p.numberOfValidPixelHits();
          GlbMu.nMatchedStations[nGlb] = muCand->numberOfMatchedStations();

          //cout<<nGlb<<" Glb muon pt  " << GlbMu.pt[nGlb]<<endl;
          nGlb++;
        }

      }
      if (muCand->isStandAloneMuon() && muCand->outerTrack().isNonnull()) {
        if (muCand->standAloneMuon().get()->hitPattern().numberOfValidMuonHits()>0 && fabs(muCand->standAloneMuon()->eta())<2.4) {
          edm::RefToBase<reco::Track> sta = edm::RefToBase<reco::Track>(muCand->standAloneMuon());
          StaMu.charge[nSta] = sta->charge();
          StaMu.pt[nSta] = sta->pt();
          StaMu.p[nSta] = sta->p();
          StaMu.eta[nSta] = sta->eta();
          StaMu.phi[nSta] = sta->phi();
          StaMu.dxy[nSta] = sta->dxy(vertex->begin()->position());
          StaMu.dz[nSta] = sta->dz(vertex->begin()->position());
          nSta++;
        }
      }
      if (nGlb >= nmax) {
        cout << "Global muons in a event exceeded maximum. \n";
        return ;
      }
      if (nSta >= nmax) {
        cout << "Standalone muons in a event exceeded maximum. \n";
        return ;
      }
    }
    GlbMu.nptl = nGlb;
    StaMu.nptl = nSta;


    //vertex probability cuts
    edm::Handle< edm::View<reco::Muon> > muons2;
    iEvent.getByLabel(tagRecoMu,muons2);

    edm::ESHandle<TransientTrackBuilder> theTTBuilder;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTBuilder);
    KalmanVertexFitter vtxFitter;

    int nDiMu = 0;

    for(unsigned int i=0; i<muons->size(); i++){
      edm::RefToBase<reco::Muon> muCand(muons,i);
      if (muCand.isNull()) continue;
      if (muCand->globalTrack().isNonnull() && muCand->innerTrack().isNonnull()) {
	if (muCand->isGlobalMuon() && muCand->isTrackerMuon() && fabs(muCand->combinedMuon()->eta()) < 2.4) {
	  for (unsigned int j=i+1; j<muons->size(); j++){
	    edm::RefToBase<reco::Muon> muCand2(muons,j);
	    if (muCand2.isNull()) continue;
	    if (muCand2->globalTrack().isNonnull() && muCand2->innerTrack().isNonnull()) {
	      if (muCand2->isGlobalMuon() && muCand2->isTrackerMuon() && fabs(muCand2->combinedMuon()->eta()) < 2.4) {
                vector<TransientTrack> t_tks;
                t_tks.push_back(theTTBuilder->build(*muCand->track()));  // pass the reco::Track, not  the reco::TrackRef (which can be transient)
                t_tks.push_back(theTTBuilder->build(*muCand2->track())); // otherwise the vertex will have transient refs inside.
                TransientVertex myVertex = vtxFitter.vertex(t_tks);
                if (myVertex.isValid()) {
                  edm::RefToBase<reco::Track> trk = edm::RefToBase<reco::Track>(muCand->innerTrack());
                  edm::RefToBase<reco::Track> glb = edm::RefToBase<reco::Track>(muCand->combinedMuon());
		  edm::RefToBase<reco::Track> trk2 = edm::RefToBase<reco::Track>(muCand2->innerTrack());
                  edm::RefToBase<reco::Track> glb2 = edm::RefToBase<reco::Track>(muCand2->combinedMuon());
                  float vChi2 = myVertex.totalChiSquared();
		  float vNDF  = myVertex.degreesOfFreedom();
		  float vProb(TMath::Prob(vChi2,(int)vNDF));
		  DiMu.vProb[nDiMu] = vProb;

                  DiMu.glbChi2_1[nDiMu] = glb->chi2()/glb->ndof();
                  DiMu.trkChi2_1[nDiMu] = trk->chi2()/trk->ndof();
                  DiMu.glbChi2_2[nDiMu] = glb2->chi2()/glb2->ndof();
                  DiMu.trkChi2_2[nDiMu] = trk2->chi2()/trk2->ndof();
                  const math::XYZTLorentzVector ZRecoGlb (muCand->px()+muCand2->px(), muCand->py()+muCand2->py() , muCand->pz()+muCand2->pz(), muCand->p()+muCand2->p());
                  DiMu.mass[nDiMu] = ZRecoGlb.mass();
                  DiMu.e[nDiMu] = ZRecoGlb.e();
                  DiMu.pt[nDiMu] = ZRecoGlb.pt();
                  DiMu.eta[nDiMu] = ZRecoGlb.eta();
                  DiMu.phi[nDiMu] = ZRecoGlb.phi();
		  DiMu.rapidity[nDiMu] = ZRecoGlb.Rapidity();

                  DiMu.pt1[nDiMu] = glb->pt();
                  DiMu.eta1[nDiMu] = glb->eta();
                  DiMu.phi1[nDiMu] = glb->phi();
                  DiMu.dxy1[nDiMu] = glb->dxy(vertex->begin()->position());
                  DiMu.dz1[nDiMu] = glb->dz(vertex->begin()->position());
                  DiMu.charge1[nDiMu] = glb->charge();
                  DiMu.pt2[nDiMu] = glb2->pt();
                  DiMu.eta2[nDiMu] = glb2->eta();
                  DiMu.phi2[nDiMu] = glb2->phi();
                  DiMu.dxy2[nDiMu] = glb2->dxy(vertex->begin()->position());
                  DiMu.dz2[nDiMu] = glb2->dz(vertex->begin()->position());
		  DiMu.charge2[nDiMu] = glb2->charge();
		  DiMu.charge[nDiMu] = glb->charge() + glb2->charge();

		  DiMu.nTrkHit1[nDiMu] = trk->hitPattern().numberOfValidTrackerHits();
		  DiMu.nTrkHit2[nDiMu] = trk2->hitPattern().numberOfValidTrackerHits();
		  DiMu.nMuHit1[nDiMu] = glb->hitPattern().numberOfValidMuonHits();
		  DiMu.nMuHit2[nDiMu] = glb2->hitPattern().numberOfValidMuonHits();
	          DiMu.nTrkLayers1[nDiMu] = glb->hitPattern().trackerLayersWithMeasurement();
	          DiMu.nTrkLayers2[nDiMu] = glb2->hitPattern().trackerLayersWithMeasurement();
        	  DiMu.nPixHit1[nDiMu] = trk->hitPattern().numberOfValidPixelHits();
        	  DiMu.nPixHit2[nDiMu] = trk2->hitPattern().numberOfValidPixelHits();
        	  DiMu.nMatchedStations1[nDiMu] = muCand->numberOfMatchedStations();
        	  DiMu.nMatchedStations2[nDiMu] = muCand2->numberOfMatchedStations();

                  muon::SelectionType st = muon::selectionTypeFromString("TrackerMuonArbitrated");
                  DiMu.isArb1[nDiMu] = muon::isGoodMuon(*muCand.get(), st);
                  muon::SelectionType st2 = muon::selectionTypeFromString("TrackerMuonArbitrated");
                  DiMu.isArb2[nDiMu] = muon::isGoodMuon(*muCand2.get(), st2);

                  //cout<<nDiMu<<" first muon pt  " << DiMu.pt1[nDiMu]<<" second muon pt  " << DiMu.pt2[nDiMu] << endl;

		  nDiMu++;
		  //cout << nDiMu << endl;
		}
	      }
            }
          }
        }
      }
    }
    DiMu.npair = nDiMu;
  } // End of doReco
  else {
    vx = -1;
    vy = -1;
    vz = -1;
  }

  // Fill a muon tree
  // if (DiMu.npair>0){
  treeMu->Fill();
  // }
}


// ------------ method called once each job just before starting event loop  ------------
void
HLTMuTree::beginJob()
{
  treeMu = foutput->make<TTree>("HLTMuTree","HLTMuTree");
  treeMu->Branch("Run",&run,"run/I");
  treeMu->Branch("Event",&event,"event/I");
  treeMu->Branch("Lumi",&lumi,"lumi/I");
  treeMu->Branch("vx",&vx,"vx/F");
  treeMu->Branch("vy",&vy,"vy/F");
  treeMu->Branch("vz",&vz,"vz/F");

  treeMu->Branch("Gen_nptl",&GenMu.nptl,"Gen_nptl/I");
  treeMu->Branch("Gen_pid",GenMu.pid,"Gen_pid[Gen_nptl]/I");
  treeMu->Branch("Gen_mom",GenMu.mom,"Gen_mom[Gen_nptl]/I");
  treeMu->Branch("Gen_status",GenMu.status,"Gen_status[Gen_nptl]/I");
  treeMu->Branch("Gen_p",GenMu.p,"Gen_p[Gen_nptl]/F");
  treeMu->Branch("Gen_pt",GenMu.pt,"Gen_pt[Gen_nptl]/F");
  treeMu->Branch("Gen_eta",GenMu.eta,"Gen_eta[Gen_nptl]/F");
  treeMu->Branch("Gen_phi",GenMu.phi,"Gen_phi[Gen_nptl]/F");

  treeMu->Branch("Glb_nptl",&GlbMu.nptl,"Glb_nptl/I");
  treeMu->Branch("Glb_charge",GlbMu.charge,"Glb_charge[Glb_nptl]/I");
  treeMu->Branch("Glb_p",GlbMu.p,"Glb_p[Glb_nptl]/F");
  treeMu->Branch("Glb_pt",GlbMu.pt,"Glb_pt[Glb_nptl]/F");
  treeMu->Branch("Glb_eta",GlbMu.eta,"Glb_eta[Glb_nptl]/F");
  treeMu->Branch("Glb_phi",GlbMu.phi,"Glb_phi[Glb_nptl]/F");
  treeMu->Branch("Glb_dxy",GlbMu.dxy,"Glb_dx[Glb_nptl]/F");
  treeMu->Branch("Glb_dz",GlbMu.dz,"Glb_dz[Glb_nptl]/F");

  treeMu->Branch("Glb_nValMuHits",GlbMu.nValMuHits,"Glb_nValMuHits[Glb_nptl]/I");
  treeMu->Branch("Glb_nValTrkHits",GlbMu.nValTrkHits,"Glb_nValTrkHits[Glb_nptl]/I");
  treeMu->Branch("Glb_nValPixHits",GlbMu.nValPixHits,"Glb_nValPixHits[Glb_nptl]/I");
  treeMu->Branch("Glb_trkLayerWMeas",GlbMu.trkLayerWMeas,"Glb_trkLayerWMeas[Glb_nptl]/I");
  treeMu->Branch("Glb_nMatchedStations",GlbMu.nMatchedStations,"Glb_nMatchedStations[Glb_nptl]/I");
  treeMu->Branch("Glb_nTrkFound",GlbMu.nTrkFound,"Glb_nTrkFound[Glb_nptl]/I");
  treeMu->Branch("Glb_glbChi2_ndof",GlbMu.glbChi2_ndof,"Glb_glbChi2_ndof[Glb_nptl]/F");
  treeMu->Branch("Glb_trkChi2_ndof",GlbMu.trkChi2_ndof,"Glb_trkChi2_ndof[Glb_nptl]/F");
  treeMu->Branch("Glb_pixLayerWMeas",GlbMu.pixLayerWMeas,"Glb_pixLayerWMeas[Glb_nptl]/I");
  treeMu->Branch("Glb_trkDxy",GlbMu.trkDxy,"Glb_trkDxy[Glb_nptl]/F");
  treeMu->Branch("Glb_trkDz",GlbMu.trkDz,"Glb_trkDz[Glb_nptl]/F");

  treeMu->Branch("Sta_nptl",&StaMu.nptl,"Sta_nptl/I");
  treeMu->Branch("Sta_charge",StaMu.charge,"Sta_charge[Sta_nptl]/I");

  treeMu->Branch("Sta_p",StaMu.p,"Sta_p[Sta_nptl]/F");
  treeMu->Branch("Sta_pt",StaMu.pt,"Sta_pt[Sta_nptl]/F");
  treeMu->Branch("Sta_eta",StaMu.eta,"Sta_eta[Sta_nptl]/F");
  treeMu->Branch("Sta_phi",StaMu.phi,"Sta_phi[Sta_nptl]/F");
  treeMu->Branch("Sta_dxy",StaMu.dxy,"Sta_dx[Sta_nptl]/F");
  treeMu->Branch("Sta_dz",StaMu.dz,"Sta_dz[Sta_nptl]/F");

  treeMu->Branch("Glb_isArbitrated",GlbMu.isArbitrated,"Glb_isArbitrated[Glb_nptl]/I");

  treeMu->Branch("Di_npair",&DiMu.npair,"Di_npair/I");
  treeMu->Branch("Di_vProb",DiMu.vProb,"Di_vProb[Di_npair]/F");
  treeMu->Branch("Di_mass",DiMu.mass,"Di_mass[Di_npair]/F");
  treeMu->Branch("Di_e",DiMu.e,"Di_e[Di_npair]/F");
  treeMu->Branch("Di_pt",DiMu.pt,"Di_pt[Di_npair]/F");
  treeMu->Branch("Di_pt1",DiMu.pt1,"Di_pt1[Di_npair]/F");
  treeMu->Branch("Di_pt2",DiMu.pt2,"Di_pt2[Di_npair]/F");
  treeMu->Branch("Di_eta",DiMu.eta,"Di_eta[Di_npair]/F");
  treeMu->Branch("Di_eta1",DiMu.eta1,"Di_eta1[Di_npair]/F");
  treeMu->Branch("Di_eta2",DiMu.eta2,"Di_eta2[Di_npair]/F");
  treeMu->Branch("Di_rapidity",DiMu.rapidity,"Di_rapidity[Di_npair]/F");
  treeMu->Branch("Di_phi",DiMu.phi,"Di_phi[Di_npair]/F");
  treeMu->Branch("Di_phi1",DiMu.phi1,"Di_phi1[Di_npair]/F");
  treeMu->Branch("Di_phi2",DiMu.phi2,"Di_phi2[Di_npair]/F");
  treeMu->Branch("Di_charge",DiMu.charge,"Di_charge[Di_npair]/I");
  treeMu->Branch("Di_charge1",DiMu.charge1,"Di_charge1[Di_npair]/I");
  treeMu->Branch("Di_charge2",DiMu.charge2,"Di_charge2[Di_npair]/I");
  treeMu->Branch("Di_isArb1",DiMu.isArb1,"Di_isArb1[Di_npair]/I");
  treeMu->Branch("Di_isArb2",DiMu.isArb2,"Di_isArb2[Di_npair]/I");
  treeMu->Branch("Di_nTrkHit1",DiMu.nTrkHit1,"Di_nTrkHit1[Di_npair]/I");
  treeMu->Branch("Di_nTrkHit2",DiMu.nTrkHit2,"Di_nTrkHit2[Di_npair]/I");
  treeMu->Branch("Di_nMuHit1",DiMu.nMuHit1,"Di_nMuHit1[Di_npair]/I");
  treeMu->Branch("Di_nMuHit2",DiMu.nMuHit2,"Di_nMuHit2[Di_npair]/I");
  treeMu->Branch("Di_nTrkLayers1",DiMu.nTrkLayers1,"Di_nTrkLayers1[Di_npair]/I");
  treeMu->Branch("Di_nTrkLayers2",DiMu.nTrkLayers2,"Di_nTrkLayers2[Di_npair]/I");
  treeMu->Branch("Di_nPixHit1",DiMu.nPixHit1,"Di_nPixHit1[Di_npair]/I");
  treeMu->Branch("Di_nPixHit2",DiMu.nPixHit2,"Di_nPixHit2[Di_npair]/I");
  treeMu->Branch("Di_nMatchedStations1",DiMu.nMatchedStations1,"Di_nMatchedStations1[Di_npair]/I");
  treeMu->Branch("Di_nMatchedStations2",DiMu.nMatchedStations2,"Di_nMatchedStations2[Di_npair]/I");
  treeMu->Branch("Di_trkChi2_1",DiMu.trkChi2_1,"Di_trkChi2_1[Di_npair]/F");
  treeMu->Branch("Di_trkChi2_2",DiMu.trkChi2_2,"Di_trkChi2_2[Di_npair]/F");
  treeMu->Branch("Di_glbChi2_1",DiMu.glbChi2_1,"Di_glbChi2_1[Di_npair]/F");
  treeMu->Branch("Di_glbChi2_2",DiMu.glbChi2_2,"Di_glbChi2_2[Di_npair]/F");
  treeMu->Branch("Di_dxy1",DiMu.dxy1,"Di_dxy1[Di_npair]/F");
  treeMu->Branch("Di_dxy2",DiMu.dxy2,"Di_dxy2[Di_npair]/F");
  treeMu->Branch("Di_dz1",DiMu.dz1,"Di_dz1[Di_npair]/F");
  treeMu->Branch("Di_dz2",DiMu.dz2,"Di_dz2[Di_npair]/F");
}

// ------------ method called once each job just after ending the event loop  ------------
void
HLTMuTree::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTMuTree);
