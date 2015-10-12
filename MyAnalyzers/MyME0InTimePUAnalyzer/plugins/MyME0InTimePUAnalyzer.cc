// -*- C++ -*-
//
// Package:    MyME0InTimePUAnalyzer
// Class:      MyME0InTimePUAnalyzer
// 
/**\class MyME0InTimePUAnalyzer MyME0InTimePUAnalyzer.cc MyAnalyzers/MyME0InTimePUAnalyzer/plugins/MyME0InTimePUAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Piet Verwilligen
//         Created:  Wed, 07 Oct 2015 08:29:01 GMT
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// root include files
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/ME0Muon.h"
#include "DataFormats/MuonReco/interface/ME0MuonCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0Segment.h" 
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "RecoMuon/MuonIdentification/interface/ME0MuonSelector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


//
// class declaration
//

class MyME0InTimePUAnalyzer : public edm::EDAnalyzer {
   public:
      explicit MyME0InTimePUAnalyzer(const edm::ParameterSet&);
      ~MyME0InTimePUAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:

      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  // virtual void beginJob() override;
  // virtual void endJob() override;

  bool checkVector(std::vector<int>&, int);

  // Output Files / TFile Service
  edm::Service<TFileService> fs;
  // std::string rootFileName;
  // std::unique_ptr<TFile> outputfile;

  // Info Bool
  bool printInfoHepMC, printInfoSignal, printInfoPU, printInfoAll, printInfoME0Match, me0genpartfound;

  // For later use in 7XY releases:
  /*
  edm::EDGetTokenT<reco::GenParticleCollection> GENParticle_Token;
  edm::EDGetTokenT<edm::HepMCProduct>           HEPMCCol_Token;
  edm::EDGetTokenT<edm::SimTrackContainer>      SIMTrack_Token;
  edm::EDGetTokenT<CSCSegmentCollection>        CSCSegment_Token;
  edm::EDGetTokenT<GEMSegmentCollection>        GEMSegment_Token;
  */

  edm::ESHandle<ME0Geometry> me0Geom;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

  double preDigiSmearX, preDigiSmearY;
  int nMatchedHits;


};

//
// constants, enums and typedefs
//
double me0mineta = 2.00;
double me0maxeta = 3.00;

//
// static data member definitions
//

//
// constructors and destructor
//
MyME0InTimePUAnalyzer::MyME0InTimePUAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed


  // rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  // outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  preDigiSmearX   = iConfig.getUntrackedParameter<double>("preDigiSmearX");
  preDigiSmearY   = iConfig.getUntrackedParameter<double>("preDigiSmearY");
  nMatchedHits    = iConfig.getUntrackedParameter<int>("nMatchedHits");

  printInfoHepMC  = iConfig.getUntrackedParameter<bool>("printInfoHepMC");
  printInfoSignal = iConfig.getUntrackedParameter<bool>("printInfoSignal");
  printInfoPU     = iConfig.getUntrackedParameter<bool>("printInfoPU");
  printInfoAll    = iConfig.getUntrackedParameter<bool>("printInfoAll");
  printInfoME0Match  = iConfig.getUntrackedParameter<bool>("printInfoME0Match");
  // For later use in 7XY releases:
  /*
  GENParticle_Token   = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  HEPMCCol_Token      = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  SIMVertex_Token     = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits")); // or consumes<std::vector<SimVertex>>
  SIMTrack_Token      = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  consumesMany<edm::PSimHitContainer>();
  PrimaryVertex_Token = consumes<std::vector<reco::Vertex>>(edm::InputTag("offlinePrimaryVertices"));
  ME0Segment_Token    = consumes<ME0SegmentCollection>(edm::InputTag("me0Segments"));
  ME0Muon_Token       = consumes<ME0MuonCollection>(edm::InputTag("me0SegmentMatching"));
  */




}


MyME0InTimePUAnalyzer::~MyME0InTimePUAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  // outputfile->cd();
  // outputfile->Close();

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MyME0InTimePUAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get Geometries
  // ----------------------
  iSetup.get<MuonGeometryRecord>().get(me0Geom);
  // ----------------------

  // Access GenParticles
  // ----------------------
  // edm::Handle<reco::GenParticleCollection> genParticles;
  // iEvent.getByLabel("genParticles", genParticles);     // 62X
  // iEvent.getByToken(GENParticle_Token, genParticles);  // 7XY
  edm::Handle<edm::HepMCProduct> hepmcevent;
  iEvent.getByLabel("generator", hepmcevent);             // 62X
  // iEvent.getByToken(HEPMCCol_Token, hepmcevent);       // 7XY
  // -----------------------

  // Access SimVertices
  // -----------------------
  edm::Handle<std::vector<SimVertex>> simVertexCollection;
  iEvent.getByLabel("g4SimHits", simVertexCollection);         // 62X
  // iEvent.getByToken(SIMVertex_Token, simVertexCollection);  // 7XY
  std::vector<SimVertex> theSimVertices; 
  theSimVertices.insert(theSimVertices.end(),simVertexCollection->begin(),simVertexCollection->end()); // more useful than seems at first sight
  // -----------------------

  // Access SimTracks
  // -----------------------
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByLabel("g4SimHits",SimTk);                   // 62X
  // iEvent.getByToken(SIMTrack_Token,SimTk);             // 7XY
  // -----------------------

  // Access SimHits
  // -----------------------
  std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers;
  iEvent.getManyByType(theSimHitContainers);              // 62X & 7XY
  std::vector<PSimHit> theSimHits;
  for (int i = 0; i < int(theSimHitContainers.size()); ++i) {
    theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
  }
  // -----------------------

  // Access Primary Vertices
  // -----------------------
  edm::Handle<std::vector<reco::Vertex>> primaryVertexCollection;
  iEvent.getByLabel("offlinePrimaryVertices", primaryVertexCollection);  // 62X
  // iEvent.getByToken(PrimaryVertex_Token, primaryVertexCollection);    // 7XY
  // -----------------------

  // Access ME0Segments
  // -----------------------
  edm::Handle<ME0SegmentCollection> me0segments;
  iEvent.getByLabel("me0Segments", me0segments);
  // iEvent.getByToken(ME0Segment_Token, me0segments);
  // -----------------------

  // Access ME0Muons
  // -----------------------
  edm::Handle <std::vector<reco::ME0Muon> > me0muons;
  iEvent.getByLabel("me0SegmentMatching", me0muons);
  // iEvent.getByToken(ME0Muon_Token, me0muons);
  std::vector<reco::ME0Muon> theME0Muons;
  theME0Muons.insert(theME0Muons.end(),me0muons->begin(),me0muons->end()); // probably not necessary
  // -----------------------

  me0genpartfound = false;

  // Analysis of SimVertices
  // =======================
  // not sure whether this is useful
  // SimVertices are all vertices used in GEANT ... 
  // so also when a delta-ray is emitted in a muon detector
  double vtx_r = 0.0, vtx_x = 0.0, vtx_y = 0.0, vtx_z = 0.0;
  /*
  for (std::vector<SimVertex>::const_iterator iVertex = theSimVertices.begin(); iVertex != theSimVertices.end(); ++iVertex) {
    SimVertex simvertex = (*iVertex);
    unsigned int simvertexid = simvertex.vertexId();
    vtx_x = simvertex.position().x(); vtx_y = simvertex.position().y(); vtx_r = sqrt(pow(vtx_x,2)+pow(vtx_y,2)); vtx_z = simvertex.position().z();
    if( vtx_r < 2 && fabs(vtx_z) < 25 ) { // constrain area to beam spot: r < 2cm and |z| < 25 cm
      if(printInfo) std::cout<<"SimVertex with id = "<<simvertexid<<" and position (in cm) : [x,y,z] = ["<<vtx_x<<","<<vtx_y<<","<<vtx_z<<"] or [r,z] = ["<<vtx_r<<","<<vtx_z<<"]"<<std::endl;
    }
  }
  */
  // =======================

  // Save Particles in separate collection [heavy]
  // std::vector< std::unique_ptr<HepMC::GenParticle> >   GEN_muons_signal, GEN_muons_bkgnd;
  // std::vector< std::unique_ptr<SimTrack> >             SIM_muons_signal, SIM_muons_bkgnd;   

  // Save index to Particles in a vector
  /*
  std::vector<unsigned int> index_genpart_signal, index_genpart_background; index_genpart_signal_mother;
  std::vector<unsigned int> index_simtrck_signal, index_simtrck_background;
  index_genpart_signal.clear(); index_genpart_background.clear(); index_simtrck_signal.clear(); index_simtrck_background.clear();
  */

  // Analysis of GenParticles, SimTracks, SimVertices and SimHits
  // ====================================================================
  // Strategy: 
  // 1) select the two muons from the Z-decay in the GenParticles collection (using HepMC::GenEvent info) ==> save the index in a vector<int>
  // 2) select the corresponding SimTrack ==> save the index to the trackId in a vector<int> and save the index to the vertexId in a vector<int>
  // 3) vertexId --> SimVertex
  // 4) loop over ME0SimHits and select the SimHits made by the SimTrack
  // 5) loop over ME0Muon --> ME0Segment --> ME0RecHits and match these rechits to simhits above
  // 6) you obtained a ME0Muon matched to the genParticle of the Z-decay! 
  // --------------------------------------------------------------------

  std::vector<int> indmu, trkmu, vtxmu;
  indmu.clear(); trkmu.clear(); vtxmu.clear();
  std::vector< std::vector<const PSimHit*> > simhitmu; 
  std::vector< ME0DetId > me0detidmu;
  std::vector< std::vector< std::pair<int,int> > > me0mu;

  // 1) loop over GenParticle container
  // --------------------------------------------------------------------
  bool skip=false;    
  bool accepted = false;
  bool foundmuons=false;
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(hepmcevent->GetEvent()));
      
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) { 
    if ( !accepted && ( (*p)->pdg_id() == 23 ) && (*p)->status() == 3 ) { 
      accepted=true;
      for( HepMC::GenVertex::particle_iterator aDaughter=(*p)->end_vertex()->particles_begin(HepMC::descendants); aDaughter !=(*p)->end_vertex()->particles_end(HepMC::descendants);aDaughter++){
	if ( abs((*aDaughter)->pdg_id())==13) {
	  foundmuons=true;
	  if ((*aDaughter)->status()!=1 ) {
	    for( HepMC::GenVertex::particle_iterator byaDaughter=(*aDaughter)->end_vertex()->particles_begin(HepMC::descendants); 
		 byaDaughter !=(*aDaughter)->end_vertex()->particles_end(HepMC::descendants);byaDaughter++){
	      if ((*byaDaughter)->status()==1 && abs((*byaDaughter)->pdg_id())==13) {
		bool found = checkVector(indmu,(*byaDaughter)->barcode());           
		if(!found) indmu.push_back((*byaDaughter)->barcode());
		if(printInfoHepMC) std::cout<<"Stable muon from Z with pdgId "<<std::showpos<<(*byaDaughter)->pdg_id()<<" and index "<<(*byaDaughter)->barcode()<<(found?" not":"")<<" added"<<std::endl;
	      }
	    }
	  }
	  else {
	    bool found = checkVector(indmu,(*aDaughter)->barcode());
	    if(!found) indmu.push_back((*aDaughter)->barcode());
	    if(printInfoHepMC) std::cout << "Stable muon from Z with pdgId "<<std::showpos<<(*aDaughter)->pdg_id()<<" and index "<<(*aDaughter)->barcode()<<(found?" not":"")<<" added"<<std::endl;
	  }       
	}           
      }
      if (!foundmuons){
	if(printInfoHepMC) std::cout << "No muons from Z ...skip event" << std::endl;
	skip=true;
      } 
    }
  }
     
  if ( !accepted) {
    if(printInfoHepMC) std::cout << "No Z particles in the event ...skip event" << std::endl;
    skip=true;
  }   
  else {
    skip=false; 
  }
  if(skip) return;

  // Ease debugging ::: run only over events that contain a GenParticle Muon within 2.00 < | eta | < 3.00
  bool forwardMuon = false;
  for(unsigned int i=0; i<indmu.size(); ++i) {
    double genparteta = myGenEvent->barcode_to_particle(indmu.at(i))->momentum().eta();
    if(fabs(genparteta) > 2.00 && fabs(genparteta) < 3.00) forwardMuon = true;
  }
  if(!forwardMuon) return; // stop program here

  if(printInfoAll) {
    for(unsigned int i=0; i<indmu.size(); ++i) {
      std::cout<<"GEN Muon: id = "<<std::showpos<<std::setw(2)<<myGenEvent->barcode_to_particle(indmu.at(i))->pdg_id()<<" | index = "<<indmu.at(i);
      std::cout<<" | eta = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().eta()<<" | phi = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().perp()<<std::endl;
    }
  }
  // --------------------------------------------------------------------
  // pre-define the vectors based on the size of indmu vector
  for(unsigned int i=0; i<indmu.size(); ++i) { 
    trkmu.push_back(-1); vtxmu.push_back(-1);
    me0detidmu.push_back(ME0DetId(1,0,0,0)); // ME0DetId(region layer chamber roll)
    std::vector<const PSimHit*> tmp1; simhitmu.push_back(tmp1);
    std::vector< std::pair<int,int> > tmp2; me0mu.push_back(tmp2); 
  }
  // --------------------------------------------------------------------


  // 2) loop over SimTrack container
  // --------------------------------------------------------------------
  for (edm::SimTrackContainer::const_iterator it = SimTk->begin(); it != SimTk->end(); ++it) {
    std::unique_ptr<SimTrack> simtrack = std::unique_ptr<SimTrack>(new SimTrack(*it));
    if(fabs(simtrack->type()) != 13) continue;
    // match to GenParticles
    if(it->genpartIndex() == indmu[0]) { 
      trkmu[0] = simtrack->trackId();    // !!! starts counting at 1, not at 0 !!!
      vtxmu[0] = simtrack->vertIndex();
    }
    if(it->genpartIndex() == indmu[1]) {
      trkmu[1] = simtrack->trackId();
      vtxmu[1] = simtrack->vertIndex();
    }
    // some printout
    if(it->genpartIndex() == indmu[0] || it->genpartIndex() == indmu[1]) { 
      // int simtrack_trackId = simtrack->trackId();
      // int simtrack_vertxId = simtrack->vertIndex();
      if(printInfoAll) {
	std::cout<<"SIM Muon: id = "<<std::setw(2)<<it->type()<<" | trackId = "<<it->trackId()<<" | vertexId = "<<it->vertIndex()<<" | genpartIndex = "<<it->genpartIndex();
	std::cout<<" | eta = "<<std::setw(9)<<it->momentum().eta()<<" | phi = "<<std::setw(9)<<it->momentum().phi();
	std::cout<<" | pt = "<<std::setw(9)<<it->momentum().pt()<<std::endl;
      }
    }
  }
  // --------------------------------------------------------------------


  // 3) pick up the associated SimVtx
  // --------------------------------------------------------------------
  for (unsigned int i=0; i<vtxmu.size(); ++i) {
    if(vtxmu[i] == -1) continue;
    SimVertex simvertex = theSimVertices.at(vtxmu[i]);
    unsigned int simvertexid = simvertex.vertexId();
    vtx_x = simvertex.position().x(); vtx_y = simvertex.position().y(); vtx_r = sqrt(pow(vtx_x,2)+pow(vtx_y,2)); vtx_z = simvertex.position().z();
    if(printInfoAll) {
      std::cout<<"|--> SimVertex with id = "<<simvertexid<<" and position (in cm) : [x,y,z] = ["<<vtx_x<<","<<vtx_y<<","<<vtx_z<<"] or [r,z] = ["<<vtx_r<<","<<vtx_z<<"]"<<std::endl;
      if(vtx_r < 2 && fabs(vtx_z) < 25) std::cout<<"     ==> is a Primary (or Secondary)Vertex"<<std::endl;
      else                              std::cout<<"     ==> must be a Decay Vertex"<<std::endl;
    }
  }
  // --------------------------------------------------------------------

      
  // 4) then loop over the SimHit Container
  // --------------------------------------------------------------------
  for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); ++iHit) {
    DetId theDetUnitId((*iHit).detUnitId());
    DetId simdetid= DetId((*iHit).detUnitId());
    /*
      int pid            = (*iHit).particleType();
      int process        = (*iHit).processType();
      double time        = (*iHit).timeOfFlight();
      double log_time    = log10((*iHit).timeOfFlight());
      double log_energy  = log10((*iHit).momentumAtEntry().perp()*1000); // MeV
      double log_deposit = log10((*iHit).energyLoss()*1000000);          // keV
    */
    int simhit_trackId = (*iHit).trackId();
    
    if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::ME0){ // Only ME0
      ME0DetId me0id(theDetUnitId);
      const ME0EtaPartition* etapart = me0Geom->etaPartition(me0id);
      GlobalPoint ME0GlobalPoint = etapart->toGlobal((*iHit).localPosition());
      // GlobalPoint ME0GlobalEntry = ME0Surface.toGlobal((*iHit).entryPoint());
      // GlobalPoint ME0GlobalExit  = ME0Surface.toGlobal((*iHit).exitPoint());
      // double ME0GlobalEntryExitDZ = fabs(ME0GlobalEntry.z()-ME0GlobalExit.z());
      // double ME0LocalEntryExitDZ  = fabs((*iHit).entryPoint().z()-(*iHit).exitPoint().z());
      
      // If SimTrack Matches SimHit Mother then save a pointer to the SimHit
      if(simhit_trackId == trkmu[0]) {
	std::vector<const PSimHit*> tmp = simhitmu[0];
	tmp.push_back(&(*iHit)); // or should we use some kind of ->clone());
	simhitmu[0] = tmp;
	me0detidmu[0] = ME0DetId(me0id.region(), 0, me0id.chamber(),0); // ME0hamber id
      }
      if(simhit_trackId == trkmu[1]) {
	std::vector<const PSimHit*> tmp = simhitmu[1];
	tmp.push_back(&(*iHit)); // or should we use some kind of ->clone());
	simhitmu[1] = tmp;
	me0detidmu[1] = ME0DetId(me0id.region(), 0, me0id.chamber(),0); // ME0hamber id
      }   
      if(simhit_trackId == trkmu[0] || simhit_trackId == trkmu[1]) {
	if(printInfoAll) {
	  std::cout<<"ME0 SimHit in "<<std::setw(12)<<(int)me0id<<me0id<<" from simtrack with trackId = "<<std::setw(9)<<(*iHit).trackId();
	  std::cout<<" | time t = "<<std::setw(12)<<(*iHit).timeOfFlight()<<" | phi = "<<std::setw(12)<<ME0GlobalPoint.phi()<<" | eta = "<<std::setw(12)<<ME0GlobalPoint.eta();
	  std::cout<<" | global position = "<<ME0GlobalPoint;
	  std::cout<<""<<std::endl;
	}
      }
    }
  }
  // --------------------------------------------------------------------


  // 5) Loop over ME0Muons and ask for the ME0RecHits of the ME0Segment
  // --------------------------------------------------------------------
  if(printInfoME0Match) {
    std::cout<<" Number of ME0Muons in this event = "<<me0muons->size()<<std::endl;
    std::cout<<" Number of ME0Sgmts in this event = "<<me0segments->size()<<std::endl;
    std::cout<<" =====     Start Matching     ===== "<<std::endl;
  }
  int me0muonpos = -1;
  for(std::vector<reco::ME0Muon>::const_iterator it=me0muons->begin(); it!=me0muons->end(); ++it) {

    ++me0muonpos;

    // 1) Neglect ME0 Muons for which the ME0Segment is not in the same chamber as the Signal SimHits
    ME0DetId       segId = ME0DetId(it->me0segment().geographicalId());
    int matchedGENMu = -1;
    if      (segId.region() == me0detidmu[0].region() && segId.chamber() == me0detidmu[0].chamber()) matchedGENMu = 0; 
    else if (segId.region() == me0detidmu[1].region() && segId.chamber() == me0detidmu[1].chamber()) matchedGENMu = 1;
    else continue;

    // 2) Neglect ME0Muons if quality is not good or innerTrack does not exist
    // if (!muon::isGoodMuon(me0Geom, *it, muon::Tight)) continue;
    // if(!it->innerTrack()) continue;
    reco::TrackRef tkRef = it->innerTrack();
    ME0Segment    segRef = it->me0segment();

    if(printInfoME0Match){
      std::cout<<"ME0Muon in "<<segId<<" with eta = "<<it->eta()<<" phi = "<<it->phi()<<" pt = "<<it->pt()<<std::endl;
      std::cout<<"        InnerTrack :: eta = "<<tkRef->eta()<<" phi = "<<tkRef->phi()<<" pt = "<<tkRef->pt()<<std::endl;
      std::cout<<"           Segment :: eta = "<<(me0Geom->etaPartition(segId)->toGlobal(segRef.localPosition())).eta()
	       <<" phi = "<<(me0Geom->etaPartition(segId)->toGlobal(segRef.localPosition())).phi()
	       <<" dir eta = "<<(me0Geom->etaPartition(segId)->toGlobal(segRef.localDirection())).eta()
	       <<" dir phi = "<<(me0Geom->etaPartition(segId)->toGlobal(segRef.localDirection())).phi()
	       <<" ME0SegRefId = "<<it->me0segid()<<" time = "<<segRef.time()<<" +/- "<<segRef.timeErr()<<" Nhits = "<<segRef.nRecHits()<<" index = "<<me0muonpos<<std::endl;
    }


    // 3) Perform Matching based on global position of SimHits and RecHits
    // Loop first over SimHits => reduce running time
    int NmatchedToSegment = 0;

    for(unsigned int j=0; j<simhitmu[matchedGENMu].size(); ++j) {
      ME0DetId simhitME0id(simhitmu[matchedGENMu][j]->detUnitId());

      if(printInfoME0Match){      
	std::cout<<"     === ME0 SimHit in "<<std::setw(12)<<simhitME0id.rawId()<<" = "<<simhitME0id<<" from simtrack with trackId = ";
	std::cout<<std::setw(9)<<simhitmu[matchedGENMu][j]->trackId()<<" | time t = "<<std::setw(12)<<simhitmu[matchedGENMu][j]->timeOfFlight();
	std::cout<<" | phi = "<<std::setw(12)<<((me0Geom->etaPartition(simhitME0id))->toGlobal(simhitmu[matchedGENMu][j]->localPosition())).phi();
	std::cout<<" | eta = "<<std::setw(12)<<((me0Geom->etaPartition(simhitME0id))->toGlobal(simhitmu[matchedGENMu][j]->localPosition())).eta()<<std::endl;
      }
      
      const std::vector<ME0RecHit> me0rechits = segRef.specificRecHits();
      for(std::vector<ME0RecHit>::const_iterator rh=me0rechits.begin(); rh!=me0rechits.end(); ++rh) {
	ME0DetId rechitME0id = rh->me0Id();
	
	// 3a verify whether simhits and rechits are in same detid
	if(rechitME0id != simhitME0id) continue;

	if(printInfoME0Match){
	  std::cout<<"          === ME0 RecHit in "<<std::setw(12)<<rechitME0id.rawId()<<" = "<<rechitME0id<<" from ME0Muon with sgmntId = ";
	  std::cout<<std::setw(9)<<it->me0segid()<<" | time t = "<<std::setw(12)<<rh->tof();
	  std::cout<<" | phi = "<<std::setw(12)<<((me0Geom->etaPartition(rechitME0id))->toGlobal(rh->localPosition())).phi();
	  std::cout<<" | eta = "<<std::setw(12)<<((me0Geom->etaPartition(rechitME0id))->toGlobal(rh->localPosition())).eta()<<std::endl;
	}

	// 3b compare global position of simhit and rechit
	GlobalPoint rechitGP = (me0Geom->etaPartition(rechitME0id))->toGlobal(rh->localPosition());
	GlobalPoint simhitGP = (me0Geom->etaPartition(simhitME0id))->toGlobal(simhitmu[matchedGENMu][j]->localPosition());
	double drGlob = sqrt(pow(rechitGP.x()-simhitGP.x(),2)+pow(rechitGP.y()-simhitGP.y(),2)+pow(rechitGP.z()-simhitGP.z(),2));
	double dRGlob = sqrt(pow(rechitGP.x()-simhitGP.x(),2)+pow(rechitGP.y()-simhitGP.y(),2));
	// given that we are in the same eta partition, we can just work with local coordinates
	LocalPoint rechitLP = rh->localPosition();
	LocalPoint simhitLP = simhitmu[matchedGENMu][j]->localPosition();
	LocalError rechitLPE = rh->localPositionError();
	double dRLoc  = sqrt(pow(rechitLP.x()-simhitLP.x(),2)+pow(rechitLP.y()-simhitLP.y(),2));
	double dXLoc = rechitLP.x()-simhitLP.x(); 
	double dYLoc = rechitLP.y()-simhitLP.y(); 
	if(printInfoME0Match){
	  std::cout<<"          === Comparison :: Local dR = "<<std::setw(9)<<dRLoc<<" | Global dR = "<<std::setw(9)<<dRGlob<<" Global dr = "<<std::setw(9)<<drGlob
		   <<" dXLoc = "<<std::setw(9)<<dXLoc<<" +/- "<<sqrt(rechitLPE.xx())<<" [cm] dYLoc = "<<std::setw(9)<<dYLoc<<" +/- "<<sqrt(rechitLPE.yy())<<" [cm]"<<std::endl;
	}
	// allow matching within 3 sigma for both local X and local Y:: look at smearing values in PseudoDigitizer (0.05 for X and 0.1 for Y)
	if(fabs(dXLoc) < 3*preDigiSmearX && fabs(dYLoc) < 3*preDigiSmearY) {
	  if(printInfoME0Match) std::cout<<"          === Matched :: |dXLoc| = "<<fabs(dXLoc)<<" < 3*sigX = "<<3*preDigiSmearX<<" && |dYLoc| = "<<fabs(dYLoc)<<" < 3*sigY = "<<3*preDigiSmearY<<std::endl;
	  ++NmatchedToSegment;
	}
      }
    }
    if(printInfoME0Match) std::cout<<"=== Number of Matched Hits :: "<<NmatchedToSegment<<std::endl;
    if(NmatchedToSegment > (nMatchedHits-1)) { // NmatchedToSegment >= nMatchedHits ==> consider the segment matched to genparticle
      std::vector< std::pair<int,int> > tmp = me0mu[matchedGENMu];
      tmp.push_back(std::make_pair(me0muonpos,NmatchedToSegment));
      me0mu[matchedGENMu] = tmp;
      // me0mu[matchedGENMu]=me0muonpos;
      // me0muMatchedHits[matchedGENMu] = NmatchedToSegment;
    }
    /*
    const std::vector<ME0RecHit> me0rechits = segRef.specificRecHits();
    for(std::vector<ME0RecHit>::const_iterator rh=me0rechits->begin(); rh!=me0rechits->end(); ++rh) {
      // here start applying the matching to the simhits
      // interesting to do, after having matched the right me0muon, I can also check which one is closest to the genparticle
      // to see in what % of cases this genparticle matching is giving the wrong match
    }
    */

    // Recalculate the time
    /*
    float averageTime=0.;
    for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
      averageTime += (*rh)->tof();
    }
    if(rechits.size() != 0) averageTime=averageTime/(rechits.size());
    float timeUncrt=0.;
    for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
      timeUncrt += pow((*rh)->tof()-averageTime,2);
    }
    if(rechits.size() > 1) timeUncrt=timeUncrt/(rechits.size()-1);
    timeUncrt = sqrt(timeUncrt);
    */

    // ME0Muon->chi2() ndof() dxy() dxyError() dz() dzError()


  }
  // --------------------------------------------------------------------


  // Do a print out of all saved information
  // ---------------------------------------
  if(printInfoSignal) {
    for(unsigned int i=0; i<indmu.size(); ++i) {
      std::cout<<"=========================="<<std::endl;
      std::cout<<"=== Muon "<<i+1<<" Information ==="<<std::endl;
      std::cout<<"=========================="<<std::endl;
      if(indmu[i] != -1) {
	std::cout<<"=== GEN Muon :: id = "<<std::showpos<<std::setw(2)<<myGenEvent->barcode_to_particle(indmu.at(i))->pdg_id();
	std::cout<<" | eta = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().eta()<<" | phi = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().phi();
	std::cout<<" | pt = "<<std::setw(9)<<myGenEvent->barcode_to_particle(indmu.at(i))->momentum().perp()<<" |        index = "<<indmu.at(i)<<std::endl;
      }
      if(trkmu[i] != -1) {
	std::cout<<"=== SIM Muon :: id = "<<std::setw(2)<<SimTk->at(trkmu.at(i)-1).type()<<" | eta = "<<std::setw(9)<<SimTk->at(trkmu.at(i)-1).momentum().eta(); // !!! starts counting at 1, not at 0 !!!
	std::cout<<" | phi = "<<std::setw(9)<<SimTk->at(trkmu.at(i)-1).momentum().phi()<<" | pt = "<<std::setw(9)<<SimTk->at(trkmu.at(i)-1).momentum().pt();     // trackId = 1 is accessed by SimTk->at(0)
	std::cout<<" | genpartIndex = "<<SimTk->at(trkmu.at(i)-1).genpartIndex()<<" | trackId = "<<SimTk->at(trkmu.at(i)-1).trackId();
	std::cout<<" | vertexId = "<<SimTk->at(trkmu.at(i)-1).vertIndex()<<std::endl;
      }
      if(vtxmu[i] != -1) {
	std::cout<<"=== SIM Vtx :: vtx = "<<std::setw(2)<<theSimVertices.at(vtxmu[i]).vertexId()<<" and position (in cm) : [x,y,z] = [";
	std::cout<<theSimVertices.at(vtxmu[i]).position().x()<<","<<theSimVertices.at(vtxmu[i]).position().y()<<","<<theSimVertices.at(vtxmu[i]).position().z()<<"] or [r,z] = [";
	std::cout<<sqrt(pow(theSimVertices.at(vtxmu[i]).position().x(),2)+pow(theSimVertices.at(vtxmu[i]).position().y(),2))<<","<<theSimVertices.at(vtxmu[i]).position().z()<<"] (units in cm)"<<std::endl;
      }
      if(simhitmu.size()>i-1) {
	std::cout<<"=== SIM Hits in ME0 :: "<<std::setw(2)<<simhitmu[i].size()<<std::endl;
	std::cout<<"--------------------------"<<std::endl;
	for(unsigned int j=0; j<simhitmu[i].size(); ++j) {
	  ME0DetId me0id(simhitmu[i][j]->detUnitId());
	  std::cout<<"=== ME0 SimHit in "<<std::setw(12)<<simhitmu[i][j]->detUnitId()<<" = "<<me0id<<" from simtrack with trackId = ";
	  std::cout<<std::setw(9)<<simhitmu[i][j]->trackId()<<" | time t = "<<std::setw(12)<<simhitmu[i][j]->timeOfFlight();
	  std::cout<<" | phi = "<<std::setw(12)<<((me0Geom->etaPartition(me0id))->toGlobal(simhitmu[i][j]->localPosition())).phi();
	  std::cout<<" | eta = "<<std::setw(12)<<((me0Geom->etaPartition(me0id))->toGlobal(simhitmu[i][j]->localPosition())).eta()<<std::endl;
	}
      }
      std::cout<<"--------------------------"<<std::endl;
      for(unsigned int j=0; j<me0mu[i].size(); ++j) {
	std::cout<<"=== ME0 Muon :: ch = "<<std::setw(2)<<theME0Muons.at(me0mu[i][j].first).charge()<<" | eta = "<<std::setw(9)<<theME0Muons.at(me0mu[i][j].first).eta(); 
	std::cout<<" | phi = "<<std::setw(9)<<theME0Muons.at(me0mu[i][j].first).phi()<<" | pt = "<<std::setw(9)<<theME0Muons.at(me0mu[i][j].first).pt();     
	std::cout<<" | ME0MuonId  = "<<std::setw(3)<<me0mu[i][j].first<<" | ME0SegRefId  = "<<std::setw(3)<<theME0Muons.at(me0mu[i][j].first).me0segid();
	std::cout<<" | time = "<<theME0Muons.at(me0mu[i][j].first).me0segment().time()<<" +/- "<<theME0Muons.at(me0mu[i][j].first).me0segment().timeErr();
	std::cout<<" | Nhits = "<<theME0Muons.at(me0mu[i][j].first).me0segment().nRecHits()<<" | matched = "<<me0mu[i][j].second /*<<std::endl*/ ;
	// std::cout<<" | Track Hits = "<<theME0Muons.at(me0mu[i][j].first).innerTrack().recHitsSize();
	std::cout<<" | Chi2/ndof = "<<theME0Muons.at(me0mu[i][j].first).innerTrack().get()->chi2()<<"/"<<theME0Muons.at(me0mu[i][j].first).innerTrack().get()->ndof();
	std::cout<<" | dxy = "<<theME0Muons.at(me0mu[i][j].first).innerTrack().get()->dxy()<<" [cm] | dz = "<<theME0Muons.at(me0mu[i][j].first).innerTrack().get()->dz()<<" [cm]"<<std::endl;  
      }
      std::cout<<"==========================\n"<<std::endl;
    }
  } // end printInfoSignal

  // =================================




}

bool MyME0InTimePUAnalyzer::checkVector(std::vector<int>& myvec, int myint) {
  bool found = false;
  for(std::vector<int>::const_iterator it=myvec.begin(); it<myvec.end(); ++it){ 
    if((*it)==myint) found = true;
  }
  return found;
}


// ------------ method called once each job just before starting event loop  ------------
/*
void 
MyME0InTimePUAnalyzer::beginJob()
{
}
*/

// ------------ method called once each job just after ending the event loop  ------------
/*
void 
MyME0InTimePUAnalyzer::endJob() 
{
}
*/

// ------------ method called when starting to processes a run  ------------
/*
void 
MyME0InTimePUAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
MyME0InTimePUAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MyME0InTimePUAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MyME0InTimePUAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MyME0InTimePUAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MyME0InTimePUAnalyzer);

//backup
  /*
  std::unique_ptr<const HepMC::GenEvent> myGenEvent = std::unique_ptr<const HepMC::GenEvent>(new HepMC::GenEvent(*(hepmcevent->GetEvent())));
  for(HepMC::GenEvent::particle_const_iterator it = myGenEvent->particles_begin(); it != myGenEvent->particles_end(); ++it) {
    std::unique_ptr<HepMC::GenParticle> genpart = std::unique_ptr<HepMC::GenParticle>(new HepMC::GenParticle(*(*it)));

    if(abs(genpart->pdg_id()) == 13 && genpart->isPromptFinalState()) {
      index_genpart_signal.push_back(genpart->barcode());
    }
    else if(abs(genpart->pdg_id()) == 13 && genpart->isPromptDecayed()) {
      index_genpart_background.push_back(genpart->barcode());
    }
    else if(abs(genpart->pdg_id()) == 13 && genpart->isDirectPromptTauDecayProductFinalState()) {
      index_genpart_signal.push_back(genpart->barcode());
    }
    else {}
  }
  */  
  /*
  for(unsigned int i=0; i<genParticles->size(); ++i) {
    // 1) consider only muons
    if(abs(genParticles->at(i).pdgId()) == 13 && genParticles->at(i).status() == 1 && genParticles->at(i).numberOfMothers() > 0) { 
      // 2) if mother of genparticle is a Z, save the genparticle index (i) and save mother index
      if(fabs(genParticles->at(i).mother()->pdgId()) == 23) { 
	index_genpart_signal.push_back(i); 
	index_genpart_signal_mother.push_back(genParticles->at(i).mother()->barcode());
	if(fabs(genParticles->at(i).eta())>me0mineta) me0genpartfound = true; 
      }
      // 3) if mother of particle is the same particle, then go one step deeper in hierarchy and repeat
      else if(abs(genParticles->at(i).pdgId()) == abs(genParticles->at(i).mother()->pdgId())) {
	if(genParticles->at(i).mother()->numberOfMothers() > 0) {
	  if(abs(genParticles->at(i).mother()->mother()->pdgId()) == 23) { 
	    index_genpart_signal.push_back(i);
	    index_genpart_signal_mother.push_back(genParticles->at(i).mother()->mother()->barcode()); 
	    if(fabs(genParticles->at(i).eta())>me0mineta) me0genpartfound = true;
	  }  
	  else{ index_genpart_background.push_back(i); }
	}
	else { index_genpart_background.push_back(i); }
      }
      else { index_genpart_background.push_back(i); }
    }
  }
  */
