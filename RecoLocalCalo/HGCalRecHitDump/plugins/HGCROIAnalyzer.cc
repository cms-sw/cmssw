#include "HGCROIAnalyzer.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/JetTools.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCAnalysisTools.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/PCAShowerAnalysis.h"

#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include <iostream>

using namespace std;

//
HGCROIAnalyzer::HGCROIAnalyzer( const edm::ParameterSet &iConfig ) : 
  slimmedRecHits_(new std::vector<SlimmedRecHit>),
  slimmedClusters_(new std::vector<SlimmedCluster>),
  slimmedROIs_(new std::vector<SlimmedROI>),
  slimmedVertices_(new std::vector<SlimmedVertex>),
  genVertex_(new TLorentzVector),
  useSuperClustersAsROIs_(false),
  useStatus3ForGenVertex_(false),
  useStatus3AsROIs_(false)
{
  //configure analyzer
  g4TracksSource_    = consumes<std::vector<SimTrack> >(iConfig.getParameter< edm::InputTag >("g4TracksSource"));
  g4VerticesSource_  = consumes<std::vector<SimVertex> >(iConfig.getParameter< edm::InputTag >("g4VerticesSource"));
  genSource_         = consumes<edm::View<reco::Candidate> >(iConfig.getParameter< edm::InputTag >("genSource"));
  genBarcodesSource_ = consumes<std::vector<int> >(iConfig.getParameter< edm::InputTag >("genSource"));
  genCandsFromSimTracksSource_ = consumes<reco::GenParticleCollection>(iConfig.getParameter< edm::InputTag >("genCandsFromSimTracksSource"));
  genJetsSource_    = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genJetsSource"));
  recoVertexSource_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("recoVertexSource"));
  useSuperClustersAsROIs_ = iConfig.getParameter<bool>("useSuperClustersAsROIs");
  useStatus3ForGenVertex_ = iConfig.getParameter<bool>("useStatus3ForGenVertex");
  useStatus3AsROIs_        = iConfig.getParameter<bool>("useStatus3AsROIs");
  superClustersSource_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("superClustersSource"));
  pfJetsSource_     = consumes<std::vector<reco::PFJet> >(iConfig.getParameter< edm::InputTag >("pfJetsSource"));
  eeSimHitsSource_  = consumes<edm::PCaloHitContainer>(iConfig.getParameter< edm::InputTag >("eeSimHitsSource"));
  hefSimHitsSource_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter< edm::InputTag >("hefSimHitsSource"));
  eeRecHitsSource_  = consumes<HGCRecHitCollection>(iConfig.getParameter< edm::InputTag >("eeRecHitsSource"));
  hefRecHitsSource_ = consumes<HGCRecHitCollection>(iConfig.getParameter< edm::InputTag >("hefRecHitsSource"));
  hepmceventSource_ = consumes<edm::HepMCProduct>(iConfig.getParameter< edm::InputTag >("hepmceventSource"));

  edm::Service<TFileService> fs;
  tree_=fs->make<TTree>("HGC","HGC");
  tree_->Branch("run",   &run_,   "run/I");
  tree_->Branch("event", &event_, "event/I");
  tree_->Branch("lumi",  &lumi_,  "lumi/I");
  tree_->Branch("RecHits",   "std::vector<SlimmedRecHit>",   &slimmedRecHits_);
  tree_->Branch("Clusters",  "std::vector<SlimmedCluster>",  &slimmedClusters_);
  tree_->Branch("ROIs",      "std::vector<SlimmedROI>",      &slimmedROIs_);
  tree_->Branch("Vertices",  "std::vector<SlimmedVertex>",   &slimmedVertices_);
  tree_->Branch("GenVertex", "TLorentzVector",               &genVertex_);
}

//
HGCROIAnalyzer::~HGCROIAnalyzer()
{
}

//
//store basic information on RecHits
//
void HGCROIAnalyzer::slimRecHits(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  slimmedRecHits_->clear();
  
  //EE hits
  edm::Handle<edm::PCaloHitContainer> eeSimHits;
  iEvent.getByToken(eeSimHitsSource_, eeSimHits);
  edm::Handle<HGCRecHitCollection> eeRecHits;
  iEvent.getByToken(eeRecHitsSource_,eeRecHits); // change to uncalib rechits

  if(eeRecHits.isValid())
    {
      edm::ESHandle<HGCalGeometry> eeGeom;
      iSetup.get<IdealGeometryRecord>().get("HGCalEESensitive",eeGeom);
      const HGCalTopology &topo=eeGeom->topology();
      const HGCalDDDConstants &dddConst=topo.dddConstants(); 
      
      float eeMipEn(55.1);
      for(HGCRecHitCollection::const_iterator hit_it=eeRecHits->begin(); 
	  hit_it!=eeRecHits->end(); 
	  hit_it++)
	{
	  uint32_t recoDetId(hit_it->id());
	  const GlobalPoint pos( std::move( eeGeom->getPosition(recoDetId) ) );	
	  slimmedRecHits_->push_back( SlimmedRecHit(recoDetId,
						    pos.x(),pos.y(),pos.z(),
						    hit_it->energy()*1e6/eeMipEn,
						    hit_it->time(),
						    0.0 ) );
	}
      
      //add the simHits -> replace with hydra
      /*
      if(eeSimHits.isValid())
	{
	  for(edm::PCaloHitContainer::const_iterator hit_it = eeSimHits->begin(); 
	      hit_it != eeSimHits->end();
	      hit_it++)
	    {
	      //gang SIM->RECO cells to get final layer assignment  
	      HGCalDetId simId(hit_it->id());
	      int layer(simId.layer()),cell(simId.cell());
	      std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,topo.detectorType());
	      cell  = recoLayerCell.first;
	      layer = recoLayerCell.second;
	      if(layer<0) continue;
	      
	      uint32_t recoDetId( (uint32_t)HGCEEDetId(ForwardSubdetector(ForwardSubdetector::HGCEE),
						       simId.zside(),
						       layer,
						       simId.sector(),
						       simId.subsector(),
						       cell));
	      SlimmedRecHitCollection::iterator theHit=std::find(slimmedRecHits_->begin(),
								 slimmedRecHits_->end(),
								 SlimmedRecHit(recoDetId));
	      if(theHit == slimmedRecHits_->end()) continue;

	      float dist2center( sqrt( theHit->x_*theHit->x_+theHit->y_*theHit->y_+theHit->z_*theHit->z_) );
	      float tof(hit_it->time()-dist2center/(0.1*CLHEP::c_light)+1.0);
	      float emf(hit_it->energyEM()/hit_it->energy());
	      theHit->addSimHit( hit_it->energy()*1e6/eeMipEn,tof,emf );
	    }
	}
      */
    }
  
  //HEF hits
  edm::Handle<edm::PCaloHitContainer> hefSimHits;
  iEvent.getByToken(hefSimHitsSource_, hefSimHits);
  edm::Handle<HGCRecHitCollection> hefRecHits;
  iEvent.getByToken(hefRecHitsSource_, hefRecHits); // change to uncalib rechits
  if(hefRecHits.isValid())
    {
      edm::ESHandle<HGCalGeometry> hefGeom;
      iSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hefGeom);
      const HGCalTopology &topo=hefGeom->topology();
      const HGCalDDDConstants &dddConst=topo.dddConstants(); 
      
      float hefMipEn(85.0);
      for(HGCRecHitCollection::const_iterator hit_it=hefRecHits->begin(); 
	  hit_it!=hefRecHits->end(); 
	  hit_it++)
	{
	  uint32_t recoDetId(hit_it->id());
	  const GlobalPoint pos( std::move( hefGeom->getPosition(recoDetId) ) );	
	  slimmedRecHits_->push_back( SlimmedRecHit(recoDetId,
						    pos.x(),pos.y(),pos.z(),
						    hit_it->energy()*1e6/hefMipEn,						    
						    hit_it->time(),
						    0.0 ) );
	}
      
      //add the simHits --> replace this with hydra
      /*
      if(hefSimHits.isValid())
	{
	  for(edm::PCaloHitContainer::const_iterator hit_it = hefSimHits->begin(); 
	      hit_it != hefSimHits->end();
	      hit_it++)
	    {
	      //gang SIM->RECO cells to get final layer assignment  
	      HGCalDetId simId(hit_it->id());
	      int layer(simId.layer()),cell(simId.cell());
	      std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,topo.detectorType());
	      cell  = recoLayerCell.first;
	      layer = recoLayerCell.second;
	      if(layer<0) continue;
	      
	      uint32_t recoDetId( (uint32_t)HGCHEDetId(ForwardSubdetector(ForwardSubdetector::HGCHEF),
						       simId.zside(),
						       layer,
						       simId.sector(),
						       simId.subsector(),
						       cell));
	      SlimmedRecHitCollection::iterator theHit=std::find(slimmedRecHits_->begin(),
								 slimmedRecHits_->end(),
								 SlimmedRecHit(recoDetId));
	      if(theHit == slimmedRecHits_->end()) continue;
	      
	      float dist2center( sqrt( theHit->x_*theHit->x_+theHit->y_*theHit->y_+theHit->z_*theHit->z_) );
	      float tof(hit_it->time()-dist2center/(0.1*CLHEP::c_light)+1.0);
	      float emf(hit_it->energyEM()/hit_it->energy());
	      theHit->addSimHit( hit_it->energy()*1e6/hefMipEn,tof,emf );
	    }
	}
      */
    }
}

//
void HGCROIAnalyzer::doMCJetMatching(edm::Handle<std::vector<reco::PFJet> > &pfJets,
				     edm::Handle<reco::GenJetCollection> &genJets,
				     edm::Handle<edm::View<reco::Candidate> > &genParticles,
				     std::unordered_map<uint32_t,uint32_t> &reco2genJet,
				     std::unordered_map<uint32_t,uint32_t> &genJet2Parton,
				     std::unordered_map<uint32_t,uint32_t> &genJet2Stable)
{
  
  //
  // match gen jets
  // gen particles
  // - find all status2 parton (after showering) within R=0.4
  // - minimize in pT
  // reco matching
  // based on http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2013_125_v3.pdf
  // - give preference to higher pT gen jets first (collections are already ordered)
  // - find reco jet which minimizes deltaR within 0.2 cone
  // - remove matched reco jet from next matches
  // - iterate until all gen jets are matched
  //
  for(size_t j=0; j<genJets->size(); j++)
    {
      const reco::GenJet& genjet=genJets->at(j);
      float pt=genjet.pt();
      float abseta=fabs(genjet.eta());
      if(abseta<1.5 || abseta>3.0) continue;
     
      //gen particle matching
      float minDPt2Stable(99999.),minDpt2Parton(9999.);
      for(size_t i = 0; i < genParticles->size(); ++ i)
	{
	  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[i] );
	  float dR(deltaR(p,genjet));
	  if(dR>0.4) continue;
	  float dPt( fabs(p.pt()-pt));

	  if(p.status()==1)
	    {
	      if(dPt>minDPt2Stable) continue;
	      minDPt2Stable=dPt;
	      genJet2Stable[j]=i;
	    }
	  else if(p.status()==2 && ( (abs(p.pdgId())==21 || abs(p.pdgId())<6) ) )
	    {
	      if(dPt>minDpt2Parton) continue;
	      minDpt2Parton=dPt;
	      genJet2Parton[j]=i;
	    }
	}
  
      //reco matching
      float minDR(0.2);
      for(size_t i=0; i<pfJets->size(); i++)
	{
	  const reco::PFJet &jet=pfJets->at(i);
	  float dR=deltaR(jet,genjet);
	  if(dR>minDR) continue;
	  minDR=dR;
	  if(reco2genJet.find(i)!=reco2genJet.end()) continue;
	  reco2genJet[i]=j;
	}
    }
}

void HGCROIAnalyzer::doMCJetMatching(edm::Handle<reco::SuperClusterCollection> &superClusters,
				     edm::Handle<reco::GenJetCollection> &genJets,
				     edm::Handle<edm::View<reco::Candidate> > &genParticles,
				     std::unordered_map<uint32_t,uint32_t> &reco2genJet,
				     std::unordered_map<uint32_t,uint32_t> &genJet2Parton,
				     std::unordered_map<uint32_t,uint32_t> &genJet2Stable)
{
  
  //
  // match gen jets
  // gen particles
  // - find all status2 parton (after showering) within R=0.4
  // - minimize in pT
  // reco matching
  // based on http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2013_125_v3.pdf
  // - give preference to higher pT gen jets first (collections are already ordered)
  // - find reco jet which minimizes deltaR within 0.2 cone
  // - remove matched reco jet from next matches
  // - iterate until all gen jets are matched
  //
  for(size_t j=0; j<genJets->size(); j++)
    {
      const reco::GenJet& genjet=genJets->at(j);
      float pt=genjet.pt();
      float abseta=fabs(genjet.eta());
      if(abseta<1.5 || abseta>3.0) continue;
     
      //gen particle matching
      float minDPt2Stable(99999.),minDpt2Parton(9999.);
      for(size_t i = 0; i < genParticles->size(); ++ i)
	{
	  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[i] );
	  float dR(deltaR(p,genjet));
	  if(dR>0.4) continue;
	  float dPt( fabs(p.pt()-pt));

	  if(p.status()==1)
	    {
	      if(dPt>minDPt2Stable) continue;
	      minDPt2Stable=dPt;
	      genJet2Stable[j]=i;
	    }
	  else if(p.status()==2 && ( (abs(p.pdgId())==21 || abs(p.pdgId())<6) ) )
	    {
	      if(dPt>minDpt2Parton) continue;
	      minDpt2Parton=dPt;
	      genJet2Parton[j]=i;
	    }
	}
  
      //reco matching
      float minDR(0.2);
      int i(0);
      for(reco::SuperClusterCollection::const_iterator c_it=superClusters->begin();
	  c_it!=superClusters->end();
	  c_it++,i++)
	{
	  float dR=deltaR(c_it->position(),genjet);
	  if(dR>minDR) continue;
	  minDR=dR;
	  if(reco2genJet.find(i)!=reco2genJet.end()) continue;
	  reco2genJet[i]=j;
	}
    }
}



//
void HGCROIAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{

  //event header
  run_   = iEvent.id().run();
  event_ = iEvent.id().event();
  lumi_  = iEvent.luminosityBlock();

  //parse rec hits
  slimRecHits(iEvent,iSetup);

  //Geant4 collections
  edm::Handle<std::vector<SimTrack> > SimTk;
  iEvent.getByToken(g4TracksSource_,SimTk);
  edm::Handle<std::vector<SimVertex> > SimVtx;
  iEvent.getByToken(g4VerticesSource_,SimVtx); 
  edm::Handle<std::vector<int> > genBarcodes;
  iEvent.getByToken(genBarcodesSource_, genBarcodes);  

  //PV collection 
  edm::Handle<reco::VertexCollection> vtxH;
  iEvent.getByToken(recoVertexSource_, vtxH);
  slimmedVertices_->clear();
  std::vector<size_t> selVtx;
  for(size_t iv=0; iv<vtxH->size(); iv++)
    {
      const reco::Vertex &vtx=vtxH->at(iv);
      if(!vtx.isValid()) continue;
      if(vtx.isFake()) continue;
      selVtx.push_back(iv);
      slimmedVertices_->push_back(SlimmedVertex(vtx.nTracks(),vtx.x(),vtx.y(),vtx.z(),vtx.p4().pt(),vtx.normalizedChi2()) );
    }
  
  // get the gen particles
  edm::Handle<edm::View<reco::Candidate> > genParticles;
  iEvent.getByToken(genSource_, genParticles);
  edm::Handle<reco::GenParticleCollection> genCandsFromSimTracks;
  iEvent.getByToken(genCandsFromSimTracksSource_, genCandsFromSimTracks);

  // get the vertex info
  edm::Handle<edm::HepMCProduct>  hepmcevent;
  iEvent.getByToken(hepmceventSource_, hepmcevent);
  const HepMC::GenEvent& genevt = hepmcevent->getHepMCData();
  genVertex_->SetXYZT(0.,0.,0.,0.);
  if( genevt.vertices_size() ) {
    HepMC::FourVector temp = (*genevt.vertices_begin())->position() ;
    genVertex_->SetXYZT(0.1*temp.x(),0.1*temp.y(),0.1*temp.z(),temp.t()/299.792458); // convert positions to cm and time to ns (it's in mm to start)
  }
  
  //jet analysis
  edm::Handle<std::vector<reco::PFJet> > pfJets;
  iEvent.getByToken(pfJetsSource_,pfJets);
  edm::Handle<reco::SuperClusterCollection> superClusters;
  iEvent.getByToken(superClustersSource_,superClusters);
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(genJetsSource_, genJets);
  std::unordered_map<uint32_t,uint32_t> reco2genJet,genJet2Parton,genJet2Stable;
  if(useSuperClustersAsROIs_) {
    doMCJetMatching(superClusters,genJets,genParticles,reco2genJet,genJet2Parton,genJet2Stable);
  } else if (useStatus3AsROIs_) {
     //doParticleMatching(superClusters,genJets,genParticles,reco2genJet,genJet2Parton,genJet2Stable);
  } else {
    doMCJetMatching(pfJets,genJets,genParticles,reco2genJet,genJet2Parton,genJet2Stable);
  }

  //
  // Analyze reco jets fiducial in HGC
  // 
  slimmedROIs_->clear();
  slimmedClusters_->clear();

  if(useSuperClustersAsROIs_)
    {
      int j(0);
      for(reco::SuperClusterCollection::const_iterator c_it=superClusters->begin();
	  c_it!=superClusters->end();
	  c_it++,j++)
	{
	  	  
	  if(c_it->energy()<10 || fabs(c_it->eta())<1.5 || fabs(c_it->eta())>3.0) continue;
	  

	  SlimmedROI slimSuperCluster(c_it->energy()/TMath::CosH(c_it->eta()),c_it->eta(),c_it->phi(),0.,0.);
	  for(size_t isv=0; isv<selVtx.size(); isv++) slimSuperCluster.addBetaStar(0);
	  
	  slimSuperCluster.setPFEnFractions(0.,1.,0.);
	  slimSuperCluster.setPFMultiplicities(0.,c_it->clustersSize(),0.);
	  
	  if( reco2genJet.find(j) != reco2genJet.end())
	    {
	      uint32_t genJetIdx=reco2genJet[j];
	      const reco::GenJet& genjet=genJets->at( genJetIdx );
	      slimSuperCluster.setGenJet(genjet.pt(),genjet.eta(),genjet.phi(),genjet.mass(),genjet.jetArea());
	      
	      if(genJet2Parton.find( genJetIdx )!=genJet2Parton.end())
		{
		  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[ genJet2Parton[genJetIdx] ] );
		  slimSuperCluster.setParton(p.pt(),p.eta(),p.phi(),p.pdgId());
		}
	      
	      if(genJet2Stable.find(genJetIdx)!=genJet2Stable.end())
		{
		  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[ genJet2Stable[genJetIdx] ] );

		  G4InteractionPositionInfo intInfo=getInteractionPosition(SimTk.product(),SimVtx.product(),genBarcodes->at(genJet2Stable[genJetIdx]));
		  math::XYZVectorD hitPos=intInfo.pos; 
		  int motherId( p.numberOfMothers() ? p.mother()->pdgId() : 0);
		  
		  //use gen particles from sim tracks instead to determine the interaction position
		  if(hitPos.x()==0 && hitPos.y()==0 && hitPos.z()==0)
		    {
		      float minDPt(99999.),minDR(99999.);
		      for(size_t ig=0; ig<genCandsFromSimTracks->size(); ig++)
			{
			  const reco::GenParticle & simtrackp=genCandsFromSimTracks->at(ig);
			  
			  //pdgId must match either for the mother or the particle 
			  int simtrackpmotherid( simtrackp.numberOfMothers() ? simtrackp.mother()->pdgId() : 0 );
			  if(simtrackpmotherid!=p.pdgId() && simtrackp.pdgId()!=p.pdgId()) continue;
			  
			  //minimize deltaR within 0.5
			  double dR=deltaR(simtrackp.eta(),simtrackp.phi(),p.eta(),p.phi());
			  if(dR>0.5) continue;
			  if(dR>minDR) continue;
			  minDR=dR;
			  
			  //minimize pT match
			  float dPt=fabs(simtrackp.pt()-p.pt());
			  if(dPt>minDPt) continue;
			  minDPt=dPt;
			  
			  //update the hit position
			  hitPos   = math::XYZVectorD( simtrackp.vx(), simtrackp.vy(), simtrackp.vz());
			}
		      
		      //
		      std::cout << "Updated G4 hit position to : " << hitPos.x() << "," << hitPos.y() << "," << hitPos.z() << std::endl;
		    }
	       
		  
		  slimSuperCluster.setStable(p.pdgId(),   motherId,
					     p.pt(),      p.eta(),     p.phi(),  
					     hitPos.x(),  hitPos.y(),  hitPos.z());
		}
	    }

	  //iterate over clusters
	  for(reco::CaloCluster_iterator cIt = c_it->clustersBegin(); cIt!=c_it->clustersEnd(); cIt++)
	    {	  
	      const reco::CaloCluster *cl = cIt->get(); 
	      SlimmedCluster slimCluster( cl->energy(),cl->eta(),cl->phi(),(int)cl->hitsAndFractions().size() );
	      slimCluster.roiidx_=slimmedROIs_->size();	  
	      
	      //run pca analysis
	      PCAShowerAnalysis pca;
	      PCAShowerAnalysis::PCASummary_t pcaSummary=pca.computeShowerParameters( *cl, *slimmedRecHits_);
	      slimCluster.center_x_ = pcaSummary.center_x;
	      slimCluster.center_y_ = pcaSummary.center_y;
	      slimCluster.center_z_ = pcaSummary.center_z;
	      slimCluster.axis_x_   = pcaSummary.axis_x;
	      slimCluster.axis_y_   = pcaSummary.axis_y;
	      slimCluster.axis_z_   = pcaSummary.axis_z;
	      slimCluster.ev_1_     = pcaSummary.ev_1;
	      slimCluster.ev_2_     = pcaSummary.ev_2;
	      slimCluster.ev_3_     = pcaSummary.ev_3;
	      slimCluster.sigma_1_  = pcaSummary.sigma_1;
	      slimCluster.sigma_2_  = pcaSummary.sigma_2;
	      slimCluster.sigma_3_  = pcaSummary.sigma_3;
	      
	      GlobalPoint pcaShowerPos(pcaSummary.center_x,pcaSummary.center_y,pcaSummary.center_z);
	      GlobalVector pcaShowerDir(pcaSummary.axis_x,pcaSummary.axis_y,pcaSummary.axis_z);
	      for (unsigned int ih=0;ih<cl->hitsAndFractions().size();++ih) 
		{
		  uint32_t id = (cl->hitsAndFractions())[ih].first.rawId();
		  SlimmedRecHitCollection::iterator theHit=std::find(slimmedRecHits_->begin(),
								     slimmedRecHits_->end(),
								     SlimmedRecHit(id));
		  if(theHit==slimmedRecHits_->end()) continue;
		  
		  theHit->clustId_=slimmedClusters_->size();
		  
		  GlobalPoint cellPos(theHit->x_,theHit->y_,theHit->z_);
		  float cellSize = theHit->cellSize_;
		  float lambda = (cellPos.z()-pcaShowerPos.z())/pcaShowerDir.z();
		  GlobalPoint interceptPos = pcaShowerPos + lambda*pcaShowerDir;
		  float absdx=std::fabs(cellPos.x()-interceptPos.x());
		  float absdy=std::fabs(cellPos.y()-interceptPos.y());
		  
		  theHit->isIn3x3_ = (absdx<cellSize*3./2. && absdy<cellSize*3./2.);
		  theHit->isIn5x5_ = (absdx<cellSize*5./2. && absdy<cellSize*5./2.);
		  theHit->isIn7x7_ = (absdx<cellSize*7./2. && absdy<cellSize*7./2.);
		}
	      
	      slimmedClusters_->push_back(slimCluster);
	    }

	  //all done with this jet
	  slimmedROIs_->push_back(slimSuperCluster);
	}
    }
  else
    {
      for(size_t j=0; j<pfJets->size(); j++)
	{
	  const reco::PFJet &jet=pfJets->at(j);
	  
	  if(jet.pt()<10 || fabs(jet.eta())<1.5 || fabs(jet.eta())>3.0) continue;
	  
	  SlimmedROI slimJet(jet.pt(),jet.eta(),jet.phi(),jet.mass(),jet.jetArea());
	  
	  for(size_t isv=0; isv<selVtx.size(); isv++)
	    {
	      size_t iv=selVtx[isv];
	      std::pair<float,float> beta=betaVariables( &jet, &(vtxH->at(iv)), *vtxH);
	      slimJet.addBetaStar(beta.second);
	    }
	  
	  slimJet.setPFEnFractions(jet.neutralHadronEnergyFraction(),
				   jet.photonEnergyFraction(),
				   jet.chargedHadronEnergyFraction());
	  slimJet.setPFMultiplicities(jet.neutralHadronMultiplicity(),
				      jet.photonMultiplicity(),
				      jet.chargedHadronMultiplicity());
	  
	  if( reco2genJet.find(j) != reco2genJet.end())
	    {
	      uint32_t genJetIdx=reco2genJet[j];
	      
	      const reco::GenJet& genjet=genJets->at( genJetIdx );
	      slimJet.setGenJet(genjet.pt(),genjet.eta(),genjet.phi(),genjet.mass(),genjet.jetArea());
	      
	      if(genJet2Parton.find( genJetIdx )!=genJet2Parton.end())
		{
		  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[ genJet2Parton[genJetIdx] ] );
		  slimJet.setParton(p.pt(),p.eta(),p.phi(),p.pdgId());
		  
		}
	      
	      if(genJet2Stable.find(genJetIdx)!=genJet2Stable.end())
		{
		  const reco::GenParticle & p = dynamic_cast<const reco::GenParticle &>( (*genParticles)[ genJet2Stable[genJetIdx] ] );
		  G4InteractionPositionInfo intInfo=getInteractionPosition(SimTk.product(),SimVtx.product(),genBarcodes->at(genJet2Stable[genJetIdx]));
		  math::XYZVectorD hitPos=intInfo.pos; 
		  int motherId( p.numberOfMothers() ? p.mother()->pdgId() : 0);
		  slimJet.setStable(p.pdgId(),   motherId,
				    p.pt(),      p.eta(),     p.phi(),  
				    hitPos.x(),  hitPos.y(),  hitPos.z());
		}
	    }
	  	  
	  //first find all pf clusters used
	  std::set<const reco::PFBlockElementCluster *> pfClusters;
	  std::vector<reco::PFCandidatePtr> jetConst(jet.getPFConstituents());
	  for(std::vector<reco::PFCandidatePtr>::iterator cIt=jetConst.begin();
	      cIt!=jetConst.end(); 
	      cIt++)
	    {
	      const reco::PFCandidate::ElementsInBlocks&einb=(*cIt)->elementsInBlocks();
	      for(size_t ieleinb=0; ieleinb<einb.size(); ieleinb++)
		{
		  const reco::PFBlockRef blockRef = einb[ieleinb].first;
		  
		  const edm::OwnVector< reco::PFBlockElement > &eleList=blockRef->elements();
		  for(unsigned int iEle=0; iEle<eleList.size(); iEle++)
		    {
		      reco::PFBlockElement::Type eletype = eleList[iEle].type();
		      if( eletype!=reco::PFBlockElement::NONE ) continue;
		      pfClusters.insert( dynamic_cast<const reco::PFBlockElementCluster*>(&(eleList[iEle])) );
		    }
		}
	    }
	  
	  //iterate of the clusters
	  for(std::set<const reco::PFBlockElementCluster *>::iterator cIt=pfClusters.begin();
	      cIt!=pfClusters.end();
	      cIt++)
	    {
	      const reco::PFClusterRef &cl=(*cIt)->clusterRef();
	      SlimmedCluster slimCluster( cl->energy(),cl->eta(),cl->phi(),(int)cl->hitsAndFractions().size() );
	      slimCluster.roiidx_=slimmedROIs_->size();	  
	      
	      //run pca analysis
	      PCAShowerAnalysis pca;
	      const reco::CaloCluster *caloCl=dynamic_cast<const reco::CaloCluster *>(cl.get());
	      PCAShowerAnalysis::PCASummary_t pcaSummary=pca.computeShowerParameters( *caloCl, *slimmedRecHits_);
	      slimCluster.center_x_ = pcaSummary.center_x;
	      slimCluster.center_y_ = pcaSummary.center_y;
	      slimCluster.center_z_ = pcaSummary.center_z;
	      slimCluster.axis_x_   = pcaSummary.axis_x;
	      slimCluster.axis_y_   = pcaSummary.axis_y;
	      slimCluster.axis_z_   = pcaSummary.axis_z;
	      slimCluster.ev_1_     = pcaSummary.ev_1;
	      slimCluster.ev_2_     = pcaSummary.ev_2;
	      slimCluster.ev_3_     = pcaSummary.ev_3;
	      slimCluster.sigma_1_  = pcaSummary.sigma_1;
	      slimCluster.sigma_2_  = pcaSummary.sigma_2;
	      slimCluster.sigma_3_  = pcaSummary.sigma_3;
	      
	      GlobalPoint pcaShowerPos(pcaSummary.center_x,pcaSummary.center_y,pcaSummary.center_z);
	      GlobalVector pcaShowerDir(pcaSummary.axis_x,pcaSummary.axis_y,pcaSummary.axis_z);
	      for (unsigned int ih=0;ih<cl->hitsAndFractions().size();++ih) 
		{
		  uint32_t id = (cl->hitsAndFractions())[ih].first.rawId();
		  SlimmedRecHitCollection::iterator theHit=std::find(slimmedRecHits_->begin(),
								     slimmedRecHits_->end(),
								     SlimmedRecHit(id));
		  if(theHit==slimmedRecHits_->end()) continue;
		  
		  theHit->clustId_=slimmedClusters_->size();
		  
		  GlobalPoint cellPos(theHit->x_,theHit->y_,theHit->z_);
		  float cellSize = theHit->cellSize_;
		  float lambda = (cellPos.z()-pcaShowerPos.z())/pcaShowerDir.z();
		  GlobalPoint interceptPos = pcaShowerPos + lambda*pcaShowerDir;
		  float absdx=std::fabs(cellPos.x()-interceptPos.x());
		  float absdy=std::fabs(cellPos.y()-interceptPos.y());
		  
		  theHit->isIn3x3_ = (absdx<cellSize*3./2. && absdy<cellSize*3./2.);
		  theHit->isIn5x5_ = (absdx<cellSize*5./2. && absdy<cellSize*5./2.);
		  theHit->isIn7x7_ = (absdx<cellSize*7./2. && absdy<cellSize*7./2.);
		}
	      
	      slimmedClusters_->push_back(slimCluster);
	    }

	  //all done with this jet
	  slimmedROIs_->push_back(slimJet);
	}
    }
  
  //remove unclustered vertices  
  slimmedRecHits_->erase(remove_if(slimmedRecHits_->begin(), slimmedRecHits_->end(), SlimmedRecHit::IsNotClustered),
			 slimmedRecHits_->end());

  //all done, fill tree
  if(slimmedROIs_->size())  tree_->Fill();
}

//
void HGCROIAnalyzer::endJob() 
{ 
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCROIAnalyzer);
