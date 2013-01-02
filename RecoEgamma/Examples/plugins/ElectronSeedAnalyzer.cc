
// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoEgamma/Examples/plugins/ElectronSeedAnalyzer.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/SimpleVector.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TTree.h"

using namespace std;
using namespace reco;

ElectronSeedAnalyzer::ElectronSeedAnalyzer(const edm::ParameterSet& conf) :
  beamSpot_(conf.getParameter<edm::InputTag>("beamSpot"))
{
  inputCollection_=conf.getParameter<edm::InputTag>("inputCollection") ;
  histfile_ = new TFile("electronpixelseeds.root","RECREATE");
}

void ElectronSeedAnalyzer::beginJob()
{
  histfile_->cd();
  tree_ = new TTree("ElectronSeeds","ElectronSeed validation ntuple");
  tree_->Branch("mcEnergy",mcEnergy,"mcEnergy[10]/F");
  tree_->Branch("mcEta",mcEta,"mcEta[10]/F");
  tree_->Branch("mcPhi",mcPhi,"mcPhi[10]/F");
  tree_->Branch("mcPt",mcPt,"mcPt[10]/F");
  tree_->Branch("mcQ",mcQ,"mcQ[10]/F");
  tree_->Branch("superclusterEnergy",superclusterEnergy,"superclusterEnergy[10]/F");
  tree_->Branch("superclusterEta",superclusterEta,"superclusterEta[10]/F");
  tree_->Branch("superclusterPhi",superclusterPhi,"superclusterPhi[10]/F");
  tree_->Branch("superclusterEt",superclusterEt,"superclusterEt[10]/F");
  tree_->Branch("seedMomentum",seedMomentum,"seedMomentum[10]/F");
  tree_->Branch("seedEta",seedEta,"seedEta[10]/F");
  tree_->Branch("seedPhi",seedPhi,"seedPhi[10]/F");
  tree_->Branch("seedPt",seedPt,"seedPt[10]/F");
  tree_->Branch("seedQ",seedQ,"seedQ[10]/F");
  tree_->Branch("seedSubdet1",seedSubdet1,"seedSubdet1[10]/I");
  tree_->Branch("seedLayer1",seedLayer1,"seedLayer1[10]/I");
  tree_->Branch("seedSide1",seedSide1,"seedSide1[10]/I");
  tree_->Branch("seedPhi1",seedPhi1,"seedPhi1[10]/F");
  tree_->Branch("seedDphi1",seedDphi1,"seedDphi1[10]/F");
  tree_->Branch("seedDrz1",seedDrz1,"seedDrz1[10]/F");
  tree_->Branch("seedRz1",seedRz1,"seedRz1[10]/F");
  tree_->Branch("seedSubdet2",seedSubdet2,"seedSubdet2[10]/I");
  tree_->Branch("seedLayer2",seedLayer2,"seedLayer2[10]/I");
  tree_->Branch("seedSide2",seedSide2,"seedSide2[10]/I");
  tree_->Branch("seedPhi2",seedPhi2,"seedPhi2[10]/F");
  tree_->Branch("seedDphi2",seedDphi2,"seedDphi2[10]/F");
  tree_->Branch("seedRz2",seedRz2,"seedRz2[10]/F");
  tree_->Branch("seedDrz2",seedDrz2,"seedDrz2[10]/F");
  histeMC_ = new TH1F("eMC","MC particle energy",100,0.,100.);
  histeMCmatched_ = new TH1F("eMCmatched","matched MC particle energy",100,0.,100.);
  histecaldriveneMCmatched_ = new TH1F("ecaldriveneMCmatched","matched MC particle energy, ecal driven",100,0.,100.);
  histtrackerdriveneMCmatched_ = new TH1F("trackerdriveneMCmatched","matched MC particle energy, tracker driven",100,0.,100.);
  histp_ = new TH1F("p","seed p",100,0.,100.);
  histeclu_ = new TH1F("clus energy","supercluster energy",100,0.,100.);
  histpt_ = new TH1F("pt","seed pt",100,0.,100.);
  histptMC_ = new TH1F("ptMC","MC particle pt",100,0.,100.);
  histptMCmatched_ = new TH1F("ptMCmatched","matched MC particle pt",100,0.,100.);
  histecaldrivenptMCmatched_ = new TH1F("ecaldrivenptMCmatched","matched MC particle pt, ecal driven",100,0.,100.);
  histtrackerdrivenptMCmatched_ = new TH1F("trackerdrivenptMCmatched","matched MC particle pt, tracker driven",100,0.,100.);
  histetclu_ = new TH1F("Et","supercluster Et",100,0.,100.);
  histeffpt_ = new TH1F("pt eff","seed effciency vs pt",100,0.,100.);
  histeta_ = new TH1F("seed eta","seed eta",100,-2.5,2.5);
  histetaMC_ = new TH1F("etaMC","MC particle eta",100,-2.5,2.5);
  histetaMCmatched_ = new TH1F("etaMCmatched","matched MC particle eta",100,-2.5,2.5);
  histecaldrivenetaMCmatched_ = new TH1F("ecaldrivenetaMCmatched","matched MC particle eta, ecal driven",100,-2.5,2.5);
  histtrackerdrivenetaMCmatched_ = new TH1F("trackerdrivenetaMCmatched","matched MC particle eta, tracker driven",100,-2.5,2.5);
  histetaclu_ = new TH1F("clus eta","supercluster eta",100,-2.5,2.5);
  histeffeta_ = new TH1F("eta eff","seed effciency vs eta",100,-2.5,2.5);
  histq_ = new TH1F("q","seed charge",100,-2.5,2.5);
  histeoverp_ = new TH1F("E/p","seed E/p",100,0.,10.);
  histnbseeds_ = new TH1I("nrs","Nr of seeds ",50,0.,25.);
  histnbclus_ = new TH1I("nrclus","Nr of superclusters ",50,0.,25.);
  histnrseeds_ = new TH1I("ns","Nr of seeds if clusters",50,0.,25.);
}

void ElectronSeedAnalyzer::endJob()
{
  histfile_->cd();
  tree_->Print();
  tree_->Write();

  // efficiency vs eta
  TH1F *histetaEff = (TH1F*)histetaMCmatched_->Clone("histetaEff");
  histetaEff->Reset();
  histetaEff->Divide(histetaMCmatched_,histeta_,1,1,"b");
  histetaEff->Print();
  histetaEff->GetXaxis()->SetTitle("#eta");
  histetaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs pt
  TH1F *histptEff = (TH1F*)histptMCmatched_->Clone("histotEff");
  histptEff->Reset();
  histptEff->Divide(histptMCmatched_,histpt_,1,1,"b");
  histptEff->Print();
  histptEff->GetXaxis()->SetTitle("p_{T}");
  histptEff->GetYaxis()->SetTitle("Efficiency");

  histeMCmatched_->Write();
  histecaldriveneMCmatched_->Write();
  histtrackerdriveneMCmatched_->Write();
  histeMC_->Write();
  histp_->Write();
  histeclu_->Write();
  histpt_->Write();
  histptMCmatched_->Write();
  histecaldrivenptMCmatched_->Write();
  histtrackerdrivenptMCmatched_->Write();
  histptMC_->Write();
  histetclu_->Write();
  histeffpt_->Write();
  histeta_->Write();
  histetaMCmatched_->Write();
  histecaldrivenetaMCmatched_->Write();
  histtrackerdrivenetaMCmatched_->Write();
  histetaMC_->Write();
  histetaclu_->Write();
  histeffeta_->Write();
  histq_->Write();
  histeoverp_->Write();
  histnbseeds_->Write();
  histnbclus_->Write();
  histnrseeds_->Write();
}

ElectronSeedAnalyzer::~ElectronSeedAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //tree_->Print();
  histfile_->Write();
  histeMC_->Write();
  histfile_->Close();
}

void ElectronSeedAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);



  edm::ESHandle<TrackerGeometry> pDD ;
  edm::ESHandle<MagneticField> theMagField ;
  iSetup.get<TrackerDigiGeometryRecord> ().get(pDD);
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  // rereads the seeds for test purposes
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;


  // get beam spot
  edm::Handle<reco::BeamSpot> theBeamSpot;
  e.getByLabel(beamSpot_, theBeamSpot);

  // get seeds

  edm::Handle<ElectronSeedCollection> elSeeds;
  e.getByLabel(inputCollection_,elSeeds);
  edm::LogInfo("")<<"\n\n =================> Treating event "<<e.id()<<" Number of seeds "<<elSeeds.product()->size();
  int is=0;

  FTSFromVertexToPointFactory   myFTS;
  float mass=.000511; // electron propagation
  PropagatorWithMaterial* prop1stLayer = new PropagatorWithMaterial(oppositeToMomentum,mass,&(*theMagField));
  PropagatorWithMaterial* prop2ndLayer = new PropagatorWithMaterial(alongMomentum,mass,&(*theMagField));

  float dphi1=0., dphi2=0., drz1=0., drz2=0.;
  float phi1=0., phi2=0., rz1=0., rz2=0.;

  for( ElectronSeedCollection::const_iterator MyS= (*elSeeds).begin(); MyS != (*elSeeds).end(); ++MyS) {

    LogDebug("") <<"\nSeed nr "<<is<<": ";
    range r=(*MyS).recHits();
     LogDebug("")<<" Number of RecHits= "<<(*MyS).nHits();
    const GeomDet *det1=0;const GeomDet *det2=0;

    TrajectorySeed::const_iterator it=r.first;
    DetId id1 = (*it).geographicalId();
    det1 = pDD->idToDet(id1);
    LogDebug("") <<" First hit local x,y,z "<<(*it).localPosition()<<" det "<<id1.det()<<" subdet "<<id1.subdetId();
    LogDebug("") <<" First hit global  "<<det1->toGlobal((*it).localPosition());
    //std::cout <<" First hit local x,y,z "<<(*it).localPosition()<<" det "<<id1.det()<<" subdet "<<id1.subdetId()<< std::endl;
    //std::cout <<" First hit global  "<<det1->toGlobal((*it).localPosition())<< std::endl;
    it++;
    DetId id2 = (*it).geographicalId();
    det2 = pDD->idToDet(id2);
    LogDebug("") <<" Second hit local x,y,z "<<(*it).localPosition()<<" det "<<id2.det()<<" subdet "<<id2.subdetId();
    LogDebug("") <<" Second hit global  "<<det2->toGlobal((*it).localPosition());
    //std::cout <<" Second hit local x,y,z "<<(*it).localPosition()<<" det "<<id2.det()<<" subdet "<<id2.subdetId()<< std::endl;
    //std::cout <<" Second hit global  "<<det2->toGlobal((*it).localPosition()) << std::endl;

    // state on last det
    const GeomDet *det=0;
    for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) det = pDD->idToDet(((*rhits)).geographicalId());
    TrajectoryStateOnSurface t=  trajectoryStateTransform::transientState((*MyS).startingState(), &(det->surface()), &(*theMagField));

    // debug

    LogDebug("")<<" ElectronSeed outermost state position: "<<t.globalPosition();
    LogDebug("")<<" ElectronSeed outermost state momentum: "<<t.globalMomentum();
    edm::RefToBase<CaloCluster> caloCluster = (*MyS).caloCluster() ;
    if (caloCluster.isNull()) continue;
    edm::Ref<SuperClusterCollection> theClus = caloCluster.castTo<SuperClusterRef>() ;
    LogDebug("")<<" ElectronSeed superCluster energy: "<<theClus->energy()<<", position: "<<theClus->position();
    LogDebug("")<<" ElectronSeed outermost state Pt: "<<t.globalMomentum().perp();
    LogDebug("")<<" ElectronSeed supercluster Et: "<<theClus->energy()*sin(2.*atan(exp(-theClus->position().eta())));
    LogDebug("")<<" ElectronSeed outermost momentum direction eta: "<<t.globalMomentum().eta();
    LogDebug("")<<" ElectronSeed supercluster eta: "<<theClus->position().eta();
    LogDebug("")<<" ElectronSeed seed charge: "<<(*MyS).getCharge();
    LogDebug("")<<" ElectronSeed E/p: "<<theClus->energy()/t.globalMomentum().mag();

    // retreive SC and compute distances between hit position and prediction the same
    // way as in the PixelHitMatcher
    
    // inputs are charge, cluster position, vertex position, cluster energy and B field
    int charge = int((*MyS).getCharge());
    GlobalPoint xmeas(theClus->position().x(),theClus->position().y(),theClus->position().z());
    GlobalPoint vprim(theBeamSpot->position().x(),theBeamSpot->position().y(),theBeamSpot->position().z());
    float energy = theClus->energy();

    FreeTrajectoryState fts = myFTS(&(*theMagField),xmeas, vprim,
				 energy, charge);
    //std::cout << "[PixelHitMatcher::compatibleSeeds] fts position, momentum " <<
    // fts.parameters().position() << " " << fts.parameters().momentum() << std::endl;

    PerpendicularBoundPlaneBuilder bpb;
    TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

    //      TrajectorySeed::range r=(*seeds.product())[i].recHits();
   // TrajectorySeed::range r=(*seeds)[i].recHits();

    // first Hit
    it=r.first;
    DetId id=(*it).geographicalId();
    const GeomDet *geomdet=pDD->idToDet((*it).geographicalId());
    LocalPoint lp=(*it).localPosition();
    GlobalPoint hitPos=geomdet->surface().toGlobal(lp);

    TrajectoryStateOnSurface tsos1;
    tsos1 = prop1stLayer->propagate(tsos,geomdet->surface()) ;

    if (tsos1.isValid()) {

      std::pair<bool,double> est;

      //UB add test on phidiff
      float SCl_phi = xmeas.phi();
      float localDphi = SCl_phi-hitPos.phi();
      if(localDphi>CLHEP::pi)localDphi-=(2*CLHEP::pi);
      if(localDphi<-CLHEP::pi)localDphi+=(2*CLHEP::pi);
      if(std::abs(localDphi)>2.5)continue;

      phi1 = hitPos.phi();
      dphi1 = hitPos.phi() - tsos1.globalPosition().phi();
      rz1 = hitPos.perp();
      drz1 = hitPos.perp() - tsos1.globalPosition().perp();
      if (id.subdetId()%2==1) {
	drz1 = hitPos.z() - tsos1.globalPosition().z();
	rz1 = hitPos.z();
      }

      // now second Hit
      it++;
      DetId id2=(*it).geographicalId();
      const GeomDet *geomdet2=pDD->idToDet((*it).geographicalId());
      TrajectoryStateOnSurface tsos2;

      // compute the z vertex from the cluster point and the found pixel hit
      double pxHit1z = hitPos.z();
      double pxHit1x = hitPos.x();
      double pxHit1y = hitPos.y();
      double r1diff = (pxHit1x-vprim.x())*(pxHit1x-vprim.x()) + (pxHit1y-vprim.y())*(pxHit1y-vprim.y());
      r1diff=sqrt(r1diff);
      double r2diff = (xmeas.x()-pxHit1x)*(xmeas.x()-pxHit1x) + (xmeas.y()-pxHit1y)*(xmeas.y()-pxHit1y);
      r2diff=sqrt(r2diff);
      double zVertexPred = pxHit1z - r1diff*(xmeas.z()-pxHit1z)/r2diff;

      GlobalPoint vertexPred(vprim.x(),vprim.y(),zVertexPred);

      FreeTrajectoryState fts2 = myFTS(&(*theMagField),hitPos,vertexPred,energy, charge);
      tsos2 = prop2ndLayer->propagate(fts2,geomdet2->surface()) ;

      if (tsos2.isValid()) {
	LocalPoint lp2=(*it).localPosition();
	GlobalPoint hitPos2=geomdet2->surface().toGlobal(lp2);
	phi2 = hitPos2.phi();
	dphi2 = hitPos2.phi() - tsos2.globalPosition().phi();
	rz2 = hitPos2.perp();
	drz2 = hitPos2.perp() - tsos2.globalPosition().perp();
    	if (id2.subdetId()%2==1) {
	  rz2 = hitPos2.z();
	  drz2 = hitPos2.z() - tsos2.globalPosition().z();
	}
      }

    }

    // fill the tree and histos

    histpt_->Fill(t.globalMomentum().perp());
    histetclu_->Fill(theClus->energy()*sin(2.*atan(exp(-theClus->position().eta()))));
    histeta_->Fill(t.globalMomentum().eta());
    histetaclu_->Fill(theClus->position().eta());
    histq_->Fill((*MyS).getCharge());
    histeoverp_->Fill(theClus->energy()/t.globalMomentum().mag());

    if (is<10) {
      superclusterEnergy[is] = theClus->energy();
      superclusterEta[is] = theClus->position().eta();
      superclusterPhi[is] = theClus->position().phi();
      superclusterEt[is] = theClus->energy()*sin(2.*atan(exp(-theClus->position().eta())));
      seedMomentum[is] = t.globalMomentum().mag();
      seedEta[is] = t.globalMomentum().eta();
      seedPhi[is] = t.globalMomentum().phi();
      seedPt[is] = t.globalMomentum().perp();
      seedQ[is] = (*MyS).getCharge();
      seedSubdet1[is] = id1.subdetId();
      switch (seedSubdet1[is]) {
        case 1:
	  {
	  
	  seedLayer1[is] = tTopo->pxbLayer( id1);
	  seedSide1[is] = 0;
	  break;
	  }
        case 2:
	{
	  
	  seedLayer1[is] = tTopo->pxfDisk( id1);
	  seedSide1[is] = tTopo->pxfSide( id1);
	  break;
	  }
        case 3:
	  {
	  
	  seedLayer1[is] = tTopo->tibLayer( id1);
	  seedSide1[is] = 0;
	  break;
	  }
        case 4:
	  {
	  
	  seedLayer1[is] = tTopo->tidWheel( id1);
	  seedSide1[is] = tTopo->tidSide( id1);
	  break;
	  }
        case 5:
	  {
	  
	  seedLayer1[is] = tTopo->tobLayer( id1);
	  seedSide1[is] = 0;
	  break;
	  }
        case 6:
	  {
	  
	  seedLayer1[is] = tTopo->tecWheel( id1);
	  seedSide1[is] = tTopo->tecSide( id1);
	  break;
	  }
      }
      seedPhi1[is] = phi1;
      seedRz1[is] = rz1;
      seedDphi1[is] = dphi1;
      seedDrz1[is] = drz1;
      seedSubdet2[is] = id2.subdetId();
      switch (seedSubdet2[is]) {
        case 1:
	  {
	  
	  seedLayer2[is] = tTopo->pxbLayer( id2);
	  seedSide2[is] = 0;
	  break;
	  }
        case 2:
	  {
	  
	  seedLayer2[is] = tTopo->pxfDisk( id2);
	  seedSide2[is] = tTopo->pxfSide( id2);
	  break;
	  }
        case 3:
	  {
	  
	  seedLayer2[is] = tTopo->tibLayer( id2);
	  seedSide2[is] = 0;
	  break;
	  }
        case 4:
	  {
	  
	  seedLayer2[is] = tTopo->tidWheel( id2);
	  seedSide2[is] = tTopo->tidSide( id2);
	  break;
	  }
        case 5:
	  {
	  
	  seedLayer2[is] = tTopo->tobLayer( id2);
	  seedSide2[is] = 0;
	  break;
	  }
        case 6:
	  {
	  
	  seedLayer2[is] = tTopo->tecWheel( id2);
	  seedSide2[is] = tTopo->tecSide( id2);
	  break;
	  }
      }
      seedDphi2[is] = dphi2;
      seedDrz2[is] = drz2;
      seedPhi2[is] = phi2;
      seedRz2[is] = rz2;
    }

    is++;

  }

  histnbseeds_->Fill(elSeeds.product()->size());

  // get input clusters

  edm::Handle<SuperClusterCollection> clusters;
  //CC to be changed according to supercluster input
  e.getByLabel("correctedHybridSuperClusters", clusters);
  histnbclus_->Fill(clusters.product()->size());
  if (clusters.product()->size()>0) histnrseeds_->Fill(elSeeds.product()->size());
  // get MC information

  edm::Handle<edm::HepMCProduct> HepMCEvt;
  // this one is empty branch in current test files
  //e.getByLabel("VtxSmeared", "", HepMCEvt);
  //e.getByLabel("source", "", HepMCEvt);
  e.getByLabel("generator", "", HepMCEvt);

  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
  HepMC::GenParticle* genPc=0;
  HepMC::FourVector pAssSim;
  int ip=0;
  for (HepMC::GenEvent::particle_const_iterator partIter = MCEvt->particles_begin();
   partIter != MCEvt->particles_end(); ++partIter) {

    for (HepMC::GenEvent::vertex_const_iterator vertIter = MCEvt->vertices_begin();
     vertIter != MCEvt->vertices_end(); ++vertIter) {

      //      CLHEP::HepLorentzVector creation = (*partIter)->CreationVertex();
      HepMC::GenVertex * creation = (*partIter)->production_vertex();
      //      CLHEP::HepLorentzVector momentum = (*partIter)->Momentum();
      HepMC::FourVector momentum = (*partIter)->momentum();
      //      HepPDT::ParticleID id = (*partIter)->particleID();  // electrons and positrons are 11 and -11
       int id = (*partIter)->pdg_id();  // electrons and positrons are 11 and -11
     LogDebug("")  << "MC particle id " << id << ", creationVertex " << (*creation) << " cm, initialMomentum " << momentum.rho() << " GeV/c" << std::endl;

      if (id == 11 || id == -11) {

      // single primary electrons or electrons from Zs or Ws
      HepMC::GenParticle* mother = 0;
      if ( (*partIter)->production_vertex() )  {
       if ( (*partIter)->production_vertex()->particles_begin(HepMC::parents) !=
           (*partIter)->production_vertex()->particles_end(HepMC::parents))
            mother = *((*partIter)->production_vertex()->particles_begin(HepMC::parents));
      }
      if ( ((mother == 0) || ((mother != 0) && (mother->pdg_id() == 23))
	                  || ((mother != 0) && (mother->pdg_id() == 32))
	                  || ((mother != 0) && (std::abs(mother->pdg_id()) == 24)))) {
      genPc=(*partIter);
      pAssSim = genPc->momentum();

      // EWK fiducial
      //if (pAssSim.perp()> 100. || std::abs(pAssSim.eta())> 2.5) continue;     
      //if (pAssSim.perp()< 20. || (std::abs(pAssSim.eta())> 1.4442 && std::abs(pAssSim.eta())< 1.56) || std::abs(pAssSim.eta())> 2.5) continue;
      // reconstruction fiducial
      //if (pAssSim.perp()< 5. || std::abs(pAssSim.eta())> 2.5) continue;
      if (std::abs(pAssSim.eta())> 2.5) continue;

      histptMC_->Fill(pAssSim.perp());
      histetaMC_->Fill(pAssSim.eta());
      histeMC_->Fill(pAssSim.rho());

     // looking for the best matching gsf electron
      bool okSeedFound = false;
      double seedOkRatio = 999999.;

      // find best matched seed
      reco::ElectronSeed bestElectronSeed;
      for( ElectronSeedCollection::const_iterator gsfIter= (*elSeeds).begin(); gsfIter != (*elSeeds).end(); ++gsfIter) {

        range r=gsfIter->recHits();
        const GeomDet *det=0;
        for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) det = pDD->idToDet(((*rhits)).geographicalId());
         TrajectoryStateOnSurface t= trajectoryStateTransform::transientState(gsfIter->startingState(), &(det->surface()), &(*theMagField));

	float eta = t.globalMomentum().eta();
	float phi = t.globalMomentum().phi();
	float p = t.globalMomentum().mag();
        double dphi = phi-pAssSim.phi();
        if (std::abs(dphi)>CLHEP::pi)
         dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
	double deltaR = sqrt(std::pow((eta-pAssSim.eta()),2) + std::pow(dphi,2));
	if ( deltaR < 0.15 ){
//	if ( deltaR < 0.3 ){
	//if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
	//(gsfIter->charge() > 0.) ){
	  double tmpSeedRatio = p/pAssSim.t();
	  if ( std::abs(tmpSeedRatio-1) < std::abs(seedOkRatio-1) ) {
	    seedOkRatio = tmpSeedRatio;
	    bestElectronSeed=*gsfIter;
	    okSeedFound = true;
	  }
	//}
	}
      } // loop over rec ele to look for the best one

      // analysis when the mc track is found
     if (okSeedFound){

	histptMCmatched_->Fill(pAssSim.perp());
	histetaMCmatched_->Fill(pAssSim.eta());
	histeMCmatched_->Fill(pAssSim.rho());
	if (ip<10) {
	  mcEnergy[ip] = pAssSim.rho();
	  mcEta[ip] = pAssSim.eta();
	  mcPhi[ip] = pAssSim.phi();
	  mcPt[ip] = pAssSim.perp();
	  mcQ[ip] = ((id == 11) ? -1.: +1.);
	}
      }
      
     // efficiency for ecal driven only
      okSeedFound = false;
      seedOkRatio = 999999.;

      // find best matched seed
      for( ElectronSeedCollection::const_iterator gsfIter= (*elSeeds).begin(); gsfIter != (*elSeeds).end(); ++gsfIter) {

        range r=gsfIter->recHits();
        const GeomDet *det=0;
        for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) det = pDD->idToDet(((*rhits)).geographicalId());
         TrajectoryStateOnSurface t= trajectoryStateTransform::transientState(gsfIter->startingState(), &(det->surface()), &(*theMagField));

	float eta = t.globalMomentum().eta();
	float phi = t.globalMomentum().phi();
	float p = t.globalMomentum().mag();
        double dphi = phi-pAssSim.phi();
        if (std::abs(dphi)>CLHEP::pi)
         dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
	double deltaR = sqrt(std::pow((eta-pAssSim.eta()),2) + std::pow(dphi,2));
	if (gsfIter->isEcalDriven()) {
	if ( deltaR < 0.15 ){
//	if ( deltaR < 0.3 ){
	//if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
	//(gsfIter->charge() > 0.) ){
	  double tmpSeedRatio = p/pAssSim.t();
	  if ( std::abs(tmpSeedRatio-1) < std::abs(seedOkRatio-1) ) {
	    seedOkRatio = tmpSeedRatio;
	    bestElectronSeed=*gsfIter;
	    okSeedFound = true;
	  }
	//}
	}
	} // end if ecal driven
      } // loop over rec ele to look for the best one

      // analysis when the mc track is found
     if (okSeedFound){

	histecaldrivenptMCmatched_->Fill(pAssSim.perp());
	histecaldrivenetaMCmatched_->Fill(pAssSim.eta());
	histecaldriveneMCmatched_->Fill(pAssSim.rho());
      }

      // efficiency for tracker driven only
      okSeedFound = false;
      seedOkRatio = 999999.;

      // find best matched seed
      for( ElectronSeedCollection::const_iterator gsfIter= (*elSeeds).begin(); gsfIter != (*elSeeds).end(); ++gsfIter) {

        range r=gsfIter->recHits();
        const GeomDet *det=0;
        for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) det = pDD->idToDet(((*rhits)).geographicalId());
         TrajectoryStateOnSurface t= trajectoryStateTransform::transientState(gsfIter->startingState(), &(det->surface()), &(*theMagField));

	float eta = t.globalMomentum().eta();
	float phi = t.globalMomentum().phi();
	float p = t.globalMomentum().mag();
        double dphi = phi-pAssSim.phi();
        if (std::abs(dphi)>CLHEP::pi)
         dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
	double deltaR = sqrt(std::pow((eta-pAssSim.eta()),2) + std::pow(dphi,2));
	if (gsfIter->isTrackerDriven()) {
	if ( deltaR < 0.15 ){
//	if ( deltaR < 0.3 ){
	//if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
	//(gsfIter->charge() > 0.) ){
	  double tmpSeedRatio = p/pAssSim.t();
	  if ( std::abs(tmpSeedRatio-1) < std::abs(seedOkRatio-1) ) {
	    seedOkRatio = tmpSeedRatio;
	    bestElectronSeed=*gsfIter;
	    okSeedFound = true;
	  }
	//}
	}
	} // end if ecal driven
      } // loop over rec ele to look for the best one

      // analysis when the mc track is found
     if (okSeedFound){

	histtrackerdrivenptMCmatched_->Fill(pAssSim.perp());
	histtrackerdrivenetaMCmatched_->Fill(pAssSim.eta());
	histtrackerdriveneMCmatched_->Fill(pAssSim.rho());
      }

     } // end if mother W or Z

    } // end if gen part is electron

    } // end loop on vertices

    ip++;

  } // end loop on gen particles

  //tree_->Fill();

}


