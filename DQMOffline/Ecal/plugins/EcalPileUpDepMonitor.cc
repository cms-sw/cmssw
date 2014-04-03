/*
 * \file EcalPileUpDepMonitor.cc
 * \author Ben Carlson - CMU
 * Last Update:
 *
 */

#include "DQMOffline/Ecal/interface/EcalPileUpDepMonitor.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

// Framework

const static int XBINS=2000;

EcalPileUpDepMonitor::EcalPileUpDepMonitor(const edm::ParameterSet& ps)
{
  VertexCollection_ = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("VertexCollection"));

  if(ps.existsAs<edm::InputTag>("basicClusterCollection") && ps.getParameter<edm::InputTag>("basicClusterCollection").label() != "")
    basicClusterCollection_ = consumes<edm::View<reco::CaloCluster> >(ps.getParameter<edm::InputTag>("basicClusterCollection"));
  else{
    basicClusterCollection_EE_ = consumes<edm::View<reco::CaloCluster> >(ps.getParameter<edm::InputTag>("basicClusterCollection_EE"));
    basicClusterCollection_EB_ = consumes<edm::View<reco::CaloCluster> >(ps.getParameter<edm::InputTag>("basicClusterCollection_EB"));
  }
  superClusterCollection_EB_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("superClusterCollection_EB"));
  superClusterCollection_EE_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("superClusterCollection_EE"));
	
  RecHitCollection_EB_       = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("RecHitCollection_EB"));
  RecHitCollection_EE_       = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("RecHitCollection_EE"));
  EleTag_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("EleTag")); 
}

EcalPileUpDepMonitor::~EcalPileUpDepMonitor()
{
}

void EcalPileUpDepMonitor::bookHistograms(DQMStore::IBooker & ibooker,
                               edm::Run const&,
                               edm::EventSetup const& eventSetup)
{
  eventSetup.get<CaloGeometryRecord>().get(geomH);
  eventSetup.get<CaloTopologyRecord>().get(caloTop);

  ibooker.cd();
  ibooker.setCurrentFolder("Ecal/EcalPileUpDepMonitor");

  std::string prof_name="bcEB_PV"; 
  std::string title="Basic clusters EB vs. PV";
  bcEB_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  bcEB_PV->setAxisTitle("N_{pv}",1); 
  bcEB_PV->setAxisTitle("Basic Clusters", 2); 

  prof_name="bcEE_PV"; 
  title="Basic Clusters EE vs. PV"; 
  bcEE_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  bcEE_PV->setAxisTitle("N_{pv}",1); 
  bcEE_PV->setAxisTitle("Basic Clusters", 2); 
	
  prof_name="scEB_PV"; 
  title="Super Clusters EB vs. PV"; 
  scEB_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  scEB_PV->setAxisTitle("N_{pv}",1);
  scEB_PV->setAxisTitle("Super Clusters", 2); 
	
  prof_name="scEE_PV"; 
  title="Super Clusters EE vs. PV"; 
  scEE_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  scEE_PV->setAxisTitle("N_{pv}",1);
  scEE_PV->setAxisTitle("Super Clusters", 2); 
	
  prof_name="scEtEB_PV"; 
  title="Super Clusters Et EB vs. PV"; 
  scEtEB_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  scEtEB_PV->setAxisTitle("N_{pv}",1);
  scEtEB_PV->setAxisTitle("Super Cluster E_{T} [GeV]", 2); 
	
	
  prof_name="scEtEE_PV"; 
  title="Super Clusters Et EE vs. PV"; 
  scEtEE_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  scEtEE_PV->setAxisTitle("N_{pv}",1);
  scEtEE_PV->setAxisTitle("Super Cluster E_{T} [GeV]", 2); 
	
  prof_name="recHitEtEB_PV"; 
  title="Reconstructed Hit Et EB vs. PV"; 
  recHitEtEB_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  recHitEtEB_PV->setAxisTitle("N_{pv}",1);
  recHitEtEB_PV->setAxisTitle("Reconstructed hit E_{T} [GeV]", 2); 

  prof_name="recHitEtEE_PV"; 
  title="Reconstructed Hit Et EE vs. PV"; 
  recHitEtEE_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350.);
  recHitEtEE_PV->setAxisTitle("N_{pv}",1);
  recHitEtEE_PV->setAxisTitle("Reconstructed hit E_{T} [GeV]", 2); 
	
  prof_name="emIso_PV"; 
  title="EM Isolation vs. PV"; 
  emIso_PV=ibooker.bookProfile(prof_name,title,50,0.,50.,50,0.,350);
  emIso_PV->setAxisTitle("N_{pv}",1);
  emIso_PV->setAxisTitle("EM_{Isolation} [GeV]", 2); 
	
  prof_name="emIso"; 
  title="EM Isolation"; 
  emIso=ibooker.book1D(prof_name,title,50,0,50);
  emIso->setAxisTitle("EM_{Isolation} [GeV]",1); 
  emIso->setAxisTitle("Events",2); 
	
  prof_name="scHitEtEB"; 
  title="Super Cluster Hit Et EB"; 
  scHitEtEB=ibooker.book1D(prof_name,title,50,0,100);
  scHitEtEB->setAxisTitle("super cluster hit E_{T} [GeV]",1); 
  scHitEtEB->setAxisTitle("Events",2); 
	
  prof_name="scHitEtEE"; 
  title="Super Cluster Hit Et EE"; 
  scHitEtEE=ibooker.book1D(prof_name,title,50,0,100);
  scHitEtEE->setAxisTitle("super cluster hit E_{T} [GeV]",1); 
  scHitEtEE->setAxisTitle("Events",2); 

	
//   prof_name="scHitE_EB"; 
//   title="Super Cluster Hit E EB"; 
//   scHitE_EB=ibooker.book1D(prof_name,title,50,0,100);
//   scHitE_EB->setAxisTitle("super cluster hit E [GeV]",1);
//   scHitE_EB->setAxisTitle("Events",2); 

	
//   prof_name="scHitE_EE"; 
//   title="Super Cluster Hit E EE"; 
//   scHitE_EE=ibooker.book1D(prof_name,title,50,0,100);
//   scHitE_EE->setAxisTitle("super cluster hit E [GeV]",1);
//   scHitE_EE->setAxisTitle("Events",2); 
	
  //SC eta
//   prof_name="scEta_EB"; 
//   title="Super Cluster #eta EB"; 
//   scEta_EB=ibooker.book1D(prof_name,title,50,-6,6);
//   scEta_EB->setAxisTitle("#eta",1); 
//   scEta_EB->setAxisTitle("Events",2); 
	
//   prof_name="scEta_EE"; 
//   title="Super Cluster #eta EE"; 
//   scEta_EE=ibooker.book1D(prof_name,title,50,-6,6);
//   scEta_EE->setAxisTitle("#eta",1); 
//   scEta_EE->setAxisTitle("Events",2); 
	
  //SC phi
//   prof_name="scPhi_EB"; 
//   title="Super Cluster #phi EB"; 
//   scPhi_EB=ibooker.book1D(prof_name,title,50,-3.14,3.14);
//   scPhi_EB->setAxisTitle("super cluster #phi",1); 
//   scPhi_EB->setAxisTitle("Events",2); 
	
//   prof_name="scPhi_EE"; 
//   title="Super Cluster #phi EE"; 
//   scPhi_EE=ibooker.book1D(prof_name,title,50,-3.14,3.14);
//   scPhi_EE->setAxisTitle("super cluster #phi",1); 
//   scPhi_EE->setAxisTitle("Events",2); 
	
  //sc sigma eta eta / eta phi
	
  prof_name="scSigmaIetaIeta_EB"; 
  title="Super Cluster sigmaIetaIeta EB"; 
  scSigmaIetaIeta_EB=ibooker.book1D(prof_name,title,50,0,0.03);
  scSigmaIetaIeta_EB->setAxisTitle("#sigma_{i#etai#eta}",1); 
  scSigmaIetaIeta_EB->setAxisTitle("Events",2); 
	
  prof_name="scSigmaIetaIeta_EE"; 
  title="Super Cluster sigmaIetaIeta EE"; 
  scSigmaIetaIeta_EE=ibooker.book1D(prof_name,title,50,0,0.1);
  scSigmaIetaIeta_EE->setAxisTitle("#sigma_{i#etai#eta}",1); 
  scSigmaIetaIeta_EE->setAxisTitle("Events",2); 
	
  //phi
  prof_name="scSigmaIetaIphi_EB"; 
  title="Super Cluster sigmaIetaIphi EB"; 
  scSigmaIetaIphi_EB=ibooker.book1D(prof_name,title,50,-5.e-4,5.e-4);
  scSigmaIetaIphi_EB->setAxisTitle("#sigma_{i#etai#phi}",1); 
  scSigmaIetaIphi_EB->setAxisTitle("Events",2); 
	
  prof_name="scSigmaIetaIphi_EE"; 
  title="Super Cluster sigmaIetaIphi EE"; 
  scSigmaIetaIphi_EE=ibooker.book1D(prof_name,title,50,-2.5e-3,2.5e-3);
  scSigmaIetaIphi_EE->setAxisTitle("#sigma_{i#etai#phi}",1); 
  scSigmaIetaIphi_EE->setAxisTitle("Events",2); 
	
  //R9
  prof_name="r9_EB"; 
  title="r9 EB"; 
  r9_EB=ibooker.book1D(prof_name,title,50,0,1.5);
  r9_EB->setAxisTitle("R_{9}",1);
  r9_EB->setAxisTitle("Events",2); 
	
  prof_name="r9_EE"; 
  title="r9 EE"; 
  r9_EE=ibooker.book1D(prof_name,title,50,0,1.5);
  r9_EE->setAxisTitle("R_{9}",1);
  r9_EE->setAxisTitle("Events",2); 

  //Rec Hit
	
  prof_name="recHitEtEB"; 
  title="RecHit Et EB"; 
  recHitEtEB=ibooker.book1D(prof_name,title,50,0,400);
  recHitEtEB->setAxisTitle("Reconstructed Hit E_{T} [GeV]",1);
  recHitEtEB->setAxisTitle("Events",2); 
	
  prof_name="recHitEtEE"; 
  title="RecHit Et EE"; 
  recHitEtEE=ibooker.book1D(prof_name,title,50,0,400);
  recHitEtEE->setAxisTitle("Reconstructed Hit E_{T} [GeV]",1);
  recHitEtEE->setAxisTitle("Events",2); 
}

void EcalPileUpDepMonitor::analyze(const edm::Event& e, const edm::EventSetup&)
{
  //Vertex collection: 
  //-----------------------------------------
  edm::Handle<reco::VertexCollection> PVCollection_h;
  e.getByToken(VertexCollection_,PVCollection_h);
  if ( ! PVCollection_h.isValid() ) {
    edm::LogWarning("VertexCollection") << "VertexCollection not found"; 
  }
  //-----------------gsfElectrons -------------------------
  edm::Handle<reco::GsfElectronCollection> electronCollection_h; 
  e.getByToken(EleTag_, electronCollection_h); 
  if( !electronCollection_h.isValid()){
    edm::LogWarning("EBRecoSummary") << "Electrons not found"; 
  }

  if(basicClusterCollection_.isUninitialized()){
    //----------------- Basic Cluster Collection Ecal Barrel  ---------
    edm::Handle<edm::View<reco::CaloCluster> > basicClusters_EB_h;
    e.getByToken( basicClusterCollection_EB_, basicClusters_EB_h );
    if ( ! basicClusters_EB_h.isValid() ) {
      edm::LogWarning("EBRecoSummary") << "basicClusters_EB_h not found"; 
    }
	
    bcEB_PV->Fill(PVCollection_h->size(), basicClusters_EB_h->size()); 
	
    //----------------- Basic Cluster Collection Ecal Endcal  ---------

    edm::Handle<edm::View<reco::CaloCluster> > basicClusters_EE_h;
    e.getByToken( basicClusterCollection_EE_, basicClusters_EE_h );
    if ( ! basicClusters_EE_h.isValid() ) {
      edm::LogWarning("EERecoSummary") << "basicClusters_EE_h not found"; 
    }

    bcEE_PV->Fill(PVCollection_h->size(), basicClusters_EE_h->size()); 
  }
  else{
    //----------------- Basic Cluster Collection Ecal Barrel  ---------
    edm::Handle<edm::View<reco::CaloCluster> > basicClusters_h;
    e.getByToken( basicClusterCollection_, basicClusters_h );
    if ( ! basicClusters_h.isValid() ) {
      edm::LogWarning("EBRecoSummary") << "basicClusters_h not found"; 
    }

    int nBarrel(0);
    int nEndcap(0);
    for(edm::View<reco::CaloCluster>::const_iterator bcItr(basicClusters_h->begin()); bcItr != basicClusters_h->end(); ++bcItr){
      if(bcItr->caloID().detector(reco::CaloID::DET_ECAL_BARREL)) ++nBarrel;
      if(bcItr->caloID().detector(reco::CaloID::DET_ECAL_ENDCAP)) ++nEndcap;
    }
	
    bcEB_PV->Fill(PVCollection_h->size(), nBarrel); 
    bcEE_PV->Fill(PVCollection_h->size(), nEndcap); 
  }

  //----------------- Reconstructed Hit Ecal barrel 
	
  edm::Handle<EcalRecHitCollection> RecHitsEB;
  e.getByToken( RecHitCollection_EB_,RecHitsEB );
  if ( ! RecHitsEB.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "RecHitsEB not found"; 
  }
	
  //----------------- Reconstructed Hit Ecal Endcap  
	
  edm::Handle<EcalRecHitCollection> RecHitsEE;
  e.getByToken( RecHitCollection_EE_, RecHitsEE );
  if ( ! RecHitsEE.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "RecHitsEB not found"; 
  }
  //----------------- Super Cluster Collection Ecal Endcap  ---------
	
  edm::Handle<reco::SuperClusterCollection> superClusters_EE_h;
  e.getByToken( superClusterCollection_EE_, superClusters_EE_h );
  if ( ! superClusters_EE_h.isValid() ) {
    edm::LogWarning("EERecoSummary") << "superClusters_EE_h not found"; 
  }
	
  //--------- Fill Isolation -----------------
	
  if(electronCollection_h.isValid()){
    for (reco::GsfElectronCollection::const_iterator recoElectron = electronCollection_h->begin ();
         recoElectron != electronCollection_h->end ();
         recoElectron++) {
        double IsoEcal =recoElectron->dr03EcalRecHitSumEt();///recoElectron->et()
        emIso_PV->Fill(PVCollection_h->size(),IsoEcal);
        emIso->Fill(IsoEcal);
      }		
  }

  //fill super clusters EE
  scEE_PV->Fill(PVCollection_h->size(), superClusters_EE_h->size()); 
	
  for (reco::SuperClusterCollection::const_iterator itSC = superClusters_EE_h->begin(); 
       itSC != superClusters_EE_h->end();
       ++itSC ) {
    double scEE_Et= itSC -> energy() * sin(2.*atan( exp(- itSC->position().eta() )));
    //    double scEE_E=itSC->energy();
		
    //fill super cluster endcap eta/phi
//     scEta_EE->Fill(itSC->position().eta());
//     scPhi_EE->Fill(itSC->position().phi());

    //get sigma eta_eta etc
		
    CaloTopology const* p_topology = caloTop.product();//get calo topology
    const EcalRecHitCollection* eeRecHits = RecHitsEE.product();
		
    reco::BasicCluster const& seedCluster(*itSC->seed());
    std::vector<float> cov = EcalClusterTools::localCovariances(seedCluster, eeRecHits, p_topology);
    float sigmaIetaIeta = std::sqrt(cov[0]);
    float sigmaIetaIphi = cov[1];

		
    float e3x3 = EcalClusterTools::e3x3(seedCluster, eeRecHits, p_topology);
    float r9 = e3x3 / itSC->energy();
		
    r9_EE->Fill(r9); 
    scSigmaIetaIeta_EE->Fill(sigmaIetaIeta);
    scSigmaIetaIphi_EE->Fill(sigmaIetaIphi);
		
    //std::cout  << " sigmaIetaIeta: " << sigmaIetaIeta << std::endl; 
    scEtEE_PV->Fill(PVCollection_h->size(),scEE_Et); 
    scHitEtEE->Fill(scEE_Et); //super cluster Et historam 
    //    scHitE_EE->Fill(scEE_E); //super cluster energy histogram
		
  }//sc-EE loop

	
  //----------------- Super Cluster Collection Ecal Barrel  ---------

  edm::Handle<reco::SuperClusterCollection> superClusters_EB_h;
  e.getByToken( superClusterCollection_EB_, superClusters_EB_h );
  if ( ! superClusters_EB_h.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "superClusters_EB_h not found"; 
  }
  scEB_PV->Fill(PVCollection_h->size(), superClusters_EB_h->size()); 

  for (reco::SuperClusterCollection::const_iterator itSC = superClusters_EB_h->begin(); 
       itSC != superClusters_EB_h->end();
       ++itSC ) {
    double scEB_Et= itSC -> energy() * sin(2.*atan( exp(- itSC->position().eta() ))); // super cluster transverse energy
    //    double scEB_E= itSC->energy(); // super cluster energy
		
    //fill super cluster Barrel eta/phi
//     scEta_EB->Fill(itSC->position().eta()); //super cluster eta
//     scPhi_EB->Fill(itSC->position().phi()); // super cluster phi
		
    //sigma ietaieta etc 
		
    CaloTopology const* p_topology = caloTop.product();//get calo topology
    const EcalRecHitCollection* ebRecHits = RecHitsEB.product();

    reco::BasicCluster const& seedCluster(*itSC->seed());
    std::vector<float> cov = EcalClusterTools::localCovariances(seedCluster, ebRecHits, p_topology);
    float sigmaIetaIeta = std::sqrt(cov[0]);
    float sigmaIetaIphi = cov[1];

		
    float e3x3 = EcalClusterTools::e3x3(seedCluster, ebRecHits, p_topology);
    float r9 = e3x3 / itSC->energy();
		
    r9_EB->Fill(r9);
    scSigmaIetaIeta_EB->Fill(sigmaIetaIeta);
    scSigmaIetaIphi_EB->Fill(sigmaIetaIphi);
		
    scEtEB_PV->Fill(PVCollection_h->size(),scEB_Et); 
    scHitEtEB->Fill(scEB_Et); 
    //    scHitE_EB->Fill(scEB_E); 
  }//sc-EB loop
	

  //-------------------Compute scalar sum of reconstructed hit Et
  double RecHitEt_EB=0; 

  for ( EcalRecHitCollection::const_iterator itr = RecHitsEB->begin () ;
        itr != RecHitsEB->end () ;
        ++itr) {	
      //RecHitEt_EB +=itr->energy();
		
      GlobalPoint const& position  = geomH->getGeometry(itr->detid())->getPosition();
      RecHitEt_EB += itr -> energy() * sin(position.theta()) ;
    }//EB Rec Hit
	
  recHitEtEB->Fill(RecHitEt_EB); 
  recHitEtEB_PV->Fill(PVCollection_h->size(),RecHitEt_EB); 

	
  //-------------------Compute scalar sum of reconstructed hit Et
  double RecHitEt_EE=0; 

  for ( EcalRecHitCollection::const_iterator itr = RecHitsEE->begin () ;
        itr != RecHitsEE->end () ;
        ++itr) {
      GlobalPoint const& position  = geomH->getGeometry(itr->detid())->getPosition();
      RecHitEt_EE += itr -> energy() * sin(position.theta()) ;
    }//EB Rec Hit
	
  recHitEtEE->Fill(RecHitEt_EE); 
  recHitEtEE_PV->Fill(PVCollection_h->size(),RecHitEt_EE); 
}

void
EcalPileUpDepMonitor::endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&)
{
}

DEFINE_FWK_MODULE(EcalPileUpDepMonitor);

