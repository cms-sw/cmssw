// -*- C++ -*-
//
// Package:     RecoEgamma/Examples
// Class  :     SiStripElectronAnalyzer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:49:38 EDT 2006
// $Id: SiStripElectronAnalyzer.cc,v 1.14 2013/01/02 20:41:37 dlange Exp $
//

// system include files
#include <memory>

// user include files
#include "RecoEgamma/Examples/plugins/SiStripElectronAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"

// for Si hits
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
SiStripElectronAnalyzer::SiStripElectronAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  fileName_ = iConfig.getParameter<std::string>("fileName");

  file_ = new TFile(fileName_.c_str(), "RECREATE");
  numCand_ = new TH1F("numCandidates", "Number of candidates found", 10, -0.5, 9.5);
  numElectrons_ = new TH1F("numElectrons", "Number of Electrons found", 10, -0.5, 9.5);
  numSuperClusters_ = new TH1F("numSuperClusters","Number of Ecal SuperClusters", 50, 0, 50);


  energySuperClusters_= new TH1F("energySuperClusters","Super Cluster Energy - all ", 200 , 0, 2000.);
  energySuperClustersEl_= new TH1F("energySuperClustersEl","Super Cluster Energy - Electron Cands ", 200, 0., 2000.);


  sizeSuperClusters_= new TH1F("sizeSuperClusters","Super Cluster Size - all ", 20, 0, 19);
  sizeSuperClustersEl_= new TH1F("sizeSuperClustersEl","Super Cluster Size - Electron Cands ", 20, 0, 19);

  emaxSuperClusters_= new TH1F("emaxSuperClusters","Super Cluster Emax - all ", 200, 0, 2000.);
  emaxSuperClustersEl_= new TH1F("emaxSuperClustersEl","Super Cluster Emax - Electron Cands ", 200, 0, 2000.);

  phiWidthSuperClusters_ = new TH1F("phiWidthSuperClusters", "Super Cluster Width - all ",20,  0., 0.05 );
  phiWidthSuperClustersEl_ = new TH1F("phiWidthSuperClustersEl", "Super Cluster Width - Electron Cands ", 20 , 0., 0.05 );





  ptDiff = new TH1F("ptDiff"," ptDiff ", 20, -10.,10.);
  pDiff = new TH1F("pDiff"," pDiff ", 100, -50.,50.);


  pElectronFailed  = new TH1F("pElectronFailed"," pElectronFailed ", 55, 0.,110.);
  ptElectronFailed  = new TH1F("ptElectronFailed"," ptElectronFailed ", 55, 0.,110.);


  pElectronPassed  = new TH1F("pElectronPassed"," pElectronPassed ", 55, 0.,110.);
  ptElectronPassed  = new TH1F("ptElectronPassed"," ptElectronPassed ", 55, 0.,110.);


  sizeSuperClustersFailed= new TH1F("sizeSuperClustersFailed","Super Cluster Size - Failed ", 20, 0, 19);
  sizeSuperClustersPassed= new TH1F("sizeSuperClustersPassed","Super Cluster Size - Passed ", 20, 0, 19);


  energySuperClustersPassed= new TH1F("energySuperClustersPassed","Super Cluster Energy - Passed ", 125, 0, 250.);
  energySuperClustersFailed= new TH1F("energySuperClustersFailed","Super Cluster Energy - Failed ", 125, 0, 250.);


  eOverPFailed = new TH1F("eOverPFailed"," E over P - Failed ", 50, 0, 10.) ;
  eOverPPassed = new TH1F("eOverPPassed"," E over P - Passed ", 50, 0, 10.) ;




  numSiStereoHits_ = new TH1F("numSiStereoHits","Number of Si StereoHits",100,0,1000);
  numSiMonoHits_ = new TH1F("numSiMonoHits","Number of Si MonoHits",100,0,1000);
  numSiMatchedHits_ = new TH1F("numSiMatchedHits","Number of Si MatchedHits",100,0,1000);



  /////////////////////////////////////////////////////



  mctruthProducer_ = iConfig.getParameter<std::string>("mctruthProducer");
  mctruthCollection_ = iConfig.getParameter<std::string>("mctruthCollection");

  superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
  superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");

  eBRecHitProducer_ = iConfig.getParameter<std::string>("recHitProducer");
  eBRecHitCollection_ = iConfig.getParameter<std::string>("recHitCollection");

  siElectronProducer_ = iConfig.getParameter<std::string>("siElectronProducer");
  siElectronCollection_ = iConfig.getParameter<std::string>("siElectronCollection");

  electronProducer_ = iConfig.getParameter<std::string>("electronProducer");
  electronCollection_ = iConfig.getParameter<std::string>("electronCollection");

  siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
  siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
  siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");
  siMatchedHitCollection_ = iConfig.getParameter<std::string>("siMatchedHitCollection");

}

// SiStripElectronAnalyzer::SiStripElectronAnalyzer(const SiStripElectronAnalyzer& rhs)
// {
//    // do actual copying here;
// }

SiStripElectronAnalyzer::~SiStripElectronAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  file_->Write();
  file_->Close();
}

//
// assignment operators
//
// const SiStripElectronAnalyzer& SiStripElectronAnalyzer::operator=(const SiStripElectronAnalyzer& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectronAnalyzer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
// init for TTree
void SiStripElectronAnalyzer::beginJob(){

  myTree_ = new TTree("myTree","my first Tree example");

  myTree_->Branch("NShowers",&NShowers_,"NShowers/I");


  // first specify the ECAL clusters
  // need to explicitly include array length.
  myTree_->Branch("EShower",&EShower_,"EShower[1000]/F");
  myTree_->Branch("XShower",&XShower_,"XShower[1000]/F");
  myTree_->Branch("YShower",&YShower_,"YShower[1000]/F");
  myTree_->Branch("ZShower",&ZShower_,"ZShower[1000]/F");

  // second specify the Si Stereo Hits
  myTree_->Branch("NStereoHits",&NStereoHits_,"NStereoHits/I");
  myTree_->Branch("StereoHitX",&StereoHitX_,"StereoHitX[1000]/F");
  myTree_->Branch("StereoHitY",&StereoHitY_,"StereoHitY[1000]/F");
  myTree_->Branch("StereoHitZ",&StereoHitZ_,"StereoHitZ[1000]/F");

  myTree_->Branch("StereoHitR",&StereoHitR_,"StereoHitR[1000]/F");
  myTree_->Branch("StereoHitPhi",&StereoHitPhi_,"StereoHitPhi[1000]/F");
  myTree_->Branch("StereoHitTheta",&StereoHitTheta_,"StereoHitTheta[1000]/F");

  myTree_->Branch("StereoHitSigX",&StereoHitSigX_,"StereoHitSigX[1000]/F");
  myTree_->Branch("StereoHitSigY",&StereoHitSigY_,"StereoHitSigY[1000]/F");
  myTree_->Branch("StereoHitCorr",&StereoHitCorr_,"StereoHitCorr[1000]/F");

  myTree_->Branch("StereoHitSignal",&StereoHitSignal_,"StereoHitSignal[1000]/F");
  myTree_->Branch("StereoHitNoise",&StereoHitNoise_,"StereoHitNoise[1000]/F");
  myTree_->Branch("StereoHitWidth",&StereoHitWidth_,"StereoHitWidth[1000]/I");

  myTree_->Branch("StereoDetector",&StereoDetector_,"StereoDetector[1000]/I");
  myTree_->Branch("StereoLayer",&StereoLayer_,"StereoLayer[1000]/I");


  // specify the Si mono (rphi) hits
  myTree_->Branch("NMonoHits",&NMonoHits_,"NMonoHits/I");
  myTree_->Branch("MonoHitX",&MonoHitX_,"MonoHitX[1000]/F");
  myTree_->Branch("MonoHitY",&MonoHitY_,"MonoHitY[1000]/F");
  myTree_->Branch("MonoHitZ",&MonoHitZ_,"MonoHitZ[1000]/F");


  myTree_->Branch("MonoHitR",&MonoHitR_,"MonoHitR[1000]/F");
  myTree_->Branch("MonoHitPhi",&MonoHitPhi_,"MonoHitPhi[1000]/F");
  myTree_->Branch("MonoHitTheta",&MonoHitTheta_,"MonoHitTheta[1000]/F");

  myTree_->Branch("MonoHitSigX",&MonoHitSigX_,"MonoHitSigX[1000]/F");
  myTree_->Branch("MonoHitSigY",&MonoHitSigY_,"MonoHitSigY[1000]/F");
  myTree_->Branch("MonoHitCorr",&MonoHitCorr_,"MonoHitCorr[1000]/F");

  myTree_->Branch("MonoHitSignal",&MonoHitSignal_,"MonoHitSignal[1000]/F");
  myTree_->Branch("MonoHitNoise",&MonoHitNoise_,"MonoHitNoise[1000]/F");
  myTree_->Branch("MonoHitWidth",&MonoHitWidth_,"MonoHitWidth[1000]/I");

  myTree_->Branch("MonoDetector",&MonoDetector_,"MonoDetector[1000]/I");
  myTree_->Branch("MonoLayer",&MonoLayer_,"MonoLayer[1000]/I");

  // specify the Si matched (rphi) hits
  myTree_->Branch("NMatchedHits",&NMatchedHits_,"NMatchedHits/I");
  myTree_->Branch("MatchedHitX",&MatchedHitX_,"MatchedHitX[1000]/F");
  myTree_->Branch("MatchedHitY",&MatchedHitY_,"MatchedHitY[1000]/F");
  myTree_->Branch("MatchedHitZ",&MatchedHitZ_,"MatchedHitZ[1000]/F");

  myTree_->Branch("MatchedHitR",&MatchedHitR_,"MatchedHitR[1000]/F");
  myTree_->Branch("MatchedHitPhi",&MatchedHitPhi_,"MatchedHitPhi[1000]/F");
  myTree_->Branch("MatchedHitTheta",&MatchedHitTheta_,"MatchedHitTheta[1000]/F");

  myTree_->Branch("MatchedHitSigX",&MatchedHitSigX_,"MatchedHitSigX[1000]/F");
  myTree_->Branch("MatchedHitSigY",&MatchedHitSigY_,"MatchedHitSigY[1000]/F");
  myTree_->Branch("MatchedHitCorr",&MatchedHitCorr_,"MatchedHitCorr[1000]/F");

  myTree_->Branch("MatchedHitSignal",&MatchedHitSignal_,"MatchedHitSignal[1000]/F");
  myTree_->Branch("MatchedHitNoise",&MatchedHitNoise_,"MatchedHitNoise[1000]/F");
  myTree_->Branch("MatchedHitWidth",&MatchedHitWidth_,"MatchedHitWidth[1000]/I");

  myTree_->Branch("MatchedDetector",&MatchedDetector_,"MatchedDetector[1000]/I");
  myTree_->Branch("MatchedLayer",&MatchedLayer_,"MatchedLayer[1000]/I");

}

void SiStripElectronAnalyzer::initNtuple(){

  LogDebug("") << " In initNtuple " ;

  NShowers_ = -999 ;
  for (int init = 0 ; init < myMaxHits ; ++init){
    EShower_[init] = -999.;
    XShower_[init] = -999.;
    YShower_[init] = -999.;
    ZShower_[init] = -999.;
  }
  NStereoHits_ = -999 ;

  for (int init = 0 ; init < myMaxHits ; ++init){
    StereoHitX_[init] = -999.;
    StereoHitY_[init] = -999.;
    StereoHitZ_[init] = -999.;
    StereoHitR_[init] = -999.;
    StereoHitPhi_[init] = -999.;
    StereoHitTheta_[init] = -999.;

    StereoHitSignal_[init] = -999.;
    StereoHitNoise_[init] = -999.;
    StereoHitWidth_[init] = -999 ;;
  }

  NMonoHits_ = -999 ;
  for (int init = 0 ; init < myMaxHits ; ++init){
    MonoHitX_[init] = -999.;
    MonoHitY_[init] = -999.;
    MonoHitZ_[init] = -999.;
    MonoHitR_[init] = -999.;
    MonoHitPhi_[init] = -999.;
    MonoHitTheta_[init] = -999.;

    MonoHitSignal_[init] = -999.;
    MonoHitNoise_[init] = -999.;
    MonoHitWidth_[init] = -999 ;;
  }

  NMatchedHits_ = -999 ;
  for (int init = 0 ; init < myMaxHits ; ++init){
    MatchedHitX_[init] = -999.;
    MatchedHitY_[init] = -999.;
    MatchedHitZ_[init] = -999.;
    MatchedHitR_[init] = -999.;
    MatchedHitPhi_[init] = -999.;
    MatchedHitTheta_[init] = -999.;

    MatchedHitSignal_[init] = -999.;
    MatchedHitNoise_[init] = -999.;
    MatchedHitWidth_[init] = -999 ;;
  }



}

// ------------ method called to produce the data  ------------
void
SiStripElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);


  using namespace std;  // so you can say "cout" and "endl"

  initNtuple();


  // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/clhep/CLHEP/HepMC/GenParticle.h
  // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/clhep/CLHEP/HepMC/GenVertex.h
  // removed by JED - causes trouble in release post 0_9_0
  //   edm::Handle<edm::HepMCProduct> mctruthHandle;
  //   iEvent.getByLabel(mctruthProducer_, mctruthCollection_, mctruthHandle);
  //   HepMC::GenEvent mctruth = mctruthHandle->getHepMCData();

  //   for (HepMC::GenEvent::particle_const_iterator partIter = mctruth.particles_begin();
  //	partIter != mctruth.particles_end();
  //	++partIter) {
  // //    for (HepMC::GenEvent::vertex_const_iterator vertIter = mctruth.vertices_begin();
  // // 	vertIter != mctruth.vertices_end();
  // // 	++vertIter) {
  //    CLHEP::HepLorentzVector creation = (*partIter)->CreationVertex();
  //    CLHEP::HepLorentzVector momentum = (*partIter)->Momentum();
  //   HepPDT::ParticleID id = (*partIter)->particleID();  // electrons and positrons are 11 and -11
  //     edm::LogInfo("") << "MC particle id " << id.pid() << ", creationVertex " << creation << " cm, initialMomentum " << momentum << " GeV/c" << endl;
  //  }

  // load the rechits for the Ecal
  edm::Handle<EcalRecHitCollection> pRecHits;
  iEvent.getByLabel(eBRecHitProducer_, eBRecHitCollection_, pRecHits);
  // Create a pointer to the RecHits  - unused for now
  //  const EcalRecHitCollection *hitCollection = pRecHits.product();


  // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/self/DataFormats/EgammaReco/interface/SuperCluster.h
  edm::Handle<reco::SuperClusterCollection> clusterHandle;
  iEvent.getByLabel(superClusterProducer_, superClusterCollection_, clusterHandle);


  ///////////////////////////////////////////////////////////////////////////////
  /////////////////  Loop over all superClusters ////////////////////////////////

  LogDebug("") << " Start loop over "
	       << clusterHandle->end()-clusterHandle->begin()
	       << "  superClusters " ;

  for (reco::SuperClusterCollection::const_iterator clusterIter = clusterHandle->begin();
       clusterIter != clusterHandle->end();
       ++clusterIter) {
    double energy = clusterIter->energy();
    math::XYZPoint position = clusterIter->position();
    std::ostringstream str;

    str  << " SuperCluster " << energy << " GeV, position "
	 << position << " cm" << "\n" ;

    energySuperClusters_->Fill(energy);
    sizeSuperClusters_->Fill(clusterIter->clustersSize());
    // this only makes sense for hybrid superclusters

    // try to point to the constituent clusters for this SuperCluster

    str << "About to loop over basicClusters" << "\n" ;

    double emaxSuperCluster = 0. ;
    double phibar = 0. ;
    double phi2bar = 0. ;
    double eTotSuperCluster = 0. ;



    for (reco::CaloCluster_iterator basicClusterIter = clusterIter->clustersBegin() ;
	 basicClusterIter != clusterIter->clustersEnd() ;
	 ++basicClusterIter ){

      //std::vector<DetId> theIds= (*basicClusterIter)->getHitsByDetId();

      str << " basicCluster Energy " << (*basicClusterIter)->energy()
	  << " Position " << (*basicClusterIter)->position()
	  << " \n"
	  << "        Position phi " << (*basicClusterIter)->position().phi()
	  << " recHits " << (*basicClusterIter)->size()
	  << " \n" ;

      double eCluster =  (*basicClusterIter)->energy();
      if(eCluster > emaxSuperCluster  ){
	emaxSuperCluster = eCluster ;
      }
      eTotSuperCluster += eCluster ;
      double phiCluster = (*basicClusterIter)->position().phi() ;
      phibar += eCluster * phiCluster ;
      phi2bar += eCluster * phiCluster * phiCluster ;


    } // end of basicClusterIter loop



    phibar= phibar /eTotSuperCluster ;
    phi2bar= phi2bar /eTotSuperCluster ;
    double phiWidth = phi2bar - phibar*phibar ;
    if(phiWidth>0.) {
      phiWidth = std::pow(phiWidth,0.5);
    }else{
      phiWidth =0.;
    }
    str << " SuperCluster stats " << "\n" ;
    str << "phibar " << phibar
	<< " phi2bar " << phi2bar
	<< " eTotSuperCluster " << eTotSuperCluster
	<< " phiWidth " << phiWidth
	<< std::endl  ;

    phiWidthSuperClusters_->Fill(phiWidth);

    emaxSuperClusters_->Fill(emaxSuperCluster);

    str << " Done with this SuperCluster " << std::endl;

    LogDebug("") << str.str() ;

  } // end of loop over superClusters

  LogDebug("") << " End loop over superClusters ";

  ///////////////////////  End of Loop over all superClusters ///////////////
  //////////////////////////////////////////////////////////////////////////


  /////////////////////////////////////////////
  //
  // loop over all EcalRecHits and print out their x,y,z,E
  //  edm::LogInfo("") << " Dumping all recHits in this event " << endl ;
  // for(EcalRecHitCollection::const_iterator _blah = hitCollection->begin();
  //     _blah != hitCollection->end() ; ++_blah ) {
  //   edm::LogInfo("") << "Ecal RecHit Energy: " << _blah->energy() << endl ;
  //   //      " Position " << _blah.position() << endl ;
  //  }
  //
  //  edm::LogInfo("") << "Dump finished " << endl ;
  //
  //////////////////////////////////////////////




  // DataFormats/EgammaCandidates/src/SiStripElectron.cc
  edm::Handle<reco::SiStripElectronCollection> siStripElectronHandle;
  iEvent.getByLabel(siElectronProducer_, siElectronCollection_, siStripElectronHandle);



  /////////////////////////////////////////////////////////////////////////////


  LogDebug("") << " Dumping Algo's guess of SiStripElectron Candidate Info " ;
  int numberOfElectrons = 0;
  // need to check if fit succeeded
  LogDebug("") << " Number of SiStripElectrons  " << siStripElectronHandle->size() ;


  for (reco::SiStripElectronCollection::const_iterator electronIter = siStripElectronHandle->begin();
       electronIter != siStripElectronHandle->end();  ++electronIter) {


    LogDebug("")  << "about to get stuff from electroncandidate "
		  << numberOfElectrons << "\n"
		  << "supercluster energy = "
		  << electronIter->superCluster()->energy() << "\n"
		  << "fit results are phi(r) = "
		  << electronIter->phiAtOrigin() << " + "
		  << electronIter->phiVsRSlope() << "*r" << "\n"
		  << " chi2 " << electronIter->chi2()
		  << " ndof " << electronIter->ndof() << "\n"
		  << " Pt " << electronIter->pt() << "\n"
		  << "P, Px, Py, Pz  "
		  << electronIter->p() << " "
		  << electronIter->px() << " "
		  << electronIter->py() << " "
		  << electronIter->pz() << "\n"
		  << "you get the idea..." ;

    // make plots for supercluster that an electron has been associ w/. here
    energySuperClustersEl_->Fill(electronIter->superCluster()->energy());
    sizeSuperClustersEl_->Fill(electronIter->superCluster()->clustersSize());

    // loop over basicClusters to get energy
    double emaxSuperCluster = 0. ;
    double phibar = 0. ;
    double phi2bar = 0. ;
    double eTotSuperCluster = 0. ;

    for (reco::CaloCluster_iterator basicClusterIter = electronIter->superCluster()->clustersBegin() ;
	 basicClusterIter != electronIter->superCluster()->clustersEnd() ;
	 ++basicClusterIter ){

      //std::vector<DetId> theIds= (*basicClusterIter)->getHitsByDetId();

      double eCluster =  (*basicClusterIter)->energy();
      if(eCluster > emaxSuperCluster  ){
	emaxSuperCluster = eCluster ;
      }
      eTotSuperCluster += eCluster ;
      double phiCluster = (*basicClusterIter)->position().phi() ;
      phibar += eCluster * phiCluster ;
      phi2bar += eCluster * phiCluster * phiCluster ;

    }

    phibar=phibar/eTotSuperCluster ;
    phi2bar=phi2bar/eTotSuperCluster ;
    double phiWidth = phi2bar - phibar*phibar ;
    if(phiWidth>0.) {
      phiWidth = std::pow(phiWidth,0.5);
    }else{
      phiWidth =0.;
    }

    phiWidthSuperClustersEl_->Fill(phiWidth);

    emaxSuperClustersEl_->Fill(emaxSuperCluster);

    numberOfElectrons++;
  }

  numCand_->Fill(siStripElectronHandle->size());

  ///////////////////////////////////////////////////////////////////////////



  // Now loop over the electrons (ie the fitted things.)

  LogDebug("")<< " About to check Electrons" ;

  edm::Handle<reco::ElectronCollection> electrons ;
  iEvent.getByLabel(electronProducer_, electronCollection_, electrons);

  numElectrons_->Fill(electrons->end()- electrons->begin());

  // set up vector of bool for SiStrips having or not having Electrons
  // this causes a warning because of variable array size at compilation time ;
  // BAD  bool hasElectron_[siStripElectronHandle->end()- siStripElectronHandle->begin()] ;
  bool* hasElectron_ = new bool[siStripElectronHandle->end()- siStripElectronHandle->begin()] ;
  for (int icount = 0 ;
       icount < siStripElectronHandle->end()- siStripElectronHandle->begin() ;
       ++icount)
    { hasElectron_[icount] = false ;}

  // also set up a counter to associate the ith electron to the jth strippy
  // Electron_to_strippy[i] = j: i-th Electron is j-th strippy
  // BAD  unsigned int Electron_to_strippy[electrons->end()- electrons->begin()];
  unsigned int* Electron_to_strippy = new unsigned int[electrons->end()- electrons->begin()];
  for (int icount = 0 ;
       icount <electrons->end()- electrons->begin();  ++icount)
    { Electron_to_strippy[icount] = 0 ;}

  unsigned int ecount=0 ;
  for (reco::ElectronCollection::const_iterator electronIter = electrons->begin();
       electronIter != electrons->end(); ++electronIter ){


    LogDebug("")<< " Associating Electrons to Strippies " ;
    LogDebug("")<< " PT is " << electronIter->track()->pt() ;

    reco::TrackRef tr =(*electronIter).track();
    uint32_t id = (*electronIter->track()->recHitsBegin())->geographicalId().rawId();
    LocalPoint pos = (*electronIter->track()->recHitsBegin())->localPosition();

    unsigned int icount = 0 ;
    LogDebug("") << " About to loop over Strippies " << " \n "
		 << " icount " << icount
		 << " max " << siStripElectronHandle->end()- siStripElectronHandle->begin() ;

    for (reco::SiStripElectronCollection::const_iterator strippyiter = siStripElectronHandle->begin();
	 strippyiter != siStripElectronHandle->end(); ++strippyiter) {

      bool hitInCommon = false;
      // loop over rphi hits
      for (std::vector<SiStripRecHit2D>::const_iterator
	     hiter = strippyiter->rphiRecHits().begin();
	   hiter != strippyiter->rphiRecHits().end();
	   ++hiter) {
	if (hiter->geographicalId().rawId() == id  &&
	    (hiter->localPosition() - pos).mag() < 1e-10) {
	  hitInCommon = true;
	  break;
	}
      }

      for (std::vector<SiStripRecHit2D>::const_iterator
	     hiter = strippyiter->stereoRecHits().begin();
	   hiter != strippyiter->stereoRecHits().end();
	   ++hiter) {
	if (hiter->geographicalId().rawId() == id  &&
	    (hiter->localPosition() - pos).mag() < 1e-10) {
	  hitInCommon = true;
	  break;
	}
      }
      if (hitInCommon) {  //this Electron belongs to this SiStripElectron.
	hasElectron_[icount] = true ;
	Electron_to_strippy[ecount]= icount ;
	ptDiff->Fill( std::abs(electronIter->track()->pt()) - std::abs(strippyiter->pt()) );
	pDiff->Fill( std::abs(electronIter->track()->p()) - std::abs(strippyiter->p()) );

      }
      icount++ ;
    } // Sistrip loop
    ecount++;
  } // Electrons

  LogDebug("") << " Done looping over Electrons " ;


  unsigned int counter = 0 ;
  for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectronHandle->begin();  strippyIter != siStripElectronHandle->end();  ++strippyIter) {


    bool skipThis = !hasElectron_[counter] ;
    if( skipThis ) {
      // plot stuff for SIStripElectrons that don't have fits associated

      LogDebug("") << " SiStrip Failed Electron " << " \n " <<
	" p " << strippyIter->p() << " \n " <<
	" pt " << strippyIter->pt() << " \n " <<
	" SuperClust size " << strippyIter->superCluster()->clustersSize() ;

      pElectronFailed->Fill( std::abs(strippyIter->p()) );
      ptElectronFailed->Fill( std::abs(strippyIter->pt()) );
      sizeSuperClustersFailed->Fill(strippyIter->superCluster()->clustersSize());
      LogDebug("") << " done filling Failed histos " ;
      //      energySuperClustersFailed->Fill(strippyIter->superCluster()->energy());
      //       if(strippyIter->p()>0.) {
      // 	eOverPFailed->Fill(strippyIter->superCluster()->energy()/strippyIter->p());
      //       }else {
      // 	eOverPFailed->Fill(-1.0);
      //       }

    } else {
      LogDebug("") << " SiStrip Passed Electron " << " \n " <<
	" p " << strippyIter->p() << " \n " <<
	" pt " << strippyIter->pt() << " \n " <<
	" SuperClust size " << strippyIter->superCluster()->clustersSize() ;
      pElectronPassed->Fill( std::abs(strippyIter->p()) );
      ptElectronPassed->Fill( std::abs(strippyIter->pt()) );
      sizeSuperClustersPassed->Fill(strippyIter->superCluster()->clustersSize());
      LogDebug("") << " done filling passed histos " ;
      //      energySuperClustersPassed->Fill(strippyIter->superCluster()->energy());
      //       if(strippyIter->p()>0.) {
      // 	eOverPPassed->Fill(strippyIter->superCluster()->energy()/strippyIter->p());
      //       }else {
      // 	eOverPPassed->Fill(-1.0);
      //       }

    } // skipThis
    counter++;
  }

  LogDebug("")<< "Dump info for all electrons ";

  for (reco::ElectronCollection::const_iterator electronIter1 = electrons->begin();
       electronIter1 != electrons->end(); ++electronIter1 ){
    reco::TrackRef tr1 =(*electronIter1).track();
    // let's find its associated SiStripElectron and SuperCluster
    unsigned int ecount1= electronIter1-electrons->begin() ;
    unsigned int stripCount1 = 0 ;
    reco::SiStripElectronCollection::const_iterator strippyIter1 ;
    for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectronHandle->begin();
	 strippyIter != siStripElectronHandle->end();  ++strippyIter) {
      if(Electron_to_strippy[ecount1]==stripCount1 ) {
	strippyIter1 = strippyIter ;
	break ; }
      stripCount1++ ;
    } // strippy loop
    ecount1++;

    std::ostringstream str;


    str << " SiStripElect p , px, py, pz " << strippyIter1->p()
	<< "  " << strippyIter1->px()
	<< "  " << strippyIter1->py()
	<< "  " << strippyIter1->pz()
	<< "\n " << std::endl ;


    str  << " Electron p px, py, pz,  = " << tr1->p()
	 << "  " << tr1->px()
	 << "  " << tr1->py()
	 << "  " << tr1->pz()
	 << "\n" <<  std::endl ;


    double EClust1 = strippyIter1->superCluster()->energy() ;
    double XClust1 = strippyIter1->superCluster()->x();
    double YClust1 = strippyIter1->superCluster()->y();
    double ZClust1 = strippyIter1->superCluster()->z();

    double rho1 = sqrt(XClust1*XClust1+YClust1*YClust1+ZClust1*ZClust1) ;
    double costheta1 = ZClust1/rho1 ;
    double sintheta1 = sqrt(1-costheta1*costheta1);
    if(ZClust1<0 ) { sintheta1 = - sintheta1 ; }
    double cosphi1 = XClust1/sqrt(XClust1*XClust1+YClust1*YClust1);
    double sinphi1 = YClust1/sqrt(XClust1*XClust1+YClust1*YClust1);

    str << " Ecal for electron E, px, py, pz "
	<< EClust1 << " "
	<< EClust1*sintheta1*cosphi1 << " "
	<< EClust1*sintheta1*sinphi1  << " "
	<< EClust1*costheta1
	<< "\n" << std::endl ;

    LogDebug("") << str.str() ;

  } // loop over electrons
 LogDebug("")<< "Done Dumping info for all electrons ";

  ///////////////////////////////////////////////////////
  // LogDebug("")<< " Checking Electrons" ;
  //  LogDebug("")<< " PT is " << electronIter->track()->pt() ;
  //    reco::TrackRef tr =(*electronIter).track();
  /// For events w/ more than 1 electron candidate, try to plot m(e,e)
  if(electrons->end()-electrons->begin()> 1) {
    edm::LogInfo("") << " Two electrons in this event " << std::endl;
    for (reco::ElectronCollection::const_iterator electronIter1 = electrons->begin();
	 electronIter1 != electrons->end()-1; ++electronIter1 ){
      reco::TrackRef tr1 =(*electronIter1).track();

      // let's find its associated SiStripElectron and SuperCluster
      // use the Electron_to_strippy[] array
      unsigned int ecount1= electronIter1-electrons->begin() ;
      // loop over strippies to find the corresponding one
      unsigned int stripCount1 = 0 ;
      reco::SiStripElectronCollection::const_iterator strippyIter1 ;
      for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectronHandle->begin();  strippyIter != siStripElectronHandle->end();  ++strippyIter) {
	if(Electron_to_strippy[ecount1]==stripCount1 ) {
	  strippyIter1 = strippyIter ;
	  break ; }
	stripCount1++ ;
      } // strippy loop

      double EClust1 = strippyIter1->superCluster()->energy() ;
      double XClust1 = strippyIter1->superCluster()->x();
      double YClust1 = strippyIter1->superCluster()->y();
      double ZClust1 = strippyIter1->superCluster()->z();

      for (reco::ElectronCollection::const_iterator electronIter2 = electronIter1+1;
	   electronIter2 != electrons->end(); ++electronIter2 ){

	reco::TrackRef tr2 =(*electronIter2).track();

	unsigned int ecount2= electronIter2-electrons->begin() ;
	unsigned int stripCount2 = 0 ;
	reco::SiStripElectronCollection::const_iterator strippyIter2 ;
	for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectronHandle->begin();  strippyIter != siStripElectronHandle->end();  ++strippyIter) {
	  if(Electron_to_strippy[ecount2]==stripCount2 ) {
	    strippyIter2 = strippyIter ;
	    break ; }
	  stripCount2++ ;
	} // strippy loop



	double EClust2 = strippyIter2->superCluster()->energy() ;
	double XClust2 = strippyIter2->superCluster()->x();
	double YClust2 = strippyIter2->superCluster()->y();
	double ZClust2 = strippyIter2->superCluster()->z();


	// now get supercluster from this:


      	edm::LogInfo("")  << " Electron p1 = " << tr1->p()
			  << " p1x " << tr1->px()
			  << " p1y " << tr1->py()
			  << " p1z " << tr1->pz()
			  << std::endl ;


  	edm::LogInfo("")  << " Electron p2 = " << tr2->p()
			  << " p2x " << tr2->px()
			  << " p2y " << tr2->py()
			  << " p2z " << tr2->pz()
			  << std::endl ;


	// combine the two in an (e,e) pair
	double Zpx = tr1->px()+tr2->px() ;
	double Zpy = tr1->py()+tr2->py() ;
	double Zpz = tr1->pz()+tr2->pz() ;
	double Ze = std::abs(tr1->p())+std::abs(tr2->p()) ;
	edm::LogInfo("") << " Z mass " <<
	  sqrt(Ze*Ze-Zpx*Zpx-Zpy*Zpy-Zpz*Zpz) << std::endl ;

	// combine the SuperClusts into a Z
	double rho1 = sqrt(XClust1*XClust1+YClust1*YClust1+ZClust1*ZClust1) ;
	double costheta1 = ZClust1/rho1 ;
	double sintheta1 = sqrt(1-costheta1*costheta1);
	if(ZClust1<0 ) { sintheta1 = - sintheta1 ; }
	double cosphi1 = XClust1/sqrt(XClust1*XClust1+YClust1*YClust1);
	double sinphi1 = YClust1/sqrt(XClust1*XClust1+YClust1*YClust1);

	double rho2 = sqrt(XClust2*XClust2+YClust2*YClust2+ZClust2*ZClust2) ;
	double costheta2 = ZClust2/rho2 ;
	double sintheta2 = sqrt(1-costheta2*costheta2);
	if(ZClust2<0 ) { sintheta2 = - sintheta2 ; }
	double cosphi2 = XClust2/sqrt(XClust2*XClust2+YClust2*YClust2);
	double sinphi2 = YClust2/sqrt(XClust2*XClust2+YClust2*YClust2);

	edm::LogInfo("") << "Energy of supercluster for 1st electron "
			 << EClust1 << " "
			 << EClust1*sintheta1*cosphi1 << " "
			 << EClust1*sintheta1*sinphi1  << " "
			 << EClust1*costheta1  << " "
			 << std::endl ;

	edm::LogInfo("") << "Energy of supercluster for 2nd electron "
			 << EClust2 << " "
			 << EClust2*sintheta2*cosphi2 << " "
			 << EClust2*sintheta2*sinphi2  << " "
			 << EClust2*costheta2  << " "
			 << std::endl ;


	// get the supercluster pair
	double Zgpx = EClust1*sintheta1*cosphi1+EClust2*sintheta2*cosphi2 ;
	double Zgpy = EClust1*sintheta1*sinphi1+EClust2*sintheta2*sinphi2 ;
	double Zgpz = EClust1*costheta1+EClust2*costheta2 ;
	double ZgE = EClust1+EClust2 ;

	edm::LogInfo("") << " Z mass from ECAL " <<
	  sqrt(ZgE*ZgE-Zgpx*Zgpx-Zgpy*Zgpy-Zgpz*Zgpz) << std::endl ;


      } //inner loop
    } // outer loop
  }// m(ee) loop

  delete[] hasElectron_;
  delete[] Electron_to_strippy;

  ///
  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////// Now for tracker hits: ///////////////////////////////////////
  LogDebug("") << " About to dump tracker info " ;

  edm::ESHandle<TrackerGeometry> trackerHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);

  edm::Handle<SiStripRecHit2DCollection> rphiHitsHandle;
  iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);

  edm::Handle<SiStripRecHit2DCollection> stereoHitsHandle;
  iEvent.getByLabel(siHitProducer_, siStereoHitCollection_, stereoHitsHandle);

  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHitsHandle;
  iEvent.getByLabel(siHitProducer_, siMatchedHitCollection_, matchedHitsHandle);

  /// loop again to get all info into myTree

  ////// get cluster
  NShowers_=0 ;
  for (reco::SuperClusterCollection::const_iterator clusterIter = clusterHandle->begin();
       clusterIter != clusterHandle->end();
       ++clusterIter) {
    double energy = clusterIter->energy();
    math::XYZPoint position = clusterIter->position();
    if(NShowers_ < myMaxHits ) {
      EShower_[NShowers_] = energy ;
      XShower_[NShowers_] = position.x() ;
      YShower_[NShowers_] = position.y() ;
      ZShower_[NShowers_] = position.z() ;
      ++NShowers_ ;
    }
    // Loop over all crystals in this supercluster - see
    // RecoEcal/EgamaClusterProducers/src/EgammaSimpleAnalyzer.cc
    // Look also at DataFormats/EgammaReco/interface/SuperCluster.h

  }
  numSuperClusters_->Fill(NShowers_);
  /////////////////////////////////////////////////////////////////////


  LogDebug("") << " Looping over stereo hits " ;


  /////// Loop over Stereo Hits
  int myHits = 0 ;
  for (SiStripRecHit2DCollection::DataContainer::const_iterator hit = stereoHitsHandle->data().begin(), hitend = stereoHitsHandle->data().end();
          hit != hitend;  ++hit) {
      DetId id(hit->geographicalId());
      if( (hit->geographicalId()).subdetId() == StripSubdetector::TIB  ||
	  (hit->geographicalId()).subdetId() == StripSubdetector::TOB    ) {
	GlobalPoint position = trackerHandle->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
	//from RecoLocalTracker/SiStripClusterizer/test/TestCluster.cc
	// cf also TrackHitAssociator.cc SiStripRecHitMatcher.cc SiStrip1DMeasurementTransformator.cc (KalmanUpdators)
	SiStripRecHit2D const rechit = *hit ;
	//	LocalPoint myposition = rechit.localPosition() ;
	LocalError myerror = rechit.localPositionError();

	// Get layer and subdetector ID here for this hit
	// see SiStripRecHitConverter/test/ValHit.cc
	Int_t siLayerNum = 0 ;
	Int_t siDetNum = 0 ;
	string siDetName = "" ;
	if( (hit->geographicalId()).subdetId() == StripSubdetector::TIB ){
	  //	   siLayerNum = tTopo->tibLayer(rechit->geographicalID());
	  siLayerNum = tTopo->tibLayer(id);
	  siDetNum = 1 ;
	  siDetName = "TIB" ;
	} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TOB ){
	  siLayerNum = tTopo->tobLayer(id);
	  siDetNum = 2 ;
	  siDetName = "TOB" ;
	  // 		} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TID ){
	  // 	  // should we use side/wheel/ring/module/stereo() ?
	  // 	  siLayerNum = tTopo->tidWheel(id);
	  // 	  siDetNum = 3 ;
	  // 	  siDetName = "TID" ;
	  // 	}else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TEC ){
	  // 	  //choices are side/petal/wheel/ring/module/glued/stereo
	  // 	  siLayerNum = tTopo->tecWheel(id);
	  // 	  siDetNum = 4 ;
	  // 	  siDetName = "TEC" ;
	}else {
	  siLayerNum = -999 ;
	  siDetNum = -999 ;
	  siDetName = "NULL" ;
	}
	//	LogDebug("") << siDetName << " " << siLayerNum ;

	const SiStripRecHit2D::ClusterRef & clust=rechit.cluster();
	double Signal = 0 ;
	double Noise2 = 0 ;
	int StripCount = 0 ;
	if(clust.isNonnull()) {
	  //	  LogDebug("") << " barycenter " << clust->barycenter() ;
	  //	  const std::vector<uint16_t> amplitudes=clust->amplitudes();
	  const std::vector<uint8_t> amplitudes=clust->amplitudes();
	  for(size_t i = 0 ; i<amplitudes.size(); i++ ){
	    Signal +=amplitudes[i] ;
	    //ignore for now	     Noise2 +=SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i)*SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i);
	    StripCount++;
	  }
	} else {
	  LogDebug("") << " null cluster " ;
	}
	//	LogDebug("") << "Signal " << Signal << " Noise2 " << Noise2 << " StripCount " << StripCount ;
	// Dump position
	// 	LogDebug("") << " Stereo "
	// 			 << "local position: "<<myposition.x()<<" "
	// 			 << myposition.y()<<" "<<myposition.z()<<"\n"
	// 			 << "local error: "<<myerror.xx()<<" "
	// 			 << myerror.xy()<<" "<<myerror.yy() << "\n"
	// 			 << "global position: " << position.x() << " "
	// 			 <<  position.y()<<" "<< position.z()<<"\n"
	// 			 << " siDetNum " << siDetNum
	// 			 << " siLayerNum " << siLayerNum ;


	if( myHits < myMaxHits ) {
	  StereoHitX_[myHits] = position.x();
	  StereoHitY_[myHits] = position.y();
	  StereoHitZ_[myHits] = position.z();

	  StereoHitR_[myHits]=position.perp();
	  StereoHitPhi_[myHits]=position.phi();
	  StereoHitTheta_[myHits]=position.theta();

	  StereoHitSigX_[myHits]=sqrt(myerror.xx());
	  StereoHitSigY_[myHits]=sqrt(myerror.yy());
	  StereoHitCorr_[myHits]=myerror.xy()/sqrt(myerror.xx()*myerror.yy());

	  StereoHitSignal_[myHits] = Signal ;
	  StereoHitNoise_[myHits] = Noise2 ;
	  StereoHitWidth_[myHits] = StripCount ;

	  StereoDetector_[myHits] = siDetNum ;
	  StereoLayer_[myHits] = siLayerNum ;

	  ++myHits ;
	}
      } // end if this is the right subdetector
  } // end loop over hits
  NStereoHits_ = myHits ;

  numSiStereoHits_->Fill(NStereoHits_);

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////

  LogDebug("") << " Looping over Mono Hits " ;
  /////// Loop over Mono Hits
  myHits = 0 ;
  for (SiStripRecHit2DCollection::DataContainer::const_iterator hit = rphiHitsHandle->data().begin(), hitend = rphiHitsHandle->data().end();
          hit != hitend;  ++hit) {
      DetId id(hit->geographicalId());

      if ((hit->geographicalId()).subdetId() == StripSubdetector::TIB ||
	  (hit->geographicalId()).subdetId() == StripSubdetector::TOB) {

	GlobalPoint position = trackerHandle->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
	//from RecoLocalTracker/SiStripClusterizer/test/TestCluster.cc
	// cf also TrackHitAssociator.cc SiStripRecHitMatcher.cc SiStrip1DMeasurementTransformator.cc (KalmanUpdators)
	SiStripRecHit2D const rechit = *hit ;
	//	LocalPoint myposition = rechit.localPosition() ;
	LocalError myerror = rechit.localPositionError();

	// Get layer and subdetector ID here for this hit
	// see SiStripRecHitConverter/test/ValHit.cc
	Int_t siLayerNum = 0 ;
	Int_t siDetNum = 0 ;
	string siDetName = "" ;
	if( (hit->geographicalId()).subdetId() == StripSubdetector::TIB ){
	  //	   siLayerNum = tTopo->tibLayer(rechit->geographicalID());
	  siLayerNum = tTopo->tibLayer(id);
	  siDetNum = 1 ;
	  siDetName = "TIB" ;
	} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TOB ){
	  siLayerNum = tTopo->tobLayer(id);
	  siDetNum = 2 ;
	  siDetName = "TOB" ;
	  // 	} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TID ){
	  // 	  // should we use side/wheel/ring/module/stereo() ?
	  // 	  siLayerNum = tTopo->tidWheel(id);
	  // 	  siDetNum = 3 ;
	  // 	  siDetName = "TID" ;
	  // 	}else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TEC ){
	  // 	  //choices are side/petal/wheel/ring/module/glued/stereo
	  // 	  siLayerNum = tTopo->tecWheel(id);
	  // 	  siDetNum = 4 ;
	  // 	  siDetName = "TEC"
	  ;
	}else {
	  siLayerNum = -999 ;
	  siDetNum = -999 ;
	  siDetName = "NULL" ;
	}
	//	LogDebug("") << siDetName << " " << siLayerNum ;
	const SiStripRecHit2D::ClusterRef & clust=rechit.cluster();
	double Signal = 0 ;
	double Noise2 = 0 ;
	int StripCount = 0 ;
	if(clust.isNonnull()) {
	  //	  LogDebug("") << " barycenter " << clust->barycenter() ;
	  //	  const std::vector<uint16_t> amplitudes=clust->amplitudes();
	  const std::vector<uint8_t> amplitudes=clust->amplitudes();
	  for(size_t i = 0 ; i<amplitudes.size(); i++ ){
	    Signal +=amplitudes[i] ;
	    //ignore for now	     Noise2 +=SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i)*SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i);
	    StripCount++;
	  }
	} else {
	  LogDebug("") << " null cluster " ;
	}
	//	LogDebug("") << "Signal " << Signal << " Noise2 " << Noise2 << " StripCount " << StripCount ;

	// Dump position info
	// 	LogDebug("") << " Mono "
	// 			 << "local position: "<<myposition.x()<<" "
	// 			 << myposition.y()<<" "<<myposition.z()<<"\n"
	// 			 <<"local error: "<<myerror.xx()<<" "
	// 			 << myerror.xy()<<" "<<myerror.yy() << "\n"
	// 			 << "global position: " << position.x() << " "
	// 			 <<  position.y()<<" "<< position.z()<<"\n"
	// 			 << " siDetNum " << siDetNum
	// 			 << " siLayerNum " << siLayerNum ;

	if( myHits < myMaxHits ) {
	  MonoHitX_[myHits] = position.x();
	  MonoHitY_[myHits] = position.y();
	  MonoHitZ_[myHits] = position.z();

	  MonoHitR_[myHits]=position.perp();
	  MonoHitPhi_[myHits]=position.phi();
	  MonoHitTheta_[myHits]=position.theta();

	  MonoHitSigX_[myHits]=sqrt(myerror.xx());
	  MonoHitSigY_[myHits]=sqrt(myerror.yy());
	  MonoHitCorr_[myHits]=myerror.xy()/sqrt(myerror.xx()*myerror.yy());


	  MonoHitSignal_[myHits] = Signal ;
	  MonoHitNoise_[myHits] = Noise2 ;
	  MonoHitWidth_[myHits] = StripCount ;

	  MonoDetector_[myHits] = siDetNum ;
	  MonoLayer_[myHits] = siLayerNum ;

	  ++myHits ;
	}  // of  if(myHits < myMaxHits)
	//	LogDebug("")<< "end of myHits < myMaxHits " ;
      } // end if this is the right subdetector
      //      LogDebug("")<< "end of TIB/TOB check " ;
  } // end loop over hits
  //    LogDebug("")<< " end of loop over hits  " ;
  NMonoHits_ = myHits ;

  numSiMonoHits_->Fill(NMonoHits_);


  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////

  LogDebug("") << "  Loop over Matched Hits " ;

  /////// Loop over Matched Hits
  myHits = 0 ;
  for (SiStripMatchedRecHit2DCollection::DataContainer::const_iterator hit = matchedHitsHandle->data().begin(), hitend = matchedHitsHandle->data().end();
          hit != hitend;  ++hit) {
        DetId id(hit->geographicalId());
        if ((hit->geographicalId()).subdetId() == StripSubdetector::TIB  ||
   	  (hit->geographicalId()).subdetId() == StripSubdetector::TOB    ) {
   	GlobalPoint position = trackerHandle->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
   	SiStripMatchedRecHit2D const rechit = *hit ;
	//	LocalPoint myposition = rechit.localPosition() ;
	LocalError myerror = rechit.localPositionError();

   	// Get layer and subdetector ID here for this hit
   	// see SiStripRecHitConverter/test/ValHit.cc
   	Int_t siLayerNum = 0 ;
   	Int_t siDetNum = 0 ;
   	string siDetName = "" ;
   	if( (hit->geographicalId()).subdetId() == StripSubdetector::TIB ){
   	  siLayerNum = tTopo->tibLayer(id);
   	  siDetNum = 1 ;
   	  siDetName = "TIB" ;
   	} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TOB ){
   	  siLayerNum = tTopo->tobLayer(id);
   	  siDetNum = 2 ;
   	  siDetName = "TOB" ;
	  //    	} else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TID ){
	  //    	  // should we use side/wheel/ring/module/stereo() ?
	  //    	  siLayerNum = tTopo->tidWheel(id);
	  //    	  siDetNum = 3 ;
	  //    	  siDetName = "TID" ;
	  //    	}else if ( (hit->geographicalId()).subdetId() == StripSubdetector::TEC ){
	  //    	  //choices are side/petal/wheel/ring/module/glued/stereo
	  //    	  siLayerNum = tTopo->tecWheel(id);
	  //    	  siDetNum = 4 ;
	  //    	  siDetName = "TEC" ;
   	}else {
   	  siLayerNum = -999 ;
   	  siDetNum = -999 ;
   	  siDetName = "NULL" ;
   	}
	//	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=rechit.cluster();
   	double Signal = 0 ;
   	double Noise2 = 0 ;
   	int StripCount = 0 ;
	// 	if(clust.isNonnull()) {
	// 	  LogDebug("") << " barycenter " << clust->barycenter() ;
	// 	  const std::vector<uint16_t> amplitudes=clust->amplitudes();
	// 	  for(size_t i = 0 ; i<amplitudes.size(); i++ ){
	// 	    Signal +=amplitudes[i] ;
	// 	    //ignore for now	     Noise2 +=SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i)*SiStripNoiseService_.getNoise(detid,clust->firstStrip()+i);
	// 	    StripCount++;
	// 	  }
	// 	} else {
	// 	  LogDebug("") << " null cluster " ;
	// 	}
	// 	LogDebug("") << "Signal " << Signal << " Noise2 " << Noise2 << " StripCount " << StripCount ;

	// Dump position info
	// 	LogDebug("") << " Matched "
	// 			 << "local position: "<<myposition.x()<<" "
	// 			 << myposition.y()<<" "<<myposition.z()<<"\n"
	// 			 << "local error: "<<myerror.xx()<<" "
	// 			 << myerror.xy()<<" "<<myerror.yy() << "\n"
	// 			 << "global position: " << position.x() << " "
	// 			 <<  position.y()<<" "<< position.z()<<"\n"
	// 			 << " siDetNum " << siDetNum
	// 			 << " siLayerNum " << siLayerNum ;

   	if( myHits < myMaxHits ) {
   	  MatchedHitX_[myHits] = position.x();
   	  MatchedHitY_[myHits] = position.y();
   	  MatchedHitZ_[myHits] = position.z();


   	  MatchedHitR_[myHits]=position.perp();
   	  MatchedHitPhi_[myHits]=position.phi();
   	  MatchedHitTheta_[myHits]=position.theta();

	  MatchedHitSigX_[myHits]=sqrt(myerror.xx());
	  MatchedHitSigY_[myHits]=sqrt(myerror.yy());
	  MatchedHitCorr_[myHits]=myerror.xy()/sqrt(myerror.xx()*myerror.yy());



   	  MatchedHitSignal_[myHits] = Signal ;
   	  MatchedHitNoise_[myHits] = Noise2 ;
   	  MatchedHitWidth_[myHits] = StripCount ;

   	  MatchedDetector_[myHits] = siDetNum ;
   	  MatchedLayer_[myHits] = siLayerNum ;

   	  ++myHits ;
   	}
      } // end if this is the right subdetector (TIB/TOB)
  } // end loop over hits
  NMatchedHits_ = myHits ;

  numSiMatchedHits_->Fill(NMatchedHits_);

  //////////////////////////////////////////////////////////////////////



  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  LogDebug("") << "Writing to myTree with " << NShowers_ << " Showers "
	       << NStereoHits_ << " Si StereoHits "
	       << NMonoHits_ << " Si MonoHits "
	       << NMatchedHits_ << " Si MatchedHits " ;

  myTree_->Fill();





} // end of Analyzer


void SiStripElectronAnalyzer::endJob(){
  LogDebug("") << "Entering endJob " ;
  file_->cd() ;
  numCand_->Write();
  numElectrons_->Write();
  numSuperClusters_->Write();

  energySuperClusters_->Write();
  sizeSuperClusters_->Write();
  emaxSuperClusters_->Write();
  phiWidthSuperClusters_->Write();

  energySuperClustersEl_->Write();
  sizeSuperClustersEl_->Write();
  emaxSuperClustersEl_->Write();
  phiWidthSuperClustersEl_->Write();

  ptDiff->Write();
  pDiff->Write();
  pElectronFailed->Write();
  ptElectronFailed->Write();
  pElectronPassed->Write();
  ptElectronPassed->Write();
  sizeSuperClustersPassed->Write();
  sizeSuperClustersFailed->Write();
  //   energySuperClustersPassed->Write();
  //   energySuperClustersFailed->Write();
  //   eOverPPassed->Write();
  //   eOverPFailed->Write();


  numSiStereoHits_->Write();
  numSiMonoHits_->Write();
  numSiMatchedHits_->Write();

  // disable for large dataset
  LogDebug("") << " Writing out ntuple is disabled for now " ;
   myTree_->Write();


  file_->Close();
}


//
// const member functions
//

//
// static member functions
//
