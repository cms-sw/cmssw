//
//
//
//
// system include files
//
// You should use the TrackerHitAssociator (in SimTracker/TrackerHitAssociation)
// which will give you the PSimHit (or the id of the sim hit). From the 
// PSimHit you can then get track and event id. Of course this
// needs the presence of the corresponding products in your input file.
// Giuseppe Cerati - a person that is developing a lot of code in this respect for studies of the 
// tracking performances + Patrizia.Azzi@cern.ch
// the RecHits are stored in the event in a RangeMap (one  / type of hit) - from this map one can retrieve the 
// hits by DetId.
// Danek's example RecoLocalTracker/SiPixelRecHits/test/ReadPixelRecHit.cc
//  How to commit to head cmssw16x
//  1. sozdanie project area
//  2. cvs co -r jetCorrections_1_6_X_retrofit JetMETCorrections/JetPlusTrack
//  3. <editing>
//     cvs commit JetMETCorrections/JetPlusTrack
//     cvs tag <tagname> JetMETCorrections/JetPlusTrack

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// MC info
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
// #include "CLHEP/HepPDT/DefaultConfig.hh"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"
//double dR = deltaR( c1, c2 );

//
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//jets
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
//#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
//#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
//
// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// tracker geometry
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
// pixel rec hits
#include "RecoLocalTracker/SiPixelRecHits/test/ReadPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
// vertixes
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
//add simhit info
//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//simtrack
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
// #include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

//
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
//#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
//#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
// candidates
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
//
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"

//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

//
// class decleration

class SinglePionEfficiencyNew : public edm::EDAnalyzer {
public:
  explicit SinglePionEfficiencyNew(const edm::ParameterSet&);
  ~SinglePionEfficiencyNew();


  double eECALmatrix(CaloNavigator<DetId>& navigator,edm::Handle<EcalRecHitCollection>& hits, int ieta, int iphi);

  double eHCALmatrix(const HcalTopology* topology, const DetId& det,const HBHERecHitCollection& hits, int ieta, int iphi);

  double eHCALneighbours(std::vector<DetId>& vNeighboursDetId, std::vector<DetId>& dets, const HcalTopology* topology, const HBHERecHitCollection& hits);

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  // output tree and root file
  string fOutputFileName ;
  TFile*      hOutputFile ;
  TTree*      t1;
  // names of modules, producing object collections
  string tracksSrc;
  string pxltracksSrc;
  string pxlhitsSrc;
  string calotowersSrc;

  string ecalHitsProducerSrc;
  string ECALbarrelHitCollectionSrc;
  string ECALendcapHitCollectionSrc;

  string hbheInputSrc;
  string hoInputSrc;  
  string hfInputSrc;
  string towermakerSrc; 

  // variables to store in ntpl
  // simulated/generated tracks
  double ptSim1, etaSim1, phiSim1; 
  double ptSim2, etaSim2, phiSim2; 
  // reconstructed tracks
  double ptTrk1, etaTrk1, phiTrk1, drTrk1, purityTrk1;
  double ptTrk2, etaTrk2, phiTrk2, drTrk2, purityTrk2;
  // track quality and number of valid hits
  int trkQuality1, trkQuality2, trkNVhits1, trkNVhits2;
  int 	idmax1, idmax2;
  // reconstructed pixel triplets
  double ptPxl1, etaPxl1, phiPxl1, drPxl1, purityPxl1; 
  double ptPxl2, etaPxl2, phiPxl2, drPxl2, purityPxl2;
  // Et and energy in cone 0.5 around every MC particle
  double etCalo1, etCalo2;
  double eCalo1, eCalo2;
  // Et in 7x7, 11x11 crystal matrix and 3x3, 5x5 HCAL matrix
  double e1ECAL7x7, e2ECAL7x7, e1ECAL11x11, e2ECAL11x11; 
  double e1HCAL3x3, e2HCAL3x3, e1HCAL5x5, e2HCAL5x5;
  // end

  // track hit associator
  TrackerHitAssociator* hitAssociator;
  // track associator to detector parameters 
  TrackAssociatorParameters parameters_;
  mutable TrackDetectorAssociator* trackAssociator_;
  //
  const edm::ParameterSet conf_;
  };
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SinglePionEfficiencyNew::SinglePionEfficiencyNew(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
   //now do what ever initialization is needed
  using namespace edm;

  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile"); 
  tracksSrc     = iConfig.getParameter<string>("tracks"); 
  pxltracksSrc  = iConfig.getParameter<string>("pxltracks"); 
  pxlhitsSrc    = iConfig.getParameter<string>("pxlhits");
  calotowersSrc = iConfig.getParameter<std::string>("calotowers");
  towermakerSrc = iConfig.getParameter<std::string>("towermaker");

  ecalHitsProducerSrc        = iConfig.getParameter< std::string >("ecalRecHitsProducer");
  ECALbarrelHitCollectionSrc = iConfig.getParameter<string>("ECALbarrelHitCollection");
  ECALendcapHitCollectionSrc = iConfig.getParameter<string>("ECALendcapHitCollection");

  hbheInputSrc = iConfig.getParameter<string>("hbheInput");
  hoInputSrc   = iConfig.getParameter<string>("hoInput");
  hfInputSrc   = iConfig.getParameter<string>("hfInput"); 

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_ =  new TrackDetectorAssociator();
  trackAssociator_->useDefaultPropagator();
}


SinglePionEfficiencyNew::~SinglePionEfficiencyNew()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

/*
double SinglePionEfficiencyNew::deltaPhi(double phi1, double phi2)
{
  double pi = 3.1415927;
  double dphi = fabs(phi1 - phi2);
  if(dphi >= pi) dphi = 2. * pi - dphi; 
  return dphi;
}
double SinglePionEfficiencyNew::deltaEta(double eta1, double eta2)
{
  double deta = fabs(eta1-eta2);
  return deta;
}
double double::deltaR(double eta1, double eta2,
		      double phi1, double phi2)
{
  double dr = sqrt( deltaEta(eta1, eta2) * deltaEta(eta1, eta2) +
		     deltaPhi(phi1, phi2) * deltaPhi(phi1, phi2) );
  return dr;
}
*/

//
// member functions
//

// ------------ method called to for each event  ------------
void
SinglePionEfficiencyNew::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   hitAssociator = new TrackerHitAssociator::TrackerHitAssociator(iEvent);

   // initialize tree variables
   // sim tracks
   ptSim1 = 0.; etaSim1 = 0.; phiSim1 = 0.; ptSim2 = 0.; etaSim2 = 0.; phiSim2 = 0.; 
   // reconstructed tracks
   ptTrk1 = 0.; etaTrk1 = 0.; phiTrk1 = 0.; drTrk1 = 1000.; purityTrk1 = 0.;
   ptTrk2 = 0.; etaTrk2 = 0.; phiTrk2 = 0.; drTrk2 = 1000.; purityTrk2 = 0.;
   // reco pixel triplets
   ptPxl1 = 0.; etaPxl1 = 0.; phiPxl1 = 0.; drPxl1 = 1000.; purityPxl1 = 0.; 
   ptPxl2 = 0.; etaPxl2 = 0.; phiPxl2 = 0.; drPxl2 = 1000.; purityPxl2 = 0.; 
   // Et of calo towers in cone 0.5 around each MC particle
   etCalo1 = 0.; etCalo2 = 0.;
   eCalo1 = 0.; eCalo2 = 0.;
   // Et in 7x7, 11x11 crystal matrix and 3x3, 5x5 HCAL matrix
   e1ECAL7x7 = 0.; e2ECAL7x7 = 0.; e1ECAL11x11 = 0.; e2ECAL11x11 = 0.; 
   e1HCAL3x3 = 0.; e2HCAL3x3 = 0.; e1HCAL5x5 = 0.; e2HCAL5x5 = 0.; 
   //
   trkQuality1 = -1; trkQuality2 = -1; trkNVhits1 = -1; trkNVhits2 = -1;
   idmax1 = -1; idmax2 = -1;

   //
   // extract tracker geometry
   //
   edm::ESHandle<TrackerGeometry> theG;
   iSetup.get<TrackerDigiGeometryRecord>().get(theG);
   //   const TrackerGeometry& theTracker(*theG);

   edm::ESHandle<CaloGeometry> pG;
   //   iSetup.get<IdealGeometryRecord>().get(pG);
   iSetup.get<CaloGeometryRecord>().get(pG);
   const CaloGeometry* geo = pG.product();
   const CaloSubdetectorGeometry* gEB = geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
   const CaloSubdetectorGeometry* gEE = geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
   const CaloSubdetectorGeometry* gHBHE = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
   //   const CaloSubdetectorGeometry* gHE = geo->getSubdetectorGeometry(DetId::Hcal,HcalEndcap);

   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
   const CaloSubdetectorTopology* theEBTopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
   const CaloSubdetectorTopology* theEETopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap);

   // get HcalTopology. should be similar way to get CaloTowerTopology
   edm::ESHandle<HcalTopology> htopo;
   iSetup.get<IdealGeometryRecord>().get(htopo);
   const HcalTopology* theHBHETopology = htopo.product();

   //   const CaloTowerTopology* theCaloTowerTopology;

   // get MC info
   edm::Handle<HepMCProduct> EvtHandle ;
   iEvent.getByLabel( "source", EvtHandle ) ;
   //  iEvent.getByLabel( "VtxSmeared", EvtHandle ) ;

   const HepMC::GenEvent* evt = EvtHandle->GetEvent() ;
   ESHandle<ParticleDataTable> pdt;
   iSetup.getData( pdt );

   vector<HepLorentzVector> genpions;
   genpions.clear();

   for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
	 p != evt->particles_end(); ++p ) {
     
     HepLorentzVector pion((*p)->momentum().px(),(*p)->momentum().py(),(*p)->momentum().pz(),(*p)->momentum().e());
     genpions.push_back(pion);
     /*
     cout <<" status : " << (*p)->status() 
	  <<" pid = " << (*p)->pdg_id() 
	  <<" eta = " << (*p)->momentum().eta()
	  <<" phi = " << (*p)->momentum().phi() 
	  <<" theta = " << (*p)->momentum().theta() 
	  <<" pt =  " << (*p)->momentum().perp() 
	  <<" charge = " << (pdt->particle((*p)->pdg_id()))->charge()
	  <<" charge3 = " << (pdt->particle((*p)->pdg_id()))->ID().threeCharge() << endl;
     */
   }
   //   edm::Handle<EBRecHitCollection> barrelRecHitsHandle;
   //   edm::Handle<EERecHitCollection> endcapRecHitsHandle;
   edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
   edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;

   iEvent.getByLabel(ecalHitsProducerSrc,ECALbarrelHitCollectionSrc,barrelRecHitsHandle);
   iEvent.getByLabel(ecalHitsProducerSrc,ECALendcapHitCollectionSrc,endcapRecHitsHandle);
   // iEvent.getByLabel( edm::InputTag(ecalHitsProducerSrc,ECALendcapHitCollectionSrc,"RECO3"), endcapRecHitsHandle);

   EBRecHitCollection::const_iterator itb;
   for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
     GlobalPoint pos = geo->getPosition(itb->detid());
     double eta = pos.eta();
     //     double the = pos.theta();
     double phi = pos.phi();
     //     double et  = itb->energy() * sin(the);
     double DR1 = deltaR(genpions[0].eta(),genpions[0].phi(),eta,phi);
     double DR2 = deltaR(genpions[1].eta(),genpions[1].phi(),eta,phi);
     /*
     cout <<" ECAL barrel rechit, energy = " << itb->energy()
     	  <<" eta = " << pos.eta()
     	  <<" phi = " << pos.phi() 
	  <<" theta = " << pos.theta() 
	  <<" et = " << et << endl;
     */
     if(DR1 < 0.5) {
       eCalo1  = eCalo1  + itb->energy();
     }
     if(DR2 < 0.5) {
       eCalo2  = eCalo2  + itb->energy();
     }
   }

   for (itb=endcapRecHitsHandle->begin(); itb!=endcapRecHitsHandle->end(); itb++) {
     GlobalPoint pos = geo->getPosition(itb->detid());
     double eta = pos.eta();
     double phi = pos.phi();
     //     double the = pos.theta();
     //     double et  = itb->energy() * sin(the);
     double DR1 = deltaR(genpions[0].eta(),genpions[0].phi(),eta,phi);
     double DR2 = deltaR(genpions[1].eta(),genpions[1].phi(),eta,phi);
     if(DR1 < 0.5) {
       eCalo1  = eCalo1  + itb->energy();
     }
     if(DR2 < 0.5) {
       eCalo2  = eCalo2  + itb->energy();
     }
     /*
     cout <<" ECAL endcap rechit, energy = " << itb->energy()
     	  <<" eta = " << pos.eta()
     	  <<" phi = " << pos.phi() 
	  <<" theta = " << pos.theta() 
	  <<" et = " << et << endl;
     */
   }

   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(hbheInputSrc,hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());

   //   cout <<" ===> ZSP HBHERecHitCollection size = " << Hithbhe.size() << endl;
   //   cout <<" ===> NO ZSP HBHERecHitCollection size = " << HithbheR.size() << endl;

   for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++) {
     GlobalPoint pos = geo->getPosition( hbheItr->detid()); 
     double eta = pos.eta();
     double phi = pos.phi();
     //     double the = pos.theta();
     //     double et  = hbheItr->energy() * sin(the);
     double DR1 = deltaR(genpions[0].eta(),genpions[0].phi(),eta,phi);
     double DR2 = deltaR(genpions[1].eta(),genpions[1].phi(),eta,phi);
     if(DR1 < 0.5) {
       eCalo1  = eCalo1  + hbheItr->energy();
     }
     if(DR2 < 0.5) {
       eCalo2  = eCalo2  + hbheItr->energy();
     }
     /*
     cout <<" HBHE rechit, energy = " << hbheItr->energy()
     	  <<" eta = " << pos.eta()
     	  <<" phi = " << pos.phi() 
	  <<" theta = " << pos.theta() 
	  <<" et = " << et << endl;
     */
   }

   /*
  // wrong case for no ZSP
   double etCalo1T = 0.;
   double eCalo1T = 0.;
   double etCalo2T = 0.;
   double eCalo2T = 0.;
   //
   edm::Handle<CaloTowerCollection> towerHandle;
   iEvent.getByLabel(towermakerSrc, towerHandle);
   const CaloTowerCollection* towers = towerHandle.product();
   for(CaloTowerCollection::const_iterator towerItr = towers->begin(); 
       towerItr != towers->end(); ++towerItr) {
     
     double towerEt  = towerItr->et();
     double towerE   = towerItr->energy();
     double towerEta = towerItr->eta();
     double towerPhi = towerItr->phi();

     double etaMC1   = genpions[0].eta();
     double phiMC1   = genpions[0].phi();

     double etaMC2   = genpions[1].eta();
     double phiMC2   = genpions[1].phi();
     
     double DR1 = deltaR(genpions[0].eta(),genpions[0].phi(),towerItr->eta(),towerItr->phi());
     double DR2 = deltaR(genpions[1].eta(),genpions[1].phi(),towerItr->eta(),towerItr->phi());

     if(DR1 < 0.5) {
       eCalo1T = eCalo1T + towerItr->energy();
           cout <<" gen eta1 = " << genpions[0].eta()
	    <<" tower eta = " << towerItr->eta()
	    <<" gen phi1 = " << genpions[0].phi()
	    <<" tower phi = " << towerItr->phi() 
	    <<" DR1 = " << DR1 
	    <<" tower Et = " << towerItr->et() 
	    <<" etCalo1 = " << etCalo1 
	    <<" SimPt1 = " << genpions[0].perp() << endl;
         }
     if(DR2 < 0.5) {
       eCalo2T = eCalo2T + towerItr->energy();
           cout <<" gen eta2 = " << genpions[1].eta()
	    <<" tower eta = " << towerItr->eta()
	    <<" gen phi2 = " << genpions[1].phi()
	    <<" tower phi = " << towerItr->phi() 
	    <<" DR2 = " << DR2
	    <<" tower Et = " << towerItr->et()
	    <<" etCalo2 = " << etCalo2 
	    <<" SimPt2 = " << genpions[1].perp() << endl;
         }
   }
*/
   etCalo1 = eCalo1 * sin(genpions[0].theta());
   etCalo2 = eCalo2 * sin(genpions[1].theta());

   /*
   etCalo1T = eCalo1T * sin(genpions[0].theta());
   etCalo2T = eCalo2T * sin(genpions[1].theta());
   
   cout  <<" eCalo1 = " << eCalo1 <<" eCalo1T = " << eCalo1T
	 <<" etCalo1 = " << etCalo1 <<" etCalo1T = " << etCalo1T
	 <<" eCalo2 = " << eCalo2 <<" eCalo2T = " << eCalo2T
	 <<" etCalo2 = " << etCalo2 <<" etCalo2T = " << etCalo2T << endl;
   */

   /*
  // wrong case for no ZSP
   // get calo towers and collect Et in cone 0.5 around every MC particle 
   edm::Handle<CandidateCollection> calotowers;
   iEvent.getByLabel(calotowersSrc, calotowers);   
   const CandidateCollection* inputCol = calotowers.product();
   CandidateCollection::const_iterator candidate;
   for( candidate = inputCol->begin(); candidate != inputCol->end(); ++candidate )
     {
       double phi   = candidate->phi();
       double theta = candidate->theta();
       double eta   = candidate->eta();
       double e     = candidate->energy();
       double et    = e*sin(theta);

       cout <<" calo towers: phi = " << phi
	    <<" eta = " << eta
	    <<" et = " << et << endl;

       HepLorentzVector tower(candidate->px(),
			      candidate->py(),
			      candidate->pz(),
			      candidate->energy());
       double DR1 = genpions[0].deltaR(tower);
       double DR2 = genpions[1].deltaR(tower);
       if(DR1 < 0.5) {
	 etCalo1 = etCalo1 + candidate->pt();
	 eCalo1 = eCalo1 + candidate->energy();
       }
       if(DR2 < 0.5) {
	 etCalo2 = etCalo2 + candidate->pt();
	 eCalo2 = eCalo2 + candidate->energy();
       }
     }
   */

   //get simtrack info
   std::vector<SimTrack> theSimTracks;
   edm::Handle<SimTrackContainer> SimTk;
   iEvent.getByLabel("g4SimHits",SimTk);
   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
   //   cout <<" number of SimTracks = " << theSimTracks.size() << endl;

   edm::Handle<SiPixelRecHitCollection> pxlrecHitColl;
   iEvent.getByLabel(pxlhitsSrc, pxlrecHitColl);
   //   cout <<"  Size of pxl rec hit collection = " << (pxlrecHitColl.product())->size() << endl;
   
   // initiate track hit associator
   std::vector<PSimHit> matched;
   std::vector<unsigned int> SimTrackIds;
   // correct
   //   hitAssociator->init(iEvent);

   // track collection
    //   Handle<TrackCollection> tracks;
   Handle<TrackCollection> tracks;
   iEvent.getByLabel(tracksSrc, tracks);
   //   cout << "====> number of reco tracks "<< tracks->size() << endl;

   // pixel track collection
   Handle<TrackCollection> pxltracks;
   iEvent.getByLabel(pxltracksSrc, pxltracks);
   //   cout << "====> number of pxl tracks "<< pxltracks->size() << endl;

   //   cout <<" Sim Track size = " << theSimTracks.size() << endl;

   for(size_t j = 0; j < theSimTracks.size(); j++){
     /*
     cout <<" sim track j = " << j
	  <<" track mom = " << theSimTracks[j].momentum().pt() 
	  <<" genpartIndex = " << theSimTracks[j].genpartIndex()
	  <<" type = " << theSimTracks[j].type()
	  <<" noGenpart = " << theSimTracks[j].noGenpart() 
	  <<" simTrackId = " << theSimTracks[j].trackId() << endl;
     */
   }

    int t = 0;

    std::string theTrackQuality = "highPurity";
    reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    ptSim1     = genpions[0].perp(); 
    etaSim1    = genpions[0].eta(); 
    phiSim1    = genpions[0].phi();

    ptSim2     = genpions[1].perp(); 
    etaSim2    = genpions[1].eta(); 
    phiSim2    = genpions[1].phi();

    //    cout <<" Reco track size = " << tracks->size() << endl;

    for(TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); track++) {
      //     const reco::TrackExtraRef & trkExtra = track->extra();

      int trkQuality = track->quality(trackQuality_);
      int trkNVhits  = track->numberOfValidHits();
      //      cout <<" track quality = " << trkQuality
      //	   <<" number of valid hits = " << trkNVhits << endl;

      double eECAL7x7i = -1000.; 
      double eECAL11x11i = -1000.; 
      double eHCAL3x3i = -1000.; 
      double eHCAL5x5i = -1000.; 

      const FreeTrajectoryState fts = trackAssociator_->getFreeTrajectoryState(iSetup, *track);
      TrackDetMatchInfo info = trackAssociator_->associate(iEvent, iSetup, fts, parameters_);
      if( info.isGoodEcal != 0 ) {
	const GlobalPoint point(info.trkGlobPosAtEcal.x(),info.trkGlobPosAtEcal.y(),info.trkGlobPosAtEcal.z());
	double etaimp = fabs(point.eta());
	/*
	  cout <<" on ECAL x = " << info.trkGlobPosAtEcal.x()
	  <<" y = " << info.trkGlobPosAtEcal.y()
	  <<" z = " << info.trkGlobPosAtEcal.z() 
	  <<" eta = " << etaimp
	  <<" phi = " << point.phi() << endl;
	*/
	// ECAL barrel or endcap and closest ecal cell
	if(etaimp < 1.479) {
	  const DetId ClosestCell = gEB->getClosestCell(point);
	  //	  EBRecHitCollection::const_iterator itEB = barrelRecHitsHandle->find(ClosestCell);
	  //	  if(itEB != barrelRecHitsHandle->end()) {
	  //	    cout <<" barrel energy of closest hit = " << itEB->energy() << endl;
	  CaloNavigator<DetId> theNavigator(ClosestCell,theEBTopology);
	  //	    EcalBarrelNavigator theNavigator(ClosestCell,theEBTopology);
	  eECAL7x7i = eECALmatrix(theNavigator,barrelRecHitsHandle,3,3);
	  eECAL11x11i = eECALmatrix(theNavigator,barrelRecHitsHandle,5,5);
	  //	  cout <<" barrel energy in 7x7 = " << eECAL7x7i << endl;
	  //	  }
	} else {
	  const DetId ClosestCell = gEE->getClosestCell(point);
	  //	  EERecHitCollection::const_iterator itEE = endcapRecHitsHandle->find(ClosestCell);
	  //	  if(itEE != endcapRecHitsHandle->end()) {
	  CaloNavigator<DetId> theNavigator(ClosestCell,theEETopology);
	  //	    EcalEndcapNavigator theNavigator(ClosestCell,theEETopology);
	  eECAL7x7i = eECALmatrix(theNavigator,endcapRecHitsHandle,3,3);
	  eECAL11x11i = eECALmatrix(theNavigator,endcapRecHitsHandle,5,5);
	  //	  cout <<" endcap energy in 7x7 ECAL cells = " << eECAL7x7i << endl;
	  //	  }
	}
	// closet HCAL cell
	
	const DetId ClosestCell = gHBHE->getClosestCell(point);
	HcalDetId hcd = ClosestCell;
	if(abs(hcd.ieta()) <= 25) {
	  /*
	    cout <<" eta inp = " << etaimp
	    <<" Hcal closest cell ieta = " << hcd.ieta()
	    <<" iphi = " << hcd.iphi()
	    <<" depth = " << hcd.depth()
	    <<" subdet = " << hcd.subdet() << endl;
	  */
	  eHCAL3x3i = eHCALmatrix(theHBHETopology, ClosestCell, Hithbhe,1,1);  
	  eHCAL5x5i = eHCALmatrix(theHBHETopology, ClosestCell, Hithbhe,2,2);  
	}
	/*
	  HBHERecHitCollection::const_iterator hbheItr = Hithbhe.find(ClosestCell);
	  if(hbheItr != Hithbhe.end()) {
	  cout <<" energy of closest hit = " << hbheItr->energy() << endl;
	  }
	  cout <<" before navigator " << endl;
	  CaloNavigator<DetId> theNavigator(ClosestCell,theHBHETopology);
	  cout <<" after navigator " << endl;
	  
	  cout <<" detector = " << ClosestCell.det()
	  <<" sub det = " << ClosestCell.subdetId()
	  <<" idsubdet = " << ClosestCell.subdet() 
	  <<" ieta = " << ClosestCell.ieta()
	  <<" iphi = " << ClosestCell.iphi() << endl;
	  
	  //	const CaloSubdetectorTopology* theHBHETopology = theCaloTopology->getSubdetectorTopology(ClosestCell);
	  if(!theHBHETopology) {cout <<" no pointer to HcalTopology " << endl;}
	  //        std::vector<DetId> vNeighboursDetId = theHBHETopology->north(ClosestCell);
	  DetId next = theNavigator.north();
	  cout <<" after north " << endl;
	*/
      }

      //      size_t Nhits = track->recHitsSize();
      //      size_t NpxlHits = track->hitPattern().numberOfValidPixelHits();
      SimTrackIds.clear();
      /*
      cout <<"    " << endl;
      cout <<" -> track " << t
	   <<" pt = " << track->pt()
	   <<" eta = " << track->eta()
	   <<" phi = " << track->phi()
	   <<" Nhits = " << track->recHitsSize()
	   <<" NpxlHits = " << track->hitPattern().numberOfValidPixelHits()
	  <<" ch2 = " << track->normalizedChi2()
	   <<" lost hits = " << track->numberOfLostHits()
	   <<" valid hits = " << track->numberOfValidHits()
	   <<" outerP = " << track->outerPt()
	   <<" outer X = " << track->outerPosition().x() << endl;
      */
      size_t ih = 0;
      for(trackingRecHit_iterator rhit = track->recHitsBegin();
	  rhit != track->recHitsEnd(); ++rhit) {
	//       const TrackingRecHitRef rhitRef = trkExtra->recHit(ih);
	if((*rhit)->isValid()) {
	  //try SimHit matching
	  float mindist = 999999;
	  float dist;
	  PSimHit closest;
	  matched.clear();	  
	  matched = hitAssociator->associateHit((**rhit));
	  /*
	  cout <<" " << endl;
	  cout <<" -----> irhit = " << ih
	       <<" rhitx = " << (*rhit)->localPosition().x()
	       <<" rhity = " << (*rhit)->localPosition().y()
	       <<" rhitz = " << (*rhit)->localPosition().z() << endl;
	  */
	  if(!matched.empty()){
	    //	    cout << "  size of matched PSimHit " << matched.size() << endl;
	    int ish = 0;
	    for(vector<PSimHit>::const_iterator m = matched.begin(); m < matched.end(); m++) {
	      if((*m).localPosition().y() < 0.0001) {
		dist = fabs((*rhit)->localPosition().x() - (*m).localPosition().x());
	      } else {
		float distx = (*rhit)->localPosition().x() - (*m).localPosition().x();
		float disty = (*rhit)->localPosition().y() - (*m).localPosition().y();
		dist = sqrt(distx*distx + disty*disty);
	      }
	      /*
	      cout <<"  --> isimhit = " << ish 
		   <<" shitx = " << (*m).localPosition().x()
		   <<" shity = " << (*m).localPosition().y()
		   <<" shitz = " << (*m).localPosition().z()
		   <<" sim track id = " << (*m).trackId() << endl; 
	      */
	      ish++;
	      if(dist < mindist) {
		mindist = dist;
		closest = (*m);
	      }
	    }
	    int trkid = closest.trackId();
	    //	    if(trkid <= (int) theSimTracks.size()) {
	      //	      if(theSimTracks[trkid-1].noGenpart() == 0) {
	    SimTrackIds.push_back(trkid);
	      /*
	      cout <<"          closest simtrack id = " << closest.trackId() 
		   <<" min dist = " << mindist << endl;
	      */
	      //	      }
	      //	    }
	  } 
	}
	ih++;
      }
      // count for sim track with max number of hits belonging to reco track 
      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n =0;
	n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[j];
	}
      }
      // track purity
      float purity = 0;
      if(SimTrackIds.size() != 0) {
	float totsim = nmax;
	float tothits = track->recHitsSize();//include pixel as well..
	purity = totsim/tothits ;
	/*
	cout << " ==> Track number # " << t 
	     << "# of rechits = " << track->recHitsSize() 
	     << " best matched simtrack id= " << idmax 
	     << " purity = " << purity << endl;
	     .*/
	//	     << " sim track mom = " << theSimTracks[idmax-1].momentum().pt() << endl;
      } else {
	//	cout <<"  !!!!  no HepMC particles associated with this pixel triplet " << endl;
      }
      // matching tracks with gen pions
      HepLorentzVector tracki(track->px(), track->py(), track->pz(), track->p());
      double DR1 = genpions[0].deltaR(tracki);
      if(DR1 < drTrk1) {
        ptTrk1     = tracki.perp();
        etaTrk1    = tracki.eta(); 
        phiTrk1    = tracki.phi();
        purityTrk1 = purity;
	drTrk1     = DR1;
	e1ECAL7x7  = eECAL7x7i;
	e1ECAL11x11= eECAL11x11i;
	e1HCAL3x3  = eHCAL3x3i;
	e1HCAL5x5  = eHCAL5x5i;
	trkQuality1 = trkQuality; 
	trkNVhits1 = trkNVhits;
	idmax1 = idmax;
      }
      double DR2 = genpions[1].deltaR(tracki);
      if(DR2 < drTrk2) {
        ptTrk2     = tracki.perp();
        etaTrk2    = tracki.eta(); 
        phiTrk2    = tracki.phi();
        purityTrk2 = purity;
	drTrk2     = DR2;
	e2ECAL7x7  = eECAL7x7i;
	e2ECAL11x11= eECAL11x11i;
	e2HCAL3x3  = eHCAL3x3i;
	e2HCAL5x5  = eHCAL5x5i;
	trkQuality2 = trkQuality; 
	trkNVhits2 = trkNVhits; 
	idmax2 = idmax;
      }
      t++;
    }

    /*
    cout <<" " << endl;

    cout <<" best matched track with 1st pion: ptTrk1 = " << ptTrk1
	 <<" etaTrk1 = " << etaTrk1
	 <<" phiTrk1 = " << phiTrk1
	 <<" DR1 = " << drTrk1
	 <<" purityTrk1 = " << purityTrk1 << endl;

    cout <<" best matched track with 2nd pion: ptTrk2 = " << ptTrk2
	 <<" etaTrk2 = " << etaTrk2
	 <<" phiTrk2 = " << phiTrk2
	 <<" DR2 = " << drTrk2
	 <<" purityTrk1 = " << purityTrk2 << endl;
    */

    // loop over the pixel triplet collection
    //    cout <<"   " << endl;
    //    cout <<" Loop over pixel triplet collection " << endl;
    t = 0;
    for(TrackCollection::const_iterator pxltrack = pxltracks->begin();
	pxltrack != pxltracks->end(); ++pxltrack) {
      SimTrackIds.clear();
      //      size_t Nhits = pxltrack->recHitsSize();
      //      size_t NpxlHits = pxltrack->hitPattern().numberOfValidPixelHits();
      /*
      cout <<"  " << endl;
      cout <<" pixel track " << t
	   <<" pt = " << pxltrack->pt()
	   <<" eta = " << pxltrack->eta()
	   <<" phi = " << pxltrack->phi()
	   <<" Nhits = " << Nhits 
	   <<" NpxlHits = " << NpxlHits 
	   <<" ch2 = " << pxltrack->normalizedChi2()
	   <<" lost hits = " << pxltrack->numberOfLostHits()
	   <<" valid hits = " << pxltrack->numberOfValidHits()
	   <<" outerP = " << pxltrack->outerPt()
	   <<" outer X = " << pxltrack->outerPosition().x() << endl;
      */
      size_t ih = 0;
      for(trackingRecHit_iterator rhit = pxltrack->recHitsBegin();
	  rhit != pxltrack->recHitsEnd(); ++rhit) {
	if((*rhit)->isValid()) {
	  //try SimHit matching
	  float mindist = 999999;
	  float dist;
	  PSimHit closest;
	  matched.clear();	  
	  matched = hitAssociator->associateHit((**rhit));
	  /*
	  cout <<" " << endl;
	  cout <<" -----> irhit = " << ih
	       <<" rhitx = " << (*rhit)->localPosition().x()
	       <<" rhity = " << (*rhit)->localPosition().y() << endl;
	  */
	  if(!matched.empty()){
	    int ish = 0;
	    for(vector<PSimHit>::const_iterator m = matched.begin(); m < matched.end(); m++) {
	      float distx = (*rhit)->localPosition().x() - (*m).localPosition().x();
	      float disty = (*rhit)->localPosition().y() - (*m).localPosition().y();
	      dist = sqrt(distx*distx + disty*disty);
	      /*
	      cout <<"  --> isimhit = " << ish 
		   <<" shitx = " << (*m).localPosition().x()
		   <<" shity = " << (*m).localPosition().y()
		   <<" sim track id = " << (*m).trackId() << endl;
	      */
	      ish++;
	      if(dist < mindist) {
		mindist = dist;
		closest = (*m);
	      }
	    }
	    int trkid = closest.trackId();
	    if(trkid <= (int) theSimTracks.size()) {
	      if(theSimTracks[trkid-1].noGenpart() == 0) {
		SimTrackIds.push_back(trkid);
		//		cout <<"          closest simtrack id = " << closest.trackId() 
		//		     <<" min dist = " << mindist << endl;
		cout <<" " << endl;
	      }
	    } else {
	      //	      cout <<" track ID > size of SimTracks " << endl;
	    }
	  } 
	}
	ih++;
      }
      // count for sim track with max number of hits belonging to reco track 
      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n =0;
	n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[j];
	}
      }
      float purity = 0;
      if(SimTrackIds.size() != 0) {
	float totsim = nmax;
	float tothits = pxltrack->recHitsSize();//include pixel as well..
	purity = totsim/tothits ;
	/*
	cout << " Track number # " << t 
	     << "# of rechits = " << pxltrack->recHitsSize() 
	     << " matched simtrack id= " << idmax 
	     << " purity = " << purity 
	     << " sim track mom = " << theSimTracks[idmax-1].momentum().perp() << endl;
	*/
      } else {
	//	cout <<"  !!!!  no HepMC particles associated with this pixel triplet " << endl;
      }
      // matching tracks with gen pions
      HepLorentzVector pxltracki(pxltrack->px(), pxltrack->py(), pxltrack->pz(), pxltrack->p());
      double DR1 = genpions[0].deltaR(pxltracki);
      if(DR1 < drPxl1) {
        ptPxl1     = pxltracki.perp();
        etaPxl1    = pxltracki.eta(); 
        phiPxl1    = pxltracki.phi();
        purityPxl1 = purity;
	drPxl1     = DR1;
      }
      double DR2 = genpions[1].deltaR(pxltracki);
      if(DR2 < drPxl2) {
        ptPxl2     = pxltracki.perp();
        etaPxl2    = pxltracki.eta(); 
        phiPxl2    = pxltracki.phi();
        purityPxl2 = purity;
	drPxl2     = DR2;
      }
      t++;
    }

    /*
    cout <<" " << endl;

    cout <<" best matched track with 1st pion: ptPxl1 = " << ptPxl1
	 <<" etaPxl1 = " << etaPxl1
	 <<" phiPxl1 = " << phiPxl1
	 <<" DR1 = " << drPxl1
	 <<" purityPxl1 = " << purityPxl1 << endl;

    cout <<" best matched track with 2nd pion: ptPxl2 = " << ptPxl2
	 <<" etaPxl2 = " << etaPxl2
	 <<" phiPxl2 = " << phiPxl2
	 <<" DR2 = " << drPxl2
	 <<" purityPxl1 = " << purityPxl2 << endl;
    */

   // fill tree
    t1->Fill();
    //    cout <<" idmax1 = " << idmax1 <<" idmax2 = " << idmax2 << endl;

    delete hitAssociator;
}


// ------------ method called once each job just before starting event loop  ------------
void 
SinglePionEfficiencyNew::beginJob(const edm::EventSetup& iSetup)
{

  using namespace edm;

  // TrackerHitAssociator::TrackerHitAssociator(const edm::Event&, const edm::ParameterSet&)
  // TrackerHitAssociator::TrackerHitAssociator(const edm::Event&)

  // create tree
  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
  t1 = new TTree("t1","analysis tree");
  // sim tracks
  t1->Branch("ptSim1",&ptSim1,"ptSim1/D");
  t1->Branch("etaSim1",&etaSim1,"etaSim1/D");
  t1->Branch("phiSim1",&phiSim1,"phiSim1/D");
  t1->Branch("ptSim2",&ptSim2,"ptSim2/D");
  t1->Branch("etaSim2",&etaSim2,"etaSim2/D");
  t1->Branch("phiSim2",&phiSim2,"phiSim2/D");
  // reco tracks
  t1->Branch("ptTrk1",&ptTrk1,"ptTrk1/D");
  t1->Branch("etaTrk1",&etaTrk1,"etaTrk1/D");
  t1->Branch("phiTrk1",&phiTrk1,"phiTrk1/D");
  t1->Branch("drTrk1",&drTrk1,"drTrk1/D");
  t1->Branch("purityTrk1",&purityTrk1,"purityTrk1/D");
  t1->Branch("ptTrk2",&ptTrk2,"ptTrk2/D");
  t1->Branch("etaTrk2",&etaTrk2,"etaTrk2/D");
  t1->Branch("phiTrk2",&phiTrk2,"phiTrk2/D");
  t1->Branch("drTrk2",&drTrk2,"drTrk2/D");
  t1->Branch("purityTrk2",&purityTrk2,"purityTrk2/D");
  // pxl tracks
  t1->Branch("ptPxl1",&ptPxl1,"ptPxl1/D");
  t1->Branch("etaPxl1",&etaPxl1,"etaPxl1/D");
  t1->Branch("phiPxl1",&phiPxl1,"phiPxl1/D");
  t1->Branch("drPxl1",&drPxl1,"drPxl1/D");
  t1->Branch("purityPxl1",&purityPxl1,"purityPxl1/D");
  t1->Branch("ptPxl2",&ptPxl2,"ptPxl2/D");
  t1->Branch("etaPxl2",&etaPxl2,"etaPxl2/D");
  t1->Branch("phiPxl2",&phiPxl2,"phiPxl2/D");
  t1->Branch("drPxl2",&drPxl2,"drPxl2/D");
  t1->Branch("purityPxl2",&purityPxl2,"purityPxl2/D");
  // Et in calo cones around MC jets
  t1->Branch("etCalo1",&etCalo1,"etCalo1/D");
  t1->Branch("etCalo2",&etCalo2,"etCalo2/D");
  t1->Branch("eCalo1",&eCalo1,"eCalo1/D");
  t1->Branch("eCalo2",&eCalo2,"eCalo2/D");
  // Et in 7x7, 11x11 ECAL crystals and 3x3, 5x5 HCAL towers
  t1->Branch("e1ECAL7x7",&e1ECAL7x7,"e1ECAL7x7/D");
  t1->Branch("e1ECAL11x11",&e1ECAL11x11,"e1ECAL11x11/D");
  t1->Branch("e2ECAL7x7",&e2ECAL7x7,"e2ECAL7x7/D");
  t1->Branch("e2ECAL11x11",&e2ECAL11x11,"e2ECAL11x11/D");
  t1->Branch("e1HCAL3x3",&e1HCAL3x3,"e1HCAL3x3/D");
  t1->Branch("e1HCAL5x5",&e1HCAL5x5,"e1HCAL5x5/D");
  t1->Branch("e2HCAL3x3",&e2HCAL3x3,"e2HCAL3x3/D");
  t1->Branch("e2HCAL5x5",&e2HCAL5x5,"e2HCAL5x5/D");
  // tracker quality and number of hits
  t1->Branch("trkQuality1",&trkQuality1,"trkQuality1/I");
  t1->Branch("trkQuality2",&trkQuality2,"trkQuality2/I");
  t1->Branch("trkNVhits1",&trkNVhits1,"trkNVhits1/I");
  t1->Branch("trkNVhits2",&trkNVhits2,"trkNVhits2/I");
  t1->Branch("idmax1",&idmax1,"idmax1/I");
  t1->Branch("idmax2",&idmax2,"idmax2/I");
  //
  return ;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SinglePionEfficiencyNew::endJob() {

  //  delete hitAssociator;
  hOutputFile->Write() ;
  hOutputFile->Close() ;
  return ;

}

double SinglePionEfficiencyNew::eECALmatrix(CaloNavigator<DetId>& navigator,edm::Handle<EcalRecHitCollection>& hits, int ieta, int iphi)
{
  DetId thisDet;
  std::vector<DetId> dets;
  dets.clear();
  EcalRecHitCollection::const_iterator hit;
  double energySum = 0.0;
  
  for (int dx = -ieta; dx < ieta+1; ++dx)
    {
      for (int dy = -iphi; dy < iphi+1; ++dy)
	{
	  //std::cout << "dx, dy " << dx << ", " << dy << std::endl;
	  thisDet = navigator.offsetBy(dx, dy);
	  navigator.home();
	  
	  if (thisDet != DetId(0))
	    {
	      hit = hits->find(thisDet);
	      if (hit != hits->end()) 
		{
		  dets.push_back(thisDet);
		  energySum += hit->energy();
		}
	    }
	}
    }
  return energySum;
}

double SinglePionEfficiencyNew::eHCALmatrix(const HcalTopology* topology, const DetId& det, const HBHERecHitCollection& hits, int ieta, int iphi)
{
  double energy = 0;

  if(ieta > 2) {
    cout <<" no matrix > 5x5 is coded ! " << endl;
    return energy;
  }
  // used dets during building the matrix
  std::vector<DetId> dets;
  dets.clear();

  //central tower
  HBHERecHitCollection::const_iterator hbheItr = hits.find(det);
  if(hbheItr != hits.end()) {
    energy = hbheItr->energy();
    dets.push_back(det);
  }
  HcalDetId depth = det;
  // max. three depths can be in endcap and we go to 2nd and 3rd from 1st where we are now
  for(int idepth = 0; idepth < 2; idepth++) {
    std::vector<DetId> vUpDetId = topology->up(depth);
    if(vUpDetId.size() != 0) {
      int n = std::count(dets.begin(),dets.end(),vUpDetId[0]);
      if(n == 0) {
	dets.push_back(vUpDetId[0]);
	HBHERecHitCollection::const_iterator hbheItrUp = hits.find(vUpDetId[0]);
	if(hbheItrUp != hits.end()) {
	  energy = energy + hbheItrUp->energy();
	}
      }
      depth = vUpDetId[0];
    }
  }

  // go to east from central tower 
  std::vector<DetId> vNeighboursDetId = topology->east(det);
  energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
  if(ieta == 2) {
    for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
      std::vector<DetId> vNeighboursDetIdc = topology->east(vNeighboursDetId[ii]);
      energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
    }
  }
  vNeighboursDetId.clear();
  // go to west from central tower 
  vNeighboursDetId = topology->west(det);
  energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
  if(ieta == 2) {
    for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
      std::vector<DetId> vNeighboursDetIdc = topology->west(vNeighboursDetId[ii]);
      energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
    }
  }
  vNeighboursDetId.clear();


  // do steps in phi to north
  DetId detnorth = det;
  for (int inorth = 0; inorth < iphi; inorth++) {
    std::vector<DetId> NorthDetId = topology->north(detnorth);
    energy = energy + eHCALneighbours(NorthDetId, dets, topology, hits);

    // go to east  
    vNeighboursDetId = topology->east(NorthDetId[0]);
    energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
    if(ieta == 2) {
      for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
	std::vector<DetId> vNeighboursDetIdc = topology->east(vNeighboursDetId[ii]);
	energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
      }
    }
    vNeighboursDetId.clear();
    // go to west 
    vNeighboursDetId = topology->west(NorthDetId[0]);
    energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
    if(ieta == 2) {
      for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
	std::vector<DetId> vNeighboursDetIdc = topology->west(vNeighboursDetId[ii]);
	energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
      }
    }
    detnorth = NorthDetId[0];
    vNeighboursDetId.clear();
  }

  // do steps in phi to south
  DetId detsouth = det;
  for (int isouth = 0; isouth < iphi; isouth++) {
    std::vector<DetId> SouthDetId = topology->south(detsouth);
    energy = energy + eHCALneighbours(SouthDetId, dets, topology, hits);

    // go to east  
    vNeighboursDetId = topology->east(SouthDetId[0]);
    energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
    if(ieta == 2) {
      for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
	std::vector<DetId> vNeighboursDetIdc = topology->east(vNeighboursDetId[ii]);
	energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
      }
    }
    vNeighboursDetId.clear();
    // go to west 
    vNeighboursDetId = topology->west(SouthDetId[0]);
    energy = energy + eHCALneighbours(vNeighboursDetId, dets, topology, hits);
    if(ieta == 2) {
      for (int ii = 0; ii < (int) vNeighboursDetId.size(); ii++) {
	std::vector<DetId> vNeighboursDetIdc = topology->west(vNeighboursDetId[ii]);
	energy = energy + eHCALneighbours(vNeighboursDetIdc, dets, topology, hits);
      }
    }
    detsouth = SouthDetId[0];
    vNeighboursDetId.clear();
  }

  // done
  /*
  HcalDetId cht = dets[0];
  if(abs(cht.ieta()) == 15 && cht.subdet() == 2) {
    cout <<" ==> Tower used " << endl; 
    double energy = 0.;
    for(int i = 0; i < dets.size(); i++) {
      HcalDetId hdet = dets[i];
      double hitenergy = 0.;
      HBHERecHitCollection::const_iterator hbheItr = hits.find(dets[i]);
      if(hbheItr != hits.end()) {
	hitenergy = hbheItr->energy();
	energy = energy + hitenergy; 
      }
      cout <<" subdet = " << hdet.subdet() 
	   <<" ieta = " << hdet.ieta()
	   <<" phi = " <<  hdet.iphi() 
	   <<" idepth = " << hdet.depth() 
	   <<" hitsenergy = " << hitenergy 
	   <<" energy = " << energy << endl;
    }
  }
  */
  return energy;
}

double SinglePionEfficiencyNew::eHCALneighbours(std::vector<DetId>& vNeighboursDetId, std::vector<DetId>& dets, 
						const HcalTopology* topology, const HBHERecHitCollection& hits)
{
  double eHCALneighbour = 0.;
  for(int i = 0; i < (int) vNeighboursDetId.size(); i++) {
    int n = std::count(dets.begin(),dets.end(),vNeighboursDetId[i]);
    if(n != 0) continue; 
    dets.push_back(vNeighboursDetId[i]);
    HBHERecHitCollection::const_iterator hbheItr = hits.find(vNeighboursDetId[i]);
    if(hbheItr != hits.end()) {
      eHCALneighbour = eHCALneighbour + hbheItr->energy();
    }
    // go into depth (for endcap)

    HcalDetId depth = vNeighboursDetId[i];
    // max. three depths can be in endcap and we go to 2nd and 3rd from 1st where we are now
    for(int idepth = 0; idepth < 2; idepth++) {
      /*
      cout <<" idepth = " << idepth
	   <<" ieta = " << depth.ieta()
	   <<" iphi = " << depth.iphi()
	   <<" depth = " << depth.depth() << endl;
      */
      std::vector<DetId> vUpDetId = topology->up(depth);
      if(vUpDetId.size() != 0) {
	int n = std::count(dets.begin(),dets.end(),vUpDetId[0]);
	if(n == 0) {
	  dets.push_back(vUpDetId[0]);
	  HBHERecHitCollection::const_iterator hbheItrUp = hits.find(vUpDetId[0]);
	  if(hbheItrUp != hits.end()) {
	    eHCALneighbour = eHCALneighbour + hbheItrUp->energy();
	  }
	}
	depth = vUpDetId[0];
      }
    }

  }
  return eHCALneighbour;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SinglePionEfficiencyNew);
