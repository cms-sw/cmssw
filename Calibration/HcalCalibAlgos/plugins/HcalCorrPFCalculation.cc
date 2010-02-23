// $Id: HcalCorrPFCalculation.cc,v 1.16 2010/01/25 22:13:27 hegner Exp $

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "FWCore/Framework/interface/Selector.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalPFCorrs.h"
#include "CondFormats/DataRecord/interface/HcalPFCorrsRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Calibration/HcalCalibAlgos/src/MaxHit_struct.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TProfile.h"

using namespace edm;
using namespace std;
using namespace reco;

class HcalCorrPFCalculation : public edm::EDAnalyzer {
 public:
  HcalCorrPFCalculation(edm::ParameterSet const& conf);
  ~HcalCorrPFCalculation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c);
  virtual void beginJob() ;
  virtual void endJob() ;
 private:
  double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint);
  
  double RecalibFactor(HcalDetId id);

  bool        Respcorr_;
  bool        PFcorr_;
  bool        Conecorr_;
  double        radius_;

  double energyECALmip;

  Bool_t doHF;
  Bool_t AddRecalib;
  int nevtot;

  const HcalRespCorrs* respRecalib;
  const HcalPFCorrs* pfRecalib;

  SteppingHelixPropagator* stepPropF;
  MagneticField *theMagField;

  edm::Service<TFileService> fs;

  TProfile *nCells, *nCellsNoise, *enHcal, *enHcalNoise;
  TH1F *enEcalB, *enEcalE;
  TTree *pfTree;
  TFile *rootFile;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;
  double taECALCone_;
  double taHCALCone_;

  const CaloGeometry* geo;

  Float_t xTrkEcal;
  Float_t yTrkEcal;
  Float_t zTrkEcal;

  Float_t xTrkHcal;
  Float_t yTrkHcal;
  Float_t zTrkHcal;

  double eEcalCone, eHcalCone, eHcalConeNoise;
  // int numrechitsEcal = 0;
  
  int UsedCells, UsedCellsNoise;

  //  Float_t etaTrack, phiTrack;
  Int_t  iPhi, iEta;
      
};


HcalCorrPFCalculation::HcalCorrPFCalculation(edm::ParameterSet const& conf) {

  //  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
    Respcorr_        = conf.getUntrackedParameter<bool>("RespcorrAdd", false);
  PFcorr_        = conf.getUntrackedParameter<bool>("PFcorrAdd", false);
  Conecorr_        = conf.getUntrackedParameter<bool>("ConeCorrAdd", true);
  radius_       = conf.getUntrackedParameter<double>("ConeRadiusCm", 40.);
  energyECALmip = conf.getParameter<double>("energyECALmip");

  edm::ParameterSet parameters = conf.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();

  taECALCone_=conf.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  taHCALCone_=conf.getUntrackedParameter<double>("TrackAssociatorHCALCone",0.6);

}

double HcalCorrPFCalculation::getDistInPlaneSimple(const GlobalPoint caloPoint,
                            const GlobalPoint rechitPoint)
{

  // Simplified version of getDistInPlane
  // Assume track direction is origin -> point of hcal intersection

  const GlobalVector caloIntersectVector(caloPoint.x(),
                                         caloPoint.y(),
                                         caloPoint.z());

  const GlobalVector caloIntersectUnitVector = caloIntersectVector.unit();

  const GlobalVector rechitVector(rechitPoint.x(),
                                  rechitPoint.y(),
                                  rechitPoint.z());

  const GlobalVector rechitUnitVector = rechitVector.unit();
  double dotprod = caloIntersectUnitVector.dot(rechitUnitVector);
  double rechitdist = caloIntersectVector.mag()/dotprod;


  const GlobalVector effectiveRechitVector = rechitdist*rechitUnitVector;
  const GlobalPoint effectiveRechitPoint(effectiveRechitVector.x(),
                                         effectiveRechitVector.y(),
                                         effectiveRechitVector.z());


  GlobalVector distance_vector = effectiveRechitPoint-caloPoint;

  if (dotprod > 0.)
  {
    return distance_vector.mag();
  }
  else
  {
    return 999999.;
  }
}

double  HcalCorrPFCalculation::RecalibFactor(HcalDetId id)
{
  Float_t resprecal = 1.;
  Float_t pfrecal = 1.;
  if(AddRecalib) {
    if(Respcorr_) resprecal = respRecalib -> getValues(id)->getValue();
    if(PFcorr_)   pfrecal = pfRecalib   -> getValues(id)->getValue();
  }
  Float_t factor = resprecal*pfrecal;
  return factor;
}

HcalCorrPFCalculation::~HcalCorrPFCalculation() {

  
}

void HcalCorrPFCalculation::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  AddRecalib=kFALSE;

  try{

    edm::ESHandle <HcalRespCorrs> recalibCorrs;
    c.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
    respRecalib = recalibCorrs.product();

    edm::ESHandle <HcalPFCorrs> pfCorrs;
    c.get<HcalPFCorrsRcd>().get("recalibrate",pfCorrs);
    pfRecalib = pfCorrs.product();

    AddRecalib = kTRUE;;
    // LogMessage("CalibConstants")<<"   OK ";

  }catch(const cms::Exception & e) {
    LogWarning("CalibConstants")<<"   Not Found!! ";
  }


  
    edm::ESHandle<CaloGeometry> pG;
    c.get<CaloGeometryRecord>().get(pG);
    geo = pG.product();
    
    parameters_.useEcal = true;
    parameters_.useHcal = true;
    parameters_.useCalo = false;
    parameters_.useMuon = false;
    parameters_.dREcal = taECALCone_;
    parameters_.dRHcal = taHCALCone_;    

    
  //  double eta_bin[42]={0.,.087,.174,.261,.348,.435,.522,.609,.696,.783,
  //.870,.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,
  //2.322,2.500,2.650,2.853,3.000,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191};
  
  // MC info 
  double phi_MC = -999999.;  // phi of initial particle from HepMC
  double eta_MC = -999999.;  // eta of initial particle from HepMC
  double mom_MC = 50.;  // P of initial particle from HepMC
  bool MC = false;
  
  // MC information
  
    
  edm::Handle<edm::HepMCProduct> evtMC;
  //  ev.getByLabel("VtxSmeared",evtMC);
  ev.getByLabel("generator",evtMC);
  if (!evtMC.isValid()) 
    {
      std::cout << "no HepMCProduct found" << std::endl;    
    } 
  else 
    {
      MC=true;
      //    std::cout << "*** source HepMCProduct found"<< std::endl;
    }  
  
  // MC particle with highest pt is taken as a direction reference  
  double maxPt = -99999.;
  int npart    = 0;
  
  GlobalPoint pos (0,0,0);
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evtMC->GetEvent()));
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) 
    {
      double phip = (*p)->momentum().phi();
      double etap = (*p)->momentum().eta();
      double pt  = (*p)->momentum().perp();
      mom_MC = (*p)->momentum().rho();
      if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
      GlobalVector mom ((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
      int charge = -1;
      
      if(abs((*p)->pdg_id())==211) charge = (*p)->pdg_id()/abs((*p)->pdg_id()); // pions only !!!
      else continue;
      
      const FreeTrajectoryState *freetrajectorystate_ =
	new FreeTrajectoryState(pos, mom ,charge , &(*theMagField));
      
      TrackDetMatchInfo info = trackAssociator_.associate(ev, c, *freetrajectorystate_ , parameters_);
      // TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,trackAssociator_.getFreeTrajectoryState(iSetup, *trit), parameters_);

      float etahcal=info.trkGlobPosAtHcal.eta();
      float phihcal=info.trkGlobPosAtHcal.phi();

      float etaecal=info.trkGlobPosAtEcal.eta();
      //    float phiecal=info.trkGlobPosAtEcal.phi();
      
      
      xTrkEcal=info.trkGlobPosAtEcal.x();
      yTrkEcal=info.trkGlobPosAtEcal.y();
      zTrkEcal=info.trkGlobPosAtEcal.z();
      
      xTrkHcal=info.trkGlobPosAtHcal.x();
      yTrkHcal=info.trkGlobPosAtHcal.y();
      zTrkHcal=info.trkGlobPosAtHcal.z();
      
      GlobalPoint gPointHcal(xTrkHcal,yTrkHcal,zTrkHcal);
      
      GlobalPoint gPointEcal(xTrkEcal,yTrkEcal,zTrkEcal);
      
      if (etahcal>2.6) doHF = kTRUE;
   
      
      
      edm::Handle<HBHERecHitCollection> hbhe;
      ev.getByType(hbhe);
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
      
      edm::Handle<HFRecHitCollection> hfcoll;
      ev.getByType(hfcoll);
      const HFRecHitCollection Hithf = *(hfcoll.product());
      
      
      edm::Handle<HORecHitCollection> hocoll;
      ev.getByType(hocoll);
      const HORecHitCollection Hitho = *(hocoll.product());
      
      
      edm::Handle<EERecHitCollection> ecalEE;
      ev.getByLabel("ecalRecHit","EcalRecHitsEE",ecalEE);
      const EERecHitCollection HitecalEE = *(ecalEE.product());

      edm::Handle<EBRecHitCollection> ecalEB;
      ev.getByLabel("ecalRecHit","EcalRecHitsEB",ecalEB);
      const EBRecHitCollection HitecalEB = *(ecalEB.product());
      
      
      
      // energy in ECAL
      eEcalCone   = 0.;
      // int numrechitsEcal = 0;
      
      //Hcal:
      eHcalCone = 0.;
      eHcalConeNoise = 0.;
      UsedCells = 0;
      UsedCellsNoise = 0;
      

      Int_t iphitrue = -10;
      Int_t ietatrue = 100;
      
      if (etahcal<1.392) 
	{
	  const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
	  //    const GlobalPoint tempPoint(newx, newy, newz);
	  //const DetId tempId = gHB->getClosestCell(tempPoint);
	  const HcalDetId tempId = gHB->getClosestCell(gPointHcal);
	  ietatrue = tempId.ieta();
	  iphitrue = tempId.iphi();
	}
      
      if (etahcal>1.392 &&  etahcal<3.0) 
	{
	  const CaloSubdetectorGeometry* gHE = geo->getSubdetectorGeometry(DetId::Hcal,HcalEndcap);
	  const HcalDetId tempId = gHE->getClosestCell(gPointHcal);
	  ietatrue = tempId.ieta();
	  iphitrue = tempId.iphi();
	}
      
      if (etahcal>3.0 &&  etahcal<5.0) 
	{
	  const CaloSubdetectorGeometry* gHF = geo->getSubdetectorGeometry(DetId::Hcal,HcalForward);
	  const HcalDetId tempId = gHF->getClosestCell(gPointHcal);
	  ietatrue = tempId.ieta();
	  iphitrue = tempId.iphi();
	}
      
      //Calculate Ecal energy:      
      for (EBRecHitCollection::const_iterator ehit=HitecalEB.begin(); ehit!=HitecalEB.end(); ehit++)	
	{
	  
	  GlobalPoint pos = geo->getPosition(ehit->detid());
	  float dr =  getDistInPlaneSimple(gPointEcal,pos);
	  if (dr < 10.) eEcalCone += ehit->energy();
	}

      for (EERecHitCollection::const_iterator ehit=HitecalEE.begin(); ehit!=HitecalEE.end(); ehit++)	
	{
	  
	  GlobalPoint pos = geo->getPosition(ehit->detid());
	  float dr =  getDistInPlaneSimple(gPointEcal,pos);
      	  if (dr < 10.) eEcalCone += ehit->energy();
	}
      if(abs(etaecal)<1.5) enEcalB -> Fill(eEcalCone); 
      if(abs(etaecal)>1.5 && abs(etaecal)<3.1) enEcalE -> Fill(eEcalCone); 


      MaxHit_struct MaxHit;

      MaxHit.hitenergy=-100.;


      Float_t recal = 1.0;


      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	{ 
	
	  recal = RecalibFactor(hhit->detid());
	  //cout<<"recal: "<<recal<<endl;

	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  float phihit = pos.phi();
	  float etahit = pos.eta();
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;

	  //Set noise RecHit opposite to track hits
	  int  iphihitNoise = iphihit >36 ? iphihit-36 : iphihit+36;
	  int ietahitNoise  = ietahit;
	  int depthhitNoise = depthhit;
	  
	  double dphi = fabs(phihcal - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  double deta = fabs(etahcal - etahit); 
	  double dr = sqrt(dphi*dphi + deta*deta);
	  
	  //dr =  getDistInPlaneSimple(gPointHcal,pos);

	  if(dr<0.5) 
	    {
	      
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  int iphihit2  = (hhit2->id()).iphi();
		  int ietahit2  = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy() * recal;
		  
		  if (iphihit==iphihit2 && ietahit==ietahit2  && depthhit!=depthhit2)  enehit = enehit+enehit2;
		
		}	  	  
	      
	      //Find a Hit with Maximum Energy
	      
	      if(enehit > MaxHit.hitenergy) 
		{
 		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.dr   = dr;
		  //MaxHit.depthhit  = (hhit->id()).depth();
		  MaxHit.depthhit  = 1;
		}
	    }
	  
	  if(dr<radius_ && enehit>0.01) 
	    {
	      eHcalCone += enehit;	    
	      UsedCells++;

	      // cout<<"track: ieta "<<ietahit<<" iphi: "<<iphihit<<" depth: "<<depthhit<<" energydepos: "<<enehit<<endl;
	      
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  recal = RecalibFactor(hhit2->detid());
		  int iphihit2 = (hhit2->id()).iphi();
		  int ietahit2 = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy()* recal;	
		  
		  if (iphihitNoise == iphihit2 && ietahitNoise == ietahit2 && depthhitNoise == depthhit2 && enehit2>0.01)
		    {
		      
		      eHcalConeNoise += hhit2->energy()*recal;
		      UsedCellsNoise++;
		      //cout<<"Noise: ieta "<<ietahit2<<" iphi: "<<iphihit2<<" depth: "<<depthhit2<<" energydepos: "<<enehit2<<endl;
		    }
		}
	    }
	} //end of all HBHE hits cycle
      
      if(doHF){
	for (HFRecHitCollection::const_iterator hhit=Hithf.begin(); hhit!=Hithf.end(); hhit++) 
	{

	  recal = RecalibFactor(hhit->detid());

	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  float phihit = pos.phi();
	  float etahit = pos.eta();
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;

	  //Set noise RecHit opposite to track hits
	  int  iphihitNoise = iphihit >36 ? iphihit-36 : iphihit+36;
	  int ietahitNoise  = ietahit;
	  int depthhitNoise = depthhit;
	  
	  
	  double dphi = fabs(phihcal - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  double deta = fabs(etahcal - etahit); 
	  double dr = sqrt(dphi*dphi + deta*deta);
	  
	  dr =  getDistInPlaneSimple(gPointHcal,pos);
	  

	  if(dr<60.) 
	    {
	      //Find a Hit with Maximum Energy
	      
	      if(enehit > MaxHit.hitenergy) 
		{
 		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.dr   = dr;
		  MaxHit.depthhit  = 1;
		}
	    }
	  
	  if(dr<radius_ && enehit>0.01) 
	    {
	      
	      eHcalCone += enehit;	    
	      UsedCells++;
	      
	      for (HFRecHitCollection::const_iterator hhit2=Hithf.begin(); hhit2!=Hithf.end(); hhit2++) 
		{
		  recal = RecalibFactor(hhit2->detid());

		  int iphihit2 = (hhit2->id()).iphi();
		  int ietahit2 = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy()* recal;	
		  
		  if (iphihitNoise == iphihit2 && ietahitNoise == ietahit2 && depthhitNoise == depthhit2 && enehit2>0.01)
		    {
		      eHcalConeNoise += hhit2->energy()*recal;
		      UsedCellsNoise++;
		    }
		}
	      
	    }
	} //end of all HF hits cycle
      } //end of doHF

      int dieta_M_P = 100;
      int diphi_M_P = 100;
      if(MaxHit.ietahitm*ietatrue>0) {dieta_M_P = abs (MaxHit.ietahitm-ietatrue);}
      if(MaxHit.ietahitm*ietatrue<0) {dieta_M_P = abs(MaxHit.ietahitm-ietatrue)-1;}
      diphi_M_P = abs(MaxHit.iphihitm-iphitrue);
      diphi_M_P =  diphi_M_P>36 ? 72-diphi_M_P : diphi_M_P; 

      float iDr = sqrt(diphi_M_P*diphi_M_P+dieta_M_P*dieta_M_P);

      
      Bool_t passCuts = kFALSE;
      //passCuts=kTRUE; 
      if(eEcalCone < energyECALmip && iDr<2.) passCuts = kTRUE;
      
      if(passCuts)
	{
	  enHcal -> Fill(ietatrue,  eHcalCone);
	  nCells -> Fill(ietatrue,  UsedCells);
	  enHcalNoise -> Fill(ietatrue,  eHcalConeNoise);
	  nCellsNoise -> Fill(ietatrue,  UsedCellsNoise); 

	  iEta = ietatrue;
	  iPhi = iphitrue;

	  pfTree->Fill();
	}
    }
}

void HcalCorrPFCalculation::beginJob(){
  
  // TProfile *nCells, *nCellsNoise, *en, *enNoise;
  //TFile *rootFile;
  
  //rootFile = new TFile(outputFile_.c_str(),"RECREATE");
  
  
  nCells = fs->make<TProfile>("nCells", "nCells", 83, -41.5, 41.5); 
  nCellsNoise = fs->make<TProfile>("nCellsNoise", "nCellsNoise", 83, -41.5, 41.5); 
  
  enHcal = fs->make<TProfile>("enHcal", "enHcal", 83, -41.5, 41.5); 
  enHcalNoise =  fs->make<TProfile>("enHcalNoise", "enHcalNoise", 83, -41.5, 41.5); 
  
  enEcalB = fs->make<TH1F>("enEcalB", "enEcalB", 500, -5,50); 
  enEcalE = fs->make<TH1F>("enEcalE", "enEcalE", 500, -5,50); 

 pfTree = new TTree("pfTree", "Tree for pf info");

 pfTree->Branch("eEcalCone", &eEcalCone, "eEcalCone/F");
 pfTree->Branch("eHcalCone", &eHcalCone, "eHcalCone/F");
 pfTree->Branch("eHcalConeNoise", &eHcalConeNoise, "eHcalConeNoise/F");

 pfTree->Branch("UsedCellsNoise", &UsedCellsNoise, "UsedCellsNoise/I");
 pfTree->Branch("UsedCells", &UsedCells, "UsedCells/I");

 
 // pfTree->Branch("etaTrack", &etaTrack, "etaTrack/F");
 //pfTree->Branch("phiTrack", &phiTrack, "phiTrack/F");
 
 pfTree->Branch("iEta", &iEta, "iEta/I");
 pfTree->Branch("iPhi", &iPhi, "iPhi/I");
 

}
void HcalCorrPFCalculation::endJob() 
{

  /*
  nCells -> Write();
  nCellsNoise -> Write();
  enHcal -> Write();
  enHcalNoise -> Write();
  
  enEcalB -> Write();
  enEcalE -> Write();

  rootFile->Close();
  */
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


//define this as a plug-in
DEFINE_FWK_MODULE(HcalCorrPFCalculation);
