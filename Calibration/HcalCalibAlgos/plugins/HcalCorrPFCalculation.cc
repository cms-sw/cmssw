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
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 private:

  // virtual void fillRecHitsTmp(int subdet_, edm::Event const& ev);
  //double dR(double eta1, double phi1, double eta2, double phi2);
  //double phi12(double phi1, double en1, double phi2, double en2);
  //double dPhiWsign(double phi1,double phi2);

double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint);

 std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string eventype_;
  std::string sign_;
  std::string mc_;
  bool        Respcorr_;
  bool        PFcorr_;
  bool        Conecorr_;
  double        radius_;

  double energyECALmip;

  Bool_t doHF;
  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  // int subdet_;

  // single/multi-particle sample (1/2)
  int etype_;
  int iz;
  //  int imc;


  int nevtot;
  int hasresp;
 const HcalRespCorrs* respRecalib;
  const HcalPFCorrs* pfRecalib;

    SteppingHelixPropagator* stepPropF;
  MagneticField *theMagField;

  TProfile *nCells, *nCellsNoise, *enHcal, *enHcalNoise;
  TH1F *enEcalB, *enEcalE;

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

};


HcalCorrPFCalculation::HcalCorrPFCalculation(edm::ParameterSet const& conf) {

  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
  
  //  Char_t histo[20];

  //hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");
  //ecalselector_ = conf.getUntrackedParameter<std::string>("ecalselector", "yes");
  //eventype_     = conf.getUntrackedParameter<std::string>("eventype", "single");
  //sign_         = conf.getUntrackedParameter<std::string>("sign", "*");
  //mc_           = conf.getUntrackedParameter<std::string>("mc", "yes");
  Respcorr_        = conf.getUntrackedParameter<bool>("RespcorrAdd", false);
  PFcorr_        = conf.getUntrackedParameter<bool>("PFcorrAdd", false);
  Conecorr_        = conf.getUntrackedParameter<bool>("ConeCorrAdd", true);
  radius_       = conf.getUntrackedParameter<double>("ConeRadiusCm", 40.);
  energyECALmip = conf.getParameter<double>("energyECALmip");

  //famos_        = conf.getUntrackedParameter<bool>("Famos", false);
  //  std::cout << "*** famos_ = " << famos_ << std::endl; 

  edm::ParameterSet parameters = conf.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();

  taECALCone_=conf.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  taHCALCone_=conf.getUntrackedParameter<double>("TrackAssociatorHCALCone",0.6);


  //iz = 1;
  //if(sign_ == "-") iz = -1;
  //if(sign_ == "*") iz = 0;

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


HcalCorrPFCalculation::~HcalCorrPFCalculation() {

  //  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void HcalCorrPFCalculation::beginJob(const edm::EventSetup& c){
  
  // TProfile *nCells, *nCellsNoise, *en, *enNoise;
  //TFile *rootFile;
  
  rootFile = new TFile(outputFile_.c_str(),"RECREATE");
  
  
  nCells = new TProfile("nCells", "nCells", 83, -41.5, 41.5); 
  nCellsNoise = new TProfile("nCellsNoise", "nCellsNoise", 83, -41.5, 41.5); 
  
  enHcal = new TProfile("enHcal", "enHcal", 83, -41.5, 41.5); 
  enHcalNoise = new TProfile("enHcalNoise", "enHcalNoise", 83, -41.5, 41.5); 
  
  enEcalB = new TH1F("enEcalB", "enEcalB", 500, -5,50); 
  enEcalE = new TH1F("enEcalE", "enEcalE", 500, -5,50); 
  
  // Response corrections w/o re-rechitting
  hasresp=0;

  try{
    
    edm::ESHandle <HcalRespCorrs> recalibCorrs;
    c.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
    respRecalib = recalibCorrs.product();
    
    edm::ESHandle <HcalPFCorrs> pfCorrs;
    c.get<HcalPFCorrsRcd>().get("recalibrate",pfCorrs);
    pfRecalib = pfCorrs.product();
    
    hasresp=1;
    // LogMessage("CalibConstants")<<"   OK ";
    
  }catch(const cms::Exception & e) {
    LogWarning("CalibConstants")<<"   Not Found!! ";
  }
  
  
    /*
  edm::ESHandle<MagneticField> bField;
  c.get<IdealMagneticFieldRecord>().get(bField);
  stepPropF  = new SteppingHelixPropagator(&*bField,alongMomentum);
  stepPropF->setMaterialMode(false);
  stepPropF->applyRadX0Correction(true);
  */
  
}
void HcalCorrPFCalculation::endJob() 
{

  nCells -> Write();
  nCellsNoise -> Write();
  enHcal -> Write();
  enHcalNoise -> Write();
  
  enEcalB -> Write();
  enEcalE -> Write();

  rootFile->Close();
}


void HcalCorrPFCalculation::analyze(edm::Event const& ev, edm::EventSetup const& c) {
  
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
      mom_MC = (*p)->momentum().mag();
      if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
      GlobalVector mom ((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
      int charge = -1;
      
      if(abs((*p)->pdg_id())==211) charge = (*p)->pdg_id()/abs((*p)->pdg_id()); // pions only !!!
      
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
      double eEcalCone   = 0.;
      // int numrechitsEcal = 0;
      
      //Hcal:
      float HcalCone    = 0.;
      float HcalConeNoise    = 0.;
      int UsedCells = 0;
      int UsedCellsNoise = 0;
      
      float dddeta = 1000.;
      float dddphi = 1000.;
      int iphitrue = 1234;
      int ietatrue = 1234;
      
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

      Float_t recal = 1.0;

      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	{ 
	  Float_t resprecal = 1.;
	  Float_t pfrecal = 1.;
	  if(hasresp==1) {
	    if(Respcorr_) resprecal = respRecalib -> getValues(hhit->detid())->getValue();
	    if(PFcorr_)   pfrecal = recal * pfRecalib   -> getValues(hhit->detid())->getValue();
	    recal = resprecal*pfrecal;
	    //cout<<"recal:"<<recal<<endl;
	  }

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
	  
	  //here looking for iEta and iPhi of track position (we know only eta and phi from info)
	  if (deta<dddeta) {
	    ietatrue = ietahit;
	    dddeta=deta;
	  }
	  
	  if (dphi<dddphi) {
	    iphitrue = iphihit;
	    dddphi=dphi;
	  }
	  
	  dr =  getDistInPlaneSimple(gPointHcal,pos);
	  
	  if(dr<radius_ && enehit>0.1) 
	    {
	      HcalCone += enehit;	    
	      UsedCells++;

	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  
		  int iphihit2 = (hhit2->id()).iphi();
		  int ietahit2 = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  
		  if (iphihitNoise == iphihit2 && ietahitNoise == ietahit2 && depthhitNoise == depthhit2)
		    {
		      HcalConeNoise += hhit2->energy()*recal;
		      UsedCellsNoise++;
		    }
		}
	      
	  }
	} //end of all HBHE hits cycle
      


      if(doHF){
	for (HFRecHitCollection::const_iterator hhit=Hithf.begin(); hhit!=Hithf.end(); hhit++) 
	{
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
	  
	  if (deta<dddeta) {
	    ietatrue = ietahit;
	    dddeta=deta;
	  }
	  
	  if (dphi<dddphi) {
	    iphitrue = iphihit;
	    dddphi=dphi;
	  }
	  
	  dr =  getDistInPlaneSimple(gPointHcal,pos);
	  
	  if(dr<radius_ && enehit>0.1) 
	    {
	      
	      HcalCone += enehit;	    
	      UsedCells++;
	      
	      for (HFRecHitCollection::const_iterator hhit2=Hithf.begin(); hhit2!=Hithf.end(); hhit2++) 
		{
		  int iphihit2 = (hhit2->id()).iphi();
		  int ietahit2 = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  
		  if (iphihitNoise == iphihit2 && ietahitNoise == ietahit2 && depthhitNoise == depthhit2)
		    {
		      HcalConeNoise += hhit2->energy()*recal;
		      UsedCellsNoise++;
		    }
		}
	      
	    }
	} //end of all HF hits cycle
      } //end of doHF

      
      Bool_t passCuts = kFALSE;
      //passCuts=kTRUE; 
      if(eEcalCone < energyECALmip) passCuts = kTRUE;
      
      if(passCuts)
	{
	  enHcal -> Fill(ietatrue,  HcalCone);
	  nCells -> Fill(ietatrue,  UsedCells);
	  enHcalNoise -> Fill(ietatrue,  HcalConeNoise);
	  nCellsNoise -> Fill(ietatrue,  UsedCellsNoise); 
	}
      
      
      
    }
  
}

  /*
double HcalCorrPFCalculation::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}
  */
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "DQMServices/Core/interface/DQMStore.h"

//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(HcalCorrPFCalculation);

//define this as a plug-in
DEFINE_FWK_MODULE(HcalCorrPFCalculation);
