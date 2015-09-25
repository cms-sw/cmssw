

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

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


#include "Calibration/HcalCalibAlgos/interface/CommonUsefulStuff.h"

#include <iostream>
#include "TProfile.h"

using namespace edm;
using namespace std;
using namespace reco;

class HcalCorrPFCalculation : public edm::EDAnalyzer {
 public:
  HcalCorrPFCalculation(edm::ParameterSet const& conf);
  ~HcalCorrPFCalculation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c) override;
  virtual void beginJob() override ;
  virtual void endJob() override ;
 private:
  
  double RecalibFactor(HcalDetId id);

  bool        Respcorr_;
  bool        PFcorr_;
  bool        Conecorr_;
  double        radius_;

  double clusterConeSize_,associationConeSize_, ecalCone_, neutralIsolationCone_, trackIsolationCone_;
  float eECAL, eECAL09cm, eECAL40cm;   
// double energyECALmip;

  float eCentHit, eTrack, eParticle;
  float CentHitFactor;
  int iEta, iPhi;
  //int iEtaTr, iPhiTr;
  float iDr, delR;
  float e3x3, e5x5;
  float phiTrack, etaTrack;
  float phiGPoint, etaGPoint;

  Bool_t doHF;
  Bool_t AddRecalib;
  int nevtot;

  const HcalRespCorrs* respRecalib;
  const HcalPFCorrs* pfRecalib;

  SteppingHelixPropagator* stepPropF;
  MagneticField *theMagField;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  const CaloGeometry* geo;
  const HcalGeometry* geoHcal;

  Float_t xTrkHcal,yTrkHcal,zTrkHcal;
  Float_t xTrkEcal, yTrkEcal, zTrkEcal;
  Float_t xAtHcal, yAtHcal, zAtHcal;

  float eEcalCone, eHcalCone, eHcalConeNoise;
  int UsedCells, UsedCellsNoise;
  float phiParticle, etaParticle;
  
  Int_t  numValidTrkHits[50], numValidTrkStrips[50], numLayers[50];
  
  Bool_t trkQual[50];

  TProfile *nCells, *nCellsNoise, *enHcal, *enHcalNoise;
  //  TH1F *enEcalB, *enEcalE;
  TTree *pfTree, *tracksTree;
  //  TFile *rootFile;
  edm::Service<TFileService> fs;
  Int_t nTracks;
  Float_t genEta,genPhi, trackEta[50],trackPhi[50], trackP[50] , delRmc[50];

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<reco::TrackCollection> tok_tracks_;
  edm::EDGetTokenT<edm::HepMCProduct> tok_gen_;
};


HcalCorrPFCalculation::HcalCorrPFCalculation(edm::ParameterSet const& iConfig) {

  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheRecHitCollectionTag"));
  tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfRecHitCollectionTag"));
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoRecHitCollectionTag")); 

  // should maybe add these options to configuration - cowden
  tok_EE_ = consumes<EcalRecHitCollection>( edm::InputTag("ecalRecHit","EcalRecHitsEE") );
  tok_EB_ = consumes<EcalRecHitCollection>( edm::InputTag("ecalRecHit","EcalRecHitsEB") );
  tok_tracks_ = consumes<reco::TrackCollection>( edm::InputTag("generalTracks") );
  tok_gen_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared")); 

  //  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
  Respcorr_        = iConfig.getUntrackedParameter<bool>("RespcorrAdd", false);
  PFcorr_        = iConfig.getUntrackedParameter<bool>("PFcorrAdd", false);
  Conecorr_        = iConfig.getUntrackedParameter<bool>("ConeCorrAdd", true);
  //radius_       = iConfig.getUntrackedParameter<double>("ConeRadiusCm", 40.);
  //energyECALmip = iConfig.getParameter<double>("energyECALmip");

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters( parameters, iC );
  trackAssociator_.useDefaultPropagator();

  associationConeSize_=iConfig.getParameter<double>("associationConeSize");
  clusterConeSize_=iConfig.getParameter<double>("clusterConeSize");
  ecalCone_=iConfig.getParameter<double>("ecalCone");
  trackIsolationCone_ = iConfig.getParameter<double>("trackIsolationCone");
  neutralIsolationCone_ = iConfig.getParameter<double>("neutralIsolationCone");

  // AxB_=iConfig.getParameter<std::string>("AxB");

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

HcalCorrPFCalculation::~HcalCorrPFCalculation() {}

void HcalCorrPFCalculation::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  AddRecalib=kFALSE;

  try{

    edm::ESHandle <HcalRespCorrs> recalibCorrs;
    c.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
    respRecalib = recalibCorrs.product();

    edm::ESHandle <HcalPFCorrs> pfCorrs;
    c.get<HcalPFCorrsRcd>().get("recalibrate",pfCorrs);
    pfRecalib = pfCorrs.product();

    AddRecalib = kTRUE;
    // LogMessage("CalibConstants")<<"   OK ";

  }catch(const cms::Exception & e) {
    LogWarning("CalibConstants")<<"   Not Found!! ";
  }

  edm::Handle<HBHERecHitCollection> hbhe;
  ev.getByToken(tok_hbhe_, hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  
  edm::Handle<HFRecHitCollection> hfcoll;
  ev.getByToken(tok_hf_, hfcoll);
  const HFRecHitCollection Hithf = *(hfcoll.product());
    
  edm::Handle<HORecHitCollection> hocoll;
  ev.getByToken(tok_ho_, hocoll);
  const HORecHitCollection Hitho = *(hocoll.product());
  
  edm::Handle<EERecHitCollection> ecalEE;
  ev.getByToken(tok_EE_,ecalEE);
  const EERecHitCollection HitecalEE = *(ecalEE.product());
  
  edm::Handle<EBRecHitCollection> ecalEB;
  ev.getByToken(tok_EB_,ecalEB);
  const EBRecHitCollection HitecalEB = *(ecalEB.product());
  
  // temporary collection of EB+EE recHits
  std::auto_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
  for(EcalRecHitCollection::const_iterator recHit = (*ecalEB).begin(); recHit != (*ecalEB).end(); ++recHit)
    {tmpEcalRecHitCollection->push_back(*recHit);}
  for(EcalRecHitCollection::const_iterator recHit = (*ecalEE).begin(); recHit != (*ecalEE).end(); ++recHit)
    {tmpEcalRecHitCollection->push_back(*recHit);}
  const EcalRecHitCollection Hitecal = *tmpEcalRecHitCollection;


  edm::Handle<reco::TrackCollection> generalTracks;
  ev.getByToken(tok_tracks_, generalTracks);
   
    edm::ESHandle<CaloGeometry> pG;
    c.get<CaloGeometryRecord>().get(pG);
    geo = pG.product();

    /*
    edm::ESHandle<HcalGeometry> hcalG;
    c.get<HcalGeometryRecord>().get(hcalG);
    geoHcal = hcalG.product();
    */

    const CaloSubdetectorGeometry* gHcal = geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    
    parameters_.useEcal = true;
    parameters_.useHcal = true;
    parameters_.useCalo = false;
    parameters_.useMuon = false;
    parameters_.dREcal = 0.5;
    parameters_.dRHcal = 0.6;    

    //parameters_.dREcal = taECALCone_;
    //parameters_.dRHcal = taHCALCone_;    

    
  edm::ESHandle<MagneticField> bField;
  c.get<IdealMagneticFieldRecord>().get(bField);
  stepPropF  = new SteppingHelixPropagator(&*bField,alongMomentum);
  stepPropF->setMaterialMode(false);
  stepPropF->applyRadX0Correction(true);


  //  double eta_bin[42]={0.,.087,.174,.261,.348,.435,.522,.609,.696,.783,
  //.870,.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,
  //2.322,2.500,2.650,2.853,3.000,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191};
  
  // MC info 
  //double phi_MC = -999999.;  // phi of initial particle from HepMC
  //double eta_MC = -999999.;  // eta of initial particle from HepMC
  double mom_MC = 50.;  // P of initial particle from HepMC
  //bool MC = false;
  
  // MC information
   
    
  edm::Handle<edm::HepMCProduct> evtMC;
  //  ev.getByLabel("generatorSmeared",evtMC);
  ev.getByToken(tok_gen_,evtMC);
  if (!evtMC.isValid()) 
    {
      std::cout << "no HepMCProduct found" << std::endl;    
    } 
  else 
    {
      //MC=true;
      //    std::cout << "*** source HepMCProduct found"<< std::endl;
    }  
  
  // MC particle with highest pt is taken as a direction reference  
  double maxPt = -99999.;
  int npart    = 0;
  
  GlobalPoint initpos (0,0,0);
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evtMC->GetEvent()));
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) 
    {
       phiParticle = (*p)->momentum().phi();
       etaParticle = (*p)->momentum().eta();
      double pt  = (*p)->momentum().perp();
      mom_MC = (*p)->momentum().rho();
      if(pt > maxPt) { npart++; maxPt = pt; /*phi_MC = phiParticle; eta_MC = etaParticle;*/ }
      GlobalVector mom ((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
      int charge = -1;
      
      if(abs((*p)->pdg_id())==211) charge = (*p)->pdg_id()/abs((*p)->pdg_id()); // pions only !!!
      else continue;
      

      /*  Propogate the partoicle to Hcal */


      const FreeTrajectoryState *freetrajectorystate_ =
	new FreeTrajectoryState(initpos, mom ,charge , &(*theMagField));
 

      TrackDetMatchInfo info = trackAssociator_.associate(ev, c, *freetrajectorystate_ , parameters_);

      GlobalPoint barrelMC(0,0,0), endcapMC(0,0,0), forwardMC1(0,0,0), forwardMC2(0,0,0);
      
      GlobalPoint gPointHcal(0.,0.,0.);
       
      /*   
      xTrkHcal=info.trkGlobPosAtHcal.x();
      yTrkHcal=info.trkGlobPosAtHcal.y();
      zTrkHcal=info.trkGlobPosAtHcal.z();
      //GlobalPoint gPointHcal(xTrkHcal,yTrkHcal,zTrkHcal);
      
      GlobalVector trackMomAtHcal = info.trkMomAtHcal;
 
      if (xTrkHcal==0 && yTrkHcal==0 && zTrkHcal==0) continue;
      */


      if(fabs(etaParticle)<1.392) {
	Cylinder *cylinder = new Cylinder(181.1, Surface::PositionType(0,0,0),
					  Surface::RotationType());
	
	TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*cylinder));
	if(steppingHelixstateinfo_.isValid() ) 
	  {    barrelMC = steppingHelixstateinfo_.freeState()->position();  }
	
      }
      
      
      doHF = kFALSE;
      if(fabs(etaParticle)>=1.392 && fabs(etaParticle)<5.191) {
	
	Surface::RotationType rot(GlobalVector(1,0,0),GlobalVector(0,1,0));
	
	Surface::PositionType pos(0., 0.,400.5*fabs(etaParticle)/etaParticle);
	PlaneBuilder::ReturnType aPlane = PlaneBuilder().plane(pos,rot);
	
	
	TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane));
	if(steppingHelixstateinfo_.isValid() && fabs(steppingHelixstateinfo_.freeState()->position().eta())<3.0 ) 
	  endcapMC = steppingHelixstateinfo_.freeState()->position();
	if(steppingHelixstateinfo_.isValid() && fabs(steppingHelixstateinfo_.freeState()->position().eta())>3.0 )
	  doHF=kTRUE;
      }
      if(doHF) {
	
	if (abs(etaParticle)>5.191) continue;
	
	if(abs(etaParticle)>2.99) {	
	  Surface::RotationType rot(GlobalVector(1,0,0),GlobalVector(0,1,0));
	  
	  Surface::PositionType pos1(0., 0.,1125*fabs(etaParticle)/etaParticle);
	  //	Surface::PositionType pos1(0., 0.,1115*fabs(etaParticle)/etaParticle);
	  Surface::PositionType pos2(0., 0.,1137*fabs(etaParticle)/etaParticle);
	  PlaneBuilder::ReturnType aPlane1 = PlaneBuilder().plane(pos1,rot);
	  PlaneBuilder::ReturnType aPlane2 = PlaneBuilder().plane(pos2,rot);
	  
	  
	  TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane1));
	  if(steppingHelixstateinfo_.isValid() ) 
	    forwardMC1 = steppingHelixstateinfo_.freeState()->position();
	  
	  steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane2));
	  if(steppingHelixstateinfo_.isValid() ) 
	    forwardMC2 = steppingHelixstateinfo_.freeState()->position();
	}
      }
      /*   ------------       ------------    -----------------------------  */
      
 

      /*  Finding the closest cell at Hcal  */
      Int_t iphitrue = -10;
      Int_t ietatrue = 100;
      HcalDetId tempId, tempId1, tempId2;
      
      
      if (abs(etaParticle)<1.392)
	{
	  gPointHcal = barrelMC;
	  tempId = gHcal->getClosestCell(gPointHcal);
	}
      if (abs(etaParticle)>=1.392 && abs(etaParticle)<3.0)  
	{
	  gPointHcal = endcapMC;
	  tempId = gHcal->getClosestCell(gPointHcal);
	}
      if (abs(etaParticle)>=3.0 && abs(etaParticle)<5.191) 
	{
	  /*
	    tempId1 = gHcal->getClosestCell(forwardMC1);	 
	    tempId2 = gHcal->getClosestCell(forwardMC2);
	    if (deltaR(tempId1.eta(), tempId1.phi(), etaParticle, phiParticle) < deltaR(tempId2.eta(), tempId2.phi(), etaParticle, phiParticle))
	    gPointHcal = forwardMC1;
	    
	    else gPointHcal = forwardMC2;
	  */
	  gPointHcal = forwardMC1;
	  tempId = gHcal->getClosestCell(gPointHcal);
	  //tempId = gHcal->CaloSubdetectorGeometry::getClosestCell(gPointHcal);
	}

      
      
      tempId = gHcal->getClosestCell(gPointHcal);
    
      ietatrue = tempId.ieta();
      iphitrue = tempId.iphi();
      
      etaGPoint = gPointHcal.eta();
      phiGPoint = gPointHcal.phi();

      //xAtHcal = gPointHcal.x();
      //yAtHcal = gPointHcal.y();
      //zAtHcal = gPointHcal.z();
      /*       -----------------   ------------------------      */

      
      if (gPointHcal.x()==0 && gPointHcal.y()==0 && gPointHcal.z()==0)
	{/*cout <<"gPointHcal is Zero!"<<endl;*/ continue;}


      float etahcal=gPointHcal.eta();
      // float phihcal=gPointHcal.phi();
      if (abs(etahcal)>5.192) continue;
      //if (abs(etahcal)>3.0 && abs(etahcal)<5.191)   
	
      //cout <<gPointHcal.x() <<"   "<<gPointHcal.y() <<"   "<<gPointHcal.z()<<"    "<<gPointHcal.eta()<<"  "<<gPointHcal.phi()<<"   "<<ietatrue<<"   "<<iphitrue <<endl;
      
      //      if (ietatrue==100 || iphitrue==-10) {cout<<"ietatrue: "<<ietatrue<<"   iphitrue: "<<iphitrue<<"  etahcal: "<<etahcal<<"  phihcal: "<<phihcal<<endl;}
	

      
      
      /*   -------------   Calculate Ecal Energy using TrackAssociator  ---------------------- */

      //float etaecal=info.trkGlobPosAtEcal.eta();
      
      xTrkEcal=info.trkGlobPosAtEcal.x();
      yTrkEcal=info.trkGlobPosAtEcal.y();
      zTrkEcal=info.trkGlobPosAtEcal.z();
       
      GlobalPoint gPointEcal(xTrkEcal,yTrkEcal,zTrkEcal);
            
      eECAL = ecalEnergyInCone(gPointEcal, ecalCone_, Hitecal, geo);
      eECAL09cm = ecalEnergyInCone(gPointEcal, 9, Hitecal, geo);
      eECAL40cm = ecalEnergyInCone(gPointEcal, 40, Hitecal, geo);
      
      eEcalCone  = eECAL;
      //if(abs(etaecal)<1.5) enEcalB -> Fill(eEcalCone); 
      //if(abs(etaecal)>1.5 && abs(etaecal)<3.1) enEcalE -> Fill(eEcalCone); 

      /*    ------------------------------              --------------------------       ----------------- */


      /*    ----------------- Find the Hottest Hcal RecHit   --------------------------- */
      MaxHit_struct MaxHit;
      
      MaxHit.hitenergy=-100.;
      Float_t recal = 1.0;

      //Hcal:
      eHcalCone = 0.;
      eHcalConeNoise = 0.;
      UsedCells = 0;
      UsedCellsNoise = 0;
      e3x3 = 0.;
      e5x5 = 0.;
      
      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	//for (HcalRecHitCollection::const_iterator hhit=Hithcal.begin(); hhit!=Hithcal.end(); hhit++) 
	{ 
	  recal = RecalibFactor(hhit->detid());
	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;
	  
	  if (depthhit!=1) continue;

	  double distAtHcal =  getDistInPlaneSimple(gPointHcal, pos);
	  
	  if(distAtHcal < clusterConeSize_) 
	    {
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		//for (HcalRecHitCollection::const_iterator hhit2=Hithcal.begin(); hhit2!=Hithcal.end(); hhit2++) 
		{
		  int iphihit2  = (hhit2->id()).iphi();
		  int ietahit2  = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy() * recal;
		  
		  if (iphihit==iphihit2 && ietahit==ietahit2  && depthhit!=depthhit2)  enehit = enehit+enehit2;
		
		}	  	  
	      
	 	      
	      if(enehit > MaxHit.hitenergy) 
		{
 		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.dr   = distAtHcal;
		  //MaxHit.depthhit  = (hhit->id()).depth();
		  MaxHit.depthhit  = 1;
		}
	    }
	}

      
      for (HFRecHitCollection::const_iterator hhit=Hithf.begin(); hhit!=Hithf.end(); hhit++) 
	{
	  
	  recal = RecalibFactor(hhit->detid());
	  
	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;
	  
	  double distAtHcal =  getDistInPlaneSimple(gPointHcal,pos);
	  
	  if(distAtHcal < associationConeSize_) 	  
	    {
	      for (HFRecHitCollection::const_iterator hhit2=Hithf.begin(); hhit2!=Hithf.end(); hhit2++) 
		{
		  int iphihit2  = (hhit2->id()).iphi();
		  int ietahit2  = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy() * recal;
		  
		  if (iphihit==iphihit2 && ietahit==ietahit2  && depthhit!=depthhit2)  enehit = enehit+enehit2;
		  
		}	  	  
	 	      
	      if(enehit > MaxHit.hitenergy) 
		{
		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.dr   = distAtHcal;
		  MaxHit.depthhit  = 1;
		}
	    }
	}
      
      /*    ----------------------       ----------------              --------------------------------------------   */


      /*    ----------- Collect Hcal Energy in a Cone (and also 3x3 and 5x5 around the hottest cell)*/
      
      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	//    for (HcalRecHitCollection::const_iterator hhit=Hithcal.begin(); hhit!=Hithcal.end(); hhit++) 
	{ 
	  
	  recal = RecalibFactor(hhit->detid());
	  //cout<<"recal: "<<recal<<endl;
	  
	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;
	  
	  //if (depthhit!=1) continue;
	  
	  //Set noise RecHit opposite to track hits
	  int  iphihitNoise = iphihit >36 ? iphihit-36 : iphihit+36;
	  int ietahitNoise  = -ietahit;
	  int depthhitNoise = depthhit;
	  
	  double distAtHcal =  getDistInPlaneSimple(gPointHcal, pos);
	  if(distAtHcal < clusterConeSize_ && MaxHit.hitenergy > 0.) 
	    {
	      eHcalCone += enehit;	    
	      UsedCells++;
	      
	      
	      int DIETA = 100;
	      if(MaxHit.ietahitm*(hhit->id()).ieta()>0)
		{ DIETA = MaxHit.ietahitm - (hhit->id()).ieta();}
	      if(MaxHit.ietahitm*(hhit->id()).ieta()<0)
	    { DIETA = MaxHit.ietahitm - (hhit->id()).ieta();  DIETA = DIETA>0 ? DIETA-1 : DIETA+1;}
	      int DIPHI = abs(MaxHit.iphihitm - (hhit->id()).iphi());
	      DIPHI = DIPHI>36 ? 72-DIPHI : DIPHI;
	      
	      
	      if (abs(DIETA)<=2 && (abs(DIPHI)<=2 || ((abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=4) && !( (abs(MaxHit.ietahitm)==21 || abs(MaxHit.ietahitm)==22) && abs((hhit->id()).ieta())<=20 && abs(DIPHI)>2) )) )
		{e5x5 += hhit->energy();}
	      if (abs(DIETA)<=1 && (abs(DIPHI)<=1 || ((abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=2) && !(abs(MaxHit.ietahitm)==21 && abs((hhit->id()).ieta())<=20 && abs(DIPHI)>1) )) )
		{e3x3  += hhit->energy();}
	      
	      // cout<<"track: ieta "<<ietahit<<" iphi: "<<iphihit<<" depth: "<<depthhit<<" energydepos: "<<enehit<<endl;
	      
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  recal = RecalibFactor(hhit2->detid());
		  int iphihit2 = (hhit2->id()).iphi();
		  int ietahit2 = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy()* recal;	
		  
		  if (iphihitNoise == iphihit2 && ietahitNoise == ietahit2 && depthhitNoise == depthhit2 && enehit2>0.)
		    {
		      eHcalConeNoise += hhit2->energy()*recal;
		      UsedCellsNoise++;
		      //cout<<"Noise: ieta "<<ietahit2<<" iphi: "<<iphihit2<<" depth: "<<depthhit2<<" energydepos: "<<enehit2<<endl;
		    }
		}
	    }
	} //end of all HBHE  hits loop
  
      
      for (HFRecHitCollection::const_iterator hhit=Hithf.begin(); hhit!=Hithf.end(); hhit++) 
	{
	  
	  recal = RecalibFactor(hhit->detid());
	  
	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  //float phihit = pos.phi();
	  //float etahit = pos.eta();
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy()* recal;
	  
	  //Set noise RecHit opposite to track hits
	  int  iphihitNoise = iphihit >36 ? iphihit-36 : iphihit+36;
	  int ietahitNoise  = -ietahit;
	  int depthhitNoise = depthhit;
	  
	  double distAtHcal =  getDistInPlaneSimple(gPointHcal,pos);
	  
	  if(distAtHcal < clusterConeSize_ &&  MaxHit.hitenergy > 0.) 
	    //if(dr<radius_ && enehit>0.) 
	    {
	      
	      eHcalCone += enehit;	    
	      UsedCells++;
	      
	      int DIETA = 100;
	      if(MaxHit.ietahitm*(hhit->id()).ieta()>0)
		{ DIETA = MaxHit.ietahitm - (hhit->id()).ieta();}
	      if(MaxHit.ietahitm*(hhit->id()).ieta()<0)
		{ DIETA = MaxHit.ietahitm - (hhit->id()).ieta();  DIETA = DIETA>0 ? DIETA-1 : DIETA+1;}
	      int DIPHI = abs(MaxHit.iphihitm - (hhit->id()).iphi());
	      DIPHI = DIPHI>36 ? 72-DIPHI : DIPHI;
	      
	      
	      if (abs(DIETA)<=2 && (abs(DIPHI)<=2 || ((abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=4) && !( (abs(MaxHit.ietahitm)==21 || abs(MaxHit.ietahitm)==22) && abs((hhit->id()).ieta())<=20 && abs(DIPHI)>2) )) )
		{e5x5 += hhit->energy();}
	      if (abs(DIETA)<=1 && (abs(DIPHI)<=1 || ((abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=2) && !(abs(MaxHit.ietahitm)==21 && abs((hhit->id()).ieta())<=20 && abs(DIPHI)>1) )) )
		{e3x3  += hhit->energy();}
	      
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
	} //end of all HF hits loop

      /*  ----------------      --------------------         ----------------------------------------------       -------- */
      

      /* ------------- -   Track-MC matching  (if any tracks are in event)    ------------    - */
      
      nTracks=0;

      delRmc[0] = 5; 
      
      float delR_track_particle = 100;
      
      for (reco::TrackCollection::const_iterator track1=generalTracks->begin(); track1!=generalTracks->end(); track1++)
	{
	  delR_track_particle = deltaR(etaParticle, phiParticle, track1->eta(), track1->phi());
	  
	  trackEta[nTracks] = track1 -> eta();
	  trackPhi[nTracks] = track1 -> phi();
	  trackP[nTracks]   = sqrt(track1->px()*track1->px() + track1->py()*track1->py() + track1->pz()*track1->pz());
	  
	  delRmc[nTracks]            = delR_track_particle;
	  numValidTrkHits[nTracks]   = track1->hitPattern().numberOfValidHits();
	  numValidTrkStrips[nTracks] = track1->hitPattern().numberOfValidStripTECHits();
	  numLayers[nTracks]         = track1->hitPattern().trackerLayersWithMeasurement(); //layers crossed
	  trkQual[nTracks]           = track1->quality(reco::TrackBase::highPurity);
	  
	  nTracks++;
	}
	  

      /*        ------------------          ------------------------------ ------- */


      int dieta_M_P = 100;
      int diphi_M_P = 100;
      if(MaxHit.ietahitm*ietatrue>0) {dieta_M_P = abs (MaxHit.ietahitm-ietatrue);}
      if(MaxHit.ietahitm*ietatrue<0) {dieta_M_P = abs(MaxHit.ietahitm-ietatrue)-1;}
      diphi_M_P = abs(MaxHit.iphihitm-iphitrue);

      diphi_M_P =  diphi_M_P>36 ? 72-diphi_M_P : diphi_M_P; 
      iDr = sqrt(diphi_M_P*diphi_M_P+dieta_M_P*dieta_M_P);

      /*      if (iDr>15) 
	{
cout<<"diphi: "<<diphi_M_P<<"  dieta: "<<dieta_M_P<<"   iDr: "<<iDr<<" ietatrue:"<<ietatrue<<"  iphitrue:"<<iphitrue<<endl;
cout<<"M ieta: "<<MaxHit.ietahitm<<"  M iphi: "<<MaxHit.iphihitm<<endl;
	
}*/


      Bool_t passCuts = kFALSE;
      passCuts=kTRUE; 
      //if(eEcalCone < energyECALmip && iDr<2.) passCuts = kTRUE;
      //if(MaxHit.hitenergy>0.) passCuts = kTRUE;
      

      if(passCuts)
	{
	  /*	  
	  enHcal -> Fill(ietatrue,  eHcalCone);
	  nCells -> Fill(ietatrue,  UsedCells);
	  enHcalNoise -> Fill(ietatrue,  eHcalConeNoise);
	  nCellsNoise -> Fill(ietatrue,  UsedCellsNoise); 
	  */


	  //e3x3=0; e5x5=0;

	  iEta = ietatrue;
	  iPhi = iphitrue;

	   //iEta = MaxHit.ietahitm;
	   //iPhi = MaxHit.iphihitm;
	  delR = MaxHit.dr;
	  eCentHit = MaxHit.hitenergy;

 	  eParticle = mom_MC;
 	  //eTrack = mom_MC;
	  //phiTrack = phiParticle;
	  //etaTrack = etaParticle;

	  pfTree->Fill();
	}

    } //Hep:MC


}

void HcalCorrPFCalculation::beginJob(){

  pfTree = fs -> make<TTree>("pfTree", "Tree for pf info");

  
  pfTree->Branch("nTracks", &nTracks, "nTracks/I"); 
  pfTree->Branch("trackEta", trackEta, "trackEta[nTracks]/F"); 
  pfTree->Branch("trackPhi", trackPhi, "trackPhi[nTracks]/F"); 
  pfTree->Branch("trackP", trackP, "trackP[nTracks]/F"); 
  
  pfTree->Branch("delRmc", delRmc, "delRmc[nTracks]/F"); 
  pfTree->Branch("numValidTrkHits", numValidTrkHits, "numValidTrkHits[nTracks]/I"); 
  pfTree->Branch("numValidTrkStrips", numValidTrkStrips, "numValidTrkStrips[nTracks]/I"); 
  pfTree->Branch("numLayers", numLayers, "numLayers[nTracks]/I"); 
  pfTree->Branch("trkQual", trkQual, "trkQual[nTracks]/O"); 
  
  
  pfTree->Branch("eEcalCone", &eEcalCone, "eEcalCone/F");
  pfTree->Branch("eHcalCone", &eHcalCone, "eHcalCone/F");
  pfTree->Branch("eHcalConeNoise", &eHcalConeNoise, "eHcalConeNoise/F");
  
  pfTree->Branch("UsedCellsNoise", &UsedCellsNoise, "UsedCellsNoise/I");
  pfTree->Branch("UsedCells", &UsedCells, "UsedCells/I");
  
  pfTree->Branch("eCentHit", &eCentHit , "eCentHit/F");

  pfTree->Branch("eParticle", &eParticle, "eParticle/F");
  pfTree->Branch("etaParticle", &etaParticle, "etaParticle/F");
  pfTree->Branch("phiParticle", &phiParticle, "phiParticle/F");

  pfTree->Branch("etaGPoint", &etaGPoint, "etaGPoint/F");
  pfTree->Branch("phiGPoint", &phiGPoint, "phiGPoint/F");

  pfTree->Branch("xAtHcal", &xAtHcal, "xAtHcal/F");
  pfTree->Branch("yAtHcal", &yAtHcal, "yAtHcal/F");
  pfTree->Branch("zAtHcal", &zAtHcal, "zAtHcal/F");
  
  pfTree->Branch("eECAL09cm", &eECAL09cm, "eECAL09cm/F");
  pfTree->Branch("eECAL40cm", &eECAL40cm, "eECAL40cm/F");
  pfTree->Branch("eECAL", &eECAL, "eECAL/F");

  pfTree->Branch("e3x3 ", &e3x3 , "e3x3/F");
  pfTree->Branch("e5x5", &e5x5 , "e5x5/F");

  pfTree->Branch("iDr", &iDr, "iDr/F");
  pfTree->Branch("delR", &delR, "delR/F");

 pfTree->Branch("iEta", &iEta, "iEta/I");
 pfTree->Branch("iPhi", &iPhi, "iPhi/I");
 
 //  pfTree->Branch("numValidTrkHits", &numValidTrkHits, "numValidTrkHits/I");
 // pfTree->Branch("numValidTrkStrips", &numValidTrkStrips, "numValidTrkStrips/I");
 // pfTree->Branch("trkQual", &trkQual, "trkQual/");
 // pfTree->Branch("numLayers", &numLayers, "numLayers/I");

}


void HcalCorrPFCalculation::endJob() 
{}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


//define this as a plug-in
DEFINE_FWK_MODULE(HcalCorrPFCalculation);
