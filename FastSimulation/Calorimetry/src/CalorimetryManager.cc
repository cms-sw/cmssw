//Framework headers 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Fast Simulation headers
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDRShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HSParameters.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
//#include "FastSimulation/Utilities/interface/Histos.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/GammaFunctionGenerator.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"  
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

// STL headers 
#include <vector>
#include <iostream>

//CMSSW headers 
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

typedef math::XYZVector XYZVector;
typedef math::XYZVector XYZPoint;

std::vector<std::pair<int,float> > CalorimetryManager::myZero_ = std::vector<std::pair<int,float> >
(1,std::pair<int,float>(0,0.));

CalorimetryManager::CalorimetryManager() : 
  myCalorimeter_(0),
  myHistos(0),
  random(0),initialized_(false)
{;}

CalorimetryManager::CalorimetryManager(FSimEvent * aSimEvent, 
				       const edm::ParameterSet& fastCalo,
				       const RandomEngine* engine)
  : 
  mySimEvent(aSimEvent), 
  random(engine),initialized_(false)

{

  aLandauGenerator = new LandauFluctuationGenerator(random);
  aGammaGenerator = new GammaFunctionGenerator(random);

  readParameters(fastCalo);

//  EBMapping_.resize(62000,myZero_);
//  EEMapping_.resize(20000,myZero_);
//  HMapping_.resize(10000,myZero_);
  EBMapping_.resize(62000);
  EEMapping_.resize(20000);
  HMapping_.resize(10000);
  theDetIds_.resize(10000);

  unsigned s=(unfoldedMode_)?5:1;
  for(unsigned ic=0;ic<62000;++ic)
    {
      EBMapping_[ic].reserve(s);
      if(ic<20000)
	EEMapping_[ic].reserve(s);
      if(ic<10000)
	HMapping_[ic].reserve(s);
    }
  


  myHistos = 0; 
#ifdef FAMOSDEBUG
  myHistos = Histos::instance();
  myHistos->book("h10",140,-3.5,3.5,100,-0.5,99.5);
  myHistos->book("h20",150,0,150.,100,-0.5,99.5);
  myHistos->book("h100",140,-3.5,3.5,100,0,0.1);
  myHistos->book("h110",140,-3.5,3.5,100,0,10.);
  myHistos->book("h120",200,-5.,5.,100,0,0.5);

  myHistos->book("h200",300,0,3.,100,0.,35.);
  myHistos->book("h210",720,-M_PI,M_PI,100,0,35.);
  myHistos->book("h212",720,-M_PI,M_PI,100,0,35.);

  myHistos->bookByNumber("h30",0,7,300,-3.,3.,100,0.,35.);
  myHistos->book("h310",75,-3.,3.,"");
  myHistos->book("h400",100,-10.,10.,100,0.,35.);
  myHistos->book("h410",720,-M_PI,M_PI);
#endif
  myCalorimeter_ = 
    new CaloGeometryHelper(fastCalo);
  myHDResponse_ = 
    new HCALResponse(fastCalo.getParameter<edm::ParameterSet>("HCALResponse"),
		     random);
  myHSParameters_ = 
    new HSParameters(fastCalo.getParameter<edm::ParameterSet>("HSParameters"));
}

void CalorimetryManager::clean()
{
  unsigned size=firedCellsEB_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      EBMapping_[firedCellsEB_[ic]].clear();
    }
  firedCellsEB_.clear();

  size=firedCellsEE_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      EEMapping_[firedCellsEE_[ic]].clear();
    }
  firedCellsEE_.clear();
  
  size=firedCellsHCAL_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      HMapping_[firedCellsHCAL_[ic]].clear();
    }
  firedCellsHCAL_.clear();

  ESMapping_.clear();

}

CalorimetryManager::~CalorimetryManager()
{
#ifdef FAMOSDEBUG
  myHistos->put("Famos.root");
#endif
  if(myCalorimeter_) delete myCalorimeter_;
  if(myHDResponse_) delete myHDResponse_;
}

void CalorimetryManager::reconstruct()
{
  // Clear the content of the calorimeters 
  if(!initialized_)
    {
      const CaloSubdetectorGeometry* geom=myCalorimeter_->getHcalGeometry();
      for(int subdetn=1;subdetn<=4;++subdetn)
	{
	  std::vector<DetId> ids=geom->getValidDetIds(DetId::Hcal,subdetn);  
	  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) 
	    {
	      HcalDetId myDetId(*i);
	      unsigned hi=myDetId.hashed_index();
	      theDetIds_[hi]=myDetId;
	    }
	}
      initialized_=true;
    }
  clean();

  LogInfo("FastCalorimetry") << "Reconstructing " << (int) mySimEvent->nTracks() << " tracks." << std::endl;
  for( int fsimi=0; fsimi < (int) mySimEvent->nTracks() ; ++fsimi) {

    FSimTrack& myTrack = mySimEvent->track(fsimi);

    int pid = abs(myTrack.type());

    if (debug_) {
      LogDebug("FastCalorimetry") << " ===> pid = "  << pid << std::endl;      
    }
    
    
    // Check that the particle hasn't decayed
    if(myTrack.noEndVertex()) {
      // Simulate energy smearing for photon and electrons
      if ( pid == 11 || pid == 22 ) {
	  
	  
	   if ( myTrack.onEcal() ) 
	    EMShowerSimulation(myTrack);
	  else if ( myTrack.onVFcal() )
	    reconstructHCAL(myTrack);
	   
      } // electron or photon
      // Simulate energy smearing for hadrons (i.e., everything 
      // but muons... and SUSY particles that deserve a special 
      // treatment.
      else if ( pid < 1000000 ) {
	if ( myTrack.onHcal() || myTrack.onVFcal() ) 	  
	  if(optionHDSim_ == 0 || pid == 13)  reconstructHCAL(myTrack);
	  else HDShowerSimulation(myTrack);
	    
      } // pid < 1000000 
    } // myTrack.noEndVertex()
  } // particle loop
  LogInfo("FastCalorimetry") << " Number of  hits (barrel)" << EBMapping_.size() << std::endl;
  LogInfo("FastCalorimetry") << " Number of  hits (Hcal)" << HMapping_.size() << std::endl;
  //  std::cout << " Nombre de hit (endcap)" << EEMapping_.size() << std::endl;

} // reconstruct

// Simulation of electromagnetic showers in PS, ECAL, HCAL 
void CalorimetryManager::EMShowerSimulation(const FSimTrack& myTrack) {
  std::vector<const RawParticle*> thePart;
  double X0depth;
  
  //  std::cout << " Simulating " << myTrack << std::endl;

  // The Particle at ECAL entrance
  //  std::cout << " Before ecalEntrance " << std::endl;
  myPart = myTrack.ecalEntrance(); 

  // protection against infinite loop.  
  if ( myTrack.type() == 22 && myPart.e()<0.055) return; 


  // Barrel or Endcap ?
  int onEcal = myTrack.onEcal();
  int onHcal = myTrack.onHcal();
  int onLayer1 = myTrack.onLayer1();
  int onLayer2 = myTrack.onLayer2();

  // The entrance in ECAL
  XYZPoint ecalentrance = myPart.vertex().Vect();
  
  //  std::cout << " Ecal entrance " << ecalentrance << std::endl;
  
  // The preshower
  PreshowerHitMaker * myPreshower = NULL ;
  if(onLayer1 || onLayer2)
    {
      XYZPoint layer1entrance,layer2entrance;
      XYZVector dir1,dir2;
      if(onLayer1) 
	{
	  layer1entrance = XYZPoint(myTrack.layer1Entrance().vertex().Vect());
	  dir1 = XYZVector(myTrack.layer1Entrance().Vect().Unit());
	}
      if(onLayer2) 
	{
	  layer2entrance = XYZPoint(myTrack.layer2Entrance().vertex().Vect());
	  dir2 = XYZVector(myTrack.layer2Entrance().Vect().Unit());
	}
      //      std::cout << " Layer1entrance " << layer1entrance << std::endl;
      //      std::cout << " Layer2entrance " << layer2entrance << std::endl;
      myPreshower = new PreshowerHitMaker(myCalorimeter_,
					  layer1entrance,
					  dir1,
					  layer2entrance,
					  dir2,
					  aLandauGenerator);
    }

  // The ECAL Properties
  EMECALShowerParametrization 
    showerparam(myCalorimeter_->ecalProperties(onEcal), 
		myCalorimeter_->hcalProperties(onHcal), 
		myCalorimeter_->layer1Properties(onLayer1), 
		myCalorimeter_->layer2Properties(onLayer2),
		theCoreIntervals_,
		theTailIntervals_,
		RCFactor_,
		RTFactor_);

  // Photons : create an e+e- pair
  if ( myTrack.type() == 22 ) {
    
    // Depth for the first e+e- pair creation (in X0)
    X0depth = -log(random->flatShoot()) * (9./7.);
    
    // Initialization
    double eMass = 0.000510998902; 
    double xe=0;
    double xm=eMass/myPart.e();
    double weight = 0.;
    
    // Generate electron energy between emass and eGamma-emass
    do {
      xe = random->flatShoot()*(1.-2.*xm) + xm;
      weight = 1. - 4./3.*xe*(1.-xe);
    } while ( weight < random->flatShoot() );
    
    // Protection agains infinite loop in Famos Shower
    if ( myPart.e()*xe < 0.055 || myPart.e()*(1.-xe) < 0.055 ) {
      
      if ( myPart.e() > 0.055 ) thePart.push_back(&myPart);
      
    } else {      
      
      myElec = (myPart) * xe;
      myPosi = (myPart) * (1.-xe);
      myElec.setVertex(myPart.vertex());
      myPosi.setVertex(myPart.vertex());
      thePart.push_back(&myElec);
      thePart.push_back(&myPosi);
    }
  // Electrons
  } else {  
    
    X0depth = 0.;
    if ( myPart.e() > 0.055 ) thePart.push_back(&myPart);
    
  } 
  
  // After the different protections, this shouldn't happen. 
  if(thePart.size()==0) 
    { 
      if(myPreshower==NULL) return; 
      delete myPreshower; 
      return; 
    } 

  // find the most energetic particle
  double maxEnergy=-1.;
  for(unsigned ip=0;ip < thePart.size();++ip)
    if(thePart[ip]->e() > maxEnergy) maxEnergy = thePart[ip]->e();
  
  // Initialize the Grid in ECAL
  int size = gridSize_;
//  if ( maxEnergy < threshold5x5 ) size = 5;
//  if ( maxEnergy < threshold3x3 ) size = 3;


  EMShower theShower(random,aGammaGenerator,&showerparam,&thePart);


  double depth((X0depth+theShower.getMaximumOfShower())*myCalorimeter_->ecalProperties(onEcal)->radLenIncm());
  XYZPoint meanShower=ecalentrance+myPart.Vect().Unit()*depth;
  
  //  if(onEcal!=1) return ; 

  // The closest crystal
  DetId pivot(myCalorimeter_->getClosestCell(meanShower, true, onEcal==1));

  //  std::cout << " After getClosestCell " << std::endl;
  
  EcalHitMaker myGrid(myCalorimeter_,ecalentrance,pivot,onEcal,size,0,random);
  //                                             ^^^^
  //                                         for EM showers
  myGrid.setPulledPadSurvivalProbability(pulledPadSurvivalProbability_);
  myGrid.setCrackPadSurvivalProbability(crackPadSurvivalProbability_);
  myGrid.setRadiusFactor(radiusFactor_);
  
  // The shower simulation
  myGrid.setTrackParameters(myPart.Vect().Unit(),X0depth,myTrack);

//  std::cout << " PS ECAL GAP HCAL X0 " << myGrid.ps1TotalX0()+myGrid.ps2TotalX0() << " " << myGrid.ecalTotalX0();
//  std::cout << " " << myGrid.ecalHcalGapTotalX0() << " " << myGrid.hcalTotalX0() << std::endl;
//  std::cout << " PS ECAL GAP HCAL L0 " << myGrid.ps1TotalL0()+myGrid.ps2TotalL0() << " " << myGrid.ecalTotalL0();
//   std::cout << " " << myGrid.ecalHcalGapTotalL0() << " " << myGrid.hcalTotalL0() << std::endl;
//   std::cout << "ECAL-HCAL " << myTrack.momentum().eta() << " " <<  myGrid.ecalHcalGapTotalL0() << std::endl;
//
//  std::cout << " Grid created " << std::endl;
  if(myPreshower) theShower.setPreshower(myPreshower);
  
  HcalHitMaker myHcalHitMaker(myGrid,(unsigned)0); 

  theShower.setGrid(&myGrid);
  theShower.setHcal(&myHcalHitMaker);
  //  std:: cout << " About to compute " << std::endl;
  theShower.compute();
  //  std::cout << " Coming back from compute" << std::endl;
  //myHistos->fill("h502", myPart->eta(),myGrid.totalX0());
  
  // Save the hits !
  std::map<uint32_t,float>::const_iterator mapitr;
  std::map<uint32_t,float>::const_iterator endmapitr=myGrid.getHits().end();
  for(mapitr=myGrid.getHits().begin();mapitr!=endmapitr;++mapitr)
    {
      if(onEcal==1)
	{
	  updateMap(EBDetId(mapitr->first).hashedIndex(), mapitr->second,myTrack.id(),EBMapping_,firedCellsEB_);
	}
	    
      else if(onEcal==2)
	updateMap(EEDetId(mapitr->first).hashedIndex(), mapitr->second,myTrack.id(),EEMapping_,firedCellsEE_);
      //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
    }

  // Now fill the HCAL hits
  endmapitr=myHcalHitMaker.getHits().end();
  for(mapitr=myHcalHitMaker.getHits().begin();mapitr!=endmapitr;++mapitr)
    {
      updateMap(HcalDetId(mapitr->first).hashed_index(),mapitr->second,myTrack.id(),HMapping_,firedCellsHCAL_);
      //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
    }

  // delete the preshower
  if(myPreshower!=0)
    {
      endmapitr=myPreshower->getHits().end();
      for(mapitr=myPreshower->getHits().begin();mapitr!=endmapitr;++mapitr)
	{
	  updateMap(mapitr->first,mapitr->second,myTrack.id(),ESMapping_);
	  //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
	}
      delete myPreshower;
    }
  //  std::cout << " Deleting myPreshower " << std::endl;
  
}



// Simulation of electromagnetic showers in VFCAL
void CalorimetryManager::reconstructECAL(const FSimTrack& track) {
  if(debug_) {
    XYZTLorentzVector moment = track.momentum();
    std::cout << "FASTEnergyReconstructor::reconstructECAL - " << std::endl
	 << "  eta " << moment.eta() << std::endl
         << "  phi " << moment.phi() << std::endl
         << "   et " << moment.Et()  << std::endl;
  }
  
  int hit; 
  
  bool central=track.onEcal()==1;
  
  //Reconstruct only electrons and photons. 

  //deal with different conventions
  // ParticlePropagator 1 <-> Barrel
  //                    2 <-> EC
  // whereas for Artur(this code):
  //                    0 <-> Barrel
  //                    1 <-> EC
  //                    2 <-> VF
  XYZTLorentzVector trackPosition;
  if( track.onEcal() ) {
    hit=track.onEcal()-1;
    trackPosition=track.ecalEntrance().vertex();
  } else {
    hit=2;
    trackPosition=track.vfcalEntrance().vertex();
  }
  
  double pathEta   = trackPosition.eta();
  double pathPhi   = trackPosition.phi();	
  double EGen      = track.ecalEntrance().e();
  

  double e=0.;
  double sigma=0.;
  // if full simulation and in HF, but without showering anyway...
  if(hit == 2 && optionHDSim_ == 2 ) { 
    std::pair<double,double> response =
      myHDResponse_->responseHCAL(EGen, pathEta, 0);//0=e/gamma 
    e     = response.first;
    sigma = response.second;
  }

  double emeas = 0.;
  
  if(sigma>0.)
    emeas = random->gaussShoot(e,sigma);
  

  if(debug_)
    std::cout << "FASTEnergyReconstructor::reconstructECAL : " 
         << "  on-calo  eta, phi  = " << pathEta << " " << pathPhi << std::endl 
	 << "  Egen  = " << EGen << std::endl 
	 << "  Eres  = " << e << std::endl 
	 << " sigma  = " << sigma << std::endl 
	 << "  Emeas = " << emeas << std::endl; 


  if(debug_)
    std::cout << "FASTEnergyReconstructor::reconstructECAL : " 
	 << " Track position - " << trackPosition.Vect() 
	 << "   bool central - " << central
         << "   hit - " << hit   << std::endl;  

  DetId detid;  
  if( hit==2 ) 
      detid = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,central);
  // Check that the detid is HCAL forward
  HcalDetId hdetid(detid);
  if(!hdetid.subdetId()!=HcalForward) return;

  if(debug_)
    std::cout << "FASTEnergyReconstructor::reconstructECAL : " 
	      << " CellID - " <<  detid.rawId() << std::endl;

  if( hit != 2  || emeas > 0.) 
    if(!detid.null()) 
      {
	updateMap(hdetid.hashed_index(),emeas,track.id(),HMapping_,firedCellsHCAL_);
      }

}


void CalorimetryManager::reconstructHCAL(const FSimTrack& myTrack)
{
  int hit;
  int pid = abs(myTrack.type());
  //  FSimTrack myTrack = mySimEvent.track(fsimi);

  //  int pid=abs(myTrack.type());
  //  std::cout << "reconstructHCAL " << std::endl;
  
  XYZTLorentzVector trackPosition;
  if (myTrack.onHcal()) {
    trackPosition=myTrack.hcalEntrance().vertex();
    hit = myTrack.onHcal()-1;
  } else {
    trackPosition=myTrack.vfcalEntrance().vertex();
    hit = 2;
  }

  double pathEta   = trackPosition.eta();
  double pathPhi   = trackPosition.phi();	
  //  double pathTheta = trackPosition.theta();

  double EGen  = myTrack.hcalEntrance().e();
  double e     = 0.;
  double sigma = 0.;
  double emeas = 0.;

  if(pid == 13) { 
    std::pair<double,double> response =
      myHDResponse_->responseHCAL(EGen, pathEta, 2); // 2=muon 
    emeas  = response.first;
    if(debug_)
      LogDebug("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - MUON !!!" << std::endl;
  }
  else if( pid == 22 || pid == 11)
    {
      
      std::pair<double,double> response =
	myHDResponse_->responseHCAL(EGen,pathEta,0); // 0=e/gamma
      e     = response.first;              //
      sigma = response.second;             //
      emeas = random->gaussShoot(e,sigma); //

      //  cout <<  "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
      if(debug_)
	LogDebug("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
    }
    else {
      e     = myHDResponse_->getHCALEnergyResponse(EGen,hit);
      sigma = myHDResponse_->getHCALEnergyResolution(EGen, hit);
      
      emeas = random->gaussShoot(e,sigma);  
    }
    

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - on-calo "   
				<< "  eta = " << pathEta 
				<< "  phi = " << pathPhi 
				<< "  Egen = " << EGen 
				<< "  Eres = " << e 
				<< "  sigma = " << sigma 
				<< "  Emeas = " << emeas << std::endl;

  if(emeas > 0.) {  
    DetId cell = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,false);
    updateMap(HcalDetId(cell).hashed_index(), emeas, myTrack.id(),HMapping_,firedCellsHCAL_);
  }
}

void CalorimetryManager::HDShowerSimulation(const FSimTrack& myTrack)
{
  //  TimeMe t(" FASTEnergyReconstructor::HDShower");
  XYZTLorentzVector moment = myTrack.momentum();

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::HDShowerSimulation - track param."
         << std::endl
	 << "  eta = " << moment.eta() << std::endl
         << "  phi = " << moment.phi() << std::endl
         << "   et = " << moment.Et()  << std::endl;

  int hit;
  //  int pid = abs(myTrack.type());

  XYZTLorentzVector trackPosition;
  if ( myTrack.onEcal() ) {
    trackPosition=myTrack.ecalEntrance().vertex();
    hit = myTrack.onEcal()-1;                               //
    myPart = myTrack.ecalEntrance();
  } else if ( myTrack.onVFcal()) {
    trackPosition=myTrack.vfcalEntrance().vertex();
    hit = 2;
    myPart = myTrack.vfcalEntrance();
  }
  else
    {
      LogDebug("FastCalorimetry") << " The particle is not in the acceptance " << std::endl;
      return;
    }

  // int onHCAL = hit + 1; - specially for myCalorimeter->hcalProperties(onHCAL)
  // (below) to get VFcal properties ...
  int onHCAL = hit + 1;
  int onECAL = myTrack.onEcal();
  
  double pathEta   = trackPosition.eta();
  double pathPhi   = trackPosition.phi();	
  //  double pathTheta = trackPosition.theta();

  double eGen  = myTrack.hcalEntrance().e();
  double e     = 0.;
  double sigma = 0.;
  
  // Here to switch between simple formulae and parameterized response 
  if(optionHDSim_ == 1) {
    e     = myHDResponse_->getHCALEnergyResponse  (eGen, hit);
    sigma = myHDResponse_->getHCALEnergyResolution(eGen, hit);
  }
  else { // optionHDsim == 2
    std::pair<double,double> response =
      myHDResponse_->responseHCAL(eGen, pathEta, 1); // 1=hadron 
    e     = response.first;
    sigma = response.second;
  }

  double emeas = 0.;
  emeas = random->gaussShoot(e,sigma);

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::HDShowerSimulation - on-calo " << std::endl  
         << "   eta  = " << pathEta << std::endl
         << "   phi  = " << pathPhi << std::endl 
	 << "  Egen  = " << eGen << std::endl
	 << "  Eres  = " << e << std::endl
	 << " sigma  = " << sigma << std::endl
	 << "  Emeas = " << emeas << std::endl;
  
  //===========================================================================
  if(emeas > 0.) {  

    // Special case - temporary protection until fix in Grid will be done
//    if(fabs(pathEta) > 1.44 && fabs(pathEta) < 1.45) {
//      CellID cell = myCalorimeter->getClosestCell(trackPosition.Vect(),false);
//      updateMap(cell,emeas);
//
//    } 
//    else {  //  Beginning of special "else"
      //=====================================

    
    // ECAL and HCAL properties to get
    HDShowerParametrization 
      theHDShowerparam(myCalorimeter_->ecalProperties(onECAL),
		       myCalorimeter_->hcalProperties(onHCAL),
		       myHSParameters_);
    
    //Making ECAL Grid (and segments calculation)
    XYZPoint caloentrance;
    XYZVector direction;
    if(myTrack.onEcal()) 
      {	
	caloentrance = myTrack.ecalEntrance().vertex().Vect();
	direction = myTrack.ecalEntrance().Vect().Unit();
      }
    else if(myTrack.onHcal())
      {
	caloentrance = myTrack.hcalEntrance().vertex().Vect();
	direction = myTrack.hcalEntrance().Vect().Unit();
      }
    else
      {
	caloentrance = myTrack.vfcalEntrance().vertex().Vect();
	direction = myTrack.vfcalEntrance().Vect().Unit();
      }

    DetId pivot;
    if(myTrack.onEcal())
      {
	pivot=myCalorimeter_->getClosestCell(caloentrance,
					     true, myTrack.onEcal()==1);
      }
    else if(myTrack.onHcal())
      {
	//	std::cout << " CalorimetryManager onHcal " <<  myTrack.onHcal() << " caloentrance" << caloentrance  << std::endl;
	pivot=myCalorimeter_->getClosestCell(caloentrance,					     
					    false, false);
      }
//    if(pivot.isZero())
//      {
//	std::cout << " HDShowerSim pivot is Zero "  << std::endl;
//	std::cout <<" Ecal entrance " << caloentrance << std::endl;
//      }
    EcalHitMaker myGrid(myCalorimeter_,caloentrance,pivot,
			pivot.null()? 0 : myTrack.onEcal(),hdGridSize_,1,
			random);
    // 1=HAD shower

    myGrid.setTrackParameters(direction,0,myTrack);
    // Build the FAMOS HCAL 
    HcalHitMaker myHcalHitMaker(myGrid,(unsigned)1); 
    
    // Shower simulation
    bool status;
    if(hdSimMethod_ == 0) {
      HDShower theShower(random,
			 &theHDShowerparam,
			 &myGrid,
			 &myHcalHitMaker,
			 onECAL,
			 emeas);
      status = theShower.compute();
    }
    else {
      HDRShower theShower(random,
			  &theHDShowerparam,
			  &myGrid,
			  &myHcalHitMaker,
			  onECAL,
			  emeas);
      status = theShower.computeShower();
    }

    if(status) {
      
      // was map<unsigned,double> but CaloHitMaker uses float
      std::map<unsigned,float>::const_iterator mapitr;
      std::map<unsigned,float>::const_iterator endmapitr;
      if(myTrack.onEcal() > 0) {
	// Save ECAL hits 
	endmapitr=myGrid.getHits().end();
	for(mapitr=myGrid.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	  double energy = mapitr->second;
	  if(energy > 0.000001) { 
	    //////////////////////////////////////////////////////////////////////////////
	    // EBRYMapping=EBMapping_? EFRYMapping=EEMapping_????
	    if(onECAL==1)
		updateMap(EBDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EBMapping_,firedCellsEB_);

	    else if(onECAL==2)
	      updateMap(EEDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EEMapping_,firedCellsEE_);

	    if(debug_)
	      LogDebug("FastCalorimetry") << " ECAL cell " << mapitr->first << " added,  E = " 
		   << energy << std::endl;  
	  }
	}
      }
      
      // Save HCAL hits
      endmapitr=myHcalHitMaker.getHits().end();
      for(mapitr=myHcalHitMaker.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	updateMap(HcalDetId(mapitr->first).hashed_index(),mapitr->second,myTrack.id(),HMapping_,firedCellsHCAL_);
	if(debug_)
	  LogDebug("FastCalorimetry") << " HCAL cell "  
	       << mapitr->first << " added    E = " 
	       << mapitr->second << std::endl;  
      }
    }      
    else {  // shower simulation failed  
//      std::cout << " Shower simulation failed " << trackPosition.Vect() << std::endl;
//      std::cout << " The FSimTrack " << myTrack << std::endl;
//      std::cout << " HF entrance on VFcal" << myTrack.onVFcal() << std::endl;
//      std::cout << " trackPosition.eta() " << trackPosition.eta() << std::endl;
      if(myTrack.onHcal() || myTrack.onVFcal())
	{
	  DetId cell = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,false);
	  updateMap(HcalDetId(cell).hashed_index(),emeas,myTrack.id(),HMapping_,firedCellsHCAL_);
	  if(debug_)
	    LogDebug("FastCalorimetry") << " HCAL simple cell "   
					<< cell.rawId() << " added    E = " 
					<< emeas << std::endl;  
	}
    }

    //    } // End of special "else" 
    //===========================

  } // e > 0. ...

  if(debug_)
    LogDebug("FastCalorimetry") << std::endl << " FASTEnergyReconstructor::HDShowerSimulation  finished "
	 << std::endl;
}

void CalorimetryManager::readParameters(const edm::ParameterSet& fastCalo) {
  edm::ParameterSet ECALparameters = fastCalo.getParameter<edm::ParameterSet>("ECAL");
  gridSize_ = ECALparameters.getParameter<int>("GridSize");
  spotFraction_ = ECALparameters.getParameter<double>("SpotFraction");
  pulledPadSurvivalProbability_ = ECALparameters.getParameter<double>("FrontLeakageProbability");
  crackPadSurvivalProbability_ = ECALparameters.getParameter<double>("GapLossProbability");
  debug_ = ECALparameters.getUntrackedParameter<bool>("Debug");
  
  theCoreIntervals_ = ECALparameters.getParameter<std::vector<double> >("CoreIntervals");
  theTailIntervals_ = ECALparameters.getParameter<std::vector<double> >("TailIntervals");
  
  RCFactor_ = ECALparameters.getParameter<double>("RCFactor");
  RTFactor_ = ECALparameters.getParameter<double>("RTFactor");
  radiusFactor_ = ECALparameters.getParameter<double>("RadiusFactor");
  
  if(gridSize_ <1) gridSize_= 7;
  if(pulledPadSurvivalProbability_ <0. || pulledPadSurvivalProbability_>1 ) pulledPadSurvivalProbability_= 1.;
  if(crackPadSurvivalProbability_ <0. || crackPadSurvivalProbability_>1 ) crackPadSurvivalProbability_= 0.9;
  
  LogInfo("FastCalorimetry") << " Fast ECAL simulation parameters " << std::endl;
  LogInfo("FastCalorimetry") << " =============================== " << std::endl;
  LogInfo("FastCalorimetry") << " Grid Size : " << gridSize_  << std::endl; 
  if(spotFraction_>0.) 
    LogInfo("FastCalorimetry") << " Spot Fraction : " << spotFraction_ << std::endl;
  else
    {
      LogInfo("FastCalorimetry") << " Core of the shower " << std::endl;
      for(unsigned ir=0; ir < theCoreIntervals_.size()/2;++ir)
	{
	  LogInfo("FastCalorimetry") << " r < " << theCoreIntervals_[ir*2] << " R_M : " << theCoreIntervals_[ir*2+1] << "        ";
	}
      LogInfo("FastCalorimetry") << std::endl;
	
      LogInfo("FastCalorimetry") << " Tail of the shower " << std::endl;
      for(unsigned ir=0; ir < theTailIntervals_.size()/2;++ir)
	{
	  LogInfo("FastCalorimetry") << " r < " << theTailIntervals_[ir*2] << " R_M : " << theTailIntervals_[ir*2+1] << "        ";
	}
      LogInfo("FastCalotimetry") << "Radius correction factor " << radiusFactor_ << std::endl;
      LogInfo("FastCalorimetry") << std::endl;
    }

  LogInfo("FastCalorimetry") << " FrontLeakageProbability : " << pulledPadSurvivalProbability_ << std::endl;
  LogInfo("FastCalorimetry") << " GapLossProbability : " << crackPadSurvivalProbability_ << std::endl;

  //FR
  edm::ParameterSet HCALparameters = fastCalo.getParameter<edm::ParameterSet>("HCAL");
  optionHDSim_ = HCALparameters.getParameter<int>("SimOption");
  hdGridSize_ = HCALparameters.getParameter<int>("GridSize");
  hdSimMethod_ = HCALparameters.getParameter<int>("SimMethod");
  //RF

  unfoldedMode_ = fastCalo.getUntrackedParameter<bool>("UnfoldedMode",false);
}


void CalorimetryManager::updateMap(uint32_t cellid,float energy,int id,std::map<uint32_t,std::vector<std::pair<int,float> > > & mymap)
{
  //  std::cout << " updateMap " << std::endl;
  std::map<unsigned,std::vector<std::pair<int,float> > >::iterator cellitr;
  cellitr = mymap.find(cellid);
  if(!unfoldedMode_) id=0;
  if( cellitr==mymap.end())
    {      
      std::vector<std::pair<int,float> > myElement;
      myElement.push_back(std::pair<int,float> (id,energy));
      mymap[cellid]=myElement;
    }
  else
    {
      if(!unfoldedMode_)
	{
	  cellitr->second[0].second+=energy;
	}
      else
	cellitr->second.push_back(std::pair<int,float>(id,energy));
    }
}

void CalorimetryManager::updateMap(int hi,float energy,int tid,std::vector<std::vector<std::pair<int,float> > > & mymap, std::vector<int>& firedCells)
{
  // Standard case first : one entry per cell 
  if(!unfoldedMode_)
    {
      // if new entry, update the list 
      if(mymap[hi].size()==0)
	{
	  firedCells.push_back(hi);
	  mymap[hi].push_back(std::pair<int,float>(0,energy));
	}
      else
	mymap[hi][0].second+=energy;
    }
  else
    {
      //      std::cout << "update map " << mymap[hi].size() << " " << hi << std::setw(8) << std::setprecision(6) <<  energy ;
      //      std::cout << " " << mymap[hi][0].second << std::endl;
      // the minimal size is always 1 ; in this case, no push_back 
      if(mymap[hi].size()==0)
	{
	  //	  if(tid==0) std::cout << " Shit ! " << std::endl;
	  firedCells.push_back(hi);
	}

      mymap[hi].push_back(std::pair<int,float>(tid,energy));
    }
  
}

void CalorimetryManager::loadFromEcalBarrel(edm::PCaloHitContainer & c) const
{ 
  unsigned size=firedCellsEB_.size();
  //  float sum=0.;
  for(unsigned ic=0;ic<size;++ic)
    {
      int hi=firedCellsEB_[ic];
      if(!unfoldedMode_)
	{
	  c.push_back(PCaloHit(EBDetId::unhashIndex(hi),EBMapping_[hi][0].second,0.,0));
	  //	  std::cout << "Adding " << hi << " " << EBDetId::unhashIndex(hi) << " " ;
	  //	  std::cout << EBMapping_[hi][0].second << " " << EBMapping_[hi][0].first << std::endl;
	}
      else
	{
	  unsigned npart=EBMapping_[hi].size();
	  for(unsigned ip=0;ip<npart;++ip)
	    {
	      c.push_back(PCaloHit(EBDetId::unhashIndex(hi),EBMapping_[hi][ip].second,0.,
				   EBMapping_[hi][ip].first));

	    }
	}
	
      //      sum+=cellit->second;
    }
  
//  for(unsigned ic=0;ic<61200;++ic) 
//    { 
//      EBDetId myCell(EBDetId::unhashIndex(ic)); 
//      if(!myCell.null()) 
//        { 
//	  float total=0.;
//	  for(unsigned id=0;id<EBMapping_[ic].size();++id)
//	    total+=EBMapping_[ic][id].second;
//	  if(EBMapping_[ic].size()>0)
//	    std::cout << "Adding " << ic << " " << myCell << " " << std::setprecision(8) <<total << std::endl; 
//        } 
//    } 


  //  std::cout << " SUM : " << sum << std::endl;
  //  std::cout << " Added " <<c.size() << " hits " <<std::endl;
}


void CalorimetryManager::loadFromEcalEndcap(edm::PCaloHitContainer & c) const
{
  unsigned size=firedCellsEE_.size();
  //  float sum=0.;
  for(unsigned ic=0;ic<size;++ic)
    {
      int hi=firedCellsEE_[ic];
      if(!unfoldedMode_)
	c.push_back(PCaloHit(EEDetId::unhashIndex(hi),EEMapping_[hi][0].second,0.,0));
      else
	{
	  unsigned npart=EEMapping_[hi].size();
	  for(unsigned ip=0;ip<npart;++ip)
	    c.push_back(PCaloHit(EEDetId::unhashIndex(hi),EEMapping_[hi][ip].second,0.,
				 EEMapping_[hi][ip].first));
	}
	
      //      sum+=cellit->second;
    }
  //  std::cout << " SUM : " << sum << std::endl;
  //  std::cout << " Added " <<c.size() << " hits " <<std::endl;
}

void CalorimetryManager::loadFromHcal(edm::PCaloHitContainer & c) const
{
  unsigned size=firedCellsHCAL_.size();
  //  float sum=0.;
  for(unsigned ic=0;ic<size;++ic)
    {
      int hi=firedCellsHCAL_[ic];
      if(!unfoldedMode_)
	c.push_back(PCaloHit(theDetIds_[hi],HMapping_[hi][0].second,0.,0));
      else
	{
	  unsigned npart=HMapping_[hi].size();
	  for(unsigned ip=0;ip<npart;++ip)
	    c.push_back(PCaloHit(theDetIds_[hi],HMapping_[hi][ip].second,0.,
				 HMapping_[hi][ip].first));
	}
	
      //      sum+=cellit->second;
    }
  //  std::cout << " SUM : " << sum << std::endl;
  //  std::cout << " Added " <<c.size() << " hits " <<std::endl;
}

void CalorimetryManager::loadFromPreshower(edm::PCaloHitContainer & c) const
{
  std::map<uint32_t,std::vector<std::pair< int,float> > >::const_iterator cellit;
  std::map<uint32_t,std::vector<std::pair <int,float> > >::const_iterator preshEnd=ESMapping_.end();
  
  for(cellit=ESMapping_.begin();cellit!=preshEnd;++cellit)
    {
      if(!unfoldedMode_)	
	c.push_back(PCaloHit(cellit->first,cellit->second[0].second,0.,0));
      else
	{
	  unsigned npart=cellit->second.size();
	  for(unsigned ip=0;ip<npart;++ip)
	    {
	      c.push_back(PCaloHit(cellit->first,cellit->second[ip].second,0.,cellit->second[ip].first));
	    }
	}
    }
}
