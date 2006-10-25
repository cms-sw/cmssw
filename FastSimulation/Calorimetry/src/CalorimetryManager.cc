// Fast Simulation headers
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDRShower.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
#include "FastSimulation/Utilities/interface/Histos.h"
  
// STL headers 
#include <vector>
#include <iostream>

// CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"

//CMSSW headers 
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

CalorimetryManager::CalorimetryManager():myCalorimeter_(0),myHistos(0)
{;}

CalorimetryManager::CalorimetryManager(FSimEvent * aSimEvent, const edm::ParameterSet& fastCalo)
  :mySimEvent(aSimEvent)
{
  readParameters(fastCalo);

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

  myHistos->bookByNumber("h30",0,7,300,0.,3.,100,0.,35.);
  myHistos->book("h310",75,0.,3.,"");
#endif
  myCalorimeter_ = new CaloGeometryHelper(fastCalo);
  myHDResponse = new HCALResponse(fastCalo.getParameter<edm::ParameterSet>("HCALResponse"));
}

CalorimetryManager::~CalorimetryManager()
{
#ifdef FAMOSDEBUG
  std::cout << " Bordel FamosDebug is enabled " << std::endl;
  myHistos->put("Famos.root");
  if(myHistos) delete myHistos;
#endif

  if(myCalorimeter_) delete myCalorimeter_;
  delete myHDResponse;
}

void CalorimetryManager::reconstruct()
{
  // Clear the content of the calorimeters 
  EBMapping_.clear();
  EEMapping_.clear();
  ESMapping_.clear();
  HMapping_.clear();

  LogInfo("FastCalorimetry") << "Reconstructing " << (int) mySimEvent->nTracks() << " tracks." << endl;
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
	    reconstructECAL(myTrack);
	   
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
  HepPoint3D ecalentrance = myPart.vertex().vect();
  
  //  std::cout << " Ecal entrance " << ecalentrance << std::endl;
  
  // The preshower
  PreshowerHitMaker * myPreshower = NULL ;
  if(onLayer1 || onLayer2)
    {
      HepPoint3D layer1entrance,layer2entrance;
      HepVector3D dir1,dir2;
      if(onLayer1) 
	{
	  layer1entrance = HepPoint3D(myTrack.layer1Entrance().vertex().vect());
	  dir1 = HepVector3D(myTrack.layer1Entrance().vect().unit());
	}
      if(onLayer2) 
	{
	  layer2entrance = HepPoint3D(myTrack.layer2Entrance().vertex().vect());
	  dir2 = HepVector3D(myTrack.layer2Entrance().vect().unit());
	}
      //      std::cout << " Layer1entrance " << layer1entrance << std::endl;
      //      std::cout << " Layer2entrance " << layer2entrance << std::endl;
      myPreshower = new PreshowerHitMaker(myCalorimeter_,layer1entrance,dir1,layer2entrance,dir2);
    }

  // The ECAL Properties
  EMECALShowerParametrization 
    showerparam(myCalorimeter_->ecalProperties(onEcal), 
		myCalorimeter_->hcalProperties(onHcal), 
		myCalorimeter_->layer1Properties(onLayer1), 
		myCalorimeter_->layer2Properties(onLayer2),
		theCoreIntervals_,
		theTailIntervals_);

  // Photons : create an e+e- pair
  if ( myTrack.type() == 22 ) {
    
    // Depth for the first e+e- pair creation (in X0)
    X0depth = -log(RandFlat::shoot()) * (9./7.);
    
    // Initialization
    double eMass = 0.000510998902; 
    double xe=0;
    double xm=eMass/myPart.e();
    double weight = 0.;
    
    // Generate electron energy between emass and eGamma-emass
    do {
      xe = RandFlat::shoot()*(1.-2.*xm) + xm;
      weight = 1. - 4./3.*xe*(1.-xe);
    } while ( weight < RandFlat::shoot() );
    
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


  EMShower theShower(&showerparam,&thePart);


  double depth((X0depth+theShower.getMaximumOfShower())*myCalorimeter_->ecalProperties(onEcal)->radLenIncm());
  HepPoint3D meanShower=ecalentrance+myPart.vect().unit()*depth;
  
  //  if(onEcal!=1) return ; 

  // The closest crystal
  DetId pivot(myCalorimeter_->getClosestCell(meanShower, true, onEcal==1));

  //  std::cout << " After getClosestCell " << std::endl;
  
  EcalHitMaker myGrid(myCalorimeter_,ecalentrance,pivot,onEcal,size,0);
  //                                             ^^^^
  //                                         for EM showers
  myGrid.setPulledPadSurvivalProbability(pulledPadSurvivalProbability_);
  myGrid.setCrackPadSurvivalProbability(crackPadSurvivalProbability_);
  myGrid.setRadiusFactor(radiusFactor_);
  
  // The shower simulation
  myGrid.setTrackParameters(myPart.vect().unit(),X0depth,myTrack);

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
      updateMap(mapitr->first,mapitr->second,(onEcal==1)?EBMapping_:EEMapping_);
      //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
    }

  // Now fill the HCAL hits
  endmapitr=myHcalHitMaker.getHits().end();
  for(mapitr=myHcalHitMaker.getHits().begin();mapitr!=endmapitr;++mapitr)
    {
      updateMap(mapitr->first,mapitr->second,HMapping_);
      //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
    }

  // delete the preshower
  if(myPreshower!=0)
    {
      endmapitr=myPreshower->getHits().end();
      for(mapitr=myPreshower->getHits().begin();mapitr!=endmapitr;++mapitr)
	{
	  updateMap(mapitr->first,mapitr->second,ESMapping_);
	  //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
	}
      delete myPreshower;
    }
  //  std::cout << " Deleting myPreshower " << std::endl;
  
}



// Simulation of electromagnetic showers in VFCAL
void CalorimetryManager::reconstructECAL(const FSimTrack& track) {
  ;
}

void CalorimetryManager::reconstructHCAL(const FSimTrack& myTrack)
{
  int hit;
  int pid = abs(myTrack.type());
  //  FSimTrack myTrack = mySimEvent.track(fsimi);

  //  int pid=abs(myTrack.type());
  //  std::cout << "reconstructHCAL " << std::endl;
  
  HepLorentzVector trackPosition;
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
    pair<double,double> response =
      myHDResponse->responseHCAL(EGen, pathEta, 2); // 2=muon 
    emeas  = response.first;
    if(debug_)
      LogDebug("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - MUON !!!" << endl;
  }
  else {
    e     = myHDResponse->getHCALEnergyResponse(EGen,hit);
    sigma = myHDResponse->getHCALEnergyResolution(EGen, hit);
  
    double emeas = 0.;
    emeas = RandGaussQ::shoot(e,sigma);  
  }

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - on-calo " << endl  
         << "   eta  = " << pathEta << endl
         << "   phi  = " << pathPhi << endl 
	 << "  Egen  = " << EGen << endl
	 << "  Eres  = " << e << endl
	 << " sigma  = " << sigma << endl
	 << "  Emeas = " << emeas << endl;


  if(emeas > 0.) {  
    DetId cell = myCalorimeter_->getClosestCell(trackPosition.vect(),false,false);
    updateMap(cell.rawId(), emeas, HMapping_);
  }
}

void CalorimetryManager::HDShowerSimulation(const FSimTrack& myTrack)
{
  //  TimeMe t(" FASTEnergyReconstructor::HDShower");
  HepLorentzVector moment = myTrack.momentum();

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::HDShowerSimulation - track param."
         << endl
	 << "  eta = " << moment.eta() << endl
         << "  phi = " << moment.phi() << endl
         << "   et = " << moment.et()  << endl;

  int hit;
  //  int pid = abs(myTrack.type());

  HepLorentzVector trackPosition;
  if ( myTrack.onEcal() ) {
    trackPosition=myTrack.ecalEntrance().vertex();
    hit = myTrack.onEcal()-1;                               //
    myPart = myTrack.ecalEntrance();
  } else {
    trackPosition=myTrack.vfcalEntrance().vertex();
    hit = 2;
    myPart = myTrack.vfcalEntrance();
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
    e     = myHDResponse->getHCALEnergyResponse  (eGen, hit);
    sigma = myHDResponse->getHCALEnergyResolution(eGen, hit);
  }
  else { // optionHDsim == 2
    pair<double,double> response =
      myHDResponse->responseHCAL(eGen, pathEta, 1); // 1=hadron 
    e     = response.first;
    sigma = response.second;
  }

  double emeas = 0.;
  emeas = RandGaussQ::shoot(e,sigma);

  if(debug_)
    LogDebug("FastCalorimetry") << "CalorimetryManager::HDShowerSimulation - on-calo " << endl  
         << "   eta  = " << pathEta << endl
         << "   phi  = " << pathPhi << endl 
	 << "  Egen  = " << eGen << endl
	 << "  Eres  = " << e << endl
	 << " sigma  = " << sigma << endl
	 << "  Emeas = " << emeas << endl;
  
  //===========================================================================
  if(emeas > 0.) {  

    // Special case - temporary protection until fix in Grid will be done
//    if(fabs(pathEta) > 1.44 && fabs(pathEta) < 1.45) {
//      CellID cell = myCalorimeter->getClosestCell(trackPosition.vect(),false);
//      updateMap(cell,emeas);
//
//    } 
//    else {  //  Beginning of special "else"
      //=====================================

    
    // ECAL and HCAL properties to get
    HDShowerParametrization 
      theHDShowerparam(myCalorimeter_->ecalProperties(onECAL),
		       myCalorimeter_->hcalProperties(onHCAL));
    
    //Making ECAL Grid (and segments calculation)
    HepPoint3D caloentrance;
    HepVector3D direction;
    if(myTrack.onEcal()) 
      {	
	caloentrance = myTrack.ecalEntrance().vertex().vect();
	direction = myTrack.ecalEntrance().vect().unit();
      }
    else if(myTrack.onHcal())
      {
	caloentrance = myTrack.hcalEntrance().vertex().vect();
	direction = myTrack.hcalEntrance().vect().unit();
      }
    else
      {
	caloentrance = myTrack.vfcalEntrance().vertex().vect();
	direction = myTrack.vfcalEntrance().vect().unit();
      }

    DetId pivot;
    if(myTrack.onEcal())
      {
	pivot=myCalorimeter_->getClosestCell(caloentrance,
					     true, myTrack.onEcal()==1);
      }
    else if(myTrack.onHcal())
      {
	pivot=myCalorimeter_->getClosestCell(caloentrance,					     
					    false, false);
      }
//    if(pivot.isZero())
//      {
//	std::cout << " HDShowerSim pivot is Zero "  << std::endl;
//	std::cout <<" Ecal entrance " << caloentrance << std::endl;
//      }
    EcalHitMaker myGrid(myCalorimeter_,caloentrance,pivot,
			pivot.null()? 0 : myTrack.onEcal(),hdGridSize_,1);
    // 1=HAD shower

    myGrid.setTrackParameters(direction,0,myTrack);
    // Build the FAMOS HCAL 
    HcalHitMaker myHcalHitMaker(myGrid,(unsigned)1); 
    
    // Shower simulation
    bool status;
    if(hdSimMethod_ == 0) {
      HDShower theShower(&theHDShowerparam,&myGrid,&myHcalHitMaker,onECAL,emeas);
      status = theShower.compute();
    }
    else {
      HDRShower theShower(&theHDShowerparam,&myGrid,&myHcalHitMaker,onECAL,emeas);
      status = theShower.computeShower();
    }

    if(status) {
      
      // was map<unsigned,double> but CaloHitMaker uses float
      std::map<unsigned,float>::const_iterator mapitr;
      std::map<unsigned,float>::const_iterator endmapitr;
      if(myTrack.onVFcal() != 2) {
	// Save ECAL hits 
	endmapitr=myGrid.getHits().end();
	for(mapitr=myGrid.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	  double energy = mapitr->second;
	  if(energy > 0.000001) { 
	    //////////////////////////////////////////////////////////////////////////////
	    // EBRYMapping=EBMapping_? EFRYMapping=EEMapping_????
	    updateMap(mapitr->first,energy,(onECAL==1)?EBMapping_:EEMapping_);
	    if(debug_)
	      LogDebug("FastCalorimetry") << " ECAL cell " << mapitr->first << " added,  E = " 
		   << energy << endl;  
	  }
	}
      }
      
      // Save HCAL hits
      endmapitr=myHcalHitMaker.getHits().end();
      for(mapitr=myHcalHitMaker.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	updateMap(mapitr->first,mapitr->second,HMapping_);
	if(debug_)
	  LogDebug("FastCalorimetry") << " HCAL cell "  
	       << mapitr->first << " added    E = " 
	       << mapitr->second << endl;  
      }
    }      
    else {  // shower simulation failed 
      DetId cell = myCalorimeter_->getClosestCell(trackPosition.vect(),false,false);
      updateMap(cell.rawId(),emeas,HMapping_);
      if(debug_)
	LogDebug("FastCalorimetry") << " HCAL simple cell "   
	     << cell.rawId() << " added    E = " 
	     << emeas << endl;  
    }

    //    } // End of special "else" 
    //===========================

  } // e > 0. ...

  if(debug_)
    LogDebug("FastCalorimetry") << endl << " FASTEnergyReconstructor::HDShowerSimulation  finished "
	 << endl;
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
}


void CalorimetryManager::updateMap(uint32_t cellid,float energy,std::map<unsigned,float>& mymap)
{
  //  std::cout << " updateMap " << std::endl;
  std::map<unsigned,float>::iterator cellitr;
  cellitr = mymap.find(cellid);
  if( cellitr==mymap.end())
    {
      mymap.insert(std::pair<unsigned,float>(cellid,energy));
    }
  else
    {
      cellitr->second+=energy;
    }
}

void CalorimetryManager::loadFromEcalBarrel(edm::PCaloHitContainer & c) const
{
  std::map<unsigned,float>::const_iterator cellit;
  std::map<unsigned,float>::const_iterator barrelEnd=EBMapping_.end();
  
  for(cellit=EBMapping_.begin();cellit!=barrelEnd;++cellit)
    {
      // Add the PCaloHit. No time, no track number 
      //      if(DetId(cellit->first).null()) std::cout << " PCaloHit. Ooops null " << std::endl;
      c.push_back(PCaloHit(cellit->first,cellit->second,0.,0));
    }
}


void CalorimetryManager::loadFromEcalEndcap(edm::PCaloHitContainer & c) const
{
  std::map<unsigned,float>::const_iterator cellit;
  std::map<unsigned,float>::const_iterator endcapEnd=EEMapping_.end();
  
  for(cellit=EEMapping_.begin();cellit!=endcapEnd;++cellit)
    {
      // Add the PCaloHit. No time, no track number 
      //      if(DetId(cellit->first).null()) std::cout << " PCaloHit. Ooops null " << std::endl;
      c.push_back(PCaloHit(cellit->first,cellit->second,0.,0));
    }
}

void CalorimetryManager::loadFromHcal(edm::PCaloHitContainer & c) const
{
  std::map<unsigned,float>::const_iterator cellit;
  std::map<unsigned,float>::const_iterator hcalEnd=HMapping_.end();
  
  for(cellit=HMapping_.begin();cellit!=hcalEnd;++cellit)
    {
      // Add the PCaloHit. No time, no track number
      //      if(DetId(cellit->first).null()) std::cout << " PCaloHit. Ooops null " << std::endl;
      c.push_back(PCaloHit(cellit->first,cellit->second,0.,0));
    }
}



