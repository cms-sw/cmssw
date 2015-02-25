//updated by Reza Goldouzian
//Framework headers 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Fast Simulation headers
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/HDShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HFShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HDRShower.h"
#include "FastSimulation/ShowerDevelopment/interface/HSParameters.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
//#include "FastSimulation/Utilities/interface/Histos.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/GammaFunctionGenerator.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"  
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FastSimulation/Event/interface/FSimTrackEqual.h"
// New headers for Muon Mip Simulation
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
// Muon Brem
#include "FastSimulation/MaterialEffects/interface/MuonBremsstrahlungSimulator.h"

//Gflash Hadronic Model
#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashPiKShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashProtonShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashAntiProtonShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashTrajectoryPoint.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"
#include "SimGeneral/GFlash/interface/Gflash3Vector.h"

//FastHFShowerLibrary
#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"

// STL headers 
#include <vector>
#include <iostream>

//CMSSW headers 
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//ROOT headers
#include "TROOT.h"
#include "TH1.h"

using namespace edm;

typedef math::XYZVector XYZVector;
typedef math::XYZVector XYZPoint;

std::vector<std::pair<int,float> > CalorimetryManager::myZero_ = std::vector<std::pair<int,float> >
(1,std::pair<int,float>(0,0.));

CalorimetryManager::CalorimetryManager() : 
  myCalorimeter_(0),
  //  myHistos(0),
  initialized_(false)
{;}

CalorimetryManager::CalorimetryManager(FSimEvent * aSimEvent, 
				       const edm::ParameterSet& fastCalo,
				       const edm::ParameterSet& fastMuECAL,
				       const edm::ParameterSet& fastMuHCAL,
                                       const edm::ParameterSet& parGflash)
  : 
  mySimEvent(aSimEvent), 
  initialized_(false),
  theMuonEcalEffects(0), theMuonHcalEffects (0), bFixedLength_(false)
{

  aLandauGenerator = new LandauFluctuationGenerator;
  aGammaGenerator = new GammaFunctionGenerator;

  //Gflash
  theProfile = new GflashHadronShowerProfile(parGflash);
  thePiKProfile = new GflashPiKShowerProfile(parGflash);
  theProtonProfile = new GflashProtonShowerProfile(parGflash);
  theAntiProtonProfile = new GflashAntiProtonShowerProfile(parGflash);

  // FastHFShowerLibrary
  theHFShowerLibrary = new FastHFShowerLibrary(fastCalo);

  readParameters(fastCalo);

  //  myHistos = 0; 

  //   myHistos = Histos::instance();
  //   myHistos->book("h10",140,-3.5,3.5,100,-0.5,99.5);
  //   myHistos->book("h20",150,0,150.,100,-0.5,99.5);
  //   myHistos->book("h100",140,-3.5,3.5,100,0,0.1);
  //   myHistos->book("h110",140,-3.5,3.5,100,0,10.);
  //   myHistos->book("h120",200,-5.,5.,100,0,0.5);
  
  //   myHistos->book("h200",300,0,3.,100,0.,35.);
  //   myHistos->book("h210",720,-M_PI,M_PI,100,0,35.);
  //   myHistos->book("h212",720,-M_PI,M_PI,100,0,35.);
  
  //   myHistos->bookByNumber("h30",0,7,300,-3.,3.,100,0.,35.);
  //   myHistos->book("h310",75,-3.,3.,"");
  //   myHistos->book("h400",100,-10.,10.,100,0.,35.);
  //   myHistos->book("h410",720,-M_PI,M_PI);

  myCalorimeter_ = 
    new CaloGeometryHelper(fastCalo);
  myHDResponse_ = 
    new HCALResponse(fastCalo.getParameter<edm::ParameterSet>("HCALResponse"));
  myHSParameters_ = 
    new HSParameters(fastCalo.getParameter<edm::ParameterSet>("HSParameters"));

  // Material Effects for Muons in ECAL (only EnergyLoss implemented so far)

  if ( fastMuECAL.getParameter<bool>("PairProduction") || 
       fastMuECAL.getParameter<bool>("Bremsstrahlung") ||
       fastMuECAL.getParameter<bool>("MuonBremsstrahlung") ||
       fastMuECAL.getParameter<bool>("EnergyLoss") || 
       fastMuECAL.getParameter<bool>("MultipleScattering") )
    theMuonEcalEffects = new MaterialEffects(fastMuECAL);

  // Material Effects for Muons in HCAL (only EnergyLoss implemented so far)

  if ( fastMuHCAL.getParameter<bool>("PairProduction") || 
       fastMuHCAL.getParameter<bool>("Bremsstrahlung") ||
       fastMuHCAL.getParameter<bool>("MuonBremsstrahlung") ||
       fastMuHCAL.getParameter<bool>("EnergyLoss") || 
       fastMuHCAL.getParameter<bool>("MultipleScattering") )
    theMuonHcalEffects = new MaterialEffects(fastMuHCAL);


}

void CalorimetryManager::clean()
{
  EBMapping_.clear();
  EEMapping_.clear();
  HMapping_.clear();
  ESMapping_.clear();
  muonSimTracks.clear();
}

CalorimetryManager::~CalorimetryManager()
{
  if(myCalorimeter_) delete myCalorimeter_;
  if(myHDResponse_) delete myHDResponse_;

  if ( theMuonEcalEffects ) delete theMuonEcalEffects;
  if ( theMuonHcalEffects ) delete theMuonHcalEffects;

  if ( theProfile ) delete theProfile;

  if ( theHFShowerLibrary ) delete theHFShowerLibrary;
}

void CalorimetryManager::reconstruct(RandomEngineAndDistribution const* random)
{

  if(evtsToDebug_.size())
    {
      std::vector<unsigned int>::const_iterator itcheck=find(evtsToDebug_.begin(),evtsToDebug_.end(),mySimEvent->id().event());
      debug_=(itcheck!=evtsToDebug_.end());
      if(debug_)
	mySimEvent->print();
    }
  // Clear the content of the calorimeters 
  if(!initialized_)
    {
      
      // Check if the preshower is really available
      if(simulatePreshower_ && !myCalorimeter_->preshowerPresent())
	{
	  std::cout << " WARNING " << std::endl;
	  std::cout << " The preshower simulation has been turned on; but no preshower geometry is available " << std::endl;
	  std::cout << " Disabling the preshower simulation " << std::endl;
	  simulatePreshower_ = false;
	}

      initialized_=true;
    }
  clean();

  LogInfo("FastCalorimetry") << "Reconstructing " << (int) mySimEvent->nTracks() << " tracks." << std::endl;
  for( int fsimi=0; fsimi < (int) mySimEvent->nTracks() ; ++fsimi) {

    FSimTrack& myTrack = mySimEvent->track(fsimi);

    int pid = abs(myTrack.type());

    if (debug_) {
      LogInfo("FastCalorimetry") << " ===> pid = "  << pid << std::endl;      
    }
    
    
    // Check that the particle hasn't decayed
    if(myTrack.noEndVertex()) {
      // Simulate energy smearing for photon and electrons
      if ( pid == 11 || pid == 22 ) {
	  
	if ( myTrack.onEcal() ) 
	  EMShowerSimulation(myTrack, random);
	else if ( myTrack.onVFcal() ) {
          if(useShowerLibrary) {
            theHFShowerLibrary->recoHFShowerLibrary(myTrack);  
            myHDResponse_->correctHF(myTrack.hcalEntrance().e(),abs(myTrack.type()));
            updateHCAL(theHFShowerLibrary->getHitsMap(),myTrack.id());
          } 
          else reconstructHCAL(myTrack, random);
	}   
      } // electron or photon
      else if (pid==13)
	{
          MuonMipSimulation(myTrack, random);
	}
      // Simulate energy smearing for hadrons (i.e., everything 
      // but muons... and SUSY particles that deserve a special 
      // treatment.
      else if ( pid < 1000000 ) {
	if ( myTrack.onHcal() || myTrack.onVFcal() ) { 	  
	  if(optionHDSim_ == 0 )  reconstructHCAL(myTrack, random);
	  else HDShowerSimulation(myTrack, random);
	}
      } // pid < 1000000 
    } // myTrack.noEndVertex()
  } // particle loop
  //  LogInfo("FastCalorimetry") << " Number of  hits (barrel)" << EBMapping_.size() << std::endl;
  //  LogInfo("FastCalorimetry") << " Number of  hits (Hcal)" << HMapping_.size() << std::endl;
  //  std::cout << " Nombre de hit (endcap)" << EEMapping_.size() << std::endl;

} // reconstruct

// Simulation of electromagnetic showers in PS, ECAL, HCAL 
void CalorimetryManager::EMShowerSimulation(const FSimTrack& myTrack,
                                            RandomEngineAndDistribution const* random) {
  std::vector<const RawParticle*> thePart;
  double X0depth;

  if (debug_) {
    LogInfo("FastCalorimetry") << " EMShowerSimulation "  <<myTrack << std::endl;      
  }
  
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
  if(simulatePreshower_ && (onLayer1 || onLayer2))
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
					  aLandauGenerator,
                                          random);
      myPreshower->setMipEnergy(mipValues_[0],mipValues_[1]);
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
  if(maxEnergy>100) size=11;
//  if ( maxEnergy < threshold5x5 ) size = 5;
//  if ( maxEnergy < threshold3x3 ) size = 3;


  EMShower theShower(random,aGammaGenerator,&showerparam,&thePart,  NULL, NULL, bFixedLength_);


  double maxShower = theShower.getMaximumOfShower();
  if (maxShower > 20.) maxShower = 2.; // simple pivot-searching protection 

  double depth((X0depth + maxShower) * 
	       myCalorimeter_->ecalProperties(onEcal)->radLenIncm());
  XYZPoint meanShower = ecalentrance + myPart.Vect().Unit()*depth;
  
  //  if(onEcal!=1) return ; 

  // The closest crystal
  DetId pivot(myCalorimeter_->getClosestCell(meanShower, true, onEcal==1));

  if(pivot.subdetId() == 0) {   // further protection against avbsence of pivot
    edm::LogWarning("CalorimetryManager") <<  "Pivot for egamma  e = "  << myTrack.hcalEntrance().e() << " is not found at depth " << depth << " and meanShower coordinates = " << meanShower << std::endl; 
    if(myPreshower) delete myPreshower;
    return;
  }
  
  EcalHitMaker myGrid(myCalorimeter_,ecalentrance,pivot,onEcal,size,0,random);
  //                                             ^^^^
  //                                         for EM showers
  myGrid.setPulledPadSurvivalProbability(pulledPadSurvivalProbability_);
  myGrid.setCrackPadSurvivalProbability(crackPadSurvivalProbability_);

  //maximumdepth dependence of the radiusfactorbehindpreshower
  //First tuning: Shilpi Jain (Mar-Apr 2010); changed after tuning - Feb-July - Shilpi Jain
  /* **************
  myGrid.setRadiusFactor(radiusFactor_);
  if(onLayer1 || onLayer2)
    {
      float b               = radiusPreshowerCorrections_[0];
      float a               = radiusFactor_*( 1.+radiusPreshowerCorrections_[1]*radiusPreshowerCorrections_[0] );
      float maxdepth        = X0depth+theShower.getMaximumOfShower();
      float newRadiusFactor = radiusFactor_;
      if(myPart.e()<=250.)
        {
	  newRadiusFactor = a/(1.+b*maxdepth); 
	}
      myGrid.setRadiusFactor(newRadiusFactor);
    }
  else // otherwise use the normal radius factor
    {
      myGrid.setRadiusFactor(radiusFactor_);
    }
  ************** */
  if(myTrack.onEcal() == 2) // if on EE  
     {
       if( (onLayer1 || onLayer2) && myPart.e()<=250.)
         {
	   double maxdepth        = X0depth+theShower.getMaximumOfShower();
	   double newRadiusFactor = radiusFactorEE_ * aTerm/(1.+bTerm*maxdepth);
	   myGrid.setRadiusFactor(newRadiusFactor);
	 }
       else // otherwise use the normal radius factor
         {
           myGrid.setRadiusFactor(radiusFactorEE_);
         }
     }//if(myTrack.onEcal() == 2)
  else                      // else if on EB
    {
      myGrid.setRadiusFactor(radiusFactorEB_);
    }
  //(end of) changed after tuning - Feb-July - Shilpi Jain

  myGrid.setPreshowerPresent(simulatePreshower_);
  
  // The shower simulation
  myGrid.setTrackParameters(myPart.Vect().Unit(),X0depth,myTrack);

//  std::cout << " PS ECAL GAP HCAL X0 " << myGrid.ps1TotalX0()+myGrid.ps2TotalX0() << " " << myGrid.ecalTotalX0();
//  std::cout << " " << myGrid.ecalHcalGapTotalX0() << " " << myGrid.hcalTotalX0() << std::endl;
//  std::cout << " PS ECAL GAP HCAL L0 " << myGrid.ps1TotalL0()+myGrid.ps2TotalL0() << " " << myGrid.ecalTotalL0();
//  std::cout << " " << myGrid.ecalHcalGapTotalL0() << " " << myGrid.hcalTotalL0() << std::endl;
//  std::cout << "ECAL-HCAL " << myTrack.momentum().eta() << " " <<  myGrid.ecalHcalGapTotalL0() << std::endl;
//
//  std::cout << " Grid created " << std::endl;
  if(myPreshower) theShower.setPreshower(myPreshower);
  
  HcalHitMaker myHcalHitMaker(myGrid,(unsigned)0); 

  theShower.setGrid(&myGrid);
  theShower.setHcal(&myHcalHitMaker);
  theShower.compute();
  //myHistos->fill("h502", myPart->eta(),myGrid.totalX0());
  
  // Save the hits !
  updateECAL(myGrid.getHits(),onEcal,myTrack.id());

  // Now fill the HCAL hits
  updateHCAL(myHcalHitMaker.getHits(),myTrack.id());

  // delete the preshower
  if(myPreshower!=0) {
    updatePreshower(myPreshower->getHits(),myTrack.id());
    delete myPreshower;
    //  std::cout << " Deleting myPreshower " << std::endl;
  }
  
}



void CalorimetryManager::reconstructHCAL(const FSimTrack& myTrack,
                                         RandomEngineAndDistribution const* random)
{
  int hit;
  int pid = abs(myTrack.type());
  if (debug_) {
    LogInfo("FastCalorimetry") << " reconstructHCAL "  << myTrack << std::endl;      
  }

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
  double emeas = 0.;
  //double emeas = -0.0001;
 
  if(pid == 13) { 
    //    std::cout << " We should not be here " << std::endl;
    emeas = myHDResponse_->responseHCAL(0, EGen, pathEta, 2, random); // 2=muon
    if(debug_)
      LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - MUON !!!" << std::endl;
  }
  else if( pid == 22 || pid == 11) {
    emeas = myHDResponse_->responseHCAL(0, EGen, pathEta, 0, random); // last par. = 0 = e/gamma
    //  cout <<  "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
    if(debug_)
	  LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
  }
  else {
    emeas = myHDResponse_->getHCALEnergyResponse(EGen, hit, random);
  }

  if(debug_)
    LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - on-calo "   
				<< "  eta = " << pathEta 
				<< "  phi = " << pathPhi 
				<< "  Egen = " << EGen 
				<< "  Emeas = " << emeas << std::endl;

  if(emeas > 0.) {  
    DetId cell = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,false);
	double tof = (myCalorimeter_->getHcalGeometry()->getGeometry(cell)->getPosition().mag())/29.98;//speed of light
	CaloHitID current_id(cell.rawId(),tof,myTrack.id());
	std::map<CaloHitID,float> hitMap;
	hitMap[current_id] = emeas;
	updateHCAL(hitMap,myTrack.id());
  }
}

void CalorimetryManager::HDShowerSimulation(const FSimTrack& myTrack, RandomEngineAndDistribution const* random){//,
					    // const edm::ParameterSet& fastCalo){

  //  TimeMe t(" FASTEnergyReconstructor::HDShower");
  XYZTLorentzVector moment = myTrack.momentum();
  
  if(debug_)
    LogInfo("FastCalorimetry") 
      << "CalorimetryManager::HDShowerSimulation - track param."
      << std::endl
      << "  eta = " << moment.eta() << std::endl
      << "  phi = " << moment.phi() << std::endl
      << "   et = " << moment.Et()  << std::endl
      << "   e  = " << myTrack.hcalEntrance().e() << std::endl;

  if (debug_) {
      LogInfo("FastCalorimetry") << " HDShowerSimulation "  << myTrack << std::endl;      
    }


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
      LogInfo("FastCalorimetry") << " The particle is not in the acceptance " << std::endl;
      return;
    }

  // int onHCAL = hit + 1; - specially for myCalorimeter->hcalProperties(onHCAL)
  // (below) to get VFcal properties ...
  int onHCAL = hit + 1;
  int onECAL = myTrack.onEcal();
  
  double pathEta   = trackPosition.eta();
  double pathPhi   = trackPosition.phi();	
  //  double pathTheta = trackPosition.theta();

  double eint  = moment.e();
  double eGen  = myTrack.hcalEntrance().e();

  double emeas = 0.;
  double pmip= myHDResponse_->getMIPfraction(eGen, pathEta);
//  std::cout << " CalorimetryManager onHcal " <<  pmip  << std::endl;

 
  //double emeas = -0.000001; 

  //===========================================================================
  if(eGen > 0.) {  

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

  if(debug_)
    LogInfo("FastCalorimetry") 
      << "CalorimetryManager::HDShowerSimulation - on-calo 1 "
      << std::endl
      << "  onEcal    = " <<  myTrack.onEcal()  << std::endl
      << "  onHcal    = " <<  myTrack.onHcal()  << std::endl
      << "  onVFcal   = " <<  myTrack.onVFcal() << std::endl
      << "  position  = " << caloentrance << std::endl;


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

    EcalHitMaker myGrid(myCalorimeter_,caloentrance,pivot,
			pivot.null()? 0 : myTrack.onEcal(),hdGridSize_,1,
			random);
    // 1=HAD shower

    myGrid.setTrackParameters(direction,0,myTrack);
    // Build the FAMOS HCAL 
    HcalHitMaker myHcalHitMaker(myGrid,(unsigned)1); 
    
    // Shower simulation
    bool status = false;
    int  mip = 2;
    // Use HFShower for HF
    if ( !myTrack.onEcal() && !myTrack.onHcal() ) {
      //      std::cout << "CalorimetryManager::HDShowerSimulation(): track entrance = "
      //        << myTrack.vfcalEntrance().vertex().X() << " "
      //        << myTrack.vfcalEntrance().vertex().Y() << " "
      //        << myTrack.vfcalEntrance().vertex().Z() << " "
      //        << " , Energy (Gen/Scale) = " << eGen << " " << e << std::endl;

      // Warning : We give here the particle energy with the response
      //           but without the resolution/gaussian smearing
      //           For HF, the resolution is due to the PE statistic

      if(useShowerLibrary) {
         theHFShowerLibrary->recoHFShowerLibrary(myTrack);  
         status = true;
      } else {
        HFShower theShower(random,
	    		   &theHDShowerparam,
	  		   &myGrid,
			   &myHcalHitMaker,
			   onECAL,
			   eGen);
			   //			 eGen);
			   //			 e); // PV Warning : temporarly set the energy to the generated E

        status = theShower.compute();
      }
    } else { 
      if(hdSimMethod_ == 0) {
	HDShower theShower(random,
			   &theHDShowerparam,
			   &myGrid,
			   &myHcalHitMaker,
			   onECAL,
			   eGen,
                           pmip);
	status = theShower.compute();
        mip    = theShower.getmip();
      }
      else if (hdSimMethod_ == 1) {
	HDRShower theShower(random,
			    &theHDShowerparam,
			    &myGrid,
			    &myHcalHitMaker,
			    onECAL,
			    eGen);
	status = theShower.computeShower();
        mip = 2;
      }
      else if (hdSimMethod_ == 2 ) {
	//        std::cout << "Using GflashHadronShowerProfile hdSimMethod_ == 2" << std::endl;

        //dynamically loading a corresponding profile by the particle type
        int particleType = myTrack.type();
        theProfile = thePiKProfile;
        if(particleType == -2212) theProfile = theAntiProtonProfile;
        else if(particleType == 2212) theProfile = theProtonProfile;

        //input variables for GflashHadronShowerProfile
        int showerType = 99 + myTrack.onEcal();
        double globalTime = 150.0; // a temporary reference hit time in nanosecond
        float charge = (float)(myTrack.charge());
        Gflash3Vector gfpos(trackPosition.X(),trackPosition.Y(),trackPosition.Z());
        Gflash3Vector gfmom(moment.X(),moment.Y(),moment.Z());

        theProfile->initialize(showerType,eGen,globalTime,charge,gfpos,gfmom);
        theProfile->loadParameters();
        theProfile->hadronicParameterization();

        //make hits
	std::vector<GflashHit>& gflashHitList = theProfile->getGflashHitList();
	std::vector<GflashHit>::const_iterator spotIter    = gflashHitList.begin();
	std::vector<GflashHit>::const_iterator spotIterEnd = gflashHitList.end();

	Gflash::CalorimeterNumber whichCalor = Gflash::kNULL;

        for( ; spotIter != spotIterEnd; spotIter++){

          double pathLength = theProfile->getGflashShowino()->getPathLengthAtShower()
            + (30*100/eGen)*(spotIter->getTime() - globalTime);

          double currentDepth = std::max(0.0,pathLength - theProfile->getGflashShowino()->getPathLengthOnEcal());

          //find the the showino position at the currentDepth
          GflashTrajectoryPoint trajectoryPoint;
          theProfile->getGflashShowino()->getHelix()->getGflashTrajectoryPoint(trajectoryPoint,pathLength);
          Gflash3Vector positionAtCurrentDepth = trajectoryPoint.getPosition();
          //find radial distrance
          Gflash3Vector lateralDisplacement = positionAtCurrentDepth - spotIter->getPosition()/CLHEP::cm;
          double rShower = lateralDisplacement.r();
          double azimuthalAngle = lateralDisplacement.phi();

          whichCalor = Gflash::getCalorimeterNumber(positionAtCurrentDepth);

          if(whichCalor==Gflash::kESPM || whichCalor==Gflash::kENCA) {
            bool statusPad = myGrid.getPads(currentDepth,true);
            if(!statusPad) continue;
            myGrid.setSpotEnergy(1.2*spotIter->getEnergy()/CLHEP::GeV);
            myGrid.addHit(rShower/Gflash::intLength[Gflash::kESPM],azimuthalAngle,0);
          }
          else if(whichCalor==Gflash::kHB || whichCalor==Gflash::kHE) {
            bool setHDdepth = myHcalHitMaker.setDepth(currentDepth,true);
            if(!setHDdepth) continue;
            myHcalHitMaker.setSpotEnergy(1.4*spotIter->getEnergy()/CLHEP::GeV);
            myHcalHitMaker.addHit(rShower/Gflash::intLength[Gflash::kHB],azimuthalAngle,0);
          }
        }
        status = true;
      }
      else {
	edm::LogInfo("FastSimulationCalorimetry") << " SimMethod " << hdSimMethod_ <<" is NOT available ";
      }
    }


    if(status) {

      // Here to switch between simple formulae and parameterized response
      if(optionHDSim_ == 1) {
        emeas = myHDResponse_->getHCALEnergyResponse(eGen, hit, random);
      }
      else { // optionHDsim == 2
        emeas = myHDResponse_->responseHCAL(mip, eGen, pathEta, 1, random); // 1=hadron
      }

      double correction = emeas / eGen;

      // RespCorrP factors (ECAL and HCAL separately) calculation
      respCorr(eint);     

      if(debug_)
	LogInfo("FastCalorimetry") 
	  << "CalorimetryManager::HDShowerSimulation - on-calo 2" << std::endl
	  << "   eta  = " << pathEta << std::endl
	  << "   phi  = " << pathPhi << std::endl
	  << "  Egen  = " << eGen << std::endl
	  << " Emeas  = " << emeas << std::endl
	  << "  corr  = " << correction << std::endl
	  << "   mip  = " << mip << std::endl;  

      if(myTrack.onEcal() > 0) {
	    // Save ECAL hits
		updateECAL(myGrid.getHits(),onECAL,myTrack.id(),correction*ecorr);
	  }

      // Save HCAL hits
      if(myTrack.onVFcal() && useShowerLibrary) {
        myHDResponse_->correctHF(eGen,abs(myTrack.type()));
        updateHCAL(theHFShowerLibrary->getHitsMap(),myTrack.id());
      } 
      else 
        updateHCAL(myHcalHitMaker.getHits(),myTrack.id(),correction*hcorr);
	  
    }      
    else {  // shower simulation failed  
//      std::cout << " Shower simulation failed " << trackPosition.Vect() << std::endl;
//      std::cout << " The FSimTrack " << myTrack << std::endl;
//      std::cout << " HF entrance on VFcal" << myTrack.onVFcal() << std::endl;
//      std::cout << " trackPosition.eta() " << trackPosition.eta() << std::endl;
      if(myTrack.onHcal() || myTrack.onVFcal())
	{
	  DetId cell = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,false);
	  double tof = (myCalorimeter_->getHcalGeometry()->getGeometry(cell)->getPosition().mag())/29.98;//speed of light
      CaloHitID current_id(cell.rawId(),tof,myTrack.id());
	  std::map<CaloHitID,float> hitMap;
	  hitMap[current_id] = emeas;
	  updateHCAL(hitMap,myTrack.id());
	  if(debug_)
	    LogInfo("FastCalorimetry") << " HCAL simple cell "   
					<< cell.rawId() << " added    E = " 
					<< emeas << std::endl;  
	}
    }

  } // e > 0. ...

  if(debug_)
    LogInfo("FastCalorimetry") << std::endl << " FASTEnergyReconstructor::HDShowerSimulation  finished "
	 << std::endl;
}


void CalorimetryManager::MuonMipSimulation(const FSimTrack& myTrack, RandomEngineAndDistribution const* random)
{
  //  TimeMe t(" FASTEnergyReconstructor::HDShower");
  XYZTLorentzVector moment = myTrack.momentum();

  // Backward compatibility behaviour
  if(!theMuonHcalEffects) 
    {
      if(myTrack.onHcal() || myTrack.onVFcal() ) 
	reconstructHCAL(myTrack, random);

      return;
    }

  if(debug_)
    LogInfo("FastCalorimetry") << "CalorimetryManager::MuonMipSimulation - track param."
         << std::endl
	 << "  eta = " << moment.eta() << std::endl
         << "  phi = " << moment.phi() << std::endl
         << "   et = " << moment.Et()  << std::endl;

  //  int hit;
  //  int pid = abs(myTrack.type());

  XYZTLorentzVector trackPosition;
  if ( myTrack.onEcal() ) {
    trackPosition=myTrack.ecalEntrance().vertex();
    //    hit = myTrack.onEcal()-1;                               //
    myPart = myTrack.ecalEntrance();
  } else if ( myTrack.onVFcal()) {
    trackPosition=myTrack.vfcalEntrance().vertex();
    //    hit = 2;
    myPart = myTrack.vfcalEntrance();
  }
  else
    {
      LogInfo("FastCalorimetry") << " The particle is not in the acceptance " << std::endl;
      return;
    }

  // int onHCAL = hit + 1; - specially for myCalorimeter->hcalProperties(onHCAL)
  // (below) to get VFcal properties ...
  // not needed ? 
  //  int onHCAL = hit + 1; 
  int onECAL = myTrack.onEcal();
  
  //  double pathEta   = trackPosition.eta();
  //  double pathPhi   = trackPosition.phi();	
  //  double pathTheta = trackPosition.theta();
  
  //===========================================================================

  // ECAL and HCAL properties to get
      
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
  
  EcalHitMaker myGrid(myCalorimeter_,caloentrance,pivot,
		      pivot.null()? 0 : myTrack.onEcal(),hdGridSize_,0,
		      random);
  // 0 =EM shower -> Unit = X0
  
  myGrid.setTrackParameters(direction,0,myTrack);
  
  // Now get the path in the Preshower, ECAL and HCAL along a straight line extrapolation 
  // but only those in the ECAL are used 
 
  const std::vector<CaloSegment>& segments=myGrid.getSegments();
  unsigned nsegments=segments.size();
  
  int ifirstHcal=-1;
  int ilastEcal=-1;
 
  EnergyLossSimulator* energyLossECAL = 0;
  if (theMuonEcalEffects) energyLossECAL = theMuonEcalEffects->energyLossSimulator();
  //  // Muon brem in ECAL
  //  MuonBremsstrahlungSimulator* muonBremECAL = 0;
  //  if (theMuonEcalEffects) muonBremECAL = theMuonEcalEffects->muonBremsstrahlungSimulator();

  for(unsigned iseg=0;iseg<nsegments&&ifirstHcal<0;++iseg)
    {
      
      // in the ECAL, there are two types of segments: PbWO4 and GAP
      float segmentSizeinX0=segments[iseg].X0length();

      // Martijn - insert your computations here
      float energy=0.0;
      if (segmentSizeinX0>0.001 && segments[iseg].material()==CaloSegment::PbWO4 ) {
	// The energy loss simulator
	float charge = (float)(myTrack.charge());
	ParticlePropagator theMuon(moment,trackPosition,charge,0);
	theMuon.setID(-(int)charge*13);
	if ( energyLossECAL ) { 
	  energyLossECAL->updateState(theMuon, segmentSizeinX0, random);
	  energy = energyLossECAL->deltaMom().E();
	  moment -= energyLossECAL->deltaMom();
	}
      } 
      // that's all for ECAL, Florian
      // Save the hit only if it is a crystal
      if(segments[iseg].material()==CaloSegment::PbWO4)
	{
	  myGrid.getPads(segments[iseg].sX0Entrance()+segmentSizeinX0*0.5);
	  myGrid.setSpotEnergy(energy);
	  myGrid.addHit(0.,0.);
	  ilastEcal=iseg;
	}
      // Check for end of loop:
      if(segments[iseg].material()==CaloSegment::HCAL)
	{
	  ifirstHcal=iseg;
	}
    }
  

  // Build the FAMOS HCAL 
  HcalHitMaker myHcalHitMaker(myGrid,(unsigned)2);     
  // float mipenergy=0.1;
  // Create the helix with the stepping helix propagator
  // to add a hit, just do
  // myHcalHitMaker.setSpotEnergy(mipenergy);
  // math::XYZVector hcalEntrance;
  // if(ifirstHcal>=0) hcalEntrance=segments[ifirstHcal].entrance();
  // myHcalHitMaker.addHit(hcalEntrance);
  ///
  /////
  ////// TEMPORARY First attempt to include HCAL (with straight-line extrapolation): 
  int ilastHcal=-1;
  float mipenergy=0.0;
 
  EnergyLossSimulator* energyLossHCAL = 0;
  if (theMuonHcalEffects) energyLossHCAL = theMuonHcalEffects->energyLossSimulator();
  //  // Muon Brem effect
  //  MuonBremsstrahlungSimulator* muonBremHCAL = 0;
  //  if (theMuonHcalEffects) muonBremHCAL = theMuonHcalEffects->muonBremsstrahlungSimulator(); 
 
  if(ifirstHcal>0 && energyLossHCAL){
    for(unsigned iseg=ifirstHcal;iseg<nsegments;++iseg)
      {
	float segmentSizeinX0=segments[iseg].X0length();
	if(segments[iseg].material()==CaloSegment::HCAL) {
	  ilastHcal=iseg;
	  if (segmentSizeinX0>0.001) {
	    // The energy loss simulator
	    float charge = (float)(myTrack.charge());
	    ParticlePropagator theMuon(moment,trackPosition,charge,0);
	    theMuon.setID(-(int)charge*13);
	    energyLossHCAL->updateState(theMuon, segmentSizeinX0, random);
	    mipenergy = energyLossHCAL->deltaMom().E();
	    moment -= energyLossHCAL->deltaMom();
	    myHcalHitMaker.setSpotEnergy(mipenergy);
	    myHcalHitMaker.addHit(segments[iseg].entrance());
	  }
	} 
      }
  }
  
  //////
  /////
  ////
  ///
  //



  // Copy the muon SimTrack (Only for Energy loss)
  FSimTrack muonTrack(myTrack);
  if(energyLossHCAL && ilastHcal>=0) {
    math::XYZVector hcalExit=segments[ilastHcal].exit();
    muonTrack.setTkPosition(hcalExit);
    muonTrack.setTkMomentum(moment);
  } else if(energyLossECAL && ilastEcal>=0) {
    math::XYZVector ecalExit=segments[ilastEcal].exit();
    muonTrack.setTkPosition(ecalExit);
    muonTrack.setTkMomentum(moment);
  } // else just leave tracker surface position and momentum...  

  muonSimTracks.push_back(muonTrack);


  // no need to change below this line
  std::map<CaloHitID,float>::const_iterator mapitr;
  std::map<CaloHitID,float>::const_iterator endmapitr;
  if(myTrack.onEcal() > 0) {
	// Save ECAL hits
	updateECAL(myGrid.getHits(),onECAL,myTrack.id());
  }

  // Save HCAL hits
  updateHCAL(myHcalHitMaker.getHits(),myTrack.id());
  
  if(debug_)
    LogInfo("FastCalorimetry") << std::endl << " FASTEnergyReconstructor::MipShowerSimulation  finished "
	 << std::endl;
}


void CalorimetryManager::readParameters(const edm::ParameterSet& fastCalo) {

  edm::ParameterSet ECALparameters = fastCalo.getParameter<edm::ParameterSet>("ECAL");

  evtsToDebug_ = fastCalo.getUntrackedParameter<std::vector<unsigned int> >("EvtsToDebug",std::vector<unsigned>());
  debug_ = fastCalo.getUntrackedParameter<bool>("Debug");

  bFixedLength_ = ECALparameters.getParameter<bool>("bFixedLength");
  //   std::cout << "bFixedLength_ = " << bFixedLength_ << std::endl;

  gridSize_ = ECALparameters.getParameter<int>("GridSize");
  spotFraction_ = ECALparameters.getParameter<double>("SpotFraction");
  pulledPadSurvivalProbability_ = ECALparameters.getParameter<double>("FrontLeakageProbability");
  crackPadSurvivalProbability_ = ECALparameters.getParameter<double>("GapLossProbability");
  theCoreIntervals_ = ECALparameters.getParameter<std::vector<double> >("CoreIntervals");
  theTailIntervals_ = ECALparameters.getParameter<std::vector<double> >("TailIntervals");
  
  RCFactor_ = ECALparameters.getParameter<double>("RCFactor");
  RTFactor_ = ECALparameters.getParameter<double>("RTFactor");
  //changed after tuning - Feb-July - Shilpi Jain
  //  radiusFactor_ = ECALparameters.getParameter<double>("RadiusFactor");
  radiusFactorEE_ = ECALparameters.getParameter<double>("RadiusFactorEE");
  radiusFactorEB_ = ECALparameters.getParameter<double>("RadiusFactorEB");
  //(end of) changed after tuning - Feb-July - Shilpi Jain
  radiusPreshowerCorrections_ = ECALparameters.getParameter<std::vector<double> >("RadiusPreshowerCorrections");
  aTerm = 1.+radiusPreshowerCorrections_[1]*radiusPreshowerCorrections_[0];
  bTerm = radiusPreshowerCorrections_[0];
  mipValues_ = ECALparameters.getParameter<std::vector<double> >("MipsinGeV");
  simulatePreshower_ = ECALparameters.getParameter<bool>("SimulatePreshower");

  if(gridSize_ <1) gridSize_= 7;
  if(pulledPadSurvivalProbability_ <0. || pulledPadSurvivalProbability_>1 ) pulledPadSurvivalProbability_= 1.;
  if(crackPadSurvivalProbability_ <0. || crackPadSurvivalProbability_>1 ) crackPadSurvivalProbability_= 0.9;
  
  LogInfo("FastCalorimetry") << " Fast ECAL simulation parameters " << std::endl;
  LogInfo("FastCalorimetry") << " =============================== " << std::endl;
  if(simulatePreshower_)
    LogInfo("FastCalorimetry") << " The preshower is present " << std::endl;
  else
    LogInfo("FastCalorimetry") << " The preshower is NOT present " << std::endl;
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
  //changed after tuning - Feb-July - Shilpi Jain
      //      LogInfo("FastCalorimetry") << "Radius correction factor " << radiusFactor_ << std::endl;
      LogInfo("FastCalorimetry") << "Radius correction factors:  EB & EE " << radiusFactorEB_ << " : "<< radiusFactorEE_ << std::endl;
  //(end of) changed after tuning - Feb-July - Shilpi Jain
      LogInfo("FastCalorimetry") << std::endl;
      if(mipValues_.size()>2) 	{
	LogInfo("FastCalorimetry") << "Improper number of parameters for the preshower ; using 95keV" << std::endl;
	mipValues_.clear();
	mipValues_.resize(2,0.000095);
	}
    }

  LogInfo("FastCalorimetry") << " FrontLeakageProbability : " << pulledPadSurvivalProbability_ << std::endl;
  LogInfo("FastCalorimetry") << " GapLossProbability : " << crackPadSurvivalProbability_ << std::endl;

  
  // RespCorrP: p (momentum), ECAL and HCAL corrections = f(p)
  edm::ParameterSet CalorimeterParam = fastCalo.getParameter<edm::ParameterSet>("CalorimeterProperties");

  rsp = CalorimeterParam.getParameter<std::vector<double> >("RespCorrP");
   LogInfo("FastCalorimetry") << " RespCorrP (rsp) size " << rsp.size() << std::endl;

  if( rsp.size()%3 !=0 )  {
    LogInfo("FastCalorimetry") 
      << " RespCorrP size is wrong -> no corrections applied !!!" 
      << std::endl;

      p_knots.push_back(14000.);
      k_e.push_back    (1.);
      k_h.push_back    (1.);
  }
  else {
    for(unsigned i = 0; i < rsp.size(); i += 3) { 
     LogInfo("FastCalorimetry") << "i = " << i/3 << "   p = " << rsp [i] 
				<< "   k_e(p) = " << rsp[i+1] 
				<< "   k_e(p) = " << rsp[i+2] << std::endl; 
      
      p_knots.push_back(rsp[i]);
      k_e.push_back    (rsp[i+1]);
      k_h.push_back    (rsp[i+2]); 
    }
  }  
 

  //FR
  edm::ParameterSet HCALparameters = fastCalo.getParameter<edm::ParameterSet>("HCAL");
  optionHDSim_ = HCALparameters.getParameter<int>("SimOption");
  hdGridSize_  = HCALparameters.getParameter<int>("GridSize");
  hdSimMethod_ = HCALparameters.getParameter<int>("SimMethod");
  //RF

  EcalDigitizer_    = ECALparameters.getUntrackedParameter<bool>("Digitizer",false);
  HcalDigitizer_    = HCALparameters.getUntrackedParameter<bool>("Digitizer",false);
  samplingHBHE_ = HCALparameters.getParameter< std::vector<double> >("samplingHBHE");
  samplingHF_   = HCALparameters.getParameter< std::vector<double> >("samplingHF");
  samplingHO_   = HCALparameters.getParameter< std::vector<double> >("samplingHO");
  ietaShiftHB_  = HCALparameters.getParameter< int >("ietaShiftHB");
  ietaShiftHE_  = HCALparameters.getParameter< int >("ietaShiftHE");
  ietaShiftHF_  = HCALparameters.getParameter< int >("ietaShiftHF");
  ietaShiftHO_  = HCALparameters.getParameter< int >("ietaShiftHO");
  timeShiftHB_  = HCALparameters.getParameter< std::vector<double> >("timeShiftHB");
  timeShiftHE_  = HCALparameters.getParameter< std::vector<double> >("timeShiftHE");
  timeShiftHF_  = HCALparameters.getParameter< std::vector<double> >("timeShiftHF");
  timeShiftHO_  = HCALparameters.getParameter< std::vector<double> >("timeShiftHO");

  // FastHFShowerLibrary
  edm::ParameterSet m_HS = fastCalo.getParameter<edm::ParameterSet>("HFShowerLibrary");
  useShowerLibrary       = m_HS.getUntrackedParameter<bool>("useShowerLibrary",false);
  useCorrectionSL        = m_HS.getUntrackedParameter<bool>("useCorrectionSL",false);
}

void CalorimetryManager::respCorr(double p) {

  int sizeP = p_knots.size();

  if(sizeP <= 1) {
    ecorr = 1.;
    hcorr = 1.;
  }
  else {
    int ip = -1;    
    for (int i = 0; i < sizeP; i++) { 
      if (p < p_knots[i]) { ip = i; break;}
    }
    if (ip == 0) {
      ecorr = k_e[0];
      hcorr = k_h[0];
    }
    else {
      if(ip == -1) {
	ecorr = k_e[sizeP-1];
	hcorr = k_h[sizeP-1];
      } 
      else {
	double x1 =  p_knots[ip-1];
	double x2 =  p_knots[ip];
	double y1 =  k_e[ip-1];
	double y2 =  k_e[ip];
	
	if(x1 == x2) {
	  //        std::cout << " equal p_knots values!!! " << std::endl;
	}	
      
	ecorr = (y1 + (y2 - y1) * (p - x1)/(x2 - x1));
	
	y1 =  k_h[ip-1];
	y2 =  k_h[ip];
	hcorr = (y1 + (y2 - y1) * (p - x1)/(x2 - x1)); 
	
      }
    }
  }

  if(debug_)
    LogInfo("FastCalorimetry") << " p, ecorr, hcorr = " << p << " "  
			        << ecorr << "  " << hcorr << std::endl;
	
}

void CalorimetryManager::updateECAL(const std::map<CaloHitID,float>& hitMap, int onEcal, int trackID, float corr)
{
  std::map<CaloHitID,float>::const_iterator mapitr;
  std::map<CaloHitID,float>::const_iterator endmapitr=hitMap.end();
  if(onEcal==1) {
    EBMapping_.reserve(EBMapping_.size()+hitMap.size());
	endmapitr=hitMap.end();
    for(mapitr=hitMap.begin();mapitr!=endmapitr;++mapitr) {
	  //correct energy
	  float energy = mapitr->second;
	  energy *= corr;
	  
	  //make finalized CaloHitID
	  CaloHitID current_id(mapitr->first.unitID(),mapitr->first.timeSlice(),trackID);
	  
      EBMapping_.push_back(std::pair<CaloHitID,float>(current_id,energy));
    }
  }
  else if(onEcal==2) {
    EEMapping_.reserve(EEMapping_.size()+hitMap.size());
	endmapitr=hitMap.end();
    for(mapitr=hitMap.begin();mapitr!=endmapitr;++mapitr) {
      //correct energy
	  float energy = mapitr->second;
	  energy *= corr;
	  
	  //make finalized CaloHitID
	  CaloHitID current_id(mapitr->first.unitID(),mapitr->first.timeSlice(),trackID);
	  
      EEMapping_.push_back(std::pair<CaloHitID,float>(current_id,energy));
    }
  }

}

void CalorimetryManager::updateHCAL(const std::map<CaloHitID,float>& hitMap, int trackID, float corr)
{
  std::vector<double> hfcorr = myHDResponse_->getCorrHF();
  std::map<CaloHitID,float>::const_iterator mapitr;
  std::map<CaloHitID,float>::const_iterator endmapitr=hitMap.end();
  HMapping_.reserve(HMapping_.size()+hitMap.size());
  for(mapitr=hitMap.begin(); mapitr!=endmapitr; ++mapitr) {
    //correct energy
	float energy = mapitr->second;
    energy *= corr;

	float time = mapitr->first.timeSlice();
	//put energy into uncalibrated state for digitizer && correct timing
	if(HcalDigitizer_){
	  HcalDetId hdetid = HcalDetId(mapitr->first.unitID());
	  if (hdetid.subdetId()== HcalBarrel){
            energy /= samplingHBHE_[hdetid.ietaAbs()-1]; //re-convert to GeV
		time = timeShiftHB_[hdetid.ietaAbs()-ietaShiftHB_];
          }
	  else if (hdetid.subdetId()== HcalEndcap){
	    energy /= samplingHBHE_[hdetid.ietaAbs()-1]; //re-convert to GeV
		time = timeShiftHE_[hdetid.ietaAbs()-ietaShiftHE_];
	  }
	  else if (hdetid.subdetId()== HcalForward){
            if(useShowerLibrary) {
              if(useCorrectionSL) energy *= hfcorr[hdetid.ietaAbs()-29];
            } else {
 	      if(hdetid.depth()== 1) energy *= samplingHF_[0];
	      if(hdetid.depth()== 2) energy *= samplingHF_[1];
            } 
		time = timeShiftHF_[hdetid.ietaAbs()-ietaShiftHF_];
	  }
	  else if (hdetid.subdetId()== HcalOuter){
            energy /= samplingHO_[hdetid.ietaAbs()-1];
		time = timeShiftHO_[hdetid.ietaAbs()-ietaShiftHO_];
	  }
	}	
	
    //make finalized CaloHitID
	CaloHitID current_id(mapitr->first.unitID(),time,trackID);
    HMapping_.push_back(std::pair<CaloHitID,float>(current_id,energy));
  }
}

void CalorimetryManager::updatePreshower(const std::map<CaloHitID,float>& hitMap, int trackID, float corr)
{
  std::map<CaloHitID,float>::const_iterator mapitr;
  std::map<CaloHitID,float>::const_iterator endmapitr=hitMap.end();
  ESMapping_.reserve(ESMapping_.size()+hitMap.size());
  for(mapitr=hitMap.begin();mapitr!=endmapitr;++mapitr) {
    //correct energy
	float energy = mapitr->second;
	energy *= corr;
	
	//make finalized CaloHitID
	CaloHitID current_id(mapitr->first.unitID(),mapitr->first.timeSlice(),trackID);
	
	ESMapping_.push_back(std::pair<CaloHitID,float>(current_id,energy));
  }
}

void CalorimetryManager::loadFromEcalBarrel(edm::PCaloHitContainer & c) const
{ 
  c.reserve(c.size()+EBMapping_.size());
  for(unsigned i=0; i<EBMapping_.size(); i++) {
    c.push_back(PCaloHit(EBDetId::unhashIndex(EBMapping_[i].first.unitID()),EBMapping_[i].second,EBMapping_[i].first.timeSlice(),EBMapping_[i].first.trackID()));
  }
}

void CalorimetryManager::loadFromEcalEndcap(edm::PCaloHitContainer & c) const
{
  c.reserve(c.size()+EEMapping_.size());
  for(unsigned i=0; i<EEMapping_.size(); i++) {
    c.push_back(PCaloHit(EEDetId::unhashIndex(EEMapping_[i].first.unitID()),EEMapping_[i].second,EEMapping_[i].first.timeSlice(),EEMapping_[i].first.trackID()));
  }
}

void CalorimetryManager::loadFromHcal(edm::PCaloHitContainer & c) const
{
  c.reserve(c.size()+HMapping_.size());
  for(unsigned i=0; i<HMapping_.size(); i++) {
    c.push_back(PCaloHit(DetId(HMapping_[i].first.unitID()),HMapping_[i].second,HMapping_[i].first.timeSlice(),HMapping_[i].first.trackID()));
  }
}


void CalorimetryManager::loadFromPreshower(edm::PCaloHitContainer & c) const
{
  c.reserve(c.size()+ESMapping_.size());
  for(unsigned i=0; i<ESMapping_.size(); i++) {
    c.push_back(PCaloHit(ESMapping_[i].first.unitID(),ESMapping_[i].second,ESMapping_[i].first.timeSlice(),ESMapping_[i].first.trackID()));
  }
}

// The main danger in this method is to screw up to relationships between particles
// So, the muon FSimTracks created by FSimEvent.cc are simply to be updated 
void CalorimetryManager::loadMuonSimTracks(edm::SimTrackContainer &muons) const
{
  unsigned size=muons.size();
  for(unsigned i=0; i<size;++i)
    {
      int id=muons[i].trackId();
      if(abs(muons[i].type())!=13) continue;
      // identify the corresponding muon in the local collection

      std::vector<FSimTrack>::const_iterator itcheck=find_if(muonSimTracks.begin(),muonSimTracks.end(),FSimTrackEqual(id));
      if(itcheck!=muonSimTracks.end())
	{
	  muons[i].setTkPosition(itcheck->trackerSurfacePosition());
	  muons[i].setTkMomentum(itcheck->trackerSurfaceMomentum());
//	  std::cout << " Found the SimTrack " << std::endl;
//	  std::cout << *itcheck << std::endl;
//	  std::cout << "SimTrack Id "<< id << " " << muons[i] << " " << std::endl;
	}
//      else
//	{
//	  std::cout << " Calorimetery Manager : this should really not happen " << std::endl;
//	  std::cout << " Was looking for " << id << " " << muons[i] << std::endl;
//	  for(unsigned i=0;i<muonSimTracks.size();++i)
//	    std::cout << muonSimTracks[i] << std::endl;
//	}
    }

}

