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
// STL headers 
#include <vector>
#include <iostream>

//DQM
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
  random(0),initialized_(false)
{;}

CalorimetryManager::CalorimetryManager(FSimEvent * aSimEvent, 
				       const edm::ParameterSet& fastCalo,
				       const edm::ParameterSet& fastMuECAL,
				       const edm::ParameterSet& fastMuHCAL,
                                       const edm::ParameterSet& parGflash,
				       const RandomEngine* engine)
  : 
  mySimEvent(aSimEvent), 
  random(engine), initialized_(false),
  theMuonEcalEffects(0), theMuonHcalEffects (0), bFixedLength_(false)

{

  aLandauGenerator = new LandauFluctuationGenerator(random);
  aGammaGenerator = new GammaFunctionGenerator(random);

  //Gflash
  theProfile = new GflashHadronShowerProfile(parGflash);
  thePiKProfile = new GflashPiKShowerProfile(parGflash);
  theProtonProfile = new GflashProtonShowerProfile(parGflash);
  theAntiProtonProfile = new GflashAntiProtonShowerProfile(parGflash);

  readParameters(fastCalo);

//  EBMapping_.resize(62000,myZero_);
//  EEMapping_.resize(20000,myZero_);
//  HMapping_.resize(10000,myZero_);
  EBMapping_.resize(62000);
  EEMapping_.resize(20000);

  unsigned s=(unfoldedMode_)?5:1;
  for(unsigned ic=0;ic<62000;++ic)
    {
      EBMapping_[ic].reserve(s);
      if(ic<20000)
	EEMapping_[ic].reserve(s);
    }

  //  myHistos = 0; 

  dbe = edm::Service<DQMStore>().operator->();

  if (useDQM_){
	TH1::SetDefaultSumw2(true); //turn on histo errors
  
	//ECAL histos
    dbe->setCurrentFolder("EMShower");
     // please keep the binning with fixed width and coherent between ShapeRhoZ and Tr/Lo shapes. Also check if you 
     // change the binning that the weight changes in the filling in EMShower.cc
    dbe->book1D("TransverseShape","Transverse Shape; #rho / Moliere radius; 1/E dE/d#rho",70, 0., 7.);
    dbe->book1D("LongitudinalShape","Longitudinal Shape; z / X0; 1/E dE/dz",40, 0.01, 40.01);
    dbe->book1D("LongitudinalShapeLayers","Longitudinal Shape in number of layers; z / Layers; 1/E dE/dz", 26, 0.01, 26.01);
    dbe->book2D("ShapeRhoZ","2D Shape; #rho / Moliere radius; z / X0", 70, 0., 7., 26, 0.01, 26.01);
    dbe->book1D("NumberOfParticles","Number Of Particles entering the Shower; #Particles; #Events", 6, -0.5, 5.5);
    dbe->book1D("ParticlesEnergy","Log Particles Energy; log10(E / GeV); #Particles", 30, 0, 3);
	
	//HCAL histos
    dbe->setCurrentFolder("HDShower");

    dbe->book1D("TransverseShapeECAL","ECAL Transverse Shape; #rho / #lambda_{int}; 1/E dE/d#rho",70, 0., 7.);
    dbe->book1D("LongitudinalShapeECAL","ECAL Longitudinal Shape; z / #lambda_{int}; 1/E dE/dz",20, 0., 2.);
    dbe->book1D("TransverseShapeHCAL","HCAL Transverse Shape; #rho / #lambda_{int}; 1/E dE/d#rho",70, 0., 7.);
    dbe->book1D("LongitudinalShapeHCAL","HCAL Longitudinal Shape; z / #lambda_{int}; 1/E dE/dz",120, 0., 12.);       
    dbe->book1D("ParticlesEnergy","Log Particles Energy; log10(E / GeV); #Particles", 30, 0, 3);

    dbe->setCurrentFolder("HDEnergies");
    dbe->book1D("EpECAL","ECAL perfect energy; E / Egen; # events",200,0,2);
    dbe->book1D("EsECAL","ECAL smeared energy; E / Egen; # events",200,0,2);
    dbe->book1D("EpHCAL","HCAL perfect energy; E / Egen; # events",200,0,2);
    dbe->book1D("EsHCAL","HCAL smeared energy; E / Egen; # events",200,0,2);
    dbe->book1D("EpTot","Total perfect energy; E / Egen; # events",200,0,2);
    dbe->book1D("EsTot","Total smeared energy; E / Egen; # events",200,0,2);
	
  }

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
    new HCALResponse(fastCalo.getParameter<edm::ParameterSet>("HCALResponse"),
		     random);
  myHSParameters_ = 
    new HSParameters(fastCalo.getParameter<edm::ParameterSet>("HSParameters"));

  // Material Effects for Muons in ECAL (only EnergyLoss implemented so far)

  if ( fastMuECAL.getParameter<bool>("PairProduction") || 
       fastMuECAL.getParameter<bool>("Bremsstrahlung") ||
       fastMuECAL.getParameter<bool>("MuonBremsstrahlung") ||
       fastMuECAL.getParameter<bool>("EnergyLoss") || 
       fastMuECAL.getParameter<bool>("MultipleScattering") )
    theMuonEcalEffects = new MaterialEffects(fastMuECAL,random);

  // Material Effects for Muons in HCAL (only EnergyLoss implemented so far)

  if ( fastMuHCAL.getParameter<bool>("PairProduction") || 
       fastMuHCAL.getParameter<bool>("Bremsstrahlung") ||
       fastMuHCAL.getParameter<bool>("MuonBremsstrahlung") ||
       fastMuHCAL.getParameter<bool>("EnergyLoss") || 
       fastMuHCAL.getParameter<bool>("MultipleScattering") )
    theMuonHcalEffects = new MaterialEffects(fastMuHCAL,random);


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
  
  HMapping_.clear();
  firedCellsHCAL_.clear();

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
}

void CalorimetryManager::reconstruct()
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
	    EMShowerSimulation(myTrack);
	  else if ( myTrack.onVFcal() )
	    reconstructHCAL(myTrack);
	   
      } // electron or photon
      else if (pid==13)
	{
	  MuonMipSimulation(myTrack);
	}
      // Simulate energy smearing for hadrons (i.e., everything 
      // but muons... and SUSY particles that deserve a special 
      // treatment.
      else if ( pid < 1000000 ) {
	if ( myTrack.onHcal() || myTrack.onVFcal() ) { 	  
	  if(optionHDSim_ == 0 )  reconstructHCAL(myTrack);
	  else HDShowerSimulation(myTrack);
	}
      } // pid < 1000000 
    } // myTrack.noEndVertex()
  } // particle loop
  //  LogInfo("FastCalorimetry") << " Number of  hits (barrel)" << EBMapping_.size() << std::endl;
  //  LogInfo("FastCalorimetry") << " Number of  hits (Hcal)" << HMapping_.size() << std::endl;
  //  std::cout << " Nombre de hit (endcap)" << EEMapping_.size() << std::endl;

} // reconstruct

// Simulation of electromagnetic showers in PS, ECAL, HCAL 
void CalorimetryManager::EMShowerSimulation(const FSimTrack& myTrack) {
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
					  aLandauGenerator);
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


  EMShower theShower(random,aGammaGenerator,&showerparam,&thePart, dbe, NULL, NULL, bFixedLength_);


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
      updateMap(HcalDetId(mapitr->first).rawId(),mapitr->second,myTrack.id(),HMapping_);
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
  //  std::cout << " Deleting myPreshower " << std::endl;
    }
  
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
      myHDResponse_->responseHCAL(0, EGen, pathEta, 0); // last par.= 0 = e/gamma 
    e     = response.first;
    sigma = response.second;
  }

  double emeas = 0.;
  if(sigma>0.) emeas = gaussShootNoNegative(e,sigma);

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
	updateMap(hdetid.rawId(),emeas,track.id(),HMapping_);
      }

}


void CalorimetryManager::reconstructHCAL(const FSimTrack& myTrack)
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
  double e     = 0.;
  double sigma = 0.;
  double emeas = 0.;
  //double emeas = -0.0001;
 
  if(pid == 13) { 
    //    std::cout << " We should not be here " << std::endl;
    std::pair<double,double> response =
      myHDResponse_->responseHCAL(0, EGen, pathEta, 2); // 2=muon 
    emeas  = response.first;
    if(debug_)
      LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - MUON !!!" << std::endl;
  }
  else if( pid == 22 || pid == 11)
    {
      
      std::pair<double,double> response =
	myHDResponse_->responseHCAL(0, EGen, pathEta, 0); // last par. = 0 = e/gamma
      e     = response.first;              //
      sigma = response.second;             //
      emeas = gaussShootNoNegative(e,sigma);

      //  cout <<  "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
      if(debug_)
	LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - e/gamma !!!" << std::endl;
    }
    else {
      e     = myHDResponse_->getHCALEnergyResponse(EGen,hit);
      sigma = myHDResponse_->getHCALEnergyResolution(EGen, hit);
      
      emeas = gaussShootNoNegative(e,sigma);
    }
    

  if(debug_)
    LogInfo("FastCalorimetry") << "CalorimetryManager::reconstructHCAL - on-calo "   
				<< "  eta = " << pathEta 
				<< "  phi = " << pathPhi 
				<< "  Egen = " << EGen 
				<< "  Eres = " << e 
				<< "  sigma = " << sigma 
				<< "  Emeas = " << emeas << std::endl;

  if(emeas > 0.) {  
    DetId cell = myCalorimeter_->getClosestCell(trackPosition.Vect(),false,false);
    updateMap(cell.rawId(), emeas, myTrack.id(),HMapping_);
  }
}

void CalorimetryManager::HDShowerSimulation(const FSimTrack& myTrack){//, 
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
  double e     = 0.;
  double sigma = 0.;

  double emeas = 0.;  
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
      //		<< myTrack.vfcalEntrance().vertex().X() << " "
      //		<< myTrack.vfcalEntrance().vertex().Y() << " "
      //		<< myTrack.vfcalEntrance().vertex().Z() << " "
      //		<< " , Energy (Gen/Scale) = " << eGen << " " << e << std::endl;

      // Warning : We give here the particle energy with the response
      //           but without the resolution/gaussian smearing
      //           For HF, the resolution is due to the PE statistic

      HFShower theShower(random,
			 &theHDShowerparam,
			 &myGrid,
			 &myHcalHitMaker,
			 onECAL,
			 eGen);
			 //			 eGen);
			 //			 e); // PV Warning : temporarly set the energy to the generated E

      status = theShower.compute();
    } else { 
      if(hdSimMethod_ == 0) {
	HDShower theShower(random,
			   &theHDShowerparam,
			   &myGrid,
			   &myHcalHitMaker,
			   onECAL,
			   eGen,
			   dbe);
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
	e     = myHDResponse_->getHCALEnergyResponse  (eGen, hit);
	sigma = myHDResponse_->getHCALEnergyResolution(eGen, hit);
      }
      else { // optionHDsim == 2
	std::pair<double,double> response =
	  myHDResponse_->responseHCAL(mip, eGen, pathEta, 1); // 1=hadron
	e     = response.first;
	sigma = response.second;
      }
      
      emeas = gaussShootNoNegative(e,sigma);
      double correction = emeas / eGen;
      
      // RespCorrP factors (ECAL and HCAL separately) calculation
      respCorr(eint);     

      if(debug_)
	LogInfo("FastCalorimetry") 
	  << "CalorimetryManager::HDShowerSimulation - on-calo 2" << std::endl
	  << "   eta  = " << pathEta << std::endl
	  << "   phi  = " << pathPhi << std::endl
	  << "  Egen  = " << eGen << std::endl
	  << "  Eres  = " << e << std::endl
	  << " sigma  = " << sigma << std::endl
	  << " Emeas  = " << emeas << std::endl
	  << "  corr  = " << correction << std::endl
	  << "   mip  = " << mip << std::endl;
      
      //perfect and smeared energy vars
      double EpE, EsE, EpH, EsH;
      EpE = EsE = EpH = EsH = 0;	  
      
      // was map<unsigned,double> but CaloHitMaker uses float
      std::map<unsigned,float>::const_iterator mapitr;
      std::map<unsigned,float>::const_iterator endmapitr;
      if(myTrack.onEcal() > 0) {
	// Save ECAL hits 
	endmapitr=myGrid.getHits().end();
	for(mapitr=myGrid.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	  double energy = mapitr->second;
	  EpE += energy;
          energy *= correction;              // RESCALING 
          energy *= ecorr;
	  EsE += energy;

	  if(energy > 0.000001) { 
	    if(onECAL==1)
		updateMap(EBDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EBMapping_,firedCellsEB_);

	    else if(onECAL==2)
	      updateMap(EEDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EEMapping_,firedCellsEE_);

	    if(debug_)
	      LogInfo("FastCalorimetry") << " ECAL cell " << mapitr->first << " added,  E = " 
		   << energy << std::endl;  
	  }
	}
      }
      
      // Save HCAL hits
      endmapitr=myHcalHitMaker.getHits().end();
      for(mapitr=myHcalHitMaker.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
	double energy = mapitr->second;
	EpH += energy;
	energy *= correction;               // RESCALING 
	energy *= hcorr;
	
	if(HcalDigitizer_){
	  HcalDetId hdetid = HcalDetId(mapitr->first);
	  if (hdetid.subdetId()== HcalBarrel || hdetid.subdetId()== HcalEndcap ) energy /= samplingHBHE_[hdetid.ietaAbs()-1]; //re-convert to GeV
	  
	  else if (hdetid.subdetId()== HcalForward){
	    if(hdetid.depth()== 1) energy /= samplingHF_[0];
	    if(hdetid.depth()== 2) energy /= samplingHF_[1];
	  }  
	  
	  else if (hdetid.subdetId()== HcalOuter  )  energy /= samplingHO_[hdetid.ietaAbs()-1]; 	
	}
	
	EsH += energy;

	updateMap(HcalDetId(mapitr->first).rawId(),energy,myTrack.id(),HMapping_);
	if(debug_)
	  LogInfo("FastCalorimetry") << " HCAL cell "  
	       << mapitr->first << " added    E = " 
	       << mapitr->second << std::endl;  
      }
	  
	  if(useDQM_){
	    //fill energy histos
	    dbe->get("HDEnergies/EpECAL")->Fill(EpE/eGen);
	    dbe->get("HDEnergies/EsECAL")->Fill(EsE/eGen);
	    dbe->get("HDEnergies/EpHCAL")->Fill(EpH/eGen);
	    dbe->get("HDEnergies/EsHCAL")->Fill(EsH/eGen);
	    dbe->get("HDEnergies/EpTot")->Fill((EpE + EpH)/eGen);
	    dbe->get("HDEnergies/EsTot")->Fill((EsE + EsH)/eGen);
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
	  updateMap(cell.rawId(),emeas,myTrack.id(),HMapping_);
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


void CalorimetryManager::MuonMipSimulation(const FSimTrack& myTrack)
{
  //  TimeMe t(" FASTEnergyReconstructor::HDShower");
  XYZTLorentzVector moment = myTrack.momentum();

  // Backward compatibility behaviour
  if(!theMuonHcalEffects) 
    {
      if(myTrack.onHcal() || myTrack.onVFcal() ) 
	reconstructHCAL(myTrack);

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
	  energyLossECAL->updateState(theMuon, segmentSizeinX0);
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
	    energyLossHCAL->updateState(theMuon, segmentSizeinX0);
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
  std::map<unsigned,float>::const_iterator mapitr;
  std::map<unsigned,float>::const_iterator endmapitr;
  if(myTrack.onEcal() > 0) {
    // Save ECAL hits 
    endmapitr=myGrid.getHits().end();
    for(mapitr=myGrid.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
      double energy = mapitr->second;
      if(onECAL==1)
	{
	  updateMap(EBDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EBMapping_,firedCellsEB_);
	}      
      else if(onECAL==2)
	{
	  updateMap(EEDetId(mapitr->first).hashedIndex(),energy,myTrack.id(),EEMapping_,firedCellsEE_);
	}
      
      if(debug_)
	LogInfo("FastCalorimetry") << " ECAL cell " << mapitr->first << " added,  E = " 
				    << energy << std::endl;  
    }
  }
      
  // Save HCAL hits
  endmapitr=myHcalHitMaker.getHits().end();
  for(mapitr=myHcalHitMaker.getHits().begin(); mapitr!=endmapitr; ++mapitr) {
    double energy = mapitr->second;
    {
      updateMap(HcalDetId(mapitr->first).rawId(),energy,myTrack.id(),HMapping_);
    }
    if(debug_)
      LogInfo("FastCalorimetry") << " HCAL cell "  
				  << mapitr->first << " added    E = " 
				  << mapitr->second << std::endl;  
  }
  
  if(debug_)
    LogInfo("FastCalorimetry") << std::endl << " FASTEnergyReconstructor::MipShowerSimulation  finished "
	 << std::endl;
}


void CalorimetryManager::readParameters(const edm::ParameterSet& fastCalo) {

  edm::ParameterSet ECALparameters = fastCalo.getParameter<edm::ParameterSet>("ECAL");

  evtsToDebug_ = fastCalo.getUntrackedParameter<std::vector<unsigned int> >("EvtsToDebug",std::vector<unsigned>());
  debug_ = fastCalo.getUntrackedParameter<bool>("Debug");
  useDQM_ = fastCalo.getUntrackedParameter<bool>("useDQM");

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

  unfoldedMode_ = fastCalo.getUntrackedParameter<bool>("UnfoldedMode",false);

  EcalDigitizer_    = ECALparameters.getUntrackedParameter<bool>("Digitizer",false);
  HcalDigitizer_    = HCALparameters.getUntrackedParameter<bool>("Digitizer",false);
  samplingHBHE_ = HCALparameters.getParameter< std::vector<double> >("samplingHBHE");
  samplingHF_   = HCALparameters.getParameter< std::vector<double> >("samplingHF");
  samplingHO_   = HCALparameters.getParameter< std::vector<double> >("samplingHO");
  smearTimeHF_  = HCALparameters.getUntrackedParameter<bool>("smearTimeHF",false);
  timeShiftHF_  = HCALparameters.getUntrackedParameter<double>("timeShiftHF",17.);
  timeSmearingHF_  = HCALparameters.getUntrackedParameter<double>("timeSmearingHF",2.);
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


void CalorimetryManager::loadFromEcalBarrel(edm::PCaloHitContainer & c) const
{ 
  unsigned size = firedCellsEB_.size();
  double time = 0.;
  for(unsigned ic=0;ic<size;++ic){ 
    int hi = firedCellsEB_[ic];
    if(!unfoldedMode_){
      if(EcalDigitizer_) time = (myCalorimeter_->getEcalGeometry(1)->getGeometry(EBDetId::unhashIndex(hi))->getPosition().mag())/29.98;//speed of light
      else time = 0.;
      c.push_back(PCaloHit(EBDetId::unhashIndex(hi),EBMapping_[hi][0].second,time,0));
      //	  std::cout << "Adding " << hi << " " << EBDetId::unhashIndex(hi) << " " ;
      //	  std::cout << EBMapping_[hi][0].second << " " << EBMapping_[hi][0].first << std::endl;
    }
    else{
      unsigned npart=EBMapping_[hi].size();
      for(unsigned ip=0;ip<npart;++ip){
	// get the time
	if(EcalDigitizer_) time = (myCalorimeter_->getEcalGeometry(1)->getGeometry(EBDetId::unhashIndex(hi))->getPosition().mag())/29.98;//speed of light
	else time = 0.;
	//	      std::cout << " Barrel " << time  << std::endl;
	c.push_back(PCaloHit(EBDetId::unhashIndex(hi),EBMapping_[hi][ip].second,time,EBMapping_[hi][ip].first));
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
  double time;
  
  for(unsigned ic=0;ic<size;++ic){
    int hi=firedCellsEE_[ic];
    if(!unfoldedMode_) {
      if(EcalDigitizer_) time=(myCalorimeter_->getEcalGeometry(2)->getGeometry(EEDetId::unhashIndex(hi))->getPosition().mag())/29.98;//speed of light
      else time = 0.;
      c.push_back(PCaloHit(EEDetId::unhashIndex(hi),EEMapping_[hi][0].second,time,0));
    }
    else{
      unsigned npart=EEMapping_[hi].size();
      for(unsigned ip=0;ip<npart;++ip) {
	if(EcalDigitizer_) time=(myCalorimeter_->getEcalGeometry(2)->getGeometry(EEDetId::unhashIndex(hi))->getPosition().mag())/29.98;//speed of light
	else time = 0.;	   
	c.push_back(PCaloHit(EEDetId::unhashIndex(hi),EEMapping_[hi][ip].second,time,EEMapping_[hi][ip].first));
      }
    }
    
    //      sum+=cellit->second;
  }
  //  std::cout << " SUM : " << sum << std::endl;
  //  std::cout << " Added " <<c.size() << " hits " <<std::endl;
}

void CalorimetryManager::loadFromHcal(edm::PCaloHitContainer & c) const
{

  double time;
  std::map<uint32_t,std::vector<std::pair< int,float> > >::const_iterator cellit;
  for (cellit=HMapping_.begin(); cellit!=HMapping_.end(); cellit++) {
    DetId hi=DetId(cellit->first);
    if(!unfoldedMode_){
      if(HcalDigitizer_) {
	time=(myCalorimeter_->getHcalGeometry()->getGeometry(hi)->getPosition().mag())/29.98;//speed of light
	if (smearTimeHF_ && hi.subdetId()== HcalForward){ // shift and smearing for HF
	  time = random->gaussShoot((time+timeShiftHF_),timeSmearingHF_);
	}
      } else time = 0.;
      c.push_back(PCaloHit(hi,cellit->second[0].second,time,0));
    }
    else{
      unsigned npart=cellit->second.size();
      for(unsigned ip=0;ip<npart;++ip){
	if(HcalDigitizer_) {
	  time=(myCalorimeter_->getHcalGeometry()->getGeometry(hi)->getPosition().mag())/29.98;//speed of light
	  if (smearTimeHF_ && hi.subdetId()== HcalForward){ // shift and smearing for HF
	    time = random->gaussShoot((time+timeShiftHF_),timeSmearingHF_);
	  }
	} else time = 0.;
	c.push_back(PCaloHit(hi,cellit->second[ip].second, time, cellit->second[ip].first));
      }
    }
  }
  
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

// Remove (most) hits with negative energies
double CalorimetryManager::gaussShootNoNegative(double e, double sigma) 
{
  double out = -0.0001;
  if (e >= 0.) {
    while (out < 0.) out = random->gaussShoot(e,sigma);
  } else { // give up on re-trying, otherwise too much time can be lost before emeas comes out positive
    out = random->gaussShoot(e,sigma);
  }
  /*
      if (out < 0.) {
	std::cout << "e = " << e << " - sigma = " << sigma << " - emeas < 0 (!)" << std::endl;
      } else {
	std::cout << "e = " << e << " - sigma = " << sigma << " - emeas > 0 " << std::endl;
      }
  */
  return out;
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

