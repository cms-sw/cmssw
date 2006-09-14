// Fast Simulation headers
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
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

//CMSSW headers 
#include "DataFormats/DetId/interface/DetId.h"

CalorimetryManager::CalorimetryManager(FSimEvent * aSimEvent, const edm::ParameterSet& fastCalo)
  :mySimEvent(aSimEvent)
{
  readParameters(fastCalo);
  myCalorimeter_ = new Calorimeter(fastCalo);
  myHistos = Histos::instance();
  myHistos->book("h10",100,90,110);
  myHistos->book("h20",100,90,110);
}

CalorimetryManager::~CalorimetryManager()
{
  myHistos->put("histos.root");
  delete myHistos;
  if(myCalorimeter_) delete myCalorimeter_;
}

void CalorimetryManager::reconstruct()
{
  // Clear the content of the calorimeters 
  EBMapping_.clear();
  EEMapping_.clear();
  ESMapping_.clear();
  HMapping_.clear();

  for( int fsimi=0; fsimi < (int) mySimEvent->nTracks() ; ++fsimi) {

    FSimTrack& myTrack = mySimEvent->track(fsimi);

    int pid = abs(myTrack.type());

    if (debug_) {
      std::cout << " ===> FASTEnergyReconstructor::reconstruct -  pid = "
		<< pid << std::endl;      
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
    } // myTrack.noEndVertex()    
  } // particle loop
  std::cout << " Number of  hits (barrel)" << EBMapping_.size() << std::endl;
  std::cout << " Number of  hits (Hcal)" << HMapping_.size() << std::endl;
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


  double depth((X0depth+theShower.getMeanDepth())*myCalorimeter_->ecalProperties(onEcal)->radLenIncm());
  //  std::cout << " Depth in cm "  << depth << std::endl;
  HepPoint3D meanShower=ecalentrance+myPart.vect().unit()*depth;

  
  if(onEcal!=1) return ; 

  // The closest crystal
  //  std::cout << " Before getClosestCell " << myCalorimeter_ <<std::endl;
  DetId pivot(myCalorimeter_->getClosestCell(meanShower, true, onEcal==1));

  //  std::cout << " After getClosestCell " << std::endl;
  
  EcalHitMaker myGrid(myCalorimeter_,ecalentrance,pivot,onEcal,size,0);
  //                                             ^^^^
  //                                         for EM showers
//  myGrid.setPulledPadSurvivalProbability(pulledPadSurvivalProbability);
//  myGrid.setCrackPadSurvivalProbability(crackPadSurvivalProbability);

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
  //  myHistos->fill("h502", myPart->eta(),myGrid.totalX0());
  
  //  std::cout << " Save the hits " << std::endl;
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
  if(myPreshower==NULL) return;
  // delete the preshower
  endmapitr=myPreshower->getHits().end();
  for(mapitr=myPreshower->getHits().begin();mapitr!=endmapitr;++mapitr)
    {
      updateMap(mapitr->first,mapitr->second,ESMapping_);
      //      std::cout << " Adding " <<mapitr->first << " " << mapitr->second <<std::endl; 
    }
  //  std::cout << " Deleting myPreshower " << std::endl;
  delete myPreshower;
}



// Simulation of electromagnetic showers in VFCAL
void CalorimetryManager::reconstructECAL(const FSimTrack& track) {
  ;
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


  if(gridSize_ <1) gridSize_= 7;
  if(pulledPadSurvivalProbability_ <0. || pulledPadSurvivalProbability_>1 ) pulledPadSurvivalProbability_= 1.;
  if(crackPadSurvivalProbability_ <0. || crackPadSurvivalProbability_>1 ) crackPadSurvivalProbability_= 0.9;
  
  std::cout << " Fast ECAL simulation parameters " << std::endl;
  std::cout << " =============================== " << std::endl;
  std::cout << " Grid Size : " << gridSize_  << std::endl; 
  if(spotFraction_>0.) 
    std::cout << " Spot Fraction : " << spotFraction_ << std::endl;
  else
    {
      std::cout << " Core of the shower " << std::endl;
      for(unsigned ir=0; ir < theCoreIntervals_.size()/2;++ir)
	{
	  std::cout << " r < " << theCoreIntervals_[ir*2] << " R_M : " << theCoreIntervals_[ir*2+1] << "        ";
	}
      std::cout << std::endl;
	
      std::cout << " Tail of the shower " << std::endl;
      for(unsigned ir=0; ir < theTailIntervals_.size()/2;++ir)
	{
	  std::cout << " r < " << theTailIntervals_[ir*2] << " R_M : " << theTailIntervals_[ir*2+1] << "        ";
	}
      std::cout << std::endl;
    }

  std::cout << " FrontLeakageProbability : " << pulledPadSurvivalProbability_ << std::endl;
  std::cout << " GapLossProbability : " << crackPadSurvivalProbability_ << std::endl;

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
      c.push_back(PCaloHit(cellit->first,cellit->second,0.,0));
    }
}
