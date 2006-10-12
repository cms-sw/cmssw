// -*- C++ -*-
//
// Package:    ElectronCalibration
// Class:      ElectronCalibration
// 
/**\class ElectronCalibration ElectronCalibration.cc Calibration/EcalCalibAlgos/src/ElectronCalibration.cc

 Description: Perform single electron calibration (tested on TB data only).

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO, Radek Ofierzynski
//         Created:  Tue Jul 18 12:17:01 CEST 2006
// $Id: ElectronCalibration.cc,v 1.1 2006/07/24 10:09:08 lorenzo Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>



// class decleration
//

class ElectronCalibration : public edm::EDAnalyzer {
   public:
      explicit ElectronCalibration(const edm::ParameterSet&);
      ~ElectronCalibration();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
   private:

      EBDetId  findMaxHit(edm::Handle<EBRecHitCollection> &);

      // ----------member data ---------------------------
      std::string rootfile_;
      std::string hitCollection_;
      std::string EBhitCollection;
      std::string digiProducer;
      std::string hitProducer_;
      std::string calibAlgo_;

       MinL3Algorithm algoL3;
       HouseholderDecomposition algoHH;
       CalibrationCluster calibCluster;
       CalibrationCluster::CalibMap ReducedMap;


       vector<int> EventsPerCrystal;
       vector<vector<float> >EventMatrix; 
       vector<float> oldCalibs;
       vector<float> newCalibs;
       vector<float> energyVector;
       vector<float> temp_solution;
       vector<float> solution;

       
       int myMaxHit_save;
       int read_events;
       int used_events;
       int nupdates;
       int checkEnergy;
       int checkOutBoundEnergy;
       unsigned int subsample_;
       unsigned int supermodule_;
       bool makeIteration;
       float BEAM_ENERGY;
       
 
       static const int MIN_IETA = 30;
       static const int MAX_IETA = 40;
       static const int MIN_IPHI = 3;
       static const int MAX_IPHI = 10;

       TH1F* e25;
};


ElectronCalibration::ElectronCalibration(const edm::ParameterSet& iConfig)
{

   rootfile_                  = iConfig.getUntrackedParameter<std::string>("rootfile","ecalSimpleTBanalysis.root");
   hitCollection_             = iConfig.getParameter<std::string>("hitCollection");
   hitProducer_               = iConfig.getParameter<std::string>("hitProducer");
   calibAlgo_       = iConfig.getParameter<std::string>("CALIBRATION_ALGO");
   
   subsample_=iConfig.getUntrackedParameter<unsigned int>("SUBSAMPLE_SIZE");

   std::cout << "*************** DATA FROM CONFIG FILE **************" << std::endl;
   std::cout << "subsample__SIZE = " << subsample_  << std::endl;
   std::cout << "*************** DATA FROM CONFIG FILE **************" << std::endl;
}


ElectronCalibration::~ElectronCalibration()
{
 

}

//========================================================================
void
ElectronCalibration::beginJob(edm::EventSetup const& iSetup) {
//========================================================================

  // Book histograms 
  e25 = new TH1F("e25","E25 energy", 1500, 0., 250.);

  // clear variables

  nupdates=0;
  read_events=0;
  used_events=0;
  checkEnergy=0;
  checkOutBoundEnergy=0;
  
  EventMatrix.clear();
  energyVector.clear();
  oldCalibs.clear();
  newCalibs.clear();

  temp_solution.clear();
  solution.clear();
  makeIteration=false;

//  used for TB -> must change to get electron momentum
  BEAM_ENERGY=120.0;  

// get Map to be calibrated  
  ReducedMap = calibCluster.getMap(MIN_IETA, MAX_IETA, MIN_IPHI, MAX_IPHI);


  EventsPerCrystal.resize(ReducedMap.size(),0);
  oldCalibs.resize(ReducedMap.size(),0.);


  CalibrationCluster::CalibMap::iterator itmap;

// get initial constants out of DB (should be set to 1 using xms file + miscalib Tool)

  edm::ESHandle<EcalIntercalibConstants> pIcal;
  
  int counter=0;
  try {
       
       iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
       std::cout << "Taken EcalIntercalibConstants" << std::endl;
       const EcalIntercalibConstants* ical = pIcal.product();
  
       for (itmap =ReducedMap.begin(); itmap != ReducedMap.end();itmap++){
         
	  int rawId_temp = itmap->first.rawId();
           
	  EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=ical->getMap().find(rawId_temp);
          
	  oldCalibs[counter]=icalit->second;
          	  
	  std::cout << "Read old constant for crystal " << itmap->first.ic() << " (" << itmap->first.ieta() << "," <<
	  itmap->first.iphi() << ") : " << icalit->second << std::endl;

	  counter++;
       }
   
       }   catch ( std::exception& ex ) {
         
	 std::cerr << "Error! can't get EcalIntercalibConstants " << std::endl;
       
       }  
 
 }

//========================================================================
void
ElectronCalibration::endJob() {
//========================================================================

  std::cout << "Total events read in file: " << read_events << std::endl;
  std::cout << "Total n. of steps: " << nupdates << std::endl;
  std::cout << "Final nupdates = " << nupdates << endl;

  if(nupdates!=0) 
    {
 
      CalibrationCluster::CalibMap::iterator itmap;
      int isize=0;
      for (itmap =ReducedMap.begin(); itmap != ReducedMap.end();itmap++){
           solution[isize]/=nupdates;
	   std::cout << "Crystal " << itmap->first.ic() << " Solution[" << isize << "] =" << solution[isize] << std::endl;
	   isize++;
      }

        newCalibs.resize(ReducedMap.size(),0.);

        calibXMLwriter write_calibrations;

        int icry=0;
 
        for (itmap=ReducedMap.begin(); itmap != ReducedMap.end();itmap++){
         
              newCalibs[icry] = oldCalibs[icry]*solution[icry];
 
              write_calibrations.writeLine(itmap->first,newCalibs[icry]);
         
	      icry++;
        }
   

    } //in nupdates=0;

////////////////////////       FINAL STATISTICS           ////////////////////

   std::cout << " " << std::endl;
   std::cout << "************* STATISTICS **************" << std::endl;
   std::cout << "Read Events: " << read_events << std::endl;
   std::cout << "Used Events: " << used_events << std::endl;


/////////////////////////////////////////////////////////////////////////////

  TFile f(rootfile_.c_str(),"RECREATE");

  e25->Write(); 
  f.Close();
 
}


//=================================================================================
EBDetId
ElectronCalibration::findMaxHit(edm::Handle<EBRecHitCollection> &  phits) {
//=================================================================================

     EcalRecHitCollection ecrh = *phits;
     EcalRecHitCollection::iterator it;
     int count=0;
     EBDetId save;
     float en_save=0;
     for (it = ecrh.begin(); it != ecrh.end(); it++)
     {
       EBDetId p = EBDetId(it->id().rawId());
       // std::cout << "Hit list " << p.ieta() << " " << p.iphi() << " " << it->energy() << std::endl;
        if(it->energy()> en_save){
	  en_save=it->energy();
	  save=p;
	  
	}
      count++;
     }
     //return save.ic();
      return save;

}


//=================================================================================
void
ElectronCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//=================================================================================
   using namespace edm;

   Handle<EBRecHitCollection> phits;
   const EBRecHitCollection* hits=0;
   try {
    std::cout << "Taken EBRecHitCollection " << std::endl;
     iEvent.getByLabel( hitProducer_, hitCollection_,phits);
     hits = phits.product(); // get a ptr to the product
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product EBRecHitCollection: " << hitCollection_.c_str() << std::endl;
   }

   if (!hits)
     return;

   if (hits->size() == 0)
     return;

////////////////////////////////////////////////////////////////////////////////////////
///                          START HERE....
///////////////////////////////////////////////////////////////////////////////////////
  read_events++;
  
  if(read_events%subsample_==0) makeIteration=true; 

  makeIteration=false; // do nothing

  //  int myMaxHit = findMaxHit(phits);


// first argument is SM number
//  EBDetId maxHitId(1,myMaxHit,EBDetId::SMCRYSTALMODE); 
  EBDetId maxHitId = findMaxHit(phits); 

  std::cout << "SubDetId = " << maxHitId.subdetId() << std::endl;
  std::cout << "RawId = " << maxHitId.rawId() << std::endl;
  std::cout << "SM number = " << maxHitId.ism() << std::endl;
  std::cout << "calibCluster Eta = " << maxHitId.ieta() << std::endl;
  std::cout << "calibCluster Phi = " << maxHitId.iphi() << std::endl;
  std::cout << "crystal number inside SM = " << maxHitId.ic() << std::endl;
  std::cout << "crystal energy = " << (hits->find(maxHitId))->energy() << std::endl;


// define boundaries of region to be calibrated
  
  if(maxHitId.ieta()<MIN_IETA || maxHitId.ieta()>MAX_IETA) return;
  if(maxHitId.iphi()<MIN_IPHI || maxHitId.iphi()>MAX_IPHI) return;
   
  int icrystal = ReducedMap.find(maxHitId)->second;
  EventsPerCrystal[icrystal]++;


// get cluster
  vector<EBDetId> Xtals5x5 = calibCluster.get5x5Id(maxHitId);
  
// fill cluster energy
   float energy[25];
   float energy3x3=0.;  
   float energy5x5=0.;  

   for (unsigned int icry=0;icry<25;icry++)
     {
       energy[icry]=(hits->find(Xtals5x5[icry]))->energy();
       energy5x5 += energy[icry];
       std::cout << "energy for hit " << icry << " = " << energy[icry] << std::endl;
 
       if ( icry == 6  || icry == 7  || icry == 8 ||
	    icry == 11 || icry == 12 || icry ==13 ||
	    icry == 16 || icry == 17 || icry ==18   )
	 {
	   energy3x3+=energy[icry];
	 }
     }


  float outBoundEnergy=0.;
  int nXtalsOut=-1;


  vector<float> aline25=calibCluster.getEnergyVector(hits,ReducedMap,Xtals5x5, outBoundEnergy, nXtalsOut);

  float resEnergy = BEAM_ENERGY-outBoundEnergy;


// Fill EventMatrix and energyVector
  EventMatrix.push_back(aline25);
  energyVector.push_back(resEnergy);


  if(makeIteration)
  {
    makeIteration=false;
    std::cout << "EventMatrix size = " << EventMatrix.size() << " energyVector size = " << energyVector.size() << std::endl;

    if(calibAlgo_=="L3_ALGORITHM") {
       temp_solution = algoL3.iterate(EventMatrix, energyVector, 20);
       }   // run it 20 times
    else if(calibAlgo_=="HH_ALGORITHM") {
       temp_solution = algoHH.iterate(EventMatrix, energyVector, 4);
       } // run it 4times
    else { 
       cout << "CALIBRATION ALGORITHM NOT SELECTED OR WRONG" << endl;
    return; 
    }
    
    if (!temp_solution.empty()){
    
       std::cout << "Temporary Solution Size = " << temp_solution.size() << std::endl;
       
       if(temp_solution.size() != ReducedMap.size())
       std::cout << "Problems with solution size: different from map size!" << std::endl;
             
	     if(nupdates==0) {
	        
		solution.resize(ReducedMap.size(),0.);
	      }
	      
	      for (int isize=0;isize<ReducedMap.size();isize++){
              
		    solution[isize]+=temp_solution[isize];
	      }
	      
	      nupdates++;
	      cout << "nupdates = " << nupdates << endl;
       	      EventMatrix.clear();
	      energyVector.clear();
	      temp_solution.clear();
	      


          } //if HH
   
   } //if iteration
  

}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronCalibration)
