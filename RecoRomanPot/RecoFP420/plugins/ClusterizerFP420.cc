///////////////////////////////////////////////////////////////////////////////
// File: ClusterizerFP420.cc
// Date: 12.2006
// Description: ClusterizerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include <memory>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "RecoRomanPot/RecoFP420/interface/ClusterizerFP420.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include <iostream> 
using namespace std;

//ClusterizerFP420::ClusterizerFP420(const edm::ParameterSet& conf):
//  conf_(conf),sClusterizerFP420_(new FP420ClusterMain(conf)) {
ClusterizerFP420::ClusterizerFP420(const edm::ParameterSet& conf):
  conf_(conf) {
    
    
    edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("ClusterizerFP420");
    verbosity    = m_Anal.getParameter<int>("Verbosity");
    sn0 = m_Anal.getParameter<int>("NumberFP420Stations");
    pn0 = m_Anal.getParameter<int>("NumberFP420SPlanes");
    if (verbosity > 0) {
      std::cout << "Creating a ClusterizerFP420" << std::endl;
      std::cout << "ClusterizerFP420: sn0=" << sn0 << " pn0=" << pn0 << std::endl;
    }
    
    sClusterizerFP420_ = new FP420ClusterMain(conf_,sn0,pn0);
  }

// Virtual destructor needed.
ClusterizerFP420::~ClusterizerFP420() { 
  delete sClusterizerFP420_;
}  

//Get at the beginning
void ClusterizerFP420::beginJob() {
  if (verbosity > 0) {
    std::cout << "BeginJob method " << std::endl;
  }
  //Getting Calibration data (Noises and BadElectrodes Flag)
  //    bool UseNoiseBadElectrodeFlagFromDB_=conf_.getParameter<bool>("UseNoiseBadElectrodeFlagFromDB");  
  //    if (UseNoiseBadElectrodeFlagFromDB_==true){
  //      iSetup.get<ClusterNoiseFP420Rcd>().get(noise);// AZ: do corrections for noise here
  //=========================================================
  // 
  // Debug: show noise for DetIDs
  //       ElectrodNoiseMapIterator mapit = noise->m_noises.begin();
  //       for (;mapit!=noise->m_noises.end();mapit++)
  // 	{
  // 	  unsigned int detid = (*mapit).first;
  // 	  std::cout << "detid " <<  detid << " # Electrode " << (*mapit).second.size()<<std::endl;
  // 	  //ElectrodNoiseVector theElectrodVector =  (*mapit).second;     
  // 	  const ElectrodNoiseVector theElectrodVector =  noise->getElectrodNoiseVector(detid);
  
  
  // 	  int electrode=0;
  // 	  ElectrodNoiseVectorIterator iter=theElectrodVector.begin();
  // 	  //for(; iter!=theElectrodVector.end(); iter++)
  // 	  {
  // 	    std::cout << " electrode " << electrode++ << " =\t"
  // 		      << iter->getNoise()     << " \t" 
  // 		      << iter->getDisable()   << " \t" 
  // 		      << std::endl; 	    
  // 	  } 
  //       }
  //===========================================================
  //    }
}

// Initialization:
//  theFP420NumberingScheme = new FP420NumberingScheme();
//  theFP420DigiMain = new FP420DigiMain();
//      void DigitizerFP420::produce(FP420G4HitCollection *   theCAFI, DigiCollectionFP420 & soutput) {

// Functions that gets called by framework every event
// void ClusterizerFP420::produce(DigiCollectionFP420 * input, ClusterCollectionFP420 & soutput)
void ClusterizerFP420::produce(DigiCollectionFP420 & input, ClusterCollectionFP420 & soutput)
{
  //  beginJob;
  
  //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
  // clean info of container from previous event
  for (int sector=1; sector<sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=1; zside<3; zside++) {
	// zside here defines just Left or Right planes, not their type !!!
	//	  int sScale = 20;
	int sScale = 2*(pn0-1);
	//      int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	// intindex is a continues numbering of FP420
	int zScale=2;  unsigned int detID = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	std::vector<ClusterFP420> collector;
	collector.clear();
	ClusterCollectionFP420::Range inputRange;
	inputRange.first = collector.begin();
	inputRange.second = collector.end();
	
	soutput.putclear(inputRange,detID);
	
      }//for
    }//for
  }//for
  
  
  //                                                                                                                      !!!!!!   
  // if we want to keep Cluster container/Collection for one event --->   uncomment the line below and vice versa
  soutput.clear();   //container_.clear() --> start from the beginning of the container
  
  //                                RUN now:                                                                                 !!!!!!     
  // startFP420ClusterMain_.run(input, soutput, noise);
  sClusterizerFP420_->run(input, soutput, noise);
  //  std::cout <<"=======           ClusterizerFP420:                    end of produce     " <<  std::endl;
  
}

