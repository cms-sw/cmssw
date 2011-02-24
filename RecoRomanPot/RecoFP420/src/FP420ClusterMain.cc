///////////////////////////////////////////////////////////////////////////////
// File: FP420ClusterMain.cc
// Date: 02.2011
// Description: FP420ClusterMain for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoRomanPot/RecoFP420/interface/FP420ClusterMain.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterProducerFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include "CLHEP/Random/RandFlat.h"

using namespace std;


FP420ClusterMain::FP420ClusterMain(const edm::ParameterSet& conf, int dn, int sn, int pn, int rn, int dh, int sh, int ph, int rh):conf_(conf),dn0(dn),sn0(sn),pn0(pn),rn0(rn),dh0(dh),sh0(sh),ph0(ph),rh0(rh)  { 
  
  verbosity         = conf_.getUntrackedParameter<int>("VerbosityLevel");

  ElectronPerADCFP420   = conf_.getParameter<double>("ElectronFP420PerAdc");
  clusterModeFP420      = conf_.getParameter<std::string>("ClusterModeFP420");
  ChannelThresholdFP420  = conf_.getParameter<double>("ChannelFP420Threshold");//6
  SeedThresholdFP420     = conf_.getParameter<double>("SeedFP420Threshold");//7
  ClusterThresholdFP420  = conf_.getParameter<double>("ClusterFP420Threshold");//7
  MaxVoidsInClusterFP420 = conf_.getParameter<int>("MaxVoidsFP420InCluster");//1
  
  ElectronPerADCHPS240   = conf_.getParameter<double>("ElectronHPS240PerAdc");
  clusterModeHPS240      = conf_.getParameter<std::string>("ClusterModeHPS240");
  ChannelThresholdHPS240  = conf_.getParameter<double>("ChannelHPS240Threshold");//6
  SeedThresholdHPS240     = conf_.getParameter<double>("SeedHPS240Threshold");//7
  ClusterThresholdHPS240  = conf_.getParameter<double>("ClusterHPS240Threshold");//7
  MaxVoidsInClusterHPS240 = conf_.getParameter<int>("MaxVoidsHPS240InCluster");//1

  if (verbosity > 0) {
    std::cout << "FP420ClusterMain constructor: ElectronPerADCFP420 = " << ElectronPerADCFP420 << std::endl;
    std::cout << " clusterModeFP420 = " << clusterModeFP420 << std::endl;
    std::cout << " ChannelThresholdFP420 = " << ChannelThresholdFP420 << std::endl;
    std::cout << " SeedThresholdFP420 = " << SeedThresholdFP420 << std::endl;
    std::cout << " ClusterThresholdFP420 = " << ClusterThresholdFP420 << std::endl;
    std::cout << " MaxVoidsInClusterFP420 = " << MaxVoidsInClusterFP420 << std::endl;
    std::cout << "FP420ClusterMain constructor: ElectronPerADCHPS240 = " << ElectronPerADCHPS240 << std::endl;
    std::cout << " clusterModeHPS240 = " << clusterModeHPS240 << std::endl;
    std::cout << " ChannelThresholdHPS240 = " << ChannelThresholdHPS240 << std::endl;
    std::cout << " SeedThresholdHPS240 = " << SeedThresholdHPS240 << std::endl;
    std::cout << " ClusterThresholdHPS240 = " << ClusterThresholdHPS240 << std::endl;
    std::cout << " MaxVoidsInClusterHPS240 = " << MaxVoidsInClusterHPS240 << std::endl;
  }

  ENCFP420 = 960.;    // 
  BadElectrodeProbabilityFP420 = 0.002;
  UseNoiseBadElectrodeFlagFromDBFP420 = false;


  ENCHPS240 = 960.;    // 
  BadElectrodeProbabilityHPS240 = 0.002;
  UseNoiseBadElectrodeFlagFromDBHPS240 = false;



  //
  xytype=2;// only X types of planes
  // pitches;
  //
//  pitchY= 0.050;// was 0.040
//  pitchX= 0.050;

  moduleThicknessY = 0.250; // mm
  moduleThicknessX = 0.250; // mm
  Thick300 = 0.300;

  // for 50x400 um2 pixels  (FP420):
  numFP420StripsY = 144;        // Y plate number of strips:144*0.050=7.2mm (xytype=1)
  numFP420StripsX = 160;        // X plate number of strips:160*0.050=8.0mm (xytype=2)
  numFP420StripsYW = 20;        // Y plate number of W strips:20 *0.400=8.0mm (xytype=1) - W have ortogonal projection
  numFP420StripsXW = 18;        // X plate number of W strips:18 *0.400=7.2mm (xytype=2) - W have ortogonal projection


  //for 100x150 um2 pixels (HPS240):
  numHPS240StripsY = 72;        // Y plate number of strips:72*0.100=7.2mm (xytype=1)
  numHPS240StripsX = 80;        // X plate number of strips:80*0.100=8.0mm (xytype=2)
  numHPS240StripsYW = 53;        // Y plate number of W strips:53 *0.150=8.0mm (xytype=1) - W have ortogonal projection
  numHPS240StripsXW = 48;        // X plate number of W strips:48 *0.150=7.2mm (xytype=2) - W have ortogonal projection


  //  sn0 = 4;
  //  pn0 = 9;


  theFP420NumberingScheme = new FP420NumberingScheme();


  if (verbosity > 1) {
    std::cout << "FP420ClusterMain constructor: sn0 = " << sn0 << " pn0=" << pn0 << " dn0=" << dn0 << " rn0=" << rn0 << std::endl;
    std::cout << "FP420ClusterMain constructor: ENCFP420 = " << ENCFP420 << std::endl;
    std::cout << "FP420ClusterMain constructor: sh0 = " << sh0 << " ph0=" << ph0 << " dh0=" << dh0 << " rh0=" << rh0 << std::endl;
    std::cout << "FP420ClusterMain constructor: ENCHPS240 = " << ENCHPS240 << std::endl;
    std::cout << " BadElectrodeProbabilityFP420 = " << BadElectrodeProbabilityFP420 << std::endl;
    std::cout << " BadElectrodeProbabilityHPS240 = " << BadElectrodeProbabilityHPS240 << std::endl;

    std::cout << " Thick300 = " << Thick300 << std::endl;
    std::cout << " numFP420StripsY = " << numFP420StripsY << " numFP420StripsX = " << numFP420StripsX << std::endl;
    std::cout << " numHPS240StripsY = " << numHPS240StripsY << " numHPS240StripsX = " << numHPS240StripsX << std::endl;
    std::cout << " moduleThicknessY = " << moduleThicknessY << " moduleThicknessX = " << moduleThicknessX << std::endl;

    //  std::cout << " pitchY = " << pitchY << " pitchX = " << pitchX << std::endl;
  }
  
  if (UseNoiseBadElectrodeFlagFromDBFP420==false){	  
    if (verbosity > 0) {
      std::cout << "FP420:  using a SingleNoiseValue and good electrode flags" << std::endl;
    }
  } else {
    if (verbosity > 0) {
      std::cout << "FP420:  using Noise and BadElectrode flags accessed from DB" << std::endl;
    }
  }
  if (UseNoiseBadElectrodeFlagFromDBHPS240==false){	  
    if (verbosity > 0) {
      std::cout << "HPS240:   using a SingleNoiseValue and good electrode flags" << std::endl;
    }
  } else {
    if (verbosity > 0) {
      std::cout << "HPS240:   using Noise and BadElectrode flags accessed from DB" << std::endl;
    }
  }
  
  if ( clusterModeFP420 == "ClusterProducerFP420" ) {
    threeThresholdFP420_ = new ClusterProducerFP420(ChannelThresholdFP420, SeedThresholdFP420, ClusterThresholdFP420, MaxVoidsInClusterFP420);
    validClusterizerFP420 = true;
  } else {
    std::cout << "ERROR:FP420ClusterMain:FP420 No valid clusterizer selected" << std::endl;
    validClusterizerFP420 = false;
  }
  if ( clusterModeHPS240 == "ClusterProducerHPS240" ) {
    threeThresholdHPS240_ = new ClusterProducerFP420(ChannelThresholdHPS240, SeedThresholdHPS240, ClusterThresholdHPS240, MaxVoidsInClusterHPS240);
    validClusterizerHPS240 = true;
  } else {
    std::cout << "ERROR:FP420ClusterMain:HPS240 No valid clusterizer selected" << std::endl;
    validClusterizerHPS240 = false;
  }


}

FP420ClusterMain::~FP420ClusterMain() {
  if ( threeThresholdFP420_ != 0 ) {delete threeThresholdFP420_;}
  if ( threeThresholdHPS240_ != 0 ) {delete threeThresholdHPS240_;}
}


//void FP420ClusterMain::run(const DigiCollectionFP420 *input, ClusterCollectionFP420 &soutput,
//			   const std::vector<ClusterNoiseFP420>& electrodnoise)
void FP420ClusterMain::run(edm::Handle<DigiCollectionFP420> &input, std::auto_ptr<ClusterCollectionFP420> &soutput)
  
{
  // unpack from iu:
  //  int  sScale = 20, zScale=2;
  //  int  sector = (iu-1)/sScale + 1 ;
  //  int  zmodule = (iu - (sector - 1)*sScale - 1) /zScale + 1 ;
  //  int  zside = iu - (sector - 1)*sScale - (zmodule - 1)*zScale ;
  
  if (verbosity > 0) {
    std::cout << "FP420ClusterMain: OK1" << std::endl;
  }
  
  int number_detunits          = 0;
  int number_localelectroderechits = 0;
  
  // get vector of detunit ids
  //    const std::vector<unsigned int> detIDs = input->detIDs();
  
  // to be used in put (besause of 0 in cluster collection for: 1) 1st cluster and 2) case of no cluster)
  // ignore 0, but to save info for 1st cluster record it second time on place 1   .
  
  // loop over detunits
  bool first = true;
  // det = 1 for +FP420 , = 2 for -FP420
  // det = 3 for +HPS240 , = 4 for -HPS240 
  int sn_end=0, pn_end=0, rn_end=0;
  int det_start = 1, det_finish = 5;
  if(dn0 < 1) det_start = 3;
  if(dh0 < 1) det_finish = 3;
  for (int det=det_start; det<det_finish; det++) {
    if(det<3) {sn_end=sn0; pn_end=pn0; rn_end=rn0;}
    else if(det<5) {sn_end=sh0; pn_end=ph0; rn_end=rh0;}
    for (int sector=1; sector<sn_end; sector++) {
      for (int zmodule=1; zmodule<pn_end; zmodule++) {
	for (int zside=1; zside<rn_end; zside++) {
	  // intindex is a continues numbering of FP420
	  unsigned int detID = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn_end, pn_end, sn_end, det, zside, sector, zmodule);
	  //	  unsigned int detID = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	  if (verbosity > 0) {
	    std::cout << " FP420ClusterMain:1 run loop   index no  iu = " << detID  << std::endl;
	  }	  
	  // Y:
	  if (xytype ==1) {
	    numFP420Strips = numFP420StripsY*numFP420StripsYW;  
	    numHPS240Strips = numHPS240StripsY*numHPS240StripsYW;  
	    moduleThickness = moduleThicknessY; 
	    //	      pitch= pitchY;
	  }
	  // X:
	  if (xytype ==2) {
	    numFP420Strips = numFP420StripsX*numFP420StripsXW;  
	    numHPS240Strips = numHPS240StripsX*numHPS240StripsXW;  
	    moduleThickness = moduleThicknessX; 
	    //	      pitch= pitchX;
	  }
	  
	  
	  //    for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); ++detunit_iterator ) {
	  //      unsigned int detID = *detunit_iterator;
	  ++number_detunits;
	  
	  //   .
	  //   GET DIGI collection  !!!!
	  //   .
	  //	  const DigiCollectionFP420::Range digiRange = input->get(detID);
	  DigiCollectionFP420::Range digiRange;
	  std::vector<HDigiFP420> dcollector;
	  // if (dcollector.size()>0){
	  if (verbosity > 0) {
	    std::cout << " FP420ClusterMain:2 number_detunits = " << number_detunits  << std::endl;
	  }	  
	  digiRange = input->get(detID);
	  //digiRange = input.get(detID);
	  // }
	  
	  if (verbosity > 0) {
	    std::cout << " FP420ClusterMain: input->get DONE dcollector.size()=" << dcollector.size() << std::endl;
	  }	  
	  
	  DigiCollectionFP420::ContainerIterator sort_begin = digiRange.first;
	  DigiCollectionFP420::ContainerIterator sort_end = digiRange.second;
	  for ( ;sort_begin != sort_end; ++sort_begin ) {
	    dcollector.push_back(*sort_begin);
	  } // for
	  if (dcollector.size()>0) {
	    
	    DigiCollectionFP420::ContainerIterator digiRangeIteratorBegin = digiRange.first;
	    DigiCollectionFP420::ContainerIterator digiRangeIteratorEnd   = digiRange.second;
	    if (verbosity > 0) {
	      std::cout << " FP420ClusterMain: channel Begin = " << (digiRangeIteratorBegin)->channel()  << std::endl;
	      std::cout << " FP420ClusterMain: channel end = " << (digiRangeIteratorEnd-1)->channel()  << std::endl;
	    }	    
	    if (verbosity > 0) {
	      std::cout << " FP420ClusterMain:3 noise treatment  " << std::endl;
	    }	  
	    //   DIGI collection  is taken  !!!!
	    
	    
	    std::vector<ClusterFP420> collector;
	    
	    //FP420:	    
	    if(det<3) {
	      if ( validClusterizerFP420) {
		if ( clusterModeFP420 == "ClusterProducerFP420" ) {
		  if (UseNoiseBadElectrodeFlagFromDBFP420==false){	  
		    
		    //Case of SingleValueNoise flags for all electrodes of a Detector
		    float noise = ENCFP420*moduleThickness/Thick300/ElectronPerADCFP420;//Noise is proportional to moduleThickness
		    
		    //vector<float> noiseVec(numElectrodes,noise);	    
		    //Construct a ElectrodNoiseVector ( in order to be compliant with the DB access)
		    ElectrodNoiseVector vnoise;
		    ClusterNoiseFP420::ElectrodData theElectrodData;       	   
		    
		    if (verbosity > 0) {std::cout << " FP420ClusterMain:4 numFP420Strips = " << numFP420Strips  << std::endl;}	  
		    for(int electrode=0; electrode < numFP420Strips; ++electrode){
		      //   discard  randomly  bad  electrode with probability BadElectrodeProbabilityFP420
		      bool badFlag= CLHEP::RandFlat::shoot(1.) < BadElectrodeProbabilityFP420 ? true : false;
		      theElectrodData.setData(noise,badFlag);
		      vnoise.push_back(theElectrodData);// fill vector vnoise
		    } // for
		    if (verbosity > 0) {std::cout << " FP420ClusterMain:5 BadElectrodeProbability added " << std::endl;}	  

		    collector.clear();
		    //   if (dcollector.size()>0){
		    collector = threeThresholdFP420_->clusterizeDetUnitPixels(digiRangeIteratorBegin,digiRangeIteratorEnd,detID,vnoise,xytype,verbosity);
		    //   }
		    if (verbosity > 0) {std::cout << " FP420ClusterMain:6 threeThresholdFP420 OK " << std::endl;}	  

		  } else {
		    //Case of Noise and BadElectrode flags access from DB
		    /*
		      std::vector<ClusterNoiseFP420>& electrodnoise
		      const ElectrodNoiseVector& vnoise = electrodnoise->getElectrodNoiseVector(detID);
		      
		      if (vnoise.size() <= 0) {
		      std::cout << "WARNING requested Noise Vector for detID " << detID << " that isn't in map " << std::endl; 
		      continue;
		      }
		      collector.clear();
		      collector = threeThresholdFP420_->clusterizeDetUnit(digiRangeIteratorBegin,digiRangeIteratorEnd,detID,vnoise);
		    */
		  }// if (UseNoiseBadElectrodeFlagFromDBFP420

		}// if ( clusterModeFP420
	      }// if ( validClusterizerFP420
	    }// if(det<3)


	    else if(det<5) {
	      if ( validClusterizerHPS240) {
		if ( clusterModeHPS240 == "ClusterProducerHPS240" ) {
		  if (UseNoiseBadElectrodeFlagFromDBHPS240==false){	  
		    
		    //Case of SingleValueNoise flags for all electrodes of a Detector
		    float noise = ENCHPS240*moduleThickness/Thick300/ElectronPerADCHPS240;//Noise is proportional to moduleThickness
		    
		    //vector<float> noiseVec(numElectrodes,noise);	    
		    //Construct a ElectrodNoiseVector ( in order to be compliant with the DB access)
		    ElectrodNoiseVector vnoise;
		    ClusterNoiseFP420::ElectrodData theElectrodData;       	   
		    
		    if (verbosity > 0) {std::cout << " HPS240ClusterMain:4 numHPS240Strips = " << numHPS240Strips  << std::endl;}	  
		    for(int electrode=0; electrode < numHPS240Strips; ++electrode){
		      //   discard  randomly  bad  electrode with probability BadElectrodeProbabilityHPS240
		      bool badFlag= CLHEP::RandFlat::shoot(1.) < BadElectrodeProbabilityHPS240 ? true : false;
		      theElectrodData.setData(noise,badFlag);
		      vnoise.push_back(theElectrodData);// fill vector vnoise
		    } // for
		    if (verbosity > 0) {std::cout << " HPS240ClusterMain:5 BadElectrodeProbability added " << std::endl;}	  

		    collector.clear();
		    //   if (dcollector.size()>0){
		    collector = threeThresholdHPS240_->clusterizeDetUnitPixels(digiRangeIteratorBegin,digiRangeIteratorEnd,detID,vnoise,xytype,verbosity);
		    //   }
		    if (verbosity > 0) {std::cout << " HPS240ClusterMain:6 threeThreshold OK " << std::endl;}	  

		  } else {
		    //Case of Noise and BadElectrode flags access from DB
		    /*
		      std::vector<ClusterNoiseFP420>& electrodnoise
		      const ElectrodNoiseVector& vnoise = electrodnoise->getElectrodNoiseVector(detID);
		      
		      if (vnoise.size() <= 0) {
		      std::cout << "WARNING requested Noise Vector for detID " << detID << " that isn't in map " << std::endl; 
		      continue;
		      }
		      collector.clear();
		      collector = threeThresholdHPS240_->clusterizeDetUnit(digiRangeIteratorBegin,digiRangeIteratorEnd,detID,vnoise);
		    */
		  }// if (UseNoiseBadElectrodeFlagFromDBHPS240

		}// if ( clusterModeHPS240
	      }// if ( validClusterizerHPS240
	    }//else if(det<5)



	    if (collector.size()>0){
	      ClusterCollectionFP420::Range inputRange;
	      inputRange.first = collector.begin();
	      inputRange.second = collector.end();
	      
	      if (verbosity > 0) {
		std::cout << " FP420ClusterMain:7 collector.size()>0 " << std::endl;
	      }	  
	      if ( first ) {
		// use it only if ClusterCollectionFP420 is the ClusterCollection of one event, otherwise, do not use (loose 1st cl. of 1st event only)
		first = false;
		unsigned int detID0 = 0;
		if (verbosity > 0) {
		  std::cout << " FP420ClusterMain:8 first soutput->put " << std::endl;
		}	  
		soutput->put(inputRange,detID0); // !!! put into adress 0 for detID which will not be used never
	      } //if ( first ) 
	      
	      // !!! put
	      if (verbosity > 0) {
		std::cout << " FP420ClusterMain:9 soutput->put " << std::endl;
	      }	  
	      soutput->put(inputRange,detID);
	      
	      number_localelectroderechits += collector.size();
	    } // if (collector.size
	    
	    
	    
	    
	    if (verbosity > 0) {
	      std::cout << "[FP420ClusterMain]:  generating " << number_localelectroderechits << " ClusterFP420s in " << number_detunits << " DetUnits." << std::endl; 
	    }//if (verb
	  }// if (dcollector.siz
	}//for
      }//for
    }//for
  }//for
  
  if (verbosity == -50 || verbosity>0 ) {
    
    std::cout <<" FP420  ===================================================      Checks" << std::endl;
    //     check of access to the collector:
    for (int det=1; det<dn0; det++) {
      for (int sector=1; sector<sn0; sector++) {
	for (int zmodule=1; zmodule<pn0; zmodule++) {
	  for (int zside=1; zside<rn0; zside++) {
	    // intindex is a continues numbering of FP420
	    unsigned int iu = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	    std::cout <<" iu = " << iu <<" sector = " << sector <<" zmodule = " << zmodule <<" zside = " << zside << "  det=" << det << std::endl;
	    std::vector<ClusterFP420> collector;
	    std::cout <<" =1=" << std::endl;
	    collector.clear();
	    std::cout <<" =2=" << std::endl;
	    ClusterCollectionFP420::Range outputRange;
	    std::cout <<" =3=" << std::endl;
	    outputRange = soutput->get(iu);
	    // fill output in collector vector (for may be sorting? or other checks)
	    std::cout <<" =4=" << std::endl;
	    ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	    std::cout <<" =5=" << std::endl;
	    ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
	    std::cout <<" =6=" << std::endl;
	    for ( ;sort_begin != sort_end; ++sort_begin ) {
	      std::cout <<" =7=" << std::endl;
	      collector.push_back(*sort_begin);
	      std::cout <<" =8=" << std::endl;
	    } // for
	    std::cout <<" =9=" << std::endl;
	    if (collector.size()>0) {
	      std::cout <<" =========== FP420ClusterMain:check: iu= " << iu <<    " zside = " << zside << std::endl;
	      std::cout <<"  ======renew collector size = " << collector.size() << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::vector<ClusterFP420>::const_iterator simHitIter = collector.begin();
	      std::vector<ClusterFP420>::const_iterator simHitIterEnd = collector.end();
	      // loop in #clusters
	      for (;simHitIter != simHitIterEnd; ++simHitIter) {
		const ClusterFP420 icluster = *simHitIter;
		//   if(icluster.amplitudes().size()>390) {
		std::cout << " ===== size of cluster= " << icluster.amplitudes().size() << std::endl;
		std::cout <<" ===" << std::endl;
		std::cout << " ===== firstStrip = " << icluster.firstStrip() << "  barycenter = " << icluster.barycenter() << "  barycenterW = " << icluster.barycenterW() << std::endl;
		std::cout <<" ===" << std::endl;
		for(unsigned int i = 0; i < icluster.amplitudes().size(); i++ ) {
		  std::cout <<  "i = " << i << "   amplitudes = "  << icluster.amplitudes()[i] << std::endl;
		}// for ampl
		std::cout <<" ===" << std::endl;
		std::cout <<" ===" << std::endl;
		std::cout <<" =======================" << std::endl;
		//  }// if(icluster.amplitudes().size()>390
	      }//for cl
	    }//  if (collector.size()>0
	    
	    /* 
	       for (DigitalMapType::const_iterator i=collector.begin(); i!=collector.end(); i++) {
	       std::cout << "DigitizerFP420:check: HDigiFP420::  " << std::endl;
	       std::cout << " strip number is as (*i).first  = " << (*i).first << "  adc is in (*i).second  = " << (*i).second << std::endl;
	       }
	    */
	    
	    //==================================
	    
	  }   // for
	}   // for
      }   // for
    }   // for
    
    //     end of check of access to the strip collection
    std::cout <<"=======            FP420ClusterMain:                    end of check     " << std::endl;
      std::cout <<" HPS240  ===================================================      Checks" << std::endl;
      // det = 3 for +HPS240 , = 4 for -HPS240 
      int detHPS240 = dn0+2;
      for (int det=3; det<detHPS240; det++) {
	for (int sector=1; sector<sn0; sector++) {
	  for (int zmodule=1; zmodule<pn0; zmodule++) {
	    for (int zside=1; zside<rn0; zside++) {
	      // intindex is a continues numbering of HPS240
	      unsigned int iu = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
		std::cout <<" iu = " << iu <<" sector = " << sector <<" zmodule = " << zmodule <<" zside = " << zside << "  det=" << det << std::endl;
	      std::vector<ClusterFP420> collector;
	      collector.clear();
	      ClusterCollectionFP420::Range outputRange;
	      outputRange = soutput->get(iu);
	      // fill output in collector vector (for may be sorting? or other checks)
	      ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	      ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
	      for ( ;sort_begin != sort_end; ++sort_begin ) {
		collector.push_back(*sort_begin);
	      } // for
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::cout <<" =========== HPS240ClusterMain:check: iu= " << iu <<    " zside = " << zside << std::endl;
	      std::cout <<"  ======renew collector size = " << collector.size() << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::vector<ClusterFP420>::const_iterator simHitIter = collector.begin();
	      std::vector<ClusterFP420>::const_iterator simHitIterEnd = collector.end();
	      // loop in #clusters
	      for (;simHitIter != simHitIterEnd; ++simHitIter) {
		const ClusterFP420 icluster = *simHitIter;
		//   if(icluster.amplitudes().size()>390) {
		std::cout << " ===== size of cluster= " << icluster.amplitudes().size() << std::endl;
		std::cout <<" ===" << std::endl;
		std::cout << " ===== firstStrip = " << icluster.firstStrip() << "  barycenter = " << icluster.barycenter() << "  barycenterW = " << icluster.barycenterW() << std::endl;
		std::cout <<" ===" << std::endl;
		for(unsigned int i = 0; i < icluster.amplitudes().size(); i++ ) {
		  std::cout <<  "i = " << i << "   amplitudes = "  << icluster.amplitudes()[i] << std::endl;
		}// for ampl
		std::cout <<" ===" << std::endl;
		std::cout <<" ===" << std::endl;
		std::cout <<" =======================" << std::endl;
		//  }// if(icluster.amplitudes().size()>390
	      }//for cl
	      
	      /* 
		 for (DigitalMapType::const_iterator i=collector.begin(); i!=collector.end(); i++) {
		 std::cout << "DigitizerFP420:check: HDigiFP420::  " << std::endl;
		 std::cout << " strip number is as (*i).first  = " << (*i).first << "  adc is in (*i).second  = " << (*i).second << std::endl;
		 }
	      */
	      
	      //==================================
	      
	    }   // for
	  }   // for
	}   // for
      }   // for
      
      //     end of check of access to the strip collection
      std::cout <<"=======            HPS240ClusterMain:                    end of check     " << std::endl;
          
  }// if (verbosit
  
}
