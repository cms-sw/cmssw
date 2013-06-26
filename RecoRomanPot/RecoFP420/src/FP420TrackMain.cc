///////////////////////////////////////////////////////////////////////////////
// File: FP420TrackMain.cc
// Date: 12.2006
// Description: FP420TrackMain for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoRomanPot/RecoFP420/interface/FP420TrackMain.h"
#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackProducerFP420.h"

#include "CLHEP/Random/RandFlat.h"

using namespace std;

//#define mytrackdebug0

//FP420TrackMain::FP420TrackMain(){ 
FP420TrackMain::FP420TrackMain(const edm::ParameterSet& conf):conf_(conf)  { 
  
  verbosity   = conf_.getUntrackedParameter<int>("VerbosityLevel");
  trackMode_  =  conf_.getParameter<std::string>("TrackModeFP420");
  dn0   = conf_.getParameter<int>("NumberFP420Detectors");
  sn0_ = conf_.getParameter<int>("NumberFP420Stations");
  pn0_ = conf_.getParameter<int>("NumberFP420SPlanes");
  rn0_ = 7;
  xytype_ = conf_.getParameter<int>("NumberFP420SPTypes");
  z420_           = conf_.getParameter<double>("z420");
  zD2_            = conf_.getParameter<double>("zD2");
  zD3_            = conf_.getParameter<double>("zD3");
  dXX_ = conf_.getParameter<double>("dXXFP420");
  dYY_ = conf_.getParameter<double>("dYYFP420");
  chiCutX_ = conf_.getParameter<double>("chiCutX420");
  chiCutY_ = conf_.getParameter<double>("chiCutY420");
  
  if (verbosity > 0) {
    std::cout << "FP420TrackMain constructor::" << std::endl;
    std::cout << "sn0=" << sn0_ << " pn0=" << pn0_ << " xytype=" << xytype_ << std::endl;
    std::cout << "trackMode = " << trackMode_ << std::endl;
    std::cout << "dXX=" << dXX_ << " dYY=" << dYY_ << std::endl;
    std::cout << "chiCutX=" << chiCutX_ << " chiCutY=" << chiCutY_ << std::endl;
  }
  ///////////////////////////////////////////////////////////////////
    // zD2_ = 1000.;  // dist between centers of 1st and 2nd stations
    // zD3_ = 8000.;  // dist between centers of 1st and 3rd stations
    
  UseHalfPitchShiftInX_= true;
  UseHalfPitchShiftInXW_= true;
  UseHalfPitchShiftInY_= true;
  UseHalfPitchShiftInYW_= true;

  pitchX_= 0.050;
  pitchY_= 0.050;// 
  pitchXW_= 0.400;
  pitchYW_= 0.400;// 

  XsensorSize_=8.0;
  YsensorSize_=7.2;

//
  zBlade_ = 5.00;
  gapBlade_ = 1.6;
  double gapSupplane = 1.6;
  ZSiPlane_=2*zBlade_+gapBlade_+gapSupplane;
  
  double ZKapton = 0.1;
  ZSiStep_=ZSiPlane_+ZKapton;
  
  double ZBoundDet = 0.020;
  double ZSiElectr = 0.250;
  double ZCeramDet = 0.500;

  double eee1=11.;
  double eee2=12.;
  zinibeg_ = (eee1-eee2)/2.;
//
  ZSiDet_ = 0.250;
//
  ZGapLDet_= zBlade_/2-(ZSiDet_+ZSiElectr+ZBoundDet+ZCeramDet/2);
//
    if (verbosity > 1) {
      std::cout << "FP420TrackMain constructor::" << std::endl;
      std::cout << " zD2=" << zD2_ << " zD3=" << zD3_ << " zinibeg =" << zinibeg_ << std::endl;
      std::cout << " UseHalfPitchShiftInX=" << UseHalfPitchShiftInX_ << " UseHalfPitchShiftInY=" << UseHalfPitchShiftInY_ << std::endl;
      std::cout << " UseHalfPitchShiftInXW=" << UseHalfPitchShiftInXW_ << " UseHalfPitchShiftInYW=" << UseHalfPitchShiftInYW_ << std::endl;
      std::cout << " pitchX=" << pitchX_ << " pitchY=" << pitchY_ << std::endl;
      std::cout << " pitchXW=" << pitchXW_ << " pitchYW=" << pitchYW_ << std::endl;
      std::cout << " zBlade_=" << zBlade_ << " gapBlade_=" << gapBlade_ << std::endl;
      std::cout << " ZKapton=" << ZKapton << " ZBoundDet=" << ZBoundDet << std::endl;
      std::cout << " ZSiElectr=" << ZSiElectr << " ZCeramDet=" << ZCeramDet << std::endl;
      std::cout << " ZSiDet=" << ZSiDet_ << " gapSupplane=" << gapSupplane << std::endl;
    }
  ///////////////////////////////////////////////////////////////////



      if ( trackMode_ == "TrackProducerSophisticatedFP420" ) {
  
  
  //trackMode_ == "TrackProducerVar1FP420" ||
  //trackMode_ == "TrackProducerVar2FP420" ||

     // if ( trackMode_ == "TrackProducerMaxAmplitudeFP420" ||
//	   trackMode_ == "TrackProducerMaxAmplitude2FP420"  ||
//	   trackMode_ == "TrackProducerSophisticatedFP420"  ||
//	   trackMode_ == "TrackProducer3DFP420" )  {

	finderParameters_ = new TrackProducerFP420(sn0_, pn0_, rn0_, xytype_, z420_, zD2_, zD3_,
						   pitchX_, pitchY_,
						   pitchXW_, pitchYW_,
						   ZGapLDet_, ZSiStep_,
						   ZSiPlane_, ZSiDet_,zBlade_,gapBlade_,
						   UseHalfPitchShiftInX_, UseHalfPitchShiftInY_,
						   UseHalfPitchShiftInXW_, UseHalfPitchShiftInYW_,
						   dXX_,dYY_,chiCutX_,chiCutY_,zinibeg_,verbosity,
						   XsensorSize_,YsensorSize_);
	validTrackerizer_ = true;
      } 
      else {
	std::cout << "ERROR:FP420TrackMain: No valid finder selected" << std::endl;
	validTrackerizer_ = false;
      }
}

FP420TrackMain::~FP420TrackMain() {
  if ( finderParameters_ != 0 ) {
    delete finderParameters_;
  }
}



void FP420TrackMain::run(edm::Handle<ClusterCollectionFP420> &input, std::auto_ptr<TrackCollectionFP420> &toutput )
{
  
  if ( validTrackerizer_ ) {
    
    int number_detunits          = 0;
    int number_localelectroderechits = 0;
    /*
      for (int sector=1; sector<sn0_; sector++) {
      for (int zmodule=1; zmodule<pn0_; zmodule++) {
      for (int zside=1; zside<rn0_; zside++) {
      int sScale = 2*(pn0-1);
      //      int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
      // intindex is a continues numbering of FP420
      int zScale=2;  unsigned int detID = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
      ClusterMap.clear();
      ClusterCollectionFP420::Range clusterRange;
      clusterRange = input.get(detID);
      ClusterCollectionFP420::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
      ClusterCollectionFP420::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
      for ( ;sort_begin != sort_end; ++sort_begin ) {
      ClusterMap.push_back(*sort_begin);
      } // for
      
      }//for
      }//for
      }//for
    */
    // get vector of detunit ids
    //    const std::vector<unsigned int> detIDs = input->detIDs();
    
    // to be used in put (besause of 0 in track collection for: 1) 1st track and 2) case of no track)
    // ignore 0, but to save info for 1st track record it second time on place 1   .
    
    bool first = true;
    // loop over detunits
    for (int det=1; det<dn0; det++) {
      ++number_detunits;
      int StID = 1111;
      if(det==2) StID = 2222;
      std::vector<TrackFP420> collector;
      // 	    std::vector<TrackFP420> collector;
      collector.clear();
      
      // if ( trackMode_ == "TrackProducerMaxAmplitudeFP420") {
      //	 collector = finderParameters_->trackFinderMaxAmplitude(input); //std::vector<TrackFP420> collector;
      // }// if ( trackMode
      // else if (trackMode_ == "TrackProducerMaxAmplitude2FP420" ) {
      //	 collector = finderParameters_->trackFinderMaxAmplitude2(input); //
      //  }// if ( trackMode
      /*
	else if (trackMode_ == "TrackProducerVar1FP420" ) {
	collector = finderParameters_->trackFinderVar1(input); //
	}// if ( trackMode
	else if (trackMode_ == "TrackProducerVar2FP420" ) {
	collector = finderParameters_->trackFinderVar2(input); //
	}// if ( trackMode
      */
      if (trackMode_ == "TrackProducerSophisticatedFP420" ) {
	collector = finderParameters_->trackFinderSophisticated(input,det); //
      }// if ( trackMode
      
      
      //  else if (trackMode_ == "TrackProducer3DFP420" ) {
      //	 collector = finderParameters_->trackFinder3D(input); //
      // }// if ( trackMode
      
      if (collector.size()>0){
	TrackCollectionFP420::Range inputRange;
	inputRange.first = collector.begin();
	inputRange.second = collector.end();
	
	if ( first ) {
	  // use it only if TrackCollectionFP420 is the TrackCollection of one event, otherwise, do not use (loose 1st cl. of 1st event only)
	  first = false;
	  unsigned int StID0 = 0;
	  toutput->put(inputRange,StID0); // !!! put into adress 0 for detID which will not be used never
	} //if ( first ) 
	
	// !!! put                                        !!! put
	toutput->put(inputRange,StID);
	
	number_localelectroderechits += collector.size();
      } // if collector.size
    }//for det loop
    
    
    if (verbosity > 0) {
      std::cout << "FP420TrackMain: execution in mode " << trackMode_ << " generating " << number_localelectroderechits << " tracks in  " << number_detunits << " detectors" << std::endl; 
    }
    
    
    if (verbosity ==-29) {
      //     check of access to the collector:
      // loop over detunits
      for (int det=1; det<dn0; det++) {
	int StID = 1111;
	if(det==2) StID = 2222;
	std::vector<TrackFP420> collector;
	collector.clear();
	TrackCollectionFP420::Range outputRange;
	outputRange = toutput->get(StID);
	// fill output in collector vector (for may be sorting? or other checks)
	TrackCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	TrackCollectionFP420::ContainerIterator sort_end = outputRange.second;
	for ( ;sort_begin != sort_end; ++sort_begin ) {
	  collector.push_back(*sort_begin);
	} // for
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<"=======FP420TrackMain:check size = " << collector.size() << "  det = " << det << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	vector<TrackFP420>::const_iterator simHitIter = collector.begin();
	vector<TrackFP420>::const_iterator simHitIterEnd = collector.end();
	// loop in #tracks
	for (;simHitIter != simHitIterEnd; ++simHitIter) {
	  const TrackFP420 itrack = *simHitIter;
	  
	  std::cout << "FP420TrackMain:check: nclusterx = " << itrack.nclusterx() << "  nclustery = " << itrack.nclustery() << std::endl;
	  std::cout << "  ax = " << itrack.ax() << "  bx = " << itrack.bx() << std::endl;
	  std::cout << "  ay = " << itrack.ay() << "  by = " << itrack.by() << std::endl;
	  std::cout << " chi2x= " << itrack.chi2x() << " chi2y= " << itrack.chi2y() << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" =======================" << std::endl;
	}
	
	//==================================
	
	//     end of check of access to the strip collection
	std::cout <<"=======            FP420TrackMain:                    end of check     " << std::endl;
	
      }//for det
    }// if verbosity
    
    
    
    
  }// if ( validTrackerizer_
  
  
  
}
