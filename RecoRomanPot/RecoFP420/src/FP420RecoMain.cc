///////////////////////////////////////////////////////////////////////////////
// File: FP420RecoMain.cc
// Date: 12.2006
// Description: FP420RecoMain for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <iostream>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoRomanPot/RecoFP420/interface/FP420RecoMain.h"

#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/RecoProducerFP420.h"

// #include "CLHEP/Vector/LorentzVector.h"
// #include "CLHEP/Random/RandFlat.h"
#include <math.h>

using namespace std;


FP420RecoMain::FP420RecoMain(const edm::ParameterSet& conf):conf_(conf)  { 
  
  verbosity   = conf_.getUntrackedParameter<int>("VerbosityLevel");
  m_rpp420_f     = conf_.getParameter<double>("RP420f");//mm
  m_rpp420_b     = conf_.getParameter<double>("RP420b");//mm
  m_zreff          = conf_.getParameter<double>("zreff");//mm
  m_zrefb          = conf_.getParameter<double>("zrefb");//mm
  dn0                = conf_.getParameter<int>("NumberFP420Detectors");
  
  if (verbosity > 0) {
    std::cout << "FP420RecoMain constructor::" << std::endl;
    std::cout << "m_rpp420_f=" << m_rpp420_f << " m_rpp420_b=" << m_rpp420_b << std::endl;
    std::cout << "m_zreff=" << m_zreff << " m_zrefb=" <<  m_zrefb<< std::endl;
  }
    
  double eee1=11.;
  double eee2=12.;
//  zinibeg_ = (eee1-eee2)/2.;
  zinibeg_ = 0.;
//
    if (verbosity > 1) {
      std::cout << "FP420RecoMain constructor::" << std::endl;
      std::cout << " eee1=" << eee1 << " eee2=" << eee2 << " zinibeg =" << zinibeg_ << std::endl;
    }
  ///////////////////////////////////////////////////////////////////
      finderParameters_ = new RecoProducerFP420(conf_);
}

FP420RecoMain::~FP420RecoMain() {
  if ( finderParameters_ != 0 ) {
    delete finderParameters_;
  }
}



void FP420RecoMain::run(edm::Handle<TrackCollectionFP420> &input, std::auto_ptr<RecoCollectionFP420> &toutput, double VtxX, double VtxY, double VtxZ)  
{
  // initialization
  bool first = true;
  //  finderParameters_->clear();
  // finderParameters_->setIP( 0., 0., 0. );
  std::vector<TrackFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  
  // loop over detunits:
  for (int number_detunits=1; number_detunits<dn0; number_detunits++) {
    unsigned int StID = number_detunits;
    std::vector<RecoFP420> rcollector;
    int restracks = 10;// max # tracks
    rcollector.reserve(restracks); 
    rcollector.clear();
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<TrackFP420> collector;
    collector.clear();
    TrackCollectionFP420::Range outputRange;
    unsigned int StIDTrack = 1111;
    double  z420    = m_rpp420_f;
    double  zref1   = m_zreff;
    double  zinibeg = zinibeg_;
    double  VtxXcur = VtxX;
    double  VtxYcur = VtxY;
    double  VtxZcur = VtxZ;
    if(StID==2)  {
      StIDTrack = 2222;
      z420    = -m_rpp420_b;
      zref1   = -m_zrefb;
      zinibeg = -zinibeg_;
      //  VtxXcur = -VtxX;
      // VtxYcur = -VtxY;
      // VtxZcur = -VtxZ;
    }
    double z1 = z420+zinibeg-VtxZcur;
    double z2 = z420+zinibeg+zref1-VtxZcur;
    if (verbosity > 1) {
      std::cout << "FP420RecoMain: StIDTrack=" << StIDTrack << std::endl;
    }
    outputRange = input->get(StIDTrack);
    //
    // fill output in collector vector (for may be sorting? or other checks)
    //
    TrackCollectionFP420::ContainerIterator sort_begin = outputRange.first;
    TrackCollectionFP420::ContainerIterator sort_end = outputRange.second;
    //
    for ( ;sort_begin != sort_end; ++sort_begin ) {
      collector.push_back(*sort_begin);
    } // for  sort_begin
    if (verbosity > 1) {
      std::cout << "FP420RecoMain: track collector.size=" << collector.size() << std::endl;
    }
    std::vector<TrackFP420>::const_iterator simHitIter = collector.begin();
    std::vector<TrackFP420>::const_iterator simHitIterEnd = collector.end();
    for (;simHitIter != simHitIterEnd; ++simHitIter) {
      const TrackFP420 itrack = *simHitIter;
      double x1 = (    itrack.bx()*z1 + (itrack.ax()-VtxXcur)       )*1000.;//um
      double y1 = (    itrack.by()*z1 + (itrack.ay()-VtxYcur)       )*1000.;//um
      double x2 = (    itrack.bx()*z2 + (itrack.ax()-VtxXcur)    )*1000.;//um
      double y2 = (    itrack.by()*z2 + (itrack.ay()-VtxYcur)    )*1000.;//um
      /////////////////////////////////////////////////////////////////
	if (verbosity == -49) {
	  std::cout << "==================================================================== " << std::endl;
	  std::cout << "FP420RecoMain: StID= " << StID << std::endl;
	  std::cout << "input coord. in mm:  z1= " <<  z1  << std::endl;
	  std::cout << "input coord. in mm:  z2= " <<  z2  << std::endl;
	  std::cout << "input:  itrack.bx()= " <<  itrack.bx()  << std::endl;
	  std::cout << "input:  itrack.ax()= " <<  itrack.ax()  << std::endl;
	  std::cout << "input:  itrack.by()= " <<  itrack.by()  << std::endl;
	  std::cout << "input:  itrack.ay()= " <<  itrack.ay()  << std::endl;

	  std::cout << "input: in um X1noVtx= " <<  (itrack.bx()*(z420+zinibeg)+itrack.ax())*1000.  << std::endl;
	  std::cout << "input: in um Y1noVtx= " <<  (itrack.by()*(z420+zinibeg)+itrack.ay())*1000.  << std::endl;
	  std::cout << "input: in um X2noVtx= " <<  (itrack.bx()*(z420+zinibeg+zref1)+itrack.ax())*1000.  << std::endl;
	  std::cout << "input: in um Y2noVtx= " <<  (itrack.by()*(z420+zinibeg+zref1)+itrack.ay())*1000.  << std::endl;


	  std::cout << "input:  in mm VtxXcur= " << VtxXcur   << std::endl;
	  std::cout << "input:  in mm VtxYcur= " << VtxYcur   << std::endl;
	  std::cout << "input:  in mm VtxZcur= " << VtxZcur   << std::endl;
	  std::cout << "input coord. in um:  x1= " <<  x1  << std::endl;
	  std::cout << "input coord. in um:  y1= " <<  y1  << std::endl;
	  std::cout << "input coord. in um:  x2= " <<  x2  << std::endl;
	  std::cout << "input coord. in um:  y2= " <<  y2  << std::endl;
	}
	double zz1=fabs(z1);
	double zz2=fabs(z2);
	rcollector = finderParameters_->reconstruct(StID, x1,y1,x2,y2,zz1,zz2); // x1,y1,x2,y2 input coord. in um; z1, z2 in mm
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    if (verbosity > 1) {
      std::cout << "FP420RecoMain: track rcollector.size=" << rcollector.size() << std::endl;
    }
    if (rcollector.size()>0){
      RecoCollectionFP420::Range rinputRange;
      rinputRange.first = rcollector.begin();
      rinputRange.second = rcollector.end();
      
      if ( first ) {
	// use it only if RecoCollectionFP420 is the RecoCollection of one event, otherwise, do not use (loose 1st cl. of 1st event only)
	first = false;
	unsigned int StID0 = 0;
	toutput->put(rinputRange,StID0); // !!! put into adress 0 for detID which will not be used never
	if (verbosity > 1) {
	  std::cout << "FP420RecoMain: put(rinputRange,StID0)" << std::endl;
	}
      } //if ( first ) 
      
      // !!! put                                        !!! put
      toutput->put(rinputRange,StID);
      if (verbosity > 1) {
	std::cout << "FP420RecoMain: put(rinputRange,StID)" << std::endl;
      }
      
    } // if rcollector.size
    
  }// for loop over detunits
  





  if (verbosity > 1) {
    //     check of access to the zcollector:
    for (int number_detunits=1; number_detunits<dn0; number_detunits++) {
      int StID = number_detunits;
      if (verbosity > 1) {
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout << "FP420RecoMain:   re-new  StID= " <<  StID << std::endl;
      }
      std::vector<RecoFP420> zcollector;
      zcollector.clear();
      RecoCollectionFP420::Range zoutputRange;
      zoutputRange = toutput->get(StID);
      // fill output in zcollector vector (for may be sorting? or other checks)
      RecoCollectionFP420::ContainerIterator sort_begin = zoutputRange.first;
      RecoCollectionFP420::ContainerIterator sort_end = zoutputRange.second;
      for ( ;sort_begin != sort_end; ++sort_begin ) {
	zcollector.push_back(*sort_begin);
      } // for
      std::cout <<"=======FP420RecoMain:check of re-new zcollector size = " << zcollector.size() << std::endl;
      std::cout <<" ===" << std::endl;
      std::cout <<" ===" << std::endl;
      std::vector<RecoFP420>::const_iterator simHitIter = zcollector.begin();
      std::vector<RecoFP420>::const_iterator simHitIterEnd = zcollector.end();
      // loop in recoess
      for (;simHitIter != simHitIterEnd; ++simHitIter) {
	const RecoFP420 itrack = *simHitIter;
	//double e0, double x0, double y0, double tx0, double ty0, double q20, int direction     
	std::cout << "FP420RecoMain:check: direction = " << itrack.direction() << std::endl;
	std::cout << "  e0 = " << itrack.e0() << "  q20 = " << itrack.q20() << std::endl;
	std::cout << "  tx0 = " << itrack.tx0() << "  ty0 = " << itrack.ty0() << std::endl;
	std::cout << " x0= " << itrack.x0() << " y0= " << itrack.y0() << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<" =======================" << std::endl;
      }
    }
    
    //==================================
    
    //     end of check of access to the strip collection
    std::cout <<"=======            FP420RecoMain:                    end of check     " << std::endl;
    
  }//if (verbos
  






}
