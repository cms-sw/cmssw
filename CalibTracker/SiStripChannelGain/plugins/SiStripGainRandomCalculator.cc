// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripGainRandomCalculator
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainRandomCalculator.cc,v 1.2 2007/05/09 16:10:14 gbruno Exp $

#include "CalibTracker/SiStripChannelGain/interface/SiStripGainRandomCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"



using namespace cms;
using namespace std;


SiStripGainRandomCalculator::SiStripGainRandomCalculator(const edm::ParameterSet& iConfig) : SiStripGainCalculator::SiStripGainCalculator(iConfig){

  //   conf_ =  iConfig;
   edm::LogInfo("SiStripGainRandomCalculator::SiStripGainRandomCalculator");

   meanGain_=iConfig.getParameter<double>("MeanGain");
   sigmaGain_=iConfig.getParameter<double>("SigmaGain");
   minimumPosValue_=iConfig.getParameter<double>("MinPositiveGain");
   printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);

}


SiStripGainRandomCalculator::~SiStripGainRandomCalculator()
{
   edm::LogInfo("SiStripGainRandomCalculator::~SiStripGainRandomCalculator");
}



void SiStripGainRandomCalculator::algoBeginRun(const edm::Run &, const edm::EventSetup& iSetup){


  edm::LogInfo("SiStripGainRandomCalculator::beginRun");


  detid_apvs.clear();


  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );     
  edm::LogInfo("SiStripGainCalculator") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
  
  for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
  
    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
      uint32_t detid=((*it)->geographicalId()).rawId();            
      const StripTopology& p = dynamic_cast<StripGeomDetUnit*>((*it))->specificTopology();
      unsigned short NAPVPairs = p.nstrips()/128;
      if(NAPVPairs<1 || NAPVPairs>6 ) {
	edm::LogError("SiStripGainCalculator")<<" Problem with Number of strips in detector.. "<< p.nstrips() <<" Exiting program"<<endl;
	exit(1);
      }
      detid_apvs.push_back( pair<uint32_t,unsigned short>(detid,NAPVPairs) );
      if (printdebug_)
	edm::LogInfo("SiStripGainCalculator")<< "detid " << detid << " apvs " << NAPVPairs;
    }
  }
}


SiStripApvGain * SiStripGainRandomCalculator::getNewObject() {

  std::cout<<"SiStripGainRandomCalculator::getNewObject called"<<std::endl;

  SiStripApvGain * obj = new SiStripApvGain();

  for(std::vector< pair<uint32_t,unsigned short> >::const_iterator it = detid_apvs.begin(); it != detid_apvs.end(); it++){
    //Generate Noise for det detid
    std::vector<float> theSiStripVector;
    for(unsigned short j=0; j<it->second; j++){
      float gain = RandGauss::shoot(meanGain_, sigmaGain_);
      if(gain<=minimumPosValue_) gain=minimumPosValue_;
      if (printdebug_)
	edm::LogInfo("SiStripGainCalculator") << "detid " << it->first << " \t"
					      << " apv " << j << " \t"
					      << gain    << " \t" 
					      << std::endl; 	    
      theSiStripVector.push_back(gain);
    }
    
    
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("SiStripGainCalculator")<<"[SiStripGainCalculator::beginJob] detid already exists"<<std::endl;

  }
  
  return obj;

}


