/*!
  \file SiStripBadStrip_PayloadInspector
  \Payload Inspector Plugin for SiStrip Bad Strip
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/08/14 14:37:22 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h" 

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT 
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  /************************************************
    test class
  *************************************************/

  class SiStripBadStripTest : public cond::payloadInspector::Histogram1D<SiStripBadStrip> {
    
  public:
    SiStripBadStripTest() : cond::payloadInspector::Histogram1D<SiStripBadStrip>("SiStrip Bad Strip test",
										 "SiStrip Bad Strip test", 10,0.0,10.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripBadStrip> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  fillWithValue(1.);
	 
	  std::stringstream ss;
	  ss << "Summary of bad strips:" << std::endl;

	  //payload->printDebug(ss);
	  //payload->printSummary(ss);
	  //std::cout<<ss.str()<<std::endl;

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);

	  for (const auto & d : detid) {
	    SiStripBadStrip::Range range=payload->getRange(d);
	    for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	      ss << "DetId="<< d << " Strip=" << payload->decode(*badStrip).firstStrip <<":"<< payload->decode(*badStrip).range << " flag="<< payload->decode(*badStrip).flag << std::endl;
	    }
	  }
	  
	  std::cout<<ss.str()<<std::endl;
 
	}// payload
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    TrackerMap of SiStripBadStrip (bad strip per detid)
  *************************************************/
  class SiStripBadModuleTrackerMap : public cond::payloadInspector::PlotImage<SiStripBadStrip> {
  public:
    SiStripBadModuleTrackerMap() : cond::payloadInspector::PlotImage<SiStripBadStrip>( "Tracker Map of SiStripAPVGains (minimum per DetId)" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "Module with at least a bad Strip (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripBadStrips"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      for (const auto & d : detid) {
	tmap->fill(d,1);
      } // loop over detIds
      
      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripBadStrip (bad strips fraction)
  *************************************************/
  class SiStripBadStripFractionTrackerMap : public cond::payloadInspector::PlotImage<SiStripBadStrip> {
  public:
    SiStripBadStripFractionTrackerMap() : cond::payloadInspector::PlotImage<SiStripBadStrip>( "Tracker Map of SiStripAPVGains (minimum per DetId)" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload( std::get<1>(iov) );

      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

      std::string titleMap = "Fraction of bad Strips per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripBadStrips"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload->getRange(d);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStripsPerDetId[d]+= payload->decode(*badStrip).range;
	  //ss << "DetId="<< d << " Strip=" << payload->decode(*badStrip).firstStrip <<":"<< payload->decode(*badStrip).range << " flag="<< payload->decode(*badStrip).flag << std::endl;
	}
	float fraction = badStripsPerDetId[d]/(128.*reader->getNumberOfApvsAndStripLength(d).first);
	tmap->fill(d,fraction);
      } // loop over detIds
      
      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      delete reader;
      return true;
    }
  };


} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripBadStrip){
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadModuleTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionTrackerMap);
}
