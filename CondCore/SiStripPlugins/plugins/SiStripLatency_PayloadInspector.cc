/*!
  \file SiStripApvGains_PayloadInspector
  \Payload Inspector Plugin for SiStrip Gain
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/07/02 17:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT 
#include "TProfile.h"
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
    1d histogram of SiStripApvGains of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripLatencyValue : public cond::payloadInspector::Histogram1D<SiStripLatency> {
    
  public:
    SiStripLatencyValue() : cond::payloadInspector::Histogram1D<SiStripLatency>("SiStripLatency values",
										 "SiStripLatency values", 200,0.0,2.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripLatency> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<SiStripLatency::Latency> lat = payload->allLatencyAndModes();
	  
	  for (const auto & l : lat) {

	    std::cout<<"APV"<<((l.detIdAndApv)&7)<<"detID"<<((l.detIdAndApv)>>3)<<std::endl;
	    std::cout<<(int)l.latency<<std::endl;
	    std::cout<<(int)l.mode<<std::endl<<std::endl;
	    fillWithValue(1.);
	    //SiStripLatency::Range range=payload->getRange(d);
	    //for(int it=0;it<range.second-range.first;it++){

	      // to be used to fill the histogram
	      //fillWithValue(payload->getApvGain(it,range));
	      
	    // }// loop over APVs
	  } // loop over detIds
	}// payload
      }// iovs
      return true;
    }// fill
  };
  

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripLatency){
  PAYLOAD_INSPECTOR_CLASS(SiStripLatencyValue);
  /* PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparator);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValuesComparator);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioComparatorByRegion);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvBarrelGainsByLayer);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteBarrelGainsByLayer);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapMinusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapPlusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteEndcapMinusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvAbsoluteEndcapPlusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAverageTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsDefaultTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaximumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMinimumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAvgDeviationRatio1sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAvgDeviationRatio2sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAvgDeviationRatio3sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaxDeviationRatio1sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaxDeviationRatio2sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaxDeviationRatio3sigmaTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);*/
}
