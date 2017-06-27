#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <memory>
#include <sstream>

namespace {

  /************************************************
    1d histogram of SiStripApvGains of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvGainsValue : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvGainsValue() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains values",
										 "SiStripApv Gains values", 200,0.0,2.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  for (const auto & d : detid) {
	    SiStripApvGain::Range range=payload->getRange(d);
	    for(int it=0;it<range.second-range.first;it++){

	      // to be used to fill the histogram
	      fillWithValue(payload->getApvGain(it,range));
	      
	    }// loop over APVs
	  } // loop over detIds
	}// payload
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    time history histogram of SiStripApvGains 
  *************************************************/

  class SiStripApvGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average","average Strip APV gain value"){}
    virtual ~SiStripApvGainByRunMeans() = default;

    float getFromPayload( SiStripApvGain& payload ){
     
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      float nAPVs=0;
      float sumOfGains=0;

      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;      
    } // payload
  };

  /************************************************
    time history histogram of TIB SiStripApvGains 
  *************************************************/

  class SiStripApvTIBGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvTIBGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average","average Tracker Inner Barrel APV gain value"){}
    virtual ~SiStripApvTIBGainByRunMeans() = default;

    float getFromPayload( SiStripApvGain& payload ){
     
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      float nAPVs=0;
      float sumOfGains=0;

      for (const auto & d : detid) {

	int subid = DetId(d).subdetId();
	if(subid!=3) continue;
	
	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;

    } // payload
  };

  /************************************************
    time history histogram of TOB SiStripApvGains 
  *************************************************/

  class SiStripApvTOBGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvTOBGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average","average Tracker Outer Barrel gain value"){}
    virtual ~SiStripApvTOBGainByRunMeans() = default;

    float getFromPayload( SiStripApvGain& payload ){
     
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      float nAPVs=0;
      float sumOfGains=0;
      
      for (const auto & d : detid) {

	int subid = DetId(d).subdetId();
	if(subid!=5) continue;

	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;

    } // payload
  };

  /************************************************
    time history histogram of TID SiStripApvGains 
  *************************************************/

  class SiStripApvTIDGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvTIDGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average","average Tracker Inner Disks APV gain value"){}
    virtual ~SiStripApvTIDGainByRunMeans() = default;

    float getFromPayload( SiStripApvGain& payload ){
     
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      float nAPVs=0;
      float sumOfGains=0;
      for (const auto & d : detid) {
	
	int subid = DetId(d).subdetId();
	if(subid!=4) continue;
	
	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;

    } // payload
  };

  /************************************************
    time history histogram of TEC SiStripApvGains 
  *************************************************/

  class SiStripApvTECGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvTECGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average in TEC","average Tracker Endcaps APV gain value"){}
    virtual ~SiStripApvTECGainByRunMeans() = default;

    float getFromPayload( SiStripApvGain& payload ){
     
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      float nAPVs=0;
      float sumOfGains=0;

      for (const auto & d : detid) {

	int subid = DetId(d).subdetId();
	if(subid!=6) continue;
	
	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;

    } // payload
  };

    
} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripApvGain){
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);
}
