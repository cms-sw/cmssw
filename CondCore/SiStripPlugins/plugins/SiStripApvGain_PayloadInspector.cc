#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

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
										 "SiStripApv Gains values", 200,0.2,1.8){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  for (size_t id=0;id<detid.size();id++){
	    SiStripApvGain::Range range=payload->getRange(detid[id]);
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
    1d histogram of means of SiStripApvGains
    for Tracker Barrel of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvBarrelGainsByLayer : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvBarrelGainsByLayer() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains averages by Barrel layer",
											 "Barrel layer (0-3: TIB), (4-9: TOB)",10,0,10,"average SiStripApv Gain"){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByLayer;

	  for (size_t id=0;id<detid.size();id++){

	    int subid = int((detid[id]>>25) & 0x7);	   
	    int layer = int((detid[id]>>14) & 0x7);
	    if(subid!=3 && subid!=5) continue;
	    if(subid==5){
	      // layers of TOB start at 5th bin
	      layer+=4;
	    }

	    SiStripApvGain::Range range=payload->getRange(detid[id]);
	    for(int it=0;it<range.second-range.first;it++){
	      sumOfGainsByLayer[layer].first+=payload->getApvGain(it,range);
	      sumOfGainsByLayer[layer].second+=1.;
	    }// loop over APVs
	  } // loop over detIds

	  // loop on the map to fill the plot
	  for (auto& data : sumOfGainsByLayer){
	    //std::cout<<"layer: "<<data.first << " payload:"<< (data.second.first/data.second.second) <<std::endl;
	    fillWithBinAndValue(data.first-1,(data.second.first/data.second.second));
	  }
	  
	}// payload
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    1d histogram of means of SiStripApvGains
    for Tracker Endcaps (minus side) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvEndcapMinusGainsByDisk : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvEndcapMinusGainsByDisk() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains averages by Endcap (minus) disk",
											     "Endcap (minus) disk (0-2: TID), (3-11: TEC)",12,0,12,"average SiStripApv Gain"){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByDisk;

	  for (size_t id=0;id<detid.size();id++){

	    int disk=-1;
	    int side=-1;
	    int subid = int((detid[id]>>25) & 0x7);
	    if(subid!=4 && subid!=6) continue;
	    
	    // TID https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiStripDetId/interface/TIDDetId.h#L112

	    if(subid==4){

	      side = int((detid[id]>>13) & 0x3);
	      disk = int((detid[id]>>11) & 0x3); 
	    } else {

	    // TEC  https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiStripDetId/interface/TECDetId.h#L122

	      side = int((detid[id]>>18) & 0x3);
	      disk = int((detid[id]>>14) & 0xF);
	      
	      // disks of TEC start at 4th bin
	      disk+=3;
	    }

	    // only negative side
	    if(side!=1) continue;

	    SiStripApvGain::Range range=payload->getRange(detid[id]);
	    for(int it=0;it<range.second-range.first;it++){
	      sumOfGainsByDisk[disk].first+=payload->getApvGain(it,range);
	      sumOfGainsByDisk[disk].second+=1.;
	    }// loop over APVs
	  } // loop over detIds

	  // loop on the map to fill the plot
	  for (auto& data : sumOfGainsByDisk){
	    fillWithBinAndValue(data.first-1,(data.second.first/data.second.second));
	  }
	  
	}// payload
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    1d histogram of means of SiStripApvGains
    for Tracker Endcaps (plus side) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripApvEndcapPlusGainsByDisk : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvEndcapPlusGainsByDisk() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains averages by Endcap (plus) disk",
											    "Endcap (plus) disk (0-2: TID), (3-11: TEC)",12,0,12,"average SiStripApv Gain"){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByDisk;

	  for (size_t id=0;id<detid.size();id++){

	    int disk=-1;
	    int side=-1;
	    int subid = int((detid[id]>>25) & 0x7);
	    if(subid!=4 && subid!=6) continue;

	    // TID https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiStripDetId/interface/TIDDetId.h#L112

	    if(subid==4){
	      side = int((detid[id]>>13) & 0x3);
	      disk = int((detid[id]>>11) & 0x3); 
	    } else {

	    // TEC https://github.com/cms-sw/cmssw/blob/master/DataFormats/SiStripDetId/interface/TECDetId.h#L122
	      
	      side = int((detid[id]>>18) & 0x3);
	      disk = int((detid[id]>>14) & 0xF); 
	      
	      // disks of TEC start at 4th bin
	      disk+=3;
	    }
	    
	    // only positive side
	    if(side!=2) continue;

	    SiStripApvGain::Range range=payload->getRange(detid[id]);
	    for(int it=0;it<range.second-range.first;it++){
	      sumOfGainsByDisk[disk].first+=payload->getApvGain(it,range);
	      sumOfGainsByDisk[disk].second+=1.;
	    }// loop over APVs
	  } // loop over detIds

	  // loop on the map to fill the plot
	  for (auto& data : sumOfGainsByDisk){
	    fillWithBinAndValue(data.first-1,(data.second.first/data.second.second));
	  }
	  
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

      for (size_t id=0;id<detid.size();id++){
	SiStripApvGain::Range range=payload.getRange(detid[id]);
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

      for (size_t id=0;id<detid.size();id++){

	int subid = int((detid[id]>>25) & 0x7);
	if(subid!=3) continue;
	
	SiStripApvGain::Range range=payload.getRange(detid[id]);
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

      for (size_t id=0;id<detid.size();id++){

	int subid = int((detid[id]>>25) & 0x7);
	if(subid!=5) continue;
	
	SiStripApvGain::Range range=payload.getRange(detid[id]);
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

      for (size_t id=0;id<detid.size();id++){

	int subid = int((detid[id]>>25) & 0x7);
	if(subid!=4) continue;
	
	SiStripApvGain::Range range=payload.getRange(detid[id]);
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

      for (size_t id=0;id<detid.size();id++){

	int subid = int((detid[id]>>25) & 0x7);
	if(subid!=6) continue;
	
	SiStripApvGain::Range range=payload.getRange(detid[id]);
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
  PAYLOAD_INSPECTOR_CLASS(SiStripApvBarrelGainsByLayer);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapMinusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvEndcapPlusGainsByDisk);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);
}
