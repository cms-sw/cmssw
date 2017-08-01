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
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"

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
    TrackerMap of SiStripApvGains (average gain per detid)
  *************************************************/
  class SiStripApvGainsAverageTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsAverageTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of average SiStripGains" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain average per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	float sumOfGains=0;
	float nAPVsPerModule=0.;
	for(int it=0;it<range.second-range.first;it++){
	  nAPVsPerModule+=1;
	  sumOfGains+=payload->getApvGain(it,range);
	} // loop over APVs
	// fill the tracker map taking the average gain on a single DetId
	tmap->fill(d,(sumOfGains/nAPVsPerModule));
      } // loop over detIds

      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName.c_str());

      return true;
    }
  };

    /************************************************
    TrackerMap of SiStripApvGains (average gain per detid)
  *************************************************/
  class SiStripApvGainsDefaultTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsDefaultTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of SiStripGains to default" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain to default per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      int totalDefaultAPVs=0;
      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	float sumOfGains=0;
	float nAPVsPerModule=0.;
	int countDefaults=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPVsPerModule+=1;
	  sumOfGains+=payload->getApvGain(it,range);
	  if(payload->getApvGain(it,range)==1) countDefaults++;
	} // loop over APVs
	// fill the tracker map taking the average gain on a single DetId
	if(countDefaults>0.) tmap->fill(d,countDefaults);
	totalDefaultAPVs+=countDefaults;
      } // loop over detIds
      
      std::cout<<"there are "<< totalDefaultAPVs << "APVs with default value (=1)" << std::endl;
      
      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName.c_str());

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (ratio with previous gain per detid)
  *************************************************/
  class SiStripApvGainsRatioWithPreviousIOVTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsRatioWithPreviousIOVTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of ratio of SiStripGains with previous IOV" ){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      
      std::vector<std::tuple<cond::Time_t,cond::Hash> > sorted_iovs = iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
	  return std::get<0>(t1) < std::get<0>(t2);
	});
      
      auto firstiov  = sorted_iovs.front();
      auto lastiov   = sorted_iovs.back();
      
      std::shared_ptr<SiStripApvGain> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::string titleMap = "SiStrip APV Gain ratio per module average (IOV: ";

      titleMap+=std::to_string(std::get<0>(firstiov));
      titleMap+="/ IOV:";
      titleMap+=std::to_string(std::get<0>(lastiov));
      titleMap+=")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);

      std::map<uint32_t,float> lastmap,firstmap;

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      // cache the last IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  Gain+=last_payload->getApvGain(it,range);
	} // loop over APVs
	lastmap[d]=(Gain/nAPV);
      } // loop over detIds
      
      detid.clear();
      
      first_payload->getDetIds(detid);
      
      // cache the first IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=first_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  Gain+=first_payload->getApvGain(it,range);
	} // loop over APVs
	firstmap[d]=(Gain/nAPV);
      } // loop over detIds
      

      std::map<uint32_t,float> cachedRatio; 
      for(const auto &d : detid){
	float ratio = firstmap[d]/lastmap[d];
	tmap->fill(d,ratio);
	cachedRatio[d] = ratio;
      }
    
      //=========================
      auto range = getTheRange(cachedRatio);

      std::string fileName(m_imageFileName);
      tmap->save(true,range.first,range.second,fileName.c_str());

      return true;
    }
  };

  /************************************************
   TrackerMap of SiStripApvGains (ratio for largest deviation with previous gain per detid)
  *************************************************/
  class SiStripApvGainsRatioMaxDeviationWithPreviousIOVTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsRatioMaxDeviationWithPreviousIOVTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of ratio (for largest deviation) of SiStripGains with previous IOV" ){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      
      std::vector<std::tuple<cond::Time_t,cond::Hash> > sorted_iovs = iovs;

      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
	  return std::get<0>(t1) < std::get<0>(t2);
	});
      
      auto firstiov  = sorted_iovs.front();
      auto lastiov   = sorted_iovs.back();
      
      std::shared_ptr<SiStripApvGain> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::string titleMap = "SiStrip APV Gain ratio for largest deviation per module (IOV: ";

      titleMap+=std::to_string(std::get<0>(firstiov));
      titleMap+="/ IOV:";
      titleMap+=std::to_string(std::get<0>(lastiov));
      titleMap+=")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);

      std::map<std::pair<uint32_t,int>,float> lastmap,firstmap;

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);
      
      // cache the last IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  Gain+=last_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  lastmap[index]=(Gain/nAPV);
	} // loop over APVs
      } // loop over detIds
      
      detid.clear();
      
      first_payload->getDetIds(detid);
      
      // cache the first IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=first_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  Gain+=first_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  firstmap[index]=(Gain/nAPV);
	} // loop over APVs
      } // loop over detIds
      
      // find the largest deviation
      std::map<uint32_t,float> cachedRatio; 

      for(const auto &item : firstmap ){
	
	// packed index (detid,APV)
	auto index   = item.first;
	auto mod     = item.first.first;
	
	float ratio = firstmap[index]/lastmap[index];
	// if we have already cached something
	if(cachedRatio[mod]){
	  if(std::abs(cachedRatio[mod])>std::abs(ratio)){
	    cachedRatio[mod]=ratio;
	  }
	} else {
	  cachedRatio[mod]=ratio;
	}
      }

      for (const auto &element : cachedRatio){
	tmap->fill(element.first,element.second);
      }

      // get the range of the TrackerMap (saturate at +/-2 std deviations)
      auto range = getTheRange(cachedRatio);
      
      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,range.first,range.second,fileName.c_str());

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (maximum gain per detid)
  *************************************************/
  class SiStripApvGainsMaximumTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsMaximumTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of SiStripAPVGains (maximum per DetId)" ){
      setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain maximum per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	float theMaxGain=0;
	for(int it=0;it<range.second-range.first;it++){
	  
	  float currentGain = payload->getApvGain(it,range);
	  if(currentGain > theMaxGain){
	    theMaxGain=currentGain;
	  }
	} // loop over APVs
	// fill the tracker map taking the average gain on a single DetId
	tmap->fill(d,theMaxGain);
      } // loop over detIds

      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName.c_str());

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (minimum gain per detid)
  *************************************************/
  class SiStripApvGainsMinimumTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsMinimumTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of SiStripAPVGains (minimum per DetId)" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain minumum per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap.c_str());
      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);
      
      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	float theMinGain=999.;
	for(int it=0;it<range.second-range.first;it++){
	  float currentGain = payload->getApvGain(it,range);
	  if(currentGain < theMinGain){
	    theMinGain=currentGain;
	  }
	} // loop over APVs
	// fill the tracker map taking the average gain on a single DetId
	tmap->fill(d,theMinGain);
      } // loop over detIds

      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName.c_str());

      return true;
    }
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
	if(subid!=StripSubdetector::TIB) continue;
	
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
	if(subid!=StripSubdetector::TOB) continue;

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
	if(subid!=StripSubdetector::TID) continue;
	
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
	if(subid!=StripSubdetector::TEC) continue;
	
	SiStripApvGain::Range range=payload.getRange(d);
	for(int it=0;it<range.second-range.first;it++){
	  nAPVs+=1;
	  sumOfGains+=payload.getApvGain(it,range);
	} // loop over APVs
      } // loop over detIds

      return sumOfGains/nAPVs;

    } // payload
  };

  class SiStripApvGainsTest : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvGainsTest() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains test",
										"SiStripApv Gains test", 10,0.0,10.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  SiStripDetSummary summaryGain;

	  for (const auto & d : detid) {
	    SiStripApvGain::Range range=payload->getRange(d);
	    for( int it=0; it < range.second - range.first; ++it ) {
	      summaryGain.add(d,payload->getApvGain(it, range));
	      fillWithValue(payload->getApvGain(it,range));
	    } 
	  }
	  std::map<unsigned int, SiStripDetSummary::Values> map = summaryGain.getCounts();

	  //	  myPrintSummary(map);

	  //std::cout<<"map size: "<<map.size()<< std::endl;
	  std::stringstream ss;
	  ss << "Summary of gain values:" << std::endl;
	  summaryGain.print(ss, true);
	  std::cout<<ss.str()<<std::endl;
	  
	}// payload
      }// iovs
      return true;
    }// fill
  };

  class SiStripApvGainsComparator : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsComparator () : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains Comparison" ){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      
      std::vector<std::tuple<cond::Time_t,cond::Hash> > sorted_iovs = iovs;
       
      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
	  return std::get<0>(t1) < std::get<0>(t2);
	});
      
      auto firstiov  = sorted_iovs.front();
      auto lastiov   = sorted_iovs.back();
      
      std::shared_ptr<SiStripApvGain> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::string lastIOVsince  = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      std::map<std::pair<uint32_t,int>,float> lastmap,firstmap;

      // loop on the last payload
      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for( int it=0; it < range.second - range.first; ++it ) {
	  nAPV+=1;
	  Gain=last_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  lastmap[index]=Gain;
	} // end loop on APVs
      } // end loop on detids

      detid.clear();
      first_payload->getDetIds(detid);

      // loop on the first payload
      for (const auto & d : detid) {
	SiStripApvGain::Range range=first_payload->getRange(d);
	float Gain=0;
	float nAPV=0;
	for( int it=0; it < range.second - range.first; ++it ) {
	  nAPV+=1;
	  Gain=first_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  firstmap[index]=Gain;
	} // end loop on APVs
      }  // end loop on detids
      
      TCanvas canvas("Payload comparison","payload comparison",1400,1000); 
      canvas.Divide(2,1);

      std::map<std::string,TH1F*> ratios;
      std::map<std::string,TH2F*> scatters;
      std::map<std::string,int> colormap;
      std::map<std::string,int> markermap;
      colormap["TIB"] = kRed;       markermap["TIB"] = kFullCircle;           
      colormap["TOB"] = kGreen;	    markermap["TOB"] = kFullTriangleUp;
      colormap["TID"] = kBlack;	    markermap["TID"] = kFullSquare;
      colormap["TEC"] = kBlue; 	    markermap["TEC"] = kFullTriangleDown; 

      std::vector<std::string> parts = {"TEC","TOB","TIB","TID"};
      
      for ( const auto &part : parts){
	ratios[part]   = new TH1F(Form("hRatio_%s",part.c_str()),Form("Gains ratio IOV: %s/ IOV: %s ;New Gain (%s) / Previous Gain (%s);Number of APV",lastIOVsince.c_str(),firstIOVsince.c_str(),lastIOVsince.c_str(),firstIOVsince.c_str()),100,0.,2.);
	scatters[part] = new TH2F(Form("hScatter_%s",part.c_str()),Form("new Gain (%s) vs previous Gain (%s);Previous Gain (%s);New Gain (%s)",lastIOVsince.c_str(),firstIOVsince.c_str(),firstIOVsince.c_str(),lastIOVsince.c_str()),100,0.5,1.8,100,0.5,1.8);
      }
      
      // now loop on the cached maps
      for(const auto &item : firstmap ) {
	
	// packed index (detid,APV)
	auto index   = item.first;
	auto mod     = item.first.first;

	int subid = DetId(mod).subdetId();
	float ratio = firstmap[index]/lastmap[index];

	if(subid==StripSubdetector::TIB){
	  ratios["TIB"]->Fill(ratio);
	  scatters["TIB"]->Fill(lastmap[index],firstmap[index]);
	}

	if(subid==StripSubdetector::TOB){
	  ratios["TOB"]->Fill(ratio);
	  scatters["TOB"]->Fill(lastmap[index],firstmap[index]);
	}

	if(subid==StripSubdetector::TID){
	  ratios["TID"]->Fill(ratio);
	  scatters["TID"]->Fill(lastmap[index],firstmap[index]);
	}

	if(subid==StripSubdetector::TEC){
	  ratios["TEC"]->Fill(ratio);
	  scatters["TEC"]->Fill(lastmap[index],firstmap[index]);
	}

      }

      auto legend = new TLegend(0.60,0.8,0.9,0.95);
      legend->SetTextSize(0.05);
      canvas.cd(1)->SetLogy(); 
      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetLeftMargin(0.13);
      canvas.cd(1)->SetRightMargin(0.08);

      for ( const auto &part : parts){
	makeNicePlotStyle(ratios[part]);
	ratios[part]->SetStats(false);
	ratios[part]->SetLineWidth(2);
	ratios[part]->SetLineColor(colormap[part]);
	if(part =="TEC")
	  ratios[part]->Draw();
	else
	  ratios[part]->Draw("same");
	legend->AddEntry(ratios[part],part.c_str(),"L");
      }

      legend->Draw("same");
      DrawStatBox(ratios,colormap,parts);
       
      auto legend2 = new TLegend(0.60,0.8,0.9,0.95);
      legend2->SetTextSize(0.05);
      canvas.cd(2);
      canvas.cd(2)->SetTopMargin(0.05);
      canvas.cd(2)->SetLeftMargin(0.13);
      canvas.cd(2)->SetRightMargin(0.08);

      for ( const auto &part : parts){
	makeNicePlotStyle(scatters[part]);
	scatters[part]->SetStats(false);
	scatters[part]->SetMarkerColor(colormap[part]);
	scatters[part]->SetMarkerStyle(markermap[part]);
	scatters[part]->SetMarkerSize(0.5);
	if(part =="TEC")
	  scatters[part]->Draw("P");
	else
	  scatters[part]->Draw("Psame");
	legend2->AddEntry(scatters[part],part.c_str(),"P");
      }

      legend2->Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());
      
     
      return true;

    }
  };

  class SiStripApvGainsComparatorByPartition : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsComparatorByPartition() : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains Comparison By Partition" ){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      
      std::vector<std::tuple<cond::Time_t,cond::Hash> > sorted_iovs = iovs;
       
      // make absolute sure the IOVs are sortd by since
      std::sort(begin(sorted_iovs), end(sorted_iovs), [](auto const &t1, auto const &t2) {
	  return std::get<0>(t1) < std::get<0>(t2);
	});
      
      auto firstiov  = sorted_iovs.front();
      auto lastiov   = sorted_iovs.back();
      
      std::shared_ptr<SiStripApvGain> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<SiStripApvGain> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);

      SiStripDetSummary summaryLastGain;

      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	for( int it=0; it < range.second - range.first; ++it ) {
	  summaryLastGain.add(d,last_payload->getApvGain(it, range));
	}
      } 

      SiStripDetSummary summaryFirstGain;

      for (const auto & d : detid) {
	SiStripApvGain::Range range=first_payload->getRange(d);
	for( int it=0; it < range.second - range.first; ++it ) {
	  summaryFirstGain.add(d,first_payload->getApvGain(it, range));
	}
      } 

      std::map<unsigned int, SiStripDetSummary::Values> firstmap = summaryFirstGain.getCounts();
      std::map<unsigned int, SiStripDetSummary::Values> lastmap = summaryLastGain.getCounts();
      //=========================
      
      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.cd();

      TH1F* hfirst = new TH1F("byPartition1","SiStrip first Gain average by partition;; average SiStrip Gain",firstmap.size(),0.,firstmap.size());
      TH1F* hlast  = new TH1F("byPartition2","SiStrip last Gain average by partition;; average SiStrip Gain",lastmap.size(),0.,lastmap.size());
      
      hfirst->SetStats(false);
      hlast->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin=0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : lastmap){
	iBin++;
	int count   = element.second.count;
	double mean = (element.second.mean)/count;
	double rms  = (element.second.rms)/count - mean*mean;

	if(rms <= 0)
	  rms = 0;
	else
	  rms = sqrt(rms);

	if(currentDetector.empty()) currentDetector="TIB";
	
	switch ((element.first)/1000) 
	  {
	  case 1:
	    detector = "TIB";
	    break;
	  case 2:
	    detector = "TOB";
	    break;
	  case 3:
	    detector = "TEC";
	    break;
	  case 4:
	    detector = "TID";
	    break;
	  }

	hlast->SetBinContent(iBin,mean);
	hlast->GetXaxis()->SetBinLabel(iBin,regionType(element.first));
	hlast->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
	  std::cout<<"detector has changed from "<<currentDetector<<" to "<<detector<<std::endl;
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      // reset the count
      iBin=0;

      for (const auto &element : firstmap){
	iBin++;
	int count   = element.second.count;
	double mean = (element.second.mean)/count;
	double rms  = (element.second.rms)/count - mean*mean;

	if(rms <= 0)
	  rms = 0;
	else
	  rms = sqrt(rms);

	hfirst->SetBinContent(iBin,mean);
	hfirst->GetXaxis()->SetBinLabel(iBin,regionType(element.first));
	hfirst->GetXaxis()->LabelsOption("v");	
      }

      hlast->SetMarkerStyle(20);
      hlast->SetMarkerSize(1);
      hlast->Draw("HIST");
      hlast->Draw("Psame");

      hfirst->SetMarkerStyle(18);
      hfirst->SetMarkerSize(1);
      hfirst->SetLineColor(kBlue);
      hfirst->SetMarkerColor(kBlue);
      hfirst->Draw("HISTsame");
      hfirst->Draw("Psame");

      canvas.Update();
      canvas.cd();

      for (const auto & line : boundaries){
	TLine* l = new TLine(hfirst->GetBinLowEdge(line),canvas.cd()->GetUymin(),hfirst->GetBinLowEdge(line),canvas.cd()->GetUymax());
	l->SetLineWidth(1);
	l->SetLineStyle(9);
	l->SetLineColor(2);
	l->Draw("same");
      }
      
      auto legend = new TLegend(0.70,0.8,0.95,0.9);
      legend->SetHeader("Comparison","C"); // option "C" allows to center the header
      legend->AddEntry(hfirst,("IOV: "+std::to_string(std::get<0>(firstiov))).c_str(),"PL");
      legend->AddEntry(hlast ,("IOV: "+std::to_string(std::get<0>(lastiov))).c_str(),"PL");
      legend->Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  class SiStripApvGainsByPartition : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsByPartition() : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains By Partition" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ){
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryGain;

      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	for( int it=0; it < range.second - range.first; ++it ) {
	  summaryGain.add(d,payload->getApvGain(it, range));
	}
      } 

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryGain.getCounts();
      //=========================
      
      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.cd();
      TH1F* h1 = new TH1F("byPartition","SiStrip Gain average by partition;; average SiStrip Gain",map.size(),0.,map.size());
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin=0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : map){
	iBin++;
	int count   = element.second.count;
	double mean = (element.second.mean)/count;
	double rms  = (element.second.rms)/count - mean*mean;

	if(rms <= 0)
	  rms = 0;
	else
	  rms = sqrt(rms);

	if(currentDetector.empty()) currentDetector="TIB";
	
	switch ((element.first)/1000) 
	  {
	  case 1:
	    detector = "TIB";
	    break;
	  case 2:
	    detector = "TOB";
	    break;
	  case 3:
	    detector = "TEC";
	    break;
	  case 4:
	    detector = "TID";
	    break;
	  }

	h1->SetBinContent(iBin,mean);
	h1->GetXaxis()->SetBinLabel(iBin,regionType(element.first));
	h1->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
	  std::cout<<"detector has changed from "<<currentDetector<<" to "<<detector<<std::endl;
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");

      canvas.Update();
      canvas.cd();

      for (const auto & line : boundaries){
	TLine* l = new TLine(h1->GetBinLowEdge(line),canvas.cd()->GetUymin(),h1->GetBinLowEdge(line),canvas.cd()->GetUymax());
	l->SetLineWidth(1);
	l->SetLineStyle(9);
	l->SetLineColor(2);
	l->Draw("same");
      }
      
      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripApvGain){
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparator);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsAverageTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsDefaultTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMaximumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsMinimumTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioMaxDeviationWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);
}
