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
#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h" 

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
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
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
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByLayer;

	  for (const auto & d : detid) {
	    
	    int subid = DetId(d).subdetId();
	    int layer(-1); 
	    if(subid!=StripSubdetector::TIB && subid!=StripSubdetector::TOB) continue;
	    if(subid==StripSubdetector::TIB){
	      layer = tTopo.tibLayer(d);
	    } else if(subid==StripSubdetector::TOB){
	      // layers of TOB start at 5th bin
	      layer = tTopo.tobLayer(d);
	      layer+=4;
	    }

	    SiStripApvGain::Range range=payload->getRange(d);
	    for(int it=0;it<range.second-range.first;it++){
	      sumOfGainsByLayer[layer].first+=payload->getApvGain(it,range);
	      sumOfGainsByLayer[layer].second+=1.;
	    }// loop over APVs
	  } // loop over detIds

	  // loop on the map to fill the plot
	  for (auto& data : sumOfGainsByLayer){
	    
	    fillWithBinAndValue(data.first-1,(data.second.first/data.second.second));
	  }
	  
	}// payload
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    2d histogram of absolute (i.e. not average)
    SiStripApvGains for Tracker Barrel of 1 IOV
  *************************************************/

  class SiStripApvAbsoluteBarrelGainsByLayer : public cond::payloadInspector::Histogram2D<SiStripApvGain> {
    public:
      SiStripApvAbsoluteBarrelGainsByLayer() : cond::payloadInspector::Histogram2D<SiStripApvGain>("SiStripApv Gains by Barrel layer", "Barrel layer (0-3: TIB), (4-9: TOB)", 10, 0, 10, "SiStripApv Gain", 200, 0.0, 2.0){
          Base::setSingleIov(true);
      }
      

      bool fill (const std::vector< std::tuple<cond::Time_t,cond::Hash> >& iovs) override{
        for (auto const& iov: iovs){
          std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload (std::get<1>(iov));
          if (payload.get()){

            TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

            std::vector<uint32_t> detid;
            payload->getDetIds(detid);
            for (const auto & d : detid){
              int subid = DetId(d).subdetId();
              if (subid!=3 && subid!=5) continue;

              SiStripApvGain::Range range = payload->getRange(d);
              for (int it=0;it<range.second-range.first;it++){
                  float gain = payload->getApvGain(it, range);
                  fillWithValue(static_cast<float>((subid == 5) ? tTopo.tobLayer(d)+4 : tTopo.tibLayer(d)),
                                (gain > 2.0)?2.0:gain);
              }
            }//loop over detIds
          }// loop over payloads
        }// loop over iovs
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
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByDisk;

	  for (const auto & d : detid) {

	    int disk=-1;
	    int side=-1;
	    int subid = DetId(d).subdetId();
	    if(subid!=StripSubdetector::TID && subid!=StripSubdetector::TEC) continue;
	    	    
	    if(subid==StripSubdetector::TID){
	      side = tTopo.tidSide(d);
	      disk = tTopo.tidWheel(d); 
	    } else {
	      side = tTopo.tecSide(d);
	      disk = tTopo.tecWheel(d);
	      // disks of TEC start at 4th bin
	      disk+=3;
	    }

	    // only negative side
	    if(side!=1) continue;

	    SiStripApvGain::Range range=payload->getRange(d);
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
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  std::map<int,std::pair<float,float> > sumOfGainsByDisk;
	  
	  for (const auto & d : detid) {

	    int disk=-1;
	    int side=-1;
	    int subid = DetId(d).subdetId();
	    if(subid!=StripSubdetector::TID && subid!=StripSubdetector::TEC) continue;

	    if(subid==StripSubdetector::TID){
	      side = tTopo.tidSide(d);
	      disk = tTopo.tidWheel(d);; 
	    } else {
	      side = tTopo.tecSide(d);
	      disk = tTopo.tecWheel(d); 
	      // disks of TEC start at 4th bin
	      disk+=3;
	    }
	    
	    // only positive side
	    if(side!=2) continue;

	    SiStripApvGain::Range range=payload->getRange(d);
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
    2D histogram of absolute (i.e. not average)
    SiStripApv Gains on the Endcap- for 1 IOV
   ************************************************/
  class SiStripApvAbsoluteEndcapMinusGainsByDisk : public cond::payloadInspector::Histogram2D<SiStripApvGain> {
  public:
    SiStripApvAbsoluteEndcapMinusGainsByDisk() : cond::payloadInspector::Histogram2D<SiStripApvGain>(
            "SiStripApv Gains averages by Endcap (minus) disk",
            "Endcap (minus) disk (0-2: TID), (3-11: TEC)",12,0,12,
            "SiStripApv Gain", 200, 0.0, 2.0){
        Base::setSingleIov(true);
    }

    bool fill (const std::vector< std::tuple<cond::Time_t,cond::Hash> >& iovs) override{
      for (auto const& iov: iovs) {
	    std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	    if( payload.get() ){

	      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

	      std::vector<uint32_t> detid;
	      payload->getDetIds(detid);

          for (const auto & d : detid){
            int subid = DetId(d).subdetId(),
                side  = -1,
                disk  = -1;

            switch (subid){
              case 4: side = tTopo.tidSide(d); disk = tTopo.tidWheel(d)     ; break;
              case 6: side = tTopo.tecSide(d); disk = tTopo.tecWheel(d) + 4 ; break;
              default: continue;
            }

            if (side!=1) continue;
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it=0;it<range.second-range.first;it++){
              float gain = payload->getApvGain(it, range);
              fillWithValue((float) disk, (gain>2.0)?2.0:gain);
            }// apvs
          }// detids
        }
      }// iovs
      return true;
    }// fill
  };

  /************************************************
    2D histogram of absolute (i.e. not average)
    SiStripApv Gains on the Endcap+ for 1 IOV
   ************************************************/
  class SiStripApvAbsoluteEndcapPlusGainsByDisk : public cond::payloadInspector::Histogram2D<SiStripApvGain> {
  public:
    SiStripApvAbsoluteEndcapPlusGainsByDisk() : cond::payloadInspector::Histogram2D<SiStripApvGain>(
            "SiStripApv Gains averages by Endcap (plus) disk",
            "Endcap (plus) disk (0-2: TID), (3-11: TEC)",12,0,12,
            "SiStripApv Gain", 200, 0.0, 2.0){
        Base::setSingleIov(true);
    }

    bool fill (const std::vector< std::tuple<cond::Time_t,cond::Hash> >& iovs) override{
      for (auto const& iov: iovs) {
	    std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	    if( payload.get() ){

	      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath());

	      std::vector<uint32_t> detid;
	      payload->getDetIds(detid);

          for (const auto & d : detid){
            int subid = DetId(d).subdetId(),
                side  = -1,
                disk  = -1;

            switch (subid){
                case 4: side = tTopo.tidSide(d); disk = tTopo.tidWheel(d)     ; break;
                case 6: side = tTopo.tecSide(d); disk = tTopo.tecWheel(d) + 4 ; break;
                default: continue;
            }

            if (side!=2) continue;
            SiStripApvGain::Range range = payload->getRange(d);
            for (int it=0;it<range.second-range.first;it++){
              float gain = payload->getApvGain(it, range);
              fillWithValue((float) disk, (gain>2.0)?2.0:gain);
            }//apvs
          }//detids
        }
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

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain average per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap);
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
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of SiStripApvGains (module with default)
  *************************************************/
  class SiStripApvGainsDefaultTrackerMap : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsDefaultTrackerMap() : cond::payloadInspector::PlotImage<SiStripApvGain>( "Tracker Map of SiStripGains to default" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));

      tmap->setPalette(1);
      
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      /*
	the defaul G1 value comes from the ratio of DefaultTickHeight/GainNormalizationFactor
	as defined in the default of the O2O producer: OnlineDB/SiStripESSources/src/SiStripCondObjBuilderFromDb.cc
       */

      float G1default = 690./640.;  
      float G2default = 1.;

      int totalG1DefaultAPVs=0;
      int totalG2DefaultAPVs=0;

      for (const auto & d : detid) {
	SiStripApvGain::Range range=payload->getRange(d);
	float sumOfGains=0;
	float nAPVsPerModule=0.;
	int countDefaults=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPVsPerModule+=1;
	  sumOfGains+=payload->getApvGain(it,range);
	  if( (payload->getApvGain(it,range))==G1default || (payload->getApvGain(it,range))==G2default) countDefaults++;
	} // loop over APVs
	// fill the tracker map taking the average gain on a single DetId
	if(countDefaults>0.){
	  tmap->fill(d,countDefaults);

	  if( std::fmod((sumOfGains/countDefaults),G1default)==0.){
	    totalG1DefaultAPVs+=countDefaults;
	  } else if ( std::fmod((sumOfGains/countDefaults),G2default)==0.){
	    totalG2DefaultAPVs+=countDefaults;
	  }
	}
      } // loop over detIds
      
      //=========================

      std::string gainType = totalG1DefaultAPVs==0 ? "G2 value (=1)" : "G1 value (=690./640.)";

      std::string titleMap = "# of APVs/module w/ default "+gainType+" (payload : "+std::get<1>(iov)+")";
      tmap->setTitle(titleMap);

      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

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

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
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
      tmap->setTitle(titleMap);
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
      auto range = SiStripPI::getTheRange(cachedRatio);

      std::string fileName(m_imageFileName);
      tmap->save(true,range.first,range.second,fileName);

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

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
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
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::map<std::pair<uint32_t,int>,float> lastmap,firstmap;

      std::vector<uint32_t> detid;
      last_payload->getDetIds(detid);
      
      // cache the last IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  float Gain=last_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  lastmap[index]=Gain;
	} // loop over APVs
      } // loop over detIds
      
      detid.clear();
      
      first_payload->getDetIds(detid);
      
      // cache the first IOV
      for (const auto & d : detid) {
	SiStripApvGain::Range range=first_payload->getRange(d);
	float nAPV=0;
	for(int it=0;it<range.second-range.first;it++){
	  nAPV+=1;
	  float Gain=first_payload->getApvGain(it,range);
	  std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
	  firstmap[index]=Gain;
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
      auto range = SiStripPI::getTheRange(cachedRatio);
      
      //=========================
      
      std::string fileName(m_imageFileName);
      tmap->save(true,range.first,range.second,fileName);

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
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain maximum per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap);
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
      tmap->save(true,0,0,fileName);

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

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "SiStrip APV Gain minumum per module (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripApvGains"));
      tmap->setTitle(titleMap);
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
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    time history histogram of SiStripApvGains 
  *************************************************/

  class SiStripApvGainByRunMeans : public cond::payloadInspector::HistoryPlot<SiStripApvGain,float> {
  public:
    SiStripApvGainByRunMeans() : cond::payloadInspector::HistoryPlot<SiStripApvGain,float>( "SiStripApv Gains average","average Strip APV gain value"){}
    ~SiStripApvGainByRunMeans() override = default;

    float getFromPayload( SiStripApvGain& payload ) override{
     
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
    ~SiStripApvTIBGainByRunMeans() override = default;

    float getFromPayload( SiStripApvGain& payload ) override{
     
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
    ~SiStripApvTOBGainByRunMeans() override = default;

    float getFromPayload( SiStripApvGain& payload ) override{
     
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
    ~SiStripApvTIDGainByRunMeans() override = default;

    float getFromPayload( SiStripApvGain& payload ) override{
     
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
    ~SiStripApvTECGainByRunMeans() override = default;

    float getFromPayload( SiStripApvGain& payload ) override{
     
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

  /************************************************
    test class
  *************************************************/

  class SiStripApvGainsTest : public cond::payloadInspector::Histogram1D<SiStripApvGain> {
    
  public:
    SiStripApvGainsTest() : cond::payloadInspector::Histogram1D<SiStripApvGain>("SiStripApv Gains test",
										"SiStripApv Gains test", 10,0.0,10.0),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripApvGain> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  SiStripDetSummary summaryGain{&m_trackerTopo};

	  for (const auto & d : detid) {
	    SiStripApvGain::Range range=payload->getRange(d);
	    for( int it=0; it < range.second - range.first; ++it ) {
	      summaryGain.add(d,payload->getApvGain(it, range));
	      fillWithValue(payload->getApvGain(it,range));
	    } 
	  }
	  std::map<unsigned int, SiStripDetSummary::Values> map = summaryGain.getCounts();

	  //SiStripPI::printSummary(map);

	  std::stringstream ss;
	  ss << "Summary of gain values:" << std::endl;
	  summaryGain.print(ss, true);
	  std::cout<<ss.str()<<std::endl;
	  
	}// payload
      }// iovs
      return true;
    }// fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Compare Gains from 2 IOVs, 2 pads canvas, firsr for ratio, second for scatter plot
  *************************************************/

  class SiStripApvGainsComparator : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsComparator () : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains Comparison" ){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
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

      std::map<std::string,std::shared_ptr<TH1F>> ratios;
      std::map<std::string,std::shared_ptr<TH2F>> scatters;
      std::map<std::string,int> colormap;
      std::map<std::string,int> markermap;
      colormap["TIB"] = kRed;       markermap["TIB"] = kFullCircle;           
      colormap["TOB"] = kGreen;	    markermap["TOB"] = kFullTriangleUp;
      colormap["TID"] = kBlack;	    markermap["TID"] = kFullSquare;
      colormap["TEC"] = kBlue; 	    markermap["TEC"] = kFullTriangleDown; 

      std::vector<std::string> parts = {"TEC","TOB","TIB","TID"};
      
      for ( const auto &part : parts){
	ratios[part]   = std::make_shared<TH1F>(Form("hRatio_%s",part.c_str()),Form("Gains ratio IOV: %s/ IOV: %s ;New Gain (%s) / Previous Gain (%s);Number of APV",lastIOVsince.c_str(),firstIOVsince.c_str(),lastIOVsince.c_str(),firstIOVsince.c_str()),100,0.,2.);
	scatters[part] = std::make_shared<TH2F>(Form("hScatter_%s",part.c_str()),Form("new Gain (%s) vs previous Gain (%s);Previous Gain (%s);New Gain (%s)",lastIOVsince.c_str(),firstIOVsince.c_str(),firstIOVsince.c_str(),lastIOVsince.c_str()),100,0.5,1.8,100,0.5,1.8);
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

      auto legend = TLegend(0.60,0.8,0.92,0.95);
      legend.SetTextSize(0.05);
      canvas.cd(1)->SetLogy(); 
      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetLeftMargin(0.13);
      canvas.cd(1)->SetRightMargin(0.08);

      for (const auto &part : parts){
	SiStripPI::makeNicePlotStyle(ratios[part].get());
	ratios[part]->SetStats(false);
	ratios[part]->SetLineWidth(2);
	ratios[part]->SetLineColor(colormap[part]);
	if(part =="TEC")
	  ratios[part]->Draw();
	else
	  ratios[part]->Draw("same");
	legend.AddEntry(ratios[part].get(),part.c_str(),"L");
      }

      legend.Draw("same");
      SiStripPI::drawStatBox(ratios,colormap,parts);
       
      auto legend2 = TLegend(0.60,0.8,0.92,0.95);
      legend2.SetTextSize(0.05);
      canvas.cd(2);
      canvas.cd(2)->SetTopMargin(0.05);
      canvas.cd(2)->SetLeftMargin(0.13);
      canvas.cd(2)->SetRightMargin(0.08);

      for (const auto &part : parts){
	SiStripPI::makeNicePlotStyle(scatters[part].get());
	scatters[part]->SetStats(false);
	scatters[part]->SetMarkerColor(colormap[part]);
	scatters[part]->SetMarkerStyle(markermap[part]);
	scatters[part]->SetMarkerSize(0.5);

	auto temp =  (TH2F*)(scatters[part]->Clone());
	temp->SetMarkerSize(1.3);

	if(part =="TEC")
	  scatters[part]->Draw("P");
	else
	  scatters[part]->Draw("Psame");

	legend2.AddEntry(temp,part.c_str(),"P");
      }

      TLine diagonal(0.5,0.5,1.8,1.8);
      diagonal.SetLineWidth(3);
      diagonal.SetLineStyle(2);
      diagonal.Draw("same");

      legend2.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;

    }
  };

  /************************************************
    Compare Gains for each tracker partition 
  *************************************************/

  class SiStripApvGainsComparatorByPartition : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsComparatorByPartition() : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains Comparison By Partition" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
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

      SiStripDetSummary summaryLastGain{&m_trackerTopo};

      for (const auto & d : detid) {
	SiStripApvGain::Range range=last_payload->getRange(d);
	for( int it=0; it < range.second - range.first; ++it ) {
	  summaryLastGain.add(d,last_payload->getApvGain(it, range));
	}
      } 

      SiStripDetSummary summaryFirstGain{&m_trackerTopo};

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

      auto hfirst = std::unique_ptr<TH1F>(new TH1F("byPartition1","SiStrip APV Gain average by partition;; average SiStrip Gain",firstmap.size(),0.,firstmap.size()));
      auto hlast  = std::unique_ptr<TH1F>(new TH1F("byPartition2","SiStrip APV Gain average by partition;; average SiStrip Gain",lastmap.size(),0.,lastmap.size()));
      
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
	hlast->SetBinError(iBin,mean/10000.);
	hlast->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first));
	hlast->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
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
	hfirst->SetBinError(iBin,mean/10000.);
	hfirst->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first));
	hfirst->GetXaxis()->LabelsOption("v");	
      }

      auto extrema = SiStripPI::getExtrema(hfirst.get(),hlast.get());
      hlast->GetYaxis()->SetRangeUser(extrema.first,extrema.second);

      hlast->SetMarkerStyle(20);
      hlast->SetMarkerSize(1);
      hlast->Draw("E1");
      hlast->Draw("Psame");

      hfirst->SetMarkerStyle(18);
      hfirst->SetMarkerSize(1);
      hfirst->SetLineColor(kBlue);
      hfirst->SetMarkerColor(kBlue);
      hfirst->Draw("E1same");
      hfirst->Draw("Psame");

      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i=0;
      for (const auto & line : boundaries){
	l[i] = TLine(hfirst->GetBinLowEdge(line),canvas.cd()->GetUymin(),hfirst->GetBinLowEdge(line),canvas.cd()->GetUymax());
	l[i].SetLineWidth(1);
	l[i].SetLineStyle(9);
	l[i].SetLineColor(2);
	l[i].Draw("same");
	i++;
      }
      
      TLegend legend = TLegend(0.70,0.8,0.95,0.9);
      legend.SetHeader("Gain Comparison","C"); // option "C" allows to center the header
      legend.AddEntry(hfirst.get(),("IOV: "+std::to_string(std::get<0>(firstiov))).c_str(),"PL");
      legend.AddEntry(hlast.get() ,("IOV: "+std::to_string(std::get<0>(lastiov))).c_str(),"PL");
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Plot gain averages by partition 
  *************************************************/

  class SiStripApvGainsByPartition : public cond::payloadInspector::PlotImage<SiStripApvGain> {
  public:
    SiStripApvGainsByPartition() : cond::payloadInspector::PlotImage<SiStripApvGain>( "SiStripGains By Partition" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripApvGain> payload = fetchPayload( std::get<1>(iov) );

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryGain{&m_trackerTopo};

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
      auto h1 = std::unique_ptr<TH1F>(new TH1F("byPartition","SiStrip Gain average by partition;; average SiStrip Gain",map.size(),0.,map.size()));
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
	h1->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first));
	h1->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->Draw("HIST");
      h1->Draw("Psame");
	    
      canvas.Update();
      
      TLine l[boundaries.size()];
      unsigned int i=0;
      for (const auto & line : boundaries){
	l[i] = TLine(h1->GetBinLowEdge(line),canvas.GetUymin(),h1->GetBinLowEdge(line),canvas.GetUymax());
	l[i].SetLineWidth(1);
	l[i].SetLineStyle(9);
	l[i].SetLineColor(2);
	l[i].Draw("same");
	i++;
      }
      
      TLegend legend = TLegend(0.52,0.82,0.95,0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(),"C"); // option "C" allows to center the header
      legend.AddEntry(h1.get(),("IOV: "+std::to_string(std::get<0>(iov))).c_str(),"PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  private:
    TrackerTopology m_trackerTopo;
  };

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiStripApvGain){
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsValue);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsByPartition);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparator);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsComparatorByPartition);
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
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainsRatioMaxDeviationWithPreviousIOVTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTIDGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTOBGainByRunMeans);
  PAYLOAD_INSPECTOR_CLASS(SiStripApvTECGainByRunMeans);
}
