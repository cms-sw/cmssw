/*!
  \file SiStripNoises_PayloadInspector
  \Payload Inspector Plugin for SiStrip Noises
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/21 13:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

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
    test class
  *************************************************/

  class SiStripNoisesTest : public cond::payloadInspector::Histogram1D<SiStripNoises> {
    
  public:
    SiStripNoisesTest() : cond::payloadInspector::Histogram1D<SiStripNoises>("SiStrip Noise test",
									     "SiStrip Noise test", 10,0.0,10.0),
			  m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripNoises> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  fillWithValue(1.);
	 
	  std::stringstream ss;
	  ss << "Summary of strips noises:" << std::endl;

	  //payload->printDebug(ss);
	  payload->printSummary(ss,&m_trackerTopo);
	  //std::cout<<ss.str()<<std::endl;

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);

	  // for (const auto & d : detid) {
	  //   SiStripNoises::Range range=payload->getRange(d);
	  //   for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  //     ss << "DetId="<< d << " Strip=" << payload->decode(*badStrip).firstStrip <<":"<< payload->decode(*badStrip).range << " flag="<< payload->decode(*badStrip).flag << std::endl;
	  //   }
	  // }
	  
	  std::cout<<ss.str()<<std::endl;
 
	}// payload
      }// iovs
      return true;
    }// fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Noise Tracker Map 
  *************************************************/

  template<SiStripPI::estimator est>  class SiStripNoiseTrackerMap : public cond::payloadInspector::PlotImage<SiStripNoises> {
    
  public:
    SiStripNoiseTrackerMap() : cond::payloadInspector::PlotImage<SiStripNoises> ( "Tracker Map of SiStripNoise "+estimatorType(est)+" per module" )
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripNoises> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "Tracker Map of Noise "+estimatorType(est)+" per module (payload : "+std::get<1>(iov)+")";
      
      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripNoises"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      // storage of info
      std::map<unsigned int,float> info_per_detid;
      
      SiStripNoises::RegistryIterator rit=payload->getRegistryVectorBegin(), erit=payload->getRegistryVectorEnd();
      uint16_t Nstrips;
      std::vector<float> vstripnoise;
      double mean,rms,min, max;
      for(;rit!=erit;++rit){
	Nstrips = (rit->iend-rit->ibegin)*8/9; //number of strips = number of chars * char size / strip noise size
	vstripnoise.resize(Nstrips);
	payload->allNoises(vstripnoise,make_pair(payload->getDataVectorBegin()+rit->ibegin,payload->getDataVectorBegin()+rit->iend));
	mean=0; rms=0; min=10000; max=0;  
	
	DetId detId(rit->detid);
	
	for(size_t i=0;i<Nstrips;++i){
	  mean+=vstripnoise[i];
	  rms+=vstripnoise[i]*vstripnoise[i];
	  if(vstripnoise[i]<min) min=vstripnoise[i];
	  if(vstripnoise[i]>max) max=vstripnoise[i];
	}
	
	mean/=Nstrips;
	if((rms/Nstrips-mean*mean)>0.){
	  rms = sqrt(rms/Nstrips-mean*mean);
	} else {
	  rms=0.;
	}       

	switch(est){
	case SiStripPI::min:
	  info_per_detid[rit->detid]=min;
	  break;
	case SiStripPI::max:
	  info_per_detid[rit->detid]=max;
	  break;
	case SiStripPI::mean:
	  info_per_detid[rit->detid]=mean;
	  break;
	case SiStripPI::rms:
	  info_per_detid[rit->detid]=rms;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknown estimator: " <<  est; 
	  break;
	}	
      }

      // loop on the map
      for (const auto &item : info_per_detid){
	tmap->fill(item.first,item.second);
      }

      auto range = SiStripPI::getTheRange(info_per_detid);
      
      //=========================
      
      std::string fileName(m_imageFileName);
      if(est==SiStripPI::rms && (range.first<0.)){
	tmap->save(true,0.,range.second,fileName);
      } else {
	tmap->save(true,range.first,range.second,fileName);
      }

      return true;
    }
  };

  typedef SiStripNoiseTrackerMap<SiStripPI::min>  SiStripNoiseMin_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::max>  SiStripNoiseMax_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::mean> SiStripNoiseMean_TrackerMap;
  typedef SiStripNoiseTrackerMap<SiStripPI::rms>  SiStripNoiseRMS_TrackerMap;

} // close namespace

PAYLOAD_INSPECTOR_MODULE(SiStripNoises){
  PAYLOAD_INSPECTOR_CLASS(SiStripNoisesTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMin_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMax_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseMean_TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripNoiseRMS_TrackerMap);
}
