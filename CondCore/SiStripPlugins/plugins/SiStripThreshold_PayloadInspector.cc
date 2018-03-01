/*!
  \file SiStripThreshold_PayloadInspector
  \Payload Inspector Plugin for SiStrip Threshold 
  \author J. Prisciandaro
  \version $Revision: 1.0 $
  \date $Date: 2018/02/22 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "DataFormats/DetId/interface/DetId.h"                     
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"   
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"  
#include "FWCore/MessageLogger/interface/MessageLogger.h"      

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"   

// auxilliary functions
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

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

using namespace std;

namespace {

  /************************************************
    test class
  *************************************************/

  class SiStripThresholdTest : public cond::payloadInspector::Histogram1D<SiStripThreshold> {
    //eredita da Histogram1D...
  public: //costruttore che chiama il costruttore della classe madre
    SiStripThresholdTest() : cond::payloadInspector::Histogram1D<SiStripThreshold>("SiStrip Threshold test",
										   "SiStrip Threshold test", 10,0.0,10.0),
			     m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())} //datamember inizializzato
    {
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{ //override ?
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripThreshold> payload = Base::fetchPayload( std::get<1>(iov) ); //prendo l'hash da ogni iov e pesco il payloas
	if( payload.get() ){
	 
	  fillWithValue(1.);
	 
	  std::stringstream ss;
	  ss << "Summary of strips threshold:" << std::endl;

	  payload->printSummary(ss,&m_trackerTopo);

	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  	  
	  std::cout<<ss.str()<<std::endl;
 
	}// payload
      }// iovs
      return true;
    }// fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************************
    1d histogram of SiStripThresholds of 1 IOV - High Threshold 
  *************************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiStripThresholdValue : public cond::payloadInspector::Histogram1D<SiStripThreshold> {
    
  public:
    SiStripThresholdValue() : cond::payloadInspector::Histogram1D<SiStripThreshold>("SiStrip Threshold values",
										    "SiStrip Threshold values", 300,0.0,10){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripThreshold> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  for (const auto & d : detid) {
	    std::cout<<d<<std::endl;
	    SiStripThreshold::Range range=payload->getRange(d);
	    for( int it=0; it<768; ++it ){
	      //std::cout<<"step3"<<std::endl;
	      auto hth = payload->getData(it,range).getHth();
	      //std::cout<<hth<<std::endl;
	      //to be used to fill the histogram
	      fillWithValue(hth);
	    }// loop over APVs
	  } // loop over detIds
	}// payload
      }// iovs
      return true;
    }// fill
  };


  /************************************************
    Tracker Map of SiStrip Threshold
  *************************************************/
  /*
  template<SiStripPI::estimator est> class SiStripThresholdTrackerMap : public cond::payloadInspector::PlotImage<SiStripThreshold> {
    
  public:
    SiStripThresholdTrackerMap() : cond::payloadInspector::PlotImage<SiStripThreshold> ( "Tracker Map of SiStripThreshold "+estimatorType(est)+" per module" )
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripThreshold> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "Tracker Map of SiStrip Threshold "+estimatorType(est)+" per module (payload : "+std::get<1>(iov)+")";
      
      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripThreshold"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      std::map<unsigned int,float> info_per_detid;
      
      for (const auto & d : detid) {
	int nstrips=0;
	double mean(0.),rms(0.),min(10000.), max(0.);
	SiStripThreshold::Range range = payload->getRange(d);
	for( int it=0; it < (range.second-range.first)*8/10; ++it ){
	  nstrips++;
	  auto ped = payload->getPed(it,range);
	  mean+=ped;
	  rms+=(ped*ped);
	  if(ped<min) min=ped;
	  if(ped>max) max=ped;
	} // end of loop on strips
	
	mean/=nstrips;
	if((rms/nstrips-mean*mean)>0.){
	  rms = sqrt(rms/nstrips-mean*mean);
	} else {
	  rms=0.;
	}       

	switch(est){
	case SiStripPI::min:
	  info_per_detid[d]=min;
	  break;
	case SiStripPI::max:
	  info_per_detid[d]=max;
	  break;
	case SiStripPI::mean:
	  info_per_detid[d]=mean;
	  break;
	case SiStripPI::rms:
	  info_per_detid[d]=rms;
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknown estimator: " <<  est; 
	  break;
	}	
      } // end of loop on detids

      for(const auto & d : detid){
	tmap->fill(d,info_per_detid[d]);
      }

      std::string fileName(m_imageFileName);
      tmap->save(true,0.,0.,fileName);

      return true;
    }
  };

  typedef SiStripPedestalsTrackerMap<SiStripPI::min>  SiStripPedestalsMin_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::max>  SiStripPedestalsMax_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::mean> SiStripPedestalsMean_TrackerMap;
  typedef SiStripPedestalsTrackerMap<SiStripPI::rms>  SiStripPedestalsRMS_TrackerMap;


  */
  /************************************************
    Tracker Map of SiStrip Threshold Summaries
  *************************************************/
  /*
  template<SiStripPI::estimator est> class SiStripPedestalsByRegion : public cond::payloadInspector::PlotImage<SiStripPedestals> {
  public:
     SiStripPedestalsByRegion() : cond::payloadInspector::PlotImage<SiStripPedestals>( "SiStrip Pedestals "+estimatorType(est)+" by Region" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
      auto iov = iovs.front();
      std::shared_ptr<SiStripPedestals> payload = fetchPayload( std::get<1>(iov) );
      
      SiStripDetSummary summaryPedestals{&m_trackerTopo};
      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	int nstrips=0;
	double mean(0.),rms(0.),min(10000.), max(0.);
	SiStripPedestals::Range range = payload->getRange(d);
	for( int it=0; it < (range.second-range.first)*8/10; ++it ){
	  nstrips++;
	  auto ped = payload->getPed(it,range);
	  mean+=ped;
	  rms+=(ped*ped);
	  if(ped<min) min=ped;
	  if(ped>max) max=ped;
	} // end of loop on strips
	
	mean/=nstrips;
	if((rms/nstrips-mean*mean)>0.){
	  rms = sqrt(rms/nstrips-mean*mean);
	} else {
	  rms=0.;
	}       
	
	switch(est){
	case SiStripPI::min:
	  summaryPedestals.add(d,min);
	  break;
	case SiStripPI::max:
	  summaryPedestals.add(d,max);
	  break;
	case SiStripPI::mean:
	  summaryPedestals.add(d,mean);
	  break;
	case SiStripPI::rms:
	  summaryPedestals.add(d,rms);
	  break;
	default:
	  edm::LogWarning("LogicError") << "Unknown estimator: " <<  est; 
	  break;
	}	
      } // loop on the detIds
      
      std::map<unsigned int, SiStripDetSummary::Values> map = summaryPedestals.getCounts();
      //=========================
      
      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(new TH1F("byRegion",Form("Average by partition of %s SiStrip Pedestals per module;;average SiStrip Pedestals %s [ADC counts]",estimatorType(est).c_str(),estimatorType(est).c_str()),map.size(),0.,map.size()));
      h1->SetStats(false);
      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.17);
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
	h1->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first).second);
	h1->GetXaxis()->LabelsOption("v");
	
	if(detector!=currentDetector) {
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      h1->SetMarkerStyle(20);
      h1->SetMarkerSize(1);
      h1->SetMaximum(h1->GetMaximum()*1.1);
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


  typedef SiStripPedestalsByRegion<SiStripPI::mean> SiStripPedestalsMeanByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::min>  SiStripPedestalsMinByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::max>  SiStripPedestalsMaxByRegion;
  typedef SiStripPedestalsByRegion<SiStripPI::rms>  SiStripPedestalsRMSByRegion;
  */
} // close namespace

PAYLOAD_INSPECTOR_MODULE(SiStripThreshold){
  PAYLOAD_INSPECTOR_CLASS(SiStripThresholdTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripThresholdValue);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMin_TrackerMap);
  // PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMax_TrackerMap);
  // PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMean_TrackerMap);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsRMS_TrackerMap);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMeanByRegion);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMinByRegion);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsMaxByRegion);
  //PAYLOAD_INSPECTOR_CLASS(SiStripPedestalsRMSByRegion);
}
