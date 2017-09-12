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

  /************************************************
    time history histogram of bad components fraction
  *************************************************/

  class SiStripBadStripFractionByRun : public cond::payloadInspector::HistoryPlot<SiStripBadStrip,float> {
  public:
    SiStripBadStripFractionByRun() : cond::payloadInspector::HistoryPlot<SiStripBadStrip,float>( "SiStrip Bad Strip fraction per run","Bad Strip fraction [%]"){}
    ~SiStripBadStripFractionByRun() override = default;

    float getFromPayload( SiStripBadStrip& payload ) override{
     
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload.getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload.decode(*badStrip).range;
	}
	badStripsPerDetId[d] = badStrips;
      } // loop over detIds 
      
      float numerator(0.),denominator(0.);
      std::vector<uint32_t> all_detids=reader->getAllDetIds();
      for (const auto & det : all_detids) {
	denominator+=128.*reader->getNumberOfApvsAndStripLength(det).first;
	if(badStripsPerDetId.count(det)!=0) numerator+= badStripsPerDetId[det];
      }
      
      delete reader;
      return (numerator/denominator)*100.;
      
    } // payload
  };

   /************************************************
    time history histogram of bad components fraction (TIB)
  *************************************************/

  class SiStripBadStripTIBFractionByRun : public cond::payloadInspector::HistoryPlot<SiStripBadStrip,float> {
  public:
    SiStripBadStripTIBFractionByRun() : cond::payloadInspector::HistoryPlot<SiStripBadStrip,float>( "SiStrip Inner Barrel Bad Strip fraction per run","TIB Bad Strip fraction [%]"){}
    ~SiStripBadStripTIBFractionByRun() override = default;

    float getFromPayload( SiStripBadStrip& payload ) override{
     
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload.getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload.decode(*badStrip).range;
	}
	badStripsPerDetId[d] = badStrips;
      } // loop over detIds 
      
      float numerator(0.),denominator(0.);
      std::vector<uint32_t> all_detids=reader->getAllDetIds();
      for (const auto & det : all_detids) {
	int subid = DetId(det).subdetId();
	if(subid != StripSubdetector::TIB) continue;
	denominator+=128.*reader->getNumberOfApvsAndStripLength(det).first;
	if(badStripsPerDetId.count(det)!=0) numerator+= badStripsPerDetId[det];
      }
      
      delete reader;
      return (numerator/denominator)*100.;
      
    } // payload
  };

   /************************************************
    time history histogram of bad components fraction (TOB)
  *************************************************/

  class SiStripBadStripTOBFractionByRun : public cond::payloadInspector::HistoryPlot<SiStripBadStrip,float> {
  public:
    SiStripBadStripTOBFractionByRun() : cond::payloadInspector::HistoryPlot<SiStripBadStrip,float>( "SiStrip Outer Barrel Bad Strip fraction per run","TOB Bad Strip fraction [%]"){}
    ~SiStripBadStripTOBFractionByRun() override = default;

    float getFromPayload( SiStripBadStrip& payload ) override{
     
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload.getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload.decode(*badStrip).range;
	}
	badStripsPerDetId[d] = badStrips;
      } // loop over detIds 
      
      float numerator(0.),denominator(0.);
      std::vector<uint32_t> all_detids=reader->getAllDetIds();
      for (const auto & det : all_detids) {
	int subid = DetId(det).subdetId();
	if(subid != StripSubdetector::TOB) continue;	
	denominator+=128.*reader->getNumberOfApvsAndStripLength(det).first;
	if(badStripsPerDetId.count(det)!=0) numerator+= badStripsPerDetId[det];
      }
      
      delete reader;
      return (numerator/denominator)*100.;
      
    } // payload
  };

   /************************************************
    time history histogram of bad components fraction (TID)
   *************************************************/

  class SiStripBadStripTIDFractionByRun : public cond::payloadInspector::HistoryPlot<SiStripBadStrip,float> {
  public:
    SiStripBadStripTIDFractionByRun() : cond::payloadInspector::HistoryPlot<SiStripBadStrip,float>( "SiStrip Inner Disks Bad Strip fraction per run","TID Bad Strip fraction [%]"){}
    ~SiStripBadStripTIDFractionByRun() override = default;

    float getFromPayload( SiStripBadStrip& payload ) override{
     
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload.getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload.decode(*badStrip).range;
	}
	badStripsPerDetId[d] = badStrips;
      } // loop over detIds 
      
      float numerator(0.),denominator(0.);
      std::vector<uint32_t> all_detids=reader->getAllDetIds();
      for (const auto & det : all_detids) {
	int subid = DetId(det).subdetId();
	if(subid != StripSubdetector::TID) continue;
	denominator+=128.*reader->getNumberOfApvsAndStripLength(det).first;
	if(badStripsPerDetId.count(det)!=0) numerator+= badStripsPerDetId[det];
      }
      
      delete reader;
      return (numerator/denominator)*100.;
      
    } // payload
  };

  /************************************************
    time history histogram of bad components fraction (TEC)
   *************************************************/

  class SiStripBadStripTECFractionByRun : public cond::payloadInspector::HistoryPlot<SiStripBadStrip,float> {
  public:
    SiStripBadStripTECFractionByRun() : cond::payloadInspector::HistoryPlot<SiStripBadStrip,float>( "SiStrip Endcaps Bad Strip fraction per run","TEC Bad Strip fraction [%]"){}
    ~SiStripBadStripTECFractionByRun() override = default;

    float getFromPayload( SiStripBadStrip& payload ) override{
     
      edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
      SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
      
      std::vector<uint32_t> detid;
      payload.getDetIds(detid);
      
      std::map<uint32_t,int> badStripsPerDetId;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload.getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload.decode(*badStrip).range;
	}
	badStripsPerDetId[d] = badStrips;
      } // loop over detIds 
      
      float numerator(0.),denominator(0.);
      std::vector<uint32_t> all_detids=reader->getAllDetIds();
      for (const auto & det : all_detids) {
	int subid = DetId(det).subdetId();
	if(subid != StripSubdetector::TEC) continue;
	denominator+=128.*reader->getNumberOfApvsAndStripLength(det).first;
	if(badStripsPerDetId.count(det)!=0) numerator+= badStripsPerDetId[det];
      }
      
      delete reader;
      return (numerator/denominator)*100.;
      
    } // payload
  };

  /************************************************
    Plot BadStrip by region 
  *************************************************/

  class SiStripBadStripByRegion : public cond::payloadInspector::PlotImage<SiStripBadStrip> {
  public:
    SiStripBadStripByRegion() : cond::payloadInspector::PlotImage<SiStripBadStrip>( "SiStrip BadStrip By Region" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripBadStrip> payload = fetchPayload( std::get<1>(iov) );

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryBadStrips{&m_trackerTopo};
      int totalBadStrips =0;

      for (const auto & d : detid) {
	SiStripBadStrip::Range range=payload->getRange(d);
	int badStrips(0);
	for( std::vector<unsigned int>::const_iterator badStrip = range.first;badStrip != range.second; ++badStrip ) {
	  badStrips+= payload->decode(*badStrip).range;
	}
	totalBadStrips+=badStrips;
	summaryBadStrips.add(d,badStrips);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapBadStrips = summaryBadStrips.getCounts();

      //=========================
      
      TCanvas canvas("BadStrip Partion summary","SiStripBadStrip region summary",1200,1000); 
      canvas.cd();
      auto h_BadStrips = std::unique_ptr<TH1F>(new TH1F("BadStripsbyRegion","SiStrip Bad Strip summary by region;; n. bad strips",mapBadStrips.size(),0.,mapBadStrips.size()));
      h_BadStrips->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.12);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin=0;

      std::string detector;
      std::string currentDetector;

      for (const auto &element : mapBadStrips){
	iBin++;
	int countBadStrips = (element.second.mean);

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

	h_BadStrips->SetBinContent(iBin,countBadStrips);
	h_BadStrips->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(element.first));
	h_BadStrips->GetXaxis()->LabelsOption("v");

	if(detector!=currentDetector) {
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      h_BadStrips->SetMarkerStyle(21);
      h_BadStrips->SetMarkerSize(1);
      h_BadStrips->SetLineColor(kBlue);
      h_BadStrips->SetLineStyle(9);
      h_BadStrips->SetMarkerColor(kBlue);
      h_BadStrips->GetYaxis()->SetRangeUser(0.,h_BadStrips->GetMaximum()*1.30);
      h_BadStrips->GetYaxis()->SetTitleOffset(1.7);
      h_BadStrips->Draw("HISTsame");
      h_BadStrips->Draw("TEXTsame");
      
      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i=0;
      for (const auto & line : boundaries){
        l[i] = TLine(h_BadStrips->GetBinLowEdge(line),canvas.cd()->GetUymin(),h_BadStrips->GetBinLowEdge(line),canvas.cd()->GetUymax());
	l[i].SetLineWidth(1);
	l[i].SetLineStyle(9);
	l[i].SetLineColor(2);
	l[i].Draw("same");
	i++;
      }
      
      TLegend legend = TLegend(0.52,0.82,0.95,0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(),"C"); // option "C" allows to center the header
      legend.AddEntry(h_BadStrips.get(),("IOV: "+std::to_string(std::get<0>(iov))+"| n. of bad strips:"+std::to_string(totalBadStrips)).c_str(),"PL");
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
PAYLOAD_INSPECTOR_MODULE(SiStripBadStrip){
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTest);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadModuleTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripFractionByRun);
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTIBFractionByRun); 
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTOBFractionByRun); 
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTIDFractionByRun); 
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripTECFractionByRun); 
  PAYLOAD_INSPECTOR_CLASS(SiStripBadStripByRegion);
}
