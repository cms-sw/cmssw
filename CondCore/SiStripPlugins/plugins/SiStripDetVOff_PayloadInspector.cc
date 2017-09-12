#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h" 

#include <memory>
#include <sstream>

// include ROOT 
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace {

  class SiStripDetVOff_LV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int>{
  public:
    SiStripDetVOff_LV(): cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with LV OFF vs time", "nLVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ) override{
      return payload.getLVoffCounts();
    }

  };

  class SiStripDetVOff_HV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int> {
  public:
    SiStripDetVOff_HV() : cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with HV OFF vs time","nHVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ) override{
      return payload.getHVoffCounts();
    }

  };

  /************************************************
    TrackerMap of Module VOff
  *************************************************/
  class SiStripDetVOff_IsModuleVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of VOff modules (HV or LV), payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of Module HVOff
  *************************************************/
  class SiStripDetVOff_IsModuleHVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleHVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleHVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleHVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of HV Off modules, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleHVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of Module LVOff
  *************************************************/
  class SiStripDetVOff_IsModuleLVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleLVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleLVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleLVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of LV Off modules, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleLVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    test class
  *************************************************/

  class SiStripDetVOffTest : public cond::payloadInspector::Histogram1D<SiStripDetVOff> {
    
  public:
    SiStripDetVOffTest() : cond::payloadInspector::Histogram1D<SiStripDetVOff>("SiStrip DetVOff test",
									       "SiStrip DetVOff test", 10,0.0,10.0),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiStripDetVOff> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  std::vector<uint32_t> detid;
	  payload->getDetIds(detid);
	  
	  SiStripDetSummary summaryHV{&m_trackerTopo};
	  SiStripDetSummary summaryLV{&m_trackerTopo};
	  
	  for (const auto & d : detid) {
	    if(payload->IsModuleLVOff(d)) summaryLV.add(d);
	    if(payload->IsModuleHVOff(d)) summaryHV.add(d);
	  }
	  std::map<unsigned int, SiStripDetSummary::Values> mapHV = summaryHV.getCounts();
	  std::map<unsigned int, SiStripDetSummary::Values> mapLV = summaryLV.getCounts();

	  // SiStripPI::printSummary(mapHV);
	  // SiStripPI::printSummary(mapLV);
 
	  std::stringstream ss;
	  ss << "Summary of HV off detectors:" << std::endl;
	  summaryHV.print(ss, true);

	  ss << "Summary of LV off detectors:" << std::endl;
	  summaryLV.print(ss, true);

	  std::cout<<ss.str()<<std::endl;
	  

	}// payload
      }// iovs
      return true;
    }// fill
  private:
    TrackerTopology m_trackerTopo;
  };

  /************************************************
    Plot DetVOff by region 
  *************************************************/

  class SiStripDetVOffByRegion : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOffByRegion() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "SiStrip DetVOff By Region" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      SiStripDetSummary summaryHV{&m_trackerTopo};
      SiStripDetSummary summaryLV{&m_trackerTopo};
      
      for (const auto & d : detid) {
	if(payload->IsModuleLVOff(d)) summaryLV.add(d);
	if(payload->IsModuleHVOff(d)) summaryHV.add(d);
      }
      std::map<unsigned int, SiStripDetSummary::Values> mapHV = summaryHV.getCounts();
      std::map<unsigned int, SiStripDetSummary::Values> mapLV = summaryLV.getCounts();
      std::vector<unsigned int> keys;
      std::transform(mapHV.begin(),
		     mapHV.end(),
		     std::back_inserter(keys),
		     [](const std::map<unsigned int, SiStripDetSummary::Values>::value_type &pair){return pair.first;});

      //=========================
      
      TCanvas canvas("DetVOff Partion summary","SiStripDetVOff region summary",1200,1000); 
      canvas.cd();
      auto h_HV = std::unique_ptr<TH1F>(new TH1F("HVbyRegion","SiStrip HV/LV summary by region;; modules with HV off",mapHV.size(),0.,mapHV.size()));
      auto h_LV = std::unique_ptr<TH1F>(new TH1F("LVbyRegion","SiStrip HV/LV summary by region;; modules with LV off",mapLV.size(),0.,mapLV.size()));

      h_HV->SetStats(false);
      h_LV->SetStats(false);

      canvas.SetBottomMargin(0.18);
      canvas.SetLeftMargin(0.10);
      canvas.SetRightMargin(0.10);
      canvas.Modified();

      std::vector<int> boundaries;
      unsigned int iBin=0;

      std::string detector;
      std::string currentDetector;

      for (const auto &index : keys){
	iBin++;
	int countHV = mapHV[index].count;
	int countLV = mapLV[index].count;

	if(currentDetector.empty()) currentDetector="TIB";
	
	switch ((index)/1000) 
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

	h_HV->SetBinContent(iBin,countHV);
	h_HV->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(index));
	h_HV->GetXaxis()->LabelsOption("v");

	h_LV->SetBinContent(iBin,countLV);
	h_LV->GetXaxis()->SetBinLabel(iBin,SiStripPI::regionType(index));
	h_LV->GetXaxis()->LabelsOption("v");

	if(detector!=currentDetector) {
	  boundaries.push_back(iBin);
	  currentDetector=detector;
	}
      }

      auto extrema = SiStripPI::getExtrema(h_LV.get(),h_HV.get());
      h_HV->GetYaxis()->SetRangeUser(extrema.first,extrema.second);
      h_LV->GetYaxis()->SetRangeUser(extrema.first,extrema.second);

      h_HV->SetMarkerStyle(20);
      h_HV->SetMarkerSize(1);
      h_HV->SetLineColor(kRed);
      h_HV->SetMarkerColor(kRed);
      h_HV->Draw("HIST");
      h_HV->Draw("TEXT45same");

      h_LV->SetMarkerStyle(21);
      h_LV->SetMarkerSize(1);
      h_LV->SetLineColor(kBlue);
      h_LV->SetLineStyle(9);
      h_LV->SetMarkerColor(kBlue);
      h_LV->Draw("HISTsame");
      h_LV->Draw("TEXT45same");
      
      canvas.Update();
      canvas.cd();

      TLine l[boundaries.size()];
      unsigned int i=0;
      for (const auto & line : boundaries){
	l[i] = TLine(h_HV->GetBinLowEdge(line),canvas.cd()->GetUymin(),h_HV->GetBinLowEdge(line),canvas.cd()->GetUymax());
	l[i].SetLineWidth(1);
	l[i].SetLineStyle(9);
	l[i].SetLineColor(2);
	l[i].Draw("same");
	i++;
      }
      
      TLegend legend = TLegend(0.45,0.80,0.90,0.9);
      legend.SetHeader((std::get<1>(iov)).c_str(),"C"); // option "C" allows to center the header
      legend.AddEntry(h_HV.get(),("HV channels: "+std::to_string(payload->getHVoffCounts())).c_str(),"PL");
      legend.AddEntry(h_LV.get(),("LV channels: "+std::to_string(payload->getLVoffCounts())).c_str(),"PL");
      legend.SetTextSize(0.025);
      legend.Draw("same");

      // Remove the current axis
      h_HV.get()->GetYaxis()->SetLabelOffset(999);
      h_HV.get()->GetYaxis()->SetTickLength(0);
      h_HV.get()->GetYaxis()->SetTitleOffset(999);

      h_LV.get()->GetYaxis()->SetLabelOffset(999);
      h_LV.get()->GetYaxis()->SetTickLength(0);
      h_LV.get()->GetYaxis()->SetTitleOffset(999);

      //draw an axis on the left side
      auto l_axis = std::unique_ptr<TGaxis>(new TGaxis(gPad->GetUxmin(),gPad->GetUymin(),gPad->GetUxmin(),gPad->GetUymax(),0,extrema.second,510));
      l_axis->SetLineColor(kRed);
      l_axis->SetTextColor(kRed);
      l_axis->SetLabelColor(kRed);
      l_axis->SetTitleOffset(1.2);
      l_axis->SetTitleColor(kRed);
      l_axis->SetTitle(h_HV.get()->GetYaxis()->GetTitle());
      l_axis->Draw();
      
      //draw an axis on the right side
      auto r_axis =  std::unique_ptr<TGaxis>(new TGaxis(gPad->GetUxmax(),gPad->GetUymin(),gPad->GetUxmax(),gPad->GetUymax(),0,extrema.second,510,"+L"));
      r_axis->SetLineColor(kBlue);
      r_axis->SetTextColor(kBlue);
      r_axis->SetLabelColor(kBlue);
      r_axis->SetTitleColor(kBlue);
      r_axis->SetTitleOffset(1.2);
      r_axis->SetTitle(h_LV.get()->GetYaxis()->GetTitle());
      r_axis->Draw();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  private:
    TrackerTopology m_trackerTopo;
  };

}

PAYLOAD_INSPECTOR_MODULE( SiStripDetVOff ){
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_LV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_HV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleVOff_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleLVOff_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleHVOff_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOffTest );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOffByRegion );
}
