/*!
  \file SiStripLorentzAngle_PayloadInspector
  \Payload Inspector Plugin for SiStrip Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/09/21 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

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

  /************************************************
    TrackerMap of SiStrip Lorentz Angle
  *************************************************/
  class SiStripLorentzAngle_TrackerMap : public cond::payloadInspector::PlotImage<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngle_TrackerMap() : cond::payloadInspector::PlotImage<SiStripLorentzAngle>( "Tracker Map SiStrip Lorentz Angle" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripLorentzAngle> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripLorentzAngle"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of SiStrip Lorentz Angle per module, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::map<uint32_t,float> LAMap_ = payload->getLorentzAngles();
      
      for(const auto &element : LAMap_){
	tmap->fill(element.first,element.second);
      } // loop over the LA MAP
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    Plot Lorentz Angle averages by partition 
  *************************************************/

  class SiStripLorentzAngleByPartition : public cond::payloadInspector::PlotImage<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngleByPartition() : cond::payloadInspector::PlotImage<SiStripLorentzAngle>( "SiStripLorentzAngle By Partition" ),
      m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml").fullPath())}
    {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripLorentzAngle> payload = fetchPayload( std::get<1>(iov) );

      SiStripDetSummary summaryLA{&m_trackerTopo};

      std::map<uint32_t,float> LAMap_ = payload->getLorentzAngles();
      
      for(const auto &element : LAMap_){
	summaryLA.add(element.first,element.second);
      } 

      std::map<unsigned int, SiStripDetSummary::Values> map = summaryLA.getCounts();
      //=========================
      
      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.cd();
      auto h1 = std::unique_ptr<TH1F>(new TH1F("byPartition","SiStrip LA average by partition;; average SiStrip Lorentz Angle [rad]",map.size(),0.,map.size()));
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

}

PAYLOAD_INSPECTOR_MODULE( SiStripLorentzAngle ){
  PAYLOAD_INSPECTOR_CLASS( SiStripLorentzAngle_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripLorentzAngleByPartition );
}
