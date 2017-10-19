/*!
  \file AlignPCLThresholds_PayloadInspector
  \Payload Inspector Plugin for Alignment PCL thresholds
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/10/19 12:51:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"

// auxilliary functions

#include <memory>
#include <sstream>
#include <iostream>
#include <regex>

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
    Display of AlignPCLThresholds
  *************************************************/
  class AlignPCLThresholds_Display : public cond::payloadInspector::PlotImage<AlignPCLThresholds> {
  public:
    AlignPCLThresholds_Display() : cond::payloadInspector::PlotImage<AlignPCLThresholds>( "Display of threshold parameters for SiPixelAli PCL" ){
    setSingleIov( true );
    }
  
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<AlignPCLThresholds> payload = fetchPayload( std::get<1>(iov) );
      AlignPCLThresholds::threshold_map m_thresholds = payload->getThreshold_Map();
      
      TCanvas canvas("Alignment PCL thresholds summary","Alignment PCL thresholds summary",1000,1000); 
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.25);
      canvas.SetLeftMargin(0.18);
      canvas.SetRightMargin(0.05);
      canvas.Modified();
      canvas.SetGrid();

      auto Thresholds = std::unique_ptr<TH2F>(new TH2F("Thresholds","Alignment parameter thresholds",m_thresholds.size(),0,m_thresholds.size(),24,0,24));
      Thresholds->SetStats(false);

      std::array<std::string,24> ylabels = {{"X","sig X","maxMove X","Error X","Y","sig Y","maxMove Y","Error Y","Z","sig Z","maxMove Z", "Error Z","#theta_{X}","sig #theta_{X}","maxMove #theta_{X}","Error #theta_{X}","#theta_{Y}","sig #theta_{Y}","maxMove #theta_{Y}","Error #theta_{Y}","#theta_{Z}","sig #theta_{Z}","maxMove #theta_{Z}","Error #theta_{Z}"}};

      unsigned int xBin=0;
      for(const auto &entry : m_thresholds){
	xBin++;
 	Thresholds->GetXaxis()->SetBinLabel(xBin,(entry.first).c_str());

	// fill the labels on y-axis
	unsigned int yBin=25;
	for (const std::string &text : ylabels ){
	  yBin--;
	  if(xBin==1){
	    Thresholds->GetYaxis()->SetBinLabel(yBin,text.c_str());
	  }
	}

	for(int foo = AlignPCLThresholds::X; foo != AlignPCLThresholds::extra_DOF; foo++ ){
	  AlignPCLThresholds::coordType type = static_cast<AlignPCLThresholds::coordType>(foo);  

	  Thresholds->SetBinContent(xBin,24-(foo*4),payload->getCut(entry.first,type));
	  Thresholds->SetBinContent(xBin,24-((foo*4)+1),payload->getSigCut(entry.first,type));
	  Thresholds->SetBinContent(xBin,24-((foo*4)+2),payload->getMaxMoveCut(entry.first,type));
	  Thresholds->SetBinContent(xBin,24-((foo*4)+3),payload->getMaxErrorCut(entry.first,type));
	}
	
      }

      Thresholds->GetXaxis()->LabelsOption("v");
      Thresholds->Draw("TEXT");
      
      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
    
    std::string getStringFromCoordEnum (const AlignPCLThresholds::coordType &type){
      switch(type){
      case AlignPCLThresholds::X : return "X";
      case AlignPCLThresholds::Y : return "Y";
      case AlignPCLThresholds::Z : return "Y";
      case AlignPCLThresholds::theta_X : return "theta_X";
      case AlignPCLThresholds::theta_Y : return "theta_Y";
      case AlignPCLThresholds::theta_Z : return "theta_Z";
      default : return "should never be here";
      }
    }
    
  };
} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(AlignPCLThresholds){
  PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholds_Display);
}
