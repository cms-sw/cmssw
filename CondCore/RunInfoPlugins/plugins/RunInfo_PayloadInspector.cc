/*!
  \file RunInfo_PayloadInspector
  \Payload Inspector Plugin for RunInfo
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/03/18 10:01:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

// helper
#include "CondCore/RunInfoPlugins/interface/RunInfoPayloadInspectoHelper.h"

// system includes
#include <memory>
#include <sstream>
#include <iostream>

// include ROOT 
#include "TProfile.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TPaletteAxis.h"

namespace {

  /************************************************
     RunInfo Payload Inspector of 1 IOV 
  *************************************************/
  class RunInfoTest : public cond::payloadInspector::Histogram1D<RunInfo> {
    
  public:
    RunInfoTest() : cond::payloadInspector::Histogram1D<RunInfo>( "Test RunInfo", "Test RunInfo",10,0.0,10.0)
    {
      Base::setSingleIov( true );
    }
  
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<RunInfo> payload = Base::fetchPayload( std::get<1>(iov) );
      
      if(payload.get() ) {
	payload->printAllValues();
      }
      return true;
    }
  };

  /************************************************
    Summary of RunInfo of 1 IOV 
  *************************************************/
  class RunInfoParameters : public cond::payloadInspector::PlotImage<RunInfo> {
  public:
    RunInfoParameters() : cond::payloadInspector::PlotImage<RunInfo>( "Display of RunInfo parameters" ){
      setSingleIov( true );
    }
  
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<RunInfo> payload = fetchPayload( std::get<1>(iov) );
      
      TCanvas canvas("Beam Spot Parameters Summary","RunInfo Parameters summary",1000,1000); 
      canvas.cd();

      gStyle->SetHistMinimumZero();

      canvas.SetTopMargin(0.08);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.3);
      canvas.SetRightMargin(0.02);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_RunInfoParameters = std::unique_ptr<TH2F>(new TH2F("Parameters","",1,0.0,1.0,11,0,11.));
      auto h2_RunInfoState      = std::unique_ptr<TH2F>(new TH2F("State","",1,0.0,1.0,11,0,11.));
      h2_RunInfoParameters->SetStats(false);
      h2_RunInfoState->SetStats(false);
      
      float fieldIntensity = RunInfoPI::theBField(payload->m_avg_current);

      std::function<float(RunInfoPI::parameters)> cutFunctor = [&payload,fieldIntensity](RunInfoPI::parameters my_param) {
	float ret(-999.);
	switch(my_param){
	case RunInfoPI::m_run		        : return float(payload->m_run);		    
	case RunInfoPI::m_start_time_ll	        : return float(payload->m_start_time_ll);	     
	case RunInfoPI::m_stop_time_ll	        : return float(payload->m_stop_time_ll);	    
	case RunInfoPI::m_start_current         : return payload->m_start_current; 	    
	case RunInfoPI::m_stop_current	        : return payload->m_stop_current;	    
	case RunInfoPI::m_avg_current	        : return payload->m_avg_current;	       
	case RunInfoPI::m_max_current	        : return payload->m_max_current;	     
	case RunInfoPI::m_min_current	        : return payload->m_min_current;	     
	case RunInfoPI::m_run_intervall_micros  : return payload->m_run_intervall_micros; 
	case RunInfoPI::m_BField                : return fieldIntensity;
	case RunInfoPI::m_fedIN                 : return float((payload->m_fed_in).size());
	case RunInfoPI::END_OF_TYPES            : return ret;
	default : return ret;
	}
      }; 

      h2_RunInfoParameters->GetXaxis()->SetBinLabel(1,"Value");
      h2_RunInfoState->GetXaxis()->SetBinLabel(1,"Value");

      unsigned int yBin=11;
      for(int foo = RunInfoPI::m_run; foo != RunInfoPI::END_OF_TYPES; foo++ ){
	RunInfoPI::parameters param = static_cast<RunInfoPI::parameters>(foo);
	std::string theLabel =  RunInfoPI::getStringFromTypeEnum(param);
	h2_RunInfoState->GetYaxis()->SetBinLabel(yBin,theLabel.c_str());
	h2_RunInfoParameters->GetYaxis()->SetBinLabel(yBin,theLabel.c_str());
	h2_RunInfoParameters->SetBinContent(1,yBin,cutFunctor(param)); 
	// non-fake payload
	if((payload->m_run)!=-1){
	  if ((payload->m_avg_current)<=-1){
	    // go in error state
	    h2_RunInfoState->SetBinContent(1,yBin,0.);
	  } else {
	    // all is OK
	    h2_RunInfoState->SetBinContent(1,yBin,1.);
	  }
	} else {
	  // this is a fake payload
	  h2_RunInfoState->SetBinContent(1,yBin,0.9);
	}
	yBin--;

      }
      
      h2_RunInfoParameters->GetXaxis()->LabelsOption("h");
      h2_RunInfoParameters->GetYaxis()->SetLabelSize(0.05);
      h2_RunInfoParameters->GetXaxis()->SetLabelSize(0.05);
      h2_RunInfoParameters->SetMarkerSize(1.5);

      h2_RunInfoState->GetXaxis()->LabelsOption("h");
      h2_RunInfoState->GetYaxis()->SetLabelSize(0.05);
      h2_RunInfoState->GetXaxis()->SetLabelSize(0.05);
      h2_RunInfoState->SetMarkerSize(1.5);

      RunInfoPI::reportSummaryMapPalette(h2_RunInfoState.get());
      h2_RunInfoState->Draw("col");

      h2_RunInfoParameters->Draw("TEXTsame");
      
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(12);
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.1,  0.98,"RunInfo parameters:");
      t1.DrawLatex(0.1,  0.95,"payload:");

      t1.SetTextFont(42);
      t1.SetTextColor(4);
      t1.DrawLatex(0.37, 0.982,Form("IOV %s",std::to_string(+std::get<0>(iov)).c_str()));
      t1.DrawLatex(0.21, 0.952,Form(" %s",(std::get<1>(iov)).c_str()));

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
    

  };

  /************************************************
    time history of Magnet currents from RunInfo
  *************************************************/

  template<RunInfoPI::parameters param> class RunInfoCurrentHistory : public cond::payloadInspector::HistoryPlot<RunInfo,float> {
  public:
    RunInfoCurrentHistory() : cond::payloadInspector::HistoryPlot<RunInfo,float>(getStringFromTypeEnum(param),getStringFromTypeEnum(param)+" value"){}
   ~RunInfoCurrentHistory() override = default;

    float getFromPayload( RunInfo& payload ) override{

      float fieldIntensity = RunInfoPI::theBField(payload.m_avg_current);

      switch(param){
      case RunInfoPI::m_start_current 	     : return payload.m_start_current; 	    
      case RunInfoPI::m_stop_current	     : return payload.m_stop_current;	    
      case RunInfoPI::m_avg_current	     : return payload.m_avg_current;	       
      case RunInfoPI::m_max_current	     : return payload.m_max_current;	     
      case RunInfoPI::m_min_current	     : return payload.m_min_current;	     
      case RunInfoPI::m_BField               : return fieldIntensity;
      default:
	edm::LogWarning("LogicError") << "Unknown parameter: " <<  param; 
	break;
      }
      
    } // payload

    /************************************************/
    std::string getStringFromTypeEnum (const RunInfoPI::parameters &parameter){
      switch(parameter){
      case RunInfoPI::m_start_current 	  : return "Magent start current [A]";
      case RunInfoPI::m_stop_current	  : return "Magnet stop current [A]";  
      case RunInfoPI::m_avg_current	  : return "Magnet average current [A]";  
      case RunInfoPI::m_max_current	  : return "Magnet max current [A]";
      case RunInfoPI::m_min_current	  : return "Magnet min current [A]";
      case RunInfoPI::m_BField               : return "B-field intensity [T]";
      default: return "should never be here";
      }
    }
  };

  typedef RunInfoCurrentHistory<RunInfoPI::m_start_current>  RunInfoStartCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_stop_current>   RunInfoStopCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_avg_current>    RunInfoAverageCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_max_current>    RunInfoMaxCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_min_current>    RunInfoMinCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_BField>         RunInfoBFieldHistory;               

} // close namespace

PAYLOAD_INSPECTOR_MODULE( RunInfo ){
  PAYLOAD_INSPECTOR_CLASS( RunInfoTest );
  PAYLOAD_INSPECTOR_CLASS( RunInfoParameters ) ;
  PAYLOAD_INSPECTOR_CLASS( RunInfoStopCurrentHistory   ) ;
  PAYLOAD_INSPECTOR_CLASS( RunInfoAverageCurrentHistory) ;
  PAYLOAD_INSPECTOR_CLASS( RunInfoMaxCurrentHistory    ) ;
  PAYLOAD_INSPECTOR_CLASS( RunInfoMinCurrentHistory    ) ;
  PAYLOAD_INSPECTOR_CLASS( RunInfoBFieldHistory        ) ;
}
