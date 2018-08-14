#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <memory>
#include <sstream>
#include "TCanvas.h"
#include "TH2F.h"

namespace {

  enum parameters {X,
		   Y,
		   Z,
		   sigmaX,
		   sigmaY,
		   sigmaZ,
		   dxdz,
		   dydz,    
		   END_OF_TYPES};

  class BeamSpot_hx : public cond::payloadInspector::HistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_hx(): cond::payloadInspector::HistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs run number", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };

  class BeamSpot_rhx : public cond::payloadInspector::RunHistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_rhx(): cond::payloadInspector::RunHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs run number", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };
  class BeamSpot_x : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_x(): cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs time", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };

  class BeamSpot_y : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >{
  public:
    BeamSpot_y(): cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "y vs time", "y"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetY(),payload.GetYError());
    }
  };

  class BeamSpot_xy : public cond::payloadInspector::ScatterPlot<BeamSpotObjects,double,double>{
  public:
    BeamSpot_xy(): cond::payloadInspector::ScatterPlot<BeamSpotObjects,double,double>("BeamSpot x vs y","x","y" ){
    }

    std::tuple<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_tuple( payload.GetX(), payload.GetY() );
    }
  };

  /************************************************
    Display of Beam Spot parameters
  *************************************************/
  class BeamSpotParameters : public cond::payloadInspector::PlotImage<BeamSpotObjects> {
  public:
    BeamSpotParameters() : cond::payloadInspector::PlotImage<BeamSpotObjects>( "Display of BeamSpot parameters" ){
    setSingleIov( true );
    }
  
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<BeamSpotObjects> payload = fetchPayload( std::get<1>(iov) );
      
      TCanvas canvas("Beam Spot Parameters Summary","BeamSpot Parameters summary",1000,1000); 
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.15);
      canvas.SetRightMargin(0.03);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_BSParameters = std::unique_ptr<TH2F>(new TH2F("Parameters","BeamSpot parameters summary",2,0.0,2.0,8,0,8.));
      h2_BSParameters->SetStats(false);
      
      std::function<double(parameters,bool)> cutFunctor = [&payload](parameters my_param,bool isError) {	
	double ret(-999.);
	if(!isError){
	  switch(my_param){
	  case X      : return payload->GetX();
	  case Y      : return payload->GetY();	    
	  case Z      : return payload->GetZ();	    
	  case sigmaX : return payload->GetBeamWidthX(); 
	  case sigmaY : return payload->GetBeamWidthY(); 
	  case sigmaZ : return payload->GetSigmaZ(); 
	  case dxdz   : return payload->Getdxdz();   
	  case dydz   : return payload->Getdydz();   
	  case END_OF_TYPES : return ret;
	  default : return ret;
	  }
	} else {
	  switch(my_param){
	  case X      : return payload->GetXError();
	  case Y      : return payload->GetYError();	    
	  case Z      : return payload->GetZError();	    
	  case sigmaX : return payload->GetBeamWidthXError(); 
	  case sigmaY : return payload->GetBeamWidthYError(); 
	  case sigmaZ : return payload->GetSigmaZError(); 
	  case dxdz   : return payload->GetdxdzError();   
	  case dydz   : return payload->GetdydzError();   
	  case END_OF_TYPES : return ret;
	  default : return ret;
	  }
	}
      };

      h2_BSParameters->GetXaxis()->SetBinLabel(1,"Value");
      h2_BSParameters->GetXaxis()->SetBinLabel(2,"Error");

      unsigned int yBin=8;
      for(int foo = parameters::X; foo != parameters::END_OF_TYPES; foo++ ){
	parameters param = static_cast<parameters>(foo);
	std::string theLabel =  getStringFromTypeEnum(param);
	h2_BSParameters->GetYaxis()->SetBinLabel(yBin,theLabel.c_str());
	h2_BSParameters->SetBinContent(1,yBin,cutFunctor(param,false)); 
	h2_BSParameters->SetBinContent(2,yBin,cutFunctor(param,true)); 
	yBin--;
      }
      
      h2_BSParameters->GetXaxis()->LabelsOption("h");
      
      h2_BSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BSParameters->GetXaxis()->SetLabelSize(0.05);

      h2_BSParameters->SetMarkerSize(1.5);
      h2_BSParameters->Draw("TEXT");
      
      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    /************************************************/
    std::string getStringFromTypeEnum (const parameters &parameter){
      switch(parameter){
      case X      : return "X [cm]";
      case Y      : return "Y [cm]";	    
      case Z      : return "Z [cm]";	    
      case sigmaX : return "#sigma_{X} [cm]"; 
      case sigmaY : return "#sigma_{Y} [cm]"; 
      case sigmaZ : return "#sigma_{Z} [cm]"; 
      case dxdz   : return "#frac{dX}{dZ} [rad]";   
      case dydz   : return "#frac{dY}{dZ} [rad]";   
      default: return "should never be here";
      }
    }
  };

} // close namespace

PAYLOAD_INSPECTOR_MODULE( BeamSpot ){
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_hx );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_rhx );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_x );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_y );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_xy );
  PAYLOAD_INSPECTOR_CLASS( BeamSpotParameters );
}
