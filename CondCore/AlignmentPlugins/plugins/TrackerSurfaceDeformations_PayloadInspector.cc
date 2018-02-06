/*!
  \file TrackerSurfaceDeformations_PayloadInspector
  \Payload Inspector Plugin for Tracker Surface Deformations
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/02/01 15:57:24 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h" 
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// needed for mapping
#include "CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h"
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

namespace {

  class TrackerSurfaceDeformationsTest : public cond::payloadInspector::Histogram1D<AlignmentSurfaceDeformations> {
    
  public:
     TrackerSurfaceDeformationsTest () : cond::payloadInspector::Histogram1D<AlignmentSurfaceDeformations>("TrackerSurfaceDeformationsTest",
													   "TrackerSurfaceDeformationsTest",2,0.0,2.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      for ( auto const & iov: iovs) {
	std::shared_ptr<AlignmentSurfaceDeformations> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  
	  int i=0;
	  auto listOfItems = payload->items();
	  std::cout << "items size:" << listOfItems.size() << std::endl;

	  for (const auto &item : listOfItems){
	    std::cout<< i << " "<< item.m_rawId <<" Det: "<<  DetId(item.m_rawId).subdetId() << " " << item.m_index << std::endl; 
 	    const auto beginEndPair = payload->parameters(i);
	    std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);
	    std::cout << "params.size()" << params.size() << std::endl;
	    for(const auto &par : params){
	      std::cout << par << std::endl;
	    }
	    i++;
	  }

	}// payload
      }// iovs
      return true;
    }// fill
  };

  //*******************************************************
  // Summary of the parameters for each partition for 1 IOV
  //*******************************************************

  template <AlignmentPI::partitions q> class TrackerAlignmentSurfaceDeformationsSummary : public cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations> {
  public:
    TrackerAlignmentSurfaceDeformationsSummary() : cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations>( "Details for "+AlignmentPI::getStringFromPart(q)){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{


      auto iov = iovs.front();
      std::shared_ptr<AlignmentSurfaceDeformations> payload = fetchPayload( std::get<1>(iov) );
      auto listOfItems = payload->items();

      int canvas_w = (q<=4) ? 1200 : 1800;
      std::pair<int,int> divisions = (q<=4) ? std::make_pair(3,1) : std::make_pair(7,2);
 
      TCanvas canvas("Summary","Summary",canvas_w,600); 
      canvas.Divide(divisions.first,divisions.second);

      std::array<std::unique_ptr<TH1F>,14> summaries;
      for (int nPar=0;nPar<14;nPar++){
	summaries[nPar] = std::make_unique<TH1F>(Form("h_summary_%i",nPar),Form("Surface Deformation parameter %i;parameter %i size;# modules",nPar,nPar),100,-0.1,0.1);
      }
      
      int i=0;
      for (const auto &item : listOfItems){
	int subid = DetId(item.m_rawId).subdetId();
	auto thePart = static_cast<AlignmentPI::partitions>(subid);

	const auto beginEndPair = payload->parameters(i);
	std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);

	// increase the counter before continuing if partition doesn't match, 
	// otherwise the cound of the parameter count gets altered 
	i++;

	// return if not the right partition
	if(thePart!=q) continue;
	int nPar=0;

	for(const auto &par : params){
	  summaries[nPar]->Fill(par);
	  nPar++;
	} // ends loop on the parameters
      } // ends loop on the item vector

      TLatex t1;

      for(int c=1;c<=(divisions.first*divisions.second);c++){
	canvas.cd(c)->SetLogy();
	canvas.cd(c)->SetTopMargin(0.02);
	canvas.cd(c)->SetBottomMargin(0.15);
	canvas.cd(c)->SetLeftMargin(0.14);
	canvas.cd(c)->SetRightMargin(0.03);

	summaries[c-1]->SetLineWidth(2);
	AlignmentPI::makeNicePlotStyle(summaries[c-1].get(),kBlack);
	summaries[c-1]->Draw("same");
	summaries[c-1]->SetTitle("");

	AlignmentPI::makeNiceStats(summaries[c-1].get(),q,kBlack);

	canvas.cd(c);

	t1.SetTextAlign(21);
	t1.SetTextSize(0.06);
	t1.SetTextColor(kBlue);
	t1.DrawLatexNDC(0.32, 0.95, Form("IOV: %s ",std::to_string(std::get<0>(iov)).c_str()));

      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
      
    }
  };
 
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::BPix>  BPixSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::FPix>  FPixSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TIB>   TIBSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TID>   TIDSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TOB>   TOBSurfaceDeformationsSummary;
  typedef TrackerAlignmentSurfaceDeformationsSummary<AlignmentPI::TEC>   TECSurfaceDeformationsSummary;  

  //*******************************************************
  // Comparison of the parameters for each partition for 1 IOV
  //*******************************************************

  template <AlignmentPI::partitions q> class TrackerAlignmentSurfaceDeformationsComparison : public cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations> {
  public:
    TrackerAlignmentSurfaceDeformationsComparison() : cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations>( "Details for "+AlignmentPI::getStringFromPart(q)){
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
      
      std::shared_ptr<AlignmentSurfaceDeformations> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::string lastIOVsince  = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      int canvas_w = (q<=4) ? 1600 : 1800;
      std::pair<int,int> divisions = (q<=4) ? std::make_pair(3,1) : std::make_pair(7,2);
 
      TCanvas canvas("Comparison","Comparison",canvas_w,600); 
      canvas.Divide(divisions.first,divisions.second);

      std::array<std::unique_ptr<TH1F>,14> deltas;
      for (int nPar=0;nPar<14;nPar++){
	deltas[nPar] = std::make_unique<TH1F>(Form("h_summary_%i",nPar),Form("Surface Deformation #Delta parameter %i;#Deltapar_{%i};# modules",nPar,nPar),100,-0.05,0.05);
      }
      
      assert(first_listOfItems.size() == last_listOfItems.size());

      for (unsigned int i=0;i<first_listOfItems.size();i++){
	auto first_id = first_listOfItems[i].m_rawId;
	int subid = DetId(first_listOfItems[i].m_rawId).subdetId();
	auto thePart = static_cast<AlignmentPI::partitions>(subid);
	if(thePart!=q) continue;

	const auto f_beginEndPair = first_payload->parameters(i);
	std::vector<align::Scalar> first_params(f_beginEndPair.first,f_beginEndPair.second);

	for (unsigned int j=0;j<last_listOfItems.size();j++){
	  auto last_id = last_listOfItems[j].m_rawId;
	  if(first_id == last_id){

	    const auto l_beginEndPair = last_payload->parameters(j);
	    std::vector<align::Scalar> last_params(l_beginEndPair.first,l_beginEndPair.second);

	    assert(first_params.size()==last_params.size());

	    for(unsigned int nPar=0;nPar<first_params.size();nPar++){
	      deltas[nPar]->Fill(last_params[nPar]-first_params[nPar]);
	    }
	    break;
	  }
	}

      } // ends loop on the item vector

      TLatex t1;

      for(int c=1;c<=(divisions.first*divisions.second);c++){
	canvas.cd(c)->SetLogy();
	canvas.cd(c)->SetTopMargin(0.015);
	canvas.cd(c)->SetBottomMargin(0.13);
	canvas.cd(c)->SetLeftMargin(0.14);
	canvas.cd(c)->SetRightMargin(0.03);

	deltas[c-1]->SetLineWidth(2);
	AlignmentPI::makeNicePlotStyle(deltas[c-1].get(),kBlack);
	deltas[c-1]->Draw("same");
	deltas[c-1]->SetTitle("");

	AlignmentPI::makeNiceStats(deltas[c-1].get(),q,kBlack);

	canvas.cd(c);
	t1.SetTextAlign(21);
	t1.SetTextSize(0.045);
	t1.SetTextColor(kBlue);
	t1.DrawLatexNDC(0.4, 0.95, Form("#DeltaIOV: %s - %s ",lastIOVsince.c_str(),firstIOVsince.c_str()));

      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
      
    }
  };
 
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::BPix>  BPixSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::FPix>  FPixSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TIB>   TIBSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TID>   TIDSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TOB>   TOBSurfaceDeformationsComparison;
  typedef TrackerAlignmentSurfaceDeformationsComparison<AlignmentPI::TEC>   TECSurfaceDeformationsComparison;  
  

  // /************************************************
  //   TrackerMap of single parameter
  // *************************************************/
  template<unsigned int par> class SurfaceDeformationTrackerMap : public cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations> {
  public:
    SurfaceDeformationTrackerMap() : cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations>( "Tracker Map of Tracker Surface deformations - parameter: "+ std::to_string(par) ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<AlignmentSurfaceDeformations> payload = fetchPayload( std::get<1>(iov) );
      auto listOfItems = payload->items();

      std::string titleMap = "Surface deformation parameter "+std::to_string(par)+" value (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("Surface Deformations"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);
   
      std::map<unsigned int,float> surfDefMap;
      
      bool isPhase0(false);
      if(listOfItems.size()==AlignmentPI::phase0size) isPhase0 = true;

      int iDet=0;
      for (const auto &item : listOfItems){

	// fill the tracker map
	int subid = DetId(item.m_rawId).subdetId();

	if(DetId(item.m_rawId).det() != DetId::Tracker){
	  edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector") << "Encountered invalid Tracker DetId:" << item.m_rawId <<" - terminating ";
	  return false;
	}

	const auto beginEndPair = payload->parameters(iDet);
	std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);

	iDet++;
	// protect against exceeding the vector of parameter size
	if(par>=params.size()) continue;

	if(isPhase0){
	  tmap->addPixel(true);
	  tmap->fill(item.m_rawId,params.at(par));
	  surfDefMap[item.m_rawId]=params.at(par);
	} else {
	  if(subid!=1 && subid!=2){
	    tmap->fill(item.m_rawId,params.at(par));
	    surfDefMap[item.m_rawId]=params.at(par);
	  }
	}
      } // loop over detIds

      //=========================
   
      // saturate at 1.5sigma
      auto autoRange = AlignmentPI::getTheRange(surfDefMap,1.5); //tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);
      // protect against uniform values (Surface deformations are defined positive)
      if (autoRange.first!=autoRange.second){
	tmap->save(true,autoRange.first,autoRange.second,fileName);
      } else {
	if(autoRange.first==0.){
	  tmap->save(true,0.,1.,fileName);
	} else {
	  tmap->save(true,autoRange.first,autoRange.second,fileName);
	}
      }

      return true;
    }
  };

  typedef SurfaceDeformationTrackerMap<0>   SurfaceDeformationParameter0TrackerMap;
  typedef SurfaceDeformationTrackerMap<1>   SurfaceDeformationParameter1TrackerMap;
  typedef SurfaceDeformationTrackerMap<2>   SurfaceDeformationParameter2TrackerMap;
  typedef SurfaceDeformationTrackerMap<3>   SurfaceDeformationParameter3TrackerMap;
  typedef SurfaceDeformationTrackerMap<4>   SurfaceDeformationParameter4TrackerMap;
  typedef SurfaceDeformationTrackerMap<5>   SurfaceDeformationParameter5TrackerMap;
  typedef SurfaceDeformationTrackerMap<6>   SurfaceDeformationParameter6TrackerMap;
  typedef SurfaceDeformationTrackerMap<7>   SurfaceDeformationParameter7TrackerMap;
  typedef SurfaceDeformationTrackerMap<8>   SurfaceDeformationParameter8TrackerMap;
  typedef SurfaceDeformationTrackerMap<9>   SurfaceDeformationParameter9TrackerMap;
  typedef SurfaceDeformationTrackerMap<10>  SurfaceDeformationParameter10TrackerMap;
  typedef SurfaceDeformationTrackerMap<11>  SurfaceDeformationParameter11TrackerMap;
  typedef SurfaceDeformationTrackerMap<12>  SurfaceDeformationParameter12TrackerMap;
  
  // /************************************************
  //   TrackerMap of single parameter
  // *************************************************/
  template<unsigned int par> class SurfaceDeformationsTkMapDelta : public cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations> {
  public:
    SurfaceDeformationsTkMapDelta() : cond::payloadInspector::PlotImage<AlignmentSurfaceDeformations>( "Tracker Map of Tracker Surface deformations - parameter: "+ std::to_string(par) ){
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
      
      std::shared_ptr<AlignmentSurfaceDeformations> last_payload  = fetchPayload( std::get<1>(lastiov) );
      std::shared_ptr<AlignmentSurfaceDeformations> first_payload = fetchPayload( std::get<1>(firstiov) );
      
      std::string lastIOVsince  = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      auto first_listOfItems = first_payload->items();
      auto last_listOfItems = last_payload->items();

      std::string titleMap = "#Delta Surface deformation parameter "+std::to_string(par)+" (IOV : "+std::to_string(std::get<0>(lastiov))+"- "+std::to_string(std::get<0>(firstiov))+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("Surface Deformations #Delta"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);
      
      std::map<unsigned int,float> surfDefMap;

      assert(first_listOfItems.size() == last_listOfItems.size());
      
      bool isPhase0(false);
      if(first_listOfItems.size()==AlignmentPI::phase0size) isPhase0 = true;
      if(isPhase0) tmap->addPixel(true);

      for (unsigned int i=0;i<first_listOfItems.size();i++){
	auto first_id = first_listOfItems[i].m_rawId;

	if(DetId(first_id).det() != DetId::Tracker){
	  edm::LogWarning("TrackerSurfaceDeformations_PayloadInspector") << "Encountered invalid Tracker DetId:" << first_id <<" - terminating ";
	  return false;
	}

	int subid = DetId(first_listOfItems[i].m_rawId).subdetId();

	const auto f_beginEndPair = first_payload->parameters(i);
	std::vector<align::Scalar> first_params(f_beginEndPair.first,f_beginEndPair.second);

	// protect against exceeding the vector of parameter size
	if(par>=first_params.size()) continue;

	for (unsigned int j=0;j<last_listOfItems.size();j++){
	  auto last_id = last_listOfItems[j].m_rawId;
	  if(first_id == last_id){

	    const auto l_beginEndPair = last_payload->parameters(j);
	    std::vector<align::Scalar> last_params(l_beginEndPair.first,l_beginEndPair.second);

	    assert(first_params.size()==last_params.size());

	    for(unsigned int nPar=0;nPar<first_params.size();nPar++){

	      float delta = last_params.at(par) - first_params.at(par);

	      if(isPhase0){
		tmap->addPixel(true);
		tmap->fill(first_id,delta);
		surfDefMap[first_id]=delta;
	      } else {
		// fill pixel map only for phase-0 (in lack of a dedicate phase-I map)
		if(subid!=1 && subid!=2){
		  tmap->fill(first_id,delta);
		  surfDefMap[first_id]=delta;
		}
	      } // if not phase-0 
	    } // loop on params
	    break;
	  } // match of the detIds
	} // loop on second list of items
      }// loop on first list of items
      
      //=========================
	 
      // saturate at 1.5sigma
      auto autoRange = AlignmentPI::getTheRange(surfDefMap,2.5); //tmap->getAutomaticRange();

      std::string fileName(m_imageFileName);
      // protect against uniform values (Surface deformations are defined positive)
      if (autoRange.first!=autoRange.second){
	tmap->save(true,autoRange.first,autoRange.second,fileName);
      } else {
	if(autoRange.first==0.){
	  tmap->save(true,0.,1.,fileName);
	} else {
	  tmap->save(true,autoRange.first,autoRange.second,fileName);
	}
      }
      
      return true;

    }
  };

  typedef SurfaceDeformationsTkMapDelta<0>   SurfaceDeformationParameter0TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<1>   SurfaceDeformationParameter1TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<2>   SurfaceDeformationParameter2TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<3>   SurfaceDeformationParameter3TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<4>   SurfaceDeformationParameter4TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<5>   SurfaceDeformationParameter5TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<6>   SurfaceDeformationParameter6TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<7>   SurfaceDeformationParameter7TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<8>   SurfaceDeformationParameter8TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<9>   SurfaceDeformationParameter9TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<10>  SurfaceDeformationParameter10TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<11>  SurfaceDeformationParameter11TkMapDelta;
  typedef SurfaceDeformationsTkMapDelta<12>  SurfaceDeformationParameter12TkMapDelta;


} // close namespace

PAYLOAD_INSPECTOR_MODULE(TrackerSurfaceDeformations){
  PAYLOAD_INSPECTOR_CLASS(TrackerSurfaceDeformationsTest);
  PAYLOAD_INSPECTOR_CLASS(BPixSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(FPixSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(TIBSurfaceDeformationsSummary);	 	
  PAYLOAD_INSPECTOR_CLASS(TIDSurfaceDeformationsSummary);	 
  PAYLOAD_INSPECTOR_CLASS(TOBSurfaceDeformationsSummary);	 
  PAYLOAD_INSPECTOR_CLASS(TECSurfaceDeformationsSummary);
  PAYLOAD_INSPECTOR_CLASS(BPixSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(FPixSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(TIBSurfaceDeformationsComparison);	 	
  PAYLOAD_INSPECTOR_CLASS(TIDSurfaceDeformationsComparison);	 
  PAYLOAD_INSPECTOR_CLASS(TOBSurfaceDeformationsComparison);	 
  PAYLOAD_INSPECTOR_CLASS(TECSurfaceDeformationsComparison);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9TrackerMap); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12TrackerMap);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter0TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter1TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter2TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter3TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter4TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter5TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter6TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter7TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter8TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter9TkMapDelta); 
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter10TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter11TkMapDelta);
  PAYLOAD_INSPECTOR_CLASS(SurfaceDeformationParameter12TkMapDelta);

}
