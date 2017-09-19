/*!
  \file TrackerALignmentErrorExtended_PayloadInspector
  \Payload Inspector Plugin for Tracker Alignment Errors (APE)
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/07/10 10:59:24 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/DetId/interface/DetId.h"

// needed for the tracker map
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

// needed for mapping
#include "CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h"
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
    1d histogram of sqrt(d_ii) of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  template<AlignmentPI::index i> class TrackerAlignmentErrorExtendedValue : public cond::payloadInspector::Histogram1D<AlignmentErrorsExtended> {
    
  public:
    TrackerAlignmentErrorExtendedValue () : cond::payloadInspector::Histogram1D<AlignmentErrorsExtended>("TrackerAlignmentErrorExtendedValue",
													 "TrackerAlignmentErrorExtendedValue sqrt(d_{"+getStringFromIndex(i)+"})",500,0.0,500.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      
      for ( auto const & iov: iovs) {
	std::shared_ptr<AlignmentErrorsExtended> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	  
	  std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;
	  auto indices = AlignmentPI::getIndices(i);
	  
	  for (const auto& it : alignErrors ){
	    
	    CLHEP::HepSymMatrix errMatrix = it.matrix();

	    if(DetId(it.rawId()).det() != DetId::Tracker){
	      edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector") << "Encountered invalid Tracker DetId:" << it.rawId() <<" - terminating ";
	      return false;
	    }
	    
	    // to be used to fill the histogram
	    fillWithValue(sqrt(errMatrix[indices.first][indices.second])*AlignmentPI::cmToUm);	    
	  } // loop on the vector of modules
	}// payload
      }// iovs
      return true;
    }// fill
  };
  
  // diagonal elements
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXValue; 
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYValue; 
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZValue; 

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYValue; 
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZValue; 
  typedef TrackerAlignmentErrorExtendedValue<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZValue; 

  // /************************************************
  //   Summary spectra of sqrt(d_ii) of 1 IOV
  // *************************************************/
 
  template<AlignmentPI::index i> class TrackerAlignmentErrorExtendedSummary : public cond::payloadInspector::PlotImage<AlignmentErrorsExtended> {
  public:
    TrackerAlignmentErrorExtendedSummary() : cond::payloadInspector::PlotImage<AlignmentErrorsExtended>( "Summary per Tracker Partition of sqrt(d_{"+getStringFromIndex(i)+"}) of APE matrix" ){
      setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload( std::get<1>(iov) );
      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      const char * path_toTopologyXML = (alignErrors.size()==AlignmentPI::phase0size) ? "Geometry/TrackerCommonData/data/trackerParameters.xml" : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath(path_toTopologyXML).fullPath()); 

      auto indices = AlignmentPI::getIndices(i);

      TCanvas canvas("Partion summary","partition summary",1200,1000); 
      canvas.Divide(3,2);
      std::map<AlignmentPI::partitions,int> colormap;
      colormap[AlignmentPI::BPix] = kBlue;
      colormap[AlignmentPI::FPix] = kBlue+2;
      colormap[AlignmentPI::TIB]  = kRed;           
      colormap[AlignmentPI::TOB]  = kRed+2;
      colormap[AlignmentPI::TID]  = kRed+4;	    
      colormap[AlignmentPI::TEC]  = kRed+6; 	    
      
      std::map<AlignmentPI::partitions,std::shared_ptr<TH1F> > APE_spectra; 
      std::vector<AlignmentPI::partitions> parts = {AlignmentPI::BPix,AlignmentPI::FPix,AlignmentPI::TIB,AlignmentPI::TID,AlignmentPI::TOB,AlignmentPI::TEC};
      
      auto s_index = getStringFromIndex(i);

      for ( const auto &part : parts){

	std::string s_part = AlignmentPI::getStringFromPart(part);
	
  	APE_spectra[part]   = std::make_shared<TH1F>(Form("hAPE_%s",s_part.c_str()),Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules",s_part.c_str(),s_index.c_str()),200,-10.,200.);       
	
      }

      for (const auto& it : alignErrors ){
	    
      	CLHEP::HepSymMatrix errMatrix = it.matrix();
      	int subid = DetId(it.rawId()).subdetId();
	double matrixElement = sqrt(errMatrix[indices.first][indices.second]);

	if(DetId(it.rawId()).det() != DetId::Tracker){
	  edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector") << "Encountered invalid Tracker DetId:" << it.rawId() <<" - terminating ";
	  return false;
	}

      	switch(subid){
      	case 1 : 
      	  APE_spectra[AlignmentPI::BPix]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
      	  break;
      	case 2 : 
      	  APE_spectra[AlignmentPI::FPix]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
      	  break;
      	case 3 : 
	  if(!tTopo.tibIsDoubleSide(it.rawId())){ // no glued DetIds
	    APE_spectra[AlignmentPI::TIB]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
	  }
	  break;
	case 4 : 
	  if(!tTopo.tidIsDoubleSide(it.rawId())){ // no glued DetIds
	    APE_spectra[AlignmentPI::TID]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
	  }
      	  break;
      	case 5 :
	  if(!tTopo.tobIsDoubleSide(it.rawId())){ // no glued DetIds
	    APE_spectra[AlignmentPI::TOB]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
	  }
      	  break;
      	case 6 :
	  if(!tTopo.tecIsDoubleSide(it.rawId())){ // no glued DetIds
	    APE_spectra[AlignmentPI::TEC]->Fill(std::min(200.,matrixElement)*AlignmentPI::cmToUm);
	  }
      	  break;
      	default : std::cout<<"will do nothing"<< std::endl;
      	  break;
      	}
      }
      
      int c_index=1;
      for (const auto &part : parts){
      	canvas.cd(c_index)->SetLogy();
	canvas.cd(c_index)->SetTopMargin(0.03);
	canvas.cd(c_index)->SetBottomMargin(0.15);
	canvas.cd(c_index)->SetLeftMargin(0.14);
	canvas.cd(c_index)->SetRightMargin(0.05);
      	APE_spectra[part]->SetLineWidth(2);
	AlignmentPI::makeNicePlotStyle(APE_spectra[part].get(),colormap[part]);
      	APE_spectra[part]->Draw("HIST");
	AlignmentPI::makeNiceStats(APE_spectra[part].get(),part,colormap[part]);
      	c_index++;
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXSummary; 
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYSummary; 
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZSummary; 

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYSummary; 
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZSummary; 
  typedef TrackerAlignmentErrorExtendedSummary<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZSummary; 

  // /************************************************
  //   TrackerMap of sqrt(d_ii) of 1 IOV
  // *************************************************/
  template<AlignmentPI::index i> class TrackerAlignmentErrorExtendedTrackerMap : public cond::payloadInspector::PlotImage<AlignmentErrorsExtended> {
  public:
    TrackerAlignmentErrorExtendedTrackerMap() : cond::payloadInspector::PlotImage<AlignmentErrorsExtended>( "Tracker Map of sqrt(d_{"+getStringFromIndex(i)+"}) of APE matrix" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload( std::get<1>(iov) );

      std::string titleMap = "APE #sqrt{d_{"+getStringFromIndex(i)+"}} value (payload : "+std::get<1>(iov)+")";

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("APE_dii"));
      tmap->setTitle(titleMap);
      tmap->setPalette(1);
   
      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      auto indices = AlignmentPI::getIndices(i);

      bool isPhase0(false);
      if(alignErrors.size()==AlignmentPI::phase0size) isPhase0 = true;
      
      for (const auto& it : alignErrors ){

	CLHEP::HepSymMatrix errMatrix = it.matrix();

	// fill the tracker map

	int subid = DetId(it.rawId()).subdetId();	    

	if(DetId(it.rawId()).det() != DetId::Tracker){
	  edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector") << "Encountered invalid Tracker DetId:" << it.rawId() <<" - terminating ";
	  return false;
	}

	if(isPhase0){
	  tmap->addPixel(true);
	  tmap->fill(it.rawId(),sqrt(errMatrix[indices.first][indices.second])*AlignmentPI::cmToUm);
	} else {
	  if(subid!=1 && subid!=2){
	    tmap->fill(it.rawId(),sqrt(errMatrix[indices.first][indices.second])*AlignmentPI::cmToUm);
	  }
	}
      } // loop over detIds

      //=========================
   
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  // diagonal elements
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XX> TrackerAlignmentErrorExtendedXXTrackerMap; 
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::YY> TrackerAlignmentErrorExtendedYYTrackerMap; 
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::ZZ> TrackerAlignmentErrorExtendedZZTrackerMap; 

  // off-diagonal elements
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XY> TrackerAlignmentErrorExtendedXYTrackerMap; 
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::XZ> TrackerAlignmentErrorExtendedXZTrackerMap; 
  typedef TrackerAlignmentErrorExtendedTrackerMap<AlignmentPI::YZ> TrackerAlignmentErrorExtendedYZTrackerMap; 
   
  // /************************************************
  //  Pixel details of 1 IOV
  // *************************************************/
  class TrackerAlignmentErrorExtendedPixelDetail : public cond::payloadInspector::PlotImage<AlignmentErrorsExtended> {
  public:
    TrackerAlignmentErrorExtendedPixelDetail() : cond::payloadInspector::PlotImage<AlignmentErrorsExtended>( "Details for BPix" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<AlignmentErrorsExtended> payload = fetchPayload( std::get<1>(iov) );
  
      std::vector<AlignTransformErrorExtended> alignErrors = payload->m_alignError;

      const char * path_toTopologyXML = (alignErrors.size()==AlignmentPI::phase0size) ? "Geometry/TrackerCommonData/data/trackerParameters.xml" : "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
      TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXML(edm::FileInPath(path_toTopologyXML).fullPath()); 

      TCanvas canvas("Pixel summary","Pixel summary",1200,1200); 
      canvas.Divide(3,2);

      std::map< std::pair<AlignmentPI::index,AlignmentPI::regions>,std::shared_ptr<TH1F> > APE_spectraByRegion;
      std::map< AlignmentPI::index,std::shared_ptr<TH1F> > summaries; 

      for(int k = AlignmentPI::XX; k<=AlignmentPI::ZZ;k++){

	AlignmentPI::index coord = (AlignmentPI::index) k;
	std::string s_coord = AlignmentPI::getStringFromIndex(coord);

	summaries[coord] = std::make_shared<TH1F>(Form("Summary_%s",s_coord.c_str()),Form("Summary for d_{%s} APE;;APE d_{%s} [#mum]",s_coord.c_str(),s_coord.c_str()),14,0,14);

	for(int j = AlignmentPI::BPixL1o; j<=AlignmentPI::FPixpL3; j++){

	  AlignmentPI::regions part = (AlignmentPI::regions) j;  
	  std::string s_part =  AlignmentPI::getStringFromRegionEnum(part);
	  
	  auto hash = std::make_pair(coord,part);
	  
	  APE_spectraByRegion[hash] = std::make_shared<TH1F>(Form("hAPE_%s_%s",s_coord.c_str(),s_part.c_str()),Form(";%s APE #sqrt{d_{%s}} [#mum];n. of modules",s_part.c_str(),s_coord.c_str()),1000,0.,1000.);       
	  	 
	}
      }

      // loop on the vector of errors
      for (const auto& it : alignErrors ){

	CLHEP::HepSymMatrix errMatrix = it.matrix();
	
	if(DetId(it.rawId()).det() != DetId::Tracker){
	  edm::LogWarning("TrackerAlignmentErrorExtended_PayloadInspector") << "Encountered invalid Tracker DetId:" << it.rawId() <<" - terminating ";
	  return false;
	}

	int subid = DetId(it.rawId()).subdetId();	    
	if(subid!=1 && subid!=2) continue;

	// fill the struct
	AlignmentPI::topolInfo t_info_fromXML;
	t_info_fromXML.init();
	t_info_fromXML.m_rawid = it.rawId();
	DetId detid(t_info_fromXML.m_rawid);
	t_info_fromXML.m_subdetid = subid;
	t_info_fromXML.fillGeometryInfo(detid,tTopo);
	t_info_fromXML.printAll();

	AlignmentPI::regions thePart = t_info_fromXML.filterThePartition();
	std::cout<< AlignmentPI::getStringFromRegionEnum(thePart) << std::endl;
	
	for(int k = AlignmentPI::XX; k<=AlignmentPI::ZZ;k++){
	
	  AlignmentPI::index coord = (AlignmentPI::index) k;
	  auto indices = AlignmentPI::getIndices(coord);
	  auto hash = std::make_pair(coord,thePart);

	  APE_spectraByRegion[hash]->Fill(sqrt(errMatrix[indices.first][indices.second])*AlignmentPI::cmToUm);

	} // loop on the coordinate indices

      } // loop over detIds

      for(int k = AlignmentPI::XX; k<=AlignmentPI::ZZ;k++){
	AlignmentPI::index coord = (AlignmentPI::index) k;
	for(int j = AlignmentPI::BPixL1o; j<=AlignmentPI::FPixpL3; j++){
	  AlignmentPI::regions part = (AlignmentPI::regions) j;  
	  auto hash = std::make_pair(coord,part);
	  summaries[coord]->GetXaxis()->SetBinLabel(j+1, AlignmentPI::getStringFromRegionEnum(part).c_str());
	  summaries[coord]->SetBinContent(j+1,APE_spectraByRegion[hash]->GetMean());
	  summaries[coord]->SetBinError(j+1,APE_spectraByRegion[hash]->GetRMS());
	  AlignmentPI::makeNicePlotStyle(summaries[coord].get(),kBlue);
	  summaries[coord]->GetXaxis()->LabelsOption("v");
	  summaries[coord]->GetXaxis()->SetLabelSize(0.6);
	}
     
	canvas.cd(k);
	canvas.cd(k)->SetTopMargin(0.07);
	canvas.cd(k)->SetBottomMargin(0.15);
	canvas.cd(k)->SetLeftMargin(0.18);
	canvas.cd(k)->SetRightMargin(0.05);
	summaries[coord]->Draw("E1");
	
      }

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };
 
} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(TrackerAlignmentErrorExtended){
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZValue);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZSummary);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXXTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYYTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedZZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXYTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedXZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedYZTrackerMap);
  PAYLOAD_INSPECTOR_CLASS(TrackerAlignmentErrorExtendedPixelDetail);
}
