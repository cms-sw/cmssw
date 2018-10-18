/*!
  \file SiPixelQulity_PayloadInspector
  \Payload Inspector Plugin for SiPixelQuality
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/10/18 14:48:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <memory>
#include <sstream>
#include <iostream>

// include ROOT 
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

namespace {

  /************************************************
    test class
  *************************************************/

  class SiPixelQualityTest : public cond::payloadInspector::Histogram1D<SiPixelQuality> {
    
  public:
    SiPixelQualityTest() : cond::payloadInspector::Histogram1D<SiPixelQuality>("SiPixelQuality test",
									       "SiPixelQuality test", 10,0.0,10.0){
      Base::setSingleIov( true );
    }
    
    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      for ( auto const & iov: iovs) {
	std::shared_ptr<SiPixelQuality> payload = Base::fetchPayload( std::get<1>(iov) );
	if( payload.get() ){
	 
	  fillWithValue(1.);
	 
	  auto theDisabledModules = payload->getBadComponentList();
	  for (const auto &mod : theDisabledModules){
	    int BadRocCount(0);
	    for (unsigned short n = 0; n < 16; n++){
	      unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
	      if (mod.BadRocs & mask) BadRocCount++;
	    }
	    std::cout<<"detId:" <<  mod.DetID << " error type:" << mod.errorType << " BadRocs:"  << BadRocCount <<  std::endl;
	  }
	}// payload
      }// iovs
      return true;
    }// fill
  };
  
  /************************************************
    summary class
  *************************************************/

  class SiPixelQualityBadRocsSummary : public cond::payloadInspector::PlotImage<SiPixelQuality> {

  public:
    SiPixelQualityBadRocsSummary() : cond::payloadInspector::PlotImage<SiPixelQuality>("SiPixel Quality Summary"){
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      std::vector<std::tuple<cond::Time_t,cond::Hash> > sorted_iovs = iovs;

      for(const auto &iov: iovs){
	std::shared_ptr<SiPixelQuality> payload = fetchPayload( std::get<1>(iov) );
	auto unpacked = unpack(std::get<0>(iov));

	std::cout<<"======================= " << unpacked.first <<" : "<< unpacked.second  << std::endl;
	auto theDisabledModules = payload->getBadComponentList();
	  for (const auto &mod : theDisabledModules){
	    std::cout<<"detId: " <<  mod.DetID << " |error type: " << mod.errorType << " |BadRocs: "  <<  mod.BadRocs <<  std::endl;
	  }
      }

      //=========================
      TCanvas canvas("Partion summary","partition summary",1200,1000);
      canvas.cd();
      canvas.SetBottomMargin(0.11);
      canvas.SetLeftMargin(0.13);
      canvas.SetRightMargin(0.05);
      canvas.Modified();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;

    }

    std::pair<unsigned int,unsigned int> unpack(cond::Time_t since){
      auto kLowMask = 0XFFFFFFFF;
      auto run  = (since >> 32);
      auto lumi = (since & kLowMask);
      return std::make_pair(run,lumi);
    }

  };

  /************************************************
    time history class
  *************************************************/

  class SiPixelQualityBadRocsTimeHistory : public cond::payloadInspector::TimeHistoryPlot<SiPixelQuality,std::pair<double,double> > {
    
  public:
    SiPixelQualityBadRocsTimeHistory() : cond::payloadInspector::TimeHistoryPlot<SiPixelQuality,std::pair<double,double> >("bad ROCs count vs time","bad ROCs count"){}

    std::pair<double,double> getFromPayload(SiPixelQuality& payload ) override{
      return std::make_pair(extractBadRocCount(payload),0.);
    }

    unsigned int extractBadRocCount(SiPixelQuality& payload){
      unsigned int BadRocCount(0);
      auto theDisabledModules = payload.getBadComponentList();
      for (const auto &mod : theDisabledModules){
	for (unsigned short n = 0; n < 16; n++){
	  unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
	  if (mod.BadRocs & mask) BadRocCount++;
	}
      }
      return BadRocCount;
    }
  };

} // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelQuality){
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsSummary);
  PAYLOAD_INSPECTOR_CLASS(SiPixelQualityBadRocsTimeHistory);
}
