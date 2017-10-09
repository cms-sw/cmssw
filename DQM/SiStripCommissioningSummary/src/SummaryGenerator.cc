#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorReadoutView.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <math.h>
#include "TH2F.h"
#include "TProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SummaryGenerator::SummaryGenerator( std::string name ) : 
  map_(),
  entries_(-1.),
  max_(-1.*sistrip::invalid_),
  min_(1.*sistrip::invalid_),
  label_(""),
  myName_(name)
{
  // TH1::SetDefaultSumw2(true); // use square of weights to calc error   
}

// -----------------------------------------------------------------------------
// 
SummaryGenerator* SummaryGenerator::instance( const sistrip::View& view ) {

  SummaryGenerator* generator = 0;
  if ( view == sistrip::CONTROL_VIEW ) {
    generator = new SummaryGeneratorControlView();
  } else if ( view == sistrip::READOUT_VIEW ) {
    generator = new SummaryGeneratorReadoutView();
  } else { generator = 0; }
  
  if ( generator ) {
    LogTrace(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]"
      << " Built \"" << generator->myName() << "\" object!";
  } else {
    edm::LogWarning(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]"
      << " Unexpected view: \"" 
      << SiStripEnumsAndStrings::view( view )
      << "\" Unable to build Generator!"
      << " Returning NULL pointer!";
  }
  
  return generator;

}

// -----------------------------------------------------------------------------
// 
std::string SummaryGenerator::name( const sistrip::RunType& run_type, 
				    const sistrip::Monitorable& mon, 
				    const sistrip::Presentation& pres,
				    const sistrip::View& view, 
				    const std::string& directory ) {
  std::stringstream ss;
  ss << sistrip::summaryHisto_ << sistrip::sep_; 
  ss << SiStripEnumsAndStrings::presentation( pres ) << sistrip::sep_; 
  ss << SiStripEnumsAndStrings::runType( run_type ) << sistrip::sep_;
  ss << SiStripEnumsAndStrings::view( view ) << sistrip::sep_;
  ss << SiStripEnumsAndStrings::monitorable( mon );
  
  //LogTrace(mlSummaryPlots_) 
  //<< "[SummaryGenerator::" << __func__ << "]"
  //<< " Histogram name: \"" << ss.str() << "\"";
  
  return ss.str();
}

// -----------------------------------------------------------------------------
// 
/*
  fix nbins for 1D distribution? to 1024? then change within summary
  methods with SetBins() methods? but must limit nbins to < 1024!!!

*/
TH1* SummaryGenerator::histogram( const sistrip::Presentation& pres,
				  const uint32_t& xbins ) {
  if ( !xbins ) { return 0; }

  TH1* summary = 0;
  if ( pres == sistrip::HISTO_1D ) { 
    summary = new TH1F( "", "", 1024, 0., static_cast<float>(1024) ); 
  } else if ( pres == sistrip::HISTO_2D_SUM ) { 
    summary = new TH1F( "", "", xbins, 0., static_cast<float>(xbins) ); 
  } else if ( pres == sistrip::HISTO_2D_SCATTER ) { 
    summary = new TH2F( "", "", 100*xbins, 0., static_cast<float>(100*xbins), 1025, 0., 1025. ); 
  } else if ( pres == sistrip::PROFILE_1D ) { 
    summary = new TProfile( "", "", xbins, 0., static_cast<float>(xbins), 0., 1025. ); 
  } else { summary = 0; }
  
  if ( summary ) {
    LogTrace(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]"
      << " Histogram name: \"" << summary->GetName() << "\"";
  } else { 
    edm::LogVerbatim(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]"
      << " Unexpected presentation: \"" 
      << SiStripEnumsAndStrings::presentation( pres )
      << "\" Unable to build summary plot!"
      << " Returning NULL pointer!";
  }
  
  return summary;
  
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::format( const sistrip::RunType& run_type, 
			       const sistrip::Monitorable& mon, 
			       const sistrip::Presentation& pres,
			       const sistrip::View& view, 
			       const std::string& directory,
			       const sistrip::Granularity& gran,
			       TH1& summary_histo ) {
  
  // Set name, title and entries
  //std::stringstream ss;
  //std::string name = SummaryGenerator::name( run_type, mon, pres, view, directory );
  //summary_histo.SetName( name.c_str() );
  //summary_histo.SetTitle( name.c_str() );
  if ( entries_ >= 0. ) { summary_histo.SetEntries( entries_ ); }
  
  // X axis
  summary_histo.GetXaxis()->SetLabelSize(0.03);
  summary_histo.GetXaxis()->SetTitleSize(0.03);
  summary_histo.GetXaxis()->SetTitleOffset(3.5); 
  //gPad->SetBottomMargin(0.2);
  
  // Y axis
  summary_histo.GetYaxis()->SetLabelSize(0.03);
  summary_histo.GetYaxis()->SetTitleSize(0.03);
  summary_histo.GetYaxis()->SetTitleOffset(1.5);
  //gPad->SetLeftMargin(0.2);
  
  // Axis label
  if ( pres == sistrip::HISTO_1D ) {
    std::string xtitle = label_ + " (for " + directory + ")";
    summary_histo.GetXaxis()->SetTitle( xtitle.c_str() );
    summary_histo.GetYaxis()->SetTitle( "Frequency" );
    summary_histo.GetXaxis()->SetTitleOffset(1.5); //@@ override value set above
  } else {
    std::string xtitle = SiStripEnumsAndStrings::granularity( gran ) + " within " + directory;
    summary_histo.GetXaxis()->SetTitle( xtitle.c_str() );
    summary_histo.GetYaxis()->SetTitle( label_.c_str() );
    //summary_histo.GetXaxis()->SetTitleOffset(1.5); //@@ override value set above (3.5?)
  }    
  
  // Formatting for 2D plots
  if ( pres == sistrip::HISTO_2D_SCATTER ) { 
    // Markers (open circles)
    summary_histo.SetMarkerStyle(2); 
    summary_histo.SetMarkerSize(0.6);
  }

  // Semi-generic formatting
  if ( pres == sistrip::HISTO_2D_SUM || 
       pres == sistrip::HISTO_2D_SCATTER ||
       pres == sistrip::PROFILE_1D ) {
    /*
    //put solid and dotted lines on summary to separate top- and
    //2nd-from-top- level bin groups.

    uint16_t topLevel = 0, topLevelOld = 0, secondLevel = 0, secondLevelOld = 0;
    std::string::size_type pos = 0;
    for ( HistoData::iterator ibin = map_.begin(); ibin != map_.end(); ibin++) {

      //draw line if top and second level numbers change.
      pos = ibin->first.find(sistrip::dot_,0);
      if (pos != std::string::npos) {
	if ((topLevel=atoi(std::string(ibin->first,0,pos).c_str())) != topLevelOld) {
	  topLevel = topLevelOld;
	  //
	}
	else if (ibin->first.find(sistrip::dot_,pos+1) != std::string::npos) {
	    if ((secondLevelOld=atoi(std::string(ibin->first,pos+1,(ibin->first.find(sistrip::dot_,pos+1)- (pos+1))).c_str())) != secondLevel) {
	      secondLevel = secondLevelOld;
	      //
	    }}}
    }
    */
  }
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::clearMap() {
  HistoData::iterator iter = map_.begin(); 
  for ( ; iter != map_.end(); iter++ ) { iter->second.clear(); }
  map_.clear();
  entries_ = -1.;
  max_ = -1.*sistrip::invalid_;
  min_ =  1.*sistrip::invalid_;
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::printMap() {

  std::stringstream ss;
  ss << "[SummaryGenerator::" << __func__ << "]"
     << " Printing contents of map: " << std::endl;
  
  HistoData::iterator iter = map_.begin(); 
  for ( ; iter != map_.end(); iter++ ) { 
    ss << " bin/entries: " << iter->first << "/" << iter->second.size() << " ";
    if ( !iter->second.empty() ) {
      ss << " value/error: ";
      std::vector<Data>::const_iterator jter = iter->second.begin();
      for ( ; jter != iter->second.end(); jter++ ) { 
	ss << jter->first << "/" << jter->second << " ";
      }
    }
    ss << std::endl;
  }
  
  ss << " Max value: " << max_ << std::endl
     << " Min value: " << min_ << std::endl;

  LogTrace(mlSummaryPlots_) << ss.str();
  
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::fillMap( const std::string& top_level_dir,
				const sistrip::Granularity& gran,
				const uint32_t& device_key, 
				const float& value,
				const float& error ) {
  
  // Check if value is valid
  if ( value > 1. * sistrip::valid_ ) { return; }
  
  // Calculate maximum and minimum values in std::map
  if ( value > max_ ) { max_ = value; }
  if ( value < min_ ) { min_ = value; }
  
  // Check if error is valid
  if ( error < 1. * sistrip::valid_ ) { 
    fill( top_level_dir, gran, device_key, value, error );
  } else { 
    fill( top_level_dir, gran, device_key, value, 0. ); 
  } 
  
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::fill( const std::string& top_level_dir,
			     const sistrip::Granularity& gran,
			     const uint32_t& device_key, 
			     const float& value,
			     const float& error ) {
  
  LogTrace(mlSummaryPlots_)
    << "[SummaryGenerator::" << __func__ << "]"
    << " Derived implementation does not exist!...";
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::histo1D( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]" 
      << " No contents in std::map to histogram!";
    return; 
  }
  
  // Retrieve histogram  
  TH1F* histo = dynamic_cast<TH1F*>(&his);
  if ( !histo ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]"
      << " NULL pointer to TH1F histogram!";
    return;
  }
  
  // Calculate bin range
  int32_t high  = static_cast<int32_t>( fabs(max_) > 20. ? max_ + 0.05 * fabs(max_) : max_ + 1. );
  int32_t low   = static_cast<int32_t>( fabs(min_) > 20. ? min_ - 0.05 * fabs(min_) : min_ - 1. );
  int32_t range = high - low;

  // increase number of bins for floats
//   if ( max_ - static_cast<int32_t>(max_) > 1.e-6 && 
//        min_ - static_cast<int32_t>(min_) > 1.e-6 ) {
//     range = 100 * range; 
//   }
  
  // Set histogram binning
  histo->SetBins( range, static_cast<float>(low), static_cast<float>(high) );
  
  // Iterate through std::map, set bin labels and fill histogram
  entries_ = 0.;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      // bin (value) and weight (error)
      histo->Fill( ii->first ); //, ii->second ); 
      entries_++;
    }
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::histo2DSum( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]" 
      << " No contents in std::map to histogram!";
    return; 
  }
  
  // Retrieve histogram  
  TH1F* histo = dynamic_cast<TH1F*>(&his);
  if ( !histo ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]"
      << " NULL pointer to TH1F histogram!";
    return;
  }
  
  // Iterate through map, set bin labels and fill histogram
  entries_ = 0.;
  uint16_t bin = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    bin++;
    histo->GetXaxis()->SetBinLabel( static_cast<Int_t>(bin), ibin->first.c_str() );
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      // x (bin), y (value) and weight (error)
      histo->Fill( static_cast<Double_t>(bin-0.5), 
		   static_cast<Double_t>(ii->first) ); //, ii->second ); 
      entries_ += 1. * ii->first;
    }
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::histo2DScatter( TH1& his ) {

  // Check number of entries in map
  if ( map_.empty() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]" 
      << " No contents in std::map to histogram!";
    return; 
  }
  
  // Retrieve histogram  
  TH2F* histo = dynamic_cast<TH2F*>(&his);
  if ( !histo ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]"
      << " NULL pointer to TH2F histogram!";
    return;
  }
  
  // Iterate through std::map, set bin labels and fill histogram
  entries_ = 0.;
  uint16_t bin = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    bin++;
    histo->GetXaxis()->SetBinLabel( static_cast<Int_t>(bin), ibin->first.c_str() );
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      // x (bin), y (value) and weight (error)
      histo->Fill( static_cast<Double_t>(bin-0.5), 
		   static_cast<Double_t>(ii->first) ); // , ii->second ); 
      entries_++;
    }
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::profile1D( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryGenerator::" << __func__ << "]" 
      << " No contents in std::map to histogram!";
    return; 
  }
  
  // Retrieve histogram  
  TProfile* histo = dynamic_cast<TProfile*>(&his);
  if ( !histo ) { 
    edm::LogWarning(mlSummaryPlots_) 
      << "[SummaryGenerator::" << __func__ << "]"
      << " NULL pointer to TProfile histogram!";
    return;
  }
  
  // Iterate through std::map, set bin labels and fill histogram
  entries_ = 0.;
  uint16_t bin = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    bin++;
    histo->GetXaxis()->SetBinLabel( static_cast<Int_t>(bin), ibin->first.c_str() );
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      // x (bin), y (value) and weight (error)
      histo->Fill( static_cast<Double_t>(bin-.5), 
		   static_cast<Double_t>(ii->first) ); //, ii->second ); 
      entries_++;
    }
  }
  
}



