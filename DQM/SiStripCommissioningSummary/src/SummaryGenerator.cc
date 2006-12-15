#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorReadoutView.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
SummaryGenerator::SummaryGenerator() : 
  map_(),
  entries_(0),
  max_(-1.*sistrip::invalid_),
  min_(1.*sistrip::invalid_),
  label_("")
{
  // TH1::SetDefaultSumw2(true); // use square of weights to calc error   
}

// -----------------------------------------------------------------------------
// 
SummaryGenerator* SummaryGenerator::instance( const sistrip::View& view ) {
  SummaryGenerator* generator = 0;
  if ( view == sistrip::CONTROL ) {
    generator = new SummaryGeneratorControlView();
  } else if ( view == sistrip::READOUT ) {
    generator = new SummaryGeneratorReadoutView();
  } else {
    generator = 0;
    cerr << "[SummaryGenerator::" << __func__ << "]"
	 << " Unexpected view: " 
	 << SiStripHistoNamingScheme::view( view ) 
	 << endl;
  }  
  return generator;
}

// -----------------------------------------------------------------------------
// 
string SummaryGenerator::name( const sistrip::Task& task, 
			       const sistrip::Monitorable& mon, 
			       const sistrip::Presentation& pres,
			       const sistrip::View& view, 
			       const std::string& directory ) {
  stringstream ss;
  ss << SiStripHistoNamingScheme::presentation( pres ) << sistrip::sep_; 
  if ( task != sistrip::UNKNOWN_TASK && 
       task != sistrip::UNDEFINED_TASK ) { 
    ss << SiStripHistoNamingScheme::task( task ) << sistrip::sep_;
  }
  ss << SiStripHistoNamingScheme::view( view ) << sistrip::sep_;
  ss << SiStripHistoNamingScheme::monitorable( mon );
  
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
  if ( pres == sistrip::SUMMARY_HISTO ) { 
    summary = new TH1F( "", "", 1024, 0., static_cast<float>(1024) ); 
  } else if ( pres == sistrip::SUMMARY_1D ) { 
    summary = new TH1F( "", "", xbins, 0., static_cast<float>(xbins) ); 
  } else if ( pres == sistrip::SUMMARY_2D ) { 
    summary = new TH2F( "", "", 100*xbins, 0., static_cast<float>(100*xbins), 1025, 0., 1025. ); 
  } else if ( pres == sistrip::SUMMARY_PROF ) { 
    summary = new TProfile( "", "", xbins, 0., static_cast<float>(xbins), 0., 1025. ); 
  } else { summary = 0; }
  return summary;
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::format( const sistrip::Task& task, 
			       const sistrip::Monitorable& mon, 
			       const sistrip::Presentation& pres,
			       const sistrip::View& view, 
			       const std::string& directory,
			       const sistrip::Granularity& gran,
			       TH1& summary_histo ) {
  
  // Set name and title
  stringstream ss;
  string name = SummaryGenerator::name( task, mon, pres, view, directory );
  summary_histo.SetName( name.c_str() );
  summary_histo.SetTitle( name.c_str() );

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
  if ( pres == sistrip::SUMMARY_HISTO ) {
    string xtitle = label_ + " (for " + directory + ")";
    summary_histo.GetXaxis()->SetTitle( xtitle.c_str() );
    summary_histo.GetYaxis()->SetTitle( "Frequency" );
    summary_histo.GetXaxis()->SetTitleOffset(1.5); //@@ override value set above
  } else {
    string xtitle = SiStripHistoNamingScheme::granularity( gran ) + " within " + directory;
    summary_histo.GetXaxis()->SetTitle( xtitle.c_str() );
    summary_histo.GetYaxis()->SetTitle( label_.c_str() );
  }    
  
  // Formatting for 2D plots
  if ( pres == sistrip::SUMMARY_2D ) { 
    // Markers (open circles)
    summary_histo.SetMarkerStyle(2); 
    summary_histo.SetMarkerSize(0.6);
  }

  // Semi-generic formatting
  if ((pres == sistrip::SUMMARY_1D) || 
      (pres == sistrip::SUMMARY_2D) ||
      (pres == sistrip::SUMMARY_PROF)) {
    /*
    //put solid and dotted lines on summary to separate top- and
    //2nd-from-top- level bin groups.

    uint16_t topLevel = 0, topLevelOld = 0, secondLevel = 0, secondLevelOld = 0;
    string::size_type pos = 0;
    for ( HistoData::iterator ibin = map_.begin(); ibin != map_.end(); ibin++) {

      //draw line if top and second level numbers change.
      pos = ibin->first.find(sistrip::dot_,0);
      if (pos != string::npos) {
	if ((topLevel=atoi(string(ibin->first,0,pos).c_str())) != topLevelOld) {
	  topLevel = topLevelOld;
	  //
	}
	else if (ibin->first.find(sistrip::dot_,pos+1) != string::npos) {
	    if ((secondLevelOld=atoi(string(ibin->first,pos+1,(ibin->first.find(sistrip::dot_,pos+1)- (pos+1))).c_str())) != secondLevel) {
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
  entries_ = 0;
  max_ = -1.*sistrip::invalid_;
  min_ =  1.*sistrip::invalid_;
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::fillMap( const string& top_level_dir,
				const sistrip::Granularity& gran,
				const uint32_t& device_key, 
				const float& value,
				const float& error ) {
  
  // Calculate maximum and minimum values in map
  if ( value > max_ ) { max_ = value; }
  if ( value < min_ ) { min_ = value; }
  
  // Fill map
  fill( top_level_dir, gran, device_key, value, error );
  
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::fill( const string& top_level_dir,
			     const sistrip::Granularity& gran,
			     const uint32_t& device_key, 
			     const float& value,
			     const float& error ) {
  
  cout << "[SummaryGenerator::" << __func__ << "]"
       << " Derived implementation does not exist!..."
       << endl;    
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::summaryHisto( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]" 
	 << " No contents in map to histogram!" << endl;
    return; 
  }
  
  // Retrieve histogram  
  TH1F* histo = dynamic_cast<TH1F*>(&his);
  if ( !histo ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]"
	 << " NULL pointer to TH1F histogram!" << endl;
    return;
  }
  
  // Calculate bin range
  float high = ceil(max_);
  float low  = ceil( fabs(min_) );
  if ( min_ < 0. ) { low *= -1.; }
  float diff = high - low;
  if ( diff < 20 ) {
    high += (20-diff) / 2.;
    low  -= (20-diff) / 2.;
  } else {
    high *= 1.2;
    low  *= 1.2;
  }
  if ( low < 20. && low > 0. ) { low = 0.; }
  high = rint(high);
  low  = rint(low);
  // Calculate binning range 
  int32_t range = static_cast<int32_t>( high-low );
  // Use finer binning if noise or timing data 
  if ( 0 ) { range *= 10; }
  // Check range 
  if ( range < 20 ) { range = 20; } 
  if ( range > 1024 ) { range = 1024; }
  // Set histogram binning
  histo->SetBins( range, low, high ); 
  
  //   cout << " binning: " 
  //        << max_ << " "
  //        << min_ << " "
  //        << high << " "
  //        << low << " "
  //        << range << " "
  //        << endl;
  
  // Iterate through map, set bin labels and fill histogram
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      histo->Fill( ii->first ); //, ii->second ); // bin (value) and weight (error)
    }
  }
  
  cout << "[SummaryGenerator::" << __func__ << "]"
       << " Added " << histo->GetEntries()
       << " entries to 1D histogram, which has " 
       << histo->GetNbinsX()
       << " bins" << endl;
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::summary1D( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]" 
	 << " No contents in map to histogram!" << endl;
    return; 
  }
  
  // Retrieve histogram  
  TH1F* histo = dynamic_cast<TH1F*>(&his);
  if ( !histo ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]"
	 << " NULL pointer to TH1F histogram!" << endl;
    return;
  }

  // Set histogram number of bins and min/max
  histo->SetBins( map_.size(), 0., (Double_t)map_.size() );
  
  // Iterate through map, set bin labels and fill histogram
  uint16_t bin = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    bin++;
    histo->GetXaxis()->SetBinLabel( static_cast<Int_t>(bin), ibin->first.c_str() );
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      histo->Fill( (Double_t)(bin-.5), (Double_t)ii->first ); //, ii->second ); // x (bin), y (value) and weight (error)
      //histo->Fill( ibin->first.c_str(), ii->first ); // x (bin) and weight (value)
//       cout << "temp " << bin << " " 
// 	   << ii->first << " "
// 	   << ii->second << endl;
    }
//     cout << "[SummaryGenerator::" << __func__ << "]"
// 	 << " Added " << ibin->second.size() 
// 	 << " contents to bin " << bin
// 	 << " with bin label '" << ibin->first.c_str()
// 	 << endl;
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::summary2D( TH1& his ) {

  // Check number of entries in map
  if ( map_.empty() ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]" 
	 << " No contents in map to histogram!" << endl;
    return; 
  }
  
  // Retrieve histogram  
  TH2F* histo = dynamic_cast<TH2F*>(&his);
  if ( !histo ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]"
	 << " NULL pointer to TH2F histogram!" << endl;
    return;
  }
  
  // Set histogram number of bins and min/max
  histo->GetXaxis()->Set( 100*map_.size(), 0., 100.*static_cast<Double_t>(map_.size()) );
  histo->GetYaxis()->Set( 1025, 0., 1025. );

  //histo->Dump();

  // Iterate through map, set bin labels and fill histogram
  uint16_t bins = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    uint16_t bin = 100*bins+50;
    histo->GetXaxis()->SetBinLabel( (Int_t)(bin), ibin->first.c_str() );
    //if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
      histo->Fill( (Double_t)(bin-.5), (Double_t)ii->first ); //, ii->second ); // x (bin), y (value) and weight (error)
    }
    bins++;
    /* cout << "[SummaryGenerator::" << __func__ << "]"
	 << " Added " << ibin->second.size() 
	 << " contents to bin " << bin
 	 << " with bin label '" << ibin->first.c_str()
 	 << endl;
    */
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGenerator::summaryProf( TH1& his ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]" 
	 << " No contents in map to histogram!" << endl;
    return; 
  }
  
  // Retrieve histogram  
  TProfile* histo = dynamic_cast<TProfile*>(&his);
  if ( !histo ) { 
    cerr << "[SummaryGenerator::" << __func__ << "]"
	 << " NULL pointer to TProfile histogram!" << endl;
    return;
  }
  
  // Set histogram number of bins and min/max
  //histo->SetBins( map_.size(), 0., static_cast<Double_t>(map_.size()) );
  histo->GetXaxis()->Set( map_.size(), 0., static_cast<Double_t>(map_.size()) );
  histo->GetYaxis()->Set( 1025, 0., 1025. );
  
  // Iterate through map, set bin labels and fill histogram
  uint16_t bin = 0;
  HistoData::const_iterator ibin = map_.begin();
  for ( ; ibin != map_.end(); ibin++ ) {
    bin++;
    histo->GetXaxis()->SetBinLabel( static_cast<Int_t>(bin), ibin->first.c_str() );
    if ( ibin->second.empty() ) { continue; }
    BinData::const_iterator ii = ibin->second.begin();
    for ( ; ii != ibin->second.end(); ii++ ) { 
//       cout << " bin: " << bin
// 	   << " value: " << ii->first
// 	   << " error: " << ii->second
// 	   << endl;
      //float wei = ii->second = 0 ? 
      //histo->AddBinContent( bin+1, ii->first ); //, ii->second ); // x (bin), y (value) and weight (error)
      histo->Fill( (Double_t)(bin-.5), (Double_t)ii->first ); //, ii->second ); // x (bin), y (value) and weight (error)
      //histo->Fill( ibin->first.c_str(), ii->first ); //, ii->second ); // x (bin), y (value) and weight (error)
    }
//     cout << "[SummaryGenerator::" << __func__ << "]"
// 	 << " Added " << ibin->second.size() 
// 	 << " contents to bin " << bin
// 	 << " with bin label '" << ibin->first.c_str()
// 	 << endl;
  }
  
}



