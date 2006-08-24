#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
#include "DQM/SiStripCommon/interface/SummaryGeneratorControlView.h"
//#include "DQM/SiStripCommon/interface/SummaryGeneratorReadoutView.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
// 
auto_ptr<SummaryGenerator> SummaryGenerator::instance( sistrip::View view ) {
  auto_ptr<SummaryGenerator> generator;
  if ( view == sistrip::CONTROL ) {
    generator = auto_ptr<SummaryGenerator>( new SummaryGeneratorControlView() );
  } else if ( view == sistrip::READOUT ) {
    //generator = auto_ptr<SummaryGenerator>( new SummaryGeneratorReadoutView() );
  } else { 
    generator = auto_ptr<SummaryGenerator>( NULL );
  }  
  return generator;
}

// -----------------------------------------------------------------------------
// 
void SummaryGenerator::format( const sistrip::SummaryHisto& histo, 
			       const sistrip::SummaryType& type,
			       const sistrip::View& view, 
			       const std::string& directory,
			       TH1& summary_histo ) {
  
  // Set name and title
  stringstream ss;
  string name = SummaryGenerator::name( histo, type, view, directory );
  summary_histo.SetName( name.c_str() );
  summary_histo.SetTitle( name.c_str() );
  
  //@@ other stuff here?...
}

// -----------------------------------------------------------------------------
// 
string SummaryGenerator::name( const sistrip::SummaryHisto& histo, 
			       const sistrip::SummaryType& type,
			       const sistrip::View& view, 
			       const std::string& directory ) {
  stringstream ss;
  ss << sistrip::summaryHisto_ << sistrip::sep_;
  ss << SiStripHistoNamingScheme::summaryHisto( histo ) << sistrip::sep_;
  ss << SiStripHistoNamingScheme::summaryType( type ) << sistrip::sep_; 
  ss << SiStripHistoNamingScheme::view( view );
  return ss.str();
}

// -----------------------------------------------------------------------------
// 
pair<float,float> SummaryGenerator::range() {

  if ( map_.empty() ) { return pair<float,float>(0.,0.); }
  
  pair<float,float> range = pair<float,float>(0.,0.);
  range.first = -1.e6;
  range.second = 1.e6;
  
  map< string, pair<float,float> >::const_iterator iter = map_.begin();
  for ( ; iter != map_.end(); iter++ ) {
    if ( iter->second.first > range.first ) { range.first = iter->second.first; }
    if ( iter->second.first < range.second ) { range.second = iter->second.first; }
  }
  return range;

}

