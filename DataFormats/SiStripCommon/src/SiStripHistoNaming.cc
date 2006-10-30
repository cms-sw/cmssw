#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::histoTitle( const HistoTitle& title ) {
  
  stringstream histo_title;

  // Append Task, KeyType and KeyValue
  histo_title << SiStripHistoNamingScheme::task( title.task_ )
	      << sistrip::sep_ 
	      << SiStripHistoNamingScheme::keyType( title.keyType_ )
	      << sistrip::hex_ << hex << setfill('0') << setw(8) << title.keyValue_ << dec
	      << sistrip::sep_;
  
  // Append Granularity (after checking if it is applicable)
  if ( title.granularity_ == sistrip::MODULE ||
       title.granularity_ == sistrip::LLD_CHAN ||
       title.granularity_ == sistrip::APV ) {
    histo_title << SiStripHistoNamingScheme::granularity( title.granularity_ );
    histo_title << title.channel_;
  } else {
    histo_title << SiStripHistoNamingScheme::granularity( sistrip::UNDEFINED_GRAN );
  }

  // Append extra info
  if ( title.extraInfo_ != "" ) { 
    histo_title << sistrip::sep_ << title.extraInfo_; 
  }
  
  return histo_title.str();
  
}

// -----------------------------------------------------------------------------
// 
HistoTitle SiStripHistoNamingScheme::histoTitle( const string& histo_title ) {
  
  HistoTitle title;
  uint32_t position = 0;
  
  // Extract Task
  title.task_ = SiStripHistoNamingScheme::task( histo_title.substr(position) );
  string task = SiStripHistoNamingScheme::task( title.task_ );
  position += histo_title.substr(position).find( task ) + task.size();
  
  // Extract KeyType
  title.keyType_ = SiStripHistoNamingScheme::keyType( histo_title.substr(position) );
  string key_type = SiStripHistoNamingScheme::keyType( title.keyType_ );
  position += histo_title.substr(position).find( key_type ) + key_type.size();
  
  // Extract KeyValue
  uint16_t key_size = 8;
  position += sistrip::hex_.size();
  stringstream key; key << histo_title.substr( position, key_size );
  key >> hex >> title.keyValue_;
  position += key_size;
  
  // Extract Granularity
  title.granularity_ = SiStripHistoNamingScheme::granularity( histo_title.substr(position) );
  string gran = SiStripHistoNamingScheme::granularity( title.granularity_ );
  position += histo_title.substr(position).find( gran ) + gran.size();

  // Extract Channel 
  uint32_t chan_size = histo_title.find( sistrip::sep_, position ) - position;
  stringstream chan; 
  chan << histo_title.substr( position, chan_size );
  chan >> dec >> title.channel_;
  position += chan_size;
  
  // Extract ExtraInfo
  uint32_t pos = histo_title.find( sistrip::sep_, position );
  if ( pos != string::npos ) { 
    title.extraInfo_ = histo_title.substr( pos+1, string::npos ); 
  }
  
  // Return HistoTitle object
  return title;
  
}
