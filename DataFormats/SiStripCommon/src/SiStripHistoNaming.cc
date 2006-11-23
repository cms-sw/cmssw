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
  string::size_type length = histo_title.length();
  string::size_type position = 0;
  string::size_type pos = 0;
  string::size_type siz = 0;
  
  // Extract Task
  siz = histo_title.find(sistrip::sep_,position) - position;
  title.task_ = SiStripHistoNamingScheme::task( histo_title.substr(position,siz) );
  string task = SiStripHistoNamingScheme::task( title.task_ );
  position += histo_title.substr(position).find( task ) + task.size() + sistrip::sep_.size();
  if ( position >= length ) { return title; }
  
  // Extract KeyType
  siz = histo_title.find(sistrip::sep_,position) - position;
  title.keyType_ = SiStripHistoNamingScheme::keyType( histo_title.substr(position,siz) );
  string key_type = SiStripHistoNamingScheme::keyType( title.keyType_ );
  position += histo_title.substr(position).find( key_type ) + key_type.size() + sistrip::hex_.size();
  if ( position >= length ) { return title; }
  
  // Extract KeyValue
  siz = 8;
  stringstream key; 
  key << histo_title.substr(position,siz);
  key >> hex >> title.keyValue_;
  position += siz + sistrip::sep_.size();
  if ( position >= length ) { return title; }
  
  // Extract Granularity
  pos = histo_title.find(sistrip::sep_,position);
  if ( pos == string::npos || pos < position ) { siz = string::npos - position; }
  else { siz = pos - position; }
  title.granularity_ = SiStripHistoNamingScheme::granularity( histo_title.substr(position,siz) );
  string gran = SiStripHistoNamingScheme::granularity( title.granularity_ );
  position += histo_title.substr(position).find( gran ) + gran.size();
  if ( position >= length ) { return title; }

  // Extract Channel 
  pos = histo_title.find(sistrip::sep_,position);
  if ( pos == string::npos || pos < position ) { siz = string::npos - position; }
  else { siz = pos - position; }
  stringstream chan; 
  chan << histo_title.substr(position,siz);
  chan >> dec >> title.channel_;
  position += siz + sistrip::sep_.size();
  if ( position >= length ) { return title; }
  
  // Extract ExtraInfo
  title.extraInfo_ = histo_title.substr( position, string::npos - position ); 
  
  // Return HistoTitle object
  return title;
  
}
