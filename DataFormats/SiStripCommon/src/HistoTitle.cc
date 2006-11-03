#include "DataFormats/SiStripCommon/interface/HistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
void HistoTitle::print( stringstream& ss ) const {
  ss << "[HistoTitle::" << __func__ << "]" << endl
     << "  CommissioningTask:  " 
     << SiStripHistoNamingScheme::task(task_)
     << endl
     << "  KeyType/Value(hex): " 
     << SiStripHistoNamingScheme::keyType(keyType_) 
     << " / "
     << hex << setfill('0') << setw(8) << keyValue_ << dec
     << endl
     << "  Gran/Channel:       "
     << SiStripHistoNamingScheme::granularity(granularity_) 
     << " / "
     << channel_
     << endl
     << "  ExtraInfo:          ";
  if ( extraInfo_ != "" ) { ss << extraInfo_; }
  else { ss << "(none)"; }
}

// -----------------------------------------------------------------------------
//
ostream& operator<< ( ostream& os, const HistoTitle& title ) {
  stringstream ss;
  title.print(ss);
  os << ss.str();
  return os;
}

