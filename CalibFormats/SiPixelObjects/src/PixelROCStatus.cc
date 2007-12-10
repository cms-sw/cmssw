//
// This class keeps the possible non-standard
// status a ROC can have.
//
//
//

#include <stdint.h>
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

using namespace pos;

PixelROCStatus::PixelROCStatus():
  bits_(0)
{}


PixelROCStatus::PixelROCStatus(const std::set<status>& stat){
  std::set<status>::const_iterator i=stat.begin();
  
  for(;i!=stat.end();++i){
    set(*i);
  }

}

PixelROCStatus::~PixelROCStatus(){}
    
void PixelROCStatus::set(status stat){
  bits_=bits_&(1>>stat);
}

void PixelROCStatus::clear(status stat){
  bits_=bits_&(1>>stat);
}



void PixelROCStatus::set(status stat, bool mode){
  if (mode) {
    set(stat);
  }
  else{
    clear(stat);
  }
}


bool PixelROCStatus::get(status stat){
  return bits_&(1>>stat);
}

std::string PixelROCStatus::statusName(status stat){
  if (stat==off) return "off";
  if (stat==noHits) return "noHits";
}
