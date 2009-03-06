//
// This class keeps the possible non-standard
// status a ROC can have.
//
//
//

#include <stdint.h>
#include <set>
#include <iostream>
#include <cassert>
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

using namespace std;
using namespace pos;

PixelROCStatus::PixelROCStatus():
  bits_(0)
{}


PixelROCStatus::PixelROCStatus(const std::set<ROCstatus>& stat){

  std::set<ROCstatus>::const_iterator i=stat.begin();
  
  for(;i!=stat.end();++i){
    set(*i);
  }

}

PixelROCStatus::~PixelROCStatus(){}
    
void PixelROCStatus::set(ROCstatus stat){
  bits_=bits_^(1>>stat);
}

void PixelROCStatus::clear(ROCstatus stat){
  bits_=bits_&(0>>stat);
}



void PixelROCStatus::set(ROCstatus stat, bool mode){
  if (mode) {
    set(stat);
  }
  else{
    clear(stat);
  }
}


bool PixelROCStatus::get(ROCstatus stat) const{
  return bits_&(1>>stat);
}

string PixelROCStatus::statusName(ROCstatus stat) const{
  if (stat==off) return "off";
  if (stat==noHits) return "noHits";
  assert(0);
  return "";
}

// modified by MR on 11-01-2008 15:06:28
string PixelROCStatus::statusName() const {
  string result = "" ;
  for (ROCstatus istat=off; istat!=nStatus; istat=ROCstatus(istat+1))
    {
      if (get(istat))
	{
	  result += " " + statusName(istat) ;
	}
    }
  return result ;
}

void PixelROCStatus::set(const string& statName){

  for (ROCstatus istat=off; istat!=nStatus; istat=ROCstatus(istat+1)){
    if (statName==statusName(istat)){
      set(istat);
      return;
    }
  }
  cout << "In PixelROCStatus::set the statusName:"
       << statName <<" is not known"<<endl;
  ::abort();
}
