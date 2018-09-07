#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;
L1MuKBMTCombinedStub::L1MuKBMTCombinedStub() :
  whNum_(0),scNum_(0),stNum_(0),phi_(0), phiB_(0),tag_(false), quality_(-1), bxNum_(17),
  eta1_(0),eta2_(0),qeta1_(-1),qeta2_(-1) {}


L1MuKBMTCombinedStub::L1MuKBMTCombinedStub(int wheel,int sector,int station,int phi,int phiB,bool tag,int bx,int quality,int eta1,int eta2, int qeta1,int qeta2):
  whNum_(wheel),
  scNum_(sector),
  stNum_(station),
  phi_(phi),
  phiB_(phiB),
  tag_(tag),
  quality_(quality),
  bxNum_(bx),
  eta1_(eta1),
  eta2_(eta2),
  qeta1_(qeta1),
  qeta2_(qeta2)  
{

}

L1MuKBMTCombinedStub::~L1MuKBMTCombinedStub() {}


bool L1MuKBMTCombinedStub::operator==(const L1MuKBMTCombinedStub& id) const {

  if ( whNum_      != id.whNum_ )     return false;
  if ( scNum_      != id.scNum_ )     return false;
  if ( stNum_      != id.stNum_ )     return false;
  if ( tag_        != id.tag_ )       return false;
  if ( phi_        != id.phi_ )       return false;
  if ( phiB_       != id.phiB_)       return false;
  if ( quality_    != id.quality_ )   return false;
  if ( bxNum_      != id.bxNum_ )     return false;
  if ( eta1_       != id.eta1_ )      return false;
  if ( eta2_       != id.eta2_ )      return false;
  if ( qeta1_      != id.qeta1_ )     return false;
  if ( qeta2_      != id.qeta2_ )     return false;
  return true;

}

//
// output stream operator for phi track segments
//
ostream& operator<<(ostream& s, const L1MuKBMTCombinedStub& id) {

  s.setf(ios::right,ios::adjustfield);
  s << "BX: "      << setw(5) << id.bxNum_  << " "
  << "wheel: "     << setw(5) << id.whNum_  << " "
  << "sector: "    << setw(5) << id.scNum_  << " "
  << "station: "   << setw(5) << id.stNum_  << " "
  << "tag: "       << setw(5) << id.tag_  << " "
  << "phi: "       << setw(5) << id.phi_  << " "
  << "phiB: "      << setw(4) << id.phiB_ << " "
  << "quality: "   << setw(4) << id.quality_ << " "
  << "eta1:"       << setw(4) <<id.eta1_ << " "
  << "eta2:"       << setw(4) <<id.eta2_ << " "
  << "qeta1:"       << setw(4) <<id.qeta1_ << " "
  << "qeta2:"       << setw(4) <<id.qeta2_;

  return s;

}
