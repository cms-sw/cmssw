#include "Geometry/CaloTopology/interface/HcalTopologyRestrictionParser.h"
#include<boost/tokenizer.hpp>
#include<sstream>
#include<iostream>
HcalTopologyRestrictionParser::HcalTopologyRestrictionParser(HcalTopology& target) : target_(target) {
}

static HcalSubdetector determineSubdet(const std::string& item) {
  if (item=="HB") return HcalBarrel;
  if (item=="HE") return HcalEndcap;
  if (item=="HF") return HcalForward;
  if (item=="HO") return HcalOuter;
  return (HcalSubdetector)0;
}

std::string HcalTopologyRestrictionParser::parse(const std::string& line) {

  std::ostringstream errors;
  boost::char_separator<char> sep(" \t",";");
  typedef boost::tokenizer<boost::char_separator<char> > myTokType;

  std::string totaline(line); totaline+=';'; // terminate
  myTokType tok(totaline, sep);
  int ieta1=0, ieta2=0, iphi1=-1, iphi2=-1, depth1=1, depth2=4;
  HcalSubdetector subdet=(HcalSubdetector)0;

  int phase=0; 
  for (myTokType::iterator beg=tok.begin(); beg!=tok.end() && phase>=0; ++beg){
    std::cout << phase << " : <" << *beg << ">\n";
    if (*beg==";") {
      if (phase==0) continue; // empty
      if (phase!=1 && phase!=5 && phase!=7) { 
	errors << "Expect 1, 5, or 7 arguments, got " << phase;
	phase=-1;
      } else {
	if (phase==1) { // reject whole subdetector...
	  target_.excludeSubdetector(subdet);
	} else {
	  target_.exclude(subdet,ieta1,ieta2,iphi1,iphi2,depth1,depth2);
	}
	phase=0;
      }
    } else {
      switch (phase) {
      case (0) : subdet=determineSubdet(*beg); break;
      case (1) : ieta1=atoi(beg->c_str()); break;
      case (2) : ieta2=atoi(beg->c_str()); break;
      case (3) : iphi1=atoi(beg->c_str()); break;
      case (4) : iphi2=atoi(beg->c_str()); depth1=1; depth2=4; break; // also set defaults...
      case (5) : depth1=atoi(beg->c_str()); break;
      case (6) : depth2=atoi(beg->c_str()); break;
      }
      phase++;
    }
  }
  
  return errors.str();

}
