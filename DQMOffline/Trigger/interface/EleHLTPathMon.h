#ifndef DQMOFFLINE_TRIGGER_ELEHLTPATHMON
#define DQMOFFLINE_TRIGGER_ELEHLTPATHMON

#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTOffData.h"

#include <vector>
#include <string>

//class: EleHLTPathMon
//
//author: Sam Harper (June 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: this object will manage all histograms associated with a particular HLT path
//     ie histograms for computing efficiency of each filter step, id efficiency of gsf electrons passing trigger etc   
//
//implimentation: 
//       has a vector of all filter path names, generates efficiency hists for each step
//       

class EleHLTPathMon {
 private:
  std::string pathName_;
  std::vector<EleHLTFilterMon*> filters_; //we own these
  
  

 public:
  explicit EleHLTPathMon(std::string pathName=""):pathName_(pathName){}
  ~EleHLTPathMon();
 
  void addFilter(const std::string& filterName);
  void addFilters(const std::vector<std::string>& names);
  //void setStdFilters();  //egamma electron triggers all have the same filter names, time saving function
  
  //does exactly what it says on the tin (fills all the filterMons)
  void fill(const EgHLTOffData& evtData,float weight);

  std::vector<std::string> getFilterNames()const; //expensive function, gets names of all the functions, if slow, will cache the results

  //sort by path name
  bool operator<(const EleHLTPathMon& rhs)const{return pathName_<rhs.pathName_;}
};



#endif
