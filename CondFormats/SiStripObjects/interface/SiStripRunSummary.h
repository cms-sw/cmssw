#ifndef SiStripRunSummary_h
#define SiStripRunSummary_h

#include<vector>
#include<string>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripRunSummary {

 public:

  SiStripRunSummary(){};
  ~SiStripRunSummary(){};
  
  bool put(std::string runSummary){runSummary_=runSummary;}
  std::string getRunSummary(){return runSummary_;} 

 private:
  std::string runSummary_; 
};

#endif
