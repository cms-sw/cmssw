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
  
  bool put( std::string runSummary ){ runSummary_ = runSummary; return true; }
  std::string getRunSummary() const { return runSummary_; } 

 private:

  std::string runSummary_; 

};

#endif
