#ifndef UEAnalysisWeight_h
#define UEAnalysisWeight_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class UEAnalysisWeight {
 public :

  UEAnalysisWeight();
  ~UEAnalysisWeight(){}
  std::vector<float> calculate(std::string,std::string,float);
  std::vector<float> calculate();
  
 private:
  
  std::vector<float> fakeTable;
  //Once we have access to teh date we have to define
  //the relative table that we must use in order to merge
  //toghether the differen stream

};

#endif
