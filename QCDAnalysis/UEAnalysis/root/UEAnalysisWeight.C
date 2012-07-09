#include "UEAnalysisWeight.h"
#include <vector>
#include <math.h>

UEAnalysisWeight::UEAnalysisWeight()
{
  std::cout << "UEAnalysisWeight constructor " <<std::endl;
  fakeTable.clear();
}

std::vector<float> UEAnalysisWeight::calculate(string tkPt, std::string trigger, float lumi)
{
  //This method will be filled once we start analyze the data
}


std::vector<float> UEAnalysisWeight::calculate()
{
  for(int i=0;i<30;++i)
    fakeTable.push_back(1.0);
  return fakeTable;
}

