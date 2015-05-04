// -*- C++ -*-

//____________________________________________________________________________||
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iterator>

//____________________________________________________________________________||
METCorrectorParameters::METCorrectorParameters(const std::string& fFile, const std::string& fSection) 
{ 
  std::cout << " MetCorrectorParameters::MetCorrectorParameters(const std::string& fFile, const std::string& fSection)  " << std::endl;
  std::cout << "fFile= " << fFile << std::endl;
  std::ifstream input(fFile.c_str());
  std::string currentSection = "";
  std::string line;
  std::string currentDefinitions = "";
  while (std::getline(input,line)) 
    {
      std::cout << " Line of parameters " << line << std::endl;
      std::istringstream iss(line);
      std::string sub;
      while (iss >> sub)
      {
        mRecord.push_back(atof(sub.c_str()));
        std::cout << "Substring: " << sub << std::endl;
       } 
     }
      std::cout << "mRecord size= " << mRecord.size() << std::endl;
}

void METCorrectorParameters::printScreen() const
{
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout<<"////////  PARAMETERS: //////////////////////"<<std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
  std::cout << " XY-Shift corection constants " << std::endl;
  std::cout << " a_x b_x a_y b_y " << std::endl;
  for(size_t i=0; i<mRecord.size(); ++i){
     std::cout << " " << mRecord[i];
  }
  std::cout << std::endl;
  std::cout<<"--------------------------------------------"<<std::endl;
}

//____________________________________________________________________________||
#include "FWCore/Utilities/interface/typelookup.h"
 
TYPELOOKUP_DATA_REG(METCorrectorParameters);
