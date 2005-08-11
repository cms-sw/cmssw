#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include <vector>
#include <map>
#include <utility>

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma extra_include "FWCore/EDProduct/interface/Wrapper.h";
#pragma extra_include "DataFormats/SiStripDigi/interface/StripDigi.h";
#pragma extra_include "DataFormats/SiStripDigi/interface/StripDigiCollection.h";
#pragma link C++ class StripDigi+;
#pragma link C++ class std::vector<StripDigi>+;
#pragma link C++ class std::pair<unsigned int, unsigned int>+;
#pragma link C++ class std::map<unsigned int, std::pair<unsigned int, unsigned int> >+;
#pragma link C++ class StripDigiCollection+;
#pragma link C++ class edm::Wrapper<StripDigiCollection>+;
#endif

