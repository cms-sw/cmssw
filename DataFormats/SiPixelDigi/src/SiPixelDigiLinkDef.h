#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include <vector>
#include <map>
#include <utility>

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma extra_include "FWCore/EDProduct/interface/Wrapper.h";
#pragma extra_include "DataFormats/SiPixelDigi/interface/PixelDigi.h";
#pragma extra_include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h";
#pragma link C++ class PixelDigi::Packing+;
#pragma link C++ class PixelDigi+;
#pragma link C++ class std::vector<PixelDigi>+;
#pragma link C++ class std::pair<unsigned int, unsigned int>+;
#pragma link C++ class std::map<unsigned int, std::pair<unsigned int, unsigned int> >+;
#pragma link C++ class PixelDigiCollection+;
#pragma link C++ class edm::Wrapper<PixelDigiCollection>+;
#endif

