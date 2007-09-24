#include <vector>
#include <string>
#include <iostream>
#include "CalibFormats/SiPixelObjects/interface/PixelDACScanRange.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include <utility>


namespace {
 struct dictionary {
   PixelROCName rocname;
   std::vector<PixelROCName> rocnamevecdummy;
   PixelDACScanRange scanrangedummy;
   std::vector<PixelDACScanRange> scanrangevecdummy;
   PixelModuleName modulenamedummy;
   std::set<PixelModuleName> modulenamesetdummy;
   std::map<PixelModuleName,uint32_t> modulenamemapdummy;
   std::map<std::string,std::string> dummyparameters;
   std::pair<uint32_t,std::vector<uint32_t> > dummy3;
   edm::RefVectorBase<std::vector<std::string> > dummyRefVectorBasePixelCalib0;
   edm::RefVectorBase<std::vector<std::vector<uint32_t> > > dummyRefVectorBasePixelCalib1;
   edm::RefBase<std::pair<uint32_t,std::vector<uint32_t> > > dummyRefVectorBasePixelCalib2;
   edm::RefVectorBase<std::vector<std::pair<uint32_t, std::vector<uint32_t> > >  > dummyRefVectorBasePixelCalib3;
 };
}
