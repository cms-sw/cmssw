#ifndef SIPIXELRAWDATA_CLASSES_H
#define SIPIXELRAWDATA_CLASSES_H

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    std::vector<SiPixelRawDataError> err0;
    std::map<int, std::vector<SiPixelRawDataError> > err1;
    edm::Wrapper< std::vector<SiPixelRawDataError>  > err2;
    edm::Wrapper< std::map<int, std::vector<SiPixelRawDataError> > > err3;
  }
}

#endif // SIPIXELRAWDATA_CLASSES_H

