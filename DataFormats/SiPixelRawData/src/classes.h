#ifndef SIPIXELRAWDATA_CLASSES_H
#define SIPIXELRAWDATA_CLASSES_H

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

namespace {
  struct dictionary {
    std::vector<SiPixelRawDataError> err0;
    std::map<int, std::vector<SiPixelRawDataError> > err1;
    edm::DetSet<SiPixelRawDataError> err2;
    std::vector<edm::DetSet<SiPixelRawDataError> > err3;
    edm::DetSetVector<SiPixelRawDataError> err4;
    edm::Wrapper< std::vector<SiPixelRawDataError>  > err5;
    edm::Wrapper< std::map<int, std::vector<SiPixelRawDataError> > > err6;
    edm::Wrapper< edm::DetSet<SiPixelRawDataError>  > err7;
    edm::Wrapper< std::vector<edm::DetSet<SiPixelRawDataError> > > err8;
    edm::Wrapper< edm::DetSetVector<SiPixelRawDataError> > err9;
  };
}

#endif // SIPIXELRAWDATA_CLASSES_H

