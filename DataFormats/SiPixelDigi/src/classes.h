#ifndef SIPIXELDIGI_CLASSES_H
#define SIPIXELDIGI_CLASSES_H

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

namespace {
  namespace {
    std::pair<short,short> patterncontainer;
    SiPixelCalibDigi::datacontainer calibdatacontainer;
    SiPixelCalibDigi calibdigiitself;
    edm::Wrapper<SiPixelCalibDigi::datacontainer> calibdatacontainer0;
    edm::Wrapper<std::vector<SiPixelCalibDigi::datacontainer> > calibdatacontainervec0;
    edm::Wrapper<SiPixelCalibDigi> calibdigi;
    edm::Wrapper<std::vector<SiPixelCalibDigi> > calibdigivec;
    edm::Wrapper<edm::DetSet<SiPixelCalibDigi> > calibdigidetvec;
    edm::Wrapper<std::vector<edm::DetSet<SiPixelCalibDigi> > > calibdigidetset;
    edm::Wrapper<edm::DetSetVector<SiPixelCalibDigi> > calibdigidetsetvec;
    
    edm::Wrapper<PixelDigi> zs0;
    edm::Wrapper<PixelDigiCollection> zsc0;
    edm::Wrapper< std::vector<PixelDigi>  > zs1;
    edm::Wrapper< edm::DetSet<PixelDigi> > zs2;
    edm::Wrapper< std::vector<edm::DetSet<PixelDigi> > > zs3;
    edm::Wrapper< edm::DetSetVector<PixelDigi> > zs4;
  }
}

#endif // SIPIXELDIGI_CLASSES_H
