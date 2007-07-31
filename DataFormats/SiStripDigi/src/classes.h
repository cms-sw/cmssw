#ifndef DataFormats_SiStripDigi_Classes_H
#define DataFormats_SiStripDigi_Classes_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
namespace {
  namespace {
    edm::Wrapper<SiStripDigi > zs0;
    edm::Wrapper<std::vector<SiStripDigi> > zs1;
    edm::Wrapper<edm::DetSet<SiStripDigi> > zs2;
    edm::Wrapper<std::vector<edm::DetSet<SiStripDigi> > > zs3;
    edm::Wrapper<edm::DetSetVector<SiStripDigi> > zs4;
  }
}

#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "boost/cstdint.hpp" 
namespace {
  namespace {
    edm::Wrapper<SiStripRawDigi > raw0;
    edm::Wrapper<std::vector<SiStripRawDigi> > raw1;
    edm::Wrapper<edm::DetSet<SiStripRawDigi> > raw2;
    edm::Wrapper<std::vector<edm::DetSet<SiStripRawDigi> > > raw3;
    edm::Wrapper<edm::DetSetVector<SiStripRawDigi> > raw4;
  }
}

#endif // DataFormats_SiStripDigi_Classes_H


 
