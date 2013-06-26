#ifndef DataFormats_SiStripDigi_Classes_H
#define DataFormats_SiStripDigi_Classes_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <vector>

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
namespace {
  struct dictionary1 {
    edm::Wrapper<SiStripDigi > zs0;
    edm::Wrapper<std::vector<SiStripDigi> > zs1;
    edm::Wrapper<edm::DetSet<SiStripDigi> > zs2;
    edm::Wrapper<std::vector<edm::DetSet<SiStripDigi> > > zs3;
    edm::Wrapper<edm::DetSetVector<SiStripDigi> > zs4;
    edm::Wrapper<edmNew::DetSetVector<SiStripDigi> > zs4_bis;
  };
}

#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "boost/cstdint.hpp" 
namespace {
  struct dictionary2 {
    edm::Wrapper<SiStripRawDigi > raw0;
    edm::Wrapper<std::vector<SiStripRawDigi> > raw1;
    edm::Wrapper<edm::DetSet<SiStripRawDigi> > raw2;
    edm::Wrapper<std::vector<edm::DetSet<SiStripRawDigi> > > raw3;
    edm::Wrapper<edm::DetSetVector<SiStripRawDigi> > raw4;
    edm::Wrapper<edmNew::DetSetVector<SiStripRawDigi> > raw4_bis;
  };
}

#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
namespace {
  struct dictionary3 {
    edm::Wrapper<SiStripProcessedRawDigi > praw0;
    edm::Wrapper<std::vector<SiStripProcessedRawDigi> > praw1;
    edm::Wrapper<edm::DetSet<SiStripProcessedRawDigi> > praw2;
    edm::Wrapper<std::vector<edm::DetSet<SiStripProcessedRawDigi> > > praw3;
    edm::Wrapper<edm::DetSetVector<SiStripProcessedRawDigi> > praw4;
    edm::Wrapper<edmNew::DetSetVector<SiStripProcessedRawDigi> > praw4_bis;
  };
}

#endif // DataFormats_SiStripDigi_Classes_H


 
