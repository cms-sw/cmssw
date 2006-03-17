#ifndef DataFormats_SiStripDigi_Classes_H
#define DataFormats_SiStripDigi_Classes_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"

namespace {
  namespace {
    edm::Wrapper<SiStripDigi> zs_digi;
    edm::Wrapper< edm::DetSetVector<SiStripDigi> > zs_digis;
    edm::Wrapper<SiStripRawDigi> rawdigi;
    edm::Wrapper< edm::DetSetVector<SiStripRawDigi> > raw_digis;
    edm::Wrapper<SiStripEventSummary> summary;
  }
}

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
namespace {
  namespace {
    edm::Wrapper<StripDigiCollection> collection;
  }
}

#endif // DataFormats_SiStripDigi_Classes_H


 
