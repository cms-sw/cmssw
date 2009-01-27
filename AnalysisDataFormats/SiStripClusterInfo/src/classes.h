#ifndef AnalysisDataFormats_SiStripClusterInfo_Classes_H
#define AnalysisDataFormats_SiStripClusterInfo_Classes_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <vector>
#include "boost/cstdint.hpp" 

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripProcessedRawDigi.h"
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

#endif // AnalysisDataFormats_SiStripClusterInfo_Classes_H
