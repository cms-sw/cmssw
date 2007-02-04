#ifndef SISTRIPCLUSTERINFO_CLASSES_H
#define SISTRIPCLUSTERINFO_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>
#include "DataFormats/Common/interface/DetSetVector.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
namespace {
  namespace {
    edm::Wrapper< SiStripClusterInfo > adummy0;
    edm::Wrapper< std::vector<SiStripClusterInfo>  > adummy1;
    edm::Wrapper< edm::DetSet<SiStripClusterInfo> > adummy2;
    edm::Wrapper< std::vector<edm::DetSet<SiStripClusterInfo> > > adummy3;
    edm::Wrapper< edm::DetSetVector<SiStripClusterInfo> > adummy4;
    //
    edm::Wrapper< std::map<unsigned int, std::vector<SiStripClusterInfo> > > adummy5;
  }
}

#endif // SISTRIPCLUSTERINFO_CLASSES_H
