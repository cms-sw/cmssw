#ifndef METOBJECTS_CLASSES_H
#define METOBJECTS_CLASSES_H

#include "DataFormats/METObjects/interface/TowerMET.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
namespace {
  edm::Wrapper<BaseMET> dummy1;
  edm::Wrapper<TowerMET> dummy2;
  edm::Wrapper<TowerMETCollection> dummy3;
  edm::Wrapper< std::vector<TowerMET> > dummy4;
  std::vector<TowerMET> dummy5;
}
}
#endif // METOBJECTS_CLASSES_H
