#ifndef METOBJECTS_CLASSES_H
#define METOBJECTS_CLASSES_H

#include "DataFormats/METObjects/interface/MET.h"
#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/METObjects/interface/TowerMET.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
namespace {
  edm::Wrapper<BaseMET> dummy1;
  edm::Wrapper<MET> dummy2;
  edm::Wrapper<METCollection> dummy3;
  edm::Wrapper< std::vector<MET> > dummy4;
  std::vector<MET> dummy5;
  edm::Wrapper<TowerMET> dummy10;
  edm::Wrapper<TowerMETCollection> dummy11;
  edm::Wrapper< std::vector<TowerMET> > dummy12;
  std::vector<TowerMET> dummy13;
}
}
#endif // METOBJECTS_CLASSES_H
