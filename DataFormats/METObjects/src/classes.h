#ifndef METOBJECTS_CLASSES_H
#define METOBJECTS_CLASSES_H

#include "DataFormats/METObjects/interface/MET.h"
#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/METObjects/interface/TowerMET.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "DataFormats/METObjects/interface/CorrMET.h"
#include "DataFormats/METObjects/interface/CorrMETCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
namespace {
  edm::Wrapper<BaseMET> dummy1;
  edm::Wrapper<MET> dummy2;
  edm::Wrapper<METCollection> dummy3;
  edm::Wrapper< std::vector<MET> > dummy4;
  std::vector<MET> dummy5;
  edm::Wrapper<TowerMET> dummy6;
  edm::Wrapper<TowerMETCollection> dummy7;
  edm::Wrapper< std::vector<TowerMET> > dummy8;
  std::vector<TowerMET> dummy9;
  edm::Wrapper<CorrMET> dummy10;
  edm::Wrapper<CorrMETCollection> dummy11;
  edm::Wrapper< std::vector<CorrMET> > dummy12;
  std::vector<CorrMET> dummy13;
}
}
#endif // METOBJECTS_CLASSES_H
