#ifndef METOBJECTS_CLASSES_H
#define METOBJECTS_CLASSES_H

#include "DataFormats/METObjects/interface/MET.h"
#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/METObjects/interface/TowerMET.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    edm::Wrapper<BaseMETv0> dummy1;
    edm::Wrapper<METv0> dummy2;
    edm::Wrapper<METv0Collection> dummy3;
    edm::Wrapper< std::vector<METv0> > dummy4;
    std::vector<METv0> dummy5;
    edm::Wrapper<TowerMETv0> dummy10;
    edm::Wrapper<TowerMETv0Collection> dummy11;
    edm::Wrapper< std::vector<TowerMETv0> > dummy12;
    std::vector<TowerMETv0> dummy13;
  };
}
#endif // METOBJECTS_CLASSES_H
