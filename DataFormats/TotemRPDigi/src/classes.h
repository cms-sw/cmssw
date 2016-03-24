#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/TotemRPDigi/interface/TotemRPDigi.h"

#include <vector>

namespace {
  namespace {
    TotemRPDigi rp_str_dig;
    edm::DetSet<TotemRPDigi> ds_rp_str_dig;
    std::vector<TotemRPDigi> vec_rp_str_dig;
    edm::DetSetVector<TotemRPDigi> dsv_rp_str_dig;
    std::vector<edm::DetSet<TotemRPDigi> > vec_ds_rp_str_dig;
    edm::Wrapper<edm::DetSet<TotemRPDigi> > wds_rp_str_dig;
    edm::Wrapper<edm::DetSetVector<TotemRPDigi> > wdsv_rp_str_dig;
  }
}
