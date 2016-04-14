/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/TotemDigi/interface/TotemRPDigi.h"

#include "DataFormats/TotemDigi/interface/TotemTriggerCounters.h"
#include "DataFormats/TotemDigi/interface/TotemVFATStatus.h"

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

    TotemTriggerCounters dummy10;
    edm::Wrapper<TotemTriggerCounters> dummy11;

    std::map<unsigned int, uint64_t> dummy27;

    TotemVFATStatus dummy30;
    edm::Wrapper< TotemVFATStatus > dummy31;
    edm::DetSetVector<TotemVFATStatus> dummy32;
    edm::Wrapper< edm::DetSetVector<TotemVFATStatus> > dummy33;

    std::bitset<8> dummy50;
    edm::Wrapper< std::bitset<8> > dummy51;
  }
}
