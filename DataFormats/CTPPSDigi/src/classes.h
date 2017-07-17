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

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemTriggerCounters.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"

#include <vector>

namespace DataFormats_CTPPSDigi {
  struct dictionary {
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

    TotemFEDInfo fi;
    std::vector<TotemFEDInfo> v_fi;
    edm::Wrapper<std::vector<TotemFEDInfo>> w_v_fi;

    CTPPSDiamondDigi rm_diamo_dig;
    edm::DetSet<CTPPSDiamondDigi> ds_rp_diamo_dig;
    std::vector<CTPPSDiamondDigi> vec_rp_diamo_dig;
    edm::DetSetVector<CTPPSDiamondDigi> dsv_rp_diamo_dig;
    std::vector<edm::DetSet<CTPPSDiamondDigi> > vec_ds_rp_diamo_dig;
    edm::Wrapper<edm::DetSet<CTPPSDiamondDigi> > wds_rp_diamo_dig;
    edm::Wrapper<edm::DetSetVector<CTPPSDiamondDigi> > wdsv_rp_diamo_dig;

    HPTDCErrorFlags rm_hptdcerr;
    CTPPSPixelDigi ff0;
    CTPPSPixelDigiCollection ffc0;
    std::vector<CTPPSPixelDigi>  ff1;
    edm::DetSet<CTPPSPixelDigi>  ff2;
    std::vector<edm::DetSet<CTPPSPixelDigi> >  ff3;
    edm::DetSetVector<CTPPSPixelDigi> ff4;


    edm::Wrapper<CTPPSPixelDigi> wff0;
    edm::Wrapper<CTPPSPixelDigiCollection> wffc0;
    edm::Wrapper< std::vector<CTPPSPixelDigi>  > wff1;
    edm::Wrapper< edm::DetSet<CTPPSPixelDigi> > wff2;
    edm::Wrapper< std::vector<edm::DetSet<CTPPSPixelDigi> > > wff3;
    edm::Wrapper< edm::DetSetVector<CTPPSPixelDigi> > wff4;

  };
}
