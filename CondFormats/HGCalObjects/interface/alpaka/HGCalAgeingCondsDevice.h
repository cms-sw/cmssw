#ifndef CondFormats_HGCalObjects_interface_alpaka_HGCalAgeingCondsDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_HGCalAgeingCondsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    using HGCalAgeingCondsDevice = PortableCollection<::hgcal::HGCalAgeingCondsSoA>;
    using HGCalAgeingCondsHost = ::hgcal::HGCalAgeingCondsHost;

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_HGCalAgeingCondsDevice_h
