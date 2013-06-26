#ifndef DataFormats_SiStripCommon_Classes_H
#define DataFormats_SiStripCommon_Classes_H

#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
namespace { 
  struct dictionary1 { 
    // edm::Wrapper< SiStripFecKey > fec;
    // edm::Wrapper< SiStripFedKey > fed;
    // edm::Wrapper< SiStripDetKey > det;
    // edm::Wrapper< SiStripNullKey > null;
  };
}

#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
namespace {
  struct dictionary2 {
    edm::Wrapper< sistrip::RunType > run_type; 
    edm::Wrapper< sistrip::ApvReadoutMode > apv_mode; 
    edm::Wrapper< sistrip::FedReadoutMode > fed_mode; 
    edm::Wrapper< SiStripEventSummary > summary;
  };
}

#endif // DataFormats_SiStripCommon_Classes_H
