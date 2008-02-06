#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
EVENTSETUP_DATA_REG(SiStripHashedDetId);

#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
EVENTSETUP_DATA_REG(SiStripFecCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
EVENTSETUP_DATA_REG(SiStripDetCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
EVENTSETUP_DATA_REG(SiStripRegionCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
EVENTSETUP_DATA_REG(SiStripGain);

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
EVENTSETUP_DATA_REG(SiStripQuality);
