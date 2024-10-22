#include "FWCore/Utilities/interface/typelookup.h"

#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
TYPELOOKUP_DATA_REG(SiStripHashedDetId);

#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
TYPELOOKUP_DATA_REG(SiStripFecCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
TYPELOOKUP_DATA_REG(SiStripDetCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
TYPELOOKUP_DATA_REG(SiStripRegionCabling);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
TYPELOOKUP_DATA_REG(SiStripGain);

#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"
TYPELOOKUP_DATA_REG(SiStripDelay);

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
TYPELOOKUP_DATA_REG(SiStripQuality);

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"
TYPELOOKUP_DATA_REG(SiStripClusterizerConditions);

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"
TYPELOOKUP_DATA_REG(stripgpu::SiStripClusterizerConditionsGPU);
