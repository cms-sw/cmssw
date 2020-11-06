/****************************************************************************
 *
 * This is a part of CMS-TOTEM  PPSoffline software.
 * Authors:
 *  Jan Kašpar (jan.kaspar@gmail.com)
 *  adapted for CondFormats by H. Malbouisson & C. Mora Herrera
 ****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include <set>

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData& CTPPSRPAlignmentCorrectionsData::getRPCorrection(unsigned int id) { return rps_[id]; }

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData CTPPSRPAlignmentCorrectionsData::getRPCorrection(unsigned int id) const {
  CTPPSRPAlignmentCorrectionData align_corr;
  auto it = rps_.find(id);
  if (it != rps_.end())
    align_corr = it->second;
  return align_corr;
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData& CTPPSRPAlignmentCorrectionsData::getSensorCorrection(unsigned int id) {
  return sensors_[id];
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData CTPPSRPAlignmentCorrectionsData::getSensorCorrection(unsigned int id) const {
  CTPPSRPAlignmentCorrectionData align_corr;
  auto it = sensors_.find(id);
  if (it != sensors_.end())
    align_corr = it->second;
  return align_corr;
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData CTPPSRPAlignmentCorrectionsData::getFullSensorCorrection(unsigned int id,
                                                                                        bool useRPErrors) const {
  // by default empty correction
  CTPPSRPAlignmentCorrectionData align_corr;

  // if found, add sensor correction (with its uncertainty)
  auto sIt = sensors_.find(id);
  if (sIt != sensors_.end())
    align_corr.add(sIt->second, true);

  // if found, add RP correction (depending on the flag, with or without its uncertainty)
  auto rpIt = rps_.find(CTPPSDetId(id).rpId());
  if (rpIt != rps_.end())
    align_corr.add(rpIt->second, useRPErrors);

  return align_corr;
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::setRPCorrection(unsigned int id, const CTPPSRPAlignmentCorrectionData& ac) {
  rps_[id] = ac;
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::setSensorCorrection(unsigned int id, const CTPPSRPAlignmentCorrectionData& ac) {
  sensors_[id] = ac;
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::addRPCorrection(
    unsigned int id, const CTPPSRPAlignmentCorrectionData& a, bool sumErrors, bool addSh, bool addRot) {
  auto it = rps_.find(id);
  if (it == rps_.end())
    rps_.insert(mapType::value_type(id, a));
  else
    it->second.add(a, sumErrors, addSh, addRot);
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::addSensorCorrection(
    unsigned int id, const CTPPSRPAlignmentCorrectionData& a, bool sumErrors, bool addSh, bool addRot) {
  auto it = sensors_.find(id);
  if (it == sensors_.end())
    sensors_.insert(mapType::value_type(id, a));
  else
    it->second.add(a, sumErrors, addSh, addRot);
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::addCorrections(const CTPPSRPAlignmentCorrectionsData& nac,
                                                     bool sumErrors,
                                                     bool addSh,
                                                     bool addRot) {
  for (const auto& it : nac.rps_)
    addRPCorrection(it.first, it.second, sumErrors, addSh, addRot);

  for (const auto& it : nac.sensors_)
    addSensorCorrection(it.first, it.second, sumErrors, addSh, addRot);
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsData::clear() {
  rps_.clear();
  sensors_.clear();
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& s, const CTPPSRPAlignmentCorrectionsData& corr) {
  for (const auto& p : corr.getRPMap()) {
    s << "RP " << p.first << ": " << p.second << std::endl;
  }

  for (const auto& p : corr.getSensorMap()) {
    s << "sensor " << p.first << ": " << p.second << std::endl;
  }

  return s;
}

//----------------------------------------------------------------------------------------------------
