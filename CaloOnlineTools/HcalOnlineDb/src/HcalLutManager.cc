// -*- C++ -*-
//
// Package: CaloOnlineTools/HcalOnlineDb
// Class:   HcalLutManager
//
/**\class HcalLutManager HcalLutManager.cc CaloOnlineTools/HcalOnlineDb/HcalLutManager.cc

 Description: Various manipulations with trigger primitive lookup tables (LUTs)

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gena Kukartsev
//         Created:  Fri, 14 Mar 2008 00:00:01 GMT
//

#include "CalibCalorimetry/CaloTPG/interface/CaloTPGTranscoderULUT.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/EMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

HcalLutManager::HcalLutManager(const HcalDbService* _conditions,
                               const HcalChannelQuality* _cq,
                               uint32_t _status_word_to_mask) {
  conditions = _conditions;
  emap = conditions->getHcalMapping();
  cq = _cq;
  status_word_to_mask = _status_word_to_mask;
}

HcalLutManager::~HcalLutManager(void) { delete lut_checksums_xml; }

// courtesy of Fedor Ratnikov
std::vector<std::string> HcalLutManager::splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || fLine[i] == '\n' || fLine[i] == '	' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

int HcalLutManager::getInt(const std::string& number) {
  int result;
  std::sscanf(number.c_str(), "%d", &result);
  return result;
}

HcalLutSet HcalLutManager::getLutSetFromFile(const std::string& _filename, int _type) {
  HcalLutSet _lutset;

  std::ifstream infile(_filename.c_str());
  std::string buf;

  if (infile.is_open()) {
    edm::LogInfo("HcalLutManager") << "File " << _filename << " is open..." << std::endl
                                   << "Reading LUTs and their eta/phi/depth/subdet ranges...";

    // get label
    std::getline(infile, _lutset.label);

    if (_type == 1) {  // for linearization LUTs get subdetectors (default)
      //get subdetectors
      std::getline(infile, buf);
      _lutset.subdet = splitString(buf);
    }

    //get min etas
    std::vector<std::string> buf_vec;
    std::getline(infile, buf);
    buf_vec = splitString(buf);
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
      _lutset.eta_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max etas
    std::getline(infile, buf);
    buf_vec = splitString(buf);
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
      _lutset.eta_max.push_back(HcalLutManager::getInt(*iter));
    }

    //get min phis
    std::getline(infile, buf);
    buf_vec = splitString(buf);
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
      _lutset.phi_min.push_back(HcalLutManager::getInt(*iter));
    }

    //get max phis
    std::getline(infile, buf);
    buf_vec = splitString(buf);
    for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
      _lutset.phi_max.push_back(HcalLutManager::getInt(*iter));
    }

    if (_type == 1) {  // for linearization LUTs get depth range (default)
      //get min depths
      std::getline(infile, buf);
      buf_vec = splitString(buf);
      for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
        _lutset.depth_min.push_back(HcalLutManager::getInt(*iter));
      }

      //get max depths
      std::getline(infile, buf);
      buf_vec = splitString(buf);
      for (std::vector<std::string>::const_iterator iter = buf_vec.begin(); iter != buf_vec.end(); iter++) {
        _lutset.depth_max.push_back(HcalLutManager::getInt(*iter));
      }
    }

    bool first_lut_entry = true;
    while (std::getline(infile, buf)) {
      buf_vec = splitString(buf);
      for (unsigned int i = 0; i < buf_vec.size(); i++) {
        if (first_lut_entry) {
          std::vector<unsigned int> _l;
          _lutset.lut.push_back(_l);
        }
        _lutset.lut[i].push_back(HcalLutManager::getInt(buf_vec[i]));
      }
      first_lut_entry = false;
    }
  }

  edm::LogInfo("HcalLutManager") << "done.";

  return _lutset;
}

//
//_____ get HO from ASCII master here ___________________________________
//
std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getLinearizationLutXmlFromAsciiMasterEmap(
    const std::string& _filename, const std::string& _tag, int _crate) {
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from ascii master file...";
  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  EMap _emap(emap);
  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map.size() << " entries";

  // read LUTs and their eta/phi/depth/subdet ranges
  HcalLutSet _set = getLutSetFromFile(_filename);
  int lut_set_size = _set.lut.size();  // number of different luts
  edm::LogInfo("HcalLutManager") << "  ==> " << lut_set_size << " sets of different LUTs read from the master file";

  // setup "zero" LUT for channel masking
  std::vector<unsigned int> zeroLut(128, 0);

  //loop over all EMap channels
  for (std::vector<EMap::EMapRow>::const_iterator row = _map.begin(); row != _map.end(); row++) {
    if ((row->subdet.find("HB") != std::string::npos || row->subdet.find("HE") != std::string::npos ||
         row->subdet.find("HO") != std::string::npos || row->subdet.find("HF") != std::string::npos) &&
        row->subdet.size() == 2) {
      LutXml::Config _cfg;

      // search for the correct LUT for a given channel,
      // higher LUT numbers have priority in case of overlapping
      int lut_index = -1;
      for (int i = 0; i < lut_set_size; i++) {
        if ((row->crate == _crate || _crate == -1) &&  // -1 stands for all crates
            _set.eta_min[i] <= row->ieta && _set.eta_max[i] >= row->ieta && _set.phi_min[i] <= row->iphi &&
            _set.phi_max[i] >= row->iphi && _set.depth_min[i] <= row->idepth && _set.depth_max[i] >= row->idepth &&
            _set.subdet[i].find(row->subdet) != std::string::npos) {
          lut_index = i;
        }
      }
      if (lut_index >= 0) {
        if (_xml.count(row->crate) == 0) {
          _xml.insert(std::pair<int, std::shared_ptr<LutXml>>(row->crate, std::make_shared<LutXml>()));
        }
        _cfg.ieta = row->ieta;
        _cfg.iphi = row->iphi;
        _cfg.depth = row->idepth;
        _cfg.crate = row->crate;
        _cfg.slot = row->slot;
        if (row->topbottom.find('t') != std::string::npos)
          _cfg.topbottom = 1;
        else if (row->topbottom.find('b') != std::string::npos)
          _cfg.topbottom = 0;
        else if (row->topbottom.find('u') != std::string::npos)
          _cfg.topbottom = 2;
        else
          edm::LogWarning("HcalLutManager") << "fpga out of range...";
        _cfg.fiber = row->fiber;
        _cfg.fiberchan = row->fiberchan;
        _cfg.lut_type = 1;
        _cfg.creationtag = _tag;
        _cfg.creationstamp = get_time_stamp();
        _cfg.targetfirmware = "1.0.0";
        _cfg.formatrevision = "1";
        _cfg.generalizedindex =
            _cfg.iphi * 10000 + _cfg.depth * 1000 + (row->ieta > 0) * 100 + abs(row->ieta) +
            (((row->subdet.find("HF") != std::string::npos) && abs(row->ieta) == 29) ? (4 * 10000) : (0));

        DetId _detId(row->rawId);
        uint32_t status_word = cq->getValues(_detId)->getValue();
        if ((status_word & status_word_to_mask) > 0) {
          _cfg.lut = zeroLut;
        } else {
          _cfg.lut = _set.lut[lut_index];
        }
        _xml[row->crate]->addLut(_cfg, lut_checksums_xml);
      }
    }
  }

  return _xml;
}

std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getMasks(int masktype, const std::string& _tag) {
  edm::LogInfo("HcalLutManager") << "Generating TDC masks...";

  EMap _emap(emap);
  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains new" << _map.size() << " entries";

  std::map<int, std::vector<uint64_t>> masks;

  for (const auto& row : _map) {
    std::string subdet = row.subdet;
    if (subdet != "HF")
      continue;
    int crate = row.crate;
    int slot = row.slot;
    int crot = 100 * crate + slot;
    int fiber = row.fiber;
    int channel = row.fiberchan;
    unsigned int finel = 4 * fiber + channel;
    if (masks.count(crot) == 0)
      masks[crot] = {};
    if (finel >= masks[crot].size())
      masks[crot].resize(finel + 1);

    if (masktype == 0) {
      HcalSubdetector _subdet;
      if (row.subdet.find("HB") != std::string::npos)
        _subdet = HcalBarrel;
      else if (row.subdet.find("HE") != std::string::npos)
        _subdet = HcalEndcap;
      else if (row.subdet.find("HO") != std::string::npos)
        _subdet = HcalOuter;
      else if (row.subdet.find("HF") != std::string::npos)
        _subdet = HcalForward;
      else
        _subdet = HcalOther;
      HcalDetId _detid(_subdet, row.ieta, row.iphi, row.idepth);
      masks[crot][finel] = conditions->getHcalTPChannelParameter(_detid)->getMask();
    } else {
      auto parameters = conditions->getHcalTPParameters();
      masks[crot][finel] = masktype == 1 ? parameters->getADCThresholdHF() : parameters->getTDCMaskHF();
    }
  }

  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  for (const auto& i : masks) {
    int crot = i.first;
    int crate = crot / 100;

    LutXml::Config _cfg;
    _cfg.lut_type = 5 + masktype;
    _cfg.crate = crate;
    _cfg.slot = crot % 100;
    _cfg.generalizedindex = crot;
    _cfg.mask = i.second;
    _cfg.creationtag = _tag;
    _cfg.targetfirmware = "1.0.0";
    _cfg.formatrevision = "1";

    if (_xml.count(crate) == 0)
      _xml[crate] = std::make_shared<LutXml>();

    _xml[crate]->addLut(_cfg);
  }

  return _xml;
}

std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getLinearizationLutXmlFromCoderEmap(const HcalTPGCoder& _coder,
                                                                                           const std::string& _tag) {
  edm::LogInfo("HcalLutManager") << "Generating linearization (input) LUTs from HcaluLUTTPGCoder...";
  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  EMap _emap(emap);
  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map.size() << " entries";

  //loop over all EMap channels
  for (std::vector<EMap::EMapRow>::const_iterator row = _map.begin(); row != _map.end(); row++) {
    if ((row->subdet.find("HB") != std::string::npos || row->subdet.find("HE") != std::string::npos ||
         row->subdet.find("HF") != std::string::npos) &&
        row->subdet.size() == 2) {
      LutXml::Config _cfg;

      if (_xml.count(row->crate) == 0) {
        _xml.insert(std::pair<int, std::shared_ptr<LutXml>>(row->crate, std::make_shared<LutXml>()));
      }

      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom.find('t') != std::string::npos)
        _cfg.topbottom = 1;
      else if (row->topbottom.find('b') != std::string::npos)
        _cfg.topbottom = 0;
      else if (row->topbottom.find('u') != std::string::npos)
        _cfg.topbottom = 2;
      else
        edm::LogWarning("HcalLutManager") << "fpga out of range...";
      _cfg.fiber = row->fiber;
      _cfg.fiberchan = row->fiberchan;
      _cfg.lut_type = 1;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp();
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1";
      _cfg.generalizedindex =
          _cfg.iphi * 10000 + _cfg.depth * 1000 + (row->ieta > 0) * 100 + abs(row->ieta) +
          (((row->subdet.find("HF") != std::string::npos) && abs(row->ieta) == 29) ? (4 * 10000) : (0));
      HcalSubdetector _subdet;
      if (row->subdet.find("HB") != std::string::npos)
        _subdet = HcalBarrel;
      else if (row->subdet.find("HE") != std::string::npos)
        _subdet = HcalEndcap;
      else if (row->subdet.find("HO") != std::string::npos)
        _subdet = HcalOuter;
      else if (row->subdet.find("HF") != std::string::npos)
        _subdet = HcalForward;
      else
        _subdet = HcalOther;
      HcalDetId _detid(_subdet, row->ieta, row->iphi, row->idepth);

      for (const auto i : _coder.getLinearizationLUT(_detid))
        _cfg.lut.push_back(i);

      _xml[row->crate]->addLut(_cfg, lut_checksums_xml);
    }
  }

  return _xml;
}

std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getHEFineGrainLUTs(const std::string& _tag) {
  edm::LogInfo("HcalLutManager") << "Generating HE fine grain LUTs";
  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  EMap _emap(emap);
  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map.size() << " entries";

  //loop over all EMap channels
  for (std::vector<EMap::EMapRow>::const_iterator row = _map.begin(); row != _map.end(); row++) {
    if (row->subdet.find("HT") != std::string::npos && row->subdet.size() == 2) {
      int abseta = abs(row->ieta);
      const HcalTopology* topo = cq->topo();
      if (abseta <= topo->lastHBRing() or abseta > topo->lastHERing())
        continue;
      if (abseta >= topo->firstHEDoublePhiRing() and row->fiberchan % 2 == 1)
        continue;  //do only actual physical towers
      LutXml::Config _cfg;

      if (_xml.count(row->crate) == 0) {
        _xml.insert(std::pair<int, std::shared_ptr<LutXml>>(row->crate, std::make_shared<LutXml>()));
      }

      _cfg.ieta = row->ieta;
      _cfg.iphi = row->iphi;
      _cfg.depth = row->idepth;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom.find('t') != std::string::npos)
        _cfg.topbottom = 1;
      else if (row->topbottom.find('b') != std::string::npos)
        _cfg.topbottom = 0;
      else if (row->topbottom.find('u') != std::string::npos)
        _cfg.topbottom = 2;
      else
        edm::LogWarning("HcalLutManager") << "fpga out of range...";
      _cfg.fiber = row->fiber;
      _cfg.fiberchan = row->fiberchan;
      _cfg.lut_type = 4;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp();
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1";
      _cfg.generalizedindex =
          _cfg.iphi * 10000 + _cfg.depth * 1000 + (row->ieta > 0) * 100 + abs(row->ieta) +
          (((row->subdet.find("HF") != std::string::npos) && abs(row->ieta) == 29) ? (4 * 10000) : (0));
      // fine grain LUTs only relevant for HE
      HcalSubdetector _subdet = HcalEndcap;
      HcalDetId _detid(_subdet, row->ieta, row->iphi, row->idepth);

      HcalFinegrainBit fg_algo(conditions->getHcalTPParameters()->getFGVersionHBHE());

      // loop over all possible configurations,
      // computing the LUT for each
      const int n_fgab_bits = 2048;
      for (int i = 0; i < 2 * n_fgab_bits; i++) {
        HcalFinegrainBit::Tower tow;
        for (int k = 0; k < 6; k++) {
          tow[0][k] = (1 << k) & i;
          tow[1][k] = (1 << (k + 6)) & i;
        }
        _cfg.lut.push_back(fg_algo.compute(tow).to_ulong());
      }

      _xml[row->crate]->addLut(_cfg, lut_checksums_xml);
    }
  }

  return _xml;
}

std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getCompressionLutXmlFromCoder(
    const CaloTPGTranscoderULUT& _coder, const std::string& _tag) {
  edm::LogInfo("HcalLutManager") << "Generating compression (output) LUTs from CaloTPGTranscoderULUT," << std::endl
                                 << "initialized from Event Setup" << std::endl;
  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  EMap _emap(emap);

  std::map<int, unsigned int> maxsize;

  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map.size() << " channels";

  //need to equalize compression LUT size in each crate-slot, needed for mixed uHTR
  for (const auto& row : _map) {
    if (row.subdet.find("HT") == std::string::npos)
      continue;
    HcalTrigTowerDetId _detid(row.rawId);
    if (!cq->topo()->validHT(_detid))
      continue;
    int crot = 100 * row.crate + row.slot;
    unsigned int size = _coder.getCompressionLUT(_detid).size();
    if (maxsize.count(crot) == 0 || size > maxsize[crot])
      maxsize[crot] = size;
  }

  for (std::vector<EMap::EMapRow>::const_iterator row = _map.begin(); row != _map.end(); row++) {
    LutXml::Config _cfg;

    if (row->subdet.find("HT") == std::string::npos)
      continue;

    HcalTrigTowerDetId _detid(row->rawId);

    if (!cq->topo()->validHT(_detid))
      continue;

    if (_xml.count(row->crate) == 0) {
      _xml.insert(std::pair<int, std::shared_ptr<LutXml>>(row->crate, std::make_shared<LutXml>()));
    }

    _cfg.ieta = row->ieta;
    _cfg.iphi = row->iphi;
    _cfg.depth = row->idepth;
    _cfg.crate = row->crate;
    _cfg.slot = row->slot;
    if (row->topbottom.find('t') != std::string::npos)
      _cfg.topbottom = 1;
    else if (row->topbottom.find('b') != std::string::npos)
      _cfg.topbottom = 0;
    else if (row->topbottom.find('u') != std::string::npos)
      _cfg.topbottom = 2;
    else
      edm::LogWarning("HcalLutManager") << "fpga out of range...";
    _cfg.fiber = row->fiber;
    _cfg.fiberchan = row->fiberchan;
    _cfg.lut_type = 2;
    _cfg.creationtag = _tag;
    _cfg.creationstamp = get_time_stamp();
    _cfg.targetfirmware = "1.0.0";
    _cfg.formatrevision = "1";
    _cfg.generalizedindex = _cfg.iphi * 10000 + (row->ieta > 0) * 100 + abs(row->ieta);

    _cfg.lut = _coder.getCompressionLUT(_detid);
    auto pWeight = conditions->getHcalTPChannelParameter(_detid, false);
    if (pWeight) {
      _cfg.weight = pWeight->getauxi1();
      _cfg.codedvetothreshold = pWeight->getauxi2();
    }

    int crot = 100 * row->crate + row->slot;
    unsigned int size = _cfg.lut.size();
    if (size < maxsize[crot]) {
      edm::LogWarning("HcalLutManager") << " resizing LUT for " << _detid << ", channel=[" << _cfg.crate << ":"
                                        << _cfg.slot << ":" << _cfg.fiber << ":" << _cfg.fiberchan
                                        << "], using value=" << _cfg.lut[size - 1] << std::endl;
      for (unsigned int i = size; i < maxsize[crot]; ++i)
        _cfg.lut.push_back(_cfg.lut[size - 1]);
    }

    _xml[row->crate]->addLut(_cfg, lut_checksums_xml);
  }

  return _xml;
}

int HcalLutManager::writeLutXmlFiles(std::map<int, std::shared_ptr<LutXml>>& _xml, const std::string& _tag) {
  for (std::map<int, std::shared_ptr<LutXml>>::const_iterator cr = _xml.begin(); cr != _xml.end(); cr++) {
    std::stringstream output_file_name;
    output_file_name << _tag << "_" << cr->first << ".xml";
    cr->second->write(output_file_name.str());
  }
  return 0;
}

void HcalLutManager::addLutMap(std::map<int, std::shared_ptr<LutXml>>& result,
                               const std::map<int, std::shared_ptr<LutXml>>& other) {
  for (std::map<int, std::shared_ptr<LutXml>>::const_iterator lut = other.begin(); lut != other.end(); lut++) {
    edm::LogInfo("HcalLutManager") << "Added LUTs for crate " << lut->first;
    if (result.count(lut->first) == 0) {
      result.insert(*lut);
    } else {
      *(result[lut->first]) += *(lut->second);
    }
  }
}

std::string HcalLutManager::get_time_stamp() {
  return std::format("{:%Y-%m-%d %H:%M:%S}",
                     std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
}

int HcalLutManager::createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC(const std::string& _tag,
                                                                    const HcalTPGCoder& _coder,
                                                                    const CaloTPGTranscoderULUT& _transcoder,
                                                                    const std::string& _lin_file) {
  std::map<int, std::shared_ptr<LutXml>> xml;
  lut_checksums_xml = new XMLDOMBlock("CFGBrick", true);

  if (!_lin_file.empty()) {
    const std::map<int, std::shared_ptr<LutXml>> _lin_lut_ascii_xml =
        getLinearizationLutXmlFromAsciiMasterEmap(_lin_file, _tag, -1);
    addLutMap(xml, _lin_lut_ascii_xml);
  }
  const std::map<int, std::shared_ptr<LutXml>> _lin_lut_xml = getLinearizationLutXmlFromCoderEmap(_coder, _tag);
  addLutMap(xml, _lin_lut_xml);

  const std::map<int, std::shared_ptr<LutXml>> _comp_lut_xml = getCompressionLutXmlFromCoder(_transcoder, _tag);
  addLutMap(xml, _comp_lut_xml);

  const std::map<int, std::shared_ptr<LutXml>> _HE_FG_lut_xml = getHEFineGrainLUTs(_tag);
  addLutMap(xml, _HE_FG_lut_xml);

  for (const auto masktype : {0, 1, 2}) {
    const auto masks = getMasks(masktype, _tag);
    addLutMap(xml, masks);
  }

  const auto _zdc_lut_xml = getZdcLutXml(_coder, _tag, false);
  addLutMap(xml, _zdc_lut_xml);

  const auto _zdc_ootpu_lut_xml = getZdcLutXml(_coder, _tag, true);
  addLutMap(xml, _zdc_ootpu_lut_xml);

  writeLutXmlFiles(xml, _tag);

  std::string checksums_file = _tag + "_checksums.xml";
  lut_checksums_xml->write(checksums_file);

  return 0;
}

std::map<int, std::shared_ptr<LutXml>> HcalLutManager::getZdcLutXml(const HcalTPGCoder& _coder,
                                                                    const std::string& _tag,
                                                                    bool ootpu_lut) {
  edm::LogInfo("HcalLutManager") << "Generating ZDC LUTs ...may the Force be with us...";
  std::map<int, std::shared_ptr<LutXml>> _xml;  // index - crate number

  EMap _emap(emap);

  const std::vector<EMap::EMapRow>& _map = _emap.get_map();
  edm::LogInfo("HcalLutManager") << "EMap contains " << _map.size() << " channels";

  const auto lutMetaDataChannels = conditions->getHcalLutMetadata()->getAllChannels();

  //loop over all EMap channels
  for (std::vector<EMap::EMapRow>::const_iterator row = _map.begin(); row != _map.end(); row++) {
    LutXml::Config _cfg;

    // only ZDC channels
    if (row->zdc_section.find("ZDC") != std::string::npos) {
      if (_xml.count(row->crate) == 0) {
        _xml.insert(std::pair<int, std::shared_ptr<LutXml>>(row->crate, std::make_shared<LutXml>()));
      }

      _cfg.ieta = row->zdc_channel;
      _cfg.depth = row->zdc_zside;
      _cfg.crate = row->crate;
      _cfg.slot = row->slot;
      if (row->topbottom.find('t') != std::string::npos)
        _cfg.topbottom = 1;
      else if (row->topbottom.find('b') != std::string::npos)
        _cfg.topbottom = 0;
      else if (row->topbottom.find('u') != std::string::npos)
        _cfg.topbottom = 2;
      else
        edm::LogWarning("HcalLutManager") << "fpga out of range...";

      if (ootpu_lut)
        _cfg.fiber = row->fiber + 6;
      else
        _cfg.fiber = row->fiber;

      _cfg.fiberchan = row->fiberchan;
      _cfg.lut_type = 1;
      _cfg.creationtag = _tag;
      _cfg.creationstamp = get_time_stamp();
      _cfg.targetfirmware = "1.0.0";
      _cfg.formatrevision = "1";
      _cfg.generalizedindex = 0;

      HcalZDCDetId::Section section = HcalZDCDetId::Unknown;
      if (row->zdc_section == "ZDC EM") {
        section = HcalZDCDetId::EM;
        _cfg.iphi = 1;
      } else if (row->zdc_section == "ZDC HAD") {
        section = HcalZDCDetId::HAD;
        _cfg.iphi = 2;
      } else {
        continue;
      }
      HcalZDCDetId _zdcdetid(section, (row->zdc_zside > 0), row->zdc_channel);

      bool isInLutMetadata = false;
      for (const auto& detid : lutMetaDataChannels) {
        if (detid.det() != DetId::Calo or detid.subdetId() != HcalZDCDetId::SubdetectorId)
          continue;

        HcalZDCDetId zdcdetid(detid.rawId());
        if (_zdcdetid == zdcdetid) {
          isInLutMetadata = true;
          break;
        }
      }

      if (!isInLutMetadata)
        continue;

      for (const auto i : _coder.getLinearizationLUT(_zdcdetid, ootpu_lut)) {
        _cfg.lut.push_back(i);
      }

      _xml[row->crate]->addLut(_cfg, lut_checksums_xml);
    }
  }

  return _xml;
}
