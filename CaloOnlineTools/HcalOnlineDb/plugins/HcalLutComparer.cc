// -*- C++ -*-
//
// Package:    CaloOnlineTools/HcalOnlineDb
// Class:      HcalLutComparer
//
/**\class HcalLutComparer HcalLutComparer.cc CaloOnlineTools/HcalOnlineDb/plugins/HcalLutComparer.cc

 Description: Does per-channel LUT payload diff-ing between two input LUT XML files

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joshua C. Hiltbrand
//         Created:  Tue, 12 Nov 2024 05:57:40 GMT
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class HcalLutComparer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalLutComparer(const edm::ParameterSet &);
  ~HcalLutComparer() override {}
  void dumpLutDiff(LutXml &xmls1, LutXml &xmls2, bool testFormat = true);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<HcalElectronicsMap, HcalElectronicsMapRcd> tok_emap_;

  std::string lutXML1_;
  std::string lutXML2_;
  unsigned int verbosity_;
};

HcalLutComparer::HcalLutComparer(const edm::ParameterSet &iConfig) {
  lutXML1_ = iConfig.getParameter<std::string>("lutXML1");
  lutXML2_ = iConfig.getParameter<std::string>("lutXML2");
  verbosity_ = iConfig.getParameter<unsigned int>("verbosity");

  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_emap_ = esConsumes<HcalElectronicsMap, HcalElectronicsMapRcd>();
}

void HcalLutComparer::dumpLutDiff(LutXml &xmls1, LutXml &xmls2, bool testFormat) {
  std::vector<int> detCodes = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 9, -9, 10, -10, 11, -11, 12, -12};
  std::vector<std::string> detNames = {"HBP",
                                       "HBM",
                                       "HEP",
                                       "HEM",
                                       "HOP",
                                       "HOM",
                                       "HFP",
                                       "HFM",
                                       "HTP",
                                       "HTM",
                                       "ZDCP_EM",
                                       "ZDCM_EM",
                                       "ZDCP_HAD",
                                       "ZDCM_HAD",
                                       "ZDCP_LUM",
                                       "ZDCM_LUM",
                                       "ZDCP_RPD",
                                       "ZDCM_RPD"};

  const int HBandHE_fgBits = 0xF000;
  const int HF_fgBits = 0x3000;

  unsigned int nvars = 5;
  enum vtype { total, extra, zeros, match, fgMatch };

  std::map<int, std::vector<int>> n;

  for (const auto &detCode : detCodes) {
    n[detCode] = std::vector<int>{};
    for (unsigned int j = 0; j < nvars; j++) {
      n[detCode].push_back(0);
    }
  }

  for (auto &x1 : xmls1) {
    auto x2 = xmls2.find(x1.first);

    HcalGenericDetId id = HcalGenericDetId(x1.first);
    int subdet = id.genericSubdet();
    if (subdet == 0 or subdet == 6)
      continue;  //'empty' or 'other'

    int side = 1;
    int section = 0;
    if (id.isHcalDetId()) {
      HcalDetId hdetId = HcalDetId(x1.first);
      side = hdetId.zside();
    } else if (id.isHcalTrigTowerDetId()) {
      HcalTrigTowerDetId htdetId = HcalTrigTowerDetId(x1.first);
      side = htdetId.zside();
    } else if (id.isHcalZDCDetId()) {
      HcalZDCDetId zdetId = HcalZDCDetId(x1.first);
      side = zdetId.zside();
      section = zdetId.section();
    }

    int detCode = side * (subdet + section);

    auto &m = n[detCode];

    m[total]++;
    if (x2 == xmls2.end()) {
      m[extra]++;
      if (testFormat)
        std::cout << "Extra detId: " << id << std::endl;
      else
        continue;
    }

    const auto &lut1 = x1.second;
    size_t size = lut1.size();

    bool zero = true;
    for (auto &i : lut1) {
      if (i > 0) {
        zero = false;
        break;
      }
    }
    if (zero) {
      m[zeros]++;
      if (verbosity_ == 1 and testFormat) {
        std::cout << "Zero LUT: " << id << std::endl;
      }
    }

    if (testFormat)
      continue;

    const auto &lut2 = x2->second;
    bool good = size == lut2.size();
    bool fgGood = size == lut2.size();
    for (size_t i = 0; i < size and (good or fgGood); ++i) {
      if (lut1[i] != lut2[i]) {
        good = false;
        if (subdet == 1 || subdet == 2) {
          if ((lut1[i] & HBandHE_fgBits) != (lut2[i] & HBandHE_fgBits))
            fgGood = false;
        } else if (subdet == 4) {
          if ((lut1[i] & HF_fgBits) != (lut2[i] & HF_fgBits))
            fgGood = false;
        }

        if (verbosity_ == 2) {
          std::cout << "Mismatch in index=" << std::setw(3) << i << ", " << std::setw(4) << lut1[i]
                    << "!=" << std::setw(4) << lut2[i] << ", " << id << '\n';
        }
      }
    }
    if (good)
      m[match]++;
    if (fgGood)
      m[fgMatch]++;
  }

  if (testFormat) {
    std::cout << std::setw(9) << "Det"
              << "  " << std::setw(6) << "total"
              << "  " << std::setw(6) << "zeroes"
              << "  " << std::setw(6) << "extra" << '\n';
    for (unsigned int i = 0; i < detCodes.size(); i++) {
      int detCode = detCodes.at(i);
      const std::string &detName = detNames.at(i);
      std::cout << std::setw(9) << detName << "  " << std::setw(6) << n[detCode][total] << "  " << std::setw(6)
                << n[detCode][zeros] << "  " << std::setw(6) << n[detCode][extra] << '\n';
      if (detCode < 0) {
        std::cout << std::setw(9) << ""
                  << "  " << std::setw(6) << n[detCode][total] + n[-detCode][total] << "  " << std::setw(6)
                  << n[detCode][zeros] + n[-detCode][zeros] << "  " << std::setw(6)
                  << n[detCode][extra] + n[-detCode][extra] << "\n\n";
      }
    }
    std::cout << "--------------------------------------------" << std::endl;
  } else {
    bool good = true;
    for (const auto &it : n) {
      if (it.second[total] != it.second[match]) {
        good = false;
      }
    }
    std::cout << std::setw(9) << "Det"
              << "  " << std::setw(6) << "total"
              << "  " << std::setw(6) << "match"
              << "  " << std::setw(8) << "mismatch"
              << "  " << std::setw(8) << "FG match"
              << "  " << std::setw(11) << "FG mismatch" << '\n';
    for (unsigned int i = 0; i < detCodes.size(); i++) {
      int detCode = detCodes.at(i);
      const std::string &detName = detNames.at(i);
      std::cout << std::setw(9) << detName << "  " << std::setw(6) << n[detCode][total] << "  " << std::setw(6)
                << n[detCode][match] << "  " << std::setw(8) << n[detCode][total] - n[detCode][match] << "  "
                << std::setw(8) << n[detCode][fgMatch] << "  " << std::setw(11)
                << n[detCode][total] - n[detCode][fgMatch] << '\n';
      if (detCode < 0) {
        std::cout << std::setw(9) << ""
                  << "  " << std::setw(6) << n[detCode][total] + n[-detCode][total] << "  " << std::setw(6)
                  << n[detCode][match] + n[-detCode][match] << "  " << std::setw(8)
                  << n[detCode][total] - n[detCode][match] + n[-detCode][total] - n[-detCode][match] << "  "
                  << std::setw(8) << n[detCode][fgMatch] + n[-detCode][fgMatch] << "  " << std::setw(11)
                  << n[detCode][total] - n[detCode][fgMatch] + n[-detCode][total] - n[-detCode][fgMatch] << "\n\n";
      }
    }
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << (good ? "PASS!" : "FAIL!") << std::endl;
  }
}

void HcalLutComparer::analyze(const edm::Event &, const edm::EventSetup &iSetup) {
  const HcalElectronicsMap *electronicsMap = &iSetup.getData(tok_emap_);

  LutXml xmls1(edm::FileInPath(lutXML1_).fullPath());
  LutXml xmls2(edm::FileInPath(lutXML2_).fullPath());

  xmls1.create_lut_map(electronicsMap);
  xmls2.create_lut_map(electronicsMap);

  std::cout << lutXML1_ << std::endl;
  dumpLutDiff(xmls1, xmls2);

  std::cout << lutXML2_ << std::endl;
  dumpLutDiff(xmls2, xmls1);

  std::cout << "Comparison" << std::endl;
  dumpLutDiff(xmls1, xmls2, false);
}

void HcalLutComparer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("lutXML1", "")->setComment("Path to a LUT XML file for diff-ing");
  desc.add<std::string>("lutXML2", "")->setComment("Path to a LUT XML file for diff-ing");
  desc.add<uint32_t>("verbosity", 0)->setComment("Verbosity level for printing out LUT diff statistics");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HcalLutComparer);
