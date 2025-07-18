// -*- C++ -*-
//
// Package:    Test/HcalLutComparer
// Class:      HcalLutComparer
//
/**\class HcalLutComparer HcalLutComparer.cc Test/HcalLutComparer/plugins/HcalLutComparer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joshua C. Hiltbrand
//         Created:  Tue, 12 Nov 2024 05:57:40 GMT
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

class HcalLutComparer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HcalLutComparer(const edm::ParameterSet &);
  ~HcalLutComparer() override {}
  void dumpLutDiff(LutXml &xmls1, LutXml &xmls2, bool testFormat);
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

void HcalLutComparer::dumpLutDiff(LutXml &xmls1, LutXml &xmls2, bool testFormat = true) {
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
          std::cout << Form("Mismatach in index=%3d, %4d!=%4d, ", int(i), lut1[i], lut2[i]) << id << std::endl;
        }
      }
    }
    if (good)
      m[match]++;
    if (fgGood)
      m[fgMatch]++;
  }

  if (testFormat) {
    std::cout << Form("%9s  %6s  %6s  %6s", "Det", "total", "zeroes", "extra") << std::endl;
    for (unsigned int i = 0; i < detCodes.size(); i++) {
      int detCode = detCodes.at(i);
      std::string detName = detNames.at(i);
      std::cout << Form("%9s  %6d  %6d  %6d", detName.c_str(), n[detCode][total], n[detCode][zeros], n[detCode][extra])
                << std::endl;
      if (detCode < 0) {
        std::cout << Form("%9s  %6d  %6d  %6d",
                          " ",
                          n[detCode][total] + n[-1 * detCode][total],
                          n[detCode][zeros] + n[-1 * detCode][zeros],
                          n[detCode][extra] + n[-1 * detCode][extra])
                  << std::endl;
        std::cout << std::endl;
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
    std::cout << Form("%9s  %6s  %6s  %8s  %8s  %11s", "Det", "total", "match", "mismatch", "FG match", "FG mismatch")
              << std::endl;
    for (unsigned int i = 0; i < detCodes.size(); i++) {
      int detCode = detCodes.at(i);
      std::string detName = detNames.at(i);
      std::cout << Form("%9s  %6d  %6d  %8d  %8d  %11d",
                        detName.c_str(),
                        n[detCode][total],
                        n[detCode][match],
                        n[detCode][total] - n[detCode][match],
                        n[detCode][fgMatch],
                        n[detCode][total] - n[detCode][fgMatch])
                << std::endl;
      if (detCode < 0) {
        std::cout << Form("%9s  %6d  %6d  %8d  %8d  %11d",
                          " ",
                          n[detCode][total] + n[-1 * detCode][total],
                          n[detCode][match] + n[-1 * detCode][match],
                          n[detCode][total] - n[detCode][match] + n[-1 * detCode][total] - n[-1 * detCode][match],
                          n[detCode][fgMatch] + n[-1 * detCode][fgMatch],
                          n[detCode][total] - n[detCode][fgMatch] + n[-1 * detCode][total] - n[-1 * detCode][fgMatch])
                  << std::endl;
        std::cout << std::endl;
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
  dumpLutDiff(xmls1, xmls2, true);

  std::cout << lutXML2_ << std::endl;
  dumpLutDiff(xmls2, xmls1, true);

  std::cout << "Comparison" << std::endl;
  dumpLutDiff(xmls1, xmls2, false);
}

void HcalLutComparer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HcalLutComparer);
