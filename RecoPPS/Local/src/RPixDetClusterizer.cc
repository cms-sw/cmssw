#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoPPS/Local/interface/RPixDetClusterizer.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

namespace {
  constexpr int maxCol = CTPPSPixelCluster::MAXCOL;
  constexpr int maxRow = CTPPSPixelCluster::MAXROW;
  constexpr double highRangeCal = 1800.;
  constexpr double lowRangeCal = 260.;
  constexpr int rocMask = 0xE000;
  constexpr int rocOffset = 13;
}  // namespace

RPixDetClusterizer::RPixDetClusterizer(edm::ParameterSet const &conf) : SeedVector_(0) {
  verbosity_ = conf.getUntrackedParameter<int>("RPixVerbosity");
  SeedADCThreshold_ = conf.getParameter<int>("SeedADCThreshold");
  ADCThreshold_ = conf.getParameter<int>("ADCThreshold");
  ElectronADCGain_ = conf.getParameter<double>("ElectronADCGain");
  VcaltoElectronGain_ = conf.getParameter<int>("VCaltoElectronGain");
  VcaltoElectronOffset_ = conf.getParameter<int>("VCaltoElectronOffset");
  doSingleCalibration_ = conf.getParameter<bool>("doSingleCalibration");
}

RPixDetClusterizer::~RPixDetClusterizer() {}

void RPixDetClusterizer::buildClusters(unsigned int detId,
                                       const std::vector<CTPPSPixelDigi> &digi,
                                       std::vector<CTPPSPixelCluster> &clusters,
                                       const CTPPSPixelGainCalibrations *pcalibrations,
                                       const CTPPSPixelAnalysisMask *maskera) {
  std::map<uint32_t, CTPPSPixelROCAnalysisMask> const &mask = maskera->analysisMask;
  std::set<std::pair<unsigned char, unsigned char> > maskedPixels;

  bool fullyMaskedRoc[6][6] = {{false}};
  CTPPSPixelDetId currentId(detId);

  // read and store masked pixels after converting ROC numbering into module numbering
  CTPPSPixelIndices pI;
  for (auto const &det : mask) {
    uint32_t planeId = det.first & ~rocMask;

    if (planeId == detId) {
      unsigned int rocNum = (det.first & rocMask) >> rocOffset;
      if (rocNum > 5)
        throw cms::Exception("InvalidRocNumber") << "roc number from mask: " << rocNum;
      if (det.second.fullMask)
        fullyMaskedRoc[currentId.plane()][rocNum] = true;
      for (auto const &paio : det.second.maskedPixels) {
        std::pair<unsigned char, unsigned char> pa = paio;
        int modCol, modRow;
        pI.transformToModule(pa.second, pa.first, rocNum, modCol, modRow);
        std::pair<int, int> modPix(modRow, modCol);
        maskedPixels.insert(modPix);
      }
    }
  }

  if (verbosity_)
    edm::LogInfo("RPixDetClusterizer") << detId << " received digi.size()=" << digi.size();

  // creating a set of CTPPSPixelDigi's and fill it

  rpix_digi_set_.clear();
  rpix_digi_set_.insert(digi.begin(), digi.end());
  SeedVector_.clear();

  // calibrate digis here
  calib_rpix_digi_map_.clear();

  for (auto const &RPdit : rpix_digi_set_) {
    uint8_t row = RPdit.row();
    uint8_t column = RPdit.column();
    if (row > maxRow || column > maxCol)
      throw cms::Exception("CTPPSDigiOutofRange") << " row = " << row << "  column = " << column;

    std::pair<unsigned char, unsigned char> pixel = std::make_pair(row, column);
    unsigned short adc = RPdit.adc();
    unsigned short electrons = 0;

    // check if pixel is noisy or dead (i.e. in the mask)
    const bool is_in = maskedPixels.find(pixel) != maskedPixels.end();

    // check if pixel inside a fully masked roc
    int piano = currentId.plane();
    int rocId = pI.getROCId(column, row);

    if (!is_in && !fullyMaskedRoc[piano][rocId]) {
      //calibrate digi and store the new ones
      electrons = calibrate(detId, adc, row, column, pcalibrations);
      if (electrons < SeedADCThreshold_ * ElectronADCGain_)
        electrons = SeedADCThreshold_ * ElectronADCGain_;
      RPixCalibDigi calibDigi(row, column, adc, electrons);
      unsigned int index = column * maxRow + row;
      calib_rpix_digi_map_.insert(std::pair<unsigned int, RPixCalibDigi>(index, calibDigi));
      SeedVector_.push_back(calibDigi);
    }
  }
  if (verbosity_)
    edm::LogInfo("RPixDetClusterizer") << " RPix set size = " << calib_rpix_digi_map_.size();

  for (auto SeedIt : SeedVector_) {
    make_cluster(SeedIt, clusters);
  }
}

void RPixDetClusterizer::make_cluster(RPixCalibDigi const &aSeed, std::vector<CTPPSPixelCluster> &clusters) {
  /// check if seed already used
  unsigned int seedIndex = aSeed.column() * maxRow + aSeed.row();
  if (calib_rpix_digi_map_.find(seedIndex) == calib_rpix_digi_map_.end()) {
    return;
  }
  // creating a temporary cluster
  RPixTempCluster atempCluster;

  // filling the cluster with the seed
  atempCluster.addPixel(aSeed.row(), aSeed.column(), aSeed.electrons());
  calib_rpix_digi_map_.erase(seedIndex);

  while (!atempCluster.empty()) {
    //This is the standard algorithm to find and add a pixel
    auto curInd = atempCluster.top();
    atempCluster.pop();
    for (auto c = std::max(0, int(atempCluster.col[curInd]) - 1);
         c < std::min(int(atempCluster.col[curInd]) + 2, maxCol);
         ++c) {
      for (auto r = std::max(0, int(atempCluster.row[curInd]) - 1);
           r < std::min(int(atempCluster.row[curInd]) + 2, maxRow);
           ++r) {
        unsigned int currIndex = c * maxRow + r;
        if (calib_rpix_digi_map_.find(currIndex) != calib_rpix_digi_map_.end()) {
          if (!atempCluster.addPixel(r, c, calib_rpix_digi_map_[currIndex].electrons())) {
            clusters.emplace_back(atempCluster.isize, atempCluster.adc, atempCluster.row, atempCluster.col);
            return;
          }
          calib_rpix_digi_map_.erase(currIndex);
        }
      }
    }

  }  // while accretion

  clusters.emplace_back(atempCluster.isize, atempCluster.adc, atempCluster.row, atempCluster.col);
}

int RPixDetClusterizer::calibrate(
    unsigned int detId, int adc, int row, int col, const CTPPSPixelGainCalibrations *pcalibrations) {
  float gain = 0;
  float pedestal = 0;
  int electrons = 0;

  if (!doSingleCalibration_) {
    const CTPPSPixelGainCalibration &DetCalibs = pcalibrations->getGainCalibration(detId);

    if (DetCalibs.getDetId() != 0) {
      gain = DetCalibs.getGain(col, row) * highRangeCal / lowRangeCal;
      pedestal = DetCalibs.getPed(col, row);
      float vcal = (adc - pedestal) * gain;
      electrons = int(vcal * VcaltoElectronGain_ + VcaltoElectronOffset_);
    } else {
      gain = ElectronADCGain_;
      pedestal = 0;
      electrons = int(adc * gain - pedestal);
    }

    if (verbosity_)
      edm::LogInfo("RPixCalibration") << "calibrate:  adc = " << adc << " electrons = " << electrons
                                      << " gain = " << gain << " pedestal = " << pedestal;
  } else {
    gain = ElectronADCGain_;
    pedestal = 0;
    electrons = int(adc * gain - pedestal);
    if (verbosity_)
      edm::LogInfo("RPixCalibration") << "calibrate:  adc = " << adc << " electrons = " << electrons
                                      << " ElectronADCGain = " << gain << " pedestal = " << pedestal;
  }
  if (electrons < 0) {
    edm::LogInfo("RPixCalibration") << "RPixDetClusterizer::calibrate: *** electrons < 0 *** --> " << electrons
                                    << "  --> electrons = 0";
    electrons = 0;
  }

  return electrons;
}
