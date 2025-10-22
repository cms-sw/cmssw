#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h"

int OmtfPhase2AngleConverter::getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const {
  constexpr int dtPhiBins = 65536;          //65536. for [-0.5,0.5] radians
  double hsPhiPitch = 2 * M_PI / nPhiBins;  // width of phi Pitch, related to halfStrip at CSC station 2

  int sector = dtScNum + 1;  //NOTE: there is a inconsistency in DT sector numb. Thus +1 needed to get detector numb.

  double scale = 0.5 / dtPhiBins / hsPhiPitch;  //was 0.8
  int scale_coeff = lround(scale * (1 << 15));

  int ichamber = sector - 1;
  if (ichamber > 6)
    ichamber = ichamber - 12;

  int offsetGlobal = (int)nPhiBins * ichamber / 12;

  int phiConverted = ((dtPhi * scale_coeff) >> 15) + offsetGlobal - phiZero;

  return config->foldPhi(phiConverted);
}

int OmtfPhase2AngleConverter::getGlobalEta(DTChamberId dTChamberId,
                                           const L1Phase2MuDTThContainer* dtThDigis,
                                           int bxNum) const {
  int dtThBins = 65536;  //65536. for [-6.3,6.3]
  float kconv = 1 / (dtThBins / 2.);

  float eta = -999;
  // get the theta digi
  bool foundeta = false;
  int thetaDigiCnt = 0;
  for (const auto& thetaDigi : (*(dtThDigis->getContainer()))) {
    if (thetaDigi.whNum() == dTChamberId.wheel() && thetaDigi.stNum() == dTChamberId.station() &&
        thetaDigi.scNum() == (dTChamberId.sector() - 1) && (thetaDigi.bxNum() - 20) == bxNum) {
      // get the theta digi
      float k = thetaDigi.k() * kconv;  //-pow(-1.,z<0)*log(tan(atan(1/k)/2.));
      eta = -1. * std::copysign( log(fabs(tan(atan(1 / k) / 2.))),  thetaDigi.z() );
      LogTrace("OMTFReconstruction") << "OmtfPhase2AngleConverter::getGlobalEta(" << dTChamberId << ") eta: " << eta
                                     << " k: " << k << " thetaDigi.k(): " << thetaDigi.k();

      thetaDigiCnt++;
      //checking if the obtained eta has reasonable range - temporary fix
      if ((dTChamberId.station() == 1 && (std::abs(eta) < 0.85 || std::abs(eta) > 1.20)) ||
          (dTChamberId.station() == 2 && (std::abs(eta) < 0.75 || std::abs(eta) > 1.04)) ||
          (dTChamberId.station() == 3 && (std::abs(eta) < 0.63 || std::abs(eta) > 0.92))) {
        foundeta = false;
        /*edm::LogVerbatim("OMTFReconstruction")
            << "OmtfPhase2AngleConverter::getGlobalEta(" << dTChamberId << ") wrong output eta: " << eta << " k: " << k
            << " thetaDigi.k(): " << thetaDigi.k() << " quality " << thetaDigi.quality();*/
      } else
        foundeta = true;
    }
  }

  //if more than 1 thetaDigi per given chamber - we don't use them, as they are ambiguous and we have no way to match them to the phi digis
  if (thetaDigiCnt > 1)
    foundeta = false;

  if (foundeta) {
    //return std::abs(config->etaToHwEta(eta)); TODO use this version
    return std::abs(std::lround(eta * 92));
  } else {
    //Returning eta of the chamber middle
    if (dTChamberId.station() == 1)
      eta = config->mb1W2Eta();
    else if (dTChamberId.station() == 2)
      eta = config->mb2W2Eta();
    else if (dTChamberId.station() == 3)
      eta = config->mb3W2Eta();

    return eta;
  }
  return 95;
}
