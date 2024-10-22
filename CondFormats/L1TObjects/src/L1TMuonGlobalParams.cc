#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

void L1TMuonGlobalParams::print(std::ostream& out) const {
  out << "L1 MicroGMT Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;

  out << "Output BX range from " << bxMin_ << " to " << bxMax_ << std::endl;

  out << "LUT paths (LUTs are generated analytically if path is empty)" << std::endl;
  out << " Abs isolation checkMem LUT path: " << this->absIsoCheckMemLUTPath() << std::endl;
  out << " Rel isolation checkMem LUT path: " << this->relIsoCheckMemLUTPath() << std::endl;
  out << " Index selMem phi LUT path: " << this->idxSelMemPhiLUTPath() << std::endl;
  out << " Index selMem eta LUT path: " << this->idxSelMemEtaLUTPath() << std::endl;
  //out << " Barrel Single MatchQual LUT path: "       << this->brlSingleMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->brlSingleMatchQualLUTMaxDR() << std::endl;
  out << " Forward pos MatchQual LUT path: " << this->fwdPosSingleMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->fwdPosSingleMatchQualLUTMaxDR() << std::endl;
  out << " Forward neg MatchQual LUT path: " << this->fwdNegSingleMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->fwdNegSingleMatchQualLUTMaxDR() << std::endl;
  out << " Overlap pos MatchQual LUT path: " << this->ovlPosSingleMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->ovlPosSingleMatchQualLUTMaxDR() << std::endl;
  out << " Overlap neg MatchQual LUT path: " << this->ovlNegSingleMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->ovlNegSingleMatchQualLUTMaxDR() << std::endl;
  out << " Barrel-Overlap pos MatchQual LUT path: " << this->bOPosMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->bOPosMatchQualLUTMaxDR()
      << ", max dR when eta-fine bit set: " << this->bOPosMatchQualLUTMaxDREtaFine() << std::endl;
  out << " Barrel-Overlap neg MatchQual LUT path: " << this->bONegMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->bONegMatchQualLUTMaxDR()
      << ", max dR when eta-fine bit set: " << this->bONegMatchQualLUTMaxDREtaFine() << std::endl;
  out << " Forward-Overlap pos MatchQual LUT path: " << this->fOPosMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->fOPosMatchQualLUTMaxDR() << std::endl;
  out << " Forward-Overlap neg MatchQual LUT path: " << this->fONegMatchQualLUTPath()
      << ", max dR (Used when LUT path empty): " << this->fONegMatchQualLUTMaxDR() << std::endl;
  out << " Barrel phi extrapolation LUT path: " << this->bPhiExtrapolationLUTPath() << std::endl;
  out << " Overlap phi extrapolation LUT path: " << this->oPhiExtrapolationLUTPath() << std::endl;
  out << " Forward phi extrapolation LUT path: " << this->fPhiExtrapolationLUTPath() << std::endl;
  out << " Barrel eta extrapolation LUT path: " << this->bEtaExtrapolationLUTPath() << std::endl;
  out << " Overlap eta extrapolation LUT path: " << this->oEtaExtrapolationLUTPath() << std::endl;
  out << " Forward eta extrapolation LUT path: " << this->fEtaExtrapolationLUTPath() << std::endl;
  out << " Sort rank LUT path: " << this->sortRankLUTPath()
      << ", pT and quality factors (Used when LUT path empty): pT factor: " << this->sortRankLUTPtFactor()
      << ", quality factor: " << this->sortRankLUTQualFactor() << std::endl;
}
