#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

void L1TMuonBarrelParams::print(std::ostream& out) const {

  out << "L1 BMTF Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
/*
  out << "LUT paths (LUTs are generated analytically if path is empty)" << std::endl;
  out << " Abs isolation checkMem LUT path: "        << this->absIsoCheckMemLUTPath() << std::endl;
  out << " Rel isolation checkMem LUT path: "        << this->relIsoCheckMemLUTPath() << std::endl;
  out << " Index selMem phi LUT path: "              << this->idxSelMemPhiLUTPath() << std::endl;
  out << " Index selMem eta LUT path: "              << this->idxSelMemEtaLUTPath() << std::endl;
  out << " Barrel Single MatchQual LUT path: "       << this->brlSingleMatchQualLUTPath() << std::endl;
  out << " Forward pos MatchQual LUT path: "         << this->fwdPosSingleMatchQualLUTPath() << std::endl;
  out << " Forward neg MatchQual LUT path: "         << this->fwdNegSingleMatchQualLUTPath() << std::endl;
  out << " Overlap pos MatchQual LUT path: "         << this->ovlPosSingleMatchQualLUTPath() << std::endl;
  out << " Overlap neg MatchQual LUT path: "         << this->ovlNegSingleMatchQualLUTPath() << std::endl;
  out << " Barrel-Overlap pos MatchQual LUT path: "  << this->bOPosMatchQualLUTPath() << std::endl;
  out << " Barrel-Overlap neg MatchQual LUT path: "  << this->bONegMatchQualLUTPath() << std::endl;
  out << " Forward-Overlap pos MatchQual LUT path: " << this->fOPosMatchQualLUTPath() << std::endl;
  out << " Forward-Overlap neg MatchQual LUT path: " << this->fONegMatchQualLUTPath() << std::endl;
  out << " Barrel phi extrapolation LUT path: "      << this->bPhiExtrapolationLUTPath() << std::endl;
  out << " Overlap phi extrapolation LUT path: "     << this->oPhiExtrapolationLUTPath() << std::endl;
  out << " Forward phi extrapolation LUT path: "     << this->fPhiExtrapolationLUTPath() << std::endl;
  out << " Barrel eta extrapolation LUT path: "      << this->bEtaExtrapolationLUTPath() << std::endl;
  out << " Overlap eta extrapolation LUT path: "     << this->oEtaExtrapolationLUTPath() << std::endl;
  out << " Forward eta extrapolation LUT path: "     << this->fEtaExtrapolationLUTPath() << std::endl;
  out << " Sort rank LUT path: "                     << this->sortRankLUTPath() << std::endl;
*/
}
