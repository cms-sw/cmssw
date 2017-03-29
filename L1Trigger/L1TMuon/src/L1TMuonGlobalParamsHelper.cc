#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"

L1TMuonGlobalParamsHelper::L1TMuonGlobalParamsHelper(const L1TMuonGlobalParams & p) : L1TMuonGlobalParams_PUBLIC(cast_to_L1TMuonGlobalParams_PUBLIC(p))
{
  if (pnodes_.size() != NUM_GMTPARAMNODES) {
    pnodes_.resize(NUM_GMTPARAMNODES);
  }
}


std::bitset<72> L1TMuonGlobalParamsHelper::inputFlags(const int &nodeIdx) const
{
  std::bitset<72> inputFlags;
  if (pnodes_[nodeIdx].uparams_.size() != 4) {
    return inputFlags;
  }

  for (size_t i = 0; i < 28; ++i) {
    inputFlags[CALOLINK1 + i] = ((pnodes_[nodeIdx].uparams_[CALOINPUTS] >> i) & 0x1);
    if (i < CALOLINK1) {
      // disable unused inputs
      inputFlags[i] = 0x1;
    }
    if (i < 12) {
      inputFlags[BMTFLINK1 + i] = ((pnodes_[nodeIdx].uparams_[BMTFINPUTS] >> i) & 0x1);
      if (i < 6) {
        inputFlags[EMTFPLINK1 + i] = ((pnodes_[nodeIdx].uparams_[EMTFINPUTS] >> i) & 0x1);
        inputFlags[OMTFPLINK1 + i] = ((pnodes_[nodeIdx].uparams_[OMTFINPUTS] >> i) & 0x1);
        inputFlags[OMTFNLINK1 + i] = ((pnodes_[nodeIdx].uparams_[OMTFINPUTS] >> (i + 6)) & 0x1);
        inputFlags[EMTFNLINK1 + i] = ((pnodes_[nodeIdx].uparams_[EMTFINPUTS] >> (i + 6)) & 0x1);
      }
    }
  }
  return inputFlags;
}


std::bitset<28> L1TMuonGlobalParamsHelper::caloInputFlags(const int &nodeIdx) const
{
  if (pnodes_[nodeIdx].uparams_.size() == 4) {
    return std::bitset<28>(pnodes_[nodeIdx].uparams_[CALOINPUTS]);
  } else {
    return std::bitset<28>();
  }
}


std::bitset<12> L1TMuonGlobalParamsHelper::tfInputFlags(const int &nodeIdx, const int &tfIdx) const
{
  if (pnodes_[nodeIdx].uparams_.size() == 4) {
    return std::bitset<12>(pnodes_[nodeIdx].uparams_[tfIdx]);
  } else {
    return std::bitset<12>();
  }
}


std::bitset<6> L1TMuonGlobalParamsHelper::eomtfInputFlags(const int &nodeIdx, const size_t &startIdx, const int &tfIdx) const
{
  std::bitset<6> inputFlags;
  if (pnodes_[nodeIdx].uparams_.size() == 4) {
    for (size_t i = 0; i < 6; ++i) {
      inputFlags[i] = ((pnodes_[nodeIdx].uparams_[tfIdx] >> (i + startIdx)) & 0x1);
    }
  }
  return inputFlags;
}


void L1TMuonGlobalParamsHelper::setFwVersion(unsigned fwVersion)
{
  pnodes_[FWVERSION].uparams_.resize(1);
  pnodes_[FWVERSION].uparams_[FWVERSION_IDX] = fwVersion;
}


void L1TMuonGlobalParamsHelper::setInputFlags(const int &nodeIdx, const std::bitset<72> &inputFlags)
{
  pnodes_[nodeIdx].uparams_.resize(4);
  for (size_t i = 0; i < 28; ++i) {
    pnodes_[nodeIdx].uparams_[CALOINPUTS] += (inputFlags.test(CALOLINK1 + i) << i);
    if (i < 12) {
      pnodes_[nodeIdx].uparams_[BMTFINPUTS] += (inputFlags.test(BMTFLINK1 + i) << i);
      if (i < 6) {
        pnodes_[nodeIdx].uparams_[OMTFINPUTS] += (inputFlags.test(OMTFPLINK1 + i) << i);
        pnodes_[nodeIdx].uparams_[OMTFINPUTS] += (inputFlags.test(OMTFNLINK1 + i) << (i + 6));
        pnodes_[nodeIdx].uparams_[EMTFINPUTS] += (inputFlags.test(EMTFPLINK1 + i) << i);
        pnodes_[nodeIdx].uparams_[EMTFINPUTS] += (inputFlags.test(EMTFNLINK1 + i) << (i + 6));
      }
    }
  }
}


void L1TMuonGlobalParamsHelper::setCaloInputFlags(const int &nodeIdx, const std::bitset<28> &inputFlags)
{
  pnodes_[nodeIdx].uparams_.resize(4);
  for (size_t i = 0; i < 28; ++i) {
    pnodes_[nodeIdx].uparams_[CALOINPUTS] += (inputFlags.test(i) << i);
  }
}


void L1TMuonGlobalParamsHelper::setTfInputFlags(const int &nodeIdx, const int &tfIdx, const std::bitset<12> &inputFlags)
{
  pnodes_[nodeIdx].uparams_.resize(4);
  for (size_t i = 0; i < 12; ++i) {
    pnodes_[nodeIdx].uparams_[tfIdx] += (inputFlags.test(i) << i);
  }
}


void L1TMuonGlobalParamsHelper::setEOmtfInputFlags(const int &nodeIdx, const size_t &startIdx, const int &tfIdx, const std::bitset<6> &inputFlags)
{
  pnodes_[nodeIdx].uparams_.resize(4);
  for (size_t i = 0; i < 6; ++i) {
    pnodes_[nodeIdx].uparams_[tfIdx] += (inputFlags.test(i) << (i + startIdx));
  }
}


// setters for cancel out LUT parameters
void L1TMuonGlobalParamsHelper::setFwdPosSingleMatchQualLUTMaxDR (double maxDR, double fEta, double fPhi)
{
  pnodes_[fwdPosSingleMatchQual].dparams_.push_back(maxDR);
  pnodes_[fwdPosSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[fwdPosSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[fwdPosSingleMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setFwdNegSingleMatchQualLUTMaxDR (double maxDR, double fEta, double fPhi)
{
  pnodes_[fwdNegSingleMatchQual].dparams_.push_back(maxDR);
  pnodes_[fwdNegSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[fwdNegSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[fwdNegSingleMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setOvlPosSingleMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[ovlPosSingleMatchQual].dparams_.push_back(maxDR);
  pnodes_[ovlPosSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[ovlPosSingleMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[ovlPosSingleMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setOvlNegSingleMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[ovlNegSingleMatchQual].dparams_.push_back(maxDR);
  pnodes_[ovlNegSingleMatchQual].dparams_.push_back(fEta);
  pnodes_[ovlNegSingleMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[ovlNegSingleMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setBOPosMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[bOPosMatchQual].dparams_.push_back(maxDR);
  pnodes_[bOPosMatchQual].dparams_.push_back(fEta);
  pnodes_[bOPosMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[bOPosMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setBONegMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[bONegMatchQual].dparams_.push_back(maxDR);
  pnodes_[bONegMatchQual].dparams_.push_back(fEta);
  pnodes_[bONegMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[bONegMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setFOPosMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[fOPosMatchQual].dparams_.push_back(maxDR);
  pnodes_[fOPosMatchQual].dparams_.push_back(fEta);
  pnodes_[fOPosMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[fOPosMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::setFONegMatchQualLUTMaxDR (double maxDR, double fEta, double fEtaCoarse, double fPhi)
{
  pnodes_[fONegMatchQual].dparams_.push_back(maxDR);
  pnodes_[fONegMatchQual].dparams_.push_back(fEta);
  pnodes_[fONegMatchQual].dparams_.push_back(fEtaCoarse);
  pnodes_[fONegMatchQual].dparams_.push_back(fPhi);
}


void L1TMuonGlobalParamsHelper::print(std::ostream& out) const {

  out << "L1 MicroGMT Parameters" << std::endl;

  out << "Firmware version: " << this->fwVersion() << std::endl;

  out << "InputsToDisable: " << this->inputsToDisable() << std::endl;
  out << "                 EMTF-|OMTF-|   BMTF    |OMTF+|EMTF+|            CALO           |  res  0" << std::endl;

  out << "Masked Inputs:   " << this->maskedInputs() << std::endl;
  out << "                 EMTF-|OMTF-|   BMTF    |OMTF+|EMTF+|            CALO           |  res  0" << std::endl;

  out << "LUT paths (LUTs are generated analytically if path is empty)" << std::endl;
  out << " Abs isolation checkMem LUT path: "        << this->absIsoCheckMemLUTPath() << std::endl;
  out << " Rel isolation checkMem LUT path: "        << this->relIsoCheckMemLUTPath() << std::endl;
  out << " Index selMem phi LUT path: "              << this->idxSelMemPhiLUTPath() << std::endl;
  out << " Index selMem eta LUT path: "              << this->idxSelMemEtaLUTPath() << std::endl;
  out << " Forward pos MatchQual LUT path: "         << this->fwdPosSingleMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->fwdPosSingleMatchQualLUTMaxDR() << std::endl;
  out << " Forward neg MatchQual LUT path: "         << this->fwdNegSingleMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->fwdNegSingleMatchQualLUTMaxDR() << std::endl;
  out << " Overlap pos MatchQual LUT path: "         << this->ovlPosSingleMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->ovlPosSingleMatchQualLUTMaxDR() << std::endl;
  out << " Overlap neg MatchQual LUT path: "         << this->ovlNegSingleMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->ovlNegSingleMatchQualLUTMaxDR() << std::endl;
  out << " Barrel-Overlap pos MatchQual LUT path: "  << this->bOPosMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->bOPosMatchQualLUTMaxDR() << ", fEta: " << this->bOPosMatchQualLUTfEta() << ", fEta when eta-fine bit isn't set: " << this->bOPosMatchQualLUTfEtaCoarse() << ", fPhi: " << this->bOPosMatchQualLUTfEta() << std::endl;
  out << " Barrel-Overlap neg MatchQual LUT path: "  << this->bONegMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->bONegMatchQualLUTMaxDR() << ", fEta: " << this->bONegMatchQualLUTfEta() << ", fEta when eta-fine bit isn't set: " << this->bONegMatchQualLUTfEtaCoarse() << ", fPhi: " << this->bONegMatchQualLUTfPhi() << std::endl;
  out << " Forward-Overlap pos MatchQual LUT path: " << this->fOPosMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->fOPosMatchQualLUTMaxDR() << std::endl;
  out << " Forward-Overlap neg MatchQual LUT path: " << this->fONegMatchQualLUTPath() << ", max dR (Used when LUT path empty): " << this->fONegMatchQualLUTMaxDR() << std::endl;
  out << " Barrel phi extrapolation LUT path: "      << this->bPhiExtrapolationLUTPath() << std::endl;
  out << " Overlap phi extrapolation LUT path: "     << this->oPhiExtrapolationLUTPath() << std::endl;
  out << " Forward phi extrapolation LUT path: "     << this->fPhiExtrapolationLUTPath() << std::endl;
  out << " Barrel eta extrapolation LUT path: "      << this->bEtaExtrapolationLUTPath() << std::endl;
  out << " Overlap eta extrapolation LUT path: "     << this->oEtaExtrapolationLUTPath() << std::endl;
  out << " Forward eta extrapolation LUT path: "     << this->fEtaExtrapolationLUTPath() << std::endl;
  out << " Sort rank LUT path: "                     << this->sortRankLUTPath() << ", pT and quality factors (Used when LUT path empty): pT factor: " << this->sortRankLUTPtFactor() << ", quality factor: " << this->sortRankLUTQualFactor() << std::endl;
}
