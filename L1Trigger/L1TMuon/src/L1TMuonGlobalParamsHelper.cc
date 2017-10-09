#include <iomanip>
#include <strstream>

#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TCommon/interface/ConvertToLUT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      inputFlags[i] = true;
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


void L1TMuonGlobalParamsHelper::loadFromOnline(l1t::TriggerSystem& trgSys, const std::string& processorId)
{
  std::string procId = processorId;
  // if the procId is an empty string use the one from the TrigSystem (the uGMT only has one processor)
  if (procId == "" ) {
    const std::map<std::string, std::string>& procRoleMap = trgSys.getProcToRoleAssignment();
    if (procRoleMap.size() != 1) {
      if (procRoleMap.empty()) {
        edm::LogError("uGMT config from online") << "No processor id found for uGMT HW configuration.";
      } else {
        edm::LogError("uGMT config from online") << "More than one processor id found for uGMT HW configuration.";
      }
    } else {
      procId = procRoleMap.cbegin()->first;
    }
  }

  // get the settings and masks for the processor id
  std::map<std::string, l1t::Parameter> settings = trgSys.getParameters(procId.c_str());
  //std::map<std::string, l1t::Mask> masks = trgSys.getMasks(procId.c_str());
  //for (auto& it: settings) {
  //   std::cout << "Key: " << it.first << ", procRole: " << it.second.getProcRole() << ", type: " << it.second.getType() << ", id: " << it.second.getId() << ", value as string: [" << it.second.getValueAsStr() << "]" << std::endl;
  //}
  //for (auto& it: masks) {
  //   std::cout << "Key: " << it.first << ", procRole: " << it.second.getProcRole() << ", id: " << it.second.getId() << std::endl;
  //}

  // Use FW version from online config if it is found there. Otherwise set it to 1
  unsigned fwVersion = 1;
  if (settings.count("algoRev") > 0) {
    fwVersion = settings["algoRev"].getValue<unsigned int>();
  }
  setFwVersion(fwVersion);

  std::stringstream ss;
  // uGMT disabled inputs
  bool disableCaloInputs = settings["caloInputsDisable"].getValue<bool>();
  std::string bmtfInputsToDisableStr = settings["bmtfInputsToDisable"].getValueAsStr();
  std::string omtfInputsToDisableStr = settings["omtfInputsToDisable"].getValueAsStr();
  std::string emtfInputsToDisableStr = settings["emtfInputsToDisable"].getValueAsStr();
  std::vector<unsigned> bmtfInputsToDisable(12, 0);
  std::vector<unsigned> omtfInputsToDisable(12, 0);
  std::vector<unsigned> emtfInputsToDisable(12, 0);
  // translate the bool and the strings to the vectors
  for (unsigned i = 0; i < 12; ++i) {
     ss.str("");
     ss << "BMTF" << i+1;
     if (bmtfInputsToDisableStr.find(ss.str()) != std::string::npos) {
        bmtfInputsToDisable[i] = 1;
     }
     ss.str("");
     ss << "OMTF";
     if (i < 6) {
        ss << "p" << i+1;
     } else {
        ss << "n" << i-5;
     }
     if (omtfInputsToDisableStr.find(ss.str()) != std::string::npos) {
        omtfInputsToDisable[i] = 1;
     }
     ss.str("");
     ss << "EMTF";
     if (i < 6) {
        ss << "p" << i+1;
     } else {
        ss << "n" << i-5;
     }
     if (emtfInputsToDisableStr.find(ss.str()) != std::string::npos) {
        emtfInputsToDisable[i] = 1;
     }
  }

  // set the condFormats parameters for uGMT disabled inputs
  if (disableCaloInputs) {
     setCaloInputsToDisable(std::bitset<28>(0xFFFFFFF));
  } else {
     setCaloInputsToDisable(std::bitset<28>());
  }

  std::bitset<12> bmtfDisables;
  for (size_t i = 0; i < bmtfInputsToDisable.size(); ++i) {
    bmtfDisables.set(i, bmtfInputsToDisable[i] > 0);
  }
  setBmtfInputsToDisable(bmtfDisables);

  std::bitset<6> omtfpDisables;
  std::bitset<6> omtfnDisables;
  for (size_t i = 0; i < omtfInputsToDisable.size(); ++i) {
    if (i < 6) {
      omtfpDisables.set(i, omtfInputsToDisable[i] > 0);
    } else {
      omtfnDisables.set(i-6, omtfInputsToDisable[i] > 0);
    }
  }
  setOmtfpInputsToDisable(omtfpDisables);
  setOmtfnInputsToDisable(omtfnDisables);

  std::bitset<6> emtfpDisables;
  std::bitset<6> emtfnDisables;
  for (size_t i = 0; i < emtfInputsToDisable.size(); ++i) {
    if (i < 6) {
      emtfpDisables.set(i, emtfInputsToDisable[i] > 0);
    } else {
      emtfnDisables.set(i-6, emtfInputsToDisable[i] > 0);
    }
  }
  setEmtfpInputsToDisable(emtfpDisables);
  setEmtfnInputsToDisable(emtfnDisables);

  // uGMT masked inputs
  bool caloInputsMasked = true;
  std::vector<unsigned> maskedBmtfInputs(12, 0);
  std::vector<unsigned> maskedOmtfInputs(12, 0);
  std::vector<unsigned> maskedEmtfInputs(12, 0);
  ss << std::setfill('0');
  // translate the bool and the strings to the vectors
  for (unsigned i = 0; i < 28; ++i) {
     ss.str("");
     ss << "inputPorts.CaloL2_" << std::setw(2) << i+1;
     // for now set as unmasked if one input is not masked
     if (!trgSys.isMasked(procId.c_str(), ss.str().c_str())) {
        caloInputsMasked = false;
     }
     if (i < 12) {
        ss.str("");
        ss << "inputPorts.BMTF_" << std::setw(2) << i+1;
        if (trgSys.isMasked(procId.c_str(), ss.str().c_str())) {
           maskedBmtfInputs[i] = 1;
        }
        ss.str("");
        ss << "inputPorts.OMTF";
        if (i < 6) {
           ss << "+_" << std::setw(2) << i+1;
        } else {
           ss << "-_" << std::setw(2) << i-5;
        }
        if (trgSys.isMasked(procId.c_str(), ss.str().c_str())) {
           maskedOmtfInputs[i] = 1;
        }
        ss.str("");
        ss << "inputPorts.EMTF";
        if (i < 6) {
           ss << "+_" << std::setw(2) << i+1;
        } else {
           ss << "-_" << std::setw(2) << i-5;
        }
        if (trgSys.isMasked(procId.c_str(), ss.str().c_str())) {
           maskedEmtfInputs[i] = 1;
        }
     }
  }
  ss << std::setfill(' ');

  // set the condFormats parameters for uGMT masked inputs
  if (caloInputsMasked) {
     setMaskedCaloInputs(std::bitset<28>(0xFFFFFFF));
  } else {
     setMaskedCaloInputs(std::bitset<28>());
  }

  std::bitset<12> bmtfMasked;
  for (size_t i = 0; i < maskedBmtfInputs.size(); ++i) {
    bmtfMasked.set(i, maskedBmtfInputs[i] > 0);
  }
  setMaskedBmtfInputs(bmtfMasked);

  std::bitset<6> omtfpMasked;
  std::bitset<6> omtfnMasked;
  for (size_t i = 0; i < maskedOmtfInputs.size(); ++i) {
    if (i < 6) {
      omtfpMasked.set(i, maskedOmtfInputs[i] > 0);
    } else {
      omtfnMasked.set(i-6, maskedOmtfInputs[i] > 0);
    }
  }
  setMaskedOmtfpInputs(omtfpMasked);
  setMaskedOmtfnInputs(omtfnMasked);

  std::bitset<6> emtfpMasked;
  std::bitset<6> emtfnMasked;
  for (size_t i = 0; i < maskedEmtfInputs.size(); ++i) {
    if (i < 6) {
      emtfpMasked.set(i, maskedEmtfInputs[i] > 0);
    } else {
      emtfnMasked.set(i-6, maskedEmtfInputs[i] > 0);
    }
  }
  setMaskedEmtfpInputs(emtfpMasked);
  setMaskedEmtfnInputs(emtfnMasked);

  // LUTs from settings with with automatic detection of address width and 31 bit output width
  setAbsIsoCheckMemLUT(l1t::convertToLUT(settings["AbsIsoCheckMem"].getVector<unsigned int>()));
  setRelIsoCheckMemLUT(l1t::convertToLUT(settings["RelIsoCheckMem"].getVector<unsigned int>()));
  setIdxSelMemPhiLUT(l1t::convertToLUT(settings["IdxSelMemPhi"].getVector<unsigned int>()));
  setIdxSelMemEtaLUT(l1t::convertToLUT(settings["IdxSelMemEta"].getVector<unsigned int>()));
  setFwdPosSingleMatchQualLUT(l1t::convertToLUT(settings["EmtfPosSingleMatchQual"].getVector<unsigned int>()));
  setFwdNegSingleMatchQualLUT(l1t::convertToLUT(settings["EmtfNegSingleMatchQual"].getVector<unsigned int>()));
  setOvlPosSingleMatchQualLUT(l1t::convertToLUT(settings["OmtfPosSingleMatchQual"].getVector<unsigned int>()));
  setOvlNegSingleMatchQualLUT(l1t::convertToLUT(settings["OmtfNegSingleMatchQual"].getVector<unsigned int>()));
  setBOPosMatchQualLUT(l1t::convertToLUT(settings["BOPosMatchQual"].getVector<unsigned int>()));
  setBONegMatchQualLUT(l1t::convertToLUT(settings["BONegMatchQual"].getVector<unsigned int>()));
  setFOPosMatchQualLUT(l1t::convertToLUT(settings["EOPosMatchQual"].getVector<unsigned int>()));
  setFONegMatchQualLUT(l1t::convertToLUT(settings["EONegMatchQual"].getVector<unsigned int>()));
  setBPhiExtrapolationLUT(l1t::convertToLUT(settings["BPhiExtrapolation"].getVector<unsigned int>()));
  setOPhiExtrapolationLUT(l1t::convertToLUT(settings["OPhiExtrapolation"].getVector<unsigned int>()));
  setFPhiExtrapolationLUT(l1t::convertToLUT(settings["EPhiExtrapolation"].getVector<unsigned int>()));
  setBEtaExtrapolationLUT(l1t::convertToLUT(settings["BEtaExtrapolation"].getVector<unsigned int>()));
  setOEtaExtrapolationLUT(l1t::convertToLUT(settings["OEtaExtrapolation"].getVector<unsigned int>()));
  setFEtaExtrapolationLUT(l1t::convertToLUT(settings["EEtaExtrapolation"].getVector<unsigned int>()));
  setSortRankLUT(l1t::convertToLUT(settings["SortRank"].getVector<unsigned int>()));
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
