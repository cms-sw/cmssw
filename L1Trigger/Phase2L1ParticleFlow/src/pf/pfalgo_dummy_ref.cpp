#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_dummy_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::PFAlgoDummyEmulator::PFAlgoDummyEmulator(const edm::ParameterSet& iConfig)
    : PFAlgoEmulatorBase(
          0, iConfig.getParameter<uint32_t>("nCalo"), iConfig.getParameter<uint32_t>("nMu"), 0, 0, 0, 0, 0) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}

edm::ParameterSetDescription l1ct::PFAlgoDummyEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<unsigned int>("nCalo");
  description.add<unsigned int>("nMu");
  description.addUntracked<bool>("debug", false);
  return description;
}
#endif

void l1ct::PFAlgoDummyEmulator::run(const PFInputRegion& in, OutputRegion& out) const {
  unsigned int nCALO = std::min<unsigned>(nCALO_, in.hadcalo.size());
  unsigned int nMU = std::min<unsigned>(nMU_, in.muon.size());

  if (debug_) {
    for (unsigned int i = 0; i < nCALO; ++i) {
      if (in.hadcalo[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t calo  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+7d ]  calo phi %+5.2f [ %+7d ]  calo emPt %8.2f [ "
          "%6d ]   emID %2d \n",
          i,
          in.hadcalo[i].floatPt(),
          in.hadcalo[i].intPt(),
          in.hadcalo[i].floatEta(),
          in.hadcalo[i].intEta(),
          in.hadcalo[i].floatPhi(),
          in.hadcalo[i].intPhi(),
          in.hadcalo[i].floatEmPt(),
          in.hadcalo[i].intEmPt(),
          in.hadcalo[i].hwEmID.to_int());
    }
    for (unsigned int i = 0; i < nMU; ++i) {
      if (in.muon[i].hwPt == 0)
        continue;
      dbgPrintf("FW  \t muon  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+7d ]  calo phi %+5.2f [ %+7d ]   \n",
                i,
                in.muon[i].floatPt(),
                in.muon[i].intPt(),
                in.muon[i].floatEta(),
                in.muon[i].intEta(),
                in.muon[i].floatPhi(),
                in.muon[i].intPhi());
    }
  }

  out.pfneutral.resize(nCALO);
  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    if (in.hadcalo[ic].hwPt > 0) {
      fillPFCand(in.hadcalo[ic], out.pfneutral[ic], in.hadcalo[ic].hwIsEM());
    } else {
      out.pfneutral[ic].clear();
    }
  }

  if (debug_) {
    for (unsigned int i = 0; i < nCALO; ++i) {
      if (out.pfneutral[i].hwPt == 0)
        continue;
      dbgPrintf("FW  \t outne %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+7d ]  calo phi %+5.2f [ %+7d ]  pid %d\n",
                i,
                out.pfneutral[i].floatPt(),
                out.pfneutral[i].intPt(),
                out.pfneutral[i].floatEta(),
                out.pfneutral[i].intEta(),
                out.pfneutral[i].floatPhi(),
                out.pfneutral[i].intPhi(),
                out.pfneutral[i].intId());
    }
  }
}
