#include "L1Trigger/Phase2L1ParticleFlow/interface/deregionizer/deregionizer_ref.h"

#include <cstdio>
#include <vector>

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::DeregionizerEmulator::DeregionizerEmulator(const edm::ParameterSet &iConfig)
    : DeregionizerEmulator(iConfig.getParameter<uint32_t>("nPuppiFinalBuffer"),
                           iConfig.getParameter<uint32_t>("nPuppiPerClk"),
                           iConfig.getParameter<uint32_t>("nPuppiFirstBuffers"),
                           iConfig.getParameter<uint32_t>("nPuppiSecondBuffers"),
                           iConfig.getParameter<uint32_t>("nPuppiThirdBuffers")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}
#endif

l1ct::DeregionizerEmulator::DeregionizerEmulator(const unsigned int nPuppiFinalBuffer /*=128*/,
                                                 const unsigned int nPuppiPerClk /*=6*/,
                                                 const unsigned int nPuppiFirstBuffers /*=12*/,
                                                 const unsigned int nPuppiSecondBuffers /*=32*/,
                                                 const unsigned int nPuppiThirdBuffers /*=64*/)
    : nPuppiFinalBuffer_(nPuppiFinalBuffer),
      nPuppiPerClk_(nPuppiPerClk),
      nPuppiFirstBuffers_(nPuppiFirstBuffers),
      nPuppiSecondBuffers_(nPuppiSecondBuffers),
      nPuppiThirdBuffers_(nPuppiThirdBuffers),
      debug_(false) {
  assert(nPuppiPerClk < nPuppiFirstBuffers && nPuppiFirstBuffers < nPuppiSecondBuffers &&
         nPuppiSecondBuffers < nPuppiThirdBuffers && nPuppiThirdBuffers <= nPuppiFinalBuffer);
}

std::vector<l1ct::PuppiObjEmu> l1ct::DeregionizerEmulator::mergeXtoY(const unsigned int X,
                                                                     const unsigned int Y,
                                                                     const std::vector<l1ct::PuppiObjEmu> &inLeft,
                                                                     const std::vector<l1ct::PuppiObjEmu> &inRight) {
  // merge X to Y with truncation
  std::vector<l1ct::PuppiObjEmu> out;

  out.insert(out.end(), inLeft.begin(), std::min(inLeft.end(), inLeft.begin() + X));
  out.insert(out.end(), inRight.begin(), std::min(inRight.end(), inRight.begin() + Y - X));

  return out;
}

std::vector<l1ct::PuppiObjEmu> l1ct::DeregionizerEmulator::mergeXtoY(const std::vector<l1ct::PuppiObjEmu> &inLeft,
                                                                     const std::vector<l1ct::PuppiObjEmu> &inRight) {
  // merge X to Y with no truncation
  std::vector<l1ct::PuppiObjEmu> out;

  out.insert(out.end(), inLeft.begin(), inLeft.end());
  out.insert(out.end(), inRight.begin(), inRight.end());

  return out;
}

void l1ct::DeregionizerEmulator::accumulateToY(const unsigned int Y,
                                               const std::vector<l1ct::PuppiObjEmu> &in,
                                               std::vector<l1ct::PuppiObjEmu> &out,
                                               std::vector<l1ct::PuppiObjEmu> &truncated) {
  unsigned int initialOutSize = out.size();
  assert(initialOutSize <= Y);
  if (initialOutSize == Y) {
    truncated.insert(truncated.end(), in.begin(), in.end());
    return;
  }
  out.insert(out.end(), in.begin(), std::min(in.end(), in.begin() + Y - initialOutSize));
  if (out.size() == Y)
    truncated.insert(truncated.end(), in.begin() + Y - initialOutSize, in.end());
  return;
}

static void debugPrint(const std::string &header, const std::vector<l1ct::PuppiObjEmu> &pup) {
  dbgCout() << " --> " << header << "\n";
  for (unsigned int iPup = 0, nPup = pup.size(); iPup < nPup; ++iPup)
    dbgCout() << "      > puppi[" << iPup << "] pT = " << pup[iPup].hwPt << "\n";
}

void l1ct::DeregionizerEmulator::run(std::vector<std::vector<std::vector<l1ct::PuppiObjEmu>>> in,
                                     std::vector<l1ct::PuppiObjEmu> &out,
                                     std::vector<l1ct::PuppiObjEmu> &truncated) {
  for (int i = 0, n = in.size(); i < n; i++) {
    std::vector<std::vector<l1ct::PuppiObjEmu>> pupsOnClock = in[i];
    std::vector<l1ct::PuppiObjEmu> intermediateTruncated;
    // Merge PF regions from this cycle. No truncation happens here
    std::vector<l1ct::PuppiObjEmu> buffer;
    for (const auto &pupsOnClockOnBoard : pupsOnClock) {
      buffer = mergeXtoY(buffer, pupsOnClockOnBoard);
    }

    // accumulate PF regions over cycles, truncation may happen here
    accumulateToY(nPuppiFinalBuffer_, buffer, out, truncated);
  }

  if (debug_) {
    dbgCout() << "\n";
    debugPrint("FINAL ARRAY", out);
    dbgCout() << "\n";
    dbgCout() << "Ran successfully!"
              << "\n";
  }
}
