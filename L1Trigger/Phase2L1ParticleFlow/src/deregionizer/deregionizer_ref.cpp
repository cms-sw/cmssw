#include "L1Trigger/Phase2L1ParticleFlow/interface/deregionizer/deregionizer_ref.h"

#include <cstdio>
#include <vector>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

l1ct::DeregionizerEmulator::DeregionizerEmulator(const edm::ParameterSet &iConfig)
    : DeregionizerEmulator(iConfig.getParameter<uint32_t>("nPuppiFinalBuffer"),
                           iConfig.getParameter<uint32_t>("nPuppiPerClk"),
                           iConfig.getParameter<uint32_t>("nPuppiFirstBuffers"),
                           iConfig.getParameter<uint32_t>("nPuppiSecondBuffers"),
                           iConfig.getParameter<uint32_t>("nPuppiThirdBuffers")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}
#else
#include "../../utils/dbgPrintf.h"
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

std::vector<std::vector<l1ct::PuppiObjEmu> > l1ct::DeregionizerEmulator::splitPFregions(
    const std::vector<std::vector<std::vector<l1ct::PuppiObjEmu> > > &regionPuppis, const int i, const int j) {
  int k = nPuppiPerClk_ * j;
  std::vector<std::vector<l1ct::PuppiObjEmu> > subregionPuppis;
  for (int l = 0, n = regionPuppis.size(); l < n; l++) {
    const auto &puppis = regionPuppis[l][i];
    std::vector<l1ct::PuppiObjEmu> tmp(std::min(puppis.begin() + k, puppis.end()),
                                       std::min(puppis.begin() + k + nPuppiPerClk_, puppis.end()));
    subregionPuppis.push_back(tmp);
  }
  return subregionPuppis;
}

std::vector<l1ct::PuppiObjEmu> l1ct::DeregionizerEmulator::mergeXtoY(const unsigned int X,
                                                                     const unsigned int Y,
                                                                     const std::vector<l1ct::PuppiObjEmu> &inLeft,
                                                                     const std::vector<l1ct::PuppiObjEmu> &inRight) {
  std::vector<l1ct::PuppiObjEmu> out;

  out.insert(out.end(), inLeft.begin(), std::min(inLeft.end(), inLeft.begin() + X));
  out.insert(out.end(), inRight.begin(), std::min(inRight.end(), inRight.begin() + Y - X));

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

void l1ct::DeregionizerEmulator::run(const l1ct::DeregionizerInput in,
                                     std::vector<l1ct::PuppiObjEmu> &out,
                                     std::vector<l1ct::PuppiObjEmu> &truncated) {
  const auto &regionPuppis = in.orderedInRegionsPuppis();
  std::vector<l1ct::PuppiObjEmu> intermediateTruncated;

  for (int i = 0, n = in.nPhiRegions; i < n; i++) {
    // Each PF region (containing at most 18 puppi candidates) is split in 3(*nPuppiPerClk=18)
    for (int j = 0; j < 3; j++) {
      std::vector<std::vector<l1ct::PuppiObjEmu> > subregionPuppis = splitPFregions(regionPuppis, i, j);

      // Merge PF regions in pairs
      std::vector<l1ct::PuppiObjEmu> buffer01 =
          mergeXtoY(nPuppiPerClk_, nPuppiFirstBuffers_, subregionPuppis[0], subregionPuppis[1]);
      std::vector<l1ct::PuppiObjEmu> buffer23 =
          mergeXtoY(nPuppiPerClk_, nPuppiFirstBuffers_, subregionPuppis[2], subregionPuppis[3]);
      std::vector<l1ct::PuppiObjEmu> buffer45 =
          mergeXtoY(nPuppiPerClk_, nPuppiFirstBuffers_, subregionPuppis[4], subregionPuppis[5]);

      // Merge 4 first regions together, forward the last 2
      std::vector<l1ct::PuppiObjEmu> buffer0123 =
          mergeXtoY(nPuppiFirstBuffers_, nPuppiSecondBuffers_, buffer01, buffer23);
      std::vector<l1ct::PuppiObjEmu> buffer45ext;
      accumulateToY(nPuppiSecondBuffers_, buffer45, buffer45ext, intermediateTruncated);

      // Merge all regions together and forward them to the final buffer
      std::vector<l1ct::PuppiObjEmu> buffer012345 =
          mergeXtoY(nPuppiSecondBuffers_, nPuppiThirdBuffers_, buffer0123, buffer45ext);
      accumulateToY(nPuppiFinalBuffer_, buffer012345, out, truncated);

      if (debug_) {
        dbgCout() << "\n";
        dbgCout() << "Phi region index : " << i << "," << j << "\n";

        debugPrint("Eta region : 0", subregionPuppis[0]);
        debugPrint("Eta region : 1", subregionPuppis[1]);
        debugPrint("Eta region : 0+1", buffer01);
        dbgCout() << "------------------ "
                  << "\n";

        debugPrint("Eta region : 2", subregionPuppis[2]);
        debugPrint("Eta region : 3", subregionPuppis[3]);
        debugPrint("Eta region : 2+3", buffer23);
        dbgCout() << "------------------ "
                  << "\n";

        debugPrint("Eta region : 4", subregionPuppis[4]);
        debugPrint("Eta region : 5", subregionPuppis[5]);
        debugPrint("Eta region : 4+5", buffer45);
        dbgCout() << "------------------ "
                  << "\n";

        debugPrint("Eta region : 0+1+2+3", buffer0123);
        dbgCout() << "------------------ "
                  << "\n";

        debugPrint("Eta region : 0+1+2+3+4+5", buffer012345);
        dbgCout() << "------------------ "
                  << "\n";

        debugPrint("Inclusive", out);
      }
    }
  }

  if (debug_) {
    dbgCout() << "\n";
    debugPrint("FINAL ARRAY", out);
    dbgCout() << "\n";
    dbgCout() << "Ran successfully!"
              << "\n";
  }
}
