#include <vector>
#include "L1Trigger/Phase2L1ParticleFlow/interface/deregionizer/deregionizer_input.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
l1ct::DeregionizerInput::DeregionizerInput(const std::vector<edm::ParameterSet> linkConfigs) {
  for (const auto &pset : linkConfigs) {
    DeregionizerInput::BoardInfo boardInfo;
    boardInfo.nOutputFramesPerBX_ = pset.getParameter<uint32_t>("nOutputFramesPerBX");
    boardInfo.nLinksPuppi_ = pset.getParameter<uint32_t>("nLinksPuppi");
    boardInfo.nPuppiPerRegion_ = pset.getParameter<uint32_t>("nPuppiPerRegion");
    boardInfo.order_ = pset.getParameter<int32_t>("outputBoard");
    boardInfo.regions_ = pset.getParameter<std::vector<uint32_t>>("outputRegions");
    boardInfo.nPuppiFramesPerRegion_ = (boardInfo.nOutputFramesPerBX_ * tmuxFactor_) / boardInfo.regions_.size();
    boardInfos_.push_back(boardInfo);
  }
}
#endif

std::vector<l1ct::DeregionizerInput::PlacedPuppi> l1ct::DeregionizerInput::inputOrderInfo(
    const std::vector<l1ct::OutputRegion> &inputRegions) const {
  // Vector of all puppis in event paired with LinkPlacementInfo
  std::vector<PlacedPuppi> linkPlacedPuppis;
  for (BoardInfo boardInfo : boardInfos_) {
    for (uint iRegion = 0; iRegion < boardInfo.regions_.size(); iRegion++) {
      uint iRegionEvent = boardInfo.regions_.at(iRegion);
      auto puppi = inputRegions.at(iRegionEvent).puppi;
      unsigned int npuppi = puppi.size();
      for (unsigned int i = 0; i < boardInfo.nLinksPuppi_ * boardInfo.nPuppiFramesPerRegion_; ++i) {
        if (i < npuppi) {
          uint iClock =
              iRegion * boardInfo.nPuppiPerRegion_ / boardInfo.nLinksPuppi_ + i % boardInfo.nPuppiFramesPerRegion_;
          uint iLink = i / boardInfo.nPuppiFramesPerRegion_;
          LPI lpi = {boardInfo.order_, iLink, iClock};
          linkPlacedPuppis.push_back(std::make_pair(puppi.at(i), lpi));
        }
      }
    }
  }
  return linkPlacedPuppis;
}

std::vector<std::vector<std::vector<l1ct::PuppiObjEmu>>> l1ct::DeregionizerInput::orderInputs(
    const std::vector<l1ct::OutputRegion> &inputRegions) const {
  std::vector<PlacedPuppi> linkPlacedPuppis = inputOrderInfo(inputRegions);
  std::vector<std::vector<std::vector<l1ct::PuppiObjEmu>>> layer2inReshape(nInputFramesPerBX_ * tmuxFactor_);
  for (uint iClock = 0; iClock < nInputFramesPerBX_ * tmuxFactor_; iClock++) {
    std::vector<std::vector<l1ct::PuppiObjEmu>> orderedPupsOnClock(boardInfos_.size());
    // Find all the puppis on this clock cycle
    for (BoardInfo boardInfo : boardInfos_) {
      // find all puppis on this clock cycle, from this board
      std::vector<l1ct::PuppiObjEmu> orderedPupsOnClockOnBoard;
      for (uint iLink = 0; iLink < boardInfo.nLinksPuppi_; iLink++) {
        // find all puppis from this clock cycle, from this board, from this link
        auto onClockOnBoardOnLink = [&](PlacedPuppi p) {
          return (p.second.clock_cycle_ == iClock) && (p.second.board_ == boardInfo.order_) &&
                 (p.second.link_ == iLink);
        };
        std::vector<PlacedPuppi> allPupsOnClockOnBoardOnLink;
        std::copy_if(std::begin(linkPlacedPuppis),
                     std::end(linkPlacedPuppis),
                     std::back_inserter(allPupsOnClockOnBoardOnLink),
                     onClockOnBoardOnLink);
        linkPlacedPuppis.erase(
            std::remove_if(std::begin(linkPlacedPuppis), std::end(linkPlacedPuppis), onClockOnBoardOnLink),
            std::end(linkPlacedPuppis));  // erase already placed pups
        if (!allPupsOnClockOnBoardOnLink.empty()) {
          orderedPupsOnClockOnBoard.push_back(allPupsOnClockOnBoardOnLink.at(0).first);
        }
      }
      orderedPupsOnClock.at(boardInfo.order_) = orderedPupsOnClockOnBoard;
    }
    layer2inReshape.at(iClock) = orderedPupsOnClock;
  }
  return layer2inReshape;
}