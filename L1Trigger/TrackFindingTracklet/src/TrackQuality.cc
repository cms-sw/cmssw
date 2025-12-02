#include "L1Trigger/TrackFindingTracklet/interface/TrackQuality.h"

#include <vector>
#include <string>
#include <numeric>
#include "conifer.h"
#include "ap_fixed.h"

namespace trklet {

  // read in and organize input tracks and stubs
  void TrackQuality::consume(const tt::StreamsTrack& tracks, const tt::StreamsStub& stubs) {
    streams_ = tracks;
    auto validT = [](int sum, const tt::FrameTrack& f) { return sum + (f.first.isNull() ? 0 : 1); };
    auto validS = [](int sum, const tt::FrameStub& f) { return sum + (f.first.isNull() ? 0 : 1); };
    const int offset = region_ * setup_->numLayers();
    const tt::StreamTrack& input = tracks[region_];
    // count tracks
    const int nTracks = std::accumulate(input.begin(), input.end(), 0, validT);
    tracks_.reserve(nTracks);
    // count stubs
    int nStubs(0);
    for (int iLayer = 0; iLayer < setup_->numLayers(); iLayer++) {
      const tt::StreamStub& layer = stubs[offset + iLayer];
      nStubs += std::accumulate(layer.begin(), layer.end(), 0, validS);
    }
    stubs_.reserve(nStubs);
    // store input
    input_.reserve(input.size());
    for (int iFrame = 0; iFrame < static_cast<int>(input.size()); iFrame++) {
      const tt::FrameTrack& frameTrack = input[iFrame];
      if (frameTrack.first.isNull()) {
        input_.emplace_back(Frame());
        continue;
      }
      tracks_.emplace_back(frameTrack, dataFormats_);
      input_.emplace_back(&tracks_.back(), setup_->numLayers());
      for (int iLayer = 0; iLayer < setup_->numLayers(); iLayer++) {
        const tt::FrameStub& frameStub = stubs[offset + iLayer][iFrame];
        if (frameStub.first.isNull())
          continue;
        stubs_.emplace_back(frameStub, dataFormats_);
        input_.back().stubs_[iLayer] = &stubs_.back();
      }
    }
  }

  // fills output products
  void TrackQuality::produce(tt::StreamsTrack& outputs) const {
    const int offset = setup_->tqNumChannel() * region_;
    outputs[offset + 0] = streams_[region_];
    tt::StreamTrack& output = outputs[offset + 1];
    const DataFormat& dfChi20 = dataFormats_->format(Variable::chi20, Process::tq);
    const DataFormat& dfChi21 = dataFormats_->format(Variable::chi21, Process::tq);
    const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::tq);
    const DataFormat& dfCot = dataFormats_->format(Variable::cot, Process::tq);
    output.reserve(input_.size());
    for (const Frame& frame : input_) {
      if (!frame.track_) {
        output.emplace_back(tt::FrameTrack());
        continue;
      }
      // analyze track and stubs
      double chi20F(0.);
      double chi21F(0.);
      TTBV hitPattern(0, setup_->numLayers());
      for (int layer = 0; layer < setup_->numLayers(); layer++) {
        StubKF* stub = frame.stubs_[layer];
        if (!stub)
          continue;
        hitPattern.set(layer);
        const double m02 = internalFormats_->m02_.digi(std::pow(stub->phi(), 2));
        const double m12 = internalFormats_->m12_.digi(std::pow(stub->z(), 2));
        const double invV0 = internalFormats_->invV0_.digi(1. / std::pow(2. * stub->dPhi(), 2));
        const double invV1 = internalFormats_->invV1_.digi(1. / std::pow(2. * stub->dZ(), 2));
        chi20F += dfChi20.limit(dfChi20.digi(m02 * invV0));
        chi21F += dfChi21.limit(dfChi21.digi(m12 * invV1));
      }
      // Accumulating all BDT Attributes
      chi20F = dfChi20.limit(chi20F);
      chi21F = dfChi21.limit(chi21F);
      const int nStubs = hitPattern.count();
      const int nGaps = hitPattern.count(hitPattern.plEncode(), hitPattern.pmEncode(), false);
      // get integer values
      const int zT = dfZT.integer(frame.track_->zT());
      const int cot = dfCot.integer(frame.track_->cot());
      const int chi20 = dfChi20.integer(chi20F);
      const int chi21 = dfChi21.integer(chi21F);
      // transform double to AP_FIXED_BDT
      static const double d = std::pow(2., 10);
      const std::vector<AP_FIXED_BDT> features({nStubs, zT / d, cot / d, chi20 / d, chi21 / d, nGaps});
      // BDT Inference
      const AP_FIXED_BDT mvaFixed = bdt_->decision_function(features).at(0);
      const AP_INT_BDT mvaInt = mvaFixed.range(mvaFixed.width - 1, 0);
      // bin mva
      const std::vector<int>& binEdges = channelAssignment_->tqBinEdges();
      int mva(0);
      for (; mva < static_cast<int>(binEdges.size()) - 2; mva++)
        if (mvaInt <= binEdges[mva + 1])
          break;
      // build output Track
      std::string s = hitPattern.str();
      std::reverse(s.begin(), s.end());
      TrackTQ trackTQ(*frame.track_, s, mva, chi20F, chi21F);
      // store result
      output.push_back(trackTQ.frame());
    }
  }

}  // namespace trklet
