/*
Track Quality Body file
C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include <vector>
#include <map>
#include <string>
#include "ap_fixed.h"

namespace trackerTFP {

  TrackQuality::TrackQuality(const ConfigTQ& iConfig, const DataFormats* dataFormats)
      : dataFormats_(dataFormats),
        bdt_digi_(iConfig.model_.fullPath()),  // load digitized BDT calculator
        baseShiftCot_(iConfig.baseShiftCot_),
        baseShiftZ0_(iConfig.baseShiftZ0_),
        baseShiftAPfixed_(iConfig.baseShiftAPfixed_),
        chi2rphiConv_(iConfig.chi2rphiConv_),
        chi2rzConv_(iConfig.chi2rzConv_),
        weightBinFraction_(iConfig.weightBinFraction_),
        dzTruncation_(iConfig.dzTruncation_),
        dphiTruncation_(iConfig.dphiTruncation_) {
    dataFormatsTQ_.reserve(+VariableTQ::end);
    fillDataFormats(iConfig);
  }

  // constructs TQ data formats
  template <VariableTQ v>
  void TrackQuality::fillDataFormats(const ConfigTQ& iConfig) {
    dataFormatsTQ_.emplace_back(makeDataFormat<v>(dataFormats_, iConfig));
    if constexpr (v + 1 != VariableTQ::end)
      fillDataFormats<v + 1>(iConfig);
  }

  // TQ MVA bin conversion LUT
  constexpr std::array<double, numBinsMVA_> TrackQuality::mvaPreSigBins() const {
    std::array<double, numBinsMVA_> lut = {};
    lut[0] = -16.;
    for (int i = 1; i < numBinsMVA_; i++)
      lut[i] = invSigmoid(TTTrack_TrackWord::tqMVABins[i]);
    return lut;
  }

  //
  template <class T>
  int TrackQuality::toBin(const T& bins, double d) const {
    int bin = 0;
    for (; bin < static_cast<int>(bins.size()) - 1; bin++)
      if (d < bins[bin + 1])
        break;
    return bin;
  }

  // Helper function to convert mvaPreSig to bin
  int TrackQuality::toBinMVA(double mva) const {
    static const std::array<double, numBinsMVA_> bins = mvaPreSigBins();
    return toBin(bins, mva);
  }

  // Helper function to convert chi2B to bin
  int TrackQuality::toBinChi2B(double chi2B) const {
    static const std::array<double, numBinsChi2B_> bins = TTTrack_TrackWord::bendChi2Bins;
    return toBin(bins, chi2B);
  }

  // Helper function to convert chi2rphi to bin
  int TrackQuality::toBinchi2rphi(double chi2rphi) const {
    static const std::array<double, numBinschi2rphi_> bins = TTTrack_TrackWord::chi2RPhiBins;
    double chi2 = chi2rphi * chi2rphiConv_;
    return toBin(bins, chi2);
  }

  // Helper function to convert chi2rz to bin
  int TrackQuality::toBinchi2rz(double chi2rz) const {
    static const std::array<double, numBinschi2rz_> bins = TTTrack_TrackWord::chi2RZBins;
    double chi2 = chi2rz * chi2rzConv_;
    return toBin(bins, chi2);
  }

  // Calculate the digitized BDT (used by HYBRID_NEWKF)
  TrackQuality::Track::Track(const tt::FrameTrack& frameTrack, const tt::StreamStub& streamStub, const TrackQuality* tq)
      : frameTrack_(frameTrack), streamStub_(streamStub) {
    static const DataFormats* df = tq->dataFormats();
    static const tt::Setup* setup = df->setup();
    const TrackDR track(frameTrack, df);
    double trackchi2rphi(0.);
    double trackchi2rz(0.);
    TTBV hitPattern(0, streamStub.size());
    std::vector<TTStubRef> ttStubRefs;
    ttStubRefs.reserve(setup->numLayers());
    for (int layer = 0; layer < (int)streamStub.size(); layer++) {
      const tt::FrameStub& frameStub = streamStub[layer];
      if (frameStub.first.isNull())
        continue;
      const StubKF stub(frameStub, df);
      hitPattern.set(layer);
      ttStubRefs.push_back(frameStub.first);

      const double m20 = tq->format(VariableTQ::m20).digi(std::pow(stub.phi(), 2));
      const double m21 = tq->format(VariableTQ::m21).digi(std::pow(stub.z(), 2));
      const double invV0 = tq->format(VariableTQ::invV0).digi(1. / std::pow(2. * stub.dPhi(), 2));
      const double invV1 = tq->format(VariableTQ::invV1).digi(1. / std::pow(2. * stub.dZ(), 2));
      const double stubchi2rphi = tq->format(VariableTQ::chi20).digi(m20 * invV0);
      const double stubchi2rz = tq->format(VariableTQ::chi21).digi(m21 * invV1);
      trackchi2rphi += stubchi2rphi;
      trackchi2rz += stubchi2rz;
    }
    if (trackchi2rphi > tq->range(VariableTQ::chi20))
      trackchi2rphi = tq->range(VariableTQ::chi20) - tq->base(VariableTQ::chi20) / 2.;
    if (trackchi2rz > tq->range(VariableTQ::chi21))
      trackchi2rz = tq->range(VariableTQ::chi21) - tq->base(VariableTQ::chi21) / 2.;
    // calc bdt inputs
    const double cot = tq->scaleCot(df->format(Variable::cot, Process::dr).integer(track.cot()));
    const double z0 =
        tq->scaleZ0(df->format(Variable::zT, Process::kf).integer(track.zT() - setup->chosenRofZ() * track.cot()));
    const int nstub = hitPattern.count();
    const int n_missint = hitPattern.count(hitPattern.plEncode() + 1, setup->numLayers(), false);
    // use simulation for bendchi2
    const TTTrackRef& ttTrackRef = frameTrack.first;
    const int region = ttTrackRef->phiSector();
    const double aRinv = -.5 * track.inv2R();
    const double aphi =
        tt::deltaPhi(track.phiT() - track.inv2R() * setup->chosenRofPhi() + region * setup->baseRegion());
    const double aTanLambda = track.cot();
    const double az0 = track.zT() - track.cot() * setup->chosenRofZ();
    const double ad0 = ttTrackRef->d0();
    static constexpr double aChi2xyfit = 0.;
    static constexpr double aChi2zfit = 0.;
    static constexpr double trkMVA1 = 0.;
    static constexpr double trkMVA2 = 0.;
    static constexpr double trkMVA3 = 0.;
    static constexpr unsigned int aHitpattern = 0;
    const unsigned int nPar = ttTrackRef->nFitPars();
    static const double Bfield = setup->bField();
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack(
        aRinv, aphi, aTanLambda, az0, ad0, aChi2xyfit, aChi2zfit, trkMVA1, trkMVA2, trkMVA3, aHitpattern, nPar, Bfield);
    ttTrack.setStubRefs(ttStubRefs);
    ttTrack.setStubPtConsistency(
        StubPtConsistency::getConsistency(ttTrack, setup->trackerGeometry(), setup->trackerTopology(), Bfield, nPar));
    const int chi2B = tq->toBinChi2B(ttTrack.chi2Bend());
    const int chi2rphi = tq->toBinchi2rphi(trackchi2rphi);
    const int chi2rz = tq->toBinchi2rz(trackchi2rz);

    // collect features and classify using bdt
    const std::vector<TrackQuality::AP_FIXED_BDT>& output =
        tq->bdt_digi().decision_function({cot, z0, chi2B, nstub, n_missint, chi2rphi, chi2rz});

    const float mva = output[0].to_float();
    // fill frame
    std::string hits = hitPattern.str();
    std::reverse(hits.begin(), hits.end());
    TTBV ttBV(hits);
    ttBV += TTBV(tq->toBinMVA(mva), widthMVA_);
    tq->format(VariableTQ::chi20).attach(trackchi2rphi, ttBV);
    tq->format(VariableTQ::chi21).attach(trackchi2rz, ttBV);
    frame_ = ttBV.bs();
  }

  template <>
  DataFormat makeDataFormat<VariableTQ::m20>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::kf>(dataFormats->setup());
    const int width = 2 * phi.width();
    const double base = std::pow(phi.base(), 2);
    const double range = std::pow(phi.range(), 2) / 4.;
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<VariableTQ::m21>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const DataFormat z = makeDataFormat<Variable::z, Process::gp>(dataFormats->setup());
    const int width = 2 * z.width();
    const double base = std::pow(z.base(), 2);
    const double range = std::pow(z.range(), 2) / 4.;
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<VariableTQ::invV0>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const DataFormat dPhi = makeDataFormat<Variable::dPhi, Process::ctb>(dataFormats->setup());
    const int width = iConfig.widthInvV0_;
    double base = std::pow(dPhi.base(), -2);
    const double range = base * std::pow(2, width) / (std::pow(2, width) - 1);
    const int shift = std::ceil(std::log2(range / base)) - width;
    base *= std::pow(2, shift);
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<VariableTQ::invV1>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const DataFormat dZ = makeDataFormat<Variable::dZ, Process::ctb>(dataFormats->setup());
    const int width = iConfig.widthInvV1_;
    double base = std::pow(dZ.base(), -2);
    const double range = base * std::pow(2, width) / (std::pow(2, width) - 1);
    const int shift = std::ceil(std::log2(range / base)) - width;
    base *= std::pow(2, shift);
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<VariableTQ::chi20>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const int shift = iConfig.baseShiftChi20_;
    const int width = iConfig.widthChi20_;
    const double base = std::pow(2., shift);
    const double range = base * std::pow(2, width);
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<VariableTQ::chi21>(const DataFormats* dataFormats, const ConfigTQ& iConfig) {
    const int shift = iConfig.baseShiftChi21_;
    const int width = iConfig.widthChi21_;
    const double base = std::pow(2., shift);
    const double range = base * std::pow(2, width);
    return DataFormat(false, width, base, range);
  }

}  // namespace trackerTFP
