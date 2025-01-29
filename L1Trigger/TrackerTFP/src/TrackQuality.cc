/*
Track Quality Body file
C.Brown & C.Savard 07/2020
*/

#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include <vector>
#include <map>
#include <string>
#include "conifer.h"
#include "ap_fixed.h"

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  TrackQuality::TrackQuality(const ConfigTQ& iConfig, const DataFormats* dataFormats)
      : dataFormats_(dataFormats),
        model_(iConfig.model_),
        featureNames_(iConfig.featureNames_),
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
    dataFormatsTQ_.emplace_back(FormatTQ<v>(dataFormats_, iConfig));
    if constexpr (++v != VariableTQ::end)
      fillDataFormats<++v>(iConfig);
  }

  // TQ MVA bin conversion LUT
  constexpr array<double, numBinsMVA_> TrackQuality::mvaPreSigBins() const {
    array<double, numBinsMVA_> lut = {};
    lut[0] = -16.;
    for (int i = 1; i < numBinsMVA_; i++)
      lut[i] = invSigmoid(TTTrack_TrackWord::tqMVABins[i]);
    return lut;
  }

  //
  template <class T>
  int TrackQuality::toBin(const T& bins, double d) const {
    int bin = 0;
    for (; bin < (int)bins.size() - 1; bin++)
      if (d < bins[bin + 1])
        break;
    return bin;
  }

  // Helper function to convert mvaPreSig to bin
  int TrackQuality::toBinMVA(double mva) const {
    static const array<double, numBinsMVA_> bins = mvaPreSigBins();
    return toBin(bins, mva);
  }

  // Helper function to convert chi2B to bin
  int TrackQuality::toBinChi2B(double chi2B) const {
    static const array<double, numBinsChi2B_> bins = TTTrack_TrackWord::bendChi2Bins;
    return toBin(bins, chi2B);
  }

  // Helper function to convert chi2rphi to bin
  int TrackQuality::toBinchi2rphi(double chi2rphi) const {
    static const array<double, numBinschi2rphi_> bins = TTTrack_TrackWord::chi2RPhiBins;
    double chi2 = chi2rphi * chi2rphiConv_;
    return toBin(bins, chi2);
  }

  // Helper function to convert chi2rz to bin
  int TrackQuality::toBinchi2rz(double chi2rz) const {
    static const array<double, numBinschi2rz_> bins = TTTrack_TrackWord::chi2RZBins;
    double chi2 = chi2rz * chi2rzConv_;
    return toBin(bins, chi2);
  }

  TrackQuality::Track::Track(const FrameTrack& frameTrack, const StreamStub& streamStub, const TrackQuality* tq)
      : frameTrack_(frameTrack), streamStub_(streamStub) {
    static const DataFormats* df = tq->dataFormats();
    static const Setup* setup = df->setup();
    const TrackDR track(frameTrack, df);
    double trackchi2rphi(0.);
    double trackchi2rz(0.);
    TTBV hitPattern(0, streamStub.size());
    vector<TTStubRef> ttStubRefs;
    ttStubRefs.reserve(setup->numLayers());
    for (int layer = 0; layer < (int)streamStub.size(); layer++) {
      const FrameStub& frameStub = streamStub[layer];
      if (frameStub.first.isNull())
        continue;
      const StubKF stub(frameStub, df);
      hitPattern.set(layer);
      ttStubRefs.push_back(frameStub.first);
      const double m20 = tq->format(VariableTQ::m20).digi(pow(stub.phi(), 2));
      const double m21 = tq->format(VariableTQ::m21).digi(pow(stub.z(), 2));
      const double invV0 = tq->format(VariableTQ::invV0).digi(1. / pow(stub.dPhi(), 2));
      const double invV1 = tq->format(VariableTQ::invV1).digi(1. / pow(stub.dZ(), 2));
      const double stubchi2rphi = tq->format(VariableTQ::chi2rphi).digi(m20 * invV0);
      const double stubchi2rz = tq->format(VariableTQ::chi2rz).digi(m21 * invV1);
      trackchi2rphi += stubchi2rphi;
      trackchi2rz += stubchi2rz;
    }
    if (trackchi2rphi > tq->range(VariableTQ::chi2rphi))
      trackchi2rphi = tq->range(VariableTQ::chi2rphi) - tq->base(VariableTQ::chi2rphi) / 2.;
    if (trackchi2rz > tq->range(VariableTQ::chi2rz))
      trackchi2rz = tq->range(VariableTQ::chi2rz) - tq->base(VariableTQ::chi2rz) / 2.;
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
    const double aphi = deltaPhi(track.phiT() - track.inv2R() * setup->chosenRofPhi() + region * setup->baseRegion());
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
    // load in bdt
    conifer::BDT<ap_fixed<10, 5>, ap_fixed<10, 5>> bdt(tq->model().fullPath());
    // collect features and classify using bdt
    const vector<ap_fixed<10, 5>>& output = bdt.decision_function({cot, z0, chi2B, nstub, n_missint, chi2rphi, chi2rz});
    const float mva = output[0].to_float();
    // fill frame
    TTBV ttBV = hitPattern;
    ttBV += TTBV(tq->toBinMVA(mva), widthMVA_);
    tq->format(VariableTQ::chi2rphi).attach(trackchi2rphi, ttBV);
    tq->format(VariableTQ::chi2rz).attach(trackchi2rz, ttBV);
    frame_ = ttBV.bs();
  }

  template <>
  FormatTQ<VariableTQ::m20>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig) : DataFormat(false) {
    const Format<Variable::phi, Process::kf> phi(dataFormats->setup());
    width_ = iConfig.widthM20_;
    base_ = pow(phi.base(), 2) * pow(2., width_ - phi.width());
    calcRange();
  }
  template <>
  FormatTQ<VariableTQ::m21>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig) : DataFormat(false) {
    const Format<Variable::z, Process::gp> z(dataFormats->setup());
    width_ = iConfig.widthM21_;
    base_ = pow(z.base(), 2) * pow(2., width_ - z.width());
    calcRange();
  }
  template <>
  FormatTQ<VariableTQ::invV0>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig) : DataFormat(false) {
    const Format<Variable::dPhi, Process::ctb> dPhi(dataFormats->setup());
    width_ = iConfig.widthInvV0_;
    range_ = 4.0 / pow(dPhi.base(), 2);
    calcBase();
  }
  template <>
  FormatTQ<VariableTQ::invV1>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig) : DataFormat(false) {
    const Format<Variable::dZ, Process::ctb> dZ(dataFormats->setup());
    width_ = iConfig.widthInvV1_;
    range_ = 4.0 / pow(dZ.base(), 2);
    calcBase();
  }
  template <>
  FormatTQ<VariableTQ::chi2rphi>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig)
      : DataFormat(false) {
    const FormatTQ<VariableTQ::m20> m20(dataFormats, iConfig);
    const FormatTQ<VariableTQ::invV0> invV0(dataFormats, iConfig);
    const int shift = iConfig.baseShiftchi2rphi_;
    width_ = iConfig.widthchi2rphi_;
    base_ = pow(2., shift);
    calcRange();
  }
  template <>
  FormatTQ<VariableTQ::chi2rz>::FormatTQ(const DataFormats* dataFormats, const ConfigTQ& iConfig) : DataFormat(false) {
    const FormatTQ<VariableTQ::m21> m21(dataFormats, iConfig);
    const FormatTQ<VariableTQ::invV1> invV1(dataFormats, iConfig);
    const int shift = iConfig.baseShiftchi2rz_;
    width_ = iConfig.widthchi2rz_;
    base_ = pow(2., shift);
    calcRange();
  }

  // Controls the conversion between TTTrack features and ML model training features
  vector<float> TrackQuality::featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                               vector<string> const& featureNames) const {
    // List input features for MVA in proper order below, the current features options are
    // {"phi", "eta", "z0", "bendchi2_bin", "nstub", "nlaymiss_interior", "chi2rphi_bin",
    // "chi2rz_bin"}
    //
    // To use more features, they must be created here and added to feature_map below
    vector<float> transformedFeatures;
    // Define feature map, filled as features are generated
    map<string, float> feature_map;
    // -------- calculate feature variables --------
    // calculate number of missed interior layers from hitpattern
    int tmp_trk_hitpattern = aTrack.hitPattern();
    int nbits = floor(log2(tmp_trk_hitpattern)) + 1;
    int lay_i = 0;
    int tmp_trk_nlaymiss_interior = 0;
    bool seq = false;
    for (int i = 0; i < nbits; i++) {
      lay_i = ((1 << i) & tmp_trk_hitpattern) >> i;  //0 or 1 in ith bit (right to left)

      if (lay_i && !seq)
        seq = true;  //sequence starts when first 1 found
      if (!lay_i && seq)
        tmp_trk_nlaymiss_interior++;
    }
    // binned chi2 variables
    int tmp_trk_bendchi2_bin = aTrack.getBendChi2Bits();
    int tmp_trk_chi2rphi_bin = aTrack.getChi2RPhiBits();
    int tmp_trk_chi2rz_bin = aTrack.getChi2RZBits();
    // get the nstub
    vector<TTStubRef> stubRefs = aTrack.getStubRefs();
    int tmp_trk_nstub = stubRefs.size();
    // get other variables directly from TTTrack
    float tmp_trk_z0 = aTrack.z0();
    float tmp_trk_z0_scaled = tmp_trk_z0 / abs(aTrack.minZ0);
    float tmp_trk_phi = aTrack.phi();
    float tmp_trk_eta = aTrack.eta();
    float tmp_trk_tanl = aTrack.tanL();
    // -------- fill the feature map ---------
    feature_map["nstub"] = float(tmp_trk_nstub);
    feature_map["z0"] = tmp_trk_z0;
    feature_map["z0_scaled"] = tmp_trk_z0_scaled;
    feature_map["phi"] = tmp_trk_phi;
    feature_map["eta"] = tmp_trk_eta;
    feature_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);
    feature_map["bendchi2_bin"] = tmp_trk_bendchi2_bin;
    feature_map["chi2rphi_bin"] = tmp_trk_chi2rphi_bin;
    feature_map["chi2rz_bin"] = tmp_trk_chi2rz_bin;
    feature_map["tanl"] = tmp_trk_tanl;
    // fill tensor with track params
    transformedFeatures.reserve(featureNames.size());
    for (const string& feature : featureNames)
      transformedFeatures.push_back(feature_map[feature]);
    return transformedFeatures;
  }

  // Passed by reference a track without MVA filled, method fills the track's MVA field
  void TrackQuality::setL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack) const {
    // load in bdt
    conifer::BDT<float, float> bdt(this->model_.fullPath());
    // collect features and classify using bdt
    vector<float> inputs = featureTransform(aTrack, this->featureNames_);
    vector<float> output = bdt.decision_function(inputs);
    aTrack.settrkMVA1(1. / (1. + exp(-output.at(0))));
  }

}  // namespace trackerTFP
