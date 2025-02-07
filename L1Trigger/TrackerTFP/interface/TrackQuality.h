/*
Track Quality Header file
C.Brown 28/07/20
*/

#ifndef L1Trigger_TrackerTFP_TrackQuality_h
#define L1Trigger_TrackerTFP_TrackQuality_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>
#include <string>
#include <cmath>
#include "ap_fixed.h"

namespace trackerTFP {

  // number of mva bit
  static constexpr int widthMVA_ = TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize;
  // number of mva bins
  static constexpr int numBinsMVA_ = 1 << widthMVA_;
  // number of chi2B bins
  static constexpr int numBinsChi2B_ = 1 << TTTrack_TrackWord::TrackBitWidths::kBendChi2Size;
  // number of chi2rphi bins
  static constexpr int numBinschi2rphi_ = 1 << TTTrack_TrackWord::TrackBitWidths::kChi2RPhiSize;
  // number of chi2rz bins
  static constexpr int numBinschi2rz_ = 1 << TTTrack_TrackWord::TrackBitWidths::kChi2RZSize;

  // track quality variables
  enum class VariableTQ { begin, m20 = begin, m21, invV0, invV1, chi2rphi, chi2rz, end, x };
  // conversion: Variable to int
  inline constexpr int operator+(VariableTQ v) { return static_cast<int>(v); }
  // increment of Variable
  inline constexpr VariableTQ operator+(VariableTQ v, int i) { return VariableTQ(+v + i); }

  // configuration
  struct ConfigTQ {
    edm::FileInPath model_;
    std::vector<std::string> featureNames_;
    double baseShiftCot_;
    double baseShiftZ0_;
    double baseShiftAPfixed_;
    int chi2rphiConv_;
    int chi2rzConv_;
    int weightBinFraction_;
    int dzTruncation_;
    int dphiTruncation_;
    int widthM20_;
    int widthM21_;
    int widthInvV0_;
    int widthInvV1_;
    int widthchi2rphi_;
    int widthchi2rz_;
    int baseShiftchi2rphi_;
    int baseShiftchi2rz_;
  };

  // function template for DataFormat generation
  template <VariableTQ v>
  DataFormat makeDataFormat(const DataFormats* dataFormats, const ConfigTQ& iConfig);

  // specializations

  template <>
  DataFormat makeDataFormat<VariableTQ::m20>(const DataFormats* dataFormats, const ConfigTQ& iConfig);
  template <>
  DataFormat makeDataFormat<VariableTQ::m21>(const DataFormats* dataFormats, const ConfigTQ& iConfig);
  template <>
  DataFormat makeDataFormat<VariableTQ::invV0>(const DataFormats* dataFormats, const ConfigTQ& iConfig);
  template <>
  DataFormat makeDataFormat<VariableTQ::invV1>(const DataFormats* dataFormats, const ConfigTQ& iConfig);
  template <>
  DataFormat makeDataFormat<VariableTQ::chi2rphi>(const DataFormats* dataFormats, const ConfigTQ& iConfig);
  template <>
  DataFormat makeDataFormat<VariableTQ::chi2rz>(const DataFormats* dataFormats, const ConfigTQ& iConfig);

  /*! \class  trackerTFP::TrackQuality
   *  \brief  Bit accurate emulation of the track quality BDT
   *  \author C.Brown
   *  \date   28/07/20
   *  \update 2024, June by Claire Savard
   *  \update 2024, Aug by Thomas Schuh
   */
  class TrackQuality {
  public:
    TrackQuality() {}
    TrackQuality(const ConfigTQ& iConfig, const DataFormats* dataFormats);
    ~TrackQuality() {}
    // object to represent tracks
    struct Track {
      Track(const tt::FrameTrack& frameTrack, const tt::StreamStub& streamStub, const TrackQuality* tq);
      // track frame
      tt::FrameTrack frameTrack_;
      // additional track variables
      tt::Frame frame_;
      // collection of stubs forming track
      tt::StreamStub streamStub_;
    };
    // provides dataformats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // Controls the conversion between TTTrack features and ML model training features
    std::vector<float> featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                        const std::vector<std::string>& featureNames) const;
    // Passed by reference a track without MVA filled, method fills the track's MVA field
    void setL1TrackQuality(TTTrack<Ref_Phase2TrackerDigi_>& aTrack) const;
    // Helper function to convert mvaPreSig to bin
    int toBinMVA(double mva) const;
    // Helper function to convert chi2B to bin
    int toBinChi2B(double chi2B) const;
    // Helper function to convert chi2rphi to bin
    int toBinchi2rphi(double chi2rphi) const;
    // Helper function to convert chi2rz to bin
    int toBinchi2rz(double chi2rz) const;
    //
    double scaleCot(int cot) const { return scaleAP(scale(cot, baseShiftCot_)); }
    //
    double scaleZ0(int z0) const { return scaleAP(scale(z0, baseShiftZ0_)); }
    // access to spedific format
    const DataFormat& format(VariableTQ v) const { return dataFormatsTQ_[+v]; }
    //
    double base(VariableTQ v) const { return dataFormatsTQ_[+v].base(); }
    //
    double range(VariableTQ v) const { return dataFormatsTQ_[+v].range(); }
    //
    const edm::FileInPath& model() const { return model_; }

  private:
    // constructs TQ data formats
    template <VariableTQ v = VariableTQ::begin>
    void fillDataFormats(const ConfigTQ& iConfig);
    // TQ MVA bin conversion LUT
    constexpr std::array<double, numBinsMVA_> mvaPreSigBins() const;
    //
    static constexpr double invSigmoid(double value) { return -log(1. / value - 1.); }
    //
    template <class T>
    int toBin(const T& bins, double d) const;
    //
    int scale(int i, int shift) const { return floor(i * pow(2., shift)); }
    //
    double scaleAP(int i) const { return i * pow(2., baseShiftAPfixed_); }
    // provides dataformats
    const DataFormats* dataFormats_;
    //
    ConfigTQ iConfig_;
    //
    edm::FileInPath model_;
    //
    std::vector<std::string> featureNames_;
    //
    double baseShiftCot_;
    //
    double baseShiftZ0_;
    //
    double baseShiftAPfixed_;
    // Conversion factor between dphi^2/weight and chi2rphi
    int chi2rphiConv_;
    // Conversion factor between dz^2/weight and chi2rz
    int chi2rzConv_;
    // Fraction of total dphi and dz ranges to calculate v0 and v1 LUT for
    int weightBinFraction_;
    // Constant used in FW to prevent 32-bit int overflow
    int dzTruncation_;
    // Constant used in FW to prevent 32-bit int overflow
    int dphiTruncation_;
    // collection of unique formats
    std::vector<DataFormat> dataFormatsTQ_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::TrackQuality, trackerTFP::DataFormatsRcd);

#endif
