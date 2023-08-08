// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1GTTInputProducer
//
/**\class L1GTTInputProducer L1GTTInputProducer.cc L1Trigger/L1TTrackMatch/plugins/L1GTTInputProducer.cc

 Description: Takes in L1TTracks and outputs the same tracks, but with modifications to the underlying track word.
   The modifications convert from Rinv --> pt and tanL --> eta.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Alexx Perloff
//         Created:  Sat, 20 Feb 2021 17:02:00 GMT
//
//

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Vivado HLS includes
#include <ap_fixed.h>
#include <ap_int.h>

// system include files
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <sstream>
#include <vector>

//
// class declaration
//

class L1GTTInputProducer : public edm::global::EDProducer<> {
public:
  explicit L1GTTInputProducer(const edm::ParameterSet&);
  ~L1GTTInputProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  static constexpr unsigned int Npars4 = 4;
  static constexpr unsigned int Npars5 = 5;
  enum ConversionBitWidths {
    kEtaMagSize = 3,     // eta output magnitude size; MAG + FRAC should be <= kEtaInputSize
    kEtaFracSize = 5,    // eta output fraction size; MAG + FRAC should be <= kEtaInputSize
    kEtaInputSize = 16,  // size of tan(lambda)

    kPTMagSize = 7,     // magnitude output size; MAG + FRAC should be <= kPTInputSize
    kPTFracSize = 3,    // fraction output size; MAG + FRAC should be <= kPTInputSize
    kPTInputSize = 15,  // size of 1/R

    kEtaOutputSize = kEtaMagSize + kEtaFracSize,  // total bit width for eta
    kPTOutputSize = kPTMagSize + kPTFracSize,     // total bit width for pT
  };

  static constexpr double kEtaErrThresh = 0.0485;  // error corresponding to 0.25 of a degree error in lambda

  static constexpr double kPTErrThresh = 5;                    // error threshold in percent
  static constexpr double kSynchrotron = (1.0 / (0.3 * 3.8));  // 1/(0.3*B) for 1/R to 1/pT conversion
  static constexpr unsigned int kPtLutSize = (1 << ConversionBitWidths::kPTOutputSize);
  static constexpr unsigned int kEtaLutSize = (1 << (ConversionBitWidths::kEtaOutputSize - 1));

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1Track;
  typedef std::vector<L1Track> TTTrackCollection;
  typedef edm::View<L1Track> TTTrackCollectionView;
  typedef ap_fixed<kEtaOutputSize, kEtaMagSize, AP_RND_CONV, AP_SAT> out_eta_t;
  typedef TTTrack_TrackWord::tanl_t in_eta_t;
  typedef ap_ufixed<kPTOutputSize, kPTMagSize, AP_RND_CONV, AP_SAT> out_pt_t;
  typedef TTTrack_TrackWord::rinv_t in_pt_t;
  typedef ap_uint<1> out_charge_t;

  // ----------member functions ----------------------
  void generate_eta_lut();
  void generate_pt_lut();
  bool getEtaBits(
      const L1Track& track, out_eta_t& etaBits, double& expected, double& maxErrPerc, double& maxErrEpsilon) const;
  bool getPtBits(const L1Track& track,
                 out_pt_t& ptBits,
                 out_charge_t& chargeBit,
                 double& expected,
                 double& maxErrPerc,
                 double& maxErrEpsilon,
                 double& minErrPerc,
                 double& minExpected) const;
  double indexTanLambda2Eta(unsigned int indexTanLambda) const;
  double inverseRT2InversePT(unsigned int indexRT) const;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  template <typename T>
  int sgn(T val) const {
    return (T(0) < val) - (val < T(0));
  }  // From https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
  double unpackSignedValue(unsigned int bits, unsigned int nBits, double lsb) const;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;
  const std::string outputCollectionName_;
  int debug_;
  std::vector<out_pt_t> pt_lut_;
  std::vector<out_eta_t> eta_lut_;
};

//
// constructors and destructor
//
L1GTTInputProducer::L1GTTInputProducer(const edm::ParameterSet& iConfig)
    : l1TracksToken_(consumes<TTTrackCollectionView>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
      debug_(iConfig.getParameter<int>("debug")) {
  // Generate the required luts
  generate_eta_lut();
  generate_pt_lut();

  // Define EDM output to be written to file (if required)
  produces<TTTrackCollection>(outputCollectionName_);
  produces<std::vector<double>>("L1GTTInputTrackPtExpected");
  produces<std::vector<double>>("L1GTTInputTrackEtaExpected");
}

L1GTTInputProducer::~L1GTTInputProducer() {}

//
// member functions
//

/**
  generate_lut: calculate the lut and write it to a file
  - argument:None
  - return (void): None
  - method:
    1) iterates through all possibilities of the input value
    2) finds values of eta from input values
    3) stores the values of eta in an array (lut)
    4) writes out the array into a file (eta_lut.h)
*/
void L1GTTInputProducer::generate_eta_lut() {
  // initialize lut array
  eta_lut_.reserve(kEtaLutSize);

  // iterate through all values in the lut
  for (unsigned int i = 0; i < kEtaLutSize; i++) {
    // -1 to ignore sign bit for input
    unsigned int index = ((i + 0.5) * pow(2, (int)(kEtaInputSize - kEtaOutputSize)));
    double newValue = indexTanLambda2Eta(index);  // map out the index to the represented eta
    out_eta_t out_eta = newValue;                 // cast it to fxp
    eta_lut_[i] = out_eta;                        // add value for the lut
  }

  if (debug_ >= 3) {
    edm::LogInfo log("L1GTTInputProducer");
    log << "generate_eta_lut::The eta_lut_[" << kEtaLutSize << "] values are ... \n";
    for (unsigned int i = 0; i < kEtaLutSize; i++) {
      log << "\t" << i << "\t" << eta_lut_[i] << "\n";
    }
  }
}

void L1GTTInputProducer::generate_pt_lut() {
  // initialize lut array
  pt_lut_.reserve(kPtLutSize);  // generate lut

  // iterate through all values in the lut
  for (unsigned int i = 0; i < kPtLutSize; i++) {
    unsigned int index = (i + 0.5) * pow(2, (int)(kPTInputSize - 1 - kPTOutputSize));
    double newValue = inverseRT2InversePT(index);  //map out the index to the represented 1/pT
    out_pt_t out_pt = 1.0 / newValue;              // take the reciprocal and cast as an AP fixed-point (1/pt ==> pt)
    pt_lut_[i] = out_pt;                           // setting the i-th value for the lut
  }

  if (debug_ >= 3) {
    edm::LogInfo log("L1GTTInputProducer");
    log << "generate_pt_lut::The pt_lut_[" << kPtLutSize << "] values are ... \n";
    for (unsigned int i = 0; i < kPtLutSize; i++) {
      log << "\t" << i << "\t" << pt_lut_[i] << "\n";
    }
  }
}

double L1GTTInputProducer::unpackSignedValue(unsigned int bits, unsigned int nBits, double lsb) const {
  // Check that none of the bits above the nBits-1 bit, in a range of [0, nBits-1], are set.
  // This makes sure that it isn't possible for the value represented by 'bits' to be
  //  any bigger than ((1 << nBits) - 1).
  assert((bits >> nBits) == 0);

  // Convert from twos compliment to C++ signed integer (normal digitized value)
  int digitizedValue = bits;
  if (bits & (1 << (nBits - 1))) {  // check if the 'bits' is negative
    digitizedValue -= (1 << nBits);
  }

  // Convert to floating point value
  return (double(digitizedValue) + 0.5) * lsb;
}

/**
    indexTanLambda2Eta: calculates eta from tan(lambda)
    - argument:
        indexTanLambda (int): the index representation for tan(lambda)
    - formula:
        f(x) = -1*ln(tan((pi/2-atan(x))/2)), where x = tan(lambda)
    - return (double): eta
*/
double L1GTTInputProducer::indexTanLambda2Eta(unsigned int indexTanLambda) const {
  double tanl = unpackSignedValue(indexTanLambda, kEtaInputSize, TTTrack_TrackWord::stepTanL);
  double theta = (M_PI / 2.0) - atan(tanl);
  double eta = -1.0 * log(tan(theta / 2.0));
  if (debug_ >= 3) {
    edm::LogInfo("L1GTTInputProducer") << "indexTanLambda2Eta::tanl index = " << indexTanLambda << "\n"
                                       << "indexTanLambda2Eta::tanl value = " << tanl << "\n"
                                       << "indexTanLambda2Eta::theta = " << theta << "\n"
                                       << "indexTanLambda2Eta::eta = " << eta;
  }
  return eta;
}

/**
  inverseRT2InversePT: calculates 1/pT from 1/rT
  - argument:
    indexRT (int): the index representation for 1/rT
  - formula:
      f(x) = 100.*(1/(0.3*3.8))*x , where x = 1/R
  - return (double): 1/pT
*/
double L1GTTInputProducer::inverseRT2InversePT(unsigned int indexRT) const {
  double inverseRT = unpackSignedValue(indexRT, kPTInputSize, TTTrack_TrackWord::stepRinv);
  return 100.0 * kSynchrotron * inverseRT;  // multiply by 100 to convert from cm to m
}

bool L1GTTInputProducer::getEtaBits(
    const L1Track& track, out_eta_t& etaBits, double& expected, double& maxErrPerc, double& maxErrEpsilon) const {
  // Conver the input to an ap_uint
  in_eta_t value = track.getTanlWord();

  // Get the expected outcome (floating point)
  out_eta_t maxValuePossible = pow(2, 15);  // saturate at max value possible for fxp
  expected = indexTanLambda2Eta(value);     // expected value for eta
  if (expected > maxValuePossible) {
    expected = maxValuePossible;
  }

  // Converted value (emulation)
  // Masking and shifting converts the efficient bit representation into an index
  // Start by setting up the masks
  in_eta_t indexTanLambda = value;
  in_eta_t mask = ~0;                                                      // mask (all 1's)
  bool sign = indexTanLambda.range(kEtaInputSize - 1, kEtaInputSize - 1);  // sign bit of indexTanLambda
  mask *= sign;  // all 0's for positive numbers, all 1's for negative numbers

  // Take the absolute value of indexTanLambda (2's complement)
  indexTanLambda ^= mask;
  indexTanLambda += sign;

  // Find the value for eta, not |eta|
  indexTanLambda =
      indexTanLambda >>
      (kEtaInputSize -
       kEtaOutputSize);  // Don't subtract 1 because we now want to take into account the sign bit of the output
  indexTanLambda =
      (indexTanLambda < (1 << (kEtaOutputSize - 1))) ? indexTanLambda : in_eta_t((1 << (kEtaOutputSize - 1)) - 1);
  etaBits = eta_lut_[indexTanLambda];

  // Reinacting the sign
  out_eta_t maskOut;
  maskOut.V = ~0;
  maskOut *= sign;
  etaBits ^= maskOut;
  etaBits.V += sign;

  // Compare the floating point calculation to the emulation
  double delta = std::abs(expected - etaBits.to_double());
  double perc_diff = (delta / std::abs(expected)) * 100.;  // calc percentage error
  if (delta > maxErrEpsilon) {
    maxErrPerc = perc_diff;
    maxErrEpsilon = delta;
  }

  if (delta >= kEtaErrThresh) {
    edm::LogError("L1GTTInputProducer") << "getEtaBits::MISMATCH!!!\n"
                                        << "\tTTTrack tanL = " << track.tanL() << "\n"
                                        << "\tTTTrack eta = " << track.momentum().eta() << "\n"
                                        << "\tTTTrack_TrackWord = " << track.getTrackWord().to_string(2) << "\n"
                                        << "\tTTTrack_TrackWord tanlWord = " << track.getTanlWord() << " ("
                                        << track.getTanlWord().to_string(2) << ")\n"
                                        << "\tin_eta_t value = " << value << " (" << value.to_string(2) << ")\n"
                                        << "\tExpected value = " << expected << "\n"
                                        << "\tCalculated eta = " << etaBits.to_double() << " (" << etaBits.to_string(2)
                                        << ") @ index " << indexTanLambda << "\n"
                                        << "\tDelta = " << delta << "\tpercentage error = " << perc_diff;
    return true;
  } else {
    if (debug_ >= 2) {
      edm::LogInfo("L1GTTInputProducer")
          << "getEtaBits::SUCCESS (TTTrack, floating eta calculation, bitwise calculation, initial index, lut index) = "
          << "(" << track.momentum().eta() << ", " << expected << ", " << etaBits << ", " << value << ", "
          << indexTanLambda << ")";
    }
  }

  return false;
}

bool L1GTTInputProducer::getPtBits(const L1Track& track,
                                   out_pt_t& ptBits,
                                   out_charge_t& chargeBit,
                                   double& expected,
                                   double& maxErrPerc,
                                   double& maxErrEpsilon,
                                   double& minErrPerc,
                                   double& minExpected) const {
  // Convert the input to an ap_uint
  in_pt_t value = track.getRinvWord();
  in_pt_t value_initial = value;

  // Get the expected outcome (floating point)
  out_pt_t maxValuePossible = pow(2, 16);       // saturate at max value possible for fxp
  expected = 1.0 / inverseRT2InversePT(value);  // expected value for inverse
  bool saturation = true;
  if (std::abs(expected) > maxValuePossible) {
    expected = maxValuePossible;
  } else {
    saturation = false;
  }

  // Converted value (emulation)
  // Masking and shifting converts the efficient bit representation into an index
  // Start by setting up the masks
  in_pt_t mask = ~0;                                            // mask (all 1's)
  bool sign = value.range(kPTInputSize - 1, kPTInputSize - 1);  // sign bit of value
  mask *= sign;  // all 0's for positive numbers, all 1's for negative numbers

  // Take the absolute value of value (2's complement)
  value ^= mask;
  value += sign;

  // Shift the value so that the index changes when the LSB of the output changes
  value = value >> (kPTInputSize - 1 - (kPTOutputSize));

  // Get the pt from the LUT
  ptBits = pt_lut_[value];

  // Set the charge bit
  chargeBit = sign;
  double charge = 1. - (2 * chargeBit.to_uint());

  // Compare the floating point calculation to the emulation
  double delta = std::abs(expected - (charge * ptBits.to_double()));
  double perc_diff = (delta / std::abs(expected)) * 100.;

  if (delta > maxErrEpsilon) {
    maxErrPerc = perc_diff;
    maxErrEpsilon = delta;
  } else if (delta < minExpected && !saturation && minErrPerc > 100.0) {
    minErrPerc = perc_diff;
    minExpected = expected;
  }

  if (std::abs(perc_diff) >= kPTErrThresh && !saturation) {
    edm::LogError("L1GTTInputProducer") << "getPtBits::MISMATCH!!!\n"
                                        << "\tTTTrack Rinv = " << track.rInv() << "\n"
                                        << "\tTTTrack pt = " << track.momentum().transverse() << "\n"
                                        << "\tTTTrack_TrackWord = " << track.getTrackWord().to_string(2) << "\n"
                                        << "\tTTTrack_TrackWord RinvWord = " << track.getRinvWord() << " ("
                                        << track.getRinvWord().to_string(2) << ")\n"
                                        << "\tin_pt_t value = " << value_initial << " (" << value_initial.to_string(2)
                                        << ")\n"
                                        << "\tExpected value = " << expected << "\n"
                                        << "\tCalculated pt = " << ptBits.to_double() << " (" << ptBits.to_string(2)
                                        << ") @ index " << value << "\n"
                                        << "\tcharge = " << charge << " (bit = " << chargeBit << ")\n"
                                        << "\tDelta = " << delta << "\tpercentage error = " << perc_diff;
    return true;
  } else {
    if (debug_ >= 2) {
      edm::LogInfo("L1GTTInputProducer") << "getPtBits::SUCCESS (TTTrack, floating pt calculation, charge, bitwise "
                                            "calculation, initial index, lut index) = "
                                         << "(" << sgn(track.rInv()) * track.momentum().transverse() << ", " << expected
                                         << ", " << charge << ", " << ptBits << ", " << value_initial << ", " << value
                                         << ")";
    }
  }

  return false;
}

// ------------ method called to produce the data  ------------
void L1GTTInputProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto vTTTrackOutput = std::make_unique<TTTrackCollection>();
  auto vPtOutput = std::make_unique<std::vector<double>>();
  auto vEtaOutput = std::make_unique<std::vector<double>>();

  edm::Handle<TTTrackCollectionView> l1TracksHandle;
  iEvent.getByToken(l1TracksToken_, l1TracksHandle);

  out_charge_t chargeBit = 0;
  out_pt_t ptBits = 0;
  out_eta_t etaBits = 0;
  in_pt_t ptBitsShifted = 0;
  in_eta_t etaBitsShifted = 0;
  unsigned int error_pt_c = 0;        // error counter
  unsigned int error_eta_c = 0;       // error counter
  double expectedPt = 0.0;            // expected value of the pt
  double expectedEta = 0.0;           // expected value of the eta
  double maxErrPercPt = 0.0;          // stores the maximum error percentage
  double maxErrPercEta = 0.0;         // stores the maximum error percentage
  double maxErrEpsilonPt = 0.0;       // keeps track of epsilon for max error
  double maxErrEpsilonEta = 0.0;      // keeps track of epsilon for max error
  double minErrPercPt = 10000000.0;   // stores the maximum error percentage
  double minExpectedPt = 10000000.0;  // keeps track of epsilon for max error

  unsigned int nOutput = l1TracksHandle->size();
  vTTTrackOutput->reserve(nOutput);
  vPtOutput->reserve(nOutput);
  vEtaOutput->reserve(nOutput);
  for (const auto& track : *l1TracksHandle) {
    if (!(track.nFitPars() == Npars4 || track.nFitPars() == Npars5)) {
      throw cms::Exception("nFitPars unknown")
          << "L1GTTInputProducer::produce method is called with numFitPars_ = " << track.nFitPars()
          << ". The only possible values are 4/5.";
    }

    // Fill the vector of tracks
    vTTTrackOutput->push_back(track);
    auto& currentTrackRef = vTTTrackOutput->back();
    if (debug_ >= 2) {
      edm::LogInfo("L1GTTInputProducer") << "produce::word before anything "
                                         << currentTrackRef.getTrackWord().to_string(2);
    }

    // Do an initial setting of the bits based on the floating point values
    currentTrackRef.setTrackWordBits();
    if (debug_ >= 2) {
      edm::LogInfo("L1GTTInputProducer") << "produce::word after initial setting of the track word "
                                         << currentTrackRef.getTrackWord().to_string(2);
    }

    // Do the conversions
    error_pt_c += getPtBits(
        currentTrackRef, ptBits, chargeBit, expectedPt, maxErrPercPt, maxErrEpsilonPt, minErrPercPt, minExpectedPt);
    error_eta_c += getEtaBits(currentTrackRef, etaBits, expectedEta, maxErrPercEta, maxErrEpsilonEta);

    // Assign the exat same bits to an ap_uint
    ptBitsShifted = ptBits.range();
    etaBitsShifted = etaBits.range();

    // Shift the bits so that the decimal is in the right spot for the GTT software
    ptBitsShifted = ptBitsShifted << 2;
    etaBitsShifted = etaBitsShifted << 8;

    // Set the MSB for the pt to the sign of the incoming word
    ptBitsShifted.set(kPTInputSize - 1, chargeBit);

    // Set the correct bits based on the converted quanteties and the existing track word components
    currentTrackRef.setTrackWord(currentTrackRef.getValidWord(),
                                 ptBitsShifted,
                                 currentTrackRef.getPhiWord(),
                                 etaBitsShifted,
                                 currentTrackRef.getZ0Word(),
                                 currentTrackRef.getD0Word(),
                                 currentTrackRef.getChi2RPhiWord(),
                                 currentTrackRef.getChi2RZWord(),
                                 currentTrackRef.getBendChi2Word(),
                                 currentTrackRef.getHitPatternWord(),
                                 currentTrackRef.getMVAQualityWord(),
                                 currentTrackRef.getMVAOtherWord());
    if (debug_ >= 2) {
      edm::LogInfo("L1GTTInputProducer") << "produce::charge after all conversions " << chargeBit << "\n"
                                         << "produce::ptBits after all conversions " << ptBits.to_string(2) << " ("
                                         << ptBitsShifted.to_string(2) << " = " << ptBitsShifted.to_uint() << ")\n"
                                         << "produce::etaBits after all conversions " << etaBits.to_string(2) << " ("
                                         << etaBitsShifted.to_string(2) << " = " << etaBitsShifted.to_uint() << ")\n"
                                         << "produce::word after all conversions "
                                         << vTTTrackOutput->back().getTrackWord().to_string(2);
    }

    // Fill the remaining outputs
    vPtOutput->push_back(expectedPt);
    vEtaOutput->push_back(expectedEta);
  }

  if (debug_ >= 1) {
    edm::LogInfo("L1GTTInputProducer") << "\nNumber of converted tracks: " << nOutput << "\n\n"
                                       << "q/r ==> pt conversion:\n"
                                       << "\tError Threshold: " << kPTErrThresh << "%\n"
                                       << "\tMax error: " << maxErrEpsilonPt
                                       << " GeV difference with percentage: " << maxErrPercPt << "% @ "
                                       << 100.0 * maxErrEpsilonPt / maxErrPercPt << " GeV"
                                       << "\n"
                                       << "\tError @ max range: " << minExpectedPt
                                       << " GeV with precentage: " << minErrPercPt << "%"
                                       << "\n"
                                       << "\tTotal number of errors: " << error_pt_c << "\n\n"
                                       << "tan(lambda) ==> eta conversion:\n"
                                       << "\tError Threshold: " << kEtaErrThresh << "\n"
                                       << "\tMax error: " << maxErrEpsilonEta << " with percentage: " << maxErrPercEta
                                       << "% @ " << 100.0 * maxErrEpsilonEta / maxErrPercEta << "\n"
                                       << "\tTotal number of errors: " << error_eta_c;
  }

  if (error_pt_c + error_eta_c) {
    edm::LogError("L1GTTInputProducer") << "produce::" << error_pt_c << "/" << error_eta_c
                                        << " pt/eta mismatches detected!!!";
  }

  // Set the GTTLinkIndex for downstream modules
  std::vector<int> index_counters(18, 0);
  for (auto& track : *vTTTrackOutput) {
    // Get the GTT Link ID (0-17) for the track
    unsigned int gtt_link_id = track.gttLinkID();
    // Assign the GTTLinkIndex to the converted tracks
    track.setGTTLinkIndex(index_counters[gtt_link_id]);
    index_counters[gtt_link_id]++;
  }

  // Put the outputs into the event
  iEvent.put(std::move(vTTTrackOutput), outputCollectionName_);
  iEvent.put(std::move(vPtOutput), "L1GTTInputTrackPtExpected");
  iEvent.put(std::move(vEtaOutput), "L1GTTInputTrackEtaExpected");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1GTTInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // L1GTTInputProducer
  edm::ParameterSetDescription desc;
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  desc.add<edm::InputTag>("l1TracksInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<std::string>("outputCollectionName", "Level1TTTracksConverted");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GTTInputProducer);
