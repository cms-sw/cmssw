#ifndef ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_
#define ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_

/*** system includes ***/
#include <array>
#include <string>
#include <iostream>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/*** Alignment ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"

class MillePedeFileReader {
  //========================== PUBLIC METHODS ==================================
public:  //====================================================================
  explicit MillePedeFileReader(const edm::ParameterSet&,
                               const std::shared_ptr<const PedeLabelerBase>&,
                               const std::shared_ptr<const AlignPCLThresholds>&);

  virtual ~MillePedeFileReader() = default;

  void read();
  bool storeAlignments();

  const std::array<double, 6>& getXobs() const { return Xobs_; }
  const std::array<double, 6>& getXobsErr() const { return XobsErr_; }
  const std::array<double, 6>& getTXobs() const { return tXobs_; }
  const std::array<double, 6>& getTXobsErr() const { return tXobsErr_; }

  const std::array<double, 6>& getYobs() const { return Yobs_; }
  const std::array<double, 6>& getYobsErr() const { return YobsErr_; }
  const std::array<double, 6>& getTYobs() const { return tYobs_; }
  const std::array<double, 6>& getTYobsErr() const { return tYobsErr_; }

  const std::array<double, 6>& getZobs() const { return Zobs_; }
  const std::array<double, 6>& getZobsErr() const { return ZobsErr_; }
  const std::array<double, 6>& getTZobs() const { return tZobs_; }
  const std::array<double, 6>& getTZobsErr() const { return tZobsErr_; }

  const AlignPCLThresholds::threshold_map getThresholdMap() const { return theThresholds_.get()->getThreshold_Map(); }

private:
  //========================= PRIVATE ENUMS ====================================
  //============================================================================

  enum class PclHLS : int {
    NotInPCL = -1,
    TPBHalfBarrelXplus = 2,
    TPBHalfBarrelXminus = 3,
    TPEHalfCylinderXplusZplus = 4,
    TPEHalfCylinderXminusZplus = 5,
    TPEHalfCylinderXplusZminus = 0,
    TPEHalfCylinderXminusZminus = 1
  };

  //========================= PRIVATE METHODS ==================================
  //============================================================================

  void readMillePedeLogFile();
  void readMillePedeResultFile();
  PclHLS getHLS(const Alignable*);
  std::string getStringFromHLS(PclHLS HLS);

  //========================== PRIVATE DATA ====================================
  //============================================================================

  // pede labeler plugin
  const std::shared_ptr<const PedeLabelerBase> pedeLabeler_;

  // thresholds from DB
  const std::shared_ptr<const AlignPCLThresholds> theThresholds_;

  // file-names
  const std::string millePedeLogFile_;
  const std::string millePedeResFile_;

  // conversion factors: cm to um & rad to urad
  static constexpr std::array<double, 6> multiplier_ = {{10000.,      // X
                                                         10000.,      // Y
                                                         10000.,      // Z
                                                         1000000.,    // tX
                                                         1000000.,    // tY
                                                         1000000.}};  // tZ

  bool updateDB_{false};
  bool vetoUpdateDB_{false};
  int Nrec_{0};

  std::array<double, 6> Xobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> XobsErr_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tXobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tXobsErr_ = {{0., 0., 0., 0., 0., 0.}};

  std::array<double, 6> Yobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> YobsErr_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tYobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tYobsErr_ = {{0., 0., 0., 0., 0., 0.}};

  std::array<double, 6> Zobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> ZobsErr_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tZobs_ = {{0., 0., 0., 0., 0., 0.}};
  std::array<double, 6> tZobsErr_ = {{0., 0., 0., 0., 0., 0.}};
};

const std::array<std::string, 8> coord_str = {{"X", "Y", "Z", "theta_X", "theta_Y", "theta_Z", "extra_DOF", "none"}};
inline std::ostream& operator<<(std::ostream& os, const AlignPCLThresholds::coordType& c) {
  if (c >= AlignPCLThresholds::endOfTypes || c < 0)
    return os << "unrecongnized coordinate";
  return os << coord_str[c];
}

#endif /* ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_ */
