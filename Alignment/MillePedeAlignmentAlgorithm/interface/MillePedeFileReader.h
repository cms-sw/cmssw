#ifndef ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_
#define ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_

/*** system includes ***/
#include <array>
#include <string>
#include <iostream>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Alignment ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"

struct mpPCLresults {
private:
  bool m_isHG;
  bool m_isDBUpdated;
  bool m_isDBUpdateVetoed;
  int m_nRecords;
  int m_exitCode;
  std::string m_exitMessage;
  std::bitset<4> m_updateBits;

public:
  mpPCLresults(bool isDBUpdated,
               bool isDBUpdateVetoed,
               int nRecords,
               int exitCode,
               std::string exitMessage,
               std::bitset<4> updateBits,
               bool isHG)
      : m_isHG(isHG),
        m_isDBUpdated(isDBUpdated),
        m_isDBUpdateVetoed(isDBUpdateVetoed),
        m_nRecords(nRecords),
        m_exitCode(exitCode),
        m_exitMessage(exitMessage),
        m_updateBits(updateBits) {}

  const bool isHighGranularity() { return m_isHG; }
  const bool getDBUpdated() { return m_isDBUpdated; }
  const bool getDBVetoed() { return m_isDBUpdateVetoed; }
  const bool exceedsThresholds() { return m_updateBits.test(0); }
  const bool exceedsCutoffs() { return m_updateBits.test(1); }
  const bool exceedsMaxError() { return m_updateBits.test(2); }
  const bool belowSignificance() { return m_updateBits.test(3); }
  const int getNRecords() { return m_nRecords; }
  const int getExitCode() { return m_exitCode; }
  const std::string getExitMessage() { return m_exitMessage; }

  void print() {
    edm::LogInfo("MillePedeFileReader") << " is DB updated: " << m_isDBUpdated
                                        << " is DB update vetoed: " << m_isDBUpdateVetoed << " nRecords: " << m_nRecords
                                        << " exitCode: " << m_exitCode << " (" << m_exitMessage << ")" << std::endl;
  }
};

class MillePedeFileReader {
  //========================== PUBLIC METHODS ==================================
public:  //====================================================================
  explicit MillePedeFileReader(const edm::ParameterSet&,
                               const std::shared_ptr<const PedeLabelerBase>&,
                               const std::shared_ptr<const AlignPCLThresholdsHG>&,
                               const std::shared_ptr<const PixelTopologyMap>&);

  virtual ~MillePedeFileReader() = default;

  void read();
  bool storeAlignments();

  enum { SIZE_LG_STRUCTS = 6, SIZE_HG_STRUCTS = 820 };

  const std::array<double, SIZE_LG_STRUCTS>& getXobs() const { return Xobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getXobsErr() const { return XobsErr_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTXobs() const { return tXobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTXobsErr() const { return tXobsErr_; }

  const std::array<double, SIZE_LG_STRUCTS>& getYobs() const { return Yobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getYobsErr() const { return YobsErr_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTYobs() const { return tYobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTYobsErr() const { return tYobsErr_; }

  const std::array<double, SIZE_LG_STRUCTS>& getZobs() const { return Zobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getZobsErr() const { return ZobsErr_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTZobs() const { return tZobs_; }
  const std::array<double, SIZE_LG_STRUCTS>& getTZobsErr() const { return tZobsErr_; }

  const std::array<double, SIZE_HG_STRUCTS>& getXobs_HG() const { return Xobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getXobsErr_HG() const { return XobsErr_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTXobs_HG() const { return tXobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTXobsErr_HG() const { return tXobsErr_HG_; }

  const std::array<double, SIZE_HG_STRUCTS>& getYobs_HG() const { return Yobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getYobsErr_HG() const { return YobsErr_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTYobs_HG() const { return tYobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTYobsErr_HG() const { return tYobsErr_HG_; }

  const std::array<double, SIZE_HG_STRUCTS>& getZobs_HG() const { return Zobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getZobsErr_HG() const { return ZobsErr_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTZobs_HG() const { return tZobs_HG_; }
  const std::array<double, SIZE_HG_STRUCTS>& getTZobsErr_HG() const { return tZobsErr_HG_; }

  const AlignPCLThresholdsHG::threshold_map getThresholdMap() const { return theThresholds_.get()->getThreshold_Map(); }

  const int binariesAmount() const { return binariesAmount_; }

  const mpPCLresults getResults() const {
    return mpPCLresults(updateDB_, vetoUpdateDB_, Nrec_, exitCode_, exitMessage_, updateBits_, isHG_);
  }

  const std::map<std::string, std::array<bool, 6>>& getResultsHG() const { return fractionExceeded_; }

private:
  //========================= PRIVATE ENUMS ====================================
  //============================================================================

  enum class PclHLS : int {
    NotInPCL = -1,
    TPEHalfCylinderXplusZminus = 0,
    TPEHalfCylinderXminusZminus = 1,
    TPBHalfBarrelXplus = 2,
    TPBHalfBarrelXminus = 3,
    TPEHalfCylinderXplusZplus = 4,
    TPEHalfCylinderXminusZplus = 5,
    TPBLadderLayer1 = 6,
    TPBLadderLayer2 = 7,
    TPBLadderLayer3 = 8,
    TPBLadderLayer4 = 9,
    TPEPanelDisk1 = 10,
    TPEPanelDisk2 = 11,
    TPEPanelDisk3 = 12,
    TPEPanelDiskM1 = 13,
    TPEPanelDiskM2 = 14,
    TPEPanelDiskM3 = 15,
  };

  //========================= PRIVATE METHODS ==================================
  //============================================================================

  void readMillePedeEndFile();
  void readMillePedeLogFile();
  void readMillePedeResultFile();
  PclHLS getHLS(const Alignable*);
  std::string getStringFromHLS(PclHLS HLS);
  int getIndexForHG(align::ID id, PclHLS HLS);
  void initializeIndexHelper();

  //========================== PRIVATE DATA ====================================
  //============================================================================

  // pede labeler plugin
  const std::shared_ptr<const PedeLabelerBase> pedeLabeler_;

  // thresholds from DB
  const std::shared_ptr<const AlignPCLThresholdsHG> theThresholds_;

  // PixelTopologyMap
  const std::shared_ptr<const PixelTopologyMap> pixelTopologyMap_;

  // input directory name
  std::string dirName_;

  // file-names
  const std::string millePedeEndFile_;
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
  const bool isHG_;

  // stores in a compact format the 4 decisions:
  // 1st bit: exceeds maximum thresholds
  // 2nd bit: exceeds cutoffs (significant movement)
  // 3rd bit: exceeds maximum errors
  // 4th bit: is below the significance
  std::bitset<4> updateBits_;

  // pede binaries available
  int binariesAmount_{0};

  // Fraction threshold booleans for HG alignment
  std::map<std::string, std::array<bool, 6>> fractionExceeded_;

  int Nrec_{0};
  int exitCode_{-1};
  std::string exitMessage_{""};

  std::array<double, SIZE_LG_STRUCTS> Xobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> XobsErr_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tXobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tXobsErr_ = std::array<double, SIZE_LG_STRUCTS>();

  std::array<double, SIZE_LG_STRUCTS> Yobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> YobsErr_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tYobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tYobsErr_ = std::array<double, SIZE_LG_STRUCTS>();

  std::array<double, SIZE_LG_STRUCTS> Zobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> ZobsErr_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tZobs_ = std::array<double, SIZE_LG_STRUCTS>();
  std::array<double, SIZE_LG_STRUCTS> tZobsErr_ = std::array<double, SIZE_LG_STRUCTS>();

  std::array<double, SIZE_HG_STRUCTS> Xobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> XobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tXobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tXobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();

  std::array<double, SIZE_HG_STRUCTS> Yobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> YobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tYobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tYobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();

  std::array<double, SIZE_HG_STRUCTS> Zobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> ZobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tZobs_HG_ = std::array<double, SIZE_HG_STRUCTS>();
  std::array<double, SIZE_HG_STRUCTS> tZobsErr_HG_ = std::array<double, SIZE_HG_STRUCTS>();

  std::unordered_map<PclHLS, std::pair<int, int>> indexHelper;
};

const std::array<std::string, 8> coord_str = {{"X", "Y", "Z", "theta_X", "theta_Y", "theta_Z", "extra_DOF", "none"}};
inline std::ostream& operator<<(std::ostream& os, const AlignPCLThresholdsHG::coordType& c) {
  if (c >= AlignPCLThresholdsHG::endOfTypes || c < AlignPCLThresholdsHG::X)
    return os << "unrecongnized coordinate";
  return os << coord_str[c];
}

#endif /* ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_ */
