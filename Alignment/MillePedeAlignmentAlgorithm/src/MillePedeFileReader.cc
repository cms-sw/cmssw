/*** Header file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"

/*** system includes ***/
#include <cmath>  // include floating-point std::abs functions
#include <fstream>

/*** core framework functionality ***/
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Alignment ***/
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

MillePedeFileReader ::MillePedeFileReader(const edm::ParameterSet& config,
                                          const std::shared_ptr<const PedeLabelerBase>& pedeLabeler,
                                          const std::shared_ptr<const AlignPCLThresholds>& theThresholds)
    : pedeLabeler_(pedeLabeler),
      theThresholds_(theThresholds),
      millePedeLogFile_(config.getParameter<std::string>("millePedeLogFile")),
      millePedeResFile_(config.getParameter<std::string>("millePedeResFile")) {}

void MillePedeFileReader ::read() {
  readMillePedeLogFile();
  readMillePedeResultFile();
}

bool MillePedeFileReader ::storeAlignments() { return (updateDB_ && !vetoUpdateDB_); }

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeFileReader ::readMillePedeLogFile() {
  std::ifstream logFile;
  logFile.open(millePedeLogFile_.c_str());

  if (logFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede log-file";
    std::string line;

    while (getline(logFile, line)) {
      std::string Nrec_string = "NREC =";

      if (line.find(Nrec_string) != std::string::npos) {
        std::istringstream iss(line);
        std::string trash;
        iss >> trash >> trash >> Nrec_;

        if (Nrec_ < theThresholds_->getNrecords()) {
          edm::LogInfo("MillePedeFileReader")
              << "Number of records used " << theThresholds_->getNrecords() << std::endl;
          updateDB_ = false;
        }
      }
    }

  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede log-file.";

    updateDB_ = false;
    Nrec_ = 0;
  }
}

void MillePedeFileReader ::readMillePedeResultFile() {
  // cutoffs by coordinate and by alignable
  std::map<std::string, std::array<float, 6> > cutoffs_;
  std::map<std::string, std::array<float, 6> > significances_;
  std::map<std::string, std::array<float, 6> > thresholds_;
  std::map<std::string, std::array<float, 6> > errors_;

  std::vector<std::string> alignables_ = theThresholds_->getAlignableList();
  for (auto& ali : alignables_) {
    cutoffs_[ali] = theThresholds_->getCut(ali);
    significances_[ali] = theThresholds_->getSigCut(ali);
    thresholds_[ali] = theThresholds_->getMaxMoveCut(ali);
    errors_[ali] = theThresholds_->getMaxErrorCut(ali);
  }

  updateDB_ = false;
  vetoUpdateDB_ = false;
  std::ifstream resFile;
  resFile.open(millePedeResFile_.c_str());

  if (resFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede result-file";

    std::string line;
    getline(resFile, line);  // drop first line

    while (getline(resFile, line)) {
      std::istringstream iss(line);

      std::vector<std::string> tokens;
      std::string token;
      while (iss >> token) {
        tokens.push_back(token);
      }

      if (tokens.size() > 4 /*3*/) {
        auto alignableLabel = std::stoul(tokens[0]);
        auto alignableIndex = alignableLabel % 10 - 1;
        const auto alignable = pedeLabeler_->alignableFromLabel(alignableLabel);

        double ObsMove = std::stof(tokens[3]) * multiplier_[alignableIndex];
        double ObsErr = std::stof(tokens[4]) * multiplier_[alignableIndex];

        auto det = getHLS(alignable);
        int detIndex = static_cast<int>(det);
        auto coord = static_cast<AlignPCLThresholds::coordType>(alignableIndex);
        std::string detLabel = getStringFromHLS(det);

        if (det != PclHLS::NotInPCL) {
          switch (coord) {
            case AlignPCLThresholds::X:
              Xobs_[detIndex] = ObsMove;
              XobsErr_[detIndex] = ObsErr;
              break;
            case AlignPCLThresholds::Y:
              Yobs_[detIndex] = ObsMove;
              YobsErr_[detIndex] = ObsErr;
              break;
            case AlignPCLThresholds::Z:
              Zobs_[detIndex] = ObsMove;
              ZobsErr_[detIndex] = ObsErr;
              break;
            case AlignPCLThresholds::theta_X:
              tXobs_[detIndex] = ObsMove;
              tXobsErr_[detIndex] = ObsErr;
              break;
            case AlignPCLThresholds::theta_Y:
              tYobs_[detIndex] = ObsMove;
              tYobsErr_[detIndex] = ObsErr;
              break;
            case AlignPCLThresholds::theta_Z:
              tZobs_[detIndex] = ObsMove;
              tZobsErr_[detIndex] = ObsErr;
              break;
            default:
              edm::LogError("MillePedeFileReader") << "Currently not able to handle DOF " << coord << std::endl;
              break;
          }
        } else {
          continue;
        }

        edm::LogVerbatim("MillePedeFileReader")
            << " alignableLabel: " << alignableLabel << " with alignableIndex " << alignableIndex << " detIndex"
            << detIndex << "\n"
            << " i.e. detLabel: " << detLabel << " (" << coord << ")\n"
            << " has movement: " << ObsMove << " +/- " << ObsErr << "\n"
            << " cutoff (cutoffs_[" << detLabel << "][" << coord << "]): " << cutoffs_[detLabel][alignableIndex] << "\n"
            << " significance (significances_[" << detLabel << "][" << coord
            << "]): " << significances_[detLabel][alignableIndex] << "\n"
            << " error thresolds (errors_[" << detLabel << "][" << coord << "]): " << errors_[detLabel][alignableIndex]
            << "\n"
            << " max movement (thresholds_[" << detLabel << "][" << coord
            << "]): " << thresholds_[detLabel][alignableIndex] << "\n"
            << "=============" << std::endl;

        if (std::abs(ObsMove) > thresholds_[detLabel][alignableIndex]) {
          edm::LogWarning("MillePedeFileReader") << "Aborting payload creation."
                                                 << " Exceeding maximum thresholds for movement: " << std::abs(ObsMove)
                                                 << " for" << detLabel << "(" << coord << ")";
          vetoUpdateDB_ = true;
          continue;

        } else if (std::abs(ObsMove) > cutoffs_[detLabel][alignableIndex]) {
          if (std::abs(ObsErr) > errors_[detLabel][alignableIndex]) {
            edm::LogWarning("MillePedeFileReader") << "Aborting payload creation."
                                                   << " Exceeding maximum thresholds for error: " << std::abs(ObsErr)
                                                   << " for" << detLabel << "(" << coord << ")";
            vetoUpdateDB_ = true;
            continue;
          } else {
            if (std::abs(ObsMove / ObsErr) < significances_[detLabel][alignableIndex]) {
              continue;
            }
          }
          updateDB_ = true;
          edm::LogInfo("MillePedeFileReader")
              << "This correction: " << ObsMove << "+/-" << ObsErr << " for " << detLabel << "(" << coord
              << ") will trigger a new Tracker Alignment payload!";
        }
      }
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede result-file.";

    updateDB_ = false;
    Nrec_ = 0;
  }
}

MillePedeFileReader::PclHLS MillePedeFileReader ::getHLS(const Alignable* alignable) {
  if (!alignable)
    return PclHLS::NotInPCL;

  const auto& tns = pedeLabeler_->alignableTracker()->trackerNameSpace();

  switch (alignable->alignableObjectId()) {
    case align::TPBHalfBarrel:
      switch (tns.tpb().halfBarrelNumber(alignable->id())) {
        case 1:
          return PclHLS::TPBHalfBarrelXminus;
        case 2:
          return PclHLS::TPBHalfBarrelXplus;
        default:
          throw cms::Exception("LogicError") << "@SUB=MillePedeFileReader::getHLS\n"
                                             << "Found a pixel half-barrel number that should not exist: "
                                             << tns.tpb().halfBarrelNumber(alignable->id());
      }
    case align::TPEHalfCylinder:
      switch (tns.tpe().endcapNumber(alignable->id())) {
        case 1:
          switch (tns.tpe().halfCylinderNumber(alignable->id())) {
            case 1:
              return PclHLS::TPEHalfCylinderXminusZminus;
            case 2:
              return PclHLS::TPEHalfCylinderXplusZminus;
            default:
              throw cms::Exception("LogicError") << "@SUB=MillePedeFileReader::getHLS\n"
                                                 << "Found a pixel half-cylinder number that should not exist: "
                                                 << tns.tpe().halfCylinderNumber(alignable->id());
          }
        case 2:
          switch (tns.tpe().halfCylinderNumber(alignable->id())) {
            case 1:
              return PclHLS::TPEHalfCylinderXminusZplus;
            case 2:
              return PclHLS::TPEHalfCylinderXplusZplus;
            default:
              throw cms::Exception("LogicError") << "@SUB=MillePedeFileReader::getHLS\n"
                                                 << "Found a pixel half-cylinder number that should not exist: "
                                                 << tns.tpe().halfCylinderNumber(alignable->id());
          }
        default:
          throw cms::Exception("LogicError")
              << "@SUB=MillePedeFileReader::getHLS\n"
              << "Found a pixel endcap number that should not exist: " << tns.tpe().endcapNumber(alignable->id());
      }
    default:
      return PclHLS::NotInPCL;
  }
}

std::string MillePedeFileReader::getStringFromHLS(MillePedeFileReader::PclHLS HLS) {
  switch (HLS) {
    case PclHLS::TPBHalfBarrelXminus:
      return "TPBHalfBarrelXminus";
    case PclHLS::TPBHalfBarrelXplus:
      return "TPBHalfBarrelXplus";
    case PclHLS::TPEHalfCylinderXminusZminus:
      return "TPEHalfCylinderXminusZminus";
    case PclHLS::TPEHalfCylinderXplusZminus:
      return "TPEHalfCylinderXplusZminus";
    case PclHLS::TPEHalfCylinderXminusZplus:
      return "TPEHalfCylinderXminusZplus";
    case PclHLS::TPEHalfCylinderXplusZplus:
      return "TPEHalfCylinderXplusZplus";
    default:
      throw cms::Exception("LogicError")
          << "@SUB=MillePedeFileReader::getStringFromHLS\n"
          << "Found an alignable structure not possible to map in the default AlignPCLThresholds partitions";
  }
}

//=============================================================================
//===   STATIC CONST MEMBER DEFINITION                                      ===
//=============================================================================
constexpr std::array<double, 6> MillePedeFileReader::multiplier_;
