/*** Header file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"

/*** system includes ***/
#include <cmath>                // include floating-point std::abs functions
#include <fstream>

/*** core framework functionality ***/
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Alignment ***/
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"


//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

MillePedeFileReader
::MillePedeFileReader(const edm::ParameterSet& config,
                      const std::shared_ptr<const PedeLabelerBase>& pedeLabeler) :
  pedeLabeler_(pedeLabeler),
  millePedeLogFile_(config.getParameter<std::string>("millePedeLogFile")),
  millePedeResFile_(config.getParameter<std::string>("millePedeResFile")),

  sigCut_     (config.getParameter<double>("sigCut")),
  Xcut_       (config.getParameter<double>("Xcut")),
  tXcut_      (config.getParameter<double>("tXcut")),
  Ycut_       (config.getParameter<double>("Ycut")),
  tYcut_      (config.getParameter<double>("tYcut")),
  Zcut_       (config.getParameter<double>("Zcut")),
  tZcut_      (config.getParameter<double>("tZcut")),
  maxMoveCut_ (config.getParameter<double>("maxMoveCut")),
  maxErrorCut_(config.getParameter<double>("maxErrorCut"))
{
}

void MillePedeFileReader
::read() {
  readMillePedeLogFile();
  readMillePedeResultFile();
}

bool MillePedeFileReader
::storeAlignments() {
  return updateDB_;
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeFileReader
::readMillePedeLogFile()
{
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

        if (Nrec_ < 25000) {
          updateDB_   = false;
        }
      }
    }

  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede log-file.";

    updateDB_   = false;
    Nrec_ = 0;
  }
}

void MillePedeFileReader
::readMillePedeResultFile()
{
  updateDB_ = false;
  std::ifstream resFile;
  resFile.open(millePedeResFile_.c_str());

  if (resFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede result-file";

    std::string line;
    getline(resFile, line); // drop first line

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
        double ObsErr  = std::stof(tokens[4]) * multiplier_[alignableIndex];

        auto det = getHLS(alignable);

        if (det != PclHLS::NotInPCL) {
          switch (alignableIndex) {
          case 0:
            Xobs_[static_cast<int>(det)] = ObsMove;
            XobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          case 1:
            Yobs_[static_cast<int>(det)] = ObsMove;
            YobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          case 2:
            Zobs_[static_cast<int>(det)] = ObsMove;
            ZobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          case 3:
            tXobs_[static_cast<int>(det)] = ObsMove;
            tXobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          case 4:
            tYobs_[static_cast<int>(det)] = ObsMove;
            tYobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          case 5:
            tZobs_[static_cast<int>(det)] = ObsMove;
            tZobsErr_[static_cast<int>(det)] = ObsErr;
            break;
          }
        } else {
          continue;
        }

        if (std::abs(ObsMove) > maxMoveCut_) {
          updateDB_    = false;
          break;

        } else if (std::abs(ObsMove) > cutoffs_[alignableIndex]) {

          if (std::abs(ObsErr) > maxErrorCut_) {
            updateDB_    = false;
            break;
          } else {
            if (std::abs(ObsMove/ObsErr) < sigCut_) {
              continue;
            }
          }
          updateDB_ = true;
        }
      }
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede result-file.";

    updateDB_   = false;
    Nrec_ = 0;
  }
}


MillePedeFileReader::PclHLS MillePedeFileReader
::getHLS(const Alignable* alignable) {
  if (!alignable) return PclHLS::NotInPCL;

  const auto& tns = pedeLabeler_->alignableTracker()->trackerNameSpace();

  switch (alignable->alignableObjectId()) {
  case align::TPBHalfBarrel:
    switch (tns.tpb().halfBarrelNumber(alignable->id())) {
    case 1: return PclHLS::TPBHalfBarrelXminus;
    case 2: return PclHLS::TPBHalfBarrelXplus;
    default:
      throw cms::Exception("LogicError")
        << "@SUB=MillePedeFileReader::getHLS\n"
        << "Found a pixel half-barrel number that should not exist: "
        << tns.tpb().halfBarrelNumber(alignable->id());
    }
  case align::TPEHalfCylinder:
    switch (tns.tpe().endcapNumber(alignable->id())) {
    case 1:
      switch (tns.tpe().halfCylinderNumber(alignable->id())) {
      case 1: return PclHLS::TPEHalfCylinderXminusZminus;
      case 2: return PclHLS::TPEHalfCylinderXplusZminus;
      default:
        throw cms::Exception("LogicError")
          << "@SUB=MillePedeFileReader::getHLS\n"
          << "Found a pixel half-cylinder number that should not exist: "
          << tns.tpe().halfCylinderNumber(alignable->id());
      }
    case 2:
      switch (tns.tpe().halfCylinderNumber(alignable->id())) {
      case 1: return PclHLS::TPEHalfCylinderXminusZplus;
      case 2: return PclHLS::TPEHalfCylinderXplusZplus;
      default:
        throw cms::Exception("LogicError")
          << "@SUB=MillePedeFileReader::getHLS\n"
          << "Found a pixel half-cylinder number that should not exist: "
          << tns.tpe().halfCylinderNumber(alignable->id());
      }
    default:
      throw cms::Exception("LogicError")
        << "@SUB=MillePedeFileReader::getHLS\n"
        << "Found a pixel endcap number that should not exist: "
        << tns.tpe().endcapNumber(alignable->id());
    }
  default: return PclHLS::NotInPCL;
  }
}

//=============================================================================
//===   STATIC CONST MEMBER DEFINITION                                      ===
//=============================================================================
constexpr std::array<double, 6> MillePedeFileReader::multiplier_;
