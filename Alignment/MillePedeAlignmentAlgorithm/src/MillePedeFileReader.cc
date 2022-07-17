/*** Header file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

/*** system includes ***/
#include <cmath>  // include floating-point std::abs functions
#include <fstream>

/*** Alignment ***/
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

MillePedeFileReader ::MillePedeFileReader(const edm::ParameterSet& config,
                                          const std::shared_ptr<const PedeLabelerBase>& pedeLabeler,
                                          const std::shared_ptr<const AlignPCLThresholdsHG>& theThresholds,
                                          const std::shared_ptr<const PixelTopologyMap>& pixelTopologyMap)
    : pedeLabeler_(pedeLabeler),
      theThresholds_(theThresholds),
      pixelTopologyMap_(pixelTopologyMap),
      dirName_(config.getParameter<std::string>("fileDir")),
      millePedeEndFile_(config.getParameter<std::string>("millePedeEndFile")),
      millePedeLogFile_(config.getParameter<std::string>("millePedeLogFile")),
      millePedeResFile_(config.getParameter<std::string>("millePedeResFile")),
      isHG_(config.getParameter<bool>("isHG")) {
  if (!dirName_.empty() && dirName_.find_last_of('/') != dirName_.size() - 1)
    dirName_ += '/';  // may need '/'
}

void MillePedeFileReader ::read() {
  if (isHG_) {
    initializeIndexHelper();
  }
  readMillePedeEndFile();
  readMillePedeLogFile();
  readMillePedeResultFile();
}

bool MillePedeFileReader ::storeAlignments() { return (updateDB_ && !vetoUpdateDB_); }

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================
void MillePedeFileReader ::readMillePedeEndFile() {
  std::ifstream endFile;
  endFile.open((dirName_ + millePedeEndFile_).c_str());

  if (endFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede end-file";
    std::string line;
    getline(endFile, line);
    std::string trash;
    if (line.find("-1") != std::string::npos) {
      getline(endFile, line);
      exitMessage_ = line;
      std::istringstream iss(line);
      iss >> exitCode_ >> trash;
      edm::LogInfo("MillePedeFileReader")
          << " Pede exit code is: " << exitCode_ << " (" << exitMessage_ << ")" << std::endl;
    } else {
      exitMessage_ = line;
      std::istringstream iss(line);
      iss >> exitCode_ >> trash;
      edm::LogInfo("MillePedeFileReader")
          << " Pede exit code is: " << exitCode_ << " (" << exitMessage_ << ")" << std::endl;
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede end-file.";
    exitMessage_ = "no exit code found";
  }
}

void MillePedeFileReader ::readMillePedeLogFile() {
  std::ifstream logFile;
  logFile.open((dirName_ + millePedeLogFile_).c_str());

  if (logFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede log-file";
    std::string line;

    while (getline(logFile, line)) {
      std::string Nrec_string = "NREC =";
      std::string Binaries_string = "C_binary";

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

      if (line.find(Binaries_string) != std::string::npos) {
        binariesAmount_ += 1;
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
  std::map<std::string, std::array<float, 6> > fractions_;

  std::map<std::string, std::array<int, 6> > countsAbove_;
  std::map<std::string, std::array<int, 6> > countsTotal_;

  AlignableObjectId alignableObjectId{AlignableObjectId::Geometry::General};

  std::vector<std::string> alignables_ = theThresholds_->getAlignableList();
  for (auto& ali : alignables_) {
    cutoffs_[ali] = theThresholds_->getCut(ali);
    significances_[ali] = theThresholds_->getSigCut(ali);
    thresholds_[ali] = theThresholds_->getMaxMoveCut(ali);
    errors_[ali] = theThresholds_->getMaxErrorCut(ali);

    if (theThresholds_->hasFloatMap(ali)) {
      fractions_[ali] = theThresholds_->getFractionCut(ali);
      countsAbove_[ali] = {{0, 0, 0, 0, 0, 0}};
      countsTotal_[ali] = {{0, 0, 0, 0, 0, 0}};
    }
  }

  updateDB_ = false;
  vetoUpdateDB_ = false;
  std::ifstream resFile;
  resFile.open((dirName_ + millePedeResFile_).c_str());

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

      auto alignableLabel = std::stoul(tokens[0]);
      const auto alignable = pedeLabeler_->alignableFromLabel(alignableLabel);
      auto det = getHLS(alignable);
      int detIndex = static_cast<int>(det);
      auto alignableIndex = alignableLabel % 10 - 1;
      std::string detLabel = getStringFromHLS(det);

      countsTotal_[detLabel][alignableIndex]++;

      if (tokens.size() > 4 /*3*/) {
        const auto paramNum = pedeLabeler_->paramNumFromLabel(alignableLabel);
        align::StructureType type = alignable->alignableObjectId();
        align::ID id = alignable->id();

        double ObsMove = std::stof(tokens[3]) * multiplier_[alignableIndex];
        double ObsErr = std::stof(tokens[4]) * multiplier_[alignableIndex];

        auto coord = static_cast<AlignPCLThresholdsHG::coordType>(alignableIndex);

        if (det != PclHLS::NotInPCL) {
          if (type != align::TPBLadder && type != align::TPEPanel) {
            switch (coord) {
              case AlignPCLThresholdsHG::X:
                Xobs_[detIndex] = ObsMove;
                XobsErr_[detIndex] = ObsErr;
                break;
              case AlignPCLThresholdsHG::Y:
                Yobs_[detIndex] = ObsMove;
                YobsErr_[detIndex] = ObsErr;
                break;
              case AlignPCLThresholdsHG::Z:
                Zobs_[detIndex] = ObsMove;
                ZobsErr_[detIndex] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_X:
                tXobs_[detIndex] = ObsMove;
                tXobsErr_[detIndex] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_Y:
                tYobs_[detIndex] = ObsMove;
                tYobsErr_[detIndex] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_Z:
                tZobs_[detIndex] = ObsMove;
                tZobsErr_[detIndex] = ObsErr;
                break;
              default:
                edm::LogError("MillePedeFileReader") << "Currently not able to handle DOF " << coord << std::endl;
                break;
            }
          } else {
            auto hgIndex = getIndexForHG(id, det);
            switch (coord) {
              case AlignPCLThresholdsHG::X:
                Xobs_HG_[hgIndex - 1] = ObsMove;
                XobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              case AlignPCLThresholdsHG::Y:
                Yobs_HG_[hgIndex - 1] = ObsMove;
                YobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              case AlignPCLThresholdsHG::Z:
                Zobs_HG_[hgIndex - 1] = ObsMove;
                ZobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_X:
                tXobs_HG_[hgIndex - 1] = ObsMove;
                tXobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_Y:
                tYobs_HG_[hgIndex - 1] = ObsMove;
                tYobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              case AlignPCLThresholdsHG::theta_Z:
                tZobs_HG_[hgIndex - 1] = ObsMove;
                tZobsErr_HG_[hgIndex - 1] = ObsErr;
                break;
              default:
                edm::LogError("MillePedeFileReader") << "Currently not able to handle DOF " << coord << std::endl;
                break;
            }
          }

        } else {
          edm::LogError("MillePedeFileReader")
              << "Currently not able to handle coordinate: " << coord << " (" << paramNum << ")  "
              << Form(" %s with ID %d (subdet %d)", alignableObjectId.idToString(type), id, DetId(id).subdetId())
              << std::endl;
          continue;
        }

        edm::LogVerbatim("MillePedeFileReader")
            << " alignableLabel: " << alignableLabel << " with alignableIndex " << alignableIndex << " detIndex "
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
            << " fraction (fractions_[" << detLabel << "][" << coord << "]): " << fractions_[detLabel][alignableIndex]
            << "\n"
            << "=============" << std::endl;

        if (std::abs(ObsMove) > thresholds_[detLabel][alignableIndex]) {
          if (!isHG_) {
            edm::LogWarning("MillePedeFileReader")
                << "Aborting payload creation."
                << " Exceeding maximum thresholds for movement: " << std::abs(ObsMove) << " for" << detLabel << "("
                << coord << ")";
          }
          updateBits_.set(0);
          vetoUpdateDB_ = true;
          continue;

        } else if (std::abs(ObsMove) > cutoffs_[detLabel][alignableIndex]) {
          updateBits_.set(1);

          if (std::abs(ObsErr) > errors_[detLabel][alignableIndex]) {
            if (!isHG_) {
              edm::LogWarning("MillePedeFileReader") << "Aborting payload creation."
                                                     << " Exceeding maximum thresholds for error: " << std::abs(ObsErr)
                                                     << " for" << detLabel << "(" << coord << ")";
            }
            updateBits_.set(2);
            vetoUpdateDB_ = true;
            continue;
          } else {
            if (std::abs(ObsMove / ObsErr) < significances_[detLabel][alignableIndex]) {
              updateBits_.set(3);
              continue;
            }
          }
          updateDB_ = true;
          if (!isHG_) {
            edm::LogInfo("MillePedeFileReader")
                << "This correction: " << ObsMove << "+/-" << ObsErr << " for " << detLabel << "(" << coord
                << ") will trigger a new Tracker Alignment payload!";
          }
          countsAbove_[detLabel][alignableIndex]++;
        }
      }
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede result-file.";

    updateDB_ = false;
    Nrec_ = 0;
  }

  if (isHG_) {          // check fractionCut
    updateDB_ = false;  // reset both booleans since only fractionCut is considered for HG
    vetoUpdateDB_ = false;
    std::stringstream ss;
    for (auto& ali : alignables_) {
      ss << ali << std::endl;
      for (long unsigned int i = 0; i < countsTotal_[ali].size(); i++) {
        if (countsTotal_[ali][i] != 0 && fractions_[ali][i] != -1) {
          float fraction_ = countsAbove_[ali][i] / (1.0 * countsTotal_[ali][i]);
          ss << static_cast<AlignPCLThresholdsHG::coordType>(i) << ":   Fraction = " << fraction_
             << "   Fraction Threshold = " << fractions_[ali][i];
          if (fraction_ >= fractions_[ali][i]) {
            updateDB_ = true;
            ss << "   above fraction threshold" << std::endl;
          } else
            ss << std::endl;
        } else
          ss << "No entries available or no fraction thresholds defined" << std::endl;
      }
      ss << "===================" << std::endl;
    }
    if (updateDB_) {
      ss << "Alignment will be updated" << std::endl;
    } else {
      ss << "Alignment will NOT be updated" << std::endl;
    }
    edm::LogWarning("MillePedeFileReader") << ss.str();
  }
}

MillePedeFileReader::PclHLS MillePedeFileReader ::getHLS(const Alignable* alignable) {
  if (!alignable)
    return PclHLS::NotInPCL;

  const auto& tns = pedeLabeler_->alignableTracker()->trackerNameSpace();
  const align::ID id = alignable->id();

  switch (alignable->alignableObjectId()) {
    case align::TPBHalfBarrel:
      switch (tns.tpb().halfBarrelNumber(id)) {
        case 1:
          return PclHLS::TPBHalfBarrelXminus;
        case 2:
          return PclHLS::TPBHalfBarrelXplus;
        default:
          throw cms::Exception("LogicError")
              << "@SUB=MillePedeFileReader::getHLS\n"
              << "Found a pixel half-barrel number that should not exist: " << tns.tpb().halfBarrelNumber(id);
      }
    case align::TPEHalfCylinder:
      switch (tns.tpe().endcapNumber(id)) {
        case 1:
          switch (tns.tpe().halfCylinderNumber(id)) {
            case 1:
              return PclHLS::TPEHalfCylinderXminusZminus;
            case 2:
              return PclHLS::TPEHalfCylinderXplusZminus;
            default:
              throw cms::Exception("LogicError")
                  << "@SUB=MillePedeFileReader::getHLS\n"
                  << "Found a pixel half-cylinder number that should not exist: " << tns.tpe().halfCylinderNumber(id);
          }
        case 2:
          switch (tns.tpe().halfCylinderNumber(id)) {
            case 1:
              return PclHLS::TPEHalfCylinderXminusZplus;
            case 2:
              return PclHLS::TPEHalfCylinderXplusZplus;
            default:
              throw cms::Exception("LogicError")
                  << "@SUB=MillePedeFileReader::getHLS\n"
                  << "Found a pixel half-cylinder number that should not exist: " << tns.tpe().halfCylinderNumber(id);
          }
        default:
          throw cms::Exception("LogicError")
              << "@SUB=MillePedeFileReader::getHLS\n"
              << "Found a pixel endcap number that should not exist: " << tns.tpe().endcapNumber(id);
      }
    case align::TPBLadder:
      switch (tns.tpb().layerNumber(id)) {
        case 1:
          return PclHLS::TPBLadderLayer1;
        case 2:
          return PclHLS::TPBLadderLayer2;
        case 3:
          return PclHLS::TPBLadderLayer3;
        case 4:
          return PclHLS::TPBLadderLayer4;
        default:
          throw cms::Exception("LogicError")
              << "@SUB=MillePedeFileReader::getHLS\n"
              << "Found a pixel layer number that should not exist: " << tns.tpb().layerNumber(id);
      }
    case align::TPEPanel:
      switch (static_cast<signed int>((tns.tpe().endcapNumber(id) == 1) ? -1 * tns.tpe().halfDiskNumber(id)
                                                                        : tns.tpe().halfDiskNumber(id))) {
        case -3:
          return PclHLS::TPEPanelDiskM3;
        case -2:
          return PclHLS::TPEPanelDiskM2;
        case -1:
          return PclHLS::TPEPanelDiskM1;
        case 3:
          return PclHLS::TPEPanelDisk3;
        case 2:
          return PclHLS::TPEPanelDisk2;
        case 1:
          return PclHLS::TPEPanelDisk1;
        default:
          throw cms::Exception("LogicError")
              << "@SUB=MillePedeFileReader::getHLS\n"
              << "Found a pixel disk number that should not exist: "
              << static_cast<signed int>((tns.tpe().endcapNumber(id) == 1) ? -1 * tns.tpe().halfDiskNumber(id)
                                                                           : tns.tpe().halfDiskNumber(id));
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
    case PclHLS::TPBLadderLayer1:
      return "TPBLadderLayer1";
    case PclHLS::TPBLadderLayer2:
      return "TPBLadderLayer2";
    case PclHLS::TPBLadderLayer3:
      return "TPBLadderLayer3";
    case PclHLS::TPBLadderLayer4:
      return "TPBLadderLayer4";
    case PclHLS::TPEPanelDisk1:
      return "TPEPanelDisk1";
    case PclHLS::TPEPanelDisk2:
      return "TPEPanelDisk2";
    case PclHLS::TPEPanelDisk3:
      return "TPEPanelDisk3";
    case PclHLS::TPEPanelDiskM1:
      return "TPEPanelDiskM1";
    case PclHLS::TPEPanelDiskM2:
      return "TPEPanelDiskM2";
    case PclHLS::TPEPanelDiskM3:
      return "TPEPanelDiskM3";
    default:
      //return "NotInPCL";
      throw cms::Exception("LogicError")
          << "@SUB=MillePedeFileReader::getStringFromHLS\n"
          << "Found an alignable structure not possible to map in the default AlignPCLThresholdsHG partitions";
  }
}

void MillePedeFileReader::initializeIndexHelper() {
  int currentSum = 0;

  indexHelper[PclHLS::TPBLadderLayer1] =
      std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXBLadders(1) / 2);
  currentSum += pixelTopologyMap_->getPXBLadders(1);
  indexHelper[PclHLS::TPBLadderLayer2] =
      std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXBLadders(2) / 2);
  currentSum += pixelTopologyMap_->getPXBLadders(2);
  indexHelper[PclHLS::TPBLadderLayer3] =
      std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXBLadders(3) / 2);
  currentSum += pixelTopologyMap_->getPXBLadders(3);
  indexHelper[PclHLS::TPBLadderLayer4] =
      std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXBLadders(4) / 2);
  currentSum += pixelTopologyMap_->getPXBLadders(4);

  indexHelper[PclHLS::TPEPanelDiskM3] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(-3));
  currentSum += pixelTopologyMap_->getPXFBlades(-3) * 2;
  indexHelper[PclHLS::TPEPanelDiskM2] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(-2));
  currentSum += pixelTopologyMap_->getPXFBlades(-2) * 2;
  indexHelper[PclHLS::TPEPanelDiskM1] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(-1));
  currentSum += pixelTopologyMap_->getPXFBlades(-1) * 2;

  indexHelper[PclHLS::TPEPanelDisk1] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(1));
  currentSum += pixelTopologyMap_->getPXFBlades(1) * 2;
  indexHelper[PclHLS::TPEPanelDisk2] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(2));
  currentSum += pixelTopologyMap_->getPXFBlades(2) * 2;
  indexHelper[PclHLS::TPEPanelDisk3] = std::make_pair(currentSum, currentSum + pixelTopologyMap_->getPXFBlades(3));
  currentSum += pixelTopologyMap_->getPXFBlades(3) * 2;
}

int MillePedeFileReader::getIndexForHG(align::ID id, PclHLS HLS) {
  const auto& tns = pedeLabeler_->alignableTracker()->trackerNameSpace();

  switch (HLS) {
    case PclHLS::TPBLadderLayer1:
      return (tns.tpb().halfBarrelNumber(id) == 1) ? tns.tpb().ladderNumber(id) + indexHelper[HLS].first
                                                   : tns.tpb().ladderNumber(id) + indexHelper[HLS].second;
    case PclHLS::TPBLadderLayer2:
      return (tns.tpb().halfBarrelNumber(id) == 1) ? tns.tpb().ladderNumber(id) + indexHelper[HLS].first
                                                   : tns.tpb().ladderNumber(id) + indexHelper[HLS].second;
    case PclHLS::TPBLadderLayer3:
      return (tns.tpb().halfBarrelNumber(id) == 1) ? tns.tpb().ladderNumber(id) + indexHelper[HLS].first
                                                   : tns.tpb().ladderNumber(id) + indexHelper[HLS].second;
    case PclHLS::TPBLadderLayer4:
      return (tns.tpb().halfBarrelNumber(id) == 1) ? tns.tpb().ladderNumber(id) + indexHelper[HLS].first
                                                   : tns.tpb().ladderNumber(id) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDisk1:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDisk2:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDisk3:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDiskM1:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDiskM2:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    case PclHLS::TPEPanelDiskM3:
      return (tns.tpe().halfCylinderNumber(id) == 1)
                 ? (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].first
                 : (tns.tpe().bladeNumber(id) * 2 - (tns.tpe().panelNumber(id) % 2)) + indexHelper[HLS].second;
    default:
      return -200;
  }
}

//=============================================================================
//===   STATIC CONST MEMBER DEFINITION                                      ===
//=============================================================================
constexpr std::array<double, 6> MillePedeFileReader::multiplier_;
