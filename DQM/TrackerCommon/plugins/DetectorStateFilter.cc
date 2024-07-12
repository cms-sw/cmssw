#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <type_traits>  // for std::is_same

class DetectorStateFilter : public edm::stream::EDFilter<> {
public:
  DetectorStateFilter(const edm::ParameterSet&);
  ~DetectorStateFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;

  const bool verbose_;
  uint64_t nEvents_, nSelectedEvents_;
  bool detectorOn_;
  const std::string detectorType_;
  const std::vector<std::string> combinations_;  // Vector of strings specifying accepted combinations
  const edm::EDGetTokenT<DcsStatusCollection> dcsStatusLabel_;
  const edm::EDGetTokenT<DCSRecord> dcsRecordToken_;

  template <typename T>
  bool checkSubdet(const T& DCS, const int index);
  template <typename T>
  bool checkDCS(const T& DCS);
  template <typename T>
  bool checkDCSCombinations(const T& DCS, const std::vector<std::string>& combinations);

  bool checkDCSStatus(const DcsStatusCollection& dcsStatus);
  bool checkDCSRecord(const DCSRecord& dcsRecord);
};

//
// auxilliary enum
//
namespace DetStateFilter {
  enum parts { BPix = 0, FPix = 1, TIBTID = 2, TOB = 3, TECp = 4, TECm = 5, Invalid };

  // Map from string to enum
  parts partNameToEnum(const std::string& partName) {
    if (partName == "BPix")
      return BPix;
    if (partName == "FPix")
      return FPix;
    if (partName == "TIBTID")
      return TIBTID;
    if (partName == "TOB")
      return TOB;
    if (partName == "TECp")
      return TECp;
    if (partName == "TECm")
      return TECm;
    return Invalid;
  }

  // Single function to parse and split the vector of strings
  std::vector<std::vector<std::string>> parseAndSplit(const std::vector<std::string>& input, char delimiter) {
    std::vector<std::vector<std::string>> parsedResult;

    for (const auto& str : input) {
      std::vector<std::string> splitStrings;
      std::stringstream ss(str);
      std::string item;

      while (std::getline(ss, item, delimiter)) {
        splitStrings.push_back(item);
      }

      parsedResult.push_back(splitStrings);
    }

    return parsedResult;
  }
}  // namespace DetStateFilter

//
// -- Constructor
//
DetectorStateFilter::DetectorStateFilter(const edm::ParameterSet& pset)
    : verbose_(pset.getUntrackedParameter<bool>("DebugOn", false)),
      detectorType_(pset.getUntrackedParameter<std::string>("DetectorType", "sistrip")),
      combinations_(pset.getUntrackedParameter<std::vector<std::string>>("acceptedCombinations")),
      dcsStatusLabel_(consumes<DcsStatusCollection>(
          pset.getUntrackedParameter<edm::InputTag>("DcsStatusLabel", edm::InputTag("scalersRawToDigi")))),
      dcsRecordToken_(consumes<DCSRecord>(
          pset.getUntrackedParameter<edm::InputTag>("DCSRecordLabel", edm::InputTag("onlineMetaDataDigis")))) {
  nEvents_ = 0;
  nSelectedEvents_ = 0;
  detectorOn_ = false;
}

//
// -- Destructor
//
DetectorStateFilter::~DetectorStateFilter() = default;

template <typename T>
//*********************************************************************//
bool DetectorStateFilter::checkSubdet(const T& DCS, const int index)
//*********************************************************************//
{
  std::vector<int> dcsStatusParts = {
      DcsStatus::BPIX, DcsStatus::FPIX, DcsStatus::TIBTID, DcsStatus::TOB, DcsStatus::TECp, DcsStatus::TECm};

  std::vector<DCSRecord::Partition> dcsRecordParts = {DCSRecord::Partition::BPIX,
                                                      DCSRecord::Partition::FPIX,
                                                      DCSRecord::Partition::TIBTID,
                                                      DCSRecord::Partition::TOB,
                                                      DCSRecord::Partition::TECp,
                                                      DCSRecord::Partition::TECm};

  if constexpr (std::is_same_v<T, DcsStatusCollection>) {
    return (DCS)[0].ready(dcsStatusParts[index]);
  } else if constexpr (std::is_same_v<T, DCSRecord>) {
    return DCS.highVoltageReady(dcsRecordParts[index]);
  } else {
    edm::LogError("DetectorStatusFilter")
        << __FILE__ << " " << __LINE__ << " passed a wrong object type, cannot deduce DCS information.\n"
        << " returning true" << std::endl;
    return true;
  }
}

template <typename T>
bool
//*********************************************************************//
DetectorStateFilter::checkDCS(const T& DCS)
//*********************************************************************//
{
  bool accepted = false;
  if (detectorType_ == "pixel") {
    if (checkSubdet(DCS, DetStateFilter::BPix) && checkSubdet(DCS, DetStateFilter::FPix)) {
      accepted = true;
      nSelectedEvents_++;
    } else {
      accepted = false;
    }
    if (verbose_) {
      edm::LogInfo("DetectorStatusFilter")
          << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << " DCS States : "
          << " BPix " << checkSubdet(DCS, DetStateFilter::BPix) << " FPix " << checkSubdet(DCS, DetStateFilter::FPix)
          << " Detector State " << accepted << std::endl;
    }
  } else if (detectorType_ == "sistrip") {
    if (checkSubdet(DCS, DetStateFilter::TIBTID) && checkSubdet(DCS, DetStateFilter::TOB) &&
        checkSubdet(DCS, DetStateFilter::TECp) && checkSubdet(DCS, DetStateFilter::TECm)) {
      accepted = true;
      nSelectedEvents_++;
    } else {
      accepted = false;
    }
    if (verbose_) {
      edm::LogInfo("DetectorStatusFilter")
          << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << " DCS States : "
          << " TEC- " << checkSubdet(DCS, DetStateFilter::TECm) << " TEC+ " << checkSubdet(DCS, DetStateFilter::TECp)
          << " TIB/TID " << checkSubdet(DCS, DetStateFilter::TIBTID) << " TOB " << checkSubdet(DCS, DetStateFilter::TOB)
          << " Detector States " << accepted << std::endl;
    }
  } else {
    throw cms::Exception("Wrong Configuration")
        << "Stated DetectorType '" << detectorType_
        << "' is neither 'pixel' or 'sistrip', please check your configuration!";
  }
  return accepted;
}

template <typename T>
bool
//*********************************************************************//
DetectorStateFilter::checkDCSCombinations(const T& DCS, const std::vector<std::string>& combinations)
//*********************************************************************//
{
  // check that the configuration is sound
  if (detectorType_ != "pixel" && detectorType_ != "sistrip") {
    throw cms::Exception("Wrong Configuration")
        << "Stated DetectorType '" << detectorType_
        << "' is neither 'pixel' or 'sistrip', please check your configuration!";
  }

  bool accepted = false;

  // first get the combinations to check
  std::vector<std::vector<std::string>> vec_to_check = DetStateFilter::parseAndSplit(combinations, '+');

  if (verbose_) {
    edm::LogInfo("DetectorStatusFilter") << "Debug Mode: Printing all possible combinations";
    for (const auto& combination : vec_to_check) {
      std::string combinationStr;
      for (const auto& part : combination) {
        if (!combinationStr.empty()) {
          combinationStr += " + ";
        }
        combinationStr += part;
      }
      edm::LogInfo("DetectorStatusFilter") << "Combination: " << combinationStr;
    }
  }

  // Initialize a vector<bool> to store the pass results
  std::vector<bool> bitset(vec_to_check.size(), false);
  for (size_t i = 0; i < vec_to_check.size(); ++i) {
    const auto& subdetectors = vec_to_check[i];
    std::vector<DetStateFilter::parts> partsToCheck;
    partsToCheck.reserve(subdetectors.size());

    // fill vector of parts to check
    for (const auto& sub : subdetectors) {
      DetStateFilter::parts partEnum = DetStateFilter::partNameToEnum(sub);
      if (partEnum == DetStateFilter::Invalid) {
        throw cms::Exception("InvalidSubdetector", "Subdetector name '" + sub + "' is invalid.");
      }
      partsToCheck.push_back(partEnum);
    }

    if (detectorType_ == "pixel") {
      for (const auto& part : partsToCheck) {
        if (part >= DetStateFilter::TIBTID) {
          throw cms::Exception("InvalidSubdetector", "Detector type 'pixel' cannot have partitions TIBTID or larger");
        }
      }
    } else if (detectorType_ == "sistrip") {
      for (const auto& part : partsToCheck) {
        if (part < DetStateFilter::TIBTID) {
          throw cms::Exception("InvalidSubdetector",
                               "Detector type 'strip' cannot have partitions smaller than TIBTID");
        }
      }
    }

    // Use std::all_of to compute the logical AND of checkSubdet(DCS, part)
    bool passes = std::all_of(partsToCheck.begin(), partsToCheck.end(), [this, &DCS](DetStateFilter::parts part) {
      return checkSubdet(DCS, part);
    });

    // Set the corresponding bit in bitset
    bitset[i] = passes;
  }

  // Set the value of accepted to the OR of all the bits in the bitset
  accepted = std::any_of(bitset.begin(), bitset.end(), [](bool bit) { return bit; });

  if (accepted)
    nSelectedEvents_++;

  if (detectorType_ == "pixel") {
    if (verbose_) {
      edm::LogInfo("DetectorStatusFilter")
          << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << " DCS States : "
          << " BPix " << checkSubdet(DCS, DetStateFilter::BPix) << " FPix " << checkSubdet(DCS, DetStateFilter::FPix)
          << " Detector State " << accepted << std::endl;
    }
  } else if (detectorType_ == "sistrip") {
    if (verbose_) {
      edm::LogInfo("DetectorStatusFilter")
          << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << " DCS States : "
          << " TEC- " << checkSubdet(DCS, DetStateFilter::TECm) << " TEC+ " << checkSubdet(DCS, DetStateFilter::TECp)
          << " TIB/TID " << checkSubdet(DCS, DetStateFilter::TIBTID) << " TOB " << checkSubdet(DCS, DetStateFilter::TOB)
          << " Detector States " << accepted << std::endl;
    }
  }

  return accepted;
}

//*********************************************************************//
bool DetectorStateFilter::filter(edm::Event& evt, edm::EventSetup const& es)
//*********************************************************************//
{
  nEvents_++;
  // Check Detector state Only for Real Data and return true for MC
  if (evt.isRealData()) {
    edm::Handle<DcsStatusCollection> dcsStatus;
    evt.getByToken(dcsStatusLabel_, dcsStatus);
    edm::Handle<DCSRecord> dcsRecord;
    evt.getByToken(dcsRecordToken_, dcsRecord);

    if (dcsStatus.isValid() && !dcsStatus->empty()) {
      // if the old style DCS status is valid (Run1 + Run2)
      if (combinations_.empty()) {
        detectorOn_ = checkDCS(*dcsStatus);
      } else {
        detectorOn_ = checkDCSCombinations(*dcsStatus, combinations_);
      }
    } else if (dcsRecord.isValid()) {
      // in case of real data check for DCSRecord content (Run >=3)
      if (combinations_.empty()) {
        detectorOn_ = checkDCS(*dcsRecord);
      } else {
        detectorOn_ = checkDCSCombinations(*dcsRecord, combinations_);
      }
    } else {
      edm::LogError("DetectorStatusFilter")
          << "Error! can't get the products, neither DCSRecord, nor scalersRawToDigi: accept in any case!";
      detectorOn_ = true;
    }
  } else {
    detectorOn_ = true;
    nSelectedEvents_++;
    if (verbose_) {
      edm::LogInfo("DetectorStatusFilter") << "Total MC Events " << nEvents_ << " Selected Events " << nSelectedEvents_
                                           << " Detector States " << detectorOn_ << std::endl;
    }
  }
  return detectorOn_;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DetectorStateFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("filters on the HV status of the Tracker (either pixels or strips)");
  desc.addUntracked<bool>("DebugOn", false)->setComment("activates debugging");
  desc.addUntracked<std::string>("DetectorType", "sistrip")->setComment("either strips or pixels");
  desc.addUntracked<std::vector<std::string>>("acceptedCombinations", {});
  desc.addUntracked<edm::InputTag>("DcsStatusLabel", edm::InputTag("scalersRawToDigi"))
      ->setComment("event data for DCS (Run2)");
  desc.addUntracked<edm::InputTag>("DCSRecordLabel", edm::InputTag("onlineMetaDataDigis"))
      ->setComment("event data for DCS (Run3)");
  descriptions.add("_detectorStateFilter", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DetectorStateFilter);
