#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/Hash.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/Transient.h"
#include <map>
#include <set>
#include <vector>

// These below are for backward compatibility only
// Note: ModuleDescription is still used, but is no longer persistent
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"


namespace {
  struct dictionary {
  std::pair<edm::BranchKey, edm::BranchDescription> dummyPairBranch;
  std::map<edm::ParameterSetID, edm::ParameterSetBlob> dummyMapParam;
  std::map<edm::ProcessHistoryID, edm::ProcessHistory> dummyMapProcH;
  std::map<edm::ProcessConfigurationID, edm::ProcessConfiguration> dummyMapProcC;
  std::map<edm::ProcessConfigurationID, edm::ParameterSetID> dummyMapParamSetID;
  std::map<edm::ProcessConfigurationID, std::string> dummyMapModuleName;
  std::set<edm::ProcessHistoryID> dummySetProcH;
  std::pair<edm::ParameterSetID, edm::ParameterSetBlob> dummyPairParam;
  std::pair<edm::ProcessHistoryID, edm::ProcessHistory> dummyPairProcH;
  std::pair<edm::ProcessConfigurationID, edm::ProcessConfiguration> dummyPairProcC;
  std::pair<edm::ProcessConfigurationID, edm::ParameterSetID> dummyPairParamSetID;
  std::pair<edm::ProcessConfigurationID, std::string> dummyPairModuleName;
  edm::ParentageID dummyParentageID;
  std::vector<edm::ProductID> dummyVectorProductID;
  std::vector<edm::BranchID> dummyVectorBranchID;
  std::set<edm::BranchID> dummySetBranchID;
  std::map<edm::BranchID, std::set<edm::BranchID> > dummyMapSetBranchID;
  std::pair<edm::BranchID, std::set<edm::BranchID> > dummyPairSetBranchID;
  std::vector<std::basic_string<char> > dummyVectorString;
  std::set<std::basic_string<char> > dummySetString;
  std::vector<edm::EventID> dummyVectorEventID;
  std::vector<std::vector<edm::EventID> > dummyVectorVectorEventID;
  std::vector<edm::ProductProvenance> dummyVectorProductProvenance;
  std::vector<std::vector<edm::ParameterSetID> > dummyVectorVectorParameterSetID;

  // The remaining ones are for backward compatibility only.
  std::map<edm::ModuleDescriptionID, edm::ModuleDescription> dummyMapMod;
  std::pair<edm::ModuleDescriptionID, edm::ModuleDescription> dummyPairMod;
  std::vector<edm::EventProcessHistoryID> dummyEventProcessHistory;
  edm::EntryDescriptionID dummyEntryDescriptionID;
  std::vector<edm::EventEntryInfo> dummyVectorEventEntryInfo;
  std::vector<edm::RunLumiEntryInfo> dummyVectorRunLumiEntryInfo;
};
}
