#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/Hash.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAux.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <map>
#include <set>
#include <vector>

namespace {
  struct dictionary {
  std::pair<edm::BranchKey, edm::BranchDescription> dummyPairBranch;
  std::map<edm::Hash<0>, edm::ModuleDescription> dummyMapMod;
  std::map<edm::Hash<1>, edm::ParameterSetBlob> dummyMapParam;
  std::map<edm::Hash<2>, edm::ProcessHistory> dummyMapProcH;
  std::map<edm::Hash<3>, edm::ProcessConfiguration> dummyMapProcC;
  std::set<edm::Hash<1> > dummySetParam;
  std::set<edm::Hash<3> > dummySetProcessDesc;
  std::pair<edm::Hash<0>, edm::ModuleDescription> dummyPairMod;
  std::pair<edm::Hash<1>, edm::ParameterSetBlob> dummyPairParam;
  std::pair<edm::Hash<2>, edm::ProcessHistory> dummyPairProcH;
  std::pair<edm::Hash<3>, edm::ProcessConfiguration> dummyPairProcC;
  std::vector<edm::ProductID> dummyVectorProductID;
  std::vector<edm::BranchID> dummyVectorBranchID;
  std::set<edm::BranchID> dummySetBranchID;
  std::map<edm::BranchID, std::set<edm::BranchID> > dummyMapSetBranchID;
  std::pair<edm::BranchID, std::set<edm::BranchID> > dummyPairSetBranchID;
  std::vector<std::basic_string<char> > dummyVectorString;
  std::set<std::basic_string<char> > dummySetString;
  std::vector<edm::EventProcessHistoryID> dummyEventProcessHistory;
  std::vector<edm::EventID> dummyVectorEventID;
  std::vector<std::vector<edm::EventID> > dummyVectorVectorEventID;
  edm::Hash<4> dummyEntryDescriptionID;
  std::vector<edm::EventEntryInfo> dummyVectorEventEntryInfo;
  std::vector<edm::RunLumiEntryInfo> dummyVectorRunLumiEntryInfo;
};
}
