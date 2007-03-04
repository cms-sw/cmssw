#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/Hash.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <map>
#include <set>
#include <vector>

namespace {
  struct dictionary {
  std::pair<edm::BranchKey, edm::BranchDescription> dummyPairBranch;
  std::map<edm::Hash<0>, edm::ModuleDescription> dummyMapMod;
  std::map<edm::Hash<2>, edm::ProcessHistory> dummyMapProc;
  std::map<edm::Hash<1>, edm::ParameterSetBlob> dummyMapParam;
  std::set<edm::Hash<1> > dummySetParam;
  std::set<edm::Hash<3> > dummySetProcessDesc;
  std::pair<edm::Hash<0>, edm::ModuleDescription> dummyPairMod;
  std::pair<edm::Hash<2>, edm::ProcessHistory> dummyPairProc;
  std::pair<edm::Hash<1>, edm::ParameterSetBlob> dummyPairParam;
  std::vector<edm::ProductID> dummyVectorProductID;
  std::vector<std::basic_string<char> > dummyVectorString;
  std::set<std::basic_string<char> > dummySetString;
};
}
