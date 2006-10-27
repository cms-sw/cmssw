#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/EventProvenance.h"
#include "DataFormats/Common/interface/FileFormatVersion.h"
#include "DataFormats/Common/interface/Hash.h"
#include "DataFormats/Common/interface/HashedTypes.h"
#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ModuleDescriptionID.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"
#include "DataFormats/Common/interface/ProcessConfigurationID.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/RunAux.h"
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "DataFormats/Common/interface/RunBlock.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <map>
#include <vector>

namespace { namespace {
  edm::Wrapper<std::vector<unsigned long> > dummy1;
  edm::Wrapper<std::vector<unsigned int> > dummy2;
  edm::Wrapper<std::vector<long> > dummy3;
  edm::Wrapper<std::vector<int> > dummy4;
  edm::Wrapper<std::vector<std::string> > dummy5;
  edm::Wrapper<std::vector<char> > dummy6;
  edm::Wrapper<std::vector<unsigned char> > dummy7;
  edm::Wrapper<std::vector<short> > dummy8;
  edm::Wrapper<std::vector<unsigned short> > dummy9;
  edm::Wrapper<std::vector<double> > dummy10;
  edm::Wrapper<std::vector<long double> > dummy11;
  edm::Wrapper<std::vector<float> > dummy12;
  edm::Wrapper<std::vector<bool> > dummy13;

  edm::Wrapper<std::pair<unsigned long, unsigned long> > dymmywp1;
  edm::Wrapper<std::pair<unsigned int, unsigned int> > dymmywp2;
  edm::Wrapper<std::pair<unsigned short, unsigned short> > dymmywp3;
  edm::Wrapper<std::map<unsigned long, unsigned long> > dymmywm1;
  edm::Wrapper<std::map<unsigned int, unsigned int> > dymmywm2;
  edm::Wrapper<std::map<unsigned short, unsigned short> > dymmypwm3;
  edm::Wrapper<std::map<unsigned long, std::vector<unsigned long> > > dymmywmv1;
  edm::Wrapper<std::map<unsigned int, std::vector<unsigned int> > > dymmywmv2;
  edm::Wrapper<std::map<unsigned short, std::vector<unsigned short> > > dymmypwmv3;

  edm::Wrapper<unsigned long> dummyw1;
  edm::Wrapper<unsigned int> dummyw2;
  edm::Wrapper<long> dummyw3;
  edm::Wrapper<int> dummyw4;
  edm::Wrapper<std::string> dummyw5;
  edm::Wrapper<char> dummyw6;
  edm::Wrapper<unsigned char> dummyw7;
  edm::Wrapper<short> dummyw8;
  edm::Wrapper<unsigned short> dummyw9;
  edm::Wrapper<double> dummyw10;
  edm::Wrapper<long double> dummyw11;
  edm::Wrapper<float> dummyw12;
  edm::Wrapper<bool> dummyw13;

  edm::Wrapper<edm::HLTPathStatus> dummyx16;
  edm::Wrapper<std::vector<edm::HLTPathStatus> > dummyx17;
  edm::Wrapper<edm::HLTGlobalStatus> dummyx18;
  edm::Wrapper<edm::TriggerResults> dummyx19;

  edm::RefItem<unsigned int> dummyRefItem1;
  edm::RefItem<int> dummyRefItem3;
  edm::RefItem<std::pair<unsigned int, unsigned int> > dummyRefItem2;
  edm::RefBase<std::vector<unsigned int>::size_type> dummRefBase1;
  edm::RefBase<std::pair<unsigned int, unsigned int> > dummRefBase2;
  edm::RefBase<int> dummyRefBase3;
  edm::RefVectorBase<std::vector<unsigned int>::size_type> dummyRefVectorBase;
  edm::RefVectorBase<int> dummyRefVectorBase2;

  std::pair<edm::BranchKey, edm::BranchDescription> dummyPairBranch;
  std::map<edm::Hash<0>, edm::ModuleDescription> dummyMapMod;
  std::map<edm::Hash<2>, edm::ProcessHistory> dummyMapProc;
  std::map<edm::Hash<1>, edm::ParameterSetBlob> dummyMapParam;
  std::set<edm::Hash<1> > dummySetParam;
  std::set<edm::Hash<3> > dummySetProcessDesc;
  std::pair<edm::Hash<0>, edm::ModuleDescription> dummyPairMod;
  std::pair<edm::Hash<2>, edm::ProcessHistory> dummyPairProc;
  std::pair<edm::Hash<1>, edm::ParameterSetBlob> dummyPairParam;
}}
