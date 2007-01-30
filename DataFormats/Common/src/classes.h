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
#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "DataFormats/Common/interface/LuminosityBlockID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ModuleDescriptionID.h"
#include "DataFormats/Common/interface/OwnVector.h"
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
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <map>
#include <vector>
#include <list>
#include <deque>
#include <set>

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
  edm::Wrapper<std::vector<unsigned long long> > dummy14;
  edm::Wrapper<std::vector<long long> > dummy15;
  edm::Wrapper<std::vector<std::pair<std::basic_string<char>,double> > > dummy16;

  edm::Wrapper<std::list<int> > dummy17;

  edm::Wrapper<std::deque<int> > dummy18;

  edm::Wrapper<std::set<int> > dummy19;

  edm::Wrapper<std::pair<unsigned long, unsigned long> > dymmywp1;
  edm::Wrapper<std::pair<unsigned int, unsigned int> > dymmywp2;
  edm::Wrapper<std::pair<unsigned short, unsigned short> > dymmywp3;
  edm::Wrapper<std::pair<int, int> > dymmywp4;
  edm::Wrapper<std::pair<unsigned int, bool> > dymmywp5;
  edm::Wrapper<std::pair<unsigned int, float> > dymmywp6;
  edm::Wrapper<std::pair<double, double> > dymmywp7;
  edm::Wrapper<std::pair<unsigned long long, std::basic_string<char> > > dymmywp8;
  edm::Wrapper<std::pair<std::basic_string<char>,int> > dummywp9;
  edm::Wrapper<std::pair<std::basic_string<char>,double> > dummywp10;
  edm::Wrapper<std::pair<std::basic_string<char>,std::vector<std::pair<std::basic_string<char>,double> > > > dummywp11;
  edm::Wrapper<std::map<unsigned long, unsigned long> > dymmywm1;
  edm::Wrapper<std::map<unsigned int, unsigned int> > dymmywm2;
  edm::Wrapper<std::map<unsigned short, unsigned short> > dymmywm3;
  edm::Wrapper<std::map<int, int> > dymmywm4;
  edm::Wrapper<std::map<unsigned int, bool> > dymmywm5;
  edm::Wrapper<std::map<unsigned long, std::vector<unsigned long> > > dymmywmv1;
  edm::Wrapper<std::map<unsigned int, std::vector<unsigned int> > > dymmywmv2;
  edm::Wrapper<std::map<unsigned short, std::vector<unsigned short> > > dymmypwmv3;
  edm::Wrapper<std::map<unsigned int, float> > dummyypwmv4;
  edm::Wrapper<std::map<unsigned long long, std::basic_string<char> > > dummyypwmv5;
  edm::Wrapper<std::multimap<double, double> > dummyypwmv6;
  edm::Wrapper<std::map<std::basic_string<char>,int> > dummyypwmv7;
  edm::Wrapper<std::map<std::basic_string<char>,std::vector<std::pair<std::basic_string<char>,double> > > > dummyypwmv8;


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
  edm::Wrapper<unsigned long long> dummyw14;
  edm::Wrapper<long long> dummyw15;

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

  std::vector<char>::iterator itc;
  std::vector<short>::iterator its;
  std::vector<unsigned short>::iterator itus;
  std::vector<int>::iterator iti;
  std::vector<unsigned int>::iterator itui;
  std::vector<long>::iterator itl;
  std::vector<unsigned long>::iterator itul;
  std::vector<long long>::iterator itll;
  std::vector<unsigned long long>::iterator itull;
  std::vector<float>::iterator itf;
  std::vector<double>::iterator itd;
  std::vector<long double>::iterator itld;
  std::vector<std::string>::iterator itstring;
  std::vector<void *>::iterator itvp;

}}
