#ifndef GeneratorInterface_Pythia8Interface_SLHAReaderBase
#define GeneratorInterface_Pythia8Interface_SLHAReaderBase

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

class TFile;
class TTree;

class SLHAReaderBase {
public:
  SLHAReaderBase(const edm::ParameterSet& conf);
  virtual ~SLHAReaderBase();

  //this function should parse the config description (e.g. with splitline() below)
  //then use the information to get the SLHA info out of the tree and return it
  virtual std::string getSLHA(const std::string& configDesc) = 0;

  static std::vector<std::string> splitline(const std::string& line, char delim);

protected:
  //members
  TFile* file_;
  TTree* tree_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<SLHAReaderBase*(const edm::ParameterSet&)> SLHAReaderFactory;

#endif
