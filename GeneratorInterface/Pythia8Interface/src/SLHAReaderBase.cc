#include "GeneratorInterface/Pythia8Interface/interface/SLHAReaderBase.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>

#include "TFile.h"
#include "TTree.h"

SLHAReaderBase::SLHAReaderBase(const edm::ParameterSet& conf) {
  auto filename = conf.getParameter<std::string>("file");
  file_ = TFile::Open(filename.c_str());
  if (!file_)
    throw cms::Exception("MissingFile") << "Could not open file " << filename;

  auto treename = conf.getParameter<std::string>("tree");
  tree_ = (TTree*)file_->Get(treename.c_str());
  if (!tree_)
    throw cms::Exception("MissingTree") << "Could not get tree " << treename << " from file " << filename;
}

SLHAReaderBase::~SLHAReaderBase() { file_->Close(); }

std::vector<std::string> SLHAReaderBase::splitline(const std::string& line, char delim) {
  std::stringstream ss(line);
  std::string field;
  std::vector<std::string> fields;
  while (getline(ss, field, delim)) {
    fields.push_back(field);
  }
  return fields;
}

EDM_REGISTER_PLUGINFACTORY(SLHAReaderFactory, "SLHAReaderFactory");
