// system include files
#include <cstdio>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>

// user include files
#include "CondFormats/DataRecord/interface/MuScleFitDBobjectRcd.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

class DBReader : public edm::one::EDAnalyzer<> {
public:
  explicit DBReader(const edm::ParameterSet&);
  ~DBReader() override;
  void initialize(const edm::EventSetup& iSetup);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  template <typename T>
  void printParameters(const T& functionPtr) {
    // Looping directly on it does not work, because it is returned by value
    // and the iterator gets invalidated on the next line. Save it to a temporary object
    // and iterate on it.
    std::vector<double> parVecVec(functionPtr->parameters());
    std::vector<double>::const_iterator parVec = parVecVec.begin();
    std::vector<int> functionId(functionPtr->identifiers());
    std::vector<int>::const_iterator id = functionId.begin();
    edm::LogPrint("DBReader") << "total number of parameters read from database = parVecVec.size() = "
                              << parVecVec.size() << std::endl;
    int iFunc = 0;
    for (; id != functionId.end(); ++id, ++iFunc) {
      int parNum = functionPtr->function(iFunc)->parNum();
      edm::LogPrint("DBReader") << "For function id = " << *id << ", with " << parNum << " parameters: " << std::endl;
      for (int par = 0; par < parNum; ++par) {
        edm::LogPrint("DBReader") << "par[" << par << "] = " << *parVec << std::endl;
        ++parVec;
      }
    }
  }

  //  uint32_t printdebug_;
  const edm::ESGetToken<MuScleFitDBobject, MuScleFitDBobjectRcd> muToken_;
  const std::string type_;
  //std::unique_ptr<BaseFunction> corrector_;
  std::shared_ptr<MomentumScaleCorrector> corrector_;
  std::shared_ptr<ResolutionFunction> resolution_;
  std::shared_ptr<BackgroundFunction> background_;
};

DBReader::DBReader(const edm::ParameterSet& iConfig)
    : muToken_(esConsumes()), type_(iConfig.getUntrackedParameter<std::string>("Type")) {}

void DBReader::initialize(const edm::EventSetup& iSetup) {
  const MuScleFitDBobject* dbObject = &iSetup.getData(muToken_);
  edm::LogInfo("DBReader") << "[DBReader::analyze] End Reading MuScleFitDBobjectRcd" << std::endl;
  edm::LogPrint("DBReader") << "identifiers size from dbObject = " << dbObject->identifiers.size() << std::endl;
  edm::LogPrint("DBReader") << "parameters size from dbObject = " << dbObject->parameters.size() << std::endl;

  // This string is one of: scale, resolution, background.
  // Create the corrector and set the parameters
  if (type_ == "scale")
    corrector_.reset(new MomentumScaleCorrector(dbObject));
  else if (type_ == "resolution")
    resolution_.reset(new ResolutionFunction(dbObject));
  else if (type_ == "background")
    background_.reset(new BackgroundFunction(dbObject));
  else {
    edm::LogPrint("DBReader") << "Error: unrecognized type. Use one of those: 'scale', 'resolution', 'background'"
                              << std::endl;
    exit(1);
  }
  // cout << "pointer = " << corrector_.get() << endl;
}

//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

DBReader::~DBReader() = default;

void DBReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  initialize(iSetup);
  if (type_ == "scale")
    printParameters(corrector_);
  else if (type_ == "resolution")
    printParameters(resolution_);
  else if (type_ == "background")
    printParameters(background_);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DBReader);
