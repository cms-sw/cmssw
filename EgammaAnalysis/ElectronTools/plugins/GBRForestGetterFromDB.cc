#ifndef CalibratedElectronProducer_h
#define CalibratedElectronProducer_h

#include <string>
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include <TFile.h>

class GBRForestGetterFromDB : public edm::one::EDAnalyzer<> {
public:
  explicit GBRForestGetterFromDB(const edm::ParameterSet &);
  ~GBRForestGetterFromDB() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  std::string theGBRForestName;
  std::string theOutputFileName;
  std::string theOutputObjectName;
  edm::ESGetToken<GBRForest, GBRWrapperRcd> theGBRForestToken_;
};

GBRForestGetterFromDB::GBRForestGetterFromDB(const edm::ParameterSet &conf)
    : theGBRForestName(conf.getParameter<std::string>("grbForestName")),
      theOutputFileName(conf.getUntrackedParameter<std::string>("outputFileName")),
      theOutputObjectName(conf.getUntrackedParameter<std::string>(
          "outputObjectName", theGBRForestName.empty() ? "GBRForest" : theGBRForestName)),
      theGBRForestToken_(esConsumes()) {}

GBRForestGetterFromDB::~GBRForestGetterFromDB() {}

void GBRForestGetterFromDB::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  auto theGBRForestHandle = iSetup.getHandle(theGBRForestToken_);
  TFile *fOut = TFile::Open(theOutputFileName.c_str(), "RECREATE");
  fOut->WriteObject(theGBRForestHandle.product(), theOutputObjectName.c_str());
  fOut->Close();
  edm::LogPrint("GBRForestGetterFromDB") << "Wrote output to " << theOutputFileName;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GBRForestGetterFromDB);

#endif
