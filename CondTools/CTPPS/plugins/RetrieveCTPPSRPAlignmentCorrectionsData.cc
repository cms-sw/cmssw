/****************************************************************************
* Author: 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/DataRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"

#include <memory>

class RetrieveCTPPSRPAlignmentCorrectionsData : public edm::one::EDAnalyzer<> {
public:
  explicit RetrieveCTPPSRPAlignmentCorrectionsData(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, CTPPSRPAlignmentCorrectionsDataRcd> esToken_;
};

RetrieveCTPPSRPAlignmentCorrectionsData::RetrieveCTPPSRPAlignmentCorrectionsData(const edm::ParameterSet &iConfig)
    : esToken_(esConsumes()) {}

void RetrieveCTPPSRPAlignmentCorrectionsData::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ctppsRPAlignmentCorrectionsData = iSetup.getData(esToken_);

  edm::LogInfo("PPS") << ctppsRPAlignmentCorrectionsData;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RetrieveCTPPSRPAlignmentCorrectionsData);
