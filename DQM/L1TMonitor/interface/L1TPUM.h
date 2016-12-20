#ifndef L1TPUM_H
#define L1TPUM_H

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

class L1TPUM : public DQMEDAnalyzer {
  public:
    L1TPUM(const edm::ParameterSet& ps);
    virtual ~L1TPUM();
  
  protected:
    void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run&, const edm::EventSetup&) override;
    virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
    void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  
  private:
    edm::EDGetTokenT<L1CaloRegionCollection> regionSource_;
    std::string histFolder_;

    MonitorElement* regionsTotalEtBxP2_;
    MonitorElement* regionsTotalEtBx0_;
    MonitorElement* regionsTotalEtBxM2_;

    MonitorElement* regionsAvgEtBxP2_;
    MonitorElement* regionsAvgEtBx0_;
    MonitorElement* regionsAvgEtBxM2_;

    MonitorElement* regionsAvgNonZeroEtBxP2_;
    MonitorElement* regionsAvgNonZeroEtBx0_;
    MonitorElement* regionsAvgNonZeroEtBxM2_;

    MonitorElement* nonZeroRegionsBxP2_;
    MonitorElement* nonZeroRegionsBx0_;
    MonitorElement* nonZeroRegionsBxM2_;

    MonitorElement* regionBxPopulation_;
    MonitorElement* regionBxEtSum_;

    std::vector<MonitorElement*> regionsPUMEtaBxP2_;
    std::vector<MonitorElement*> regionsPUMEtaBx0_;
    std::vector<MonitorElement*> regionsPUMEtaBxM2_;
};

#endif
