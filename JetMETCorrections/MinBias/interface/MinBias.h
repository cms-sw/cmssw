// system include files
#include <memory>
#include <string>
#include <iostream>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "TTree.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//
// class decleration
//
namespace cms {
  class MinBias : public edm::EDAnalyzer {
  public:
    explicit MinBias(const edm::ParameterSet&);

    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void beginJob() override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void endJob() override;

  private:
    // ----------member data ---------------------------
    // names of modules, producing object collections

    std::string hbheLabel_, hoLabel_, hfLabel_;
    edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
    edm::EDGetTokenT<HORecHitCollection> hoToken_;
    edm::EDGetTokenT<HFRecHitCollection> hfToken_;
    // stuff for histogramms
    bool allowMissingInputs_;
    //
    //   TH1D*       hCalo1[8000], *hCalo2;
    TTree* myTree_;
    //
    int mydet, mysubd, depth, iphi, ieta;
    float phi, eta;
    float mom1, mom2, mom3, mom4, occup;
    const CaloGeometry* geo_;
    // counters
    std::map<DetId, double> theFillDetMap0_;
    std::map<DetId, double> theFillDetMap1_;
    std::map<DetId, double> theFillDetMap2_;
    std::map<DetId, double> theFillDetMap3_;
    std::map<DetId, double> theFillDetMap4_;
  };
}  // namespace cms
