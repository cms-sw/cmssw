#ifndef HcalConstantsASCIIWriter_h
#define HcalConstantsASCIIWriter_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include <vector>
#include <map>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//
// class decleration
//
namespace cms {
  class HcalConstantsASCIIWriter : public edm::EDAnalyzer {
  public:
    explicit HcalConstantsASCIIWriter(const edm::ParameterSet &);
    ~HcalConstantsASCIIWriter();

    virtual void analyze(const edm::Event &, const edm::EventSetup &);
    virtual void beginJob();
    virtual void endJob();

  private:
    // ----------member data ---------------------------
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
    edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_resp_;

    std::ofstream *myout_hcal;
    std::string file_input;
    std::string file_output;
  };
}  // namespace cms
#endif
