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

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TFile.h"
#include "TTree.h"



namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

//
// class decleration
//
namespace cms
{
class MinBias : public edm::EDAnalyzer {
   public:
      explicit MinBias(const edm::ParameterSet&);
      ~MinBias();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void endJob() ;

   private:
  // ----------member data ---------------------------
  // names of modules, producing object collections

     std::string hbheLabel_,hoLabel_,hfLabel_;
     edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
     edm::EDGetTokenT<HORecHitCollection> hoToken_;
     edm::EDGetTokenT<HFRecHitCollection> hfToken_;
  // stuff for histogramms
  //  output file name with histograms
     std::string fOutputFileName ;
     bool allowMissingInputs_;
  //
     TFile*      hOutputFile ;
  //   TH1D*       hCalo1[8000], *hCalo2;
     TTree * myTree;
  //
     int mydet, mysubd, depth, iphi, ieta, ievent;
     float phi,eta;
     float mom1,mom2,mom3,mom4,occup;
     const CaloGeometry* geo;
  // counters
     std::map<DetId,double> theFillDetMap0;
     std::map<DetId,double> theFillDetMap1;
     std::map<DetId,double> theFillDetMap2;
     std::map<DetId,double> theFillDetMap3;
     std::map<DetId,double> theFillDetMap4;

};
}
