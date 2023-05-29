#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"  //
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"

#include <iostream>

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"

#include "TH1F.h"

using namespace std;
using namespace edm;

class LCTPrinter : public edm::one::EDAnalyzer<> {
public:
  explicit LCTPrinter(const edm::ParameterSet&);
  ~LCTPrinter();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // ----------member data ---------------------------
  CSCSectorReceiverLUT* srLUTs_[5];
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LCTPrinter::LCTPrinter(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed

  int endcap = 1, sector = 1;  // assume SR LUTs are all same for every sector in either of endcaps
  bool TMB07 = true;           // specific TMB firmware

  edm::ParameterSet srLUTset;
  srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  srLUTset.addUntrackedParameter<bool>("Binary", false);
  srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  for (int station = 1, fpga = 0; station <= 4 && fpga < 5; station++) {
    if (station == 1)
      for (int subSector = 0; subSector < 2 && fpga < 5; subSector++)
        srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, subSector + 1, station, srLUTset, TMB07);
    else
      srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
  }
}

LCTPrinter::~LCTPrinter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void LCTPrinter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ///////////////////
  //Setup Stuff//////
  ///////////////////

  std::vector<csctf::TrackStub> stubs;
  std::vector<csctf::TrackStub>::const_iterator stub;

  //MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi>    "csctfunpacker"          ""        "CsctfFilter"
  edm::Handle<CSCCorrelatedLCTDigiCollection> lctDigiColls;
  iEvent.getByLabel("csctfunpacker", lctDigiColls);
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator lctDigiColl;
  for (lctDigiColl = lctDigiColls->begin(); lctDigiColl != lctDigiColls->end(); lctDigiColl++) {
    CSCCorrelatedLCTDigiCollection::const_iterator lctDigi = (*lctDigiColl).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator lctDigiEnd = (*lctDigiColl).second.second;

    for (; lctDigi != lctDigiEnd; lctDigi++) {
      //int endcap  = (*lctDigiColl).first.endcap()-1;
      int station = (*lctDigiColl).first.station() - 1;
      //int sector  = (*lctDigiColl).first.triggerSector()-1;
      int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*lctDigiColl).first);
      int cscId = (*lctDigiColl).first.triggerCscId() - 1;
      int fpga = (subSector ? subSector - 1 : station + 1);

      lclphidat lclPhi;
      gblphidat gblPhi;
      gbletadat gblEta;

      try {
        lclPhi = srLUTs_[fpga]->localPhi(
            lctDigi->getStrip(), lctDigi->getPattern(), lctDigi->getQuality(), lctDigi->getBend());
      } catch (...) {
        bzero(&lclPhi, sizeof(lclPhi));
        std::cout << "Bad local phi!" << std::endl;
      }
      try {
        gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lctDigi->getKeyWG(), cscId + 1);
      } catch (...) {
        bzero(&gblPhi, sizeof(gblPhi));
        std::cout << "Bad global phi!" << std::endl;
      }
      try {
        gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lctDigi->getKeyWG(), cscId + 1);
      } catch (...) {
        bzero(&gblEta, sizeof(gblEta));
        std::cout << "Bad global eta!" << std::endl;
      }

      csctf::TrackStub theStub((*lctDigi), (*lctDigiColl).first);
      theStub.setPhiPacked(gblPhi.global_phi);
      theStub.setEtaPacked(gblEta.global_eta);

      stubs.push_back(theStub);
    }
  }

  //////////////
  //Analysis////
  //////////////

  for (stub = stubs.begin(); stub != stubs.end(); stub++) {
    stub->print();
    std::cout << "Endcap: " << stub->endcap() << " Station:  " << stub->station() << " Sector: " << stub->sector()
              << " Ring: " << CSCDetId(stub->getDetId()).ring() << " SubSector: " << stub->subsector()
              << " CscId: " << stub->cscid() << " ChamberId: " << CSCDetId(stub->getDetId()).chamber()
              << " TriggerChamberId: " << CSCTriggerNumbering::triggerCscIdFromLabels(stub->getDetId())
              << " Bx: " << stub->BX() << std::endl;
    std::cout << "Phi Packed: " << stub->phiPacked() << std::endl;
    std::cout << "Eta Packed: " << stub->etaPacked() << std::endl;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void LCTPrinter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void LCTPrinter::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(LCTPrinter);
