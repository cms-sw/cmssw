// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "TPostScript.h"
#include "TCanvas.h"

#include <math.h>
#include <vector>
#include <sstream>

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
//***************************************************

//
// class declaration
//

class testTkHistoMap : public DQMOneEDAnalyzer<> {
public:
  explicit testTkHistoMap(const edm::ParameterSet&);
  ~testTkHistoMap();

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void endJob(void) override;

private:
  void read(const TkDetMap* tkDetMap);
  void create(const TkDetMap* tkDetMap);

  bool readFromFile;
  std::unique_ptr<TkHistoMap> tkhisto, tkhistoBis, tkhistoZ, tkhistoPhi, tkhistoR, tkhistoCheck;
};

//
testTkHistoMap::testTkHistoMap(const edm::ParameterSet& iConfig)
    : readFromFile(iConfig.getParameter<bool>("readFromFile")) {}

void testTkHistoMap::create(const TkDetMap* tkDetMap) {
  tkhisto = std::make_unique<TkHistoMap>(tkDetMap, "detId", "detId", -1);
  tkhistoBis = std::make_unique<TkHistoMap>(
      tkDetMap,
      "detIdBis",
      "detIdBis",
      0,
      1);  //here the baseline (the value of the empty,not assigned bins) is put to -1 (default is zero)
  tkhistoZ = std::make_unique<TkHistoMap>(tkDetMap, "Zmap", "Zmap");
  tkhistoPhi = std::make_unique<TkHistoMap>(tkDetMap, "Phi", "Phi");
  tkhistoR = std::make_unique<TkHistoMap>(
      tkDetMap,
      "Rmap",
      "Rmap",
      -99.);  //here the baseline (the value of the empty,not assigned bins) is put to -99 (default is zero)
  tkhistoCheck = std::make_unique<TkHistoMap>(tkDetMap, "check", "check");
}

/*Check that is possible to load in tkhistomaps histograms already stored in a DQM root file (if the folder and name are known)*/
void testTkHistoMap::read(const TkDetMap* tkDetMap) {
  edm::Service<DQMStore>().operator->()->open("test.root");

  tkhisto = std::make_unique<TkHistoMap>(tkDetMap);
  tkhistoBis = std::make_unique<TkHistoMap>(tkDetMap);
  tkhistoZ = std::make_unique<TkHistoMap>(tkDetMap);
  tkhistoPhi = std::make_unique<TkHistoMap>(tkDetMap);
  tkhistoR = std::make_unique<TkHistoMap>(tkDetMap);
  tkhistoCheck = std::make_unique<TkHistoMap>(tkDetMap);

  tkhisto->loadTkHistoMap("detId", "detId");
  tkhistoBis->loadTkHistoMap("detIdBis", "detIdBis", 1);
  tkhistoZ->loadTkHistoMap("Zmap", "Zmap");
  tkhistoPhi->loadTkHistoMap("Phi", "Phi");
  tkhistoR->loadTkHistoMap("Rmap", "Rmap");
  tkhistoCheck->loadTkHistoMap("check", "check");
}

testTkHistoMap::~testTkHistoMap() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void testTkHistoMap::endJob(void) {
  /*Test extraction of detid from histogram title and ix, iy*/
  size_t ilayer = 1;
  std::string histoTitle = tkhisto->getMap(ilayer)->getTitle();
  uint32_t detB = tkhisto->getDetId(ilayer, 5, 5);
  uint32_t detA = tkhisto->getDetId(histoTitle, 5, 5);
  if (detA == 0 || detA != detB)
    edm::LogError("testTkHistoMap") << " for layer " << ilayer << " the extracted detid in a bin is wrong " << detA
                                    << " " << detB << std::endl;

  /*Test Drawing functions*/
  TCanvas C("c", "c");
  C.Divide(3, 3);
  C.Update();
  TPostScript ps("test.ps", 121);
  ps.NewPage();
  for (size_t ilayer = 1; ilayer < 34; ++ilayer) {
    C.cd(1);
    tkhisto->getMap(ilayer)->getTProfile2D()->Draw("TEXT");
    C.cd(2);
    tkhistoZ->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(3);
    tkhistoPhi->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(4);
    tkhistoR->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(5);
    tkhistoCheck->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.cd(6);
    tkhistoBis->getMap(ilayer)->getTProfile2D()->Draw("BOXCOL");
    C.Update();
    ps.NewPage();
  }
  ps.Close();

  if (!readFromFile)
    edm::Service<DQMStore>().operator->()->save("test.root");

  tkhisto->saveAsCanvas("test.canvas.root", "LEGO", "RECREATE");
  tkhistoBis->saveAsCanvas("test.canvas.root", "LEGO", "RECREATE");
  tkhistoZ->saveAsCanvas("test.canvas.root", "LEGO", "UPDATE");
  tkhistoPhi->saveAsCanvas("test.canvas.root", "LEGO", "UPDATE");
  tkhistoR->saveAsCanvas("test.canvas.root", "LEGO", "UPDATE");
  tkhistoCheck->saveAsCanvas("test.canvas.root", "LEGO", "UPDATE");

  /* test Dump in TkMap*/
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
  TrackerMap tkmap, tkmapZ, tkmapPhi, tkmapR;

  tkmap.setPalette(1);
  tkmapZ.setPalette(2);
  tkmapPhi.setPalette(2);
  tkmapR.setPalette(2);

  tkhisto->dumpInTkMap(&tkmap);
  tkhistoZ->dumpInTkMap(&tkmapZ);
  tkhistoPhi->dumpInTkMap(&tkmapPhi);
  tkhistoR->dumpInTkMap(&tkmapR);

  tkmap.save(true, 0, 0, "testTkMap.png");
  tkmapZ.save(true, 0, 0, "testTkMapZ.png");
  tkmapPhi.save(true, 0, 0, "testTkMapPhi.png");
  tkmapR.save(true, 0, 0, "testTkMapR.png");
}

//
// member functions
//

// // ------------ method called to produce the data  ------------
void testTkHistoMap::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TkDetMap> tkDetMapHandle;
  iSetup.get<TrackerTopologyRcd>().get(tkDetMapHandle);
  const TkDetMap* tkDetMap = tkDetMapHandle.product();
  if (!readFromFile) {
    create(tkDetMap);
  } else {
    read(tkDetMap);
  }

  if (readFromFile)
    return;

  edm::ESHandle<TrackerGeometry> tkgeom;
  iSetup.get<TrackerDigiGeometryRecord>().get(tkgeom);

  float value;
  LocalPoint localPos(0., 0., 0.);
  GlobalPoint globalPos;

  tkhisto->fillFromAscii("test.txt");
  tkhistoBis->fillFromAscii("test2.txt");

  for (const auto det : tkgeom->detUnits()) {
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    if (stripDet != nullptr) {
      globalPos = stripDet->surface().toGlobal(localPos);
      const DetId id = stripDet->geographicalId();

      value = id % 1000000;

      //tkhisto->fill(id,value);
      //tkhistoBis->fill(id,value);
      tkhistoZ->fill(id, globalPos.z());
      tkhistoPhi->fill(id, globalPos.phi());
      tkhistoR->fill(id, globalPos.perp());
      tkhistoCheck->add(id, 1.);
      tkhistoCheck->add(id, 1.);

      edm::LogInfo("testTkHistoMap") << "detid " << id.rawId() << " pos z " << globalPos.z() << " phi "
                                     << globalPos.phi() << " r " << globalPos.perp() << std::endl;

      if (value != tkhisto->getValue(id))
        edm::LogError("testTkHistoMap") << " input value " << value << " differs from read value "
                                        << tkhisto->getValue(id) << std::endl;

      // For usage that reset histo content use setBinContent instead than fill
      /* 
      tkhisto->setBinContent(id,value);
      tkhistoZ->setBinContent(id,globalPos.z());
      tkhistoPhi->setBinContent(id,globalPos.phi());
      tkhistoR->setBinContent(id,globalPos.perp());
      */
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testTkHistoMap);
