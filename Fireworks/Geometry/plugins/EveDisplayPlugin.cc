// -*- C++ -*-
//
// Package:    EveDisplayPlugin
// Class:      EveDisplayPlugin
//
/**\class EveDisplayPlugin EveDisplayPlugin.cc Reve/EveDisplayPlugin/src/EveDisplayPlugin.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Wed Sep 26 08:27:23 EDT 2007
//
//

#include "TROOT.h"
#include "TSystem.h"
#include "TColor.h"
#include "TStyle.h"
#include "TEnv.h"

// system include files
#include <memory>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Fireworks/Geometry/interface/DisplayPlugin.h"
#include "Fireworks/Geometry/interface/DisplayPluginFactory.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TGeoManager.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"

//
// class decleration
//

class EveDisplayPlugin : public fireworks::geometry::DisplayPlugin {
public:
  explicit EveDisplayPlugin(edm::ConsumesCollector);
  ~EveDisplayPlugin() override;

private:
  void run(const edm::EventSetup&) override;
  const edm::ESGetToken<TGeoManager, DisplayGeomRecord> m_geomToken;
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
EveDisplayPlugin::EveDisplayPlugin(edm::ConsumesCollector iCollector) : m_geomToken(iCollector.esConsumes()) {
  //now do what ever initialization is needed
}

EveDisplayPlugin::~EveDisplayPlugin() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
// ------------ method called to for each event  ------------
void EveDisplayPlugin::run(const edm::EventSetup& iSetup) {
  std::cout << "In the EveDisplayPlugin::analyze method..." << std::endl;
  using namespace edm;

  TGeoManager const& geom = iSetup.getData(m_geomToken);

  TEveManager::Create();

  TEveGeoTopNode* trk = new TEveGeoTopNode(const_cast<TGeoManager*>(&geom), geom.GetTopNode());
  trk->SetVisLevel(2);
  gEve->AddGlobalElement(trk);
}

//define this as a plug-in
DEFINE_FIREWORKS_GEOM_DISPLAY(EveDisplayPlugin);
