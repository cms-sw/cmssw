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
// $Id: EveDisplayPlugin.cc,v 1.1 2010/04/01 21:58:00 chrjones Exp $
//
//

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
      explicit EveDisplayPlugin();
      ~EveDisplayPlugin();


   private:
      virtual void run(const edm::EventSetup&);

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
EveDisplayPlugin::EveDisplayPlugin()
{
   //now do what ever initialization is needed

}


EveDisplayPlugin::~EveDisplayPlugin()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
// ------------ method called to for each event  ------------
void
EveDisplayPlugin::run(const edm::EventSetup& iSetup)
{
  std::cout << "In the EveDisplayPlugin::analyze method..." << std::endl;
   using namespace edm;

   ESHandle<TGeoManager> geom;
   iSetup.get<DisplayGeomRecord>().get(geom);


   TEveManager::Create();

   TEveGeoTopNode* trk = new TEveGeoTopNode(const_cast<TGeoManager*>(geom.product()),
					    geom->GetTopNode());
   trk->SetVisLevel(2);
   gEve->AddGlobalElement(trk);

}


//define this as a plug-in
DEFINE_FIREWORKS_GEOM_DISPLAY(EveDisplayPlugin);
