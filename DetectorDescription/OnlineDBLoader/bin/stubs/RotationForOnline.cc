//
// Original Author:  Jie Chen
//         Created:  Mon Apr  9 11:36:53 CDT 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <DetectorDescription/Core/interface/DDExpandedView.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/OfflineDBLoader/interface/ReadWriteORA.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>

#include <iostream>
#include <istream>
#include <fstream>
#include <string>



//
// class decleration
//

class RotationForOnline : public edm::EDAnalyzer {
   public:
      explicit RotationForOnline(const edm::ParameterSet&);
      ~RotationForOnline();
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


   private:

      // ----------member data ---------------------------
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
RotationForOnline::RotationForOnline(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


RotationForOnline::~RotationForOnline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RotationForOnline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "analyze does nothing" << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
RotationForOnline::beginJob(const edm::EventSetup& iSetup)
{
  std::string rotationFileName("ROTATIONS.dat");
  std::ofstream rotationOS(rotationFileName.c_str());
  std::cout << "RotationForOnline Analyzer..." << std::endl;

  edm::ESHandle<DDCompactView> pDD;

  iSetup.get<IdealGeometryRecord>().get( "", pDD );

  DDRotation::iterator<DDRotation> rit(DDRotation::begin()), red(DDRotation::end());
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    if ( rit->matrix()->isIdentity() ) continue;
    const DDRotation& rota = *rit;
    bool reflection = false;

    Hep3Vector xv=rota.rotation()->colX();
    Hep3Vector yv = rota.rotation()->colY();
    Hep3Vector zv = rota.rotation()->colZ();
    if ( xv.cross(yv) * zv  < 0) {
                reflection = true;
    }
 
    rotationOS<< *(rota.isDefined().first); //rota name

    rotationOS<< "," << rota.rotation()->thetaX()
	      << "," << rota.rotation()->phiX()
	      << "," << rota.rotation()->thetaY()
	      << "," << rota.rotation()->phiY()
	      << "," << rota.rotation()->thetaZ()
	      << "," << rota.rotation()->phiZ()
	      << "," << (int)reflection
	      <<std::endl;
  } 

  rotationOS.close();

}

// ------------ method called once each job just after ending the event loop  ------------
void 
RotationForOnline::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RotationForOnline);
