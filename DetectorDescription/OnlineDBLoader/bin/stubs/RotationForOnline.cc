//
// Original Author:  Jie Chen
//         Created:  Mon Apr  9 11:36:53 CDT 2007
// $Id: RotationForOnline.cc,v 1.7 2010/03/25 21:55:36 case Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDValue.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include <Geometry/Records/interface/IdealGeometryRecord.h>


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
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
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
RotationForOnline::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{
  // set tolerance for "near zero"
  double tolerance= 1.0e-3;
  std::string rotationFileName("ROTATIONS.dat");
  std::ofstream rotationOS(rotationFileName.c_str());
  std::cout << "RotationForOnline Analyzer..." << std::endl;


  edm::ESTransientHandle<DDCompactView> pDD;

  iSetup.get<IdealGeometryRecord>().get( "", pDD );
  DDRotationMatrix ident;

  DDRotation::iterator<DDRotation> rit(DDRotation::begin()), red(DDRotation::end());
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    //    if ( rit->matrix()->isIdentity() ) continue;
    if ( *(rit->matrix()) == ident ) continue;
    const DDRotation& rota = *rit;
    bool reflection = false;

    DD3Vector x, y, z;
    rit->matrix()->GetComponents(x, y, z);
    if ( (1.0 + (x.Cross(y)).Dot(z)) <= tolerance ) {
      reflection = true;
    }
 
    rotationOS<< *(rota.isDefined().first); //rota name
    double thetaX, phiX, thetaY, phiY, thetaZ, phiZ;

    thetaX = std::acos(x.z());
    phiX = ( x.y() == 0 && x.x() == 0.0) ? 0.0 : std::atan2(x.y(), x.x());

    thetaY = std::acos(y.z());
    phiY = ( y.y() == 0 && y.x() == 0.0) ? 0.0 : std::atan2(y.y(), y.x());

    thetaZ = std::acos(z.z());
    phiZ = ( z.y() == 0 && z.x() == 0.0) ? 0.0 : std::atan2(z.y(), z.x());

    rotationOS<< "," << thetaX
    	      << "," << phiX
    	      << "," << thetaY
    	      << "," << phiY
	      << "," << thetaZ
	      << "," << phiZ
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
