// -*- C++ -*-
//
// Package:    InsertNoisyPixelsInDB
// Class:      InsertNoisyPixelsInDB
// 
/**\class InsertNoisyPixelsInDB InsertNoisyPixelsInDB.cc CondTools/InsertNoisyPixelsInDB/src/InsertNoisyPixelsInDB.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Romain Rougny
//         Created:  Tue Feb  3 15:18:02 CET 2009
// $Id$
//
//


#include "SiPixelGainCalibrationRejectNoisyAndDead.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationRejectNoisyAndDead::SiPixelGainCalibrationRejectNoisyAndDead(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


SiPixelGainCalibrationRejectNoisyAndDead::~SiPixelGainCalibrationRejectNoisyAndDead()
{

}


// ------------ method called to for each event  ------------
void
SiPixelGainCalibrationRejectNoisyAndDead::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   cout<<"test"<<endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationRejectNoisyAndDead::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationRejectNoisyAndDead::endJob() {
}

//define this as a plug-in
