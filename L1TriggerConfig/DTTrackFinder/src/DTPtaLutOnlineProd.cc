// -*- C++ -*-
//
// Package:    DTPtaLutOnlineProd
// Class:      DTPtaLutOnlineProd
// 
/**\class DTPtaLutOnlineProd DTPtaLutOnlineProd.h L1Trigger/DTPtaLutProducers/src/DTPtaLutOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Oct  2 19:40:12 CEST 2008
// $Id$
//
//


// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTPtaLutRcd.h"

//
// class declaration
//

class DTPtaLutOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTPtaLutRcd, L1MuDTPtaLut >
{
   public:
      DTPtaLutOnlineProd(const edm::ParameterSet&);
      ~DTPtaLutOnlineProd();

  virtual boost::shared_ptr< L1MuDTPtaLut > newObject(
    const std::string& objectKey ) ;

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
DTPtaLutOnlineProd::DTPtaLutOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTPtaLutRcd,
			    L1MuDTPtaLut >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTPtaLutOnlineProd::~DTPtaLutOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTPtaLut >
DTPtaLutOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1MuDTPtaLut object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1MuDTPtaLut >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTPtaLutOnlineProd);
