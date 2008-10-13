// -*- C++ -*-
//
// Package:    DTExtLutOnlineProd
// Class:      DTExtLutOnlineProd
// 
/**\class DTExtLutOnlineProd DTExtLutOnlineProd.h L1Trigger/DTExtLutProducers/src/DTExtLutOnlineProd.cc

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

#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"

//
// class declaration
//

class DTExtLutOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTExtLutRcd, L1MuDTExtLut >
{
   public:
      DTExtLutOnlineProd(const edm::ParameterSet&);
      ~DTExtLutOnlineProd();

  virtual boost::shared_ptr< L1MuDTExtLut > newObject(
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
DTExtLutOnlineProd::DTExtLutOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTExtLutRcd,
			    L1MuDTExtLut >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTExtLutOnlineProd::~DTExtLutOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTExtLut >
DTExtLutOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1MuDTExtLut object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1MuDTExtLut >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTExtLutOnlineProd);
