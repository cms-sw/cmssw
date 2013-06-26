// -*- C++ -*-
//
// Package:    DTPhiLutOnlineProd
// Class:      DTPhiLutOnlineProd
// 
/**\class DTPhiLutOnlineProd DTPhiLutOnlineProd.h L1Trigger/DTPhiLutProducers/src/DTPhiLutOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Oct  2 19:40:12 CEST 2008
// $Id: DTPhiLutOnlineProd.cc,v 1.1 2008/10/13 03:24:50 wsun Exp $
//
//


// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"

//
// class declaration
//

class DTPhiLutOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTPhiLutRcd, L1MuDTPhiLut >
{
   public:
      DTPhiLutOnlineProd(const edm::ParameterSet&);
      ~DTPhiLutOnlineProd();

  virtual boost::shared_ptr< L1MuDTPhiLut > newObject(
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
DTPhiLutOnlineProd::DTPhiLutOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTPhiLutRcd,
			    L1MuDTPhiLut >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTPhiLutOnlineProd::~DTPhiLutOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTPhiLut >
DTPhiLutOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1MuDTPhiLut object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1MuDTPhiLut >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTPhiLutOnlineProd);
