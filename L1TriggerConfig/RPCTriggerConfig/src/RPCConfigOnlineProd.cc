// -*- C++ -*-
//
// Package:    RPCConfigOnlineProd
// Class:      RPCConfigOnlineProd
// 
/**\class RPCConfigOnlineProd RPCConfigOnlineProd.h L1Trigger/RPCConfigProducers/src/RPCConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Oct  2 19:40:12 CEST 2008
// $Id: RPCConfigOnlineProd.cc,v 1.1 2008/10/13 02:41:02 wsun Exp $
//
//


// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"

//
// class declaration
//

class RPCConfigOnlineProd : public L1ConfigOnlineProdBase< L1RPCConfigRcd,
							   L1RPCConfig > {
   public:
      RPCConfigOnlineProd(const edm::ParameterSet&);
      ~RPCConfigOnlineProd();

  virtual boost::shared_ptr< L1RPCConfig > newObject(
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
RPCConfigOnlineProd::RPCConfigOnlineProd(const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1RPCConfigRcd, L1RPCConfig >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


RPCConfigOnlineProd::~RPCConfigOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1RPCConfig >
RPCConfigOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1RPCConfig object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1RPCConfig >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCConfigOnlineProd);
