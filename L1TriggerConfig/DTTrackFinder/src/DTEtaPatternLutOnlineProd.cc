// -*- C++ -*-
//
// Package:    DTEtaPatternLutOnlineProd
// Class:      DTEtaPatternLutOnlineProd
// 
/**\class DTEtaPatternLutOnlineProd DTEtaPatternLutOnlineProd.h L1Trigger/DTEtaPatternLutProducers/src/DTEtaPatternLutOnlineProd.cc

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

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"

//
// class declaration
//

class DTEtaPatternLutOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTEtaPatternLutRcd, L1MuDTEtaPatternLut >
{
   public:
      DTEtaPatternLutOnlineProd(const edm::ParameterSet&);
      ~DTEtaPatternLutOnlineProd();

  virtual boost::shared_ptr< L1MuDTEtaPatternLut > newObject(
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
DTEtaPatternLutOnlineProd::DTEtaPatternLutOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTEtaPatternLutRcd,
			    L1MuDTEtaPatternLut >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTEtaPatternLutOnlineProd::~DTEtaPatternLutOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTEtaPatternLut >
DTEtaPatternLutOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1MuDTEtaPatternLut object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1MuDTEtaPatternLut >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTEtaPatternLutOnlineProd);
