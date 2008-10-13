// -*- C++ -*-
//
// Package:    DTQualPatternLutOnlineProd
// Class:      DTQualPatternLutOnlineProd
// 
/**\class DTQualPatternLutOnlineProd DTQualPatternLutOnlineProd.h L1Trigger/DTQualPatternLutProducers/src/DTQualPatternLutOnlineProd.cc

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

#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"

//
// class declaration
//

class DTQualPatternLutOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTQualPatternLutRcd, L1MuDTQualPatternLut >
{
   public:
      DTQualPatternLutOnlineProd(const edm::ParameterSet&);
      ~DTQualPatternLutOnlineProd();

  virtual boost::shared_ptr< L1MuDTQualPatternLut > newObject(
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
DTQualPatternLutOnlineProd::DTQualPatternLutOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTQualPatternLutRcd,
			    L1MuDTQualPatternLut >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTQualPatternLutOnlineProd::~DTQualPatternLutOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTQualPatternLut >
DTQualPatternLutOnlineProd::newObject( const std::string& objectKey )
{
  edm::LogError( "L1-O2O" ) << "L1MuDTQualPatternLut object with key "
			    << objectKey << " not in ORCON!" ;

  return boost::shared_ptr< L1MuDTQualPatternLut >() ;
}

//
// member functions
//


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTQualPatternLutOnlineProd);
