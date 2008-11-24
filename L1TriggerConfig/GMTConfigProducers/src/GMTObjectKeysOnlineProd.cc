// -*- C++ -*-
//
// Package:    L1TriggerConfig
// Class:      GMTScaleKeysOnlineProd
// 
/**\class GMTScaleKeysOnlineProd GMTScaleKeysOnlineProd.h L1TriggerConfig/GMTConfigProducers/src/GMTScaleKeysOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//  Author:  Thomas Themel
// $Id: GMTScaleKeysOnlineProd.cc,v 1.2 2008/09/30 20:35:18 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class GMTScaleKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      GMTScaleKeysOnlineProd(const edm::ParameterSet&);
      ~GMTScaleKeysOnlineProd();

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
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
GMTScaleKeysOnlineProd::GMTScaleKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig )
{}


GMTScaleKeysOnlineProd::~GMTScaleKeysOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GMTScaleKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  std::string rctKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kGMT ) ;

  // SELECT GMT_PARAMETER FROM GMT_CONF WHERE GMT_CONF.GMT_KEY = rctKey
  l1t::OMDSReader::QueryResults paremKeyResults =
    m_omdsReader.basicQuery( "SCALES_KEY",
			     "CMS_GMT",
			     "GMT_CONFIG",
			     "GMT_CONF.GMT_KEY",
			     m_omdsReader.singleAttribute( rctKey  ) );


  if( paremKeyResults.queryFailed() ||
      paremKeyResults.numberRows() != 1 ) // check query successful
    {
      edm::LogError( "L1-O2O" ) << "Problem with GMT key." ;
      return ;
    }

  std::string paremKey ;
  paremKeyResults.fillVariable( paremKey ) ;

  pL1TriggerKey->add( "L1MuGMTScalesRcd",
		      "L1MuGMTScales",
		      paremKey ) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(GMTScaleKeysOnlineProd);
