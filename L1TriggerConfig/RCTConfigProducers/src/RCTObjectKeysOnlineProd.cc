// -*- C++ -*-
//
// Package:    L1TriggerConfig
// Class:      RCTObjectKeysOnlineProd
// 
/**\class RCTObjectKeysOnlineProd RCTObjectKeysOnlineProd.h L1TriggerConfig/RCTConfigProducers/src/RCTObjectKeysOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Fri Aug 22 19:51:36 CEST 2008
// $Id$
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class RCTObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      RCTObjectKeysOnlineProd(const edm::ParameterSet&);
      ~RCTObjectKeysOnlineProd();

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
RCTObjectKeysOnlineProd::RCTObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig )
{}


RCTObjectKeysOnlineProd::~RCTObjectKeysOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RCTObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  std::string rctKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kRCT ) ;

  // SELECT RCT_PARAMETER FROM RCT_CONF WHERE RCT_CONF.RCT_KEY = rctKey
  l1t::OMDSReader::QueryResults paremKeyResults =
    m_omdsReader.basicQuery( "RCT_PARAMETER",
			     "CMS_RCT",
			     "RCT_CONF",
			     "RCT_CONF.RCT_KEY",
			     m_omdsReader.singleAttribute( rctKey  ) );


  if( paremKeyResults.second.size() != 1 ) // check query successful
    {
      edm::LogError( "L1-O2O" ) << "Problem with RCT key." ;
      return ;
    }

  std::string paremKey =
    paremKeyResults.second.front()[ "RCT_PARAMETER" ].data< std::string >() ;

  pL1TriggerKey->add( "L1RCTParametersRcd",
		      "L1RCTParameters",
		      paremKey ) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTObjectKeysOnlineProd);
