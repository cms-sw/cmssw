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
// $Id: L1RCT_RSKeysOnlineProd.cc,v 1.1 2009/03/11 10:15:45 jleonard Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include <sstream>
#include "CoralBase/TimeStamp.h"
#include <math.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class L1RCT_RSKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
public: 
  L1RCT_RSKeysOnlineProd(const edm::ParameterSet& iConfig);
  ~L1RCT_RSKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
      // ----------member data ---------------------------
       bool m_enableL1RCTChannelMask;

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
L1RCT_RSKeysOnlineProd::L1RCT_RSKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig ),
    m_enableL1RCTChannelMask ( iConfig.getParameter< bool >( "enableL1RCTChannelMask" ) )
{}



//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1RCT_RSKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{

  if( m_enableL1RCTChannelMask )
    {

      l1t::OMDSReader::QueryResults objectKeyResults =
	m_omdsReader.basicQuery( "ID",
				 "CMS_RCT",
				 "RCT_RUN_SETTINGS_KEY_CURRENT");

      std::string objectKey;


      if( objectKeyResults.queryFailed())

	{
	  edm::LogError( "L1-O2O" ) << "Problem with jey for record L1RCTCHannelMaskRcd: query failed." ;
	}
      else if(objectKeyResults.numberRows() != 1){
	      edm::LogError("L1-O2O")
	      << "Problem with key for record L1RCTChannelMaskRcd: "
	      << (objectKeyResults.numberRows()) << " rows were returned";
      }
      else
	{

	  objectKeyResults.fillVariable( objectKey ) ;

	}
      pL1TriggerKey->add( "L1RCTChannelMaskRcd",
			  "L1RCTChannelMask",
			  objectKey ) ;
    }
}
  
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RCT_RSKeysOnlineProd);
