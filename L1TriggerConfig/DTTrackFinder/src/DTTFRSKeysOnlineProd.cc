// -*- C++ -*-
//
// Package:    L1TriggerConfig
// Class:      DTTFRSKeysOnlineProd
// 
/**\class DTTFRSKeysOnlineProd DTTFRSKeysOnlineProd.h L1TriggerConfig/DTTFConfigProducers/src/DTTFRSKeysOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Troconiz - UAM Madrid
//         Created:  Thu Oct  2 21:43:50 CEST 2008
// $Id: DTTFRSKeysOnlineProd.cc,v 1.1 2009/05/14 14:13:40 troco Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class DTTFRSKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      DTTFRSKeysOnlineProd(const edm::ParameterSet&);
      ~DTTFRSKeysOnlineProd();

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
DTTFRSKeysOnlineProd::DTTFRSKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBase( iConfig )
{}


DTTFRSKeysOnlineProd::~DTTFRSKeysOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DTTFRSKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  // SELECT HW_SETTINGS FROM CMS_DT_TF.DTTF_SETTINGS_KEY_CURRENT
  l1t::OMDSReader::QueryResults rsKeyResults =
    m_omdsReader.basicQuery( "HW_SETTINGS",
			     "CMS_DT_TF",
			     "DTTF_SETTINGS_KEY_CURRENT" );

  if( rsKeyResults.queryFailed() ||
      rsKeyResults.numberRows() != 1 ) // check query successful
    {
      edm::LogError( "L1-O2O" ) << "Problem with DTTF RS key." ;
      return ;
    }

  std::string rsKey ;
  rsKeyResults.fillVariable( rsKey ) ;

  pL1TriggerKey->add( "L1MuDTTFMasksRcd",
		      "L1MuDTTFMasks",
		      rsKey ) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTTFRSKeysOnlineProd);
