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
// $Id: RCTObjectKeysOnlineProd.cc,v 1.8 2010/05/21 13:36:40 efron Exp $
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

  if( !rctKey.empty() )
    {
      std::string paremKey, scaleKey, ecalScaleKey , hcalScaleKey;
      // SELECT RCT_PARAMETER FROM RCT_CONF WHERE RCT_CONF.RCT_KEY = rctKey
      l1t::OMDSReader::QueryResults paremKeyResults =
	m_omdsReader.basicQuery( "RCT_PARAMETER",
				 "CMS_RCT",
				 "RCT_CONF",
				 "RCT_CONF.RCT_KEY",
				 m_omdsReader.singleAttribute( rctKey  ) );


      if( paremKeyResults.queryFailed() ||
	  paremKeyResults.numberRows() != 1 ) // check query successful
	{
	  edm::LogError( "L1-O2O" ) << "Problem with RCT Parameter key." ;
	  return ;
	}


      l1t::OMDSReader::QueryResults scaleKeyResults =
	m_omdsReader.basicQuery( "L1T_SCALE_CALO_ET_THRESHOLD_ID",
				 "CMS_RCT",
				 "PAREM_CONF",
				 "PAREM_CONF.PAREM_KEY",
				 paremKeyResults );  // not null no need to check

      l1t::OMDSReader::QueryResults ecalKeyResults =
	m_omdsReader.basicQuery("ECAL_CONF",
				"CMS_RCT",
				"RCT_CONF",
				"RCT_CONF.RCT_KEY",
				m_omdsReader.singleAttribute(rctKey));

      if( ecalKeyResults.queryFailed() ||
	  ecalKeyResults.numberRows() > 1 ) // check query successful)	
	{
	  edm::LogError( "L1-O2O" ) << "Problem with rct_conf.ecal_conf." ;
	  return ;
	}
      std::string ecalKey;
      if(!ecalKeyResults.fillVariable(ecalKey))
	 ecalScaleKey = "NULL";
      else { // fill variable not null

	if( ecalKey == "NULL")
	  ecalScaleKey = "NULL";
	else if(ecalKey == "IDENTITY")
	  ecalScaleKey = "IDENTITY";
	else { // not identity or null
      

	l1t::OMDSReader::QueryResults ecalScaleKeyResults =
	  m_omdsReader.basicQuery( "ECAL_LUT_CONFIG_ID",
				     "CMS_RCT",
				     "ECAL_SCALE_KEY",
				   "ECAL_SCALE_KEY.ECAL_TAG",
				     ecalKeyResults);
	
	  if( ecalScaleKeyResults.queryFailed() ||
	      ecalScaleKeyResults.numberRows() > 1 ) // check query successful)
	    {
	      std::cout << " nrows " << ecalScaleKeyResults.numberRows() <<std::endl;
	      edm::LogError( "L1-O2O" ) << "bad results from lut_config_id." ;
	      return ;
	    }
	  int ecalScaleTemp = -1;
	 
	  ecalScaleKeyResults.fillVariable( ecalScaleTemp );

	


	 std::stringstream ss;
	 ss << ecalScaleTemp;
	 ecalScaleKey = ss.str();
	}
      }    

      l1t::OMDSReader::QueryResults hcalKeyResults =
	m_omdsReader.basicQuery("HCAL_CONF",
				"CMS_RCT",
				"RCT_CONF",
				"RCT_CONF.RCT_KEY",
				m_omdsReader.singleAttribute(rctKey));

      if( hcalKeyResults.queryFailed() ||
	  hcalKeyResults.numberRows() > 1 ) // check query successful)	
	{
	  edm::LogError( "L1-O2O" ) << "Problem with rct_conf.hcal_conf." ;
	  return ;
	}
      std::string hcalKey;
      if(!hcalKeyResults.fillVariable(hcalScaleKey))
	hcalScaleKey = "NULL";
  
    

      paremKeyResults.fillVariable( paremKey ) ;
      scaleKeyResults.fillVariable( scaleKey ) ;
  
      
      pL1TriggerKey->add( "L1RCTParametersRcd",
			  "L1RCTParameters",
			  paremKey ) ;
      pL1TriggerKey->add( "L1EmEtScaleRcd",
			  "L1CaloEtScale",
			  scaleKey ) ;
      pL1TriggerKey->add( "L1CaloEcalScaleRcd",
			  "L1CaloEcalScale",
			  ecalScaleKey ) ;
      pL1TriggerKey->add( "L1CaloHcalScaleRcd",
			  "L1CaloHcalScale",
			  hcalScaleKey ) ;
    }
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTObjectKeysOnlineProd);
