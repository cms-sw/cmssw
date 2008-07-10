// -*- C++ -*-
//
// Package:    L1TriggerKeyOnlineProd
// Class:      L1TriggerKeyOnlineProd
// 
/**\class L1TriggerKeyOnlineProd L1TriggerKeyOnlineProd.h CondTools/L1TriggerKeyOnlineProd/src/L1TriggerKeyOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sun Mar  2 03:03:32 CET 2008
// $Id: L1TriggerKeyOnlineProd.cc,v 1.5 2008/07/04 23:26:14 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerKeyOnlineProd.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TriggerKeyOnlineProd::L1TriggerKeyOnlineProd(const edm::ParameterSet& iConfig)
   : m_tscKey( iConfig.getParameter< std::string >( "tscKey" ) ),
     m_omdsReader(
	iConfig.getParameter< std::string >( "onlineDB" ),
	iConfig.getParameter< std::string >( "onlineAuthentication" ) ),
     m_forceGeneration( iConfig.getParameter< bool >( "forceGeneration" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1TriggerKeyOnlineProd::~L1TriggerKeyOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyOnlineProd::ReturnType
L1TriggerKeyOnlineProd::produce(const L1TriggerKeyRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TriggerKey> pL1TriggerKey ;
   pL1TriggerKey = boost::shared_ptr< L1TriggerKey >( new L1TriggerKey() ) ;
   pL1TriggerKey->setTSCKey( m_tscKey ) ;

   // Get L1TriggerKeyList
   const L1TriggerKeyListRcd& keyListRcd =
      iRecord.getRecord< L1TriggerKeyListRcd >() ;
   edm::ESHandle< L1TriggerKeyList > keyList ;
   keyListRcd.get( keyList ) ;

   // If L1TriggerKeyList does not contain TSC key, token is empty
   if( keyList->token( m_tscKey ) == std::string() ||
       m_forceGeneration )
   {
      // Instantiate new L1TriggerKey
      pL1TriggerKey = boost::shared_ptr< L1TriggerKey >(
         new L1TriggerKey() ) ;
      pL1TriggerKey->setTSCKey( m_tscKey ) ;

      if( !m_forceGeneration )
	{
	  // Get subsystem keys from OMDS

	  // SELECT CSCTF_KEY, DTTF_KEY, RPC_KEY, GMT_KEY, RCT_KEY, GCT_KEY, GT_KEY FROM TRIGGERSUP_CONF WHERE TRIGGERSUP_CONF.TS_KEY = m_tscKey
	  std::vector< std::string > queryStrings ;
	  queryStrings.push_back( "CSCTF_KEY" ) ;
	  queryStrings.push_back( "DTTF_KEY" ) ;
	  queryStrings.push_back( "RPC_KEY" ) ;
	  queryStrings.push_back( "GMT_KEY" ) ;
	  queryStrings.push_back( "RCT_KEY" ) ;
	  queryStrings.push_back( "GCT_KEY" ) ;
	  queryStrings.push_back( "GT_KEY" ) ;
	  //	  queryStrings.push_back( "TSP0_KEY" ) ;

	  l1t::OMDSReader::QueryResults subkeyResults =
	    m_omdsReader.basicQuery( queryStrings,
				     "TRIGGERSUP_CONF",
				     "TRIGGERSUP_CONF.TS_KEY",
				     m_omdsReader.singleAttribute(m_tscKey) ) ;
	  const coral::AttributeList& row = subkeyResults.second.front() ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  std::string csctfKey = row[ "CSCTF_KEY" ].data< std::string >() ;
	  pL1TriggerKey->add( "L1MuCSCPtLut",
			      "L1MuCSCPtLutRcd",
			      csctfKey ) ;
	  pL1TriggerKey->add( "L1MuCSCTFAlignment",
			      "L1MuCSCTFAlignmentRcd",
			      csctfKey ) ;
	  pL1TriggerKey->add( "L1MuCSCTFConfiguration",
			      "L1MuCSCTFConfigurationRcd",
			      csctfKey ) ;
	  std::cout << "CSCTF_KEY " << csctfKey << std::endl ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  std::string dttfKey = row[ "DTTF_KEY" ].data< std::string >() ;
	  pL1TriggerKey->add( "L1MuDTEtaPatternLut",
			      "L1MuDTEtaPatternLutRcd",
			      dttfKey ) ;
	  pL1TriggerKey->add( "L1MuDTExtLut",
			      "L1MuDTExtLutRcd",
			      dttfKey ) ;
	  pL1TriggerKey->add( "L1MuDTPhiLut",
			      "L1MuDTPhiLutRcd",
			      dttfKey ) ;
	  pL1TriggerKey->add( "L1MuDTPtaLut",
			      "L1MuDTPtaLutRcd",
			      dttfKey ) ;
	  pL1TriggerKey->add( "L1MuDTQualPatternLut",
			      "L1MuDTQualPatternLutRcd",
			      dttfKey ) ;
	  std::cout << "DTTF_KEY " << dttfKey << std::endl ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  std::string rpcKey = row[ "RPC_KEY" ].data< std::string >() ;
	  pL1TriggerKey->add( "L1RPCConfig",
			      "L1RPCConfigRcd",
			      rpcKey ) ;
	  std::cout << "RPC_KEY " << rpcKey << std::endl ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  // SELECT GMT_SOFTWARE_CONFIG FROM GMT_LUTS WHERE GMT_LUTS.KEY =
	  // ( SELECT LUT_KEY FROM GMT_CONFIG WHERE GMT_CONFIG.KEY = GMT_KEY [from subkeyResults] )

	  l1t::OMDSReader::QueryResults gmtSWKeyResults =
	    m_omdsReader.basicQuery(
	      "GMT_SOFTWARE_CONFIG",
	      "GMT_LUTS",
	      "GMT_LUTS.KEY",
	      m_omdsReader.basicQuery( "LUT_KEY",
				       "GMT_CONFIG",
				       "GMT_CONFIG.KEY",
				       subkeyResults, "GMT_KEY" ) ) ;

	  const coral::AttributeList& gmtRow = gmtSWKeyResults.second.front() ;
	  std::string gmtSwKey =
	    gmtRow[ "GMT_SOFTWARE_CONFIG" ].data< std::string >() ;

	  pL1TriggerKey->add( "L1MuGMTParameters",
			      "L1MuGMTParametersRcd",
			      gmtSwKey ) ;
	  std::cout << "GMT_SOFTWARE_CONFIG KEY " << gmtSwKey << std::endl ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  std::string rctKey = row[ "RCT_KEY" ].data< std::string >() ;
	  pL1TriggerKey->add( "L1RCTParameters",
			      "L1RCTParametersRcd",
			      rctKey ) ;
	  std::cout << "RCT_KEY " << rctKey << std::endl ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  std::string gctKey = row[ "GCT_KEY" ].data< std::string >() ;

	  // ~~~~~~~~~~~~~~~~~~~~

	  // SELECT PARTITION0_SETUP_FK FROM GT_SETUP WHERE GT_SETUP.ID = GT_KEY [from subkeyResults]

	  // In the future, do this only if TSP0_KEY is null.

	  std::string gtKey = row[ "GT_KEY" ].data< std::string >() ;

	  l1t::OMDSReader::QueryResults gtPartitionKeyResults =
	    m_omdsReader.basicQuery( "PARTITION0_SETUP_FK",
				     "GT_SETUP",
				     "GT_SETUP.ID",
				     subkeyResults, "GT_KEY" ) ;

	  const coral::AttributeList& gtPartitionKeyRow =
	    gtPartitionKeyResults.second.front() ;
	  std::string gtPartitionKey =
	    gtPartitionKeyRow[ "PARTITION0_SETUP_FK" ].data< std::string >() ;

	  pL1TriggerKey->add( "L1GtPrescaleFactors",
			      "L1GtPrescaleFactorsAlgoTrigRcd",
			      gtKey ) ;
	  pL1TriggerKey->add( "L1GtPrescaleFactors",
			      "L1GtPrescaleFactorsTechTrigRcd",
			      gtKey ) ;
	  pL1TriggerKey->add( "L1GtTriggerMask",
			      "L1GtTriggerMaskAlgoTrigRcd",
			      gtPartitionKey ) ;
	  pL1TriggerKey->add( "L1GtTriggerMask",
			      "L1GtTriggerMaskTechTrigRcd",
			      gtPartitionKey ) ;
	  pL1TriggerKey->add( "L1GtParameters",
			      "L1GtParametersRcd",
			      gtKey ) ;
	  pL1TriggerKey->add( "L1GtStableParameters",
			      "L1GtStableParametersRcd",
			      gtKey ) ;
	  pL1TriggerKey->add( "L1GtBoardMaps",
			      "L1GtBoardMapsRcd",
			      gtKey ) ;
	  pL1TriggerKey->add( "L1GtTriggerMenu",
			      "L1GtTriggerMenuRcd",
			      gtKey ) ;

	  std::cout << "GT_KEY " << gtKey << std::endl ;
	  std::cout << "GT_PARTITION_KEY " << gtPartitionKey << std::endl ;

	  // Muon scales

	  // SELECT SCALE_MUON_PT_REF, SCALE_MUON_ETA_REF, SCALE_MUON_PHI_REF FROM L1T_SCALES WHERE L1T_SCALES.ID =
	  // ( SELECT SCALES_FK FROM L1T_MENU WHERE L1T_MENU.ID =
	  // ( SELECT L1T_MENU_FK FROM GT_SETUP WHERE GT_SETUP.ID GT_KEY [from subkeyResults] ) )

	  std::vector< std::string > gtQueryStrings ;
	  gtQueryStrings.push_back( "SCALE_MUON_PT_REF" ) ;
	  gtQueryStrings.push_back( "SCALE_MUON_ETA_REF" ) ;
	  gtQueryStrings.push_back( "SCALE_MUON_PHI_REF" ) ;

	  l1t::OMDSReader::QueryResults muonScaleKeyResults =
	    m_omdsReader.basicQuery(
	      gtQueryStrings,
	      "L1T_SCALES",
	      "L1T_SCALES.ID",
	      m_omdsReader.basicQuery(
		"SCALES_FK",
		"L1T_MENU",
		"L1T_MENU.ID",
		m_omdsReader.basicQuery(
		  "L1T_MENU_FK",
		  "GT_SETUP",
		  "GT_SETUP.ID",
		  subkeyResults, "GT_KEY" ) ) ) ;

	  const coral::AttributeList& gtRow =
	    muonScaleKeyResults.second.front() ;
	  std::string gtMuPtScaleId =
	    gtRow[ "SCALE_MUON_PT_REF" ].data< std::string >() ;
	  std::string gtMuEtaScaleId =
	    gtRow[ "SCALE_MUON_ETA_REF" ].data< std::string >() ;
	  std::string gtMuPhiScaleId =
	    gtRow[ "SCALE_MUON_PHI_REF" ].data< std::string >() ;

	  pL1TriggerKey->add( "L1MuTriggerPtScale",
			      "L1MuTriggerPtScaleRcd",
			      gtMuPtScaleId ) ;

	  // Concatenate eta and phi keys, separated by comma;
	  // need to parse in L1TriggerConfigOnlineProd
	  std::string muGeomKey = gtMuEtaScaleId + "," + gtMuPhiScaleId ;
	  pL1TriggerKey->add( "L1MuTriggerScales",
			      "L1MuTriggerScalesRcd",
			      muGeomKey ) ;
	  pL1TriggerKey->add( "L1MuGMTScales",
			      "L1MuGMTScalesRcd",
			      muGeomKey ) ;
	  std::cout << "Mu pt scale key " << gtMuPtScaleId << std::endl ;
	  std::cout << "Mu eta scale key " << gtMuEtaScaleId << std::endl ;
	  std::cout << "Mu phi scale key " << gtMuPhiScaleId << std::endl ;

	  // Test string parsing
	  int loc = muGeomKey.find( "," ) ;
	  std::string etaKey = muGeomKey.substr( 0,      // start position
						 loc ) ; // length
	  int phiKeyLength = muGeomKey.size() - etaKey.size() - 1 ;
	  std::string phiKey = muGeomKey.substr( loc+1,
						 phiKeyLength ) ;
	  std::cout << "Parsed mu eta scale key " << etaKey << std::endl ;
	  std::cout << "Parsed mu phi scale key " << phiKey << std::endl ;
	}
   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1TriggerKey for TSC key " + m_tscKey + " already in CondDB." ) ;
   }

   return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProd);
