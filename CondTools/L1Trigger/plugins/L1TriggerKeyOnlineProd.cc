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
// $Id: L1TriggerKeyOnlineProd.cc,v 1.7 2008/07/23 16:38:08 wsun Exp $
//
//


// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
	iConfig.getParameter< std::string >( "onlineAuthentication" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
   std::vector< std::string > tmp =
     iConfig.getParameter< std::vector< std::string > >( "recordsToInclude" ) ;
   std::vector< std::string >::const_iterator itr = tmp.begin() ;
   std::vector< std::string >::const_iterator end = tmp.end() ;
   for( ; itr != end ; ++itr )
     {
       m_recordsToInclude.insert( make_pair( *itr, false ) ) ;
     }
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
   if( keyList->token( m_tscKey ) == std::string() )
   {
      // Instantiate new L1TriggerKey
      pL1TriggerKey = boost::shared_ptr< L1TriggerKey >(
         new L1TriggerKey() ) ;
      pL1TriggerKey->setTSCKey( m_tscKey ) ;

      edm::LogVerbatim( "L1-O2O" ) << "TSC KEY " << m_tscKey ;

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
				 "CMS_TRG_L1_CONF",
				 "TRIGGERSUP_CONF",
				 "TRIGGERSUP_CONF.TS_KEY",
				 m_omdsReader.singleAttribute(m_tscKey) ) ;
      if( subkeyResults.second.size() != 1 ) // check query successful
	{
	  edm::LogError( "L1-O2O" ) << "Problem with subsystem keys." ;
	  return pL1TriggerKey ;
	}

      const coral::AttributeList& row = subkeyResults.second.front() ;

      // ~~~~~~~~~~~~~~~~~~~~

      std::string csctfKey = row[ "CSCTF_KEY" ].data< std::string >() ;

      if( listContains( "L1MuCSCPtLutRcd" ) )
	pL1TriggerKey->add( "L1MuCSCPtLutRcd",
			    "L1MuCSCPtLut",
			    csctfKey ) ;
      if( listContains( "L1MuCSCTFAlignmentRcd" ) )
	pL1TriggerKey->add( "L1MuCSCTFAlignmentRcd",
			    "L1MuCSCTFAlignment",
			    csctfKey ) ;
      if( listContains( "L1MuCSCTFConfigurationRcd" ) )
	pL1TriggerKey->add( "L1MuCSCTFConfigurationRcd",
			    "L1MuCSCTFConfiguration",
			    csctfKey ) ;
      edm::LogVerbatim( "L1-O2O" ) << "CSCTF_KEY " << csctfKey ;

      // ~~~~~~~~~~~~~~~~~~~~

      std::string dttfKey = row[ "DTTF_KEY" ].data< std::string >() ;

      if( listContains( "L1MuDTEtaPatternLutRcd" ) )
	pL1TriggerKey->add( "L1MuDTEtaPatternLutRcd",
			    "L1MuDTEtaPatternLut",
			    dttfKey ) ;
      if( listContains( "L1MuDTExtLutRcd" ) )
	pL1TriggerKey->add( "L1MuDTExtLutRcd",
			    "L1MuDTExtLut",
			    dttfKey ) ;
      if( listContains( "L1MuDTPhiLutRcd" ) )
	pL1TriggerKey->add( "L1MuDTPhiLutRcd",
			    "L1MuDTPhiLut",
			    dttfKey ) ;
      if( listContains( "L1MuDTPtaLutRcd" ) )
	pL1TriggerKey->add( "L1MuDTPtaLutRcd",
			    "L1MuDTPtaLut",
			    dttfKey ) ;
      if( listContains( "L1MuDTQualPatternLutRcd" ) )
	pL1TriggerKey->add( "L1MuDTQualPatternLutRcd",
			    "L1MuDTQualPatternLut",
			    dttfKey ) ;
      edm::LogVerbatim( "L1-O2O" ) << "DTTF_KEY " << dttfKey ;

      // ~~~~~~~~~~~~~~~~~~~~

      std::string rpcKey = row[ "RPC_KEY" ].data< std::string >() ;
      if( listContains( "L1RPCConfigRcd" ) )
	pL1TriggerKey->add( "L1RPCConfigRcd",
			    "L1RPCConfig",
			    rpcKey ) ;
      edm::LogVerbatim( "L1-O2O" ) << "RPC_KEY " << rpcKey ;

      // ~~~~~~~~~~~~~~~~~~~~

      // SELECT GMT_SOFTWARE_CONFIG FROM GMT_LUTS WHERE GMT_LUTS.KEY =
      // ( SELECT LUT_KEY FROM GMT_CONFIG WHERE GMT_CONFIG.KEY =
      // GMT_KEY [from subkeyResults] )

      l1t::OMDSReader::QueryResults gmtSWKeyResults =
	m_omdsReader.basicQuery(
	   "GMT_SOFTWARE_CONFIG",
	   "CMS_GMT",
	   "GMT_LUTS",
	   "GMT_LUTS.KEY",
	   m_omdsReader.basicQuery( "LUT_KEY",
				    "CMS_GMT",
				    "GMT_CONFIG",
				    "GMT_CONFIG.KEY",
				    subkeyResults, "GMT_KEY" ) ) ;

      if( gmtSWKeyResults.second.size() == 1 ) // check query successful
	{
	  const coral::AttributeList& gmtRow = gmtSWKeyResults.second.front() ;
	  std::string gmtSwKey =
	    gmtRow[ "GMT_SOFTWARE_CONFIG" ].data< std::string >() ;

	  if( listContains( "L1MuGMTParametersRcd" ) )
	    pL1TriggerKey->add( "L1MuGMTParametersRcd",
				"L1MuGMTParameters",
				gmtSwKey ) ;
	  edm::LogVerbatim( "L1-O2O" )
	    << "GMT_KEY " << row[ "GMT_KEY" ].data< std::string >() ;
	  edm::LogVerbatim( "L1-O2O" )
	    << "GMT_SOFTWARE_CONFIG KEY " << gmtSwKey ;
	}
      else
	{
	  edm::LogError( "L1-O2O" )
	    << "Problem with object key for L1MuGMTParametersRcd." ;
	}

      // ~~~~~~~~~~~~~~~~~~~~

      std::string rctKey = row[ "RCT_KEY" ].data< std::string >() ;
      if( listContains( "L1RCTParametersRcd" ) )
	pL1TriggerKey->add( "L1RCTParametersRcd",
			    "L1RCTParameters",
			    rctKey ) ;
      if( listContains( "L1CaloEcalScaleRcd" ) )
	pL1TriggerKey->add( "L1CaloEcalScaleRcd",
			    "L1CaloEcalScale",
			    rctKey ) ;
      if( listContains( "L1CaloHcalScaleRcd" ) )
	pL1TriggerKey->add( "L1CaloHcalScaleRcd",
			    "L1CaloHcalScale",
			    rctKey ) ;
      edm::LogVerbatim( "L1-O2O" ) << "RCT_KEY " << rctKey ;

      // ~~~~~~~~~~~~~~~~~~~~

      std::string gctKey = row[ "GCT_KEY" ].data< std::string >() ;
      if( listContains( "L1JetEtScaleRcd" ) )
	pL1TriggerKey->add( "L1JetEtScaleRcd",
			    "L1CaloEtScale",
			    gctKey ) ;
      if( listContains( "L1EmEtScaleRcd" ) )
	pL1TriggerKey->add( "L1EmEtScaleRcd",
			    "L1CaloEtScale",
			    gctKey ) ;
      if( listContains( "L1GctJetFinderParamsRcd" ) )
	pL1TriggerKey->add( "L1GctJetFinderParamsRcd",
			    "L1GctJetFinderParams",
			    gctKey ) ;
      if( listContains( "L1GctJetCalibFunRcd" ) )
	pL1TriggerKey->add( "L1GctJetCalibFunRcd",
			    "L1GctJetEtCalibrationFunction",
			    gctKey ) ;
      if( listContains( "L1GctJetCounterNegativeEtaRcd" ) )
	pL1TriggerKey->add( "L1GctJetCounterNegativeEtaRcd",
			    "L1GctJetCounterSetup",
			    gctKey ) ;
      if( listContains( "L1GctJetCounterPositiveEtaRcd" ) )
	pL1TriggerKey->add( "L1GctJetCounterPositiveEtaRcd",
			    "L1GctJetCounterSetup",
			    gctKey ) ;
      if( listContains( "L1CaloGeometryRecord" ) )
	pL1TriggerKey->add( "L1CaloGeometryRecord",
			    "L1CaloGeometry",
			    gctKey ) ;
      edm::LogVerbatim( "L1-O2O" ) << "GCT_KEY " << gctKey ;

      // ~~~~~~~~~~~~~~~~~~~~

      // SELECT PARTITION0_SETUP_FK FROM GT_SETUP WHERE GT_SETUP.ID =
      // GT_KEY [from subkeyResults]

      // In the future, do this only if TSP0_KEY is null.

      std::string gtKey = row[ "GT_KEY" ].data< std::string >() ;

//       l1t::OMDSReader::QueryResults gtPartitionKeyResults =
// 	m_omdsReader.basicQuery( "PARTITION0_SETUP_FK",
// 				 "CMS_GT",
// 				 "GT_SETUP",
// 				 "GT_SETUP.ID",
// 				 subkeyResults, "GT_KEY" ) ;

//       const coral::AttributeList& gtPartitionKeyRow =
// 	gtPartitionKeyResults.second.front() ;
//       std::string gtPartitionKey ;
// 	gtPartitionKeyRow[ "PARTITION0_SETUP_FK" ].data< std::string >() ;
 
      if( listContains( "L1GtPrescaleFactorsAlgoTrigRcd" ) )
	pL1TriggerKey->add( "L1GtPrescaleFactorsAlgoTrigRcd",
			    "L1GtPrescaleFactors",
			    gtKey ) ;
      if( listContains( "L1GtPrescaleFactorsTechTrigRcd" ) )
	pL1TriggerKey->add( "L1GtPrescaleFactorsTechTrigRcd",
			    "L1GtPrescaleFactors",
			    gtKey ) ;
      if( listContains( "L1GtTriggerMaskAlgoTrigRcd" ) )
	pL1TriggerKey->add( "L1GtTriggerMaskAlgoTrigRcd",
			  "L1GtTriggerMask",
			  gtKey ) ;
      //			  gtPartitionKey ) ;
      if( listContains( "L1GtTriggerMaskTechTrigRcd" ) )
	pL1TriggerKey->add( "L1GtTriggerMaskTechTrigRcd",
			    "L1GtTriggerMask",
			    gtKey ) ;
      //			    gtPartitionKey ) ;
      if( listContains( "L1GtTriggerMaskVetoAlgoTrigRcd" ) )
	pL1TriggerKey->add( "L1GtTriggerMaskVetoAlgoTrigRcd",
			  "L1GtTriggerMask",
			  gtKey ) ;
      //			  gtPartitionKey ) ;
      if( listContains( "L1GtTriggerMaskVetoTechTrigRcd" ) )
	pL1TriggerKey->add( "L1GtTriggerMaskVetoTechTrigRcd",
			    "L1GtTriggerMask",
			    gtKey ) ;
      //			    gtPartitionKey ) ;
      if( listContains( "L1GtParametersRcd" ) )
	pL1TriggerKey->add( "L1GtParametersRcd",
			    "L1GtParameters",
			    gtKey ) ;
      if( listContains( "L1GtStableParametersRcd" ) )
	pL1TriggerKey->add( "L1GtStableParametersRcd",
			    "L1GtStableParameters",
			    gtKey ) ;
      if( listContains( "L1GtBoardMapsRcd" ) )
	pL1TriggerKey->add( "L1GtBoardMapsRcd",
			    "L1GtBoardMaps",
			    gtKey ) ;
      if( listContains( "L1GtTriggerMenuRcd" ) )
	pL1TriggerKey->add( "L1GtTriggerMenuRcd",
			    "L1GtTriggerMenu",
			    gtKey ) ;

      edm::LogVerbatim( "L1-O2O" ) << "GT_KEY " << gtKey ;
      // edm::LogVerbatim( "L1-O2O" ) << "GT_PARTITION_KEY " << gtPartitionKey ;

      // Muon scales

      // SELECT SCALE_MUON_PT_REF, SCALE_MUON_ETA_REF,
      // SCALE_MUON_PHI_REF FROM L1T_SCALES WHERE L1T_SCALES.ID =
      // ( SELECT SCALES_FK FROM L1T_MENU WHERE L1T_MENU.ID =
      // ( SELECT L1T_MENU_FK FROM GT_SETUP WHERE GT_SETUP.ID =
      // GT_KEY [from subkeyResults] ) )

      std::vector< std::string > gtQueryStrings ;
      gtQueryStrings.push_back( "SC_MUON_ETA_FK" ) ;
      gtQueryStrings.push_back( "SC_MUON_PHI_FK" ) ;
      gtQueryStrings.push_back( "SC_MUON_PT_FK" ) ;

      l1t::OMDSReader::QueryResults muonScaleKeyResults =
	m_omdsReader.basicQuery(
	   gtQueryStrings,
	   "CMS_GT",
	   "L1T_SCALES",
	   "L1T_SCALES.ID",
	   m_omdsReader.basicQuery( "SCALES_KEY",
				    "CMS_GMT",
				    "GMT_CONFIG",
				    "GMT_CONFIG.KEY",
				    subkeyResults, "GMT_KEY" ) ) ;

      if( muonScaleKeyResults.second.size() == 1 ) // check query successful
	{
	  const coral::AttributeList& gtRow =
	    muonScaleKeyResults.second.front() ;
	  std::string gtMuEtaScaleId =
	    gtRow[ "SC_MUON_ETA_FK" ].data< std::string >() ;
	  std::string gtMuPhiScaleId =
	    gtRow[ "SC_MUON_PHI_FK" ].data< std::string >() ;
	  std::string gtMuPtScaleId =
	    gtRow[ "SC_MUON_PT_FK" ].data< std::string >() ;

	  if( listContains( "L1MuTriggerPtScaleRcd" ) )
	    pL1TriggerKey->add( "L1MuTriggerPtScaleRcd",
				"L1MuTriggerPtScale",
				gtMuPtScaleId ) ;

	  // Concatenate eta and phi keys, separated by comma;
	  // need to parse in L1TriggerConfigOnlineProd
	  std::string muGeomKey = gtMuEtaScaleId + "," + gtMuPhiScaleId ;
	  if( listContains( "L1MuTriggerScalesRcd" ) )
	    pL1TriggerKey->add( "L1MuTriggerScalesRcd",
				"L1MuTriggerScales",
				muGeomKey ) ;
	  if( listContains( "L1MuGMTScalesRcd" ) )
	    pL1TriggerKey->add( "L1MuGMTScalesRcd",
				"L1MuGMTScales",
				muGeomKey ) ;
	  edm::LogVerbatim("L1-O2O") << "Mu pt scale key " << gtMuPtScaleId ;
	  edm::LogVerbatim("L1-O2O") << "Mu eta scale key " << gtMuEtaScaleId ;
	  edm::LogVerbatim("L1-O2O") << "Mu phi scale key " << gtMuPhiScaleId ;

// 	  // Test string parsing
// 	  int loc = muGeomKey.find( "," ) ;
// 	  std::string etaKey = muGeomKey.substr( 0,      // start position
// 						 loc ) ; // length
// 	  int phiKeyLength = muGeomKey.size() - etaKey.size() - 1 ;
// 	  std::string phiKey = muGeomKey.substr( loc+1,
// 						 phiKeyLength ) ;
// 	  std::cout << "Parsed mu eta scale key " << etaKey ;
// 	  std::cout << "Parsed mu phi scale key " << phiKey ;
	}
      else
	{
	  edm::LogError( "L1-O2O")
	    << "Problem with object keys for L1MuTriggerPtScaleRcd, "
	    << "L1MuTriggerScalesRcd, L1MuGMTScalesRcd." ;
	}

      // Print out unknown records.
      std::map< std::string, bool >::const_iterator itr =
	m_recordsToInclude.begin() ;
      std::map< std::string, bool >::const_iterator end =
	m_recordsToInclude.end() ;
      for( ; itr != end ; ++itr )
	{
	  if( !itr->second )
	    {
	      edm::LogVerbatim( "L1-O2O" ) << "Unknown record ignored: "
					   << itr->first ;
	    }
	}
   }
   else
     {
       throw l1t::DataAlreadyPresentException(
	 "L1TriggerKey for TSC key " + m_tscKey + " already in CondDB." ) ;
     }

   return pL1TriggerKey ;
}

bool
L1TriggerKeyOnlineProd::listContains( const std::string& toMatch )
{
  std::map< std::string, bool >::iterator itr =
    m_recordsToInclude.find( toMatch ) ;
  //    find( m_recordsToInclude.begin(), m_recordsToInclude.end(), toMatch ) ;
  //  return itr != m_recordsToInclude.end() ;

  if( itr != m_recordsToInclude.end() )
    {
      itr->second = true ;
      return true ;
    }

  return false ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProd);
