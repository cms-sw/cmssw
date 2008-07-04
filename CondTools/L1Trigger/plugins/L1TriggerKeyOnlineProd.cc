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
// $Id: L1TriggerKeyOnlineProd.cc,v 1.4 2008/06/23 20:04:35 wsun Exp $
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

#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

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
	  std::string tableString = "TRIGGERSUP_CONF" ;

	  std::vector< std::string > queryStrings ;
	  queryStrings.push_back( "CSCTF_KEY" ) ;
	  queryStrings.push_back( "DTTF_KEY" ) ;
	  queryStrings.push_back( "RPC_KEY" ) ;
	  queryStrings.push_back( "GMT_KEY" ) ;
	  queryStrings.push_back( "RCT_KEY" ) ;
	  queryStrings.push_back( "GCT_KEY" ) ;
	  queryStrings.push_back( "GT_KEY" ) ;
	  //	  queryStrings.push_back( "TSP0_KEY" ) ;

	  std::string conditionString = "TRIGGERSUP_CONF.TS_KEY = :tscKey" ;
	  coral::AttributeList attributes ;
	  attributes.extend( "tscKey", typeid( std::string ) ) ;
	  attributes[ "tscKey" ].data< std::string >() = m_tscKey ;

	  boost::shared_ptr< coral::IQuery > query
	    ( m_omdsReader.newQuery( tableString, queryStrings,
				     conditionString, attributes ) ) ;
	  coral::ICursor& cursor = query->execute() ;

	  cursor.next() ;
	  const coral::AttributeList& row = cursor.currentRow() ;

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

	  std::string gmtKey = row[ "GMT_KEY" ].data< std::string >() ;

	  std::vector< std::string > gmtQueryStrings1 ;
	  gmtQueryStrings1.push_back( "LUT_KEY" ) ;
	  std::string gmtConditionString1 = "GMT_CONFIG.KEY = :gmtKey" ;
	  coral::AttributeList gmtAttributes1 ;
	  gmtAttributes1.extend( "gmtKey", typeid( std::string ) ) ;
	  gmtAttributes1[ "gmtKey" ].data< std::string >() = gmtKey ;
	  boost::shared_ptr< coral::IQuery > gmtQuery1
	    ( m_omdsReader.newQuery( "GMT_CONFIG", gmtQueryStrings1,
				     gmtConditionString1, gmtAttributes1 ) ) ;
	  coral::ICursor& gmtCursor1 = gmtQuery1->execute() ;
	  gmtCursor1.next() ;
	  const coral::AttributeList& gmtRow1 = gmtCursor1.currentRow() ;
	  std::string gmtLutKey = gmtRow1[ "LUT_KEY" ].data< std::string >() ;

	  std::vector< std::string > gmtQueryStrings2 ;
	  gmtQueryStrings2.push_back( "GMT_SOFTWARE_CONFIG" ) ;
	  std::string gmtConditionString2 = "GMT_LUTS.KEY = :gmtLutKey" ;
	  coral::AttributeList gmtAttributes2 ;
	  gmtAttributes2.extend( "gmtLutKey", typeid( std::string ) ) ;
	  gmtAttributes2[ "gmtLutKey" ].data< std::string >() = gmtLutKey ;
	  boost::shared_ptr< coral::IQuery > gmtQuery2
	    ( m_omdsReader.newQuery( "GMT_LUTS", gmtQueryStrings2,
				     gmtConditionString2, gmtAttributes2 ) ) ;
	  coral::ICursor& gmtCursor2 = gmtQuery2->execute() ;
	  gmtCursor2.next() ;
	  const coral::AttributeList& gmtRow2 = gmtCursor2.currentRow() ;
	  std::string gmtSwKey =
	    gmtRow2[ "GMT_SOFTWARE_CONFIG" ].data< std::string >() ;

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

	  std::string gtKey = row[ "GT_KEY" ].data< std::string >() ;

	  std::vector< std::string > gtQueryStrings ;
	  gtQueryStrings.push_back( "PARTITION0_SETUP_FK" ) ;
	  std::string gtConditionString = "GT_SETUP.ID = :gtKey" ;
	  coral::AttributeList gtAttributes ;
	  gtAttributes.extend( "gtKey", typeid( std::string ) ) ;
	  gtAttributes[ "gtKey" ].data< std::string >() = gtKey ;
	  boost::shared_ptr< coral::IQuery > gtQuery
	    ( m_omdsReader.newQuery( "GT_SETUP", gtQueryStrings,
				     gtConditionString, gtAttributes ) ) ;
	  coral::ICursor& gtCursor = gtQuery->execute() ;
	  gtCursor.next() ;
	  const coral::AttributeList& gtRow = gtCursor.currentRow() ;
	  std::string gtPartitionKey =
	    gtRow[ "PARTITION0_SETUP_FK" ].data< std::string >() ;

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

	  std::vector< std::string > gtQueryStrings1 ;
	  gtQueryStrings1.push_back( "L1T_MENU_FK" ) ;
	  std::string gtConditionString1 = "GT_SETUP.ID = :gtKey" ;
	  coral::AttributeList gtAttributes1 ;
	  gtAttributes1.extend( "gtKey", typeid( std::string ) ) ;
	  gtAttributes1[ "gtKey" ].data< std::string >() = gtKey ;
	  boost::shared_ptr< coral::IQuery > gtQuery1
	    ( m_omdsReader.newQuery( "GT_SETUP", gtQueryStrings1,
				     gtConditionString1, gtAttributes1 ) ) ;
	  coral::ICursor& gtCursor1 = gtQuery1->execute() ;
	  gtCursor1.next() ;
	  const coral::AttributeList& gtRow1 = gtCursor1.currentRow() ;
	  std::string gtMenuId =
	    gtRow1[ "L1T_MENU_FK" ].data< std::string >() ;

	  std::vector< std::string > gtQueryStrings2 ;
	  gtQueryStrings2.push_back( "SCALES_FK" ) ;
	  std::string gtConditionString2 = "L1T_MENU.ID = :gtMenuId" ;
	  coral::AttributeList gtAttributes2 ;
	  gtAttributes2.extend( "gtMenuId", typeid( std::string ) ) ;
	  gtAttributes2[ "gtMenuId" ].data< std::string >() = gtMenuId ;
	  boost::shared_ptr< coral::IQuery > gtQuery2
	    ( m_omdsReader.newQuery( "L1T_MENU", gtQueryStrings2,
				     gtConditionString2, gtAttributes2 ) ) ;
	  coral::ICursor& gtCursor2 = gtQuery2->execute() ;
	  gtCursor2.next() ;
	  const coral::AttributeList& gtRow2 = gtCursor2.currentRow() ;
	  std::string gtScalesId =
	    gtRow2[ "SCALES_FK" ].data< std::string >() ;

	  std::vector< std::string > gtQueryStrings3 ;
	  gtQueryStrings3.push_back( "SCALE_MUON_PT_REF" ) ;
	  gtQueryStrings3.push_back( "SCALE_MUON_ETA_REF" ) ;
	  gtQueryStrings3.push_back( "SCALE_MUON_PHI_REF" ) ;
	  std::string gtConditionString3 = "L1T_SCALES.ID = :gtScalesId" ;
	  coral::AttributeList gtAttributes3 ;
	  gtAttributes3.extend( "gtScalesId", typeid( std::string ) ) ;
	  gtAttributes3[ "gtScalesId" ].data< std::string >() = gtScalesId ;
	  boost::shared_ptr< coral::IQuery > gtQuery3
	    ( m_omdsReader.newQuery( "L1T_SCALES", gtQueryStrings3,
				     gtConditionString3, gtAttributes3 ) ) ;
	  coral::ICursor& gtCursor3 = gtQuery3->execute() ;
	  gtCursor3.next() ;
	  const coral::AttributeList& gtRow3 = gtCursor3.currentRow() ;
	  std::string gtMuPtScaleId =
	    gtRow3[ "SCALE_MUON_PT_REF" ].data< std::string >() ;
	  std::string gtMuEtaScaleId =
	    gtRow3[ "SCALE_MUON_ETA_REF" ].data< std::string >() ;
	  std::string gtMuPhiScaleId =
	    gtRow3[ "SCALE_MUON_PHI_REF" ].data< std::string >() ;

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
