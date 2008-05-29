// -*- C++ -*-
//
// Package:    L1TriggerConfigOnlineProd
// Class:      L1TriggerConfigOnlineProd
// 
/**\class L1TriggerConfigOnlineProd L1TriggerConfigOnlineProd.h CondTools/L1TriggerConfigOnlineProd/src/L1TriggerConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu May 29 23:36:18 CEST 2008
// $Id$
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerConfigOnlineProd.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

// ------------ method called to produce the data  ------------

boost::shared_ptr<L1RCTParameters>
L1TriggerConfigOnlineProd::produceL1RCTParameters( const L1RCTParametersRcd& iRecord )
{
   using namespace edm::es;
   boost::shared_ptr<L1RCTParameters> pL1RCTParameters ;

   // Get subsystem key and check if already in ORCON
   std::string key ;
   if( getSubsystemKey( iRecord, pL1RCTParameters, key ) ||
       m_forceGeneration )
   {
     // Key not in ORCON -- get data from OMDS and make C++ object
//      std::cout << "RCT key = " << key << std::endl ;

     // ~~~~~~~~~ Get rct_parameter from rct_key ~~~~~~~~~

     std::string table = "RCT_CONF" ;

     std::vector< std::string > queryStrings ;
     queryStrings.push_back( "RCT_PARAMETER" ) ;

     std::string condition = "RCT_CONF.RCT_KEY = :key" ;

     coral::AttributeList attributes ;
     attributes.extend( "key", typeid( std::string ) ) ;
     attributes[ "key" ].data< std::string >() = key ;

     boost::shared_ptr< coral::IQuery > query
       ( m_omdsReader.newQuery( table, queryStrings,
				condition, attributes ) ) ;
     coral::ICursor& cursor = query->execute() ;
     cursor.next() ;
     const coral::AttributeList& paremKeyAttributes = cursor.currentRow() ;

//      std::string rct_parameter =
//        paremKeyAttributes[ "RCT_PARAMETER" ].data< std::string >() ;
//      std::cout << "rct_parameter = " << rct_parameter << std::endl ;

     std::string paremKeyCondition =
       "PAREM_CONF.PAREM_KEY = :RCT_PARAMETER" ;

     // ~~~~~~~~~ Cut values ~~~~~~~~~

     // select egamma_lsb, jetmet_lsb, e_min_for_fg_cut, e_max_for_fg_cut, h_over_e_cut, e_min_for_h_over_e_cut, e_activity_cut, h_activity_cut, eic_isolation_threshold, noisevetohb, noisevetoheplus, noisevetoheminus from parem_conf where parem_conf.parem_key = (select rct_parameter from rct_conf where rct_conf.rct_key = 'rct_cmssw_def');

     table = "PAREM_CONF" ;

     queryStrings.clear() ;
     queryStrings.push_back( "EGAMMA_LSB" ) ;
     queryStrings.push_back( "JETMET_LSB" ) ;
     queryStrings.push_back( "E_MIN_FOR_FG_CUT" ) ;
     queryStrings.push_back( "E_MAX_FOR_FG_CUT" ) ;
     queryStrings.push_back( "H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_MIN_FOR_H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_MAX_FOR_H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_ACTIVITY_CUT" ) ;
     queryStrings.push_back( "H_ACTIVITY_CUT" ) ;
     queryStrings.push_back( "EIC_ISOLATION_THRESHOLD" ) ;
     queryStrings.push_back( "NOISEVETOHB" ) ;
     queryStrings.push_back( "NOISEVETOHEPLUS" ) ;
     queryStrings.push_back( "NOISEVETOHEMINUS" ) ;

     boost::shared_ptr< coral::IQuery > query2
       ( m_omdsReader.newQuery( table, queryStrings,
				paremKeyCondition, paremKeyAttributes ));
     coral::ICursor& cursor2 = query2->execute() ;
     cursor2.next() ;
     const coral::AttributeList& row2 = cursor2.currentRow() ;

     double eGammaLSB = row2[ "EGAMMA_LSB" ].data< double >() ;
     double jetMETLSB = row2[ "JETMET_LSB" ].data< double >() ;
     double eMinForFGCut = row2[ "E_MIN_FOR_FG_CUT" ].data< double >() ;
     double eMaxForFGCut = row2[ "E_MAX_FOR_FG_CUT" ].data< double >() ;
     double hOeCut = row2[ "H_OVER_E_CUT" ].data< double >() ;
     double eMinForHoECut = row2[ "E_MIN_FOR_H_OVER_E_CUT" ].data< double >();
     double eMaxForHoECut = row2[ "E_MAX_FOR_H_OVER_E_CUT" ].data< double >();
     double eActivityCut = row2[ "E_ACTIVITY_CUT" ].data< double >() ;
     double hActivityCut = row2[ "H_ACTIVITY_CUT" ].data< double >() ;
     unsigned int eicIsolationThreshold =
       ( unsigned int ) row2[ "EIC_ISOLATION_THRESHOLD" ].data< double >() ;
     bool noiseVetoHB = row2[ "NOISEVETOHB" ].data< bool >() ;
     bool noiseVetoHEplus = row2[ "NOISEVETOHEPLUS" ].data< bool >() ;
     bool noiseVetoHEminus = row2[ "NOISEVETOHEMINUS" ].data< bool >() ;

//      std::cout << "eGammaLSB = " << eGammaLSB << std::endl ;
//      std::cout << "jetMETLSB = " << jetMETLSB << std::endl ;
//      std::cout << "eMinForFGCut = " << eMinForFGCut << std::endl ;
//      std::cout << "eMaxForFGCut = " << eMaxForFGCut << std::endl ;
//      std::cout << "hOeCut = " << hOeCut << std::endl ;
//      std::cout << "eMinForHoECut = " << eMinForHoECut << std::endl ;
//      std::cout << "eMaxForHoECut = " << eMaxForHoECut << std::endl ;
//      std::cout << "eActivityCut = " << eActivityCut << std::endl ;
//      std::cout << "hActivityCut = " << hActivityCut << std::endl ;
//      std::cout << "eicIsolationThreshold = " << eicIsolationThreshold << std::endl ;
//      std::cout << "noiseVetoHB = " << noiseVetoHB << std::endl ;
//      std::cout << "noiseVetoHEplus = " << noiseVetoHEplus << std::endl ;
//      std::cout << "noiseVetoHEminus = " << noiseVetoHEminus << std::endl ;

     // ~~~~~~~~~ EGamma ECAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from egamma_ecal_scalefactor where egamma_ecal_scalefactor.fk_version = (select egamma_ecal from parem_conf where parem_conf.parem_key = (select rct_parameter from rct_conf where rct_conf.rct_key='rct_cmssw_def'));

     // First, get version number.

     table = "PAREM_CONF" ;
     queryStrings.clear() ;
     queryStrings.push_back( "EGAMMA_ECAL" ) ;

     // Use previous condition.
     boost::shared_ptr< coral::IQuery > query3
       ( m_omdsReader.newQuery( table, queryStrings,
				paremKeyCondition, paremKeyAttributes ));
     coral::ICursor& cursor3 = query3->execute() ;
     cursor3.next() ;
     const coral::AttributeList& egammaEcalVersion = cursor3.currentRow() ;

     // Now get scale factors

     table = "EGAMMA_ECAL_SCALEFACTOR" ;
     condition = "EGAMMA_ECAL_SCALEFACTOR.FK_VERSION = :EGAMMA_ECAL" ;

     std::vector< std::string > scaleFactorQueryStrings ;
     scaleFactorQueryStrings.push_back( "SCALEFACTOR") ;
     scaleFactorQueryStrings.push_back( "FK_RCT_ETA" ) ;

     boost::shared_ptr< coral::IQuery > query4
       ( m_omdsReader.newQuery( table, scaleFactorQueryStrings,
				condition, egammaEcalVersion ) ) ;
     coral::ICursor& cursor4 = query4->execute() ;

     // Store scale factors in temporary array to get ordering right.
     // Reserve space for 100 bins.

     static const int reserve = 100 ;
     double sfTmp[ reserve ] ;
     for( int i = 0 ; i < reserve ; ++i )
       {
	 sfTmp[ i ] = 0. ;
       }

     int maxBin = 0 ;
     while( cursor4.next() )
       {
	 const coral::AttributeList& row4 = cursor4.currentRow() ;
	 double sf = row4[ "SCALEFACTOR" ].data< double >() ;
	 int ieta = ( int ) row4[ "FK_RCT_ETA" ].data< short >() ;

	 sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.

	 if( ieta > maxBin )
	   {
	     maxBin = ieta ;
	   }
       }

//      std::cout << "egammaEcal maxBin = " << maxBin << std::endl ;
     std::vector< double > egammaEcalScaleFactors ;
     for( int i = 0 ; i < maxBin ; ++i )
       {
	 egammaEcalScaleFactors.push_back( sfTmp[ i ] ) ;
// 	 std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ EGamma HCAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from egamma_hcal_scalefactor where egamma_hcal_scalefactor.fk_version = (select egamma_hcal from parem_conf where parem_conf.parem_key = (select rct_parameter from rct_conf where rct_conf.rct_key='rct_cmssw_def'));

     // First, get version number.

     table = "PAREM_CONF" ;
     queryStrings.clear() ;
     queryStrings.push_back( "EGAMMA_HCAL" ) ;

     // Use previous condition.
     boost::shared_ptr< coral::IQuery > query5
       ( m_omdsReader.newQuery( table, queryStrings,
				paremKeyCondition, paremKeyAttributes ));
     coral::ICursor& cursor5 = query5->execute() ;
     cursor5.next() ;
     const coral::AttributeList& egammaHcalVersion = cursor5.currentRow() ;

     // Now get scale factors

     table = "EGAMMA_HCAL_SCALEFACTOR" ;
     condition = "EGAMMA_HCAL_SCALEFACTOR.FK_VERSION = :EGAMMA_HCAL" ;

     boost::shared_ptr< coral::IQuery > query6
       ( m_omdsReader.newQuery( table, scaleFactorQueryStrings,
				condition, egammaHcalVersion ) ) ;
     coral::ICursor& cursor6 = query6->execute() ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
	 sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     while( cursor6.next() )
       {
	 const coral::AttributeList& row6 = cursor6.currentRow() ;
	 double sf = row6[ "SCALEFACTOR" ].data< double >() ;
	 int ieta = ( int ) row6[ "FK_RCT_ETA" ].data< short >() ;

	 sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.

	 if( ieta > maxBin )
	   {
	     maxBin = ieta ;
	   }
       }

//      std::cout << "egammaHcal maxBin = " << maxBin << std::endl ;
     std::vector< double > egammaHcalScaleFactors ;
     for( int i = 0 ; i < maxBin ; ++i )
       {
	 egammaHcalScaleFactors.push_back( sfTmp[ i ] ) ;
// 	 std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ JetMET ECAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from jetmet_ecal_scalefactor where jetmet_ecal_scalefactor.fk_version = (select jetmet_ecal from parem_conf where parem_conf.parem_key = (select rct_parameter from rct_conf where rct_conf.rct_key='rct_cmssw_def'));

     // First, get version number.

     table = "PAREM_CONF" ;
     queryStrings.clear() ;
     queryStrings.push_back( "JETMET_ECAL" ) ;

     // Use previous condition.
     boost::shared_ptr< coral::IQuery > query7
       ( m_omdsReader.newQuery( table, queryStrings,
				paremKeyCondition, paremKeyAttributes ));
     coral::ICursor& cursor7 = query7->execute() ;
     cursor7.next() ;
     const coral::AttributeList& jetmetEcalVersion = cursor7.currentRow() ;

     // Now get scale factors

     table = "JETMET_ECAL_SCALEFACTOR" ;
     condition = "JETMET_ECAL_SCALEFACTOR.FK_VERSION = :JETMET_ECAL" ;

     boost::shared_ptr< coral::IQuery > query8
       ( m_omdsReader.newQuery( table, scaleFactorQueryStrings,
				condition, jetmetEcalVersion ) ) ;
     coral::ICursor& cursor8 = query8->execute() ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
	 sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     while( cursor8.next() )
       {
	 const coral::AttributeList& row8 = cursor8.currentRow() ;
	 double sf = row8[ "SCALEFACTOR" ].data< double >() ;
	 int ieta = ( int ) row8[ "FK_RCT_ETA" ].data< short >() ;

	 sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.

	 if( ieta > maxBin )
	   {
	     maxBin = ieta ;
	   }
       }

//      std::cout << "jetmetEcal maxBin = " << maxBin << std::endl ;
     std::vector< double > jetmetEcalScaleFactors ;
     for( int i = 0 ; i < maxBin ; ++i )
       {
	 jetmetEcalScaleFactors.push_back( sfTmp[ i ] ) ;
// 	 std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ JetMET HCAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from jetmet_hcal_scalefactor where jetmet_hcal_scalefactor.fk_version = (select jetmet_hcal from parem_conf where parem_conf.parem_key = (select rct_parameter from rct_conf where rct_conf.rct_key='rct_cmssw_def'));

     // First, get version number.

     table = "PAREM_CONF" ;
     queryStrings.clear() ;
     queryStrings.push_back( "JETMET_HCAL" ) ;

     // Use previous condition.
     boost::shared_ptr< coral::IQuery > query9
       ( m_omdsReader.newQuery( table, queryStrings,
				paremKeyCondition, paremKeyAttributes ));
     coral::ICursor& cursor9 = query9->execute() ;
     cursor9.next() ;
     const coral::AttributeList& jetmetHcalVersion = cursor9.currentRow() ;

     // Now get scale factors

     table = "JETMET_HCAL_SCALEFACTOR" ;
     condition = "JETMET_HCAL_SCALEFACTOR.FK_VERSION = :JETMET_HCAL" ;

     boost::shared_ptr< coral::IQuery > query10
       ( m_omdsReader.newQuery( table, scaleFactorQueryStrings,
				condition, jetmetHcalVersion ) ) ;
     coral::ICursor& cursor10 = query10->execute() ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
	 sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     while( cursor10.next() )
       {
	 const coral::AttributeList& row10 = cursor10.currentRow() ;
	 double sf = row10[ "SCALEFACTOR" ].data< double >() ;
	 int ieta = ( int ) row10[ "FK_RCT_ETA" ].data< short >() ;

	 sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.

	 if( ieta > maxBin )
	   {
	     maxBin = ieta ;
	   }
       }

//      std::cout << "jetmetHcal maxBin = " << maxBin << std::endl ;
     std::vector< double > jetmetHcalScaleFactors ;
     for( int i = 0 ; i < maxBin ; ++i )
       {
	 jetmetHcalScaleFactors.push_back( sfTmp[ i ] ) ;
// 	 std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }


     //~~~~~~~~~ Instantiate new L1RCTParameters object. ~~~~~~~~~

     pL1RCTParameters = boost::shared_ptr< L1RCTParameters >(
	new L1RCTParameters( eGammaLSB,
			     jetMETLSB,
			     eMinForFGCut,
			     eMaxForFGCut,
			     hOeCut,
			     eMinForHoECut,
			     eMaxForHoECut,
			     3.0, // hMinForHoECut,
			     eActivityCut,
			     hActivityCut,
			     eicIsolationThreshold,
			     3, // jscQuietThresholdBarrel,
			     3, // jscQuietThresholdEndcap,
			     noiseVetoHB,
			     noiseVetoHEplus,
			     noiseVetoHEminus,
			     egammaEcalScaleFactors,
			     egammaHcalScaleFactors,
			     jetmetEcalScaleFactors,
			     jetmetHcalScaleFactors ) ) ;
   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1RCTParameters for key " + key + " already in CondDB." ) ;
   }

   return pL1RCTParameters ;
}
