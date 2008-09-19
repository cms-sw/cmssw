// -*- C++ -*-
//
// Package:    RCTConfigOnlineProd
// Class:      RCTConfigOnlineProd
// 
/**\class RCTConfigOnlineProd RCTConfigOnlineProd.h L1Trigger/RCTConfigProducers/src/RCTConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
// $Id$
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

// #include "FWCore/Framework/interface/HCTypeTagTemplate.h"
// #include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

class RCTConfigOnlineProd : public L1ConfigOnlineProdBase< L1RCTParametersRcd,
							   L1RCTParameters > {
   public:
      RCTConfigOnlineProd(const edm::ParameterSet&);
      ~RCTConfigOnlineProd();

  virtual void makeObject( const std::string& objectKey,
			   boost::shared_ptr< L1RCTParameters >& output ) ;
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
RCTConfigOnlineProd::RCTConfigOnlineProd(const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1RCTParametersRcd, L1RCTParameters >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


RCTConfigOnlineProd::~RCTConfigOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void
RCTConfigOnlineProd::makeObject( const std::string& objectKey,
				 boost::shared_ptr< L1RCTParameters >& output )
{
     using namespace edm::es;

     std::string rctSchema = "CMS_RCT" ;
     const l1t::OMDSReader::QueryResults paremKeyResults =
       m_omdsReader.singleAttribute( objectKey ) ;

     // ~~~~~~~~~ Cut values ~~~~~~~~~

     // select egamma_lsb, jetmet_lsb, e_min_for_fg_cut, e_max_for_fg_cut,
     // h_ove r_e_cut, e_min_for_h_over_e_cut, e_max_for_h_over_e_cut,
     // h_min_for_h_over_e_cut, e_activity_cut, h_activity_cut,
     // eic_isolation_threshold, jsc_quiet_threshold_barrel,
     // jsc_quiet_threshold_endcap, noisevetohb, noisevetoheplus,
     // noisevetoheminus from parem_conf where parem_conf.parem_key =
     // (select rct_parameter from rct_conf where rct_conf.rct_key =
     // 'rct_cmssw_def');

     std::vector< std::string > queryStrings ;
     queryStrings.push_back( "EGAMMA_LSB" ) ;
     queryStrings.push_back( "JETMET_LSB" ) ;
     queryStrings.push_back( "E_MIN_FOR_FG_CUT" ) ;
     queryStrings.push_back( "E_MAX_FOR_FG_CUT" ) ;
     queryStrings.push_back( "H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_MIN_FOR_H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_MAX_FOR_H_OVER_E_CUT" ) ;
     queryStrings.push_back( "H_MIN_FOR_H_OVER_E_CUT" ) ;
     queryStrings.push_back( "E_ACTIVITY_CUT" ) ;
     queryStrings.push_back( "H_ACTIVITY_CUT" ) ;
     queryStrings.push_back( "EIC_ISOLATION_THRESHOLD" ) ;
     queryStrings.push_back( "JSC_QUIET_THRESHOLD_BARREL" ) ;
     queryStrings.push_back( "JSC_QUIET_THRESHOLD_ENDCAP" ) ;
     queryStrings.push_back( "NOISEVETOHB" ) ;
     queryStrings.push_back( "NOISEVETOHEPLUS" ) ;
     queryStrings.push_back( "NOISEVETOHEMINUS" ) ;

     l1t::OMDSReader::QueryResults results2 =
       m_omdsReader.basicQuery( queryStrings,
                                rctSchema,
                                "PAREM_CONF",
                                "PAREM_CONF.PAREM_KEY",
                                paremKeyResults ) ;

     if( results2.second.size() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1RCTParameters key." ;
	 return ;
       }

     const coral::AttributeList& row2 = results2.second.front() ;

     double eGammaLSB = row2[ "EGAMMA_LSB" ].data< double >() ;
     double jetMETLSB = row2[ "JETMET_LSB" ].data< double >() ;
     double eMinForFGCut = row2[ "E_MIN_FOR_FG_CUT" ].data< double >() ;
     double eMaxForFGCut = row2[ "E_MAX_FOR_FG_CUT" ].data< double >() ;
     double hOeCut = row2[ "H_OVER_E_CUT" ].data< double >() ;
     double eMinForHoECut = row2[ "E_MIN_FOR_H_OVER_E_CUT" ].data< double >();
     double eMaxForHoECut = row2[ "E_MAX_FOR_H_OVER_E_CUT" ].data< double >();
     double hMinForHoECut = row2[ "H_MIN_FOR_H_OVER_E_CUT" ].data< double >();
     double eActivityCut = row2[ "E_ACTIVITY_CUT" ].data< double >() ;
     double hActivityCut = row2[ "H_ACTIVITY_CUT" ].data< double >() ;
     double jscQuietThreshBarrel =
       row2[ "JSC_QUIET_THRESHOLD_BARREL" ].data< double >() ;
     double jscQuietThreshEndcap =
       row2[ "JSC_QUIET_THRESHOLD_ENDCAP" ].data< double >() ;
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
//      std::cout << "hMinForHoECut = " << hMinForHoECut << std::endl ;
//      std::cout << "eActivityCut = " << eActivityCut << std::endl ;
//      std::cout << "hActivityCut = " << hActivityCut << std::endl ;
//      std::cout << "eicIsolationThreshold = " << eicIsolationThreshold << std::endl ;
//      std::cout << "jscQuietThreshBarrel = " << jscQuietThreshBarrel << std::endl ;
//      std::cout << "jscQuietThreshEndcap = " << jscQuietThreshEndcap << std::endl ;
//      std::cout << "noiseVetoHB = " << noiseVetoHB << std::endl ;
//      std::cout << "noiseVetoHEplus = " << noiseVetoHEplus << std::endl ;
//      std::cout << "noiseVetoHEminus = " << noiseVetoHEminus << std::endl ;

     // ~~~~~~~~~ EGamma ECAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from egamma_ecal_scalefactor where
     // egamma_ecal_scalefactor.fk_version =
     // (select egamma_ecal from parem_conf where parem_conf.parem_key =
     // (select rct_parameter from rct_conf where rct_conf.rct_key=
     // 'rct_cmssw_def'));

     std::vector< std::string > scaleFactorQueryStrings ;
     scaleFactorQueryStrings.push_back( "SCALEFACTOR") ;
     scaleFactorQueryStrings.push_back( "FK_RCT_ETA" ) ;

     l1t::OMDSReader::QueryResults egammaEcalResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "EGAMMA_ECAL_SCALEFACTOR",
         "EGAMMA_ECAL_SCALEFACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "EGAMMA_ECAL",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     // Store scale factors in temporary array to get ordering right.
     // Reserve space for 100 bins.

     static const int reserve = 100 ;
     double sfTmp[ reserve ] ;
     for( int i = 0 ; i < reserve ; ++i )
       {
         sfTmp[ i ] = 0. ;
       }

     int maxBin = 0 ;
     std::vector< coral::AttributeList >::const_iterator itr =
       egammaEcalResults.second.begin() ;
     std::vector< coral::AttributeList >::const_iterator end =
       egammaEcalResults.second.end() ;
     for( ; itr != end ; ++itr )
       {
         const coral::AttributeList& row = *itr ;
         double sf = row[ "SCALEFACTOR" ].data< double >() ;
         int ieta = ( int ) row[ "FK_RCT_ETA" ].data< short >() ;

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
//       std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ EGamma HCAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from egamma_hcal_scalefactor where
     // egamma_hcal_scalefactor.fk_version =
     // (select egamma_hcal from parem_conf where parem_conf.parem_key =
     // (select rct_parameter from rct_conf where rct_conf.rct_key=
     // 'rct_cmssw_def'));

     l1t::OMDSReader::QueryResults egammaHcalResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "EGAMMA_HCAL_SCALEFACTOR",
         "EGAMMA_HCAL_SCALEFACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "EGAMMA_HCAL",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
         sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     itr = egammaHcalResults.second.begin() ;
     end = egammaHcalResults.second.end() ;
     for( ; itr != end ; ++itr )
       {
         const coral::AttributeList& row = *itr ;
         double sf = row[ "SCALEFACTOR" ].data< double >() ;
         int ieta = ( int ) row[ "FK_RCT_ETA" ].data< short >() ;

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
//       std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ JetMET ECAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from jetmet_ecal_scalefactor where
     // jetmet_ecal_scalefactor.fk_version =
     // (select jetmet_ecal from parem_conf where parem_conf.parem_key =
     // (select rct_parameter from rct_conf where rct_conf.rct_key=
     // 'rct_cmssw_def'));

     l1t::OMDSReader::QueryResults jetmetEcalResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "JETMET_ECAL_SCALEFACTOR",
         "JETMET_ECAL_SCALEFACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "JETMET_ECAL",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
         sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     itr = jetmetEcalResults.second.begin() ;
     end = jetmetEcalResults.second.end() ;
     for( ; itr != end ; ++itr )
       {
         const coral::AttributeList& row = *itr ;
         double sf = row[ "SCALEFACTOR" ].data< double >() ;
         int ieta = ( int ) row[ "FK_RCT_ETA" ].data< short >() ;

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
//       std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     // ~~~~~~~~~ JetMET HCAL scale factors ~~~~~~~~~

     // select scalefactor, fk_rct_eta from jetmet_hcal_scalefactor where
     // jetmet_hcal_scalefactor.fk_version =
     // (select jetmet_hcal from parem_conf where parem_conf.parem_key =
     // (select rct_parameter from rct_conf where rct_conf.rct_key=
     // 'rct_cmssw_def'));

     l1t::OMDSReader::QueryResults jetmetHcalResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "JETMET_HCAL_SCALEFACTOR",
         "JETMET_HCAL_SCALEFACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "JETMET_HCAL",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     // Store scale factors in temporary array to get ordering right.
     for( int i = 0 ; i < reserve ; ++i )
       {
         sfTmp[ i ] = 0. ;
       }

     maxBin = 0 ;
     itr = jetmetHcalResults.second.begin() ;
     end = jetmetHcalResults.second.end() ;
     for( ; itr != end ; ++itr )
       {
         const coral::AttributeList& row = *itr ;
         double sf = row[ "SCALEFACTOR" ].data< double >() ;
         int ieta = ( int ) row[ "FK_RCT_ETA" ].data< short >() ;

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
//       std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
       }

     //~~~~~~~~~ Instantiate new L1RCTParameters object. ~~~~~~~~~

     output = boost::shared_ptr< L1RCTParameters >(
        new L1RCTParameters( eGammaLSB,
                             jetMETLSB,
                             eMinForFGCut,
                             eMaxForFGCut,
                             hOeCut,
                             eMinForHoECut,
                             eMaxForHoECut,
                             hMinForHoECut,
                             eActivityCut,
                             hActivityCut,
                             eicIsolationThreshold,
                             (int) jscQuietThreshBarrel,
                             (int) jscQuietThreshEndcap,
                             noiseVetoHB,
                             noiseVetoHEplus,
                             noiseVetoHEminus,
                             egammaEcalScaleFactors,
                             egammaHcalScaleFactors,
                             jetmetEcalScaleFactors,
                             jetmetHcalScaleFactors ) ) ;
}

//
// member functions
//

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTConfigOnlineProd);
