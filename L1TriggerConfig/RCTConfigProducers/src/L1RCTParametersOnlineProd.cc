// -*- C++ -*-
//
// Package:    L1RCTParametersOnlineProd
// Class:      L1RCTParametersOnlineProd
// 
/**\class L1RCTParametersOnlineProd L1RCTParametersOnlineProd.h L1Trigger/L1RCTParametersProducers/src/L1RCTParametersOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
// $Id: L1RCTParametersOnlineProd.cc,v 1.1 2008/10/13 02:30:12 wsun Exp $
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

class L1RCTParametersOnlineProd :
  public L1ConfigOnlineProdBase< L1RCTParametersRcd, L1RCTParameters > {
   public:
      L1RCTParametersOnlineProd(const edm::ParameterSet&);
      ~L1RCTParametersOnlineProd();

  virtual boost::shared_ptr< L1RCTParameters > newObject(
    const std::string& objectKey ) ;

  void fillScaleFactors(
    const l1t::OMDSReader::QueryResults& results,
    std::vector< double >& output ) ;

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
L1RCTParametersOnlineProd::L1RCTParametersOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1RCTParametersRcd, L1RCTParameters >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


L1RCTParametersOnlineProd::~L1RCTParametersOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1RCTParameters >
L1RCTParametersOnlineProd::newObject( const std::string& objectKey )
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

     l1t::OMDSReader::QueryResults paremResults =
       m_omdsReader.basicQuery( queryStrings,
                                rctSchema,
                                "PAREM_CONF",
                                "PAREM_CONF.PAREM_KEY",
                                paremKeyResults ) ;

     if( paremResults.queryFailed() ||
	 paremResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1RCTParameters key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

     double eGammaLSB, jetMETLSB, eMinForFGCut, eMaxForFGCut, hOeCut ;
     double eMinForHoECut, eMaxForHoECut, hMinForHoECut ;
     double eActivityCut, hActivityCut ;
     double jscQuietThreshBarrel, jscQuietThreshEndcap, eicIsolationThreshold ;
     bool noiseVetoHB, noiseVetoHEplus, noiseVetoHEminus ;

     paremResults.fillVariable( "EGAMMA_LSB", eGammaLSB ) ;
     paremResults.fillVariable( "JETMET_LSB", jetMETLSB ) ;
     paremResults.fillVariable( "E_MIN_FOR_FG_CUT", eMinForFGCut ) ;
     paremResults.fillVariable( "E_MAX_FOR_FG_CUT", eMaxForFGCut ) ;
     paremResults.fillVariable( "H_OVER_E_CUT", hOeCut ) ;
     paremResults.fillVariable( "E_MIN_FOR_H_OVER_E_CUT", eMinForHoECut ) ;
     paremResults.fillVariable( "E_MAX_FOR_H_OVER_E_CUT", eMaxForHoECut ) ;
     paremResults.fillVariable( "H_MIN_FOR_H_OVER_E_CUT", hMinForHoECut ) ;
     paremResults.fillVariable( "E_ACTIVITY_CUT", eActivityCut ) ;
     paremResults.fillVariable( "H_ACTIVITY_CUT", hActivityCut ) ;
     paremResults.fillVariable( "JSC_QUIET_THRESHOLD_BARREL",
				jscQuietThreshBarrel ) ;
     paremResults.fillVariable( "JSC_QUIET_THRESHOLD_ENDCAP",
				jscQuietThreshEndcap ) ;
     paremResults.fillVariable( "EIC_ISOLATION_THRESHOLD",
				eicIsolationThreshold ) ;
     paremResults.fillVariable( "NOISEVETOHB", noiseVetoHB ) ;
     paremResults.fillVariable( "NOISEVETOHEPLUS", noiseVetoHEplus ) ;
     paremResults.fillVariable( "NOISEVETOHEMINUS", noiseVetoHEminus ) ;

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
     scaleFactorQueryStrings.push_back( "SCALEFACTOR" ) ;
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

     if( egammaEcalResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with EgammaEcal key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "egammaEcal " ;
     std::vector< double > egammaEcalScaleFactors ;
     fillScaleFactors( egammaEcalResults, egammaEcalScaleFactors ) ;

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

     if( egammaHcalResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with EgammaHcal key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "egammaHcal " ;
     std::vector< double > egammaHcalScaleFactors ;
     fillScaleFactors( egammaHcalResults, egammaHcalScaleFactors ) ;

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

     if( jetmetEcalResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with JetmetEcal key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "jetmetEcal " ;
     std::vector< double > jetmetEcalScaleFactors ;
     fillScaleFactors( jetmetEcalResults, jetmetEcalScaleFactors ) ;

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

     if( jetmetHcalResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with JetmetHcal key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "jetmetHcal " ;
     std::vector< double > jetmetHcalScaleFactors ;
     fillScaleFactors( jetmetHcalResults, jetmetHcalScaleFactors ) ;

     //~~~~~~~~~ Instantiate new L1RCTParameters object. ~~~~~~~~~

     // Default objects for Lindsey 

     return boost::shared_ptr< L1RCTParameters >(
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
                             (unsigned int) eicIsolationThreshold,
                             (int) jscQuietThreshBarrel,
                             (int) jscQuietThreshEndcap,
                             noiseVetoHB,
                             noiseVetoHEplus,
                             noiseVetoHEminus,
			     false, // useLindsey
                             egammaEcalScaleFactors,
                             egammaHcalScaleFactors,
                             jetmetEcalScaleFactors,
                             jetmetHcalScaleFactors,
			     std::vector<double>(),
			     std::vector<double>(),
			     std::vector<double>(),
			     std::vector<double>(),
			     std::vector<double>(),
			     std::vector<double>()
			     ) ) ;
}

//
// member functions
//

void
L1RCTParametersOnlineProd::fillScaleFactors(
  const l1t::OMDSReader::QueryResults& results,
  std::vector< double >& output )
{
  // Store scale factors in temporary array to get ordering right.
  // Reserve space for 100 bins.

  static const int reserve = 100 ;
  double sfTmp[ reserve ] ;
  for( int i = 0 ; i < reserve ; ++i )
    {
      sfTmp[ i ] = 0. ;
    }

  short maxBin = 0 ;
  for( int i = 0 ; i < results.numberRows() ; ++i )
    {
      double sf ;
      results.fillVariableFromRow( "SCALEFACTOR", i, sf ) ;

      short ieta ;
      results.fillVariableFromRow( "FK_RCT_ETA", i, ieta ) ;

      sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.

      if( ieta > maxBin )
	{
	  maxBin = ieta ;
	}
    }

//   std::cout << "maxBin = " << maxBin << std::endl ;
  for( short i = 0 ; i < maxBin ; ++i )
    {
      output.push_back( sfTmp[ i ] ) ;
//       std::cout << i+1 << " " << sfTmp[ i ] << std::endl ;
    }
}

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RCTParametersOnlineProd);
