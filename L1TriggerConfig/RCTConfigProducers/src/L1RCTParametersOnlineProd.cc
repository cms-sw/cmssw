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
// $Id: L1RCTParametersOnlineProd.cc,v 1.4 2012/06/11 18:21:04 wmtan Exp $
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
    std::vector< double >& output, int nfactor = 1 ) ;
  /*

  void fillScaleFactors(
    const l1t::OMDSReader::QueryResults& results,
    std::vector< std::vector< double  > >& output ) ;
  */
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
     queryStrings.push_back( "USECORR" );
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
     bool noiseVetoHB, noiseVetoHEplus, noiseVetoHEminus, useCorr ;

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
     paremResults.fillVariable( "USECORR", useCorr ) ;
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



     //Lindsay variables
    
        /*
    std::vector< std::vector< double > > hcalCalibScaleFactors ;
       std::vector< std::vector< double > > hcalCalibHighScaleFactors ;
       std::vector< std::vector< double > > ecalCalibScaleFactors ;
       std::vector< std::vector< double > > crossTermsScaleFactors ;
     */
  std::vector< double > lowHoverE_smear, highHoverE_smear ;
     std::vector< double > hcalCalibScaleFactors,ecalCalibScaleFactors;
     std::vector< double > hcalCalibHighScaleFactors,crossTermsScaleFactors;

     if(useCorr) {  //lindsay corrections
 
       std::vector< std::string > scaleFactorQuery3Strings ;
       scaleFactorQuery3Strings.push_back( "SCALEFACTOR" ) ;
       scaleFactorQuery3Strings.push_back( "SF2" ) ;
       scaleFactorQuery3Strings.push_back( "SF3" ) ;
       scaleFactorQuery3Strings.push_back( "FK_RCT_ETA" ) ;

       l1t::OMDSReader::QueryResults hcalCalibResults =
	 m_omdsReader.basicQuery(
				 scaleFactorQuery3Strings,
				 rctSchema,
				 "HCAL_CALIB_FACTOR",
				 "HCAL_CALIB_FACTOR.VERSION",
				 m_omdsReader.basicQuery( "HCAL_CALIB_VERSION",
							  rctSchema,
							  "PAREM_CONF",
							  "PAREM_CONF.PAREM_KEY",
							  paremKeyResults ) ) ;

       if( hcalCalibResults.queryFailed() ) {// check query successful
	 
	 edm::LogError( "L1-O2O" ) << "Problem with JetmetHcal key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }
       
//      std::cout << "jetmetHcal " ;
       
       fillScaleFactors( hcalCalibResults, hcalCalibScaleFactors, 3 ) ;
     
       l1t::OMDSReader::QueryResults hcalCalibHighResults =
	 m_omdsReader.basicQuery(
				 scaleFactorQuery3Strings,
				 rctSchema,
				 "HCAL_CALIB_HIGH_FACTOR",
				 "HCAL_CALIB_HIGH_FACTOR.VERSION",
				 m_omdsReader.basicQuery( "HCAL_CALIB_HIGH_VERSION",
							  rctSchema,
							  "PAREM_CONF",
							  "PAREM_CONF.PAREM_KEY",
							  paremKeyResults ) ) ;

     if( hcalCalibHighResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with hcalHigh key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }



     fillScaleFactors( hcalCalibHighResults, hcalCalibHighScaleFactors,3 ) ;

     l1t::OMDSReader::QueryResults ecalCalibResults =
       m_omdsReader.basicQuery(
			       scaleFactorQuery3Strings,
			       rctSchema,
				 "ECAL_CALIB_FACTOR",
			       "ECAL_CALIB_FACTOR.VERSION",
			       m_omdsReader.basicQuery( "ECAL_CALIB_VERSION",
							  rctSchema,
							"PAREM_CONF",
							"PAREM_CONF.PAREM_KEY",
							paremKeyResults ) ) ;

     if( ecalCalibResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with ecal calib key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

     
     fillScaleFactors( ecalCalibResults, ecalCalibScaleFactors,3 ) ;
     
     
     std::vector< std::string > scaleFactorQuery6Strings ;
     scaleFactorQuery6Strings.push_back( "SCALEFACTOR" ) ;
     scaleFactorQuery6Strings.push_back( "SF2" ) ;
     scaleFactorQuery6Strings.push_back( "SF3" ) ;
     scaleFactorQuery6Strings.push_back( "SF4" ) ;
     scaleFactorQuery6Strings.push_back( "SF5" ) ;
     scaleFactorQuery6Strings.push_back( "SF6" ) ;
     scaleFactorQuery6Strings.push_back( "FK_RCT_ETA" ) ;
     l1t::OMDSReader::QueryResults crossTermResults =
       m_omdsReader.basicQuery(
			       scaleFactorQuery6Strings,
			       rctSchema,
			       "CROSS_TERMS_FACTOR",
			       "CROSS_TERMS_FACTOR.VERSION",
			       m_omdsReader.basicQuery( "CROSS_TERMS_VERSION",
							rctSchema,
							"PAREM_CONF",
							"PAREM_CONF.PAREM_KEY",
							paremKeyResults ) ) ;
     
     if( crossTermResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with crossTerms key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

     fillScaleFactors( crossTermResults, crossTermsScaleFactors,6 ) ;

     l1t::OMDSReader::QueryResults hoveresmearhighResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "H_OVER_E_SMEAR_HIGH_FACTOR",
         "H_OVER_E_SMEAR_HIGH_FACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "H_OVER_E_SMEAR_HIGH_VERSION",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     if( hoveresmearhighResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with low h over e smear key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "egammaEcal " ;
      fillScaleFactors( hoveresmearhighResults, highHoverE_smear ) ;


     l1t::OMDSReader::QueryResults hoveresmearlowResults =
       m_omdsReader.basicQuery(
         scaleFactorQueryStrings,
         rctSchema,
         "H_OVER_E_SMEAR_LOW_FACTOR",
         "H_OVER_E_SMEAR_LOW_FACTOR.FK_VERSION",
         m_omdsReader.basicQuery( "H_OVER_E_SMEAR_LOW_VERSION",
                                  rctSchema,
                                  "PAREM_CONF",
                                  "PAREM_CONF.PAREM_KEY",
                                  paremKeyResults ) ) ;

     if( hoveresmearlowResults.queryFailed() ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with low h over e smear key." ;
	 return boost::shared_ptr< L1RCTParameters >() ;
       }

//      std::cout << "egammaEcal " ;
      fillScaleFactors( hoveresmearlowResults, lowHoverE_smear ) ;
     }


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
			     useCorr, // useLindsey
                             egammaEcalScaleFactors,
                             egammaHcalScaleFactors,
                             jetmetEcalScaleFactors,
                             jetmetHcalScaleFactors,
			    ecalCalibScaleFactors,
			    hcalCalibScaleFactors,
			    hcalCalibHighScaleFactors,
			    crossTermsScaleFactors,
			     lowHoverE_smear,
			     highHoverE_smear
			     ) ) ;
}

//
// member functions
//

void
L1RCTParametersOnlineProd::fillScaleFactors(
  const l1t::OMDSReader::QueryResults& results,
  std::vector< double >& output, int nfactors )
{
  if( (nfactors < 1) || (nfactors > 6)){
    edm::LogError( "L1-O2O" ) <<"invalid number of factors in scale factors fill";
    return;
  }

    std::vector< std::string > scaleFactorQuery6Strings ;
    scaleFactorQuery6Strings.push_back( "SCALEFACTOR" ) ;
    scaleFactorQuery6Strings.push_back( "SF2" ) ;
    scaleFactorQuery6Strings.push_back( "SF3" ) ;
    scaleFactorQuery6Strings.push_back( "SF4" ) ;
    scaleFactorQuery6Strings.push_back( "SF5" ) ;
    scaleFactorQuery6Strings.push_back( "SF6" ) ;
       // Store scale factors in temporary array to get ordering right.
  // Reserve space for 100 bins.

    //  static const int reserve = 100 ;
  std::vector <double> sfTmp[100] ;
  /*  
  for( int i = 0 ; i < reserve ; ++i )
    {
      sfTmp[ i ] = 0. ;
    }
  */
  short maxBin = 0 ;
  for( int i = 0 ; i < results.numberRows() ; ++i )
    {
      double sf[6] ;
      for(int nf = 0; nf < nfactors; nf++){
	results.fillVariableFromRow( scaleFactorQuery6Strings.at(nf), i, sf[nf] ) ;
      }
      short ieta = 0;
      results.fillVariableFromRow( "FK_RCT_ETA", i, ieta ) ;
      
      for(int nf = 0; nf< nfactors; nf++)
	//      sfTmp[ ieta-1 ] = sf ; // eta bins start at 1.
	sfTmp[ieta-1].push_back(sf[nf]);

      if( ieta > maxBin )
	{
	  maxBin = ieta ;
	}
    }
  

  for( short i = 0 ; i < maxBin ; ++i )
    {
      for( short nf = 0; nf < nfactors; nf++)
	output.push_back( sfTmp[ i ].at(nf) ) ;
      
    }
}

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RCTParametersOnlineProd);
