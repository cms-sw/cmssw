// -*- C++ -*-
//
// Package:    L1HfRingEtScaleOnlineProd
// Class:      L1HfRingEtScaleOnlineProd
// 
/**\class L1HfRingEtScaleOnlineProd L1HfRingEtScaleOnlineProd.h L1TriggerConfig/L1ScalesProducers/src/L1HfRingEtScaleOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
// $Id: L1HfRingEtScaleOnlineProd.cc,v 1.5 2012/06/18 10:17:56 eulisse Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"


//
// class declaration
//

class L1HfRingEtScaleOnlineProd :
  public L1ConfigOnlineProdBase< L1HfRingEtScaleRcd, L1CaloEtScale > {
   public:
      L1HfRingEtScaleOnlineProd(const edm::ParameterSet&);
      ~L1HfRingEtScaleOnlineProd();

  virtual boost::shared_ptr< L1CaloEtScale > newObject(
    const std::string& objectKey ) ;


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
L1HfRingEtScaleOnlineProd::L1HfRingEtScaleOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1HfRingEtScaleRcd, L1CaloEtScale >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


L1HfRingEtScaleOnlineProd::~L1HfRingEtScaleOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1CaloEtScale >
L1HfRingEtScaleOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;

     // get scales keys
     l1t::OMDSReader::QueryResults scalesKeyResults =
       m_omdsReader.basicQuery(
			       "GCT_SCALES_KEY",
			       "CMS_GCT",
			       "GCT_PHYS_PARAMS",
			       "GCT_PHYS_PARAMS.CONFIG_KEY",
			       m_omdsReader.singleAttribute( objectKey ) );
     
     std::string scalesKey ;
     
     if( scalesKeyResults.queryFailed() ) {
       edm::LogError("L1-O2O")
	 << "Problem with key for L1JetEtScaleRcd : GCT scales key query failed ";
     }
     else if( scalesKeyResults.numberRows() != 1 ) {
       edm::LogError("L1-O2O")
	 << "Problem with key for L1JetEtScaleRcd : "
	 << (scalesKeyResults.numberRows()) << " rows were returned when getting GCT scales key";
     }
     else {
       scalesKeyResults.fillVariable( scalesKey );
     }
     

     // get ring scale key
     l1t::OMDSReader::QueryResults hfRingScaleKeyResults =
       m_omdsReader.basicQuery(
			       "SC_HF_ET_SUM_FK",
			       "CMS_GT",
			       "L1T_SCALES",
			       "L1T_SCALES.ID",
			       scalesKeyResults );

     std::string hfRingScaleKey ;

     if( hfRingScaleKeyResults.queryFailed() ) {
       edm::LogError("L1-O2O")
	 << "Problem with key for L1HfRingEtScaleRcd : HF ring Et scale key query failed ";
     }
     else if( hfRingScaleKeyResults.numberRows() != 1 ) {
       edm::LogError("L1-O2O")
	 << "Problem with key for L1HfRingEtScaleRcd : "
	 << (hfRingScaleKeyResults.numberRows()) << " rows were returned when getting HF ring Et scale key";
     }
     else {
       hfRingScaleKeyResults.fillVariable( hfRingScaleKey ) ;
     }
 
     // get thresholds
     std::vector< std::string > queryStrings ;
     queryStrings.push_back( "E_GEV_BIN_LOW_0");
     queryStrings.push_back( "E_GEV_BIN_LOW_1");
     queryStrings.push_back( "E_GEV_BIN_LOW_2");
     queryStrings.push_back( "E_GEV_BIN_LOW_3");
     queryStrings.push_back( "E_GEV_BIN_LOW_4");
     queryStrings.push_back( "E_GEV_BIN_LOW_5");
     queryStrings.push_back( "E_GEV_BIN_LOW_6");
     queryStrings.push_back( "E_GEV_BIN_LOW_7");

     l1t::OMDSReader::QueryResults scaleResults =
       m_omdsReader.basicQuery( queryStrings,
                                "CMS_GT",
                                "L1T_SCALE_HF_ET_SUM",
                                "L1T_SCALE_HF_ET_SUM.ID",
				hfRingScaleKeyResults
				);

     std::vector<double> thresholds;

     if( scaleResults.queryFailed() ) {
	 edm::LogError( "L1-O2O" ) << "Problem with L1HfRingEtScale key : scale query failed." ;
     }
     else if ( scaleResults.numberRows() != 1 ) {
	 edm::LogError( "L1-O2O" ) << "Problem with L1HfRingEtScale key : scale query failed." ;
     }
     else {
       for( std::vector< std::string >::iterator thresh = queryStrings.begin();
	    thresh != queryStrings.end(); ++thresh) {
	 float tempScale = 0.;
	 scaleResults.fillVariable(*thresh,tempScale);
	 thresholds.push_back(tempScale);
       }
     }

     // get region LSB
     double rgnEtLsb=0.;
     
     l1t::OMDSReader::QueryResults lsbResults =
       m_omdsReader.basicQuery( "GCT_RGN_ET_LSB",
				"CMS_GCT",
				"GCT_PHYS_PARAMS",
				"GCT_PHYS_PARAMS.CONFIG_KEY",
				m_omdsReader.singleAttribute( objectKey ) ) ;
     
     if( lsbResults.queryFailed() ) {
	 edm::LogError( "L1-O2O" ) << "Problem with L1JetEtScale key." ;
     }
     else {
       lsbResults.fillVariable( "GCT_RGN_ET_LSB", rgnEtLsb );
     }

     //~~~~~~~~~ Instantiate new L1HfRingEtScale object. ~~~~~~~~~
     return boost::shared_ptr< L1CaloEtScale >( new L1CaloEtScale(0xff, 0x7, rgnEtLsb, thresholds ) );
}


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1HfRingEtScaleOnlineProd);
