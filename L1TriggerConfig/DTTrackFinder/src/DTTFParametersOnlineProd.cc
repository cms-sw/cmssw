// -*- C++ -*-
//
// Package:    DTTFParametersOnlineProd
// Class:      DTTFParametersOnlineProd
// 
/**\class DTTFParametersOnlineProd DTTFParametersOnlineProd.h L1TriggerConfig/DTTrackFinder/src/DTTFParametersOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Fri Oct  3 00:26:52 CEST 2008
// $Id: DTTFParametersOnlineProd.cc,v 1.7 2009/06/01 07:06:53 troco Exp $
//
//


// system include files
#include <iostream>

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"

//
// class declaration
//

class DTTFParametersOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTTFParametersRcd, L1MuDTTFParameters > {
   public:
      DTTFParametersOnlineProd(const edm::ParameterSet&);
      ~DTTFParametersOnlineProd();

      virtual boost::shared_ptr< L1MuDTTFParameters > newObject(
        const std::string& objectKey ) ;

   private:

      // ----------member data ---------------------------
};

//
// constructors and destructor
//
DTTFParametersOnlineProd::DTTFParametersOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTTFParametersRcd, L1MuDTTFParameters >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTTFParametersOnlineProd::~DTTFParametersOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTTFParameters >
DTTFParametersOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;

     boost::shared_ptr< L1MuDTTFParameters > pDTTFParameters(
       new L1MuDTTFParameters() ) ;

     pDTTFParameters->reset() ;

     std::string dttfSchema = "CMS_DT_TF" ;

     // Order of strings is used below -- don't change!
     std::vector< std::string > crateKeyColumns ;
     crateKeyColumns.push_back( "WEDGE_CRATE_1" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_2" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_3" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_4" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_5" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_6" ) ;

     l1t::OMDSReader::QueryResults crateKeyResults =
       m_omdsReader.basicQuery( crateKeyColumns,
                                dttfSchema,
                                "DTTF_CONF",
                                "DTTF_CONF.ID",
                                m_omdsReader.singleAttribute( objectKey ) ) ;

     if( crateKeyResults.queryFailed() ||
	 crateKeyResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" )
	   << "Problem with L1MuDTTFParameters key " << objectKey ;
	 return boost::shared_ptr< L1MuDTTFParameters >() ;
       }

     // print crate keys -- delete when done debugging
     std::string crateKeys[ 6 ] ;
     for( int icrate = 0 ; icrate < 6 ; ++icrate )
       {
	 crateKeyResults.fillVariable( crateKeyColumns[ icrate ],
				       crateKeys[ icrate ] ) ;
	 std::cout << "Crate " << icrate << " key "
		   << crateKeys[ icrate ] << std::endl ;
       }

     // Map of sector (0-11) to name (L/R)
     std::string sectorNames[ 12 ] = {
       "R", "L", "R", "L", "L", "R", "L", "R", "R", "L", "R", "L" } ;

     // Map of sector (0-11) to crate (0-5)
     int crateNumbers[ 12 ] = { 3, 3, 4, 4, 5, 5, 2, 2, 1, 1, 0, 0 } ;

     // Map of wheel array index to wheel number (+- 3, 2, 1).
     int wheelNumbers[ 6 ] = { -3, -2, -1, 1, 2, 3 } ;

     // Map of wheel array index to name ({N,P}{0,1,2}).
     std::string wheelNames[ 6 ] = { "N2", "N1", "N0", "P0", "P1", "P2" } ;

     // Needed over and over later
     std::vector< std::string > phtfMaskColumns ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST1" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST2" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST3" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST4" ) ;
     phtfMaskColumns.push_back( "SOC_QUAL_CSC" ) ;
     phtfMaskColumns.push_back( "SOC_STDIS_N" ) ;
     phtfMaskColumns.push_back( "SOC_STDIS_WL" ) ;
     phtfMaskColumns.push_back( "SOC_STDIS_WR" ) ;
     phtfMaskColumns.push_back( "SOC_STDIS_ZL" ) ;
     phtfMaskColumns.push_back( "SOC_STDIS_ZR" ) ;
     phtfMaskColumns.push_back( "SOC_QCUT_ST1" ) ;
     phtfMaskColumns.push_back( "SOC_QCUT_ST2" ) ;
     phtfMaskColumns.push_back( "SOC_QCUT_ST4" ) ;
     phtfMaskColumns.push_back( "SOC_RUN_21" ) ;
     phtfMaskColumns.push_back( "SOC_NBX_DEL" ) ;
     phtfMaskColumns.push_back( "SOC_CSC_ETACANC" ) ;
     phtfMaskColumns.push_back( "SOC_OPENLUT_EXTR" ) ;

     // Loop over sectors 0-11
     for( int isc = 0 ; isc < 12 ; ++isc )
       {
	 int crateNumber = crateNumbers[ isc ] ;
	 std::cout << "isc " << isc << " icr " << crateNumber << std::endl ;

	 // Loop over wheels 0-5
	 for( int iwh = 0 ; iwh < 6 ; ++iwh )
	   {
	     std::string sectorWheelName =
	       sectorNames[ isc ] + wheelNames[ iwh ] ;

	     int nwh = wheelNumbers[ iwh ] ;

	     // Check if non-null crate key
	     std::string crateKey ;
	     if( crateKeyResults.fillVariable( crateKeyColumns[ crateNumber ],
					       crateKey ) )
	       {
		 // Get PHTF key
		 std::vector< std::string > phtfKeyColumns ;
		 phtfKeyColumns.push_back( "PHTF_" + sectorWheelName ) ;

		 l1t::OMDSReader::QueryResults phtfKeyResults =
		   m_omdsReader.basicQuery( phtfKeyColumns,
					    dttfSchema,
					    "WEDGE_CRATE_CONF",
					    "WEDGE_CRATE_CONF.ID",
					    crateKeyResults,
					    crateKeyColumns[ crateNumber ] ) ;

		 if( phtfKeyResults.queryFailed() ||
		     phtfKeyResults.numberRows() != 1 )
		   {
		     edm::LogError( "L1-O2O" )
		       << "Problem with WEDGE_CRATE_CONF key." ;
		     return boost::shared_ptr< L1MuDTTFParameters >() ;
		   }
		 
		 std::string dummy ;
		 if( phtfKeyResults.fillVariable( dummy ) )
		   {
		     std::cout << "PHTF key " << dummy << std::endl ;
		     
		     l1t::OMDSReader::QueryResults phtfMaskResults =
		       m_omdsReader.basicQuery( phtfMaskColumns,
						dttfSchema,
						"PHTF_CONF",
						"PHTF_CONF.ID",
						phtfKeyResults ) ;
		     
		     if( phtfMaskResults.queryFailed() ||
			 phtfMaskResults.numberRows() != 1 )
		       {
			 edm::LogError( "L1-O2O" )
			   << "Problem with PHTF_CONF key." ;
			 return boost::shared_ptr< L1MuDTTFParameters >() ;
		       }
		     
		     long long tmp ;
		     
		     phtfMaskResults.fillVariable( "INREC_QUAL_ST1", tmp ) ;
		     std::cout << " INREC_QUAL_ST1 " << tmp ;
		     pDTTFParameters->set_inrec_qual_st1( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "INREC_QUAL_ST2", tmp ) ;
		     std::cout << " INREC_QUAL_ST2 " << tmp ;
		     pDTTFParameters->set_inrec_qual_st2( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "INREC_QUAL_ST3", tmp ) ;
		     std::cout << " INREC_QUAL_ST3 " << tmp ;
		     pDTTFParameters->set_inrec_qual_st3( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "INREC_QUAL_ST4", tmp ) ;
		     std::cout << " INREC_QUAL_ST4 " << tmp << std::endl ;
		     pDTTFParameters->set_inrec_qual_st4( nwh, isc, tmp ) ;
		     std::cout << " SOC_QUAL_CSC " << tmp << std::endl ;
		     pDTTFParameters->set_soc_qual_csc( nwh, isc, tmp ) ;
		     
		     phtfMaskResults.fillVariable( "SOC_STDIS_N", tmp ) ;
		     std::cout << " SOC_STDIS_N " << tmp ;
		     pDTTFParameters->set_soc_stdis_n( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_STDIS_WL", tmp ) ;
		     std::cout << " SOC_STDIS_WL " << tmp ;
		     pDTTFParameters->set_soc_stdis_wl( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_STDIS_WR", tmp ) ;
		     std::cout << " SOC_STDIS_WR " << tmp ;
		     pDTTFParameters->set_soc_stdis_wr( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_STDIS_ZL", tmp ) ;
		     std::cout << " SOC_STDIS_ZL " << tmp ;
		     pDTTFParameters->set_soc_stdis_zl( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_STDIS_ZR", tmp ) ;
		     std::cout << " SOC_STDIS_ZR " << tmp << std::endl ;
		     pDTTFParameters->set_soc_stdis_zr( nwh, isc, tmp ) ;
		     
		     phtfMaskResults.fillVariable( "SOC_QCUT_ST1", tmp ) ;
		     std::cout << " SOC_QCUT_ST1 " << tmp ;
		     pDTTFParameters->set_soc_qcut_st1( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_QCUT_ST2", tmp ) ;
		     std::cout << " SOC_QCUT_ST2 " << tmp ;
		     pDTTFParameters->set_soc_qcut_st2( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_QCUT_ST4", tmp ) ;
		     std::cout << " SOC_QCUT_ST4 " << tmp << std::endl ;
		     pDTTFParameters->set_soc_qcut_st4( nwh, isc, tmp ) ;
		     
		     phtfMaskResults.fillVariable( "SOC_RUN_21", tmp ) ;
		     std::cout << " SOC_RUN_21 " << tmp ;
		     pDTTFParameters->set_soc_run_21( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_NBX_DEL", tmp ) ;
		     std::cout << " SOC_NBX_DEL " << tmp ;
		     pDTTFParameters->set_soc_nbx_del( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_CSC_ETACANC", tmp ) ;
		     std::cout << " SOC_CSC_ETACANC " << tmp ;
		     pDTTFParameters->set_soc_csc_etacanc( nwh, isc, tmp ) ;
		     phtfMaskResults.fillVariable( "SOC_OPENLUT_EXTR", tmp ) ;
		     std::cout << " SOC_OPENLUT_EXTR " << tmp << std::endl ;
		     pDTTFParameters->set_soc_openlut_extr( nwh, isc, tmp ) ;
		   }
	       }
	   }
       }

     return pDTTFParameters ;
}

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTTFParametersOnlineProd);
