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
// $Id: DTTFParametersOnlineProd.cc,v 1.2 2008/11/10 08:30:43 troco Exp $
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
       void disablePHTF( 
         boost::shared_ptr< L1MuDTTFParameters >& dttfParameters,
         int nwh,    // wheel
         int isc ) ; // sector

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

     std::string dttfSchema = "CMS_DT_TF" ;

     // Order of strings is used below -- don't change!
     std::vector< std::string > crateKeyColumns ;
     crateKeyColumns.push_back( "WEDGE_CRATE_1" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_2" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_3" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_4" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_5" ) ;
     crateKeyColumns.push_back( "WEDGE_CRATE_6" ) ;
     crateKeyColumns.push_back( "HW_SETTINGS" ) ;

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

     // Wedge crate masks
     // Order of strings is used below -- don't change!
     std::vector< std::string > crateMaskColumns ;
     crateMaskColumns.push_back( "WEDGE_CRATE_1" ) ;
     crateMaskColumns.push_back( "WEDGE_CRATE_2" ) ;
     crateMaskColumns.push_back( "WEDGE_CRATE_3" ) ;
     crateMaskColumns.push_back( "WEDGE_CRATE_4" ) ;
     crateMaskColumns.push_back( "WEDGE_CRATE_5" ) ;
     crateMaskColumns.push_back( "WEDGE_CRATE_6" ) ;

     l1t::OMDSReader::QueryResults crateMaskResults =
       m_omdsReader.basicQuery( crateMaskColumns,
                                dttfSchema,
                                "DTTF_SETTINGS",
                                "DTTF_SETTINGS.ID",
                                crateKeyResults, "HW_SETTINGS" ) ;

     if( crateMaskResults.queryFailed() ||
	 crateMaskResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" )
	   << "Problem with DTTF_SETTINGS key." ;
	 return boost::shared_ptr< L1MuDTTFParameters >() ;
       }

     // Cache crate masks
     unsigned long crateMaskL[ 6 ] ;
     unsigned long crateMaskR[ 6 ] ;
     for( int icrate = 0 ; icrate < 6 ; ++icrate )
       {
	 std::string crateMask ;
	 crateMaskResults.fillVariable( crateMaskColumns[ icrate ],
					crateMask ) ;
         char* pEnd;
	 crateMaskL[ icrate ] = std::strtol( crateMask.c_str(), &pEnd, 16 ) ;
	 crateMaskR[ icrate ] = std::strtol( pEnd, (char **)NULL, 16 ) ;
	 std::cout << "Crate " << icrate << " masks"
		   << " L: " << std::hex << crateMaskL[ icrate ]
	           << " R: " << std::hex << crateMaskR[ icrate ] << std::endl ;
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

     // Map of sector+wheel name to bit number in crate mask
     std::map< std::string, unsigned int > crateMaskBitmap ;
     crateMaskBitmap.insert( std::make_pair( "N2",  0 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "N1",  4 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "N0",  8 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P0", 16 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P1", 20 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P2", 24 ) ) ;

     // Needed over and over later
     std::vector< std::string > phtfMaskColumns ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST1" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST2" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST3" ) ;
     phtfMaskColumns.push_back( "INREC_QUAL_ST4" ) ;
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
	 unsigned long crateMask = crateMaskL[ crateNumber ] ;
         if ( sectorNames[ isc ] == "R" ) crateMask = crateMaskR[ crateNumber ] ;
	 std::cout << "isc " << isc << " icr " << crateNumber << std::endl ;

	 // Loop over wheels 0-5
	 for( int iwh = 0 ; iwh < 6 ; ++iwh )
	   {
	     // Check appropriate bit of crate mask
	     // 0 = PHTF disabled, other = PHTF enabled
	     std::string sectorWheelName =
	       sectorNames[ isc ] + wheelNames[ iwh ] ;

	     unsigned int maskBit = 30 ;
	     std::map< std::string, unsigned int >::const_iterator itr =
	       crateMaskBitmap.find( wheelNames[ iwh ] ) ;
	     if( itr != crateMaskBitmap.end() )
	       {
		 maskBit = itr->second ;
	       }

	     unsigned long phtfEnabled = ( crateMask >> maskBit ) & 0xF ;

             if ( wheelNames[ iwh ] == "P2" ) phtfEnabled += ( crateMask >> 24 ) & 0x10 ;
             if ( wheelNames[ iwh ] == "N2" ) phtfEnabled += ( crateMask >> 25 ) & 0x10 ;

	     std::cout << "Bits " << maskBit << " (" << sectorWheelName
		       << ") of mask " << std::hex << crateMask << " is "
		       << std::hex << phtfEnabled << std::endl ;

	     int nwh = wheelNumbers[ iwh ] ;

	     disablePHTF( pDTTFParameters, nwh, isc ) ;

	     // Check if PHTF enabled
	     if( phtfEnabled & 0xF )
               {
	         unsigned long chmask = phtfEnabled & 0x1;
		 std::cout << " INREC_CHDIS_ST1 " << 1-chmask ;
		 pDTTFParameters->set_inrec_chdis_st1( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 1 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST2 " << 1-chmask ;
		 pDTTFParameters->set_inrec_chdis_st2( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 2 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST3 " << 1-chmask ;
		 pDTTFParameters->set_inrec_chdis_st3( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 3 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST4 " << 1-chmask ;
		 pDTTFParameters->set_inrec_chdis_st4( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 4 ) & 0x1;
		 std::cout << " INREC_QUAL_CSC " << 7*(1-chmask) << std::endl ;
		 pDTTFParameters->set_soc_qual_csc( nwh, isc, 7*(1-chmask) ) ;

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
       }

     return pDTTFParameters ;
}

//
// member functions
//

void
DTTFParametersOnlineProd::disablePHTF( 
  boost::shared_ptr< L1MuDTTFParameters >& dttfParameters,
  int nwh,  // wheel
  int isc ) // sector
{
  dttfParameters->set_inrec_chdis_st1( nwh, isc, true ) ;
  dttfParameters->set_inrec_chdis_st2( nwh, isc, true ) ;
  dttfParameters->set_inrec_chdis_st3( nwh, isc, true ) ;
  dttfParameters->set_inrec_chdis_st4( nwh, isc, true ) ;

  dttfParameters->set_inrec_qual_st1( nwh, isc, 0x7 ) ;
  dttfParameters->set_inrec_qual_st2( nwh, isc, 0x7 ) ;
  dttfParameters->set_inrec_qual_st3( nwh, isc, 0x7 ) ;
  dttfParameters->set_inrec_qual_st4( nwh, isc, 0x7 ) ;

  dttfParameters->set_soc_stdis_n( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_stdis_wl( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_stdis_wr( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_stdis_zl( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_stdis_zr( nwh, isc, 0x7 ) ;

  dttfParameters->set_soc_qcut_st1( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_qcut_st2( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_qcut_st4( nwh, isc, 0x7 ) ;
  dttfParameters->set_soc_qual_csc( nwh, isc, 0x7 ) ;

  dttfParameters->set_soc_run_21( nwh, isc, true ) ;
  dttfParameters->set_soc_nbx_del( nwh, isc, true ) ;
  dttfParameters->set_soc_csc_etacanc( nwh, isc, true ) ;
}

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTTFParametersOnlineProd);
