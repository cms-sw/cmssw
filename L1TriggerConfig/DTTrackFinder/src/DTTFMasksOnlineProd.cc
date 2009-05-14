// -*- C++ -*-
//
// Package:    DTTFMasksOnlineProd
// Class:      DTTFMasksOnlineProd
// 
/**\class DTTFMasksOnlineProd DTTFMasksOnlineProd.h L1TriggerConfig/DTTrackFinder/src/DTTFMasksOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Troconiz - UAM Madrid
//         Created:  Fri Apr  3 00:26:52 CEST 2009
// $Id: DTTFMasksOnlineProd.cc,v 1.4 2008/11/24 14:43:35 troco Exp $
//
//


// system include files
#include <iostream>

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

//
// class declaration
//

class DTTFMasksOnlineProd :
  public L1ConfigOnlineProdBase< L1MuDTTFMasksRcd, L1MuDTTFMasks > {
   public:
      DTTFMasksOnlineProd(const edm::ParameterSet&);
      ~DTTFMasksOnlineProd();

      virtual boost::shared_ptr< L1MuDTTFMasks > newObject(
        const std::string& objectKey ) ;

   private:
       void disablePHTF( 
         boost::shared_ptr< L1MuDTTFMasks >& dttfMasks,
         int nwh,    // wheel
         int isc ) ; // sector

      // ----------member data ---------------------------
};

//
// constructors and destructor
//
DTTFMasksOnlineProd::DTTFMasksOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1MuDTTFMasksRcd, L1MuDTTFMasks >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


DTTFMasksOnlineProd::~DTTFMasksOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1MuDTTFMasks >
DTTFMasksOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;

     boost::shared_ptr< L1MuDTTFMasks > pDTTFMasks(
       new L1MuDTTFMasks() ) ;

     std::string dttfSchema = "CMS_DT_TF" ;

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
                                "DTTF_RUN_SETTINGS",
                                "DTTF_RUN_SETTINGS.ID",
                                m_omdsReader.singleAttribute( objectKey ) ) ;

     if( crateMaskResults.queryFailed() ||
	 crateMaskResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" )
	   << "Problem with L1MuDTTFMasks key " << objectKey ;
	 return boost::shared_ptr< L1MuDTTFMasks >() ;
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
     crateMaskBitmap.insert( std::make_pair( "N2", 24 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "N1", 20 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "N0", 16 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P0",  8 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P1",  4 ) ) ;
     crateMaskBitmap.insert( std::make_pair( "P2",  0 ) ) ;

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

	     std::cout << "Bits " << std::dec << maskBit << " (" << sectorWheelName
		       << ") of mask " << std::hex << crateMask << " is "
		       << std::hex << phtfEnabled << std::endl ;

	     int nwh = wheelNumbers[ iwh ] ;

	     disablePHTF( pDTTFMasks, nwh, isc ) ;

	     // Check if PHTF enabled
	     if( phtfEnabled & 0xF )
               {
	         unsigned long chmask = phtfEnabled & 0x1;
		 std::cout << " INREC_CHDIS_ST1 " << 1-chmask ;
		 pDTTFMasks->set_inrec_chdis_st1( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 1 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST2 " << 1-chmask ;
		 pDTTFMasks->set_inrec_chdis_st2( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 2 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST3 " << 1-chmask ;
		 pDTTFMasks->set_inrec_chdis_st3( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 3 ) & 0x1;
		 std::cout << " INREC_CHDIS_ST4 " << 1-chmask ;
		 pDTTFMasks->set_inrec_chdis_st4( nwh, isc, 1-chmask ) ;
		 chmask = ( phtfEnabled >> 4 ) & 0x1;
		 std::cout << " INREC_CHDIS_CSC " << 1-chmask << std::endl ;
		 pDTTFMasks->set_inrec_chdis_csc( nwh, isc, 1-chmask ) ;
	       }
	   }
       }

     return pDTTFMasks ;
}

//
// member functions
//

void
DTTFMasksOnlineProd::disablePHTF( 
  boost::shared_ptr< L1MuDTTFMasks >& dttfMasks,
  int nwh,  // wheel
  int isc ) // sector
{
  dttfMasks->set_inrec_chdis_st1( nwh, isc, true ) ;
  dttfMasks->set_inrec_chdis_st2( nwh, isc, true ) ;
  dttfMasks->set_inrec_chdis_st3( nwh, isc, true ) ;
  dttfMasks->set_inrec_chdis_st4( nwh, isc, true ) ;
  dttfMasks->set_inrec_chdis_csc( nwh, isc, true ) ;
}

// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DTTFMasksOnlineProd);
