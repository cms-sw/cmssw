// -*- C++ -*-
//
// Package:    L1HtMissScaleOnlineProd
// Class:      L1HtMissScaleOnlineProd
// 
/**\class L1HtMissScaleOnlineProd L1HtMissScaleOnlineProd.h L1TriggerConfig/L1ScalesProducers/src/L1HtMissScaleOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
// $Id: L1HtMissScaleOnlineProd.cc,v 1.1 2009/03/26 15:48:09 jbrooke Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"


//
// class declaration
//

class L1HtMissScaleOnlineProd :
  public L1ConfigOnlineProdBase< L1HtMissScaleRcd, L1CaloEtScale > {
   public:
      L1HtMissScaleOnlineProd(const edm::ParameterSet&);
      ~L1HtMissScaleOnlineProd();

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
L1HtMissScaleOnlineProd::L1HtMissScaleOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1HtMissScaleRcd, L1CaloEtScale >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


L1HtMissScaleOnlineProd::~L1HtMissScaleOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1CaloEtScale >
L1HtMissScaleOnlineProd::newObject( const std::string& objectKey )
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
  l1t::OMDSReader::QueryResults htMissScaleKeyResults =
    m_omdsReader.basicQuery(
			    "SC_CENJET_ET_THRESHOLD_FK",
			    "CMS_GT",
			    "L1T_SCALES",
			    "L1T_SCALES.ID",
			    scalesKeyResults );
  
  std::string htMissScaleKey ;
  
  if( htMissScaleKeyResults.queryFailed() ) {
    edm::LogError("L1-O2O")
      << "Problem with key for L1HtMissScaleRcd : HF ring Et scale key query failed ";
  }
  else if( htMissScaleKeyResults.numberRows() != 1 ) {
    edm::LogError("L1-O2O")
      << "Problem with key for L1HtMissScaleRcd : "
      << (htMissScaleKeyResults.numberRows()) << " rows were returned when getting HF ring Et scale key";
  }
  else {
    htMissScaleKeyResults.fillVariable( htMissScaleKey ) ;
  }
  
  // thresholds
  std::vector< std::string > queryStrings ;
  queryStrings.push_back( "ET_GEV_BIN_LOW_0");
  queryStrings.push_back( "ET_GEV_BIN_LOW_1");
  queryStrings.push_back( "ET_GEV_BIN_LOW_2");
  queryStrings.push_back( "ET_GEV_BIN_LOW_3");
  queryStrings.push_back( "ET_GEV_BIN_LOW_4");
  queryStrings.push_back( "ET_GEV_BIN_LOW_5");
  queryStrings.push_back( "ET_GEV_BIN_LOW_6");
  queryStrings.push_back( "ET_GEV_BIN_LOW_7");
  queryStrings.push_back( "ET_GEV_BIN_LOW_8");
  queryStrings.push_back( "ET_GEV_BIN_LOW_9");
  queryStrings.push_back( "ET_GEV_BIN_LOW_10");
  queryStrings.push_back( "ET_GEV_BIN_LOW_11");
  queryStrings.push_back( "ET_GEV_BIN_LOW_12");
  queryStrings.push_back( "ET_GEV_BIN_LOW_13");
  queryStrings.push_back( "ET_GEV_BIN_LOW_14");
  queryStrings.push_back( "ET_GEV_BIN_LOW_15");
  queryStrings.push_back( "ET_GEV_BIN_LOW_16");
  queryStrings.push_back( "ET_GEV_BIN_LOW_17");
  queryStrings.push_back( "ET_GEV_BIN_LOW_18");
  queryStrings.push_back( "ET_GEV_BIN_LOW_19");
  queryStrings.push_back( "ET_GEV_BIN_LOW_20");
  queryStrings.push_back( "ET_GEV_BIN_LOW_21");
  queryStrings.push_back( "ET_GEV_BIN_LOW_22");
  queryStrings.push_back( "ET_GEV_BIN_LOW_23");
  queryStrings.push_back( "ET_GEV_BIN_LOW_24");
  queryStrings.push_back( "ET_GEV_BIN_LOW_25");
  queryStrings.push_back( "ET_GEV_BIN_LOW_26");
  queryStrings.push_back( "ET_GEV_BIN_LOW_27");
  queryStrings.push_back( "ET_GEV_BIN_LOW_28");
  queryStrings.push_back( "ET_GEV_BIN_LOW_29");
  queryStrings.push_back( "ET_GEV_BIN_LOW_30");
  queryStrings.push_back( "ET_GEV_BIN_LOW_31");
  queryStrings.push_back( "ET_GEV_BIN_LOW_32");
  queryStrings.push_back( "ET_GEV_BIN_LOW_33");
  queryStrings.push_back( "ET_GEV_BIN_LOW_34");
  queryStrings.push_back( "ET_GEV_BIN_LOW_35");
  queryStrings.push_back( "ET_GEV_BIN_LOW_36");
  queryStrings.push_back( "ET_GEV_BIN_LOW_37");
  queryStrings.push_back( "ET_GEV_BIN_LOW_38");
  queryStrings.push_back( "ET_GEV_BIN_LOW_39");
  queryStrings.push_back( "ET_GEV_BIN_LOW_40");
  queryStrings.push_back( "ET_GEV_BIN_LOW_41");
  queryStrings.push_back( "ET_GEV_BIN_LOW_42");
  queryStrings.push_back( "ET_GEV_BIN_LOW_43");
  queryStrings.push_back( "ET_GEV_BIN_LOW_44");
  queryStrings.push_back( "ET_GEV_BIN_LOW_45");
  queryStrings.push_back( "ET_GEV_BIN_LOW_46");
  queryStrings.push_back( "ET_GEV_BIN_LOW_47");
  queryStrings.push_back( "ET_GEV_BIN_LOW_48");
  queryStrings.push_back( "ET_GEV_BIN_LOW_49");
  queryStrings.push_back( "ET_GEV_BIN_LOW_50");
  queryStrings.push_back( "ET_GEV_BIN_LOW_51");
  queryStrings.push_back( "ET_GEV_BIN_LOW_52");
  queryStrings.push_back( "ET_GEV_BIN_LOW_53");
  queryStrings.push_back( "ET_GEV_BIN_LOW_54");
  queryStrings.push_back( "ET_GEV_BIN_LOW_55");
  queryStrings.push_back( "ET_GEV_BIN_LOW_56");
  queryStrings.push_back( "ET_GEV_BIN_LOW_57");
  queryStrings.push_back( "ET_GEV_BIN_LOW_58");
  queryStrings.push_back( "ET_GEV_BIN_LOW_59");
  queryStrings.push_back( "ET_GEV_BIN_LOW_60");
  queryStrings.push_back( "ET_GEV_BIN_LOW_61");
  queryStrings.push_back( "ET_GEV_BIN_LOW_62");
  queryStrings.push_back( "ET_GEV_BIN_LOW_63");
  
  std::vector<double> thresholds;

  
//   l1t::OMDSReader::QueryResults scaleResults =
//     m_omdsReader.basicQuery( queryStrings,
// 			     "CMS_GT",
// 			     "L1T_SCALE_CALO_ET_THRESHOLD",
// 			     "L1T_SCALE_CALO_ET_THRESHOLD.ID",
// 			     scaleKeyResults
// 			     );
  

  
//   if( scaleResults.queryFailed() ||
//       scaleResults.numberRows() != 1 ) // check query successful
//     {
//       edm::LogError( "L1-O2O" ) << "Problem with L1HtMissScale key." ;
//       return boost::shared_ptr< L1CaloEtScale >() ;
//     }
  
  
//   for( std::vector< std::string >::iterator thresh = queryStrings.begin();
//        thresh != queryStrings.end(); ++thresh) {
//     float tempScale;
//     scaleResults.fillVariable(*thresh,tempScale);
//     m_thresholds.push_back(tempScale);
//   }
  
  
  //~~~~~~~~~ Instantiate new L1HtMissScale object. ~~~~~~~~~  
  return boost::shared_ptr< L1CaloEtScale >( new L1CaloEtScale( 0, 0x7f, 0. ,thresholds ) );
}


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1HtMissScaleOnlineProd);
