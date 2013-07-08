// -*- C++ -*-
//
// Package:    L1EmEtScaleOnlineProd
// Class:      L1EmEtScaleOnlineProd
// 
/**\class L1EmEtScaleOnlineProd L1EmEtScaleOnlineProd.h L1Trigger/L1EmEtScaleProducers/src/L1EmEtScaleOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
// $Id: L1EmEtScaleOnlineProd.cc,v 1.3 2012/06/18 10:17:56 eulisse Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"


//
// class declaration
//

class L1EmEtScaleOnlineProd :
  public L1ConfigOnlineProdBase< L1EmEtScaleRcd, L1CaloEtScale > {
   public:
      L1EmEtScaleOnlineProd(const edm::ParameterSet&);
      ~L1EmEtScaleOnlineProd();

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
L1EmEtScaleOnlineProd::L1EmEtScaleOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1EmEtScaleRcd, L1CaloEtScale >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


L1EmEtScaleOnlineProd::~L1EmEtScaleOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1CaloEtScale >
L1EmEtScaleOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;

     // ~~~~~~~~~ Cut values ~~~~~~~~~


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

     const l1t::OMDSReader::QueryResults scaleKeyResults =
       m_omdsReader.singleAttribute( objectKey ) ;

    
     l1t::OMDSReader::QueryResults scaleResults =
       m_omdsReader.basicQuery( queryStrings,
                                "CMS_GT",
                                "L1T_SCALE_CALO_ET_THRESHOLD",
                                "L1T_SCALE_CALO_ET_THRESHOLD.ID",
				m_omdsReader.basicQuery( "L1T_SCALE_CALO_ET_THRESHOLD_ID",
							"CMS_RCT",
							"L1CALOEMETTHRESH",
							"L1CALOEMETTHRESH.NAME",
							scaleKeyResults
							));



     if( scaleResults.queryFailed() ||
	 scaleResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1EmEtScale key." ;
	 return boost::shared_ptr< L1CaloEtScale >() ;
       }
     std::vector<double> m_thresholds;

     for( std::vector< std::string >::iterator thresh = queryStrings.begin();
	  thresh != queryStrings.end(); ++thresh) {
       float tempScale = 0.0;
       scaleResults.fillVariable(*thresh,tempScale);
       m_thresholds.push_back(tempScale);
     }
  
     l1t::OMDSReader::QueryResults lsbResults =
       m_omdsReader.basicQuery("INPUTLSB",
			       "CMS_RCT",
			       "L1CALOEMETTHRESH",
			       "L1CALOEMETTHRESH.NAME",
			       scaleKeyResults
			       );
   if( lsbResults.queryFailed() ||
	 lsbResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1EmEtScale key." ;
	 return boost::shared_ptr< L1CaloEtScale >() ;
       }

     double m_lsb = 0.;
     lsbResults.fillVariable(m_lsb);

     //     std::cout << " input lsb " << m_lsb <<std::endl;

     //~~~~~~~~~ Instantiate new L1EmEtScale object. ~~~~~~~~~

     // Default objects for Lindsey 

     return boost::shared_ptr< L1CaloEtScale >( new L1CaloEtScale( m_lsb,m_thresholds ) );
}


// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1EmEtScaleOnlineProd);
