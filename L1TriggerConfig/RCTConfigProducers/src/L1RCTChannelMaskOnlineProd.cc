
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
// $Id: L1RCTChannelMaskOnlineProd.cc,v 1.3 2012/06/11 18:21:04 wmtan Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

// #include "FWCore/Framework/interface/HCTypeTagTemplate.h"
// #include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

class L1RCTChannelMaskOnlineProd :
  public L1ConfigOnlineProdBase< L1RCTChannelMaskRcd, L1RCTChannelMask > {
   public:
  L1RCTChannelMaskOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase< L1RCTChannelMaskRcd, L1RCTChannelMask > (iConfig) {}
  ~L1RCTChannelMaskOnlineProd() {}
  
  virtual boost::shared_ptr< L1RCTChannelMask > newObject(const std::string& objectKey ) ;


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

boost::shared_ptr< L1RCTChannelMask >
L1RCTChannelMaskOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;


      std::cout << " Current key is " << objectKey <<std::endl;

     std::string rctSchema = "CMS_RCT" ;

     std::vector< std::string > dc_maskStrings;
     dc_maskStrings.push_back( "HCAL_MASK_CR00_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR00_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR01_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR01_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR02_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR02_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR03_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR03_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR04_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR04_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR05_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR05_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR06_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR06_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR07_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR07_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR08_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR08_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR09_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR09_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR10_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR10_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR11_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR11_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR12_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR12_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR13_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR13_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR14_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR14_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR15_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR15_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR16_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR16_ODD" );
     dc_maskStrings.push_back( "HCAL_MASK_CR17_EVEN" );
     dc_maskStrings.push_back( "HCAL_MASK_CR17_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR00_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR00_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR01_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR01_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR02_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR02_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR03_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR03_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR04_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR04_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR05_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR05_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR06_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR06_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR07_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR07_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR08_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR08_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR09_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR09_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR10_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR10_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR11_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR11_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR12_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR12_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR13_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR13_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR14_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR14_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR15_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR15_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR16_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR16_ODD" );
     dc_maskStrings.push_back( "ECAL_MASK_CR17_EVEN" );
     dc_maskStrings.push_back( "ECAL_MASK_CR17_ODD" );

     l1t::OMDSReader::QueryResults dcMaskResults =
     m_omdsReader.basicQuery(
			     dc_maskStrings,
			     rctSchema,
			     "RCT_DEADCHANNEL_SUMMARY",
			     "RCT_DEADCHANNEL_SUMMARY.ID",
			     m_omdsReader.basicQuery( "DC_SUM_ID",
                                rctSchema,
                                "RCT_RUN_SETTINGS_KEY",
                                "RCT_RUN_SETTINGS_KEY.ID",
                                m_omdsReader.singleAttribute( objectKey)))  ;



     if( dcMaskResults.queryFailed() ||
	 dcMaskResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1RCTChannelMask key." ;
	 

	 std::cout << " Returened rows " << dcMaskResults.numberRows() <<std::endl;
	 return boost::shared_ptr< L1RCTChannelMask >() ;
       }
     
      L1RCTChannelMask* m = new L1RCTChannelMask;

     long long hcal_temp = 0LL;
     int ecal_temp = 0;
     for(int i = 0 ; i < 36 ; i++) {
       dcMaskResults.fillVariable(dc_maskStrings.at(i),hcal_temp);
       for(int j = 0; j < 32 ;  j++) 
	 if(j< 28)
	   m->hcalMask[i/2][i%2][j] = ((hcal_temp >> j ) & 1) == 1;
	 else
	   m->hfMask[i/2][i%2][j-28] = ((hcal_temp >> j ) & 1) == 1;
     }
     for(int i = 36; i < 72 ; i++) {
       dcMaskResults.fillVariable(dc_maskStrings.at(i),ecal_temp);
       for(int j = 0; j < 28 ;  j++) {
	 int k = i - 36;
	 m->ecalMask[k/2][k%2][j] = ((ecal_temp >> j ) & 1) ==1;
       }
     }

     // FIND dummy cards from TSC key in crate conf
     

     
     std::vector< std::string > cardMaskStrings;
     cardMaskStrings.push_back("RC0");
     cardMaskStrings.push_back("RC1");
     cardMaskStrings.push_back("RC2");
     cardMaskStrings.push_back("RC3");
     cardMaskStrings.push_back("RC4");
     cardMaskStrings.push_back("RC5");
     cardMaskStrings.push_back("RC6");
     cardMaskStrings.push_back("JSC");
     

     std::vector< std::string > crateIDStrings;
     crateIDStrings.push_back("RCT_CRATE_0");
     crateIDStrings.push_back("RCT_CRATE_1");
     crateIDStrings.push_back("RCT_CRATE_2");
     crateIDStrings.push_back("RCT_CRATE_3");
     crateIDStrings.push_back("RCT_CRATE_4");
     crateIDStrings.push_back("RCT_CRATE_5");
     crateIDStrings.push_back("RCT_CRATE_6");
     crateIDStrings.push_back("RCT_CRATE_7");
     crateIDStrings.push_back("RCT_CRATE_8");
     crateIDStrings.push_back("RCT_CRATE_9");
     crateIDStrings.push_back("RCT_CRATE_10");
     crateIDStrings.push_back("RCT_CRATE_11");
     crateIDStrings.push_back("RCT_CRATE_12");
     crateIDStrings.push_back("RCT_CRATE_13");
     crateIDStrings.push_back("RCT_CRATE_14");
     crateIDStrings.push_back("RCT_CRATE_15");
     crateIDStrings.push_back("RCT_CRATE_16");
     crateIDStrings.push_back("RCT_CRATE_17");


     l1t::OMDSReader::QueryResults crate_conf =
       m_omdsReader.basicQuery( "CRATE_CONF",
                                rctSchema,
                                "RCT_RUN_SETTINGS_KEY",
                                "RCT_RUN_SETTINGS_KEY.ID",
                                m_omdsReader.singleAttribute( objectKey))  ;
			       
     int crateNum = 0;
     for( std::vector<std::string>::iterator crate = crateIDStrings.begin(); crate !=crateIDStrings.end() ; ++crate) {
       //       std::cout << "crate conf " << *crate <<std::endl;
       l1t::OMDSReader::QueryResults cardConfResults =
	 m_omdsReader.basicQuery(cardMaskStrings,
				 rctSchema,
				 "CRATE_CONF_DUMMY",
				 "CRATE_CONF_DUMMY.CRATE_CONF",
				 m_omdsReader.basicQuery( *crate,
							  rctSchema,
							  "RCT_CRATE_CONF",
							  "RCT_CRATE_CONF.RCT_KEY",
							  crate_conf
				 ));
       bool extantCard[8];
       int cardNum =0 ;
       for(std::vector<std::string>::iterator card = cardMaskStrings.begin(); card!=cardMaskStrings.end(); ++card){
	 cardConfResults.fillVariable(*card,extantCard[cardNum]);

	 if(!extantCard[cardNum]){
	   switch(cardNum){
	     case 6 :

	       for ( int k = 0 ; k <4 ; k++) {

		 m->ecalMask[crateNum][0][(cardNum/2)*8 + k ] |= !extantCard[cardNum];
		 m->ecalMask[crateNum][1][(cardNum/2)*8 + k ] |= !extantCard[cardNum];
		 m->hcalMask[crateNum][0][(cardNum/2)*8 + k ] |= !extantCard[cardNum];
		 m->hcalMask[crateNum][1][(cardNum/2)*8 + k ] |= !extantCard[cardNum];
	       }
	       break;
	   case 7 :

	     for ( int k = 0 ; k <4 ; k++) {
	       m->hfMask[crateNum][0][k] |= !extantCard[cardNum];
	       m->hfMask[crateNum][1][k] |= !extantCard[cardNum];
	     }
	     break;
	   default:

	     for(int k = 0; k < 8 ; k++ ){
	       m->hcalMask[crateNum][cardNum%2][(cardNum/2)*8 + k] |= !extantCard[cardNum];
	       m->ecalMask[crateNum][cardNum%2][(cardNum/2)*8 + k] |= !extantCard[cardNum];
	     }
	   }		 
	 }
	 cardNum++;
       }
       crateNum++;
     }
     /*
    std::cout << "check fill" <<std::endl;
     for(int i = 0; i< 18; i++)
       for(int j =0; j< 2; j++){
	 for(int k =0; k<28; k++){
	   if(m->ecalMask[i][j][k])
	     std::cout << "ecal masked channel: crate " << i << " phi " << j <<" ieta " <<k <<std::endl; 
	   if(m->hcalMask[i][j][k])
	     std::cout << "hcal masked channel: crate " << i << " phi " << j <<" ieta " <<k <<std::endl; 
	 }
	 for(int k =0; k<4;k++)
	   if(m->hfMask[i][j][k])
	     std::cout << "hf masked channel: crate " << i << " phi " << j <<" ieta " <<k <<std::endl; 
       }
     
     */
     //~~~~~~~~~ Instantiate new L1RCTChannelMask object. ~~~~~~~~~


     return boost::shared_ptr< L1RCTChannelMask >(m);
} 
	

//
// member functions
//




//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RCTChannelMaskOnlineProd);
