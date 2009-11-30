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
// $Id: L1CaloHcalScaleConfigOnlineProd.cc,v 1.3 2009/10/25 19:30:36 efron Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
//#include "CondFormats/HcalObjects/interface/HcalTPGLutIdMap.h"
//#include "CondFormats/HcalObjects/interface/HcalTPGLutGroup.h"
//#include "CondFormats/HcalObjects/interface/HcalTPGPhysicsConst.h"
//#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
//#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
//#include "Geometry/HcalMapping/interface/HcalElectronicsMapping.h"
//#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CondTools/L1Trigger/interface/DataManager.h"

#include <iostream>
#include <iomanip>

//
// class declaration
//

class L1CaloHcalScaleConfigOnlineProd :
  public L1ConfigOnlineProdBase< L1CaloHcalScaleRcd, L1CaloHcalScale > {
   public:
      L1CaloHcalScaleConfigOnlineProd(const edm::ParameterSet& iConfig);
      ~L1CaloHcalScaleConfigOnlineProd();

  virtual boost::shared_ptr< L1CaloHcalScale > newObject(
    const std::string& objectKey ) ;


   private:
  // const HcalElectronicsMapping * theMapping_ ;
  //  std::map<int, std::vector<int>* > groupInfo;
  // HcalTPGGroups*  lutGrpMap;
  L1CaloHcalScale* hcalScale;
  HcalTrigTowerGeometry theTrigTowerGeometry;
  CaloTPGTranscoderULUT caloTPG;
  typedef std::vector<double> RCTdecompression;
  std::vector<RCTdecompression> hcaluncomp;
  cond::CoralTransaction* m_coralTransaction ;
  //  DataManager a;
 cond::DBSession * session;
 cond::Connection * connection ;

  
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
L1CaloHcalScaleConfigOnlineProd::L1CaloHcalScaleConfigOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1CaloHcalScaleRcd, L1CaloHcalScale >( iConfig )
{
  hcalScale = new L1CaloHcalScale(0);
  //  theMapping_ = new HcalElectronicsMapping();
  //   lutGrpMap = new HcalTPGGroups();



 session = new cond::DBSession();
 session->configuration().setMessageLevel( cond::Debug ) ;

 
  std::string  connectString =    iConfig.getParameter< std::string >( "onlineDB" );
  std::string authenticationPath  =   iConfig.getParameter< std::string >( "onlineAuthentication" );
  

  session->configuration().setAuthenticationMethod(cond::XML);
  std::cout << "authenticating  crap " << std::endl;
  session->configuration().setAuthenticationPath( authenticationPath ) ;
  std::cout << "initializing pthing  crap " << std::endl << std::flush;
  session->configuration().setBlobStreamer("COND/Services/TBufferBlobStreamingService") ;
  std::cout << "initializing blobbing crap " << std::endl <<std::flush;
  session->open() ;
  std::cout << "initializing opening crap " << std::endl << std::flush;

 connection = new cond::Connection( connectString ) ;
     connection->connect( session ) ;


     m_coralTransaction = &( connection->coralTransaction() ) ;
     m_coralTransaction->start( true ) ;

   
     //        DataM a ( connectString, authenticationPath, true );

}


L1CaloHcalScaleConfigOnlineProd::~L1CaloHcalScaleConfigOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  //  delete theMapping_;
 
  connection->disconnect() ;
  //  delete connection ;
  //   delete session ;

    delete hcalScale;
  //  delete lutGrpMap;
  //  groupInfo.clear();


}

boost::shared_ptr< L1CaloHcalScale >
L1CaloHcalScaleConfigOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;
 
     std:: cout << "object Key " << objectKey <<std::endl;

     if(objectKey == "NULL" || objectKey == "")  // return default blank ecal scale	 
        return boost::shared_ptr< L1CaloHcalScale >( hcalScale );
     if(objectKey == "IDENTITY"){  // return identity ecal scale  
       hcalScale = 0;
       hcalScale = new L1CaloHcalScale(1);
       return boost::shared_ptr< L1CaloHcalScale >( hcalScale);
     }
     
  // TODO cms::Error log
     //   if (1024 != (unsigned int) 0x400) std::cout << "Error: Analytic compression expects 10-bit LUT; found LUT with " << 1024 << " entries instead" << std::endl;
 
     std::vector<unsigned int> analyticalLUT(1024, 0);
     std::vector<unsigned int> identityLUT(1024, 0);
     
     // Compute compression LUT
     for (unsigned int i=0; i < 1024; i++) {
       analyticalLUT[i] = (unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5);
       identityLUT[i] = std::min(i,0xffu);
       //        std::cout << "output lsb " <<std::endl;
       //	 for (unsigned int k = threshold; k < 1024; ++k)
       //      std::cout << i << " "<<analyticalLUT[i] << " ";
     }
     
     hcaluncomp.clear();
    for (int i = 0; i < 4176; i++){
       RCTdecompression decompressionTable(0x100,0);
       hcaluncomp.push_back(decompressionTable);
    }



     std::vector < std::string > mainStrings;
     mainStrings.push_back("HCAL_LUT_METADATA");
     mainStrings.push_back("HCAL_LUT_CHAN_DATA");

     // ~~~~~~~~~ Cut values ~~~~~~~~~

 
     std::vector< std::string > metaStrings ;
     metaStrings.push_back("RCTLSB");  
     metaStrings.push_back("NOMINAL_GAIN");  
   
     
   

 
    l1t::OMDSReader::QueryResults paramResults =
       m_omdsReader.basicQueryView( metaStrings,
                                "CMS_HCL_HCAL_COND",
                                "V_HCAL_LUT_METADATA_V1",
                                "V_HCAL_LUT_METADATA_V1.TAG_NAME",
				m_omdsReader.basicQuery(
							"HCAL_LUT_METADATA",
							"CMS_RCT",
							"HCAL_SCALE_KEY",
							"HCAL_SCALE_KEY.HCAL_TAG",
							m_omdsReader.singleAttribute(objectKey)));
    

    
    
    if( paramResults.queryFailed()
	|| (paramResults.numberRows()!=1) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloHcalScale key.  Unable to find lutparam dat table" ;
	 return boost::shared_ptr< L1CaloHcalScale >() ;
       }

    

    //     for(int i = 0; i < paramResults.numberRows() ; i++){
       
    //HcalTPGPhysicsConst::Item item;
    double hcalLSB, nominal_gain;
    paramResults.fillVariable("RCTLSB",hcalLSB);    
    paramResults.fillVariable("NOMINAL_GAIN",nominal_gain);
 
    float    rctlsb = hcalLSB == 0.25 ? 1./4 : 1./8;




    std::cout << " god fucking mother fucker lsb "   << rctlsb << " ng " << nominal_gain <<std::endl;

    l1t::OMDSReader::QueryResults chanKey =m_omdsReader.basicQuery(
								  "HCAL_LUT_CHAN_DATA",
								  "CMS_RCT",
								  "HCAL_SCALE_KEY",
								  "HCAL_SCALE_KEY.HCAL_TAG",
								  m_omdsReader.singleAttribute(objectKey));
      
    //coral::AttributeList myresult;
    //    myresult.extend(
    
    /*
    l1t::OMDSReader::QueryResults chanResults 

					 
       m_omdsReader.basicQueryView( channelStrings,
                                "CMS_HCL_HCAL_COND",
                                "V_HCAL_LUT_CHAN_DATA_V1",
                                "V_HCAL_LUT_CHAN_DATA_V1.TAG_NAME",
				m_omdsReader.basicQuery(
							"HCAL_LUT_CHAN_DATA",
							"CMS_RCT",
							"HCAL_SCALE_KEY",
							"HCAL_SCALE_KEY.HCAL_TAG",
							m_omdsReader.singleAttribute(objectKey)));
    
    */

    std::string schemaName("CMS_HCL_HCAL_COND");
    coral::ISchema& schema = schemaName.empty() ?
      m_coralTransaction->nominalSchema() :
      m_coralTransaction->coralSessionProxy().schema( schemaName ) ;
     coral::IQuery* query = schema.newQuery(); ;



     
    std::vector< std::string > channelStrings;
    channelStrings.push_back("IPHI");
    channelStrings.push_back("IETA");
    channelStrings.push_back("LUT_GRANULARITY");
    channelStrings.push_back("OUTPUT_LUT_THRESHOLD");
    channelStrings.push_back("OBJECTNAME");

     std::vector< std::string >::const_iterator it = channelStrings.begin() ;
     std::vector< std::string >::const_iterator end = channelStrings.end() ;
     for( ; it != end ; ++it )
       {
         query->addToOutputList( *it ) ;
       }

    std::string ob = "OBJECTNAME";
    coral::AttributeList myresult; 
    myresult.extend("IPHI", typeid(int)); 
    myresult.extend("IETA", typeid(int)); 
    myresult.extend("LUT_GRANULARITY", typeid(int)); 
    myresult.extend("OUTPUT_LUT_THRESHOLD", typeid(int)); 
    myresult.extend( ob,typeid(std::string));//, typeid(std::string)); 
    //   query->addToOutputList(*constIt);
    query->defineOutput( myresult ); 

    //  query->addToOutputList(ob);

    query->addToTableList( "V_HCAL_LUT_CHAN_DATA_V1");


    query->setCondition(
			"V_HCAL_LUT_CHAN_DATA_V1.TAG_NAME = :" + chanKey.columnNames().front(),
			chanKey.attributeLists().front());


    coral::ICursor& cursor = query->execute();

 // when the query goes out of scope.
    std::vector<coral::AttributeList> atts;
    while (cursor.next()) {
        atts.push_back(cursor.currentRow());
    };

    delete query;

    l1t::OMDSReader::QueryResults chanResults(channelStrings,atts); 
    if( chanResults.queryFailed()
	|| (chanResults.numberRows()==0) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloHcalScale key.  Unable to find lutparam dat table nrows" << chanResults.numberRows() ;
	 return boost::shared_ptr< L1CaloHcalScale >() ;
       }
    std::cout << " god fucking mother fucker 2" << std::endl;


    chanResults.attributeLists();
     for(int i = 0; i < chanResults.numberRows() ; ++i){
       std::string objectName;
       chanResults.fillVariableFromRow("OBJECTNAME",i, objectName);
       //       int
       if(objectName == "HcalTrigTowerDetId") { //trig tower
	 int ieta, iphi, lutGranularity, threshold;
	 
	 
	 chanResults.fillVariableFromRow("LUT_GRANULARITY",i,lutGranularity);
	 chanResults.fillVariableFromRow("IPHI",i,iphi);
	 chanResults.fillVariableFromRow("IETA",i,ieta);
	 chanResults.fillVariableFromRow("OUTPUT_LUT_THRESHOLD",i,threshold);
	 
	 if(ieta == 1 && iphi == 1)
	   std::cout << "11 data  gran " << lutGranularity << " thresh " << threshold <<std::endl;

	 if(ieta == 29 && iphi > 0 && iphi <= 4)
	   std::cout << "hf data  gran " << lutGranularity << " thresh " << threshold <<std::endl;

	 unsigned int outputLut[1024];
	 int lutId = caloTPG.GetOutputLUTId(ieta,iphi);



	 double eta_low = 0., eta_high = 0.;
	 theTrigTowerGeometry.towerEtaBounds(ieta,eta_low,eta_high); 
	 double cosh_ieta = fabs(cosh((eta_low + eta_high)/2.));

	  //	  for (int iphi = 1; iphi <= 72; iphi++) {
	 if (!caloTPG.HTvalid(ieta, iphi)) continue;
	 double factor = 0.;
	 if (abs(ieta) >= theTrigTowerGeometry.firstHFTower())
	   factor = hcalLSB;
	 else 
	   factor = nominal_gain / cosh_ieta * lutGranularity;
	 for (int k = 0; k < threshold; ++k)
	   outputLut[k] = 0;
	 
         for (unsigned int k = threshold; k < 1024; ++k)
	   outputLut[k] = (abs(ieta) < theTrigTowerGeometry.firstHFTower()) ? analyticalLUT[k] : identityLUT[k];
	 /*
	 if((ieta == 1 && iphi ==1) || (ieta == 29 && iphi > 0 && iphi <= 4) ){
	   std::cout << "output lsb " <<std::endl;
	   for (unsigned int k = 0; k < 1024; ++k)
	     std::cout << k << " "<<outputLut[k]<< " ";
	   
	   std::cout <<std::endl;
	 }
	 */
	   // tpg - compressed value
	   unsigned int tpg = outputLut[0];
          
	   int low = 0;
	  if(ieta == 1 && iphi ==1)
	    std::cout << "11 data  gran low " << low << " tpg " << tpg <<std::endl;
	  
	  if(ieta == 29 && iphi > 0 && iphi <= 4)
	    std::cout << "hf data  gran "<< low << " tpg " << tpg <<std::endl;
          for (unsigned int k = 0; k < 1024; ++k){
             if (outputLut[k] != tpg){
                unsigned int mid = (low + k)/2;
                hcaluncomp[lutId][tpg] = (tpg == 0 ? low : factor * mid);
                low = k;
		if((ieta == 1 && iphi ==1) || (ieta == 29 && iphi > 0 && iphi <= 4) )
	   
		  std::cout << tpg << " " << hcaluncomp[lutId][tpg] << " " ;
                tpg = outputLut[k];


             }
          }
          hcaluncomp[lutId][tpg] = factor * low;
	if((ieta == 1 && iphi ==1) || (ieta == 29 && iphi ==1))
	  std::cout << std::endl;
       }
     }
     


     for( unsigned short ieta = 1 ; ieta <= L1CaloHcalScale::nBinEta; ++ieta ){
       for(int pos = 0; pos <=1; pos++){
	 for( unsigned short irank = 0 ; irank < L1CaloHcalScale::nBinRank; ++irank ){
     
	   
	   
	   int zside = (int)  pow(-1,pos);
	   int nphi = 0;
	   double etvalue = 0.;
	   
	   //	 std::cout << "ieta " <<zside*ieta ;
	   for(int iphi = 1; iphi<=72; iphi++){

	     if(!caloTPG.HTvalid(ieta, iphi))
	       continue;
	     int lutId = caloTPG.GetOutputLUTId(ieta,iphi);
	     nphi++;
	     etvalue += hcaluncomp[lutId][irank];

	   } // phi
	   if (nphi > 0) etvalue /= nphi;

	   
	   hcalScale->setBin(irank, ieta, zside, etvalue);

		   //		 std::cout << " irank " << irank << " etValue " << et_lsb*tpgValue[irank] << std::endl;		 

	 } // rank
       } // zside
     }// eta

     std::cout << std::setprecision(10);
     hcalScale->print(std::cout);
// ------------ method called to produce the data  ------------
     return boost::shared_ptr< L1CaloHcalScale >( hcalScale );

}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloHcalScaleConfigOnlineProd);
