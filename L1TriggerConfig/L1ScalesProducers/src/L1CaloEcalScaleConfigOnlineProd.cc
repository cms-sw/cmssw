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
// $Id: L1CaloEcalScaleConfigOnlineProd.cc,v 1.4 2010/12/21 04:08:28 efron Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

//
// class declaration
//

class L1CaloEcalScaleConfigOnlineProd :
  public L1ConfigOnlineProdBase< L1CaloEcalScaleRcd, L1CaloEcalScale > {
   public:
      L1CaloEcalScaleConfigOnlineProd(const edm::ParameterSet&);
      ~L1CaloEcalScaleConfigOnlineProd();

  virtual boost::shared_ptr< L1CaloEcalScale > newObject(
    const std::string& objectKey ) ;


   private:
 const EcalElectronicsMapping * theMapping_ ;
  std::map<int, std::vector<int>* > groupInfo;
  EcalTPGGroups*  lutGrpMap;
  L1CaloEcalScale* ecalScale;
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
L1CaloEcalScaleConfigOnlineProd::L1CaloEcalScaleConfigOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1CaloEcalScaleRcd, L1CaloEcalScale >( iConfig )
{
  ecalScale = new L1CaloEcalScale(0);
  theMapping_ = new EcalElectronicsMapping();
   lutGrpMap = new EcalTPGGroups();

}


L1CaloEcalScaleConfigOnlineProd::~L1CaloEcalScaleConfigOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete theMapping_;
 
  //  delete ecalScale;
  //  delete lutGrpMap;
  groupInfo.clear();


}

boost::shared_ptr< L1CaloEcalScale >
L1CaloEcalScaleConfigOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;
 
     std:: cout << "object Key " << objectKey <<std::endl;

     if(objectKey == "NULL" || objectKey == "")  // return default blank ecal scale	 
        return boost::shared_ptr< L1CaloEcalScale >( ecalScale );
     if(objectKey == "IDENTITY"){  // return identity ecal scale  
       ecalScale = 0;
       ecalScale = new L1CaloEcalScale(1);
       return boost::shared_ptr< L1CaloEcalScale >( ecalScale);
     }
     

     double ee_lsb = 0.;
     double eb_lsb = 0.;

     std::vector < std::string > mainStrings;
  
     // ~~~~~~~~~ Cut values ~~~~~~~~~

 
     std::vector< std::string > paramStrings ;
     paramStrings.push_back("LOGIC_ID");  // EB/EE
     paramStrings.push_back("ETSAT");  //Only object needed
    
    
     std::vector< std::string> IDStrings;
     IDStrings.push_back("NAME");
     IDStrings.push_back("ID1");
     IDStrings.push_back("ID2");
     IDStrings.push_back("maps_to");

    l1t::OMDSReader::QueryResults paramResults =
       m_omdsReader.basicQuery( paramStrings,
                                "CMS_ECAL_CONF",
                                "FE_CONFIG_LUTPARAM_DAT",
                                "FE_CONFIG_LUTPARAM_DAT.LUT_CONF_ID",
				m_omdsReader.singleAttribute(objectKey)	
				);

    if( paramResults.queryFailed()
	|| (paramResults.numberRows()==0) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key.  Unable to find lutparam dat table" ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }

    

     for(int i = 0; i < paramResults.numberRows() ; i++){
       
       //EcalTPGPhysicsConst::Item item;
       float etSat;
       paramResults.fillVariableFromRow("ETSAT",i,etSat);

       std::string  ecid_name; 
       int logic_id;
       paramResults.fillVariableFromRow("LOGIC_ID",i, logic_id);
       
       
       l1t::OMDSReader::QueryResults logicID =
	 m_omdsReader.basicQuery(IDStrings,
				 "CMS_ECAL_COND",
				 "CHANNELVIEW",
				 "CHANNELVIEW.LOGIC_ID",
				 m_omdsReader.singleAttribute(logic_id)
				 );
       
       logicID.fillVariable("NAME",ecid_name);

       if(ecid_name =="EB")
	 eb_lsb = etSat/1024;
       else if("EE" == ecid_name)
	 ee_lsb = etSat/1024;
       else {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale  LOGIC_ID.  unable to find channel view with appropiate logic id" ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }

     }
     //     std::cout << " eb lsb " << eb_lsb << " ee_lsb " << ee_lsb << std::endl;
  
     std::vector< std::string > grpLUT;
     grpLUT.push_back("GROUP_ID");
     grpLUT.push_back("LUT_ID");
     grpLUT.push_back("LUT_VALUE");
     
     l1t::OMDSReader::QueryResults lutGrpResults = 
       m_omdsReader.basicQuery( grpLUT,
				"CMS_ECAL_CONF",
				"FE_LUT_PER_GROUP_DAT",
				"FE_LUT_PER_GROUP_DAT.LUT_CONF_ID",
				m_omdsReader.singleAttribute(objectKey)
				);

    if( lutGrpResults.queryFailed()
	|| (lutGrpResults.numberRows()%1024 !=0) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key.  No group info" ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }


     int nEntries = lutGrpResults.numberRows();
     for(int i = 0; i <  nEntries; i++) {
       int group, lutID;
       float lutValue;

       lutGrpResults.fillVariableFromRow("GROUP_ID",i,group);
       if(groupInfo.find(group) == groupInfo.end()){
	 groupInfo[group] = new std::vector<int>;
	 (groupInfo[group])->resize(1024);
       }
      
       lutGrpResults.fillVariableFromRow("LUT_ID",i,lutID);
       lutGrpResults.fillVariableFromRow("LUT_VALUE",i,lutValue);
       groupInfo[group]->at(lutID) = (int) lutValue;
     }

     std::map<int, std::vector<int> >  tpgValueMap;
       
       std::map<int, std::vector<int>* >::iterator grpIt;
     for ( grpIt = groupInfo.begin(); grpIt != groupInfo.end() ; ++grpIt){
       const std::vector<int> * lut_ = grpIt->second;
     
       std::vector<int> tpgValue; 
       tpgValue.resize(256);
       int lastValue = 0;
       for(int tpg = 0; tpg < 256 ; tpg++){

	 for(int i = 0; i < 1024 ; i++) {

	   if(tpg == (0xff & (lut_->at(i)))){
	     tpgValue[tpg] = i; 
	     lastValue = i;
	     break;
	   }
	   tpgValue[tpg] = lastValue;
	 }
       }
       tpgValueMap[grpIt->first] = tpgValue;
     }


     std::vector < std::string > groupMap;
     groupMap.push_back("LOGIC_ID");
     groupMap.push_back("GROUP_ID");


     
     l1t::OMDSReader::QueryResults grpMapResults = 
       m_omdsReader.basicQuery( groupMap,
				"CMS_ECAL_CONF",
				"FE_CONFIG_LUT_DAT",
				"FE_CONFIG_LUT_DAT.LUT_CONF_ID",
				m_omdsReader.singleAttribute(objectKey)
				);
     if( grpMapResults.queryFailed()
	|| (grpMapResults.numberRows()==0) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key. No fe_config_lut_dat info" ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }

     nEntries = grpMapResults.numberRows();
     for(int i = 0; i< nEntries; ++i){
       std::string  ecid_name; 
       int logic_id;
       grpMapResults.fillVariableFromRow("LOGIC_ID",i, logic_id);
       int group_id;
       grpMapResults.fillVariableFromRow("GROUP_ID",i, group_id);
       //       if(logic_id >= 2100001901 && logic_id <= 2100001916)
	 //	 std::cout<< "missing logic id found " <<logic_id <<std::endl;
       l1t::OMDSReader::QueryResults IDResults =
	 m_omdsReader.basicQuery( IDStrings,
				  "CMS_ECAL_COND",
				  "CHANNELVIEW",
				  "CHANNELVIEW.LOGIC_ID",
				  m_omdsReader.singleAttribute(logic_id)
				  );
       if( paramResults.queryFailed()
	   || (paramResults.numberRows()==0) ) // check query successful
	 {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key.  Unable to find logic_id channel view" ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }
       for(int j = 0; j < IDResults.numberRows(); j++){

	 std::string ecid_name, maps_to;

	 IDResults.fillVariableFromRow("NAME",j, ecid_name);
	 IDResults.fillVariableFromRow("maps_to",j, maps_to);
	 if(logic_id >= 2100001901 && logic_id <= 2100001916)
	   //	   std::cout << " name " << ecid_name << " maps to " << maps_to <<std::endl;
	 if(ecid_name != maps_to){
	   continue;               // make sure they match
	 }
	 if(ecid_name== "EB_trigger_tower" || ecid_name == "EE_trigger_tower") {	   
	   int id1,id2;
	   IDResults.fillVariableFromRow("ID1",j, id1);
	   IDResults.fillVariableFromRow("ID2",j, id2);
	 
	   if(ecid_name == "EB_trigger_tower")
	     id1+=36;  //lowest TCC for barrel 37
	   EcalTrigTowerDetId temp = theMapping_->getTrigTowerDetId(id1,id2);
	   /*	   if(ecid_name == "EE_trigger_tower"){
	     int testID = theMapping_->TCCid(temp);

	     if( testID != id1 ){
	      	       std::cout << " unmatched mapping testID " <<testID <<std::endl;
	       std::cout << "id1 " << id1 << " id2 " <<id2<< " iphi " << temp.iphi() <<"  ieta " << temp.ieta() <<std::endl;
	     }
	   }
	   if(ecid_name == "EB_trigger_tower"){
	     int testID = theMapping_->TCCid(temp);
	     if( testID != id1 ){
	       std::cout << " unmatched mapping testID " <<testID <<std::endl;
	       std::cout << "id1 " << id1 << " id2 " <<id2<< " iphi " << temp.iphi() <<"  ieta " << temp.ieta() <<std::endl;
	     }
	   }
	   */
	   //	   if(temp.ieta() == -18 || temp.ietaAbs() == 28)
	     //	   if(logic_id >= 2100001901 && logic_id <= 2100001916)


	   lutGrpMap->setValue(temp, group_id);  // assume ee has less than 68 tt
	   break;
	 }
       }
     }
     
     const EcalTPGGroups::EcalTPGGroupsMap & gMap = lutGrpMap->getMap();
     


     for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta ){
       EcalSubdetector subdet = ( ieta <= 17  ) ? EcalBarrel : EcalEndcap ;
       double et_lsb = (ieta<=17) ?  eb_lsb : ee_lsb;
       for(int pos = 0; pos <=1; pos++){
	 int zside = (int)  pow(-1,pos);

	 //	 std::cout << "ieta " <<zside*ieta ;
	 for(int iphi = 1; iphi<=72; iphi++){
	   if(!EcalTrigTowerDetId::validDetId(zside,subdet,ieta, iphi))
	     continue;
	   EcalTrigTowerDetId test(zside, subdet, ieta, iphi);
	   EcalTPGGroups::EcalTPGGroupsMapItr  itLut = gMap.find(test) ;
	   if(itLut != gMap.end()) {
	     //	     std::cout << " non mapped section iphi " << iphi << " ieta " <<ieta << " tccid " << theMapping_->TCCid(test) << " iTT " << theMapping_->iTT(test)<< std::endl;
	     std::vector<int> tpgValue = tpgValueMap[itLut->second]; 

	     for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	       {
		 ecalScale->setBin(irank, ieta, zside, et_lsb * tpgValue[irank]);

		 //		 std::cout << " irank " << irank << " etValue " << et_lsb*tpgValue[irank] << std::endl;		 
	       }
	     
	     break;
	   }
	 }
       }
     }

     
     //     ecalScale->print(std::cout);

 
// ------------ method called to produce the data  ------------
     return boost::shared_ptr< L1CaloEcalScale >( ecalScale );
}
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloEcalScaleConfigOnlineProd);
