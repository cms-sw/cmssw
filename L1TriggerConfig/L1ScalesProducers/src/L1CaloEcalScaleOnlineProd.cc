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
// $Id: L1CaloEcalScaleOnlineProd.cc,v 1.1 2009/03/18 11:03:04 efron Exp $
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


//
// class declaration
//

class L1CaloEcalScaleOnlineProd :
  public L1ConfigOnlineProdBase< L1CaloEcalScaleRcd, L1CaloEcalScale > {
   public:
      L1CaloEcalScaleOnlineProd(const edm::ParameterSet&);
      ~L1CaloEcalScaleOnlineProd();

  virtual boost::shared_ptr< L1CaloEcalScale > newObject(
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
L1CaloEcalScaleOnlineProd::L1CaloEcalScaleOnlineProd(
  const edm::ParameterSet& iConfig)
  : L1ConfigOnlineProdBase< L1CaloEcalScaleRcd, L1CaloEcalScale >( iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced

   //now do what ever other initialization is needed
}


L1CaloEcalScaleOnlineProd::~L1CaloEcalScaleOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr< L1CaloEcalScale >
L1CaloEcalScaleOnlineProd::newObject( const std::string& objectKey )
{
     using namespace edm::es;


     //     const EcalTPGPhysicsConst & physMap = new EcalTPGPhysicsConst;
     double ee_lsb = 0.;
     double eb_lsb = 0.;


     //     `WILL assume to inputs for now
     // LIN_CONF_ID and LUT_CONF_ID
     //  HOw I extract these will come later.

     std::vector < std::string > mainStrings;
     mainStrings.push_back("LUT_CONFIG_ID");
     mainStrings.push_back("LIN_CONFIG_ID");
     
     l1t::OMDSReader::QueryResults mainResults =
       m_omdsReader.basicQuery(mainStrings,
			       "CMS_ECAL_CONF_TEST",
			       "FE_CONFIG_MAIN",
			       "FE_CONFIG_MAIN.CONF_ID",
			       m_omdsReader.singleAttribute(objectKey)
			       );
     if( mainResults.queryFailed() ||
	 mainResults.numberRows() != 1 ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key." ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }
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
                                "FE_CONFIG_PARAM_DAT",
                                "FE_CONFIG_PARAM_DAT.LIN_CONF_ID",
				mainResults,
				"LIN_CONF_ID"
				);
     

     for(int i = 0; i < paramResults.numberRows() ; i++){
       
       //EcalTPGPhysicsConst::Item item;
       double etSat;
       paramResults.fillVariableFromRow("ETSTAT",i,etSat);
 
       std::string logic_id, ecid_name; 
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
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale  LOGIC_ID." ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }

     }
     /*
     l1t::OMDSReader::QueryResults grpNumber =
       m_omdsReader.basicQuery( "NUMBER_OF_GROUPS",
				"CMS_ECAL_CONF",
				"FE_CONFIG_LUT_INFO",
				"FE_CONFIG_LUT_INFO.LUT_CONF_ID",
				mainResults,
				"LUT_CONF_ID");  

     int nGrps;
     
     grpNumber.fillVariable(nGrps);
     */
     //     EcalTPGLutGroup::EcalTPGLutGroup & lut = new EcalTPGLutGroup();

     std::vector< std::string > grpLUT;
     grpLUT.push_back("GROUP_ID");
     grpLUT.push_back("LUT_ID");
     grpLUT.push_back("LUT_VALUE");
     
     l1t::OMDSReader::QueryResults lutGrpResults = 
       m_omdsReader.basicQuery( grpLUT,
				"CMS_ECAL_CONF",
				"FE_LUT_PER_GROUP_DAT",
				"FE_LUT_PER_GROUP_DAT.LUT_CONF_ID",
				mainResults,
				"LUT_CONF_ID");

    if( lutGrpResults.queryFailed()
	|| (lutGrpResults.numberRows()%1024 !=0) ) // check query successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem with L1CaloEcalScale key." ;
	 return boost::shared_ptr< L1CaloEcalScale >() ;
       }
    std::map<int, std::vector<int>* > groupInfo;

     int nEntries = lutGrpResults.numberRows();
     for(int i = 0; i <  nEntries; i++) {
       int group, lutID;
       int lutValue;
       lutGrpResults.fillVariableFromRow("GROUP_ID",i,group);
       if(groupInfo.find(group) == groupInfo.end()){
	 groupInfo[group] = new std::vector<int>;
	 (groupInfo[group])->resize(1024);
       }
       lutGrpResults.fillVariableFromRow("LUT_ID",i,lutID);
       lutGrpResults.fillVariableFromRow("LUT_VALUE",i,lutValue);
       groupInfo[group]->at(lutID) = lutValue;
     }
     
     std::map<int, std::vector<int> >  tpgValueMap;
     std::map<int, std::vector<int>* >::iterator grpIt;
     for ( grpIt = groupInfo.begin(); grpIt != groupInfo.end() ; ++grpIt){
       const std::vector<int> * lut_ = grpIt->second;
       std::vector<int> tpgValue;
       for(int tpg = 0; tpg < 256 ; tpg++){
	 for(int i = 0; i < 1024 ; i++) 
	   if(tpg == (0xff & lut_->at(i))){
	     tpgValue.push_back(i);
	     break;
	   }
       tpgValueMap[grpIt->first] = tpgValue;
     }

     std::vector < std::string > groupMap;
     groupMap.push_back("LOGIC_ID");
     groupMap.push_back("GROUP_ID");

     EcalTPGGroups*  lutGrpMap = new EcalTPGGroups(); 
     l1t::OMDSReader::QueryResults grpMapResults = 
       m_omdsReader.basicQuery( groupMap,
				"CMS_ECAL_CONF",
				"FE_CONFIG_LUT_DAT",
				"FE_CONFIG_LUT_DAT.LUT_CONF_ID",
				mainResults,
				"LUT_CONF_ID");


     nEntries = grpMapResults.numberRows();
     for(int i = 0; i< nEntries; ++i){
       std::string logic_id, ecid_name; 
       grpMapResults.fillVariableFromRow("LOGIC_ID",i, logic_id);
       int group_id;
       grpMapResults.fillVariableFromRow("GROUP_ID",i, group_id);

       // Logic ID
       //      fillVariableFromRow("LOGIC_ID", i, ecid_xt);
       l1t::OMDSReader::QueryResults IDResults =
	 m_omdsReader.basicQuery( IDStrings,
				  "CMS_ECAL_COND",
				  "CHANNELVIEW",
				  "CHANNELVIEW.LOGIC_ID",
				  m_omdsReader.singleAttribute(logic_id)
				  );
       for(int j = 0; j < IDResults.numberRows(); j++){
	 std::string ecid_name, maps_to;

	 IDResults.fillVariableFromRow("NAME",j, ecid_name);
	 IDResults.fillVariableFromRow("maps_to",j, maps_to);
	 if(ecid_name != maps_to)
	   continue;               // make sure they match
	 else if(ecid_name== "EB_trigger_tower" || ecid_name == "EE_trigger_tower") {
	   int id1,id2;
	   IDResults.fillVariableFromRow("ID1",i, id1);
	   IDResults.fillVariableFromRow("ID2",i, id2);
	   char ch[10];
	   sprintf(ch, "%d%d",id1,id2);  
	   std::string s = "";
	   s.insert(0,ch);
	   
	   int towerID = atoi(s.c_str());
	   lutGrpMap->setValue(towerID, group_id);
	   break;
	 }
       }
     }



     
     const EcalTPGGroups::EcalTPGGroupsMap & gMap = lutGrpMap->getMap();
   
     L1CaloEcalScale* ecalScale = new L1CaloEcalScale();

     for( unsigned short ieta = 1 ; ieta <= L1CaloEcalScale::nBinEta; ++ieta )
       for( unsigned short irank = 0 ; irank < 2 * L1CaloEcalScale::nBinRank; ++irank )
	 {
	   EcalSubdetector subdet = ( ieta <= 17  ) ? EcalBarrel : EcalEndcap ;

	   std::vector<int> tpgValuePos =  tpgValueMap[gMap.find(EcalTrigTowerDetId(1, subdet, ieta, 1).rawId())->second];
	   std::vector<int> tpgValueNeg =  tpgValueMap[gMap.find(EcalTrigTowerDetId(-1, subdet, ieta, 2).rawId())->second];
	   double et_lsb = (ieta<=17) ?  eb_lsb : ee_lsb;
	   for( unsigned short irank = 0 ; irank < L1CaloEcalScale::nBinRank; ++irank )
	     {
	       ecalScale->setBin(irank, ieta, +1, et_lsb * tpgValuePos[irank]);
	       ecalScale->setBin(irank, ieta, -1, et_lsb * tpgValueNeg[irank]);

	     }
	 }
     return boost::shared_ptr< L1CaloEcalScale >( ecalScale );

}
 
// ------------ method called to produce the data  ------------


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloEcalScaleOnlineProd);
