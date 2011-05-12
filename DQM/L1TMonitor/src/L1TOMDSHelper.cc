#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

//_____________________________________________________________________
L1TOMDSHelper::L1TOMDSHelper(){
  
  m_omdsReader = 0;

}

//_____________________________________________________________________
L1TOMDSHelper::~L1TOMDSHelper(){

  delete m_omdsReader;

}

//_____________________________________________________________________
bool L1TOMDSHelper::connect(string iOracleDB,string iPathCondDB,string &error){

  // Handeling inputs
  m_oracleDB   = iOracleDB;
  m_pathCondDB = iPathCondDB;
  error        = "";

  // Initializing variables 
  bool SessionExists    = false;
  bool SessionOpen      = false;
  bool ConnectionExists = false;
  bool ConnectionOpen   = false;
  bool out              = false;

  m_omdsReader = new l1t::OMDSReader();
  m_omdsReader->connect(m_oracleDB,m_pathCondDB);

  // Testing session
  if(m_omdsReader->dbSession()){
    SessionExists = true;
    if(m_omdsReader->dbSession()->isOpen()){SessionOpen = true;}
  }
  
  // Testing connection
  if(m_omdsReader->dbConnection()){
    ConnectionExists = true;
    if(m_omdsReader->dbSession()->isOpen()){ConnectionOpen = true;}
  }

  // Defining output and error message if needed
  if     (SessionExists && SessionOpen && ConnectionExists && ConnectionOpen){out = true;}
  else if(!SessionExists || !ConnectionExists) {error = "WARNING: DB connection failed (Session or Connection do not exist)";}
  else if(!SessionOpen)                        {error = "WARNING: DB connection failed (Session is not open)";}
  else if(!ConnectionOpen)                     {error = "WARNING: DB connection failed (Conection is not open)";}

  return out;

}


//_____________________________________________________________________
map<string,WbMTriggerXSecFit> L1TOMDSHelper::getWbMTriggerXsecFits(string iTable,std::string &error){
  
  map<string,WbMTriggerXSecFit>       out;
  const l1t::OMDSReader::QueryResults qresults;
  error = "";

  // Parameters
  string qSchema = "CMS_WBM";

  // Fields to retrive in the query
  vector<string> qStrings;
  qStrings.push_back("BIT");
  qStrings.push_back("NAME");
  qStrings.push_back("PM1");  //Inverse
  qStrings.push_back("P0");   //Constant
  qStrings.push_back("P1");   //Linear
  qStrings.push_back("P2");   //Quadratic

  l1t::OMDSReader::QueryResults paramResults = m_omdsReader->basicQuery(qStrings,qSchema,iTable,"",qresults);

  if(paramResults.queryFailed()){ 
    edm::LogError("L1TOMDSHelper") << "OMDS query error: Query Failed";
    error = "WARNING: OMDS query failed";
  }

  for(int i=0; i<paramResults.numberRows();++i){
    
    WbMTriggerXSecFit tFit;
    tFit.fitFunction = "[0]/x+[1]+[2]*x+[3]*x*x"; // Fitting function hardcoded for now

    string tBitName;

    paramResults.fillVariableFromRow("BIT" ,i,tFit.bitNumber);
    paramResults.fillVariableFromRow("NAME",i,tBitName);
    paramResults.fillVariableFromRow("PM1" ,i,tFit.pm1);
    paramResults.fillVariableFromRow("P0"  ,i,tFit.p0);
    paramResults.fillVariableFromRow("P1"  ,i,tFit.p1);
    paramResults.fillVariableFromRow("P2"  ,i,tFit.p2);

    tFit.bitName = tBitName;

    out[tBitName] = tFit;

  }

  return out;

}

//_____________________________________________________________________
map<string,WbMTriggerXSecFit> L1TOMDSHelper::getWbMAlgoXsecFits(std::string &error){
  return getWbMTriggerXsecFits("LEVEL1_ALGO_CROSS_SECTION",error);
}

//_____________________________________________________________________
map<string,WbMTriggerXSecFit> L1TOMDSHelper::getWbMTechXsecFits(std::string &error){
  return getWbMTriggerXsecFits("LEVEL1_TECH_CROSS_SECTION",error);
}

//_____________________________________________________________________
int L1TOMDSHelper::getNumberCollidingBunches(int lhcFillNumber,string &error){

  int nCollidingBunches = 0;
  error                 = "";

  // Parameters
  string rtlSchema = "CMS_RUNTIME_LOGGER";
  string table     = "FILL_INITLUMIPERBUNCH";
  string atribute1 = "LHCFILL";

  // Fields to retrive in the query
  vector<std::string> qStrings ;
  qStrings.push_back("BUNCH");
  qStrings.push_back("BEAM1CONFIG"); 
  qStrings.push_back("BEAM2CONFIG");  

  l1t::OMDSReader::QueryResults qResults = m_omdsReader->basicQuery(qStrings,rtlSchema,table,atribute1,m_omdsReader->singleAttribute(lhcFillNumber));

  // Check query successful
  if(qResults.queryFailed()){ 
    edm::LogError( "L1TOMDSHelper" ) << "OMDS query error: Query Failed";
    error = "WARNING: OMDS query failed\n";
  }
  else{

    if(qResults.numberRows() != 3564){
      error = "WARNING: Initial bunch luminosity was not correctly retrived from DB\n" ;
    }
    else{

      // Now we count the number of bunches with both beam 1 and 2 configured
      for(int i=0; i<qResults.numberRows();++i){    
        int   bunch;
        bool  beam1config,beam2config;
        qResults.fillVariableFromRow("BUNCH"        ,i,bunch);
        qResults.fillVariableFromRow("BEAM1CONFIG"  ,i,beam1config);
        qResults.fillVariableFromRow("BEAM2CONFIG"  ,i,beam2config);

        if(beam1config && beam2config){nCollidingBunches++;}
      }
    }
  }

  return nCollidingBunches;

}

//_____________________________________________________________________
vector<bool> L1TOMDSHelper::getBunchStructure(int lhcFillNumber,string &error){

  vector<bool> BunchStructure;
  error = "";

  // Fields to retrive in the query
  string rtlSchema = "CMS_RUNTIME_LOGGER";
  string table     = "FILL_INITLUMIPERBUNCH";
  string atribute1 = "LHCFILL";

  // Setting the strings we want to recover from OMDS
  vector<std::string> qStrings ;
  qStrings.push_back("BUNCH");
  qStrings.push_back("BEAM1CONFIG"); 
  qStrings.push_back("BEAM2CONFIG");  

  l1t::OMDSReader::QueryResults qResults = m_omdsReader->basicQuery(qStrings,rtlSchema,table,atribute1,m_omdsReader->singleAttribute(lhcFillNumber));

  if(qResults.queryFailed()){
    edm::LogError( "L1TOMDSHelper" ) << "OMDS query error: Query Failed";
    error = "WARNING: OMDS query failed\n";
  }
  else{

    if(qResults.numberRows() != 3564){
      error = "WARNING: Initial bunch luminosity was not correctly retrived from DB\n" ;
    }
    else{

      for(int i=0; i<qResults.numberRows();++i){    
        int   bunch;
        bool  beam1config,beam2config;
        qResults.fillVariableFromRow("BUNCH"      ,i,bunch);
        qResults.fillVariableFromRow("BEAM1CONFIG",i,beam1config);
        qResults.fillVariableFromRow("BEAM2CONFIG",i,beam2config);

        if(beam1config && beam2config){BunchStructure.push_back(true);}
        else                          {BunchStructure.push_back(false);}

      }
    }
  }

  return BunchStructure;

}

//_____________________________________________________________________
vector<float> L1TOMDSHelper::getInitBunchLumi(int lhcFillNumber,string &error){

  vector<float> InitBunchLumi;
  error = "";

  // Fields to retrive in the query
  string rtlSchema = "CMS_RUNTIME_LOGGER";
  string table     = "FILL_INITLUMIPERBUNCH";
  string atribute1 = "LHCFILL";

  // Setting the strings we want to recover from OMDS
  vector<std::string> qStrings ;
  qStrings.push_back("BUNCH");
  qStrings.push_back("INITBUNCHLUMI");

  l1t::OMDSReader::QueryResults qResults = m_omdsReader->basicQuery(qStrings,rtlSchema,table,atribute1,m_omdsReader->singleAttribute(lhcFillNumber));

  if(qResults.queryFailed()){
    edm::LogError( "L1TOMDSHelper" ) << "OMDS query error: Query Failed";
    error = "WARNING: OMDS query failed\n";
  }
  else{

    if(qResults.numberRows() != 3564){
      error = "WARNING: Initial bunch luminosity was not correctly retrived from DB\n" ;
    }
    else{

      for(int i=0; i<qResults.numberRows();++i){    
        int   bunch;
        float initbunchlumi;
        qResults.fillVariableFromRow("BUNCH"        ,i,bunch);
        qResults.fillVariableFromRow("INITBUNCHLUMI",i,initbunchlumi);

        InitBunchLumi.push_back(initbunchlumi);
      }
    }
  }

  return InitBunchLumi;

}

//_____________________________________________________________________
vector<double> L1TOMDSHelper::getRelativeBunchLumi(int lhcFillNumber,string &error){

  vector<double> RelativeBunchLumi;
  error = "";

  // Fields to retrive in the query
  string rtlSchema = "CMS_RUNTIME_LOGGER";
  string table     = "FILL_INITLUMIPERBUNCH";
  string atribute1 = "LHCFILL";

  // Setting the strings we want to recover from OMDS
  vector<std::string> qStrings ;
  qStrings.push_back("BUNCH");
  qStrings.push_back("INITBUNCHLUMI");

  l1t::OMDSReader::QueryResults qResults = m_omdsReader->basicQuery(qStrings,rtlSchema,table,atribute1,m_omdsReader->singleAttribute(lhcFillNumber));

  if(qResults.queryFailed()){
    edm::LogError( "L1TOMDSHelper" ) << "OMDS query error: Query Failed";
    error = "WARNING: OMDS query failed\n";
  }
  else{

    if(qResults.numberRows() != 3564){
      error = "WARNING: Initial bunch luminosity was not correctly retrived from DB\n" ;
    }
    else{

      //-> Get the inicial bunch luminosity add calculate the total luminosity of the fill
      double        InitTotalLumi = 0;
      vector<float> InitBunchLumi;

      for(int i=0; i<qResults.numberRows();++i){    
        int   bunch;
        float initbunchlumi;
        qResults.fillVariableFromRow("BUNCH"        ,i,bunch);
        qResults.fillVariableFromRow("INITBUNCHLUMI",i,initbunchlumi);

        InitTotalLumi += initbunchlumi;
        InitBunchLumi.push_back(initbunchlumi);
      }

      //-> We calculate the relative luminosity for each bunch
      for(unsigned int i=0 ; i<InitBunchLumi.size() ; i++){
        RelativeBunchLumi.push_back(InitBunchLumi[i]/InitTotalLumi);
      }
    }
  }

  return RelativeBunchLumi;

}
