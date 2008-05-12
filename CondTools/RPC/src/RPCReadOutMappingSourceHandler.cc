#include "CondTools/RPC/interface/RPCReadOutMappingSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::RPCReadOutMappingSourceHandler::RPCReadOutMappingSourceHandler(const edm::ParameterSet& ps):
  m_name(ps.getUntrackedParameter<std::string>("name","RPCReadOutMappingSourceHandler")),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_host(ps.getUntrackedParameter<std::string>("OnlineDBHost","lxplus.cern.ch")),
  m_sid(ps.getUntrackedParameter<std::string>("OnlineDBSID","blah")),
  m_user(ps.getUntrackedParameter<std::string>("OnlineDBUser","blaah")),
  m_pass(ps.getUntrackedParameter<std::string>("OnlineDBPass","blaaah")),
  m_port(ps.getUntrackedParameter<int>("OnlineDBPort",1521))
{
}

popcon::RPCReadOutMappingSourceHandler::~RPCReadOutMappingSourceHandler()
{
}

void popcon::RPCReadOutMappingSourceHandler::getNewObjects()
{

//	std::cout << "RPCReadOutMappingSourceHandler: RPCReadOutMappingSourceHandler::getNewObjects begins\n";

        edm::Service<cond::service::PoolDBOutputService> mydbservice;

// first check what is already there in offline DB
	const RPCReadOutMapping* cabling_prev;
        if(m_validate==1) {
//          std::cout<<" Validation was requested, so will check present contents"<<std::endl;
          std::cout<<" Sorry, validation not available for the moment..."<<std::endl;
        }

// now construct new cabling map from online DB
        ConnectOnlineDB(m_host,m_sid,m_user,m_pass,m_port);
        readCablingMap();
        DisconnectOnlineDB();

        cond::Time_t snc=mydbservice->currentTime();

// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2Cablings(cabling_prev,cabling);
        if (!difference) cout<<"No changes - will not write anything!!!"<<endl;
        if (difference==1) {
          cout<<"Will write new object to offline DB!!!"<<endl;
          m_to_transfer.push_back(std::make_pair((RPCReadOutMapping*)cabling,snc));
        }

	std::cout << "RPCReadOutMappingSourceHandler: RPCReadOutMappingSourceHandler::getNewObjects ends\n";
}

void popcon::RPCReadOutMappingSourceHandler::ConnectOnlineDB(string host, string sid, string user, string pass, int port=1521)
{
  stringstream ss;
  ss << "//" << host << ":" << port << "/" << sid;

  cout << "RPCReadOutMappingSourceHandler: connecting to " << host << "..." << flush;
  env = Environment::createEnvironment(Environment::OBJECT);
  conn = env->createConnection(user, pass, ss.str());
  cout << "Done." << endl;
}

void popcon::RPCReadOutMappingSourceHandler::DisconnectOnlineDB()
{
  env->terminateConnection(conn);
  Environment::terminateEnvironment(env);
}

void popcon::RPCReadOutMappingSourceHandler::readCablingMap()
{

//  string cabling_version = "test";
  time_t rawtime;
  time(&rawtime); //time since January 1, 1970
  tm * ptm = gmtime(&rawtime);//GMT time
  char buffer[20];
  strftime(buffer,20,"%d/%m/%Y_%H:%M:%S",ptm);
  string cabling_version=(string)buffer;

  Statement* stmt = conn->createStatement();
  string sqlQuery ="";

  cout << endl <<"RPCReadOutMappingSourceHandler: start to build RPC cabling..." << flush << endl << endl;
  cabling =  new RPCReadOutMapping(cabling_version);

  // Get FEDs
  sqlQuery=" SELECT DCCBoardId, FEDNumber FROM DCCBoard WHERE DCCBoardId>0 ORDER BY FEDNumber ";
  stmt->setSQL(sqlQuery.c_str());
  ResultSet* rset = stmt->executeQuery();
  std::pair<int,int> tmp_tbl;
  std::vector< std::pair<int,int> > theDAQ;
  while (rset->next()) {
    tmp_tbl.first=rset->getInt(1);
    tmp_tbl.second=rset->getInt(2);
    theDAQ.push_back(tmp_tbl);
  }
  for(unsigned int iFED=0;iFED<theDAQ.size();iFED++) {
    std::vector<std::pair<int,int> > theTB;
    DccSpec dcc(theDAQ[iFED].second);
    sqlQuery = " SELECT TriggerBoardId, DCCInputChannelNum FROM TriggerBoard ";
    sqlQuery += " WHERE DCCBoard_DCCBoardId= ";
    sqlQuery += IntToString(theDAQ[iFED].first);
    sqlQuery += " ORDER BY DCCInputChannelNum ";
    stmt->setSQL(sqlQuery.c_str());
    rset = stmt->executeQuery();
    while (rset->next()) {
      tmp_tbl.first=rset->getInt(1);
      tmp_tbl.second=rset->getInt(2);
      theTB.push_back(tmp_tbl);
    }
    for(unsigned int iTB=0;iTB<theTB.size();iTB++) {
      std::vector<std::pair<int,int> > theLink;
      TriggerBoardSpec tb(theTB[iTB].second);
      sqlQuery = " SELECT Board_BoardId, TriggerBoardInputNum FROM LinkConn ";
      sqlQuery += " WHERE TB_TriggerBoardId= ";
      sqlQuery +=  IntToString(theTB[iTB].first);
      sqlQuery += " ORDER BY TriggerBoardInputNum ";
      stmt->setSQL(sqlQuery.c_str());
      rset = stmt->executeQuery();
      while (rset->next()) {
        tmp_tbl.first=rset->getInt(1);
        tmp_tbl.second=rset->getInt(2);
        theLink.push_back(tmp_tbl);
      }
      for(unsigned int iLink=0;iLink<theLink.size();iLink++) {
        std::vector<std::pair<int,string> > theLB;
        std::pair<int,string> tmpLB;
        // Get master first...
        sqlQuery = " SELECT Name ";
        sqlQuery += " FROM Board ";
        sqlQuery += " WHERE BoardId= ";
        sqlQuery +=  IntToString(theLink[iLink].first);
        stmt->setSQL(sqlQuery.c_str());
        rset = stmt->executeQuery();
        while (rset->next()) {
          tmpLB.first=theLink[iLink].first;
          tmpLB.second=rset->getString(1);
          theLB.push_back(tmpLB);
        }
        // then slaves
        sqlQuery = " SELECT LinkBoard.LinkBoardId, Board.Name ";
        sqlQuery += " FROM LinkBoard, Board ";
        sqlQuery += " WHERE LinkBoard.MasterId= ";
        sqlQuery +=  IntToString(theLink[iLink].first);
        sqlQuery += " AND Board.BoardId=LinkBoard.LinkBoardId";
        sqlQuery += " AND LinkBoard.MasterId<>LinkBoard.LinkBoardId";
        sqlQuery += " ORDER BY LinkBoard.LinkBoardId ";
        stmt->setSQL(sqlQuery.c_str());
        rset = stmt->executeQuery();
        while (rset->next()) {
          tmpLB.first=rset->getInt(1);
          tmpLB.second=rset->getString(2);
          theLB.push_back(tmpLB);
        }
        LinkConnSpec  lc(theLink[iLink].second);
        int linkChannel;
        for(unsigned int iLB=0; iLB<theLB.size(); iLB++) {
          linkChannel=atoi(((theLB[iLB].second).substr((theLB[iLB].second).length()-1,1)).c_str());
          bool master = (theLB[iLB].first==theLink[iLink].first);
          std::string name=theLB[iLB].second;
          LinkBoardSpec lb(master,linkChannel,name);
          FEBStruct tmpFEB;
          std::vector<FEBStruct> theFEB;
          sqlQuery = " SELECT FEBLocation.FEBLocationId,";
          sqlQuery += "  FEBLocation.CL_ChamberLocationId,";
          sqlQuery += "  FEBConnector.FEBConnectorId,";
          sqlQuery += "  FEBLocation.FEBLocalEtaPartition,"; 
          sqlQuery += "  FEBLocation.PosInLocalEtaPartition,";
          sqlQuery += "  FEBLocation.FEBCMSEtaPartition,";
          sqlQuery += "  FEBLocation.PosInCMSEtaPartition,";
          sqlQuery += "  FEBConnector.LinkBoardInputNum ";
          sqlQuery += " FROM FEBLocation, FEBConnector ";
          sqlQuery += " WHERE FEBLocation.LB_LinkBoardId= ";
          sqlQuery +=  IntToString(theLB[iLB].first);
          sqlQuery += "  AND FEBLocation.FEBLocationId=FEBConnector.FL_FEBLocationId";
          sqlQuery += " ORDER BY FEBLocation.FEBLocationId, FEBConnector.FEBConnectorId";
          stmt->setSQL(sqlQuery.c_str());
          rset = stmt->executeQuery();
          while (rset->next()) {
            tmpFEB.febId=rset->getInt(1);
            tmpFEB.chamberId=rset->getInt(2);
            tmpFEB.connectorId=rset->getInt(3);
            tmpFEB.localEtaPart=rset->getString(4);
            tmpFEB.posInLocalEtaPart=rset->getInt(5);
            tmpFEB.cmsEtaPart=rset->getString(6);
            tmpFEB.posInCmsEtaPart=rset->getInt(7);
            tmpFEB.lbInputNum=rset->getInt(8);
            theFEB.push_back(tmpFEB);
}
          for(unsigned int iFEB=0; iFEB<theFEB.size(); iFEB++) {
            FebLocationSpec febLocation = {theFEB[iFEB].cmsEtaPart,theFEB[iFEB].posInCmsEtaPart,theFEB[iFEB].localEtaPart,theFEB[iFEB].posInLocalEtaPart};
// Get chamber 
            ChamberLocationSpec chamber;
            sqlQuery = "SELECT DiskOrWheel, Layer, Sector, Subsector,";
            sqlQuery += " ChamberLocationName,";
            sqlQuery += " FEBZOrnt, FEBRadOrnt, BarrelOrEndcap";
            sqlQuery += " FROM ChamberLocation ";
            sqlQuery += " WHERE ChamberLocationId= ";
            sqlQuery +=  IntToString(theFEB[iFEB].chamberId);
            stmt->setSQL(sqlQuery.c_str());
            rset = stmt->executeQuery();
            while (rset->next()) {
              chamber.diskOrWheel=rset->getInt(1);
              chamber.layer=rset->getInt(2);
              chamber.sector=rset->getInt(3);
              chamber.subsector=rset->getString(4);
              if (chamber.subsector=="") chamber.subsector="0";
              chamber.chamberLocationName=rset->getString(5);
              chamber.febZOrnt=rset->getString(6);
              chamber.febZRadOrnt=rset->getString(7);
              if (chamber.febZRadOrnt=="") chamber.febZRadOrnt="N/A";
              chamber.barrelOrEndcap=rset->getString(8);
            }
            FebConnectorSpec febConnector(theFEB[iFEB].lbInputNum,chamber,febLocation);
            // Get Strips
            sqlQuery = "SELECT CableChannelNum, ChamberStripNumber, CmsStripNumber";
            sqlQuery += " FROM ChamberStrip ";
            sqlQuery += " WHERE FC_FEBConnectorId= ";
            sqlQuery +=  IntToString(theFEB[iFEB].connectorId);
            sqlQuery += " ORDER BY CableChannelNum";
            stmt->setSQL(sqlQuery.c_str());
            rset = stmt->executeQuery();
            unsigned int iStripEntry=0;
            while (rset->next()) {
              ChamberStripSpec strip = {rset->getInt(1),rset->getInt(2),rset->getInt(3)};
              febConnector.add(strip);
              iStripEntry++;
            }
            lb.add(febConnector); 
          }
          lc.add(lb);
        }
        tb.add(lc);
      }
      dcc.add(tb);
    }
    std::cout<<"--> Adding DCC"<<std::endl;
    cabling->add(dcc);
  }
  cout << endl <<"Building RPC Cabling done!" << flush << endl << endl;
}

int popcon::RPCReadOutMappingSourceHandler::Compare2Cablings(const RPCReadOutMapping* map1, RPCReadOutMapping* map2) {
  vector<const DccSpec *> dccs1 = map1->dccList();
  vector<const DccSpec *> dccs2 = map2->dccList();
  if(dccs1.size()!=dccs2.size()) {
//    std::cout<<"Compare2Cablings: map sizes do not match :"<<dccs1.size()<<" "<<dccs2.size()<<std::endl;
    return 1;
  }
  pair<int,int> dccRange1 = map1->dccNumberRange();
  pair<int,int> dccRange2 = map2->dccNumberRange();
  if(dccRange1.first!=dccRange2.first) {
//    std::cout<<"Compare2Cablings: DCC range begins do not match :"<<dccRange1.first<<" "<<dccRange2.first<<std::endl;
    return 1;
  }
  if(dccRange1.second!=dccRange2.second) {
//    std::cout<<"Compare2Cablings: DCC range ends do not match :"<<dccRange1.second<<" "<<dccRange2.second<<std::endl;
    return 1;
  }
  typedef vector<const DccSpec *>::const_iterator IDCC;
  IDCC idcc2 = dccs2.begin();
  for (IDCC idcc1 = dccs1.begin(); idcc1 != dccs1.end(); idcc1++) {
    int dccNo = (**idcc1).id();
    std::string dccContents = (**idcc1).print(4);
    if ((**idcc2).id()!=dccNo) {
//      std::cout<<"Compare2Cablings: DCC numbers do not match :"<<dccNo<<" "<<(**idcc2).id()<<std::endl;
      return 1;
    }
    if ((**idcc2).print(4)!=dccContents) {
//      std::cout<<"Compare2Cablings: DCC contents do not match for DCC "<<dccNo<<std::endl;
      return 1;
    }
    idcc2++;
  }
  return 0;
}
