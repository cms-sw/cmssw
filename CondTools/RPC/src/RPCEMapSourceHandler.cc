#include "CondTools/RPC/interface/RPCEMapSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::RPCEMapSourceHandler::RPCEMapSourceHandler(const edm::ParameterSet& ps) :
  m_name(ps.getUntrackedParameter<std::string>("name","RPCEMapSourceHandler")),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_host(ps.getUntrackedParameter<std::string>("OnlineDBHost","oracms.cern.ch")),
  m_sid(ps.getUntrackedParameter<std::string>("OnlineDBSID","omds")),
  m_user(ps.getUntrackedParameter<std::string>("OnlineDBUser","RPC_CONFIGURATION")),
  m_pass(ps.getUntrackedParameter<std::string>("OnlineDBPass","blahblah")),
  m_port(ps.getUntrackedParameter<int>("OnlineDBPort",1521))
{
//	std::cout << "RPCEMapSourceHandler: RPCEMapSourceHandler constructor" << std::endl;
}

popcon::RPCEMapSourceHandler::~RPCEMapSourceHandler()
{
}

void popcon::RPCEMapSourceHandler::getNewObjects()
{

//	std::cout << "RPCEMapSourceHandler: RPCEMapSourceHandler::getNewObjects begins\n";

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

// first check what is already there in offline DB
  Ref payload;
  if(m_validate==1 && tagInfo().size>0) {
    std::cout<<" Validation was requested, so will check present contents"<<std::endl;
    std::cout<<"Name of tag : "<<tagInfo().name << ", tag size : " << tagInfo().size 
            << ", last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  
    payload = lastPayload();
  }

// now construct new cabling map from online DB
        ConnectOnlineDB(m_host,m_sid,m_user,m_pass,m_port);
        readEMap();
        DisconnectOnlineDB();

        cond::Time_t snc=mydbservice->currentTime();
	
// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2EMaps(payload,eMap);
        if (!difference) cout<<"No changes - will not write anything!!!"<<endl;
        if (difference==1) {
          cout<<"Will write new object to offline DB!!!"<<endl;
          m_to_transfer.push_back(std::make_pair((RPCEMap*)eMap,snc));
//          delete eMap;
        }

//	std::cout << "RPCEMapSourceHandler: RPCEMapSourceHandler::getNewObjects ends\n";
}

void popcon::RPCEMapSourceHandler::ConnectOnlineDB(string host, string sid, string user, string pass, int port=1521)
{
  stringstream ss;
  ss << "//" << host << ":" << port << "/" << sid;

  cout << "RPCEMapSourceHandler: connecting to " << host << "..." << flush;
  env = Environment::createEnvironment(Environment::OBJECT);
  conn = env->createConnection(user, pass, ss.str());
  cout << "Done." << endl;
}

void popcon::RPCEMapSourceHandler::DisconnectOnlineDB()
{
  env->terminateConnection(conn);
  Environment::terminateEnvironment(env);
}

void popcon::RPCEMapSourceHandler::readEMap()
{

//  string eMap_version = "test";
  time_t rawtime;
  time(&rawtime); //time since January 1, 1970
  tm * ptm = gmtime(&rawtime);//GMT time
  char buffer[20];
  strftime(buffer,20,"%d/%m/%Y_%H:%M:%S",ptm);
  string eMap_version=(string)buffer;

  Statement* stmt = conn->createStatement();
  string sqlQuery ="";

  cout << endl <<"RPCEMapSourceHandler: start to build RPC e-Map..." << flush << endl << endl;
  eMap =  new RPCEMap(eMap_version);

  // Get FEDs
  RPCEMap::dccItem thisDcc;
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
    thisDcc.theId=theDAQ[iFED].second;
    std::vector<std::pair<int,int> > theTB;
// get TBs
    RPCEMap::tbItem thisTB;
    sqlQuery = " SELECT TriggerBoardId, DCCInputChannelNum FROM TriggerBoard ";
    sqlQuery += " WHERE DCCBoard_DCCBoardId= ";
    sqlQuery += IntToString(theDAQ[iFED].first);
    sqlQuery += " ORDER BY DCCInputChannelNum ";
    stmt->setSQL(sqlQuery.c_str());
    rset = stmt->executeQuery();
    int ntbs=0;
    while (rset->next()) {
      ntbs++;
      tmp_tbl.first=rset->getInt(1);
      tmp_tbl.second=rset->getInt(2);
      theTB.push_back(tmp_tbl);
    }
    for(unsigned int iTB=0;iTB<theTB.size();iTB++) {
      thisTB.theNum=theTB[iTB].second;
      thisTB.theMaskedLinks=0;
      std::vector<std::pair<int,int> > theLink;
// get links
      RPCEMap::linkItem thisLink;
      sqlQuery = " SELECT Board_BoardId, TriggerBoardInputNum FROM LinkConn ";
      sqlQuery += " WHERE TB_TriggerBoardId= ";
      sqlQuery +=  IntToString(theTB[iTB].first);
      sqlQuery += " ORDER BY TriggerBoardInputNum ";
      stmt->setSQL(sqlQuery.c_str());
      rset = stmt->executeQuery();
      int nlinks=0;
      while (rset->next()) {
        nlinks++;
        tmp_tbl.first=rset->getInt(1);
        tmp_tbl.second=rset->getInt(2);
        theLink.push_back(tmp_tbl);
      }
      for(unsigned int iLink=0;iLink<theLink.size();iLink++) {
        int boardId=theLink[iLink].first;
        thisLink.theTriggerBoardInputNumber=theLink[iLink].second;
        std::vector<std::pair<int,string> > theLB;
        std::pair<int,string> tmpLB;
        // Get master LBs first...
        RPCEMap::lbItem thisLB;
        sqlQuery = " SELECT Name ";
        sqlQuery += " FROM Board ";
        sqlQuery += " WHERE BoardId= ";
        sqlQuery +=  IntToString(theLink[iLink].first);
        stmt->setSQL(sqlQuery.c_str());
        rset = stmt->executeQuery();
        int nlbs=0;
        while (rset->next()) {
          nlbs++;
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
          nlbs++;
          tmpLB.first=rset->getInt(1);
          tmpLB.second=rset->getString(2);
          theLB.push_back(tmpLB);
        }
        for(unsigned int iLB=0; iLB<theLB.size(); iLB++) {
          thisLB.theLinkBoardNumInLink=atoi(((theLB[iLB].second).substr((theLB[iLB].second).length()-1,1)).c_str());
          thisLB.theMaster = (theLB[iLB].first==boardId);
          FEBStruct tmpFEB;
          std::vector<FEBStruct> theFEB;
// get FEBs
          RPCEMap::febItem thisFeb;
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
          int nfebs=0;
          while (rset->next()) {
            nfebs++;
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
            thisFeb.localEtaPartition=theFEB[iFEB].localEtaPart;
            thisFeb.positionInLocalEtaPartition=theFEB[iFEB].posInLocalEtaPart;
            thisFeb.cmsEtaPartition=theFEB[iFEB].cmsEtaPart;
            thisFeb.positionInCmsEtaPartition=theFEB[iFEB].posInCmsEtaPart;
            thisFeb.theLinkBoardInputNum=theFEB[iFEB].lbInputNum;
            FebLocationSpec febLocation = {theFEB[iFEB].cmsEtaPart,theFEB[iFEB].posInCmsEtaPart,theFEB[iFEB].localEtaPart,theFEB[iFEB].posInLocalEtaPart};
            // Get chamber 
            sqlQuery = "SELECT DiskOrWheel, Layer, Sector, Subsector,";
            sqlQuery += " ChamberLocationName,";
            sqlQuery += " FEBZOrnt, FEBRadOrnt, BarrelOrEndcap";
            sqlQuery += " FROM ChamberLocation ";
            sqlQuery += " WHERE ChamberLocationId= ";
            sqlQuery +=  IntToString(theFEB[iFEB].chamberId);
            stmt->setSQL(sqlQuery.c_str());
            rset = stmt->executeQuery();
            while (rset->next()) {
              thisFeb.diskOrWheel=rset->getInt(1);
              thisFeb.layer=rset->getInt(2);
              thisFeb.sector=rset->getInt(3);
              thisFeb.subsector=rset->getString(4);
              if (thisFeb.subsector=="") thisFeb.subsector="0";
              thisFeb.chamberLocationName=rset->getString(5);
              thisFeb.febZOrnt=rset->getString(6);
              thisFeb.febZRadOrnt=rset->getString(7);
              if (thisFeb.febZRadOrnt=="") thisFeb.febZRadOrnt="N/A";
              thisFeb.barrelOrEndcap=rset->getString(8);
              ChamberLocationSpec chamber = {thisFeb.diskOrWheel,thisFeb.layer,thisFeb.sector,thisFeb.subsector,thisFeb.chamberLocationName,thisFeb.febZOrnt,thisFeb.febZRadOrnt,thisFeb.barrelOrEndcap};
              DBSpecToDetUnit toDU;
              thisFeb.theRawId=toDU(chamber,febLocation);
            }
            // Get Strips
            RPCEMap::stripItem thisStrip;
            sqlQuery = "SELECT CableChannelNum, ChamberStripNumber, CmsStripNumber";
            sqlQuery += " FROM ChamberStrip ";
            sqlQuery += " WHERE FC_FEBConnectorId= ";
            sqlQuery +=  IntToString(theFEB[iFEB].connectorId);
            sqlQuery += " ORDER BY CableChannelNum";
            stmt->setSQL(sqlQuery.c_str());
            rset = stmt->executeQuery();
            int nstrips=0;
            while (rset->next()) {
              thisStrip.cablePinNumber=rset->getInt(1);
              thisStrip.chamberStripNumber=rset->getInt(2);
              thisStrip.cmsStripNumber=rset->getInt(3);
              eMap->theStrips.push_back(thisStrip);
              nstrips++;
            }
            thisFeb.nStrips=nstrips;
            eMap->theFebs.push_back(thisFeb);
          }
          thisLB.nFebs=nfebs;
          eMap->theLBs.push_back(thisLB);
        }
        thisLink.nLBs=nlbs;
        eMap->theLinks.push_back(thisLink);
      }
      thisTB.nLinks=nlinks;
      eMap->theTBs.push_back(thisTB);
    }
    thisDcc.nTBs=ntbs;
    std::cout<<"DCC added"<<std::endl;
    eMap->theDccs.push_back(thisDcc);
  }
  cout << endl <<"Building RPC e-Map done!" << flush << endl << endl;
}

//int popcon::RPCEMapSourceHandler::Compare2EMaps(const RPCEMap* map1, RPCEMap* map2) {
int popcon::RPCEMapSourceHandler::Compare2EMaps(Ref map1, RPCEMap* map2) {
  RPCReadOutMapping* oldmap1 = map1->convert();
  RPCReadOutMapping* oldmap2 = map2->convert();
  vector<const DccSpec *> dccs1 = oldmap1->dccList();
  vector<const DccSpec *> dccs2 = oldmap2->dccList();
  if(dccs1.size()!=dccs2.size()) {
//    std::cout<<"Compare2EMaps: map sizes do not match :"<<dccs1.size()<<" "<<dccs2.size()<<std::endl;
    return 1;
  }
  pair<int,int> dccRange1 = oldmap1->dccNumberRange();
  pair<int,int> dccRange2 = oldmap2->dccNumberRange();
  if(dccRange1.first!=dccRange2.first) {
//    std::cout<<"Compare2EMaps: DCC range begins do not match :"<<dccRange1.first<<" "<<dccRange2.first<<std::endl;
    return 1;
  }
  if(dccRange1.second!=dccRange2.second) {
//    std::cout<<"Compare2EMaps: DCC range ends do not match :"<<dccRange1.second<<" "<<dccRange2.second<<std::endl;
    return 1;
  }
  typedef vector<const DccSpec *>::const_iterator IDCC;
  IDCC idcc2 = dccs2.begin();
  for (IDCC idcc1 = dccs1.begin(); idcc1 != dccs1.end(); idcc1++) {
    int dccNo = (**idcc1).id();
    std::string dccContents = (**idcc1).print(4);
    if ((**idcc2).id()!=dccNo) {
//      std::cout<<"Compare2EMaps: DCC numbers do not match :"<<dccNo<<" "<<(**idcc2).id()<<std::endl;
      return 1;
    }
    if ((**idcc2).print(4)!=dccContents) {
//      std::cout<<"Compare2EMaps: DCC contents do not match for DCC "<<dccNo<<std::endl;
      return 1;
    }
    idcc2++;
  }
  return 0;
}
