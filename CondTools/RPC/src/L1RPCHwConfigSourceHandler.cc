#include "CondTools/RPC/interface/L1RPCHwConfigSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::L1RPCHwConfigSourceHandler::L1RPCHwConfigSourceHandler(const edm::ParameterSet& ps):
  m_name(ps.getUntrackedParameter<std::string>("name","L1RPCHwConfigSourceHandler")),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_connect(ps.getUntrackedParameter<std::string>("OnlineConn","")),
  m_authpath(ps.getUntrackedParameter<std::string>("OnlineAuthPath",".")),
  m_host(ps.getUntrackedParameter<std::string>("OnlineDBHost","oracms.cern.ch")),
  m_sid(ps.getUntrackedParameter<std::string>("OnlineDBSID","omds")),
  m_user(ps.getUntrackedParameter<std::string>("OnlineDBUser","RPC_CONFIGURATION")),
  m_pass(ps.getUntrackedParameter<std::string>("OnlineDBPass","****")),
  m_port(ps.getUntrackedParameter<int>("OnlineDBPort",10121))
{
}

popcon::L1RPCHwConfigSourceHandler::~L1RPCHwConfigSourceHandler()
{
}

void popcon::L1RPCHwConfigSourceHandler::getNewObjects()
{

  std::cout << "L1RPCHwConfigSourceHandler: L1RPCHwConfigSourceHandler::getNewObjects begins\n";
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

//  std::cerr << "------- " << m_name 
//           << " - > getNewObjects" << std::endl;
//  std::cerr<<"Got offlineInfo, tag: "<<std::endl;
//  std::cerr << tagInfo().name << " , last object valid from " 
//          << tagInfo().lastInterval.first << " to "
//            << tagInfo().lastInterval.second << " , token is "
//            << tagInfo().token << " and this is the payload "
//            << tagInfo().lastPayloadToken << std::endl;

// first check what is already there in offline DB
  Ref payload;

  if(m_validate==1) {
    std::cout<<" Validation was requested, so will check present contents"<<std::endl;
    std::cout<<"Name of tag : "<<tagInfo().name << ", tag size : " << tagInfo().size
            << ", last object valid since "
            << tagInfo().lastInterval.first << std::endl;
    payload = lastPayload();
  } else {
    std::cout << "L1RPCHwConfigSourceHandler: no validation requested"<<std::endl;
  }

// now construct new object from online DB
        if (m_connect=="") {
          ConnectOnlineDB(m_host,m_sid,m_user,m_pass,m_port);
          readHwConfig0();
        } else {
          ConnectOnlineDB(m_connect,m_authpath);
          readHwConfig1();
        }
        DisconnectOnlineDB();

        cond::Time_t snc=mydbservice->currentTime();

// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2Configs(payload,disabledDevs);
        if (!difference) cout<<"No changes - will not write anything!!!"<<endl;
        if (difference==1) {
          cout<<"Will write new object to offline DB!!!"<<endl;
          m_to_transfer.push_back(std::make_pair((L1RPCHwConfig*)disabledDevs,snc));
        }

        std::cout << "L1RPCHwConfigSourceHandler: L1RPCHwConfigSourceHandler::getNewObjects ends\n";
}

void popcon::L1RPCHwConfigSourceHandler::ConnectOnlineDB(string host, string sid, string user, string pass, int port=10121)
{
  stringstream ss;
  ss << "//" << host << ":" << port << "/" << sid;
  cout << "L1RPCHwConfigSourceHandler: connecting to " << host << "..." << flush;
  env = Environment::createEnvironment(Environment::OBJECT);
  conn = env->createConnection(user, pass, ss.str());
  cout << "Done." << endl;
}

void popcon::L1RPCHwConfigSourceHandler::ConnectOnlineDB(string connect, string authPath)
{
  cout << "L1RPCHwConfigSourceHandler: connecting to " << connect << "..." << flush;
  session = new cond::DBSession();
  session->configuration().setAuthenticationMethod(cond::XML);
  session->configuration().setAuthenticationPath( authPath ) ;
  session->open() ;
  connection = new cond::Connection( connect ) ;
  connection->connect( session ) ;
  coralTr = & (connection->coralTransaction()) ;
  cout << "Done." << endl;
}

void popcon::L1RPCHwConfigSourceHandler::DisconnectOnlineDB()
{
  if (m_connect=="") {
    env->terminateConnection(conn);
    Environment::terminateEnvironment(env);
  } else {
    connection->disconnect() ;
    delete connection ;
    delete session ;
  }
}

void popcon::L1RPCHwConfigSourceHandler::readHwConfig0()
{
  Statement* stmt = conn->createStatement();
  string sqlQuery ="";
  cout << endl <<"L1RPCHwConfigSourceHandler: start to build L1RPC Hw Config..." << flush << endl << endl;
  disabledDevs =  new L1RPCHwConfig();

// get disabled crates and translate into towers/sectors/segments
  sqlQuery = "select tb.towerto, tb.towerfrom, tb.sector ";
  sqlQuery += "from CRATEDISABLED cd, CRATE c, BOARD b, TRIGGERBOARD tb "; 
  sqlQuery += "where c.crateid=cd.crate_crateid and b.crate_crateid=c.crateid and b.boardid=tb.triggerboardid and c.type='TRIGGERCRATE' ";
  sqlQuery += "order by tb.towerto, tb.sector ";
  stmt->setSQL(sqlQuery.c_str());
  ResultSet* rset = stmt->executeQuery();
  while (rset->next()) {
//    std::cout<<"  found board on disabled crate..."<<std::endl;
    int sector=atoi((rset->getString(3)).c_str());
    int first=atoi((rset->getString(1)).c_str());
    int last=atoi((rset->getString(2)).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  }

// get disabled triggerboards and translate into towers/sectors/segments
  sqlQuery = "select tb.towerto, tb.towerfrom, tb.sector ";
  sqlQuery += "from BOARDDISABLED bd, BOARD b, TRIGGERBOARD tb ";
  sqlQuery += "where b.boardid=bd.board_boardid and b.boardid=tb.triggerboardid ";
  sqlQuery += "order by tb.towerto, tb.sector ";
  stmt->setSQL(sqlQuery.c_str());
  rset = stmt->executeQuery();
  while (rset->next()) {
//    std::cout<<"  found disabled board..."<<std::endl;
    int sector=atoi((rset->getString(3)).c_str());
    int first=atoi((rset->getString(1)).c_str());
    int last=atoi((rset->getString(2)).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  }

// get disabled links - this is not usable here
/*
  sqlQuery = "select tb.towerto, tb.towerfrom, tb.sector, l.triggerboardinputnum ";
  sqlQuery += "from LINKDISABLED ld, LINKCONN l, TRIGGERBOARD tb ";
  sqlQuery += " where ld.link_linkconnid=l.linkconnid and l.tb_triggerboardid=tb.triggerboardid ";
  sqlQuery += "order by tb.towerto, tb.sector, l.triggerboardinputnum ";
  stmt->setSQL(sqlQuery.c_str());
  rset = stmt->executeQuery();
  while (rset->next()) {
    int sector=atoi((rset->getString(3)).c_str());
    int first=atoi((rset->getString(1)).c_str());
    int last=atoi((rset->getString(2)).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  } */

// get disabled chips and translate into towers/sectors
// for the moment assume that chip position 8 corresponds to lowest tower number
// and so on, ignoring bogus chip at position 11 if given TB operates 3 towers.
  sqlQuery = "select tb.towerto, tb.towerfrom, tb.sector, c.position ";
  sqlQuery += "from CHIPDISABLED cd, CHIP c, BOARD b, TRIGGERBOARD tb ";
  sqlQuery += " where cd.chip_chipid=c.chipid and c.board_boardid=b.boardid and b.boardid=tb.triggerboardid and c.type='PAC' ";
  sqlQuery += "order by tb.towerto, tb.sector, c.position ";
  stmt->setSQL(sqlQuery.c_str());
  rset = stmt->executeQuery();
  while (rset->next()) {
//    std::cout<<"  found disabled chip, not sure what to do with it..."<<std::endl;
    int sector=atoi((rset->getString(3)).c_str());
    int first=atoi((rset->getString(1)).c_str());
    int last=atoi((rset->getString(2)).c_str());
    int chipPos=rset->getInt(4);
    int tower=first+chipPos-8;
    if (tower<=last) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(tower,sector,jSegment,false);
      } 
    }
  }
}
    
void popcon::L1RPCHwConfigSourceHandler::readHwConfig1()
{
  coralTr->start( true );
  coral::ISchema& schema = coralTr->nominalSchema();
  std::string condition="";
  coral::AttributeList conditionData;
  cout << endl <<"L1RPCHwConfigSourceHandler: start to build L1RPC Hw Config..." << flush << endl << endl;

  disabledDevs =  new L1RPCHwConfig();

// get disabled crates and translate into towers/sectors/segments
  coral::IQuery* query1 = schema.newQuery();
  query1->addToTableList( "CRATEDISABLED" );
  query1->addToTableList( "CRATE" );
  query1->addToTableList( "BOARD" );
  query1->addToTableList( "TRIGGERBOARD" );
  query1->addToOutputList("TRIGGERBOARD.TOWERTO","TOWERTO");
  query1->addToOutputList("TRIGGERBOARD.TOWERFROM","TOWERFROM");
  query1->addToOutputList("TRIGGERBOARD.SECTOR","SECTOR");
  query1->addToOrderList( "TOWERTO" );
  query1->addToOrderList( "SECTOR" );
  condition = "CRATE.CRATEID=CRATEDISABLED.CRATE_CRATEID AND BOARD.CRATE_CRATEID=CRATE.CRATEID AND BOARD.BOARDID=TRIGGERBOARD.TRIGGERBOARDID AND CRATE.TYPE='TRIGGERCRATE'";
  query1->setCondition( condition, conditionData );
  coral::ICursor& cursor1 = query1->execute();
  while ( cursor1.next() ) {
//    cursor1.currentRow().toOutputStream( std::cout ) << std::endl;
    const coral::AttributeList& row = cursor1.currentRow();
    int sector = atoi((row["SECTOR"].data<std::string>()).c_str());
    int first = atoi((row["TOWERTO"].data<std::string>()).c_str());
    int last = atoi((row["TOWERFROM"].data<std::string>()).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  }
  delete query1;

// get disabled triggerboards and translate into towers/sectors/segments
  coral::IQuery* query2 = schema.newQuery();
  query2->addToTableList( "BOARDDISABLED" );
  query2->addToTableList( "BOARD" );
  query2->addToTableList( "TRIGGERBOARD" );
  query2->addToOutputList("TRIGGERBOARD.TOWERTO","TOWERTO");
  query2->addToOutputList("TRIGGERBOARD.TOWERFROM","TOWERFROM");
  query2->addToOutputList("TRIGGERBOARD.SECTOR","SECTOR");
  query2->addToOrderList( "TOWERTO" );
  query2->addToOrderList( "SECTOR" );
  condition = "BOARD.BOARDID=BOARDDISABLED.BOARD_BOARDID AND BOARD.BOARDID=TRIGGERBOARD.TRIGGERBOARDID";
  query2->setCondition( condition, conditionData );
  coral::ICursor& cursor2 = query2->execute();
  while ( cursor2.next() ) {
//    cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
    const coral::AttributeList& row = cursor2.currentRow();
    int sector = atoi((row["SECTOR"].data<std::string>()).c_str());
    int first = atoi((row["TOWERTO"].data<std::string>()).c_str());
    int last = atoi((row["TOWERFROM"].data<std::string>()).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  }
  delete query2;

// get disabled links - this is not usable here
/*
  coral::IQuery* query3 = schema.newQuery();
  query3->addToTableList( "LINKDISABLED" );
  query3->addToTableList( "LINKCONN" );
  query3->addToTableList( "TRIGGERBOARD" );
  query3->addToOutputList("TRIGGERBOARD.TOWERTO","TOWERTO");
  query3->addToOutputList("TRIGGERBOARD.TOWERFROM","TOWERFROM");
  query3->addToOutputList("TRIGGERBOARD.SECTOR","SECTOR");
  query3->addToOutputList("LINKCONN.TRIGGERBOARDINPUTNUM","TBINPUTNUM");
  query3->addToOrderList( "TOWERTO" );
  query3->addToOrderList( "SECTOR" );
  query3->addToOrderList( "TBINPUTNUM" );
  condition = "LINKCONN.LINKCONNID=LINKDISABLED.LINK_LINKCONNID AND LINKCONN.TB_TRIGGERBOARDID=TRIGGERBOARD.TRIGGERBOARDID";
  query3->setCondition( condition, conditionData );
  coral::ICursor& cursor3 = query3->execute();
  while ( cursor3.next() ) {
//    cursor3.currentRow().toOutputStream( std::cout ) << std::endl;
    const coral::AttributeList& row = cursor3.currentRow();
    int sector = atoi((row["SECTOR"].data<std::string>()).c_str());
    int first = atoi((row["TOWERTO"].data<std::string>()).c_str());
    int last = atoi((row["TOWERFROM"].data<std::string>()).c_str());
    for (int iTower=first; iTower<=last; iTower++) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(iTower,sector,jSegment,false);
      }
    }
  }
  delete query3;*/

// get disabled chips and translate into towers/sectors
// for the moment assume that chip position 8 corresponds to lowest tower number
// and so on, ignoring bogus chip at position 11 if given TB operates 3 towers.
  coral::IQuery* query4 = schema.newQuery();
  query4->addToTableList( "CHIPDISABLED" );
  query4->addToTableList( "CHIP" );
  query4->addToTableList( "BOARD" );
  query4->addToTableList( "TRIGGERBOARD" );
  query4->addToOutputList("TRIGGERBOARD.TOWERTO","TOWERTO");
  query4->addToOutputList("TRIGGERBOARD.TOWERFROM","TOWERFROM");
  query4->addToOutputList("TRIGGERBOARD.SECTOR","SECTOR");
  query4->addToOutputList("CHIP.POSITION","POSITION");
  query4->addToOrderList( "TOWERTO" );
  query4->addToOrderList( "SECTOR" );
  query4->addToOrderList( "POSITION" );
  condition = "CHIP.CHIPID=CHIPDISABLED.CHIP_CHIPID AND CHIP.BOARD_BOARDID=BOARD.BOARDID AND BOARD.BOARDID=TRIGGERBOARD.TRIGGERBOARDID AND CHIP.TYPE='PAC'";
  query4->setCondition( condition, conditionData );
  coral::ICursor& cursor4 = query4->execute();
  while ( cursor4.next() ) {
//    cursor4.currentRow().toOutputStream( std::cout ) << std::endl;
    const coral::AttributeList& row = cursor4.currentRow();
    int sector = atoi((row["SECTOR"].data<std::string>()).c_str());
    int first = atoi((row["TOWERTO"].data<std::string>()).c_str());
    int last = atoi((row["TOWERFROM"].data<std::string>()).c_str());
    int chipPos=row["POSITION"].data<short>();
    int tower=first+chipPos-8;
    if (tower<=last) {
      for (int jSegment=0; jSegment<12; jSegment++) {
        disabledDevs->enablePAC(tower,sector,jSegment,false);
      }
    }
  }
  delete query4;

  coralTr->commit();
}

int popcon::L1RPCHwConfigSourceHandler::Compare2Configs(Ref set1, L1RPCHwConfig* set2)
{
  std::cout<<"Size of new object is : "<<flush;
  std::cout<<set2->size()<<std::endl;
  std::cout<<"Size of ref object is : "<<flush;
  std::cout<<set1->size()<<std::endl;

  if (set1->size() != set2->size()) {
    std::cout<<" Number of disabled devices changed "<<std::endl;
    return 1;
  }
  for (int tower=-16; tower<17; tower++) {
    for (int sector=0; sector<12; sector++) {
      for (int segment=0; segment<12; segment++)
      if (set1->isActive(tower,sector,segment) != set2->isActive(tower,sector,segment)) {
        std::cout<<" Configuration changed for tower "<<tower<<", sector "<<sector<<", segment "<<segment<<std::endl;
        return 1;
      }
    }
  }
  return 0;
}
