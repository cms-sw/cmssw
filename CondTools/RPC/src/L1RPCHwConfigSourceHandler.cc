#include "CondTools/RPC/interface/L1RPCHwConfigSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::L1RPCHwConfigSourceHandler::L1RPCHwConfigSourceHandler(const edm::ParameterSet& ps):
  m_name(ps.getUntrackedParameter<std::string>("name","L1RPCHwConfigSourceHandler")),
  m_dummy(ps.getUntrackedParameter<int>("WriteDummy",0)),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_disableCrates(ps.getUntrackedParameter<std::vector<int> >("DisabledCrates")),
  m_disableTowers(ps.getUntrackedParameter<std::vector<int> >("DisabledTowers")),
  m_connect(ps.getUntrackedParameter<std::string>("OnlineConn","")),
  m_authpath(ps.getUntrackedParameter<std::string>("OnlineAuthPath","."))
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
  disabledDevs =  new L1RPCHwConfig();
      if (m_dummy==1) {
        std::vector<int>::iterator crIt = m_disableCrates.begin();
        for (; crIt!=m_disableCrates.end(); ++crIt){
          disabledDevs->enableCrate(*crIt,false);
        }
        std::vector<int>::iterator twIt = m_disableTowers.begin();
        for (; twIt!=m_disableTowers.end(); ++twIt){
          disabledDevs->enableTower(*twIt,false);
        }
      } else {
        ConnectOnlineDB(m_connect,m_authpath);
        readHwConfig1();
        DisconnectOnlineDB();
      }

        cond::Time_t snc=mydbservice->currentTime();

// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2Configs(payload,disabledDevs);
        if (!difference) std::cout<<"No changes - will not write anything!!!"<<std::endl;
        if (difference==1) {
          std::cout<<"Will write new object to offline DB!!!"<<std::endl;
          m_to_transfer.push_back(std::make_pair((L1RPCHwConfig*)disabledDevs,snc+1));
        }

        std::cout << "L1RPCHwConfigSourceHandler: L1RPCHwConfigSourceHandler::getNewObjects ends\n";
}

void popcon::L1RPCHwConfigSourceHandler::ConnectOnlineDB(std::string connect, std::string authPath)
{
  std::cout << "L1RPCHwConfigSourceHandler: connecting to " << connect << "..." << std::flush;
  cond::persistency::ConnectionPool connection;
//  session->configuration().setAuthenticationMethod(cond::XML);
  connection.setAuthenticationPath( authPath ) ;
  connection.configure();
  session = connection.createSession( connect,true );
  std::cout << "Done." << std::endl;
}

void popcon::L1RPCHwConfigSourceHandler::DisconnectOnlineDB()
{
  session.close();
}

void popcon::L1RPCHwConfigSourceHandler::readHwConfig1()
{
  session.transaction().start( true );
  coral::ISchema& schema = session.nominalSchema();
  std::string condition="";
  coral::AttributeList conditionData;
  std::cout << std::endl <<"L1RPCHwConfigSourceHandler: start to build L1RPC Hw Config..." << std::flush << std::endl << std::endl;

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

}

int popcon::L1RPCHwConfigSourceHandler::Compare2Configs(const Ref& _set1, L1RPCHwConfig* set2)
{
  Ref set1 = _set1;
  std::cout<<"Size of new object is : "<<std::flush;
  std::cout<<set2->size()<<std::endl;
  std::cout<<"Size of ref object is : "<<std::flush;
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
