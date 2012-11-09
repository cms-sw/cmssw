#include "CondTools/RPC/interface/RPCEMapSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::RPCEMapSourceHandler::RPCEMapSourceHandler(const edm::ParameterSet& ps) :
  m_name(ps.getUntrackedParameter<std::string>("name","RPCEMapSourceHandler")),
  m_dummy(ps.getUntrackedParameter<int>("WriteDummy",0)),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_connect(ps.getUntrackedParameter<std::string>("OnlineConn","")),
  m_authpath(ps.getUntrackedParameter<std::string>("OnlineAuthPath","."))
{
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
  time_t rawtime;
  time(&rawtime); //time since January 1, 1970
  tm * ptm = gmtime(&rawtime);//GMT time
  char buffer[20];
  strftime(buffer,20,"%d/%m/%Y_%H:%M:%S",ptm);
  std::string eMap_version=(std::string)buffer;

  eMap =  new RPCEMap(eMap_version);
      if (m_dummy==0) {
        ConnectOnlineDB(m_connect,m_authpath);
        readEMap1();
        DisconnectOnlineDB();
      }

        cond::Time_t snc=mydbservice->currentTime();
	
// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2EMaps(payload,eMap);
        if (!difference) std::cout<<"No changes - will not write anything!!!"<<std::endl;
        if (difference==1) {
          std::cout<<"Will write new object to offline DB!!!"<<std::endl;
          m_to_transfer.push_back(std::make_pair((RPCEMap*)eMap,snc));
        }

//	std::cout << "RPCEMapSourceHandler: RPCEMapSourceHandler::getNewObjects ends\n";
}

void popcon::RPCEMapSourceHandler::ConnectOnlineDB(std::string connect, std::string authPath)
{
  std::cout << "RPCEMapConfigSourceHandler: connecting to " << connect << "..." << std::flush;
  connection = new cond::DbConnection() ;
//  session->configuration().setAuthenticationMethod(cond::XML);
  connection->configuration().setAuthenticationPath( authPath ) ;
  connection->configure();
  session = new cond::DbSession(connection->createSession());
  session->open(connect,true) ;
  std::cout << "Done." << std::endl;
}

void popcon::RPCEMapSourceHandler::DisconnectOnlineDB()
{
  connection->close() ;
  delete connection ;
  session->close();
  delete session ;
}

void popcon::RPCEMapSourceHandler::readEMap1()
{
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  std::string condition="";
  coral::AttributeList conditionData;

  std::cout << std::endl <<"RPCEMapSourceHandler: start to build RPC e-Map..." << std::flush << std::endl << std::endl;

  // Get FEDs
  RPCEMap::dccItem thisDcc;
  coral::IQuery* query1 = schema.newQuery();
  query1->addToTableList( "DCCBOARD" );
  query1->addToOutputList("DCCBOARD.DCCBOARDID","DCCBOARDID");
  query1->addToOutputList("DCCBOARD.FEDNUMBER","FEDNUMBER");
  query1->addToOrderList("FEDNUMBER");
  condition = "DCCBOARD.DCCBOARDID>0";
  query1->setCondition( condition, conditionData );
//  std::cout<<"Getting DCCBOARD...";
  coral::ICursor& cursor1 = query1->execute();
  std::cout<<"OK"<<std::endl;
  std::pair<int,int> tmp_tbl;
  std::vector< std::pair<int,int> > theDAQ;
  while ( cursor1.next() ) {
//    cursor1.currentRow().toOutputStream( std::cout ) << std::endl;
    const coral::AttributeList& row = cursor1.currentRow();
    tmp_tbl.first=row["DCCBOARDID"].data<long long>();
    tmp_tbl.second=row["FEDNUMBER"].data<long long>();
    theDAQ.push_back(tmp_tbl);
  }
  delete query1;

  for(unsigned int iFED=0;iFED<theDAQ.size();iFED++) {
    thisDcc.theId=theDAQ[iFED].second;
    std::vector<std::pair<int,int> > theTB;
// get TBs
    RPCEMap::tbItem thisTB;
    coral::IQuery* query2 = schema.newQuery();
    query2->addToTableList( "TRIGGERBOARD" );
    query2->addToOutputList("TRIGGERBOARD.TRIGGERBOARDID","TRIGGERBOARDID");
    query2->addToOutputList("TRIGGERBOARD.DCCINPUTCHANNELNUM","DCCCHANNELNUM");
    query2->addToOrderList("DCCCHANNELNUM");
    condition = "TRIGGERBOARD.DCCBOARD_DCCBOARDID="+IntToString(theDAQ[iFED].first);
    query2->setCondition( condition, conditionData );
    coral::ICursor& cursor2 = query2->execute();
    int ntbs=0;
    while ( cursor2.next() ) {
      ntbs++;
//      cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
      const coral::AttributeList& row = cursor2.currentRow();
      tmp_tbl.first=row["TRIGGERBOARDID"].data<long long>();
      tmp_tbl.second=row["DCCCHANNELNUM"].data<long long>();
      theTB.push_back(tmp_tbl);
    }
    delete query2;
    for(unsigned int iTB=0;iTB<theTB.size();iTB++) {
      thisTB.theNum=theTB[iTB].second;
      std::vector<std::pair<int,int> > theLink;
// get links
      RPCEMap::linkItem thisLink;
      coral::IQuery* query3 = schema.newQuery();
      query3->addToTableList("BOARDBOARDCONN");
      query3->addToOutputList("BOARDBOARDCONN.BOARD_BOARDID","BOARDID");
      query3->addToOutputList("BOARDBOARDCONN.COLLECTORBOARDINPUTNUM","TBINPUTNUM");
      query3->addToOrderList("TBINPUTNUM");
      condition = "BOARDBOARDCONN.BOARD_COLLECTORBOARDID="+IntToString(theTB[iTB].first);
      query3->setCondition( condition, conditionData );
      coral::ICursor& cursor3 = query3->execute();
      int nlinks=0;
      while (cursor3.next()) {
        nlinks++;
        const coral::AttributeList& row = cursor3.currentRow();
        tmp_tbl.first=row["BOARDID"].data<long long>();
        tmp_tbl.second=row["TBINPUTNUM"].data<long long>();
        theLink.push_back(tmp_tbl);
      }
      delete query3;
      for(unsigned int iLink=0;iLink<theLink.size();iLink++) {
        int boardId=theLink[iLink].first;
        thisLink.theTriggerBoardInputNumber=theLink[iLink].second;
        std::vector<std::pair<int,std::string> > theLB;
        std::pair<int,std::string> tmpLB;
        // Get master LBs first...
        RPCEMap::lbItem thisLB;
        coral::IQuery* query4 = schema.newQuery();
        query4->addToTableList("BOARD");
        query4->addToOutputList("BOARD.NAME","NAME");
        condition = "BOARD.BOARDID="+IntToString(theLink[iLink].first);
        query4->setCondition( condition, conditionData );
        coral::ICursor& cursor4 = query4->execute();
        int nlbs=0;
        while (cursor4.next()) {
          nlbs++;
          const coral::AttributeList& row = cursor4.currentRow();
          tmpLB.first=theLink[iLink].first;
          tmpLB.second=row["NAME"].data<std::string>();
          theLB.push_back(tmpLB);
        }
        delete query4;
        // then slaves
        coral::IQuery* query5 = schema.newQuery();
        query5->addToTableList("LINKBOARD");
        query5->addToTableList("BOARD");
        query5->addToOutputList("LINKBOARD.LINKBOARDID","LINKBOARDID");
        query5->addToOutputList("BOARD.NAME","NAME");
        query5->addToOrderList("LINKBOARDID");
        condition = "LINKBOARD.MASTERID="+IntToString(theLink[iLink].first)+" AND BOARD.BOARDID=LINKBOARD.LINKBOARDID AND LINKBOARD.MASTERID<>LINKBOARD.LINKBOARDID";
        query5->setCondition( condition, conditionData );
        coral::ICursor& cursor5 = query5->execute();
        while (cursor5.next()) {
          nlbs++;
          const coral::AttributeList& row = cursor5.currentRow();
          tmpLB.first=row["LINKBOARDID"].data<long long>();
          tmpLB.second=row["NAME"].data<std::string>();
          theLB.push_back(tmpLB);
        }
        delete query5;
        for(unsigned int iLB=0; iLB<theLB.size(); iLB++) {
          thisLB.theMaster = (theLB[iLB].first==boardId);
// extract all relevant information from BOARD.NAME
          std::string theName = theLB[iLB].second;
          int slength = theName.length();
          thisLB.theLinkBoardNumInLink=atoi((theName.substr(slength-1,1)).c_str());
          int wheel=atoi((theName.substr(6,1)).c_str());
          std::string char1=(theName.substr(4,1)).c_str();
          std::string char2=(theName.substr(slength-7,1)).c_str();
          int num3=atoi((theName.substr(slength-6,1)).c_str());
          std::string char4=(theName.substr(slength-5,1)).c_str();
          bool itsS1to9=(theName.substr(slength-11,1)=="S");
          int n1=10;
          int n2=1;
          int n3=0;
          if (!itsS1to9) {
            n1=11;
            n2=2;
          }
          int sector=atoi((theName.substr(slength-n1,n2)).c_str());
          std::string char1Val[2]={"B","E"};                              // 1,2
          std::string char2Val[3]={"N","M","P"};                          // 0,1,2
          std::string char4Val[9]={"0","1","2","3","A","B","C","D","E"};  // 0,...,8
          for (int i=0; i<2; i++) if (char1==char1Val[i]) n1=i+1;
          for (int i=0; i<3; i++) if (char2==char2Val[i]) n2=i;
          for (int i=0; i<9; i++) if (char4==char4Val[i]) n3=i;
          thisLB.theCode=n3+num3*10+n2*100+n1*1000+wheel*10000+sector*100000;
          FEBStruct tmpFEB;
          std::vector<FEBStruct> theFEB;
// get FEBs
          RPCEMap::febItem thisFeb;
          coral::IQuery* query6 = schema.newQuery();
          query6->addToTableList("FEBLOCATION");
          query6->addToTableList("FEBCONNECTOR");
          query6->addToOutputList("FEBLOCATION.FEBLOCATIONID","FEBLOCATIONID");
          query6->addToOutputList("FEBLOCATION.CL_CHAMBERLOCATIONID","CHAMBERLOCATIONID");
          query6->addToOutputList("FEBCONNECTOR.FEBCONNECTORID","FEBCONNECTORID");
          query6->addToOutputList("FEBLOCATION.FEBLOCALETAPARTITION","LOCALETAPART"); 
          query6->addToOutputList("FEBLOCATION.POSINLOCALETAPARTITION","POSINLOCALETAPART");
          query6->addToOutputList("FEBLOCATION.FEBCMSETAPARTITION","CMSETAPART");
          query6->addToOutputList("FEBLOCATION.POSINCMSETAPARTITION","POSINCMSETAPART");
          query6->addToOutputList("FEBCONNECTOR.LINKBOARDINPUTNUM","LINKBOARDINPUTNUM");
          query6->addToOrderList("FEBLOCATIONID");
          query6->addToOrderList("FEBCONNECTORID");
          condition = "FEBLOCATION.LB_LINKBOARDID="+IntToString(theLB[iLB].first)+" AND FEBLOCATION.FEBLOCATIONID=FEBCONNECTOR.FL_FEBLOCATIONID";
          query6->setCondition( condition, conditionData );
          coral::ICursor& cursor6 = query6->execute();
          int nfebs=0;
          while (cursor6.next()) {
            nfebs++;
            const coral::AttributeList& row = cursor6.currentRow();
            tmpFEB.febId=row["FEBLOCATIONID"].data<long long>();
            tmpFEB.chamberId=row["CHAMBERLOCATIONID"].data<long long>();
            tmpFEB.connectorId=row["FEBCONNECTORID"].data<long long>();
            tmpFEB.localEtaPart=row["LOCALETAPART"].data<std::string>();
            tmpFEB.posInLocalEtaPart=row["POSINLOCALETAPART"].data<short>();
            tmpFEB.cmsEtaPart=row["CMSETAPART"].data<std::string>();
            tmpFEB.posInCmsEtaPart=row["POSINCMSETAPART"].data<short>();
            tmpFEB.lbInputNum=row["LINKBOARDINPUTNUM"].data<short>();
            theFEB.push_back(tmpFEB);
          }
          delete query6;
          for(unsigned int iFEB=0; iFEB<theFEB.size(); iFEB++) {
            std::string temp=theFEB[iFEB].localEtaPart;
            std::string localEtaVal[6]={"Forward","Central","Backward","A","B","C"};
            char localEtaPartition=0;
            for (int i=0; i<6; i++) if (temp==localEtaVal[i]) localEtaPartition=i+1;
            char positionInLocalEtaPartition=theFEB[iFEB].posInLocalEtaPart;
            temp=theFEB[iFEB].cmsEtaPart;
            std::string cmsEtaVal[6]={"1","2","3","A","B","C"};
            char cmsEtaPartition=0;
            for (int i=0; i<6; i++) if (temp==cmsEtaVal[i]) cmsEtaPartition=i+1;
            char positionInCmsEtaPartition=theFEB[iFEB].posInCmsEtaPart;
            thisFeb.thePartition=positionInLocalEtaPartition+10*localEtaPartition+100*positionInCmsEtaPartition+1000*cmsEtaPartition;
            thisFeb.theLinkBoardInputNum=theFEB[iFEB].lbInputNum;
            // Get chamber 
            coral::IQuery* query7 = schema.newQuery();
            query7->addToTableList("CHAMBERLOCATION");
            query7->addToOutputList("CHAMBERLOCATION.DISKORWHEEL","DISKORWHEEL");
            query7->addToOutputList("CHAMBERLOCATION.LAYER","LAYER");
            query7->addToOutputList("CHAMBERLOCATION.SECTOR","SECTOR");
            query7->addToOutputList("CHAMBERLOCATION.SUBSECTOR","SUBSECTOR");
            query7->addToOutputList("CHAMBERLOCATION.CHAMBERLOCATIONNAME","NAME");
            query7->addToOutputList("CHAMBERLOCATION.FEBZORNT","FEBZORNT");
            query7->addToOutputList("CHAMBERLOCATION.FEBRADORNT","FEBRADORNT");
            query7->addToOutputList("CHAMBERLOCATION.BARRELORENDCAP","BARRELORENDCAP");
            condition = "CHAMBERLOCATION.CHAMBERLOCATIONID="+IntToString(theFEB[iFEB].chamberId);
            query7->setCondition( condition, conditionData );
            coral::ICursor& cursor7 = query7->execute();
            while (cursor7.next()) {
              const coral::AttributeList& row = cursor7.currentRow();
              char diskOrWheel=row["DISKORWHEEL"].data<short>()+4;
              char layer=row["LAYER"].data<short>();
              int sector=row["SECTOR"].data<short>();
              temp=row["SUBSECTOR"].data<std::string>();
// TEMPORARY TO CORRECT A LITTLE BUG IN OMDS
//              std::string chname=row["NAME"].data<std::string>();
//              if (layer==6 && (sector==9 || sector==11)) {
//                if (temp=="+") {
//                  temp="";
//                  std::cout<<"Changed subsector + to null for "<<chname<<std::endl;
//                }
//              }

              std::string subsVal[5]={"--","-","","+","++"};
              char subsector=0;
              for (int i=0; i<5; i++) if (temp==subsVal[i]) subsector=i;
              temp=row["FEBZORNT"].data<std::string>();
              char febZOrnt=0;
              if (temp=="+z") febZOrnt=1;
              temp=row["FEBRADORNT"].data<std::string>();
              char febZRadOrnt=0;
              std::string febZRVal[3]={"","IN","OUT"};
              for (int i=0; i<3; i++) if (temp==febZRVal[i]) febZRadOrnt=i;
              temp=row["BARRELORENDCAP"].data<std::string>();
              char barrelOrEndcap=0;
              if (temp=="Barrel") barrelOrEndcap=1;
              thisFeb.theChamber=sector+100*subsector+1000*febZRadOrnt+5000*febZOrnt+10000*diskOrWheel+100000*layer+1000000*barrelOrEndcap;
            }
            delete query7;
            // Get Strips
            coral::IQuery* query8 = schema.newQuery();
            query8->addToTableList("CHAMBERSTRIP");
            query8->addToOutputList("CHAMBERSTRIP.CABLECHANNELNUM","CABLECHANNELNUM");
            query8->addToOutputList("CHAMBERSTRIP.CHAMBERSTRIPNUMBER","CHAMBERSTRIPNUM");
//            query8->addToOutputList("CHAMBERSTRIP.CMSSTRIPNUMBER","CMSSTRIPNUM");
            query8->addToOrderList("CABLECHANNELNUM");
            condition = "CHAMBERSTRIP.FC_FEBCONNECTORID="+IntToString(theFEB[iFEB].connectorId);
            query8->setCondition( condition, conditionData );
            coral::ICursor& cursor8 = query8->execute();
// NEW: do not store all the strip information as goes, only the minimum data needed to
// reproduce it later on
            int nstrips=0;
            int firstChamberStrip=0;
            int firstPin=0;
            int lastChamberStrip=0;
            int lastPin=0;
            while (cursor8.next()) {
              const coral::AttributeList& row = cursor8.currentRow();
              lastChamberStrip=row["CHAMBERSTRIPNUM"].data<int>();
              lastPin=row["CABLECHANNELNUM"].data<short>();
              if (nstrips==0) {
                firstChamberStrip=lastChamberStrip;
                firstPin=lastPin;
              }
              nstrips++;
            }
            delete query8;
            int algo = -1;
            if (firstPin == 1 && lastPin == nstrips) 
              { algo = 1; }
            else if (firstPin == 2 && lastPin == nstrips+1)
              { algo = 2; }
            else if (firstPin == 3 && lastPin == nstrips+2)
              { algo = 3; }
            else if (firstPin == 2 && lastPin == nstrips+2)
              { algo = 0;}
            else
              { std::cout<<" Unknown algo : "<<firstPin<<" "<<lastPin<<std::endl; }
            if ((lastPin-firstPin)*(lastChamberStrip-firstChamberStrip) < 0) algo=algo+4;
            thisFeb.theAlgo=algo+100*firstChamberStrip+10000*nstrips;
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
  std::cout << std::endl <<"Building RPC e-Map done!" << std::flush << std::endl << std::endl;
}

int popcon::RPCEMapSourceHandler::Compare2EMaps(Ref map1, RPCEMap* map2) {
  RPCReadOutMapping* oldmap1 = map1->convert();
  RPCReadOutMapping* oldmap2 = map2->convert();
  std::vector<const DccSpec *> dccs1 = oldmap1->dccList();
  std::vector<const DccSpec *> dccs2 = oldmap2->dccList();
  if(dccs1.size()!=dccs2.size()) {
    return 1;
  }
  std::pair<int,int> dccRange1 = oldmap1->dccNumberRange();
  std::pair<int,int> dccRange2 = oldmap2->dccNumberRange();
  if(dccRange1.first!=dccRange2.first) {
    return 1;
  }
  if(dccRange1.second!=dccRange2.second) {
    return 1;
  }
  typedef std::vector<const DccSpec *>::const_iterator IDCC;
  IDCC idcc2 = dccs2.begin();
  for (IDCC idcc1 = dccs1.begin(); idcc1 != dccs1.end(); idcc1++) {
    int dccNo = (**idcc1).id();
    std::string dccContents = (**idcc1).print(4);
    if ((**idcc2).id()!=dccNo) {
      return 1;
    }
    if ((**idcc2).print(4)!=dccContents) {
      return 1;
    }
    idcc2++;
  }
  return 0;
}
