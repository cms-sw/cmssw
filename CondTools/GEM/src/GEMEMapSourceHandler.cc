#include "CondTools/GEM/interface/GEMEMapSourceHandler.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include <TString.h>

#include <fstream>
#include <cstdlib>
#include <vector>

#include <sstream>

#include <DataFormats/MuonDetId/interface/GEMDetId.h>

popcon::GEMEMapSourceHandler::GEMEMapSourceHandler( const edm::ParameterSet& ps ):
  m_name( ps.getUntrackedParameter<std::string>( "name", "GEMEMapSourceHandler" ) ),
  m_dummy( ps.getUntrackedParameter<int>( "WriteDummy", 0 ) ),
  m_validate( ps.getUntrackedParameter<int>( "Validate", 1 ) ),
  m_connect( ps.getParameter<std::string>( "connect" ) ),
  m_connectionPset( ps.getParameter<edm::ParameterSet>( "DBParameters" ) )
{
}

popcon::GEMEMapSourceHandler::~GEMEMapSourceHandler()
{
}

void popcon::GEMEMapSourceHandler::getNewObjects()
{
  
  edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                         << "BEGIN" << std::endl;
  
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  // first check what is already there in offline DB
  Ref payload;
  if(m_validate==1 && tagInfo().size>0) {
    edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                           << "Validation was requested, so will check present contents\n"
                                           << "Destination Tag Info: name " << tagInfo().name
                                           << ", size " << tagInfo().size 
                                           << ", last object valid since " << tagInfo().lastInterval.first
                                           << ", hash " << tagInfo().lastPayloadToken << std::endl;  
    payload = lastPayload();
  }
  
  // now construct new cabling map from online DB
  // FIXME: use boost::ptime
  time_t rawtime;
  time(&rawtime); //time since January 1, 1970
  tm * ptm = gmtime(&rawtime);//GMT time
  char buffer[20];
  strftime(buffer,20,"%d/%m/%Y_%H:%M:%S",ptm);
  std::string eMap_version( buffer );
  edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                         << "GEM eMap version: " << eMap_version << std::endl;
  eMap =  new GEMEMap(eMap_version);
  if (m_dummy==0) {
    ConnectOnlineDB( m_connect, m_connectionPset );
    readGEMEMap();
    DisconnectOnlineDB();
  }



  /*
  // additional work
  //FIXME: you need a coral::ISessionProxy for accessing GEM data
  //not a cond::Session designed for condition access.
  //If so, the cond::Session data member is not needed.
  if (m_dummy==0) {
    //ConnectOnlineDB( m_connect, m_connectionPset );
    //DisconnectOnlineDB();
  }
 
  //FIXME: use edm::FileInPath
  //TODO: data files go in a separate git repo, if needed
  std::string baseCMS = std::string(getenv("CMSSW_BASE"))+std::string("/src/CondTools/GEM/data/");  
  std::vector<std::string> mapfiles;

  //TString WhichConf = "CMSGE1/1";

  mapfiles.push_back("vfat_position.csv");
  
  for (unsigned int ifm=0;ifm<mapfiles.size();ifm++){  
    GEMEMap::GEMVFatMaptype vmtype;
    std::string filename(baseCMS+mapfiles[ifm]);
    edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
					   <<"Opening CSV file "<< filename << std::endl;
    vmtype.VFATmapTypeId=ifm+1;//this is 1 and 2 if there are two input files
    std::ifstream maptype(filename.c_str());
    std::string buf("");
    
    
    std::string field, line, tmp_sec;
    while(std::getline(maptype, line)){
      //mapping v1:      VFAT_POSN	Z	IETA	IPHI	DEPTH	Detector Strip Number	VFAT channel Number	Px Connector Pin #
      //mapping v2:             SUBDET   SECTOR	         TYPE	ZPOSN	IETA   IPHI   DEPTH   VFAT_POSN	  DET_STRIP   VFAT_CHAN   CONN_PIN
      //mapping CS (7 Nov 2016) SUBDET   TSCOL   TSROW   TYPE   ZPOSN   IETA   IPHI   DEPTH   VFAT_POSN   DET_STRIP   VFAT_CHAN   CONN_PIN
      
      int vfat_pos, z_dir, ieta, iphi, dep, str_num, vfat_chn_num, sec;
      uint16_t vfat_add;
      std::stringstream ssline(line);   
      getline( ssline, field, ',' );
      tmp_sec = "";
      tmp_sec.push_back(field[4]);
      tmp_sec.push_back(field[5]);
      std::cout << tmp_sec << std::endl;
      std::stringstream Sec(tmp_sec);
      getline( ssline, field, ',' );
      std::stringstream Z_dir(field);
      getline( ssline, field, ',' );
      std::stringstream Ieta(field);
      getline( ssline, field, ',' );
      std::stringstream Iphi(field);
      getline( ssline, field, ',' );
      std::stringstream Dep(field);
      getline( ssline, field, ',' );
      std::stringstream Vfat_pos(field);
      getline( ssline, field, ',' );
      std::stringstream Str_num(field);
      getline( ssline, field, ',' );
      std::stringstream Vfat_chn_num(field);
      getline( ssline, field, ',' );
      char* chr = strdup(field.c_str());
      std::cout << chr << std::endl;
      vfat_add = strtol(chr,NULL,16);
      Sec >> sec;Z_dir >> z_dir; Ieta >> ieta; Iphi >> iphi; Dep >> dep; Vfat_pos >> vfat_pos; Str_num >> str_num; Vfat_chn_num >> vfat_chn_num; //(uint16_t)chr >> vfat_add;
     
      LogDebug( "GEMMapSourceHandler" ) << ", z_direction="<< z_dir
					<< ", ieta="<< ieta
					<< ", iphi="<< iphi
					<< ", depth="<< dep
					<< ", vfat position="<< vfat_pos
					<< ", strip no.=" << str_num
					<< ", vfat channel no.="<< vfat_chn_num
					<< std::endl;
      
      std::cout<<" Sector="<<sec<<" z_direction="<<z_dir<<" ieta="<<ieta<<" iphi="<<iphi<<" depth="<<dep<<" vfat position="<<vfat_pos<<" strip no.="<<str_num<<" vfat channel no.="<<vfat_chn_num<<" vfat address = " << vfat_add <<std::endl;
      //GEMDetId id(z_dir, 1, 1, dep, sec, ieta);
      //std::cout  << id.rawId() << std::endl;    
      vmtype.sec.push_back(sec);
      vmtype.vfat_position.push_back(vfat_pos);
      vmtype.z_direction.push_back(z_dir);
      vmtype.iEta.push_back(ieta);
      vmtype.iPhi.push_back(iphi);
      vmtype.depth.push_back(dep);
      vmtype.strip_number.push_back(str_num);
      vmtype.vfat_chnnel_number.push_back(vfat_chn_num);
      vmtype.vfatId.push_back(vfat_add);
    }
      eMap->theVFatMaptype.push_back(vmtype); 
  }
  */   
  cond::Time_t snc = mydbservice->currentTime();  
  // look for recent changes
  int difference=1;
  if (difference==1) {
    m_to_transfer.push_back(std::make_pair((GEMEMap*)eMap,snc));
    edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                           << "Emap size: " << eMap->theVFatMaptype.size()
                                           << ", payloads to transfer: " << m_to_transfer.size() << std::endl;
  }
  edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                         << "END." << std::endl;
}

// // additional work (I added these two functions: ConnectOnlineDB and DisconnectOnlineDB)
void popcon::GEMEMapSourceHandler::ConnectOnlineDB( const std::string& connect, const edm::ParameterSet& connectionPset )
{
  cond::persistency::ConnectionPool connection;
  edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                         << "GEMEMapConfigSourceHandler: connecting to " << connect << "..." << std::endl;
  connection.setParameters( connectionPset );
  connection.configure();
  session = connection.createSession( connect,true );
  edm::LogInfo( "GEMEMapSourceHandler" ) << "[" << "GEMEMapSourceHandler::" << __func__ << "]:" << m_name << ": "
                                         << "Done." << std::endl;
}

void popcon::GEMEMapSourceHandler::DisconnectOnlineDB()
{
  session.close();
}

void popcon::GEMEMapSourceHandler::readGEMEMap()
{
  session.transaction().start( true );
  coral::ISchema& schema = session.nominalSchema();
  std::string condition="";
  coral::AttributeList conditionData;

  std::cout << std::endl <<"GEMEMapSourceHandler: start to build GEM e-Map..." << std::flush << std::endl << std::endl;

  coral::IQuery* query1 = schema.newQuery();
  /*select c.SECTOR, c.ZPOSN,c.IETA,c.IPHI,c.DEPTH,c.VFAT_POSN,c.DET_STRIP,c.VFAT_CHAN, b.VFAT_ADDRESS  from gem_omds.gem_vfat_channels c inner join  gem_omds.gem_sprchmbr_opthyb_vfats_v b on c.vfat_posn = b.vfat_posn and c.sector= b.sector and c.depth = b.depth*/

  query1->addToTableList( "CMS_GEM_MUON_COND.gem_vfat_channels" );
  query1->addToTableList( "CMS_GEM_MUON_VIEW.gem_sprchmbr_opthyb_vfats_view" );
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.SECTOR", "SECTOR");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.ZPOSN", "ZPOSN");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.IETA", "IETA");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.IPHI", "IPHI");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.DEPTH", "DEPTH");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.VFAT_POSN", "VFAT_POSN");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.DET_STRIP", "DET_STRIP");
  query1->addToOutputList("CMS_GEM_MUON_COND.gem_vfat_channels.VFAT_CHAN", "VFAT_CHAN");
  query1->addToOutputList("CMS_GEM_MUON_VIEW.gem_sprchmbr_opthyb_vfats_view.VFAT_ADDRESS", "VFAT_ADDRESS");

  condition = "CMS_GEM_MUON_COND.gem_vfat_channels.IETA>0";

  query1->setCondition( condition, conditionData );


  coral::ICursor& cursor1 = query1->execute();
  std::cout<<"OK"<<std::endl;
  GEMEMap::GEMVFatMaptype vmtype;
  std::pair<int,int> tmp_tbl;
  std::vector< std::pair<int,int> > theDAQ;
  while ( cursor1.next() ) {
    const coral::AttributeList& row = cursor1.currentRow();
    vmtype.iEta.push_back( row["IETA"].data<int>() );
    vmtype.iPhi.push_back( row["IPHI"].data<int>() );
    vmtype.depth.push_back( row["DEPTH"].data<int>()  );
    vmtype.vfat_position.push_back( row["VFAT_POSN"].data<int>()  );
    vmtype.strip_number.push_back( row["DET_STRIP"].data<int>() );
    vmtype.vfat_chnnel_number.push_back( row["VFAT_CHAN"].data<int>()  );
    vmtype.z_direction.push_back( row["ZPOSN"].data<int>() );
    vmtype.vfatId.push_back( row["VFAT_ADDRESS"].data<uint16_t>()  );
    std::string a = row["VFAT_ADDRESS"].data<std::string>();
  
    int sector;
    std::string b=a.substr(a.find("GEM")+3,a.npos);
    //std::cout <<" a  "<<a<<" b "<<b<<std::endl;
    std::stringstream os;
    os<<b;
    os>>sector;
    //std::cout <<" sector "<<sector<<std::endl;

 
    vmtype.sec.push_back(sector);
  }
  eMap->theVFatMaptype.push_back(vmtype); 
  delete query1;

}
