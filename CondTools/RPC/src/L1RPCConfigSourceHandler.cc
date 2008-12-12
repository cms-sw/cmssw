#include "CondTools/RPC/interface/L1RPCConfigSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

popcon::L1RPCConfigSourceHandler::L1RPCConfigSourceHandler(const edm::ParameterSet& ps):
  m_name(ps.getUntrackedParameter<std::string>("name","L1RPCConfigSourceHandler")),
  m_validate(ps.getUntrackedParameter<int>("Validate",0)),
  m_ppt(ps.getUntrackedParameter<int>("PACsPerTower")),
  m_dataDir(ps.getUntrackedParameter<std::string>("filedir","L1Trigger/RPCTrigger/data/Eff90PPT12/"))
{
	//	std::cout << "L1RPCConfigSourceHandler: L1RPCConfigSourceHandler constructor" << std::endl;
  edm::FileInPath fp(m_dataDir+"pacPat_t0sc0sg0.xml");
  std::string patternsDirNameUnstriped = fp.fullPath();
  m_patternsDir = patternsDirNameUnstriped.substr(0,patternsDirNameUnstriped.find_last_of("/")+1);
}

popcon::L1RPCConfigSourceHandler::~L1RPCConfigSourceHandler()
{
}

void popcon::L1RPCConfigSourceHandler::getNewObjects()
{

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

//  std::cout << "L1RPCConfigSourceHandler: L1RPCConfigSourceHandler::getNewObjects begins\n";
//  std::cerr << "------- " << m_name 
//	     << " - > getNewObjects" << std::endl;
//  std::cerr<<"Got offlineInfo, tag: "<<std::endl;
//  std::cerr << tagInfo().name << " , last object valid from " 
//	    << tagInfo().lastInterval.first << " to "
//            << tagInfo().lastInterval.second << " , token is "
//            << tagInfo().token << " and this is the payload "
//            << tagInfo().lastPayloadToken << std::endl;

// first check what is already there in offline DB
	const L1RPCConfig* patterns_prev = 0; //just to avoid warnings

        if(m_validate==1) {
          std::cout<<" Validation was requested, so will check present contents"<<std::endl;
        }

        readConfig();

        cond::Time_t snc=mydbservice->currentTime();

// look for recent changes
        int difference=1;
        if (m_validate==1) difference=Compare2Configs(patterns_prev,patterns);
        if (!difference) cout<<"No changes - will not write anything!!!"<<endl;
        if (difference==1) {
          cout<<"Will write new object to offline DB!!!"<<endl;
          m_to_transfer.push_back(std::make_pair((L1RPCConfig*)patterns,snc));
        }

	std::cout << "L1RPCConfigSourceHandler: L1RPCConfigSourceHandler::getNewObjects ends\n";
}

void popcon::L1RPCConfigSourceHandler::readConfig()
{
  patterns =  new L1RPCConfig();
  patterns->setPPT(m_ppt);
// parse and insert patterns
  int scCnt = 0, sgCnt = 0;
  if(m_ppt == 1) {
    scCnt = 1;
    sgCnt = 1;
  }
  else if(m_ppt == 12) {
    scCnt = 1;
    sgCnt = 12;
  }
  else if(m_ppt == 144) {
    scCnt = 12;
    sgCnt = 12;
  }
  else {
    throw cms::Exception("BadConfig") << "Bad number of ppt requested: " << m_ppt << "\n";
  }

  for (int tower = 0; tower < RPCConst::m_TOWER_COUNT; ++tower) {
    for (int logSector = 0; logSector < scCnt; ++logSector) {
      for (int logSegment = 0; logSegment < sgCnt; ++logSegment) {

        std::stringstream fname;
        fname << m_patternsDir
              << "pacPat_t" << tower 
              << "sc" << logSector 
	      << "sg" <<logSegment 
              << ".xml";

        cout<<"Parsing "<<fname.str()<<flush;
        RPCPatternsParser parser;
        parser.parse(fname.str());
        cout<<" - done "<<endl;

        RPCPattern::RPCPatVec npats = parser.getPatternsVec(tower, logSector, logSegment);
           
        for (unsigned int ip=0; ip<npats.size(); ip++) {
          npats[ip].setCoords(tower,logSector,logSegment);
          patterns->m_pats.push_back(npats[ip]);
        }

        RPCPattern::TQualityVec nquals = parser.getQualityVec(); 
        for (unsigned int iq=0; iq<nquals.size(); iq++) {
          nquals[iq].m_tower=tower;
          nquals[iq].m_logsector=logSector;
          nquals[iq].m_logsegment=logSegment;
          patterns->m_quals.push_back(nquals[iq]);
        }
	    
      } // segments
    } // sectors
  } // towers
}

int popcon::L1RPCConfigSourceHandler::Compare2Configs(const L1RPCConfig* pat1, L1RPCConfig* pat2) {
  std::cout<<" L1RPCConfigSourceHandler::Compare2Configs: sorry, option not available - different configs assumed"<<std::endl;
  return 1;
}
