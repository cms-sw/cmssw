#ifndef L1Trigger_RPCPacManager_h
#define L1Trigger_RPCPacManager_h
/** \class RPCPacManager
 *
 * The singleton object of thise class stores all PACs of L1RPC trigger.
 * The tempalte type TPacTypeshould be derived from RPCPacBase,
 * and containe the constructor:
 * RPCPacData(std::string patFilesDir, int m_tower, int logSector, int logSegment).
 * 3 configuration are suported:
 * ONE_PAC_PER_TOWER - the same m_PAC (set of patterns etc.) for every LogCone in a m_tower
 * _12_PACS_PER_TOWER - the same m_PAC in the same segment in every sector,
 * (i.e. 12 PACs in sector (one for LogicCone (segment)), all sectors are treat as one)
 * _144_PACS_PER_TOWER - one m_PAC for every LogicCone of given m_tower
 *
 * \author Karol Bunkowski (Warsaw)
 *
 */
//------------------------------------------------------------------------------
#include <string>
#include <vector>
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"

#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <cstdlib>
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE

///Suported configurations
// TODO: should be moved to L1RPConfig
enum L1RpcPACsCntEnum {
  ONE_PAC_PER_TOWER = 1,
  _12_PACS_PER_TOWER = 12, //the same m_PAC in the same segment in every sector,
  _144_PACS_PER_TOWER =144,
  TB_TESTS
};


template<class TPacType> class RPCPacManager {
public:
  ~RPCPacManager() {
    for (unsigned int m_tower = 0; m_tower < m_PacTab.size(); m_tower++)
      for (unsigned int logSector = 0; logSector < m_PacTab[m_tower].size(); logSector++) {
        for (unsigned int logSegment = 0; logSegment < m_PacTab[m_tower][logSector].size(); logSegment++) {
          TPacType* pac = m_PacTab[m_tower][logSector][logSegment];
          delete pac;
        }
      }
  }



  /** Gets data for PACs. @param patFilesDirectory The directory where files defining
    * PACs are stored. The files should be named acording to special convencion.
    * @param  _PACsCnt The configuration version.
    * Should be caled once, before using PACs
    */
  void init(std::string patFilesDirectory, L1RpcPACsCntEnum _PACsCnt)  {
    destroy(); 
    m_PACsCnt = _PACsCnt;
    if(m_PACsCnt == ONE_PAC_PER_TOWER) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 1;
    }
    else if(m_PACsCnt == _12_PACS_PER_TOWER) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 12;
    }
    else if(m_PACsCnt == _144_PACS_PER_TOWER) {
      m_SectorsCnt = 12;
      m_SegmentCnt = 12;
    }
    else if(m_PACsCnt == TB_TESTS) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 4;
    }

    for (int m_tower = 0; m_tower < RPCConst::m_TOWER_COUNT; m_tower++) {
      m_PacTab.push_back(std::vector<std::vector<TPacType*> >());
      for (int logSector = 0; logSector < m_SectorsCnt; logSector++) {
        m_PacTab[m_tower].push_back(std::vector<TPacType*>());
        for (int logSegment = 0; logSegment < m_SegmentCnt; logSegment++) {
          TPacType* pac  = new TPacType(patFilesDirectory, m_tower, logSector, logSegment); 
          m_PacTab[m_tower][logSector].push_back(pac);                   
        }
      } 
    } 
    xercesc::XMLPlatformUtils::Terminate();
  };

  
  void init(const L1RPCConfig *rpcconf)  {
    destroy(); 
    switch (rpcconf->getPPT()){
      case 1:
        m_PACsCnt = ONE_PAC_PER_TOWER;
        break;
      case 12:
        m_PACsCnt = _12_PACS_PER_TOWER;
        break;
      case 144:
        m_PACsCnt = _144_PACS_PER_TOWER;
        break;
    
    }
    
    if(m_PACsCnt == ONE_PAC_PER_TOWER) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 1;
    }
    else if(m_PACsCnt == _12_PACS_PER_TOWER) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 12;
    }
    else if(m_PACsCnt == _144_PACS_PER_TOWER) {
      m_SectorsCnt = 12;
      m_SegmentCnt = 12;
    }
    else if(m_PACsCnt == TB_TESTS) {
      m_SectorsCnt = 1;
      m_SegmentCnt = 4;
    }

    /*
    std::vector<std::vector<std::vector<RPCPattern::RPCPatVec> > > patvec;
    std::vector<std::vector<std::vector<RPCPattern::TQualityVec> > > qualvec;
    for (int tower = 0; tower < RPCConst::m_TOWER_COUNT; ++tower) {
      patvec.push_back(std::vector< std::vector< RPCPattern::RPCPatVec > >());
      qualvec.push_back(std::vector< std::vector< RPCPattern::TQualityVec > >());
      for (int logSector = 0; logSector < m_SectorsCnt; ++logSector) {
        patvec[tower].push_back(std::vector< RPCPattern::RPCPatVec >());
        qualvec[tower].push_back(std::vector< RPCPattern::TQualityVec >());
        for (int logSegment = 0; logSegment < m_SegmentCnt; ++logSegment) {
          patvec[tower][logSector].push_back(RPCPattern::RPCPatVec());
          qualvec[tower][logSector].push_back(RPCPattern::TQualityVec());
        }
      }
    }

    for (unsigned int ipat=0; ipat<rpcconf->m_pats.size(); ipat++)
      patvec[rpcconf->m_pats[ipat].getTower()][rpcconf->m_pats[ipat].getLogSector()][rpcconf->m_pats[ipat].getLogSegment()].push_back(rpcconf->m_pats[ipat]);
    for (unsigned int iqual=0; iqual<rpcconf->m_quals.size(); iqual++)
      qualvec[rpcconf->m_quals[iqual].m_tower][rpcconf->m_quals[iqual].m_logsector][rpcconf->m_quals[iqual].m_logsegment].push_back(rpcconf->m_quals[iqual]);
    */


    for (int tower = 0; tower < RPCConst::m_TOWER_COUNT; tower++) {
      m_PacTab.push_back(std::vector<std::vector<TPacType*> >());
      for (int logSector = 0; logSector < m_SectorsCnt; logSector++) {
        m_PacTab[tower].push_back(std::vector<TPacType*>());
        for (int logSegment = 0; logSegment < m_SegmentCnt; logSegment++) {
          /*L1RPCConfig* rpcconf1=new L1RPCConfig();
          rpcconf1->setPPT(rpcconf->getPPT());
          for (unsigned int ipat=0; ipat<patvec[tower][logSector][logSegment].size(); ipat++)
            rpcconf1->m_pats.push_back(patvec[tower][logSector][logSegment][ipat]);
          for (unsigned int iqual=0; iqual<qualvec[tower][logSector][logSegment].size(); iqual++)
            rpcconf1->m_quals.push_back(qualvec[tower][logSector][logSegment][iqual]);
          //TPacType* pac  = new TPacType(rpcconf1->m_pats,rpcconf1->m_quals);*/
          TPacType* pac  = new TPacType(rpcconf, tower, logSector, logSegment);
          m_PacTab[tower][logSector].push_back(pac);
        }
      } 
    } 
    xercesc::XMLPlatformUtils::Terminate();
  };
  
  /** Returns the pointer to m_PAC for given LogCone defined by m_tower, logSector, logSegment.
    * Here you do not have to care, what configuration is curent used.
    * @param m_tower -16 : 16, @param logSector 0 : 11, @param logSegment 0 : 11.
    */
  //const
  TPacType* getPac(int m_tower, int logSector, int logSegment) const {
    if (m_PacTab.size() <= (unsigned int) abs(m_tower))
     throw RPCException("RPCPacManager::getPac: given towerNum to big");
     // edm::LogError("RPCTrigger") << "RPCPacManager::getPac: given towerNum to big" << std::endl;

    //int curLogSector = logSector;
    //int curlogSegment = logSegment;

    if(m_PACsCnt == ONE_PAC_PER_TOWER) {
      logSector = 0;
      logSegment = 0;
    }
    else if(m_PACsCnt == _12_PACS_PER_TOWER) {
      logSector = 0;
    }

    //XXXX//m_PacTab[abs(m_tower)][logSector][logSegment]->setCurrentPosition(m_tower, curLogSector, curlogSegment);
    return  m_PacTab[std::abs(m_tower)][logSector][logSegment];
  };
  
  //const 
  TPacType* getPac(const RPCConst::l1RpcConeCrdnts& coneCrdnts) const {
    return getPac(coneCrdnts.m_Tower, coneCrdnts.m_LogSector, coneCrdnts.m_LogSegment);
  }
  
  private:
    std::vector<std::vector<std::vector<TPacType*> > > m_PacTab; //!< m_PacTab[m_tower][logSector][m_LogSegment]

    int m_SectorsCnt; //!< Count of used differnt sectors.

    int m_SegmentCnt; //!< Count of used differnt segments.

    L1RpcPACsCntEnum m_PACsCnt; //Used configuration version.

    void destroy(){
      for (size_t tower = 0; tower < m_PacTab.size() ; ++tower) {
        for (size_t logSector = 0; logSector < m_PacTab.at(tower).size(); logSector++) {
          for (size_t logSegment = 0; logSegment < m_PacTab.at(tower).at(logSector).size() ; logSegment++) {
            TPacType* pac = m_PacTab.at(tower).at(logSector).at(logSegment);
            delete pac;
          }
        }
      }
      m_PacTab.clear();
    }
};

#endif
