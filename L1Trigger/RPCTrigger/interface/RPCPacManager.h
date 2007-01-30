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
#include <xercesc/util/PlatformUtils.hpp>
#include <cstdlib>
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE

///Suported configurations
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

  /** Creates the PACs. @param patFilesDirectory The directory where files defining
    * PACs are stored. The files should be named acording to special convencion.
    * @param  _PACsCnt The configuration version.
    * Should be caled once, before using PACs
    */
  void init(std::string patFilesDirectory, L1RpcPACsCntEnum _PACsCnt)  {
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
};

#endif
