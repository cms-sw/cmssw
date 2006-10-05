/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/

#ifndef L1RpcPacManagerH
#define L1RpcPacManagerH
/** \class L1RpcPacManager
 *
 * The singleton object of thise class stores all PACs of L1RPC trigger.
 * The tempalte type TPacTypeshould be derived from L1RpcPacBase,
 * and containe the constructor:
 * L1RpcPac(std::string patFilesDir, int tower, int logSector, int logSegment).
 * 3 configuration are suported:
 * ONE_PAC_PER_TOWER - the same PAC (set of patterns etc.) for every LogCone in a tower
 * _12_PACS_PER_TOWER - the same PAC in the same segment in every sector,
 * (i.e. 12 PACs in sector (one for LogicCone (segment)), all sectors are treat as one)
 * _144_PACS_PER_TOWER - one PAC for every LogicCone of given tower
 *
 * \author Karol Bunkowski (Warsaw)
 *
 */
//------------------------------------------------------------------------------
#include <string>
#include <vector>
#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"
#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"
#include "L1Trigger/RPCTrigger/src/RPCException.h"
#include <xercesc/util/PlatformUtils.hpp>

#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE

///Suported configurations
enum L1RpcPACsCntEnum {
  ONE_PAC_PER_TOWER = 1,
  _12_PACS_PER_TOWER = 12, //the same PAC in the same segment in every sector,
  _144_PACS_PER_TOWER =144,
  TB_TESTS
};


template<class TPacType> class L1RpcPacManager {
private:
  std::vector<std::vector<std::vector<TPacType*> > > PacTab; //!< PacTab[tower][logSector][LogSegment]

  int SectorsCnt; //!< Count of used differnt sectors.

  int SegmentCnt; //!< Count of used differnt segments.

  L1RpcPACsCntEnum PACsCnt; //Used configuration version.

public:
  ~L1RpcPacManager() {
    for (unsigned int tower = 0; tower < PacTab.size(); tower++)
      for (unsigned int logSector = 0; logSector < PacTab[tower].size(); logSector++) {
        for (unsigned int logSegment = 0; logSegment < PacTab[tower][logSector].size(); logSegment++) {
          TPacType* pac = PacTab[tower][logSector][logSegment];
          delete pac;
        }
      }
  }

  /** Creates the PACs. @param patFilesDirectory The directory where files defining
    * PACs are stored. The files should be named acording to special convencion.
    * @param  _PACsCnt The configuration version.
    * Should be caled once, before using PACs
    */
  void Init(std::string patFilesDirectory, L1RpcPACsCntEnum _PACsCnt)  {
    PACsCnt = _PACsCnt;
    if(PACsCnt == ONE_PAC_PER_TOWER) {
      SectorsCnt = 1;
      SegmentCnt = 1;
    }
    else if(PACsCnt == _12_PACS_PER_TOWER) {
      SectorsCnt = 1;
      SegmentCnt = 12;
    }
    else if(PACsCnt == _144_PACS_PER_TOWER) {
      SectorsCnt = 12;
      SegmentCnt = 12;
    }
    else if(PACsCnt == TB_TESTS) {
      SectorsCnt = 1;
      SegmentCnt = 4;
    }

    for (int tower = 0; tower < L1RpcConst::TOWER_COUNT; tower++) {
      PacTab.push_back(std::vector<std::vector<TPacType*> >() );
      for (int logSector = 0; logSector < SectorsCnt; logSector++) {
        PacTab[tower].push_back(std::vector<TPacType*>() );
        for (int logSegment = 0; logSegment < SegmentCnt; logSegment++) {
          TPacType* pac  = new TPacType(patFilesDirectory, tower, logSector, logSegment); 
          PacTab[tower][logSector].push_back(pac);                   
        }
      } 
    } 
    xercesc::XMLPlatformUtils::Terminate();
  };

  /** Returns the pointer to PAC for given LogCone defined by tower, logSector, logSegment.
    * Here you do not have to care, what configuration is curent used.
    * @param tower -16 : 16, @param logSector 0 : 11, @param logSegment 0 : 11.
    */
  //const
  TPacType* GetPac(int tower, int logSector, int logSegment) const {
    if (PacTab.size() <= (unsigned int) abs(tower) )
     throw L1RpcException("L1RpcPacManager::GetPac: given towerNum to big");
     // edm::LogError("RPCTrigger") << "L1RpcPacManager::GetPac: given towerNum to big" << std::endl;

    int curLogSector = logSector;
    int curlogSegment = logSegment;

    if(PACsCnt == ONE_PAC_PER_TOWER) {
      logSector = 0;
      logSegment = 0;
    }
    else if(PACsCnt == _12_PACS_PER_TOWER) {
      logSector = 0;
    }

    PacTab[abs(tower)][logSector][logSegment]->SetCurrentPosition(tower, curLogSector, curlogSegment);
    return  PacTab[abs(tower)][logSector][logSegment];
  };
  
  //const 
  TPacType* GetPac(const L1RpcConst::L1RpcConeCrdnts& coneCrdnts) const {
    return GetPac(coneCrdnts.Tower, coneCrdnts.LogSector, coneCrdnts.LogSegment);
  }
};

#endif
