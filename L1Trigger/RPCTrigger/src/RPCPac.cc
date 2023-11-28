#include "L1Trigger/RPCTrigger/interface/RPCPac.h"
//#include "L1Trigger/RPCTrigger/interface/TEPatternsGroup.h"

RPCPac::RPCPac(const RPCPacData* pacData, int tower, int logSector, int logSegment)
    : RPCPacBase(tower, logSector, logSegment) {
  m_pacData = pacData;
}

/** 
 *
 * Runs Pattern Comparator algorithm for hits from the cone.
 * Calls the runTrackPatternsGroup() and runEnergeticPatternsGroups().
 * @return found track candidate (empty if hits does not fit to eny pattern)
 *
 */
RPCPacMuon RPCPac::run(const RPCLogCone& cone) const {  //symualcja

  RPCPacMuon bestMuon;
  //track
  if (!m_pacData->m_TrackPatternsGroup.m_PatternsItVec.empty())
    bestMuon = runTrackPatternsGroup(cone);

  //energetic
  if (!m_pacData->m_EnergeticPatternsGroupList.empty()) {
    RPCPacMuon bufMuon = runEnergeticPatternsGroups(cone);
    if (bufMuon > bestMuon)
      bestMuon = bufMuon;
  }

  bestMuon.setConeCrdnts(m_CurrConeCrdnts);

  //bestMuon.setConeCrdnts(cone.);
  bestMuon.setLogConeIdx(cone.getIdx());
  /*
  int refStripNum = m_pacData->getPattern(bestMuon.getPatternNum())
                             .getStripFrom(RPCConst::m_REF_PLANE[abs(m_CurrConeCrdnts.m_Tower)])
                              +  m_CurrConeCrdnts.m_LogSector * 96
                              + m_CurrConeCrdnts.m_LogSegment * 8;
  bestMuon.setRefStripNum(refStripNum);*/
  return bestMuon;
}

RPCPacMuon RPCPac::runTrackPatternsGroup(const RPCLogCone& cone) const {
  RPCPacMuon bestMuon;

  for (unsigned int vecNum = 0; vecNum < m_pacData->m_TrackPatternsGroup.m_PatternsItVec.size(); vecNum++) {
    RPCMuon::TDigiLinkVec digiIdx;
    unsigned short firedPlanes = 0;
    int firedPlanesCount = 0;
    unsigned short one = 1;
    const RPCPattern& pattern = *(m_pacData->m_TrackPatternsGroup.m_PatternsItVec[vecNum]);
    for (int logPlane = RPCConst::m_FIRST_PLANE;
         logPlane < RPCConst::m_USED_PLANES_COUNT[std::abs(m_ConeCrdnts.m_Tower)];
         logPlane++) {
      if (pattern.getStripFrom(logPlane) == RPCConst::m_NOT_CONECTED) {
        //firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        continue;
      }
      int fromBit = pattern.getStripFrom(logPlane);
      int toBit = pattern.getStripTo(logPlane);
      for (int bitNumber = fromBit; bitNumber < toBit; bitNumber++) {
        if (cone.getLogStripState(logPlane, bitNumber) == true) {
          firedPlanes = firedPlanes | one;
          firedPlanesCount++;
          std::vector<int> dIVec = cone.getLogStripDigisIdxs(logPlane, bitNumber);
          if (!dIVec.empty())
            digiIdx.push_back(RPCMuon::TDigiLink(logPlane, *dIVec.begin()));

          break;
        }
      }

      if ((RPCConst::m_USED_PLANES_COUNT[std::abs(m_ConeCrdnts.m_Tower)] - logPlane) == 3)
        if (firedPlanesCount == 0)
          break;

      one = one << 1;
    }

    if (firedPlanesCount >= 3) {
      short quality = m_pacData->m_QualityTabsVec[pattern.getQualityTabNumber()][firedPlanes];
      if (quality != -1) {
        if (quality >= bestMuon.getQuality()) {
          RPCPacMuon bufMuon(pattern, quality, firedPlanes);
          if (bufMuon > bestMuon) {
            bestMuon = bufMuon;
            bestMuon.setDigiIdxVec(digiIdx);
          }
        }
      }
    }
  }
  return bestMuon;
}

RPCPacMuon RPCPac::runEnergeticPatternsGroups(const RPCLogCone& cone) const {
  RPCPacMuon bestMuon;
  unsigned short firedPlanes = 0;
  // int firedPlanesCount = 0;
  RPCPacData::TEPatternsGroupList::const_iterator iEGroup = m_pacData->m_EnergeticPatternsGroupList.begin();
  for (; iEGroup != m_pacData->m_EnergeticPatternsGroupList.end(); iEGroup++) {
    firedPlanes = 0;
    // firedPlanesCount = 0;
    unsigned short one = 1;
    for (int logPlane = RPCConst::m_FIRST_PLANE;
         logPlane < RPCConst::m_USED_PLANES_COUNT[std::abs(m_ConeCrdnts.m_Tower)];
         logPlane++) {  //or po paskach ze stozka

      if (cone.getHitsCnt(logPlane) > 0) {
        RPCLogCone::TLogPlane lp = cone.getLogPlane(logPlane);
        RPCLogCone::TLogPlane::const_iterator itLP = lp.begin();
        RPCLogCone::TLogPlane::const_iterator itLPE = lp.end();
        for (; itLP != itLPE; ++itLP) {
          int strip = itLP->first;
          if (iEGroup->m_GroupShape.getLogStripState(logPlane, strip)) {
            firedPlanes = firedPlanes | one;
            // firedPlanesCount++;
            break;
          }
        }
      }
      /*
          for(unsigned int bitNum = 0;
               bitNum <
               RPCConst::
               m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)][logPlane];
               bitNum++)
            {
              if(iEGroup->m_GroupShape.getLogStripState(logPlane, bitNum)
                  && cone.getLogStripState(logPlane, bitNum))
                {
                  firedPlanes = firedPlanes | one;
                  firedPlanesCount++;
                  break;
                }
            }
           */
      one = one << 1;
    }

    short quality = m_pacData->m_QualityTabsVec[iEGroup->m_QualityTabNumber][firedPlanes];
    if (quality == -1)
      continue;

    RPCPacMuon bufMuon;
    for (unsigned int vecNum = 0; vecNum < iEGroup->m_PatternsItVec.size(); vecNum++) {
      RPCMuon::TDigiLinkVec digiIdx;
      const RPCPattern::RPCPatVec::const_iterator patternIt = iEGroup->m_PatternsItVec[vecNum];
      const RPCPattern& pattern = *patternIt;
      bool wasHit = false;
      unsigned short one1 = 1;
      for (int logPlane = RPCConst::m_FIRST_PLANE;
           logPlane < RPCConst::m_USED_PLANES_COUNT[std::abs(m_ConeCrdnts.m_Tower)];
           logPlane++, one1 = one1 << 1) {
        if (pattern.getStripFrom(logPlane) == RPCConst::m_NOT_CONECTED) {
          //          firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          continue;
        }
        if ((firedPlanes & one1) != 0) {
          int fromBit = pattern.getStripFrom(logPlane);
          int toBit = pattern.getStripTo(logPlane);
          wasHit = false;
          for (int bitNumber = fromBit; bitNumber < toBit; bitNumber++) {
            wasHit = wasHit || cone.getLogStripState(logPlane, bitNumber);
            if (wasHit) {  // no sense to check more
              std::vector<int> dIVec = cone.getLogStripDigisIdxs(logPlane, bitNumber);
              if (!dIVec.empty())
                digiIdx.push_back(RPCMuon::TDigiLink(logPlane, *dIVec.begin()));
              break;
            }
          }

          if (!wasHit) {
            break;
          }
        }
      }
      if (wasHit) {
        bufMuon.setAll(pattern, quality, firedPlanes);
        bufMuon.setDigiIdxVec(digiIdx);
        break;  //if one pattern fits, thers no point to check other patterns from group
      }
    }  //end of patterns loop
    if (bufMuon > bestMuon) {
      bestMuon = bufMuon;
    }
    //if(bestMuon.getQuality() == m_pacData->m_MaxQuality)
    //  return bestMuon;
  }  //end of EGroup loop
  return bestMuon;
}
