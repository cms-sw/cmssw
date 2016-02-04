//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/interface/RPCHalfSorter.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include <algorithm>

#include "L1Trigger/RPCTrigger/interface/MuonsGrabber.h"
//---------------------------------------------------------------------------
#include <set>
using namespace std;

/**
 *
 * Defualt constructor
 *
*/
RPCHalfSorter::RPCHalfSorter(RPCTriggerConfiguration* triggerConfig) {
  
  m_TrigCnfg = triggerConfig;
  
  m_GBOutputMuons.assign(2, L1RpcTBMuonsVec());
  
}
/** 
 *
 * Runs GB algorithm for half of the detector - 6 TC (sectors).
 * @return 4 munons from barrel (m_GBOutputMuons[0]),
 * and 4 from endcaps (m_GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 RPCHalfSorter::runHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  //1+6+1 TC, wazne sa tylko te 6 w srodku
  for(unsigned int iTC = 0; iTC < tcsMuonsVec2.size()-1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++) {
      if(tcsMuonsVec2[iTC][iMu].getCode() == 0)
        continue;
      if(tcsMuonsVec2[iTC][iMu].gBDataKilledLast()) {
        for(unsigned int iMuN = 0; iMuN < tcsMuonsVec2[iTC+1].size(); iMuN++) {
          if(tcsMuonsVec2[iTC+1][iMuN].getCode() == 0)
            continue;
          if(tcsMuonsVec2[iTC+1][iMuN].gBDataKilledFirst())
          {
            int eta1 = tcsMuonsVec2[iTC][iMu].getEtaAddr();
            int eta2 = tcsMuonsVec2[iTC+1][iMuN].getEtaAddr();
            if ( eta1 > 16 ) eta1 = - ( (~eta1 & 63) + 1);
            if ( eta2 > 16 ) eta2 = - ( (~eta2 & 63) + 1);
            if(abs(eta1 - eta2) <= 1) 
            {
              if(tcsMuonsVec2[iTC][iMu].getCode() <= tcsMuonsVec2[iTC+1][iMuN].getCode()) 
              {
                if(tcsMuonsVec2[iTC][iMu].getSegmentAddr() == RPCConst::m_SEGMENTS_IN_SECTOR_CNT-1) {
                  tcsMuonsVec2[iTC][iMu].kill();
                }
              }    
              else 
              {
                tcsMuonsVec2[iTC+1][iMuN].kill();
              }
            }
          }
        }
      }
    }
  }

  L1RpcTBMuonsVec outputBarrelMuons;
  L1RpcTBMuonsVec outputEndcapMuons;
  
  for(unsigned int iTC = 1; iTC < tcsMuonsVec2.size()-1; iTC++)
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      if(tcsMuonsVec2[iTC][iMu].isLive()){
       // if(abs(16 - tcsMuonsVec2[iTC][iMu].getEtaAddr()) <=7){
       // etaAddr should be encoded here in 6 bits with 2compl. 
        if( tcsMuonsVec2[iTC][iMu].getEtaAddr() >= 57 ||
            tcsMuonsVec2[iTC][iMu].getEtaAddr() <= 7  )
        {
          outputBarrelMuons.push_back(tcsMuonsVec2[iTC][iMu]);
        }
        else{
          outputEndcapMuons.push_back(tcsMuonsVec2[iTC][iMu]);
        }
      }
      
  sort(outputBarrelMuons.begin(), outputBarrelMuons.end(), RPCTBMuon::TMuonMore());
  sort(outputEndcapMuons.begin(), outputEndcapMuons.end(), RPCTBMuon::TMuonMore());

  while(outputBarrelMuons.size() < RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.push_back(RPCTBMuon());
  while(outputBarrelMuons.size() > RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.pop_back();
  
  while(outputEndcapMuons.size() < RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.push_back(RPCTBMuon());
  while(outputEndcapMuons.size() > RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.pop_back();

  m_GBOutputMuons[0].insert(m_GBOutputMuons[0].end(),
                            outputBarrelMuons.begin(),
                            outputBarrelMuons.end());
  m_GBOutputMuons[1].insert(m_GBOutputMuons[1].end(),
                            outputEndcapMuons.begin(),
                            outputEndcapMuons.end());
  return m_GBOutputMuons;
}
void RPCHalfSorter::maskHSBInput(L1RpcTBMuonsVec & newVec, int mask) {

    if ( mask < 0 || mask > 3) {
         throw cms::Exception("RPCHalfSorter::maskHSBInput") << " hsbMask has wrong value - " << mask <<" \n";
    }

    if ( mask == 1) { // leave the best muons
      newVec.at(2) = RPCTBMuon();
      newVec.at(3) = RPCTBMuon();
    } else if (mask == 2){
      newVec.at(0) = RPCTBMuon();
      newVec.at(1) = RPCTBMuon();
    } else {
      newVec.at(0) = RPCTBMuon();
      newVec.at(1) = RPCTBMuon();
      newVec.at(2) = RPCTBMuon();
      newVec.at(3) = RPCTBMuon();
    }

}



/** 
 * Runs runHalf() for 2 detecors parts.
 * Converts m_tower number (eta addr) from continous (0 - 32, m_tower 0 = 16)
 * to 2'complement.
 * @return 4 munons from barrel (m_GBOutputMuons[0]),
 * and 4 from endcaps (m_GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 RPCHalfSorter::run(L1RpcTBMuonsVec2 &tcsMuonsVec2, edm::ESHandle<L1RPCHsbConfig> hsbConf) {
  
  m_GBOutputMuons[0].clear();
  m_GBOutputMuons[1].clear();

  L1RpcTBMuonsVec2 firstHalfTcsMuonsVec2;

  if ( tcsMuonsVec2[m_TrigCnfg->getTCsCnt()-1].size()==0 || hsbConf->getHsbMask(0,0) == 3 ) {
    firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[m_TrigCnfg->getTCsCnt()-1]); //TC=11 (last one)
  } else {
    L1RpcTBMuonsVec newVec = tcsMuonsVec2[m_TrigCnfg->getTCsCnt()-1];
    maskHSBInput(newVec, hsbConf->getHsbMask(0,0));
    firstHalfTcsMuonsVec2.push_back(newVec); 
  }

  // update sectorAddr. Dont update sectorAddr of last tc (it will be done in other half)
  int secAddr = 1;  
  //                                         <6+1
  for(int iTC = 0; iTC < m_TrigCnfg->getTCsCnt()/2 +1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++){
      if ( secAddr != 0 && secAddr != 7  ){ 
        tcsMuonsVec2[iTC][iMu].setSectorAddr(secAddr); // |
                                                   // iTC=0 - firstTrigger crate (no=1) 
                                                   //       - in hw it has sectorAddr=1
        //tcsMuonsVec2[iTC][iMu].setGBData(0);       // gbData is used nowhere from now, we 
                                                   //      want to act same as hw
      }
    } // iter. over muons end
    ++secAddr; // Next trigger crate. Update the address

    if ( tcsMuonsVec2[iTC].size()==0 || hsbConf->getHsbMask(0, iTC+1) == 3 ) {
      firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
    } else {
      L1RpcTBMuonsVec newVec = tcsMuonsVec2[iTC];
      maskHSBInput(newVec, hsbConf->getHsbMask(0, iTC+1));
      firstHalfTcsMuonsVec2.push_back(newVec); 
    }

  }

  runHalf(firstHalfTcsMuonsVec2);

  unsigned int fhBMuons = m_GBOutputMuons[0].size(); // Number of first half barrel muons
  unsigned int fhEMuons = m_GBOutputMuons[1].size(); // Number of first half endcap muons
  
  L1RpcTBMuonsVec2 secondHalfTcsMuonsVec2;
  secAddr = 0; 
  //        5                                           <12
  for(int iTC = m_TrigCnfg->getTCsCnt()/2-1; iTC < m_TrigCnfg->getTCsCnt(); iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++){
      if ( secAddr != 0 && secAddr != 7  ){ 
        tcsMuonsVec2[iTC][iMu].setSectorAddr(secAddr);
        //tcsMuonsVec2[iTC][iMu].setGBData(0);       // gbData is used nowhere from now, we 
                                                   //      want to act same as hw
      }
    }
    ++secAddr;
    if ( tcsMuonsVec2[iTC].size()==0 || hsbConf->getHsbMask(1, iTC-5) == 3 ) {
      secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
    } else {
      L1RpcTBMuonsVec newVec = tcsMuonsVec2[iTC];
      maskHSBInput(newVec, hsbConf->getHsbMask(1, iTC-5));
      secondHalfTcsMuonsVec2.push_back(newVec); 
    }
  }

  //secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[0]);
  if ( tcsMuonsVec2[0].size()==0 || hsbConf->getHsbMask(1 , 7) == 3 ) {
    secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[0]);
  } else {
    L1RpcTBMuonsVec newVec = tcsMuonsVec2[0];
    maskHSBInput(newVec, hsbConf->getHsbMask(1, 7));
    secondHalfTcsMuonsVec2.push_back(newVec); 
  }

  runHalf(secondHalfTcsMuonsVec2);
  // Debug
    for (unsigned  int region = 0; region < m_GBOutputMuons.size(); ++region){ // region: 0- barrel,1-endcaps
        for (unsigned  int i = 0; i < m_GBOutputMuons[region].size(); ++i)
	{
	
	  unsigned int halfNum = 0; // Number of halfsorter: 0 - first half, 1 - second half
	  int iMod=0;

	  // After second call of runHalf muons are written at the end of m_GBOutputMuons[0,1] vector
	  // not needed - fhBMuons ==4 (always)
  	  if ( (region == 0 && i >= fhBMuons ) ||
	       (region == 1 && i >= fhEMuons ) )
	  {
	    halfNum = 1;
	    iMod=4;
	  }
          // Print out 
          if (m_TrigCnfg->getDebugLevel()==1){
#ifndef _STAND_ALONE
           // LogDebug("RPCHwDebug")<<"GB 3"<< region <<halfNum  
	   //     << " " << i - iMod << " "
           //     << m_GBOutputMuons[region][i].printDebugInfo(m_TrigCnfg->getDebugLevel());
           //MuonsGrabber::Instance().writeDataForRelativeBX(iBx);  
           MuonsGrabber::Instance().addMuon(m_GBOutputMuons[region][i], 3, region, halfNum, i - iMod);  

#else
            std::cout <<"GB 3" << region<< halfNum
	        << " " << i - iMod << " "
                << m_GBOutputMuons[region][i].printDebugInfo(m_TrigCnfg->getDebugLevel())
                << std::endl;
#endif 
          }
          // Re-number the phi addr. This should be done by fs, temporary fix (doesnt change the logic)
          int segment = m_GBOutputMuons[region][i].getSegmentAddr();
          int sector = m_GBOutputMuons[region][i].getSectorAddr()-1+6*halfNum;
          int pt = m_GBOutputMuons[region][i].getPtCode();
          if (pt != 0){// dont touch empty muons
            m_GBOutputMuons[region][i].setPhiAddr( (sector*12 + segment + 2)%144 );
          }
       }
     }

  return m_GBOutputMuons;
}
