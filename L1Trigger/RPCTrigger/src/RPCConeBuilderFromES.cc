// -*- C++ -*-
//
// Package:     RPCTrigger
// Class  :     RPCConeBuilderFromES
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Mon Mar  3 13:34:20 CET 2008
// $Id: RPCConeBuilderFromES.cc,v 1.11 2011/02/25 16:56:18 fruboes Exp $
//

// system include files

// user include files
#include "L1Trigger/RPCTrigger/interface/RPCConeBuilderFromES.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCConeBuilderFromES::RPCConeBuilderFromES()
{
}

// RPCConeBuilderFromES::RPCConeBuilderFromES(const RPCConeBuilderFromES& rhs)
// {
//    // do actual copying here;
// }

RPCConeBuilderFromES::~RPCConeBuilderFromES()
{
}

L1RpcLogConesVec RPCConeBuilderFromES::getConesFromES(edm::Handle<RPCDigiCollection> rpcDigis, 
                                                      edm::ESHandle<L1RPCConeBuilder> coneBuilder,
                                                      edm::ESHandle<L1RPCConeDefinition> coneDef,
                                                      edm::ESHandle<L1RPCBxOrConfig> bxOrDef,
                                                      edm::ESHandle<L1RPCHwConfig> hwConfig, int bx)
{
  std::vector<RPCLogHit> logHits;
  std::vector<RPCLogHit> logHitsFromUncomp;
  
  // Build cones from digis
  // first build loghits

  short int digiIndex = 0; 
  RPCDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=rpcDigis->begin();
       detUnitIt!=rpcDigis->end();
       ++detUnitIt)
  {
    const RPCDetId& id = (*detUnitIt).first;

    uint32_t rawId = id.rawId();

    const RPCDigiCollection::Range& range = (*detUnitIt).second;

    std::pair<L1RPCConeBuilder::TCompressedConVec::const_iterator, L1RPCConeBuilder::TCompressedConVec::const_iterator> 
          compressedConnPair = coneBuilder->getCompConVec(rawId);

    // iterate over strips
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;
         ++digiIt)
    {
      
      ++digiIndex;
      if ( digiIt->bx() < bxOrDef->getFirstBX() + bx || digiIt->bx() > bxOrDef->getLastBX() +bx  ){
      //if ( digiIt->bx() < hwConfig->getFirstBX() + bx || digiIt->bx() > hwConfig->getLastBX() +bx  ){
        continue;
      }
      
      //std::cout << digiIt->bx() << " D " << rawId << " " << id << " S " <<  digiIt->strip() << std::endl;
      // for uncompressed connections
      std::pair<L1RPCConeBuilder::TStripConVec::const_iterator, L1RPCConeBuilder::TStripConVec::const_iterator> 
          itPair = coneBuilder->getConVec(rawId,digiIt->strip());

      L1RPCConeBuilder::TStripConVec::const_iterator it = itPair.first;
      // Iterate over uncompressed connections, convert digis to logHits 
      for (; it!=itPair.second;++it){
         //std::cout << " Not empty!" << std::endl;
         if ( hwConfig->isActive(it->m_tower, it->m_PAC)  ){

             RPCLogHit lh(it->m_tower, it->m_PAC, it->m_logplane, it->m_logstrip); 
             lh.setDigiIdx(digiIndex);  
             logHitsFromUncomp.push_back( lh );
         } 
      }

      /*
      bool printOut = false;
      if (digiIt->strip() == 62 || digiIt->strip() == 63 ){
        std::cout << "Strip " << digiIt->strip() << std::endl;
        printOut = true;
      }
      */
      
      L1RPCConeBuilder::TCompressedConVec::const_iterator itComp = compressedConnPair.first;
      for (; itComp!=compressedConnPair.second; ++itComp){
         if ( hwConfig->isActive(itComp->m_tower, itComp->m_PAC)){
           int logstrip = itComp->getLogStrip(digiIt->strip(),coneDef->getLPSizeVec());
           if (logstrip!=-1){
               RPCLogHit lh(itComp->m_tower, itComp->m_PAC, itComp->m_logplane, logstrip );
               lh.setDigiIdx(digiIndex);
               logHits.push_back( lh );
           }
           /*
           if (printOut){
             std::cout << "T " << (int)itComp->m_tower << " P " 
                 << (int)itComp->m_PAC << " LP " 
                 << (int)itComp->m_logplane << " LS " 
                 << (int)logstrip << std::endl;
         }*/
           
         }
      }

    } // strip iteration ends
    
  }
  

  // check if we dont have any preferable uncompressed loghits
  std::vector<RPCLogHit>::iterator itLHitUncomp = logHitsFromUncomp.begin();
  std::vector<RPCLogHit>::iterator itLHitComp;

  // overwrite uncompressed with those coming from compressed
  for(;itLHitUncomp != logHitsFromUncomp.end(); ++itLHitUncomp) {
    for (itLHitComp = logHits.begin();  itLHitComp != logHits.end(); ++itLHitComp){

      if ( itLHitComp->getTower() == itLHitUncomp->getTower() 
           && itLHitComp->getLogSector() == itLHitUncomp->getLogSector()   
           && itLHitComp->getLogSegment() == itLHitUncomp->getLogSegment()   
           && itLHitComp->getlogPlaneNumber() == itLHitUncomp->getlogPlaneNumber()  )
      {
//         std::cout<< "Overwrite " << std::endl;
        //std::cout.flush();
          *itLHitUncomp = *itLHitComp;
      } 

    }
  }

  // copy missing from compressed to uncompressed  
  for(;itLHitUncomp != logHitsFromUncomp.end(); ++itLHitUncomp) {
    bool present = false;
    for (unsigned int i=0;  i < logHits.size(); ++i)  
    {

      if ( logHits[i].getTower() == itLHitUncomp->getTower()
           && logHits[i].getLogSector() == itLHitUncomp->getLogSector()
           && logHits[i].getLogSegment() == itLHitUncomp->getLogSegment()
           && logHits[i].getlogPlaneNumber() == itLHitUncomp->getlogPlaneNumber()  )
      {
         present = true;
      }
    }
    if (!present)
    {
//       std::cout<< "Copy " << std::endl;
      //std::cout.flush();

      logHits.push_back(*itLHitUncomp);
    }
  }

  // build cones
  L1RpcLogConesVec ActiveCones;

  std::vector<RPCLogHit>::iterator p_lhit;
  for (p_lhit = logHits.begin(); p_lhit != logHits.end(); ++p_lhit){

    bool hitTaken = false;
    L1RpcLogConesVec::iterator p_cone;
    for (p_cone = ActiveCones.begin(); p_cone != ActiveCones.end(); p_cone++){
      hitTaken = p_cone->addLogHit(*p_lhit);
      if(hitTaken)
        break;
    }

    if(!hitTaken) {
      RPCLogCone newcone(*p_lhit);
      newcone.setIdx(ActiveCones.size());
      ActiveCones.push_back(newcone);
    }
  }// for loghits

  /*
  for (int tower = -16; tower<17;++tower)
  {
    for (int sector = 0; sector<12;++sector)
    {
      for (int segment = 0; segment<12;++segment)
      {
        for (L1RpcLogConesVec::iterator it =  ActiveCones.begin(); it!=ActiveCones.end(); ++it)
        {
          if (it->getTower()==tower 
              && it->getLogSector()==sector
              && it->getLogSegment()==segment)
          {
           std::cout << it->toString() << std::endl;
          }
        }
      }
    }
  }
  // */
  
  return ActiveCones;
  
}

