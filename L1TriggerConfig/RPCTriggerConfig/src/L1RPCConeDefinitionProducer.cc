// -*- C++ -*-
//
// Package:    L1RPCConeDefinitionProducer
// Class:      L1RPCConeDefinitionProducer
// 
/**\class L1RPCConeDefinitionProducer L1RPCConeDefinitionProducer.h L1TriggerConfig/L1RPCConeDefinitionProducer/src/L1RPCConeDefinitionProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Mon Feb 23 12:09:06 CET 2009
// $Id: L1RPCConeDefinitionProducer.cc,v 1.1 2009/03/19 14:38:13 fruboes Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"


#include "CondFormats/RPCObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"

//
// class decleration
//

class L1RPCConeDefinitionProducer : public edm::ESProducer {
   public:
      L1RPCConeDefinitionProducer(const edm::ParameterSet&);
      ~L1RPCConeDefinitionProducer();

      typedef boost::shared_ptr<L1RPCConeDefinition> ReturnType;

      ReturnType produce(const L1RPCConeDefinitionRcd&);
   private:
      // ----------member data ---------------------------
     int m_towerBeg;
     int m_towerEnd;
     int m_rollBeg;
     int m_rollEnd;
     int m_hwPlaneBeg;
     int m_hwPlaneEnd;
     
     L1RPCConeDefinition::TLPSizesInTowers m_LPSizesInTowers;
     //L1RPCConeDefinition::TRingsToTowers m_RingsToTowers;
     L1RPCConeDefinition::TRingToTowerVec m_ringToTowerVec;
     L1RPCConeDefinition::TRingsToLP m_RingsToLP;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1RPCConeDefinitionProducer::L1RPCConeDefinitionProducer(const edm::ParameterSet& iConfig):
      m_towerBeg(iConfig.getParameter<int>("towerBeg")),
      m_towerEnd(iConfig.getParameter<int>("towerEnd")),
      m_rollBeg(iConfig.getParameter<int>("rollBeg")),
      m_rollEnd(iConfig.getParameter<int>("rollEnd")),
      m_hwPlaneBeg(iConfig.getParameter<int>("hwPlaneBeg")),
      m_hwPlaneEnd(iConfig.getParameter<int>("hwPlaneEnd"))
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  for (int i = m_towerBeg; i <= m_towerEnd; ++i){
      
    std::stringstream name;
    name << "lpSizeTower" << i;
      
    L1RPCConeDefinition::TLogPlaneSize newSizes = 
        iConfig.getParameter<std::vector<int> >(name.str().c_str());
      
    m_LPSizesInTowers.push_back(newSizes);
    
  }



   //now do what ever other initialization is needed
   
   //  hw planes numbered from 0 to 5
   // rolls from 0 to 17 (etaPartition)
   //
   //  rollConnLP_[roll]_[hwPlane-1]
   //  rollConnLP_5_3 = cms.vint32(6, 0, 0),
   //     ----- roll 5, hwPlane 4 (3+1) is logplane 6 (OK)
   //
   //  rollConnT_[roll]_[hwPlane-1]
   //  rollConnT_5_3 = cms.vint32(4, -1, -1),
   //     ----- roll 5, hwPlane 4 (3+1) contirubtes to tower 4 (OK)
  
   for (int roll = m_rollBeg; roll <= m_rollEnd; ++roll){
    //L1RPCConeDefinition::THWplaneToTower newHwPlToTower;
    L1RPCConeDefinition::THWplaneToLP newHWplaneToLP;
    for (int hwpl = m_hwPlaneBeg; hwpl <= m_hwPlaneEnd; ++hwpl){
      std::stringstream name;
      name << "rollConnLP_" << roll << "_" << hwpl;
            
      L1RPCConeDefinition::TLPList newListLP = 
      iConfig.getParameter<std::vector<int> >(name.str().c_str());
      newHWplaneToLP.push_back(newListLP);
            
            
      std::stringstream name1;
      name1 << "rollConnT_" << roll << "_" << hwpl;
            
      /*L1RPCConeDefinition::TLPList newListT =  
      iConfig.getParameter<std::vector<int> >(name1.str().c_str());
      newHwPlToTower.push_back(newListT);*/
      std::vector<int> hwPl2TowerVec = iConfig.getParameter<std::vector<int> >(name1.str().c_str());
      
      for (unsigned int i = 0;i < hwPl2TowerVec.size();++i){
        
        if (hwPl2TowerVec[i]>=0)
        {
          L1RPCConeDefinition::TRingToTower rt(roll, hwpl, hwPl2TowerVec[i],i);
          m_ringToTowerVec.push_back(rt);
        }
      }


    }
    //m_RingsToTowers.push_back(newHwPlToTower);
    
    m_RingsToLP.push_back(newHWplaneToLP);
  }
}


L1RPCConeDefinitionProducer::~L1RPCConeDefinitionProducer(){}



//
// member functions
//

// ------------ method called to produce the data  ------------
L1RPCConeDefinitionProducer::ReturnType
L1RPCConeDefinitionProducer::produce(const L1RPCConeDefinitionRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1RPCConeDefinition> pL1RPCConeDefinition(new L1RPCConeDefinition);

   pL1RPCConeDefinition->setFirstTower(m_towerBeg);
   pL1RPCConeDefinition->setLastTower(m_towerEnd);
   
   pL1RPCConeDefinition->setLPSizeForTowers(m_LPSizesInTowers);
   pL1RPCConeDefinition->setRingsToLP(m_RingsToLP);
   pL1RPCConeDefinition->setRingToTowerVec(m_ringToTowerVec);
   
   
   return pL1RPCConeDefinition ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RPCConeDefinitionProducer);
