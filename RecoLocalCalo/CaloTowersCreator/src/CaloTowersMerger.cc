// -*- C++ -*-
//
// Package:    CaloTowersMerger
// Class:      CaloTowersMerger
// 
/**\class CaloTowersMerger CaloTowersMerger.cc RecoLocalCalo/CaloTowersMerger/src/CaloTowersMerger.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Thu Nov  4 16:36:30 CET 2010
// $Id: CaloTowersMerger.cc,v 1.2 2010/11/24 19:52:16 anastass Exp $
//
//

// Anton Anastassov (Northwestern):
// Add code for actual tower merging, define two distinct inputs of
// "regular" and "extra" towers
// This functionality is subject to some restrictions
// (see notes below)



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

//
// class declaration
//

class CaloTowersMerger : public edm::EDProducer {
   public:
      explicit CaloTowersMerger(const edm::ParameterSet&);
      ~CaloTowersMerger();

  CaloTower mergedTower(CaloTower t1, CaloTower t2);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag regularTowerTag,extraTowerTag;
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
CaloTowersMerger::CaloTowersMerger(const edm::ParameterSet& iConfig)
{
  regularTowerTag=iConfig.getParameter<edm::InputTag>("regularTowerTag");
  extraTowerTag=iConfig.getParameter<edm::InputTag>("extraTowerTag");

   //register your products
   produces<CaloTowerCollection>();
}


CaloTowersMerger::~CaloTowersMerger()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloTowersMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<CaloTowerCollection> regTower,extraTower;

  iEvent.getByLabel(regularTowerTag,regTower);
  iEvent.getByLabel(extraTowerTag,extraTower);

  std::auto_ptr<CaloTowerCollection> output;

  if (!regTower.isValid() && !extraTower.isValid()){
    edm::LogError("CaloTowersMerger")<<"both input tag:"<<regularTowerTag<<" and "<<extraTowerTag<<" are invalid. empty merged collection";
    output.reset(new CaloTowerCollection());
    iEvent.put(output);
    return;
  }else if (!regTower.isValid()  || !extraTower.isValid()){
    if (!regTower.isValid() && extraTower.isValid())
      regTower=extraTower;
    output.reset(new CaloTowerCollection(*regTower));
    iEvent.put(output);
    return;
  }
  else{
    //both valid input collections: merging
    output.reset(new CaloTowerCollection());
    output->reserve(regTower->size()+extraTower->size());
  
    CaloTowerCollection::const_iterator rt_begin = regTower->begin();
    CaloTowerCollection::const_iterator rt_end = regTower->end();
    CaloTowerCollection::const_iterator rt_it = rt_begin;

    //vector of overlapping towers
    std::vector<CaloTowerCollection::const_iterator> overlappingTowers;
    overlappingTowers.reserve(extraTower->size());

    for (;rt_it!=rt_end;++rt_it){
      CaloTowerCollection::const_iterator et_it = extraTower->find(rt_it->id());
      if (et_it != extraTower->end()){
	//need to merge the components
	//FIXME

	/////	CaloTower mergedTower(*t1);
	//one needs to merge t1 and t2 into mergedTower
	//end FIXME
	CaloTower mt = mergedTower(*rt_it, *et_it);

	output->push_back(mt);
	overlappingTowers.push_back(et_it);

      }else{
	//just copy the regular tower over
	output->push_back(*rt_it);
      }
    }
    CaloTowerCollection::const_iterator et_begin = extraTower->begin();
    CaloTowerCollection::const_iterator et_end = extraTower->end();
    CaloTowerCollection::const_iterator et_it=et_begin;
    for (;et_it!=et_end;++et_it){
      if (std::find(overlappingTowers.begin(),overlappingTowers.end(),et_it)==overlappingTowers.end())
	//non overlapping tower
	//copy the extra tower over
	output->push_back(*et_it);
    }
    iEvent.put(output);
  }
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
CaloTowersMerger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CaloTowersMerger::endJob() {
}





// Make a new tower by merging two towers containing exclusive hits
// This cannot be done fully consistently and independent of the
// creation mechnaism...
// This functionlaity it to be used only for testing the effects 
// of rejected bad hits.

CaloTower CaloTowersMerger::mergedTower(const CaloTower rt, const CaloTower et) {

  double newOuterE = 0;

  // HO energies are always saved (even if useHO=false)
  // make sure there is no double counting
  // crude test if one has HO energy and the other not (possibly due to bad hit cleanup)
  // However: for |iEta|>15 E_outer has different meening:
  // it holds either the energy in the outer depths in HCAL, etc
  
  if (rt.ietaAbs()<16 && (fabs(rt.outerEnergy()-et.outerEnergy())<0.00001 ) ) {
    // there is no differnce in the store HO energies
    newOuterE = rt.outerEnergy();
  }
  else {
    newOuterE = rt.outerEnergy()+et.outerEnergy();
  }


  bool rt_hasEcalConstit = false;
  bool et_hasEcalConstit = false;
  
  bool rt_hasHcalConstit = false;
  bool et_hasHcalConstit = false;


  // check if there are HCAL/ECAL constituents in the towers

  std::vector<DetId>::const_iterator rc_begin=rt.constituents().begin();
  std::vector<DetId>::const_iterator rc_end=rt.constituents().end();
  std::vector<DetId>::const_iterator rc_it;


   for (rc_it=rc_begin; rc_it!=rc_end; ++rc_it) {
     if (rc_it->det()==DetId::Hcal) rt_hasHcalConstit=true;
     break;
   }
   for (rc_it=rc_begin; rc_it!=rc_end; ++rc_it) {
     if (rc_it->det()==DetId::Ecal) rt_hasEcalConstit=true;
     break;
   }

  std::vector<DetId>::const_iterator ec_begin=et.constituents().begin();
  std::vector<DetId>::const_iterator ec_end=et.constituents().end();
  std::vector<DetId>::const_iterator ec_it;

   for (ec_it=ec_begin; ec_it!=ec_end; ++ec_it) {
     if (ec_it->det()==DetId::Hcal) et_hasHcalConstit=true;
     break;
   }
   for (ec_it=ec_begin; ec_it!=ec_end; ++ec_it) {
     if (ec_it->det()==DetId::Ecal) et_hasEcalConstit=true;
     break;
   }


   std::vector<DetId> combinedConstituents = rt.constituents();
   for (ec_it=ec_begin; ec_it!=ec_end; ++ec_it) {
     // if properly resconstructed, the only possible overlap should be for HO hits that
     // are always listed as constituents if above thereshold (old JetMET request)
     if (std::find(combinedConstituents.begin(),combinedConstituents.end(), *ec_it)==combinedConstituents.end())
       combinedConstituents.push_back(*ec_it);
   }



   GlobalPoint newEmPosition(0.0, 0.0, 0.0);

   // The following assumes the current default
   // momentum reconstruction method (1) and 
   // accepted rechits with some threshod >0


   if (rt_hasEcalConstit && et_hasEcalConstit) {
    
     if (rt.emEnergy()>0 && et.emEnergy()>0) {
       double sumEmE = rt.emEnergy()+ et.emEnergy();
       
       double x =   rt.emEnergy()*rt.emPosition().x() + et.emEnergy()*et.emPosition().x();
       double y =   rt.emEnergy()*rt.emPosition().y() + et.emEnergy()*et.emPosition().y();   
       double z =   rt.emEnergy()*rt.emPosition().z() + et.emEnergy()*et.emPosition().z();
      
       GlobalPoint weightedEmdPosition(x/sumEmE,y/sumEmE,z/sumEmE);
       newEmPosition = weightedEmdPosition;
     }
     

   }
   else if (rt_hasEcalConstit && !et_hasEcalConstit) {
     newEmPosition = rt.emPosition();
   }
   else if (!rt_hasEcalConstit && et_hasEcalConstit) {
     newEmPosition = et.emPosition();
   }
   
   
   GlobalPoint newHadPosition(0.0, 0.0, 0.0);
   // had positions are the same if there is at least one constituent
   if (rt_hasHcalConstit) {
     newHadPosition = rt.hadPosition();
   }
   else if (et_hasHcalConstit) {
     newHadPosition = et.hadPosition();
   }
   

   // MAke the new tower and set all values  
    
   CaloTower mergedTower(rt.id(), rt.emEnergy()+et.emEnergy(), rt.hadEnergy()+et.hadEnergy(), newOuterE, -1, -1, 
			 rt.p4()+et.p4(), newEmPosition, newHadPosition);
   
   mergedTower.addConstituents(combinedConstituents);

   (rt.hottestCellE() > et.hottestCellE())?
     mergedTower.setHottestCellE(rt.hottestCellE()) : 
     mergedTower.setHottestCellE(et.hottestCellE());

  unsigned int numBadHcalChan = rt.numBadHcalCells() - et.numProblematicHcalCells() - rt.numRecoveredHcalCells();
   unsigned int numBadEcalChan = rt.numBadEcalCells() - et.numProblematicEcalCells() - rt.numRecoveredEcalCells();

   unsigned int numProbHcalChan = rt.numProblematicHcalCells() + et.numProblematicHcalCells();
   unsigned int numProbEcalChan = rt.numProblematicEcalCells() + et.numProblematicEcalCells();

   unsigned int numRecHcalChan = rt.numRecoveredHcalCells() + et.numRecoveredHcalCells();
   unsigned int numRecEcalChan = rt.numRecoveredEcalCells() + et.numRecoveredEcalCells();

   mergedTower.setCaloTowerStatus(numBadHcalChan, numBadEcalChan,    
				  numRecHcalChan, numRecEcalChan,    
				  numProbHcalChan, numProbEcalChan);

   // use timing from the good tower only (for now, in default reco we use information from good hits only)
   // time is saved as integer but returned as float in (ns)
   mergedTower.setEcalTime( int(rt.ecalTime()*100.0 + 0.5) );
   mergedTower.setHcalTime( int(rt.hcalTime()*100.0 + 0.5) );


  return mergedTower;

}



//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowersMerger);
