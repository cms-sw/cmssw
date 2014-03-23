#ifndef EcalDigiToRaw_H
#define EcalDigiToRaw_H

//
// Package:    EcalDigiToRaw
// Class:      EcalDigiToRaw
// 
/**\class EcalDigiToRaw EcalDigiToRaw.cc SimCalorimetry/EcalDigiToRaw/src/EcalDigiToRaw.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Emmanuelle Perez
//         Created:  Sat Nov 25 13:59:51 CET 2006
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/EcalDigiToRaw/interface/TowerBlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/TCCBlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/SRBlockFormatter.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"


//
// class decleration
//

class EcalDigiToRaw : public edm::EDProducer {
   public:
       EcalDigiToRaw(const edm::ParameterSet& pset);
       virtual ~EcalDigiToRaw();

      void beginJob();
      void produce(edm::Event& e, const edm::EventSetup& c);
      void endJob() ;

      typedef long long Word64;
      typedef unsigned int Word32;

    	int* GetCounter() {return &counter_ ;}
	bool GetDebug() {return debug_ ;}
	int* GetOrbit() {return &orbit_number_ ;}
	int* GetBX() {return &bx_ ;}
	int* GetLV1() {return &lv1_ ;}
	int* GetRunNumber() {return &runnumber_ ;}
	bool GetDoBarrel() {return doBarrel_ ;}
	bool GetDoEndCap() {return doEndCap_ ;}
	bool GetDoSR() {return doSR_ ;}
	bool GetDoTower() {return doTower_ ;}
	bool GetDoTCC() {return doTCC_ ;}

        std::vector<int32_t>* GetListDCCId() {return &listDCCId_ ;}
    
	static const int BXMAX = 2808;


   private:


      // ----------member data ---------------------------

        int  counter_;
	int orbit_number_;
	bool debug_;
	int runnumber_;
	int bx_;
	int lv1_;

	bool doTCC_;
	bool doSR_;
	bool doTower_;

	edm::EDGetTokenT<EcalTrigPrimDigiCollection> labelTT_ ;
	edm::EDGetTokenT<EBSrFlagCollection> labelEBSR_ ;
	edm::EDGetTokenT<EESrFlagCollection> labelEESR_ ;
	edm::EDGetTokenT<EBDigiCollection> EBDigiToken_ ;
	edm::EDGetTokenT<EEDigiCollection> EEDigiToken_;

	bool doBarrel_;
	bool doEndCap_;

        std::vector<int32_t> listDCCId_;
    
	std::string label_;
	std::string instanceNameEB_;
	std::string instanceNameEE_;

        TowerBlockFormatter* Towerblockformatter_;
        TCCBlockFormatter*   TCCblockformatter_;
	BlockFormatter*	     Headerblockformatter_;
	SRBlockFormatter*    SRblockformatter_;


};

//define this as a plug-in
// DEFINE_FWK_MODULE(EcalDigiToRaw);

#endif

