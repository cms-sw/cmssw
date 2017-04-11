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
#include "FWCore/Framework/interface/global/EDProducer.h"

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

class EcalDigiToRaw : public edm::global::EDProducer<> {
   public:
       EcalDigiToRaw(const edm::ParameterSet& pset);
       virtual ~EcalDigiToRaw();

       virtual void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

       typedef long long Word64;
       typedef unsigned int Word32;

       bool GetDebug() const {return debug_ ;}
       bool GetDoBarrel() const {return doBarrel_ ;}
	bool GetDoEndCap() const {return doEndCap_ ;}
	bool GetDoSR() const {return doSR_ ;}
	bool GetDoTower() const{return doTower_ ;}
	bool GetDoTCC() const {return doTCC_ ;}

        const std::vector<int32_t>* GetListDCCId() const {return &listDCCId_ ;}
    
	static const int BXMAX = 2808;


   private:


      // ----------member data ---------------------------

	const bool doTCC_;
	const bool doSR_;
	const bool doTower_;

	const bool doBarrel_;
	const bool doEndCap_;

        const std::vector<int32_t> listDCCId_;
    
	const std::string label_;
	const std::string instanceNameEB_;
	const std::string instanceNameEE_;

	const edm::EDGetTokenT<EBDigiCollection> EBDigiToken_ ;
	const edm::EDGetTokenT<EEDigiCollection> EEDigiToken_;
	const edm::EDGetTokenT<EcalTrigPrimDigiCollection> labelTT_ ;
	const edm::EDGetTokenT<EBSrFlagCollection> labelEBSR_ ;
	const edm::EDGetTokenT<EESrFlagCollection> labelEESR_ ;
	const bool debug_;
        const std::unique_ptr<TowerBlockFormatter> Towerblockformatter_;
        const std::unique_ptr<TCCBlockFormatter>   TCCblockformatter_;
	const std::unique_ptr<BlockFormatter>	     Headerblockformatter_;
	const std::unique_ptr<SRBlockFormatter>    SRblockformatter_;


};

//define this as a plug-in
// DEFINE_FWK_MODULE(EcalDigiToRaw);

#endif

