// -*- C++ -*-
//
// Package:   EcalChannelCheckerTreeHists 
// Class:     EcalChannelCheckerTreeHists 
// 
/**\class EcalChannelCheckerTreeHists EcalChannelCheckerTreeHists.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//          Author:  Caterina DOGLIONI
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalChannelCheckerTreeHists.h,v 1.5 2008/05/05 13:30:45 doglioni Exp $
//
// 

// system include files
#include <memory>
#include <vector>
#include <set>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "TFile.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH2D.h"
#include "TProfile2D.h"
#include "TProfile.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#define MAX_XTALS 61200

//
// class declaration
//

class EcalChannelCheckerTreeHists : public edm::EDAnalyzer {
public:
	explicit EcalChannelCheckerTreeHists(const edm::ParameterSet&);
	~EcalChannelCheckerTreeHists();

private:
	virtual void beginJob(const edm::EventSetup&) ;
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void endJob() ;
	std::string intToString(int num);
	void makeTree();
        float getEntriesAvg();


// ----------member data ---------------------------

	edm::InputTag EBDigis_;
	edm::InputTag EBUncalibratedRecHitCollection_;
	edm::InputTag headerProducer_;
        const EcalElectronicsMapping* ecalElectronicsMap_;
        EcalFedMap* fedMap_;

	//histograms - for each quantity, one 2d map and one vector of crystal histogram 
	
	TProfile2D * prof2_XtalJitter_;
	TH1D* v_h1_XtalJitter_[MAX_XTALS];

	TProfile2D * prof2_XtalAmpli_;
	TH1D* v_h1_XtalAmpli_[MAX_XTALS];

	TProfile2D * prof2_XtalPed_;
	TH1D* v_h1_XtalPed_[MAX_XTALS];

	//for jitter: vector of crystal profiles
        TProfile * v_prof_XtalPulse_[MAX_XTALS];

	edm::Service<TFileService> fs_;
	TFileDirectory XtalJitterDir_;
	TFileDirectory XtalAmpliDir_;
	TFileDirectory XtalPedDir_;
        TFileDirectory XtalPulseDir_;

	//other parameters
	int runNum_;
	int eventNum_;
};
