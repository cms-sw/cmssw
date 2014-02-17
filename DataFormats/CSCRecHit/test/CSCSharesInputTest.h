//
// Package:    SharesInputTest
// Class:      SharesInputTest
// 

//
// Original Author:  Phillip Killewald
//         Created:  Thu Jan 29 17:33:51 CET 2009
// $Id: CSCSharesInputTest.h,v 1.4 2010/02/11 00:11:26 wmtan Exp $
//


#include <map>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TNtuple.h"


class CSCSharesInputTest : public edm::EDAnalyzer {
	public:
		explicit CSCSharesInputTest(const edm::ParameterSet &myConfig);
		
		~CSCSharesInputTest();
		
	private:
		virtual void beginJob();
		
		virtual void analyze(const edm::Event &myEvent, const edm::EventSetup &mySetup);
		
		virtual void endJob();
		
		edm::InputTag cscRecHitTag_;
		edm::InputTag muonTag_;
		
		std::map<std::string, uint64_t> counts_;
		
		edm::Service<TFileService> rootFile_;
		std::map<std::string, TNtuple *> ntuples_;
};

