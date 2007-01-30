#ifndef CSCTFPacker_h
#define CSCTFPacker_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"

class CSCTFPacker : public edm::EDAnalyzer {
private:
	CSCTriggerMappingFromFile* TFMapping;

	bool zeroSuppression;
	int  nTBINs;
	unsigned short activeSectors;

	FILE *file;

public:
	virtual void beginJob(const edm::EventSetup&){}
	virtual void endJob(void){}
	virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);

	explicit CSCTFPacker(const edm::ParameterSet &conf);
	~CSCTFPacker(void);
};

#endif
