#ifndef CSCFileDumper_h
#define CSCFileDumper_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <stdio.h>

class CSCFileDumper: public edm::EDAnalyzer {
public:
	std::map<int,FILE*> dump_files;
	std::set<unsigned long> eventsToDump;
//	std::??? rangesToDump; to be implemented

	std::string output, events;
	int fedID_first, fedID_last;

	CSCFileDumper(const edm::ParameterSet & pset);
	virtual ~CSCFileDumper(void);

    void analyze(const edm::Event & e, const edm::EventSetup& c);

private:
   
   std::string source_;
};

#endif
