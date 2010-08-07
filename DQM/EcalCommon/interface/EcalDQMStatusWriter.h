#ifndef EcalDQMStatusWriter_h
#define EcalDQMStatusWriter_h

/*
 * \file EcalDQMStatusWriter.h
 *
 * $Date: 2010/08/07 10:50:57 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
*/

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <typeinfo>
#include <sstream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class EcalDQMStatusWriter : public edm::EDAnalyzer {

public:

EcalDQMStatusWriter(const edm::ParameterSet& ps);
virtual ~EcalDQMStatusWriter();

void analyze(const edm::Event & e, const edm::EventSetup & c);

void beginJob(void);
void endJob(void);

void beginRun(const edm::Run & r, const edm::EventSetup & c);
void endRun(const edm::Run & r, const edm::EventSetup & c);

private:

EcalDQMChannelStatus* readEcalDQMChannelStatusFromFile(const char *);
EcalDQMTowerStatus* readEcalDQMTowerStatusFromFile(const char *);

uint32_t convert(uint32_t c);

bool verbose_;
std::vector<std::string> objectName_;
std::vector<std::string> inpFileName_;
std::vector<unsigned long long> since_;

const EcalElectronicsMapping* map_;

};
#endif
