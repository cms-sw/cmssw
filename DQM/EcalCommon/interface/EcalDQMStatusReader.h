#ifndef EcalDQMStatusReader_h
#define EcalDQMStatusReader_h

/*
 * \file EcalDQMStatusReader.h
 *
 * $Date: 2010/08/09 09:00:10 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class EcalDQMStatusReader : public edm::EDAnalyzer {

public:

EcalDQMStatusReader(const edm::ParameterSet& ps);
virtual ~EcalDQMStatusReader() {};

void analyze(const edm::Event & e, const edm::EventSetup & c) {};

void beginRun(const edm::Run & r, const edm::EventSetup & c);

private:

bool verbose_;

};
#endif
