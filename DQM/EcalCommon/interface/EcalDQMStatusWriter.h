#ifndef EcalDQMStatusWriter_h
#define EcalDQMStatusWriter_h

/*
 * \file EcalDQMStatusWriter.h
 *
 * $Date: 2010/08/06 15:34:49 $
 * $Revision: 1.1 $
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

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class EcalDQMStatusWriter : public edm::EDAnalyzer {

 public:

  EcalDQMStatusWriter(const edm::ParameterSet& ps);
  virtual ~EcalDQMStatusWriter();

  void analyze(const edm::Event & e, const edm::EventSetup & c);
  void endJob(void);

 private:

  EcalDQMChannelStatus* readEcalDQMChannelStatusFromFile(const char *);
  EcalDQMTowerStatus* readEcalDQMTowerStatusFromFile(const char *);

  int convert(int c);

  std::vector<std::string> objectName_;
  std::vector<std::string> inpFileName_;
  std::vector<unsigned long long> since_;

};
#endif
