#ifndef Gen_DataCardFileWriter_H
#define Gen_DataCardFileWriter_H

// I. M. Nugent


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace gen {

class DataCardFileWriter : public  edm::EDAnalyzer {
 public:
  DataCardFileWriter(const edm::ParameterSet&);
  ~DataCardFileWriter() override{};

  void beginJob() override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override{};
  void endJob() override{};

};

};

#endif
