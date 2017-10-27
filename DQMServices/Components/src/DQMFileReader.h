#ifndef DQMSERVICES_COMPONENTS_DQMFileReader_H
# define DQMSERVICES_COMPONENTS_DQMFileReader_H
// -*- C++ -*-
//
// Package:    DQMFileReader
// Class:      DQMFileReader
// 
/*
   
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kenichi Hatakeyama
//
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMFileReader : public edm::EDAnalyzer {
public:
  explicit DQMFileReader(const edm::ParameterSet&);
  ~DQMFileReader() override;
  

private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  
  DQMStore *dbe_;  
  edm::ParameterSet pset_;

  std::vector<std::string > filenames_;
  std::string referenceFileName_;

};

#endif
