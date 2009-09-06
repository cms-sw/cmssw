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
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMFileReader : public edm::EDAnalyzer {
public:
  explicit DQMFileReader(const edm::ParameterSet&);
  ~DQMFileReader();
  

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  DQMStore *dbe_;  
  std::vector<std::string > filenames_;

};

#endif
