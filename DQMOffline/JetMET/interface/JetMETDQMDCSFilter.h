#ifndef JetMETDQMDCSFilter_H
#define JetMETDQMDCSFilter_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

class JetMETDQMDCSFilter {

 public:
  JetMETDQMDCSFilter( const edm::ParameterSet & );
  ~JetMETDQMDCSFilter();
  bool filter(const edm::Event& evt, const edm::EventSetup& es);
  bool passPIX, passSiStrip, passECAL, passHBHE, passHF, passHO, passES, passMuon;

  private:
  bool verbose_;
  bool filter_;
  bool detectorOn_;
  std::string detectorTypes_;

};

#endif
