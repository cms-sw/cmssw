#ifndef CondTools_SiStrip_FedCablingReader_H
#define CondTools_SiStrip_FedCablingReader_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class SiStripFedCablingReader : public edm::EDAnalyzer {

 public:
  
  SiStripFedCablingReader( const edm::ParameterSet& );
  
  ~SiStripFedCablingReader() {;}

  void beginRun( const edm::Run&, const edm::EventSetup& );

  void analyze(const edm::Event&, const edm::EventSetup&){;}
  
 private:

  bool printFecCabling_;
  bool printDetCabling_;
  bool printRegionCabling_;

};

#endif // CondTools_SiStrip_FedCablingReader_H
