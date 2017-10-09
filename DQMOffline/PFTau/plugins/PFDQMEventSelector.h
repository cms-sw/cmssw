#ifndef PFDQMEventSelector_H
#define PFDQMEventSelector_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class DQMStore;
class PFDQMEventSelector : public edm::EDFilter {

public:
   PFDQMEventSelector( const edm::ParameterSet & );
 ~PFDQMEventSelector();

private:
  void beginJob();
  bool filter( edm::Event &, edm::EventSetup const& );
  void endJob();

  bool openInputFile();

  uint64_t nEvents_, nSelectedEvents_;
  bool verbose_;

  std::vector<std::string> folderNames_;
  std::string inputFileName_; 
  bool fileOpened_; 

  DQMStore* dqmStore_;
};

#endif
