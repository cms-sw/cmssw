#ifndef ESPedestalTask_H
#define ESPedestalTask_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
class MonitorElement;
class DQMStore;

class ESPedestalTask : public edm::EDAnalyzer {
   public:
      ESPedestalTask(const edm::ParameterSet& ps);
      virtual ~ESPedestalTask();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      edm::InputTag digilabel_;
      edm::FileInPath lookup_;
      std::string outputFile_;
      std::string prefixME_;


      DQMStore* dqmStore_;
      MonitorElement* hADC_[2][2][40][40][32];

     int nLines_, runNum_, eCount_; 
     int runtype_, seqtype_, dac_, gain_, precision_;
     int firstDAC_, nDAC_, isPed_, vDAC_[5], layer_;

};

#endif
