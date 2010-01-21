#ifndef ESPedestalTask_H
#define ESPedestalTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESPedestalTask : public edm::EDAnalyzer {

   public:

      ESPedestalTask(const edm::ParameterSet& ps);
      virtual ~ESPedestalTask();

   private:

      void beginJob(void);
      void analyze(const edm::Event&, const edm::EventSetup&);
      void endJob(void);
      void setup(void);
      void reset(void);
      void cleanup(void);
      void beginRun(const edm::Run & r, const edm::EventSetup & c);
      void endRun(const edm::Run& r, const edm::EventSetup& c);

      edm::InputTag digilabel_;
      edm::FileInPath lookup_;
      std::string outputFile_;
      std::string prefixME_;

      bool enableCleanup_;
      bool mergeRuns_;

      DQMStore* dqmStore_;
      MonitorElement* meADC_[4288][32];

      bool init_;
      int nLines_, runNum_, ievt_, senCount_[2][2][40][40]; 
      int runtype_, seqtype_, dac_, gain_, precision_;
      int firstDAC_, nDAC_, isPed_, vDAC_[5];

};

#endif
