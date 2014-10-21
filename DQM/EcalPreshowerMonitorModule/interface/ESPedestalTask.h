#ifndef ESPedestalTask_H
#define ESPedestalTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MonitorElement;

class ESPedestalTask : public DQMEDAnalyzer {

   public:

      ESPedestalTask(const edm::ParameterSet& ps);
      virtual ~ESPedestalTask() {}

   private:

      void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob(void);

      edm::EDGetTokenT<ESDigiCollection> digitoken_;
      edm::FileInPath lookup_;
      std::string outputFile_;
      std::string prefixME_;

      MonitorElement* meADC_[4288][32];

      int nLines_, runNum_, ievt_, senCount_[2][2][40][40]; 
      int runtype_, seqtype_, dac_, gain_, precision_;
      int firstDAC_, nDAC_, isPed_, vDAC_[5];

};

#endif
