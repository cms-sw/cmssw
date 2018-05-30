#ifndef ESDcsInfoTask_h
#define ESDcsInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"

class MonitorElement;
class DQMStore;

class ESDcsInfoTask: public edm::EDAnalyzer{

   public:

      /// Constructor
      ESDcsInfoTask(const edm::ParameterSet& ps);

      /// Destructor
      ~ESDcsInfoTask() override;

   protected:

      /// Analyze
      void analyze(const edm::Event& e, const edm::EventSetup& c) override;

      /// BeginJob
      void beginJob(void) override;

      /// EndJob
      void endJob(void) override;

      /// BeginLuminosityBlock
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup) override;

      /// Reset
      void reset(void);

      /// Cleanup
      void cleanup(void);

   private:

      DQMStore* dqmStore_;

      std::string prefixME_;

      bool enableCleanup_;

      bool mergeRuns_;

      edm::EDGetTokenT<DcsStatusCollection> dcsStatustoken_;

      MonitorElement* meESDcsFraction_;
      MonitorElement* meESDcsActiveMap_;

      int ievt_;

};

#endif
