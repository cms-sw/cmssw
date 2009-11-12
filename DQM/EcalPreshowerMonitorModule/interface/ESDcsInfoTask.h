#ifndef ESDcsInfoTask_h
#define ESDcsInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESDcsInfoTask: public edm::EDAnalyzer{

   public:

      /// Constructor
      ESDcsInfoTask(const edm::ParameterSet& ps);

      /// Destructor
      virtual ~ESDcsInfoTask();

   protected:

      /// Analyze
      void analyze(const edm::Event& e, const edm::EventSetup& c);

      /// BeginJob
      void beginJob(void);

      /// EndJob
      void endJob(void);

      /// BeginLuminosityBlock
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup);

      /// EndLuminosityBlock
      void endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup);

      /// Reset
      void reset(void);

      /// Cleanup
      void cleanup(void);

   private:

      DQMStore* dqmStore_;

      std::string prefixME_;

      bool enableCleanup_;

      bool mergeRuns_;

      edm::InputTag dcsStatuslabel_;

      MonitorElement* meESDcsFraction_;
      MonitorElement* meESDcsActiveMap_;

      int ievt_;

};

#endif
