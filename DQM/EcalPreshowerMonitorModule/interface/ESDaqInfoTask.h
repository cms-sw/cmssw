#ifndef ESDaqInfoTask_h
#define ESDaqInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//class ESElectronicsMapper;
class ESDaqInfoTask: public edm::EDAnalyzer{

   public:

      /// Constructor
      ESDaqInfoTask(const edm::ParameterSet& ps);

      /// Destructor
      virtual ~ESDaqInfoTask();

   protected:

      /// Analyze
      void analyze(const edm::Event& e, const edm::EventSetup& c);

      /// BeginJob
      void beginJob(const edm::EventSetup& c);

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

      MonitorElement* meESDaqFraction_;
      MonitorElement* meESDaqActive_[56];
      MonitorElement* meESDaqActiveMap_;

      MonitorElement* meESDaqError_;

      int ESFedRangeMin_;
      int ESFedRangeMax_;

      ESElectronicsMapper * es_mapping_;

      bool ESOnFed_[56];

};

#endif

