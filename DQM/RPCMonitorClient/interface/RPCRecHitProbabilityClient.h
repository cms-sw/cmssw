#ifndef RPCRecHitProbabilityClient_H
#define RPCRecHitProbabilityClient_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>


class RPCRecHitProbabilityClient:public edm::EDAnalyzer{

public:

  /// Constructor
 RPCRecHitProbabilityClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~ RPCRecHitProbabilityClient();

  /// BeginJob
  void beginJob( );

  //Begin Run
   void beginRun(const edm::Run& , const edm::EventSetup&);
    
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& ) ;

  /// Analyze  
  void analyze(const edm::Event& , const edm::EventSetup& );

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& );
 
  //End Run
  void endRun(const edm::Run& , const edm::EventSetup& ); 		
  
  /// Endjob
  void endJob();

 private:

    std::string  globalFolder_;
  
    DQMStore* dbe_;
 
  
};
#endif
