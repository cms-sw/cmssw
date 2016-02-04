#ifndef RPCFEDIntegrity_H
#define RPCFEDIntegrity_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <string>
#include <map>
class DQMStore;
class MonitorElement;

class RPCFEDIntegrity:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCFEDIntegrity(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCFEDIntegrity();

  /// BeginJob
  void beginJob();

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  
  void endJob();
  
 private:
  
  void labelBins( MonitorElement * myMe);
  edm::InputTag rawCountsLabel_;
  void reset(void);
  void bookFEDMe(void);

  std::string  prefixDir_;
  
  bool merge_, init_;

  DQMStore* dbe_;

  int FATAL_LIMIT;

  enum fedHisto{Entries, Fatal, NonFatal};

  MonitorElement * fedMe_[3];

  int  numOfFED_ ,  minFEDNum_ ,  maxFEDNum_ ;
  std::vector<std::string> histoName_; 
};

#endif
