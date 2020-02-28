#ifndef ESDaqInfoTask_h
#define ESDaqInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"  // definition in line 75
#include "DQMServices/Core/interface/DQMStore.h"

class ESDaqInfoTask : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  /// Constructor
  ESDaqInfoTask(const edm::ParameterSet& ps);

  /// Destructor
  ~ESDaqInfoTask() override;

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// BeginJob
  void beginJob(void) override;

  /// EndJob
  void endJob(void) override;

  /// BeginLuminosityBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) override;

  /// Reset
  void reset(void);

private:
  DQMStore* dqmStore_;

  std::string prefixME_;

  bool mergeRuns_;

  MonitorElement* meESDaqFraction_;
  MonitorElement* meESDaqActive_[56];
  MonitorElement* meESDaqActiveMap_;

  MonitorElement* meESDaqError_;

  int ESFedRangeMin_;
  int ESFedRangeMax_;

  ESElectronicsMapper* es_mapping_;

  bool ESOnFed_[56];

  int getFEDNumber(const int x, const int y) {
    int iz = (x < 40) ? 1 : 2;
    int ip = (y >= 40) ? 1 : 2;
    int ix = (x < 40) ? x : x - 40;
    int iy = (y < 40) ? y : y - 40;
    return (*es_mapping_).getFED(iz, ip, ix + 1, iy + 1);
  }
};

#endif
