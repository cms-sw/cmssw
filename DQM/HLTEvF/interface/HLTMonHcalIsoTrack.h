#include <memory>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>

class HLTMonHcalIsoTrack : public edm::EDAnalyzer {
public:
  explicit HLTMonHcalIsoTrack(const edm::ParameterSet&);
  ~HLTMonHcalIsoTrack();

  double getDist(double,double,double,double);
  std::pair<int, int> towerIndex(double, double);

private:

  int evtBuf;

  DQMStore* dbe_;

  virtual void beginJob(const edm::EventSetup&) ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string folderName_;
  std::string outRootFileName_;

  std::string hltRAWEventTag_;
  std::string hltAODEventTag_;

  std::string l2collectionLabel_;
  std::string l3collectionLabel_;

  std::string l3filterLabel_;
  std::string l1filterLabel_;
  std::string l2filterLabel_;

  std::vector<std::string> trigNames_;
  std::vector<std::string> l2collectionLabels_;
  std::vector<std::string> l3collectionLabels_;

  std::vector<std::string> l3filterLabels_;
  std::vector<std::string> l1filterLabels_;
  std::vector<std::string> l2filterLabels_;

  std::string hltProcess_;

  bool useProducerCollections_;

  bool saveToRootFile_;

  std::vector<edm::ParameterSet> triggers;

  std::vector<MonitorElement*> hL2TowerOccupancy;
  std::vector<MonitorElement*> hL3TowerOccupancy;
  std::vector<MonitorElement*> hL2L3acc;
  std::vector<MonitorElement*> hL3L2trackMatch;
  std::vector<MonitorElement*> hL3candL2rat;
  std::vector<MonitorElement*> hL2isolationP;
  std::vector<MonitorElement*> hL1eta;
  std::vector<MonitorElement*> hL1phi;

};


