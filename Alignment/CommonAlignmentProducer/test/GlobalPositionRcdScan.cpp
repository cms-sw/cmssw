#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESWatcher.h"


class  GlobalPositionRcdScan : public edm::EDAnalyzer
{
public:
  explicit GlobalPositionRcdScan( const edm::ParameterSet& iConfig )
  {}
  ~GlobalPositionRcdScan() {}
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup); 
  
private:

  edm::ESWatcher<GlobalPositionRcd> watcher_;
};

void GlobalPositionRcdScan::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{

  if (watcher_.check(evtSetup)) {
  
    edm::ESHandle<Alignments> globalPositionRcd;
    evtSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

    std::cout << "GlobalPositionRcd content starting from run " << evt.run() << ":" << std::endl;
  
    for (std::vector<AlignTransform>::const_iterator i = globalPositionRcd->m_align.begin();
	 i != globalPositionRcd->m_align.end();  ++i) {
      std::cout << "  Component ";
      if (i->rawId() == DetId(DetId::Tracker).rawId()) {
	std::cout << "Tracker";
      }
      else if (i->rawId() == DetId(DetId::Muon).rawId()) {
	std::cout << "Muon   ";
      }
      else if (i->rawId() == DetId(DetId::Ecal).rawId()) {
	std::cout << "Ecal   ";
      }
      else if (i->rawId() == DetId(DetId::Hcal).rawId()) {
	std::cout << "Hcal   ";
      }
      else if (i->rawId() == DetId(DetId::Calo).rawId()) {
	std::cout << "Calo   ";
      }
      else {
	std::cout << "Unknown";
      }
      std::cout << " entry " << i->rawId() 
		<< " translation " << i->translation() 
		<< " angles " << i->rotation().eulerAngles() << std::endl;
      // std::cout << i->rotation() << std::endl;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdScan);
