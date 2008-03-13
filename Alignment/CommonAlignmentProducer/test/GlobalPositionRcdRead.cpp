#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

class  GlobalPositionRcdRead : public edm::EDAnalyzer {
   public:
      explicit  GlobalPositionRcdRead(const edm::ParameterSet& iConfig ) {};
      ~GlobalPositionRcdRead() {}
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
    std::cout << "Calling empty GlobalPositionRcdRead::analyze which is empty. "
	      << "Set 'untracked PSet maxEvents = {untracked int32 input = 0}'"; 
  };
};
  
void GlobalPositionRcdRead::beginJob(const edm::EventSetup& evtSetup) 
{
   std::cout << "Reading from database in GlobalPositionRcdRead::beginJob..." << std::endl;

   edm::ESHandle<Alignments> globalPositionRcd;
   evtSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

   std::cout << "Expecting entries in " << DetId(DetId::Tracker).rawId() << " " 
	     << DetId(DetId::Muon).rawId() << " " << DetId(DetId::Ecal).rawId() << " " 
	     << DetId(DetId::Hcal).rawId() << std::endl;
   for (std::vector<AlignTransform>::const_iterator i = globalPositionRcd->m_align.begin();
	i != globalPositionRcd->m_align.end();  ++i) {
      std::cout << "entry " << i->rawId() 
		<< " translation " << i->translation() 
		<< " angles " << i->rotation().eulerAngles() << std::endl;
      std::cout << i->rotation() << std::endl;
   }

   std::cout << "done!" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdRead);
