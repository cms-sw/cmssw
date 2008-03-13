#include <string>
#include <map>
#include <vector>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DetId/interface/DetId.h"

// Database
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

class  GlobalPositionRcdWrite : public edm::EDAnalyzer {
   public:
      explicit  GlobalPositionRcdWrite(const edm::ParameterSet& iConfig)
         : m_tracker(iConfig.getParameter<edm::ParameterSet>("tracker"))
         , m_muon(iConfig.getParameter<edm::ParameterSet>("muon"))
         , m_ecal(iConfig.getParameter<edm::ParameterSet>("ecal"))
         , m_hcal(iConfig.getParameter<edm::ParameterSet>("hcal"))
	 , nEventCalls_(0)
      {};
      ~GlobalPositionRcdWrite() {}
  virtual void beginJob(const edm::EventSetup& iSetup) {};
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

   private:
      edm::ParameterSet m_tracker, m_muon, m_ecal, m_hcal;
  unsigned int nEventCalls_;
};
  
void GlobalPositionRcdWrite::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{
   if (nEventCalls_ > 0) {
     std::cout << "Writing to DB to be done only once, "
	       << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'."
	       << "(Your writing should be fine.)" << std::endl;
     return;
   }

   Alignments* globalPositions = new Alignments();

   AlignTransform tracker(AlignTransform::Translation(m_tracker.getParameter<double>("x"),
						      m_tracker.getParameter<double>("y"),
						      m_tracker.getParameter<double>("z")),
			  AlignTransform::EulerAngles(m_tracker.getParameter<double>("alpha"),
						      m_tracker.getParameter<double>("beta"),
						      m_tracker.getParameter<double>("gamma")),
			  DetId(DetId::Tracker).rawId());
   AlignTransform muon(AlignTransform::Translation(m_muon.getParameter<double>("x"),
						   m_muon.getParameter<double>("y"),
						   m_muon.getParameter<double>("z")),
		       AlignTransform::EulerAngles(m_muon.getParameter<double>("alpha"),
						   m_muon.getParameter<double>("beta"),
						   m_muon.getParameter<double>("gamma")),
		       DetId(DetId::Muon).rawId());
   AlignTransform ecal(AlignTransform::Translation(m_ecal.getParameter<double>("x"),
						   m_ecal.getParameter<double>("y"),
						   m_ecal.getParameter<double>("z")),
		       AlignTransform::EulerAngles(m_ecal.getParameter<double>("alpha"),
						   m_ecal.getParameter<double>("beta"),
						   m_ecal.getParameter<double>("gamma")),
		       DetId(DetId::Ecal).rawId());
   AlignTransform hcal(AlignTransform::Translation(m_hcal.getParameter<double>("x"),
						   m_hcal.getParameter<double>("y"),
						   m_hcal.getParameter<double>("z")),
		       AlignTransform::EulerAngles(m_hcal.getParameter<double>("alpha"),
						   m_hcal.getParameter<double>("beta"),
						   m_hcal.getParameter<double>("gamma")),
		       DetId(DetId::Hcal).rawId());


   std::cout << "Tracker (" << tracker.rawId() << ") at " << tracker.translation() 
	     << " " << tracker.rotation().eulerAngles() << std::endl;
   std::cout << tracker.rotation() << std::endl;

   std::cout << "Muon (" << muon.rawId() << ") at " << muon.translation() 
	     << " " << muon.rotation().eulerAngles() << std::endl;
   std::cout << muon.rotation() << std::endl;

   std::cout << "Ecal (" << ecal.rawId() << ") at " << ecal.translation() 
	     << " " << ecal.rotation().eulerAngles() << std::endl;
   std::cout << ecal.rotation() << std::endl;

   std::cout << "Hcal (" << hcal.rawId() << ") at " << hcal.translation()
	     << " " << hcal.rotation().eulerAngles() << std::endl;
   std::cout << hcal.rotation() << std::endl;

   globalPositions->m_align.push_back(tracker);
   globalPositions->m_align.push_back(muon);
   globalPositions->m_align.push_back(ecal);
   globalPositions->m_align.push_back(hcal);

   std::cout << "Uploading to the database..." << std::endl;

   edm::Service<cond::service::PoolDBOutputService> poolDbService;

   if (!poolDbService.isAvailable())
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
                  
//    if (poolDbService->isNewTagRequest("GlobalPositionRcd")) {
//       poolDbService->createNewIOV<Alignments>(&(*globalPositions), poolDbService->endOfTime(), "GlobalPositionRcd");
//    } else {
//       poolDbService->appendSinceTime<Alignments>(&(*globalPositions), poolDbService->currentTime(), "GlobalPositionRcd");
//    }
   poolDbService->writeOne<Alignments>(&(*globalPositions), 
				       poolDbService->currentTime(),
				       //poolDbService->beginOfTime(),
                                       "GlobalPositionRcd");



   std::cout << "done!" << std::endl;
   nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdWrite);
