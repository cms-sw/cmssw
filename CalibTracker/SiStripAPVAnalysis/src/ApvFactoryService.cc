#include "CalibTracker/SiStripAPVAnalysis/interface/ApvFactoryService.h"

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

using namespace edm;
using namespace std;
//using namespace sistrip;

// -----------------------------------------------------------------------------
//

// -----------------------------------------------------------------------------
//

ApvFactoryService::ApvFactoryService(const edm::ParameterSet& pset, edm::ActivityRegistry& activity) : apvFactory_() {
  apvFactory_ = new ApvAnalysisFactory(pset);
  std::cout << " Print pedalgo inside ApvFactoryService constructor "
            << pset.getParameter<string>("CalculatorAlgorithm") << std::endl;
  //  activity.watchPostProcessEvent(this, &ApvFactoryService::postProcessEvent);
}

// -----------------------------------------------------------------------------
//
ApvFactoryService::~ApvFactoryService() {}

// -----------------------------------------------------------------------------

void ApvFactoryService::postProcessEvent(const edm::Event& ie, const edm::EventSetup& ies) {
  if (gotPed)
    return;

  /*
  std::cout << "ApvFactoryService::post" << std::endl;

  edm::ESHandle<SiStripPedestals> ped;
  ies.get<SiStripPedestalsRcd>().get(ped);

  //apvFactory_->SetPed(ped);

  gotPed=true;

 
  std::vector<uint32_t> pdetid;
  ped->getDetIds(pdetid);
  edm::LogInfo("SiStripO2O") << " Peds Found " << pdetid.size() << " DetIds";

  // pedDB_=ped;
  
 for (size_t id=0;id<pdetid.size();id++){
    SiStripPedestals::Range range=ped->getRange(pdetid[id]);
   //   SiStripPedestals::Range range=pedDB_.getRange(pdetid[id]);
    int strip=0;
   
   edm::LogInfo("SiStripO2O")  << "PED detid " << pdetid[id] << " \t"
			    << " strip " << strip << " \t"
			    << ped->getPed   (strip,range)   << " \t" 
			    << ped->getLowTh (strip,range)   << " \t" 
			    << ped->getHighTh(strip,range)   << " \t" 
			    << std::endl;     

  } 
 
*/
}

// -----------------------------------------------------------------------------
//

// -----------------------------------------------------------------------------
//

int ApvFactoryService::getEventInitNumber() { return 0; }
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
//

ApvAnalysisFactory* const ApvFactoryService::getApvFactory() const { return apvFactory_; }
