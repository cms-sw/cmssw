// Last commit: $Id: ApvFactoryService.cc,v 1.33 2007/11/07 15:55:33 bainbrid Exp $


#include "CalibTracker/SiStripAPVAnalysis/interface/PedFactoryService.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"


using namespace edm;
using namespace std;
//using namespace sistrip;

// -----------------------------------------------------------------------------
// 

// -----------------------------------------------------------------------------
// 

PedFactoryService::PedFactoryService( const edm::ParameterSet& pset,
				      edm::ActivityRegistry& activity ) :
  pedDB_(),
  gotPed(false)
{
 
 
  std::cout << " In principle should read form CFG db true or false " << pset.getParameter<bool>("useDB") << std::endl;

  activity.watchPostProcessEvent(this, &PedFactoryService::postProcessEvent);

  


}

// -----------------------------------------------------------------------------
//
PedFactoryService::~PedFactoryService() {
}

// -----------------------------------------------------------------------------


void PedFactoryService::postProcessEvent(const edm::Event& ie, const edm::EventSetup& ies){



  if (gotPed) 
    return;

  std::cout << "PedFactoryService::postProcessEvent" << std::endl;
 
  edm::ESHandle<SiStripPedestals> ped;
  ies.get<SiStripPedestalsRcd>().get(ped);

  //apvFactory_->SetPed(ped);

  gotPed=true;

  pedDB_=ped.product();

  /*
  std::vector<uint32_t> pdetid;
  ped->getDetIds(pdetid);

  cout << "in PedFactoryService::postProcessEvent Peds Found " << pdetid.size() << " DetIds";

 for (size_t id=0;id<pdetid.size();id++){
    SiStripPedestals::Range range=ped->getRange(pdetid[id]);
   //   SiStripPedestals::Range range=pedDB_.getRange(pdetid[id]);
    int strip=0;
   
   //  edm::LogInfo("SiStripO2O")  << "PED detid " << pdetid[id] << " \t"
   std::cout  << "PED detid " << pdetid[id] << " \t"
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




const SiStripPedestals*  PedFactoryService::getPedDB() const {
 
return pedDB_;

}








  
