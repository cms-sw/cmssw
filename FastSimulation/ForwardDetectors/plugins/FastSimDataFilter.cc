#include "FastSimulation/ForwardDetectors/plugins/FastSimDataFilter.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

using namespace std;
namespace cms {

  FastSimDataFilter::FastSimDataFilter(const edm::ParameterSet& conf)
      : tokTowers_(consumes<CaloTowerCollection>(edm::InputTag("towerMaker"))),
        towercut(conf.getUntrackedParameter<double>("towercut", 1.4)) {}

  void FastSimDataFilter::beginJob() {
    ntotal = 0;
    npassed = 0;
  }
  void FastSimDataFilter::endJob() {
    edm::LogVerbatim("FastSimDataFilter") << " FastSimDataFilter: accepted " << npassed << "  out of total " << ntotal;
  }

  bool FastSimDataFilter::filter(edm::Event& event, const edm::EventSetup& setup) {
    ntotal++;
    bool result = false;

    const edm::Handle<CaloTowerCollection>& towers = event.getHandle(tokTowers_);
    CaloTowerCollection::const_iterator cal;

    int nplusP = 0;
    int nminusP = 0;
    int nplusR = 0;
    int nminusR = 0;

    for (cal = towers->begin(); cal != towers->end(); ++cal) {
      //    double eE   = cal->emEnergy();
      //    double eH   = cal->hadEnergy();
      //    double eHO  = cal->outerEnergy();
      //    double phi  = cal->phi();

      double eta = cal->eta();
      double en = cal->energy();

      // cylindrical
      math::RhoEtaPhiVector mom(cal->et(), cal->eta(), cal->phi());

      /*  
    // cell properties    
    CaloTowerDetId idT = cal->id();
    int ieta = idT.ieta();
    if(ieta > 0) ieta -= 1;
    int iphi = idT.iphi();
    */

      // Towers approx. in the region of BSC Paddles and Rings
      if (en > towercut && eta > 3.1 && eta < 3.5)
        nplusP++;
      if (en > towercut && eta < -3.1 && eta > -3.5)
        nminusP++;
      if (en > towercut && eta > 3.9 && eta < 4.6)
        nplusR++;
      if (en > towercut && eta < -3.9 && eta > -4.6)
        nminusR++;

    }  // towers cycle

    if ((nplusP * nminusP >= 1) || (nplusR * nminusR >= 1) || (nplusP * nminusR >= 1) || (nplusR * nminusP >= 1)) {
      result = true;
      npassed++;
    }

    return result;
  }
}  // namespace cms

// define this class as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
using namespace cms;
DEFINE_FWK_MODULE(FastSimDataFilter);
