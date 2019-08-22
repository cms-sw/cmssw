#include "FastSimulation/ForwardDetectors/plugins/FastSimDataFilter.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

using namespace std;
namespace cms {

  FastSimDataFilter::FastSimDataFilter(const edm::ParameterSet& conf) {
    towercut = conf.getUntrackedParameter<double>("towercut", 1.4);
  }

  FastSimDataFilter::~FastSimDataFilter() {}

  void FastSimDataFilter::beginJob() {
    ntotal = 0;
    npassed = 0;
  }
  void FastSimDataFilter::endJob() {
    cout << " FastSimDataFilter: accepted " << npassed << "  out of total " << ntotal << endl;
  }

  bool FastSimDataFilter::filter(edm::Event& event, const edm::EventSetup& setup) {
    ntotal++;
    bool result = false;

    edm::Handle<CaloTowerCollection> towers;
    event.getByLabel("towerMaker", towers);
    CaloTowerCollection::const_iterator cal;

    int nplusP = 0;
    int nminusP = 0;
    int nplusR = 0;
    int nminusR = 0;
    double metx = 0.;
    double mety = 0.;
    double sumEt = 0.;

    for (cal = towers->begin(); cal != towers->end(); ++cal) {
      //    double eE   = cal->emEnergy();
      //    double eH   = cal->hadEnergy();
      //    double eHO  = cal->outerEnergy();
      //    double phi  = cal->phi();

      double eta = cal->eta();
      double en = cal->energy();
      double et = cal->et();

      // cylindrical
      math::RhoEtaPhiVector mom(cal->et(), cal->eta(), cal->phi());

      /*  
    // cell properties    
    CaloTowerDetId idT = cal->id();
    int ieta = idT.ieta();
    if(ieta > 0) ieta -= 1;
    int iphi = idT.iphi();
    */

      metx += mom.x();
      mety += mom.y();
      sumEt += et;

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
      //    std::cout << "... passed" << std::endl;
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
