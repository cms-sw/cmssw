
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "FWCore/Utilities/interface/typelookup.h"  ///// added as a try for CaloTopology


class EcalLaserCorrFilter : public edm::EDFilter {

  public:

    explicit EcalLaserCorrFilter(const edm::ParameterSet & iConfig);
    ~EcalLaserCorrFilter() {}

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    edm::EDGetTokenT<EcalRecHitCollection> ebRHSrcToken_;
    edm::EDGetTokenT<EcalRecHitCollection> eeRHSrcToken_;

    // thresholds to laser corr to set kPoorCalib
    const double EBLaserMIN_, EELaserMIN_, EBLaserMAX_, EELaserMAX_, EBEnegyMIN_, EEEnegyMIN_;
    const bool   taggingMode_, debug_;
};


EcalLaserCorrFilter::EcalLaserCorrFilter(const edm::ParameterSet & iConfig)
  : ebRHSrcToken_      (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitSource")))
  , eeRHSrcToken_      (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , EBLaserMIN_   (iConfig.getParameter<double>("EBLaserMIN"))
  , EELaserMIN_   (iConfig.getParameter<double>("EELaserMIN"))
  , EBLaserMAX_   (iConfig.getParameter<double>("EBLaserMAX"))
  , EELaserMAX_   (iConfig.getParameter<double>("EELaserMAX"))
  , EBEnegyMIN_   (iConfig.getParameter<double>("EBEnegyMIN"))
  , EEEnegyMIN_   (iConfig.getParameter<double>("EEEnegyMIN"))
  , taggingMode_  (iConfig.getParameter<bool>("taggingMode"))
  , debug_        (iConfig.getParameter<bool>("Debug"))
{
  produces<bool>();
}


bool EcalLaserCorrFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {

   using namespace edm;
   using namespace reco;
   using namespace std;

  edm::Handle<EcalRecHitCollection> ebRHs;
  iEvent.getByToken(ebRHSrcToken_, ebRHs);

  edm::Handle<EcalRecHitCollection> eeRHs;
  iEvent.getByToken(eeRHSrcToken_, eeRHs);

  // Laser corrections
  edm::ESHandle<EcalLaserDbService> laser;
  iSetup.get<EcalLaserDbRecord>().get(laser);


  bool goodCalib = true;

  // check EE RecHits
  for (EcalRecHitCollection::const_iterator eerh = eeRHs->begin(); eerh != eeRHs->end(); ++eerh) {

    EcalRecHit hit    = (*eerh);
    EEDetId    eeDet  = hit.id();
    double     energy = eerh->energy();
    double     time   = eerh->time();
    int        jx     = EEDetId((*eerh).id()).ix();
    int        jy     = EEDetId((*eerh).id()).iy();
    int        jz     = EEDetId((*eerh).id()).zside();

    // get laser coefficient
    float lasercalib = laser->getLaserCorrection( EEDetId(eeDet), iEvent.time());

    if( energy>EEEnegyMIN_ && (lasercalib < EELaserMIN_ || lasercalib > EELaserMAX_) ) {
      goodCalib = false;
      if(debug_) {
	std::cout << "RecHit EE "
		  << iEvent.id().run()<< ":" << iEvent.luminosityBlock() <<":"<<iEvent.id().event()
		  << " lasercalib " << lasercalib << " rechit ene " << energy << " time " << time
		  << " ix, iy, z = " << jx << " " << jy  << " " << jz
		  << std::endl;
      }
    }

    //if (!goodCalib) break;
  }

  // check EB RecHits
  for (EcalRecHitCollection::const_iterator ebrh = ebRHs->begin(); ebrh != ebRHs->end(); ++ebrh) {

    EcalRecHit hit    = (*ebrh);
    EBDetId    ebDet  = hit.id();
    double     energy = ebrh->energy();
    double     time   = ebrh->time();
    int etarec        = EBDetId((*ebrh).id()).ieta();
    int phirec        = EBDetId((*ebrh).id()).iphi();
    int zrec          = EBDetId((*ebrh).id()).zside();


    // get laser coefficient
    float lasercalib = laser->getLaserCorrection( EBDetId(ebDet), iEvent.time());

    if (energy>EBEnegyMIN_ && (lasercalib < EBLaserMIN_ || lasercalib > EBLaserMAX_) ) {
      goodCalib = false;
      if(debug_) {
	std::cout << "RecHit EB "
		  << iEvent.id().run()<< ":" << iEvent.luminosityBlock() <<":"<<iEvent.id().event()
		  << " lasercalib " << lasercalib << " rechit ene " << energy << " time " << time
		  << " eta, phi, z = " << etarec << " " << phirec  << " " << zrec
		  << std::endl;
      }
    }
    //if (!goodCalib) break;
  }


  bool result = goodCalib;
  //std::cout << " *********** Result ******** " << result << std::endl;

  iEvent.put( std::auto_ptr<bool>(new bool(result)) );

  return taggingMode_ || result;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EcalLaserCorrFilter);
