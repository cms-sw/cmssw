// my includes
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Calibration/Tools/plugins/SingleEleCalibSelector.h"

//CLHEP
#include <CLHEP/Vector/LorentzVector.h>

#include <iostream>

SingleEleCalibSelector::SingleEleCalibSelector(const edm::ParameterSet& iConfig) {
  ESCOPinMin_ = iConfig.getParameter<double>("ESCOPinMin");
  ESCOPinMax_ = iConfig.getParameter<double>("ESCOPinMax");
  ESeedOPoutMin_ = iConfig.getParameter<double>("ESeedOPoutMin");
  ESeedOPoutMax_ = iConfig.getParameter<double>("ESeedOPoutMax");
  PinMPoutOPinMin_ = iConfig.getParameter<double>("PinMPoutOPinMin");
  PinMPoutOPinMax_ = iConfig.getParameter<double>("PinMPoutOPinMax");
  E5x5OPoutMin_ = iConfig.getParameter<double>("E5x5OPoutMin");
  E5x5OPoutMax_ = iConfig.getParameter<double>("E5x5OPoutMax");
  E3x3OPinMin_ = iConfig.getParameter<double>("E3x3OPinMin");
  E3x3OPinMax_ = iConfig.getParameter<double>("E3x3OPinMax");
  E3x3OE5x5Min_ = iConfig.getParameter<double>("E3x3OE5x5Min");
  E3x3OE5x5Max_ = iConfig.getParameter<double>("E3x3OE5x5Max");
  EBrecHitLabel_ = iConfig.getParameter<edm::InputTag>("alcaBarrelHitCollection");
  EErecHitLabel_ = iConfig.getParameter<edm::InputTag>("alcaEndcapHitCollection");
}

// ------------------------------------------------------------

void 

SingleEleCalibSelector::select (edm::Handle<collection> inputHandle, 
			const edm::Event& iEvent , const edm::EventSetup& iSetup) 
{
  selected_.clear();

  //Get the EB rechit collection
  edm::Handle<EBRecHitCollection> barrelRecHitsHandle;
  iEvent.getByLabel(EBrecHitLabel_, barrelRecHitsHandle);
  const EBRecHitCollection* EBHitsColl = barrelRecHitsHandle.product();

  //Get the EE rechit collection
  edm::Handle<EERecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel(EErecHitLabel_, endcapRecHitsHandle);
  const EERecHitCollection* EEHitsColl = endcapRecHitsHandle.product();

  //To deal with Geometry
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);

  //Loop over electrons
  for (collection::const_iterator ele = (*inputHandle).begin(); ele != (*inputHandle).end(); ++ele) {
    //Find DetID max hit
    DetId maxHitId = findMaxHit((*ele).superCluster()->hitsAndFractions(), EBHitsColl, EEHitsColl);

    if (maxHitId.null()) {
      std::cout << " Null Id" << std::endl;
      continue;
    }

    // Find 3x3 and 5x5 windows around xtal max and compute E3x3,E5x5
    double E5x5 = 0.;
    double E3x3 = 0.;

    const CaloSubdetectorTopology* topology = theCaloTopology->getSubdetectorTopology(DetId::Ecal, maxHitId.subdetId());
    int WindowSize = 5;
    std::vector<DetId> m5x5aroundMax = topology->getWindow(maxHitId, WindowSize, WindowSize);
    E5x5 = EnergyNxN(m5x5aroundMax, EBHitsColl, EEHitsColl);
    WindowSize = 3;
    std::vector<DetId> m3x3aroundMax = topology->getWindow(maxHitId, WindowSize, WindowSize);
    E3x3 = EnergyNxN(m3x3aroundMax, EBHitsColl, EEHitsColl);

    double pin = ele->trackMomentumAtVtx().R();
    double piMpoOpi = (pin - ele->trackMomentumOut().R()) / pin;
    double E5x5OPout = E5x5 / ele->trackMomentumOut().R();
    double E3x3OPin = E3x3 / pin;
    double E3x3OE5x5 = E3x3 / E5x5;
    double EseedOPout = ele->eSeedClusterOverPout();
    double EoPin = ele->eSuperClusterOverP();

    if (piMpoOpi > PinMPoutOPinMin_ && piMpoOpi < PinMPoutOPinMax_ && EseedOPout > ESeedOPoutMin_ &&
        EseedOPout < ESeedOPoutMax_ && EoPin > ESCOPinMin_ && EoPin < ESCOPinMax_ && E5x5OPout > E5x5OPoutMin_ &&
        E5x5OPout < E5x5OPoutMax_ && E3x3OPin > E3x3OPinMin_ && E3x3OPin < E3x3OPinMax_ && E3x3OE5x5 > E3x3OE5x5Min_ &&
        E3x3OE5x5 < E3x3OE5x5Max_) {
      selected_.push_back(&(*ele));
    }
  }

  return;
}

// ------------------------------------------------------------

SingleEleCalibSelector::~SingleEleCalibSelector() {}

// ------------------------------------------------------------

// To find Max Hit
DetId SingleEleCalibSelector::findMaxHit(const std::vector<std::pair<DetId, float> >& v1,
                                         const EBRecHitCollection* EBhits,
                                         const EERecHitCollection* EEhits) {
  double currEnergy = 0.;
  DetId maxHit;
  for (std::vector<std::pair<DetId, float> >::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
    if (idsIt->first.subdetId() == EcalBarrel) {
      EBRecHitCollection::const_iterator itrechit;
      itrechit = EBhits->find((*idsIt).first);
      if (itrechit == EBhits->end()) {
        edm::LogInfo("reading") << "[findMaxHit] rechit not found! ";
        continue;
      }
      //FIXME: use fraction ??
      if (itrechit->energy() > currEnergy) {
        currEnergy = itrechit->energy();
        maxHit = (*idsIt).first;
      }
    } else {
      EERecHitCollection::const_iterator itrechit;
      itrechit = EEhits->find((*idsIt).first);
      if (itrechit == EEhits->end()) {
        edm::LogInfo("reading") << "[findMaxHit] rechit not found! ";
        continue;
      }

      if (itrechit->energy() > currEnergy) {
        currEnergy = itrechit->energy();
        maxHit = (*idsIt).first;
      }
    }
  }
  return maxHit;
}

// Energy in a window NxN
double SingleEleCalibSelector::EnergyNxN(const std::vector<DetId>& vNxN,
                                         const EBRecHitCollection* EBhits,
                                         const EERecHitCollection* EEhits) {
  double dummy = 0.;
  int window_size = vNxN.size();
  for (int ixtal = 0; ixtal < window_size; ixtal++) {
    if (vNxN[ixtal].subdetId() == EcalBarrel) {
      EBRecHitCollection::const_iterator it_rechit;
      it_rechit = EBhits->find(vNxN[ixtal]);
      dummy += it_rechit->energy();
    } else {
      EERecHitCollection::const_iterator it_rechit;
      it_rechit = EEhits->find(vNxN[ixtal]);
      dummy += it_rechit->energy();
    }
  }

  return dummy;
}
