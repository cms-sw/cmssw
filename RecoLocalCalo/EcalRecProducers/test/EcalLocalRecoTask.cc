/*
 * \file EcalLocalReco.cc
 *
 *
*/

#include <RecoLocalCalo/EcalRecProducers/test/EcalLocalRecoTask.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

EcalLocalRecoTask::EcalLocalRecoTask(const edm::ParameterSet& ps) {
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");

  EBrecHitToken_ = consumes<EBRecHitCollection>(ps.getParameter<edm::InputTag>("ebrechits"));
  EErecHitToken_ = consumes<EERecHitCollection>(ps.getParameter<edm::InputTag>("eerechits"));
  ESrecHitToken_ = consumes<ESRecHitCollection>(ps.getParameter<edm::InputTag>("esrechits"));

  EBurecHitToken_ = consumes<EBUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("eburechits"));
  EEurecHitToken_ = consumes<EEUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("eeurechits"));

  EBdigiToken_ = consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("ebdigis"));
  EEdigiToken_ = consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("eedigis"));
  ESdigiToken_ = consumes<ESDigiCollection>(ps.getParameter<edm::InputTag>("esdigis"));

  cfToken_ = consumes<CrossingFrame<PCaloHit>>(edm::InputTag("mix", "EcalHitsEB"));

  if (outputFile_.size() != 0) {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "histograms will NOT be saved";
  }

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if (verbose_) {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "verbose switch is ON";
  } else {
    edm::LogInfo("EcalLocalRecoTaskInfo") << "verbose switch is OFF";
  }

  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

  meEBUncalibRecHitMaxSampleRatio_ = 0;
  meEBUncalibRecHitPedestal_ = 0;
  meEBUncalibRecHitOccupancy_ = 0;
  meEBRecHitSimHitRatio_ = 0;

  Char_t histo[70];

  if (dbe_) {
    dbe_->setCurrentFolder("EcalLocalRecoTask");

    sprintf(histo, "EcalLocalRecoTask Barrel occupancy");
    meEBUncalibRecHitOccupancy_ = dbe_->book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

    sprintf(histo, "EcalLocalRecoTask Barrel Reco pedestals");
    meEBUncalibRecHitPedestal_ = dbe_->book1D(histo, histo, 1000, 0., 1000.);

    sprintf(histo, "EcalLocalRecoTask RecHit Max Sample Ratio");
    meEBUncalibRecHitMaxSampleRatio_ = dbe_->book1D(histo, histo, 200, 0., 2.);

    sprintf(histo, "EcalLocalRecoTask RecHit SimHit Ratio");
    meEBRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 200, 0., 2.);
  }
}

EcalLocalRecoTask::~EcalLocalRecoTask() {
  if (outputFile_.size() != 0 && dbe_)
    dbe_->save(outputFile_);
}

void EcalLocalRecoTask::beginJob() {}

void EcalLocalRecoTask::endJob() {}

void EcalLocalRecoTask::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::LogInfo("EcalLocalRecoTaskInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  edm::Handle<EBDigiCollection> pEBDigis;
  edm::Handle<EEDigiCollection> pEEDigis;
  edm::Handle<ESDigiCollection> pESDigis;

  e.getByToken(EBdigiToken_, pEBDigis);
  e.getByToken(EEdigiToken_, pEEDigis);
  e.getByToken(ESdigiToken_, pESDigis);

  edm::Handle<EBUncalibratedRecHitCollection> pEBUncalibRecHit;
  edm::Handle<EEUncalibratedRecHitCollection> pEEUncalibRecHit;

  e.getByToken(EBurecHitToken_, pEBUncalibRecHit);
  e.getByToken(EEurecHitToken_, pEEUncalibRecHit);

  edm::Handle<EBRecHitCollection> pEBRecHit;
  edm::Handle<EERecHitCollection> pEERecHit;
  edm::Handle<ESRecHitCollection> pESRecHit;

  e.getByToken(EBrecHitToken_, pEBRecHit);
  e.getByToken(EBrecHitToken_, pEERecHit);
  e.getByToken(ESrecHitToken_, pESRecHit);

  edm::Handle<CrossingFrame<PCaloHit>> crossingFrame;

  e.getByToken(cfToken_, crossingFrame);

  edm::ESHandle<EcalPedestals> pPeds;
  c.get<EcalPedestalsRcd>().get(pPeds);

  auto barrelHits = std::make_unique<MixCollection<PCaloHit>>(crossingFrame.product());

  MapType EBSimMap;

  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin(); hitItr != barrelHits->end(); ++hitItr) {
    EBDetId EBid = EBDetId(hitItr->id());

    LogDebug("EcalLocalRecoTaskDebug") << " CaloHit " << hitItr->getName() << " DetID = " << hitItr->id() << "\n"
                                       << "Energy = " << hitItr->energy() << " Time = " << hitItr->time() << "\n"
                                       << "EBDetId = " << EBid.ieta() << " " << EBid.iphi();

    uint32_t crystid = EBid.rawId();
    EBSimMap[crystid] += hitItr->energy();
  }

  const EBDigiCollection* EBDigi = pEBDigis.product();
  const EBUncalibratedRecHitCollection* EBUncalibRecHit = pEBUncalibRecHit.product();
  const EBRecHitCollection* EBRecHit = pEBRecHit.product();

  // loop over uncalibRecHit
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin();
       uncalibRecHit != EBUncalibRecHit->end();
       ++uncalibRecHit) {
    EBDetId EBid = EBDetId(uncalibRecHit->id());
    if (meEBUncalibRecHitOccupancy_)
      meEBUncalibRecHitOccupancy_->Fill(EBid.iphi(), EBid.ieta());
    if (meEBUncalibRecHitPedestal_)
      meEBUncalibRecHitPedestal_->Fill(uncalibRecHit->pedestal());

    // Find corresponding recHit
    EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
    // Find corresponding digi
    EBDigiCollection::const_iterator myDigi = EBDigi->find(EBid);

    double eMax = 0.;

    if (myDigi != EBDigi->end()) {
      for (unsigned int sample = 0; sample < myDigi->size(); ++sample) {
        double analogSample = EcalMGPASample((*myDigi)[sample]).adc();
        if (eMax < analogSample) {
          eMax = analogSample;
        }
      }
    } else
      continue;

    const EcalPedestals* myped = pPeds.product();
    EcalPedestals::const_iterator it = myped->getMap().find(EBid);
    if (it != myped->getMap().end()) {
      if (eMax > (*it).mean_x1 + 5 * (*it).rms_x1)  //only real signal RecHit
      {
        if (meEBUncalibRecHitMaxSampleRatio_)
          meEBUncalibRecHitMaxSampleRatio_->Fill((uncalibRecHit->amplitude() + uncalibRecHit->pedestal()) / eMax);
        edm::LogInfo("EcalLocalRecoTaskInfo")
            << " eMax = " << eMax << " Amplitude = " << uncalibRecHit->amplitude() + uncalibRecHit->pedestal();
      } else
        continue;
    } else
      continue;

    if (myRecHit != EBRecHit->end()) {
      if (EBSimMap[EBid.rawId()] != 0.)
        if (meEBRecHitSimHitRatio_)
          meEBRecHitSimHitRatio_->Fill(myRecHit->energy() / EBSimMap[EBid.rawId()]);
    } else
      continue;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLocalRecoTask);
