#include "../interface/OccupancyClient.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  OccupancyClient::OccupancyClient() :
    DQWorkerClient(),
    minHits_(0),
    deviationThreshold_(0.)
  {
    qualitySummaries_.insert("QualitySummary");
  }

  void
  OccupancyClient::setParams(edm::ParameterSet const& _params)
  {
    minHits_ = _params.getUntrackedParameter<int>("minHits");
    deviationThreshold_ = _params.getUntrackedParameter<double>("deviationThreshold");
  }

  void
  OccupancyClient::producePlots(ProcessType)
  {
    using namespace std;

    // number of allowed ieta indices
    // EE-: -28 to -1 with -27, -25 empty
    // EE+: 1 to 28 with 26, 28 empty
    unsigned const nPhiRings(56);

    MESet& meQualitySummary(MEs_.at("QualitySummary"));
//     MESet& meHotDigi(MEs_.at("HotDigi"));
//     MESet& meHotRecHitThr(MEs_.at("HotRecHitThr"));
//     MESet& meHotTPDigiThr(MEs_.at("HotTPDigiThr"));

    MESet const& sDigi(sources_.at("DigiAll"));
    MESet const& sRecHitThr(sources_.at("RecHitThrAll"));
    MESet const& sTPDigiThr(sources_.at("TPDigiThrAll"));

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR);

    double digiPhiRingMean[nPhiRings];
    std::fill_n(digiPhiRingMean, nPhiRings, 0.);
    double rechitPhiRingMean[nPhiRings];
    std::fill_n(rechitPhiRingMean, nPhiRings, 0.);
    int numCrystals[nPhiRings]; // this is static, but is easier to count now
    std::fill_n(numCrystals, nPhiRings, 0);

    MESet::const_iterator dEnd(sDigi.end());
    MESet::const_iterator rItr(sRecHitThr);
    for(MESet::const_iterator dItr(sDigi.beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      rItr = dItr;

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      DetId id(dItr->getId());
      int ieta(0);
      if(id.subdetId() == EcalTriggerTower) // barrel
        ieta = EcalTrigTowerDetId(id).ieta();
      else{
        std::vector<DetId> ids(scConstituents(EcalScDetId(id)));
        if(ids.size() == 0) continue;
        ieta = getTrigTowerMap()->towerOf(ids[0]).ieta();
      }

      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      digiPhiRingMean[index] += entries;
      rechitPhiRingMean[index] += rhentries;
      numCrystals[index] += 1;
    }

    for(unsigned ie(0); ie < nPhiRings; ie++){
      digiPhiRingMean[ie] /= numCrystals[ie];
      rechitPhiRingMean[ie] /= numCrystals[ie];
    }

    // Store # of entries for Occupancy analysis
    std::vector<float> Nentries(nDCC,0.);   // digis
    std::vector<float> Nrhentries(nDCC,0.); // (filtered) rechits

    // second round to find hot towers
    for(MESet::const_iterator dItr(sDigi.beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      DetId id(dItr->getId());

      bool doMask(meQualitySummary.maskMatches(id, mask, statusManager_));

      rItr = dItr;

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      int ieta(0);
      if(id.subdetId() == EcalTriggerTower) // barrel
        ieta = EcalTrigTowerDetId(id).ieta();
      else{
        std::vector<DetId> ids(scConstituents(EcalScDetId(id)));
        if(ids.size() == 0) continue;
        ieta = getTrigTowerMap()->towerOf(ids[0]).ieta();
      }

      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      int quality(doMask ? kMGood : kGood);

      if(entries > minHits_ && entries > digiPhiRingMean[index] * deviationThreshold_){
        //        meHotDigi->fill(id);
        quality = doMask ? kMBad : kBad;
      }
      if(rhentries > minHits_ && rhentries > rechitPhiRingMean[index] * deviationThreshold_){
        //        meHotRecHitThr->fill(id);
        quality = doMask ? kMBad : kBad;
      }

      meQualitySummary.setBinContent(id, double(quality));

      // Keep count of digis & rechits for Occupancy analysis
      unsigned iDCC( dccId(id)-1 );
      if ( entries   > minHits_ ) Nentries[iDCC]   += entries;
      if ( rhentries > minHits_ ) Nrhentries[iDCC] += rhentries;

    }

    double tpdigiPhiRingMean[nPhiRings];
    std::fill_n(tpdigiPhiRingMean, nPhiRings, 0.);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sTPDigiThr.getBinContent(ttid));

      unsigned index(ttid.ieta() < 0 ? ttid.ieta() + 28 : ttid.ieta() + 27);

      tpdigiPhiRingMean[index] += entries;
    }

    for(int ie(0); ie < 28; ie++){
      float denom(-1.);
      if(ie < 27) denom = 72.;
      else denom = 36.;
      tpdigiPhiRingMean[ie] /= denom;
      tpdigiPhiRingMean[55 - ie] /= denom;
    }

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      float entries(sTPDigiThr.getBinContent(ttid));

      unsigned index(ttid.ieta() < 0 ? ttid.ieta() + 28 : ttid.ieta() + 27);

      int quality(kGood);

      if(entries > minHits_ && entries > tpdigiPhiRingMean[index] * deviationThreshold_){
        //        meHotTPDigiThr.fill(ttid);
        quality = kBad;
      }

      if(quality != kBad) continue;

      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      for(unsigned iD(0); iD < ids.size(); ++iD){
        DetId& id(ids[iD]);

        int quality(meQualitySummary.getBinContent(id));
        if(quality == kMBad || quality == kBad) continue;

        meQualitySummary.setBinContent(id, meQualitySummary.maskMatches(id, mask, statusManager_) ? kMBad : kBad);
      }
    }

    // Quality check: set entire FED to BAD if its occupancy begins to vanish
    // Fill FED statistics from (filtered) RecHit Occupancy 
    float meanFEDEB(0), meanFEDEE(0), rmsFEDEB(0), rmsFEDEE(0);
    unsigned int nFEDEB(0), nFEDEE(0);
    for ( unsigned iDCC(0); iDCC < nDCC; iDCC++ ) {
      if ( iDCC >=kEBmLow && iDCC <= kEBpHigh) {
        meanFEDEB += Nrhentries[iDCC];
        rmsFEDEB  += Nrhentries[iDCC]*Nrhentries[iDCC];
        nFEDEB++;
      }
      else {
        meanFEDEE += Nrhentries[iDCC];
        rmsFEDEE  += Nrhentries[iDCC]*Nrhentries[iDCC];
        nFEDEE++;
      }
    }
    meanFEDEB /= float( nFEDEB ); rmsFEDEB /= float( nFEDEB );
    meanFEDEE /= float( nFEDEE ); rmsFEDEE /= float( nFEDEE );
    rmsFEDEB   = sqrt( abs(rmsFEDEB - meanFEDEB*meanFEDEB) );
    rmsFEDEE   = sqrt( abs(rmsFEDEE - meanFEDEE*meanFEDEE) );
    // Analyze FED statistics
    float meanFED(0.), rmsFED(0.), nRMS(5.);
    for ( MESet::iterator qsItr(meQualitySummary.beginChannel()); qsItr != meQualitySummary.end(); qsItr.toNextChannel() ) {
      DetId id( qsItr->getId() );
      unsigned iDCC( dccId(id)-1 );
      if ( iDCC >= kEBmLow && iDCC <= kEBpHigh ) {
        meanFED = meanFEDEB;
        rmsFED  = rmsFEDEB;
      }
      else {
        meanFED = meanFEDEE;
        rmsFED  = rmsFEDEE;
      }
      if ( meanFED > 1000. && Nrhentries[iDCC] < meanFED - nRMS*rmsFED )
        meQualitySummary.setBinContent( id, meQualitySummary.maskMatches(id, mask, statusManager_) ? kMBad : kBad );
    }

  } // producePlots()

  DEFINE_ECALDQM_WORKER(OccupancyClient);
}
