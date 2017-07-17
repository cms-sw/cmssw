#include "../interface/IntegrityClient.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm
{
  IntegrityClient::IntegrityClient() :
    DQWorkerClient(),
    errFractionThreshold_(0.)
  {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  IntegrityClient::setParams(edm::ParameterSet const& _params)
  {
    errFractionThreshold_ = _params.getUntrackedParameter<double>("errFractionThreshold");
  }

  // Check Channel Status Record at every endLumi
  // Used to fill Channel Status Map MEs
  void
  IntegrityClient::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& _es)
  {
    _es.get<EcalChannelStatusRcd>().get( chStatus );
  }

  void
  IntegrityClient::producePlots(ProcessType)
  {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR |
                  1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
                  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR |
                  1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    MESet& meQuality(MEs_.at("Quality"));
    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meChStatus( MEs_.at("ChStatus") );

    MESet const& sOccupancy(sources_.at("Occupancy"));
    MESet const& sGain(sources_.at("Gain"));
    MESet const& sChId(sources_.at("ChId"));
    MESet const& sGainSwitch(sources_.at("GainSwitch"));
    MESet const& sTowerId(sources_.at("TowerId"));
    MESet const& sBlockSize(sources_.at("BlockSize"));

    // Fill Channel Status Map MEs
    // Record is checked for updates at every endLumi and filled here
    MESet::iterator chSEnd( meChStatus.end() );
    for( MESet::iterator chSItr(meChStatus.beginChannel()); chSItr != chSEnd; chSItr.toNextChannel() ){

      DetId id( chSItr->getId() );

      EcalChannelStatusMap::const_iterator chIt(0);

      // Set appropriate channel map (EB or EE)
      if( id.subdetId() == EcalBarrel ){
        EBDetId ebid(id);
        chIt = chStatus->find( ebid );
      }
      else {
        EEDetId eeid(id);
        chIt = chStatus->find( eeid );
      }

      // Get status code and fill ME
      if ( chIt != chStatus->end() ){
        uint16_t code( chIt->getEncodedStatusCode() );
        chSItr->setBinContent( code );
      }

    } // Channel Status Map

    MESet::iterator qEnd(meQuality.end());
    MESet::const_iterator occItr(sOccupancy);
    for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      occItr = qItr;

      DetId id(qItr->getId());

      bool doMask(meQuality.maskMatches(id, mask, statusManager_));

      float entries(occItr->getBinContent());

      float gain(sGain.getBinContent(id));
      float chid(sChId.getBinContent(id));
      float gainswitch(sGainSwitch.getBinContent(id));

      float towerid(sTowerId.getBinContent(id));
      float blocksize(sBlockSize.getBinContent(id));

      if(entries + gain + chid + gainswitch + towerid + blocksize < 1.){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
        continue;
      }

      float chErr((gain + chid + gainswitch + towerid + blocksize) / (entries + gain + chid + gainswitch + towerid + blocksize));

      if(chErr > errFractionThreshold_){
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
      }
      else{
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
      }
    }

    // Quality check: set an entire FED to BAD if "any" DCC-SRP or DCC-TCC mismatch errors are detected
    // Fill mismatch statistics
    MESet const& sBXSRP(sources_.at("BXSRP"));
    MESet const& sBXTCC(sources_.at("BXTCC"));
    std::vector<bool> hasMismatchDCC(nDCC,false);
    for ( unsigned iDCC(0); iDCC < nDCC; ++iDCC ) {
      if ( sBXSRP.getBinContent(iDCC + 1) > 50. || sBXTCC.getBinContent(iDCC + 1) > 50. ) // "any" => 50
        hasMismatchDCC[iDCC] = true;
    }
    // Analyze mismatch statistics
    for ( MESet::iterator qsItr(meQualitySummary.beginChannel()); qsItr != meQualitySummary.end(); qsItr.toNextChannel() ) {
      DetId id( qsItr->getId() );
      unsigned iDCC( dccId(id)-1 );
      if ( hasMismatchDCC[iDCC] )
        meQualitySummary.setBinContent( id, meQualitySummary.maskMatches(id, mask, statusManager_) ? kMBad : kBad );
    }

  } // producePlots()

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}
