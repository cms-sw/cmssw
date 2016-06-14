#include "../interface/TrigPrimClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm
{
  TrigPrimClient::TrigPrimClient() :
    DQWorkerClient(),
    minEntries_(0),
    errorFractionThreshold_(0.)
  {
    qualitySummaries_.insert("EmulQualitySummary");
  }
  
  void
  TrigPrimClient::setParams(edm::ParameterSet const& _params)
  {
    minEntries_ = _params.getUntrackedParameter<int>("minEntries");
    errorFractionThreshold_ = _params.getUntrackedParameter<double>("errorFractionThreshold");
  }

  void
  TrigPrimClient::producePlots(ProcessType)
  {
    MESet& meTimingSummary(MEs_.at("TimingSummary"));
    MESet& meNonSingleSummary(MEs_.at("NonSingleSummary"));
    MESet& meEmulQualitySummary(MEs_.at("EmulQualitySummary"));

    MESet const& sEtEmulError(sources_.at("EtEmulError"));
    MESet const& sMatchedIndex(sources_.at("MatchedIndex"));
    MESet const& sTPDigiThrAll(sources_.at("TPDigiThrAll"));

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    // Store # of entries for Occupancy analysis
    std::vector<float> Nentries(nDCC,0.);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      bool doMask(meEmulQualitySummary.maskMatches(ttid, mask, statusManager_));

      float towerEntries(0.);
      float tMax(0.5);
      float nMax(0.);
      for(int iBin(0); iBin < 6; iBin++){
	float entries(sMatchedIndex.getBinContent(ttid, iBin + 1));
	towerEntries += entries;

	if(entries > nMax){
	  nMax = entries;
	  tMax = iBin == 0 ? -0.5 : iBin + 0.5; // historical reasons.. much clearer to say "no entry = -0.5"
	}
      }

      meTimingSummary.setBinContent(ttid, tMax);

      if(towerEntries < minEntries_){
	meEmulQualitySummary.setBinContent(ttid, doMask ? kMUnknown : kUnknown);
	continue;
      }

      float nonsingleFraction(1. - nMax / towerEntries);

      if(nonsingleFraction > 0.)
	meNonSingleSummary.setBinContent(ttid, nonsingleFraction);

      if(sEtEmulError.getBinContent(ttid) / towerEntries > errorFractionThreshold_)
        meEmulQualitySummary.setBinContent(ttid, doMask ? kMBad : kBad);
      else
        meEmulQualitySummary.setBinContent(ttid, doMask ? kMGood : kGood);
      
      // Keep count for Occupancy analysis
      unsigned iDCC( dccId(ttid)-1 );
      Nentries[iDCC] += sTPDigiThrAll.getBinContent(ttid); 
    }

    // Fill TTF4 v Masking ME
    // NOT an occupancy plot: only tells you if non-zero TTF4 occupancy was seen
    // without giving info about how many were seen
    MESet& meTTF4vMask(MEs_.at("TTF4vMask"));
    MESet const& sTTFlags4(sources_.at("TTFlags4"));
    MESet const& sTTMaskMapAll(sources_.at("TTMaskMapAll"));
    
    // Loop over all TTs
    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++) {
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      bool isMasked( sTTMaskMapAll.getBinContent(ttid) > 0. );
      bool hasTTF4( sTTFlags4.getBinContent(ttid) > 0. );
      if ( isMasked ) { 
        if ( hasTTF4 )
          meTTF4vMask.setBinContent( ttid,12 ); // Masked, has TTF4
        else
          meTTF4vMask.setBinContent( ttid,11 ); // Masked, no TTF4
      } else {
        if ( hasTTF4 )
          meTTF4vMask.setBinContent( ttid,13 ); // not Masked, has TTF4
      }   
    } // TT loop 

    // Quality check: set an entire FED to BAD if an "entire" FED shows any DCC-SRP flag mismatch errors
    // Fill flag mismatch statistics
    std::vector<float> nTTs(nDCC,0.);
    std::vector<float> nTTFMismath(nDCC,0.);
    MESet const& sTTFMismatch(sources_.at("TTFMismatch"));
    for ( unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++ ) {
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      unsigned iDCC( dccId(ttid)-1 );
      if ( sTTFMismatch.getBinContent(ttid) > 0. )
          nTTFMismath[iDCC]++;
      nTTs[iDCC]++;
    }
    // Analyze flag mismatch statistics
    for ( unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++ ) {
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      unsigned iDCC( dccId(ttid)-1 );
      if ( nTTFMismath[iDCC] > 0.8*nTTs[iDCC] ) // "entire" => 80%
        meEmulQualitySummary.setBinContent( ttid, meEmulQualitySummary.maskMatches(ttid, mask, statusManager_) ? kMBad : kBad );
    }

    // Quality check: set entire FED to BAD if its occupancy begins to vanish
    // Fill FED statistics from TP digi occupancy
    float meanFEDEB(0.), meanFEDEE(0.), rmsFEDEB(0.), rmsFEDEE(0.);
    unsigned int nFEDEB(0), nFEDEE(0);
    for ( unsigned iDCC(0); iDCC < nDCC; iDCC++ ) {
      if ( iDCC >=kEBmLow && iDCC <= kEBpHigh) {
        meanFEDEB += Nentries[iDCC];
        rmsFEDEB  += Nentries[iDCC]*Nentries[iDCC];
        nFEDEB++;
      }
      else {
        meanFEDEE += Nentries[iDCC];
        rmsFEDEE  += Nentries[iDCC]*Nentries[iDCC];
        nFEDEE++;
      }
    }
    meanFEDEB /= float( nFEDEB ); rmsFEDEB /= float( nFEDEB );
    meanFEDEE /= float( nFEDEE ); rmsFEDEE /= float( nFEDEE );
    rmsFEDEB   = sqrt( abs(rmsFEDEB - meanFEDEB*meanFEDEB) );
    rmsFEDEE   = sqrt( abs(rmsFEDEE - meanFEDEE*meanFEDEE) );
    // Analyze FED statistics
    float meanFED(0.), rmsFED(0.), nRMS(5.);
    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      unsigned iDCC( dccId(ttid)-1 );
      if ( iDCC >= kEBmLow && iDCC <= kEBpHigh ) {
        meanFED = meanFEDEB;
        rmsFED  = rmsFEDEB;
      }
      else {
        meanFED = meanFEDEE;
        rmsFED  = rmsFEDEE;
      }
      if ( meanFED > 100. && Nentries[iDCC] < meanFED - nRMS*rmsFED )
        meEmulQualitySummary.setBinContent( ttid, meEmulQualitySummary.maskMatches(ttid, mask, statusManager_) ? kMBad : kBad );
    }

  } // producePlots()

  DEFINE_ECALDQM_WORKER(TrigPrimClient);
}
