
#include "DQM/SiStripCommissioningSources/interface/PedsFullNoiseTask.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/lexical_cast.hpp"


// -----------------------------------------------------------------------------
//
PedsFullNoiseTask::PedsFullNoiseTask(DQMStore * dqm, const FedChannelConnection & conn, const edm::ParameterSet & pset) :
  CommissioningTask(dqm, conn, "PedsFullNoiseTask"),
  nstrips_(256)
{
  LogTrace(sistrip::mlDqmSource_)
    << "[PedsFullNoiseTask::" << __func__ << "]"
    << " Constructing object...";
  edm::ParameterSet params = pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters");
  nskip_ = params.getParameter<int>("NrEvToSkipAtStart");
  skipped_ = false,
  ntempstab_ = params.getParameter<int>("NrEvUntilStable");
  tempstable_ = false;
  nadcnoise_ = params.getParameter<int>("NrPosBinsNoiseHist");
}

// -----------------------------------------------------------------------------
//
PedsFullNoiseTask::~PedsFullNoiseTask()
{
  LogTrace(sistrip::mlDqmSource_)
    << "[PedsFullNoiseTask::" << __func__ << "]"
    << " Destructing object...";

  // Readjust noise measurements
  // Here we correct for the drift in the pedestal (the rough pedestal was used
  //   as a reference for the noise measurement) and for the fact that the mean
  //   of the noise histogram is also not correct because since it is filled
  //   with SetBinContent() rather than with Fill()
  // In principle we could just take the mean of the histogram (meanrough below)
  //   and shift to mean 0, but to do things properly in cases of non-
  //   Gaussian noise we calculate the shift to be applied as below, using:
  //     1) meangood = pedgood - pedrough
  //     2) meanrough + shift = 0
  // get the noise histograms
  TH2S * hist = static_cast<TH2S *>(noisehist_.histo()->getTH2S());
  TProfile * prof = static_cast<TProfile *>(noiseprof_.histo()->getTProfile());
  for ( uint16_t istrip = 0; istrip < nstrips_; ++istrip ) {
    // get the 'rough' mean from the noise histogram [don't use Th1 projections! it's awfully slow]
    float meanrough = 0, tot = 0;
    for ( uint16_t ibin = 0; ibin < 2*nadcnoise_; ++ibin ) {
      float noisebin = hist->GetBinContent(hist->GetBin(ibin+1, istrip+1));
      meanrough += noisebin * hist->GetBinCenter(ibin+1);
      tot += noisebin;
    }
    meanrough /= tot;
    // get the 'correct' mean from the noise profile, taking the binning properly into account
    float meangood = prof->GetBinContent(istrip+1);
    // get the rough pedestal from the start of the run, that was used to make the noise histogram
    float pedrough = 1. * pedroughhist_.vSumOfContents_.at(istrip) / pedroughhist_.vNumOfEntries_.at(istrip);
    // get the good pedestal that was determined during the remainder of the run
    float pedgood  = 1. * pedhist_.vSumOfContents_.at(istrip) / pedhist_.vNumOfEntries_.at(istrip);
    // calculate the shift to apply. It should bring the mean very close to 0.
    float shift = meangood - meanrough - pedgood + pedrough;
    // determine how to loop depending on whether the shift is up or down
    int noisebinstart = (shift < 0 ? 1 : 2 * nadcnoise_);
    int noisebinstop  = (shift < 0 ? 2 * nadcnoise_ : 1);
    int noisebinincr  = (shift < 0 ? 1 : -1);
    // add the to be overwritten bins to the overflow - checked to keep integral the same
    for (int noisebin = noisebinstart; noisebin != noisebinstart-int(shift+0.5); noisebin += noisebinincr) {
      // get the value of the bin that will be overwritten when applying shift
      int binnrsource = hist->GetBin(noisebin, istrip+1);
      int binnrtarget = hist->GetBin(noisebinstart - noisebinincr, istrip+1);
      // get the content at the source and target
      float contentsource = hist->GetBinContent(binnrsource);
      float contenttarget = hist->GetBinContent(binnrtarget);
      // set the new content of the overview bin
      hist->SetBinContent(binnrtarget, contenttarget+contentsource);
    }
    // shift the bins in the 2D histogram - centers around zero and keeps the integral the same
    for (int noisebin = noisebinstart; noisebin != noisebinstop+noisebinincr; noisebin += noisebinincr) {
      int noisebinsource = noisebin - int(shift+0.5); // take the middle of the bin
      // check if the source bin would be out of range
      if (noisebinsource <= 2*nadcnoise_ && noisebinsource >= 1) {
        // retrieve the linear binnrs through the histogram
        int binnrsource = hist->GetBin(noisebinsource, istrip+1);
        int binnrtarget = hist->GetBin(noisebin, istrip+1);
        // get the content at the source bin
        float content = hist->GetBinContent(binnrsource);
        // now fill
        hist->SetBinContent(binnrtarget, content);
        // note: this will give a histogram with the mean between [-0.5,0.5]
        //   this is an unavoidable consequence of binning on a 2D histogram,
        //   where the true mean of each X projection is not stored
        //   nevertheless we still need to correct for the binning-biased mean
        //   in the histogram before shifting, otherwise we pile up errors
      } else { // source bin out of range
        // fill with zero ; overflow stays overflow
        int binnrtarget = hist->GetBin(noisebin, istrip+1);
        hist->SetBinContent(binnrtarget, 0);
      } // end if source in range
    } // end loop over bins
    // now also shift the noise profile such that it reflects the proper mean, without residual binning effect (0 if Gaussian)
    prof->SetBinContent(istrip+1, meangood - pedgood + pedrough);
    
  } // end loop over strips

}

// -----------------------------------------------------------------------------
//
void PedsFullNoiseTask::book()
{

  LogTrace(sistrip::mlDqmSource_) << "[PedsFullNoiseTask::" << __func__ << "]";

  // pedestal estimate profile histo
  // in principle we don't need to store this one, but let's keep it anyway
  pedroughhist_.isProfile_ = true;
  pedroughhist_.explicitFill_ = false;
  if (!pedroughhist_.explicitFill_) {
    pedroughhist_.vNumOfEntries_.resize(nstrips_,0);
    pedroughhist_.vSumOfContents_.resize(nstrips_,0);
    pedroughhist_.vSumOfSquares_.resize(nstrips_,0);
  }
  std::string titlepedrough = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                                                 sistrip::PEDS_FULL_NOISE,
                                                 sistrip::FED_KEY,
                                                 fedKey(),
                                                 sistrip::LLD_CHAN,
                                                 connection().lldChannel(),
                                                 sistrip::extrainfo::roughPedestals_).title();
  pedroughhist_.histo( dqm()->bookProfile( titlepedrough, titlepedrough,
                                           nstrips_, -0.5, nstrips_*1.-0.5,
                                           1025, 0., 1025. ) );

  // pedestal profile histo
  pedhist_.isProfile_ = true;
  pedhist_.explicitFill_ = false;
  if (!pedhist_.explicitFill_) {
    pedhist_.vNumOfEntries_.resize(nstrips_,0);
    pedhist_.vSumOfContents_.resize(nstrips_,0);
    pedhist_.vSumOfSquares_.resize(nstrips_,0);
  }
  std::string titleped = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                                            sistrip::PEDS_FULL_NOISE,
                                            sistrip::FED_KEY,
                                            fedKey(),
                                            sistrip::LLD_CHAN,
                                            connection().lldChannel(),
                                            sistrip::extrainfo::pedestals_).title();
  pedhist_.histo( dqm()->bookProfile( titleped, titleped,
                                      nstrips_, -0.5, nstrips_*1.-0.5,
                                      1025, 0., 1025. ) );

  // Common mode 1D histograms
  cmhist_.resize(2);
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    cmhist_[iapv].isProfile_ = false;
    cmhist_[iapv].vNumOfEntries_.resize(1024,0);
    std::string titlecm = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                             sistrip::PEDS_FULL_NOISE, 
                                             sistrip::FED_KEY, 
                                             fedKey(),
                                             sistrip::APV, 
                                             connection().i2cAddr(iapv),
                                             sistrip::extrainfo::commonMode_ ).title();
    titlecm = titlecm + boost::lexical_cast<std::string>(iapv);
    cmhist_[iapv].histo( dqm()->book1S( titlecm, titlecm,
                                        1024, -0.5, 1024.-0.5 ) );
  }

  // Noise profile
  noiseprof_.isProfile_ = true;
  noiseprof_.explicitFill_ = false;
  if (!noiseprof_.explicitFill_) {
    noiseprof_.vNumOfEntries_.resize(nstrips_,0);
    noiseprof_.vSumOfContents_.resize(nstrips_,0);
    noiseprof_.vSumOfSquares_.resize(nstrips_,0);
  }
  std::string titlenoise = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                              sistrip::PEDS_FULL_NOISE, 
                                              sistrip::FED_KEY, 
                                              fedKey(),
                                              sistrip::LLD_CHAN, 
                                              connection().lldChannel(),
                                              sistrip::extrainfo::noiseProfile_ ).title();
  noiseprof_.histo( dqm()->bookProfile( titlenoise, titlenoise,
                                        nstrips_, -0.5, nstrips_*1.-0.5,
                                        1025, 0., 1025. ) );

  // noise 2D compact histo
  noisehist_.explicitFill_ = true;
  if (!noisehist_.explicitFill_) {
    noisehist_.vNumOfEntries_.resize((nstrips_+2)*2*(nadcnoise_+2), 0);
  }
  std::string titlenoise2d = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                                sistrip::PEDS_FULL_NOISE, 
                                                sistrip::FED_KEY, 
                                                fedKey(),
                                                sistrip::LLD_CHAN, 
                                                connection().lldChannel(),
                                                sistrip::extrainfo::noise2D_ ).title();
  noisehist_.histo( dqm()->book2S( titlenoise2d, titlenoise2d,
                                   2*nadcnoise_, -nadcnoise_, nadcnoise_,
                                   nstrips_, -0.5, nstrips_*1.-0.5 ) );

}

// -----------------------------------------------------------------------------
//
void PedsFullNoiseTask::fill( const SiStripEventSummary & summary,
                         const edm::DetSet<SiStripRawDigi> & digis )
{

  // Check number of digis
  uint16_t nbins = digis.data.size();
  if (nbins != nstrips_) {
    edm::LogWarning(sistrip::mlDqmSource_)
      << "[PedsFullNoiseTask::" << __func__ << "]"
      << " " << nstrips_ << " digis expected, but got " << nbins << ". Skipping.";
    return;
  }

  // get the event number of the first event, not necessarily 1 (parallel processing on FUs)
  static uint32_t firstev = summary.event();

  // skipping events
  if (summary.event() - firstev < nskip_) return;
  // when all events are skipped
  if (!skipped_ && summary.event() - firstev >= nskip_) {
    skipped_ = true;
    if (nskip_ > 0) LogTrace(sistrip::mlDqmSource_) << "[PedsFullNoiseTask::" << __func__ << "]"
      << " Done skipping events. Now starting rough pedestals.";
  }

  // while stabilizing temperature...
  if (summary.event() - firstev < ntempstab_ + nskip_) {
    // estimate peds roughly
    for ( uint16_t istrip = 0; istrip < nstrips_; ++istrip ) {
      updateHistoSet( pedroughhist_, istrip, digis.data[istrip].adc() );
    }
    return;
  }
  // when temperature has stabilized
  if (!tempstable_ && summary.event() - firstev >= ntempstab_ + nskip_) {
    tempstable_ = true;
    LogTrace(sistrip::mlDqmSource_) << "[PedsFullNoiseTask::" << __func__ << "]"
      << " Rough pedestals done. Now starting noise measurements.";
  }

  // Calc common mode for both APVs
  uint16_t napvs = 2 * nbins / nstrips_;
  std::vector<uint32_t> cm; cm.resize(napvs, 0);
  std::vector<uint16_t> adc;
  for ( uint16_t iapv = 0; iapv < napvs; iapv++ ) { 
    adc.clear(); adc.reserve(128);
    for ( uint16_t ibin = 0; ibin < 128; ibin++ ) { 
      if ( (iapv*128)+ibin < nbins ) { 
        adc.push_back( digis.data[(iapv*128)+ibin].adc() );
      }
    }
    sort( adc.begin(), adc.end() ); 
    // take median as common mode
    uint16_t index = adc.size()%2 ? adc.size()/2 : adc.size()/2-1;
    if ( !adc.empty() ) cm[iapv] = (uint32_t) adc[index];
  }
  updateHistoSet( cmhist_[0], cm[0] );
  updateHistoSet( cmhist_[1], cm[1] );

  // 1D noise histogram
  for ( uint16_t istrip = 0; istrip < nstrips_; ++istrip ) {
    // calculate noise by subtracting common mode
    //float noiseval = static_cast<float>( digis.data[istrip].adc() ) - static_cast<float>( cm[istrip/128] );
    // calculate the noise wrt the rough pedestal estimate
    float noiseval = digis.data.at(istrip).adc() - 1.*pedroughhist_.vSumOfContents_.at(istrip)/pedroughhist_.vNumOfEntries_.at(istrip);
    // store the noise value in the profile
    updateHistoSet( noiseprof_, istrip, noiseval );
  }

  // 2D noise histogram and pedestal
  TH2S * hist = (TH2S *) noisehist_.histo()->getTH2S();
  // calculate pedestal and noise and store in the histograms
  for ( uint16_t istrip = 0; istrip < nstrips_; ++istrip ) {
    // store the pedestal
    updateHistoSet( pedhist_, istrip, digis.data[istrip].adc() );
    // calculate the noise wrt the rough pedestal estimate
    float noiseval = digis.data.at(istrip).adc() - 1.*pedroughhist_.vSumOfContents_.at(istrip)/pedroughhist_.vNumOfEntries_.at(istrip);
    // retrieve the linear binnr through the histogram
    short binnr = hist->GetBin((short) (noiseval+nadcnoise_), istrip+1) - 1; // -1 accounts for the fact that updateHistoSet() adds 1 later again
    // store the noise value in the 2D histo
    updateHistoSet( noisehist_, binnr );
  }


}

// -----------------------------------------------------------------------------
//
void PedsFullNoiseTask::update()
{

  // estimated pedestals 
  updateHistoSet( pedroughhist_ );

  // pedestals 
  updateHistoSet( pedhist_ );

  // commonmode
  updateHistoSet( cmhist_[0] );
  updateHistoSet( cmhist_[1] );

  // noise profile
  updateHistoSet( noiseprof_ );

  // noise 2D histo
  updateHistoSet( noisehist_ );

}
// -----------------------------------------------------------------------------
