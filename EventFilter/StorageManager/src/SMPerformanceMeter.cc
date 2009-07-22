/*
     For performance statistics for
     Storage Manager and SMProxyServer.

     $Id: SMPerformanceMeter.cc,v 1.7 2008/10/13 13:05:36 hcheung Exp $
*/

#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"

stor::SMPerfStats::SMPerfStats():
  samples_(1000),
  period4samples_(5),
  maxBandwidth_(0.),
  minBandwidth_(999999.),
  maxBandwidth2_(0.),
  minBandwidth2_(999999.)
{
  longTermCounter_.reset(new ForeverCounter());
  shortTermCounter_.reset(new RollingSampleCounter(samples_, samples_, samples_,
                                                   RollingSampleCounter::INCLUDE_SAMPLES_IMMEDIATELY));
  shortPeriodCounter_.reset(new RollingIntervalCounter(10*period4samples_, period4samples_, period4samples_));
}

void stor::SMPerfStats::reset()
{
  maxBandwidth_    = 0.;
  minBandwidth_    = 999999.;
  maxBandwidth2_    = 0.;
  minBandwidth2_    = 999999.;
}

void stor::SMPerfStats::fullReset()
{
  longTermCounter_.reset(new ForeverCounter());
  shortTermCounter_.reset(new RollingSampleCounter(samples_, samples_, samples_,
                                                   RollingSampleCounter::INCLUDE_SAMPLES_IMMEDIATELY));
  shortPeriodCounter_.reset(new RollingIntervalCounter(10*period4samples_, period4samples_, period4samples_));
  maxBandwidth_    = 0.;
  minBandwidth_    = 999999.;
  maxBandwidth2_    = 0.;
  minBandwidth2_    = 999999.;
}

stor::SMOnlyStats::SMOnlyStats():
  instantBandwidth_(0.0),
  instantRate_(0.0),
  instantLatency_(0.0),
  totalSamples_(0.0),
  duration_(0.0),
  meanBandwidth_(0.0),
  meanRate_(0.0),
  meanLatency_(0.0),
  maxBandwidth_(0.0),
  minBandwidth_(999999.0),
  instantBandwidth2_(0.0),
  instantRate2_(0.0),
  instantLatency2_(0.0),
  totalSamples2_(0.0),
  duration2_(0.0),
  meanBandwidth2_(0.0),
  meanRate2_(0.0),
  meanLatency2_(0.0),
  maxBandwidth2_(0.0),
  minBandwidth2_(999999.0),
  receivedVolume_(0.0)
{
}

stor::SMPerformanceMeter::SMPerformanceMeter():
 loopCounter_(0)
{
  stats_.fullReset();
}
			
void stor::SMPerformanceMeter::init(unsigned long samples, unsigned long time_period) 
{
  // samples is the number of samples that will be collected
  // before statistics for those samples are calculated.
  // A running mean statistics is also kept and not changed
  // by changing the number samples_.
  boost::mutex::scoped_lock sl(data_lock_);
  loopCounter_ = 0;
  stats_.samples_ = samples;
  stats_.period4samples_ = time_period;
  stats_.fullReset();
}
	
bool stor::SMPerformanceMeter::addSample(unsigned long size) 
{	
  // Take the received amount of data in bytes and
  // use the counters to keep statistics for the last
  // samples_ samples; for the last time period period4samples_
  // and for the whole period init was called
  // Return true if set of samples is reached.
  boost::mutex::scoped_lock sl(data_lock_);

  stats_.longTermCounter_->addSample( (double) size / (double) 0x100000 );
  stats_.shortTermCounter_->addSample( (double) size / (double) 0x100000 );
  stats_.shortPeriodCounter_->addSample( (double) size / (double) 0x100000 );

  if ( stats_.shortPeriodCounter_->hasValidResult() )
  {
    double testVal = stats_.shortPeriodCounter_->getValueRate();
    if (testVal > stats_.maxBandwidth2_)
      stats_.maxBandwidth2_ = testVal;
    if (testVal < stats_.minBandwidth2_)
      stats_.minBandwidth2_ = testVal;
  }	

  if ( loopCounter_ == (stats_.samples_ - 1  ) )
  {
    loopCounter_ = 0;

    double testVal = stats_.shortTermCounter_->getValueRate();
    if (testVal > stats_.maxBandwidth_)
      stats_.maxBandwidth_ = testVal;
    if (testVal < stats_.minBandwidth_)
      stats_.minBandwidth_ = testVal;

    return true;			
  }	
  ++loopCounter_;

  return false;
}	

void stor::SMPerformanceMeter::setSamples(unsigned long num_samples)
{
  boost::mutex::scoped_lock sl(data_lock_);
  stats_.samples_ = num_samples;
  stats_.fullReset();
}

void stor::SMPerformanceMeter::setPeriod4Samples(unsigned long time_period)
{
  boost::mutex::scoped_lock sl(data_lock_);
  stats_.period4samples_ = time_period;
  stats_.fullReset();
}

stor::SMPerfStats stor::SMPerformanceMeter::getStats() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_;
}

unsigned long stor::SMPerformanceMeter::samples() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.samples_;
}

double stor::SMPerformanceMeter::totalvolumemb() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.longTermCounter_->getValueSum();
}


