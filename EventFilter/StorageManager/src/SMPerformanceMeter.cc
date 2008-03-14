/*
     For performance statistics for
     Storage Manager and SMProxyServer.

     $Id$
*/

#include "EventFilter/StorageManager/interface/SMPerformanceMeter.h"

stor::SMPerfStats::SMPerfStats():
  samples_(100),
  totalMB4mean_(0.0),
  meanThroughput_(0.0),
  meanRate_(0.0),
  meanLatency_(0.0),
  sampleCounter_(0),
  allTime_(0.0),
  maxBandwidth_(0.),
  minBandwidth_(999999.),
  totalMB_(0.),
  throughput_(0.0),
  rate_(0.0),
  latency_(0.0)
{
}

void stor::SMPerfStats::reset()
{
  latency_ = 0.0;
  throughput_= 0.0;
  rate_ = 0.0;
  totalMB_ = 0.;
}

void stor::SMPerfStats::fullReset()
{
  latency_ = 0.0;
  throughput_= 0.0;
  rate_ = 0.0;
  totalMB_ = 0.;
  totalMB4mean_= 0.0;
  meanThroughput_= 0.0;
  meanRate_= 0.0;
  meanLatency_= 0.0;
  sampleCounter_= 0;
  allTime_= 0.0;
  maxBandwidth_     = 0.;
  minBandwidth_     = 999999.;
}

stor::SMPerformanceMeter::SMPerformanceMeter()
{
  stats_.fullReset();
}
			
void stor::SMPerformanceMeter::init(unsigned long samples) 
{
  // samples is the number of samples that will be collected
  // before statistics for those samples are calculated.
  // A running mean statistics is also kept and not changed
  // by changing the number samples_.
  boost::mutex::scoped_lock sl(data_lock_);
  loopCounter_ = 0;
  stats_.reset();
  stats_.samples_ = samples;
}
	
bool stor::SMPerformanceMeter::addSample(unsigned long size) 
{	
  // Take the received amount of data in bytes and
  // add it to the total amount of received bytes.
  //   Update the statistics after addSample
  // is called samples_ times. Keep a runnning mean.
  // Return true if set of samples is reached.
  boost::mutex::scoped_lock sl(data_lock_);
  if ( loopCounter_ == 0 )
  {	
    if(stats_.sampleCounter_ == 0) chrono_.start(0);
    stats_.totalMB_ = 0;  // MB per set of samples
  }

  stats_.totalMB_ += ( (double) size / (double) 0x100000 );
  stats_.totalMB4mean_ += ( (double) size / (double) 0x100000 );

  if ( loopCounter_ == (stats_.samples_ - 1  ) )
  {
    chrono_.stop(0);
    double usecs = (double) chrono_.dusecs();
    chrono_.start(0);
    loopCounter_ = 0;

    stats_.throughput_ = stats_.totalMB_ / (usecs/ (double) 1000000.0);
    stats_.rate_ = ((double) stats_.samples_) / (usecs/ (double) 1000000.0);
    stats_.latency_ = ((double) 1000000.0)/stats_.rate_;
    // for mean measurements
    //double secs = (double) chrono_.dsecs();
    stats_.allTime_ += usecs/ (double) 1000000.0;
    stats_.meanThroughput_ = stats_.totalMB4mean_ / stats_.allTime_;
    stats_.meanRate_ = ((double) stats_.sampleCounter_) / stats_.allTime_;
    stats_.meanLatency_ = ((double) 1000000.0)/stats_.meanRate_;

    // Determine minimum and maximum "instantaneous" bandwidth
    if (stats_.throughput_ > stats_.maxBandwidth_)
      stats_.maxBandwidth_ = stats_.throughput_;
    if (stats_.throughput_ < stats_.minBandwidth_)
      stats_.minBandwidth_ = stats_.throughput_;

    ++stats_.sampleCounter_;
    return true;			
  }	
  ++loopCounter_;
  ++stats_.sampleCounter_;

  return false;
}	

void stor::SMPerformanceMeter::setSamples(unsigned long num_samples)
{
  boost::mutex::scoped_lock sl(data_lock_);
  stats_.samples_ = num_samples;
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

double stor::SMPerformanceMeter::bandwidth() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.throughput_;
}

double stor::SMPerformanceMeter::rate() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.rate_;
}

double stor::SMPerformanceMeter::latency() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.latency_;
}

double stor::SMPerformanceMeter::meanbandwidth() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.meanThroughput_;
}

double stor::SMPerformanceMeter::maxbandwidth() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.maxBandwidth_;
}

double stor::SMPerformanceMeter::minbandwidth() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.minBandwidth_;
}

double stor::SMPerformanceMeter::meanrate() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.meanRate_;
}

double stor::SMPerformanceMeter::meanlatency() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.meanLatency_;
}

unsigned long stor::SMPerformanceMeter::totalsamples() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.sampleCounter_;
}

double stor::SMPerformanceMeter::totalvolumemb() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.totalMB4mean_;
}

double stor::SMPerformanceMeter::duration() 
{
  boost::mutex::scoped_lock sl(data_lock_);
  return stats_.allTime_;
}

