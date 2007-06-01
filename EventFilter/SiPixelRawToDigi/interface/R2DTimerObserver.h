#ifndef SiPixelRawToDigi_R2DTimerObserver_H 
#define SiPixelRawToDigi_R2DTimerObserver_H

#include <string>
#include "Utilities/Timing/interface/TimingReport.h"

class R2DTimerObserver :  public TimingReport::ItemObserver {
/// \class R2DTimerObserver
/// utility to get the real/cpu time betwen last TimingReport::Item start-stop.
/// actual timer initialisation by initTiming(..) from PixelRecoUtilities.
/// example usage:
///R2DTimerObserver tm("timer name");
/// ....   {  TimeMe t(tm.item(),false);   ...   }
/// last_real = tm.lastMeasurement().real();


private:

  typedef TimingReport::ItemObserver::event TimerState;

  class LastMeasurement {
  public:
    LastMeasurement(double real=0., double cpu=0.) : real_(real), cpu_(cpu) { }
    double real() const { return real_;}
    double cpu() const { return cpu_;}
  private:
    double real_,cpu_;
  };
 
  TimingReport::Item * timer_item;
  TimerState lastState;
  LastMeasurement theMeasurement;

  /// from base class
  virtual void operator()(const TimerState & timerState) {
    theMeasurement = LastMeasurement(timerState.first-lastState.first,
                                     timerState.second-lastState.second);
    lastState = timerState;
  }

public:

  void init(const std::string & name) {
    timer_item = &(*TimingReport::current())[name];
    timer_item->switchCPU(false);
    timer_item->setObs(this);
  }

  R2DTimerObserver() : timer_item(0), lastState(0,0) { }
  R2DTimerObserver(const std::string name) : lastState(0,0) { init( name) ; }

  const LastMeasurement & lastMeasurement() { return theMeasurement; }
  TimingReport::Item & item() { return *timer_item; }

  void start() { timer_item->start(); }
  void stop() { timer_item->stop(); }

};

#endif
