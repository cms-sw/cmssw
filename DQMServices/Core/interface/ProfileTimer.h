#ifndef _PROFILE_TIMER_H_
#define _PROFILE_TIMER_H_

/// set to true/false to turn profiling on/off
/** Note: setting this to true will generate root-tuple with timing informartion 
   for senders; this will appear to cause a memory leak; however, this is due to
   the never-ending expanding of the root-tree containing the timing info; 
   furthermore, the root file is never properly closed (since the collector 
   never stops running; one can still access the information (despite the 
   root warnings) - Christos, July 2005
*/
#define DO_TIME_PROFILING false

#include <TStopwatch.h>

const bool do_profiling = DO_TIME_PROFILING;

/// --------------------------------------------------------------------------------
/// An intelligent way of turning time profiling on & off (thanks to jbk@fnal.gov)
/// --------------------------------------------------------------------------------

/// default (time profiling turned off)
template <bool b> 
class TimerTool
{
public:
    class DoProfile
    {
       public:
       explicit DoProfile(TimerTool<b>&) { }
       /// add destructor to suppres the "unused variable" warning
       ~DoProfile(){}
    };

    /// stinky interface
    TStopwatch* get() { return 0; }
    bool profilingEnabled() const { return false; }
    /// get timing in secs
    Float_t getTime(void){return -999;}
};

/// used if profiling is requested
template <>
class TimerTool<true>
{
 public:
  class DoProfile
  {
  public:
    /// manipulate the timer object: start timer the moment 
    /// TimerTool::DoProfile ctor is called, stop it when dtor is called!
    explicit DoProfile(TimerTool& p):timer_(p.get()) 
    {timer_->Start(); }
    ~DoProfile() { timer_->Stop(); }
  private:
    TStopwatch* timer_; /// same object as in argument of DoProfile ctor
  };
  
  TimerTool(): timer(new TStopwatch) { }
  ~TimerTool() { delete timer; }
  
  TStopwatch* get() { return timer; }
  bool profilingEnabled() const { return true; }
  /// get timing in secs
  Float_t getTime(void){return timer->RealTime();}
 private:
  TStopwatch* timer;
};

/// instantiate the correct version using a typedef
typedef TimerTool<do_profiling> OperatorTimer;

#endif // #ifndef _PROFILE_TIMER_H_
