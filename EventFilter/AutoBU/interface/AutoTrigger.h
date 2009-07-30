#ifndef EVENTFILTER_AUTOBU_AUTOTRIGGER_H
#define EVENTFILTER_AUTOBU_AUTOTRIGGER_H

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/TriggerAdapter/interface/Application.h"
#include "EventFilter/AutoBU/interface/TriggerGeneratorWithPayload.h"

#include <boost/random.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_smallint.hpp>

#include <xdata/Double.h>

#include <toolbox/task/WorkLoop.h>

#include <semaphore.h>

namespace evf{

class AutoTrigger : public rubuilder::ta::Application
{
 private:
  typedef boost::variate_generator<
    boost::mt19937, boost::poisson_distribution<>
    >                              rnd_poisson_t;


 public:
  /**
   * Define factory method for the instantion of TA applications.
   */
  XDAQ_INSTANTIATOR();


  AutoTrigger(xdaq::ApplicationStub *s)
    throw (xdaq::exception::Exception) : rubuilder::ta::Application(s),
    wlLumiSectionCycle_(0), asLumiSectionCycle_(0), 
    lsExit_(false), lsCurrent_(0), orbitCurrent_(0), bx_(0), 
    targetRate_(100), lambdaPoiss_(10)
    {
      sem_init(&triggerSem_,0,0);
      sem_init(&deadtimeSem_,0,0);
      boost::mt19937 rnd_gen;    
      rndPoiss_ = new rnd_poisson_t( rnd_gen,
				      boost::poisson_distribution<>( lambdaPoiss_ ));
      delta_ = int((2.-double(targetRate_.value_)/100000.)/double(targetRate_.value_)*1.e6+3.); //empyrical formula, will depend on platform
      std::cout << "const setting target rate and delta " << delta_ << std::endl;
      getApplicationInfoSpace()->fireItemAvailable("totalTriggerRate",&measuredRate_);
      getApplicationInfoSpace()->fireItemAvailable("totalTriggerNumber",&eventNumber_);
    }


 private:

  /**
   * The generator of dummy triggers.
   */
  evf::TriggerGeneratorWithPayload triggerGenerator_;

  // workloop / action signature for LS
  toolbox::task::WorkLoop         *wlLumiSectionCycle_;      
  toolbox::task::ActionSignature  *asLumiSectionCycle_;

  bool                             lsExit_;
  xdata::UnsignedInteger32         lsCurrent_;
  xdata::UnsignedInteger32         orbitCurrent_;
  timeval                          lsStartTime_;
  timeval                          lsEndTime_;

  // workloop / action signature for trigger
  toolbox::task::WorkLoop         *wlMachineCycle_;      
  toolbox::task::ActionSignature  *asMachineCycle_;

  bool                             machineExit_;
  uint32_t                         bx_;
  xdata::UnsignedInteger32         deadBuckets_;
  xdata::UnsignedInteger32         targetRate_;
  xdata::Double                    measuredRate_;
  xdata::Double                    fractionDead_;
  uint32_t                         delta_;
  double                           lambdaPoiss_;
  boost::mt19937                   rndGen_;   //Mersenne Twister generator
  rnd_poisson_t                   *rndPoiss_;
  sem_t                            triggerSem_;
  sem_t                            deadtimeSem_;

  // workloop / action signature for monitor
  toolbox::task::WorkLoop         *wlMonitorCycle_;      
  toolbox::task::ActionSignature  *asMonitorCycle_;
  bool                             monitorExit_;
  timeval                          monitorStartTime_;
  timeval                          monitorEndTime_;


    /**
     * Callback implementing the action to be executed on the
     * Ready->Enabled transition.
     */
  void enableAction(toolbox::Event::Reference e)
    throw (toolbox::fsm::exception::Exception);
  void haltAction(toolbox::Event::Reference e)
    throw (toolbox::fsm::exception::Exception);
  void sendNTriggers(const unsigned int n)
    throw (rubuilder::ta::exception::Exception);
  void defaultWebPage(xgi::Input *in, xgi::Output *out)
    throw (xgi::exception::Exception);
  void startLsCycle() throw (evf::Exception);
  void stopLsCycle() throw (evf::Exception);
  
  bool lsCycle(toolbox::task::WorkLoop* wl);
  bool machineCycle(toolbox::task::WorkLoop* wl);

  void startMonitorCycle() throw (evf::Exception);
  void stopMonitorCycle() throw (evf::Exception);
  
  bool monitorCycle(toolbox::task::WorkLoop* wl);


  typedef rubuilder::ta::Application Base;  
};
}
#endif
