#ifndef _Updater_h_
#define _Updater_h_

#include <pthread.h>
#include <list>

class MonitorUserInterface;

namespace dqm
{
  class UpdateObserver;

  class Updater
    {
    private:
      typedef std::list<dqm::UpdateObserver *> olist;
      typedef olist::const_iterator obi;
      pthread_t worker;
      MonitorUserInterface *mui; 
      olist obs_;
      bool observed;
      bool running;

    public:
      
      Updater(MonitorUserInterface *the_mui);
      
      ~Updater()
	{
	  /// recommended by Benigno Gobbo to fix crash for following sequence:
	  /// Configure/Enable/Halt + Configure/Enable
	  pthread_cancel( worker );
	}
      
      void registerObserver(dqm::UpdateObserver *);

      void request_update();
      
      /// this is the function the thread will execute
      static void *start(void *pthis);
      
      void setRunning()
	{ 
	  running = true; 
	}

      bool checkRunning() const 
	{
	  return running;
	}

      void setStopped()
	{
	  running=false;
	}

    };
}

#endif
