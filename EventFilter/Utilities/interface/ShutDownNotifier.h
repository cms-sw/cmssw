#ifndef EVENTFILTER_UTILITIES_SHUTDOWNNotifier_H
#define EVENTFILTER_UTILITIES_SHUTDOWNNotifier_H

#include <vector>

namespace evf{
  
  class ShutDownListener;

  class ShutDownNotifier
    {

    public:
      
      void registerListener(ShutDownListener *);
      void removeListener(ShutDownListener *);

      static ShutDownNotifier *instance(){return instance_;}
      
    private:
      
      void notify();
      void cleanup();
      ShutDownNotifier(){instance_ = this;}
      static ShutDownNotifier *instance_;
      std::vector<ShutDownListener *> listeners_;
      friend class FUEventProcessor;
    };


}
#endif
