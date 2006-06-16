#ifndef EVENTFILTER_UTILITIES_SHUTDOWNLISTENER_H
#define EVENTFILTER_UTILITIES_SHUTDOWNLISTENER_H

namespace evf
{

  class ShutDownListener
  { 
  public:

    ShutDownListener();
    virtual ~ShutDownListener();
    virtual void onShutDown()= 0;

  private:
    
    

  };
}
#endif
