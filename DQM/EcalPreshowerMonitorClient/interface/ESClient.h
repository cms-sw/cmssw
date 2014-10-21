#ifndef ESClient_H
#define ESClient_H

#include <string>

#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;

class ESClient {

   public:

      virtual void analyze(void)      = 0;
      virtual void beginJob(DQMStore* dqmStore)     = 0;
      virtual void endJob(void)       = 0;
      virtual void beginRun(void)     = 0;
      virtual void endRun(void)       = 0;
      virtual void setup(void)	=0;
      virtual void cleanup(void)	=0;
      //  virtual int getEvtPerJob( void ) = 0;
      //  virtual int getEvtPerRun( void ) = 0;
      virtual void endLumiAnalyze(void)   =0;

      virtual ~ESClient(void) {}

      template<typename T> T* getHisto(MonitorElement*, bool = false, T* = 0) const;

};

template<typename T>
T*
ESClient::getHisto(MonitorElement* _me, bool _clone/* = false*/, T* _current/* = 0*/) const
{
  if(!_me){
    if(_clone) return _current;
    else return 0;
  }

  TObject* obj(_me->getRootObject());

  if(!obj) return 0;

  if(_clone){
    delete _current;
    _current = dynamic_cast<T*>(obj->Clone(("ME " + _me->getName()).c_str()));
    if(_current) _current->SetDirectory(0);
    return _current;
  }
  else
    return dynamic_cast<T*>(obj);
}

#endif // ESClient_H

