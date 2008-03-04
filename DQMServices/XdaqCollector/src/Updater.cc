#include "DQMServices/XdaqCollector/interface/Updater.h"
#include "DQMServices/XdaqCollector/interface/UpdateObserver.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

using namespace dqm;

Updater::Updater(MonitorUserInterface *the_mui) : mui(the_mui), observed(false),
						  running(false)
{
  // create a thread and pass a pointer to the static function start
  pthread_create(&worker, NULL, start, (void *)this);
}

void *Updater::start(void *pthis)
{
  // We receive a pointer to the parent Updater object
  Updater *updater_pt = (Updater *)pthis;
  updater_pt->setRunning();
  while(1)
    {
      // call the updater request_update method
      updater_pt->request_update();
      if(!updater_pt->checkRunning()) break;
    }
  
  return (void *)updater_pt;
}

void Updater::request_update()
{
  mui->update();
  if(observed)
    for(obi i = obs_.begin(); i != obs_.end(); i++)
      (*i)->onUpdate();
}

void Updater::registerObserver(UpdateObserver *obs)
{
  observed = true;
  obs_.push_back(obs);
}
