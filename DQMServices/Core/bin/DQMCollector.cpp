#include <string>
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "TROOT.h"

#include "DQMServices/Core/interface/CollectorRoot.h"
#include "DQMServices/Core/src/ClientRoot.h"

class DummyCollector : public CollectorRoot
{

public:
  virtual void process(){}
  DummyCollector(bool keepStaleSources, int port_no) : 
    CollectorRoot("Collector", port_no, keepStaleSources)
  {
    inputAvail_=true;
    run();
  }
  static CollectorRoot *instance(bool keepStaleSources = false,
				 int port_no = CollectorRoot::defListenPort)
  {
    if(instance_==0)
      instance_ = new DummyCollector(keepStaleSources, port_no);
    return instance_;
  }

private:
  static CollectorRoot * instance_;
};

CollectorRoot *DummyCollector::instance_=0; 

extern void InitGui(); 
VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
TROOT producer("producer","Simple histogram producer",initfuncs);

void interrupt (int sig) 
{
  close (ClientRoot::ss);
  exit (1);
}

// usage: DQMCollector <keepStaleSources> <port_no>
// <keepStaleSources>: enter "1" if corresponding monitoring information 
//                     should stay in memory after sources are done processing 
//                     and/or disconnect (default: 0)
// <port_no>         : port number for connection with sources and clients 
//                     (default: 9090)
int main(int argc, char *argv[])
{
  bool keepStaleSources = true;
  if(argc >= 2)
    keepStaleSources = (atoi(argv[1]) != 0);

  // default port #
  int port_no = 9090;
  if(argc >= 3) port_no = atoi(argv[2]);

  signal (SIGINT, interrupt);
  try
    {
      DummyCollector::instance(keepStaleSources, port_no);
    }
  catch (...)
    {
      return -1;
    }
  //  gROOT->SetBatch(kTRUE);
  //TApplication app("app",&argc,argv);
  return 0;
}
