#ifndef CollectorRoot_h
#define CollectorRoot_h

#include "DQMServices/Core/src/ClientServerRoot.h"

#include <string>

class CollectorRoot : public ClientServerRoot
{
 public:

  virtual ~CollectorRoot();
  virtual void process()=0;

  /// default connecting port
  static const Int_t defListenPort = 9090;

 protected:
  /// can only be constructed through the "instance" method of the derived class
  CollectorRoot(std::string name, int listenport = defListenPort, 
		///whether to keep MonitorElements in memory after sources go down
		bool keepStaleSources = false);
  /// the infinite loop
  void run(void);
  /// the REAL main loop
  virtual bool run_once(void);

  bool inputAvail_;
  /// port for connections
  int listenport_;

 private:

};

#endif
