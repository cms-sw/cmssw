#ifndef AuxCollectorRoot_h
#define AuxCollectorRoot_h

#include "DQMServices/Core/interface/CollectorRoot.h"

#include <string>
#include <vector>

class AuxCollectorRoot : public CollectorRoot
{
 public:

  virtual ~AuxCollectorRoot();
  virtual void process()=0;

  /// default connecting port
  static const Int_t defListenPort = 9090;

 protected:
  /// can only be constructed through the "instance" method of the derived class
  AuxCollectorRoot(std::string host, std::string name, 
		   int listenport = defListenPort);
  AuxCollectorRoot(std::vector<std::string> hosts, std::string name, 
		   int listenport = defListenPort);

  /// the REAL main loop
  bool run_once(void);

  /// port for connections
  int listenport_;

 
 private:

};

#endif
