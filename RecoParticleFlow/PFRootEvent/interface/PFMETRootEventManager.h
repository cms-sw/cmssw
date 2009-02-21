#ifndef Demo_PFRootEvent_PFMETRootEventManager_h
#define Demo_PFRootEvent_PFMETRootEventManager_h

#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"
#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"


class PFMETRootEventManager : public PFRootEventManager {
  
 public:
  PFMETRootEventManager(const char* file);

  ~PFMETRootEventManager();

  bool processEntry(int entry);
  float DeltaMET(int entry);
  float DeltaPhi(int entry);
  void write();


 private:

  PFMETBenchmark benchmark_;

};

#endif
