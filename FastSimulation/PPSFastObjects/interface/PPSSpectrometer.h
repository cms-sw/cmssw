#ifndef PPSSPECTROMETER_H
#define PPSSPECTROMETER_H
#include <vector>
#include <utility>
#include "TObject.h"
#include "FastSimulation/PPSFastObjects/interface/PPSGenData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSSimData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSRecoData.h"
#include "FastSimulation/PPSFastObjects/interface/PPSGenVertex.h"
#include "FastSimulation/PPSFastObjects/interface/PPSRecoVertex.h"

typedef PPSRecoData Reco;
typedef PPSGenData  Gen;
typedef PPSSimData  Sim;

template<class T>
class PPSSpectrometer: public TObject {
public:
      PPSSpectrometer();
      virtual ~PPSSpectrometer(){};

      int     Nvtx() {return Vertices->size();};

      PPSBaseVertex* Vertices;
      T ArmF;
      T ArmB;

      void clear() { if (Vertices) Vertices->clear();ArmF.clear(); ArmB.clear(); };

ClassDef(PPSSpectrometer,1);
};
#endif
