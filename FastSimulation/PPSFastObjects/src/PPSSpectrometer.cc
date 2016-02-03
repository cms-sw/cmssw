#include "FastSimulation/PPSFastObjects/interface/PPSSpectrometer.h"
template<class T>
PPSSpectrometer<T>::PPSSpectrometer(){Vertices = new PPSBaseVertex();};


template<>
PPSSpectrometer<PPSSimData>::PPSSpectrometer() {Vertices = new PPSBaseVertex();};
template<>
PPSSpectrometer<PPSGenData>::PPSSpectrometer() {Vertices = new PPSGenVertex();};
template<>
PPSSpectrometer<PPSRecoData>::PPSSpectrometer(){Vertices = new PPSRecoVertex();};

