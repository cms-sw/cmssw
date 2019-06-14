//#include <map>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DPGAnalysis/SiStripTools/interface/SiStripTKNumbers.h"

SiStripTKNumbers::SiStripTKNumbers() {
  DetId tk(DetId::Tracker, 0);
  _nmodules[tk.rawId()] = (3540 - 816) + 816 + 5208 + 6400;
  _nfibres[tk.rawId()] = (9192 - 2208) + 2208 + 12906 + 15104;
  _nmodules[tk.subdetId()] = _nmodules[tk.rawId()];
  _nfibres[tk.subdetId()] = _nfibres[tk.rawId()];

  DetId tib(DetId::Tracker, StripSubdetector::TIB);
  _nmodules[tib.rawId()] = 3540 - 816;
  _nfibres[tib.rawId()] = 9192 - 2208;
  _nmodules[tib.subdetId()] = _nmodules[tib.rawId()];
  _nfibres[tib.subdetId()] = _nfibres[tib.rawId()];

  DetId tid(DetId::Tracker, StripSubdetector::TID);
  _nmodules[tid.rawId()] = 816;
  _nfibres[tid.rawId()] = 2208;
  _nmodules[tid.subdetId()] = _nmodules[tid.rawId()];
  _nfibres[tid.subdetId()] = _nfibres[tid.rawId()];

  DetId tob(DetId::Tracker, StripSubdetector::TOB);
  _nmodules[tob.rawId()] = 5208;
  _nfibres[tob.rawId()] = 12906;
  _nmodules[tob.subdetId()] = _nmodules[tob.rawId()];
  _nfibres[tob.subdetId()] = _nfibres[tob.rawId()];

  DetId tec(DetId::Tracker, StripSubdetector::TEC);
  _nmodules[tec.rawId()] = 6400;
  _nfibres[tec.rawId()] = 15104;
  _nmodules[tec.subdetId()] = _nmodules[tec.rawId()];
  _nfibres[tec.subdetId()] = _nfibres[tec.rawId()];

  DetId tecp(DetId(DetId::Tracker, StripSubdetector::TEC).rawId() | ((1 & 0x3) << 18));
  _nmodules[tecp.rawId()] = 3200;
  _nfibres[tecp.rawId()] = 7552;

  DetId tecm(DetId(DetId::Tracker, StripSubdetector::TEC).rawId() | ((2 & 0x3) << 18));
  _nmodules[tecm.rawId()] = 3200;
  _nfibres[tecm.rawId()] = 7552;
}

int SiStripTKNumbers::nmodules(const DetId& detid) const {
  int subd = detid.subdetId();
  if (_nmodules.find(subd) != _nmodules.end())
    return _nmodules.find(subd)->second;

  return 0;
}

int SiStripTKNumbers::nmodules(const int id) const {
  if (_nmodules.find(id) != _nmodules.end())
    return _nmodules.find(id)->second;

  return 0;
}

int SiStripTKNumbers::nfibres(const DetId& detid) const {
  int subd = detid.subdetId();
  if (_nfibres.find(subd) != _nfibres.end())
    return _nfibres.find(subd)->second;

  return 0;
}

int SiStripTKNumbers::nfibres(const int id) const {
  if (_nfibres.find(id) != _nfibres.end())
    return _nfibres.find(id)->second;

  return 0;
}

int SiStripTKNumbers::napvs(const DetId& detid) const { return nfibres(detid) * _apvsperfibre; }

int SiStripTKNumbers::napvs(const int id) const { return nfibres(id) * _apvsperfibre; }

int SiStripTKNumbers::nstrips(const DetId& detid) const { return nfibres(detid) * _apvsperfibre * _stripsperapv; }

int SiStripTKNumbers::nstrips(const int id) const { return nfibres(id) * _apvsperfibre * _stripsperapv; }
