#ifndef DPGAnalysis_SiStripTools_SiStripTKNumbers_H
#define DPGAnalysis_SiStripTools_SiStripTKNumbers_H

#include <map>

class DetId;

class SiStripTKNumbers {
public:
  SiStripTKNumbers();

  int nmodules(const DetId& detid) const;
  int nmodules(const int id) const;

  int nfibres(const DetId& detid) const;
  int nfibres(const int id) const;

  int napvs(const DetId& detid) const;
  int napvs(const int id) const;

  int nstrips(const DetId& detid) const;
  int nstrips(const int id) const;

private:
  std::map<int, int> _nmodules;
  std::map<int, int> _nfibres;

  static const int _apvsperfibre = 2;
  static const int _stripsperapv = 128;
};

#endif  //  DPGAnalysis_SiStripTools_SiStripTKNumbers_H
