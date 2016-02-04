#ifndef DPGAnalysis_SiStripTools_SiStripTKNumbers_H
#define DPGAnalysis_SiStripTools_SiStripTKNumbers_H

#include <map>

class SiStripDetId;

class SiStripTKNumbers {

 public:

  SiStripTKNumbers();

  int  nmodules(const SiStripDetId& detid) const;
  int  nmodules(const int id) const;

  int  nfibres(const SiStripDetId& detid) const;
  int  nfibres(const int id) const;

  int  napvs(const SiStripDetId& detid) const;
  int  napvs(const int id) const;

  int  nstrips(const SiStripDetId& detid) const;
  int  nstrips(const int id) const;
  
 private:
  
  std::map<int, int> _nmodules;
  std::map<int, int> _nfibres;
  
  
  static const int _apvsperfibre = 2;
  static const int _stripsperapv = 128;
  
};

#endif  //  DPGAnalysis_SiStripTools_SiStripTKNumbers_H
