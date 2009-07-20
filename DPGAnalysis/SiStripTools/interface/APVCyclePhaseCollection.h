#ifndef DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h
#define DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h

#include <string>
#include <map>


class APVCyclePhaseCollection {

 public:
  APVCyclePhaseCollection(): _apvmap() { };
  ~APVCyclePhaseCollection() { };

  const std::map<std::string,int>& get() const { return _apvmap; };

  std::map<std::string,int>& get() { return _apvmap; };


 private:


  std::map<std::string,int> _apvmap;

};

#endif // DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h
