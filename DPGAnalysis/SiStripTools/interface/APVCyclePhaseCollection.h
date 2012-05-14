#ifndef DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h
#define DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h

#include <string>
#include <map>
#include <vector>


class APVCyclePhaseCollection {

 public:
  APVCyclePhaseCollection(): _apvmap() { };
  ~APVCyclePhaseCollection() { };

  const std::map<std::string,int>& get() const { return _apvmap; };

  std::map<std::string,int>& get() { return _apvmap; };

  const int getPhase(const std::string partition) const;

  const std::vector<int> getPhases(const std::string partition) const;

  enum{nopartition=-91,multiphase=-92,empty=-98,invalid=-99};

 private:


  std::map<std::string,int> _apvmap;

};

#endif // DPGAnalysis_SiStripTools_APVCyclePhaseCollection_h
