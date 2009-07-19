#ifndef DPGAnalysis_SiStripTools_APVShotFinder_h
#define DPGAnalysis_SiStripTools_APVShotFinder_h

#include <vector>
#include "DPGAnalysis/SiStripTools/interface/APVShot.h"

class SiStripDigi;


namespace edm {
  template <class T> class DetSet;
  template <class T> class DetSetVector;
}

class APVShotFinder {

 public:
  APVShotFinder();
  APVShotFinder(const edm::DetSet<SiStripDigi>& digis);
  APVShotFinder(const edm::DetSetVector<SiStripDigi>& digicoll);

  void computeShots(const edm::DetSet<SiStripDigi>& digis);
  void computeShots(const edm::DetSetVector<SiStripDigi>& digicoll);

  const std::vector<APVShot>& getShots() const;

 private:

  void addShots(const edm::DetSet<SiStripDigi>& digis);

  std::vector<APVShot> _shots;

};

#endif // DPGAnalysis_SiStripTools_APVShotFinder_h
