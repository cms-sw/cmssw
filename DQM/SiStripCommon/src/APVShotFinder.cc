#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/APVShot.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DQM/SiStripCommon/interface/APVShotFinder.h"

APVShotFinder::APVShotFinder(const bool zs) : _zs(zs), _shots() {}

APVShotFinder::APVShotFinder(const edm::DetSet<SiStripDigi>& digis, const bool zs) : _zs(zs), _shots() {
  computeShots(digis);
}

APVShotFinder::APVShotFinder(const edm::DetSetVector<SiStripDigi>& digicoll, const bool zs) : _zs(zs), _shots() {
  computeShots(digicoll);
}

void APVShotFinder::computeShots(const edm::DetSet<SiStripDigi>& digis) {
  _shots.clear();
  addShots(digis);
}

void APVShotFinder::computeShots(const edm::DetSetVector<SiStripDigi>& digicoll) {
  _shots.clear();

  for (edm::DetSetVector<SiStripDigi>::const_iterator it = digicoll.begin(); it != digicoll.end(); ++it) {
    addShots(*it);
  }
}

void APVShotFinder::addShots(const edm::DetSet<SiStripDigi>& digis) {
  DetId detid(digis.detId());

  int laststrip = -1;
  int apv = -1;
  std::vector<SiStripDigi> temp;

  for (edm::DetSet<SiStripDigi>::const_iterator digi = digis.begin(); digi != digis.end(); digi++) {
    if (!_zs || digi->adc() > 0) {
      if (laststrip >= digi->strip())
        edm::LogWarning("StripNotInOrder") << "Strips not in order in DetSet<SiStripDigi>";
      laststrip = digi->strip();

      int newapv = digi->strip() / 128;
      if (newapv != apv) {
        if (apv >= 0) {
          if (temp.size() > 64) {
            APVShot shot(temp, detid, _zs);
            _shots.push_back(shot);
          }
          temp.clear();
        }
        apv = newapv;
      }

      temp.push_back(*digi);
    }
  }
  // last strip
  if (temp.size() > 64) {
    APVShot shot(temp, detid, _zs);
    _shots.push_back(shot);
  }
  temp.clear();
}

const std::vector<APVShot>& APVShotFinder::getShots() const { return _shots; }
