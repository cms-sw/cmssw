#include <cmath>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalCommon.h"

using namespace std;

int EcalCommon::crystalToTriggerTower(int xtal) noexcept(false) {
  if (xtal < 1 || xtal > 1700) {
    throw(std::runtime_error("ERROR:  crystalToTriggerTower:  crystal number out of bounds"));
  }

  int i = (int)floor((xtal - 1) / 20.0);
  int j = (xtal - 1) - 20 * i;
  int tti = (int)floor(i / 5.0);
  int ttj = (int)floor(j / 5.0);
  int tt = ttj + 4 * tti + 1;

  return tt;
}
