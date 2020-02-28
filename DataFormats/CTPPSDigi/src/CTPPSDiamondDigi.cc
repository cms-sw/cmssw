/** \file
 *
 *
 * \author Seyed Mohsen Etesami
 */

#include <DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h>

using namespace std;

CTPPSDiamondDigi::CTPPSDiamondDigi(
    unsigned int ledgt_, unsigned int tedgt_, unsigned int threvolt_, bool mhit_, unsigned short hptdcerror_)
    : ledgt(ledgt_), tedgt(tedgt_), threvolt(threvolt_), mhit(mhit_), hptdcerror(HPTDCErrorFlags(hptdcerror_)) {}

CTPPSDiamondDigi::CTPPSDiamondDigi() : ledgt(0), tedgt(0), threvolt(0), mhit(false), hptdcerror(HPTDCErrorFlags(0)) {}

// Comparison
bool CTPPSDiamondDigi::operator==(const CTPPSDiamondDigi& digi) const {
  if (ledgt != digi.leadingEdge() || tedgt != digi.trailingEdge() || threvolt != digi.thresholdVoltage() ||
      mhit != digi.multipleHit() || hptdcerror.errorFlag() != digi.hptdcErrorFlags().errorFlag())
    return false;
  else
    return true;
}
