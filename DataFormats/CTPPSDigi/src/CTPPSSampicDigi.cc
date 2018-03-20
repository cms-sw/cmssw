/** \file
 * 
 *
 * \author Seyed Mohsen Etesami
 */

#include <DataFormats/CTPPSDigi/interface/CTPPSSampicDigi.h>

using namespace std;

CTPPSSampicDigi::CTPPSSampicDigi(unsigned int ledgt_, unsigned int tedgt_, unsigned int threvolt_, bool mhit_, unsigned short hptdcerror_) :
  ledgt(ledgt_), tedgt(tedgt_), threvolt(threvolt_), mhit(mhit_), hptdcerror(HPTDCErrorFlags(hptdcerror_))
{}

CTPPSSampicDigi::CTPPSSampicDigi() :
  ledgt(0), tedgt(0), threvolt(0), mhit(false), hptdcerror(HPTDCErrorFlags(0))
{}

// Comparison
bool
CTPPSSampicDigi::operator==(const CTPPSSampicDigi& digi) const
{
  if ( ledgt     != digi.getLeadingEdge()
   || tedgt     != digi.getTrailingEdge()
   || threvolt  != digi.getThresholdVoltage()
   || mhit      != digi.getMultipleHit()
   || hptdcerror.getErrorFlag() != digi.getHPTDCErrorFlags().getErrorFlag()) return false;
  else  
    return true; 
}

