/** \file
 * 
 *  $Date: 2007/05/03 23:27:45 $
 *  $Revision: 1.12 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <bitset>

using namespace std;

// Constructors
CSCStripDigi::CSCStripDigi (const int & istrip, const vector<int> & vADCCounts, const vector<uint16_t> & vADCOverflow,
			    const vector<uint16_t> & vOverlap, const vector<uint16_t> & vErrorstat ):
  strip(istrip),
  ADCCounts(vADCCounts),
  ADCOverflow(vADCOverflow),
  OverlappedSample(vOverlap),
  Errorstat(vErrorstat)
{
}

CSCStripDigi::CSCStripDigi (const int & istrip, const vector<int> & vADCCounts):
  strip(istrip),
  ADCCounts(vADCCounts),
  ADCOverflow(8,0),
  OverlappedSample(8,0),
  Errorstat(8,0)
{
}


CSCStripDigi::CSCStripDigi ():
  strip(0),
  ADCCounts(8,0),
  ADCOverflow(8,0),
  OverlappedSample(8,0),
  Errorstat(8,0)
{
}

// Comparison
bool
CSCStripDigi::operator == (const CSCStripDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getADCCounts().size() != digi.getADCCounts().size() ) return false;
  if ( getADCCounts() != digi.getADCCounts() ) return false;
  return true;
}

// Getters
int CSCStripDigi::getStrip() const { return strip; }
std::vector<int> CSCStripDigi::getADCCounts() const { return ADCCounts; }


// Setters
void CSCStripDigi::setStrip(int istrip) {
  strip = istrip;
}
void CSCStripDigi::setADCCounts(vector<int>vADCCounts) {
  bool badVal = false;
  for (int i=0; i<(int)vADCCounts.size(); i++) {
    if (vADCCounts[i] < 1) badVal = true;
  }
  if ( !badVal ) {
    ADCCounts = vADCCounts;
  } else {
    vector<int> ZeroCounts(8,0);
    ADCCounts = ZeroCounts;
  }
}

// Debug
void
CSCStripDigi::print() const {
  cout << "CSC Strip: " << getStrip() << "  ADC Counts: ";
  for (int i=0; i<(int)getADCCounts().size(); i++) {cout << getADCCounts()[i] << " ";}
  cout << "\n";
}




