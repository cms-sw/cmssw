#include <L1Trigger/TrackFindingTMTT/interface/HTcell.h>
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"

namespace TMTT {

//=== Initialization with cfg params, 
//=== rapidity range of current sector, and estimated q/Pt of cell,
//=== and the bin number of the cell along the q/Pt axis of the r-phi HT array,
//=== and a flag indicating if this cell is the merge of smaller HT cells.


void HTcell::init(const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg,
		  float etaMinSector, float etaMaxSector, float qOverPt, unsigned int ibin_qOverPt, 
		  bool mergedCell, bool miniHTcell) 
{
  settings_ = settings;

  // Sector number
  iPhiSec_ = iPhiSec;
  iEtaReg_ = iEtaReg;

  // Note track q/Pt. 
  // In this case of an r-phi HT, each cell corresponds to a unique q/Pt.
  // In the case of an r-z HT, it is assumed that we know q/Pt from previously run r-phi HT.
  qOverPtCell_ = qOverPt;
  // Note bin number of cell along q/Pt axis of r-phi HT array. (Not used if r-z HT).
  ibin_qOverPt_ = ibin_qOverPt;
  mergedCell_ = mergedCell;
  // Is cell in Mini-HT?
  miniHTcell_ = miniHTcell;
  // Rapidity range of sector.
  etaMinSector_ = etaMinSector;
  etaMaxSector_ = etaMaxSector;

  invPtToDphi_   = settings->invPtToDphi();  // B*c/2E11

  // Use filter in each HT cell using only stubs which have consistent bend?
  useBendFilter_ = settings->useBendFilter();

  // A filter is used each HT cell, which prevents more than the specified number of stubs being stored in the cell. (Reflecting memory limit of hardware).
  if (miniHTcell_) {
    maxStubsInCell_ = settings->maxStubsInCellMiniHough();
  } else {
    maxStubsInCell_ = settings->maxStubsInCell();
  }

  // Check if subsectors are being used within each sector. These are only ever used for r-phi HT.
  numSubSecs_ = settings->numSubSecsEta();
}

//=== Termination. Search for track in this HT cell etc.

void HTcell::end(){
  // Produce list of filtered stubs by applying all requested filters (e.g. on stub bend).
  // (If no filters are requested, then filtered & unfiltered stub collections will be identical).

  // N.B. Other filters,  such as the r-z filters, which the firmware runs after the HT because they are too slow within it,
  // are not defined here, but instead inside class TrkFilterAfterRphiHT.


  vFilteredStubs_ = vStubs_;
  if (useBendFilter_) vFilteredStubs_ = this->bendFilter(vFilteredStubs_);

  // Prevent too many stubs being stored in a single HT cell if requested (to reflect hardware memory limits).
  // N.B. This MUST be the last filter applied.
  if (maxStubsInCell_ <= 99) vFilteredStubs_ = this->maxStubCountFilter(vFilteredStubs_);

  // Calculate the number of layers the filtered stubs in this cell are in.
  numFilteredLayersInCell_ = this->calcNumFilteredLayers();

  if (numSubSecs_ > 1) { 
    // If using subsectors within each sector, calculate the number of layers the filters stubs in this cell are in,
    // when one considers only the subset of the stubs within each subsector.
    // Look for the "best" subsector.
    numFilteredLayersInCellBestSubSec_ = 0;
    for (unsigned int i = 0; i < numSubSecs_; i++) {
      unsigned int numLaySubSec = this->calcNumFilteredLayers(i);
      numFilteredLayersInCellBestSubSec_ = max(numFilteredLayersInCellBestSubSec_, numLaySubSec);
    }
  } else {
    // If only 1 sub-sector, then subsector and sector are identical.
    numFilteredLayersInCellBestSubSec_ = numFilteredLayersInCell_;
  }
}

// Calculate how many tracker layers the filter stubs in this cell are in, when only the subset of those stubs
// that are in the specified subsector are counted.

unsigned int HTcell::calcNumFilteredLayers(unsigned int iSubSec) const {
  vector<const Stub*> stubsInSubSec;
  for (const Stub* s : vFilteredStubs_) {
    const vector<bool>& inSubSec = subSectors_.at(s); // Find out which subsectors this stub is in.
    if (inSubSec[iSubSec]) stubsInSubSec.push_back(s);
  }
  return Utility::countLayers( settings_, stubsInSubSec );
}


//=== Produce a filtered collection of stubs in this cell that all have consistent bend.
//=== Only called for r-phi Hough transform.

vector<const Stub*> HTcell::bendFilter( const vector<const Stub*>& stubs ) const {

  // Create bend-filtered stub collection.
  vector<const Stub*> filteredStubs;
  for (const Stub* s : stubs) {

    // Require stub bend to be consistent with q/Pt of this cell.

    unsigned int minBin = s->min_qOverPt_bin();
    unsigned int maxBin = s->max_qOverPt_bin();
    if ( mergedCell_ ) {
      if ( minBin % 2 == 1 ) minBin--;
      // Next line not wanted with current m-bin range definition in Stub::calcQoverPtRange().
      //if ( maxBin % 2 == 1 ) maxBin++;
    }
    if (minBin <= ibin_qOverPt_ && ibin_qOverPt_ <= maxBin )  filteredStubs.push_back(s);
  }

  return filteredStubs;
}

//=== Filter stubs so as to prevent more than specified number of stubs being stored in one cell.
//=== This reflects finite memory of hardware.

vector<const Stub*> HTcell::maxStubCountFilter( const vector<const Stub*>& stubs ) const {
  vector<const Stub*> filteredStubs;
  // If there are too many stubs in a cell, the hardware keeps (maxStubsInCell - 1) of the first stubs in the list
  // plus the last stub.  
  if (stubs.size() > maxStubsInCell_) {
    for (unsigned int i = 0; i < maxStubsInCell_ - 1; i++) { // first stubs
      filteredStubs.push_back(stubs[i]);
    }
    filteredStubs.push_back(stubs[ stubs.size() - 1] ); // plus last stub
  } else {
    filteredStubs = stubs;
  }
  return filteredStubs;
}

}
