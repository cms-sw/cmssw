#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"

void CSCIndexerBase::fillChamberLabel() const {
  // Fill the member vector which permits decoding of the linear chamber index
  // Logically const since initializes cache only,
  // Beware that the ME42 indices 235-270 within this vector do NOT correspond to
  // their 'real' linear indices (which are 469-504 for +z)
   chamberLabel.resize( 271 ); // one more than #chambers per endcap. Includes ME42.
   IndexType count = 0;
   chamberLabel[count] = 0;

   for ( IndexType is = 1 ; is != 5; ++is ) {
      IndexType irmax = ringsInStation(is);
      for ( IndexType ir = 1; ir != irmax+1; ++ir ) {
         IndexType icmax = chambersInRingOfStation(is, ir);
         for ( IndexType ic = 1; ic != icmax+1; ++ic ) {
     chamberLabel[ ++count ] = is*1000 + ir*100 + ic ;
         }
      }
   }
}

CSCIndexerBase::IndexType CSCIndexerBase::chamberLabelFromChamberIndex( IndexType ici ) const {

  // This is just for cross-checking

  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  if ( ici > 468 ) {
    // ME42
    ici -= 234; // now in range 235-306
    if ( ici > 270 ) { // -z
      ici -= 36; // now in range 235-270
    }
  }
  else { // in range 1-468
    if ( ici > 234 ) { // -z
      ici -= 234; // now in range 1-234
    }
  }
  if (chamberLabel.empty()) fillChamberLabel();
  return chamberLabel[ici];

}

CSCIndexerBase::IndexType CSCIndexerBase::hvSegmentIndex(IndexType is, IndexType ir, IndexType iwire ) const {

    IndexType hvSegment = 1;   // There is only one HV segment in ME1/1

    if (is > 2 && ir == 1) {        // HV segments are the same in ME3/1 and ME4/1
      if      ( iwire >= 33 && iwire <= 64 ) { hvSegment = 2; }
      else if ( iwire >= 65 && iwire <= 96 ) { hvSegment = 3; }

    } else if (is > 1 && ir == 2) { // HV segments are the same in ME2/2, ME3/2, and ME4/2
      if      ( iwire >= 17 && iwire <= 28 ) { hvSegment = 2; }
      else if ( iwire >= 29 && iwire <= 40 ) { hvSegment = 3; }
      else if ( iwire >= 41 && iwire <= 52 ) { hvSegment = 4; }
      else if ( iwire >= 53 && iwire <= 64 ) { hvSegment = 5; }

    } else if (is == 1 && ir == 2) {
      if      ( iwire >= 25 && iwire <= 48 ) { hvSegment = 2; }
      else if ( iwire >= 49 && iwire <= 64 ) { hvSegment = 3; }

    } else if (is == 1 && ir == 3) {
      if      ( iwire >= 13 && iwire <= 22 ) { hvSegment = 2; }
      else if ( iwire >= 23 && iwire <= 32 ) { hvSegment = 3; }

    } else if (is == 2 && ir == 1) {
      if      ( iwire >= 45 && iwire <= 80 ) { hvSegment = 2; }
      else if ( iwire >= 81 && iwire <= 112) { hvSegment = 3; }

    }

    return hvSegment;
  }

CSCDetId CSCIndexerBase::detIdFromChamberLabel( IndexType ie, IndexType label ) const {

  IndexType is = label/1000;
  label -= is*1000;
  IndexType ir = label/100;
  label -= ir*100;
  IndexType ic = label;

  return CSCDetId( ie, is, ir, ic );
}

CSCDetId CSCIndexerBase::detIdFromChamberIndex( IndexType ici ) const {
  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  IndexType ie = 1;
  if ( ici > 468 ) {
    // ME42
    ici -= 234; // now in range 235-306
    if ( ici > 270 ) { // -z
      ie = 2;
      ici -= 36; // now in range 235-270
    }
  }
  else { // in range 1-468
    if ( ici > 234 ) { // -z
      ie = 2;
      ici -= 234; // now in range 1-234
    }
  }
  if (chamberLabel.empty()) fillChamberLabel();
  IndexType label = chamberLabel[ici];
  return detIdFromChamberLabel( ie, label );
}

CSCDetId CSCIndexerBase::detIdFromLayerIndex( IndexType ili ) const {

  IndexType il = (ili-1)%6 + 1;
  IndexType ici = (ili-1)/6 + 1;
  CSCDetId id = detIdFromChamberIndex( ici );

  return CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), il);
}
