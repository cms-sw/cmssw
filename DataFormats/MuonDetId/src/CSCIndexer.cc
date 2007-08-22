#include <DataFormats/MuonDetId/interface/CSCIndexer.h>
#include <iostream>

void CSCIndexer::fillChamberLabel() const {
  // Fill the member vector which permits decoding of the linear chamber index
  // Logically const since initializes cache only
  chamberLabel.resize( 235 ); // one more than #chambers per endcap
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

CSCDetId CSCIndexer::detIdFromChamberIndex_OLD( IndexType ici ) const {
  IndexType ie = 1;
  if (ici > 234 ) {
     ie = 2;
     ici -= 234; 
  }
  // Now ici is in range 1-234 (assuming valid input in range 1-468)

  // MEij pairs...
  const IndexType station[] = {0,1,1,1,2,2,3,3,4};
  const IndexType ring[]    = {0,1,2,3,1,2,1,2,1};

  // MEij preceding a given MEij matching linear index above
  const IndexType prevs[] =  {0,0,1,1,1,2,2,3,3}; 
  const IndexType prevr[] =  {0,0,1,2,3,1,2,1,2};

  IndexType is = 4;
  IndexType ir = 1;
  for ( IndexType i = 2; i<=8; ++i) {
    IndexType js = station[i];
    IndexType jr = ring[i];
    // if it's before start of MEjs/jr then it's in the previous MEis/ir
      if ( ici < startChamberIndexInEndcap(js,jr) ) {
	is = prevs[i];
	ir = prevr[i];
	break;
      }
      // otherwise it's in ME41
  }
  IndexType ic = ici - startChamberIndexInEndcap(is,ir) + 1;

  return CSCDetId( ie, is, ir, ic );
}
 
CSCDetId CSCIndexer::detIdFromChamberIndex( IndexType ici ) const {
  IndexType ie = 1;
  if ( ici > 234 ) {
	  ie = 2;
	  ici -= 234;
  }

  if (chamberLabel.empty()) fillChamberLabel();
  IndexType label = chamberLabel[ici];  
  return detIdFromChamberLabel( ie, label );
}

CSCDetId CSCIndexer::detIdFromChamberLabel( IndexType ie, IndexType label ) const {

  IndexType is = label/1000;
  label -= is*1000;
  IndexType ir = label/100;
  label -= ir*100;
  IndexType ic = label;

  return CSCDetId( ie, is, ir, ic );
}

CSCDetId CSCIndexer::detIdFromLayerIndex( IndexType ili ) const {
  IndexType ie = 1;
  if ( ili > 1404 ) {
     ie = 2;
     ili -= 1404;
  }
  IndexType il = (ili-1)%6 + 1;
  IndexType ici = (ili-1)/6 + 1;
  if (chamberLabel.empty()) fillChamberLabel();
  IndexType label = chamberLabel[ici];
  CSCDetId id = detIdFromChamberLabel(ie, label);
	
  return CSCDetId(ie, id.station(), id.ring(), id.chamber(), il);
}

std::pair<CSCDetId, CSCIndexer::IndexType>  CSCIndexer::detIdFromStripChannelIndex( LongIndexType isi ) const {

   const LongIndexType lastPlusEndcap = 108864; // = 217728/2
   const LongIndexType firstme13  = 34561; // First channel of ME13
   const LongIndexType lastme13   = 48384; // Last channel of ME13

   const IndexType lastPlusEndcapLayer = 1404; // = 2808/2
   const IndexType firstme13layer  = 433; // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
   const IndexType lastme13layer   = 648; // = 108*6
	
   IndexType ie = 1;
   if ( isi > lastPlusEndcap ) {
      ie = 2;
      isi -= lastPlusEndcap;
   }
	
   // Defaults: before ME13
   LongIndexType istart = 0;
   IndexType nchan = 80;
   IndexType layerOffset = 0;
	
   if ( isi > lastme13 ) { // after ME13
      istart = lastme13;
      layerOffset = lastme13layer;
   }
   else if ( isi >= firstme13 ) { // ME13
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchan = 64;
   }

   isi -= istart; // remove earlier group(s)
   IndexType ichan = (isi-1)%nchan + 1;
   IndexType ili = (isi-1)/nchan + 1;
   ili += layerOffset; // add appropriate offset for earlier group(s)
   if ( ie != 1 ) ili+= lastPlusEndcapLayer; // add offset to -z endcap
	
   return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichan);
}

 
CSCIndexer::IndexType CSCIndexer::checkLabel( IndexType ici ) const {
  if (chamberLabel.empty()) fillChamberLabel();
  return chamberLabel[ici];  
}
