#include "MuonAnalysis/MuonAssociators/interface/SegmentLCTMatchBox.h"

#include <TMath.h>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

const int SegmentLCTMatchBox::_alctEnvelopes[] = { 2, 1, 0, 1, 2, 2 };
const int SegmentLCTMatchBox::_clctEnvelopes[] = { 5, 2, 0, 2, 4, 5 };


SegmentLCTMatchBox::SegmentLCTMatchBox( int nHitsSegment, int nHitsALCT, int nHitsCLCT, int printLevel ):
  _printLevel    ( printLevel   ),
  _nHitsSegment  ( nHitsSegment ), 
  _nHitsALCT     ( nHitsALCT    ),
  _nHitsCLCT     ( nHitsCLCT    )
{
}

SegmentLCTMatchBox::~SegmentLCTMatchBox(){

}

int SegmentLCTMatchBox::me11aNormalize ( int halfStrip ){

  int retVal = (halfStrip-1)%32+1;

    return retVal;

}

int SegmentLCTMatchBox::halfStrip( const CSCRecHit2D &hit ){

  int retVal;

  if (hit.channels().size() == 3)

    retVal =  2.0* (hit.channels()[1] + hit.positionWithinStrip() - 0.5 );
  
  else

    retVal =  2.0 * (hit.channels()[0] - hit.positionWithinStrip() - 0.5 );

  bool evenLayer =  hit.cscDetId().layer() % 2 == 0;

  if ( evenLayer )
    retVal -= 1;

  if ( (hit.cscDetId().station() == 1) && (hit.cscDetId().layer() != 3) && evenLayer )
    retVal += 1;

  return retVal;
  
}

int SegmentLCTMatchBox::wireGroup ( const CSCRecHit2D &hit ){

  return ( hit.wgroups()[0] -1 );

}

bool SegmentLCTMatchBox::isMatched ( const CSCSegment &segment, edm::Handle<CSCCorrelatedLCTDigiCollection> CSCTFlcts,
				     int *match_report ){

  if (_printLevel > 2)
    std::cout << "*** BEGIN MATCHING *** " << std::endl;
  
  if (match_report)
    (*match_report) = 0;

  bool retVal=false;

  int LCT_key_strip = -999;
  int LCT_key_wireg = -999;

  const int noMatchVal = 9999;

  int keyStripEstim = noMatchVal;
  int keyWiregEstim = noMatchVal;

  CSCDetId *tempDetId= 0;
  const CSCDetId &origId = segment.cscDetId();

  // index for histogram filling ... 0 for ME11a, station() for the rest..

  // if we're in ME11a, we have to worry about triple-ganging of strips.

  bool me11aStation = false;

  if (segment.cscDetId().ring() == 4){
    
    me11aStation = true;
    
    tempDetId = new CSCDetId ( origId.endcap(), origId.station(), 1,origId.chamber());

  } else {

    tempDetId = new CSCDetId ( origId );

  }

  double stripSum = 0, wiregSum = 0, numHits = 0;


  // first, find the estimator for the key strip and wire group from recHits

  const std::vector<CSCRecHit2D>& theHits = segment.specificRecHits();
  std::vector<CSCRecHit2D>::const_iterator hitIter;

  bool hadKeyInfo = false;
  
  for (hitIter = theHits.begin(); hitIter!= theHits.end(); hitIter++){

    if ( hitIter -> cscDetId() . layer() == 3){

      hadKeyInfo = true;

      keyStripEstim = halfStrip ( *hitIter );
      keyWiregEstim = wireGroup ( *hitIter );

    }
    
    stripSum += halfStrip ( *hitIter );
    wiregSum += wireGroup ( *hitIter );
    numHits  += 1.0;
  }

  if (!hadKeyInfo){ // no key info .. have to improvise with averages..
    
    if (_printLevel > 1)
      std::cout << "MATCHING: NO KEY INFO!!!" << std::endl;

    keyStripEstim = stripSum / numHits;
    keyWiregEstim = wiregSum / numHits;

  }

  if (me11aStation){
    keyStripEstim = me11aNormalize (keyStripEstim);
  }

  int numLCTsChamber = 0;

  CSCCorrelatedLCTDigiCollection::Range lctRange = CSCTFlcts -> get( *tempDetId );

  int deltaWireg = 999, deltaStrip = 999;
  
  if ( _printLevel >= 0) 
    std::cout << " segment CSCDetId " << segment.cscDetId() << std::endl;
  

  bool lctWiregMatch = false;
  bool lctStripMatch = false;

  for(CSCCorrelatedLCTDigiCollection::const_iterator lct = lctRange.first ; lct != lctRange.second; lct++ ){

    if (match_report)
      (*match_report) |= MATCH_CHAMBER;
    
    if ( _printLevel >= 0)
      std::cout << (*lct) << std::endl;

    LCT_key_wireg = lct -> getKeyWG();
    LCT_key_strip = lct -> getStrip();

    numLCTsChamber++;
   
    if (me11aStation)
      LCT_key_strip = me11aNormalize( LCT_key_strip );

    deltaWireg = keyWiregEstim - LCT_key_wireg;
    deltaStrip = keyStripEstim - LCT_key_strip;

    if (me11aStation){ // the ganging of ME11a causes wraparound effects at the boundaries for delta strip 

      if (deltaStrip > 16) deltaStrip -= 32;
      if (deltaStrip < -16) deltaStrip += 32;

    }

    lctWiregMatch |= ( abs(deltaWireg) <=  5 );
    lctStripMatch |= ( abs(deltaStrip) <= 10 );

  }
  
  if (lctWiregMatch && match_report)
    (*match_report) |= MATCH_WIREG;
  
  if (lctStripMatch && match_report)
    (*match_report) |= MATCH_STRIP;
  
  retVal =  lctWiregMatch && lctStripMatch ;
	  
  if (!retVal && (_printLevel > 1) )
    std::cout << "FAIL: retVal was " << retVal 
	      << " numLCTS was: " << numLCTsChamber 
	      << segment.cscDetId() << std::endl;

  if (_printLevel > 3)
    std::cout << "*** END MATCHING *** " << std::endl;

  delete tempDetId;

  return retVal;
  
}

bool SegmentLCTMatchBox::isLCTAble ( const CSCSegment &segment, int *match_report ){

  if (match_report)
    (*match_report) = 0;

  if (segment . nRecHits() < 4 )         return false;

  if (_printLevel >= 2)
    std::cout << "*** BEGIN FIDUCIAL *** " << std::endl;

  bool hadKeyInfo = false;

  int thisStation = segment.cscDetId().station();  

  int keyStrip = 999, keyWireg = 999;

  const std::vector<CSCRecHit2D>& theHits = segment . specificRecHits();
	    
  std::vector<CSCRecHit2D>::const_iterator hitIter;

  double sumStrip = 0, sumWireg = 0, nHits = 0;

  for (hitIter = theHits.begin(); hitIter!= theHits.end(); hitIter++){

    if (hitIter -> cscDetId(). layer() == 3){

      hadKeyInfo = true;
    
      keyStrip = halfStrip (*hitIter);
      keyWireg = wireGroup (*hitIter);

      if (match_report)
	(*match_report) |= MATCH_HASKEY;

    } 

    sumStrip += halfStrip( *hitIter );
    sumWireg += wireGroup( *hitIter );
    nHits+= 1.0;

    if (_printLevel > 1){
      std::cout << "layer: " << hitIter -> cscDetId(). layer() << " " 
		<< " number of strips participating: " << hitIter -> channels().size() << std::endl;
      if ( hitIter -> channels().size()==1 )
	std::cout << hitIter -> channels()[0] << " " << hitIter -> positionWithinStrip() 
		  << halfStrip(*hitIter) << std::endl;
    }
    
  }	    

  if (!hadKeyInfo){ // no hit in key layer... improvize with the averages

    keyStrip = TMath::FloorNint( sumStrip / nHits + 0.5 );
    keyWireg = TMath::FloorNint( sumWireg / nHits + 0.5 );

    if (_printLevel > 1)
      std::cout << "sumStrip: " << sumStrip << " sumWireg: " <<  sumWireg << " nHits: " << nHits << std::endl;
    

  }

  int hitsFidAlct = 0;
  int hitsFidClct = 0;

  if (_printLevel > 1)
    std::cout << "key wg, strip: " << keyWireg <<  ", " << keyStrip << std::endl;


  for (hitIter = theHits.begin(); hitIter!= theHits.end(); hitIter++){

    int thisLayer = hitIter -> cscDetId() . layer();

    int delWgroup = wireGroup( *hitIter ) - keyWireg;
    int delStrip  = halfStrip( *hitIter ) - keyStrip;
    
    if (_printLevel > 1){ // debug why this match didn't work
      
      std::cout << "layer: " << thisLayer << "wg,st: " << wireGroup( *hitIter ) << ", " << halfStrip ( *hitIter )
		<< "deltas: " << delWgroup << ", " << delStrip ;
      
    }
    
    int histoFillIndex;
    
    if (segment.cscDetId().ring() == 4)
      histoFillIndex = 0;
    else
      histoFillIndex = segment.cscDetId().station();
    
    if (thisLayer <=3)
      delWgroup = -delWgroup;
    
    if (thisStation == 3)
      delWgroup = -delWgroup;
    
    if (thisStation == 4)
      delWgroup = -delWgroup;
    
    if ( delWgroup >=0 )
      if ( delWgroup <= (_alctEnvelopes[ thisLayer - 1 ]) ) hitsFidAlct++;
    
    if ( abs(delStrip)  <= (_clctEnvelopes[ thisLayer - 1 ]) ) hitsFidClct++;
    
    if (_printLevel >1)
      std::cout << " hitsFid alct: " << hitsFidAlct << " clct: " << hitsFidClct << std::endl;

  }

  if (!hadKeyInfo && (_printLevel > 2)) std::cout << "NO KEY INFO!" << std::endl;

  if (_printLevel >= 2)
    std::cout << "*** END FIDUCIAL *** " << std::endl;

  if ( hitsFidAlct < 3) return false;
  if ( hitsFidClct < 3) return false;

  if (!hadKeyInfo && (_printLevel > 2) ) std::cout << "NO KEY INFO AND FIDUCIAL! " << segment.cscDetId() << std::endl;

  
  return true;

}

// it will return true if the segment is matched by an LCT and this LCT belongs
// to one of the CSCTF triggering tracks
bool SegmentLCTMatchBox::belongsToTrigger ( const CSCSegment &segment, 
                                            edm::Handle<L1CSCTrackCollection> CSCTFtracks,
                                            edm::Handle<CSCCorrelatedLCTDigiCollection> CSCTFlcts){


  if (_printLevel > 2)
    std::cout << "*** CHECKING IF THE SEGMENT BELONGS TO A TRIGGER *** " 
              << std::endl;
  
  bool retVal=false;

  int LCT_key_strip = -999;
  int LCT_key_wireg = -999;

  const int noMatchVal = 9999;

  int keyStripEstim = noMatchVal;
  int keyWiregEstim = noMatchVal;

  CSCDetId *tempDetId= 0;
  const CSCDetId &origId = segment.cscDetId();


  // if we're in ME11a, we have to worry about triple-ganging of strips.

  bool me11aStation = false;

  if (segment.cscDetId().ring() == 4){
    
    me11aStation = true;
    
    tempDetId = new CSCDetId ( origId.endcap(), origId.station(), 1,origId.chamber());

  } else {

    tempDetId = new CSCDetId ( origId );

  }

  double stripSum = 0, wiregSum = 0, numHits = 0;


  // first, find the estimator for the key strip and wire group from recHits

  const std::vector<CSCRecHit2D>& theHits = segment.specificRecHits();
  std::vector<CSCRecHit2D>::const_iterator hitIter;

  bool hadKeyInfo = false;
  
  for (hitIter = theHits.begin(); hitIter!= theHits.end(); hitIter++){

    if ( hitIter -> cscDetId() . layer() == 3){

      hadKeyInfo = true;

      keyStripEstim = halfStrip ( *hitIter );
      keyWiregEstim = wireGroup ( *hitIter );

    }
    
    stripSum += halfStrip ( *hitIter );
    wiregSum += wireGroup ( *hitIter );
    numHits  += 1.0;
  }

  if (!hadKeyInfo){ // no key info .. have to improvise with averages..
    
    if (_printLevel > 1)
      std::cout << "MATCHING: NO KEY INFO!!!" << std::endl;

    keyStripEstim = stripSum / numHits;
    keyWiregEstim = wiregSum / numHits;

  }

  if (me11aStation){
    keyStripEstim = me11aNormalize (keyStripEstim);
  }

  int numLCTsChamber = 0;

  CSCCorrelatedLCTDigiCollection::Range lctRange = CSCTFlcts -> get( *tempDetId );

  int deltaWireg = 999, deltaStrip = 999;
  
  if ( _printLevel >= 0) 
    std::cout << " segment CSCDetId " << segment.cscDetId() << std::endl;
  

  bool lctWiregMatch = false;
  bool lctStripMatch = false;

  for(CSCCorrelatedLCTDigiCollection::const_iterator lct = lctRange.first ; lct != lctRange.second; lct++ ){

    if ( _printLevel >= 0)
      std::cout << (*lct) << std::endl;

    LCT_key_wireg = lct -> getKeyWG();
    LCT_key_strip = lct -> getStrip();

    numLCTsChamber++;
   
    if (me11aStation)
      LCT_key_strip = me11aNormalize( LCT_key_strip );

    deltaWireg = keyWiregEstim - LCT_key_wireg;
    deltaStrip = keyStripEstim - LCT_key_strip;

    if (me11aStation){ // the ganging of ME11a causes wraparound effects at the boundaries for delta strip 

      if (deltaStrip > 16) deltaStrip -= 32;
      if (deltaStrip < -16) deltaStrip += 32;

    }

    lctWiregMatch |= ( abs(deltaWireg) <=  5 );
    lctStripMatch |= ( abs(deltaStrip) <= 10 );

    bool isMatched = lctWiregMatch && lctStripMatch ;

    if (isMatched) {

      // loop over the track to see if the matched segment belong to a triggering TF track 
      for(L1CSCTrackCollection::const_iterator trk=CSCTFtracks->begin(); trk<CSCTFtracks->end(); trk++){
        
        // For each trk, get the list of its LCTs
        CSCCorrelatedLCTDigiCollection lctsOfTrack = trk -> second;
  
        // loop over all the lcts of the track
        for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator lctOfTrk = lctsOfTrack.begin(); lctOfTrk  != lctsOfTrack.end()  ; lctOfTrk++){
	  
          //CSCCorrelatedLCTDigiCollection::Range lctOfTrkRange = 
          //lctsOfTracks.get((*lctOfTrk).first);
	  CSCCorrelatedLCTDigiCollection::Range lctOfTrkRange = (*lctOfTrk).second;

          if (lctRange == lctOfTrkRange) retVal = true;
          
      
        }// loop over lcts of a track
      
      }// loop over the CSCTF tracks
    
    } // is Matched?
  }
  
  if (!retVal && (_printLevel > 1) )
    std::cout << "FAIL: retVal was " << retVal 
              << segment.cscDetId() << std::endl;

  if (_printLevel > 3)
    std::cout << "*** END MATCHING TO THE CSCTF TRIGGER*** " << std::endl;

  delete tempDetId;

  return retVal;
  
}


// -999: no matching
//  1: bad phi road. Not good extrapolation, but still triggering
// 11: singles
// 15: halo
// 2->10 and 12->14: coincidence trigger with good extrapolation

// it is equivalent to belongsToTrigger but instead of a boolean it returns
// the track mode. -999 mean no trigger found. 
int SegmentLCTMatchBox::whichMode ( const CSCSegment &segment, 
                                    edm::Handle<L1CSCTrackCollection> CSCTFtracks,
                                    edm::Handle<CSCCorrelatedLCTDigiCollection> CSCTFlcts){


  if (_printLevel > 2)
    std::cout << "*** CHECKING IF THE SEGMENT BELONGS TO A TRIGGER *** " 
              << std::endl;
  
  int retVal=-999;

  int LCT_key_strip = -999;
  int LCT_key_wireg = -999;

  const int noMatchVal = 9999;

  int keyStripEstim = noMatchVal;
  int keyWiregEstim = noMatchVal;

  CSCDetId *tempDetId= 0;
  const CSCDetId &origId = segment.cscDetId();


  // if we're in ME11a, we have to worry about triple-ganging of strips.

  bool me11aStation = false;

  if (segment.cscDetId().ring() == 4){
    
    me11aStation = true;
    
    tempDetId = new CSCDetId ( origId.endcap(), origId.station(), 1,origId.chamber());

  } else {

    tempDetId = new CSCDetId ( origId );

  }

  double stripSum = 0, wiregSum = 0, numHits = 0;


  // first, find the estimator for the key strip and wire group from recHits

  const std::vector<CSCRecHit2D>& theHits = segment.specificRecHits();
  std::vector<CSCRecHit2D>::const_iterator hitIter;

  bool hadKeyInfo = false;
  
  for (hitIter = theHits.begin(); hitIter!= theHits.end(); hitIter++){

    if ( hitIter -> cscDetId() . layer() == 3){

      hadKeyInfo = true;

      keyStripEstim = halfStrip ( *hitIter );
      keyWiregEstim = wireGroup ( *hitIter );

    }
    
    stripSum += halfStrip ( *hitIter );
    wiregSum += wireGroup ( *hitIter );
    numHits  += 1.0;
  }

  if (!hadKeyInfo){ // no key info .. have to improvise with averages..
    
    if (_printLevel > 1)
      std::cout << "MATCHING: NO KEY INFO!!!" << std::endl;

    keyStripEstim = stripSum / numHits;
    keyWiregEstim = wiregSum / numHits;

  }

  if (me11aStation){
    keyStripEstim = me11aNormalize (keyStripEstim);
  }

  int numLCTsChamber = 0;

  CSCCorrelatedLCTDigiCollection::Range lctRange = CSCTFlcts -> get( *tempDetId );

  int deltaWireg = 999, deltaStrip = 999;
  
  if ( _printLevel >= 0) 
    std::cout << " segment CSCDetId " << segment.cscDetId() << std::endl;
  

  bool lctWiregMatch = false;
  bool lctStripMatch = false;

  for(CSCCorrelatedLCTDigiCollection::const_iterator lct = lctRange.first ; lct != lctRange.second; lct++ ){

    if ( _printLevel >= 0)
      std::cout << (*lct) << std::endl;

    LCT_key_wireg = lct -> getKeyWG();
    LCT_key_strip = lct -> getStrip();

    numLCTsChamber++;
   
    if (me11aStation)
      LCT_key_strip = me11aNormalize( LCT_key_strip );

    deltaWireg = keyWiregEstim - LCT_key_wireg;
    deltaStrip = keyStripEstim - LCT_key_strip;

    if (me11aStation){ // the ganging of ME11a causes wraparound effects at the boundaries for delta strip 

      if (deltaStrip > 16) deltaStrip -= 32;
      if (deltaStrip < -16) deltaStrip += 32;

    }

    lctWiregMatch |= ( abs(deltaWireg) <=  5 );
    lctStripMatch |= ( abs(deltaStrip) <= 10 );
    
    bool isMatched = lctWiregMatch && lctStripMatch ;
    
    if (isMatched) {
      
      // debugging
      //std::cout << "I AM MATCHED\n";
      //std::cout << "detId: "   << *tempDetId        << std::endl;
      //std::cout << "strip: "   << lct -> getStrip() << std::endl;
      //std::cout << "wg: "      << lct -> getKeyWG() << std::endl;
      
      // loop over the track to see if the matched segment belong to a triggering TF track 
      for(L1CSCTrackCollection::const_iterator trk=CSCTFtracks->begin(); trk<CSCTFtracks->end(); trk++){
        
        // For each trk, get the list of its LCTs
        CSCCorrelatedLCTDigiCollection lctsOfTrack = trk -> second;
        
        // loop over all the lcts of the track
        for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator lctOfTrk = lctsOfTrack.begin(); lctOfTrk  != lctsOfTrack.end()  ; lctOfTrk++){
          
          CSCCorrelatedLCTDigiCollection::Range lctOfTrkRange = lctsOfTrack.get((*lctOfTrk).first);
          
          for(CSCCorrelatedLCTDigiCollection::const_iterator lctTRK=lctOfTrkRange.first; lctTRK!=lctOfTrkRange.second; lctTRK++){
            
            // debugging
            //std::cout << "test detId: "   << (*lctOfTrk).first    << std::endl;
            //std::cout << "test strip: "   << lctTRK -> getStrip() << std::endl;
            //std::cout << "test wg: "      << lctTRK -> getKeyWG() << std::endl;
 
            //std::cout << "  test DetId: " << (*lctOfTrk).first  << std::endl;   	
            //std::cout << "  lctRange2: " <<  *tempDetId << std::endl; 
  	
            // matching DetId, strip and WG
            if ( (*tempDetId        == (*lctOfTrk).first   ) &&
                 (lct -> getStrip() == lctTRK -> getStrip()) &&
                 (lct -> getKeyWG() == lctTRK -> getKeyWG()) ){
              
              //std::cout << "MATCHED\n";
              // PtAddress gives an handle on other parameters such as the trk mode
              ptadd thePtAddress(trk->first.ptLUTAddress());
         
              retVal = thePtAddress.track_mode;  
            }
          }
          
        }// loop over lcts of a track
        
      }// loop over the CSCTF tracks
      
    }// is Matched?
  }
  
  
  if (_printLevel > 3)
    std::cout << "*** END MATCHING TO THE CSCTF TRIGGER*** " << std::endl;
  
  delete tempDetId;
  
  return retVal;
  
}

