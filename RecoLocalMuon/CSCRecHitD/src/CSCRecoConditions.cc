#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>
#include <CondFormats/CSCObjects/interface/CSCChannelTranslator.h>
#include <iostream>

CSCRecoConditions::CSCRecoConditions( const edm::ParameterSet & ps ) : theConditions( ps ) {
}

CSCRecoConditions::~CSCRecoConditions() {
}

void CSCRecoConditions::initializeEvent( const edm::EventSetup& es ) {
  theConditions.initializeEvent( es );
}


float CSCRecoConditions::gain(const CSCDetId & id, int channel) const { 
  CSCChannelTranslator translate;
  CSCDetId idraw = translate.rawCSCDetId( id );
  int iraw = translate.rawStripChannel( id, channel );
  return theConditions.gain(idraw, iraw);
}

float CSCRecoConditions::pedestal(const CSCDetId & id, int channel) const { 
  CSCChannelTranslator translate;
  CSCDetId idraw = translate.rawCSCDetId( id );
  int iraw = translate.rawStripChannel( id, channel );
  return theConditions.pedestal(idraw, iraw);
}

float CSCRecoConditions::pedestalSigma(const CSCDetId & id, int channel) const { 
  CSCChannelTranslator translate;
  CSCDetId idraw = translate.rawCSCDetId( id );
  int iraw = translate.rawStripChannel( id, channel );
  return theConditions.pedestalSigma(idraw, iraw);
}

float CSCRecoConditions::chipCorrection(const CSCDetId & id, int geomStrip) const { 
  //printf("RecoCondition before translation e:%d s:%d r:%d c:%d l:%d strip:%d \n",id.endcap(),id.station(), id.ring(),id.chamber(),id.layer(),geomStrip);
  CSCChannelTranslator translate;
  // If ME1/4, set ring = 1
  CSCDetId idraw = translate.rawCSCDetId( id );
  // If ME1/4, collapse 48 chips into 16 electronics channel (1-48) -> (1-16)
  int geomChannel = translate.channelFromStrip( id, geomStrip );
  // Translate geometry-oriented strip channels into raw channels 
  // reordering ME+1/1a and ME-1/1b and moving ME1/1a (1-16)->(65-80)
  int iraw = translate.rawStripChannel( id, geomChannel);
  //printf("RecoCondition after  translation e:%d s:%d r:%d c:%d l:%d strip:%d \n",idraw.endcap(),idraw.station(), idraw.ring(),idraw.chamber(),idraw.layer(),iraw);
  return theConditions.chipCorrection(idraw, iraw);
}

float CSCRecoConditions::chamberTimingCorrection(const CSCDetId & id) const { 
  CSCChannelTranslator translate;
  // If ME1/4, set ring = 1
  CSCDetId idraw = translate.rawCSCDetId( id );
  return theConditions.chamberTimingCorrection(idraw);
}

float CSCRecoConditions::anodeBXoffset(const CSCDetId & id) const { 
  CSCChannelTranslator translate;
  // If ME1/4, set ring = 1
  CSCDetId idraw = translate.rawCSCDetId( id );
  return theConditions.anodeBXoffset(idraw);
}

void CSCRecoConditions::stripWeights( const CSCDetId& id, float* weights ) const {

  short int is = id.station();
  short int ir = id.ring();

  short int strip1 = 1;
  short int nStrips = 80;
  if ( is == 1 && ir == 1) nStrips = 64; // ME1b
  if ( is == 1 && ir == 3) nStrips = 64; // ME13

  if ( ir == 4 ) { // ME1a
    const CSCDetId testId( id.endcap(), 1, 1, id.chamber(), id.layer() ); // create ME11 detId
    strip1 = 65; // ME1a channels are 65-80 in ME11
    for ( short int i = 0; i < 16; ++i) {
      float w = stripWeight(testId, strip1+i);

      // Unfold ganged channels in ME1a
      weights[i]    = w;
      weights[i+16] = w;
      weights[i+32] = w;
    }
  } 
  else { // non-ME1a chambers
    for ( short int i = 0; i < nStrips; ++i) {
      weights[i] = stripWeight(id, strip1+i);
    }
  }
}

void CSCRecoConditions::noiseMatrix( const CSCDetId& id, int centralStrip, std::vector<float>& nMatrix ) const {

  // nMatrix will be filled with expanded noise matrix elements for 
  // channel 'centralStrip' and its immediate neighbours
  nMatrix.clear();

  // Initialize values in case we can't find chamber with constants
  // These are ME1/2 constants...
  float elem[15];
  elem[0] = 8.64;
  elem[1] = 3.47;
  elem[2] = 2.45;
  elem[3] = 8.60;
  elem[4] = 3.28;
  elem[5] = 1.88;
  elem[6] = 8.61;
  elem[7] = 3.18;
  elem[8] = 1.99;
  elem[9] = 7.67;
  elem[10] = 2.64;
  elem[11] = 0.;
  elem[12] = 7.71;
  elem[13] = 0.;
  elem[14] = 0.;

  short int strip1 = centralStrip;
  short int triplet[3];
  bool isME1a = (id.ring()==4);
  CSCDetId id1;

  // ME1/a constants are stored in channels 65-80 of ME11, ME1/b in channels 1-64.
  // ME1/a channels are ganged - 48 strips to 16 channels.
  if ( isME1a ) { // ME1a

    strip1 = centralStrip%16;     // strip#   1-48
    if (strip1 == 0) strip1 = 16; // channel# 1-16
    strip1 += 64;                 // strip1   65-80
    id1=CSCDetId( id.endcap(), 1, 1, id.chamber(), id.layer() ); // ME11 detId
    
    if (strip1 == 65) {
      triplet[0]=80;
      triplet[1]=65;
      triplet[2]=66;
    } else if (strip1 == 80) {
      triplet[0]=79;
      triplet[1]=80;
      triplet[2]=65;
    } else {
      triplet[0]=strip1-1;
      triplet[1]=strip1;
      triplet[2]=strip1+1;

    }
  } 
  else { // non-ME1a
      triplet[0]=strip1-1;
      triplet[1]=strip1;
      triplet[2]=strip1+1;
  }

  for ( short int i = 0; i < 3; ++i) {
  
    short int channel = triplet[i];
    float w;
    std::vector<float> me(12);

    if ( isME1a ) {
      w = stripWeight(id1, channel);
      theConditions.noiseMatrixElements(id1, channel, me);
    }
    else {
      w = stripWeight(id, channel);
      theConditions.noiseMatrixElements(id, channel, me);
    }
    w = w*w;
    for ( short int j=0; j<11; ++j ) {
      elem[j] = me[j] * w;     
    }
    elem[11]= 0.; 
    elem[12]= me[11] * w;
    elem[13]= 0.;
    elem[14]= 0.;
      
    // Test that elements make sense:
    bool isFlawed = false;      
    for ( int k = 0; k < 15; ++k) {
      // make sure the number isn't too close to zero...
      if (elem[k] < 0.001) elem[k] = 0.001;
      // make sure the number isn't too big...
      if (elem[k] > 50.) isFlawed = true; 
    }

    if ( isFlawed ) {
      // These are fake ME1/2:
      elem[0] = 8.64;
      elem[1] = 3.47;
      elem[2] = 2.45;
      elem[3] = 8.60;
      elem[4] = 3.28;
      elem[5] = 1.88;
      elem[6] = 8.61;
      elem[7] = 3.18;
      elem[8] = 1.99;
      elem[9] = 7.67;
      elem[10] = 2.64;
      elem[11] = 0.;
      elem[12] = 7.71;
      elem[13] = 0.;
      elem[14] = 0.;
    }

    for (int k = 0; k < 15; ++k) nMatrix.push_back( elem[k] );
  }
}

void CSCRecoConditions::crossTalk( const CSCDetId& id, int centralStrip, std::vector<float>& xtalks) const {

  xtalks.clear();

  short int strip1 = centralStrip;
  short int triplet[3];
  bool isME1a = (id.ring()==4);
  CSCDetId id1;

  // ME1/a constants are stored in channels 65-80 of ME11, ME1/b in channels 1-64.
  // ME1/a channels are ganged - 48 strips to 16 channels.
  if ( isME1a ) { // ME1a

    strip1 = centralStrip%16;     // strip#   1-48
    if (strip1 == 0) strip1 = 16; // channel# 1-16
    strip1 += 64;                 // strip1   65-80
    id1=CSCDetId( id.endcap(), 1, 1, id.chamber(), id.layer() ); // ME11 detId
    
    if (strip1 == 65) {
      triplet[0]=80;
      triplet[1]=65;
      triplet[2]=66;
    } else if (strip1 == 80) {
      triplet[0]=79;
      triplet[1]=80;
      triplet[2]=65;
    } else {
      triplet[0]=strip1-1;
      triplet[1]=strip1;
      triplet[2]=strip1+1;
    }
  } 
  else { // non-ME1a
      triplet[0]=strip1-1;
      triplet[1]=strip1;
      triplet[2]=strip1+1;
  }

  // For 3 neighbouring strips get crosstalks
  short int idx = 0;
  for ( short int i = 0; i < 3; ++i) {
  
    short int channel = triplet[i];
    std::vector<float> ct(4);

    if ( isME1a ) {
      theConditions.crossTalk(id1, channel, ct);
    }
    else {
      theConditions.crossTalk(id, channel, ct);
    }

    xtalks.push_back(ct[0]);
    xtalks.push_back(ct[1]);
    xtalks.push_back(ct[2]);
    xtalks.push_back(ct[3]);
    ++idx;
  }
}

/// Test for neighbouring bad strip
/// I'm a bit confused about this - it returns true if strip is
/// not at edge, and either of its neighbours is bad.

bool CSCRecoConditions::nearBadStrip( const CSCDetId& id, int geomStrip ) const {
  // Note ME1A strip runs 1-48 
  /*
  CSCChannelTranslator translate;
  CSCDetId idraw = translate.rawCSCDetId( id );
  int geomChan = translate.channelFromStrip( id, geomStrip ); 
  int rawChan = translate.rawStripChannel( id, geomChan ); 

  const std::bitset<80>& badStrips = theConditions.badStripWord(idraw);

  bool nearBad = false;
  if( rawChan>1 && rawChan<80 ){ // 80 bits max, labelled 0-79. Test 1-78 for neighbours.
    nearBad = (badStrips.test(rawChan) || badStrips.test(rawChan-2));
  }
  */
  //
  bool nearBad = (badStrip(id,geomStrip-1) || badStrip(id,geomStrip+1));


  return nearBad;
}
//
/// Test for a bad strip

bool CSCRecoConditions::badStrip( const CSCDetId& id, int geomStrip ) const {
  // Note ME1A strip runs 1-48 
  bool aBadS = false;
  if(geomStrip>0 && geomStrip<81){ 
    CSCChannelTranslator translate;
    CSCDetId idraw = translate.rawCSCDetId( id );
    int geomChan = translate.channelFromStrip( id, geomStrip ); 
    int rawChan = translate.rawStripChannel( id, geomChan ); 

    const std::bitset<80>& badStrips = theConditions.badStripWord(idraw);

    if( rawChan>0 && rawChan<81 ){ // 80 bits max, labelled 0-79. Test 0-79 (i.e. any - that's the idea)
      aBadS = badStrips.test(rawChan-1);
    }
  }
  return aBadS;
}

/// Get bad wiregroup word
const std::bitset<112>& CSCRecoConditions::badWireWord( const CSCDetId& id ) const {
    return theConditions.badWireWord( id );
}

// This expects raw ME11 detId for ME1b (channels 1-64) & for ME1a (channels 65-80)
float CSCRecoConditions::stripWeight( const CSCDetId& id, int channel ) const {
   float w = averageGain() / theConditions.gain(id, channel);
  
   // Weights are forced to lie within 0.5 and 1.5
   if (w > 1.5) w = 1.5;
   if (w < 0.5) w = 0.5;
   return w;
}
