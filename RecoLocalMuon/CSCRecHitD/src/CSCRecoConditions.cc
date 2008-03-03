#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>
#include <iostream>

CSCRecoConditions::CSCRecoConditions( const edm::ParameterSet & ps ) : theConditions( ps ) {
}

CSCRecoConditions::~CSCRecoConditions() {
}

void CSCRecoConditions::initializeEvent( const edm::EventSetup& es ) {
  theConditions.initializeEvent( es );
}

// This expects ME11 detId for ME1b (channels 1-64) AND for ME1a (channels 65-80)
float CSCRecoConditions::stripWeight( const CSCDetId& id, int channel ) const {
   float w = averageGain() / gain(id, channel);
  
   // Weights are forced to lie within 0.5 and 1.5
   if (w > 1.5) w = 1.5;
   if (w < 0.5) w = 0.5;
   return w;
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

