#ifndef CSCToAFEB_h
#define CSCToAFEB_h

/**\class  CSCToAFEB
 *
 * CSC layer, wire vs AFEB channel map. 
 *
 * \author N. Terentiev, CMU
 */

class CSCToAFEB{

private:
  int layer_wire_to_channel_[6][8];
  int layer_wire_to_board_[6][8];
  int station_ring_to_nmxafeb_[4][3];
  int station_ring_to_nmxwire_[4][3];
public:

  /// Constructor

CSCToAFEB (){

  for(int i=1; i<=6; i++) for(int j=1;j<=8;j++) {
    if(i==1 || i==3 || i==5) {
      if(j<5) layer_wire_to_channel_[i-1][j-1] =j+4;
      if(j>4) layer_wire_to_channel_[i-1][j-1] =j+8;
    }
    if(i==2 || i==4 || i==6) {
      if(j<5) layer_wire_to_channel_[i-1][j-1] =j;
      if(j>4) layer_wire_to_channel_[i-1][j-1] =j+4;
    }
  }

  for(int i=1; i<=6; i++) for(int j=1;j<=8;j++) 
    layer_wire_to_board_[i-1][j-1]=(i-1)/2+1;
  
  for(int i=1; i<=4; i++) for(int j=1;j<=3;j++) {
    if(i==1) {
      if(j==1) station_ring_to_nmxafeb_[i-1][j-1]=18;
      if(j==2) station_ring_to_nmxafeb_[i-1][j-1]=24;
      if(j==3) station_ring_to_nmxafeb_[i-1][j-1]=12;
    }
    if(i==2) {
      if(j==1) station_ring_to_nmxafeb_[i-1][j-1]=42;
      if(j==2) station_ring_to_nmxafeb_[i-1][j-1]=24;
      if(j==3) station_ring_to_nmxafeb_[i-1][j-1]=0;
    }
    if(i==3) {
      if(j==1) station_ring_to_nmxafeb_[i-1][j-1]=36;
      if(j==2) station_ring_to_nmxafeb_[i-1][j-1]=24;
      if(j==3) station_ring_to_nmxafeb_[i-1][j-1]=0;
    }
    if(i==4) {
      if(j==1) station_ring_to_nmxafeb_[i-1][j-1]=36;
      if(j==2) station_ring_to_nmxafeb_[i-1][j-1]=24;
      if(j==3) station_ring_to_nmxafeb_[i-1][j-1]=0;
    }
  } 

  for(int i=1; i<=4; i++) for(int j=1;j<=3;j++) {
    if(i==1) {
      if(j==1) station_ring_to_nmxwire_[i-1][j-1]=48;
      if(j==2) station_ring_to_nmxwire_[i-1][j-1]=64;
      if(j==3) station_ring_to_nmxwire_[i-1][j-1]=32;
    }
    if(i==2) {
      if(j==1) station_ring_to_nmxwire_[i-1][j-1]=112;
      if(j==2) station_ring_to_nmxwire_[i-1][j-1]=64;
      if(j==3) station_ring_to_nmxwire_[i-1][j-1]=0;
    }
    if(i==3) {
      if(j==1) station_ring_to_nmxwire_[i-1][j-1]=96;
      if(j==2) station_ring_to_nmxwire_[i-1][j-1]=64;
      if(j==3) station_ring_to_nmxwire_[i-1][j-1]=0;
    }
    if(i==4) {
      if(j==1) station_ring_to_nmxwire_[i-1][j-1]=96;
      if(j==2) station_ring_to_nmxwire_[i-1][j-1]=64;
      if(j==3) station_ring_to_nmxwire_[i-1][j-1]=0;
    }
  } 


  /*
  layer_wire_to_channel_[6][8] =     {{ 5,6,7,8,13,14,15,16 },
                                      { 1,2,3,4, 9,10,11,12 },
                                      { 5,6,7,8,13,14,15,16 },
                                      { 1,2,3,4, 9,10,11,12 },
                                      { 5,6,7,8,13,14,15,16 },
                                      { 1,2,3,4, 9,10,11,12 }};
  gives AFEB channel number for given layer and wire numbers.

  layer_wire_to_board_[6][8]   =     {{ 1,1,1,1,1,1,1,1 },
                                      { 1,1,1,1,1,1,1,1 },
                                      { 2,2,2,2,2,2,2,2 },
                                      { 2,2,2,2,2,2,2,2 },
                                      { 3,3,3,3,3,3,3,3 },
                                      { 3,3,3,3,3,3,3,3 }};
  gives position of AFEB in column for given layer and wire numbers.

  station_ring_to_nmxafeb_[4][3]=    {{18?,24,12},
                                      {42, 24,0 },
                                      {36, 24,0 ],
                                      {36, 24,0 }}; 
  gives max. # of AFEBs in CSC of different types for given station and ring.

  station_ring_to_nmxwire_[4][3]=    {{48?,64,32},
                                      {112,64,0 },
                                      {96, 64,0 ],
                                      {96, 64,0 }};
  gives max. # of wiregroups in one layer of CSC of different types
  for given station and ring.
  */
  }

  /// return AFEB channel number
  int getAfebCh(int layer, int wiregroup) const;
  /// return AFEB position number
  int getAfebPos(int layer, int wiregroup) const; 
  /// return layer number
  int getLayer(int afeb, int channel) const;
  /// return wiregroup number
  int getWireGroup(int afeb, int channel) const; 
  /// return max. number of AFEBs
  int getMaxAfeb(int station, int ring) const;
  /// return max. number of wiregroups per layer
  int getMaxWire(int station, int ring) const;
 
  /// Print content
  void print() const;

};

#endif
