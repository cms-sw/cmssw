#define MAXPRO 56 // MAX number of reconstructed tracks

struct MyPPSTracks {
  int n;
  int zside[MAXPRO];
  int station[MAXPRO];
  float x[MAXPRO];
  float y[MAXPRO];

};


