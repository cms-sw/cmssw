#ifndef TPNCor_hh
#define TPNCor_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//
// logical navigation in doubly linked list of channels

#include <string>
#include <TROOT.h>

class TPNCor
{
public:

  TPNCor( string filename );
  virtual  ~TPNCor();
  double getPNCorrectionFactor( double val0 , int gain );
 
private:  
  double corParams[2][10][3];
  int isFileOK;

  
};

#endif

