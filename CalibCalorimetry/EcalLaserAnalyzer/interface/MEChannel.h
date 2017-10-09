#ifndef MEChannel_hh
#define MEChannel_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//
// logical navigation in doubly linked list of channels

#include <vector>
#include <TROOT.h>

class MEChannel
{
public:

  MEChannel( int ix, int iy, int ii, MEChannel* mother );
  virtual  ~MEChannel();
  
  int id()       const;
  int ix() const { return _ix; }
  int iy() const { return _iy; }
  
  MEChannel* getDaughter( int ix, int iy, int ig );
  MEChannel* m() { return _m; }
  MEChannel* d( unsigned ii ) { if( ii>=n() ) return 0; return _d[ii]; }
  unsigned n() const { return _d.size(); }
  int ig() const { return _ig; }
  
  bool getListOfChannels(            std::vector< MEChannel* >& );
  bool getListOfAncestors(           std::vector< MEChannel* >& );
  bool getListOfDescendants(         std::vector< MEChannel* >& );
  bool getListOfDescendants( int ig, std::vector< MEChannel* >& );
  MEChannel* getFirstDescendant( int ig );
  MEChannel* getDescendant( int ig, int ii );
  MEChannel* getChannel(          int ix, int iy );
  MEChannel* getChannel(  int ig, int ix, int iy );
  MEChannel* getAncestor( int ig );
  
  void print(std::ostream& o, bool recursif=false ) const;
  TString oneLine( int ig );
  TString oneLine() { return oneLine( _ig ); }
  TString oneWord( int ig );
  TString oneWord() { return oneWord( _ig ); }
 
private:
  
  // doubly-linked tree
  MEChannel* _m;
  std::vector< MEChannel* > _d;
  
  // granularity
  int _ig;
  
  // (local) coordinates
  int _ix;
  int _iy;
  
  // defines a channel
  std::vector< int > _id;
  
  MEChannel* addDaughter( int ix, int iy, int ii );
  
  //  ClassDef(MEChannel,0) // MEChannel -- A channel or group of channels  
};

#endif

