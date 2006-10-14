#ifndef PixelToFEDAssociate_H
#define PixelToFEDAssociate_H

/** \class PixelToFEDAssociate
 *  Check to which FED pixel module belongs to.
 *  The associacions are read from the datafile
 */

#include <vector>
#include <string>
#include <iostream>

#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"

class PixelBarrelName;
class PixelEndcapName;
class PixelModuleName;


class PixelToFEDAssociate {
public:

  /// FED id for module
  int operator()(const PixelModuleName &) const;

  /// FED id to which barrel modul (identified by name) should be assigned 
  int operator()(const PixelBarrelName &) const;

  /// FED id to which endcape modul (identified by name) should be assigned 
  int operator()(const PixelEndcapName &) const;

  void init() const;
private:
  typedef TRange<int> Range; 

  /// define allowed (layer,module,ladder) ranges for barrel units,
  /// check if module represented by name falls in allowed ranges
  struct Bdu { Range l,z,f; bool operator()(const PixelBarrelName&) const;};

  /// define allowed (endcap,disk,blade) ranges for endcap units,
  /// check if module represented by name falls in allowed ranges
  struct Edu { Range e,d,b; bool operator()(const PixelEndcapName&) const;};

  typedef vector< pair< int, vector<Bdu> > > BarrelConnections;
  typedef vector< pair< int, vector<Edu> > > EndcapConnections;
  static BarrelConnections theBarrel;
  static EndcapConnections theEndcap;

  static bool isInitialised;

private:

  /// initialisation (read input file)
  void send (pair< int, vector<Bdu> > & , pair< int, vector<Edu> > & ) const;
  Bdu getBdu( string ) const;
  Edu getEdu( string ) const;
  Range readRange( const string &) const;
};
#endif 
