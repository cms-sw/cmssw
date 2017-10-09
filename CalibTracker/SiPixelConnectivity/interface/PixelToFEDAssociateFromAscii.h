#ifndef PixelToFEDAssociateFromAscii_H
#define PixelToFEDAssociateFromAscii_H

/** \class PixelToFEDAssociateFromAscii
 *  Check to which FED pixel module belongs to.
 *  The associacions are read from the datafile
 */

#include <vector>
#include <string>

#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"

class PixelBarrelName;
class PixelEndcapName;
class PixelModuleName;



class PixelToFEDAssociateFromAscii : public PixelToFEDAssociate {
public:

  PixelToFEDAssociateFromAscii(const std::string & fileName);

  /// FED id for module
  virtual int operator()(const PixelModuleName &) const override;

  /// version
  virtual std::string version() const override;


  /// FED id to which barrel modul (identified by name) should be assigned 
  int operator()(const PixelBarrelName &) const;

  /// FED id to which endcape modul (identified by name) should be assigned 
  int operator()(const PixelEndcapName &) const;

private:
  /// initialisatin (read file)
  void init( const std::string & fileName);


  typedef TRange<int> Range; 

  /// define allowed (layer,module,ladder) ranges for barrel units,
  /// check if module represented by name falls in allowed ranges
  struct Bdu { int b; Range l,z,f; bool operator()(const PixelBarrelName&) const;};

  /// define allowed (endcap,disk,blade) ranges for endcap units,
  /// check if module represented by name falls in allowed ranges
  struct Edu { int e; Range d,b; bool operator()(const PixelEndcapName&) const;};

  typedef std::vector< std::pair< int, std::vector<Bdu> > > BarrelConnections;
  typedef std::vector< std::pair< int, std::vector<Edu> > > EndcapConnections;
  BarrelConnections theBarrel;
  EndcapConnections theEndcap;

private:

  std::string theVersion;

  /// initialisation (read input file)
  void send (std::pair< int, std::vector<Bdu> > & , 
      std::pair< int, std::vector<Edu> > & ) ;
  Bdu getBdu( std::string ) const;
  Edu getEdu( std::string ) const;
  Range readRange( const std::string &) const;
};
#endif 
