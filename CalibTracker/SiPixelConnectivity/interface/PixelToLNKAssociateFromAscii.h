#ifndef PixelToLNKAssociateFromAscii_H
#define PixelToLNKAssociateFromAscii_H

/** \class PixelToLNKAssociateFromAscii
 *  Check to which FED pixel module belongs to.
 *  The associacions are read from the datafile
 */

#include <vector>
#include <string>
#include <map>

#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"

class PixelBarrelName;
class PixelEndcapName;
class PixelModuleName;

class PixelToLNKAssociateFromAscii : public PixelToFEDAssociate {
public:

  typedef PixelToFEDAssociate::CablingRocId CablingRocId;
  typedef PixelToFEDAssociate::DetectorRocId DetectorRocId;

  PixelToLNKAssociateFromAscii(const std::string & fileName);

  virtual const CablingRocId * operator()(const DetectorRocId& roc) const;

  /// version
  virtual std::string version() const;

private:
  typedef TRange<int> Range; 

  /// initialisatin (read file)
  void init( const std::string & fileName);
  void addConnections( int fedId, int linkId, std::string module, Range rocDetIds);


  std::string theVersion;
  std::vector< std::pair<DetectorRocId,CablingRocId> > theConnection;

  Range readRange( const std::string &) const;
    
};
#endif 
