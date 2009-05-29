#ifndef MagGeoBuilderFromDDD_H
#define MagGeoBuilderFromDDD_H

/** \class MagGeoBuilderFromDDD
 *  Parse the XML magnetic geometry, build individual volumes and match their
 *  shared surfaces. Build MagVolume6Faces and organise them in a hierarchical
 *  structure. Build MagGeometry out of it.
 *
 *  $Date: 2009/01/21 12:05:14 $
 *  $Revision: 1.11 $
 *  \author N. Amapane - INFN Torino
 */
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h" 
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <string>
#include <vector>
#include <map>

class Surface;
class MagBLayer;
class MagESector;
class MagVolume6Faces;
namespace magneticfield {
  class VolumeBasedMagneticFieldESProducer;
  class AutoMagneticFieldESProducer;
}


class MagGeoBuilderFromDDD  {
public:
  /// Constructor. 
  /// overrideMasterSector is a hack to allow switching between 
  /// phi-symmetric maps and maps with sector-specific tables. 
  /// It won't be necessary anymore once the geometry is decoupled from 
  /// the specification of tables, ie when tables will come from the DB.
  MagGeoBuilderFromDDD(std::string version_, bool debug=false, bool overrideMasterSector=false);

  /// Destructor
  virtual ~MagGeoBuilderFromDDD();

  ///  Set scaling factors for individual volumes. 
  /// "keys" is a vector of 100*volume number + sector (sector 0 = all sectors)
  /// "values" are the corresponding scaling factors 
  void setScaling(std::vector<int> keys, std::vector<double> values);

  /// Get barrel layers
  std::vector<MagBLayer*> barrelLayers() const;

  /// Get endcap layers
  std::vector<MagESector*> endcapSectors() const;

  float maxR() const;

  float maxZ() const;  

private:
  typedef ConstReferenceCountingPointer<Surface> RCPS;

  // Build the geometry. 
  //virtual void build();
  virtual void build(const DDCompactView & cpv);


  // FIXME: only for temporary tests and debug, to be removed
  friend class TestMagVolume;
  friend class MagGeometry;
  friend class magneticfield::VolumeBasedMagneticFieldESProducer;
  friend class magneticfield::AutoMagneticFieldESProducer;


  std::vector<MagVolume6Faces*> barrelVolumes() const;  
  std::vector<MagVolume6Faces*> endcapVolumes() const;

  // Temporary container to manipulate volumes and their surfaces.
  class volumeHandle;
  typedef std::vector<volumeHandle*> handles;

  // Build interpolator for the volume with "correct" rotation
  void buildInterpolator(const volumeHandle * vol, 
			 std::map<std::string, MagProviderInterpol*>& interpolators);

  // Build all MagVolumes setting the MagProviderInterpol
  void buildMagVolumes(const handles & volumes,
		       std::map<std::string, MagProviderInterpol*> & interpolators);

  // Print checksums for surfaces.
  void summary(handles & volumes);

  // Perform simple sanity checks
  void testInside(handles & volumes);

  // A layer of barrel volumes.
  class bLayer;
  // A sector of volumes in a layer.
  class bSector;
  // A rod of volumes in a sector.
  class bRod;
  // A slab of volumes in a rod.
  class bSlab;
  // A sector of endcap volumes.
  class eSector;  
  // A layer of volumes in an endcap sector.
  class eLayer;
 
  
  // Extractors for precomputed_value_sort (to sort containers of volumeHandles). 
  typedef std::unary_function<const volumeHandle*, double> uFcn;
  struct ExtractZ;
  struct ExtractAbsZ;
  struct ExtractPhi;
  struct ExtractPhiMax;
  struct ExtractR;
  struct ExtractRN;
  struct LessDPhi;
  // This one to be used only for max_element and min_element
  struct LessZ;

  handles bVolumes;  // the barrel volumes.
  handles eVolumes;  // the endcap volumes.

  std::vector<MagBLayer*> mBLayers; // Finally built barrel geometry
  std::vector<MagESector*> mESectors; // Finally built barrel geometry

  std::string version; // Version of the data files to be used
  
  std::map<int, double> theScalingFactors;

  bool overrideMasterSector; // see comment on ctor

  static bool debug;

};
#endif
