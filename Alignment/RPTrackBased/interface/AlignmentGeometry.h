/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_AlignmentGeometry
#define Alignment_RPTrackBased_AlignmentGeometry

#include <map>
#include <set>
#include <string>

/**
 *\brief A structure to hold relevant geometrical information about one detector/sensor.
 **/
struct DetGeometry {
  double z;                     ///< mm
  double dx, dy;                ///< sensitive direction
  double sx, sy;                ///< detector nominal shift = detector center - beam pipe; in mm
  double s;                     ///< detector nominal shift in sensitive direction; in mm

  unsigned int matrixIndex;     ///< index (0 ... AlignmentGeometry::Detectors()) within a S matrix block (for detector-related quantities)
                                ///< assigned by StraightTrackAlignment::PrepareGeometry

  unsigned int rpMatrixIndex;   ///< index (0 ... AlignmentGeometry::RPs()) within a S matrix block (for RP-related quantities)
                                ///< assigned by StraightTrackAlignment::PrepareGeometry

  bool isU;                     ///< true for U detectors, false for V detectors
                                ///< global U, V frame is used - that matches with u, v frame of the 1200 detector

  DetGeometry(double _z = 0., double _dx = 0., double _dy = 0., double _sx = 0., double _sy = 0., bool _isU = false) :
     z(_z), dx(_dx), dy(_dy), sx(_sx), sy(_sy), s(sx*dx + sy*dy), matrixIndex(0), isU(_isU) {}
};



/**
 *\brief A collection of geometrical information.
 * Map: (decimal) detector ID --> DetGeometry
 **/
class AlignmentGeometry : public std::map<unsigned int, DetGeometry>
{
  protected:
    std::set<unsigned int> rps;

  public:
    /// a characteristic z in mm
    double z0;

    /// puts an element to the map
    void Insert(unsigned int id, const DetGeometry &g)
      { insert(value_type(id, g)); rps.insert(id / 10); }

    /// returns the number of RPs in the collection
    unsigned int RPs()
      { return rps.size(); }

    /// returns the number of detectors in the collection
    unsigned int Detectors()
      { return size(); }
    
    /// returns detector id corresponding to the given matrix index
    unsigned int MatrixIndexToDetId(unsigned int) const;

    /// returns reference the the geometry of the detector with the given matrix index
    const_iterator FindByMatrixIndex(unsigned int) const;
    
    /// returns reference the the geometry of the first detector in the RP with the given rpMatrix index
    const_iterator FindFirstByRPMatrixIndex(unsigned int) const;

    /// check whether the sensor Id is valid (present in the map)
    bool ValidSensorId(unsigned int id) const
      { return (find(id) != end()); }

    /// check whether the RP Id is valid (present in the set)
    bool ValidRPId(unsigned int id) const
      { return (rps.find(id) != rps.end()); }

    /// TODO
    void Print() const;

    /// loads geometry from a text file of 5 columns:
    /// id | center x, y, z (all in mm) | read-out direction x projection, y projection
    void LoadFromFile(const std::string filename);
};

#endif

