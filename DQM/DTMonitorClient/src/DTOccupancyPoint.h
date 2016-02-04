#ifndef DTOccupancyPoint_H
#define DTOccupancyPoint_H

/** \class DTOccupancyPoint
 *  This class is used for evaluation of layer occupancy in DTOccupancyTest.
 *  It describes a point in the 2D plane (average cell occupancy vs cell occupancy RMS).
 *
 *  $Date: 2008/07/02 16:50:29 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include <DataFormats/MuonDetId/interface/DTLayerId.h>

class DTOccupancyPoint {
public:
  /// Constructor
  DTOccupancyPoint();

  DTOccupancyPoint(double mean, double rms);

  DTOccupancyPoint(double mean, double rms, DTLayerId layerId);

  /// Destructor
  virtual ~DTOccupancyPoint();

  // Operations

  /// average cell occupancy in the layer
  double mean() const;
  
  /// RMS of the distribution of the cell occupancies in the layer
  double rms() const;

  /// distance from another point in the 2D plane
  double distance(const DTOccupancyPoint& anotherPoint) const;
  
  double deltaMean(const DTOccupancyPoint& anotherPoint) const;

  double deltaRMS(const DTOccupancyPoint& anotherPoint) const;

  bool operator==(const DTOccupancyPoint& other) const;

  bool operator!=(const DTOccupancyPoint& other) const;
  
  bool operator<(const DTOccupancyPoint& other) const;

  void setLayerId(DTLayerId layerId);

  DTLayerId layerId() const;

private:

  double theMean;
  double theRMS;
  DTLayerId theLayerId;
  
  bool debug; // FIXME: to be removed

};

// Compute the average RMS among two DTOccupancyPoints
double computeAverageRMS(const DTOccupancyPoint& onePoint, const DTOccupancyPoint& anotherPoint);

// Compute the min RMS among two DTOccupancyPoints
double computeMinRMS(const DTOccupancyPoint& onePoint, const DTOccupancyPoint& anotherPoint);

#endif

