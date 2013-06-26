#ifndef DTOccupancyCluster_H
#define DTOccupancyCluster_H

/** \class DTOccupancyCluster
 *  Cluster of DTOccupancyPoint used bt DTOccupancyTest to spot problematic layers.
 *  Layers are clusterized in the plane average cell occupancy - RMS of the cell occupancies.
 *  
 *  $Date: 2008/10/16 09:33:39 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTOccupancyPoint.h"

#include <vector>
#include <string>
#include <math.h>
#include <set>

class TH2F;

class DTOccupancyCluster {
public:
  /// Constructor
  // Contruct the cluster from a couple of point
  DTOccupancyCluster(const DTOccupancyPoint& firstPoint, const DTOccupancyPoint& secondPoint);

  /// Constructor
  // Contruct the cluster from a single point: you can not add points to such a cluster
  DTOccupancyCluster(const DTOccupancyPoint& singlePoint);

  /// Destructor
  virtual ~DTOccupancyCluster();

  // Operations
  
  /// Check if the cluster candidate satisfies the quality requirements
  bool isValid() const;

  /// Add a point to the cluster: returns false if the point does not satisfy the
  /// quality requirement
  bool addPoint(const DTOccupancyPoint& anotherPoint);

  /// Compute the distance of a single point from the cluster
  /// (minimum distance with respect to the cluster points)
  double distance(const DTOccupancyPoint& point) const;

  /// average cell occupancy of the layers in the cluster
  double averageMean() const;

  /// average RMS of the cell occpuancy distributions of the layers in the cluster
  double averageRMS() const;

  /// max average cell occupancy of the layers in the cluster
  double maxMean() const;
  
  /// max RMS of the cell occpuancy distributions of the layers in the cluster
  double maxRMS() const;

  /// get a TH2F displaying the cluster
  TH2F * getHisto(std::string histoName, int nBinsX, double minX, double maxX,
		  int nBinsY, double minY, double maxY, int fillColor) const;

  /// # of layers belonging to the cluster
  int nPoints() const;

  std::set<DTLayerId> getLayerIDs() const;

protected:

private:
  
  bool qualityCriterion(const DTOccupancyPoint& firstPoint, const DTOccupancyPoint& secondPoint);
  
  bool  qualityCriterion(const DTOccupancyPoint& anotherPoint);

  void computeRadius();

  bool theValidity;
  double radius;
  std::vector<DTOccupancyPoint> thePoints;

  double theMaxMean;
  double theMaxRMS;
  double theMeanSum;
  double theRMSSum;

};

/// for DTOccupancyCluster sorting
bool clusterIsLessThan(const DTOccupancyCluster& clusterOne, const DTOccupancyCluster& clusterTwo);

#endif

