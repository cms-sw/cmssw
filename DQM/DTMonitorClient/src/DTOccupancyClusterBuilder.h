#ifndef DTOccupancyClusterBuilder_H
#define DTOccupancyClusterBuilder_H

/** \class DTOccupancyClusterBuilder
 *  Build clusters of layer occupancies (DTOccupancyCluster) to spot problematic layers.
 *  It's used by DTOccupancyTest.
 *
 *  $Date: 2008/10/16 09:33:39 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTOccupancyPoint.h"
#include "DTOccupancyCluster.h"

#include <set>
#include <map>
#include <vector>
#include <string>


class DTOccupancyClusterBuilder {
public:
  /// Constructor
  DTOccupancyClusterBuilder();

  /// Destructor
  virtual ~DTOccupancyClusterBuilder();

  // Operations
  /// Add an occupancy point for a given layer
  void addPoint(const DTOccupancyPoint& point);

  /// build the clusters
  void buildClusters();

  /// draw a TH2F histograms showing the clusters 
  void drawClusters(std::string canvasName);

  /// get the cluster correspondig to "normal" cell occupancy.
  DTOccupancyCluster getBestCluster() const;

  bool isProblematic(DTLayerId layerId) const;

protected:

private:
  std::pair<DTOccupancyPoint, DTOccupancyPoint> getInitialPair();

  void computePointToPointDistances();

  void computeDistancesToCluster(const DTOccupancyCluster& cluster);

  bool buildNewCluster();

  void sortClusters();
  
  std::set<DTOccupancyPoint> thePoints;
  std::map<double, std::pair<DTOccupancyPoint, DTOccupancyPoint> > theDistances;
  std::map<double, DTOccupancyPoint> theDistancesFromTheCluster;
  std::vector<DTOccupancyCluster> theClusters;
  std::set<DTLayerId> theProblematicLayers;

  double maxMean;
  double maxRMS;

};

#endif

