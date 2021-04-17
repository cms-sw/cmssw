/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_AlignmentTask_h
#define CalibPPS_AlignmentRelative_AlignmentTask_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
class AlignmentConstraint;
class CTPPSGeometry;

#include <vector>

/**
 *\brief Represents an alignment task.
 **/
class AlignmentTask {
public:
  // -------------------- config file parameters --------------------

  /// whether to resolve detector shifts in readout direction(s)
  bool resolveShR;

  /// whether to resolve detector shifts in z
  bool resolveShZ;

  /// whether to resolve detector rotations around z
  bool resolveRotZ;

  /// whether to resolve only 1 rot_z per RP
  bool oneRotZPerPot;

  /// whether to apply the constraint mean U = mean V RotZ for strips ("standard" set of constraints only)
  bool useEqualMeanUMeanVRotZConstraints;

  /// fixed detectors constraints from config file
  edm::ParameterSet fixedDetectorsConstraints;

  /// settings of "standard" constraints from config file
  edm::ParameterSet standardConstraints;

  // -------------------- geometry-related members --------------------

  /// the geometry for this task
  AlignmentGeometry geometry;

  /// builds the alignment geometry
  static void buildGeometry(const std::vector<unsigned int> &rpDecIds,
                            const std::vector<unsigned int> &excludedSensors,
                            const CTPPSGeometry *,
                            double z0,
                            AlignmentGeometry &geometry);

  // -------------------- quantity-class-related members --------------------

  /// quantity classes
  enum QuantityClass {
    qcShR1,  ///< detector shifts in first readout direction
    qcShR2,  ///< detector shifts in second readout direction
    qcShZ,   ///< detector shifts in z
    qcRotZ,  ///< detector rotations around z
  };

  /// list of quantity classes to be optimized
  std::vector<QuantityClass> quantityClasses;

  /// returns a string tag for the given quantity class
  std::string quantityClassTag(QuantityClass) const;

  struct DetIdDirIdxPair {
    unsigned int detId;
    unsigned int dirIdx;

    bool operator<(const DetIdDirIdxPair &other) const {
      if (detId < other.detId)
        return true;
      if (detId > other.detId)
        return false;
      if (dirIdx < other.dirIdx)
        return true;

      return false;
    }
  };

  /// for each quantity class contains mapping (detector id, direction) --> measurement index
  std::map<QuantityClass, std::map<DetIdDirIdxPair, unsigned int>> mapMeasurementIndeces;

  /// for each quantity class contains mapping detector id --> quantity index
  std::map<QuantityClass, std::map<unsigned int, unsigned int>> mapQuantityIndeces;

  /// builds "mapMatrixIndeces" from "geometry"
  void buildIndexMaps();

  /// returns the number of quantities of the given class
  unsigned int measurementsOfClass(QuantityClass) const;

  /// returns the number of quantities of the given class
  unsigned int quantitiesOfClass(QuantityClass) const;

  /// returns measurement index (if non-existent, returns -1)
  signed int getMeasurementIndex(QuantityClass cl, unsigned int detId, unsigned int dirIdx) const;

  /// returns measurement index (if non-existent, returns -1)
  signed int getQuantityIndex(QuantityClass cl, unsigned int detId) const;

  // -------------------- constraint-related members --------------------

  /// builds a set of fixed-detector constraints
  void buildFixedDetectorsConstraints(std::vector<AlignmentConstraint> &) const;

  /// builds the standard constraints
  void buildStandardConstraints(std::vector<AlignmentConstraint> &) const;

  /// adds constraints such that only 1 rot_z per RP is left
  void buildOneRotZPerPotConstraints(std::vector<AlignmentConstraint> &) const;

  /// adds constraints such that only mean-U and mean-V RotZ are equal for each strip RP
  void buildEqualMeanUMeanVRotZConstraints(std::vector<AlignmentConstraint> &constraints) const;

  // -------------------- constructors --------------------

  /// dummy constructor (not to be used)
  AlignmentTask() {}

  /// normal constructor
  AlignmentTask(const edm::ParameterSet &ps);
};

#endif
