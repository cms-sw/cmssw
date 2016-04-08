/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_AlignmentTask
#define Alignment_RPTrackBased_AlignmentTask

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"
class AlignmentConstraint;
class TotemRPGeometry;

#include <vector>


/**
 *\brief Represents an alignment task.
 **/
class AlignmentTask {
  public:
    /// quantity classes
    enum QuantityClass {
      qcShR,    ///< detector shifts in readout direction
      qcShZ,    ///< detector shifts in z
      qcRotZ,   ///< detector rotations around z
      qcRPShZ   ///< RP shifts in z
    };
    
    /// list of quantity classes to be optimized
    std::vector<QuantityClass> quantityClasses;

    /// returns a string tag for the given quantity class
    static std::string QuantityClassTag(QuantityClass);

    /// whether to resolve detector shifts in readout direction
    bool resolveShR;

    /// whether to resolve detector shifts in z
    bool resolveShZ;

    /// whether to resolve detector rotations around z
    bool resolveRotZ;

    /// whether to resolve RP shifts in z
    bool resolveRPShZ;

    /// whether the per-group constraint shall be applied
    bool useExtendedRotZConstraint;
    
    /// whether the per-group constraint shall be applied
    bool useZeroThetaRotZConstraint;

    /// whether the per-group constraints shall be applied
    bool useExtendedShZConstraints;

    /// whether the second (c_i ~ z^RP_i) constraint shall be applied
    bool useExtendedRPShZConstraint;

    /// whether to resolve only 1 rot_z per RP
    bool oneRotZPerPot;

    /// the geometry for this task
    AlignmentGeometry geometry;
   
    /// dummy constructor (not to be used)
    AlignmentTask() {}
    
    /// normal constructor
    AlignmentTask(const edm::ParameterSet& ps);

    /// builds the alignment geometry
    static void BuildGeometry(const std::vector<unsigned int> &RPIds,
        const std::vector<unsigned int> excludePlanes, const TotemRPGeometry *,
        double z0, AlignmentGeometry &geometry);
    
    /// homogeneous constraints from config file
    edm::ParameterSet homogeneousConstraints;

    /// fixed detectors constraints from config file
    edm::ParameterSet fixedDetectorsConstraints;

    /// returns the number of quantities of the given class
    unsigned int QuantitiesOfClass(QuantityClass);
    
    /// returns the number of constraints of the given class
    unsigned int ConstraintsForClass(QuantityClass);
    
    /// builds a set of homogeneous constraints
    void BuildHomogeneousConstraints(std::vector<AlignmentConstraint>&);
    
    /// builds a set of fixed-detector constraints
    void BuildFixedDetectorsConstraints(std::vector<AlignmentConstraint>&);
    
    /// builds the agreed constraints for final analysis
    void BuildOfficialConstraints(std::vector<AlignmentConstraint>&);

    /// adds constraints such that only 1 rot_z per RP is left
    void BuildOneRotZPerPotConstraints(std::vector<AlignmentConstraint>&);
};

#endif

