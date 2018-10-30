#include <vector>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

class DDCompactView;
class DDExpandedView;

/// DDComparators need to know if names of DDRotation matter.
/**
   Therefore the constructors (default) are set to care about rotation names
   but if one really wants to compare without regard to the name, for example
   in the case of two DDCompactViews created in-memory from reading two 
   separate sets of XML rather than from DB objects, then one must set this
   to false.
 **/
struct DDCompOptions {
  DDCompOptions() : compRotName_(false), attResync_(false),
		    contOnError_(false), distTol_(0.0004),
		    rotTol_(0.0004), specTol_(0.0004)
  { }

  bool compRotName_;
  bool attResync_;
  bool contOnError_;
  double distTol_;
  double rotTol_;
  double specTol_;
};

bool DDCompareEPV(DDExpandedView& lhs, DDExpandedView& rhs, const DDCompOptions& ddco);
bool DDCompareCPV(const DDCompactView& lhs, const DDCompactView& rhs, const DDCompOptions& ddco);

/// LogicalParts have solids which could be BooleanSolids.
/**
   This means they need to know if the DDRotation naems matter.
 **/
bool DDCompareLP(const DDLogicalPart& lhs, const DDLogicalPart& rhs, const DDCompOptions& ddco);

/// Needs to know about rotmat because of BooleanSolid
bool DDCompareSolid(const DDSolid& lhs, const DDSolid& rhs, const DDCompOptions& ddco);

bool DDCompareDBLVEC(const std::vector<double>& lhs, const std::vector<double>& rhs, double tol=0.0004);

/// Needs to know because of Rotation Matrix of Boolean Relationship.
bool DDCompareBoolSol(const DDBooleanSolid& lhs, const DDBooleanSolid& rhs, const DDCompOptions& ddco);
bool DDCompareDDTrans(const DDTranslation& lhs, const DDTranslation& rhs, double tol=0.0004);

/// Allows to compare name or not. If not, compares only values of the rotation matrix.
bool DDCompareDDRot(const DDRotation& lhs, const DDRotation& rhs, const DDCompOptions& ddco);

bool DDCompareDDRotMat(const DDRotationMatrix& lhs, const DDRotationMatrix& rhs, double tol=0.0004);
