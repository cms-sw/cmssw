#include <functional>
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

struct DDCompareEPV : public std::binary_function<DDExpandedView, DDExpandedView, bool> {
  DDCompareEPV();
  DDCompareEPV(const DDCompOptions& ddco);
  bool operator()(DDExpandedView& lhs, DDExpandedView& rhs) const ;
  DDCompOptions ddco_;
};

struct DDCompareCPV : public std::binary_function<DDCompactView, DDCompactView, bool> {
  DDCompareCPV();
  DDCompareCPV(const DDCompOptions& ddco);
  bool operator()(const DDCompactView& lhs, const DDCompactView& rhs) const ;
  DDCompOptions ddco_;
};

/// LogicalParts have solids which could be BooleanSolids.
/**
   This means they need to know if the DDRotation naems matter.
 **/
struct DDCompareLP : public std::binary_function<DDLogicalPart, DDLogicalPart, bool> {
  DDCompareLP();
  DDCompareLP(const DDCompOptions& ddco);
  bool operator()(const DDLogicalPart& lhs, const DDLogicalPart& rhs) const ;
  DDCompOptions ddco_;
};

/// Needs to know about rotmat because of BooleanSolid
struct DDCompareSolid : public std::binary_function<DDSolid, DDSolid, bool> {
  DDCompareSolid();
  DDCompareSolid(const DDCompOptions& ddco);
  bool operator()(const DDSolid& lhs, const DDSolid& rhs) const ;
  DDCompOptions ddco_;
};

struct DDCompareDBLVEC : public std::binary_function<std::vector<double>, std::vector<double>, bool> {
  DDCompareDBLVEC();
  DDCompareDBLVEC(double tol);
  bool operator()(const std::vector<double>& lhs, const std::vector<double>& rhs) const ;
  double tol_;
};

/// Needs to know because of Rotation Matrix of Boolean Relationship.
struct DDCompareBoolSol : public std::binary_function<DDBooleanSolid, DDBooleanSolid, bool> {
  DDCompareBoolSol();
  DDCompareBoolSol(const DDCompOptions& ddco);
  bool operator()(const DDBooleanSolid& lhs, const DDBooleanSolid& rhs) const ;
  DDCompOptions ddco_;
};

struct DDCompareDDTrans : public std::binary_function<DDTranslation, DDTranslation, bool> {
  DDCompareDDTrans();
  DDCompareDDTrans(double tol);
  bool operator()(const DDTranslation& lhs, const DDTranslation& rhs) const;
  double tol_;
};

/// Allows to compare name or not. If not, compares only values of the rotation matrix.
struct DDCompareDDRot : public std::binary_function<DDRotation, DDRotation, bool> {
  DDCompareDDRot();
  DDCompareDDRot(const DDCompOptions& ddco);
  bool operator()(const DDRotation& lhs, const DDRotation& rhs) const;
  DDCompOptions ddco_;
};

struct DDCompareDDRotMat : public std::binary_function<DDRotationMatrix, DDRotationMatrix, bool> {
  DDCompareDDRotMat();
  DDCompareDDRotMat(double tol);
  bool operator()(const DDRotationMatrix& lhs, const DDRotationMatrix& rhs) const;
  double tol_;
};




