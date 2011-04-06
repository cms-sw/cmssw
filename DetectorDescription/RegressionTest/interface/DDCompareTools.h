#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

/// DDComparators need to know if names of DDRotation matter.
/**
   Therefore the constructors (default) are set to care about rotation names
   but if one really wants to compare without regard to the name, for example
   in the case of two DDCompactViews created in-memory from reading two 
   separate sets of XML rather than from DB objects, then one must set this
   to false.
 **/
struct DDCompareEPV : public std::binary_function<DDExpandedView, DDExpandedView, bool> {
  DDCompareEPV();
  DDCompareEPV(bool compmat);
  bool operator()(DDExpandedView& lhs, DDExpandedView& rhs) const ;
  bool compMatOnly_;
};

struct DDCompareCPV : public std::binary_function<DDCompactView, DDCompactView, bool> {
  DDCompareCPV();
  DDCompareCPV(bool compmat);
  bool operator()(const DDCompactView& lhs, const DDCompactView& rhs) const ;
  bool compMatOnly_;
};

/// LogicalParts have solids which could be BooleanSolids.
/**
   This means they need to know if the DDRotation naems matter.
 **/
struct DDCompareLP : public std::binary_function<DDLogicalPart, DDLogicalPart, bool> {
  DDCompareLP();
  DDCompareLP(bool compmat);
  bool operator()(const DDLogicalPart& lhs, const DDLogicalPart& rhs) const ;
  bool compMatOnly_;
};

/// Needs to know about rotmat because of BooleanSolid
struct DDCompareSolid : public std::binary_function<DDSolid, DDSolid, bool> {
  DDCompareSolid();
  DDCompareSolid(bool compmat);
  bool operator()(const DDSolid& lhs, const DDSolid& rhs) const ;
  bool compMatOnly_;
};

struct DDCompareDBLVEC : public std::binary_function<std::vector<double>, std::vector<double>, bool> {
  bool operator()(const std::vector<double>& lhs, const std::vector<double>& rhs) const ;
};

/// Needs to know because of Rotation Matrix of Boolean Relationship.
struct DDCompareBoolSol : public std::binary_function<DDBooleanSolid, DDBooleanSolid, bool> {
  DDCompareBoolSol();
  DDCompareBoolSol(bool compmat);
  bool operator()(const DDBooleanSolid& lhs, const DDBooleanSolid& rhs) const ;
  bool compMatOnly_;
};

struct DDCompareDDTrans : public std::binary_function<DDTranslation, DDTranslation, bool> {
  bool operator()(const DDTranslation& lhs, const DDTranslation& rhs) const;
};

/// Allows to compare name or not. If not, compares only values of the rotation matrix.
struct DDCompareDDRot : public std::binary_function<DDRotation, DDRotation, bool> {
  // special constructor if you want to compare the name.
  DDCompareDDRot();
  DDCompareDDRot( bool compmat );
  bool operator()(const DDRotation& lhs, const DDRotation& rhs) const;
  bool compMatOnly_;
};

struct DDCompareDDRotMat : public std::binary_function<DDRotationMatrix, DDRotationMatrix, bool> {
  bool operator()(const DDRotationMatrix& lhs, const DDRotationMatrix& rhs) const;
};




