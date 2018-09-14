#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H

#include "FWCore/ParameterSet/interface/types.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerAbstractConstruction.h"
#include <string>

class GeometricDet;

bool subDetByType(const GeometricDet* a, const GeometricDet* b);
bool phiSortNP(const GeometricDet* a, const GeometricDet* b); // NP** Phase2 BarrelEndcap
bool isLessZ(const GeometricDet* a, const GeometricDet* b);
bool isLessModZ(const GeometricDet* a, const GeometricDet* b);
double getPhi(const GeometricDet* a);
double getPhiModule(const GeometricDet* a);
double getPhiGluedModule(const GeometricDet* a);
double getPhiMirror(const GeometricDet* a);
double getPhiModuleMirror(const GeometricDet* a);
double getPhiGluedModuleMirror(const GeometricDet* a);
bool isLessRModule(const GeometricDet* a, const GeometricDet* b);
bool isLessR(const GeometricDet* a, const GeometricDet* b);

/**
 * Abstract Class to construct a Level in the hierarchy
 */

class CmsTrackerLevelBuilder : public CmsTrackerAbstractConstruction {
public:
    void build(DDFilteredView&, GeometricDet*, std::string) override;
    ~CmsTrackerLevelBuilder() override {}

private:
    virtual void buildComponent(DDFilteredView&, GeometricDet*, std::string) = 0;

protected:
    CmsTrackerStringToEnum theCmsTrackerStringToEnum;

private:
    virtual void sortNS(DDFilteredView&, GeometricDet*) {}
    CmsTrackerStringToEnum _CmsTrackerStringToEnum;
};

#endif
