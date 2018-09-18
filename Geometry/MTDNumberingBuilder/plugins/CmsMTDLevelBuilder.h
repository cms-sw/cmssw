#ifndef Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H

#include "FWCore/ParameterSet/interface/types.h"
#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDAbstractConstruction.h"
#include <string>

class GeometricTimingDet;

// it relies on the fact that the GeometricTimingDet::GDEnumType enumerators
// used to identify the subdetectors in the upgrade geometries are equal to the
// ones of the present detector + n*100
bool subDetByType(const GeometricTimingDet* a, const GeometricTimingDet* b);
// NP** Phase2 BarrelEndcap
bool phiSortNP(const GeometricTimingDet* a, const GeometricTimingDet* b);
bool isLessZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
bool isLessModZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
double getPhi(const GeometricTimingDet* a);
double getPhiModule(const GeometricTimingDet* a);
double getPhiGluedModule(const GeometricTimingDet* a);
double getPhiMirror(const GeometricTimingDet* a);
double getPhiModuleMirror(const GeometricTimingDet* a);
double getPhiGluedModuleMirror(const GeometricTimingDet* a);
bool isLessRModule(const GeometricTimingDet* a, const GeometricTimingDet* b);
bool isLessR(const GeometricTimingDet* a, const GeometricTimingDet* b);

/**
 * Abstract Class to construct a Level in the hierarchy
 */

class CmsMTDLevelBuilder : public CmsMTDAbstractConstruction {
public:
    void build(DDFilteredView&, GeometricTimingDet*, std::string) override;
    ~CmsMTDLevelBuilder() override {}

private:
    virtual void buildComponent(DDFilteredView&, GeometricTimingDet*, std::string) = 0;

protected:
    CmsMTDStringToEnum theCmsMTDStringToEnum;

private:
    virtual void sortNS(DDFilteredView&, GeometricTimingDet*) {}
    CmsMTDStringToEnum _CmsMTDStringToEnum;
};

#endif
