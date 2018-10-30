#ifndef Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H

#include "FWCore/ParameterSet/interface/types.h"
#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDAbstractConstruction.h"
#include <string>

class GeometricTimingDet;

/**
 * Abstract Class to construct a Level in the hierarchy
 */

class CmsMTDLevelBuilder : public CmsMTDAbstractConstruction {
public:
    // it relies on the fact that the GeometricTimingDet::GDEnumType enumerators
    // used to identify the subdetectors in the upgrade geometries are equal to the
    // ones of the present detector + n*100
    static bool subDetByType(const GeometricTimingDet* a, const GeometricTimingDet* b);
    // NP** Phase2 BarrelEndcap
    static bool phiSortNP(const GeometricTimingDet* a, const GeometricTimingDet* b);
    static bool isLessZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
    static bool isLessModZ(const GeometricTimingDet* a, const GeometricTimingDet* b);
    static double getPhi(const GeometricTimingDet* a);
    static double getPhiModule(const GeometricTimingDet* a);
    static double getPhiGluedModule(const GeometricTimingDet* a);
    static double getPhiMirror(const GeometricTimingDet* a);
    static double getPhiModuleMirror(const GeometricTimingDet* a);
    static double getPhiGluedModuleMirror(const GeometricTimingDet* a);
    static bool isLessRModule(const GeometricTimingDet* a, const GeometricTimingDet* b);
    static bool isLessR(const GeometricTimingDet* a, const GeometricTimingDet* b);

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
