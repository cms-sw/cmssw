#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"

#include <cmath>

bool CmsMTDLevelBuilder::subDetByType(const GeometricTimingDet* a, const GeometricTimingDet* b)
{
    // it relies on the fact that the GeometricTimingDet::GDEnumType
    // enumerators used to identify the subdetectors in the upgrade geometries
    // are equal to the ones of the present detector + n*100
    return a->type() < b->type();
}

// NP** Phase2 BarrelEndcap
bool CmsMTDLevelBuilder::phiSortNP(const GeometricTimingDet* a, const GeometricTimingDet* b)
{
    if (std::abs(a->translation().rho() - b->translation().rho()) < 0.01
        && (std::abs(a->translation().phi() - b->translation().phi()) < 0.01
               || std::abs(a->translation().phi() - b->translation().phi()) > 6.27)
        && a->translation().z() * b->translation().z() > 0.0) {
        return (std::abs(a->translation().z()) < std::abs(b->translation().z()));
    } else
        return false;
}

bool CmsMTDLevelBuilder::isLessZ(const GeometricTimingDet* a, const GeometricTimingDet* b)
{
    // NP** change for Phase 2 Tracker
    if (a->translation().z() == b->translation().z()) {
        return a->translation().rho() < b->translation().rho();
    } else {
        // Original version
        return a->translation().z() < b->translation().z();
    }
}

bool CmsMTDLevelBuilder::isLessModZ(const GeometricTimingDet* a, const GeometricTimingDet* b)
{
    return std::abs(a->translation().z()) < std::abs(b->translation().z());
}

double CmsMTDLevelBuilder::getPhi(const GeometricTimingDet* a)
{
    double phi = a->phi();
    return (phi >= 0 ? phi : phi + 2 * M_PI);
}

double CmsMTDLevelBuilder::getPhiModule(const GeometricTimingDet* a)
{
    std::vector<const GeometricTimingDet*> const& comp = a->components().back()->components();
    float phi = 0.;
    bool sum = true;

    for (auto i : comp) {
        if (std::abs(i->phi()) > M_PI / 2.) {
            sum = false;
            break;
        }
    }

    if (sum) {
        for (auto i : comp) {
            phi += i->phi();
        }

        double temp = phi / float(comp.size()) < 0. ? 2 * M_PI + phi / float(comp.size()) : phi / float(comp.size());
        return temp;

    } else {
        for (auto i : comp) {
            double phi1 = i->phi() >= 0 ? i->phi() : i->phi() + 2 * M_PI;
            phi += phi1;
        }

        double com = comp.front()->phi() >= 0 ? comp.front()->phi() : 2 * M_PI + comp.front()->phi();
        double temp
            = std::abs(phi / float(comp.size()) - com) > 2. ? M_PI - phi / float(comp.size()) : phi / float(comp.size());
        temp = temp >= 0 ? temp : 2 * M_PI + temp;
        return temp;
    }
}

double CmsMTDLevelBuilder::getPhiGluedModule(const GeometricTimingDet* a)
{
    std::vector<const GeometricTimingDet*> comp;
    a->deepComponents(comp);
    float phi = 0.;
    bool sum = true;

    for (auto& i : comp) {
        if (std::abs(i->phi()) > M_PI / 2.) {
            sum = false;
            break;
        }
    }

    if (sum) {
        for (auto& i : comp) {
            phi += i->phi();
        }

        double temp = phi / float(comp.size()) < 0. ? 2 * M_PI + phi / float(comp.size()) : phi / float(comp.size());
        return temp;

    } else {
        for (auto& i : comp) {
            double phi1 = i->phi() >= 0 ? i->phi() : i->translation().phi() + 2 * M_PI;
            phi += phi1;
        }

        double com = comp.front()->phi() >= 0 ? comp.front()->phi() : 2 * M_PI + comp.front()->phi();
        double temp
            = std::abs(phi / float(comp.size()) - com) > 2. ? M_PI - phi / float(comp.size()) : phi / float(comp.size());
        temp = temp >= 0 ? temp : 2 * M_PI + temp;
        return temp;
    }
}

double CmsMTDLevelBuilder::getPhiMirror(const GeometricTimingDet* a)
{
    double phi = a->phi();
    phi = (phi >= 0 ? phi : phi + 2 * M_PI); // (-pi,pi] --> [0,2pi)
    return ((M_PI - phi) >= 0 ? (M_PI - phi) : (M_PI - phi) + 2 * M_PI); // (-pi,pi] --> [0,2pi)
}

double CmsMTDLevelBuilder::getPhiModuleMirror(const GeometricTimingDet* a)
{
    double phi = getPhiModule(a); // [0,2pi)
    phi = (phi <= M_PI ? phi : phi - 2 * M_PI); // (-pi,pi]
    return (M_PI - phi);
}

double CmsMTDLevelBuilder::getPhiGluedModuleMirror(const GeometricTimingDet* a)
{
    double phi = getPhiGluedModule(a); // [0,2pi)
    phi = (phi <= M_PI ? phi : phi - 2 * M_PI); // (-pi,pi]
    return (M_PI - phi);
}

bool CmsMTDLevelBuilder::isLessRModule(const GeometricTimingDet* a, const GeometricTimingDet* b)
{
    return a->deepComponents().front()->rho() < b->deepComponents().front()->rho();
}

bool CmsMTDLevelBuilder::isLessR(const GeometricTimingDet* a, const GeometricTimingDet* b) { return a->rho() < b->rho(); }

void CmsMTDLevelBuilder::build(DDFilteredView& fv, GeometricTimingDet* tracker, std::string attribute)
{

    LogTrace("GeometricTimingDetBuilding")
        << std::string(3 * fv.history().size(), '-') << "+ " << ExtractStringFromDDD::getString(attribute, &fv) << " "
        << tracker->type() << " " << tracker->name() << std::endl;

    bool doLayers = fv.firstChild(); // descend to the next Layer

    while (doLayers) {
        buildComponent(fv, tracker, attribute);
        doLayers = fv.nextSibling(); // go to the next adjacent thingy
    }

    fv.parent();

    sortNS(fv, tracker);
}
