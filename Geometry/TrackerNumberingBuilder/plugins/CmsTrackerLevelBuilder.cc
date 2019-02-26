#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"

#include <cmath>

bool CmsTrackerLevelBuilder::subDetByType(const GeometricDet* a, const GeometricDet* b)
{
    // it relies on the fact that the GeometricDet::GDEnumType enumerators used
    // to identify the subdetectors in the upgrade geometries are equal to the
    // ones of the present detector + n*100
    return a->type() % 100 < b->type() % 100;
}

// NP** Phase2 BarrelEndcap
bool CmsTrackerLevelBuilder::phiSortNP(const GeometricDet* a, const GeometricDet* b)
{
    if (std::abs(a->translation().rho() - b->translation().rho()) < 0.01
        && (std::abs(a->translation().phi() - b->translation().phi()) < 0.01
               || std::abs(a->translation().phi() - b->translation().phi()) > 6.27)
        && a->translation().z() * b->translation().z() > 0.0) {
        return (std::abs(a->translation().z()) < std::abs(b->translation().z()));
    } else
        return false;
}

bool CmsTrackerLevelBuilder::isLessZ(const GeometricDet* a, const GeometricDet* b)
{
    // NP** change for Phase 2 Tracker
    if (a->translation().z() == b->translation().z()) {
        return a->translation().rho() < b->translation().rho();
    } else {
        // Original version
        return a->translation().z() < b->translation().z();
    }
}

bool CmsTrackerLevelBuilder::isLessModZ(const GeometricDet* a, const GeometricDet* b)
{
    return std::abs(a->translation().z()) < std::abs(b->translation().z());
}

double CmsTrackerLevelBuilder::getPhi(const GeometricDet* a)
{
    double phi = a->phi();
    return (phi >= 0 ? phi : phi + 2 * M_PI);
}

double CmsTrackerLevelBuilder::getPhiModule(const GeometricDet* a)
{
    std::vector<const GeometricDet*> const& comp = a->components().back()->components();
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

double CmsTrackerLevelBuilder::getPhiGluedModule(const GeometricDet* a)
{
    std::vector<const GeometricDet*> comp;
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

double CmsTrackerLevelBuilder::getPhiMirror(const GeometricDet* a)
{
    double phi = a->phi();
    phi = (phi >= 0 ? phi : phi + 2 * M_PI); // (-pi,pi] --> [0,2pi)
    return ((M_PI - phi) >= 0 ? (M_PI - phi) : (M_PI - phi) + 2 * M_PI); // (-pi,pi] --> [0,2pi)
}

double CmsTrackerLevelBuilder::getPhiModuleMirror(const GeometricDet* a)
{
    double phi = getPhiModule(a); // [0,2pi)
    phi = (phi <= M_PI ? phi : phi - 2 * M_PI); // (-pi,pi]
    return (M_PI - phi);
}

double CmsTrackerLevelBuilder::getPhiGluedModuleMirror(const GeometricDet* a)
{
    double phi = getPhiGluedModule(a); // [0,2pi)
    phi = (phi <= M_PI ? phi : phi - 2 * M_PI); // (-pi,pi]
    return (M_PI - phi);
}

bool CmsTrackerLevelBuilder::isLessRModule(const GeometricDet* a, const GeometricDet* b)
{
    return a->deepComponents().front()->rho() < b->deepComponents().front()->rho();
}

bool CmsTrackerLevelBuilder::isLessR(const GeometricDet* a, const GeometricDet* b)
{
    return a->rho() < b->rho();
}

void CmsTrackerLevelBuilder::build(DDFilteredView& fv, GeometricDet* tracker, std::string attribute)
{

    LogTrace("GeometricDetBuilding") << std::string(3 * fv.history().size(), '-') << "+ "
                                     << ExtractStringFromDDD::getString(attribute, &fv) << " " << tracker->type() << " "
                                     << tracker->name() << std::endl;

    bool doLayers = fv.firstChild(); // descend to the first Layer

    while (doLayers) {
        buildComponent(fv, tracker, attribute);
        doLayers = fv.nextSibling(); // go to next layer
    }

    fv.parent();

    sortNS(fv, tracker);
}
