#include "CondDBCmsMTDConstruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDD.h"

using namespace cms;

std::unique_ptr<GeometricTimingDet> CondDBCmsMTDConstruction::construct(const PGeometricTimingDet& pgd) {
  auto mtd = std::make_unique<GeometricTimingDet>(pgd.pgeomdets_[0], GeometricTimingDet::MTD);

  size_t detMax = pgd.pgeomdets_.size();
  size_t tri = 1;
  std::vector<GeometricTimingDet*> hier;
  GeometricTimingDet* subdet = mtd.get();
  hier.emplace_back(subdet);
  while (tri < detMax && pgd.pgeomdets_[tri].level_ == 1) {
    subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
    ++tri;
    hier.back()->addComponent(subdet);
    hier.emplace_back(subdet);
    while (tri < detMax && pgd.pgeomdets_[tri].level_ == 2) {
      subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
      ++tri;
      hier.back()->addComponent(subdet);
      hier.emplace_back(subdet);
      while (tri < detMax && pgd.pgeomdets_[tri].level_ == 3) {
        subdet =
            new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
        ++tri;
        hier.back()->addComponent(subdet);
        hier.emplace_back(subdet);
        while (tri < detMax && pgd.pgeomdets_[tri].level_ == 4) {
          subdet =
              new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
          ++tri;
          hier.back()->addComponent(subdet);
          hier.emplace_back(subdet);
          while (tri < detMax && pgd.pgeomdets_[tri].level_ == 5) {
            subdet =
                new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
            ++tri;
            hier.back()->addComponent(subdet);
            hier.emplace_back(subdet);
            while (tri < detMax && pgd.pgeomdets_[tri].level_ == 6) {
              subdet = new GeometricTimingDet(pgd.pgeomdets_[tri],
                                              GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
              ++tri;
              hier.back()->addComponent(subdet);
            }
            hier.pop_back();
          }
          hier.pop_back();
        }
        hier.pop_back();
      }
      hier.pop_back();
    }
    hier.pop_back();
  }
  return mtd;
}
