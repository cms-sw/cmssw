#include "CondDBCmsMTDConstruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDDetIdBuilder.h"

using namespace cms;

CondDBCmsMTDConstruction::CondDBCmsMTDConstruction() { }

const GeometricTimingDet* CondDBCmsMTDConstruction::construct(const PGeometricTimingDet& pgd) {
  GeometricTimingDet* mtd  = new GeometricTimingDet(pgd.pgeomdets_[0],GeometricTimingDet::MTD);

  size_t detMax =  pgd.pgeomdets_.size();
  size_t tri = 1;
  std::vector<GeometricTimingDet*> hier;
  int lev=1;
  GeometricTimingDet* subdet = mtd;
  hier.emplace_back(subdet);
    while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 1 ) {
      subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
      ++tri;
      hier.back()->addComponent(subdet);
      hier.emplace_back(subdet);
      ++lev;
      while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 2 ) {
	subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
	++tri;
	hier.back()->addComponent(subdet);
	hier.emplace_back(subdet);
	++lev;
	while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 3 ) {
	  subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
	  ++tri;
	  hier.back()->addComponent(subdet);
	  hier.emplace_back(subdet);
	  ++lev;
	  while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 4 ) {
	    subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
	    ++tri;
	    hier.back()->addComponent(subdet);
	    hier.emplace_back(subdet);
	    ++lev;
	    while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 5 ) {
	      subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
	      ++tri;
	      hier.back()->addComponent(subdet);
	      hier.emplace_back(subdet);
	      ++lev;
	      while ( tri < detMax && pgd.pgeomdets_[tri].level_ == 6 ) {
		subdet = new GeometricTimingDet(pgd.pgeomdets_[tri], GeometricTimingDet::GTDEnumType(pgd.pgeomdets_[tri].type_));
		++tri;
		hier.back()->addComponent(subdet);
	      }
	      --lev;
	      hier.pop_back();
	    }
	    --lev;
	    hier.pop_back();
	  }
	  --lev;
	  hier.pop_back();
	}
	--lev;
	hier.pop_back();
      }
      --lev;
      hier.pop_back();
    }
  return mtd;
}

