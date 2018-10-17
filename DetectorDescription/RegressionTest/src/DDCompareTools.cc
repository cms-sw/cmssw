#include "DetectorDescription/RegressionTest/interface/DDCompareTools.h"

#include <cstddef>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Math/GenVector/Rotation3D.h"

bool DDCompareEPV( DDExpandedView& lhs, DDExpandedView& rhs, const DDCompOptions& ddco)
{
  bool ret(true);

  std::cout <<"*********FIRST BY firstChild, firstChild, nextSibling, nextSibling*********" << std::endl;
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.firstChild();
  rhs.firstChild();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.firstChild();
  rhs.firstChild();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.nextSibling();
  rhs.nextSibling();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.nextSibling();
  rhs.nextSibling();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.parent();
  rhs.parent();
  std::cout <<"*********THEN BY next, next, next, next*********" << std::endl;
  lhs.parent();
  rhs.parent();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.next();
  rhs.next();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  lhs.next();
  rhs.next();
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  std::cout << lhs.depth() << " depth " << rhs.depth() << std::endl;
  lhs.next();
  rhs.next();
  std::cout << lhs.depth() << " depth " << rhs.depth() << std::endl;
  std::cout << lhs.logicalPart().name().name() << ":" << lhs.copyno()
	    << " !=? " << rhs.logicalPart().name().name() << ":" << rhs.copyno()
	    << std::endl;
  return ret;
}

bool DDCompareCPV(const DDCompactView& lhs, const DDCompactView& rhs, const DDCompOptions& ddco)
{
  bool ret(true);

  const auto & g1 = lhs.graph();
  const auto & g2 = rhs.graph();

  using Graph = DDCompactView::Graph;
  using adjl_iterator = Graph::const_adj_iterator;

  adjl_iterator git1 = g1.begin();
  adjl_iterator gend1 = g1.end();
  adjl_iterator git2 = g2.begin();
  adjl_iterator gend2 = g2.end();

  Graph::index_type i=0;

  while ( git1 != gend1 && git2 != gend2 && ret ) {
    const DDLogicalPart & ddLP1 = g1.nodeData(git1);
    const DDLogicalPart & ddLP2 = g2.nodeData(git2);
    std::cout << ++i << " P " << ddLP1.name() << " " << ddLP2.name() << std::endl;
    if ( ! DDCompareLP(ddLP1, ddLP2, ddco) ) {
      ret = false;
      break;
    } else if (!git1->empty() && !git2->empty() ) { 
      auto cit1  = git1->begin();
      auto cend1 = git1->end();
      auto cit2  = git2->begin();
      auto cend2 = git2->end();

      while ( cit1 != cend1 && cit2 != cend2 ) {
	const DDLogicalPart & ddcurLP1 = g1.nodeData(cit1->first);
	const DDLogicalPart & ddcurLP2 = g2.nodeData(cit2->first);
	std::cout << ++i << " c1--> " << g1.edgeData(cit1->second)->copyno() << " " << ddcurLP1.name().fullname() << std::endl;
	std::cout << ++i << " c2--> " << g2.edgeData(cit2->second)->copyno() << " " << ddcurLP2.name().fullname() << std::endl;
	const DDPosData* p1(g1.edgeData(cit1->second));
	const DDPosData* p2(g2.edgeData(cit2->second));

	if ( p1->copyno() != p2->copyno() || 
	     ! DDCompareLP(ddcurLP1,ddcurLP2,ddco) ) {
	  std::cout << "Failed to match node (fullname:copy_no): 1: " 
		    << ddcurLP1.name().fullname() << ":" << p1->copyno() << " 2: " 
		    << ddcurLP2.name().fullname() << ":" << p2->copyno() << std::endl;
	  ret = false;
	  break;
	} else if ( ! DDCompareDDTrans(p1->trans(), p2->trans()) ) {
	  std::cout << "Failed to match translation " << std::endl;
	  ret = false;
	  break;
	} else if ( ! DDCompareDDRot(p1->ddrot(), p2->ddrot(), ddco) ) {
	  std::cout << "Failed to match rotation " << std::endl;
	  ret = false;
	  break;
	}
	++cit1;
	++cit2;
      }
    } else if ( git1->size() != git2->size() ) {
      ret = false;
      std::cout << "DDCompactViews are different because number of children do not match" << std::endl;
      std::cout << "graph1 size of edge_list: " << git1->size() << " and graph2 size of edge_list: " << git2->size() << std::endl;
      break;
    }
    ++git1;
    ++git2;
  }
  return ret;
}

bool DDCompareLP(const DDLogicalPart& lhs, const DDLogicalPart& rhs, const DDCompOptions& ddco)
{
  bool ret(true);
  // for a logical part to be equal, the solid must be equal and the name must be equal.
  if ( lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "LogicalPart names do not match " << lhs.name().fullname() 
	      << " and " << rhs.name().fullname() << std::endl;
  } else if ( ! DDCompareSolid(lhs.solid(), rhs.solid(), ddco) ){
    ret = false;
    std::cout << "LogicalPart Solids do not match " << lhs.name().fullname() 
	      << " and " << rhs.name().fullname() << std::endl;
  }
  return ret;
}

bool DDCompareSolid(const DDSolid& lhs, const DDSolid& rhs, const DDCompOptions& ddco)
{
  bool ret(true);
    switch ( lhs.shape() ) {
    case DDSolidShape::dd_not_init:
    case DDSolidShape::ddbox:
    case DDSolidShape::ddtubs:
    case DDSolidShape::ddcuttubs:
    case DDSolidShape::ddtrap: 
    case DDSolidShape::ddcons:
    case DDSolidShape::ddpolycone_rz:
    case DDSolidShape::ddpolyhedra_rz:
    case DDSolidShape::ddpolycone_rrz:
    case DDSolidShape::ddpolyhedra_rrz:
    case DDSolidShape::ddextrudedpolygon:
    case DDSolidShape::ddtorus:
    case DDSolidShape::ddpseudotrap:
    case DDSolidShape::ddtrunctubs:
    case DDSolidShape::ddsphere:
    case DDSolidShape::ddellipticaltube:
    case DDSolidShape::ddshapeless: 
      {
	if ( lhs.name().fullname() != rhs.name().fullname() ) {
	  ret = false;
	  std::cout << "Solid names do not match for solid " << lhs.name().fullname() << " and " << rhs.name().fullname() << std::endl;
	} else if ( lhs.shape() != rhs.shape() ) {
	  ret = false;
	  std::cout << "Shape types do not match for solids " << lhs.name().fullname() 
		    << " and " << rhs.name().fullname() 
		    << " even though their names match " << std::endl;
	} else if ( ! DDCompareDBLVEC(lhs.parameters(), rhs.parameters()) ) {
	  ret = false;
	  std::cout << "Parameters do not match for solids " << lhs.name().fullname() 
		    << " and " << rhs.name().fullname() 
		    << " even though their names and shape type match." << std::endl;
	  std::cout << "size: " << lhs.parameters().size() << " " << rhs.parameters().size() << std::endl;
	} 
	break;
      }
    case DDSolidShape::ddunion:
    case DDSolidShape::ddsubtraction:
    case DDSolidShape::ddintersection: 
      {
	if ( ! DDCompareBoolSol(lhs, rhs, ddco) ) {
	  ret = false;
	}
	break;
      }
    default:
      break;
    }
    return ret;
}

// Default tolerance was 0.0004
bool DDCompareDBLVEC( const std::vector<double>& lhs, const std::vector<double>& rhs, double tol)
{
  bool ret(true);
  std::ios_base::fmtflags originalFlags = std::cout.flags();
  int originalPrecision = std::cout.precision();
  if( lhs.size() != rhs.size())
  {
    ret = false;
    std::cout << "Size of vectors do not match." << std::endl;
  }
  else
  {
    for( size_t i = 0; i < lhs.size() ; ++i )
    {
      if( std::fabs( lhs[i] - rhs[i] ) > tol )
      {
	ret = false;
	std::cout << "Vector content at index " << i << " does not match " ;
	std::cout << std::setw(12) << std::fixed << std::setprecision(4) << lhs[i] << " != " << rhs[i] << std::endl;
	break;
      }
    }
  }
  // Now set everything back to defaults
  std::cout.flags( originalFlags );
  std::cout.precision( originalPrecision );
  return ret;
}

bool DDCompareBoolSol( const DDBooleanSolid& lhs, const DDBooleanSolid& rhs, const DDCompOptions& ddco)
{
  bool ret(true);
  if ( lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "BooleanSolid names do not match ";
  } else if ( lhs.shape() != rhs.shape() ) {
    ret = false;
    std::cout << "BooleanSolid shape types do not match ";
  } else if ( ! DDCompareDBLVEC(lhs.parameters(), rhs.parameters(), ddco.distTol_) ) {
    ret = false;
    std::cout << "BooleanSolid parameters do not match ";
  } else if ( ! DDCompareSolid(lhs.solidA(), rhs.solidA(), ddco) ) {
    ret = false;
    std::cout << "BooleanSolid SolidA solids do not match ";
  } else if ( ! DDCompareSolid(lhs.solidB(), rhs.solidB(), ddco) ) {
    ret= false;
    std::cout << "BooleanSolid SolidB solids do not match ";
  } else if ( ! DDCompareDDTrans(lhs.translation(), rhs.translation(), ddco.distTol_) ) {
    ret = false;
    std::cout << "BooleanSolid Translations do not match ";
  } else if ( ! DDCompareDDRot(lhs.rotation(), rhs.rotation(), ddco) ) {
    ret = false;
    std::cout << "BooleanSolid Rotations do not match ";
  }
  if ( ! ret ) {
    std::cout << "for boolean solids " 
	      << lhs.name().fullname() << " and " 
	      << rhs.name().fullname() << std::endl;
  }
  return ret;
}

// Default tolerance was 0.0004
bool DDCompareDDTrans( const DDTranslation& lhs, const DDTranslation& rhs, double tol)
{
  bool ret(true);
  if ( std::fabs(lhs.x() - rhs.x()) > tol
       || std::fabs(lhs.y() - rhs.y()) > tol
       || std::fabs(lhs.z() - rhs.z()) > tol ) {  
    ret=false;
  }
  return ret;
}

bool DDCompareDDRot( const DDRotation& lhs, const DDRotation& rhs, const DDCompOptions& ddco)
{
  bool ret(true);
  if ( ddco.compRotName_ && lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "DDRotation names do not match " 
	      << lhs.name().fullname() << " and " 
	      << rhs.name().fullname() << std::endl;
  } else if ( ! DDCompareDDRotMat( lhs.rotation(), rhs.rotation()) ) {
    ret = false;
    std::cout << "DDRotationMatrix values do not match " 
	      << lhs.name().fullname() << " and " 
	      << rhs.name().fullname() << std::endl;
  }
  return ret;
}

// Default tolerance was 0.0004
bool DDCompareDDRotMat( const DDRotationMatrix& lhs, const DDRotationMatrix& rhs, double tol )
{
  bool ret(true);
  // manual way to do it... postponed.  Tested with Distance method from root::math
  //DD3Vector x1, y1, z1;
  //lhs.GetComponents(x1,y1,z1);
  //DD3Vector x2, y2, z2;
  //rhs.GetComponents(x2,y2,z2);
  double dist = Distance(lhs,rhs);
  if ( std::fabs(dist) > tol ) {
    std::cout << "Rotation matrices do not match." << std::endl;
    ret = false;
    DD3Vector x, y, z;
    std::cout << "FIRST" << std::endl;
    lhs.GetComponents(x,y,z);
    std::cout << std::setw(12) << std::fixed << std::setprecision(4) << x.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Z();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Z();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Z() << std::endl;
    std::cout << "SECOND" << std::endl;
    rhs.GetComponents(x,y,z);
    std::cout << std::setw(12) << std::fixed << std::setprecision(4) << x.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.X();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Y();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Z();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Z();
    std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Z() << std::endl;
  }

  return ret;
}
