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

DDCompareEPV::DDCompareEPV() : ddco_() { }

DDCompareEPV::DDCompareEPV(const DDCompOptions& ddco) : ddco_(ddco) { }

bool DDCompareEPV::operator()(DDExpandedView& lhs, DDExpandedView& rhs) const {
  bool ret(true);
  /* struct DDCompareCPVGraph : public std::binary_function<DDCompactView::graph_type, DDCompactView::graph_type, bool> { */
  /*   bool operator()(const DDCompactView::graph_type& lhs, const DDCompactView::graph_type& rhs) const { */
  /*     bool ret; */
  //    const graph<DDLogicalPart, DDPosData*>& g1= lhs;
  //    graph<DDLogicalPart, DDPosData*>::const_iterator beg1 = g1.begin();
  //DDCompactView::graph_type::const_iterator beg1 = lhs.begin();
  /*     std::cout << "size of lhs = " << lhs.edge_size() << std::endl;     */
  /*     std::cout << "size of rhs = " << rhs.edge_size() << std::endl;     */
  /*     std::cout << "iterating over lhs edges?" << std::endl; */
  /*     graph<DDLogicalPart, DDPosData*>::const_iterator beg1(lhs), end1(lhs); */
  /*     beg1 = lhs.begin_iter(); */
  /*     end1 = lhs.end_iter(); */
  /*     size_t countem(0); */
  /*     for ( ; beg1 != end1 ; ++beg1 ) { */
  /*       ++countem; */
  /*     } */
  /*     std::cout << "between lhs.begin_iter() and lhs.end_iter() there are " << countem << std::endl; */
  /*     graph<DDLogicalPart,DDPosData*>::const_edge_range cer = lhs.graph().edges(lhs.root()); */
  /*     graph<DDLogicalPart,DDPosData*>::adj_iterator i = const_edge_range.first */
  //    DDExpandedView v1(lhs), v2(rhs);
  //    if (v1.next()) std::cout << "v1 is true" << std::endl;
  //    if (v2.next()) std::cout << "v2 is true" << std::endl;
  //    DDNodeComparator ddnc;
  //if ( ddnc (v1, v2) ) { 
  //  ret = true;
  //}
  /*     while (v1.next()) { */
  /*       if (v2.next()) { */
  /* 	std::cout << v1.logicalPart().name().name() << ":" << v1.copyno() */
  /* 		  << " !=? " << v2.logicalPart().name().name() << ":" << v2.copyno() */
  /* 		  << std::endl; */
	
  /* 	if ( v1.logicalPart().name().name() != v2.logicalPart().name().name() ) { */
  /* 	  std::cout << v1.logicalPart().name().name() << ":" << v1.copyno() */
  /* 		    << " !=? " << v2.logicalPart().name().name() << ":" << v2.copyno() */
  /* 		    << std::endl; */
  /* 	//	  std::cout << v1.logicalPart().name().name() << " !=? " << v2.logicalPart().name().name() << std::endl; */
  /* 	  ret=false; */
  /* 	  break; */
  /* 	} */
  /*       } */
  /*     } */
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

DDCompareCPV::DDCompareCPV() : ddco_() { }

DDCompareCPV::DDCompareCPV(const DDCompOptions& ddco) : ddco_(ddco) { }

bool DDCompareCPV::operator()(const DDCompactView& lhs, const DDCompactView& rhs) const {
  bool ret(true);

  const DDCompactView::graph_type & g1 = lhs.graph();
  const DDCompactView::graph_type & g2 = rhs.graph();

  typedef DDCompactView::graph_type::const_adj_iterator adjl_iterator;
  adjl_iterator git1 = g1.begin();
  adjl_iterator gend1 = g1.end();
  adjl_iterator git2 = g2.begin();
  adjl_iterator gend2 = g2.end();
  /*     std::cout << "uniqueness test: " << &(*git1) << " != " << &(*git2)  */
  /* 	      << " and " << &(*gend1) << " != " << &(*gend2) << std::endl; */
  //    DDCompactView::graph_type::const_iterator bit = g1.begin_iter();
  DDCompactView::graph_type::index_type i=0;
  //    for (; git1 != gend1; ++git1) {
  while ( git1 != gend1 && git2 != gend2 && ret ) {
    const DDLogicalPart & ddLP1 = g1.nodeData(git1);
    const DDLogicalPart & ddLP2 = g2.nodeData(git2);
    std::cout << ++i << " P " << ddLP1.name() << " " << ddLP2.name() << std::endl;
    if ( ! DDCompareLP(ddco_)(ddLP1, ddLP2) ) {
      ret = false;
      break;
    } else if (!git1->empty() && !git2->empty() ) { 
      DDCompactView::graph_type::edge_list::const_iterator cit1  = git1->begin();
      DDCompactView::graph_type::edge_list::const_iterator cend1 = git1->end();
      DDCompactView::graph_type::edge_list::const_iterator cit2  = git2->begin();
      DDCompactView::graph_type::edge_list::const_iterator cend2 = git2->end();
      //for (; cit != cend; ++cit) {
      while ( cit1 != cend1 && cit2 != cend2 ) {
	const DDLogicalPart & ddcurLP1 = g1.nodeData(cit1->first);
	const DDLogicalPart & ddcurLP2 = g2.nodeData(cit2->first);
	std::cout << ++i << " c1--> " << g1.edgeData(cit1->second)->copyno() << " " << ddcurLP1.name().fullname() << std::endl;
	std::cout << ++i << " c2--> " << g2.edgeData(cit2->second)->copyno() << " " << ddcurLP2.name().fullname() << std::endl;
	const DDPosData* p1(g1.edgeData(cit1->second));
	const DDPosData* p2(g2.edgeData(cit2->second));
	//	  if ( g1.edgeData(cit1->second)->copyno_ != g2.edgeData(cit2->second)->copyno_
// 	if ( p1->copyno_ != p2->copyno_
// 	     || ddcurLP1.name().fullname() != ddcurLP2.name().fullname() ) {
// 	  std::cout << "Failed to match node (fullname:copy_no): 1: " 
// 		    << ddcurLP1.name().fullname() << ":" << p1->copyno_ << " 2: " 
// 		    << ddcurLP2.name().fullname() << ":" << p2->copyno_ << std::endl;
// 	  ret = false;
// 	  break;
	if ( p1->copyno() != p2->copyno() || 
	     ! DDCompareLP(ddco_)(ddcurLP1,ddcurLP2) ) {
	  std::cout << "Failed to match node (fullname:copy_no): 1: " 
		    << ddcurLP1.name().fullname() << ":" << p1->copyno() << " 2: " 
		    << ddcurLP2.name().fullname() << ":" << p2->copyno() << std::endl;
	  ret = false;
	  break;
	} else if ( ! DDCompareDDTrans()(p1->trans(), p2->trans()) ) {
	  std::cout << "Failed to match translation " << std::endl;
// 	  std::cout << "1: " << std::setw(12) << std::fixed << std::setprecision(4) << p1->trans_.x();
// 	  std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << p1->trans_.y();
// 	  std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << p1->trans_.z() << std::endl;
// 	  std::cout << "2: " << std::setw(12) << std::fixed << std::setprecision(4) << p2->trans_.x();
// 	  std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << p2->trans_.y();
// 	  std::cout << "," << std::setw(12) << std::fixed << std::setprecision(4) << p2->trans_.z() << std::endl;
	  ret = false;
	  break;
	} else if ( ! DDCompareDDRot(ddco_)(p1->ddrot(), p2->ddrot()) ) {
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

DDCompareLP::DDCompareLP() : ddco_() { }

DDCompareLP::DDCompareLP(const DDCompOptions& ddco) : ddco_(ddco) { }

bool DDCompareLP::operator()(const DDLogicalPart& lhs, const DDLogicalPart& rhs) const {
  bool ret(true);
  // for a logical part to be equal, the solid must be equal and the name must be equal.
  if ( lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "LogicalPart names do not match " << lhs.name().fullname() 
	      << " and " << rhs.name().fullname() << std::endl;
  } else if ( ! DDCompareSolid(ddco_)(lhs.solid(), rhs.solid()) ){
    ret = false;
    std::cout << "LogicalPart Solids do not match " << lhs.name().fullname() 
	      << " and " << rhs.name().fullname() << std::endl;
  }
  return ret;
}

DDCompareSolid::DDCompareSolid() : ddco_() { }

DDCompareSolid::DDCompareSolid(const DDCompOptions& ddco) : ddco_(ddco) { }

bool DDCompareSolid::operator()(const DDSolid& lhs, const DDSolid& rhs) const {
  bool ret(true);
    switch ( lhs.shape() ) {
    case dd_not_init:
    case ddbox:
    case ddtubs:
    case ddcuttubs:
    case ddtrap: 
    case ddcons:
    case ddpolycone_rz:
    case ddpolyhedra_rz:
    case ddpolycone_rrz:
    case ddpolyhedra_rrz:
    case ddextrudedpolygon:
    case ddtorus:
    case ddpseudotrap:
    case ddtrunctubs:
    case ddsphere:
    case ddorb:
    case ddellipticaltube:
    case ddellipsoid:
    case ddparallelepiped:
    case ddshapeless: 
      {
	if ( lhs.name().fullname() != rhs.name().fullname() ) {
	  ret = false;
	  std::cout << "Solid names do not match for solid " << lhs.name().fullname() << " and " << rhs.name().fullname() << std::endl;
	} else if ( lhs.shape() != rhs.shape() ) {
	  ret = false;
	  std::cout << "Shape types do not match for solids " << lhs.name().fullname() 
		    << " and " << rhs.name().fullname() 
		    << " even though their names match " << std::endl;
	} else if ( ! DDCompareDBLVEC()(lhs.parameters(), rhs.parameters()) ) {
	  ret = false;
	  std::cout << "Parameters do not match for solids " << lhs.name().fullname() 
		    << " and " << rhs.name().fullname() 
		    << " even though their names and shape type match." << std::endl;
	  std::cout << "size: " << lhs.parameters().size() << " " << rhs.parameters().size() << std::endl;
	} 
	break;
      }
    case ddunion:
    case ddsubtraction:
    case ddintersection: 
      {
	if ( ! DDCompareBoolSol(ddco_)(lhs, rhs) ) {
	  ret = false;
	}
	break;
      }
    case ddreflected:
      {
	DDReflectionSolid rs1(lhs);
	DDReflectionSolid rs2(rhs);
	if ( ! DDCompareSolid(ddco_)( rs1.unreflected(), rs2.unreflected()) ) {
	  ret = false;
	  std::cout << "Unreflected volumes of DDReflections do not match. Reflections are " 
		    << lhs.name().fullname() << " and " << rhs.name().fullname() << std::endl;
	}
	break;
      }
    default:
      break;
    }
    return ret;
}

DDCompareDBLVEC::DDCompareDBLVEC() : tol_(0.0004) { }

DDCompareDBLVEC::DDCompareDBLVEC(double tol) : tol_(tol) { }

bool
DDCompareDBLVEC::operator() ( const std::vector<double>& lhs, const std::vector<double>& rhs ) const
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
      if( std::fabs( lhs[i] - rhs[i] ) > tol_ )
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

DDCompareBoolSol::DDCompareBoolSol() : ddco_() { }

DDCompareBoolSol::DDCompareBoolSol(const DDCompOptions& ddco) : ddco_(ddco) { }
bool DDCompareBoolSol::operator() ( const DDBooleanSolid& lhs, const DDBooleanSolid& rhs ) const {
  bool ret(true);
  if ( lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "BooleanSolid names do not match ";
  } else if ( lhs.shape() != rhs.shape() ) {
    ret = false;
    std::cout << "BooleanSolid shape types do not match ";
  } else if ( ! DDCompareDBLVEC(ddco_.distTol_)(lhs.parameters(), rhs.parameters()) ) {
    ret = false;
    std::cout << "BooleanSolid parameters do not match ";
  } else if ( ! DDCompareSolid(ddco_)(lhs.solidA(), rhs.solidA()) ) {
    ret = false;
    std::cout << "BooleanSolid SolidA solids do not match ";
  } else if ( ! DDCompareSolid(ddco_)(lhs.solidB(), rhs.solidB()) ) {
    ret= false;
    std::cout << "BooleanSolid SolidB solids do not match ";
  } else if ( ! DDCompareDDTrans(ddco_.distTol_)(lhs.translation(), rhs.translation()) ) {
    ret = false;
    std::cout << "BooleanSolid Translations do not match ";
  } else if ( ! DDCompareDDRot(ddco_)(lhs.rotation(), rhs.rotation()) ) {
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

DDCompareDDTrans::DDCompareDDTrans() : tol_(0.0004) { }

DDCompareDDTrans::DDCompareDDTrans(double tol) : tol_(tol) { }

bool DDCompareDDTrans::operator() ( const DDTranslation& lhs, const DDTranslation& rhs ) const {
  bool ret(true);
  if ( std::fabs(lhs.x() - rhs.x()) > tol_
       || std::fabs(lhs.y() - rhs.y()) > tol_
       || std::fabs(lhs.z() - rhs.z()) > tol_ ) {  
    ret=false;
  }
  return ret;
}

DDCompareDDRot::DDCompareDDRot( ) : ddco_() { }

DDCompareDDRot::DDCompareDDRot( const DDCompOptions& ddco ) : ddco_(ddco) { }

bool DDCompareDDRot::operator() ( const DDRotation& lhs, const DDRotation& rhs ) const {
  bool ret(true);
  if ( ddco_.compRotName_ && lhs.name().fullname() != rhs.name().fullname() ) {
    ret = false;
    std::cout << "DDRotation names do not match " 
	      << lhs.name().fullname() << " and " 
	      << rhs.name().fullname() << std::endl;
  } else if ( ! DDCompareDDRotMat()(*lhs.rotation(), *rhs.rotation()) ) {
    ret = false;
    std::cout << "DDRotationMatrix values do not match " 
	      << lhs.name().fullname() << " and " 
	      << rhs.name().fullname() << std::endl;
  }
  return ret;
}

DDCompareDDRotMat::DDCompareDDRotMat() : tol_(0.0004) { }

DDCompareDDRotMat::DDCompareDDRotMat(double tol) : tol_(tol) { }

bool DDCompareDDRotMat::operator() ( const DDRotationMatrix& lhs, const DDRotationMatrix& rhs ) const {
  bool ret(true);
  // manual way to do it... postponed.  Tested with Distance method from root::math
  //DD3Vector x1, y1, z1;
  //lhs.GetComponents(x1,y1,z1);
  //DD3Vector x2, y2, z2;
  //rhs.GetComponents(x2,y2,z2);
  double dist = Distance(lhs,rhs);
  if ( std::fabs(dist) > tol_ ) {
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


