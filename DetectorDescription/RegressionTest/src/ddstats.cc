#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/RegressionTest/src/ddstats.h"
#include "DataFormats/Math/interface/Graph.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

struct DDPosData;

void ddstats(std::ostream & os)
{

  os << "DDD in memory stats:" << std::endl 
     << "====================" << std::endl << std::endl;
  DDCompactView cpv;
  
  // What we will count:
  // ----------------------
  
  int noEdges(0); // number of graph-multi-edges
  int noNodes(0); // number of graph-nodes
  int noExpNodes(1); // number of expanded-nodes

  // number of Logical- and PosParts, Solids, Materials
  int noLog(0), noSol(0), noMat(0), noRot(0); 
  
  // accumulative number of name-characters (logparts,solids,rotation,materials)
  int noCLog(0), noCSol(0), noCMat(0), noCRot(0);
  
  int noSolidP(0); // accumulative number of solid-parameters
 
  // fetch the acyclic multigraph 
  const auto & g = cpv.graph();
  
  DDExpandedView exv(cpv);
  while (exv.next()) ++noExpNodes;

  // iterate over the adjacency-list
  auto it = g.begin();
  for(; it != g.end(); ++it) {
    ++noNodes;
    noEdges += it->size();
  } 
 
  typedef DDLogicalPart::StoreT::value_type lpst_type;
  lpst_type & lpst = DDLogicalPart::StoreT::instance();
  lpst_type::iterator lpstit = lpst.begin();
  for(; lpstit != lpst.end(); ++lpstit) {
    noCLog += lpstit->first.name().size();
    ++noLog;
  }
  
  typedef DDMaterial::StoreT::value_type mast_type;
  mast_type & mast = DDMaterial::StoreT::instance();
  mast_type::iterator mastit = mast.begin();
  for(; mastit != mast.end(); ++mastit) {
    noCMat += mastit->first.name().size();
    ++noMat;
  }

  typedef DDSolid::StoreT::value_type sost_type;
  sost_type & sost = DDSolid::StoreT::instance();
  sost_type::iterator sostit = sost.begin();
  for(; sostit != sost.end(); ++sostit) {
    noCSol += sostit->first.name().size();
    DDSolid s(sostit->first);
    noSolidP += s.parameters().size(); 
    ++noSol;
  }
  
  typedef DDRotation::StoreT::value_type rost_type;
  rost_type & rost = DDRotation::StoreT::instance();
  rost_type::iterator rostit = rost.begin();
  for(; rostit != rost.end(); ++rostit) {
    noCRot += rostit->first.name().size();
    ++noRot;
  }
 
  // derived quantities
  std::cout << "sizeof(void*)=" << sizeof(void*) << std::endl;
  std::cout << "sizeof(DDLogicalPart)="<<sizeof(DDLogicalPart)<<std::endl;
  std::cout << "sizeof(DDTranslation)="<< sizeof(DDTranslation)<<std::endl;
  std::cout << "sizeof(DDRotationMatrix)=" << sizeof(DDRotationMatrix)<<std::endl;
  int store = 4*sizeof(void*); // overhead for data-management (est.)
  int byRot = noRot * (sizeof(DDRotationMatrix) + store); // bytes in rotations
  int bySol = noSolidP * sizeof(double) + noSol*store; // bytes in solids
  int byMat = noMat * (5*sizeof(double) + store); // bytes in materials
  int byPos = noEdges * (sizeof(DDTranslation) + sizeof(DDRotation) + sizeof(int)); 
  int byNam = noCLog + noCSol + noCMat + noCRot; // bytes in strings for names
  int byLog = noLog * (sizeof(DDMaterial) + sizeof(DDSolid) + store); // LogicalPart
  int byGra = (noEdges + noNodes)*store; // est. graph structure
  int bytes = byRot + bySol + byMat + byPos + byNam + byLog + byGra;
  bytes += noNodes*sizeof(DDLogicalPart) + noEdges*sizeof(DDPosData*);
  double mb = 1024.*1024.;
  
  os << "noNodes=" << noNodes << std::endl
     << "noEdges=" << noEdges << std::endl 
     << "noExNod=" << noExpNodes << std::endl << std::endl;   
  os << "noLog=" << noLog << std::endl
     << "noSol=" << noSol << " noSolidP=" << noSolidP << std::endl
     << "noMat=" << noMat << std::endl
     << "noRot=" << noRot << std::endl << std::endl;
  os << "noCLog=" << noCLog << std::endl
     << "noCSol=" << noCSol << std::endl      
     << "noCMat=" << noCMat << std::endl
     << "noCRot=" << noCRot << std::endl
     << "       --------" << std::endl
     << "       " << byNam
     <<  " chars used for naming." << std::endl << std::endl;
  os << "byLog = " << byLog/mb << " logicalparts " << std::endl
     << "byNam = " << byNam/mb << " string for names " << std::endl
     << "byRot = " << byRot/mb << " rotations " << std::endl
     << "bySol = " << bySol/mb << " solids " << std::endl
     << "byMat = " << byMat/mb << " materials " << std::endl
     << "byPos = " << byPos/mb << " posparts " << std::endl
     << "byGra = " << byGra/mb << " graph-struct " << std::endl
     << "-----------------------" << std::endl 
     << "OVERALL: " << bytes / mb << " MByte" << std::endl;
  
}
