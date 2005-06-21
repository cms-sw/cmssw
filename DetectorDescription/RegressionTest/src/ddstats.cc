#include "DetectorDescription/DDRegressionTest/src/ddstats.h"
#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "DetectorDescription/DDCore/interface/DDMaterial.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"
#include "DetectorDescription/DDCore/interface/DDTransform.h"
#include "DetectorDescription/DDCore/interface/DDExpandedView.h"
void ddstats(ostream & os)
{

  os << "DDD in memory stats:" << endl 
     << "====================" << endl << endl;
  DDCompactView cpv;
  
  // What we will count:
  // ----------------------
  
  int noEdges(0); // number of graph-multi-edges
  int noNodes(0); // number of graph-nodes
  int noExpNodes(1); // number of expanded-nodes

  // number of Logical- and PosParts, Solids, Materials
  int noLog(0), noPos(0), noSol(0), noMat(0), noRot(0); 
  noPos = noEdges;
  
  // accumulative number of name-characters (logparts,solids,rotation,materials)
  int noCLog(0), noCSol(0), noCMat(0), noCRot(0);
  
  int noSolidP(0); // accumulative number of solid-parameters
 
  // fetch the acyclic multigraph 
  const graph_type & g = cpv.graph();
  
  DDExpandedView exv(cpv);
  while (exv.next()) noExpNodes++;

  // iterate over the adjacency-list
  graph_type::const_adj_iterator it = g.begin();
  for ( ; it != g.end(); ++it) {
    noNodes++;
    noEdges += it->size();
  } 
 
  typedef DDLogicalPart::StoreT::value_type lpst_type;
  lpst_type & lpst = DDLogicalPart::StoreT::instance();
  lpst_type::iterator lpstit = lpst.begin();
  for(; lpstit != lpst.end(); ++lpstit) {
    noCLog += lpstit->first.name().size();
    noLog++;
  }
  
  typedef DDMaterial::StoreT::value_type mast_type;
  mast_type & mast = DDMaterial::StoreT::instance();
  mast_type::iterator mastit = mast.begin();
  for(; mastit != mast.end(); ++mastit) {
    noCMat += mastit->first.name().size();
    noMat++;
  }

  typedef DDSolid::StoreT::value_type sost_type;
  sost_type & sost = DDSolid::StoreT::instance();
  sost_type::iterator sostit = sost.begin();
  for(; sostit != sost.end(); ++sostit) {
    noCSol += sostit->first.name().size();
    DDSolid s(sostit->first);
    noSolidP += s.parameters().size(); 
    noSol++;
  }
  
  typedef DDRotation::StoreT::value_type rost_type;
  rost_type & rost = DDRotation::StoreT::instance();
  rost_type::iterator rostit = rost.begin();
  for(; rostit != rost.end(); ++rostit) {
    noCRot += rostit->first.name().size();
    noRot++;
  }
 
  // derived quantities
  cout << "sizeof(void*)=" << sizeof(void*) << endl;
  cout << "sizeof(DDLogicalPart)="<<sizeof(DDLogicalPart)<<endl;
  cout << "sizeof(DDTranslation)="<< sizeof(DDTranslation)<<endl;
  cout << "sizeof(DDRotationMatrix)=" << sizeof(DDRotationMatrix)<<endl;
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
  
  os << "noNodes=" << noNodes << endl
     << "noEdges=" << noEdges << endl 
     << "noExNod=" << noExpNodes << endl << endl;   
  os << "noLog=" << noLog << endl
     << "noSol=" << noSol << " noSolidP=" << noSolidP << endl
     << "noMat=" << noMat << endl
     << "noRot=" << noRot << endl << endl;
  os << "noCLog=" << noCLog << endl
     << "noCSol=" << noCSol << endl      
     << "noCMat=" << noCMat << endl
     << "noCRot=" << noCRot << endl
     << "       --------" << endl
     << "       " << byNam
     <<  " chars used for naming." << endl << endl;
  os << "byLog = " << byLog/mb << " logicalparts " << endl
     << "byNam = " << byNam/mb << " string for names " << endl
     << "byRot = " << byRot/mb << " rotations " << endl
     << "bySol = " << bySol/mb << " solids " << endl
     << "byMat = " << byMat/mb << " materials " << endl
     << "byPos = " << byPos/mb << " posparts " << endl
     << "byGra = " << byGra/mb << " graph-struct " << endl
     << "-----------------------" << endl 
     << "OVERALL: " << bytes / mb << " MByte" << endl;
  
}
