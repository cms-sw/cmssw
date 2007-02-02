namespace std { } using namespace std;

#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include <iostream>
#include <fstream>

DDErrorDetection::DDErrorDetection()
{
  scan();
}


void DDErrorDetection::scan()
{
  cout << "DDErrorDetection::scan(): Scanning for DDD errors ..." << flush;
  
  DDLogicalPart lp_dummy;
  DDMaterial ma_dummy;
  DDRotation ro_dummy;
  DDSpecifics sp_dummy;
  DDSolid so_dummy;
  
  lp_err::instance() = dd_error_scan(lp_dummy);
  ma_err::instance() = dd_error_scan(ma_dummy);
  ro_err::instance() = dd_error_scan(ro_dummy);
  sp_err::instance() = dd_error_scan(sp_dummy);
  so_err::instance() = dd_error_scan(so_dummy);  

  cout << " ... finished." << endl;
}

void DDErrorDetection::errors()
{
}

void DDErrorDetection::warnings()
{
}





// ddname as string, set<edges>
const map<string, set<DDLogicalPart> > & DDErrorDetection::lp_cpv()
{
  static map<string, set<DDLogicalPart> > result_;
  if (result_.size()) return result_;
  
  DDCompactView cpv;
  const graph_type & g = cpv.graph();
  
  map<string, set<DDLogicalPart> >::const_iterator it(lp_err::instance().begin()),
                                                       ed(lp_err::instance().end());
  for (; it != ed; ++it) {
    set<DDLogicalPart>::const_iterator sit(it->second.begin()), sed(it->second.end());
    for( ; sit != sed; ++sit) {
      const DDLogicalPart & lp = *sit;
      graph_type::const_edge_range er = g.edges(lp);
      if (g.nodeIndex(lp).second) {
        result_.insert(make_pair(string(lp.ddname()), set<DDLogicalPart>()));  
      }
      for (; er.first != er.second; ++er.first) {
         result_[lp.ddname()].insert(g.nodeData(er.first->first));
      }
    }						       
  }		
  return result_;				       
}


const map<DDSolid, set<DDLogicalPart> > & DDErrorDetection::so_lp()
{
  static map<DDSolid, set<DDLogicalPart> > result_;
  if (result_.size()) return result_;
  
  const map<DDSolid, set<DDSolid> > & err_mat = so();
  map<DDSolid, set<DDSolid> >::const_iterator it(err_mat.begin()), ed(err_mat.end());
  for (; it != ed; ++it) {
    set<DDLogicalPart> s;
    DDSolid m(it->first);
    result_[m]=s;
    set<DDSolid>::const_iterator sit(it->second.begin()), sed(it->second.end());
    for(; sit != sed; ++sit) {
      result_[*sit] = s;
    }
    //cout << "insert: " << m.name() << endl;
  }
  DDLogicalPart::iterator<DDLogicalPart> lpit,lped; lped.end();
  for (; lpit != lped; ++lpit) {
    if (lpit->isDefined().second) {
      map<DDSolid, set<DDLogicalPart> >::iterator i = result_.find(lpit->solid());
      //cout << "searching: " << lpit->name() << endl;
      if ( i != result_.end() ) {
      //cout << endl << "FOUND: " << lpit->name() << endl << endl;
      i->second.insert(*lpit);
     } 
    }  
  }
  return result_;
}

/*
const map<DDSpecifics, set<pair<DDLogicalPart, string> > & DDErrorDetection::sp()
{
  static map<DDSpecifics, set<pair<DDLogicalPart, string> result_;
  if (result_.size()) return result_;
}
*/

const map<DDMaterial, set<DDLogicalPart> > & DDErrorDetection::ma_lp()
{
  static map<DDMaterial, set<DDLogicalPart> > result_;
  if (result_.size()) return result_;
  
  const vector<pair<string,DDName> > & err_mat = ma();
  vector<pair<string,DDName> >::const_iterator it(err_mat.begin()), ed(err_mat.end());
  for (; it != ed; ++it) {
    set<DDLogicalPart> s;
    DDMaterial m(it->second);
    result_[m]=s;
    //cout << "insert: " << m.name() << endl;
  }
  DDLogicalPart::iterator<DDLogicalPart> lpit,lped; lped.end();
  for (; lpit != lped; ++lpit) {
    if (lpit->isDefined().second) {
      map<DDMaterial, set<DDLogicalPart> >::iterator i = result_.find(lpit->material());
      //cout << "searching: " << lpit->name() << endl;
      if ( i != result_.end() ) {
      //cout << endl << "FOUND: " << lpit->name() << endl << endl;
      i->second.insert(*lpit);
     } 
    }  
  }
  return result_;
}

  
const vector<pair<string,DDName> > & DDErrorDetection::ma()
{
  static vector<pair<string,DDName> > result_;
  ofstream o("/dev/null");

  if (result_.size()) return result_;
  
  DDCheckMaterials(o,&result_);
  return result_;

/*
  */
}


const map<DDSolid,set<DDSolid> > & DDErrorDetection::so()
{
  static map<DDSolid, set<DDSolid> > result_;
  if (result_.size()) return result_;
 
  // build the material dependency graph
  typedef graph<DDSolid,double> ma_graph_t;
  typedef graphwalker<DDSolid,double> ma_walker_t;
    
  ma_graph_t mag;
  vector<DDSolid> errs;
  DDSolid::iterator<DDSolid> it, ed; ed.end();
  for (; it != ed; ++it) {
    DDSolid  ma = *it;
    if (ma.isDefined().second) {
      DDSolidShape sh = ma.shape();
      if ( (sh == ddunion) || (sh == ddintersection) || (sh == ddsubtraction) ) {
       DDBooleanSolid bs(ma);
       DDSolid a(bs.solidA()),b(bs.solidB());
       //DDRotation r(bs.rotation());
       //DDTranslation t(bs.translation);
       mag.addEdge(a, ma, 0);
       mag.addEdge(b, ma, 0);
     }  
    }
    else {
      errs.push_back(ma);
    }
  }
  
    vector<DDSolid>::const_iterator mit(errs.begin()),
                                      med(errs.end());
    for (; mit != med; ++mit) {

    try {
      // loop over erroreous materials
      ma_walker_t w(mag,*mit);
      while (w.next()) {
        result_[*mit].insert(w.current().first);
      }
      cout << endl;
    } 
    catch(DDSolid m) {
      ;
      //cout << "no such material: " << m << " for creating a walker." << endl;
    }
   } 
   return result_;
}


void DDErrorDetection::report(ostream & o)
{
  
  o << endl << endl << "---> DDD ERROR REPORT <---" << endl << endl;
  o << "MISSING DEFINITIONS:" << endl << endl;
  o << "LogicalParts:" << endl
    << lp_err::instance() << endl;
  o << "Materials:" << endl
    << ma_err::instance()  << endl;
  o << "Solids:" << endl
    << so_err::instance() << endl;
  o << "Rotations:" << endl
    << ro_err::instance() << endl;
  o << "Specifics:" << endl
    << sp_err::instance() << endl;
  o << endl << "IMPLICATIONS OF MISSING DEFINITIONS:" << endl << endl;
 
  o << "A) LogicalParts that have missing definitions but are used in the geometr. hierarchy (PosParts):" << endl
    << "   Format: namespace:name: [name of child]*" << endl;
  o << lp_cpv() << endl;
  
  o << "B) Detailed report on Materials:" << endl;
  const vector<pair<string,DDName> > & res = ma();
  vector<pair<string,DDName> >::const_iterator it(res.begin()), ed(res.end());
  for (; it != ed; ++it) {
    cout << it->second << ":  " << it->first << endl;
  }
  cout << endl;

  
  o << "C) Solids affected by Solids that have missing definitions:" << endl;
  o << so() << endl;
   
  o << "D) LogicalParts affected by Materials of B):" << endl;
  o << ma_lp() << endl;
  
  o << "E) LogicalParts affected by Solids of C):" << endl;
  o << so_lp() << endl;
  
  o << "F) Parent-Child positionings affected by Rotations that have missing definitions:"  << endl;
  
  o << "E) " << endl;
  
  o << endl;
  //nix();
}
