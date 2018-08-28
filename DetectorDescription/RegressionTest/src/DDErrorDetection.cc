#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

#include <fstream>

#include "DetectorDescription/Core/interface/Store.h"
//***** Explicit template instantiation of Singleton
#include "DetectorDescription/Core/interface/Singleton.icc"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
//**** to get rid of compile errors about ambiguous delete of Stores
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Specific.h"

using namespace std;

template class DDI::Singleton<std::map<std::string,std::set<DDLogicalPart> > >;
template class DDI::Singleton<std::map<std::string,std::set<DDMaterial> > >;
template class DDI::Singleton<std::map<std::string,std::set<DDSolid> > >;
template class DDI::Singleton<std::map<std::string,std::set<DDRotation> > >;
template class DDI::Singleton<std::map<std::string,std::set<DDSpecifics> > >;

//*****

DDErrorDetection::DDErrorDetection( const DDCompactView& cpv)
{
  DDMaterial::StoreT::instance().setReadOnly(false);
  DDSolid::StoreT::instance().setReadOnly(false);
  DDLogicalPart::StoreT::instance().setReadOnly(false);
  DDSpecifics::StoreT::instance().setReadOnly(false);
  DDRotation::StoreT::instance().setReadOnly(false);

  scan(cpv);
}

DDErrorDetection::~DDErrorDetection() {
  DDMaterial::StoreT::instance().setReadOnly(true);
  DDSolid::StoreT::instance().setReadOnly(true);
  DDLogicalPart::StoreT::instance().setReadOnly(true);
  DDSpecifics::StoreT::instance().setReadOnly(true);
  DDRotation::StoreT::instance().setReadOnly(true); 
}


void DDErrorDetection::scan( const DDCompactView& cpv )
{
  std::cout << "DDErrorDetection::scan(): Scanning for DDD errors ..." << std::flush;
  
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

  std::cout << " ... finished." << std::endl;
}

void DDErrorDetection::errors()
{
  std::cout << "What does DDErrorDetection::errors() do? nothing." << std::endl;
}

void DDErrorDetection::warnings()
{
  std::cout << "What does DDErrorDetection::warnings() do? nothing." << std::endl;
}

// ddname as std::string, std::set<edges>
const std::map<std::string, std::set<DDLogicalPart> > & DDErrorDetection::lp_cpv( const DDCompactView & cpv)
{
  static std::map<std::string, std::set<DDLogicalPart> > result_;
  if (!result_.empty()) return result_;
  
  const auto & g = cpv.graph();
  
  std::map<std::string, std::set<DDLogicalPart> >::const_iterator it(lp_err::instance().begin()),
                                                       ed(lp_err::instance().end());
  for (; it != ed; ++it) {
    std::set<DDLogicalPart>::const_iterator sit(it->second.begin()), sed(it->second.end());
    for( ; sit != sed; ++sit) {
      const DDLogicalPart & lp = *sit;
      auto er = g.edges(lp);
      if (g.nodeIndex(lp).second) {
        result_.insert(make_pair(lp.ddname().fullname(), std::set<DDLogicalPart>()));  
      }
      for (; er.first != er.second; ++er.first) {
         result_[lp.ddname().fullname()].insert(g.nodeData(er.first->first));
      }
    }						       
  }		
  return result_;				       
}


const std::map<DDSolid, std::set<DDLogicalPart> > & DDErrorDetection::so_lp()
{
  static std::map<DDSolid, std::set<DDLogicalPart> > result_;
  if (!result_.empty()) return result_;
  
  const std::map<DDSolid, std::set<DDSolid> > & err_mat = so();
  std::map<DDSolid, std::set<DDSolid> >::const_iterator it(err_mat.begin()), ed(err_mat.end());
  for (; it != ed; ++it) {
    std::set<DDLogicalPart> s;
    DDSolid m(it->first);
    result_[m]=s;
    std::set<DDSolid>::const_iterator sit(it->second.begin()), sed(it->second.end());
    for(; sit != sed; ++sit) {
      result_[*sit] = s;
    }
    //std::cout << "insert: " << m.name() << std::endl;
  }
  DDLogicalPart::iterator<DDLogicalPart> lpit,lped; lped.end();
  for (; lpit != lped; ++lpit) {
    if (lpit->isDefined().second) {
      std::map<DDSolid, std::set<DDLogicalPart> >::iterator i = result_.find(lpit->solid());
      //std::cout << "searching: " << lpit->name() << std::endl;
      if ( i != result_.end() ) {
      //std::cout << std::endl << "FOUND: " << lpit->name() << std::endl << std::endl;
      i->second.insert(*lpit);
     } 
    }  
  }
  return result_;
}

/*
const std::map<DDSpecifics, std::set<pair<DDLogicalPart, std::string> > & DDErrorDetection::sp()
{
  static std::map<DDSpecifics, std::set<pair<DDLogicalPart, std::string> result_;
  if (result_.size()) return result_;
}
*/

const std::map<DDMaterial, std::set<DDLogicalPart> > & DDErrorDetection::ma_lp()
{
  static std::map<DDMaterial, std::set<DDLogicalPart> > result_;
  if (!result_.empty()) return result_;
  
  const std::vector<pair<std::string, std::string> > & err_mat = ma();
  std::vector<pair<std::string, std::string> >::const_iterator it(err_mat.begin()), ed(err_mat.end());
  for (; it != ed; ++it) {
    std::set<DDLogicalPart> s;
    DDMaterial m(it->second);
    result_[m]=s;
  }
  DDLogicalPart::iterator<DDLogicalPart> lpit,lped; lped.end();
  for (; lpit != lped; ++lpit) {
    if (lpit->isDefined().second) {
      std::map<DDMaterial, std::set<DDLogicalPart> >::iterator i = result_.find(lpit->material());
      if ( i != result_.end() ) {
      //std::cout << std::endl << "FOUND: " << lpit->name() << std::endl << std::endl;
      i->second.insert(*lpit);
     } 
    }  
  }
  return result_;
}

  
const std::vector<pair<std::string, std::string> > & DDErrorDetection::ma()
{
  static std::vector<pair<std::string, std::string> > result_;
  ofstream o("/dev/null");

  if (!result_.empty()) return result_;
  
  DDCheckMaterials(o,&result_);
  return result_;

/*
  */
}


const std::map<DDSolid,std::set<DDSolid> > & DDErrorDetection::so()
{
  static std::map<DDSolid, std::set<DDSolid> > result_;
  if (!result_.empty()) return result_;
 
  // build the material dependency graph
  using ma_graph_t = math::Graph<DDSolid,double>;
  using ma_walker_t = math::GraphWalker<DDSolid,double>;
    
  ma_graph_t mag;
  std::vector<DDSolid> errs;
  DDSolid::iterator<DDSolid> it, ed; ed.end();
  for (; it != ed; ++it) {
    DDSolid  ma = *it;
    if (ma.isDefined().second) {
      DDSolidShape sh = ma.shape();
      if ( (sh == DDSolidShape::ddunion) || (sh == DDSolidShape::ddintersection) || (sh == DDSolidShape::ddsubtraction) ) {
       DDBooleanSolid bs(ma);
       DDSolid a(bs.solidA()),b(bs.solidB());
       //DDRotation r(bs.rotation());
       //DDTranslation t(bs.translation);
       mag.addEdge(a, ma, 0);
       mag.addEdge(b, ma, 0);
     }  
    }
    else {
      errs.emplace_back(ma);
    }
  }
  
  std::vector<DDSolid>::const_iterator mit(errs.begin()),
    med(errs.end());
  for (; mit != med; ++mit) {

    ma_walker_t w(mag,*mit);
    while (w.next()) {
      result_[*mit].insert(w.current().first);
    }
    std::cout << std::endl;
  } 
  
  return result_;
}


void DDErrorDetection::report(const DDCompactView& cpv, ostream & o)
{
  
  o << std::endl << std::endl << "---> DDD ERROR REPORT <---" << std::endl << std::endl;
  o << "MISSING DEFINITIONS:" << std::endl << std::endl;
  o << "LogicalParts:" << std::endl
    << lp_err::instance() << std::endl;
  o << "Materials:" << std::endl
    << ma_err::instance()  << std::endl;
  o << "Solids:" << std::endl
    << so_err::instance() << std::endl;
  o << "Rotations:" << std::endl
    << ro_err::instance() << std::endl;
  o << "Specifics:" << std::endl
    << sp_err::instance() << std::endl;
  o << std::endl << "IMPLICATIONS OF MISSING DEFINITIONS:" << std::endl << std::endl;
 
  o << "A) LogicalParts that have missing definitions but are used in the geometr. hierarchy (PosParts):" << std::endl
    << "   Format: namespace:name: [name of child]*" << std::endl;
  o << lp_cpv(cpv) << std::endl;
  
  o << "B) Detailed report on Materials:" << std::endl;
  const std::vector<pair<std::string,std::string>> & res = ma();
  std::vector<pair<std::string,std::string>>::const_iterator it(res.begin()), ed(res.end());
  for (; it != ed; ++it) {
    std::cout << it->second << ":  " << it->first << std::endl;
  }
  std::cout << std::endl;

  
  o << "C) Solids affected by Solids that have missing definitions:" << std::endl;
  o << so() << std::endl;
   
  o << "D) LogicalParts affected by Materials of B):" << std::endl;
  o << ma_lp() << std::endl;
  
  o << "E) LogicalParts affected by Solids of C):" << std::endl;
  o << so_lp() << std::endl;
  
  // Did this ever exist?
  //  o << "F) Parent-Child positionings affected by Rotations that have missing definitions:"  << std::endl;
  o << std::endl;
  //nix();
}

bool DDErrorDetection::noErrorsInTheReport(const DDCompactView& cpv)
{
  return lp_err::instance().empty() &&
         ma_err::instance().empty() &&
         so_err::instance().empty() &&
         ro_err::instance().empty() &&
         sp_err::instance().empty() &&
         lp_cpv(cpv).empty() &&
         ma().empty() &&
         so().empty() &&
         ma_lp().empty() &&
         so_lp().empty();
}
