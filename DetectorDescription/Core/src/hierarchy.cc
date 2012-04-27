// Two modules of CLHEP are partly used in DDD
// . unit definitions (such as m, cm, GeV, ...) of module CLHEP/Units
// . rotation matrices and translation std::vectors of module CLHEP/Vector
//   (they are typedef'd to DDRotationMatrix and DDTranslation in
//   DDD/DDCore/interface/DDTransform.h
#include "CLHEP/Units/GlobalSystemOfUnits.h"

/*
  Doc!
*/
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDName.h"

#include <fstream>
#include <string>
#include <set>
#include <iostream>
#include <map>

std::string link(std::string & nm, std::string & ns)
{
   return ns + nm + ".html";
}

void streamSolid(DDSolid s, std::ostream & os) {
  DDSolidShape sp = s.shape();
  if ( (sp==ddpolycone_rrz) || (sp==ddpolyhedra_rrz) ) {
    unsigned int i = 0;
    const std::vector<double> & p = s.parameters();
    if (sp==ddpolyhedra_rrz){
      os << "numSides=" << p[0] << " ";
      ++i;
    }
    os <<"startPhi[deg]=" << p[i]/deg << " deltaPhi[deg]" << p[i+1]/deg << std::endl;
    i +=2;
    os << "<table><tr><td>z[cm]</td><td>rMin[cm]</td><td>rMax[cm]</td></tr>";
    while ( i+1 < p.size()) {
      os << "<tr><td>" << p[i]/cm << "</td>";
      os << "<td>" << p[i+1]/cm << "</td>";
      os << "<td>" << p[i+2]/cm << "</td></tr>" << std::endl;
      i = i+3;  
    }
    os << "</table>" << std::endl;
  } else
  {
    os << s;
  }
}

void generateHtml(std::vector<DDLogicalPart> & v, std::map<DDLogicalPart,int> & c)
{
   static DDCompactView cpv;
   std::string name = v.back().name().name();
   std::string ns   = v.back().name().ns();
   std::string fname = link(name,ns);
   std::ofstream file(fname.c_str());
   
   
   file << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">" << std::endl;
   file << "<html><head><title>DDD-part:" << ns << " " << name << "</title>";
   file << "<body><h2>"
        <<  "name=" << name << "  namespace=" << ns << "</h2>" << std::endl;
   file << "<br>weight = " << cpv.weight(v.back())/kg << "kg<br>" << std::endl;
   //file << "volume = " << v.back().solid().volume()/m3 << "m3<br>" << std::endl;
   //file << "solid  = " << typeid(v.back().solid().rep()).name() << "  ";
           streamSolid(v.back().solid(),file);
   file << "<br>" << std::endl;
   std::string mname = v.back().material().name().name();
   std::string mns   = v.back().material().name().ns();
   std::string lk = mns + ".html#" + mname;
   file << "material = " << "<a href=\""  << lk <<  "\">"  
                         << mname << "</a>" << std::endl;
   file << "density = "  << v.back().material().density()/g*cm3 << "g/cm3<br>" << std::endl;
   			 
   // link children
   file << "<br>Children:<br>" << std::endl;
   std::map<DDLogicalPart,int>::iterator cit = c.begin();
   for (; cit != c.end(); ++cit) {
     std::string nm, nsp;
     nm = cit->first.name().name(); nsp = cit->first.name().ns();
     file << "<a href=\"" << link(nm,nsp) << "\">"
          << nm  << "(" << cit->second << ")</a> " << std::endl;
   }
   
   file << "<br><br>Ancestors:<br>";
   int s = v.size();
   --s; --s;
   for (; s>=0; --s) {
     std::string nm,nsp;
     nm = v[s].name().name();
     nsp = v[s].name().ns();
     file << std::string(2*s,' ') << "<a href=\"" << link(nm,nsp) << "\">"
              << nm << "</a><br> " << std::endl;
   }
   file << "<br>SpecPars:<br>" << std::endl;
   file << "<table><tbody>" << std::endl;
   typedef std::vector< std::pair<DDPartSelection*,DDsvalues_type*> > sv_type;
   sv_type sv = v.back().attachedSpecifics();
   sv_type::iterator sit = sv.begin();
   for (; sit != sv.end(); ++sit) {
     file << "<tr>" << std::endl
          << " <td>" << *(sit->first) <<"</td>" << std::endl;
     file << " <td>";	  
     DDsvalues_type::iterator svit = sit->second->begin();
     for(; svit != sit->second->end(); ++svit) {
       file << svit->second << "<br>" <<std::endl;
     }
     file << "</td>" << std::endl
          << "</tr>" << std::endl;
     
   }
   file << "</table></tbody>" << std::endl;
   
   file << "<br></body></html>" <<std::endl;
   file.close();
}

void writeMaterials(std::map<std::string,std::set<DDMaterial> > & m)
{
   std::map<std::string, std::set<DDMaterial> >::iterator it = m.begin();
   for (; it != m.end(); ++it) {
     std::string fname = it->first + ".html"; 
     std::ofstream file(fname.c_str());
     file << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">" << std::endl;
     file << "<html><head><title>DDD-Materials</title>";
     file << "<body>";
     std::set<DDMaterial>::iterator mit = it->second.begin();
     for (; mit != it->second.end(); ++mit) {
       file << "<a name=\"" << mit->name().name() << "\">";
       file << mit->name().name() << " d=" << mit->density()/g*cm3 
            << "g/cm3";
	    
       file << "</a><br>";
     }	    
     file << "</body></html>" << std::endl;
     file.close();
   }

}

void hierarchy(const DDLogicalPart & parent)
{
  static  DDCompactView cpv ;
  static DDCompactView::graph_type g = cpv.graph();
  static int count=0;
  static std::vector<DDLogicalPart> history;
  static std::map<std::string,std::set<DDMaterial> > materials;
  //DDCompactView::graph_type::adj_iterator it = g.begin();
  
  history.push_back(parent);
  materials[parent.material().name().ns()].insert(parent.material());
  std::cout << history.size() << std::string(2*count,' ') << " " << parent.ddname() << std::endl;
  DDCompactView::graph_type::edge_range er = g.edges(parent);
  DDCompactView::graph_type::edge_iterator eit = er.first;
  std::map<DDLogicalPart,int> children;
  for (; eit != er.second; ++eit) {  
     children[g.nodeData(*eit)]++;
  }
  
  generateHtml(history,children);
  
  std::map<DDLogicalPart,int>::iterator cit = children.begin();
  for (; cit != children.end(); ++cit) {
     ++count;
     hierarchy(cit->first);
     history.pop_back();
     --count;
  }
  
  writeMaterials(materials);
}

