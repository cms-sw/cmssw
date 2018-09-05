#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/Singleton.icc"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDEnums.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Specific.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

using namespace std;
using namespace dd::operators;

ostream & operator<<(ostream & o, const DDHtmlFormatter & f)
{
  o << f.os_.str(); f.os_.str("");
  return o;
}

DDHtmlFormatter DDHtmlFormatter::header(const string & title, const string & style)
{
  pre();
  os_ << "<html>\n<head>\n<title>" << title << "</title>" << endl;
  os_ << "<link rel=\"stylesheet\" type=\"text/css\" href=\"" << style << "\">" << endl;
  //os_ << "</head>\n<body>" << endl;
  return *this;
}

DDHtmlFormatter DDHtmlFormatter::link(const string & url, const string & txt, const string & target)
{
  pre();
  os_ << lnk(url,txt,target) << endl;
  return *this;
}

string DDHtmlFormatter::lnk(const string & url, const string & txt, const string & target)
{
   string result;
   result = string("<a href=\"") + url;
   result += string("\" target=\"") + target;
   result += string("\">")+ txt + string("</a>");
   return result; 
}

DDHtmlFormatter DDHtmlFormatter::footer()
{
  pre();
  os_ << "</body></html>" << endl;
  return *this;
}


//=============================================================================================================
//=============================================================================================================


void DDNsGenerator::doit()
{  
  DDHtmlFormatter f;
  os_ << f.header(title_,"../style.css");
  os_ << f.h2(title_) << f.p(text_);// << endl;
  ns_type::const_iterator it = n_.begin(); 
  ns_type::const_iterator ed = n_.end(); 
  os_ << f.ul();
  for (; it != ed; ++it) {
    os_ << f.li(f.lnk(it->first + "/list.html"  , it->first, target_));
  }
  os_ << f.ulEnd() << endl;
  os_ << f.footer() << endl;
} 

DDHtmlDetails::DDHtmlDetails(const string & cat, const string & txt) : cat_(cat), txt_(txt) { }

ns_type & DDHtmlLpDetails::names() 
{
   DDLogicalPart lp;
   findNameSpaces(lp, names_);
   return names_;
}

ns_type & DDHtmlMaDetails::names() 
{
   DDMaterial lp;
   findNameSpaces(lp, names_);
   return names_;
}

ns_type & DDHtmlSoDetails::names() 
{
   DDSolid lp;
   findNameSpaces(lp, names_);
   return names_;
}

ns_type & DDHtmlSpDetails::names() 
{
   DDSpecifics lp;
   findNameSpaces(lp, names_);
   return names_;
}

ns_type & DDHtmlRoDetails::names() 
{
   DDRotation lp;
   findNameSpaces(lp, names_);
   return names_;
}


bool DDHtmlSoDetails::details(ostream & os, const DDName & nm) 
{
  os << f_.header("Solid Details")
     << f_.h3(">> formatting under construction <<"); 
  os << DDSolid(nm); return true; 
}

bool DDHtmlSpDetails::details(ostream & os, const DDName & nm) 
{
  os << f_.header("SpecPars Details")
     << f_.h3(">> formatting under construction <<"); 
    os << DDSpecifics(nm); return true; 
}

bool DDHtmlRoDetails::details(ostream & os, const DDName & nm) 
{ 
  os << f_.header("Rotations Details");

  DDRotation ddr(nm);
  if ( ddr.isDefined().second == false ) {
    os << "<b>ERROR!</b><br><p>The Rotation " << nm << " is not defined!</p>" << endl;
    return false;
  }
  DD3Vector x, y, z;
  ddr.matrix().GetComponents(x, y, z);
  os << f_.h2("Rotation: " + nm.fullname());
  os << f_.h3("GEANT3 style:"); 
  os << "<table border=\"0\">" << endl
     << "<tr><td>thetaX =</td><td>" << CONVERT_TO( x.Theta(), deg ) << " deg</td><tr>" << endl
     << "<tr><td>phiX =</td><td>" << CONVERT_TO( x.Phi(), deg ) << " deg</td><tr>" << endl
     << "<tr><td>thetaY =</td><td>" << CONVERT_TO( y.Theta(), deg ) << " deg</td><tr>" << endl
     << "<tr><td>phiY =</td><td>" << CONVERT_TO( y.Phi(), deg ) << " deg</td><tr>" << endl     
     << "<tr><td>thetaZ =</td><td>" << CONVERT_TO( z.Theta(), deg ) << " deg</td><tr>" << endl
     << "<tr><td>phiZ =</td><td>" << CONVERT_TO( z.Phi(), deg ) << " deg</td><tr>" << endl     
     << "</table>";
     
  os << f_.h3("Rotation axis & angle (theta,phi,angle)") << endl;   
  os << DDRotation(nm); return true; 
}

bool DDHtmlMaDetails::details(ostream & os, const DDName & nm)
{
  typedef DDI::Singleton<map<DDMaterial, set<DDLogicalPart> > > parts_t;
  static bool once = false;
  if (!once) {
    once=true;
    DDLogicalPart::iterator<DDLogicalPart> it, ed;
    ed.end();
    
    for (; it != ed; ++it) {
      if (it->isDefined().second)
        parts_t::instance()[it->material()].insert(*it);
    }
  }
  
  string s = nm.ns() + " : " + nm.name();
  DDMaterial ma(nm);
  os << f_.header(s);
  os << f_.h2("Material <b>" + s + "</b>");
  os << f_.br();
  if ( ma.isDefined().second == false ) {
    os << "<b>ERROR!<b><br><p>The Material is not defined in namespace " << nm.ns() << "! </p>" << endl;
    return false;
  }
  
  os << "<p>density = " << CONVERT_TO( ma.density(), g_per_cm3 ) << " g/cm3 </p>" << endl;
  int co = ma.noOfConstituents();
  if ( co ) {
    os << f_.p("Composites by fraction-mass:");
    os << f_.table() 
       << f_.tr() << f_.td("<b>fm</b>") << f_.td("<b>Material</b>") << f_.td("<b>elementary?</b>") << f_.trEnd();
    for(int i=0; i<co; ++i) {
      pair<DDMaterial,double> fm = ma.constituent(i);
      string elem = "ERROR";
      DDMaterial m = fm.first;
      double frac = fm.second;

      if (m.isDefined().second) {
        if (m.noOfConstituents()) 
	    elem = "no";
	 else
	    elem = "yes";
      }
      os << f_.tr() << "<td>" << frac << "</td>" 
         << f_.td(f_.lnk("../" + m.ddname().ns() + "/" + m.ddname().name() + ".html", m.ddname().fullname(), "_popup"))
	  << f_.td(elem) << f_.trEnd();
    }
    os << f_.tableEnd();
  }
  else { // if ( co ) ...
    os << f_.p("ElementaryMaterial:");
    os << "<p>z = " << ma.z() << "</p>" << endl;
    os << "<p>a = " << CONVERT_TO( ma.a(), g_per_mole ) << "g/mole</p>" << endl;
  }
  
   
  const set<DDLogicalPart> & lps = parts_t::instance()[ma];
  set<DDLogicalPart>::const_iterator it(lps.begin()), ed(lps.end());
  if ( it != ed ) {
    os << f_.h3("Material used in following LogicalParts:") << endl;
    os << "<p>" << endl;
  }
  for (; it != ed; ++it ) {
    const DDName & n = it->ddname();
    os << f_.link("../../lp/" + n.ns() + "/" + n.name() + ".html", n.fullname(), "_popup" );
  }
  os << "</p>" << endl;
  return true;
}

bool DDHtmlLpDetails::details(ostream & os, const DDName & nm) 
{
  static bool once = false;
  typedef DDI::Singleton< map<DDLogicalPart, set<DDSpecifics> > > lp_sp_t;
  if ( !once ) {
    once = true;
    DDSpecifics::iterator<DDSpecifics> it, ed;
    ed.end();
    for (; it != ed; ++it ) {
      if (it->isDefined().second) {
        const vector<DDPartSelection> & ps = it->selection();
	 vector<DDPartSelection>::const_iterator pit(ps.begin()), ped(ps.end());
	 for (; pit != ped; ++pit) {
	   if (!pit->empty()) {
	     lp_sp_t::instance()[pit->back().lp_].insert(*it);
	   }
	 }
      }
    }
  }
  string s = nm.ns() + " : " + nm.name();
  DDLogicalPart lp(nm);
  os << f_.header(s);
  os << f_.h2("LogicalPart <b>" + s + "</b>");
  os << f_.br();
  if ( lp.isDefined().second == false ) {
    os << "<b>ERROR!<b><br><p>The LogicalPart is not defined in namespace " << nm.ns() << "! </p>" << endl;
    return false;
  }
  
  string so_url = "../../so/" + lp.solid().ddname().ns() + "/" + lp.solid().ddname().name() + ".html";
  string ma_url = "../../ma/" + lp.material().ddname().ns() + "/" + lp.material().ddname().name() + ".html";
  string so_nm = lp.solid().ddname().ns() + ":" + lp.solid().ddname().name();
  string ma_nm = lp.material().ddname().ns() + ":" + lp.material().ddname().name();
  os << f_.table() 
     << f_.tr() << f_.td("Category") << f_.td( DDEnums::categoryName(lp.category()) ) << f_.trEnd()
     << f_.tr() << f_.td("Solid") << f_.td( f_.lnk(so_url, so_nm, "_popup" )) << f_.trEnd()
     << f_.tr() << f_.td("Material") << f_.td(f_.lnk(ma_url, ma_nm, "_popup")) << f_.trEnd();
  os << f_.tableEnd();     
  
  typedef map<DDLogicalPart, set<DDSpecifics> > lp_sp_type;
  const lp_sp_type & lp_sp = lp_sp_t::instance();
  lp_sp_type::const_iterator lpspit = lp_sp.find(lp);
  if (lpspit != lp_sp.end()) {
    os << f_.h3("assigned SpecPars (Specifics):");
    set<DDSpecifics>::const_iterator it(lpspit->second.begin()), ed(lpspit->second.end());
    os << "<p>" << endl;
    for (; it != ed; ++it) {
      os << f_.link("../../sp/" + it->ddname().ns() + "/" + it->ddname().name() + ".html", it->ddname().fullname(), "_popup")
         << " " << endl;
    }
    os << "</p>" << endl;
  }
        
  os << f_.footer();
  return true;
}

//=============================================================================================================

void dd_to_html(DDHtmlDetails & dtls)
{
  cout << "---> dd_to_html() called with category=" << dtls.category() << endl;
  const string & category = dtls.category();
  const string & text     = dtls.text();
  ns_type & names         = dtls.names();
  
  mkdir( category.c_str(), 0755 );
  
  // first the namespaces
  string ns_fname = category + "/ns.html";
  ofstream ns_file(ns_fname.c_str());
  DDNsGenerator ns_gen(ns_file, text, "_list", names, "");  
  ns_gen.doit();
  ns_file.close();
  
  // list all logical parts per namespace
  ns_type::const_iterator it(names.begin()), ed(names.end());
  for( ; it != ed; ++it ) {
    
    const string & ns = it->first;
    
    // create directories named like the namespaces
    string dir = category + "/" + ns;
    mkdir( dir.c_str(), 0755 );

    // create a html file listing all instances of a namespace
    string fname = category + "/" + ns + "/list.html";
    ofstream list_file(fname.c_str());
    DDHtmlFormatter f;
    list_file << f.header(text)
              << f.p("Instances in Namespace <b>" + ns + "</b><br>");
    list_file << f.ul();
    // loop over all instances of a single namespace
    set<string>::const_iterator nit(it->second.begin()), ned(it->second.end());
    for(; nit != ned; ++nit) {

      const string & nm = *nit;
      string result_s = nm;

      // details for each instance 
      string d_fname = category + "/" + ns + "/" + nm + ".html";
      ofstream detail_file(d_fname.c_str());
      DDName an(nm,ns);
      bool result = dtls.details(detail_file, an);

      if (!result) result_s = ">> ERROR: " + nm + " <<";
      list_file << f.li(f.lnk(nm+".html", result_s, "_details"));	   
      
    }
    list_file << f.ulEnd() << f.footer();
  }
}

//=============================================================================================================
//=============================================================================================================


void DDFrameGenerator::doit()
{
  DDHtmlFormatter f;
  os_ << f.header(t_);
  os_ << "<frameset cols=\"25%,*\">" << endl;
  os_ << "  <frameset rows=\"50%,*\">" << endl;
  os_ << "   <frame src=\"" << u1_ << "\" name=\"" << n1_ << "\">" << endl;
  os_ << "   <frame src=\"" << u2_ << "\" name=\"" << n2_ << "\">" << endl;
  os_ << "  </frameset>" << endl;
  os_ << " <frame src=\"" << u3_ << "\" name=\"" << n3_ << "\">" << endl;
  os_ << "</frameset>" << endl;
  os_ << f.footer() << endl;  
}

void dd_html_frameset(ostream & os)
{
  DDHtmlFormatter f;
  os << f.header("DDD Reports");
  os << "<frameset rows=\"50%,50%\"> " << endl
     << "  <frameset cols=\"50%,50%\">" << endl
     << "    <frame name=\"_ns\" src=\"ns.html\">" << endl
     << "    <frame name=\"_list\">" << endl
     << "  </frameset>" << endl
     << "  <frameset cols=\"50%,50%\">" << endl
     << "    <frame name=\"_details\">" << endl
     << "    <frame name=\"_popup\">" << endl
     << "  </frameset>" << endl
     << "</frameset>" << endl
     << endl;
  os << f.footer();

}

void dd_html_menu_frameset(ostream & os)
{
  DDHtmlFormatter f;
  os << f.header("DDD Web Representation");
  os << "<frameset cols=\"20%,80%\">" << endl
     << "  <frame name=\"_menu\" src=\"menu.html\">" << endl
     << "  <frame name=\"_selection\" >" << endl
     << "</frameset>" << endl;
     
  os << f.footer();    
}


void dd_html_menu(ostream & os)
{
  DDHtmlFormatter f;
  os << f.header("DDD Web Main Menu","style.css");
  os << f.h1("Select a Category:") 
     << f.p(f.lnk("lp/index.html", "LogicalParts", "_selection")) 
     << f.p(f.lnk("ma/index.html", "Materials", "_selection"))
     << f.p(f.lnk("so/index.html", "Solids", "_selection"))
     << f.p(f.lnk("ro/index.html", "Rotations", "_selection"))
     << f.p(f.lnk("sp/index.html", "SpecPars", "_selection"))
     ;
     
  os << f.footer();
}



