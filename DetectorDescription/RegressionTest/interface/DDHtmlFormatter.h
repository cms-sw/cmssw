#ifndef DDHtmlFormatter_h
#define DDHtmlFormatter_h


#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include "DetectorDescription/DDRegressionTest/interface/DDErrorDetection.h"


class DDHtmlFormatter;

ostream & operator<<(ostream & o, const DDHtmlFormatter & f);

class DDHtmlFormatter
{
public:
 
 typedef std::map<std::string,std::set<std::string> > ns_type;
 
 explicit DDHtmlFormatter() { }
 DDHtmlFormatter(const DDHtmlFormatter & f) : os_(f.os_.str()) { }
 
 DDHtmlFormatter header(const string & text, const string & style="../../style.css");
 DDHtmlFormatter footer();
 
 DDHtmlFormatter br() { pre(); os_ << "<br>" << endl; return *this; }
 DDHtmlFormatter p(const string & content) { pre(); os_ << "<p>" << endl << content << endl << "</p>" << endl; return *this; }
 
 DDHtmlFormatter ul() { pre(); os_ << "<ul>" << endl; return *this;}
 DDHtmlFormatter li(const string & content) { pre(); os_ << "<li>" << content << "</li>" << endl; return *this;}
 DDHtmlFormatter ulEnd() { pre(); os_ << "</ul>" << endl; return *this;}
 
 DDHtmlFormatter h1(const string & content) { pre(); os_ << "<h1>" << content << "</h1>" << endl; return *this;}
 DDHtmlFormatter h2(const string & content) { pre(); os_ << "<h2>" << content << "</h2>" << endl; return *this;}
 DDHtmlFormatter h3(const string & content) { pre(); os_ << "<h3>" << content << "</h3>" << endl; return *this;}
 
 DDHtmlFormatter link(const string & url, const string & text, const string & target="_self");
 string           lnk(const string & url, const string & text, const string & target="_self");
 
 DDHtmlFormatter table(int border=0){ pre(); os_ << "<table border=\"" << border << "\">" << endl; return *this;}
 DDHtmlFormatter tableEnd() { pre(); os_ << "</table>" << endl; return *this;}
 DDHtmlFormatter tr() { pre(); os_ << " <tr>" << endl; return *this;}
 DDHtmlFormatter trEnd() { pre(); os_ << " </tr>" << endl; return *this;}
 DDHtmlFormatter td(const string & content) { pre(); os_ << "  <td>" << content << endl << "  </td>" << endl; return *this;}
 
 DDHtmlFormatter color(int red, int green, int blue){return *this;};
 
 void pre() { os_.str(""); }
 
// string operator<<(string o) { o << os_; }
  mutable stringstream os_;
 
};






/** 
 Generates HTML for DD-namespaces 
*/
class DDNsGenerator 
{
 
public:
 DDNsGenerator(ostream & os, 
              const string & title, 
              const string & target, 
	       const ns_type & n, 
	       const string & text="") 
 : os_(os), title_(title), text_(text), target_(target), n_(n){ }
 
 void doit();

private:

 std::ostream & os_;  
 std::string title_, text_, target_; 
 const ns_type & n_;
}; 

class DDFrameGenerator
{
public:
  DDFrameGenerator(ostream & os,
                   const string & title,
                   const string & n1 = "_ns", //frame names
		     const string & n2 = "_list",
		     const string & n3 = "_details",
		     const string & u1 = "ns.html", //url to be displayed in each frame
		     const string & u2 = "list.html",
		     const string & u3 = "details.html")
   : t_(title), n1_(n1), n2_(n2), n3_(n3), u1_(u1), u2_(u2), u3_(u3), os_(os) { }
   
  void doit();   

private:
  std::string t_, n1_, n2_, n3_, u1_, u2_, u3_;
  std::ostream & os_;
};  		     		     
	
// =============================================================================================================
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class DDHtmlDetails
{
public:
  DDHtmlDetails(const string & cat, const string & txt);
  virtual bool details(ostream & os, const DDName &) = 0;	
  virtual ns_type & names() = 0;
  virtual ~DDHtmlDetails(){};
  const string & category() { return cat_; }
  const string & text() {return txt_; }
protected:  
  mutable ns_type names_;  
  std::string cat_, txt_;
  DDHtmlFormatter f_;
};		


class DDHtmlLpDetails : public DDHtmlDetails
{
public: 
  DDHtmlLpDetails(const string & cat, const string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(ostream & os, const DDName &);
  ns_type & names();
  
};

class DDHtmlMaDetails : public DDHtmlDetails
{
public: 
  DDHtmlMaDetails(const string & cat, const string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(ostream & os, const DDName &);
  ns_type & names();
  
};

class DDHtmlSoDetails : public DDHtmlDetails
{
public: 
  DDHtmlSoDetails(const string & cat, const string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(ostream & os, const DDName &);
  ns_type & names();
  
};

class DDHtmlRoDetails : public DDHtmlDetails
{
public: 
  DDHtmlRoDetails(const string & cat, const string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(ostream & os, const DDName &);
  ns_type & names();
  
};

class DDHtmlSpDetails : public DDHtmlDetails
{
public: 
  DDHtmlSpDetails(const string & cat, const string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(ostream & os, const DDName &);
  ns_type & names();
  
};


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// =============================================================================================================
	
void dd_to_html(DDHtmlDetails & det);
			 	         
void dd_html_frameset(ostream & os);

void dd_html_menu_frameset(ostream & os);

void dd_html_menu(ostream & os);

void dd_html_ro();



#endif
