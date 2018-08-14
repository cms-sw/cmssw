#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_DD_HTML_FORMATTER_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_DD_HTML_FORMATTER_H

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

class DDHtmlFormatter;

std::ostream & operator<<(std::ostream & o, const DDHtmlFormatter & f);

class DDHtmlFormatter
{
public:
 
  typedef std::map<std::string,std::set<std::string> > ns_type;
 
  explicit DDHtmlFormatter() { }
  DDHtmlFormatter(const DDHtmlFormatter & f) : os_(f.os_.str()) { }
 
  DDHtmlFormatter header(const std::string & text, const std::string & style="../../style.css");
  DDHtmlFormatter footer();
 
  DDHtmlFormatter br() { pre(); os_ << "<br>" << std::endl; return *this; }
  DDHtmlFormatter p(const std::string & content) { pre(); os_ << "<p>" << std::endl << content << std::endl << "</p>" << std::endl; return *this; }
 
  DDHtmlFormatter ul() { pre(); os_ << "<ul>" << std::endl; return *this;}
  DDHtmlFormatter li(const std::string & content) { pre(); os_ << "<li>" << content << "</li>" << std::endl; return *this;}
  DDHtmlFormatter ulEnd() { pre(); os_ << "</ul>" << std::endl; return *this;}
 
  DDHtmlFormatter h1(const std::string & content) { pre(); os_ << "<h1>" << content << "</h1>" << std::endl; return *this;}
  DDHtmlFormatter h2(const std::string & content) { pre(); os_ << "<h2>" << content << "</h2>" << std::endl; return *this;}
  DDHtmlFormatter h3(const std::string & content) { pre(); os_ << "<h3>" << content << "</h3>" << std::endl; return *this;}
 
  DDHtmlFormatter link(const std::string & url, const std::string & text, const std::string & target="_self");
  std::string           lnk(const std::string & url, const std::string & text, const std::string & target="_self");
 
  DDHtmlFormatter table(int border=0){ pre(); os_ << "<table border=\"" << border << "\">" << std::endl; return *this;}
  DDHtmlFormatter tableEnd() { pre(); os_ << "</table>" << std::endl; return *this;}
  DDHtmlFormatter tr() { pre(); os_ << " <tr>" << std::endl; return *this;}
  DDHtmlFormatter trEnd() { pre(); os_ << " </tr>" << std::endl; return *this;}
  DDHtmlFormatter td(const std::string & content) { pre(); os_ << "  <td>" << content << std::endl << "  </td>" << std::endl; return *this;}
 
  DDHtmlFormatter color(int red, int green, int blue){return *this;};
 
  void pre() { os_.str(""); }
 
  // std::string operator<<(std::string o) { o << os_; }
  mutable std::stringstream os_;

  DDHtmlFormatter& operator= ( const DDHtmlFormatter& ) = delete;
};

/** 
    Generates HTML for DD-namespaces 
*/
class DDNsGenerator 
{ 
public:
  DDNsGenerator(std::ostream & os, 
		const std::string & title, 
		const std::string & target, 
		const ns_type & n, 
		const std::string & text="") 
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
  DDFrameGenerator(std::ostream & os,
                   const std::string & title,
                   const std::string & n1 = "_ns", //frame names
		   const std::string & n2 = "_list",
		   const std::string & n3 = "_details",
		   const std::string & u1 = "ns.html", //url to be displayed in each frame
		   const std::string & u2 = "list.html",
		   const std::string & u3 = "details.html")
    : t_(title), n1_(n1), n2_(n2), n3_(n3), u1_(u1), u2_(u2), u3_(u3), os_(os) { }
   
  void doit();   

private:
  std::string t_, n1_, n2_, n3_, u1_, u2_, u3_;
  std::ostream & os_;
};  		     		     

class DDHtmlDetails
{
public:
  DDHtmlDetails(const std::string & cat, const std::string & txt);
  virtual bool details(std::ostream & os, const DDName &) = 0;	
  virtual ns_type & names() = 0;
  virtual ~DDHtmlDetails(){};
  const std::string & category() { return cat_; }
  const std::string & text() {return txt_; }
protected:  
  mutable ns_type names_;  
  std::string cat_, txt_;
  DDHtmlFormatter f_;
};		


class DDHtmlLpDetails : public DDHtmlDetails
{
public: 
  DDHtmlLpDetails(const std::string & cat, const std::string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(std::ostream & os, const DDName &) override;
  ns_type & names() override;
  
};

class DDHtmlMaDetails : public DDHtmlDetails
{
public: 
  DDHtmlMaDetails(const std::string & cat, const std::string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(std::ostream & os, const DDName &) override;
  ns_type & names() override;
  
};

class DDHtmlSoDetails : public DDHtmlDetails
{
public: 
  DDHtmlSoDetails(const std::string & cat, const std::string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(std::ostream & os, const DDName &) override;
  ns_type & names() override;
  
};

class DDHtmlRoDetails : public DDHtmlDetails
{
public: 
  DDHtmlRoDetails(const std::string & cat, const std::string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(std::ostream & os, const DDName &) override;
  ns_type & names() override;
  
};

class DDHtmlSpDetails : public DDHtmlDetails
{
public: 
  DDHtmlSpDetails(const std::string & cat, const std::string & txt) : DDHtmlDetails(cat,txt) {}
  bool details(std::ostream & os, const DDName &) override;
  ns_type & names() override;
  
};

void dd_to_html(DDHtmlDetails & det);
			 	         
void dd_html_frameset(std::ostream & os);

void dd_html_menu_frameset(std::ostream & os);

void dd_html_menu(std::ostream & os);

void dd_html_ro();

#endif
