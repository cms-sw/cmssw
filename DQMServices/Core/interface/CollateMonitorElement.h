#ifndef CollateMonitorElement_h
#define CollateMonitorElement_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/StringUtil.h"

#include <set>
#include <string>

/** The base class for the collation of Monitor Elements
 */
class CollateMonitorElement : public StringUtil
{

 public:
  
  CollateMonitorElement(const std::string & name, const std::string title, 
			const std::string pathname);
  virtual ~CollateMonitorElement();

  // get # of MEs used in collation last time sum was calculated
  int getNumUsed(void){return numUsed_;}
      
  std::string getCName(void){return cname_;}

 private:
  
 protected:
  // false till 1st Monitoring Element has been added
  bool canUse_;
  // CME name, title, pathname
  std::string cname_, ctitle_, cpathname_;
  // # of MEs used in collation last time sum was calculated
  int numUsed_;
  
  // all search strings that form the collation ME
  std::set<std::string> searchStrings;

  // add Monitor Element to CollateMonitorElement's contents; return success flag 
  virtual bool addIt(MonitorElement * me, const std::string & pathname,  
		     const std::string & name) = 0;
  // map of MEs that form summary ME
  dqm::me_util::monit_map contents_;
  // calculate summary (to be called at end of monitoring cycle)
  virtual void summary(void) = 0;

  // add <search_string> to cme's contents; look for match in global_map
  void add(const std::string & search_string, 
	   const dqm::me_util::global_map & look_here);

  // add <search_string> to summary ME; 
  // <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
  // (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  virtual void add2search(const std::string & search_string);

  // look for all ME matching <search_string> in <look_here>; 
  // if found, add to contents_
  void scanContents(const std::string & search_string, const 
		    dqm::me_util::global_map & look_here);

  // come here when the 1st ME (component of sum) has been obtained
  virtual void createCollateBase(MonitorElement * me) = 0;

  friend class MonitorUserInterface;
  //  friend class MonitorUIRoot;
};



#endif









