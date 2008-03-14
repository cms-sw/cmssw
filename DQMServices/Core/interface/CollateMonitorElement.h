#ifndef CollateMonitorElement_h
#define CollateMonitorElement_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/StringUtil.h"

#include <set>
#include <string>
#include <vector>

class MonitorElementRootFolder;
class DaqMonitorBEInterface;

/** The base class for the collation of Monitor Elements
 */
class CollateMonitorElement : public StringUtil
{

 public:
  
  /// get # of MEs used in collation last time sum was calculated
  int getNumUsed(void){return numUsed_;}
     
  /// get Collate ME name
  std::string getCName(void){return cname_;}

  /// get MonitorElement that is obtained by summing the MonitorElements
  virtual MonitorElement * getMonitorElement() = 0;

 private:
  
 protected:

  CollateMonitorElement(const std::string & name, const std::string title, 
			const std::string pathname);
  virtual ~CollateMonitorElement();

  /// get hold of back-end interface instance
  DaqMonitorBEInterface * bei;
  /// false till 1st Monitoring Element has been added
  bool canUse_;
  /// CME name, title, pathname
  std::string cname_, ctitle_, cpathname_;
  /// # of MEs used in collation last time sum was calculated
  int numUsed_;
  
  /// all search strings and rules that form the collation ME
  dqm::me_util::searchCriteria rules;
  /// map of MEs that form summary ME
  dqm::me_util::rootDir contents_;

  /// add MEs to contents_
  void addME(std::vector<MonitorElement *> & allMEs);

  /// add Monitor Element to CollateMonitorElement's contents; return success flag 
  virtual bool addIt(MonitorElement * me) = 0;
  /// calculate summary (to be called at end of monitoring cycle)
  virtual void summary(void) = 0;

  /// add search_string to rules.search.search_path
  void add2search_path(const std::string & search_string, unsigned int tag);
  /// add pathname to rules.search.folders (flag=false) 
  /// or rules.search.foldersFull (flag=true)
  void add2folders(const std::string & pathname, bool useSubfolders, 
		   unsigned int tag);
  /// add tag to rules.tags
  void add2tags(unsigned int tag);
  /// add <search_string> to cme's contents; look for match in directory structure;
  /// if tag != 0, this applies to tagged contents
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
  /// this action applies to all MEs already available or future ones
  void add(unsigned int tag, const std::string & search_string, 
	   const dqm::me_util::rootDir & Dir);
  /// add directory contents to summary ME ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// if tag != 0, this applies to tagged contents
  /// use flag to specify whether subfolders (and their contents) should be included;
  /// this action applies to all MEs already available or future ones
  void add(unsigned tag, std::string & pathname, const dqm::me_util::rootDir
	   & Dir, bool useSubfolds);

  /// add tagged MEs to summary ME ==> FAST
  /// this action applies to all MEs already available or future ones
  void add(unsigned tag, const dqm::me_util::rootDir & Dir);

  /// add directory contents to put_here
  /// use flag to specify whether subfolders (and their contents) 
  /// should be included;
  void scanContents(std::string & pathname, bool useSubfolders,
		    const dqm::me_util::rootDir & Dir,
		    std::vector<MonitorElement *> & put_here);

  /// come here when the 1st ME (component of sum) has been obtained
  virtual void createCollateBase(MonitorElement * me) = 0;

  /// check if need to update collate-ME
  void checkAddedContents();

  friend class MonitorUserInterface;
  friend class DaqMonitorBEInterface;
  //  friend class MonitorUIRoot;
};



#endif









