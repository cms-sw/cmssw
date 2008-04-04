#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>

using namespace dqm::me_util;

using std::string; using std::vector;

CollateMonitorElement::CollateMonitorElement(const string & name, 
					     const string title, 
					     const string pathname)
{
  rules.search.clear(); rules.tags.clear();
  canUse_ = false;
  numUsed_ = 0; cname_ = name; ctitle_ = title; cpathname_ = pathname;
}


CollateMonitorElement::~CollateMonitorElement() 
{
}

// add directory contents to put_here
// use flag to specify whether subfolders (and their contents) 
// should be included;
void CollateMonitorElement::scanContents(string & pathname, 
					 bool useSubfolders, const rootDir & Dir,
					 vector<MonitorElement *> & put_here)
{
  if(useSubfolders)
    bei->getAllContents(pathname, Dir, put_here);
  else
    bei->getContents(pathname, Dir, put_here);
}

// add <search_string> to cme's contents; look for match in directory structure;
// if tag != 0, this applies to tagged contents
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
void CollateMonitorElement::add(unsigned int tag, const string & search_string, 
				const rootDir & Dir)
{
  add2search_path(search_string, tag);
  vME allMEs;
  bei->scanContents(search_string, Dir, allMEs);
  addME(allMEs);
}

// add directory contents to summary ME ==> FAST
// (need exact pathname without wildcards, e.g. A/B/C);
// if tag != 0, this applies to tagged contents
// use flag to specify whether subfolders (and their contents) should be included;
// this action applies to all MEs already available or future ones
void CollateMonitorElement::add(unsigned int tag, string & pathname, 
				const rootDir & Dir, bool useSubfolds)
{
  add2folders(pathname, useSubfolds, tag);
  vME allMEs;
  scanContents(pathname, useSubfolds, Dir, allMEs);
  addME(allMEs);
}

// add tagged MEs to summary ME ==> FAST
// this action applies to all MEs already available or future ones
void CollateMonitorElement::add(unsigned int tag, const rootDir & Dir)
{
  add2tags(tag);
  vME allMEs;
  bei->get(Dir.paths, allMEs);
  addME(allMEs);
}

// add MEs to contents_
void CollateMonitorElement::addME(vector<MonitorElement *> & allMEs)
{
  for(vMEIt it = allMEs.begin(); it != allMEs.end(); ++it)
    {
      if(addIt(*it))
	{
	  // check if we need to define histogram/profile
	  if(!canUse_)
	    createCollateBase(*it);
	}
    }
}

// add search_string to rules.search.search_path
void CollateMonitorElement::add2search_path(const string & search_string, 
					    unsigned int tag)
{
  rules.add2search_path(search_string, tag);
}

// add pathname to rules.search.folders (flag=false) 
// or rules.search.foldersFull (flag=true)
void CollateMonitorElement::add2folders(const string & pathname, 
					bool useSubfolders, unsigned int tag)
{
  rules.add2folders(pathname, useSubfolders, tag);
}

// add tag to rules.tags
void CollateMonitorElement::add2tags(unsigned int tag)
{
  rules.add2tags(tag);
}

// check if need to update collate-ME
void CollateMonitorElement::checkAddedContents()
{
  vME allMEs;
  bei->checkAddedContents(rules, allMEs);
  addME(allMEs);
}
