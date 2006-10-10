#ifndef _DQM_DEFINITIONS_H_
#define _DQM_DEFINITIONS_H_

#include <vector>
#include <map>
#include <set>
#include <list>
#include <string>

class MonitorElement;
class MonitorElementRootFolder;
class CollateMonitorElement;

namespace dqm
{
  namespace me_util
  {
    // define some "nicknames" here
    
    typedef std::vector<std::string>::iterator vIt;
    typedef std::vector<std::string>::const_iterator cvIt;
    typedef std::list<std::string>::iterator lIt;
    typedef std::set<std::string>::iterator sIt;
    typedef std::set<std::string>::const_iterator csIt;
    typedef std::set<MonitorElement *>::iterator meIt;
    typedef std::vector<MonitorElement *> vME;
    typedef vME::iterator vMEIt;
    typedef vME::const_iterator vMEcIt;
    // key: ME name, value: monitoring element address
    typedef std::map<std::string, MonitorElement *> ME_map;
    typedef ME_map::iterator ME_it;
    typedef ME_map::const_iterator cME_it;
    
    // key: folder pathname, value: folder address
    typedef std::map<std::string, MonitorElementRootFolder *> dir_map;
    typedef dir_map::iterator dir_it;
    typedef dir_map::const_iterator cdir_it;
    // key: folder pathname, value: set of ME names of objects 
    // (subfolders not included)
    typedef std::map<std::string, std::set<std::string> > monit_map;
    typedef std::map<std::string, std::set<std::string> >::iterator 
      monit_it;
    typedef std::map<std::string, std::set<std::string> >::const_iterator 
      cmonit_it;

    struct rootDir_ {
      // map with pathnmames containing MEs
      dir_map paths;
      // top directory: everything hangs down from here
      MonitorElementRootFolder * top;
      rootDir_(){top = 0;}
    };
    typedef struct rootDir_ rootDir;

    // key: subscribers name, value: directory structure
    typedef std::map<std::string, rootDir> subscriber_map;
    typedef subscriber_map::iterator sdir_it;
    typedef subscriber_map::const_iterator csdir_it;
    // key: tag (eg. detector-ID), value: directory structure
    typedef std::map<unsigned int, rootDir> tag_map;
    typedef tag_map::iterator tdir_it;
    typedef tag_map::const_iterator ctdir_it;
     // key: ME name, value: set with tags
    typedef std::map<std::string, std::set<unsigned int> > ME_tags;
    typedef ME_tags::iterator tags_it;
    typedef ME_tags::const_iterator ctags_it;
    // key: folder pathname, value: set w/ ME names and its tags
    typedef std::map<std::string, ME_tags> dir_tags;
    typedef dir_tags::iterator dirt_it;
    typedef dir_tags::const_iterator cdirt_it;

    // key: ME of summary, value: CME
    // (note: can use CollateMonitorElement::getMonitorElement() 
    // for reverse linking)
    typedef std::map<MonitorElement *, CollateMonitorElement *> cme_map;
    typedef cme_map::iterator cmemIt;

    typedef std::set<CollateMonitorElement *> cme_set;
    typedef cme_set::iterator cmesIt;

    // pathname for root folder
    static const std::string ROOT_PATHNAME = ".";


    struct Channel_ {
      int binx; // bin # in x-axis (or bin # for 1D histogram)
      int biny; // bin # in y-axis (for 2D or 3D histograms)
      int binz; // bin # in z-axis (for 3D histograms)
      float content; // bin content
      float RMS; // RMS of bin content

      int getBin(){return getBinX();}
      int getBinX(){return binx;}
      int getBinY(){return biny;}
      int getBinZ(){return binz;}
      float getContents(){return content;}
      float getRMS(){return RMS;}
      Channel_(int bx, int by, int bz, float data, float rms)
      {binx = bx; biny = by; binz = bz; content = data; RMS = rms;}
      
      Channel_() {Channel_(0, 0, 0, 0, 0);}
    };
    typedef struct Channel_ Channel;

    // search criterion; to be used by CollateMonitorElement and QualityTest
    struct searchCriterion_ {
      std::vector<std::string> search_path; // exact pathname, or w/ wildcards
      std::vector<std::string> folders; // folder contents (no subfolders)
      std::vector<std::string> foldersFull; // folder contents (w/ subfolders)
    };
    typedef struct searchCriterion_ searchCriterion;

    typedef std::map<unsigned int, searchCriterion> search_map;
    typedef search_map::iterator sMapIt;
    typedef search_map::const_iterator csMapIt;

    struct searchCriteria_{
      // key: tag (use =0 for no tags), value: all search criteria for tag
      search_map search;
      // vector of tags (no search strings or folders)
      std::vector<unsigned int> tags;
      // add search_string to search.search_path
      void add2search_path(const std::string & search_string, unsigned int tag)
      {
	sMapIt it = search.find(tag);
	if(it == search.end())
	  {
	    searchCriterion tmp; tmp.search_path.push_back(search_string);
	    search[tag] = tmp;
	  }
	else
	  it->second.search_path.push_back(search_string);  
      }
      // add pathname to search.folders (flag=false) 
      // or search.foldersFull (flag=true)
      void add2folders(const std::string & pathname, bool useSubfolders, 
		       unsigned int tag)
      {
	sMapIt it = search.find(tag);
	if(it == search.end())
	  {
	    searchCriterion tmp; 
	    if(useSubfolders)
	      tmp.foldersFull.push_back(pathname);
	    else
	      tmp.folders.push_back(pathname);
	    search[tag] = tmp;
	  }
	else
	  {
	    if(useSubfolders)
	      it->second.foldersFull.push_back(pathname);
	    else
	      it->second.folders.push_back(pathname);
	  }
      }
      // add tag to tags member
      void add2tags(unsigned int tag)
      {
	tags.push_back(tag);
      }

    };
      typedef struct searchCriteria_ searchCriteria;

  } // namespace me_util

} // namespace dqm

#endif
