#ifndef _DQM_DEFINITIONS_H_
#define _DQM_DEFINITIONS_H_

#include <vector>
#include <map>
#include <set>
#include <list>
#include <string>

class MonitorElement;

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

    // key: ME name, value: monitoring element address
    typedef std::map<std::string, MonitorElement *> ME_map;
    typedef ME_map::iterator ME_it;
    typedef ME_map::const_iterator cME_it;
    
    // key: folder name, value: folder address
    typedef std::map<std::string, MonitorElement *> dir_map;
    typedef dir_map::iterator dir_it;
    typedef dir_map::const_iterator cdir_it;
    // key: folder pathname,value: folder address 
    typedef std::map<std::string, MonitorElement *> fulldir_map;
    typedef fulldir_map::iterator fdir_it;
    typedef fulldir_map::const_iterator cfdir_it;
    // key: folder pathname, value: address of folder's map of objects 
    typedef std::map<std::string, const ME_map *> global_map;
    typedef global_map::iterator glob_it;
    typedef global_map::const_iterator cglob_it;
    // key: folder pathname, value: set of ME names of objects 
    // (subfolders not included)
    typedef std::map<std::string, std::set<std::string> > monit_map;
    typedef std::map<std::string, std::set<std::string> >::iterator 
      monit_it;
    typedef std::map<std::string, std::set<std::string> >::const_iterator 
      cmonit_it;
    
    // structure holding folder hierarchy w/ ME
    struct MonitorStruct_ {
      // map of folder hierarchy; 
      // key: folder pathname, value: folder address
      dir_map folders_;
      // key: folder pathname, value: folder's contents address 
      // (subfolders not included)
      global_map global_;
      // default constructor
      MonitorStruct_() {folders_.clear(); global_.clear();}
    };
    typedef struct MonitorStruct_ MonitorStruct;

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


  } // namespace me_util

} // namespace dqm

#endif
