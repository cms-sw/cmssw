#ifndef MonitorElement_h
#define MonitorElement_h


#include <sys/time.h>
#include <string>
#include <vector>
#include <set>
#include <list>
#include <map>

#include "Utilities/Threads/interface/ThreadUtils.h"

#include "DQMServices/Core/interface/QReport.h"

/** The base class for all MonitorElements (ME)
 */
class MonitorElement
{

 public:
  
  MonitorElement();
  MonitorElement(const char*);
  virtual ~MonitorElement();

  // true if ME was updated in last monitoring cycle
  bool wasUpdated() const;
  // specify whether ME should be reset at end of monitoring cycle (default: false);
  // (typically called by Sources that control the original ME)
  void setResetMe(bool flag);
  // get name of ME
  virtual std::string getName() const  = 0;
  // "Fill" ME methods:
  // can be used with 1D histograms or scalars
  virtual void Fill(float)=0;
  // can be used with 2D (x,y) or 1D (x, w) histograms
  virtual void Fill(float, float)=0;
  // can be used with 3D (x, y, z) or 2D (x, y, w) histograms
  virtual void Fill(float, float, float)=0;
  // can be used with 3D (x, y, z, w) histograms
  virtual void Fill(float, float, float, float)=0;
  // returns value of ME in string format (eg. "f = 3.14151926" for float numbers);
  // relevant only for scalar or string MEs
  virtual std::string valueString() const=0;
  // return tagged value of ME in string format 
  // (eg. <name>f=3.14151926</name> for float numbers);
  // relevant only for sending scalar or string MEs over TSocket
  std::string tagString() const
  {return "<" + getName() + ">" + valueString() + "</" + getName() + ">";}
  // true if ME is a folder
  bool isFolder(void) const;
  // opposite of isFolder method
  bool isNotFolder(void) const;

  // get QReport corresponding to <qtname> (null pointer if QReport does not exist)
  const QReport * getQReport(std::string qtname) const;

  // get map of QReports
  dqm::qtests::QR_map getQReports(void) const {return qreports_;}
  // get warnings from last set of quality tests
  std::vector<QReport *> getQWarnings(void) const {return qwarnings_;}
  // get errors from last set of quality tests
  std::vector<QReport *> getQErrors(void) const {return qerrors_;}

  // run all quality tests
  void runQTests(void);

  LockMutex::Mutex mutex;

 private:
  
 protected:

  void update();
  // reset ME (ie. contents, errors, etc)
  virtual void Reset()=0;

  // ------------ Operations for MEs that are normally never reset ---------

  // reset contents (does not erase contents permanently)
  // (makes copy of current contents; will be subtracted from future contents)
  virtual void softReset(void){}
  // if true: will subtract contents copied at "soft-reset" from now on
  // if false: will NO longer subtract contents (default)
  virtual void enableSoftReset(bool flag){};

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  // if true, will accumulate ME contents (over many periods)
  // until method is called with flag = false again
  void setAccumulate(bool flag);

  // whether soft-reset is enabled
  bool softReset_on; // default: false
  // whether ME contents should be accumulated over multiple monitoring periods
  bool accumulate_on; // default: false

  // true if ME should be reset at end of monitoring cycle
  bool resetMe(void) const;
  // reset "was updated" flag
  void resetUpdate();
  // add quality report (to be called by DaqMonitorROOTBackEnd)
  void addQReport(QReport * qr);
  // true if QReport with name <qtname> already exists
  bool qreportExists(std::string qtname) const;

  class manage{
  public:
    manage() : variedSince(true), folder_flag(false),
      resetMe(false){}
    // has content changed?
    bool variedSince; 
    // is this a folder? (if not, it's a monitoring object)
    bool folder_flag;
    // should contents be reset at end of monitoring cycle?
    bool resetMe;
    // creation time
    timeval time_stamp;
  };

  manage man;

  dqm::qtests::QR_map qreports_;

  std::vector<QReport *> qwarnings_;
  std::vector<QReport *> qerrors_;

  friend class DaqMonitorBEInterface;
  friend class CollateMET;

};

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

  } // namespace me_util

} // namespace dqm

#endif









