#ifndef MonitorElementRootT_h
#define MonitorElementRootT_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Core/interface/QReport.h"

#include <string>
#include <vector>
#include <iostream>
#include <map>

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TH3F.h>
#include <TFolder.h>
#include <TObjString.h>

class QCriterion;

namespace edm {
  class DQMHttpSource;
}

namespace dqm
{
  namespace me_util
  {
    /// boolean flag for ME propertiess (key: ME name)
    typedef std::map<std::string, bool> ME_flag;
    typedef ME_flag::iterator bIt;
    
    struct neverSent_data_
    {
      ME_flag qr; /// key: QReport name, value: neverSent flag
      bool me;   /// neverSent flag for ME
      neverSent_data_(){me = true; qr.clear();}
    };
    typedef struct neverSent_data_ neverSent_data;
    
    /// key: ME_name, value: neverSent values for ME and QReports
    typedef std::map<std::string, neverSent_data> NeverSentData;
    typedef NeverSentData::iterator nsdIt;

  }
}

//
class MonitorElementRootObject : public MonitorElementT<TNamed>
{

 public:
  MonitorElementRootObject(TNamed *val, const std::string name) : 
  MonitorElementT<TNamed>(val,name) {}
  virtual ~MonitorElementRootObject(){}

  void Reset() {}

 private:
  void Fill(float x){}
  void Fill(float x, float y){}
  void Fill(float x, float y, float z){}
  void Fill(float x, float y, float z, float w){}

 protected:
  void copyFunctions(TH1 * copyFromThis, TH1 * copyToThis);
};

class FoldableMonitor
{
 public:
  FoldableMonitor(void){valobj = 0; tagObject = 0;}
  /// use this for saving scalars/strings into root-tuple
  TNamed *getRootObject(){return valobj;}
  /// use this for sending scalars/strings over TSocket
  TObjString * getTagObject(){return tagObject;}
  virtual ~FoldableMonitor()
  {if(valobj)delete valobj; if(tagObject)delete tagObject;}

  /// for folders: get pathname;
  /// (base class method to be overwritten for MonitorElementRootFolder class)
  ///  for other MEs: get pathname of parent folder
  std::string getPathname() const;

 protected:
  /// useful for saving scalars and strings into root-tuple
  TNamed *valobj;
  /// useful for shipping scalars and strings over TSocket
  TObjString * tagObject;
  void updateTagObject(const std::string & s)
  {if(tagObject)tagObject->SetString(s.c_str());}
};

class MonitorElementRootFolder : public MonitorElementRootObject
{

 private:
  /// objects (histograms, profiles, scalars) of directory
  dqm::me_util::ME_map objects_;

  /// whether ME has never been sent at least once (to given subscriber);
  /// true by default; to be used only for subscribers' directories 
  /// in SenderBase class
  dqm::me_util::NeverSentData neverSent;
  /// whether ME should be removed
  /// from subscription menu if removeElement is called;
  /// true by default; change values only via ReceiverBase::removeMonitorable,
  /// DaqMonitorROOTBackEnd::extractXXX (TH1F, TH2F, ...)
  dqm::me_util::ME_flag canDeleteFromMenu;
  /// whether ME has been requested; false by default; change value via
  /// method modifySubscription in ReceiverBase class
  dqm::me_util::ME_flag isDesired;
    
  /// subfolders of directory (key: subfodler name, not pathname)
  dqm::me_util::dir_map subfolds_;
  /// there is a difference between name (e.g. B3) and pathname (e.g. B1/B2/B3)
  std::string pathname_;
  /// verbose level
  unsigned verbose_;
  /// make new directory <name> and attach to directory "parent"
  MonitorElementRootFolder * makeDir(MonitorElementRootFolder * parent,
				     const std::string & name);

  /// true if inpath exists
  bool pathExists(std::string inpath) const;

  /// whether folder owns (and can delete) its children
  bool ownsChildren_;

  /// name of root folder
  /// (for Own: "root", for Subscribers: subscriber's name, for Tags: "Tag=<TagNo>")
  std::string rootFolderName_;
  /// set name of root folder
  void setRootName();
  /// get name of root folder
  std::string getRootName() const {return rootFolderName_;}
  
  friend class DaqMonitorROOTBackEnd;
  friend class SenderBase;
  friend class ReceiverBase;
  friend class DQMTagHelper;
  /// for fast access to objects_
  friend class MonitorUIRoot;
  friend class edm::DQMHttpSource;

  /// get (pointer to) last directory in inpath (create necessary subdirs)
  MonitorElementRootFolder * makeDirectory(std::string inpath);
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  MonitorElementRootFolder * getDirectory(std::string inpath) const;
    
  /// get parent directory
  MonitorElementRootFolder * getParent(void) {return parent_;}
  /// set parent directory
  void setParent(MonitorElementRootFolder * parent) 
  {parent_ = parent; setRootName();}

  /// add monitoring element to contents (could be folder)
  void addElement(MonitorElement * obj);
  /// add null monitoring element to contents (can NOT be folder)
  void addElement(const std::string name)
  {
    if(!findObject(name))
      objects_[name] = (MonitorElement *) 0;
    else
      std::cout << " *** Warning: MonitorElement " << name 
		<< " already defined in " << pathname_ << std::endl;
  }
  /// remove monitoring element from contents; return success flag
  /// if warning = true, print message if element does not exist
  bool removeElement(const std::string & name,  bool warning = true);

  /// same as above with pointer instead of name
  void removeElement(MonitorElement * me, const std::string & name);

  /// remove MonitorElement name from neverSent, isDesired, objects_ (if applicable)
  void removeMEName(const std::string & name);

  /// remove all monitoring elements in directory (not including subfolders)
  void removeContents();

  /// true if at least one of the folder objects has been updated
  bool wasUpdated(void) const;

  /// true if folder has no objects
  bool empty(void) const;

  /// (a) cleanup of folder's subfolder list 
  /// (this calls the clear method of folder's subfolders as well)
  /// (b) deletion of folder's objects 
  /// (c) removes "this" entry from parent's list of subfolders
  void clear();
  /// collect all subfolders of "this"
  void getSubfolders(std::vector<MonitorElementRootFolder*> & put_here);
  /// true if "this" (or any subfolder at any level below "this") contains
  /// at least one valid (i.e. non-null) monitoring element
  bool containsAnyMEs(void) const;
  /// true if "this" (or any subfolder at any level below "this") contains
  /// at least one monitorable element
  bool containsAnyMonitorable(void) const;

  MonitorElementRootFolder(TFolder *f, const std::string name);
  virtual ~MonitorElementRootFolder(void);

  /// update vector with all children of folder
  /// (does NOT include contents of subfolders)
  void getContents(std::vector<MonitorElement*> & put_here) const;

  /// update vector with all children of folder and all subfolders
  void getAllContents(std::vector<MonitorElement*> & put_here) const;
  

 public:
  /// get full pathname of folder (i.e. wrt root)
  std::string getPathname(void) const {return pathname_;}
  /// look for subfolder "name" in current dir; return pointer (null if not found)
  MonitorElementRootFolder * findFolder(std::string name) const;
  /// look for monitor element "name" in current dir; 
  /// return pointer (or null if not found)
  MonitorElement * findObject(std::string name) const;
  /// get folder children in string of the form <obj1>,<obj2>,<obj3>;
  /// (empty string for folder containing no objects)
  /// if skipNull = true, skip null monitoring elements
  std::string getChildren(bool skipNull) const;  
  /// true if ME <name> is included in folder's objects_
  /// (i.e. name exists, but could correspond to null pointer)
  bool hasMonitorable(const std::string & name) const
  {return objects_.find(name) != objects_.end();}
  /// set verbose level
  void setVerbose(unsigned level){verbose_ = level;}

  /// whether folder owns (and can delete) its children
  bool ownsChildren() const {return ownsChildren_;}

  /// true if at least of one of contents or subfolders gave hasWarning = true
  bool hasWarning(void) const;
  /// true if at least of one of contents or subfolders gave hasError = true
  bool hasError(void) const;
  /// true if at least of one of contents or subfolders gave hasOtherReport = true
  bool hasOtherReport(void) const;

  /// get status of folder (one of: STATUS_OK, WARNING, ERROR, OTHER);
  /// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  /// see Core/interface/QTestStatus.h for details on "OTHER" 
  int getStatus(void) const;

};


class MonitorElementRootInt : public MonitorElementInt, public FoldableMonitor
{
 public:
  MonitorElementRootInt(int *val, const std::string &name) : 
  MonitorElementInt(val,name)
  {
    valobj = new TNamed(name.c_str(),valueString().c_str());
    tagObject = new TObjString(tagString().c_str());
  }

  virtual ~MonitorElementRootInt(){}

  void Fill(float x)
  {
    LockMutex a(mutex);
    MonitorElementInt::Fill(x);
    this->MonitorElementInt::operator*()=(int)x;
    valobj->SetTitle(valueString().c_str());
    updateTagObject(tagString());
  }
 private:
  void Fill(float x, float w)
  {
    std::cerr << " *** Fill method for integers needs to be called\n"
	      << " with one argument! " << std::endl;
  }
  void Fill(float x, float y, float w) {Fill(x,w);}
  void Fill(float x, float y, float z, float w){Fill(x,w);}
};

class MonitorElementRootFloat : public MonitorElementFloat, 
  public FoldableMonitor
{
 public:
  MonitorElementRootFloat(float *val, const std::string &name) : 
  MonitorElementFloat(val,name)
  {
    valobj = new TNamed(name.c_str(),valueString().c_str());
    tagObject = new TObjString(tagString().c_str());
  }
  
  void Fill(float x)
  {
    LockMutex a(mutex);
    MonitorElementFloat::Fill(x);
    this->MonitorElementFloat::operator*()=x;
    valobj->SetTitle(valueString().c_str());
    updateTagObject(tagString());
  }

  virtual ~MonitorElementRootFloat(){}
 private:
  void Fill(float x, float w)
  {
    std::cerr << " *** Fill method for floats needs to be called\n"
	      << " with one argument! " << std::endl;
  }

  void Fill(float x, float y, float w) {Fill(x,w);}
  void Fill(float x, float y, float z, float w){Fill(x, w);}

};

class MonitorElementRootString : public MonitorElementString, 
  public FoldableMonitor
{
 public:
  MonitorElementRootString(std::string *val, const std::string &name) : 
  MonitorElementString(val,name)
  {
    valobj = new TNamed(name.c_str(),this->valueString().c_str());
    tagObject = new TObjString(tagString().c_str());
  }
  
  virtual ~MonitorElementRootString(){}
 private:

  void Fill(float x, float w){}
  void Fill(float x, float y, float w){}
  void Fill(float x, float y, float z, float w){}

};

/// class should be created (ctor) by DaqMonitorROOTBackEnd class 
/// and deleted (dtor) by MonitorElement class 
class MERootQReport : public MonitorElementRootString, public QReport
{
 public:
  /// redefine MonitorElementString method
  std::string valueString() const
  { std::ostringstream retval;
    retval << "qr=st." << getStatus() << "." << getValue(); 
    return retval.str();
  }

 protected:
  /// ctor #1
  MERootQReport(std::string *val, std::string ME_name, 
		std::string qtname) : 
  MonitorElementRootString(val, ME_name+"."+qtname), QReport(qtname) 
  {updateReport();}
  /// ctor #2
  MERootQReport(std::string *val, std::string ME_name, 
		std::string qtname, QCriterion * qc) : 
  MonitorElementRootString(val, ME_name+"."+qtname), QReport(qc)
  {updateReport();}
 private:
  ///
  virtual ~MERootQReport(){}

  /// method to be called after test has run (and status/message have been updated)
  void updateReport(void)
  {
    /// this is the (quality test) message 
    /// (this should also call MonitorElement::update)
    this->MonitorElementString::operator*()= getMessage();
    /// this is the updated valobj
    valobj->SetTitle(valueString().c_str());
    /// this is the updated tagObject
    updateTagObject(tagString());
  }
  void resetUpdate(){this->MonitorElementString::resetUpdate();}
  /// for calling the constructors
  friend class DaqMonitorROOTBackEnd;
  /// for calling the destructors
  friend class MonitorElement;
};

class MonitorElementRootH1 : public MonitorElementRootObject
{

 public:
  MonitorElementRootH1(TH1F *h, const std::string name) : 
  MonitorElementRootObject(h,name){}
  void Fill(float x, float w)
  {
    LockMutex a(mutex);
    ((TH1F*)operator->())->Fill(x,w);
  }
  void Fill(float x){Fill(x, 1);}
  
  void Reset() {
    ((TH1F*)operator->())->Reset();   
  }
  
  virtual ~MonitorElementRootH1(){}

  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  float getBinContent(int binx) const;
  float getBinError(int binx) const;
  float getEntries(void) const;
  int getNbinsX() const;
  int getNbinsY() const;
  int getNbinsZ() const;
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle() const;

  void setBinContent(int binx, float content);
  void setBinError(int binx, float error);
  void setEntries(float nentries);
  void setBinLabel(int bin, std::string label, int axis = 1);
  void setAxisRange(float min, float max, int axis = 1);
  void setAxisTitle(std::string axis_title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);

  void setTitle(std::string new_title);

 private:
  void Fill(float x, float y, float w)
  {
    std::cerr << " *** Fill method for 1D histograms needs to be called\n"
	      << " with one (x) or two (x, w) arguments! " << std::endl;
  }

  void Fill(float x, float y, float z, float w){Fill(x,y,w);}

  friend class DaqMonitorROOTBackEnd;

  void copyFrom(TH1F * just_in);
  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void softReset(void);
  /// get x-, y-, or z-axis
  TAxis * getAxis(int axis) const
  {
    if(axis == 1)
      return ( (TH1F *) this->const_ptr() )->GetXaxis();
    else if (axis == 2)
      return ( (TH1F *) this->const_ptr() )->GetYaxis();
    else
      return 0;
  }
  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  void unresetContents(void);
};

class MonitorElementRootH2 : public MonitorElementRootObject
{
  
 public:
  MonitorElementRootH2(TH2F *h, const std::string name) : 
  MonitorElementRootObject(h,name){}
  void Fill(float x, float y, float w)
  {
    LockMutex a(mutex);
    ((TH2F*)operator->())->Fill(x,y,w);
  }
  void Fill(float x, float y){Fill(x,y,1);}
  
  void Reset() {
    ((TH2F*)operator->())->Reset();   
  }
  
  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  float getBinContent(int binx, int biny) const;
  float getBinError(int binx, int biny) const;
  float getEntries(void) const;
  int getNbinsX() const;
  int getNbinsY() const;
  int getNbinsZ() const;
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle() const;

  void setBinContent(int binx, int biny, float content);
  void setBinError(int binx, int biny, float error);
  void setEntries(float nentries);
  void setBinLabel(int bin, std::string label, int axis = 1);
  void setAxisRange(float min, float max, int axis = 1);
  void setAxisTitle(std::string axis_title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);
  void setTitle(std::string new_title);

  virtual ~MonitorElementRootH2(){}

 private:
  void Fill(float x, float y, float z, float w)
  {std::cerr << " *** Fill method for 2D histogram needs to be called\n"
	     << " with two (x,y) or three (x,y,w) arguments! " << std::endl;
  }
  void Fill(float x){Fill(x, 0, 0, 0);}

  friend class DaqMonitorROOTBackEnd;
  void copyFrom(TH2F * just_in);
  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void softReset(void);
  /// get x-, y-, or z-axis
  TAxis * getAxis(int axis) const
  {
    if(axis == 1)
      return ( (TH2F *) this->const_ptr() )->GetXaxis();
    else if(axis == 2)
      return ( (TH2F *) this->const_ptr() )->GetYaxis();
    else
      return 0;
  }
  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  void unresetContents(void);
};

class MonitorElementRootProf : public MonitorElementRootObject
{
  
 public:
  MonitorElementRootProf(TProfile *h, const std::string name) : 
  MonitorElementRootObject(h,name){}
  void Fill(float x, float y, float w)
  {
    LockMutex a(mutex);
    ((TProfile*)operator->())->Fill(x,y,w);
  }
  void Fill(float x, float y){Fill(x, y, 1);}
  
  void Reset() {
    ((TProfile*)operator->())->Reset();   
  }

  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  float getBinContent(int binx) const;
  float getBinError(int binx) const;
  float getEntries(void) const;
  float getBinEntries(int bin) const;
  float getYmin(void) const;
  float getYmax(void) const;
  int getNbinsX() const;
  int getNbinsY() const;
  int getNbinsZ() const;
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle() const;

  void setBinContent(int binx, float content);
  void setBinError(int binx, float error);
  void setEntries(float nentries);
  void setBinLabel(int bin, std::string label, int axis = 1);
  void setAxisRange(float min, float max, int axis = 1);
  void setAxisTitle(std::string axis_title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);
  void setTitle(std::string new_title);
  void setBinEntries(int binx, float nentries);

  virtual ~MonitorElementRootProf(){}

 private:
  void Fill(float x, float y, float z, float w)
  {std::cerr << " *** Fill method for profiles needs to be called\n"
	     << " with two (x,y) or three (x,y,w) arguments! " << std::endl;
  }
  void Fill(float x){Fill(x, 0, 0, 0);}
 
  friend class DaqMonitorROOTBackEnd;
  void copyFrom(TProfile * just_in);
  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void softReset(void);
  /// get x-, y-, or z-axis
  TAxis * getAxis(int axis) const
  {
    if(axis == 1)
      return ( (TProfile *) this->const_ptr() )->GetXaxis();
    else if(axis == 2)
      return ( (TProfile *) this->const_ptr() )->GetYaxis();
    else
      return 0;
  }

  /// used for subtracting reference from received profile;
  /// Can be called with sum = h1 or sum = h2
  void addProfiles(TProfile * h1, TProfile * h2, TProfile * sum, 
		   float c1, float c2);
  ///
  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  void unresetContents(void);
};

class MonitorElementRootProf2D : public MonitorElementRootObject
{
  
 public:
  MonitorElementRootProf2D(TProfile2D *h, const std::string name) : 
    MonitorElementRootObject(h,name){}
  void Fill(float x, float y, float z, float w)
  {
    LockMutex a(mutex);
    ((TProfile2D*)operator->())->Fill(x,y,z,w);
  }
  
  void Fill(float x, float y, float z){Fill(x, y, z, 1);}
  
  void Reset() {
    ((TProfile2D*)operator->())->Reset();   
  }

  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  float getBinContent(int binx, int biny) const;
  float getBinError(int binx, int biny) const;
  float getEntries(void) const;
  float getBinEntries(int bin) const;
  int getNbinsX() const;
  int getNbinsY() const;
  int getNbinsZ() const;
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle() const;

  void setBinContent(int binx, int biny, float content);
  void setBinError(int binx, int biny, float error);
  void setEntries(float nentries);
  void setBinLabel(int bin, std::string label, int axis = 1);
  void setAxisRange(float min, float max, int axis = 1);
  void setAxisTitle(std::string axis_title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);
  void setTitle(std::string new_title);
  void setBinEntries(int binx, float nentries);

  virtual ~MonitorElementRootProf2D(){}

 private:
  void Fill(float x, float y)
  {std::cerr << " *** Fill method for 2-D profiles needs to be called\n"
	     << " with three (x,y,z) or four (x,y,z,w) arguments! " 
	     << std::endl;
  }
  void Fill(float x){Fill(x, 0);}
 
  friend class DaqMonitorROOTBackEnd;
  void copyFrom(TProfile2D * just_in);
  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void softReset(void);
  /// get x-, y-, or z-axis
  TAxis * getAxis(int axis) const
  {
    if(axis == 1)
      return ( (TProfile2D *) this->const_ptr() )->GetXaxis();
    else if(axis == 2)
      return ( (TProfile2D *) this->const_ptr() )->GetYaxis();
    else if(axis == 3)
      return ( (TProfile2D *) this->const_ptr() )->GetZaxis();
    else
      return 0;
  }
  /// used for subtracting reference from received profile;
  /// Can be called with sum = h1 or sum = h2
  void addProfiles(TProfile2D * h1, TProfile2D * h2, TProfile2D * sum, 
	      float c1, float c2);
  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  void unresetContents(void);
  ///
};

class MonitorElementRootH3 :  public MonitorElementRootObject
{

 public:
  MonitorElementRootH3(TH3F *h, const std::string name) : 
  MonitorElementRootObject(h,name){}
  void Fill(float x, float y, float z, float w)
  {
    LockMutex a(mutex);
    ((TH3F*)operator->())->Fill(x,y,z,w);
    
  }
  void Fill(float x, float y, float z){Fill(x,y,z,1);}
  void Reset() {
    ((TH3F*)operator->())->Reset();   
  }
  
  float getMean(int axis = 1) const;
  float getMeanError(int axis = 1) const;
  float getRMS(int axis = 1) const;
  float getRMSError(int axis = 1) const;
  float getBinContent(int binx, int biny, int binz) const;
  float getBinError(int binx, int biny, int binz) const;
  float getEntries(void) const;
  int getNbinsX() const;
  int getNbinsY() const;
  int getNbinsZ() const;
  std::string getAxisTitle(int axis = 1) const;
  std::string getTitle() const;

  void setBinContent(int binx, int biny, int binz, float content);
  void setBinError(int binx, int biny, int binz, float error);
  void setEntries(float nentries);
  void setBinLabel(int bin, std::string label, int axis = 1);
  void setAxisRange(float min, float max, int axis = 1);
  void setAxisTitle(std::string axis_title, int axis = 1);
  void setAxisTimeDisplay(int value, int axis = 1);
  void setAxisTimeFormat(const char *format = "", int axis = 1);
  void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1);
  void setTitle(std::string new_title);

  virtual ~MonitorElementRootH3(){}

 private:
  void Fill(float x, float y)
  {std::cerr << " *** Fill method for 3D histograms needs to be called\n"
	     << " with three (x,y,z) or four (x,y,z,w) arguments! " 
	     << std::endl;
  }
  void Fill(float x){Fill(x, 0);}

  friend class DaqMonitorROOTBackEnd;
  void copyFrom(TH3F * just_in);

  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void softReset(void);
  /// get x-, y-, or z-axis
  TAxis * getAxis(int axis) const
  {
    if(axis == 1)
      return ( (TH3F *) this->const_ptr() )->GetXaxis();
    else if(axis == 2)
      return ( (TH3F *) this->const_ptr() )->GetYaxis();
    else if(axis == 3)
      return ( (TH3F *) this->const_ptr() )->GetZaxis();
    else
      return 0;
  }
  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  void unresetContents(void);
};

#endif









