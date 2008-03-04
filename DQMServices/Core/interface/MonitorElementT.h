#ifndef MonitorElementT_h
#define MonitorElementT_h

#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

template<class T>
class MonitorElementT : public MonitorElement
{
 public:
  MonitorElementT(T *val, const std::string name="") : 
  name_(name), val_(val) 
  {reference_ = 0;}
  virtual ~MonitorElementT() 
  {
    delete val_;
    if(reference_)deleteReference();
  }

  void clear(){} 
  /// pointer to val_
  T * operator->(){
    update();
    return val_;
  }
  /// const pointer to val_
  const T * const_ptr() const
  {
    return (const T *) val_;
  }

  /// return *val_ by reference
  T & operator*()
    {
      update();
      return *val_;
    }
  /// return *val_ by value
  T operator*() const {return *val_;}
  
  /// noarg functor (do we need this?)
  T operator()(){return *val_;}

  /// explicit cast overload
  operator T(){return *val_;}

  virtual std::string getName() const {return name_;}
  virtual T getValue() const {return *val_;}

  virtual void Reset()=0;

  virtual std::string valueString() const {return std::string();};

  /// whether soft-reset is enabled
  bool isSoftResetEnabled(void) const{return reference_ != 0;}
  
  float doNotUseMethod(std::string method) const
  {
    std::cerr << " *** Cannot use method " << method << 
      " with MonitorElement " << getName() << std::endl;
    return -999;
  }

  /// get mean value of histogram along x, y or z axis (axis=1, 2, 3 respectively)
  virtual float getMean(int axis = 1) const
  {return doNotUseMethod("getMean");}
  /// get mean value uncertainty of histogram along x, y or z axis 
  /// get (axis=1, 2, 3 respectively)
  virtual float getMeanError(int axis = 1) const
  {return doNotUseMethod("getMeanError");}
  /// get RMS of histogram along x, y or z axis (axis=1, 2, 3 respectively)
  virtual float getRMS(int axis = 1) const
  {return doNotUseMethod("getRMS");}
  /// get RMS uncertainty of histogram along x, y or z axis(axis=1,2,3 respectively)
  virtual float getRMSError(int axis = 1) const
  {return doNotUseMethod("getRMSError");}
  /// get # of bins in X-axis
  virtual int getNbinsX() const
  {return int(doNotUseMethod("getNbinsX"));}
  /// get # of bins in Y-axis
  virtual int getNbinsY() const
  {return int(doNotUseMethod("getNbinsY"));}
  /// get # of bins in Z-axis
  virtual int getNbinsZ() const
  {return int(doNotUseMethod("getNbinsZ"));}
  /// get content of bin (1-D)
  virtual float getBinContent(int binx) const
  {return doNotUseMethod("getBinContent(binx)");}
  /// get content of bin (2-D)
  virtual float getBinContent(int binx, int biny) const
  {return doNotUseMethod("getBinContent(binx,biny)");}
  /// get content of bin (3-D)
  virtual float getBinContent(int binx, int biny, int binz) const
  {return doNotUseMethod("getBinContent(binx,biny,binz)");}
  /// get uncertainty on content of bin (1-D) - See TH1::GetBinError for details
  virtual float getBinError(int binx) const
  {return doNotUseMethod("getBinError(binx)");}
  /// get uncertainty on content of bin (2-D) - See TH1::GetBinError for details
  virtual float getBinError(int binx, int biny) const
  {return doNotUseMethod("getBinError(binx,biny)");}
  /// get uncertainty on content of bin (3-D) - See TH1::GetBinError for details
  virtual float getBinError(int binx, int biny, int binz) const
  {return doNotUseMethod("getBinError(binx,biny,binz)");}
  /// get # of entries
  virtual float getEntries(void) const {return 1;}
  /// get # of bin entries (for profiles)
  virtual float getBinEntries(int bin) const
  {return doNotUseMethod("getBinEntries");}
  /// get min Y value (for profiles)
  virtual float getYmin(void) const
  {return doNotUseMethod("getYmin");}
  /// get max Y value (for profiles)
  virtual float getYmax(void) const 
  {return doNotUseMethod("getXmin");}
  /// get x-, y- or z-axis title (axis=1, 2, 3 respectively)
  virtual std::string getAxisTitle(int axis = 1) const 
  {doNotUseMethod("getAxisTitle"); return std::string("");}
  /// get histogram/profile title
  virtual std::string getTitle() const 
  {doNotUseMethod("getTitle"); return std::string("");}

  /// set content of bin (1-D)
  virtual void setBinContent(int binx, float content)
  {doNotUseMethod("setBinContent(binx,content)");}
  /// set content of bin (2-D)
  virtual void setBinContent(int binx, int biny, float content)
  {doNotUseMethod("setBinContent(binx,biny,content)");}
  /// set content of bin (3-D)
  virtual void setBinContent(int binx, int biny, int binz, float content)
  {doNotUseMethod("setBinContent(binx,biny,binz,content)");}
  /// set uncertainty on content of bin (1-D)
  virtual void setBinError(int binx, float error)
  {doNotUseMethod("setBinError(binx,error)");}
  /// set uncertainty on content of bin (2-D)
  virtual void setBinError(int binx, int biny, float error)
  {doNotUseMethod("setBinError(binx,biny,error)");}
  /// set uncertainty on content of bin (3-D)
  virtual void setBinError(int binx, int biny, int binz, float error)
  {doNotUseMethod("setBinError(binx,biny,binz,error)");}
  /// set # of entries
  virtual void setEntries(float nentries){}
  /// set bin label for x, y or z axis (axis=1, 2, 3 respectively)
  virtual void setBinLabel(int bin, std::string label, int axis = 1)
  {doNotUseMethod("setBinLabel");}
  /// set x-, y- or z-axis range (axis=1, 2, 3 respectively)
  virtual void setAxisRange(float xmin, float xmax, int axis = 1)
  {doNotUseMethod("setAxisRange");}
  /// set x-, y- or z-axis title (axis=1, 2, 3 respectively)
  virtual void setAxisTitle(std::string axis_title, int axis = 1)
  {doNotUseMethod("setAxisTitle");}
  /// set x-, y-, or z-axis to display time values
  virtual void setAxisTimeDisplay(int value, int axis = 1)
  {doNotUseMethod("setAxisTimeDisplay");}
  /// set the format of the time values that are displayed on an axis
  virtual void setAxisTimeFormat(const char *format = "", int axis = 1)
  {doNotUseMethod("setAxisTimeFormat");}
  /// set the time offset, if option = "gmt" then the offset is treated as a GMT time
  virtual void setAxisTimeOffset(double toffset, const char *option="local", int axis = 1)
  {doNotUseMethod("setAxisTimeOffset");}
  /// set (ie. change) histogram/profile title
  virtual void setTitle(std::string new_title)
  {doNotUseMethod("setTitle");}
  /// set # of bin entries (to be used for profiles)
  virtual void setBinEntries(int bin, float nentries) 
  {doNotUseMethod("setBinEntries");}

 private:
  
  std::string name_;
  T * val_;
 protected:

  /// make sure axis is one of 1 (x), 2 (y) or 3 (z)
  bool checkAxis(int axis) const
  {
    if (axis < 1 || axis > 3) return false;
    return true;
  }

  T * reference_; /// this is set to "val_" upon a "softReset"

  /// delete reference_
  void deleteReference(void)
  {
    if(!reference_)
      {
	std::cerr << " *** Cannot delete null reference for " 
		  << getName() << std::endl;
	return;
      }
    delete reference_;
    reference_ = 0;
  }

  /// adds reference_ back into val_ contents (ie. reverts action of softReset)
  virtual void unresetContents(void){}

  /// for description: see DQMServices/Core/interface/MonitorElement.h
  void disableSoftReset(void)
  {
    if(isSoftResetEnabled())
      {
	std::cout << " \"soft-reset\" option has been disabled for " 
		  << getName() << std::endl;

	unresetContents();
	deleteReference();
      }
  }

  // this is really bad; unfortunately, gcc 3.2.3 won't let me define 
  // template classes, so I have to find a workaround for now
  // error: "...is not a template type" - christos May26, 2005
  friend class CollateMERootH1;
  friend class CollateMERootH2;
  friend class CollateMERootH3;
  friend class CollateMERootProf;
  friend class CollateMERootProf2D;

};
#endif









