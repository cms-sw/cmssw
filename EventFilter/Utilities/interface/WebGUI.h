#ifndef WEBGUI_H
#define WEBGUI_H 1


#include "EventFilter/Utilities/interface/EPStateMachine.h"
#include "EventFilter/Utilities/interface/Css.h"

#include "xdaq/include/xdaq/Application.h"

#include "toolbox/include/toolbox/lang/Class.h"

#include "xdata/include/xdata/Serializable.h"
#include "xdata/include/xdata/UnsignedInteger32.h"

#include "xgi/include/xgi/Method.h"
#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"
#include "xgi/include/xgi/exception/Exception.h"

#include <string>
#include <vector>


namespace evf {
  
  class WebGUI : public toolbox::lang::Class
  {
  public:
    //
    // typedefs
    //
    typedef xdata::Serializable                             Param_t;
    typedef xdata::UnsignedInteger32                        Counter_t;
    typedef std::vector<std::pair<std::string,Param_t*> >   ParamVec_t;
    typedef std::vector<std::pair<std::string,Counter_t*> > CounterVec_t;
    typedef xgi::Input                                      Input_t;
    typedef xgi::Output                                     Output_t;
    typedef const std::string                               CString_t;
    typedef xgi::exception::Exception                       XgiException_t;
    
    
    //
    // construction/destruction
    //
    WebGUI(xdaq::Application* app,EPStateMachine* fsm);
    virtual ~WebGUI();
    
    
    //
    // public memeber functions
    //
    void defaultWebPage(Input_t *in, Output_t *out) throw (XgiException_t);
    void debugWebPage(Input_t *in, Output_t *out)   throw (XgiException_t);
    void css(Input_t *in,Output_t *out)             throw (XgiException_t);
    
    void addStandardParam(CString_t& name,Param_t* param);
    void addMonitorParam(CString_t& name,Param_t* param);
    void addDebugParam(CString_t& name,Param_t* param);
    
    void addStandardCounter(CString_t& name,Counter_t* counter);
    void addMonitorCounter(CString_t& name,Counter_t* counter);
    void addDebugCounter(CString_t& name,Counter_t* counter);
    
    void exportParameters(); // must be called once after registering all params!
    void resetCounters();
    
    void setLargeAppIcon(CString_t& icon) { largeAppIcon_=icon; }
    void setSmallAppIcon(CString_t& icon) { smallAppIcon_=icon; }
    void setSmallDbgIcon(CString_t& icon) { smallDbgIcon_=icon; }
    void setHyperDAQIcon(CString_t& icon) { hyperDAQIcon_=icon; }

    xdata::InfoSpace* appInfoSpace() { return appInfoSpace_; }
    xdata::InfoSpace* monInfoSpace() { return monInfoSpace_; }
    

  private:
    //
    // private member functions
    //
    void addParamsToInfoSpace(const ParamVec_t& params,xdata::InfoSpace* infoSpace);
    void addCountersToParams();
    
    void htmlTable(Input_t*in,Output_t*out,
		   CString_t& title,const ParamVec_t& params);
    void htmlHead(Input_t*in,Output_t*out,
		  CString_t& pageTitle);
    void htmlHeadline(Input_t*in,Output_t*out,
		      CString_t& link,CString_t& icon);
    
    
  private:
    //
    // member data
    //
    xdaq::Application *app_;
    EPStateMachine    *fsm_;
    Css                css_;
    Logger             log_;

    std::string        sourceId_;
    std::string        urn_;
    
    xdata::InfoSpace  *appInfoSpace_;
    xdata::InfoSpace  *monInfoSpace_;

    ParamVec_t         standardParams_;
    ParamVec_t         monitorParams_;
    ParamVec_t         debugParams_;
    CounterVec_t       standardCounters_;
    CounterVec_t       monitorCounters_;
    CounterVec_t       debugCounters_;
    
    bool               parametersExported_;
    bool               countersAddedToParams_;

    std::string        largeAppIcon_;
    std::string        smallAppIcon_;
    std::string        smallDbgIcon_;
    std::string        hyperDAQIcon_;

  };
  

} // namespace evf


#endif
