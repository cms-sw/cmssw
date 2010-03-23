#ifndef EVENTFILTER_GOODIES_IDIE_H
#define EVENTFILTER_GOODIES_IDIE_H

#include "EventFilter/Utilities/interface/Exception.h"

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"

#include "xgi/Utils.h"
#include "xgi/Input.h"
#include "xgi/Output.h"
#include "xgi/Method.h"

#include "xdaq/Application.h"

#include "toolbox/net/URN.h"
#include "toolbox/fsm/exception/Exception.h"


#include <vector>
#include <queue>

#include <sys/time.h>


namespace evf {


  namespace internal{
    struct path{
      path():l1pass(0),paccept(0),preject(0),pexcept(0),pspass(0){}
      unsigned int l1pass;
      unsigned int paccept;
      unsigned int preject;
      unsigned int pexcept;
      unsigned int pspass;
    };
    struct fu{
      time_t tstamp;
      unsigned int ccount;
      std::vector<pid_t> cpids;
      std::vector<std::string> signals;
      std::vector<std::string> stacktraces;
    };
  }
  typedef std::map<std::string,internal::fu> fmap;
  typedef fmap::iterator ifmap;
  
  class iDie : public xdaq::Application
  {
  public:
    //
    // xdaq instantiator macro
    //
    XDAQ_INSTANTIATOR();
  
    
    //
    // construction/destruction
    //
    iDie(xdaq::ApplicationStub *s);
    virtual ~iDie();
    //UI
    void defaultWeb(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void summaryTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void detailsTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void dumpTable(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    void updater(xgi::Input *in,xgi::Output *out)
      throw (xgi::exception::Exception);
    //AI
    void postEntry(xgi::Input*in,xgi::Output*out)
      throw (xgi::exception::Exception);
    
  private:
    //
    // private member functions
    //
    
    void reset();
    //
    // member data
    //

    // message logger
    Logger                          log_;

    // monitored parameters
    xdata::String                   url_;
    xdata::String                   class_;
    xdata::UnsignedInteger32        instance_;
    xdata::String                   hostname_;
    xdata::UnsignedInteger32        runNumber_;

    fmap                            fus_;

    std::vector<std::vector<internal::path> > paths_;

    unsigned int                    totalCores_;

  }; // class iDie


} // namespace evf


#endif
