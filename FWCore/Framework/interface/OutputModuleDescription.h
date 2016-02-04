#ifndef FWCore_Framework_OutputModuleDescription_h
#define FWCore_Framework_OutputModuleDescription_h

/*----------------------------------------------------------------------

OutputModuleDescription : the stuff that is needed to configure an
output module that does not come in through the ParameterSet  

$Id: OutputModuleDescription.h,v 1.2 2008/02/21 22:47:51 wdd Exp $
----------------------------------------------------------------------*/
namespace edm {

  struct OutputModuleDescription {
    OutputModuleDescription() : maxEvents_(-1) {}
    OutputModuleDescription(int maxEvents) :
      maxEvents_(maxEvents)
    {}
    int maxEvents_;
  };
}

#endif
