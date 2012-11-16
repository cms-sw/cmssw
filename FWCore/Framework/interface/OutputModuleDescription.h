#ifndef FWCore_Framework_OutputModuleDescription_h
#define FWCore_Framework_OutputModuleDescription_h

/*----------------------------------------------------------------------

OutputModuleDescription : the stuff that is needed to configure an
output module that does not come in through the ParameterSet  

$Id: OutputModuleDescription.h,v 1.1 2007/11/30 07:06:55 wmtan Exp $
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
