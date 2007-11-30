#ifndef FWCore_Framework_OutputModuleDescription_h
#define FWCore_Framework_OutputModuleDescription_h

/*----------------------------------------------------------------------

OutputModuleDescription : the stuff that is needed to configure an
output module that does not come in through the ParameterSet  

$Id: OutputModuleDescription.h,v 1.8 2007/11/29 17:27:38 wmtan Exp $
----------------------------------------------------------------------*/
namespace edm {

  struct OutputModuleDescription {
    OutputModuleDescription() : maxEvents_(-1), maxLumis_(-1) {}
    OutputModuleDescription(int maxEvents, int maxLumis) :
      maxEvents_(maxEvents),
      maxLumis_(maxLumis)
    {}
    int maxEvents_;
    int maxLumis_;
  };
}

#endif
