#ifndef __EPV_h_
#define __EPV_h_

#include "toolbox/include/PackageInfo.h"

namespace EventProcessor {
    const string package     = "EventProcessor";
    const string versions    = "cmssw_0_2_0";
    const string description = "hlt event processor application";
    toolbox::PackageInfo getPackageInfo();
    void checkPackageDependencies() throw (toolbox::PackageInfo::VersionException);
    set<string, less<string> > getPackageDependencies();
}

#endif
