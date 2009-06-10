// $Id$

#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"

#include <xercesc/util/PlatformUtils.hpp>

using namespace xercesc;

XHTMLMonitor::XHTMLMonitor()
{
  XMLPlatformUtils::Initialize();
}

XHTMLMonitor::~XHTMLMonitor()
{
  XMLPlatformUtils::Terminate();
}
