// $Id: XHTMLMonitor.cc,v 1.2 2009/06/10 08:15:29 dshpakov Exp $
/// @file: XHTMLMonitor.cc

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
