// $Id: XHTMLMonitor.cc,v 1.4 2011/03/07 15:31:32 mommsen Exp $
/// @file: XHTMLMonitor.cc

#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"

#include <xercesc/util/PlatformUtils.hpp>

using namespace xercesc;

boost::mutex stor::XHTMLMonitor::xhtmlMakerMutex_;

stor::XHTMLMonitor::XHTMLMonitor()
{
  xhtmlMakerMutex_.lock();
  XMLPlatformUtils::Initialize();
}

stor::XHTMLMonitor::~XHTMLMonitor()
{
  XMLPlatformUtils::Terminate();
  xhtmlMakerMutex_.unlock();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
