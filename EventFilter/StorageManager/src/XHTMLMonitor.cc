// $Id: XHTMLMonitor.cc,v 1.3.14.3 2011/02/28 17:56:06 mommsen Exp $
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
