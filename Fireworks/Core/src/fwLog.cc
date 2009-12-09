// -*- C++ -*-
//
// Package:     Core
// Class  :     fwLog
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  8 23:10:10 CST 2009
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/fwLog.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
using namespace fwlog;
static LogLevel s_presentLevel = kInfo;

static const char* const s_levelNames[] = {"Debug","Info","Warning","Error"};

const char* levelName(LogLevel iLevel) {
   return s_levelNames[iLevel];
}

static std::ostream* s_logger = &std::cerr;

std::ostream& logger() {
   return *s_logger;
}

void setLogger(std::ostream* iNewLogger) {
   if (0==iNewLogger) {
      s_logger=&std::cout;
   } else {
      s_logger=iNewLogger;
   }
}

LogLevel presentLogLevel() {
   return s_presentLevel;
}
void setPresentLogLevel(LogLevel iLevel) {
   s_presentLevel=iLevel;
}
