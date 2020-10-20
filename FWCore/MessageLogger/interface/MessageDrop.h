#ifndef MessageLogger_MessageDrop_h
#define MessageLogger_MessageDrop_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     MessageDrop
//
/**\class MessageDrop MessageDrop.h 

 Description: <one line class summary>

 Usage:
    <usage>

*/

//
// Original Author:  M. Fischler and Jim Kowalkowski
//         Created:  Tues Feb 14 16:38:19 CST 2006
//

// Framework include files

#include "FWCore/Utilities/interface/EDMException.h"  // change log 4
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// system include files

#include <string>
#include <string_view>

// user include files

namespace edm {

  namespace messagedrop {
    class StringProducer;
    class StringProducerWithPhase;
    class StringProducerPath;
    class StringProducerSinglet;
  }  // namespace messagedrop

  struct MessageDrop {
  private:
    MessageDrop();

  public:
    MessageDrop(MessageDrop const&) = delete;
    MessageDrop& operator=(MessageDrop const&) = delete;

    ~MessageDrop();
    static MessageDrop* instance();
    std::string moduleContext();
    void setModuleWithPhase(std::string const& name, std::string const& label, unsigned int moduleID, const char* phase);
    void setPath(const char* type, std::string const& pathname);
    void setSinglet(const char* sing);
    void clear();

    std::string_view runEvent;
    unsigned int streamID;
    bool debugEnabled;
    bool infoEnabled;
    bool fwkInfoEnabled;
    bool warningEnabled;
    bool errorEnabled;

    CMS_THREAD_SAFE static std::string jobMode;
    CMS_THREAD_SAFE static unsigned char messageLoggerScribeIsRunning;
    CMS_THREAD_SAFE static bool debugAlwaysSuppressed;
    CMS_THREAD_SAFE static bool infoAlwaysSuppressed;
    CMS_THREAD_SAFE static bool fwkInfoAlwaysSuppressed;
    CMS_THREAD_SAFE static bool warningAlwaysSuppressed;

  private:
    edm::propagate_const<messagedrop::StringProducerWithPhase*> spWithPhase;
    edm::propagate_const<messagedrop::StringProducerPath*> spPath;
    edm::propagate_const<messagedrop::StringProducerSinglet*> spSinglet;
    messagedrop::StringProducer const* moduleNameProducer;
  };

  static const unsigned char MLSCRIBE_RUNNING_INDICATOR = 29;

}  // end of namespace edm

#endif  // MessageLogger_MessageDrop_h
