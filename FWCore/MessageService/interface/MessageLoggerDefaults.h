#ifndef MessageLogger_MessageLoggerDefaults_h
#define MessageLogger_MessageLoggerDefaults_h

// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     MessageLoggerDefaults
//
/**\class MessageLoggerDefaults MessageLoggerDefaults.h 

 Description: Structure holding defaults for MessageLogger service configuration

 Usage:
    #include "FWCore/Utilities/interface/JobMode.h"
    edm::JobMode mode = somehowObtainJobMode();
    MessageLoggerDefaults mlDefaults (mode);
    ...
    PSet p;
    std::string threshold = 
       p.getUntrackedParameter<std::string>("threshold", mlDefaults.threshold);

*/

//
// Original Author:  M. Fischler 
//         Created:  Tues Jun 14 10:38:19 CST 2007
// $Id: MessageLoggerDefaults.h,v 1.2 2008/07/09 21:47:29 fischler Exp $
//

// Framework include files

#include "FWCore/Utilities/interface/JobMode.h"

// system include files

#include <string>
#include <vector>
#include <map>
#include <cassert>

// Change log
//
// 6/15/07 mf	Original version
//
// -------------------------------------------------------

// user include files

// ----------------
// Maintenance Tips
// ----------------
//
// When default values change for one job mode of operation or another,
// implement that change in the appropriate member function implemented
// in HardwiredDefaults.cc.  For example, hardwireGridJobMode().
// 
// When a new default is needed, add the item to the struct, and then place
// the default value into the appropriate sections of the various member
// functions in HardwiredDefaults.cc. 
// It may be necessary to supply values of that default for each of the modes,
// even though the ErrorLogger default is suitable for all the modes but one.
//
// When the new default is of a type not already being accessed, also add
// the accessor method in this header, and implement it in
// MessageLoggerDefaults.cc following the pattern shown for existing items

namespace edm {
namespace service {

struct MessageLoggerDefaults {
public:
  static const int NO_VALUE_SET = -45654;

  struct Category {
    std::string threshold;
    int limit;
    int reportEvery;
    int timespan;
    Category() : 
      threshold("")
    , limit(NO_VALUE_SET)  
    , reportEvery(NO_VALUE_SET)  
    , timespan(NO_VALUE_SET) {}
  };
  
  struct Destination {
    std::string threshold;
    std::map<std::string,Category> category; 
    std::map<std::string,Category> sev; 
    std::string output;
  };
  
  // publicly available collections and structures

  std::vector<std::string> categories;
  std::vector<std::string> destinations;
  std::vector<std::string> fwkJobReports;
  std::vector<std::string> statistics;
  std::map<std::string,Destination> destination;
      
  // access to values set
  
  std::string threshold (std::string const & dest);
  std::string output    (std::string const & dest);

  int limit      (std::string const & dest, std::string const & cat);
  int reportEvery(std::string const & dest, std::string const & cat);
  int timespan   (std::string const & dest, std::string const & cat);
    
  int sev_limit      (std::string const & dest, std::string const & sev);
  int sev_reportEvery(std::string const & dest, std::string const & sev);
  int sev_timespan   (std::string const & dest, std::string const & sev);
    
  // Modes with hardwired defaults
  
  void hardwireGridJobMode();
  void hardwireReleaseValidationJobMode();
  void hardwireAnalysisJobMode();
  void hardwireNilJobMode();

  static edm::JobMode mode(std::string const & jm);
   
  // ctor

  explicit MessageLoggerDefaults (edm::JobMode mode = GridJobMode) {
    // mode-independent defaults
    
    // mode-dependent defaults
    switch (mode) {
      // GridJobMode:	Intended for automated batch-like processing, in which 
      //               	the output stream for cerr is routed to an apropro
      //               	file.  LogInfo messages are enabled, and at least 
      //	      	one such message is delivered from the framework per
      //		event, so this mode is not suitable for many-Hz light
      //		event processing. 
      case GridJobMode: 
        hardwireGridJobMode();
	break;       
      case ReleaseValidationJobMode:
        hardwireReleaseValidationJobMode();
	break;       
      case AnalysisJobMode:
        hardwireAnalysisJobMode();
	break;       
      case NilJobMode:
        hardwireNilJobMode();
	break;       
      default:
        // this should never happen!  No user error can get here.
	bool    Invalid_JobMode_in_ctor_of_MessageLoggerDefaults = false;
	assert (Invalid_JobMode_in_ctor_of_MessageLoggerDefaults);
    } // end of switch on mode
  }

};



} // end of namespace service
} // end of namespace edm


#endif // MessageLogger_MessageLoggerDefaults_h
