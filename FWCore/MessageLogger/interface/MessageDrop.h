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
// $Id: MessageDrop.h,v 1.9 2007/03/31 00:17:07 fischler Exp $
//

// Framework include files

#include "FWCore/Utilities/interface/EDMException.h"	// change log 4


// system include files

#include <string>

// Change log
//
//  1  mf 5/12/06	initialize debugEnabled to true, to avoid unitialized
//			data detection in memory checks (and to be safe in
//			getting enabled output independant of timings) 
//
//  4  mf 2/22/07	static ex_p to have a way to convey exceptions to throw
//			(this is needed when configuring could lead to an 
//			exception, for example)
//
//  5  mf 2/22/07	jobreport_name to have a way to convey content
//			of jobreport option from cmsRun to MessageLogger class
//
//  6  mf 6/18/07	jobMode to have a way to convey choice of hardwired
//			MessageLogger defaults

// user include files

namespace edm {

struct MessageDrop {
private:
  MessageDrop() 
  : moduleName ("")
  , runEvent("pre-events")
  , jobreport_name()					// change log 5
  , jobMode("")						// change log 6
  , debugEnabled(true) 					// change log 1
  , infoEnabled(true) 					// change log 3
  , warningEnabled(true)				// change log 3
  {  } 
public:
  static MessageDrop * instance ();
  std::string moduleName;
  std::string runEvent;
  std::string jobreport_name;				// change log 5
  std::string jobMode;					// change log 6
  bool debugEnabled;
  bool infoEnabled;
  bool warningEnabled;
  static edm::Exception * ex_p;				// change log 4
};

} // end of namespace edm


#endif // MessageLogger_MessageDrop_h
