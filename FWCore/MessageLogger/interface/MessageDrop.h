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
// Original Author:  M. Fischler and Jim Kowalkowsi
//         Created:  Tues Feb 14 16:38:19 CST 2006
// $Id: MessageDrop.h,v 1.1 2006/02/15 00:39:09 fischler Exp $
//

// system include files

#include <string>


// user include files

namespace edm {

struct MessageDrop {
private:
  MessageDrop() :  moduleName (""), runEvent("pre-services") { } 
public:
  static MessageDrop * instance ();
  std::string moduleName;
  std::string runEvent;
  bool debugEnabled;
};



} // end of namespace edm


#endif // MessageLogger_MessageDrop_h
