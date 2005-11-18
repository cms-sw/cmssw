#ifndef Services_MessageLogger_h
#define Services_MessageLogger_h

// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
//
/**\class MessageLogger MessageLogger.h FWCore/Services/interface/MessageLogger.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
// $Id: MessageLogger.h,v 1.2 2005/11/14 20:08:17 fischler Exp $
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm  {
namespace service  {


class MessageLogger
{
public:
  MessageLogger( ParameterSet const &, ActivityRegistry & );

  void  postBeginJob();
  void  postEndJob();

  void  preEventProcessing ( edm::EventID const &, edm::Timestamp const & );
  void  postEventProcessing( Event const &, EventSetup const & );

  void  preModule ( ModuleDescription const & );
  void  postModule( ModuleDescription const & );

private:
  // put an ErrorLog object here, and maybe more

};  // MessageLogger


}  // namespace service
}  // namespace edm



#endif  // Services_MessageLogger_h
