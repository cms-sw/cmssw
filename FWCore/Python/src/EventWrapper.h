#ifndef Python_EventWrapper_h
#define Python_EventWrapper_h
// -*- C++ -*-
//
// Package:     Python
// Class  :     EventWrapper
// 
/**\class EventWrapper EventWrapper.h FWCore/Python/interface/EventWrapper.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 28 10:57:11 CDT 2006
// $Id: EventWrapper.h,v 1.1 2006/07/18 12:17:07 chrjones Exp $
//

// system include files
#include <string>
// user include files
#include "FWCore/Framework/interface/GenericHandle.h"

// forward declarations
namespace edm {
  class Event;
  
  namespace python {
    class ConstEventWrapper
  {

   public:
    ConstEventWrapper() : event_(0) {}
    ConstEventWrapper(const edm::Event&);
    //virtual ~ConstEventWrapper();

      // ---------- const member functions ---------------------
    void getByLabel(std::string const& , edm::GenericHandle& ) const;
    void getByLabel(std::string const& , std::string const&, edm::GenericHandle& ) const;
    
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //ConstEventWrapper(const ConstEventWrapper&); // stop default

      //const ConstEventWrapper& operator=(const ConstEventWrapper&); // stop default

      // ---------- member data --------------------------------
      edm::Event const* event_;
};
  }
}

#endif
