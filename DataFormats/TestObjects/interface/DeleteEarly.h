#ifndef DataFormats_TestObjects_DeleteEarly_h
#define DataFormats_TestObjects_DeleteEarly_h
// -*- C++ -*-
//
// Package:     TestObjects
// Class  :     DeleteEarly
// 
/**\class DeleteEarly DeleteEarly.h DataFormats/TestObjects/interface/DeleteEarly.h

 Description: Data type used to test early deletion feature

 Usage:

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Feb  7 14:40:52 CST 2012
// $Id: DeleteEarly.h,v 1.1 2012/02/09 22:09:35 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace edmtest {
  class DeleteEarly
  {
    
  public:
    DeleteEarly() {};
    ~DeleteEarly() {++s_nDeletes;}
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    static unsigned int nDeletes() {return s_nDeletes;}
    
    // ---------- member functions ---------------------------
    
  private:
    
    // ---------- member data --------------------------------
    static unsigned int s_nDeletes;
    
  };

}

#endif
