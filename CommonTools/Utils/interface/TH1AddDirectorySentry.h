#ifndef CommonTools_Utils_TH1AddDirectorySentry_h
#define CommonTools_Utils_TH1AddDirectorySentry_h
// -*- C++ -*-
//
// Package:     UtilAlgos
// Class  :     TH1AddDirectorySentry
// 
/**\class TH1AddDirectorySentry TH1AddDirectorySentry.h CommonTools/UtilAlgos/interface/TH1AddDirectorySentry.h

 Description: Manages the status of the ROOT directory

 Usage:
    Construct an instance of this object in a routine in which you expect a ROOT histogram to be
 automatically added to the current directory in a file. The destructor will be sure to reset ROOT
 to its previous setting.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Nov  8 12:16:13 EST 2007
// $Id: TH1AddDirectorySentry.h,v 1.1 2010/01/05 14:21:14 hegner Exp $
//

class TH1AddDirectorySentry
{

   public:
      TH1AddDirectorySentry();
      ~TH1AddDirectorySentry();


   private:
      TH1AddDirectorySentry(const TH1AddDirectorySentry&);
      TH1AddDirectorySentry& operator=(const TH1AddDirectorySentry&);
      bool status_;
};


#endif
