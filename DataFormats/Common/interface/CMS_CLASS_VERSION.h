#ifndef DataFormats_Common_CMS_CLASS_VERSION_h
#define DataFormats_Common_CMS_CLASS_VERSION_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     CMS_CLASS_VERSION
// 
/**\class CMS_CLASS_VERSION CMS_CLASS_VERSION.h DataFormats/Common/interface/CMS_CLASS_VERSION.h

 Description: Macro used to add versioning info needed by ROOT to a CMS templated class

 Usage:
    Add the following line to a template class' public declaration area
      CMS_CLASS_VERSION(<number>)

  For classes that have been stored into ROOT files before the addition of the macro, we suggest starting the <number> at 10. This was chosen to be larger than any known number of stored changes to a templated class.
  For new classes that have never been stored, we suggest starting the <number> at 2 (0 and 1 have special meanings to ROOT).
 
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Mar  3 14:25:29 CST 2011
// $Id$
//

// system include files

// user include files

// forward declarations
#define CMS_CLASS_VERSION(_version_) static short Class_Version() { return _version_;}

#endif
