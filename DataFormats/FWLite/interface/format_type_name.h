#ifndef DataFormats_FWLite_format_type_name_h
#define DataFormats_FWLite_format_type_name_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     format_type_name
// 
/**\class format_type_name format_type_name.h DataFormats/FWLite/interface/format_type_name.h

 Description: functions used to format and unformat class names for conditions access

 Usage:
    Used internally 

*/
//
// Original Author:  
//         Created:  Thu Dec  3 16:52:43 CST 2009
// $Id$
//

// system include files

// user include files

// forward declarations

namespace fwlite {

  ///given a C++ class name returned a mangled name 
  std::string format_type_to_mangled(const std::string&);

  ///given a mangled name return the C++ class name
  std::string unformat_mangled_to_type(const std::string&);

}

#endif
