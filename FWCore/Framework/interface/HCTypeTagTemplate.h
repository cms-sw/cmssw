#ifndef Framework_HCTypeTagTemplate_h
#define Framework_HCTypeTagTemplate_h
// -*- C++ -*-
//
// Package:     HeteroContainer
// Module:      HCTypeTagTemplate
// 
// Description: initializer class for HCTypeTag
//
// Usage:
//    NOTE: This class will go away once CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h switches to using className<T>()
//
//    See HCTypeTag.h for usage.
//
//    This templated class does NOT have a definition of the method
//    className().  Instead className() is supposed to be specialized
//    for each parameter T.  We have included a cpp macro to make it 
//    easier to do this specialization.  To use the macro simple do
//
//    #include "interface/HCTypeTagTemplate.h"
//    HCTYPETAGTEMPLATE_CLASS(Foo, MyGroup)
//     
// Author:      Chris D. Jones
// Created:     Sun Sep 20 15:31:56 EDT 1998
// $Id: HCTypeTagTemplate.h,v 1.8 2009/04/26 22:18:35 chrjones Exp $
//
// Revision history
//
// $Log: HCTypeTagTemplate.h,v $
// Revision 1.8  2009/04/26 22:18:35  chrjones
// Now register type at shared object load time rather than at dynamic load time. This should decrease 'code bloat' caused by the function local static.
//
// Revision 1.7  2005/11/16 00:17:39  chrjones
// moved constructor definition to avoid physical coupling problems with the template parameters
//
// Revision 1.6  2005/11/11 20:55:54  chrjones
// use new TypeIDBase for basis of all type comparisons
//
// Revision 1.5  2005/09/01 23:30:48  wmtan
// fix rule violations found by rulechecker
//
// Revision 1.4  2005/09/01 05:20:05  wmtan
// Fix Rules violations found by RuleChecker
//
// Revision 1.3  2005/07/14 22:50:52  wmtan
// Rename packages
//
// Revision 1.2  2005/06/23 19:59:30  wmtan
// fix rules violations found by rulechecker
//
// Revision 1.1  2005/05/29 02:29:53  wmtan
// initial population
//
// Revision 1.3  2005/04/18 15:35:59  chrjones
// classValue code now in .icc file to avoid problems with static function variables
//
// Revision 1.2  2005/04/04 20:30:10  chrjones
// added namespace
//
// Revision 1.1  2005/03/28 15:03:19  chrjones
// first submission
//
// Revision 1.6  2002/06/04 18:22:45  cleo3
// use new explicit template specialization syntax
//
// Revision 1.5  2000/07/25 13:42:35  cdj
// HCTypeTag can now find a TypeTag from the name of a type
//
// Revision 1.4  1998/10/20 18:06:22  cdj
// modified so .cc file can not be multiply included
//
// Revision 1.3  1998/09/30 15:42:42  cdj
// removed inlined version of className(), now get from HCTypeTagTemplate.cc
//
// Revision 1.2  1998/09/29 18:57:01  cdj
// fixed cpp macro
//
// Revision 1.1.1.1  1998/09/23 14:13:12  cdj
// first submission
//

// system include files

// user include files
#include "FWCore/Framework/interface/HCTypeTag.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
template < class T, class Group >
class HCTypeTagTemplate 
{
   public:

      // ---------- Constructors and destructor ---------------- 
      HCTypeTagTemplate();
      //virtual ~HCTypeTagTemplate(); //not needed

      // ---------- static member functions --------------------
      static const char* className() {
         return edm::eventsetup::heterocontainer::className<T>();
      }

   protected:

   private:
      // ---------- Constructors and destructor ----------------
      //HCTypeTagTemplate(const HCTypeTagTemplate&); // stop default

      // ---------- assignment operator(s) ---------------------
      //const HCTypeTagTemplate& operator=(const HCTypeTagTemplate&); // stop default

};

// inline function definitions

      }
   }
}

#endif
