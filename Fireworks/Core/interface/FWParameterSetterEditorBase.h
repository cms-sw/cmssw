#ifndef Fireworks_Core_FWParameterSetterEditorBase_h
#define Fireworks_Core_FWParameterSetterEditorBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterSetterEditorBase
//
/**\class FWParameterSetterEditorBase FWParameterSetterEditorBase.h Fireworks/Core/interface/FWParameterSetterEditorBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Jun 26 11:17:59 EDT 2008
// $Id: FWParameterSetterEditorBase.h,v 1.4 2012/02/22 03:45:57 amraktad Exp $
//

// system include files

// user include files

// forward declarations

class FWParameterSetterEditorBase
{

public:
   FWParameterSetterEditorBase();
   virtual ~FWParameterSetterEditorBase();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual void updateEditor();

private:
   FWParameterSetterEditorBase(const FWParameterSetterEditorBase&);    // stop default

   const FWParameterSetterEditorBase& operator=(const FWParameterSetterEditorBase&);    // stop default

   // ---------- member data --------------------------------

};


#endif
