#ifndef Fireworks_Core_FWParameterSetterBase_h
#define Fireworks_Core_FWParameterSetterBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterSetterBase
//
/**\class FWParameterSetterBase FWParameterSetterBase.h Fireworks/Core/interface/FWParameterSetterBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:16:14 EST 2008
// $Id: FWParameterSetterBase.h,v 1.8 2012/02/22 03:45:57 amraktad Exp $
//

// system include files
#include <boost/shared_ptr.hpp>

// user include files

// forward declarations
class FWParameterBase;
class FWParameterSetterEditorBase;
class TGFrame;

class FWParameterSetterBase
{
public:
   FWParameterSetterBase();
   virtual ~FWParameterSetterBase();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   static boost::shared_ptr<FWParameterSetterBase> makeSetterFor(FWParameterBase*);

   // ---------- member functions ---------------------------

   void             attach(FWParameterBase*, FWParameterSetterEditorBase*);
   virtual TGFrame* build(TGFrame* iParent, bool labelBack = true) = 0;

   virtual void     setEnabled(bool);

protected:
   void update() const;
   FWParameterSetterEditorBase* frame() const { return m_frame; }

private:
   virtual void attach(FWParameterBase*) = 0;

   FWParameterSetterBase(const FWParameterSetterBase&);                  // stop default
   const FWParameterSetterBase& operator=(const FWParameterSetterBase&); // stop default

   // ---------- member data --------------------------------

   FWParameterSetterEditorBase* m_frame;
};

#endif
