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
//

// system include files
#include <memory>

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

   static std::shared_ptr<FWParameterSetterBase> makeSetterFor(FWParameterBase*);

   // ---------- member functions ---------------------------

   void             attach(FWParameterBase*, FWParameterSetterEditorBase*);
   virtual TGFrame* build(TGFrame* iParent, bool labelBack = true) = 0;

   virtual void     setEnabled(bool);

protected:
   void update() const;
   FWParameterSetterEditorBase* frame() const { return m_frame; }

private:
   virtual void attach(FWParameterBase*) = 0;

   FWParameterSetterBase(const FWParameterSetterBase&) = delete;                  // stop default
   const FWParameterSetterBase& operator=(const FWParameterSetterBase&) = delete; // stop default

   // ---------- member data --------------------------------

   FWParameterSetterEditorBase* m_frame;
};

#endif
