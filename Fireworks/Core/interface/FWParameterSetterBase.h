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
// $Id$
//

// system include files

// user include files

// forward declarations
class FWParameterBase;
class TGFrame;
class TGedFrame;

class FWParameterSetterBase
{

   public:
      FWParameterSetterBase();
      virtual ~FWParameterSetterBase();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
   
      static FWParameterSetterBase* makeSetterFor(FWParameterBase*);
      // ---------- member functions ---------------------------
      void attach(FWParameterBase*, TGedFrame*);
      virtual TGFrame* build(TGFrame* iParent) = 0;
   protected:
      void update() const;
      TGedFrame* frame() const { return m_frame;}
   private:
      virtual void attach(FWParameterBase*) = 0;
   
      FWParameterSetterBase(const FWParameterSetterBase&); // stop default

      const FWParameterSetterBase& operator=(const FWParameterSetterBase&); // stop default

      // ---------- member data --------------------------------
      TGedFrame* m_frame;
};


#endif
