#ifndef Fireworks_Core_FWViewManagerBase_h
#define Fireworks_Core_FWViewManagerBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewManagerBase
//
/**\class FWViewManagerBase FWViewManagerBase.h Fireworks/Core/interface/FWViewManagerBase.h

   Description: Base class for a Manger for a specific type of View

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Sat Jan  5 10:29:00 EST 2008
// $Id: FWViewManagerBase.h,v 1.16 2009/04/07 14:02:33 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>
#include <map>

// user include files

//Needed for gccxml
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

// forward declarations
class FWEventItem;
class TClass;
class DetIdToMatrix;
class FWModelId;
class FWModelChangeManager;
class FWColorManager;
class FWTypeToRepresentations;

class FWViewManagerBase
{

public:
   virtual ~FWViewManagerBase();

   // ---------- const member functions ---------------------
   virtual FWTypeToRepresentations supportedTypesAndRepresentations() const = 0;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void newItem(const FWEventItem*) = 0;

   void setGeom(const DetIdToMatrix* geom){
      m_detIdToGeo = geom;
   }
   const DetIdToMatrix* getGeom(const DetIdToMatrix* geom){
      return m_detIdToGeo;
   }

   void setChangeManager(FWModelChangeManager* iCM);
   void setColorManager(FWColorManager* iCM);


   void modelChangesComingSlot();
   void modelChangesDoneSlot();
   void colorsChangedSlot();
   virtual void eventBegin(){
   };
   virtual void eventEnd(){
   };

protected:
   FWViewManagerBase();

   /**handles dynamic loading of a library or macro containing the class
      named iNameOfClass which inherits from iBaseClass.  The returned
      void* will correspond to the address of the 'BaseClass'
    */
   void* createInstanceOf(const TClass* iBaseClass,
                          const char* iNameOfClass);

   /** called when models have changed and so the display must be updated*/
   virtual void modelChangesComing() = 0;
   virtual void modelChangesDone() = 0;
   virtual void colorsChanged() = 0;

   FWModelChangeManager& changeManager() const;
   FWColorManager& colorManager() const;
   const DetIdToMatrix* detIdToGeo() const;
private:
   FWViewManagerBase(const FWViewManagerBase&);    // stop default

   const FWViewManagerBase& operator=(const FWViewManagerBase&);    // stop default

   // ---------- member data --------------------------------
   const DetIdToMatrix* m_detIdToGeo;

   FWModelChangeManager* m_changeManager;
   FWColorManager* m_colorManager;

};


#endif
