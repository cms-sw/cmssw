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
// $Id: FWViewManagerBase.h,v 1.10 2008/06/10 19:26:21 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>
#include <map>

// user include files

// forward declarations
class FWEventItem;
class TClass;
class DetIdToMatrix;
class FWModelId;
class FWModelChangeManager;

class FWViewManagerBase
{

   public:
      virtual ~FWViewManagerBase();

      // ---------- const member functions ---------------------
      /** returning an empty vector means this type can not be handled*/
      virtual std::vector<std::string> purposeForType(const std::string& iTypeName) const = 0;
   
      virtual std::set<std::pair<std::string,std::string> > supportedTypesAndPurpose() const = 0;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newItem(const FWEventItem*) = 0;

      void setGeom(const DetIdToMatrix* geom){ m_detIdToGeo = geom; }
      const DetIdToMatrix* getGeom(const DetIdToMatrix* geom){ return m_detIdToGeo; }
      
      void setChangeManager(FWModelChangeManager* iCM);
      

      void modelChangesComingSlot();
      void modelChangesDoneSlot();
   
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

      FWModelChangeManager& changeManager() const;
      const DetIdToMatrix* detIdToGeo() const;
   private:
      FWViewManagerBase(const FWViewManagerBase&); // stop default

      const FWViewManagerBase& operator=(const FWViewManagerBase&); // stop default

      // ---------- member data --------------------------------
      const DetIdToMatrix* m_detIdToGeo;
   
      FWModelChangeManager* m_changeManager;

};


#endif
