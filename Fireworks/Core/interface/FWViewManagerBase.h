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
// $Id: FWViewManagerBase.h,v 1.4 2008/01/19 04:51:13 dmytro Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>

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
      /** returns 'true' if the name of the builder matches the naming
       conventions for builders used by this View
       */
      bool useableBuilder(const std::string& iBuilderName) const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable() = 0;

      virtual void newItem(const FWEventItem*) = 0;

      virtual void registerProxyBuilder(const std::string&, 
					const std::string&) = 0;
      void setGeom(const DetIdToMatrix* geom){ m_detIdToGeo = geom; }
      const DetIdToMatrix* getGeom(const DetIdToMatrix* geom){ return m_detIdToGeo; }
      
      void setChangeManager(FWModelChangeManager* iCM);
      

      void modelsHaveChangedSlot(const std::set<FWModelId>& );

   protected:
      FWViewManagerBase(const char* iBuilderNamePostfix);
      template<typename TIter>
        FWViewManagerBase(TIter iBegin,TIter iEnd):
         m_builderNamePostfixes(iBegin,iEnd) {}
      
      /**handles dynamic loading of a library or macro containing the class
       named iNameOfClass which inherits from iBaseClass.  The returned
       void* will correspond to the address of the 'BaseClass'
       */
      void* createInstanceOf(const TClass* iBaseClass,
			     const char* iNameOfClass);

      /** called when models have changed and so the display must be updated*/
      virtual void modelsHaveChanged(const std::set<FWModelId>& ) = 0;

      FWModelChangeManager& changeManager() const;
      const DetIdToMatrix* detIdToGeo() const;
   private:
      FWViewManagerBase(const FWViewManagerBase&); // stop default

      const FWViewManagerBase& operator=(const FWViewManagerBase&); // stop default

      // ---------- member data --------------------------------
      const DetIdToMatrix* m_detIdToGeo;
   
      std::vector<std::string> m_builderNamePostfixes;
      FWModelChangeManager* m_changeManager;

};


#endif
