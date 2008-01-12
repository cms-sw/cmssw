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
// $Id: FWViewManagerBase.h,v 1.2 2008/01/11 00:49:03 chrjones Exp $
//

// system include files
#include <string>
#include <vector>

// user include files

// forward declarations
class FWEventItem;
class TClass;

class FWViewManagerBase
{

   public:
      virtual ~FWViewManagerBase();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable() = 0;

      virtual void newItem(const FWEventItem*) = 0;

      virtual void registerProxyBuilder(const std::string&, 
					const std::string&) = 0;
      
      /** returns 'true' if the name of the builder matches the naming
       conventions for builders used by this View
       */
      bool useableBuilder(const std::string& iBuilderName) const;

   protected:
      FWViewManagerBase(const char* iBuilderNamePostfix);
      template<typename TIter>
        FWViewManagerBase(TIter iBegin,TIter iEnd):
         m_builderNamePostfixes(iBegin,iEnd) {}

      /*handles dynamic loading of a library or macro containing the class
         named iNameOfClass which inherits from iBaseClass.  The returned
	 void* will correspond to the address of the 'BaseClass'
      */
      void* createInstanceOf(const TClass* iBaseClass,
			     const char* iNameOfClass);
   private:
      FWViewManagerBase(const FWViewManagerBase&); // stop default

      const FWViewManagerBase& operator=(const FWViewManagerBase&); // stop default

      // ---------- member data --------------------------------
      std::vector<std::string> m_builderNamePostfixes;

};


#endif
