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
// $Id: FWViewManagerBase.h,v 1.1 2008/01/07 05:48:45 chrjones Exp $
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
      
      bool useableBuilder(const std::string&) const;

   protected:
      FWViewManagerBase(const char* iBuilderNamePostfix);

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
