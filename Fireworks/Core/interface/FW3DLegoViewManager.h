#ifndef Fireworks_Core_FW3DLegoViewManager_h
#define Fireworks_Core_FW3DLegoViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoViewManager
// 
/**\class FW3DLegoViewManager FW3DLegoViewManager.h Fireworks/Core/interface/FW3DLegoViewManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 22:01:21 EST 2008
// $Id: FW3DLegoViewManager.h,v 1.2 2008/01/21 01:17:08 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"

// forward declarations
class TList;
class THStack;
class TH2F;
class TCanvas;
class FW3DLegoDataProxyBuilder;
class FWEventItem;

struct FW3DLegoModelProxy
{
   boost::shared_ptr<FW3DLegoDataProxyBuilder>   builder;
   TH2F*                           product; //owned by builder
   FW3DLegoModelProxy():product(0){}
   FW3DLegoModelProxy(boost::shared_ptr<FW3DLegoDataProxyBuilder> iBuilder):
    builder(iBuilder),product(0) {}
};

class FW3DLegoViewManager : public FWViewManagerBase
{

   public:
      FW3DLegoViewManager();
      virtual ~FW3DLegoViewManager();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable();

      virtual void newItem(const FWEventItem*);

      void registerProxyBuilder(const std::string&, 
				const std::string&);

      void DynamicCoordinates();
      void exec3event(int event, int x, int y, TObject *selected);
      void pixel2wc(const Int_t PixelX, const Int_t PixelY, 
		    Double_t& WCX, Double_t& WCY, const Double_t WCZ = 0);

   protected:
   virtual void modelChangesComing();
   virtual void modelChangesDone();

   private:
      FW3DLegoViewManager(const FW3DLegoViewManager&); // stop default

      const FW3DLegoViewManager& operator=(const FW3DLegoViewManager&); // stop default

      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::string> TypeToBuilder;
      TypeToBuilder m_typeToBuilder;
      std::vector<FW3DLegoModelProxy> m_modelProxies;

      TCanvas* m_legoCanvas;
      THStack* m_stack;
      TH2F* m_background;
      int  m_legoRebinFactor;
};


#endif
