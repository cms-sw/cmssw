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
// $Id$
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
class TH2;
class TH2F;
class TH2C;
class TCanvas;
class FW3DLegoDataProxyBuilder;
class FWEventItem;
class FWGUIManager;
class TGFrame;
class FW3DLegoView;
class FWViewBase;
class TObject;

struct FW3DLegoModelProxy
{
   boost::shared_ptr<FW3DLegoDataProxyBuilder>   builder;
   TObject*                           product; //owned by builder
   bool ignore;
   FW3DLegoModelProxy():product(0), ignore(false){}
   FW3DLegoModelProxy(boost::shared_ptr<FW3DLegoDataProxyBuilder> iBuilder):
    builder(iBuilder),product(0), ignore(false){}
};

class FW3DLegoViewManager : public FWViewManagerBase
{

   public:
      FW3DLegoViewManager(FWGUIManager*);
      virtual ~FW3DLegoViewManager();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable();

      virtual void newItem(const FWEventItem*);

      void registerProxyBuilder(const std::string&, 
				const std::string&,
                                const FWEventItem*);
   
      FWViewBase* buildView(TGFrame* iParent);

      void exec3event(int event, int x, int y, TObject *selected);
      void pixel2wc(const Int_t PixelX, const Int_t PixelY, 
		    Double_t& WCX, Double_t& WCY, const Double_t WCZ = 0);
   
   protected:
   virtual void modelChangesComing();
   virtual void modelChangesDone();

   private:
      FW3DLegoViewManager(const FW3DLegoViewManager&); // stop default

      const FW3DLegoViewManager& operator=(const FW3DLegoViewManager&); // stop default

      void makeProxyBuilderFor(const FWEventItem* iItem);
   
      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::vector<std::string> > TypeToBuilders;
      TypeToBuilders m_typeToBuilders;
      std::vector<FW3DLegoModelProxy> m_modelProxies;

      //TCanvas* m_legoCanvas;
      std::vector<boost::shared_ptr<FW3DLegoView> > m_views;
      THStack* m_stack;
      TH2F* m_background;
      TH2F* m_highlight;
      TH2C* m_highlight_map;
      int  m_legoRebinFactor;
      static const char* const m_builderPrefixes[];
};


#endif
