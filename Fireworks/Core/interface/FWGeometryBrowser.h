#ifndef Fireworks_Core_FWGeometryBrowser_h
#define Fireworks_Core_FWGeometryBrowser_h

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#endif
#include "TGFrame.h"

#ifndef __CINT__
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWStringParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"
#endif

class TFile;
class TGTextButton;
class TGeoNode;
class TGeoVolume;
class TGTextEntry;
class TGComboBox;
class TGStatusBar;
class TGeoManager;
class TEveGeoTopNode;

class FWGUIManager;
class FWTableWidget;
class FWGeometryTableManager;
class FWConfiguration;

class FWParameterBase;

class FWGeometryBrowser : public TGMainFrame
#ifndef __CINT__
                      , public FWConfigurableParameterizable , public FWParameterSetterEditorBase
#endif
{
   friend class FWGeometryTableManager;

public:
   enum EMode { kNode, kVolume };

   FWGeometryBrowser(FWGUIManager*);
   virtual ~FWGeometryBrowser();
  
   void cellClicked(Int_t iRow, Int_t iColumn, 
                    Int_t iButton, Int_t iKeyMod, 
                    Int_t iGlobalX, Int_t iGlobalY);
  
   void windowIsClosing();
   void chosenItem(int);
   void browse();
   void reset();
   void readFile();
   void updateStatusBar(const char* status);

   TGeoManager*   geoManager() { return m_geoManager;}

   virtual void setFrom(const FWConfiguration&);
   Bool_t HandleKey(Event_t *event);
   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   
protected:
#ifndef __CINT__
   FWEnumParameter         m_mode;
   FWStringParameter       m_filter; 
   FWLongParameter         m_autoExpand; 
   FWLongParameter         m_maxDaughters; 
#endif

private:
   FWGeometryBrowser(const FWGeometryBrowser&);
   const FWGeometryBrowser& operator=(const FWGeometryBrowser&);

   FWGUIManager           *m_guiManager;

   FWTableWidget          *m_tableWidget;
   FWGeometryTableManager *m_tableManager;

   TFile                  *m_geometryFile;
   TGStatusBar            *m_statBar;
   TGCompositeFrame       *m_settersFrame;

   int                     m_selectedIdx;

   TGeoManager            *m_geoManager;
   TEveGeoTopNode         *m_eveTopNode;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   void resetSetters();
   void makeSetter(TGCompositeFrame* frame, FWParameterBase* param);
   void loadGeometry();

   ClassDef(FWGeometryBrowser, 0);
};

#endif

