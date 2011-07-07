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

class FWGUIManager;
class FWColorManager;
class FWTableWidget;
class FWGeometryTableManager;
class FWConfiguration;
class FWColorPopup;
class FWGeoTopNode;

class FWParameterBase;


class FWGeometryBrowser : public TGMainFrame
#ifndef __CINT__
                        , public FWConfigurableParameterizable , public FWParameterSetterEditorBase
#endif
{

public:
   enum EMode { kNode, kVolume };


   FWGeometryBrowser(FWGUIManager*, FWColorManager*);
   virtual ~FWGeometryBrowser();
  
   void cellClicked(Int_t iRow, Int_t iColumn, 
                    Int_t iButton, Int_t iKeyMod, 
                    Int_t iGlobalX, Int_t iGlobalY);
  
   void windowIsClosing();
   void chosenItem(int);
   void browse();
   void readFile();
   void updateStatusBar(const char* status = 0);
   void updateFilter();

   void printTable();

   void cdNode(int);
   void cdTop();
   void cdUp();
   void setPath(int, std::string&);

   bool getVolumeMode()      const { return m_mode.value(); }
   std::string getFilter ()  const { return m_filter.value(); }
   int getAutoExpand()       const { return m_autoExpand.value(); }
   int getVisLevel()         const  {return m_visLevel.value(); }

   int getTopNodeIdx() const { return m_topNodeIdx.value(); }

   TGeoManager*   geoManager() { return m_geoManager; }
   FWGeometryTableManager*  getTableManager() { return m_tableManager;} 
   virtual void setFrom(const FWConfiguration&);
   Bool_t HandleKey(Event_t *event);
   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   void nodeColorChangeRequested(Color_t);

private:
   FWGeometryBrowser(const FWGeometryBrowser&);
   const FWGeometryBrowser& operator=(const FWGeometryBrowser&);


#ifndef __CINT__
   FWEnumParameter         m_mode;
   FWStringParameter       m_filter; 
   FWLongParameter         m_autoExpand;
   FWLongParameter         m_visLevel;   
   FWLongParameter         m_topNodeIdx;   
#endif

   FWGUIManager           *m_guiManager;
   FWColorManager         *m_colorManager;

   FWTableWidget          *m_tableWidget;
   FWGeometryTableManager *m_tableManager;

   TFile                  *m_geometryFile;
   TGStatusBar            *m_statBar;
   TGCompositeFrame       *m_settersFrame;

   // int                     m_selectedIdx;

   TGeoManager            *m_geoManager;
   FWGeoTopNode           *m_eveTopNode;

   FWColorPopup           *m_colorPopup;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   void resetSetters();
   void makeSetter(TGCompositeFrame* frame, FWParameterBase* param);
   void loadGeometry();

   void backgroundChanged();
   void autoExpandChanged();
   void refreshTable3D();


   ClassDef(FWGeometryBrowser, 0);
};

#endif

