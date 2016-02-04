#ifndef Fireworks_Core_FWGeometryTableView_h
#define Fireworks_Core_FWGeometryTableView_h

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#endif


#include "Rtypes.h"
#include "TGFrame.h"

#ifndef __CINT__
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWStringParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"
#endif


class TGTextButton;
class TGeoNode;
class TGeoVolume;
class TGTextEntry;
class TGComboBox;
class TGStatusBar;
class TGeoManager;
class TEveWindowSlot;
class TEveWindowFrame;

class FWTableWidget;
class FWGeometryTableManager;
class FWConfiguration;
class FWColorPopup;
class FWColorManager;
class FWGeoTopNode;

class FWParameterBase;
class FWViewCombo;
class FWGUIValidatingTextEntry;
class FWGeoMaterialValidator;

class FWGeometryTableView
#ifndef __CINT__
   : public  FWViewBase,
     public  FWParameterSetterEditorBase
#endif
{

public:
   enum EMode { kNode, kVolume };

   FWGeometryTableView(TEveWindowSlot*, FWColorManager*, TGeoNode*, TObjArray*);
   virtual ~FWGeometryTableView();
  
   void cellClicked(Int_t iRow, Int_t iColumn, 
                    Int_t iButton, Int_t iKeyMod, 
                    Int_t iGlobalX, Int_t iGlobalY);
  
   void chosenItem(int);
   void selectView(int);
   void filterListCallback();
   void filterTextEntryCallback();
   void updateFilter(std::string&);

   void printTable();

   void cdNode(int);
   void cdTop();
   void cdUp();
   void setPath(int, std::string&);

   bool getVolumeMode()      const { return m_mode.value(); }
   std::string getFilter ()  const { return m_filter.value(); }
   int getAutoExpand()       const { return m_autoExpand.value(); }
   int getVisLevel()         const  {return m_visLevel.value(); }
   bool getIgnoreVisLevelWhenFilter() const  {return m_visLevelFilter.value(); }

   int getTopNodeIdx() const { return m_topNodeIdx.value(); }
   FWGeometryTableManager*  getTableManager() { return m_tableManager;} 
   virtual void setFrom(const FWConfiguration&);

   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   virtual void saveImageTo( const std::string& iName ) const {}
   void nodeColorChangeRequested(Color_t);

   void setBackgroundColor();
   void populate3DViewsFromConfig();

private:
   FWGeometryTableView(const FWGeometryTableView&);
   const FWGeometryTableView& operator=(const FWGeometryTableView&);


#ifndef __CINT__ 
   FWEnumParameter         m_mode;
   FWStringParameter       m_filter; 
   FWLongParameter         m_autoExpand;
   FWLongParameter         m_visLevel;
   FWBoolParameter         m_visLevelFilter;      
   FWLongParameter         m_topNodeIdx;  
#endif

   FWColorManager         *m_colorManager;
   FWTableWidget          *m_tableWidget;
   FWGeometryTableManager *m_tableManager;

   TGCompositeFrame       *m_settersFrame;
   FWGeoTopNode           *m_eveTopNode;

   FWColorPopup           *m_colorPopup;

   TEveWindowFrame*        m_eveWindow;
   TGCompositeFrame*       m_frame;

   FWViewCombo*            m_viewBox;

   FWGUIValidatingTextEntry* m_filterEntry;
   FWGeoMaterialValidator*   m_filterValidator;

   const FWConfiguration*  m_viewersConfig;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   void resetSetters();
   void makeSetter(TGCompositeFrame* frame, FWParameterBase* param);
   void loadGeometry();

   void autoExpandChanged();
   void modeChanged();
   void refreshTable3D();

   ClassDef(FWGeometryTableView, 0);
};

#endif

