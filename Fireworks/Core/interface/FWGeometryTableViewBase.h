#ifndef Fireworks_Core_FWGeometryTableViewBase_h
#define Fireworks_Core_FWGeometryTableViewBase_h

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#endif


#include "Rtypes.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "Fireworks/Core/interface/FWViewType.h"
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


class TGeoNode;
class TGeoVolume;
class TGStatusBar;
class TGeoManager;
class TEveWindowSlot;
class TEveWindowFrame;
class TEveElement;
class FWEveDigitSetScalableMarker;
class TEveScene;

class FWTableWidget;
class FWGeometryTableManagerBase;
class FWConfiguration;
class FWColorPopup;
class FWColorManager;
class FWGeoTopNode;
class FWParameterBase;

class FWGeometryTableViewBase
#ifndef __CINT__
   : public  FWViewBase,
     public  FWParameterSetterEditorBase
#endif
{
public:
   class FWViewCombo : public TGTextButton
   {
   private:
      FWGeometryTableViewBase* m_tableView;
      TEveElement* m_el;
   public:
      FWViewCombo(const TGWindow *p, FWGeometryTableViewBase* t): 
         TGTextButton(p, "Select Views", -1, TGButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame | kDoubleBorder  ), m_tableView(t), m_el(0) {}
      virtual ~FWViewCombo() {}
      void setElement(TEveElement* x) {m_el = x;}
      virtual Bool_t  HandleButton(Event_t* event);
   };


   FWGeometryTableViewBase(TEveWindowSlot*, FWViewType::EType, FWColorManager*);
   virtual ~FWGeometryTableViewBase();
  
   virtual  void cellClicked(Int_t iRow, Int_t iColumn, 
                             Int_t iButton, Int_t iKeyMod, 
                             Int_t iGlobalX, Int_t iGlobalY);
  
   // void chosenItemFrom3DView(int);
   virtual void chosenItem(int);
   void selectView(int);
 
   bool getEnableHighlight() { return m_enableHighlight.value(); } 
   virtual  FWGeometryTableManagerBase*  getTableManager() { return 0; }

   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   virtual void saveImageTo( const std::string& iName ) const {}
   void nodeColorChangeRequested(Color_t);

   void setBackgroundColor();
   void populate3DViewsFromConfig();
   virtual void refreshTable3D();


   void cdNode(int);
   virtual void cdTop();
   virtual void cdUp();
   virtual void setPath(int, std::string&);

   void checkExpandLevel();

   int getTopNodeIdx() const { return TMath::Max((int)m_topNodeIdx.value(), 0); }
  
   FWEveDigitSetScalableMarker* getMarker()  {return m_marker;}
   void transparencyChanged();
   
   void  reloadColors();
   
   long getParentTransparencyFactor() const { return m_parentTransparencyFactor.value(); }
   long getLeafTransparencyFactor()   const { return m_leafTransparencyFactor.value(); }
   long getMinParentTransparency() const { return m_minParentTransparency.value(); }
   long getMinLeafTransparency()   const { return m_minLeafTransparency.value(); }
   
protected:

#ifndef __CINT__      
   FWLongParameter         m_topNodeIdx; 
   FWLongParameter         m_autoExpand;
   FWBoolParameter         m_enableHighlight;
   
   FWLongParameter         m_parentTransparencyFactor;
   FWLongParameter         m_leafTransparencyFactor;
   FWLongParameter         m_minParentTransparency;
   FWLongParameter         m_minLeafTransparency;
#endif

   FWColorManager         *m_colorManager;
   FWTableWidget          *m_tableWidget;

   //  TGCompositeFrame       *m_settersFrame;

   FWColorPopup           *m_colorPopup;

   TEveWindowFrame*        m_eveWindow;
   TGCompositeFrame*       m_frame;

   FWViewCombo*            m_viewBox;


   const FWConfiguration*  m_viewersConfig;
  
   bool m_enableRedraw;
   
   FWEveDigitSetScalableMarker* m_marker;
   FWGeoTopNode* m_eveTopNode;
   TEveScene*    m_eveScene;

#ifndef __CINT__
   // std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   //   void resetSetters();
   //   void makeSetter(TGCompositeFrame* frame, FWParameterBase* param);

   void postConst();
   
   void setTopNodePathFromConfig(const FWConfiguration& iFrom);

   virtual void populateController(ViewerParameterGUI&) const;

private:
   int m_tableRowIndexForColorPopup;

   FWGeometryTableViewBase(const FWGeometryTableViewBase&);                  // stop default
   const FWGeometryTableViewBase& operator=(const FWGeometryTableViewBase&); // stop default
   void setColumnSelected(int idx);
   ClassDef(FWGeometryTableViewBase, 0);
};

#endif

