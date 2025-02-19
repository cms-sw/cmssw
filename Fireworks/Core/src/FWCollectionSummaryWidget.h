#ifndef Fireworks_Core_FWCollectionSummaryWidget_h
#define Fireworks_Core_FWCollectionSummaryWidget_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryWidget
// 
/**\class FWCollectionSummaryWidget FWCollectionSummaryWidget.h Fireworks/Core/interface/FWCollectionSummaryWidget.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Feb 14 10:02:30 CST 2009
// $Id: FWCollectionSummaryWidget.h,v 1.8 2010/06/16 14:04:40 matevz Exp $
//

// system include files
#include "TGFrame.h"

// user include files

// forward declarations
class TGPictureButton;
class TGCheckButton;
class TGLayoutHints;
class FWEventItem;
class FWColorSelect;
class FWBoxIconButton;
class FWBoxIconBase;
class FWCheckBoxIcon;
class FWColorBoxIcon;
class FWCustomIconsButton;
class FWColorPopup;
class FWCollectionSummaryTableManager;
class FWTableWidget;
class TGGC;
class TGTextButton;

namespace fireworks {
   class Context;
}

class FWCollectionSummaryWidgetConnectionHolder;

class FWCollectionSummaryWidget : public TGCompositeFrame
{

public:
   FWCollectionSummaryWidget(TGFrame* iParent, FWEventItem& iItem, TGLayoutHints*);
   virtual ~FWCollectionSummaryWidget();
   
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void displayChanged();
   void itemChanged();
   
   //pass true if white or false if black
   void setBackgroundToWhite(bool);
   
   void colorChangeRequested(Color_t iColorIndex);
   void toggleItemVisible();
   void toggleShowHide();
   void colorClicked();
   
   void requestForInfo(FWEventItem*); // *SIGNAL*
   void requestForFilter(FWEventItem*); // *SIGNAL*
   void requestForErrorInfo(FWEventItem*); // *SIGNAL*
   void requestForController(FWEventItem*); // *SIGNAL*
   void requestForModelContextMenu(Int_t,Int_t); // *SIGNAL*
   
   void infoClicked();
   void stateClicked();
   void labelClicked();
   
   ClassDef(FWCollectionSummaryWidget,0);
   
   void itemColorClicked(int iIndex, Int_t iRootX, Int_t iRootY);
   void modelSelected(Int_t iRow,Int_t iButton,Int_t iKeyMod, Int_t iGlobalX, Int_t iGlobalY);

private:
   FWCollectionSummaryWidget(const FWCollectionSummaryWidget&); // stop default
   
   const FWCollectionSummaryWidget& operator=(const FWCollectionSummaryWidget&); // stop default
   
   void createColorPopup();
   void colorTable();
   // ---------- member data --------------------------------
   FWEventItem* m_collection;
   TGLayoutHints* m_hints;
   TGFrame* m_parent;
   FWCustomIconsButton* m_showHideButton;
   FWBoxIconButton* m_isVisibleButton;
   FWCheckBoxIcon* m_isVisibleCheckBox;
   //FWColorSelect* m_colorSelectWidget;
   FWBoxIconButton* m_colorSelectWidget;
   FWColorBoxIcon* m_colorSelectBox;
   FWCustomIconsButton* m_stateButton;
   FWCustomIconsButton* m_infoButton;
   TGTextButton* m_label;
   TGFrame* m_holder;
   bool m_collectionShown;
   TGGC* m_graphicsContext;
   TGGC* m_tableContext;
   int m_indexForColor;
   FWColorPopup* m_colorPopup;
   
   FWCollectionSummaryTableManager* m_tableManager;
   FWTableWidget* m_tableWidget;
   bool m_backgroundIsWhite;
   
   FWCollectionSummaryWidgetConnectionHolder* m_connectionHolder;
};


#endif
