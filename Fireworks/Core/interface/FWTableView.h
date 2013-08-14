// -*- C++ -*-
#ifndef Fireworks_Core_FWTableView_h
#define Fireworks_Core_FWTableView_h
//
// Package:     Core
// Class  :     FWTableView
//
/**\class FWTableView FWTableView.h Fireworks/Core/interface/FWTableView.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWTableView.h,v 1.9 2011/01/26 11:57:32 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TGCompositeFrame;
class TGComboBox;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveGeoShape;
class TGLMatrix;
class TGTextEntry;
class FWEventItem;
class FWTableViewManager;
class FWTableWidget;
class TEveWindowFrame;
class TEveWindowSlot;
class FWTableViewManager;
class FWTableViewTableManager;
class FWCustomIconsButton;
class FWGUIValidatingTextEntry;
class FWExpressionValidator;

class FWTableView : public FWViewBase {
     friend class FWTableViewTableManager;

public:
     FWTableView(TEveWindowSlot *, FWTableViewManager *);
     virtual ~FWTableView();

     // ---------- const member functions ---------------------
     virtual void addTo(FWConfiguration&) const;

     virtual void saveImageTo(const std::string& iName) const;

     // ---------- static member functions --------------------

     // ---------- member functions ---------------------------
     virtual void setFrom(const FWConfiguration&);
     void setBackgroundColor(Color_t);
     void resetColors (const class FWColorManager &);
     void updateItems ();
     void updateEvaluators ();
     void selectCollection (Int_t);
     void dataChanged ();
     const FWEventItem *item () const;
     void modelSelected(Int_t iRow,Int_t iButton,Int_t iKeyMod,Int_t,Int_t);
     void columnSelected (Int_t iCol, Int_t iButton, Int_t iKeyMod);
     void toggleShowHide ();
     void addColumn ();
     void deleteColumn ();
     void modifyColumn ();

private:
     FWTableView(const FWTableView&);    // stop default
     const FWTableView& operator=(const FWTableView&);    // stop default

protected:
     // ---------- member data --------------------------------
     TEveWindowFrame *m_eveWindow;
     TGComboBox *m_collection;
     TGCompositeFrame *m_vert, *m_column_control;
     int m_iColl;
     FWTableViewManager *m_manager;
     FWTableViewTableManager *m_tableManager;
     FWTableWidget *m_tableWidget;
     bool m_showColumnUI;
     FWCustomIconsButton *m_columnUIButton;
     TGTextEntry *m_column_name_field;
     FWGUIValidatingTextEntry *m_column_expr_field;
     FWExpressionValidator *m_validator;
     TGTextEntry *m_column_prec_field;
     int m_currentColumn;
     bool m_useColumnsFromConfig;
};


#endif
