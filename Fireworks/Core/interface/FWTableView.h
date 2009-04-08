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
// $Id: FWTableView.h,v 1.3 2009/04/08 15:07:53 jmuelmen Exp $
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
class FWTableViewManager;
class FWEveValueScaler;
class TEveWindowFrame;
class TEveWindowSlot;
class FWTableViewManager;

class FWTableView : public FWViewBase {

public:
     FWTableView(TEveWindowSlot *, const FWTableViewManager *);
     virtual ~FWTableView();

     // ---------- const member functions ---------------------
     TGFrame* frame() const;
     const std::string& typeName() const;
     virtual void addTo(FWConfiguration&) const;

     virtual void saveImageTo(const std::string& iName) const;

     // ---------- static member functions --------------------
     static const std::string& staticTypeName();

     // ---------- member functions ---------------------------
     virtual void setFrom(const FWConfiguration&);
     void setBackgroundColor(Color_t);
     void updateItems ();

private:
     FWTableView(const FWTableView&);    // stop default
     const FWTableView& operator=(const FWTableView&);    // stop default

     // ---------- member data --------------------------------
     TEveWindowFrame *m_frame;
     TGComboBox *m_collection;
     const FWTableViewManager *m_manager;
};


#endif
