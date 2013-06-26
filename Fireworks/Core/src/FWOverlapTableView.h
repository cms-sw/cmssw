#ifndef Fireworks_Core_FWOverlapTableView_h
#define Fireworks_Core_FWOverlapTableView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWOverlapTableView
// 
/**\class FWOverlapTableView FWOverlapTableView.h Fireworks/Core/interface/FWOverlapTableView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Jan  4 00:06:31 CET 2012
// $Id: FWOverlapTableView.h,v 1.6 2013/04/14 20:41:07 amraktad Exp $
//

#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"


class FWOverlapTableManager;
class TEvePointSet;
class FWEveOverlap;
//class FWGUIValidatingTextEntry;
//class FWGeoPathValidator;
class TGNumberEntry;
class RGTextButton;
class TGCheckButton;

class FWOverlapTableView : public FWGeometryTableViewBase
{
public:

   FWOverlapTableView(TEveWindowSlot* iParent, FWColorManager* colMng);
   virtual ~FWOverlapTableView();

   virtual  FWGeometryTableManagerBase*  getTableManager(); 
   

   void precisionCallback(Long_t);
   void recalculate();

   virtual void setFrom(const FWConfiguration&);
   virtual void populateController(ViewerParameterGUI&) const;

   virtual void cdTop();
   virtual void cdUp();

   void drawPoints();
   void pointSize();

   bool listAllNodes() const;
   void setListAllNodes();
   virtual void chosenItem(int x);

protected:
   virtual TEveElement* getEveGeoElement() const;

private:  
  
   FWOverlapTableView(const FWOverlapTableView&); // stop default
   const FWOverlapTableView& operator=(const FWOverlapTableView&); // stop default
   
   void setCheckerState(bool);
   TGTextButton* m_applyButton;
   TGCheckButton* m_listOptionButton;  

public:
   // ---------- member data --------------------------------

   FWOverlapTableManager *m_tableManager;
   TGNumberEntry*  m_numEntry;

   bool            m_runChecker;
   virtual void    refreshTable3D();

  
#ifndef __CINT__
   FWStringParameter       m_path; 
   FWDoubleParameter     m_precision;

   FWBoolParameter         m_listAllNodes;

   FWBoolParameter         m_rnrOverlap;
   FWBoolParameter         m_rnrExtrusion;

   FWBoolParameter         m_drawPoints;
   FWLongParameter        m_pointSize;
   FWLongParameter        m_extrusionMarkerColor;
   FWLongParameter        m_overlapMarkerColor;

   
#endif
   ClassDef(FWOverlapTableView, 0);
};


#endif
