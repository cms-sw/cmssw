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
   ~FWOverlapTableView() override;

    FWGeometryTableManagerBase*  getTableManager() override; 
   

   void precisionCallback(Long_t);
   void recalculate();

   void setFrom(const FWConfiguration&) override;
   void populateController(ViewerParameterGUI&) const override;

   void cdTop() override;
   void cdUp() override;

   void drawPoints();
   void pointSize();

   bool listAllNodes() const;
   void setListAllNodes();
   void chosenItem(int x) override;

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
   void    refreshTable3D() override;

  
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
   ClassDefOverride(FWOverlapTableView, 0);
};


#endif
