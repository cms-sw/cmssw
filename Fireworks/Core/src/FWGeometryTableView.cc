#include <iostream>
#include <boost/bind.hpp>
#include <boost/regex.hpp>

#include "Fireworks/Core/interface/FWGeometryTableView.h"
#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWValidatorBase.h"

#include "Fireworks/Core/interface/fwLog.h"

#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "TGComboBox.h"
#include "KeySymbols.h"

// #define PERFTOOL_BROWSER
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveScene.h"
#include "TEveSceneInfo.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#ifdef PERFTOOL_BROWSER 
#include <google/profiler.h>
#endif

bool geodebug = 0;

enum GeoMenuOptions {
   kSetTopNode,
   kSetTopNodeCam,
   kVisOn,
   kVisOff,
   kInspectMaterial,
   kInspectShape,
   kCamera,
   kTableDebug
};


class FWGeoMaterialValidator : public FWValidatorBase {
public:
   struct Material
   {
      TGeoMaterial* g;
      std::string n;
      bool operator< (const Material& x) const { return n < x.n ;}
      Material( TGeoMaterial* x) {  g= x; n = x ? x->GetName() : "<show-all>";}
   };

   FWGeometryTableManager* m_table;
   mutable std::vector<Material> m_list;

   FWGeoMaterialValidator( FWGeometryTableManager* t) { m_table = t;}
   virtual ~FWGeoMaterialValidator() {}


   virtual void fillOptions(const char* iBegin, const char* iEnd, std::vector<std::pair<boost::shared_ptr<std::string>, std::string> >& oOptions) const 
   {
      oOptions.clear();
      std::string part(iBegin,iEnd);
      unsigned int part_size = part.size();

      m_list.clear();
      m_list.push_back(0);

      FWGeometryTableManager::Entries_i it = m_table->refSelected();
      int nLevel = it->m_level;
      it++;
      while (it->m_level > nLevel)
      {
         TGeoMaterial* g = it->m_node->GetVolume()->GetMaterial();
         bool duplicate = false;
         for (std::vector<Material>::iterator j = m_list.begin(); j!=m_list.end(); ++j) {
            if (j->g == g) {
               duplicate = true;
               break;
            }
         }
         if (!duplicate)
            m_list.push_back(g);

         ++it;
      }
      std::vector<Material>::iterator startIt = m_list.begin();
      startIt++;
      std::sort(startIt, m_list.end());

      std::string h = "";
      oOptions.push_back(std::make_pair(boost::shared_ptr<std::string>(new std::string(m_list.begin()->n)), h));
      for (std::vector<Material>::iterator i = startIt; i!=m_list.end(); ++i) {
         if(part == (*i).n.substr(0,part_size) )
         {
            //  std::cout << i->n <<std::endl;
            oOptions.push_back(std::make_pair(boost::shared_ptr<std::string>(new std::string((*i).n)), (*i).n.substr(part_size, (*i).n.size()-part_size)));
         }
      }

   }

   bool isStringValid(std::string& exp) 
   {
      if (exp.empty()) return true;

      for (std::vector<Material>::iterator i = m_list.begin(); i!=m_list.end(); ++i) {
         if (exp == (*i).n) 
            return true;
    
      }
      return false;
   }
};

//______________________________________________________________________________

class FWViewCombo : public TGTextButton
{
private:
   FWGeometryTableView* m_tableView;
   TEveElement* m_el;

public:
   FWViewCombo(const TGWindow *p, FWGeometryTableView* t): 
      TGTextButton(p, "Select Views", -1, TGButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame | kDoubleBorder  ), m_tableView(t), m_el(0) {}
   virtual ~FWViewCombo() {}
   void setElement(TEveElement* x) {m_el = x;}

   virtual Bool_t  HandleButton(Event_t* event) 
   {
      if (event->fType == kButtonPress)
      {
         bool map = false;

         FWPopupMenu* m_viewPopup = new FWPopupMenu(0);

         TEveElementList* views = gEve->GetViewers();
         int idx = 0;

         for (TEveElement::List_i it = views->BeginChildren(); it != views->EndChildren(); ++it)
         { 
            TEveViewer* v = ((TEveViewer*)(*it));
            if (strstr( v->GetElementName(), "3D") )
            {     
               bool added = false;          
               m_viewPopup->AddEntry(v->GetElementName(), idx);
               TEveSceneInfo* si = ( TEveSceneInfo*)v->FindChild(Form("SI - EventScene %s",v->GetElementName() ));
               if (m_el) {
                  for (TEveElement::List_i it = m_el->BeginParents(); it != m_el->EndParents(); ++it ){
                     if (*it == si->GetScene()) {
                        added = true;
                        break;
                     }
                  }
               }
               map = true;
               if (added)
                  m_viewPopup->CheckEntry(idx);
            }
            ++idx;
         }

         if (map) {

            Window_t wdummy;
            Int_t ax,ay;
            gVirtualX->TranslateCoordinates(GetId(),
                                            gClient->GetDefaultRoot()->GetId(),
                                            event->fX, event->fY, //0,0 in local coordinates
                                            ax,ay, //coordinates of screen
                                            wdummy);


            m_viewPopup->PlaceMenu(ax, ay, true,true);
            m_viewPopup->Connect("Activated(Int_t)",
                                 "FWGeometryTableView",
                                 const_cast<FWGeometryTableView*>(m_tableView),
                                 "selectView(Int_t)");
         }
         else
         {
            fwLog(fwlog::kError) << "No 3D View added. \n";
         }
      }
      return true;
   }

};

//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================

FWGeometryTableView::FWGeometryTableView(TEveWindowSlot* iParent,FWColorManager* colMng, TGeoNode* tn, TObjArray* volumes )
   : FWViewBase(FWViewType::kGeometryTable),
     m_mode(this, "Mode:", 0l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"ExpandList:", 1l, 0l, 100l),
     m_visLevel(this,"VisLevel:", 3l, 1l, 100l),
     m_visLevelFilter(this,"IgnoreVisLevelOnFilter", true),
     m_topNodeIdx(this, "TopNodeIndex", -1l, 0, 1e7),
     m_colorManager(colMng),
     m_tableManager(0),
     m_eveTopNode(0),
     m_colorPopup(0),
     m_eveWindow(0),
     m_frame(0),
     m_viewBox(0),
     m_filterEntry(0),
     m_filterValidator(0),
     m_viewersConfig(0)
{
   m_eveWindow = iParent->MakeFrame(0);
   TGCompositeFrame* xf = m_eveWindow->GetGUICompositeFrame();

   m_frame = new TGVerticalFrame(xf);
   xf->AddFrame(m_frame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);
   m_mode.changed_.connect(boost::bind(&FWGeometryTableView::modeChanged,this));
   m_autoExpand.changed_.connect(boost::bind(&FWGeometryTableView::autoExpandChanged, this));
   m_visLevel.changed_.connect(boost::bind(&FWGeometryTableView::refreshTable3D,this));
   m_visLevelFilter.changed_.connect(boost::bind(&FWGeometryTableView::refreshTable3D,this));


   // top row
   {
      TGHorizontalFrame* hp =  new TGHorizontalFrame(m_frame);
 
      if (0) { TGTextButton* fileOpen = new TGTextButton (hp, "Open Geometry File");
         hp->AddFrame(fileOpen);
         fileOpen->Connect("Clicked()","FWGeometryTableView",this,"browse()");
      }
      {
         TGTextButton* rb = new TGTextButton (hp, "CdTop");
         hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0) );
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdTop()");
      } {
         TGTextButton* rb = new TGTextButton (hp, "CdUp");
         hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdUp()");
      }

      {
         m_viewBox = new FWViewCombo(hp, this);
         hp->AddFrame( m_viewBox,new TGLayoutHints(kLHintsExpandY, 2, 2, 0, 0));
      }
      {
         hp->AddFrame(new TGLabel(hp, "Filter:"), new TGLayoutHints(kLHintsBottom, 10, 0, 0, 2));
         m_filterEntry = new FWGUIValidatingTextEntry(hp);
         m_filterEntry->SetHeight(20);
         m_filterValidator = new FWGeoMaterialValidator(m_tableManager);
         m_filterEntry->setValidator(m_filterValidator);
         hp->AddFrame(m_filterEntry, new TGLayoutHints(kLHintsExpandX,  1, 2, 1, 0));
         m_filterEntry->setMaxListBoxHeight(150);
         m_filterEntry->getListBox()->Connect("Selected(int)", "FWGeometryTableView",  this, "filterListCallback()");
         m_filterEntry->Connect("ReturnPressed()", "FWGeometryTableView",  this, "filterTextEntryCallback()");

         gVirtualX->GrabKey( m_filterEntry->GetId(),gVirtualX->KeysymToKeycode((int)kKey_A),  kKeyControlMask, true);
      }
      m_frame->AddFrame(hp,new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 4, 2, 2, 0));
   }

   m_settersFrame = new TGHorizontalFrame(m_frame);
   m_frame->AddFrame( m_settersFrame, new TGLayoutHints(kLHintsExpandX,4,2,2,2));
   m_settersFrame->SetCleanup(kDeepCleanup);

   m_tableWidget = new FWTableWidget(m_tableManager, m_frame); 
   m_frame->AddFrame(m_tableWidget,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,2,2,0,0));
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGeometryTableView",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->disableGrowInWidth();
   resetSetters();

   if (tn)
   {
      m_tableManager->loadGeometry(tn, volumes);
      cdTop();
   }

   m_frame->MapSubwindows();
   m_frame->Layout();
   xf->Layout();
   m_frame->MapWindow();
}

FWGeometryTableView::~FWGeometryTableView()
{
   if (m_eveTopNode)
   {
      while ( m_eveTopNode->HasParents()) {
         TEveElement* x =  *m_eveTopNode->BeginParents();
        x->RemoveElement(m_eveTopNode);
      }
      m_eveTopNode->DecDenyDestroy();
   }

   // take out composite frame and delete it directly (zwithout the timeout)
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();
   frame->RemoveFrame( m_frame );
   delete m_frame;

   m_eveWindow->DestroyWindowAndSlot();
   delete m_tableManager;
}

//==============================================================================

void
FWGeometryTableView::addTo(FWConfiguration& iTo) const
{
   FWConfigurableParameterizable::addTo(iTo);

   FWConfiguration viewers(1);
   if (m_eveTopNode)
   { 
      for (TEveElement::List_i it = m_eveTopNode->BeginParents(); it != m_eveTopNode->EndParents(); ++it )
      {
         FWConfiguration tempArea;
         TEveScene* scene = dynamic_cast<TEveScene*>(*it);
         std::string n = scene->GetElementName();
         viewers.addKeyValue(n, tempArea);
      }
   }
   iTo.addKeyValue("Viewers", viewers, true);
}
  
void
FWGeometryTableView::setFrom(const FWConfiguration& iFrom)
{ 
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);

   }  
   m_filterEntry->SetText(m_filter.value().c_str(), false);
   resetSetters();
   cdNode(m_topNodeIdx.value());
   m_viewersConfig = iFrom.valueForKey("Viewers");
}

void
FWGeometryTableView::populate3DViewsFromConfig()
{
   // post-config 
   if (m_viewersConfig) {
      TEveElementList* scenes = gEve->GetScenes();
      const FWConfiguration::KeyValues* keyVals = m_viewersConfig->keyValues();
      if(0!=keyVals) 
      {
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it) {
    
            TString sname = it->first;
            TEveElement* s = scenes->FindChild(sname);
            if (s)
            {
               // std::cout << sname.Data() << std::endl;   
               if (!m_eveTopNode) {
                  m_eveTopNode = new FWGeoTopNode(this);
                  m_eveTopNode->IncDenyDestroy();
                  m_viewBox->setElement(m_eveTopNode);
               }
               s->AddElement(m_eveTopNode);
            }
         }   
      }
   }
}

void
FWGeometryTableView::resetSetters()
{

   if (!m_settersFrame->GetList()->IsEmpty())
   {
      m_setters.clear();

      TGFrameElement *el = (TGFrameElement*) m_settersFrame->GetList()->First();
      TGHorizontalFrame* f = (TGHorizontalFrame*) el->fFrame;
      m_settersFrame->RemoveFrame(f);
   }

   TGHorizontalFrame* frame =  new TGHorizontalFrame(m_settersFrame);
   m_settersFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX,4,2,2,2) );
   m_settersFrame->SetCleanup(kDeepCleanup);
   makeSetter(frame, &m_mode);
   makeSetter(frame, &m_autoExpand);
   makeSetter(frame, &m_visLevel);
   makeSetter(frame, &m_visLevelFilter);
   m_settersFrame->MapSubwindows();
   m_frame->Layout();
}

void
FWGeometryTableView::makeSetter(TGCompositeFrame* frame, FWParameterBase* param) 
{
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(param) );
   ptr->attach(param, this);
 
   TGFrame* m_frame = ptr->build(frame, false);
   frame->AddFrame(m_frame, new TGLayoutHints(kLHintsExpandX));

   m_setters.push_back(ptr);
}


//==============================================================================

void 
FWGeometryTableView::selectView(int idx)
{
   TEveElement::List_i it = gEve->GetViewers()->BeginChildren();
   std::advance(it, idx);
   TEveViewer* v = (TEveViewer*)(*it);
   TEveSceneInfo* si = (TEveSceneInfo*)v->FindChild(Form("SI - EventScene %s",v->GetElementName()));

   bool added = false;
   if (!m_eveTopNode) {
      m_eveTopNode = new FWGeoTopNode(this);
      m_eveTopNode->IncDenyDestroy();
      m_viewBox->setElement(m_eveTopNode);
   }
   else
   {
      for (TEveElement::List_i it = m_eveTopNode->BeginParents(); it != m_eveTopNode->EndParents(); ++it ){
         if (*it == si->GetScene()) {
            added = true;
            break;
         }
      }
   }
   printf("add node %s \n", si->GetElementName());

   if (added)
      si->GetScene()->RemoveElement(m_eveTopNode);
   else
      si->GetScene()->AddElement(m_eveTopNode);

   m_eveTopNode->ElementChanged();
   gEve->Redraw3D();
}



//==============================================================================
void 
FWGeometryTableView::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t x, Int_t y)
{
   m_tableManager->setSelection(iRow, iColumn, iButton);
   FWGeometryTableManager::NodeInfo& ni = *m_tableManager->refSelected();

   if (iButton == kButton1) 
   {
      if (iColumn == FWGeometryTableManager::kName)
      {
         m_tableManager->firstColumnClicked(iRow);
         return;
      }
      else if (iColumn == FWGeometryTableManager::kColor)
      { 
         std::vector<Color_t> colors;
         m_colorManager->fillLimitedColors(colors);
      
         if (!m_colorPopup) {
            m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), colors.front());
            m_colorPopup->InitContent("", colors);
            m_colorPopup->Connect("ColorSelected(Color_t)","FWGeometryTableView", const_cast<FWGeometryTableView*>(this), "nodeColorChangeRequested(Color_t)");
         }
         m_colorPopup->SetName("Selected");
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         m_colorPopup->PlacePopup(x, y, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
         return;
      }
      else
      {
         bool elementChanged = false;
         if (iColumn ==  FWGeometryTableManager::kVisSelf)
         {
            m_tableManager->setVisibility(ni, !m_tableManager->getVisibility(ni));
            elementChanged = true;
         }
         if (iColumn ==  FWGeometryTableManager::kVisChild)
         { 
            m_tableManager->setVisibilityChld(ni, !m_tableManager->getVisibilityChld(ni));; 
            elementChanged = true;
         }


         if (m_eveTopNode && elementChanged)
         {
            m_eveTopNode->ElementChanged();
            gEve->RegisterRedraw3D();
         }
      }
        

      m_tableManager->dataChanged();

   }
   else if (iColumn == FWGeometryTableManager::kName)
   {
      FWPopupMenu* m_nodePopup = new FWPopupMenu();
      m_nodePopup->AddEntry("Set As Top Node", kSetTopNode);
      m_nodePopup->AddEntry("Set As Top Node And Camera Center", kSetTopNodeCam);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("Rnr Off For All Children", kVisOff);
      m_nodePopup->AddEntry("Rnr On For All Children", kVisOn);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("Set Camera Center", kCamera);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("InspectMaterial", kInspectMaterial);
      m_nodePopup->AddEntry("InspectShape", kInspectShape);
     m_nodePopup->AddEntry("Table Debug", kTableDebug);

      m_nodePopup->PlaceMenu(x,y,true,true);
      m_nodePopup->Connect("Activated(Int_t)",
                           "FWGeometryTableView",
                           const_cast<FWGeometryTableView*>(this),
                           "chosenItem(Int_t)");
   }
}

void FWGeometryTableView::chosenItem(int x)
{
   FWGeometryTableManager::NodeInfo& ni = *m_tableManager->refSelected();
   TGeoVolume* gv = ni.m_node->GetVolume();
   bool visible = true;
   bool resetHome = false;
   if (gv)
   {
      switch (x) {
        case kVisOff:
            visible = false;
        case kVisOn:
           m_tableManager->setDaughtersSelfVisibility(visible);
            refreshTable3D();
            break;

         case kInspectMaterial:
            gv->InspectMaterial();
            break;
         case kInspectShape:
            gv->InspectShape();
            break;
         case kTableDebug:
            // std::cout << "node name " << ni.name() << "parent " <<m_tableManager->refEntries()[ni.m_parent].name() <<  std::endl;
            // printf("node expanded [%d] imported[%d] children[%d]\n", ni.m_expanded,m_tableManager->nodeImported(m_tableManager->m_selectedIdx) ,  ni.m_node->GetNdaughters());
            //            m_tableManager->printChildren(
            // m_tableManager->m_selectedIdx);
            m_tableManager->printMaterials();
            break;

         case kSetTopNode:
            cdNode(m_tableManager->m_selectedIdx);
            break;

         case kSetTopNodeCam:
            cdNode(m_tableManager->m_selectedIdx);
            resetHome = true;
         case kCamera:
         {
            TGeoHMatrix mtx;
            m_tableManager->getNodeMatrix( ni, mtx);

            static double pnt[3];
            TGeoBBox* bb = static_cast<TGeoBBox*>( ni.m_node->GetVolume()->GetShape());
            const double* origin = bb->GetOrigin();
            mtx.LocalToMaster(origin, pnt);

            TEveElementList* vl = gEve->GetViewers();
            for (TEveElement::List_i it = vl->BeginChildren(); it != vl->EndChildren(); ++it)
            {
               TEveViewer* v = ((TEveViewer*)(*it));
               TString name = v->GetElementName();
               if (name.Contains("3D"))
               {
                  v->GetGLViewer()->SetDrawCameraCenter(true);
                  TGLCamera& cam = v->GetGLViewer()->CurrentCamera();
                  cam.SetExternalCenter(true);
                  cam.SetCenterVec(pnt[0], pnt[1], pnt[2]);
               }
            }
            if (resetHome) gEve->FullRedraw3D(true, true);
            break;
         }
      }
   }
}

void FWGeometryTableView::setBackgroundColor()
{
   bool backgroundIsWhite = m_colorManager->backgroundColorIndex()==FWColorManager::kWhiteIndex;
   if(backgroundIsWhite) {
      m_tableWidget->SetBackgroundColor(0xffffff);
      m_tableWidget->SetLineSeparatorColor(0x000000);
   } else {
      m_tableWidget->SetBackgroundColor(0x000000);
      m_tableWidget->SetLineSeparatorColor(0xffffff);
   }
   m_tableManager->setBackgroundToWhite(backgroundIsWhite);
   gClient->NeedRedraw(m_tableWidget);
}

void FWGeometryTableView::nodeColorChangeRequested(Color_t col)
{
   FWGeometryTableManager::NodeInfo& ni = *m_tableManager->refSelected();
   ni.m_color = col;
   ni.m_node->GetVolume()->SetLineColor(col);
   refreshTable3D();
}

void
FWGeometryTableView::printTable()
{
   // print all entries
   m_tableManager->printChildren(-1);
}

//______________________________________________________________________________


void FWGeometryTableView::cdNode(int idx)
{
   std::string p;
   m_tableManager->getNodePath(idx, p);
   setPath(idx, p);
}

void FWGeometryTableView::cdTop()
{
   std::string path = "/" ;
   path += m_tableManager->refEntries().at(0).name();
   setPath(-1, path ); 
}

void FWGeometryTableView::cdUp()
{   
   if ( getTopNodeIdx() != -1)
   {
      int pIdx   = m_tableManager->refEntries()[getTopNodeIdx()].m_parent;
      std::string p;
      m_tableManager->getNodePath(pIdx, p);
      setPath(pIdx, p);
   }
}

void FWGeometryTableView::setPath(int parentIdx, std::string& path)
{
   m_topNodeIdx.set(parentIdx);
#ifdef PERFTOOL_BROWSER  
   ProfilerStart(Form("cdPath%d.prof", parentIdx));
#endif

   m_tableManager->topGeoNodeChanged(parentIdx);
   m_tableManager->updateFilter();
   m_tableManager->checkExpandLevel();

   refreshTable3D();
   // printf("END Set Path to [%s], curren node %s \n", m_path.value().c_str(), topNode->GetName()); 

   m_tableManager->redrawTable();
   if ( m_eveTopNode) {
      m_eveTopNode->ElementChanged();
      gEve->FullRedraw3D(false, true);
   } 

#ifdef PERFTOOL_BROWSER  
   ProfilerStop();
#endif 
}
//______________________________________________________________________________
void FWGeometryTableView::filterTextEntryCallback()
{
   // std::cout << "text entry click ed \n" ;
   std::string exp = m_filterEntry->GetText();
   if ( m_filterValidator->isStringValid(exp)) 
   {
      updateFilter(exp);
   }
   else
   {
      fwLog(fwlog::kError) << "filter expression not valid." << std::endl;
      return;
   }
}

void FWGeometryTableView::filterListCallback()
{ 
   //   std::cout << "list click ed \n" ;
   TGListBox* list =   m_filterEntry->getListBox();
   TList selected;
   list->GetSelectedEntries(&selected);
   if (selected.GetEntries() == 1)
   {
      const TGLBEntry* entry = dynamic_cast<TGLBEntry*> (selected.First());
      updateFilter( m_filterValidator->m_list[ entry->EntryId()].n);
   } 
}



void FWGeometryTableView::updateFilter(std::string& exp)
{
   // std::cout << "=FWGeometryTableView::updateFilter()" << m_filterEntry->GetText() <<std::endl;
  
   if (exp == m_filterValidator->m_list.begin()->n) 
      exp.clear();

   if (exp == m_filter.value()) return;

   if (exp.empty())
   {
      // std::cout << "FITLER OFF \n";
      for (FWGeometryTableManager::Entries_i i = m_tableManager->refEntries().begin(); i !=  m_tableManager->refEntries().end(); ++i)
      {
         m_tableManager->setVisibility(*i, true);
         m_tableManager->setVisibilityChld(*i, true);
      }

      // NOTE: entry should be cleared automatically
      m_filterEntry->Clear();

      m_tableManager->checkExpandLevel();
   }
  
   m_filter.set(exp);
   m_tableManager->updateFilter();
   refreshTable3D();

}

//______________________________________________________________________________
void FWGeometryTableView::modeChanged()
{
   // reset filter when change mode
   //   std::cout << "chage mode \n";
   m_tableManager->updateFilter();
   refreshTable3D();
}

//______________________________________________________________________________

void FWGeometryTableView::autoExpandChanged()
{
   m_tableManager->checkExpandLevel();
   m_tableManager->redrawTable();
}
//______________________________________________________________________________

void FWGeometryTableView::refreshTable3D()
{
   m_tableManager->redrawTable();

   if ( m_eveTopNode) {
      m_eveTopNode->ElementChanged();
      gEve->FullRedraw3D(false, true);
   } 
}

