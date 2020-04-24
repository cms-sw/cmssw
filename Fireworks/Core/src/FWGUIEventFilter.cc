#include "TGLabel.h"
#include "TG3DLine.h"
#include "TGResourcePool.h"
#include "Fireworks/Core/interface/RootGuiUtils.h"
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"
#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"

FWGUIEventFilter::FWGUIEventFilter(CmsShowNavigator* n):
   TGMainFrame(gClient->GetRoot(), 560, 300),


   m_origFilterMode(CmsShowNavigator::kOr),
   m_isOpen(false),
   m_filtersRemoved(false),

   m_eventSelectionFrameParent(nullptr),
   m_eventSelectionFrame(nullptr),
   m_triggerSelectionFrameParent(nullptr),
   m_triggerSelectionFrame(nullptr),

   m_rad1(nullptr),
   m_rad2(nullptr),
   m_stateLabel(nullptr),
   m_disableFilteringBtn(nullptr),
   m_addBtn(nullptr),

   m_navigator(n)
{
   SetWindowName("Event Filters");

   TGVerticalFrame* v1 = new TGVerticalFrame(this);
   AddFrame(v1, new TGLayoutHints(kLHintsExpandX |kLHintsExpandY));


   //----------------------- Event selection
  
   {
      m_eventSelectionFrameParent =  new TGVerticalFrame(v1, GetWidth(), s_entryHeight, 0);
      v1->AddFrame(m_eventSelectionFrameParent, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 2, 2, 0,0));

      // headers
      TGHorizontalFrame* selH = new TGHorizontalFrame(m_eventSelectionFrameParent);
      m_eventSelectionFrameParent->AddFrame(selH, new TGLayoutHints(kLHintsExpandX));

      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH);
         selH->AddFrame(cfr, new TGLayoutHints(kLHintsExpandX));
         cfr->AddFrame(new TGLabel(cfr, "Event Filter Expression:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 6, 4));
      }
      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 122, 22, kFixedSize);
         selH->AddFrame(cfr);
         cfr->AddFrame(new TGLabel(cfr, "Comment:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 2, 0));
      }
      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 105, 22, kFixedSize);
         selH->AddFrame(cfr);
         cfr->AddFrame(new TGLabel(cfr, "Pass:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 2, 0));
      }


      TGHorizontalFrame* addBtnFrame = new TGHorizontalFrame(v1);
      v1->AddFrame(addBtnFrame, new TGLayoutHints(kLHintsExpandX));
      addBtnFrame->AddFrame(new TGHorizontal3DLine(addBtnFrame),  new TGLayoutHints(kLHintsExpandX | kLHintsCenterY,4 ,8, 2, 2));
    
      m_addBtn = new FWCustomIconsButton(addBtnFrame, fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign.png"),
                                         fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-over.png"),
                                         fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-disabled.png"));

      addBtnFrame->AddFrame(m_addBtn, new TGLayoutHints(kLHintsRight/*|kLHintsExpandX|kLHintsExpandY*/, 0, 6, 4, 1));
      TQObject::Connect(m_addBtn, "Clicked()", "FWGUIEventFilter",  this, "newEventEntry()");
   }

   //----------------------- TriggerResults selection

   {
      m_triggerSelectionFrameParent =  new TGVerticalFrame(v1, GetWidth(), s_entryHeight, 0);
      v1->AddFrame(m_triggerSelectionFrameParent, new TGLayoutHints(kLHintsExpandX|kLHintsTop, 2, 2, 0,0));

      // headers
      TGHorizontalFrame* selH = new TGHorizontalFrame(m_triggerSelectionFrameParent);
      m_triggerSelectionFrameParent->AddFrame(selH, new TGLayoutHints(kLHintsExpandX));

      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH);
         selH->AddFrame(cfr, new TGLayoutHints(kLHintsExpandX));
         cfr->AddFrame(new TGLabel(cfr, "TriggerResults Filter Expression:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 6, 04));
      }
      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 122, 22, kFixedSize);
         selH->AddFrame(cfr);
         cfr->AddFrame(new TGLabel(cfr, "Comment:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 0, 0));
      }
      {
         TGCompositeFrame *cfr = new TGHorizontalFrame(selH, 105, 22, kFixedSize);
         selH->AddFrame(cfr);
         cfr->AddFrame(new TGLabel(cfr, "Pass:"), new TGLayoutHints(kLHintsLeft|kLHintsBottom, 2, 2, 0, 0));
      }

      //-------------------- adding new selection

      TGHorizontalFrame* addBtnFrame = new TGHorizontalFrame(v1);
      v1->AddFrame(addBtnFrame, new TGLayoutHints(kLHintsExpandX));

      addBtnFrame->AddFrame(new TGHorizontal3DLine(addBtnFrame),  new TGLayoutHints(kLHintsExpandX | kLHintsCenterY,4 ,8, 2, 2));
      m_addBtn = new FWCustomIconsButton(addBtnFrame, fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign.png"),
                                         fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-over.png"),
                                         fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "plus-sign-disabled.png"));

      addBtnFrame->AddFrame(m_addBtn, new TGLayoutHints(kLHintsRight/*|kLHintsExpandX|kLHintsExpandY*/, 0, 6, 4, 1));
      TQObject::Connect(m_addBtn, "Clicked()", "FWGUIEventFilter",  this, "newTriggerEntry()");

   }

   //-------------------- logical operations

   TGHorizontalFrame* headerFrame = new TGHorizontalFrame(v1/*, 360, 61, kHorizontalFrame | kFixedSize*/);

   {
      TGHorizontalFrame* xx = new TGHorizontalFrame(v1);
      fireworks_root_gui::makeLabel(xx, "Combine Expression Width:", 152, 2,2,2,2);
      m_rad1 =  new TGRadioButton(xx, "OR", 81);
      xx->AddFrame(m_rad1, new TGLayoutHints(kLHintsNormal, 2,10, 0, 0));
      m_rad1->SetState(kButtonDown);
      m_rad2 =  new TGRadioButton(xx, "AND", 82);
      xx->AddFrame(m_rad2);
      m_rad1->Connect("Clicked()", "FWGUIEventFilter", this, "changeFilterMode()");
      m_rad2->Connect("Clicked()", "FWGUIEventFilter", this, "changeFilterMode()");

      v1->AddFrame(xx, new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));
   }
   v1->AddFrame(headerFrame, new TGLayoutHints(kLHintsNormal, 1, 1, 1, 1));


   //-------------------- status
   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(v1);
      v1->AddFrame(hf, new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 20));
      fireworks_root_gui::makeLabel(hf, "Status:", 37, 2,2,2,2);
      {     
         TGGC *fTextGC;
         const TGFont *font = gClient->GetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
         if (!font)
            font = gClient->GetResourcePool()->GetDefaultFont();
         GCValues_t   gval;
         gval.fMask = kGCBackground | kGCFont | kGCForeground;
         gval.fFont = font->GetFontHandle();
         fTextGC = gClient->GetGC(&gval, kTRUE);

         TGHorizontalFrame *labFrame = new TGHorizontalFrame(hf, 380, 22, kHorizontalFrame | kFixedWidth);
         hf->AddFrame(labFrame, new TGLayoutHints(kLHintsNormal));

         m_stateLabel = new TGLabel(labFrame, "x", fTextGC->GetGC());
         labFrame->AddFrame(m_stateLabel, new TGLayoutHints( kLHintsLeft,2,2,2,2));
                                     
      }
   }
   //-------------------- external actions

   TGHorizontalFrame* btnFrame = new TGHorizontalFrame(v1, 280, 30);
   v1->AddFrame(btnFrame, new TGLayoutHints(kLHintsCenterX | kLHintsExpandX | kLHintsBottom , 0, 0, 2, 4));

   TGTextButton* cancel = new TGTextButton(btnFrame," Close ");
   btnFrame->AddFrame(cancel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY , 20, 20, 2, 4));
   cancel->Connect("Clicked()","FWGUIEventFilter", this, "CloseWindow()");

   {
      TGHorizontalFrame* f = new TGHorizontalFrame(btnFrame);
      btnFrame->AddFrame(f, new TGLayoutHints(kLHintsRight, 4, 18, 2, 4));
      m_disableFilteringBtn = new TGTextButton(f, " Disable Filtering ");
      f->AddFrame(m_disableFilteringBtn, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 4, 10, 2, 4));
      m_disableFilteringBtn->Connect("Clicked()","FWGUIEventFilter", this, "disableFilters()");

      m_applyBtn = new TGTextButton(f,"Apply Filters");
      f->AddFrame(m_applyBtn, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 4, 8, 2, 4));
      m_applyBtn->Connect("Clicked()","FWGUIEventFilter", this, "apply()");
      m_applyBtn->SetToolTipText("Enable filtering and apply changes.");
   }


}

FWGUIEventFilter::~FWGUIEventFilter()
{
}

void
FWGUIEventFilter::addSelector(FWEventSelector* sel)
{
   TGCompositeFrame* parent = sel->m_triggerProcess.empty() ? m_eventSelectionFrame : m_triggerSelectionFrame;
   FWGUIEventSelector* es = new FWGUIEventSelector(parent, sel, m_navigator->getProcessList() );
   parent->AddFrame(es, new TGLayoutHints(kLHintsExpandX));
   TQObject::Connect(es, "removeSelector(FWGUIEventSelector*)", "FWGUIEventFilter",  this, "deleteEntry(FWGUIEventSelector*)");
   TQObject::Connect(es, "selectorChanged()", "FWGUIEventFilter",  this, "checkApplyButton()");

   m_guiSelectors.push_back(es);
 
}

void
FWGUIEventFilter::changeFilterMode()
{
   TGButton *btn = (TGButton *) gTQSender;

   if (btn->WidgetId() == 81)
      m_rad2->SetState(kButtonUp);
   else
      m_rad1->SetState(kButtonUp);
   checkApplyButton();
}

void
FWGUIEventFilter::show( std::list<FWEventSelector*>* sels, int filterMode, int filterState)
{
   m_applyBtn->SetForegroundColor(0x000000);
   m_filtersRemoved = false;

   m_isOpen = true;

   m_origFilterMode = filterMode;

   if (filterMode == CmsShowNavigator::kOr)
   {
      m_rad1->SetState(kButtonDown, false);
      m_rad2->SetState(kButtonUp, false);
   }
   else
   {
      m_rad2->SetState(kButtonDown, false);
      m_rad1->SetState(kButtonUp, false);
   }

   assert(m_eventSelectionFrame == nullptr);
   
   m_eventSelectionFrame = new TGVerticalFrame(m_eventSelectionFrameParent);
   m_eventSelectionFrameParent->AddFrame(m_eventSelectionFrame,  new TGLayoutHints(kLHintsExpandX));

   m_triggerSelectionFrame = new TGVerticalFrame(m_triggerSelectionFrameParent);
   m_triggerSelectionFrameParent->AddFrame(m_triggerSelectionFrame,  new TGLayoutHints(kLHintsExpandX));

   for(std::list<FWEventSelector*>::iterator i = sels->begin(); i != sels->end(); ++i)
      addSelector(*i);

   updateFilterStateLabel(filterState);

   MapSubwindows();
   Layout();
   MapRaised();
}

void
FWGUIEventFilter::reset()
{
   // called on load of configuration
   if (m_eventSelectionFrameParent) {
      m_eventSelectionFrameParent->RemoveFrame(m_eventSelectionFrame);
      m_eventSelectionFrame = nullptr;
   }
   if (m_triggerSelectionFrameParent) {
      m_triggerSelectionFrameParent->RemoveFrame(m_triggerSelectionFrame);
      m_triggerSelectionFrame = nullptr;
   }

   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
      delete *i;
   
   m_guiSelectors.clear();
}

///////////////////////////////////////////
//   Callbacks
///////////////////////////////////////////

void
FWGUIEventFilter::deleteEntry(FWGUIEventSelector* sel)
{
   m_filtersRemoved = true;

   m_guiSelectors.remove(sel);

   TGCompositeFrame* p = nullptr;
   if (sel->origSelector()->m_triggerProcess.empty())
      p = m_eventSelectionFrame;
   else
      p = m_triggerSelectionFrame;

   p->RemoveFrame(sel);
   Resize(GetWidth(), GetDefaultHeight());
   Layout();
   gClient->NeedRedraw(this);
}

void
FWGUIEventFilter::newTriggerEntry()
{ 
   FWEventSelector* s = new FWEventSelector;
   s->m_triggerProcess = "HLT";
   addSelector(s);
   MapSubwindows();

   Resize(GetWidth(), GetDefaultHeight());

   Layout();
}

void
FWGUIEventFilter::newEventEntry()
{
   addSelector(new FWEventSelector());
   MapSubwindows();
   Resize(GetWidth(), GetDefaultHeight());
   Layout();
}

void
FWGUIEventFilter::apply()
{
   m_navigator->applyFiltersFromGUI();

   m_origFilterMode = getFilterMode();
   m_filtersRemoved = false;
   m_applyBtn->SetForegroundColor(0x000000);
   fClient->NeedRedraw( this);

}

void
FWGUIEventFilter::setupDisableFilteringButton(bool x)
{
   m_disableFilteringBtn->SetEnabled(x);
}

void
FWGUIEventFilter::disableFilters()
{
   m_navigator->toggleFilterEnable();
}

int
FWGUIEventFilter::getFilterMode()
{
   if (m_rad1->IsOn())
      return CmsShowNavigator::kOr;
   else
      return CmsShowNavigator::kAnd;
}

void
FWGUIEventFilter::CloseWindow()
{
   m_isOpen = false;
   m_eventSelectionFrameParent->RemoveFrame(m_eventSelectionFrame);
   m_eventSelectionFrame = nullptr;
   m_triggerSelectionFrameParent->RemoveFrame(m_triggerSelectionFrame);
   m_triggerSelectionFrame = nullptr;

   FWGUIEventSelector* gs;
   for (std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin(); i != m_guiSelectors.end(); ++i)
   {
      gs = *i;
      delete gs;
   }

   m_guiSelectors.clear();
   UnmapWindow();
   m_navigator->editFiltersExternally();
}


void
FWGUIEventFilter::checkApplyButton()
{
   // set color of apply button if changed

   bool changed = ( m_filtersRemoved || (getFilterMode() != m_origFilterMode) );

   if (!changed)
   {
      std::list<FWGUIEventSelector*>::iterator i = m_guiSelectors.begin();
      while (i != m_guiSelectors.end())
      {
         if ((*i)->origSelector() == nullptr ||
             (*i)->guiSelector()->m_enabled    != (*i)->origSelector()->m_enabled  ||
             (*i)->guiSelector()->m_expression != (*i)->origSelector()->m_expression )
         {
            changed = true;
            break;
         }

         ++i;
      }
   }


   m_applyBtn->SetForegroundColor(changed ? 0x40FF80 : 0x000000);
   gClient->NeedRedraw( m_applyBtn);
}

void
FWGUIEventFilter::updateFilterStateLabel(int state)
{
   if (state == CmsShowNavigator::kOn)
      m_stateLabel->SetText(Form("%d events selected from %d   ", m_navigator->getNSelectedEvents(), m_navigator->getNTotalEvents()));
   else if (state == CmsShowNavigator::kOff)
      m_stateLabel->SetText("Filtering Disabled  ");
   else
      m_stateLabel->SetText("Filtering Withdrawn  ");

   Layout();
}

/*
  AMT  no effect after resize
void
FWGUIEventFilter::addTo(FWConfiguration& iTo) const
{
   FWConfiguration tmp;
   {
      std::stringstream s;
      s << GetWidth();
      tmp.addKeyValue("width", s.str());
   } 
   {
      std::stringstream s;
      s << GetHeight();
      tmp.addKeyValue("height", s.str());
   }

   iTo.addKeyValue("EventFilterGUI", tmp, true);
    
}

void
FWGUIEventFilter::setFrom(const FWConfiguration& iFrom)
{
   const FWConfiguration* conf = iFrom.valueForKey("EventFilterGUI");
   if (conf) {
      UInt_t w = atoi(conf->valueForKey("width")->value().c_str());
      UInt_t h = atoi(conf->valueForKey("height")->value().c_str());
      Resize(w, h);
   }
}
*/

bool FWGUIEventFilter::HandleKey(Event_t *event)
{ 
   // AMT workaround for problems  to override root's action  for 'Ctrl+s'
  
   if (GetBindList()->IsEmpty())
      FWGUIManager::getGUIManager()->getMainFrame()->bindCSGActionKeys(this);

   TIter next(fBindList);
   TGMapKey *m;
   TGFrame  *w = nullptr;

   while ((m = (TGMapKey *) next())) {
      if (m->fKeyCode == event->fCode) {
         w = (TGFrame *) m->fWindow;
         if (w->HandleKey(event)) return kTRUE;
      }
   }
   return kFALSE;
}
