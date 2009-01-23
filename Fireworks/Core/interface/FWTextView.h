// -*- C++ -*-
#ifndef Fireworks_Core_FWTextView_h
#define Fireworks_Core_FWTextView_h

#include "RQ_OBJECT.h"
#include <vector>
#include <string>

namespace fwlite {
   class Event;
}

class ElectronTableManager;
class MuonTableManager;
class JetTableManager;
class HLTTableManager;
class L1TableManager;
class TrackTableManager;
class VertexTableManager;
class TGMainFrame;
class TGPicture;
class TGTab;
class TGTextEntry;
class TGTextView;
class TGCheckButton;
class TGCompositeFrame;
class TGTransientFrame;
class FWTableManager;
class FWTextView;
class FWDisplayEvent;
class FWSelectionManager;
class FWModelChangeManager;
class FWEventItem;
class FWGUIManager;
class CmsShowMain;

class FWTextViewHeader {
public:
   void dump (FILE *) const;
   void setContent (int run, int event, double met, double metPhi,
                    double sumEt, double mEtSig);
   void update (TGTextView *) const;   // one header serves multiple views

public:
   long int run, event;
   double met, metPhi, sumEt, mEtSig;
};

class FWTextViewPage {
   RQ_OBJECT("FWTextViewPage")
public:
   FWTextViewPage (const std::string &title,
                   const std::vector<FWTableManager *> &tables,
                   TGCompositeFrame *frame,
                   TGTab *parent_tab,
                   FWTextView *view, unsigned int layout);
   void       setNext (FWTextViewPage *);
   void       setPrev (FWTextViewPage *);
   void       deselect ();
   void       select ();
   void       update ();
   void       undock ();
   void       redock ();
   void       dumpToFile ();
   void       dumpToTerminal ();
   void       copyToSelection ();
   void       dumpToPrinter ();
   static const TGPicture *copyIcon ();

public:
   std::string title;
   std::vector<FWTableManager *>      tables;
   TGTextView                         *header_view;
   TGCompositeFrame                   *frame;
   TGTab                              *parent_tab;
   TGTransientFrame                   *undocked;
   TGCompositeFrame                   *parent;
   TGTextEntry                        *file_name;
   TGCheckButton                      *append_button;
   TGTextEntry                        *print_command;
   FWTextView                         *view;
   FWTextViewPage                     *prev;
   FWTextViewPage                     *next;
   TGCompositeFrame                   *nav_frame;
};

class FWTextView {
   RQ_OBJECT("FWTextView")
public:
   FWTextView (CmsShowMain *, FWSelectionManager *, FWModelChangeManager *,
               FWGUIManager *);
   void newEvent (const fwlite::Event &, const CmsShowMain *);
   void nextPage ();
   void prevPage ();
   void selectionChanged (const FWSelectionManager &);
   void itemChanged (const FWEventItem *);
   void changesDone (const CmsShowMain *);
   void update (int tab);

public:
   // objects
   ElectronTableManager       *el_manager;
   MuonTableManager           *mu_manager;
   JetTableManager            *jet_manager;
   // trigger
   L1TableManager             *l1em_manager;
   L1TableManager             *l1mu_manager;
   L1TableManager             *l1jet_manager;
   HLTTableManager            *hlt_manager;
   // tracks
   TrackTableManager          *track_manager;
   VertexTableManager         *vertex_manager;
//       TGMainFrame		*fMain;

   FWTextViewHeader header;

   std::vector<FWTableManager *>      managers;

   // display pages
   FWTextViewPage             *page;
   FWTextViewPage             *pages[3];

   FWSelectionManager         *seleman;
   FWModelChangeManager       *changeman;

   TGTab                      *parent_tab;
};

#endif
