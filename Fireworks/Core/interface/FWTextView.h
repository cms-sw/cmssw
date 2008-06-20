// -*- C++ -*-
#ifndef Fireworks_Core_FWTextView_h
#define Fireworks_Core_FWTextView_h

#include "RQ_OBJECT.h"
#include <vector>

namespace fwlite {
     class Event;
}

class ElectronTableManager;
class MuonTableManager;
class JetTableManager;
class TGMainFrame;
class TGCompositeFrame;
class FWTableManager;
class FWTextView;
class FWDisplayEvent;
class FWSelectionManager;
class CmsShowMain;

class FWTextViewPage {
public:
     FWTextViewPage (const std::string &title, 
		     const std::vector<FWTableManager *> &tables,
		     TGMainFrame *frame,
		     FWTextView *view);
     void	setNext (FWTextViewPage *);
     void	setPrev (FWTextViewPage *);
     void	deselect ();
     void	select ();
     void	update ();
		     
public:
     std::string			title;
     std::vector<FWTableManager *>	tables;
     TGMainFrame			*frame;
     FWTextView				*view;
     FWTextViewPage			*prev;
     FWTextViewPage			*next;
     TGCompositeFrame			*nav_frame;
};

class FWTextView {
     RQ_OBJECT("FWTextView") 
public:
     FWTextView (CmsShowMain *, FWSelectionManager *);
     void newEvent (const fwlite::Event &, const CmsShowMain *);
     void nextPage ();
     void prevPage ();
     void selectionChanged (const FWSelectionManager &);

protected:
     // objects
     ElectronTableManager	*el_manager;
     MuonTableManager		*mu_manager;
     JetTableManager		*jet_manager;
     // trigger
     ElectronTableManager	*l1_manager;
     ElectronTableManager	*hlt_manager;
     // tracks
     ElectronTableManager	*track_manager;
     ElectronTableManager	*vertex_manager;
     TGMainFrame		*fMain;

     std::vector<FWTableManager *>	managers;

     // display pages
     FWTextViewPage		*page;
     FWTextViewPage		*pages[3];

     FWSelectionManager		*seleman;
};

#endif
