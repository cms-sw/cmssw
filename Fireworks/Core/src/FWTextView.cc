#define private public
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Fireworks/Core/interface/FWTextView.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "CmsShowMain.h"
#undef private
#include "TEveBrowser.h"
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TGTextView.h"
#include "TGTextEntry.h"
#include "TGButton.h"
#include "TStopwatch.h"
#include "TGTab.h"
#include "TString.h"
#include "TSystem.h"
#include "TGClient.h"

#include "boost/bind.hpp"
#include <iostream>

#include "TableManagers.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"

void FWTextViewHeader::update (TGTextView *view) const
{
   view->Clear();
   char s[1024];
   int n = snprintf(s, 1024, "Run %ld%16sEvent %ld", run, "", event);
   view->AddLineFast(s);
   n = snprintf(s, 1024, "MET %5.1f GeV%8sMET phi %6.3f%8sSum ET %5.1f GeV"
                "%8sMET significance %5.2f (GeV)^1/2",
                met, "", metPhi, "", sumEt, "", mEtSig);
   view->AddLineFast(s);
   view->Update();
}

void FWTextViewHeader::dump (FILE *f) const
{
   char s1[1024], s2[1024];
   int n1 = snprintf(s1, 1024, "Run %ld%8s%8sEvent %ld", run, "", "", event);
   int n2 = snprintf(s2, 1024, "MET %5.1f GeV%8sMET phi %6.3f%8sSum ET %5.1f GeV"
                     "%8sMET significance %5.2f (GeV)^1/2",
                     met, "", metPhi, "", sumEt, "", mEtSig);
   int len = std::max(n1, n2);
   for (int i = 0; i < len; ++i)
      fprintf(f, "=");
   fprintf(f, "\n%s\n", s1);
   fprintf(f, "%s\n", s2);
   for (int i = 0; i < len; ++i)
      fprintf(f, "=");
   fprintf(f, "\n");
}

void FWTextViewHeader::setContent (int run_, int event_, double met_,
                                   double metPhi_, double sumEt_, double mEtSig_)
{
   run        = run_;
   event      = event_;
   met        = met_    ;
   metPhi     = metPhi_ ;
   sumEt      = sumEt_  ;
   mEtSig     = mEtSig_ ;
}

FWTextViewPage::FWTextViewPage (const std::string &title_,
                                const std::vector<FWTableManager *> &tables_,
                                TGCompositeFrame *frame_,
                                TGTab *parent_tab_,
                                FWTextView *view_,
                                unsigned int layout)
   : title(title_),
     tables(tables_),
     frame(frame_),
     parent_tab(parent_tab_),
     undocked(0),
     view(view_),
     prev(0),
     next(0)
{
   const int width=frame->GetWidth();
   const int height=frame->GetHeight();
   TGHorizontalFrame *m_buttons = new TGHorizontalFrame(frame, width, 25, kFixedHeight);
   TGPictureButton *m_undockButton =
      new TGPictureButton(m_buttons, FWGUISubviewArea::undockIcon());
   m_undockButton->SetToolTipText("Undock view to own window");
   m_buttons->AddFrame(m_undockButton, new TGLayoutHints(kLHintsExpandY));
   m_undockButton->Connect("Clicked()", "FWTextViewPage", this, "undock()");
   TGTextButton *m_dumpButton =
      new TGTextButton(m_buttons, "Dump to file");
   m_dumpButton->SetToolTipText("Dump tables to file");
   m_buttons->AddFrame(m_dumpButton, new TGLayoutHints(kLHintsExpandY));
   m_dumpButton->Connect("Clicked()", "FWTextViewPage", this, "dumpToFile()");
   file_name = new TGTextEntry(m_buttons, "event_dump.txt");
   file_name->SetToolTipText("File name for dump (- for stdout)");
   file_name->Connect("ReturnPressed()", "FWTextViewPage", this, "dumpToFile()");
   m_buttons->AddFrame(file_name, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   append_button = new TGCheckButton(m_buttons, "append to file");
   append_button->SetToolTipText("Append to dump file");
   m_buttons->AddFrame(append_button, new TGLayoutHints(kLHintsExpandY));
   TGTextButton *m_termDumpButton =
      new TGTextButton(m_buttons, "Dump to terminal");
   m_termDumpButton->SetToolTipText("Dump tables to terminal");
   m_buttons->AddFrame(m_termDumpButton, new TGLayoutHints(kLHintsExpandY));
   m_termDumpButton->Connect("Clicked()", "FWTextViewPage", this, "dumpToTerminal()");
#if 0
   TGPictureButton *m_copyButton =
      new TGPictureButton(m_buttons, copyIcon());
   m_copyButton->SetToolTipText("Copy tables to X server selection");
   m_buttons->AddFrame(m_copyButton, new TGLayoutHints(kLHintsExpandY));
   m_copyButton->Connect("Clicked()", "FWTextViewPage", this, "copyToSelection()");
#endif
   TGTextButton *m_printButton =
      new TGTextButton(m_buttons, "Dump to printer");
   m_printButton->SetToolTipText("Dump tables to printer");
   m_buttons->AddFrame(m_printButton, new TGLayoutHints(kLHintsExpandY));
   m_printButton->Connect("Clicked()", "FWTextViewPage", this, "dumpToPrinter()");
   print_command = new TGTextEntry(m_buttons, "enscript -r -f Courier7");
   print_command->SetToolTipText("Print command");
   print_command->Connect("ReturnPressed()", "FWTextViewPage", this, "dumpToPrinter()");
   m_buttons->AddFrame(print_command, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   frame->AddFrame(m_buttons, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   header_view = new TGTextView(frame, width, 70);
   frame->AddFrame(header_view, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   header_view->SetBackground         (header_view->GetBlackPixel());
   header_view->SetForegroundColor    (header_view->GetWhitePixel());
   header_view->SetSelectBack(header_view->GetWhitePixel());
   header_view->SetSelectFore(header_view->GetBlackPixel());
   TGCompositeFrame *table_frame = frame;
   if (layout & kLHintsTop) {
      table_frame = new TGHorizontalFrame(frame);
      frame->AddFrame(table_frame, new TGLayoutHints(kLHintsTop | kLHintsExpandX |
                                                     kLHintsExpandY));
   }
   for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
        i != tables.end(); ++i) {
      (*i)->MakeFrame(table_frame, width, height, layout);
//        frame->AddFrame((*i)->frame, tFrameHints);
//        printf("frame: adding frame\n");
   }
   frame->MapSubwindows();
   frame->Layout();
   frame->MapWindow();
}

void FWTextViewPage::undock ()
{
   // Extract the frame contained in this split frame an reparent it in a
   // transient frame. Keep a pointer on the transient frame to be able to
   // swallow the child frame back to this.
   if (undocked == 0) {
      parent = (TGCompositeFrame *)dynamic_cast<const TGCompositeFrame *>(frame->GetParent());
      assert(parent != 0);
//      printf("this frame: %d, parent: %d; TGTab: %d\n",
//             frame->GetId(), parent->GetId(), parent_tab->GetId());
//      frame->UnmapWindow();
      undocked = new
                 TGTransientFrame(gClient->GetDefaultRoot(), frame, 800, 600);
      frame->ReparentWindow(undocked);
      undocked->AddFrame(frame, new TGLayoutHints(kLHintsExpandX |
                                                  kLHintsExpandY));
      // Layout...
      undocked->MapSubwindows();
      undocked->Layout();
      undocked->MapWindow();
//      parent->RemoveFrame(frame);
      undocked->Connect("CloseWindow()", "FWTextViewPage", this,
                        "redock()");
   }
}

void FWTextViewPage::redock ()
{
   if (undocked) {
      TGFrame *frame = (TGFrame *)
                       dynamic_cast<TGFrameElement*>(undocked->GetList()->First())->fFrame;
//        printf("undocked window: %d, frame to reparent: %d; parent: %d\n",
//               undocked->GetId(), frame->GetId(), parent->GetId());
      frame->UnmapWindow();
      undocked->RemoveFrame(frame);
      frame->ReparentWindow(parent);
//        parent->AddFrame(frame, new TGLayoutHints(kLHintsExpandX |
//                                                  kLHintsExpandY));
      // Layout...
      parent->MapSubwindows();
      parent->Layout();
      undocked->CloseWindow();
      undocked = 0;
   }
}

void FWTextViewPage::setNext (FWTextViewPage *p)
{
   next = p;
}

void FWTextViewPage::setPrev (FWTextViewPage *p)
{
   prev = p;
}

void FWTextViewPage::select ()
{
   // not needed with tabbed FWTextViewPages
}

void FWTextViewPage::deselect ()
{
   // not needed with tabbed FWTextViewPages
}

void FWTextViewPage::update ()
{
//      printf("size of frame:%d lines\n", frame->GetHeight() / 25);
//      printf("%d tables\n", tables.size());
//      for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
//        i != tables.end(); ++i) {
//        printf("\t%s: %d entries\n", (*i)->title().c_str(), (*i)->NumberOfRows());
//      }
//      printf("current tab is %d\n", parent_tab->GetCurrent());
   view->header.update(header_view);
//      int total_lines = 0;
//      for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
//        i != tables.end(); ++i) {
//        total_lines += (*i)->NumberOfRows() + 3;
//      }
   for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
        i != tables.end(); ++i) {
      (*i)->Update();
//        (*i)->Update(std::min(10, (*i)->NumberOfRows()));
   }
}

void FWTextViewPage::dumpToTerminal ()
{
   view->header.dump(stdout);
   for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
        i != tables.end(); ++i) {
      (*i)->dump(stdout);
   }
}

void FWTextViewPage::dumpToFile ()
{
   if (strcmp(file_name->GetBuffer()->GetString(), "-") == 0) {
      view->header.dump(stdout);
      for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
           i != tables.end(); ++i) {
         (*i)->dump(stdout);
      }
   } else {
      FILE *f = fopen(file_name->GetBuffer()->GetString(),
                      append_button->IsOn() ? "a" : "w");
      if (f == 0) {
         perror(file_name->GetBuffer()->GetString());
         return;
      }
      printf("dumping to file... ");
      view->header.dump(f);
      for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
           i != tables.end(); ++i) {
         (*i)->dump(f);
      }
      fclose(f);
      printf("done.\n");
   }
}

void FWTextViewPage::dumpToPrinter ()
{
   FILE *f = popen(print_command->GetBuffer()->GetString(), "w");
   if (f == 0) {
      perror(print_command->GetBuffer()->GetString());
      return;
   }
   printf("dumping to printer... ");
   view->header.dump(f);
   for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
        i != tables.end(); ++i) {
      (*i)->dump(f);
   }
   printf("done.\n");
   pclose(f);
}

void FWTextViewPage::copyToSelection ()
{

}

const TGPicture *FWTextViewPage::copyIcon ()
{
   const char* cmspath = gSystem->Getenv("ROOTSYS");
   if (0 == cmspath) {
      throw std::runtime_error("ROOTSYS environment variable not set");
   }
   TString coreIcondir(Form("%s/icons/", gSystem->Getenv("ROOTSYS")));
   return gClient->GetPicture(coreIcondir+"bld_copy.xpm");
}


FWTextView::FWTextView (CmsShowMain *de, FWSelectionManager *sel,
                        FWModelChangeManager *chg,
                        FWGUIManager *gui)
   : el_manager(new ElectronTableManager),
     mu_manager(new MuonTableManager),
     jet_manager(new JetTableManager),
     l1em_manager     (new L1TableManager),
     l1mu_manager     (new L1TableManager),
     l1jet_manager    (new L1TableManager),
     hlt_manager      (new HLTTableManager),
     track_manager    (new TrackTableManager),
     vertex_manager   (new VertexTableManager),
     seleman          (sel),
     changeman        (chg),
     parent_tab       (gui->m_textViewTab)
//        main		(de)
{
   // stick managers in a vector for easier collective operations
   FWTableManager *managers_retreat[] = {
      el_manager,
      mu_manager,
      jet_manager,
      l1em_manager,
      l1mu_manager,
      l1jet_manager,
      hlt_manager,
      track_manager,
      vertex_manager,
   };
   l1em_manager->setTitle("L1 EM objects");
   l1mu_manager->setTitle("L1 muons");
   l1jet_manager->setTitle("L1 jets");
   managers.insert(managers.begin(),
                   managers_retreat,
                   managers_retreat +
                   sizeof(managers_retreat) / sizeof(FWTableManager *));
   // connect to the selection manager
   seleman->selectionChanged_.
   connect(boost::bind(&FWTextView::selectionChanged,this,_1));
   // and to the change manager
   changeman->changeSignalsAreDone_.
   connect(boost::bind(&FWTextView::changesDone,this, de));
//      for (std::vector<FWItemChangeSignal>::iterator i =
//             changeman->m_itemChangeSignals.begin();
//        i != changeman->m_itemChangeSignals.end();
//        ++i) {
//        i->connect(boost::bind(&FWTextView::itemChanged, this, _1));
//        printf("connecting to something\n");
//      }
   //------------------------------------------------------------
   // widget up some tables (moved to CmsShowMainFrame)
   //------------------------------------------------------------
//   int width=1200;
//   int height=600;
//   TEveBrowser *b = gEve->GetBrowser();
//   b->StartEmbedding();
//   fMain = new TGMainFrame(gClient->GetRoot(),width,height);
//   b->StopEmbedding();

//      mu_manager->MakeFrame(fMain, width, height);
//      el_manager->MakeFrame(fMain, width, height);
//      jet_manager->MakeFrame(fMain, width, height);

//      l1_manager->MakeFrame(fMain, width, height);
//      hlt_manager->MakeFrame(fMain, width, height);

//      track_manager->MakeFrame(fMain, width, height);
//      vertex_manager->MakeFrame(fMain, width, height);

// //      // use hierarchical cleaning
//       fMain->SetCleanup(kDeepCleanup);

// //      // Set a name to the main frame
//       fMain->SetWindowName("Text view");

//      // Map all subwindows of main frame
//      fMain->MapSubwindows();

//      // Map main frame
//      fMain->MapWindow();

   // resize main window to tableWidth plus size of scroller and for first few cells heights.
   // so far I don't know how to figure out scroller size, I just found it's about 30 units.
//      fMain->Resize(width+20,height+20);

   //------------------------------------------------------------
   // set up the display pages
   //------------------------------------------------------------
   std::vector<FWTableManager *> v_objs;
   v_objs.push_back(jet_manager);
   v_objs.push_back(el_manager);
   v_objs.push_back(mu_manager);
   FWTextViewPage *objects = new FWTextViewPage("Physics objects", v_objs,
                                                gui->m_textViewFrame[0],
                                                gui->m_textViewTab,
                                                this, kLHintsLeft);
   std::vector<FWTableManager *> v_trigger;
   v_trigger.push_back(l1mu_manager);
   v_trigger.push_back(l1em_manager);
   v_trigger.push_back(l1jet_manager);
//      v_trigger.push_back(hlt_manager);
   FWTextViewPage *trigger = new FWTextViewPage("Trigger information", v_trigger,
                                                gui->m_textViewFrame[1],
                                                gui->m_textViewTab, this,
                                                kLHintsTop);
   std::vector<FWTableManager *> v_tracks;
   v_tracks.push_back(track_manager);
   v_tracks.push_back(vertex_manager);
   FWTextViewPage *tracks = new FWTextViewPage("Tracking", v_tracks,
                                               gui->m_textViewFrame[2],
                                               gui->m_textViewTab, this,
                                               kLHintsLeft);
   page = objects;
   objects->setNext(trigger);
   trigger->setPrev(objects);
   trigger->setNext(tracks);
   tracks->setPrev(trigger);
   // select current page
   page->select();
   // keep track of all of them
   pages[0] = objects;
   pages[1] = trigger;
   pages[2] = tracks;
   parent_tab->Connect("Selected(Int_t)", "FWTextView", this, "update(int)");
}

void FWTextView::update (int tab)
{
   // update a tab, either because of tab click or because of new
   // event
   if (tab > 0 && tab < 4)
      pages[tab - 1]->update();
}

void FWTextView::newEvent (const fwlite::Event &ev, const CmsShowMain *de)
{
   // TStopwatch stopwatch_read;
   // TStopwatch stopwatch_table;
   //------------------------------------------------------------
   // get event items
   //------------------------------------------------------------
   const reco::GsfElectronCollection *els = 0;
   const reco::MuonCollection *mus = 0;
   const reco::CaloJetCollection *jets = 0;
   const reco::CaloMETCollection *mets = 0;
   const reco::TrackCollection *tracks = 0;
   const std::vector<reco::Vertex> *vertices = 0;
   const l1extra::L1EmParticleCollection      *l1ems = 0;
   const l1extra::L1MuonParticleCollection    *l1mus = 0;
   const l1extra::L1JetParticleCollection     *l1jets = 0;
   for (std::vector<FWEventItem *>::const_iterator
        i = de->m_eiManager->m_items.begin(),
        end = de->m_eiManager->m_items.end();
        i != end; ++i) {
      if (*i == 0)     // event items can be null after they'be been removed
         continue;
//        printf("event item: %s found\n", (*i)->name().c_str());
      if ((*i)->name() == "Electrons") {
         (*i)->get(els);
//             (*i)->select(0);
//             (*i)->select(2);
      } else if ((*i)->name() == "Muons") {
         (*i)->get(mus);
      } else if ((*i)->name() == "Jets") {
         (*i)->get(jets);
      } else if ((*i)->name() == "MET") {
         (*i)->get(mets);
      } else if ((*i)->name() == "Tracks") {
         (*i)->get(tracks);
      } else if ((*i)->name() == "Vertices") {
         (*i)->get(vertices);
      } else if ((*i)->name() == "L1EmTrig") {
         (*i)->get(l1ems);
      } else if ((*i)->name() == "L1-Muons") {
         (*i)->get(l1mus);
      } else if ((*i)->name() == "L1-Jets") {
         (*i)->get(l1jets);
      }
   }
   // if I try to do this in the ctor, someone pulls the items out
   // from under me later
   for (std::vector<FWEventItem *>::const_iterator
        i = de->m_eiManager->m_items.begin(),
        end = de->m_eiManager->m_items.end();
        i != end; ++i) {
      if (*i == 0)     // event items can be null after they'be been removed
         continue;
//        printf("event item: %s\n", (*i)->name().c_str());
      if ((*i)->name() == "Electrons") {
         el_manager->setItem(*i);
      } else if ((*i)->name() == "Muons") {
         mu_manager->setItem(*i);
      } else if ((*i)->name() == "Jets") {
         jet_manager->setItem(*i);
      } else if ((*i)->name() == "Tracks") {
         track_manager->setItem(*i);
      } else if ((*i)->name() == "Vertices") {
         vertex_manager->setItem(*i);
      } else if ((*i)->name() == "L1EmTrig") {
         l1em_manager->setItem(*i);
      } else if ((*i)->name() == "L1-Muons") {
         l1mu_manager->setItem(*i);
      } else if ((*i)->name() == "L1-Jets") {
         l1jet_manager->setItem(*i);
      }
   }
   //------------------------------------------------------------
   // electrons
   //------------------------------------------------------------
   int n_els = 0;
   if (els != 0)
      n_els = els->size();
   //------------------------------------------------------------
   // muons
   //------------------------------------------------------------
   int n_mus = 0;
   if (mus != 0)
      n_mus = mus->size();
   //------------------------------------------------------------
   // jets
   //------------------------------------------------------------
   int n_jets = 0;
   if (jets != 0)
      n_jets = jets->size();
   //------------------------------------------------------------
   // MET
   //------------------------------------------------------------
   int n_mets = 0;
   if (mets != 0)
      n_mets = mets->size();
   //------------------------------------------------------------
   // tracks
   //------------------------------------------------------------
   int n_tracks = 0;
   if (tracks != 0)
      n_tracks = tracks->size();
   //------------------------------------------------------------
   // vertices
   //------------------------------------------------------------
   int n_vertices = 0;
   if (vertices != 0)
      n_vertices = vertices->size();
   //------------------------------------------------------------
   // L1 EMs
   //------------------------------------------------------------
   int n_l1ems = 0;
   if (l1ems != 0)
      n_l1ems = l1ems->size();
   //------------------------------------------------------------
   // L1 mus
   //------------------------------------------------------------
   int n_l1mus = 0;
   if (l1mus != 0)
      n_l1mus = l1mus->size();
   //------------------------------------------------------------
   // L1 jets
   //------------------------------------------------------------
   int n_l1jets = 0;
   if (l1jets != 0)
      n_l1jets = l1jets->size();

   //------------------------------------------------------------
   // print header
   //------------------------------------------------------------
#if 0
   printf("run %d event %d:\n", ev.aux_.run(), ev.aux_.event());
   std::cout << ev.aux_;
#if 1
   printf("MET %5.1f\t MET phi %.2f\t Sum Et %5.1f\t sig(MET) %.2f\n",
          met.p4().Et(), met.phi(), met.sumEt(), met.mEtSig());
#endif
   printf("%d electrons\t%d muons\t%d jets\t\n",
          n_els, n_mus, n_jets);
#endif
   if (n_mets == 1) {   // this should always be the case for non-sick files
      const CaloMET &met = *mets->begin();
      header.setContent(ev.aux_.run(), ev.aux_.event(),
                        met.p4().Et(), met.phi(), met.sumEt(), met.mEtSig());
   } else {
      header.setContent(ev.aux_.run(), ev.aux_.event(),
                        -1, -1, -1, -1);
   }
   //------------------------------------------------------------
   // print muons
   //------------------------------------------------------------
   mu_manager->rows.clear();
#if 0
   printf("Muons\n");
   printf("pt\t global\t tk\t SA\t calo\t iso(3)\t iso(5)\t tr pt\t eta\t"
          " phi\t chi^2/ndof\t matches\t d0\t sig(d0)\t"
          " loose(match)\t tight(match)\t loose(depth)\t tight(depth)\n");
#endif
   for (int i = 0; i < n_mus; ++i) {
      const reco::Muon &muon = mus->at(i);
//        if (i == 5 || muon.pt() < 1) {
//             printf("skipping %d muons\n", n_mus - i);
//             break;
//        }
      double trpt = -1;     // need a better way to flag invalid track
      double trd0 = -1;
      double trsd0 = 1;
      if (muon.track().isNonnull()) {
         trpt = muon.track()->pt();
         trd0 = muon.track()->d0();
         trsd0 = muon.track()->d0Error();
      }
#if 0
      printf("%5.1f\t %c\t %c\t %c\t %c\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t"
             "%6.3f\t\t %d\t\t %6.3f\t %7.3f\t %c\t\t %c\t\t %c\t\t %c\n",
             muon.pt(),     // is this right?
             muon.isGlobalMuon()            ? 'y' : 'n',
             muon.isTrackerMuon()           ? 'y' : 'n',
             muon.isStandAloneMuon()        ? 'y' : 'n',
             muon.isCaloMuon()              ? 'y' : 'n',
             0., 0.,     // what iso?
             trpt, muon.eta(), muon.phi(),
             0.,     // how to get chi^2?
             muon.numberOfMatches(Muon::SegmentArbitration),     // is this the right arbitration?
             trd0, trd0 / trsd0,
             'm', 'm', 'm', 'm' // get these flags!
             );
#endif
      MuonRowStruct row = {
         i,
         muon.pt(),       // is this right?
         muon.isGlobalMuon()              ? FLAG_YES : FLAG_NO,
         muon.isTrackerMuon()             ? FLAG_YES : FLAG_NO,
         muon.isStandAloneMuon()  ? FLAG_YES : FLAG_NO,
         muon.isCaloMuon()                ? FLAG_YES : FLAG_NO,
//             0., 0., // what iso?
         trpt, muon.eta(), muon.phi(),
//             0., // how to get chi^2?
//             muon.numberOfMatches(Muon::SegmentArbitration), // is this the right arbitration?
         trd0, trd0 / trsd0,
//             FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE // get these flags!
      };
      mu_manager->rows.push_back(row);
   }
   mu_manager->sort(0, true);
   //------------------------------------------------------------
   // print electrons
   //------------------------------------------------------------
   el_manager->rows.clear();
#if 0
   printf("Electrons\n");
   printf("Et\t eta\t phi\t E/p\t H/E\t fbrem\t dei\t dpi\t see\t spp\t"
          " iso\t robust\t loose\t tight\n");
#endif
   for (int i = 0; i < n_els; ++i) {
      const reco::GsfElectron &electron = els->at(i);
      const double et = electron.caloEnergy() / cosh(electron.eta());     // is this the right way to get ET?
//        if (i == 5 || et < 1) {
//             printf("skipping %d electrons\n", n_els - i);
//             break;
//        }
      const double pin  = electron.trackMomentumAtVtx().R();
      const double pout = electron.trackMomentumOut().R();
#if 0
      printf("%5.1f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t"
             "%6.3f\t %6.3f\t %6.3f\t %c\t %c\t %c\n",
             et, electron.eta(), electron.phi(),
             electron.eSuperClusterOverP(), electron.hadronicOverEm(),
             (pin - pout) / pin,
             electron.deltaEtaSuperClusterTrackAtVtx(),
             electron.deltaPhiSuperClusterTrackAtVtx(),
             0., 0.,     // can we get the shape?
             0.,     // can we get the iso?
             'm', 'm', 'm'  // can we get these flags?
             );
#endif
      ElectronRowStruct row = {
         i,
         et, electron.eta(), electron.phi(),
         electron.eSuperClusterOverP(), electron.hadronicOverEm(),
         (pin - pout) / pin,
         electron.deltaEtaSuperClusterTrackAtVtx(),
         electron.deltaPhiSuperClusterTrackAtVtx(),
//             0., 0., // can we get the shape?
//             0., // can we get the iso?
//             FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE // can we get these flags?
      };
      el_manager->rows.push_back(row);
   }
   el_manager->sort(0, true);
   //------------------------------------------------------------
   // print jets
   //------------------------------------------------------------
   jet_manager->rows.clear();
#if 0
   printf("Jets\n");
   printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
#endif
   for (int i = 0; i < n_jets; ++i) {
      const reco::CaloJet &jet = jets->at(i);
      const double et = jet.p4().Et();
//        if (i == 20 || et < 1) {
//             printf("skipping %d jets\n", n_jets - i);
//             break;
//        }
#if 0
      printf("%5.1f\t %6.3f\t %6.3f\t %5.1f\t %5.1f\t %6.3f\t %6.3f\n",
             et, jet.eta(), jet.phi(),
             jet.p4().E() * jet.emEnergyFraction(),             // this has got
             jet.p4().E() * jet.energyFractionHadronic(),       // to be a joke
             jet.emEnergyFraction(),
             0.         // how do we get the charge fraction?
             );
#endif
      JetRowStruct row = {
         i,
         et, jet.eta(), jet.phi(),
         jet.p4().E() * jet.emEnergyFraction(),                 // this has got
         jet.p4().E() * jet.energyFractionHadronic(),           // to be a joke
         jet.emEnergyFraction(),
//	       0.	// how do we get the charge fraction?
      };
      jet_manager->rows.push_back(row);
   }
   jet_manager->sort(0, true);
   //------------------------------------------------------------
   // print tracks
   //------------------------------------------------------------
//      printf("%d tracks\n", n_tracks);
   track_manager->rows.clear();
//      printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
   for (int i = 0; i < n_tracks; ++i) {
      const reco::Track &tr = tracks->at(i);
      TrackRowStruct row = {
         i,
         tr.pt(), tr.eta(), tr.phi(),
         tr.d0(), tr.d0Error(), tr.dz(), tr.dzError(),
         tr.vx(), tr.vy(), tr.vz(),
         tr.hitPattern().numberOfValidPixelHits(),
         tr.hitPattern().numberOfValidStripHits(),
//             125,
         tr.chi2(), tr.ndof()
      };
      track_manager->rows.push_back(row);
   }
   track_manager->sort(0, true);
   //------------------------------------------------------------
   // print vertices
   //------------------------------------------------------------
//     printf("Vertices\n");
   vertex_manager->rows.clear();
//      printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
   for (int i = 0; i < n_vertices; ++i) {
      const reco::Vertex &vtx = vertices->at(i);
      VertexRowStruct row = {
         i,
         vtx.x(), vtx.xError(),
         vtx.y(), vtx.yError(),
         vtx.z(), vtx.zError(),
         vtx.tracksSize(), vtx.chi2(), vtx.ndof()
      };
      vertex_manager->rows.push_back(row);
   }
   vertex_manager->sort(0, true);
   //------------------------------------------------------------
   // print L1 EMs
   //------------------------------------------------------------
//      printf("%d tracks\n", n_tracks);
   l1em_manager->rows.clear();
//      printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
   for (int i = 0; i < n_l1ems; ++i) {
      const l1extra::L1EmParticle &l1 = l1ems->at(i);
      L1RowStruct row = {
         i,
         l1.et(), l1.eta(), l1.phi()
      };
      l1em_manager->rows.push_back(row);
   }
   l1em_manager->sort(0, true);
   //------------------------------------------------------------
   // print L2 mus
   //------------------------------------------------------------
//      printf("%d tracks\n", n_tracks);
   l1mu_manager->rows.clear();
//      printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
   for (int i = 0; i < n_l1mus; ++i) {
      const l1extra::L1MuonParticle &l1 = l1mus->at(i);
      L1RowStruct row = {
         i,
         l1.et(), l1.eta(), l1.phi()
      };
      l1mu_manager->rows.push_back(row);
   }
   l1mu_manager->sort(0, true);
   //------------------------------------------------------------
   // print L1 jets
   //------------------------------------------------------------
//      printf("%d tracks\n", n_tracks);
   l1jet_manager->rows.clear();
//      printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
   for (int i = 0; i < n_l1jets; ++i) {
      const l1extra::L1JetParticle &l1 = l1jets->at(i);
      L1RowStruct row = {
         i,
         l1.et(), l1.eta(), l1.phi()
      };
      l1jet_manager->rows.push_back(row);
   }
   l1jet_manager->sort(0, true);
//      static int i = 0;
//      i++;
//      if (i == 3) {
//        page->deselect();
//        page = page->next;
//        page->select();
//      }
   changesDone(0);
   // printf("read: ");
   // stopwatch_read.Stop();
   // stopwatch_read.Print("m");
   for (int i = 0; i < 3; ++i)
      pages[i]->update();
   // printf("table: ");
   // stopwatch_table.Stop();
   // stopwatch_table.Print("m");
}

void FWTextView::nextPage ()
{
   if (page->next != 0) {
      page->deselect();
      page = page->next;
      page->select();
   }
}

void FWTextView::prevPage ()
{
   if (page->prev != 0) {
      page->deselect();
      page = page->prev;
      page->select();
   }
}

void FWTextView::selectionChanged (const FWSelectionManager &m)
{
//      printf("selectionChanged\n");
   // clear old selection
   for (std::vector<FWTableManager *>::iterator
        i = managers.begin(), end = managers.end();
        i != end; ++i)
      (*i)->sel_indices.clear();
   // propagate new selection
   for (std::set<FWModelId>::const_iterator i = m.m_selection.begin(),
                                            end = m.m_selection.end(); i != end; ++i) {
      for (std::vector<FWTableManager *>::iterator
           j = managers.begin(), end = managers.end();
           j != end; ++j) {
         if (i->item() == (*j)->item)
            (*j)->sel_indices.insert(i->index());
      }
   }
   for (std::vector<FWTableManager *>::iterator
        i = managers.begin(), end = managers.end();
        i != end; ++i)
      if ((*i)->item != 0) {
         (*i)->selectRows();
      }
}

void FWTextView::changesDone (const CmsShowMain *)
{
//      printf("changesDone\n");
//      selectionChanged(*main->m_selectionManager);
   for (std::vector<FWTableManager *>::iterator
        i = managers.begin(), end = managers.end();
        i != end; ++i) {
      (*i)->vis_indices.clear();
      if ((*i)->item == 0)
         continue;
      (*i)->widget->SetTextColor((*i)->item->m_displayProperties.color());
      for (unsigned int j = 0; j < (*i)->item->m_itemInfos.size(); ++j) {
         if ((*i)->item->m_itemInfos[j].displayProperties().isVisible()) {
            (*i)->vis_indices.insert(j);
         }
      }
      (*i)->resort();     // because invisible lines are supposed to
                          // be at the bottom
      (*i)->selectRows();
   }
}

void FWTextView::itemChanged (const FWEventItem *item)
{
   printf("item %s changed!\n", item->name().c_str());
}

