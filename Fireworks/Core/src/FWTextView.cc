#define private public
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "CmsShowMain.h"
#undef private
#include "TEveBrowser.h"
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TGTextView.h"
#include "TStopwatch.h"

#include "boost/bind.hpp"
#include <iostream>

#include "TableManagers.h"

FWTextViewPage::FWTextViewPage (const std::string &title_, 
				const std::vector<FWTableManager *> &tables_,
				TGCompositeFrame *frame_,
				FWTextView *view_)
     : title(title_),
       tables(tables_),
       frame(frame_),
       view(view_),
       prev(0),
       next(0)
{
     for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
	  i != tables.end(); ++i) {
	  int width=1200;
	  int height=600;
	  (*i)->MakeFrame(frame, width, height);
// 	  frame->AddFrame((*i)->frame, tFrameHints);
// 	  printf("frame: adding frame\n");
     }
     frame->MapSubwindows();
     frame->MapWindow(); 
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
     for (std::vector<FWTableManager *>::const_iterator i = tables.begin();
	  i != tables.end(); ++i) {
	  (*i)->Update();
     }
}

FWTextView::FWTextView (CmsShowMain *de, FWSelectionManager *sel,
			FWGUIManager *gui)
     : el_manager(new ElectronTableManager),
       mu_manager(new MuonTableManager),
       jet_manager(new JetTableManager),
       l1_manager    	(new ElectronTableManager),
       hlt_manager   	(new ElectronTableManager),
       track_manager 	(new TrackTableManager),
       vertex_manager  	(new ElectronTableManager),
       seleman		(sel)
{      
     // stick managers in a vector for easier collective operations
     FWTableManager *managers_retreat[] = {
	  el_manager		,
	  mu_manager		,
	  jet_manager		,
	  l1_manager    	,	
	  hlt_manager   	,	
	  track_manager 	,	
	  vertex_manager	,
     };
     managers.insert(managers.begin(), 
		     managers_retreat, 
		     managers_retreat + 
		     sizeof(managers_retreat) / sizeof(FWTableManager *));
#if 0
     // apparently this is no good
     for (std::vector<FWEventItem *>::const_iterator 
	       i = de->m_eiManager->m_items.begin(),
	       end = de->m_eiManager->m_items.end(); 
	  i != end; ++i) {
	  printf("event item: %s\n", (*i)->name().c_str());
	  if ((*i)->name() == "Electrons") { 
	       el_manager->item = *i;
	  } else if ((*i)->name() == "Muons") { 
	       mu_manager->item = *i;
	  } else if ((*i)->name() == "Jets") { 
	       jet_manager->item = *i;
	  } else if ((*i)->name() == "Tracks") { 
	       track_manager->item = *i;
	  } else if ((*i)->name() == "Vertices") { 
	       vertex_manager->item = *i;
	  } 
     }
#endif
     // connect to the selection manager
     seleman->selectionChanged_.
	  connect(boost::bind(&FWTextView::selectionChanged,this,_1));
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
     v_objs.push_back(mu_manager);
     v_objs.push_back(el_manager);
     v_objs.push_back(jet_manager);
     FWTextViewPage *objects = new FWTextViewPage("Physics objects", v_objs, 
						  gui->m_textViewFrame[0], this);
     std::vector<FWTableManager *> v_trigger;
     v_trigger.push_back(l1_manager);
     v_trigger.push_back(hlt_manager);
     FWTextViewPage *trigger = new FWTextViewPage("Trigger information", v_trigger,
						  gui->m_textViewFrame[1], this);
     std::vector<FWTableManager *> v_tracks;
     v_tracks.push_back(track_manager);
     v_tracks.push_back(vertex_manager);
     FWTextViewPage *tracks = new FWTextViewPage("Tracking", v_tracks,
						  gui->m_textViewFrame[2], this);
     page = objects;
     objects->setNext(trigger);
     trigger->setPrev(objects);
     trigger->setNext(tracks);
     tracks->setPrev(trigger);
     // select current page
     page->select();
}

void FWTextView::newEvent (const fwlite::Event &ev, const CmsShowMain *de)
{
     TStopwatch stopwatch_read;
     TStopwatch stopwatch_table;
     //------------------------------------------------------------
     // get event items
     //------------------------------------------------------------
     const reco::GsfElectronCollection *els = 0;
     const reco::MuonCollection *mus = 0;
     const reco::CaloJetCollection *jets = 0;
     const reco::CaloMETCollection *mets = 0;
     const reco::TrackCollection *tracks = 0;
     const std::vector<reco::Vertex> *vertices = 0;
     for (std::vector<FWEventItem *>::const_iterator 
	       i = de->m_eiManager->m_items.begin(),
	       end = de->m_eiManager->m_items.end(); 
	  i != end; ++i) {
	  printf("event item: %s\n", (*i)->name().c_str());
	  if ((*i)->name() == "Electrons") { 
	       (*i)->get(els);
//   	       (*i)->select(0);
//   	       (*i)->select(2);
	  } else if ((*i)->name() == "Muons") { 
	       (*i)->get(mus);
	  } else if ((*i)->name() == "Jets") { 
	       (*i)->get(jets);
	  } else if ((*i)->name() == "METs") { 
	       (*i)->get(mets);
	  } else if ((*i)->name() == "Tracks") { 
	       (*i)->get(tracks);
	  } else if ((*i)->name() == "Vertices") { 
	       (*i)->get(vertices);
	  } 
     }
     // if I try to do this in the ctor, someone pulls the items out
     // from under me later
     for (std::vector<FWEventItem *>::const_iterator 
	       i = de->m_eiManager->m_items.begin(),
	       end = de->m_eiManager->m_items.end(); 
	  i != end; ++i) {
	  printf("event item: %s\n", (*i)->name().c_str());
	  if ((*i)->name() == "Electrons") { 
	       el_manager->item = *i;
	  } else if ((*i)->name() == "Muons") { 
	       mu_manager->item = *i;
	  } else if ((*i)->name() == "Jets") { 
	       jet_manager->item = *i;
	  } else if ((*i)->name() == "Tracks") { 
	       track_manager->item = *i;
	  } else if ((*i)->name() == "Vertices") { 
	       vertex_manager->item = *i;
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
     if (mets != 0)
	  n_tracks = tracks->size();
     //------------------------------------------------------------
     // vertices
     //------------------------------------------------------------
     int n_vertices = 0;
     if (vertices != 0)
	  n_vertices = vertices->size();

     //------------------------------------------------------------
     // print header
     //------------------------------------------------------------
     printf("run %d event %d:\n", ev.aux_.run(), ev.aux_.event());
     std::cout << ev.aux_;
#if 1
     assert(n_mets == 1);
     const CaloMET &met = *mets->begin();
     printf("MET %5.1f\t MET phi %.2f\t Sum Et %5.1f\t sig(MET) %.2f\n",
	    met.p4().Et(), met.phi(), met.sumEt(), met.mEtSig());
#endif
     printf("%d electrons\t%d muons\t%d jets\t\n", 
	    n_els, n_mus, n_jets);
     //------------------------------------------------------------
     // print muons
     //------------------------------------------------------------
     mu_manager->rows.clear();
     printf("Muons\n");
     printf("pt\t global\t tk\t SA\t calo\t iso(3)\t iso(5)\t tr pt\t eta\t"
	    " phi\t chi^2/ndof\t matches\t d0\t sig(d0)\t"
	    " loose(match)\t tight(match)\t loose(depth)\t tight(depth)\n");
     for (int i = 0; i < n_mus; ++i) {
	  const reco::Muon &muon = mus->at(i);
// 	  if (i == 5 || muon.pt() < 1) {
// 	       printf("skipping %d muons\n", n_mus - i);
// 	       break;
// 	  }
 	  double trpt = -1; // need a better way to flag invalid track
	  double trd0 = -1;
	  double trsd0 = 1;
	  if (muon.track().isNonnull()) {
	       trpt = muon.track()->pt();
	       trd0 = muon.track()->d0();
	       trsd0 = muon.track()->d0Error();
	  }
	  printf("%5.1f\t %c\t %c\t %c\t %c\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t"
		 "%6.3f\t\t %d\t\t %6.3f\t %7.3f\t %c\t\t %c\t\t %c\t\t %c\n",
		 muon.pt(), // is this right?
		 muon.isGlobalMuon() 		? 'y' : 'n', 
		 muon.isTrackerMuon()		? 'y' : 'n', 
		 muon.isStandAloneMuon()	? 'y' : 'n', 
		 muon.isCaloMuon()		? 'y' : 'n',
		 0., 0., // what iso? 
		 trpt, muon.eta(), muon.phi(),
		 0., // how to get chi^2?
		 muon.numberOfMatches(Muon::SegmentArbitration), // is this the right arbitration?
		 trd0, trd0 / trsd0,
		 'm', 'm', 'm', 'm' // get these flags!
	       );
	  MuonRowStruct row = {
	       i,
	       muon.pt(), // is this right?
	       muon.isGlobalMuon() 		? FLAG_YES : FLAG_NO, 
	       muon.isTrackerMuon()		? FLAG_YES : FLAG_NO, 
	       muon.isStandAloneMuon()	? FLAG_YES : FLAG_NO, 
	       muon.isCaloMuon()		? FLAG_YES : FLAG_NO,
	       0., 0., // what iso? 
	       trpt, muon.eta(), muon.phi(),
	       0., // how to get chi^2?
	       muon.numberOfMatches(Muon::SegmentArbitration), // is this the right arbitration?
	       trd0, trd0 / trsd0,
	       FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE // get these flags!
	  };
	  mu_manager->rows.push_back(row);
     }
     mu_manager->Sort(0, true);
     //------------------------------------------------------------
     // print electrons
     //------------------------------------------------------------
     el_manager->rows.clear();
     printf("Electrons\n");
     printf("Et\t eta\t phi\t E/p\t H/E\t fbrem\t dei\t dpi\t see\t spp\t"
	    " iso\t robust\t loose\t tight\n");
     for (int i = 0; i < n_els; ++i) {
	  const reco::GsfElectron &electron = els->at(i);
	  const double et = electron.caloEnergy() / cosh(electron.eta()); // is this the right way to get ET?
// 	  if (i == 5 || et < 1) {
// 	       printf("skipping %d electrons\n", n_els - i);
// 	       break;
// 	  }
	  const double pin  = electron.trackMomentumAtVtx().R();
	  const double pout = electron.trackMomentumOut().R();
	  printf("%5.1f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t %6.3f\t"
		 "%6.3f\t %6.3f\t %6.3f\t %c\t %c\t %c\n",
		 et, electron.eta(), electron.phi(),
		 electron.eSuperClusterOverP(), electron.hadronicOverEm(),
		 (pin - pout) / pin,
		 electron.deltaEtaSuperClusterTrackAtVtx(),
		 electron.deltaPhiSuperClusterTrackAtVtx(),
		 0., 0., // can we get the shape?
		 0., // can we get the iso?
		 'm', 'm', 'm' // can we get these flags?
	       );
	  ElectronRowStruct row = { 		 
	       i,
	       et, electron.eta(), electron.phi(),
	       electron.eSuperClusterOverP(), electron.hadronicOverEm(),
	       (pin - pout) / pin,
	       electron.deltaEtaSuperClusterTrackAtVtx(),
	       electron.deltaPhiSuperClusterTrackAtVtx(),
	       0., 0., // can we get the shape?
	       0., // can we get the iso?
	       FLAG_MAYBE, FLAG_MAYBE, FLAG_MAYBE // can we get these flags?
	  };
	  el_manager->rows.push_back(row);
     }
     el_manager->Sort(0, true);
     //------------------------------------------------------------
     // print jets
     //------------------------------------------------------------
     jet_manager->rows.clear();
     printf("Jets\n");
     printf("Et\t eta\t phi\t ECAL\t HCAL\t emf\t chf\n");
     for (int i = 0; i < n_jets; ++i) {
	  const reco::CaloJet &jet = jets->at(i);
	  const double et = jet.p4().Et();
// 	  if (i == 20 || et < 1) {
// 	       printf("skipping %d jets\n", n_jets - i);
// 	       break;
// 	  }
	  printf("%5.1f\t %6.3f\t %6.3f\t %5.1f\t %5.1f\t %6.3f\t %6.3f\n",
		 et, jet.eta(), jet.phi(),
		 jet.p4().E() * jet.emEnergyFraction(),		// this has got
		 jet.p4().E() * jet.energyFractionHadronic(),	// to be a joke
		 jet.emEnergyFraction(), 
		 0.	// how do we get the charge fraction?
	       );
	  JetRowStruct row = {
	       i,
	       et, jet.eta(), jet.phi(),
	       jet.p4().E() * jet.emEnergyFraction(),		// this has got
	       jet.p4().E() * jet.energyFractionHadronic(),	// to be a joke
	       jet.emEnergyFraction(), 
	       0.	// how do we get the charge fraction?
	  };
	  jet_manager->rows.push_back(row);
     }
     jet_manager->Sort(0, true);
     //------------------------------------------------------------
     // print tracks
     //------------------------------------------------------------
     printf("%d Tracks\n", n_tracks);
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
	       125,
	       tr.chi2(), int(tr.ndof())
	  };
	  track_manager->rows.push_back(row);
     }
     track_manager->Sort(0, true);
//      static int i = 0; 
//      i++;
//      if (i == 3) {
// 	  page->deselect();
// 	  page = page->next;
// 	  page->select();
//      }
     printf("read: ");
     stopwatch_read.Stop();
     stopwatch_read.Print("m");
     page->update();
     printf("table: ");
     stopwatch_table.Stop();
     stopwatch_table.Print("m");
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
	  if ((*i)->item != 0)
	       (*i)->selectRows();
}
