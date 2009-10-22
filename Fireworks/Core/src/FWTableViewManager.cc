// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTableViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWTableViewManager.cc,v 1.9 2009/09/24 14:55:25 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"
#include "Reflex/Base.h"
#include "Reflex/Type.h"

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTableViewManager::FWTableViewManager(FWGUIManager* iGUIMgr) :
     FWViewManagerBase()
{
     FWGUIManager::ViewBuildFunctor f;
     f=boost::bind(&FWTableViewManager::buildView,
		   this, _1);
     iGUIMgr->registerViewBuilder(FWTableView::staticTypeName(), f);

     // ---------- for some object types, we have default table contents ----------
     TableEntry genparticle_table_entries[] = { 
	  { "pt"	, "pT"		, 1 			},
	  { "eta"	, "eta"		, 3 			},
	  { "phi"	, "phi"		, 3 			},
	  { "status"	, "status"	, TableEntry::INT 	},
	  { "pdgId"	, "pdgId"	, TableEntry::INT 	},
     };
     TableEntry muon_table_entries[] = { 
	  { "charge"				, "q"			, TableEntry::INT	},
	  { "pt"				, "pT"			, 1 			},
	  { "isGlobalMuon"			, "global"		, TableEntry::BOOL	},
	  { "isTrackerMuon"			, "tracker"		, TableEntry::BOOL	},
	  { "isStandAloneMuon"			, "SA"			, TableEntry::BOOL	},
	  { "isCaloMuon"			, "calo"		, TableEntry::BOOL	},
	  { "track().pt()"			, "tr pt"		, 1 			},
	  { "eta"				, "eta"			, 3 			},
	  { "phi"				, "phi"			, 3 			},
	  { "numberOfMatches('SegmentArbitration')"	, "matches"	, TableEntry::INT	},
	  { "track().d0()"			, "d0"			, 3			},
	  { "track().d0() / track().d0Error()"	, "d0 / d0Err"		, 3			},
     };
     TableEntry electron_table_entries[] = { 
	  { "charge"				, "q"		, TableEntry::INT	},
	  { "pt"				, "pT"		, 1 	},
	  { "eta"				, "eta"		, 3 	},
	  { "phi"				, "phi"		, 3 	},
	  { "eSuperClusterOverP"		, "E/p"		, 3 	},
	  { "hadronicOverEm"			, "H/E"		, 3 	},
	  { "(trackMomentumAtVtx().R() - trackMomentumOut().R()) / trackMomentumAtVtx().R()"			, "fbrem"	, 3 	},
	  { "deltaEtaSuperClusterTrackAtVtx()"	, "dei"		, 3 	},
	  { "deltaPhiSuperClusterTrackAtVtx()"	, "dpi"		, 3 	} 
     };
     TableEntry photon_table_entries[] = { 
	  { "pt"				, "pT"		, 1 	},
	  { "eta"				, "eta"		, 3 	},
	  { "phi"				, "phi"		, 3 	},
	  { "hadronicOverEm"			, "H/E"		, 3 	},
     };
     TableEntry calojet_table_entries[] = { 
	  { "pt"	, "pT"		, 1 	},
	  { "eta"	, "eta"		, 3 	},
	  { "phi"	, "phi"		, 3 	},
	  { "p4().E() * emEnergyFraction()"		, "ECAL"	, 1 	},
	  { "p4().E() * energyFractionHadronic()"	, "HCAL"	, 1 	},
	  { "emEnergyFraction()"			, "emf"		, 3 	},
     };
     TableEntry jet_table_entries[] = { 
	  { "pt"	, "pT"		, 1 	},
	  { "eta"	, "eta"		, 3 	},
	  { "phi"	, "phi"		, 3 	},
     };
     TableEntry met_table_entries[] = { 
	  { "et"	, "MET"		, 1 	},
	  { "phi"	, "phi"		, 3 	},
	  { "sumEt"	, "sumEt"	, 1 	},
	  { "mEtSig"	, "mEtSig"	, 3 	},
     };
     TableEntry track_table_entries[] = { 
	  { "charge"	, "q"		, TableEntry::INT	},
	  { "pt"	, "pT"		, 1 	},
	  { "eta"	, "eta"		, 3 	},
	  { "phi"	, "phi"		, 3 	},
	  { "d0"	, "d0"		, 5 	},
	  { "d0Error"	, "d0Err"	, 5 	},
	  { "dz"	, "dz"		, 5 	},
	  { "dzError"	, "dzErr"	, 5 	},
	  { "vx"	, "vx"		, 5 	},
	  { "vy"	, "vy"		, 5 	},
	  { "vz"	, "vz"		, 5 	},
	  { "hitPattern().numberOfValidPixelHits()"	, "pixel hits"	, TableEntry::INT 	},
	  { "hitPattern().numberOfValidStripHits()"	, "strip hits"	, TableEntry::INT 	},
	  { "chi2"	, "chi2"	, 3 	},
	  { "ndof"	, "ndof"	, TableEntry::INT 	},
     };
     TableEntry vertex_table_entries[] = { 
	  { "x"		, "x"		, 5 	},
	  { "xError"	, "xError"	, 5 	},
	  { "y"		, "y"		, 5 	},
	  { "yError"	, "yError"	, 5 	},
	  { "z"		, "z"		, 5 	},
	  { "zError"	, "zError"	, 5 	},
	  { "tracksSize", "tracks"	, TableEntry::INT 	},
	  { "chi2"	, "chi2"	, 3 	},
	  { "ndof"	, "ndof"	, TableEntry::INT 	},
     };
     TableEntry calotower_table_entries[] = { 
	  { "emEt"	, "emEt"	, 1 	},
	  { "hadEt"	, "hadEt"	, 1 	},
	  { "Et"	, "et"		, 1 	},
	  { "eta"	, "eta"		, 3 	},
	  { "phi"	, "phi"		, 3 	},
     };
     m_tableFormats["reco::GenParticle"	].insert(m_tableFormats["reco::GenParticle"	].end(), genparticle_table_entries	, genparticle_table_entries 	+ sizeof(genparticle_table_entries	) / sizeof(TableEntry));
     m_tableFormats["reco::Muon"	].insert(m_tableFormats["reco::Muon"		].end(), muon_table_entries		, muon_table_entries 		+ sizeof(muon_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::GsfElectron"	].insert(m_tableFormats["reco::GsfElectron"	].end(), electron_table_entries		, electron_table_entries 	+ sizeof(electron_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::Photon"	].insert(m_tableFormats["reco::Photon"		].end(), photon_table_entries		, photon_table_entries	 	+ sizeof(photon_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::CaloJet"	].insert(m_tableFormats["reco::CaloJet"		].end(), calojet_table_entries		, calojet_table_entries 	+ sizeof(calojet_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::Jet"		].insert(m_tableFormats["reco::Jet"		].end(), jet_table_entries		, jet_table_entries 		+ sizeof(jet_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::MET"		].insert(m_tableFormats["reco::MET"		].end(), met_table_entries		, met_table_entries 		+ sizeof(met_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::Track"	].insert(m_tableFormats["reco::Track"		].end(), track_table_entries		, track_table_entries 		+ sizeof(track_table_entries		) / sizeof(TableEntry));
     m_tableFormats["reco::Vertex"	].insert(m_tableFormats["reco::Vertex"		].end(), vertex_table_entries		, vertex_table_entries 		+ sizeof(vertex_table_entries		) / sizeof(TableEntry));
     m_tableFormats["CaloTower"		].insert(m_tableFormats["CaloTower"		].end(), calotower_table_entries	, calotower_table_entries 	+ sizeof(calotower_table_entries	) / sizeof(TableEntry));
}

FWTableViewManager::~FWTableViewManager()
{
}

//
// member functions
//
std::map<std::string, std::vector<FWTableViewManager::TableEntry> >::iterator 
FWTableViewManager::tableFormatsImpl (const Reflex::Type &key) 
{
//      printf("trying to find a table for %s\n", key.Name(ROOT::Reflex::SCOPED).c_str());
     std::map<std::string, std::vector<FWTableViewManager::TableEntry> >::iterator 
	  ret = m_tableFormats.find(key.Name(ROOT::Reflex::SCOPED));
     if (ret != m_tableFormats.end())
	  return ret;
//      for (Reflex::Base_Iterator it = key.Base_Begin(); it != key.Base_End(); ++i) {
// 	  ret = m_tableFormats.find(it->Name(ROOT::Reflex::SCOPED));
// 	  if (ret != m_tableFormats.end())
// 	       return ret;
//      }
     // if there is no exact match for the type, try the base classes
     for (Reflex::Base_Iterator it = key.Base_Begin(); it != key.Base_End(); ++it) {
	  ret = tableFormatsImpl(it->ToType());
 	  if (ret != m_tableFormats.end()) 
	       return ret;
     }
     // if there is no match at all, we just start with a blank table
     return m_tableFormats.end();
}

std::map<std::string, std::vector<FWTableViewManager::TableEntry> >::iterator 
FWTableViewManager::tableFormats (const Reflex::Type &key) 
{
     std::map<std::string, std::vector<FWTableViewManager::TableEntry> >::iterator 
	  ret = m_tableFormats.find(key.Name(ROOT::Reflex::SCOPED));
     if (ret != m_tableFormats.end())
	  return ret;
     else ret = tableFormatsImpl(key); // recursive search for base classes
     if (ret != m_tableFormats.end()) {
	  std::pair<std::string, std::vector<FWTableViewManager::TableEntry> > 
	       new_format(key.Name(ROOT::Reflex::SCOPED), ret->second);
	  std::cout << "adding new type " << key.Name(ROOT::Reflex::SCOPED) << std::endl;
	  return m_tableFormats.insert(new_format).first;
     } else {
	  TableEntry default_table_entries[] = { 
	       { "pt"	, "pt"	, 1 	},
	       { "eta"	, "eta"	, 3 	},
	       { "phi"	, "phi"	, 3 	},
	  };
	  std::pair<std::string, std::vector<FWTableViewManager::TableEntry> > new_format(
	       key.Name(ROOT::Reflex::SCOPED), 
	       std::vector<FWTableViewManager::TableEntry>(
		    default_table_entries, 
		    default_table_entries + 
		    sizeof(default_table_entries) / sizeof(TableEntry)));
	  std::cout << "adding new type " << key.Name(ROOT::Reflex::SCOPED) << std::endl;
	  return m_tableFormats.insert(new_format).first;
     }
}

std::map<std::string, std::vector<FWTableViewManager::TableEntry> >::iterator 
FWTableViewManager::tableFormats (const TClass &key) 
{
     return tableFormats(Reflex::Type::ByName(key.GetName()));
}

class FWViewBase*
FWTableViewManager::buildView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWTableView> view( new FWTableView(iParent, this) );
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   view->beingDestroyed_.connect(boost::bind(&FWTableViewManager::beingDestroyed,
					     this,_1));
   return view.get();
}

void
FWTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      if(it->get() == iView) {
         m_views.erase(it);
         return;
      }
   }
}

void
FWTableViewManager::newItem(const FWEventItem* iItem)
{
     m_items.push_back(iItem);
     iItem->goingToBeDestroyed_.connect(boost::bind(&FWTableViewManager::destroyItem,
						    this, _1));
     // tell the views to update their item lists
     for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
	      m_views.begin(), itEnd = m_views.end();
	 it != itEnd; ++it) {
	  (*it)->updateItems();
	  (*it)->dataChanged();
     }
}

void FWTableViewManager::destroyItem (const FWEventItem *item)
{
     // remove the item from the list
     for (std::vector<const FWEventItem *>::iterator it = m_items.begin(), 
	       itEnd = m_items.end();
	  it != itEnd; ++it) {
	  if (*it == item) {
	       m_items.erase(it);
	       break;
	  }
     }
     // tell the views to update their item lists
     for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
	      m_views.begin(), itEnd = m_views.end();
	 it != itEnd; ++it) {
	  (*it)->updateItems();
	  (*it)->dataChanged();
     }
}

void
FWTableViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
   // printf("changes coming\n");
}

void
FWTableViewManager::modelChangesDone()
{
     gEve->EnableRedraw();
     // tell the views to update their item lists
     for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
	      m_views.begin(), itEnd = m_views.end();
	 it != itEnd; ++it) {
	  (*it)->dataChanged();
     }
     // printf("changes done\n");
}

void
FWTableViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
       m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
	(*it)->resetColors(colorManager());
//       printf("Changed the background color for a table to 0x%x\n", 
// 	     colorManager().background());
   }
}

void
FWTableViewManager::dataChanged()
{
     for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
	      m_views.begin(), itEnd = m_views.end();
	 it != itEnd;
	 ++it) {
	  (*it)->dataChanged();
//       printf("Changed the background color for a table to 0x%x\n", 
// 	     colorManager().background());
   }
}

FWTypeToRepresentations
FWTableViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
//    const std::string kSimple("simple#");

//    for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
//        it != itEnd;
//        ++it) {
//       for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
//             builderName != it->second.end(); ++builderName )
//       {
//          if(builderName->substr(0,kSimple.size()) == kSimple) {
//             returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWSimpleRepresentationChecker(
//                                                                                builderName->substr(kSimple.size(),
//                                                                                                    builderName->find_first_of('@')-kSimple.size()),
//                                                                                it->first)));
//          } else {

//             returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWEDProductRepresentationChecker(
//                                                                                builderName->substr(0,builderName->find_first_of('@')),
//                                                                                it->first)));
//          }
//       }
//    }
   return returnValue;
}

const std::string FWTableViewManager::kConfigTypeNames = "typeNames";

void FWTableViewManager::addTo (FWConfiguration &iTo) const
{
     std::cout << "writing configuration" << std::endl;
     // if there are views, it's the job of the first view to store
     // the configuration (this is to avoid ordering problems in the
     // case of multiple views)
     if (m_views.size() > 0)
	  return;
     // if there are no views, then it's up to us to store the column
     // formats.  This is done in addToImpl, which can be called by
     // FWTableView as well
     addToImpl(iTo);
}
     
void FWTableViewManager::addToImpl (FWConfiguration &iTo) const
{
     FWConfiguration typeNames(1);
     for (std::map<std::string, std::vector<TableEntry> >::const_iterator 
	       iType = m_tableFormats.begin(),
	       iType_end = m_tableFormats.end();
	  iType != iType_end; ++iType) {
	  typeNames.addValue(iType->first);
	  FWConfiguration columns(1);
	  for (std::vector<FWTableViewManager::TableEntry>::const_iterator 
		    i = iType->second.begin(),
		    iEnd = iType->second.end();
	       i != iEnd; ++i) {
	       columns.addValue(i->name);
	       columns.addValue(i->expression);
	       char prec[100];
	       snprintf(prec, 100, "%d", i->precision);
	       columns.addValue(prec);
	  }
	  iTo.addKeyValue(iType->first, columns);
     }
     iTo.addKeyValue(kConfigTypeNames, typeNames);
}

void FWTableViewManager::setFrom(const FWConfiguration &iFrom)
{
     try {
	  const FWConfiguration *typeNames = iFrom.valueForKey(kConfigTypeNames);
	  if (typeNames != 0) {
               //NOTE: FWTableViewTableManagers hold pointers into m_tableFormats so if we
               // clear it those pointers would be invalid
               // instead we will just clear the lists and fill them with their new values
	       //m_tableFormats.clear();
	       for (FWConfiguration::StringValuesIt 
			 iType = typeNames->stringValues()->begin(),
			 iTypeEnd = typeNames->stringValues()->end(); 
		    iType != iTypeEnd; ++iType) {
		    //std::cout << "reading type " << *iType << std::endl;
		    const FWConfiguration *columns = iFrom.valueForKey(*iType);
		    assert(columns != 0);
		    std::vector<TableEntry> &formats = m_tableFormats[*iType];
                    formats.clear();
		    for (FWConfiguration::StringValuesIt 
			      it = columns->stringValues()->begin(),
			      itEnd = columns->stringValues()->end(); 
			 it != itEnd; ++it) {
			 const std::string &name = *it++;
			 const std::string &expr = *it++;
			 int prec = atoi(it->c_str());
			 FWTableViewManager::TableEntry e = { expr, name, prec };
			 formats.push_back(e);
		    }
	       }
	  } else {
	       std::cout << "no table column configuration stored, using defaults\n";
	  }
     } catch (...) {
	  // No info about types in the configuration; this is not an
	  // error, it merely means that the types are handled by the
	  // first FWTableView.
     }
}

