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
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>

#include "TEveManager.h"
#include "TClass.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/fwLog.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTableViewManager::FWTableViewManager(FWGUIManager* iGUIMgr)
:FWViewManagerBase()
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWTableViewManager::buildView,
                 this, _1, _2);
   iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kTable), f);

   // ---------- for some object types, we have default table contents ----------
   table("reco::GenParticle").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("status", TableEntry::INT).
   column("pdgId", TableEntry::INT);

   table("reco::Muon").
   column("q", TableEntry::INT, "charge").
   column("pT", 1, "pt").
   column("global", TableEntry::BOOL, "isGlobalMuon").
   column("tracker", TableEntry::BOOL, "isTrackerMuon").
   column("SA", TableEntry::BOOL, "isStandAloneMuon").
   column("calo", TableEntry::BOOL, "isCaloMuon").
   column("tr pt", 1, "track().pt()").
   column("eta", 3).
   column("phi", 3).
   column("matches", TableEntry::INT, "numberOfMatches('SegmentArbitration')").
   column("d0", 3, "track().d0()").
   column("d0 / d0Err", 3, "track().d0() / track().d0Error()");

   table("reco::GsfElectron").
   column("q", TableEntry::INT, "charge").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("E/p", 3, "eSuperClusterOverP").
   column("H/E", 3, "hadronicOverEm").
   column("fbrem", 3, "(trackMomentumAtVtx().R() - trackMomentumOut().R()) / trackMomentumAtVtx().R()").
   column("dei", 3, "deltaEtaSuperClusterTrackAtVtx()").
   column("dpi", 3, "deltaPhiSuperClusterTrackAtVtx()");

   table("reco::Photon").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("H/E", 3, "hadronicOverEm");

   table("reco::CaloJet").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("ECAL", 1, "p4().E() * emEnergyFraction()").
   column("HCAL", 1, "p4().E() * energyFractionHadronic()").
   column("emf", 3, "emEnergyFraction()");

   table("reco::Jet").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("electronEnergyFraction", 3, "electronEnergyFraction()").
   column("muonEnergyFraction", 3, "muonEnergyFraction()").
   column("photonEnergyFraction", 3, "photonEnergyFraction()");

   table("reco::MET").
   column("et", 1).
   column("phi", 3).
   column("sumEt", 1).
   column("mEtSig", 3);

   table("reco::Track").
   column("q", TableEntry::INT, "charge").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("d0", 5).
   column("d0Err", 5, "d0Error").
   column("dz", 5).
   column("dzErr", 5, "dzError").
   column("vx", 5).
   column("vy", 5).
   column("vz", 5).
   column("pixel hits", TableEntry::INT, "hitPattern().numberOfValidPixelHits()").
   column("strip hits", TableEntry::INT, "hitPattern().numberOfValidStripHits()").
   column("chi2", 3).
   column("ndof", TableEntry::INT);

   table("DTRecSegment4D").
   column("wheel", 0, "chamberId.wheel").
   column("station", 0, "chamberId.station").
   column("sector", 0, "chamberId.sector").
   column("t0phi", 2, "phiSegment.t0").
   column("t0theta", 2, "zSegment.t0").
   column("hasPhi", -2, "hasPhi").
   column("hasZed", -2, "hasZed").
   column("chi2", 2, "chi2").
   column("dof", 0, "degreesOfFreedom"); 

   table("DTRecHit1DPair").
   column("wheel", 0, "wireId.wheel").
   column("station", 0, "wireId.station").
   column("sector", 0, "wireId.sector").
   column("SL", 0, "wireId.superlayer").
   column("layer", 0, "wireId.layer").
   column("wire", 0, "wireId.wire").
   column("digiTime", 2, "digiTime");

   table("CSCSegment").
   column("endcap", 0, "cscDetId.endcap").
   column("station", 0, "cscDetId.station").
   column("ring", 0, "cscDetId.ring").
   column("chamber", 0, "cscDetId.chamber");

   table("reco::Vertex").
   column("x", 5).
   column("xError", 5).
   column("y", 5).
   column("yError", 5).
   column("z", 5).
   column("zError", 5).
   column("tracks", TableEntry::INT, "tracksSize").
   column("chi2", 3).
   column("ndof", 3);

   table("CaloTower").
   column("emEt", 1).
   column("hadEt", 1).
   column("et", 1, "Et").
   column("eta", 3).
   column("phi", 3);
   
   table("CaloRecHit").
   column("id", TableEntry::INT,"detid.rawId").
   column("energy",3).
   column("time",3).
   column("flags",TableEntry::INT,"flags");

   table("reco::PFCandidate").
   column("et", 1, "Et").
   column("eta", 3).
   column("phi", 3).
   column("ecalEnergy", 3,"ecalEnergy()").
   column("hcalEnergy", 3,"hcalEnergy()").
   column("track pt", 3,"trackRef().pt()");

   table("reco::Electron").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("E/p", 3, "eSuperClusterOverP").
   column("H/E", 3, "hadronicOverEm").
   column("fbrem", 3,"(trackMomentumAtVtx().R() - trackMomentumOut().R()) / trackMomentumAtVtx().R()" ).
   column("dei",3, "deltaEtaSuperClusterTrackAtVtx" ).
   column("dpi", 3, "deltaPhiSuperClusterTrackAtVtx()").
   column("charge", 0, "charge").
   column("isPF", 0, "isPF()").
   column("sieie", 3, "sigmaIetaIeta").
   column("isNotConv", 1, "passConversionVeto");

   table("pat::PackedCandidate").
   column("pT", 1, "pt").
   column("eta", 3).
   column("phi", 3).
   column("pdgId", 0).
   column("charge", 0).
   column("dxy", 3).
   column("dzAssociatedPV", 3, "dzAssociatedPV()");
}

FWTableViewManager::~FWTableViewManager()
{
}

//
// member functions
//

/** Define a new table for type @a name
 
    @a name the typename of the object contained in the table.

    @returns the TableHandle for this table, which can be
             used to create columns via the ::columns() method. 
    All subsequent calls for method column will be 
    relative to this table.

    If a table with the same name is already there, its entries
    are reset. 
  */
FWTableViewManager::TableHandle
FWTableViewManager::table(const char *name)
{
   TableHandle handle(name, m_tableFormats);
   return handle;
}

/** Define a column in the current table. 
 
    @a name to be used as header of the column.

    @a precision specifying the number of significant digits in the fractional part.

    @a expression to be used to retrieve the value from the object.
  */
FWTableViewManager::TableHandle &
FWTableViewManager::TableHandle::column(const char *name, int precision, const char *expression)
{
   TableEntry columnEntry;
   columnEntry.name = name;
   columnEntry.precision = precision;
   columnEntry.expression = expression;
   
   m_specs[m_name].push_back(columnEntry);
   return *this;
}

/** Helper function to do recursive lookup of specialized 
    table description for a given type @a key.
 */
FWTableViewManager::TableSpecs::iterator 
FWTableViewManager::tableFormatsImpl(const edm::TypeWithDict &key) 
{
   TableSpecs::iterator ret = m_tableFormats.find(key.name());
   if (ret != m_tableFormats.end())
      return ret;

   // if there is no exact match for the type, try the base classes
   edm::TypeBases bases(key);
   for (auto const& base : bases)  
   {
      ret = tableFormatsImpl(edm::BaseWithDict(base).typeOf());
      if (ret != m_tableFormats.end()) 
         return ret;
   }

   return m_tableFormats.end();
}

/** Find the entries for a given type @a key, possibly recursively
    searching recursively in the class hierarchy for a base class 
    that matches. 

    - If the recursion succeeds return the specific table.
    - Otherwise, create a dummy table with most common properties.

    @a key the edm::TypeWithDict of the collection for which we want
           to have the key definition.

    FIXME: how about actually inspecting the type and show all the int and floats 
           if no description is found??
  */
FWTableViewManager::TableSpecs::iterator
FWTableViewManager::tableFormats(const edm::TypeWithDict &key) 
{
   static const std::string isint("int");
   static const std::string isbool("bool");
   static const std::string isdouble("double");
   static const std::string isfloat("float");

   std::string keyType = key.name();

   TableSpecs::iterator ret = m_tableFormats.find(keyType);

   if (ret != m_tableFormats.end())
      return ret;
   
   ret = tableFormatsImpl(key); // recursive search for base classes

   if (ret != m_tableFormats.end()) 
      return ret;

   TableHandle handle = table(keyType.c_str());
   edm::TypeFunctionMembers functionMembers(key);
   for (auto const& member : functionMembers)
   {
      edm::FunctionWithDict m(member);
      if (m.functionParameterSize())
         continue;
      if (!m.isPublic())
         continue;
      if (!m.isConst())
         continue;
      if (m.finalReturnType().name() == isint)
         handle.column(m.name().c_str(), TableEntry::INT);
      else if (m.finalReturnType().name() == isbool)
         handle.column(m.name().c_str(), TableEntry::BOOL);
      else if (m.finalReturnType().name() == isdouble)
         handle.column(m.name().c_str(), 5);
      else if (m.finalReturnType().name() == isfloat)
         handle.column(m.name().c_str(), 3);
   }
   edm::TypeDataMembers dataMembers(key);
   for (auto const& member : dataMembers)
   {
      edm::MemberWithDict m(member);
      if (!m.isPublic())
         continue;
      if (!m.isConst())
         continue;
      if (m.typeOf().name() == isint)
         handle.column(m.name().c_str(), TableEntry::INT);
      else if (m.typeOf().name() == isbool)
         handle.column(m.name().c_str(), TableEntry::BOOL);
      else if (m.typeOf().name() == isdouble)
         handle.column(m.name().c_str(), 5);
      else if (m.typeOf().name() == isfloat)
         handle.column(m.name().c_str(), 3);
   }
   return m_tableFormats.find(keyType);
}

/** Helper function which uses TClass rather than edm::TypeWithDict.

    Otherwise identical to FWTableViewManager::tableFormats(const TClass &key).

  */
FWTableViewManager::TableSpecs::iterator 
FWTableViewManager::tableFormats(const TClass &key) 
{
   return tableFormats(edm::TypeWithDict::byName(key.GetName()));
}

class FWViewBase*
FWTableViewManager::buildView(TEveWindowSlot* iParent, const std::string& /*type*/)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWTableView> view(new FWTableView(iParent, this));
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   view->beingDestroyed_.connect(boost::bind(&FWTableViewManager::beingDestroyed,
					     this, _1));
   return view.get();
}

void
FWTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(Views::iterator it = m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) 
   {
      if(it->get() == iView) 
      {
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
   notifyViews();
}

/** Tell the views to update their item list. */
void
FWTableViewManager::notifyViews(void)
{
   for(size_t i = 0, e = m_views.size(); i != e; ++i)
   { 
      FWTableView *view = m_views[i].get();
      view->updateItems();
      view->dataChanged();
   } 
}

/** Remove @a iItem from the list 
    
    @a iItem the item to be removed.

 */
void 
FWTableViewManager::destroyItem(const FWEventItem *iItem)
{
   // remove the item from the list
   // FIXME: why doesn't it use erase?? Boh...
   for (size_t i = 0, e = m_items.size(); i != e; ++i)
   {
      if (m_items[i] != iItem)
         continue;
      m_items[i] = 0;
   }

   notifyViews();
}

/** Remove all items present in the view.
    
    This should watch the FWEventItemsManager::goingToClearItems_ signal.
  */
void
FWTableViewManager::removeAllItems(void)
{
   m_items.clear();
   notifyViews();
}

void
FWTableViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}

void
FWTableViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
   // tell the views to update their item lists
   // FIXME: doesn't this need to call updateItems as well
   // and hence notifyViews would be more appropriate?? Boh...
   dataChanged();
}

/** Notify all the views that colors have changed */
void
FWTableViewManager::colorsChanged()
{
   for(size_t i = 0, e = m_views.size(); i != e; ++i)
      m_views[i].get()->resetColors(colorManager());
}

void
FWTableViewManager::dataChanged()
{
   for(size_t i = 0, e = m_views.size(); i != e; ++i)
      m_views[i].get()->dataChanged();
}

FWTypeToRepresentations
FWTableViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   return returnValue;
}

const std::string FWTableViewManager::kConfigTypeNames = "typeNames";

void 
FWTableViewManager::addTo (FWConfiguration &iTo) const
{
   // if there are views, it's the job of the first view to store
   // the configuration (this is to avoid ordering problems in the
   // case of multiple views)
   if (!m_views.empty())
      return;
   // if there are no views, then it's up to us to store the column
   // formats.  This is done in addToImpl, which can be called by
   // FWTableView as well
   addToImpl(iTo);
}
     
void 
FWTableViewManager::addToImpl(FWConfiguration &iTo) const
{
   FWConfiguration typeNames(1);
   char prec[100];

   for (TableSpecs::const_iterator 
	iType = m_tableFormats.begin(),
	iType_end = m_tableFormats.end();
	iType != iType_end; ++iType) 
   {
      const std::string &typeName = iType->first;
      typeNames.addValue(typeName);
      FWConfiguration columns(1);
      const TableEntries &entries = iType->second;
      for (size_t ei = 0, ee = entries.size(); ei != ee; ++ei)
      {
         const TableEntry &entry = entries[ei];
         columns.addValue(entry.name);
         columns.addValue(entry.expression);
         columns.addValue((snprintf(prec, 100, "%d", entry.precision), prec));
      }
      iTo.addKeyValue(typeName, columns);
   }
   iTo.addKeyValue(kConfigTypeNames, typeNames);
}

void 
FWTableViewManager::setFrom(const FWConfiguration &iFrom)
{
   try
   {
      const FWConfiguration *typeNames = iFrom.valueForKey(kConfigTypeNames);
      if (typeNames == 0)
      {
         fwLog(fwlog::kWarning) << "no table column configuration stored, using defaults\n";
         return;
      }
            
      //NOTE: FWTableViewTableManagers hold pointers into m_tableFormats so if we
      // clear it those pointers would be invalid
      // instead we will just clear the lists and fill them with their new values
      //m_tableFormats.clear();
      for (FWConfiguration::StringValuesIt 
	   iType = typeNames->stringValues()->begin(),
	   iTypeEnd = typeNames->stringValues()->end(); 
           iType != iTypeEnd; ++iType) 
      {
         //std::cout << "reading type " << *iType << std::endl;
	 const FWConfiguration *columns = iFrom.valueForKey(*iType);
         if (!columns) continue;
         TableHandle handle = table(iType->c_str());
	 for (FWConfiguration::StringValuesIt 
	      it = columns->stringValues()->begin(),
	      itEnd = columns->stringValues()->end(); 
	      it != itEnd; ++it) 
         {
	    const std::string &name = *it++;
	    const std::string &expr = *it++;
	    int prec = atoi(it->c_str());
            handle.column(name.c_str(), prec, expr.c_str());
	 }
      }
   } 
   catch (...) 
   {
      // No info about types in the configuration; this is not an
      // error, it merely means that the types are handled by the
      // first FWTableView.
   }
}
