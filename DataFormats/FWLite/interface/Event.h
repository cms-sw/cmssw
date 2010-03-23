#ifndef DataFormats_FWLite_Event_h
#define DataFormats_FWLite_Event_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Event
//
/**\class Event Event.h DataFormats/FWLite/interface/Event.h

   Description: Provide event data access in FWLite

   Usage:
   This class is meant to allow one to loop over all events in a TFile and then
 read the data in an Event in a manner analogous to how data is read in the full framework.
 A typical use would be
 \code
 TFile f("foo.root");
 fwlite::Event ev(&f);
 for(ev.toBeing(); ! ev.atEnd(); ++ev) {
    fwlite::Handle<std::vector<Foo> > foos;
    foos.getByLabel(ev, "myFoos");
 }
 \endcode
 The above example will work for both CINT and compiled code. However, it is possible to exactly
 match the full framework if you only intend to compile your code.  In that case the access
 would look like

 \code
 TFile f("foo.root");
 fwlite::Event ev(&f);
 
 edm::InputTag fooTag("myFoos");
 for(ev.toBeing(); ! ev.atEnd(); ++ev) {
    edm::Handle<std::vector<Foo> > foos;
    ev.getByLabel(fooTag, foos);
 }
 \endcode
 
*/
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:01:20 EDT 2007
// $Id: Event.h,v 1.33 2010/03/12 14:55:38 ewv Exp $
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <cstring>
#include <string>

#include "TBranch.h"
#include "Rtypes.h"
#include "Reflex/Object.h"

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"

// forward declarations
namespace edm {
   class EDProduct;
   class ProductRegistry;
   class BranchDescription;
   class EDProductGetter;
   class EventAux;
   class Timestamp;
   class TriggerResults;
   class TriggerNames;
   class TriggerResultsByName;
}
class TCut;

namespace fwlite {
   class BranchMapReader;
   class HistoryGetterBase;
   class DataGetterHelper;
   class RunFactory;
   class Event : public EventBase
   {

      public:
         // NOTE: Does NOT take ownership so iFile must remain around
         // at least as long as Event
         Event(TFile* iFile);
         virtual ~Event();

         ///Advance to next event in the TFile
         const Event& operator++();

         ///Go to the event at index iIndex
         bool to (Long64_t iIndex);

         ///Go to event by Run & Event number
         bool to(const edm::EventID &id);
         bool to(edm::RunNumber_t run, edm::EventNumber_t event);
         bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

         /// Go to the very first Event.
         const Event& toBegin();
      
         // ---------- const member functions ---------------------
         ///Return the branch name in the TFile which contains the data 
         virtual const std::string getBranchNameFor(const std::type_info&,
                                                    const char* iModuleLabel,
                                                    const char* iProductInstanceLabel,
                                                    const char* iProcessName) const;

         using fwlite::EventBase::getByLabel;
         /// This function should only be called by fwlite::Handle<>
         virtual bool getByLabel(const std::type_info&, const char*, const char*, const char*, void*) const;
         //void getByBranchName(const std::type_info&, const char*, void*&) const;

         ///Properly setup for edm::Ref, etc and then call TTree method
         void       draw(Option_t* opt);
         Long64_t   draw(const char* varexp, const TCut& selection, Option_t* option = "", Long64_t nentries = 1000000000, Long64_t firstentry = 0);
         Long64_t   draw(const char* varexp, const char* selection, Option_t* option = "", Long64_t nentries = 1000000000, Long64_t firstentry = 0); 
         Long64_t   scan(const char* varexp = "", const char* selection = "", Option_t* option = "", Long64_t nentries = 1000000000, Long64_t firstentry = 0);

         bool isValid() const;
         operator bool () const;
         virtual bool atEnd() const;

         ///Returns number of events in the file
         Long64_t size() const;

         virtual edm::EventAuxiliary const& eventAuxiliary() const;

         const std::vector<edm::BranchDescription>& getBranchDescriptions() const {
            return branchMap_.getBranchDescriptions();
         }
         const std::vector<std::string>& getProcessHistory() const;
         TFile* getTFile() const {
            return branchMap_.getFile();
         }

         edm::EDProduct const* getByProductID(edm::ProductID const&) const;

         virtual edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const;

         virtual edm::TriggerResultsByName triggerResultsByName(std::string const& process) const;

         fwlite::LuminosityBlock const& getLuminosityBlock() const;
         fwlite::Run             const& getRun() const;

         // ---------- static member functions --------------------
         static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);


      private:
         friend class internal::ProductGetter;
         friend class ChainEvent;
         friend class EventHistoryGetter;

         Event(const Event&); // stop default

         const Event& operator=(const Event&); // stop default

         const edm::ProcessHistory& history() const;
         void updateAux(Long_t eventIndex) const;
         void fillFileIndex() const;
         void fillParameterSetRegistry() const;
         void setGetter( boost::shared_ptr<edm::EDProductGetter> getter ) { return dataHelper_.setGetter(getter);}

         // ---------- member data --------------------------------
         TFile* file_;
         // TTree* eventTree_;
         TTree* eventHistoryTree_;
         // Long64_t eventIndex_;
         mutable boost::shared_ptr<fwlite::LuminosityBlock>  lumi_;
         mutable boost::shared_ptr<fwlite::Run>  run_;
         mutable fwlite::BranchMapReader branchMap_;

         //takes ownership of the strings used by the DataKey keys in data_
         mutable std::vector<const char*> labels_;
         mutable edm::ProcessHistoryMap historyMap_;
         mutable std::vector<edm::EventProcessHistoryID> eventProcessHistoryIDs_;
         mutable std::vector<std::string> procHistoryNames_;
         mutable edm::EventAuxiliary aux_;
         mutable edm::FileIndex fileIndex_;
         edm::EventAuxiliary* pAux_;
         edm::EventAux* pOldAux_;
         TBranch* auxBranch_;
         int fileVersion_;
         mutable bool parameterSetRegistryFilled_;

         fwlite::DataGetterHelper dataHelper_;
         mutable boost::shared_ptr<RunFactory> runFactory_;
   };

}
#endif /*__CINT__ */
#endif
