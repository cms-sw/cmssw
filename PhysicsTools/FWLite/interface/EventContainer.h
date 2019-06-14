// -*- C++ -*-

#if !defined(EventContainer_H)
#define EventContainer_H

#include <map>
#include <string>
#include <typeinfo>

#include "TH1.h"
#include "TFile.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "PhysicsTools/FWLite/interface/TH1Store.h"

namespace fwlite {

  class EventContainer : public EventBase {
  public:
    //////////////////////
    // Public Constants //
    //////////////////////

    typedef std::map<std::string, std::string> SSMap;
    typedef void (*FuncPtr)(std::string&);

    /////////////
    // friends //
    /////////////
    // tells particle data how to print itself out
    friend std::ostream& operator<<(std::ostream& o_stream, const EventContainer& rhs);

    //////////////////////////
    //            _         //
    // |\/|      |_         //
    // |  |EMBER | UNCTIONS //
    //                      //
    //////////////////////////

    /////////////////////////////////
    // Constructors and Destructor //
    /////////////////////////////////
    EventContainer(optutl::CommandLineParser& parser, FuncPtr funcPtr = nullptr);
    ~EventContainer() override;

    ////////////////
    // One Liners //
    ////////////////

    // return number of events seen
    int eventsSeen() const { return m_eventsSeen; }

    //////////////////////////////
    // Regular Member Functions //
    //////////////////////////////

    // adds a histogram pointer to the map.  You can specify a
    // directory as well if you wish.
    void add(TH1* histPtr, const std::string& directory = "");

    // given a string, returns corresponding histogram pointer
    TH1* hist(const std::string& name);
    TH1* hist(const char* name) { return hist((const std::string)name); }
    TH1* hist(const TString& name) { return hist((const char*)name); }

    // return this containers parser
    optutl::CommandLineParser& parser();

    ///////////////////////////////////////////////////////////////////
    // Implement the two functions needed to make this an EventBase. //
    ///////////////////////////////////////////////////////////////////
    bool getByLabel(const std::type_info& iInfo,
                    const char* iModuleLabel,
                    const char* iProductInstanceLabel,
                    const char* iProcessLabel,
                    void* oData) const override;

    const std::string getBranchNameFor(const std::type_info& iInfo,
                                       const char* iModuleLabel,
                                       const char* iProductInstanceLabel,
                                       const char* iProcessLabel) const override;

    const EventContainer& operator++() override;

    const EventContainer& toBegin() override;

    bool atEnd() const override;

    edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const override {
      return m_eventBasePtr->triggerNames(triggerResults);
    }

    edm::TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const override {
      return m_eventBasePtr->triggerResultsByName(triggerResults);
    }

    Long64_t fileIndex() const override { return m_eventBasePtr->fileIndex(); }
    Long64_t secondaryFileIndex() const override { return m_eventBasePtr->secondaryFileIndex(); }

    edm::EventAuxiliary const& eventAuxiliary() const override { return m_eventBasePtr->eventAuxiliary(); }

    template <class T>
    bool getByLabel(const edm::InputTag& tag, edm::Handle<T>& handle) const {
      return m_eventBasePtr->getByLabel(tag, handle);
    }
    /////////////////////////////
    // Static Member Functions //
    /////////////////////////////

  private:
    //////////////////////////////
    // Private Member Functions //
    //////////////////////////////

    // stop the copy constructor
    EventContainer(const EventContainer& rhs) {}

    /////////////////////////
    // Private Member Data //
    /////////////////////////

    fwlite::EventBase* m_eventBasePtr;
    TH1Store m_histStore;
    std::string m_outputName;
    int m_eventsSeen;
    int m_maxWanted;
    int m_outputEvery;
    optutl::CommandLineParser* m_parserPtr;

    ////////////////////////////////
    // Private Static Member Data //
    ////////////////////////////////

    static bool sm_autoloaderCalled;
  };
}  // namespace fwlite

#endif  // EventContainer_H
