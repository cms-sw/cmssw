// these includes are FWLite-safe
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
// these are from ROOT, so they're safe too
#include <TString.h>
#include <TNamed.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "PhysicsTools/FWLite/interface/ScannerHelpers.h"
#endif

namespace fwlite {
    class EventSelector : public TNamed {
        public:
            EventSelector(const char *name="", const char *title="") : TNamed(name,title) {}
            virtual ~EventSelector() {}
            virtual bool accept(const fwlite::EventBase &ev) const = 0;
    };    

    class RunLumiSelector : public EventSelector {
        public:
            RunLumiSelector(const char *name="", const char *title="") : EventSelector(name,title) {}
            RunLumiSelector(int run, int firstLumi=0, int lastLumi = 9999999) :
                EventSelector(TString::Format("run%d_lumi%d_%d", run, firstLumi, lastLumi),
                              TString::Format("Run %d, Lumi range [%d, %d]", run, firstLumi, lastLumi))
                { add(run, firstLumi, lastLumi); }

            virtual ~RunLumiSelector() {}
            virtual bool accept(const fwlite::EventBase &ev) const {
                return accept(ev.id().run(), ev.luminosityBlock());
            }
            void add(int run, int firstLumi=0, int lastLumi = 9999999) {
                runs.push_back(run);
                firstLumis.push_back(firstLumi);
                lastLumis.push_back(lastLumi);
            }
            void clear() {
                runs.clear();
                firstLumis.clear();
                lastLumis.clear();
            }
            bool accept(int run, int luminosityBlock) const {
                if (runs.empty()) return true;
                for (int i = 0, n = runs.size(); i < n; ++i) {
                    if (runs[i] == run) {
                        if ((firstLumis[i] <= luminosityBlock) && (luminosityBlock <= lastLumis[i])) return true;
                    }
                }
                return false;
            }
                        
        private:
            std::vector<int> runs, firstLumis, lastLumis;
    };

    template<typename Collection>
    class ObjectCountSelector :  public EventSelector {
        public:
            ObjectCountSelector(const char *label, const char *instance, const char *process,
                                const char *cut,   int minNumber=1, int maxNumber=-1) :
                label_(label), instance_(instance),
                min_(minNumber), max_(maxNumber),
                scanner(new helper::ScannerBase(helper::Parser::elementType(edm::TypeWithDict(HandleT::TempWrapT::typeInfo()))))
            {
                scanner->setCut(cut);
                scanner->setIgnoreExceptions(true);
            }
            ~ObjectCountSelector() { delete scanner; }
            virtual bool accept(const fwlite::EventBase &ev) const {
                int nfound = 0;
                HandleT handle; // here, not as a datamember, otherwise CINT segfaults (!?)
                handle.getByLabel(ev, label_.c_str(), instance_.c_str(), process_.c_str());
                const Collection & vals = *handle;
                for (size_t j = 0, n = vals.size(); j < n; ++j) {
                    if (scanner->test(&vals[j])) nfound++;
                }
                return (nfound >= min_ && (max_ == -1 || nfound <= max_));
            }
            void setCut(const char *cut) { scanner->setCut(cut); }
            void setMin(int minNumber)   { min_ = minNumber; }
            void setMax(int maxNumber)   { max_ = maxNumber; }
            void setIgnoreExceptions(bool ignoreThem=true) { scanner->setIgnoreExceptions(ignoreThem); }
        protected:
            typedef fwlite::Handle<Collection> HandleT;
            std::string    label_, instance_, process_;
            int min_, max_;
            helper::ScannerBase *scanner; // has to be a pointer, otherwise CINT segfaults in setCut (!?)
            // prevent copy c-tor and assignment
            ObjectCountSelector(const fwlite::ObjectCountSelector<Collection> &other) ;
            ObjectCountSelector & operator=(const fwlite::ObjectCountSelector<Collection> &other) ;
    };
}
