#ifndef ROO_SIMPLE_CACHE_SENTRY
#define ROO_SIMPLE_CACHE_SENTRY

#include "RooRealVar.h"
#include "RooSetProxy.h"
#include "TIterator.h"

class SimpleCacheSentry : public RooAbsArg {
    public:
        SimpleCacheSentry() ;
        SimpleCacheSentry(const RooRealVar &var) ;
        SimpleCacheSentry(const RooAbsCollection &vars) ;
        SimpleCacheSentry(const RooAbsArg &func, const RooArgSet *obs=0) ;
        SimpleCacheSentry(const SimpleCacheSentry &other, const char *newname = 0) ;
        RooSetProxy & deps() { return _deps; }
        const RooArgSet & deps() const { return _deps; }
        void addVar(const RooRealVar &var) { _deps.add(var); } 
        void addVars(const RooAbsCollection &vars) ; 
        void addFunc(const RooAbsArg &func, const RooArgSet *obs=0) ;
        bool good() const { return !isValueDirty(); } 
        bool empty() const { return _deps.getSize() == 0; }
        void reset() { clearValueDirty(); } 
        // base class methods to be implemented
        virtual TObject* clone(const char* newname) const { return new SimpleCacheSentry(*this, newname); }
        virtual RooAbsArg *createFundamental(const char* newname=0) const { return 0; }
        virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) { return false; }
        virtual void writeToStream(ostream& os, Bool_t compact) const { }
        virtual Bool_t operator==(const RooAbsArg& other) { return this == &other; }
        virtual void syncCache(const RooArgSet* nset=0) {}
        virtual void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE, Bool_t setValDirty=kTRUE) {}
        virtual void attachToTree(TTree& t, Int_t bufSize=32000) {}
        virtual void attachToVStore(RooVectorDataStore& vstore) {}
        virtual void setTreeBranchStatus(TTree& t, Bool_t active) {}
        virtual void fillTreeBranch(TTree& t) {}
    private:
        RooSetProxy _deps;
        ClassDef(SimpleCacheSentry,1) 
};

#endif
