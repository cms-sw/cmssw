#include "../interface/SimpleCacheSentry.h"

SimpleCacheSentry::SimpleCacheSentry() :  _deps("deps","deps",this)   {}

SimpleCacheSentry::SimpleCacheSentry(const RooRealVar &var) :
    _deps("deps","deps",this)   
{
    addVar(var);
}

SimpleCacheSentry::SimpleCacheSentry(const RooAbsCollection &vars) :
    _deps("deps","deps",this)   
{
    addVars(vars);
}


SimpleCacheSentry::SimpleCacheSentry(const RooAbsArg &func, const RooArgSet *obs) :
    _deps("deps","deps",this)   
{
    addFunc(func,obs);
}

SimpleCacheSentry::SimpleCacheSentry(const SimpleCacheSentry &other, const char *newname) :
    _deps("deps",this,other._deps)   
{
}

void SimpleCacheSentry::addVars(const RooAbsCollection &vars) 
{
    TIterator *iter = vars.createIterator();
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (_deps.containsInstance(*a)) continue;
        if (a->isDerived()) addFunc(*a);
        else _deps.add(*a);
    }
    delete iter;
}

void SimpleCacheSentry::addFunc(const RooAbsArg &func, const RooArgSet *obs) 
{
    RooArgSet *deps = func.getParameters(obs,false);
    addVars(*deps);
    delete deps;
}


