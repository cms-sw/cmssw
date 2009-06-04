#include "PhysicsTools/PatUtils/interface/StringParserTools.h"
#include <typeinfo>

PATStringObjectFunction::PATStringObjectFunction(const std::string &string) :
    expr_(string)
{
   candFunc_ = tryGet<reco::Candidate>(string);
   if (!candFunc_.get()) {
       eleFunc_ = tryGet<pat::Electron>(string);
       muFunc_  = tryGet<pat::Muon>(string);
       tauFunc_ = tryGet<pat::Tau>(string);
       gamFunc_ = tryGet<pat::Photon>(string);
       jetFunc_ = tryGet<pat::Jet>(string);
       metFunc_ = tryGet<pat::MET>(string);
       gpFunc_  = tryGet<pat::GenericParticle>(string);
       pfFunc_  = tryGet<pat::PFParticle>(string);
   } 
}

double
PATStringObjectFunction::operator()(const reco::Candidate &c) const  {
    if (candFunc_.get()) return (*candFunc_)(c);
    const std::type_info &type = typeid(c);
    if      (type == typeid(pat::Electron       )) return tryEval<pat::Electron       >(c, eleFunc_);
    else if (type == typeid(pat::Muon           )) return tryEval<pat::Muon           >(c,  muFunc_);
    else if (type == typeid(pat::Tau            )) return tryEval<pat::Tau            >(c, tauFunc_);
    else if (type == typeid(pat::Photon         )) return tryEval<pat::Photon         >(c, gamFunc_);
    else if (type == typeid(pat::Jet            )) return tryEval<pat::Jet            >(c, jetFunc_);
    else if (type == typeid(pat::MET            )) return tryEval<pat::MET            >(c, metFunc_);
    else if (type == typeid(pat::GenericParticle)) return tryEval<pat::GenericParticle>(c,  gpFunc_);
    else if (type == typeid(pat::PFParticle     )) return tryEval<pat::PFParticle     >(c,  pfFunc_);
    else throw cms::Exception("Type Error") << "Cannot evaluate '" << expr_ << "' on an object of unsupported type " << type.name() << "\n"; 
}

void 
PATStringObjectFunction::throwBadType(const std::type_info &ty) const  {
    throw cms::Exception("Type Error")  << "Expression '" << expr_ << "' can't be evaluated on an object of type " << ty.name() << "\n(hint: use c++filt to demangle the type name)\n";
}

PATStringCutObjectSelector::PATStringCutObjectSelector(const std::string &string) :
    expr_(string)
{
   candFunc_ = tryGet<reco::Candidate>(string);
   if (!candFunc_.get()) {
       eleFunc_ = tryGet<pat::Electron>(string);
       muFunc_  = tryGet<pat::Muon>(string);
       tauFunc_ = tryGet<pat::Tau>(string);
       gamFunc_ = tryGet<pat::Photon>(string);
       jetFunc_ = tryGet<pat::Jet>(string);
       metFunc_ = tryGet<pat::MET>(string);
       gpFunc_  = tryGet<pat::GenericParticle>(string);
       pfFunc_  = tryGet<pat::PFParticle>(string);
   } 
}

bool
PATStringCutObjectSelector::operator()(const reco::Candidate &c) const  {
    if (candFunc_.get()) return (*candFunc_)(c);
    const std::type_info &type = typeid(c);
    if      (type == typeid(pat::Electron       )) return tryEval<pat::Electron       >(c, eleFunc_);
    else if (type == typeid(pat::Muon           )) return tryEval<pat::Muon           >(c,  muFunc_);
    else if (type == typeid(pat::Tau            )) return tryEval<pat::Tau            >(c, tauFunc_);
    else if (type == typeid(pat::Photon         )) return tryEval<pat::Photon         >(c, gamFunc_);
    else if (type == typeid(pat::Jet            )) return tryEval<pat::Jet            >(c, jetFunc_);
    else if (type == typeid(pat::MET            )) return tryEval<pat::MET            >(c, metFunc_);
    else if (type == typeid(pat::GenericParticle)) return tryEval<pat::GenericParticle>(c,  gpFunc_);
    else if (type == typeid(pat::PFParticle     )) return tryEval<pat::PFParticle     >(c,  pfFunc_);
    else throw cms::Exception("Type Error") << "Cannot evaluate '" << expr_ << "' on an object of unsupported type " << type.name() << "\n"; 
}

void 
PATStringCutObjectSelector::throwBadType(const std::type_info &ty) const  {
    throw cms::Exception("Type Error")  << "Expression '" << expr_ << "' can't be evaluated on an object of type " << ty.name() << "\n(hint: use c++filt to demangle the type name)\n";
}
