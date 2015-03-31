#include "DataFormats/PatCandidates/interface/UserData.h"
// Note: these two below are allowed in FWLite even if they come from FWCore
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"

void pat::UserData::checkDictionaries(const std::type_info &type) {
    if (!edm::hasDictionary(type)) {
        int status = 0;
        char * demangled = abi::__cxa_demangle(type.name(),  0, 0, &status);
        std::string typeName(status == 0 ? demangled : type.name());
        if ((demangled != 0) && (status == 0)) free(demangled);
        throw edm::Exception(edm::errors::DictionaryNotFound)
            << "   No REFLEX data dictionary found for the following class:\n\t"
            << typeName 
            << "\n   Most likely the dictionary was never generated,\n"
            << "   but it may be that it was generated in the wrong package.\n"
            << "   Please add (or move) the specification\n"
            << "\t<class name=\"" << typeName << "\" />\n"
            << "   to the appropriate classes_def.xml file.\n"
            << "   If the class is a template instance, you may need\n"
            << "   to define a dummy variable of this type in classes.h.\n"
            << "   Also, if this class has any transient members,\n"
            << "   you need to specify them in classes_def.xml.\n";
    } // check for dictionary
}
