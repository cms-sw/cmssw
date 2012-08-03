// -*- C++ -*-
//
// Package:     Modules
// Class  :     XMLOutputModule
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Aug  4 20:45:44 EDT 2006
//

// system include files
#include <fstream>
#include <string>
#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/Selections.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files

//
// constants, enums and typedefs
//

namespace edm {
  class XMLOutputModule : public OutputModule {

   public:
      XMLOutputModule(ParameterSet const&);
      virtual ~XMLOutputModule();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void write(EventPrincipal const& e);
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&) {}
      virtual void writeRun(RunPrincipal const&) {}

      XMLOutputModule(XMLOutputModule const&); // stop default

      XMLOutputModule const& operator=(XMLOutputModule const&); // stop default

      // ---------- member data --------------------------------
      std::ofstream stream_;
      std::string indentation_;
  };

  namespace {
    void doNotDelete(void*) {}
    void callDestruct(ObjectWithDict* iObj) {
      iObj->destruct();
    }
    //Handle memory for calls to invoke
    // We handle Ref's by using an external void* buffer (which we do not delete) while everything else
    // we create the proper object (and therefore must delete it)
    boost::shared_ptr<ObjectWithDict> initReturnValue(MemberWithDict const& iMember,
                                                      ObjectWithDict* iObj,
                                                      void** iRefBuffer) {
      TypeWithDict returnType = iMember.typeOf().returnType();
      if(returnType.isReference()) {
        *iObj = ObjectWithDict(returnType, iRefBuffer);
        return boost::shared_ptr<ObjectWithDict>(iObj, doNotDelete);
      }
      *iObj = returnType.construct();
      return boost::shared_ptr<ObjectWithDict>(iObj, callDestruct);
    }

    //remove characters from a string which are not allowed to be used in XML
    std::string formatXML(std::string const& iO) {
      std::string result(iO);
      static std::string const kSubs("<>&");
      static std::string const kLeft("&lt;");
      static std::string const kRight("&gt;");
      static std::string const kAmp("&");

      std::string::size_type i = 0;
      while(std::string::npos != (i = result.find_first_of(kSubs, i))) {
        switch(result.at(i)) {
          case '<':
            result.replace(i, 1, kLeft);
            break;
          case '>':
            result.replace(i, 1, kRight);
            break;
          case '&':
            result.replace(i, 1, kAmp);
        }
        ++i;
      }
      return result;
    }

    char const* kNameValueSep = "\">";
    char const* kContainerOpen = "<container size=\"";
    char const* kContainerClose = "</container>";
    std::string const kObjectOpen = "<object type=\"";
    std::string const kObjectClose = "</object>";
    ///convert the object information to the correct type and print it
    #define FILLNAME(_type_) s_toName[typeid(_type_).name()]= #_type_;
    std::string const& typeidToName(std::type_info const& iID) {
      static std::map<std::string, std::string> s_toName;
      if(s_toName.empty()) {
        FILLNAME(short);
        FILLNAME(int);
        FILLNAME(long);
        FILLNAME(long long);

        FILLNAME(unsigned short);
        FILLNAME(unsigned int);
        FILLNAME(unsigned long);
        FILLNAME(unsigned long long);

        FILLNAME(double);
        FILLNAME(float);
      }
      return s_toName[iID.name()];
    }

    template<typename T>
    void doPrint(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, ObjectWithDict const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << typeidToName(typeid(T)) << kNameValueSep
        << *reinterpret_cast<T*>(iObject.address()) << iPostfix << "\n";
    }

    template<>
    void doPrint<char>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, ObjectWithDict const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "char" << kNameValueSep
        << static_cast<int>(*reinterpret_cast<char*>(iObject.address())) << iPostfix << "\n";
    }

    template<>
    void doPrint<unsigned char>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, ObjectWithDict const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "unsigned char" << kNameValueSep << static_cast<unsigned int>(*reinterpret_cast<unsigned char*>(iObject.address())) << iPostfix << "\n";
    }

    template<>
    void doPrint<bool>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, ObjectWithDict const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "bool" << kNameValueSep
        << ((*reinterpret_cast<bool*>(iObject.address()))?"true":"false") << iPostfix << "\n";
    }


    typedef void(*FunctionType)(std::ostream&, std::string const&,
                                std::string const&, ObjectWithDict const&, std::string const&);
    typedef std::map<std::string, FunctionType> TypeToPrintMap;

    template<typename T>
    void addToMap(TypeToPrintMap& iMap){
      iMap[typeid(T).name()]=doPrint<T>;
    }

    bool printAsBuiltin(std::ostream& oStream,
                               std::string const& iPrefix,
                               std::string const& iPostfix,
                               ObjectWithDict const iObject,
                               std::string const& iIndent){
      typedef void(*FunctionType)(std::ostream&, std::string const&, std::string const&, ObjectWithDict const&, std::string const&);
      typedef std::map<std::string, FunctionType> TypeToPrintMap;
      static TypeToPrintMap s_map;
      static bool isFirst = true;
      if(isFirst){
        addToMap<bool>(s_map);
        addToMap<char>(s_map);
        addToMap<short>(s_map);
        addToMap<int>(s_map);
        addToMap<long>(s_map);
        addToMap<unsigned char>(s_map);
        addToMap<unsigned short>(s_map);
        addToMap<unsigned int>(s_map);
        addToMap<unsigned long>(s_map);
        addToMap<float>(s_map);
        addToMap<double>(s_map);
        isFirst=false;
      }
      TypeToPrintMap::iterator itFound =s_map.find(iObject.typeName());
      if(itFound == s_map.end()){

        return false;
      }
      itFound->second(oStream, iPrefix, iPostfix, iObject, iIndent);
      return true;
    }

    bool printAsContainer(std::ostream& oStream,
                          std::string const& iPrefix,
                          std::string const& iPostfix,
                          ObjectWithDict const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta);

    void printDataMembers(std::ostream& oStream,
                          ObjectWithDict const& iObject,
                          TypeWithDict const& iType,
                          std::string const& iIndent,
                          std::string const& iIndentDelta);

    void printObject(std::ostream& oStream,
                     std::string const& iPrefix,
                     std::string const& iPostfix,
                     ObjectWithDict const& iObject,
                     std::string const& iIndent,
                     std::string const& iIndentDelta) {
      ObjectWithDict objectToPrint = iObject;
      std::string indent(iIndent);
      if(iObject.isPointer()) {
        oStream << iIndent << iPrefix << formatXML(iObject.typeOf().name(TypeNameHandling::Scoped)) << "\">\n";
        indent +=iIndentDelta;
        int size = (0!=iObject.address()) ? (0!=*reinterpret_cast<void**>(iObject.address())?1:0) : 0;
        oStream << indent << kContainerOpen << size << "\">\n";
        if(size) {
          std::string indent2 = indent + iIndentDelta;
          ObjectWithDict obj(iObject.toType(), *reinterpret_cast<void**>(iObject.address()));
          obj = obj.castObject(obj.dynamicType());
          printObject(oStream, kObjectOpen, kObjectClose, obj, indent2, iIndentDelta);
        }
        oStream << indent << kContainerClose << "\n";
        oStream << iIndent << iPostfix << "\n";
        TypeWithDict pointedType = iObject.toType();
        if(TypeWithDict::byName("void") == pointedType || pointedType.isPointer() || iObject.address()==0) {
          return;
        }
        return;
        /*
        //have the code that follows print the contents of the data to which the pointer points
        objectToPrint = ObjectWithDict(pointedType, iObject.address());
        //try to convert it to its actual type (assuming the original type was a base class)
        objectToPrint = ObjectWithDict(objectToPrint.castObject(objectToPrint.dynamicType()));
        indent +=iIndentDelta;
        */
      }
      std::string typeName(objectToPrint.typeOf().name(TypeNameHandling::Scoped));
      if(typeName.empty()){
        typeName="{unknown}";
      }

      //see if we are dealing with a typedef
      TypeWithDict objectType = objectToPrint.typeOf();
      bool wasTypedef = false;
      while(objectType.isTypedef()) {
         objectType = objectType.toType();
         wasTypedef = true;
      }
      if(wasTypedef){
         ObjectWithDict tmp(objectType, objectToPrint.address());
         objectToPrint = tmp;
      }
      if(printAsBuiltin(oStream, iPrefix, iPostfix, objectToPrint, indent)) {
        return;
      }
      if(printAsContainer(oStream, iPrefix, iPostfix, objectToPrint, indent, iIndentDelta)){
        return;
      }

      oStream << indent << iPrefix << formatXML(typeName) << "\">\n";
      printDataMembers(oStream, objectToPrint, objectType, indent+iIndentDelta, iIndentDelta);
      oStream << indent << iPostfix << "\n";

    }

    void printDataMembers(std::ostream& oStream,
                          ObjectWithDict const& iObject,
                          TypeWithDict const& iType,
                          std::string const& iIndent,
                          std::string const& iIndentDelta) {
      //print all the base class data members
      TypeBases bases(iType);
      for(auto const& baseMember : bases) {
        BaseWithDict base(baseMember);
        printDataMembers(oStream, iObject.castObject(base.toType()), base.toType(), iIndent, iIndentDelta);
      }
      static std::string const kPrefix("<datamember name=\"");
      static std::string const ktype("\" type=\"");
      static std::string const kPostfix("</datamember>");

      TypeDataMembers dataMembers(iType);
      for(auto const& dataMember : dataMembers) {
        MemberWithDict member(dataMember);
        //std::cout << "     debug " << member.name() << " " << member.typeOf().name() << "\n";
        if (member.isTransient()) {
          continue;
        }
        try {
          std::string prefix = kPrefix + member.name() + ktype;
          printObject(oStream,
                      prefix,
                      kPostfix,
                      member.get(iObject),
                      iIndent,
                      iIndentDelta);
        }catch(std::exception& iEx) {
          std::cout << iIndent << member.name() << " <exception caught("
          << iEx.what() << ")>\n";
        }
      }
    }

    bool printContentsOfStdContainer(std::ostream& oStream,
                                     std::string const& iPrefix,
                                     std::string const& iPostfix,
                                     ObjectWithDict iBegin,
                                     ObjectWithDict const& iEnd,
                                     std::string const& iIndent,
                                     std::string const& iIndentDelta){
      size_t size = 0;
      std::ostringstream sStream;
      if(iBegin.typeOf() != iEnd.typeOf()) {
        std::cerr << " begin (" << iBegin.typeOf().name(TypeNameHandling::Scoped) << ") and end ("
          << iEnd.typeOf().name(TypeNameHandling::Scoped) << ") are not the same type" << std::endl;
        throw std::exception();
      }
      try {
        MemberWithDict compare(iBegin.typeOf().memberByName("operator!="));
        if(!compare) {
          //std::cerr << "no 'operator!=' for " << iBegin.typeOf().name() << std::endl;
          return false;
        }
        MemberWithDict incr(iBegin.typeOf().memberByName("operator++"));
        if(!incr) {
          //std::cerr << "no 'operator++' for " << iBegin.typeOf().name() << std::endl;
          return false;
        }
        MemberWithDict deref(iBegin.typeOf().memberByName("operator*"));
        if(!deref) {
          //std::cerr << "no 'operator*' for " << iBegin.typeOf().name() << std::endl;
          return false;
        }

        std::string indexIndent = iIndent+iIndentDelta;
        int dummy=0;
        //std::cerr << "going to loop using iterator " << iBegin.typeOf().name(NameHAndling::Scoped) << std::endl;

        std::vector<void*> compareArgs;
        compareArgs.push_back(iEnd.address());
        std::vector<void*> incArgs;
        incArgs.push_back(&dummy);
        bool compareResult;
        ObjectWithDict objCompareResult(TypeWithDict(typeid(bool)), &compareResult);
        ObjectWithDict objIncr;
        void* objIncrRefBuffer;
        boost::shared_ptr<ObjectWithDict> incrMemHolder = initReturnValue(incr, &objIncr, &objIncrRefBuffer);
        for(;
          compare.invoke(iBegin, &objCompareResult, compareArgs), compareResult;
          incr.invoke(iBegin, &objIncr, incArgs), ++size) {
          //std::cerr << "going to print" << std::endl;
          ObjectWithDict iTemp;
          void* derefRefBuffer;
          boost::shared_ptr<ObjectWithDict> derefMemHolder = initReturnValue(deref, &iTemp, &derefRefBuffer);
          deref.invoke(iBegin, &iTemp);
          if(iTemp.isReference()) {
            iTemp = ObjectWithDict(iTemp.typeOf(), derefRefBuffer);
          }
          printObject(sStream, kObjectOpen, kObjectClose, iTemp, indexIndent, iIndentDelta);
          //std::cerr << "printed" << std::endl;
        }
      } catch(std::exception const& iE) {
        std::cerr << "while printing std container caught exception " << iE.what() << std::endl;
        return false;
      }
      oStream << iPrefix << iIndent << kContainerOpen << size << "\">\n";
      oStream << sStream.str();
      oStream << iIndent << kContainerClose << std::endl;
      oStream << iPostfix;
      //std::cerr << "finished loop" << std::endl;
      return true;
    }

    bool printAsContainer(std::ostream& oStream,
                          std::string const& iPrefix, std::string const& iPostfix,
                          ObjectWithDict const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta) {
      ObjectWithDict sizeObj;
      try {
        size_t temp; //used to hold the memory for the return value
        sizeObj = ObjectWithDict(TypeWithDict(typeid(size_t)), &temp);
        iObject.invoke("size", &sizeObj);

        if(sizeObj.typeOf().typeInfo() != typeid(size_t)) {
          throw std::exception();
        }
        size_t size = *reinterpret_cast<size_t*>(sizeObj.address());
        MemberWithDict atMember;
        atMember = iObject.typeOf().memberByName("at");
        if(!atMember) {
          throw std::exception();
        }
        std::string typeName(iObject.typeOf().name(TypeNameHandling::Scoped));
        if(typeName.empty()){
          typeName="{unknown}";
        }

        oStream << iIndent << iPrefix << formatXML(typeName) << "\">\n"
          << iIndent << kContainerOpen << size << "\">\n";
        ObjectWithDict contained;
        std::string indexIndent=iIndent+iIndentDelta;
        for(size_t index = 0; index != size; ++index) {
          void* atRefBuffer;
          boost::shared_ptr<ObjectWithDict> atMemHolder = initReturnValue(atMember, &contained, &atRefBuffer);

          std::vector<void*> args;
          args.push_back(&index);
          atMember.invoke(iObject, &contained, args);
          if(contained.isReference()) {
            contained = ObjectWithDict(contained.typeOf(), atRefBuffer);
          }
          //std::cout << "invoked 'at'" << std::endl;
          try {
            printObject(oStream, kObjectOpen, kObjectClose, contained, indexIndent, iIndentDelta);
          }catch(std::exception& iEx) {
            std::cout << iIndent << " <exception caught("
            << iEx.what() << ")>\n";
          }
        }
        oStream << iIndent << kContainerClose << std::endl;
        oStream << iIndent << iPostfix << std::endl;
        return true;
      } catch(std::exception const& x){
        //std::cerr << "failed to invoke 'at' because " << x.what() << std::endl;
        try {
          //oStream << iIndent << iPrefix << formatXML(typeName) << "\">\n";
          std::string typeName(iObject.typeOf().name(TypeNameHandling::Scoped));
          if(typeName.empty()){
            typeName="{unknown}";
          }
          ObjectWithDict iObjBegin;
          void* beginRefBuffer;
          MemberWithDict beginMember = iObject.typeOf().memberByName("begin");
          boost::shared_ptr<ObjectWithDict> beginMemHolder = initReturnValue(beginMember, &iObjBegin, &beginRefBuffer);
          ObjectWithDict iObjEnd;
          void* endRefBuffer;
          MemberWithDict endMember = iObject.typeOf().memberByName("end");
          boost::shared_ptr<ObjectWithDict> endMemHolder = initReturnValue(endMember, &iObjEnd, &endRefBuffer);

          beginMember.invoke(iObject, &iObjBegin);
          endMember.invoke(iObject, &iObjEnd);
          if(printContentsOfStdContainer(oStream,
                                         iIndent+iPrefix+formatXML(typeName)+"\">\n",
                                         iIndent+iPostfix,
                                         iObjBegin,
                                         iObjEnd,
                                         iIndent,
                                         iIndentDelta)) {
            if(typeName.empty()){
              typeName="{unknown}";
            }
            return true;
          }
        } catch(std::exception const& x) {
        }
        return false;
      }
      return false;
    }

    void printObject(std::ostream& oStream,
                     Event const& iEvent,
                     std::string const& iClassName,
                     std::string const& iModuleLabel,
                     std::string const& iInstanceLabel,
                     std::string const& iIndent,
                     std::string const& iIndentDelta) {
      try {
        GenericHandle handle(iClassName);
      }catch(edm::Exception const&) {
        std::cout << iIndent << " \"" << iClassName << "\"" << " is an unknown type" << std::endl;
        return;
      }
      GenericHandle handle(iClassName);
      iEvent.getByLabel(iModuleLabel, iInstanceLabel, handle);
      std::string className = formatXML(iClassName);
      printObject(oStream, kObjectOpen, kObjectClose, *handle, iIndent, iIndentDelta);
    }
  }

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  XMLOutputModule::XMLOutputModule(ParameterSet const& iPSet) :
      OutputModule(iPSet),
      stream_(iPSet.getUntrackedParameter<std::string>("fileName").c_str()),
      indentation_("  ") {
    if(!stream_){
      throw edm::Exception(errors::Configuration) << "failed to open file " << iPSet.getUntrackedParameter<std::string>("fileName");
    }
    stream_ << "<cmsdata>" << std::endl;
  }

  // XMLOutputModule::XMLOutputModule(XMLOutputModule const& rhs)
  // {
  //    // do actual copying here;
  // }

  XMLOutputModule::~XMLOutputModule() {
    stream_ << "</cmsdata>" << std::endl;
  }

  //
  // assignment operators
  //
  // XMLOutputModule const& XMLOutputModule::operator=(XMLOutputModule const& rhs)
  // {
  //   //An exception safe implementation is
  //   XMLOutputModule temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  void
  XMLOutputModule::write(EventPrincipal const& iEP) {
    ModuleDescription desc;
    Event event(const_cast<EventPrincipal&>(iEP), desc);
    stream_ << "<event run=\"" << event.id().run() << "\" number=\"" << event.id().event() << "\" >\n";
    std::string startIndent = indentation_;
    for(Selections::const_iterator itBD = keptProducts()[InEvent].begin(), itBDEnd = keptProducts()[InEvent].end();
        itBD != itBDEnd;
        ++itBD) {
      stream_ << "<product type=\"" << (*itBD)->friendlyClassName()
             << "\" module=\"" << (*itBD)->moduleLabel()
      << "\" productInstance=\"" << (*itBD)->productInstanceName() << "\">\n";
      printObject(stream_,
                   event,
                  (*itBD)->className(),
                  (*itBD)->moduleLabel(),
                  (*itBD)->productInstanceName(),
                  startIndent,
                  indentation_);
      stream_ << "</product>\n";
    }
    stream_ << "</event>" << std::endl;
  }

  void
  XMLOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Prints event information into a file in XML format.");
    desc.addUntracked<std::string>("fileName");
    OutputModule::fillDescription(desc);
    descriptions.add("XMLoutput", desc);
  }
}
using edm::XMLOutputModule;
DEFINE_FWK_MODULE(XMLOutputModule);
