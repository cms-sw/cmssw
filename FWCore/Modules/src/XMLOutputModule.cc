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

#include "Reflex/Base.h"

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
    void callDestruct(Reflex::Object* iObj) {
      iObj->Destruct();
    }
    //Handle memory for calls to Reflex Invoke
    // We handle Ref's by using an external void* buffer (which we do not delete) while everything else
    // we ask Reflex to create the proper object (and therefore must ask Reflex to delete it)
    boost::shared_ptr<Reflex::Object> initReturnValue(Reflex::Member const& iMember,
                                                      Reflex::Object* iObj,
                                                      void** iRefBuffer) {
      Reflex::Type returnType = iMember.TypeOf().ReturnType();
      if(returnType.IsReference()) {
        *iObj = Reflex::Object(returnType, iRefBuffer);
        return boost::shared_ptr<Reflex::Object>(iObj, doNotDelete);
      }
      *iObj = returnType.Construct();
      return boost::shared_ptr<Reflex::Object>(iObj, callDestruct);
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
    void doPrint(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, Reflex::Object const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << typeidToName(typeid(T)) << kNameValueSep
        << *reinterpret_cast<T*>(iObject.Address()) << iPostfix << "\n";
    }

    template<>
    void doPrint<char>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, Reflex::Object const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "char" << kNameValueSep
        << static_cast<int>(*reinterpret_cast<char*>(iObject.Address())) << iPostfix << "\n";
    }

    template<>
    void doPrint<unsigned char>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, Reflex::Object const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "unsigned char" << kNameValueSep << static_cast<unsigned int>(*reinterpret_cast<unsigned char*>(iObject.Address())) << iPostfix << "\n";
    }

    template<>
    void doPrint<bool>(std::ostream& oStream, std::string const& iPrefix, std::string const& iPostfix, Reflex::Object const& iObject, std::string const& iIndent) {
      oStream << iIndent << iPrefix << "bool" << kNameValueSep
        << ((*reinterpret_cast<bool*>(iObject.Address()))?"true":"false") << iPostfix << "\n";
    }


    typedef void(*FunctionType)(std::ostream&, std::string const&,
                                std::string const&, Reflex::Object const&, std::string const&);
    typedef std::map<std::string, FunctionType> TypeToPrintMap;

    template<typename T>
    void addToMap(TypeToPrintMap& iMap){
      iMap[typeid(T).name()]=doPrint<T>;
    }

    bool printAsBuiltin(std::ostream& oStream,
                               std::string const& iPrefix,
                               std::string const& iPostfix,
                               Reflex::Object const iObject,
                               std::string const& iIndent){
      typedef void(*FunctionType)(std::ostream&, std::string const&, std::string const&, Reflex::Object const&, std::string const&);
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
      TypeToPrintMap::iterator itFound =s_map.find(iObject.TypeOf().TypeInfo().name());
      if(itFound == s_map.end()){

        return false;
      }
      itFound->second(oStream, iPrefix, iPostfix, iObject, iIndent);
      return true;
    }

    bool printAsContainer(std::ostream& oStream,
                          std::string const& iPrefix,
                          std::string const& iPostfix,
                          Reflex::Object const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta);

    void printDataMembers(std::ostream& oStream,
                          Reflex::Object const& iObject,
                          Reflex::Type const& iType,
                          std::string const& iIndent,
                          std::string const& iIndentDelta);

    void printObject(std::ostream& oStream,
                     std::string const& iPrefix,
                     std::string const& iPostfix,
                     Reflex::Object const& iObject,
                     std::string const& iIndent,
                     std::string const& iIndentDelta) {
      Reflex::Object objectToPrint = iObject;
      std::string indent(iIndent);
      if(iObject.TypeOf().IsPointer()) {
        oStream << iIndent << iPrefix << formatXML(iObject.TypeOf().Name(Reflex::SCOPED)) << "\">\n";
        indent +=iIndentDelta;
        int size = (0!=iObject.Address()) ? (0!=*reinterpret_cast<void**>(iObject.Address())?1:0) : 0;
        oStream << indent << kContainerOpen << size << "\">\n";
        if(size) {
          std::string indent2 = indent + iIndentDelta;
          Reflex::Object obj(iObject.TypeOf().ToType(), *reinterpret_cast<void**>(iObject.Address()));
          obj = obj.CastObject(obj.DynamicType());
          printObject(oStream, kObjectOpen, kObjectClose, obj, indent2, iIndentDelta);
        }
        oStream << indent << kContainerClose << "\n";
        oStream << iIndent << iPostfix << "\n";
        Reflex::Type pointedType = iObject.TypeOf().ToType();
        if(Reflex::Type::ByName("void") == pointedType || pointedType.IsPointer() || iObject.Address()==0) {
          return;
        }
        return;

        //have the code that follows print the contents of the data to which the pointer points
        objectToPrint = Reflex::Object(pointedType, iObject.Address());
        //try to convert it to its actual type (assuming the original type was a base class)
        objectToPrint = Reflex::Object(objectToPrint.CastObject(objectToPrint.DynamicType()));
        indent +=iIndentDelta;
      }
      std::string typeName(objectToPrint.TypeOf().Name(Reflex::SCOPED));
      if(typeName.empty()){
        typeName="{unknown}";
      }

      //see if we are dealing with a typedef
      Reflex::Type objectType = objectToPrint.TypeOf();
      bool wasTypedef = false;
      while(objectType.IsTypedef()) {
         objectType = objectType.ToType();
         wasTypedef = true;
      }
      if(wasTypedef){
         Reflex::Object tmp(objectType, objectToPrint.Address());
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
                          Reflex::Object const& iObject,
                          Reflex::Type const& iType,
                          std::string const& iIndent,
                          std::string const& iIndentDelta) {
      //print all the base class data members
      for(Reflex::Base_Iterator itBase = iType.Base_Begin();
          itBase != iType.Base_End();
          ++itBase) {
        printDataMembers(oStream, iObject.CastObject(itBase->ToType()), itBase->ToType(), iIndent, iIndentDelta);
      }
      static std::string const kPrefix("<datamember name=\"");
      static std::string const ktype("\" type=\"");
      static std::string const kPostfix("</datamember>");

      for(Reflex::Member_Iterator itMember = iType.DataMember_Begin();
          itMember != iType.DataMember_End();
          ++itMember){
        //std::cout << "     debug " << itMember->Name() << " " << itMember->TypeOf().Name() << "\n";
        if (itMember->IsTransient()) {
          continue;
        }
        try {
          std::string prefix = kPrefix + itMember->Name() + ktype;
          printObject(oStream,
                      prefix,
                      kPostfix,
                      itMember->Get(iObject),
                      iIndent,
                      iIndentDelta);
        }catch(std::exception& iEx) {
          std::cout << iIndent << itMember->Name() << " <exception caught("
          << iEx.what() << ")>\n";
        }
      }
    }

    bool printContentsOfStdContainer(std::ostream& oStream,
                                     std::string const& iPrefix,
                                     std::string const& iPostfix,
                                     Reflex::Object iBegin,
                                     Reflex::Object const& iEnd,
                                     std::string const& iIndent,
                                     std::string const& iIndentDelta){
      size_t size = 0;
      std::ostringstream sStream;
      if(iBegin.TypeOf() != iEnd.TypeOf()) {
        std::cerr << " begin (" << iBegin.TypeOf().Name(Reflex::SCOPED) << ") and end ("
          << iEnd.TypeOf().Name(Reflex::SCOPED) << ") are not the same type" << std::endl;
        throw std::exception();
      }
      try {
        Reflex::Member compare(iBegin.TypeOf().MemberByName("operator!="));
        if(!compare) {
          //std::cerr << "no 'operator!=' for " << iBegin.TypeOf().Name() << std::endl;
          return false;
        }
        Reflex::Member incr(iBegin.TypeOf().MemberByName("operator++"));
        if(!incr) {
          //std::cerr << "no 'operator++' for " << iBegin.TypeOf().Name() << std::endl;
          return false;
        }
        Reflex::Member deref(iBegin.TypeOf().MemberByName("operator*"));
        if(!deref) {
          //std::cerr << "no 'operator*' for " << iBegin.TypeOf().Name() << std::endl;
          return false;
        }

        std::string indexIndent = iIndent+iIndentDelta;
        int dummy=0;
        //std::cerr << "going to loop using iterator " << iBegin.TypeOf().Name(Reflex::SCOPED) << std::endl;

        std::vector<void*> compareArgs = Reflex::Tools::MakeVector((iEnd.Address()));
        std::vector<void*> incrArgs = Reflex::Tools::MakeVector(static_cast<void*>(&dummy));
        bool compareResult;
        Reflex::Object objCompareResult(Reflex::Type::ByTypeInfo(typeid(bool)), &compareResult);
        Reflex::Object objIncr;
        void* objIncrRefBuffer;
        boost::shared_ptr<Reflex::Object> incrMemHolder = initReturnValue(incr, &objIncr, &objIncrRefBuffer);
        for(;
    	 compare.Invoke(iBegin, &objCompareResult, compareArgs), compareResult;
    	 incr.Invoke(iBegin, &objIncr, incrArgs), ++size) {
          //std::cerr << "going to print" << std::endl;
          Reflex::Object iTemp;
          void* derefRefBuffer;
          boost::shared_ptr<Reflex::Object> derefMemHolder = initReturnValue(deref, &iTemp, &derefRefBuffer);
          deref.Invoke(iBegin, &iTemp);
          if(iTemp.TypeOf().IsReference()) {
            iTemp = Reflex::Object(iTemp.TypeOf(), derefRefBuffer);
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
                          Reflex::Object const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta) {
      Reflex::Object sizeObj;
      try {
        size_t temp; //used to hold the memory for the return value
        sizeObj = Reflex::Object(Reflex::Type::ByTypeInfo(typeid(size_t)), &temp);
        iObject.Invoke("size", &sizeObj);

        if(sizeObj.TypeOf().TypeInfo() != typeid(size_t)) {
          throw std::exception();
        }
        size_t size = *reinterpret_cast<size_t*>(sizeObj.Address());
        Reflex::Member atMember;
        atMember = iObject.TypeOf().MemberByName("at");
        if(!atMember) {
          throw std::exception();
        }
        std::string typeName(iObject.TypeOf().Name(Reflex::SCOPED));
        if(typeName.empty()){
          typeName="{unknown}";
        }

        oStream << iIndent << iPrefix << formatXML(typeName) << "\">\n"
          << iIndent << kContainerOpen << size << "\">\n";
        Reflex::Object contained;
        std::string indexIndent=iIndent+iIndentDelta;
        for(size_t index = 0; index != size; ++index) {
          void* atRefBuffer;
          boost::shared_ptr<Reflex::Object> atMemHolder = initReturnValue(atMember, &contained, &atRefBuffer);

          atMember.Invoke(iObject, &contained, Reflex::Tools::MakeVector(static_cast<void*>(&index)));
          if(contained.TypeOf().IsReference()) {
            contained = Reflex::Object(contained.TypeOf(), atRefBuffer);
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
          std::string typeName(iObject.TypeOf().Name(Reflex::SCOPED));
          if(typeName.empty()){
            typeName="{unknown}";
          }
          Reflex::Object iObjBegin;
          void* beginRefBuffer;
          Reflex::Member beginMember = iObject.TypeOf().MemberByName("begin");
          boost::shared_ptr<Reflex::Object> beginMemHolder = initReturnValue(beginMember, &iObjBegin, &beginRefBuffer);
          Reflex::Object iObjEnd;
          void* endRefBuffer;
          Reflex::Member endMember = iObject.TypeOf().MemberByName("end");
          boost::shared_ptr<Reflex::Object> endMemHolder = initReturnValue(endMember, &iObjEnd, &endRefBuffer);

          beginMember.Invoke(iObject, &iObjBegin);
          endMember.Invoke(iObject, &iObjEnd);
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
    desc.addUntracked<std::string>("fileName");
    OutputModule::fillDescription(desc);
    descriptions.add("XMLOutputModule", desc);
  }
}
using edm::XMLOutputModule;
DEFINE_FWK_MODULE(XMLOutputModule);
