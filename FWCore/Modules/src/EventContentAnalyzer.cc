// -*- C++ -*-
//
// Package:    Modules
// Class:      EventContentAnalyzer
//
/**
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep 19 11:47:28 CEST 2005
//
//


// system include files
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <string>
#include <map>
#include <vector>

// user include files
#include "Reflex/Member.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edm {
  class ConfigurationDescriptions;
  namespace {
    std::string formatClassName(std::string const& iName) {
       return std::string("(")+iName+")";
    }

    char const* kNameValueSep = "=";
    ///convert the object information to the correct type and print it
    template<typename T>
    void doPrint(std::string const& iName, Reflex::Object const& iObject, std::string const& iIndent) {
      LogAbsolute("EventContent") << iIndent<< iName <<kNameValueSep<<*reinterpret_cast<T*>(iObject.Address());//<<"\n";
    }

    template<>
    void doPrint<char>(std::string const& iName, Reflex::Object const& iObject, std::string const& iIndent) {
      LogAbsolute("EventContent") << iIndent<< iName <<kNameValueSep<<static_cast<int>(*reinterpret_cast<char*>(iObject.Address()));//<<"\n";
    }

    template<>
    void doPrint<unsigned char>(std::string const& iName, Reflex::Object const& iObject, std::string const& iIndent) {
      LogAbsolute("EventContent") << iIndent<< iName <<kNameValueSep<<static_cast<unsigned int>(*reinterpret_cast<unsigned char*>(iObject.Address()));//<<"\n";
    }

    template<>
    void doPrint<bool>(std::string const& iName, Reflex::Object const& iObject, std::string const& iIndent) {
      LogAbsolute("EventContent") << iIndent<< iName <<kNameValueSep<<((*reinterpret_cast<bool*>(iObject.Address()))?"true":"false");//<<"\n";
    }

    typedef void(*FunctionType)(std::string const&, Reflex::Object const&, std::string const&);
    typedef std::map<std::string, FunctionType> TypeToPrintMap;

    template<typename T>
    void addToMap(TypeToPrintMap& iMap) {
       iMap[typeid(T).name()]=doPrint<T>;
    }

    bool printAsBuiltin(std::string const& iName,
                               Reflex::Object const& iObject,
                               std::string const& iIndent) {
       typedef void(*FunctionType)(std::string const&, Reflex::Object const&, std::string const&);
       typedef std::map<std::string, FunctionType> TypeToPrintMap;
       static TypeToPrintMap s_map;
       static bool isFirst = true;
       if(isFirst) {
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
       if(itFound == s_map.end()) {

          return false;
       }
       itFound->second(iName, iObject, iIndent);
       return true;
    }

    bool printAsContainer(std::string const& iName,
                          Reflex::Object const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta);

    void printObject(std::string const& iName,
                     Reflex::Object const& iObject,
                     std::string const& iIndent,
                     std::string const& iIndentDelta) {
       std::string printName = iName;
       Reflex::Object objectToPrint = iObject;
       std::string indent(iIndent);
       if(iObject.TypeOf().IsPointer()) {
         LogAbsolute("EventContent")<<iIndent<<iName<<kNameValueSep<<formatClassName(iObject.TypeOf().Name())<<std::hex<<iObject.Address()<<std::dec;//<<"\n";
          Reflex::Type pointedType = iObject.TypeOf().ToType();
          if(Reflex::Type::ByName("void") == pointedType ||
             pointedType.IsPointer() ||
             iObject.Address() == 0) {
             return;
          }
          return;

          //have the code that follows print the contents of the data to which the pointer points
          objectToPrint = Reflex::Object(pointedType, iObject.Address());
          //try to convert it to its actual type (assuming the original type was a base class)
          objectToPrint = Reflex::Object(objectToPrint.CastObject(objectToPrint.DynamicType()));
          printName = std::string("*")+iName;
          indent +=iIndentDelta;
       }
       std::string typeName(objectToPrint.TypeOf().Name());
       if(typeName.empty()) {
          typeName="<unknown>";
       }

       //see if we are dealing with a typedef
       if(objectToPrint.TypeOf().IsTypedef()) {
         objectToPrint = Reflex::Object(objectToPrint.TypeOf().ToType(), objectToPrint.Address());
       }
       if(printAsBuiltin(printName, objectToPrint, indent)) {
          return;
       }
       if(printAsContainer(printName, objectToPrint, indent, iIndentDelta)) {
          return;
       }

       LogAbsolute("EventContent")<<indent<<printName<<" "<<formatClassName(typeName);//<<"\n";
       indent+=iIndentDelta;
       //print all the data members
       for(Reflex::Member_Iterator itMember = objectToPrint.TypeOf().DataMember_Begin();
           itMember != objectToPrint.TypeOf().DataMember_End();
           ++itMember) {
          //LogAbsolute("EventContent") <<"     debug "<<itMember->Name()<<" "<<itMember->TypeOf().Name()<<"\n";
          try {
             printObject(itMember->Name(),
                          itMember->Get(objectToPrint),
                          indent,
                          iIndentDelta);
          }catch(std::exception& iEx) {
    	LogAbsolute("EventContent") <<indent<<itMember->Name()<<" <exception caught("
    		  <<iEx.what()<<")>\n";
          }
       }
    }

    bool printAsContainer(std::string const& iName,
                          Reflex::Object const& iObject,
                          std::string const& iIndent,
                          std::string const& iIndentDelta) {
       Reflex::Object sizeObj;
       try {
          size_t temp; //used to hold the memory for the return value
          sizeObj = Reflex::Object(Reflex::Type::ByTypeInfo(typeid(size_t)), &temp);
          iObject.Invoke("size", &sizeObj);
          assert(iObject.TypeOf().FunctionMemberByName("size").TypeOf().ReturnType().TypeInfo()==typeid(size_t));
          //std::cout <<"size of type '"<<sizeObj.TypeOf().Name()<<"' "<<sizeObj.TypeOf().TypeInfo().name()<<std::endl;
          assert(sizeObj.TypeOf().FinalType().TypeInfo() == typeid(size_t));
          size_t size = *reinterpret_cast<size_t*>(sizeObj.Address());
          Reflex::Member atMember;
          try {
             atMember = iObject.TypeOf().MemberByName("at");
          } catch(std::exception const& x) {
             //std::cerr <<"could not get 'at' member because "<< x.what()<<std::endl;
             return false;
          }
          LogAbsolute("EventContent") <<iIndent<<iName<<kNameValueSep<<"[size="<<size<<"]";//"\n";
          Reflex::Object contained;
          std::string indexIndent=iIndent+iIndentDelta;
          Reflex::Type atReturnType = atMember.TypeOf().ReturnType();
          //std::cout <<"return type "<<atReturnType.Name()<<" size of "<<atReturnType.SizeOf()
          //<<" pointer? "<<atReturnType.IsPointer()<<" ref? "<<atReturnType.IsReference()<<std::endl;

          //Return by reference must be treated differently since Reflex will not properly create
          // memory for a ref (which should just be a pointer to the object and not the object itself)
          //So we will create memory on the stack which can be used to hold a reference
          bool const isRef = atReturnType.IsReference();
          void* refMemoryBuffer = 0;
          size_t index = 0;
          //The argument to the 'at' function is the index. Since the argument list holds pointers to the arguments
          // we only need to create it once and then when the value of index changes the pointer already
          // gets the new value
          std::vector<void*> args;
          args.push_back(&index);
          for( ; index != size; ++index) {
             std::ostringstream sizeS;
             sizeS << "["<<index<<"]";
             if(isRef) {
                Reflex::Object refObject(atReturnType, &refMemoryBuffer);
                atMember.Invoke(iObject, &refObject, args);
                //Although to hold the return value from a reference Reflex requires you to pass it a
                // void** when it tries to call methods on the reference it expects to be given a void*
                contained = Reflex::Object(atReturnType, refMemoryBuffer);
             } else {
                contained = atReturnType.Construct();
                atMember.Invoke(iObject, &contained, args);
             }
             //LogAbsolute("EventContent") <<"invoked 'at'"<<std::endl;
             try {
                printObject(sizeS.str(), contained, indexIndent, iIndentDelta);
             } catch(std::exception& iEx) {
    	        LogAbsolute("EventContent") <<indexIndent<<iName<<" <exception caught("
    		  <<iEx.what()<<")>\n";
             }
             if(!isRef) {
                contained.Destruct();
             }
          }
          return true;
       } catch(std::exception const& x) {
          //std::cerr <<"failed to invoke 'at' because "<<x.what()<<std::endl;
          return false;
       }
       return false;
    }

    void printObject(Event const& iEvent,
                     std::string const& iClassName,
                     std::string const& iModuleLabel,
                     std::string const& iInstanceLabel,
                     std::string const& iProcessName,
                     std::string const& iIndent,
                     std::string const& iIndentDelta) {
       try {
          GenericHandle handle(iClassName);
       }catch(edm::Exception const&) {
          LogAbsolute("EventContent") <<iIndent<<" \""<<iClassName<<"\""<<" is an unknown type"<<std::endl;
          return;
       }
       GenericHandle handle(iClassName);
       iEvent.getByLabel(InputTag(iModuleLabel, iInstanceLabel, iProcessName), handle);
       std::string className = formatClassName(iClassName);
       printObject(className, *handle, iIndent, iIndentDelta);
    }
  }

  class EventContentAnalyzer : public EDAnalyzer {
  public:
     explicit EventContentAnalyzer(ParameterSet const&);
     ~EventContentAnalyzer();

     virtual void analyze(Event const&, EventSetup const&);
     virtual void endJob();

     static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:

     // ----------member data ---------------------------
     std::string indentation_;
     std::string verboseIndentation_;
     std::vector<std::string> moduleLabels_;
     bool        verbose_;
     std::vector<std::string> getModuleLabels_;
     bool        getData_;
     int         evno_;
     std::map<std::string, int> cumulates_;
     bool        listContent_;
  };

  //
  // constructors and destructor
  //
  EventContentAnalyzer::EventContentAnalyzer(ParameterSet const& iConfig) :
    indentation_(iConfig.getUntrackedParameter("indentation", std::string("++"))),
    verboseIndentation_(iConfig.getUntrackedParameter("verboseIndentation", std::string("  "))),
    moduleLabels_(iConfig.getUntrackedParameter("verboseForModuleLabels", std::vector<std::string>())),
    verbose_(iConfig.getUntrackedParameter("verbose", false) || moduleLabels_.size()>0),
    getModuleLabels_(iConfig.getUntrackedParameter("getDataForModuleLabels", std::vector<std::string>())),
    getData_(iConfig.getUntrackedParameter("getData", false) || getModuleLabels_.size()>0),
    evno_(1),
    listContent_(iConfig.getUntrackedParameter("listContent",true)){
     //now do what ever initialization is needed
     sort_all(moduleLabels_);
  }

  EventContentAnalyzer::~EventContentAnalyzer() {

     // do anything here that needs to be done at destruction time
     // (e.g. close files, deallocate resources etc.)

  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void
  EventContentAnalyzer::analyze(Event const& iEvent, EventSetup const& iSetup) {
     typedef std::vector<Provenance const*> Provenances;
     Provenances provenances;

     iEvent.getAllProvenance(provenances);

     if(listContent_) {
       LogAbsolute("EventContent") << "\n" << indentation_ << "Event " << std::setw(5) << evno_ << " contains "
				   << provenances.size() << " product" << (provenances.size() == 1 ? "" : "s")
				   << " with friendlyClassName, moduleLabel, productInstanceName and processName:"
				   << std::endl;
     }

     std::string startIndent = indentation_+verboseIndentation_;
     for(Provenances::iterator itProv = provenances.begin(), itProvEnd = provenances.end();
                               itProv != itProvEnd;
                             ++itProv) {
         std::string const& className = (*itProv)->className();

         std::string const& friendlyName = (*itProv)->friendlyClassName();
         //if(friendlyName.empty())  friendlyName = std::string("||");

         std::string const& modLabel = (*itProv)->moduleLabel();
         //if(modLabel.empty()) modLabel = std::string("||");

         std::string const& instanceName = (*itProv)->productInstanceName();
         //if(instanceName.empty()) instanceName = std::string("||");

         std::string const& processName = (*itProv)->processName();

	 bool doVerbose = verbose_ && (moduleLabels_.empty() ||
				       binary_search_all(moduleLabels_, modLabel));

	 if(listContent_ || doVerbose ) {
	   LogAbsolute("EventContent") << indentation_ << friendlyName
				       << " \"" << modLabel
				       << "\" \"" << instanceName << "\" \""
				       << processName << "\""
				       << " (productId = " << (*itProv)->productID() << ")"
				       << std::endl;
	 }

         std::string key = friendlyName
  	 + std::string(" + \"") + modLabel
  	 + std::string("\" + \"") + instanceName + "\" \"" + processName + "\"";
         ++cumulates_[key];

         if(doVerbose) {

  	   //indent one level before starting to print
  	   printObject(iEvent,
  		       className,
  		       modLabel,
  		       instanceName,
                       processName,
  		       startIndent,
  		       verboseIndentation_);
             continue;

         }
         if(getData_) {
           if(getModuleLabels_.empty() ||
             binary_search_all(getModuleLabels_, modLabel)) {
             try {
               GenericHandle handle(className);
             } catch(edm::Exception const&) {
               LogAbsolute("EventContent") << startIndent << " \"" << className << "\"" << " is an unknown type" << std::endl;
               return;
             }
             GenericHandle handle(className);
             iEvent.getByLabel(InputTag(modLabel,
                                        instanceName,
                                        processName),
                                        handle);
           }
         }
     }
     //std::cout <<"Mine"<<std::endl;
     ++evno_;
  }

  // ------------ method called at end of job -------------------
  void
  EventContentAnalyzer::endJob() {
     typedef std::map<std::string, int> nameMap;

     LogAbsolute("EventContent") << "\nSummary for key being the concatenation of friendlyClassName, moduleLabel, productInstanceName and processName" << std::endl;
     for(nameMap::const_iterator it = cumulates_.begin(), itEnd = cumulates_.end();
                                 it != itEnd;
                                 ++it) {
        LogAbsolute("EventContent") << std::setw(6) << it->second << " occurrences of key " << it->first << std::endl;
     }

  // Test boost::lexical_cast  We don't need this right now so comment it out.
  // int k = 137;
  // std::string ktext = boost::lexical_cast<std::string>(k);
  // std::cout << "\nInteger " << k << " expressed as a string is |" << ktext << "|" << std::endl;
  }

  void
  EventContentAnalyzer::fillDescriptions(ConfigurationDescriptions & descriptions) {

     descriptions.setComment("This plugin will print a list of all products in the event "
                             "provenance.  It also has options to print and/or get each product.");

     ParameterSetDescription desc;

     ParameterDescriptionNode *np;

     std::string defaultString("++");
     np = desc.addOptionalUntracked<std::string>("indentation", defaultString);
     np->setComment("This string is printed at the beginning of every line printed during event processing.");

     np = desc.addOptionalUntracked<bool>("verbose", false);
     np->setComment("If true, the contents of products are printed using Reflex.");

     defaultString = "  ";
     np = desc.addOptionalUntracked<std::string>("verboseIndentation", defaultString);
     np->setComment("This string is used to further indent lines when printing the contents of products in verbose mode.");

     std::vector<std::string> defaultVString;

     np = desc.addOptionalUntracked<std::vector<std::string> >("verboseForModuleLabels", defaultVString);
     np->setComment("If this vector is not empty, then only products with module labels on this list are printed using Reflex.");

     np = desc.addOptionalUntracked<bool>("getData", false);
     np->setComment("If true the products will be retrieved using getByLabel.");

     np = desc.addOptionalUntracked<std::vector<std::string> >("getDataForModuleLabels", defaultVString);
     np->setComment("If this vector is not empty, then only products with module labels on this list are retrieved by getByLabel.");

     np = desc.addOptionalUntracked<bool>("listContent", true);
     np->setComment("If true then print a list of all the event content.");


     descriptions.add("printContent", desc);
  }
}

using edm::EventContentAnalyzer;
DEFINE_FWK_MODULE(EventContentAnalyzer);
