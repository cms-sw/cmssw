// -*- C++ -*-
//
// Package:    Modules
// Class:      TestInitRootHandlers
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


// system include files
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
// #include "FWCore/Utilities/interface/Algorithms.h"

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Services/test/TestInitRootHandlers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/Utilities/interface/EDMException.h"


#include "FWCore/Framework/interface/GenericHandle.h"
//
// class declarations
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

///consistently format class names
static
std::string formatClassName(const std::string& iName) {
   return std::string("(")+iName+")";
}

static const char* kNameValueSep = "=";
///convert the object information to the correct type and print it
template<typename T>
static void doPrint(const std::string&iName,const Reflex::Object& iObject, const std::string& iIndent) {
   std::cout << iIndent<< iName <<kNameValueSep<<*reinterpret_cast<T*>(iObject.Address())<<"\n";
}

template<>
static void doPrint<char>(const std::string&iName,const Reflex::Object& iObject, const std::string& iIndent) {
   std::cout << iIndent<< iName <<kNameValueSep<<static_cast<int>(*reinterpret_cast<char*>(iObject.Address()))<<"\n";
}
template<>
static void doPrint<unsigned char>(const std::string&iName,const Reflex::Object& iObject, const std::string& iIndent) {
   std::cout << iIndent<< iName <<kNameValueSep<<static_cast<unsigned int>(*reinterpret_cast<unsigned char*>(iObject.Address()))<<"\n";
}

template<>
static void doPrint<bool>(const std::string&iName,const Reflex::Object& iObject, const std::string& iIndent) {
   std::cout << iIndent<< iName <<kNameValueSep<<((*reinterpret_cast<bool*>(iObject.Address()))?"true":"false")<<"\n";
}


typedef void(*FunctionType)(const std::string&,const Reflex::Object&, const std::string&);
typedef std::map<std::string, FunctionType> TypeToPrintMap;

template<typename T>
static void addToMap(TypeToPrintMap& iMap){
   iMap[typeid(T).name()]=doPrint<T>;
}

static bool printAsBuiltin(const std::string& iName,
                           const Reflex::Object iObject,
                           const std::string& iIndent){
   typedef void(*FunctionType)(const std::string&,const Reflex::Object&, const std::string&);
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
   itFound->second(iName,iObject,iIndent);
   return true;
}
static bool printAsContainer(const std::string& iName,
                             const Reflex::Object& iObject,
                             const std::string& iIndent,
                             const std::string& iIndentDelta);

static void printObject(const std::string& iName,
                        const Reflex::Object& iObject,
                        const std::string& iIndent,
                        const std::string& iIndentDelta) {
   std::string printName = iName;
   Reflex::Object objectToPrint = iObject;
   std::string indent(iIndent);
   if(iObject.TypeOf().IsPointer()) {
     std::cout<<iIndent<<iName<<kNameValueSep<<formatClassName(iObject.TypeOf().Name())<<std::hex<<iObject.Address()<<std::dec<<"\n";
      Reflex::Type pointedType = iObject.TypeOf().ToType();
      if(Reflex::Type::ByName("void") == pointedType ||
         pointedType.IsPointer() ||
         iObject.Address()==0) {
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
   if(typeName.empty()){
      typeName="<unknown>";
   }

   //see if we are dealing with a typedef
   if(objectToPrint.TypeOf().IsTypedef()){
     objectToPrint = Reflex::Object(objectToPrint.TypeOf().ToType(),objectToPrint.Address());
   } 
   if(printAsBuiltin(printName,objectToPrint,indent)) {
      return;
   }
   if(printAsContainer(printName,objectToPrint,indent,iIndentDelta)){
      return;
   }
   
   std::cout<<indent<<printName<<" "<<formatClassName(typeName)<<"\n";
   indent+=iIndentDelta;
   //print all the data members
   for(Reflex::Member_Iterator itMember = objectToPrint.TypeOf().DataMember_Begin();
       itMember != objectToPrint.TypeOf().DataMember_End();
       ++itMember){
      //std::cout <<"     debug "<<itMember->Name()<<" "<<itMember->TypeOf().Name()<<"\n";
      try {
         printObject( itMember->Name(),
                      itMember->Get( objectToPrint),
                      indent,
                      iIndentDelta);
      }catch(std::exception& iEx) {
	std::cout <<indent<<itMember->Name()<<" <exception caught("
		  <<iEx.what()<<")>\n";
      }
      catch(...) {
	std::cout <<indent<<itMember->Name()<<"<unknown exception caught>"<<"\n";
      }
   }
}

static bool printAsContainer(const std::string& iName,
                             const Reflex::Object& iObject,
                             const std::string& iIndent,
                             const std::string& iIndentDelta){
   Reflex::Object sizeObj;
   try {
      sizeObj = iObject.Invoke("size");
      assert(sizeObj.TypeOf().TypeInfo() == typeid(size_t));
      size_t size = *reinterpret_cast<size_t*>(sizeObj.Address());
      Reflex::Member atMember;
      try {
         atMember = iObject.TypeOf().MemberByName("at");
      } catch(const std::exception& x) {
         //std::cerr <<"could not get 'at' member because "<< x.what()<<std::endl;
         return false;
      }
      std::cout <<iIndent<<iName<<kNameValueSep<<"[size="<<size<<"]\n";
      Reflex::Object contained;
      std::string indexIndent=iIndent+iIndentDelta;
      for(size_t index = 0; index != size; ++index) {
         std::ostringstream sizeS;
         sizeS << "["<<index<<"]";
         contained = atMember.Invoke(iObject, Reflex::Tools::MakeVector(static_cast<void*>(&index)));
         //std::cout <<"invoked 'at'"<<std::endl;
         try {
            printObject(sizeS.str(),contained,indexIndent,iIndentDelta);
         }catch(...) {
            std::cout <<iIndent<<"<exception caught>"<<"\n";
         }
      }
      return true;
   } catch(const std::exception& x){
      //std::cerr <<"failed to invoke 'at' because "<<x.what()<<std::endl;
      return false;
   }
   return false;
}


static void printObject(const edm::Event& iEvent,
                        const std::string& iClassName,
                        const std::string& iModuleLabel,
                        const std::string& iInstanceLabel,
                        const std::string& iIndent,
                        const std::string& iIndentDelta) {
   using namespace edm;
   try {
      GenericHandle handle(iClassName);
   }catch(const edm::Exception&) {
      std::cout <<iIndent<<" \""<<iClassName<<"\""<<" is an unknown type"<<std::endl;
      return;
   }
   GenericHandle handle(iClassName);
   iEvent.getByLabel(iModuleLabel,iInstanceLabel,handle);
   std::string className = formatClassName(iClassName);
   printObject(className,*handle,iIndent,iIndentDelta);   
}

void RootErrorHandler(int level, bool die, const char* location, const char* message)
{
// Translate ROOT severity level to MessageLogger severity level

  edm::ELseverityLevel el_severity = edm::ELseverityLevel::ELsev_info;

  if (level >= 5000) {
    el_severity = edm::ELseverityLevel::ELsev_fatal;
    die = true;
  }
  else if (level >= 4000) {
    el_severity = edm::ELseverityLevel::ELsev_severe;
    die = true;
  }
  else if (level >= 2000) {
    el_severity = edm::ELseverityLevel::ELsev_error;
    die = true;
  }
  else if (level >= 1000) {
    el_severity = edm::ELseverityLevel::ELsev_error;
    die = true;
  }

// Adapt C-strings to std::strings
// Arrange to report the error location as furnished by Root

  std::string el_location = "@SUB=?";
  if (location != 0) el_location = std::string("@SUB=")+std::string(location);

  std::string el_message  = "?";
  if (message != 0)  el_message  = message;

// Try to create a meaningful id string using knowledge of ROOT error messages
//
// id ==     "ROOT-ClassName" where ClassName is the affected class
//      else "ROOT/ClassName" where ClassName is the error-declaring class
//      else "ROOT"

  std::string el_identifier = "ROOT";

  std::string precursor("class ");
  size_t index1 = el_message.find(precursor);
  if (index1 != std::string::npos) {
    size_t index2 = index1 + precursor.length();
    size_t index3 = el_message.find_first_of(" :", index2);
    if (index3 != std::string::npos) {
      size_t substrlen = index3-index2;
      el_identifier += "-";
      el_identifier += el_message.substr(index2,substrlen);
    }
  } else {
    index1 = el_location.find("::");
    if (index1 != std::string::npos) {
      el_identifier += "/";
      el_identifier += el_location.substr(0, index1);
    }
  }

// Intercept some messages and downgrade the severity

    if (el_message.find("dictionary") != std::string::npos) {
      el_severity = edm::ELseverityLevel::ELsev_info;
      die = false;
    }

    if (el_message.find("already in TClassTable") != std::string::npos) {
      el_severity = edm::ELseverityLevel::ELsev_info;
      die = false;
    }

    if (el_message.find("matrix not positive definite") != std::string::npos) {
      el_severity = edm::ELseverityLevel::ELsev_info;
      die = false;
    }

// Intercept some messages and upgrade the severity

    if ((el_location.find("TBranchElement::Fill") != std::string::npos)
     && (el_message.find("fill branch") != std::string::npos)
     && (el_message.find("address") != std::string::npos)
     && (el_message.find("not set") != std::string::npos)) {
      el_severity = edm::ELseverityLevel::ELsev_fatal;
      die = true;
    }

    if ((el_message.find("Tree branches") != std::string::npos)
     && (el_message.find("different numbers of entries") != std::string::npos)) {
      el_severity = edm::ELseverityLevel::ELsev_fatal;
      die = true;
    }

// Feed the message to the MessageLogger... let it choose to suppress or not.

  if (el_severity == edm::ELseverityLevel::ELsev_fatal && !die) {
    edm::LogError("Root_Fatal") << el_location << el_message;
  }
  else if (el_severity == edm::ELseverityLevel::ELsev_severe && !die) {
    edm::LogError("Root_Severe") << el_location << el_message;
  }
  else if (el_severity == edm::ELseverityLevel::ELsev_error && !die) {
    edm::LogError("Root_Error") << el_location << el_message;
  }
  else if (el_severity == edm::ELseverityLevel::ELsev_warning && !die) {
    edm::LogWarning("Root_Warning") << el_location << el_message ;
  }
  else if (el_severity == edm::ELseverityLevel::ELsev_info && !die) {
    edm::LogInfo("Root_Information") << el_location << el_message ;
  }

// Root has declared a fatal error.  Throw an EDMException.

   if (die) {
// Throw an edm::Exception instead of just aborting
     std::ostringstream sstr;
     sstr << "Fatal Root Error: " << el_location << "\n" << el_message << '\n';
     edm::Exception except(edm::errors::FatalRootError, sstr.str());
     throw except;
   }
}
//
// constructors and destructor
//
TestInitRootHandlers::TestInitRootHandlers(const edm::ParameterSet& iConfig) :
  indentation_(iConfig.getUntrackedParameter("indentation",std::string("++"))),
  verboseIndentation_(iConfig.getUntrackedParameter("verboseIndention",std::string("  "))),
  moduleLabels_(iConfig.getUntrackedParameter("verboseForModuleLabels",std::vector<std::string>())),
  verbose_(iConfig.getUntrackedParameter("verbose",false) || moduleLabels_.size()>0),
  evno_(0)
{
   //now do what ever initialization is needed
  //edm::sort_all(moduleLabels_);
  std::sort(moduleLabels_.begin(), moduleLabels_.end());
}

TestInitRootHandlers::~TestInitRootHandlers()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestInitRootHandlers::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   typedef std::vector< Provenance const*> Provenances;
   Provenances provenances;
   std::string friendlyName;
   std::string modLabel;
   std::string instanceName;
   std::string key;

   iEvent.getAllProvenance(provenances);
   
   std::cout << "\n" << indentation_ << "Event " << std::setw(5) << evno_ << " contains "
             << provenances.size() << " product" << (provenances.size()==1 ?"":"s")
             << " with friendlyClassName, moduleLabel and productInstanceName:"
             << std::endl;

   for(Provenances::iterator itProv = provenances.begin(), itProvEnd = provenances.end();
                             itProv != itProvEnd;
                           ++itProv) {
      friendlyName = (*itProv)->friendlyClassName();
      //if(friendlyName.empty())  friendlyName = std::string("||");

      modLabel = (*itProv)->moduleLabel();
      //if(modLabel.empty())  modLabel = std::string("||");

      instanceName = (*itProv)->productInstanceName();
      //if(instanceName.empty())  instanceName = std::string("||");
      
      std::cout << indentation_ << friendlyName
                << " \"" << modLabel
                << "\" \"" << instanceName <<"\"" << std::endl;
      if(verbose_){
         if( moduleLabels_.empty() ||
	     //edm::binary_search_all(moduleLabels_, modLabel)) {
	     std::binary_search(moduleLabels_.begin(),moduleLabels_.end(), modLabel)) {
            //indent one level before starting to print
            std::string startIndent = indentation_+verboseIndentation_;
            printObject(iEvent,
                        (*itProv)->className(),
                        (*itProv)->moduleLabel(),
                        (*itProv)->productInstanceName(),
                        startIndent,
                        verboseIndentation_);
         }
      }
      
      key = friendlyName
          + std::string(" + \"") + modLabel
          + std::string("\" + \"") + instanceName+"\"";
      ++cumulates_[key];
   }
   ++evno_;

   if( evno_ == 1 ) RootErrorHandler( 1000, false, "ContentTest", "Simulated Root Warning" );
   if( evno_ == 2 ) RootErrorHandler( 2000, false, "ContentTest", "Simulated Root Error" );
   if( evno_ == 3 ) RootErrorHandler( 4000, false, "ContentTest", "Simulated Root SysError" );
   if( evno_ == 4 ) RootErrorHandler( 5000, true,  "ContentTest", "Simulated Fatal Root error" );
}

// ------------ method called at end of job -------------------
void
TestInitRootHandlers::endJob() 
{
   typedef std::map<std::string,int> nameMap;

   std::cout <<"\nSummary for key being the concatenation of friendlyClassName, moduleLabel and productInstanceName" << std::endl;
   for(nameMap::const_iterator it = cumulates_.begin(), itEnd = cumulates_.end();
                               it != itEnd;
                             ++it) {
      std::cout << std::setw(6) << it->second << " occurrences of key " << it->first << std::endl;
   }

// Test boost::lexical_cast  We don't need this right now so comment it out.
// int k = 137;
// std::string ktext = boost::lexical_cast<std::string>(k);
// std::cout << "\nInteger " << k << " expressed as a string is |" << ktext << "|" << std::endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(TestInitRootHandlers);
