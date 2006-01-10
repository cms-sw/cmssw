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
// $Id: EventContentAnalyzer.cc,v 1.3 2006/01/10 17:23:08 chrjones Exp $
//
//


// system include files
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Provenance.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Modules/src/EventContentAnalyzer.h"

#include "boost/lexical_cast.hpp"

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
static void doPrint(const std::string&iName,const seal::reflex::Object& iObject, const std::string& iIndent) {
   std::cout << iIndent<< iName <<kNameValueSep<<*reinterpret_cast<T*>(iObject.address())<<"\n";
};

typedef void(*FunctionType)(const std::string&,const seal::reflex::Object&, const std::string&);
typedef std::map<std::string, FunctionType> TypeToPrintMap;

template<typename T>
static void addToMap(TypeToPrintMap& iMap){
   iMap[typeid(T).name()]=doPrint<T>;
}

static bool printAsBuiltin(const std::string& iName,
                           const seal::reflex::Object iObject,
                           const std::string& iIndent){
   typedef void(*FunctionType)(const std::string&,const seal::reflex::Object&, const std::string&);
   typedef std::map<std::string, FunctionType> TypeToPrintMap;
   static TypeToPrintMap s_map;
   static bool isFirst = true;
   if(isFirst){
      addToMap<short>(s_map);
      addToMap<int>(s_map);
      addToMap<long>(s_map);
      addToMap<unsigned short>(s_map);
      addToMap<unsigned int>(s_map);
      addToMap<unsigned long>(s_map);
      addToMap<float>(s_map);
      addToMap<double>(s_map);
      isFirst=false;
   }
   TypeToPrintMap::iterator itFound =s_map.find(iObject.type().typeInfo().name());
   if(itFound == s_map.end()){
      
      if(iObject.type().isPointer()) {
         std::cout<<iIndent<<iName<<kNameValueSep<<formatClassName(iObject.type().name())<<std::hex<<iObject.address()<<"\n";
         return true;
      }
      return false;
   }
   itFound->second(iName,iObject,iIndent);
   return true;
};
static bool printAsContainer(const std::string& iName,
                             const seal::reflex::Object& iObject,
                             const std::string& iIndent,
                             const std::string& iIndentDelta);

static void printObject(const std::string& iName,
                        const seal::reflex::Object& iObject,
                        const std::string& iIndent,
                        const std::string& iIndentDelta) {
   if(printAsBuiltin(iName,iObject,iIndent)) {
      return;
   }
   if(printAsContainer(iName,iObject,iIndent,iIndentDelta)){
      return;
   }
   
   using namespace seal::reflex;
   std::cout<<iIndent<<iName<<" "<<formatClassName(iObject.type().name())<<"\n";

   std::string indent(iIndent+iIndentDelta);
   //print all the data members
   for(seal::reflex::member_iterator itMember = iObject.type().dataMember_begin();
       itMember != iObject.type().dataMember_end();
       ++itMember){
      //std::cout <<"     debug "<<itMember->name()<<" "<<itMember->type().name()<<"\n";
      printObject( itMember->name(),
                   itMember->get( iObject),
                   indent,
                   iIndentDelta);
   }
};

static bool printAsContainer(const std::string& iName,
                             const seal::reflex::Object& iObject,
                             const std::string& iIndent,
                             const std::string& iIndentDelta){
   using namespace seal::reflex;
   Object sizeObj;
   try {
      sizeObj = iObject.invoke("size");
      assert(sizeObj.type().typeInfo() == typeid(size_t));
      size_t size = *reinterpret_cast<size_t*>(sizeObj.address());
      Member atMember;
      try {
         atMember = iObject.type().member("at");
      } catch(const std::exception& x) {
         //std::cerr <<"could not get 'at' member because "<< x.what()<<std::endl;
         return false;
      }
      std::cout <<iIndent<<iName<<kNameValueSep<<"[size="<<size<<"]\n";
      Object contained;
      std::string indexIndent=iIndent+iIndentDelta;
      for(size_t index = 0; index != size; ++index) {
         std::ostringstream sizeS;
         sizeS << "["<<index<<"]";
         contained = atMember.invoke(iObject, Tools::makeVector(static_cast<void*>(&index)));
         //std::cout <<"invoked 'at'"<<std::endl;
         printObject(sizeS.str(),contained,indexIndent,iIndentDelta);
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

//
// constructors and destructor
//
EventContentAnalyzer::EventContentAnalyzer(const edm::ParameterSet& iConfig) :
  indentation_(iConfig.getUntrackedParameter("indentation",std::string("++"))),
  verboseIndentation_(iConfig.getUntrackedParameter("verboseIndention",std::string("  "))),
  moduleLabels_(iConfig.getUntrackedParameter("verboseLabels",std::vector<std::string>())),
  verbose_(iConfig.getUntrackedParameter("verbose",false) || moduleLabels_.size()>0),
  evno_(0)
{
   //now do what ever initialization is needed
   std::sort(moduleLabels_.begin(),moduleLabels_.end());
}

EventContentAnalyzer::~EventContentAnalyzer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
EventContentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

   for(Provenances::iterator itProv  = provenances.begin();
                             itProv != provenances.end();
                           ++itProv) {
      friendlyName = (*itProv)->product.friendlyClassName_;
      //if(friendlyName.empty())  friendlyName = std::string("||");

      modLabel = (*itProv)->product.module.moduleLabel_;
      //if(modLabel.empty())  modLabel = std::string("||");

      instanceName = (*itProv)->product.productInstanceName_;
      //if(instanceName.empty())  instanceName = std::string("||");
      
      std::cout << indentation_ << friendlyName
                << " \"" << modLabel
                << "\" \"" << instanceName <<"\"" << std::endl;
      if(verbose_){
         if( moduleLabels_.size() ==0 ||
             std::binary_search(moduleLabels_.begin(),moduleLabels_.end(),modLabel)) {
            //indent one level before starting to print
            std::string startIndent = indentation_+verboseIndentation_;
            printObject(iEvent,
                        (*itProv)->product.fullClassName_,
                        (*itProv)->product.module.moduleLabel_,
                        (*itProv)->product.productInstanceName_,
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
}

// ------------ method called at end of job -------------------
void
EventContentAnalyzer::endJob() 
{
   typedef std::map<std::string,int> nameMap;

   std::cout <<"\nSummary for key being the concatenation of friendlyClassName, moduleLabel and productInstanceName" << std::endl;
   for(nameMap::const_iterator it =cumulates_.begin();
                               it!=cumulates_.end();
                             ++it) {
      std::cout << std::setw(6) << it->second << " occurrences of key " << it->first << std::endl;
   }

// Test boost::lexical_cast  We don't need this right now so comment it out.
// int k = 137;
// std::string ktext = boost::lexical_cast<std::string>(k);
// std::cout << "\nInteger " << k << " expressed as a string is |" << ktext << "|" << std::endl;
}
