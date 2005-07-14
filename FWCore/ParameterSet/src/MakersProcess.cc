// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     MakersProcess
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 18 19:09:01 EDT 2005
// $Id: MakersProcess.cc,v 1.3 2005/06/27 16:35:01 paterno Exp $
//

// system include files
#include <iostream>

// user include files
#include "FWCore/ParameterSet/src/BuilderVPSet.h"
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;
//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace edm {
   namespace pset {

struct FillProcess : public edm::pset::Visitor
{
   FillProcess(edm::ParameterSet& oToFill, std::vector< boost::shared_ptr<edm::pset::WrapperNode> >& oPathSegments) :
   pset_(oToFill), wrappers_(oPathSegments) {
      static const std::string kModule("module");
      static const std::string kESModule("es_module");
      static const std::string kSource("source");
      static const std::string kESSource("es_source");
      moduleTypes_[kModule];
      moduleTypes_[kESModule];
      moduleTypes_[kSource];
      moduleTypes_[kESSource];
      handleTypes_[kModule] = &FillProcess::handleModule;
      handleTypes_[kSource] = &FillProcess::handleModule;
      handleTypes_[kESModule] = &FillProcess::handleESModule;
      handleTypes_[kESSource] = &FillProcess::handleESSource;

      assert(moduleTypes_.size() == handleTypes_.size());
      namesToTypes_["all_modules"]=kModule;
      namesToTypes_["all_sources"]=kSource;
      namesToTypes_["all_esmodules"]=kESModule;
      namesToTypes_["all_essources"]=kESSource;
   }

   virtual void visitContents(const edm::pset::ContentsNode& iNode) {
      //print(cout, "Contents", iNode);
      //endPrint(cout);
      iNode.acceptForChildren(*this);
      //visitChildren(iNode);
   }
   virtual void visitPSet(const edm::pset::PSetNode& iNode) {
      //print(cout, "PSet", iNode);
      //endPrint(cout);
      
      static const std::string kPSet("PSet");
      static const std::string kBlock("block");
      if(iNode.type() == kPSet) {
	if(iNode.value_.value_->empty()==true)
	{
	  throw edm::Exception(errors::Configuration,"EmptySet")
	    << "ParameterSet: Empty ParameterSets are not allowed.\n"
	    << "name = " << iNode.name();
	}
         pSets_.insert(make_pair(iNode.name(), makePSet(*(iNode.value_.value_),
                                                          usingBlocks_,
                                                          pSets_)));
      } else if (iNode.type() == kBlock) {
	if(iNode.value_.value_->empty()==true)
	{
	  throw edm::Exception(errors::Configuration,"EmptySet")
	    << "ParameterSet: Empty Blocks are not allowed.\n"
	    << "name = " << iNode.name();
	}
         usingBlocks_.insert(make_pair(iNode.name(), makePSet(*(iNode.value_.value_),
                                                                usingBlocks_,
                                                                pSets_)));
      } else {
         assert(false);
      }
      
      //PrintNodes handleChildren;
      //iNode.acceptForChildren(handleChildren);
   }
   virtual void visitVPSet(const edm::pset::VPSetNode& iNode) {
      std::vector<ParameterSet> sets;
      BuilderVPSet builder(sets, usingBlocks_, pSets_);
      iNode.acceptForChildren(builder);
      pset_.insert(false, iNode.name_, Entry(sets, true));
   }
   virtual void visitModule(const edm::pset::ModuleNode& iNode) {
      using namespace edm;
      //print(cout, "Module", iNode) <<" class=\""<<iNode.class_<<"\"";
      //endPrint(cout);
      ModuleTypes::iterator itFound = moduleTypes_.find(iNode.type_);
      assert(itFound != moduleTypes_.end());
      
      boost::shared_ptr<ParameterSet> modulePSet = makePSet(*iNode.nodes_,
                                                     usingBlocks_,
                                                     pSets_);
      std::string name = (this->*(handleTypes_[iNode.type_]))(iNode, *modulePSet);
      pset_.insert(true, name , Entry(*modulePSet,true));
      itFound->second.push_back(name);
      
      //visitChildren(iNode);
   }
   virtual void visitWrapper(const edm::pset::WrapperNode& iNode) {
      //std::cout <<" Found Path Fragment "<<std::endl;
      //print(cout, "Wrapper", iNode);
      //endPrint(cout);
      wrappers_.push_back(boost::shared_ptr<edm::pset::WrapperNode>(new edm::pset::WrapperNode(iNode)));
   }
   
   void fillFrom(const edm::pset::PSetNode& iNode) {
      iNode.acceptForChildren(*this);
      //add the 'all*' items to the ParameterSet
      for(std::map<std::string, std::string>::const_iterator itMods = namesToTypes_.begin();
           itMods != namesToTypes_.end();
           ++itMods) {
         pset_.insert(true, itMods->first, Entry(moduleTypes_[itMods->second], true));
      }
   }
private:
   std::string handleModule(const edm::pset::ModuleNode&iNode , edm::ParameterSet& oPSet) {
      oPSet.insert(true, "module_label", Entry(iNode.name_, true));
      oPSet.insert(true, "module_type", Entry(iNode.class_,true));
      return iNode.name_;
   }
   std::string handleESModule(const edm::pset::ModuleNode&iNode, edm::ParameterSet& oPSet) {
      std::string label("");
      if(iNode.name_ != "nameless") {
         label = iNode.name_;
      }
      oPSet.insert(true, "module_label", Entry(label, true));
      oPSet.insert(true, "module_type", Entry(iNode.class_,true));
      return iNode.class_+"@"+label;
   }
   std::string handleESSource(const edm::pset::ModuleNode&iNode, edm::ParameterSet& oPSet) {
      std::string label("");
      if(iNode.name_ != "main_es_input") {
         label = iNode.name_;
      }
      oPSet.insert(true, "module_label", Entry(label, true));
      oPSet.insert(true, "module_type", Entry(iNode.class_,true));
      return iNode.class_+"@"+label;
   }
   edm::ParameterSet& pset_;
   std::vector< boost::shared_ptr<edm::pset::WrapperNode> >& wrappers_;
   std::map< std::string, boost::shared_ptr<edm::ParameterSet> > usingBlocks_;
   std::map< std::string, boost::shared_ptr<edm::ParameterSet> > pSets_;
   typedef std::map<std::string, std::vector<std::string> > ModuleTypes;
   ModuleTypes moduleTypes_;
   typedef std::string (FillProcess::*pMemFunc)(const edm::pset::ModuleNode&, edm::ParameterSet&);
   std::map<std::string, pMemFunc> handleTypes_;
   std::map<std::string, std::string> namesToTypes_;
};

struct BuildProcess : public edm::pset::Visitor
{
   void fill(const boost::shared_ptr<edm::pset::NodePtrList>& iNodes,
              edm::ProcessDesc& iToFill) {
      assert(iNodes->size() == 1);
      procDesc_ = & iToFill;
      (*iNodes->begin())->accept(*this);
   }
   
   
   //explicit PrintNodes(PSetPtr fillme);
   //virtual ~BuilderPSet();
   virtual void visitPSet(const edm::pset::PSetNode& iNode) {
      //print(cout, "PSet", iNode);
      //endPrint(cout);
      if("process" != iNode.type()) {
	throw edm::Exception(errors::Configuration,"InvalidType")
	  << "ParameterSet: "
	  << "The configuration does not start with a 'process' block.\n"
	  << "found type " << iNode.type()
	  << " with name " << iNode.name();
      }
      procDesc_->pset_.insert(true, "process_name", edm::Entry(iNode.name(), true));
      
      FillProcess handleChildren(procDesc_->pset_, procDesc_->pathFragments_);
      
      handleChildren.fillFrom(iNode);
   }
   
private:
   edm::ProcessDesc* procDesc_;
   
};

boost::shared_ptr<edm::ProcessDesc> makeProcess(NodePtrListPtr nodes) {
   BuildProcess builder;
   boost::shared_ptr<edm::ProcessDesc> procDesc(new ProcessDesc());
   builder.fill(nodes, *procDesc);
   return procDesc;
}

   }
}
