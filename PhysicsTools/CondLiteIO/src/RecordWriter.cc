// -*- C++ -*-
//
// Package:     FWCondLite
// Class  :     RecordWriter
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Dec  9 17:01:03 CST 2009
// $Id: RecordWriter.cc,v 1.6 2013/01/24 23:11:18 wmtan Exp $
//

// system include files
#include <cassert>
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

// user include files
#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/FWLite/interface/format_type_name.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace fwlite;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RecordWriter::RecordWriter(const char* iName, TFile* iFile):
pAux_(&aux_)
{
   tree_ = new TTree(fwlite::format_type_to_mangled(iName).c_str(),"Holds data for an EventSetup Record");
   tree_->SetDirectory(iFile);
   
   auxBranch_ = tree_->Branch("ESRecordAuxiliary","edm::ESRecordAuxiliary",&pAux_);
}

// RecordWriter::RecordWriter(const RecordWriter& rhs)
// {
//    // do actual copying here;
// }

RecordWriter::~RecordWriter()
{
}

//
// assignment operators
//
// const RecordWriter& RecordWriter::operator=(const RecordWriter& rhs)
// {
//   //An exception safe implementation is
//   RecordWriter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
RecordWriter::update(const void* iData, const std::type_info& iType, const char* iLabel)
{
   const char* label = iLabel;
   if(0==iLabel) {
      label = "";
   }
   std::map<std::pair<edm::TypeIDBase,std::string>, DataBuffer>::iterator itFound = idToBuffer_.find(std::make_pair(edm::TypeIDBase(iType),
      std::string(iLabel)));
   if(itFound == idToBuffer_.end()) {
      //first request
      DataBuffer buffer;
      buffer.pBuffer_=iData;
      edm::TypeWithDict t(iType);
      assert(bool(t));
      
      std::string className = t.name();
      
      //now find actual type
      edm::ObjectWithDict o(t,const_cast<void*>(iData));
      edm::TypeWithDict trueType(o.dynamicType());
      buffer.trueType_ = edm::TypeIDBase(trueType.typeInfo());
      std::string trueClassName = trueType.name();
      
      buffer.branch_ = tree_->Branch((fwlite::format_type_to_mangled(className)+"__"+label).c_str(),
                                     trueClassName.c_str(),
                                     &buffer.pBuffer_);
      idToBuffer_.insert(std::make_pair(std::make_pair(edm::TypeIDBase(iType),std::string(iLabel)),buffer));
      itFound = idToBuffer_.find(std::make_pair(edm::TypeIDBase(iType),
         std::string(iLabel)));
   }
   edm::TypeWithDict t(iType);
   edm::ObjectWithDict o(t,const_cast<void*>(iData));
   edm::TypeWithDict trueType(o.dynamicType());
   assert(edm::TypeIDBase(trueType.typeInfo())==itFound->second.trueType_);
   itFound->second.branch_->SetAddress(&(itFound->second.pBuffer_));
   itFound->second.pBuffer_ = iData;
}

//call update before calling write
void 
RecordWriter::fill(const edm::ESRecordAuxiliary& iValue)
{
   for(std::map<std::pair<edm::TypeIDBase,std::string>, DataBuffer>::iterator it=idToBuffer_.begin(),itEnd=idToBuffer_.end();
       it!=itEnd;++it) {
      if(0==it->second.pBuffer_) {
         throw cms::Exception("MissingESData")<<"The EventSetup data "<<it->first.first.name()<<" '"<<it->first.second<<"' was not supplied";
      }
   }
   
   aux_ = iValue;
   tree_->Fill();
   for(std::map<std::pair<edm::TypeIDBase,std::string>, DataBuffer>::iterator it=idToBuffer_.begin(),itEnd=idToBuffer_.end();
       it!=itEnd;++it) {
          it->second.pBuffer_=0;
   }
}

void
RecordWriter::write()
{
   tree_->Write();
}
//
// const member functions
//

//
// static member functions
//
