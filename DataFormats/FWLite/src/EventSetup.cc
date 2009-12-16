// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventSetup
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Thu Dec 10 15:58:26 CST 2009
// $Id$
//

// system include files
#include <cassert>
#include <algorithm>
#include "boost/bind.hpp"
#include "TTree.h"
#include "TFile.h"

// user include files
#include "DataFormats/FWLite/interface/EventSetup.h"
#include "DataFormats/FWLite/interface/format_type_name.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//
using namespace fwlite;
static const char* const kRecordAuxiliaryBranchName="ESRecordAuxiliary";
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetup::EventSetup(TFile* iFile):
m_event(0),
m_file(iFile)
{
}

// EventSetup::EventSetup(const EventSetup& rhs)
// {
//    // do actual copying here;
// }

EventSetup::~EventSetup()
{
   for(std::vector<Record*>::iterator it = m_records.begin(), itEnd=m_records.end();
    it !=itEnd; ++it) {
       delete *it;
   }
}

//
// assignment operators
//
// const EventSetup& EventSetup::operator=(const EventSetup& rhs)
// {
//   //An exception safe implementation is
//   EventSetup temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EventSetup::syncTo(const edm::EventID& iID, const edm::Timestamp& iTime) {
   std::for_each(m_records.begin(),
                 m_records.end(),
                 boost::bind(&Record::syncTo,_1,iID,iTime));
}

//
// const member functions
//
bool 
EventSetup::exists(const char* iRecordName) const
{
   std::string realName = unformat_mangled_to_type(iRecordName);
   TObject* obj = m_file->Get(realName.c_str());
   if(0==obj) {
      return false;
   }
   TTree* tree = dynamic_cast<TTree*>(obj);
   if(0==tree) {
      return false;
   }
   return 0 != tree->FindBranch(kRecordAuxiliaryBranchName);
}

RecordID 
EventSetup::recordID(const char* iRecordName) const
{
   std::string treeName = format_type_to_mangled(iRecordName);
   TObject* obj = m_file->Get(treeName.c_str());
   if(0==obj) {
      throw cms::Exception("UnknownRecord")<<"The TTree for the record "<<iRecordName<<" does not exist "<<m_file->GetName();
   }
   TTree* tree = dynamic_cast<TTree*>(obj);
   if(0==tree) {
      throw cms::Exception("UnknownRecord")<<"The object corresponding to "<<iRecordName<<" in file "<<m_file->GetName()<<" is not a TTree and therefore is not a Record";   
   }
   if(0 == tree->FindBranch(kRecordAuxiliaryBranchName)) {
      throw cms::Exception("UnknownRecord")<<"The TTree corresponding to "<<iRecordName<<" in file "<<m_file->GetName()<<" does not have the proper structure to be a Record";
   }
   //do we already have this Record?
   std::string name(iRecordName);
   for(std::vector<Record*>::iterator it = m_records.begin(), itEnd=m_records.end(); it!=itEnd;++it){
      if((*it)->name()==name) {
         return it - m_records.begin();
      }
   }
   
   //Not found so need to make a new one
   Record* rec = new Record(iRecordName, tree);
   m_records.push_back(rec);
   return m_records.size()-1;
}

const Record& 
EventSetup::get(const RecordID& iID) const
{
   assert(iID<m_records.size());
   return *(m_records[iID]);
}

//
// static member functions
//
