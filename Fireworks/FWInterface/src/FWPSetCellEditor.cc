// -*- C++ -*-
//
// Package:     FWInterface
// Class  :     FWPSetCellEditor
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon Feb 28 20:44:59 CET 2011
// $Id: FWPSetCellEditor.cc,v 1.7 2012/09/08 06:27:35 amraktad Exp $
//
#include <boost/algorithm/string.hpp>
#include <sstream>
#include "KeySymbols.h"

// user include files
#include "Fireworks/FWInterface/src/FWPSetCellEditor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "FWCore/Utilities/interface/Parse.h"

//______________________________________________________________________________

template <class T>
bool editNumericParameter(edm::ParameterSet &ps, bool tracked, 
                          const std::string &label, 
                          const std::string &value) 
{
   std::stringstream  str(value);
   T v;
   str >> v;
   bool fail = str.fail();
   if (tracked)
      ps.addParameter(label, v);
   else
      ps.addUntrackedParameter(label, v);
         
   return fail;
}
//______________________________________________________________________________

void editStringParameter(edm::ParameterSet &ps, bool tracked,
                         const std::string &label,
                         const std::string &value)
{
   if (tracked)
      ps.addParameter(label, value);
   else
      ps.addUntrackedParameter(label, value);
}

//______________________________________________________________________________

void editBoolParameter(edm::ParameterSet &ps, bool tracked,
		       const std::string &label,
		       const std::string &value)
{
   bool x = false;

   if (boost::iequals(value, "true")) {
      x = true;
   }
   else if (boost::iequals(value, "false")){
      x = false;
   }
   else {
      fwLog(fwlog::kError) << "Invalid value. Possible values are true/false case insensitive." << std::endl;
      return;
   }
   if (tracked)
      ps.addParameter<bool>(label, x);
   else
      ps.addUntrackedParameter<bool>(label, x);
}

//______________________________________________________________________________
void editFileInPath(edm::ParameterSet &ps, bool tracked,
                    const std::string &label,
                    const std::string &value)
{
   if (tracked)
      ps.addParameter(label, edm::FileInPath(value));
   else
      ps.addUntrackedParameter(label, edm::FileInPath(value));
}

//______________________________________________________________________________

bool editVInputTag(edm::ParameterSet &ps, bool tracked,
                   const std::string &label,
                   const std::string &value)
{ 
   std::vector<edm::InputTag> inputTags;
   std::stringstream iss(value);
   std::string vitem;
   bool fail = false;
   size_t fst, lst;

   while (getline(iss, vitem, ','))
   {
      fst = vitem.find("[");
      lst = vitem.find("]");
        
      if ( fst != std::string::npos )
         vitem.erase(fst,1);
      if ( lst != std::string::npos )
         vitem.erase(lst,1);
        
      std::vector<std::string> tokens = edm::tokenize(vitem, ":");
      size_t nwords = tokens.size();
        
      if ( nwords > 3 )
      {
         fail = true;
         return fail;
      }
      else 
      {
         std::string it_label("");
         std::string it_instance("");
         std::string it_process("");

         if ( nwords > 0 ) 
            it_label = tokens[0];
         if ( nwords > 1 ) 
            it_instance = tokens[1];
         if ( nwords > 2 ) 
            it_process  = tokens[2];
        
         inputTags.push_back(edm::InputTag(it_label, it_instance, it_process));
      }
   }
     
   if (tracked)
      ps.addParameter(label, inputTags);
   else
      ps.addUntrackedParameter(label, inputTags);

   return fail;
}

//______________________________________________________________________________

bool editInputTag(edm::ParameterSet &ps, bool tracked,
                  const std::string &label,
                  const std::string &value)
{
   std::vector<std::string> tokens = edm::tokenize(value, ":");
   size_t nwords = tokens.size();
     
   bool fail;

   if ( nwords > 3 ) 
   {
      fail = true;
   }
   else
   {           
      std::string it_label("");
      std::string it_instance("");
      std::string it_process("");

      if ( nwords > 0 ) 
         it_label = tokens[0];
      if ( nwords > 1 ) 
         it_instance = tokens[1];
      if ( nwords > 2 ) 
         it_process  = tokens[2];

      if ( tracked )
         ps.addParameter(label, edm::InputTag(it_label, it_instance, it_process));
      else
         ps.addUntrackedParameter(label, edm::InputTag(it_label, it_instance, it_process));
            
      fail = false;
   }
           
   return fail;
}

//______________________________________________________________________________

bool editESInputTag(edm::ParameterSet &ps, bool tracked,
                    const std::string &label,
                    const std::string &value)
{
   std::vector<std::string> tokens = edm::tokenize(value, ":");
   size_t nwords = tokens.size();
      
   bool fail;
  
   if ( nwords > 2 )
   {
      fail = true;    
   }
   else
   {             
      std::string it_module("");
      std::string it_data("");

      if ( nwords > 0 ) 
         it_module = tokens[0];
      if ( nwords > 1 ) 
         it_data = tokens[1];

      if ( tracked )
         ps.addParameter(label, edm::ESInputTag(it_module, it_data));
      else
         ps.addUntrackedParameter(label, edm::ESInputTag(it_module, it_data));
        
      fail = false;
   }

   return fail;
}
  
//______________________________________________________________________________
template <typename T>
void editVectorParameter(edm::ParameterSet &ps, bool tracked,
                         const std::string &label,
                         const std::string &value)
{
   std::vector<T> valueVector;
      
   std::stringstream iss(value);
   std::string vitem;
      
   size_t fst, lst;

   while (getline(iss, vitem, ','))
   {
      fst = vitem.find("[");
      lst = vitem.find("]");
        
      if ( fst != std::string::npos )
         vitem.erase(fst,1);
      if ( lst != std::string::npos )
         vitem.erase(lst,1);
        
      std::stringstream oss(vitem);
      T on;
      oss >> on;

      valueVector.push_back(on);
   }
     
   if (tracked)
      ps.addParameter(label, valueVector);
   else
      ps.addUntrackedParameter(label, valueVector);
}

//______________________________________________________________________________

bool FWPSetCellEditor::apply(FWPSetTableManager::PSetData &data, FWPSetTableManager::PSetData &parent)
{
   switch (data.type)
   {
      case 'I':
         editNumericParameter<int32_t>(*parent.pset, data.tracked, data.label, GetText());
         break;
       case 'B':
         editBoolParameter(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'U':
         editNumericParameter<uint32_t>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'D':
         editNumericParameter<double>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'L':
         editNumericParameter<long long>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'X':
         editNumericParameter<unsigned long long>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'S':
         editStringParameter(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'i':
         editVectorParameter<int32_t>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'u':
         editVectorParameter<uint32_t>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'l':
         editVectorParameter<long long>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'x':
         editVectorParameter<unsigned long long>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'd':
         editVectorParameter<double>(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 's':
         editVectorParameter<std::string>(*parent.pset, data.tracked, data.label, GetText());
         break; 
      case 't':
         editInputTag(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'g':
         editESInputTag(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'v':
         editVInputTag(*parent.pset, data.tracked, data.label, GetText());
         break;
      case 'F':
         editFileInPath(*parent.pset, data.tracked, data.label, GetText());
         break;
      default:
         fwLog(fwlog::kError) << "unsupported parameter" << std::endl;
         UnmapWindow();
         return false;
   }
   return true;
}

//______________________________________________________________________________

bool FWPSetCellEditor::HandleKey(Event_t*event)
{
   UInt_t keysym = event->fCode;

   if (keysym == (UInt_t) gVirtualX->KeysymToKeycode(kKey_Escape))
   {
      TGFrame *p = dynamic_cast<TGFrame*>(const_cast<TGWindow*>(GetParent()));
      while (p)
      {
         TGMainFrame *mp = dynamic_cast<TGMainFrame*>(p);
         //   printf("editor find parent %p, %s, %p\n", p, p->ClassName(), mp);
         if (mp)
         {
            return mp->HandleKey(event);
         }
         p = dynamic_cast<TGFrame*>(const_cast<TGWindow*>(p->GetParent()));
      }
   }

   return TGTextEntry::HandleKey(event);
}
