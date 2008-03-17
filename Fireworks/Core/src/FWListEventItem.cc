// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListEventItem
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 28 11:13:37 PST 2008
// $Id: FWListEventItem.cc,v 1.10 2008/03/16 23:12:51 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include <iostream>
#include <sstream>
#include "TEveManager.h"
#include "TEveSelection.h"

#include "TClass.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "Reflex/Base.h"

// user include files
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWListModel.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
const std::vector<std::string>&
defaultMemberFunctionNames()
{
   static std::vector<std::string> s_names;
   if(s_names.empty()){
      s_names.push_back("pt");
      s_names.push_back("et");
      s_names.push_back("energy");
   }
   return s_names;
}

static
ROOT::Reflex::Member
recursiveFindMember(const std::string& iName, 
                    const ROOT::Reflex::Type& iType)
{
   using namespace ROOT::Reflex;

   Member temp = iType.MemberByName(iName);
   if(temp) {return temp;}
   
   //try all base classes
   for(Base_Iterator it = iType.Base_Begin(), itEnd = iType.Base_End();
       it != itEnd;
       ++it) {
      temp = recursiveFindMember(iName,it->ToType());
      if(temp) {break;}
   }
   return temp;
}


static
ROOT::Reflex::Member
findDefaultMember(const TClass* iClass) {
   using namespace ROOT::Reflex;
   if(0==iClass) {
      return Member();
   }
   
   Type rType = Type::ByTypeInfo(*(iClass->GetTypeInfo()));
   assert(rType != Type() );
   //std::cout <<"Type "<<rType.Name()<<std::endl;
   
   Member returnValue;
   const std::vector<std::string>& names = defaultMemberFunctionNames();
   for(std::vector<std::string>::const_iterator it = names.begin(), itEnd=names.end();
       it != itEnd;
       ++it) {
      //std::cout <<" trying function "<<*it<<std::endl;
      Member temp = recursiveFindMember(*it,rType);
      if(temp) {
         if(0==temp.FunctionParameterSize(true)) {
            //std::cout <<"    FOUND "<<temp.Name()<<std::endl;
            returnValue = temp;
            break;
         }
      }
   }
   return returnValue;
}

namespace {
   template <class T>
   std::string valueToString(const std::string& iName, const ROOT::Reflex::Object& iObj) {
      std::stringstream s;
      s.setf(std::ios_base::fixed,std::ios_base::floatfield);
      s.precision(2);
      s<<iName <<" = "<<*(reinterpret_cast<T*>(iObj.Address()));
      return s.str();
   }

   typedef std::string(*FunctionType)(const std::string&,const ROOT::Reflex::Object&);
   typedef std::map<std::string, FunctionType> TypeToPrintMap;
    
   template<typename T>
   static void addToMap(TypeToPrintMap& iMap) {
      iMap[typeid(T).name()]=valueToString<T>;
   }
}

static
std::string
valueFor(const ROOT::Reflex::Object& iObj, const ROOT::Reflex::Member& iMember) {
   static TypeToPrintMap s_map;
   if(s_map.empty() ) {
      addToMap<float>(s_map);
      addToMap<double>(s_map);
   }

   ROOT::Reflex::Object val = iMember.Invoke(iObj);

   TypeToPrintMap::iterator itFound =s_map.find(val.TypeOf().TypeInfo().name());
   if(itFound == s_map.end()) {
      //std::cout <<" could not print because type is "<<iObj.TypeOf().TypeInfo().name()<<std::endl;
      return std::string();
   }
   
   return itFound->second(iMember.Name(),val);
 }

//
// constructors and destructor
//
FWListEventItem::FWListEventItem(FWEventItem* iItem,
                                 FWDetailViewManager* iDV):
TEveElementList(iItem->name().c_str(),"",kTRUE),
m_item(iItem),
m_detailViewManager(iDV),
m_memberFunction(findDefaultMember(iItem->modelType()))
{
   m_item->itemChanged_.connect(boost::bind(&FWListEventItem::itemChanged,this,_1));
   m_item->changed_.connect(boost::bind(&FWListEventItem::modelsChanged,this,_1));
   TEveElementList::SetMainColor(iItem->defaultDisplayProperties().color());
}

// FWListEventItem::FWListEventItem(const FWListEventItem& rhs)
// {
//    // do actual copying here;
// }

FWListEventItem::~FWListEventItem()
{
}

//
// assignment operators
//
// const FWListEventItem& FWListEventItem::operator=(const FWListEventItem& rhs)
// {
//   //An exception safe implementation is
//   FWListEventItem temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool 
FWListEventItem::doSelection(bool iToggleSelection)
{
   return true;
}

void 
FWListEventItem::SetMainColor(Color_t iColor)
{
   FWChangeSentry sentry(*(m_item->changeManager()));
   FWDisplayProperties prop(iColor,m_item->defaultDisplayProperties().isVisible());
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetMainColor(iColor);
   
   for(int index=0; index <static_cast<int>(m_item->size()); ++index) {
      FWDisplayProperties prop=m_item->modelInfo(index).displayProperties();
      if(iColor !=prop.color()) {
          prop.setColor(iColor);
         m_item->setDisplayProperties(index,prop);
      }
   }
}


void 
FWListEventItem::SetRnrState(Bool_t rnr)
{
   FWChangeSentry sentry(*(m_item->changeManager()));
   FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetRnrState(rnr);

   for(int index=0; index <static_cast<int>(m_item->size()); ++index) {
      FWDisplayProperties prop=m_item->modelInfo(index).displayProperties();
      if(rnr !=prop.isVisible()) {
          prop.setIsVisible(rnr);
         m_item->setDisplayProperties(index,prop);
      }
   }
   
}
Bool_t 
FWListEventItem::SingleRnrState() const
{
   return kTRUE;
}


void 
FWListEventItem::itemChanged(const FWEventItem* iItem)
{
   //std::cout <<"item changed "<<eventItem()->size()<<std::endl;
   this->DestroyElements();
   for(unsigned int index = 0; index < eventItem()->size(); ++index) {
      ROOT::Reflex::Object obj;
      std::string data;
      if(m_memberFunction) {
         //the const_cast is fine since I'm calling a const member function
         ROOT::Reflex::Object temp(m_memberFunction.DeclaringType(),
                                   const_cast<void*>(eventItem()->modelData(index)));
         obj=temp;
         data = valueFor(obj,m_memberFunction);
      }
      FWListModel* model = new FWListModel(FWModelId(eventItem(),index), 
                                           m_detailViewManager,
                                           data);
      this->AddElement( model );
      model->SetMainColor(m_item->defaultDisplayProperties().color());
   }
}

void 
FWListEventItem::modelsChanged( const std::set<FWModelId>& iModels )
{
   //std::cout <<"modelsChanged "<<std::endl;
   bool aChildChanged = false;
   TEveElement::List_i itElement = this->BeginChildren();
   int index = 0;
   for(FWModelIds::const_iterator it = iModels.begin(), itEnd = iModels.end();
       it != itEnd;
       ++it,++itElement,++index) {
      assert(itElement != this->EndChildren());         
      while(index < it->index()) {
         ++itElement;
         ++index;
         assert(itElement != this->EndChildren());         
      }
      //std::cout <<"   "<<index<<std::endl;
      bool modelChanged = false;
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
      FWListModel* model = static_cast<FWListModel*>(*itElement);
      modelChanged = model->update(info.displayProperties());
      if(info.isSelected() xor (*itElement)->GetSelectedLevel()==1) {
         modelChanged = true;
         if(info.isSelected()) {         
            gEve->GetSelection()->AddElement(*itElement);
         } else {
            gEve->GetSelection()->RemoveElement(*itElement);
         }
      }      
      if(modelChanged) {
         (*itElement)->ElementChanged();
         aChildChanged=true;
         //(*itElement)->UpdateItems();  //needed to force list tree to update immediately
      }
   }
   if(aChildChanged) {
      this->UpdateItems();
   }
   //std::cout <<"modelsChanged done"<<std::endl;

}

//
// const member functions
//
FWEventItem* 
FWListEventItem::eventItem() const
{
   return m_item;
}

void 
FWListEventItem::openDetailViewFor(int index) const
{
   m_detailViewManager->openDetailViewFor( FWModelId(m_item,index));
}

//
// static member functions
//

ClassImp(FWListEventItem)
