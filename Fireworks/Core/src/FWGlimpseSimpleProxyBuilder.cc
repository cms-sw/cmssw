// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseSimpleProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 09:46:41 EST 2008
// $Id: FWGlimpseSimpleProxyBuilder.cc,v 1.1 2008/12/02 21:11:53 chrjones Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilder.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGlimpseSimpleProxyBuilder::FWGlimpseSimpleProxyBuilder(const std::type_info& iType):
m_helper(iType)
{
}

// FWGlimpseSimpleProxyBuilder::FWGlimpseSimpleProxyBuilder(const FWGlimpseSimpleProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWGlimpseSimpleProxyBuilder::~FWGlimpseSimpleProxyBuilder()
{
}

//
// assignment operators
//
// const FWGlimpseSimpleProxyBuilder& FWGlimpseSimpleProxyBuilder::operator=(const FWGlimpseSimpleProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWGlimpseSimpleProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWGlimpseSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   if(0!=iItem) {
      m_helper.itemChanged(iItem);
   }   
}

void 
FWGlimpseSimpleProxyBuilder::build(const FWEventItem* iItem,
                                   TEveElementList** product)
{
   if(0==*product) {
      *product = new TEveElementList();
   } else {
      (*product)->DestroyElements();
   }
   size_t size = iItem->size();
   for(int index = 0; index < static_cast<int>(size); ++index) {
      const void* modelData = iItem->modelData(index);
      std::string name;
      m_helper.fillTitle(*iItem,index,name);
      std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
      {
         itemHolder->OpenCompound();
         //guarantees that CloseCompound will be called no matter what happens
         boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                boost::mem_fn(&TEveCompound::CloseCompound));
         build(m_helper.offsetObject(modelData),index,*itemHolder);
      }
      const FWEventItem::ModelInfo& info = iItem->modelInfo(index);
      changeElementAndChildren(itemHolder.get(), info);
      itemHolder->SetRnrSelf(info.displayProperties().isVisible());
      itemHolder->SetRnrChildren(info.displayProperties().isVisible());
      itemHolder->ElementChanged();
      (*product)->AddElement(itemHolder.release());
   }   
}

//
// const member functions
//

//
// static member functions
//
std::string 
FWGlimpseSimpleProxyBuilder::typeOfBuilder()
{
   return std::string("simple#");
}
