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
// $Id: FW3DSimpleProxyBuilder.cc,v 1.2 2008/12/03 01:15:17 chrjones Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilder.h"
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
FW3DSimpleProxyBuilder::FW3DSimpleProxyBuilder(const std::type_info& iType):
m_helper(iType)
{
}

// FW3DSimpleProxyBuilder::FW3DSimpleProxyBuilder(const FW3DSimpleProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DSimpleProxyBuilder::~FW3DSimpleProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DSimpleProxyBuilder& FW3DSimpleProxyBuilder::operator=(const FW3DSimpleProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DSimpleProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FW3DSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   if(0!=iItem) {
      m_helper.itemChanged(iItem);
   }   
}

void 
FW3DSimpleProxyBuilder::build(const FWEventItem* iItem,
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
FW3DSimpleProxyBuilder::typeOfBuilder()
{
   return std::string("simple#");
}
