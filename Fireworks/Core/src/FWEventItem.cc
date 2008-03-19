// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEventItem
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Jan  3 14:59:23 EST 2008
// $Id: FWEventItem.cc,v 1.11 2008/03/05 15:11:32 chrjones Exp $
//

// system include files
#include <TClass.h>
#include "TVirtualCollectionProxy.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEventItem::FWEventItem(FWModelChangeManager* iCM,
                         FWSelectionManager* iSM,
                         unsigned int iId,
                         const std::string& iName,
			 const TClass* iClass,
			 const FWDisplayProperties& iProperties,
			 const std::string& iModuleLabel,
			 const std::string& iProductInstanceLabel,
			 const std::string& iProcessName,
                         unsigned int iLayer) :
  m_changeManager(iCM),
  m_selectionManager(iSM),
  m_id(iId),
  m_name(iName),
  m_type(iClass),
  m_colProxy(iClass->GetCollectionProxy()?iClass->GetCollectionProxy()->Generate():
                                          static_cast<TVirtualCollectionProxy*>(0)),
  m_data(0),
  m_collectionOffset(0),
  m_displayProperties(iProperties),
  m_layer(iLayer),
  m_moduleLabel(iModuleLabel),
  m_productInstanceLabel(iProductInstanceLabel),
  m_processName(iProcessName),
  m_event(0),
m_filter("","")
{
  assert(m_type->GetTypeInfo());
  ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*(m_type->GetTypeInfo())));
  assert(dataType != ROOT::Reflex::Type() );

  std::string wrapperName = std::string("edm::Wrapper<")+dataType.Name(ROOT::Reflex::SCOPED)+" >";
  std::cout <<wrapperName<<std::endl;
  m_wrapperType = ROOT::Reflex::Type::ByName(wrapperName);

  assert(m_wrapperType != ROOT::Reflex::Type());
}

FWEventItem::FWEventItem(FWModelChangeManager* iCM,
                         FWSelectionManager* iSM,
                         unsigned int iId,
                         const FWPhysicsObjectDesc& iDesc) :
m_changeManager(iCM),
m_selectionManager(iSM),
m_id(iId),
m_name(iDesc.name()),
m_type(iDesc.type()),
m_colProxy(m_type->GetCollectionProxy()?m_type->GetCollectionProxy()->Generate():
                                        static_cast<TVirtualCollectionProxy*>(0)),
m_data(0),
m_collectionOffset(0),
m_displayProperties(iDesc.displayProperties()),
m_layer(iDesc.layer()),
m_moduleLabel(iDesc.moduleLabel()),
m_productInstanceLabel(iDesc.productInstanceLabel()),
m_processName(iDesc.processName()),
m_event(0),
m_filter(iDesc.filterExpression(),"")
{
   assert(m_type->GetTypeInfo());
   ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*(m_type->GetTypeInfo())));
   assert(dataType != ROOT::Reflex::Type() );
   
   std::string wrapperName = std::string("edm::Wrapper<")+dataType.Name(ROOT::Reflex::SCOPED)+" >";
   //std::cout <<wrapperName<<std::endl;
   m_wrapperType = ROOT::Reflex::Type::ByName(wrapperName);
   
   assert(m_wrapperType != ROOT::Reflex::Type());
   if(0==m_colProxy) {
      //is this an object which has only one member item and that item is a container?
      if(dataType.DataMemberSize()==1) {
         ROOT::Reflex::Type memType( dataType.DataMemberAt(0).TypeOf() );
         assert(memType != ROOT::Reflex::Type());
         const TClass* rootMemType = TClass::GetClass(memType.TypeInfo());
         assert(rootMemType != 0);
         if(rootMemType->GetCollectionProxy()) {
            m_colProxy=boost::shared_ptr<TVirtualCollectionProxy>(rootMemType->GetCollectionProxy()->Generate());
         }
      }
      if(0==m_colProxy) {
         m_itemInfos.reserve(1);
      }
   }
   m_filter.setClassName(modelType()->GetName());
   //only want to listen to this signal when we need to run the filter
   m_shouldFilterConnection = m_changeManager->changeSignalsAreDone_.connect(sigc::mem_fun(*this,&FWEventItem::runFilter));
   m_shouldFilterConnection.block(true);

}
// FWEventItem::FWEventItem(const FWEventItem& rhs)
// {
//    // do actual copying here;
// }
/*
FWEventItem::~FWEventItem()
{
}
*/
//
// assignment operators
//
// const FWEventItem& FWEventItem::operator=(const FWEventItem& rhs)
// {
//   //An exception safe implementation is
//   FWEventItem temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWEventItem::setEvent(const fwlite::Event* iEvent) 
{
   m_event = iEvent;
   m_data = 0;
   if(m_colProxy.get()) {
      m_colProxy->PopProxy();
   }
   m_itemInfos.clear();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   m_changeManager->changed(this);
}

void 
FWEventItem::setLabels(const std::string& iModule,
		       const std::string& iProductInstance,
		       const std::string& iProcess) 
{
  m_moduleLabel = iModule;
  m_productInstanceLabel = iProductInstance;
  m_processName = iProcess;
  m_data = 0;
   if(m_colProxy.get()) {
      m_colProxy->PopProxy();
   }
   m_itemInfos.clear();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   m_changeManager->changed(this);
}

void 
FWEventItem::setName(const std::string& iName) 
{
  m_name = iName;
}

void 
FWEventItem::setDefaultDisplayProperties(const FWDisplayProperties& iProp)
{
   m_displayProperties= iProp;
}

void 
FWEventItem::setFilterExpression(const std::string& iExpression)
{
   m_filter.setExpression(iExpression);
   runFilter();
}

void
FWEventItem::runFilter()
{   
   m_shouldFilterConnection.block(true);

   if(not m_filter.trivialFilter() && m_colProxy.get() && m_data) {
      std::cout <<"runFilter"<<std::endl;
      FWChangeSentry sentry(*(this->changeManager()));
      int size = m_colProxy->Size();
      std::vector<ModelInfo>::iterator itInfo = m_itemInfos.begin();
      for(int index = 0; index != size; ++index,++itInfo) {
         if(not m_filter.passesFilter(m_colProxy->At(index))) {
            itInfo->m_displayProperties.setIsVisible(false);
         } else {
            itInfo->m_displayProperties.setIsVisible(true);
         }
         FWModelId id(this,index);
         m_changeManager->changed(id);
      }
   }
}

void 
FWEventItem::unselect(int iIndex) const
{
   //check if this is a change
   if(bool& sel = m_itemInfos.at(iIndex).m_isSelected) {
      sel=false;
      FWModelId id(this,iIndex);
      m_selectionManager->unselect(id);
      m_changeManager->changed(id);
   }
}
void
FWEventItem::select(int iIndex) const
{
   bool& sel = m_itemInfos.at(iIndex).m_isSelected;
   if(not sel) {
      sel = true;
      FWModelId id(this,iIndex);
      m_selectionManager->select(id);
      m_changeManager->changed(id);
   }
}
void 
FWEventItem::toggleSelect(int iIndex) const
{
   bool& sel = m_itemInfos.at(iIndex).m_isSelected;
   sel = not sel;
   FWModelId id(this,iIndex);
   m_selectionManager->select(id);
   m_changeManager->changed(id);
}

void 
FWEventItem::setDisplayProperties(int iIndex, const FWDisplayProperties& iProps) const
{
   FWDisplayProperties& prop = m_itemInfos.at(iIndex).m_displayProperties;
   if( prop
      != iProps ) {
      prop = iProps;
      FWModelId id(this,iIndex);
      //m_selectionManager->select(id);
      m_changeManager->changed(id);
   }
}

//
// const member functions
//
const void* 
FWEventItem::data(const std::type_info& iInfo) const
{
  //std::cerr <<"asked to get data "<<m_type->GetName()<<std::endl;
  using namespace ROOT::Reflex;
  //At the moment this is a programming error
  assert(iInfo == *(m_type->GetTypeInfo()) );

  //lookup data if we don't already have it
  if(0 == m_data) {
    void* wrapper=0;
    void* temp = &wrapper;
    if(m_event) {
      m_event->getByLabel(m_wrapperType.TypeInfo(),
			  m_moduleLabel.c_str(),
			  m_productInstanceLabel.c_str(),
			  m_processName.size()?m_processName.c_str():0,
			  temp);
      if(wrapper==0) {
	//should report a problem
	std::cerr<<"failed getByLabel"<<std::endl;
	return 0;
      }
      //Get Reflex to do the work
      Object wrapperObj(m_wrapperType,wrapper);

      //Convert our wrapper to its EDProduct base class
      static Type s_edproductType(Type::ByTypeInfo(typeid(edm::EDProduct)));
      Object edproductObj(wrapperObj.CastObject(s_edproductType));
      const edm::EDProduct* prod = reinterpret_cast<const edm::EDProduct*>(edproductObj.Address());
      if(not prod->isPresent()) {
	//not actually in this event
	std::cerr <<"data unavailable for this event"<<std::endl;
	return 0;
      }

      //get the Event data from the wrapper
      Object product(wrapperObj.Get("obj"));
      if(product.TypeOf().IsTypedef()) {
	product = Object(product.TypeOf().ToType(),product.Address());
      }
      setData(product.Address());
    }
  }
  return m_data;
}

void 
FWEventItem::setData(const void* iData) const
{
   m_data = iData;
   if(m_colProxy) {
      m_colProxy->PushProxy(static_cast<char*>(const_cast<void*>(m_data))+m_collectionOffset);
      m_itemInfos.reserve(m_colProxy->Size());
      m_itemInfos.resize(m_colProxy->Size(),ModelInfo(m_displayProperties,false));
   } else {
      m_itemInfos.push_back(ModelInfo(m_displayProperties,false));
   }
}

void 
FWEventItem::getPrimaryData() const
{
   if(0!=m_data) return;
   this->data(*(m_type->GetTypeInfo()));
}

const FWDisplayProperties& 
FWEventItem::defaultDisplayProperties() const
{
  return m_displayProperties;
}

unsigned int 
FWEventItem::layer() const
{
   return m_layer;
}


unsigned int 
FWEventItem::id() const
{
   return m_id;
}

const std::string& 
FWEventItem::name() const
{
  return m_name;
}

const TClass* 
FWEventItem::type() const
{
  return m_type;
}

const std::string& 
FWEventItem::moduleLabel() const
{
  return m_moduleLabel;
}
const std::string& 
FWEventItem::productInstanceLabel() const
{
  return m_productInstanceLabel;
}

const std::string& 
FWEventItem::processName() const
{
  return m_processName;
}

const FWEventItem::ModelInfo& 
FWEventItem::modelInfo(int iIndex) const
{
   getPrimaryData();
   return m_itemInfos.at(iIndex);
}

size_t
FWEventItem::size() const
{
   getPrimaryData();
   return m_itemInfos.size();
}

const TClass* 
FWEventItem::modelType() const
{
   return 0 != m_colProxy.get()? m_colProxy->GetValueClass() : m_type;
}

const void* 
FWEventItem::modelData(int iIndex) const
{
   getPrimaryData();
   if ( 0 == m_data) { return m_data; }
   return 0 != m_colProxy.get()? m_colProxy->At(iIndex) : m_data;
}

const std::string& 
FWEventItem::filterExpression() const
{
   return m_filter.expression();
}

//
// static member functions
//
