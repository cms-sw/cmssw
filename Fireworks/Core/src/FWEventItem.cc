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
// $Id: FWEventItem.cc,v 1.35 2009/07/31 15:42:14 chrjones Exp $
//

// system include files
#include <iostream>
#include <exception>
#include <TClass.h>
#include "TVirtualCollectionProxy.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#define private public
#include "DataFormats/FWLite/interface/Event.h"
#undef private
#include "DataFormats/Common/interface/EDProduct.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWItemAccessorBase.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
const std::vector<std::pair<std::string,std::string> >&
defaultMemberFunctionNames()
{
   static std::vector<std::pair<std::string,std::string> > s_names;
   if(s_names.empty()){
      s_names.push_back(std::pair<std::string,std::string>("pt","GeV"));
      s_names.push_back(std::pair<std::string,std::string>("et","GeV"));
      s_names.push_back(std::pair<std::string,std::string>("energy","GeV"));
   }
   return s_names;
}

//
// constructors and destructor
//
FWEventItem::FWEventItem(fireworks::Context* iContext,
                         unsigned int iId,
                         boost::shared_ptr<FWItemAccessorBase> iAccessor,
                         const FWPhysicsObjectDesc& iDesc) :
   m_context(iContext),
   m_id(iId),
   m_name(iDesc.name()),
   m_type(iDesc.type()),
   m_purpose(iDesc.purpose()),
   m_accessor(iAccessor),
   m_displayProperties(iDesc.displayProperties()),
   m_layer(iDesc.layer()),
   m_moduleLabel(iDesc.moduleLabel()),
   m_productInstanceLabel(iDesc.productInstanceLabel()),
   m_processName(iDesc.processName()),
   m_event(0),
   m_interestingValueGetter(ROOT::Reflex::Type::ByTypeInfo(*(m_accessor->modelType()->GetTypeInfo())),
                            defaultMemberFunctionNames()),
   m_filter(iDesc.filterExpression(),""),
   m_printedNoDataError(false),
   m_printedErrorThisEvent(false)
{
   assert(m_type->GetTypeInfo());
   ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*(m_type->GetTypeInfo())));
   assert(dataType != ROOT::Reflex::Type() );

   std::string wrapperName = std::string("edm::Wrapper<")+dataType.Name(ROOT::Reflex::SCOPED)+" >";
   //std::cout <<wrapperName<<std::endl;
   m_wrapperType = ROOT::Reflex::Type::ByName(wrapperName);

   assert(m_wrapperType != ROOT::Reflex::Type());
   if(!m_accessor->isCollection()) {
      m_itemInfos.reserve(1);
   }
   m_filter.setClassName(modelType()->GetName());
   //only want to listen to this signal when we need to run the filter
   m_shouldFilterConnection = changeManager()->changeSignalsAreDone_.connect(sigc::mem_fun(*this,&FWEventItem::runFilter));
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
   if ( m_event != iEvent ) m_printedNoDataError = false;
   m_printedErrorThisEvent = false;
   m_event = iEvent;
   m_accessor->reset();
   m_itemInfos.clear();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   changeManager()->changed(this);
}

void
FWEventItem::setLabels(const std::string& iModule,
                       const std::string& iProductInstance,
                       const std::string& iProcess)
{
   m_moduleLabel = iModule;
   m_productInstanceLabel = iProductInstance;
   m_processName = iProcess;
   m_accessor->reset();
   m_itemInfos.clear();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   changeManager()->changed(this);
}

void
FWEventItem::setName(const std::string& iName)
{
   m_name = iName;
}

void
FWEventItem::setDefaultDisplayProperties(const FWDisplayProperties& iProp)
{
   bool visChange = m_displayProperties.isVisible() != iProp.isVisible();
   bool colorChanged = m_displayProperties.color() != iProp.color();

   if(!visChange && !colorChanged) {
      return;
   }
   //If the default visibility is changed, we want to also change the the visibility of the children
   // BUT we want to remember the old visibility so if the visibility is changed again we return
   // to the previous state.
   // only the visible ones need to be marked as 'changed'
   FWChangeSentry sentry(*(changeManager()));

   for(int index=0; index <static_cast<int>(size()); ++index) {
      FWDisplayProperties prp = m_itemInfos[index].displayProperties();
      bool vis=prp.isVisible();
      bool changed = false;
      changed = visChange && vis;
      if(colorChanged) {
         if(m_displayProperties.color()==prp.color()) {
            prp.setColor(iProp.color());
            m_itemInfos[index].m_displayProperties=prp;
            changed = true;
         }
      }
      if(changed) {
         FWModelId id(this,index);
         changeManager()->changed(id);
      }
   }
   m_displayProperties= iProp;
   defaultDisplayPropertiesChanged_(this);
}

void
FWEventItem::setFilterExpression(const std::string& iExpression)
{
   m_filter.setExpression(iExpression);
   filterChanged_(this);
   runFilter();
}

void
FWEventItem::runFilter()
{
   m_shouldFilterConnection.block(true);

   if(m_accessor->isCollection() && m_accessor->data()) {
      //std::cout <<"runFilter"<<std::endl;
      FWChangeSentry sentry(*(this->changeManager()));
      int size = m_accessor->size();
      std::vector<ModelInfo>::iterator itInfo = m_itemInfos.begin();
      try {
         for(int index = 0; index != size; ++index,++itInfo) {
            bool changed = false;
            bool wasVisible = itInfo->m_displayProperties.isVisible();
            if(not m_filter.passesFilter(m_accessor->modelData(index))) {
               itInfo->m_displayProperties.setIsVisible(false);
               changed = wasVisible==true;
            } else {
               itInfo->m_displayProperties.setIsVisible(true);
               changed = wasVisible==false;
            }
            if(changed) {
               FWModelId id(this,index);
               changeManager()->changed(id);
            }
         }
      } catch( const std::exception& iException) {
         //Should log this error
         std::cerr <<"Exception occurred while running filter on "<<name()<<"\n"
         <<iException.what()<<std::endl;
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
      selectionManager()->unselect(id);
      changeManager()->changed(id);
   }
}
void
FWEventItem::select(int iIndex) const
{
   bool& sel = m_itemInfos.at(iIndex).m_isSelected;
   if(not sel) {
      sel = true;
      FWModelId id(this,iIndex);
      selectionManager()->select(id);
      changeManager()->changed(id);
   }
}
void
FWEventItem::toggleSelect(int iIndex) const
{
   bool& sel = m_itemInfos.at(iIndex).m_isSelected;
   sel = not sel;
   FWModelId id(this,iIndex);
   if (sel)
      selectionManager()->select(id);
   else selectionManager()->unselect(id);
   changeManager()->changed(id);
}

void
FWEventItem::setDisplayProperties(int iIndex, const FWDisplayProperties& iProps) const
{
   FWDisplayProperties& prop = m_itemInfos.at(iIndex).m_displayProperties;
   if(m_displayProperties.isVisible()) {
      if( prop
          != iProps ) {
         prop = iProps;
         FWModelId id(this,iIndex);
         //selectionManager()->select(id);
         changeManager()->changed(id);
      }
   } else {
      if(iProps.isVisible()) {
         FWChangeSentry sentry(*(this->changeManager()));
         int size = m_accessor->size();
         std::vector<ModelInfo>::iterator itInfo = m_itemInfos.begin();
         for(int index = 0; index != size; ++index,++itInfo) {
            if( itInfo->m_displayProperties.isVisible() ) {
               itInfo->m_displayProperties.setIsVisible(false);
               FWModelId id(this,index);
               changeManager()->changed(id);
            }
         }
         m_itemInfos.at(iIndex).m_displayProperties.setIsVisible(true);
         FWModelId id(this,iIndex);
         changeManager()->changed(id);
         const_cast<FWEventItem*>(this)->m_displayProperties.setIsVisible(true);
         //NOTE: need to send out a signal here
         defaultDisplayPropertiesChanged_(this);
      }
   }
}

void
FWEventItem::moveToFront()
{
   assert(0!=m_context->eventItemsManager());
   int largest = layer();
   for(FWEventItemsManager::const_iterator it = m_context->eventItemsManager()->begin(),
                                           itEnd = m_context->eventItemsManager()->end();
       it != itEnd;
       ++it) {
      if( (*it) && (*it)->layer() > largest) {
         largest= (*it)->layer();
      }
   }

   if(largest != layer()) {
      m_layer = largest+1;
   }

   m_itemInfos.clear();
   m_accessor->reset();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   changeManager()->changed(this);
}
void
FWEventItem::moveToBack()
{
   assert(0!=m_context->eventItemsManager());
   int smallest = layer();
   for(FWEventItemsManager::const_iterator it = m_context->eventItemsManager()->begin(),
                                           itEnd = m_context->eventItemsManager()->end();
       it != itEnd;
       ++it) {
      if( (*it) && (*it)->layer() < smallest) {
         smallest= (*it)->layer();
      }
   }

   if(smallest != layer()) {
      m_layer = smallest-1;
   }
   FWChangeSentry sentry(*(this->changeManager()));

   m_itemInfos.clear();
   m_accessor->reset();
   preItemChanged_(this);
   //want filter to rerun after all changes have been made
   m_shouldFilterConnection.block(false);
   changeManager()->changed(this);
}

//
// const member functions
//
const void*
FWEventItem::data(const std::type_info& iInfo) const
{
   using namespace Reflex;
   //At the moment this is a programming error
   assert(iInfo == *(m_type->GetTypeInfo()) );

   //lookup data if we don't already have it
   if(0==m_accessor->data()) {
      void* wrapper=0;
      void* temp = &wrapper;
      if(m_event) {
         try {
            m_event->getByLabel(m_wrapperType.TypeInfo(),
                                m_moduleLabel.c_str(),
                                m_productInstanceLabel.c_str(),
                                m_processName.size() ? m_processName.c_str() : 0,
                                temp);
         } catch (std::exception& iException) {
            if ( !m_printedNoDataError ) {
               std::cerr << "Failed to get "<<name()<<" because \n" <<iException.what()<<std::endl;
               m_printedNoDataError = true;
            }
            return 0;
         }
         if(0==wrapper) {
            if ( !m_printedNoDataError ) {
               std::cerr << "Failed to get "<<name()<<" because branch does not exist in this file"<<std::endl;
               m_printedNoDataError = true;
            }
            return 0;
         }
//       printf("%s: wrapper address: 0x%x 0x%x 0x%x\n", name().c_str(), wrapper, &wrapper, *(int *)wrapper);
         std::string fullbranch_classname = (edm::TypeID(iInfo)).friendlyClassName();
         std::string fullbranch_module, fullbranch_product, fullbranch_process;
         for (fwlite::Event::KeyToDataMap::const_iterator i = m_event->data_.begin(); i != m_event->data_.end(); ++i) {
            if (i->second->pObj_ != wrapper)
               continue;
            if (i->first.module_ != 0 && strlen(i->first.module_) > 0)
               fullbranch_module = i->first.module_;
            if (i->first.product_ != 0 && strlen(i->first.product_) > 0)
               fullbranch_product = i->first.product_;
            if (i->first.process_ != 0 && strlen(i->first.process_) > 0)
               fullbranch_process = i->first.process_;
         }
         m_fullBranchName  = fullbranch_classname + "_"; // the quoted separator
         m_fullBranchName += fullbranch_module + "_";   // is required, but
         m_fullBranchName += fullbranch_product + "_";  // looks so very Japanese
         m_fullBranchName += fullbranch_process;
//       printf("full branch name for event item %s is %s\n", name().c_str(), m_fullBranchName.c_str());

         //Get Reflex to do the work
         Object wrapperObj(m_wrapperType,wrapper);

         //Convert our wrapper to its EDProduct base class
         static Type s_edproductType(Type::ByTypeInfo(typeid(edm::EDProduct)));
         Object edproductObj(wrapperObj.CastObject(s_edproductType));
         const edm::EDProduct* prod = reinterpret_cast<const edm::EDProduct*>(edproductObj.Address());

         if(not prod->isPresent()) {
            //not actually in this event
            if(!m_printedErrorThisEvent) {
               std::cerr <<name()<<" is registered in the file but is unavailable for this event"<<std::endl;
               m_printedErrorThisEvent = true;
            }
            return 0;
         }

         setData(wrapperObj);
      }
   }
   //return m_data;
   return m_accessor->data();
}


void
FWEventItem::setData(const Reflex::Object& iData) const
{
   m_accessor->setWrapper(iData);
   //std::cout <<"size "<<m_accessor->size()<<std::endl;
   if(m_accessor->isCollection()) {
      m_itemInfos.reserve(m_accessor->size());
      m_itemInfos.resize(m_accessor->size(),ModelInfo(m_displayProperties,false));
   } else {
      m_itemInfos.push_back(ModelInfo(m_displayProperties,false));
   }
}

void
FWEventItem::getPrimaryData() const
{
   //if(0!=m_data) return;
   if(0!=m_accessor->data()) return;
   this->data(*(m_type->GetTypeInfo()));
}

const FWDisplayProperties&
FWEventItem::defaultDisplayProperties() const
{
   return m_displayProperties;
}

int
FWEventItem::layer() const
{
   return m_layer;
}

bool
FWEventItem::isInFront() const
{
   assert(0!=m_context->eventItemsManager());
   for(FWEventItemsManager::const_iterator it = m_context->eventItemsManager()->begin(),
                                           itEnd = m_context->eventItemsManager()->end();
       it != itEnd;
       ++it) {
      if((*it) && (*it)->layer() > layer()) {
         return false;
      }
   }
   return true;
}

bool
FWEventItem::isInBack() const
{
   assert(0!=m_context->eventItemsManager());
   for(FWEventItemsManager::const_iterator it = m_context->eventItemsManager()->begin(),
                                           itEnd = m_context->eventItemsManager()->end();
       it != itEnd;
       ++it) {
      if((*it) && (*it)->layer() < layer()) {
         return false;
      }
   }
   return true;
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
FWEventItem::purpose() const
{
   return m_purpose;
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

FWEventItem::ModelInfo
FWEventItem::modelInfo(int iIndex) const
{
   getPrimaryData();
   if(m_displayProperties.isVisible()) {
      return m_itemInfos.at(iIndex);
   }
   FWDisplayProperties dp(m_itemInfos.at(iIndex).displayProperties());
   dp.setIsVisible(false);
   ModelInfo t(dp,m_itemInfos.at(iIndex).isSelected());
   return t;
}

size_t
FWEventItem::size() const
{
   getPrimaryData();
   return m_itemInfos.size();
}

bool
FWEventItem::isCollection() const
{
   return m_accessor->isCollection();
}

const TClass*
FWEventItem::modelType() const
{
   return m_accessor->modelType();
}

const void*
FWEventItem::modelData(int iIndex) const
{
   getPrimaryData();
   return m_accessor->modelData(iIndex);
}

std::string
FWEventItem::modelName(int iIndex) const
{
   std::ostringstream s;
   size_t lastChar = name().size();
   //if name ends in 's' assume it is plural and remove the s for the individual object
   if(name()[lastChar-1]=='s') {
      --lastChar;
   }
   s<<name().substr(0,lastChar)<<" "<<iIndex;
   return s.str();
}

bool
FWEventItem::haveInterestingValue() const
{
   return m_interestingValueGetter.isValid();
}

double
FWEventItem::modelInterestingValue(int iIndex) const
{
   getPrimaryData();
   return m_interestingValueGetter.valueFor(m_accessor->modelData(iIndex));
}
std::string
FWEventItem::modelInterestingValueAsString(int iIndex) const
{
   getPrimaryData();
   return m_interestingValueGetter.stringValueFor(m_accessor->modelData(iIndex));
}


const std::string&
FWEventItem::filterExpression() const
{
   return m_filter.expression();
}

void
FWEventItem::destroy() const
{
   goingToBeDestroyed_(this);
   delete this;
}

//
// static member functions
//
