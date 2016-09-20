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
//

// system include files
#include <iostream>
#include <algorithm>
#include <exception>
#include <TClass.h>

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/FWLite/interface/Event.h"
// Needed to test edm::Event access
// #include "FWCore/Framework/interface/Event.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWItemAccessorBase.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/src/FWGenericHandle.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

//
// static data member definitions
//

int FWEventItem::minLayerValue()
{
   return -100;
}

int FWEventItem::maxLayerValue()
{
   return 100;
}


//
// constructors and destructor
//
FWEventItem::FWEventItem(fireworks::Context* iContext,
                         unsigned int iId,
                         boost::shared_ptr<FWItemAccessorBase> iAccessor,
                         const FWPhysicsObjectDesc& iDesc,  const FWConfiguration* pbc) :
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
   m_interestingValueGetter(edm::TypeWithDict(*(m_accessor->modelType()->GetTypeInfo())), m_purpose),
   m_filter(iDesc.filterExpression(),""),
   m_printedErrorThisEvent(false),
   m_isSelected(false),
   m_origColor(0),
   m_proxyBuilderConfig(0)
{
   //assert(m_type->GetTypeInfo());
   //edm::TypeWithDict dataType(*(m_type->GetTypeInfo()));
   //assert(dataType != edm::TypeWithDict() );
   //
   //std::string dataTypeName = dataType.name();
   //if (dataTypeName[dataTypeName.size() -1] == '>')
   //   dataTypeName += " ";
   //std::string wrapperName = "edm::Wrapper<" + dataTypeName + ">";
   //
   //fwLog(fwlog::kDebug) << "Looking for the wrapper name" 
   //                    << wrapperName << std::endl;
   //m_wrapperType = edm::TypeWithDict::byName(wrapperName);
   //
   //assert(m_wrapperType != edm::TypeWithDict());
   if(!m_accessor->isCollection()) {
      m_itemInfos.reserve(1);
   }
   m_filter.setClassName(modelType()->GetName());
   m_proxyBuilderConfig = new FWProxyBuilderConfiguration(pbc, this);
   m_origColor = iDesc.displayProperties().color();
}
// FWEventItem::FWEventItem(const FWEventItem& rhs)
// {
//    // do actual copying here;
// }

FWEventItem::~FWEventItem()
{
  delete m_proxyBuilderConfig;
}
 
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
FWEventItem::setEvent(const edm::EventBase* iEvent)
{
   m_printedErrorThisEvent = false;
   m_event = iEvent;
   m_accessor->reset();
   m_itemInfos.clear();
   handleChange();
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
   handleChange();
}

void
FWEventItem::setName(const std::string& iName)
{
   m_name = iName;
}

/** This is the place where not only display properties are changed,
    but which is also responsible to notify the FWModelChangeManager about
    the change. If you've just added some property, you have a nice GUI for
    it and still nothing works, this is probably the place where you want to 
    look. 
  */
void
FWEventItem::setDefaultDisplayProperties(const FWDisplayProperties& iProp)
{
   bool visChange = m_displayProperties.isVisible() != iProp.isVisible();
   bool colorChanged = m_displayProperties.color() != iProp.color();
   bool transparencyChanged = m_displayProperties.transparency() != iProp.transparency();

   if(!visChange && !colorChanged && !transparencyChanged) {
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
            changed = true;
         }
      }
      if (transparencyChanged) {
         if(m_displayProperties.transparency() == prp.transparency()) {
            prp.setTransparency(iProp.transparency());
            changed = true;
         }
      }
      if(changed) {
         m_itemInfos[index].m_displayProperties=prp;
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
      //want to make it obvious what type of object was selected
      // therefore we also select the item
      const_cast<FWEventItem*>(this)->selectItem();
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
      if ((*it) && (*it != this) && (*it)->layer() > largest) {
         largest= (*it)->layer();
      }
   }

   if(largest >= layer()) {
      m_layer = std::min(largest+1, maxLayerValue());
   }

   m_itemInfos.clear();
   m_accessor->reset();
   handleChange();
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
      if((*it) && (*it != this) && (*it)->layer() < smallest) {
         smallest= (*it)->layer();
      }
   }

   if(smallest <= layer()) {
      m_layer = std::max(smallest-1, minLayerValue());
   }

   m_itemInfos.clear();
   m_accessor->reset();
   handleChange();
}

void
FWEventItem::moveToLayer(int layer)
{
   assert(0!=m_context->eventItemsManager());

   m_layer = std::max(std::min(layer, maxLayerValue()), minLayerValue());

   m_itemInfos.clear();
   m_accessor->reset();
   handleChange();
}

void
FWEventItem::proxyConfigChanged()
{
   m_itemInfos.clear();
   m_accessor->reset();
   if (m_context->eventItemsManager())
      handleChange();

}

void 
FWEventItem::handleChange()
{
   preItemChanged_(this);
   FWChangeSentry sentry(*(this->changeManager()));
   //want filter to rerun after all changes have been made
   changeManager()->changed(this);
   getPrimaryData();
   runFilter();
}

//
// const member functions
//
const void*
FWEventItem::data(const std::type_info& iInfo) const
{
   //At the moment this is a programming error
   assert(iInfo == *(m_type->GetTypeInfo()));

   //lookup data if we don't already have it
   if (m_accessor->data())
      return m_accessor->data();

   m_errorMessage.clear();
   if (!m_event)
      return m_accessor->data();
   
   // Retrieve the data from the event.
   edm::InputTag tag(m_moduleLabel, m_productInstanceLabel, m_processName);
   edm::TypeWithDict type(iInfo);
   edm::FWGenericHandle handle(type);
   try
   {
      m_event->getByLabel(tag, handle);
      setData(*handle);
   }
   catch (std::exception& iException)
   {
      if (!m_printedErrorThisEvent) 
      {
         std::ostringstream s;
         s << "Failed to get " << name() << " because \n" <<iException.what();
         m_errorMessage=s.str();
         m_printedErrorThisEvent = true;
      }
      return 0;
   }
   
   return m_accessor->data();
}

void
FWEventItem::setData(const edm::ObjectWithDict& iData) const
{
   m_accessor->setData(iData);
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
      if((*it) && (*it != this) && (*it)->layer() >= layer()) {
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
      if((*it) && (*it != this) && (*it)->layer() <= layer()) {
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
   return true; //m_interestingValueGetter.isValid();
}

 
const std::string&
FWEventItem::modelInterestingValueAsString(int iIndex) const
{
   getPrimaryData();
   return m_interestingValueGetter.getToolTip(m_accessor->modelData(iIndex));
}


const std::string&
FWEventItem::filterExpression() const
{
   return m_filter.expression();
}

void
FWEventItem::destroy() const
{
   //NOTE: need to unselect first before announcing destruction
   // because some items are listening to the display change and may
   // not properly release their connection to that signal after they
   // are destroyed via a connection to goingToBeDestroyed_
   const_cast<FWEventItem*>(this)->unselectItem();
   {
      FWChangeSentry sentry(*(changeManager()));
   
      for(int index=0; index <static_cast<int>(size()); ++index) {
         if(m_itemInfos.at(index).m_isSelected) {
            FWModelId id(this,index);
            selectionManager()->unselect(id);
            changeManager()->changed(id);
         }
      }
   }
   goingToBeDestroyed_(this);
   delete this;
}


void
FWEventItem::selectItem()
{
   if(!m_isSelected) {
      m_isSelected=true;
      selectionManager()->selectItem(this);
      defaultDisplayPropertiesChanged_(this);
   }
}
void
FWEventItem::unselectItem()
{
   if(m_isSelected) {
      m_isSelected=false;
      selectionManager()->unselectItem(this);
      defaultDisplayPropertiesChanged_(this);
   }
}
void
FWEventItem::toggleSelectItem()
{
   m_isSelected = !m_isSelected;
   if(m_isSelected) {
      selectionManager()->selectItem(this);
   }else {
      selectionManager()->unselectItem(this);
   }
   defaultDisplayPropertiesChanged_(this);
}
bool
FWEventItem::itemIsSelected() const
{
   return m_isSelected;
}

bool
FWEventItem::hasError() const {
   return !errorMessage().empty();
}

const std::string&
FWEventItem::errorMessage() const
{
   if(m_errorMessage.empty()) {
      getPrimaryData();
   }
   return m_errorMessage;
}

const FWGeometry* 
FWEventItem::getGeom() const {
   return m_context->getGeom();
}

void FWEventItem::resetColor()
{ 
   m_displayProperties.setColor(m_origColor);

   FWChangeSentry sentry(*(this->changeManager()));
   for(int index=0; index <static_cast<int>(size()); ++index) {
      m_itemInfos.at(index).m_displayProperties.setColor(m_origColor);
      FWModelId id(this,index);
      changeManager()->changed(id);
   }

   defaultDisplayPropertiesChanged_(this);
}
//
// static member functions
//
