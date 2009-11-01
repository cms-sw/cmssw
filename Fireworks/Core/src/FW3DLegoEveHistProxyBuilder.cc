// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoEveHistProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul  5 11:26:11 EDT 2008
// $Id: FW3DLegoEveHistProxyBuilder.cc,v 1.3 2008/11/06 22:05:24 amraktad Exp $
//

// system include files
#include "TEveCaloData.h"
#include "TH2F.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveHistProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoEveHistProxyBuilder::FW3DLegoEveHistProxyBuilder() :
   m_hist(0), m_data(0), m_sliceIndex(-1)
{
}

// FW3DLegoEveHistProxyBuilder::FW3DLegoEveHistProxyBuilder(const FW3DLegoEveHistProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoEveHistProxyBuilder::~FW3DLegoEveHistProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DLegoEveHistProxyBuilder& FW3DLegoEveHistProxyBuilder::operator=(const FW3DLegoEveHistProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoEveHistProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DLegoEveHistProxyBuilder::attach(TEveElement* iElement,
                                    TEveCaloDataHist* iHist)
{
   m_data = iHist;
}

void
FW3DLegoEveHistProxyBuilder::build()
{
   build(item(),&m_hist);
   if(0!=m_hist && -1 == m_sliceIndex) {
      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(item()->name().c_str(), 0., item()->defaultDisplayProperties().color());
   }
   m_data->DataChanged();
}

void
FW3DLegoEveHistProxyBuilder::modelChangesImp(const FWModelIds&)
{
   applyChangesToAllModels();
   m_data->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
   m_data->DataChanged();
}

void
FW3DLegoEveHistProxyBuilder::itemChangedImp(const FWEventItem*)
{

}

void
FW3DLegoEveHistProxyBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   m_hist->Reset();
   m_data->DataChanged();
}


//
// const member functions
//

//
// static member functions
//
