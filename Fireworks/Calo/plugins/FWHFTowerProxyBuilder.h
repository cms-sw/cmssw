#ifndef Fireworks_Calo_FWHFTowerProxyBuilder_h
#define Fireworks_Calo_FWHFTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHFTowerProxyBuilder
// 
/**\class FWHFTowerProxyBuilder FWHFTowerProxyBuilder.h Fireworks/Calo/interface/FWHFTowerProxyBuilder.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon May 31 16:41:23 CEST 2010
// $Id: FWHFTowerProxyBuilder.h,v 1.1 2010/05/31 15:35:00 amraktad Exp $
//

// system include files

// user include files

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// base
//
class FWHFTowerProxyBuilder : public FWCaloDataHistProxyBuilder
{
public:
   FWHFTowerProxyBuilder();
   virtual ~FWHFTowerProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   virtual void setCaloData(const fireworks::Context&);
   virtual void addSliceSelector();
   const HFRecHitCollection* m_hits;

private:

   FWHFTowerProxyBuilder(const FWHFTowerProxyBuilder&); // stop default

   const FWHFTowerProxyBuilder& operator=(const FWHFTowerProxyBuilder&); // stop default

   virtual void build(const FWEventItem* iItem,
                      TEveElementList* product, const FWViewContext*);
   // ---------- member data --------------------------------

};


//
// ShortFiber
//

class FWHFShortTowerProxyBuilder : public FWHFTowerProxyBuilder
{
public:
   FWHFShortTowerProxyBuilder() {
   }
   virtual ~FWHFShortTowerProxyBuilder() {
   }

   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "HFShort";
   }
   virtual void fillCaloData(); 

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWHFShortTowerProxyBuilder(const FWHFShortTowerProxyBuilder&); // stop default
   const FWHFShortTowerProxyBuilder& operator=(const FWHFShortTowerProxyBuilder&); // stop default
};


//
// LongFiber
//

class FWHFLongTowerProxyBuilder : public FWHFTowerProxyBuilder
{
public:
   FWHFLongTowerProxyBuilder() {
   }
   virtual ~FWHFLongTowerProxyBuilder(){
   }

   // ---------- const member functions ---------------------
   virtual const std::string histName() const {
      return "HFLong";
   }

   virtual void fillCaloData(); 

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWHFLongTowerProxyBuilder(const FWHFLongTowerProxyBuilder&); // stop default

   const FWHFLongTowerProxyBuilder& operator=(const FWHFLongTowerProxyBuilder&); // stop default
};

#endif
