#ifndef Fireworks_Calo_FWPFCandTowerProxyBuilder_h
#define Fireworks_Calo_FWPFCandTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerProxyBuilderBase
//
/**\class FWCaloTowerProxyBuilderBase FWCaloTowerProxyBuilderBase.h Fireworks/Calo/interface/FWCaloTowerProxyBuilderBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:08 EST 2008
//

#include "Rtypes.h"
#include <string>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

class FWHistSliceSelector;

class FWPFCandidateTowerProxyBuilder : public FWCaloDataHistProxyBuilder
{
public:
   FWPFCandidateTowerProxyBuilder();
   ~FWPFCandidateTowerProxyBuilder() override;

   virtual double getEt(const reco::PFCandidate&) const = 0;

protected:
   void fillCaloData() override;
   FWHistSliceSelector* instantiateSliceSelector() override;
   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

private:
   FWPFCandidateTowerProxyBuilder(const FWPFCandidateTowerProxyBuilder&) = delete; // stop default
   const FWPFCandidateTowerProxyBuilder& operator=(const FWPFCandidateTowerProxyBuilder&) = delete; // stop default
  
   // ---------- member data --------------------------------
   const reco::PFCandidateCollection* m_towers;
};




//
// Ecal
//

class FWECalPFCandidateProxyBuilder : public FWPFCandidateTowerProxyBuilder {
public:
   FWECalPFCandidateProxyBuilder() {
   }
   ~FWECalPFCandidateProxyBuilder() override {
   }

   // ---------- const member functions ---------------------

   double getEt(const reco::PFCandidate& iTower) const override {
      return iTower.ecalEnergy()* TMath::Sin(iTower.theta());
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWECalPFCandidateProxyBuilder(const FWECalPFCandidateProxyBuilder&) = delete; // stop default
   const FWECalPFCandidateProxyBuilder& operator=(const FWECalPFCandidateProxyBuilder&) = delete; // stop default
};


//
// Hcal
//

class FWHCalPFCandidateProxyBuilder : public FWPFCandidateTowerProxyBuilder {
public:
   FWHCalPFCandidateProxyBuilder() {
   }
   ~FWHCalPFCandidateProxyBuilder() override{
   }

   // ---------- const member functions ---------------------

   double getEt(const reco::PFCandidate& iTower) const override {
      return iTower.hcalEnergy() * TMath::Sin(iTower.theta());
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWHCalPFCandidateProxyBuilder(const FWHCalPFCandidateProxyBuilder&) = delete; // stop default

   const FWHCalPFCandidateProxyBuilder& operator=(const FWHCalPFCandidateProxyBuilder&) = delete; // stop default
};

#endif
