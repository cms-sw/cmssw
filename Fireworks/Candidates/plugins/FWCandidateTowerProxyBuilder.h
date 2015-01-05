#ifndef Fireworks_Calo_FWCandTowerProxyBuilder_h
#define Fireworks_Calo_FWCandTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
#include "Rtypes.h"
#include <string>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
//#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

class FWHistSliceSelector;

class FWCandidateTowerProxyBuilder : public FWCaloDataHistProxyBuilder
{
public:
   FWCandidateTowerProxyBuilder();
   virtual ~FWCandidateTowerProxyBuilder();

   virtual double getEt(const reco::Candidate& cand) const { return cand.pt(); }

   REGISTER_PROXYBUILDER_METHODS();
protected:
   virtual void fillCaloData();
   virtual FWHistSliceSelector* instantiateSliceSelector();
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

private:
   FWCandidateTowerProxyBuilder(const FWCandidateTowerProxyBuilder&); // stop default
   const FWCandidateTowerProxyBuilder& operator=(const FWCandidateTowerProxyBuilder&); // stop default
  
   // ---------- member data --------------------------------
   const pat::PackedCandidateCollection* m_towers;
};



/*
//
// Ecal
//

class FWECalCandidateProxyBuilder : public FWCandidateTowerProxyBuilder {
public:
   FWECalCandidateProxyBuilder() {
   }
   virtual ~FWECalCandidateProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   virtual double getEt(const reco::Candidate& iTower) const;

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWECalCandidateProxyBuilder(const FWECalCandidateProxyBuilder&); // stop default
   const FWECalCandidateProxyBuilder& operator=(const FWECalCandidateProxyBuilder&); // stop default
};


//
// Hcal
//

class FWHCalCandidateProxyBuilder : public FWCandidateTowerProxyBuilder {
public:
   FWHCalCandidateProxyBuilder() {
   }
   virtual ~FWHCalCandidateProxyBuilder(){
   }

   // ---------- const member functions ---------------------

   virtual double getEt(const reco::Candidate& iTower) const;

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWHCalCandidateProxyBuilder(const FWHCalCandidateProxyBuilder&); // stop default

   const FWHCalCandidateProxyBuilder& operator=(const FWHCalCandidateProxyBuilder&); // stop default
};
*/

#endif
