#ifndef Fireworks_Calo_FWCandTowerProxyBuilder_h
#define Fireworks_Calo_FWCandTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
#include "Rtypes.h"
#include <string>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

class FWHistSliceSelector;

class FWCandidateTowerProxyBuilder : public FWCaloDataHistProxyBuilder
{
public:
   FWCandidateTowerProxyBuilder();
   virtual ~FWCandidateTowerProxyBuilder();
   static std::string typeOfBuilder() { return std::string("simple#");}
   virtual double getEt(const reco::Candidate& cand) const { return cand.pt(); }


   REGISTER_PROXYBUILDER_METHODS();
protected:
   virtual void fillCaloData();
   virtual FWHistSliceSelector* instantiateSliceSelector();
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);

private:
   FWCandidateTowerProxyBuilder(const FWCandidateTowerProxyBuilder&); // stop default
   const FWCandidateTowerProxyBuilder& operator=(const FWCandidateTowerProxyBuilder&); // stop default
  
   virtual void itemChangedImp(const FWEventItem*);
   // ---------- member data --------------------------------
   FWSimpleProxyHelper m_helper;
};

#endif
