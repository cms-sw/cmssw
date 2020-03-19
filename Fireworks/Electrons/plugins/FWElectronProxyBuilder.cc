// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
//
#include "TEveCompound.h"
#include "TEveTrack.h"
#include "TEveScalableStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"

////////////////////////////////////////////////////////////////////////////////
//
//   3D and RPZ proxy builder with shared track list
//
////////////////////////////////////////////////////////////////////////////////

class FWElectronProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {
public:
  FWElectronProxyBuilder();
  ~FWElectronProxyBuilder() override;

  using FWProxyBuilderBase::haveSingleProduct;
  bool haveSingleProduct() const override { return false; }
  using FWProxyBuilderBase::cleanLocal;
  void cleanLocal() override;
  using FWSimpleProxyBuilderTemplate<reco::GsfElectron>::buildViewType;
  void buildViewType(const reco::GsfElectron& iData,
                     unsigned int iIndex,
                     TEveElement& oItemHolder,
                     FWViewType::EType type,
                     const FWViewContext*) override;

  using FWSimpleProxyBuilderTemplate<reco::GsfElectron>::setItem;
  void setItem(const FWEventItem* iItem) override;

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWElectronProxyBuilder(const FWElectronProxyBuilder&) = delete;                   // stop default
  const FWElectronProxyBuilder& operator=(const FWElectronProxyBuilder&) = delete;  // stop default

  TEveElementList* requestCommon();

  TEveElementList* m_common;
};

FWElectronProxyBuilder::FWElectronProxyBuilder() : m_common(nullptr) {
  m_common = new TEveElementList("common electron scene");
  m_common->IncDenyDestroy();
}

FWElectronProxyBuilder::~FWElectronProxyBuilder() { m_common->DecDenyDestroy(); }

void FWElectronProxyBuilder::setItem(const FWEventItem* iItem) {
  FWProxyBuilderBase::setItem(iItem);

  if (iItem) {
    iItem->getConfig()->assertParam("LineWidth", long(1), long(1), long(4));
  }
}

TEveElementList* FWElectronProxyBuilder::requestCommon() {
  if (m_common->HasChildren() == false) {
    int width = item()->getConfig()->value<long>("LineWidth");
    for (int i = 0; i < static_cast<int>(item()->size()); ++i) {
      const reco::GsfElectron& electron = modelData(i);

      TEveTrack* track(nullptr);
      if (electron.gsfTrack().isAvailable())
        track = fireworks::prepareTrack(*electron.gsfTrack(), context().getTrackPropagator());
      else
        track = fireworks::prepareCandidate(electron, context().getTrackPropagator());
      track->MakeTrack();
      track->SetLineWidth(width);
      setupElement(track);
      m_common->AddElement(track);
    }
  }
  return m_common;
}

void FWElectronProxyBuilder::cleanLocal() { m_common->DestroyElements(); }

void FWElectronProxyBuilder::buildViewType(const reco::GsfElectron& electron,
                                           unsigned int iIndex,
                                           TEveElement& oItemHolder,
                                           FWViewType::EType type,
                                           const FWViewContext*) {
  TEveElementList* tracks = requestCommon();
  TEveElement::List_i trkIt = tracks->BeginChildren();
  std::advance(trkIt, iIndex);
  setupAddElement(*trkIt, &oItemHolder);

  if (type == FWViewType::kRhoPhi || type == FWViewType::kRhoPhiPF)
    fireworks::makeRhoPhiSuperCluster(this, electron.superCluster(), electron.phi(), oItemHolder);
  else if (type == FWViewType::kRhoZ)
    fireworks::makeRhoZSuperCluster(this, electron.superCluster(), electron.phi(), oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWElectronProxyBuilder,
                        reco::GsfElectron,
                        "Electrons",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

////////////////////////////////////////////////////////////////////////////////
//
//   GLIMPSE specific proxy builder
//
////////////////////////////////////////////////////////////////////////////////

class FWElectronGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {
public:
  FWElectronGlimpseProxyBuilder() {}
  ~FWElectronGlimpseProxyBuilder() override {}
  using FWSimpleProxyBuilderTemplate<reco::GsfElectron>::build;

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder&) = delete;  // stop default

  const FWElectronGlimpseProxyBuilder& operator=(const FWElectronGlimpseProxyBuilder&) = delete;  // stop default

  void build(const reco::GsfElectron& iData,
             unsigned int iIndex,
             TEveElement& oItemHolder,
             const FWViewContext*) override;
};

void FWElectronGlimpseProxyBuilder::build(const reco::GsfElectron& iData,
                                          unsigned int iIndex,
                                          TEveElement& oItemHolder,
                                          const FWViewContext*) {
  TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("", "");
  marker->SetLineWidth(2);
  fireworks::addStraightLineSegment(marker, &iData, 1.0);
  setupAddElement(marker, &oItemHolder);
  //add to scaler at end so that it can scale the line after all ends have been added
  // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate: the scaler is not set!
  //    assert(scaler());
  //    scaler()->addElement(marker);
}
REGISTER_FWPROXYBUILDER(FWElectronGlimpseProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kGlimpseBit);
