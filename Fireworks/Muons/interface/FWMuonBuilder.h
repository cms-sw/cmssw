#ifndef Fireworks_Muons_FWMuonBuilder_h
#define Fireworks_Muons_FWMuonBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonBuilder
//
//
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
namespace reco {
  class Muon;
}

class FWEventItem;
class TEveElementList;
class TEveTrackPropagator;
class FWMagField;
class FWProxyBuilderBase;

class FWMuonBuilder {
public:
  FWMuonBuilder();
  virtual ~FWMuonBuilder();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void buildMuon(
      FWProxyBuilderBase*, const reco::Muon* muon, TEveElement* tList, bool showEndcap, bool onlyTracks = false);

  void setLineWidth(int w) { m_lineWidth = w; }

  FWMuonBuilder(const FWMuonBuilder&) = delete;  // stop default

  const FWMuonBuilder& operator=(const FWMuonBuilder&) = delete;  // stop default

private:
  void calculateField(const reco::Muon& iData, FWMagField* field);

  // ---------- member data --------------------------------
  int m_lineWidth;
};

#endif
