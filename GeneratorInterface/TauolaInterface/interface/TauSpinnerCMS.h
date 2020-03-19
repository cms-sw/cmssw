// -*- C++ -*-
//
// Package: TauSpinnerInterface
// Class: TauSpinnerCMS
//
/**\class TauSpinnerCMS TauSpinnerCMS.cc

 */
//
// Original Author: Ian Nugent
// Created: Fri Feb 15 2013

#ifndef TauSpinnerCMS_h
#define TauSpinnerCMS_h

#include <iostream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
//#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "TauSpinner/SimpleParticle.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

class TauSpinnerCMS : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit TauSpinnerCMS(const edm::ParameterSet &);
  ~TauSpinnerCMS() override{};  // no need to delete ROOT stuff

  void produce(edm::Event &, const edm::EventSetup &) final;
  void beginJob() final;
  void endJob() final;
  static double flat();
  void setRandomEngine(CLHEP::HepRandomEngine *v) { fRandomEngine = v; }
  virtual void initialize();

private:
  bool isReco_;
  bool isTauolaConfigured_;
  bool isLHPDFConfigured_;
  std::string LHAPDFname_;
  double CMSEnergy_;
  edm::InputTag gensrc_;
  int MotherPDGID_, Ipol_, nonSM2_, nonSMN_;
  static bool isTauSpinnerConfigure;

  // Additional funtionms for Reco (not provided by Tauola/TauSpinner authors)
  int readParticlesfromReco(edm::Event &e,
                            TauSpinner::SimpleParticle &X,
                            TauSpinner::SimpleParticle &tau,
                            TauSpinner::SimpleParticle &tau2,
                            std::vector<TauSpinner::SimpleParticle> &tau_daughters,
                            std::vector<TauSpinner::SimpleParticle> &tau2_daughters);
  void GetLastSelf(const reco::GenParticle *Particle);
  void GetRecoDaughters(const reco::GenParticle *Particle,
                        std::vector<TauSpinner::SimpleParticle> &daughters,
                        int parentpdgid);
  bool isFirst(const reco::GenParticle *Particle);
  double roundOff_;

  static CLHEP::HepRandomEngine *fRandomEngine;
  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> GenParticleCollectionToken_;
  static bool fInitialized;
};
#endif
