#ifndef RecoHIHiHiJetAlgosHiGenCleaner_H
#define RecoHIHiHiJetAlgosHiGenCleaner_H
// -*- C++ -*-
//
// Package:    HiGenCleaner
// Class:      HiGenCleaner
//
/**\class HiGenCleaner HiGenCleaner.cc yetkin/HiGenCleaner/src/HiGenCleaner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Jul 21 04:26:01 EDT 2009
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"

//
// class decleration
//

template <class T2>
class HiGenCleaner : public edm::global::EDProducer<> {
public:
  typedef std::vector<T2> T2Collection;
  explicit HiGenCleaner(const edm::ParameterSet&);
  ~HiGenCleaner() override;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::View<T2> > jetSrc_;
  const double deltaR_;
  const double ptCut_;
  const bool makeNew_;
  const bool fillDummy_;
};

#endif
