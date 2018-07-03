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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"

//
// class decleration
//

template <class T2>
class HiGenCleaner : public edm::EDProducer {
public:
  typedef std::vector<T2> T2Collection;
      explicit HiGenCleaner(const edm::ParameterSet&);
      ~HiGenCleaner() override;

   private:
      void produce(edm::Event&, const edm::EventSetup&) override;
      // ----------member data ---------------------------

   edm::EDGetTokenT<edm::View<T2> > jetSrc_;
   double deltaR_;
   double ptCut_;
   bool makeNew_;
   bool fillDummy_;

};


#endif
