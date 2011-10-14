#ifndef PhysicsTools_PatUtils_SmearedJetProducerT_h
#define PhysicsTools_PatUtils_SmearedJetProducerT_h

/** \class SmearedJetProducerT
 *
 * Produce collection of "smeared" jets.
 * The aim of this correction is to account for the difference in jet energy resolution
 * between Monte Carlo simulation and Data. 
 * The jet energy resolutions have been measured in QCD di-jet and gamma + jets events selected in 2010 data,
 * as documented in the PAS JME-10-014.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SmearedJetProducerT.h,v 1.1 2011/09/30 12:29:05 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/CandidateTools/interface/generalAuxFunctions.h"

#include <TMath.h>

#include <TFile.h>
#include <TH2.h>

namespace SmearedJetProducer_namespace
{
  template <typename T>
  class GenJetMatcherT
  {
    public:

     GenJetMatcherT(const edm::ParameterSet& cfg) 
       : srcGenJets_(cfg.getParameter<edm::InputTag>("srcGenJets")),
         dRmaxGenJetMatch_(cfg.getParameter<double>("dRmaxGenJetMatch"))
     {}
     ~GenJetMatcherT() {}

     const reco::GenJet* operator()(const T& jet, edm::Event* evt = 0) const
     {
       assert(evt);
       
       edm::Handle<reco::GenJetCollection> genJets;
       evt->getByLabel(srcGenJets_, genJets);

       const reco::GenJet* retVal = 0;

       double dRbestMatch = dRmaxGenJetMatch_;
       for ( reco::GenJetCollection::const_iterator genJet = genJets->begin();
	     genJet != genJets->end(); ++genJet ) {
	 double dR = deltaR(jet.p4(), genJet->p4());
	 if ( dR < dRbestMatch ) {
	   retVal = &(*genJet);
	   dRbestMatch = dR;
	 }
       }

       return retVal;
     }

    private:

//--- configuration parameter
     edm::InputTag srcGenJets_;

     double dRmaxGenJetMatch_;
  };
}

template <typename T>
class SmearedJetProducerT : public edm::EDProducer 
{
  typedef std::vector<T> JetCollection;

 public:

  explicit SmearedJetProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      genJetMatcher_(cfg)
  {
    src_ = cfg.getParameter<edm::InputTag>("src");

    edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
    std::string lutName = cfg.getParameter<std::string>("lutName");
    if ( !inputFileName.isLocal() ) 
      throw cms::Exception("JetMETsmearInputProducer") 
        << " Failed to find File = " << inputFileName << " !!\n";

    inputFile_ = new TFile(inputFileName.fullPath().data());
    lut_ = dynamic_cast<TH2*>(inputFile_->Get(lutName.data()));
    if ( !lut_ ) 
      throw cms::Exception("SmearedJetProducer") 
        << " Failed to load LUT = " << lutName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

    smearBy_ = ( cfg.exists("smearBy") ) ? cfg.getParameter<double>("smearBy") : 1.0;

    produces<JetCollection>();
  }
  ~SmearedJetProducerT()
  {
    // nothing to be done yet...
  }
    
 private:

  virtual void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    //std::cout << "<SmearedJetProducer::produce>:" << std::endl;
    //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;

    std::auto_ptr<JetCollection> smearedJets(new JetCollection);
    
    edm::Handle<JetCollection> jets;
    evt.getByLabel(src_, jets);

    for ( typename JetCollection::const_iterator jet = jets->begin();
	  jet != jets->end(); ++jet ) {
      reco::Candidate::LorentzVector jetP4 = jet->p4();

      T smearedJet = (*jet);

      const reco::GenJet* genJet = genJetMatcher_(*jet, &evt);
      if ( genJet ) {
	int binIndex = lut_->FindBin(TMath::Abs(jetP4.eta()), jetP4.pt());
	double smearFactor = lut_->GetBinContent(binIndex);
	smearFactor = TMath::Power(smearFactor, smearBy_);

	reco::Candidate::LorentzVector smearP4 = jet->p4() - genJet->p4();
	smearP4 *= smearFactor;
	
	smearedJet.setP4(genJet->p4() + smearP4);
      }

      smearedJets->push_back(smearedJet);
    }

//--- add collection of "smeared" jets to the event
    evt.put(smearedJets);
  } 

  std::string moduleLabel_;
      
  SmearedJetProducer_namespace::GenJetMatcherT<T> genJetMatcher_;

//--- configuration parameters
 
  // collection of pat::Jets (with L2L3/L2L3Residual corrections applied)
  edm::InputTag src_;

  std::string jetCorrLabel_;

  TFile* inputFile_;
  TH2* lut_;

  double smearBy_; // option to "smear" jet energy by N standard-deviations, useful for template morphing
};

#endif
