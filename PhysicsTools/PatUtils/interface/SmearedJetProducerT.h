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
 * \version $Revision: 1.4 $
 *
 * $Id: SmearedJetProducerT.h,v 1.4 2011/11/02 14:03:07 veelken Exp $
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

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include <TFile.h>
#include <TH2.h>
#include <TMath.h>
#include <TRandom3.h>

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

  template <typename T>
  class JetResolutionExtractorT
  {
    public:

     JetResolutionExtractorT(const edm::ParameterSet&) {}
     ~JetResolutionExtractorT() {}

     double operator()(const T&) const
     {
       throw cms::Exception("SmearedJetProducer::produce")
	 << " Jets of type other than PF not supported yet !!\n";       
     }
  };

  template <typename T>
  class RawJetExtractorT // this template is neccessary to support pat::Jets
                         // (because pat::Jet->p4() returns the JES corrected, not the raw, jet momentum)
  {
    public:

     reco::Candidate::LorentzVector operator()(const T& jet) const 
     {        
       return jet.p4();
     } 
  };
}

template <typename T>
class SmearedJetProducerT : public edm::EDProducer 
{
  typedef std::vector<T> JetCollection;

 public:

  explicit SmearedJetProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      genJetMatcher_(cfg),
      jetResolutionExtractor_(cfg.getParameter<edm::ParameterSet>("jetResolutions"))
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

    jetCorrLabel_ = ( cfg.exists("jetCorrLabel") ) ?
      cfg.getParameter<std::string>("jetCorrLabel") : "";

    smearBy_ = ( cfg.exists("smearBy") ) ? cfg.getParameter<double>("smearBy") : 1.0;

    shiftBy_ = ( cfg.exists("shiftBy") ) ? cfg.getParameter<double>("shiftBy") : 0.;

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

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& jet = jets->at(jetIndex);

      //std::cout << "jet: E = " << jet.energy() << "," 
      //	  << " px = " << jet.px() << ", py = " << jet.py() << ", pz = " << jet.pz() << std::endl;
      
      reco::Candidate::LorentzVector corrJetP4 = jet.p4();
      if ( jetCorrLabel_ != "" ) {
	static SmearedJetProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
	reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(jet);

	corrJetP4 = jetCorrExtractor_(jet, jetCorrLabel_, &evt, &es, jetCorrEtaMax_, &rawJetP4);

	//std::cout << "corrJet: E = " << corrJetP4.E() << "," 
	//	    << " px = " << corrJetP4.px() << ", py = " << corrJetP4.py() << ", pz = " << corrJetP4.pz() << std::endl;
      }

      double smearFactor = 1.;
      
      double x = TMath::Abs(corrJetP4.eta());
      double y = corrJetP4.pt();
      if ( x > lut_->GetXaxis()->GetXmin() && x < lut_->GetXaxis()->GetXmax() &&
	   y > lut_->GetYaxis()->GetXmin() && y < lut_->GetYaxis()->GetXmax() ) {
	int binIndex = lut_->FindBin(x, y);
	
	if ( smearBy_ > 0. ) smearFactor += smearBy_*(lut_->GetBinContent(binIndex) - 1.);
	double smearFactorErr = lut_->GetBinError(binIndex);
	//std::cout << "x = " << x << ", y = " << y << ":" 
	//          << " smearFactor = " << smearFactor << " +/- " << smearFactorErr << std::endl;

	if ( shiftBy_ != 0. ) smearFactor += (shiftBy_*smearFactorErr);
	//std::cout << "smearBy = " << smearBy_ << ", shiftBy = " << shiftBy_ 
	//          << " --> final smearFactor = " << smearFactor << std::endl;
      }

      double smearedJetEn = corrJetP4.E();

      const reco::GenJet* genJet = genJetMatcher_(jet, &evt);
      if ( genJet ) { 
//--- case 1: reconstructed jet matched to generator level jet, 
//            smear difference between reconstructed and "true" jet energy

	double dEn = corrJetP4.E() - genJet->energy();
	//std::cout << " case 1: dEn = " << dEn << std::endl;

	smearedJetEn = genJet->energy() + smearFactor*dEn;
      } else {
//--- case 2: reconstructed jet **not** matched to generator level jet, 
//            smear jet energy using MC resolution functions implemented in PFMEt significance algorithm (CMS AN-10/400)

	if ( smearFactor > 1. ) {
	  // CV: MC resolution already accounted for in reconstructed jet,
	  //     add additional Gaussian smearing of width = sqrt(smearFactor^2 - 1) 
	  //     to account for Data/MC **difference** in jet resolutions
	  double sigmaEn = jetResolutionExtractor_(jet)*TMath::Sqrt(smearFactor*smearFactor - 1.);
	  //std::cout << " case 2: sigmaEn = " << sigmaEn << std::endl;

	  smearedJetEn = corrJetP4.E() + rnd_.Gaus(0., sigmaEn);
	}
      }
	  
      // CV: keep minimum jet energy, in order not to loose direction information
      const double minJetEn = 1.e-2; 
      if ( smearedJetEn < minJetEn ) smearedJetEn = minJetEn;
	  
      reco::Candidate::LorentzVector smearedJetP4 = jet.p4();
      smearedJetP4 *= (smearedJetEn/corrJetP4.E());
	  
      T smearedJet = (jet);
      smearedJet.setP4(smearedJetP4);
      
      //std::cout << "smearedJet: E = " << smearedJet.energy() << "," 
      //	  << " px = " << smearedJet.px() << ", py = " << smearedJet.py() << ", pz = " << smearedJet.pz() << std::endl;
      
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

  TFile* inputFile_;
  TH2* lut_;

  SmearedJetProducer_namespace::JetResolutionExtractorT<T> jetResolutionExtractor_;
  TRandom3 rnd_;

  std::string jetCorrLabel_; // e.g. 'ak5PFJetL1FastL2L3' (reco::PFJets) / '' (pat::Jets)
  JetCorrExtractorT<T> jetCorrExtractor_;

  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html

  double smearBy_; // option to "smear" jet energy by N standard-deviations, useful for template morphing

  double shiftBy_; // option to increase/decrease within uncertainties the jet energy resolution used for smearing 
};

#endif
