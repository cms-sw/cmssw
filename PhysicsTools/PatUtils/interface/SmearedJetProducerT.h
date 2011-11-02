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
 * \version $Revision: 1.3 $
 *
 * $Id: SmearedJetProducerT.h,v 1.3 2011/11/01 14:11:55 veelken Exp $
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

    for ( typename JetCollection::const_iterator jet = jets->begin();
	  jet != jets->end(); ++jet ) {
      reco::Candidate::LorentzVector jetP4 = jet->p4();

      //std::cout << "jet: E = " << jet->energy() << "," 
      //	  << " px = " << jet->px() << ", py = " << jet->py() << ", pz = " << jet->pz() << std::endl;
      
      double smearFactor = 1.;
      
      double x = TMath::Abs(jetP4.eta());
      double y = jetP4.pt();
      if ( x > lut_->GetXaxis()->GetXmin() && x < lut_->GetXaxis()->GetXmax() &&
	   y > lut_->GetYaxis()->GetXmin() && y < lut_->GetYaxis()->GetXmax() ) {
	int binIndex = lut_->FindBin(x, y);

	smearFactor = lut_->GetBinContent(binIndex);
	double smearFactorErr = lut_->GetBinError(binIndex);

	//std::cout << "x = " << x << ", y = " << y << ": smearFactor = " << smearFactor << " +/- " << smearFactorErr << std::endl;

	if ( shiftBy_ != 0. ) smearFactor += (shiftBy_*smearFactorErr);
	smearFactor = TMath::Power(smearFactor, smearBy_);
	//std::cout << "smearBy = " << smearBy_ << " --> final smearFactor = " << smearFactor << std::endl;
      }

      double smearedJetEn = jet->energy();

      const reco::GenJet* genJet = genJetMatcher_(*jet, &evt);
      if ( genJet ) { 
//--- case 1: reconstructed jet matched to generator level jet, 
//            smear difference between reconstructed and "true" jet energy

	double dEn = jet->energy() - genJet->energy();
	//std::cout << " case 1: dEn = " << dEn << std::endl;

	smearedJetEn = genJet->energy() + smearFactor*dEn;
      } else {
//--- case 2: reconstructed jet **not** matched to generator level jet, 
//            smear jet energy using MC resolution functions implemented in PFMEt significance algorithm (CMS AN-10/400)

	if ( smearFactor > 1. ) {
	  // CV: MC resolution already accounted for in reconstructed jet,
	  //     add additional Gaussian smearing of width = sqrt(smearFactor^2 - 1) 
	  //     to account for Data/MC **difference** in jet resolutions
	  double sigmaEn = jetResolutionExtractor_(*jet)*TMath::Sqrt(smearFactor*smearFactor - 1.);
	  //std::cout << " case 2: sigmaEn = " << sigmaEn << std::endl;

	  smearedJetEn = jet->energy() + rnd_.Gaus(0., sigmaEn);
	}
      }
	  
      // CV: keep minimum jet energy, in order not to loose direction information
      const double minJetEn = 1.e-2; 
      if ( smearedJetEn < minJetEn ) smearedJetEn = minJetEn;
	  
      reco::Candidate::LorentzVector smearedJetP4 = jetP4;
      smearedJetP4 *= (smearedJetEn/jet->energy());
	  
      T smearedJet = (*jet);
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

  std::string jetCorrLabel_;

  TFile* inputFile_;
  TH2* lut_;

  SmearedJetProducer_namespace::JetResolutionExtractorT<T> jetResolutionExtractor_;
  TRandom3 rnd_;

  double smearBy_; // option to "smear" jet energy by N standard-deviations, useful for template morphing

  double shiftBy_; // option to increase/decrease within uncertainties the jet energy resolution used for smearing 
};

#endif
