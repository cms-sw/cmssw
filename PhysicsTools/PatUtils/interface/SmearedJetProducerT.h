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
 *
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <TFile.h>
#include <TFormula.h>
#include <TH2.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TString.h>

#include <iostream>
#include <iomanip>

namespace SmearedJetProducer_namespace
{
  template <typename T>
  class GenJetMatcherT
  {
    public:

     GenJetMatcherT(const edm::ParameterSet& cfg, edm::ConsumesCollector && iC)
       : srcGenJetsToken_(iC.consumes<reco::GenJetCollection>(cfg.getParameter<edm::InputTag>("srcGenJets"))),
         dRmaxGenJetMatch_(0)
     {
       TString dRmaxGenJetMatch_formula = cfg.getParameter<std::string>("dRmaxGenJetMatch").data();
       dRmaxGenJetMatch_formula.ReplaceAll("genJetPt", "x");
       dRmaxGenJetMatch_ = new TFormula("dRmaxGenJetMatch", dRmaxGenJetMatch_formula.Data());
     }
     ~GenJetMatcherT()
     {
       delete dRmaxGenJetMatch_;
     }

     const reco::GenJet* operator()(const T& jet, edm::Event* evt = 0) const
     {
       assert(evt);

       edm::Handle<reco::GenJetCollection> genJets;
       evt->getByToken(srcGenJetsToken_, genJets);

       const reco::GenJet* retVal = 0;

       double dRbestMatch = 1.e+6;
       for ( reco::GenJetCollection::const_iterator genJet = genJets->begin();
	     genJet != genJets->end(); ++genJet ) {
	 double dRmax = dRmaxGenJetMatch_->Eval(genJet->pt());
	 //std::cout << "genJetPt = " << genJet->pt() << ": dRmax = " << dRmax << std::endl;
	 double dR = deltaR(jet.p4(), genJet->p4());
	 if ( dR < dRbestMatch && dR < dRmax ) {
	   retVal = &(*genJet);
	   dRbestMatch = dR;
	 }
       }

       return retVal;
     }

    private:

//--- configuration parameter
     edm::InputTag srcGenJets_;
     edm::EDGetTokenT<reco::GenJetCollection> srcGenJetsToken_;

     TFormula* dRmaxGenJetMatch_;
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

  template <>
  class RawJetExtractorT<pat::Jet>
  {
    public:

     reco::Candidate::LorentzVector operator()(const pat::Jet& jet) const
     {
       if ( jet.jecSetsAvailable() ) return jet.correctedP4("Uncorrected");
       else return jet.p4();
     }
  };
}

template <typename T, typename Textractor>
class SmearedJetProducerT : public edm::EDProducer
{
  typedef std::vector<T> JetCollection;

 public:

  explicit SmearedJetProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      genJetMatcher_(cfg, consumesCollector()),
      jetResolutionExtractor_(cfg.getParameter<edm::ParameterSet>("jetResolutions")),
      skipJetSelection_(0)
  {
    //std::cout << "<SmearedJetProducer::SmearedJetProducer>:" << std::endl;
    //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;

    src_ = cfg.getParameter<edm::InputTag>("src");
    srcToken_ = consumes<JetCollection>(src_);

    edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
    std::string lutName = cfg.getParameter<std::string>("lutName");
    if (inputFileName.location() == edm::FileInPath::Unknown)
      throw cms::Exception("JetMETsmearInputProducer")
        << " Failed to find File = " << inputFileName << " !!\n";

    inputFile_ = new TFile(inputFileName.fullPath().data());
    lut_ = dynamic_cast<TH2*>(inputFile_->Get(lutName.data()));
    if ( !lut_ )
      throw cms::Exception("SmearedJetProducer")
        << " Failed to load LUT = " << lutName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

    jetCorrLabel_ = ( cfg.exists("jetCorrLabel") ) ?
      cfg.getParameter<std::string>("jetCorrLabel") : "";
    jetCorrEtaMax_ = ( cfg.exists("jetCorrEtaMax") ) ?
      cfg.getParameter<double>("jetCorrEtaMax") : 9.9;

    sigmaMaxGenJetMatch_ = cfg.getParameter<double>("sigmaMaxGenJetMatch");

    smearBy_ = ( cfg.exists("smearBy") ) ? cfg.getParameter<double>("smearBy") : 1.0;
    //std::cout << "smearBy = " << smearBy_ << std::endl;

    shiftBy_ = ( cfg.exists("shiftBy") ) ? cfg.getParameter<double>("shiftBy") : 0.;
    //std::cout << "shiftBy = " << shiftBy_ << std::endl;

    if ( cfg.exists("skipJetSelection") ) {
      std::string skipJetSelection_string = cfg.getParameter<std::string>("skipJetSelection");
      skipJetSelection_ = new StringCutObjectSelector<T>(skipJetSelection_string);
    }

    skipRawJetPtThreshold_  = ( cfg.exists("skipRawJetPtThreshold")  ) ?
      cfg.getParameter<double>("skipRawJetPtThreshold")  : 1.e-2;
    skipCorrJetPtThreshold_ = ( cfg.exists("skipCorrJetPtThreshold") ) ?
      cfg.getParameter<double>("skipCorrJetPtThreshold") : 1.e-2;

    verbosity_ = ( cfg.exists("verbosity") ) ?
      cfg.getParameter<int>("verbosity") : 0;

    produces<JetCollection>();
  }
  ~SmearedJetProducerT()
  {
    delete skipJetSelection_;
    delete inputFile_;
    delete lut_;
  }

 private:

  virtual void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    if ( verbosity_ ) {
      std::cout << "<SmearedJetProducerT::produce>:" << std::endl;
      std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
      std::cout << " src = " << src_.label() << std::endl;
    }

    std::auto_ptr<JetCollection> smearedJets(new JetCollection);

    edm::Handle<JetCollection> jets;
    evt.getByToken(srcToken_, jets);

    int numJets = jets->size();
    for ( int jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      const T& jet = jets->at(jetIndex);

      static const SmearedJetProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
      reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(jet);
      if ( verbosity_ ) {
	std::cout << "rawJet: Pt = " << rawJetP4.pt() << ", eta = " << rawJetP4.eta() << ", phi = " << rawJetP4.phi() << std::endl;
      }

      reco::Candidate::LorentzVector corrJetP4 = jet.p4();
      if ( jetCorrLabel_ != "" ) corrJetP4 = jetCorrExtractor_(jet, jetCorrLabel_, &evt, &es, jetCorrEtaMax_, &rawJetP4);
      if ( verbosity_ ) {
	std::cout << "corrJet: Pt = " << corrJetP4.pt() << ", eta = " << corrJetP4.eta() << ", phi = " << corrJetP4.phi() << std::endl;
      }

      double smearFactor = 1.;
      double x = TMath::Abs(corrJetP4.eta());
      double y = corrJetP4.pt();
      if ( x > lut_->GetXaxis()->GetXmin() && x < lut_->GetXaxis()->GetXmax() &&
	   y > lut_->GetYaxis()->GetXmin() && y < lut_->GetYaxis()->GetXmax() ) {
	int binIndex = lut_->FindBin(x, y);

	if ( smearBy_ > 0. ) smearFactor += smearBy_*(lut_->GetBinContent(binIndex) - 1.);
	double smearFactorErr = lut_->GetBinError(binIndex);
	if ( verbosity_ ) std::cout << "smearFactor = " << smearFactor << " +/- " << smearFactorErr << std::endl;

	if ( shiftBy_ != 0. ) {
	  smearFactor += (shiftBy_*smearFactorErr);
	  if ( verbosity_ ) std::cout << "smearFactor(shifted) = " << smearFactor << std::endl;
	}
      }

      double smearedJetEn = jet.energy();
      double sigmaEn = jetResolutionExtractor_(jet)*TMath::Sqrt(smearFactor*smearFactor - 1.);
      const reco::GenJet* genJet = genJetMatcher_(jet, &evt);
      bool isGenMatched = false;
      if ( genJet ) {
	if ( verbosity_ ) {
	  std::cout << "genJet: Pt = " << genJet->pt() << ", eta = " << genJet->eta() << ", phi = " << genJet->phi() << std::endl;
	}
	double dEn = corrJetP4.E() - genJet->energy();
	if ( dEn < (sigmaMaxGenJetMatch_*sigmaEn) ) {
//--- case 1: reconstructed jet matched to generator level jet,
//            smear difference between reconstructed and "true" jet energy

	  if ( verbosity_ ) {
	    std::cout << " successfully matched to genJet" << std::endl;
	    std::cout << "corrJetEn = " << corrJetP4.E() << ", genJetEn = " << genJet->energy() << " --> dEn = " << dEn << std::endl;
	  }

	  smearedJetEn = jet.energy()*(1. + (smearFactor - 1.)*dEn/TMath::Max(rawJetP4.E(), corrJetP4.E()));
	  isGenMatched = true;
	}
      }
      if ( !isGenMatched ) {
//--- case 2: reconstructed jet **not** matched to generator level jet,
//            smear jet energy using MC resolution functions implemented in PFMEt significance algorithm (CMS AN-10/400)

	if ( verbosity_ ) {
	  std::cout << " not matched to genJet" << std::endl;
	  std::cout << "corrJetEn = " << corrJetP4.E() << ", sigmaEn = " << sigmaEn << std::endl;
	}

	if ( smearFactor > 1. ) {
	  // CV: MC resolution already accounted for in reconstructed jet,
	  //     add additional Gaussian smearing of width = sqrt(smearFactor^2 - 1)
	  //     to account for Data/MC **difference** in jet resolutions.
	  //     Take maximum(rawJetEn, corrJetEn) to avoid pathological cases
	  //    (e.g. corrJetEn << rawJetEn, due to L1Fastjet corrections)

	  smearedJetEn = jet.energy()*(1. + rnd_.Gaus(0., sigmaEn)/TMath::Max(rawJetP4.E(), corrJetP4.E()));
	}
      }

      // CV: keep minimum jet energy, in order not to loose direction information
      const double minJetEn = 1.e-2;
      if ( smearedJetEn < minJetEn ) smearedJetEn = minJetEn;

      // CV: skip smearing in case either "raw" or "corrected" jet energy is very low
      //     or jet passes selection configurable via python
      //    (allows for protection against "pathological cases",
      //     cf. PhysicsTools/PatUtils/python/tools/metUncertaintyTools.py)
      reco::Candidate::LorentzVector smearedJetP4 = jet.p4();
      if ( !((skipJetSelection_ && (*skipJetSelection_)(jet)) ||
	     rawJetP4.pt()  < skipRawJetPtThreshold_          ||
	     corrJetP4.pt() < skipCorrJetPtThreshold_         ) ) {
	if ( verbosity_ ) {
	  std::cout << " smearing jetP4 by factor = " << (smearedJetEn/jet.energy()) << " --> smearedJetEn = " << smearedJetEn << std::endl;
	}
	smearedJetP4 *= (smearedJetEn/jet.energy());
      }

      if ( verbosity_ ) {
	std::cout << "smearedJet: Pt = " << smearedJetP4.pt() << ", eta = " << smearedJetP4.eta() << ", phi = " << smearedJetP4.phi() << std::endl;
	std::cout << " dPt = " << (smearedJetP4.pt() - jet.pt())
		  << " (Px = " << (smearedJetP4.px() - jet.px()) << ", Py = " << (smearedJetP4.py() - jet.py()) << ")" << std::endl;
      }

      T smearedJet = (jet);
      smearedJet.setP4(smearedJetP4);

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
  edm::EDGetTokenT<JetCollection> srcToken_;

  TFile* inputFile_;
  TH2* lut_;

  SmearedJetProducer_namespace::JetResolutionExtractorT<T> jetResolutionExtractor_;
  TRandom3 rnd_;

  std::string jetCorrLabel_; // e.g. 'ak5PFJetL1FastL2L3' (reco::PFJets) / '' (pat::Jets)
  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
  Textractor jetCorrExtractor_;

  double sigmaMaxGenJetMatch_; // maximum difference between energy of reconstructed jet and matched generator level jet
                               // (if the difference between reconstructed and generated jet energy exceeds this threshold,
                               //  the jet is considered to have substantial pile-up contributions are is considered to be unmatched)

  double smearBy_; // option to "smear" jet energy by N standard-deviations, useful for template morphing

  double shiftBy_; // option to increase/decrease within uncertainties the jet energy resolution used for smearing

  StringCutObjectSelector<T>* skipJetSelection_; // jets passing this cut are **not** smeared
  double skipRawJetPtThreshold_;  // jets with transverse momenta below this value (either on "raw" or "corrected" level)
  double skipCorrJetPtThreshold_; // are **not** smeared

  int verbosity_; // flag to enabled/disable debug output
};

#endif
