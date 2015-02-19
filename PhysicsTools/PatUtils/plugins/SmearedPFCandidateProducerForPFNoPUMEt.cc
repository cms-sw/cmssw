#include "PhysicsTools/PatUtils/plugins/SmearedPFCandidateProducerForPFNoPUMEt.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"


const double defJetPtThr = 0.01;
const double dR2Match = 0.01*0.01;
const double etaMaxBound = 9.9;


template <typename T, typename Textractor>
SmearedPFCandidateProducerForPFNoPUMEtT<T, Textractor>::SmearedPFCandidateProducerForPFNoPUMEtT(const edm::ParameterSet& cfg)
  : genJetMatcher_(cfg,consumesCollector()),
    jetResolutionExtractor_(cfg.getParameter<edm::ParameterSet>("jetResolutions")),
    jetCorrLabel_(""),
    skipJetSelection_(nullptr)
{

  srcPFCandidates_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates") );
  srcJets_ = consumes<JetCollection>(cfg.getParameter<edm::InputTag>("srcJets"));

  edm::FileInPath inputFileName = cfg.getParameter<edm::FileInPath>("inputFileName");
  std::string lutName = cfg.getParameter<std::string>("lutName");
  if ( inputFileName.location() == edm::FileInPath::Unknown )
    throw cms::Exception("SmearedPFCandidateProducerForPFNoPUMEt")
      << " Failed to find File = " << inputFileName << " !!\n";

  inputFile_ = new TFile(inputFileName.fullPath().data());
  lut_ = dynamic_cast<TH2*>(inputFile_->Get(lutName.data()));
  if ( !lut_ )
    throw cms::Exception("SmearedPFCandidateProducerForPFNoPUMEt")
      << " Failed to load LUT = " << lutName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";

    if ( cfg.exists("jetCorrLabel") ) {
      jetCorrLabel_ = cfg.getParameter<edm::InputTag>("jetCorrLabel");
      jetCorrToken_ = consumes<reco::JetCorrector>(jetCorrLabel_);
    }

  jetCorrEtaMax_ = ( cfg.exists("jetCorrEtaMax") ) ?
    cfg.getParameter<double>("jetCorrEtaMax") : etaMaxBound;

  sigmaMaxGenJetMatch_ = cfg.getParameter<double>("sigmaMaxGenJetMatch");

  smearBy_ = ( cfg.exists("smearBy") ) ? cfg.getParameter<double>("smearBy") : 1.0;
  //std::cout << "smearBy = " << smearBy_ << std::endl;

  shiftBy_ = ( cfg.exists("shiftBy") ) ? cfg.getParameter<double>("shiftBy") : 0.;
  //std::cout << "shiftBy = " << shiftBy_ << std::endl;

  skipJetSel_ = cfg.exists("skipJetSelection");
  if ( skipJetSel_ ) {
    std::string skipJetSelection_string = cfg.getParameter<std::string>("skipJetSelection");
    skipJetSelection_ = new StringCutObjectSelector<T>(skipJetSelection_string);
  }


  skipRawJetPtThreshold_  = ( cfg.exists("skipRawJetPtThreshold")  ) ?
    cfg.getParameter<double>("skipRawJetPtThreshold")  : defJetPtThr;
  skipCorrJetPtThreshold_ = ( cfg.exists("skipCorrJetPtThreshold") ) ?
    cfg.getParameter<double>("skipCorrJetPtThreshold") : defJetPtThr;

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<reco::PFCandidateCollection>();
}

template <typename T, typename Textractor>
SmearedPFCandidateProducerForPFNoPUMEtT<T, Textractor>::~SmearedPFCandidateProducerForPFNoPUMEtT()
{
  delete inputFile_;
  if ( skipJetSel_ ) {
    delete skipJetSelection_;
  }
}

template <typename T, typename Textractor>
void SmearedPFCandidateProducerForPFNoPUMEtT<T, Textractor>::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByToken(srcPFCandidates_, originalPFCandidates);

  edm::Handle<JetCollection> jets;
  evt.getByToken(srcJets_, jets);

  std::auto_ptr<reco::PFCandidateCollection> smearedPFCandidates(new reco::PFCandidateCollection);

  for ( reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
	originalPFCandidate != originalPFCandidates->end(); ++originalPFCandidate ) {

    const T* jet_matched = nullptr;
    for ( typename JetCollection::const_iterator jet = jets->begin();
	  jet != jets->end(); ++jet ) {
      std::vector<reco::PFCandidatePtr> jetConstituents = jet->getPFConstituents();
      for ( std::vector<reco::PFCandidatePtr>::const_iterator jetConstituent = jet->getPFConstituents().begin();
	    jetConstituent != jet->getPFConstituents().end() && jet_matched==nullptr; ++jetConstituent ) {
	if ( deltaR2(originalPFCandidate->p4(), (*jetConstituent)->p4()) < dR2Match ) jet_matched = &(*jet);
      }
    }

    if ( jet_matched==nullptr ) continue;

    const static SmearedJetProducer_namespace::RawJetExtractorT<T> rawJetExtractor;
    reco::Candidate::LorentzVector rawJetP4 = rawJetExtractor(*jet_matched);
    // if ( verbosity_ ) {
    //   std::cout << "rawJet: Pt = " << rawJetP4.pt() << ", eta = " << rawJetP4.eta() << ", phi = " << rawJetP4.phi() << std::endl;
    // }

    reco::Candidate::LorentzVector corrJetP4 = jet_matched->p4();
    if ( !jetCorrLabel_.label().empty() ) {
      edm::Handle<reco::JetCorrector> jetCorr;
      evt.getByToken(jetCorrToken_, jetCorr);
      corrJetP4 = jetCorrExtractor_(*jet_matched, jetCorr.product(), jetCorrEtaMax_, &rawJetP4);
    }
    // if ( verbosity_ ) {
    //   std::cout << "corrJet: Pt = " << corrJetP4.pt() << ", eta = " << corrJetP4.eta() << ", phi = " << corrJetP4.phi() << std::endl;
    // }

    double smearFactor = 1.;
    double x = std::abs(corrJetP4.eta());
    double y = corrJetP4.pt();
    if ( x > lut_->GetXaxis()->GetXmin() && x < lut_->GetXaxis()->GetXmax() &&
	 y > lut_->GetYaxis()->GetXmin() && y < lut_->GetYaxis()->GetXmax() ) {
      int binIndex = lut_->FindBin(x, y);

      if ( smearBy_ > 0. ) smearFactor += smearBy_*(lut_->GetBinContent(binIndex) - 1.);
      double smearFactorErr = lut_->GetBinError(binIndex);
      //if ( verbosity_ ) std::cout << "smearFactor = " << smearFactor << " +/- " << smearFactorErr << std::endl;

      if ( shiftBy_ != 0. ) {
	smearFactor += (shiftBy_*smearFactorErr);
	//if ( verbosity_ ) std::cout << "smearFactor(shifted) = " << smearFactor << std::endl;
      }
    }

    double smearedJetEn = jet_matched->energy();

    T rawJet(*jet_matched);
    rawJet.setP4(rawJetP4);
    double jetResolution = jetResolutionExtractor_(rawJet);
    double sigmaEn = jetResolution;

    const reco::GenJet* genJet = genJetMatcher_(*jet_matched, &evt);
    bool isGenMatched = false;
    if ( genJet!=nullptr ) {
      // if ( verbosity_ ) {
      // 	std::cout << "genJet: Pt = " << genJet->pt() << ", eta = " << genJet->eta() << ", phi = " << genJet->phi() << std::endl;
      // }
      double dEn = corrJetP4.E() - genJet->energy();
      if ( std::abs(dEn) < (sigmaMaxGenJetMatch_*sigmaEn) ) {
//--- case 1: reconstructed jet matched to generator level jet,
//            smear difference between reconstructed and "true" jet energy

	// if ( verbosity_ ) {
	//   std::cout << " successfully matched to genJet" << std::endl;
	//   std::cout << "corrJetEn = " << corrJetP4.E() << ", genJetEn = " << genJet->energy() << " --> dEn = " << dEn << std::endl;
	// }

	//smearedJetEn = jet_matched->energy()*(1. + (smearFactor - 1.)*dEn/std::max(rawJetP4.E(), corrJetP4.E()));
	smearedJetEn = jet_matched->energy() + (smearFactor - 1.)*dEn;
	isGenMatched = true;
      }
    }
    if ( !isGenMatched ) {
//--- case 2: reconstructed jet **not** matched to generator level jet,
//            smear jet energy using MC resolution functions implemented in PFMEt significance algorithm (CMS AN-10/400)

      // if ( verbosity_ ) {
      // 	std::cout << " not matched to genJet" << std::endl;
      // 	std::cout << "corrJetEn = " << corrJetP4.E() << ", sigmaEn = " << sigmaEn << std::endl;
      // }

      if ( smearFactor > 1. ) {
	// CV: MC resolution already accounted for in reconstructed jet,
	//     add additional Gaussian smearing of width = sqrt(smearFactor^2 - 1)
	//     to account for Data/MC **difference** in jet resolutions.
	//     Take maximum(rawJetEn, corrJetEn) to avoid pathological cases
	//    (e.g. corrJetEn << rawJetEn, due to L1Fastjet corrections)

	double addSigmaEn = jetResolution*sqrt(smearFactor*smearFactor - 1.);
	//smearedJetEn = jet_matched->energy()*(1. + rnd_.Gaus(0., addSigmaEn)/std::max(rawJetP4.E(), corrJetP4.E()));
	smearedJetEn = jet_matched->energy() + rnd_.Gaus(0., addSigmaEn);
      }
    }

    // CV: keep minimum jet energy, in order not to loose direction information
    const double minJetEn = 1.e-2;
    if ( smearedJetEn < minJetEn ) smearedJetEn = minJetEn;

    // CV: skip smearing in case either "raw" or "corrected" jet energy is very low
    //     or jet passes selection configurable via python
    //    (allows for protection against "pathological cases",
    //     cf. PhysicsTools/PatUtils/python/tools/metUncertaintyTools.py)
    reco::Candidate::LorentzVector smearedJetP4 = jet_matched->p4();
    if ( !((skipJetSelection_ && (*skipJetSelection_)(*jet_matched)) ||
	   rawJetP4.pt()  < skipRawJetPtThreshold_                   ||
	   corrJetP4.pt() < skipCorrJetPtThreshold_                  ) ) {
      // if ( verbosity_ ) {
      // 	std::cout << " multiplying jetP4 by factor = " << (smearedJetEn/jet_matched->energy()) << " --> smearedJetEn = " << smearedJetEn << std::endl;
      // }
      smearedJetP4 *= (smearedJetEn/jet_matched->energy());
    }

    // if ( verbosity_ ) {
    //   std::cout << "smearedJet: Pt = " << smearedJetP4.pt() << ", eta = " << smearedJetP4.eta() << ", phi = " << smearedJetP4.phi() << std::endl;
    //   std::cout << " dPt = " << (smearedJetP4.pt() - jet_matched->pt())
    // 		<< " (dPx = " << (smearedJetP4.px() - jet_matched->px()) << ", dPy = " << (smearedJetP4.py() - jet_matched->py()) << ")" << std::endl;
    // }

    double scaleFactor = ( jet_matched->p4().energy() > 0. ) ?
      (smearedJetP4.energy()/jet_matched->p4().energy()) : 1.0;

    double smearedPx = scaleFactor*originalPFCandidate->px();
    double smearedPy = scaleFactor*originalPFCandidate->py();
    double smearedPz = scaleFactor*originalPFCandidate->pz();
    double mass      = originalPFCandidate->mass();
    double smearedEn = sqrt(smearedPx*smearedPx + smearedPy*smearedPy + smearedPz*smearedPz + mass*mass);
    reco::Candidate::LorentzVector smearedPFCandidateP4(smearedPx, smearedPy, smearedPz, smearedEn);

    reco::PFCandidate smearedPFCandidate(*originalPFCandidate);
    smearedPFCandidate.setP4(smearedPFCandidateP4);

    smearedPFCandidates->push_back(smearedPFCandidate);
  }

  evt.put(smearedPFCandidates);
}

#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"

namespace SmearedJetProducer_namespace
{
  template <>
  class JetResolutionExtractorT<reco::PFJet>
  {
    public:

     JetResolutionExtractorT(const edm::ParameterSet& cfg)
       : jetResolutions_(cfg)
     {}
     ~JetResolutionExtractorT() {}

     double operator()(const reco::PFJet& jet) const
     {
       metsig::SigInputObj pfJetResolution = jetResolutions_.evalPFJet(&jet);
       if ( pfJetResolution.get_energy() > 0. ) {
	 return jet.energy()*(pfJetResolution.get_sigma_e()/pfJetResolution.get_energy());
       } else {
	 return 0.;
       }
     }

     metsig::SignAlgoResolutions jetResolutions_;
  };
}

typedef SmearedPFCandidateProducerForPFNoPUMEtT<reco::PFJet, JetCorrExtractorT<reco::PFJet> > SmearedPFCandidateProducerForPFNoPUMEt;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedPFCandidateProducerForPFNoPUMEt);

