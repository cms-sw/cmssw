/** \class ShiftedMETcorrInputProducer
 *
 * Vary px, py and sumEt of "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)
 * by +/- 1 standard deviation, in order to estimate resulting uncertainty on MET
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <string>

class ShiftedMETcorrInputProducer : public edm::stream::EDProducer<> {
public:
  explicit ShiftedMETcorrInputProducer(const edm::ParameterSet&);
  ~ShiftedMETcorrInputProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag src_;
  std::vector<edm::EDGetTokenT<CorrMETData> > srcTokens_;

  struct binningEntryType {
    binningEntryType(double uncertainty) : binLabel_(""), binUncertainty_(uncertainty) {}
    binningEntryType(const edm::ParameterSet& cfg)
        : binLabel_(cfg.getParameter<std::string>("binLabel")),
          binUncertainty_(cfg.getParameter<double>("binUncertainty")) {}
    std::string getInstanceLabel_full(const std::string& instanceLabel) {
      std::string retVal = instanceLabel;
      if (!instanceLabel.empty() && !binLabel_.empty())
        retVal.append("#");
      retVal.append(binLabel_);
      return retVal;
    }
    ~binningEntryType() {}
    std::string binLabel_;
    double binUncertainty_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_;
};

ShiftedMETcorrInputProducer::ShiftedMETcorrInputProducer(const edm::ParameterSet& cfg) {
  src_ = cfg.getParameter<vInputTag>("src");

  //--- check that all InputTags refer to the same module label
  //   (i.e. differ by instance label only)
  for (vInputTag::const_iterator src_ref = src_.begin(); src_ref != src_.end(); ++src_ref) {
    for (vInputTag::const_iterator src_test = src_ref; src_test != src_.end(); ++src_test) {
      if (src_test->label() != src_ref->label())
        throw cms::Exception("ShiftedMETcorrInputProducer")
            << "InputTags specified by 'src' Configuration parameter must not refer to different module labels !!\n";
    }
  }

  shiftBy_ = cfg.getParameter<double>("shiftBy");

  if (cfg.exists("binning")) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for (vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin(); cfgBinningEntry != cfgBinning.end();
         ++cfgBinningEntry) {
      binning_.push_back(new binningEntryType(*cfgBinningEntry));
    }
  } else {
    double uncertainty = cfg.getParameter<double>("uncertainty");
    binning_.push_back(new binningEntryType(uncertainty));
  }

  for (vInputTag::const_iterator src_i = src_.begin(); src_i != src_.end(); ++src_i) {
    for (std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin(); binningEntry != binning_.end();
         ++binningEntry) {
      srcTokens_.push_back(consumes<CorrMETData>(
          edm::InputTag(src_i->label(), (*binningEntry)->getInstanceLabel_full(src_i->instance()))));
      produces<CorrMETData>((*binningEntry)->getInstanceLabel_full(src_i->instance()));
    }
  }
}

ShiftedMETcorrInputProducer::~ShiftedMETcorrInputProducer() {
  for (std::vector<binningEntryType*>::const_iterator it = binning_.begin(); it != binning_.end(); ++it) {
    delete (*it);
  }
}

void ShiftedMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  unsigned countToken(0);
  for (vInputTag::const_iterator src_i = src_.begin(); src_i != src_.end(); ++src_i) {
    for (std::vector<binningEntryType*>::iterator binningEntry = binning_.begin(); binningEntry != binning_.end();
         ++binningEntry) {
      edm::Handle<CorrMETData> originalObject;
      evt.getByToken(srcTokens_.at(countToken), originalObject);
      ++countToken;

      double shift = shiftBy_ * (*binningEntry)->binUncertainty_;

      auto shiftedObject = std::make_unique<CorrMETData>(*originalObject);
      //--- MET balances momentum of reconstructed particles,
      //    hence variations of "unclustered energy" and MET are opposite in sign
      shiftedObject->mex = -shift * originalObject->mex;
      shiftedObject->mey = -shift * originalObject->mey;
      shiftedObject->sumet = shift * originalObject->sumet;

      evt.put(std::move(shiftedObject), (*binningEntry)->getInstanceLabel_full(src_i->instance()));
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedMETcorrInputProducer);
