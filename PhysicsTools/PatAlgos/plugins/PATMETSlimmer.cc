/**
  \class    pat::PATMETSlimmer PATMETSlimmer.h "PhysicsTools/PatAlgos/interface/PATMETSlimmer.h"
  \brief    Slimmer of PAT METs 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/PatCandidates/interface/MET.h"

namespace pat {

  class PATMETSlimmer : public edm::EDProducer {
    public:
      explicit PATMETSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATMETSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      struct OneMETShift {
          OneMETShift() {}
          OneMETShift(pat::MET::METUncertainty shift_, pat::MET::METUncertaintyLevel level_, const edm::InputTag & baseTag, edm::ConsumesCollector && cc) ;
          pat::MET::METUncertainty shift;
          pat::MET::METUncertaintyLevel level;
          edm::EDGetTokenT<pat::METCollection> token;
          void readAndSet(const edm::Event &ev, pat::MET &met) ;
      };
      void maybeReadShifts(const edm::ParameterSet &basePSet, const std::string &name, pat::MET::METUncertaintyLevel level) ;

      edm::EDGetTokenT<pat::METCollection> src_;
      std::vector<OneMETShift> shifts_;
  };

} // namespace

pat::PATMETSlimmer::PATMETSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<pat::METCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
    maybeReadShifts( iConfig, "rawUncertainties", pat::MET::Raw );
    maybeReadShifts( iConfig, "type1Uncertainties", pat::MET::Type1 );
    maybeReadShifts( iConfig, "type1p2Uncertainties", pat::MET::Type1p2 );
    produces<std::vector<pat::MET> >();
}

void pat::PATMETSlimmer::maybeReadShifts(const edm::ParameterSet &basePSet, const std::string &name, pat::MET::METUncertaintyLevel level) {
    if (basePSet.existsAs<edm::ParameterSet>(name)) {
        throw cms::Exception("Unsupported", "Reading PSets not supported, for now just use input tag");
    } else if (basePSet.existsAs<edm::InputTag>(name)) {
        const edm::InputTag & baseTag = basePSet.getParameter<edm::InputTag>(name);
        shifts_.push_back(OneMETShift(pat::MET::JetEnUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::JetEnDown, level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::JetResUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::JetResDown, level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::MuonEnUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::MuonEnDown, level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::ElectronEnUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::ElectronEnDown, level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::TauEnUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::TauEnDown, level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::UnclusteredEnUp,   level, baseTag, consumesCollector()));
        shifts_.push_back(OneMETShift(pat::MET::UnclusteredEnDown, level, baseTag, consumesCollector()));
    }
}

pat::PATMETSlimmer::OneMETShift::OneMETShift(pat::MET::METUncertainty shift_, pat::MET::METUncertaintyLevel level_, const edm::InputTag & baseTag, edm::ConsumesCollector && cc) :
    shift(shift_), level(level_)
{
    std::string baseTagStr = baseTag.encode();
    char buff[1024];
    switch (shift) {
        case pat::MET::JetEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "JetEnUp");   break;
        case pat::MET::JetEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "JetEnDown"); break;
        case pat::MET::JetResUp  : snprintf(buff, 1023, baseTagStr.c_str(), "JetResUp");   break;
        case pat::MET::JetResDown: snprintf(buff, 1023, baseTagStr.c_str(), "JetResDown"); break;
        case pat::MET::MuonEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "MuonEnUp");   break;
        case pat::MET::MuonEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "MuonEnDown"); break;
        case pat::MET::ElectronEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "ElectronEnUp");   break;
        case pat::MET::ElectronEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "ElectronEnDown"); break;
        case pat::MET::TauEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "TauEnUp");   break;
        case pat::MET::TauEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "TauEnDown"); break;
        case pat::MET::UnclusteredEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "UnclusteredEnUp");   break;
        case pat::MET::UnclusteredEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "UnclusteredEnDown"); break;
        default: throw cms::Exception("LogicError", "OneMETShift constructor called wih bogus shift");
    }
    token = cc.consumes<pat::METCollection>(edm::InputTag(buff));
}

void 
pat::PATMETSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<pat::METCollection>  src;
    iEvent.getByToken(src_, src);
    if (src->size() != 1) throw cms::Exception("CorruptData", "More than one MET in the collection");

    auto_ptr<vector<pat::MET> >  out(new vector<pat::MET>(1, src->front()));
    pat::MET & met = out->back();
    for (OneMETShift &shift : shifts_) {
        shift.readAndSet(iEvent, met);
    }

    iEvent.put(out);
}

void
pat::PATMETSlimmer::OneMETShift::readAndSet(const edm::Event &ev, pat::MET &met) {
    edm::Handle<pat::METCollection>  src;
    ev.getByToken(token, src);
    if (src->size() != 1) throw cms::Exception("CorruptData", "More than one MET in the shifted collection");
    const pat::MET &met2 = src->front();
    met.setShift(met2.px(), met2.py(), met2.sumEt(), shift, level);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATMETSlimmer);
