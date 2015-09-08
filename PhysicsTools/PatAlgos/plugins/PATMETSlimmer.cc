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
          OneMETShift(pat::MET::METUncertainty shift_, pat::MET::METCorrectionType level_, const edm::InputTag & baseTag, edm::ConsumesCollector && cc,
		      bool t0FromMiniAOD_, bool corShift_, bool uncShift_, bool isSmeared_=false) ;
          pat::MET::METUncertainty shift;
          pat::MET::METCorrectionType level;
          edm::EDGetTokenT<pat::METCollection> token;
	  bool t0FromMiniAOD;
	  bool corShift;
	  bool uncShift;
	  bool isSmeared;
          void readAndSet(const edm::Event &ev, pat::MET &met) ;
      };
    void maybeReadShifts(const edm::ParameterSet &basePSet, const std::string &name, pat::MET::METCorrectionType level, bool readFromMiniAOD=false) ;

      edm::EDGetTokenT<pat::METCollection> src_;
      std::vector<OneMETShift> shifts_;

      bool onMiniAOD_;
  };

} // namespace

pat::PATMETSlimmer::PATMETSlimmer(const edm::ParameterSet & iConfig) :
  src_(consumes<pat::METCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  onMiniAOD_ =false;
  if(iConfig.existsAs<bool>("runningOnMiniAOD")) {
    onMiniAOD_ = iConfig.getParameter<bool>("runningOnMiniAOD");
  }

  maybeReadShifts( iConfig, "rawVariation", pat::MET::None );
  maybeReadShifts( iConfig, "t1Uncertainties", pat::MET::T1 );
  maybeReadShifts( iConfig, "t01Variation", pat::MET::T0, onMiniAOD_ );
  maybeReadShifts( iConfig, "t1SmearedVarsAndUncs", pat::MET::Smear );
  
  maybeReadShifts( iConfig, "tXYUncForRaw", pat::MET::TXYForRaw );
  maybeReadShifts( iConfig, "tXYUncForT1", pat::MET::TXY );
  maybeReadShifts( iConfig, "tXYUncForT01", pat::MET::TXYForT01 ); 
  maybeReadShifts( iConfig, "tXYUncForT1Smear", pat::MET::TXYForT1Smear );
  maybeReadShifts( iConfig, "tXYUncForT01Smear", pat::MET::TXYForT01Smear );
  maybeReadShifts( iConfig, "caloMET", pat::MET::Calo );

  produces<std::vector<pat::MET> >();
}

void pat::PATMETSlimmer::maybeReadShifts(const edm::ParameterSet &basePSet, const std::string &name, pat::MET::METCorrectionType level, bool readFromMiniAOD) {
 
    if (basePSet.existsAs<edm::ParameterSet>(name)) {
        throw cms::Exception("Unsupported", "Reading PSets not supported, for now just use input tag");
    } else if (basePSet.existsAs<edm::InputTag>(name) ) {
        const edm::InputTag & baseTag = basePSet.getParameter<edm::InputTag>(name);

	if(level==pat::MET::T1) {
	  shifts_.push_back(OneMETShift(pat::MET::NoShift, level, baseTag, consumesCollector(), readFromMiniAOD, true, false, false));
	  shifts_.push_back(OneMETShift(pat::MET::NoShift, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetResUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetResDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::MuonEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::MuonEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::ElectronEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::ElectronEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::PhotonEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::PhotonEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::TauEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::TauEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::UnclusteredEnUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::UnclusteredEnDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true));
	}
	else if(level==pat::MET::Smear) {
	  shifts_.push_back(OneMETShift(pat::MET::NoShift, level, baseTag, consumesCollector(), readFromMiniAOD, true, false, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetResUp,   level, baseTag, consumesCollector(), readFromMiniAOD, false, true, true));
	  shifts_.push_back(OneMETShift(pat::MET::JetResDown, level, baseTag, consumesCollector(), readFromMiniAOD, false, true, true));
	}
	else {
	  shifts_.push_back(OneMETShift(pat::MET::NoShift, level, baseTag, consumesCollector(), readFromMiniAOD, true, false));
	}
    }

}

pat::PATMETSlimmer::OneMETShift::OneMETShift(pat::MET::METUncertainty shift_, pat::MET::METCorrectionType level_, const edm::InputTag & baseTag,
					     edm::ConsumesCollector && cc, bool t0FromMiniAOD_, bool corShift_, bool uncShift_, bool isSmeared) :
  shift(shift_), level(level_), t0FromMiniAOD(t0FromMiniAOD_), corShift(corShift_), uncShift(uncShift_), isSmeared(isSmeared)
{

    std::string baseTagStr = baseTag.encode();
    char buff[1024];
    switch (shift) {
        case pat::MET::NoShift  : snprintf(buff, 1023, baseTagStr.c_str(), "");   break;
        case pat::MET::JetEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "JetEnUp");   break;
        case pat::MET::JetEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "JetEnDown"); break;
        case pat::MET::JetResUp  : snprintf(buff, 1023, baseTagStr.c_str(), isSmeared?"JetResUp":"");   break;
        case pat::MET::JetResDown: snprintf(buff, 1023, baseTagStr.c_str(), isSmeared?"JetResDown":""); break;
        case pat::MET::MuonEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "MuonEnUp");   break;
        case pat::MET::MuonEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "MuonEnDown"); break;
        case pat::MET::ElectronEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "ElectronEnUp");   break;
        case pat::MET::ElectronEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "ElectronEnDown"); break;
        case pat::MET::PhotonEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "PhotonEnUp");   break;
        case pat::MET::PhotonEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "PhotonEnDown"); break;
        case pat::MET::TauEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "TauEnUp");   break;
        case pat::MET::TauEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "TauEnDown"); break;
        case pat::MET::UnclusteredEnUp  : snprintf(buff, 1023, baseTagStr.c_str(), "UnclusteredEnUp");   break;
        case pat::MET::UnclusteredEnDown: snprintf(buff, 1023, baseTagStr.c_str(), "UnclusteredEnDown"); break;
        default: throw cms::Exception("LogicError", "OneMETShift constructor called with bogus shift");
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


    if(t0FromMiniAOD) {
      if(uncShift) met.setUncShift(met2.shiftedPx( shift, pat::MET::Type01), met2.shiftedPy(shift, pat::MET::Type01), 
				   met2.shiftedSumEt(shift, pat::MET::Type01), shift, isSmeared);
      if(corShift) met.setCorShift(met2.corPx(pat::MET::Type01), met2.corPy(pat::MET::Type01), 
				   met2.corSumEt(pat::MET::Type01), level);
    }
    else {
      if(uncShift) met.setUncShift(met2.px(), met2.py(), met2.sumEt(), shift, isSmeared);
      if(corShift) met.setCorShift(met2.px(), met2.py(), met2.sumEt(), level);
    }

}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATMETSlimmer);
