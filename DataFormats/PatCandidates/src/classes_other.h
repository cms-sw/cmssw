#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"

#include "DataFormats/PatCandidates/interface/StringMap.h"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"

#include "DataFormats/PatCandidates/interface/Vertexing.h"

#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "DataFormats/PatCandidates/interface/CandKinResolution.h"

namespace {
  struct dictionary {

  std::pair<std::string, std::vector<float> > jcfcf;
  edm::Wrapper<std::pair<std::string, std::vector<float> > > w_jcfcf;
  std::vector<pat::JetCorrFactors::CorrectionFactor> v_jcfcf;
  edm::Wrapper<std::vector<pat::JetCorrFactors::CorrectionFactor> > w_v_jcfcf;
  std::vector<pat::JetCorrFactors> v_jcf;
  edm::Wrapper<std::vector<pat::JetCorrFactors> >  w_v_jcf;
  edm::ValueMap<pat::JetCorrFactors> vm_jcf;
  edm::Wrapper<edm::ValueMap<pat::JetCorrFactors> >  w_vm_jcf;
  //std::vector<pat::TauJetCorrFactors::CorrectionFactor> v_tjcfcf;
  //edm::Wrapper<std::vector<pat::TauJetCorrFactors::CorrectionFactor> > w_v_tjcfcf;
  std::vector<pat::TauJetCorrFactors> v_tjcf;
  edm::Wrapper<std::vector<pat::TauJetCorrFactors> >  w_v_tjcf;
  edm::ValueMap<pat::TauJetCorrFactors> vm_tjcf;
  edm::Wrapper<edm::ValueMap<pat::TauJetCorrFactors> >  w_vm_tjcf;

  edm::Wrapper<StringMap>   w_sm;

  edm::Wrapper<edm::ValueMap<pat::VertexAssociation> >	 w_vm_va;

  edm::Wrapper<std::vector<pat::EventHypothesis> >	 w_v_eh;

  std::pair<pat::IsolationKeys,reco::IsoDeposit>	 p_ik_id;
  std::vector<std::pair<pat::IsolationKeys,reco::IsoDeposit> >	 v_p_ik_id;

  edm::Wrapper<edm::ValueMap<pat::LookupTableRecord> >	 w_vm_p_lutr;

  pat::CandKinResolution ckr;
  std::vector<pat::CandKinResolution>  v_ckr;
  pat::CandKinResolutionValueMap vm_ckr;
  edm::Wrapper<pat::CandKinResolutionValueMap> w_vm_ckr;

  };

}
