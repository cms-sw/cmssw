import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

string_func = """
result_type asCandidate(const argument_type& obj) const override final { 
  std::cout << "lol I was written in python!" << std::endl;
  return obj->pt() < 5.0;
}  
"""

string_value = """
double value(const reco::CandidatePtr& obj) const override final { 
  return obj->pt();
}  
"""

trivialCutFlow = cms.PSet(
    idName = cms.string("trivialCutFlow"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MinPtCut"),
                  minPt = cms.double(10.0),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                ),
        cms.PSet( cutName = cms.string("MaxAbsEtaCut"),
                  maxEta = cms.double(2.5),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                ),
        cms.PSet( cutName = cms.string("ExpressionEvaluatorCut"),
                  realCutName = cms.string("StringMinPtCut"),
                  candidateType = cms.string("NONE"),
                  functionDef = cms.string(string_func),
                  valueDef = cms.string(string_value),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                )
    )
)

central_id_registry.register(trivialCutFlow.idName,
                             '406a42716bb40f14256446a98e25c1de')
