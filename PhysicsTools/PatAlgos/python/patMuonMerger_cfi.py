import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.mergedMuonsNoCuts_cfi as _mod
mergedMuons = _mod.mergedMuonsNoCuts.clone(
                             muonCut         = "pt>15 && abs(eta)<2.4",
                             pfCandidatesCut = "pt>15 && abs(eta)<2.4",
                             lostTrackCut    = "pt>15 && abs(eta)<2.4"
                         )
