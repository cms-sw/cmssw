from builtins import range
import FWCore.ParameterSet.Config as cms

et_vs_et_bins = []
et_vs_et_bins.extend(list(range(0, 50, 1)))
et_vs_et_bins.extend(list(range(50, 150, 5)))
et_vs_et_bins.extend(list(range(150, 500, 50)))

histDefinitions = cms.PSet(
    nVertex=cms.PSet(
        name=cms.untracked.string('nVertex'),
        title=cms.untracked.string('Number of event vertices in collection'),
        nbinsX=cms.untracked.uint32(100),
        xmin=cms.untracked.double(-0.5),
        xmax=cms.untracked.double(99.5),
    ),
    ETvsET=cms.PSet(
        name=cms.untracked.string('ETvsET'),
        title=cms.untracked.string('Template for ET vs ET histograms'),
        binsX=cms.untracked.vdouble(et_vs_et_bins),
        binsY=cms.untracked.vdouble(et_vs_et_bins),
        nbinsX=cms.untracked.uint32(len(et_vs_et_bins) - 1),
        nbinsY=cms.untracked.uint32(len(et_vs_et_bins) - 1),
    ),
    PHIvsPHI=cms.PSet(
        name=cms.untracked.string('PHIvsPHI'),
        title=cms.untracked.string('Template for \phi vs \phi histograms'),
        nbinsX=cms.untracked.uint32(80),
        xmin=cms.untracked.double(-3.2),
        xmax=cms.untracked.double(3.2),
        nbinsY=cms.untracked.uint32(80),
        ymin=cms.untracked.double(-3.2),
        ymax=cms.untracked.double(3.2),
    ),
)
