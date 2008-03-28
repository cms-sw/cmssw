# The following comments couldn't be translated into the new config version:

# ! hard QCD on',
# ! to avoid stopping run',
# ! tolerance parameter to adjust fragmentation',
# ! pp multiple scattering off',
# ! b quark mass',
# ! t quark mass'
# ! ptMin
# q+q->q+q
# q+qbar->q+qbar
# q+qbar->g+g
# q+g->q+g
# g+g->q+qbar
# g+g->g+g
# q+qbar->g+gamma
# q+qbar->gamma+gamma
# q+g->q+gamma
# g+g->gamma+gamma
# g+g->g+gamma
# ! Only user defined processes,
import FWCore.ParameterSet.Config as cms

pythiaDefault = cms.vstring('MSEL=1', 'MSTU(21)=1', 'PARU(14)=1.', 'MSTP(81)=0', 'PMAS(5,1)=4.8', 'PMAS(6,1)=175.0', 'CKIN(3)=7.')
pythiaJets = cms.vstring('MSUB(11)=1', 'MSUB(12)=1', 'MSUB(13)=1', 'MSUB(28)=1', 'MSUB(53)=1', 'MSUB(68)=1')
pythiaPromptPhotons = cms.vstring('MSUB(14)=1', 'MSUB(18)=1', 'MSUB(29)=1', 'MSUB(114)=1', 'MSUB(115)=1')
orcaSettings = cms.vstring('MSEL=0', 'PARP(67)=1.', 'PARP(82)=1.9', 'PARP(85)=0.33', 'PARP(86)=0.66', 'PARP(89)=1000.', 'PARP(91)=1.0', 'MSTJ(11)=3', 'MSTJ(22)=2')

