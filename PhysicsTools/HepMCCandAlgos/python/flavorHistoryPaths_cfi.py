
import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.flavorHistoryProducer_cfi import *
from PhysicsTools.HepMCCandAlgos.flavorHistoryFilter_cfi import *


from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.sisCone5GenJets_cff import *

#
# Create prioritized paths to separate HF composition samples.
# 
# These are exclusive priorities, so sample "i" will not overlap with "i+1".
# Note that the "dr" values below correspond to the dr between the
# matched genjet, and the sister genjet. 
#
# 1) W+bb with >= 2 jets from the ME (dr > 0.5)
# 2) W+b or W+bb with 1 jet from the ME
# 3) W+cc from the ME (dr > 0.5)
# 4) W+c or W+cc with 1 jet from the ME
# 5) W+bb with 1 jet from the parton shower (dr == 0.0)
# 6) W+cc with 1 jet from the parton shower (dr == 0.0)
#
# These are the "trash bin" samples that we're throwing away:
#
# 7) W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
# 8) W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
# 9) W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
# 10)W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)
#
# And here is the true "light flavor" sample:
#
# 11) Veto of all the previous (W+ light jets)
#


#wjetsAna.verbose = cms.bool(True)

# 1) W+bb with >= 2 jets from the ME (dr > 0.5)
wbb = cms.Path(  
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    wbbMEFlavorHistoryFilter
    )

# 2) W+b or W+bb with 1 jet from the ME
wb = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    wbFEFlavorHistoryFilter
    )

# 3) W+cc from the ME (dr > 0.5)
wcc = cms.Path( 
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    wccMEFlavorHistoryFilter
    )

# 4) W+c or W+cc with 1 jet from the ME
wc = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    wcFEFlavorHistoryFilter
    )

# 5) W+bb with 1 jet from the parton shower (dr == 0.0)
wbb_gs = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    wbbGSFlavorHistoryFilter
    )

# 6) W+cc with 1 jet from the parton shower (dr == 0.0)
wcc_gs = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    wccGSFlavorHistoryFilter
    )

#
# And these are the "trash bin" samples that we're throwing away:
#
# 7) W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
wbb_comp = cms.Path( 
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    ~wccGSFlavorHistoryFilter*
    wbbMEComplimentFlavorHistoryFilter
    )


# 8) W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
wcc_comp = cms.Path( 
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    ~wccGSFlavorHistoryFilter*
    ~wbbMEComplimentFlavorHistoryFilter*
    wccMEComplimentFlavorHistoryFilter
    )

# 9) W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
wbb_gs_comp = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    ~wccGSFlavorHistoryFilter*
    ~wbbMEComplimentFlavorHistoryFilter*
    ~wccMEComplimentFlavorHistoryFilter*
    wbbGSComplimentFlavorHistoryFilter
    )

# 10)W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)
wcc_gs_comp = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    ~wccGSFlavorHistoryFilter*
    ~wbbMEComplimentFlavorHistoryFilter*
    ~wccMEComplimentFlavorHistoryFilter*
    ~wbbGSComplimentFlavorHistoryFilter*
    wccGSComplimentFlavorHistoryFilter
    )

#
# Here is the "true" light flavor sample:
#
# 11) Veto of all the previous (W+ light jets)

wjets = cms.Path(
    genJetParticles*sisCone5GenJets*
    bFlavorHistoryProducer*
    cFlavorHistoryProducer*
    ~wbbMEFlavorHistoryFilter*
    ~wbFEFlavorHistoryFilter*
    ~wccMEFlavorHistoryFilter*
    ~wcFEFlavorHistoryFilter*
    ~wbbGSFlavorHistoryFilter*
    ~wccGSFlavorHistoryFilter*
    ~wbbMEComplimentFlavorHistoryFilter*
    ~wccMEComplimentFlavorHistoryFilter*
    ~wbbGSComplimentFlavorHistoryFilter*
    ~wccGSComplimentFlavorHistoryFilter
    )
