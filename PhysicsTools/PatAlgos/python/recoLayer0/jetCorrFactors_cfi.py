import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.JetCorrFactorsProducer_cfi as _mod

# module to produce jet correction factors associated in a valuemap
patJetCorrFactors = _mod.JetCorrFactorsProducer.clone(
    ## the use of emf in the JEC is not yet implemented
    emf = False,
    ## input collection of jets
    src = "ak4PFJetsCHS",
    ## payload postfix for testing
    payload = 'AK4PFchs',
    ## correction levels
    levels = [
        ## tags for the individual jet corrections; when
        ## not available the string should be set to 'none'
        'L1FastJet', 'L2Relative', 'L3Absolute',#'L5Flavor', 'L7Parton'
    ],
    ## define the type of L5Flavor corrections for here. These can
    ## be of type 'J' for dijet derived, or of type 'T' for ttbar
    ## derived.
    flavorType = 'J',
    ## in case you are using JPT jets you must have specified the L1Offset
    ## corrections by a dedicated L1JPTOffset correction level. This dedi-
    ## cated correction level has an ordinary L1Offset or L1FastJet corrector
    ## as input, which needs to be specified via this additional parameter
    extraJPTOffset = "L1FastJet",
    ## in case that L1Offset or L1FastJet corrections are part
    ## of the parameter levels add the optional parameter
    ## primaryVertices here to specify the primary vertex
    ## collection, which was used to determine the L1Offset
    ## or L1FastJet correction from. This parameter will ONLY
    ## be read out if the correction level L1Offset or
    ## L1FastJet is found in levels.
    useNPV = True,
    primaryVertices = 'offlinePrimaryVertices',
    ## in case that L1FastJet corrections are part of the
    ## parameter levels add the optional parameter rho
    ## here to specify the energy density parameter for
    ## the corresponding jet collection (this variable is
    ## typically taken from kt6PFJets).
    useRho = True,
    rho = 'fixedGridRhoFastjetAll',
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(patJetCorrFactors,
                                           useNPV = False,
                                           useRho = False,
                                           levels = ['L2Relative','L3Absolute'],
                                           payload = "AK4PF",
                                           src = "akCs4PFJets",
                                       )
