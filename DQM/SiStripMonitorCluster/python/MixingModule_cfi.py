import FWCore.ParameterSet.Config as cms

# from Configuration/Applications/data/DIGI-RECO.cfg 
mix = cms.EDFilter("MixingModule",
    #      secsource input = EmbeddedRootSource
    #      {
    #
    # starting 050, you can skip secsource block at all if you don't
    # wnat to model mixing/pileup at all;
    # however, an at least "empty mixing" need to be present if you
    # want to perform Ecal/Hcal/CSC/DT digitization - those explicitely
    # require presence of CrossingFrame in the edm::Event
    #
    # alternatively, you can set averageNumber=0 if you don't want
    # to model the pileup
    #
    # to the secsource/EmbeddedRootSource, you can give just 1 file or more;
    # this files will make a "concatinated buffer", which will go circular
    # until the loop of primary events is done - thus, it'll never run out
    #
    # WARNING: you can only give miltiple files, if they're generated with
    #          identical sets of tracked parameters;
    #          for example, you canNOT give a file made with a single muon
    #          gun and a file made with a single pion gun, because PartID
    #          is a *tracked* parameter in the gun's PSet;
    #          however, you can merge together files made with other generators,
    #          because all parameters of it would be
    #          *untracked*
    #
    #         untracked vstring fileNames =
    #         {'file:/afs/cern.ch/cms/geant4rep/genntpl/muon_simhit_for_pileup.060pre1.root' }
    #         untracked vstring fileNames =
    #         {'file:simevent.root' }
    #         string type = "poisson"
    #         double averageNumber = 3      # setting this param. to 0 means "No pile-up",
    # that is, digitize current crossing only
    #         int32 minBunch = -3
    #         int32 maxBunch = 5
    #         int32 seed = 1234567
    #      }
    bunchspace = cms.int32(25),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    Label = cms.string('')
)


