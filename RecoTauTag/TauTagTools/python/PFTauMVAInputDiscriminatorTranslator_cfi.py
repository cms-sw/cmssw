import FWCore.ParameterSet.Config as cms

produceTancMVAInputDiscriminators = cms.EDProducer(
    "PFTauMVAInputDiscriminantTranslator",
    pfTauSource = cms.InputTag("shrinkingConePFTauProducer"),
    # List of MVA inputs to turn into discriminants
    # Each PFTauDiscirminator corresponds to a variable name
    # (see RecoTauTag/TauTagTools/src/DiscriminantList.cc)
    # an index in case of multiple values (i.e. TrackPt) and
    # a default value in case the desired index does not exist.
    # If the default value is ommitted, zero is used.
    discriminants = cms.VPSet(),
    #discriminants = cms.VPSet(
        #cms.PSet(
            #name=cms.string('DecayMode')
        #),
        #cms.PSet(name=cms.string('MainTrackPt') ),
        #cms.PSet(name=cms.string('MainTrackAngle') ),
        #cms.PSet(name=cms.string('TrackPt'),
                 #indices=cms.vuint32(0,1)),
        #cms.PSet(name=cms.string('TrackAngle'),
                 #indices=cms.vuint32(0,1),
                 #default=cms.double(1.0)
                #),
        #cms.PSet(name=cms.string('PiZeroPt'),
                 #indices=cms.vuint32(0,1),
                #),
        #cms.PSet(name=cms.string('PiZeroAngle') ,
                 #indices=cms.vuint32(0,1),
                 #default=cms.double(1.0)
                #),
        #cms.PSet(name=cms.string('Dalitz'),
                 #indices=cms.vuint32(0,1),
                #),
        #cms.PSet(name=cms.string('InvariantMassOfSignal') ),
        ##cms.PSet(name=cms.string('InvariantMass') ),
        #cms.PSet(name=cms.string('Pt') ),
        #cms.PSet(name=cms.string('Eta') ),
        #cms.PSet(name=cms.string('OutlierPt'),
                 #indices=cms.vuint32(0,1, 2, 3),
                #),
        #cms.PSet(name=cms.string('OutlierAngle'),
                 #indices=cms.vuint32(0,1, 2, 3),
                 #default=cms.double(1.0)
                #),
        #cms.PSet(name=cms.string('ChargedOutlierPt'),
                 #indices=cms.vuint32(0,1, 2, 3),
                #),
        #cms.PSet(name=cms.string('ChargedOutlierAngle'),
                 #indices=cms.vuint32(0,1, 2, 3),
                 #default=cms.double(1.0)
                #),
        #cms.PSet(name=cms.string('NeutralOutlierPt'),
                 #indices=cms.vuint32(0,1, 2, 3),
                #),
        #cms.PSet(name=cms.string('NeutralOutlierAngle'),
                 #indices=cms.vuint32(0,1, 2, 3),
                 #default=cms.double(1.0)
                #),
        #cms.PSet(name=cms.string('OutlierNCharged')),
        #cms.PSet(name=cms.string('OutlierN') ),
        #cms.PSet(name=cms.string('OutlierSumPt') ),
        #cms.PSet(name=cms.string('OutlierMass') ),
        #cms.PSet(name=cms.string('ChargedOutlierSumPt') ),
        #cms.PSet(name=cms.string('NeutralOutlierSumPt') ),
        ##cms.PSet(name='GammaOccupancy'),
        ##cms.PSet(name='GammaPt' ),
        ##cms.PSet(name='FilteredObjectPt' ),
        ##cms.PSet(name='InvariantMassOfSignalWithFiltered' ),
    #)
)

def loadMVAInputsIntoPatTauDiscriminants(thePatTauProducer):
    return
    " Add all of the MVA inputs discriminators to the tauID inputs of a patTau Producer "
    print "Embedding MVA inputs into PAT Tau producer "
    patTauIDConfig = thePatTauProducer.tauIDSources
    for tancInputInfo in produceTancMVAInputDiscriminators.discriminants:
        name = tancInputInfo.name.value()
        if hasattr(tancInputInfo, "indices"):
            # multiple input
            for index in tancInputInfo.indices:
                collectionName = name + str(index)
                setattr(patTauIDConfig, "TaNC"+collectionName, cms.InputTag(
                    "produceTancMVAInputDiscriminators", collectionName))
        else:
            # single input
            setattr(patTauIDConfig, "TaNC"+name, cms.InputTag(
                    "produceTancMVAInputDiscriminators", name))

if __name__ == "__main__":
    class Dummy:
        pass
    test = Dummy()
    test.tauIDSources = cms.PSet()
    loadMVAInputsIntoPatTauDiscriminants(test)
    print test.tauIDSources










