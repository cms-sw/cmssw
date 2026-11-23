import FWCore.ParameterSet.Config as cms

# Used by testSplitterSendsOnlyConsumedProducts.sh. Offloads fromSource,
# viaSourceAlias, fromLocalOther, viaLocalAlias, thingsViaMixed, remoteSum and
# remoteViaAlias to a remote process.
#
# The splitter should send, in each direction, only the data products that are
# consumed on the other side, matched by type and product instance name.
# EDAliases are not sent themselves: the data products they point to are sent
# under their original labels, and the EDAliases are recreated on the other side.
#
# The data products:
#    - the Source produces an edmtest::IntProduct with value 4.
#    - localSum produces two edmtest::IntProduct, with instances "" and "other",
#      both with value 3.
#    - remoteSum adds 4 + 4 + 3 + 3 = 14.
#
# The local process should send to the remote one:
#    - ("source", ""): read directly by fromSource, and through aliasOfSource by
#      viaSourceAlias.
#    - ("localSum", "other"): read directly by fromLocalOther, and through
#      aliasOfLocalSum by viaLocalAlias.
#    - ("things", ""): read through the mixed EDAlias by thingsViaMixed.
#
# The local process should not send:
#    - localProducer: read only by localSum, which stays local.
#    - ("localSum", ""): no offloaded module reads it.
#
# The remote process should send back to the local one:
#    - ("remoteSum", "other"): read directly by checkRemoteSum, and through
#      aliasOfRemoteSum by checkAliasOfRemoteSum.
#    - ("remoteViaAlias", ""): read by checkRemoteViaAlias.
#
# The remote process should not send back:
#    - ("remoteSum", "") and ("remoteViaAlias", "other"): no local module reads
#      them.
#    - the results of fromSource, viaSourceAlias, fromLocalOther, viaLocalAlias
#      and thingsViaMixed: they are read only in the remote process, or not at all.
#
# The mixed EDAlias points to data products of both the Source and things, but
# thingsViaMixed reads only the one of things. When thingsViaMixed is the only
# offloaded module, only things should be sent, and nothing of the Source.
#
# The local IntTestAnalyzers check that they read 14. A data product that is not
# sent, or is sent under the wrong label, makes them fail or throw
# ProductNotFound.

process = cms.Process("SPLITPRODUCTS")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("IntSource")
process.maxEvents.input = 10

# local producers
process.localProducer = cms.EDProducer("IntProducer", ivalue=cms.int32(3))
process.localSum = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("localProducer")))
process.things = cms.EDProducer("ThingProducer")

# EDAliases of local data products
process.aliasOfSource = cms.EDAlias(source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))))
process.aliasOfLocalSum = cms.EDAlias(
    localSum=cms.VPSet(
        cms.PSet(
            type=cms.string("edmtestIntProduct"),
            fromProductInstance=cms.string("other"),
            toProductInstance=cms.string("fromOther"),
        )
    )
)
process.mixed = cms.EDAlias(
    source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct"))),
    things=cms.VPSet(cms.PSet(type=cms.string("edmtestThings"), fromProductInstance=cms.string(""))),
)

# offloaded modules
process.fromSource = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("source")))
process.viaSourceAlias = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSource")))
process.fromLocalOther = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("localSum", "other")))
process.viaLocalAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfLocalSum", "fromOther"))
)
process.thingsViaMixed = cms.EDProducer("OtherThingProducer", thingTag=cms.InputTag("mixed"))
process.remoteSum = cms.EDProducer(
    "AddIntsProducer",
    labels=cms.VInputTag("fromSource", "viaSourceAlias", "fromLocalOther", "viaLocalAlias"),
)
process.aliasOfRemoteSum = cms.EDAlias(
    remoteSum=cms.VPSet(
        cms.PSet(
            type=cms.string("edmtestIntProduct"),
            fromProductInstance=cms.string("other"),
            toProductInstance=cms.string(""),
        )
    )
)
process.remoteViaAlias = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfRemoteSum")))

# local checks
process.checkRemoteSum = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("remoteSum", "other"), valueMustMatch=cms.untracked.int32(14)
)
process.checkAliasOfRemoteSum = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("aliasOfRemoteSum"), valueMustMatch=cms.untracked.int32(14)
)
process.checkRemoteViaAlias = cms.EDAnalyzer(
    "IntTestAnalyzer", moduleLabel=cms.untracked.InputTag("remoteViaAlias"), valueMustMatch=cms.untracked.int32(14)
)

process.path = cms.Path(
    process.localProducer
    + process.localSum
    + process.things
    + process.fromSource
    + process.viaSourceAlias
    + process.fromLocalOther
    + process.viaLocalAlias
    + process.thingsViaMixed
    + process.remoteSum
    + process.remoteViaAlias
)
process.checks = cms.EndPath(process.checkRemoteSum + process.checkAliasOfRemoteSum + process.checkRemoteViaAlias)
