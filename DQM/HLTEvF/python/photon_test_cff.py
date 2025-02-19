import FWCore.ParameterSet.Config as cms

PhotonIsoEt10SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(

#get these parameters from the HLT tables for each release

#iso single photon
    

#non-iso Et 10

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltPhotonEcalNonIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltNonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),
    PtMax = cms.untracked.double(200.0)
)


PhotonIsoEt15SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(

#non-iso Et 15

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),
    PtMax = cms.untracked.double(200.0)
)

PhotonIsoEt20SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(


#non-iso Et 20

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),
    PtMax = cms.untracked.double(200.0)
)

PhotonIsoEt25SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(


#non-iso Et 25

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),

    PtMax = cms.untracked.double(200.0)
)

PhotonIsoEt30SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(

#move me back please
#iso single photon (et30)
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltPhotonEcalNonIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltSingleEgammaHcalNonIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltPhotonNonIsoTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),

    PtMax = cms.untracked.double(200.0)
)

PhotonNonIsoEt15SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(

        
#non-iso et 15


        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),

    PtMax = cms.untracked.double(200.0)
)

PhotonNonIsoEt25SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(

#non-iso Et 25
        
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),

    PtMax = cms.untracked.double(200.0)
)


PhotonNonIsoEt40SourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(


#non-iso Single Photon

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),


#    disableROOToutput = cms.untracked.bool(False),
    PtMax = cms.untracked.double(200.0)
)

PhotonIsoDoubleSourcePlots = cms.EDAnalyzer("HLTMonPhotonSource",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(False),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(


#iso double photon

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltPhotonEcalNonIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltSingleEgammaHcalNonIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltPhotonNonIsoTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
),


#    disableROOToutput = cms.untracked.bool(False),
    PtMax = cms.untracked.double(200.0)
)


#iso et30 (default)

PhotonIsoEt30ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoEt30SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter"),
        cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter"),
        cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"),
        cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter")
        

    )
)



#iso et10
PhotonIsoEt10ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoEt10SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoSinglePhotonEt10EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonEt10HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter")
        

    )
)

#iso et15
PhotonIsoEt15ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoEt15SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter")
        

    )
)

#iso et20
PhotonIsoEt20ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoEt20SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter")
        

    )
)

#iso et25
PhotonIsoEt25ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoEt25SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter")
        

    )
)

#noniso et15
PhotonNonIsoEt15ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonNonIsoEt15SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter")
        

    )
)

#noniso et25
PhotonNonIsoEt25ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonNonIsoEt25SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter")
        

    )
)

#noniso et40 (default)
PhotonNonIsoEt40ClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonNonIsoEt40SourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"),
        cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter")
        

    )
)

#iso double
PhotonIsoDoubleClientPlots = cms.EDAnalyzer("HLTMonPhotonClient",
    outputFile = cms.untracked.string('./PhotonDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("PhotonIsoDoubleSourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter"),
        cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter"),
        cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter"),
        cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter")
        

    )
)




egammaPhotonDQMpath = cms.Path(PhotonIsoEt30SourcePlots * PhotonIsoEt30ClientPlots * PhotonIsoEt10SourcePlots* PhotonIsoEt10ClientPlots* PhotonIsoEt15SourcePlots* PhotonIsoEt15ClientPlots* PhotonIsoEt20SourcePlots* PhotonIsoEt20ClientPlots* PhotonIsoEt25SourcePlots* PhotonIsoEt25ClientPlots* PhotonNonIsoEt15SourcePlots* PhotonNonIsoEt15ClientPlots* PhotonNonIsoEt25SourcePlots* PhotonNonIsoEt25ClientPlots* PhotonNonIsoEt40SourcePlots* PhotonNonIsoEt40ClientPlots * PhotonIsoDoubleSourcePlots * PhotonIsoDoubleClientPlots)

#egammaPhotonDQMpath = cms.Path(PhotonIsoEt30SourcePlots*PhotonClientPlots)
