import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltHIGmonitoring = topMonitoring.clone(
    #FolderName = 'HLT/Higgs/default/',
    FolderName = 'HLT/HIG/default/',

    histoPSet = dict(
        lsPSet = dict(
            nbins =  2500,
            xmin  =  0.,
            xmax  = 2500.),
        
        metPSet = dict(
            nbins = 30,
            xmin  =  0 ,
            xmax  =  300),

        ptPSet = dict(
            nbins =   60 ,
            xmin  =   0 ,
            xmax  =  300),

        phiPSet = dict(
            nbins =  32 ,
            xmin  = -3.2,
            xmax  =  3.2),

        etaPSet = dict(
             nbins =  30 ,
             xmin  =  -3.0,
             xmax  =  3.0),

        htPSet = dict(
            nbins =   60 ,
            xmin  =   0 ,
            xmax  =  600),
        # Marina
        csvPSet = dict(
            nbins =  50 ,
            xmin  =  0.0 ,
            xmax  =  1.0 ),

        DRPSet = dict(
            nbins =  60 ,
            xmin  =  0.0 ,
            xmax  = 6.0),

        invMassPSet = dict(
            nbins =  40 ,
             xmin  =  0.0 ,
             xmax  =  80.0),

         MHTPSet = dict(
            nbins =  80 ,
            xmin  =   60 ,
            xmax  =  300),

         #MET and HT binning
         metBinning = [0,20,40,60,80,100,125,150,175,200],
         HTBinning  = [0,20,40,60,80,100,125,150,175,200,300,400,500,700],
         #Eta binning
         eleEtaBinning = [-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5],
         jetEtaBinning = [-4.7,-3.2,-3.0,-2.5,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.5,3.0,3.2,4.7],
         muEtaBinning  = [-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4],
         #pt binning
         elePtBinning = [0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400],
         jetPtBinning = [0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400],
         muPtBinning  = [0,3,5,8,10,15,20,25,30,40,50,60,80,120,200,400],
         #Eta binning 2D
         eleEtaBinning2D = [-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5],
         jetEtaBinning2D = [-4.7,-3.2,-3.0,-2.5,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.5,3.0,3.2,4.7],
         muEtaBinning2D  = [-2.4,-2.1,-1.7,-1.2,-0.9,-0.6,-0.3,-0.1,0,0.1,0.3,0.6,0.9,1.2,1.7,2.1,2.4],

         #pt binning 2D
         elePtBinning2D = [0,15,20,30,40,60,80,100,200,400],
         jetPtBinning2D = [0,15,20,30,40,60,80,100,200,400],
         muPtBinning2D  = [0,15,20,30,40,60,80,100,200,400],
         #HT and phi binning 2D
         HTBinning2D  = [0,20,40,70,100,150,200,400,700],
         phiBinning2D = [-3.1416,-2.5132,-1.8849,-1.2566,-0.6283,0,0.6283,1.2566,1.8849,2.5132,3.1416],
    ),
    applyLeptonPVcuts = True,
    leptonPVcuts = dict(
         dxy =  0.5,
         dz  =  1.),

    met       = "pfMet", # pfMet
    jets      = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !
    vertices  = "offlinePrimaryVertices",

    HTdefinition = 'pt>30 & abs(eta)<2.5',
    leptJetDeltaRmin = 0.4,
    eleSelection =  'pt > 7. && abs(eta) < 2.5', 
    muoSelection =  'pt > 5 &&  abs(eta) < 2.4 && (isGlobalMuon || (isTrackerMuon && numberOfMatches>0)) && muonBestTrackType != 2',
    vertexSelection = '!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2',

    nmuons     = 0,
    nelectrons = 0,
    njets      = 0,


    numGenericTriggerEventPSet = dict(
             andOr         =  False,
             andOrHlt      = True,# True:=OR; False:=AND
             verbosityLevel = 1,
             hltInputTag   = "TriggerResults::HLT",
             errorReplyHlt =  False ),

    denGenericTriggerEventPSet = dict(
            andOr         = False,
            dcsInputTag   = "scalersRawToDigi",
            dcsRecordInputTag = "onlineMetaDataDigis",
            dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
            andOrDcs      =  False,
            errorReplyDcs =  True,
            verbosityLevel = 1)
)

