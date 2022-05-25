AlCaRecoMatrix = {
                  "ALCALumiPixelsCountsExpress" : "AlCaPCCRandom",
                  "AlCaLumiPixelsCountsPrompt"  : "AlCaPCCZeroBias+RawPCCProducer",
                  # These two (AlCaPhiSym, AlCaP0) cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                  "AlCaPhiSym"                  : "",
                  "AlCaP0"                      : "",
                  "ALCAPPS"                     : "PPSCalMaxTracks", # Express producer
                  "AlCaPPS"                     : "PPSCalMaxTracks", # Prompt  producer
                  "Charmonium"                  : "TkAlJpsiMuMu",
                  "Commissioning"               : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                  "Cosmics"                     : "SiPixelCalCosmics+SiStripCalCosmics+TkAlCosmics0T+MuAlGlobalCosmics",
                  "DoubleMuon"                  : "TkAlZMuMu+TkAlDiMuonAndVertex+MuAlCalIsolatedMu",
                  "DoubleMuParked"              : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu",
                  "EGamma"                      : "EcalESAlign+EcalUncalWElectron+EcalUncalZElectron+HcalCalIsoTrkProducerFilter+HcalCalIterativePhiSym",
                  "Express"                     : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+SiPixelCalZeroBias",
                  "ExpressAlignment"            : "TkAlMinBias",
                  "ExpressCosmics"              : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+SiPixelCalZeroBias",
                  "HcalNZS"                     : "HcalCalMinBias",
                  "HLTPhysics"                  : "TkAlMinBias",
                  "JetHT"                       : "HcalCalIsoTrkProducerFilter+TkAlMinBias",
                  "MET"                         : "HcalCalNoise",
                  "MinimumBias"                 : "SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias",
                  "MuOnia"                      : "TkAlUpsilonMuMu",
                  "MuOniaParked"                : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                  "NoBPTX"                      : "TkAlCosmicsInCollisions",
                  "SingleMuon"                  : "SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+MuAlCalIsolatedMu+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonProducerFilter",
                  "StreamExpress"               : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+SiPixelCalZeroBias+SiPixelCalSingleMuon+PPSCalTrackBasedSel",
                  "StreamExpressHI"             : "SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+SiPixelCalZeroBias",
                  # These (TestEnablesTracker, TestEnablesEcalHcal) are in the AlCaRecoMatrix, but no RelVals are produced
                  # 'TestEnablesTracker'        : 'TkAlLAS'
                  # 'TestEnablesEcalHcal'       : 'HcalCalPedestal'
                  "ZeroBias"                    : "SiStripCalZeroBias+TkAlMinBias+SiStripCalMinBias",
                  }


def buildList(pdList, matrix):
    """Takes a list of primary datasets (PDs) and the AlCaRecoMatrix (a dictinary) and returns a string with all the AlCaRecos for the selected PDs separated by the '+' character without duplicates."""
    alCaRecoList = []
    for pd in pdList:
        alCaRecoList.extend(matrix[pd].split("+"))
    # remove duplicates converting to a set
    alCaRecoList = set(alCaRecoList)
    stringList = ''
    for alCaReco in alCaRecoList:
        if stringList == '':
            stringList += alCaReco
        else:
            stringList += '+'+alCaReco
    return stringList

# Update the lists anytime a new PD is added to the matrix
autoAlca = { 'allForPrompt'         : buildList(['Charmonium', 'Commissioning', 'DoubleMuParked', 'DoubleMuon', 'EGamma', 'HLTPhysics', 'HcalNZS', 'JetHT', 'MET', 'MinimumBias', 'MuOnia', 'MuOniaParked', 'NoBPTX', 'SingleMuon', 'ZeroBias'], AlCaRecoMatrix),
             'allForExpress'        : buildList(['StreamExpress', 'ALCALumiPixelsCountsExpress'], AlCaRecoMatrix),
             'allForExpressHI'      : buildList(['StreamExpressHI'], AlCaRecoMatrix),
             'allForPromptCosmics'  : buildList(['Cosmics'], AlCaRecoMatrix),
             'allForExpressCosmics' : buildList(['ExpressCosmics'], AlCaRecoMatrix) }
autoAlca.update(AlCaRecoMatrix)

# list of AlCa sequences that have modules that do not support concurrent LuminosityBlocks
AlCaNoConcurrentLumis = [
    'PromptCalibProd',                 # AlcaBeamSpotProducer
    'PromptCalibProdSiPixelAli',       # AlignmentProducerAsAnalyzer, MillePedeFileConverter
    'PromptCalibProdBeamSpotHP',       # AlcaBeamSpotProducer
    'PromptCalibProdBeamSpotHPLowPU',  # AlcaBeamSpotProducer
]
