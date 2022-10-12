AlCaRecoMatrix = {
                  "ALCALumiPixelsCountsExpress" : "AlCaPCCRandom",
                  "AlCaLumiPixelsCountsPrompt"  : "AlCaPCCZeroBias+RawPCCProducer",
                  # These two (AlCaPhiSym, AlCaP0) cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                  "AlCaPhiSym"                  : "",
                  "AlCaP0"                      : "",
                  "ALCAPPSExpress"              : "PPSCalMaxTracks", # Express producer
                  "AlCaPPSPrompt"               : "PPSCalMaxTracks", # Prompt  producer
                  "Commissioning"               : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                  "Cosmics"                     : "SiPixelCalCosmics+SiStripCalCosmics+TkAlCosmics0T+MuAlGlobalCosmics",
                  "EGamma"                      : "EcalESAlign+EcalUncalWElectron+EcalUncalZElectron+HcalCalIsoTrkProducerFilter+HcalCalIterativePhiSym",
                  "Express"                     : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+SiPixelCalZeroBias",
                  "ExpressAlignment"            : "TkAlMinBias",
                  "ExpressCosmics"              : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+SiPixelCalZeroBias",
                  "HcalNZS"                     : "HcalCalMinBias",
                  "HLTPhysics"                  : "TkAlMinBias",
                  "JetMET"                      : "HcalCalIsoTrkProducerFilter+TkAlJetHT+HcalCalNoise",
                  "Muon"                        : "TkAlZMuMu+TkAlDiMuonAndVertex+MuAlCalIsolatedMu+SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonProducerFilter",
                  "NoBPTX"                      : "TkAlCosmicsInCollisions",
                  "ParkingDoubleMuonLowMass"    : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                  "StreamExpress"               : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+SiPixelCalZeroBias+SiPixelCalSingleMuon",
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
autoAlca = { 'allForPrompt'         : buildList(['Commissioning', 'EGamma', 'HLTPhysics', 'HcalNZS', 'JetMET', 'Muon', 'NoBPTX', 'ParkingDoubleMuonLowMass', 'ZeroBias'], AlCaRecoMatrix),
             'allForExpress'        : buildList(['StreamExpress'], AlCaRecoMatrix),
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
