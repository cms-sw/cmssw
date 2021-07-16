AlCaRecoMatrix = {"AlCaLumiPixels" : "AlCaPCCZeroBias+AlCaPCCRandom",
                  "Charmonium"     : "TkAlJpsiMuMu",
                  "Commissioning"  : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                  "Cosmics"        : "SiPixelCalCosmics+SiStripCalCosmics+TkAlCosmics0T+MuAlGlobalCosmics",
                  "DoubleEG"       : "EcalCalZElectron+EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter",
                  "DoubleMuon"     : "TkAlZMuMu+TkAlDiMuonAndVertex+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalLowPUHBHEMuonFilter",
                  # New PD in 2018 to replace SinglePhoton SingleElectron and DoubleEG in 2017
                  "EGamma"         : "EcalESAlign+EcalUncalWElectron+EcalUncalZElectron+HcalCalIsoTrkFilter+HcalCalIterativePhiSym",
                  "HLTPhysics"     : "TkAlMinBias+HcalCalIterativePhiSym",
                  "JetHT"          : "HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter+TkAlMinBias",
                  "MinimumBias"    : "SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias",
                  "MuOnia"         : "TkAlUpsilonMuMu",
                  "NoBPTX"         : "TkAlCosmicsInCollisions",
                  "SingleElectron" : "EcalUncalWElectron+EcalUncalZElectron+HcalCalIterativePhiSym+EcalESAlign",
                  "SingleMuon"     : "SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonFilter+HcalCalHEMuonFilter",
                  "SinglePhoton"   : "HcalCalGammaJet",
                  "ZeroBias"       : "SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias+AlCaPCCZeroBiasFromRECO", 

                  "Express"        : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                  "ExpressCosmics" : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+SiPixelCalZeroBias",
                  "ExpressAlignment":"TkAlMinBias",
                  # Used for new PCC PCL introduced in 2018
                  "AlcaLumiPixelsExpress":"AlCaPCCRandom",
                  # These two cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                  "AlCaPhiSym"     : "",
                  "AlCaP0"         : "",
                  # ---------------------------------------------------------------------------------------------------------------------------
                  "HcalNZS"        : "HcalCalMinBias",
                  # This is in the AlCaRecoMatrix, but no RelVals are produced
                  # 'TestEnablesTracker'  : 'TkAlLAS'
                  # 'TestEnablesEcalHcal' : 'HcalCalPedestal'
                  "MET" : "HcalCalNoise",
                  "SingleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+MuAlZMuMu+HcalCalHO",
                  "DoubleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu",
                  "DoubleMuParked" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu",
                  "MuOniaParked" : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                  "DoubleElectron" : "EcalCalZElectron+EcalUncalZElectron+HcalCalIsoTrkFilter",
                  "StreamExpress" : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                  "StreamExpressHI" : "SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+SiPixelCalZeroBias"
                  }

# AlCaReco matrix used in CMSSW releases in 2017
AlCaRecoMatrix2017 = {"AlCaLumiPixels" : "AlCaPCCZeroBias+AlCaPCCRandom",
                      "Charmonium"     : "TkAlJpsiMuMu",
                      "Commissioning"  : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                      "Cosmics"        : "TkAlCosmics0T+MuAlGlobalCosmics",
                      "DoubleEG"       : "EcalCalZElectron+EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter",
                      "DoubleMuon"     : "TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu",
                      "HLTPhysics"     : "TkAlMinBias",
                      "JetHT"          : "HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter",
                      "MinimumBias"    : "SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias",
                      "MuOnia"         : "TkAlUpsilonMuMu",
                      "NoBPTX"         : "TkAlCosmicsInCollisions",
                      "SingleElectron" : "EcalUncalWElectron+EcalUncalZElectron+HcalCalIterativePhiSym+EcalESAlign",
                      "SingleMuon"     : "TkAlMuonIsolated+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonFilter+HcalCalHEMuonFilter",
                      "SinglePhoton"   : "HcalCalGammaJet",
                      "ZeroBias"       : "SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias+AlCaPCCZeroBiasFromRECO", 

                      "Express"  : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                      "ExpressCosmics" : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+SiPixelCalZeroBias",
                      "ExpressAlignment":"TkAlMinBias",
                      # These two cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                      "AlCaPhiSym"     : "",
                      "AlCaP0"         : "",
                      # ---------------------------------------------------------------------------------------------------------------------------
                      "HcalNZS"        : "HcalCalMinBias",
                      # This is in the AlCaRecoMatrix, but no RelVals are produced
                      # 'TestEnablesTracker'  : 'TkAlLAS'
                      # 'TestEnablesEcalHcal' : 'HcalCalPedestal'
                      "MET" : "HcalCalNoise",
                      "SingleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+MuAlZMuMu+HcalCalHO",
                      "DoubleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu",
                      "DoubleMuParked" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu",
                      "MuOniaParked" : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                      "DoubleElectron" : "EcalCalZElectron+EcalUncalZElectron+HcalCalIsoTrkFilter",
                      "StreamExpress" : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                      "StreamExpressHI" : "SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+SiPixelCalZeroBias"
}

# this matrix will be used for the legacy reprocessing of the 2016 2016B-H dataset;
# with the exception of ZeroBias, it was also used for the 23Sept16 reprocessing of 2016B-G 
AlCaRecoMatrixRereco = {'AlCaLumiPixels' : 'LumiPixels',
                        'Charmonium'     : 'TkAlJpsiMuMu',
                        'Commissioning'  : 'TkAlMinBias+SiStripCalMinBias+HcalCalIsoTrk+HcalCalIsolatedBunchSelector',
                        'Cosmics'        : 'SiPixelCalCosmics+TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics',
                        'DoubleEG'       : 'EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter',
                        'DoubleElectron' : 'EcalUncalZElectron+HcalCalIsoTrkFilter',
                        'DoubleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu+MuAlZMuMu+TkAlZMuMu+TkAlJpsiMuMu+TkAlUpsilonMuMu+HcalCalIsoTrkFilter',
                        'DoubleMuon'     : 'TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu',
                        'DoubleMuParked' : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlZMuMu',
                        'HLTPhysics'     : 'SiStripCalMinBias+TkAlMinBias+HcalCalIsoTrkFilter+HcalCalIterativePhiSym',
                        'JetHT'          : 'HcalCalDijets+HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter',
                        'NoBPTX'         : 'TkAlCosmicsInCollisions',
                        'MET'            : 'HcalCalNoise',
                        'MinimumBias'    : 'SiStripCalMinBias+TkAlMinBias',
                        'MuOnia'         : 'TkAlUpsilonMuMu',
                        'SingleElectron' : 'EcalUncalWElectron+EcalUncalZElectron+EcalESAlign+HcalCalIterativePhiSym+HcalCalIsoTrkFilter',
                        'SingleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+MuAlZMuMu+HcalCalHO',
                        'SingleMuon'     : 'SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalIterativePhiSym+HcalCalHO',
                        'SinglePhoton'   : 'HcalCalGammaJet',
                        'ZeroBias'       : 'SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias+SiStripCalMinBiasAfterAbortGap',
                        'HcalNZS'        : 'HcalCalMinBias'
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
autoAlca = { 'allForPrompt'         : buildList(['Charmonium', 'Commissioning', 'DoubleEG', 'DoubleElectron', 'DoubleMu', 'DoubleMuParked', 'DoubleMuon', 'HLTPhysics', 'HcalNZS', 'JetHT', 'MET', 'MinimumBias', 'MuOnia', 'MuOniaParked', 'NoBPTX' , 'SingleElectron', 'SingleMu', 'SingleMuon', 'SinglePhoton', 'ZeroBias'], AlCaRecoMatrix),
             'allForExpress'        : buildList(['StreamExpress'], AlCaRecoMatrix),
             'allForExpressHI'      : buildList(['StreamExpressHI'], AlCaRecoMatrix),
             'allForPromptCosmics'  : buildList(['Cosmics'], AlCaRecoMatrix),
             'allForExpressCosmics' : buildList(['ExpressCosmics'], AlCaRecoMatrix) }
autoAlca.update(AlCaRecoMatrix)
