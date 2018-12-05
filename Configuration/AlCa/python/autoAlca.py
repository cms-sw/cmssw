AlCaRecoMatrix = {"AlCaLumiPixels" : "AlCaPCCZeroBias+AlCaPCCRandom",
                  "Charmonium"     : "TkAlJpsiMuMu",
                  "Commissioning"  : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                  "Cosmics"        : "TkAlCosmics0T+MuAlGlobalCosmics+DtCalibCosmics",
                  "DoubleEG"       : "EcalCalZElectron+EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter",
                  "DoubleMuon"     : "TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+DtCalib",
                  # New PD in 2018 to replace SinglePhoton SingleElectron and DoubleEG in 2017
                  "EGamma"         : "EcalESAlign+EcalUncalWElectron+EcalUncalZElectron+HcalCalIsoTrkFilter+HcalCalIterativePhiSym",
                  "HLTPhysics"     : "TkAlMinBias",
                  "JetHT"          : "HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter",
                  "MinimumBias"    : "SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias",
                  "MuOnia"         : "TkAlUpsilonMuMu",
                  "NoBPTX"         : "TkAlCosmicsInCollisions",
                  "SingleElectron" : "EcalUncalWElectron+EcalUncalZElectron+HcalCalIterativePhiSym+EcalESAlign",
                  "SingleMuon"     : "TkAlMuonIsolated+DtCalib+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonFilter+HcalCalHEMuonFilter",
                  "SinglePhoton"   : "HcalCalGammaJet",
                  "ZeroBias"       : "SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias+AlCaPCCZeroBiasFromRECO", 

                  "Express"        : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalib+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                  "ExpressCosmics" : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+DtCalibCosmics+SiPixelCalZeroBias",
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
                  "SingleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+MuAlZMuMu+HcalCalHO",
                  "DoubleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu",
                  "DoubleMuParked" : "MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu",
                  "MuOniaParked" : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                  "DoubleElectron" : "EcalCalZElectron+EcalUncalZElectron+HcalCalIsoTrkFilter",
                  "StreamExpress" : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalib+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                  "StreamExpressHI" : "SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalibHI+SiPixelCalZeroBias"
                  }

# AlCaReco matrix used in CMSSW releases in 2017
AlCaRecoMatrix2017 = {"AlCaLumiPixels" : "AlCaPCCZeroBias+AlCaPCCRandom",
                      "Charmonium"     : "TkAlJpsiMuMu",
                      "Commissioning"  : "HcalCalIsoTrk+HcalCalIsolatedBunchSelector+TkAlMinBias+SiStripCalMinBias",
                      "Cosmics"        : "TkAlCosmics0T+MuAlGlobalCosmics+DtCalibCosmics",
                      "DoubleEG"       : "EcalCalZElectron+EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter",
                      "DoubleMuon"     : "TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+DtCalib",
                      "HLTPhysics"     : "TkAlMinBias",
                      "JetHT"          : "HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter",
                      "MinimumBias"    : "SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias",
                      "MuOnia"         : "TkAlUpsilonMuMu",
                      "NoBPTX"         : "TkAlCosmicsInCollisions",
                      "SingleElectron" : "EcalUncalWElectron+EcalUncalZElectron+HcalCalIterativePhiSym+EcalESAlign",
                      "SingleMuon"     : "TkAlMuonIsolated+DtCalib+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonFilter+HcalCalHEMuonFilter",
                      "SinglePhoton"   : "HcalCalGammaJet",
                      "ZeroBias"       : "SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias+AlCaPCCZeroBiasFromRECO", 

                      "Express"  : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalib+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                      "ExpressCosmics" : "SiStripPCLHistos+SiStripCalZeroBias+TkAlCosmics0T+DtCalibCosmics+SiPixelCalZeroBias",
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
                      "SingleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+MuAlZMuMu+HcalCalHO",
                      "DoubleMu" : "MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu",
                      "DoubleMuParked" : "MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu",
                      "MuOniaParked" : "TkAlJpsiMuMu+TkAlUpsilonMuMu",
                      "DoubleElectron" : "EcalCalZElectron+EcalUncalZElectron+HcalCalIsoTrkFilter",
                      "StreamExpress" : "SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalib+Hotline+LumiPixelsMinBias+SiPixelCalZeroBias",
                      "StreamExpressHI" : "SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAAG+DtCalibHI+SiPixelCalZeroBias"
}

# this matrix will be used for the legacy reprocessing of the 2016 2016B-H dataset;
# with the exception of ZeroBias, it was also used for the 23Sept16 reprocessing of 2016B-G 
AlCaRecoMatrixRereco = {'AlCaLumiPixels' : 'LumiPixels',
                        'Charmonium'     : 'TkAlJpsiMuMu',
                        'Commissioning'  : 'TkAlMinBias+SiStripCalMinBias+HcalCalIsoTrk+HcalCalIsolatedBunchSelector',
                        'Cosmics'        : 'TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DtCalibCosmics',
                        'DoubleEG'       : 'EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter',
                        'DoubleElectron' : 'EcalUncalZElectron+HcalCalIsoTrkFilter',
                        'DoubleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu+MuAlZMuMu+TkAlZMuMu+TkAlJpsiMuMu+TkAlUpsilonMuMu+HcalCalIsoTrkFilter',
                        'DoubleMuon'     : 'TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+DtCalib',
                        'DoubleMuParked' : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu',
                        'HLTPhysics'     : 'SiStripCalMinBias+TkAlMinBias+HcalCalIsoTrkFilter',
                        'JetHT'          : 'HcalCalDijets+HcalCalIsoTrkFilter+HcalCalIsolatedBunchFilter',
                        'NoBPTX'         : 'TkAlCosmicsInCollisions',
                        'MET'            : 'HcalCalNoise',
                        'MinimumBias'    : 'SiStripCalMinBias+TkAlMinBias',
                        'MuOnia'         : 'TkAlUpsilonMuMu',
                        'SingleElectron' : 'EcalUncalWElectron+EcalUncalZElectron+EcalESAlign+HcalCalIterativePhiSym+HcalCalIsoTrkFilter',
                        'SingleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+MuAlZMuMu+HcalCalHO',
                        'SingleMuon'     : 'TkAlMuonIsolated+DtCalib+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalIterativePhiSym+HcalCalHO',
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
