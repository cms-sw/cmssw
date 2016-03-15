AlCaRecoMatrix = {'AlCaLumiPixels' : 'LumiPixels',
                  'Charmonium'     : 'TkAlJpsiMuMu',
                  'Commissioning'  : 'HcalCalIsoTrk',
                  'Cosmics'        : 'TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DtCalibCosmics',
                  'DoubleEG'       : 'EcalCalZElectron+EcalUncalZElectron+HcalCalIterativePhiSym+HcalCalIsoTrkFilter',
                  'DoubleElectron' : 'EcalCalZElectron+EcalUncalZElectron+HcalCalIsoTrkFilter',
                  'DoubleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu',
                  'DoubleMuon'     : 'TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+DtCalib',
                  'DoubleMuParked' : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu',
                  'HLTPhysics'     : 'SiStripCalMinBias+TkAlMinBias+HcalCalIsoTrkFilter',
                  'JetHT'          : 'HcalCalDijets+HcalCalIsoTrkFilter',
                  'MET'            : 'HcalCalNoise',
                  'MinimumBias'    : 'SiStripCalMinBias+TkAlMinBias',
                  'MuOnia'         : 'TkAlJpsiMuMu+TkAlUpsilonMuMu',
                  'MuOniaParked'   : 'TkAlJpsiMuMu+TkAlUpsilonMuMu',
                  'SingleElectron' : 'EcalCalWElectron+EcalUncalWElectron+EcalCalZElectron+EcalUncalZElectron+EcalESAlign+HcalCalIterativePhiSym',
                  'SingleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+MuAlZMuMu+HcalCalHO',
                  'SingleMuon'     : 'TkAlMuonIsolated+DtCalib+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+HcalCalIterativePhiSym',
                  'SinglePhoton'   : 'HcalCalGammaJet',
                  'ZeroBias'       : 'SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+SiStripCalMinBias', 
                  'StreamExpress'  : 'SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAfterAbortGap+DtCalib+Hotline+LumiPixelsMinBias',
                  'StreamExpressHI': 'SiStripCalZeroBias+TkAlMinBiasHI+SiStripPCLHistos+SiStripCalMinBias+SiStripCalMinBiasAfterAbortGap+DtCalibHI',
                  'ExpressCosmics' : 'SiStripCalZeroBias+TkAlCosmics0T',
                  # These two cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                  'AlCaP0'         : '',
                  # ---------------------------------------------------------------------------------------------------------------------------
                  'HcalNZS'        : 'HcalCalMinBias'
                  # This is in the AlCaRecoMatrix, but no RelVals are produced
                  # 'TestEnablesTracker'  : 'TkAlLAS'
                  # 'TestEnablesEcalHcal' : 'HcalCalPedestal'
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
autoAlca = { 'allForPrompt'         : buildList(['Charmonium', 'Commissioning', 'DoubleEG', 'DoubleElectron', 'DoubleMu', 'DoubleMuParked', 'DoubleMuon', 'HLTPhysics', 'HcalNZS', 'JetHT', 'MET', 'MinimumBias', 'MuOnia', 'MuOniaParked', 'SingleElectron', 'SingleMu', 'SingleMuon', 'SinglePhoton', 'ZeroBias'], AlCaRecoMatrix),
             'allForExpress'        : buildList(['StreamExpress'], AlCaRecoMatrix),
             'allForExpressHI'      : buildList(['StreamExpressHI'], AlCaRecoMatrix),
             'allForPromptCosmics'  : buildList(['Cosmics'], AlCaRecoMatrix),
             'allForExpressCosmics' : buildList(['ExpressCosmics'], AlCaRecoMatrix) }
autoAlca.update(AlCaRecoMatrix)
