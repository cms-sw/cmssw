AlCaRecoMatrix = {'AlCaLumiPixels' : 'LumiPixels',
                  'Charmonium'     : 'TkAlJpsiMuMu',
                  'Commissioning'  : 'HcalCalIsoTrk',
                  'Cosmics'        : 'TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DtCalibCosmics',
                  'DoubleEG'       : 'EcalCalZElectron+EcalUncalZElectron',
                  'DoubleElectron' : 'EcalCalZElectron+EcalUncalZElectron',
                  'DoubleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu',
                  'DoubleMuon'     : 'TkAlZMuMu+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu+DtCalib',
                  'DoubleMuParked' : 'MuAlCalIsolatedMu+MuAlOverlaps+DtCalib+TkAlZMuMu',
                  'HLTPhysics'     : 'SiStripCalMinBias+TkAlMinBias',
                  'JetHT'          : 'HcalCalDijets',
                  'MET'            : 'HcalCalNoise',
                  'MinimumBias'    : 'SiStripCalMinBias+TkAlMinBias',
                  'MuOnia'         : 'TkAlJpsiMuMu+TkAlUpsilonMuMu',
                  'MuOniaParked'   : 'TkAlJpsiMuMu+TkAlUpsilonMuMu',
                  'SingleElectron' : 'EcalCalWElectron+EcalUncalWElectron+EcalCalZElectron+EcalUncalZElectron',
                  'SingleMu'       : 'MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+MuAlZMuMu',
                  'SingleMuon'     : 'TkAlMuonIsolated+DtCalib+MuAlCalIsolatedMu+MuAlOverlaps+MuAlZMuMu',
                  'SinglePhoton'   : 'HcalCalGammaJet',
                  'StreamExpress'  : 'SiStripCalZeroBias+TkAlMinBias+SiStripPCLHistos+SiStripCalMinBias+DtCalib+Hotline',
                  'ExpressCosmics' : 'SiStripCalZeroBias+TkAlCosmics0T',
                  # These two cannot run on RAW, they are just meant to run on the dedicated AlcaRAW so they do not enter the allForPrompt list
                  'AlCaP0'         : '',
                  # ---------------------------------------------------------------------------------------------------------------------------
                  'HcalNZS'        : 'HcalCalMinBias'
                  # This is in the AlCaRecoMatrix, but no RelVals are produced
                  # 'TestEnablesTracker' : 'TkAlLAS'
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
autoAlca = { 'allForPrompt'         : buildList(['Charmonium', 'Commissioning', 'DoubleEG', 'DoubleElectron', 'DoubleMu', 'DoubleMuParked', 'DoubleMuon', 'HLTPhysics', 'HcalNZS', 'JetHT', 'MET', 'MinimumBias', 'MuOnia', 'MuOniaParked', 'SingleElectron', 'SingleMu', 'SingleMuon', 'SinglePhoton'], AlCaRecoMatrix),
             'allForExpress'        : buildList(['StreamExpress'], AlCaRecoMatrix),
             'allForPromptCosmics'  : buildList(['Cosmics'], AlCaRecoMatrix),
             'allForExpressCosmics' : buildList(['ExpressCosmics'], AlCaRecoMatrix) }
autoAlca.update(AlCaRecoMatrix)
