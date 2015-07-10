from CMGTools.Utilities.metRecoilCorrection.rootfile_dir import rootfile_dir

def lookup( fileName, stringToFind ):
    '''predicate for identifying samples. could be more solid'''
    if fileName.find( stringToFind )>-1:
        return True
    else:
        return False


recoilfits_leptonic_5X = dict(
    zmm = 'recoilfit_zmm53XRR_2012_njet.root',
    hig = 'recoilfit_higgs53X_20pv_njet.root',
    wjets = 'recoilfit_wjets53X_20pv_njet.root',
    )

recoilfits_hadronic_5X = dict(
    zmm = 'recoilfit_ztt53X_20pv_njet.root',
    hig = 'recoilfit_htt53X_20pv_njet.root',
    wjets = 'recoilfit_wjets53X_20pv_njet.root'
    )

recoilfits_leptonic_4X = dict(
    zmm = 'recoilfit_zmm42X_20pv_njet.root',
    hig = 'recoilfit_higgs42X_20pv_njet.root',
    wjets = 'recoilfit_wjets42X_20pv_njet.root'
    )

recoilfits_hadronic_4X = recoilfits_leptonic_4X 



def fileAndLeg(fileName, is53X, mode=None, channel=None):
    correctFileName = None
    leptonLeg = None
    recoilfits = None
    if is53X:
        if channel == 'di-tau':
            recoilfits = recoilfits_hadronic_5X
        else:
            recoilfits = recoilfits_leptonic_5X
    else:
        if channel == 'di-tau':
            recoilfits = recoilfits_hadronic_4X
        else:
            recoilfits = recoilfits_leptonic_4X
    if mode:
        # forcing a mode
        fileName = mode
    sample = None
    if lookup( fileName, 'DYJets' ) or \
           lookup( fileName, 'DY1Jet' ) or \
           lookup( fileName, 'DY2Jet' ) or \
           lookup( fileName, 'DY3Jet' ) or \
           lookup( fileName, 'DY4Jet' ):
        print '\tENABLED : Z->l tau mode (tau is true)'
        sample = 'zmm'
        leptonLeg = 0
    if lookup( fileName, 'GluGluToHToTauTau' ) or \
           lookup( fileName, 'VBF_HToTauTau' ) or \
           lookup( fileName, 'VBFHToTauTau' ) or \
           lookup( fileName, 'SUSYBBHToTauTau' ) or \
           lookup( fileName, 'SUSYGluGluToHToTauTau' ) or \
           lookup( fileName, 'WH_ZH_TTH_HToTauTau' ):
        print '\tENABLED : Higgs mode (tau is true)'
        sample = 'hig'
        leptonLeg = 0
    elif lookup( fileName, 'WJetsToLNu' ) or \
             lookup( fileName, 'W1Jet' ) or \
             lookup( fileName, 'W2Jets' ) or \
             lookup( fileName, 'W3Jets' ) or \
             lookup( fileName, 'W4Jets' ):
        print '\tENABLED : W+jet mode (tau is fake)'
        sample = 'wjets'
        if channel == 'di-tau':
            leptonLeg = 1 # taking the leading tau
        else:
            leptonLeg = 2 # taking the second leg (lepton for mutau and etau)
    else:
        pass
    if sample is None:
        return None, None
    correctFileName = '/'.join([rootfile_dir,recoilfits[sample]])
    if correctFileName:
        print '\tCorrecting to:', correctFileName
        print '\tLeg number   :',leptonLeg
    return correctFileName, leptonLeg



def basicParameters(is53X):
    fileZmmData = None
    fileZmmMC = None
    correctionType = None
    if is53X:
        print 'picking up 53X recoil fits'
        fileZmmData = rootfile_dir + 'recoilfit_datamm53XRR_2012_njet.root'
        fileZmmMC = rootfile_dir + 'recoilfit_zmm53XRR_2012_njet.root'
        correctionType = 1
    else:
        print 'picking up 44X recoil fits'
        fileZmmData = rootfile_dir + 'recoilfit_datamm42X_20pv_njet.root'
        fileZmmMC = rootfile_dir + 'recoilfit_zmm42X_20pv_njet.root'        
        correctionType = 1
    print '\tZmm data:',fileZmmData
    print '\tZmm MC  :',fileZmmMC
    print '\ttype    :',correctionType
    return fileZmmData, fileZmmMC, correctionType


def setupRecoilCorrection( process, runOnMC, enable=True, is53X=True, mode=None, channel=None):

    print '# setting up recoil corrections:'
    if not runOnMC:
        enable=False

    if not hasattr( process, 'recoilCorMETTauMu')  and \
       not hasattr( process, 'recoilCorMETTauEle') and \
       not hasattr( process, 'recoilCorMETDiTau') :
        print 'recoil correction module not in the path -> DISABLED'
        return False
        
    fileName = process.source.fileNames[0]
    print fileName

    if mode:
        print 'FORCING TO', mode

    fileZmmData, fileZmmMC, correctionType = basicParameters(is53X)

    correctFileName = None
    leptonLeg = None
    if enable:
        correctFileName, leptonLeg = fileAndLeg(fileName, is53X, mode, channel)
        if correctFileName is None:
            enable = False
    if enable:
        if lookup( fileName, 'WH_ZH_TTH_HToTauTau' ):
            # in this case, several bosons, we only want the higgs
            process.genWorZ.select = ['keep status()==3 & pdgId = {h0}']
        if hasattr( process, 'recoilCorMETTauMu'):
            if mode:
                process.recoilCorMETTauMu.force = True
            process.recoilCorMETTauMu.enable = True
            process.recoilCorMETTauMu.fileCorrectTo = correctFileName
            process.recoilCorMETTauMu.leptonLeg = leptonLeg
            process.recoilCorMETTauMu.fileZmmData = fileZmmData
            process.recoilCorMETTauMu.fileZmmMC = fileZmmMC
            process.recoilCorMETTauMu.correctionType = correctionType
        if hasattr( process, 'recoilCorMETTauEle'):
            if mode:
                process.recoilCorMETTauEle.force = True
            process.recoilCorMETTauEle.enable = True
            process.recoilCorMETTauEle.fileCorrectTo = correctFileName
            process.recoilCorMETTauEle.leptonLeg = leptonLeg 
            process.recoilCorMETTauEle.fileZmmData = fileZmmData
            process.recoilCorMETTauEle.fileZmmMC = fileZmmMC
            process.recoilCorMETTauEle.correctionType = correctionType
        if hasattr( process, 'recoilCorMETDiTau'):
            if mode:
                process.recoilCorMETDiTau.force = True
            process.recoilCorMETDiTau.enable = True
            process.recoilCorMETDiTau.fileCorrectTo = correctFileName
            process.recoilCorMETDiTau.leptonLeg = leptonLeg 
            process.recoilCorMETDiTau.fileZmmData = fileZmmData
            process.recoilCorMETDiTau.fileZmmMC = fileZmmMC
            process.recoilCorMETDiTau.correctionType = correctionType
    else:
        print '\tDISABLED'
        if runOnMC:
            process.metRecoilCorrectionInputSequence.remove( process.genWorZ ) 
        if hasattr( process, 'recoilCorMETTauMu'):
            process.recoilCorMETTauMu.enable = False
        if hasattr( process, 'recoilCorMETTauEle'):
            process.recoilCorMETTauEle.enable = False
        if hasattr( process, 'recoilCorMETDiTau'):
            process.recoilCorMETDiTau.enable = False
            

if __name__ == '__main__':

    import sys
    from PhysicsTools.Heppy.utils.cmsswRelease import isNewerThan

    data, mc, type = basicParameters(isNewerThan('CMSSW_5_2_X'))
    
    for line in sys.stdin:
        print 
        line = line.rstrip()
        print line
        print fileAndLeg(line, True, None, 'di-tau')
