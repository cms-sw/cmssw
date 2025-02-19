# Colin, March 2012

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

def _getattrGenerator( process, postfix ):
    '''A function generator to simplify the getattr syntax'''
    def fun(name):
        return getattr(process, name+postfix)
    return fun


_PFBRECO_loaded = False

def _loadPFBRECO(process):
    '''The particle-flow based reconstruction sequence should be loaded once in the process.
    Not optimal, should load it only if it is not detected (hasattr)'''
    global _PFBRECO_loaded
    if _PFBRECO_loaded is False: 
        _PFBRECO_loaded = True
        process.load("CommonTools.ParticleFlow.PFBRECO_cff")


def setupPFIso(process, leptonCollection, particleName, newpostfix='PFIso', postfix='', runPF2PAT = False):
    '''Generic function to setup particle-based isolation for a given lepton collection.
    Returns the isolation sequence.
    You are responsible for adding it to your path.

    leptonCollection could e.g. be "gsfElectrons" or "muons"
    particleName must be either "Electron" or "Muon".
    newpostfix can be specified to define several particle-flow isolation sequences
    '''
    lepshort = None
    if particleName=='Electron':
        lepshort='el'
    elif particleName=='Muon':
        lepshort='mu'
    else:
        raise ValueError('particleName should be equal to "Electron" or "Muon"')

    if runPF2PAT != True : 
      _loadPFBRECO(process)

    # ADD VETOES IN ENDCAPS!
    fullpostfix = postfix+newpostfix
    ga = _getattrGenerator( process, postfix )
    ganew = _getattrGenerator( process, fullpostfix )

    leptonSeq = cms.Sequence(
        ga('pf{lepton}IsolationSequence'.format(lepton=particleName))  
        )
    setattr( process, 'std{lepton}Sequence{postfix}'.format(lepton=particleName,
                                                   postfix=postfix), leptonSeq)

    leptonSource = leptonCollection
    cloneProcessingSnippet(process,
                           ga('std{lepton}Sequence'.format(lepton=particleName)),
                           newpostfix)

    ganew("{lepshort}PFIsoDepositCharged".format(lepshort=lepshort) ).src = leptonSource
    ganew("{lepshort}PFIsoDepositChargedAll".format(lepshort=lepshort)).src = leptonSource
    ganew("{lepshort}PFIsoDepositNeutral".format(lepshort=lepshort)).src = leptonSource
    ganew("{lepshort}PFIsoDepositGamma".format(lepshort=lepshort)).src = leptonSource
    ganew("{lepshort}PFIsoDepositPU".format(lepshort=lepshort)).src = leptonSource

    return ganew('std{lepton}Sequence'.format(lepton=particleName))

def setupPFIsoPhoton(process, photonCollection, particleName, newpostfix='PFIso'):
    '''Generic function to setup particle-based isolation for a given lepton collection.
    Returns the isolation sequence.
    You are responsible for adding it to your path.

    leptonCollection could e.g. be "gsfElectrons" or "muons"
    particleName must be either "Electron" or "Muon".
    newpostfix can be specified to define several particle-flow isolation sequences
    '''
    phoshort = None
    if particleName=='Photon':
        phoshort='ph'
    else:
        raise ValueError('particleName should be equal to "Photon"')
    
    _loadPFBRECO(process)

    postfix = ''
    # ADD VETOES IN ENDCAPS!
    fullpostfix = postfix+newpostfix
    #fullpostfix = ''
    ga = _getattrGenerator( process, postfix )
    ganew = _getattrGenerator( process, fullpostfix )

    photonSeq = cms.Sequence(
        ga('pf{photon}IsolationSequence'.format(photon=particleName))  
        )
    setattr( process, 'std{photon}Sequence{postfix}'.format(photon=particleName,
                                                   postfix=postfix), photonSeq)

    photonSource = photonCollection
    cloneProcessingSnippet(process,
                           ga('std{photon}Sequence'.format(photon=particleName)),
                           newpostfix)

    ganew("{phoshort}PFIsoDepositCharged".format(phoshort=phoshort) ).src = photonSource
    ganew("{phoshort}PFIsoDepositChargedAll".format(phoshort=phoshort)).src = photonSource
    ganew("{phoshort}PFIsoDepositNeutral".format(phoshort=phoshort)).src = photonSource
    ganew("{phoshort}PFIsoDepositGamma".format(phoshort=phoshort)).src = photonSource
    ganew("{phoshort}PFIsoDepositPU".format(phoshort=phoshort)).src = photonSource

    return ganew('std{photon}Sequence'.format(photon=particleName))


def setupPFMuonIso(process, muonCollection, postfix='PFIso' ):
    '''Set up particle-based isolation for the muons in muonCollection.

    Calls setupPFIso.
    '''
    return setupPFIso( process, muonCollection, 'Muon', postfix)



def setupPFElectronIso(process, electronCollection, newpostfix='PFIso', postfix='', runPF2PAT = False ):
    '''Set up particle-based isolation for the electrons in electronCollection.

    Calls setupPFIso.
    '''
    #    print 'WARNING!!! the vetoes are the ones defined for the PF e-s (no veto...).'
    #    print 'Vetoes should be applied in the endcaps when doing particle-based isolation on gsfElectrons.'
    #    print 'Need a volunteer to implement that.'
    return setupPFIso( process, electronCollection, 'Electron', newpostfix, postfix, runPF2PAT)


def setupPFPhotonIso(process, photonCollection, postfix='PFIso' ):
    '''Set up particle-based isolation for the electrons in electronCollection.

    Calls setupPFIsoPhoton.
    '''
    #    print 'WARNING!!! the vetoes are the ones defined for the PF e-s (no veto...).'
    #    print 'Please make sure that your file with vetoes is up to date'
    return setupPFIsoPhoton( process, photonCollection, 'Photon', postfix)

