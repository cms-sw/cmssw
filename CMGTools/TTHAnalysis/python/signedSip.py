import ROOT

#ROOT.gSystem.Load("libCMGToolsTTHAnalysis")
SignedImpactParameterComputer = ROOT.SignedImpactParameter()

def signedSip3D(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    dir   = lepton.jet.momentum()         if hasattr(lepton,'jet')      else lepton.momentum()
    track = lepton.gsfTrack() if abs(lepton.pdgId()) == 11  else lepton.track()
    meas = SignedImpactParameterComputer.signedIP3D(track.get(), vertex, dir)
    return meas.significance()

def signedIp3D(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    dir   = lepton.jet.momentum()         if hasattr(lepton,'jet')      else lepton.momentum()
    track = lepton.gsfTrack() if abs(lepton.pdgId()) == 11  else lepton.track()
    meas = SignedImpactParameterComputer.signedIP3D(track.get(), vertex, dir)
    return meas.value()


def maxSignedSip3Djettracks(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    if hasattr(lepton,'jet') and lepton.jet != lepton :
       dir = lepton.jet.momentum()   
       jetTracks = [ dau.pseudoTrack() for dau in lepton.jet.daughterPtrVector() if dau.charge() != 0 and dau.pt() > 1.0 ]
       if len(jetTracks) !=0:
           meas = max((SignedImpactParameterComputer.signedIP3D(track,vertex,dir)).significance() for track in jetTracks )
           return meas
       else:
           return -1.
    else : 
        return -1.


def maxSip3Djettracks(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    if hasattr(lepton,'jet') and lepton.jet != lepton :
       dir = lepton.jet.momentum()   
       jetTracks = [ dau.pseudoTrack() for dau in lepton.jet.daughterPtrVector() if dau.charge() != 0 and dau.pt() > 1.0 ]
       if len(jetTracks) !=0:
           meas = max( (SignedImpactParameterComputer.IP3D(track,vertex)).significance() for track in jetTracks )
           return meas
       else:
           return -1.
    else : 
        return -1.


def maxSignedSip2Djettracks(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    if hasattr(lepton,'jet') and lepton.jet != lepton :
       dir = lepton.jet.momentum()   
       jetTracks = [ dau.pseudoTrack() for dau in lepton.jet.daughterPtrVector() if dau.charge() != 0 and dau.pt() > 1.0 ]
       if len(jetTracks) !=0:
           meas = max( (SignedImpactParameterComputer.signedIP2D(track,vertex,dir)).significance() for track in jetTracks )
           return meas
       else:
           return -1.
    else : 
        return -1.



def maxSip2Djettracks(lepton, vertex=None):
    if vertex is None:
        vertex = lepton.associatedVertex
    if hasattr(lepton,'jet') and lepton.jet != lepton :
       dir = lepton.jet.momentum()   
       jetTracks = [ dau.pseudoTrack() for dau in lepton.jet.daughterPtrVector() if dau.charge() != 0 and dau.pt() > 1.0 ]
       if len(jetTracks) !=0:
           meas = max( (SignedImpactParameterComputer.IP2D(track,vertex)).significance() for track in jetTracks )
           return meas
       else:
           return -1.
    else : 
        return -1.







def twoTrackChi2(lepton1,lepton2):
    track1 = lepton1.gsfTrack() if abs(lepton1.pdgId()) == 11  else lepton1.track()
    track2 = lepton2.gsfTrack() if abs(lepton2.pdgId()) == 11  else lepton2.track()
    pair = SignedImpactParameterComputer.twoTrackChi2(track1.get(),track2.get())
    return (pair.first,pair.second)

#For the vertex related variables
#A = selectedLeptons[0], B = selectedLeptons[1], C = selectedLeptons[2], D = selectedLeptons[3] 
##Variables related to IP
#Of one lepton w.r.t. the PV of the event
def absIP3D(lepton, pv=None):
    if pv is None:
        pv = lepton.associatedVertex
    track = lepton.gsfTrack() if abs(lepton.pdgId()) == 11  else lepton.track()
    pairvalerr  = SignedImpactParameterComputer.absIP3D(track.get(), pv)
    return (pairvalerr.first,pairvalerr.second)
#Of one lepton w.r.t. the PV of the PV of the other leptons only
def absIP3Dtrkpvtrks(leptonA,leptonB,leptonC,leptonD,nlep,iptrk):
    leptrkA = leptonA.gsfTrack() if abs(leptonA.pdgId()) == 11  else leptonA.track()
    leptrkB = leptonB.gsfTrack() if abs(leptonB.pdgId()) == 11  else leptonB.track()
    leptrkC = leptonC.gsfTrack() if abs(leptonC.pdgId()) == 11  else leptonC.track()
    leptrkD = leptonD.gsfTrack() if abs(leptonD.pdgId()) == 11  else leptonD.track()     
    pairvalerr = SignedImpactParameterComputer.absIP3Dtrkpvtrks(leptrkA.get(),leptrkB.get(),leptrkC.get(),leptrkD.get(),nlep,iptrk)
    return (pairvalerr.first,pairvalerr.second)      
##Variables related to chi2
def chi2pvtrks(leptonA,leptonB,leptonC,leptonD,nlep):
    leptrkA = leptonA.gsfTrack() if abs(leptonA.pdgId()) == 11  else leptonA.track()
    leptrkB = leptonB.gsfTrack() if abs(leptonB.pdgId()) == 11  else leptonB.track()
    leptrkC = leptonC.gsfTrack() if abs(leptonC.pdgId()) == 11  else leptonC.track()
    leptrkD = leptonD.gsfTrack() if abs(leptonD.pdgId()) == 11  else leptonD.track() 
    pairvalerr = SignedImpactParameterComputer.chi2pvtrks(leptrkA.get(),leptrkB.get(),leptrkC.get(),leptrkD.get(),nlep)
    return (pairvalerr.first,pairvalerr.second)
