from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import printOut 
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import GenParticle 

def findStatus1Leptons(particle):
    '''Returns status 1 e and mu among the particle daughters'''
    leptons = []
    for i in range( particle.numberOfDaughters() ):
        dau = particle.daughter(i)
        if dau.status() == 1:
            if abs(dau.pdgId())==11 or abs(dau.pdgId())==13:
                leptons.append( dau )
            else:
                continue
        else:
            leptons = findStatus1Leptons( dau, leptons )
    return leptons


def allDaughters(particle, daughters, rank ):
    '''Fills daughters with all the daughters of particle.
    Recursive function.'''
    rank += 1 
    for i in range( particle.numberOfDaughters() ):
        dau = GenParticle(particle.daughter(i))
        dau.rank = rank
        daughters.append( dau )
        daughters = allDaughters( dau, daughters, rank )
    return daughters


def bosonToX(particles, bosonType, xType):
    bosons = filter(lambda x: x.status()==3 and x.pdgId()==bosonType, particles)
    daughters = []
    if len(bosons)==0:
        return [], False
    boson = bosons[0]
    daus = []
    allDaughters( boson, daus, 0)
    xDaus = filter(lambda x: x.status()==3 and abs(x.pdgId())==xType, daus)
    # print printOut(xDaus)
    return xDaus, True 

def isNotHadronicId(pdgId,includeSMLeptons=True):
    if abs(pdgId) in [11,12,13,14,15,16]:
        return includeSMLeptons
    i = (abs(pdgId) % 1000)
    return i > 10 and i != 21 and i < 100

def isPromptLepton(lepton, beforeFSR, includeMotherless=True, includeTauDecays=False):
    if abs(lepton.pdgId()) not in [11,13,15]:
        return False
    if lepton.numberOfMothers() == 0:
        return includeMotherless;
    mom = lepton.mother()
    if mom.pdgId() == lepton.pdgId():
        if beforeFSR: return False
        return isPromptLepton(mom, beforeFSR, includeMotherless, includeTauDecays)
    elif abs(mom.pdgId()) == 15:
        if not includeTauDecays: return False
        return isPromptLepton(mom, beforeFSR, includeMotherless, includeTauDecays)
    else:
        return isNotHadronicId(mom.pdgId(), includeSMLeptons=False)



