###############################################
#
#   Library of functions used when dealing 
#   with jets, especially reformat_jets.py
#
###############################################

from pyjet import cluster
from particle import Particle
import numpy as np


# Takes an entry from a tree and extracts lists of key parameters.
def getLists(entry, pTcut=0):
        # This applies to the entire event
        pdgidList = entry.sim_pdgId
        evLen = len(pdgidList)

        pTList = np.zeros(evLen, dtype=np.float64)
        etaList = np.zeros(evLen, dtype=np.float64)
        phiList = np.zeros(evLen, dtype=np.float64)
        massList = np.zeros(evLen, dtype=np.float64)

        # Putting the data in the right format
        for j in range(evLen):
            # j is one particle within the event
            pdgid = pdgidList[j]
            massList[j] = Particle.from_pdgid(pdgid).mass

            pTList[j] = entry.sim_pt[j]
            etaList[j] = entry.sim_eta[j]
            phiList[j] = entry.sim_phi[j]

        # Perform pT cut, optional
        if(pTcut!=0):
                maskList = pTList # eh I can probably use maskList = pTList>pTcut
                pTList = pTList[maskList>pTcut]
                etaList = etaList[maskList>pTcut]
                phiList = phiList[maskList>pTcut]
                massList = massList[maskList>pTcut]

        return pTList, etaList, phiList, massList


# Takes an entry from a tree and extracts particles from it. Uses
# those particles to create jets, which it returns.
def createJets(pTList, etaList, phiList, massList):

        length = np.size(pTList)
        vectors = np.array([],dtype=np.dtype([('pT', 'f8'), ('eta', 'f8'), 
                                            ('phi', 'f8'), ('mass', 'f8')]))

        for i in range(length):
                vectors = np.append(vectors, np.array([(pTList[i], etaList[i], phiList[i], massList[i])], 
                                                      dtype=vectors.dtype))
        
        # Actual jet step
        sequence = cluster(vectors, R=0.4, p=-1) # p=-1 gives anti-kt
        jets = sequence.inclusive_jets()  # list of PseudoJets

        return jets

def plotOneJet(jet, name):
    const = jet.constituents_array()
    plt.scatter(const["eta"], const["phi"], c='green')
    plt.scatter([jet.eta], [jet.phi], c="red")
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')
    plt.savefig(name)
    plt.clf()

def matchArr(jetArr, treeArr):
        # Convert to int so np.where() is available
        intjetArr = (10000*jetArr).astype(int)
        inttreeArr = (10000*treeArr).astype(int)

        # Stores recovered index of particle in jet
        indexArr = np.zeros(len(jetArr))
        
        for i in range(len(intjetArr)):
                indexArr[i] = np.where(inttreeArr == intjetArr[i])[0][0]

        return indexArr