###############################################
#
#   Library of functions used when dealing 
#   with jets, especially reformat_jets.py
#
###############################################

import fastjet
from pyjet import cluster
from particle import Particle
import numpy as np
import awkward as ak
import vector


# Takes an entry from a tree and extracts lists of key parameters.
def getLists(entry, hardSc = False): #, pTcut=0):
        # This applies to the entire event
        pdgidList = entry.sim_pdgId
        evLen = len(pdgidList)

        # pTList = np.zeros(evLen, dtype=np.float64)
        # etaList = np.zeros(evLen, dtype=np.float64)
        # phiList = np.zeros(evLen, dtype=np.float64)
        # massList = np.zeros(evLen, dtype=np.float64)

        pTList = np.ones(evLen, dtype=np.float64)*-999
        etaList = np.ones(evLen, dtype=np.float64)*-999
        phiList = np.ones(evLen, dtype=np.float64)*-999
        massList = np.ones(evLen, dtype=np.float64)*-999

        # Putting the data in the right format
        for j in range(evLen):#range(len(simEvnp)):
                if (hardSc and entry.sim_event[j] !=0 ): 
                        # print(f"event != 0 {entry.sim_event[j]}")
                        continue
                if(entry.sim_q[j] != 0 and entry.sim_pt[j] < 0.75 ):
                        continue
                pdgid = pdgidList[j]
                massList[j] = Particle.from_pdgid(pdgid).mass/1000.

                pTList[j] = entry.sim_pt[j]
                etaList[j] = entry.sim_eta[j]
                phiList[j] = entry.sim_phi[j]

        massList = massList[massList != -999]
        pTList = pTList[pTList != -999]
        etaList = etaList[etaList != -999]
        phiList = phiList[phiList != -999]

        # Perform pT cut, optional
        # if(pTcut!=0):
        #         maskList = pTList # eh I can probably use maskList = pTList>pTcut
        #         pTList = pTList[maskList>pTcut]
        #         etaList = etaList[maskList>pTcut]
        #         phiList = phiList[maskList>pTcut]
        #         massList = massList[maskList>pTcut]

        return pTList, etaList, phiList, massList


# Takes an entry from a tree and extracts particles from it. Uses
# those particles to create jets, which it returns.
def createJets(pTList, etaList, phiList, massList):

        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

        length = np.size(pTList)
        fjs = []

        for j in range(length):
                f=fastjet.PseudoJet(pseudojet_from_pt_eta_phi_m(pTList[j], etaList[j], phiList[j], massList[j]))
                fjs.append(f)

        # Left over from pyjet implementation
        # jetInput = np.array([],dtype=np.dtype([('pt', 'f8'), ('eta', 'f8'), 
        #                                     ('phi', 'f8'), ('M', 'f8')]))
        # for i in range(length):
        #         jetInput = np.append(jetInput, np.array([(pTList[i], etaList[i], phiList[i], massList[i])], 
        #                                               dtype=jetInput.dtype))
        # Actual jet step
        # sequence = cluster(jetInput, R=0.4, p=-1) # p=-1 gives anti-kt
        # jets = sequence.inclusive_jets(ptmin=20)  # list of PseudoJets
        
        # FastJet implementation with awkward arrays-- do not use
        # vector.register_awkward()
        # awkJetInput = ak.Array(jetInput, with_name="Momentum4D")
        # cluster = fastjet.ClusterSequence(awkJetInput, jetdef)
        # jets = cluster.inclusive_jets(min_pt = 20)

        cluster = fastjet.ClusterSequence(fjs, jetdef)
        jets = cluster.inclusive_jets(ptmin=20)

        return cluster, jets

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

def pseudojet_from_pt_eta_phi_m(pt, eta, phi, mass):
    # Convert (pt, eta, phi, mass) to (px, py, pz, E) and create a PseudoJet.
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    return fastjet.PseudoJet(px, py, pz, energy)