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
def getLists(entry, hardSc = False, pTcut=True):
        # This applies to the entire event
        pdgidList = entry.sim_pdgId
        evLen = len(pdgidList)

        pTList = np.ones(evLen, dtype=np.float64)*-999
        etaList = np.ones(evLen, dtype=np.float64)*-999
        phiList = np.ones(evLen, dtype=np.float64)*-999
        massList = np.ones(evLen, dtype=np.float64)*-999

        # Putting the data in the right format
        for j in range(evLen):#range(len(simEvnp)):
                if (hardSc and entry.sim_event[j] !=0 ): 
                        # print(f"event != 0 {entry.sim_event[j]}")
                        continue
                if(pTcut and entry.sim_q[j] != 0 and entry.sim_pt[j] < 0.75 ):
                        continue
                vtxX = entry.simvtx_x[entry.sim_parentVtxIdx[j]]
                vtxY = entry.simvtx_y[entry.sim_parentVtxIdx[j]]
                if(np.sqrt(vtxX**2 + vtxY**2)>10):
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

def matchArr(jetArrpT, jetArreta, jetArrphi, treeArrpT, treeArreta, treeArrphi, evnum, jetnum):
        indexArr = np.ones(len(jetArrpT))*-999

        for i in range(len(jetArrpT)):
                for j in range(len(treeArrpT)):
                        if(np.abs(jetArrpT[i]-treeArrpT[j])<0.00001 and np.abs(jetArreta[i]-treeArreta[j])<0.00001 
                                and np.abs(np.cos(jetArrphi[i])-np.cos(treeArrphi[j]))<0.00001):
                                if(indexArr[i]!=-999): 
                                       print(f"Error: double matched at Event={evnum}, Jet={jetnum} i={i}")
                                       continue 
                                indexArr[i] = j
        if(indexArr[i]==-999.0):
                print(f"Error: Event={evnum}, Jet={jetnum} i={i}, pT = {jetArrpT[i]}, eta = {jetArreta[i]}, phi = {jetArrphi[i]}")

        return indexArr

def pseudojet_from_pt_eta_phi_m(pt, eta, phi, mass):
    # Convert (pt, eta, phi, mass) to (px, py, pz, E) and create a PseudoJet.
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    return fastjet.PseudoJet(px, py, pz, energy)