#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import math
import numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

ptLSB = 0.25;
etaLSB = 0.0043633231;
phiLSB = 0.0043633231;
hwJets = []
emJets = []
hwData = []
emData = []
hwDataNoZ = []
emDataNoZ = []
nHw = 0
nEm = 0
nEv = 5000
nEvNoZ = 0


for hwRx in range(0,70):
    with open("datFull/" + str(hwRx) + "/tx_summary.txt", "r") as inFile:
        frameIt = -1
        for line in inFile:
            if('1v' in line):
                frameIt += 1
                if frameIt < 20:
                    continue
                if hwRx == 0 and frameIt < 33:
                    continue
                linkData = line.split('1v')
                for wordIt in range(1,25):
                    word = linkData[wordIt].replace(' ','').replace('\n','')
                    if int(word, 16) & 0xffff:
                        jet = word[8:]
                        hwJets.append([(int(jet,16)&0xffff)*ptLSB, 
                                         ((((int(jet,16)>>24)&0xff)*19)+9)*etaLSB,
                                         ((((int(jet,16)>>16)&0xff)*20)+10)*phiLSB])
                    if (int(word, 16)>>32) & 0xffff:
                        jet = word[:8]
                        hwJets.append([(int(jet,16)&0xffff)*ptLSB, 
                                         ((((int(jet,16)>>24)&0xff)*19)+9)*etaLSB,
                                         ((((int(jet,16)>>16)&0xff)*20)+10)*phiLSB])
                if (frameIt%13) == 6:
                    if(nHw>=nEv):
                        break
                    nHw+=1
                    if len(hwJets)==0:
                        hwJets.append([0,0,0])
                    hwData.append(hwJets)
                    del hwJets
                    hwJets = []


with open("emuout.txt", "r") as inFile:
    for line in inFile:
        if " " in line:
            if(nEm>=nEv):
                break
            nEm+=1
            if len(emJets)>0:
                emData.append(emJets)
            del emJets
            emJets = []
        else:
            jet = [float(line.split("\t")[0]),
                   float(line.split("\t")[1]),
                   float(line.split("\t")[2])]
            emJets.append(jet)


nDiff = 0

for evIt in range(0,nEv):
    if len(hwData[evIt]) != len(emData[evIt]):
        nDiff+=1
        continue
    goodJet=0
    for hwJet in hwData[evIt]:
        for emJet in emData[evIt]:
            if hwJet[0] == emJet[0]:
                if (hwJet[1]-emJet[1])<0.01:
                    if (hwJet[2]-emJet[2])<0.01:
                        goodJet+=1
    if goodJet < len(hwData[evIt]):
        nDiff+=1


print("\n\nnEvent = " + str(nEv) + "\nnDiff = " + str(nDiff) + "\nGood events = " + str((1-float(nDiff)/float(nEv))*100) + "%")
        


print("\n\n=====================================================================================")
print("\t\tFirmware Events: " + str(nHw) + "\t\t" + "Emulator Events: " + str(nEm))
print("=====================================================================================")
print("\t\tpT\t" + "eta\t" + "phi\t\t" + "pT\t" + "eta\t" + "phi\t")
print("=====================================================================================")


for evIt in range(0,nEv):
    if hwData[evIt][0][0] > 0:
        hwDataNoZ.append(hwData[evIt])
    if emData[evIt][0][0] > 0:
        emDataNoZ.append(emData[evIt])
    nEvNoZ+=1


for evIt in range(0,nEv):
    if hwData[evIt][0][0] ==0 and emData[evIt][0][0] == 0:
        continue
    jetCount=0
    jetDiff = len(hwData[evIt]) - len(emData[evIt])
    print("")
    if jetDiff==0:
        for jetIt in range(len(hwData[evIt])):
            print(str(evIt) + "\t\t" + str(hwData[evIt][jetIt][0]) + "\t" + str(hwData[evIt][jetIt][1])[:4] + "\t" + str(hwData[evIt][jetIt][2])[:4] + "\t\t" +
                  str(emData[evIt][jetIt][0]) + "\t" + str(emData[evIt][jetIt][1])[:4] + "\t" + str(emData[evIt][jetIt][2])[:4])
    if jetDiff>0:
        for jetIt in range(len(hwData[evIt])):
            jetCount+=1
            if jetCount > len(emData[evIt]):
                emData[evIt].append([0,0,0])
            print(str(evIt) + "\t\t" + str(hwData[evIt][jetIt][0]) + "\t" + str(hwData[evIt][jetIt][1])[:4] + "\t" + str(hwData[evIt][jetIt][2])[:4]  + "\t\t" +
                  str(emData[evIt][jetIt][0]) + "\t" + str(emData[evIt][jetIt][1])[:4] + "\t" + str(emData[evIt][jetIt][2])[:4])
    if jetDiff<0:
        for jetIt in range(len(emData[evIt])):
            jetCount+=1
            if jetCount > len(hwData[evIt]):
                hwData[evIt].append([0,0,0])
            print(str(evIt) + "\t\t" + str(hwData[evIt][jetIt][0]) + "\t" + str(hwData[evIt][jetIt][1])[:4] + "\t" + str(hwData[evIt][jetIt][2])[:4]  + "\t\t" +
                  str(emData[evIt][jetIt][0]) + "\t" + str(emData[evIt][jetIt][1])[:4] + "\t" + str(emData[evIt][jetIt][2])[:4])
        



fig, axs =   plt.subplots(2,3, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})

fig.patch.set_facecolor( '#ffffff')


nPtHw,  bPtHw  = np.histogram([jet[0] for event in hwDataNoZ for jet in event], bins=50, range=(0,200))
nEtaHw, bEtaHw = np.histogram([jet[1] for event in hwDataNoZ for jet in event], bins=18, range=(0,1.5))
nPhiHw, bPhiHw = np.histogram([jet[2] for event in hwDataNoZ for jet in event], bins=8,  range=(0,0.7))

meansPt  = [0.5*(bPtHw[i]  + bPtHw[i+1])  for i in range(len(nPtHw))]
meansEta = [0.5*(bEtaHw[i] + bEtaHw[i+1]) for i in range(len(nEtaHw))]
meansPhi = [0.5*(bPhiHw[i] + bPhiHw[i+1]) for i in range(len(nPhiHw))]

nPtEm  = axs[0,0].hist([jet[0] for event in emDataNoZ for jet in event], bins=50, range=(0,200), histtype='bar', linewidth=1.5, label='Emulator', color='#929591', zorder=0)[0]
nEtaEm = axs[0,1].hist([jet[1] for event in emDataNoZ for jet in event], bins=18, range=(0,1.5), histtype='bar', linewidth=1.5, label='Emulator', color='#929591', zorder=0)[0]
nPhiEm = axs[0,2].hist([jet[2] for event in emDataNoZ for jet in event], bins=8,  range=(0,0.7), histtype='bar', linewidth=1.5, label='Emulator', color='#929591', zorder=0)[0]

axs[0,0].scatter(meansPt,  nPtHw,  label='Firmware', c='#000000', linewidths=0.5, s=25, marker='+')
axs[0,1].scatter(meansEta, nEtaHw, label='Firmware', c='#000000', linewidths=0.5, s=25, marker='+')
axs[0,2].scatter(meansPhi, nPhiHw, label='Firmware', c='#000000', linewidths=0.5, s=25, marker='+')


axs[1,0].scatter(meansPt,  [(hw/em) for hw,em in zip(nPtHw,nPtEm)] ,  c='#000000', linewidths=0.5, s=15, zorder=1)
axs[1,1].scatter(meansEta, [(hw/em) for hw,em in zip(nEtaHw,nEtaEm)], c='#000000', linewidths=0.5, s=15, zorder=1)
axs[1,2].scatter(meansPhi, [(hw/em) for hw,em in zip(nPhiHw,nPhiEm)], c='#000000', linewidths=0.5, s=15, zorder=1)

axs[1,0].axhline(y=1, linewidth=1, linestyle='--', c='#929591')
axs[1,1].axhline(y=1, linewidth=1, linestyle='--', c='#929591')
axs[1,2].axhline(y=1, linewidth=1, linestyle='--', c='#929591')

axs[1,0].set(ylim=(0.5,1.5))
axs[1,1].set(ylim=(0.5,1.5))
axs[1,2].set(ylim=(0.5,1.5))

axs[0,0].set(ylabel="Events")
axs[1,0].set(ylabel="FW / EMU")

axs[0,0].legend(prop={'size': 10})
axs[0,1].legend(prop={'size': 10})
axs[0,2].legend(prop={'size': 10})

ymaxPt  = max(np.concatenate([nPtHw,nPtEm]))
ymaxEta = max(np.concatenate([nEtaHw,nEtaEm]))
ymaxPhi = max(np.concatenate([nPhiHw,nPhiEm]))

axs[0,0].set(xlim=(0,200))
axs[0,1].set(xlim=(0,1.5))
axs[0,2].set(xlim=(0,0.7))

axs[0,0].set(ylim=(0,ymaxPt +(0.05*ymaxPt)))
axs[0,1].set(ylim=(0,ymaxEta+(0.05*ymaxEta)))
axs[0,2].set(ylim=(0,ymaxPhi+(0.05*ymaxPhi)))

axs[0,0].set(xlabel="Jet $p_T$ (GeV)")
axs[0,1].set(xlabel="Jet $\eta$")
axs[0,2].set(xlabel="Jet $\phi$")

plt.savefig('ttbarPU200_3900.pdf', bbox_inches='tight')
