#!/usr/bin/env python3

from __future__ import print_function
import sys
import string
from ROOT import *
from array import array

def getFileInPath(rfile):
   import os
   for dir in os.environ['CMSSW_SEARCH_PATH'].split(":"):
     if os.path.exists(os.path.join(dir,rfile)): return os.path.join(dir,rfile)
   return None


detIDsFileName = getFileInPath('DQM/SiStripMonitorClient/data/detids.dat')

filename_online=sys.argv[1]
filename_offline=sys.argv[2]

runNum=filename_offline[19:25]

dir="DQMData/Run " + runNum + "/PixelPhase1/Run summary/Pahse1_MechanicalView/"


dirBPix=dir + "PXBarrel/"
dirFPix=dir + "PXForward/"

hnameB="digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_"

minlad=[-6,-14,-22,-32]

shell=""

#Barrel
def BPIX_list(inputFile):
    DQMfile=TFile(inputFile)
    BPIXCounter = []
    BPIXCounter_v0 = []
    for l in range(1,5):
        hname=hnameB + str(l)
    
        RocMap=DQMfile.FindObjectAny(hname)
    
        for j in range(1,RocMap.GetNbinsY()+1):
        
            lad=minlad[l-1] + int(j-1)/2
            alad=abs(lad)
        
            for i in range(1,RocMap.GetNbinsX()+1):
            
                roc=0
            
                bin=RocMap.GetBin(i,j)
                digi=RocMap.GetBinContent(bin)

                if (digi!=0): continue
            
                mod=-4 + int((i-1)/8)

                if (lad==0 or mod==0): continue

                if (lad < 0 and mod < 0) :

                    shell="BmO"

                    if (alad%2==1):
                        if (j%2==1): roc=(i-1)%8
                        if (j%2==0): roc= 15-(i-1)%8
                    
                    elif (alad%2==0):
                        if (j%2 ==0): roc=(i-1)%8
                        if (j%2 ==1): roc= 15-(i-1)%8

                if (lad > 0 and mod < 0):

                    shell= "BmI";
                
                    if (lad%2==1):
                        if (j%2 ==0): roc=(i-1)%8
                        if (j%2 ==1): roc= 15-(i-1)%8
                    
                    if (lad%2==0):
                        if (j%2 ==1): roc=(i-1)%8
                        if (j%2 ==0): roc= 15-(i-1)%8

                if (lad > 0 and mod > 0):                                                                  
                
                    shell= "BpI"
                
                    if (lad%2==1):
                        if (j%2 ==1): roc=7-(i-1)%8
                        if (j%2 ==0): roc=8+(i-1)%8
                    
                    if (lad%2==0):
                        if (j%2 ==0): roc=7-(i-1)%8
                        if (j%2 ==1): roc=8+(i-1)%8

                if (lad < 0 and mod > 0):     
                    shell= "BpO"
                
                    if (alad%2==1):
                        if (j%2 ==0): roc=7-(i-1)%8
                        if (j%2 ==1): roc=8+(i-1)%8
                    
                    if (alad%2==0):
                        if (j%2 ==1): roc=7-(i-1)%8
                        if (j%2 ==0): roc=8+(i-1)%8

                f1=open(getFileInPath('DQM/SiStripMonitorClient/data/detids.dat'))
                modwritten=False
                Mod_check = "LYR"+str(l) + "_LDR" + str(abs(lad)) + "F_MOD" +str(abs(mod))
                shell_check = "BPix_" + str(shell)

                for line in f1:
                    refName=line.split(" ")[1]
                    if modwritten: break
                    shell_ref = str(refName[:8]).strip()
                    module_ref = str(refName[14:]).strip()

                    if (Mod_check == module_ref) and (shell_check == shell_ref):

                        ModuleName_BPIX = refName.strip()+"_ROC "
                        BmLYR1_check = ModuleName_BPIX.split('_')                        
                        if ((BmLYR1_check[1] == "BmI" or BmLYR1_check[1] == "BmO") and (BmLYR1_check[3] == "LYR1")):
                           if int(roc) <= 7:
                              roc = str(int(roc)+8)
                           elif int(roc) >= 8:
                              roc =str(int(roc)-8)

                        BPix_Name = ModuleName_BPIX + str(roc)
                        BPIXCounter_v0.append(BPix_Name)
                        BPIXCounter = list(set(BPIXCounter_v0))
                        modwritten=True
    return BPIXCounter

#End of Barrel

#Doing FPix

hnameF="digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_"
minbld=[-11,-17]

def FPIX_list(inputFile):
    FPIXCounter = []
    DQMfile=TFile(inputFile)
    for r in range (1,3):

        hname=hnameF + str(r)
        RocMap=DQMfile.FindObjectAny(hname)

        for j in range(1,RocMap.GetNbinsY()+1):        

            bld=minbld[r-1] + int(j-1)/4
            abld=abs(bld)
        
            for i in range(1,RocMap.GetNbinsX()+1):
            
                roc=0

                bin=RocMap.GetBin(i,j)
                digi=RocMap.GetBinContent(bin)

                if (digi!=0): continue

                disk=-3 + int(i-1)/8

                if (bld==0 or disk==0): continue

                pnl=0

                if ((j-1)%4==0 or (j-1)%4==1): pnl=2
                if ((j-1)%4==2 or (j-1)%4==3): pnl=1

                if (disk < 0 and bld <0):

                    shell= "BmO"

                    if ((j-1)%4==0 or (j-1)%4==3): roc= 15-(i-1)%8 
                    if ((j-1)%4==1 or (j-1)%4==2): roc= (i-1)%8

                if (disk < 0 and bld >0):
                    
                    shell= "BmI";
                
                    if ((j-1)%4==0 or (j-1)%4==3): roc= 15-(i-1)%8 
                    if ((j-1)%4==1 or (j-1)%4==2): roc= (i-1)%8

                if (disk > 0 and bld >0):
                    
                    shell= "BpI"
                
                    if ((j-1)%4==0 or (j-1)%4==3): roc=7-(i-1)%8 
                    if ((j-1)%4==1 or (j-1)%4==2): roc=8+(i-1)%8
                
                if (disk > 0 and bld <0):

                    shell= "BpO"
                
                    if ((j-1)%4==0 or (j-1)%4==3): roc=7-(i-1)%8; 
                    if ((j-1)%4==1 or (j-1)%4==2): roc=8+(i-1)%8;


                FPix_Name = "FPix_" + str(shell) + "_D" + str(abs(disk)) + '_BLD'+ str(abs(bld)) + '_PNL' + str(abs(pnl)) + '_RNG'+ str(abs(r)) + "_ROC " + str(roc) +""
                FPIXCounter.append(FPix_Name)
    return FPIXCounter


deadROCList_online = list(set(BPIX_list(filename_online))) + list(set(FPIX_list(filename_online)))
deadROCList_offline = list(set(BPIX_list(filename_offline))) + list(set(FPIX_list(filename_offline)))

MaskedROC_DuringRun = list(set(deadROCList_online) - set(deadROCList_offline))
print('Number of New Dead ROC: '+ str(len(MaskedROC_DuringRun)))

outFileName = 'DeadROC_Diff.txt'
outFileName_online = 'DeadROC_online.txt'
outFileName_offline = 'DeadROC_offline.txt'
f = open(outFileName,"w")
f1 = open(outFileName_online,"w")
f2 = open(outFileName_offline,"w")

for i in range(len(MaskedROC_DuringRun)):

    f.write(MaskedROC_DuringRun[i]+"\n")
f.close()

for i in range(len(deadROCList_online)):

    f1.write(deadROCList_online[i]+"\n")
f1.close()

for i in range(len(deadROCList_offline)):

    f2.write(deadROCList_offline[i]+"\n")
f2.close()


