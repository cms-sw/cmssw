#!/usr/bin/env python
# encoding: utf-8

# File        : makeTrackConversionLUTs.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2021 Apr 13
#
# Description : 

from __future__ import division
import math
import numpy as np
from collections import defaultdict, Counter
from pprint import pprint

BITSABSCURV=14
BITSPT=13
maxCurv = 0.00855
ptLSB=0.025
ptLUT=[]
pts = []
ptshifts = []


BITSTTTANL=16-1
BITSETA=13-1
maxTanL=8.0
etaLSB = math.pi/ (1<<BITSETA)
etaLUT=[]
etas = []
etashifts = []

def GetPtLUT():
    for i in range(1,(1<<BITSABSCURV)):
        k = (maxCurv*i)/(1<<BITSABSCURV)
        pOB=0.3*3.8*0.01/(k)
        pts.append(pOB)
        pINT = int(round(pOB/ptLSB))
        if pINT<(1<<BITSPT):
            ptLUT.append(str(pINT))
        else:
            ptLUT.append(str((1<<BITSPT)-1))


def GetEtaLUT():
    for i in range(0,(1<<(BITSTTTANL))):
        tanL = (maxTanL*i)/(1<<BITSTTTANL)
        lam =math.atan(tanL)
        theta =math.pi/2.0-lam
        eta = -math.log(math.tan(theta/2.0))
        etas.append(eta)
        etaINT = int(round(eta*(1<<BITSETA)/math.pi))
        if abs(eta<math.pi):
            etaLUT.append(str(etaINT))

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def GetLUTwrtLSB(fltLUT, lsb, isPT=False, minshift=0,maxshift=5,
                 lowerbound=None, upperbound=None, nbits=None):
    length = len(fltLUT)
    steps  = np.diff(fltLUT)
    if isPT:
        steps  = -1* steps
    if nbits is None:
        nbits = range(minshift, maxshift)
    cross = length - np.searchsorted(steps[::-1], [lsb/(2**i) for i in nbits])

    x = []
    val = []
    shifts = []
    bfrom = 0
    for i, nb in enumerate(nbits):
        shifty_= 0
        bitcross = cross[i]
        if bitcross == 1:
            continue
        if nb == nbits[-1]:
            bitcross = length
            # bitcross = length if isPT else length-1
        orgx = np.arange(bfrom, bitcross)
        val_ = np.take(fltLUT, orgx)
        if upperbound is not None and np.any(val_ >= upperbound):
            sel = orgx[val_>=upperbound]
            uppershift = -1 if isPT else -2
            ## sel[-1]+1 , since we will use < in the final function
            sht = [sel[0], sel[-1]+1, uppershift, 0, int(upperbound/lsb)]
            shifts.append(sht)
            orgx = orgx[val_<=upperbound]
            val_ = val_[val_<=upperbound]
            if isPT:
                bfrom=orgx[0]
            else:
                bitcross=orgx[-1]
        if lowerbound is not None and np.any(val_ <= lowerbound):
            sel = orgx[val_<=lowerbound]
            lowershift = -2 if isPT else -1
            sht = [sel[0], sel[-1]+1, lowershift, 0, lowerbound/lsb]
            shifts.append(sht)
            orgx = orgx[val_>=lowerbound]
            val_ = val_[val_>= lowerbound]
            bitcross=orgx[-1]

        if nb > 1:
            ### Important: We can only shift by nb-1 bit to keep the precision
            shiftx_ = ((orgx >> (nb-1)) )
            if len(shiftx_) == 0:
                continue
            if len(x) > 0:
                shifty_ = int(x[-1] + 1 - shiftx_[0]) ## +1 to make sure it won't overlap
                shiftx_ = shiftx_ + shifty_
        else:
            shiftx_ = orgx
        x_, pickx_ = np.unique(shiftx_, return_index=True)
        val_ = np.take(val_, pickx_) 
        x = np.append(x, x_)
        val = np.append(val, val_)
        if nb == 0:
            sht = [bfrom, bitcross+1, 0, shifty_, 0 ]
        else:
            sht = [bfrom, bitcross+1, nb-1, shifty_, 0 ]
        shifts.append(sht)
        # print("Shifting {nbit} bits with intercept {itsect}, input start from {bfrom} ({ffrom}) to {bto} ({fto}), LUT size {nLUT} ".format(
            # nbit=nb, itsect=shifty_, bfrom=bfrom, bto=bitcross, ffrom=fltLUT[bfrom], fto=fltLUT[bitcross],  nLUT =len(val_)))
        bfrom = bitcross+1

    return shifts

def Modification(inval, intINT, config):
    for cfg in config:
        if inval >= cfg[0] and inval < cfg[1]:
            if cfg[2] < 0 and cfg[4]!=0:
                if cfg[2] == -1:
                    return cfg[2], -1, cfg[4]
                if cfg[2] == -2:
                    return cfg[2], 9999999, cfg[4]
            elif cfg[2] < 0 and cfg[4]==0:
                return cfg[2], inval, intINT
            else:
                return cfg[2], (inval >> cfg[2] ) + cfg[3], intINT

def GetLUTModified(orgLUT, shiftMap, isPT=False):
    tempmap = defaultdict(list)
    x= []
    y= []
    for i, pINT in enumerate(orgLUT):
        ii = i
        if isPT:
            ii+=1
        nshift, newidx, newINT = Modification(ii, pINT, shiftMap)
        tempmap[newidx].append(newINT)

    con = consecutive(list(tempmap.keys()))
    for k, v in tempmap.items():
        if k == -1 or k == 9999999:
            continue
        setv = set(v)
        x.append(k)
        if len(setv) == 1:
            y.append(v[0])
        elif len(setv) == 2 or len(setv) ==3:
            contv = Counter(v)
            ## The counter was sorted, descending in python3 and asending in python2
            ## This will result in slightly different LUT when running 
            isallequal = (len(set(contv.values())) == 1)
            if isallequal:
                ## Using min for now. To be decided
                y.append(min(contv.keys()))
            else:
                y.append(contv.most_common(1)[0][0])
        else:
            print("----- allow up to 3 values per bins")
    return x, y

def ProducedFinalLUT(LUT, k, isPT=False, bounderidx=False):
    k = np.asarray(k).astype(int)
    if isPT:
        k[:, [0, 1]] +=1
        k[k[:, 2] > 0, 3] +=1
    x, y = GetLUTModified(LUT, k,isPT)
    if x[0] != 0:
        for i in k:
            if i[2] < 0:
                continue
            else:
                i[3] -= x[0]
    k = k[k[:, 0].argsort()]
    x, y = GetLUTModified(LUT, k, isPT)
    if bounderidx:
        ### Has 
        if np.any(k[:,2] == -1):
            y.insert(0, str(k[k[:,2] == -1, 4][0]))
            k[k[:,2] == -1, 4] = 0
            k[k[:, 2] >= 0, 3] +=1
        if np.any(k[:,2] == -2):
            y.append(str(k[k[:,2] == -2, 4][0]))
            k[k[:,2] == -2, 4] = len(y)-1
    return k, x, y

### PT
def LookUp(inpt, shiftmap, LUT, bounderidx):
    for i in shiftmap:
        if inpt >=i[0] and inpt < i[1]:
            if i[2] < 0:
                if bounderidx:
                    return i[4], LUT[i[4]]
                else:
                    return -1, i[4]
            else:
                return (inpt >> i[2])+i[3], LUT[(inpt >> i[2])+i[3]]

def ptChecks(shiftmap, LUT, bounderidx=False):
    for i in range(1,(1<<BITSABSCURV)-1):
        k = (maxCurv*i)/(1<<BITSABSCURV)
        pOB=0.3*3.8*0.01/(k)
        idx, pINT = LookUp(i, shiftmap, LUT, bounderidx)
        ## We don't need to check beyond the boundary
        if pOB > (1<<BITSPT)*0.025 or pOB < 2:
            continue
        # Allow +-1 1LSB
        if (abs(pOB - float(pINT)*0.025) > 0.025 ):
            print("pt : ", i, pOB, pts[i-1], ptLUT[i-1], idx, pINT, int(pINT)*0.025)

def etaChecks(shiftmap, LUT, bounderidx=False):
    for i in range(0,(1<<(BITSTTTANL))):
        tanL = (maxTanL*i)/(1<<BITSTTTANL)
        lam =math.atan(tanL)
        theta =math.pi/2.0-lam
        eta = -math.log(math.tan(theta/2.0))
        ## We don't need to check beyond the boundary
        if eta > 2.45:
            continue
        eINT = int(eta*(1<<BITSETA)/math.pi)
        idx, etaINT = LookUp(i, shiftmap, LUT, bounderidx)
        if eta < 1.59 and (abs(eta - int(etaINT)*etaLSB) > etaLSB  ):
            print("eta : ", i, eta, eINT, idx, etaINT, int(etaINT)*etaLSB)
        ## For high eta region, we allow up to 2LSB
        if eta >= 1.59 and (abs(eta - int(etaINT)*etaLSB) > etaLSB * 2 ):
            print("eta : ", i, eta, eINT, idx, etaINT, int(etaINT)*etaLSB)


def PrintPTLUT(k, ptLUT):
    shiftout = []
    for i in k:
        ii = [str(j) for j in i]
        temp = ",".join(ii)
        shiftout.append("{" + temp +"}")
    print("int ptShifts[{nOps}][5]={{".format(nOps=len(k)) + ", ".join(shiftout) + "};")
    print("ap_uint<BITSPT> ptLUT[{address}]={{".format(address=len(ptLUT))+', '.join(ptLUT)+'};')

def PrintEtaLUT(k, etaLUT):
    shiftout = []
    for i in k:
        ii = [str(j) for j in i]
        temp = ",".join(ii)
        shiftout.append("{" + temp +"}")
    print("int etaShifts[{nOps}][5]={{".format(nOps=len(k)) + ", ".join(shiftout) + "};")
    print("ap_uint<BITSETA> etaLUT[{address}]={{".format(address=len(etaLUT)) +', '.join(etaLUT)+'};')

if __name__ == "__main__":
    bounderidx=True
    GetPtLUT()
    k = GetLUTwrtLSB(pts, ptLSB, isPT=True, nbits=[ 1, 2, 3, 4, 5, 6, 7], lowerbound=2, upperbound=((1<<BITSPT)-1)*ptLSB)
    k, x, y = ProducedFinalLUT(ptLUT, k, isPT=True, bounderidx=bounderidx)
    con = consecutive(x)
    if len(con) > 1:
        print("index is not continuous: ", con)
    # ptChecks(k, y, bounderidx=bounderidx)
    # print("Total size of LUT is %d" % len(y))
    PrintPTLUT(k, y)

    # # ### Eta
    GetEtaLUT()
    k =  GetLUTwrtLSB(etas, etaLSB, isPT=False, nbits=[0, 1, 2, 3, 5], upperbound =2.45)
    k, x, y = ProducedFinalLUT(etaLUT, k, bounderidx=bounderidx)
    con = consecutive(x)
    if len(con) > 1:
        print("index is not continuous: ", con)
    # etaChecks(k, y, bounderidx=bounderidx)
    # print("Total size of LUT is %d" % len(y))
    PrintEtaLUT(k, y)
