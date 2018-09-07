from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import bmtfKalmanTrackingSettings as settings

initialK = settings.initialK
initialK2 = settings.initialK2

def bits(number, size_in_bits):
    """
    The bin() function is *REALLY* unhelpful when working with negative numbers.
    It outputs the binary representation of the positive version of that number
    with a '-' at the beginning. Woop-di-do. Here's how to derive the two's-
    complement binary of a negative number:

        complement(bin(+n - 1))

    `complement` is a function that flips each bit. `+n` is the negative number
    made positive.

    """
    if number < 0:
        return compliment(bin(abs(number) - 1)[2:]).rjust(size_in_bits, '1')
    else:
        return bin(number)[2:].rjust(size_in_bits, '0')

def compliment(value):
    return ''.join(COMPLEMENT[x] for x in value)

COMPLEMENT = {'1': '0', '0': '1'}



for i in range(0,4):
    arr = []
    for k in range(0,1024):
        arr.append(0)
    for phiB in range(-512,512):
        address = int(bits(phiB,10),2)       
        factor=phiB    
        if i in [2,3]:
            if abs(phiB)>63:
                factor = 63*phiB/abs(phiB)
        if i in [1,0]:
            if abs(phiB)>127:
                factor = 127*phiB/abs(phiB)
        if factor!=0:         
            charge= phiB/abs(phiB)
        else:
            charge=0;
        
        K = int(initialK[i]*8*factor/(1+initialK2[i]*8*charge*factor))
        if K>8191:
            K=8191
        if K<-8191:
            K=-8191
        arr[address]=str(K)    
#    import pdb;pdb.set_trace()    
    print('\n\n\n')
    if i in [0,1]:
        lut = 'ap_int<14> initialK_'+str(i+1)+'[1024] = {'+','.join(arr)+'};'
    if i in [2,3]:
        lut = 'ap_int<14> initialK_'+str(i+1)+'[1024] = {'+','.join(arr)+'};'
    print(lut)

        



