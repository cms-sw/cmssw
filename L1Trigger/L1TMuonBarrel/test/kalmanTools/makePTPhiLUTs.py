from __future__ import print_function
from math import pi,floor


print(int(((-330+1024)*pi/(6.0*2048.0))/(0.625*pi/180.0)))



#phi=[]
#for i in range(0,2048):
#    p = int((i*pi/(6.0*2048.0)+15.0*pi/180.0)/(0.625*pi/180.0))
#    p = int((i*2*pi/(6.0*2048.0))/(0.625*pi/180.0))
#    phi.append(str(p))

#print('const ap_int<8> phiLUT[2047] = {'+','.join(phi)+'};')
#import pdb;pdb.set_trace()









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







phiLUT=[]
kPHI = 57.2958/0.625/1024.;

for i in range(0,1024):
    phiLUT.append(0)
for phi in range(-512,512):
    address = int(bits(phi,10),2)       
    phiF=float(phi)
    phiNew = 24+int(floor(kPHI*phiF));
    if phiNew >  69: 
        phiNew =  69;
    if phiNew < -8: 
        phiNew = -8;
    phiLUT[address]=(str(phiNew))
print('const ap_int<8> phiLUT[1024] = {'+','.join(phiLUT)+'};')
    


ptLUT=[]
lsb = 1.25/(1<<13)
for i in range(0,4096):
    ptLUT.append(6)
for K in range(-2048,2048):
    address = int(bits(K,12),2)       
    if K>=0:
        charge=1
    else:
        charge=-1
    

    
    FK=lsb*abs(K)

    if abs(K)>2047:
        FK=lsb*2047

    if abs(K)<26:
        FK=lsb*26

    FK = 0.898*FK/(1.0-0.6*FK);
    FK=FK-26.382*FK*FK*FK*FK*FK;
    FK=FK-charge*1.408e-3;
    FK=FK/1.17;
    if (FK!=0.0):
        pt=int(2.0/FK)
    else:
        pt=511

    if pt>511:
        pt=511

    if pt<6:
        pt=6;
    ptLUT[address]=str(pt)    


    
 
print('const ap_uint<9> ptLUT[4096] = {'+','.join(ptLUT)+'};')
