from __future__ import print_function
from math import pi


print(int(((-330+1024)*pi/(6.0*2048.0))/(0.625*pi/180.0)))



phi=[]
for i in range(0,2048):
#    p = int((i*pi/(6.0*2048.0)+15.0*pi/180.0)/(0.625*pi/180.0))
    p = int((i*2*pi/(6.0*2048.0))/(0.625*pi/180.0))
    phi.append(str(p))

print('const ap_int<8> phiLUT[2047] = {'+','.join(phi)+'};')
#import pdb;pdb.set_trace()
pt=[]
lsb = 1.25/(1<<13)
for i in range(0,4096):
    K=i
    if i<22:
        K=22;
    ptF = (2*(1.0/(lsb*float(K))));
    KF=1.0/ptF
    KF = 0.797*KF+0.454*KF*KF-5.679e-4;
    if KF!=0:
        p=int(1.0/KF);
    else:
        p=511
    if p>511:
        p=511;
    pt.append(str(p))
    
 
print('const ap_uint<9> ptLUT[4096] = {'+','.join(pt)+'};')
