#!/usr/bin/env python

import math
import sys
import random

NSector = 9
rcrit = 55.0

rinvmax=0.0057
d0max=3.0;
tdisk=2.4
tLLD =1.7

#If true it will use a TrackletProcessor to replace TEs, TCs
combinedTP=False

#If true it will use a MatchProcessor to replace PRs, MEs, MC
combinedMP=False

# if true use L2L3 seeding
extraseeding=True

# if true generate displaced seeding
displacedseeding=False

two_pi=8*math.atan(1.0)

rlayers = [25.0 , 37.0 , 50.0, 68.0, 80.0, 110.0 ]

rmaxdisk=110.0

rpsmax = 67.0

zdisks = [131.2, 155.0, 185.34, 221.62, 265.0 ]

nallstubslayers = [ 8, 4, 4, 4, 4, 4 ]

nvmtelayers = [4, 8, 4, 8, 4, 8 ]

nvmteextralayers = [-1, 4, 4, -1, -1, -1 ] #only L2&L3 used

nallprojlayers = [ 8, 4, 4, 4, 4, 4 ]

nvmmelayers = [4, 8, 8, 8, 8, 8 ]

nallstubsdisks = [4, 4, 4, 4, 4 ]

nvmtedisks = [4, 4, 4, 4, 4 ]

nallprojdisks = [ 4, 4, 4, 4, 4 ]

nvmmedisks = [8, 4, 4, 4, 4 ]

#for seeding in L1D1 L2D1
nallstubsoverlaplayers = [ 8, 4] 
nvmteoverlaplayers = [2, 2]

nallstubsoverlapdisks = [4] 
nvmteoverlapdisks = [4]

#displaced configs

# layers
#currently use the same VM divisions as prompt layers
dispLLL = [[3,4,2],[5,6,4]] 

# disks to layer overlap
# use prompt VM divisions for disk and overlap divisions for layer 
dispDDL = [[1,2,2]]

# layers to disk overlap
# use prompt VM divisions for layers and overlap divisions for disk 
dispLLD = [[2,3,1]]

xx="" # can be set to "XX"

def phiRange():

    phicrit=math.asin(0.5*rinvmax*rcrit)

    phimax=0.0
    
    for r in rlayers :
        dphi=math.fabs(math.asin(0.5*rinvmax*r)-phicrit)
        if dphi>phimax :
            phimax=dphi
    
    return two_pi/NSector+2*phimax


phirange=phiRange()

print "phi ranage : ",phirange

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

def letter(i) :
    if i==1 :
        return "A"
    if i==2 :
        return "B"
    if i==3 :
        return "C"
    if i==4 :
        return "D"
    if i==5 :
        return "E"
    if i==6 :
        return "F"
    if i==7 :
        return "G"
    if i==8 :
        return "H"
    if i==9 :
        return "I"
    if i==10 :
        return "J"
    if i==11 :
        return "K"
    if i==12 :
        return "L"
    if i==13 :
        return "M"
    if i==14 :
        return "N"
    if i==15 :
        return "O"
    return "letter can not handle input = "+str(i)

def letter_triplet(i) :
    if i==1 :
        return "a"
    if i==2 :
        return "b"
    if i==3 :
        return "c"
    if i==4 :
        return "d"
    if i==5 :
        return "e"
    if i==6 :
        return "f"
    if i==7 :
        return "g"
    if i==8 :
        return "h"
    if i==9 :
        return "i"
    if i==10 :
        return "j"
    if i==11 :
        return "k"
    if i==12 :
        return "l"
    if i==13 :
        return "m"
    if i==14 :
        return "n"
    if i==15 :
        return "o"
    return "letter can not handle input = "+str(i)

def letterextra(i) :
    if i==1 :
        return "I"
    if i==2 :
        return "J"
    if i==3 :
        return "K"
    if i==4 :
        return "L"
    return "extraletter can not handle input = "+str(i)


def letteroverlap(i) :
    if i==1 :
        return "X"
    if i==2 :
        return "Y"
    if i==3 :
        return "Z"
    if i==4 :
        return "W"
    if i==5 :
        return "Q"
    if i==6 :
        return "R"
    if i==7 :
        return "S"
    if i==8 :
        return "T"
    return "letteroverlap can not handle input = "+str(i)

def letteroverlap_triplet(i) :
    if i==1 :
        return "x"
    if i==2 :
        return "y"
    if i==3 :
        return "z"
    if i==4 :
        return "w"
    if i==5 :
        return "q"
    if i==6 :
        return "r"
    if i==7 :
        return "s"
    if i==8 :
        return "t"
    return "letteroverlap can not handle input = "+str(i)

def letter_as(s) :
    if( s=="A") :
        return s
    if( s=="B") :
        return s
    if( s=="C") :
        return s
    if( s=="D") :
        return s
    if( s=="E") :
        return s
    if( s=="F") :
        return s
    if( s=="G") :
        return s
    if( s=="H") :
        return s
    #extra seeding VMs point to same AS memories as others
    if( s=="I") :
        return "A"
    if( s=="J") :
        return "B"
    if( s=="K") :
        return "C"
    if( s=="L") :
        return "D"
    #overlap VMs point to same AS memories as others
    if( s=="X") :
        return "A"
    if( s=="Y") :
        return "B"
    if( s=="Z") :
        return "C"
    if( s=="W") :
        return "D"
    if( s=="Q") :
        return "E"
    if( s=="R") :
        return "F"
    if( s=="S") :
        return "G"
    if( s=="T") :
        return "H"
    #same for triplets
    if( s=="a") :
        return "A"
    if( s=="b") :
        return "B"
    if( s=="c") :
        return "C"
    if( s=="d") :
        return "D"
    if( s=="e") :
        return "E"
    if( s=="f") :
        return "F"
    if( s=="g") :
        return "G"
    if( s=="h") :
        return "H"
    #overlap VMs point to same AS memories as others
    if( s=="x") :
        return "A"
    if( s=="y") :
        return "B"
    if( s=="z") :
        return "C"
    if( s=="w") :
        return "D"
    if( s=="q") :
        return "E"
    if( s=="r") :
        return "F"
    if( s=="s") :
        return "G"
    if( s=="t") :
        return "H"
    print "letter_as can not handle input ", s
    return ""
    
def d0(ilayer,phiinner,phiouter,maxrinv)   :
    d01 = -rlayers[ilayer-1]*rlayers[ilayer]/(rlayers[ilayer]-rlayers[ilayer-1])*(phiouter-phiinner)
    d02 = rlayers[ilayer-1]*rlayers[ilayer]*maxrinv / 2.
    return [d01-d02, d01+d02]

def d0disk(idisk,phiinner,phiouter,maxrinv)   :
    r1  = zdisks[idisk-1]/tdisk;
    r2  = zdisks[idisk]  /tdisk;
    d01 = -r1*r2/(r2-r1)*(phiouter-phiinner)
    d02 = r1*r2*maxrinv / 2.
    return [d01-d02, d01+d02]
    
def rinv5(ilayer,phiinner,phiouter, d0) :
    return 2*(phiinner-phiouter)/(rlayers[ilayer-1]-rlayers[ilayer]) + 2*d0/(rlayers[ilayer-1]*rlayers[ilayer])

def rinv5disk(idisk,phiinner,phiouter, d0) :
    r1  = zdisks[idisk-1]/tdisk;
    r2  = zdisks[idisk]  /tdisk;
    return 2*(phiinner-phiouter)/(r1-r2) + 2*d0/(r1*r2)

def phiproj5(r,phi,rinv,d0, rproj) :
    dphi = 0.5*(rproj-r)*rinv-d0*(rproj-r)/r/rproj
    return phi+dphi

def phiproj5range(ilayer,phiinner,phiouter, maxrinv, maxd0, rproj) :

    d1 = d0(ilayer,phiinner,phiouter,maxrinv)
    #fp.write("\n----------------\n   "+str(maxrinv)+" "+str(maxd0)+" "+str(rproj)+" : "+str(d1)+"\n")
    if d1[0]>maxd0 or d1[1]<-maxd0 :
        return [-10, -10]
    prinv = maxrinv
    nrinv = -maxrinv
    if d1[0]< -maxd0 :
        d1[0] = -maxd0
    if d1[1]> maxd0 :
        d1[1] = maxd0
    prinv = rinv5(ilayer,phiinner,phiouter,d1[1])
    nrinv = rinv5(ilayer,phiinner,phiouter,d1[0])
    phi1 = phiproj5(rlayers[ilayer],phiouter, nrinv,d1[0],rproj)
    phi2 = phiproj5(rlayers[ilayer],phiouter, prinv,d1[1],rproj)

    phirange = []
    if phi1 < phi2 :
        phirange = [phi1, phi2]
    else :
        phirange = [phi2, phi1]
    return phirange

def circle(r1,phi1,r2,phi2,r3,phi3) :
    x1 = r1 * math.cos(phi1)
    x2 = r2 * math.cos(phi2)
    x3 = r3 * math.cos(phi3)
    y1 = r1 * math.sin(phi1)
    y2 = r2 * math.sin(phi2)
    y3 = r3 * math.sin(phi3)
    k1 = - (x2-x1)/(y2-y1)
    k2 = - (x3-x2)/(y3-y2)
    b1 = 0.5*(y2+y1)-0.5*(x1+x2)*k1
    b2 = 0.5*(y3+y2)-0.5*(x2+x3)*k2
    y0 = (b1*k2-b2*k1)/(k2-k1)
    x0 = (b1-b2)/(k2-k1)
    R  = math.sqrt(pow(x1-x0,2)+pow(y1-y0,2))
    rinv = 1./R
    d0 = R - math.sqrt(x0*x0+y0*y0)
    return [rinv, d0]
    
def phiproj5projrange(stlist, rproj) :
    phimax = 0
    phimin = phirange
    for stname in stlist :
        l1 = int(stname[4])
        l2 = int(stname[7])
        l3 = int(stname[11])
        n1 = nallstubslayers[l1-1] if stname[3]  == "L" else nallstubsdisks[l1-1]
        n2 = nallstubslayers[l2-1] if stname[6]  == "L" else nallstubsdisks[l2-1]
        n3 = nallstubslayers[l3-1] if stname[10] == "L" else nallstubsdisks[l3-1]

        t = tdisk if stname[3]  == "D" else tLLD
        
        r1 = rlayers[l1-1] if stname[3]  == "L" else zdisks[l1-1]/t
        r2 = rlayers[l2-1] if stname[6]  == "L" else zdisks[l2-1]/t
        r3 = rlayers[l3-1] if stname[10] == "L" else zdisks[l3-1]/t

        vm1 = ord(letter_as(stname[5]))-ord("A")
        vm2 = ord(letter_as(stname[8]))-ord("A")
        vm3s = stname[12:].split("_")[0]
        for vm3c in vm3s :
            vm3 = ord(letter_as(vm3c))-ord("A")
            phi1  = phirange * vm1 / n1
            dphi1 = phirange       / n1
            phi2  = phirange * vm2 / n2
            dphi2 = phirange       / n2
            phi3  = phirange * vm3 / n3
            dphi3 = phirange       / n3

            #print l1, l2, l3, " : ",vm1, vm2, vm3
            ntries = 100
            niter  = 100000
            while ntries > 0 and niter > 0:
                niter = niter-1
                p1 = phi1 + dphi1*random.random()
                p2 = phi2 + dphi2*random.random()
                p3 = phi3 + dphi3*random.random()
                c = circle(r1,p1,r2,p2,r3,p3)
                if abs(c[1]) < d0max and abs(c[0])<rinvmax :
                    ntries = ntries-1
                    p = phiproj5(r3,p3, c[0], c[1], rproj)
                    if p < phimin:
                        phimin = p
                    if p > phimax:
                        phimax = p
            if niter == 0:
                print "Reached max number of tries: bad triplet??",ntries,l1,l2,l3,rproj
                print stlist
                
                        
    return [phimin, phimax]
                

def phiproj5diskrange(idisk,phiinner,phiouter, maxrinv, maxd0, rproj) :

    d1 = d0disk(idisk,phiinner,phiouter,maxrinv)
    #fp.write("\n----------------\n   "+str(maxrinv)+" "+str(maxd0)+" "+str(rproj)+" : "+str(d1)+"\n")
    if d1[0]>maxd0 or d1[1]<-maxd0 :
        return [-10, -10]
    prinv = maxrinv
    nrinv = -maxrinv
    if d1[0]< -maxd0 :
        d1[0] = -maxd0
    if d1[1]> maxd0 :
        d1[1] = maxd0
    prinv = rinv5disk(idisk,phiinner,phiouter,d1[1])
    nrinv = rinv5disk(idisk,phiinner,phiouter,d1[0])
    phi1 = phiproj5(zdisks[idisk]/tdisk,phiouter, nrinv,d1[0],rproj)
    phi2 = phiproj5(zdisks[idisk]/tdisk,phiouter, prinv,d1[1],rproj)

    phirange = []
    if phi1 < phi2 :
        phirange = [phi1, phi2]
    else :
        phirange = [phi2, phi1]
    return phirange

def phiproj5stlayer(ilayer,projlayer, ivminner, ivmouter) :

    dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
    dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

    phiinner2=dphiinner*ivminner
    phiinner1=phiinner2-dphiinner
    
    phiouter2=dphiouter*ivmouter
    phiouter1=phiouter2-dphiouter

    p11=phiproj5range(ilayer,phiinner1,phiouter1,rinvmax, d0max, rlayers[projlayer-1])
    p12=phiproj5range(ilayer,phiinner1,phiouter2,rinvmax, d0max, rlayers[projlayer-1])
    p21=phiproj5range(ilayer,phiinner2,phiouter1,rinvmax, d0max, rlayers[projlayer-1])
    p22=phiproj5range(ilayer,phiinner2,phiouter2,rinvmax, d0max, rlayers[projlayer-1])

    #fp.write("\n"+str(phiinner1)+" "+str(phiouter1)+" : "+str(p11)+" "+str(p12)+" "+str(p21)+" "+str(p22)+"\n")
    
    minp=phirange
    maxp=0

    if p11[0]>-9 :
        if p11[0] < minp :
            minp = p11[0]
        if p11[1] > maxp :
            maxp = p11[1]

    if p12[0]>-9 :
        if p12[0] < minp :
            minp = p12[0]
        if p12[1] > maxp :
            maxp = p12[1]

    if p21[0]>-9 :
        if p21[0] < minp :
            minp = p21[0]
        if p21[1] > maxp :
            maxp = p21[1]

    if p22[0]>-9 :
        if p22[0] < minp :
            minp = p22[0]
        if p22[1] > maxp :
            maxp = p22[1]


    return [minp,maxp]

def phiproj5stlayer_to_disk(ilayer,projdisk, ivminner, ivmouter) :

    dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
    dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

    phiinner2=dphiinner*ivminner
    phiinner1=phiinner2-dphiinner
    
    phiouter2=dphiouter*ivmouter
    phiouter1=phiouter2-dphiouter

    p11=phiproj5range(ilayer,phiinner1,phiouter1,rinvmax, d0max, zdisks[projdisk-1]/tLLD)
    p12=phiproj5range(ilayer,phiinner1,phiouter2,rinvmax, d0max, zdisks[projdisk-1]/tLLD)
    p21=phiproj5range(ilayer,phiinner2,phiouter1,rinvmax, d0max, zdisks[projdisk-1]/tLLD)
    p22=phiproj5range(ilayer,phiinner2,phiouter2,rinvmax, d0max, zdisks[projdisk-1]/tLLD)

    #fp.write("\n"+str(phiinner1)+" "+str(phiouter1)+" : "+str(p11)+" "+str(p12)+" "+str(p21)+" "+str(p22)+"\n")
    
    minp=phirange
    maxp=0

    if p11[0]>-9 :
        if p11[0] < minp :
            minp = p11[0]
        if p11[1] > maxp :
            maxp = p11[1]

    if p12[0]>-9 :
        if p12[0] < minp :
            minp = p12[0]
        if p12[1] > maxp :
            maxp = p12[1]

    if p21[0]>-9 :
        if p21[0] < minp :
            minp = p21[0]
        if p21[1] > maxp :
            maxp = p21[1]

    if p22[0]>-9 :
        if p22[0] < minp :
            minp = p22[0]
        if p22[1] > maxp :
            maxp = p22[1]


    return [minp,maxp]

def phiproj5stdisk_to_layer(idisk,projlayer, ivminner, ivmouter) :

    dphiinner=phirange/nallstubsdisks[idisk-1]/nvmtedisks[idisk-1]
    dphiouter=phirange/nallstubsdisks[idisk]/nvmtedisks[idisk]

    phiinner2=dphiinner*ivminner
    phiinner1=phiinner2-dphiinner
    
    phiouter2=dphiouter*ivmouter
    phiouter1=phiouter2-dphiouter

    p11=phiproj5diskrange(idisk,phiinner1,phiouter1,rinvmax, d0max, rlayers[projlayer-1])
    p12=phiproj5diskrange(idisk,phiinner1,phiouter2,rinvmax, d0max, rlayers[projlayer-1])
    p21=phiproj5diskrange(idisk,phiinner2,phiouter1,rinvmax, d0max, rlayers[projlayer-1])
    p22=phiproj5diskrange(idisk,phiinner2,phiouter2,rinvmax, d0max, rlayers[projlayer-1])

    #fp.write("\n"+str(phiinner1)+" "+str(phiouter1)+" : "+str(p11)+" "+str(p12)+" "+str(p21)+" "+str(p22)+"\n")
    
    minp=phirange
    maxp=0

    if p11[0]>-9 :
        if p11[0] < minp :
            minp = p11[0]
        if p11[1] > maxp :
            maxp = p11[1]

    if p12[0]>-9 :
        if p12[0] < minp :
            minp = p12[0]
        if p12[1] > maxp :
            maxp = p12[1]

    if p21[0]>-9 :
        if p21[0] < minp :
            minp = p21[0]
        if p21[1] > maxp :
            maxp = p21[1]

    if p22[0]>-9 :
        if p22[0] < minp :
            minp = p22[0]
        if p22[1] > maxp :
            maxp = p22[1]


    return [minp,maxp]

    
def rinv(ilayer,phiinner,phiouter) :
    return 2*math.sin(phiinner-phiouter)/(rlayers[ilayer-1]-rlayers[ilayer])

def rinvdisk(idisk,phiinner,phiouter) :
    return 2*math.sin(phiinner-phiouter)/(rpsmax*(zdisks[idisk-1]-zdisks[idisk])/zdisks[idisk-1])

def validtepair(ilayer,ivminner,ivmouter) :

    dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
    dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    rinv11=rinv(ilayer,phiinner1,phiouter1)
    rinv12=rinv(ilayer,phiinner1,phiouter2)
    rinv21=rinv(ilayer,phiinner2,phiouter1)
    rinv22=rinv(ilayer,phiinner2,phiouter2)
    
    #print rinv11,rinv12,rinv21,rinv22

    if rinv11>rinvmax and rinv12>rinvmax and rinv21>rinvmax and rinv22>rinvmax :
        return False

    if rinv11<-rinvmax and rinv12<-rinvmax and rinv21<-rinvmax and rinv22<-rinvmax :
        return False
    
    return True

def validtepairextra(ilayer,ivminner,ivmouter) :

    dphiinner=phirange/nallstubslayers[ilayer-1]/nvmteextralayers[ilayer-1]
    dphiouter=phirange/nallstubslayers[ilayer]/nvmteextralayers[ilayer]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    rinv11=rinv(ilayer,phiinner1,phiouter1)
    rinv12=rinv(ilayer,phiinner1,phiouter2)
    rinv21=rinv(ilayer,phiinner2,phiouter1)
    rinv22=rinv(ilayer,phiinner2,phiouter2)
    
    #print rinv11,rinv12,rinv21,rinv22

    if rinv11>rinvmax and rinv12>rinvmax and rinv21>rinvmax and rinv22>rinvmax :
        return False

    if rinv11<-rinvmax and rinv12<-rinvmax and rinv21<-rinvmax and rinv22<-rinvmax :
        return False
    
    return True


def validtepairdisk(idisk,ivminner,ivmouter) :

    dphiinner=phirange/nallstubsdisks[idisk-1]/nvmtedisks[idisk-1]
    dphiouter=phirange/nallstubsdisks[idisk]/nvmtedisks[idisk]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    rinv11=rinvdisk(idisk,phiinner1,phiouter1)
    rinv12=rinvdisk(idisk,phiinner1,phiouter2)
    rinv21=rinvdisk(idisk,phiinner2,phiouter1)
    rinv22=rinvdisk(idisk,phiinner2,phiouter2)
    
    #print rinv11,rinv12,rinv21,rinv22

    if rinv11>rinvmax and rinv12>rinvmax and rinv21>rinvmax and rinv22>rinvmax :
        return False

    if rinv11<-rinvmax and rinv12<-rinvmax and rinv21<-rinvmax and rinv22<-rinvmax :
        return False
    
    return True


def validtepairoverlap(ilayer,ivminner,ivmouter) :

    dphiinner=phirange/nallstubsoverlaplayers[ilayer-1]/nvmteoverlaplayers[ilayer-1]
    dphiouter=phirange/nallstubsoverlapdisks[0]/nvmteoverlapdisks[0]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    rinv11=rinv(ilayer,phiinner1,phiouter1)
    rinv12=rinv(ilayer,phiinner1,phiouter2)
    rinv21=rinv(ilayer,phiinner2,phiouter1)
    rinv22=rinv(ilayer,phiinner2,phiouter2)
    
    #print rinv11,rinv12,rinv21,rinv22

    if rinv11>rinvmax and rinv12>rinvmax and rinv21>rinvmax and rinv22>rinvmax :
        return False

    if rinv11<-rinvmax and rinv12<-rinvmax and rinv21<-rinvmax and rinv22<-rinvmax :
        return False
    
    return True

def validtedpair(ilayer,ivminner,ivmouter) :

    dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
    dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

    phiinner2=dphiinner*ivminner
    phiinner1=phiinner2-dphiinner
    
    phiouter2=dphiouter*ivmouter
    phiouter1=phiouter2-dphiouter

    d11=d0(ilayer,phiinner1,phiouter1,rinvmax)
    d12=d0(ilayer,phiinner1,phiouter2,rinvmax)
    d21=d0(ilayer,phiinner2,phiouter1,rinvmax)
    d22=d0(ilayer,phiinner2,phiouter2,rinvmax)
    
    if d11[0]>d0max and d12[0]>d0max and d21[0]>d0max and d22[0]>d0max and d11[1]>d0max and d12[1]>d0max and d21[1]>d0max and d22[1]>d0max :
        return False
    if d11[0]<-d0max and d12[0]<-d0max and d21[0]<-d0max and d22[0]<-d0max and d11[1]<-d0max and d12[1]<-d0max and d21[1]<-d0max and d22[1]<-d0max :
        return False
    
    return True

def validtedpairdisk(idisk,ivminner,ivmouter) :

    dphiinner=phirange/nallstubsdisks[idisk-1]/nvmtedisks[idisk-1]
    dphiouter=phirange/nallstubsdisks[idisk]/nvmtedisks[idisk]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    d11=d0disk(idisk,phiinner1,phiouter1,rinvmax)
    d12=d0disk(idisk,phiinner1,phiouter2,rinvmax)
    d21=d0disk(idisk,phiinner2,phiouter1,rinvmax)
    d22=d0disk(idisk,phiinner2,phiouter2,rinvmax)
    
    if d11[0]>d0max and d12[0]>d0max and d21[0]>d0max and d22[0]>d0max and d11[1]>d0max and d12[1]>d0max and d21[1]>d0max and d22[1]>d0max :
        return False
    if d11[0]<-d0max and d12[0]<-d0max and d21[0]<-d0max and d22[0]<-d0max and d11[1]<-d0max and d12[1]<-d0max and d21[1]<-d0max and d22[1]<-d0max :
        return False
   
    return True

def validtedpairoverlap(ilayer,ivminner,ivmouter) :

    dphiinner=phirange/nallstubsoverlaplayers[ilayer-1]/nvmteoverlaplayers[ilayer-1]
    dphiouter=phirange/nallstubsoverlapdisks[0]/nvmteoverlapdisks[0]

    phiinner1=dphiinner*ivminner
    phiinner2=phiinner1+dphiinner
    
    phiouter1=dphiouter*ivmouter
    phiouter2=phiouter1+dphiouter

    d11=d0(ilayer,phiinner1,phiouter1,rinvmax)
    d12=d0(ilayer,phiinner1,phiouter2,rinvmax)
    d21=d0(ilayer,phiinner2,phiouter1,rinvmax)
    d22=d0(ilayer,phiinner2,phiouter2,rinvmax)
    
    if d11[0]>d0max and d12[0]>d0max and d21[0]>d0max and d22[0]>d0max and d11[1]>d0max and d12[1]>d0max and d21[1]>d0max and d22[1]>d0max :
        return False
    if d11[0]<-d0max and d12[0]<-d0max and d21[0]<-d0max and d22[0]<-d0max and d11[1]<-d0max and d12[1]<-d0max and d21[1]<-d0max and d22[1]<-d0max :
        return False
   
    return True

def asmems(sp_list):
    as_list1 = []
    as_list2 = []
    as_list3 = []
    as_list  = []
    for sp in sp_list:
        i = sp.find("PHI");
        j = 0
        while (i >= 0):
           as_i  = letter_as(sp[i+3])
           as_name = "AS_"+sp[i-2:i]+"PHI"+as_i
           if as_name not in as_list:
               as_list.append(as_name)
               if   j==0 :
                   as_list1.append(as_name)
               elif j==1:
                   as_list2.append(as_name)
               elif j==2 :
                   as_list3.append(as_name)
               else:
                   print "too many PHI segments in a name!", sp
           i = sp.find("PHI",i+1);
           j = j+1

    #return as_list       
    return as_list1 + as_list2 + as_list3

def asmems3(st_list):
    as_list  = []
    for st in st_list:
        as_name = "AS_"+st[3:5]+"PHI"+letter_as(st[5])
        if as_name not in as_list:
            as_list.append(as_name)
        as_name = "AS_"+st[6:8]+"PHI"+letter_as(st[8])
        if as_name not in as_list:
            as_list.append(as_name)
        i = st.find("_",12)
        ls = st[12:i]
        for l in ls:
            as_name = "AS_"+st[10:12]+"PHI"+letter_as(l)
            if as_name not in as_list:
                as_list.append(as_name)

    return as_list       


def phiproj(ilayer,phi,rinv,projlayer) :

    dphi=math.asin(0.5*rlayers[projlayer-1]*rinv)-math.asin(0.5*rlayers[ilayer-1]*rinv)

    return phi+dphi;


def phiprojdisk(idisk,phi,rinv,projdisk) :

    rproj=min(rmaxdisk,rpsmax*zdisks[projdisk-1]/zdisks[idisk-1])
    
    dphi=math.asin(0.5*rproj*rinv)-math.asin(0.5*rpsmax*rinv)

    return phi+dphi;

def phiprojdisktolayer(idisk,phi,rinv,projlayer) :

    dphi=math.asin(0.5*rlayers[projlayer-1]*rinv)-math.asin(0.5*rmaxdisk*rinv)

    return phi+dphi;

def phiprojlayertodisk(ilayer,phi,rinv,projdisk) :

    dphi=math.asin(0.5*rmaxdisk*rinv)-math.asin(0.5*rlayers[ilayer-1]*rinv)

    return phi+dphi;

    

def phiprojrange(ilayer, projlayer, splist) :

    projrange=[]
    
    for spname in splist :
        ivminner=int(spname.split("PHI")[1].split("_")[0][1:])
        ivmouter=int(spname.split("PHI")[2][1:])
        #print projlayer, spname, spname.split("PHI"),ivminner,ivmouter

        dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
        dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

        phiinner1=dphiinner*ivminner
        phiinner2=phiinner1+dphiinner
    
        phiouter1=dphiouter*ivmouter
        phiouter2=phiouter1+dphiouter

        rinv11=rinv(ilayer,phiinner1,phiouter1)
        rinv12=rinv(ilayer,phiinner1,phiouter2)
        rinv21=rinv(ilayer,phiinner2,phiouter1)
        rinv22=rinv(ilayer,phiinner2,phiouter2)

        minrinv=rinv11
        maxrinv=rinv11

        if rinv12<minrinv :
            minrinv=rinv12
        if rinv12>maxrinv :
            maxrinv=rinv12
            
        if rinv21<minrinv :
            minrinv=rinv21
        if rinv21>maxrinv :
            maxrinv=rinv21

        if rinv22<minrinv :
            minrinv=rinv22
        if rinv22>maxrinv :
            maxrinv=rinv22

        if minrinv<-rinvmax :
            minrinv=-rinvmax

        if maxrinv>rinvmax :
            maxrinv=rinvmax

        if minrinv>rinvmax :
            minrinv=rinvmax

            
        phiminproj1=phiproj(ilayer,phiinner1,minrinv,projlayer)
        phiminproj2=phiproj(ilayer+1,phiouter1,minrinv,projlayer)

        phimin=min(phiminproj1,phiminproj2)-0.05
        
        phimaxproj1=phiproj(ilayer,phiinner2,maxrinv,projlayer)
        phimaxproj2=phiproj(ilayer+1,phiouter2,maxrinv,projlayer)

        phimax=max(phimaxproj1,phimaxproj2)+0.05

        if len(projrange)==0 :
            projrange=[phimin,phimax]
        else :
            projrange[0]=min(projrange[0],phimin)
            projrange[1]=max(projrange[1],phimax)


    return projrange


def phiprojrangedisk(idisk, projdisk, splist) :

    projrange=[]
    
    for spname in splist :
        ivminner=int(spname.split("PHI")[1].split("_")[0][1:])
        ivmouter=int(spname.split("PHI")[2][1:])
        #print projlayer, spname, spname.split("PHI"),ivminner,ivmouter

        dphiinner=phirange/nallstubsdisks[idisk-1]/nvmtedisks[idisk-1]
        dphiouter=phirange/nallstubsdisks[idisk]/nvmtedisks[idisk]

        phiinner1=dphiinner*ivminner
        phiinner2=phiinner1+dphiinner
    
        phiouter1=dphiouter*ivmouter
        phiouter2=phiouter1+dphiouter

        rinv11=rinvdisk(idisk,phiinner1,phiouter1)
        rinv12=rinvdisk(idisk,phiinner1,phiouter2)
        rinv21=rinvdisk(idisk,phiinner2,phiouter1)
        rinv22=rinvdisk(idisk,phiinner2,phiouter2)

        minrinv=rinv11
        maxrinv=rinv11

        if rinv12<minrinv :
            minrinv=rinv12
        if rinv12>maxrinv :
            maxrinv=rinv12
            
        if rinv21<minrinv :
            minrinv=rinv21
        if rinv21>maxrinv :
            maxrinv=rinv21

        if rinv22<minrinv :
            minrinv=rinv22
        if rinv22>maxrinv :
            maxrinv=rinv22

        if minrinv<-rinvmax :
            minrinv=-rinvmax

        if maxrinv>rinvmax :
            maxrinv=rinvmax

        phiminproj1=phiprojdisk(idisk,phiinner1,minrinv,projdisk)
        phiminproj2=phiprojdisk(idisk+1,phiouter1,minrinv,projdisk)

        phimin=min(phiminproj1,phiminproj2)-0.10
        
        phimaxproj1=phiprojdisk(idisk,phiinner2,maxrinv,projdisk)
        phimaxproj2=phiprojdisk(idisk+1,phiouter2,maxrinv,projdisk)

        phimax=max(phimaxproj1,phimaxproj2)+0.10

        if len(projrange)==0 :
            projrange=[phimin,phimax]
        else :
            projrange[0]=min(projrange[0],phimin)
            projrange[1]=max(projrange[1],phimax)


    return projrange

def phiprojrangedisktolayer(idisk,projlayer,splist) :

    projrange=[]
    
    for spname in splist :
        ivminner=int(spname.split("PHI")[1].split("_")[0][1:])
        ivmouter=int(spname.split("PHI")[2][1:])
        #print projlayer, spname, spname.split("PHI"),ivminner,ivmouter

        dphiinner=phirange/nallstubsdisks[idisk-1]/nvmtedisks[idisk-1]
        dphiouter=phirange/nallstubsdisks[idisk]/nvmtedisks[idisk]

        phiinner1=dphiinner*ivminner
        phiinner2=phiinner1+dphiinner
    
        phiouter1=dphiouter*ivmouter
        phiouter2=phiouter1+dphiouter

        rinv11=rinvdisk(idisk,phiinner1,phiouter1)
        rinv12=rinvdisk(idisk,phiinner1,phiouter2)
        rinv21=rinvdisk(idisk,phiinner2,phiouter1)
        rinv22=rinvdisk(idisk,phiinner2,phiouter2)

        minrinv=rinv11
        maxrinv=rinv11

        if rinv12<minrinv :
            minrinv=rinv12
        if rinv12>maxrinv :
            maxrinv=rinv12
            
        if rinv21<minrinv :
            minrinv=rinv21
        if rinv21>maxrinv :
            maxrinv=rinv21

        if rinv22<minrinv :
            minrinv=rinv22
        if rinv22>maxrinv :
            maxrinv=rinv22

        if minrinv<-rinvmax :
            minrinv=-rinvmax

        if maxrinv>rinvmax :
            maxrinv=rinvmax

        phiminproj1=phiprojdisktolayer(idisk,phiinner1,minrinv,projlayer)
        phiminproj2=phiprojdisktolayer(idisk+1,phiouter1,minrinv,projlayer)

        phimin=min(phiminproj1,phiminproj2)-0.15
        
        phimaxproj1=phiprojdisktolayer(idisk,phiinner2,maxrinv,projlayer)
        phimaxproj2=phiprojdisktolayer(idisk+1,phiouter2,maxrinv,projlayer)

        phimax=max(phimaxproj1,phimaxproj2)+0.15

        if len(projrange)==0 :
            projrange=[phimin,phimax]
        else :
            projrange[0]=min(projrange[0],phimin)
            projrange[1]=max(projrange[1],phimax)


    return projrange


def phiprojrangelayertodisk(ilayer,projdisk,splist) :

    projrange=[]
    
    for spname in splist :
        ivminner=int(spname.split("PHI")[1].split("_")[0][1:])
        ivmouter=int(spname.split("PHI")[2][1:])
        #print projlayer, spname, spname.split("PHI"),ivminner,ivmouter

        dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
        dphiouter=phirange/nallstubslayers[ilayer]/nvmtelayers[ilayer]

        phiinner1=dphiinner*ivminner
        phiinner2=phiinner1+dphiinner
    
        phiouter1=dphiouter*ivmouter
        phiouter2=phiouter1+dphiouter

        rinv11=rinv(ilayer,phiinner1,phiouter1)
        rinv12=rinv(ilayer,phiinner1,phiouter2)
        rinv21=rinv(ilayer,phiinner2,phiouter1)
        rinv22=rinv(ilayer,phiinner2,phiouter2)

        minrinv=rinv11
        maxrinv=rinv11

        if rinv12<minrinv :
            minrinv=rinv12
        if rinv12>maxrinv :
            maxrinv=rinv12
            
        if rinv21<minrinv :
            minrinv=rinv21
        if rinv21>maxrinv :
            maxrinv=rinv21

        if rinv22<minrinv :
            minrinv=rinv22
        if rinv22>maxrinv :
            maxrinv=rinv22

        if minrinv<-rinvmax :
            minrinv=-rinvmax

        if maxrinv>rinvmax :
            maxrinv=rinvmax

        if minrinv>maxrinv :
            minrinv=maxrinv

        if maxrinv<minrinv :
            maxrinv=minrinv

        #print minrinv, maxrinv
            
        phiminproj1=phiprojlayertodisk(ilayer,phiinner1,minrinv,projdisk)
        phiminproj2=phiprojlayertodisk(ilayer+1,phiouter1,minrinv,projdisk)

        phimin=min(phiminproj1,phiminproj2)-0.05
        
        phimaxproj1=phiprojlayertodisk(ilayer,phiinner2,maxrinv,projdisk)
        phimaxproj2=phiprojlayertodisk(ilayer+1,phiouter2,maxrinv,projdisk)

        phimax=max(phimaxproj1,phimaxproj2)+0.05

        if len(projrange)==0 :
            projrange=[phimin,phimax]
        else :
            projrange[0]=min(projrange[0],phimin)
            projrange[1]=max(projrange[1],phimax)


    return projrange

def phiprojrangeoverlaplayertodisk(ilayer,projdisk,splist) :

    projrange=[]
    
    for spname in splist :
        ivminner=int(spname.split("PHI")[1].split("_")[0][1:])
        ivmouter=int(spname.split("PHI")[2][1:])
        #print projlayer, spname, spname.split("PHI"),ivminner,ivmouter

        dphiinner=phirange/nallstubslayers[ilayer-1]/nvmtelayers[ilayer-1]
        dphiouter=phirange/nallstubsdisks[0]/nvmtedisks[0]

        phiinner1=dphiinner*ivminner
        phiinner2=phiinner1+dphiinner
    
        phiouter1=dphiouter*ivmouter
        phiouter2=phiouter1+dphiouter

        rinv11=rinv(ilayer,phiinner1,phiouter1)
        rinv12=rinv(ilayer,phiinner1,phiouter2)
        rinv21=rinv(ilayer,phiinner2,phiouter1)
        rinv22=rinv(ilayer,phiinner2,phiouter2)

        minrinv=rinv11
        maxrinv=rinv11

        if rinv12<minrinv :
            minrinv=rinv12
        if rinv12>maxrinv :
            maxrinv=rinv12
            
        if rinv21<minrinv :
            minrinv=rinv21
        if rinv21>maxrinv :
            maxrinv=rinv21

        if rinv22<minrinv :
            minrinv=rinv22
        if rinv22>maxrinv :
            maxrinv=rinv22

        if minrinv<-rinvmax :
            minrinv=-rinvmax

        if maxrinv>rinvmax :
            maxrinv=rinvmax

        if minrinv>maxrinv :
            minrinv=maxrinv

        if maxrinv<minrinv :
            maxrinv=minrinv

        #print minrinv, maxrinv
            
        phiminproj1=phiprojlayertodisk(ilayer,phiinner1,minrinv,projdisk)
        phiminproj2=phiprojlayertodisk(ilayer+1,phiouter1,minrinv,projdisk)

        phimin=min(phiminproj1,phiminproj2)-0.15
        
        phimaxproj1=phiprojlayertodisk(ilayer,phiinner2,maxrinv,projdisk)
        phimaxproj2=phiprojlayertodisk(ilayer+1,phiouter2,maxrinv,projdisk)

        phimax=max(phimaxproj1,phimaxproj2)+0.15

        if len(projrange)==0 :
            projrange=[phimin,phimax]
        else :
            projrange[0]=min(projrange[0],phimin)
            projrange[1]=max(projrange[1],phimax)


    return projrange

def readSPoccupancy() :
    fi = open("SPmap.txt","r")

    sp_occupancy = {}
    for line in fi:
        spname = line.split()[0]
        occ = float(line.split()[1])/9./100.
        sp_occupancy[spname] = occ

    print sp_occupancy
    return sp_occupancy

def readUnusedProj() :
    fi = open("unusedproj.txt","r")

    unusedproj=[]

    for line in fi:
        unusedproj.append(line.split('\n')[0])

    return unusedproj

def findInputLinks(dtcphirange) :

    fi = open(dtcphirange,"r")

    ilinks=[]
    
    for line in fi:
        dtcname=line.split()[0]
        layerdisk=int(line.split()[1])
        phimin=float(line.split()[2])
        phimax=float(line.split()[3])
        #print "Line: ",dtcname,layerdisk,phimin,phimax
        phimin1=phimin-two_pi/9.0+0.5*phirange
        phimax1=phimax-two_pi/9.0+0.5*phirange
        phimin2=phimin+0.5*phirange
        phimax2=phimax+0.5*phirange

        #print "phimin1, phimax1 : ",phimin1,phimax1
        
        if layerdisk<7 :
            layer=layerdisk
            #print "layer : ",layer
            nallstubs=nallstubslayers[layer-1]
            dphi=phirange/nallstubs
            for iallstub in range(0,nallstubs) :
                phiminallstub=iallstub*dphi
                phimaxallstub=phiminallstub+dphi
                #print "Allstub phimin,max :",phiminallstub,phimaxallstub
                if (phiminallstub<phimax1 and phimaxallstub>phimin1) or (phiminallstub<phimax2 and phimaxallstub>phimin2) :
                    if iallstub<nallstubs/2 :
                        il="IL_L"+str(layer)+"PHI"+letter(iallstub+1)+"_"+dtcname+"_A"
                        ilinks.append(il)
                        #print "Inputlink : ",il
                    if iallstub>=nallstubs/2 :
                        il="IL_L"+str(layer)+"PHI"+letter(iallstub+1)+"_"+dtcname+"_B"
                        ilinks.append(il)
                        #print "Inputlink : ",il
        else :
            disk=layerdisk-6
            #print "layerdisk disk : ",layerdisk,disk
            nallstubs=nallstubsdisks[disk-1]
            dphi=phirange/nallstubs
            for iallstub in range(0,nallstubs) :
                phiminallstub=iallstub*dphi
                phimaxallstub=phiminallstub+dphi
                #print "Allstub phimin,max :",phiminallstub,phimaxallstub
                if (phiminallstub<phimax1 and phimaxallstub>phimin1) or (phiminallstub<phimax2 and phimaxallstub>phimin2) :
                    if iallstub<nallstubs/2 :
                        il="IL_D"+str(disk)+"PHI"+letter(iallstub+1)+"_"+dtcname+"_A"
                        ilinks.append(il)
                        #print "Inputlink : ",il
                    if iallstub>=nallstubs/2 :
                        il="IL_D"+str(disk)+"PHI"+letter(iallstub+1)+"_"+dtcname+"_B"
                        ilinks.append(il)
                        #print "Inputlink : ",il

    return ilinks

inputlinks=findInputLinks("dtcphirange.txt")

print "Inputlinks :",len(inputlinks),inputlinks

unusedproj=readUnusedProj()

print "Unusedproj :",unusedproj


fp = open("wires.input.hourglass","w")

#
# Do the VM routers for the TE in the layers
#

#
# triplets VMs:
# FIRST  (same as inner): L3,L5,D1 ->same memories as pairs;   L2abcdefg for L2L3D1
# SECOND (same as outer): L4,L6,D2 -> same memeories as pairs; L3abcdefg for L2L3D1
# THIRD  (same as outer): L2,L4    -> same memories as pairs;  L2xyz, D1xyz for D1D2L2 and L2L3D1


for ilayer in range(1,7) :
    print "layer =",ilayer,"allstub memories",nallstubslayers[ilayer-1]
    fp.write("\n")
    fp.write("#\n")
    fp.write("# VMRouters for the TEs in layer "+str(ilayer)+" \n")
    fp.write("#\n")
    for iallstubmem in range(1,nallstubslayers[ilayer-1]+1) :
        allstubsmemname="L"+str(ilayer)+"PHI"+letter(iallstubmem)
        for il in inputlinks :
            if allstubsmemname in il :
                fp.write(il+" ")
        fp.write("> VMR_L"+str(ilayer)+"PHI"+letter(iallstubmem)+" > ")
        fp.write("AS_L"+str(ilayer)+"PHI"+letter(iallstubmem))
        for ivm in range(1,nvmmelayers[ilayer-1]+1) :
            fp.write(" VMSME_L"+str(ilayer)+"PHI"+letter(iallstubmem)+str((iallstubmem-1)*nvmmelayers[ilayer-1]+ivm))
        for ivm in range(1,nvmtelayers[ilayer-1]+1) :
            fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letter(iallstubmem)+str((iallstubmem-1)*nvmtelayers[ilayer-1]+ivm))
        if extraseeding :
            if (nvmteextralayers[ilayer-1]>0) :
                for ivm in range(1,nvmteextralayers[ilayer-1]+1) :
                    fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letterextra(iallstubmem)+str((iallstubmem-1)*nvmteextralayers[ilayer-1]+ivm))
        if ilayer in range(1,3) :
            for ivm in range(1,nvmteoverlaplayers[ilayer-1]+1) :
                fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letteroverlap(iallstubmem)+str((iallstubmem-1)*nvmteoverlaplayers[ilayer-1]+ivm))
        if displacedseeding :
            if ilayer == 2:
                for ivm in range(1,nvmtelayers[ilayer-1]+1) :
                    fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letter_triplet(iallstubmem)+str((iallstubmem-1)*nvmtelayers[ilayer-1]+ivm))
                for ivm in range(1,nvmteoverlaplayers[ilayer-1]+1) :
                    fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letteroverlap_triplet(iallstubmem)+str((iallstubmem-1)*nvmteoverlaplayers[ilayer-1]+ivm))
            if ilayer == 3:
                for ivm in range(1,nvmtelayers[ilayer-1]+1) :
                    fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letter_triplet(iallstubmem)+str((iallstubmem-1)*nvmtelayers[ilayer-1]+ivm))
                
        fp.write("\n\n")

#
# Do the VM routers for the TE in the overlap layers
#

#for ilayer in range(1,3) :
#    print "layer =",ilayer,"allstub memories",nallstubslayers[ilayer-1]
#    fp.write("\n")
#    fp.write("#\n")
#    fp.write("# VMRouters for the TEs in overlap layer "+str(ilayer)+" \n")
#    fp.write("#\n")
#    for iallstubmem in range(1,nallstubsoverlaplayers[ilayer-1]+1) :
#        allstubsmemname="L"+str(ilayer)+"PHI"+letter(iallstubmem)
#        for il in inputlinks :
#            if allstubsmemname in il :
#                fp.write(il+" ")
#        fp.write("> VMRTE_L"+str(ilayer)+"PHI"+letteroverlap(iallstubmem)+" > "#)
#        fp.write("AS_L"+str(ilayer)+"PHI"+letteroverlap(iallstubmem))
#        for ivm in range(1,nvmteoverlaplayers[ilayer-1]+1) :
#            fp.write(" VMSTE_L"+str(ilayer)+"PHI"+letteroverlap(iallstubmem)+str((iallstubmem-1)*nvmteoverlaplayers[ilayer-1]+ivm))
#        fp.write("\n\n")



#
# Do the VM routers for the TE in the disks
#

for idisk in range(1,6) :
    print "disk =",idisk,"allstub memories",nallstubsdisks[idisk-1]
    fp.write("\n")
    fp.write("#\n")
    fp.write("# VMRouters for the TEs in disk "+str(idisk)+" \n")
    fp.write("#\n")
    for iallstubmem in range(1,nallstubsdisks[idisk-1]+1) :
        allstubsmemname="D"+str(idisk)+"PHI"+letter(iallstubmem)
        for il in inputlinks :
            if allstubsmemname in il :
                fp.write(il+" ")
        fp.write("> VMR_D"+str(idisk)+"PHI"+letter(iallstubmem)+" > ")
        fp.write("AS_D"+str(idisk)+"PHI"+letter(iallstubmem))
        for ivm in range(1,nvmmedisks[idisk-1]+1) :
            fp.write(" VMSME_D"+str(idisk)+"PHI"+letter(iallstubmem)+str((iallstubmem-1)*nvmmedisks[idisk-1]+ivm))
        if idisk in range(1,5) :
            for ivm in range(1,nvmtedisks[idisk-1]+1) :
                fp.write(" VMSTE_D"+str(idisk)+"PHI"+letter(iallstubmem)+str((iallstubmem-1)*nvmtedisks[idisk-1]+ivm))
        if idisk in range(1,2) :
            for ivm in range(1,nvmteoverlapdisks[idisk-1]+1) :
                fp.write(" VMSTE_D"+str(idisk)+"PHI"+letteroverlap(iallstubmem)+str((iallstubmem-1)*nvmteoverlapdisks[idisk-1]+ivm))
            if displacedseeding :
                for ivm in range(1,nvmteoverlapdisks[idisk-1]+1) :
                    fp.write(" VMSTE_D"+str(idisk)+"PHI"+letteroverlap_triplet(iallstubmem)+str((iallstubmem-1)*nvmteoverlapdisks[idisk-1]+ivm))
        fp.write("\n\n")


#
# Do the VM routers for the TE in the overlap disks
#

#for idisk in range(1,2) :
#    print "disk =",idisk,"allstub memories overlap ",nallstubsoverlapdisks[idisk-1]
#    fp.write("\n")
#    fp.write("#\n")
#    fp.write("# VMRouters for the TEs in overlap disk "+str(idisk)+" \n")
#    fp.write("#\n")
#    for iallstubmem in range(1,nallstubsoverlapdisks[idisk-1]+1) :
#        allstubsmemname="D"+str(idisk)+"PHI"+letter(iallstubmem)
#        for il in inputlinks :
#            if allstubsmemname in il :
#                fp.write(il+" ")
#        fp.write("> VMRTE_D"+str(idisk)+"PHI"+letteroverlap(iallstubmem)+" > ")
#        fp.write("AS_D"+str(idisk)+"PHI"+letteroverlap(iallstubmem))
#        for ivm in range(1,nvmteoverlapdisks[idisk-1]+1) :
#            fp.write(" VMSTE_D"+str(idisk)+"PHI"+letteroverlap(iallstubmem)+str((iallstubmem-1)*nvmteoverlapdisks[idisk-1]+ivm))
#        fp.write("\n\n")



#
# Do the VM routers for the ME in the layers
#

#for ilayer in range(1,7) :
#    print "layer =",ilayer,"allproj memories",nallprojlayers[ilayer-1]
#    fp.write("\n")
#    fp.write("#\n")
#    fp.write("# VMRouters for the MEs in layer "+str(ilayer)+" \n")
#    fp.write("#\n")
#    for iallprojmem in range(1,nallprojlayers[ilayer-1]+1) :
#        allstubsmemname="L"+str(ilayer)+"PHI"+letter(iallprojmem)
#        for il in inputlinks :
#            if allstubsmemname in il :
#                fp.write(il+" ")
#        fp.write("> VMRME_L"+str(ilayer)+"PHI"+letter(iallprojmem)+" > ")
#        fp.write("\n\n")


#
# Do the VM routers for the ME in the disks
#

#for idisk in range(1,6) :
#    print "disk =",idisk,"allproj memories",nallprojdisks[idisk-1]
#    fp.write("\n")
#    fp.write("#\n")
#    fp.write("# VMRouters for the MEs in disk "+str(idisk)+" \n")
#    fp.write("#\n")
#    for iallprojmem in range(1,nallprojdisks[idisk-1]+1) :
#        allstubsmemname="D"+str(idisk)+"PHI"+letter(iallprojmem)
#        for il in inputlinks :
#            if allstubsmemname in il :
#                fp.write(il+" ")
#        fp.write("> VMRME_D"+str(idisk)+"PHI"+letter(iallprojmem)+" > ")
#        fp.write("AS_D"+str(idisk)+"PHI"+letter(iallprojmem))
#        for ivm in range(1,nvmmedisks[idisk-1]+1) :
#            fp.write(" VMSME_D"+str(idisk)+"PHI"+letter(iallprojmem)+str((iallprojmem-1)*nvmmedisks[idisk-1]+ivm))
#        fp.write("\n\n")



#
# Do the TED for the LL->L
#

SPD_list=[]
PairAMs = []

if displacedseeding :

    for lll in dispLLL :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for Displaced seeding layer"+str(lll[0])+"+layer"+str(lll[1])+"->layer"+str(lll[2])+"\n")
        fp.write("#\n")
        #print "layer = ",lll[0]
        for ivminner in range(1,nallstubslayers[lll[0]-1]*nvmtelayers[lll[0]-1]+1) :
            for ivmouter in range(1,nallstubslayers[lll[1]-1]*nvmtelayers[lll[1]-1]+1) :
                if validtedpair(lll[0],ivminner,ivmouter) :
                    amn = "L"+str(lll[0])+letter((ivminner-1)/nvmtelayers[lll[0]-1]+1)+"L"+str(lll[1])+letter((ivmouter-1)/nvmtelayers[lll[1]-1]+1)
                    if amn not in PairAMs :
                        PairAMs.append(amn)
                    fp.write("VMSTE_L"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner))
                    fp.write(" VMSTE_L"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > TED_L"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner))
                    fp.write("_L"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > ")
                    prange = phiproj5stlayer(lll[0],lll[2], ivminner, ivmouter)
                    #fp.write(str(prange[0])+" "+str(prange[1])+"\n")                         
                    for ivmproj in range(1, nallstubslayers[lll[2]-1]*nvmtelayers[lll[2]-1]+1) :
                        phiprojmin=phirange/nallstubslayers[lll[2]-1]/nvmtelayers[lll[2]-1]*(ivmproj-1)
                        phiprojmax=phirange/nallstubslayers[lll[2]-1]/nvmtelayers[lll[2]-1]*ivmproj
                        if prange[0]<phiprojmax and prange[1]>phiprojmin :
                            spd_name="SPD_L"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner)+"_L"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter)+"_L"+str(lll[2])+"PHI"+letter((ivmproj-1)/nvmtelayers[lll[2]-1]+1)+str(ivmproj)
                            fp.write(spd_name+" ")
                            SPD_list.append(spd_name)
                    fp.write("\n\n")

    #
    # Do the TED for the LL->D
    #

    for lll in dispLLD :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for Displaced seeding layer"+str(lll[0])+"+layer"+str(lll[1])+"->disk"+str(lll[2])+"\n")
        fp.write("#\n")
        #print "layer = ",lll[0]
        for ivminner in range(1,nallstubslayers[lll[0]-1]*nvmtelayers[lll[0]-1]+1) :
            for ivmouter in range(1,nallstubslayers[lll[1]-1]*nvmtelayers[lll[1]-1]+1) :
                if validtedpair(lll[0],ivminner,ivmouter) :
                    amn = "L"+str(lll[0])+letter_triplet((ivminner-1)/nvmtelayers[lll[0]-1]+1)+"L"+str(lll[1])+letter_triplet((ivmouter-1)/nvmtelayers[lll[1]-1]+1)
                    if amn not in PairAMs :
                        PairAMs.append(amn)
                    fp.write("VMSTE_L"+str(lll[0])+"PHI"+letter_triplet((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner))
                    fp.write(" VMSTE_L"+str(lll[1])+"PHI"+letter_triplet((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > TED_L"+str(lll[0])+"PHI"+letter_triplet((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner))
                    fp.write("_L"+str(lll[1])+"PHI"+letter_triplet((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > ")
                    prange = phiproj5stlayer_to_disk(lll[0],lll[2], ivminner, ivmouter)
                    #fp.write(str(prange[0])+" "+str(prange[1])+"\n")                         
                    for ivmproj in range(1, nallstubsdisks[lll[2]-1]*nvmteoverlapdisks[lll[2]-1]+1) :
                        phiprojmin=phirange/nallstubsdisks[lll[2]-1]/nvmteoverlapdisks[lll[2]-1]*(ivmproj-1)
                        phiprojmax=phirange/nallstubsdisks[lll[2]-1]/nvmteoverlapdisks[lll[2]-1]*ivmproj
                        if prange[0]<phiprojmax and prange[1]>phiprojmin :
                            spd_name="SPD_L"+str(lll[0])+"PHI"+letter_triplet((ivminner-1)/nvmtelayers[lll[0]-1]+1)+str(ivminner)+"_L"+str(lll[1])+"PHI"+letter_triplet((ivmouter-1)/nvmtelayers[lll[1]-1]+1)+str(ivmouter)+"_D"+str(lll[2])+"PHI"+letteroverlap_triplet((ivmproj-1)/nvmteoverlapdisks[lll[2]-1]+1)+str(ivmproj)
                            fp.write(spd_name+" ")
                            SPD_list.append(spd_name)
                    fp.write("\n\n")

    #
    # Do the TED for the DD->L
    #

    for lll in dispDDL :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for Displaced seeding disk"+str(lll[0])+"+disk"+str(lll[1])+"->layer"+str(lll[2])+"\n")
        fp.write("#\n")
        #print "disk = ",lll[0]
        for ivminner in range(1,nallstubsdisks[lll[0]-1]*nvmtedisks[lll[0]-1]+1) :
            for ivmouter in range(1,nallstubsdisks[lll[1]-1]*nvmtedisks[lll[1]-1]+1) :
                if validtedpairdisk(lll[0],ivminner,ivmouter) :
                    amn = "D"+str(lll[0])+letter((ivminner-1)/nvmtedisks[lll[0]-1]+1)+"D"+str(lll[1])+letter((ivmouter-1)/nvmtedisks[lll[1]-1]+1)
                    if amn not in PairAMs :
                        PairAMs.append(amn)
                    fp.write("VMSTE_D"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtedisks[lll[0]-1]+1)+str(ivminner))
                    fp.write(" VMSTE_D"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtedisks[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > TED_D"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtedisks[lll[0]-1]+1)+str(ivminner))
                    fp.write("_D"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtedisks[lll[1]-1]+1)+str(ivmouter))
                    fp.write(" > ")
                    prange = phiproj5stdisk_to_layer(lll[0],lll[2], ivminner, ivmouter)
                    #fp.write(str(prange[0])+" "+str(prange[1])+"\n")                         
                    for ivmproj in range(1, nallstubslayers[lll[2]-1]*nvmteoverlaplayers[lll[2]-1]+1) :
                        phiprojmin=phirange/nallstubslayers[lll[2]-1]/nvmteoverlaplayers[lll[2]-1]*(ivmproj-1)
                        phiprojmax=phirange/nallstubslayers[lll[2]-1]/nvmteoverlaplayers[lll[2]-1]*ivmproj
                        if prange[0]<phiprojmax and prange[1]>phiprojmin :
                            spd_name="SPD_D"+str(lll[0])+"PHI"+letter((ivminner-1)/nvmtedisks[lll[0]-1]+1)+str(ivminner)+"_D"+str(lll[1])+"PHI"+letter((ivmouter-1)/nvmtedisks[lll[1]-1]+1)+str(ivmouter)+"_L"+str(lll[2])+"PHI"+letteroverlap_triplet((ivmproj-1)/nvmteoverlaplayers[lll[2]-1]+1)+str(ivmproj)
                            fp.write(spd_name+" ")
                            SPD_list.append(spd_name)
                    fp.write("\n\n")

TPROJ_list=[]
TPAR_list=[]

SP_list=[]


if combinedTP :

    #
    # Do the TP for the layers
    #
                
    for ilayer in (1,2,3,5) :
        if ilayer==2 and not extraseeding :
            continue
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Processors for seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")

        if ilayer!=2 : 
            for ivminner in range(1,nallstubslayers[ilayer-1]*nvmtelayers[ilayer-1]+1) :
                for ivmouter in range(1,nallstubslayers[ilayer]*nvmtelayers[ilayer]+1) :
                    if validtepair(ilayer,ivminner,ivmouter) :
                        sp_name="SP_L"+str(ilayer)+"PHI"+letter((ivminner-1)/nvmtelayers[ilayer-1]+1)+str(ivminner)+"_L"+str(ilayer+1)+"PHI"+letter((ivmouter-1)/nvmtelayers[ilayer]+1)+str(ivmouter)
                        SP_list.append(sp_name)
        else :
            for ivminner in range(1,nallstubslayers[ilayer-1]*nvmteextralayers[ilayer-1]+1) :
                for ivmouter in range(1,nallstubslayers[ilayer]*nvmteextralayers[ilayer]+1) :
                    if validtepairextra(ilayer,ivminner,ivmouter) :
                        sp_name="SP_L"+str(ilayer)+"PHI"+letterextra((ivminner-1)/nvmteextralayers[ilayer-1]+1)+str(ivminner)+"_L"+str(ilayer+1)+"PHI"+letterextra((ivmouter-1)/nvmteextralayers[ilayer]+1)+str(ivmouter)
                        SP_list.append(sp_name)

                    
        sp_layer=[]
        for sp_name in SP_list :
            if "_L"+str(ilayer) in sp_name and "_L"+str(ilayer+1) in sp_name :
                #print ilayer,sp_name
                sp_layer.append(sp_name)

        tcs=12
        if ilayer==2 :
            tcs=2
        if ilayer==3 :
            tcs=8
        if ilayer==5 :
            tcs=4

        sp_per_tc=split(sp_layer,tcs)
    
        tp_count=0
        for sps in  sp_per_tc :
            print len(sps), sps
            innervms=[]
            outervms=[]
            for sp_name in sps :
                innervm=sp_name.split("_")[1]
                outervm=sp_name.split("_")[2]
                fp.write("VMSTE_"+innervm+" VMSTE_"+outervm+" ")
            tp_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    
            tpar_name="TPAR_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tp_count)
            fp.write(" > TP_L"+str(ilayer)+"L"+str(ilayer+1)+letter(tp_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            for projlayer in range(1,7) :
                if ilayer==2 and projlayer==6:
                    continue #seeding in L2L3 assumed not to project to L6
                if projlayer!=ilayer and projlayer!=ilayer+1 :
                    projrange=phiprojrange(ilayer,projlayer,sps)
                    for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                        phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tp_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projdisks=[]
            if ilayer<5 :
                projdisks.append(1)
                projdisks.append(2)
            if ilayer==2 :
                projdisks.append(3)
                projdisks.append(4)
            if ilayer==1 :
                projdisks.append(3)
                projdisks.append(4)
                projdisks.append(5)
            for projdisk in projdisks :
                projrange=phiprojrangelayertodisk(ilayer,projdisk,sps)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1) 
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tp_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")
            
    
    #
    # Do the TP for the disks
    #
                
    for idisk in (1,3) :
        fp.write("#\n")
        fp.write("# Tracklet Processors for seeding disk "+str(idisk)+" \n")
        fp.write("#\n")

        for ivminner in range(1,nallstubsdisks[idisk-1]*nvmtedisks[idisk-1]+1) :
            for ivmouter in range(1,nallstubsdisks[idisk]*nvmtedisks[idisk]+1) :
                if validtepairdisk(idisk,ivminner,ivmouter) :
                    sp_name="SP_D"+str(idisk)+"PHI"+letter((ivminner-1)/nvmtedisks[idisk-1]+1)+str(ivminner)+"_D"+str(idisk+1)+"PHI"+letter((ivmouter-1)/nvmtedisks[idisk]+1)+str(ivmouter)
                    SP_list.append(sp_name)

        
        sp_disk=[]
        for sp_name in SP_list :
            if "_D"+str(idisk) in sp_name and "_D"+str(idisk+1) in sp_name :
                #print idisk,sp_name
                sp_disk.append(sp_name)


        tcs=6
        if idisk==3 :
            tcs=2

        sp_per_tc=split(sp_disk,tcs)
    
        tc_count=0
        for sps in  sp_per_tc :
            #print len(sps), sps
            for sp_name in sps :
                innervm=sp_name.split("_")[1]
                outervm=sp_name.split("_")[2]
                fp.write("VMSTE_"+innervm+" VMSTE_"+outervm+" ")
            tc_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    
            tpar_name="TPAR_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)
            fp.write(" > TP_D"+str(idisk)+"D"+str(idisk+1)+letter(tc_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            for projdisk in range(1,6) :
                if projdisk!=idisk and projdisk!=idisk+1 :
                    #print "idisk, projdisk, sps:",idisk, projdisk,sps
                    projrange=phiprojrangedisk(idisk,projdisk,sps)
                    for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                        print "looking for projection to disk iallproj",projdisk,iallproj
                        phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projlayers=[]
            projlayers.append(1)
            if idisk==1 :
                projlayers.append(2)
            for projlayer in projlayers :
                projrange=phiprojrangedisktolayer(idisk,projlayer,sps)
                for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                    phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                    if projrange[0]<=phiprojmax and projrange[1]>=phiprojmin :
                        proj_name="TPROJ_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")

    #
    # Do the TP for the overlaps
    #


    for ilayer in (1,2) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for overlap seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")
        #print "layer = ",ilayer
        for ivminner in range(1,nallstubsoverlaplayers[ilayer-1]*nvmteoverlaplayers[ilayer-1]+1) :
            for ivmouter in range(1,nallstubsoverlapdisks[0]*nvmteoverlapdisks[0]+1) :
                if validtepairoverlap(ilayer,ivminner,ivmouter) :
                    sp_name="SP_L"+str(ilayer)+"PHI"+letteroverlap((ivminner-1)/nvmteoverlaplayers[ilayer-1]+1)+str(ivminner)+"_D"+str(1)+"PHI"+letteroverlap((ivmouter-1)/nvmteoverlapdisks[0]+1)+str(ivmouter)
                    SP_list.append(sp_name)



    for ilayer in (1,2) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Calculators for seeding in overlap layer "+str(ilayer)+" \n")
        fp.write("#\n")

        sp_layer=[]
        for sp_name in SP_list :
            if "_L"+str(ilayer) in sp_name and "_D1" in sp_name :
                #print ilayer,sp_name
                sp_layer.append(sp_name)

        tcs=6
        if ilayer==2 :
            tcs=2

        sp_per_tc=split(sp_layer,tcs)
    
        tc_count=0
        for sps in  sp_per_tc :
            #print len(sps), sps
            for sp_name in sps :
                innervm=sp_name.split("_")[1]
                outervm=sp_name.split("_")[2]
                fp.write("VMSTE_"+innervm+" VMSTE_"+outervm+" ")
            tc_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    

            tpar_name="TPAR_L"+str(ilayer)+"D1"+xx+letter(tc_count)
            fp.write(" > TP_L"+str(ilayer)+"D1"+letter(tc_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            if ilayer==2 :
                for projlayer in range(1,2) :
                    projrange=phiprojrange(ilayer,projlayer,sps)
                    #print ilayer, iallstubmeminner,projlayer,projrange
                    for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                        phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_L"+str(ilayer)+"D"+str(1)+xx+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projdisks=[2,3,4,5]
            for projdisk in projdisks :
                projrange=phiprojrangeoverlaplayertodisk(ilayer,projdisk,sps)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_L"+str(ilayer)+"D"+str(1)+xx+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")
                    

            
else :    
    
    #
    # Do the TE for the layers
    #

    for ilayer in (1,3,5) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")
        #print "layer = ",ilayer
        for ivminner in range(1,nallstubslayers[ilayer-1]*nvmtelayers[ilayer-1]+1) :
            for ivmouter in range(1,nallstubslayers[ilayer]*nvmtelayers[ilayer]+1) :
                if validtepair(ilayer,ivminner,ivmouter) :
                    fp.write("VMSTE_L"+str(ilayer)+"PHI"+letter((ivminner-1)/nvmtelayers[ilayer-1]+1)+str(ivminner))
                    fp.write(" VMSTE_L"+str(ilayer+1)+"PHI"+letter((ivmouter-1)/nvmtelayers[ilayer]+1)+str(ivmouter))
                    fp.write(" > TE_L"+str(ilayer)+"PHI"+letter((ivminner-1)/nvmtelayers[ilayer-1]+1)+str(ivminner))
                    fp.write("_L"+str(ilayer+1)+"PHI"+letter((ivmouter-1)/nvmtelayers[ilayer]+1)+str(ivmouter))
                    sp_name="SP_L"+str(ilayer)+"PHI"+letter((ivminner-1)/nvmtelayers[ilayer-1]+1)+str(ivminner)+"_L"+str(ilayer+1)+"PHI"+letter((ivmouter-1)/nvmtelayers[ilayer]+1)+str(ivmouter)
                    fp.write(" > "+sp_name)
                    fp.write("\n\n")
                    SP_list.append(sp_name)


    for ilayer in [2] :
        if not extraseeding :
            continue
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for extra seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")
        print "layer = ",ilayer
        for ivminner in range(1,nallstubslayers[ilayer-1]*nvmteextralayers[ilayer-1]+1) :
            for ivmouter in range(1,nallstubslayers[ilayer]*nvmteextralayers[ilayer]+1) :
                if validtepairextra(ilayer,ivminner,ivmouter) :
                    fp.write("VMSTE_L"+str(ilayer)+"PHI"+letterextra((ivminner-1)/nvmteextralayers[ilayer-1]+1)+str(ivminner))
                    fp.write(" VMSTE_L"+str(ilayer+1)+"PHI"+letterextra((ivmouter-1)/nvmteextralayers[ilayer]+1)+str(ivmouter))
                    fp.write(" > TE_L"+str(ilayer)+"PHI"+letterextra((ivminner-1)/nvmteextralayers[ilayer-1]+1)+str(ivminner))
                    fp.write("_L"+str(ilayer+1)+"PHI"+letterextra((ivmouter-1)/nvmteextralayers[ilayer]+1)+str(ivmouter))
                    sp_name="SP_L"+str(ilayer)+"PHI"+letterextra((ivminner-1)/nvmteextralayers[ilayer-1]+1)+str(ivminner)+"_L"+str(ilayer+1)+"PHI"+letterextra((ivmouter-1)/nvmteextralayers[ilayer]+1)+str(ivmouter)
                    fp.write(" > "+sp_name)
                    fp.write("\n\n")
                    SP_list.append(sp_name)




    #
    # Do the TE for the disks
    #

    for idisk in (1,3) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for seeding disk "+str(idisk)+" \n")
        fp.write("#\n")
        #print "disk = ",idisk
        for ivminner in range(1,nallstubsdisks[idisk-1]*nvmtedisks[idisk-1]+1) :
            for ivmouter in range(1,nallstubsdisks[idisk]*nvmtedisks[idisk]+1) :
                if validtepairdisk(idisk,ivminner,ivmouter) :
                    fp.write("VMSTE_D"+str(idisk)+"PHI"+letter((ivminner-1)/nvmtedisks[idisk-1]+1)+str(ivminner))
                    fp.write(" VMSTE_D"+str(idisk+1)+"PHI"+letter((ivmouter-1)/nvmtedisks[idisk]+1)+str(ivmouter))
                    fp.write(" > TE_D"+str(idisk)+"PHI"+letter((ivminner-1)/nvmtedisks[idisk-1]+1)+str(ivminner))
                    fp.write("_D"+str(idisk+1)+"PHI"+letter((ivmouter-1)/nvmtedisks[idisk]+1)+str(ivmouter))
                    sp_name="SP_D"+str(idisk)+"PHI"+letter((ivminner-1)/nvmtedisks[idisk-1]+1)+str(ivminner)+"_D"+str(idisk+1)+"PHI"+letter((ivmouter-1)/nvmtedisks[idisk]+1)+str(ivmouter)
                    fp.write(" > "+sp_name)
                    fp.write("\n\n")
                    SP_list.append(sp_name)



    #
    # Do the TE for the overlap
    #

    for ilayer in (1,2) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Engines for overlap seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")
        #print "layer = ",ilayer
        for ivminner in range(1,nallstubsoverlaplayers[ilayer-1]*nvmteoverlaplayers[ilayer-1]+1) :
            for ivmouter in range(1,nallstubsoverlapdisks[0]*nvmteoverlapdisks[0]+1) :
                if validtepairoverlap(ilayer,ivminner,ivmouter) :
                    fp.write("VMSTE_L"+str(ilayer)+"PHI"+letteroverlap((ivminner-1)/nvmteoverlaplayers[ilayer-1]+1)+str(ivminner))
                    fp.write(" VMSTE_D"+str(1)+"PHI"+letteroverlap((ivmouter-1)/nvmteoverlapdisks[0]+1)+str(ivmouter))
                    fp.write(" > TE_L"+str(ilayer)+"PHI"+letteroverlap((ivminner-1)/nvmteoverlaplayers[ilayer-1]+1)+str(ivminner))
                    fp.write("_D"+str(1)+"PHI"+letteroverlap((ivmouter-1)/nvmteoverlapdisks[0]+1)+str(ivmouter))
                    sp_name="SP_L"+str(ilayer)+"PHI"+letteroverlap((ivminner-1)/nvmteoverlaplayers[ilayer-1]+1)+str(ivminner)+"_D"+str(1)+"PHI"+letteroverlap((ivmouter-1)/nvmteoverlapdisks[0]+1)+str(ivmouter)
                    fp.write(" > "+sp_name)
                    fp.write("\n\n")
                    SP_list.append(sp_name)


                
                
    #
    # Do the TC for the layers
    #

    for ilayer in (1,2,3,5) :
        if ilayer==2 and not extraseeding :
            continue
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Calculators for seeding layer "+str(ilayer)+" \n")
        fp.write("#\n")

        sp_layer=[]
        for sp_name in SP_list :
            if "_L"+str(ilayer) in sp_name and "_L"+str(ilayer+1) in sp_name :
                #print ilayer,sp_name
                sp_layer.append(sp_name)

        tcs=12
        if ilayer==2 :
            tcs=2
        if ilayer==3 :
            tcs=8
        if ilayer==5 :
            tcs=4

        sp_per_tc=split(sp_layer,tcs)
    
        tc_count=0
        for sps in  sp_per_tc :
            print len(sps), sps
            for sp_name in sps :
                fp.write(sp_name+" ")
            tc_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    
            tpar_name="TPAR_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tc_count)
            fp.write(" > TC_L"+str(ilayer)+"L"+str(ilayer+1)+letter(tc_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            for projlayer in range(1,7) :
                if ilayer==2 and projlayer==6:
                    continue #seeding in L2L3 assumed not to project to L6
                if projlayer!=ilayer and projlayer!=ilayer+1 :
                    projrange=phiprojrange(ilayer,projlayer,sps)
                    for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                        phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projdisks=[]
            if ilayer<5 :
                projdisks.append(1)
                projdisks.append(2)
            if ilayer==2 :
                projdisks.append(3)
                projdisks.append(4)
            if ilayer==1 :
                projdisks.append(3)
                projdisks.append(4)
                projdisks.append(5)
            for projdisk in projdisks :
                projrange=phiprojrangelayertodisk(ilayer,projdisk,sps)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_L"+str(ilayer)+"L"+str(ilayer+1)+xx+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")
            

    #
    # Do the TC for the disks
    #

                
    for idisk in (1,3) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Calculators for seeding diks "+str(idisk)+" \n")
        fp.write("#\n")

        sp_disk=[]
        for sp_name in SP_list :
            if "_D"+str(idisk) in sp_name and "_D"+str(idisk+1) in sp_name :
                #print idisk,sp_name
                sp_disk.append(sp_name)

        tcs=6
        if idisk==3 :
            tcs=2

        sp_per_tc=split(sp_disk,tcs)
    
        tc_count=0
        for sps in  sp_per_tc :
            #print len(sps), sps
            for sp_name in sps :
                fp.write(sp_name+" ")
            tc_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    
            tpar_name="TPAR_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)
            fp.write(" > TC_D"+str(idisk)+"D"+str(idisk+1)+letter(tc_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            for projdisk in range(1,6) :
                if projdisk!=idisk and projdisk!=idisk+1 :
                    projrange=phiprojrangedisk(idisk,projdisk,sps)
                    for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                        print "looking for projection to disk iallproj",projdisk,iallproj
                        phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projlayers=[]
            projlayers.append(1)
            if idisk==1 :
                projlayers.append(2)
            for projlayer in projlayers :
                projrange=phiprojrangedisktolayer(idisk,projlayer,sps)
                for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                    phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                    if projrange[0]<=phiprojmax and projrange[1]>=phiprojmin :
                        proj_name="TPROJ_D"+str(idisk)+"D"+str(idisk+1)+xx+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")


    #
    # Do the TC for the overlap
    #
                
    for ilayer in (1,2) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Tracklet Calculators for seeding in overlap layer "+str(ilayer)+" \n")
        fp.write("#\n")

        sp_layer=[]
        for sp_name in SP_list :
            if "_L"+str(ilayer) in sp_name and "_D1" in sp_name :
                #print ilayer,sp_name
                sp_layer.append(sp_name)

        tcs=6
        if ilayer==2 :
            tcs=2

        sp_per_tc=split(sp_layer,tcs)
    
        tc_count=0
        for sps in  sp_per_tc :
            #print len(sps), sps
            for sp_name in sps :
                fp.write(sp_name+" ")
            tc_count+=1
            as_names = asmems(sps)
            for asn in as_names:
                fp.write(asn+" ")    

            tpar_name="TPAR_L"+str(ilayer)+"D1"+xx+letter(tc_count)
            fp.write(" > TC_L"+str(ilayer)+"D1"+letter(tc_count)+" > "+tpar_name)
            TPAR_list.append(tpar_name)
            if ilayer==2 :
                for projlayer in range(1,2) :
                    projrange=phiprojrange(ilayer,projlayer,sps)
                    #print ilayer, iallstubmeminner,projlayer,projrange
                    for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                        phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_L"+str(ilayer)+"D"+str(1)+xx+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projdisks=[2,3,4,5]
            for projdisk in projdisks :
                projrange=phiprojrangeoverlaplayertodisk(ilayer,projdisk,sps)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_L"+str(ilayer)+"D"+str(1)+xx+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")


if displacedseeding : 

    #
    # triplet finding
    #

    ST_list = []

    print "+++++++++++++"
    print PairAMs
    print "+++++++++++++"

    spd_occ = readSPoccupancy()
    spd_occ_max = 5

    for pn in PairAMs :
        #print "*** debug Triplet Engines for ",pn
        fp.write("#\n# Triplet Engines for "+pn+"\n#\n")
        if   pn[0:2] == "L5":
            spd_occ_max = 1.0
        elif pn[0:2] == "L3":
            spd_occ_max = 1.5
        elif pn[0:2] == "L2":
            spd_occ_max = 0.2
        elif pn[0:2] == "D1":
            spd_occ_max = 0.2

        spall = []
        spn3 = []
        for spn in SPD_list :
            if spn not in spd_occ:
                print "****** did not find ",spn," in the occupancy list!! Skipping..."
            else:
                #  third VMs
                spnparts = spn.split("_")
                if pn[0:2]==spnparts[1][0:2] and pn[2]==spnparts[1][5] and pn[3:5]==spnparts[2][0:2] and pn[5]==spnparts[2][5] :
                    spall.append(spn)
                    if spnparts[3] not in spn3:
                        spn3.append(spnparts[3])

        #spall are all sp from pn
        #spn3 now is a list of all third VMs
        print "&&&&&&&&&&&&&&&&&&&&&\n"
        print pn
        print len(spall)
        print spall
        print spn3
        print "&&&&&&&&&&&&&&&&&&&&&&\n"

        toprint_vmte = []
        toprint_aste = []
        toprint_sps  = []
        tre_counter = 0
        occ = 0
        first_counter = 1
        for vm3 in spn3:
            print "^^^^",vm3
            for spn in spall :
                if vm3 in spn :
                    #now looping over SPDs that are from pn and pointing to vm3
                    occ = occ + spd_occ[spn]
                    print "^^^^",spn,occ
                    if occ > spd_occ_max and first_counter == 0:
                        #reach the limit, print
                        print "^^^^^",len(toprint_vmte),len(toprint_sps)
                        tre_counter = tre_counter + 1
                        first_counter = 1
                        for i in toprint_vmte :
                            fp.write("VMSTE_"+i+" ")
                        for i in toprint_sps :
                            fp.write(i+" ")
                        STn = "ST_"+pn+"_"+vm3[0:2]
                        for i in toprint_aste :
                            STn = STn + i                    
                        STn = STn+"_"+str(tre_counter)
                        ST_list.append(STn)
                        fp.write("> TRE_"+pn+"_"+str(tre_counter)+" > "+STn+"\n")
                        toprint_vmte = []
                        toprint_aste = []
                        toprint_sps = []
                        occ = spd_occ[spn]
                    if vm3 not in toprint_vmte:
                        toprint_vmte.append(vm3)
                        if vm3[5] not in toprint_aste:
                            toprint_aste.append(vm3[5])
                    toprint_sps.append(spn)
                    first_counter = 0
        if len(toprint_sps)>0 :
            #print the remainder
            print "^^^^^",len(toprint_vmte),len(toprint_sps)
            tre_counter = tre_counter + 1
            for i in toprint_vmte :
                fp.write("VMSTE_"+i+" ")
            for i in toprint_sps :
                fp.write(i+" ")
            STn = "ST_"+pn+"_"+vm3[0:2]
            for i in toprint_vmte :
                STn = STn + i[5]                    
            STn = STn+"_"+str(tre_counter)
            ST_list.append(STn)
            fp.write("> TRE_"+pn+"_"+str(tre_counter)+" > "+STn+"\n")
            toprint_vmte = []
            toprint_aste = []
            toprint_sps = []
            occ = 0



    print "*********************************"    
    print ST_list
    print "*********************************"    

    for lll in dispLLL :
        fp.write("\n")
        fp.write("#\n")
        l1 = lll[0];
        l2 = lll[1];
        l3 = lll[2];
        fp.write("# Tracklet Calculators for LLL triplets "+str(l1)+str(l2)+str(l3)+" \n")
        fp.write("#\n")
        tcn = "L"+str(l1)+"L"+str(l2)+"L"+str(l3)

        st_lll = []
        for st_name in ST_list :
            if "L"+str(l1) in st_name and "L"+str(l2) in st_name and "L"+str(l3) in st_name :
                st_lll.append(st_name)

        tcs = 10
        st_per_tc = split(st_lll,tcs)

        tc_count = 0
        for sts in st_per_tc :
            print len(sts), sts

            for st_name in sts:
                fp.write(st_name+" ")

            asmem = asmems3(sts)
            for asn in asmem:
                fp.write(asn+" ")    

            tc_count+=1
            fp.write("> TCD_"+tcn+letter(tc_count)+" > TPAR_"+tcn+letter(tc_count)+" ")
            TPAR_list.append("TPAR_"+tcn+letter(tc_count))

            for projlayer in range(1,7) :
                if projlayer!=l1 and projlayer!=l2 and projlayer!=l3 :
                    projrange=phiproj5projrange(sts,rlayers[projlayer-1])
                    for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                        phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                        phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                        if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                            proj_name="TPROJ_"+tcn+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                            if proj_name not in unusedproj :
                                fp.write(" "+proj_name)
                                TPROJ_list.append(proj_name)
            projdisks=[]
            if l3 == 2:
                projdisks = [1,2,3]
            for projdisk in projdisks :
                projrange=phiproj5projrange(sts,rmaxdisk)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_"+tcn+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")


    for lll in dispLLD :
        fp.write("\n")
        fp.write("#\n")
        l1 = lll[0];
        l2 = lll[1];
        l3 = lll[2];
        fp.write("# Tracklet Calculators for LLD triplets "+str(l1)+str(l2)+str(l3)+" \n")
        fp.write("#\n")
        tcn = "L"+str(l1)+"L"+str(l2)+"D"+str(l3)

        st_lll = []
        for st_name in ST_list :
            if "L"+str(l1) in st_name and "L"+str(l2) in st_name and "D"+str(l3) in st_name :
                st_lll.append(st_name)

        tcs = 10
        st_per_tc = split(st_lll,tcs)

        tc_count = 0
        for sts in st_per_tc :
            print len(sts), sts

            for st_name in sts:
                fp.write(st_name+" ")

            asmem = asmems3(sts)
            for asn in asmem:
                fp.write(asn+" ")    

            tc_count+=1
            fp.write("> TCD_"+tcn+letter(tc_count)+" > TPAR_"+tcn+letter(tc_count)+" ")
            TPAR_list.append("TPAR_"+tcn+letter(tc_count))

            projlayers = [1,4]
            for projlayer in projlayers :
                projrange=phiproj5projrange(sts,rlayers[projlayer-1])
                for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                    phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_"+tcn+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            projdisks=[2,3,4]
            for projdisk in projdisks :
                projrange=phiproj5projrange(sts,rmaxdisk)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_"+tcn+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")


    for lll in dispDDL :
        fp.write("\n")
        fp.write("#\n")
        l1 = lll[0];
        l2 = lll[1];
        l3 = lll[2];
        fp.write("# Tracklet Calculators for DDL triplets "+str(l1)+str(l2)+str(l3)+" \n")
        fp.write("#\n")
        tcn = "D"+str(l1)+"D"+str(l2)+"L"+str(l3)

        st_lll = []
        for st_name in ST_list :
            if "D"+str(l1) in st_name and "D"+str(l2) in st_name and "L"+str(l3) in st_name :
                st_lll.append(st_name)

        tcs = 10
        st_per_tc = split(st_lll,tcs)

        tc_count = 0
        for sts in st_per_tc :
            print len(sts), sts

            for st_name in sts:
                fp.write(st_name+" ")

            asmem = asmems3(sts)
            for asn in asmem:
                fp.write(asn+" ")    

            tc_count+=1
            fp.write("> TCD_"+tcn+letter(tc_count)+" > TPAR_"+tcn+letter(tc_count)+" ")
            TPAR_list.append("TPAR_"+tcn+letter(tc_count))

            projlayers = [1,3]
            for projlayer in projlayers :
                projrange=phiproj5projrange(sts,rlayers[projlayer-1])
                for iallproj in range(1,nallprojlayers[projlayer-1]+1) :
                    phiprojmin=phirange/nallprojlayers[projlayer-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojlayers[projlayer-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_"+tcn+letter(tc_count)+"_L"+str(projlayer)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            projdisks=[3,4,5]
            for projdisk in projdisks :
                projrange=phiproj5projrange(sts,rmaxdisk)
                for iallproj in range(1,nallprojdisks[projdisk-1]+1) :
                    phiprojmin=phirange/nallprojdisks[projdisk-1]*(iallproj-1)
                    phiprojmax=phirange/nallprojdisks[projdisk-1]*iallproj
                    if projrange[0]<phiprojmax and projrange[1]>phiprojmin :
                        proj_name="TPROJ_"+tcn+letter(tc_count)+"_D"+str(projdisk)+"PHI"+letter(iallproj)
                        if proj_name not in unusedproj :
                            fp.write(" "+proj_name)
                            TPROJ_list.append(proj_name)
            fp.write("\n\n")



FM_list=[]
CM_list=[]


if combinedMP :

    for ilayer in range(1,7) :
        print "layer =",ilayer,"allstub memories",nallprojlayers[ilayer-1]
        fp.write("\n")
        fp.write("#\n")
        fp.write("# PROJRouters+MatchEngines+MatchCalculator in layer "+str(ilayer)+" \n")
        fp.write("#\n")
        for iallprojmem in range(1,nallprojlayers[ilayer-1]+1) :
            projmemname="L"+str(ilayer)+"PHI"+letter(iallprojmem)
            for proj_name in TPROJ_list :
                if projmemname in proj_name :
                    fp.write(proj_name+" ")
            fp.write("AS_L"+str(ilayer)+"PHI"+letter(iallprojmem))
            for ivm in range(1,nallprojlayers[ilayer-1]*nvmmelayers[ilayer-1]+1) :
                phiregion=1+(ivm-1)/nvmmelayers[ilayer-1]
                if phiregion!=iallprojmem :
                    continue
                fp.write(" VMSME_L"+str(ilayer)+"PHI"+letter(phiregion)+str(ivm))

            fp.write(" > MP_L"+str(ilayer)+"PHI"+letter(iallprojmem)+" > ")
            fm_name="FM_L1L2_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer!=1 and ilayer!=2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2L3_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer!=2 and ilayer!=3 and ilayer!=6:  #do not allow L2L3 projections to L6
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L3L4_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer!=3 and ilayer!=4 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L5L6_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer!=5 and ilayer!=6 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D1D2_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer==1 or ilayer==2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D3D4_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer==1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2D1_L"+str(ilayer)+"PHI"+letter(iallprojmem)
            if ilayer==1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fp.write("\n\n")

    for idisk in range(1,6) :
        print "disk =",idisk,"allstub memories",nallprojdisks[idisk-1]
        fp.write("\n")
        fp.write("#\n")
        fp.write("# PROJRouters+MatchEngine+MatchCalculator in disk "+str(idisk)+" \n")
        fp.write("#\n")
        for iallprojmem in range(1,nallprojdisks[idisk-1]+1) :
            projmemname="D"+str(idisk)+"PHI"+letter(iallprojmem)
            for proj_name in TPROJ_list :
                if projmemname in proj_name :
                    fp.write(proj_name+" ")
            fp.write("AS_D"+str(idisk)+"PHI"+letter(iallprojmem))
            for ivm in range(1,nallprojdisks[idisk-1]*nvmmedisks[idisk-1]+1) :
                phiregion=1+(ivm-1)/nvmmedisks[idisk-1]
                if phiregion!=iallprojmem :
                    continue
                fp.write(" VMSME_D"+str(idisk)+"PHI"+letter(phiregion)+str(ivm))
            fp.write(" > MP_D"+str(idisk)+"PHI"+letter(iallprojmem)+" > ")
            fm_name="FM_D1D2_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=1 and idisk!=2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D3D4_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=3 and idisk!=4 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L1L2_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=5 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2L3_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=5 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L3L4_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk==1 or idisk==2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L1D1_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2D1_D"+str(idisk)+"PHI"+letter(iallprojmem)
            if idisk!=1 and idisk!=5 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)

            fp.write("\n\n")

            
    
else :
                
    #
    # Do the PROJRouters for the layers
    #

    for ilayer in range(1,7) :
        print "layer =",ilayer,"allstub memories",nallprojlayers[ilayer-1]
        fp.write("\n")
        fp.write("#\n")
        fp.write("# PROJRouters for the MEs in layer "+str(ilayer)+" \n")
        fp.write("#\n")
        for iallprojmem in range(1,nallprojlayers[ilayer-1]+1) :
            projmemname="L"+str(ilayer)+"PHI"+letter(iallprojmem)
            for proj_name in TPROJ_list :
                if projmemname in proj_name :
                    fp.write(proj_name+" ")
            fp.write("> PR_L"+str(ilayer)+"PHI"+letter(iallprojmem)+" > AP_L"+str(ilayer)+"PHI"+letter(iallprojmem))
            for ivm in range(1,nvmmelayers[ilayer-1]+1) :
                fp.write(" VMPROJ_L"+str(ilayer)+"PHI"+letter(iallprojmem)+str((iallprojmem-1)*nvmmelayers[ilayer-1]+ivm))
            fp.write("\n\n")


    #
    # Do the PROJRouters for the disks
    #

    for idisk in range(1,6) :
        print "disk =",idisk,"allstub memories",nallprojdisks[idisk-1]
        fp.write("\n")
        fp.write("#\n")
        fp.write("# PROJRouters for the MEs in disk "+str(idisk)+" \n")
        fp.write("#\n")
        for iallprojmem in range(1,nallprojdisks[idisk-1]+1) :
            projmemname="D"+str(idisk)+"PHI"+letter(iallprojmem)
            for proj_name in TPROJ_list :
                if projmemname in proj_name :
                    fp.write(proj_name+" ")
            fp.write("> PR_D"+str(idisk)+"PHI"+letter(iallprojmem)+" > AP_D"+str(idisk)+"PHI"+letter(iallprojmem))
            for ivm in range(1,nvmmedisks[idisk-1]+1) :
                fp.write(" VMPROJ_D"+str(idisk)+"PHI"+letter(iallprojmem)+str((iallprojmem-1)*nvmmedisks[idisk-1]+ivm))
            fp.write("\n\n")


    #
    # Do the ME for the layers
    #

    CM_list=[]

    for ilayer in range(1,7) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Match Engines for layer "+str(ilayer)+" \n")
        fp.write("#\n")
        print "layer = ",ilayer
        for ivm in range(1,nallprojlayers[ilayer-1]*nvmmelayers[ilayer-1]+1) :
            fp.write("VMSME_L"+str(ilayer)+"PHI"+letter(1+(ivm-1)/nvmmelayers[ilayer-1])+str(ivm))
            fp.write(" VMPROJ_L"+str(ilayer)+"PHI"+letter(1+(ivm-1)/nvmmelayers[ilayer-1])+str(ivm)+" >")
            fp.write(" ME_L"+str(ilayer)+"PHI"+letter(1+(ivm-1)/nvmmelayers[ilayer-1])+str(ivm)+" > ")
            CM_name="CM_L"+str(ilayer)+"PHI"+letter(1+(ivm-1)/nvmmelayers[ilayer-1])+str(ivm)
            fp.write(CM_name)
            CM_list.append(CM_name)
            fp.write("\n\n")
 

    #
    # Do the ME for the disks
    #

    for idisk in range(1,6) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Match Engines for disk "+str(idisk)+" \n")
        fp.write("#\n")
        print "disk = ",idisk
        for ivm in range(1,nallprojdisks[idisk-1]*nvmmedisks[idisk-1]+1) :
            fp.write("VMSME_D"+str(idisk)+"PHI"+letter(1+(ivm-1)/nvmmedisks[idisk-1])+str(ivm))
            fp.write(" VMPROJ_D"+str(idisk)+"PHI"+letter(1+(ivm-1)/nvmmedisks[idisk-1])+str(ivm)+" >")
            fp.write(" ME_D"+str(idisk)+"PHI"+letter(1+(ivm-1)/nvmmedisks[idisk-1])+str(ivm)+" > ")
            CM_name="CM_D"+str(idisk)+"PHI"+letter(1+(ivm-1)/nvmmedisks[idisk-1])+str(ivm)
            fp.write(CM_name)
            CM_list.append(CM_name)
            fp.write("\n\n")
 


    #
    # Do the MC for the layers
    #

    for ilayer in range(1,7) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Match Calculator for layer "+str(ilayer)+" \n")
        fp.write("#\n")
        print "layer = ",ilayer
        for iproj in range(1,nallprojlayers[ilayer-1]+1) :
            for ivm in range(1,nvmmelayers[ilayer-1]+1) :
                fp.write("CM_L"+str(ilayer)+"PHI"+letter(iproj)+str((iproj-1)*nvmmelayers[ilayer-1]+ivm)+" ")
            fp.write("AP_L"+str(ilayer)+"PHI"+letter(iproj)+" ")
            fp.write("AS_L"+str(ilayer)+"PHI"+letter(iproj)+" > ")
            fp.write("MC_L"+str(ilayer)+"PHI"+letter(iproj)+" > ")
            
            fm_name="FM_L1L2"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer!=1 and ilayer!=2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            if extraseeding :
                fm_name="FM_L2L3"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
                if ilayer!=2 and ilayer!=3 and ilayer!=6 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
            fm_name="FM_L3L4"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer!=3 and ilayer!=4 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L5L6"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer!=5 and ilayer!=6 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D1D2"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer==1 or ilayer==2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D3D4"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer==1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2D1"+xx+"_L"+str(ilayer)+"PHI"+letter(iproj)
            if ilayer==1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            #now triplets:
            if displacedseeding :
                fm_name="FM_L3L4L2_L"+str(ilayer)+"PHI"+letter(iproj)
                if ilayer!=2 and ilayer!=3 and ilayer!=4 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
                fm_name="FM_L5L6L4_L"+str(ilayer)+"PHI"+letter(iproj)
                if ilayer!=4 and ilayer!=5 and ilayer!=6 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
                fm_name="FM_L2L3D1_L"+str(ilayer)+"PHI"+letter(iproj)
                if ilayer==1 or ilayer==4:
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
                fm_name="FM_D1D2L2_L"+str(ilayer)+"PHI"+letter(iproj)
                if ilayer==1 or ilayer==3:
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
            fp.write("\n\n")
        
    #
    # Do the MC for the disks
    #

    for idisk in range(1,6) :
        fp.write("\n")
        fp.write("#\n")
        fp.write("# Match Calculator for disk "+str(idisk)+" \n")
        fp.write("#\n")
        print "disk = ",idisk
        for iproj in range(1,nallprojdisks[idisk-1]+1) :
            for ivm in range(1,nvmmedisks[idisk-1]+1) :
                fp.write("CM_D"+str(idisk)+"PHI"+letter(iproj)+str((iproj-1)*nvmmedisks[idisk-1]+ivm)+" ")
            fp.write("AP_D"+str(idisk)+"PHI"+letter(iproj)+" ")
            fp.write("AS_D"+str(idisk)+"PHI"+letter(iproj)+" > ")
            fp.write("MC_D"+str(idisk)+"PHI"+letter(iproj)+" > ")
            
            fm_name="FM_D1D2"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk!=1 and idisk!=2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_D3D4"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk!=3 and idisk!=4 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L1L2"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk!=5 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            if extraseeding :
                fm_name="FM_L2L3"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
                if idisk==1 or idisk==2 or idisk==3 or idisk==4 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
            fm_name="FM_L3L4"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk==1 or idisk==2 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L1D1"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk!=1 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            fm_name="FM_L2D1"+xx+"_D"+str(idisk)+"PHI"+letter(iproj)
            if idisk!=1 and idisk!=5 :
                fp.write(fm_name+" ")
                FM_list.append(fm_name)
            #now triplets:
            if displacedseeding :
                fm_name="FM_L3L4L2_D"+str(idisk)+"PHI"+letter(iproj)
                if idisk==1 or idisk==2 or idisk==3 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
                fm_name="FM_L2L3D1_D"+str(idisk)+"PHI"+letter(iproj)
                if idisk==2 or idisk==3 or idisk==4 :
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)
                fm_name="FM_D1D2L2_D"+str(idisk)+"PHI"+letter(iproj)
                if idisk!=1 and idisk!=2:
                    fp.write(fm_name+" ")
                    FM_list.append(fm_name)

            fp.write("\n\n")
        
        
#
# Do the Track Fits
#

fits = ["L1L2"+xx,"L3L4"+xx,"L5L6"+xx,"D1D2"+xx,"D3D4"+xx,"L1D1"+xx,"L2D1"+xx]

if extraseeding :
    fits.append("L2L3"+xx)

if displacedseeding : 
    fits = fits + ["L3L4L2","L5L6L4","L2L3D1","D1D2L2"]



for fitname in fits:
    fp.write("\n")
    fp.write("#\n")
    fp.write("# Tracklet Fit for seeding "+fitname+" \n")
    fp.write("#\n")
    for fm_name in FM_list :
        if fitname in fm_name :
            fp.write(fm_name+" ")
    for tpar_name in TPAR_list :
        if fitname in tpar_name :
            fp.write(tpar_name+" ")
    fp.write(" > FT_"+fitname+" > TF_"+fitname)
    fp.write("\n\n")
    


#
# purge duplicates
#
fp.write("\n")
fp.write("#\n")
fp.write("# Purge Duplicates\n")
fp.write("#\n")
for fitname in fits:
    fp.write("TF_"+fitname+" ")
fp.write("> PD > ")
for fitname in fits:
    fp.write("CT_"+fitname+" ")
   

print "=========================Summary========================================"
print "StubPair          ",len(SP_list)
if displacedseeding :
    print "StubPair Displaced",len(SPD_list)
    print "StubTriplet       ",len(ST_list)
print "TPAR              ",len(TPAR_list)
print "TPROJ             ",len(TPROJ_list)
print "Cand. Match       ",len(CM_list)
print "Full Match        ",len(FM_list)
