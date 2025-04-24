#! /usr/bin/env python

import sys
import os
import math
import random
from scipy import optimize
from scipy import interpolate
import numpy
import copy

Q2max=1.0 # 1 GeV^2 as the maximally allowed Q2
ion_Form=1 # Form1: Q**2=kT**2+(mn*x)**2, Qmin**2=(mn*x)**2; 
           # Form2: Q**2=kT**2/(1-x)+(mn*x)**2/(1-x), Qmin**2=(mn*x)**2/(1-x)

files=[arg for arg in sys.argv[1:] if arg.startswith('--file=')]
nuclei=[arg for arg in sys.argv[1:] if arg.startswith('--beams=')]
if not files or not nuclei:
    raise Exception( "The usage of it should be e.g., ./lhe_ktsmearing_UPC --beams='Pb208 Pb208' --file='/PATH/TO/file.lhe' --out='ktsmearing.lhe4upc' ")
files=files[0]
files=files.replace('--file=','')
#files=[file.lower() for file in files.split(' ')]
files=[file for file in files.split(' ')]
files=[files[0]]
nuclei=nuclei[0]
nuclei=nuclei.replace('--beams=','')
nuclei=[nucleus.rstrip().lstrip() for nucleus in nuclei.split(' ')]

# name:(RA,aA,wA), RA and aA are in fm, need divide by GeVm12fm to get GeV-1
GeVm12fm=0.1973
WoodsSaxon={'H2':(0.01,0.5882,0),'Li7':(1.77,0.327,0),'Be9':(1.791,0.611,0),'B10':(1.71,0.837,0),'B11':(1.69,0.811,0),\
                'C13':(1.635,1.403,0),'C14':(1.73,1.38,0),'N14':(2.570,0.5052,-0.180),'N15':(2.334,0.498,0.139),'O16':(2.608,0.513,-0.051),'Ne20':(2.791,0.698,-0.168),\
                'Mg24':(3.108,0.607,-0.163),'Mg25':(3.22,0.58,-0.236),'Al27':(3.07,0.519,0),'Si28':(3.340,0.580,-0.233),'Si29':(3.338,0.547,-0.203),'Si30':(3.338,0.547,-0.203),\
                'P31':(3.369,0.582,-0.173),'Cl35':(3.476,0.599,-0.10),'Cl37':(3.554,0.588,-0.13),'Ar40':(3.766,0.586,-0.161),'K39':(3.743,0.595,-0.201),'Ca40':(3.766,0.586,-0.161),\
                'Ca48':(3.7369,0.5245,-0.030),'Ni58':(4.3092,0.5169,-0.1308),'Ni60':(4.4891,0.5369,-0.2668),'Ni61':(4.4024,0.5401,-0.1983),'Ni62':(4.4425,0.5386,-0.2090),'Ni64':(4.5211,0.5278,-0.2284),\
                'Cu63':(4.214,0.586,0),'Kr78':(4.5,0.5,0),'Ag110':(5.33,0.535,0),'Sb122':(5.32,0.57,0),'Xe129':(5.36,0.59,0),'Xe132':(5.4,0.61,0),\
                'Nd142':(5.6135,0.5868,0.096),'Er166':(5.98,0.446,0.19),'W186':(6.58,0.480,0),'Au197':(6.38,0.535,0),'Pb207':(6.62,0.546,0),'Pb208':(6.624,0.549,0)}

if nuclei[0] != 'p' and nuclei[0] not in WoodsSaxon.keys():
    raise ValueError('do not know the first beam type = %s'%nuclei[0])

if nuclei[1] != 'p' and nuclei[1] not in WoodsSaxon.keys():
    raise ValueError('do not know the second beam type = %s'%nuclei[1])

outfile=[arg for arg in sys.argv[1:] if arg.startswith('--out=')]
if not outfile:
    outfile=['ktsmearing.lhe4upc']

outfile=outfile[0]
outfile=outfile.replace('--out=','')

currentdir=os.getcwd()

p_Q2max_save=1
p_x_array=None # an array of log10(1/x)
p_xmax_array=None # an array of maximal function value at logQ2/Q02, where Q02=0.71
p_fmax_array=None # an array of maximal function value
p_xmax_interp=None
p_fmax_interp=None

offset=100

def generate_Q2_epa_proton(x,Q2max):
    if x >= 1.0 or x <= 0:
        raise ValueError( "x >= 1 or x <= 0")
    mp=0.938272081 # proton mass in unit of GeV
    mupomuN=2.793
    Q02=0.71  # in unit of GeV**2
    mp2=mp**2
    Q2min=mp2*x**2/(1-x)

    def xmaxvalue(Q2MAX):
        val=(math.sqrt(Q2MAX*(4*mp2+Q2MAX))-Q2MAX)/(2*mp2)
        return val

    global p_x_array
    global p_Q2max_save
    global p_xmax_array
    global p_fmax_array
    global p_xmax_interp
    global p_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max) : return Q2max

    logQ2oQ02max = math.log(Q2max/Q02)
    logQ2oQ02min = math.log(Q2min/Q02)

    def distfun(xx,logQ2oQ02):
        exp=math.exp(logQ2oQ02)
        funvalue=(-8*mp2**2*xx**2+exp**2*mupomuN**2*Q02**2*\
                       (2-2*xx+xx**2)+2*exp*mp2*Q02*(4-4*xx+mupomuN**2*xx**2))\
                       /(2*exp*(1+exp)**4*Q02*(4*mp2+exp*Q02))
        return funvalue

    if p_x_array is None or (p_Q2max_save != Q2max):
        # we need to generate the grid first
        p_Q2max_save = Q2max
        xmaxQ2max=xmaxvalue(Q2max)
        log10xmaxQ2maxm1=math.log10(1/xmaxQ2max)
        p_x_array=[]
        p_xmax_array=[]
        p_fmax_array=[]
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1=log10xmaxQ2maxm1+0.1*j+log10xm1
                p_x_array.append(tlog10xm1)
                xx=10**(-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx,max_Q2)
                    p_xmax_array.append(max_Q2)
                    p_fmax_array.append(max_fun)
                else:
                    max_Q2 = optimize.fmin(lambda x0: -distfun(xx,x0),\
                                                    (logQ2oQ02max+logQ2oQ02min)/2,\
                                               full_output=False,disp=False)
                    max_fun = distfun(xx,max_Q2[0])
                    p_xmax_array.append(max_Q2[0])
                    p_fmax_array.append(max_fun)
        p_x_array=numpy.array(p_x_array)
        p_xmax_array=numpy.array(p_xmax_array)
        p_fmax_array=numpy.array(p_fmax_array)
        p_xmax_interp=interpolate.interp1d(p_x_array,p_xmax_array)
        p_fmax_interp=interpolate.interp1d(p_x_array,p_fmax_array)
    log10xm1=math.log10(1/x)
    max_x = p_xmax_interp(log10xm1)
    max_fun = p_fmax_interp(log10xm1)
    logQ2oQ02now=logQ2oQ02min
    while True:
        r1=random.random() # a random float number between 0 and 1
        logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
        w=distfun(x,logQ2oQ02now)/max_fun
        r2=random.random() # a random float number between 0 and 1
        if r2 <= w: break
    Q2v=math.exp(logQ2oQ02now)*Q02
    return Q2v

A_Q2max_save=[1,1]
A_x_array=[None,None]  # an array of log10(1/x)
A_xmax_array=[None,None] # an array of maximal function value at logQ2/Q02, where Q02=0.71
A_fmax_array=[None,None] # an array of maximal function value
A_xmax_interp=[None,None]
A_fmax_interp=[None,None]

# first beam: ibeam=0; second beam: ibeam=1
def generate_Q2_epa_ion(ibeam,x,Q2max,RA,aA,wA):
    if x >= 1.0 or x <= 0:
        raise ValueError( "x >= 1 or x <= 0")
    if ibeam not in [0,1]:
        raise ValueError( "ibeam != 0,1")
    mn=0.9315 # averaged nucleon mass in unit of GeV
    Q02=0.71
    mn2=mn**2
    if ion_Form == 2:
        Q2min=mn2*x**2/(1-x)
    else:
        Q2min=mn2*x**2
    RAA=RA/GeVm12fm # from fm to GeV-1
    aAA=aA/GeVm12fm # from fm to GeV-1
    
    
    def xmaxvalue(Q2MAX):
        val=(math.sqrt(Q2MAX*(4*mn2+Q2MAX))-Q2MAX)/(2*mn2)
        return val

    global A_x_array
    global A_Q2max_save
    global A_xmax_array
    global A_fmax_array
    global A_xmax_interp
    global A_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max) : return Q2max

    logQ2oQ02max = math.log(Q2max/Q02)
    logQ2oQ02min = math.log(Q2min/Q02)

    # set rhoA0=1 (irrelvant for this global factor)
    def FchA1(q):
        piqaA=math.pi*q*aAA
        funval=4*math.pi**4*aAA**3/(piqaA**2*math.sinh(piqaA)**2)*\
            (piqaA*math.cosh(piqaA)*math.sin(q*RAA)*(1-wA*aAA**2/RAA**2*\
            (6*math.pi**2/math.sinh(piqaA)**2+math.pi**2-3*RAA**2/aAA**2))\
            -q*RAA*math.sinh(piqaA)*math.cos(q*RAA)*(1-wA*aAA**2/RAA**2*\
            (6*math.pi**2/math.sinh(piqaA)**2+3*math.pi**2-RAA**2/aAA**2)))
        return funval

    # set rhoA0=1 (irrelvant for this global factor
    def FchA2(q):
        funval=0
        # only keep the first two terms
        for n in range(1,3):
            funval=funval+(-1)**(n-1)*n*math.exp(-n*RAA/aAA)/(n**2+q**2*aAA**2)**2*\
                (1+12*wA*aAA**2/RAA**2*(n**2-q**2*aAA**2)/(n**2+q**2*aAA**2)**2)
        funval=funval*8*math.pi*aAA**3
        return funval

    def distfun(xx,logQ2oQ02):
        exp=math.exp(logQ2oQ02)*Q02
        if ion_Form == 2:
            FchA=FchA1(math.sqrt((1-xx)*exp))+FchA2(math.sqrt((1-xx)*exp))
        else:
            FchA=FchA1(math.sqrt(exp))+FchA2(math.sqrt(exp))
        funvalue=(1-Q2min/exp)*FchA**2
        return funvalue

    if A_x_array[ibeam] == None or (A_Q2max_save[ibeam] != Q2max):
        # we need to generate the grid first
        A_Q2max_save[ibeam] = Q2max
        xmaxQ2max=xmaxvalue(Q2max)
        log10xmaxQ2maxm1=math.log10(1/xmaxQ2max)
        A_x_array[ibeam]=[]
        A_xmax_array[ibeam]=[]
        A_fmax_array[ibeam]=[]
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1=log10xmaxQ2maxm1+0.1*j+log10xm1
                A_x_array[ibeam].append(tlog10xm1)
                xx=10**(-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx,max_Q2)
                    A_xmax_array[ibeam].append(max_Q2)
                    A_fmax_array[ibeam].append(max_fun)
                else:
                    max_Q2 = optimize.fmin(lambda x0: -distfun(xx,x0),\
                                                    (logQ2oQ02max+logQ2oQ02min)/2,\
                                               full_output=False,disp=False)
                    max_fun = distfun(xx,max_Q2[0])
                    A_xmax_array[ibeam].append(max_Q2[0])
                    A_fmax_array[ibeam].append(max_fun)
        A_x_array[ibeam]=numpy.array(A_x_array[ibeam])
        A_xmax_array[ibeam]=numpy.array(A_xmax_array[ibeam])
        A_fmax_array[ibeam]=numpy.array(A_fmax_array[ibeam])
        A_xmax_interp[ibeam]=interpolate.interp1d(A_x_array[ibeam],A_xmax_array[ibeam])
        A_fmax_interp[ibeam]=interpolate.interp1d(A_x_array[ibeam],A_fmax_array[ibeam])
    log10xm1=math.log10(1/x)
    max_x = A_xmax_interp[ibeam](log10xm1)
    max_fun = A_fmax_interp[ibeam](log10xm1)
    logQ2oQ02now=logQ2oQ02min
    while True:
        r1=random.random() # a random float number between 0 and 1
        logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
        w=distfun(x,logQ2oQ02now)/max_fun
        r2=random.random() # a random float number between 0 and 1
        if r2 <= w: break
    Q2v=math.exp(logQ2oQ02now)*Q02
    return Q2v

#stream=open("Q2.dat",'w')
#for i in range(100000):
#    Q2v=generate_Q2_epa_ion(1,1e-1,1.0,WoodsSaxon['Pb208'][0],\
#                                WoodsSaxon['Pb208'][1],WoodsSaxon['Pb208'][2])
#    stream.write('%12.7e\n'%Q2v)
#stream.close()

def boostl(Q,PBOO,P):
    """Boost P via PBOO with PBOO^2=Q^2 to PLB"""
    # it boosts P from (Q,0,0,0) to PBOO
    # if P=(PBOO[0],-PBOO[1],-PBOO[2],-PBOO[3])
    # it will boost P to (Q,0,0,0)
    PLB=[0,0,0,0] # energy, px, py, pz in unit of GeV
    PLB[0]=(PBOO[0]*P[0]+PBOO[3]*P[3]+PBOO[2]*P[2]+PBOO[1]*P[1])/Q
    FACT=(PLB[0]+P[0])/(Q+PBOO[0])
    for j in range(1,4):
        PLB[j]=P[j]+FACT*PBOO[j]
    return PLB

def boostl2(Q,PBOO1,PBOO2,P):
    """Boost P from PBOO1 (PBOO1^2=Q^2) to PBOO2 (PBOO2^2=Q^2) frame"""
    PBOO10=[PBOO1[0],-PBOO1[1],-PBOO1[2],-PBOO1[3]]
    PRES=boostl(Q,PBOO10,P) # PRES is in (Q,0,0,0) frame
    PLB=boostl(Q,PBOO2,PRES) # PLB is in PBOO2 frame
    return PLB

def boostToEcm(E1,E2,pext):
    Ecm=2*math.sqrt(E1*E2)
    PBOO=[E1+E2,0,0,E2-E1]
    pext2=copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j]=boostl(Ecm,PBOO,pext[j])
    return pext2

def boostFromEcm(E1,E2,pext):
    Ecm=2*math.sqrt(E1*E2)
    PBOO=[E1+E2,0,0,E1-E2]
    pext2=copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j]=boostl(Ecm,PBOO,pext[j])
    return pext2

def InitialMomentumReshuffle(Ecm,x1,x2,Q1,Q2,pext):
    r1=random.random() # a random float number between 0 and 1
    r2=random.random() # a random float number between 0 and 1
    ph1=2*math.pi*r1
    ph2=2*math.pi*r2
    Kperp2=Q1**2+Q2**2+2*Q1*Q2*math.cos(ph1-ph2)
    Kperp2max=Ecm**2*(min(1,x1/x2,x2/x1)-x1*x2)
    if Kperp2 >= Kperp2max:
        return None
    x1bar=math.sqrt(x1/x2*Kperp2/Ecm**2+x1**2)
    x2bar=math.sqrt(x2/x1*Kperp2/Ecm**2+x2**2)
    if x1bar >= 1.0 or x2bar >= 1.0: return None
    pext2=copy.deepcopy(pext)
    # new initial state
    pext2[0][0]=Ecm/2*x1bar
    pext2[0][1]=Q1*math.cos(ph1)
    pext2[0][2]=Q1*math.sin(ph1)
    pext2[0][3]=Ecm/2*x1bar
    pext2[1][0]=Ecm/2*x2bar
    pext2[1][1]=Q2*math.cos(ph2)
    pext2[1][2]=Q2*math.sin(ph2)
    pext2[1][3]=-Ecm/2*x2bar
    # new final state
    PBOO1=[0,0,0,0]
    PBOO2=[0,0,0,0]
    for j in range(4):
        PBOO1[j]=pext[0][j]+pext[1][j]
        PBOO2[j]=pext2[0][j]+pext2[1][j]
    Q=math.sqrt(x1*x2)*Ecm
    for j in range(2,len(pext)):
        pext2[j]=boostl2(Q,PBOO1,PBOO2,pext[j])
    return pext2


headers=[]
inits=[]
events=[]
ninit0=0
ninit1=0
firstinit=""
E_beam1=0
E_beam2=0
PID_beam1=0
PID_beam2=0

nevent=0

ilil=0
for i,file in enumerate(files):
    stream=open(file,'r')
    headQ=True
    initQ=False
    iinit=-1
    ievent=-1
    eventQ=False
    this_event=[]
    n_particles=0
    rwgtQ=False
    mgrwtQ=False
    procid=None
    proc_dict={}
    for line in stream:
        sline=line.replace('\n','')
        if "<init>" in line or "<init " in line:
            initQ=True
            headQ=False
            iinit=iinit+1
            if i==0: inits.append(sline)
        elif headQ and i == 0:
            headers.append(sline)
        elif "</init>" in line or "</init " in line:
            initQ=False
            iinit=-1
            if i==0: inits.append(sline)
        elif initQ:
            iinit=iinit+1
            if iinit == 1:
                if i == 0:
                    firstinit=sline
                    ninit0=len(inits)
                    inits.append(sline)
                    firstinit=' '.join(firstinit.split()[:-1])
                    ff=firstinit.strip().split()
                    PID_beam1=int(ff[0])
                    PID_beam2=int(ff[1])
                    E_beam1=float(ff[2])
                    E_beam2=float(ff[3])
                    if abs(PID_beam1) != 2212 or abs(PID_beam2) != 2212:
                        raise ValueError( "Not a proton-proton collider")
                    ninit1=int(sline.split()[-1])
                else:
                    ninit1=ninit1+int(sline.split()[-1])
                    sline=' '.join(sline.split()[:-1])
                    if not sline == firstinit:
                        raise Exception( "the beam information of the LHE files is not identical")
            elif iinit == 2:
                procid=sline.split()[-1]
                ilil=ilil+1
                sline=' '.join(sline.split()[:-1])+(' %d'%(offset+ilil))
                proc_dict[procid]=offset+ilil
                if i == 0:
                    inits.append(sline)
                else:
                    inits.insert(-1,sline)
            elif iinit >= 3:
                if i == 0:
                    inits.append(sline)
                else:
                    inits.insert(-1,sline)
            else:
                raise Exception( "should not reach here. Do not understand the <init> block")
        elif "<event>" in line or "<event " in line:
            eventQ=True
            ievent=ievent+1
            events.append(sline)
        elif "</event>" in line or "</event " in line:
            nevent=nevent+1
            eventQ=False
            rwgtQ=False
            mgrwtQ=False
            ievent=-1
            this_event=[]
            n_particles=0
            events.append(sline)
            #if nevent >= 10: break
        elif eventQ:
            ievent=ievent+1
            if ievent == 1:
                found=False
                for procid,new_procid in proc_dict.items():
                    if ' '+procid+' ' not in sline: continue
                    procpos=sline.index(' '+procid+' ')
                    found=True
                    sline=sline[:procpos]+(' %d'%(new_procid))+sline[procpos+len(' '+procid):]
                    break
                if not found: raise Exception( "do not find the correct proc id !")
                n_particles=int(sline.split()[0])
                #procpos=sline.index(' '+procid)
                #sline=sline[:procpos]+(' %d'%(1+i))+sline[procpos+len(' '+procid):]
            elif "<rwgt" in sline:
                rwgtQ=True
            elif "</rwgt" in sline:
                rwgtQ=False
            elif "<mgrwt" in sline:
                mgrwtQ=True
            elif "</mgrwt" in sline:
                mgrwtQ=False                
            elif not rwgtQ and not mgrwtQ:
                sline2=sline.split()
                particle=[int(sline2[0]),int(sline2[1]),int(sline2[2]),int(sline2[3]),\
                              int(sline2[4]),int(sline2[5]),float(sline2[6]),float(sline2[7]),\
                              float(sline2[8]),float(sline2[9]),float(sline2[10]),\
                              float(sline2[11]),float(sline2[12])]
                this_event.append(particle)
                if ievent == n_particles+1:
                    # get the momenta and masses
                    x1=this_event[0][9]/E_beam1
                    x2=this_event[1][9]/E_beam2
                    pext=[]
                    mass=[]
                    for j in range(n_particles):
                        pext.append([this_event[j][9],this_event[j][6],\
                                         this_event[j][7],this_event[j][8]])
                        mass.append(this_event[j][10])
                    # first we need to boost from antisymmetric beams to symmetric beams
                    if E_beam1 != E_beam2:
                        pext=boostToEcm(E_beam1,E_beam2,pext)
                    Ecm=2*math.sqrt(E_beam1*E_beam2)
                    pext_new = None
                    Q1=0
                    Q2=0
                    while pext_new == None:
                        # generate Q1 and Q2
                        if nuclei[0] == 'p':
                            Q12=generate_Q2_epa_proton(x1,Q2max)
                        else:
                            RA,aA,wA=WoodsSaxon[nuclei[0]]
                            Q12=generate_Q2_epa_ion(0,x1,Q2max,RA,aA,wA)
                        if nuclei[1] == 'p':
                            Q22=generate_Q2_epa_proton(x2,Q2max)
                        else:
                            if nuclei[0] == nuclei[1]:
                                RA,aA,wA=WoodsSaxon[nuclei[0]]
                                Q22=generate_Q2_epa_ion(0,x2,Q2max,RA,aA,wA)
                            else:
                                RA,aA,wA=WoodsSaxon[nuclei[1]]
                                Q22=generate_Q2_epa_ion(1,x2,Q2max,RA,aA,wA)
                        Q1=math.sqrt(Q12)
                        Q2=math.sqrt(Q22)
                        # perform the initial momentum reshuffling
                        pext_new=InitialMomentumReshuffle(Ecm,x1,x2,Q1,Q2,pext)

                    if E_beam1 != E_beam2:
                        # boost back from the symmetric beams to antisymmetric beams
                        pext_new=boostFromEcm(E_beam1,E_beam2,pext_new)
                    # update the event information
                    # negative invariant mass means negative invariant mass square (-Q**2, spacelike)
                    this_event[0][10]=-Q1
                    this_event[1][10]=-Q2
                    for j in range(n_particles):
                        this_event[j][9]=pext_new[j][0]
                        this_event[j][6]=pext_new[j][1]
                        this_event[j][7]=pext_new[j][2]
                        this_event[j][8]=pext_new[j][3]
                        newsline="      %d    %d     %d    %d    %d    %d  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e"%tuple(this_event[j])
                        events.append(newsline)
                continue
            events.append(sline)
    stream.close()

# modify the number of process information
firstinit=firstinit+(' %d'%ninit1)
inits[ninit0]=firstinit

text='\n'.join(headers)+'\n'
text=text+'\n'.join(inits)+'\n'
text=text+'\n'.join(events)
if '<LesHouchesEvents' in headers[0]: text=text+'\n</LesHouchesEvents>\n'

stream=open(outfile,'w')
stream.write(text)
stream.close()
print ("INFO: The final produced lhe file is %s"%outfile)

