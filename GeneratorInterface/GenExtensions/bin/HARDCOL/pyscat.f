C=====================================================================
C  This routine was modified from the PYTHIA 6.418 code, which is
C              (C) Torbjorn Sjostrand, Lund 2008.
C
C  The modifications are part of the HARDCOL package for 
C  hard color singlet exchange, and refer to PYTHIA version 6.418.
C  Modifications implemented by Rikard Enberg, 2001-2003 and 2008.  
C  See http://www.isv.uu.se/thep/hardcol/      
C=====================================================================
 
C*********************************************************************
 
C...PYSCAT
C...Finds outgoing flavours and event type; sets up the kinematics
C...and colour flow of the hard scattering
 
      SUBROUTINE PYSCAT
 
C...Double precision and integer declarations
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Parameter statement to help give large particle numbers.
      PARAMETER (KSUSY1=1000000,KSUSY2=2000000,KTECHN=3000000,
     &KEXCIT=4000000,KDIMEN=5000000)
C...Parameter statement for maximum size of showers.
      PARAMETER (MAXNUR=1000)
 
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/
 
C...Commonblocks.
      COMMON/PYPART/NPART,NPARTD,IPART(MAXNUR),PTPART(MAXNUR)
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)
      COMMON/PYINT4/MWID(500),WIDS(500,5)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON/PYSSMT/ZMIX(4,4),UMIX(2,2),VMIX(2,2),SMZ(4),SMW(2),
     &SFMIX(16,4),ZMIXI(4,4),UMIXI(2,2),VMIXI(2,2)
      COMMON/PYTCSM/ITCM(0:99),RTCM(0:99)
      COMMON/PYPUED/IUED(0:99),RUED(0:99)
      SAVE /PYPART/,/PYJETS/,/PYDAT1/,/PYDAT2/,/PYDAT3/,/PYSUBS/,
     &/PYPARS/,/PYINT1/,/PYINT2/,/PYINT3/,/PYINT4/,/PYINT5/,/PYSSMT/,
     &/PYTCSM/,/PYPUED/
C...Local arrays and saved variables
      DIMENSION WDTP(0:400),WDTE(0:400,0:5),PMQ(2),Z(2),CTHE(2),
     &PHI(2),KUPPO(100),VINTSV(41:66),ILAB(100)
      INTEGER IOKFLA(6),IIFLAV
C...UED related declarations:
C...equivalences between ordered particles (451->475)
C...and UED particle code (5 000 000 + id)
      DIMENSION IUEDEQ(475),MUED(2)
      DATA (IUEDEQ(I),I=451,475)/
     & 6100001,6100002,6100003,6100004,6100005,6100006, 
     & 5100001,5100002,5100003,5100004,5100005,5100006, 
     & 6100011,6100013,6100015,                         
     & 5100012,5100011,5100014,5100013,5100016,5100015, 
     & 5100021,5100022,5100023,5100024/                 
      SAVE VINTSV
 
C...Read out process
      ISUB=MINT(1)
      ISUBSV=ISUB
 
C...Restore information for low-pT processes
      IF(ISUB.EQ.95.AND.MINT(57).GE.1) THEN
        DO 100 J=41,66
  100   VINT(J)=VINTSV(J)
      ENDIF
 
C...Convert H' or A process into equivalent H one
      IHIGG=1
      KFHIGG=25
      IF((ISUB.GE.151.AND.ISUB.LE.160).OR.(ISUB.GE.171.AND.
     &ISUB.LE.190)) THEN
        IHIGG=2
        IF(MOD(ISUB-1,10).GE.5) IHIGG=3
        KFHIGG=33+IHIGG
        IF(ISUB.EQ.151.OR.ISUB.EQ.156) ISUB=3
        IF(ISUB.EQ.152.OR.ISUB.EQ.157) ISUB=102
        IF(ISUB.EQ.153.OR.ISUB.EQ.158) ISUB=103
        IF(ISUB.EQ.171.OR.ISUB.EQ.176) ISUB=24
        IF(ISUB.EQ.172.OR.ISUB.EQ.177) ISUB=26
        IF(ISUB.EQ.173.OR.ISUB.EQ.178) ISUB=123
        IF(ISUB.EQ.174.OR.ISUB.EQ.179) ISUB=124
        IF(ISUB.EQ.181.OR.ISUB.EQ.186) ISUB=121
        IF(ISUB.EQ.182.OR.ISUB.EQ.187) ISUB=122
        IF(ISUB.EQ.183.OR.ISUB.EQ.188) ISUB=111
        IF(ISUB.EQ.184.OR.ISUB.EQ.189) ISUB=112
        IF(ISUB.EQ.185.OR.ISUB.EQ.190) ISUB=113
      ENDIF
 
      IF(ISUB.EQ.401.OR.ISUB.EQ.402) KFHIGG=KFPR(ISUB,1)
 
C...Convert bottomonium process into equivalent charmonium ones.
      IF(ISUB.GE.461.AND.ISUB.LE.479) ISUB=ISUB-40
 
C...Choice of subprocess, number of documentation lines
      IDOC=6+ISET(ISUB)
      IF(ISUB.EQ.95) IDOC=8
      IF(ISET(ISUB).EQ.5) IDOC=9
      IF(ISET(ISUB).EQ.11) IDOC=4+NUP
      MINT(3)=IDOC-6
      IF(IDOC.GE.9.AND.ISET(ISUB).LE.4) IDOC=IDOC+2
      MINT(4)=IDOC
      IPU1=MINT(84)+1
      IPU2=MINT(84)+2
      IPU3=MINT(84)+3
      IPU4=MINT(84)+4
      IPU5=MINT(84)+5
      IPU6=MINT(84)+6
 
C...Reset K, P and V vectors. Store incoming particles
      DO 120 JT=1,MSTP(126)+100
        I=MINT(83)+JT
        IF(I.GT.MSTU(4)) GOTO 120
        DO 110 J=1,5
          K(I,J)=0
          P(I,J)=0D0
          V(I,J)=0D0
  110   CONTINUE
  120 CONTINUE
      DO 140 JT=1,2
        I=MINT(83)+JT
        K(I,1)=21
        K(I,2)=MINT(10+JT)
        DO 130 J=1,5
          P(I,J)=VINT(285+5*JT+J)
  130   CONTINUE
  140 CONTINUE
      MINT(6)=2
      KFRES=0
 
C...Store incoming partons in their CM-frame. Save pdf value.
      SH=VINT(44)
      SHR=SQRT(SH)
      SHP=VINT(26)*VINT(2)
      SHPR=SQRT(SHP)
      SHUSER=SHR
      IF(ISET(ISUB).GE.3.AND.ISET(ISUB).LE.5) SHUSER=SHPR
      DO 150 JT=1,2
        I=MINT(84)+JT
        K(I,1)=14
        K(I,2)=MINT(14+JT)
        K(I,3)=MINT(83)+2+JT
        P(I,3)=0.5D0*SHUSER*(-1D0)**(JT-1)
        P(I,4)=0.5D0*SHUSER
        VINT(38+JT)=XSFX(JT,MINT(14+JT))
  150 CONTINUE
 
C...Copy incoming partons to documentation lines
      DO 170 JT=1,2
        I1=MINT(83)+4+JT
        I2=MINT(84)+JT
        K(I1,1)=21
        K(I1,2)=K(I2,2)
        K(I1,3)=I1-2
        DO 160 J=1,5
          P(I1,J)=P(I2,J)
  160   CONTINUE
  170 CONTINUE
 
C...Choose new quark/lepton flavour for relevant annihilation graphs
      IF(ISUB.EQ.12.OR.ISUB.EQ.53.OR.ISUB.EQ.54.OR.ISUB.EQ.58.OR.
     &ISUB.EQ.314.OR.ISUB.EQ.319.OR.ISUB.EQ.316.OR.
     &(ISUB.GE.135.AND.ISUB.LE.140).OR.ISUB.EQ.382.OR.ISUB.EQ.385) THEN
        IGLGA=21
        IF(ISUB.EQ.58.OR.(ISUB.GE.137.AND.ISUB.LE.140)) IGLGA=22
        CALL PYWIDT(IGLGA,SH,WDTP,WDTE)
  180   RKFL=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))*PYR(0)
        DO 190 I=1,MDCY(IGLGA,3)
          KFLF=KFDP(I+MDCY(IGLGA,2)-1,1)
          RKFL=RKFL-(WDTE(I,1)+WDTE(I,2)+WDTE(I,4))
          IF(RKFL.LE.0D0) GOTO 200
  190   CONTINUE
  200   CONTINUE
        IF((ISUB.EQ.53.OR.ISUB.EQ.385.OR.ISUB.EQ.314.OR.ISUB.EQ.319
     &      .OR.ISUB.EQ.316).AND.MINT(2).LE.2) THEN
          IF(KFLF.GE.4) GOTO 180
        ELSEIF((ISUB.EQ.53.OR.ISUB.EQ.385.OR.ISUB.EQ.314.OR.ISUB.EQ.319.
     &       OR.ISUB.EQ.316).AND.MINT(2).LE.4) THEN
          KFLF=4
          MINT(2)=MINT(2)-2
        ELSEIF(ISUB.EQ.53.OR.ISUB.EQ.385.OR.ISUB.EQ.314.OR.ISUB.EQ.319.
     &        OR.ISUB.EQ.316) THEN
          KFLF=5
          MINT(2)=MINT(2)-4
        ELSEIF(ISUB.EQ.382.AND.ITCM(5).EQ.1.AND.IABS(MINT(15)).LE.2
     &  .AND.IABS(KFLF).GE.3) THEN
          FACQQB=VINT(58)**2*4D0/9D0*(VINT(45)**2+VINT(46)**2)/
     &    VINT(44)**2
          FACCIB=VINT(46)**2/RTCM(41)**4
          IF(FACQQB/(FACQQB+FACCIB).LT.PYR(0)) GOTO 180
        ELSEIF(ISUB.EQ.382.AND.ITCM(5).EQ.5.AND.MINT(2).EQ.2) THEN
          KFLF=5
          MINT(2)=1
        ELSEIF(ISUB.EQ.382.AND.ITCM(5).EQ.5.AND.MINT(2).EQ.1) THEN
          IF(KFLF.EQ.5) GOTO 180
        ELSEIF(ISUB.EQ.54.OR.ISUB.EQ.135.OR.ISUB.EQ.136) THEN
          IF((KCHG(PYCOMP(KFLF),1)/2D0)**2.LT.PYR(0)) GOTO 180
        ELSEIF(ISUB.EQ.58.OR.(ISUB.GE.137.AND.ISUB.LE.140)) THEN
          IF((KCHG(PYCOMP(KFLF),1)/3D0)**2.LT.PYR(0)) GOTO 180
        ENDIF
      ENDIF
 
C...Final state flavours and colour flow: default values
      JS=1
      MINT(21)=MINT(15)
      MINT(22)=MINT(16)
      MINT(23)=0
      MINT(24)=0
      KCC=20
      KCS=ISIGN(1,MINT(15))
 
      IF(ISET(ISUB).EQ.11) THEN
C...User-defined processes: find products
        MINT(3)=0
        DO 210 IUP=3,NUP
          IF(ISTUP(IUP).LT.1.OR.ISTUP(IUP).GT.3) THEN
          ELSEIF(NUP.EQ.5.AND.IUP.GE.4.AND.MOTHUP(1,4).EQ.3) THEN
            MINT(21+IUP)=IDUP(IUP)
          ELSEIF(ISTUP(IUP).EQ.1.AND.(ISTUP(MOTHUP(1,IUP)).EQ.2.OR.
     &    ISTUP(MOTHUP(1,IUP)).EQ.3).AND.IDUP(MOTHUP(1,IUP)).NE.0) THEN
          ELSEIF(IDUP(IUP).EQ.0) THEN
          ELSE
            MINT(3)=MINT(3)+1
            IF(MINT(3).LE.6) MINT(20+MINT(3))=IDUP(IUP)
          ENDIF
  210   CONTINUE
 
      ELSEIF(ISUB.LE.10) THEN
        IF(ISUB.EQ.1) THEN
C...f + fbar -> gamma*/Z0
          KFRES=23
 
        ELSEIF(ISUB.EQ.2) THEN
C...f + fbar' -> W+/-
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.3) THEN
C...f + fbar -> h0 (or H0, or A0)
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.4) THEN
C...gamma + W+/- -> W+/-
 
        ELSEIF(ISUB.EQ.5) THEN
C...Z0 + Z0 -> h0
          XH=SH/SHP
          MINT(21)=MINT(15)
          MINT(22)=MINT(16)
          PMQ(1)=PYMASS(MINT(21))
          PMQ(2)=PYMASS(MINT(22))
  220     JT=INT(1.5D0+PYR(0))
          ZMIN=2D0*PMQ(JT)/SHPR
          ZMAX=1D0-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/
     &    (SHPR*(SHPR-PMQ(3-JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          Z(JT)=ZMIN+(ZMAX-ZMIN)*PYR(0)
          IF(-1D0+(1D0+XH)/(1D0-Z(JT))-XH/(1D0-Z(JT))**2.LT.
     &    (1D0-XH)**2/(4D0*XH)*PYR(0)) GOTO 220
          SQC1=1D0-4D0*PMQ(JT)**2/(Z(JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 220
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP)
          CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(JT)=MIN(1D0,MAX(-1D0,CTHE(JT)))
          Z(3-JT)=1D0-XH/(1D0-Z(JT))
          SQC1=1D0-4D0*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 220
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP)
          CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(3-JT)=MIN(1D0,MAX(-1D0,CTHE(3-JT)))
          PHIR=PARU(2)*PYR(0)
          CPHI=COS(PHIR)
          ANG=CTHE(1)*CTHE(2)-SQRT(1D0-CTHE(1)**2)*
     &    SQRT(1D0-CTHE(2)**2)*CPHI
          Z1=2D0-Z(JT)
          Z2=ANG*SQRT(Z(JT)**2-4D0*PMQ(JT)**2/SHP)
          Z3=1D0-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP
          Z(3-JT)=2D0/(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*
     &    PMQ(3-JT)**2/SHP))
          ZMIN=2D0*PMQ(3-JT)/SHPR
          ZMAX=1D0-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 220
          KCC=22
          KFRES=25
 
        ELSEIF(ISUB.EQ.6) THEN
C...Z0 + W+/- -> W+/-
 
        ELSEIF(ISUB.EQ.7) THEN
C...W+ + W- -> Z0
 
        ELSEIF(ISUB.EQ.8) THEN
C...W+ + W- -> h0
          XH=SH/SHP
  230     DO 260 JT=1,2
            I=MINT(14+JT)
            IA=IABS(I)
            IF(IA.LE.10) THEN
              RVCKM=VINT(180+I)*PYR(0)
              DO 240 J=1,MSTP(1)
                IB=2*J-1+MOD(IA,2)
                IPM=(5-ISIGN(1,I))/2
                IDC=J+MDCY(IA,2)+2
                IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 240
                MINT(20+JT)=ISIGN(IB,I)
                RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
                IF(RVCKM.LE.0D0) GOTO 250
  240         CONTINUE
            ELSE
              IB=2*((IA+1)/2)-1+MOD(IA,2)
              MINT(20+JT)=ISIGN(IB,I)
            ENDIF
  250       PMQ(JT)=PYMASS(MINT(20+JT))
  260     CONTINUE
          JT=INT(1.5D0+PYR(0))
          ZMIN=2D0*PMQ(JT)/SHPR
          ZMAX=1D0-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/
     &    (SHPR*(SHPR-PMQ(3-JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(ZMIN.GE.ZMAX) GOTO 230
          Z(JT)=ZMIN+(ZMAX-ZMIN)*PYR(0)
          IF(-1D0+(1D0+XH)/(1D0-Z(JT))-XH/(1D0-Z(JT))**2.LT.
     &    (1D0-XH)**2/(4D0*XH)*PYR(0)) GOTO 230
          SQC1=1D0-4D0*PMQ(JT)**2/(Z(JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 230
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(24,1)**2-PMQ(JT)**2)/(Z(JT)*SHP)
          CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(JT)=MIN(1D0,MAX(-1D0,CTHE(JT)))
          Z(3-JT)=1D0-XH/(1D0-Z(JT))
          SQC1=1D0-4D0*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 230
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(24,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP)
          CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(3-JT)=MIN(1D0,MAX(-1D0,CTHE(3-JT)))
          PHIR=PARU(2)*PYR(0)
          CPHI=COS(PHIR)
          ANG=CTHE(1)*CTHE(2)-SQRT(1D0-CTHE(1)**2)*
     &    SQRT(1D0-CTHE(2)**2)*CPHI
          Z1=2D0-Z(JT)
          Z2=ANG*SQRT(Z(JT)**2-4D0*PMQ(JT)**2/SHP)
          Z3=1D0-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP
          Z(3-JT)=2D0/(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*
     &    PMQ(3-JT)**2/SHP))
          ZMIN=2D0*PMQ(3-JT)/SHPR
          ZMAX=1D0-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 230
          KCC=22
          KFRES=25
 
        ELSEIF(ISUB.EQ.10) THEN
C...f + f' -> f + f' (gamma/Z/W exchange); th = (p(f)-p(f))**2
          IF(MINT(2).EQ.1) THEN
            KCC=22
          ELSE
C...W exchange: need to mix flavours according to CKM matrix
            DO 280 JT=1,2
              I=MINT(14+JT)
              IA=IABS(I)
              IF(IA.LE.10) THEN
                RVCKM=VINT(180+I)*PYR(0)
                DO 270 J=1,MSTP(1)
                  IB=2*J-1+MOD(IA,2)
                  IPM=(5-ISIGN(1,I))/2
                  IDC=J+MDCY(IA,2)+2
                  IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 270
                  MINT(20+JT)=ISIGN(IB,I)
                  RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
                  IF(RVCKM.LE.0D0) GOTO 280
  270           CONTINUE
              ELSE
                IB=2*((IA+1)/2)-1+MOD(IA,2)
                MINT(20+JT)=ISIGN(IB,I)
              ENDIF
  280       CONTINUE
            KCC=22
          ENDIF
        ENDIF
 
      ELSEIF(ISUB.LE.20) THEN
        IF(ISUB.EQ.11) THEN
C...f + f' -> f + f' (g exchange); th = (p(f)-p(f))**2
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.12) THEN
C...f + fbar -> f' + fbar'; th = (p(f)-p(f'))**2
          MINT(21)=ISIGN(KFLF,MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
 
        ELSEIF(ISUB.EQ.13) THEN
C...f + fbar -> g + g; th arbitrary
          MINT(21)=21
          MINT(22)=21
          KCC=MINT(2)+4
 
        ELSEIF(ISUB.EQ.14) THEN
C...f + fbar -> g + gamma; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=22
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.15) THEN
C...f + fbar -> g + Z0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=23
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.16) THEN
C...f + fbar' -> g + W+/-; th = (p(f)-p(W-))**2 or (p(fbar')-p(W+))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=ISIGN(24,KCH1+KCH2)
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.17) THEN
C...f + fbar -> g + h0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=25
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.18) THEN
C...f + fbar -> gamma + gamma; th arbitrary
          MINT(21)=22
          MINT(22)=22
 
        ELSEIF(ISUB.EQ.19) THEN
C...f + fbar -> gamma + Z0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=22
          MINT(23-JS)=23
 
        ELSEIF(ISUB.EQ.20) THEN
C...f + fbar' -> gamma + W+/-; th = (p(f)-p(W-))**2 or
C...(p(fbar')-p(W+))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2
          MINT(20+JS)=22
          MINT(23-JS)=ISIGN(24,KCH1+KCH2)
        ENDIF
 
      ELSEIF(ISUB.LE.30) THEN
        IF(ISUB.EQ.21) THEN
C...f + fbar -> gamma + h0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=22
          MINT(23-JS)=25
 
        ELSEIF(ISUB.EQ.22) THEN
C...f + fbar -> Z0 + Z0; th arbitrary
          MINT(21)=23
          MINT(22)=23
 
        ELSEIF(ISUB.EQ.23) THEN
C...f + fbar' -> Z0 + W+/-; th = (p(f)-p(W-))**2 or (p(fbar')-p(W+))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2
          MINT(20+JS)=23
          MINT(23-JS)=ISIGN(24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.24) THEN
C...f + fbar -> Z0 + h0 (or H0, or A0); th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=23
          MINT(23-JS)=KFHIGG
 
        ELSEIF(ISUB.EQ.25) THEN
C...f + fbar -> W+ + W-; th = (p(f)-p(W-))**2
          MINT(21)=-ISIGN(24,MINT(15))
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.26) THEN
C...f + fbar' -> W+/- + h0 (or H0, or A0);
C...th = (p(f)-p(W-))**2 or (p(fbar')-p(W+))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2
          MINT(20+JS)=ISIGN(24,KCH1+KCH2)
          MINT(23-JS)=KFHIGG
 
        ELSEIF(ISUB.EQ.27) THEN
C...f + fbar -> h0 + h0
 
        ELSEIF(ISUB.EQ.28) THEN
C...f + g -> f + g; th = (p(f)-p(f))**2
          IF(MINT(15).EQ.21) JS=2
          KCC=MINT(2)+6
          IF(MINT(15).EQ.21) KCC=KCC+2
          IF(MINT(15).NE.21) KCS=ISIGN(1,MINT(15))
          IF(MINT(16).NE.21) KCS=ISIGN(1,MINT(16))
 
        ELSEIF(ISUB.EQ.29) THEN
C...f + g -> f + gamma; th = (p(f)-p(f))**2
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=22
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.30) THEN
C...f + g -> f + Z0; th = (p(f)-p(f))**2
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=23
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
        ENDIF
 
      ELSEIF(ISUB.LE.40) THEN
        IF(ISUB.EQ.31) THEN
C...f + g -> f' + W+/-; th = (p(f)-p(f'))**2; choose flavour f'
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(24,KCHG(IA,1)*I)
          RVCKM=VINT(180+I)*PYR(0)
          DO 290 J=1,MSTP(1)
            IB=2*J-1+MOD(IA,2)
            IPM=(5-ISIGN(1,I))/2
            IDC=J+MDCY(IA,2)+2
            IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 290
            MINT(20+JS)=ISIGN(IB,I)
            RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
            IF(RVCKM.LE.0D0) GOTO 300
  290     CONTINUE
  300     KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.32) THEN
C...f + g -> f + h0; th = (p(f)-p(f))**2
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=25
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.33) THEN
C...f + gamma -> f + g; th=(p(f)-p(f))**2
          IF(MINT(15).EQ.22) JS=2
          MINT(23-JS)=21
          KCC=24+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.34) THEN
C...f + gamma -> f + gamma; th=(p(f)-p(f))**2
          IF(MINT(15).EQ.22) JS=2
          KCC=22
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.35) THEN
C...f + gamma -> f + Z0; th=(p(f)-p(f))**2
          IF(MINT(15).EQ.22) JS=2
          MINT(23-JS)=23
          KCC=22
 
        ELSEIF(ISUB.EQ.36) THEN
C...f + gamma -> f' + W+/-; th=(p(f)-p(f'))**2
          IF(MINT(15).EQ.22) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(24,KCHG(IA,1)*I)
          IF(IA.LE.10) THEN
            RVCKM=VINT(180+I)*PYR(0)
            DO 310 J=1,MSTP(1)
              IB=2*J-1+MOD(IA,2)
              IPM=(5-ISIGN(1,I))/2
              IDC=J+MDCY(IA,2)+2
              IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 310
              MINT(20+JS)=ISIGN(IB,I)
              RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
              IF(RVCKM.LE.0D0) GOTO 320
  310       CONTINUE
          ELSE
            IB=2*((IA+1)/2)-1+MOD(IA,2)
            MINT(20+JS)=ISIGN(IB,I)
          ENDIF
  320     KCC=22
 
        ELSEIF(ISUB.EQ.37) THEN
C...f + gamma -> f + h0
 
        ELSEIF(ISUB.EQ.38) THEN
C...f + Z0 -> f + g
 
        ELSEIF(ISUB.EQ.39) THEN
C...f + Z0 -> f + gamma
 
        ELSEIF(ISUB.EQ.40) THEN
C...f + Z0 -> f + Z0
        ENDIF
 
      ELSEIF(ISUB.LE.50) THEN
        IF(ISUB.EQ.41) THEN
C...f + Z0 -> f' + W+/-
 
        ELSEIF(ISUB.EQ.42) THEN
C...f + Z0 -> f + h0
 
        ELSEIF(ISUB.EQ.43) THEN
C...f + W+/- -> f' + g
 
        ELSEIF(ISUB.EQ.44) THEN
C...f + W+/- -> f' + gamma
 
        ELSEIF(ISUB.EQ.45) THEN
C...f + W+/- -> f' + Z0
 
        ELSEIF(ISUB.EQ.46) THEN
C...f + W+/- -> f' + W+/-
 
        ELSEIF(ISUB.EQ.47) THEN
C...f + W+/- -> f' + h0
 
        ELSEIF(ISUB.EQ.48) THEN
C...f + h0 -> f + g
 
        ELSEIF(ISUB.EQ.49) THEN
C...f + h0 -> f + gamma
 
        ELSEIF(ISUB.EQ.50) THEN
C...f + h0 -> f + Z0
        ENDIF
 
      ELSEIF(ISUB.LE.60) THEN
        IF(ISUB.EQ.51) THEN
C...f + h0 -> f' + W+/-
 
        ELSEIF(ISUB.EQ.52) THEN
C...f + h0 -> f + h0
 
        ELSEIF(ISUB.EQ.53) THEN
C...g + g -> f + fbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.54) THEN
C...g + gamma -> f + fbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=27
          IF(MINT(16).EQ.21) KCC=28
 
        ELSEIF(ISUB.EQ.55) THEN
C...g + Z0 -> f + fbar
 
        ELSEIF(ISUB.EQ.56) THEN
C...g + W+/- -> f + fbar'
 
        ELSEIF(ISUB.EQ.57) THEN
C...g + h0 -> f + fbar
 
        ELSEIF(ISUB.EQ.58) THEN
C...gamma + gamma -> f + fbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=21
 
        ELSEIF(ISUB.EQ.59) THEN
C...gamma + Z0 -> f + fbar
 
        ELSEIF(ISUB.EQ.60) THEN
C...gamma + W+/- -> f + fbar'
        ENDIF
 
      ELSEIF(ISUB.LE.70) THEN
        IF(ISUB.EQ.61) THEN
C...gamma + h0 -> f + fbar
 
        ELSEIF(ISUB.EQ.62) THEN
C...Z0 + Z0 -> f + fbar
 
        ELSEIF(ISUB.EQ.63) THEN
C...Z0 + W+/- -> f + fbar'
 
        ELSEIF(ISUB.EQ.64) THEN
C...Z0 + h0 -> f + fbar
 
        ELSEIF(ISUB.EQ.65) THEN
C...W+ + W- -> f + fbar
 
        ELSEIF(ISUB.EQ.66) THEN
C...W+/- + h0 -> f + fbar'
 
        ELSEIF(ISUB.EQ.67) THEN
C...h0 + h0 -> f + fbar
 
        ELSEIF(ISUB.EQ.68) THEN
C...g + g -> g + g; th arbitrary
          KCC=MINT(2)+12
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.EQ.69) THEN
C...gamma + gamma -> W+ + W-; th arbitrary
          MINT(21)=24
          MINT(22)=-24
          KCC=21
 
        ELSEIF(ISUB.EQ.70) THEN
C...gamma + W+/- -> Z0 + W+/-; th=(p(W)-p(W))**2
          IF(MINT(15).EQ.22) MINT(21)=23
          IF(MINT(16).EQ.22) MINT(22)=23
          KCC=21
        ENDIF
 
      ELSEIF(ISUB.LE.80) THEN
        IF(ISUB.EQ.71.OR.ISUB.EQ.72) THEN
C...Z0 + Z0 -> Z0 + Z0; Z0 + Z0 -> W+ + W-
          XH=SH/SHP
          MINT(21)=MINT(15)
          MINT(22)=MINT(16)
          PMQ(1)=PYMASS(MINT(21))
          PMQ(2)=PYMASS(MINT(22))
  330     JT=INT(1.5D0+PYR(0))
          ZMIN=2D0*PMQ(JT)/SHPR
          ZMAX=1D0-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/
     &    (SHPR*(SHPR-PMQ(3-JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          Z(JT)=ZMIN+(ZMAX-ZMIN)*PYR(0)
          IF(-1D0+(1D0+XH)/(1D0-Z(JT))-XH/(1D0-Z(JT))**2.LT.
     &    (1D0-XH)**2/(4D0*XH)*PYR(0)) GOTO 330
          SQC1=1D0-4D0*PMQ(JT)**2/(Z(JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 330
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP)
          CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(JT)=MIN(1D0,MAX(-1D0,CTHE(JT)))
          Z(3-JT)=1D0-XH/(1D0-Z(JT))
          SQC1=1D0-4D0*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 330
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP)
          CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(3-JT)=MIN(1D0,MAX(-1D0,CTHE(3-JT)))
          PHIR=PARU(2)*PYR(0)
          CPHI=COS(PHIR)
          ANG=CTHE(1)*CTHE(2)-SQRT(1D0-CTHE(1)**2)*
     &    SQRT(1D0-CTHE(2)**2)*CPHI
          Z1=2D0-Z(JT)
          Z2=ANG*SQRT(Z(JT)**2-4D0*PMQ(JT)**2/SHP)
          Z3=1D0-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP
          Z(3-JT)=2D0/(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*
     &    PMQ(3-JT)**2/SHP))
          ZMIN=2D0*PMQ(3-JT)/SHPR
          ZMAX=1D0-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 330
          KCC=22
 
        ELSEIF(ISUB.EQ.73) THEN
C...Z0 + W+/- -> Z0 + W+/-
          JS=MINT(2)
          XH=SH/SHP
  340     JT=3-MINT(2)
          I=MINT(14+JT)
          IA=IABS(I)
          IF(IA.LE.10) THEN
            RVCKM=VINT(180+I)*PYR(0)
            DO 350 J=1,MSTP(1)
              IB=2*J-1+MOD(IA,2)
              IPM=(5-ISIGN(1,I))/2
              IDC=J+MDCY(IA,2)+2
              IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 350
              MINT(20+JT)=ISIGN(IB,I)
              RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
              IF(RVCKM.LE.0D0) GOTO 360
  350       CONTINUE
          ELSE
            IB=2*((IA+1)/2)-1+MOD(IA,2)
            MINT(20+JT)=ISIGN(IB,I)
          ENDIF
  360     PMQ(JT)=PYMASS(MINT(20+JT))
          MINT(23-JT)=MINT(17-JT)
          PMQ(3-JT)=PYMASS(MINT(23-JT))
          JT=INT(1.5D0+PYR(0))
          ZMIN=2D0*PMQ(JT)/SHPR
          ZMAX=1D0-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/
     &    (SHPR*(SHPR-PMQ(3-JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(ZMIN.GE.ZMAX) GOTO 340
          Z(JT)=ZMIN+(ZMAX-ZMIN)*PYR(0)
          IF(-1D0+(1D0+XH)/(1D0-Z(JT))-XH/(1D0-Z(JT))**2.LT.
     &    (1D0-XH)**2/(4D0*XH)*PYR(0)) GOTO 340
          SQC1=1D0-4D0*PMQ(JT)**2/(Z(JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 340
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP)
          CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(JT)=MIN(1D0,MAX(-1D0,CTHE(JT)))
          Z(3-JT)=1D0-XH/(1D0-Z(JT))
          SQC1=1D0-4D0*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 340
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP)
          CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(3-JT)=MIN(1D0,MAX(-1D0,CTHE(3-JT)))
          PHIR=PARU(2)*PYR(0)
          CPHI=COS(PHIR)
          ANG=CTHE(1)*CTHE(2)-SQRT(1D0-CTHE(1)**2)*
     &    SQRT(1D0-CTHE(2)**2)*CPHI
          Z1=2D0-Z(JT)
          Z2=ANG*SQRT(Z(JT)**2-4D0*PMQ(JT)**2/SHP)
          Z3=1D0-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP
          Z(3-JT)=2D0/(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*
     &    PMQ(3-JT)**2/SHP))
          ZMIN=2D0*PMQ(3-JT)/SHPR
          ZMAX=1D0-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 340
          KCC=22
 
        ELSEIF(ISUB.EQ.74) THEN
C...Z0 + h0 -> Z0 + h0
 
        ELSEIF(ISUB.EQ.75) THEN
C...W+ + W- -> gamma + gamma
 
        ELSEIF(ISUB.EQ.76.OR.ISUB.EQ.77) THEN
C...W+ + W- -> Z0 + Z0; W+ + W- -> W+ + W-
          XH=SH/SHP
  370     DO 400 JT=1,2
            I=MINT(14+JT)
            IA=IABS(I)
            IF(IA.LE.10) THEN
              RVCKM=VINT(180+I)*PYR(0)
              DO 380 J=1,MSTP(1)
                IB=2*J-1+MOD(IA,2)
                IPM=(5-ISIGN(1,I))/2
                IDC=J+MDCY(IA,2)+2
                IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 380
                MINT(20+JT)=ISIGN(IB,I)
                RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
                IF(RVCKM.LE.0D0) GOTO 390
  380         CONTINUE
            ELSE
              IB=2*((IA+1)/2)-1+MOD(IA,2)
              MINT(20+JT)=ISIGN(IB,I)
            ENDIF
  390       PMQ(JT)=PYMASS(MINT(20+JT))
  400     CONTINUE
          JT=INT(1.5D0+PYR(0))
          ZMIN=2D0*PMQ(JT)/SHPR
          ZMAX=1D0-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/
     &    (SHPR*(SHPR-PMQ(3-JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(ZMIN.GE.ZMAX) GOTO 370
          Z(JT)=ZMIN+(ZMAX-ZMIN)*PYR(0)
          IF(-1D0+(1D0+XH)/(1D0-Z(JT))-XH/(1D0-Z(JT))**2.LT.
     &    (1D0-XH)**2/(4D0*XH)*PYR(0)) GOTO 370
          SQC1=1D0-4D0*PMQ(JT)**2/(Z(JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 370
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(24,1)**2-PMQ(JT)**2)/(Z(JT)*SHP)
          CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(JT)=MIN(1D0,MAX(-1D0,CTHE(JT)))
          Z(3-JT)=1D0-XH/(1D0-Z(JT))
          SQC1=1D0-4D0*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)
          IF(SQC1.LT.1D-8) GOTO 370
          C1=SQRT(SQC1)
          C2=1D0+2D0*(PMAS(24,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP)
          CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2D0*PYR(0)-1D0)*C1))/C1
          CTHE(3-JT)=MIN(1D0,MAX(-1D0,CTHE(3-JT)))
          PHIR=PARU(2)*PYR(0)
          CPHI=COS(PHIR)
          ANG=CTHE(1)*CTHE(2)-SQRT(1D0-CTHE(1)**2)*
     &    SQRT(1D0-CTHE(2)**2)*CPHI
          Z1=2D0-Z(JT)
          Z2=ANG*SQRT(Z(JT)**2-4D0*PMQ(JT)**2/SHP)
          Z3=1D0-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP
          Z(3-JT)=2D0/(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*
     &    PMQ(3-JT)**2/SHP))
          ZMIN=2D0*PMQ(3-JT)/SHPR
          ZMAX=1D0-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))
          ZMAX=MIN(1D0-XH,ZMAX)
          IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 370
          KCC=22
 
        ELSEIF(ISUB.EQ.78) THEN
C...W+/- + h0 -> W+/- + h0
 
        ELSEIF(ISUB.EQ.79) THEN
C...h0 + h0 -> h0 + h0
 
        ELSEIF(ISUB.EQ.80) THEN
C...q + gamma -> q' + pi+/-; th=(p(q)-p(q'))**2
          IF(MINT(15).EQ.22) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(211,KCHG(IA,1)*I)
          IB=3-IA
          MINT(20+JS)=ISIGN(IB,I)
          KCC=22
        ENDIF
 
      ELSEIF(ISUB.LE.90) THEN
        IF(ISUB.EQ.81) THEN
C...q + qbar -> Q + Qbar; th = (p(q)-p(Q))**2
          MINT(21)=ISIGN(MINT(55),MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
 
        ELSEIF(ISUB.EQ.82) THEN
C...g + g -> Q + Qbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(MINT(55),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.83) THEN
C...f + q -> f' + Q; th = (p(f) - p(f'))**2
          KFOLD=MINT(16)
          IF(MINT(2).EQ.2) KFOLD=MINT(15)
          KFAOLD=IABS(KFOLD)
          IF(KFAOLD.GT.10) THEN
            KFANEW=KFAOLD+2*MOD(KFAOLD,2)-1
          ELSE
            RCKM=VINT(180+KFOLD)*PYR(0)
            IPM=(5-ISIGN(1,KFOLD))/2
            KFANEW=-MOD(KFAOLD+1,2)
  410       KFANEW=KFANEW+2
            IDC=MDCY(KFAOLD,2)+(KFANEW+1)/2+2
            IF(MDME(IDC,1).EQ.1.OR.MDME(IDC,1).EQ.IPM) THEN
              IF(MOD(KFAOLD,2).EQ.0) RCKM=RCKM-
     &        VCKM(KFAOLD/2,(KFANEW+1)/2)
              IF(MOD(KFAOLD,2).EQ.1) RCKM=RCKM-
     &        VCKM(KFANEW/2,(KFAOLD+1)/2)
            ENDIF
            IF(KFANEW.LE.6.AND.RCKM.GT.0D0) GOTO 410
          ENDIF
          IF(MINT(2).EQ.1) THEN
            MINT(21)=ISIGN(MINT(55),MINT(15))
            MINT(22)=ISIGN(KFANEW,MINT(16))
          ELSE
            MINT(21)=ISIGN(KFANEW,MINT(15))
            MINT(22)=ISIGN(MINT(55),MINT(16))
            JS=2
          ENDIF
          KCC=22
 
        ELSEIF(ISUB.EQ.84) THEN
C...g + gamma -> Q + Qbar; th arbitary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(MINT(55),KCS)
          MINT(22)=-MINT(21)
          KCC=27
          IF(MINT(16).EQ.21) KCC=28
 
        ELSEIF(ISUB.EQ.85) THEN
C...gamma + gamma -> F + Fbar; th arbitary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(MINT(56),KCS)
          MINT(22)=-MINT(21)
          KCC=21
 
        ELSEIF(ISUB.GE.86.AND.ISUB.LE.89) THEN
C...g + g -> (J/Psi, chi_0c, chi_1c or chi_2c) + g
          MINT(21)=KFPR(ISUB,1)
          MINT(22)=KFPR(ISUB,2)
          KCC=24
          KCS=(-1)**INT(1.5D0+PYR(0))
        ENDIF
 
      ELSEIF(ISUB.LE.100) THEN
        IF(ISUB.EQ.95) THEN
C...Low-pT ( = energyless g + g -> g + g)
          KCC=MINT(2)+12
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.EQ.96) THEN
C...Multiple interactions (should be reassigned to QCD process)
        ENDIF
 
      ELSEIF(ISUB.LE.110) THEN
        IF(ISUB.EQ.101) THEN
C...g + g -> gamma*/Z0
          KCC=21
          KFRES=22
 
        ELSEIF(ISUB.EQ.102) THEN
C...g + g -> h0 (or H0, or A0)
          KCC=21
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.103) THEN
C...gamma + gamma -> h0 (or H0, or A0)
          KCC=21
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.104.OR.ISUB.EQ.105) THEN
C...g + g -> chi_0c or chi_2c.
          KCC=21
          KFRES=KFPR(ISUB,1)
 
        ELSEIF(ISUB.EQ.106) THEN
C...g + g -> J/Psi + gamma
          MINT(21)=KFPR(ISUB,1)
          MINT(22)=KFPR(ISUB,2)
          KCC=21
 
        ELSEIF(ISUB.EQ.107) THEN
C...g + gamma -> J/Psi + g
          MINT(21)=KFPR(ISUB,1)
          MINT(22)=KFPR(ISUB,2)
          KCC=22
          IF(MINT(16).EQ.22) KCC=33
 
        ELSEIF(ISUB.EQ.108) THEN
C...gamma + gamma -> J/Psi + gamma
          MINT(21)=KFPR(ISUB,1)
          MINT(22)=KFPR(ISUB,2)
 
        ELSEIF(ISUB.EQ.110) THEN
C...f + fbar -> gamma + h0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=22
          MINT(23-JS)=KFHIGG
        ENDIF
 
      ELSEIF(ISUB.LE.120) THEN
        IF(ISUB.EQ.111) THEN
C...f + fbar -> g + h0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=KFHIGG
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.112) THEN
C...f + g -> f + h0; th = (p(f) - p(f))**2
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=KFHIGG
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.113) THEN
C...g + g -> g + h0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(23-JS)=KFHIGG
          KCC=22+JS
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.EQ.114) THEN
C...g + g -> gamma + gamma; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(21)=22
          MINT(22)=22
          KCC=21
 
        ELSEIF(ISUB.EQ.115) THEN
C...g + g -> g + gamma; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(23-JS)=22
          KCC=22+JS
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.EQ.116) THEN
C...g + g -> gamma + Z0
 
        ELSEIF(ISUB.EQ.117) THEN
C...g + g -> Z0 + Z0
 
        ELSEIF(ISUB.EQ.118) THEN
C...g + g -> W+ + W-
        ENDIF
 
      ELSEIF(ISUB.LE.140) THEN
        IF(ISUB.EQ.121) THEN
C...g + g -> Q + Qbar + h0
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUBSV,2),KCS)
          MINT(22)=-MINT(21)
          KCC=11+INT(0.5D0+PYR(0))
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.122) THEN
C...q + qbar -> Q + Qbar + h0
          MINT(21)=ISIGN(KFPR(ISUBSV,2),MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.123) THEN
C...f + f' -> f + f' + h0 (or H0, or A0) (Z0 + Z0 -> h0 as
C...inner process)
          KCC=22
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.124) THEN
C...f + f' -> f" + f"' + h0 (or H0, or A) (W+ + W- -> h0 as
C...inner process)
          DO 430 JT=1,2
            I=MINT(14+JT)
            IA=IABS(I)
            IF(IA.LE.10) THEN
              RVCKM=VINT(180+I)*PYR(0)
              DO 420 J=1,MSTP(1)
                IB=2*J-1+MOD(IA,2)
                IPM=(5-ISIGN(1,I))/2
                IDC=J+MDCY(IA,2)+2
                IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 420
                MINT(20+JT)=ISIGN(IB,I)
                RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
                IF(RVCKM.LE.0D0) GOTO 430
  420         CONTINUE
            ELSE
              IB=2*((IA+1)/2)-1+MOD(IA,2)
              MINT(20+JT)=ISIGN(IB,I)
            ENDIF
  430     CONTINUE
          KCC=22
          KFRES=KFHIGG
 
        ELSEIF(ISUB.EQ.131.OR.ISUB.EQ.132) THEN
C...f + gamma*_(T,L) -> f + g; th=(p(f)-p(f))**2
          IF(MINT(15).EQ.22) JS=2
          MINT(23-JS)=21
          KCC=24+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.133.OR.ISUB.EQ.134) THEN
C...f + gamma*_(T,L) -> f + gamma; th=(p(f)-p(f))**2
          IF(MINT(15).EQ.22) JS=2
          KCC=22
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.135.OR.ISUB.EQ.136) THEN
C...g + gamma*_(T,L) -> f + fbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=27
          IF(MINT(16).EQ.21) KCC=28
 
        ELSEIF(ISUB.GE.137.AND.ISUB.LE.140) THEN
C...gamma*_(T,L) + gamma*_(T,L) -> f + fbar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=21
 
        ENDIF
 
      ELSEIF(ISUB.LE.160) THEN
        IF(ISUB.EQ.141) THEN
C...f + fbar -> gamma*/Z0/Z'0
          KFRES=32
 
        ELSEIF(ISUB.EQ.142) THEN
C...f + fbar' -> W'+/-
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(34,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.143) THEN
C...f + fbar' -> H+/-
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(37,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.144) THEN
C...f + fbar' -> R
          KFRES=ISIGN(41,MINT(15)+MINT(16))
 
        ELSEIF(ISUB.EQ.145) THEN
C...q + l -> LQ (leptoquark)
          IF(IABS(MINT(16)).LE.8) JS=2
          KFRES=ISIGN(42,MINT(14+JS))
          KCC=28+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.146) THEN
C...e + gamma -> e* (excited lepton)
          IF(MINT(15).EQ.22) JS=2
          KFRES=ISIGN(KFPR(ISUB,1),MINT(14+JS))
          KCC=22
 
        ELSEIF(ISUB.EQ.147.OR.ISUB.EQ.148) THEN
C...q + g -> q* (excited quark)
          IF(MINT(15).EQ.21) JS=2
          KFRES=ISIGN(KFPR(ISUB,1),MINT(14+JS))
          KCC=30+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.149) THEN
C...g + g -> eta_tc
          KFRES=KTECHN+331
          KCC=23
          KCS=(-1)**INT(1.5D0+PYR(0))
        ENDIF
 
      ELSEIF(ISUB.LE.200) THEN
        IF(ISUB.EQ.161) THEN
C...f + g -> f' + H+/-; th = (p(f)-p(f'))**2
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(37,KCHG(IA,1)*I)
          IB=IA+MOD(IA,2)-MOD(IA+1,2)
          MINT(20+JS)=ISIGN(IB,I)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.162) THEN
C...q + g -> LQ + lbar; LQ=leptoquark; th=(p(q)-p(LQ))^2
          IF(MINT(15).EQ.21) JS=2
          MINT(20+JS)=ISIGN(42,MINT(14+JS))
          KFLQL=KFDP(MDCY(42,2),2)
          MINT(23-JS)=-ISIGN(KFLQL,MINT(14+JS))
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.163) THEN
C...g + g -> LQ + LQbar; LQ=leptoquark; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(42,KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.164) THEN
C...q + qbar -> LQ + LQbar; LQ=leptoquark; th=(p(q)-p(LQ))**2
          MINT(21)=ISIGN(42,MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
 
        ELSEIF(ISUB.EQ.165) THEN
C...q + qbar -> l- + l+; th=(p(q)-p(l-))**2
          MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.166) THEN
C...q + qbar' -> l + nu; th=(p(u)-p(nu))**2 or (p(ubar)-p(nubar))**2
          IF(MOD(MINT(15),2).EQ.0) THEN
            MINT(21)=ISIGN(KFPR(ISUB,1)+1,MINT(15))
            MINT(22)=ISIGN(KFPR(ISUB,1),MINT(16))
          ELSE
            MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
            MINT(22)=ISIGN(KFPR(ISUB,1)+1,MINT(16))
          ENDIF
 
        ELSEIF(ISUB.EQ.167.OR.ISUB.EQ.168) THEN
C...q + q' -> q" + q* (excited quark)
          KFQSTR=KFPR(ISUB,2)
          KFQEXC=MOD(KFQSTR,KEXCIT)
          JS=MINT(2)
          MINT(20+JS)=ISIGN(KFQSTR,MINT(14+JS))
          IF(IABS(MINT(15)).NE.KFQEXC.AND.IABS(MINT(16)).NE.KFQEXC)
     &    MINT(23-JS)=ISIGN(KFQEXC,MINT(17-JS))
          KCC=22
          JS=3-JS
 
        ELSEIF(ISUB.EQ.169) THEN
C...q + qbar -> e + e* (excited lepton)
          KFQSTR=KFPR(ISUB,2)
          KFQEXC=MOD(KFQSTR,KEXCIT)
          JS=MINT(2)
          MINT(20+JS)=ISIGN(KFQSTR,MINT(14+JS))
          MINT(23-JS)=ISIGN(KFQEXC,MINT(17-JS))
          JS=3-JS
 
        ELSEIF(ISUB.EQ.191) THEN
C...f + fbar -> rho_tc0.
          KFRES=KTECHN+113
 
        ELSEIF(ISUB.EQ.192) THEN
C...f + fbar' -> rho_tc+/-
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(KTECHN+213,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.193) THEN
C...f + fbar -> omega_tc0.
          KFRES=KTECHN+223
 
        ELSEIF(ISUB.EQ.194) THEN
C...f + fbar -> f' + fbar' via mixture of s-channel
C...rho_tc and omega_tc; th=(p(f)-p(f'))**2
          MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.195) THEN
C...f + fbar' -> f'' + fbar''' via s-channel
C...rho_tc+ th=(p(f)-p(f'))**2
C...q + qbar' -> l + nu; th=(p(u)-p(nu))**2 or (p(ubar)-p(nubar))**2
          IF(MOD(MINT(15),2).EQ.0) THEN
            MINT(21)=ISIGN(KFPR(ISUB,1)+1,MINT(15))
            MINT(22)=ISIGN(KFPR(ISUB,1),MINT(16))
          ELSE
            MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
            MINT(22)=ISIGN(KFPR(ISUB,1)+1,MINT(16))
          ENDIF
        ENDIF
 
CMRENNA++
      ELSEIF(ISUB.LE.215) THEN
        IF(ISUB.EQ.201) THEN
C...f + fbar -> ~e_L + ~e_Lbar
          MINT(21)=ISIGN(KSUSY1+11,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.202) THEN
C...f + fbar -> ~e_R + ~e_Rbar
          MINT(21)=ISIGN(KSUSY2+11,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.203) THEN
C...f + fbar -> ~e_L + ~e_Rbar
          IF(MINT(15).LT.0) JS=2
          IF(MINT(2).EQ.1) THEN
            MINT(20+JS)=KFPR(ISUB,1)
            MINT(23-JS)=-KFPR(ISUB,2)
          ELSE
            MINT(20+JS)=-KFPR(ISUB,1)
            MINT(23-JS)=KFPR(ISUB,2)
          ENDIF
 
        ELSEIF(ISUB.EQ.204) THEN
C...f + fbar -> ~mu_L + ~mu_Lbar
          MINT(21)=ISIGN(KSUSY1+13,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.205) THEN
C...f + fbar -> ~mu_R + ~mu_Rbar
          MINT(21)=ISIGN(KSUSY2+13,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.206) THEN
C...f + fbar -> ~mu_L + ~mu_Rbar
          IF(MINT(15).LT.0) JS=2
          IF(MINT(2).EQ.1) THEN
            MINT(20+JS)=KFPR(ISUB,1)
            MINT(23-JS)=-KFPR(ISUB,2)
          ELSE
            MINT(20+JS)=-KFPR(ISUB,1)
            MINT(23-JS)=KFPR(ISUB,2)
          ENDIF
 
        ELSEIF(ISUB.EQ.207) THEN
C...f + fbar -> ~tau_1 + ~tau_1bar
          MINT(21)=ISIGN(KSUSY1+15,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.208) THEN
C...f + fbar -> ~tau_2 + ~tau_2bar
          MINT(21)=ISIGN(KSUSY2+15,KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.209) THEN
C...f + fbar -> ~tau_1 + ~tau_2bar
          IF(MINT(15).LT.0) JS=2
          IF(MINT(2).EQ.1) THEN
            MINT(20+JS)=KFPR(ISUB,1)
            MINT(23-JS)=-KFPR(ISUB,2)
          ELSE
            MINT(20+JS)=-KFPR(ISUB,1)
            MINT(23-JS)=KFPR(ISUB,2)
          ENDIF
 
        ELSEIF(ISUB.EQ.210) THEN
C...q + qbar' -> ~l_L + ~nulbar; th arbitrary
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          MINT(21)=-ISIGN(KFPR(ISUB,1),KCH1+KCH2)
          MINT(22)=ISIGN(KFPR(ISUB,2),KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.211) THEN
C...q + qbar'-> ~tau_1 + ~nutaubar; th arbitrary
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          MINT(21)=-ISIGN(KSUSY1+15,KCH1+KCH2)
          MINT(22)=ISIGN(KSUSY1+16,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.212) THEN
C...q + qbar'-> ~tau_2 + ~nutaubar; th arbitrary
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          MINT(21)=-ISIGN(KSUSY2+15,KCH1+KCH2)
          MINT(22)=ISIGN(KSUSY1+16,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.213) THEN
C...f + fbar -> ~nul + ~nulbar
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.214) THEN
C...f + fbar -> ~nutau + ~nutaubar
          MINT(21)=ISIGN(KSUSY1+16,KCS)
          MINT(22)=-MINT(21)
        ENDIF
 
      ELSEIF(ISUB.LE.225) THEN
        IF(ISUB.EQ.216) THEN
C...f + fbar -> ~chi01 + ~chi01
          MINT(21)=KSUSY1+22
          MINT(22)=KSUSY1+22
 
        ELSEIF(ISUB.EQ.217) THEN
C...f + fbar -> ~chi02 + ~chi02
          MINT(21)=KSUSY1+23
          MINT(22)=KSUSY1+23
 
        ELSEIF(ISUB.EQ.218 ) THEN
C...f + fbar -> ~chi03 + ~chi03
          MINT(21)=KSUSY1+25
          MINT(22)=KSUSY1+25
 
        ELSEIF(ISUB.EQ.219 ) THEN
C...f + fbar -> ~chi04 + ~chi04
          MINT(21)=KSUSY1+35
          MINT(22)=KSUSY1+35
 
        ELSEIF(ISUB.EQ.220 ) THEN
C...f + fbar -> ~chi01 + ~chi02
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+22
          MINT(23-JS)=KSUSY1+23
 
        ELSEIF(ISUB.EQ.221 ) THEN
C...f + fbar -> ~chi01 + ~chi03
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+22
          MINT(23-JS)=KSUSY1+25
 
        ELSEIF(ISUB.EQ.222) THEN
C...f + fbar -> ~chi01 + ~chi04
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+22
          MINT(23-JS)=KSUSY1+35
 
        ELSEIF(ISUB.EQ.223) THEN
C...f + fbar -> ~chi02 + ~chi03
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+23
          MINT(23-JS)=KSUSY1+25
 
        ELSEIF(ISUB.EQ.224) THEN
C...f + fbar -> ~chi02 + ~chi04
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+23
          MINT(23-JS)=KSUSY1+35
 
        ELSEIF(ISUB.EQ.225) THEN
C...f + fbar -> ~chi03 + ~chi04
          IF(MINT(15).LT.0) JS=2
C          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+25
          MINT(23-JS)=KSUSY1+35
        ENDIF
 
      ELSEIF(ISUB.LE.236) THEN
        IF(ISUB.EQ.226) THEN
C...f + fbar -> ~chi+-1 + ~chi-+1
C...th=(p(q)-p(chi+))**2 or (p(qbar)-p(chi-))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          MINT(21)=ISIGN(KSUSY1+24,KCH1)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.227) THEN
C...f + fbar -> ~chi+-2 + ~chi-+2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          MINT(21)=ISIGN(KSUSY1+37,KCH1)
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.228) THEN
C...f + fbar -> ~chi+-1 + ~chi-+2
C...th=(p(q)-p(chi1+))**2 or th=(p(qbar)-p(chi1-))**2
C...js=1 if pyr<.5, js=2 if pyr>.5
C...if 15=q, 16=qbar and js=1, chi1+ + chi2-, th=(q-chi1+)**2
C...if 15=qbar, 16=q and js=1, chi2- + chi1+, th=(q-chi1+)**2
C...if 15=q, 16=qbar and js=2, chi1- + chi2+, th=(qbar-chi1-)**2
C...if 15=qbar, 16=q and js=2, chi2+ + chi1-, th=(q-chi1-)**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=INT(1-KCH1)/2
          IF(MINT(2).EQ.1) THEN
            MINT(21)= ISIGN(KSUSY1+24,KCH1)
            MINT(22)= -ISIGN(KSUSY1+37,KCH1)
c            IF(KCH2.EQ.0) JS=2
          ELSE
            MINT(21)= ISIGN(KSUSY1+37,KCH1)
            MINT(22)= -ISIGN(KSUSY1+24,KCH1)
            JS=2
c            IF(KCH2.EQ.1) JS=2
          ENDIF
 
        ELSEIF(ISUB.EQ.229) THEN
C...q + qbar' -> ~chi01 + ~chi+-1
C...th=(p(u)-p(chi+))**2 or (p(ubar)-p(chi-))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
C...CHECK THIS
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+22
          MINT(23-JS)=ISIGN(KSUSY1+24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.230) THEN
C...q + qbar' -> ~chi02 + ~chi+-1
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+23
          MINT(23-JS)=ISIGN(KSUSY1+24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.231) THEN
C...q + qbar' -> ~chi03 + ~chi+-1
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+25
          MINT(23-JS)=ISIGN(KSUSY1+24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.232) THEN
C...q + qbar' -> ~chi04 + ~chi+-1
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+35
          MINT(23-JS)=ISIGN(KSUSY1+24,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.233) THEN
C...q + qbar' -> ~chi01 + ~chi+-2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+22
          MINT(23-JS)=ISIGN(KSUSY1+37,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.234) THEN
C...q + qbar' -> ~chi02 + ~chi+-2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+23
          MINT(23-JS)=ISIGN(KSUSY1+37,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.235) THEN
C...q + qbar' -> ~chi03 + ~chi+-2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+25
          MINT(23-JS)=ISIGN(KSUSY1+37,KCH1+KCH2)
 
        ELSEIF(ISUB.EQ.236) THEN
C...q + qbar' -> ~chi04 + ~chi+-2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MOD(MINT(15),2).EQ.0) JS=2
          MINT(20+JS)=KSUSY1+35
          MINT(23-JS)=ISIGN(KSUSY1+37,KCH1+KCH2)
        ENDIF
 
      ELSEIF(ISUB.LE.245) THEN
        IF(ISUB.EQ.237) THEN
C...q + qbar -> ~chi01 + ~g
C...th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=KSUSY1+22
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.238) THEN
C...q + qbar -> ~chi02 + ~g
C...th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=KSUSY1+23
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.239) THEN
C...q + qbar -> ~chi03 + ~g
C...th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=KSUSY1+25
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.240) THEN
C...q + qbar -> ~chi04 + ~g
C...th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=KSUSY1+35
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.241) THEN
C...q + qbar' -> ~chi+-1 + ~g
C...if 15=u, 16=dbar, then (kch1+kch2)>0, js=1, chi+
C...if 15=d, 16=ubar, then (kch1+kch2)<0, js=2, chi-
C...if 15=ubar, 16=d, then (kch1+kch2)<0, js=1, chi-
C...if 15=dbar, 16=u, then (kch1+kch2)>0, js=2, chi+
C...th=(p(q)-p(chi+))**2 or (p(qbar')-p(chi-))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          JS=1
          IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=ISIGN(KSUSY1+24,KCH1+KCH2)
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.242) THEN
C...q + qbar' -> ~chi+-2 + ~g
C...if 15=u, 16=dbar, then (kch1+kch2)>0, js=1, chi+
C...if 15=d, 16=ubar, then (kch1+kch2)<0, js=2, chi-
C...if 15=ubar, 16=d, then (kch1+kch2)<0, js=1, chi-
C...if 15=dbar, 16=u, then (kch1+kch2)>0, js=2, chi+
C...th=(p(q)-p(chi+))**2 or (p(qbar')-p(chi-))**2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          JS=1
          IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2
          MINT(20+JS)=KSUSY1+21
          MINT(23-JS)=ISIGN(KSUSY1+37,KCH1+KCH2)
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.243) THEN
C...q + qbar -> ~g + ~g ; th arbitrary
          MINT(21)=KSUSY1+21
          MINT(22)=KSUSY1+21
          KCC=MINT(2)+4
 
        ELSEIF(ISUB.EQ.244) THEN
C...g + g -> ~g + ~g ; th arbitrary
          KCC=MINT(2)+12
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=KSUSY1+21
          MINT(22)=KSUSY1+21
        ENDIF
 
      ELSEIF(ISUB.LE.260) THEN
        IF(ISUB.EQ.246) THEN
C...qj + g -> ~qj_L + ~chi01
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+22
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.247) THEN
C...qj + g -> ~qj_R + ~chi01
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+22
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.248) THEN
C...qj + g -> ~qj_L + ~chi02
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+23
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.249) THEN
C...qj + g -> ~qj_R + ~chi02
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+23
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.250) THEN
C...qj + g -> ~qj_L + ~chi03
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+25
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.251) THEN
C...qj + g -> ~qj_R + ~chi03
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+25
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.252) THEN
C...qj + g -> ~qj_L + ~chi04
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+35
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.253) THEN
C...qj + g -> ~qj_R + ~chi04
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+35
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.254) THEN
C...qj + g -> ~qk_L + ~chi+-1
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(KSUSY1+24,KCHG(IA,1)*I)
          IB=-IA+INT((IA+1)/2)*4-1
          MINT(20+JS)=ISIGN(KSUSY1+IB,I)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.255) THEN
C...qj + g -> ~qk_L + ~chi+-1
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(23-JS)=ISIGN(KSUSY1+24,KCHG(IA,1)*I)
          IB=-IA+INT((IA+1)/2)*4-1
          MINT(20+JS)=ISIGN(KSUSY2+IB,I)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.256) THEN
C...qj + g -> ~qk_L + ~chi+-2
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          IB=-IA+INT((IA+1)/2)*4-1
          MINT(20+JS)=ISIGN(KSUSY1+IB,I)
          MINT(23-JS)=ISIGN(KSUSY1+37,KCHG(IA,1)*I)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.257) THEN
C...qj + g -> ~qk_R + ~chi+-2
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          IB=-IA+INT((IA+1)/2)*4-1
          MINT(20+JS)=ISIGN(KSUSY2+IB,I)
          MINT(23-JS)=ISIGN(KSUSY1+37,KCHG(IA,1)*I)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.258) THEN
C...qj + g -> ~qj_L + ~g
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+21
          KCC=MINT(2)+6
          IF(JS.EQ.2) KCC=KCC+2
          KCS=ISIGN(1,I)
 
        ELSEIF(ISUB.EQ.259) THEN
C...qj + g -> ~qj_R + ~g
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+21
          KCC=MINT(2)+6
          IF(JS.EQ.2) KCC=KCC+2
          KCS=ISIGN(1,I)
        ENDIF
 
      ELSEIF(ISUB.LE.270) THEN
        IF(ISUB.EQ.261) THEN
C...f + fbar -> ~t_1 + ~t_1bar; th = (p(q)-p(sq))**2
          ISGN=1
          IF(MINT(43).EQ.1.AND.PYR(0).GT.0.5D0) ISGN=-1
          MINT(21)=ISGN*ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
C...Correct color combination
          IF(MINT(43).EQ.4) KCC=4
 
        ELSEIF(ISUB.EQ.262) THEN
C...f + fbar -> ~t_2 + ~t_2bar; th = (p(q)-p(sq))**2
          ISGN=1
          IF(MINT(43).EQ.1.AND.PYR(0).GT.0.5D0) ISGN=-1
          MINT(21)=ISGN*ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
C...Correct color combination
          IF(MINT(43).EQ.4) KCC=4
 
        ELSEIF(ISUB.EQ.263) THEN
C...f + fbar -> ~t_1 + ~t_2bar; th = (p(q)-p(sq))**2
          IF((KCS.GT.0.AND.MINT(2).EQ.1).OR.
     &    (KCS.LT.0.AND.MINT(2).EQ.2)) THEN
            MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
            MINT(22)=-ISIGN(KFPR(ISUB,2),KCS)
          ELSE
            JS=2
            MINT(21)=ISIGN(KFPR(ISUB,2),KCS)
            MINT(22)=-ISIGN(KFPR(ISUB,1),KCS)
          ENDIF
C...Correct color combination
          IF(MINT(43).EQ.4) KCC=4
 
        ELSEIF(ISUB.EQ.264) THEN
C...g + g -> ~t_1 + ~t_1bar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.265) THEN
C...g + g -> ~t_2 + ~t_2bar; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
        ENDIF
 
      ELSEIF(ISUB.LE.296) THEN
        IF(ISUB.EQ.271.OR.ISUB.EQ.281.OR.ISUB.EQ.291) THEN
C...qi + qj -> ~qi_L + ~qj_L
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
          MINT(21)=ISIGN(KSUSY1+IABS(MINT(15)),MINT(15))
          MINT(22)=ISIGN(KSUSY1+IABS(MINT(16)),MINT(16))
 
        ELSEIF(ISUB.EQ.272.OR.ISUB.EQ.282.OR.ISUB.EQ.292) THEN
C...qi + qj -> ~qi_R + ~qj_R
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
          MINT(21)=ISIGN(KSUSY2+IABS(MINT(15)),MINT(15))
          MINT(22)=ISIGN(KSUSY2+IABS(MINT(16)),MINT(16))
 
        ELSEIF(ISUB.EQ.273.OR.ISUB.EQ.283.OR.ISUB.EQ.293) THEN
C...qi + qj -> ~qi_L + ~qj_R
          MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
          MINT(22)=ISIGN(KFPR(ISUB,2),MINT(16))
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.274.OR.ISUB.EQ.284) THEN
C...qi + qjbar -> ~qi_L + ~qj_Lbar; th = (p(f)-p(sf'))**2
          MINT(21)=ISIGN(KSUSY1+IABS(MINT(15)),MINT(15))
          MINT(22)=ISIGN(KSUSY1+IABS(MINT(16)),MINT(16))
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.275.OR.ISUB.EQ.285) THEN
C...qi + qjbar -> ~qi_R + ~qj_Rbar ; th = (p(f)-p(sf'))**2
          MINT(21)=ISIGN(KSUSY2+IABS(MINT(15)),MINT(15))
          MINT(22)=ISIGN(KSUSY2+IABS(MINT(16)),MINT(16))
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.276.OR.ISUB.EQ.286.OR.ISUB.EQ.296) THEN
C...qi + qjbar -> ~qi_L + ~qj_Rbar ; th = (p(f)-p(sf'))**2
          MINT(21)=ISIGN(KFPR(ISUB,1),MINT(15))
          MINT(22)=ISIGN(KFPR(ISUB,2),MINT(16))
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.277.OR.ISUB.EQ.287) THEN
C...f + fbar -> ~qi_L + ~qi_Lbar ; th = (p(q)-p(sq))**2
          ISGN=1
          IF(MINT(43).EQ.1.AND.PYR(0).GT.0.5D0) ISGN=-1
          MINT(21)=ISGN*ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          IF(MINT(43).EQ.4) KCC=4
 
        ELSEIF(ISUB.EQ.278.OR.ISUB.EQ.288) THEN
C...f + fbar -> ~qi_R + ~qi_Rbar; th = (p(q)-p(sq))**2
          ISGN=1
          IF(MINT(43).EQ.1.AND.PYR(0).GT.0.5D0) ISGN=-1
          MINT(21)=ISGN*ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          IF(MINT(43).EQ.4) KCC=4
 
        ELSEIF(ISUB.EQ.279.OR.ISUB.EQ.289) THEN
C...g + g -> ~qi_L + ~qi_Lbar ; th arbitrary
C...pure LL + RR
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.280.OR.ISUB.EQ.290) THEN
C...g + g -> ~qi_R + ~qi_Rbar ; th arbitrary
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.294) THEN
C...qj + g -> ~qj_L + ~g
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY1+IA,I)
          MINT(23-JS)=KSUSY1+21
          KCC=MINT(2)+6
          IF(JS.EQ.2) KCC=KCC+2
          KCS=ISIGN(1,I)
 
        ELSEIF(ISUB.EQ.295) THEN
C...qj + g -> ~qj_R + ~g
          IF(MINT(15).EQ.21) JS=2
          I=MINT(14+JS)
          IA=IABS(I)
          MINT(20+JS)=ISIGN(KSUSY2+IA,I)
          MINT(23-JS)=KSUSY1+21
          KCC=MINT(2)+6
          IF(JS.EQ.2) KCC=KCC+2
          KCS=ISIGN(1,I)
        ENDIF
 
      ELSEIF(ISUB.LE.330) THEN
        IF(ISUB.EQ.311)THEN
C...g + g -> g* + g* (UED)
          KCC=MINT(2)+12
          KCS=(-1)**INT(1.5D0+PYR(0))
          MUED(1)=472
          MUED(2)=472
          MINT(21)=IUEDEQ(472)
          MINT(22)=IUEDEQ(472)
        ELSEIF(ISUB.EQ.312)THEN
C...q + g -> q*_D + g*, q*_S + g*
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456
          IF(MINT(15).EQ.21) JS=2
          KCC=MINT(2)+6
          IF(MINT(15).EQ.21)KCC=KCC+2
          IF(MINT(15).NE.21)THEN
            KCS=ISIGN(1,MINT(15))
            MUED(2)=472
            MUED(1)=KCS*(KKFLMI+IABS(MINT(15)))
            MINT(22)=IUEDEQ(472)
            MINT(21)=KCS*IUEDEQ(KKFLMI+IABS(MINT(15)))
          ENDIF
          IF(MINT(16).NE.21)THEN
            KCS=ISIGN(1,MINT(16))
            MUED(2)=KCS*(KKFLMI+IABS(MINT(16)))
            MUED(1)=472
            MINT(22)=KCS*IUEDEQ(KKFLMI+IABS(MINT(16)))
            MINT(21)=IUEDEQ(472)
          ENDIF
        ELSEIF(ISUB.EQ.313)THEN
C...q + q' -> q*_D + q*_D',q*_S+q*_S'
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456
          KCC=MINT(2)         
          IF(MINT(15).EQ.MINT(16))THEN
            MUED(1)=SIGN(1,MINT(15))*(KKFLMI+IABS(MINT(15)))
            MUED(2)=MINT(21)
            MINT(21)=SIGN(1,MINT(15))*IUEDEQ(KKFLMI+IABS(MINT(15)))
            MINT(22)=MINT(21)
          ELSE
            MUED(1)=SIGN(1,MINT(15))*(KKFLMI+IABS(MINT(15)))
            MUED(2)=SIGN(1,MINT(16))*(KKFLMI+IABS(MINT(16)))
            MINT(21)=SIGN(1,MINT(15))*IUEDEQ(KKFLMI+IABS(MINT(15)))
            MINT(22)=SIGN(1,MINT(16))*IUEDEQ(KKFLMI+IABS(MINT(16)))
          ENDIF
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2        
        ELSEIF(ISUB.EQ.314)THEN
C...g + g -> q*_D + q*_D_bar, q*_S + q*_S_bar
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456
          KCS=(-1)**INT(1.5D0+PYR(0))    
          XFLAOUT=PYR(0)
          IF(XFLAOUT.LE.0.2)THEN
            MUED(1)=ISIGN(1,KCS)*(KKFLMI+1)
            MINT(21)=ISIGN(1,KCS)*IUEDEQ(KKFLMI+1)
          ELSEIF(XFLAOUT.LE.0.4)THEN
            MUED(1)=ISIGN(1,KCS)*(KKFLMI+2)
            MINT(21)=ISIGN(1,KCS)*IUEDEQ(KKFLMI+2)
          ELSEIF(XFLAOUT.LE.0.6)THEN
            MUED(1)=ISIGN(1,KCS)*(KKFLMI+3)
            MINT(21)=ISIGN(1,KCS)*IUEDEQ(KKFLMI+3)
          ELSEIF(XFLAOUT.LE.0.8)THEN
            MUED(1)=ISIGN(1,KCS)*(KKFLMI+4)
            MINT(21)=ISIGN(1,KCS)*IUEDEQ(KKFLMI+4)
          ELSE
            MUED(1)=ISIGN(1,KCS)*(KKFLMI+5)
            MINT(21)=ISIGN(1,KCS)*IUEDEQ(KKFLMI+5)
          ENDIF
          MINT(22)=-MINT(21)
          MUED(2)=-MUED(1)
          KCC=MINT(2)+10
        ELSEIF(ISUB.EQ.315)THEN
C...q + qbar -> q*_D + q*_D_bar, q*_S + q*_S_bar
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456
          MUED(1)=ISIGN(1,MINT(15))*(KKFLMI+IABS(MINT(15)))
          MUED(2)=-MINT(21)
          MINT(21)=ISIGN(1,MINT(15))*IUEDEQ(KKFLMI+IABS(MINT(15)))
          MINT(22)=-MINT(21)
          KCC=4
        ELSEIF(ISUB.EQ.316)THEN
C...q + qbar'    -> q*_D + q*_S_bar'
          MUED(1)=ISIGN(1,MINT(15))*(456+IABS(MINT(15)))
          MUED(2)=ISIGN(1,MINT(16))*(450+IABS(MINT(16)))
          MINT(21)=ISIGN(1,MINT(15))*IUEDEQ(456+IABS(MINT(15)))
          MINT(22)=ISIGN(1,MINT(16))*IUEDEQ(450+IABS(MINT(16)))
          KCC=MINT(2)+2
        ELSEIF(ISUB.EQ.317)THEN
C...q + qbar'    -> q*_D + q*_D_bar', q*_S + q*_S_bar
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456      
          MUED(1)=ISIGN(1,MINT(15))*(KKFLMI+IABS(MINT(15)))
          MUED(2)=ISIGN(1,MINT(16))*(KKFLMI+IABS(MINT(16)))
          MINT(21)=ISIGN(1,MINT(15))*IUEDEQ(KKFLMI+IABS(MINT(15)))
          MINT(22)=ISIGN(1,MINT(16))*IUEDEQ(KKFLMI+IABS(MINT(16)))
          KCC=MINT(2)+2
        ELSEIF(ISUB.EQ.318)THEN
C...q + q'    -> q*_D + q*_S'     
          KCC=MINT(2)         
          MUED(1)=SIGN(1,MINT(15))*(456+IABS(MINT(15)))
          MUED(2)=SIGN(1,MINT(16))*(450+IABS(MINT(16)))               
          MINT(21)=SIGN(1,MINT(15))*IUEDEQ(456+IABS(MINT(15)))
          MINT(22)=SIGN(1,MINT(16))*IUEDEQ(450+IABS(MINT(16)))
        ELSEIF(ISUB.EQ.319)THEN
C...q + qbar -> q*_D' + q*_D_bar', q*_S' + q*_S_bar'
C...The two channels have the same cross section
          KKFLMI=450
          IF(PYR(0).GT.0.5)KKFLMI=456
          XFLAOUT=PYR(0)
          IIFLAV=0
C...N.B. NFLAVOURS=IUED(3)
C   DO I=1,NFLAVOURS
          DO 433 I=1,IUED(3)
            IF(I.NE.IABS(MINT(15)))THEN
              IIFLAV=IIFLAV+1
              IOKFLA(IIFLAV)=I
            ENDIF
 433      CONTINUE
          FLASTEP=1./(IUED(3)-1)
          DO I=1,IUED(3)-1
            FLAVV=FLASTEP*I
            IF(XFLAOUT.LE.FLAVV)THEN                  
              MUED(1)=ISIGN(1,MINT(15))*(KKFLMI+IOKFLA(I))
              MINT(21)=ISIGN(1,MINT(15))*IUEDEQ(KKFLMI+IOKFLA(I))
              GOTO 435
            ENDIF
          ENDDO
 435      CONTINUE
          IF(IABS(MUED(1)).LT.451.AND.IABS(MUED(1)).GT.462)THEN
            WRITE(MSTU(11),*) 'IN PYSCAT: KK FLAVORS PROBLEM !!!'
            CALL PYSTOP(5000000)
          ENDIF
          MINT(22)=-MINT(21)
          KCC=4
        ENDIF
        
      ELSEIF(ISUB.LE.340) THEN
 
        IF(ISUB.EQ.297.OR.ISUB.EQ.298) THEN
C...q + qbar' -> H+ + H0
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2
          MINT(20+JS)=ISIGN(37,KCH1+KCH2)
          MINT(23-JS)=KFPR(ISUB,2)
        ELSEIF(ISUB.EQ.299.OR.ISUB.EQ.300) THEN
C...f + fbar -> A0 + H0; th arbitrary
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KFPR(ISUB,1)
          MINT(23-JS)=KFPR(ISUB,2)
        ELSEIF(ISUB.EQ.301) THEN
C...f + fbar -> H+ H-
          MINT(21)=ISIGN(KFPR(ISUB,1),KCS)
          MINT(22)=-MINT(21)
        ENDIF
CMRENNA--
 
      ELSEIF(ISUB.LE.360) THEN
 
        IF(ISUB.EQ.341.OR.ISUB.EQ.342) THEN
C...l + l -> H_L++/--, H_R++/--
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(KFPR(ISUB,1),KCH1+KCH2)
 
        ELSEIF(ISUB.GE.343.AND.ISUB.LE.348) THEN
C...l + gamma -> l' + H++/--; th=(p(l)-p(H))**2
          IF(MINT(15).EQ.22) JS=2
          MINT(20+JS)=ISIGN(KFPR(ISUB,1),-MINT(14+JS))
          MINT(23-JS)=ISIGN(KFPR(ISUB,2),-MINT(14+JS))
          KCC=22
 
        ELSEIF(ISUB.EQ.349.OR.ISUB.EQ.350) THEN
C...f + fbar -> H++ + H--; th = (p(f)-p(H--))**2
          MINT(21)=-ISIGN(KFPR(ISUB,1),MINT(15))
          MINT(22)=-MINT(21)
 
        ELSEIF(ISUB.EQ.351.OR.ISUB.EQ.352) THEN
C...f + f' -> f" + f"' + H++/-- (W+/- + W+/- -> H++/--
C...as inner process).
          DO 450 JT=1,2
            I=MINT(14+JT)
            IA=IABS(I)
            IF(IA.LE.10) THEN
              RVCKM=VINT(180+I)*PYR(0)
              DO 440 J=1,MSTP(1)
                IB=2*J-1+MOD(IA,2)
                IPM=(5-ISIGN(1,I))/2
                IDC=J+MDCY(IA,2)+2
                IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 440
                MINT(20+JT)=ISIGN(IB,I)
                RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)
                IF(RVCKM.LE.0D0) GOTO 450
  440         CONTINUE
            ELSE
              IB=2*((IA+1)/2)-1+MOD(IA,2)
              MINT(20+JT)=ISIGN(IB,I)
            ENDIF
  450     CONTINUE
          KCC=22
          KFRES=ISIGN(KFPR(ISUB,1),MINT(15))
          IF(MOD(MINT(15),2).EQ.1) KFRES=-KFRES
 
        ELSEIF(ISUB.EQ.353) THEN
C...f + fbar -> Z_R0
          KFRES=KFPR(ISUB,1)
 
        ELSEIF(ISUB.EQ.354) THEN
C...f + fbar' -> W+/-
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          KFRES=ISIGN(KFPR(ISUB,1),KCH1+KCH2)
 
        ENDIF
 
      ELSEIF(ISUB.LE.380) THEN
 
        IF(ISUB.LE.363.OR.ISUB.EQ.368) THEN
C...f + fbar -> charged+ charged- technicolor
          KSW=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUB,1),KSW)
          MINT(22)=-ISIGN(KFPR(ISUB,2),KSW)
 
        ELSEIF(ISUB.LE.367.OR.ISUB.EQ.379.OR.ISUB.EQ.380) THEN
C...f + fbar -> neutral neutral technicolor
          MINT(21)=KFPR(ISUB,1)
          MINT(22)=KFPR(ISUB,2)
 
        ELSEIF(ISUB.EQ.374.OR.ISUB.EQ.375.OR.ISUB.EQ.378) THEN
C...f + fbar' -> neutral charged technicolor
          IN=1
          IC=2
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2
          MINT(23-JS)=ISIGN(KFPR(ISUB,IC),KCH1+KCH2)
          MINT(20+JS)=KFPR(ISUB,IN)
 
        ELSEIF(ISUB.GE.370.AND.ISUB.LE.377) THEN
C...f + fbar' -> charged neutral technicolor
          IN=2
          IC=1
          KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))
          KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))
          IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2
          MINT(20+JS)=ISIGN(KFPR(ISUB,IC),KCH1+KCH2)
          MINT(23-JS)=KFPR(ISUB,IN)
        ENDIF
 
      ELSEIF(ISUB.LE.400) THEN
        IF(ISUB.EQ.381) THEN
C...f + f' -> f + f' (g exchange); th = (p(f)-p(f))**2, TC extensions
          KCC=MINT(2)
          IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2
 
        ELSEIF(ISUB.EQ.382) THEN
C...f + fbar -> f' + fbar'; th = (p(f)-p(f'))**2, TC extensions
          MINT(21)=ISIGN(KFLF,MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
 
        ELSEIF(ISUB.EQ.383) THEN
C...f + fbar -> g + g; th arbitrary, TC extensions
          MINT(21)=21
          MINT(22)=21
          KCC=MINT(2)+4
 
        ELSEIF(ISUB.EQ.384) THEN
C...f + g -> f + g; th = (p(f)-p(f))**2, TC extensions
          IF(MINT(15).EQ.21) JS=2
          KCC=MINT(2)+6
          IF(MINT(15).EQ.21) KCC=KCC+2
          IF(MINT(15).NE.21) KCS=ISIGN(1,MINT(15))
          IF(MINT(16).NE.21) KCS=ISIGN(1,MINT(16))
 
        ELSEIF(ISUB.EQ.385) THEN
C...g + g -> f + fbar; th arbitrary, TC extensions
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFLF,KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.386) THEN
C...g + g -> g + g; th arbitrary, TC extensions
          KCC=MINT(2)+12
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.EQ.387) THEN
C...q + qbar -> Q + Qbar; th = (p(q)-p(Q))**2, TC extensions
          MINT(21)=ISIGN(MINT(55),MINT(15))
          MINT(22)=-MINT(21)
          KCC=4
 
        ELSEIF(ISUB.EQ.388) THEN
C...g + g -> Q + Qbar; th arbitrary, TC extensions
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(MINT(55),KCS)
          MINT(22)=-MINT(21)
          KCC=MINT(2)+10
 
        ELSEIF(ISUB.EQ.391) THEN
C...f + fbar -> G*.
          KFRES=KFPR(ISUB,1)
 
        ELSEIF(ISUB.EQ.392) THEN
C...g + g -> G*.
          KCC=21
          KFRES=KFPR(ISUB,1)
 
        ELSEIF(ISUB.EQ.393) THEN
C...q + qbar -> g + G*;  th arbitrary.
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(20+JS)=KFPR(ISUB,1)
          MINT(23-JS)=KFPR(ISUB,2)
          KCC=17+JS
 
        ELSEIF(ISUB.EQ.394) THEN
C...q + g -> q + G*;  th = (p(f) - p(f))**2
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=KFPR(ISUB,2)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.EQ.395) THEN
C...g + g -> G* + g;  th arbitrary.
          IF(PYR(0).GT.0.5D0) JS=2
          MINT(23-JS)=KFPR(ISUB,2)
          KCC=22+JS
        ENDIF
 
      ELSEIF(ISUB.LE.420) THEN
        IF(ISUB.EQ.401) THEN
C...g + g -> t + b + H+/-
          KCS=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(KFPR(ISUBSV,2),KCS)
          MINT(22)=ISIGN(5,-KCS)
          KCC=11+INT(0.5D0+PYR(0))
          KFRES=ISIGN(KFHIGG,-KCS)
 
        ELSEIF(ISUB.EQ.402) THEN
C...q + qbar -> t + b + H+/-
          KFL=(-1)**INT(1.5D0+PYR(0))
          MINT(21)=ISIGN(INT(6.+.5*KFL),KCS)
          MINT(22)=ISIGN(INT(6.-.5*KFL),-KCS)
          KCC=4
          KFRES=ISIGN(KFHIGG,-KFL*KCS)

C=====================================================================
C BEGIN HARDCOL MODIFICATION
C=====================================================================
C...The processes 403, 404, 405 and 406, 407, 408 represent 
C...color singlet exchange (in two different approximations),
C...so the colors of the scattered partons should be the same before 
C...and after the scattering. Thus there will be no connection between
C...the two remnants, and a rapidity gap with no string.

        ELSEIF(ISUB.EQ.403) THEN
C...q + q' -> q + q' color singlet exchange
          KCC=22
          
        ELSEIF(ISUB.EQ.404) THEN
C...q + g -> q + g color singlet exchange
          KCC=22
          KCS=ISIGN(1,MINT(15)*MINT(16))
          
        ELSEIF(ISUB.EQ.405) THEN
C...g + g -> g + g color singlet exchange
          KCC=22

        ELSEIF(ISUB.EQ.406) THEN
C...q + q' -> q + q' color singlet exchange
          KCC=22
          
        ELSEIF(ISUB.EQ.407) THEN
C...q + g -> q + g color singlet exchange
          KCC=22
          KCS=ISIGN(1,MINT(15)*MINT(16))
          
        ELSEIF(ISUB.EQ.408) THEN
C...g + g -> g + g colour singlet exchange
          KCC=22
        
C=====================================================================
C END HARDCOL MODIFICATION
C=====================================================================
        ENDIF
 
C...QUARKONIA+++
C...Additional code by Stefan Wolf
      ELSEIF(ISUB.LE.430) THEN
        IF(ISUB.GE.421.AND.ISUB.LE.424) THEN
C...g + g -> QQ~[n] + g
C...MINT(21), MINT(22) copied from ISUB.EQ.86-89
C...[g + g -> (J/Psi, chi_0c, chi_1c or chi_2c) + g]
C...KCC and KCS copied from ISUB.EQ.86-89 (for ISUB.EQ.421)
C...[g + g -> (J/Psi, chi_0c, chi_1c or chi_2c) + g]
C...or from ISUB.EQ.68 (for ISUB.NE.421)
C...[g + g -> g + g; th arbitrary]
          MINT(21)=KFPR(ISUBSV,1)
          MINT(22)=KFPR(ISUBSV,2)
          IF(ISUB.EQ.421) THEN
             KCC=24
             KCS=(-1)**INT(1.5D0+PYR(0))
          ELSE
             KCC=MINT(2)+12
             KCS=(-1)**INT(1.5D0+PYR(0))
          ENDIF
 
        ELSEIF(ISUB.GE.425.AND.ISUB.LE.427) THEN
C...q + g -> q + QQ~[n]
C...MINT(21), MINT(22) "copied" from ISUB.EQ.112
C...[f + g -> f + h0; th = (p(f)-p(f))**2; (q + g -> q + h0 only)]
C...KCC copied from ISUB.EQ.28
C...[f + g -> f + g;  th = (p(f)-p(f))**2; (q + g -> q + g  only)]
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=KFPR(ISUBSV,2)
          KCC=MINT(2)+6
          IF(MINT(15).EQ.21) KCC=KCC+2
          IF(MINT(15).NE.21) KCS=ISIGN(1,MINT(15))
          IF(MINT(16).NE.21) KCS=ISIGN(1,MINT(16))
 
        ELSEIF(ISUB.GE.428.AND.ISUB.LE.430) THEN
C...q + q~ -> g + QQ~[n]
C...MINT(21), MINT(22) "copied" from ISUB.EQ.111
C...[f + fbar -> g + h0; th arbitrary; (q + qbar -> g + h0 only)]
C...KCC copied from ISUB.EQ.13
C...[f + fbar -> g + g;  th arbitrary; (q + qbar -> g + g  only)]
          IF(PYR(0).GT.0.5) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=KFPR(ISUBSV,2)
          KCC=MINT(2)+4
        ENDIF
 
      ELSEIF(ISUB.LE.440) THEN
        IF(ISUB.GE.431.AND.ISUB.LE.433) THEN
C...g + g -> QQ~[n] + g
C...MINT(21), MINT(22) copied from ISUB.EQ.86-89
C...[g + g -> (J/Psi, chi_0c, chi_1c or chi_2c) + g]
C...KCC and KCS copied from ISUB.EQ.86-89
C...[g + g -> (J/Psi, chi_0c, chi_1c or chi_2c) + g]
          MINT(21)=KFPR(ISUBSV,1)
          MINT(22)=KFPR(ISUBSV,2)
          KCC=24
          KCS=(-1)**INT(1.5D0+PYR(0))
 
        ELSEIF(ISUB.GE.434.AND.ISUB.LE.436) THEN
C...q + g -> q + QQ~[n]
C...MINT(21), MINT(22) "copied" from ISUB.EQ.112
C...[f + g -> f + h0; th = (p(f)-p(f))**2; (q + g -> q + h0 only)]
C...KCC and KCS copied from ISUB.EQ.112
C...[f + g -> f + h0; th = (p(f)-p(f))**2; (q + g -> q + h0 only)]
          IF(MINT(15).EQ.21) JS=2
          MINT(23-JS)=KFPR(ISUBSV,2)
          KCC=15+JS
          KCS=ISIGN(1,MINT(14+JS))
 
        ELSEIF(ISUB.GE.437.AND.ISUB.LE.439) THEN
C...q + q~ -> g + QQ~[n]
C...MINT(21), MINT(22) "copied" from ISUB.EQ.111
C...[f + fbar -> g + h0; th arbitrary; (q + qbar -> g + h0 only)]
C...KCC copied from ISUB.EQ.111
C...[f + fbar -> g + h0; th arbitrary; (q + qbar -> g + h0 only)]
          IF(PYR(0).GT.0.5) JS=2
          MINT(20+JS)=21
          MINT(23-JS)=KFPR(ISUBSV,2)
          KCC=17+JS
        ENDIF
C...QUARKONIA---
 
      ENDIF
 
      IF(ISET(ISUB).EQ.11) THEN
C...Store documentation for user-defined processes
        BEZUP=(PUP(3,1)+PUP(3,2))/(PUP(4,1)+PUP(4,2))
        KUPPO(1)=MINT(83)+5
        KUPPO(2)=MINT(83)+6
        I=MINT(83)+6
        DO 470 IUP=3,NUP
          KUPPO(IUP)=0
          IF(MSTP(128).GE.2.AND.MOTHUP(1,IUP).GE.3) THEN
            IDOC=IDOC-1
            MINT(4)=MINT(4)-1
            GOTO 470
          ENDIF
          I=I+1
          KUPPO(IUP)=I
          K(I,1)=21
          K(I,2)=IDUP(IUP)
          IF(IDUP(IUP).EQ.0) K(I,2)=90
          K(I,3)=0
          IF(MOTHUP(1,IUP).GE.3) K(I,3)=KUPPO(MOTHUP(1,IUP))
          K(I,4)=0
          K(I,5)=0
          DO 460 J=1,5
            P(I,J)=PUP(J,IUP)
  460     CONTINUE
          V(I,5)=VTIMUP(IUP)
  470   CONTINUE
        CALL PYROBO(MINT(83)+7,MINT(83)+4+NUP,0D0,VINT(24),0D0,0D0,
     &  -BEZUP)
 
C...Store final state partons for user-defined processes
        N=IPU2
        DO 490 IUP=3,NUP
          N=N+1
          K(N,1)=1
          IF(ISTUP(IUP).EQ.2.OR.ISTUP(IUP).EQ.3) K(N,1)=11
          K(N,2)=IDUP(IUP)
          IF(IDUP(IUP).EQ.0) K(N,2)=90
          IF(MSTP(128).LE.0.OR.MOTHUP(1,IUP).EQ.0) THEN
            K(N,3)=KUPPO(IUP)
          ELSE
            K(N,3)=MINT(84)+MOTHUP(1,IUP)
          ENDIF
          K(N,4)=0
          K(N,5)=0
C...Search for daughters of intermediate colourless particles.
          IF(K(N,1).EQ.11.AND.KCHG(PYCOMP(K(N,2)),2).EQ.0) THEN
            DO 475 IUPDAU=IUP+1,NUP
              IF(MOTHUP(1,IUPDAU).EQ.IUP.AND.K(N,4).EQ.0) K(N,4)=
     &        N+IUPDAU-IUP
              IF(MOTHUP(1,IUPDAU).EQ.IUP) K(N,5)=N+IUPDAU-IUP
  475       CONTINUE
          ENDIF
          DO 480 J=1,5
            P(N,J)=PUP(J,IUP)
  480     CONTINUE
          V(N,5)=VTIMUP(IUP)
  490   CONTINUE
        CALL PYROBO(IPU3,N,0D0,VINT(24),0D0,0D0,-BEZUP)
 
C...Arrange colour flow for user-defined processes
        NLBL=0
        DO 540 IUP1=1,NUP
          I1=MINT(84)+IUP1
          IF(KCHG(PYCOMP(K(I1,2)),2).EQ.0) GOTO 540
          IF(K(I1,1).EQ.1) K(I1,1)=3
          IF(K(I1,1).EQ.11) K(I1,1)=14
C...Find a not yet considered colour/anticolour line.
          DO 530 ISDE1=1,2
            IF(ICOLUP(ISDE1,IUP1).EQ.0) GOTO 530
            NMAT=0
            DO 500 ILBL=1,NLBL
              IF(ICOLUP(ISDE1,IUP1).EQ.ILAB(ILBL)) NMAT=1
  500       CONTINUE
            IF(NMAT.EQ.0) THEN
              NLBL=NLBL+1
              ILAB(NLBL)=ICOLUP(ISDE1,IUP1)
C...Find all others belonging to same line.
              I3=I1
              I4=0
              DO 520 IUP2=IUP1+1,NUP
                I2=MINT(84)+IUP2
                DO 510 ISDE2=1,2
                  IF(ICOLUP(ISDE2,IUP2).EQ.ICOLUP(ISDE1,IUP1)) THEN
                    IF(ISDE2.EQ.ISDE1) THEN
                      K(I3,3+ISDE2)=K(I3,3+ISDE2)+I2
                      K(I2,3+ISDE2)=K(I2,3+ISDE2)+MSTU(5)*I3
                      I3=I2
                    ELSEIF(I4.NE.0) THEN
                      K(I4,3+ISDE2)=K(I4,3+ISDE2)+I2
                      K(I2,3+ISDE2)=K(I2,3+ISDE2)+MSTU(5)*I4
                      I4=I2
                    ELSEIF(IUP2.LE.2) THEN
                      K(I1,3+ISDE1)=K(I1,3+ISDE1)+I2
                      K(I2,3+ISDE2)=K(I2,3+ISDE2)+I1
                      I4=I2
                    ELSE
                      K(I1,3+ISDE1)=K(I1,3+ISDE1)+MSTU(5)*I2
                      K(I2,3+ISDE2)=K(I2,3+ISDE2)+MSTU(5)*I1
                      I4=I2
                    ENDIF
                  ENDIF
  510           CONTINUE
  520         CONTINUE
            ENDIF
  530     CONTINUE
  540   CONTINUE
 
      ELSEIF(IDOC.EQ.7) THEN
C...Resonance not decaying; store kinematics
        I=MINT(83)+7
        K(IPU3,1)=1
        K(IPU3,2)=KFRES
        K(IPU3,3)=I
        P(IPU3,4)=SHUSER
        P(IPU3,5)=SHUSER
        K(I,1)=21
        K(I,2)=KFRES
        P(I,4)=SHUSER
        P(I,5)=SHUSER
        N=IPU3
        MINT(21)=KFRES
        MINT(22)=0
 
C...Special cases: colour flow in coloured resonances
        KCRES=PYCOMP(KFRES)
        IF(KCHG(KCRES,2).NE.0) THEN
          K(IPU3,1)=3
          DO 550 J=1,2
            JC=J
            IF(KCS.EQ.-1) JC=3-J
            IF(ICOL(KCC,1,JC).NE.0.AND.K(IPU1,1).EQ.14) K(IPU1,J+3)=
     &      MINT(84)+ICOL(KCC,1,JC)
            IF(ICOL(KCC,2,JC).NE.0.AND.K(IPU2,1).EQ.14) K(IPU2,J+3)=
     &      MINT(84)+ICOL(KCC,2,JC)
            IF(ICOL(KCC,3,JC).NE.0.AND.K(IPU3,1).EQ.3) K(IPU3,J+3)=
     &      MSTU(5)*(MINT(84)+ICOL(KCC,3,JC))
  550     CONTINUE
        ELSE
          K(IPU1,4)=IPU2
          K(IPU1,5)=IPU2
          K(IPU2,4)=IPU1
          K(IPU2,5)=IPU1
        ENDIF
 
      ELSEIF(IDOC.EQ.8) THEN
C...2 -> 2 processes: store outgoing partons in their CM-frame
        DO 560 JT=1,2
          I=MINT(84)+2+JT
          KCA=PYCOMP(MINT(20+JT))
          K(I,1)=1
          IF(KCHG(KCA,2).NE.0) K(I,1)=3
          K(I,2)=MINT(20+JT)
          K(I,3)=MINT(83)+IDOC+JT-2
          KFAA=IABS(K(I,2))
          IF(KFPR(ISUBSV,1+MOD(JS+JT,2)).NE.0) THEN
            P(I,5)=SQRT(VINT(63+MOD(JS+JT,2)))
          ELSE
            P(I,5)=PYMASS(K(I,2))
          ENDIF
          IF((KFAA.EQ.6.OR.KFAA.EQ.7.OR.KFAA.EQ.8).AND.
     &    P(I,5).LT.PARP(42)) P(I,5)=PYMASS(K(I,2))
  560   CONTINUE
        IF(P(IPU3,5)+P(IPU4,5).GE.SHR) THEN
          KFA1=IABS(MINT(21))
          KFA2=IABS(MINT(22))
          IF((KFA1.GT.3.AND.KFA1.NE.21).OR.(KFA2.GT.3.AND.KFA2.NE.21))
     &    THEN
            MINT(51)=1
            RETURN
          ENDIF
          P(IPU3,5)=0D0
          P(IPU4,5)=0D0
        ENDIF
        P(IPU3,4)=0.5D0*(SHR+(P(IPU3,5)**2-P(IPU4,5)**2)/SHR)
        P(IPU3,3)=SQRT(MAX(0D0,P(IPU3,4)**2-P(IPU3,5)**2))
        P(IPU4,4)=SHR-P(IPU3,4)
        P(IPU4,3)=-P(IPU3,3)
        N=IPU4
        MINT(7)=MINT(83)+7
        MINT(8)=MINT(83)+8
 
C...Rotate outgoing partons using cos(theta)=(th-uh)/lam(sh,sqm3,sqm4)
        CALL PYROBO(IPU3,IPU4,ACOS(VINT(23)),VINT(24),0D0,0D0,0D0)
 
      ELSEIF(IDOC.EQ.9) THEN
C...2 -> 3 processes: store outgoing partons in their CM frame
        DO 570 JT=1,2
          I=MINT(84)+2+JT
          KCA=PYCOMP(MINT(20+JT))
          K(I,1)=1
          IF(KCHG(KCA,2).NE.0) K(I,1)=3
          K(I,2)=MINT(20+JT)
          K(I,3)=MINT(83)+IDOC+JT-3
          JTA=JT
C...t and b in opposide order in event list as compared to
C...matrix element?
          IF(ISUB.EQ.402.AND.IABS(MINT(21)).EQ.5) JTA=3-JT
          IF(IABS(K(I,2)).LE.22) THEN
            P(I,5)=PYMASS(K(I,2))
          ELSE
            P(I,5)=SQRT(VINT(63+MOD(JS+JTA,2)))
          ENDIF
          PT=SQRT(MAX(0D0,VINT(197+5*JTA)-P(I,5)**2+VINT(196+5*JTA)**2))
          P(I,1)=PT*COS(VINT(198+5*JTA))
          P(I,2)=PT*SIN(VINT(198+5*JTA))
  570   CONTINUE
        K(IPU5,1)=1
        K(IPU5,2)=KFRES
        K(IPU5,3)=MINT(83)+IDOC
        P(IPU5,5)=SHR
        P(IPU5,1)=-P(IPU3,1)-P(IPU4,1)
        P(IPU5,2)=-P(IPU3,2)-P(IPU4,2)
        PMS1=P(IPU3,5)**2+P(IPU3,1)**2+P(IPU3,2)**2
        PMS2=P(IPU4,5)**2+P(IPU4,1)**2+P(IPU4,2)**2
        PMS3=P(IPU5,5)**2+P(IPU5,1)**2+P(IPU5,2)**2
        PMT3=SQRT(PMS3)
        P(IPU5,3)=PMT3*SINH(VINT(211))
        P(IPU5,4)=PMT3*COSH(VINT(211))
        PMS12=(SHPR-P(IPU5,4))**2-P(IPU5,3)**2
        SQL12=(PMS12-PMS1-PMS2)**2-4D0*PMS1*PMS2
        IF(SQL12.LE.0D0) THEN
          MINT(51)=1
          RETURN
        ENDIF
        P(IPU3,3)=(-P(IPU5,3)*(PMS12+PMS1-PMS2)+
     &  VINT(213)*(SHPR-P(IPU5,4))*SQRT(SQL12))/(2D0*PMS12)
        P(IPU4,3)=-P(IPU3,3)-P(IPU5,3)
        IF(ISUB.EQ.402.AND.IABS(MINT(21)).EQ.5) THEN
C...t and b in opposide order in event list as compared to
C...matrix element
          P(IPU4,3)=(-P(IPU5,3)*(PMS12+PMS2-PMS1)+
     &    VINT(213)*(SHPR-P(IPU5,4))*SQRT(SQL12))/(2D0*PMS12)
          P(IPU3,3)=-P(IPU4,3)-P(IPU5,3)
        END IF
        P(IPU3,4)=SQRT(PMS1+P(IPU3,3)**2)
        P(IPU4,4)=SQRT(PMS2+P(IPU4,3)**2)
        MINT(23)=KFRES
        N=IPU5
        MINT(7)=MINT(83)+7
        MINT(8)=MINT(83)+8
 
      ELSEIF(IDOC.EQ.11) THEN
C...Z0 + Z0 -> h0, W+ + W- -> h0: store Higgs and outgoing partons
        PHI(1)=PARU(2)*PYR(0)
        PHI(2)=PHI(1)-PHIR
        DO 580 JT=1,2
          I=MINT(84)+2+JT
          K(I,1)=1
          IF(KCHG(PYCOMP(MINT(20+JT)),2).NE.0) K(I,1)=3
          K(I,2)=MINT(20+JT)
          K(I,3)=MINT(83)+IDOC+JT-2
          P(I,5)=PYMASS(K(I,2))
          IF(0.5D0*SHPR*Z(JT).LE.P(I,5)) THEN
            MINT(51)=1
            RETURN
          ENDIF
          PABS=SQRT(MAX(0D0,(0.5D0*SHPR*Z(JT))**2-P(I,5)**2))
          PTABS=PABS*SQRT(MAX(0D0,1D0-CTHE(JT)**2))
          P(I,1)=PTABS*COS(PHI(JT))
          P(I,2)=PTABS*SIN(PHI(JT))
          P(I,3)=PABS*CTHE(JT)*(-1)**(JT+1)
          P(I,4)=0.5D0*SHPR*Z(JT)
          IZW=MINT(83)+6+JT
          K(IZW,1)=21
          K(IZW,2)=23
          IF(ISUB.EQ.8) K(IZW,2)=ISIGN(24,PYCHGE(MINT(14+JT)))
          K(IZW,3)=IZW-2
          P(IZW,1)=-P(I,1)
          P(IZW,2)=-P(I,2)
          P(IZW,3)=(0.5D0*SHPR-PABS*CTHE(JT))*(-1)**(JT+1)
          P(IZW,4)=0.5D0*SHPR*(1D0-Z(JT))
          P(IZW,5)=-SQRT(MAX(0D0,P(IZW,3)**2+PTABS**2-P(IZW,4)**2))
  580   CONTINUE
        I=MINT(83)+9
        K(IPU5,1)=1
        K(IPU5,2)=KFRES
        K(IPU5,3)=I
        P(IPU5,5)=SHR
        P(IPU5,1)=-P(IPU3,1)-P(IPU4,1)
        P(IPU5,2)=-P(IPU3,2)-P(IPU4,2)
        P(IPU5,3)=-P(IPU3,3)-P(IPU4,3)
        P(IPU5,4)=SHPR-P(IPU3,4)-P(IPU4,4)
        K(I,1)=21
        K(I,2)=KFRES
        DO 590 J=1,5
          P(I,J)=P(IPU5,J)
  590   CONTINUE
        N=IPU5
        MINT(23)=KFRES
 
      ELSEIF(IDOC.EQ.12) THEN
C...Z0 and W+/- scattering: store bosons and outgoing partons
        PHI(1)=PARU(2)*PYR(0)
        PHI(2)=PHI(1)-PHIR
        JTRAN=INT(1.5D0+PYR(0))
        DO 600 JT=1,2
          I=MINT(84)+2+JT
          K(I,1)=1
          IF(KCHG(PYCOMP(MINT(20+JT)),2).NE.0) K(I,1)=3
          K(I,2)=MINT(20+JT)
          K(I,3)=MINT(83)+IDOC+JT-2
          P(I,5)=PYMASS(K(I,2))
          IF(0.5D0*SHPR*Z(JT).LE.P(I,5)) P(I,5)=0D0
          PABS=SQRT(MAX(0D0,(0.5D0*SHPR*Z(JT))**2-P(I,5)**2))
          PTABS=PABS*SQRT(MAX(0D0,1D0-CTHE(JT)**2))
          P(I,1)=PTABS*COS(PHI(JT))
          P(I,2)=PTABS*SIN(PHI(JT))
          P(I,3)=PABS*CTHE(JT)*(-1)**(JT+1)
          P(I,4)=0.5D0*SHPR*Z(JT)
          IZW=MINT(83)+6+JT
          K(IZW,1)=21
          IF(MINT(14+JT).EQ.MINT(20+JT)) THEN
            K(IZW,2)=23
          ELSE
            K(IZW,2)=ISIGN(24,PYCHGE(MINT(14+JT))-PYCHGE(MINT(20+JT)))
          ENDIF
          K(IZW,3)=IZW-2
          P(IZW,1)=-P(I,1)
          P(IZW,2)=-P(I,2)
          P(IZW,3)=(0.5D0*SHPR-PABS*CTHE(JT))*(-1)**(JT+1)
          P(IZW,4)=0.5D0*SHPR*(1D0-Z(JT))
          P(IZW,5)=-SQRT(MAX(0D0,P(IZW,3)**2+PTABS**2-P(IZW,4)**2))
          IPU=MINT(84)+4+JT
          K(IPU,1)=3
          K(IPU,2)=KFPR(ISUB,JT)
          IF(ISUB.EQ.72.AND.JT.EQ.JTRAN) K(IPU,2)=-K(IPU,2)
          IF(ISUB.EQ.73.OR.ISUB.EQ.77) K(IPU,2)=K(IZW,2)
          K(IPU,3)=MINT(83)+8+JT
          IF(IABS(K(IPU,2)).LE.10.OR.K(IPU,2).EQ.21) THEN
            P(IPU,5)=PYMASS(K(IPU,2))
          ELSE
            P(IPU,5)=SQRT(VINT(63+MOD(JS+JT,2)))
          ENDIF
          MINT(22+JT)=K(IPU,2)
  600   CONTINUE
C...Find rotation and boost for hard scattering subsystem
        I1=MINT(83)+7
        I2=MINT(83)+8
        BEXCM=(P(I1,1)+P(I2,1))/(P(I1,4)+P(I2,4))
        BEYCM=(P(I1,2)+P(I2,2))/(P(I1,4)+P(I2,4))
        BEZCM=(P(I1,3)+P(I2,3))/(P(I1,4)+P(I2,4))
        GAMCM=(P(I1,4)+P(I2,4))/SHR
        BEPCM=BEXCM*P(I1,1)+BEYCM*P(I1,2)+BEZCM*P(I1,3)
        PX=P(I1,1)+GAMCM*(GAMCM/(1D0+GAMCM)*BEPCM-P(I1,4))*BEXCM
        PY=P(I1,2)+GAMCM*(GAMCM/(1D0+GAMCM)*BEPCM-P(I1,4))*BEYCM
        PZ=P(I1,3)+GAMCM*(GAMCM/(1D0+GAMCM)*BEPCM-P(I1,4))*BEZCM
        THECM=PYANGL(PZ,SQRT(PX**2+PY**2))
        PHICM=PYANGL(PX,PY)
C...Store hard scattering subsystem. Rotate and boost it
        SQLAM=(SH-P(IPU5,5)**2-P(IPU6,5)**2)**2-4D0*P(IPU5,5)**2*
     &  P(IPU6,5)**2
        PABS=SQRT(MAX(0D0,SQLAM/(4D0*SH)))
        CTHWZ=VINT(23)
        STHWZ=SQRT(MAX(0D0,1D0-CTHWZ**2))
        PHIWZ=VINT(24)-PHICM
        P(IPU5,1)=PABS*STHWZ*COS(PHIWZ)
        P(IPU5,2)=PABS*STHWZ*SIN(PHIWZ)
        P(IPU5,3)=PABS*CTHWZ
        P(IPU5,4)=SQRT(PABS**2+P(IPU5,5)**2)
        P(IPU6,1)=-P(IPU5,1)
        P(IPU6,2)=-P(IPU5,2)
        P(IPU6,3)=-P(IPU5,3)
        P(IPU6,4)=SQRT(PABS**2+P(IPU6,5)**2)
        CALL PYROBO(IPU5,IPU6,THECM,PHICM,BEXCM,BEYCM,BEZCM)
        DO 620 JT=1,2
          I1=MINT(83)+8+JT
          I2=MINT(84)+4+JT
          K(I1,1)=21
          K(I1,2)=K(I2,2)
          DO 610 J=1,5
            P(I1,J)=P(I2,J)
  610     CONTINUE
  620   CONTINUE
        N=IPU6
        MINT(7)=MINT(83)+9
        MINT(8)=MINT(83)+10
      ENDIF
 
      IF(ISET(ISUB).EQ.11) THEN
      ELSEIF(IDOC.GE.8) THEN
C...Store colour connection indices
        DO 630 J=1,2
          JC=J
          IF(KCS.EQ.-1) JC=3-J
          IF(ICOL(KCC,1,JC).NE.0.AND.K(IPU1,1).EQ.14) K(IPU1,J+3)=
     &    K(IPU1,J+3)+MINT(84)+ICOL(KCC,1,JC)
          IF(ICOL(KCC,2,JC).NE.0.AND.K(IPU2,1).EQ.14) K(IPU2,J+3)=
     &    K(IPU2,J+3)+MINT(84)+ICOL(KCC,2,JC)
          IF(ICOL(KCC,3,JC).NE.0.AND.K(IPU3,1).EQ.3) K(IPU3,J+3)=
     &    MSTU(5)*(MINT(84)+ICOL(KCC,3,JC))
          IF(ICOL(KCC,4,JC).NE.0.AND.K(IPU4,1).EQ.3) K(IPU4,J+3)=
     &    MSTU(5)*(MINT(84)+ICOL(KCC,4,JC))
  630   CONTINUE
 
C...Copy outgoing partons to documentation lines
        IMAX=2
        IF(IDOC.EQ.9) IMAX=3
        DO 650 I=1,IMAX
          I1=MINT(83)+IDOC-IMAX+I
          I2=MINT(84)+2+I
          K(I1,1)=21
          K(I1,2)=K(I2,2)
          IF(IDOC.LE.9) K(I1,3)=0
          IF(IDOC.GE.11) K(I1,3)=MINT(83)+2+I
          DO 640 J=1,5
            P(I1,J)=P(I2,J)
  640     CONTINUE
  650   CONTINUE
 
      ELSEIF(IDOC.EQ.9) THEN
C...Store colour connection indices
        DO 660 J=1,2
          JC=J
          IF(KCS.EQ.-1) JC=3-J
          IF(ICOL(KCC,1,JC).NE.0.AND.K(IPU1,1).EQ.14) K(IPU1,J+3)=
     &    K(IPU1,J+3)+MINT(84)+ICOL(KCC,1,JC)+
     &    MAX(0,MIN(1,ICOL(KCC,1,JC)-2))
          IF(ICOL(KCC,2,JC).NE.0.AND.K(IPU2,1).EQ.14) K(IPU2,J+3)=
     &    K(IPU2,J+3)+MINT(84)+ICOL(KCC,2,JC)+
     &    MAX(0,MIN(1,ICOL(KCC,2,JC)-2))
          IF(ICOL(KCC,3,JC).NE.0.AND.K(IPU4,1).EQ.3) K(IPU4,J+3)=
     &    MSTU(5)*(MINT(84)+ICOL(KCC,3,JC))
          IF(ICOL(KCC,4,JC).NE.0.AND.K(IPU5,1).EQ.3) K(IPU5,J+3)=
     &    MSTU(5)*(MINT(84)+ICOL(KCC,4,JC))
  660   CONTINUE
 
C...Copy outgoing partons to documentation lines
        DO 680 I=1,3
          I1=MINT(83)+IDOC-3+I
          I2=MINT(84)+2+I
          K(I1,1)=21
          K(I1,2)=K(I2,2)
          K(I1,3)=0
          DO 670 J=1,5
            P(I1,J)=P(I2,J)
  670     CONTINUE
  680   CONTINUE
      ENDIF
 
C...Copy outgoing partons to list of allowed radiators.
      NPART=0
      IF(MINT(35).GE.2.AND.ISET(ISUB).NE.0) THEN
        DO 690 I=MINT(84)+3,N
          NPART=NPART+1
          IPART(NPART)=I
          PTPART(NPART)=SQRT(P(I,5)**2+P(I,1)**2+P(I,2)**2)
  690   CONTINUE
      ENDIF
 
C...Low-pT events: remove gluons used for string drawing purposes
      IF(ISUB.EQ.95) THEN
        IF(MINT(35).LE.1) THEN
          K(IPU3,1)=K(IPU3,1)+10
          K(IPU4,1)=K(IPU4,1)+10
        ENDIF
        DO 700 J=41,66
          VINTSV(J)=VINT(J)
          VINT(J)=0D0
  700   CONTINUE
        DO 720 I=MINT(83)+5,MINT(83)+8
          DO 710 J=1,5
            P(I,J)=0D0
  710     CONTINUE
  720   CONTINUE
      ENDIF
 
      RETURN
      END
