c     Version 1.383
c     The variables I_SNG in HIJSFT and JL in ATTRAD were not initialized.
c     The version initialize them. (as found by Fernando Marroquim)
c
c
c
c     Version 1.382
c     Nuclear distribution for deuteron is taken as the Hulthen wave
c     function as provided by Brian Cole (Columbia)
c
c
c     Version 1.381
c
c     The parameters for Wood-Saxon distribution for deuteron are
c     constrained to give the right rms ratius 2.116 fm
c     (R=0.0, D=0.5882)
c
c
c     Version 1.38
c
c     The following common block is added to record the number of elastic
c     (NELT, NELP) and inelastic (NINT, NINP) participants
c
c        COMMON/HIJGLBR/NELT,NINT,NELP,NINP
c        SAVE  /HIJGLBR/
c
c     Version 1.37
c
c     A bug in the quenching subroutine is corrected. When calculating the
c     distance between two wounded nucleons, the displacement of the
c     impact parameter was not inculded. This bug was discovered by
c     Dr. V.Uzhinskii JINR, Dubna, Russia
c
c
C     Version 1.36
c
c     Modification Oct. 8, 1998. In hijing, log(ran(nseed)) occasionally
c     causes overfloat. It is modified to log(max(ran(nseed),1.0e-20)).
c
c
C     Nothing important has been changed here. A few 'garbage' has been
C     cleaned up here, like common block HIJJET3 for the sea quark strings
C     which were originally created to implement the DPM scheme which
C     later was abadoned in the final version. The lines which operate
C     on these data are also deleted in the program.
C
C
C     Version 1.35
C     There are some changes in the program: subroutine HARDJET is now
C     consolidated with HIJHRD. HARDJET is used to re-initiate PYTHIA
C     for the triggered hard processes. Now that is done  altogether
C     with other normal hard processes in modified JETINI. In the new
C     version one calls JETINI every time one calls HIJHRD. In the new
C     version the effect of the isospin of the nucleon on hard processes,
C     especially direct photons is correctly considered.
C     For A+A collisions, one has to initilize pythia
C     separately for each type of collisions, pp, pn,np and nn,
C     or hp and hn for hA collisions. In JETINI we use the following
C     catalogue for different types of collisions:
C     h+h: h+h (I_TYPE=1)
C     h+A: h+p (I_TYPE=1), h+n (I_TYPE=2)
C     A+h: p+h (I_TYPE=1), n+h (I_TYPE=2)
C     A+A: p+p (I_TYPE=1), p+n (I_TYPE=2), n+p (I_TYPE=3), n+n (I_TYPE=4)
C*****************************************************************
c
C
C     Version 1.34
C     Last modification on January 5, 1998. Two mistakes are corrected in
C     function G. A Mistake in the subroutine Parton is also corrected.
C     (These are pointed out by Ysushi Nara).
C
C
C       Last modifcation on April 10, 1996. To conduct final
C       state radiation, PYTHIA reorganize the two scattered
C       partons and their final momenta will be a little
C       different. The summed total momenta of the partons
C       from the final state radiation are stored in HINT1(26-29)
C       and HINT1(36-39) which are little different from 
C       HINT1(21-24) and HINT1(41-44).
C
C       Version 1.33
C
C       Last modfication  on September 11, 1995. When HIJING and
C       PYTHIA are initialized, the shadowing is evaluated at
C       b=0 which is the maximum. This will cause overestimate
C       of shadowing for peripheral interactions. To correct this
C       problem, shadowing is set to zero when initializing. Then
C       use these maximum  cross section without shadowing as a
C       normalization of the Monte Carlo. This however increase
C       the computing time. IHNT2(16) is used to indicate whether
C       the sturcture function is called for (IHNT2(16)=1) initialization
C       or for (IHNT2(16)=0)normal collisions simulation
C
C       Last modification on Aagust 28, 1994. Two bugs associate
C       with the impact parameter dependence of the shadowing is
C       corrected.
C
C
c       Last modification on October 14, 1994. One bug is corrected
c       in the direct photon production option in subroutine
C       HIJHRD.( this problem was reported by Jim Carroll and Mike Beddo).
C       Another bug associated with keeping the decay history
C       in the particle information is also corrected.(this problem
C       was reported by Matt Bloomer)
C
C
C       Last modification on July 15, 1994. The option to trig on
C       heavy quark production (charm IHPR2(18)=0 or beauty IHPR2(18)=1) 
C       is added. To do this, set IHPR2(3)=3. For inclusive production,
C       one should reset HIPR1(10)=0.0. One can also trig larger pt
C       QQbar production by giving HIPR1(10) a nonvanishing value.
C       The mass of the heavy quark in the calculation of the cross
C       section (HINT1(59)--HINT1(65)) is given by HIPR1(7) (the
C       default is the charm mass D=1.5). We also include a separate
C       K-factor for heavy quark and direct photon production by
C       HIPR1(23)(D=2.0).
C
C       Last modification on May 24, 1994.  The option to
C       retain the information of all particles including those
C       who have decayed is IHPR(21)=1 (default=0). KATT(I,3) is 
C       added to contain the line number of the parent particle 
C       of the current line which is produced via a decay. 
C       KATT(I,4) is the status number of the particle: 11=particle
C       which has decayed; 1=finally produced particle.
C
C
C       Last modification on May 24, 1994( in HIJSFT when valence quark
C       is quenched, the following error is corrected. 1.2*IHNT2(1) --> 
C       1.2*IHNT2(1)**0.333333, 1.2*IHNT2(3) -->1.2*IHNT(3)**0.333333)
C
C
C       Last modification on March 16, 1994 (heavy flavor production
C       processes MSUB(81)=1 MSUB(82)=1 have been switched on,
C       charm production is the default, B-quark option is
C       IHPR2(18), when it is switched on, charm quark is 
C       automatically off)
C
C
C       Last modification on March 23, 1994 (an error is corrected
C       in the impact parameter dependence of the jet cross section)
C
C       Last modification Oct. 1993 to comply with non-vax
C       machines compiler 
C
C*********************************************
C	LAST MODIFICATION April 5, 1991
CQUARK DISTRIBUTIOIN (1-X)**A/(X**2+C**2/S)**B 
C(A=HIPR1(44),B=HIPR1(46),C=HIPR1(45))
C STRING FLIP, VENUS OPTION IHPR2(15)=1,IN WHICH ONE CAN HAVE ONE AND
C TWO COLOR CHANGES, (1-W)**2,W*(1-W),W*(1-W),AND W*2, W=HIPR1(18), 
C AMONG PT DISTRIBUTION OF SEA QUARKS IS CONTROLLED BY HIPR1(42)
C
C	gluon jets can form a single string system
C
C	initial state radiation is included
C	
C	all QCD subprocesses are included
c
c	direct particles production is included(currently only direct
C		photon)
c
C	Effect of high P_T trigger bias on multiple jets distribution
c
C******************************************************************
C	                        HIJING.10                         *
C	          Heavy Ion Jet INteraction Generator        	  *
C	                           by                       	  *
C		   X. N. Wang      and   M. Gyulassy           	  *
C	 	      Lawrence Berkeley Laboratory		  *
C								  *
C******************************************************************
C
C******************************************************************
C NFP(K,1),NFP(K,2)=flavor of q and di-q, NFP(K,3)=present ID of  *
C proj, NFP(K,4) original ID of proj.  NFP(K,5)=colli status(0=no,*
C 1=elastic,2=the diffrac one in single-diffrac,3= excited string.*
C |NFP(K,6)| is the total # of jet production, if NFP(K,6)<0 it   *
C can not produce jet anymore. NFP(K,10)=valence quarks scattering*
C (0=has not been,1=is going to be, -1=has already been scattered *
C NFP(k,11) total number of interactions this proj has suffered   *
C PP(K,1)=PX,PP(K,2)=PY,PP(K,3)=PZ,PP(K,4)=E,PP(K,5)=M(invariant  *
C mass), PP(K,6,7),PP(K,8,9)=transverse momentum of quark and     *
C diquark,PP(K,10)=PT of the hard scattering between the valence  *
C quarks; PP(K,14,15)=the mass of quark,diquark.       		  * 
C******************************************************************
C
C****************************************************************
C
C	SUBROUTINE HIJING
C
C****************************************************************
	SUBROUTINE HIJING(FRAME,BMIN0,BMAX0)
	CHARACTER FRAME*8
	DIMENSION SCIP(300,300),RNIP(300,300),SJIP(300,300),JTP(3),
     &			IPCOL(90000),ITCOL(90000)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
C
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
        COMMON/HIJGLBR/NELT,NINT,NELP,NINP
c	COMMON/HIMAIN1/NATT,EATT,JATT,NT,NP,N0,N01,N10,N11,IERRSTAT -man
	COMMON/HIMAIN1/NATT,EATT,JATT,NT,NP,N0,N01,N10,N11
	COMMON/HIMAIN2/KATT(130000,4),PATT(130000,4),VATT(130000,4)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HIJJET4/NDR,IADR(900,2),KFDR(900),PDR(900,5),VDR(900,4)
	COMMON/RANSEED/NSEED
C
	COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)   
	COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
        SAVE

C     Initialize error return code        
        IERRSTAT = 0
	BMAX=MIN(BMAX0,HIPR1(34)+HIPR1(35))
	BMIN=MIN(BMIN0,BMAX)
	IF(IHNT2(1).LE.1 .AND. IHNT2(3).LE.1) THEN
		BMIN=0.0
		BMAX=2.5*SQRT(HIPR1(31)*0.1/HIPR1(40))
	ENDIF
C			********HIPR1(31) is in mb =0.1fm**2
C*******THE FOLLOWING IS TO SELECT THE COORDINATIONS OF NUCLEONS 
C       BOTH IN PROJECTILE AND TARGET NUCLEAR( in fm)
C
	YP(1,1)=0.0
	YP(2,1)=0.0
	YP(3,1)=0.0
	IF(IHNT2(1).LE.1) GO TO 14
	DO 10 KP=1,IHNT2(1)
5	R=HIRND(1)
c
        if(IHNT2(1).EQ.2) then
           rnd1=max(RLU(NSEED),1.0e-20)
           rnd2=max(RLU(NSEED),1.0e-20)
           rnd3=max(RLU(NSEED),1.0e-20)
           R=-0.5*(log(rnd1)*4.38/2.0+log(rnd2)*0.85/2.0
     &          +4.38*0.85*log(rnd3)/(4.38+0.85))
        endif
c
	X=RLU(NSEED)
	CX=2.0*X-1.0
	SX=SQRT(1.0-CX*CX)
C		********choose theta from uniform cos(theta) distr
	PHI=RLU(NSEED)*2.0*HIPR1(40)
C		********choose phi form uniform phi distr 0 to 2*pi
	YP(1,KP)=R*SX*COS(PHI)
	YP(2,KP)=R*SX*SIN(PHI)
	YP(3,KP)=R*CX
	IF(HIPR1(29).EQ.0.0) GO TO 10
	DO 8  KP2=1,KP-1
		DNBP1=(YP(1,KP)-YP(1,KP2))**2
		DNBP2=(YP(2,KP)-YP(2,KP2))**2
		DNBP3=(YP(3,KP)-YP(3,KP2))**2
		DNBP=DNBP1+DNBP2+DNBP3
		IF(DNBP.LT.HIPR1(29)*HIPR1(29)) GO TO 5
C			********two neighbors cannot be closer than 
C				HIPR1(29)
8	CONTINUE
10	CONTINUE
c*******************************
        if(IHNT2(1).EQ.2) then
           YP(1,2)=-YP(1,1)
           YP(2,2)=-YP(2,1)
           YP(3,2)=-YP(3,1)
        endif
c********************************
	DO 12 I=1,IHNT2(1)-1
	DO 12 J=I+1,IHNT2(1)
	IF(YP(3,I).GT.YP(3,J)) GO TO 12
	Y1=YP(1,I)
	Y2=YP(2,I)
	Y3=YP(3,I)
	YP(1,I)=YP(1,J)
	YP(2,I)=YP(2,J)
	YP(3,I)=YP(3,J)
	YP(1,J)=Y1
	YP(2,J)=Y2
	YP(3,J)=Y3
12	CONTINUE
C
C******************************
14	YT(1,1)=0.0
	YT(2,1)=0.0
	YT(3,1)=0.0
	IF(IHNT2(3).LE.1) GO TO 24
	DO 20 KT=1,IHNT2(3)
15	R=HIRND(2)
c
        if(IHNT2(3).EQ.2) then
           rnd1=max(RLU(NSEED),1.0e-20)
           rnd2=max(RLU(NSEED),1.0e-20)
           rnd3=max(RLU(NSEED),1.0e-20)
           R=-0.5*(log(rnd1)*4.38/2.0+log(rnd2)*0.85/2.0
     &          +4.38*0.85*log(rnd3)/(4.38+0.85))
        endif
c
	X=RLU(NSEED)
	CX=2.0*X-1.0
	SX=SQRT(1.0-CX*CX)
C		********choose theta from uniform cos(theta) distr
	PHI=RLU(NSEED)*2.0*HIPR1(40)
C		********chose phi form uniform phi distr 0 to 2*pi
	YT(1,KT)=R*SX*COS(PHI)
	YT(2,KT)=R*SX*SIN(PHI)
	YT(3,KT)=R*CX
	IF(HIPR1(29).EQ.0.0) GO TO 20
	DO 18  KT2=1,KT-1
		DNBT1=(YT(1,KT)-YT(1,KT2))**2
		DNBT2=(YT(2,KT)-YT(2,KT2))**2
		DNBT3=(YT(3,KT)-YT(3,KT2))**2
		DNBT=DNBT1+DNBT2+DNBT3
		IF(DNBT.LT.HIPR1(29)*HIPR1(29)) GO TO 15
C			********two neighbors cannot be closer than 
C				HIPR1(29)
18	CONTINUE
20	CONTINUE
c**********************************
        if(IHNT2(3).EQ.2) then
           YT(1,2)=-YT(1,1)
           YT(2,2)=-YT(2,1)
           YT(3,2)=-YT(3,1)
        endif
c*********************************
	DO 22 I=1,IHNT2(3)-1
	DO 22 J=I+1,IHNT2(3)
	IF(YT(3,I).LT.YT(3,J)) GO TO 22
	Y1=YT(1,I)
	Y2=YT(2,I)
	Y3=YT(3,I)
	YT(1,I)=YT(1,J)
	YT(2,I)=YT(2,J)
	YT(3,I)=YT(3,J)
	YT(1,J)=Y1
	YT(2,J)=Y2
	YT(3,J)=Y3
22	CONTINUE
C********************
24	MISS=-1

50	MISS=MISS+1
	IF(MISS.GT.50) THEN
	   WRITE(6,*) 'infinite loop happened in  HIJING'
	   STOP
	ENDIF

	NATT=0
	JATT=0
	EATT=0.0
	CALL HIJINI
        NLOP=0
C			********Initialize for a new event
60	NT=0
	NP=0
	N0=0
	N01=0
	N10=0
	N11=0
        NELT=0
        NINT=0
        NELP=0
        NINP=0
	NSG=0
	NCOLT=0

C****	BB IS THE ABSOLUTE VALUE OF IMPACT PARAMETER,BB**2 IS 
C       RANDOMLY GENERATED AND ITS ORIENTATION IS RANDOMLY SET 
C       BY THE ANGLE PHI  FOR EACH COLLISION.******************
C
	BB=SQRT(BMIN**2+RLU(NSEED)*(BMAX**2-BMIN**2))
	PHI=2.0*HIPR1(40)*RLU(NSEED)
	BBX=BB*COS(PHI)
	BBY=BB*SIN(PHI)
	HINT1(19)=BB
	HINT1(20)=PHI
C
	DO 70 JP=1,IHNT2(1)
	DO 70 JT=1,IHNT2(3)
	   SCIP(JP,JT)=-1.0
	   B2=(YP(1,JP)+BBX-YT(1,JT))**2+(YP(2,JP)+BBY-YT(2,JT))**2
	   R2=B2*HIPR1(40)/HIPR1(31)/0.1
C		********mb=0.1*fm, YP is in fm,HIPR1(31) is in mb
	   RRB1=MIN((YP(1,JP)**2+YP(2,JP)**2)
     &          /1.2**2/REAL(IHNT2(1))**0.6666667,1.0)
	   RRB2=MIN((YT(1,JT)**2+YT(2,JT)**2)
     &          /1.2**2/REAL(IHNT2(3))**0.6666667,1.0)
	   APHX1=HIPR1(6)*4.0/3.0*(IHNT2(1)**0.3333333-1.0)
     &           *SQRT(1.0-RRB1)
	   APHX2=HIPR1(6)*4.0/3.0*(IHNT2(3)**0.3333333-1.0)
     &           *SQRT(1.0-RRB2)
	   HINT1(18)=HINT1(14)-APHX1*HINT1(15)
     &			-APHX2*HINT1(16)+APHX1*APHX2*HINT1(17)
	   IF(IHPR2(14).EQ.0.OR.
     &          (IHNT2(1).EQ.1.AND.IHNT2(3).EQ.1)) THEN
	      GS=1.0-EXP(-(HIPR1(30)+HINT1(18))*ROMG(R2)/HIPR1(31))
	      RANTOT=RLU(NSEED)
	      IF(RANTOT.GT.GS) GO TO 70
	      GO TO 65
	   ENDIF
	   GSTOT_0=2.0*(1.0-EXP(-(HIPR1(30)+HINT1(18))
     &             /HIPR1(31)/2.0*ROMG(0.0)))
	   R2=R2/GSTOT_0
	   GS=1.0-EXP(-(HIPR1(30)+HINT1(18))/HIPR1(31)*ROMG(R2))
	   GSTOT=2.0*(1.0-SQRT(1.0-GS))
	   RANTOT=RLU(NSEED)*GSTOT_0
	   IF(RANTOT.GT.GSTOT) GO TO 70
	   IF(RANTOT.GT.GS) THEN
	      CALL HIJCSC(JP,JT)
	      GO TO 70
C			********perform elastic collisions
	   ENDIF
 65	   SCIP(JP,JT)=R2
	   RNIP(JP,JT)=RANTOT
	   SJIP(JP,JT)=HINT1(18)
	   NCOLT=NCOLT+1
	   IPCOL(NCOLT)=JP
	   ITCOL(NCOLT)=JT
70	CONTINUE
C		********total number interactions proj and targ has
C				suffered
	IF(NCOLT.EQ.0) THEN
	   NLOP=NLOP+1
           IF(NLOP.LE.20.OR.
     &           (IHNT2(1).EQ.1.AND.IHNT2(3).EQ.1)) GO TO 60
           RETURN
	ENDIF
C               ********At large impact parameter, there maybe no
C                       interaction at all. For NN collision
C                       repeat the event until interaction happens
C
	IF(IHPR2(3).NE.0) THEN
	   NHARD=1+INT(RLU(NSEED)*(NCOLT-1)+0.5)
	   NHARD=MIN(NHARD,NCOLT)
	   JPHARD=IPCOL(NHARD)
	   JTHARD=ITCOL(NHARD)
	ENDIF
C
	IF(IHPR2(9).EQ.1) THEN
		NMINI=1+INT(RLU(NSEED)*(NCOLT-1)+0.5)
		NMINI=MIN(NMINI,NCOLT)
		JPMINI=IPCOL(NMINI)
		JTMINI=ITCOL(NMINI)
	ENDIF
C		********Specifying the location of the hard and
C			minijet if they are enforced by user
C
	DO 200 JP=1,IHNT2(1)
	DO 200 JT=1,IHNT2(3)
	IF(SCIP(JP,JT).EQ.-1.0) GO TO 200
		NFP(JP,11)=NFP(JP,11)+1
		NFT(JT,11)=NFT(JT,11)+1
	IF(NFP(JP,5).LE.1 .AND. NFT(JT,5).GT.1) THEN
		NP=NP+1
		N01=N01+1
	ELSE IF(NFP(JP,5).GT.1 .AND. NFT(JT,5).LE.1) THEN
		NT=NT+1
		N10=N10+1
	ELSE IF(NFP(JP,5).LE.1 .AND. NFT(JT,5).LE.1) THEN
		NP=NP+1
		NT=NT+1
		N0=N0+1
	ELSE IF(NFP(JP,5).GT.1 .AND. NFT(JT,5).GT.1) THEN
		N11=N11+1
	ENDIF
c
	JOUT=0
	NFP(JP,10)=0
	NFT(JT,10)=0
C*****************************************************************
	IF(IHPR2(8).EQ.0 .AND. IHPR2(3).EQ.0) GO TO 160
C		********When IHPR2(8)=0 no jets are produced
	IF(NFP(JP,6).LT.0 .OR. NFT(JT,6).LT.0) GO TO 160
C		********jets can not be produced for (JP,JT)
C			because not enough energy avaible for 
C				JP or JT 
	R2=SCIP(JP,JT)
	HINT1(18)=SJIP(JP,JT)
	TT=ROMG(R2)*HINT1(18)/HIPR1(31)
	TTS=HIPR1(30)*ROMG(R2)/HIPR1(31)
	NJET=0
	IF(IHPR2(3).NE.0 .AND. JP.EQ.JPHARD .AND. JT.EQ.JTHARD) THEN
           CALL JETINI(JP,JT,1)
           CALL HIJHRD(JP,JT,0,JFLG,0)
           HINT1(26)=HINT1(47)
           HINT1(27)=HINT1(48)
           HINT1(28)=HINT1(49)
           HINT1(29)=HINT1(50)
           HINT1(36)=HINT1(67)
           HINT1(37)=HINT1(68)
           HINT1(38)=HINT1(69)
           HINT1(39)=HINT1(70)
C
	   IF(ABS(HINT1(46)).GT.HIPR1(11).AND.JFLG.EQ.2) NFP(JP,7)=1
	   IF(ABS(HINT1(56)).GT.HIPR1(11).AND.JFLG.EQ.2) NFT(JT,7)=1
	   IF(MAX(ABS(HINT1(46)),ABS(HINT1(56))).GT.HIPR1(11).AND.
     &				JFLG.GE.3) IASG(NSG,3)=1
	   IHNT2(9)=IHNT2(14)
	   IHNT2(10)=IHNT2(15)
	   DO 105 I05=1,5
	      HINT1(20+I05)=HINT1(40+I05)
	      HINT1(30+I05)=HINT1(50+I05)
 105	   CONTINUE
	   JOUT=1
	   IF(IHPR2(8).EQ.0) GO TO 160
	   RRB1=MIN((YP(1,JP)**2+YP(2,JP)**2)/1.2**2
     &		/REAL(IHNT2(1))**0.6666667,1.0)
	   RRB2=MIN((YT(1,JT)**2+YT(2,JT)**2)/1.2**2
     &		/REAL(IHNT2(3))**0.6666667,1.0)
	   APHX1=HIPR1(6)*4.0/3.0*(IHNT2(1)**0.3333333-1.0)
     &           *SQRT(1.0-RRB1)
	   APHX2=HIPR1(6)*4.0/3.0*(IHNT2(3)**0.3333333-1.0)
     &           *SQRT(1.0-RRB2)
	   HINT1(65)=HINT1(61)-APHX1*HINT1(62)
     &			-APHX2*HINT1(63)+APHX1*APHX2*HINT1(64)
	   TTRIG=ROMG(R2)*HINT1(65)/HIPR1(31)
	   NJET=-1
C		********subtract the trigger jet from total number
C			of jet production  to be done since it has
C				already been produced here
	   XR1=-ALOG(EXP(-TTRIG)+RLU(NSEED)*(1.0-EXP(-TTRIG)))
 106	   NJET=NJET+1
	   XR1=XR1-ALOG(max(RLU(NSEED),1.0e-20))
	   IF(XR1.LT.TTRIG) GO TO 106
	   XR=0.0
 107	   NJET=NJET+1
	   XR=XR-ALOG(max(RLU(NSEED),1.0e-20))
	   IF(XR.LT.TT-TTRIG) GO TO 107
	   NJET=NJET-1
	   GO TO 112
	ENDIF
C		********create a hard interaction with specified P_T
c				 when IHPR2(3)>0
	IF(IHPR2(9).EQ.1.AND.JP.EQ.JPMINI.AND.JT.EQ.JTMINI) GO TO 110
C		********create at least one pair of mini jets 
C			when IHPR2(9)=1
C
	IF(IHPR2(8).GT.0 .AND.RNIP(JP,JT).LT.EXP(-TT)*
     &		(1.0-EXP(-TTS))) GO TO 160
C		********this is the probability for no jet production
 110    XTMP=EXP(-TT)
        XR=-ALOG(XTMP+HIJRAN(NSEED)*(1.0-XTMP))
111	NJET=NJET+1
	XR=XR-ALOG(max(RLU(NSEED),1.0e-20))
	IF(XR.LT.TT) GO TO 111
112	NJET=MIN(NJET,IHPR2(8))
	IF(IHPR2(8).LT.0)  NJET=ABS(IHPR2(8))
C		******** Determine number of mini jet production
C
	DO 150 I_JET=1,NJET
           CALL JETINI(JP,JT,0)
	   CALL HIJHRD(JP,JT,JOUT,JFLG,1)
C		********JFLG=1 jets valence quarks, JFLG=2 with 
C			gluon jet, JFLG=3 with q-qbar prod for
C			(JP,JT). If JFLG=0 jets can not be produced 
C			this time. If JFLG=-1, error occured abandon
C			this event. JOUT is the total hard scat for
C			(JP,JT) up to now.
	   IF(JFLG.EQ.0) GO TO 160
	   IF(JFLG.LT.0) THEN
	      IF(IHPR2(10).NE.0) WRITE(6,*) 'error occured in HIJHRD'
	      GO TO 50
	   ENDIF
	   JOUT=JOUT+1
	   IF(ABS(HINT1(46)).GT.HIPR1(11).AND.JFLG.EQ.2) NFP(JP,7)=1
	   IF(ABS(HINT1(56)).GT.HIPR1(11).AND.JFLG.EQ.2) NFT(JT,7)=1
	   IF(MAX(ABS(HINT1(46)),ABS(HINT1(56))).GT.HIPR1(11).AND.
     &			JFLG.GE.3) IASG(NSG,3)=1
C		******** jet with PT>HIPR1(11) will be quenched
 150	CONTINUE
 160	CONTINUE
	CALL HIJSFT(JP,JT,JOUT,IERROR)
	IF(IERROR.NE.0) THEN
	   IF(IHPR2(10).NE.0) WRITE(6,*) 'error occured in HIJSFT'
	   GO TO 50
	ENDIF
C
C		********conduct soft scattering between JP and JT
	JATT=JATT+JOUT

200	CONTINUE
c
c**************************
c
	DO 201 JP=1,IHNT2(1)
           IF(NFP(JP,5).GT.2) THEN
              NINP=NINP+1
           ELSE IF(NFP(JP,5).EQ.2.OR.NFP(JP,5).EQ.1) THEN
              NELP=NELP+1
           ENDIF
 201    continue
	DO 202 JT=1,IHNT2(3)
           IF(NFT(JT,5).GT.2) THEN
              NINT=NINT+1
           ELSE IF(NFT(JT,5).EQ.2.OR.NFT(JT,5).EQ.1) THEN
              NELT=NELT+1
           ENDIF
 202    continue
c     
c*******************************


C********perform jet quenching for jets with PT>HIPR1(11)**********

	IF((IHPR2(8).NE.0.OR.IHPR2(3).NE.0).AND.IHPR2(4).GT.0.AND.
     &			IHNT2(1).GT.1.AND.IHNT2(3).GT.1) THEN
		DO 271 I=1,IHNT2(1)
			IF(NFP(I,7).EQ.1) CALL QUENCH(I,1)
271		CONTINUE
		DO 272 I=1,IHNT2(3)
			IF(NFT(I,7).EQ.1) CALL QUENCH(I,2)
272		CONTINUE
		DO 273 ISG=1,NSG
			IF(IASG(ISG,3).EQ.1) CALL QUENCH(ISG,3)
273		CONTINUE
	ENDIF
C
C**************fragment all the string systems in the following*****
C
C********N_ST is where particle information starts
C********N_STR+1 is the number of strings in fragmentation
C********the number of strings before a line is stored in K(I,4)
C********IDSTR is id number of the string system (91,92 or 93)
C
        IF(IHPR2(20).NE.0) THEN
	   DO 360 ISG=1,NSG
		CALL HIJFRG(ISG,3,IERROR)
        	IF(MSTU(24).NE.0 .OR.IERROR.GT.0) THEN
		   MSTU(24)=0
		   MSTU(28)=0
		   IF(IHPR2(10).NE.0) THEN
		      call lulist(1)
		      WRITE(6,*) 'error occured, repeat the event'
		   ENDIF
		   GO TO 50
		ENDIF
C			********Check errors
C
		N_ST=1
		IDSTR=92
		IF(IHPR2(21).EQ.0) THEN
		   CALL LUEDIT(2)
C     Dont perform the search for string if N=1 because it will search
C     beyond the end of the valid particle list
                ELSE IF(N.GT.1) THEN
351		   N_ST=N_ST+1
C     Check for inconsistency -- no string line found
                   IF(N_ST.GT.N) THEN
                      IERRSTAT=2
                      RETURN
                   ENDIF

		   IF(K(N_ST,2).LT.91.OR.K(N_ST,2).GT.93) GO TO  351
		   IDSTR=K(N_ST,2)
		   N_ST=N_ST+1
		ENDIF
C
		IF(FRAME.EQ.'LAB') THEN
			CALL HIBOOST
		ENDIF
C		******** boost back to lab frame(if it was in)
C
		N_STR=0
		DO 360 I=N_ST,N
		   IF(K(I,2).EQ.IDSTR) THEN
                      IF(K(I,3).LT.N_ST) THEN
                         N_STR=N_STR+1
                         GO TO 360
                      ENDIF 
		   ENDIF
		   K(I,4)=N_STR
		   NATT=NATT+1
C     Add a check on array overflow
                   IF(NATT.GT.130000) THEN
                      IERRSTAT=1
                      RETURN
                   ENDIF
		   KATT(NATT,1)=K(I,2)
		   KATT(NATT,2)=20
		   KATT(NATT,4)=K(I,1)
		   IF(K(I,3).EQ.0 .OR. K(I,3).LT.N_ST .OR.
     &                  (K(K(I,3),2).EQ.IDSTR .AND. 
     &                  K(K(I,3),3).LT.N_ST)) THEN
		      KATT(NATT,3)=0
		   ELSE
		      KATT(NATT,3)=NATT-I+K(I,3)+N_STR-K(K(I,3),4)
		   ENDIF
C       ****** identify the mother particle
		   PATT(NATT,1)=P(I,1)
		   PATT(NATT,2)=P(I,2)
		   PATT(NATT,3)=P(I,3)
		   PATT(NATT,4)=P(I,4)
		   EATT=EATT+P(I,4)
                   VATT(NATT,1)=V(I,1)
                   VATT(NATT,2)=V(I,2)
                   VATT(NATT,3)=V(I,3)
                   VATT(NATT,4)=V(I,4)
360	   CONTINUE
C		********Fragment the q-qbar jets systems *****
C
	   JTP(1)=IHNT2(1)
	   JTP(2)=IHNT2(3)
	   DO 400 NTP=1,2
	   DO 400 J_JTP=1,JTP(NTP)
		CALL HIJFRG(J_JTP,NTP,IERROR)
        	IF(MSTU(24).NE.0 .OR. IERROR.GT.0) THEN
		   MSTU(24)=0
		   MSTU(28)=0
		   IF(IHPR2(10).NE.0) THEN
		      call lulist(1)
		      WRITE(6,*) 'error occured, repeat the event'
		   ENDIF
		   GO TO 50
		ENDIF
C			********check errors
C
		N_ST=1
		IDSTR=92
		IF(IHPR2(21).EQ.0) THEN
		   CALL LUEDIT(2)
C     Dont perform the search for string if N=1 because it will search
C     beyond the end of the valid particle list
                ELSE IF(N.GT.1) THEN
381		   N_ST=N_ST+1
C     Check for inconsistency -- no string line found
                   IF(N_ST.GT.N) THEN
                      IERRSTAT=2
                      RETURN
                   ENDIF
		   IF(K(N_ST,2).LT.91.OR.K(N_ST,2).GT.93) GO TO  381
		   IDSTR=K(N_ST,2)
		   N_ST=N_ST+1
		ENDIF
		IF(FRAME.EQ.'LAB') THEN
			CALL HIBOOST
		ENDIF
C		******** boost back to lab frame(if it was in)
C
		NFTP=NFP(J_JTP,5)
		IF(NTP.EQ.2) NFTP=10+NFT(J_JTP,5)
		N_STR=0
		DO 390 I=N_ST,N
		   IF(K(I,2).EQ.IDSTR) THEN
                      IF(K(I,3).LT.N_ST) THEN
                         N_STR=N_STR+1
                         GO TO 390
                      ENDIF
		   ENDIF
		   K(I,4)=N_STR
		   NATT=NATT+1
C     Add a check on array overflow
                   IF(NATT.GT.130000) THEN
                      IERRSTAT=1
                      RETURN
                   ENDIF
		   KATT(NATT,1)=K(I,2)
		   KATT(NATT,2)=NFTP
		   KATT(NATT,4)=K(I,1)
		   IF(K(I,3).EQ.0 .OR. K(I,3).LT.N_ST .OR.
     &                  (K(K(I,3),2).EQ.IDSTR .AND. 
     &                  K(K(I,3),3).LT.N_ST)) THEN
		      KATT(NATT,3)=0
		   ELSE
		      KATT(NATT,3)=NATT-I+K(I,3)+N_STR-K(K(I,3),4)
		   ENDIF
C       ****** identify the mother particle
		   PATT(NATT,1)=P(I,1)
		   PATT(NATT,2)=P(I,2)
		   PATT(NATT,3)=P(I,3)
		   PATT(NATT,4)=P(I,4)
		   EATT=EATT+P(I,4)
                   VATT(NATT,1)=V(I,1)
                   VATT(NATT,2)=V(I,2)
                   VATT(NATT,3)=V(I,3)
                   VATT(NATT,4)=V(I,4)
390		CONTINUE 
400	   CONTINUE
C		********Fragment the q-qq related string systems
	ENDIF

	DO 450 I=1,NDR
		NATT=NATT+1
C   Add a check on array overflow
                IF(NATT.GT.130000) THEN
                   IERRSTAT=1
                   RETURN
                ENDIF
		KATT(NATT,1)=KFDR(I)
		KATT(NATT,2)=40
		KATT(NATT,3)=0
		PATT(NATT,1)=PDR(I,1)
		PATT(NATT,2)=PDR(I,2)
		PATT(NATT,3)=PDR(I,3)
		PATT(NATT,4)=PDR(I,4)
		EATT=EATT+PDR(I,4)
                VATT(NATT,1)=VDR(I,1)
                VATT(NATT,2)=VDR(I,2)
                VATT(NATT,3)=VDR(I,3)
                VATT(NATT,4)=VDR(I,4)
450	CONTINUE
C			********store the direct-produced particles
C
	DENGY=EATT/(IHNT2(1)*HINT1(6)+IHNT2(3)*HINT1(7))-1.0
	IF(ABS(DENGY).GT.HIPR1(43).AND.IHPR2(20).NE.0
     &     .AND.IHPR2(21).EQ.0) THEN
	IF(IHPR2(10).NE.0) WRITE(6,*) 'Energy not conserved, '//
     &          'repeat the event'
C		call lulist(1)
		GO TO 50
	ENDIF
	RETURN
	END
C
C
C
	SUBROUTINE HIJSET(EFRM,FRAME,PROJ,TARG,IAP,IZP,IAT,IZT)
	CHARACTER FRAME*4,PROJ*4,TARG*4,EFRAME*4
	DOUBLE PRECISION  DD1,DD2,DD3,DD4
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
        COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
	EXTERNAL FNKICK,FNKICK2,FNSTRU,FNSTRUM,FNSTRUS
        SAVE

	CALL TITLE
	IHNT2(1)=IAP
	IHNT2(2)=IZP
	IHNT2(3)=IAT
	IHNT2(4)=IZT
	IHNT2(5)=0
	IHNT2(6)=0
C
	HINT1(8)=MAX(ULMASS(2112),ULMASS(2212))
	HINT1(9)=HINT1(8)
C
	IF(PROJ.NE.'A') THEN
		IF(PROJ.EQ.'P') THEN
		    IHNT2(5)=2212
		ELSE IF(PROJ.EQ.'PBAR') THEN 
		    IHNT2(5)=-2212
		ELSE IF(PROJ.EQ.'PI+') THEN
		    IHNT2(5)=211
		ELSE IF(PROJ.EQ.'PI-') THEN
		    IHNT2(5)=-211
		ELSE IF(PROJ.EQ.'K+') THEN
		    IHNT2(5)=321
		ELSE IF(PROJ.EQ.'K-') THEN
		    IHNT2(5)=-321
		ELSE IF(PROJ.EQ.'N') THEN
		    IHNT2(5)=2112
		ELSE IF(PROJ.EQ.'NBAR') THEN
		    IHNT2(5)=-2112
		ELSE
		    WRITE(6,*) PROJ, 'wrong or unavailable proj name'
		    STOP
		ENDIF
		HINT1(8)=ULMASS(IHNT2(5))
	ENDIF
	IF(TARG.NE.'A') THEN
		IF(TARG.EQ.'P') THEN
		    IHNT2(6)=2212
		ELSE IF(TARG.EQ.'PBAR') THEN 
		    IHNT2(6)=-2212
		ELSE IF(TARG.EQ.'PI+') THEN
		    IHNT2(6)=211
		ELSE IF(TARG.EQ.'PI-') THEN
		    IHNT2(6)=-211
		ELSE IF(TARG.EQ.'K+') THEN
		    IHNT2(6)=321
		ELSE IF(TARG.EQ.'K-') THEN
		    IHNT2(6)=-321
		ELSE IF(TARG.EQ.'N') THEN
		    IHNT2(6)=2112
		ELSE IF(TARG.EQ.'NBAR') THEN
		    IHNT2(6)=-2112
		ELSE
		    WRITE(6,*) TARG,'wrong or unavailable targ name'
		    STOP
		ENDIF
		HINT1(9)=ULMASS(IHNT2(6))
	ENDIF

C...Switch off decay of pi0, K0S, Lambda, Sigma+-, Xi0-, Omega-.
C... Does nothing because we changed pyset, to do this there - Matt Nguyen, Dec 15 2012
	IF(IHPR2(12).GT.0) THEN
           CALL LUGIVE('MDCY(C111,1)=0')
           CALL LUGIVE('MDCY(C310,1)=0')
           CALL LUGIVE('MDCY(C411,1)=0;MDCY(C-411,1)=0')
           CALL LUGIVE('MDCY(C421,1)=0;MDCY(C-421,1)=0')
           CALL LUGIVE('MDCY(C431,1)=0;MDCY(C-431,1)=0')
           CALL LUGIVE('MDCY(C511,1)=0;MDCY(C-511,1)=0')
           CALL LUGIVE('MDCY(C521,1)=0;MDCY(C-521,1)=0')
           CALL LUGIVE('MDCY(C531,1)=0;MDCY(C-531,1)=0')
           CALL LUGIVE('MDCY(C3122,1)=0;MDCY(C-3122,1)=0')
           CALL LUGIVE('MDCY(C3112,1)=0;MDCY(C-3112,1)=0')
           CALL LUGIVE('MDCY(C3212,1)=0;MDCY(C-3212,1)=0')
           CALL LUGIVE('MDCY(C3222,1)=0;MDCY(C-3222,1)=0')
           CALL LUGIVE('MDCY(C3312,1)=0;MDCY(C-3312,1)=0')
           CALL LUGIVE('MDCY(C3322,1)=0;MDCY(C-3322,1)=0')
           CALL LUGIVE('MDCY(C3334,1)=0;MDCY(C-3334,1)=0')
	ENDIF
	MSTU(12)=0
	MSTU(21)=1
	IF(IHPR2(10).EQ.0) THEN
		MSTU(22)=0
		MSTU(25)=0
		MSTU(26)=0
	ENDIF
	MSTJ(12)=IHPR2(11)
	PARJ(21)=HIPR1(2)
	PARJ(41)=HIPR1(3)
	PARJ(42)=HIPR1(4)
C			******** set up for jetset
	IF(FRAME.EQ.'LAB') THEN
	   DD1=EFRM
	   DD2=HINT1(8)
	   DD3=HINT1(9)
	   HINT1(1)=SQRT(HINT1(8)**2+2.0*HINT1(9)*EFRM+HINT1(9)**2)
	   DD4=DSQRT(DD1**2-DD2**2)/(DD1+DD3)
	   HINT1(2)=DD4
	   HINT1(3)=0.5*DLOG((1.D0+DD4)/(1.D0-DD4))
	   DD4=DSQRT(DD1**2-DD2**2)/DD1
	   HINT1(4)=0.5*DLOG((1.D0+DD4)/(1.D0-DD4))
	   HINT1(5)=0.0
	   HINT1(6)=EFRM
	   HINT1(7)=HINT1(9)
	ELSE IF(FRAME.EQ.'CMS') THEN
	   HINT1(1)=EFRM
	   HINT1(2)=0.0
	   HINT1(3)=0.0
	   DD1=HINT1(1)
	   DD2=HINT1(8)
	   DD3=HINT1(9)
	   DD4=DSQRT(1.D0-4.D0*DD2**2/DD1**2)
	   HINT1(4)=0.5*DLOG((1.D0+DD4)/(1.D0-DD4))
	   DD4=DSQRT(1.D0-4.D0*DD3**2/DD1**2)
	   HINT1(5)=-0.5*DLOG((1.D0+DD4)/(1.D0-DD4))
	   HINT1(6)=HINT1(1)/2.0
	   HINT1(7)=HINT1(1)/2.0
	ENDIF
C		********define Lorentz transform to lab frame
c
C		********calculate the cross sections involved with
C			nucleon collisions.
	IF(IHNT2(1).GT.1) THEN
		CALL HIJWDS(IHNT2(1),1,RMAX)
		HIPR1(34)=RMAX
C			********set up Wood-Sax distr for proj.
	ENDIF
	IF(IHNT2(3).GT.1) THEN
		CALL HIJWDS(IHNT2(3),2,RMAX)
		HIPR1(35)=RMAX
C			********set up Wood-Sax distr for  targ.
	ENDIF
C
C
	I=0
20	I=I+1
	IF(I.EQ.10) GO TO 30
	IF(HIDAT0(10,I).LE.HINT1(1)) GO TO 20
30	IF(I.EQ.1) I=2
	DO 40 J=1,9
	   HIDAT(J)=HIDAT0(J,I-1)+(HIDAT0(J,I)-HIDAT0(J,I-1))
     &	   *(HINT1(1)-HIDAT0(10,I-1))/(HIDAT0(10,I)-HIDAT0(10,I-1))
40	CONTINUE
	HIPR1(31)=HIDAT(5)
	HIPR1(30)=2.0*HIDAT(5)
C
C
	CALL HIJCRS
C
	IF(IHPR2(5).NE.0) THEN
		CALL HIFUN(3,0.0,36.0,FNKICK)
C		********booking for generating pt**2 for pt kick
	ENDIF
	CALL HIFUN(7,0.0,6.0,FNKICK2)
	CALL HIFUN(4,0.0,1.0,FNSTRU)
	CALL HIFUN(5,0.0,1.0,FNSTRUM)
	CALL HIFUN(6,0.0,1.0,FNSTRUS)
C		********booking for x distribution of valence quarks
	EFRAME='Ecm'
	IF(FRAME.EQ.'LAB') EFRAME='Elab'
	WRITE(6,100) EFRAME,EFRM,PROJ,IHNT2(1),IHNT2(2),
     &               TARG,IHNT2(3),IHNT2(4) 
100     FORMAT(//10X,'****************************************
     &  **********'/
     &  10X,'*',48X,'*'/
     &  10X,'*         HIJING has been initialized at         *'/
     &  10X,'*',13X,A4,'= ',F10.2,' GeV/n',13X,'*'/
     &  10X,'*',48X,'*'/
     &  10X,'*',8X,'for ',
     &  A4,'(',I3,',',I3,')',' + ',A4,'(',I3,',',I3,')',7X,'*'/
     &  10X,'**************************************************')
        RETURN
        END
C
C
C
	FUNCTION FNKICK(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	FNKICK=1.0/(X+HIPR1(19)**2)/(X+HIPR1(20)**2)
     &		/(1+EXP((SQRT(X)-HIPR1(20))/0.4))
	RETURN
	END
C
C
	FUNCTION FNKICK2(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	FNKICK2=X*EXP(-2.0*X/HIPR1(42))
	RETURN
	END
C
C
C
	FUNCTION FNSTRU(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	FNSTRU=(1.0-X)**HIPR1(44)/
     &		(X**2+HIPR1(45)**2/HINT1(1)**2)**HIPR1(46)
	RETURN
	END
C
C
C
	FUNCTION FNSTRUM(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	FNSTRUM=1.0/((1.0-X)**2+HIPR1(45)**2/HINT1(1)**2)**HIPR1(46)
     &          /(X**2+HIPR1(45)**2/HINT1(1)**2)**HIPR1(46)
	RETURN
	END
C
C
	FUNCTION FNSTRUS(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	FNSTRUS=(1.0-X)**HIPR1(47)/
     &		(X**2+HIPR1(45)**2/HINT1(1)**2)**HIPR1(48)
	RETURN
	END
C
C
C
C
	SUBROUTINE HIBOOST
      	IMPLICIT DOUBLE PRECISION(D)  
      	COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5) 
      	COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	DO 100 I=1,N
	   DBETA=P(I,3)/P(I,4)
	   IF(ABS(DBETA).GE.1.D0) THEN
	      DB=HINT1(2)
	      IF(DB.GT.0.99999999D0) THEN 
C		********Rescale boost vector if too close to unity. 
		 WRITE(6,*) '(HIBOOT:) boost vector too large' 
		 DB=0.99999999D0
	      ENDIF 
	      DGA=1D0/SQRT(1D0-DB**2)
	      DP3=P(I,3)
	      DP4=P(I,4)
	      P(I,3)=(DP3+DB*DP4)*DGA  
	      P(I,4)=(DP4+DB*DP3)*DGA  
	      GO TO 100
	   ENDIF
	   Y=0.5*DLOG((1.D0+DBETA)/(1.D0-DBETA))
	   AMT=SQRT(P(I,1)**2+P(I,2)**2+P(I,5)**2)
	   P(I,3)=AMT*SINH(Y+HINT1(3))
	   P(I,4)=AMT*COSH(Y+HINT1(3))
100	CONTINUE
	RETURN
	END
C
C
C
C
	SUBROUTINE QUENCH(JPJT,NTP)
	DIMENSION RDP(300),LQP(300),RDT(300),LQT(300)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
C
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/RANSEED/NSEED
	SAVE  
C
        BB=HINT1(19)                                            ! Uzhi
        PHI=HINT1(20)                                           ! Uzhi
        BBX=BB*COS(PHI)                                         ! Uzhi
        BBY=BB*SIN(PHI)                                         ! Uzhi
c
	IF(NTP.EQ.2) GO TO 400
	IF(NTP.EQ.3) GO TO 2000 
C*******************************************************
C Jet interaction for proj jet in the direction PHIP
C******************************************************
C
	IF(NFP(JPJT,7).NE.1) RETURN

	JP=JPJT
	DO 290 I=1,NPJ(JP)
	   PTJET0=SQRT(PJPX(JP,I)**2+PJPY(JP,I)**2)
	   IF(PTJET0.LE.HIPR1(11)) GO TO 290
	   PTOT=SQRT(PTJET0*PTJET0+PJPZ(JP,I)**2)
	   IF(PTOT.LT.HIPR1(8)) GO TO 290
	   PHIP=ULANGL(PJPX(JP,I),PJPY(JP,I))
C******* find the wounded proj which can interact with jet***
	   KP=0
	   DO 100 I2=1,IHNT2(1)
	      IF(NFP(I2,5).NE.3 .OR. I2.EQ.JP) GO TO 100
	      DX=YP(1,I2)-YP(1,JP)
	      DY=YP(2,I2)-YP(2,JP)
	      PHI=ULANGL(DX,DY)
	      DPHI=ABS(PHI-PHIP)
              IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI      ! Uzhi
	      IF(DPHI.GE.HIPR1(40)/2.0) GO TO 100
	      RD0=SQRT(DX*DX+DY*DY)
	      IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 100
	      KP=KP+1
	      LQP(KP)=I2
	      RDP(KP)=COS(DPHI)*RD0
 100	   CONTINUE
C*******	rearrange according decending rd************
	   DO 110 I2=1,KP-1
	      DO 110 J2=I2+1,KP
		 IF(RDP(I2).LT.RDP(J2)) GO TO 110
		 RD=RDP(I2)
		 LQ=LQP(I2)
		 RDP(I2)=RDP(J2)
		 LQP(I2)=LQP(J2)
		 RDP(J2)=RD
		 LQP(J2)=LQ
 110	      CONTINUE
C****** find wounded targ which can interact with jet********
	      KT=0
	      DO 120 I2=1,IHNT2(3)
		 IF(NFT(I2,5).NE.3) GO TO 120
		 DX=YT(1,I2)-YP(1,JP)-BBX
		 DY=YT(2,I2)-YP(2,JP)-BBY
		 PHI=ULANGL(DX,DY)
		 DPHI=ABS(PHI-PHIP)
                 IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI   ! Uzhi
		 IF(DPHI.GT.HIPR1(40)/2.0) GO TO 120
		 RD0=SQRT(DX*DX+DY*DY)
		 IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 120
		 KT=KT+1
		 LQT(KT)=I2
		 RDT(KT)=COS(DPHI)*RD0
 120	      CONTINUE
C*******	rearrange according decending rd************
	      DO 130 I2=1,KT-1
		 DO 130 J2=I2+1,KT
		    IF(RDT(I2).LT.RDT(J2)) GO TO 130
		    RD=RDT(I2)
		    LQ=LQT(I2)
		    RDT(I2)=RDT(J2)
		    LQT(I2)=LQT(J2)
		    RDT(J2)=RD
		    LQT(J2)=LQ
 130		 CONTINUE
		
		 MP=0
		 MT=0
		 R0=0.0
		 NQ=0
		 DP=0.0
		 PTOT=SQRT(PJPX(JP,I)**2+PJPY(JP,I)**2+PJPZ(JP,I)**2)
		 V1=PJPX(JP,I)/PTOT
		 V2=PJPY(JP,I)/PTOT
		 V3=PJPZ(JP,I)/PTOT

 200		 RN=RLU(NSEED)
 210		 IF(MT.GE.KT .AND. MP.GE.KP) GO TO 290
		 IF(MT.GE.KT) GO TO 220
		 IF(MP.GE.KP) GO TO 240
		 IF(RDP(MP+1).GT.RDT(MT+1)) GO TO 240
 220		 MP=MP+1
		 DRR=RDP(MP)-R0
		 IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 210
		 DP=DRR*HIPR1(14)
		 IF(KFPJ(JP,I).NE.21) DP=0.5*DP
C	********string tension of quark jet is 0.5 of gluon's 
		 IF(DP.LE.0.2) GO TO 210
		 IF(PTOT.LE.0.4) GO TO 290
		 IF(PTOT.LE.DP) DP=PTOT-0.2
		 DE=DP

		 IF(KFPJ(JP,I).NE.21) THEN
		    PRSHU=PP(LQP(MP),1)**2+PP(LQP(MP),2)**2
     &                   +PP(LQP(MP),3)**2
		    DE=SQRT(PJPM(JP,I)**2+PTOT**2)
     &			-SQRT(PJPM(JP,I)**2+(PTOT-DP)**2)
		    ERSHU=(PP(LQP(MP),4)+DE-DP)**2
		    AMSHU=ERSHU-PRSHU
		    IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 210
		    PP(LQP(MP),4)=SQRT(ERSHU)
		    PP(LQP(MP),5)=SQRT(AMSHU)
		 ENDIF
C		********reshuffle the energy when jet has mass
		 R0=RDP(MP)
		 DP1=DP*V1
		 DP2=DP*V2
		 DP3=DP*V3
C		********momentum and energy transfer from jet
		 
		 NPJ(LQP(MP))=NPJ(LQP(MP))+1
		 KFPJ(LQP(MP),NPJ(LQP(MP)))=21
		 PJPX(LQP(MP),NPJ(LQP(MP)))=DP1
		 PJPY(LQP(MP),NPJ(LQP(MP)))=DP2
		 PJPZ(LQP(MP),NPJ(LQP(MP)))=DP3
		 PJPE(LQP(MP),NPJ(LQP(MP)))=DP
		 PJPM(LQP(MP),NPJ(LQP(MP)))=0.0
		 GO TO 260

 240		 MT=MT+1
		 DRR=RDT(MT)-R0
		 IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 210
		 DP=DRR*HIPR1(14)
		 IF(DP.LE.0.2) GO TO 210
		 IF(PTOT.LE.0.4) GO TO 290
		 IF(PTOT.LE.DP) DP=PTOT-0.2
		 DE=DP

		 IF(KFPJ(JP,I).NE.21) THEN
		    PRSHU=PT(LQT(MT),1)**2+PT(LQT(MT),2)**2
     &                   +PT(LQT(MT),3)**2
		    DE=SQRT(PJPM(JP,I)**2+PTOT**2)
     &			-SQRT(PJPM(JP,I)**2+(PTOT-DP)**2)
		    ERSHU=(PT(LQT(MT),4)+DE-DP)**2
		    AMSHU=ERSHU-PRSHU
		    IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 210
		    PT(LQT(MT),4)=SQRT(ERSHU)
		    PT(LQT(MT),5)=SQRT(AMSHU)
		 ENDIF
C		********reshuffle the energy when jet has mass

		 R0=RDT(MT)
		 DP1=DP*V1
		 DP2=DP*V2
		 DP3=DP*V3
C		********momentum and energy transfer from jet
		 NTJ(LQT(MT))=NTJ(LQT(MT))+1
		 KFTJ(LQT(MT),NTJ(LQT(MT)))=21
		 PJTX(LQT(MT),NTJ(LQT(MT)))=DP1
		 PJTY(LQT(MT),NTJ(LQT(MT)))=DP2
		 PJTZ(LQT(MT),NTJ(LQT(MT)))=DP3
		 PJTE(LQT(MT),NTJ(LQT(MT)))=DP
		 PJTM(LQT(MT),NTJ(LQT(MT)))=0.0

 260		 PJPX(JP,I)=(PTOT-DP)*V1
		 PJPY(JP,I)=(PTOT-DP)*V2
		 PJPZ(JP,I)=(PTOT-DP)*V3
		 PJPE(JP,I)=PJPE(JP,I)-DE

		 PTOT=PTOT-DP
		 NQ=NQ+1
		 GO TO 200
 290	      CONTINUE

	      RETURN

C*******************************************************
C Jet interaction for target jet in the direction PHIT
C******************************************************
C
C******* find the wounded proj which can interact with jet***

 400	      IF(NFT(JPJT,7).NE.1) RETURN
	      JT=JPJT
	      DO 690 I=1,NTJ(JT)
		 PTJET0=SQRT(PJTX(JT,I)**2+PJTY(JT,I)**2)
		 IF(PTJET0.LE.HIPR1(11)) GO TO 690
		 PTOT=SQRT(PTJET0*PTJET0+PJTZ(JT,I)**2)
		 IF(PTOT.LT.HIPR1(8)) GO TO 690
		 PHIT=ULANGL(PJTX(JT,I),PJTY(JT,I))
		 KP=0
		 DO 500 I2=1,IHNT2(1)
		    IF(NFP(I2,5).NE.3) GO TO 500
		    DX=YP(1,I2)+BBX-YT(1,JT)
		    DY=YP(2,I2)+BBY-YT(2,JT)
		    PHI=ULANGL(DX,DY)
		    DPHI=ABS(PHI-PHIT)
                    IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI ! Uzhi
		    IF(DPHI.GT.HIPR1(40)/2.0) GO TO 500
		    RD0=SQRT(DX*DX+DY*DY)
		    IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 500
		    KP=KP+1
		    LQP(KP)=I2
		    RDP(KP)=COS(DPHI)*RD0
 500		 CONTINUE
C*******	rearrange according to decending rd************
		 DO 510 I2=1,KP-1
		    DO 510 J2=I2+1,KP
		       IF(RDP(I2).LT.RDP(J2)) GO TO 510
		       RD=RDP(I2)
		       LQ=LQP(I2)
		       RDP(I2)=RDP(J2)
		       LQP(I2)=LQP(J2)
		       RDP(J2)=RD
		       LQP(J2)=LQ
 510		    CONTINUE
C****** find wounded targ which can interact with jet********
		    KT=0
		    DO 520 I2=1,IHNT2(3)
		       IF(NFT(I2,5).NE.3 .OR. I2.EQ.JT) GO TO 520
		       DX=YT(1,I2)-YT(1,JT)
		       DY=YT(2,I2)-YT(2,JT)
		       PHI=ULANGL(DX,DY)
		       DPHI=ABS(PHI-PHIT)
                       IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI ! Uzhi
		       IF(DPHI.GT.HIPR1(40)/2.0) GO TO 520
		       RD0=SQRT(DX*DX+DY*DY)
		       IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 520
		       KT=KT+1
		       LQT(KT)=I2
		       RDT(KT)=COS(DPHI)*RD0
 520		    CONTINUE
C*******	rearrange according to decending rd************
		    DO 530 I2=1,KT-1
		       DO 530 J2=I2+1,KT
			  IF(RDT(I2).LT.RDT(J2)) GO TO 530
			  RD=RDT(I2)
			  LQ=LQT(I2)
			  RDT(I2)=RDT(J2)
			  LQT(I2)=LQT(J2)
			  RDT(J2)=RD
			  LQT(J2)=LQ
 530		       CONTINUE
		       
		       MP=0
		       MT=0
		       NQ=0
		       DP=0.0
		       R0=0.0
		PTOT=SQRT(PJTX(JT,I)**2+PJTY(JT,I)**2+PJTZ(JT,I)**2)
		V1=PJTX(JT,I)/PTOT
		V2=PJTY(JT,I)/PTOT
		V3=PJTZ(JT,I)/PTOT

 600		RN=RLU(NSEED)
 610		IF(MT.GE.KT .AND. MP.GE.KP) GO TO 690
		IF(MT.GE.KT) GO TO 620
		IF(MP.GE.KP) GO TO 640
		IF(RDP(MP+1).GT.RDT(MT+1)) GO TO 640
620		MP=MP+1
		DRR=RDP(MP)-R0
		IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 610
		DP=DRR*HIPR1(14)
		IF(KFTJ(JT,I).NE.21) DP=0.5*DP
C	********string tension of quark jet is 0.5 of gluon's 
		IF(DP.LE.0.2) GO TO 610
		IF(PTOT.LE.0.4) GO TO 690
		IF(PTOT.LE.DP) DP=PTOT-0.2
		DE=DP
C
		IF(KFTJ(JT,I).NE.21) THEN
		   PRSHU=PP(LQP(MP),1)**2+PP(LQP(MP),2)**2
     &                   +PP(LQP(MP),3)**2
		   DE=SQRT(PJTM(JT,I)**2+PTOT**2)
     &		     -SQRT(PJTM(JT,I)**2+(PTOT-DP)**2)
		   ERSHU=(PP(LQP(MP),4)+DE-DP)**2
		   AMSHU=ERSHU-PRSHU
		   IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 610
		   PP(LQP(MP),4)=SQRT(ERSHU)
		   PP(LQP(MP),5)=SQRT(AMSHU)
		ENDIF
C		********reshuffle the energy when jet has mass
C
		R0=RDP(MP)
		DP1=DP*V1
		DP2=DP*V2
		DP3=DP*V3
C		********momentum and energy transfer from jet
		NPJ(LQP(MP))=NPJ(LQP(MP))+1
		KFPJ(LQP(MP),NPJ(LQP(MP)))=21
		PJPX(LQP(MP),NPJ(LQP(MP)))=DP1
		PJPY(LQP(MP),NPJ(LQP(MP)))=DP2
		PJPZ(LQP(MP),NPJ(LQP(MP)))=DP3
		PJPE(LQP(MP),NPJ(LQP(MP)))=DP
		PJPM(LQP(MP),NPJ(LQP(MP)))=0.0

		GO TO 660

640		MT=MT+1
		DRR=RDT(MT)-R0
		IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 610
		DP=DRR*HIPR1(14)
		IF(DP.LE.0.2) GO TO 610
		IF(PTOT.LE.0.4) GO TO 690
		IF(PTOT.LE.DP) DP=PTOT-0.2
		DE=DP

		IF(KFTJ(JT,I).NE.21) THEN
		   PRSHU=PT(LQT(MT),1)**2+PT(LQT(MT),2)**2
     &                   +PT(LQT(MT),3)**2
		   DE=SQRT(PJTM(JT,I)**2+PTOT**2)
     &		     -SQRT(PJTM(JT,I)**2+(PTOT-DP)**2)
		   ERSHU=(PT(LQT(MT),4)+DE-DP)**2
		   AMSHU=ERSHU-PRSHU
		   IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 610
		   PT(LQT(MT),4)=SQRT(ERSHU)
		   PT(LQT(MT),5)=SQRT(AMSHU)
		ENDIF
C		********reshuffle the energy when jet has mass

		R0=RDT(MT)
		DP1=DP*V1
		DP2=DP*V2
		DP3=DP*V3
C		********momentum and energy transfer from jet
		NTJ(LQT(MT))=NTJ(LQT(MT))+1
		KFTJ(LQT(MT),NTJ(LQT(MT)))=21
		PJTX(LQT(MT),NTJ(LQT(MT)))=DP1
		PJTY(LQT(MT),NTJ(LQT(MT)))=DP2
		PJTZ(LQT(MT),NTJ(LQT(MT)))=DP3
		PJTE(LQT(MT),NTJ(LQT(MT)))=DP
		PJTM(LQT(MT),NTJ(LQT(MT)))=0.0

660		PJTX(JT,I)=(PTOT-DP)*V1
		PJTY(JT,I)=(PTOT-DP)*V2
		PJTZ(JT,I)=(PTOT-DP)*V3
		PJTE(JT,I)=PJTE(JT,I)-DE

		PTOT=PTOT-DP
		NQ=NQ+1
		GO TO 600
690	CONTINUE
	RETURN
C********************************************************
C	Q-QBAR jet interaction
C********************************************************
2000	ISG=JPJT
	IF(IASG(ISG,3).NE.1) RETURN
C
	JP=IASG(ISG,1)
	JT=IASG(ISG,2)
	XJ=(YP(1,JP)+BBX+YT(1,JT))/2.0
	YJ=(YP(2,JP)+BBY+YT(2,JT))/2.0
	DO 2690 I=1,NJSG(ISG)
	   PTJET0=SQRT(PXSG(ISG,I)**2+PYSG(ISG,I)**2)
	   IF(PTJET0.LE.HIPR1(11).OR.PESG(ISG,I).LT.HIPR1(1))
     &            GO TO 2690
	   PTOT=SQRT(PTJET0*PTJET0+PZSG(ISG,I)**2)
	   IF(PTOT.LT.MAX(HIPR1(1),HIPR1(8))) GO TO 2690
	   PHIQ=ULANGL(PXSG(ISG,I),PYSG(ISG,I))
	   KP=0
	   DO 2500 I2=1,IHNT2(1)
	      IF(NFP(I2,5).NE.3.OR.I2.EQ.JP) GO TO 2500
	      DX=YP(1,I2)+BBX-XJ
	      DY=YP(2,I2)+BBY-YJ
	      PHI=ULANGL(DX,DY)
	      DPHI=ABS(PHI-PHIQ)
              IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI      ! Uzhi
	      IF(DPHI.GT.HIPR1(40)/2.0) GO TO 2500
	      RD0=SQRT(DX*DX+DY*DY)
	      IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 2500
	      KP=KP+1
	      LQP(KP)=I2
	      RDP(KP)=COS(DPHI)*RD0
 2500	   CONTINUE
C*******	rearrange according to decending rd************
	   DO 2510 I2=1,KP-1
	      DO 2510 J2=I2+1,KP
		 IF(RDP(I2).LT.RDP(J2)) GO TO 2510
		 RD=RDP(I2)
		 LQ=LQP(I2)
		 RDP(I2)=RDP(J2)
		 LQP(I2)=LQP(J2)
		 RDP(J2)=RD
		 LQP(J2)=LQ
 2510	      CONTINUE
C****** find wounded targ which can interact with jet********
	      KT=0
	      DO 2520 I2=1,IHNT2(3)
		 IF(NFT(I2,5).NE.3 .OR. I2.EQ.JT) GO TO 2520
		 DX=YT(1,I2)-XJ
		 DY=YT(2,I2)-YJ
		 PHI=ULANGL(DX,DY)
		 DPHI=ABS(PHI-PHIQ)
                 IF(DPHI.GE.HIPR1(40)) DPHI=2.*HIPR1(40)-DPHI ! Uzhi
		 IF(DPHI.GT.HIPR1(40)/2.0) GO TO 2520
		 RD0=SQRT(DX*DX+DY*DY)
		 IF(RD0*SIN(DPHI).GT.HIPR1(12)) GO TO 2520
		 KT=KT+1
		 LQT(KT)=I2
		 RDT(KT)=COS(DPHI)*RD0
 2520	      CONTINUE
C*******	rearrange according to decending rd************
	      DO 2530 I2=1,KT-1
		 DO 2530 J2=I2+1,KT
		    IF(RDT(I2).LT.RDT(J2)) GO TO 2530
		    RD=RDT(I2)
		    LQ=LQT(I2)
		    RDT(I2)=RDT(J2)
		    LQT(I2)=LQT(J2)
		    RDT(J2)=RD
		    LQT(J2)=LQ
 2530		 CONTINUE
		
		 MP=0
		 MT=0
		 NQ=0
		 DP=0.0
		 R0=0.0
		 PTOT=SQRT(PXSG(ISG,I)**2+PYSG(ISG,I)**2
     &                +PZSG(ISG,I)**2)
		 V1=PXSG(ISG,I)/PTOT
		 V2=PYSG(ISG,I)/PTOT
		 V3=PZSG(ISG,I)/PTOT

 2600		 RN=RLU(NSEED)
 2610		 IF(MT.GE.KT .AND. MP.GE.KP) GO TO 2690
		 IF(MT.GE.KT) GO TO 2620
		 IF(MP.GE.KP) GO TO 2640
		 IF(RDP(MP+1).GT.RDT(MT+1)) GO TO 2640
 2620		 MP=MP+1
		 DRR=RDP(MP)-R0
		 IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 2610
		 DP=DRR*HIPR1(14)/2.0
		 IF(DP.LE.0.2) GO TO 2610
		 IF(PTOT.LE.0.4) GO TO 2690
		 IF(PTOT.LE.DP) DP=PTOT-0.2
		 DE=DP
C
		 IF(K2SG(ISG,I).NE.21) THEN
		    IF(PTOT.LT.DP+HIPR1(1)) GO TO 2690
		    PRSHU=PP(LQP(MP),1)**2+PP(LQP(MP),2)**2
     &                    +PP(LQP(MP),3)**2
		    DE=SQRT(PMSG(ISG,I)**2+PTOT**2)
     &		       -SQRT(PMSG(ISG,I)**2+(PTOT-DP)**2)
		    ERSHU=(PP(LQP(MP),4)+DE-DP)**2
		    AMSHU=ERSHU-PRSHU
		    IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 2610
		    PP(LQP(MP),4)=SQRT(ERSHU)
		    PP(LQP(MP),5)=SQRT(AMSHU)
		 ENDIF
C		********reshuffle the energy when jet has mass
C
		 R0=RDP(MP)
		 DP1=DP*V1
		 DP2=DP*V2
		 DP3=DP*V3
C		********momentum and energy transfer from jet
		 NPJ(LQP(MP))=NPJ(LQP(MP))+1
		 KFPJ(LQP(MP),NPJ(LQP(MP)))=21
		 PJPX(LQP(MP),NPJ(LQP(MP)))=DP1
		 PJPY(LQP(MP),NPJ(LQP(MP)))=DP2
		 PJPZ(LQP(MP),NPJ(LQP(MP)))=DP3
		 PJPE(LQP(MP),NPJ(LQP(MP)))=DP
		 PJPM(LQP(MP),NPJ(LQP(MP)))=0.0

		 GO TO 2660

 2640		 MT=MT+1
		 DRR=RDT(MT)-R0
		 IF(RN.GE.1.0-EXP(-DRR/HIPR1(13))) GO TO 2610
		 DP=DRR*HIPR1(14)
		 IF(DP.LE.0.2) GO TO 2610
		 IF(PTOT.LE.0.4) GO TO 2690
		 IF(PTOT.LE.DP) DP=PTOT-0.2
		 DE=DP

		 IF(K2SG(ISG,I).NE.21) THEN
		    IF(PTOT.LT.DP+HIPR1(1)) GO TO 2690
		    PRSHU=PT(LQT(MT),1)**2+PT(LQT(MT),2)**2
     &                    +PT(LQT(MT),3)**2
		    DE=SQRT(PMSG(ISG,I)**2+PTOT**2)
     &		       -SQRT(PMSG(ISG,I)**2+(PTOT-DP)**2)
		    ERSHU=(PT(LQT(MT),4)+DE-DP)**2
		    AMSHU=ERSHU-PRSHU
		    IF(AMSHU.LT.HIPR1(1)*HIPR1(1)) GO TO 2610
		    PT(LQT(MT),4)=SQRT(ERSHU)
		    PT(LQT(MT),5)=SQRT(AMSHU)
		 ENDIF
C               ********reshuffle the energy when jet has mass

		 R0=RDT(MT)
		 DP1=DP*V1
		 DP2=DP*V2
		 DP3=DP*V3
C		********momentum and energy transfer from jet
		 NTJ(LQT(MT))=NTJ(LQT(MT))+1
		 KFTJ(LQT(MT),NTJ(LQT(MT)))=21
		 PJTX(LQT(MT),NTJ(LQT(MT)))=DP1
		 PJTY(LQT(MT),NTJ(LQT(MT)))=DP2
		 PJTZ(LQT(MT),NTJ(LQT(MT)))=DP3
		 PJTE(LQT(MT),NTJ(LQT(MT)))=DP
		 PJTM(LQT(MT),NTJ(LQT(MT)))=0.0

 2660		 PXSG(ISG,I)=(PTOT-DP)*V1
		 PYSG(ISG,I)=(PTOT-DP)*V2
		 PZSG(ISG,I)=(PTOT-DP)*V3
		 PESG(ISG,I)=PESG(ISG,I)-DE

		 PTOT=PTOT-DP
		 NQ=NQ+1
		 GO TO 2600
 2690	CONTINUE
	RETURN
	END

C
C
C
C
	SUBROUTINE HIJFRG(JTP,NTP,IERROR)
C	NTP=1, fragment proj string, NTP=2, targ string, 
C       NTP=3, independent 
C	strings from jets.  JTP is the line number of the string
C*******Fragment all leading strings of proj and targ**************
C	IHNT2(1)=atomic #, IHNT2(2)=proton #(=-1 if anti-proton)  *
C******************************************************************
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
C
        COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
	COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
	COMMON/RANSEED/NSEED
	
	IERROR=0
	CALL LUEDIT(0)
	N=0
C			********initialize the document lines
	IF(NTP.EQ.3) THEN
		ISG=JTP
		N=NJSG(ISG)
		DO 100 I=1,NJSG(ISG)
			K(I,1)=K1SG(ISG,I)
			K(I,2)=K2SG(ISG,I)
			P(I,1)=PXSG(ISG,I)
			P(I,2)=PYSG(ISG,I)
			P(I,3)=PZSG(ISG,I)
			P(I,4)=PESG(ISG,I)
			P(I,5)=PMSG(ISG,I)
C       Clear the starting point information in the Pythia arrays
                        V(I,1)=0
                        V(I,2)=0
                        V(I,3)=0
                        V(I,4)=0
                        V(I,5)=0
100		CONTINUE
C		IF(IHPR2(1).GT.0) CALL ATTRAD(IERROR)
c		IF(IERROR.NE.0) RETURN
C		CALL LULIST(1)
		CALL LUEXEC
		RETURN
	ENDIF
C
	IF(NTP.EQ.2) GO TO 200
	IF(JTP.GT.IHNT2(1))   RETURN
	IF(NFP(JTP,5).NE.3.AND.NFP(JTP,3).NE.0
     &	    .AND.NPJ(JTP).EQ.0.AND.NFP(JTP,10).EQ.0) GO TO 1000
	IF(NFP(JTP,15).EQ.-1) THEN
		KF1=NFP(JTP,2)
		KF2=NFP(JTP,1)
		PQ21=PP(JTP,6)
		PQ22=PP(JTP,7)
		PQ11=PP(JTP,8)
		PQ12=PP(JTP,9)
		AM1=PP(JTP,15)
		AM2=PP(JTP,14)
	ELSE
		KF1=NFP(JTP,1)
		KF2=NFP(JTP,2)
		PQ21=PP(JTP,8)
		PQ22=PP(JTP,9)
		PQ11=PP(JTP,6)
		PQ12=PP(JTP,7)
		AM1=PP(JTP,14)
		AM2=PP(JTP,15)	
	ENDIF
C	********for NFP(JTP,15)=-1 NFP(JTP,1) IS IN -Z DIRECTION
	PB1=PQ11+PQ21
	PB2=PQ12+PQ22
	PB3=PP(JTP,3)
	PECM=PP(JTP,5)
	BTZ=PB3/PP(JTP,4)
	IF((ABS(PB1-PP(JTP,1)).GT.0.01.OR.
     &    ABS(PB2-PP(JTP,2)).GT.0.01).AND.IHPR2(10).NE.0)
     &	  WRITE(6,*) '  Pt of Q and QQ do not sum to the total'

	GO TO 300

200	IF(JTP.GT.IHNT2(3))  RETURN
	IF(NFT(JTP,5).NE.3.AND.NFT(JTP,3).NE.0
     &	   .AND.NTJ(JTP).EQ.0.AND.NFT(JTP,10).EQ.0) GO TO 1200
	IF(NFT(JTP,15).EQ.1) THEN
		KF1=NFT(JTP,1)
		KF2=NFT(JTP,2)
		PQ11=PT(JTP,6)
		PQ12=PT(JTP,7)
		PQ21=PT(JTP,8)
		PQ22=PT(JTP,9)
		AM1=PT(JTP,14)
		AM2=PT(JTP,15)
	ELSE
		KF1=NFT(JTP,2)
		KF2=NFT(JTP,1)
		PQ11=PT(JTP,8)
		PQ12=PT(JTP,9)
		PQ21=PT(JTP,6)
		PQ22=PT(JTP,7)
		AM1=PT(JTP,15)
		AM2=PT(JTP,14)
	ENDIF	
C	********for NFT(JTP,15)=1 NFT(JTP,1) IS IN +Z DIRECTION
	PB1=PQ11+PQ21
	PB2=PQ12+PQ22
	PB3=PT(JTP,3)
	PECM=PT(JTP,5)
	BTZ=PB3/PT(JTP,4)

	IF((ABS(PB1-PT(JTP,1)).GT.0.01.OR.
     &     ABS(PB2-PT(JTP,2)).GT.0.01).AND.IHPR2(10).NE.0)
     &     WRITE(6,*) '  Pt of Q and QQ do not sum to the total'

300	IF(PECM.LT.HIPR1(1)) THEN
	   IERROR=1
	   IF(IHPR2(10).EQ.0) RETURN
	   WRITE(6,*) ' ECM=',PECM,' energy of the string is too small'
	   RETURN
	ENDIF
	AMT=PECM**2+PB1**2+PB2**2
	AMT1=AM1**2+PQ11**2+PQ12**2
	AMT2=AM2**2+PQ21**2+PQ22**2
	PZCM=SQRT(ABS(AMT**2+AMT1**2+AMT2**2-2.0*AMT*AMT1
     &       -2.0*AMT*AMT2-2.0*AMT1*AMT2))/2.0/SQRT(AMT)
C		*******PZ of end-partons in c.m. frame of the string
	K(1,1)=2
	K(1,2)=KF1
	P(1,1)=PQ11
	P(1,2)=PQ12
	P(1,3)=PZCM
	P(1,4)=SQRT(AMT1+PZCM**2)
	P(1,5)=AM1
	K(2,1)=1
	K(2,2)=KF2
	P(2,1)=PQ21
	P(2,2)=PQ22
	P(2,3)=-PZCM
	P(2,4)=SQRT(AMT2+PZCM**2)
	P(2,5)=AM2
C       Clear the starting point information in the Pythia arrays
        V(1,1)=0
        V(1,2)=0
        V(1,3)=0
        V(1,4)=0
        V(1,5)=0
        V(2,1)=0
        V(2,2)=0
        V(2,3)=0
        V(2,4)=0
        V(2,5)=0
	N=2
C*****
	CALL HIROBO(0.0,0.0,0.0,0.0,BTZ)
	JETOT=0
	IF((PQ21**2+PQ22**2).GT.(PQ11**2+PQ12**2)) THEN
		PMAX1=P(2,1)
		PMAX2=P(2,2)
		PMAX3=P(2,3)
	ELSE
		PMAX1=P(1,1)
		PMAX2=P(1,2)
		PMAX3=P(1,3)
	ENDIF
	IF(NTP.EQ.1) THEN
		PP(JTP,10)=PMAX1
		PP(JTP,11)=PMAX2
		PP(JTP,12)=PMAX3
	ELSE IF(NTP.EQ.2) THEN
		PT(JTP,10)=PMAX1
		PT(JTP,11)=PMAX2
		PT(JTP,12)=PMAX3
	ENDIF
C*******************attach produced jets to the leading partons****
	IF(NTP.EQ.1.AND.NPJ(JTP).NE.0) THEN
		JETOT=NPJ(JTP)
C		IF(NPJ(JTP).GE.2) CALL HIJSRT(JTP,1)
C			********sort jets in order of y
		IEX=0
		IF((ABS(KF1).GT.1000.AND.KF1.LT.0)
     &			.OR.(ABS(KF1).LT.1000.AND.KF1.GT.0)) IEX=1
		DO 520 I=N,2,-1
		DO 520 J=1,5
			II=NPJ(JTP)+I
			K(II,J)=K(I,J)
			P(II,J)=P(I,J)
			V(II,J)=V(I,J)
520		CONTINUE
		DO 540 I=1,NPJ(JTP)
			DO 542 J=1,5
				K(I+1,J)=0
				V(I+1,J)=0
542			CONTINUE				
			I0=I
			IF(IEX.EQ.1) I0=NPJ(JTP)-I+1
C				********reverse the order of jets
			KK1=KFPJ(JTP,I0)
			K(I+1,1)=2
			K(I+1,2)=KK1
			IF(KK1.NE.21 .AND. KK1.NE.0)  K(I+1,1)=
     &			  1+(ABS(KK1)+(2*IEX-1)*KK1)/2/ABS(KK1)
			P(I+1,1)=PJPX(JTP,I0)
			P(I+1,2)=PJPY(JTP,I0)
			P(I+1,3)=PJPZ(JTP,I0)
			P(I+1,4)=PJPE(JTP,I0)
			P(I+1,5)=PJPM(JTP,I0)
540		CONTINUE
		N=N+NPJ(JTP)
	ELSE IF(NTP.EQ.2.AND.NTJ(JTP).NE.0) THEN
		JETOT=NTJ(JTP)
c		IF(NTJ(JTP).GE.2)  CALL HIJSRT(JTP,2)
C			********sort jets in order of y
		IEX=1
		IF((ABS(KF2).GT.1000.AND.KF2.LT.0)
     &			.OR.(ABS(KF2).LT.1000.AND.KF2.GT.0)) IEX=0
		DO 560 I=N,2,-1
		DO 560 J=1,5
			II=NTJ(JTP)+I
			K(II,J)=K(I,J)
			P(II,J)=P(I,J)
			V(II,J)=V(I,J)
560		CONTINUE
		DO 580 I=1,NTJ(JTP)
			DO 582 J=1,5
				K(I+1,J)=0
				V(I+1,J)=0
582			CONTINUE				
			I0=I
			IF(IEX.EQ.1) I0=NTJ(JTP)-I+1
C				********reverse the order of jets
			KK1=KFTJ(JTP,I0)
			K(I+1,1)=2
			K(I+1,2)=KK1
			IF(KK1.NE.21 .AND. KK1.NE.0) K(I+1,1)=
     &		           1+(ABS(KK1)+(2*IEX-1)*KK1)/2/ABS(KK1)
			P(I+1,1)=PJTX(JTP,I0)
			P(I+1,2)=PJTY(JTP,I0)
			P(I+1,3)=PJTZ(JTP,I0)
			P(I+1,4)=PJTE(JTP,I0)
			P(I+1,5)=PJTM(JTP,I0)
580		CONTINUE
		N=N+NTJ(JTP)
	ENDIF
	IF(IHPR2(1).GT.0.AND.RLU(NSEED).LE.HIDAT(3)) THEN
	     HIDAT20=HIDAT(2)
	     HIPR150=HIPR1(5)
	     IF(IHPR2(8).EQ.0.AND.IHPR2(3).EQ.0.AND.IHPR2(9).EQ.0)
     &			HIDAT(2)=2.0
	     IF(HINT1(1).GE.1000.0.AND.JETOT.EQ.0)THEN
		HIDAT(2)=3.0
		HIPR1(5)=5.0
	     ENDIF
	     CALL ATTRAD(IERROR)
	     HIDAT(2)=HIDAT20
	     HIPR1(5)=HIPR150
	ELSE IF(JETOT.EQ.0.AND.IHPR2(1).GT.0.AND.
     &                       HINT1(1).GE.1000.0.AND.
     &		RLU(NSEED).LE.0.8) THEN
		HIDAT20=HIDAT(2)
		HIPR150=HIPR1(5)
		HIDAT(2)=3.0
		HIPR1(5)=5.0
	     IF(IHPR2(8).EQ.0.AND.IHPR2(3).EQ.0.AND.IHPR2(9).EQ.0)
     &			HIDAT(2)=2.0
		CALL ATTRAD(IERROR)
		HIDAT(2)=HIDAT20
		HIPR1(5)=HIPR150
	ENDIF
	IF(IERROR.NE.0) RETURN
C		******** conduct soft radiations
C****************************
C
C
C	CALL LULIST(1)
	CALL LUEXEC
	RETURN

1000	N=1
	K(1,1)=1
       	K(1,2)=NFP(JTP,3)
	DO 1100 JJ=1,5
       		P(1,JJ)=PP(JTP,JJ)
C       Clear the starting point information in the Pythia arrays
                V(1,JJ)=0
1100		CONTINUE
C			********proj remain as a nucleon or delta
	CALL LUEXEC
C	call lulist(1)
	RETURN
C
1200	N=1
	K(1,1)=1
	K(1,2)=NFT(JTP,3)
	DO 1300 JJ=1,5
		P(1,JJ)=PT(JTP,JJ)
C       Clear the starting point information in the Pythia arrays
                V(1,JJ)=0
1300	CONTINUE
C			********targ remain as a nucleon or delta
	CALL LUEXEC
C	call lulist(1)
	RETURN
	END
C
C
C
C********************************************************************
C	Sort the jets associated with a nucleon in order of their
C	rapdities
C********************************************************************
	SUBROUTINE HIJSRT(JPJT,NPT)
	DIMENSION KF(100),PX(100),PY(100),PZ(100),PE(100),PM(100)
	DIMENSION Y(100),IP(100,2)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	SAVE
	IF(NPT.EQ.2) GO TO 500
	JP=JPJT
	IQ=0
	I=1
100	KF(I)=KFPJ(JP,I)
	PX(I)=PJPX(JP,I)
	PY(I)=PJPY(JP,I)
	PZ(I)=PJPZ(JP,I)
	PE(I)=PJPE(JP,I)
	PM(I)=PJPM(JP,I)
	Y(I-IQ)=0.5*ALOG((ABS(PE(I)+PZ(I))+1.E-5)
     &          /(ABS(PE(I)-PZ(I))+1.E-5))
	IP(I-IQ,1)=I
	IP(I-IQ,2)=0
	IF(KF(I).NE.21) THEN
		IP(I-IQ,2)=1
		IQ=IQ+1
		I=I+1
		KF(I)=KFPJ(JP,I)
		PX(I)=PJPX(JP,I)
		PY(I)=PJPY(JP,I)
		PZ(I)=PJPZ(JP,I)
		PE(I)=PJPE(JP,I)
		PM(I)=PJPM(JP,I)
	ENDIF
	I=I+1
	IF(I.LE.NPJ(JP)) GO TO 100
			
	DO 200 I=1,NPJ(JP)-IQ
	DO 200 J=I+1,NPJ(JP)-IQ
		IF(Y(I).GT.Y(J)) GO TO 200
		IP1=IP(I,1)
		IP2=IP(I,2)
		IP(I,1)=IP(J,1)
		IP(I,2)=IP(J,2)
		IP(J,1)=IP1
		IP(J,2)=IP2
200	CONTINUE
C			********sort in decending y
	IQQ=0
	I=1
300	KFPJ(JP,I)=KF(IP(I-IQQ,1))
	PJPX(JP,I)=PX(IP(I-IQQ,1))
	PJPY(JP,I)=PY(IP(I-IQQ,1))
	PJPZ(JP,I)=PZ(IP(I-IQQ,1))
	PJPE(JP,I)=PE(IP(I-IQQ,1))
	PJPM(JP,I)=PM(IP(I-IQQ,1))
	IF(IP(I-IQQ,2).EQ.1) THEN
		KFPJ(JP,I+1)=KF(IP(I-IQQ,1)+1)
		PJPX(JP,I+1)=PX(IP(I-IQQ,1)+1)
		PJPY(JP,I+1)=PY(IP(I-IQQ,1)+1)
		PJPZ(JP,I+1)=PZ(IP(I-IQQ,1)+1)
		PJPE(JP,I+1)=PE(IP(I-IQQ,1)+1)
		PJPM(JP,I+1)=PM(IP(I-IQQ,1)+1)
		I=I+1
		IQQ=IQQ+1
	ENDIF
	I=I+1
	IF(I.LE.NPJ(JP)) GO TO 300

	RETURN

500	JT=JPJT
	IQ=0
	I=1
600	KF(I)=KFTJ(JT,I)
	PX(I)=PJTX(JT,I)
	PY(I)=PJTY(JT,I)
	PZ(I)=PJTZ(JT,I)
	PE(I)=PJTE(JT,I)
	PM(I)=PJTM(JT,I)
	Y(I-IQ)=0.5*ALOG((ABS(PE(I)+PZ(I))+1.E-5)
     &          /(ABS(PE(I)-PZ(I))+1.E-5))
	IP(I-IQ,1)=I
	IP(I-IQ,2)=0
	IF(KF(I).NE.21) THEN
		IP(I-IQ,2)=1
		IQ=IQ+1
		I=I+1
		KF(I)=KFTJ(JT,I)
		PX(I)=PJTX(JT,I)
		PY(I)=PJTY(JT,I)
		PZ(I)=PJTZ(JT,I)
		PE(I)=PJTE(JT,I)
		PM(I)=PJTM(JT,I)
	ENDIF
	I=I+1
	IF(I.LE.NTJ(JT)) GO TO 600
			
	DO 700 I=1,NTJ(JT)-IQ
	DO 700 J=I+1,NTJ(JT)-IQ
		IF(Y(I).LT.Y(J)) GO TO 700
		IP1=IP(I,1)
		IP2=IP(I,2)
		IP(I,1)=IP(J,1)
		IP(I,2)=IP(J,2)
		IP(J,1)=IP1
		IP(J,2)=IP2
700	CONTINUE
C			********sort in acending y
	IQQ=0
	I=1
800	KFTJ(JT,I)=KF(IP(I-IQQ,1))
	PJTX(JT,I)=PX(IP(I-IQQ,1))
	PJTY(JT,I)=PY(IP(I-IQQ,1))
	PJTZ(JT,I)=PZ(IP(I-IQQ,1))
	PJTE(JT,I)=PE(IP(I-IQQ,1))
	PJTM(JT,I)=PM(IP(I-IQQ,1))
	IF(IP(I-IQQ,2).EQ.1) THEN
		KFTJ(JT,I+1)=KF(IP(I-IQQ,1)+1)
		PJTX(JT,I+1)=PX(IP(I-IQQ,1)+1)
		PJTY(JT,I+1)=PY(IP(I-IQQ,1)+1)
		PJTZ(JT,I+1)=PZ(IP(I-IQQ,1)+1)
		PJTE(JT,I+1)=PE(IP(I-IQQ,1)+1)
		PJTM(JT,I+1)=PM(IP(I-IQQ,1)+1)
		I=I+1
		IQQ=IQQ+1
	ENDIF
	I=I+1
	IF(I.LE.NTJ(JT)) GO TO 800
	RETURN
	END	

C
C
C
C****************************************************************
C	conduct soft radiation according to dipole approxiamtion
C****************************************************************
	SUBROUTINE ATTRAD(IERROR)
C
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
	COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
	COMMON/RANSEED/NSEED
	SAVE  
	IERROR=0

C.....S INVARIANT MASS-SQUARED BETWEEN PARTONS I AND I+1......
C.....SM IS THE LARGEST MASS-SQUARED....

40	SM=0.
        JL=1
	DO 30 I=1,N-1
	   S=2.*(P(I,4)*P(I+1,4)-P(I,1)*P(I+1,1)-P(I,2)*P(I+1,2)
     &		-P(I,3)*P(I+1,3))+P(I,5)**2+P(I+1,5)**2
	   IF(S.LT.0.) S=0.
	   WP=SQRT(S)-1.5*(P(I,5)+P(I+1,5))
	   IF(WP.GT.SM) THEN
	      PBT1=P(I,1)+P(I+1,1)
	      PBT2=P(I,2)+P(I+1,2)
	      PBT3=P(I,3)+P(I+1,3)
	      PBT4=P(I,4)+P(I+1,4)
	      BTT=(PBT1**2+PBT2**2+PBT3**2)/PBT4**2
	      IF(BTT.GE.1.0-1.0E-10) GO TO 30
	      IF((I.NE.1.OR.I.NE.N-1).AND.
     &             (K(I,2).NE.21.AND.K(I+1,2).NE.21)) GO TO 30
	      JL=I
	      SM=WP
	   ENDIF
30	CONTINUE
	S=(SM+1.5*(P(JL,5)+P(JL+1,5)))**2
      	IF(SM.LT.HIPR1(5)) GOTO 2
     
C.....MAKE PLACE FOR ONE GLUON.....
      	IF(JL+1.EQ.N) GOTO 190
      	DO 160 J=N,JL+2,-1
      		K(J+1,1)=K(J,1)
		K(J+1,2)=K(J,2)
      		DO 150 M=1,5
150   			P(J+1,M)=P(J,M)
160   		CONTINUE
190   	N=N+1
     
C.....BOOST TO REST SYSTEM FOR PARTICLES JL AND JL+1.....
      	P1=P(JL,1)+P(JL+1,1)
      	P2=P(JL,2)+P(JL+1,2)
      	P3=P(JL,3)+P(JL+1,3)
      	P4=P(JL,4)+P(JL+1,4)
      	BEX=-P1/P4
      	BEY=-P2/P4
      	BEZ=-P3/P4
	IMIN=JL
	IMAX=JL+1
      	CALL ATROBO(0.,0.,BEX,BEY,BEZ,IMIN,IMAX,IERROR)
	IF(IERROR.NE.0) RETURN
C.....ROTATE TO Z-AXIS....
      	CTH=P(JL,3)/SQRT(P(JL,4)**2-P(JL,5)**2)
      	IF(ABS(CTH).GT.1.0)  CTH=MAX(-1.,MIN(1.,CTH))
      	THETA=ACOS(CTH)
      	PHI=ULANGL(P(JL,1),P(JL,2))
      	CALL ATROBO(0.,-PHI,0.,0.,0.,IMIN,IMAX,IERROR)
      	CALL ATROBO(-THETA,0.,0.,0.,0.,IMIN,IMAX,IERROR)
     
C.....CREATE ONE GLUON AND ORIENTATE.....
     
1	CALL AR3JET(S,X1,X3,JL)
      	CALL ARORIE(S,X1,X3,JL)		
	IF(HIDAT(2).GT.0.0) THEN
      	   PTG1=SQRT(P(JL,1)**2+P(JL,2)**2)
      	   PTG2=SQRT(P(JL+1,1)**2+P(JL+1,2)**2)
      	   PTG3=SQRT(P(JL+2,1)**2+P(JL+2,2)**2)
	   PTG=MAX(PTG1,PTG2,PTG3)
	   IF(PTG.GT.HIDAT(2)) THEN
	      FMFACT=EXP(-(PTG**2-HIDAT(2)**2)/HIPR1(2)**2)
	      IF(RLU(NSEED).GT.FMFACT) GO TO 1
	   ENDIF
	ENDIF
C.....ROTATE AND BOOST BACK.....
	IMIN=JL
	IMAX=JL+2
      	CALL ATROBO(THETA,PHI,-BEX,-BEY,-BEZ,IMIN,IMAX,IERROR)
	IF(IERROR.NE.0) RETURN
C.....ENUMERATE THE GLUONS.....
      	K(JL+2,1)=K(JL+1,1)
	K(JL+2,2)=K(JL+1,2)
	K(JL+2,3)=K(JL+1,3)
	K(JL+2,4)=K(JL+1,4)
	K(JL+2,5)=K(JL+1,5)
      	P(JL+2,5)=P(JL+1,5)
      	K(JL+1,1)=2
	K(JL+1,2)=21
	K(JL+1,3)=0
	K(JL+1,4)=0
	K(JL+1,5)=0
      	P(JL+1,5)=0.
C----THETA FUNCTION DAMPING OF THE EMITTED GLUONS. FOR HADRON-HADRON.
C----R0=VFR(2)
C      	IF(VFR(2).GT.0.) THEN
C      	PTG=SQRT(P(JL+1,1)**2+P(JL+1,2)**2)
C      	PTGMAX=WSTRI/2.
C      	DOPT=SQRT((4.*PAR(71)*VFR(2))/WSTRI)
C      	PTOPT=(DOPT*WSTRI)/(2.*VFR(2))
C      	IF(PTG.GT.PTOPT) IORDER=IORDER-1
C      	IF(PTG.GT.PTOPT) GOTO 1
C      	ENDIF
C-----
     	IF(SM.GE.HIPR1(5)) GOTO 40

2      	K(1,1)=2
	K(1,3)=0
	K(1,4)=0
	K(1,5)=0
      	K(N,1)=1
	K(N,3)=0
	K(N,4)=0
	K(N,5)=0

      	RETURN
      	END


	SUBROUTINE AR3JET(S,X1,X3,JL)
C     
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
	COMMON/RANSEED/NSEED
	SAVE  
C     
	C=1./3.
      	IF(K(JL,2).NE.21 .AND. K(JL+1,2).NE.21) C=8./27.
      	EXP1=3
      	EXP3=3
      	IF(K(JL,2).NE.21) EXP1=2
      	IF(K(JL+1,2).NE.21) EXP3=2
      	A=0.24**2/S
      	YMA=ALOG(.5/SQRT(A)+SQRT(.25/A-1))
      	D=4.*C*YMA
      	SM1=P(JL,5)**2/S
      	SM3=P(JL+1,5)**2/S
      	XT2M=(1.-2.*SQRT(SM1)+SM1-SM3)*(1.-2.*SQRT(SM3)-SM1+SM3)
      	XT2M=MIN(.25,XT2M)
      	NTRY=0
1     	IF(NTRY.EQ.5000) THEN
	        X1=.5*(2.*SQRT(SM1)+1.+SM1-SM3)
		X3=.5*(2.*SQRT(SM3)+1.-SM1+SM3)
		RETURN
      	ENDIF
      	NTRY=NTRY+1
     
      	XT2=A*(XT2M/A)**(RLU(NSEED)**(1./D))
     
      	YMAX=ALOG(.5/SQRT(XT2)+SQRT(.25/XT2-1.))
      	Y=(2.*RLU(NSEED)-1.)*YMAX
      	X1=1.-SQRT(XT2)*EXP(Y)
      	X3=1.-SQRT(XT2)*EXP(-Y)
      	X2=2.-X1-X3
      	NEG=0
      	IF(K(JL,2).NE.21 .OR. K(JL+1,2).NE.21) THEN
        IF((1.-X1)*(1.-X2)*(1.-X3)-X2*SM1*(1.-X1)-X2*SM3*(1.-X3).
     &  LE.0..OR.X1.LE.2.*SQRT(SM1)-SM1+SM3.OR.X3.LE.2.*SQRT(SM3)
     &  -SM3+SM1) NEG=1
        X1=X1+SM1-SM3
        X3=X3-SM1+SM3
     	ENDIF
      	IF(NEG.EQ.1) GOTO 1
     
      	FG=2.*YMAX*C*(X1**EXP1+X3**EXP3)/D
      	XT2M=XT2
      	IF(FG.LT.RLU(NSEED)) GOTO 1
     
      	RETURN
      	END
C*************************************************************


	SUBROUTINE ARORIE(S,X1,X3,JL)
C     
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
	COMMON/RANSEED/NSEED
	SAVE  
C     
     	W=SQRT(S)
     	X2=2.-X1-X3
     	E1=.5*X1*W
     	E3=.5*X3*W
     	P1=SQRT(E1**2-P(JL,5)**2)
	P3=SQRT(E3**2-P(JL+1,5)**2)
	CBET=1.
	IF(P1.GT.0..AND.P3.GT.0.) CBET=(P(JL,5)**2
     &           +P(JL+1,5)**2+2.*E1*E3-S*(1.-X2))/(2.*P1*P3)
      	IF(ABS(CBET).GT.1.0) CBET=MAX(-1.,MIN(1.,CBET))
      	BET=ACOS(CBET)
     
C.....MINIMIZE PT1-SQUARED PLUS PT3-SQUARED.....
      	IF(P1.GE.P3) THEN
	   PSI=.5*ULANGL(P1**2+P3**2*COS(2.*BET),-P3**2*SIN(2.*BET))
	   PT1=P1*SIN(PSI)
	   PZ1=P1*COS(PSI)
	   PT3=P3*SIN(PSI+BET)
	   PZ3=P3*COS(PSI+BET)
      	ELSE IF(P3.GT.P1) THEN
	   PSI=.5*ULANGL(P3**2+P1**2*COS(2.*BET),-P1**2*SIN(2.*BET))
	   PT1=P1*SIN(BET+PSI)
	   PZ1=-P1*COS(BET+PSI)
	   PT3=P3*SIN(PSI)
	   PZ3=-P3*COS(PSI)
      	ENDIF
     
      	DEL=2.0*HIPR1(40)*RLU(NSEED)
      	P(JL,4)=E1
      	P(JL,1)=PT1*SIN(DEL)
      	P(JL,2)=-PT1*COS(DEL)
      	P(JL,3)=PZ1
      	P(JL+2,4)=E3
      	P(JL+2,1)=PT3*SIN(DEL)
      	P(JL+2,2)=-PT3*COS(DEL)
      	P(JL+2,3)=PZ3
      	P(JL+1,4)=W-E1-E3
      	P(JL+1,1)=-P(JL,1)-P(JL+2,1)
      	P(JL+1,2)=-P(JL,2)-P(JL+2,2)
      	P(JL+1,3)=-P(JL,3)-P(JL+2,3)
      	RETURN
      	END


C
C*******************************************************************
C	make  boost and rotation to entries from IMIN to IMAX
C*******************************************************************
	SUBROUTINE ATROBO(THE,PHI,BEX,BEY,BEZ,IMIN,IMAX,IERROR)
        COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
	DIMENSION ROT(3,3),PV(3)
	DOUBLE PRECISION DP(4),DBEX,DBEY,DBEZ,DGA,DGA2,DBEP,DGABEP
        SAVE
	IERROR=0
     
      	IF(IMIN.LE.0 .OR. IMAX.GT.N .OR. IMIN.GT.IMAX) RETURN

      	IF(THE**2+PHI**2.GT.1E-20) THEN
C...ROTATE (TYPICALLY FROM Z AXIS TO DIRECTION THETA,PHI)
	   ROT(1,1)=COS(THE)*COS(PHI)
	   ROT(1,2)=-SIN(PHI)
	   ROT(1,3)=SIN(THE)*COS(PHI)
	   ROT(2,1)=COS(THE)*SIN(PHI)
	   ROT(2,2)=COS(PHI)
	   ROT(2,3)=SIN(THE)*SIN(PHI)
	   ROT(3,1)=-SIN(THE)
	   ROT(3,2)=0.
	   ROT(3,3)=COS(THE)
	   DO 120 I=IMIN,IMAX
C**************	   IF(MOD(K(I,1)/10000,10).GE.6) GOTO 120
	      DO 100 J=1,3
 100		 PV(J)=P(I,J)
		 DO 110 J=1,3
 110		    P(I,J)=ROT(J,1)*PV(1)+ROT(J,2)*PV(2)
     &                     +ROT(J,3)*PV(3)
 120		 CONTINUE
	ENDIF
     
      	IF(BEX**2+BEY**2+BEZ**2.GT.1E-20) THEN
C...LORENTZ BOOST (TYPICALLY FROM REST TO MOMENTUM/ENERGY=BETA)
        	DBEX=BEX
        	DBEY=BEY
        	DBEZ=BEZ
		DGA2=1D0-DBEX**2-DBEY**2-DBEZ**2
		IF(DGA2.LE.0D0) THEN
			IERROR=1
			RETURN
		ENDIF
        	DGA=1D0/DSQRT(DGA2)
        	DO 140 I=IMIN,IMAX
C*************	   IF(MOD(K(I,1)/10000,10).GE.6) GOTO 140
        	   DO 130 J=1,4
130   		   DP(J)=P(I,J)
        	   DBEP=DBEX*DP(1)+DBEY*DP(2)+DBEZ*DP(3)
        	   DGABEP=DGA*(DGA*DBEP/(1D0+DGA)+DP(4))
        	   P(I,1)=DP(1)+DGABEP*DBEX
        	   P(I,2)=DP(2)+DGABEP*DBEY
        	   P(I,3)=DP(3)+DGABEP*DBEZ
        	   P(I,4)=DGA*(DP(4)+DBEP)
140   		CONTINUE
      	ENDIF
     
      	RETURN
      	END
C
C
C
	SUBROUTINE HIJHRD(JP,JT,JOUT,JFLG,IOPJET0)
C
C	IOPTJET=1, ALL JET WILL FORM SINGLE STRING SYSTEM
C		0, ONLY Q-QBAR JET FORM SINGLE STRING SYSTEM
C*******Perform jets production and fragmentation when JP JT *******
C     scatter. JOUT-> number of hard scatterings precede this one  *
C     for the the same pair(JP,JT). JFLG->a flag to show whether   *
C     jets can be produced (with valence quark=1,gluon=2, q-qbar=3)*
C     or not(0). Information of jets are in  COMMON/ATTJET and     *
C     /MINJET. ABS(NFP(JP,6)) is the total number jets produced by *
C    JP. If NFP(JP,6)<0 JP can not produce jet anymore.	           *
C*******************************************************************
	DIMENSION IP(100,2),IPQ(50),IPB(50),IT(100,2),ITQ(50),ITB(50)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HIJJET4/NDR,IADR(900,2),KFDR(900),PDR(900,5),VDR(900,4)
	COMMON/RANSEED/NSEED
C************************************ HIJING common block
        COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
        COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
        COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200)
        COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
        COMMON/PYINT1/MINT(400),VINT(400)
        COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2)
        COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3)
        COMMON/HIPYINT/MINT4,MINT5,ATCO(200,20),ATXS(0:200)
	SAVE  
C********************************** LU common block
	MXJT=500
C		SIZE OF COMMON BLOCK FOR # OF PARTON PER STRING
	MXSG=900
C		SIZE OF COMMON BLOCK FOR # OF SINGLE STRINGS
	MXSJ=100
C		SIZE OF COMMON BLOCK FOR # OF PARTON PER SINGLE
C		STRING
	JFLG=0
	IHNT2(11)=JP
	IHNT2(12)=JT
C
	IOPJET=IOPJET0
	IF(IOPJET.EQ.1.AND.(NFP(JP,6).NE.0.OR.NFT(JT,6).NE.0))
     &                   IOPJET=0
	IF(JP.GT.IHNT2(1) .OR. JT.GT.IHNT2(3)) RETURN
	IF(NFP(JP,6).LT.0 .OR. NFT(JT,6).LT.0) RETURN
C		******** JP or JT can not produce jet anymore
C
	IF(JOUT.EQ.0) THEN
		EPP=PP(JP,4)+PP(JP,3)
		EPM=PP(JP,4)-PP(JP,3)
		ETP=PT(JT,4)+PT(JT,3)
		ETM=PT(JT,4)-PT(JT,3)
		IF(EPP.LT.0.0) GO TO 1000
		IF(EPM.LT.0.0) GO TO 1000
		IF(ETP.LT.0.0) GO TO 1000
		IF(ETM.LT.0.0) GO TO 1000
		IF(EPP/(EPM+0.01).LE.ETP/(ETM+0.01)) RETURN
	ENDIF
C		********for the first hard scattering of (JP,JT)
C			have collision only when Ycm(JP)>Ycm(JT)

	ECUT1=HIPR1(1)+HIPR1(8)+PP(JP,14)+PP(JP,15)
	ECUT2=HIPR1(1)+HIPR1(8)+PT(JT,14)+PT(JT,15)
	IF(PP(JP,4).LE.ECUT1) THEN
		NFP(JP,6)=-ABS(NFP(JP,6))
		RETURN
	ENDIF
	IF(PT(JT,4).LE.ECUT2) THEN
		NFT(JT,6)=-ABS(NFT(JT,6))
		RETURN
	ENDIF
C		*********must have enough energy to produce jets

	MISS=0
	MISP=0
	MIST=0
C
	IF(NFP(JP,10).EQ.0 .AND. NFT(JT,10).EQ.0) THEN
		MINT(44)=MINT4
		MINT(45)=MINT5
		XSEC(0,1)=ATXS(0)
		XSEC(11,1)=ATXS(11)
		XSEC(12,1)=ATXS(12)
		XSEC(28,1)=ATXS(28)
		DO 120 I=1,20
		COEF(11,I)=ATCO(11,I)
		COEF(12,I)=ATCO(12,I)
		COEF(28,I)=ATCO(28,I)
120		CONTINUE
	ELSE
		ISUB11=0
		ISUB12=0
		ISUB28=0
		IF(XSEC(11,1).NE.0) ISUB11=1
		IF(XSEC(12,1).NE.0) ISUB12=1
		IF(XSEC(28,1).NE.0) ISUB28=1		
		MINT(44)=MINT4-ISUB11-ISUB12-ISUB28
		MINT(45)=MINT5-ISUB11-ISUB12-ISUB28
		XSEC(0,1)=ATXS(0)-ATXS(11)-ATXS(12)-ATXS(28)
		XSEC(11,1)=0.0
		XSEC(12,1)=0.0
		XSEC(28,1)=0.0	
		DO 110 I=1,20
		COEF(11,I)=0.0
		COEF(12,I)=0.0
		COEF(28,I)=0.0
110		CONTINUE
	ENDIF		
C	********Scatter the valence quarks only once per NN 
C       collision,
C		afterwards only gluon can have hard scattering.
 155	CALL PYTHIA
	JJ=MINT(31)
	IF(JJ.NE.1) GO TO 155
C		*********one hard collision at a time
	IF(K(7,2).EQ.-K(8,2)) THEN
		QMASS2=(P(7,4)+P(8,4))**2-(P(7,1)+P(8,1))**2
     &			-(P(7,2)+P(8,2))**2-(P(7,3)+P(8,3))**2
		QM=ULMASS(K(7,2))
		IF(QMASS2.LT.(2.0*QM+HIPR1(1))**2) GO TO 155
	ENDIF
C		********q-qbar jets must has minimum mass HIPR1(1)
	PXP=PP(JP,1)-P(3,1)
	PYP=PP(JP,2)-P(3,2)
	PZP=PP(JP,3)-P(3,3)
	PEP=PP(JP,4)-P(3,4)
	PXT=PT(JT,1)-P(4,1)
	PYT=PT(JT,2)-P(4,2)
	PZT=PT(JT,3)-P(4,3)
	PET=PT(JT,4)-P(4,4)

	IF(PEP.LE.ECUT1) THEN
		MISP=MISP+1
		IF(MISP.LT.50) GO TO 155
		NFP(JP,6)=-ABS(NFP(JP,6))
		RETURN
	ENDIF
	IF(PET.LE.ECUT2) THEN
		MIST=MIST+1
		IF(MIST.LT.50) GO TO 155
		NFT(JT,6)=-ABS(NFT(JT,6))
		RETURN
	ENDIF
C		******** if the remain energy<ECUT the proj or targ
C			 can not produce jet anymore

	WP=PEP+PZP+PET+PZT
	WM=PEP-PZP+PET-PZT
	IF(WP.LT.0.0 .OR. WM.LT.0.0) THEN
		MISS=MISS+1
		IF(MISS.LT.50) GO TO 155
		RETURN
	ENDIF
C		********the total W+, W- must be positive
	SW=WP*WM
	AMPX=SQRT((ECUT1-HIPR1(8))**2+PXP**2+PYP**2+0.01)
	AMTX=SQRT((ECUT2-HIPR1(8))**2+PXT**2+PYT**2+0.01)
	SXX=(AMPX+AMTX)**2
	IF(SW.LT.SXX.OR.VINT(43).LT.HIPR1(1)) THEN
		MISS=MISS+1
		IF(MISS.LT.50) GO TO 155
		RETURN
	ENDIF  
C		********the proj and targ remnants must have at least
C			a CM energy that can produce two strings
C			with minimum mass HIPR1(1)(see HIJSFT HIJFRG)
C
	HINT1(41)=P(7,1)
	HINT1(42)=P(7,2)
	HINT1(43)=P(7,3)
	HINT1(44)=P(7,4)
	HINT1(45)=P(7,5)
	HINT1(46)=SQRT(P(7,1)**2+P(7,2)**2)
	HINT1(51)=P(8,1)
	HINT1(52)=P(8,2)
	HINT1(53)=P(8,3)
	HINT1(54)=P(8,4)
	HINT1(55)=P(8,5)
	HINT1(56)=SQRT(P(8,1)**2+P(8,2)**2) 
	IHNT2(14)=K(7,2)
	IHNT2(15)=K(8,2)
C
	PINIRAD=(1.0-EXP(-2.0*(VINT(47)-HIDAT(1))))
     &		/(1.0+EXP(-2.0*(VINT(47)-HIDAT(1))))
	I_INIRAD=0
	IF(RLU(NSEED).LE.PINIRAD) I_INIRAD=1
	IF(K(7,2).EQ.-K(8,2)) GO TO 190
	IF(K(7,2).EQ.21.AND.K(8,2).EQ.21.AND.IOPJET.EQ.1) GO TO 190
C*******************************************************************
C	gluon  jets are going to be connectd with
C	the final leading string of quark-aintquark
C*******************************************************************
	JFLG=2
	JPP=0
	LPQ=0
	LPB=0
	JTT=0
	LTQ=0
	LTB=0
	IS7=0
	IS8=0
        HINT1(47)=0.0
        HINT1(48)=0.0
        HINT1(49)=0.0
        HINT1(50)=0.0
        HINT1(67)=0.0
        HINT1(68)=0.0
        HINT1(69)=0.0
        HINT1(70)=0.0
	DO 180 I=9,N
	   IF(K(I,3).EQ.1 .OR. K(I,3).EQ.2.OR.
     &                   ABS(K(I,2)).GT.30) GO TO 180
C************************************************************
           IF(K(I,3).EQ.7) THEN
              HINT1(47)=HINT1(47)+P(I,1)
              HINT1(48)=HINT1(48)+P(I,2)
              HINT1(49)=HINT1(49)+P(I,3)
              HINT1(50)=HINT1(50)+P(I,4)
           ENDIF
           IF(K(I,3).EQ.8) THEN
              HINT1(67)=HINT1(67)+P(I,1)
              HINT1(68)=HINT1(68)+P(I,2)
              HINT1(69)=HINT1(69)+P(I,3)
              HINT1(70)=HINT1(70)+P(I,4)
           ENDIF
C************************modifcation made on Apr 10. 1996*****
	   IF(K(I,2).GT.21.AND.K(I,2).LE.30) THEN
	      NDR=NDR+1
	      IADR(NDR,1)=JP
	      IADR(NDR,2)=JT
	      KFDR(NDR)=K(I,2)
	      PDR(NDR,1)=P(I,1)
	      PDR(NDR,2)=P(I,2)
	      PDR(NDR,3)=P(I,3)
	      PDR(NDR,4)=P(I,4)
	      PDR(NDR,5)=P(I,5)
              VDR(NDR,1)=V(I,1)
              VDR(NDR,2)=V(I,2)
              VDR(NDR,3)=V(I,3)
              VDR(NDR,4)=V(I,4)
C************************************************************
	      GO TO 180
C************************correction made on Oct. 14,1994*****
	   ENDIF
	   IF(K(I,3).EQ.7.OR.K(I,3).EQ.3) THEN
	      IF(K(I,3).EQ.7.AND.K(I,2).NE.21.AND.K(I,2).EQ.K(7,2)
     &	             .AND.IS7.EQ.0) THEN
		 PP(JP,10)=P(I,1)
		 PP(JP,11)=P(I,2)
		 PP(JP,12)=P(I,3)
		 PZP=PZP+P(I,3)
		 PEP=PEP+P(I,4)
		 NFP(JP,10)=1
		 IS7=1
		 GO TO 180
	      ENDIF
	      IF(K(I,3).EQ.3.AND.(K(I,2).NE.21.OR.
     &                               I_INIRAD.EQ.0)) THEN
		 PXP=PXP+P(I,1)
		 PYP=PYP+P(I,2)
		 PZP=PZP+P(I,3)
		 PEP=PEP+P(I,4)
		 GO TO 180 
	      ENDIF
	      JPP=JPP+1
	      IP(JPP,1)=I
	      IP(JPP,2)=0
	      IF(K(I,2).NE.21) THEN
		 IF(K(I,2).GT.0) THEN
		    LPQ=LPQ+1
		    IPQ(LPQ)=JPP
		    IP(JPP,2)=LPQ
		 ELSE IF(K(I,2).LT.0) THEN
		    LPB=LPB+1
		    IPB(LPB)=JPP
		    IP(JPP,2)=-LPB
		 ENDIF
	      ENDIF
	   ELSE IF(K(I,3).EQ.8.OR.K(I,3).EQ.4) THEN
	      IF(K(I,3).EQ.8.AND.K(I,2).NE.21.AND.K(I,2).EQ.K(8,2)
     &				.AND.IS8.EQ.0) THEN
		 PT(JT,10)=P(I,1)
		 PT(JT,11)=P(I,2)
		 PT(JT,12)=P(I,3)
		 PZT=PZT+P(I,3)
		 PET=PET+P(I,4)
		 NFT(JT,10)=1
		 IS8=1
		 GO TO 180
	      ENDIF			
	      IF(K(I,3).EQ.4.AND.(K(I,2).NE.21.OR.
     &                             I_INIRAD.EQ.0)) THEN
		 PXT=PXT+P(I,1)
		 PYT=PYT+P(I,2)
		 PZT=PZT+P(I,3)
		 PET=PET+P(I,4)
		 GO TO 180
	      ENDIF
	      JTT=JTT+1
	      IT(JTT,1)=I
	      IT(JTT,2)=0
	      IF(K(I,2).NE.21) THEN
		 IF(K(I,2).GT.0) THEN
		    LTQ=LTQ+1
		    ITQ(LTQ)=JTT
		    IT(JTT,2)=LTQ
		 ELSE IF(K(I,2).LT.0) THEN
		    LTB=LTB+1
		    ITB(LTB)=JTT
		    IT(JTT,2)=-LTB
		 ENDIF
	      ENDIF
	   ENDIF
 180	CONTINUE
c
c
	IF(LPQ.NE.LPB .OR. LTQ.NE.LTB) THEN
		MISS=MISS+1
		IF(MISS.LE.50) GO TO 155
		WRITE(6,*) ' Q -QBAR NOT MATCHED IN HIJHRD'
		JFLG=0
		RETURN
	ENDIF
C****The following will rearrange the partons so that a quark is***
C****allways followed by an anti-quark ****************************

	J=0
181	J=J+1
	IF(J.GT.JPP) GO TO 182
	IF(IP(J,2).EQ.0) THEN
		GO TO 181
	ELSE IF(IP(J,2).NE.0) THEN
		LP=ABS(IP(J,2))
		IP1=IP(J,1)
		IP2=IP(J,2)
		IP(J,1)=IP(IPQ(LP),1)
		IP(J,2)=IP(IPQ(LP),2)
		IP(IPQ(LP),1)=IP1
		IP(IPQ(LP),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPQ(LP)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPQ(LP)
		ENDIF
C		********replace J with a quark
		IP1=IP(J+1,1)
		IP2=IP(J+1,2)
		IP(J+1,1)=IP(IPB(LP),1)
		IP(J+1,2)=IP(IPB(LP),2)
		IP(IPB(LP),1)=IP1
		IP(IPB(LP),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPB(LP)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPB(LP)
		ENDIF
C		******** replace J+1 with anti-quark
		J=J+1
		GO TO 181
	ENDIF

182	J=0
183	J=J+1
	IF(J.GT.JTT) GO TO 184
	IF(IT(J,2).EQ.0) THEN
		GO TO 183
	ELSE IF(IT(J,2).NE.0) THEN
		LT=ABS(IT(J,2))
		IT1=IT(J,1)
		IT2=IT(J,2)
		IT(J,1)=IT(ITQ(LT),1)
		IT(J,2)=IT(ITQ(LT),2)
		IT(ITQ(LT),1)=IT1
		IT(ITQ(LT),2)=IT2
		IF(IT2.GT.0) THEN
			ITQ(IT2)=ITQ(LT)
		ELSE IF(IT2.LT.0) THEN
			ITB(-IT2)=ITQ(LT)
		ENDIF
C		********replace J with a quark
		IT1=IT(J+1,1)
		IT2=IT(J+1,2)
		IT(J+1,1)=IT(ITB(LT),1)
		IT(J+1,2)=IT(ITB(LT),2)
		IT(ITB(LT),1)=IT1
		IT(ITB(LT),2)=IT2
		IF(IT2.GT.0) THEN
			ITQ(IT2)=ITB(LT)
		ELSE IF(IT2.LT.0) THEN
			ITB(-IT2)=ITB(LT)
		ENDIF
C		******** replace J+1 with anti-quark
		J=J+1
		GO TO 183

	ENDIF

184	CONTINUE
	IF(NPJ(JP)+JPP.GT.MXJT.OR.NTJ(JT)+JTT.GT.MXJT) THEN
		JFLG=0
		WRITE(6,*) 'number of partons per string exceeds'
		WRITE(6,*) 'the common block size'
		RETURN
	ENDIF
C			********check the bounds of common blocks
	DO 186 J=1,JPP
		KFPJ(JP,NPJ(JP)+J)=K(IP(J,1),2)
		PJPX(JP,NPJ(JP)+J)=P(IP(J,1),1)
		PJPY(JP,NPJ(JP)+J)=P(IP(J,1),2)
		PJPZ(JP,NPJ(JP)+J)=P(IP(J,1),3)
		PJPE(JP,NPJ(JP)+J)=P(IP(J,1),4)
		PJPM(JP,NPJ(JP)+J)=P(IP(J,1),5)
186	CONTINUE
	NPJ(JP)=NPJ(JP)+JPP
	DO 188 J=1,JTT
		KFTJ(JT,NTJ(JT)+J)=K(IT(J,1),2)
		PJTX(JT,NTJ(JT)+J)=P(IT(J,1),1)
		PJTY(JT,NTJ(JT)+J)=P(IT(J,1),2)
		PJTZ(JT,NTJ(JT)+J)=P(IT(J,1),3)
		PJTE(JT,NTJ(JT)+J)=P(IT(J,1),4)
		PJTM(JT,NTJ(JT)+J)=P(IT(J,1),5)
188	CONTINUE
	NTJ(JT)=NTJ(JT)+JTT
	GO TO 900
C*****************************************************************
CThis is the case of a quark-antiquark jet it will fragment alone
C****************************************************************
190	JFLG=3
	IF(K(7,2).NE.21.AND.K(8,2).NE.21.AND.
     &                   K(7,2)*K(8,2).GT.0) GO TO 155
	JPP=0
	LPQ=0
	LPB=0
        DO 200 I=9,N
	   IF(K(I,3).EQ.1.OR.K(I,3).EQ.2.OR.
     &                  ABS(K(I,2)).GT.30) GO TO 200
		IF(K(I,2).GT.21.AND.K(I,2).LE.30) THEN
			NDR=NDR+1
			IADR(NDR,1)=JP
			IADR(NDR,2)=JT
			KFDR(NDR)=K(I,2)
			PDR(NDR,1)=P(I,1)
			PDR(NDR,2)=P(I,2)
			PDR(NDR,3)=P(I,3)
			PDR(NDR,4)=P(I,4)
			PDR(NDR,5)=P(I,5)
                        VDR(NDR,1)=V(I,1)
                        VDR(NDR,2)=V(I,2)
                        VDR(NDR,3)=V(I,3)
                        VDR(NDR,4)=V(I,4)
C************************************************************
			GO TO 200
C************************correction made on Oct. 14,1994*****
		ENDIF
		IF(K(I,3).EQ.3.AND.(K(I,2).NE.21.OR.
     &                              I_INIRAD.EQ.0)) THEN
			PXP=PXP+P(I,1)
			PYP=PYP+P(I,2)
			PZP=PZP+P(I,3)
			PEP=PEP+P(I,4)
			GO TO 200
		ENDIF
		IF(K(I,3).EQ.4.AND.(K(I,2).NE.21.OR.
     &                                I_INIRAD.EQ.0)) THEN
			PXT=PXT+P(I,1)
			PYT=PYT+P(I,2)
			PZT=PZT+P(I,3)
			PET=PET+P(I,4)
			GO TO 200
		ENDIF
		JPP=JPP+1
		IP(JPP,1)=I
		IP(JPP,2)=0
		IF(K(I,2).NE.21) THEN
			IF(K(I,2).GT.0) THEN
				LPQ=LPQ+1
				IPQ(LPQ)=JPP
				IP(JPP,2)=LPQ
			ELSE IF(K(I,2).LT.0) THEN
				LPB=LPB+1
				IPB(LPB)=JPP
				IP(JPP,2)=-LPB
			ENDIF
		ENDIF
200	CONTINUE
	IF(LPQ.NE.LPB) THEN
	   MISS=MISS+1
	   IF(MISS.LE.50) GO TO 155
	   WRITE(6,*) LPQ,LPB, 'Q-QBAR NOT CONSERVED OR NOT MATCHED'
	   JFLG=0
	   RETURN
	ENDIF

C**** The following will rearrange the partons so that a quark is***
C**** allways followed by an anti-quark ****************************
	J=0
220	J=J+1
	IF(J.GT.JPP) GO TO 222
	IF(IP(J,2).EQ.0) GO TO 220
		LP=ABS(IP(J,2))
		IP1=IP(J,1)
		IP2=IP(J,2)
		IP(J,1)=IP(IPQ(LP),1)
		IP(J,2)=IP(IPQ(LP),2)
		IP(IPQ(LP),1)=IP1
		IP(IPQ(LP),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPQ(LP)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPQ(LP)
		ENDIF
		IPQ(LP)=J
C		********replace J with a quark
		IP1=IP(J+1,1)
		IP2=IP(J+1,2)
		IP(J+1,1)=IP(IPB(LP),1)
		IP(J+1,2)=IP(IPB(LP),2)
		IP(IPB(LP),1)=IP1
		IP(IPB(LP),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPB(LP)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPB(LP)
		ENDIF
C		******** replace J+1 with an anti-quark
		IPB(LP)=J+1
		J=J+1
		GO TO 220

222	CONTINUE
	IF(LPQ.GE.1) THEN
		DO 240 L0=2,LPQ
			IP1=IP(2*L0-3,1)
			IP2=IP(2*L0-3,2)
			IP(2*L0-3,1)=IP(IPQ(L0),1)
			IP(2*L0-3,2)=IP(IPQ(L0),2)
			IP(IPQ(L0),1)=IP1
			IP(IPQ(L0),2)=IP2
			IF(IP2.GT.0) THEN
				IPQ(IP2)=IPQ(L0)
			ELSE IF(IP2.LT.0) THEN
				IPB(-IP2)=IPQ(L0)
			ENDIF
			IPQ(L0)=2*L0-3
C
			IP1=IP(2*L0-2,1)
			IP2=IP(2*L0-2,2)
			IP(2*L0-2,1)=IP(IPB(L0),1)
			IP(2*L0-2,2)=IP(IPB(L0),2)
			IP(IPB(L0),1)=IP1
			IP(IPB(L0),2)=IP2
			IF(IP2.GT.0) THEN
				IPQ(IP2)=IPB(L0)
			ELSE IF(IP2.LT.0) THEN
				IPB(-IP2)=IPB(L0)
			ENDIF
			IPB(L0)=2*L0-2
240		CONTINUE
C		********move all the qqbar pair to the front of 
C				the list, except the first pair
		IP1=IP(2*LPQ-1,1)
		IP2=IP(2*LPQ-1,2)
		IP(2*LPQ-1,1)=IP(IPQ(1),1)
		IP(2*LPQ-1,2)=IP(IPQ(1),2)
		IP(IPQ(1),1)=IP1
		IP(IPQ(1),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPQ(1)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPQ(1)
		ENDIF
		IPQ(1)=2*LPQ-1
C		********move the first quark to the beginning of
C				the last string system
		IP1=IP(JPP,1)
		IP2=IP(JPP,2)
		IP(JPP,1)=IP(IPB(1),1)
		IP(JPP,2)=IP(IPB(1),2)
		IP(IPB(1),1)=IP1
		IP(IPB(1),2)=IP2
		IF(IP2.GT.0) THEN
			IPQ(IP2)=IPB(1)
		ELSE IF(IP2.LT.0) THEN
			IPB(-IP2)=IPB(1)
		ENDIF
		IPB(1)=JPP
C		********move the first anti-quark to the end of the 
C			last string system
	ENDIF
	IF(NSG.GE.MXSG) THEN
	   JFLG=0
	   WRITE(6,*) 'number of jets forming single strings exceeds'
	   WRITE(6,*) 'the common block size'
	   RETURN
	ENDIF
	IF(JPP.GT.MXSJ) THEN
	   JFLG=0
	   WRITE(6,*) 'number of partons per single jet system'
	   WRITE(6,*) 'exceeds the common block size'
	   RETURN
	ENDIF
C		********check the bounds of common block size
	NSG=NSG+1
	NJSG(NSG)=JPP
	IASG(NSG,1)=JP
	IASG(NSG,2)=JT
	IASG(NSG,3)=0
	DO 300 I=1,JPP
		K1SG(NSG,I)=2
		K2SG(NSG,I)=K(IP(I,1),2)
		IF(K2SG(NSG,I).LT.0) K1SG(NSG,I)=1
		PXSG(NSG,I)=P(IP(I,1),1)
		PYSG(NSG,I)=P(IP(I,1),2)
		PZSG(NSG,I)=P(IP(I,1),3)
		PESG(NSG,I)=P(IP(I,1),4)
		PMSG(NSG,I)=P(IP(I,1),5)
300	CONTINUE
	K1SG(NSG,1)=2
	K1SG(NSG,JPP)=1
C******* reset the energy-momentum of incoming particles ********
900	PP(JP,1)=PXP
	PP(JP,2)=PYP
	PP(JP,3)=PZP
	PP(JP,4)=PEP
	PP(JP,5)=0.0
	PT(JT,1)=PXT
	PT(JT,2)=PYT
	PT(JT,3)=PZT
	PT(JT,4)=PET
	PT(JT,5)=0.0

	NFP(JP,6)=NFP(JP,6)+1
	NFT(JT,6)=NFT(JT,6)+1
	RETURN
C
1000	JFLG=-1
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) 'Fatal HIJHRD error'
	WRITE(6,*) JP, ' proj E+,E-',EPP,EPM,' status',NFP(JP,5)
	WRITE(6,*) JT, ' targ E+,E_',ETP,ETM,' status',NFT(JT,5)
	RETURN
	END
C
C
C
C
C
	SUBROUTINE JETINI(JP,JT,I_TRIG)
C*******Initialize PYTHIA for jet production**********************
C	I_TRIG=0: for normal processes
C	I_TRIG=1: for triggered processes
C       JP: sequence number of the projectile
C       JT: sequence number of the target
C     For A+A collisions, one has to initilize pythia
C     separately for each type of collisions, pp, pn,np and nn,
C     or hp and hn for hA collisions. In this subroutine we use the following
C     catalogue for different type of collisions:
C     h+h: h+h (I_TYPE=1)
C     h+A: h+p (I_TYPE=1), h+n (I_TYPE=2)
C     A+h: p+h (I_TYPE=1), n+h (I_TYPE=2)
C     A+A: p+p (I_TYPE=1), p+n (I_TYPE=2), n+p (I_TYPE=3), n+n (I_TYPE=4)
C*****************************************************************
	CHARACTER BEAM*16,TARG*16
	DIMENSION XSEC0(8,0:200),COEF0(8,200,20),INI(8),
     &		MINT44(8),MINT45(8)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
        COMMON/HIPYINT/MINT4,MINT5,ATCO(200,20),ATXS(0:200)
C
        COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
	COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
        COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200)
        COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
        COMMON/PYINT1/MINT(400),VINT(400)
        COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2)
        COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3)
        SAVE
	DATA INI/8*0/I_LAST/-1/
C
        IHNT2(11)=JP
        IHNT2(12)=JT
        IF(IHNT2(5).NE.0 .AND. IHNT2(6).NE.0) THEN
           I_TYPE=1
        ELSE IF(IHNT2(5).NE.0 .AND. IHNT2(6).EQ.0) THEN
           I_TYPE=1
           IF(NFT(JT,4).EQ.2112) I_TYPE=2
        ELSE IF(IHNT2(5).EQ.0 .AND. IHNT2(6).NE.0) THEN
           I_TYPE=1
           IF(NFP(JP,4).EQ.2112) I_TYPE=2
        ELSE
           IF(NFP(JP,4).EQ.2212 .AND. NFT(JT,4).EQ.2212) THEN
              I_TYPE=1
           ELSE IF(NFP(JP,4).EQ.2212 .AND. NFT(JT,4).EQ.2112) THEN
              I_TYPE=2
           ELSE IF(NFP(JP,4).EQ.2112 .AND. NFT(JT,4).EQ.2212) THEN
              I_TYPE=3
           ELSE
              I_TYPE=4
           ENDIF
        ENDIF
c
	IF(I_TRIG.NE.0) GO TO 160
        IF(I_TRIG.EQ.I_LAST) GO TO 150
        MSTP(2)=2
c			********second order running alpha_strong
        MSTP(33)=1
        PARP(31)=HIPR1(17)
C			********inclusion of K factor
        MSTP(51)=3
C			********Duke-Owens set 1 structure functions
        MSTP(61)=1
C			********INITIAL STATE RADIATION
        MSTP(71)=1
C			********FINAL STATE RADIATION
        IF(IHPR2(2).EQ.0.OR.IHPR2(2).EQ.2) MSTP(61)=0
        IF(IHPR2(2).EQ.0.OR.IHPR2(2).EQ.1) MSTP(71)=0
c
        MSTP(81)=0
C			******** NO MULTIPLE INTERACTION
        MSTP(82)=1
C			*******STRUCTURE OF MUTLIPLE INTERACTION
        MSTP(111)=0
C		********frag off(have to be done by local call)
        IF(IHPR2(10).EQ.0) MSTP(122)=0
C		********No printout of initialization information
        PARP(81)=HIPR1(8)
        CKIN(5)=HIPR1(8)
        CKIN(3)=HIPR1(8)
        CKIN(4)=HIPR1(9)
        IF(HIPR1(9).LE.HIPR1(8)) CKIN(4)=-1.0
        CKIN(9)=-10.0
        CKIN(10)=10.0
        MSEL=0
        DO 100 ISUB=1,200
           MSUB(ISUB)=0
 100    CONTINUE
        MSUB(11)=1
        MSUB(12)=1
        MSUB(13)=1
        MSUB(28)=1
        MSUB(53)=1
        MSUB(68)=1
        MSUB(81)=1
        MSUB(82)=1
        DO 110 J=1,MIN(8,MDCY(21,3))
 110    MDME(MDCY(21,2)+J-1,1)=0
        ISEL=4
        IF(HINT1(1).GE.20.0 .and. IHPR2(18).EQ.1) ISEL=5
        MDME(MDCY(21,2)+ISEL-1,1)=1
C			********QCD subprocesses
        MSUB(14)=1
        MSUB(18)=1
        MSUB(29)=1
C                       ******* direct photon production
 150    IF(INI(I_TYPE).NE.0) GO TO 800
	GO TO 400
C
C	*****triggered subprocesses, jet, photon, heavy quark and DY
C
 160    I_TYPE=4+I_TYPE
        IF(I_TRIG.EQ.I_LAST) GO TO 260
        PARP(81)=ABS(HIPR1(10))-0.25
        CKIN(5)=ABS(HIPR1(10))-0.25
        CKIN(3)=ABS(HIPR1(10))-0.25
        CKIN(4)=ABS(HIPR1(10))+0.25
        IF(HIPR1(10).LT.HIPR1(8)) CKIN(4)=-1.0
c
        MSEL=0
        DO 101 ISUB=1,200
           MSUB(ISUB)=0
 101    CONTINUE
        IF(IHPR2(3).EQ.1) THEN
           MSUB(11)=1
           MSUB(12)=1
           MSUB(13)=1
           MSUB(28)=1
           MSUB(53)=1
           MSUB(68)=1
           MSUB(81)=1
           MSUB(82)=1
           MSUB(14)=1
           MSUB(18)=1
           MSUB(29)=1
           DO 102 J=1,MIN(8,MDCY(21,3))
 102	   MDME(MDCY(21,2)+J-1,1)=0
           ISEL=4
           IF(HINT1(1).GE.20.0 .and. IHPR2(18).EQ.1) ISEL=5
           MDME(MDCY(21,2)+ISEL-1,1)=1
C			********QCD subprocesses
        ELSE IF(IHPR2(3).EQ.2) THEN
           MSUB(14)=1
           MSUB(18)=1
           MSUB(29)=1
C		********Direct photon production
c		q+qbar->g+gamma,q+qbar->gamma+gamma, q+g->q+gamma
        ELSE IF(IHPR2(3).EQ.3) THEN
           CKIN(3)=MAX(0.0,HIPR1(10))
           CKIN(5)=HIPR1(8)
           PARP(81)=HIPR1(8)
           MSUB(81)=1
           MSUB(82)=1
           DO 105 J=1,MIN(8,MDCY(21,3))
 105	   MDME(MDCY(21,2)+J-1,1)=0
           ISEL=4
           IF(HINT1(1).GE.20.0 .and. IHPR2(18).EQ.1) ISEL=5
           MDME(MDCY(21,2)+ISEL-1,1)=1
C             **********Heavy quark production
        ENDIF
260	IF(INI(I_TYPE).NE.0) GO TO 800
C
C
400	INI(I_TYPE)=1
	IF(IHPR2(10).EQ.0) MSTP(122)=0
	IF(NFP(JP,4).EQ.2212) THEN
		BEAM='P'
	ELSE IF(NFP(JP,4).EQ.-2212) THEN
		BEAM='P~'
	ELSE IF(NFP(JP,4).EQ.2112) THEN
		BEAM='N'
	ELSE IF(NFP(JP,4).EQ.-2112) THEN
		BEAM='N~'
	ELSE IF(NFP(JP,4).EQ.211) THEN
		BEAM='PI+'
	ELSE IF(NFP(JP,4).EQ.-211) THEN
		BEAM='PI-'
	ELSE IF(NFP(JP,4).EQ.321) THEN
		BEAM='PI+'
	ELSE IF(NFP(JP,4).EQ.-321) THEN
		BEAM='PI-'
	ELSE
		WRITE(6,*) 'unavailable beam type', NFP(JP,4)
	ENDIF
	IF(NFT(JT,4).EQ.2212) THEN
		TARG='P'
	ELSE IF(NFT(JT,4).EQ.-2212) THEN
		TARG='P~'
	ELSE IF(NFT(JT,4).EQ.2112) THEN
		TARG='N'
	ELSE IF(NFT(JT,4).EQ.-2112) THEN
		TARG='N~'
	ELSE IF(NFT(JT,4).EQ.211) THEN
		TARG='PI+'
	ELSE IF(NFT(JT,4).EQ.-211) THEN
		TARG='PI-'
	ELSE IF(NFT(JT,4).EQ.321) THEN
		TARG='PI+'
	ELSE IF(NFT(JT,4).EQ.-321) THEN
		TARG='PI-'
	ELSE
		WRITE(6,*) 'unavailable target type', NFT(JT,4)
	ENDIF
C
	IHNT2(16)=1
C       ******************indicate for initialization use when
C                         structure functions are called in PYTHIA
C
	CALL PYINIT('CMS',BEAM,TARG,HINT1(1))
	MINT4=MINT(44)
	MINT5=MINT(45)
	MINT44(I_TYPE)=MINT(44)
	MINT45(I_TYPE)=MINT(45)
	ATXS(0)=XSEC(0,1)
	XSEC0(I_TYPE,0)=XSEC(0,1)
	DO 500 I=1,200
		ATXS(I)=XSEC(I,1)
		XSEC0(I_TYPE,I)=XSEC(I,1)
		DO 500 J=1,20
			ATCO(I,J)=COEF(I,J)
			COEF0(I_TYPE,I,J)=COEF(I,J)
500	CONTINUE
C
	I_LAST=I_TRIG
	IHNT2(16)=0
C
	RETURN
C		********Store the initialization information for
C				late use
C
C
800	MINT(44)=MINT44(I_TYPE)
	MINT(45)=MINT45(I_TYPE)
	MINT4=MINT(44)
	MINT5=MINT(45)
	XSEC(0,1)=XSEC0(I_TYPE,0)
	ATXS(0)=XSEC(0,1)
	DO 900 I=1,200
		XSEC(I,1)=XSEC0(I_TYPE,I)
		ATXS(I)=XSEC(I,1)
	DO 900 J=1,20
		COEF(I,J)=COEF0(I_TYPE,I,J)
		ATCO(I,J)=COEF(I,J)
900	CONTINUE
        I_LAST=I_TRIG
        MINT(11)=NFP(JP,4)
        MINT(12)=NFT(JT,4)
	RETURN
	END
C            
C
C
	SUBROUTINE HIJINI
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HIJJET4/NDR,IADR(900,2),KFDR(900),PDR(900,5),VDR(900,4)
	COMMON/RANSEED/NSEED
	SAVE  
C****************Reset the momentum of initial particles************
C             and assign flavors to the proj and targ string       *
C*******************************************************************
	NSG=0
	NDR=0
	IPP=2212
	IPT=2212
	IF(IHNT2(5).NE.0) IPP=IHNT2(5)
	IF(IHNT2(6).NE.0) IPT=IHNT2(6)
C		********in case the proj or targ is a hadron.
C
	DO 100 I=1,IHNT2(1)
	PP(I,1)=0.0
	PP(I,2)=0.0
	PP(I,3)=SQRT(HINT1(1)**2/4.0-HINT1(8)**2)
	PP(I,4)=HINT1(1)/2
	PP(I,5)=HINT1(8)
	PP(I,6)=0.0
	PP(I,7)=0.0
	PP(I,8)=0.0
	PP(I,9)=0.0
	PP(I,10)=0.0
	NFP(I,3)=IPP
	NFP(I,4)=IPP
	NFP(I,5)=0
	NFP(I,6)=0
	NFP(I,7)=0
	NFP(I,8)=0
	NFP(I,9)=0
	NFP(I,10)=0
	NFP(I,11)=0
	NPJ(I)=0
	IF(I.GT.ABS(IHNT2(2))) NFP(I,3)=2112
	CALL ATTFLV(NFP(I,3),IDQ,IDQQ)
	NFP(I,1)=IDQ
	NFP(I,2)=IDQQ
	NFP(I,15)=-1
	IF(ABS(IDQ).GT.1000.OR.(ABS(IDQ*IDQQ).LT.100.AND.
     &		RLU(NSEED).LT.0.5)) NFP(I,15)=1
	PP(I,14)=ULMASS(IDQ)
	PP(I,15)=ULMASS(IDQQ)
100	CONTINUE
C
	DO 200 I=1,IHNT2(3)
	PT(I,1)=0.0
	PT(I,2)=0.0
	PT(I,3)=-SQRT(HINT1(1)**2/4.0-HINT1(9)**2)
	PT(I,4)=HINT1(1)/2.0
	PT(I,5)=HINT1(9)
	PT(I,6)=0.0
	PT(I,7)=0.0
	PT(I,8)=0.0
	PT(I,9)=0.0
	PT(I,10)=0.0
	NFT(I,3)=IPT
	NFT(I,4)=IPT
	NFT(I,5)=0
	NFT(I,6)=0
	NFT(I,7)=0
	NFT(I,8)=0
	NFT(I,9)=0
	NFT(I,10)=0
	NFT(I,11)=0
	NTJ(I)=0
	IF(I.GT.ABS(IHNT2(4))) NFT(I,3)=2112
	CALL ATTFLV(NFT(I,3),IDQ,IDQQ)
	NFT(I,1)=IDQ
	NFT(I,2)=IDQQ
	NFT(I,15)=1
	IF(ABS(IDQ).GT.1000.OR.(ABS(IDQ*IDQQ).LT.100.AND.
     &			RLU(NSEED).LT.0.5)) NFT(I,15)=-1
	PT(I,14)=ULMASS(IDQ)
	PT(I,15)=ULMASS(IDQQ)
200	CONTINUE
	RETURN
	END
C
C
C
	SUBROUTINE ATTFLV(ID,IDQ,IDQQ)
	COMMON/RANSEED/NSEED
	SAVE  
C
	IF(ABS(ID).LT.100) THEN
		NSIGN=1
		IDQ=ID/100
		IDQQ=-ID/10+IDQ*10
		IF(ABS(IDQ).EQ.3) NSIGN=-1
		IDQ=NSIGN*IDQ
		IDQQ=NSIGN*IDQQ
		IF(IDQ.LT.0) THEN
			ID0=IDQ
			IDQ=IDQQ
			IDQQ=ID0
		ENDIF
		RETURN
	ENDIF
C		********return ID of quark(IDQ) and anti-quark(IDQQ)
C			for pions and kaons
c
C	Return LU ID for quarks and diquarks for proton(ID=2212) 
C	anti-proton(ID=-2212) and nuetron(ID=2112)
C	LU ID for d=1,u=2, (ud)0=2101, (ud)1=2103, 
C       (dd)1=1103,(uu)1=2203.
C	Use SU(6)  weight  proton=1/3d(uu)1 + 1/6u(ud)1 + 1/2u(ud)0
C			  nurtron=1/3u(dd)1 + 1/6d(ud)1 + 1/2d(ud)0
C 
	IDQ=2
	IF(ABS(ID).EQ.2112) IDQ=1
	IDQQ=2101
	X=RLU(NSEED)
	IF(X.LE.0.5) GO TO 30
	IF(X.GT.0.666667) GO TO 10
	IDQQ=2103
	GO TO 30
10	IDQ=1
	IDQQ=2203
	IF(ABS(ID).EQ.2112) THEN
		IDQ=2
		IDQQ=1103
	ENDIF
30	IF(ID.LT.0) THEN
		ID00=IDQQ
		IDQQ=-IDQ
		IDQ=-ID00
	ENDIF
	RETURN
	END	
C
C*******************************************************************
C	This subroutine performs elastic scatterings and possible 
C	elastic cascading within their own nuclei
c*******************************************************************
	SUBROUTINE HIJCSC(JP,JT)
	DIMENSION PSC1(5),PSC2(5)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/RANSEED/NSEED
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	SAVE  
	IF(JP.EQ.0 .OR. JT.EQ.0) GO TO 25
	DO 10 I=1,5
	PSC1(I)=PP(JP,I)
	PSC2(I)=PT(JT,I)
10	CONTINUE
	CALL HIJELS(PSC1,PSC2)
	DPP1=PSC1(1)-PP(JP,1)
	DPP2=PSC1(2)-PP(JP,2)
	DPT1=PSC2(1)-PT(JT,1)
	DPT2=PSC2(2)-PT(JT,2)
	PP(JP,6)=PP(JP,6)+DPP1/2.0
	PP(JP,7)=PP(JP,7)+DPP2/2.0
	PP(JP,8)=PP(JP,8)+DPP1/2.0
	PP(JP,9)=PP(JP,9)+DPP2/2.0
	PT(JT,6)=PT(JT,6)+DPT1/2.0
	PT(JT,7)=PT(JT,7)+DPT2/2.0
	PT(JT,8)=PT(JT,8)+DPT1/2.0
	PT(JT,9)=PT(JT,9)+DPT2/2.0
	DO 20 I=1,4
	PP(JP,I)=PSC1(I)
	PT(JT,I)=PSC2(I)
20	CONTINUE
	NFP(JP,5)=MAX(1,NFP(JP,5))
	NFT(JT,5)=MAX(1,NFT(JT,5))
C		********Perform elastic scattering between JP and JT
	RETURN
C		********The following is for possible elastic cascade
c
25	IF(JP.EQ.0) GO TO 45
	PABS=SQRT(PP(JP,1)**2+PP(JP,2)**2+PP(JP,3)**2)
	BX=PP(JP,1)/PABS
	BY=PP(JP,2)/PABS
	BZ=PP(JP,3)/PABS
	DO 40 I=1,IHNT2(1)
		IF(I.EQ.JP) GO TO 40
		DX=YP(1,I)-YP(1,JP)
		DY=YP(2,I)-YP(2,JP)
		DZ=YP(3,I)-YP(3,JP)
		DIS=DX*BX+DY*BY+DZ*BZ
		IF(DIS.LE.0) GO TO 40
		BB=DX**2+DY**2+DZ**2-DIS**2
		R2=BB*HIPR1(40)/HIPR1(31)/0.1
C		********mb=0.1*fm, YP is in fm,HIPR1(31) is in mb
		GS=1.0-EXP(-(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
     &			*ROMG(R2))**2
		GS0=1.0-EXP(-(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
     &			*ROMG(0.0))**2
		IF(RLU(NSEED).GT.GS/GS0) GO TO 40
		DO 30 K=1,5
			PSC1(K)=PP(JP,K)
			PSC2(K)=PP(I,K)
30		CONTINUE
		CALL HIJELS(PSC1,PSC2)
		DPP1=PSC1(1)-PP(JP,1)
		DPP2=PSC1(2)-PP(JP,2)
		DPT1=PSC2(1)-PP(I,1)
		DPT2=PSC2(2)-PP(I,2)
		PP(JP,6)=PP(JP,6)+DPP1/2.0
		PP(JP,7)=PP(JP,7)+DPP2/2.0
		PP(JP,8)=PP(JP,8)+DPP1/2.0
		PP(JP,9)=PP(JP,9)+DPP2/2.0
		PP(I,6)=PP(I,6)+DPT1/2.0
		PP(I,7)=PP(I,7)+DPT2/2.0
		PP(I,8)=PP(I,8)+DPT1/2.0
		PP(I,9)=PP(I,9)+DPT2/2.0
		DO 35 K=1,5
			PP(JP,K)=PSC1(K)
			PP(I,K)=PSC2(K)
35		CONTINUE
		NFP(I,5)=MAX(1,NFP(I,5))
		GO TO 45
40	CONTINUE
45	IF(JT.EQ.0) GO TO 80
	PABS=SQRT(PT(JT,1)**2+PT(JT,2)**2+PT(JT,3)**2)
	BX=PT(JT,1)/PABS
	BY=PT(JT,2)/PABS
	BZ=PT(JT,3)/PABS
	DO 70 I=1,IHNT2(3)
		IF(I.EQ.JT) GO TO 70
		DX=YT(1,I)-YT(1,JT)
		DY=YT(2,I)-YT(2,JT)
		DZ=YT(3,I)-YT(3,JT)
		DIS=DX*BX+DY*BY+DZ*BZ
		IF(DIS.LE.0) GO TO 70
		BB=DX**2+DY**2+DZ**2-DIS**2
		R2=BB*HIPR1(40)/HIPR1(31)/0.1
C		********mb=0.1*fm, YP is in fm,HIPR1(31) is in mb
		GS=(1.0-EXP(-(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
     &			*ROMG(R2)))**2
		GS0=(1.0-EXP(-(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
     &			*ROMG(0.0)))**2
		IF(RLU(NSEED).GT.GS/GS0) GO TO 70
		DO 60 K=1,5
			PSC1(K)=PT(JT,K)
			PSC2(K)=PT(I,K)
60		CONTINUE
		CALL HIJELS(PSC1,PSC2)
		DPP1=PSC1(1)-PT(JT,1)
		DPP2=PSC1(2)-PT(JT,2)
		DPT1=PSC2(1)-PT(I,1)
		DPT2=PSC2(2)-PT(I,2)
		PT(JT,6)=PT(JT,6)+DPP1/2.0
		PT(JT,7)=PT(JT,7)+DPP2/2.0
		PT(JT,8)=PT(JT,8)+DPP1/2.0
		PT(JT,9)=PT(JT,9)+DPP2/2.0
		PT(I,6)=PT(I,6)+DPT1/2.0
		PT(I,7)=PT(I,7)+DPT2/2.0
		PT(I,8)=PT(I,8)+DPT1/2.0
		PT(I,9)=PT(I,9)+DPT2/2.0
		DO 65 K=1,5
			PT(JT,K)=PSC1(K)
			PT(I,K)=PSC2(K)
65		CONTINUE
		NFT(I,5)=MAX(1,NFT(I,5))
		GO TO 80
70	CONTINUE
80	RETURN
	END
C
C
C*******************************************************************
CThis subroutine performs elastic scattering between two nucleons
C
C*******************************************************************
	SUBROUTINE HIJELS(PSC1,PSC2)
	IMPLICIT DOUBLE PRECISION(D)
	DIMENSION PSC1(5),PSC2(5)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/RANSEED/NSEED
	SAVE  
C
	CC=1.0-HINT1(12)/HINT1(13)
	RR=(1.0-CC)*HINT1(13)/HINT1(12)/(1.0-HIPR1(33))-1.0
	BB=0.5*(3.0+RR+SQRT(9.0+10.0*RR+RR**2))
	EP=SQRT((PSC1(1)-PSC2(1))**2+(PSC1(2)-PSC2(2))**2
     &		+(PSC1(3)-PSC2(3))**2)
	IF(EP.LE.0.1) RETURN
	ELS0=98.0/EP+52.0*(1.0+RR)**2
	PCM1=PSC1(1)+PSC2(1)
	PCM2=PSC1(2)+PSC2(2)
	PCM3=PSC1(3)+PSC2(3)
	ECM=PSC1(4)+PSC2(4)
	AM1=PSC1(5)**2
	AM2=PSC2(5)**2
	AMM=ECM**2-PCM1**2-PCM2**2-PCM3**2
	IF(AMM.LE.PSC1(5)+PSC2(5)) RETURN
C		********elastic scattering only when approaching
C				to each other
	PMAX=(AMM**2+AM1**2+AM2**2-2.0*AMM*AM1-2.0*AMM*AM2
     &			-2.0*AM1*AM2)/4.0/AMM
	PMAX=ABS(PMAX)
20	TT=RLU(NSEED)*MIN(PMAX,1.5)
	ELS=98.0*EXP(-2.8*TT)/EP
     &		+52.0*EXP(-9.2*TT)*(1.0+RR*EXP(-4.6*(BB-1.0)*TT))**2
	IF(RLU(NSEED).GT.ELS/ELS0) GO TO 20
	PHI=2.0*HIPR1(40)*RLU(NSEED)
C
	DBX=PCM1/ECM
	DBY=PCM2/ECM
	DBZ=PCM3/ECM
        DB=SQRT(DBX**2+DBY**2+DBZ**2)
        IF(DB.GT.0.99999999D0) THEN 
          DBX=DBX*(0.99999999D0/DB) 
          DBY=DBY*(0.99999999D0/DB) 
          DBZ=DBZ*(0.99999999D0/DB) 
          DB=0.99999999D0   
	  WRITE(6,*) ' (HIJELS) boost vector too large' 
C		********Rescale boost vector if too close to unity. 
        ENDIF   
        DGA=1D0/SQRT(1D0-DB**2)      
C
	DP1=SQRT(TT)*SIN(PHI)
	DP2=SQRT(TT)*COS(PHI)
	DP3=SQRT(PMAX-TT)
	DP4=SQRT(PMAX+AM1)
        DBP=DBX*DP1+DBY*DP2+DBZ*DP3   
        DGABP=DGA*(DGA*DBP/(1D0+DGA)+DP4) 
        PSC1(1)=DP1+DGABP*DBX
        PSC1(2)=DP2+DGABP*DBY  
        PSC1(3)=DP3+DGABP*DBZ  
        PSC1(4)=DGA*(DP4+DBP)    
C	
	DP1=-SQRT(TT)*SIN(PHI)
	DP2=-SQRT(TT)*COS(PHI)
	DP3=-SQRT(PMAX-TT)
	DP4=SQRT(PMAX+AM2)
        DBP=DBX*DP1+DBY*DP2+DBZ*DP3   
        DGABP=DGA*(DGA*DBP/(1D0+DGA)+DP4) 
        PSC2(1)=DP1+DGABP*DBX
        PSC2(2)=DP2+DGABP*DBY  
        PSC2(3)=DP3+DGABP*DBZ  
        PSC2(4)=DGA*(DP4+DBP)
	RETURN
	END
C
C	
C*******************************************************************
C	   				                           *
C		Subroutine HIJSFT		                   *
C						                   *
C  Scatter two excited strings, JP from proj and JT from target    *
C*******************************************************************
	SUBROUTINE HIJSFT(JP,JT,JOUT,IERROR)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
	COMMON/RANSEED/NSEED
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &               PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &               PJPM(300,500),NTJ(300),KFTJ(300,500),
     &               PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &               PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/DPMCOM1/JJP,JJT,AMP,AMT,APX0,ATX0,AMPN,AMTN,AMP0,AMT0,
     &		NFDP,NFDT,WP,WM,SW,XREMP,XREMT,DPKC1,DPKC2,PP11,PP12,
     &		PT11,PT12,PTP2,PTT2
	COMMON/DPMCOM2/NDPM,KDPM(20,2),PDPM1(20,5),PDPM2(20,5)
	SAVE  
C*******************************************************************
C	JOUT-> the number
C	of hard scatterings preceding this soft collision. 
C       IHNT2(13)-> 1=
C	double diffrac 2=single diffrac, 3=non-single diffrac.
C*******************************************************************
	IERROR=0
	JJP=JP
	JJT=JT
	NDPM=0
	IOPMAIN=0
	IF(JP.GT.IHNT2(1) .OR. JT.GT.IHNT2(3)) RETURN
	I_SNG=0

	EPP=PP(JP,4)+PP(JP,3)
	EPM=PP(JP,4)-PP(JP,3)
	ETP=PT(JT,4)+PT(JT,3)
	ETM=PT(JT,4)-PT(JT,3)

	WP=EPP+ETP
	WM=EPM+ETM
	SW=WP*WM
C		********total W+,W- and center-of-mass energy

	IF(WP.LT.0.0 .OR. WM.LT.0.0) GO TO 1000

	IF(JOUT.EQ.0) THEN
		IF(EPP.LT.0.0) GO TO 1000
		IF(EPM.LT.0.0) GO TO 1000
		IF(ETP.LT.0.0) GO TO 1000
		IF(ETM.LT.0.0) GO TO 1000    
		IF(EPP/(EPM+0.01).LE.ETP/(ETM+0.01)) RETURN
	ENDIF
C		********For strings which does not follow a jet-prod,
C			scatter only if Ycm(JP)>Ycm(JT). When jets
C			are produced just before this collision
C			this requirement has already be enforced
C			(see SUBROUTINE HIJHRD)
	IHNT2(11)=JP
	IHNT2(12)=JT
C
C
C
	MISS=0
	PKC1=0.0
	PKC2=0.0
	PKC11=0.0
	PKC12=0.0
	PKC21=0.0
	PKC22=0.0
	DPKC11=0.0
	DPKC12=0.0
	DPKC21=0.0
	DPKC22=0.0
	IF(NFP(JP,10).EQ.1.OR.NFT(JT,10).EQ.1) THEN
	   IF(NFP(JP,10).EQ.1) THEN
	      PHI1=ULANGL(PP(JP,10),PP(JP,11))
	      PPJET=SQRT(PP(JP,10)**2+PP(JP,11)**2)
	      PKC1=PPJET
	      PKC11=PP(JP,10)
	      PKC12=PP(JP,11)
	   ENDIF
	   IF(NFT(JT,10).EQ.1) THEN
	      PHI2=ULANGL(PT(JT,10),PT(JT,11))
	      PTJET=SQRT(PT(JT,10)**2+PT(JT,11)**2)
	      PKC2=PTJET
	      PKC21=PT(JT,10)
	      PKC22=PT(JT,11)
	   ENDIF
	   IF(IHPR2(4).GT.0.AND.IHNT2(1).GT.1.AND.IHNT2(3).GT.1) THEN
	      IF(NFP(JP,10).EQ.0) THEN
		 PHI=-PHI2
	      ELSE IF(NFT(JT,10).EQ.0) THEN
		 PHI=PHI1
	      ELSE
		 PHI=(PHI1+PHI2-HIPR1(40))/2.0
	      ENDIF
	      BX=HINT1(19)*COS(HINT1(20))
	      BY=HINT1(19)*SIN(HINT1(20))
	      XP0=YP(1,JP)
	      YP0=YP(2,JP)
	      XT0=YT(1,JT)+BX
	      YT0=YT(2,JT)+BY
	      R1=MAX(1.2*IHNT2(1)**0.3333333,
     &               SQRT(XP0**2+YP0**2))
	      R2=MAX(1.2*IHNT2(3)**0.3333333,
     &               SQRT((XT0-BX)**2+(YT0-BY)**2))
	      IF(ABS(COS(PHI)).LT.1.0E-5) THEN
		 DD1=R1
		 DD2=R1
		 DD3=ABS(BY+SQRT(R2**2-(XP0-BX)**2)-YP0)
		 DD4=ABS(BY-SQRT(R2**2-(XP0-BX)**2)-YP0)
		 GO TO 5
	      ENDIF
	      BB=2.0*SIN(PHI)*(COS(PHI)*YP0-SIN(PHI)*XP0)
	      CC=(YP0**2-R1**2)*COS(PHI)**2+XP0*SIN(PHI)*(
     &				XP0*SIN(PHI)-2.0*YP0*COS(PHI))
	      DD=BB**2-4.0*CC
	      IF(DD.LT.0.0) GO TO 10
	      XX1=(-BB+SQRT(DD))/2.0
	      XX2=(-BB-SQRT(DD))/2.0
	      DD1=ABS((XX1-XP0)/COS(PHI))
	      DD2=ABS((XX2-XP0)/COS(PHI))
C			
	      BB=2.0*SIN(PHI)*(COS(PHI)*(YT0-BY)-SIN(PHI)*XT0)-2.0*BX
	      CC=(BX**2+(YT0-BY)**2-R2**2)*COS(PHI)**2+XT0*SIN(PHI)
     &           *(XT0*SIN(PHI)-2.0*COS(PHI)*(YT0-BY))
     &		 -2.0*BX*SIN(PHI)*(COS(PHI)*(YT0-BY)-SIN(PHI)*XT0)
	      DD=BB**2-4.0*CC
	      IF(DD.LT.0.0) GO TO 10
	      XX1=(-BB+SQRT(DD))/2.0
	      XX2=(-BB-SQRT(DD))/2.0
	      DD3=ABS((XX1-XT0)/COS(PHI))
	      DD4=ABS((XX2-XT0)/COS(PHI))
C
 5	      DD1=MIN(DD1,DD3)
	      DD2=MIN(DD2,DD4)
	      IF(DD1.LT.HIPR1(13)) DD1=0.0
	      IF(DD2.LT.HIPR1(13)) DD2=0.0
	      IF(NFP(JP,10).EQ.1.AND.PPJET.GT.HIPR1(11)) THEN
		 DP1=DD1*HIPR1(14)/2.0
		 DP1=MIN(DP1,PPJET-HIPR1(11))
		 PKC1=PPJET-DP1
		 DPX1=COS(PHI1)*DP1
		 DPY1=SIN(PHI1)*DP1
		 PKC11=PP(JP,10)-DPX1
		 PKC12=PP(JP,11)-DPY1
		 IF(DP1.GT.0.0) THEN
		    CTHEP=PP(JP,12)/SQRT(PP(JP,12)**2+PPJET**2)
		    DPZ1=DP1*CTHEP/SQRT(1.0-CTHEP**2)
		    DPE1=SQRT(DPX1**2+DPY1**2+DPZ1**2)
		    EPPPRM=PP(JP,4)+PP(JP,3)-DPE1-DPZ1
		    EPMPRM=PP(JP,4)-PP(JP,3)-DPE1+DPZ1
		    IF(EPPPRM.LE.0.0.OR.EPMPRM.LE.0.0) GO TO 15
		    EPP=EPPPRM
		    EPM=EPMPRM
		    PP(JP,10)=PKC11
		    PP(JP,11)=PKC12
		    NPJ(JP)=NPJ(JP)+1
		    KFPJ(JP,NPJ(JP))=21
		    PJPX(JP,NPJ(JP))=DPX1
		    PJPY(JP,NPJ(JP))=DPY1
		    PJPZ(JP,NPJ(JP))=DPZ1
		    PJPE(JP,NPJ(JP))=DPE1
		    PJPM(JP,NPJ(JP))=0.0
		    PP(JP,3)=PP(JP,3)-DPZ1
		    PP(JP,4)=PP(JP,4)-DPE1
		 ENDIF
	      ENDIF
 15	      IF(NFT(JT,10).EQ.1.AND.PTJET.GT.HIPR1(11)) THEN
		 DP2=DD2*HIPR1(14)/2.0
		 DP2=MIN(DP2,PTJET-HIPR1(11))
		 PKC2=PTJET-DP2
		 DPX2=COS(PHI2)*DP2
		 DPY2=SIN(PHI2)*DP2
		 PKC21=PT(JT,10)-DPX2
		 PKC22=PT(JT,11)-DPY2
		 IF(DP2.GT.0.0) THEN
		    CTHET=PT(JT,12)/SQRT(PT(JT,12)**2+PTJET**2)
		    DPZ2=DP2*CTHET/SQRT(1.0-CTHET**2)
		    DPE2=SQRT(DPX2**2+DPY2**2+DPZ2**2)
		    ETPPRM=PT(JT,4)+PT(JT,3)-DPE2-DPZ2
		    ETMPRM=PT(JT,4)-PT(JT,3)-DPE2+DPZ2
		    IF(ETPPRM.LE.0.0.OR.ETMPRM.LE.0.0) GO TO 16
		    ETP=ETPPRM
		    ETM=ETMPRM
		    PT(JT,10)=PKC21
		    PT(JT,11)=PKC22
		    NTJ(JT)=NTJ(JT)+1
		    KFTJ(JT,NTJ(JT))=21
		    PJTX(JT,NTJ(JT))=DPX2
		    PJTY(JT,NTJ(JT))=DPY2
		    PJTZ(JT,NTJ(JT))=DPZ2
		    PJTE(JT,NTJ(JT))=DPE2
		    PJTM(JT,NTJ(JT))=0.0
		    PT(JT,3)=PT(JT,3)-DPZ2
		    PT(JT,4)=PT(JT,4)-DPE2
		 ENDIF
	      ENDIF
 16	      DPKC11=-(PP(JP,10)-PKC11)/2.0
	      DPKC12=-(PP(JP,11)-PKC12)/2.0
	      DPKC21=-(PT(JT,10)-PKC21)/2.0
	      DPKC22=-(PT(JT,11)-PKC22)/2.0
	      WP=EPP+ETP
	      WM=EPM+ETM
	      SW=WP*WM
	   ENDIF
	ENDIF
C		********If jet is quenched the pt from valence quark
C			hard scattering has to reduced by d*kapa
C
C   
10	PTP02=PP(JP,1)**2+PP(JP,2)**2
	PTT02=PT(JT,1)**2+PT(JT,2)**2
C	
	AMQ=MAX(PP(JP,14)+PP(JP,15),PT(JT,14)+PT(JT,15))
	AMX=HIPR1(1)+AMQ
C		********consider mass cut-off for strings which
C			must also include quark's mass
	AMP0=AMX
	DPM0=AMX
	NFDP=0
	IF(NFP(JP,5).LE.2.AND.NFP(JP,3).NE.0) THEN
		AMP0=ULMASS(NFP(JP,3))
		NFDP=NFP(JP,3)+2*NFP(JP,3)/ABS(NFP(JP,3))
		DPM0=ULMASS(NFDP)
		IF(DPM0.LE.0.0) THEN
			NFDP=NFDP-2*NFDP/ABS(NFDP)
			DPM0=ULMASS(NFDP)
		ENDIF
	ENDIF
	AMT0=AMX
	DTM0=AMX
	NFDT=0
	IF(NFT(JT,5).LE.2.AND.NFT(JT,3).NE.0) THEN
		AMT0=ULMASS(NFT(JT,3))
		NFDT=NFT(JT,3)+2*NFT(JT,3)/ABS(NFT(JT,3))
		DTM0=ULMASS(NFDT)
		IF(DTM0.LE.0.0) THEN
			NFDT=NFDT-2*NFDT/ABS(NFDT)
			DTM0=ULMASS(NFDT)
		ENDIF
	ENDIF
C	
	AMPN=SQRT(AMP0**2+PTP02)
	AMTN=SQRT(AMT0**2+PTT02)
	SNN=(AMPN+AMTN)**2+0.001
C
	IF(SW.LT.SNN+0.001) GO TO 4000
C		********Scatter only if SW>SNN
C*****give some PT kick to the two exited strings******************
	SWPTN=4.0*(MAX(AMP0,AMT0)**2+MAX(PTP02,PTT02))
	SWPTD=4.0*(MAX(DPM0,DTM0)**2+MAX(PTP02,PTT02))
	SWPTX=4.0*(AMX**2+MAX(PTP02,PTT02))
	IF(SW.LE.SWPTN) THEN
		PKCMX=0.0
	ELSE IF(SW.GT.SWPTN .AND. SW.LE.SWPTD
     &		.AND.NPJ(JP).EQ.0.AND.NTJ(JT).EQ.0) THEN
	   PKCMX=SQRT(SW/4.0-MAX(AMP0,AMT0)**2)
     &           -SQRT(MAX(PTP02,PTT02))
	ELSE IF(SW.GT.SWPTD .AND. SW.LE.SWPTX
     &		.AND.NPJ(JP).EQ.0.AND.NTJ(JT).EQ.0) THEN
	   PKCMX=SQRT(SW/4.0-MAX(DPM0,DTM0)**2)
     &           -SQRT(MAX(PTP02,PTT02))
	ELSE IF(SW.GT.SWPTX) THEN
	   PKCMX=SQRT(SW/4.0-AMX**2)-SQRT(MAX(PTP02,PTT02))
	ENDIF
C		********maximun PT kick
C*********************************************************
C
	IF(NFP(JP,10).EQ.1.OR.NFT(JT,10).EQ.1) THEN
		IF(PKC1.GT.PKCMX) THEN
			PKC1=PKCMX
			PKC11=PKC1*COS(PHI1)
			PKC12=PKC1*SIN(PHI1)
			DPKC11=-(PP(JP,10)-PKC11)/2.0
			DPKC12=-(PP(JP,11)-PKC12)/2.0
		ENDIF
		IF(PKC2.GT.PKCMX) THEN
			PKC2=PKCMX
			PKC21=PKC2*COS(PHI2)
			PKC22=PKC2*SIN(PHI2)
			DPKC21=-(PT(JT,10)-PKC21)/2.0
			DPKC22=-(PT(JT,11)-PKC22)/2.0
		ENDIF
		DPKC1=DPKC11+DPKC21
		DPKC2=DPKC12+DPKC22
		NFP(JP,10)=-NFP(JP,10)
		NFT(JT,10)=-NFT(JT,10)
		GO TO 40
	ENDIF
C		********If the valence quarks had a hard-collision
C			the pt kick is the pt from hard-collision.
	I_SNG=0
	IF(IHPR2(13).NE.0 .AND. RLU(NSEED).LE.HIDAT(4)) I_SNG=1
	IF((NFP(JP,5).EQ.3 .OR.NFT(JT,5).EQ.3).OR.
     &		(NPJ(JP).NE.0.OR.NFP(JP,10).NE.0).OR.
     &		(NTJ(JT).NE.0.OR.NFT(JT,10).NE.0)) I_SNG=0
C
C               ********decite whether to have single-diffractive
	IF(IHPR2(5).EQ.0) THEN
		PKC=HIPR1(2)*SQRT(-ALOG(1.0-RLU(NSEED)
     &			*(1.0-EXP(-PKCMX**2/HIPR1(2)**2))))
		GO TO 30
	ENDIF
	PKC=HIRND2(3,0.0,PKCMX**2)
	PKC=SQRT(PKC)
	IF(PKC.GT.HIPR1(20)) 
     &	   PKC=HIPR1(2)*SQRT(-ALOG(EXP(-HIPR1(20)**2/HIPR1(2)**2)
     &	       -RLU(NSEED)*(EXP(-HIPR1(20)**2/HIPR1(2)**2)-
     &	       EXP(-PKCMX**2/HIPR1(2)**2))))
C
	IF(I_SNG.EQ.1) PKC=0.65*SQRT(
     &		-ALOG(1.0-RLU(NSEED)*(1.0-EXP(-PKCMX**2/0.65**2))))
C			********select PT kick
30	PHI0=2.0*HIPR1(40)*RLU(NSEED)
	PKC11=PKC*SIN(PHI0)
	PKC12=PKC*COS(PHI0)
	PKC21=-PKC11
	PKC22=-PKC12
	DPKC1=0.0
	DPKC2=0.0
40	PP11=PP(JP,1)+PKC11-DPKC1
	PP12=PP(JP,2)+PKC12-DPKC2
	PT11=PT(JT,1)+PKC21-DPKC1
	PT12=PT(JT,2)+PKC22-DPKC2
	PTP2=PP11**2+PP12**2
	PTT2=PT11**2+PT12**2
C
	AMPN=SQRT(AMP0**2+PTP2)
	AMTN=SQRT(AMT0**2+PTT2)
	SNN=(AMPN+AMTN)**2+0.001
C***************************************
	WP=EPP+ETP
	WM=EPM+ETM
	SW=WP*WM
C****************************************
	IF(SW.LT.SNN) THEN
	   MISS=MISS+1
	   IF(MISS.LE.100) then
	      PKC=0.0
	      GO TO 30
	   ENDIF
	   IF(IHPR2(10).NE.0) 
     &	     WRITE(6,*) 'Error occured in Pt kick section of HIJSFT'
	   GO TO 4000
	ENDIF
C******************************************************************
	AMPD=SQRT(DPM0**2+PTP2)
	AMTD=SQRT(DTM0**2+PTT2)

	AMPX=SQRT(AMX**2+PTP2)
	AMTX=SQRT(AMX**2+PTT2)

	DPN=AMPN**2/SW
	DTN=AMTN**2/SW
	DPD=AMPD**2/SW
	DTD=AMTD**2/SW
	DPX=AMPX**2/SW
	DTX=AMTX**2/SW
C
	SPNTD=(AMPN+AMTD)**2
	SPNTX=(AMPN+AMTX)**2
C			********CM energy if proj=N,targ=N*
	SPDTN=(AMPD+AMTN)**2
	SPXTN=(AMPX+AMTN)**2
C			********CM energy if proj=N*,targ=N
	SPDTX=(AMPD+AMTX)**2
	SPXTD=(AMPX+AMTD)**2
	SDD=(AMPD+AMTD)**2
	SXX=(AMPX+AMTX)**2

C
C	
C		********CM energy if proj=delta, targ=delta
C****************There are many different cases**********
c	IF(IHPR2(15).EQ.1) GO TO 500
C
C		********to have DPM type soft interactions
C
	CONTINUE
	IF(SW.GT.SXX+0.001) THEN
	   IF(I_SNG.EQ.0) THEN
 	      D1=DPX
	      D2=DTX
	      NFP3=0
	      NFT3=0
	      GO TO 400
	   ELSE
c**** 5/30/1998 this is identical to the above statement. Added to
c**** avoid questional branching to block.
	      IF((NFP(JP,5).EQ.3 .AND.NFT(JT,5).EQ.3).OR.
     &	         (NPJ(JP).NE.0.OR.NFP(JP,10).NE.0).OR.
     &		 (NTJ(JT).NE.0.OR.NFT(JT,10).NE.0)) THEN
                 D1=DPX
                 D2=DTX
                 NFP3=0
                 NFT3=0
                 GO TO 400
              ENDIF
C		********do not allow excited strings to have 
C			single-diffr 
	      IF(RLU(NSEED).GT.0.5.OR.(NFT(JT,5).GT.2.OR.
     &		      NTJ(JT).NE.0.OR.NFT(JT,10).NE.0)) THEN
		 D1=DPN
		 D2=DTX
		 NFP3=NFP(JP,3)
		 NFT3=0
		 GO TO 220
	      ELSE
		 D1=DPX
		 D2=DTN
		 NFP3=0
		 NFT3=NFT(JT,3)
		 GO TO 240
	      ENDIF
C		********have single diffractive collision
	   ENDIF
	ELSE IF(SW.GT.MAX(SPDTX,SPXTD)+0.001 .AND.
     &				SW.LE.SXX+0.001) THEN
	   IF(((NPJ(JP).EQ.0.AND.NTJ(JT).EQ.0.AND.
     &         RLU(NSEED).GT.0.5).OR.(NPJ(JP).EQ.0
     &         .AND.NTJ(JT).NE.0)).AND.NFP(JP,5).LE.2) THEN
	      D1=DPD
	      D2=DTX
	      NFP3=NFDP
	      NFT3=0
	      GO TO 220
	   ELSE IF(NTJ(JT).EQ.0.AND.NFT(JT,5).LE.2) THEN
	      D1=DPX
	      D2=DTD
	      NFP3=0
	      NFT3=NFDT
	      GO TO 240
	   ENDIF
	   GO TO 4000
	ELSE IF(SW.GT.MIN(SPDTX,SPXTD)+0.001.AND.
     &			SW.LE.MAX(SPDTX,SPXTD)+0.001) THEN
	   IF(SPDTX.LE.SPXTD.AND.NPJ(JP).EQ.0
     &                       .AND.NFP(JP,5).LE.2) THEN
	      D1=DPD
	      D2=DTX
	      NFP3=NFDP
	      NFT3=0
	      GO TO 220
	   ELSE IF(SPDTX.GT.SPXTD.AND.NTJ(JT).EQ.0
     &                       .AND.NFT(JT,5).LE.2) THEN
	      D1=DPX
	      D2=DTD
	      NFP3=0
	      NFT3=NFDT
	      GO TO 240
	   ENDIF
c*** 5/30/1998 added to avoid questional branching to another block
c*** this is identical to the statement following the next ELSE IF
	   IF(((NPJ(JP).EQ.0.AND.NTJ(JT).EQ.0
     &       .AND.RLU(NSEED).GT.0.5).OR.(NPJ(JP).EQ.0
     &        .AND.NTJ(JT).NE.0)).AND.NFP(JP,5).LE.2) THEN
	      D1=DPN
	      D2=DTX
	      NFP3=NFP(JP,3)
	      NFT3=0
	      GO TO 220
	   ELSE IF(NTJ(JT).EQ.0.AND.NFT(JT,5).LE.2) THEN
	      D1=DPX
	      D2=DTN
	      NFP3=0
	      NFT3=NFT(JT,3)
	      GO TO 240
	   ENDIF
	   GO TO 4000
	ELSE IF(SW.GT.MAX(SPNTX,SPXTN)+0.001 .AND.
     &			SW.LE.MIN(SPDTX,SPXTD)+0.001) THEN
	   IF(((NPJ(JP).EQ.0.AND.NTJ(JT).EQ.0
     &       .AND.RLU(NSEED).GT.0.5).OR.(NPJ(JP).EQ.0
     &        .AND.NTJ(JT).NE.0)).AND.NFP(JP,5).LE.2) THEN
	      D1=DPN
	      D2=DTX
	      NFP3=NFP(JP,3)
	      NFT3=0
	      GO TO 220
	   ELSE IF(NTJ(JT).EQ.0.AND.NFT(JT,5).LE.2) THEN
	      D1=DPX
	      D2=DTN
	      NFP3=0
	      NFT3=NFT(JT,3)
	      GO TO 240
	   ENDIF
	   GO TO 4000
	ELSE IF(SW.GT.MIN(SPNTX,SPXTN)+0.001 .AND.
     &			SW.LE.MAX(SPNTX,SPXTN)+0.001) THEN
	   IF(SPNTX.LE.SPXTN.AND.NPJ(JP).EQ.0
     &                           .AND.NFP(JP,5).LE.2) THEN
	      D1=DPN
	      D2=DTX
	      NFP3=NFP(JP,3)
	      NFT3=0
	      GO TO 220
	   ELSEIF(SPNTX.GT.SPXTN.AND.NTJ(JT).EQ.0
     &                           .AND.NFT(JT,5).LE.2) THEN
	      D1=DPX
	      D2=DTN
	      NFP3=0
	      NFT3=NFT(JT,3)
	      GO TO 240
	   ENDIF
	   GO TO 4000
	ELSE IF(SW.LE.MIN(SPNTX,SPXTN)+0.001 .AND.
     &			(NPJ(JP).NE.0 .OR.NTJ(JT).NE.0)) THEN
	   GO TO 4000
	ELSE IF(SW.LE.MIN(SPNTX,SPXTN)+0.001 .AND.
     &		NFP(JP,5).GT.2.AND.NFT(JT,5).GT.2) THEN
	   GO TO 4000
	ELSE IF(SW.GT.SDD+0.001.AND.SW.LE.
     &                     MIN(SPNTX,SPXTN)+0.001) THEN
	   D1=DPD
	   D2=DTD
	   NFP3=NFDP
	   NFT3=NFDT
	   GO TO 100
	ELSE IF(SW.GT.MAX(SPNTD,SPDTN)+0.001 
     &                      .AND. SW.LE.SDD+0.001) THEN
	   IF(RLU(NSEED).GT.0.5) THEN
	      D1=DPD
	      D2=DTN
	      NFP3=NFDP
	      NFT3=NFT(JT,3)
	      GO TO 100
	   ELSE
	      D1=DPN
	      D2=DTD
	      NFP3=NFP(JP,3)
	      NFT3=NFDT
	      GO TO 100
	   ENDIF
	ELSE IF(SW.GT.MIN(SPNTD,SPDTN)+0.001
     &		.AND. SW.LE.MAX(SPNTD,SPDTN)+0.001) THEN
	   IF(SPNTD.GT.SPDTN) THEN
	      D1=DPD
	      D2=DTN
	      NFP3=NFDP
	      NFT3=NFT(JT,3)
	      GO TO 100
	   ELSE
	      D1=DPN
	      D2=DTD
	      NFP3=NFP(JP,3)
	      NFT3=NFDT
	      GO TO 100
	   ENDIF
	ELSE IF(SW.LE.MIN(SPNTD,SPDTN)+0.001) THEN
	   D1=DPN
	   D2=DTN
	   NFP3=NFP(JP,3)
	   NFT3=NFT(JT,3)
	   GO TO 100
	ENDIF
	WRITE(6,*) ' Error in HIJSFT: There is no path to here'
	RETURN
C
C***************  elastic scattering ***************
C	this is like elastic, both proj and targ mass
C	must be fixed
C***************************************************
100	NFP5=MAX(2,NFP(JP,5))
	NFT5=MAX(2,NFT(JT,5))
	BB1=1.0+D1-D2
	BB2=1.0+D2-D1
	IF(BB1**2.LT.4.0*D1 .OR. BB2**2.LT.4.0*D2) THEN
		MISS=MISS+1
		IF(MISS.GT.100.OR.PKC.EQ.0.0) GO TO 3000
		PKC=PKC*0.5
		GO TO 30
	ENDIF
	IF(RLU(NSEED).LT.0.5) THEN
		X1=(BB1-SQRT(BB1**2-4.0*D1))/2.0
		X2=(BB2-SQRT(BB2**2-4.0*D2))/2.0
	ELSE
		X1=(BB1+SQRT(BB1**2-4.0*D1))/2.0
		X2=(BB2+SQRT(BB2**2-4.0*D2))/2.0
	ENDIF
	IHNT2(13)=2
	GO TO 600
C
C********** Single diffractive ***********************
C either proj or targ's mass is fixed
C*****************************************************
220	NFP5=MAX(2,NFP(JP,5))
	NFT5=3
	IF(NFP3.EQ.0) NFP5=3
	BB2=1.0+D2-D1
	IF(BB2**2.LT.4.0*D2) THEN
		MISS=MISS+1
		IF(MISS.GT.100.OR.PKC.EQ.0.0) GO TO 3000
		PKC=PKC*0.5
		GO TO 30
	ENDIF
	XMIN=(BB2-SQRT(BB2**2-4.0*D2))/2.0
	XMAX=(BB2+SQRT(BB2**2-4.0*D2))/2.0
	MISS4=0
222	X2=HIRND2(6,XMIN,XMAX)
	X1=D1/(1.0-X2)
	IF(X2*(1.0-X1).LT.(D2+1.E-4/SW)) THEN
		MISS4=MISS4+1
		IF(MISS4.LE.1000) GO TO 222
		GO TO 5000
	ENDIF
	IHNT2(13)=2
	GO TO 600
C			********Fix proj mass*********
240	NFP5=3
	NFT5=MAX(2,NFT(JT,5))
	IF(NFT3.EQ.0) NFT5=3
	BB1=1.0+D1-D2
	IF(BB1**2.LT.4.0*D1) THEN
		MISS=MISS+1
		IF(MISS.GT.100.OR.PKC.EQ.0.0) GO TO 3000
		PKC=PKC*0.5
		GO TO 30
	ENDIF
	XMIN=(BB1-SQRT(BB1**2-4.0*D1))/2.0
	XMAX=(BB1+SQRT(BB1**2-4.0*D1))/2.0
	MISS4=0
242	X1=HIRND2(6,XMIN,XMAX)
	X2=D2/(1.0-X1)
	IF(X1*(1.0-X2).LT.(D1+1.E-4/SW)) THEN
		MISS4=MISS4+1
		IF(MISS4.LE.1000) GO TO 242
		GO TO 5000
	ENDIF
	IHNT2(13)=2
	GO TO 600
C			********Fix targ mass*********
C
C*************non-single diffractive**********************
C	both proj and targ may not be fixed in mass 
C*********************************************************
C
400	NFP5=3
	NFT5=3
	BB1=1.0+D1-D2
	BB2=1.0+D2-D1
	IF(BB1**2.LT.4.0*D1 .OR. BB2**2.LT.4.0*D2) THEN
		MISS=MISS+1
		IF(MISS.GT.100.OR.PKC.EQ.0.0) GO TO 3000
		PKC=PKC*0.5
		GO TO 30
	ENDIF
	XMIN1=(BB1-SQRT(BB1**2-4.0*D1))/2.0
	XMAX1=(BB1+SQRT(BB1**2-4.0*D1))/2.0
	XMIN2=(BB2-SQRT(BB2**2-4.0*D2))/2.0
	XMAX2=(BB2+SQRT(BB2**2-4.0*D2))/2.0
	MISS4=0	
410	X1=HIRND2(4,XMIN1,XMAX1)
	X2=HIRND2(4,XMIN2,XMAX2)
	IF(NFP(JP,5).EQ.3.OR.NFT(JT,5).EQ.3) THEN
		X1=HIRND2(6,XMIN1,XMAX1)
		X2=HIRND2(6,XMIN2,XMAX2)
	ENDIF
C			********
	IF(ABS(NFP(JP,1)*NFP(JP,2)).GT.1000000.OR.
     &			ABS(NFP(JP,1)*NFP(JP,2)).LT.100) THEN
		X1=HIRND2(5,XMIN1,XMAX1)
	ENDIF
	IF(ABS(NFT(JT,1)*NFT(JT,2)).GT.1000000.OR.
     &			ABS(NFT(JT,1)*NFT(JT,2)).LT.100) THEN
		X2=HIRND2(5,XMIN2,XMAX2)
	ENDIF
c	IF(IOPMAIN.EQ.3) X1=HIRND2(6,XMIN1,XMAX1)
c	IF(IOPMAIN.EQ.2) X2=HIRND2(6,XMIN2,XMAX2) 
C	********For q-qbar or (qq)-(qq)bar system use symetric
C		distribution, for q-(qq) or qbar-(qq)bar use
C		unsymetrical distribution
C
	IF(ABS(NFP(JP,1)*NFP(JP,2)).GT.1000000) X1=1.0-X1
	XXP=X1*(1.0-X2)
	XXT=X2*(1.0-X1)
	IF(XXP.LT.(D1+1.E-4/SW) .OR. XXT.LT.(D2+1.E-4/SW)) THEN
		MISS4=MISS4+1
		IF(MISS4.LE.1000) GO TO 410
		GO TO 5000
	ENDIF
	IHNT2(13)=3
C***************************************************
C***************************************************
600	CONTINUE
	IF(X1*(1.0-X2).LT.(AMPN**2-1.E-4)/SW.OR.
     &			X2*(1.0-X1).LT.(AMTN**2-1.E-4)/SW) THEN
		MISS=MISS+1
		IF(MISS.GT.100.OR.PKC.EQ.0.0) GO TO 2000
		PKC=0.0
		GO TO 30
	ENDIF
C
	EPP=(1.0-X2)*WP
	EPM=X1*WM
	ETP=X2*WP
	ETM=(1.0-X1)*WM
	PP(JP,3)=(EPP-EPM)/2.0
	PP(JP,4)=(EPP+EPM)/2.0
	IF(EPP*EPM-PTP2.LT.0.0) GO TO 6000
	PP(JP,5)=SQRT(EPP*EPM-PTP2)
	NFP(JP,3)=NFP3
	NFP(JP,5)=NFP5

	PT(JT,3)=(ETP-ETM)/2.0
	PT(JT,4)=(ETP+ETM)/2.0
	IF(ETP*ETM-PTT2.LT.0.0) GO TO 6000
	PT(JT,5)=SQRT(ETP*ETM-PTT2)
	NFT(JT,3)=NFT3
	NFT(JT,5)=NFT5
C*****recoil PT from hard-inter is shared by two end-partons 
C       so that pt=p1+p2
	PP(JP,1)=PP11-PKC11
	PP(JP,2)=PP12-PKC12

	KICKDIP=1
	KICKDIT=1
	IF(ABS(NFP(JP,1)*NFP(JP,2)).GT.1000000.OR.
     &			ABS(NFP(JP,1)*NFP(JP,2)).LT.100) THEN
		KICKDIP=0
	ENDIF
	IF(ABS(NFT(JT,1)*NFT(JT,2)).GT.1000000.OR.
     &			ABS(NFT(JT,1)*NFT(JT,2)).LT.100) THEN
		KICKDIT=0
	ENDIF
	IF((KICKDIP.EQ.0.AND.RLU(NSEED).LT.0.5)
     &     .OR.(KICKDIP.NE.0.AND.RLU(NSEED)
     &     .LT.0.5/(1.0+(PKC11**2+PKC12**2)/HIPR1(22)**2))) THEN
	   PP(JP,6)=(PP(JP,1)-PP(JP,6)-PP(JP,8)-DPKC1)/2.0+PP(JP,6)
	   PP(JP,7)=(PP(JP,2)-PP(JP,7)-PP(JP,9)-DPKC2)/2.0+PP(JP,7)
	   PP(JP,8)=(PP(JP,1)-PP(JP,6)-PP(JP,8)-DPKC1)/2.0
     &              +PP(JP,8)+PKC11
	   PP(JP,9)=(PP(JP,2)-PP(JP,7)-PP(JP,9)-DPKC2)/2.0
     &              +PP(JP,9)+PKC12
	ELSE
	   PP(JP,8)=(PP(JP,1)-PP(JP,6)-PP(JP,8)-DPKC1)/2.0+PP(JP,8)
	   PP(JP,9)=(PP(JP,2)-PP(JP,7)-PP(JP,9)-DPKC2)/2.0+PP(JP,9)
	   PP(JP,6)=(PP(JP,1)-PP(JP,6)-PP(JP,8)-DPKC1)/2.0
     &              +PP(JP,6)+PKC11
	   PP(JP,7)=(PP(JP,2)-PP(JP,7)-PP(JP,9)-DPKC2)/2.0
     &              +PP(JP,7)+PKC12
	ENDIF
	PP(JP,1)=PP(JP,6)+PP(JP,8)
	PP(JP,2)=PP(JP,7)+PP(JP,9)
C				********pt kick for proj
	PT(JT,1)=PT11-PKC21
	PT(JT,2)=PT12-PKC22
	IF((KICKDIT.EQ.0.AND.RLU(NSEED).LT.0.5)
     &     .OR.(KICKDIT.NE.0.AND.RLU(NSEED)
     &     .LT.0.5/(1.0+(PKC21**2+PKC22**2)/HIPR1(22)**2))) THEN
	   PT(JT,6)=(PT(JT,1)-PT(JT,6)-PT(JT,8)-DPKC1)/2.0+PT(JT,6)
	   PT(JT,7)=(PT(JT,2)-PT(JT,7)-PT(JT,9)-DPKC2)/2.0+PT(JT,7)
	   PT(JT,8)=(PT(JT,1)-PT(JT,6)-PT(JT,8)-DPKC1)/2.0
     &              +PT(JT,8)+PKC21
	   PT(JT,9)=(PT(JT,2)-PT(JT,7)-PT(JT,9)-DPKC2)/2.0
     &              +PT(JT,9)+PKC22
	ELSE
	   PT(JT,8)=(PT(JT,1)-PT(JT,6)-PT(JT,8)-DPKC1)/2.0+PT(JT,8)
	   PT(JT,9)=(PT(JT,2)-PT(JT,7)-PT(JT,9)-DPKC2)/2.0+PT(JT,9)
	   PT(JT,6)=(PT(JT,1)-PT(JT,6)-PT(JT,8)-DPKC1)/2.0
     &              +PT(JT,6)+PKC21
	   PT(JT,7)=(PT(JT,2)-PT(JT,7)-PT(JT,9)-DPKC2)/2.0
     &              +PT(JT,7)+PKC22
	ENDIF
	PT(JT,1)=PT(JT,6)+PT(JT,8)
	PT(JT,2)=PT(JT,7)+PT(JT,9)
C			********pt kick for targ

	IF(NPJ(JP).NE.0) NFP(JP,5)=3
	IF(NTJ(JT).NE.0) NFT(JT,5)=3
C			********jets must be connected to string
	IF(EPP/(EPM+0.0001).LT.ETP/(ETM+0.0001).AND.
     &			ABS(NFP(JP,1)*NFP(JP,2)).LT.1000000)THEN
		DO 620 JSB=1,15
		PSB=PP(JP,JSB)
		PP(JP,JSB)=PT(JT,JSB)
		PT(JT,JSB)=PSB
		NSB=NFP(JP,JSB)
		NFP(JP,JSB)=NFT(JT,JSB)
		NFT(JT,JSB)=NSB
620		CONTINUE
C		********when Ycm(JP)<Ycm(JT) after the collision
C			exchange the positions of the two   
	ENDIF
C
	RETURN
C**************************************************
C**************************************************
1000	IERROR=1
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) '	Fatal HIJSFT start error,abandon this event'
	WRITE(6,*) '	PROJ E+,E-,W+',EPP,EPM,WP
	WRITE(6,*) '	TARG E+,E-,W-',ETP,ETM,WM
	WRITE(6,*) '	W+*W-, (APN+ATN)^2',SW,SNN
	RETURN
2000	IERROR=0
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) '	(2)energy partition fail,'
	WRITE(6,*) ' 	HIJSFT not performed, but continue'
	WRITE(6,*) '	MP1,MPN',X1*(1.0-X2)*SW,AMPN**2
	WRITE(6,*) '	MT2,MTN',X2*(1.0-X1)*SW,AMTN**2
	RETURN
3000	IERROR=0
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) '	(3)something is wrong with the pt kick, '
	WRITE(6,*) '	HIJSFT not performed, but continue'
	WRITE(6,*) '	D1=',D1,' D2=',D2,' SW=',SW
	WRITE(6,*) '	HISTORY NFP5=',NFP(JP,5),' NFT5=',NFT(JT,5)
	WRITE(6,*) '	THIS COLLISON NFP5=',NFP5, ' NFT5=',NFT5
	WRITE(6,*) '	# OF JET IN PROJ',NPJ(JP),' IN TARG',NTJ(JT)
	RETURN
4000	IERROR=0
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) '	(4)unable to choose process, but not harmful'
	WRITE(6,*) '	HIJSFT not performed, but continue'
	WRITE(6,*) '	PTP=',SQRT(PTP2),' PTT=',SQRT(PTT2),' SW=',SW
	WRITE(6,*) '	AMCUT=',AMX,' JP=',JP,' JT=',JT
	WRITE(6,*) '	HISTORY NFP5=',NFP(JP,5),' NFT5=',NFT(JT,5)
	RETURN
5000	IERROR=0
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) '	energy partition failed(5),for limited try'
	WRITE(6,*) '    HIJSFT not performed, but continue'
	WRITE(6,*) '	NFP5=',NFP5,' NFT5=',NFT5
	WRITE(6,*) '	D1',D1,' X1(1-X2)',X1*(1.0-X2)
	WRITE(6,*) '	D2',D2,' X2(1-X1)',X2*(1.0-X1)
	RETURN
6000	PKC=0.0
	MISS=MISS+1
	IF(MISS.LT.100) GO TO 30
	IERROR=1
	IF(IHPR2(10).EQ.0) RETURN
	WRITE(6,*) ' ERROR OCCURED, HIJSFT NOT PERFORMED'
        WRITE(6,*) ' Abort this event'
	WRITE(6,*) 'MTP,PTP2',EPP*EPM,PTP2,'  MTT,PTT2',ETP*ETM,PTT2 
	RETURN
	END
C
C	
C***************************************
	SUBROUTINE HIJFLV(ID)
	COMMON/RANSEED/NSEED
	SAVE  
	ID=1
	RNID=RLU(NSEED)
	IF(RNID.GT.0.43478) THEN
		ID=2
		IF(RNID.GT.0.86956) ID=3
	ENDIF
	RETURN
	END
C
C
C
	SUBROUTINE HIPTDI(PT,PTMAX,IOPT)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/RANSEED/NSEED
	SAVE  
	IF(IOPT.EQ.2) THEN
		PT=HIRND2(7,0.0,PTMAX)
		IF(PT.GT.HIPR1(8)) 
     &		PT=HIPR1(2)*SQRT(-ALOG(EXP(-HIPR1(8)**2/HIPR1(2)**2)
     &			-RLU(NSEED)*(EXP(-HIPR1(8)**2/HIPR1(2)**2)-
     &			EXP(-PTMAX**2/HIPR1(2)**2))))

	ELSE
		PT=HIPR1(2)*SQRT(-ALOG(1.0-RLU(NSEED)*
     &			(1.0-EXP(-PTMAX**2/HIPR1(2)**2))))
	ENDIF
	PTMAX0=MAX(PTMAX,0.01)
	PT=MIN(PTMAX0-0.01,PT)
	RETURN
	END
C*************************
C
C
C
C
C ********************************************************
C ************************              WOOD-SAX
        SUBROUTINE HIJWDS(IA,IDH,XHIGH)
C     SETS UP HISTOGRAM IDH WITH RADII FOR
C     NUCLEUS IA DISTRIBUTED ACCORDING TO THREE PARAM WOOD SAXON
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/WOOD/R,D,FNORM,W        
        DIMENSION IAA(20),RR(20),DD(20),WW(20),RMS(20)
        EXTERNAL RWDSAX,WDSAX
        SAVE
C
C   PARAMETERS OF SPECIAL NUCLEI FROM ATOMIC DATA AND NUC DATA TABLES
C     VOL 14, 5-6 1974
        DATA IAA/2,4,12,16,27,32,40,56,63,93,184,197,208,7*0./
        DATA RR/0.01,.964,2.355,2.608,2.84,3.458,3.766,3.971,4.214,
     1        4.87,6.51,6.38,6.624,7*0./
        DATA DD/0.5882,.322,.522,.513,.569,.61,.586,.5935,.586,.573,
     1        .535,.535,.549,7*0./
        DATA WW/0.0,.517,-0.149,-0.051,0.,-0.208,-0.161,13*0./
        DATA RMS/2.11,1.71,2.46,2.73,3.05,3.247,3.482,3.737,3.925,4.31,
     1        5.42,5.33,5.521,7*0./
C
      	A=IA
C
C 		********SET WOOD-SAX PARAMS FIRST  AS IN DATE ET AL
      	D=0.54
C			********D IS WOOD SAX DIFFUSE PARAM IN FM
	R=1.19*A**(1./3.) - 1.61*A**(-1./3.)
C 			********R IS RADIUS PARAM
	W=0.
C 		********W IS The third of three WOOD-SAX PARAM
C
C      		********CHECK TABLE FOR SPECIAL CASES
	DO 10 I=1,13
		IF (IA.EQ.IAA(I)) THEN
			R=RR(I)
     			D=DD(I)
      			W=WW(I)
      			RS=RMS(I)
      		END IF
10    	CONTINUE
C     			********FNORM is the normalize factor
      	FNORM=1.0
      	XLOW=0.
      	XHIGH=R+ 12.*D
      	IF (W.LT.-0.01)  THEN
      		IF (XHIGH.GT.R/SQRT(ABS(W))) XHIGH=R/SQRT(ABS(W))
      	END IF
      	FGAUS=GAUSS1(RWDSAX,XLOW,XHIGH,0.001)
      	FNORM=1./FGAUS
C
        IF (IDH.EQ.1) THEN
           HINT1(72)=R
           HINT1(73)=D
           HINT1(74)=W
           HINT1(75)=FNORM/4.0/HIPR1(40)
        ELSE IF (IDH.EQ.2) THEN
           HINT1(76)=R
           HINT1(77)=D
           HINT1(78)=W
           HINT1(79)=FNORM/4.0/HIPR1(40)
        ENDIF
C
C     	NOW SET UP HBOOK FUNCTIONS IDH FOR  R**2*RHO(R)
C     	THESE HISTOGRAMS ARE USED TO GENERATE RANDOM RADII
      	CALL HIFUN(IDH,XLOW,XHIGH,RWDSAX)
      	RETURN
      	END
C
C
	FUNCTION WDSAX(X)
C     			********THREE PARAMETER WOOD SAXON
      	COMMON/WOOD/R,D,FNORM,W
	SAVE
      	WDSAX=FNORM*(1.+W*(X/R)**2)/(1+EXP((X-R)/D))
       	IF (W.LT.0.) THEN
       		IF (X.GE.R/SQRT(ABS(W))) WDSAX=0.
       	ENDIF
      	RETURN
      	END
C
C
	FUNCTION RWDSAX(X)
      	RWDSAX=X*X*WDSAX(X)
      	RETURN
      	END
C
C
C
C
	FUNCTION WDSAX1(X)
C     			********THREE PARAMETER WOOD SAXON 
C                               FOR  PROJECTILE
C       HINT1(72)=R, HINT1(73)=D, HINT1(74)=W, HINT1(75)=FNORM
C
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE
      	WDSAX1=HINT1(75)*(1.+HINT1(74)*(X/HINT1(72))**2)/
     &       (1+EXP((X-HINT1(72))/HINT1(73)))
       	IF (HINT1(74).LT.0.) THEN
       		IF (X.GE.HINT1(72)/SQRT(ABS(HINT1(74)))) WDSAX1=0.
       	ENDIF
      	RETURN
      	END
C
C
	FUNCTION WDSAX2(X)
C     			********THREE PARAMETER WOOD SAXON 
C                               FOR  TARGET
C       HINT1(76)=R,HINT1(77)=D, HINT1(78)=W, HINT1(79)=FNORM
C
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
      	WDSAX2=HINT1(79)*(1.+HINT1(78)*(X/HINT1(76))**2)/
     &       (1+EXP((X-HINT1(76))/HINT1(77)))
       	IF (HINT1(78).LT.0.) THEN
       		IF (X.GE.HINT1(76)/SQRT(ABS(HINT1(78)))) WDSAX2=0.
       	ENDIF
      	RETURN
      	END
C
C
C	THIS FUNCTION IS TO CALCULATE THE NUCLEAR PROFILE FUNCTION
C       OF THE  COLLIDERING SYSTEM (IN UNITS OF 1/mb)
C
	FUNCTION  PROFILE(XB)
        COMMON/PACT/BB,B1,PHI,Z1
        COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	EXTERNAL FLAP, FLAP1, FLAP2
        SAVE
C
        BB=XB
        PROFILE=1.0
        IF(IHNT2(1).GT.1 .AND. IHNT2(3).GT.1) THEN
           PROFILE=float(IHNT2(1))*float(IHNT2(3))*0.1*
     &          GAUSS1(FLAP,0.0,HIPR1(34),0.01)
        ELSE IF(IHNT2(1).EQ.1 .AND. IHNT2(3).GT.1) THEN
           PROFILE=0.2*float(IHNT2(3))*
     &          GAUSS1(FLAP2,0.0,HIPR1(35),0.001)
        ELSE IF(IHNT2(1).GT.1 .AND. IHNT2(3).EQ.1) THEN
           PROFILE=0.2*float(IHNT2(1))*
     &          GAUSS1(FLAP1,0.0,HIPR1(34),0.001)
        ENDIF
	RETURN
	END
C
C
	FUNCTION FLAP(X)
        COMMON/PACT/BB,B1,PHI,Z1
        COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        EXTERNAL FGP1
        SAVE
        B1=X
        FLAP=GAUSS2(FGP1,0.0,2.0*HIPR1(40),0.01)
	RETURN
	END
C
	FUNCTION FGP1(X)
        COMMON/PACT/BB,B1,PHI,Z1
        COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        EXTERNAL FGP2
        SAVE
        PHI=X
        FGP1=2.0*GAUSS3(FGP2,0.0,HIPR1(34),0.01)
	RETURN
	END
C
	FUNCTION FGP2(X)
        COMMON/PACT/BB,B1,PHI,Z1
        COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        EXTERNAL FGP3
        SAVE
        Z1=X
        FGP2=2.0*GAUSS4(FGP3,0.0,HIPR1(35),0.01)
	RETURN
	END
C
        FUNCTION FGP3(X)
        COMMON/PACT/BB,B1,PHI,Z1
        SAVE  
        R1=SQRT(B1**2+Z1**2)
        R2=SQRT(BB**2+B1**2-2.0*B1*BB*COS(PHI)+X**2)
        FGP3=B1*WDSAX1(R1)*WDSAX2(R2)
        RETURN
        END
C
C
        FUNCTION FLAP1(X)
        COMMON/PACT/BB,B1,PHI,Z1
        SAVE  
        R=SQRT(BB**2+X**2)
        FLAP1=WDSAX1(R)
        RETURN
        END
C
C
        FUNCTION FLAP2(X)
        COMMON/PACT/BB,B1,PHI,Z1
        SAVE  
        R=SQRT(BB**2+X**2)
        FLAP2=WDSAX2(R)
        RETURN
        END
C
C
C The next three subroutines are for Monte Carlo generation 
C according to a given function FHB. One calls first HIFUN 
C with assigned channel number I, low and up limits. Then to 
C generate the distribution one can call HIRND(I) which gives 
C you a random number generated according to the given function.
C 
	SUBROUTINE HIFUN(I,XMIN,XMAX,FHB)
	COMMON/HIJHB/RR(10,4097),XX(10,4097)
	EXTERNAL FHB
        SAVE
	FNORM=GAUSS1(FHB,XMIN,XMAX,0.001)
	DO 100 J=1,4097
		XX(I,J)=XMIN+(XMAX-XMIN)*(J-1)/4096.0
		XDD=XX(I,J)
		RR(I,J)=GAUSS1(FHB,XMIN,XDD,0.001)/FNORM
100	CONTINUE
	RETURN
	END
C
C
C
	FUNCTION HIRND(I)
	COMMON/HIJHB/RR(10,4097),XX(10,4097)
	COMMON/RANSEED/NSEED
	SAVE  
	RX=RLU(NSEED)
	JL=0
	JU=4098
10	IF(JU-JL.GT.1) THEN
	   JM=(JU+JL)/2
	   IF((RR(I,4097).GT.RR(I,1)).EQV.(RX.GT.RR(I,JM))) THEN
	      JL=JM
	   ELSE
	      JU=JM
	   ENDIF
	GO TO 10
	ENDIF
	J=JL
	IF(J.LT.1) J=1
	IF(J.GE.4097) J=4096
C     Fix problem with taking average of adjacent values instead of
C     interpolating
	FRAC=(RX-RR(I,J))/(RR(I,J+1)-RR(I,J))
	HIRND=XX(I,J)+FRAC*(XX(I,J+1)-XX(I,J))
	RETURN
	END	
C
C
C
C
C	This generate random number between XMIN and XMAX
	FUNCTION HIRND2(I,XMIN,XMAX)
	COMMON/HIJHB/RR(10,4097),XX(10,4097)
	COMMON/RANSEED/NSEED
	SAVE  
	IF(XMIN.LT.XX(I,1)) XMIN=XX(I,1)
	IF(XMAX.GT.XX(I,4097)) XMAX=XX(I,4097)
	JMIN=1+4096*(XMIN-XX(I,1))/(XX(I,4097)-XX(I,1))
	JMAX=1+4096*(XMAX-XX(I,1))/(XX(I,4097)-XX(I,1))
	RX=RR(I,JMIN)+(RR(I,JMAX)-RR(I,JMIN))*RLU(NSEED)
	JL=0
	JU=4098
10	IF(JU-JL.GT.1) THEN
	   JM=(JU+JL)/2
	   IF((RR(I,4097).GT.RR(I,1)).EQV.(RX.GT.RR(I,JM))) THEN
	      JL=JM
	   ELSE
	      JU=JM
	   ENDIF
	GO TO 10
	ENDIF
	J=JL
	IF(J.LT.1) J=1
	IF(J.GE.4097) J=4096
C     Fix problem with taking average of adjacent values instead of
C     interpolating
	FRAC=(RX-RR(I,J))/(RR(I,J+1)-RR(I,J))
	HIRND2=XX(I,J)+FRAC*(XX(I,J+1)-XX(I,J))
	RETURN
	END	
C
C
C
C
	SUBROUTINE HIJCRS
C	THIS IS TO CALCULATE THE CROSS SECTIONS OF JET PRODUCTION AND
C	THE TOTAL INELASTIC CROSS SECTIONS.
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/NJET/N,IP_CRS
	EXTERNAL FHIN,FTOT,FNJET,FTOTJET,FTOTRIG
        SAVE
	IF(HINT1(1).GE.10.0) CALL CRSJET
C			********calculate jet cross section(in mb)
C
	APHX1=HIPR1(6)*(IHNT2(1)**0.3333333-1.0)
	APHX2=HIPR1(6)*(IHNT2(3)**0.3333333-1.0)
	HINT1(11)=HINT1(14)-APHX1*HINT1(15)
     &			-APHX2*HINT1(16)+APHX1*APHX2*HINT1(17)
	HINT1(10)=GAUSS1(FTOTJET,0.0,20.0,0.01)
	HINT1(12)=GAUSS1(FHIN,0.0,20.0,0.01)
	HINT1(13)=GAUSS1(FTOT,0.0,20.0,0.01)
	HINT1(60)=HINT1(61)-APHX1*HINT1(62)
     &			-APHX2*HINT1(63)+APHX1*APHX2*HINT1(64)
	HINT1(59)=GAUSS1(FTOTRIG,0.0,20.0,0.01)
	IF(HINT1(59).EQ.0.0) HINT1(59)=HINT1(60)
	IF(HINT1(1).GE.10.0) Then
	   DO 20 I=0,20
	      N=I
	      HINT1(80+I)=GAUSS1(FNJET,0.0,20.0,0.01)/HINT1(12)
 20	   CONTINUE
	ENDIF
	HINT1(10)=HINT1(10)*HIPR1(31)
	HINT1(12)=HINT1(12)*HIPR1(31)
	HINT1(13)=HINT1(13)*HIPR1(31)
	HINT1(59)=HINT1(59)*HIPR1(31)
C		********Total and Inel cross section are calculated
C			by Gaussian integration.
	IF(IHPR2(13).NE.0) THEN
	HIPR1(33)=1.36*(1.0+36.0/HINT1(1)**2)
     &             *ALOG(0.6+0.1*HINT1(1)**2)
	HIPR1(33)=HIPR1(33)/HINT1(12)
	ENDIF
C		********Parametrized cross section for single
C			diffractive reaction(Goulianos)
	RETURN
	END
C
C
C
C
	FUNCTION FTOT(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	OMG=OMG0(X)*(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
	FTOT=2.0*(1.0-EXP(-OMG))
	RETURN
	END
C
C
C
	FUNCTION FHIN(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	OMG=OMG0(X)*(HIPR1(30)+HINT1(11))/HIPR1(31)/2.0
	FHIN=1.0-EXP(-2.0*OMG)
	RETURN
	END
C
C
C
	FUNCTION FTOTJET(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	OMG=OMG0(X)*HINT1(11)/HIPR1(31)/2.0
	FTOTJET=1.0-EXP(-2.0*OMG)
	RETURN
	END
C
C
C
	FUNCTION FTOTRIG(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	SAVE  
	OMG=OMG0(X)*HINT1(60)/HIPR1(31)/2.0
	FTOTRIG=1.0-EXP(-2.0*OMG)
	RETURN
	END
C
C
C
C
        FUNCTION FNJET(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/NJET/N,IP_CRS
	SAVE  
        OMG1=OMG0(X)*HINT1(11)/HIPR1(31)
        C0=EXP(N*ALOG(OMG1)-SGMIN(N+1))
        IF(N.EQ.0) C0=1.0-EXP(-2.0*OMG0(X)*HIPR1(30)/HIPR1(31)/2.0)
        FNJET=C0*EXP(-OMG1)
        RETURN
        END
C
C
C
C
C
        FUNCTION SGMIN(N)
        GA=0.
        IF(N.LE.2) GO TO 20
        DO 10 I=1,N-1
        Z=I
        GA=GA+ALOG(Z)
10      CONTINUE
20      SGMIN=GA
        RETURN
        END
C
C
C
	FUNCTION OMG0(X)
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON /BESEL/X4
	EXTERNAL BK
	SAVE   
	X4=HIPR1(32)*SQRT(X)
	OMG0=HIPR1(32)**2*GAUSS2(BK,X4,X4+20.0,0.01)/96.0
	RETURN
	END
C
C
C
	FUNCTION ROMG(X)
C		********This gives the eikonal function from a table
C			calculated in the first call
	DIMENSION FR(0:1000)
	SAVE 
	DATA I0/0/
	IF(I0.NE.0) GO TO 100
	DO 50 I=1,1001
	XR=(I-1)*0.01
	FR(I-1)=OMG0(XR)
50	CONTINUE
100	I0=1
	IF(X.GE.10.0) THEN
		ROMG=0.0
		RETURN
	ENDIF
	IX=INT(X*100)
	ROMG=(FR(IX)*((IX+1)*0.01-X)+FR(IX+1)*(X-IX*0.01))/0.01
	RETURN
	END
C
C
C
	FUNCTION BK(X)
	COMMON /BESEL/X4
	SAVE  
	BK=EXP(-X)*(X**2-X4**2)**2.50/15.0
	RETURN
	END
C
C
C	THIS PROGRAM IS TO CALCULATE THE JET CROSS SECTION
C	THE INTEGRATION IS DONE BY USING VEGAS
C
	SUBROUTINE CRSJET
	IMPLICIT REAL*8(A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	COMMON/NJET/N,IP_CRS
	COMMON/BVEG1/XL(10),XU(10),ACC,NDIM,NCALL,ITMX,NPRN
	COMMON/BVEG2/XI(50,10),SI,SI2,SWGT,SCHI,NDO,IT
	COMMON/BVEG3/F,TI,TSI
	COMMON/SEEDVAX/NUM1
	EXTERNAL FJET,FJETRIG
        SAVE
C
c************************
c	NCALL give the number of inner-iteration, ITMX 
C       gives the limit of out-iteration. Nprn is an option
C       ( 1: print the integration process. 0: do not print)
C
	NDIM=3
	IP_CRS=0
	CALL VEGAS(FJET,AVGI,SD,CHI2A)
	HINT1(14)=AVGI/2.5682
	IF(IHPR2(6).EQ.1 .AND. IHNT2(1).GT.1) THEN
		IP_CRS=1
		CALL VEGAS(FJET,AVGI,SD,CHI2A)
		HINT1(15)=AVGI/2.5682
	ENDIF
	IF(IHPR2(6).EQ.1 .AND. IHNT2(3).GT.1) THEN
		IP_CRS=2
		CALL VEGAS(FJET,AVGI,SD,CHI2A)
		HINT1(16)=AVGI/2.5682
	ENDIF
	IF(IHPR2(6).EQ.1.AND.IHNT2(1).GT.1.AND.IHNT2(3).GT.1) THEN
		IP_CRS=3
		CALL VEGAS(FJET,AVGI,SD,CHI2A)
		HINT1(17)=AVGI/2.5682
	ENDIF
C		********Total inclusive jet cross section(Pt>P0) 
C
	IF(IHPR2(3).NE.0) THEN
	   IP_CRS=0
	   CALL VEGAS(FJETRIG,AVGI,SD,CHI2A)
	   HINT1(61)=AVGI/2.5682
	   IF(IHPR2(6).EQ.1 .AND. IHNT2(1).GT.1) THEN
	      IP_CRS=1
	      CALL VEGAS(FJETRIG,AVGI,SD,CHI2A)
	      HINT1(62)=AVGI/2.5682
	   ENDIF
	   IF(IHPR2(6).EQ.1 .AND. IHNT2(3).GT.1) THEN
	      IP_CRS=2
	      CALL VEGAS(FJETRIG,AVGI,SD,CHI2A)
	      HINT1(63)=AVGI/2.5682
	   ENDIF
	   IF(IHPR2(6).EQ.1.AND.IHNT2(1).GT.1.AND.IHNT2(3).GT.1) THEN
	      IP_CRS=3
	      CALL VEGAS(FJETRIG,AVGI,SD,CHI2A)
	      HINT1(64)=AVGI/2.5682
	   ENDIF
	ENDIF
C			********cross section of trigger jet
C
	RETURN
	END
C
C
C
	FUNCTION FJET(X,WGT)
	IMPLICIT REAL*8(A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	DIMENSION X(10)
        SAVE
        WGT=WGT
	PT2=(HINT1(1)**2/4.0-HIPR1(8)**2)*X(1)+HIPR1(8)**2
	XT=2.0*DSQRT(PT2)/HINT1(1)
	YMX1=DLOG(1.0/XT+DSQRT(1.0/XT**2-1.0))
	Y1=2.0*YMX1*X(2)-YMX1
	YMX2=DLOG(2.0/XT-DEXP(Y1))
	YMN2=DLOG(2.0/XT-DEXP(-Y1))
	Y2=(YMX2+YMN2)*X(3)-YMN2
	FJET=2.0*YMX1*(YMX2+YMN2)*(HINT1(1)**2/4.0-HIPR1(8)**2)
     &		*G(Y1,Y2,PT2)/2.0
	RETURN
	END
C
C
C
	FUNCTION FJETRIG(X,WGT)
	IMPLICIT REAL*8(A-H,O-Z)
	REAL HIPR1(100),HINT1(100),PTMAX,PTMIN
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	DIMENSION X(10)
        SAVE 
	PTMIN=ABS(HIPR1(10))-0.25
	PTMIN=MAX(PTMIN,HIPR1(8))
	AM2=0.D0
	IF(IHPR2(3).EQ.3) THEN
	   AM2=HIPR1(7)**2
	   PTMIN=MAX(0.0,HIPR1(10))
	ENDIF
	PTMAX=ABS(HIPR1(10))+0.25
	IF(HIPR1(10).LE.0.0) PTMAX=HINT1(1)/2.0-AM2
	IF(PTMAX.LE.PTMIN) PTMAX=PTMIN+0.25
	PT2=(PTMAX**2-PTMIN**2)*X(1)+PTMIN**2
	AMT2=PT2+AM2
	XT=2.0*DSQRT(AMT2)/HINT1(1)
	YMX1=DLOG(1.0/XT+DSQRT(1.0/XT**2-1.0))
	Y1=2.0*YMX1*X(2)-YMX1
	YMX2=DLOG(2.0/XT-DEXP(Y1))
	YMN2=DLOG(2.0/XT-DEXP(-Y1))
	Y2=(YMX2+YMN2)*X(3)-YMN2
	IF(IHPR2(3).EQ.3) THEN
	   GTRIG=2.0*GHVQ(Y1,Y2,AMT2)
	ELSE IF(IHPR2(3).EQ.2) THEN
	   GTRIG=2.0*GPHOTON(Y1,Y2,PT2)
	ELSE
	   GTRIG=G(Y1,Y2,PT2)
	ENDIF
	FJETRIG=2.0*YMX1*(YMX2+YMN2)*(PTMAX**2-PTMIN**2)
     &		*GTRIG/2.0
	RETURN
	END
C
C
C
	FUNCTION GHVQ(Y1,Y2,AMT2)
	IMPLICIT REAL*8 (A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	DIMENSION F(2,7)
        SAVE  
	XT=2.0*DSQRT(AMT2)/HINT1(1)
	X1=0.50*XT*(DEXP(Y1)+DEXP(Y2))
	X2=0.50*XT*(DEXP(-Y1)+DEXP(-Y2))
	SS=X1*X2*HINT1(1)**2
	AF=4.0
	IF(IHPR2(18).NE.0) AF=5.0
	DLAM=HIPR1(15)
	APH=12.0*3.1415926/(33.0-2.0*AF)/DLOG(AMT2/DLAM**2)
C
	CALL PARTON(F,X1,X2,AMT2)
C
	Gqq=4.0*(COSH(Y1-Y2)+HIPR1(7)**2/AMT2)/(1.D0+COSH(Y1-Y2))/9.0
     &      *(F(1,1)*F(2,2)+F(1,2)*F(2,1)+F(1,3)*F(2,4)
     &        +F(1,4)*F(2,3)+F(1,5)*F(2,6)+F(1,6)*F(2,5))
	Ggg=(8.D0*COSH(Y1-Y2)-1.D0)*(COSH(Y1-Y2)+2.0*HIPR1(7)**2/AMT2
     &      -2.0*HIPR1(7)**4/AMT2**2)/(1.0+COSH(Y1-Y2))/24.D0
     &      *F(1,7)*F(2,7)
C
	GHVQ=(Gqq+Ggg)*HIPR1(23)*3.14159*APH**2/SS**2
	RETURN
	END
C
C
C
	FUNCTION GPHOTON(Y1,Y2,PT2)
	IMPLICIT REAL*8 (A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	DIMENSION F(2,7)
        SAVE  
	XT=2.0*DSQRT(PT2)/HINT1(1)
	X1=0.50*XT*(DEXP(Y1)+DEXP(Y2))
	X2=0.50*XT*(DEXP(-Y1)+DEXP(-Y2))
	Z=DSQRT(1.D0-XT**2/X1/X2)
	SS=X1*X2*HINT1(1)**2
	T=-(1.0-Z)/2.0
	U=-(1.0+Z)/2.0
	AF=3.0
	DLAM=HIPR1(15)
	APH=12.0*3.1415926/(33.0-2.0*AF)/DLOG(PT2/DLAM**2)
	APHEM=1.0/137.0
C
	CALL PARTON(F,X1,X2,PT2)
C
	G11=-(U**2+1.0)/U/3.0*F(1,7)*(4.0*F(2,1)+4.0*F(2,2)
     &      +F(2,3)+F(2,4)+F(2,5)+F(2,6))/9.0
	G12=-(T**2+1.0)/T/3.0*F(2,7)*(4.0*F(1,1)+4.0*F(1,2)
     &      +F(1,3)+F(1,4)+F(1,5)+F(1,6))/9.0
	G2=8.0*(U**2+T**2)/U/T/9.0*(4.0*F(1,1)*F(2,2)
     &     +4.0*F(1,2)*F(2,1)+F(1,3)*F(2,4)+F(1,4)*F(2,3)
     &     +F(1,5)*F(2,6)+F(1,6)*F(2,5))/9.0
C
	GPHOTON=(G11+G12+G2)*HIPR1(23)*3.14159*APH*APHEM/SS**2
	RETURN
	END
C
C
C
C
	FUNCTION G(Y1,Y2,PT2)
	IMPLICIT REAL*8 (A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	DIMENSION F(2,7)
        SAVE 
	XT=2.0*DSQRT(PT2)/HINT1(1)
	X1=0.50*XT*(DEXP(Y1)+DEXP(Y2))
	X2=0.50*XT*(DEXP(-Y1)+DEXP(-Y2))
	Z=DSQRT(1.D0-XT**2/X1/X2)
	SS=X1*X2*HINT1(1)**2
	T=-(1.0-Z)/2.0
	U=-(1.0+Z)/2.0
	AF=3.0
	DLAM=HIPR1(15)
	APH=12.0*3.1415926/(33.0-2.0*AF)/DLOG(PT2/DLAM**2)
C
	CALL PARTON(F,X1,X2,PT2)
C
	G11=( (F(1,1)+F(1,2))*(F(2,3)+F(2,4)+F(2,5)+F(2,6))
     &      +(F(1,3)+F(1,4))*(F(2,5)+F(2,6)) )*SUBCRS1(T,U)
C
	G12=( (F(2,1)+F(2,2))*(F(1,3)+F(1,4)+F(1,5)+F(1,6))
     &      +(F(2,3)+F(2,4))*(F(1,5)+F(1,6)) )*SUBCRS1(U,T)
C
	G13=(F(1,1)*F(2,1)+F(1,2)*F(2,2)+F(1,3)*F(2,3)+F(1,4)*F(2,4)
     &      +F(1,5)*F(2,5)+F(1,6)*F(2,6))*(SUBCRS1(U,T)
     &      +SUBCRS1(T,U)-8.D0/T/U/27.D0)
C
	G2=(AF-1)*(F(1,1)*F(2,2)+F(2,1)*F(1,2)+F(1,3)*F(2,4)
     &     +F(2,3)*F(1,4)+F(1,5)*F(2,6)+F(2,5)*F(1,6))*SUBCRS2(T,U)
C
	G31=(F(1,1)*F(2,2)+F(1,3)*F(2,4)+F(1,5)*F(2,6))*SUBCRS3(T,U)
	G32=(F(2,1)*F(1,2)+F(2,3)*F(1,4)+F(2,5)*F(1,6))*SUBCRS3(U,T)
C
	G4=(F(1,1)*F(2,2)+F(2,1)*F(1,2)+F(1,3)*F(2,4)+F(2,3)*F(1,4)+
     1	F(1,5)*F(2,6)+F(2,5)*F(1,6))*SUBCRS4(T,U)
C
	G5=AF*F(1,7)*F(2,7)*SUBCRS5(T,U)
C
	G61=F(1,7)*(F(2,1)+F(2,2)+F(2,3)+F(2,4)+F(2,5)
     &      +F(2,6))*SUBCRS6(T,U)
	G62=F(2,7)*(F(1,1)+F(1,2)+F(1,3)+F(1,4)+F(1,5)
     &      +F(1,6))*SUBCRS6(U,T)
C
	G7=F(1,7)*F(2,7)*SUBCRS7(T,U)
C
	G=(G11+G12+G13+G2+G31+G32+G4+G5+G61+G62+G7)*HIPR1(17)*
     1	3.14159D0*APH**2/SS**2
	RETURN
	END
C
C
C
	FUNCTION SUBCRS1(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS1=4.D0/9.D0*(1.D0+U**2)/T**2
	RETURN
	END
C
C
	FUNCTION SUBCRS2(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS2=4.D0/9.D0*(T**2+U**2)
	RETURN
	END
C
C
	FUNCTION SUBCRS3(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS3=4.D0/9.D0*(T**2+U**2+(1.D0+U**2)/T**2
     1	-2.D0*U**2/3.D0/T)
	RETURN
	END
C
C
	FUNCTION SUBCRS4(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS4=8.D0/3.D0*(T**2+U**2)*(4.D0/9.D0/T/U-1.D0)
	RETURN
	END
C
C
C
	FUNCTION SUBCRS5(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS5=3.D0/8.D0*(T**2+U**2)*(4.D0/9.D0/T/U-1.D0)
	RETURN
	END
C
C
	FUNCTION SUBCRS6(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS6=(1.D0+U**2)*(1.D0/T**2-4.D0/U/9.D0)
	RETURN
	END
C
C
	FUNCTION SUBCRS7(T,U)
	IMPLICIT REAL*8(A-H,O-Z)
	SUBCRS7=9.D0/2.D0*(3.D0-T*U-U/T**2-T/U**2)
	RETURN
	END
C
C
C
	SUBROUTINE PARTON(F,X1,X2,QQ)
	IMPLICIT REAL*8(A-H,O-Z)
	REAL HIPR1(100),HINT1(100)
        COMMON/HIPARNT/HIPR1,IHPR2(50),HINT1,IHNT2(50)
	COMMON/NJET/N,IP_CRS
	DIMENSION F(2,7) 
        SAVE
	DLAM=HIPR1(15)
	Q0=HIPR1(16)
	S=DLOG(DLOG(QQ/DLAM**2)/DLOG(Q0**2/DLAM**2))
	IF(IHPR2(7).EQ.2) GO TO 200
C*******************************************************
	AT1=0.419+0.004*S-0.007*S**2
	AT2=3.460+0.724*S-0.066*S**2
	GMUD=4.40-4.86*S+1.33*S**2
	AT3=0.763-0.237*S+0.026*S**2
	AT4=4.00+0.627*S-0.019*S**2
	GMD=-0.421*S+0.033*S**2
C*******************************************************
	CAS=1.265-1.132*S+0.293*S**2
	AS=-0.372*S-0.029*S**2
	BS=8.05+1.59*S-0.153*S**2
	APHS=6.31*S-0.273*S**2
	BTAS=-10.5*S-3.17*S**2
	GMS=14.7*S+9.80*S**2
C********************************************************
C	CAC=0.135*S-0.075*S**2
C	AC=-0.036-0.222*S-0.058*S**2
C	BC=6.35+3.26*S-0.909*S**2
C	APHC=-3.03*S+1.50*S**2
C	BTAC=17.4*S-11.3*S**2
C	GMC=-17.9*S+15.6*S**2
C***********************************************************
	CAG=1.56-1.71*S+0.638*S**2
	AG=-0.949*S+0.325*S**2
	BG=6.0+1.44*S-1.05*S**2
	APHG=9.0-7.19*S+0.255*S**2
	BTAG=-16.5*S+10.9*S**2
	GMG=15.3*S-10.1*S**2
	GO TO 300
C********************************************************
200	AT1=0.374+0.014*S
	AT2=3.33+0.753*S-0.076*S**2
	GMUD=6.03-6.22*S+1.56*S**2
	AT3=0.761-0.232*S+0.023*S**2
	AT4=3.83+0.627*S-0.019*S**2
	GMD=-0.418*S+0.036*S**2
C************************************
	CAS=1.67-1.92*S+0.582*S**2
	AS=-0.273*S-0.164*S**2
	BS=9.15+0.530*S-0.763*S**2
	APHS=15.7*S-2.83*S**2
	BTAS=-101.0*S+44.7*S**2
	GMS=223.0*S-117.0*S**2
C*********************************
C	CAC=0.067*S-0.031*S**2
C	AC=-0.120-0.233*S-0.023*S**2
C	BC=3.51+3.66*S-0.453*S**2
C	APHC=-0.474*S+0.358*S**2
C	BTAC=9.50*S-5.43*S**2
C	GMC=-16.6*S+15.5*S**2
C**********************************
	CAG=0.879-0.971*S+0.434*S**2
	AG=-1.16*S+0.476*S**2
	BG=4.0+1.23*S-0.254*S**2
	APHG=9.0-5.64*S-0.817*S**2
	BTAG=-7.54*S+5.50*S**2
	GMG=-0.596*S+1.26*S**2
C*********************************
300	B12=DEXP(GMRE(AT1)+GMRE(AT2+1.D0)-GMRE(AT1+AT2+1.D0))
	B34=DEXP(GMRE(AT3)+GMRE(AT4+1.D0)-GMRE(AT3+AT4+1.D0))
	CNUD=3.D0/B12/(1.D0+GMUD*AT1/(AT1+AT2+1.D0))
	CND=1.D0/B34/(1.D0+GMD*AT3/(AT3+AT4+1.D0))
C********************************************************
C	FUD=X*(U+D)
C	FS=X*2(UBAR+DBAR+SBAR)  AND UBAR=DBAR=SBAR
C*******************************************************
	FUD1=CNUD*X1**AT1*(1.D0-X1)**AT2*(1.D0+GMUD*X1)
	FS1=CAS*X1**AS*(1.D0-X1)**BS*(1.D0+APHS*X1
     &      +BTAS*X1**2+GMS*X1**3)
	F(1,3)=CND*X1**AT3*(1.D0-X1)**AT4*(1.D0+GMD*X1)+FS1/6.D0
	F(1,1)=FUD1-F(1,3)+FS1/3.D0
	F(1,2)=FS1/6.D0
	F(1,4)=FS1/6.D0
	F(1,5)=FS1/6.D0
	F(1,6)=FS1/6.D0
	F(1,7)=CAG*X1**AG*(1.D0-X1)**BG*(1.D0+APHG*X1
     &         +BTAG*X1**2+GMG*X1**3)
C
	FUD2=CNUD*X2**AT1*(1.D0-X2)**AT2*(1.D0+GMUD*X2)
	FS2=CAS*X2**AS*(1.D0-X2)**BS*(1.D0+APHS*X2
     &      +BTAS*X2**2+GMS*X2**3)
	F(2,3)=CND*X2**AT3*(1.D0-X2)**AT4*(1.D0+GMD*X2)+FS2/6.D0
	F(2,1)=FUD2-F(2,3)+FS2/3.D0
	F(2,2)=FS2/6.D0
	F(2,4)=FS2/6.D0
	F(2,5)=FS2/6.D0
	F(2,6)=FS2/6.D0
	F(2,7)=CAG*X2**AG*(1.D0-X2)**BG*(1.D0+APHG*X2
     &         +BTAG*X2**2+GMG*X2**3)
C***********Nuclear effect on the structure function****************
C
        IF(IHPR2(6).EQ.1 .AND. IHNT2(1).GT.1) THEN
	   AAX=1.193*ALOG(FLOAT(IHNT2(1)))**0.16666666
	   RRX=AAX*(X1**3-1.2*X1**2+0.21*X1)+1.0
     &	       +1.079*(FLOAT(IHNT2(1))**0.33333333-1.0)
     &         /DLOG(IHNT2(1)+1.0D0)*DSQRT(X1)*DEXP(-X1**2/0.01)
	   IF(IP_CRS.EQ.1 .OR.IP_CRS.EQ.3) RRX=DEXP(-X1**2/0.01)
	   DO 400 I=1,7
	      F(1,I)=RRX*F(1,I)
 400	   CONTINUE
        ENDIF
        IF(IHPR2(6).EQ.1 .AND. IHNT2(3).GT.1) THEN
	   AAX=1.193*ALOG(FLOAT(IHNT2(3)))**0.16666666
	   RRX=AAX*(X2**3-1.2*X2**2+0.21*X2)+1.0
     &	       +1.079*(FLOAT(IHNT2(3))**0.33333-1.0)
     &         /DLOG(IHNT2(3)+1.0D0)*DSQRT(X2)*DEXP(-X2**2/0.01)
	   IF(IP_CRS.EQ.2 .OR. IP_CRS.EQ.3) RRX=DEXP(-X2**2/0.01)
	   DO 500 I=1,7
	      F(2,I)=RRX*F(2,I)
 500	   CONTINUE
        ENDIF
c
	RETURN
	END
C
C
C
        FUNCTION GMRE(X)
        IMPLICIT REAL*8(A-H,O-Z)
        Z=X
        IF(X.GT.3.0D0) GO TO 10
        Z=X+3.D0
10      GMRE=0.5D0*DLOG(2.D0*3.14159265D0/Z)+Z*DLOG(Z)-Z+DLOG(1.D0
     1	+1.D0/12.D0/Z+1.D0/288.D0/Z**2-139.D0/51840.D0/Z**3
     1	-571.D0/2488320.D0/Z**4)
        IF(Z.EQ.X) GO TO 20
        GMRE=GMRE-DLOG(Z-1.D0)-DLOG(Z-2.D0)-DLOG(Z-3.D0)
20      CONTINUE
        RETURN
        END
c
C
C
	FUNCTION GMIN(N)
	IMPLICIT REAL*8(A-H,O-Z)
	GA=0.
	IF(N.LE.2) GO TO 20
	DO 10 I=1,N-1
	Z=I
	GA=GA+DLOG(Z)
10	CONTINUE
20	GMIN=GA
	RETURN
	END
C
C
C***************************************************************

	BLOCK DATA HIDATA
	REAL*8 XL(10),XU(10),ACC
	COMMON/BVEG1/XL,XU,ACC,NDIM,NCALL,ITMX,NPRN
	COMMON/SEEDVAX/NUM1
	COMMON/HIPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
	COMMON/RANSEED/NSEED
	COMMON/HIMAIN1/ NATT,EATT,JATT,NT,NP,N0,N01,N10,N11
	COMMON/HIMAIN2/KATT(130000,4),PATT(130000,4),VATT(130000,4)
	COMMON/HISTRNG/NFP(300,15),PP(300,15),NFT(300,15),PT(300,15)
	COMMON/HIJCRDN/YP(3,300),YT(3,300)
	COMMON/HIJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &               PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &               PJPM(300,500),NTJ(300),KFTJ(300,500),
     &               PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &               PJTE(300,500),PJTM(300,500)
	COMMON/HIJJET2/NSG,NJSG(900),IASG(900,3),K1SG(900,100),
     &		K2SG(900,100),PXSG(900,100),PYSG(900,100),
     &		PZSG(900,100),PESG(900,100),PMSG(900,100)
	COMMON/HIJDAT/HIDAT0(10,10),HIDAT(10)
	COMMON/HIPYINT/MINT4,MINT5,ATCO(200,20),ATXS(0:200)
        SAVE
	DATA NUM1/30123984/,XL/10*0.D0/,XU/10*1.D0/
	DATA NCALL/1000/ITMX/100/ACC/0.01/NPRN/0/
C...give all the switchs and parameters the default values
	DATA NSEED/74769375/
	DATA HIPR1/
     &	1.5,  0.35, 0.5,  0.9,  2.0,  0.1,  1.5,  2.0, -1.0, -2.25,
     &	2.0,  0.5,  1.0,  2.0,  0.2,  2.0,  2.5,  0.3,  0.1,  1.4,
     &	1.6,  1.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.4,  57.0,
     &	28.5, 3.9,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  
     &	3.141592654,
     &	0.0,  0.4,  0.1,  1.5,  0.1, 0.25, 0.0,  0.5,  0.0,  0.0,
     &	50*0.0/

	DATA IHPR2/
     &	1,    3,    0,    1,    1,    1,    1,    10,    0,    0,
     &	1,    0,    1,    1,    0,    0,    1,     0,    0,    1,
     &  1,    29*0/

	DATA HINT1/100*0/
	DATA IHNT2/50*0/

C...initialize all the data common blocks
	DATA NATT/0/EATT/0.0/JATT/0/NT/0/NP/0/N0/0/N01/0/N10/0/N11/0/
	DATA KATT/520000*0/PATT/520000*0.0/

	DATA NFP/4500*0/PP/4500*0.0/NFT/4500*0/PT/4500*0.0/

	DATA YP/900*0.0/YT/900*0.0/

	DATA NPJ/300*0/KFPJ/150000*0/PJPX/150000*0.0/PJPY/150000*0.0/
     &	PJPZ/150000*0.0/PJPE/150000*0.0/PJPM/150000*0.0/
	DATA NTJ/300*0/KFTJ/150000*0/PJTX/150000*0.0/PJTY/150000*0.0/
     &	PJTZ/150000*0.0/PJTE/150000*0.0/PJTM/150000*0.0/

	DATA NSG/0/NJSG/900*0/IASG/2700*0/K1SG/90000*0/K2SG/90000*0/
     &	PXSG/90000*0.0/PYSG/90000*0.0/PZSG/90000*0.0/PESG/90000*0.0/
     &	PMSG/90000*0.0/
	DATA MINT4/0/MINT5/0/ATCO/4000*0.0/ATXS/201*0.0/
	DATA (HIDAT0(1,I),I=1,10)/0.0,0.0,0.0,0.0,0.0,0.0,2.25,
     &          2.5,4.0,4.1/
	DATA (HIDAT0(2,I),I=1,10)/2.0,3.0,5.0,6.0,7.0,8.0,8.0,10.0,
     &		10.0,10.0/
	DATA (HIDAT0(3,I),I=1,10)/1.0,0.8,0.8,0.7,0.45,0.215,
     &          0.21,0.19,0.19,0.19/
	DATA (HIDAT0(4,I),I=1,10)/0.35,0.35,0.3,0.3,0.3,0.3,
     &          0.5,0.6,0.6,0.6/
	DATA (HIDAT0(5,I),I=1,10)/23.8,24.0,26.0,26.2,27.0,28.5,28.5,
     &		28.5,28.5,28.5/
	DATA ((HIDAT0(J,I),I=1,10),J=6,9)/40*0.0/
	DATA (HIDAT0(10,I),I=1,10)/5.0,20.0,53.0,62.0,100.0,200.0,
     &          546.0,900.0,1800.0,4000.0/
	DATA HIDAT/10*0.0/
	END
C*******************************************************************
C
C
C
C
C*******************************************************************
C   SUBROUTINE PERFORMS N-DIMENSIONAL MONTE CARLO INTEG'N
C      - BY G.P. LEPAGE   SEPT 1976/(REV)APR 1978
C*******************************************************************
C
      SUBROUTINE VEGAS(FXN,AVGI,SD,CHI2A)
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/BVEG1/XL(10),XU(10),ACC,NDIM,NCALL,ITMX,NPRN
      COMMON/BVEG2/XI(50,10),SI,SI2,SWGT,SCHI,NDO,IT
      COMMON/BVEG3/F,TI,TSI   
      EXTERNAL FXN
      DIMENSION D(50,10),DI(50,10),XIN(50),R(50),DX(10),DT(10),X(10)
     1   ,KG(10),IA(10)
      REAL*4 QRAN(10)
      SAVE
      DATA NDMX/50/,ALPH/1.5D0/,ONE/1.D0/,MDS/-1/
C
      NDO=1
      DO 1 J=1,NDIM
1     XI(1,J)=ONE
C
      ENTRY VEGAS1(FXN,AVGI,SD,CHI2A)
C         - INITIALIZES CUMMULATIVE VARIABLES, BUT NOT GRID
      IT=0
      SI=0.
      SI2=SI
      SWGT=SI
      SCHI=SI
C
      ENTRY VEGAS2(FXN,AVGI,SD,CHI2A)
C         - NO INITIALIZATION
      ND=NDMX
      NG=1
      IF(MDS.EQ.0) GO TO 2
      NG=(NCALL/2.)**(1./NDIM)
      MDS=1
      IF((2*NG-NDMX).LT.0) GO TO 2
      MDS=-1
      NPG=NG/NDMX+1
      ND=NG/NPG
      NG=NPG*ND
2     K=NG**NDIM
      NPG=NCALL/K
      IF(NPG.LT.2) NPG=2
      CALLS=NPG*K
      DXG=ONE/NG
      DV2G=(CALLS*DXG**NDIM)**2/NPG/NPG/(NPG-ONE)
      XND=ND
      NDM=ND-1
      DXG=DXG*XND
      XJAC=ONE/CALLS
      DO 3 J=1,NDIM
c***this is the line 50
      DX(J)=XU(J)-XL(J)
3     XJAC=XJAC*DX(J)
C
C   REBIN PRESERVING BIN DENSITY
C
      IF(ND.EQ.NDO) GO TO 8
      RC=NDO/XND
      DO 7 J=1,NDIM
      K=0
      XN=0.
      DR=XN
      I=K
4     K=K+1
      DR=DR+ONE
      XO=XN
      XN=XI(K,J)
5     IF(RC.GT.DR) GO TO 4
      I=I+1
      DR=DR-RC
      XIN(I)=XN-(XN-XO)*DR
      IF(I.LT.NDM) GO TO 5
      DO 6 I=1,NDM
6     XI(I,J)=XIN(I)
7     XI(ND,J)=ONE
      NDO=ND
C
8     CONTINUE
      IF(NPRN.NE.0) WRITE(16,200) NDIM,CALLS,IT,ITMX,ACC,MDS,ND
     1                           ,(XL(J),XU(J),J=1,NDIM)
C
      ENTRY VEGAS3(FXN,AVGI,SD,CHI2A)
C         - MAIN INTEGRATION LOOP
9     IT=IT+1
      TI=0.
      TSI=TI
      DO 10 J=1,NDIM
      KG(J)=1
      DO 10 I=1,ND
      D(I,J)=TI
10    DI(I,J)=TI
C
11    FB=0.
      F2B=FB
      K=0
12    K=K+1
      CALL ARAN9(QRAN,NDIM)
      WGT=XJAC
      DO 15 J=1,NDIM
      XN=(KG(J)-QRAN(J))*DXG+ONE
c*****this is the line 100
      IA(J)=XN
      IF(IA(J).GT.1) GO TO 13
      XO=XI(IA(J),J)
      RC=(XN-IA(J))*XO
      GO TO 14
13    XO=XI(IA(J),J)-XI(IA(J)-1,J)
      RC=XI(IA(J)-1,J)+(XN-IA(J))*XO
14    X(J)=XL(J)+RC*DX(J)
      WGT=WGT*XO*XND
15    CONTINUE
C
      F=WGT
      F=F*FXN(X,WGT)
      F2=F*F
      FB=FB+F
      F2B=F2B+F2
      DO 16 J=1,NDIM
      DI(IA(J),J)=DI(IA(J),J)+F
16    IF(MDS.GE.0) D(IA(J),J)=D(IA(J),J)+F2
      IF(K.LT.NPG) GO TO 12
C
      F2B=DSQRT(F2B*NPG)
      F2B=(F2B-FB)*(F2B+FB)
      TI=TI+FB
      TSI=TSI+F2B
      IF(MDS.GE.0) GO TO 18
      DO 17 J=1,NDIM
17    D(IA(J),J)=D(IA(J),J)+F2B
18    K=NDIM
19    KG(K)=MOD(KG(K),NG)+1
      IF(KG(K).NE.1) GO TO 11
      K=K-1
      IF(K.GT.0) GO TO 19
C
C   FINAL RESULTS FOR THIS ITERATION
C
      TSI=TSI*DV2G
      TI2=TI*TI
      WGT=TI2/(TSI+1.0d-37)
      SI=SI+TI*WGT
      SI2=SI2+TI2
      SWGT=SWGT+WGT
      SWGT=SWGT+1.0D-37
      SI2=SI2+1.0D-37
      SCHI=SCHI+TI2*WGT
      AVGI=SI/(SWGT)
      SD=SWGT*IT/(SI2)
      CHI2A=SD*(SCHI/SWGT-AVGI*AVGI)/(IT-.999)
      SD=DSQRT(ONE/SD)
C****this is the line 150
      IF(NPRN.EQ.0) GO TO 21
      TSI=DSQRT(TSI)
      WRITE(16,201) IT,TI,TSI,AVGI,SD,CHI2A
      IF(NPRN.GE.0) GO TO 21
      DO 20 J=1,NDIM
20    WRITE(16,202) J,(XI(I,J),DI(I,J),D(I,J),I=1,ND)
C
C   REFINE GRID
C
21    DO 23 J=1,NDIM
      XO=D(1,J)
      XN=D(2,J)
      D(1,J)=(XO+XN)/2.
      DT(J)=D(1,J)
      DO 22 I=2,NDM
      D(I,J)=XO+XN
      XO=XN
      XN=D(I+1,J)
      D(I,J)=(D(I,J)+XN)/3.
22    DT(J)=DT(J)+D(I,J)
      D(ND,J)=(XN+XO)/2.
23    DT(J)=DT(J)+D(ND,J)
C
      DO 28 J=1,NDIM
      RC=0.
      DO 24 I=1,ND
      R(I)=0.
      IF (DT(J).GE.1.0D18) THEN
       WRITE(6,*) '************** A SINGULARITY >1.0D18'
C      WRITE(5,1111)
C1111  FORMAT(1X,'**************IMPORTANT NOTICE***************')
C      WRITE(5,1112)
C1112  FORMAT(1X,'THE INTEGRAND GIVES RISE A SINGULARITY >1.0D18')
C      WRITE(5,1113)
C1113  FORMAT(1X,'PLEASE CHECK THE INTEGRAND AND THE LIMITS')
C      WRITE(5,1114)
C1114  FORMAT(1X,'**************END NOTICE*************')
      END IF    
      IF(D(I,J).LE.1.0D-18) GO TO 24
      XO=DT(J)/D(I,J)
      R(I)=((XO-ONE)/XO/DLOG(XO))**ALPH
24    RC=RC+R(I)
      RC=RC/XND
      K=0
      XN=0.
      DR=XN
      I=K
25    K=K+1
      DR=DR+R(K)
      XO=XN
c****this is the line 200
      XN=XI(K,J)
26    IF(RC.GT.DR) GO TO 25
      I=I+1
      DR=DR-RC
      XIN(I)=XN-(XN-XO)*DR/(R(K)+1.0d-30)
      IF(I.LT.NDM) GO TO 26
      DO 27 I=1,NDM
27    XI(I,J)=XIN(I)
28    XI(ND,J)=ONE
C
      IF(IT.LT.ITMX.AND.ACC*DABS(AVGI).LT.SD) GO TO 9
200   FORMAT('0INPUT PARAMETERS FOR VEGAS:  NDIM=',I3,'  NCALL=',F8.0
     1    /28X,'  IT=',I5,'  ITMX=',I5/28X,'  ACC=',G9.3
     2    /28X,'  MDS=',I3,'   ND=',I4/28X,'  (XL,XU)=',
     3    (T40,'( ',G12.6,' , ',G12.6,' )'))
201   FORMAT(///' INTEGRATION BY VEGAS' / '0ITERATION NO.',I3,
     1    ':   INTEGRAL =',G14.8/21X,'STD DEV  =',G10.4 /
     2    ' ACCUMULATED RESULTS:   INTEGRAL =',G14.8 /
     3    24X,'STD DEV  =',G10.4 / 24X,'CHI**2 PER IT''N =',G10.4)
202   FORMAT('0DATA FOR AXIS',I2 / ' ',6X,'X',7X,'  DELT I  ',
     1    2X,' CONV''CE  ',11X,'X',7X,'  DELT I  ',2X,' CONV''CE  '
     2   ,11X,'X',7X,'  DELT I  ',2X,' CONV''CE  ' /
     2    (' ',3G12.4,5X,3G12.4,5X,3G12.4))
      RETURN
      END
C
C
      SUBROUTINE ARAN9(QRAN,NDIM)
      DIMENSION QRAN(10)
      COMMON/SEEDVAX/NUM1
      DO 1 I=1,NDIM
    1 QRAN(I)=RLU(NUM1)
      RETURN
      END

C
C
C*********GAUSSIAN ONE-DIMENSIONAL INTEGRATION PROGRAM*************
C
	FUNCTION GAUSS1(F,A,B,EPS)
	EXTERNAL F
	DIMENSION W(12),X(12)
	DATA CONST/1.0E-12/
	DATA W/0.1012285,.2223810,.3137067,.3623838,.0271525,
     &         .0622535,0.0951585,.1246290,.1495960,.1691565,
     &         .1826034,.1894506/
	DATA X/0.9602899,.7966665,.5255324,.1834346,.9894009,
     &         .9445750,0.8656312,.7554044,.6178762,.4580168,
     &         .2816036,.0950125/
	DELTA=CONST*ABS(A-B)
	GAUSS1=0.0
	AA=A
5	Y=B-AA
	IF(ABS(Y).LE.DELTA) RETURN
2	BB=AA+Y
	C1=0.5*(AA+BB)
	C2=C1-AA
	S8=0.0
	S16=0.0
	DO 1 I=1,4
	U=X(I)*C2
1	S8=S8+W(I)*(F(C1+U)+F(C1-U))
	DO 3 I=5,12
	U=X(I)*C2
3	S16=S16+W(I)*(F(C1+U)+F(C1-U))
	S8=S8*C2
	S16=S16*C2
	IF(ABS(S16-S8).GT.EPS*(1.+ABS(S16))) GOTO 4
	GAUSS1=GAUSS1+S16
	AA=BB
	GOTO 5
4	Y=0.5*Y
	IF(ABS(Y).GT.DELTA) GOTO 2
	WRITE(6,7)
	GAUSS1=0.0
	RETURN
7	FORMAT(1X,'GAUSS1....TOO HIGH ACURACY REQUIRED')
	END
C
C
C
	FUNCTION GAUSS2(F,A,B,EPS)
	EXTERNAL F
	DIMENSION W(12),X(12)
	DATA CONST/1.0E-12/
	DATA W/0.1012285,.2223810,.3137067,.3623838,.0271525,
     &         .0622535,0.0951585,.1246290,.1495960,.1691565,
     &         .1826034,.1894506/
	DATA X/0.9602899,.7966665,.5255324,.1834346,.9894009,
     &         .9445750,0.8656312,.7554044,.6178762,.4580168,
     &         .2816036,.0950125/
	DELTA=CONST*ABS(A-B)
	GAUSS2=0.0
	AA=A
5	Y=B-AA
	IF(ABS(Y).LE.DELTA) RETURN
2	BB=AA+Y
	C1=0.5*(AA+BB)
	C2=C1-AA
	S8=0.0
	S16=0.0
	DO 1 I=1,4
	U=X(I)*C2
1	S8=S8+W(I)*(F(C1+U)+F(C1-U))
	DO 3 I=5,12
	U=X(I)*C2
3	S16=S16+W(I)*(F(C1+U)+F(C1-U))
	S8=S8*C2
	S16=S16*C2
	IF(ABS(S16-S8).GT.EPS*(1.+ABS(S16))) GOTO 4
	GAUSS2=GAUSS2+S16
	AA=BB
	GOTO 5
4	Y=0.5*Y
	IF(ABS(Y).GT.DELTA) GOTO 2
	WRITE(6,7)
	GAUSS2=0.0
	RETURN
7	FORMAT(1X,'GAUSS2....TOO HIGH ACURACY REQUIRED')
	END
C
C
C
	FUNCTION GAUSS3(F,A,B,EPS)
	EXTERNAL F
	DIMENSION W(12),X(12)
	DATA CONST/1.0E-12/
	DATA W/0.1012285,.2223810,.3137067,.3623838,.0271525,
     &         .0622535,0.0951585,.1246290,.1495960,.1691565,
     &         .1826034,.1894506/
	DATA X/0.9602899,.7966665,.5255324,.1834346,.9894009,
     &         .9445750,0.8656312,.7554044,.6178762,.4580168,
     &         .2816036,.0950125/
	DELTA=CONST*ABS(A-B)
	GAUSS3=0.0
	AA=A
5	Y=B-AA
	IF(ABS(Y).LE.DELTA) RETURN
2	BB=AA+Y
	C1=0.5*(AA+BB)
	C2=C1-AA
	S8=0.0
	S16=0.0
	DO 1 I=1,4
	U=X(I)*C2
1	S8=S8+W(I)*(F(C1+U)+F(C1-U))
	DO 3 I=5,12
	U=X(I)*C2
3	S16=S16+W(I)*(F(C1+U)+F(C1-U))
	S8=S8*C2
	S16=S16*C2
	IF(ABS(S16-S8).GT.EPS*(1.+ABS(S16))) GOTO 4
	GAUSS3=GAUSS3+S16
	AA=BB
	GOTO 5
4	Y=0.5*Y
	IF(ABS(Y).GT.DELTA) GOTO 2
	WRITE(6,7)
	GAUSS3=0.0
	RETURN
7	FORMAT(1X,'GAUSS3....TOO HIGH ACURACY REQUIRED')
	END
C
C
C
C
	FUNCTION GAUSS4(F,A,B,EPS)
	EXTERNAL F
	DIMENSION W(12),X(12)
	DATA CONST/1.0E-12/
	DATA W/0.1012285,.2223810,.3137067,.3623838,.0271525,
     &         .0622535,0.0951585,.1246290,.1495960,.1691565,
     &         .1826034,.1894506/
	DATA X/0.9602899,.7966665,.5255324,.1834346,.9894009,
     &         .9445750,0.8656312,.7554044,.6178762,.4580168,
     &         .2816036,.0950125/
	DELTA=CONST*ABS(A-B)
	GAUSS4=0.0
	AA=A
5	Y=B-AA
	IF(ABS(Y).LE.DELTA) RETURN
2	BB=AA+Y
	C1=0.5*(AA+BB)
	C2=C1-AA
	S8=0.0
	S16=0.0
	DO 1 I=1,4
	U=X(I)*C2
1	S8=S8+W(I)*(F(C1+U)+F(C1-U))
	DO 3 I=5,12
	U=X(I)*C2
3	S16=S16+W(I)*(F(C1+U)+F(C1-U))
	S8=S8*C2
	S16=S16*C2
	IF(ABS(S16-S8).GT.EPS*(1.+ABS(S16))) GOTO 4
	GAUSS4=GAUSS4+S16
	AA=BB
	GOTO 5
4	Y=0.5*Y
	IF(ABS(Y).GT.DELTA) GOTO 2
	WRITE(6,7)
	GAUSS4=0.0
	RETURN
7	FORMAT(1X,'GAUSS4....TOO HIGH ACURACY REQUIRED')
	END
C
C
C
C
C
	SUBROUTINE TITLE
	WRITE(6,200)
200	FORMAT(//10X,
     &  '**************************************************'/10X,
     &  '*     |      \\       _______      /  ------/     *'/10X,
     &  '*   ----- ------     |_____|     /_/     /       *'/10X,
     &  '*    ||\\    /        |_____|      /    / \\       *'/10X,
     &  '*    /| \\  /_/       /_______    /_  /    \\_     *'/10X,
     &  '*   / |     / /     /  /  / |        -------     *'/10X,
     &  '*     |    / /\\       /  /  |     /     |        *'/10X,
     &  '*     |   / /  \\     /  / \\_|    /   -------     *'/10X,
     &  '*                                                *'/10X,
     &  '**************************************************'/10X,
     &  '                      HIJING                      '/10X,
     &  '       Heavy Ion Jet INteraction Generator        '/10X,
     &  '                        by                        '/10X,
     &  '	    X. N. Wang  and  M. Gyulassy           '/10X,
     &  ' 	    Lawrence Berkeley Laboratory           '//)	
	RETURN
	END
