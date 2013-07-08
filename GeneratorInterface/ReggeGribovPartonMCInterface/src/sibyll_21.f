C=======================================================================
C          SSSSSS   IIIIIII  BBBBB   YY      YY   L        L
C         S            I     B    B    YY  YY     L        L
C          SSSSS       I     BBBBB       YY       L        L
C               S      I     B    B      YY       L        L
C         SSSSSS    IIIIIII  BBBBB       YY       LLLLLLL  LLLLLLL
C=======================================================================
C  Code for SIBYLL:  hadronic interaction Monte Carlo event generator
C=======================================================================
C
C   Version 2.1     (28-Sep-2001)
C
C       By   Ralph Engel
C            R.S. Fletcher
C            T.K. Gaisser
C            Paolo Lipari
C            Todor Stanev
C
C-----------------------------------------------------------------------
C***  Please  have people who want this code contact one of the authors.
C***  Please report any problems.       ****
C
C      For a correct copy contact:
C                REngel@bartol.udel.edu
C                Gaisser@bartol.udel.edu
C                Stanev@bartol.udel.edu
C                Lipari@roma1.infn.it
C
c Sept 15, 2008:  all COMMONS aligned for double precision   by D. Heck
C-----------------------------------------------------------------------



      SUBROUTINE SIBYLL (K_beam, IATARG, Ecm)
C-----------------------------------------------------------------------
C...Main routine for the production of hadronic events,
C.  generates an inelastic hadronic interaction of
C.  a `projectile particle' of code K_beam with a
C.  target nucleus of mass number A = IATARG (integer)
C.  IATARG = 0 is an "air" nucleus  (superposition of oxygen and nitrogen)
C.  with c.m. energy for the hadron-nucleon system Ecm (GeV)
C.
C.  Allowed values of K_beam: 7,8,9,10,11,12,13,14,-13,-14
C.                            pi+-,K+-,KL,KS,p,n,pbar,nbar
C.
C.  The output is contained in COMMON /S_PLIST/ that contains:
C.
C.     NP           number of final particles
C.     P(1:NP, 1:5) 4-momenta + masses of the final particles
C.     LLIST (1:NP) codes of final particles.
C.  the reaction is studied in the c.m. of  hadron-nucleon system
C.
C.  The COMMON block /S_CHIST/ contains information about the
C.  the structure of the  generated event:
C.    NW   = number of wounded nucleons
C.    NJET = total number of hard interactions
C.    NSOF = total number of soft interactions
C.    NNSOF (1:NW) = number of soft pomeron cuts in each interaction
C.    NNJET (1:NW) = number of minijets produced in each interaction
C.    XJ1 (1:Index) = x1  for each string
C.    XJ2 (1:Index) = x2   "   "     "
C.    PTJET (1:Index) = pT   "   "     "
C.    NNPJET (1:Index) = total number of particles in each string
C.    NNPSTR (1:2*NW) = number of particles in each `beam string'
C.    JDIF(1:NW) = diffraction code
C----------------------------------------------------------------------
      SAVE

      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      PARAMETER (NW_max = 20)
      PARAMETER (NS_max = 20, NH_max = 50)
      PARAMETER (NJ_max = (NS_max+NH_max)*NW_max)
      COMMON /S_CHIST/ X1J(NJ_max),X2J(NJ_max),
     &    X1JSUM(NW_max),X2JSUM(NW_max),PTJET(NJ_max),PHIJET(NJ_max),
     &    NNPJET(NJ_max),NNPSTR(2*NW_max),NNSOF(NW_max),NNJET(NW_max),
     &    JDIF(NW_max),NW,NJET,NSOF
      COMMON /S_CCSTR/ X1(2*NW_max),X2(2*NW_max),
     &    PXB(2*NW_max),PYB(2*NW_max),PXT(2*NW_max),PYT(2*NW_max),
     &    IFLB(2*NW_max),IFLT(2*NW_max)
      COMMON /S_CLDIF/ LDIFF
      COMMON /S_CQDIS/ PPT0 (33),ptflag
      COMMON /S_CUTOFF/ STR_mass_val, STR_mass_sea

      DIMENSION LL(6:14)
      DATA LL /7*2,2*1/
      DATA FOX /0.257/

      if(Ndebug.gt.1)
     &  print *,' SIBYLL: called with (K_beam,IATARG,Ecm):',
     &  K_beam,IATARG,Ecm

      kb = K_beam
      SQS = Ecm
      S = SQS*SQS

      Ncall = Ncall+1

 100  CONTINUE

      NP = 0
      NJET = 0
      NSOF = 0
      IATARGET = IATARG

C...Generate an 'air' interaction by choosing Nitrogen or Oxygen

      IF (IATARGET .EQ. 0) THEN
          R = S_RNDM(NP)
          IATARGET = 14
          IF (R .LT. FOX)  IATARGET = 16
      ENDIF
      L = LL(IABS(KB))

C...Generate number NW wounded nucleons, and diffraction code.

1000  CALL SIB_START_EV (Ecm, L, IATARGET, NW, JDIF)

C...limits on simulation of pure diffraction dissociation
      IF((LDIFF.NE.0).and.(NW.EQ.1)) THEN
         IF((LDIFF.EQ.-1) .AND. (JDIF(1).NE.0) ) GOTO 1000
         IF((LDIFF.EQ. 1) .AND. ((JDIF(1).NE.0).AND.(JDIF(1).NE.3)))
     +     GOTO 1000
         IF((LDIFF.EQ. 5) .AND. (JDIF(1).EQ.2)) GOTO 1000
         IF((LDIFF.GE. 2) .AND. (LDIFF.LE.4)) THEN
           JDIF(1) = LDIFF-1
         ENDIF
      ENDIF

C...Diffractive/non-diffractive interactions

      IF((NW.EQ.1).and.(JDIF(1).NE.0)) THEN
        CALL SIB_DIFF (KB, JDIF(1), Ecm, 1, IREJ)
      ELSE
        CALL SIB_NDIFF (KB, IATARGET, Ecm, 1, IREJ)
      ENDIF

      IF (IREJ.NE.0) THEN
        if(Ndebug.gt.2) print *,
     &    'SIBYLL: rejection (Ecm,Ncall,Nw,JDIF):',Ecm,Ncall,NW,JDIF(1)
        GOTO 100
      ENDIF

C...Check energy-momentum conservation

      CALL PFsum(1,NP,Esum,PXsum,PYsum,PZsum,NF)
      IF (ABS(Esum/(0.5*Ecm*FLOAT(NW+1)) - 1.) .GT. 1.E-03)  THEN
ctp         WRITE(*,*) ' SIBYLL: energy not conserved (L,call): ',L,Ncall
ctp         WRITE(*,*) ' sqs_inp = ', Ecm, ' sqs_out = ', Esum
ctp         CALL SIB_LIST(6)
ctp         WRITE(*,*) ' SIBYLL: event rejected'
         goto 100
      ENDIF

C...list final state particles
      if(Ndebug.gt.10) call sib_list(6)

      RETURN
      END


      SUBROUTINE SIB_NDIFF (K_beam, IATARGET, Ecm, Irec, IREJ)
C----------------------------------------------------------------------
C...Non-diffractive or multiple non-diff./diff. interactions
C.    Irec  flag to avoid recursive calls of SIB_DIFF and SIB_NDIFF
C----------------------------------------------------------------------
      SAVE

      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)

      PARAMETER (NW_max = 20)
      PARAMETER (NS_max = 20, NH_max = 50)
      PARAMETER (NJ_max = (NS_max+NH_max)*NW_max)
      COMMON /S_CHIST/ X1J(NJ_max),X2J(NJ_max),
     &    X1JSUM(NW_max),X2JSUM(NW_max),PTJET(NJ_max),PHIJET(NJ_max),
     &    NNPJET(NJ_max),NNPSTR(2*NW_max),NNSOF(NW_max),NNJET(NW_max),
     &    JDIF(NW_max),NW,NJET,NSOF
      COMMON /S_CCSTR/ X1(2*NW_max),X2(2*NW_max),
     &    PXB(2*NW_max),PYB(2*NW_max),PXT(2*NW_max),PYT(2*NW_max),
     &    IFLB(2*NW_max),IFLT(2*NW_max)

      COMMON /S_CLDIF/ LDIFF
      COMMON /S_CQDIS/ PPT0 (33),ptflag
      COMMON /S_CUTOFF/ STR_mass_val, STR_mass_sea

      DIMENSION X2JET(NW_max),BET(2*NW_max),GAM(2*NW_max),EE(2*NW_max)

      DIMENSION QMAS(33)
      DIMENSION LL(6:14)
      DATA QMAS
     &  /2*0.35,0.6,7*0.,2*1.1,1.25,7*0.,1.25,1.1,1.25,7*0,2*1.25,1.5/
      DATA LL /7*2,2*1/

      if(Ndebug.gt.1)
     &  print *,' SIB_NDIFF: called with (K_beam,IATARGET,Ecm,Irec):',
     &  K_beam,IATARGET,Ecm,Irec

      IREJ = 1

      NP_0    = NP
      SQS_0   = SQS

      SQS   = Ecm
      S     = SQS*SQS

*        print *,' current NP,QSQ (-3) ',NP,SQS,NP_0

C...`soft increase of pT'

C Setting ptflag = 0 will result in
C underestimating the P_t at high energies.
      if (ptflag.gt.0.0) then
            ptu=.3+.08*log10(sqs/30.)
            pts=.45+.08*log10(sqs/30.)
            ptqq=.6+.08*log10(sqs/30.)
            PPT0 (1) = PTU
            PPT0 (2) = PTU
            PPT0 (3) = PTS
            PPT0 (10) = PTQQ
            DO J=11,33
                PPT0(J) = PTQQ
            ENDDO
      endif

C...energy-dependent transverse momentum cutoff

      PTmin = PAR(10)+PAR(11)*EXP(PAR(12)*SQRT(LOG(SQS)))
      XMIN = 4.*PTmin**2/S
      ZMIN = LOG(XMIN)

2000  CONTINUE

C...sample multiple interaction configuration
*        print *,' current NP,QSQ (-2a) ',NP,SQS,NP_0

      L = LL(IABS(K_beam))
      DO I=1,NW
        if(JDIF(I).eq.0) then
          CALL CUT_PRO(L, SQS, PTmin, NNSOF(I), NNJET(I))
        else
          NNSOF(I) = 1
          NNJET(I) = 0
        endif
      ENDDO

*        print *,' current NP,QSQ (-2b) ',NP,SQS,NP_0

C...sample x values

      ITRY = 0
3000  CONTINUE
      ITRY = ITRY+1
      IF(ITRY.GT.5) GOTO 2000
      NP = NP_0
      NJET = 0
      NSOF = 0
      Nall = 0
      X1JET = 0.
      DO JW=1,NW
C...hard sea-sea interactions
         X2JET(JW) = 0.
         X1JSUM(JW) = 0.
         X2JSUM(JW) = 0.
         DO JJ=1,NNJET(JW)
           Nall = Nall+1
           NJET = NJET+1
*          print *,' Ncall,JW,NW,Njet,NNJET(JW),Nall',
*    &       Ncall,JW,NW,Njet,NNJET(JW),Nall
           CALL SAMPLE_hard (L,X1J(Nall),X2J(Nall),PTJET(Nall))
*        print *,' current NP,QSQ (-2c) ',NP,SQS,NP_0
           X1JET = X1JET + X1J(Nall)
           X2JET(JW) = X2JET(JW)+X2J(Nall)
           if(Ndebug.gt.2)
     &       print *,' SIB_NDIFF: hard JJ,JW,X1JET,X2JET(JW):',
     &       JJ,JW,X1JET,X2JET(JW)
           IF ((X2JET(JW).GT.0.9).OR.(X1JET.GT.0.9)) then
             if(Ndebug.gt.2) print *,
     &         ' SIB_NDIFF: not enough phase space (Ncall,Njet):',
     &         Ncall,Njet
             GOTO 3000
           ENDIF
           X1JSUM(JW) = X1JSUM(JW)+X1J(Nall)
           X2JSUM(JW) = X2JSUM(JW)+X2J(Nall)
         ENDDO
C...soft sea-sea interactions
         NSOF_JW = 0
         DO JJ=1,NNSOF(JW)-1
ctp060203           CALL SAMPLE_soft (L,STR_mass_sea,X1S,X2S,PTSOF)
           CALL SAMPLE_soft (STR_mass_sea,X1S,X2S,PTSOF)
*        print *,' current NP,QSQ (-2d) ',NP,SQS,NP_0
           IF ((X2JET(JW)+X2S.LT.0.9).AND.(X1JET+X1S.LT.0.9)) THEN
             NSOF = NSOF+1
             Nall = Nall+1
*            print *,' Ncall,JW,NW,Nsof,NNSOF(JW),Nall',
*    &         Ncall,JW,NW,Nsof,NNSOF(JW),Nall
             NSOF_JW = NSOF_JW+1
             X1J(Nall) = X1S
             X2J(Nall) = X2S
             PTjet(Nall) = PTsof
             X1JSUM(JW) = X1JSUM(JW)+X1S
             X2JSUM(JW) = X2JSUM(JW)+X2S
             X1JET = X1JET + X1S
             X2JET(JW) = X2JET(JW)+X2S
           ENDIF
           if(Ndebug.gt.2)
     &       print *,' SIB_NDIFF: soft JJ,JW,X1JET,X2JET(JW):',
     &       JJ,JW,X1JET,X2JET(JW)
         ENDDO
         NNSOF(JW) = NSOF_JW+1
ctp060203 3500    CONTINUE
      ENDDO

*        print *,' current NP,QSQ (-1) ',NP,SQS,NP_0

C...Prepare 2*NW valence/sea color strings.

      CALL BEAM_SPLIT (K_beam, NW, X1, IFLB, X1JET, LXBAD, STR_mass_val)
      IF (LXBAD .EQ. 1) then
        if(Ndebug.gt.2) print *,' BEAM_SPLIT: rejection (Ncall):',Ncall
        NP    = NP_0
        SQS   = SQS_0
        S     = SQS*SQS
        return
      ENDIF
*        print *,' current NP,QSQ (-1a) ',NP,SQS,NP_0
      DO J=1,NW
         J1=2*(J-1)+1
         J2=J1+1
*        print *,' J,J1,J2,NW ',J,J1,J2,NW
         KT=13
         IF (IATARGET .GT. 1)  KT = 13+INT(2.*S_RNDM(0))
         CALL HSPLI (KT,IFLT(J2),IFLT(J1))
*        XMINA = 2.*STR_mass_val/(SQS*(1.-X2JET(J)))
         XMINA = 1./(SQS*(1.-X2JET(J)))**2
C        XMINA = 2.*0.20/(SQS*(1.-X2JET(J)))  ! change RSF. 5-92
         CHI=CHIDIS (KT,IFLT(J2),IFLT(J1))
         XVAL=1.-X2JET(J)
         IF (XVAL.LT.XMINA) GOTO 3000
         X2(J2) = MAX(CHI*XVAL,XMINA)
         X2(J2) = MIN(X2(J2),XVAL-XMINA)
         X2(J1) = XVAL-X2(J2)
      ENDDO

C...Generates primordial pT for the partons
*        print *,' current NP,QSQ (-1b) ',NP,SQS,NP_0

      DO J=1,NW
         J1 = 2*(J-1)+1
         J2 = J1+1
         CALL PTDIS (10,PXT(J1),PYT(J1))
         if (j.eq.1) then
            CALL PTDIS (10,PXB(J2),PYB(J2))
         else
            CALL PTDIS (IFLB(J2),PXB(J2),PYB(J2))
         endif
         PXB(J1) = -PXB(J2)
         PYB(J1) = -PYB(J2)
         PXT(J2) = -PXT(J1)
         PYT(J2) = -PYT(J1)
      ENDDO

*        print *,' current NP,QSQ (-1c) ',NP,SQS,NP_0
C...Check consistency of kinematics

      DO J=1,2*NW
         EE(J) = SQS*SQRT(X1(J)*X2(J))
         XM1 = SQRT(PXB(J)**2+PYB(J)**2+QMAS(IABS(IFLB(J)))**2)
         XM2 = SQRT(PXT(J)**2+PYT(J)**2+QMAS(IABS(IFLT(J)))**2)
*        print *,' current NP,QSQ (-1d) ',NP,SQS,NP_0
*        print *,' J,IFLB(J),IFLT(J),NW ',J,IFLB(J),IFLT(J),NW
         IF (EE(J) .LT. XM1+XM2+0.3)  GOTO 2000
      ENDDO

C...Fragmentation of soft/hard sea color strings

*     print *,' current NP,SQS (0)',NP,SQS,NP_0

      DO I=1,Nall
        NOLD=NP
        CALL JET_FRAG (I)
        NNPJET (I) = NP-NOLD
*       print *,' current NP,SQS (1)',NP,SQS,NP_0
      ENDDO

C...Fragment the 2*NW valence/sea color strings

      DO JW=1,NW
        if((Irec.eq.1).and.(JDIF(JW).ne.0)) then
          J1 = 2*JW-1
          J2 = J1+1
          X1D = X1(J1)+X1(J2)
          X2D = X2(J1)+X2(J2)
          EE (J1) = SQS*SQRT(X1D*X2D)
          BET(J1) = (X1D-X2D)/(X1D+X2D)
          GAM(J1) = (X1D+X2D)/(2.*SQRT(X1D*X2D))
          if(JW.eq.1) then
            KD = K_beam
          else
            KD = 9
          endif
          Nold = NP
          call SIB_DIFF(KD, JDIF(JW), EE(J1), 0, IREJ)
          if(IREJ.ne.0) print *,' SIB_NDIFF: SIB_DIFF rejection:',Ncall
          DO K=NOLD+1,NP
            PZ = P(K,3)
            P(K,3) = GAM(J1)*(PZ+BET(J1)*P(K,4))
            P(K,4) = GAM(J1)*(P(K,4)+BET(J1)*PZ)
          ENDDO
          NNPSTR(J1) = NP-Nold
          NNPSTR(J2) = 0
        else
          DO J=2*JW-1,2*JW
            EE (J) = SQS*SQRT(X1(J)*X2(J))
            BET(J) = (X1(J)-X2(J))/(X1(J)+X2(J))
            GAM(J) = (X1(J)+X2(J))/(2.*SQRT(X1(J)*X2(J)))
            NOLD=NP
            CALL STRING_FRAG
     &        (EE(J),IFLB(J),IFLT(J),PXB(J),PYB(J),PXT(J),PYT(J),IFBAD)
            IF (IFBAD .EQ. 1) then
              if(Ndebug.gt.2)
     &          print *,' STRING_FRAG: rejection (Ncall):',Ncall
              GOTO 2000
            ENDIF
            DO K=NOLD+1,NP
              PZ = P(K,3)
              P(K,3) = GAM(J)*(PZ+BET(J)*P(K,4))
              P(K,4) = GAM(J)*(P(K,4)+BET(J)*PZ)
            ENDDO
            NNPSTR(J) = NP-NOLD
          ENDDO
        endif
      ENDDO

      IREJ = 0
      SQS   = SQS_0
      S     = SQS*SQS

      if(Ndebug.gt.2)
     &  print *,'SIB_NDIFF: generated interactions (Ns,Nh):',
     &  NSOF+NW,NJET

      RETURN
      END


      SUBROUTINE SIBNUC (IAB, IAT, SQS)
C-----------------------------------------------------------------------
C.  Routine that generates the interaction of a nucleus of
C.  mass number IAB with a  target nucleus  of mass IAT
C.  (IAT=0 : air).
C.  SQS (GeV) is the  center of mass energy of each
C.  nucleon - nucleon cross section
C-----------------------------------------------------------------------
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_PLNUC/ PA(5,40000), LLA(40000), NPA
      COMMON /S_MASS1/ AM(49), AM2(49)
      COMMON /CKFRAG/ KODFRAG
      PARAMETER (IAMAX=56)
      COMMON /CNUCMS/ B, BMAX, NTRY, NA, NB, NI, NAEL, NBEL
     +         ,JJA(IAMAX), JJB(IAMAX), JJINT(IAMAX,IAMAX)
     +         ,JJAEL(IAMAX), JJBEL(IAMAX)
      COMMON /FRAGMENTS/ PPP(3,60)
      DIMENSION SIGDIF(3)
      DIMENSION IAF(60)
      SAVE
      DATA RPOX /0.3624/

C...Target mass
      IF (IAT .EQ. 0) THEN
         IATARGET = 14 + 2*INT((1.+RPOX)*S_RNDM(0))
      ELSE
         IATARGET = IAT
      ENDIF

C...Single nucleon (proton) case

      IF (IAB .EQ. 1)  THEN
         NPA = 0
         CALL SIBYLL (13,IATARGET, SQS)
         CALL DECSIB
         DO J=1,NP
            LA = IABS(LLIST(J))
            IF (LA .LT. 10000)  THEN
               NPA = NPA + 1
               LLA(NPA) = LLIST(J)
               DO K=1,5
                  PA(K,NPA) = P(J,K)
               ENDDO
            ENDIF
         ENDDO
         RETURN
      ENDIF


C...Nuclei

      CALL SIB_SIGMA_HP(1,SQS,SIGT,SIGEL,SIG0,SIGDIF,SLOPE,RHO)
      CALL INT_NUC (IATARGET, IAB, SIG0, SIGEL)

C...fragment spectator nucleons
      NBT = NB + NBEL
      IF (KODFRAG .EQ. 1)  THEN
          CALL FRAGM1(IAB,NBT, NF, IAF)
      ELSE IF(KODFRAG .EQ. 2)  THEN
          CALL FRAGM2(IAB,NBT, NF, IAF)
      ELSE
          CALL FRAGM (IATARGET, IAB, NBT,B, NF, IAF)
      ENDIF

C...Spectator fragments
      NPA = 0
      DO J=1,NF
         NPA = NPA+1
         if(NPA.gt.40000) then
           write(6,'(1x,a,2i8)')
     &       'SIBNUC: no space left in S_PLNUC (NPA,NF)',NPA,NF
           NPA = NPA-1
           return
         endif
         LLA(NPA) = 1000+IAF(J)
         PA(1,NPA) = 0.
         PA(2,NPA) = 0.
         PA(3,NPA) = SQS/2.
         PA(4,NPA) = SQS/2.
         PA(5,NPA) = FLOAT(IAF(J))*0.5*(AM(13)+AM(14))
      ENDDO

C...Elastically scattered fragments
      DO J=1,NBEL
         NPA = NPA+1
         if(NPA.gt.40000) then
           write(6,'(1x,a,2i8)')
     &       'SIBNUC: no space left in S_PLNUC (NPA,NBEL)',NPA,NBEL
           NPA = NPA-1
           return
         endif
         LLA(NPA) = 1001
         PA(1,NPA) = 0.
         PA(2,NPA) = 0.
         PA(3,NPA) = SQS/2.
         PA(4,NPA) = SQS/2.
         PA(5,NPA) = 0.5*(AM(13)+AM(14))
      ENDDO

C...Superimpose NB  nucleon interactions
      DO JJ=1,NB
          CALL SIBYLL (13,IATARGET, SQS)
          CALL DECSIB
          DO J=1,NP
             LA = IABS(LLIST(J))
             IF (LA .LT. 10000)   THEN
                NPA = NPA + 1
                if(NPA.gt.40000) then
                  write(6,'(1x,a,2i8)')
     &              'SIBNUC: no space left in S_PLNUC (NPA,NP)',NPA,NP
                  NPA = NPA-1
                  return
                endif
                LLA(NPA) = LLIST(J)
                DO K=1,5
                    PA(K,NPA) = P(J,K)
                ENDDO
            ENDIF
         ENDDO
      ENDDO

      RETURN
      END



      FUNCTION CHIDIS (KPARTin, IFL1, IFL2)
C...Generate CHI (fraction of energy of a hadron carried by
C.                the valence quark, or diquark, as specified by IFL1)
C.  INPUT KPART = code of particle
C.        IFL1, IFL2 = codes of partons (3, 3bar of color)
C.........................................................
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CPSPL/ CCHIK(3,6:14)
      COMMON /S_CUTOFF/ STR_mass_val, STR_mass_sea
      SAVE

      kpart=IABS(kpartin)
      IFQ=IABS(IFL1)
      IF (IFQ.GT.10) IFQ=IABS(IFL2)
      CUT=2.*STR_mass_val/SQS
100   CHIDIS=S_RNDM(0)**2
      if (chidis.lt.cut) goto 100
      if (chidis.gt.(1.-cut)) goto 100
      IF((CHIDIS**2/(CHIDIS**2+CUT**2))**0.5
     +   *(1.-CHIDIS)**CCHIK(IFQ,KPART).LT.S_RNDM(0)) GOTO 100
      CHIDIS = MAX(0.5*CUT,CHIDIS)
      CHIDIS = MIN(1.-CUT,CHIDIS)
      IF (IABS(IFL1).GT.10)  CHIDIS=1.-CHIDIS
      RETURN
      END



      SUBROUTINE HSPLI (KF, KP1,KP2)
C...This subroutine splits one hadron of code KF
C.  into 2 partons of code KP1 and KP2
C.  KP1 refers to a color triplet [q or (qq)bar]
C.  KP2 to a a color anti-triplet [qbar or (qq)]
C.  allowed inputs:
C.  KF = 6:14 pi0,pi+-,k+-,k0L,k0s, p,n
C.     = -13,-14  pbar,nbar
C-------------------------------------------------
      SAVE
      L = IABS(KF)-5
C...Test for good input
      IF ( (L .LE. 0) .OR. (L.GT. 9) ) THEN
         WRITE(6,*)
     &      'HSPLI : Routine entered with illegal particle code ',KF
      ENDIF
      GOTO (50,100,200,300,400,500,500,600,700), L

50    R = S_RNDM(0)              ! pi0
      IF (R.LE.0.)  THEN
         KP1 = 1
         KP2 = -1
      ELSE
        KP1 = 2
        KP2 = -2
      ENDIF
      RETURN
100   KP1 = 1                  ! pi+
      KP2 = -2
      RETURN
200   KP1 = 2                  ! pi-
      KP2 = -1
      RETURN
300   KP1 = 1                  ! k+
      KP2 = -3
      RETURN
400   KP1 = 3                  ! k-
      KP2 = -1
      RETURN
500   KP1 = 2                  ! k0l, k0s
      KP2 = -3
      IF (S_RNDM(0).GT. 0.5)  THEN
        KP1 = 3
        KP2 = -2
      ENDIF
      RETURN
600   R = 6.*S_RNDM(0)            ! p/pbar
      IF (R .LT.3.)       THEN
        KP1 = 1
        KP2 = 12
      ELSEIF (R .LT. 4.)  THEN
        KP1 = 1
        KP2 = 21
      ELSE
        KP1 = 2
        KP2 = 11
      ENDIF
      IF (KF .LT. 0)      THEN
        KPP = KP1
        KP1 = -KP2
        KP2 = -KPP
      ENDIF
      RETURN
700   R = 6.*S_RNDM(0)                  ! n/nbar
      IF (R .LT.3.)       THEN
         KP1 = 2
         KP2 = 12
      ELSEIF (R .LT. 4.)  THEN
        KP1 = 2
        KP2 = 21
      ELSE
        KP1 = 1
        KP2 = 22
      ENDIF
      IF (KF .LT. 0)      THEN
        KPP = KP1
        KP1 = -KP2
        KP2 = -KPP
      ENDIF
      RETURN
      END



      SUBROUTINE JET_FRAG (Index)
C-----------------------------------------------------------------------
C.   Fragmentation of a jet-jet system
C.   Input : kinematical variables of a jet-jet system,
C.           taken from /S_CHIST/
C-----------------------------------------------------------------------

      REAL*8 DX1J, DX2J, DBETJ
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      PARAMETER (NW_max = 20)
      PARAMETER (NS_max = 20, NH_max = 50)
      PARAMETER (NJ_max = (NS_max+NH_max)*NW_max)
      COMMON /S_CHIST/ X1J(NJ_max),X2J(NJ_max),
     &    X1JSUM(NW_max),X2JSUM(NW_max),PTJET(NJ_max),PHIJET(NJ_max),
     &    NNPJET(NJ_max),NNPSTR(2*NW_max),NNSOF(NW_max),NNJET(NW_max),
     &    JDIF(NW_max),NW,NJET,NSOF
      SAVE
      DATA PGG /1./

      if(Ndebug.gt.2) then
        print *,' JET_FRAG: called for entry (I,NP):',Index,NP
        print *,' JET_FRAG: (X1J,X2J,PTjet):',X1J(Index),X2J(Index),
     &    PTjet(Index)
      endif

      E0 = SQRT(S*X1J(Index)*X2J(Index))
      TH = ASIN(MIN(0.999999,2.*PTJET(Index)/E0))
      FI = 6.283185*S_RNDM(0)
      NOLD = NP
      IF ( (E0.LT.8.) .OR. (S_RNDM(0).GT.PGG)) THEN
         IS = -1 + 2.*INT(1.9999*S_RNDM(0))
 100     IFL1 = IS*(INT((2.+0.3)*S_RNDM(0))+1)
         XM = 2.*QMASS(IFL1)+0.3
         if(E0.LE.XM) GOTO 100
         CALL STRING_FRAG (E0,IFL1,-IFL1,0.,0.,0.,0.,IFBAD)
         if(IFBAD.ne.0) print *,
     &     ' JET_FRAG: rejection in STRING_FRAG (IFL,E0):',IFL1,E0
      ELSE
         CALL GG_FRAG(E0)
      ENDIF
      DX1J = X1J(Index)
      DX2J = X2J(Index)
      DBETJ = (DX1J-DX2J)/(DX1J+DX2J)
      CALL SIROBO (NOLD+1,NP,TH,FI,0.D0,0.D0,DBETJ)

      if(Ndebug.gt.2) print *,' JET_FRAG: particles produced:',NP-NOLD

      RETURN
      END



      SUBROUTINE STRING_FRAG(E0,IFL1,IFL2,PX1,PY1,PX2,PY2,IFBAD)
C-----------------------------------------------------------------------
C.  This routine fragments a string of energy E0
C.  the ends of the strings  have flavors IFL1 and IFL2
C.  the particles produced are in the  jet-jet frame
C.  with IFL1 going in the +z direction
C.     E0 = total energy in jet-jet system
C.  This version consider also a primordial pT attached
C.  to the ends of the string PX1,PY1,  PX2,PY2
C.  OUTPUT:  IFBAD =1  kinematically impossible decay
c
c      Modified Nov. 91.  RSF and TSS to fragment symmetrically
c      ie forward and backward are fragmented as leading.
c      Change- Dec. 92  RSF.  call to ptdis moved- to use flavor
c      of NEW quark in fragmentation.
C-----------------------------------------------------------------------

      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_MASS1/ AM(49), AM2(49)
      DIMENSION WW(2,2), PTOT(4), PX(3),PY(3),IFL(3)
      DIMENSION LPOINT(3000), PMQ(3)
      LOGICAL LRANK
      SAVE
      DATA LRANK/.true./

      if(Ndebug.gt.2) then
        print *,
     &    ' STRING_FRAG: called with (E0,IFL1,IFL2,PX1,PY1,PX2,PY2)',
     &    E0,IFL1,IFL2,PX1,PY1,PX2,PY2
        print *,' STRING_FRAG: NP before fragmentation:',NP
      endif

C...initialise
      NTRY = 0
      IFBAD = 0
200      NTRY = NTRY + 1
      IF (NTRY .GT. 50)  THEN
         IFBAD = 1
         RETURN
      ENDIF
      I = NP
      DO K=1,2
         WW(K,1) = 1.
         WW(K,2) = 0.
      ENDDO
      PX(1) = PX1
      PY(1) = PY1
      PX(2) = PX2
      PY(2) = PY2
      PX(3) = 0.
      PY(3) = 0.
      PTOT (1) = PX1+PX2
      PTOT (2) = PY1+PY2
      PTOT (3) = 0.
      PTOT (4) = E0
      IFL(1) = IFL1
      IFL(2) = IFL2
      PMQ(1) = QMASS(IFL(1))
      PMQ(2) = QMASS(IFL(2))

      IBLEAD = 0
C
C      SET FLAG FOR GENERATION OF LEADING PARTICLES.
C      "AND" IS FOR PPBAR ( DIQUARK AT BOTH ENDS)
C      "OR" IS FOR PP, PPI, ( DIQUARK AT ONE END.)
C
      IF (IABS(IFL1) .GT. 10 .AND. IABS(IFL2) .GT. 10)  THEN
         IBLEAD = 2
         I = I+1
         JT = 1.5+S_RNDM(0)
         GOTO 350
      ENDIF
      IF (IABS(IFL1) .GT. 10 .OR. IABS(IFL2) .GT. 10)  THEN
         IBLEAD = 1
         I = I+1
         JT = 2
         IF (IABS(IFL2) .GT. 10) JT = 1
         GOTO 350
      ENDIF

C...produce new particle: side, pT
300   continue
      I=I+1
      if(i.gt.8000) then
        write(6,'(1x,a,i8)')
     &    'STRING_FRAG: no space left in S_PLIST:',I
        stop
      endif
      IF (IBLEAD .GT. 0)  THEN
           JT = 3 - JT
           GO TO 350
       ENDIF
c
ctp060203 349     continue
         JT=1.5+S_RNDM(0)
 350      JR=3-JT
      LPOINT(I) = JT

C...particle ID and pt.
ctp060203 999        continue
      CALL SIB_IFLAV (IFL(JT), 0, IFL(3), LLIST(I))
ctp060302 991    continue
      PMQ(3) = QMASS(IFL(3))
      P(I,5) = AM(IABS(LLIST(I)))
      CALL PTDIS (IFL(3), PX(3),PY(3))
C...fill transverse momentum
      P(I,1) = PX(JT) + PX(3)
      P(I,2) = PY(JT) + PY(3)
      XMT2 = P(I,5)**2+P(I,1)**2+P(I,2)**2


C...test end of fragmentation

      WREM2 = PTOT(4)**2-PTOT(1)**2-PTOT(2)**2-PTOT(3)**2
      IF (WREM2 .LT. 0.1)  GOTO 200
*     WMIN = PMQ(1)+PMQ(2)+2.*PMQ(3)+ 0.6 + (2.*S_RNDM(0)-1.)*0.2
      WMIN = PMQ(1)+PMQ(2)+2.*PMQ(3)+ 1.1 + (2.*S_RNDM(0)-1.)*0.2
c      WMIN = PMQ(jr)+sqrt(xmt2)+pmq(3)+ 1.1 +(2.*S_RNDM(0)-1.)*0.2
c      IF (WREM2 .LT. WMIN**2) goto 400
      IF (WREM2 .LT. WMIN**2)    Then!   goto 400
         if (abs(ifl(3)).ne.3) GOTO 400
          goto 200
      endif

C...Choose z
      IF (IBLEAD .GT. 0.and.abs(ifl(jt)).gt.10)  THEN
C        Special frag. for leading Baryon only
         Z = ZBLEAD (IABS(LLIST(I)))
         IBLEAD = IBLEAD - 1
      ELSE
         Z = ZDIS (IFL(3),ifl(jt),XMT2)
      ENDIF

      WW(JT,2) = Z*WW(JT,1)
      WW(JR,2) = XMT2/(WW(JT,2)*E0**2)

      P(I,3) = WW(1,2)*0.5*E0 - WW(2,2)*0.5*E0
      P(I,4) = WW(1,2)*0.5*E0 + WW(2,2)*0.5*E0

      DO J=1,4
         PTOT (J) = PTOT(J) - P(I,J)
      ENDDO
      DO K=1,2
         WW(K,1) = WW(K,1) - WW(K,2)
      ENDDO

C...Reset pT and flavor at ends of the string
      PX(JT) = -PX(3)
      PY(JT) = -PY(3)
      IFL(JT) =-IFL(3)
      PMQ(JT) = PMQ(3)
      GOTO 300

C...Final two hadrons
400      IF (IFL(JR)*IFL(3) .GT. 100)  GOTO 200
      CALL SIB_IFLAV (IFL(JR), -IFL(3), IFLA, LLIST(I+1))
      P(I+1,5) = AM(IABS(LLIST(I+1)))
      P(I,1)   = PX(JT)+PX(3)
      P(I,2)   = PY(JT)+PY(3)
      I1 = I+1
      P(I+1,1) = PX(JR)-PX(3)
      P(I+1,2) = PY(JR)-PY(3)
      XM1 = P(I,5)**2+P(I,1)**2+P(I,2)**2
      XM2 = P(I1,5)**2+P(I1,1)**2+P(I1,2)**2
      IF (SQRT(XM1)+SQRT(XM2) .GT. SQRT(WREM2)) GOTO 200
      WREM = SQRT(WREM2)
      EA1 = (WREM2+XM1-XM2)/(2.*WREM)
      PA2 = (EA1**2-XM1)
      if (pa2.gt.0)  then
            PA = SQRT(PA2)
      else
            goto 200
      endif
      BA = PTOT(3)/PTOT(4)
      GA = PTOT(4)/WREM
      S = FLOAT(3-2*JT)
      P(I,3) = GA*(BA*EA1+S*PA)
      P(I,4) = GA*(EA1+BA*S*PA)
      P(I+1,3) = PTOT(3)-P(I,3)
      P(I+1,4) = PTOT(4)-P(I,4)
      NA= NP+1
      NP=I+1

C...reorder  particles along chain (in rank)
      IF (LRANK)  THEN
      N1 = NA-1
      N2 = 0
      DO J=NA,NP
         IF(LPOINT(J) .EQ. 2)  THEN
            N2=N2+1
            LLIST (NP+N2) = LLIST(J)
            DO K=1,5
               P(NP+N2,K)=P(J,K)
            ENDDO
         ELSE
            N1= N1+1
            IF (N1.LT.J)   THEN
               LLIST(N1) = LLIST(J)
               DO K=1,5
                  P(N1,K) = P(J,K)
               ENDDO
            ENDIF
         ENDIF
      ENDDO
      JJ=N1
      DO J=NP+N2,NP+1,-1
         JJ= JJ+1
         LLIST(JJ) = LLIST(J)
         DO K=1,5
             P(JJ,K) = P(J,K)
         ENDDO
      ENDDO
      ENDIF

      if(Ndebug.gt.2)
     &  print *,' STRING_FRAG: NP after fragmentation:',NP

      RETURN
      END



      FUNCTION ZDIS (IFL1,ifl2, XMT2)
C...z distribution
      COMMON /S_CZDIS/ FAin, FB0in
      COMMON /S_CZDISs/ FAs1, fAs2
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      SAVE

      fa=fain
      fb0=fb0in
CDH   correction  may 10-1996
      if (iabs(kb).ge.13) then   ! baryons only
          if (abs(ifl2).eq.3)  fa=fain+fas2
          if (abs(ifl1).eq.3)  fa=fain+fas1
      endif
      FB = FB0*XMT2
      IF(FA.GT.0.01.AND.ABS(FA-1.)/FB.LE.0.01) ZMAX=FB/(1.+FB)+
     +  (1.-FA)*FB**2/(1.+FB)**3
      IF(FA.GT.0.01.AND.ABS(FA-1.)/FB.GT.0.01) ZMAX=0.5*(1.+FB-
     +  SQRT((1.-FB)**2+4.*FA*FB))/(1.-FA)
      IF(ZMAX.LT.0.1)  ZDIV=2.75*ZMAX
      IF(ZMAX.GT.0.85)
     +     ZDIV=ZMAX-0.6/FB**2+(FA/FB)*ALOG((0.01+FA)/FB)
C...Choice if z, preweighted for peaks at low or high z
100   Z=S_RNDM(0)
      IDIV=1
      FPRE=1.
      IF (ZMAX.LT.0.1)  THEN
         IF(1..LT.S_RNDM(0)*(1.-ALOG(ZDIV)))  IDIV=2
         IF (IDIV.EQ.1)  Z=ZDIV*Z
         IF (IDIV.EQ.2)  Z=ZDIV**Z
         IF (IDIV.EQ.2)  FPRE=ZDIV/Z
      ELSEIF (ZMAX.GT.0.85)  THEN
         IF(1..LT.S_RNDM(0)*(FB*(1.-ZDIV)+1.)) IDIV=2
         IF (IDIV.EQ.1)  Z=ZDIV+ALOG(Z)/FB
         IF (IDIV.EQ.1)  FPRE=EXP(FB*(Z-ZDIV))
         IF (IDIV.EQ.2)  Z=ZDIV+Z*(1.-ZDIV)
      ENDIF
C...weighting according to the correct formula
      IF (Z.LE.FB/(50.+FB).OR.Z.GE.1.)  GOTO 100
      FVAL=(ZMAX/Z)*EXP(FB*(1./ZMAX-1./Z))
      IF(FA.GT.0.01)  FVAL=((1.-Z)/(1.-ZMAX))**FA*FVAL
      IF(FVAL.LT.S_RNDM(0)*FPRE)  GOTO 100
      ZDIS=Z
      RETURN
      END



      FUNCTION ZBLEAD (LB)
C...fragmentation function for leading baryon
C.  simple form:  f(z) = a + x**b
C   INPUT : LB = particle code.
C..................................................
      COMMON /S_CZLEAD/ CLEAD, FLEAD
c      COMMON /S_SZLEAD/ CLEADs, FLEADs
      COMMON /S_CHP/ ICHP(49), ISTR(49), IBAR(49)
      SAVE

            IC = ICHP(Lb)*ISIGN(1,Lb)

      if (lb.ge.34.and.lb.le.39)  then  ! Lambda's and Sigma's
  665               ZBLEAD = S_RNDM(0)
                if (zblead.le..01) goto 665
c          zblead=zdisn(1) ! blead**2   ! soft
      else if (ic.eq.0)     then
          zblead=zdisn(1)   ! blead**2   !soft
      else if (ic.eq.1)  then  ! fast protons only
            if (abs(lb).eq.13) then
              IF (S_RNDM(0) .LT. CLEAD)  THEN
  666               ZBLEAD = S_RNDM(0)
                if (zblead.le..01) goto 666
              ELSE
                  zblead=1.-zdisn(1)  ! zblead**2   !hard
              ENDIF
            continue
           else
               zblead=zdisn(1)  ! zblead**2   !hard
           endif
      else if (ic.eq.2)  then  ! fast delta++
          zblead=1.- zdisn(1)  ! (zblead)**.3333
      else
               zblead=S_RNDM(0) ! zdisn(1)     !hard
      endif
       RETURN
      END



      FUNCTION ZDISN (n)
C...Generate (1-x)**n
      SAVE
666   rmin=1.1
      do i=1,n+1
c  use dummy argument to prevent compiler optimization
         R1=S_RNDM(i)
         IF (R1.LE.RMIN) RMIN=R1
      ENDDO
      ZDISn=RMIN
      if (zdisn.le..01) goto 666
      if (zdisn.ge..99) goto 666
      RETURN
      END



      SUBROUTINE GG_FRAG (E0)
C...This routine fragments a  gluon-gluon system
C.  of mass E0 (GeV)
C.  the particles produced are in the  jet-jet frame
C.  oriented along the z axis
C...........................................................
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_MASS1/ AM(49), AM2(49)
      DIMENSION WW(2,2),PTOT(4),PX(3),PY(3),IFL(3),PMQ(3)
      SAVE

C...Generate the 'forward' leading particle.
100   I = NP+1
      I0 = -1 + 2.*INT(1.9999*S_RNDM(0))
      CALL SIB_IFLAV(I0,0,IFL1, LDUM)
      CALL SIB_IFLAV(IFL1,0,IFL2, LLIST(I))
      CALL PTDIS(IFL1,PX1,PY1)
      CALL PTDIS(IFL2,PX2,PY2)
      P(I,1) = PX1+PX2
      P(I,2) = PY1+PY2
      P(I,5) = AM(IABS(LLIST(I)))
      XM1 = P(I,5)**2+P(I,1)**2+P(I,2)**2
      Z1 = ZDIS (IFL1,1,0.25*XM1)
      Z2 = ZDIS (IFL2,1,0.25*XM1)
      T1  = 4.*XM1/(E0*E0*(Z1+Z2))
      P(I,4) = 0.25*E0*(Z1+Z2 + T1)
      P(I,3) = 0.25*E0*(Z1+Z2 - T1)

C...Generate the 'backward' leading particle.
      I = I+1
      CALL SIB_IFLAV(-I0,0,IFL3, LDUM)
      CALL SIB_IFLAV(IFL3,0,IFL4, LLIST(I))
      CALL PTDIS(IFL3,PX3,PY3)
      CALL PTDIS(IFL4,PX4,PY4)
      P(I,1) = PX3+PX4
      P(I,2) = PY3+PY4
      P(I,5) = AM(IABS(LLIST(I)))
      XM2 = P(I,5)**2+P(I,1)**2+P(I,2)**2
      Z3 = ZDIS (IFL3,1,0.25*XM2)
      Z4 = ZDIS (IFL4,1,0.25*XM2)
      T2  = 4.*XM2/(E0*E0*(Z3+Z4))
      P(I,4) = 0.25*E0*( Z3+Z4 + T2)
      P(I,3) = 0.25*E0*(-Z3-Z4 + T2)

C...Fragment the two remaning strings
      N0 = 0
      DO KS=1,2

      NTRY = 0
200      NTRY = NTRY+1
      I = NP+2+N0
      IF (NTRY .GT. 30)  GOTO 100

      IF (KS .EQ. 1)  THEN
         WW(1,1) = 0.5 * (1 - Z1 - 0.5*T2)
         WW(2,1) = 0.5 * (1 - Z3 - 0.5*T1)
         PX(1) = -PX1
         PY(1) = -PY1
         PX(2) = -PX3
         PY(2) = -PY3
         IFL(1) = -IFL1
         IFL(2) = -IFL3
      ELSE
         WW(1,1) = 0.5 * (1 - Z2 - 0.5*T2)
         WW(2,1) = 0.5 * (1 - Z4 - 0.5*T1)
         PX(1) = -PX2
         PY(1) = -PY2
         PX(2) = -PX4
         PY(2) = -PY4
         IFL(1) = -IFL2
         IFL(2) = -IFL4
      ENDIF
      PX(3) = 0.
      PY(3) = 0.
      PTOT (1) = PX(1)+PX(2)
      PTOT (2) = PY(1)+PY(2)
      PTOT (3) = 0.5*E0*(WW(1,1)-WW(2,1))
      PTOT (4) = 0.5*E0*(WW(1,1)+WW(2,1))

      PMQ(1) = QMASS(IFL(1))
      PMQ(2) = QMASS(IFL(2))

C...produce new particle: side, pT
300      I=I+1
      if(i.gt.8000) then
        write(6,'(1x,a,i8)')
     &    'GG_FRAG: no space left in S_PLIST:',I
        stop
      endif
      JT=1.5+S_RNDM(0)
      JR=3-JT
c      CALL PTDIS (IFL(JT), PX(3),PY(3))

C...particle ID
      CALL SIB_IFLAV (IFL(JT), 0, IFL(3), LLIST(I))
      PMQ(3) = QMASS(IFL(3))
      P(I,5) = AM(IABS(LLIST(I)))

      CALL PTDIS (IFL(3), PX(3),PY(3))

C...test end of fragmentation
      WREM2 = PTOT(4)**2-PTOT(1)**2-PTOT(2)**2-PTOT(3)**2
      IF (WREM2 .LT. 0.1)  GOTO 200
      WMIN = PMQ(1)+PMQ(2)+2.*PMQ(3)+1.1 + (2.*S_RNDM(0)-1.)*0.2
      IF (WREM2 .LT. WMIN**2)  GOTO 400

C...fill transverse momentum
      P(I,1) = PX(JT) + PX(3)
      P(I,2) = PY(JT) + PY(3)

C...Choose z
      XMT2 = P(I,5)**2+P(I,1)**2+P(I,2)**2
      Z = ZDIS (ifl(3),IFL(JT), XMT2)

      WW(JT,2) = Z*WW(JT,1)
      WW(JR,2) = XMT2/(WW(JT,2)*E0**2)

      P(I,3) = WW(1,2)*0.5*E0 - WW(2,2)*0.5*E0
      P(I,4) = WW(1,2)*0.5*E0 + WW(2,2)*0.5*E0

      DO J=1,4
         PTOT (J) = PTOT(J) - P(I,J)
      ENDDO
      DO K=1,2
         WW(K,1) = WW(K,1) - WW(K,2)
      ENDDO

C...Reset pT and flavor at ends of the string
      PX(JT) = -PX(3)
      PY(JT) = -PY(3)
      IFL(JT) =-IFL(3)
      PMQ(JT) = PMQ(3)
      GOTO 300

C...Final two hadrons
400   IF (IFL(JR)*IFL(3) .GT. 100)  GOTO 200
      CALL SIB_IFLAV (IFL(JR), -IFL(3), IFLA, LLIST(I+1))
      P(I+1,5) = AM(IABS(LLIST(I+1)))
      P(I,1)   = PX(JT)+PX(3)
      P(I,2)   = PY(JT)+PY(3)
      I1 = I+1
      P(I1,1) = PX(JR)-PX(3)
      P(I1,2) = PY(JR)-PY(3)
      XM1 = P(I,5)**2+P(I,1)**2+P(I,2)**2
      XM2 = P(I1,5)**2+P(I1,1)**2+P(I1,2)**2
      IF (SQRT(XM1)+SQRT(XM2) .GT. SQRT(WREM2)) GOTO 200
      if (ptot(4).le.0) goto 200
      WREM = SQRT(WREM2)
      EA1 = (WREM2+XM1-XM2)/(2.*WREM)
      PA2 = (EA1**2-XM1)
      if (pa2.ge.0.0) then
        PA = SQRT(pa2)
      else
       goto 200
      endif
      BA = PTOT(3)/PTOT(4)
      GA = PTOT(4)/WREM
      S = FLOAT(3-2*JT)
      P(I,3) = GA*(BA*EA1+S*PA)
      P(I,4) = GA*(EA1+BA*S*PA)
      P(I+1,3) = PTOT(3)-P(I,3)
      P(I+1,4) = PTOT(4)-P(I,4)
      N0 = I-NP-1
      ENDDO                  ! loop on two `remaining strings'
      NP = I+1
      RETURN
      END



      FUNCTION QMASS(IFL)
C-----------------------------------------------------------------------
C...Return quark or diquark constituent masses
C-----------------------------------------------------------------------
      DIMENSION QMAS(3)
      SAVE
      DATA QMAS /0.325,0.325,0.5/

      IFLA = IABS(IFL)
      IF (IFLA .LE. 3)       THEN
         QMASS = QMAS(IFLA)
      ELSE
         QMA = QMAS(IFLA/10)
         QMB = QMAS(MOD(IFLA,10))
         QMASS = QMA+QMB
      ENDIF
      RETURN
      END



      SUBROUTINE SIB_IFLAV (IFL1,IFL2_A, IFL2, KF)
C-----------------------------------------------------------------------
C.  This subroutine receives as input IFL1 the flavor code
C.  of a quark (antiquark) and  generates the antiquark (quark)
C.  of flavor code IFL2 that combine with the original parton
C.  to compose an hadron of code KF. ONLY 3 FLAVORS
C.  If (IFL2_A.NE.0) returns an hadron KF composed of IFL1 and IFL2_A
C-----------------------------------------------------------------------

      COMMON /S_CFLAFR/ PAR(20), IPAR(10)

      DIMENSION KFLA(3,3,2), CDIAG(12), KDIAG(6)
      DIMENSION KBAR(30), CFR(12), KFR(80)
      SAVE
      DATA KFLA /0,8,10,7,0,22,9,21,0,0,26,29,25,0,31,28,30,0/
      DATA CDIAG /0.5,0.25,0.5,0.25,1.,0.5,0.5,0.,0.5,0.,1.,1./
      DATA KDIAG /6,23,24,27,32,33/
      DATA KBAR /13,14,34,35,36,37,38,9*0,39,3*0,40,41,42,43,44,
     +             45,46,47,48,49/
      DATA CFR /0.75,0.,0.5,0.,0.,1.,0.1667,0.3333,0.0833,0.6667,
     +            0.1667,0.3333/
      DATA KFR/0,16,17,19,100,104,109,115,0,26,27,29,122,126,131,137
     +  ,0,40,42,47,144,158,178,205,0,1,3,6,10,15,21,28,0,0,56,57,240,
     +  246,256,271,0,0,1,3,6,10,15,21,60,61,64,70,292,307,328,356,
     +  0,1,3,6,10,15,21,28,16*0/


      IFLA = IABS(IFL1)
      IFL2A = IFL2_A
      IF (IFL2A .NE. 0)  THEN
         IFL2A = MOD(IFL2A,100)
         IFL2 = IFL2A
         IFLB = IABS(IFL2A)
         MB = 0
         IF (IFLB .GT. 10)   MB=1
         IF (IFLA .GT. 10)   MB=2
      ELSE
          MB = 2
         IF (IFLA .LT. 10)   THEN
             MB = 1
             IF ((1.+PAR(1))*S_RNDM(0).LT. 1.)  MB=0
         ENDIF
      ENDIF

      IF (MB .EQ. 0)  THEN
         IF (IFL2A.EQ.0)
     +        IFL2=ISIGN(1+INT((2.+PAR(2))*S_RNDM(0)),-IFL1)
         IFLD = MAX(IFL1,IFL2)
         IFLE = MIN(IFL1,IFL2)
         GOTO 100
      ENDIF

C...Decide if the diquark must be split
      IF (MB .EQ. 2 .AND. IFLA .GT. 100)   THEN
         IFLA = MOD(IFLA,100)
           GOTO 200
      ENDIF
      IF (MB .EQ. 2 .AND. IFLA .EQ. 0)   THEN
          IF (S_RNDM(0) .LT. PAR(8))  THEN
             MB = 0
             IFLG = MOD(IFL1,10)
             IFLH =(IFL1-IFLG)/10
             IF (S_RNDM(0) .GT. 0.5)  THEN
                IFLDUM = IFLG
                IFLG = IFLH
                IFLH = IFLDUM
             ENDIF
             IFL11=IFLG
             IFL22=ISIGN(1+INT((2.+PAR(2))*S_RNDM(0)),-IFL1)
             IFLD = MAX(IFL11,IFL22)
             IFLE = MIN(IFL11,IFL22)
             IFL2 = -IFLH*10+IFL22
             IF (S_RNDM(0) .GT. 0.5)  IFL2 = IFL22*10-IFLH
             IFL2 = IFL2+ISIGN(100,IFL2)
          ENDIF
      ENDIF

C...Form a meson: consider spin and flavor mixing for the diagonal states
100      IF (MB .EQ. 0)  THEN
         IF1 = IABS(IFLD)
         IF2 = IABS(IFLE)
         IFLC = MAX(IF1,IF2)
         KSP = INT(PAR(5)+S_RNDM(0))
         KSP = MIN(KSP,1)
         IF (IFLC.EQ.3)  KSP = INT(PAR(6)+S_RNDM(0))
         IF (IF1 .NE. IF2)   THEN
            KF = KFLA(IF1,IF2,KSP+1)
         ELSE
            R = S_RNDM(0)
            JF=1+INT(R+CDIAG(6*KSP+2*IF1-1))+
     +             INT(R+CDIAG(6*KSP+2*IF1))
            JF = MIN(JF,3)
            KF=KDIAG(JF+3*KSP)
         ENDIF
         RETURN
      ENDIF

C...Form a baryon
200      IF (IFL2A .NE. 0)   THEN
          IF (MB .EQ. 1)  THEN
             IFLD = IFLA
             IFLE = IFLB/10
             IFLF = MOD(IFLB,10)
          ELSE
             IFLD = IFLB
             IFLE = IFLA/10
             IFLF = MOD(IFLA,10)
          ENDIF
          LFR = 3+2*((2*(IFLE-IFLF))/(1+IABS(IFLE-IFLF)))
          IF(IFLD.NE.IFLE.AND.IFLD.NE.IFLF)  LFR=LFR+1
      ELSE
110          CONTINUE
          IF(MB.EQ.1)   THEN            ! generate diquark
             IFLD = IFLA
120             IFLE = 1+INT((2.+PAR(2)*PAR(3))*S_RNDM(0))
             IFLF = 1+INT((2.+PAR(2)*PAR(3))*S_RNDM(0))
             IF(IFLE.GE.IFLF.AND.PAR(4).LT.S_RNDM(0))    GOTO 120
             IF(IFLE.LT.IFLF.AND.PAR(4)*S_RNDM(0).GT.1.) GOTO 120
             IFL2=ISIGN(10*IFLE+IFLF,IFL1)
          ELSE                  ! generate quark
             IFL2=ISIGN(1+INT((2.+PAR(2))*S_RNDM(0)),IFL1)
             IFLD=IABS(IFL2)
             IFLE=IFLA/10
             IFLF=MOD(IFLA,10)
          ENDIF
C...SU(6) factors for baryon formation
             LFR=3+2*((2*(IFLE-IFLF))/(1+IABS(IFLE-IFLF)))
          IF(IFLD.NE.IFLE.AND.IFLD.NE.IFLF)  LFR=LFR+1
          WT = CFR(2*LFR-1)+PAR(7)*CFR(2*LFR)
          IF(IFLE.LT.IFLF)   WT=WT/3.
          IF (WT.LT.S_RNDM(0)) GOTO 110
      ENDIF

C...Form Baryon
      IFLG=MAX(IFLD,IFLE,IFLF)
      IFLI=MIN(IFLD,IFLE,IFLF)
      IFLH=IFLD+IFLE+IFLF-IFLG-IFLI
      KSP=2+2*INT(1.-CFR(2*LFR-1)+(CFR(2*LFR-1)+PAR(7)*
     1       CFR(2*LFR))*S_RNDM(0))

C...Distinguish Lambda- and Sigma- like particles
      IF (KSP.EQ.2.AND.IFLG.GT.IFLH.AND.IFLH.GT.IFLI)  THEN
      IF(IFLE.GT.IFLF.AND.IFLD.NE.IFLG) KSP=2+INT(0.75+S_RNDM(0))
       IF(IFLE.LT.IFLF.AND.IFLD.EQ.IFLG) KSP=3
       IF(IFLE.LT.IFLF.AND.IFLD.NE.IFLG) KSP=2+INT(0.25+S_RNDM(0))
      ENDIF
      KF=KFR(16*KSP-16+IFLG)+KFR(16*KSP-8+IFLH)+IFLI
      KF=ISIGN(KBAR(KF-40),IFL1)

      RETURN
      END



      SUBROUTINE PTDIS (IFL,PX,PY)
C...Generate pT
      COMMON /S_CQDIS/ PPT0(33),ptflag
      SAVE

      PT = PPT0(IABS(IFL))*SQRT(-ALOG(MAX(1E-10,S_RNDM(0))))
      PHI= 6.2831853*S_RNDM(0)
      PX=PT*COS(PHI)
      PY=PT*SIN(PHI)
      RETURN
      END



      SUBROUTINE SIROBO( NBEG, NEND, THE, PHI, DBEX, DBEY, DBEZ)
C **********************************************************************
C   THIS IS A SLIGHTLY ALTERED VERSION OF "LUROBO" [JETSET63.PYTHIA]   *
C SET TO WORK IN THE SIBYL ENVIROMENT. THE TRANSFORMATION IS PERFORMED *
C ON PARTICLES NUMBER FROM NBEG TO NEND. COMMON BLOCKS CHANGED.        *
C                                      TSS,   Oct '87                  *
C  modification  use directly BETA in double precision in input (PL)   *
C **********************************************************************
      COMMON /S_PLIST/ PLIST(8000,5), LLIST(8000), NP
      DIMENSION ROT(3,3),PV(3)
      DOUBLE PRECISION DP(4),DBEX,DBEY,DBEZ,DGA,DBEP,DGABEP
      SAVE

      IF(THE**2+PHI**2 .LE. 1E-20) GO TO 131
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
       DO 120 I=NBEG,NEND
       DO 100 J=1,3
 100   PV(J)=PLIST(I,J)
       DO 110 J=1,3
 110   PLIST(I,J)=ROT(J,1)*PV(1)+ROT(J,2)*PV(2)+ROT(J,3)*PV(3)
 120   CONTINUE
 131    IF(DBEX**2+DBEY**2+DBEZ**2 .LE. 1D-20) GO TO 151
C...LORENTZ BOOST (TYPICALLY FROM REST TO MOMENTUM/ENERGY=BETA)
       DGA=1D0/DSQRT(1D0-DBEX**2-DBEY**2-DBEZ**2)
       DO 140 I=NBEG, NEND
       DO 130 J=1,4
 130   DP(J)=PLIST(I,J)
       DBEP=DBEX*DP(1)+DBEY*DP(2)+DBEZ*DP(3)
       DGABEP=DGA*(DGA*DBEP/(1D0+DGA)+DP(4))
       PLIST(I,1)=DP(1)+DGABEP*DBEX
       PLIST(I,2)=DP(2)+DGABEP*DBEY
       PLIST(I,3)=DP(3)+DGABEP*DBEZ
       PLIST(I,4)=DGA*(DP(4)+DBEP)
 140   CONTINUE
 151   RETURN
      END


      SUBROUTINE BEAM_SPLIT (L, NW, XX, IFL, XJET, LXBAD, STR_mass)
C...This subroutine split a hadron of code L
C.  into 2*NW partons, each of energy XX(j) and
C.  flavor IFL.  The minimum fractional energy of
C.  each parton is X_min = 2*STR_mass/sqrt(s)
C.
C.  Variable qmas changed to STR_mass to agree with name in SIBYLL
C.      and added to calling sequenceto insure symmetry.
C.     Also a factor of (1-xjet) is added to the def. of xmin for nw=1
C.                               RSF  Apr-2-92
C---------------------------------------------------------------------

      PARAMETER (NW_max = 20)
      DIMENSION XX(2*NW_max), IFL(2*NW_max)

      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      SAVE

      DATA AC /-0.2761856692/             ! log(2) - gamma(Eulero)
      DATA GAMMA /2./
      DATA NBAD / 0 /

c-------
c  New code to handle low energy p nuc problem.
c------
      LXBAD = 0
      XMIN = 2.*STR_mass/SQS
      IF (1.-XJET .LT. FLOAT(2*NW)*XMIN)  THEN
         NBAD = NBAD + 1
         LXBAD = 1
ctp         IF (NBAD .LE. 20) THEN
ctp           WRITE (6, *) 'BEAM_SPLIT: kinematically forbidden situation'
ctp           WRITE (6, 5)  NBAD, SQS, XJET, NW
ctp         ENDIF
ctp 5       FORMAT(1X,'NBAD = ',I3,3X,'sqs = ',E10.3,
ctp     &            3X, 'x_jet = ', F9.3, 3X, ' NW = ',I2)
ctp         IF (NBAD .eq. 20) THEN
ctp           WRITE (6, *)
ctp     &     ' BEAM_SPLIT : Last warning about bad splittings '
ctp           WRITE (6, *) ' The energy threshold is probably too low.'
ctp         ENDIF
         RETURN
      ENDIF

      IF (NW .EQ. 1)  THEN
         XVAL = 1.-XJET
         GOTO 200
      ENDIF

C...Choose total energy of sea partons
      N = 2*(NW-1)
      Z1 = LOG(FLOAT(N))
      Z2 = LOG(0.5*SQS*(1.-XJET)/STR_mass-2.)
100   R=S_RNDM(0)
      Z=(Z1+AC)*(1.+R*(((Z2+AC)/(Z1+AC))**N-1.))**(1./FLOAT(N))-AC
      XSEA = XMIN*EXP(Z)
      IF ( (1.-XSEA)**GAMMA .LT. S_RNDM(0)) GOTO 100
C...Split the energy  of sea partons among the different partons
      XREM = XSEA - FLOAT(N)*XMIN
      DO J=3,N+1
c  use dummy argument to prevent compiler from optimizing
         XA = XREM*S_RNDM(j)
         XREM = XREM - XA
*        print *,' BEAM_SPLIT: XX index ',J
         XX(J) = XMIN + XA
      ENDDO
*     print *,' BEAM_SPLIT: XX index ',N+2
      XX(N+2) = XMIN + XREM
      XVAL = 1.-XSEA-XJET
C...Flavor of sea partons
      DO J=1,N/2
         J1 =  3 + (J-1)*2
*        print *,' BEAM_SPLIT: flavour indices ',J1,J1+1
c  use dummy argument to prevent compiler from optimizing
         IFL(J1) = INT(1.+1.99*S_RNDM(j))
         IFL(J1+1) = -IFL(J1)
      ENDDO
C...Prepare the valence partons
200   CALL HSPLI (L,IFL(1),IFL(2))
      CHI = CHIDIS(L,IFL(1),IFL(2))
      XX(1) = MAX(CHI*XVAL,XMIN)
      XX(1) = MIN(XX(1),XVAL-XMIN)
C      FOR MESONS, SPLIT ENERGY SYMETRICALLY.
C????? SPLIT K'S WITH ENERGY TO S QUARK?
C
      if (abs(l).le.12.and.S_RNDM(0).le.0.5) xx(1)=XVAL-XX(1)
      XX(2) = XVAL-XX(1)

      RETURN
      END

C--------------------------------------------------------------------------
C    CODE OF ANALYSIS (not needed to generate events)
C--------------------------------------------------------------------------

      SUBROUTINE PFsum(N1,N2,ETOT,PXT,PYT,PZT,NF)
C...Return the energy,px,py,pz and the number of stable
C.  particles in the list between N1 and N2
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      SAVE

      NF=0
      ETOT=0.
      PXT=0.
      PYT=0.
      PZT=0.
      DO J=N1,N2
         L = LLIST(J)
         IF (IABS(L) .LT. 10000)  THEN
           NF = NF+1
           ETOT = ETOT + ABS( P(J,4) )
           PXT = PXT + P(J,1)
           PYT = PYT + P(J,2)
           PZT = PZT + P(J,3)
         ENDIF
      ENDDO
      RETURN
      END


      SUBROUTINE QNUM (JQ,JS,JB,JBA, NC, NF)
C...Return the quantum numbers of one event
C.  JQ = charge, JB = baryon number, JS = strangeness
C.  JBA = (number of baryons+antibaryons)
C.  NC  = number of charged particles
C.  NF  = number of final particles
C..................................................
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_CHP/ ICHP(49), ISTR(49), IBAR(49)
      SAVE

      JQ = 0
      JB = 0
      JS = 0
      JBA= 0
      NC = 0
      NF = 0
      DO J=1,NP
          L = LLIST(J)
          LL = IABS(L)
          IF (LL .LT. 10000)  THEN
              IF(ICHP(LL) .NE. 0) NC = NC + 1
              NF = NF + 1
              JQ = JQ + ICHP(LL)*ISIGN(1,L)
              JB = JB + IBAR(LL)*ISIGN(1,L)
              JBA= JBA+ IBAR(LL)
              JS = JS + ISTR(LL)*ISIGN(1,L)
          ENDIF
      ENDDO
      RETURN
      END


      SUBROUTINE SIB_LIST(LUN)
C-----------------------------------------------------------------------
C...This routine prints the event record for the
C.  current event on unit LUN
C-----------------------------------------------------------------------

      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_PLIST1/ LLIST1(8000)
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      PARAMETER (NW_max = 20)
      PARAMETER (NS_max = 20, NH_max = 50)
      PARAMETER (NJ_max = (NS_max+NH_max)*NW_max)
      COMMON /S_CHIST/ X1J(NJ_max),X2J(NJ_max),
     &    X1JSUM(NW_max),X2JSUM(NW_max),PTJET(NJ_max),PHIJET(NJ_max),
     &    NNPJET(NJ_max),NNPSTR(2*NW_max),NNSOF(NW_max),NNJET(NW_max),
     &    JDIF(NW_max),NW,NJET,NSOF
      COMMON /S_CCSTR/ X1(2*NW_max),X2(2*NW_max),
     &    PXB(2*NW_max),PYB(2*NW_max),PXT(2*NW_max),PYT(2*NW_max),
     &    IFLB(2*NW_max),IFLT(2*NW_max)
      CHARACTER*6 NAMP
      COMMON /S_CNAM/ NAMP (0:49)
      COMMON /S_CHP/ ICHP(49), ISTR(49), IBAR(49)

      CHARACTER CODE*18
      CHARACTER*18 NAMDIF(0:3)
      SAVE
      DATA NAMDIF /'Non-diff. event   ',
     &  'Beam diffraction  ','Target diffraction','Double diffraction'/

      WRITE (LUN,*)
      WRITE (LUN, *) ' Event record '
      if(NW.eq.1) WRITE (LUN,*) '  ',NAMDIF(JDIF(1))
      WRITE (LUN,*) '  N_w/N_s/N_j = ', NW, NSOF, NJET
      WRITE (LUN,100)

C...Print particle list
      ichar = 0
      ibary = 0
      DO J=1,NP
        L = MOD(LLIST(J),10000)
        CODE = '                  '
        CODE(1:6) = NAMP(IABS(L))
        IF (L .LT. 0) CODE(7:9) = 'bar'
        IF(IABS(LLIST(J)) .GT. 10000)   CODE(10:10) = '*'
        WRITE (LUN,120) J, CODE, LLIST1(J), (P(J,K),K=1,4)
        if(abs(LLIST(J)).LT.10000) then
          ichar = ichar+sign(1,l)*ICHP(iabs(l))
          ibary = ibary+sign(1,l)*IBAR(iabs(l))
        endif
      ENDDO
      CALL PFsum(1,NP,Esum,PXsum,PYsum,PZsum,NF)
      WRITE(LUN,140) PXsum,PYsum,PZsum,Esum
100      FORMAT(3X,'N  Particle',12X,'Ori',6x,'PX',9x,'PY',9x,'PZ'
     +         ,9x,'E', /, 3X,70('-'))
120      FORMAT(1X,I4,1X,A18,1X,I4,2X,2(F9.3,2X),2(E9.3,2X))
140      FORMAT(1X,'Tot = ',24X,2(F9.3,2X),G9.3,2X,E9.3)
      write(LUN,'(1x,a,i3,3x,a,i3)') 'Total charge:',ichar,
     &  'baryon number:',ibary

      RETURN
      END



      SUBROUTINE KCODE (J,CODE,NC)
C...Produce the code for parton J
C.  Input K, Output CODE, NC=number of characters
C..................................................
      CHARACTER*5 CODE
      CHARACTER*1 NAMQ(3)
      SAVE
      DATA NAMQ /'U','D','S'/

      CODE = '     '
      IF(J.EQ.0)  THEN
         CODE(1:3) = 'GLU'
         NC = 3
         RETURN
      ENDIF
      JA = IABS(J)
      J1 = MOD(JA,10)
      J2 = (JA-J1)/10
      IF(JA .GT. 10) THEN
         CODE(1:1) = NAMQ(J2)
         CODE(2:2) = NAMQ(J1)
         NC = 2
      ELSE
         CODE(1:1) = NAMQ(J1)
         NC = 1
      ENDIF
      IF (J .LT. 0)  THEN
         CODE(NC+1:NC+3) = 'bar'
         NC = NC+3
      ENDIF
      RETURN
      END



C----------------------------------------------------------------------------
C  Code for sampling
C-----------------------------------------------------------------------------

ctp060302      SUBROUTINE SAMPLE_soft (L, STR_mass_min, X1,X2,PT)
      SUBROUTINE SAMPLE_soft (STR_mass_min, X1,X2,PT)
C-----------------------------------------------------------------------
C...Routine for the sampling the kinematical variables
C.  that characterize a soft cut pomeron (x1,x2, pT)
C.  from the differential cross section:
C.     d3sigma/(dx1 dx2 dpT)
C.  INPUT:  L=1 incident proton, L=2  incident pi
C.          (soft strings identical for pi and p interactions)
C.  OUTPUT:  X1, X2, PT (GeV)
C-----------------------------------------------------------------------

      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CQDIS/ PPT0(33),ptflag
      SAVE

      ZSOF = 2.*LOG(STR_mass_min/SQS)
 100  Z1=-ZSOF*S_RNDM(0)+ZSOF
      Z2=-ZSOF*S_RNDM(0)+ZSOF
      IF(Z1+Z2.LE.ZSOF) GOTO 100
      X1=EXP(Z1)
      X2=EXP(Z2)
      STR_mass2 = sqrt(X1*X2*S)/2.
 150  PT = PPT0(10)*SQRT(-ALOG(MAX(1E-10,S_RNDM(0))))
      IF(PT.GT.PTmin) GOTO 150
      IF(PT.GE.STR_mass2) GOTO 150

      RETURN
      END



      SUBROUTINE SAMPLE_hard (L, X1,X2,PT)
C-----------------------------------------------------------------------
C...Routine for the sampling the kinematical variables
C.  that determine a  jet-jet  system (x1,x2, pT)
C.  from the differential cross section:
C.     d3sigma/(dx1 dx2 dpT)
C.  This version assumes the `single parton approximation'
C.  INPUT:  L=1 incident proton, L=2  incident pi
C.  OUTPUT:  X1, X2, PT (GeV)
C-----------------------------------------------------------------------

      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      SAVE

100   Z1=ZSAMPLE (ZMIN,L)
      Z2=ZSAMPLE (ZMIN,1)
      SIG=1.-XMIN*EXP(-Z1-Z2)
      IF (SIG .LT. S_RNDM(0))  GOTO 100
      X1=EXP(Z1)
      X2=EXP(Z2)
      Q2=PTmin**2/(1.-S_RNDM(0)*SIG)
      PT=SQRT(Q2*(1.-Q2/(S*X1*X2)))

      RETURN
      END



      FUNCTION ZSAMPLE (ZMIN,L)
C...This function returns as output a value z=log(x)
C.  distributed as f(x) = g(x) + 4/9 *(q(x) + qbar(x))
C.  from a minimum value ZMIN to 0,
C.  for a proton (L=1) or a pi (L=2)
C.  needs to be initialised with: CALL ZSAMPLE_INI
C.....................................................
      COMMON /S_CZGEN/ XA,XB,XMAX,ZA,ZB,ZMAX,DX,DZ,      APART(2),
     +   FFA(2),FFB(2),
     +   DFX(2),DFZ(2),XX(200,2),ZZ(200,2),FFX(200,2),FFZ(200,2),
     +   NX,NZ
      SAVE

      F = PART_INT(ZMIN,L)*S_RNDM(0)
      IF (F .GE. FFA(L))  THEN
         ZSAMPLE = ZA - (F-FFA(L))/APART(L)
      ELSE IF (F .GE. FFB(L))  THEN
         JF = (F-FFB(L))/DFZ(L) + 1
         F0 = FFB(L) + DFZ(L)*FLOAT(JF-1)
         T = (F-F0)/DFZ(L)
         ZSAMPLE = ZZ(JF,L)*(1.-T)+ZZ(JF+1,L)*T
      ELSE
         JF = F/DFX(L)+1
         F0 = DFX(L)*FLOAT(JF-1)
         T = (F-F0)/DFX(L)
         X = XX(JF,L)*(1.-T)+XX(JF+1,L)*T
         ZSAMPLE = LOG(X)
      ENDIF

      RETURN
      END



      FUNCTION PART_INT (ZMIN,L)
C...This function returns as output the integral of
C.  the parton structure function:
C.     f(x) = g(x) + 4/9 *(q(x) + qbar(x))
C.  from xmin = exp(zmin) to 1
C.  for a proton (L=1) or a pi (L=2)
C.  needs to be initialised with: CALL ZSAMPLE_INI
C.....................................................
      COMMON /S_CZGEN/ XA,XB,XMAX,ZA,ZB,ZMAX,DX,DZ,      APART(2),
     +   FFA(2),FFB(2),
     +   DFX(2),DFZ(2),XX(200,2),ZZ(200,2),FFX(200,2),FFZ(200,2),
     +   NX,NZ
      SAVE

      IF (ZMIN .LT. ZA)  THEN
         PART_INT = FFA(L) + APART(L)*(ZA-ZMIN)
      ELSE IF (ZMIN .LT. ZB) THEN
         JZ = (ZB-ZMIN)/DZ+1
         JZ = min(JZ,199)
         Z0 = ZB-DZ*FLOAT(JZ-1)
         T = (Z0-ZMIN)/DZ
         PART_INT = FFZ(JZ,L)*(1.-T) + FFZ(JZ+1,L)*T
      ELSE
         X = EXP(ZMIN)
         JX = (XMAX-X)/DX+1
         JX = min(JX,199)
         X0 = XMAX-DX*FLOAT(JX-1)
         T = (X0-X)/DX
         PART_INT = FFX(JX,L)*(1.-T) + FFX(JX+1,L)*T
      ENDIF
      RETURN
      END



      SUBROUTINE ZSAMPLE_INI
C...This subroutine initialise the generation of
C.  z = log(x)  for the generation  of z according
C.  to the structure functions
C..................................................
      COMMON /S_CZGEN/ XA,XB,XMAX,ZA,ZB,ZMAX,DX,DZ,      APART(2),
     +   FFA(2),FFB(2),
     +   DFX(2),DFZ(2),XX(200,2),ZZ(200,2),FFX(200,2),FFZ(200,2),
     +   NX,NZ
      SAVE

      XA = 1.E-04
      XB = 1.E-01
      XMAX = 0.80
      ZA = LOG(XA)
      ZB = LOG(XB)
      ZMAX = LOG(XMAX)
      NX = 200
      NZ = 200
      DX = (XMAX-XB)/FLOAT(NX-1)
      DZ = (ZB-ZA)/FLOAT(NZ-1)

      DO L=1,2
C         very small x:  f(x) = A/x
         APART(L) = PARTON(0.,L)

C         large x: interpolation in x
         FFX(1,L) = 0.
         DO J=2,NX
            X = XMAX - DX*(FLOAT(J)-0.5)
             G = PARTON(X,L)/X
            FFX(J,L) = FFX(J-1,L)+G*DX
         ENDDO
         CALL INVERT_ARRAY (FFX(1,L),XMAX,-DX,NX,XX(1,L),FMIN,
     +                        DFX(L))

C         small x: interpolation in log(x)
         FFZ(1,L) = FFX(NX,L)
         DO J=2,NZ
            Z = ZB - DZ*(FLOAT(J)-0.5)
            X = EXP(Z)
            G = PARTON(X,L)
            FFZ(J,L) = FFZ(J-1,L)+G*DZ
         ENDDO
         CALL INVERT_ARRAY (FFZ(1,L),ZB,-DZ,NZ,ZZ(1,L),FMIN,DFZ(L))
         FFA(L) = FFZ(NZ,L)
         FFB(L) = FFX(NX,L)
      ENDDO
      RETURN
      END



      FUNCTION PARTON(X,L)
C...This function returns the structure function
C.   f(x) = x * [ g(x) + 4/9 *(q(x) + qbar(x)) ]
C.  for a proton.
C................................................

      parameter (beta=1.925978)
      SAVE

      IF (L .EQ. 2)  GOTO 1000

C...Eichten et al.  (set 1)
ctp060203 100      uv = 1.78 * x**0.5 * (1.-x**1.51)**3.5
      uv = 1.78 * x**0.5 * (1.-x**1.51)**3.5
      dv = 0.67 * x**0.4 * (1.-x**1.51)**4.5
      us = 0.182 * (1.-x)**8.54
      ss = 0.081 * (1.-x)**8.54
      qq0 = uv + dv + 4.*us + 2.*ss
      glu0 = (2.62 + 9.17*x)* (1.-x)**5.90
      parton = glu0 + 4./9.*qq0
      return

1000      continue

C...Owens set 1   from STRF from Wisc. Pheno. group. for q2=q2_min
      AV=.4
      BV=.7
c      BETA=GGAMMA(AV)*GGAMMA(BV+1.)/GGAMMA(AV+BV+1.)  =1.925978
      uv=X**(AV)*(1.-X)**BV/BETA
      dv=uv

      A=.9
      BET=5.
      us=(A*(1.-X)**BET)/6.

      A=.888
      BET=3.11
      GA1=6.0
      glu0=A*(1.-X)**BET*(1.+GA1*X)
c   Bug Fix thanks to Sue Kashahara- correct factor in front of
c   sea quarks for Owens S.F.  5-94
      qq0 = uv + dv + 6.*us
      parton = (glu0 + 4./9.*qq0)
      return

      end



      BLOCK DATA PARAM_INI
C-----------------------------------------------------------------------
C....This block data contains default values
C.   of the parameters used in fragmentation
C-----------------------------------------------------------------------

      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CZDIS/ FA, FB0
      COMMON /S_CZDISs/ FAs1, fAs2
      COMMON /S_CZLEAD/ CLEAD, FLEAD
      COMMON /S_CPSPL/ CCHIK(3,6:14)
      COMMON /S_CQDIS/ PPT0 (33),ptflag
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      COMMON /S_CUTOFF/ STR_mass_val, STR_mass_sea
      COMMON /CKFRAG/ KODFRAG
      SAVE

C...mass cutoff for soft strings
      data STR_mass_val /.35/
      data STR_mass_sea /1./
C...Longitudinal Fragmentation function
      DATA FA /0.5/, FB0 /0.8/
C...Longitudinal Fragmentation function for leading baryons
       DATA CLEAD  /0.6/, FLEAD  /0.6/
c      strange fragmentation
      data FAs1 /3./, fAs2 /3./
c      data FAs1 /0./, fAs2 /0./
C...pT of sea partons
      DATA PTFLAG /1./
      DATA PPT0 /0.30,0.30,0.450,30*0.60/
C...Splitting parameters
      DATA CCHIK /21*2.,6*3./
C...Parameters of flavor formation
      DATA PAR /0.04,0.25,0.25,0.14,0.3,0.3,0.15,0.,
     &          7.0, 11*0. /
C...Fragmentation of nuclei
      DATA KODFRAG /0/
C...Debug label and event counter
      DATA Ndebug /0/
      DATA Ncall /0/

      END



      SUBROUTINE PARAM_PRINT(LUN)

      COMMON /S_CZDIS/ FA, FB0
      COMMON /S_CZLEAD/ CLEAD, FLEAD
      COMMON /S_CPSPL/ CCHIK(3,6:14)
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CQDIS/ PPT0 (33),ptflag
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      SAVE

      WRITE (LUN, 25)
25      FORMAT( //,1x,40('-'), /
     +   ' SIBYLL MONTE CARLO PROGRAM. Version 2.1',/,1x,40('-'),/
     +   ' List of parameters: ' )

      WRITE (LUN, 31) FA, FB0
31      FORMAT (' Parameters of longitudinal fragmentation: ', /,
     +          '  f(z) = (1-z)**a * exp(-b * mt**2/z) ', /,
     +          '  a = ', f9.3, 3x, ' b = ', f9.3, ' GeV**-2' )
      WRITE (LUN, 32) CLEAD, 1./FLEAD-1.
32      FORMAT (' Parameters of leading fragmentation: ', /,
     +   '  f(z) = c + (1-z)**a ', /,
     +   '  c = ',f9.3,3x,' a = ',f9.3)

      WRITE (LUN, 35) PPT0(1), PPT0(3), PPT0(11),ppt0(10)
35      FORMAT (' <pT> of sea partons ', /,
     +   2x,'<pT>(u/d) ',F8.3,2x,'<pT>(s) ',f8.3,2x,'<pT>(qq) ',f8.3,
     +     2x,'<pT>(val) ',f8.3)

      WRITE (LUN, 120) (PAR(K),K=1,12)
120      FORMAT (1x, 'Parameters of flavor formation: ',/,
     +   3x,'PAR(1) = Prob(qq)/Prob(q) =              ',F10.2,/,
     +   3x,'PAR(2) = Prob(s)/Prob(u)  =              ',F10.2,/,
     +   3x,'PAR(3) = Prob(us)/Prob(ud) =             ',F10.2,/,
     +   3x,'PAR(4) = Prob(ud_0)/Prob(ud_1) =         ',F10.2,/,
     +   3x,'PAR(5) = Prob(Vector)/Prob(Scalar) =     ',F10.2,/,
     +   3x,'PAR(6) = Prob(K*)/Prob(K) =              ',F10.2,/,
     +   3x,'PAR(7) = Prob(spin 3/2)/Prob(spin=1/2) = ',F10.2,/,
     +   3x,'PAR(8) = Prob(B-M-Bbar)/Prob(B-Bbar) =   ',F10.2,/,
     +   3x,'PAR(9) = Phase space suppression of MI = ',F10.2,/,
     +   3x,'PAR(10)= Low-energy limit for pt cutoff= ',F10.2,/,
     +   3x,'PAR(11)= Pt cutoff factor for exp      = ',F10.2,/,
     +   3x,'PAR(12)= Pt cutoff factor in exp       = ',F10.2)

      WRITE (LUN, 40)
      WRITE (LUN, 41) CCHIK (1,13), CCHIK(2,13)
40      FORMAT(' Parameters of hadron splitting ' )
41      FORMAT('   p -> [(ud) u] splitting: alpha = ', F10.3, /,
     +         '   p -> [(uu) d] splitting: alpha = ', F10.3 )

      RETURN
      END



C-----------------------------------------------------------------------
C  Code for diffraction
C-----------------------------------------------------------------------


      SUBROUTINE SIB_DIFF (L0, JDIF1, Ecm, Irec, IREJ)
C-----------------------------------------------------------------------
C...diffraction dissociation
C.  INPUT L0 = index of "beam particle"
C.             the target is assumed to be a proton.
C.    JDIF1 = 1  "beam diffraction"
C.          = 2  "target diffraction"
C.          = 3  "double diffraction"
C     Irec  flag to avoid recursive calls of SIB_DIFF and SIB_NDIFF
C-----------------------------------------------------------------------

      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_RUN/ SQS, S, PTmin, XMIN, ZMIN, kb ,kt
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_MASS1/ AM(49), AM2(49)
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      DIMENSION XM2MIN(3), ALXMIN(3)
      DIMENSION P0(5)
      DIMENSION KK(6:14)
      SAVE

      DATA PI /3.1415926/
      DATA KK /3*2,4*3,2*1/
      DATA XM2MIN /1.5, 0.2, 0.6/                  ! M_x**2(min) GeV**2
      DATA ALXMIN /0.405465,-1.6094379,-0.5108256/ ! log[M_x**2(min)]
      DATA SLOP0 /6.5/                 ! b (slope_ for Mx**2 > 5 GeV**2
      DATA ASLOP /31.10362/            ! fit to the slope parameter.
      DATA BSLOP /-15.29012/

      if(Ndebug.gt.1)
     &  print *,' SIB_DIFF: called with (L0,JDIF1,Ecm):',
     &  L0,JDIF1,Ecm

      IREJ = 1
      LA = IABS(L0)
      XM2MAX = PAR(13)*Ecm*Ecm

C...Double diffraction
      IF (JDIF1 .EQ. 3)   THEN
         K = KK(LA)
         AL = LOG(XM2MAX/XM2MIN(K))
         ALX = ALXMIN(K) + AL*S_RNDM(0)
         XMB2 = EXP(ALX)
         XMB = SQRT (XMB2)
         AL = LOG(XM2MAX/XM2MIN(1))
         ALX = ALXMIN(1) + AL*S_RNDM(0)
         XMT2 = EXP(ALX)
         XMT = SQRT (XMT2)
         X1 = 1.+(XMB2-XMT2)/(Ecm*Ecm)
         X2 = 2.-X1
         SLOPE = MAX(SLOP0, ASLOP+BSLOP*ALX)
50       T = -LOG(S_RNDM(0))/SLOPE
         PT = SQRT(T)
         PZ1 = 0.25*Ecm*Ecm*X1*X1-XMB2-PT*PT
         PZ2 = 0.25*Ecm*Ecm*X2*X2-XMT2-PT*PT
         IF (PZ1.LT.0. .OR. PZ2.LT.0.)   GOTO 50
         PHI = PI*S_RNDM(0)
         P0(5) = XMB
         P0(4) = 0.5*Ecm*X1
         P0(1) = PT*COS(PHI)
         P0(2) = PT*SIN(PHI)
         P0(3) = SQRT(PZ1)
         CALL DIFDEC (L0, Irec, P0)
         P0(5) = XMT
         P0(4) = 0.5*Ecm*X2
         P0(1) = -P0(1)
         P0(2) = -P0(2)
         P0(3) = -SQRT(PZ2)
         CALL DIFDEC (13, Irec, P0)
         IREJ = 0
         RETURN
      ENDIF

C...Single diffraction
      IF (JDIF1.EQ. 1)  THEN
         K = KK(LA)
         EM  = AM(13)
         EM2 = AM2(13)
         L = 13
         ZD = -1.
      ELSE
         K = 1
         EM  = AM(LA)
         EM2 = AM2(LA)
         L = L0
         ZD = +1.
      ENDIF
C...Generate the mass of the diffracted system Mx (1/Mx**2 distribution)
      AL = LOG(XM2MAX/XM2MIN(K))
      ALX = ALXMIN(K) + AL*S_RNDM(0)
      XM2 = EXP(ALX)
      XM = SQRT (XM2)
      XMB = XM
      XMT = XM
C...Generate the Kinematics of the pseudoelastic hadron
      X = 1.-(XM2-EM2)/(Ecm*Ecm)
      NP = NP+1
      P(NP,4) = 0.5*Ecm*X
      SLOPE = MAX(SLOP0, ASLOP+BSLOP*ALX)
60    T = -LOG(MAX(1.E-10,S_RNDM(0)))/SLOPE
      PT = SQRT(T*X)
      PZ2 = P(NP,4)**2-EM2 - PT*PT
      IF (PZ2 .LT.0.)   GOTO 60
      PHI = PI*S_RNDM(0)
      P(NP,3) = SQRT(PZ2)*ZD
      P(NP,1) = PT*COS(PHI)
      P(NP,2) = PT*SIN(PHI)
      P(NP,5) = EM
      LLIST(NP) = L
C...Generating the hadronic system recoling against the produced particle
      P0(5) = SQRT(XM2)
      P0(4) = 0.5*Ecm*(2.-X)
      DO J=1,3
         P0(J) = -P(NP,J)
      ENDDO
      CALL DIFDEC (L0, Irec, P0)
      IREJ = 0

      RETURN
      END



      SUBROUTINE DIFDEC (L0, Irec, P0)
C-----------------------------------------------------------------------
C..."decay" of an excited state with the quantum numbers
C.   of particle L0 and the 5-momentum P0
C.   - low energy: phase space decay (fire ball model)
C.   - intermediate energy: one-string decay (longitudinal phase space)
C.   - high energy: pomeron-hadron scattering (multi-string model)
C-----------------------------------------------------------------------

      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_MASS1/ AM(49), AM2(49)
      COMMON /S_CHP/ ICHP(49), ISTR(49), IBAR(49)
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      DIMENSION P0(5), LL(10), PD(10,5), BE(3), LCON(6:14)
      SAVE
      DATA EMIN /0.7/
      DATA EMIN2 /10./
      DATA LCON /7,6,6,11,11,9,9,14,13/
      DATA PCHEX /0.33/            ! probability of charge exchange

      LA = IABS(L0)
      DELTAE = P0(5) - AM(LA)

C...pomeron-hadron scattering (pi0 is used instead of pomeron)
      IF ((IPAR(10).gt.0).and.(Irec.gt.0).and.(DELTAE.gt.EMIN2))  THEN
         N1 = NP+1

 50      CONTINUE
           CALL SIB_NDIFF(LA, 6, P0(5), 0, IREJ)
         IF(IREJ.NE.0) THEN
           NP = N1-1
           GOTO 50
         ENDIF

         DO J=1,3
            BE(J)=P0(J)/P0(4)
         ENDDO
         GA=P0(4)/P0(5)
         if(P0(3).gt.0.) then
           do i=N1,NP
             P(I,3) = -P(I,3)
           enddo
         endif
         DO I=N1,NP
            BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
            DO J=1,3
               P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)
            ENDDO
            P(I,4)=GA*(P(I,4)+BEP)
         ENDDO

C..."string-like" decay
      ELSE IF (DELTAE .GT. EMIN)  THEN
           N1 = NP+1
         CALL HSPLI(L0,IFL1,IFL2)
         IF (P0(3) .GT. 0.)  THEN
            IFLA = IFL2
            IFL2 = IFL1
            IFL1 = IFLA
         ENDIF
10         CALL STRING_FRAG (P0(5),IFL1,IFL2,0.,0.,0.,0.,IFBAD)
         IF (IFBAD .EQ. 1)  GOTO 10
         DO J=1,3
            BE(J)=P0(J)/P0(4)
         ENDDO
         GA=P0(4)/P0(5)
         DO I=N1,NP
            BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
            DO J=1,3
               P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)
            ENDDO
            P(I,4)=GA*(P(I,4)+BEP)
         ENDDO

C...Phase space decay of the excited state
      ELSE
        AV = 2.*SQRT(DELTAE)
ctp060203 100     NPI = AV*(1.+0.5*GASDEV(0))
100     NPI = AV*(1.+0.5*GASDEV(LA))
        IF(NPI.LE.0.OR.NPI.GT.9.OR.AM(LA)+NPI*AM(7)+0.02
     .            .GT.P0(5))  GOTO 100
        IF (S_RNDM(0).LT.PCHEX)  THEN
            LL(NPI+1) = LCON(LA)*ISIGN(1,L0)
            IF( (L0 .EQ. 6) .OR. (L0 .EQ. 11) )
     .             LL(NPI+1) = LL(NPI+1)+INT(1.99999*S_RNDM(0))
        ELSE
            LL(NPI+1) = L0
        ENDIF
        JQQ = ICHP(LA)*ISIGN(1,L0)-
     .            ICHP(IABS(LL(NPI+1)))*ISIGN(1,LL(NPI+1))
120     JQTOT = 0.
        DO K=1,NPI-1
           LL(K) = 6+INT(S_RNDM(0)*2.99999)
           JQTOT = JQTOT + ICHP(LL(K))
        ENDDO
        JQR = JQQ-JQTOT
        IF (JQR.LT.-1.OR.JQR.GT.1)  GOTO 120
        LL(NPI) = 6+JQR
        IF (LL(NPI) .EQ. 5)  LL(NPI)=8
        CALL DECPAR (0,P0,NPI+1,LL, PD)
        DO J=1,NPI+1
           NP = NP+1
           LLIST(NP) = LL(J)
           DO K=1,5
              P(NP,K) = PD(J,K)
           ENDDO
        ENDDO
      ENDIF

      RETURN
      END


      SUBROUTINE CUT_PRO (L, SQS, PTmin, NSOFR, NJETR)
C-----------------------------------------------------------------------
C...Generate a number of soft/hard (jet-)pairs for a 'projectile'
C.  (K=1:p),(K=2:pi) interacting with a nucleon at sqrt(s)=SQS(GeV)
C-----------------------------------------------------------------------

      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      COMMON /S_CUTOFF/ STR_mass_val, STR_mass_sea
      SAVE

      K = L
      if(K.eq.3) K = 2

      AL = LOG10 (SQS)
      IF (AL .LT. ASQSMIN)  THEN
          WRITE(*,*)  ' CUT_PRO:  low sqrt(s) ', SQS
          NSOFR = 1
          NJETR = 0
          RETURN
      ENDIF
      IF (AL .GT. ASQSMAX)  THEN
          WRITE(*,*)  ' CUT_PRO:  sqrt(s) out of bounds ', SQS
          NJETR = 0
          RETURN
      ENDIF

      J1 = (AL - ASQSMIN)/DASQS + 1
      J1 = MIN(J1,60)
      J1 = MAX(J1,1)
      J2 = J1+1
      T = (AL-ASQSMIN)/DASQS - FLOAT(J1-1)

      R = 0.9999*S_RNDM(0)
      DO I=0,NS_max
        DO J=0,NH_max
          IF (R.LT.(1.-T)*PJETC(I,J,J1,K)+T*PJETC(I,J,J2,K)) GOTO 100
        ENDDO
      ENDDO
100   CONTINUE

C...phase space limitation

 120  CONTINUE
      XM = FLOAT(2*I)*STR_mass_sea + FLOAT(2*J)*PTmin
      PACC = EXP(PAR(9)*(2.-XM)/SQS)
      IF(S_RNDM(0).GT.PACC) THEN
        IF(I+J.GT.1) THEN
          IF(I.GT.0) THEN
            I = I-1
            GOTO 120
          ELSE IF(J.GT.0) THEN
            J = J-1
            GOTO 120
          ENDIF
        ENDIF
      ENDIF

      NSOFR = I
      NJETR = J

      if(Ndebug.gt.2)
     &  print *,' CUT_PRO: (L,SQS,PTmin,Ns,Nh)',K,SQS,PTmin,I,J

      RETURN
      END


C===========================================================================
C  Code for initialization
C===========================================================================


      SUBROUTINE SIBYLL_INI
C-----------------------------------------------------------------------
C...Initialization routine for SYBILL
C.
C.  the routine fills the COMMON block /CCSIG/ that contains
C.  important information for the generation of events
C.
C     PARAMETER (NS_max = 20, NH_max = 50)
C     COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
C    &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
C.
C.  NSQS = number of energy points  (61 is current version)
C.  ASQSMIN = log_10 [sqrt(s) GeV]   minimum value
C.  ASQSMIN = log_10 [sqrt(s) GeV]   maximum value
C.  DASQS   = step  in log_10[sqrt(s)]
C.            DASQS = (ASQSMAX - ASQSMIN)/(NSQS-1)
C.
C.  SSIG(J,1) inelastic cross section for pp interaction
C.            at energy: sqrt(s)(GeV) = 10**[ASQSMIN+DASQS*(J-1)]
C.  SSIG(J,2)  inelastic cross section for pi-p interaction
C.  SSIGN(J,1) inelastic cross section for p-Air interaction
C.  SSIGN(J,2) inelastic cross section for pi-Air interaction
C.
C.  PJETC(n_s,n_j,J,1) Cumulative  probability distribution
C.                 for the production of n_s soft interactions and
C.                 n_j (n_j=0:30) jet pairs at sqrt(s) labeled
C.                 by J, for p-p interaction
C.  PJETC(n_s,n_j,J,2) Same as above for pi-p interaction
C.  ALINT(J,1)   proton-air  interaction length (g cm-2)
C.  ALINT(J,2)   pi-air  interaction length (g cm-2)
C-----------------------------------------------------------------------
      SAVE
cdh
      WRITE(*,100)
 100  FORMAT(' ','====================================================',
     *     /,' ','|                                                  |',
     *     /,' ','|                 S I B Y L L  2.1                 |',
     *     /,' ','|                                                  |',
     *     /,' ','|         HADRONIC INTERACTION MONTE CARLO         |',
     *     /,' ','|                        BY                        |',
     *     /,' ','|                   Ralph ENGEL                    |',
     *     /,' ','|           R.S. FLETCHER, T.K. GAISSER            |',
     *     /,' ','|               P. LIPARI, T. STANEV               |',
     *     /,' ','|                                                  |',
     *     /,' ','| Publication to be cited when using this program: |',
     *     /,' ','| R. Engel et al., Proc. 26th ICRC, 1 (1999) 415   |',
     *     /,' ','|                                                  |',
     *     /,' ','| last modified:  28. Sept. 2001 by R. Engel       |',
     *     /,' ','====================================================',
     *     /)
cdh

*     WRITE(*,*) ' Initialization of SIBYLL 2.1 event generator '
      CALL PAR_INI
      CALL JET_INI
      CALL ZSAMPLE_INI
      CALL BLOCK_INI
      CALL NUC_GEOM_INI
      CALL SIG_AIR_INI

      RETURN
      END



      SUBROUTINE PAR_INI
C------------------------------------------------------------
      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      SAVE

C...Model switches

C...amplitude/cross section fit parameter set
      IPAR(1) = 1
      IPAR(2) = 0

C...recursive diffraction (default=on)
      IPAR(10) = 1

C...Model parameters

C...energy dependence of PTmin
      PAR(10) = 1.
      PAR(11) = 0.065
      IF(IPAR(2).EQ.0) THEN
        PAR(12) = 0.9
      ELSE
        PAR(12) = 1.12
      ENDIF

C...max mass in diffraction dissociation (Md_max**2/s)
      PAR(13) = 0.2


      RETURN
      END



      SUBROUTINE JET_INI
C-----------------------------------------------------------------------
C...Compute table of cross sections, and table of probability
C.  for the production of multiple soft and hard interactions
C.
C.  The output of this routine  is the COMMON block /S_CCSIG/
C.  that contains  the cross sections h-p, h-Air, and the
C.  cumulative probability of NS soft and NH hard interactions
C-----------------------------------------------------------------------

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      COMMON /S_CCSIG2/ SSIG_TOT(61,3),SSIG_SD1(61,3),SSIG_SD2(61,3),
     &    SSIG_DD(61,3),SSIG_B(61,3),SSIG_RHO(61,3)

      DIMENSION Pjet(0:NS_max,0:NH_max)
      DIMENSION SIG_df(3),SIGDIF(3),SIGDIF_pi(3),
     &          PS_tab(61),PH_tab(61),PT_tab(61)
      SAVE


C...spacing in energy for table of cross sections.

      NSQS = 61
      ASQSMIN = 1.
      ASQSMAX = 7.
      DASQS = (ASQSMAX-ASQSMIN)/FLOAT(NSQS-1)

C...initialization of proton and pion tables

      DO KK=1,2

ctp         WRITE(6,'(2(/,1X,A,A))')
ctp     &     'Table: J, sqs,  PT_cut,  SIG_tot,  SIG_inel,  B_el,  ',
ctp     &     'rho,  <n_s>,  <n_h>',
ctp     &     '-----------------------------------------------------',
ctp     &     '-------------------'

         JINT = KK
         DO J=1, NSQS
           ASQS = ASQSMIN + DASQS*FLOAT(J-1)
           SQS = 10.**ASQS

           CALL SIB_SIG (JINT, SQS, PTmin,
     &                   SIG_tot, SIG_inel, SIG_df, B_el, Pjet)

C...low-energy interpolation with data-parametrizations
           call SIB_HADCSL(JINT,SQS,
     &                     SIGTOT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
           if(SQS.le.100.) then
             SIG_TOT  = SIGTOT
             SIG_inel = SIGINEL
             B_EL     = SLOPE
           else if(SQS.le.1000.) then
             Xi = log(SQS/100.)/2.30258509299405
             SIG_TOT  = Xi*SIG_TOT+(1.-Xi)*SIGTOT
             SIG_inel = Xi*SIG_inel+(1.-Xi)*SIGINEL
             B_EL     = Xi*B_EL+(1.-Xi)*SLOPE
           endif

           SSIG_TOT(J,KK) = SIG_TOT
           SSIG(J,KK)     = SIG_inel
           SSIG_SD1(J,KK) = SIGDIF(1)
           SSIG_SD2(J,KK) = SIGDIF(2)
           SSIG_DD(J,KK)  = SIG_df(3)
           SSIG_B(J,KK)   = B_EL
           SSIG_RHO(J,KK) = RHO

           PSUM = 0.
           PH = 0.
           PS = 0.
           DO NS=0,NS_max
             DO NJ=0,NH_max

               PS = PS+FLOAT(NS)*Pjet(NS,NJ)
               PH = PH+FLOAT(NJ)*Pjet(NS,NJ)

               PSUM = PSUM+Pjet(NS,NJ)
               PJETC(NS,NJ,J,KK) = PSUM

             ENDDO
           ENDDO
           PS_tab(J) = PS
           PH_tab(J) = PH
           PT_tab(J) = PTmin

ctp           WRITE(6,'(3X,I2,1P,E12.3,0P,4F8.2,3F8.3)')
ctp     &       JINT,SQS,PTmin,SIG_tot,SIG_inel,B_el,RHO,PS,PH

         ENDDO
      ENDDO

C...initialization of kaon tables

      JINT = 3

ctp      WRITE(6,'(2(/,1X,A,A))')
ctp     &  'Table: J, sqs,  PT_cut,  SIG_tot,  SIG_inel,  B_el,  ',
ctp     &  'rho,  <n_s>,  <n_h>',
ctp     &  '-----------------------------------------------------',
ctp     &  '-------------------'
      DO J=1, NSQS
        ASQS = ASQSMIN + DASQS*FLOAT(J-1)
        SQS = 10.**ASQS
C...use pion cross section rescaled for high-energy extrapolation
        SIG_tot   = SSIG_TOT(J,2)
        SIG_inel  = SSIG(J,2)
        SIG_df(1) = SSIG_SD1(J,2)
        SIG_df(2) = SSIG_SD2(J,2)
        SIG_df(3) = SSIG_DD(J,2)
        B_el = SSIG_B(J,2)
        PTmin = PT_tab(J)
        PS = PS_tab(J)
        PH = PH_tab(J)

C...low-energy interpolation with data-parametrizations
        call SIB_HADCSL(2,SQS,
     &                  SIGTOT_pi,SIGEL_pi,SIGINEL,SIGDIF_pi,SLOPE,RHO)
        call SIB_HADCSL(3,SQS,
     &                  SIGTOT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
        SIG_el    = (SIGEL/SIGEL_pi)*(SIG_TOT-SIG_inel)
        SIG_TOT   = (SIGTOT/SIGTOT_pi)*SIG_TOT
        SIG_inel  = SIG_TOT-SIG_el
        SIG_df(3) = (SIGDIF(3)/SIGDIF_pi(3))*SIG_df(3)
        if(SQS.le.100.) then
          SIG_TOT  = SIGTOT
          SIG_inel = SIGINEL
          B_EL     = SLOPE
        else if(SQS.le.1000.) then
          Xi = log(SQS/100.)/2.30258509299405
          SIG_TOT  = Xi*SIG_TOT+(1.-Xi)*SIGTOT
          SIG_inel = Xi*SIG_inel+(1.-Xi)*SIGINEL
          B_EL     = Xi*B_EL+(1.-Xi)*SLOPE
        endif

        SSIG_TOT(J,3) = SIG_TOT
        SSIG(J,3)     = SIG_inel
        SSIG_SD1(J,3) = SIGDIF(1)
        SSIG_SD2(J,3) = SIGDIF(2)
        SSIG_DD(J,3)  = SIG_df(3)
        SSIG_B(J,3)   = B_EL
        SSIG_RHO(J,3) = RHO

ctp        WRITE(6,'(3X,I2,1P,E12.3,0P,4F8.2,3F8.3)')
ctp     &    JINT,SQS,PTmin,SIG_tot,SIG_inel,B_el,RHO,PS,PH

      ENDDO


      RETURN
      END


      SUBROUTINE INI_WRITE (LUN)
C-----------------------------------------------------------------------
C   This subroutine prints on unit LUN
C   a table of the cross sections  used in the program
C   and of the average number of hard interactions, and the average
C   number of wounded nucleons in a hadron-air interaction
C-----------------------------------------------------------------------

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      DIMENSION PJ(2),PS(2),PW(2)
      SAVE

      DATA ATARG /14.514/

*      CALL PARAM_PRINT(LUN)
      WRITE (LUN, 10)
      WRITE (LUN, 15)
      WRITE (LUN, 16)
      WRITE (LUN, 18)
10    FORMAT(//,' Table of cross sections, and average number',
     &         ' of minijets and wounded nucleons ')
15    FORMAT('        [sqrt(s) in GeV, cross sections in mbarn]. ')
16    FORMAT(' sqrt(s) sig(pp) sig(pA) <n_s> <n_j> <n_w>',
     &    ' sig(pip) sig(piA) <n_s> <n_j> <n_w>')
18    FORMAT(1X,77('-') )
      DO J=1,61,1
         SQS = 10.**(ASQSMIN + DASQS*FLOAT(J-1))

         DO K=1,2

           PW(K) = ATARG*SSIG(J,K)/SSIGN(J,K)

           PJ(K) = 0.
           PS(K) = 0.
           DO NS=0,NS_max
             DO NJ=0,NH_max
               IF(NJ.GT.0) THEN
                 PROB = PJETC(NS,NJ,J,K) - PJETC(NS,NJ-1,J,K)
               ELSE IF(NS.GT.0) THEN
                 PROB = PJETC(NS,NJ,J,K) - PJETC(NS-1,NH_max,J,K)
               ELSE
                 PROB = 0.
               ENDIF
               PJ(K) = PJ(K)+FLOAT(NJ)*PROB
               PS(K) = PS(K)+FLOAT(NS)*PROB
             ENDDO
           ENDDO

         ENDDO

         WRITE(LUN,20) SQS,SSIG(J,1),SSIGN(J,1),PS(1),PJ(1),PW(1)
     &                      ,SSIG(J,2),SSIGN(J,2),PS(2),PJ(2),PW(2)

      ENDDO
      WRITE (LUN, 18)
20    FORMAT (1X,E8.2, 2(2F7.1,1X,3F6.2,1X))

      RETURN
      END

C*************************************************************************
C=========================================================================
C. UTILITIES ROUTINES
C=========================================================================
C***********************************************************************

C=======================================================================
C. Code for the wounded nucleon distribution
C=======================================================================


      SUBROUTINE SIB_START_EV (SQS, L, IA, NW, JDIF)
C-----------------------------------------------------------------------
C...Beginning of a SIBYLL interaction
C.
C.  INPUT : SQS = c.m.s. energy (GeV)
C.          L = 1:proton, 2:charged pion
C.          IA = mass of target nucleon
C.
C.  OUTPUT: NW    = number of wounded nucleons
C.          JDIF(JW)  = diffraction code    !!!! changed to field !!!!
C.                  (0 : non-diffractive interaction)
C.                  (1 : forward diffraction)
C.                  (2 : backward diffraction)
C.                  (3 : double diffraction)
C.
C-----------------------------------------------------------------------

      PARAMETER (NW_max = 20)
      DIMENSION JDIF(NW_max)
      COMMON /S_CNCM0/ B, BMAX, NTRY, NA

      DIMENSION SIGDIF(3)
      SAVE

C...sample number of wounded nucleons
      CALL SIB_SIGMA_HP(L,SQS,SIGT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
      IF (IA .GT. 1)  THEN
         CALL INT_H_NUC (IA, SIGT, SLOPE, RHO)
      ELSE
         NA = 1
      ENDIF
      NW = NA

C...new treatment of diffraction
      PF = SIGDIF(1)/SIGINEL
      PB = SIGDIF(2)/SIGINEL
      PD = SIGDIF(3)/SIGINEL
      P0 = 1.-PF-PB-PD
      P1 = P0 + PF
      P2 = P1 + PB
      DO K=1, NW
        R = S_RNDM(0)
        IF (R .LT. P0)  THEN
          JDIF(K) = 0
        ELSE IF (R .LT. P1)  THEN
          JDIF(K) = 1
        ELSE IF (R .LT. P2)  THEN
          JDIF(K) = 2
        ELSE
          JDIF(K) = 3
        ENDIF
      ENDDO

      RETURN
      END



      SUBROUTINE INT_H_NUC (IA, SIGT, SLOPE, RHO)
C...Compute with a montecarlo method the "multiple interaction structure"
C.  of an hadron-nucleus collision.
C.
C.
C.  INPUT : IA               = mass of target nucleus
C.          SIGT (mbarn)     = total hp cross section
C.          SLOPE (GeV**-2)  = slope of hp elastic scattering
C.          RHO              = real/imaginary part of forward elastic
C.                             scattering amplitude
C.
C.  OUTPUT : in COMMON block /CNCMS0/
C.           B = impact parameter (fm)
C.           BMAX = maximum impact parameter for generation
C.           NTRY = number of "trials" before one interaction
C.           NA = number of wounded nucleons in A
C. Author : P.Lipari  (may 1993)
C---------------------------------------------------------------------------
      PARAMETER (IAMAX=56)
      COMMON /S_CNCM0/ B, BMAX, NTRY, NA
      DIMENSION XA(IAMAX), YA(IAMAX)
      SAVE
      DATA PI /3.1415926/
      DATA CMBARN /0.389385/

      CC = SIGT/(4.*PI*SLOPE*CMBARN)
      DEN = 2.*SLOPE*CMBARN*0.1
      BMAX = 10.                             ! fm
      NTRY = 0
      CALL NUC_CONF (IA, XA, YA)
1000  B = BMAX*SQRT(S_RNDM(0))
      PHI = 2.*PI*S_RNDM(0)
      BX = B*COS(PHI)
      BY = B*SIN(PHI)
      NTRY = NTRY+1
      NA = 0
      DO JA=1,IA
         S = (XA(JA)-BX)**2 + (YA(JA)-BY)**2
         F = EXP(-S/DEN)
         PEL = CC*CC*(1.+RHO*RHO)*F*F
         PINEL  = 2.*CC*F-PEL
         R = S_RNDM(0)
         IF (R .LT. PINEL)  THEN
            NA = NA + 1
         ENDIF
      ENDDO
      IF (NA .EQ. 0)  GOTO 1000
      RETURN
      END


C==========================================================================
C. Cross sections
C==========================================================================


      SUBROUTINE SIG_AIR_INI
C-----------------------------------------------------------------------
C...Initialize the cross section and interaction lengths on air
C.  (this version initializes p-air, pi-air, and K-air cross sections)
C-----------------------------------------------------------------------

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      COMMON /S_CCSIG2/ SSIG_TOT(61,3),SSIG_SD1(61,3),SSIG_SD2(61,3),
     &    SSIG_DD(61,3),SSIG_B(61,3),SSIG_RHO(61,3)

      DIMENSION SIGDIF(3)
      SAVE

      DATA AVOG /6.0221367E-04/

      ATARGET = 14.514

C...particle loop (p, pi, K)
      DO K=1,3
        DO J=1,NSQS

           ASQS = ASQSMIN + DASQS*FLOAT(J-1)
           SQS = 10.**ASQS

           CALL SIB_SIGMA_HP(K,SQS,SIGT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
           CALL SIG_H_AIR(SIGT, SLOPE, RHO, SSIGT, SSIGEL, SSIGQE)

C  particle production cross section
           SSIGN(J,K) = SSIGT-SSIGQE
           ALINT(J,K) = 1./(AVOG*SSIGn(j,K)/ATARGET)

        ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE SIB_SIGMA_HP(L,SQS,SIGT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
C-----------------------------------------------------------------------
C     Hadron-proton cross sections, taken from interpolation table
C     calculated by SIBYLL_INI
C
C     input:       L     1      proton-proton
C                        2      pi-proton
C                        3      K-proton
C                  SQS   sqrt(s)
C
C     output:      SIGT       total cross section (mb)
C                  SIGEL      elastic cross section (mb)
C                  SIGINEL    inelastic cross section (mb)
C                  SIGDIF     diffraction dissociation CS (mb)
C                  SLOPE      elastic slope parameter (GeV^-2)
C                  RHO        real/imaginary part of forward amplitude
C-----------------------------------------------------------------------

      DIMENSION SIGDIF(3)

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      COMMON /S_CCSIG2/ SSIG_TOT(61,3),SSIG_SD1(61,3),SSIG_SD2(61,3),
     &    SSIG_DD(61,3),SSIG_B(61,3),SSIG_RHO(61,3)
      SAVE

      IF(NSQS.LE.0) THEN
        WRITE(6,'(//,1X,A)')
     &    'SIB_SIGMA_HP: interpolation table not initialized.'
        STOP
      ENDIF

      AL = LOG10(SQS)
      J1 = (AL - 1.)*10. + 1
      if((j1.lt.1).or.(j1.ge.NSQS)) then
c        write (6,'(1x,a,i3,1p,e12.3)')
c     &    'SIB_SIGMA_HP: energy out of range ',L,sqs
        J1 = min(J1,NSQS-1)
        J1 = max(J1,1)
      endif
      T = (AL-1.)*10. - FLOAT(J1-1)
      SIGT    = SSIG_TOT(J1,L)*(1.-T) + SSIG_TOT(J1+1,L)*T
      SIGINEL = SSIG(J1,L)*(1.-T) + SSIG(J1+1,L)*T
      SIGEL   = SIGT-SIGINEL
      SIGDIF(1) = SSIG_SD1(J1,L)*(1.-T) + SSIG_SD1(J1+1,L)*T
      SIGDIF(2) = SSIG_SD2(J1,L)*(1.-T) + SSIG_SD2(J1+1,L)*T
      SIGDIF(3) = SSIG_DD(J1,L)*(1.-T) + SSIG_DD(J1+1,L)*T
      SLOPE   = SSIG_B(J1,L) *(1.-T) + SSIG_B(J1+1,L)*T
      RHO     = SSIG_RHO(J1,L) *(1.-T) + SSIG_RHO(J1+1,L)*T

      RETURN
      END


      SUBROUTINE SIB_SIGMA_HAIR (L,SQS,SIGprod)
C-----------------------------------------------------------------------
C     Hadron-air cross sections, taken from interpolation table
C     calculated by SIBYLL_INI
C
C     input:       L     1      proton-air
C                        2      pi-air
C                        3      K-air
C                  SQS   sqrt(s)
C
C     output:      SIGprod    particle production cross section (mb)
C-----------------------------------------------------------------------

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CCSIG/ SSIG(61,3), PJETC(0:NS_max,0:NH_max,61,2),
     &    SSIGN(61,3), ALINT(61,3), ASQSMIN, ASQSMAX, DASQS, NSQS
      SAVE

      IF(NSQS.LE.0) THEN
        WRITE(6,'(//,1X,A)')
     &    'SIB_SIGMA_HAIR: interpolation table not initialized.'
        STOP
      ENDIF

      AL = LOG10(SQS)
      J1 = (AL - 1.)*10. + 1
      if((j1.lt.1).or.(j1.ge.NSQS)) then
        write (6,'(1x,a,i3,1p,e12.3)')
     &    'SIB_SIGMA_HAIR: energy out of range ',L,sqs
        J1 = min(J1,NSQS-1)
        J1 = max(J1,1)
      endif
      T = (AL-1.)*10. - FLOAT(J1-1)
      SIGprod = SSIGN(J1,L)*(1.-T) + SSIGN(J1+1,L)*T

      RETURN
      END


      SUBROUTINE SIB_HADCSL(L,ECM,SIGTOT,SIGEL,SIGINEL,SIGDIF,SLOPE,RHO)
C-----------------------------------------------------------------------
C     low-energy cross section parametrizations (target always proton)
C
C     input:   L           beam particle: (1 - proton,
C                                          2 - pion,
C                                          3 - kaon)
C                          target is always proton
C              ECM         c.m. energy (GeV)
C
C     output:  SIGTOT      total cross section (mb)
C              SIGEL       elastic cross section (mb)
C              SIGDIF      diffractive cross section (sd-1,sd-2,dd, mb)
C              SLOPE       forward elastic slope (GeV**-2)
C              RHO         real/imaginary part of elastic amplitude
C-----------------------------------------------------------------------
      DIMENSION SIGDIF(3)

      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      SAVE

C  proton-proton cross section as reference
      CALL SIB_HADCS1(1,ECM,SIGTOT,SIGEL,SIGINEL,SLOPE,RHO)

C  parametrization for diffraction
      Xi_min = 1.5/(ECM*ECM)
      Xi_max = PAR(13)
      SIGeff = SIGEL
      call SIB_HADCS2(ECM,Xi_min,Xi_max,SIGeff,SIGDIF)

      if(L.eq.1) return

C  regge motivated rescaling of diffraction dissociation
      sigtot_pp = SIGTOT
      sigel_pp  = SIGEL
      slope_pp  = SLOPE
      CALL SIB_HADCS1(L,ECM,SIGTOT,SIGEL,SIGINEL,SLOPE,RHO)
      SIGDIF(1) = slope_pp/SLOPE*SIGTOT/sigtot_pp*SIGDIF(1)
      SIGDIF(2) = slope_pp/SLOPE*SIGEL/sigel_pp*SIGDIF(2)
      SIGDIF(3) = SIGTOT/sigtot_pp*SIGDIF(3)

      RETURN
      END


      SUBROUTINE SIB_HADCS1(L,ECM,SIGTOT,SIGEL,SIGINEL,SLOPE,RHO)
C-----------------------------------------------------------------------
C     low-energy cross section parametrizations
C
C     input:   L           beam particle: (1 - proton,
C                                          2 - pion,
C                                          3 - kaon)
C                          target is always proton
C              ECM         c.m. energy (GeV)
C
C     output:  SIGTOT      total cross section (mb)
C              SIGEL       elastic cross section (mb)
C              SIGDIF      diffractive cross section (sd-1,sd-2,dd, mb)
C              SLOPE       forward elastic slope (GeV**-2)
C              RHO         real/imaginary part of elastic amplitude
C
C     comments:
C     - low-energy data interpolation uses PDG fits from 1992
C     - slopes from ???, new fit to pp data
C     - high-energy extrapolation by Donnachie-Landshoff like fit made
C       by PDG 1996
C     - analytic extension of amplitude to calculate rho
C-----------------------------------------------------------------------

      DIMENSION TPDG92(7,2,6),TPDG96(9,6),BURQ83(3,6),XMA(6)
      SAVE

      DATA TPDG92  /
     &  3.D0, 2100.D0, 48.D0, 0.D0, 1.D0, 0.522D0, -4.51D0,
     &  3.D0, 2100.D0, 11.9D0, 26.9D0, -1.21D0, 0.169D0, -1.85D0,
     &  5.D0, 2100.D0, 38.4D0, 77.6D0, -0.64D0, 0.26D0, -1.2D0,
     &  5.D0, 2100.D0, 10.2D0, 52.7D0, -1.16D0, 0.125D0, -1.28D0,
     &  4.D0, 340.D0,  16.4D0, 19.3D0, -0.42D0, 0.19D0, 0.D0,
     &  4.D0, 340.D0,  0.D0, 11.4D0, -0.4D0, 0.079D0, 0.D0,
     &  2.5D0, 370.D0, 33.D0, 14.D0, -1.36D0, 0.456D0, -4.03D0,
     &  2.5D0, 370.D0, 1.76D0, 11.2D0, -0.64D0, 0.043D0, 0.D0,
     &  2.D0, 310.D0,  18.1D0, 0.D0, 1.D0, 0.26D0, -1.D0,
     &  2.D0, 310.D0,  5.D0, 8.1D0, -1.8D0, 0.16D0, -1.3D0,
     &  3.D0, 310.D0,  32.1D0, 0.D0, 1.D0, 0.66D0, -5.6D0,
     &  3.D0, 310.D0,  7.3D0, 0.D0, 1.D0, 0.29D0, -2.4D0  /

      DATA TPDG96  /
     &  50.D0, 22.D0,0.079D0,0.25D0,0.D0,
     &         77.15D0,-21.05D0,0.46D0,0.9D0,
     &  50.D0, 22.D0,0.079D0,0.25D0,0.D0,
     &         77.15D0,21.05D0,0.46D0,0.9D0,
     &  10.D0, 13.70,0.079D0,0.25D0,0.D0,
     &         31.85D0,-4.05D0,0.45D0,0.9D0,
     &  10.D0, 13.70,0.079D0,0.25D0,0.D0,
     &         31.85D0,4.05D0,0.45D0,0.9D0,
     &  10.D0, 12.20,0.079D0,0.25D0,0.D0,
     &         17.35D0,-9.05D0,0.50D0,0.9D0,
     &  10.D0, 12.20,0.079D0,0.25D0,0.D0,
     &         17.35D0,9.05D0,0.50D0,0.9D0  /

      DATA BURQ83 /
     &  8.557D0,  0.00D0, 0.574D0,
     &  11.13D0,  7.23D0, 0.30D0,
     &  9.11D0,  -0.73D0, 0.28D0,
     &  9.11D0,   0.65D0, 0.28D0,
     &  8.55D0,  -5.98D0, 0.28D0,
     &  8.55D0,   1.60D0, 0.28D0  /

      DATA XMA / 2*0.93956563, 2*0.13956995, 2*0.493677 /
      DATA GEV2MB /0.389365/
      DATA PI /3.14159265358979/

C  find index
      IF(L.eq.1) THEN
        K = 1                            ! p p
      ELSE IF(L.eq.2) THEN
        K = 3                            ! pi+ p
*       K = 4                            ! pi- p
      ELSE IF(L.eq.3) THEN
        K = 5                            ! K+ p
*       K = 6                            ! K- p
      ELSE
        GOTO 100
      ENDIF

C  calculate lab momentum
      SS = ECM**2
      E1 = (SS-XMA(1)**2-XMA(K)**2)/(2.*XMA(1))
      PL = SQRT((E1-XMA(K))*(E1+XMA(K)))
      PLL = LOG(PL)

C  check against lower limit
      IF(ECM.LE.XMA(1)+XMA(K)) GOTO 200

      XP  = TPDG96(2,K)*SS**TPDG96(3,K)
      YP  = TPDG96(6,K)/SS**TPDG96(8,K)
      YM  = TPDG96(7,K)/SS**TPDG96(8,K)

      PHR = TAN(PI/2.*(1.-TPDG96(8,K)))
      PHP = TAN(PI/2.*(1.+TPDG96(3,K)))
      RHO = (-YP/PHR + YM*PHR - XP/PHP)/(YP+YM+XP)

      SLOPE = BURQ83(1,K)+BURQ83(2,K)/SQRT(PL)+BURQ83(3,K)*PLL

C  select energy range and interpolation method
      IF(PL.LT.TPDG96(1,K)) THEN
        SIGTOT = TPDG92(3,1,K)+TPDG92(4,1,K)*PL**TPDG92(5,1,K)
     &          + TPDG92(6,1,K)*PLL**2+TPDG92(7,1,K)*PLL
        SIGEL  = TPDG92(3,2,K)+TPDG92(4,2,K)*PL**TPDG92(5,2,K)
     &          + TPDG92(6,2,K)*PLL**2+TPDG92(7,2,K)*PLL
      ELSE IF(PL.LT.TPDG92(2,1,K)) THEN
        SIGTO1 = TPDG92(3,1,K)+TPDG92(4,1,K)*PL**TPDG92(5,1,K)
     &          + TPDG92(6,1,K)*PLL**2+TPDG92(7,1,K)*PLL
        SIGEL1 = TPDG92(3,2,K)+TPDG92(4,2,K)*PL**TPDG92(5,2,K)
     &          + TPDG92(6,2,K)*PLL**2+TPDG92(7,2,K)*PLL
        SIGTO2 = YP+YM+XP
        SIGEL2 = SIGTO2**2/(16.*PI*SLOPE*GEV2MB)*(1.+RHO**2)
        X2 = LOG(PL/TPDG96(1,K))/LOG(TPDG92(2,1,K)/TPDG96(1,K))
        X1 = 1. - X2
        SIGTOT = SIGTO2*X2 + SIGTO1*X1
        SIGEL  = SIGEL2*X2 + SIGEL1*X1
      ELSE
        SIGTOT = YP+YM+XP
        SIGEL  = SIGTOT**2/(16.*PI*SLOPE*GEV2MB)*(1.+RHO**2)
      ENDIF
      SIGINEL = SIGTOT-SIGEL

      RETURN

 100  CONTINUE
        WRITE(6,'(1X,2A,2I7)') 'SIB_HADCSL: ',
     &    'invalid beam particle: ',L
        RETURN

 200  CONTINUE
        WRITE(6,'(1X,2A,1P,E12.4)') 'SIB_HADCSL: ',
     &    'energy too small (Ecm): ',ECM

      RETURN
      END


      SUBROUTINE SIB_HADCS2(SQS,Xi_min,Xi_max,SIGeff,SIGDIF)
C-----------------------------------------------------------------------
C   cross section for diffraction dissociation
C
C   - single diffraction dissociation:
C     Goulianos' parametrization (Ref: PL B358 (1995) 379)
C   - double diffration dissociation: simple scaling model using
C     single diff. cross section
C
C     in addition rescaling for different particles is applied using
C     internal rescaling tables (not implemented yet)
C
C     input:     SQS         c.m. energy (GeV)
C                Xi_min      min. diff mass (squared) = Xi_min*SQS**2
C                Xi_max      max. diff mass (squared) = Xi_max*SQS**2
C                SIGeff      effective cross section for DD scaling
C
C     output:    sig_sd1     cross section for diss. of particle 1 (mb)
C                sig_sd2     cross section for diss. of particle 2 (mb)
C                sig_dd      cross section for diss. of both particles
C-----------------------------------------------------------------------

      DIMENSION SIGDIF(3)
      DOUBLE PRECISION Xpos1(96),Xwgh1(96),Xpos2(96),Xwgh2(96)
      DOUBLE PRECISION xil,xiu,tl,tu
      SAVE

C  model parameters
      DATA delta    / 0.104 /
      DATA alphap   / 0.25 /
      DATA beta0    / 6.56 /
      DATA gpom0    / 1.21 /
      DATA xm_p     / 0.938 /
      DATA x_rad2   / 0.71 /

C  integration precision
      DATA Ngau1    / 32 /
      DATA Ngau2    / 32 /

      DATA PI /3.14159265358979/
      DATA GEV2MB /0.389365/


      SIGDIF(1) = 0.
      SIGDIF(2) = 0.
      SIGDIF(3) = 0.

      XIL = LOG(Xi_min)
      XIU = LOG(Xi_max)

      if(XIL.ge.XIU) return

      SS = SQS*SQS
      xm4_p2 = 4.*xm_p**2
      fac = beta0**2/(16.*PI)

      t1 = -5.
      t2 = 0.
      tl = x_rad2/3./(1.-t1/x_rad2)**3
      tu = x_rad2/3./(1.-t2/x_rad2)**3

C  flux renormalization and cross section for pp/ppbar case

      Xnorm  = 0.

      xil = log(1.5/SS)
      xiu = log(0.1)

      IF(xiu.LE.xil) goto 1000

      CALL SIB_GAUSET(xil,xiu,Ngau1,xpos1,xwgh1)
      CALL SIB_GAUSET(tl,tu,Ngau2,xpos2,xwgh2)

      do i1=1,Ngau1

        xi = exp(xpos1(i1))
        w_xi = Xwgh1(i1)

        do i2=1,Ngau2

          tt = x_rad2-x_rad2*(x_rad2/(3.*xpos2(i2)))**(1./3.)

          alpha_t =  1.+delta+alphap*tt
          f2_t = ((xm4_p2-2.8*tt)/(xm4_p2-tt))**2

          Xnorm = Xnorm
     &      + f2_t*xi**(2.-2.*alpha_t)*Xwgh2(i2)*w_xi

        enddo
      enddo

      Xnorm = Xnorm*fac

 1000 continue

      XIL = LOG(Xi_min)
      XIU = LOG(Xi_max)

      T1 = -5.
      T2 = 0.

      TL = x_rad2/3./(1.-t1/x_rad2)**3
      TU = x_rad2/3./(1.-t2/x_rad2)**3

C  single diffraction diss. cross section

      CSdiff = 0.

      CALL SIB_GAUSET(XIL,XIU,NGAU1,XPOS1,XWGH1)
      CALL SIB_GAUSET(TL,TU,NGAU2,XPOS2,XWGH2)

      do i1=1,Ngau1

        xi = exp(xpos1(i1))
        w_xi = Xwgh1(i1)*beta0*gpom0*(xi*ss)**delta

        do i2=1,Ngau2

          tt = x_rad2-x_rad2*(x_rad2/(3.*xpos2(i2)))**(1./3.)

          alpha_t =  1.+delta+alphap*tt
          f2_t = ((xm4_p2-2.8*tt)/(xm4_p2-tt))**2

          CSdiff = CSdiff
     &      + f2_t*xi**(2.-2.*alpha_t)*Xwgh2(i2)*w_xi

        enddo
      enddo

      CSdiff = CSdiff*fac*GEV2MB/MAX(1.,Xnorm)

*     write(6,'(1x,1p,4e14.3)')
*    &  sqrt(SS),Xnorm,2.*CSdiff*MAX(1.,Xnorm),2.*CSdiff

      SIGDIF(1) = CSdiff
      SIGDIF(2) = CSdiff

C  double diff. dissociation from simple probability consideration
*     Pdiff = 0.5-sqrt(0.25-CSdiff/SIGeff)
      Pdiff = CSdiff/SIGeff
      SIGDIF(3) = Pdiff*Pdiff*SIGeff

      RETURN
      END


      SUBROUTINE DECSIB
C-----------------------------------------------------------------------
C...Decay all unstable particle in Sibyll
C.  decayed particle have the code increased by 10000
C
C   changed to allow for multiple calls to DECSIB in one event
C-----------------------------------------------------------------------
      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_PLIST1/ LLIST1(8000)
      COMMON /S_MASS1/ AM(49), AM2(49)
      DIMENSION P0(5), LL(10), PD(10,5)
      SAVE

      NN = 1
      DO J=1,NP
         LLIST1(J) = 0
      ENDDO
      DO WHILE (NN .LE. NP)
         L= LLIST(NN)
         LA = IABS(L)
         if(LA.lt.50) then
           IF (IDB(LA) .GT. 0)  THEN
              DO K=1,5
                P0(K) = P(NN,K)
              ENDDO
              CALL DECPAR (L,P0,ND,LL,PD)
              LLIST(NN) = LLIST(NN)+ISIGN(10000,LLIST(NN))
              DO J=1,ND
                NP = NP+1
                if(NP.gt.8000) then
                  write(6,'(1x,a,2i8)')
     &              'DECSIB: no space left in S_PLIST (NP,ND):',NP,ND
                  NP = NP-1
                  return
                endif
                DO K=1,5
                  P(NP,K) = PD(J,K)
                ENDDO
                LLIST(NP)=LL(J)
                LLIST1(NP)=NN
              ENDDO
           ENDIF
         endif
         NN = NN+1
      ENDDO

      RETURN
      END



      SUBROUTINE DECPAR (LA,P0,ND,LL,P)
C-----------------------------------------------------------------------
C...This subroutine generates the decay of a particle
C.  with ID = LA, and 5-momentum P0(1:5)
C.  into ND particles of 5-momenta P(j,1:5) (j=1:ND)
C.
C.  If the initial particle code is LA=0
C.  then ND and LL(1:ND) are considered as  input and
C.  the routine generates a phase space decay into ND
C.  particles of codes LL(1:nd)
C.
C.  june 1992
C.  This version  contains the decay of polarized muons
C.  The muon codes are  L =  4 : mu+ R
C.                          -4 : mu+ L
C.                           5 : mu- L
C.                          -5 : mu- R
C-----------------------------------------------------------------------
      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)
      COMMON /S_MASS1/ AM(49), AM2(49)
      DIMENSION P0(5), LL(10), P(10,5)
      DIMENSION PV(10,5), RORD(10), UE(3),BE(3), FACN(3:10)
      SAVE
      DATA FACN /2.,5.,15.,60.,250.,1500.,12000.,120000./
      DATA PI /3.1415926/

C...c.m.s. Momentum in two particle decays
cdh  corrected 25.4.02
      PAWT(A,B,C) = SQRT((A**2-(B+C)**2+1.e-5)*(A**2-(B-C)**2))/(2.*A)

C...Phase space decay into the particles in the list
      IF (LA .EQ. 0)  THEN
          MAT = 0
          MBST = 0
          PS = 0.
          DO J=1,ND
CDH          following statements corrected by D.H. dec 20.,1995
             P (J,5) = AM(IABS(LL(J)))
             PV(J,5) = AM(IABS(LL(J)))
             PS = PS+P(J,5)
          ENDDO
          DO J=1,4
             PV(1,J) = P0(J)
          ENDDO
          PV(1,5) = P0(5)
          GOTO 140
      ENDIF

C...Choose decay channel
      L = IABS(LA)
      ND=0
      IDC = IDB(L)-1
      IF (IDC+1 .LE.0)  RETURN
      RBR = S_RNDM(0)
110   IDC=IDC+1
      IF(RBR.GT.CBR(IDC))  GOTO 110

      KD =6*(IDC-1)+1
      ND = KDEC(KD)
      MAT= KDEC(KD+1)
      MBST=0
      IF (MAT .GT.0 .AND. P0(4) .GT. 20*P0(5)) MBST=1
      IF (MAT .GT.0 .AND. MBST .EQ. 0)
     +        BETA = SQRT(P0(1)**2+P0(2)**2+P0(3)**2)/P0(4)
      PS = 0.
      DO J=1,ND
         LL(J) = KDEC(KD+1+J)
         P(J,5)  = AM(LL(J))
         PV(J,5) = AM(LL(J))
         PS = PS + P(J,5)
      ENDDO
      DO J=1,4
         PV(1,J) = 0.
         IF (MBST .EQ. 0)  PV(1,J) = P0(J)
      ENDDO
      IF (MBST .EQ. 1)  PV(1,4) = P0(5)
      PV(1,5) = P0(5)

140   IF (ND .EQ. 2) GOTO 280

      IF (ND .EQ. 1)  THEN
         DO J=1,4
            P(1,J) = P0(J)
         ENDDO
         RETURN
      ENDIF

C...Calculate maximum weight for ND-particle decay
      WWTMAX = 1./FACN(ND)
      PMAX=PV(1,5)-PS+P(ND,5)
      PMIN=0.
      DO IL=ND-1,1,-1
         PMAX = PMAX+P(IL,5)
         PMIN = PMIN+P(IL+1,5)
         WWTMAX = WWTMAX*PAWT(PMAX,PMIN,P(IL,5))
      ENDDO

C...generation of the masses, compute weight, if rejected try again
240   RORD(1) = 1.
      DO 260 IL1=2,ND-1
      RSAV = S_RNDM(il1)
      DO 250 IL2=IL1-1,1,-1
      IF(RSAV.LE.RORD(IL2))   GOTO 260
250     RORD(IL2+1)=RORD(IL2)
260     RORD(IL2+1)=RSAV
      RORD(ND) = 0.
      WT = 1.
      DO 270 IL=ND-1,1,-1
      PV(IL,5)=PV(IL+1,5)+P(IL,5)+(RORD(IL)-RORD(IL+1))*(PV(1,5)-PS)
270   WT=WT*PAWT(PV(IL,5),PV(IL+1,5),P(IL,5))
      IF (WT.LT.S_RNDM(0)*WWTMAX)   GOTO 240

C...Perform two particle decays in respective cm frame
280   DO 300 IL=1,ND-1
      PA=PAWT(PV(IL,5),PV(IL+1,5),P(IL,5))
      UE(3)=2.*S_RNDM(il)-1.
      PHI=2.*PI*S_RNDM(il)
      UT = SQRT(1.-UE(3)**2)
      UE(1) = UT*COS(PHI)
      UE(2) = UT*SIN(PHI)
      DO 290 J=1,3
      P(IL,J)=PA*UE(J)
290   PV(IL+1,J)=-PA*UE(J)
      P(IL,4)=SQRT(PA**2+P(IL,5)**2)
300   PV(IL+1,4)=SQRT(PA**2+PV(IL+1,5)**2)

C...Lorentz transform decay products to lab frame
      DO 310 J=1,4
310   P(ND,J)=PV(ND,J)
      DO 340 IL=ND-1,1,-1
      DO 320 J=1,3
320   BE(J)=PV(IL,J)/PV(IL,4)
      GA=PV(IL,4)/PV(IL,5)
      DO 340 I=IL,ND
      BEP = BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
      DO 330 J=1,3
330   P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)
340   P(I,4)=GA*(P(I,4)+BEP)

C...Weak decays
      IF (MAT .EQ. 1)  THEN
         F1=P(2,4)*P(3,4)-P(2,1)*P(3,1)-P(2,2)*P(3,2)-P(2,3)*P(3,3)
         IF (MBST.EQ.1)  THEN
C        WT = P0(5)*P(1,4)*F1
         WT = P0(5)*(P(1,4)+FLOAT(LA/L)*P(1,3))*F1
       ENDIF
         IF (MBST.EQ.0)  THEN
         WT=F1*(P(1,4)*P0(4)-P(1,1)*P0(1)-P(1,2)*P0(2)-P(1,3)*P0(3))
         WT= WT-FLOAT(LA/L)*(P0(4)*BETA*P(1,4)-P0(4)*P(1,3))*F1
       ENDIF
         WTMAX = P0(5)**4/8.
         IF(WT.LT.S_RNDM(0)*WTMAX)   GOTO 240
      ENDIF

C...Boost back for rapidly moving particle
      IF (MBST .EQ. 1)   THEN
         DO 440 J=1,3
440      BE(J)=P0(J)/P0(4)
         GA= P0(4)/P0(5)
         DO 460 I=1,ND
         BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
         DO 450 J=1,3
450         P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)
460         P(I,4)=GA*(P(I,4)+BEP)
      ENDIF

C...labels for antiparticle decay
      IF (LA .LT. 0 .AND. L .GT. 18)  THEN
           DO J=1,ND
            LL(J) = LBARP(LL(J))
         ENDDO
      ENDIF

      RETURN
      END



      BLOCK DATA DATDEC
C-----------------------------------------------------------------------
C...initialization of SIBYLL particle data
C-----------------------------------------------------------------------

      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)
      COMMON /S_MASS1/ AM(49), AM2(49)
      COMMON /S_CHP/ ICHP(49), ISTR(49), IBAR(49)
      COMMON /S_CNAM/ NAMP (0:49)
      CHARACTER NAMP*6
      SAVE
      DATA CBR /3*1.,0.,1.,1.,0.6351,0.8468,0.9027,0.9200,0.9518,1.,
     +   0.6351,0.8468,0.9027,0.9200,0.9518,1.,0.2160,0.3398,0.4748,
     +   0.6098,0.8049,1.,0.6861,1.,3*0.,0.5,1.,0.5,1.,
     +   0.3890,0.7080,0.9440,0.9930,1.,0.,0.4420,0.6470,0.9470,0.9770,
     +   0.9990,4*1.,0.6670,1.,9*0.,0.6670,1.,0.6670,1.,0.6670,1.,
     +   0.8880,0.9730,1.,0.4950,0.8390,0.9870,1.,0.5160,5*1.,0.6410,1.,
     +   1.,0.67,1.,0.33,1.,1.,0.88,0.94,1.,0.88,0.94,1.,0.88,0.94,1.,
     +   0.33,1.,0.67,1.,0.678,0.914,1./
      DATA AM / 0.,2*0.511E-3, 2*0.10566, 0.13497, 2*0.13957,
     +   2*0.49365, 2*0.49767, 0.93827, 0.93957, 4*0.,0.93827,
     +   0.93957, 2*0.49767, 0.54880,0.95750,2*0.76830,0.76860,
     +   2*0.89183,2*0.89610,0.78195,1.01941,1.18937,1.19255,
     +   1.19743,1.31490,1.32132,1.11563,1.23100,1.23500,
     +   1.23400,1.23300,1.38280,1.38370,1.38720,
     +   1.53180,1.53500,1.67243 /
      DATA AM2 /0.,2*2.61121E-07,2*0.011164,0.018217,0.019480,
     + 0.019480,0.243690,0.243690,0.247675,0.247675,0.880351,0.882792,
     + 0.000000,0.000000,0.000000,0.000000,0.880351,0.882792,0.247675,
     + 0.247675,0.301181,0.916806,0.590285,0.590285,0.590746,0.795361,
     + 0.795361,0.802995,0.802995,0.611446,1.039197,1.414601,1.422176,
     + 1.433839,1.728962,1.745887,1.244630,1.515361,1.525225,1.522765,
     + 1.520289,1.912136,1.914626,1.924324,2.346411,2.356225,2.797022/
      DATA IDB /
     +    0,0,0,1,2,3,5,6,7,13,19,25,8*0,30,32,34,40,46,47,48,49,60,62,
     +    64,66,69,73,75,76,77,78,79,81,82,84,86,87,90,93,96,98,100/
      DATA KDEC /
     + 3,1,15,2,18,0,3,1,16,3,17,0,2,0,1,1,8*0,2,0,4,17,0,0,2,0,5,18,0,
     + 0,2,0,4,17,0,0,2,0,7,6,0,0,3,0,7,7,8,0,3,0,7,6,6,0,3,1,17,4,6,0,
     + 3,1,15,2,6,0,2,0,5,18,0,0,2,0,8,6,0,0,3,0,8,8,7,0,3,0,8,6,6,0,3,
     + 1,18,5,6,0,3,1,16,3,6,0,3,0,6,6,6,0,3,0,7,8,6,0,3,1,18,5,7,0,3,
     + 1,17,4,8,0,3,1,16,3,7,0,3,1,15,2,8,0,2,0,7,8,0,0,2,0,6,6,20*0,1,
     + 0,11,3*0,1,0,12,0,0,0,1,0,11,0,0,0,1,0,12,0,0,0,2,0,1,1,0,0,3,0,
     + 6,6,6,0,3,0,7,8,6,0,3,0,1,7,8,0,3,0,1,3,2,7*0,3,0,7,8,23,0,3,0,6
     + ,6,23,0,2,0,1,27,0,0,2,0,1,32,0,0,2,0,1,1,0,0,3,0,6,6,6,0,2,0,7,
     + 6,0,0,2,0,8,6,0,0,2,0,7,8,0,0,2,0,21,7,0,0,2,0,9,6,0,0,54*0,2,0,
     + 22,8,0,0,2,0,10,6,0,0,2,0,9,8,0,0,2,0,21,6,0,0,2,0,10,7,0,0,
     + 2,0,22,6,0,0,3,0,7,8,6,0,2,0,1,6,0,0,2,0,7,8,0,0,2,0,9,10,0,
     + 0,2,0,11,12,0,0,3,0,7,
     + 8,6,0,2,0,1,23,0,0,2,0,13,6,0,0,2,0,14,7,0,0,2,0,39,1,0,0,2,
     + 0,14,8,0,0,2,0,39,6,0,0,2,0,39,8,0,0,2,0,13,8,0,0,2,0,
     + 14,6,0,0,2,0,13,7,0,0,2,0,13,6,
     + 0,0,2,0,14,7,0,0,2,0,13,8,0,0,2,0,14,6,0,0,2,0,14,8,0,0,2,0,
     + 39,7,0,0,2,0,34,6,0,0,2,0,35,7,0,0,2,0,39,6,0,0,2,0,34,8,0,0,
     + 2,0,36,7,0,0,2,0,39,8,0,0,2,
     + 0,35,8,0,0,2,0,36,6,0,0,2,0,37,6,0,0,2,0,38,7,0,0,2,0,
     + 37,8,0,0,2,0,38,6,0,0,2,0,39,10,0,0,2,0,37,8,0,0,2,0,38,6,0,0/
      DATA LBARP/1,3,2,5,4,6,8,7,10,9,11,12,-13,-14,16,15,18,17,13,14,
     +  22,21,23,24,26,25,27,29,28,31,30,32,33,-34,-35,-36,-37,-38,-39,
     +  -40,-41,-42,-43,-44,-45,-46,-47,-48,-49/
      DATA ICHP /0,1,-1,1,-1,0,1,-1,1,-1,0,0,1,0,4*0,-1,0,4*0,
     +    1,-1,0,1,-1,4*0,1,0,-1,0,-1,0,2,1,0,-1,1,0,-1,0,-1,-1/
      DATA ISTR /8*0,-1,+1,10,10,8*0,-1,+1,5*0,-1,+1,-1,+1,2*0,
     +           3*1,2*2,1,4*0,3*1,2*2,3 /
      DATA IBAR /12*0,2*1,4*0,2*-1,13*0,16*1/
      DATA NAMP /
     +     '     ','gam   ','e+','e-','mu+','mu-','pi0',
     +     'pi+','pi-','k+', 'k-', 'k0l','k0s',
     +     'p', 'n', 'nue', 'nueb', 'num', 'numb', 'pbar', 'nbar',
     +     'k0', 'k0b', 'eta', 'etap', 'rho+', 'rho-','rho0',
     +     'k*+','k*-','k*0','k*0b','omeg', 'phi', 'SIG+', 'SIG0',
     +     'SIG-','XI0','XI-','LAM','DELT++','DELT+','DELT0','DELT-',
     +     'SIG*+ ','SIG*0','SIG*-', 'XI*0', 'XI*-', 'OME*-'/
      END
C->
      SUBROUTINE DECPR (LUN)
C...Print on unit LUN the list of particles and decay channels
      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)
      COMMON /S_MASS1/ AM(49), AM2(49)
      COMMON /S_CNAM/ NAMP (0:49)
      CHARACTER*6 NAMP
      DIMENSION LL(3)
      SAVE

      DO L=1,49
         IDC = IDB(L)-1
         NC = 0
         WRITE (LUN,10) L,NAMP(L), AM(L)
         IF(IDC+1 .GT. 0)  THEN
            CB = 0.
110         IDC=IDC+1
            NC = NC+1
            CBOLD = CB
            CB = CBR(IDC)
            BR = CB-CBOLD
            KD = 6*(IDC-1)+1
            ND = KDEC(KD)
            MAT= KDEC(KD+1)
            DO J=1,ND
              LL(J) = KDEC(KD+1+J)
            ENDDO
            WRITE (LUN,15) NC,BR,ND,MAT, (NAMP(LL(J)),J=1,ND)
            IF (CB .LT. 1.)  GOTO 110
         ENDIF
      ENDDO
      RETURN
10    FORMAT(1X,I3,2X,A6,3X,F10.4)
15    FORMAT(5X,I2,2X,F9.4,I4,I4,2X,3(A6,2X))
      END



      SUBROUTINE DEC_DEBUG (L,P0, ND, LL, PD)
      COMMON /S_CNAM/ NAMP (0:49)
      CHARACTER*6 NAMP
      DIMENSION P0(5), LL(10), PD(10,5)
      SAVE

      ETOT = 0.
      DO J=1,ND
         ETOT = ETOT + PD(J,4)
      ENDDO
      WRITE(*,*)  NAMP(IABS(L)),' -> ', (NAMP(IABS(LL(J))),J=1,ND)
      WRITE(*,*)  ' Ei, Ef = ', P0(4), ETOT, ' L = ', L
      RETURN
      END



      SUBROUTINE SIB_SIG(Jint,SIB_SQS,SIB_PTmin,SIB_SIG_tot,
     &                   SIB_SIG_ine,SIB_diff,SIB_B_el,SIB_PJET)
C-----------------------------------------------------------------------
C
C...SIBYLL 2.1 cross sections
C
C   input parameter: SIB_SQS   c.m.s. energy (GeV)
C                    Jint      1 p-p cross sections
C                              2 pi-p cross sections
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)

      PARAMETER (NS_max = 20, NH_max = 50)
      REAL SIB_PJET(0:NS_max,0:NH_max)
      REAL SIB_SQS,SIB_PTmin,
     &     SIB_SIG_ine,SIB_SIG_tot,SIB_diff(3),SIB_B_el

      COMMON /S_CFLAFR/ PAR(20), IPAR(10)
      REAL PAR


      COMMON /SIGMAS/SQS,SIGTOT,SIGEL,SIGINE,
     &               SIGSD1(2),SIGSD2(2),SIGDD(2),
     &               SLOPE,SLOPEc,RHO,PROB(0:NS_max,0:NH_max),SIGSUM
C
      COMMON /PROFILE/XNUS2,XMUS2,XNUSPI2,
     &                XNUH2,XMUH2,XNUHPI2,
     &                ENHPP,ENHPIP,al1,be1,al2,be2
C
      COMMON /S_CHDCNV/ABR(2,400),ABP(2,400),ABH(2,400),DB,NB

      DIMENSION XI(20)
C
      DIMENSION SIG_BRN(3)
      DIMENSION SIG_dif_1(2),SIG_dif_2(2),SIG_dd(2)

      DIMENSION IHAR(2),SIGQCD(61,2)
      SAVE

      DATA (SIGQCD(K,1),K=    1,   61) /
     &8.4925E-02,1.8301E-01,3.4031E-01,5.7346E-01,9.0097E-01,1.3443E+00,
     &1.9284E+00,2.6830E+00,3.6426E+00,4.8474E+00,6.3440E+00,8.1867E+00,
     &1.0438E+01,1.3169E+01,1.6462E+01,2.0412E+01,2.5128E+01,3.0733E+01,
     &3.7368E+01,4.5196E+01,5.4401E+01,6.5191E+01,7.7807E+01,9.2520E+01,
     &1.0964E+02,1.2951E+02,1.5254E+02,1.7918E+02,2.0993E+02,2.4538E+02,
     &2.8618E+02,3.3307E+02,3.8689E+02,4.4859E+02,5.1923E+02,6.0003E+02,
     &6.9234E+02,7.9769E+02,9.1782E+02,1.0547E+03,1.2104E+03,1.3876E+03,
     &1.5888E+03,1.8174E+03,2.0767E+03,2.3707E+03,2.7038E+03,3.0810E+03,
     &3.5077E+03,3.9903E+03,4.5357E+03,5.1517E+03,5.8471E+03,6.6317E+03,
     &7.5163E+03,8.5134E+03,9.6365E+03,1.0901E+04,1.2324E+04,1.3925E+04,
     &1.5724E+04/

      DATA (SIGQCD(K,2),K=    1,   61) /
     &1.5891E-01,2.9085E-01,4.8190E-01,7.4586E-01,1.0985E+00,1.5582E+00,
     &2.1466E+00,2.8890E+00,3.8146E+00,4.9572E+00,6.3556E+00,8.0544E+00,
     &1.0104E+01,1.2564E+01,1.5500E+01,1.8987E+01,2.3112E+01,2.7972E+01,
     &3.3679E+01,4.0359E+01,4.8154E+01,5.7228E+01,6.7762E+01,7.9965E+01,
     &9.4071E+01,1.1035E+02,1.2909E+02,1.5063E+02,1.7536E+02,2.0370E+02,
     &2.3613E+02,2.7321E+02,3.1553E+02,3.6379E+02,4.1875E+02,4.8129E+02,
     &5.5238E+02,6.3311E+02,7.2470E+02,8.2854E+02,9.4614E+02,1.0793E+03,
     &1.2298E+03,1.3999E+03,1.5920E+03,1.8089E+03,2.0534E+03,2.3291E+03,
     &2.6396E+03,2.9892E+03,3.3825E+03,3.8248E+03,4.3219E+03,4.8803E+03,
     &5.5073E+03,6.2109E+03,7.0001E+03,7.8849E+03,8.8764E+03,9.9871E+03,
     &1.1231E+04/


      DATA CMBARN /0.389385/
      DATA PI /3.1415926/
      DATA INIT /0/

      IF(INIT.EQ.0) THEN
*        CALL HAR_INI
        CALL FACT_INI
        IHAR(1) = 0
        IHAR(2) = 0
        INIT = 1
      ENDIF

      ECM = SIB_SQS

      IF(JINT.EQ.1) THEN

        XI( 1) =  4.989E+01
        XI( 2) =  8.203E-05
        XI( 3) =  2.449E-02
        XI( 4) = -4.000E-01
        XI( 5) =  2.000E-01
        XI( 6) =  5.000E-01
        XI( 7) =  0.000E+00
        XI( 8) =  6.000E-01
        XI( 9) =  9.000E-02
        XI(10) =  1.000E+00
        XI(11) =  2.000E+00
        XI(12) =  3.175E+00
        XI(13) =  2.500E-01
        XI(14) =  5.400E-01
        XI(15) =  7.700E-01
        XI(16) = -8.800E-01
        XI(17) =  5.400E-01
        XI(18) =  5.000E-01
        XI(19) =  9.000E-01

      ELSE IF(JINT.EQ.2) THEN

        XI( 1) = 2.653E+01
        XI( 2) = 2.707E+01
        XI( 3) = 2.449E-02
        XI( 4) =-4.000E-01
        XI( 5) = 2.000E-01
        XI( 6) = 5.000E-01
        XI( 7) = 0.000E+00
        XI( 8) = 6.000E-01
        XI( 9) = 9.000E-02
        XI(10) = 1.000E+00
        XI(11) = 2.000E+00
        XI(12) = 1.216E+00
        XI(13) = 2.500E-01
        XI(14) = 5.400E-01
        XI(15) = 7.700E-01
        XI(16) =-8.800E-01
        XI(17) = 5.400E-01
        XI(18) = 8.640E+00
        XI(19) = 9.000E-01

      ENDIF

      XNUS2   = XI(12)
      XMUS2   = XI(13)
      XNUSPI2 = XI(14)

      XNUH2   = XI(15)
      XMUH2   = XI(16)
      XNUHPI2 = XI(17)

      CALL HAD_CONV(IABS(JINT))

      PTCUT = XI(10)+0.065D0*EXP(0.9D0*SQRT(2.D0*LOG(ECM)))
      INDX = abs(JINT)
      IHAR(INDX) = IHAR(INDX)+1
      SIGHAR = SIGQCD(IHAR(INDX),INDX)

      S = ECM**2

      BREG =  ABS(XI(18)) + XI(19)*LOG(S)
      BPOM =  ABS(XI(12)) + XI(13)*LOG(S)
      IK = ABS(JINT)
      DO JB=1,NB
        B = DB*FLOAT(JB-1)
        ABR(IK,JB) = 2./(8.*PI*BREG)*EXP(-B**2/(4.*BREG))
        ABP(IK,JB) = 2./(8.*PI*BPOM)*EXP(-B**2/(4.*BPOM))
      ENDDO

C  reggeon
      SIGSR = ABS(XI(2))*S**(-ABS(XI(4)))
      SIG_BRN(1) = SIGSR/CMBARN
C  pomeron (soft part)
      SIGSP = ABS(XI(1))*S**ABS(XI(3))
      SIG_BRN(2) = SIGSP/CMBARN
C  pomeron (hard part)
      SIG_BRN(3) = SIGHAR/CMBARN

C  2x2 channel low-mass model and separate high-mass diffraction

      al1 = XI(5)
      be1 = XI(6)
      al2 = al1
      be2 = be1
      EnhPP  = XI(9)
      EnhPiP = EnhPP

      CALL SIG_JET_3 (SIG_brn,JINT,SIG_tot,SIG_ela,SIG_ine,SIG_sum,
     &                SIG_dif_1,SIG_dif_2,SIG_dd,B_el,PROB)

      SIGTOT = SIG_tot*CMBARN
      SIGINE = SIG_ine*CMBARN
      SIGSUM = SIG_sum*CMBARN
      SIGELc = SIGTOT-SIGINE
      SIGEL  = SIG_ela*CMBARN
      SIGSD1(1) = SIG_dif_1(1)*CMBARN
      SIGSD1(2) = SIG_dif_1(2)*CMBARN
      SIGSD2(1) = SIG_dif_2(1)*CMBARN
      SIGSD2(2) = SIG_dif_2(2)*CMBARN
      SIGDD(1)  = SIG_dd(1)*CMBARN
      SIGDD(2)  = SIG_dd(2)*CMBARN
      SLOPE  = B_EL
      SLOPEc = SIG_tot**2/(16.*Pi*SIG_ela)

      DE = ABS(SIGEL+SIGINE-SIGTOT)/SIGTOT
      IF(DE.GT.0.01) THEN
        print *,'SIBSIG:      Ecm: ',ECM
        print *,'          SIGTOT: ',SIGTOT
        print *,'        SIGEL1/2: ',SIGEL,SIGELc
        print *,'        SLOPE1/2: ',SLOPE,SLOPEc
        print *,'        SIGDIF 1: ',SIGSD1
        print *,'        SIGDIF 2: ',SIGSD2
        print *,'         SIGDDIF: ',SIGDD
        print *,'      SUM-SIGTOT: ',SIGEL+SIGINE-SIGTOT
      ENDIF

C  SIBYLL interface to single precision

      SIB_PTmin   = PTCUT
      SIB_SIG_tot = SIGTOT
      SIB_SIG_ine = SIGINE
      SIB_diff(1) = SIGSD1(1)+SIGSD1(2)
      SIB_diff(2) = SIGSD2(1)+SIGSD2(2)
      SIB_diff(3) = SIGDD(1)+SIGDD(2)
      SIB_B_el    = SLOPE
      DO I=0,NS_max
        DO K=0,NH_max
          SIB_PJET(I,K) = PROB(I,K)
        ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE SIG_JET_3 (SIG_brn, JINT, SIG_TOT, SIG_ELA,
     &        SIG_INE, SIG_sum, SIG_DIF1, SIG_DIF2, SIG_DD, B_EL, P_int)
C-----------------------------------------------------------------------
C
C...This subroutine  receives in INPUT:
C.       SIG_brn (GeV-2)  Born graph cross sections
C.       JINT (1 = pp interaction)    (2 pi-p interaction)
C.       neg. value: without calculation of interaction probabilities
C.
C.  and returns as output:
C.       SIG_???  , B_el
C.       and P_int(0:NS_max,0:NH_max)   interaction probabilities
C
C   two x two -channel approximation for diffraction
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)

      DIMENSION SIG_brn(3)
      PARAMETER (NS_max = 20, NH_max = 50)

      COMMON /S_CFACT/ FACT(0:NH_max), CO_BIN(0:NH_max,0:NH_max)
      COMMON /S_CHDCNV/ABR(2,400),ABP(2,400),ABH(2,400),DB,NB

      COMMON /PROFILE/XNUS2,XMUS2,XNUSPI2,
     &                XNUH2,XMUH2,XNUHPI2,
     &                EnhPP,EnhPiP,al1,be1,al2,be2

      DIMENSION SIG_DIF1(2),SIG_DIF2(2),SIG_DD(2),
     &          P_int(0:NS_max,0:NH_max)
      SAVE
      DATA PI /3.1415926/

      DO J=0,NH_max
        DO I=0,NS_max
          P_int(I,J) = 0.
        ENDDO
      ENDDO

      ga1 = sqrt(al1*al1+be1*be1)
      ga2 = sqrt(al2*al2+be2*be2)

      fe_a_1  = (1.+al1/ga1)/2.
      fe_a_2  = (1.-al1/ga1)/2.
      fd_a_1  = sqrt(1.-(al1/ga1)**2)/2.
      fd_a_2  = -fd_a_1

      fe_b_1  = (1.+al2/ga2)/2.
      fe_b_2  = (1.-al2/ga2)/2.
      fd_b_1  = sqrt(1.-(al2/ga2)**2)/2.
      fd_b_2  = -fd_b_1

      fe_11 = fe_a_1*fe_b_1
      fe_22 = fe_a_2*fe_b_2
      fe_12 = fe_a_1*fe_b_2
      fe_21 = fe_a_2*fe_b_1

      fd_a_11 = fd_a_1*fe_b_1
      fd_a_22 = fd_a_2*fe_b_2
      fd_a_12 = fd_a_1*fe_b_2
      fd_a_21 = fd_a_2*fe_b_1

      fd_b_11 = fe_a_1*fd_b_1
      fd_b_22 = fe_a_2*fd_b_2
      fd_b_12 = fe_a_1*fd_b_2
      fd_b_21 = fe_a_2*fd_b_1

      fdd_11 = fd_a_1*fd_b_1
      fdd_22 = fd_a_2*fd_b_2
      fdd_12 = fd_a_1*fd_b_2
      fdd_21 = fd_a_2*fd_b_1


      sum_abs = 0.
      sum_tot = 0.
      sum_ela = 0.
      sum_sd_a = 0.
      sum_sd_b = 0.
      sum_dd  = 0.
      sum_B   = 0.

      IK = ABS(JINT)
      if(JINT.GT.0) then
        I0MAX = NS_max
        J0MAX = NH_max
      ELSE
        I0MAX = 1
        J0MAX = 1
      ENDIF
      SIG_REG = SIG_BRN(1)
      SIG_POM = SIG_BRN(2)
      SIG_HAR = SIG_BRN(3)

      DO JB=1,NB

         B = DB*FLOAT(JB-1)

         ABREG = ABR(IK,JB)
         ABPOM = ABP(IK,JB)
         ABHAR = ABH(IK,JB)

         chi2_soft = ABREG*SIG_REG+ABPOM*SIG_POM
         chi2_soft_11 = (1.-al1+ga1)*(1.-al2+ga2)*chi2_soft
         chi2_soft_22 = (1.-al1-ga1)*(1.-al2-ga2)*chi2_soft
         chi2_soft_12 = (1.-al1+ga1)*(1.-al2-ga2)*chi2_soft
         chi2_soft_21 = (1.-al1-ga1)*(1.-al2+ga2)*chi2_soft

         chi2_hard = ABHAR*SIG_HAR
         chi2_hard_11 = (1.-al1+ga1)*(1.-al2+ga2)*chi2_hard
         chi2_hard_22 = (1.-al1-ga1)*(1.-al2-ga2)*chi2_hard
         chi2_hard_12 = (1.-al1+ga1)*(1.-al2-ga2)*chi2_hard
         chi2_hard_21 = (1.-al1-ga1)*(1.-al2+ga2)*chi2_hard


         ef_11  = exp(-0.5*(chi2_soft_11+chi2_hard_11))
         ef_22  = exp(-0.5*(chi2_soft_22+chi2_hard_22))
         ef_12 = exp(-0.5*(chi2_soft_12+chi2_hard_12))
         ef_21 = exp(-0.5*(chi2_soft_21+chi2_hard_21))

         esf_11  = ef_11**2
         esf_22  = ef_22**2
         esf_12  = ef_12**2
         esf_21  = ef_21**2

         F_ine = B*(1. - fe_11*esf_11 - fe_12*esf_12
     &                 - fe_21*esf_21 - fe_22*esf_22)
         F_tot = 1. - fe_11*ef_11 - fe_12*ef_12
     &              - fe_21*ef_21 - fe_22*ef_22
         F_ela = B*F_tot**2
         F_tot = B*F_tot

         F_sd_a = B*(fd_a_11*ef_11 + fd_a_12*ef_12
     &              + fd_a_21*ef_21 + fd_a_22*ef_22)**2
         F_sd_b = B*(fd_b_11*ef_11 + fd_b_12*ef_12
     &              + fd_b_21*ef_21 + fd_b_22*ef_22)**2
         F_dd  = B*(fdd_11*ef_11 + fdd_12*ef_12
     &              + fdd_21*ef_21 + fdd_22*ef_22)**2

         sum_abs = sum_abs+F_ine
         sum_tot = sum_tot+F_tot
         sum_ela = sum_ela+F_ela

         sum_sd_a = sum_sd_a+F_sd_a
         sum_sd_b = sum_sd_b+F_sd_b
         sum_dd  = sum_dd +F_dd

         sum_B   = sum_b+B**2*F_tot

         fac_11 = B*esf_11
         fac_22 = B*esf_22
         fac_12 = B*esf_12
         fac_21 = B*esf_21
         soft_rec_11 = 1./chi2_soft_11
         soft_rec_22 = 1./chi2_soft_22
         soft_rec_12 = 1./chi2_soft_12
         soft_rec_21 = 1./chi2_soft_21
         chi2_hard_11 = max(chi2_hard_11,1.d-10)
         chi2_hard_22 = max(chi2_hard_22,1.d-10)
         chi2_hard_12 = max(chi2_hard_12,1.d-10)
         chi2_hard_21 = max(chi2_hard_21,1.d-10)
         DO I=0,I0MAX
           soft_rec_11 = soft_rec_11*chi2_soft_11
           soft_rec_22 = soft_rec_22*chi2_soft_22
           soft_rec_12 = soft_rec_12*chi2_soft_12
           soft_rec_21 = soft_rec_21*chi2_soft_21
           hard_rec_11 = 1./chi2_hard_11
           hard_rec_22 = 1./chi2_hard_22
           hard_rec_12 = 1./chi2_hard_12
           hard_rec_21 = 1./chi2_hard_21
           DO J=0,J0MAX
             hard_rec_11 = hard_rec_11*chi2_hard_11
             hard_rec_22 = hard_rec_22*chi2_hard_22
             hard_rec_12 = hard_rec_12*chi2_hard_12
             hard_rec_21 = hard_rec_21*chi2_hard_21
             P_int(I,J) = P_int(I,J)
     &                + fe_11*soft_rec_11*hard_rec_11*fac_11
     &                + fe_22*soft_rec_22*hard_rec_22*fac_22
     &                + fe_12*soft_rec_12*hard_rec_12*fac_12
     &                + fe_21*soft_rec_21*hard_rec_21*fac_21
           ENDDO
         ENDDO

      ENDDO

      SIG_abs  = SUM_abs*2.*PI*DB
      SIG_tot  = SUM_tot*4.*PI*DB
      SIG_ela  = SUM_ela*2.*PI*DB
      SIG_dif1(1) = SUM_sd_a*2.*PI*DB
      SIG_dif2(1) = SUM_sd_b*2.*PI*DB
      SIG_dd(1)   = SUM_dd*2.*PI*DB
      SIG_ine  = SIG_abs + SIG_dif1(1) + SIG_dif2(1) + SIG_dd(1)
      B_EL     = sum_B/SUM_tot/2.

      SA = 0.
      P_int(0,0) = 0.
      DO I=0,I0MAX
        DO J=0,J0MAX
          fac = FACT(I)*FACT(J)
          P_int(I,J) = P_int(I,J)/fac
          SA = SA + P_int(I,J)
        ENDDO
      ENDDO

      SIG_hmsd = EnhPP*(P_int(1,0)+P_int(0,1))*2.*PI*DB
      SIG_hmdd = be1**2*SIG_hmsd + be2**2*SIG_hmsd
     &          + EnhPP**2*P_int(1,1)*2.*PI*DB

      SIG_dif1(2) = SIG_hmsd
      SIG_dif2(2) = SIG_hmsd
      SIG_dd(2)   = SIG_hmdd

      SIG_sum = SA*2.*PI*DB

      DO I=0,I0MAX
        DO J=0,J0MAX
          P_int(I,J) = P_int(I,J)/SA
        ENDDO
      ENDDO

      RETURN
      END
C
C
      SUBROUTINE HAD_CONV(JINT)
C-----------------------------------------------------------------------
C
C...Convolution of hadrons profile
C.  [function A(b) of Durand and Pi]
C.  precalculate and put  in COMMON block
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C
      COMMON /S_CHDCNV/ABR(2,400),ABP(2,400),ABH(2,400),DB,NB

      DOUBLE PRECISION NU2, MU2, NUPI2, NU, MU, NUPI
      COMMON /S_CH0CNV/ NU2, MU2, NUPI2, NU, MU, NUPI

C
      COMMON /PROFILE/XNUS2,XMUS2,XNUSPI2,
     &                XNUH2,XMUH2,XNUHPI2,
     &                ENHPP,ENHPIP,al1,be1,al2,be2
      SAVE

C...integration constants
      BMAX = 50.
      NB  = 400
      DB = BMAX/FLOAT(NB)

C  soft reggeon interactions

      NU2   = XNUS2
      MU2   = XMUS2
      NUPI2 = XNUSPI2

      NU = SQRT(NU2)
      MU = SQRT(ABS(MU2))
      NUPI = SQRT(NUPI2)

      DO JB=1,NB
         B = DB*FLOAT(JB-1)
         IF(JINT.EQ.1) THEN
           ABR(JINT,JB) = A_PP(B)
         ELSE
           ABR(JINT,JB) = A_PIP(B)
         ENDIF
      ENDDO

C  soft pomeron interactions

      NU2   = XNUS2
      MU2   = XMUS2
      NUPI2 = XNUSPI2

      NU = SQRT(NU2)
      MU = SQRT(ABS(MU2))
      NUPI = SQRT(NUPI2)

      DO JB=1,NB
         B = DB*FLOAT(JB-1)
         IF(JINT.EQ.1) THEN
           ABP(JINT,JB) = A_PP(B)
         ELSE
           ABP(JINT,JB) = A_PIP(B)
         ENDIF
      ENDDO

C  hard pomeron interactions

      NU2   = XNUH2
      MU2   = XMUH2
      NUPI2 = XNUHPI2

      NU = SQRT(NU2)
      MU = SQRT(ABS(MU2))
      NUPI = SQRT(NUPI2)

      DB = BMAX/FLOAT(NB)
      DO JB=1,NB
         B = DB*FLOAT(JB-1)
         IF(JINT.EQ.1) THEN
           ABH(JINT,JB) = A_PP(B)
         ELSE
           ABH(JINT,JB) = A_PIP(B)
         ENDIF
      ENDDO

      RETURN
      END
C
C
      DOUBLE PRECISION FUNCTION A_pp (b)
C...Convolution of parton distribution for pp interaction
      IMPLICIT DOUBLE PRECISION (A-Z)
C
      DOUBLE PRECISION NU2, MU2, NUPI2, NU, MU, NUPI
      COMMON /S_CH0CNV/ NU2, MU2, NUPI2, NU, MU, NUPI
      SAVE
      data pi / 3.1415926/

      ETA = NU2/MU2

      IF(ETA.LT.0.D0) THEN

        c = nu**5/(96.*pi)
        if (b .gt. 0.0001D0)  then
           A_pp = c*b**3 * bessk (3, b*nu)
        else
           A_pp = nu**2/(12.*pi)
        endif

      ELSE

        X = B*NU
        Y = B*MU
        C = NU2/(12.*PI)/(1.-ETA)**2
        IF(X.GT.0.0001D0) THEN
          A_PP = C*(1./8.*X**3*BESSK(3,X)
     &          -3./2.*ETA/(1.-ETA)*X**2*BESSK(2,X)
     &          +9*ETA**2/(1.-ETA)**2*X*BESSK1(X)
     &          -24*ETA**3/(1.-ETA)**3*(BESSK0(X)-BESSK0(Y))
     &          +3.*ETA**3/(1.-ETA)**2*Y*BESSK1(Y))
        ELSE
          A_PP = C*(1./8.*8.
     &          -3./2.*ETA/(1.-ETA)*2.
     &          +9*ETA**2/(1.-ETA)**2*1.
     &          -24*ETA**3/(1.-ETA)**3*LOG(MU/NU)
     &          +3.*ETA**3/(1.-ETA)**2*1.)
        ENDIF

      ENDIF

      RETURN
      END
C
C
      DOUBLE PRECISION FUNCTION A_pip (b)
C...Convolution of parton distribution for pip interaction
      IMPLICIT DOUBLE PRECISION (A-Z)
C
      DOUBLE PRECISION NU2, MU2, NUPI2, NU, MU, NUPI
      COMMON /S_CH0CNV/ NU2, MU2, NUPI2, NU, MU, NUPI
      SAVE
      data pi / 3.1415926/

      eta = nu2/nupi2
      c = nu2/(2.*pi) * 1./(1.-eta)

      if (b .gt. 0.0001D0)  then
         b1 = b*nu
         b2 = b*nupi
         f1 = 0.5*b1 * bessk1(b1)
         f2 = eta/(1.-eta)*(bessk0(b2)- bessk0(b1))
         A_pip = c*(f1+f2)
      else
         A_pip = c*(0.5 + eta/(1.-eta)*log(nu/nupi))
      endif
      return
      end
C
C
C----------------------------------------------------------------------------
C  Bessel functions
C----------------------------------------------------------------------------
C
      FUNCTION BESSK0(X)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
*     REAL*8 Y,P1,P2,P3,P4,P5,P6,P7,
*    *    Q1,Q2,Q3,Q4,Q5,Q6,Q7
      SAVE
C
      DATA P1,P2,P3,P4,P5,P6,P7/-0.57721566D0,0.42278420D0,
     *    0.23069756D0,0.3488590D-1,0.262698D-2,0.10750D-3,0.74D-5/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,-0.7832358D-1,
     * 0.2189568D-1,-0.1062446D-1,0.587872D-2,-0.251540D-2,0.53208D-3/

      IF (X.LE.2.0) THEN
        Y=X*X/4.0
        BESSK0=(-LOG(X/2.0)*BESSI0(X))+(P1+Y*(P2+Y*(P3+
     *        Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
        Y=(2.0/X)
        BESSK0=(EXP(-X)/SQRT(X))*(Q1+Y*(Q2+Y*(Q3+
     *        Y*(Q4+Y*(Q5+Y*(Q6+Y*Q7))))))
      ENDIF
      RETURN
      END
C
C
      FUNCTION BESSK1(X)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C
*     REAL*8 Y,P1,P2,P3,P4,P5,P6,P7,
*    *    Q1,Q2,Q3,Q4,Q5,Q6,Q7
      SAVE
      DATA P1,P2,P3,P4,P5,P6,P7/1.0D0,0.15443144D0,-0.67278579D0,
     *    -0.18156897D0,-0.1919402D-1,-0.110404D-2,-0.4686D-4/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,0.23498619D0,
     *    -0.3655620D-1,0.1504268D-1,-0.780353D-2,0.325614D-2,
     *    -0.68245D-3/

      IF (X.LE.2.0) THEN
        Y=X*X/4.0
        BESSK1=(LOG(X/2.0)*BESSI1(X))+(1.0/X)*(P1+Y*(P2+
     *      Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
        Y=2.0/X
        BESSK1=(EXP(-X)/SQRT(X))*(Q1+Y*(Q2+Y*(Q3+
     *      Y*(Q4+Y*(Q5+Y*(Q6+Y*Q7))))))
      ENDIF
      RETURN
      END
C
C
      FUNCTION BESSK(N,X)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      SAVE
C
      IF (N.LT.2) STOP 'bad argument N in BESSK'
      TOX=2.0/X
      BKM=BESSK0(X)
      BK=BESSK1(X)
      DO 11 J=1,N-1
        BKP=BKM+J*TOX*BK
        BKM=BK
        BK=BKP
11    CONTINUE
      BESSK=BK
      RETURN
      END
C
C
      FUNCTION BESSI0(X)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C
*     REAL*8 Y,P1,P2,P3,P4,P5,P6,P7,
*    *    Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9
      SAVE
      DATA P1,P2,P3,P4,P5,P6,P7/1.0D0,3.5156229D0,3.0899424D0,
     *    1.2067492D0,
     *    0.2659732D0,0.360768D-1,0.45813D-2/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,0.1328592D-1,
     *    0.225319D-2,-0.157565D-2,0.916281D-2,-0.2057706D-1,
     *    0.2635537D-1,-0.1647633D-1,0.392377D-2/

      IF (ABS(X).LT.3.75) THEN
        Y=(X/3.75)**2
        BESSI0=P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))
      ELSE
        AX=ABS(X)
        Y=3.75/AX
        BESSI0=(EXP(AX)/SQRT(AX))*(Q1+Y*(Q2+Y*(Q3+Y*(Q4
     *      +Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9))))))))
      ENDIF
      RETURN
      END
C
C
      FUNCTION BESSI1(X)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C
*     REAL*8 Y,P1,P2,P3,P4,P5,P6,P7,
*    *    Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9
      SAVE
      DATA P1,P2,P3,P4,P5,P6,P7/0.5D0,0.87890594D0,0.51498869D0,
     *    0.15084934D0,0.2658733D-1,0.301532D-2,0.32411D-3/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,-0.3988024D-1,
     *    -0.362018D-2,0.163801D-2,-0.1031555D-1,0.2282967D-1,
     *    -0.2895312D-1,0.1787654D-1,-0.420059D-2/

      IF (ABS(X).LT.3.75) THEN
        Y=(X/3.75)**2
        BESSI1=X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
        AX=ABS(X)
        Y=3.75/AX
        BESSI1=(EXP(AX)/SQRT(AX))*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+
     *      Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9))))))))
      ENDIF
      RETURN
      END


      SUBROUTINE FACT_INI
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)

      PARAMETER (NS_max = 20, NH_max = 50)
      COMMON /S_CFACT/ FACT(0:NH_max), CO_BIN(0:NH_max,0:NH_max)
      SAVE

      FACT(0) = 1.
      DO J=1,NH_max
         FACT(J) = FACT(J-1)*FLOAT(J)
      ENDDO
      DO J=0,NH_max
         DO K=0,J
            CO_BIN(J,K) = FACT(J)/(FACT(K)*FACT(J-K))
         ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE SIB_GAUSET(AX,BX,NX,Z,W)
C-----------------------------------------------------------------------
C
C     N-point gauss zeros and weights for the interval (AX,BX) are
C           stored in  arrays Z and W respectively.
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      COMMON /GQCOM/A(273),X(273),KTAB(96)
      DIMENSION Z(NX),W(NX)
      SAVE
      DATA INIT/0/
C
      ALPHA=0.5*(BX+AX)
      BETA=0.5*(BX-AX)
      N=NX
*
*  the N=1 case:
      IF(N.NE.1) GO TO 1
      Z(1)=ALPHA
      W(1)=BX-AX
      RETURN
*
*  the Gauss cases:
    1 IF((N.LE.16).AND.(N.GT.1)) GO TO 2
      IF(N.EQ.20) GO TO 2
      IF(N.EQ.24) GO TO 2
      IF(N.EQ.32) GO TO 2
      IF(N.EQ.40) GO TO 2
      IF(N.EQ.48) GO TO 2
      IF(N.EQ.64) GO TO 2
      IF(N.EQ.80) GO TO 2
      IF(N.EQ.96) GO TO 2
*
*  the extended Gauss cases:
      IF((N/96)*96.EQ.N) GO TO 3
*
C  jump to center of intervall intrgration:
      GO TO 100
*
C  get Gauss point array
*
    2 CALL PO106BD
C     -print out message
*     IF(INIT.LE.20)THEN
*       INIT=init+1
*       WRITE (6,*) ' initialization of Gauss int. N=',N
*     ENDIF
C  extract real points
      K=KTAB(N)
      M=N/2
      DO 21 J=1,M
C       extract values from big array
        JTAB=K-1+J
        WTEMP=BETA*A(JTAB)
        DELTA=BETA*X(JTAB)
C       store them backward
        Z(J)=ALPHA-DELTA
        W(J)=WTEMP
C       store them forward
        JP=N+1-J
        Z(JP)=ALPHA+DELTA
        W(JP)=WTEMP
   21 CONTINUE
C     store central point (odd N)
      IF((N-M-M).EQ.0) RETURN
      Z(M+1)=ALPHA
      JMID=K+M
      W(M+1)=BETA*A(JMID)
      RETURN
C
C  get ND96 times chained 96 Gauss point array
C
    3 CALL PO106BD
C  print out message
      IF(INIT.LE.20)THEN
        INIT=init+1
        WRITE (6,*) ' initialization of extended Gauss int. N=',N
      ENDIF
C     -extract real points
      K=KTAB(96)
      ND96=N/96
      DO 31 J=1,48
C       extract values from big array
        JTAB=K-1+J
        WTEMP=BETA*A(JTAB)
        DELTA=BETA*X(JTAB)
        WTeMP=WTEMP/ND96
        DeLTA=DELTA/ND96
        DO 32 JD96=0,ND96-1
          ZCNTR= (ALPHA-BETA)+ BETA*FLOAT(2*JD96+1)/FLOAT(ND96)
C         store them backward
          Z(J+JD96*96)=ZCNTR-DELTA
          W(J+JD96*96)=WTEMP
C         store them forward
          JP=96+1-J
          Z(JP+JD96*96)=ZCNTR+DELTA
          W(JP+JD96*96)=WTEMP
   32   CONTINUE
   31 CONTINUE
      RETURN
*
C  the center of intervall cases:
  100 CONTINUE
C  print out message
      IF(INIT.LE.20)THEN
        INIT=init+1
        WRITE (6,*) ' init. of center of intervall int. N=',N
      ENDIF
C  put in constant weight and equally spaced central points
      N=IABS(N)
      DO 111 IN=1,N
        WIN=(BX-AX)/FLOAT(N)
        Z(IN)=AX  + (FLOAT(IN)-.5)*WIN
  111 W(IN)=WIN
      RETURN
      END
C
C
      SUBROUTINE PO106BD
C-----------------------------------------------------------------------
C
C     store big arrays needed for Gauss integral, CERNLIB D106BD
C     (arrays A,X,ITAB copied on B,Y,LTAB)
C
C-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      COMMON /GQCOM/ B(273),Y(273),LTAB(96)
      DIMENSION      A(273),X(273),KTAB(96)
      SAVE
C
C-----TABLE OF INITIAL SUBSCRIPTS FOR N=2(1)16(4)96
      DATA KTAB(2)/1/
      DATA KTAB(3)/2/
      DATA KTAB(4)/4/
      DATA KTAB(5)/6/
      DATA KTAB(6)/9/
      DATA KTAB(7)/12/
      DATA KTAB(8)/16/
      DATA KTAB(9)/20/
      DATA KTAB(10)/25/
      DATA KTAB(11)/30/
      DATA KTAB(12)/36/
      DATA KTAB(13)/42/
      DATA KTAB(14)/49/
      DATA KTAB(15)/56/
      DATA KTAB(16)/64/
      DATA KTAB(20)/72/
      DATA KTAB(24)/82/
      DATA KTAB(28)/82/
      DATA KTAB(32)/94/
      DATA KTAB(36)/94/
      DATA KTAB(40)/110/
      DATA KTAB(44)/110/
      DATA KTAB(48)/130/
      DATA KTAB(52)/130/
      DATA KTAB(56)/130/
      DATA KTAB(60)/130/
      DATA KTAB(64)/154/
      DATA KTAB(68)/154/
      DATA KTAB(72)/154/
      DATA KTAB(76)/154/
      DATA KTAB(80)/186/
      DATA KTAB(84)/186/
      DATA KTAB(88)/186/
      DATA KTAB(92)/186/
      DATA KTAB(96)/226/
C
C-----TABLE OF ABSCISSAE (X) AND WEIGHTS (A) FOR INTERVAL (-1,+1).
C
C-----N=2
      DATA X(1)/0.577350269189626D0  /, A(1)/1.000000000000000D0  /
C-----N=3
      DATA X(2)/0.774596669241483D0  /, A(2)/0.555555555555556D0  /
      DATA X(3)/0.000000000000000D0  /, A(3)/0.888888888888889D0  /
C-----N=4
      DATA X(4)/0.861136311594053D0  /, A(4)/0.347854845137454D0  /
      DATA X(5)/0.339981043584856D0  /, A(5)/0.652145154862546D0  /
C-----N=5
      DATA X(6)/0.906179845938664D0  /, A(6)/0.236926885056189D0  /
      DATA X(7)/0.538469310105683D0  /, A(7)/0.478628670499366D0  /
      DATA X(8)/0.000000000000000D0  /, A(8)/0.568888888888889D0  /
C-----N=6
      DATA X(9)/0.932469514203152D0  /, A(9)/0.171324492379170D0  /
      DATA X(10)/0.661209386466265D0 /, A(10)/0.360761573048139D0 /
      DATA X(11)/0.238619186083197D0 /, A(11)/0.467913934572691D0 /
C-----N=7
      DATA X(12)/0.949107912342759D0 /, A(12)/0.129484966168870D0 /
      DATA X(13)/0.741531185599394D0 /, A(13)/0.279705391489277D0 /
      DATA X(14)/0.405845151377397D0 /, A(14)/0.381830050505119D0 /
      DATA X(15)/0.000000000000000D0 /, A(15)/0.417959183673469D0 /
C-----N=8
      DATA X(16)/0.960289856497536D0 /, A(16)/0.101228536290376D0 /
      DATA X(17)/0.796666477413627D0 /, A(17)/0.222381034453374D0 /
      DATA X(18)/0.525532409916329D0 /, A(18)/0.313706645877887D0 /
      DATA X(19)/0.183434642495650D0 /, A(19)/0.362683783378362D0 /
C-----N=9
      DATA X(20)/0.968160239507626D0 /, A(20)/0.081274388361574D0 /
      DATA X(21)/0.836031107326636D0 /, A(21)/0.180648160694857D0 /
      DATA X(22)/0.613371432700590D0 /, A(22)/0.260610696402935D0 /
      DATA X(23)/0.324253423403809D0 /, A(23)/0.312347077040003D0 /
      DATA X(24)/0.000000000000000D0 /, A(24)/0.330239355001260D0 /
C-----N=10
      DATA X(25)/0.973906528517172D0 /, A(25)/0.066671344308688D0 /
      DATA X(26)/0.865063366688985D0 /, A(26)/0.149451349150581D0 /
      DATA X(27)/0.679409568299024D0 /, A(27)/0.219086362515982D0 /
      DATA X(28)/0.433395394129247D0 /, A(28)/0.269266719309996D0 /
      DATA X(29)/0.148874338981631D0 /, A(29)/0.295524224714753D0 /
C-----N=11
      DATA X(30)/0.978228658146057D0 /, A(30)/0.055668567116174D0 /
      DATA X(31)/0.887062599768095D0 /, A(31)/0.125580369464905D0 /
      DATA X(32)/0.730152005574049D0 /, A(32)/0.186290210927734D0 /
      DATA X(33)/0.519096129206812D0 /, A(33)/0.233193764591990D0 /
      DATA X(34)/0.269543155952345D0 /, A(34)/0.262804544510247D0 /
      DATA X(35)/0.000000000000000D0 /, A(35)/0.272925086777901D0 /
C-----N=12
      DATA X(36)/0.981560634246719D0 /, A(36)/0.047175336386512D0 /
      DATA X(37)/0.904117256370475D0 /, A(37)/0.106939325995318D0 /
      DATA X(38)/0.769902674194305D0 /, A(38)/0.160078328543346D0 /
      DATA X(39)/0.587317954286617D0 /, A(39)/0.203167426723066D0 /
      DATA X(40)/0.367831498998180D0 /, A(40)/0.233492536538355D0 /
      DATA X(41)/0.125233408511469D0 /, A(41)/0.249147045813403D0 /
C-----N=13
      DATA X(42)/0.984183054718588D0 /, A(42)/0.040484004765316D0 /
      DATA X(43)/0.917598399222978D0 /, A(43)/0.092121499837728D0 /
      DATA X(44)/0.801578090733310D0 /, A(44)/0.138873510219787D0 /
      DATA X(45)/0.642349339440340D0 /, A(45)/0.178145980761946D0 /
      DATA X(46)/0.448492751036447D0 /, A(46)/0.207816047536889D0 /
      DATA X(47)/0.230458315955135D0 /, A(47)/0.226283180262897D0 /
      DATA X(48)/0.000000000000000D0 /, A(48)/0.232551553230874D0 /
C-----N=14
      DATA X(49)/0.986283808696812D0 /, A(49)/0.035119460331752D0 /
      DATA X(50)/0.928434883663574D0 /, A(50)/0.080158087159760D0 /
      DATA X(51)/0.827201315069765D0 /, A(51)/0.121518570687903D0 /
      DATA X(52)/0.687292904811685D0 /, A(52)/0.157203167158194D0 /
      DATA X(53)/0.515248636358154D0 /, A(53)/0.185538397477938D0 /
      DATA X(54)/0.319112368927890D0 /, A(54)/0.205198463721296D0 /
      DATA X(55)/0.108054948707344D0 /, A(55)/0.215263853463158D0 /
C-----N=15
      DATA X(56)/0.987992518020485D0 /, A(56)/0.030753241996117D0 /
      DATA X(57)/0.937273392400706D0 /, A(57)/0.070366047488108D0 /
      DATA X(58)/0.848206583410427D0 /, A(58)/0.107159220467172D0 /
      DATA X(59)/0.724417731360170D0 /, A(59)/0.139570677926154D0 /
      DATA X(60)/0.570972172608539D0 /, A(60)/0.166269205816994D0 /
      DATA X(61)/0.394151347077563D0 /, A(61)/0.186161000015562D0 /
      DATA X(62)/0.201194093997435D0 /, A(62)/0.198431485327111D0 /
      DATA X(63)/0.000000000000000D0 /, A(63)/0.202578241925561D0 /
C-----N=16
      DATA X(64)/0.989400934991650D0 /, A(64)/0.027152459411754D0 /
      DATA X(65)/0.944575023073233D0 /, A(65)/0.062253523938648D0 /
      DATA X(66)/0.865631202387832D0 /, A(66)/0.095158511682493D0 /
      DATA X(67)/0.755404408355003D0 /, A(67)/0.124628971255534D0 /
      DATA X(68)/0.617876244402644D0 /, A(68)/0.149595988816577D0 /
      DATA X(69)/0.458016777657227D0 /, A(69)/0.169156519395003D0 /
      DATA X(70)/0.281603550779259D0 /, A(70)/0.182603415044924D0 /
      DATA X(71)/0.095012509837637D0 /, A(71)/0.189450610455069D0 /
C-----N=20
      DATA X(72)/0.993128599185094D0 /, A(72)/0.017614007139152D0 /
      DATA X(73)/0.963971927277913D0 /, A(73)/0.040601429800386D0 /
      DATA X(74)/0.912234428251325D0 /, A(74)/0.062672048334109D0 /
      DATA X(75)/0.839116971822218D0 /, A(75)/0.083276741576704D0 /
      DATA X(76)/0.746331906460150D0 /, A(76)/0.101930119817240D0 /
      DATA X(77)/0.636053680726515D0 /, A(77)/0.118194531961518D0 /
      DATA X(78)/0.510867001950827D0 /, A(78)/0.131688638449176D0 /
      DATA X(79)/0.373706088715419D0 /, A(79)/0.142096109318382D0 /
      DATA X(80)/0.227785851141645D0 /, A(80)/0.149172986472603D0 /
      DATA X(81)/0.076526521133497D0 /, A(81)/0.152753387130725D0 /
C-----N=24
      DATA X(82)/0.995187219997021D0 /, A(82)/0.012341229799987D0 /
      DATA X(83)/0.974728555971309D0 /, A(83)/0.028531388628933D0 /
      DATA X(84)/0.938274552002732D0 /, A(84)/0.044277438817419D0 /
      DATA X(85)/0.886415527004401D0 /, A(85)/0.059298584915436D0 /
      DATA X(86)/0.820001985973902D0 /, A(86)/0.073346481411080D0 /
      DATA X(87)/0.740124191578554D0 /, A(87)/0.086190161531953D0 /
      DATA X(88)/0.648093651936975D0 /, A(88)/0.097618652104113D0 /
      DATA X(89)/0.545421471388839D0 /, A(89)/0.107444270115965D0 /
      DATA X(90)/0.433793507626045D0 /, A(90)/0.115505668053725D0 /
      DATA X(91)/0.315042679696163D0 /, A(91)/0.121670472927803D0 /
      DATA X(92)/0.191118867473616D0 /, A(92)/0.125837456346828D0 /
      DATA X(93)/0.064056892862605D0 /, A(93)/0.127938195346752D0 /
C-----N=32
      DATA X(94)/0.997263861849481D0 /, A(94)/0.007018610009470D0 /
      DATA X(95)/0.985611511545268D0 /, A(95)/0.016274394730905D0 /
      DATA X(96)/0.964762255587506D0 /, A(96)/0.025392065309262D0 /
      DATA X(97)/0.934906075937739D0 /, A(97)/0.034273862913021D0 /
      DATA X(98)/0.896321155766052D0 /, A(98)/0.042835898022226D0 /
      DATA X(99)/0.849367613732569D0 /, A(99)/0.050998059262376D0 /
      DATA X(100)/0.794483795967942D0/, A(100)/0.058684093478535D0/
      DATA X(101)/0.732182118740289D0/, A(101)/0.065822222776361D0/
      DATA X(102)/0.663044266930215D0/, A(102)/0.072345794108848D0/
      DATA X(103)/0.587715757240762D0/, A(103)/0.078193895787070D0/
      DATA X(104)/0.506899908932229D0/, A(104)/0.083311924226946D0/
      DATA X(105)/0.421351276130635D0/, A(105)/0.087652093004403D0/
      DATA X(106)/0.331868602282127D0/, A(106)/0.091173878695763D0/
      DATA X(107)/0.239287362252137D0/, A(107)/0.093844399080804D0/
      DATA X(108)/0.144471961582796D0/, A(108)/0.095638720079274D0/
      DATA X(109)/0.048307665687738D0/, A(109)/0.096540088514727D0/
C-----N=40
      DATA X(110)/0.998237709710559D0/, A(110)/0.004521277098533D0/
      DATA X(111)/0.990726238699457D0/, A(111)/0.010498284531152D0/
      DATA X(112)/0.977259949983774D0/, A(112)/0.016421058381907D0/
      DATA X(113)/0.957916819213791D0/, A(113)/0.022245849194166D0/
      DATA X(114)/0.932812808278676D0/, A(114)/0.027937006980023D0/
      DATA X(115)/0.902098806968874D0/, A(115)/0.033460195282547D0/
      DATA X(116)/0.865959503212259D0/, A(116)/0.038782167974472D0/
      DATA X(117)/0.824612230833311D0/, A(117)/0.043870908185673D0/
      DATA X(118)/0.778305651426519D0/, A(118)/0.048695807635072D0/
      DATA X(119)/0.727318255189927D0/, A(119)/0.053227846983936D0/
      DATA X(120)/0.671956684614179D0/, A(120)/0.057439769099391D0/
      DATA X(121)/0.612553889667980D0/, A(121)/0.061306242492928D0/
      DATA X(122)/0.549467125095128D0/, A(122)/0.064804013456601D0/
      DATA X(123)/0.483075801686178D0/, A(123)/0.067912045815233D0/
      DATA X(124)/0.413779204371605D0/, A(124)/0.070611647391286D0/
      DATA X(125)/0.341994090825758D0/, A(125)/0.072886582395804D0/
      DATA X(126)/0.268152185007253D0/, A(126)/0.074723169057968D0/
      DATA X(127)/0.192697580701371D0/, A(127)/0.076110361900626D0/
      DATA X(128)/0.116084070675255D0/, A(128)/0.077039818164247D0/
      DATA X(129)/0.038772417506050D0/, A(129)/0.077505947978424D0/
C-----N=48
      DATA X(130)/0.998771007252426D0/, A(130)/0.003153346052305D0/
      DATA X(131)/0.993530172266350D0/, A(131)/0.007327553901276D0/
      DATA X(132)/0.984124583722826D0/, A(132)/0.011477234579234D0/
      DATA X(133)/0.970591592546247D0/, A(133)/0.015579315722943D0/
      DATA X(134)/0.952987703160430D0/, A(134)/0.019616160457355D0/
      DATA X(135)/0.931386690706554D0/, A(135)/0.023570760839324D0/
      DATA X(136)/0.905879136715569D0/, A(136)/0.027426509708356D0/
      DATA X(137)/0.876572020274247D0/, A(137)/0.031167227832798D0/
      DATA X(138)/0.843588261624393D0/, A(138)/0.034777222564770D0/
      DATA X(139)/0.807066204029442D0/, A(139)/0.038241351065830D0/
      DATA X(140)/0.767159032515740D0/, A(140)/0.041545082943464D0/
      DATA X(141)/0.724034130923814D0/, A(141)/0.044674560856694D0/
      DATA X(142)/0.677872379632663D0/, A(142)/0.047616658492490D0/
      DATA X(143)/0.628867396776513D0/, A(143)/0.050359035553854D0/
      DATA X(144)/0.577224726083972D0/, A(144)/0.052890189485193D0/
      DATA X(145)/0.523160974722233D0/, A(145)/0.055199503699984D0/
      DATA X(146)/0.466902904750958D0/, A(146)/0.057277292100403D0/
      DATA X(147)/0.408686481990716D0/, A(147)/0.059114839698395D0/
      DATA X(148)/0.348755886292160D0/, A(148)/0.060704439165893D0/
      DATA X(149)/0.287362487355455D0/, A(149)/0.062039423159892D0/
      DATA X(150)/0.224763790394689D0/, A(150)/0.063114192286254D0/
      DATA X(151)/0.161222356068891D0/, A(151)/0.063924238584648D0/
      DATA X(152)/0.097004699209462D0/, A(152)/0.064466164435950D0/
      DATA X(153)/0.032380170962869D0/, A(153)/0.064737696812683D0/
C-----N=64
      DATA X(154)/0.999305041735772D0/, A(154)/0.001783280721696D0/
      DATA X(155)/0.996340116771955D0/, A(155)/0.004147033260562D0/
      DATA X(156)/0.991013371476744D0/, A(156)/0.006504457968978D0/
      DATA X(157)/0.983336253884625D0/, A(157)/0.008846759826363D0/
      DATA X(158)/0.973326827789910D0/, A(158)/0.011168139460131D0/
      DATA X(159)/0.961008799652053D0/, A(159)/0.013463047896718D0/
      DATA X(160)/0.946411374858402D0/, A(160)/0.015726030476024D0/
      DATA X(161)/0.929569172131939D0/, A(161)/0.017951715775697D0/
      DATA X(162)/0.910522137078502D0/, A(162)/0.020134823153530D0/
      DATA X(163)/0.889315445995114D0/, A(163)/0.022270173808383D0/
      DATA X(164)/0.865999398154092D0/, A(164)/0.024352702568710D0/
      DATA X(165)/0.840629296252580D0/, A(165)/0.026377469715054D0/
      DATA X(166)/0.813265315122797D0/, A(166)/0.028339672614259D0/
      DATA X(167)/0.783972358943341D0/, A(167)/0.030234657072402D0/
      DATA X(168)/0.752819907260531D0/, A(168)/0.032057928354851D0/
      DATA X(169)/0.719881850171610D0/, A(169)/0.033805161837141D0/
      DATA X(170)/0.685236313054233D0/, A(170)/0.035472213256882D0/
      DATA X(171)/0.648965471254657D0/, A(171)/0.037055128540240D0/
      DATA X(172)/0.611155355172393D0/, A(172)/0.038550153178615D0/
      DATA X(173)/0.571895646202634D0/, A(173)/0.039953741132720D0/
      DATA X(174)/0.531279464019894D0/, A(174)/0.041262563242623D0/
      DATA X(175)/0.489403145707052D0/, A(175)/0.042473515123653D0/
      DATA X(176)/0.446366017253464D0/, A(176)/0.043583724529323D0/
      DATA X(177)/0.402270157963991D0/, A(177)/0.044590558163756D0/
      DATA X(178)/0.357220158337668D0/, A(178)/0.045491627927418D0/
      DATA X(179)/0.311322871990210D0/, A(179)/0.046284796581314D0/
      DATA X(180)/0.264687162208767D0/, A(180)/0.046968182816210D0/
      DATA X(181)/0.217423643740007D0/, A(181)/0.047540165714830D0/
      DATA X(182)/0.169644420423992D0/, A(182)/0.047999388596458D0/
      DATA X(183)/0.121462819296120D0/, A(183)/0.048344762234802D0/
      DATA X(184)/0.072993121787799D0/, A(184)/0.048575467441503D0/
      DATA X(185)/0.024350292663424D0/, A(185)/0.048690957009139D0/
C-----N=80
      DATA X(186)/0.999553822651630D0/, A(186)/0.001144950003186D0/
      DATA X(187)/0.997649864398237D0/, A(187)/0.002663533589512D0/
      DATA X(188)/0.994227540965688D0/, A(188)/0.004180313124694D0/
      DATA X(189)/0.989291302499755D0/, A(189)/0.005690922451403D0/
      DATA X(190)/0.982848572738629D0/, A(190)/0.007192904768117D0/
      DATA X(191)/0.974909140585727D0/, A(191)/0.008683945269260D0/
      DATA X(192)/0.965485089043799D0/, A(192)/0.010161766041103D0/
      DATA X(193)/0.954590766343634D0/, A(193)/0.011624114120797D0/
      DATA X(194)/0.942242761309872D0/, A(194)/0.013068761592401D0/
      DATA X(195)/0.928459877172445D0/, A(195)/0.014493508040509D0/
      DATA X(196)/0.913263102571757D0/, A(196)/0.015896183583725D0/
      DATA X(197)/0.896675579438770D0/, A(197)/0.017274652056269D0/
      DATA X(198)/0.878722567678213D0/, A(198)/0.018626814208299D0/
      DATA X(199)/0.859431406663111D0/, A(199)/0.019950610878141D0/
      DATA X(200)/0.838831473580255D0/, A(200)/0.021244026115782D0/
      DATA X(201)/0.816954138681463D0/, A(201)/0.022505090246332D0/
      DATA X(202)/0.793832717504605D0/, A(202)/0.023731882865930D0/
      DATA X(203)/0.769502420135041D0/, A(203)/0.024922535764115D0/
      DATA X(204)/0.744000297583597D0/, A(204)/0.026075235767565D0/
      DATA X(205)/0.717365185362099D0/, A(205)/0.027188227500486D0/
      DATA X(206)/0.689637644342027D0/, A(206)/0.028259816057276D0/
      DATA X(207)/0.660859898986119D0/, A(207)/0.029288369583267D0/
      DATA X(208)/0.631075773046871D0/, A(208)/0.030272321759557D0/
      DATA X(209)/0.600330622829751D0/, A(209)/0.031210174188114D0/
      DATA X(210)/0.568671268122709D0/, A(210)/0.032100498673487D0/
      DATA X(211)/0.536145920897131D0/, A(211)/0.032941939397645D0/
      DATA X(212)/0.502804111888784D0/, A(212)/0.033733214984611D0/
      DATA X(213)/0.468696615170544D0/, A(213)/0.034473120451753D0/
      DATA X(214)/0.433875370831756D0/, A(214)/0.035160529044747D0/
      DATA X(215)/0.398393405881969D0/, A(215)/0.035794393953416D0/
      DATA X(216)/0.362304753499487D0/, A(216)/0.036373749905835D0/
      DATA X(217)/0.325664370747701D0/, A(217)/0.036897714638276D0/
      DATA X(218)/0.288528054884511D0/, A(218)/0.037365490238730D0/
      DATA X(219)/0.250952358392272D0/, A(219)/0.037776364362001D0/
      DATA X(220)/0.212994502857666D0/, A(220)/0.038129711314477D0/
      DATA X(221)/0.174712291832646D0/, A(221)/0.038424993006959D0/
      DATA X(222)/0.136164022809143D0/, A(222)/0.038661759774076D0/
      DATA X(223)/0.097408398441584D0/, A(223)/0.038839651059051D0/
      DATA X(224)/0.058504437152420D0/, A(224)/0.038958395962769D0/
      DATA X(225)/0.019511383256793D0/, A(225)/0.039017813656306D0/
C-----N=96
      DATA X(226)/0.999689503883230D0/, A(226)/0.000796792065552D0/
      DATA X(227)/0.998364375863181D0/, A(227)/0.001853960788946D0/
      DATA X(228)/0.995981842987209D0/, A(228)/0.002910731817934D0/
      DATA X(229)/0.992543900323762D0/, A(229)/0.003964554338444D0/
      DATA X(230)/0.988054126329623D0/, A(230)/0.005014202742927D0/
      DATA X(231)/0.982517263563014D0/, A(231)/0.006058545504235D0/
      DATA X(232)/0.975939174585136D0/, A(232)/0.007096470791153D0/
      DATA X(233)/0.968326828463264D0/, A(233)/0.008126876925698D0/
      DATA X(234)/0.959688291448742D0/, A(234)/0.009148671230783D0/
      DATA X(235)/0.950032717784437D0/, A(235)/0.010160770535008D0/
      DATA X(236)/0.939370339752755D0/, A(236)/0.011162102099838D0/
      DATA X(237)/0.927712456722308D0/, A(237)/0.012151604671088D0/
      DATA X(238)/0.915071423120898D0/, A(238)/0.013128229566961D0/
      DATA X(239)/0.901460635315852D0/, A(239)/0.014090941772314D0/
      DATA X(240)/0.886894517402420D0/, A(240)/0.015038721026994D0/
      DATA X(241)/0.871388505909296D0/, A(241)/0.015970562902562D0/
      DATA X(242)/0.854959033434601D0/, A(242)/0.016885479864245D0/
      DATA X(243)/0.837623511228187D0/, A(243)/0.017782502316045D0/
      DATA X(244)/0.819400310737931D0/, A(244)/0.018660679627411D0/
      DATA X(245)/0.800308744139140D0/, A(245)/0.019519081140145D0/
      DATA X(246)/0.780369043867433D0/, A(246)/0.020356797154333D0/
      DATA X(247)/0.759602341176647D0/, A(247)/0.021172939892191D0/
      DATA X(248)/0.738030643744400D0/, A(248)/0.021966644438744D0/
      DATA X(249)/0.715676812348967D0/, A(249)/0.022737069658329D0/
      DATA X(250)/0.692564536642171D0/, A(250)/0.023483399085926D0/
      DATA X(251)/0.668718310043916D0/, A(251)/0.024204841792364D0/
      DATA X(252)/0.644163403784967D0/, A(252)/0.024900633222483D0/
      DATA X(253)/0.618925840125468D0/, A(253)/0.025570036005349D0/
      DATA X(254)/0.593032364777572D0/, A(254)/0.026212340735672D0/
      DATA X(255)/0.566510418561397D0/, A(255)/0.026826866725591D0/
      DATA X(256)/0.539388108324357D0/, A(256)/0.027412962726029D0/
      DATA X(257)/0.511694177154667D0/, A(257)/0.027970007616848D0/
      DATA X(258)/0.483457973920596D0/, A(258)/0.028497411065085D0/
      DATA X(259)/0.454709422167743D0/, A(259)/0.028994614150555D0/
      DATA X(260)/0.425478988407300D0/, A(260)/0.029461089958167D0/
      DATA X(261)/0.395797649828908D0/, A(261)/0.029896344136328D0/
      DATA X(262)/0.365696861472313D0/, A(262)/0.030299915420827D0/
      DATA X(263)/0.335208522892625D0/, A(263)/0.030671376123669D0/
      DATA X(264)/0.304364944354496D0/, A(264)/0.031010332586313D0/
      DATA X(265)/0.273198812591049D0/, A(265)/0.031316425596861D0/
      DATA X(266)/0.241743156163840D0/, A(266)/0.031589330770727D0/
      DATA X(267)/0.210031310460567D0/, A(267)/0.031828758894411D0/
      DATA X(268)/0.178096882367618D0/, A(268)/0.032034456231992D0/
      DATA X(269)/0.145973714654896D0/, A(269)/0.032206204794030D0/
      DATA X(270)/0.113695850110665D0/, A(270)/0.032343822568575D0/
      DATA X(271)/0.081297495464425D0/, A(271)/0.032447163714064D0/
      DATA X(272)/0.048812985136049D0/, A(272)/0.032516118713868D0/
      DATA X(273)/0.016276744849602D0/, A(273)/0.032550614492363D0/
      DATA IBD/0/
      IF(IBD.NE.0) RETURN
      IBD=1
      DO 10 I=1,273
        B(I) = A(I)
10      Y(I) = X(I)
      DO 20 I=1,96
20      LTAB(I) = KTAB(I)
      RETURN
      END
