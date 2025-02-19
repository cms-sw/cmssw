C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
C     KTCLUS: written by Mike Seymour, July 1992.
C     Last modified November 2000.
C     Please send comments or suggestions to Mike.Seymour@rl.ac.uk
C
C     This is a general-purpose kt clustering package.
C     It can handle ee, ep and pp collisions.
C     It is loosely based on the program of Siggi Bethke.
C
C     The time taken (on a 10MIP machine) is (0.2microsec)*N**3
C     where N is the number of particles.
C     Over 90 percent of this time is used in subroutine KTPMIN, which
C     simply finds the minimum member of a one-dimensional array.
C     It is well worth thinking about optimization: on the SPARCstation
C     a factor of two increase was obtained simply by increasing the
C     optimization level from its default value.
C
C     The approach is to separate the different stages of analysis.
C     KTCLUS does all the clustering and records a merging history.
C     It returns a simple list of the y values at which each merging
C     occured. Then the following routines can be called to give extra
C     information on the most recently analysed event.
C     KTCLUR is identical but includes an R parameter, see below.
C     KTYCUT gives the number of jets at each given YCUT value.
C     KTYSUB gives the number of sub-jets at each given YCUT value.
C     KTBEAM gives same info as KTCLUS but only for merges with the beam
C     KTJOIN gives same info as KTCLUS but for merges of sub-jets.
C     KTRECO reconstructs the jet momenta at a given value of YCUT.
C     It also gives information on which jets at scale YCUT belong to
C     which macro-jets at scale YMAC, for studying sub-jet properties.
C     KTINCL reconstructs the jet momenta according to the inclusive jet
C     definition of Ellis and Soper.
C     KTISUB, KTIJOI and KTIREC are like KTYSUB, KTJOIN and KTRECO,
C     except that they only apply to one inclusive jet at a time,
C     with the pt of that jet automatically used for ECUT.
C     KTWICH gives a list of which particles ended up in which jets.
C     KTWCHS gives the same thing, but only for subjets.
C     Note that the numbering of jets used by these two routines is
C     guaranteed to be the same as that used by KTRECO.
C
C     The collision type and analysis type are indicated by the first
C     argument of KTCLUS. IMODE=<TYPE><ANGLE><MONO><RECOM> where
C     TYPE:  1=>ee, 2=>ep with p in -z direction, 3=>pe, 4=>pp
C     ANGLE: 1=>angular kt def., 2=>DeltaR, 3=>f(DeltaEta,DeltaPhi)
C            where f()=2(cosh(eta)-cos(phi)) is the QCD emission metric
C     MONO:  1=>derive relative pseudoparticle angles from jets
C            2=>monotonic definitions of relative angles
C     RECOM: 1=>E recombination scheme, 2=>pt scheme, 3=>pt**2 scheme
C
C     There are also abbreviated forms for the most common combinations:
C     IMODE=1 => E scheme in e+e-                              (=1111)
C           2 => E scheme in ep                                (=2111)
C           3 => E scheme in pe                                (=3111)
C           4 => E scheme in pp                                (=4111)
C           5 => covariant E scheme in pp                      (=4211)
C           6 => covariant pt-scheme in pp                     (=4212)
C           7 => covariant monotonic pt**2-scheme in pp        (=4223)
C
C     KTRECO no longer needs to reconstruct the momenta according to the
C     same recombination scheme in which they were clustered. Its first
C     argument gives the scheme, taking the same values as RECOM above.
C
C     Note that unlike previous versions, all variables which hold y
C     values have been named in a consistent way:
C     Y()  is the output scale at which jets were merged,
C     YCUT is the input scale at which jets should be counted, and
C          jet-momenta reconstructed etc,
C     YMAC is the input macro-jet scale, used in determining whether
C          or not each jet is a sub-jet.
C     The original scheme defined in our papers is equivalent to always
C     setting YMAC=1.
C     Whenever a YCUT or YMAC variable is used, it is rounded down
C     infinitesimally, so that for example, setting YCUT=Y(2) refers
C     to the scale where the event is 2-jet, even if rounding errors
C     have shifted its value slightly.
C
C     An R parameter can be used in hadron-hadron collisions by
C     calling KTCLUR instead of KTCLUS.  This is as suggested by
C     Ellis and Soper, but implemented slightly differently,
C     as in M.H. Seymour, LU TP 94/2 (submitted to Nucl. Phys. B.).
C     R**2 multiplies the single Kt everywhere it is used.
C     Calling KTCLUR with R=1 is identical to calling KTCLUS.
C     R plays a similar role to the jet radius in a cone-type algorithm,
C     but is scaled up by about 40% (ie R=0.7 in a cone algorithm is
C     similar to this algorithm with R=1).
C     Note that R.EQ.1 must be used for the e+e- and ep versions,
C     and is strongly recommended for the hadron-hadron version.
C     However, R values smaller than 1 have been found to be useful for
C     certain applications, particularly the mass reconstruction of
C     highly-boosted colour-singlets such as high-pt hadronic Ws,
C     as in M.H. Seymour, LU TP 93/8 (to appear in Z. Phys. C.).
C     Situations in which R<1 is useful are likely to also be those in
C     which the inclusive reconstruction method is more useful.
C
C     Also included is a set of routines for doing Lorentz boosts:
C     KTLBST finds the boost matrix to/from the cm frame of a 4-vector
C     KTRROT finds the rotation matrix from one vector to another
C     KTMMUL multiplies together two matrices
C     KTVMUL multiplies a vector by a matrix
C     KTINVT inverts a transformation matrix (nb NOT a general 4 by 4)
C     KTFRAM boosts a list of vectors between two arbitrary frames
C     KTBREI boosts a list of vectors between the lab and Breit frames
C     KTHADR boosts a list of vectors between the lab and hadronic cmf
C       The last two need the momenta in the +z direction of the lepton
C       and hadron beams, and the 4-momentum of the outgoing lepton.
C
C     The main reference is:
C       S. Catani, Yu.L. Dokshitzer, M.H. Seymour and B.R. Webber,
C         Nucl.Phys.B406(1993)187.
C     The ep version was proposed in:
C       S. Catani, Yu.L. Dokshitzer and B.R. Webber,
C         Phys.Lett.285B(1992)291.
C     The inclusive reconstruction method was proposed in:
C       S.D. Ellis and D.E. Soper,
C         Phys.Rev.D48(1993)3160.
C
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
      SUBROUTINE KTCLUS(IMODE,PP,NN,ECUT,Y,*)
      IMPLICIT NONE
C---DO CLUSTER ANALYSIS OF PARTICLES IN PP
C
C   IMODE   = INPUT  : DESCRIBED ABOVE
C   PP(I,J) = INPUT  : 4-MOMENTUM OF Jth PARTICLE: I=1,4 => PX,PY,PZ,E
C   NN      = INPUT  : NUMBER OF PARTICLES
C   ECUT    = INPUT  : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   Y(J)    = OUTPUT : VALUE OF Y FOR WHICH EVENT CHANGES FROM BEING
C                        J JET TO J-1 JET
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED (MOST LIKELY DUE TO TOO MANY PARTICLES)
C
C   NOTE THAT THE MOMENTA ARE DECLARED DOUBLE PRECISION,
C   AND ALL OTHER FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER IMODE,NN
      DOUBLE PRECISION PP(4,*)
      DOUBLE PRECISION ECUT,Y(*),ONE
      ONE=1
      CALL KTCLUR(IMODE,PP,NN,ONE,ECUT,Y,*999)
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTCLUR(IMODE,PP,NN,R,ECUT,Y,*)
      IMPLICIT NONE
C---DO CLUSTER ANALYSIS OF PARTICLES IN PP
C
C   IMODE   = INPUT  : DESCRIBED ABOVE
C   PP(I,J) = INPUT  : 4-MOMENTUM OF Jth PARTICLE: I=1,4 => PX,PY,PZ,E
C   NN      = INPUT  : NUMBER OF PARTICLES
C   R       = INPUT  : ELLIS AND SOPER'S R PARAMETER, SEE ABOVE.
C   ECUT    = INPUT  : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   Y(J)    = OUTPUT : VALUE OF Y FOR WHICH EVENT CHANGES FROM BEING
C                        J JET TO J-1 JET
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED (MOST LIKELY DUE TO TOO MANY PARTICLES)
C
C   NOTE THAT THE MOMENTA ARE DECLARED DOUBLE PRECISION,
C   AND ALL OTHER FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,IM,IMODE,TYPE,ANGL,MONO,RECO,N,I,J,NN,
     &     IMIN,JMIN,KMIN,NUM,HIST,INJET,IABBR,NABBR
      PARAMETER (NMAX=512,NABBR=7)
      DOUBLE PRECISION PP(4,*)
      DOUBLE PRECISION R,ECUT,Y(*),P,KT,ETOT,RSQ,KTP,KTS,KTPAIR,KTSING,
     &     KTMIN,ETSQ,KTLAST,KTMAX,KTTMP
      LOGICAL FIRST
      CHARACTER TITLE(4,4)*10
C---KT RECORDS THE KT**2 OF EACH MERGING.
C---KTLAST RECORDS FOR EACH MERGING, THE HIGHEST ECUT**2 FOR WHICH THE
C   RESULT IS NOT MERGED WITH THE BEAM (COULD BE LARGER THAN THE
C   KT**2 AT WHICH IT WAS MERGED IF THE KT VALUES ARE NOT MONOTONIC).
C   THIS MAY SOUND POINTLESS, BUT ITS USEFUL FOR DETERMINING WHETHER
C   SUB-JETS SURVIVED TO SCALE Y=YMAC OR NOT.
C---HIST RECORDS MERGING HISTORY:
C   N=>DELETED TRACK N, M*NMAX+N=>MERGED TRACKS M AND N (M<N).
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      DIMENSION INJET(NMAX),IABBR(NABBR)
      DATA FIRST,TITLE,IABBR/.TRUE.,
     &     'e+e-      ','ep        ','pe        ','pp        ',
     &     'angle     ','DeltaR    ','f(DeltaR) ','**********',
     &     'no        ','yes       ','**********','**********',
     &     'E         ','Pt        ','Pt**2     ','**********',
     &     1111,2111,3111,4111,4211,4212,4223/
C---CHECK INPUT
      IM=IMODE
      IF (IM.GE.1.AND.IM.LE.NABBR) IM=IABBR(IM)
      TYPE=MOD(IM/1000,10)
      ANGL=MOD(IM/100 ,10)
      MONO=MOD(IM/10  ,10)
      RECO=MOD(IM     ,10)
      IF (NN.GT.NMAX.OR.NN.LT.1.OR.(NN.LT.2.AND.TYPE.EQ.1))
     &     CALL KTWARN('KTCLUS',100,*999)
      IF (TYPE.LT.1.OR.TYPE.GT.4.OR.ANGL.LT.1.OR.ANGL.GT.4.OR.
     &    MONO.LT.1.OR.MONO.GT.2.OR.RECO.LT.1.OR.RECO.GT.3)
     &     CALL KTWARN('KTCLUS',101,*999)
      IF (FIRST) THEN
         WRITE (6,'(/,1X,54(''*'')/A)')
     &   ' KTCLUS: written by Mike Seymour, July 1992.'
         WRITE (6,'(A)')
     &   ' Last modified November 2000.'
         WRITE (6,'(A)')
     &   ' Please send comments or suggestions to Mike.Seymour@rl.ac.uk'
         WRITE (6,'(/A,I2,2A)')
     &   '       Collision type =',TYPE,' = ',TITLE(TYPE,1)
         WRITE (6,'(A,I2,2A)')
     &   '     Angular variable =',ANGL,' = ',TITLE(ANGL,2)
         WRITE (6,'(A,I2,2A)')
     &   ' Monotonic definition =',MONO,' = ',TITLE(MONO,3)
         WRITE (6,'(A,I2,2A)')
     &   ' Recombination scheme =',RECO,' = ',TITLE(RECO,4)
         IF (R.NE.1) THEN
         WRITE (6,'(A,F5.2)')
     &   '     Radius parameter =',R
         IF (TYPE.NE.4) WRITE (6,'(A)')
     &   ' R.NE.1 is strongly discouraged for this collision type!'
         ENDIF
         WRITE (6,'(1X,54(''*'')/)')
         FIRST=.FALSE.
      ENDIF
C---COPY PP TO P
      N=NN
      NUM=NN
      CALL KTCOPY(PP,N,P,(RECO.NE.1))
      ETOT=0
      DO 100 I=1,N
         ETOT=ETOT+P(4,I)
 100  CONTINUE
      IF (ETOT.EQ.0) CALL KTWARN('KTCLUS',102,*999)
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
      RSQ=R**2
C---CALCULATE ALL PAIR KT's
      DO 210 I=1,N-1
         DO 200 J=I+1,N
            KTP(J,I)=-1
            KTP(I,J)=KTPAIR(ANGL,P(1,I),P(1,J),KTP(J,I))
 200     CONTINUE
 210  CONTINUE
C---CALCULATE ALL SINGLE KT's
      DO 230 I=1,N
         KTS(I)=KTSING(ANGL,TYPE,P(1,I))
 230  CONTINUE
      KTMAX=0
C---MAIN LOOP
 300  CONTINUE
C---FIND MINIMUM MEMBER OF KTP
      CALL KTPMIN(KTP,NMAX,N,IMIN,JMIN)
C---FIND MINIMUM MEMBER OF KTS
      CALL KTSMIN(KTS,NMAX,N,KMIN)
C---STORE Y VALUE OF TRANSITION FROM N TO N-1 JETS
      KTMIN=KTP(IMIN,JMIN)
      KTTMP=RSQ*KTS(KMIN)
      IF ((TYPE.GE.2.AND.TYPE.LE.4).AND.
     &     (KTTMP.LE.KTMIN.OR.N.EQ.1))
     &     KTMIN=KTTMP
      KT(N)=KTMIN
      Y(N)=KT(N)*ETSQ
C---IF MONO.GT.1, SEQUENCE IS SUPPOSED TO BE MONOTONIC, IF NOT, WARN
      IF (KTMIN.LT.KTMAX.AND.MONO.GT.1) CALL KTWARN('KTCLUS',1,*999)
      IF (KTMIN.GE.KTMAX) KTMAX=KTMIN
C---IF LOWEST KT IS TO A BEAM, THROW IT AWAY AND MOVE LAST ENTRY UP
      IF (KTMIN.EQ.KTTMP) THEN
         CALL KTMOVE(P,KTP,KTS,NMAX,N,KMIN,1)
C---UPDATE HISTORY AND CROSS-REFERENCES
         HIST(N)=KMIN
         INJET(N)=KMIN
         DO 400 I=N,NN
            IF (INJET(I).EQ.KMIN) THEN
               KTLAST(I)=KTMAX
               INJET(I)=0
            ELSEIF (INJET(I).EQ.N) THEN
               INJET(I)=KMIN
            ENDIF
 400     CONTINUE
C---OTHERWISE MERGE JETS IMIN AND JMIN AND MOVE LAST ENTRY UP
      ELSE
         CALL KTMERG(P,KTP,KTS,NMAX,IMIN,JMIN,N,TYPE,ANGL,MONO,RECO)
         CALL KTMOVE(P,KTP,KTS,NMAX,N,JMIN,1)
C---UPDATE HISTORY AND CROSS-REFERENCES
         HIST(N)=IMIN*NMAX+JMIN
         INJET(N)=IMIN
         DO 600 I=N,NN
            IF (INJET(I).EQ.JMIN) THEN
               INJET(I)=IMIN
            ELSEIF (INJET(I).EQ.N) THEN
               INJET(I)=JMIN
            ENDIF
 600     CONTINUE
      ENDIF
C---THATS ALL THERE IS TO IT
      N=N-1
      IF (N.GT.1 .OR. N.GT.0.AND.(TYPE.GE.2.AND.TYPE.LE.4)) GOTO 300
      IF (N.EQ.1) THEN
         KT(N)=1D20
         Y(N)=KT(N)*ETSQ
      ENDIF
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTYCUT(ECUT,NY,YCUT,NJET,*)
      IMPLICIT NONE
C---COUNT THE NUMBER OF JETS AT EACH VALUE OF YCUT, FOR EVENT WHICH HAS
C   ALREADY BEEN ANALYSED BY KTCLUS.
C
C   ECUT    = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   NY      = INPUT : NUMBER OF YCUT VALUES
C   YCUT(J) = INPUT : Y VALUES AT WHICH NUMBERS OF JETS ARE COUNTED
C   NJET(J) =OUTPUT : NUMBER OF JETS AT YCUT(J)
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NY,NJET(NY),NMAX,HIST,I,J,NUM
      PARAMETER (NMAX=512)
      DOUBLE PRECISION YCUT(NY),ETOT,RSQ,P,KT,KTP,KTS,ETSQ,ECUT,KTLAST,
     &     ROUND
      PARAMETER (ROUND=0.99999D0)
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      IF (ETOT.EQ.0) CALL KTWARN('KTYCUT',100,*999)
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
      DO 100 I=1,NY
         NJET(I)=0
 100  CONTINUE
      DO 210 I=NUM,1,-1
         DO 200 J=1,NY
            IF (NJET(J).EQ.0.AND.KT(I)*ETSQ.GE.ROUND*YCUT(J)) NJET(J)=I
 200     CONTINUE
 210  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTYSUB(ECUT,NY,YCUT,YMAC,NSUB,*)
      IMPLICIT NONE
C---COUNT THE NUMBER OF SUB-JETS AT EACH VALUE OF YCUT, FOR EVENT WHICH
C   HAS ALREADY BEEN ANALYSED BY KTCLUS.
C   REMEMBER THAT A SUB-JET IS DEFINED AS A JET AT Y=YCUT WHICH HAS NOT
C   YET BEEN MERGED WITH THE BEAM AT Y=YMAC.
C
C   ECUT    = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   NY      = INPUT : NUMBER OF YCUT VALUES
C   YCUT(J) = INPUT : Y VALUES AT WHICH NUMBERS OF SUB-JETS ARE COUNTED
C   YMAC    = INPUT : Y VALUE USED TO DEFINE MACRO-JETS, TO DETERMINE
C                       WHICH JETS ARE SUB-JETS
C   NSUB(J) =OUTPUT : NUMBER OF SUB-JETS AT YCUT(J)
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NY,NSUB(NY),NMAX,HIST,I,J,NUM
      PARAMETER (NMAX=512)
      DOUBLE PRECISION YCUT(NY),YMAC,ETOT,RSQ,P,KT,KTP,KTS,ETSQ,ECUT,
     &     KTLAST,ROUND
      PARAMETER (ROUND=0.99999D0)
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      IF (ETOT.EQ.0) CALL KTWARN('KTYSUB',100,*999)
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
      DO 100 I=1,NY
         NSUB(I)=0
 100  CONTINUE
      DO 210 I=NUM,1,-1
         DO 200 J=1,NY
            IF (NSUB(J).EQ.0.AND.KT(I)*ETSQ.GE.ROUND*YCUT(J)) NSUB(J)=I
            IF (NSUB(J).NE.0.AND.KTLAST(I)*ETSQ.LT.ROUND*YMAC)
     &          NSUB(J)=NSUB(J)-1
 200     CONTINUE
 210  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTBEAM(ECUT,Y,*)
      IMPLICIT NONE
C---GIVE SAME INFORMATION AS LAST CALL TO KTCLUS EXCEPT THAT ONLY
C   TRANSITIONS WHERE A JET WAS MERGED WITH THE BEAM JET ARE RECORDED
C
C   ECUT    = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   Y(J)    =OUTPUT : Y VALUE WHERE Jth HARDEST JET WAS MERGED WITH BEAM
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,HIST,NUM,I,J
      PARAMETER (NMAX=512)
      DOUBLE PRECISION ETOT,RSQ,P,KT,KTP,KTS,ECUT,ETSQ,Y(*),KTLAST
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      IF (ETOT.EQ.0) CALL KTWARN('KTBEAM',100,*999)
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
      J=1
      DO 100 I=1,NUM
         IF (HIST(I).LE.NMAX) THEN
            Y(J)=ETSQ*KT(I)
            J=J+1
         ENDIF
 100  CONTINUE
      DO 200 I=J,NUM
         Y(I)=0
 200  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTJOIN(ECUT,YMAC,Y,*)
      IMPLICIT NONE
C---GIVE SAME INFORMATION AS LAST CALL TO KTCLUS EXCEPT THAT ONLY
C   TRANSITIONS WHERE TWO SUB-JETS WERE JOINED ARE RECORDED
C   REMEMBER THAT A SUB-JET IS DEFINED AS A JET AT Y=YCUT WHICH HAS NOT
C   YET BEEN MERGED WITH THE BEAM AT Y=YMAC.
C
C   ECUT    = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   YMAC    = INPUT : VALUE OF Y USED TO DEFINE MACRO-JETS
C   Y(J)    =OUTPUT : Y VALUE WHERE EVENT CHANGED FROM HAVING
C                         N+J SUB-JETS TO HAVING N+J-1, WHERE N IS
C                         THE NUMBER OF MACRO-JETS AT SCALE YMAC
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,HIST,NUM,I,J
      PARAMETER (NMAX=512)
      DOUBLE PRECISION ETOT,RSQ,P,KT,KTP,KTS,ECUT,ETSQ,Y(*),YMAC,KTLAST,
     &     ROUND
      PARAMETER (ROUND=0.99999D0)
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      IF (ETOT.EQ.0) CALL KTWARN('KTJOIN',100,*999)
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
      J=1
      DO 100 I=1,NUM
         IF (HIST(I).GT.NMAX.AND.ETSQ*KTLAST(I).GE.ROUND*YMAC) THEN
            Y(J)=ETSQ*KT(I)
            J=J+1
         ENDIF
 100  CONTINUE
      DO 200 I=J,NUM
         Y(I)=0
 200  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTRECO(RECO,PP,NN,ECUT,YCUT,YMAC,PJET,JET,NJET,NSUB,*)
      IMPLICIT NONE
C---RECONSTRUCT KINEMATICS OF JET SYSTEM, WHICH HAS ALREADY BEEN
C   ANALYSED BY KTCLUS. NOTE THAT NO CONSISTENCY CHECK IS MADE: USER
C   IS TRUSTED TO USE THE SAME PP VALUES AS FOR KTCLUS
C
C   RECO     = INPUT : RECOMBINATION SCHEME (NEED NOT BE SAME AS KTCLUS)
C   PP(I,J)  = INPUT : 4-MOMENTUM OF Jth PARTICLE: I=1,4 => PX,PY,PZ,E
C   NN       = INPUT : NUMBER OF PARTICLES
C   ECUT     = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   YCUT     = INPUT : Y VALUE AT WHICH TO RECONSTRUCT JET MOMENTA
C   YMAC     = INPUT : Y VALUE USED TO DEFINE MACRO-JETS, TO DETERMINE
C                        WHICH JETS ARE SUB-JETS
C   PJET(I,J)=OUTPUT : 4-MOMENTUM OF Jth JET AT SCALE YCUT
C   JET(J)   =OUTPUT : THE MACRO-JET WHICH CONTAINS THE Jth JET,
C                        SET TO ZERO IF JET IS NOT A SUB-JET
C   NJET     =OUTPUT : THE NUMBER OF JETS
C   NSUB     =OUTPUT : THE NUMBER OF SUB-JETS (EQUAL TO THE NUMBER OF
C                        NON-ZERO ENTRIES IN JET())
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT THE MOMENTA ARE DECLARED DOUBLE PRECISION,
C   AND ALL OTHER FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,RECO,NUM,N,NN,NJET,NSUB,JET(*),HIST,IMIN,JMIN,I,J
      PARAMETER (NMAX=512)
      DOUBLE PRECISION PP(4,*),PJET(4,*)
      DOUBLE PRECISION ECUT,P,KT,KTP,KTS,ETOT,RSQ,ETSQ,YCUT,YMAC,KTLAST,
     &     ROUND
      PARAMETER (ROUND=0.99999D0)
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
C---CHECK INPUT
      IF (RECO.LT.1.OR.RECO.GT.3) THEN
        PRINT *,'RECO=',RECO
        CALL KTWARN('KTRECO',100,*999)
      ENDIF
C---COPY PP TO P
      N=NN
      IF (NUM.NE.NN) CALL KTWARN('KTRECO',101,*999)
      CALL KTCOPY(PP,N,P,(RECO.NE.1))
      IF (ECUT.EQ.0) THEN
         ETSQ=1/ETOT**2
      ELSE
         ETSQ=1/ECUT**2
      ENDIF
C---KEEP MERGING UNTIL YCUT
 100  IF (ETSQ*KT(N).LT.ROUND*YCUT) THEN
         IF (HIST(N).LE.NMAX) THEN
            CALL KTMOVE(P,KTP,KTS,NMAX,N,HIST(N),0)
         ELSE
            IMIN=HIST(N)/NMAX
            JMIN=HIST(N)-IMIN*NMAX
            CALL KTMERG(P,KTP,KTS,NMAX,IMIN,JMIN,N,0,0,0,RECO)
            CALL KTMOVE(P,KTP,KTS,NMAX,N,JMIN,0)
         ENDIF
         N=N-1
         IF (N.GT.0) GOTO 100
      ENDIF
C---IF YCUT IS TOO LARGE THERE ARE NO JETS
      NJET=N
      NSUB=N
      IF (N.EQ.0) RETURN
C---SET UP OUTPUT MOMENTA
      DO 210 I=1,NJET
         IF (RECO.EQ.1) THEN
            DO 200 J=1,4
               PJET(J,I)=P(J,I)
 200        CONTINUE
         ELSE
            PJET(1,I)=P(6,I)*COS(P(8,I))
            PJET(2,I)=P(6,I)*SIN(P(8,I))
            PJET(3,I)=P(6,I)*SINH(P(7,I))
            PJET(4,I)=P(6,I)*COSH(P(7,I))
         ENDIF
         JET(I)=I
 210  CONTINUE
C---KEEP MERGING UNTIL YMAC TO FIND THE FATE OF EACH JET
 300  IF (ETSQ*KT(N).LT.ROUND*YMAC) THEN
         IF (HIST(N).LE.NMAX) THEN
            IMIN=0
            JMIN=HIST(N)
            NSUB=NSUB-1
         ELSE
            IMIN=HIST(N)/NMAX
            JMIN=HIST(N)-IMIN*NMAX
            IF (ETSQ*KTLAST(N).LT.ROUND*YMAC) NSUB=NSUB-1
         ENDIF
         DO 310 I=1,NJET
            IF (JET(I).EQ.JMIN) JET(I)=IMIN
            IF (JET(I).EQ.N) JET(I)=JMIN
 310     CONTINUE
         N=N-1
         IF (N.GT.0) GOTO 300
      ENDIF
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTINCL(RECO,PP,NN,PJET,JET,NJET,*)
      IMPLICIT NONE
C---RECONSTRUCT KINEMATICS OF JET SYSTEM, WHICH HAS ALREADY BEEN
C   ANALYSED BY KTCLUS ACCORDING TO THE INCLUSIVE JET DEFINITION. NOTE
C   THAT NO CONSISTENCY CHECK IS MADE: USER IS TRUSTED TO USE THE SAME
C   PP VALUES AS FOR KTCLUS
C
C   RECO     = INPUT : RECOMBINATION SCHEME (NEED NOT BE SAME AS KTCLUS)
C   PP(I,J)  = INPUT : 4-MOMENTUM OF Jth PARTICLE: I=1,4 => PX,PY,PZ,E
C   NN       = INPUT : NUMBER OF PARTICLES
C   PJET(I,J)=OUTPUT : 4-MOMENTUM OF Jth JET AT SCALE YCUT
C   JET(J)   =OUTPUT : THE JET WHICH CONTAINS THE Jth PARTICLE
C   NJET     =OUTPUT : THE NUMBER OF JETS
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT THE MOMENTA ARE DECLARED DOUBLE PRECISION,
C   AND ALL OTHER FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,RECO,NUM,N,NN,NJET,JET(*),HIST,IMIN,JMIN,I,J
      PARAMETER (NMAX=512)
      DOUBLE PRECISION PP(4,*),PJET(4,*)
      DOUBLE PRECISION P,KT,KTP,KTS,ETOT,RSQ,KTLAST
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
C---CHECK INPUT
      IF (RECO.LT.1.OR.RECO.GT.3) CALL KTWARN('KTINCL',100,*999)
C---COPY PP TO P
      N=NN
      IF (NUM.NE.NN) CALL KTWARN('KTINCL',101,*999)
      CALL KTCOPY(PP,N,P,(RECO.NE.1))
C---INITIALLY EVERY PARTICLE IS IN ITS OWN JET
      DO 100 I=1,NN
         JET(I)=I
 100  CONTINUE
C---KEEP MERGING TO THE BITTER END
      NJET=0
 200  IF (N.GT.0) THEN
         IF (HIST(N).LE.NMAX) THEN
            IMIN=0
            JMIN=HIST(N)
            NJET=NJET+1
            IF (RECO.EQ.1) THEN
               DO 300 J=1,4
                  PJET(J,NJET)=P(J,JMIN)
 300           CONTINUE
            ELSE
               PJET(1,NJET)=P(6,JMIN)*COS(P(8,JMIN))
               PJET(2,NJET)=P(6,JMIN)*SIN(P(8,JMIN))
               PJET(3,NJET)=P(6,JMIN)*SINH(P(7,JMIN))
               PJET(4,NJET)=P(6,JMIN)*COSH(P(7,JMIN))
            ENDIF
            CALL KTMOVE(P,KTP,KTS,NMAX,N,JMIN,0)
         ELSE
            IMIN=HIST(N)/NMAX
            JMIN=HIST(N)-IMIN*NMAX
            CALL KTMERG(P,KTP,KTS,NMAX,IMIN,JMIN,N,0,0,0,RECO)
            CALL KTMOVE(P,KTP,KTS,NMAX,N,JMIN,0)
         ENDIF
         DO 400 I=1,NN
            IF (JET(I).EQ.JMIN) JET(I)=IMIN
            IF (JET(I).EQ.N) JET(I)=JMIN
            IF (JET(I).EQ.0) JET(I)=-NJET
 400     CONTINUE
         N=N-1
         GOTO 200
      ENDIF
C---FINALLY EVERY PARTICLE MUST BE IN AN INCLUSIVE JET
      DO 500 I=1,NN
C---IF THERE ARE ANY UNASSIGNED PARTICLES SOMETHING MUST HAVE GONE WRONG
         IF (JET(I).GE.0) CALL KTWARN('KTINCL',102,*999)
         JET(I)=-JET(I)
 500  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTISUB(N,NY,YCUT,NSUB,*)
      IMPLICIT NONE
C---COUNT THE NUMBER OF SUB-JETS IN THE Nth INCLUSIVE JET OF AN EVENT
C   THAT HAS ALREADY BEEN ANALYSED BY KTCLUS.
C
C   N       = INPUT : WHICH INCLUSIVE JET TO USE
C   NY      = INPUT : NUMBER OF YCUT VALUES
C   YCUT(J) = INPUT : Y VALUES AT WHICH NUMBERS OF SUB-JETS ARE COUNTED
C   NSUB(J) =OUTPUT : NUMBER OF SUB-JETS AT YCUT(J)
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER N,NY,NSUB(NY),NMAX,HIST,I,J,NUM,NM
      PARAMETER (NMAX=512)
      DOUBLE PRECISION YCUT(NY),ETOT,RSQ,P,KT,KTP,KTS,KTLAST,ROUND,EPS
      PARAMETER (ROUND=0.99999D0)
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      DATA EPS/1D-6/
      DO 100 I=1,NY
         NSUB(I)=0
 100  CONTINUE
C---FIND WHICH MERGING CORRESPONDS TO THE NTH INCLUSIVE JET
      NM=0
      J=0
      DO 110 I=NUM,1,-1
        IF (HIST(I).LE.NMAX) J=J+1
        IF (J.EQ.N) THEN
          NM=I
          GOTO 120
        ENDIF
 110  CONTINUE
 120  CONTINUE
C---GIVE UP IF THERE ARE LESS THAN N INCLUSIVE JETS
      IF (NM.EQ.0) CALL KTWARN('KTISUB',100,*999)
      DO 210 I=NUM,1,-1
         DO 200 J=1,NY
            IF (NSUB(J).EQ.0.AND.RSQ*KT(I).GE.ROUND*YCUT(J)*KT(NM))
     &          NSUB(J)=I
            IF (NSUB(J).NE.0.AND.ABS(KTLAST(I)-KTLAST(NM)).GT.EPS)
     &          NSUB(J)=NSUB(J)-1
 200     CONTINUE
 210  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTIJOI(N,Y,*)
      IMPLICIT NONE
C---GIVE SAME INFORMATION AS LAST CALL TO KTCLUS EXCEPT THAT ONLY
C   MERGES OF TWO SUB-JETS INSIDE THE Nth INCLUSIVE JET ARE RECORDED
C
C   N       = INPUT : WHICH INCLUSIVE JET TO USE
C   Y(J)    =OUTPUT : Y VALUE WHERE JET CHANGED FROM HAVING
C                         J+1 SUB-JETS TO HAVING J
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,HIST,NUM,I,J,N,NM
      PARAMETER (NMAX=512)
      DOUBLE PRECISION ETOT,RSQ,P,KT,KTP,KTS,Y(*),KTLAST,EPS
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      DATA EPS/1D-6/
C---FIND WHICH MERGING CORRESPONDS TO THE NTH INCLUSIVE JET
      NM=0
      J=0
      DO 100 I=NUM,1,-1
        IF (HIST(I).LE.NMAX) J=J+1
        IF (J.EQ.N) THEN
          NM=I
          GOTO 105
        ENDIF
 100  CONTINUE
 105  CONTINUE
C---GIVE UP IF THERE ARE LESS THAN N INCLUSIVE JETS
      IF (NM.EQ.0) CALL KTWARN('KTIJOI',100,*999)
      J=1
      DO 110 I=1,NUM
         IF (HIST(I).GT.NMAX.AND.ABS(KTLAST(I)-KTLAST(NM)).LT.EPS) THEN
            Y(J)=RSQ*KT(I)/KT(NM)
            J=J+1
         ENDIF
 110  CONTINUE
      DO 200 I=J,NUM
         Y(I)=0
 200  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTIREC(RECO,PP,NN,N,YCUT,PSUB,NSUB,*)
      IMPLICIT NONE
C---RECONSTRUCT KINEMATICS OF SUB-JET SYSTEM IN THE Nth INCLUSIVE JET
C   OF AN EVENT THAT HAS ALREADY BEEN ANALYSED BY KTCLUS
C
C   RECO     = INPUT : RECOMBINATION SCHEME (NEED NOT BE SAME AS KTCLUS)
C   PP(I,J)  = INPUT : 4-MOMENTUM OF Jth PARTICLE: I=1,4 => PX,PY,PZ,E
C   NN       = INPUT : NUMBER OF PARTICLES
C   N        = INPUT : WHICH INCLUSIVE JET TO USE
C   YCUT     = INPUT : Y VALUE AT WHICH TO RECONSTRUCT JET MOMENTA
C   PSUB(I,J)=OUTPUT : 4-MOMENTUM OF Jth SUB-JET AT SCALE YCUT
C   NSUB     =OUTPUT : THE NUMBER OF SUB-JETS
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT THE MOMENTA ARE DECLARED DOUBLE PRECISION,
C   AND ALL OTHER FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,RECO,NUM,NN,NJET,NSUB,JET,HIST,I,J,N,NM
      PARAMETER (NMAX=512)
      DOUBLE PRECISION PP(4,*),PSUB(4,*)
      DOUBLE PRECISION ECUT,P,KT,KTP,KTS,ETOT,RSQ,YCUT,YMAC,KTLAST
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      DIMENSION JET(NMAX)
C---FIND WHICH MERGING CORRESPONDS TO THE NTH INCLUSIVE JET
      NM=0
      J=0
      DO 100 I=NUM,1,-1
         IF (HIST(I).LE.NMAX) J=J+1
         IF (J.EQ.N) THEN
            NM=I
            GOTO 110
         ENDIF
 100  CONTINUE
 110  CONTINUE
C---GIVE UP IF THERE ARE LESS THAN N INCLUSIVE JETS
      IF (NM.EQ.0) CALL KTWARN('KTIREC',102,*999)
C---RECONSTRUCT THE JETS AT THE APPROPRIATE SCALE
      ECUT=SQRT(KT(NM)/RSQ)
      YMAC=RSQ
      CALL KTRECO(RECO,PP,NN,ECUT,YCUT,YMAC,PSUB,JET,NJET,NSUB,*999)
C---GET RID OF THE ONES THAT DO NOT END UP IN THE JET WE WANT
      NSUB=0
      DO 210 I=1,NJET
         IF (JET(I).EQ.HIST(NM)) THEN
            NSUB=NSUB+1
            DO 200 J=1,4
               PSUB(J,NSUB)=PSUB(J,I)
 200        CONTINUE
         ENDIF
 210  CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTWICH(ECUT,YCUT,JET,NJET,*)
      IMPLICIT NONE
C---GIVE A LIST OF WHICH JET EACH ORIGINAL PARTICLE ENDED UP IN AT SCALE
C   YCUT, TOGETHER WITH THE NUMBER OF JETS AT THAT SCALE.
C
C   ECUT     = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   YCUT     = INPUT : Y VALUE AT WHICH TO DEFINE JETS
C   JET(J)   =OUTPUT : THE JET WHICH CONTAINS THE Jth PARTICLE,
C                        SET TO ZERO IF IT WAS PUT INTO THE BEAM JETS
C   NJET     =OUTPUT : THE NUMBER OF JETS AT SCALE YCUT (SO JET()
C                        ENTRIES WILL BE IN THE RANGE 0 -> NJET)
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER JET(*),NJET,NTEMP
      DOUBLE PRECISION ECUT,YCUT
      CALL KTWCHS(ECUT,YCUT,YCUT,JET,NJET,NTEMP,*999)
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTWCHS(ECUT,YCUT,YMAC,JET,NJET,NSUB,*)
      IMPLICIT NONE
C---GIVE A LIST OF WHICH SUB-JET EACH ORIGINAL PARTICLE ENDED UP IN AT
C   SCALE YCUT, WITH MACRO-JET SCALE YMAC, TOGETHER WITH THE NUMBER OF
C   JETS AT SCALE YCUT AND THE NUMBER OF THEM WHICH ARE SUB-JETS.
C
C   ECUT     = INPUT : DENOMINATOR OF KT MEASURE. IF ZERO, ETOT IS USED
C   YCUT     = INPUT : Y VALUE AT WHICH TO DEFINE JETS
C   YMAC     = INPUT : Y VALUE AT WHICH TO DEFINE MACRO-JETS
C   JET(J)   =OUTPUT : THE JET WHICH CONTAINS THE Jth PARTICLE,
C                        SET TO ZERO IF IT WAS PUT INTO THE BEAM JETS
C   NJET     =OUTPUT : THE NUMBER OF JETS AT SCALE YCUT (SO JET()
C                        ENTRIES WILL BE IN THE RANGE 0 -> NJET)
C   NSUB     =OUTPUT : THE NUMBER OF SUB-JETS AT SCALE YCUT, WITH
C                        MACRO-JETS DEFINED AT SCALE YMAC (SO ONLY NSUB
C                        OF THE JETS 1 -> NJET WILL APPEAR IN JET())
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL FLOATING POINT VARIABLES ARE DECLARED DOUBLE PRECISION
C
      INTEGER NMAX,JET(*),NJET,NSUB,HIST,NUM,I,J,JSUB
      PARAMETER (NMAX=512)
      DOUBLE PRECISION P1(4,NMAX),P2(4,NMAX)
      DOUBLE PRECISION ECUT,YCUT,YMAC,ZERO,ETOT,RSQ,P,KTP,KTS,KT,KTLAST
      COMMON /KTCOMM/ETOT,RSQ,P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),
     &  KT(NMAX),KTLAST(NMAX),HIST(NMAX),NUM
      DIMENSION JSUB(NMAX)
C---THE MOMENTA HAVE TO BEEN GIVEN LEGAL VALUES,
C   EVEN THOUGH THEY WILL NEVER BE USED
      DATA ((P1(J,I),I=1,NMAX),J=1,4),ZERO
     &  /NMAX*1,NMAX*0,NMAX*0,NMAX*1,0/
C---FIRST GET A LIST OF WHICH PARTICLE IS IN WHICH JET AT YCUT
      CALL KTRECO(1,P1,NUM,ECUT,ZERO,YCUT,P2,JET,NJET,NSUB,*999)
C---THEN FIND OUT WHICH JETS ARE SUBJETS
      CALL KTRECO(1,P1,NUM,ECUT,YCUT,YMAC,P2,JSUB,NJET,NSUB,*999)
C---AND MODIFY JET() ACCORDINGLY
      DO 10 I=1,NUM
        IF (JET(I).NE.0) THEN
          IF (JSUB(JET(I)).EQ.0) JET(I)=0
        ENDIF
 10   CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTFRAM(IOPT,CMF,SIGN,Z,XZ,N,P,Q,*)
      IMPLICIT NONE
C---BOOST PARTICLES IN P TO/FROM FRAME GIVEN BY CMF, Z, XZ.
C---IN THIS FRAME CMZ IS STATIONARY,
C                   Z IS ALONG THE (SIGN)Z-AXIS (SIGN=+ OR -)
C                  XZ IS IN THE X-Z PLANE (WITH POSITIVE X COMPONENT)
C---IF Z HAS LENGTH ZERO, OR SIGN=0, NO ROTATION IS PERFORMED
C---IF XZ HAS ZERO COMPONENT PERPENDICULAR TO Z IN THAT FRAME,
C   NO AZIMUTHAL ROTATION IS PERFORMED
C
C   IOPT    = INPUT  : 0=TO FRAME, 1=FROM FRAME
C   CMF(I)  = INPUT  : 4-MOMENTUM WHICH IS STATIONARY IN THE FRAME
C   SIGN    = INPUT  : DIRECTION OF Z IN THE FRAME, NOTE THAT
C                        ONLY ITS SIGN IS USED, NOT ITS MAGNITUDE
C   Z(I)    = INPUT  : 4-MOMENTUM WHICH LIES ON THE (SIGN)Z-AXIS
C   XZ(I)   = INPUT  : 4-MOMENTUM WHICH LIES IN THE X-Z PLANE
C   N       = INPUT  : NUMBER OF PARTICLES IN P
C   P(I,J)  = INPUT  : 4-MOMENTUM OF JTH PARTICLE BEFORE
C   Q(I,J)  = OUTPUT : 4-MOMENTUM OF JTH PARTICLE AFTER
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED
C
C   NOTE THAT ALL MOMENTA ARE DOUBLE PRECISION
C
C   NOTE THAT IT IS SAFE TO CALL WITH P=Q
C   
      INTEGER IOPT,I,N
      DOUBLE PRECISION CMF(4),SIGN,Z(4),XZ(4),P(4,N),Q(4,N),
     &  R(4,4),NEW(4),OLD(4)
      IF (IOPT.LT.0.OR.IOPT.GT.1) CALL KTWARN('KTFRAM',200,*999)
C---FIND BOOST TO GET THERE FROM LAB
      CALL KTUNIT(R)
      CALL KTLBST(0,R,CMF,*999)
C---FIND ROTATION TO PUT BOOSTED Z ON THE (SIGN)Z AXIS
      IF (SIGN.NE.0) THEN
        CALL KTVMUL(R,Z,OLD)
        IF (OLD(1).NE.0.OR.OLD(2).NE.0.OR.OLD(3).NE.0) THEN
          NEW(1)=0
          NEW(2)=0
          NEW(3)=SIGN
          NEW(4)=ABS(SIGN)
          CALL KTRROT(R,OLD,NEW,*999)
C---FIND ROTATION TO PUT BOOSTED AND ROTATED XZ INTO X-Z PLANE
          CALL KTVMUL(R,XZ,OLD)
          IF (OLD(1).NE.0.OR.OLD(2).NE.0) THEN
            NEW(1)=1
            NEW(2)=0
            NEW(3)=0
            NEW(4)=1
            OLD(3)=0
C---NOTE THAT A POTENTIALLY AWKWARD SPECIAL CASE IS AVERTED, BECAUSE IF
C   OLD AND NEW ARE EXACTLY BACK-TO-BACK, THE ROTATION AXIS IS UNDEFINED
C   BUT IN THAT CASE KTRROT WILL USE THE Z AXIS, AS REQUIRED
            CALL KTRROT(R,OLD,NEW,*999)
          ENDIF
        ENDIF
      ENDIF
C---INVERT THE TRANSFORMATION IF NECESSARY
      IF (IOPT.EQ.1) CALL KTINVT(R,R)
C---APPLY THE RESULT TO ALL THE VECTORS
      DO 30 I=1,N
        CALL KTVMUL(R,P(1,I),Q(1,I))
 30   CONTINUE
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTBREI(IOPT,PLEP,PHAD,POUT,N,P,Q,*)
      IMPLICIT NONE
C---BOOST PARTICLES IN P TO/FROM BREIT FRAME
C
C   IOPT    = INPUT  : 0/2=TO BREIT FRAME, 1/3=FROM BREIT FRAME
C                      0/1=NO AZIMUTHAL ROTATION AFTERWARDS
C                      2/3=LEPTON PLANE ROTATED INTO THE X-Z PLANE
C   PLEP    = INPUT  : MOMENTUM OF INCOMING LEPTON IN +Z DIRECTION
C   PHAD    = INPUT  : MOMENTUM OF INCOMING HADRON IN +Z DIRECTION
C   POUT(I) = INPUT  : 4-MOMENTUM OF OUTGOING LEPTON
C   N       = INPUT  : NUMBER OF PARTICLES IN P
C   P(I,J)  = INPUT  : 4-MOMENTUM OF JTH PARTICLE BEFORE
C   Q(I,J)  = OUTPUT : 4-MOMENTUM OF JTH PARTICLE AFTER
C   LAST ARGUMENT IS LABEL TO JUMP TO IF FOR ANY REASON THE EVENT
C   COULD NOT BE PROCESSED (MOST LIKELY DUE TO PARTICLES HAVING SMALLER
C   ENERGY THAN MOMENTUM)
C
C   NOTE THAT ALL MOMENTA ARE DOUBLE PRECISION
C
C   NOTE THAT IT IS SAFE TO CALL WITH P=Q
C   
      INTEGER IOPT,N
      DOUBLE PRECISION PLEP,PHAD,POUT(4),P(4,N),Q(4,N),
     &  CMF(4),Z(4),XZ(4),DOT,QDQ
C---CHECK INPUT
      IF (IOPT.LT.0.OR.IOPT.GT.3) CALL KTWARN('KTBREI',200,*999)
C---FIND 4-MOMENTUM OF BREIT FRAME (TIMES AN ARBITRARY FACTOR)
      DOT=ABS(PHAD)*(ABS(PLEP)-POUT(4))-PHAD*(PLEP-POUT(3))
      QDQ=(ABS(PLEP)-POUT(4))**2-(PLEP-POUT(3))**2-POUT(2)**2-POUT(1)**2
      CMF(1)=DOT*(         -POUT(1))
      CMF(2)=DOT*(         -POUT(2))
      CMF(3)=DOT*(    PLEP -POUT(3))-QDQ*    PHAD
      CMF(4)=DOT*(ABS(PLEP)-POUT(4))-QDQ*ABS(PHAD)
C---FIND ROTATION TO PUT INCOMING HADRON BACK ON Z-AXIS
      Z(1)=0
      Z(2)=0
      Z(3)=PHAD
      Z(4)=ABS(PHAD)
      XZ(1)=0
      XZ(2)=0
      XZ(3)=0
      XZ(4)=0
C---DO THE BOOST
      IF (IOPT.LE.1) THEN
        CALL KTFRAM(IOPT,CMF,PHAD,Z,XZ,N,P,Q,*999)
      ELSE
        CALL KTFRAM(IOPT-2,CMF,PHAD,Z,POUT,N,P,Q,*999)
      ENDIF
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTHADR(IOPT,PLEP,PHAD,POUT,N,P,Q,*)
      IMPLICIT NONE
C---BOOST PARTICLES IN P TO/FROM HADRONIC CMF
C
C   ARGUMENTS ARE EXACTLY AS FOR KTBREI
C
C   NOTE THAT ALL MOMENTA ARE DOUBLE PRECISION
C
C   NOTE THAT IT IS SAFE TO CALL WITH P=Q
C   
      INTEGER IOPT,N
      DOUBLE PRECISION PLEP,PHAD,POUT(4),P(4,N),Q(4,N),
     &  CMF(4),Z(4),XZ(4)
C---CHECK INPUT
      IF (IOPT.LT.0.OR.IOPT.GT.3) CALL KTWARN('KTHADR',200,*999)
C---FIND 4-MOMENTUM OF HADRONIC CMF
      CMF(1)=         -POUT(1)
      CMF(2)=         -POUT(2)
      CMF(3)=    PLEP -POUT(3)+    PHAD
      CMF(4)=ABS(PLEP)-POUT(4)+ABS(PHAD)
C---FIND ROTATION TO PUT INCOMING HADRON BACK ON Z-AXIS
      Z(1)=0
      Z(2)=0
      Z(3)=PHAD
      Z(4)=ABS(PHAD)
      XZ(1)=0
      XZ(2)=0
      XZ(3)=0
      XZ(4)=0
C---DO THE BOOST
      IF (IOPT.LE.1) THEN
        CALL KTFRAM(IOPT,CMF,PHAD,Z,XZ,N,P,Q,*999)
      ELSE
        CALL KTFRAM(IOPT-2,CMF,PHAD,Z,POUT,N,P,Q,*999)
      ENDIF
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      FUNCTION KTPAIR(ANGL,P,Q,ANGLE)
      IMPLICIT NONE
C---CALCULATE LOCAL KT OF PAIR, USING ANGULAR SCHEME:
C   1=>ANGULAR, 2=>DeltaR, 3=>f(DeltaEta,DeltaPhi)
C   WHERE f(eta,phi)=2(COSH(eta)-COS(phi)) IS THE QCD EMISSION METRIC
C---IF ANGLE<0, IT IS SET TO THE ANGULAR PART OF THE LOCAL KT ON RETURN
C   IF ANGLE>0, IT IS USED INSTEAD OF THE ANGULAR PART OF THE LOCAL KT
      INTEGER ANGL
      DOUBLE PRECISION P(9),Q(9),KTPAIR,R,KTMDPI,ANGLE,ETA,PHI,ESQ
C---COMPONENTS OF MOMENTA ARE PX,PY,PZ,E,1/P,PT,ETA,PHI,PT**2
      R=ANGLE
      IF (ANGL.EQ.1) THEN
         IF (R.LE.0) R=2*(1-(P(1)*Q(1)+P(2)*Q(2)+P(3)*Q(3))*(P(5)*Q(5)))
         ESQ=MIN(P(4),Q(4))**2
      ELSEIF (ANGL.EQ.2.OR.ANGL.EQ.3) THEN
         IF (R.LE.0) THEN
            ETA=P(7)-Q(7)
            PHI=KTMDPI(P(8)-Q(8))
            IF (ANGL.EQ.2) THEN
               R=ETA**2+PHI**2
            ELSE
               R=2*(COSH(ETA)-COS(PHI))
            ENDIF
         ENDIF
         ESQ=MIN(P(9),Q(9))
      ELSEIF (ANGL.EQ.4) THEN
        ESQ=(1d0/(P(5)*Q(5))-P(1)*Q(1)-P(2)*Q(2)-
     &P(3)*Q(3))*2D0/(P(5)*Q(5))/(0.0001D0+1d0/P(5)+1d0/Q(5))**2        
        R=1d0
      ELSE
         ktpair = 0D0
	 CALL KTWARN('KTPAIR',200,*999)
         STOP
      ENDIF
      KTPAIR=ESQ*R
      IF (ANGLE.LT.0) ANGLE=R
 999  END
C-----------------------------------------------------------------------
      FUNCTION KTSING(ANGL,TYPE,P)
      IMPLICIT NONE
C---CALCULATE KT OF PARTICLE, USING ANGULAR SCHEME:
C   1=>ANGULAR, 2=>DeltaR, 3=>f(DeltaEta,DeltaPhi)
C---TYPE=1 FOR E+E-, 2 FOR EP, 3 FOR PE, 4 FOR PP
C   FOR EP, PROTON DIRECTION IS DEFINED AS -Z
C   FOR PE, PROTON DIRECTION IS DEFINED AS +Z
      DOUBLE PRECISION KTSING,P(9)
      DOUBLE PRECISION COSTH,R,SMALL
      INTEGER ANGL,TYPE
      DATA SMALL/1D-4/
      IF (ANGL.EQ.1.OR.ANGL.EQ.4) THEN
         COSTH=P(3)*P(5)
         IF (TYPE.EQ.2) THEN
            COSTH=-COSTH
         ELSEIF (TYPE.EQ.4) THEN
            COSTH=ABS(COSTH)
         ELSEIF (TYPE.NE.1.AND.TYPE.NE.3) THEN
	    ktsing = 0D0
            CALL KTWARN('KTSING',200,*999)
            STOP
         ENDIF
         R=2*(1-COSTH)
C---IF CLOSE TO BEAM, USE APPROX 2*(1-COS(THETA))=SIN**2(THETA)
         IF (R.LT.SMALL) R=(P(1)**2+P(2)**2)*P(5)**2
         KTSING=P(4)**2*R
      ELSEIF (ANGL.EQ.2.OR.ANGL.EQ.3) THEN
         KTSING=P(9)
      ELSE
	 ktsing = 0D0
         CALL KTWARN('KTSING',201,*999)
         STOP
      ENDIF
 999  END
C-----------------------------------------------------------------------
      SUBROUTINE KTPMIN(A,NMAX,N,IMIN,JMIN)
      IMPLICIT NONE
C---FIND THE MINIMUM MEMBER OF A(NMAX,NMAX) WITH IMIN < JMIN <= N
      INTEGER NMAX,N,IMIN,JMIN,KMIN,I,J,K
C---REMEMBER THAT A(X+(Y-1)*NMAX)=A(X,Y)
C   THESE LOOPING VARIABLES ARE J=Y-2, I=X+(Y-1)*NMAX
      DOUBLE PRECISION A(*),AMIN
      K=1+NMAX
      KMIN=K
      AMIN=A(KMIN)
      DO 110 J=0,N-2
         DO 100 I=K,K+J
            IF (A(I).LT.AMIN) THEN
               KMIN=I
               AMIN=A(KMIN)
            ENDIF
 100     CONTINUE
         K=K+NMAX
 110  CONTINUE
      JMIN=KMIN/NMAX+1
      IMIN=KMIN-(JMIN-1)*NMAX
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTSMIN(A,NMAX,N,IMIN)
      IMPLICIT NONE
C---FIND THE MINIMUM MEMBER OF A
      INTEGER N,NMAX,IMIN,I
      DOUBLE PRECISION A(NMAX)
      IMIN=1
      DO 100 I=1,N
         IF (A(I).LT.A(IMIN)) IMIN=I
 100  CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTCOPY(A,N,B,ONSHLL)
      IMPLICIT NONE
C---COPY FROM A TO B. 5TH=1/(3-MTM), 6TH=PT, 7TH=ETA, 8TH=PHI, 9TH=PT**2
C   IF ONSHLL IS .TRUE. PARTICLE ENTRIES ARE PUT ON-SHELL BY SETTING E=P
      INTEGER I,N
      DOUBLE PRECISION A(4,N)
      LOGICAL ONSHLL
      DOUBLE PRECISION B(9,N),ETAMAX,SINMIN,EPS
      DATA ETAMAX,SINMIN,EPS/10,0,1D-6/
C---SINMIN GETS CALCULATED ON FIRST CALL
      IF (SINMIN.EQ.0) SINMIN=1/COSH(ETAMAX)
      DO 100 I=1,N
         B(1,I)=A(1,I)
         B(2,I)=A(2,I)
         B(3,I)=A(3,I)
         B(4,I)=A(4,I)
         B(5,I)=SQRT(A(1,I)**2+A(2,I)**2+A(3,I)**2)
         IF (ONSHLL) B(4,I)=B(5,I)
         IF (B(5,I).EQ.0) B(5,I)=1D-10
         B(5,I)=1/B(5,I)
         B(9,I)=A(1,I)**2+A(2,I)**2
         B(6,I)=SQRT(B(9,I))
         B(7,I)=B(6,I)*B(5,I)
         IF (B(7,I).GT.SINMIN) THEN
            B(7,I)=A(4,I)**2-A(3,I)**2
            IF (B(7,I).LE.EPS*B(4,I)**2.OR.ONSHLL) B(7,I)=B(9,I)
            B(7,I)=LOG((B(4,I)+ABS(B(3,I)))**2/B(7,I))/2
         ELSE
            B(7,I)=ETAMAX+2
         ENDIF
         B(7,I)=SIGN(B(7,I),B(3,I))
         IF (A(1,I).EQ.0 .AND. A(2,I).EQ.0) THEN
            B(8,I)=0
         ELSE
            B(8,I)=ATAN2(A(2,I),A(1,I))
         ENDIF
 100  CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTMERG(P,KTP,KTS,NMAX,I,J,N,TYPE,ANGL,MONO,RECO)
      IMPLICIT NONE
C---MERGE THE Jth PARTICLE IN P INTO THE Ith PARTICLE
C   J IS ASSUMED GREATER THAN I. P CONTAINS N PARTICLES BEFORE MERGING.
C---ALSO RECALCULATING THE CORRESPONDING KTP AND KTS VALUES IF MONO.GT.0
C   FROM THE RECOMBINED ANGULAR MEASURES IF MONO.GT.1
C---NOTE THAT IF MONO.LE.0, TYPE AND ANGL ARE NOT USED
      INTEGER ANGL,RECO,TYPE,I,J,K,N,NMAX,MONO
      DOUBLE PRECISION P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX),PT,PTT,
     &     KTMDPI,KTUP,PI,PJ,ANG,KTPAIR,KTSING,ETAMAX,EPS
      KTUP(I,J)=KTP(MAX(I,J),MIN(I,J))
      DATA ETAMAX,EPS/10,1D-6/
      IF (J.LE.I) CALL KTWARN('KTMERG',200,*999)
C---COMBINE ANGULAR MEASURES IF NECESSARY
      IF (MONO.GT.1) THEN
         DO 100 K=1,N
            IF (K.NE.I.AND.K.NE.J) THEN
               IF (RECO.EQ.1) THEN
                  PI=P(4,I)
                  PJ=P(4,J)
               ELSEIF (RECO.EQ.2) THEN
                  PI=P(6,I)
                  PJ=P(6,J)
               ELSEIF (RECO.EQ.3) THEN
                  PI=P(9,I)
                  PJ=P(9,J)
               ELSE
                  CALL KTWARN('KTMERG',201,*999)
                  STOP
               ENDIF
               IF (PI.EQ.0.AND.PJ.EQ.0) THEN
                  PI=1
                  PJ=1
               ENDIF
               KTP(MAX(I,K),MIN(I,K))=
     &              (PI*KTUP(I,K)+PJ*KTUP(J,K))/(PI+PJ)
            ENDIF
 100     CONTINUE
      ENDIF
      IF (RECO.EQ.1) THEN
C---VECTOR ADDITION
         P(1,I)=P(1,I)+P(1,J)
         P(2,I)=P(2,I)+P(2,J)
         P(3,I)=P(3,I)+P(3,J)
c         P(4,I)=P(4,I)+P(4,J) ! JA
         P(5,I)=SQRT(P(1,I)**2+P(2,I)**2+P(3,I)**2)
         P(4,I)=P(5,I) ! JA (Massless scheme)
         IF (P(5,I).EQ.0) THEN
            P(5,I)=1
         ELSE
            P(5,I)=1/P(5,I)
         ENDIF
      ELSEIF (RECO.EQ.2) THEN
C---PT WEIGHTED ETA-PHI ADDITION
         PT=P(6,I)+P(6,J)
         IF (PT.EQ.0) THEN
            PTT=1
         ELSE
            PTT=1/PT
         ENDIF
         P(7,I)=(P(6,I)*P(7,I)+P(6,J)*P(7,J))*PTT
         P(8,I)=KTMDPI(P(8,I)+P(6,J)*PTT*KTMDPI(P(8,J)-P(8,I)))
         P(6,I)=PT
         P(9,I)=PT**2
      ELSEIF (RECO.EQ.3) THEN
C---PT**2 WEIGHTED ETA-PHI ADDITION
         PT=P(9,I)+P(9,J)
         IF (PT.EQ.0) THEN
            PTT=1
         ELSE
            PTT=1/PT
         ENDIF
         P(7,I)=(P(9,I)*P(7,I)+P(9,J)*P(7,J))*PTT
         P(8,I)=KTMDPI(P(8,I)+P(9,J)*PTT*KTMDPI(P(8,J)-P(8,I)))
         P(6,I)=P(6,I)+P(6,J)
         P(9,I)=P(6,I)**2
      ELSE
         CALL KTWARN('KTMERG',202,*999)
         STOP
      ENDIF
C---IF MONO.GT.0 CALCULATE NEW KT MEASURES. IF MONO.GT.1 USE ANGULAR ONES.
      IF (MONO.LE.0) RETURN
C---CONVERTING BETWEEN 4-MTM AND PT,ETA,PHI IF NECESSARY
      IF (ANGL.NE.1.AND.RECO.EQ.1) THEN
         P(9,I)=P(1,I)**2+P(2,I)**2
         P(7,I)=P(4,I)**2-P(3,I)**2
         IF (P(7,I).LE.EPS*P(4,I)**2) P(7,I)=P(9,I)
         IF (P(7,I).GT.0) THEN
            P(7,I)=LOG((P(4,I)+ABS(P(3,I)))**2/P(7,I))/2
            IF (P(7,I).GT.ETAMAX) P(7,I)=ETAMAX+2
         ELSE
            P(7,I)=ETAMAX+2
         ENDIF
         P(7,I)=SIGN(P(7,I),P(3,I))
         IF (P(1,I).NE.0.AND.P(2,I).NE.0) THEN
            P(8,I)=ATAN2(P(2,I),P(1,I))
         ELSE
            P(8,I)=0
         ENDIF
      ELSEIF (ANGL.EQ.1.AND.RECO.NE.1) THEN
         P(1,I)=P(6,I)*COS(P(8,I))
         P(2,I)=P(6,I)*SIN(P(8,I))
         P(3,I)=P(6,I)*SINH(P(7,I))
         P(4,I)=P(6,I)*COSH(P(7,I))
         IF (P(4,I).NE.0) THEN
            P(5,I)=1/P(4,I)
         ELSE
            P(5,I)=1
         ENDIF
      ENDIF
      ANG=0
      DO 200 K=1,N
         IF (K.NE.I.AND.K.NE.J) THEN
            IF (MONO.GT.1) ANG=KTUP(I,K)
            KTP(MIN(I,K),MAX(I,K))=
     &           KTPAIR(ANGL,P(1,I),P(1,K),ANG)
         ENDIF
 200  CONTINUE
      KTS(I)=KTSING(ANGL,TYPE,P(1,I))
 999  END
C-----------------------------------------------------------------------
      SUBROUTINE KTMOVE(P,KTP,KTS,NMAX,N,J,IOPT)
      IMPLICIT NONE
C---MOVE THE Nth PARTICLE IN P TO THE Jth POSITION
C---ALSO MOVING KTP AND KTS IF IOPT.GT.0
      INTEGER I,J,N,NMAX,IOPT
      DOUBLE PRECISION P(9,NMAX),KTP(NMAX,NMAX),KTS(NMAX)
      DO 100 I=1,9
         P(I,J)=P(I,N)
 100  CONTINUE
      IF (IOPT.LE.0) RETURN
      DO 110 I=1,J-1
         KTP(I,J)=KTP(I,N)
         KTP(J,I)=KTP(N,I)
 110  CONTINUE
      DO 120 I=J+1,N-1
         KTP(J,I)=KTP(I,N)
         KTP(I,J)=KTP(N,I)
 120  CONTINUE
      KTS(J)=KTS(N)
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTUNIT(R)
      IMPLICIT NONE
C   SET R EQUAL TO THE 4 BY 4 IDENTITY MATRIX
      DOUBLE PRECISION R(4,4)
      INTEGER I,J
      DO 20 I=1,4
        DO 10 J=1,4
          R(I,J)=0
          IF (I.EQ.J) R(I,J)=1
 10     CONTINUE
 20   CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTLBST(IOPT,R,A,*)
      IMPLICIT NONE
C   PREMULTIPLY R BY THE 4 BY 4 MATRIX TO
C   LORENTZ BOOST TO/FROM THE CM FRAME OF A
C   IOPT=0 => TO
C   IOPT=1 => FROM
C
C   LAST ARGUMENT IS LABEL TO JUMP TO IF A IS NOT TIME-LIKE
C
      INTEGER IOPT,I,J
      DOUBLE PRECISION R(4,4),A(4),B(4),C(4,4),M
      DO 10 I=1,4
        B(I)=A(I)
 10   CONTINUE
      M=B(4)**2-B(1)**2-B(2)**2-B(3)**2
      IF (M.LE.0) CALL KTWARN('KTLBST',100,*999)
      M=SQRT(M)
      B(4)=B(4)+M
      M=1/(M*B(4))
      IF (IOPT.EQ.0) THEN
        B(4)=-B(4)
      ELSEIF (IOPT.NE.1) THEN
        CALL KTWARN('KTLBST',200,*999)
        STOP
      ENDIF
      DO 30 I=1,4
        DO 20 J=1,4
          C(I,J)=B(I)*B(J)*M
          IF (I.EQ.J) C(I,J)=C(I,J)+1
 20     CONTINUE
 30   CONTINUE
      C(4,4)=C(4,4)-2
      CALL KTMMUL(C,R,R)
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTRROT(R,A,B,*)
      IMPLICIT NONE
C   PREMULTIPLY R BY THE 4 BY 4 MATRIX TO
C   ROTATE FROM VECTOR A TO VECTOR B BY THE SHORTEST ROUTE
C   IF THEY ARE EXACTLY BACK-TO-BACK, THE ROTATION AXIS IS THE VECTOR
C   WHICH IS PERPENDICULAR TO THEM AND THE X AXIS, UNLESS THEY ARE
C   PERPENDICULAR TO THE Y AXIS, WHEN IT IS THE VECTOR WHICH IS
C   PERPENDICULAR TO THEM AND THE Y AXIS.
C   NOTE THAT THESE CONDITIONS GUARANTEE THAT IF BOTH ARE PERPENDICULAR
C   TO THE Z AXIS, IT WILL BE USED AS THE ROTATION AXIS.
C
C   LAST ARGUMENT IS LABEL TO JUMP TO IF EITHER HAS LENGTH ZERO
C
      DOUBLE PRECISION R(4,4),M(4,4),A(4),B(4),C(4),D(4),AL,BL,CL,DL,EPS
C---SQRT(2*EPS) IS THE ANGLE IN RADIANS OF THE SMALLEST ALLOWED ROTATION
C   NOTE THAT IF YOU CONVERT THIS PROGRAM TO SINGLE PRECISION, YOU WILL
C   NEED TO INCREASE EPS TO AROUND 0.5E-4
      PARAMETER (EPS=0.5D-6)
      AL=A(1)**2+A(2)**2+A(3)**2
      BL=B(1)**2+B(2)**2+B(3)**2
      IF (AL.LE.0.OR.BL.LE.0) CALL KTWARN('KTRROT',100,*999)
      AL=1/SQRT(AL)
      BL=1/SQRT(BL)
      CL=(A(1)*B(1)+A(2)*B(2)+A(3)*B(3))*AL*BL
C---IF THEY ARE COLLINEAR, DON'T NEED TO DO ANYTHING
      IF (CL.GE.1-EPS) THEN
        RETURN
C---IF THEY ARE BACK-TO-BACK, USE THE AXIS PERP TO THEM AND X AXIS
      ELSEIF (CL.LE.-1+EPS) THEN
        IF (ABS(B(2)).GT.EPS) THEN
          C(1)= 0
          C(2)=-B(3)
          C(3)= B(2)
C---UNLESS THEY ARE PERPENDICULAR TO THE Y AXIS,
        ELSE
          C(1)= B(3)
          C(2)= 0
          C(3)=-B(1)
        ENDIF
C---OTHERWISE FIND ROTATION AXIS
      ELSE
        C(1)=A(2)*B(3)-A(3)*B(2)
        C(2)=A(3)*B(1)-A(1)*B(3)
        C(3)=A(1)*B(2)-A(2)*B(1)
      ENDIF
      CL=C(1)**2+C(2)**2+C(3)**2
      IF (CL.LE.0) CALL KTWARN('KTRROT',101,*999)
      CL=1/SQRT(CL)
C---FIND ROTATION TO INTERMEDIATE AXES FROM A
      D(1)=A(2)*C(3)-A(3)*C(2)
      D(2)=A(3)*C(1)-A(1)*C(3)
      D(3)=A(1)*C(2)-A(2)*C(1)
      DL=AL*CL
      M(1,1)=A(1)*AL
      M(1,2)=A(2)*AL
      M(1,3)=A(3)*AL
      M(1,4)=0
      M(2,1)=C(1)*CL
      M(2,2)=C(2)*CL
      M(2,3)=C(3)*CL
      M(2,4)=0
      M(3,1)=D(1)*DL
      M(3,2)=D(2)*DL
      M(3,3)=D(3)*DL
      M(3,4)=0
      M(4,1)=0
      M(4,2)=0
      M(4,3)=0
      M(4,4)=1
      CALL KTMMUL(M,R,R)
C---AND ROTATION FROM INTERMEDIATE AXES TO B
      D(1)=B(2)*C(3)-B(3)*C(2)
      D(2)=B(3)*C(1)-B(1)*C(3)
      D(3)=B(1)*C(2)-B(2)*C(1)
      DL=BL*CL
      M(1,1)=B(1)*BL
      M(2,1)=B(2)*BL
      M(3,1)=B(3)*BL
      M(1,2)=C(1)*CL
      M(2,2)=C(2)*CL
      M(3,2)=C(3)*CL
      M(1,3)=D(1)*DL
      M(2,3)=D(2)*DL
      M(3,3)=D(3)*DL
      CALL KTMMUL(M,R,R)
      RETURN
 999  RETURN 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTVMUL(M,A,B)
      IMPLICIT NONE
C   4 BY 4 MATRIX TIMES 4 VECTOR: B=M*A.
C   ALL ARE DOUBLE PRECISION
C   IT IS SAFE TO CALL WITH B=A
C   FIRST SUBSCRIPT=ROWS, SECOND=COLUMNS
      DOUBLE PRECISION M(4,4),A(4),B(4),C(4)
      INTEGER I,J
      DO 20 I=1,4
        C(I)=0
        DO 10 J=1,4
          C(I)=C(I)+M(I,J)*A(J)
 10     CONTINUE
 20   CONTINUE
      DO 30 I=1,4
        B(I)=C(I)
 30   CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTMMUL(A,B,C)
      IMPLICIT NONE
C   4 BY 4 MATRIX MULTIPLICATION: C=A*B.
C   ALL ARE DOUBLE PRECISION
C   IT IS SAFE TO CALL WITH C=A OR B.
C   FIRST SUBSCRIPT=ROWS, SECOND=COLUMNS
      DOUBLE PRECISION A(4,4),B(4,4),C(4,4),D(4,4)
      INTEGER I,J,K
      DO 30 I=1,4
        DO 20 J=1,4
          D(I,J)=0
          DO 10 K=1,4
            D(I,J)=D(I,J)+A(I,K)*B(K,J)
 10       CONTINUE
 20     CONTINUE
 30   CONTINUE
      DO 50 I=1,4
        DO 40 J=1,4
          C(I,J)=D(I,J)
 40     CONTINUE
 50   CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTINVT(A,B)
      IMPLICIT NONE
C---INVERT TRANSFORMATION MATRIX A
C
C   A = INPUT  : 4 BY 4 TRANSFORMATION MATRIX
C   B = OUTPUT : INVERTED TRANSFORMATION MATRIX
C
C   IF A IS NOT A TRANSFORMATION MATRIX YOU WILL GET STRANGE RESULTS
C
C   NOTE THAT IT IS SAFE TO CALL WITH A=B
C
      DOUBLE PRECISION A(4,4),B(4,4),C(4,4)
      INTEGER I,J
C---TRANSPOSE
      DO 20 I=1,4
        DO 10 J=1,4
          C(I,J)=A(J,I)
 10     CONTINUE
 20   CONTINUE
C---NEGATE ENERGY-MOMENTUM MIXING TERMS
      DO 30 I=1,3
        C(4,I)=-C(4,I)
        C(I,4)=-C(I,4)
 30   CONTINUE
C---OUTPUT
      DO 50 I=1,4
        DO 40 J=1,4
          B(I,J)=C(I,J)
 40     CONTINUE
 50   CONTINUE
      END
C-----------------------------------------------------------------------
      FUNCTION KTMDPI(PHI)
      IMPLICIT NONE
C---RETURNS PHI, MOVED ONTO THE RANGE [-PI,PI)
      DOUBLE PRECISION KTMDPI,PHI,PI,TWOPI,THRPI,EPS
      PARAMETER (PI=3.14159265358979324D0,TWOPI=6.28318530717958648D0,
     &     THRPI=9.42477796076937972D0)
      PARAMETER (EPS=1D-15)
      KTMDPI=PHI
      IF (KTMDPI.LE.PI) THEN
        IF (KTMDPI.GT.-PI) THEN
          GOTO 100
        ELSEIF (KTMDPI.GT.-THRPI) THEN
          KTMDPI=KTMDPI+TWOPI
        ELSE
          KTMDPI=-MOD(PI-KTMDPI,TWOPI)+PI
        ENDIF
      ELSEIF (KTMDPI.LE.THRPI) THEN
        KTMDPI=KTMDPI-TWOPI
      ELSE
        KTMDPI=MOD(PI+KTMDPI,TWOPI)-PI
      ENDIF
 100  IF (ABS(KTMDPI).LT.EPS) KTMDPI=0
      END
C-----------------------------------------------------------------------
      SUBROUTINE KTWARN(SUBRTN,ICODE,*)
C     DEALS WITH ERRORS DURING EXECUTION
C     SUBRTN = NAME OF CALLING SUBROUTINE
C     ICODE  = ERROR CODE:    - 99 PRINT WARNING & CONTINUE
C                          100-199 PRINT WARNING & JUMP
C                          200-    PRINT WARNING & STOP DEAD
C-----------------------------------------------------------------------
      INTEGER ICODE
      CHARACTER*6 SUBRTN
      WRITE (6,10) SUBRTN,ICODE
   10 FORMAT(/' KTWARN CALLED FROM SUBPROGRAM ',A6,': CODE =',I4/)
      IF (ICODE.LT.100) RETURN
      IF (ICODE.LT.200) RETURN 1
      STOP
      END
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
