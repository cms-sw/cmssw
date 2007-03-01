c ************************************************************************* 
c
c This file contains the routines used by the MC@NLO to integrate the
c NLO cross sections, and to generate weighted/unweighted events.
c Most of these routines are identical to those of the original
c package BASES/SPRING. They have been modified in order to treat
c the case of functions with arbitrary sign. The user should be
c only interested in the two wrappers
c     RUN_BASES
c     RUN_SPRING
c which perform the integration and generate unweighted events respectively.
c See the comments at the beginning of the bodies of these routines.
c
c ------------------------------ SIDE EFFECTS -----------------------------
c
c When the user is interested in generating weighted events, he can access 
c the Vegas weight by inserting in the integrand function the following
c common block:
c      COMMON /CBSWGT/BSEWGT
c This common block is filled by BASES on event-by-event basis. BSEWGT
c is actually the Vegas weight, divided by a normalization factor equal
c to the number of Vegas calls per iteration, times the number of iterations
c in the integration step; BSEWGT can thus be used to fill histograms,
c with no further normalization.
c
c It is possible to generate unweighted events with an integrand function
c of arbitrary sign. Suppose the integrand function is FF0; then one has
c to define FF=ABS(FF0), and give FF as input to BASES/SPRING. The
c information on the sign of FF0 must be passed by the user to BASES/SPRING
c by inserting the common block
c      COMMON /CBSSGN/BSFSGN
c in the body of FF, where BSFSGN is such that FF0=BSFSGN*FF (thus,
c BSFSGN is either +1 or -1). Tipically, the definition of BSFSGN is 
c the last line of code in the body of the function. 
c
c ------------------------------ EVENT RECORD -----------------------------
c
c After SPRING successfully generates an event, it calls (from RUN_OF_SPRING) 
c the routine 
c      SPRFIN
c The user must use this routine to finalize his computation: selecting 
c (on statistical basis) a given flavour/color/kinematics configuration,
c if the event corresponds to several configurations, and writing the
c event on file. It must be stressed that SPRFIN is called
c immediately after SPRING has gone through the input function FF in
c the generation of an event: thus, FF must pass all the relevant
c quantities to SPRFIN through common blocks.
c 
c ************************************************************************* 
c 
c 
c 
c
      subroutine run_bases(ff,prefix,ndim,nwild,ncall,it1,it2,ac1,ac2,
     #                    res,sig,resneg,signeg,ctime,itd1,itd2,iseed,
     #                    iwrite,ircall)
c Master routine to integrate a real*8 function using bases. The entries are:
c
c  ff:     the integrand
c  prefix: bases will write the relevant information on prefix_bs.data
c  ndim:   the dimension of the space that contains the support of ff
c  nwild:  the number of wild variables
c  ncall:  the number of sampling points per iteration (user's input)
c  it1:    the maximum number of iterations in the grid optimization step
c  it2:    the maximum number of iterations in the integration step
c  ac1:    the relative accuracy required in the grid optimization step
c  ac2:    the relative accuracy required in the integration step
c  res:    the result of the integration of ff
c  sig:    its standard deviation
c  resneg: the result of the integration of ff0, where ff=abs(ff0)
c  signeg: its standard deviation
c  ctime:  the computing time used in the integration step
c  itd1:   iterations performed in the grid optimization step
c  itd2:   iterations performed in the integration step
c  iseed:  the initial seed for random number generation
c  iwrite: if equal to zero, bases does NOT write prefix_bs.data
c  ircall: the actual number of sampling points per iteration (bases output)
c
c ******************************** WARNING ******************************** 
c Resneg and signeg return sensible results only if the common block
c      COMMON /CBSSGN/BSFSGN
c is inserted in the body of the integrand function ff
c ******************************** WARNING ******************************** 
c Bases has been modified in order to output weighted events, thus
c avoiding the use of spring. For this option to work, the common block
c      COMMON /CBSWGT/BSEWGT
c must be intserted in the body of the integrand function ff
c
c In the case of the wild variables, bases saves the probability
c information for each hypercube associated to those variables;
c this is crucial when spring is attached. If a variable is not
c wild, the information on the shape of the integrand is lost 
c during the phase of event generation.
c Bases also leaves to the user the choice of whether a variable
c is optimized (i.e., the grid is iteratively adjusted), or not.
c This option is controlled by the value of IG(i) (1=optimization,
c 0=no optimization). In the present interface, we always set IG(i)=1
      implicit none
      real*8 ff,ac1,ac2,res,sig,resneg,signeg,ctime
      integer ndim,nwild,ncall,it1,it2,itd1,itd2,iseed,iwrite,ircall
      integer ing,inpg,insp
      common/brcall/ing,inpg,insp
      character * 60 prefix,fname
      integer isunit
      parameter (isunit=23)
      logical verbose
      data verbose/.false./
      external ff
c
      call init_of_bases(ndim,nwild,ncall,it1,it2,iseed,ac1,ac2)
      call bases(ff,res,sig,resneg,signeg,ctime,itd1,itd2)
      ircall=insp*inpg
      if(iwrite.ne.0)then
        call fk88strcat(prefix,'_bs.data',fname)
        open(unit=isunit,file=fname,
     #       form='unformatted',status='unknown')
        call bswrit(isunit)
        close(isunit)
      endif
      if(verbose)then
        write(6,*)'integral: ',res,' +- ',sig
        write(6,*)'elapsed time: ',ctime
        write(6,*)'itns: ',itd1,itd2
      endif
      return
      end


      subroutine run_spring(ff,prefix,mxevts,mxtrls,nevts,ntrls,
     #                      ndim,nwild,iseed)
c Master routine to generate events using spring. The entries are:
c
c  ff:     the function previously integrated by bases
c  prefix: spring will read the relevant information from prefix_bs.data
c  mxevts: the required number of generated events
c  mxtrls: the maximum number of trials to generate mxevts
c  nevts:  the number of generated events
c  ntrls:  the number of trials
c  ndim:   the dimension of the space that contains the support of ff
c  nwild:  the number of wild variables
c  iseed:  the initial seed for random number generation
c
c ******************************** WARNING ******************************** 
c For the event to be finalised (for example, written on file), 
c the routine 
c      SPRFIN
c must be written by the user; the relevant information is passed
c from FF to SPRFIN by means of common blocks
      implicit none
      real*8 ff,dum
      integer mxevts,mxtrls,nevts,ntrls,ndim,nwild,iseed,idum
      character * 60 prefix
      parameter (dum=-1.d0)
      parameter (idum=-1)
      external ff
c
      if(mxevts.eq.0)return
      call init_of_spring(prefix)
      call init_of_bases(ndim,nwild,idum,idum,idum,iseed,dum,dum)
      if(mxtrls.le.0)then
        write(6,*)'no events will be generated'
        return
      endif
      if(mxtrls.lt.mxevts)then
        write(6,*)
     #   'SPRING warning: the required number of events cannot be'
        write(6,*)
     #   'generated: mxevts, mxtrls=',mxevts,mxtrls
      endif
      call run_of_spring(ff,mxevts,mxtrls,nevts,ntrls)
      if(nevts.ne.mxevts.and.ntrls.ne.mxtrls)then
        write(6,*)'SPRING: fatal error'
        write(6,*)mxevts,nevts
        write(6,*)mxtrls,ntrls
        stop
      endif
      return
      end

c
c
c Begin of BASES routines
c
c
      subroutine init_of_bases(mdim,mwild,mcall,it1,it2,iseed,ac1,ac2)
c This subroutine is called in order to initialize bases parameters;
c it has been obtained by getting together bsdims.f, bsgrid.f, and
c bsparm.f of the original package (BASES50/SPRING50). When the last
c four entries are set to negative values, default values (defined
c in bsinit) are used
      implicit none
      integer mdim,mwild,mcall,it1,it2,iseed
      real * 8 ac1,ac2
      integer mxdim,mxwl
      parameter (mxdim = 50)
      parameter (mxwl = 15)
* NDIM   : The number of dimension of integral                     *
* NWILD  : The number of wild variables                            *
* NCALL  : The number of sample points per iteration.              *
*          This actual number is to be determined by taking the    *
*          number of dimensions into account.                      *
* XL(i)  : The lower value of the i-th integral variable           *
* XU(i)  : The upper value of the i-th integral variable           *
* IG(i)  : The flag switches whether the grid of i-th variable     *
*          is to be optimized ( 1 ) or kept uniform ( 0 ).         *
      real * 8 xl(mxdim),xu(mxdim)
      integer ndim,nwild,ncall,ig(mxdim)
      common/bparm1/xl,xu,ndim,nwild,ig,ncall
* ACC1   : The required accuracy at the grid optimization step     *
* ACC2   : The required accuracy at the integration step.          *
* ITMX1  : The max. number of iteration at the grid opt. step.     *
* ITMX2  : The max. number of iteration at the integration step.   *
      real*8 acc1,acc2
      integer itmx1,itmx2
      common/bparm2/acc1,acc2,itmx1,itmx2
      integer jflag,ibases
      common/base0/jflag,ibases
      integer i
      logical lhisto
      common/bfxlhisto/lhisto
      data lhisto/.false./
c
      if(mdim.gt.mxdim.or.mwild.gt.mxwl)then
        write(6,*)'just kidding....'
        stop
      endif
      ibases=0
      call bsinit
      call drnset(iseed)
      ndim=mdim
      nwild=mwild
      ncall=mcall
      do i=1,ndim
        xl(i)=0.d0
        xu(i)=1.d0
        ig(i)=1
      enddo
      if(it1.ge.0)itmx1=it1
      if(it2.ge.0)itmx2=it2
      if(ac1.ge.0.d0)acc1=ac1
      if(ac2.ge.0.d0)acc2=ac2
      return
      end


************************************************************************
*    ====================================================              *
      SUBROUTINE BASES(FXN,S,SIGMA,SNEG,SIGMANEG,CTIME,IT1,IT2 )
*    ====================================================              *
*      Subroutine BASES for the Numerical integration.                 *
*      In terms of this program Integration can be done, furthermore   *
*      a probability distribution can be made for the event generation.*
*      The event with weight one is generated by program SPRING.       *
* ((Input))                                                            *
*    from the arguement                                                *
*      FXN    : Name of function program                               *
*    from the labeled common /BASE1/                                   *
*      XL(50) : Lower limits of the integration variabels              *
*      XU(50) : upper limits of the integration variabels              *
*      NDIM   : Dimension of the integration                           *
*      NCALL  : Number of sampling points per iteration                *
*    from the lebeled common /BASE2/                                   *
*      ITMX*  : Number of iteration                                    *
*      ACC*   : Required accuracies                                    *
* ((Output))                                                           *
*      S      : Estimate of the integral                               *
*      SIGMA  : Standard deviation of the estimate                     *
*      SNEG   : Estimate of the integral with sign (see run_bases)     *
*      SIGMANEG: Standard deviation of the estimate for sneg           *
*      CTIME  : Computing time required for integration                *
*      IT1    : Number of iterations for the grid defining step        *
*      IT2    : Number of iterations for the integration step          *
C*                                                                     *
C*       Coded by S.Kawabata         April '94                         *
C*                                                                     *
C* Modified by S. Frixione (sneg, sigmaneg, and common block brsneg    *
C* have been added)                                                    *
C***********************************************************************
C
C
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL FXN
      PARAMETER (MXDIM = 50)
*
*     JFLAG =  0 : First trial for defining grids.
*     JFLAG =  1 : First trial for event accumulation.
*     JFLAG =  2 : Second or more trial for defining grids.
*     JFLAG =  3 : Second or more trial for accumulation.
*                                                                      *
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE2/ ACC1,ACC2,ITMX1,ITMX2
      REAL*4 STIME
      COMMON /BSRSLT/AVGI,SD,CHI2A,STIME,ITG,ITF
      COMMON /BSRNEG/AVGINEG,SDNEG
      CHARACTER*80 ERROR
      COMMON /BWARN1/ NERROR
      COMMON /BWARN2/ ERROR(3,3)
*        INTV = ( 0 / 1 / any ) = ( Batch / Batch(Unix) / Interactive )
*        IPNT = ( 0 / any ) = ( IBM Type / Ascii printer )
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
 
       COMMON/NINFO/ NODEID, NUMNOD
       COMMON /BDATE/ IDATE(3),ITIME(2)
*            IDATE(1) : year        ITIME(1) : hour
*            IDATE(2) : month       ITIME(2) : minute
*            IDATE(3) : day
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
 
*-------------------------------------------------
*     Check the parameters defined by user
*------------------------------------------------------
 
      CALL BSCHCK

* ---------------------------------------------------------------
*          Initialize timer
* ---------------------------------------------------------------
 
       CALL BSDATE
 
       JFLAG  = 0
       LU     = 6
       IF( INTV .GT. 1 ) THEN
           CALL BSPRNT( LU, 1, IDUM1, IDUM2 )
       ENDIF

C  -----------------------------------------------------
C     Defining grids
C  -----------------------------------------------------
*
       DO 100 I = 1, NWILD
          IG(I) = 1
  100  CONTINUE
 
       CALL BSETGU
 
       IF( INTV .GT. 1 ) THEN
           CALL BSPRNT( LU, 4, IDUM1, IDUM2 )
       ENDIF
 
       CALL BSUTIM( 0, 2 )
 
*     ===================
       CALL BSINTG( FXN )
*     ===================        For a parallel computer
C                                      CALL BSCAST( JFLAG, 1 )
 
*  ----------------------------------------------------
*     Accumulation to make probability distribution
*  ----------------------------------------------------
*     ===================
       CALL BSINTG( FXN )
*     ===================        For a parallel computer
C                                      CALL BSCAST( JFLAG, 1 )
       S     = AVGI
       SIGMA = SD
       CTIME = STIME
       IT1   = ITG
       IT2   = ITF
       SNEG     = AVGINEG
       SIGMANEG = SDNEG
 
       CALL BSUTIM( 0, 2 )
       TIMEB2 = RTIME
 
       IF( NERROR .GT. 0 ) THEN
           WRITE(6,9900)
 9900      FORMAT(1X,'****************************************',
     .               '***************************************',
     .           /1X,'* (((( Warning in the integration step ',
     .               '))))                                   *',
     .           /1X,'*                                      ',
     .               '                                       *')
           DO 990 J = 1,NERROR
           DO 990 I = 1,3
              WRITE(6,9901) ERROR(I,J)
 9901         FORMAT(1X,A79)
  990      CONTINUE
           WRITE(6,9902)
 9902      FORMAT(1X,'*                                      ',
     .               '                                       *',
     .           /1X,'*(( Suggestion ))                      ',
     .               '                                       *',
     .           /1X,'* (1) Try integration again with larger ',
     .               'number of sample points than this job.*',
     .           /1X,'* or                                   ',
     .               '                                       *',
     .           /1X,'* (2) The integral variables are not sui',
     .               'ted for the function.                 *',
     .           /1X,'*     Take another integral variables !!',
     .               '                                      *',
     .           /1X,'*                                       ',
     .               '                                      *',
     .           /1X,'****************************************',
     .               '***************************************')
       ENDIF
 
       IF( INTV .GT. 1 ) THEN
           CALL BSPRNT( LU, 2, IDUM1, IDUM2 )
       ENDIF
 
       RETURN
       END


************************************************************************
*    ===================                                               *
      SUBROUTINE BSCHCK
*    ===================                                               *
* ((Purpose))                                                          *
*     To check user's initialization parameters.                       *
*                                                                      *
*        Coded by S.Kawabata        Oct. '85                           *
*                                                                      *
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( MXDIM = 50)
      COMMON /BPARM1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BPARM2/ ACC1,ACC2,ITMX1,ITMX2
 
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE1/ XLT(MXDIM),XUT(MXDIM),NDIMT,NWILDT,
     .               IGT(MXDIM),NCALLT
      COMMON /BASE2/ ACC1T,ACC2T,ITMX1T,ITMX2T
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
      COMMON /XHCNTL/ LOCK
 
      LOCK  = 1
 
      IF( IBASES .NE.  1 ) THEN
          WRITE(6,9000)
 9000     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   BSINIT was not called before calling BASES  *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
          STOP
      ENDIF
 
 
      IF( NDIM .LT. 1) THEN
          WRITE(6,9100)
 9100     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   NDIM was not set before calling BASES.      *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
          STOP
      ENDIF
 
      NDIMT = NDIM
 
      DO 200 I = 1,NDIM
         IF( XU(I) .LE. -1.0D37) THEN
             WRITE(6,9200) I,I
 9200        FORMAT(
     .        5X,'*************************************************',
     .       /5X,'*                                               *',
     .       /5X,'*   XL(',I6,' ).  XU(',I6,' ) were not set      *',
     .       /5X,'*    before calling BASES.                      *',
     .       /5X,'*   Process was terminated due to this error.   *',
     .       /5X,'*                                               *',
     .       /5X,'*************************************************')
             STOP
         ENDIF
 
         IGT(I)  = IG(I)
         XLT(I)  = XL(I)
         XUT(I)  = XU(I)
 
  200 CONTINUE
C
C  Change the maximum number of the wild variables
C 10 ===> 15
      IF( NWILD .LT.  0) THEN
          NWILD = MIN( NDIM, 15)
          WRITE(6,9300) NWILD
 9300     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   NWILD was not set before calling BASES.     *',
     .    /5X,'*                                               *',
     .    /5X,'*   NWILD is set equal to the value(',I6,' ).   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
      ELSE
     .IF( NWILD .GT. 15) THEN
          NWILDO = NWILD
          NWILD  = MIN( NDIM, 15)
          WRITE(6,9400) NWILDO, NWILD
 9400     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   NWILD(',I6,' ) was too large number.        *',
     .    /5X,'*                                               *',
     .    /5X,'*   NWILD is set equal to the value(',I6,' ).   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
      ENDIF
 
      NWILDT = NWILD
      NCALLT = NCALL
 
      ITMX1T = ITMX1
      ITMX2T = ITMX2
      ACC1T  = ACC1
      ACC2T  = ACC2
C
      RETURN
      END


C***********************************************************************
C*                                                                     *
C*========================                                             *
C*    SUBROUTINE BSETGU                                                *
C*========================                                             *
C*((Function))                                                         *
C*     Initialization of Bases progam                                  *
C*     This is called only when IFLAG=0.                               *
C*     ( IFLAG = 0 ; First Trial of Defining Grid step )               *
C*                                                                     *
C*    Changed by S.Kawabata    Aug. 1984 at Nagoya Univ.               *
C*    Last update              Oct. 1985 at KEK                        *
C*                                                                     *
C* Modified by S. Frixione (common block brcall has been added in      *
C* order to pass npg, ng, and nsp without using base4)                 *
C***********************************************************************
C
      SUBROUTINE BSETGU
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
      COMMON /BASE6/ D(NDMX,MXDIM),
     .               ALPH,XSAVE(NDMX,MXDIM),XTI,XTSI,XACC,ITSX
      COMMON /BRCALL/ING,INPG,INSP
 
      DIMENSION  XIN(NDMX)
      DATA  ONE/ 1.0D0/
C
C---------------------------------------------------------------
C           Define the number of grids and sub-regions
C---------------------------------------------------------------
C==> Determine NG : Number of grids
          NG    = (NCALL/2.)**(1./NWILD)
         IF(NG .GT. 25) NG  = 25
  100    IF(NG .LT.  2) NG  =  1
         IF(NG**NWILD .GT. LENG) THEN
            NG  = NG - 1
            GO TO 100
         ENDIF
C
C==> Determine ND : Number of sub-regions
          M     = NDMX/NG
          ND    = M*NG
C
C==> Determine NPG: Number of sampling points per subhypercube
          NSP   = NG**NWILD
          NPG   = NCALL/NSP

          ING    = NG
          INPG   = NPG
          INSP   = NSP
 
          XI(1,1)= ONE
          MA(1)  = 1
          DX(1)  = XU(1)-XL(1)
 
          IF( NDIM .GT. 1 ) THEN
              DO 130 J = 2,NDIM
                 XI(1,J)= ONE
                 DX(J)  = XU(J)-XL(J)
                 IF( J .LE. NWILD ) THEN
                    MA(J)  = NG*MA(J-1)
                 ENDIF
  130         CONTINUE
          ENDIF
C
C---------------------------------------------------------------
C           Set size of subregions uniform
C---------------------------------------------------------------
          NDM   = ND-1
          RC    = ONE/ND
          DO 155 J =1,NDIM
             K     = 0
             XN    = 0.D0
             DR    = XN
             I     = K
  140        K     = K+1
             DR    = DR+ONE
             XO    = XN
             XN    = XI(K,J)
  145       IF(RC .GT. DR) GO TO 140
             I     = I+1
             DR    = DR-RC
             XIN(I)= XN-(XN-XO)*DR
            IF(I .LT. NDM) GO TO 145
             DO 150 I  = 1,NDM
                XI(I,J)= XIN(I)
  150        CONTINUE
             XI(ND,J)  = ONE
  155     CONTINUE
********************************************* Updated Feb.08 '94
          IF( ITSX .GT. 0 ) THEN
              IPSAVE = 1
              XACC    = 1.0D37
              XTI     = 0.0D0
              XTSI    = XACC
              ITSX    = 1
              DO 200 J = 1, NDIM
              DO 200 I = 1, ND
                 XSAVE(I,J) = XI(I,J)
  200         CONTINUE
          ENDIF
C
      RETURN
      END


C***********************************************************************
C*                                                                     *
C*========================                                             *
C*    SUBROUTINE BSETGV( IFLAG )                                       *
C*========================                                             *
C*((Function))                                                         *
C*    Refine the grid sizes                                            *
C*                                                                     *
C*    Coded   by S.Kawabata    Aug. 1984 at Nagoya Univ.               *
C*    Last update              Oct. 1985 at KEK                        *
C*                                                                     *
C***********************************************************************
C
      SUBROUTINE BSETGV( IFLAG )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
      COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
      COMMON /BASE6/ D(NDMX,MXDIM),
     .               ALPH,XSAVE(NDMX,MXDIM),XTI,XTSI,XACC,ITSX
      REAL*4 STIME
      COMMON /BSRSLT/AVGI,SD,CHI2A,STIME,ITG,ITF
*
 
      DIMENSION  XIN(NDMX),R(NDMX),DT(MXDIM),DDX(NDMX)
      DATA  ONE/1.0D0/,ZERO/0.0D0/,N0/0/,N1/1/
*
*========= Save the grid information for the best accuracy ===========
*
      IF( ITSX .GT. 0 ) THEN
          IF( IFLAG .EQ. 0 ) THEN
              IF( IT .GE. 5 ) THEN
                  IF( ( TI .GT. AVGI+SD) .AND. TSI .LT. XTSI ) THEN
                      DO 400 J = 1, NDIM
                      DO 400 I = 1, ND
                         XSAVE(I,J) = XI(I,J)
  400                 CONTINUE
                      XACC         = TACC
                      ITSX         = IT
                      XTI          = TI
                      XTSI         = TSI
                  ENDIF
              ENDIF
          ELSE
              IF( ( XTI .GT. TI) .AND. XTSI .LT. TSI ) THEN
                  DO 500 J = 1, NDIM
                  DO 500 I = 1, ND
                     XI(I,J) = XSAVE(I,J)
  500             CONTINUE
*                ==========
                   RETURN
*                ==========
              ENDIF
          ENDIF
      ENDIF
 
C======= SMOOTHING THE FUNCTION D(I,J)
C
        CLOGE   = 1.0D0/LOG(10.0D0)
 
        NDM     = ND-1
        DO 780 J= N1,NDIM
         IF( IG(J) .EQ. 1 ) THEN
          DDX(1)= 0.5D0*(D(1,J) + D(2,J))
          DO 710 I=2,NDM
            DDX(I)= (D(I+1,J) + D(I,J) + D(I-1,J))/3.D0
  710     CONTINUE
          DDX(ND) = 0.5D0*(D(NDM,J) + D(ND,J))
          DT(J) = 0.D0
          DO 720 I = 1, ND
             D(I,J) = DDX(I)
             DT(J)  = DT(J)+D(I,J)
  720     CONTINUE
C
C=========== REDEFINE THE GRID
C
 
          DTLOG   = LOG(DT(J))
          DT10    = CLOGE*DTLOG
          RC    = ZERO
          DO 730 I= N1,ND
            R(I)  = ZERO
            IF(D(I,J) .GT. ZERO) THEN
               DILOG = LOG(D(I,J))
               IF( DT10 - CLOGE*DILOG  .LE. 70.0D0 ) THEN
                   XO    = DT(J)/D(I,J)
                   R(I)  = ((XO-ONE)/(XO*(DTLOG-DILOG)))**ALPH
               ELSE
C                  XO    = DT(J)/D(I,J)
                   R(I)  = (DTLOG-DILOG)**(-ALPH)
               ENDIF
            ENDIF
            RC    = RC+R(I)
  730     CONTINUE
          RC    = RC/ND
          K     = N0
          XN    = N0
          DR    = XN
          I     = K
  740  K     = K + N1
          DR    = DR+R(K)
          XO    = XN
          XN    = XI(K,J)
  750 IF(RC.GT.DR)GO TO 740
          I     = I + N1
          DR    = DR-RC
          XIN(I)= XN-(XN-XO)*DR/R(K)
      IF(I.LT.NDM)GO TO 750
          DO 760 I= N1,NDM
            XI(I,J)= XIN(I)
  760     CONTINUE
          XI(ND,J)= ONE
         ENDIF
  780   CONTINUE
C
      RETURN
      END


C***********************************************************************
C*                                                                     *
C*========================                                             *
C*    SUBROUTINE BSGETW( WEIGHT )                                      *
C*========================                                             *
C*((Function))                                                         *
C*    Get Weight                                                       *
C*                                                                     *
C*    Coded   by T.Ishikawa    Jun. 1995 at KEK                        *
C*    Last update              Jun. 1995 at KEK                        *
C*                                                                     *
C***********************************************************************
C
      SUBROUTINE BSGETW( WEIGHT )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
*
*========= Save the grid information for the best accuracy ===========
*
      WEIGHT = WGT
C
      RETURN
      END


***********************************************************************
*============================                                         *
      SUBROUTINE BSINFO( LU )
*============================                                         *
*((Purpose))                                                          *
*    Print the information for                                        *
*        (1) BASES parameters                                         *
*        (2) Computer time information                                *
*        (3) Convergency behavior of the Grid optimization step       *
*        (4) Convergency behavior of the integration step             *
*(( Input ))                                                          *
*    LU  :  Logical unit number of printer                            *
*                                                                     *
*           by S.Kawabata    March 1994 at KEK
*                                                                     *
***********************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*4 STIME
      COMMON /BSRSLT/AVGI,SD,CHI2A,STIME,ITG,ITF
 
*  Print Title
 
      CALL BSPRNT( LU, 1, IDUM1, IDUM2 )
 
*  Print Bases parameters
 
      CALL BSPRNT( LU, 4, IDUM1, IDUM2 )
 
*  Print Computing time information
 
      CALL BSPRNT( LU, 3, IDUM1, IDUM2 )
 
*  Print Convergency Behaviors
 
      DO 100 ISTEP = 0, 1
         ITX  = ITG
         IF( ISTEP .EQ. 1 ) ITX = ITF

      IF( ITX .GT. 0 ) THEN

         CALL BSPRNT( LU, 8, ITX, ISTEP )
 
      ENDIF
  100 CONTINUE
 
      RETURN
      END


************************************************************************
*    ===================                                               *
      SUBROUTINE BSINIT
*    ===================                                               *
* ((Purpose))                                                          *
*     Initialization of BASE50/SPRING50.                               *
*     Function of this routine is                                      *
*       (0) Set the size of histogram and scatter plot buffers         *
*       (1) Set the parameters INTV and IPNT                           *
*             INTV = ( 0 / 1 / any )                                   *
*                  = ( Batch / Batch(Unix) / Interactive )             *
*             IPNT = ( 0 / any )                                       *
*                  = ( IBM Type / Ascii printer )                      *
*       (2) Set the acceleration factor ALPHA by 1.5                   *
*            The range of this value is from 0.0 to 2.0.               *
*            ALPHA = 0.0 results in no grid-optimization.              *
*       (3) Set the grid-optimization flag IGOPT ( Default value 0 )   *
*             IGOPT = 0  :  The grid is optimized by VEGAS algorithm   *
*             IGOPT = 1  :  The grid is optimized so that the accuracy *
*                           of each iteration be minimized.            *
*       (4) Set Node-ID number NODEID and the number of nodes NUMNOD   *
*       (5) Set seed of radom number                                   *
*       (6) Set the values of BASES paremeters with default ones.      *
*       (7) Set the values of parameters with non-sense values,        *
*            which should be set again with the true values by User    *
*            before running BASES.                                     *
*                                                                      *
*        Coded by S.Kawabata         March '94                         *
*                                                                      *
* Modified by S. Frixione in order to exclude the built-in             *
* histogramming package. The common block bfxlhisto has been added     *
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50 )
      COMMON /BPARM1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BPARM2/ ACC1,ACC2,ITMX1,ITMX2
 
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE6/ D(NDMX,MXDIM),
     .               ALPH,XSAVE(NDMX,MXDIM),XTI,XTSI,XACC,IGOPT
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
       COMMON/NINFO/ NODEID, NUMNOD
       COMMON /BDATE/ IDATE(3),ITIME(2)
*            IDATE(1) : year        ITIME(1) : hour
*            IDATE(2) : month       ITIME(2) : minute
*            IDATE(3) : day
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
      LOGICAL LHISTO
      COMMON/BFXLHISTO/LHISTO
*=========================================================
* (0) Initialization of timer and Histogram buffer
*     Timer initialization
       CALL BSTIME( TIME0, 0 )
       TIMEB1 = TIME0
       TIMINT = 0
 
*     Histogram buffer initialization
       IF(LHISTO)THEN
         LU  = 6
         CALL BHINIT( LU )
       ENDIF
 
*=========================================================
 
* (1) Set the parameters INTV and IPNT
       INTV  = 2
       IPNT  = 1
* (2) Set the acceleration factor ALPHA by 1.5
       ALPH  = 1.5D0
* (3) Set the grid-optimization flag IGOPT
       IGOPT = 0
* (4) Set Node-ID number NODEID and the number of nodes NUMNOD
*      IF( INTV .EQ. 0 ) THEN
           NODEID = 0
           NUMNOD = 1
*      ELSE
*          NODEID = 0
*          NUMNOD = 1
*      ENDIF
 
C---------------------------------------------------------------
C (5)  Set initial seeds of random number generator
c Modified by SF: this is now in init_of_bases
C---------------------------------------------------------------
c       ISEED = 12345
C
c       CALL DRNSET( ISEED )
C ---------------------------------------------------------------
C (6),(7)  Set BASES parameters equal to default values
C ---------------------------------------------------------------
C
       NDIM   = -1
       NWILD  =  1
       ITMX1  = 15
       ITMX2  = 100
       NCALL  = 1000
       ACC1   = 0.2D0
       ACC2   = 0.01D0
       DO 100 I = 1,MXDIM
          IG(I) = 1
          XU(I)  = -1.0D37
  100  CONTINUE
 
*    Initialization of computing time table of BASES
       DO 200 I = 0, 2
          TIMEBS(I) = 0.0
  200  CONTINUE
 
*-------------------------------------------
*      Don't change IBASES from this value
*-------------------------------------------
       IBASES =  1
 
       RETURN
       END


***********************************************************************
*                                                                     *
*    ==========================                                       *
      SUBROUTINE BSINTG( FXN )
*    ==========================                                       *
*((Function))                                                         *
*    Subroutine performs N-dimensional Monte Carlo integration        *
*    for four vector generation of simulated events                   *
*                                                                     *
*       JFLAG = 0 ; First Trial of Defining Grid                      *
*       JFLAG = 1 ; First Trial of Data Accumulation                  *
*       JFLAG = 2 ; Second Trial of Defining Grid                     *
*       JFLAG = 3 ; Second Trial of Data Accumulation                 *
*                                                                     *
*    Coded   by S.Kawabata    July 1980 at DESY, Hamburg              *
*    Last update              March 1994                              *
*                                                                     *
* Modified by S. Frixione in order to exclude the built-in            *
* histogramming package. The common block bfxlhisto has been added    *
* The common block bsrneg and CBSSGN have been added to treat         *
* functions with sign; CBSSGN must mandatorily be inserted also in    *
* in the integrand function. The commond block CBSWGT has been        *
* added in order to deal with weighted events; it must be inserted    *
* also in the integrand function.                                     *
***********************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)

      EXTERNAL FXN
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE2/ ACC1,ACC2,ITMX1,ITMX2
      COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
      PARAMETER (ITM = 50)
      REAL*4 TIME, EFF, WRONG, TRSLT, TSTD, PCNT
      COMMON /BASE5/ ITRAT(ITM,0:1),TIME(ITM,0:2),EFF(ITM,0:1),
     .               WRONG(ITM,0:1),RESLT(ITM,0:1),ACSTD(ITM,0:1),
     .               TRSLT(ITM,0:1),TSTD(ITM,0:1),PCNT(ITM,0:1)
      COMMON /BASE6/ D(NDMX,MXDIM),
     .               ALPH,XSAVE(NDMX,MXDIM),XTI,XTSI,XACC,ITSX
      REAL*4 STIME
      COMMON /BSRSLT/AVGI,SD,CHI2A,STIME,ITG,ITF
      COMMON /BSRNEG/AVGINEG,SDNEG
      COMMON /BRCALL/ING,INPG,INSP
      COMMON /CBSSGN/BSFSGN
      COMMON /CBSWGT/BSEWGT
      CHARACTER*80 ERROR
      COMMON /BWARN1/ NERROR
      COMMON /BWARN2/ ERROR(3,3)
*
*        INTV = ( 0 / 1 / any ) = ( Batch / Batch(Unix) / Interactive )
*        IPNT = ( 0 / any ) = ( IBM Type / Ascii printer )
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
 
      REAL*8  X(MXDIM)
      INTEGER KG(MXDIM),IA(MXDIM)
 
      COMMON/NINFO/ NODEID, NUMNOD
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
C     REAL*8  TX(2)
      INTEGER NCNODE(2,512),NPNODE(2,512)
C     INTEGER NEFF(2)
      LOGICAL LHISTO
      COMMON/BFXLHISTO/LHISTO
*
*     Parameters for checking convergency
*
      DATA ACLMT,FC / 25.0D0, 5.0D0 /
 
 
      DATA  ONE/ 1.0D0/, ZERO/0.0D0/, LU / 6/
      DATA  N0/0/, N1/1/, HUNDRT/100.0D0/
 
************************************************************************
*                       Initialization Part
************************************************************************
*=======================================================================
*          Determine the number of hypercubes NSP
*=======================================================================
 
      XND     = ND
      NSP     = NG**NWILD
      XJAC    = 1.0D0
      DO  5 I = 1, NDIM
         XJAC = XJAC*DX(I)
    5 CONTINUE
      CALLS   = NSP*NPG
      DXG     = 1.0D0/NG
      DV2G    = DXG**(2*NWILD)/NPG/NPG/(NPG-1)
      DXG     = DXG*XND
 
      IF( NSP .EQ. 1 ) THEN
*=======================================================================
*           Determination of the number of sampling points
*               per node in the single hypercube case
*=======================================================================
          MEX     = MOD(NPG,NUMNOD)
          NPERCP  = NPG/NUMNOD
          NPGT    = 0
          DO  12 NODEX = 1,NUMNOD
             NPGS  = NPGT + 1
             NPGT  = NPGT + NPERCP
             IF( NODEX .LE. MEX ) NPGT = NPGT + 1
             NCNODE(1,NODEX) = 1
             NCNODE(2,NODEX) = 1
             NPNODE(1,NODEX) = NPGS
             NPNODE(2,NODEX) = NPGT
   12     CONTINUE
      ELSE
*=======================================================================
*          Determination of the number of hypercubes
*              per node in many hypercubes case
*=======================================================================
          MEX     = MOD(NSP,NUMNOD)
          NPERCP  = NSP/NUMNOD
          NSPT    = 0
          DO  15 NODEX = 1,NUMNOD
             NSPS  = NSPT + 1
             NSPT  = NSPT + NPERCP
             IF( NODEX .LE. MEX ) NSPT = NSPT + 1
             NCNODE(1,NODEX) = NSPS
             NCNODE(2,NODEX) = NSPT
             NPNODE(1,NODEX) = 1
             NPNODE(2,NODEX) = NPG
   15     CONTINUE
      ENDIF
*=======================================================================
      NEND    = N0
      ATACC   = ZERO
      NERROR  = N0
      NER1    = N0
      NER2    = N0
      NER3    = N0
      SUMTI   = ZERO
      SUMTSI  = ZERO
 

      IF(JFLAG .EQ. N0 .OR. JFLAG .EQ. N1 ) THEN
*-----------------------------------------------------------------------
*        JFLAG = 0  : The first trial of the grid optim. step
*        JFLAG = 1  : The first trial of the integration step
*-----------------------------------------------------------------------
         DO 10 J  = N1,NSP
           DXD(J) = ZERO
           DXP(J) = ZERO
   10    CONTINUE
*       -----------------
         ISTEP   = JFLAG
*       -----------------
         IT1   = N1
         SI    = ZERO
         SI2   = ZERO
         SWGT  = ZERO
         SCHI  = ZERO
         SINEG    = ZERO
         SI2NEG   = ZERO
         SWGTNEG  = ZERO
         SCHINEG  = ZERO
*       =============
         IF(LHISTO)THEN
           CALL BHRSET
         ENDIF
*       =============
         NSU     = N0
         SCALLS= ZERO
      ELSE
*-----------------------------------------------------------------------
*        JFLAG = 2  : The continuation of the grid optim. step
*        JFLAG = 3  : The continuation of the integration step
*-----------------------------------------------------------------------
C        IF( JFLAG .EQ. 2 ) THEN
*           -------------
C            ISTEP  = N0
*           -------------
C        ELSE
C    .   IF( JFLAG .EQ. 3 ) THEN
*           -------------
C            ISTEP  = N1
*           -------------
C        ELSE
C                *****************
C                      STOP
C                *****************
C         ENDIF
C
C         IT1   = IT + 1
      ENDIF
 
*------- Set the expected accuracy and the max. iteration number -------
 
      ITMX   = ITMX1
      ACC    = ACC1*0.01D0
      IF( ISTEP .EQ. N1 ) THEN
         ITMX = ITMX2
         ACC  = ACC2*0.01D0
      ENDIF
 
*-------- Print the title of the convergency behavior table -----------
*                  in the interactive mode
      IF( INTV .GT. 1 ) THEN
*         -----------------------------------
           CALL BSPRNT( LU, 5, ISTEP, IDUM2 )
*         -----------------------------------
      ENDIF
      NEGFLG     = 0
 
*    =====================
      CALL BSUTIM( 0, 2 )
*    =====================
 

*********************************************************************
*               Main Integration Loop
*********************************************************************
*    ========
      DO 500  IT = IT1,ITMX
*    ========
*=======================================================================
*                 Initialization for the iteration
*=======================================================================
 
         SCALLS  = SCALLS + CALLS
         NGOOD   = N0
         NEGTIV  = N0
         TI      = ZERO
         TSI     = TI
         TINEG   = ZERO
         TSINEG  = TINEG
 
         IF( ISTEP .EQ. N0 ) THEN
             DO 200 J= N1,NDIM
             DO 200 I=1,ND
                D(I,J)= TI
  200        CONTINUE
         ENDIF
 
         NODEX  = NODEID
         IF( NODEID .EQ. 0 )  NODEX = NUMNOD
 
*---------------------------------------------------------------------
*        Distributing hyper cubes to NumNode nodes
*           NCNODE(1,NODEX)   : 1st cube number for the node NODEX
*           NCNODE(2,NODEX)   : Last cube number for the node NODEX
*                    NODEX    : node number 1 => NumNode(=0)
*                    NODEX    : node number 1 => NumNode(=0)
*---------------------------------------------------------------------
 
         NSP1  = NCNODE(1,NODEX)
         NSP2  = NCNODE(2,NODEX)
*                                 Dummy loopfor a parallel processor
C                                 IF( NSP1 .GT. 1 ) THEN
C                                     CALL DRLOOP( NDIM*NPG*(NSP1-1) )
C                                 ENDIF
 
*=====================================================================
*      Loop for hypercube from NSP1 to NSP2 in the NodeX-th node
*=====================================================================
*       ========
         DO 400 NCB = NSP1, NSP2
*       ========
            FB      = 0.0
            F2B     = 0.0
            FBNEG   = 0.0
            F2BNEG  = 0.0
            NP      = NCB - 1
            IF( NWILD .GT. 1 ) THEN
                DO 210 J = 1,NWILD-1
                   NUM   = MOD(NP,MA(J+1))
                   KG(J) = NUM/MA(J) + 1
  210           CONTINUE
            ENDIF
            KG(NWILD)     = NP/MA(NWILD) + 1
 
*---------------------------------------------------------------------
*       If number of hypercubes is only one,
*        Distributing sampling points to NumNode nodes
*           NPNODE(1,NODEX)   : 1st sample point for the node NODEX
*           NPNODE(2,NODEX)   : Last sample point for the node NODEX
*                    NODEX    : node number 1 => NumNode(=0)
*---------------------------------------------------------------------
 
            NPG1  = NPNODE(1,NODEX)
            NPG2  = NPNODE(2,NODEX)
*                                 Dummy loop for a parallel processor
C                                 IF( NPG1 .GT. 1 ) THEN
C                                     CALL DRLOOP( NDIM*(NPG1-1) )
C                                 ENDIF
 
*=====================================================================
*          Loop for sampling points from NPG1 to NPG2
*                in the single hypercube case
*=====================================================================
*          ========
            DO 300 NTY = NPG1,NPG2
*          ========
*---------------------------------------------------------------------
*        Determine the integration variables by random numbers
*---------------------------------------------------------------------
 
               WGT   = XJAC
               DO 250 J= 1,NDIM
                  IF( J .LE. NWILD ) THEN
                      XN  = (KG(J)-DRN(IDUMY))*DXG+1.D0
                  ELSE
                      XN  = ND*DRN(IDUMY)+1.D0
                  ENDIF
                  IA(J)   = XN
                  IAJ     = IA(J)
                  IF( IAJ .EQ. 1) THEN
                      XO  = XI(IAJ,J)
                      RC  = (XN-IA(J))*XO
                  ELSE
                      XO  = XI(IAJ,J)-XI(IAJ-1,J)
                      RC  = XI(IAJ-1,J)+(XN-IAJ)*XO
                  ENDIF
                  X(J)    = XL(J)+RC*DX(J)
                  WGT     = WGT*XO*XND
  250          CONTINUE
*-----------------------------------------------------------------------
*                     =======
c The Vegas weight passed to the integrand function is zero in
c the grid optimisation step
               BSEWGT = 0.D0
               IF(JFLAG.EQ.1)BSEWGT = WGT/(ITMX2*INSP*INPG)
               FXG  =  FXN(X)*WGT
               FXGNEG  =  FXG*BSFSGN
*                     =======
*-----------------------------------------------------------------------
*             Check the value of the integrand
*-----------------------------------------------------------------------
 
               IF( FXG .NE. 0.0 ) THEN
                   NGOOD = NGOOD + 1
                   IF( ISTEP .EQ. 1 ) THEN
                       DXD(NCB) = DXD(NCB) + FXG
                       IF( FXG .GT. DXP(NCB) ) DXP(NCB) = FXG
                   ENDIF
                   IF( FXG .LT. 0.0 ) THEN
                       NEGTIV= NEGTIV+ 1
                       IF( NEGFLG .EQ. 0 ) THEN
                          WRITE(6,9200) IT,NODEID
 9200                     FORMAT(1X,
     .                       '******* WARNING FROM BASES ********',
     .                       '***********',
     .                       /1X,'*  Negative FUNCTION at IT =',I3,1X,
     .                       ', node = ',I3,1X,'*',
     .                       /1X,'***********************************',
     .                       '***********')
                          NEGFLG  = 1
                       ENDIF
                   ENDIF
               ENDIF
 
*-----------------------------------------------------------------------
*              Accumulation of FXG and FXG*FXG
*-----------------------------------------------------------------------
 
               F2    = FXG*FXG
               FB    = FB + FXG
               F2B   = F2B + F2
               F2NEG    = FXGNEG*FXGNEG
               FBNEG    = FBNEG + FXGNEG
               F2BNEG   = F2BNEG + F2NEG
               IF( ISTEP .EQ. 0 ) THEN
                   DO 260  J = 1,NDIM
                      D(IA(J),J)= D(IA(J),J)+F2
  260              CONTINUE
               ENDIF
*======
  300       CONTINUE
*======
*------------------------------------------- for a parallel processor
*                                 Dummy loop for a parallel processor
C                                 IF( NPG2 .LT. NPG ) THEN
C                                     CALL DRLOOP(NDIM*(NPG-NPG1))
C                                 ENDIF
*                                 Global sum of FB and F2B
C                                 IF( NSP .EQ. 1 ) THEN
C                                     CALL BSDSUM(  FB, 1 )
C                                     CALL BSDSUM( F2B, 1 )
C                                 ENDIF
*-----------------------------------------------------------------------
 
*-----------------------------------------------------------------------
*         Calculate the estimate and variance in the hypercube
*-----------------------------------------------------------------------
 
            F2B   = DSQRT(F2B*NPG)
            F2S   = (F2B-FB)*(F2B+FB)
            TI    = TI+FB
            TSI   = TSI + F2S
            F2BNEG   = DSQRT(F2BNEG*NPG)
            F2SNEG   = (F2BNEG-FBNEG)*(F2BNEG+FBNEG)
            TINEG    = TINEG+FBNEG
            TSINEG   = TSINEG + F2SNEG
*======
  400    CONTINUE
*======
*------------------------------------------- for a parallel processor
*                                 Dummy loop
C                                 IF( NSP2 .LT. NSP ) THEN
C                                     CALL DRLOOP(NDIM*NPG*(NSP-NSP2))
C                                 ENDIF
 
*                                 Global sum of efficiency and frequency
*                                     of negative valued function
C                                 NEFF(1) = NGOOD
C                                 NEFF(2) = NEGTIV
C                                 CALL BSISUM( NEFF, 2 )
 
C                                 TX(1) = TI
C                                 TX(2) = TSI
C                                 IF( NSP .EQ. 1 ) THEN
C                                     CALL BSDSUM(   TX, 2 )
C                                 ENDIF
 
*                                 Global sum of grid information
C                                 IF( ISTEP .EQ. 0 ) THEN
C                                     NOWORK = NDMX*NDIM
C                                     CALL BSDSUM(    D, NOWORK )
C                                 ENDIF
 
*=====================================================================
*           Compute Result of this Iteration
*=====================================================================
*--------------------------------------------------------------------
*           Accumulate the histogram entries
*--------------------------------------------------------------------
*       -------------
         IF(LHISTO)THEN
           CALL BHSAVE
         ENDIF
*       -------------
*--------------------------------------------------------------------
 
C        TI     = TX(1)
C        TSI    = TX(2)
C        NGOOD  = NEFF(1)
C        NEGTIV = NEFF(2)

 
         TI    = TI/CALLS
         TSI   = TSI*DV2G
         TINEG    = TINEG/CALLS
         TSINEG   = TSINEG*DV2G
**
         IF( TSI .LE. 1.0D-37 ) TSI = 1.0D-37
         IF( TSINEG .LE. 1.0D-37 ) TSINEG = 1.0D-37
**
         TI2   = TI*TI
         TI2NEG   = TINEG*TINEG
 
         IF( NGOOD .LE. 10 ) THEN
*           --------------------------------
             CALL BSPRNT( LU, 9, IDUM1, IDUM2 )
*           --------------------------------
*            *****************
                   STOP
*            *****************
 
         ENDIF
 
*--------------------------------------------------------------------
*               Calculate the cumulative result
*--------------------------------------------------------------------
         WGT   = ONE/TSI
         SI    = SI+TI*WGT
         SWGT  = SWGT+WGT
         SCHI  = SCHI+TI2*WGT
         AVGI  = SI/SWGT
         CHI2A = ZERO
         IF(IT .GT. N1 ) CHI2A = (SCHI - SI*AVGI)/(IT-.999D0)
         SD    = DSQRT(ONE/SWGT)
         WGTNEG   = ONE/TSINEG
         SINEG    = SINEG+TINEG*WGTNEG
         SWGTNEG  = SWGTNEG+WGTNEG
         SCHINEG  = SCHINEG+TI2NEG*WGTNEG
         AVGINEG  = SINEG/SWGTNEG
         SDNEG    = DSQRT(ONE/SWGTNEG)
 
*---------------------------------------------------------------------
*             Save the results in the buffer
*---------------------------------------------------------------------
 
         TSI   = DSQRT(TSI)
         ITX         = MOD( IT, ITM)
         IF( ITX .EQ. 0 ) ITX = ITM
         ITRAT(ITX,ISTEP)  = IT
         EFF  (ITX,ISTEP)  = NGOOD/CALLS*HUNDRT
         WRONG(ITX,ISTEP)  = NEGTIV/CALLS*HUNDRT
         RESLT(ITX,ISTEP)  = AVGI
         ACSTD(ITX,ISTEP)  = SD
         TRSLT(ITX,ISTEP)  = TI
         TACC              = ABS(TSI/TI*HUNDRT)
         TSTD (ITX,ISTEP)  = TACC
         PCNT (ITX,ISTEP)  = ABS(SD/AVGI*HUNDRT)
 
*----------------------------------------------------------------------
*                  Check cumulative accuracy
*----------------------------------------------------------------------
 
         IF( NODEID .EQ. 0 ) THEN
 
*-------------------  Check cumulative accuracy -----------------------
 
             SDAV  = SD/AVGI
             IF((ABS(SDAV) .LE. ACC)) NEND = N1
 
             IF( ISTEP .EQ. N1 ) THEN
                 IF( TACC .GT. ACLMT ) THEN
                     IF( NER1 .EQ. 0 ) THEN
                         NERROR = NERROR + 1
                         WRITE(ERROR(1,NERROR),9900) NERROR,IT,ACLMT
 9900                    FORMAT('* (',I1,') Temp. accuracy of it-#',
     .                         I3,' is too large comparing to',
     .                         F6.2,' percent.',6X,'*')
                         WRITE(ERROR(2,NERROR),9901) TACC,ACLMT 
 9901                    FORMAT('*',8X,'Temp. accuracy (',
     .                         F7.4,' % )  >>   (',
     .                         F7.4,' % )',23X,'*')
                         WRITE(ERROR(3,NERROR),9902)
 9902                    FORMAT('*',77X,'*')
                         NER1  = 1
                     ENDIF
                 ENDIF
                 IF( IT .GT. 1 ) THEN
                     IF(( TI .GT. AVTI+FDEVI ) .OR.
     .                  ( TI .LT. AVTI-FDEVI )      ) THEN
                          IF( NER2 .EQ. 0 ) THEN
                              NERROR = NERROR + 1
                              WRITE(ERROR(1,NERROR),9910) NERROR,IT,FC
 9910                         FORMAT('* (',I1,') Temp. estimate of ',
     .                        'it-#',I3,' fluctuates more than ',
     .                               F4.1,'*average-sigma.',6X,'*')
                              RE = TI
*patch TI:1995/08/25
                              ARE = ABS(RE)
*old                          CALL BSORDR( RE, FX2, ORDER, IORDR )
                              CALL BSORDR( ARE, FX2, ORDER, IORDR )
*patch end
                              RE = TI/ORDER
                              RE1 = AVTI
                              AC  = FDEVI 
*patch TI:1995/08/25
                              ARE1 = ABS(AVTI)
                              AAC  = ABS(FDEVI)
                              IF( ARE1 .GE. AAC ) THEN
                                  CALL BSORDR( ARE1, FX2, ORDR1, IORDR1)
                              ELSE
                                  CALL BSORDR( AAC, FX2, ORDR1, IORDR1)
                              ENDIF
*                             IF( RE1 .GE. AC ) THEN
*                                 CALL BSORDR( RE1, FX2, ORDR1, IORDR1)
*                             ELSE
*                                 CALL BSORDR( AC, FX2, ORDR1, IORDR1)
*                             ENDIF
*patch end
                              RE1 = AVTI/ORDR1
                              AC  = AC/ORDR1
                              WRITE(ERROR(2,NERROR),9911) RE,IORDR, 
     .                                          RE1,AC,IORDR1 
 9911                         FORMAT('*        Temp. Estimate (',
     .                         F10.6,' E',I3,')  >  (',F10.6,'+',F8.6,
     .                         ' ) E',I3,', or',1X,'*')
                              WRITE(ERROR(3,NERROR),9912) RE,IORDR, 
     .                                          RE1,AC,IORDR1 
 9912                         FORMAT('*        Temp. Estimate (',
     .                         F10.6,' E',I3,')  <  (',F10.6,'-',F8.6,
     .                         ' ) E',I3,5X,'*')
                              NER2 = 1
                          ENDIF
                     ENDIF
                     IF( TSI .GT. FDEVI ) THEN
                         IF( NER3 .EQ. 0 ) THEN
                             NERROR = NERROR + 1
                             WRITE(ERROR(1,NERROR),9920) NERROR,IT,FC
 9920                        FORMAT('* (',I1,') Error of it-#',
     .                              I3,' fluctuates more than',F4.1,
     .                              '*average-sigma.',16X,'*')
                             RE1 = TSI
*patch TI:1995/08/25
                             ARE1 = ABS(TSI)
*                            CALL BSORDR( RE1, FX2, ORDER, IORDR)
                             CALL BSORDR( ARE1, FX2, ORDER, IORDR)
*patch end;
                             RE1 = TSI/ORDER
                             AC  = FDEVI 
*patch TI:1995/08/25
                             AAC  = ABS(FDEVI)
*                            CALL BSORDR( AC, FX2, ORDR1, IORDR1)
                             CALL BSORDR( AAC, FX2, ORDR1, IORDR1)
*patch end;
                             AC  = AC/ORDR1
                             WRITE(ERROR(2,NERROR),9921) RE1,IORDR, 
     .                                         AC,IORDR1 
 9921                        FORMAT('*        Temp. Error (',
     .                         F10.6,' E',I3,')  >  (',F10.6,
     .                         ' E',I3,')',18X,'*')
                             WRITE(ERROR(3,NERROR),9902)
                             NER3  = 1
                         ENDIF
                     ENDIF
                 ENDIF
                 SUMTSI = SUMTSI + TSI
                 SUMTI  = SUMTI  + TI
                 AVTSI  = SUMTSI/FLOAT(IT)
                 AVTI   = SUMTI/FLOAT(IT)
                 FDEVI  = FC*AVTSI
             ENDIF
         ENDIF
 
*------------------------------------------- for a parallel processor
 
*                                  Broadcast
C                                  CALL BSCAST( NEND, 1 )
 
*----------------------------------------------------------------------
*        Smoothing the Distribution D(I,J) and refine the grids
*----------------------------------------------------------------------
 
         IF( ISTEP .LE. N0 ) THEN
             IF( IT .EQ. ITMX ) NEND = N1
*           ---------------------
             CALL BSETGV( NEND )
*           ---------------------
         ENDIF
*       ==========================
         CALL BSUTIM( 0, ISTEP )
*       ==========================
 
         TIME (ITX,ISTEP)  = TIMINT
         STIME             = TIMINT
 
*---- Print the convergency behavior table in the interactive mode ----
         IF( INTV .GT. 1 ) THEN
*            ---------------------------------
              CALL BSPRNT ( LU, 6, ISTEP, IDUM2 )
*            ---------------------------------
         ENDIF
 
         IF( NEND .EQ. N1 ) GO TO 600
 
*       ======================
         CALL BSUTIM( 0, 2 )
*       ======================
*======
  500 CONTINUE
*======
      IT    = IT - N1
      NEND  = N1
 
***********************************************************************
*                   Termination of BASES
***********************************************************************
*======
  600 CONTINUE
*======
*---------------------------------------------- For a parallel computer
 
*                                 Global sum of histograms
C                                 CALL BHSUM
*                                 Global sum of probabilities
C                                 CALL BSDSUM(  DXD, NSP )
*                                 Global sum of the max.value in each HC
C                                 CALL BSDSUM(  DXP, NSP )
 
 
*======================= End of the step ? ============================
 
      IF( NEND .EQ. N1 ) THEN
          IF( INTV .GT. 1 ) THEN
*            ---------------------------------
              CALL BSPRNT ( LU, 7, IDUM1, IDUM2 )
*            ---------------------------------
          ENDIF
          IF( ISTEP .EQ. N0) THEN
              JFLAG   = N1
              ITG     = IT
          ELSE
              JFLAG   = N0
              ITF     = IT
          ENDIF
      ENDIF
*    ======================
       CALL BSUTIM( 0, 2 )
*    ======================
 
      RETURN
      END


***********************************************************************
*    ===================================                              *
      SUBROUTINE BSLIST( LU, I, ISTEP )
*    ===================================                              *
* ((purpose))                                                         *
*     Print out results of each iteration and cumulative result       *
* ((Argument))                                                        *
*  (Input)                                                            *
*     LU      : Logical unit number for the printer                   *
*     I       : Address in the arrays of common /BASE5/               *
*     ISTEP   : The Set-Identifier                                    *
*               ISTEP = ( 0 / 1 ) = ( Grid opt. / Integration step )  *
*                                                                     *
*     S. Kawabata   March '94                                         *
***********************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (ITM = 50)
      REAL*4 TIME, EFF, WRONG, TRSLT, TSTD, PCNT
      COMMON /BASE5/ ITRAT(ITM,0:1),TIME(ITM,0:2),EFF(ITM,0:1),
     .               WRONG(ITM,0:1),RESLT(ITM,0:1),ACSTD(ITM,0:1),
     .               TRSLT(ITM,0:1),TSTD(ITM,0:1),PCNT(ITM,0:1)
 
      CALL BSTCNV( TIME(I,ISTEP), IH, MN, IS1, IS2 )
 
      RE  = RESLT(I,ISTEP)
      AC  = ABS(ACSTD(I,ISTEP))
      ARE = ABS(RE)
      IF( ARE .GE. AC) THEN
          CALL BSORDR( ARE, F2, ORDER, IORDR)
      ELSE
          CALL BSORDR(  AC, F2, ORDER, IORDR )
      ENDIF
      RE  = RE/ORDER
      AC  = AC/ORDER
      IEFF = EFF(I,ISTEP)
      WRITE(LU,9631) ITRAT(I,ISTEP),IEFF,WRONG(I,ISTEP),
     .              TRSLT(I,ISTEP),TSTD(I,ISTEP),
     .              RE,AC,IORDR,PCNT(I,ISTEP),IH,MN,IS1,IS2
 9631 FORMAT(I4,I4,F6.2,1P,E11.3, 0P,1X,F6.3,
     .              F10.6,'(+-',F8.6,')E',I3.2,1X,F6.3,
     .          1X,I3,':',I2,':',I2,'.',I2.2)
 
 
      RETURN
      END


C***********************************************************************
C*                                                                     *
C*=============================================                        *
C*    SUBROUTINE BSORDR( VAL, F2, ORDER, IORDR)                        *
C*=============================================                        *
C*((Function))                                                         *
C*    To resolve the real number VAL into mantester and exponent parts.*
C*  When VAL = 1230.0 is given, output are                             *
C*        F2 = 1.2  and ORDER = 4.0.                                   *
C*((Input))                                                            *
C*  VAL  : Real*8 value                                                *
C*((Output))                                                           *
C*  F2   : The upper two digits is given                               *
C*  ORDER: Order is given                                              *
C*  IORDR: Exponent is given                                           *
C*((Author))                                                           *
C*  S.Kawabata                                                         *
C*                                                                     *
C***********************************************************************
 
      SUBROUTINE BSORDR(VAL, F2, ORDER, IORDR)
      IMPLICIT REAL*8 (A-H,O-Z)
 
      IF( VAL .NE. 0.0 ) THEN
          ORDER    =  LOG10( VAL )
          IORDR    =  INT( ORDER )
          IF( ORDER .LT. 0.0D0 ) IORDR = IORDR - 1
          ORDER  = 10.D0**IORDR
          F2     = VAL/ORDER
      ELSE
          IORDR  = 0
          ORDER  = 1.0D0
          F2    = 0.0D0
      ENDIF
 
      RETURN
      END


***********************************************************************
*    =======================================                          *
      SUBROUTINE BSPRNT( LU, ID, IP1, IP2 )
*    =======================================                          *
* ((purpose))                                                         *
*     Print out routine of BASES.                                     *
*  (Argument)                                                         *
*     ID  : Identity number of printouts.                             *
*     IP1... IP2 : Integer                                            *
*  (Author)                                                           *
*     S. Kawabata   May 1992                                          *
*     Last update   March 1994                                        *
***********************************************************************
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE2/ ACC1,ACC2,ITMX1,ITMX2
      COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
      PARAMETER (ITM = 50)
      REAL*4 TIME, EFF, WRONG, TRSLT, TSTD, PCNT
      COMMON /BASE5/ ITRAT(ITM,0:1),TIME(ITM,0:2),EFF(ITM,0:1),
     .               WRONG(ITM,0:1),RESLT(ITM,0:1),ACSTD(ITM,0:1),
     .               TRSLT(ITM,0:1),TSTD(ITM,0:1),PCNT(ITM,0:1)
      REAL*4 STIME
      COMMON /BSRSLT/AVGI,SD,CHI2A,STIME,IT1,ITF
      CHARACTER*51 ICH(0:1)
      CHARACTER*1 CN
*        INTV = ( 0 / 1 / any ) = ( Batch / Batch(Unix) / Interactive )
*        IPNT = ( 0 / any ) = ( IBM Type / Ascii printer )
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
*
       COMMON /BDATE/ IDATE(3),ITIME(2)
*            IDATE(1) : year        ITIME(1) : hour
*            IDATE(2) : month       ITIME(2) : minute
*            IDATE(3) : day
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
      REAL*4 XTIME
*
       COMMON/NINFO/ NODEID, NUMNOD
*
      DATA  ICH / 'Convergency Behavior for the Grid Optimization Step',
     .            'Convergency Behavior for the Integration Step      '/
 

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      common/fstbases/ basesoutput

      if(basesoutput.lt.2) then 
         return
      endif
CCCC  ================================================================


      IF( NODEID .NE. 0 ) RETURN
      CN = CHAR(12)
 
      GO TO ( 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ), ID
C----------------------------------------------------------- BSMAIN
 
  100 IF( IPNT .EQ. 0 ) THEN
          WRITE(LU,9600)
 9600     FORMAT(/1H1,/1H )
      ELSE
          WRITE(LU,9610) CN
 9610     FORMAT(A1)
      ENDIF
      WRITE(LU,9620) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
 9620 FORMAT(55X,'Date: ',I2,'/',I2,'/',I2,2X,I2.2,':',I2.2)
      WRITE(LU,9050)
 9050 FORMAT(
     . 8X,'**********************************************************',
     ./8X,'*                                                        *',
     ./8X,'*     BBBBBBB     AAAA     SSSSSS   EEEEEE   SSSSSS      *',
     ./8X,'*     BB    BB   AA  AA   SS    SS  EE      SS    SS     *',
     ./8X,'*     BB    BB  AA    AA  SS        EE      SS           *',
     ./8X,'*     BBBBBBB   AAAAAAAA   SSSSSS   EEEEEE   SSSSSS      *',
     ./8X,'*     BB    BB  AA    AA        SS  EE            SS     *',
     ./8X,'*     BB    BB  AA    AA  SS    SS  EE      SS    SS     *',
     ./8X,'*     BBBB BB   AA    AA   SSSSSS   EEEEEE   SSSSSS      *',
     ./8X,'*                                                        *',
     ./8X,'*                   BASES Version 5.1                    *',
     ./8X,'*           coded by S.Kawabata KEK, March 1994          *',
     ./8X,'**********************************************************')
 
          RETURN
C----------------------------------------------------------- BSMAIN
 
  200     IF( IPNT .EQ. 0 ) THEN
              WRITE(LU,9600)
          ELSE
              WRITE(LU,9610) CN
          ENDIF
          WRITE(LU,9300)
 9300     FORMAT(20X,
     .         '****** END OF BASES *********')
 
C----------------------------------------------------------- BSMAIN
 
  300 CONTINUE
      WRITE(LU,9305)
 9305 FORMAT(
     .//5X,'<<   Computing Time Information   >>')
 
*     WRITE(LU,9310) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
*9310 FORMAT(/15X,'Start at: ',I2,'/',I2,'/',I2,2X,I2.2,':',I2.2)
*     CALL BSDATE
*     WRITE(LU,9320) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
*9320 FORMAT(15X,'End   at: ',I2,'/',I2,'/',I2,2X,I2.2,':',I2.2)
      WRITE(LU,9330)
 9330 FORMAT(/15X,'(1) For BASES              H: M:  Sec')
      CALL BSTCNV(TIMEBS(2),IH,MN,IS1,IS2)
      WRITE(LU,9340) IH, MN, IS1, IS2
 9340 FORMAT(19X,'Overhead           : ',I3,':',I2,':',I2,'.',I2.2)
      CALL BSTCNV(TIMEBS(0),IH,MN,IS1,IS2)
      WRITE(LU,9350) IH, MN, IS1, IS2
 9350 FORMAT(19X,'Grid Optim. Step   : ',I3,':',I2,':',I2,'.',I2.2)
      CALL BSTCNV(TIMEBS(1),IH,MN,IS1,IS2)
      WRITE(LU,9360) IH, MN, IS1, IS2
 9360 FORMAT(19X,'Integration Step   : ',I3,':',I2,':',I2,'.',I2.2)
      XTIME = TIMEB2 - TIMEB1
      CALL BSTCNV(XTIME,IH,MN,IS1,IS2)
      WRITE(LU,9365) IH, MN, IS1, IS2
 9365 FORMAT(19X,'Go time for all    : ',I3,':',I2,':',I2,'.',I2.2)
      EXTIM  = TIMEBS(1)*1000.0/SCALLS/0.7
      WRITE(LU,9375)
 9375 FORMAT(/15X,'(2) Expected event generation time')
      WRITE(LU,9376) EXTIM
 9376 FORMAT(19X,'Expected time for 1000 events :',F10.2,' Sec')
      RETURN
 
C----------------------------------------------------------- BASES
 
  400 NSP   = NG**NWILD
      MCALL = NSP*NPG
      WRITE(LU,9400) NDIM,NWILD,MCALL,NCALL,ND,NG,NSP
 9400 FORMAT(
     .//5X,'<<   Parameters for BASES    >>',
     .//5X,' (1) Dimensions of integration etc.',
     . /5X,'     # of dimensions :    Ndim    =',I9,3X,'( 50 at max.)',
     . /5X,'     # of Wilds      :    Nwild   =',I9,3X,'( 15 at max.)',
     . /5X,'     # of sample points : Ncall   =',I9,'(real)',
     .                                         I9,'(given)',
     . /5X,'     # of subregions    : Ng      =',I9,' / variable',
     . /5X,'     # of regions       : Nregion =',I9,' / variable',
     . /5X,'     # of Hypercubes    : Ncube   =',I9,
     .//5X,' (2) About the integration variables')
      WRITE(LU,9405)
 9405 FORMAT(10X,'------',2('+---------------'),'+-------+-------')
      WRITE(LU,9410)
 9410 FORMAT(10X,'    i       XL(i)           XU(i)     ',
     .           '  IG(i)   Wild')
      WRITE(LU,9405)
       DO 450 I = 1,NDIM
          IF( I .LE. NWILD ) THEN
          WRITE(LU,9420) I,XL(I),XU(I),IG(I)
 9420     FORMAT(10X,I5,1P,2('  ',E14.6),'  ',3X,0P,I1,3X,
     .                       '   yes')
          ELSE
          WRITE(LU,9421) I,XL(I),XU(I),IG(I)
 9421     FORMAT(10X,I5,1P,2('  ',E14.6),'  ',3X,0P,I1,3X,
     .                        '    no')
          ENDIF
  450  CONTINUE
       WRITE(LU,9405)
       WRITE(LU,9450) ITMX1,ACC1,ITMX2,ACC2
 9450  FORMAT(
     . /5X,' (3) Parameters for the grid optimization step',
     . /5X,'     Max.# of iterations: ITMX1 =',I9,
     . /5X,'     Expected accuracy  : Acc1  =',F9.4,' %',
     .//5X,' (4) Parameters for the integration step',
     . /5X,'     Max.# of iterations: ITMX2 =',I9,
     . /5X,'     Expected accuracy  : Acc2  =',F9.4,' %')
 
          RETURN
C----------------------------------------------------------- BASES
 
  500    IF( INTV .LE. 1 )    RETURN
         ISTEP  = IP1
         IF( IPNT .EQ. 0 ) THEN
             WRITE(LU,9600)
         ELSE
             WRITE(LU,9610) CN
         ENDIF
         WRITE(LU,9620) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
         WRITE(LU,9500) ICH(ISTEP)
 9500    FORMAT(15X,A)
         WRITE(LU,9570)
         WRITE(LU,9550)
 9550    FORMAT(1X,'<- Result of  each iteration ->',
     .          2X,'<-     Cumulative Result     ->',
     .          1X,'< CPU  time >',
     .         /1X,' IT Eff R_Neg   Estimate  Acc %',
     .          2X,'Estimate(+- Error )order  Acc %',
     .          1X,'( H: M: Sec )')
         WRITE(LU,9570)
 9570    FORMAT(1X,7('----------'),'--------')
         RETURN
 
C----------------------------------------------------------- BASES
 
  600    IF( INTV .LE. 1 ) RETURN
         ISTEP  = IP1
         ITX = MOD( IT, ITM)
         IF( ITX .EQ. 0 ) ITX = ITM
 
         CALL BSLIST( LU, ITX, ISTEP )
 
         RETURN
 
  700    IF( INTV .LE. 1 ) RETURN
         WRITE(LU,9570)
 
         RETURN
C----------------------------------------------------------- BASES
 
  800    ITJ    = IP1
         ISTEP  = IP2
         ITX  = MOD( ITJ, ITM )
         IF( ITX .EQ. 0 ) ITX = ITM
 
         IF( ITRAT(1,ISTEP) .EQ. 1 ) THEN
             NDEV   = 1
         ELSE
             NDEV   = 2
             ITFN   = ITM
             ITMN   = 10000
             DO 610 I = 1,ITM
                IF( ITRAT(I,ISTEP) .LT. ITMN ) THEN
                    ITST = I
                    ITMN = ITRAT(I,ISTEP)
                ENDIF
  610        CONTINUE
             IF( ITST .EQ. 1 ) NDEV = 1
         ENDIF
 
         IF( IPNT .EQ. 0 ) THEN
             WRITE(LU,9600)
         ELSE
             WRITE(LU,9610) CN
         ENDIF
         WRITE(LU,9620) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
         WRITE(LU,9500) ICH(ISTEP)
         WRITE(LU,9570)
         WRITE(LU,9550)
         WRITE(LU,9570)
 
  625    IF( NDEV .EQ. 1 ) THEN
             ITST = 1
             ITFN = ITX
         ENDIF
 
         DO 650 I = ITST, ITFN
 
            CALL BSLIST( LU, I, ISTEP )
 
  650    CONTINUE
         NDEV  = NDEV - 1
         IF( NDEV .GT. 0 ) GO TO 625
         WRITE(LU,9570)
 
      RETURN
 
C----------------------------------------------------------- BASES
 
  900 WRITE(LU,9950)
 9950 FORMAT(1X,'******** FATAL ERROR IN BASES **************',
     .      /1X,'There are no enough good points in this iteration.',
     .      /1X,'Process was terminated due to this error.')
 
      RETURN
 
C-----------------------------------------------------------------
 1000 LOOP = IP1
      IF( IP2 .NE. 0 ) THEN
          IF( IPNT .EQ. 0 ) THEN
              WRITE(LU,9600)
           ELSE
              WRITE(LU,9610) CN
           ENDIF
           WRITE(LU,9620) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
           WRITE(LU,9650)
 9650      FORMAT(
     .      20X,'Results of Integration',
     .     /10X,5('----------'),'------',
     .     /10X,' Loop#  Estimate(+- Error )order',
     .                     '  It1  It2 ( H: M: Sec )',
     .     /10X,5('----------'),'------')
      ENDIF
 
      RE  = AVGI
      AC  = ABS(SD)
      ARE = ABS(RE)
      IF( ARE .GE. AC) THEN
          CALL BSORDR( ARE, F2, ORDER, IORDR)
      ELSE
          CALL BSORDR(  AC, F2, ORDER, IORDR )
      ENDIF
      RE  = RE/ORDER
      AC  = AC/ORDER
      CALL BSTCNV( STIME, IH, MN, IS1, IS2)
      WRITE(LU,9660) LOOP,RE,AC,IORDR,IT1,IT,IH,MN,IS1,IS2
 9660 FORMAT(10X,I6,F10.6,'(+-',F8.6,')E',I3.2,2I5,
     .        1X,I3,':',I2,':',I2,'.',I2.2,
     .      /10X,5('----------'),'------')
 
      RETURN
      END


C***********************************************************************
C*                                                                     *
C*========================                                             *
C*    SUBROUTINE BSPUTW( WEIGHT )                                      *
C*========================                                             *
C*((Function))                                                         *
C*    Put Weight                                                       *
C*                                                                     *
C*    Coded   by T.Ishikawa    Jun. 1995 at KEK                        *
C*    Last update              Jun. 1995 at KEK                        *
C*                                                                     *
C***********************************************************************
C
      SUBROUTINE BSPUTW( WEIGHT )
C
      IMPLICIT REAL*8 (A-H,O-Z)
      COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
*
*========= Save the grid information for the best accuracy ===========
*
      WGT = WEIGHT
C
      RETURN
      END


************************************************************************
*                                                                      *
*    ==========================                                        *
      SUBROUTINE BSREAD( LUN )
*    ==========================                                        *
* ((Function))                                                         *
*     Read temporary result from the logocal unit LUN                  *
* ((Auther))                                                           *
*     S.Kawabata    June '90 at KEK                                    *
*                                                                      *
************************************************************************
 
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ ND1(5*MXDIM+3)
*     COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
*    .               IG(MXDIM),NCALL
C     COMMON /BASE2/ ND2(6)
*     COMMON /BASE2/ ACC1,ACC2,ITMX1,ITMX2
      COMMON /BASE3/ ND3(11)
*     COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
      COMMON /BASE4/ ND4(2*MXDIM*(NDMX+1)+4*LENG+MXDIM+3)
*     COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
*    .               ND,NG,NPG,MA(MXDIM)
      PARAMETER (ITM  = 50 )
*     COMMON /BASE5/ ND5(22*ITM)
      COMMON /BASE5/ ND5(23*ITM)
*     REAL*4 TIME, EFF, WRONG, TRSLT, TSTD, PCNT
*     COMMON /BASE5/ ITRAT(ITM,0:1),TIME(ITM,0:2),EFF(ITM,0:1),
*    .               WRONG(ITM,0:1),RESLT(ITM,0:1),ACSTD(ITM,0:1),
*    .               TRSLT(ITM,0:1),TSTD(ITM,0:1),PCNT(ITM,0:1)
      COMMON /RANDM/ ND6(45)
 
      PARAMETER ( NHS = 50, NSC = 50 )
      COMMON /PLOTH/ NPH(18*(NHS+NSC)+29),NW
      COMMON /PLOTB/ IBUF( 281*NHS + 2527*NSC )
*     INTEGER*4 XHASH,DHASH,NHIST,MAPL,IFBASE,NSCAT,MAPD
*     COMMON/PLOTH/ XHASH(NHS+1,13),DHASH(NSC+1,14),IFBASE(NHS),
*    .              NHIST, MAPL(4,NHS),
*    .              NSCAT, MAPD(4,NSC),
*    .              NW
 
      COMMON/NINFO/ NODEID, NUMNOD
 
      IF( NODEID .NE. 0 ) RETURN
 
      REWIND LUN
      READ(LUN) ND1,ND3,ND4,ND5,ND6,NPH
C     READ(LUN) ND1,ND2,ND3,ND4,ND5,ND6,NPH
 
      READ(LUN) NW,(IBUF(I),I=1,NW)
C
      RETURN
      END


************************************************************************
*=================================================
      SUBROUTINE BSTCNV( TIME, IH, MN, IS1, IS2 )
*=================================================
* (Purpose)
*    Resolve TIME in second into IH, MN, IS1, IS2
* (Input)
*    TIME : in the unit of second
* (Output)
*    IH   : Hours
*    MN   : Minute
*    IS1  : Second
*    IS2  : 0.xx Second
* (Author)
*    S.Kawabata 1992 June 15
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*4 TIME
      INTEGER  HOUR
      DATA HOUR, MINUT, N100/ 360000, 6000, 100 /
 
      ISEC  = TIME*N100
      IH    = 0
      MN    = IH
      IF( ISEC .GE. MINUT ) THEN
          ITIME = ISEC
          IF( ISEC .GE. HOUR ) THEN
              IH    = ITIME/HOUR
              IHX   = IH*HOUR
              ITIME = ITIME - IHX
              ISEC  = ISEC - IHX
          ENDIF
          MN    = ITIME/MINUT
          ISEC  = ISEC - MN*MINUT
      ENDIF
      IS1  = ISEC/N100
      IS2  = MOD( ISEC, N100)
 
      RETURN
      END


      SUBROUTINE BSUTIM( JOB, ID )
 
C     COMMON/NINFO/ NODEID, NUMNOD
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
 
*  Prior to call thisroutine, BSTIME( TIME0, 1 ) should be called
*  for initialize the time offset TIME0.
*
*     print *,'bsutim .. job, id ',job,id
      CALL BSTIME( RTIME, 1)
      DTIME      = RTIME - TIME0
 
      IF( JOB .EQ. 0 ) THEN
*       For BASES computing time
*         ID  = 0  : Grid defining step
*               1  : Integration step
*               2  : Others
 
          TIMEBS(ID) = TIMEBS(ID) + DTIME
 
          IF( ID .LE. 1 ) THEN
              TIMINT = TIMINT + DTIME
          ENDIF
      ELSE
*       For SPRING computing time
*         ID  = 0  : Event generation
*               1  : Overhead
*               2  : Others
 
          TIMESP(ID) = TIMESP(ID) + DTIME
 
      ENDIF
 
      TIME0      = RTIME
 
      RETURN
      END


************************************************************************
*                                                                      *
*    ==========================                                        *
      SUBROUTINE BSWRIT( LUN )
*    =====================                                             *
* ((Purpose))                                                          *
*     Read temporary result from disk file.                            *
* ((Auther))                                                           *
*     S.Kawabata  June '90 at KEK                                      *
*                                                                      *
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ ND1(5*MXDIM+3)
*     COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
*    .               IG(MXDIM),NCALL
C     COMMON /BASE2/ ND2(6)
*     COMMON /BASE2/ ACC1,ACC2,ITMX1,ITMX2
      COMMON /BASE3/ ND3(11)
*     COMMON /BASE3/ SCALLS,WGT,TI,TSI,TACC,IT
      COMMON /BASE4/ ND4(2*MXDIM*(NDMX+1)+4*LENG+MXDIM+3)
*     COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
*    .               ND,NG,NPG,MA(MXDIM)
      PARAMETER (ITM  = 50 )
*     COMMON /BASE5/ ND5(22*ITM)
      COMMON /BASE5/ ND5(23*ITM)
*     REAL*4 TIME, EFF, WRONG, TRSLT, TSTD, PCNT
*     COMMON /BASE5/ ITRAT(ITM,0:1),TIME(ITM,0:2),EFF(ITM,0:1),
*    .               WRONG(ITM,0:1),RESLT(ITM,0:1),ACSTD(ITM,0:1),
*    .               TRSLT(ITM,0:1),TSTD(ITM,0:1),PCNT(ITM,0:1)
      COMMON /RANDM/ ND6(45)
 
      PARAMETER ( NHS = 50, NSC = 50 )
      COMMON /PLOTH/ NPH(18*(NHS+NSC)+29),NW
      COMMON /PLOTB/ IBUF( 281*NHS + 2527*NSC )
*     INTEGER*4 XHASH,DHASH,NHIST,MAPL,IFBASE,NSCAT,MAPD
*     COMMON/PLOTH/ XHASH(NHS+1,13),DHASH(NSC+1,14),IFBASE(NHS),
*    .              NHIST, MAPL(4,NHS),
*    .              NSCAT, MAPD(4,NSC),
*    .              NW
 
      COMMON/NINFO/ NODEID, NUMNOD
 
      IF( NODEID .NE. 0 ) RETURN
 
      REWIND LUN
      WRITE(LUN) ND1,ND3,ND4,ND5,ND6,NPH
C     WRITE(LUN) ND1,ND2,ND3,ND4,ND5,ND6,NPH
      IF(NW .EQ. 0 ) NW = 281
      WRITE(LUN) NW,(IBUF(I),I=1,NW)
C
      RETURN
      END


C**********************************************************************
C*======================                                              *
C* FUNCTION DRN( ISEED)                                               *
C*======================                                              *
C*  Machine-independent Random number generator                       *
C*     General purpose Version,  OK as long as >= 32 bits             *
C*((Arguement))                                                       *
C*  ISEED: Seed                                                       *
C*                                                                    *
C**********************************************************************
 
*     REAL FUNCTION DRN*8(ISEED)
      DOUBLE PRECISION FUNCTION DRN(ISEED)
 
      COMMON/RANDM/RDM(31),RM1,RM2,IA1,IC1,M1,IX1,
     .                             IA2,IC2,M2,IX2,
     .                             IA3,IC3,M3,IX3
 
C Generate Next number in sequence
 
      IX1    = MOD( IA1*IX1+IC1, M1 )
      IX2    = MOD( IA2*IX2+IC2, M2 )
      IX3    = MOD( IA3*IX3+IC3, M3 )
      J      = 1 + (31*IX3)/M3
      DRN    = RDM(J)
      RDM(J) = ( FLOAT(IX1)+FLOAT(IX2)*RM2 )*RM1
 
C Omit following statement if function arguement passed by value:
 
      ISEED = IX1
      RETURN
      END


C**********************************************************************
C*============================                                        *
C* Subroutine DRNSET( ISEED )                                         *
C*============================                                        *
C*((Purpose))                                                         *
C*  Initialization routine of                                         *
C*         Machine-independent Random number generator                *
C*         General purpose Version,  OK as long as >= 32 bits         *
C*((Arguement))                                                       *
C*  ISEED: SEED                                                       *
C*                                                                    *
C**********************************************************************
 
      SUBROUTINE DRNSET( ISEED )
 
      COMMON/RANDM/RDM(31),RM1,RM2,IA1,IC1,M1,IX1,
     .                             IA2,IC2,M2,IX2,
     .                             IA3,IC3,M3,IX3
 
      IA1 =    1279
      IC1 =  351762
      M1  = 1664557
      IA2 =    2011
      IC2 =  221592
      M2  = 1048583
      IA3 =   15091
      IC3 =    6171
      M3  =   29201
 
C Initialization
 
      IX1  = MOD( ISEED, M1 )
      IX1  = MOD( IA1*IX1+IC1, M1 )
      IX2  = MOD( IX1, M2 )
      IX1  = MOD( IA1*IX1+IC1, M1 )
      IX3  = MOD( IX1,M3)
      RM1  = 1./FLOAT(M1)
      RM2  = 1./FLOAT(M2)
      DO 100 J = 1,31
         IX1   = MOD( IA1*IX1+IC1, M1 )
         IX2   = MOD( IA2*IX2+IC2, M2 )
         RDM(J)= ( FLOAT(IX1)+FLOAT(IX2)*RM2 )*RM1
  100 CONTINUE
 
      RETURN
      END
c
c
c End of BASES routines
c
c
c
c
c Begin of SPRING routines
c
c
      subroutine init_of_spring(prefix)
c This subroutine is called in order to initialize spring parameters
      implicit none
      integer ispring
      common/sprng0/ispring
      character * 60 prefix,fname
      integer isunit
      parameter (isunit=23)
c
      call fk88strcat(prefix,'_bs.data',fname)
      open(unit=isunit,file=fname,
     #     form='unformatted',status='old')
      call bsread(isunit)
      close(isunit)
      ispring=1
      return
      end


      subroutine run_of_spring(ff,mxevts,mxtrls,nevts,ntrls)
      implicit none
      real * 8 ff
      integer mxevts,mxtrls,nevts,ntrls
      integer mxtry
      parameter (mxtry=50)
      integer iounit
      parameter (iounit=6)
      integer mxtryp,nevent,ntrial,miss
      common/sprng2/mxtryp,nevent,ntrial,miss
      external ff
c

      integer mmdecide
      common/decpar/mmdecide

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      common/fstbases/ basesoutput
CCCC  ================================================================

      nevent=0
      ntrial=0
      dowhile(ntrial.lt.mxtrls.and.nevent.lt.mxevts)
        call spring(ff,mxtry,mxevts)
CCC ===================================================================
CCC   added by fabian stoeckli (fabian.stoeckli@cern.ch)
CCC   13.2.2007
CCC   in order to call correct finalizing procedure
        if(mmdecide.eq.1) then
           call sprfin()
        elseif(mmdecide.eq.2) then
           call sprfhg()
        elseif(mmdecide.eq.3) then
           call sprfvh()
        elseif(mmdecide.eq.4) then
           call sprfvb()
        elseif(mmdecide.eq.5) then
           call sprfqq()
        elseif(mmdecide.eq.6) then
           call sprfsb()
        elseif(mmdecide.eq.7) then
           call sprfst()
        else
           write(*,*) 'ERROR #1'
           stop
        endif
      enddo
      if(basesoutput.gt.2) then
         call spinfo(iounit)
      endif
      nevts=nevent
      ntrls=ntrial
      return
      end


c It appears that the main unweighting routine is sprgen; spring
c seems only to be a wrapper for the former. In sprgen the information
c is available on the values of xx(i) when the event is accepted.
c
c There is a serious logical mistake in spring: the initialization
c is performed only if ibases is different from zero. Ibases is
c a flag which is set to 1 in bsinit, and it is connected to bases,
c and not to spring. Thus, if one runs spring by using the bases data
c files saved at a earlier stage, spring is initialized only the first
c time, and this could force the code to crash, but it could also
c lead to incorrect results without notice.
c
c A new common block /sprng0/ispring has therefore been introduced,
c which controls the initialization of spring. 
************************************************************************
*    ==================================                                *
      SUBROUTINE SPRING(FUNC, MXTRY, MXEVTS )
*    ==================================                                *
*         Main Program for the Event generation program SPRING.        *
*                                                                      *
*        Coded by S.Kawabata        September '84                      *
*                                                                      *
* Modified by S. Frixione in order to exclude the built-in             *
* histogramming package. The common block bfxlhisto has been added     *
* The common block SPRNG0 has been added, to correct the logical       *
* flaw pointed out in the comments above. A further modification       *
* occurred on 6/4/2005. Entry MXEVTS has been added to this routine,   *
* in order to modify the warning relevant to the number of             *
* mis-generations, now issued only after the mis-generations are       *
* than 1% of the total number events to be generated                   *
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      EXTERNAL FUNC
      COMMON/SPRNG0/ISPRING
      COMMON /BASE0/ NDUM,IBASES
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
      COMMON /BDATE/ IDATE(3),ITIME(2)
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
 
      COMMON /SPRNG1/ XND, DXG, XJAC, DXMAX, NSP
      COMMON /SPRNG2/ MXTRYP,NEVENT, NTRIAL,MISS
 
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
      LOGICAL LHISTO
      COMMON/BFXLHISTO/LHISTO
*                                                                      *
*----------------------------- Entry point ----------------------------*
*                                                                      *
*======================================================================*
*                  Initialization of the program                       *
*======================================================================*
*----------------------------------------------------------------------*
*                     initialize timer etc.                            *
*----------------------------------------------------------------------*
*                                                                      *
c       IF( IBASES .GT. 0 ) THEN
       IF( ISPRING .GT. 0 ) THEN
 
           CALL SPCHCK
 
           CALL BSTIME( TIME0, 0 )
           TIMES1 = TIME0
 
           MXTRYP = MXTRY
           INTV   = 0
c           IBASES = 0
           ISPRING = 0
           MISFLG = 0
 
           CALL BSDATE
 
           DO 10 I = 0,2
              TIMESP(I) = 0.0
   10      CONTINUE
*                                                                      *
            IF( MXTRY .LT. 10 ) MXTRY = 50
            NBIN    = MXTRY
            IF( MXTRY .GT. 50) NBIN = 50
            MXTRY1  = MXTRY + 1
            MISS    = 0
            NEVENT  = 0
            NTRIAL  = 0
 

            IF(LHISTO)THEN
              CALL SHINIT( MXTRY1 )
              CALL SHRSET
            ENDIF
*----------------------------------------------------------------------*
*             Make the cumulative probability distribution             *
*----------------------------------------------------------------------*
*                                                                      *
            XND     = ND
            DXG     = XND/NG
            NSP     = NG**NWILD
 
*///// DEBUG
*       MCALL   = NSP*NPG
*       CALL BSPRNT( 4, MCALL, IDUM2, IDUM3, IDUM4 )
*
            XJAC    = 1.0
            DO 50 I = 1, NDIM
               XJAC = XJAC*DX(I)
   50       CONTINUE
            DXMAX   = 0.0D0
            DO 100  I = 1,NSP
               IF( DXD( I ) .LT. 0.0D0 ) THEN
                   WRITE(6,9100) I
 9100              FORMAT(
     .             /5X,'********** FATAL ERROR IN SPRING **********',
     .             /5X,'*     Negative probability was found      *',
     .             /5X,'*        in the ',I6,'-th Hypercube.      *',
     .             /5X,'*******************************************')
                   STOP
               ENDIF
               DXMAX    = DXMAX + DXD( I )
               DXD(I)   = DXMAX
  100       CONTINUE
*        =====================
          CALL BSUTIM( 1, 1 )
*        =====================
      ENDIF
*     =====================
       CALL BSUTIM( 1, 2 )
*     =====================
c      IF( IBASES .EQ. 1 ) THEN
      IF( ISPRING .EQ. 1 ) THEN
          WRITE(6,9000)
 9000     FORMAT(
     .      1X,'**************************************************',
     .     /1X,'*    Flag ISPRING was not equal to "0".          *',
     .     /1X,'*                                                *',
     .     /1X,'*   Process was terminated by this error.        *',
     .     /1X,'*   Call S.Kawabata.                             *',
     .     /1X,'**************************************************')
           STOP
       ENDIF
*                                                                      *
*======================================================================*
*                       Event generation                               *
*======================================================================*
*     =====================
  500  CALL BSUTIM( 1, 1 )
*     =====================
 
*     ==================================
        CALL SPRGEN( FUNC, MXTRY, IRET)
*     ==================================
 
*     =====================
       CALL BSUTIM( 1, 0 )
*     =====================
 
      IF(LHISTO)THEN
        CALL SHFILL( IRET )
      ENDIF
 
      IF( IRET .LE. MXTRY ) THEN
          NTRIAL =NTRIAL + IRET
          NEVENT = NEVENT + 1
          IF(LHISTO)THEN
            CALL SHUPDT
          ENDIF
      ELSE
          NTRIAL =NTRIAL + IRET - 1
          MISS = MISS + 1
          IF( MISFLG .EQ. 0 .AND. MISS .GT. INT(0.01*MXEVTS) ) THEN
              WRITE(6,9600) MXTRY
 9600         FORMAT(1X,'****************************************',
     .                  '****************************************',
     .              /1X,'* (((( Warning ))))                     ',
     .                  '                                       *',
     .              /1X,'*                                       ',
     .                  '                                       *',
     .              /1X,'*  The number of mis-generations is foun',
     .                  'd more than',I3,' times.                  *')
              WRITE(6,9610)
 9610         FORMAT(1X,'*                                       ',
     .                  '                                       *',
     .              /1X,'*(( Suggestion ))                       ',
     .                  '                                       *',
     .              /1X,'* (1) Try integration again with larger ',
     .                  'number of sample points than this job. *',
     .              /1X,'* or                                    ',
     .                  '                                       *',
     .              /1X,'* (2) The integral variables are not sui',
     .                  'ted for the function.                  *',
     .              /1X,'*     Take another integral variables !!',
     .                  '                                       *',
     .              /1X,'*                                       ',
     .                  '                                       *',
     .              /1X,'****************************************',
     .                  '****************************************')
            MISFLG = 1
          ENDIF
          GO TO 500
      ENDIF
*     =====================
  600  CALL BSUTIM( 1, 1 )
*     =====================
 
      RETURN
      END


************************************************************************
*    ===================                                               *
      SUBROUTINE SPCHCK
*    ===================                                               *
* ((Purpose))                                                          *
*     To check user's initialization parameters.                       *
*                                                                      *
*        Coded by S.Kawabata      April  '94                           *
*                                                                      *
************************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER ( MXDIM = 50)
      COMMON /BPARM1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BPARM2/ ACC1,ACC2,ITMX1,ITMX2
 
      COMMON /BASE0/ JFLAG,IBASES
      COMMON /BASE1/ XLT(MXDIM),XUT(MXDIM),NDIMT,NWILDT,
     .               IGT(MXDIM),NCALLT
      COMMON /BASE2/ ACC1T,ACC2T,ITMX1T,ITMX2T
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
 
      IF( NDIM .NE. NDIMT ) THEN
          WRITE(6,9100) NDIM,NDIMT
 9100     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   Given NDIM(',I6,' ) does not match          *',
     .    /5X,'*      to NDIM(',I6,' ) in BASES.               *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
          STOP
      ENDIF
 
      IF( NWILD .NE. NWILDT ) THEN
          WRITE(6,9110) NWILD,NWILDT
 9110     FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   Given NWILD(',I6,' ) does not match         *',
     .    /5X,'*      to NWILD(',I6,' ) in BASES.              *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
          STOP
      ENDIF
 
      DO 200 I = 1,NDIM
         IF( XL(I) .NE. XLT(I) ) THEN
             WRITE(6,9200) I,XL(I),I,XLT(I)
 9200        FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   Given XL(',I3,' ) = ',D15.8,'            *',
     .    /5X,'*      does not match to                        *',
     .    /5X,'*      to XL(',I3,' ) = ',D15.8,' in BASES   *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
             STOP
         ENDIF
         IF( XU(I) .NE. XUT(I) ) THEN
             WRITE(6,9210) I,XU(I),I,XUT(I)
 9210        FORMAT(
     .     5X,'*************************************************',
     .    /5X,'*                                               *',
     .    /5X,'*   Given XU(',I3,' ) = ',D15.8,'            *',
     .    /5X,'*      does not match to                        *',
     .    /5X,'*      to XU(',I3,' ) = ',D15.8,' in BASES   *',
     .    /5X,'*                                               *',
     .    /5X,'*   Process was terminated due to this error.   *',
     .    /5X,'*                                               *',
     .    /5X,'*************************************************')
             STOP
         ENDIF
  200 CONTINUE
 
      RETURN
      END


***********************************************************************
*============================                                         *
      SUBROUTINE SPINFO( LU )
*============================                                         *
*((Purpose))                                                          *
*    Print the information for                                        *
*        (1) BASES parameters                                         *
*        (2) Computer time information                                *
*        (3) Convergency behavior of the Grid optimization step       *
*        (4) Convergency behavior of the integration step             *
*(( Input ))                                                          *
*    LU  :  Logical unit number of printer                            *
*                                                                     *
*           by S.Kawabata    March 1994 at KEK
*                                                                     *
* Modified by S. Frixione in order to exclude the built-in            *
* histogramming package. The common block bfxlhisto has been added    *
***********************************************************************
 
      IMPLICIT REAL*8 (A-H,O-Z)
      COMMON /BDATE/ IDATE(3),ITIME(2)
      COMMON /BSCNTL/ INTV, IPNT, NLOOP, MLOOP
 
      COMMON /SPRNG2/ MXTRY,NEVENT, NTRIAL, MISS
 
      PARAMETER ( NHS = 50, NSC = 50 )
      INTEGER*4 XHASH,DHASH,NHIST,MAPL,IFBASE,NSCAT,MAPD
*     COMMON/PLOTH/ XHASH(ILH,13),DHASH(IDH,14),IFBASE(ILH),
*    .              MAXL, NHIST, MAPL(4,ILH),
*    .              MAXD, NSCAT, MAPD(4,IDH),
*    .              NW
      COMMON/PLOTH/ XHASH(NHS+1,13),DHASH(NSC+1,14),IFBASE(NHS),
     .              NHIST, MAPL(4,NHS),
     .              NSCAT, MAPD(4,NSC),
     .              NW
 
      REAL*4 TIMEBS,TIMINT,TIMESP,TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME1/ TIME0,RTIME,TIMEB1,TIMEB2,TIMES1
      COMMON /BTIME2/ TIMEBS(0:2),TIMINT,TIMESP(0:2)
      REAL*4 XTIME
      LOGICAL LHISTO
      COMMON/BFXLHISTO/LHISTO
 
      CHARACTER*1 CN
 
       IF( IPNT .EQ. 0 ) THEN
           WRITE(LU,9300)
       ELSE
           CN     = CHAR(12)
           WRITE(LU,9350) CN
       ENDIF
 9300  FORMAT(/1H1,////1H )
 9350  FORMAT(A1,////1X)
       WRITE(LU,9360) (IDATE(I),I=1,3),(ITIME(J),J=1,2)
 9360  FORMAT(55X,'Date: ',I2,'/',I2,'/',I2,2X,I2.2,':',I2.2)
       WRITE(LU,9400)
 9400 FORMAT(
     . 8X,'**********************************************************',
     ./8X,'*                                                        *',
     ./8X,'*    SSSSS   PPPPPP   RRRRRR   IIIII  N    NN   GGGGG    *',
     ./8X,'*   SS   SS  PP   PP  RR   RR   III   NN   NN  GG   GG   *',
     ./8X,'*   SS       PP   PP  RR   RR   III   NNN  NN  GG        *',
     ./8X,'*    SSSSS   PPPPPP   RRRRR     III   NNNN NN  GG  GGGG  *',
     ./8X,'*        SS  PP       RR  RR    III   NN NNNN  GG   GG   *',
     ./8X,'*   SS   SS  PP       RR   RR   III   NN  NNN  GG   GG   *',
     ./8X,'*    SSSSS   PP       RR    RR IIIII  NN   NN   GGGGG    *',
     ./8X,'*                                                        *',
     ./8X,'*                  SPRING Version 5.1                    *',
     ./8X,'*           coded by S.Kawabata KEK, March 1994          *',
     ./8X,'**********************************************************')
*                                                                      *
          EFF   = FLOAT(NEVENT)/FLOAT(NTRIAL)*100.D0
          CALL BSTIME( RTIME, 1 )
          XTIME = RTIME - TIMES1
          WRITE(LU,9500) NEVENT,EFF,(TIMESP(I),I=0,2),XTIME,MXTRY,MISS
 9500     FORMAT(/5X,'Number of generated events    =',I10,
     .         /5X,'Generation efficiency         =',F10.3,' Percent',
     .         /5X,'Computing time for generation =',F10.3,' Seconds',
     .         /5X,'               for Overhead   =',F10.3,' Seconds',
     .         /5X,'               for Others     =',F10.3,' Seconds',
     .         /5X,'GO time for event generation  =',F10.3,' Seconds',
     .         /5X,'Max. number of trials MXTRY   =',I10,' per event',
     .         /5X,'Number of mis-generations     =',I10,' times')
 
      IF(LHISTO)THEN
        CALL SPHIST( LU )
      ENDIF
 
      RETURN
      END


C***********************************************************************
C*====================================                                 *
C* SUBROUTINE SPRGEN( F, MXTRY, NTRY )                                 *
C*====================================                                 *
C*                                                                     *
C*     Generation of events according to the probability density       *
C*     which is stored in a disk file.                                 *
C*                                                                     *
C*    Coded   by S.Kawabata   at July,1980                             *
C*    Update     S.Kawabata   September '84                            *
C*                                                                     *
C* Modified by S. Frixione in order to exclude the built-in            *
C* histogramming package. The common block bfxlhisto has been added    *
C***********************************************************************
C
       SUBROUTINE SPRGEN(F,MXTRY,NTRY)
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
      EXTERNAL F
      PARAMETER (MXDIM = 50, NDMX = 50, LENG = 32768)
      COMMON /BASE1/ XL(MXDIM),XU(MXDIM),NDIM,NWILD,
     .               IG(MXDIM),NCALL
      COMMON /BASE4/ XI(NDMX,MXDIM),DX(MXDIM),DXD(LENG),DXP(LENG),
     .               ND,NG,NPG,MA(MXDIM)
 
      COMMON /SPRNG1/ XND, DXG, XJAC, DXMAX, NSP
 
      LOGICAL LHISTO
      COMMON/BFXLHISTO/LHISTO
      DIMENSION Y(MXDIM),KG(MXDIM)
      DATA ONE/1.0D0/
C
C
      RX    = DRN(IDUMY)*DXMAX
C
C  -------------- Binary Search  --------------------------------
C
      IPMIN = 1
      IPMAX = NSP
C
 300  IC    = (IPMIN+IPMAX)/2
        IF(RX .LT. DXD(IC)) THEN
          IPMAX = IC
        ELSE
          IPMIN = IC
        ENDIF
      IF(IPMAX-IPMIN .GT.  2) GO TO 300
C
      IC    = IPMIN-1
 350  IC    = IC+1
      IF(DXD(IC) .LT. RX) GO TO 350
C
C --------------------------------------------------------------------
C      Identify the hypecube number from sequential number IC
C --------------------------------------------------------------------
C
       FMAX  = DXP(IC)
C
       IX    = IC-1
 
       KG(NWILD) = IX/MA(NWILD) + 1
       IF( NWILD .GT. 1 ) THEN
           DO 400 J = 1,NWILD-1
              NUM   = MOD(IX,MA(J+1))
              KG(J) = NUM/MA(J) + 1
  400      CONTINUE
       ENDIF
C
C  ------------------------------------------------------------------
C                     Sample and test a event
C  ------------------------------------------------------------------
C
      DO 600 NTRY = 1,MXTRY
        WGT   = XJAC
        DO 550 J=1,NDIM
          IF( J .LE. NWILD) THEN
             XN    = (KG(J)-DRN(IDUMY))*DXG+ONE
          ELSE
             XN    = ND*DRN(IDUMY) + ONE
          ENDIF
          IAJ   = XN
          IF(IAJ .EQ. 1) THEN
            XO    = XI(IAJ,J)
            RC    = (XN-IAJ)*XO
          ELSE
            XO    = XI(IAJ,J)-XI(IAJ-1,J)
            RC    = XI(IAJ-1,J)+(XN-IAJ)*XO
          ENDIF
          Y(J)  = XL(J) + RC*DX(J)
          WGT   = WGT*XO*XND
  550   CONTINUE
C
*       FX    = F(Y)*WGT
        FF    = F(Y)
        FX    = FF*WGT
        FUNCT = FX/FMAX
C
        IF( FX .GT. 0.0D0 ) THEN
*           IF( DRN(IDUMY) .LE. FUNCT ) GO TO 700
            XJ = DRN(IDUMY)
            IF( XJ .LE. FUNCT ) GO TO 700
*           IF( XJ .LE. FUNCT ) THEN
*               WRITE(6,9999) NTRY,IC,FF,WGT,XJ,FUNCT
*9999           FORMAT(1X,'NTRY,IC,FF,WGT,XJ,FUNCT = ',2I5,4E12.4)
*               GO TO 700
*           ENDIF
        ELSE
     .  IF( FX .LT. 0.0D0 ) THEN
            WRITE(6,9100) IC
 9100       FORMAT(
     .      /5X,'********** FATAL ERROR IN SPRING **********',
     .      /5X,'* A negative value of function was found  *',
     .      /5X,'*        in the ',I6,'-th Hypercube.      *',
     .      /5X,'*******************************************')
            WRITE(6,9405)
 9405       FORMAT(5X,'------',3('+---------------'),'+')
            WRITE(6,9410)
 9410       FORMAT(5X,'    i       XL(i)             X       ',
     .                '     XU(i)')
            WRITE(6,9405)
            DO 450 I = 1,NDIM
                WRITE(6,9420) I,XL(I),Y(I),XU(I)
 9420           FORMAT(5X,I5,1P,3('  ',E14.6))
  450       CONTINUE
            WRITE(6,9405)
            STOP
        ENDIF
C
        IF(LHISTO)THEN
          CALL SHCLER
        ENDIF
C
  600 CONTINUE
 
      NTRY  = MXTRY + 1
 
  700 RETURN
      END
c
c
c End of SPRING routines
c
c
c
c
c Begin of utilities
c
c
      subroutine shinit(mxtry1)
      write(6,*)'shinit is not here'
      stop
      end


      subroutine shrset
      write(6,*)'shrset is not here'
      stop
      end


      subroutine shfill(iret)
      write(6,*)'shfill is not here'
      stop
      end


      subroutine shupdt
      write(6,*)'shupdt is not here'
      stop
      end


      subroutine sphist(lu)
      write(6,*)'sphist is not here'
      stop
      end


      subroutine shcler
      write(6,*)'shcler is not here'
      stop
      end


      subroutine bhinit(lu)
      write(6,*)'bhinit is not here'
      stop
      end


      subroutine bhrset
      write(6,*)'bhrset is not here'
      stop
      end


      subroutine bhsave
      write(6,*)'bhsave is not here'
      stop
      end
c
c
c End of utilities
c
c
