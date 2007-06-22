***********************************************************************
* H1 2006 DPDF Fits parameterization                                  *
* ----------------------------------                                  *
* Reference: H1 Collaboration, A. Aktas et al., "Measurement and QCD  *
* Analysis of the Diffractive Deep-Inelastic Scattering Cross Section *
* at HERA", DESY06-049, hep-ex/0606004, subm. to Eur.Phys.J           *
*                                                                     *
* Contact persons in case of questions:                               *
* Frank-Peter Schilling (frank-peter.schilling@cern.ch)               *
* Paul Newman (newmanpr@mail.desy.de)                                 *
* ------------------------------------------------------------------- *
* This routine returns the pomeron dpdf's (z*pdf(z,Q^2)) and the      * 
* pomeron and reggeon structure functions F2_{IP,IR}(beta,Q^2),       *
* FL_{IP,IR}(beta,Q^2), F2c_{IP,IR}(beta,Q^2), FLc_{IP,IR}(beta,Q^2)  *
* Range of validity: 0.0043 < z < 0.8 ; 8.5 < Q^2 < 1600 GeV^2        *
* Outside, the pdf's are in z an extrapolation beyond the measured    *
* range and in Q^2 evolved to higher/lower scales using the NLO DGLAP *
* equations                                                           *
* The grids are provided for 0.001 < z < 1.0 ; 1 < Q^2 < 30000 GeV^2  *
* ------------------------------------------------------------------- *
* Input:  z:     z-pom (parton momentum fraction)                     *
*         q2:    scale^2 where dpdf's are evaluated (in GeV^2)        *
*         ifit:  1: Fit A  2: Fit B                                   *
*                0: fit used by preceding call (no re-initialization  *
*                   of data tables, much faster than using 1 or 2)    *
* Output: xpq(-6:6):  PDG style array of pomeron dpdf's at (z,Q2)     *
*         f2(2):      Structure function F2  (1=pomeron,2=reggeon)    *
*         fl(2):      Structure function FL  (1=pomeron,2=reggeon)    *
*         c2(2):      Structure function F2c (1=pomeron,2=reggeon)    *
*         cl(2):      Structure function FLc (1=pomeron,2=reggeon)    *
***********************************************************************

      SUBROUTINE QCD_2006(Z,Q2,IFIT,XPQ,F2,FL,C2,CL)

      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

      integer nz,nq2
      double precision q2l,q2u,zl,zu

      parameter (nz=100,nq2=30)
      parameter (zl =0.001d0 , zu=0.99d0 )
      parameter (q2l=1.0d0 , q2u=30000.0d0 )

      PARAMETER (NQ2MAX=nq2,NZMAX=nz)

      INTEGER IQ2,IZ,IFIT
      DIMENSION XPQ(-6:6)
      double precision F2(2),FL(2),C2(2),CL(2)

      integer iwarn
      data iwarn /0/
      save iwarn
      parameter ( maxwarn = 10 )

      DIMENSION U_GRD(NZMAX,NQ2MAX),
     &          G_GRD(NZMAX,NQ2MAX),
     &          F2GRD(NZMAX,NQ2MAX),
     &          FLGRD(NZMAX,NQ2MAX),
     &          C2GRD(NZMAX,NQ2MAX),
     &          CLGRD(NZMAX,NQ2MAX),
     &          F2GRR(NZMAX,NQ2MAX),
     &          FLGRR(NZMAX,NQ2MAX),
     &          C2GRR(NZMAX,NQ2MAX),
     &          CLGRR(NZMAX,NQ2MAX)

      COMMON /CPARAM/ U_GRD,G_GRD,
     &                F2GRD,FLGRD,C2GRD,CLGRD,
     &                F2GRR,FLGRR,C2GRR,CLGRR

C     Initialise Parameterisation from data statements

      IF(IFIT.GT.0) THEN
         IF(IFIT.EQ.1) THEN 
           WRITE(6,*)'[QCD_2006] Initialising H1 2006 DPDF Fit A'
           CALL I_2006_FITA
         ELSEIF(IFIT.EQ.2) THEN 
           WRITE(6,*)'[QCD_2006] Initialising H1 2006 DPDF Fit B'
           CALL I_2006_FITB
         ELSE
           WRITE(6,*)'[QCD_2006] Error: Unknown ifit=',ifit
           STOP
         ENDIF
      ENDIF

c     health warning

      if (z<0.0043d0.or.z>0.8d0
     &    .or.q2.lt.8.5d0.or.q2.gt.1600.d0) then
         if (iwarn.lt.maxwarn) then
            write(6,*)'[QCD_2006] Warning: using DPDFs outside range '//
     &        ' of validity: z,q2:',z,q2
            iwarn=iwarn+1
            if (iwarn.eq.maxwarn) then
              write(6,*)'[QCD_2006] Last warning.'
            endif
         endif
      endif

C     Initialise xpq

      DO I=-6,6
         XPQ(I)=0.D0
      ENDDO
      do i=1,2
         F2(i)=0.D0
         FL(i)=0.D0
         C2(i)=0.D0
         CL(i)=0.D0
      enddo

c     check input
      ZT = Z
      Q2T = Q2
      Q2T=MIN(Q2U-0.01D0,MAX(Q2L,Q2T))
      ZT =MIN(ZU,MAX(ZL,ZT))

c     get binwidth
      DQ2 = DLOG(Q2U/Q2L)/DFLOAT(NQ2-1)
      DZ  = DLOG10(ZU/ZL)/DFLOAT(NZ-1)

C     Lower Grid Point
      IQ2 = INT(DLOG(Q2T/Q2L)/DQ2)+1
      IZ  = INT(DLOG10(ZT/ZL)/DZ)+1

C     Central and boundary z,Q2 points for interpolation
      DLQ2  = DLOG(Q2T)
      DLQ21 = DLOG(Q2L)+DFLOAT(IQ2-1)*DQ2
      DLQ22 = DLOG(Q2L)+DFLOAT(IQ2)*DQ2

      DLZ  = DLOG10(ZT)
      DLZ1 = DLOG10(ZL)+DFLOAT(IZ-1)*DZ
      DLZ2 = DLOG10(ZL)+DFLOAT(IZ)*DZ

C     Light Flavour Singlet
      U_T = XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               U_GRD(IZ,IQ2),  U_GRD(IZ,IQ2+1),
     +               U_GRD(IZ+1,IQ2),U_GRD(IZ+1,IQ2+1))

C     Gluon
      G_T = XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               G_GRD(IZ,IQ2),  G_GRD(IZ,IQ2+1),
     +               G_GRD(IZ+1,IQ2),G_GRD(IZ+1,IQ2+1))

c     f2-pom
      F2P_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               F2GRD(IZ,IQ2),  F2GRD(IZ,IQ2+1),
     +               F2GRD(IZ+1,IQ2),F2GRD(IZ+1,IQ2+1))
c     fL-pom
      FLP_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               FLGRD(IZ,IQ2),  FLGRD(IZ,IQ2+1),
     +               FLGRD(IZ+1,IQ2),FLGRD(IZ+1,IQ2+1))
c     C2-pom
      C2P_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               C2GRD(IZ,IQ2),  C2GRD(IZ,IQ2+1),
     +               C2GRD(IZ+1,IQ2),C2GRD(IZ+1,IQ2+1))
c     CL-pom
      CLP_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               CLGRD(IZ,IQ2),  CLGRD(IZ,IQ2+1),
     +               CLGRD(IZ+1,IQ2),CLGRD(IZ+1,IQ2+1))
c     f2-mes
      F2R_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               F2GRR(IZ,IQ2),  F2GRR(IZ,IQ2+1),
     +               F2GRR(IZ+1,IQ2),F2GRR(IZ+1,IQ2+1))
c     fL-mes
      FLR_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               FLGRR(IZ,IQ2),  FLGRR(IZ,IQ2+1),
     +               FLGRR(IZ+1,IQ2),FLGRR(IZ+1,IQ2+1))
c     C2-mes
      C2R_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               C2GRR(IZ,IQ2),  C2GRR(IZ,IQ2+1),
     +               C2GRR(IZ+1,IQ2),C2GRR(IZ+1,IQ2+1))
c     CL-mes
      CLR_T= XYINT06(DLZ1,DLZ2,DLQ21,DLQ22,DLZ,DLQ2,
     +               CLGRR(IZ,IQ2),  CLGRR(IZ,IQ2+1),
     +               CLGRR(IZ+1,IQ2),CLGRR(IZ+1,IQ2+1))


C     Fill XPQ array
      XPQ(-3) = U_T
      XPQ(-2) = U_T
      XPQ(-1) = U_T
      XPQ( 0) = G_T
      XPQ( 1) = U_T
      XPQ( 2) = U_T
      XPQ( 3) = U_T

      F2(1)=F2P_T
      F2(2)=F2R_T

      FL(1)=FLP_T
      FL(2)=FLR_T

      C2(1)=C2P_T
      C2(2)=C2R_T

      CL(1)=CLP_T
      CL(2)=CLR_T

      RETURN
      END

***********************************************************************

      FUNCTION XYINT06(X1,X2,Y1,Y2,XT,YT,F11,F12,F21,F22)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

      XEXP = (XT-X1)/(X2-X1)
      YEXP = (YT-Y1)/(Y2-Y1)

      XYINT06 = F11 + XEXP*(F21-F11)
     &          + YEXP*(F12 + XEXP*(F22-F12) - (F11 + XEXP*(F21-F11)))

      RETURN
      END
