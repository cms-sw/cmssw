      SUBROUTINE FLUX(F,Z,TMIN,TMAX,NSTRU)
      IMPLICIT NONE
* Returns H1 pomeron flux if NSTRU=6
* Returns H1 reggeon flux if NSTRU=7
* Returns flux for user defined structure function if NSTRU=8
* B.Cox and J. Forshaw 11/05/00
      DOUBLE PRECISION F,Z,TMIN,TMAX
      DOUBLE PRECISION alpha,B,alphap
      DOUBLE PRECISION alphar,alpharp,Br,Cr
* H1 best fits 
*      PARAMETER (alpha=1.203,alphap=0.26,B=4.6)
*      PARAMETER (alphar=0.50,alpharp=0.90,Br=2.0,Cr=16.0)
* H1 parameters with no interference (best fit to H1 F2D3 using POMWIG)
      PARAMETER (alpha=1.200,alphap=0.26,B=4.6)
      PARAMETER (alphar=0.57,alpharp=0.90,Br=2.0,Cr=48.0)      
      DOUBLE PRECISION V,W,X      
      INTEGER NSTRU  
      
      if (NSTRU.EQ.9) then
         V = DEXP(-(B+2.D0*alphap*DLOG(1.D0/Z))*TMIN)-
     +        DEXP(-(B+2.D0*alphap*DLOG(1.D0/Z))*TMAX)
         W = 1.D0/(B+2.D0*alphap*DLOG(1.D0/Z))
         X = 1.D0/(Z**(2.D0*alpha-1.D0))
         F = X*W*V
      elseif (NSTRU.EQ.10) then
         V = DEXP(-(Br+2.D0*alpharp*DLOG(1.D0/Z))*TMIN)-
     +        DEXP(-(Br+2.D0*alpharp*DLOG(1.D0/Z))*TMAX)
         W = 1.D0/(Br+2.D0*alpharp*DLOG(1.D0/Z))
         X = 1.D0/(Z**(2.D0*alphar-1.D0))
         F = Cr*X*W*V         
      elseif (NSTRU.EQ.11) then
         V = DEXP(-(B+2.D0*alphap*DLOG(1.D0/Z))*TMIN)-
     +        DEXP(-(B+2.D0*alphap*DLOG(1.D0/Z))*TMAX)
         W = 1.D0/(B+2.D0*alphap*DLOG(1.D0/Z))
         X = 1.D0/(Z**(2.D0*alpha-1.D0))
         F = X*W*V
      else
         write(*,*) 'pomwig : NSTRU must be 9, 10 or 11 in herwig65'
         STOP
      endif
      RETURN
      END
