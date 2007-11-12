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
* This routine returns the pomeron or reggeon flux factors            *
* f_{IP,IR/p}(xpom,t)                                                 *
* Both IP and IR flux factors are normalized such that xpom*flux=1    *
* at xpom=0.003; In addition, the IR flux is multiplied by a further  *
* normalization parameter as determined in the fit to the data        *
* The routine returns either the flux at fixed values (xpom,t), or    *
* the t-integrated flux (integrated from t to tmin)                   *
* ------------------------------------------------------------------- *
* Input:  xpom:  xpomeron value                                       *
*         t:     t value (negative!)                                  *
*         int:   0: flux at (xpom,t) is returned                      *
*                1: t-integrated (t...tmin) flux at xpom is returned  *
*         ifit:  1: Fit A  2: Fit B                                   *
*         ipom:  1: Pomeron flux 2: Reggeon flux                      *
* Output: flux:  flux value                                           *
***********************************************************************

      subroutine h12006flux(xpom,t,int,ifit,ipom,flux)

      implicit none

      integer int,ifit,ipom
      double precision xpom,t,a0,ap,b0,c,flux,norm,dm

      if ((ipom.lt.1).or.(ipom.gt.2)) then
         print *,'[H12006FLUX] Unknown ipom: ',ipom
         stop
      endif
      if ((ifit.lt.1).or.(ifit.gt.2)) then
         print *,'[H12006FLUX] Unknown ifit: ',ifit
         stop
      endif

      if (ipom.eq.1) then ! pomeron
        if (ifit.eq.1) then ! fit a
          a0 = 1.1182d0 
        elseif (ifit.eq.2) then ! fit b
          a0 = 1.1110d0 
        endif
        ap = 0.06d0 
        b0 = 5.5d0 
        c = 1.0d0 

      else ! meson
        a0 = 0.5d0 
        ap = 0.3d0 
        b0 = 1.6d0 
        if (ifit.eq.1) then ! fit a
          c = 0.001697d0 
        elseif (ifit.eq.2) then ! fit b
          c = 0.001398d0 
        endif
      endif

c     normalization
      call rflux2006(0.003d0,a0,ap,b0,-1.d0,1.0d0,1,dm)
      norm=(1./(0.003d0*dm))*c

c     actual flux
      call rflux2006(xpom,a0,ap,b0,t,norm,int,flux)

      return
      end

*******************************************************************************

      subroutine rflux2006(x_pom,a0,ap,b0,tcut,c,int,fl)

      implicit none

      double precision x_pom,a0,tmin,tcut,ap,b0,c,fl,b
      integer int

c     calc min. kinematically  allowed t
      tmin= -(0.93827231D0*x_pom)**2/(1.D0-x_pom)

c     c*xpom**(-(2apom-1))
      fl = c * dexp((2.0d0*a0-1.)*dlog(1.0d0/x_pom))
      b=(b0+2.0d0*ap*dlog(1.0d0/x_pom))

      if (int.eq.0) then
c       at fixed t:  exp(Bt) 
        fl = fl * dexp(b*tcut)
      else
c       t integrated: (1/B)*[exp(-B*tmax)-exp(-B*tmin)]
        fl = fl * (dexp(tmin*b)-dexp(tcut*b))/b
      endif

      return
      end
