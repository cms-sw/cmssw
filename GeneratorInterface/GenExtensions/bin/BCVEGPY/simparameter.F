
c******************************************************
c...     some frequently used parameters.
c******************************************************
      subroutine parameters() 
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "inclcon.h"
      common/ppp/pp(4,40),guv(4)
      common/rconst/pi
       double complex colmat, bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)

      hbcvm=pmbc
      hbcm=pmbc
      fmb=pmb
      fmc=pmc

      wwpi=pi
	hbcp=fbcc

      fb1=(3*hbcm**2)/(8*fmb*fmc)
      fb2=(3*(fmb-fmc)*hbcm)/(8*fmb*fmc)
      fb3=hbcm**2/(8*fmb*fmc)
      fb4=((fmb-fmc)*hbcm)/(8*fmb*fmc)

c...the color factor \delta_{ij}/3 of the bound state has
c...been included in the following definiation for
c...the p-wave matrix elements times the factor from bound
c...state, <0|p|0>*c_p.
      cbc1p1=dsqrt(hbcp/(72.0d0*hbcm**3))
      cbcp0 =dsqrt(hbcp/(216.0d0*hbcm**3))
      cbcp1 =dsqrt(hbcp/(144.0d0*hbcm**3))
c...here for simplicity one (hbcm) has been absorbed into
c...matrix element.
      cbcp2 =dsqrt(hbcp/(72.0d0*hbcm))

      return
      end

c************************************************
c...   this is only used for p-wave generation.
c...   some simplified paremters.
c************************************************
      subroutine dparameters()
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "inclcon.h"
      common/ppp/pp(4,40),guv(4)
 
      guv(1)= 1.0d0
      guv(2)=-1.0d0
      guv(3)=-1.0d0
      guv(4)=-1.0d0

      fmc2=fmc**2
      fmb2=fmb**2
      fmb3=fmb**3
      fmc3=fmc**3

      hbcvm2=hbcvm**2
      hbcm2 =hbcm**2
      hbcm3 =hbcm**3
      hbcm4 =hbcm**4
      hbcm5 =hbcm**5

      dhbcvm2=1.0d0/hbcvm2
      dhbcm2 =1.0d0/hbcm2

      ffmcfmb=fmc/(fmb+fmc)

      return
      end

c***********************************************
c******** parameters for s-wave ****************
c***********************************************

	subroutine paraswave(ibco)
c...preamble: declarations.
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	common/rconst/pi
        double complex colmat, bundamp
       common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/wavezero/fbc
      common/coloct/ioctet
	common/octmatrix/coeoct

	ioctet =0
	fbc =1.241d0
	fbcc=dsqrt(3.0d0*fbc**2/pi/pmbc)

c....for color-octet states. 
	if(ibco.eq.7.or.ibco.eq.8) then
	  ioctet =1
	  coeoct =0.2d0  ! the value of \delta_s(v^2)
	                 ! defined in prd71,074012(2005)
	end if
       
	call parameters()
      call dparameters()
      call coupling()
	
	return
	end

c***********************************************
c******** parameters for p-wave ****************
c***********************************************

	subroutine parapwave
c...preamble: declarations.
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	common/rconst/pi
        double complex colmat, bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/wavezero/fbc
      common/coloct/ioctet
	common/octmatrix/coeoct

	ioctet   =0
	fbc =0.44833d0
	fbcc=fbc**2*9.0d0/(2.0d0*pi)
       
	call parameters()
      call dparameters()
      call coupling()
	
	return
	end
