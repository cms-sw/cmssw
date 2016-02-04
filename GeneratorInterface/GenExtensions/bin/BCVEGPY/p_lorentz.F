c************************************************************
c...some subroutines for p-wave calculation.
c************************************************************
c*** genpolar is used to generate the polarization vector ***
c*** ep of the vector boson with momentum p and mass pm.  ***       
c************************************************************
      subroutine genpolar3(p,ep,pm)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p(4),ep(4,2:4)

      pxy=dsqrt(p(2)**2+p(3)**2)
      pxyz=dsqrt(p(2)**2+p(3)**2+p(4)**2)

      if(pxy.lt.1.0d-16) then
         if(p(4).gt.0.0d0) then
           sign=1.0d0
         else 
           sign=-1.0d0
         endif 
c******    ploarization vector 1 of momentum p *****
         ep(1,2)=0.0d0
         ep(2,2)=sign
         ep(3,2)=0.0d0
         ep(4,2)=0.0d0
c******    ploarization vector 2 of momentum p *****
         ep(1,3)=0.0d0 
         ep(2,3)=0.0d0
         ep(3,3)=sign
         ep(4,3)=0.0d0 
c******    ploarization vector 3 of momentum p *****
         ep(1,4)=p(4)/pm
         ep(2,4)=0.0d0
         ep(3,4)=0.0d0
         ep(4,4)=p(1)/pm
      else        
c******    ploarization vector 1 of momentum p *****
         ep(1,2)=0.0d0
         ep(2,2)=-p(3)/pxy
         ep(3,2)=p(2)/pxy
         ep(4,2)=0.0d0
c******    ploarization vector 2 of momentum p *****
         ep(1,3)=0.0d0
         ep(2,3)=(p(2)*p(4))/(pxyz*pxy)
         ep(3,3)=(p(3)*p(4))/(pxyz*pxy)
         ep(4,3)=-pxy/pxyz

c******    ploarization vector 3 of momentum p *****
         ep(1,4)=pxyz/pm
         ep(2,4)=p(2)*p(1)/(pxyz*pm)
         ep(3,4)=p(3)*p(1)/(pxyz*pm)
         ep(4,4)=p(4)*p(1)/(pxyz*pm)
      endif

      return
      end 
c*********************************************
      subroutine genpolar(p,ep)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p(4),ep(4,2:3)

      pxy=dsqrt(p(2)**2+p(3)**2)
      pxyz=dsqrt(p(2)**2+p(3)**2+p(4)**2)

      if (pxy.lt.1.0d-16) then
         if(p(4).gt.0.0d0) then
           sign=1.0d0
         else 
           sign=-1.0d0
         endif 
c******    ploarization vector 1 of momentum p *****
         ep(1,2)=0.0d0 
         ep(2,2)=sign
         ep(3,2)=0.0d0
         ep(4,2)=0.0d0
c******    ploarization vector 2 of momentum p *****
         ep(1,3)=0.0d0 
         ep(2,3)=0.0d0
         ep(3,3)=sign
         ep(4,3)=0.0d0 
      else
c******    ploarization vector 1 of momentum p *****
         ep(1,2)=0.0d0
         ep(2,2)=-p(3)/pxy
         ep(3,2)=p(2)/pxy
         ep(4,2)=0.0d0
c******    ploarization vector 2 of momentum p *****
         ep(1,3)=0.0d0
         ep(2,3)=-(p(2)*p(4))/(pxyz*pxy)
         ep(3,3)=-(p(3)*p(4))/(pxyz*pxy)
         ep(4,3)=pxy/pxyz
      endif

      return
      end 
c*************************************************************
c*** gentensor is used to generate the polarization tensor ***
c*** ept with momentum p.                                  ***
c...this is for the general results (m > 0).
      subroutine gentensor(p,ept,pm)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p(4),ept(4,4,2:6),ep(4,2:4)

      call genpolar3(p,ep,pm)

      do i=1,4
       do j=1,4
        ept(i,j,2)=1.0d0/dsqrt(2.0d0)*(ep(i,2)*ep(j,3)+ep(i,3)*ep(j,2))
        ept(i,j,3)=1.0d0/dsqrt(2.0d0)*(ep(i,2)*ep(j,4)+ep(i,4)*ep(j,2))
        ept(i,j,4)=1.0d0/dsqrt(2.0d0)*(ep(i,4)*ep(j,3)+ep(i,3)*ep(j,4))
        ept(i,j,5)=1.0d0/dsqrt(2.0d0)*(ep(i,2)*ep(j,2)-ep(i,3)*ep(j,3))
        ept(i,j,6)=1.0d0/dsqrt(6.0d0)*(ep(i,2)*ep(j,2)+ep(i,3)*ep(j,3)-
     &            2.0d0*ep(i,4)*ep(j,4))
       enddo
      enddo

      return
      end
c********************************************************
      subroutine def_p_eq_p_dot_tp(n,k,e,j)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension e(4,4,2:6)

      do i=1,4
        pp(i,n)=pp(1,k)*e(1,i,j)-pp(2,k)*e(2,i,j)-pp(3,k)*e(3,i,j)
     .         -pp(4,k)*e(4,i,j)
      enddo

      return
      end
c*************************************************  
      double precision function sprpp(k1,k2)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)

      sprpp=pp(1,k1)*pp(1,k2)-pp(2,k1)*pp(2,k2)-pp(3,k1)*pp(3,k2)-
     .      pp(4,k1)*pp(4,k2)
      
	return
	end 
c*************************************************
      double precision function epsp(n1,n2,n3,n4)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/ pp(4,40),guv(4)

      epsp=pp(1,n1)*(pp(2,n2)*(pp(3,n3)*pp(4,n4)-pp(4,n3)*pp(3,n4))
     .     -pp(3,n2)*(pp(2,n3)*pp(4,n4)-pp(4,n3)*pp(2,n4))
     .     +pp(4,n2)*(pp(2,n3)*pp(3,n4)-pp(3,n3)*pp(2,n4)))  
     .     -pp(2,n1)*(pp(1,n2)*(pp(3,n3)*pp(4,n4)-pp(4,n3)*pp(3,n4))
     .     -pp(3,n2)*(pp(1,n3)*pp(4,n4)-pp(4,n3)*pp(1,n4))
     .     +pp(4,n2)*(pp(1,n3)*pp(3,n4)-pp(3,n3)*pp(1,n4)))  
     .     -pp(3,n1)*(pp(2,n2)*(pp(1,n3)*pp(4,n4)-pp(4,n3)*pp(1,n4))
     .     -pp(1,n2)*(pp(2,n3)*pp(4,n4)-pp(4,n3)*pp(2,n4))
     .     +pp(4,n2)*(pp(2,n3)*pp(1,n4)-pp(1,n3)*pp(2,n4)))  
     .     -pp(4,n1)*(pp(2,n2)*(pp(3,n3)*pp(1,n4)-pp(1,n3)*pp(3,n4))
     .     -pp(3,n2)*(pp(2,n3)*pp(1,n4)-pp(1,n3)*pp(2,n4))
     .     +pp(1,n2)*(pp(2,n3)*pp(3,n4)-pp(3,n3)*pp(2,n4)))

      return 
      end 
c********************************************************
      subroutine genppp()
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
#include "inclppp.h"
#include "inclcon.h"
      common/counter/ibcstate,nev

	if(ibcstate.eq.3) then
        p1p2=sprpp(2,1)
        p1p3=sprpp(3,1)
        p1p4=sprpp(4,1)
        p1p5=sprpp(5,1)
        p1p6=sprpp(6,1)
        p1p7=sprpp(7,1)
        p1p8=sprpp(8,1)
        p2p3=sprpp(3,2)
        p2p4=sprpp(4,2)
        p2p5=sprpp(5,2)
        p2p6=sprpp(6,2)
        p2p7=sprpp(7,2)
        p2p8=sprpp(8,2)
        p3p4=sprpp(4,3)
        p3p5=sprpp(5,3)
        p3p6=sprpp(6,3)
        p3p7=sprpp(7,3)
        p3p8=sprpp(8,3)
        p4p5=sprpp(5,4)
        p4p6=sprpp(6,4)
        p4p7=sprpp(7,4)
        p4p8=sprpp(8,4)
        p5p6=sprpp(6,5)
        p5p7=sprpp(7,5)
        p5p8=sprpp(8,5)
        p6p6=sprpp(6,6)
        p6p7=sprpp(7,6)
        p6p8=sprpp(8,6)
        p7p7=sprpp(7,7)
        p7p8=sprpp(8,7)
        p8p8=sprpp(8,8)
	end if

	if(ibcstate.eq.4) then
        p1p2=sprpp(2,1)
        p1p3=sprpp(3,1)
        p1p4=sprpp(4,1)
        p1p5=sprpp(5,1)
        p1p6=sprpp(6,1)
        p1p7=sprpp(7,1)
        p2p3=sprpp(3,2)
        p2p4=sprpp(4,2)
        p2p5=sprpp(5,2)
        p2p6=sprpp(6,2)
        p2p7=sprpp(7,2)
        p3p4=sprpp(4,3)
        p3p5=sprpp(5,3)
        p3p6=sprpp(6,3)
        p3p7=sprpp(7,3)
        p4p5=sprpp(5,4)
        p4p6=sprpp(6,4)
        p4p7=sprpp(7,4)
        p5p6=sprpp(6,5)
        p5p7=sprpp(7,5)
        p6p6=sprpp(6,6)
        p6p7=sprpp(7,6)
        p7p7=sprpp(7,7)
        p3p5s2=p3p5**2 
        p2p3s2=p2p3**2 
        p3p4s2=p3p4**2 
        p1p3s2=p1p3**2 
        p3p6s2=p3p6**2 
	end if

	if(ibcstate.eq.5) then
        p1p2=sprpp(2,1)
        p1p3=sprpp(3,1)
        p1p4=sprpp(4,1)
        p1p5=sprpp(5,1)
        p1p6=sprpp(6,1)
        p1p7=sprpp(7,1)
        p1p8=sprpp(8,1)
        p2p3=sprpp(3,2)
        p2p4=sprpp(4,2)
        p2p5=sprpp(5,2)
        p2p6=sprpp(6,2)
        p2p7=sprpp(7,2)
        p2p8=sprpp(8,2)
        p3p4=sprpp(4,3)
        p3p5=sprpp(5,3)
        p3p6=sprpp(6,3)
        p3p7=sprpp(7,3)
        p3p8=sprpp(8,3)
        p4p5=sprpp(5,4)
        p4p6=sprpp(6,4)
        p4p7=sprpp(7,4)
        p4p8=sprpp(8,4)
        p5p6=sprpp(6,5)
        p5p7=sprpp(7,5)
        p5p8=sprpp(8,5)
        p6p6=sprpp(6,6)
        p6p7=sprpp(7,6)
        p6p8=sprpp(8,6)
        p7p7=sprpp(7,7)
        p7p8=sprpp(8,7)
        p8p8=sprpp(8,8)
        p3p5s2=p3p5**2 
        p2p3s2=p2p3**2 
        p1p3s2=p1p3**2 
        p3p4s2=p3p4**2 
	end if

      if(ibcstate.eq.6) then
        p1p2=sprpp(2,1)
        p1p3=sprpp(3,1)
        p1p4=sprpp(4,1)
        p1p5=sprpp(5,1)
        p1p6=sprpp(6,1)
        p1p7=sprpp(7,1)
        p1p8=sprpp(8,1)
        p1p9=sprpp(9,1)
        p1p10=sprpp(10,1)
        p1p11=sprpp(11,1)
        p1p12=sprpp(12,1)
        p1p13=sprpp(13,1)
        p2p3=sprpp(3,2)
        p2p4=sprpp(4,2)
        p2p5=sprpp(5,2)
        p2p6=sprpp(6,2)
        p2p7=sprpp(7,2)
        p2p8=sprpp(8,2)
        p2p9=sprpp(9,2)
        p2p10=sprpp(10,2)
        p2p11=sprpp(11,2)
        p2p12=sprpp(12,2)
        p2p13=sprpp(13,2)
        p3p4=sprpp(4,3)
        p3p5=sprpp(5,3)
        p3p6=sprpp(6,3)
        p3p7=sprpp(7,3)
        p3p8=sprpp(8,3)
        p3p9=sprpp(9,3)
        p3p10=sprpp(10,3)
        p3p11=sprpp(11,3)
        p3p12=sprpp(12,3)
        p3p13=sprpp(13,3)
        p4p5=sprpp(5,4)
        p4p6=sprpp(6,4)
        p4p7=sprpp(7,4)
        p4p8=sprpp(8,4)
        p4p9=sprpp(9,4)
        p4p10=sprpp(10,4)
        p4p11=sprpp(11,4)
        p4p12=sprpp(12,4)
        p4p13=sprpp(13,4)
        p5p6=sprpp(6,5)
        p5p7=sprpp(7,5)
        p5p8=sprpp(8,5)
        p5p9=sprpp(9,5)
        p5p10=sprpp(10,5)
        p5p11=sprpp(11,5)
        p5p12=sprpp(12,5)
        p5p13=sprpp(13,5)
        p6p6=sprpp(6,6)
        p6p7=sprpp(7,6)
        p6p8=sprpp(8,6)
        p6p9=sprpp(9,6)
        p6p10=sprpp(10,6)
        p6p11=sprpp(11,6)
        p6p12=sprpp(12,6)
        p6p13=sprpp(13,6)
        p7p7=sprpp(7,7)
        p7p8=sprpp(8,7)
        p7p9=sprpp(9,7)
        p7p10=sprpp(10,7)
        p7p11=sprpp(11,7)
        p7p12=sprpp(12,7)
        p7p13=sprpp(13,7)
        p8p8=sprpp(8,8)
        p8p9=sprpp(9,8)
        p8p10=sprpp(10,8)
        p8p11=sprpp(11,8)
        p8p12=sprpp(12,8)
        p8p13=sprpp(13,8)
        p9p9=sprpp(9,9)
        p9p10=sprpp(10,9)
        p9p11=sprpp(11,9)
        p9p12=sprpp(12,9)
        p9p13=sprpp(13,9)
        p10p10=sprpp(10,10)
        p10p11=sprpp(11,10)
        p10p12=sprpp(12,10)
        p10p13=sprpp(13,10)
        p11p11=sprpp(11,11)
        p11p12=sprpp(12,11)
        p11p13=sprpp(13,11)
        p12p12=sprpp(12,12)
        p12p13=sprpp(13,12)
        p13p13=sprpp(13,13)
	end if
      
	w20=1.0d0/(2*p1p2+2*p1p3*ffmcfmb-2*p1p3+2*p2p3*ffmcfmb-2*p2p3+
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-fmb2+hbcm2)
      w19=1.0d0/(2*p1p2)
      w18=1.0d0/(2*p3p4*ffmcfmb+2*p3p5*ffmcfmb+2*p4p5+ffmcfmb**2*
     . hbcm2+fmc2)
      w17=1.0d0/(2*p1p2-2*p1p3*ffmcfmb-2*p2p3*ffmcfmb+ffmcfmb**2*
     . hbcm2-fmc2)
      w16=1.0d0/(2*p2p3*ffmcfmb-2*p2p3-2*p2p5-2*p3p5*ffmcfmb+2*p3p5+
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+fmb2+hbcm2)
      w15=1.0d0/(2*p1p3*ffmcfmb-2*p1p3-2*p1p5-2*p3p5*ffmcfmb+2*p3p5+
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+fmb2+hbcm2)
      w14=1.0d0/(2*p3p4-fmb2+fmc2+hbcm2)
      w13=1.0d0/(2*p1p3*ffmcfmb-2*p1p3+ffmcfmb**2*hbcm2-2*ffmcfmb*
     . hbcm2-fmb2+hbcm2)
      w12=(-1.0d0)/(2*p1p3*ffmcfmb-ffmcfmb**2*hbcm2+fmc2)
      w11=(-1.0d0)/(2*p2p3*ffmcfmb+2*p2p4-2*p3p4*ffmcfmb-ffmcfmb**2*
     . hbcm2-fmc2)
      w10=(-1.0d0)/(2*p2p4)
      w9=(-1.0d0)/(2*p1p5)
      w8=1.0d0/(2*p2p3*ffmcfmb-2*p2p3+ffmcfmb**2*hbcm2-2*ffmcfmb*
     . hbcm2-fmb2+hbcm2)
      w7=(-1.0d0)/(2*p2p3*ffmcfmb-ffmcfmb**2*hbcm2+fmc2)
      w6=1.0d0/(2*p3p5+fmb2-fmc2+hbcm2)
      w5=(-1.0d0)/(2*p1p3*ffmcfmb+2*p1p4-2*p3p4*ffmcfmb-ffmcfmb**2*
     . hbcm2-fmc2)
      w4=(-1.0d0)/(2*p1p4)
      w3=(-1.0d0)/(2*p2p5)
      w2=(-1.0d0)/(2*p3p5*ffmcfmb-2*p3p5-ffmcfmb**2*hbcm2+2*ffmcfmb*
     . hbcm2-fmb2-hbcm2)
      w1=1.0d0/(2*p3p4*ffmcfmb+ffmcfmb**2*hbcm2+fmc2)

      return 
      end
c******************************************************
      subroutine coupling()
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
#include "inclcon.h"
      common/cccc/c36,c35,c34,c33,c32,c31,c30,c29 
     . ,c28,c27,c26,c25,c24,c23,c22,c21,c20,c19,c18,c17 
     . ,c16,c15,c14,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3 
     . ,c2,c1s2,c1s1
      common/counter/ibcstate,nev
      
      	c0 = 0
	if(ibcstate.eq.3) c0=cbc1p1
      if(ibcstate.eq.4) c0=cbcp0
      if(ibcstate.eq.5) c0=cbcp1
	if(ibcstate.eq.6) c0=cbcp2

	c36=-c0
      c35=-c0
      c34=-c0
      c33=c0
      c32=c0
      c31=c0
      c30=c0
      c29=c0
      c28=c0
      c27=c0
      c26=c0
      c25=c0
      c24=c0
      c23=c0
      c22=c0
      c21=c0
      c20=c0
      c19=c0
      c18=c0
      c17=c0
      c16=c0
      c15=c0
      c14=c0
      c13=c0
      c12=c0
      c11=c0
      c10=c0
      c9=c0
      c8=c0
      c7=c0
      c6=c0
      c5=c0
      c4=c0
      c3=c0
      c2=c0
      c1s2=-c0
      c1s1=-c0

      return
      end
c************************ 
      subroutine genpppn(n) 
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
#include "inclppp.h"
#include "inclamp.h"
      common/ppp/pp(4,40),guv(4)
      common/counter/ibcstate,nev

	if(ibcstate.eq.3) then
      do i=1,4
         pp(i,12)=0.d0
         pp(i,11)=0.d0
         pp(i,10)=0.d0
         pp(i,9)=0.d0
      enddo
        call wa0(3,9,10,c(2,n))
        call wa0(6,9,10,c(5,n))
        call wa0(7,9,10,c(8,n))
        call wa0(8,9,10,c(9,n))
        call wa0(2,9,10,c(31,n))
        call wa1(3,8,11,12,c(1,n))
        call wa1(3,2,11,12,c(28,n))
        call wa1(6,3,11,12,c(3,n))
        call wa1(6,8,11,12,c(4,n))
        call wa1(6,7,11,12,c(13,n))
        call wa1(6,2,11,12,c(18,n))
        call wa1(7,3,11,12,c(6,n))
        call wa1(7,8,11,12,c(7,n))
        call wa1(2,7,11,12,c(15,n))
        call wa1(2,8,11,12,c(27,n))
        call wa2(3,8,7,9,10,c(26,n))
        call wa2(3,2,7,9,10,c(16,n))
        call wa2(6,3,7,9,10,c(11,n))
        call wa2(6,3,8,9,10,c(25,n))
        call wa2(6,3,2,9,10,c(20,n))
        call wa2(6,8,7,9,10,c(14,n))
        call wa2(6,8,2,9,10,c(23,n))
        call wa2(6,2,7,9,10,c(17,n))
        call wa2(8,2,7,9,10,c(24,n))
        call wa2(2,3,8,9,10,c(32,n))
        call wa3(3,8,2,7,11,12,c(29,n))
        call wa3(6,3,8,7,11,12,c(12,n))
        call wa3(6,3,2,7,11,12,c(19,n))
        call wa3(6,3,2,8,11,12,c(30,n))
        call wa3(6,8,2,7,11,12,c(22,n))
        call wa4(6,3,8,2,7,9,10,c(21,n))
        epspn0=epsp(10,4,5,9)
        epspn1=epsp3(10,11,4)
        epspn2=epsp3(10,11,5)
        epspn3=epsp3(12,4,9)
        epspn4=epsp3(12,5,9)
        epspn5=epsp(11,12,4,5)
        epspn6=epsp3(11,12,4)
        epspn7=epsp3(12,4,5)
        p12p12=sprpp(12,12)
        p11p12=sprpp(11,12)
        p11p11=sprpp(11,11)
        p10p12=sprpp(10,12)
        p10p11=sprpp(10,11)
        p10p10=sprpp(10,10)
        p9p12=sprpp(9,12)
        p9p11=sprpp(9,11)
        p9p10=sprpp(9,10)
        p9p9=sprpp(9,9)
        p5p12=sprpp(5,12)
        p5p11=sprpp(5,11)
        p5p10=sprpp(5,10)
        p5p9=sprpp(5,9)
        p4p12=sprpp(4,12)
        p4p11=sprpp(4,11)
        p4p10=sprpp(4,10)
        p4p9=sprpp(4,9)
        p0p12=pp(1,12)
        p0p11=pp(1,11)
        p0p10=pp(1,10)
        p0p9=pp(1,9)
        p0p5=pp(1,5)
        p0p4=pp(1,4)
	end if

	if(ibcstate.eq.5) then
        do i=1,4
         pp(i,12)=0.d0
         pp(i,11)=0.d0
         pp(i,10)=0.d0
         pp(i,9)=0.d0
        enddo
        call wa0(3,11,12,c(3,n))
        call wa0(6,11,12,c(7,n))
        call wa0(7,11,12,c(10,n))
        call wa0(8,11,12,c(11,n))
        call wa0(2,11,12,c(30,n))
        call wa1(3,8,9,10,c(2,n))
        call wa1(3,2,9,10,c(31,n))
        call wa1(6,3,9,10,c(5,n))
        call wa1(6,8,9,10,c(6,n))
        call wa1(6,7,9,10,c(15,n))
        call wa1(6,2,9,10,c(27,n))
        call wa1(7,3,9,10,c(8,n))
        call wa1(7,8,9,10,c(9,n))
        call wa1(2,7,9,10,c(28,n))
        call wa1(2,8,9,10,c(29,n))
        call wa2(3,8,7,11,12,c(1,n))
        call wa2(3,8,2,11,12,c(32,n))
        call wa2(3,2,7,11,12,c(17,n))
        call wa2(6,3,8,11,12,c(4,n))
        call wa2(6,3,7,11,12,c(13,n))
        call wa2(6,3,2,11,12,c(21,n))
        call wa2(6,8,7,11,12,c(16,n))
        call wa2(6,8,2,11,12,c(25,n))
        call wa2(6,2,7,11,12,c(19,n))
        call wa2(8,2,7,11,12,c(26,n))
        call wa3(3,8,2,7,9,10,c(18,n))
        call wa3(6,3,8,7,9,10,c(14,n))
        call wa3(6,3,8,2,9,10,c(23,n))
        call wa3(6,3,2,7,9,10,c(20,n))
        call wa3(6,8,2,7,9,10,c(24,n))
        call wa4(6,3,8,2,7,11,12,c(22,n))
        epspn0=epsp(10,4,5,9)
        epspn1=epsp3(10,4,9)
        epspn2=epsp3(10,11,4)
        epspn3=epsp3(10,11,5)
        epspn4=epsp3(10,4,5)
        epspn5=epsp3(12,4,9)
        epspn6=epsp3(12,5,9)
        epspn7=epsp(11,12,4,5)
        p12p12=sprpp(12,12)
        p11p12=sprpp(11,12)
        p11p11=sprpp(11,11)
        p10p12=sprpp(10,12)
        p10p11=sprpp(10,11)
        p10p10=sprpp(10,10)
        p9p12=sprpp(9,12)
        p9p11=sprpp(9,11)
        p9p10=sprpp(9,10)
        p9p9=sprpp(9,9)
        p5p12=sprpp(5,12)
        p5p11=sprpp(5,11)
        p5p10=sprpp(5,10)
        p5p9=sprpp(5,9)
        p4p12=sprpp(4,12)
        p4p11=sprpp(4,11)
        p4p10=sprpp(4,10)
        p4p9=sprpp(4,9)
        p0p12=pp(1,12)
        p0p11=pp(1,11)
        p0p10=pp(1,10)
        p0p9=pp(1,9)
        p0p5=pp(1,5)
        p0p4=pp(1,4)
  	end if

	if(ibcstate.eq.6) then
        do i=1,4
         pp(i,17)=0.d0
         pp(i,16)=0.d0
         pp(i,15)=0.d0
         pp(i,14)=0.d0
        enddo
        call wa0(3,14,15,c(33,n))
        call wa0(6,14,15,c(6,n))
        call wa0(7,14,15,c(10,n))
        call wa0(8,14,15,c(11,n))
        call wa0(9,14,15,c(12,n))
        call wa0(1,14,15,c(66,n))
        call wa0(10,14,15,c(29,n))
        call wa0(11,14,15,c(40,n))
        call wa0(12,14,15,c(62,n))
        call wa0(13,14,15,c(83,n))
        call wa1(3,8,16,17,c(1,n))
        call wa1(3,9,16,17,c(2,n))
        call wa1(3,11,16,17,c(41,n))
        call wa1(3,1,16,17,c(43,n))
        call wa1(3,12,16,17,c(48,n))
        call wa1(3,10,16,17,c(64,n))
        call wa1(3,13,16,17,c(84,n))
        call wa1(6,3,16,17,c(3,n))
        call wa1(6,8,16,17,c(4,n))
        call wa1(6,9,16,17,c(5,n))
        call wa1(6,1,16,17,c(38,n))
        call wa1(6,10,16,17,c(70,n))
        call wa1(6,11,16,17,c(51,n))
        call wa1(6,7,16,17,c(76,n))
        call wa1(6,12,16,17,c(92,n))
        call wa1(7,3,16,17,c(7,n))
        call wa1(7,8,16,17,c(8,n))
        call wa1(7,9,16,17,c(9,n))
        call wa1(7,1,16,17,c(77,n))
        call wa1(1,10,16,17,c(63,n))
        call wa1(1,8,16,17,c(65,n))
        call wa1(1,11,16,17,c(39,n))
        call wa1(1,9,16,17,c(42,n))
        call wa1(1,13,16,17,c(82,n))
        call wa1(10,7,16,17,c(28,n))
        call wa1(11,7,16,17,c(44,n))
        call wa1(12,1,16,17,c(93,n))
        call wa1(13,7,16,17,c(85,n))
        call wa2(3,8,1,14,15,c(74,n))
        call wa2(3,9,7,14,15,c(32,n))
        call wa2(3,11,1,14,15,c(73,n))
        call wa2(3,11,7,14,15,c(46,n))
        call wa2(3,10,1,14,15,c(81,n))
        call wa2(3,13,7,14,15,c(87,n))
        call wa2(3,6,12,14,15,c(94,n))
        call wa2(6,3,8,14,15,c(91,n))
        call wa2(6,3,9,14,15,c(80,n))
        call wa2(6,3,11,14,15,c(90,n))
        call wa2(6,3,10,14,15,c(88,n))
        call wa2(6,3,7,14,15,c(75,n))
        call wa2(6,8,7,14,15,c(15,n))
        call wa2(6,9,7,14,15,c(16,n))
        call wa2(6,1,10,14,15,c(20,n))
        call wa2(6,1,7,14,15,c(21,n))
        call wa2(6,1,8,14,15,c(26,n))
        call wa2(6,1,3,14,15,c(35,n))
        call wa2(6,1,9,14,15,c(37,n))
        call wa2(6,1,11,14,15,c(49,n))
        call wa2(6,1,12,14,15,c(61,n))
        call wa2(6,10,7,14,15,c(22,n))
        call wa2(6,11,7,14,15,c(59,n))
        call wa2(7,3,1,14,15,c(78,n))
        call wa2(8,3,7,14,15,c(31,n))
        call wa2(1,10,7,14,15,c(17,n))
        call wa2(1,8,7,14,15,c(23,n))
        call wa2(1,11,7,14,15,c(58,n))
        call wa2(1,9,7,14,15,c(60,n))
        call wa2(1,3,9,14,15,c(79,n))
        call wa2(10,3,7,14,15,c(27,n))
        call wa2(13,1,7,14,15,c(89,n))
        call wa3(3,8,1,7,16,17,c(68,n))
        call wa3(3,9,1,7,16,17,c(47,n))
        call wa3(3,11,1,7,16,17,c(45,n))
        call wa3(3,10,1,7,16,17,c(67,n))
        call wa3(3,13,1,7,16,17,c(86,n))
        call wa3(6,3,8,7,16,17,c(13,n))
        call wa3(6,3,9,7,16,17,c(14,n))
        call wa3(6,3,11,7,16,17,c(56,n))
        call wa3(6,3,10,7,16,17,c(72,n))
        call wa3(6,1,10,7,16,17,c(19,n))
        call wa3(6,1,8,7,16,17,c(25,n))
        call wa3(6,1,3,9,16,17,c(52,n))
        call wa3(6,1,3,11,16,17,c(50,n))
        call wa3(6,1,3,7,16,17,c(53,n))
        call wa3(6,1,3,12,16,17,c(57,n))
        call wa3(6,1,3,10,16,17,c(69,n))
        call wa3(6,1,3,8,16,17,c(71,n))
        call wa3(6,1,9,7,16,17,c(36,n))
        call wa3(6,1,11,7,16,17,c(54,n))
        call wa4(6,1,10,3,7,14,15,c(18,n))
        call wa4(6,1,8,3,7,14,15,c(24,n))
        call wa4(6,1,3,9,7,14,15,c(34,n))
        call wa4(6,1,3,11,7,14,15,c(55,n))
        epspn0=epsp(14,15,4,5)
        epspn1=epsp3(15,16,4)
        epspn2=epsp3(15,16,5)
        epspn3=epsp3(14,17,4)
        epspn4=epsp3(14,17,5)
        epspn5=epsp(16,17,4,5)
        epspn6=epsp3(16,17,4)
        epspn7=epsp3(17,4,5)
        p17p17=sprpp(17,17)
        p16p17=sprpp(16,17)
        p16p16=sprpp(16,16)
        p15p17=sprpp(15,17)
        p15p16=sprpp(15,16)
        p15p15=sprpp(15,15)
        p14p17=sprpp(14,17)
        p14p16=sprpp(14,16)
        p14p15=sprpp(14,15)
        p14p14=sprpp(14,14)
        p5p17=sprpp(5,17)
        p5p16=sprpp(5,16)
        p5p15=sprpp(5,15)
        p5p14=sprpp(5,14)
        p4p17=sprpp(4,17)
        p4p16=sprpp(4,16)
        p4p15=sprpp(4,15)
        p4p14=sprpp(4,14)
        p0p17=pp(1,17)
        p0p16=pp(1,16)
        p0p15=pp(1,15)
        p0p14=pp(1,14)
        p0p5=pp(1,5)
        p0p4=pp(1,4)
      end if

	return
      end
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c%    the subroutine is used to do calcualtion for and plus the results     % 
c%  wa1 -- c* p1_hat * p0_hat                                               %
c%  wa2 -- c* p2_hat * p1_hat * p0_hat                                      % 
c%  wa3 -- c* p3_hat * p2_hat * p1_hat * p0_hat                             % 
c%  wa4 -- c* p4_hat * p3_hat * p2_hat * p1_hat * p0_hat                    % 
c%  wa5 -- c* p5_hat * p4_hat * p3_hat * p2_hat * p1_hat * p0_hat           % 
c%  wa6 -- c* p6_hat * p5_hat * p4_hat * p3_hat * p2_hat * p1_hat * p0_hat  % 
c%         ........                                                         %
c%  wan -- c* pn_hat * .......................................... * p0_hat  %
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      subroutine wa0(n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)

      pp(1,na)=pp(1,na)+c*pp(1,n0)
      pp(2,na)=pp(2,na)+c*pp(2,n0)
      pp(3,na)=pp(3,na)+c*pp(3,n0)
      pp(4,na)=pp(4,na)+c*pp(4,n0)

      return
      end
c*********************
      subroutine wa1(n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)

      call pabp(pp(1,n1),pp(1,n0),pp(1,na),pp(1,nb),c)

      return
      end
c********************
      subroutine wa2(n2,n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension pa(4),pb(4)

      call    pab(pp(1,n1),pp(1,n0),pa,pb)
      call paabbp(pp(1,n2),pa,pb,pp(1,na),pp(1,nb),c)

      return
      end
c************************
      subroutine wa3(n3,n2,n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension pa(4),pb(4),qa(4),qb(4)

      call    pab(pp(1,n1),pp(1,n0),pa,pb) 
      call  paabb(pp(1,n2),pa,pb,qa,qb)
      call paabbp(pp(1,n3),qa,qb,pp(1,na),pp(1,nb),c)

      return
      end
c*********************
      subroutine wa4(n4,n3,n2,n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension pa(4),pb(4),qa(4),qb(4)

      call    pab(pp(1,n1),pp(1,n0),pa,pb) 
      call  paabb(pp(1,n2),pa,pb,qa,qb)
      call  paabb(pp(1,n3),qa,qb,pa,pb) 
      call paabbp(pp(1,n4),pa,pb,pp(1,na),pp(1,nb),c)

      return
      end
c********************
      subroutine wa5(n5,n4,n3,n2,n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension pa(4),pb(4),qa(4),qb(4)

      call    pab(pp(1,n1),pp(1,n0),pa,pb) 
      call  paabb(pp(1,n2),pa,pb,qa,qb)
      call  paabb(pp(1,n3),qa,qb,pa,pb) 
      call  paabb(pp(1,n4),pa,pb,qa,qb) 
      call paabbp(pp(1,n5),qa,qb,pp(1,na),pp(1,nb),c)

      return
      end
c***************
      subroutine wa6(n6,n5,n4,n3,n2,n1,n0,na,nb,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)
      dimension pa(4),pb(4),qa(4),qb(4)

      call    pab(pp(1,n1),pp(1,n0),pa,pb) 
      call  paabb(pp(1,n2),pa,pb,qa,qb)
      call  paabb(pp(1,n3),qa,qb,pa,pb) 
      call  paabb(pp(1,n4),pa,pb,qa,qb) 
      call  paabb(pp(1,n5),qa,qb,pa,pb) 
      call paabbp(pp(1,n6),pa,pb,pp(1,na),pp(1,nb),c)

      return
      end
c*****************
      double precision function epsp3(n1,n2,n3) 
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/ppp/pp(4,40),guv(4)

      epsp3=pp(2,n1)*(pp(3,n2)*pp(4,n3)-pp(4,n2)*pp(3,n3))
     .     +pp(3,n1)*(pp(4,n2)*pp(2,n3)-pp(2,n2)*pp(4,n3))
     .     +pp(4,n1)*(pp(2,n2)*pp(3,n3)-pp(3,n2)*pp(2,n3))

      return 
      end 
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c%      this is written by j. x. wnag; june 20, 2004           %
c%      there are   16 times,  9 plus   for subroutinue pab    %
c%      there are   32 times, 24 plus   for subroutinue paabb  %
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      subroutine pab(p1,q,p1a,p1b)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p1(4),q(4),p1a(4),p1b(4)
c*******************************************************************   
c***     p1a(1)= p1.q 
c***     p1a_v=q(1)*p1_v - p1(1)*q_v 
c*******************************************************************
      p1a(1)=p1(1)*q(1)-p1(2)*q(2)-p1(3)*q(3)-p1(4)*q(4)
      p1a(2)=p1(2)*q(1)-p1(1)*q(2)
      p1a(3)=p1(3)*q(1)-p1(1)*q(3)
      p1a(4)=p1(4)*q(1)-p1(1)*q(4)
c*******************************************************************   
c***     p1b(1)=0
c***     p1b_v= p1_v x q_v 
c*******************************************************************
      p1b(1)=0.0d0
      p1b(2)=p1(3)*q(4)-p1(4)*q(3)
      p1b(3)=p1(4)*q(2)-p1(2)*q(4)
      p1b(4)=p1(2)*q(3)-p1(3)*q(2)

      return
      end 
c******************************
      subroutine paabb(p2,pa,pb,p2a,p2b)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p2(4),pa(4),pb(4),p2a(4),p2b(4)  
c*******************************************************************   
c***     p2a(1)= p2.pa 
c***     p2a_v=pa(1)*p2_v - p2(1)*pa_v + p2_v x pb_v
c*******************************************************************
      p2a(1)=p2(1)*pa(1)-p2(2)*pa(2)-p2(3)*pa(3)-p2(4)*pa(4)
      p2a(2)=p2(2)*pa(1)-p2(1)*pa(2)+p2(3)*pb(4)-p2(4)*pb(3)
      p2a(3)=p2(3)*pa(1)-p2(1)*pa(3)+p2(4)*pb(2)-p2(2)*pb(4)
      p2a(4)=p2(4)*pa(1)-p2(1)*pa(4)+p2(2)*pb(3)-p2(3)*pb(2)
c*******************************************************************   
c***     p2b(1)=-p2.pb 
c***     p2b_v= p2(1)*pb_v -pb(1)*p2_v + p2_v x pa_v 
c*******************************************************************
      p2b(1)=p2(2)*pb(2)+p2(3)*pb(3)+p2(4)*pb(4)-p2(1)*pb(1)   
      p2b(2)=p2(1)*pb(2)-p2(2)*pb(1)+p2(3)*pa(4)-p2(4)*pa(3)
      p2b(3)=p2(1)*pb(3)-p2(3)*pb(1)+p2(4)*pa(2)-p2(2)*pa(4)
      p2b(4)=p2(1)*pb(4)-p2(4)*pb(1)+p2(2)*pa(3)-p2(3)*pa(2)

      return
      end 
c******************************
      subroutine pabp(p1,q,p1a,p1b,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p1(4),q(4),p1a(4),p1b(4)
c*******************************************************************   
c***     p1a(1)= p1.q 
c***     p1a_v=q(1)*p1_v - p1(1)*q_v 
c*******************************************************************
      p1a(1)=p1a(1)+c*(p1(1)*q(1)-p1(2)*q(2)-p1(3)*q(3)-p1(4)*q(4))
      p1a(2)=p1a(2)+c*(p1(2)*q(1)-p1(1)*q(2))
      p1a(3)=p1a(3)+c*(p1(3)*q(1)-p1(1)*q(3))
      p1a(4)=p1a(4)+c*(p1(4)*q(1)-p1(1)*q(4))
c*******************************************************************   
c***     p1b(1)=0
c***     p1b_v= p1_v x q_v 
c*******************************************************************
c      p1b(1)=0.0d0+pb1(1)
      p1b(2)=p1b(2)+c*(p1(3)*q(4)-p1(4)*q(3))
      p1b(3)=p1b(3)+c*(p1(4)*q(2)-p1(2)*q(4))
      p1b(4)=p1b(4)+c*(p1(2)*q(3)-p1(3)*q(2))

      return
      end 
c**********************
      subroutine paabbp(p2,pa,pb,p2a,p2b,c)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      dimension p2(4),pa(4),pb(4),p2a(4),p2b(4)  
c*******************************************************************   
c***     p2a(1)=p2a(1)+c*(p2.pa) 
c***     p2a_v=p2a_v+c*(pa(1)*p2_v - p2(1)*pa_v + p2_v x pb_v)
c*******************************************************************
      p2a(1)=p2a(1)+c*(p2(1)*pa(1)-p2(2)*pa(2)-p2(3)*pa(3)-p2(4)*pa(4))
      p2a(2)=p2a(2)+c*(p2(2)*pa(1)-p2(1)*pa(2)+p2(3)*pb(4)-p2(4)*pb(3))
      p2a(3)=p2a(3)+c*(p2(3)*pa(1)-p2(1)*pa(3)+p2(4)*pb(2)-p2(2)*pb(4))
      p2a(4)=p2a(4)+c*(p2(4)*pa(1)-p2(1)*pa(4)+p2(2)*pb(3)-p2(3)*pb(2))
c*******************************************************************   
c***     p2b(1)=p2b(1)-c*p2.pb 
c***     p2b_v= p2b_v+c*(p2(1)*pb_v -pb(1)*p2_v + p2_v x pa_v) 
c*******************************************************************
      p2b(1)=p2b(1)+c*(p2(2)*pb(2)+p2(3)*pb(3)+p2(4)*pb(4)-p2(1)*pb(1))
      p2b(2)=p2b(2)+c*(p2(1)*pb(2)-p2(2)*pb(1)+p2(3)*pa(4)-p2(4)*pa(3))
      p2b(3)=p2b(3)+c*(p2(1)*pb(3)-p2(3)*pb(1)+p2(4)*pa(2)-p2(2)*pa(4))
      p2b(4)=p2b(4)+c*(p2(1)*pb(4)-p2(4)*pb(1)+p2(2)*pa(3)-p2(3)*pa(2))

      return
      end
      
