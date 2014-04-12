c...this is the 36 amplitude for (3p2) state.
c...obtained from the program fdc.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) z.x. zhang, j.x. wang and x.g. wu                    c
c reference: hep-ph/0309120                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine amp10_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(91) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,91 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(8)=8*ccc*w1*p5p6*(-fmb*hbcm+fmc*hbcm)
      b(10)=w1*ccc*(16*p5p6*(-hbcm*p3p8+hbcm*p4p8)+16*hbcm*p1p8*p3p6)
      b(15)=8*ccc*hbcm*p1p3*w1
      b(21)=8*ccc*w1*(-hbcm*p3p8+hbcm*p4p8)
      b(23)=-8*ccc*hbcm*p3p6*w1
      b(24)=4*ccc*hbcm*w1
      b(25)=4*ccc*w1*(fmb*hbcm-fmc*hbcm)
      b(30)=w1*ccc*(16*p5p6*(fmb*hbcm*p7p8-fmc*hbcm*p7p8)+16*p6p7*(
     . fmb*hbcm*p1p8-fmc*hbcm*p1p8))
      b(31)=8*ccc*hbcm*p5p6*w1
      b(65)=8*ccc*w1*p6p7*(-fmb*hbcm+fmc*hbcm)
      b(66)=16*ccc*w1*(hbcm*p3p7*p6p8+hbcm*p4p8*p6p7)
      b(74)=-8*ccc*hbcm*p6p7*w1
      b(75)=-8*ccc*hbcm*p1p8*w1
      b(76)=8*ccc*w1*(-fmb*hbcm*p1p8+fmc*hbcm*p1p8)
      b(77)=8*ccc*w1*(-fmb*hbcm*p6p8+fmc*hbcm*p6p8)
      b(78)=-8*ccc*hbcm*p6p8*w1
      do 200 n=1,91 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp18_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      b(1)=ccc*(w18*((p5p6*p3p7)*(-16*ffmcfmb+16)+(p4p6*p3p7)*(-16*
     . ffmcfmb-32)+(p5p7*p3p6)*(-32*ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*
     . hbcm2+2*ffmcfmb*hbcm2-2*fmb2+3*fmc2)+16*(-ffmcfmb*p2p3*p6p7-2*
     . ffmcfmb*p3p6*p3p7-2*p2p4*p6p7+p2p5*p6p7+2*p3p4*p6p7-p3p5*p6p7-
     . 2*p3p6*p4p7-3*p4p6*p4p7-2*p4p6*p5p7+p5p6*p5p7))+w1*((p5p6*p3p7
     . )*(-16*ffmcfmb+16)+(p4p6*p3p7)*(-16*ffmcfmb-32)+(p5p7*p3p6)*(-
     . 32*ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*hbcm2+2*ffmcfmb*hbcm2-2*
     . fmb2+3*fmc2)+16*(-ffmcfmb*p2p3*p6p7-2*ffmcfmb*p3p6*p3p7-2*p2p4
     . *p6p7+p2p5*p6p7+2*p3p4*p6p7-p3p5*p6p7-2*p3p6*p4p7-3*p4p6*p4p7-
     . 2*p4p6*p5p7+p5p6*p5p7)))
      b(2)=w18*ccc*((p5p6*p3p7)*(-16*ffmcfmb+16)+(p4p6*p3p7)*(-16*
     . ffmcfmb-32)+(p5p7*p3p6)*(-32*ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*
     . hbcm2+2*ffmcfmb*hbcm2-2*fmb2+3*fmc2)+16*(-ffmcfmb*p2p3*p6p7-2*
     . ffmcfmb*p3p6*p3p7-2*p2p4*p6p7+p2p5*p6p7+2*p3p4*p6p7-p3p5*p6p7-
     . 2*p3p6*p4p7-3*p4p6*p4p7-2*p4p6*p5p7+p5p6*p5p7))
      ans=ccc*(w18*(16*p3p4*(p7p8+p7p9)+16*p5p7*(p2p8+p2p9-p5p8-p5p9)
     . +16*p4p7*(-2*p2p8-2*p2p9+3*p4p8+3*p4p9+2*p5p8+2*p5p9)+16*p2p4*
     . (-p7p8-p7p9)+16*p3p7*(-ffmcfmb*p2p8-ffmcfmb*p2p9+ffmcfmb*p4p8+
     . ffmcfmb*p4p9+ffmcfmb*p5p8+ffmcfmb*p5p9+p4p8+p4p9)+16*p2p3*(-
     . ffmcfmb*p7p8-ffmcfmb*p7p9)+8*(-ffmcfmb**2*hbcm2*p7p8-ffmcfmb**
     . 2*hbcm2*p7p9+2*ffmcfmb*hbcm2*p7p8+2*ffmcfmb*hbcm2*p7p9+fmc2*
     . p7p8+fmc2*p7p9))+w1*(16*p5p7*(p2p8-p5p8)+16*p4p7*(-2*p2p8+3*
     . p4p8+2*p5p8)+16*p3p7*(-ffmcfmb*p2p8+ffmcfmb*p4p8+ffmcfmb*p5p8+
     . p4p8)+8*(-ffmcfmb**2*hbcm2*p7p8+2*ffmcfmb*hbcm2*p7p8-2*ffmcfmb
     . *p2p3*p7p8+fmc2*p7p8-2*p2p4*p7p8+2*p3p4*p7p8))+w8*(16*p5p7*(
     . p2p11-p5p11)+16*p4p7*(-2*p2p11+3*p4p11+2*p5p11)+16*p3p7*(-
     . ffmcfmb*p2p11+ffmcfmb*p4p11+ffmcfmb*p5p11+p4p11)+8*(-ffmcfmb**
     . 2*hbcm2*p7p11+2*ffmcfmb*hbcm2*p7p11-2*ffmcfmb*p2p3*p7p11+fmc2*
     . p7p11-2*p2p4*p7p11+2*p3p4*p7p11))+8*(-3*p1p12-p2p12+3*p3p12+4*
     . p4p12+p5p12-2*p7p8+p7p9))
      b(3)=ans
      b(4)=ccc*(w18*((p3p7*p3p5)*(-16*ffmcfmb-16)+(p3p7*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm+2*hbcm2)
     . +8*p4p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-4*hbcm2)+8*p3p7*(
     . ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-fmc2)+16*
     . (2*ffmcfmb*p2p3*p3p7+2*p2p3*p4p7-p2p3*p5p7+p2p4*p3p7+3*p3p4*
     . p5p7-3*p3p5*p4p7))+w1*((p3p7*p3p5)*(-16*ffmcfmb-16)+(p3p7*p3p4
     . )*(-16*ffmcfmb+16)+8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm+2
     . *hbcm2)+8*p4p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-4*hbcm2)+8*
     . p3p7*(ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-
     . fmc2)+16*(2*ffmcfmb*p2p3*p3p7+2*p2p3*p4p7-p2p3*p5p7+p2p4*p3p7+
     . 3*p3p4*p5p7-3*p3p5*p4p7)))
      b(5)=w18*ccc*((p3p7*p3p5)*(-16*ffmcfmb-16)+(p3p7*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm+2*hbcm2)
     . +8*p4p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-4*hbcm2)+8*p3p7*(
     . ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-fmc2)+16*
     . (2*ffmcfmb*p2p3*p3p7+2*p2p3*p4p7-p2p3*p5p7+p2p4*p3p7+3*p3p4*
     . p5p7-3*p3p5*p4p7))
      ans2=w8*(16*p5p7*(-hbcm*p2p11+2*hbcm*p4p11)+16*p4p7*(hbcm*p2p11
     . -2*hbcm*p5p11)+16*p3p7*(ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p4p11-
     . ffmcfmb*hbcm*p5p11-fmc*p2p11+fmc*p4p11+fmc*p5p11+hbcm*p4p11-
     . hbcm*p5p11)+16*p1p3*(ffmcfmb*hbcm*p7p11-fmc*p7p11)+8*(ffmcfmb
     . **2*hbcm3*p7p11+2*ffmcfmb*hbcm*p2p3*p7p11-2*ffmcfmb*hbcm3*
     . p7p11-fmc2*hbcm*p7p11+2*hbcm*p2p4*p7p11-2*hbcm*p3p4*p7p11))+8*
     . (fmb*p3p12+fmc*p3p12+2*hbcm*p1p12-hbcm*p2p12-3*hbcm*p3p12-hbcm
     . *p4p12-2*hbcm*p5p12+2*hbcm*p7p11-hbcm*p7p8)
      ans1=w18*(16*p3p4*(-hbcm*p7p8-hbcm*p7p9)+16*p5p7*(-hbcm*p2p8-
     . hbcm*p2p9+2*hbcm*p4p8+2*hbcm*p4p9)+16*p4p7*(hbcm*p2p8+hbcm*
     . p2p9-2*hbcm*p5p8-2*hbcm*p5p9)+16*p2p4*(hbcm*p7p8+hbcm*p7p9)+16
     . *p3p7*(ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p2p9-ffmcfmb*hbcm*p4p8-
     . ffmcfmb*hbcm*p4p9-ffmcfmb*hbcm*p5p8-ffmcfmb*hbcm*p5p9-fmc*p2p8
     . -fmc*p2p9+fmc*p4p8+fmc*p4p9+fmc*p5p8+fmc*p5p9+hbcm*p4p8+hbcm*
     . p4p9-hbcm*p5p8-hbcm*p5p9)+16*p2p3*(ffmcfmb*hbcm*p7p8+ffmcfmb*
     . hbcm*p7p9)+16*p1p3*(ffmcfmb*hbcm*p7p8+ffmcfmb*hbcm*p7p9-fmc*
     . p7p8-fmc*p7p9)+8*(ffmcfmb**2*hbcm3*p7p8+ffmcfmb**2*hbcm3*p7p9-
     . 2*ffmcfmb*hbcm3*p7p8-2*ffmcfmb*hbcm3*p7p9-fmc2*hbcm*p7p8-fmc2*
     . hbcm*p7p9))+w1*(16*p5p7*(-hbcm*p2p8+2*hbcm*p4p8)+16*p4p7*(hbcm
     . *p2p8-2*hbcm*p5p8)+16*p3p7*(ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*
     . p4p8-ffmcfmb*hbcm*p5p8-fmc*p2p8+fmc*p4p8+fmc*p5p8+hbcm*p4p8-
     . hbcm*p5p8)+16*p1p3*(ffmcfmb*hbcm*p7p8-fmc*p7p8)+8*(ffmcfmb**2*
     . hbcm3*p7p8+2*ffmcfmb*hbcm*p2p3*p7p8-2*ffmcfmb*hbcm3*p7p8-fmc2*
     . hbcm*p7p8+2*hbcm*p2p4*p7p8-2*hbcm*p3p4*p7p8))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w8*(16*p5p6*(-p3p11-p4p11)+16*p4p6*(-p2p11+3*p3p11+p5p11)+
     . 16*p3p6*(-ffmcfmb*p2p11+2*ffmcfmb*p3p11+2*ffmcfmb*p4p11-2*
     . p4p11+p5p11)+8*(-ffmcfmb**2*hbcm2*p6p11+2*ffmcfmb*hbcm2*p6p11-
     . 2*ffmcfmb*p2p3*p6p11-2*fmb2*p6p11+3*fmc2*p6p11-4*p2p4*p6p11+2*
     . p2p5*p6p11+4*p3p4*p6p11-2*p3p5*p6p11))+8*(p1p13+4*p2p13-4*
     . p3p13-5*p4p13-p6p11+2*p6p8-p6p9)
      ans1=w18*(16*p5p6*(-p3p8-p3p9-p4p8-p4p9)+16*p3p5*(-p6p8-p6p9)+
     . 32*p3p4*(p6p8+p6p9)+16*p4p6*(-p2p8-p2p9+3*p3p8+3*p3p9+p5p8+
     . p5p9)+16*p2p5*(p6p8+p6p9)+32*p2p4*(-p6p8-p6p9)+16*p3p6*(-
     . ffmcfmb*p2p8-ffmcfmb*p2p9+2*ffmcfmb*p3p8+2*ffmcfmb*p3p9+2*
     . ffmcfmb*p4p8+2*ffmcfmb*p4p9-2*p4p8-2*p4p9+p5p8+p5p9)+16*p2p3*(
     . -ffmcfmb*p6p8-ffmcfmb*p6p9)+8*(-ffmcfmb**2*hbcm2*p6p8-ffmcfmb
     . **2*hbcm2*p6p9+2*ffmcfmb*hbcm2*p6p8+2*ffmcfmb*hbcm2*p6p9-2*
     . fmb2*p6p8-2*fmb2*p6p9+3*fmc2*p6p8+3*fmc2*p6p9))+w1*(16*p5p6*(-
     . p3p8-p4p8)+16*p4p6*(-p2p8+3*p3p8+p5p8)+16*p3p6*(-ffmcfmb*p2p8+
     . 2*ffmcfmb*p3p8+2*ffmcfmb*p4p8-2*p4p8+p5p8)+8*(-ffmcfmb**2*
     . hbcm2*p6p8+2*ffmcfmb*hbcm2*p6p8-2*ffmcfmb*p2p3*p6p8-2*fmb2*
     . p6p8+3*fmc2*p6p8-4*p2p4*p6p8+2*p2p5*p6p8+4*p3p4*p6p8-2*p3p5*
     . p6p8))+ans2
      ans=ccc*ans1
      b(7)=ans
      b(8)=ccc*(w18*(8*p5p6*(ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb*
     . hbcm2+fmc*hbcm-2*hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmb2+2*fmc*
     . hbcm-3*fmc2)+16*(2*ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6+p2p3*
     . p4p6+2*p2p4*p3p6-p2p5*p3p6+p3p4*p5p6-p3p5*p4p6))+w1*(8*p5p6*(
     . ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb*hbcm2+fmc*hbcm-2*hbcm2
     . )+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-3
     . *ffmcfmb*hbcm2-fmb*hbcm+2*fmb2+2*fmc*hbcm-3*fmc2)+16*(2*
     . ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6+p2p3*p4p6+2*p2p4*p3p6-
     . p2p5*p3p6+p3p4*p5p6-p3p5*p4p6)))
      b(9)=w18*ccc*(8*p5p6*(ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb*
     . hbcm2+fmc*hbcm-2*hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmb2+2*fmc*
     . hbcm-3*fmc2)+16*(2*ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6+p2p3*
     . p4p6+2*p2p4*p3p6-p2p5*p3p6+p3p4*p5p6-p3p5*p4p6))
      ans4=16*p3p6*(ffmcfmb**2*hbcm*p3p8-ffmcfmb*fmb*p3p8-2*ffmcfmb*
     . fmc*p3p8-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p3p8
     . +ffmcfmb*hbcm*p3p9+ffmcfmb*hbcm*p5p8+ffmcfmb*hbcm*p5p9+fmb*
     . p1p8+fmb*p1p9-fmb*p3p9+3*fmc*p1p8+3*fmc*p1p9+fmc*p2p8+fmc*p2p9
     . -3*fmc*p3p9-fmc*p4p8-fmc*p4p9-fmc*p5p8-fmc*p5p9+hbcm*p4p8+hbcm
     . *p4p9-hbcm*p5p8-hbcm*p5p9)+16*(ffmcfmb*hbcm3*p6p8+ffmcfmb*
     . hbcm3*p6p9-fmb*hbcm2*p6p8-fmb*hbcm2*p6p9+fmb2*hbcm*p6p8+fmb2*
     . hbcm*p6p9-2*fmc*hbcm2*p6p8-2*fmc*hbcm2*p6p9-fmc2*hbcm*p6p8-
     . fmc2*hbcm*p6p9)
      ans3=16*p2p5*(-hbcm*p6p8-hbcm*p6p9)+16*p2p4*(hbcm*p6p8+hbcm*
     . p6p9)+16*p5p6*(-ffmcfmb*hbcm*p3p8+fmc*p3p8-hbcm*p1p8-hbcm*p1p9
     . +hbcm*p3p8+hbcm*p3p9+hbcm*p4p8+hbcm*p4p9)+16*p4p6*(-ffmcfmb*
     . hbcm*p3p8+fmc*p3p8+2*hbcm*p1p8+2*hbcm*p1p9+hbcm*p2p8+hbcm*p2p9
     . -2*hbcm*p3p8-2*hbcm*p3p9-hbcm*p5p8-hbcm*p5p9)+16*p3p5*(ffmcfmb
     . *hbcm*p6p8+ffmcfmb*hbcm*p6p9-fmb*p6p8-fmb*p6p9-2*fmc*p6p8-2*
     . fmc*p6p9+hbcm*p6p8+hbcm*p6p9)+16*p3p4*(ffmcfmb*hbcm*p6p8+
     . ffmcfmb*hbcm*p6p9-fmb*p6p8-fmb*p6p9-2*fmc*p6p8-2*fmc*p6p9-hbcm
     . *p6p8-hbcm*p6p9)+16*p2p3*(-ffmcfmb*hbcm*p6p8-ffmcfmb*hbcm*p6p9
     . +fmb*p6p8+fmb*p6p9+2*fmc*p6p8+2*fmc*p6p9)+16*p1p3*(-2*ffmcfmb*
     . hbcm*p6p8-2*ffmcfmb*hbcm*p6p9+fmb*p6p8+fmb*p6p9+3*fmc*p6p8+3*
     . fmc*p6p9)+ans4
      ans2=w18*ans3
      ans5=w1*(16*p5p6*(-ffmcfmb*hbcm*p3p8+fmc*p3p8-hbcm*p1p8+hbcm*
     . p3p8+hbcm*p4p8)+16*p4p6*(-ffmcfmb*hbcm*p3p8+fmc*p3p8+2*hbcm*
     . p1p8+hbcm*p2p8-2*hbcm*p3p8-hbcm*p5p8)+16*p3p5*(ffmcfmb*hbcm*
     . p6p8-fmb*p6p8-2*fmc*p6p8+hbcm*p6p8)+16*p3p4*(ffmcfmb*hbcm*p6p8
     . -fmb*p6p8-2*fmc*p6p8-hbcm*p6p8)+16*p2p3*(-ffmcfmb*hbcm*p6p8+
     . fmb*p6p8+2*fmc*p6p8)+16*p1p3*(-2*ffmcfmb*hbcm*p6p8+fmb*p6p8+3*
     . fmc*p6p8)+16*p3p6*(ffmcfmb**2*hbcm*p3p8-ffmcfmb*fmb*p3p8-2*
     . ffmcfmb*fmc*p3p8-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p3p8+ffmcfmb*
     . hbcm*p5p8+fmb*p1p8+3*fmc*p1p8+fmc*p2p8-fmc*p4p8-fmc*p5p8+hbcm*
     . p4p8-hbcm*p5p8)+16*(ffmcfmb*hbcm3*p6p8-fmb*hbcm2*p6p8+fmb2*
     . hbcm*p6p8-2*fmc*hbcm2*p6p8-fmc2*hbcm*p6p8+hbcm*p2p4*p6p8-hbcm*
     . p2p5*p6p8))+8*(4*fmb*p3p13+4*fmc*p3p13-hbcm*p1p13-3*hbcm*p2p13
     . +3*hbcm*p4p13+hbcm*p6p11-hbcm*p6p8+hbcm*p6p9)
      ans1=w8*(16*p5p6*(-hbcm*p1p11+hbcm*p3p11+hbcm*p4p11)+16*p4p6*(2
     . *hbcm*p1p11+hbcm*p2p11-2*hbcm*p3p11-hbcm*p5p11)+16*p3p5*(
     . ffmcfmb*hbcm*p6p11-fmb*p6p11-2*fmc*p6p11+hbcm*p6p11)+16*p3p4*(
     . ffmcfmb*hbcm*p6p11-fmb*p6p11-2*fmc*p6p11-hbcm*p6p11)+16*p2p3*(
     . -ffmcfmb*hbcm*p6p11+fmb*p6p11+2*fmc*p6p11)+16*p1p3*(-2*ffmcfmb
     . *hbcm*p6p11+fmb*p6p11+3*fmc*p6p11)+16*p3p6*(-ffmcfmb*hbcm*
     . p1p11+ffmcfmb*hbcm*p3p11+ffmcfmb*hbcm*p5p11+fmb*p1p11-fmb*
     . p3p11+3*fmc*p1p11+fmc*p2p11-3*fmc*p3p11-fmc*p4p11-fmc*p5p11+
     . hbcm*p4p11-hbcm*p5p11)+16*(ffmcfmb*hbcm3*p6p11-fmb*hbcm2*p6p11
     . +fmb2*hbcm*p6p11-2*fmc*hbcm2*p6p11-fmc2*hbcm*p6p11+hbcm*p2p4*
     . p6p11-hbcm*p2p5*p6p11))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      ans2=w1*(16*(p5p7*p3p6)*(-fmb-2*fmc+hbcm)+16*p6p7*(-ffmcfmb*
     . hbcm3+fmb*hbcm2-fmb2*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(p5p6*p3p7
     . )*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm-hbcm)+16*
     . (p4p7*p3p6)*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+16*(p6p7*p3p5)*(-
     . ffmcfmb*hbcm+fmb+2*fmc-hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm+fmb
     . +2*fmc+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-fmb-2*fmc)+16*(p6p7*
     . p1p3)*(2*ffmcfmb*hbcm-fmb-3*fmc)+16*(p3p7*p3p6)*(-ffmcfmb*fmc-
     . fmb-fmc)+16*(-hbcm*p2p4*p6p7+hbcm*p2p5*p6p7-2*hbcm*p4p6*p4p7-
     . hbcm*p4p6*p5p7+hbcm*p5p6*p5p7))
      ans1=w18*(16*(p5p7*p3p6)*(-fmb-2*fmc+hbcm)+16*p6p7*(-ffmcfmb*
     . hbcm3+fmb*hbcm2-fmb2*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(p5p6*p3p7
     . )*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm-hbcm)+16*
     . (p4p7*p3p6)*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+16*(p6p7*p3p5)*(-
     . ffmcfmb*hbcm+fmb+2*fmc-hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm+fmb
     . +2*fmc+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-fmb-2*fmc)+16*(p6p7*
     . p1p3)*(2*ffmcfmb*hbcm-fmb-3*fmc)+16*(p3p7*p3p6)*(-ffmcfmb*fmc-
     . fmb-fmc)+16*(-hbcm*p2p4*p6p7+hbcm*p2p5*p6p7-2*hbcm*p4p6*p4p7-
     . hbcm*p4p6*p5p7+hbcm*p5p6*p5p7))+ans2
      ans=ccc*ans1
      b(11)=ans
      b(12)=w18*ccc*(16*(p5p7*p3p6)*(-fmb-2*fmc+hbcm)+16*p6p7*(-
     . ffmcfmb*hbcm3+fmb*hbcm2-fmb2*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(
     . p5p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm-
     . hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+16*(p6p7*
     . p3p5)*(-ffmcfmb*hbcm+fmb+2*fmc-hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*
     . hbcm+fmb+2*fmc+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-fmb-2*fmc)+
     . 16*(p3p7*p3p6)*(-ffmcfmb*fmc-fmb-fmc)+16*(-hbcm*p2p4*p6p7+hbcm
     . *p2p5*p6p7-2*hbcm*p4p6*p4p7-hbcm*p4p6*p5p7+hbcm*p5p6*p5p7))
      b(13)=ccc*(8*w18*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3+fmb2-2*fmc2+3*p2p4-p2p5-3*p3p4+p3p5)+8*w1*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3+fmb2-2*fmc2+3*p2p4-p2p5-3
     . *p3p4+p3p5))
      b(14)=8*ccc*w18*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3+fmb2-2*fmc2+3*p2p4-p2p5-3*p3p4+p3p5)
      b(15)=ccc*(w18*(8*p3p5*(ffmcfmb*hbcm-fmc-hbcm)+8*p3p4*(ffmcfmb*
     . hbcm-fmc+2*hbcm)+8*p2p3*(-2*ffmcfmb*hbcm+fmc)+8*p1p3*(-3*
     . ffmcfmb*hbcm+fmb+4*fmc)+4*(-ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3-2
     . *fmb2*hbcm-2*fmc*hbcm2+3*fmc2*hbcm-4*hbcm*p2p4+2*hbcm*p2p5))+
     . w1*(8*p3p5*(ffmcfmb*hbcm-fmc-hbcm)+8*p3p4*(ffmcfmb*hbcm-fmc+2*
     . hbcm)+8*p2p3*(-2*ffmcfmb*hbcm+fmc)+8*p1p3*(-3*ffmcfmb*hbcm+fmb
     . +4*fmc)+4*(-ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3-2*fmb2*hbcm-2*fmc
     . *hbcm2+3*fmc2*hbcm-4*hbcm*p2p4+2*hbcm*p2p5)))
      b(16)=w18*ccc*(8*p3p5*(ffmcfmb*hbcm-fmc-hbcm)+8*p3p4*(ffmcfmb*
     . hbcm-fmc+2*hbcm)+8*p2p3*(-2*ffmcfmb*hbcm+fmc)+8*p1p3*(-3*
     . ffmcfmb*hbcm+fmb+4*fmc)+4*(-ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3-2
     . *fmb2*hbcm-2*fmc*hbcm2+3*fmc2*hbcm-4*hbcm*p2p4+2*hbcm*p2p5))
      b(21)=ccc*(8*w8*(-2*hbcm*p4p11+hbcm*p5p11)+8*w18*(3*ffmcfmb*
     . hbcm*p3p8-fmb*p3p8-4*fmc*p3p8-2*hbcm*p4p8-2*hbcm*p4p9+hbcm*
     . p5p8+hbcm*p5p9)+8*w1*(3*ffmcfmb*hbcm*p3p8-fmb*p3p8-4*fmc*p3p8-
     . 2*hbcm*p4p8+hbcm*p5p8))
      b(23)=ccc*(w18*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6
     . +hbcm*p5p6))+w1*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*
     . p4p6+hbcm*p5p6)))
      b(24)=ccc*(4*w18*(-3*ffmcfmb*hbcm+fmb+4*fmc)+4*w1*(-3*ffmcfmb*
     . hbcm+fmb+4*fmc))
      b(25)=ccc*(4*w18*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)+4*w1*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*p3p5)
     . )
      b(26)=ccc*(w18*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm
     . *p5p7))+w1*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7)))
      ans5=16*p3p6*(-ffmcfmb**2*hbcm2*p7p8-ffmcfmb**2*hbcm2*p7p9-
     . ffmcfmb*fmb*hbcm*p7p8-ffmcfmb*fmb*hbcm*p7p9+2*ffmcfmb*hbcm2*
     . p7p8+2*ffmcfmb*hbcm2*p7p9+fmb*hbcm*p7p8+fmb*hbcm*p7p9-2*fmb2*
     . p7p8-2*fmb2*p7p9-fmc*hbcm*p7p8-fmc*hbcm*p7p9+2*fmc2*p7p8+2*
     . fmc2*p7p9)
      ans4=32*(p3p7*p3p5)*(-ffmcfmb*p6p8-ffmcfmb*p6p9+2*p6p8+2*p6p9)+
     . 32*(p3p7*p3p4)*(-ffmcfmb*p6p8-ffmcfmb*p6p9-2*p6p8-2*p6p9)+32*(
     . p3p6*p3p4)*(ffmcfmb*p7p8+ffmcfmb*p7p9)+32*(p3p7*p3p6)*(2*
     . ffmcfmb*p2p8+2*ffmcfmb*p2p9-2*ffmcfmb*p3p8-2*ffmcfmb*p3p9-2*
     . ffmcfmb*p4p8-2*ffmcfmb*p4p9-ffmcfmb*p5p8-ffmcfmb*p5p9+2*p4p8+2
     . *p4p9-2*p5p8-2*p5p9)+32*(p3p7*p2p3)*(ffmcfmb*p6p8+ffmcfmb*p6p9
     . )+32*(p3p6*p2p3)*(-ffmcfmb*p7p8-ffmcfmb*p7p9)+16*p5p6*(-
     . ffmcfmb*hbcm2*p7p8-ffmcfmb*hbcm2*p7p9-fmc*hbcm*p7p8-fmc*hbcm*
     . p7p9)+16*p4p6*(-ffmcfmb*hbcm2*p7p8-ffmcfmb*hbcm2*p7p9-fmc*hbcm
     . *p7p8-fmc*hbcm*p7p9+2*hbcm2*p7p8+2*hbcm2*p7p9)+16*p3p7*(-2*
     . ffmcfmb*hbcm2*p6p8-2*ffmcfmb*hbcm2*p6p9+fmb*hbcm*p6p8+fmb*hbcm
     . *p6p9+2*fmb2*p6p8+2*fmb2*p6p9-fmc*hbcm*p6p8-fmc*hbcm*p6p9-2*
     . fmc2*p6p8-2*fmc2*p6p9)+ans5
      ans3=32*(p5p6*p3p7)*(p3p8+p3p9+p4p8+p4p9)+32*(p5p7*p3p5)*(p6p8+
     . p6p9)+32*(p4p6*p3p5)*(p7p8+p7p9)+32*(p5p7*p3p4)*(-p6p8-p6p9)+
     . 32*(p5p6*p3p4)*(-p7p8-p7p9)+64*(p4p7*p3p4)*(-p6p8-p6p9)+32*(
     . p4p6*p3p7)*(2*p2p8+2*p2p9-3*p3p8-3*p3p9-p4p8-p4p9-2*p5p8-2*
     . p5p9)+32*(p5p7*p3p6)*(-p2p8-p2p9+2*p4p8+2*p4p9)+32*(p4p7*p3p6)
     . *(p2p8+p2p9-2*p5p8-2*p5p9)+32*(p6p7*p3p5)*(p2p8+p2p9-p5p8-p5p9
     . )+32*(p6p7*p3p4)*(-p2p8-p2p9+2*p4p8+2*p4p9+p5p8+p5p9)+32*(p3p7
     . *p2p5)*(-p6p8-p6p9)+32*(p3p6*p2p5)*(p7p8+p7p9)+32*(p3p7*p2p4)*
     . (p6p8+p6p9)+32*(p3p6*p2p4)*(-p7p8-p7p9)+32*(p6p7*p2p3)*(-p4p8-
     . p4p9)+32*(p4p7*p2p3)*(p6p8+p6p9)+32*(p4p6*p2p3)*(-p7p8-p7p9)+
     . 16*p5p7*(fmb*hbcm*p6p8+fmb*hbcm*p6p9-fmc*hbcm*p6p8-fmc*hbcm*
     . p6p9)+16*p4p7*(fmb*hbcm*p6p8+fmb*hbcm*p6p9-fmc*hbcm*p6p8-fmc*
     . hbcm*p6p9-2*hbcm2*p6p8-2*hbcm2*p6p9)+16*p6p7*(fmb*hbcm*p2p8+
     . fmb*hbcm*p2p9-fmb*hbcm*p4p8-fmb*hbcm*p4p9-fmb*hbcm*p5p8-fmb*
     . hbcm*p5p9-fmc*hbcm*p2p8-fmc*hbcm*p2p9+fmc*hbcm*p4p8+fmc*hbcm*
     . p4p9+fmc*hbcm*p5p8+fmc*hbcm*p5p9+2*hbcm2*p4p8+2*hbcm2*p4p9)+
     . ans4
      ans2=w18*ans3
      ans8=32*(-ffmcfmb*p2p3*p3p6*p7p8+ffmcfmb*p2p3*p3p7*p6p8+ffmcfmb
     . *p3p4*p3p6*p7p8-p2p3*p4p6*p7p8+p2p3*p4p7*p6p8-p2p3*p4p8*p6p7-
     . p2p4*p3p6*p7p8+p2p4*p3p7*p6p8+p2p5*p3p6*p7p8-p2p5*p3p7*p6p8-2*
     . p3p4*p4p7*p6p8-p3p4*p5p6*p7p8-p3p4*p5p7*p6p8+p3p5*p4p6*p7p8+
     . p3p5*p5p7*p6p8)
      ans7=32*(p5p6*p3p7)*(p3p8+p4p8)+32*(p4p6*p3p7)*(2*p2p8-3*p3p8-
     . p4p8-2*p5p8)+32*(p5p7*p3p6)*(-p2p8+2*p4p8)+32*(p4p7*p3p6)*(
     . p2p8-2*p5p8)+32*(p6p7*p3p5)*(p2p8-p5p8)+32*(p6p7*p3p4)*(-p2p8+
     . 2*p4p8+p5p8)+16*p5p7*(fmb*hbcm*p6p8-fmc*hbcm*p6p8)+16*p4p7*(
     . fmb*hbcm*p6p8-fmc*hbcm*p6p8-2*hbcm2*p6p8)+16*p6p7*(fmb*hbcm*
     . p2p8-fmb*hbcm*p4p8-fmb*hbcm*p5p8-fmc*hbcm*p2p8+fmc*hbcm*p4p8+
     . fmc*hbcm*p5p8+2*hbcm2*p4p8)+32*(p3p7*p3p5)*(-ffmcfmb*p6p8+2*
     . p6p8)+32*(p3p7*p3p4)*(-ffmcfmb*p6p8-2*p6p8)+32*(p3p7*p3p6)*(2*
     . ffmcfmb*p2p8-2*ffmcfmb*p3p8-2*ffmcfmb*p4p8-ffmcfmb*p5p8+2*p4p8
     . -2*p5p8)+16*p5p6*(-ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8)+16*p4p6*(
     . -ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8+2*hbcm2*p7p8)+16*p3p7*(-2*
     . ffmcfmb*hbcm2*p6p8+fmb*hbcm*p6p8+2*fmb2*p6p8-fmc*hbcm*p6p8-2*
     . fmc2*p6p8)+16*p3p6*(-ffmcfmb**2*hbcm2*p7p8-ffmcfmb*fmb*hbcm*
     . p7p8+2*ffmcfmb*hbcm2*p7p8+fmb*hbcm*p7p8-2*fmb2*p7p8-fmc*hbcm*
     . p7p8+2*fmc2*p7p8)+ans8
      ans6=w1*ans7
      ans9=(16*p3p5*(p6p12+p7p13)+16*p3p4*(-2*p6p12+5*p7p13)+16*p6p7*
     . (-3*p3p11+2*p3p8-p3p9)+32*p2p3*(p6p12-2*p7p13)+16*p3p7*(-3*
     . p1p13-4*p2p13+6*p3p13+6*p4p13+3*p5p13-p6p11-p6p8+p6p9)+16*p3p6
     . *(-ffmcfmb*p3p12+p1p12-p2p12-2*p3p12-p4p12-2*p5p12+2*p7p11-
     . p7p8)+8*(ffmcfmb*hbcm2*p7p13+2*fmb*hbcm*p6p12-2*fmb*hbcm*p7p13
     . -2*fmc*hbcm*p6p12+3*fmc*hbcm*p7p13-hbcm2*p6p12+5*hbcm2*p7p13-2
     . *p1p3*p7p13-2*p3p12*p4p6))
      ans1=w8*(32*(p5p6*p3p7)*(p3p11+p4p11)+32*(p4p6*p3p7)*(2*p2p11-3
     . *p3p11-p4p11-2*p5p11)+32*(p5p7*p3p6)*(-p2p11+2*p4p11)+32*(p4p7
     . *p3p6)*(p2p11-2*p5p11)+32*(p6p7*p3p5)*(p2p11-p5p11)+32*(p6p7*
     . p3p4)*(-p2p11+2*p4p11+p5p11)+16*p5p7*(fmb*hbcm*p6p11-fmc*hbcm*
     . p6p11)+16*p4p7*(fmb*hbcm*p6p11-fmc*hbcm*p6p11-2*hbcm2*p6p11)+
     . 16*p6p7*(fmb*hbcm*p2p11-fmb*hbcm*p4p11-fmb*hbcm*p5p11-fmc*hbcm
     . *p2p11+fmc*hbcm*p4p11+fmc*hbcm*p5p11+2*hbcm2*p4p11)+32*(p3p7*
     . p3p5)*(-ffmcfmb*p6p11+2*p6p11)+32*(p3p7*p3p4)*(-ffmcfmb*p6p11-
     . 2*p6p11)+32*(p3p7*p3p6)*(2*ffmcfmb*p2p11-2*ffmcfmb*p3p11-2*
     . ffmcfmb*p4p11-ffmcfmb*p5p11+2*p4p11-2*p5p11)+16*p3p7*(-2*
     . ffmcfmb*hbcm2*p6p11+fmb*hbcm*p6p11+2*fmb2*p6p11-fmc*hbcm*p6p11
     . -2*fmc2*p6p11)+16*p3p6*(-ffmcfmb*fmc*hbcm*p7p11-ffmcfmb*hbcm2*
     . p7p11+fmc*hbcm*p7p11-fmc2*p7p11)+32*(ffmcfmb*p2p3*p3p6*p7p11+
     . ffmcfmb*p2p3*p3p7*p6p11-ffmcfmb*p3p4*p3p6*p7p11-p2p3*p4p11*
     . p6p7+p2p3*p4p7*p6p11+p2p4*p3p6*p7p11+p2p4*p3p7*p6p11-p2p5*p3p7
     . *p6p11-2*p3p4*p4p7*p6p11-p3p4*p5p7*p6p11+p3p5*p5p7*p6p11))+
     . ans2+ans6+ans9
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w18*(8*p5p6*(ffmcfmb*hbcm-fmc)+8*p4p6*(ffmcfmb*hbcm-
     . fmc)+8*p3p6*(-ffmcfmb**2*hbcm+ffmcfmb*fmb+2*ffmcfmb*fmc+2*
     . ffmcfmb*hbcm-fmb-3*fmc))+w1*(8*p5p6*(ffmcfmb*hbcm-fmc)+8*p4p6*
     . (ffmcfmb*hbcm-fmc)+8*p3p6*(-ffmcfmb**2*hbcm+ffmcfmb*fmb+2*
     . ffmcfmb*fmc+2*ffmcfmb*hbcm-fmb-3*fmc)))
      b(32)=w18*ccc*(8*p5p6*(-ffmcfmb*hbcm+fmc)+8*p4p6*(-ffmcfmb*hbcm
     . +fmc)+8*p3p6*(ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc-2*
     . ffmcfmb*hbcm+fmb+3*fmc))
      ans2=w8*(16*p5p7*(ffmcfmb*hbcm*p6p11-fmc*p6p11)+16*p4p7*(
     . ffmcfmb*hbcm*p6p11-fmc*p6p11)+16*p3p7*(ffmcfmb*hbcm*p6p11-fmc*
     . p6p11)+16*p6p7*(-2*ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p2p11+
     . ffmcfmb*hbcm*p4p11+ffmcfmb*hbcm*p5p11+fmb*p1p11+fmb*p2p11-fmb*
     . p4p11-fmb*p5p11+3*fmc*p1p11+2*fmc*p2p11-2*fmc*p4p11-2*fmc*
     . p5p11)+16*p3p6*(-ffmcfmb**2*hbcm*p7p11+ffmcfmb*fmc*p7p11+
     . ffmcfmb*hbcm*p7p11-fmc*p7p11))+8*(ffmcfmb*hbcm*p7p13-fmc*p7p13
     . )
      ans1=w18*(16*p5p7*(ffmcfmb*hbcm*p6p8+ffmcfmb*hbcm*p6p9-fmc*p6p8
     . -fmc*p6p9)+16*p4p7*(ffmcfmb*hbcm*p6p8+ffmcfmb*hbcm*p6p9-fmc*
     . p6p8-fmc*p6p9)+16*p3p7*(ffmcfmb*hbcm*p6p8+ffmcfmb*hbcm*p6p9-
     . fmc*p6p8-fmc*p6p9)+16*p6p7*(-2*ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*
     . p2p8-ffmcfmb*hbcm*p2p9+ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p4p9+
     . ffmcfmb*hbcm*p5p8+ffmcfmb*hbcm*p5p9+fmb*p1p8+fmb*p2p8+fmb*p2p9
     . -fmb*p4p8-fmb*p4p9-fmb*p5p8-fmb*p5p9+3*fmc*p1p8+2*fmc*p2p8+2*
     . fmc*p2p9-2*fmc*p4p8-2*fmc*p4p9-2*fmc*p5p8-2*fmc*p5p9)+16*p3p6*
     . (-ffmcfmb**2*hbcm*p7p8-ffmcfmb**2*hbcm*p7p9+ffmcfmb*fmc*p7p8+
     . ffmcfmb*fmc*p7p9+ffmcfmb*hbcm*p7p8+ffmcfmb*hbcm*p7p9-fmc*p7p8-
     . fmc*p7p9))+w1*(16*p5p7*(ffmcfmb*hbcm*p6p8-fmc*p6p8)+16*p4p7*(
     . ffmcfmb*hbcm*p6p8-fmc*p6p8)+16*p3p7*(ffmcfmb*hbcm*p6p8-fmc*
     . p6p8)+16*p6p7*(-2*ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p2p8+ffmcfmb*
     . hbcm*p4p8+ffmcfmb*hbcm*p5p8+fmb*p1p8+fmb*p2p8-fmb*p4p8-fmb*
     . p5p8+3*fmc*p1p8+2*fmc*p2p8-2*fmc*p4p8-2*fmc*p5p8)+16*p3p6*(-
     . ffmcfmb**2*hbcm*p7p8+ffmcfmb*fmc*p7p8+ffmcfmb*hbcm*p7p8-fmc*
     . p7p8))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(34)=4*ccc*w18*(3*ffmcfmb*hbcm-fmb-4*fmc)
      b(35)=ccc*(8*w18*(-ffmcfmb*hbcm*p7p8-ffmcfmb*hbcm*p7p9+fmc*p7p8
     . +fmc*p7p9)+8*w1*(-ffmcfmb*hbcm*p7p8+fmc*p7p8)+8*w8*(-ffmcfmb*
     . hbcm*p7p11+fmc*p7p11))
      b(36)=4*ccc*w18*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)
      b(37)=w18*ccc*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7))
      b(38)=ccc*(w18*(16*p3p7*(p4p8+p4p9)+16*p3p4*(-p7p8-p7p9)+8*(-
     . ffmcfmb*hbcm2*p7p8-ffmcfmb*hbcm2*p7p9-fmc*hbcm*p7p8-fmc*hbcm*
     . p7p9))+8*w1*(-ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8-2*p3p4*p7p8+2*
     . p3p7*p4p8)+8*w8*(-ffmcfmb*hbcm2*p7p11-fmc*hbcm*p7p11-2*p3p4*
     . p7p11+2*p3p7*p4p11))
      b(39)=w8*ccc*(8*p6p7*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+16*(2
     . *ffmcfmb*p3p6*p3p7-2*p3p4*p6p7+p3p5*p6p7+2*p3p6*p4p7-p3p6*p5p7
     . +p3p7*p4p6))
      b(40)=w8*ccc*(16*(p5p7*p3p6)*(-fmb-2*fmc+hbcm)+16*p6p7*(-
     . ffmcfmb*hbcm3+fmb*hbcm2-fmb2*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(
     . p5p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm-
     . hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+16*(p6p7*
     . p3p5)*(-ffmcfmb*hbcm+fmb+2*fmc-hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*
     . hbcm+fmb+2*fmc+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-fmb-2*fmc)+
     . 16*(p6p7*p1p3)*(2*ffmcfmb*hbcm-fmb-3*fmc)+16*(p3p7*p3p6)*(-
     . ffmcfmb*fmc-fmb-fmc)+16*(-hbcm*p2p4*p6p7+hbcm*p2p5*p6p7-2*hbcm
     . *p4p6*p4p7-hbcm*p4p6*p5p7+hbcm*p5p6*p5p7))
      b(41)=ccc*(w8*((p5p6*p3p7)*(-16*ffmcfmb+16)+(p4p6*p3p7)*(-16*
     . ffmcfmb-32)+(p5p7*p3p6)*(-32*ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*
     . hbcm2+2*ffmcfmb*hbcm2-2*fmb2+3*fmc2)+16*(-ffmcfmb*p2p3*p6p7-2*
     . ffmcfmb*p3p6*p3p7-2*p2p4*p6p7+p2p5*p6p7+2*p3p4*p6p7-p3p5*p6p7-
     . 2*p3p6*p4p7-3*p4p6*p4p7-2*p4p6*p5p7+p5p6*p5p7))+8*p6p7)
      b(42)=w18*ccc*(8*p6p7*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+16*(
     . 2*ffmcfmb*p3p6*p3p7-2*p3p4*p6p7+p3p5*p6p7+2*p3p6*p4p7-p3p6*
     . p5p7+p3p7*p4p6))
      b(43)=ccc*(w18*(16*p5p7*(-p6p8-p6p9)+16*p6p7*(-2*p4p8-2*p4p9+
     . p5p8+p5p9)+32*p4p7*(p6p8+p6p9)+16*p4p6*(p7p8+p7p9)+16*p3p7*(
     . ffmcfmb*p6p8+ffmcfmb*p6p9)+16*p3p6*(ffmcfmb*p7p8+ffmcfmb*p7p9)
     . )+w1*(16*p6p7*(-2*p4p8+p5p8)+16*(ffmcfmb*p3p6*p7p8+ffmcfmb*
     . p3p7*p6p8+p4p6*p7p8+2*p4p7*p6p8-p5p7*p6p8))+w8*(16*p6p7*(-2*
     . p4p11+p5p11)+16*(ffmcfmb*p3p6*p7p11+ffmcfmb*p3p7*p6p11+p4p6*
     . p7p11+2*p4p7*p6p11-p5p7*p6p11))+8*(p6p12-3*p7p13))
      b(44)=ccc*(w8*(8*p5p6*(-ffmcfmb*hbcm2-fmc*hbcm)+8*p4p6*(-
     . ffmcfmb*hbcm2-fmc*hbcm+2*hbcm2)+8*p3p6*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+fmb*hbcm-2*
     . fmb2-2*fmc*hbcm+3*fmc2)+16*(-2*ffmcfmb*p2p3*p3p6+2*ffmcfmb*
     . p3p4*p3p6-p2p3*p4p6-2*p2p4*p3p6+p2p5*p3p6-p3p4*p5p6+p3p5*p4p6)
     . )+8*p3p6)
      b(45)=8*ccc*w8*(2*ffmcfmb*p3p6+3*p4p6-p5p6)
      b(46)=w8*ccc*(8*p5p6*(-ffmcfmb*hbcm+fmc)+8*p4p6*(-ffmcfmb*hbcm+
     . fmc)+8*p3p6*(ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc-2*
     . ffmcfmb*hbcm+fmb+3*fmc))
      b(47)=8*ccc*w18*(2*ffmcfmb*p3p6+3*p4p6-p5p6)
      b(48)=8*ccc*(p3p6+p4p6+p5p6)
      b(49)=w8*ccc*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7))
      b(50)=8*ccc*w8*(2*ffmcfmb*p3p7+3*p4p7-p5p7)
      b(51)=ccc*(w8*((p3p7*p3p5)*(-16*ffmcfmb-16)+(p3p7*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm+2*hbcm2)
     . +8*p4p7*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-4*hbcm2)+8*p3p7*(
     . ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-fmc2)+16*
     . (2*ffmcfmb*p2p3*p3p7+2*p2p3*p4p7-p2p3*p5p7+p2p4*p3p7+3*p3p4*
     . p5p7-3*p3p5*p4p7))-8*p3p7)
      b(52)=8*ccc*w18*(2*ffmcfmb*p3p7+3*p4p7-p5p7)
      b(53)=ccc*(8*w18*(-3*p4p8-3*p4p9+p5p8+p5p9)+8*w1*(-3*p4p8+p5p8)
     . +8*w8*(-3*p4p11+p5p11))
      b(54)=4*ccc*w8*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)
      b(55)=4*ccc*w8*(3*ffmcfmb*hbcm-fmb-4*fmc)
      b(56)=ccc*(8*w8*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3+fmb2-2*fmc2+3*p2p4-p2p5-3*p3p4+p3p5)-8)
      b(57)=-8*ccc
      b(58)=w8*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6+
     . hbcm*p5p6))
      b(59)=ccc*(w8*(8*p3p5*(ffmcfmb*hbcm-fmc-hbcm)+8*p3p4*(ffmcfmb*
     . hbcm-fmc+2*hbcm)+8*p2p3*(-2*ffmcfmb*hbcm+fmc)+8*p1p3*(-3*
     . ffmcfmb*hbcm+fmb+4*fmc)+4*(-ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3-2
     . *fmb2*hbcm-2*fmc*hbcm2+3*fmc2*hbcm-4*hbcm*p2p4+2*hbcm*p2p5))+4
     . *hbcm)
      b(60)=w18*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6+
     . hbcm*p5p6))
      b(61)=-4*ccc*hbcm
      b(62)=8*ccc*(ffmcfmb*hbcm*p3p6+hbcm*p4p6+hbcm*p5p6)
      b(65)=ccc*(w18*(8*p6p7*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+16*
     . (2*ffmcfmb*p3p6*p3p7-2*p3p4*p6p7+p3p5*p6p7+2*p3p6*p4p7-p3p6*
     . p5p7+p3p7*p4p6))+w1*(8*p6p7*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*
     . hbcm)+16*(2*ffmcfmb*p3p6*p3p7-2*p3p4*p6p7+p3p5*p6p7+2*p3p6*
     . p4p7-p3p6*p5p7+p3p7*p4p6)))
      b(66)=ccc*(w18*(16*p5p7*(-hbcm*p6p8-hbcm*p6p9)+16*p6p7*(-hbcm*
     . p4p8-hbcm*p4p9+hbcm*p5p8+hbcm*p5p9)+16*p4p7*(hbcm*p6p8+hbcm*
     . p6p9)+16*p4p6*(hbcm*p7p8+hbcm*p7p9)+16*p3p6*(fmc*p7p8+fmc*p7p9
     . )+16*p3p7*(-ffmcfmb*hbcm*p6p8-ffmcfmb*hbcm*p6p9+fmb*p6p8+fmb*
     . p6p9+2*fmc*p6p8+2*fmc*p6p9))+w1*(16*p6p7*(-hbcm*p4p8+hbcm*p5p8
     . )+16*p3p7*(-ffmcfmb*hbcm*p6p8+fmb*p6p8+2*fmc*p6p8)+16*(fmc*
     . p3p6*p7p8+hbcm*p4p6*p7p8+hbcm*p4p7*p6p8-hbcm*p5p7*p6p8))+w8*(
     . 16*p6p7*(-hbcm*p4p11+hbcm*p5p11)+16*p3p7*(-ffmcfmb*hbcm*p6p11+
     . fmb*p6p11+2*fmc*p6p11)+16*(fmc*p3p6*p7p11+hbcm*p4p6*p7p11+hbcm
     . *p4p7*p6p11-hbcm*p5p7*p6p11))+8*(hbcm*p6p12-2*hbcm*p7p13))
      b(68)=ccc*(8*w18*(2*ffmcfmb*p3p6+3*p4p6-p5p6)+8*w1*(2*ffmcfmb*
     . p3p6+3*p4p6-p5p6))
      b(71)=ccc*(8*w18*(2*ffmcfmb*p3p7+3*p4p7-p5p7)+8*w1*(2*ffmcfmb*
     . p3p7+3*p4p7-p5p7))
      b(73)=8*ccc*w8*p6p7*(2*ffmcfmb*hbcm-fmb-3*fmc)
      b(74)=ccc*(8*w18*p6p7*(2*ffmcfmb*hbcm-fmb-3*fmc)+8*w1*p6p7*(2*
     . ffmcfmb*hbcm-fmb-3*fmc))
      b(75)=ccc*(8*w18*(3*ffmcfmb*hbcm*p1p8+3*ffmcfmb*hbcm*p1p9+
     . ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p2p9-3*ffmcfmb*hbcm*p3p8-3*
     . ffmcfmb*hbcm*p3p9-ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p4p9-ffmcfmb*
     . hbcm*p5p8-ffmcfmb*hbcm*p5p9-fmb*p1p8-fmb*p1p9+fmb*p3p8+fmb*
     . p3p9-4*fmc*p1p8-4*fmc*p1p9-fmc*p2p8-fmc*p2p9+4*fmc*p3p8+4*fmc*
     . p3p9+fmc*p4p8+fmc*p4p9+fmc*p5p8+fmc*p5p9)+8*w1*(3*ffmcfmb*hbcm
     . *p1p8+ffmcfmb*hbcm*p2p8-3*ffmcfmb*hbcm*p3p8-ffmcfmb*hbcm*p4p8-
     . ffmcfmb*hbcm*p5p8-fmb*p1p8+fmb*p3p8-4*fmc*p1p8-fmc*p2p8+4*fmc*
     . p3p8+fmc*p4p8+fmc*p5p8)+8*w8*(3*ffmcfmb*hbcm*p1p11+ffmcfmb*
     . hbcm*p2p11-3*ffmcfmb*hbcm*p3p11-ffmcfmb*hbcm*p4p11-ffmcfmb*
     . hbcm*p5p11-fmb*p1p11+fmb*p3p11-4*fmc*p1p11-fmc*p2p11+4*fmc*
     . p3p11+fmc*p4p11+fmc*p5p11))
      ans2=w8*(16*p2p3*(-2*p4p11+p5p11)+16*p3p5*(p1p11-p3p11+2*p4p11-
     . p5p11)+16*p3p4*(-3*p1p11-p2p11+3*p3p11+3*p4p11)+8*(-2*ffmcfmb*
     . hbcm2*p1p11-ffmcfmb*hbcm2*p2p11+2*ffmcfmb*hbcm2*p3p11+ffmcfmb*
     . hbcm2*p4p11+ffmcfmb*hbcm2*p5p11+fmb*hbcm*p1p11-fmb*hbcm*p3p11-
     . 3*fmc*hbcm*p1p11-fmc*hbcm*p2p11+3*fmc*hbcm*p3p11+fmc*hbcm*
     . p4p11+fmc*hbcm*p5p11+4*hbcm2*p4p11-2*hbcm2*p5p11))
      ans1=w18*(16*p2p3*(-2*p4p8-2*p4p9+p5p8+p5p9)+16*p3p5*(p1p8+p1p9
     . -p3p8-p3p9+2*p4p8+2*p4p9-p5p8-p5p9)+16*p3p4*(-3*p1p8-3*p1p9-
     . p2p8-p2p9+3*p3p8+3*p3p9+3*p4p8+3*p4p9)+8*(-2*ffmcfmb*hbcm2*
     . p1p8-2*ffmcfmb*hbcm2*p1p9-ffmcfmb*hbcm2*p2p8-ffmcfmb*hbcm2*
     . p2p9+2*ffmcfmb*hbcm2*p3p8+2*ffmcfmb*hbcm2*p3p9+ffmcfmb*hbcm2*
     . p4p8+ffmcfmb*hbcm2*p4p9+ffmcfmb*hbcm2*p5p8+ffmcfmb*hbcm2*p5p9+
     . fmb*hbcm*p1p8+fmb*hbcm*p1p9-fmb*hbcm*p3p8-fmb*hbcm*p3p9-3*fmc*
     . hbcm*p1p8-3*fmc*hbcm*p1p9-fmc*hbcm*p2p8-fmc*hbcm*p2p9+3*fmc*
     . hbcm*p3p8+3*fmc*hbcm*p3p9+fmc*hbcm*p4p8+fmc*hbcm*p4p9+fmc*hbcm
     . *p5p8+fmc*hbcm*p5p9+4*hbcm2*p4p8+4*hbcm2*p4p9-2*hbcm2*p5p8-2*
     . hbcm2*p5p9))+w1*(16*p2p3*(-2*p4p8+p5p8)+16*p3p5*(p1p8-p3p8+2*
     . p4p8-p5p8)+16*p3p4*(-3*p1p8-p2p8+3*p3p8+3*p4p8)+8*(-2*ffmcfmb*
     . hbcm2*p1p8-ffmcfmb*hbcm2*p2p8+2*ffmcfmb*hbcm2*p3p8+ffmcfmb*
     . hbcm2*p4p8+ffmcfmb*hbcm2*p5p8+fmb*hbcm*p1p8-fmb*hbcm*p3p8-3*
     . fmc*hbcm*p1p8-fmc*hbcm*p2p8+3*fmc*hbcm*p3p8+fmc*hbcm*p4p8+fmc*
     . hbcm*p5p8+4*hbcm2*p4p8-2*hbcm2*p5p8))+ans2
      ans=ccc*ans1
      b(76)=ans
      b(77)=ccc*(w18*(16*p3p6*(2*p4p8+2*p4p9-p5p8-p5p9)+16*p3p5*(p6p8
     . +p6p9)+32*p3p4*(-p6p8-p6p9)+8*(-ffmcfmb*hbcm2*p6p8-ffmcfmb*
     . hbcm2*p6p9+fmb*hbcm*p6p8+fmb*hbcm*p6p9-2*fmc*hbcm*p6p8-2*fmc*
     . hbcm*p6p9))+w1*(16*p3p6*(2*p4p8-p5p8)+8*(-ffmcfmb*hbcm2*p6p8+
     . fmb*hbcm*p6p8-2*fmc*hbcm*p6p8-4*p3p4*p6p8+2*p3p5*p6p8))+w8*(16
     . *p3p6*(2*p4p11-p5p11)+8*(-ffmcfmb*hbcm2*p6p11+fmb*hbcm*p6p11-2
     . *fmc*hbcm*p6p11-4*p3p4*p6p11+2*p3p5*p6p11))+8*p3p13)
      b(78)=ccc*(8*w18*(2*ffmcfmb*hbcm*p6p8+2*ffmcfmb*hbcm*p6p9-fmb*
     . p6p8-fmb*p6p9-3*fmc*p6p8-3*fmc*p6p9)+8*w1*(2*ffmcfmb*hbcm*p6p8
     . -fmb*p6p8-3*fmc*p6p8)+8*w8*(2*ffmcfmb*hbcm*p6p11-fmb*p6p11-3*
     . fmc*p6p11))
      b(79)=8*ccc*w18*p6p7*(2*ffmcfmb*hbcm-fmb-3*fmc)
      b(80)=w18*ccc*(8*p5p7*(-2*ffmcfmb*hbcm+fmb+3*fmc)+8*p4p7*(-2*
     . ffmcfmb*hbcm+fmb+3*fmc)+8*p3p7*(-2*ffmcfmb*hbcm+fmb+3*fmc))
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(fmc+hbcm)+8*(hbcm*p4p7+2*hbcm*p5p7))
      b(84)=8*ccc*(2*p3p7+p4p7+3*p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-4*hbcm2+2*p1p3+4*p2p3-6*p3p4-2*
     . p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(89)=-4*ccc*hbcm
      b(90)=w8*ccc*(8*p5p7*(-2*ffmcfmb*hbcm+fmb+3*fmc)+8*p4p7*(-2*
     . ffmcfmb*hbcm+fmb+3*fmc)+8*p3p7*(-2*ffmcfmb*hbcm+fmb+3*fmc))
      b(91)=ccc*(w18*(8*p5p7*(-2*ffmcfmb*hbcm+fmb+3*fmc)+8*p4p7*(-2*
     . ffmcfmb*hbcm+fmb+3*fmc)+8*p3p7*(-2*ffmcfmb*hbcm+fmb+3*fmc))+w1
     . *(8*p5p7*(-2*ffmcfmb*hbcm+fmb+3*fmc)+8*p4p7*(-2*ffmcfmb*hbcm+
     . fmb+3*fmc)+8*p3p7*(-2*ffmcfmb*hbcm+fmb+3*fmc)))
      b(92)=8*ccc*(-2*hbcm2+3*p1p3+2*p2p3-2*p3p4-2*p3p5)
      b(93)=8*ccc*p3p6
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp17_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(2)=w2*ccc*((p6p7*p1p3)*(-16*ffmcfmb+32)+8*p6p7*(ffmcfmb**2*
     . hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*
     . fmb*fmc+fmc*hbcm+fmc2)+16*(ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*
     . p5p7-p1p2*p6p7+3*p1p5*p6p7))
      ans=ccc*(w2*(16*p1p3*(-ffmcfmb*p7p9-p7p9)+8*(ffmcfmb**2*hbcm2*
     . p7p9+2*ffmcfmb*fmb*hbcm*p7p9-2*ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*
     . hbcm2*p7p9-2*ffmcfmb*p1p9*p3p7-2*fmb*fmc*p7p9+fmc*hbcm*p7p9+
     . fmc2*p7p9+4*p1p2*p7p9-6*p1p4*p7p9-2*p1p9*p4p7))+w17*(48*p1p4*(
     . -p7p10-p7p11)+32*p1p2*(p7p10+p7p11)+16*p4p7*(-p1p10-p1p11)+16*
     . p1p3*(-ffmcfmb*p7p10-ffmcfmb*p7p11-p7p10-p7p11)+16*p3p7*(-
     . ffmcfmb*p1p10-ffmcfmb*p1p11)+8*(ffmcfmb**2*hbcm2*p7p10+ffmcfmb
     . **2*hbcm2*p7p11+2*ffmcfmb*fmb*hbcm*p7p10+2*ffmcfmb*fmb*hbcm*
     . p7p11-2*ffmcfmb*fmc*hbcm*p7p10-2*ffmcfmb*fmc*hbcm*p7p11-
     . ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p11-2*fmb*fmc*p7p10-2*fmb*
     . fmc*p7p11+fmc*hbcm*p7p10+fmc*hbcm*p7p11+fmc2*p7p10+fmc2*p7p11)
     . )+w12*(16*p1p3*(-ffmcfmb*p7p10-p7p10)+8*(ffmcfmb**2*hbcm2*
     . p7p10+2*ffmcfmb*fmb*hbcm*p7p10-2*ffmcfmb*fmc*hbcm*p7p10-
     . ffmcfmb*hbcm2*p7p10-2*ffmcfmb*p1p10*p3p7-2*fmb*fmc*p7p10+fmc*
     . hbcm*p7p10+fmc2*p7p10-2*p1p10*p4p7+4*p1p2*p7p10-6*p1p4*p7p10))
     . +8*(-p3p12-p4p12-p5p12-p7p10+p7p11))
      b(3)=ans
      b(5)=w2*ccc*(16*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-ffmcfmb
     . **2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*fmb*fmc-fmc2
     . )+16*(2*ffmcfmb*p1p3*p3p7-2*p1p2*p3p7-p1p3*p5p7+3*p1p4*p3p7))
      ans3=w12*(32*p3p4*(-ffmcfmb*hbcm*p7p10+fmc*p7p10)+16*p2p3*(
     . ffmcfmb*hbcm*p7p10-fmc*p7p10)+16*p1p3*(2*ffmcfmb*hbcm*p7p10+
     . fmb*p7p10-fmc*p7p10)+16*p3p7*(ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*
     . p2p10-2*ffmcfmb*hbcm*p5p10-fmb*p1p10-fmc*p1p10-fmc*p2p10+2*fmc
     . *p5p10)+8*(ffmcfmb*fmb*hbcm2*p7p10+ffmcfmb*fmc*hbcm2*p7p10-2*
     . ffmcfmb*hbcm3*p7p10-fmb*fmc*hbcm*p7p10+2*fmc*hbcm2*p7p10-fmc2*
     . hbcm*p7p10-2*hbcm*p1p10*p5p7-2*hbcm*p1p2*p7p10+4*hbcm*p1p4*
     . p7p10))+8*(hbcm*p2p12+hbcm*p3p12+hbcm*p4p12+hbcm*p5p12-2*hbcm*
     . p7p11)
      ans2=w17*(32*p1p4*(hbcm*p7p10+hbcm*p7p11)+16*p1p2*(-hbcm*p7p10-
     . hbcm*p7p11)+16*p5p7*(-hbcm*p1p10-hbcm*p1p11)+32*p3p4*(-ffmcfmb
     . *hbcm*p7p10-ffmcfmb*hbcm*p7p11+fmc*p7p10+fmc*p7p11)+16*p2p3*(
     . ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p11-fmc*p7p10-fmc*p7p11)+16*
     . p1p3*(2*ffmcfmb*hbcm*p7p10+2*ffmcfmb*hbcm*p7p11+fmb*p7p10+fmb*
     . p7p11-fmc*p7p10-fmc*p7p11)+16*p3p7*(ffmcfmb*hbcm*p1p10+ffmcfmb
     . *hbcm*p1p11+ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p2p11-2*ffmcfmb*
     . hbcm*p5p10-2*ffmcfmb*hbcm*p5p11-fmb*p1p10-fmb*p1p11-fmc*p1p10-
     . fmc*p1p11-fmc*p2p10-fmc*p2p11+2*fmc*p5p10+2*fmc*p5p11)+8*(
     . ffmcfmb*fmb*hbcm2*p7p10+ffmcfmb*fmb*hbcm2*p7p11+ffmcfmb*fmc*
     . hbcm2*p7p10+ffmcfmb*fmc*hbcm2*p7p11-2*ffmcfmb*hbcm3*p7p10-2*
     . ffmcfmb*hbcm3*p7p11-fmb*fmc*hbcm*p7p10-fmb*fmc*hbcm*p7p11+2*
     . fmc*hbcm2*p7p10+2*fmc*hbcm2*p7p11-fmc2*hbcm*p7p10-fmc2*hbcm*
     . p7p11))+ans3
      ans1=w2*(32*p3p4*(-ffmcfmb*hbcm*p7p9+fmc*p7p9)+16*p2p3*(ffmcfmb
     . *hbcm*p7p9-fmc*p7p9)+16*p3p7*(ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*
     . p2p9-2*ffmcfmb*hbcm*p5p9-fmb*p1p9-fmc*p1p9-fmc*p2p9+2*fmc*p5p9
     . )+16*p1p3*(2*ffmcfmb*hbcm*p7p9+fmb*p7p9-fmc*p7p9)+8*(ffmcfmb*
     . fmb*hbcm2*p7p9+ffmcfmb*fmc*hbcm2*p7p9-2*ffmcfmb*hbcm3*p7p9-fmb
     . *fmc*hbcm*p7p9+2*fmc*hbcm2*p7p9-fmc2*hbcm*p7p9-2*hbcm*p1p2*
     . p7p9+4*hbcm*p1p4*p7p9-2*hbcm*p1p9*p5p7))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans=ccc*(w2*(16*p3p6*(ffmcfmb*p2p9-2*ffmcfmb*p3p9-2*ffmcfmb*
     . p4p9-p1p9)+16*p1p3*(-ffmcfmb*p6p9+p6p9)+8*(ffmcfmb**2*hbcm2*
     . p6p9+ffmcfmb*fmb*hbcm*p6p9-ffmcfmb*fmc*hbcm*p6p9-ffmcfmb*hbcm2
     . *p6p9-fmb*fmc*p6p9+fmc*hbcm*p6p9+2*p1p5*p6p9-6*p1p9*p4p6))+w17
     . *(16*p1p5*(p6p10+p6p11)+48*p4p6*(-p1p10-p1p11)+16*p3p6*(
     . ffmcfmb*p2p10+ffmcfmb*p2p11-2*ffmcfmb*p3p10-2*ffmcfmb*p3p11-2*
     . ffmcfmb*p4p10-2*ffmcfmb*p4p11-p1p10-p1p11)+16*p1p3*(-ffmcfmb*
     . p6p10-ffmcfmb*p6p11+p6p10+p6p11)+8*(ffmcfmb**2*hbcm2*p6p10+
     . ffmcfmb**2*hbcm2*p6p11+ffmcfmb*fmb*hbcm*p6p10+ffmcfmb*fmb*hbcm
     . *p6p11-ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*fmc*hbcm*p6p11-ffmcfmb*
     . hbcm2*p6p10-ffmcfmb*hbcm2*p6p11-fmb*fmc*p6p10-fmb*fmc*p6p11+
     . fmc*hbcm*p6p10+fmc*hbcm*p6p11))+w12*(16*p3p6*(ffmcfmb*p2p10-2*
     . ffmcfmb*p3p10-2*ffmcfmb*p4p10-p1p10)+16*p1p3*(-ffmcfmb*p6p10+
     . p6p10)+8*(ffmcfmb**2*hbcm2*p6p10+ffmcfmb*fmb*hbcm*p6p10-
     . ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*hbcm2*p6p10-fmb*fmc*p6p10+fmc*
     . hbcm*p6p10-6*p1p10*p4p6+2*p1p5*p6p10))+8*(-3*p1p13-2*p2p13+4*
     . p3p13+p4p13+3*p5p13-p6p10))
      b(7)=ans
      b(9)=w2*ccc*(16*p4p6*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p6*(-ffmcfmb
     . **2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*ffmcfmb*
     . hbcm2+fmb*fmc)+16*(ffmcfmb*p1p3*p3p6-ffmcfmb*p2p3*p3p6+2*
     . ffmcfmb*p3p4*p3p6+3*p1p3*p4p6-p1p5*p3p6))
      ans3=w12*(16*p3p5*(ffmcfmb*hbcm*p6p10-fmc*p6p10)+16*p4p6*(-3*
     . ffmcfmb*hbcm*p3p10+3*fmc*p3p10-hbcm*p1p10)+16*p3p6*(2*ffmcfmb*
     . fmb*p3p10+2*ffmcfmb*fmc*p3p10-2*ffmcfmb*hbcm*p1p10-ffmcfmb*
     . hbcm*p2p10+2*ffmcfmb*hbcm*p4p10+fmc*p1p10+fmc*p2p10-fmc*p4p10)
     . +8*(-ffmcfmb**2*hbcm3*p6p10+ffmcfmb*fmb*hbcm2*p6p10+ffmcfmb*
     . fmc*hbcm2*p6p10+ffmcfmb*hbcm3*p6p10-fmb*fmc*hbcm*p6p10-2*fmb*
     . p1p3*p6p10-fmc*hbcm2*p6p10-2*hbcm*p1p5*p6p10))+8*(-3*fmb*p3p13
     . -3*fmc*p3p13+2*hbcm*p1p13+2*hbcm*p2p13-hbcm*p3p13-hbcm*p4p13-2
     . *hbcm*p5p13+hbcm*p6p10)
      ans2=w17*(16*p1p5*(-hbcm*p6p10-hbcm*p6p11)+16*p1p3*(-fmb*p6p10-
     . fmb*p6p11)+16*p3p5*(ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p11-fmc*
     . p6p10-fmc*p6p11)+16*p4p6*(-3*ffmcfmb*hbcm*p3p10+3*fmc*p3p10-
     . hbcm*p1p10-hbcm*p1p11)+16*p3p6*(2*ffmcfmb*fmb*p3p10+2*ffmcfmb*
     . fmc*p3p10-2*ffmcfmb*hbcm*p1p10-2*ffmcfmb*hbcm*p1p11-ffmcfmb*
     . hbcm*p2p10-ffmcfmb*hbcm*p2p11+2*ffmcfmb*hbcm*p3p11+2*ffmcfmb*
     . hbcm*p4p10+2*ffmcfmb*hbcm*p4p11+fmc*p1p10+fmc*p1p11+fmc*p2p10+
     . fmc*p2p11-fmc*p3p11-fmc*p4p10-fmc*p4p11)+8*(-ffmcfmb**2*hbcm3*
     . p6p10-ffmcfmb**2*hbcm3*p6p11+ffmcfmb*fmb*hbcm2*p6p10+ffmcfmb*
     . fmb*hbcm2*p6p11+ffmcfmb*fmc*hbcm2*p6p10+ffmcfmb*fmc*hbcm2*
     . p6p11+ffmcfmb*hbcm3*p6p10+ffmcfmb*hbcm3*p6p11-fmb*fmc*hbcm*
     . p6p10-fmb*fmc*hbcm*p6p11-fmc*hbcm2*p6p10-fmc*hbcm2*p6p11))+
     . ans3
      ans1=w2*(16*p3p5*(ffmcfmb*hbcm*p6p9-fmc*p6p9)+16*p3p6*(-2*
     . ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+2*ffmcfmb*hbcm*p3p9+2*
     . ffmcfmb*hbcm*p4p9+fmc*p1p9+fmc*p2p9-fmc*p3p9-fmc*p4p9)+8*(-
     . ffmcfmb**2*hbcm3*p6p9+ffmcfmb*fmb*hbcm2*p6p9+ffmcfmb*fmc*hbcm2
     . *p6p9+ffmcfmb*hbcm3*p6p9-fmb*fmc*hbcm*p6p9-2*fmb*p1p3*p6p9-fmc
     . *hbcm2*p6p9-2*hbcm*p1p5*p6p9-2*hbcm*p1p9*p4p6))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(12)=w2*ccc*(16*(p6p7*p1p3)*(-fmb-fmc+2*hbcm)+32*(p4p6*p3p7)*(
     . -ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6)*(2*ffmcfmb*hbcm-fmc)+32*(
     . p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-
     . fmc)+16*(p3p7*p3p6)*(ffmcfmb*fmb+2*ffmcfmb*fmc)+8*p6p7*(-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(-hbcm*p1p2*p6p7+2*hbcm*p1p5*
     . p6p7))
      b(14)=w2*ccc*(p1p3*(16*ffmcfmb+8)+4*(-2*ffmcfmb**2*hbcm2-3*
     . ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+3*fmb*fmc-
     . 2*fmc*hbcm-fmc2-6*p1p2+8*p1p4))
      b(16)=w2*ccc*(24*p3p4*(ffmcfmb*hbcm-fmc)+16*p2p3*(-ffmcfmb*hbcm
     . +fmc)+8*p1p3*(-4*ffmcfmb*hbcm-fmb+2*fmc)+4*(ffmcfmb**2*hbcm3-2
     . *ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+2*fmb*
     . fmc*hbcm-3*fmc*hbcm2+fmc2*hbcm+4*hbcm*p1p2-6*hbcm*p1p4))
      b(17)=ccc*(w17*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)+w12*(
     . 8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6))
      b(18)=ccc*(4*w17*(-3*ffmcfmb*hbcm-4*fmb-fmc+3*hbcm)+4*w12*(-3*
     . ffmcfmb*hbcm-4*fmb-fmc+3*hbcm))
      b(19)=ccc*(4*w17*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+
     . 6*p1p3+6*p2p3-8*p3p4)+4*w12*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*
     . hbcm-4*hbcm2+6*p1p3+6*p2p3-8*p3p4))
      b(20)=ccc*(w17*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)+
     . w12*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7))
      b(21)=ccc*(8*w2*(-2*hbcm*p1p9-2*hbcm*p2p9+3*hbcm*p3p9+3*hbcm*
     . p4p9)+8*w17*(3*ffmcfmb*hbcm*p3p10+4*fmb*p3p10+fmc*p3p10-2*hbcm
     . *p1p10-2*hbcm*p1p11-2*hbcm*p2p10-2*hbcm*p2p11+3*hbcm*p3p11+3*
     . hbcm*p4p10+3*hbcm*p4p11)+8*w12*(3*ffmcfmb*hbcm*p3p10+4*fmb*
     . p3p10+fmc*p3p10-2*hbcm*p1p10-2*hbcm*p2p10+3*hbcm*p4p10))
      b(22)=ccc*(w17*(24*p3p4*(ffmcfmb*hbcm-fmc)+16*p2p3*(-ffmcfmb*
     . hbcm+fmc)+8*p1p3*(-4*ffmcfmb*hbcm-fmb+2*fmc)+4*(ffmcfmb**2*
     . hbcm3-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+
     . 2*fmb*fmc*hbcm-3*fmc*hbcm2+fmc2*hbcm+4*hbcm*p1p2-6*hbcm*p1p4))
     . +w12*(24*p3p4*(ffmcfmb*hbcm-fmc)+16*p2p3*(-ffmcfmb*hbcm+fmc)+8
     . *p1p3*(-4*ffmcfmb*hbcm-fmb+2*fmc)+4*(ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+2*fmb*
     . fmc*hbcm-3*fmc*hbcm2+fmc2*hbcm+4*hbcm*p1p2-6*hbcm*p1p4))-4*
     . hbcm)
      b(27)=ccc*(w17*(24*p4p6*(ffmcfmb*hbcm-fmc)+8*p3p6*(-2*ffmcfmb*
     . fmb-2*ffmcfmb*fmc+2*ffmcfmb*hbcm-fmc))+w12*(24*p4p6*(ffmcfmb*
     . hbcm-fmc)+8*p3p6*(-2*ffmcfmb*fmb-2*ffmcfmb*fmc+2*ffmcfmb*hbcm-
     . fmc)))
      b(28)=ccc*(w17*(16*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*
     . ffmcfmb*hbcm2-fmb*fmc)+16*(-ffmcfmb*p1p3*p3p6+ffmcfmb*p2p3*
     . p3p6-2*ffmcfmb*p3p4*p3p6-3*p1p3*p4p6+p1p5*p3p6))+w12*(16*p4p6*
     . (ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-fmb*fmc)+16*(-
     . ffmcfmb*p1p3*p3p6+ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6-3*p1p3
     . *p4p6+p1p5*p3p6))-8*p3p6)
      b(29)=ccc*(w17*(32*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p5p7*
     . p3p6)*(2*ffmcfmb*hbcm-fmc)+32*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+
     . 16*(p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+32*(p6p7*p1p3)*(ffmcfmb*hbcm
     . +fmb)+16*(p3p7*p3p6)*(ffmcfmb*fmb+2*ffmcfmb*fmc)+8*p6p7*(-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(-hbcm*p1p2*p6p7+2*hbcm*p1p5*
     . p6p7))+w12*(32*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6)*
     . (2*ffmcfmb*hbcm-fmc)+32*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(
     . p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+32*(p6p7*p1p3)*(ffmcfmb*hbcm+fmb
     . )+16*(p3p7*p3p6)*(ffmcfmb*fmb+2*ffmcfmb*fmc)+8*p6p7*(-ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm+2*fmc
     . *hbcm2+fmc2*hbcm)+16*(-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7)))
      ans4=16*p6p7*(ffmcfmb*hbcm2*p2p10+ffmcfmb*hbcm2*p2p11-ffmcfmb*
     . hbcm2*p5p10-ffmcfmb*hbcm2*p5p11+fmb*hbcm*p1p10+fmb*hbcm*p1p11-
     . fmc*hbcm*p1p10-fmc*hbcm*p1p11-fmc*hbcm*p2p10-fmc*hbcm*p2p11+
     . fmc*hbcm*p5p10+fmc*hbcm*p5p11+5*hbcm2*p1p10+5*hbcm2*p1p11)+16*
     . p3p7*(-ffmcfmb**2*hbcm2*p6p10-ffmcfmb**2*hbcm2*p6p11+ffmcfmb*
     . fmc*hbcm*p6p10+ffmcfmb*fmc*hbcm*p6p11+ffmcfmb*hbcm2*p6p10+
     . ffmcfmb*hbcm2*p6p11-fmc*hbcm*p6p10-fmc*hbcm*p6p11)+16*p3p6*(-
     . ffmcfmb**2*hbcm2*p7p10-ffmcfmb**2*hbcm2*p7p11-ffmcfmb*fmb*hbcm
     . *p7p10-ffmcfmb*fmb*hbcm*p7p11+ffmcfmb*fmc*hbcm*p7p10+ffmcfmb*
     . fmc*hbcm*p7p11+ffmcfmb*hbcm2*p7p10+ffmcfmb*hbcm2*p7p11+fmb*fmc
     . *p7p10+fmb*fmc*p7p11)
      ans3=32*(p3p6*p1p5)*(-p7p10-p7p11)+32*(p5p7*p1p3)*(-p6p10-p6p11
     . )+64*(p4p6*p1p3)*(p7p10+p7p11)+128*(p4p6*p3p7)*(p1p10+p1p11)+
     . 32*(p5p7*p3p6)*(p1p10+p1p11)+32*(p4p7*p3p6)*(p1p10+p1p11)+96*(
     . p6p7*p3p5)*(p1p10+p1p11)+64*(p6p7*p3p4)*(p1p10+p1p11)+64*(p6p7
     . *p2p3)*(-p1p10-p1p11)+32*(p6p7*p1p3)*(-3*p1p10-3*p1p11-p2p10-
     . p2p11+2*p5p10+2*p5p11)+32*(p3p6*p3p4)*(ffmcfmb*p7p10+ffmcfmb*
     . p7p11)+32*(p3p7*p3p6)*(-2*ffmcfmb*p2p10-2*ffmcfmb*p2p11+2*
     . ffmcfmb*p3p10+2*ffmcfmb*p3p11+2*ffmcfmb*p4p10+2*ffmcfmb*p4p11+
     . ffmcfmb*p5p10+ffmcfmb*p5p11+2*p1p10+2*p1p11)+32*(p3p7*p1p3)*(
     . ffmcfmb*p6p10+ffmcfmb*p6p11-p6p10-p6p11)+32*(p3p6*p1p3)*(
     . ffmcfmb*p7p10+ffmcfmb*p7p11)+16*p5p7*(ffmcfmb*hbcm2*p6p10+
     . ffmcfmb*hbcm2*p6p11-fmc*hbcm*p6p10-fmc*hbcm*p6p11)+16*p4p6*(-
     . ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p11+fmc*hbcm*p7p10+fmc*
     . hbcm*p7p11)+ans4
      ans2=w17*ans3
      ans5=w12*(32*(p6p7*p1p3)*(-3*p1p10-p2p10+2*p5p10)+32*(p3p7*p3p6
     . )*(-2*ffmcfmb*p2p10+2*ffmcfmb*p3p10+2*ffmcfmb*p4p10+ffmcfmb*
     . p5p10+2*p1p10)+32*(p3p7*p1p3)*(ffmcfmb*p6p10-p6p10)+16*p5p7*(
     . ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+16*p4p6*(-ffmcfmb*hbcm2*
     . p7p10+fmc*hbcm*p7p10)+16*p6p7*(ffmcfmb*hbcm2*p2p10-ffmcfmb*
     . hbcm2*p5p10+fmb*hbcm*p1p10-fmc*hbcm*p1p10-fmc*hbcm*p2p10+fmc*
     . hbcm*p5p10+5*hbcm2*p1p10)+16*p3p7*(-ffmcfmb**2*hbcm2*p6p10+
     . ffmcfmb*fmc*hbcm*p6p10+ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+16*
     . p3p6*(-ffmcfmb**2*hbcm2*p7p10-ffmcfmb*fmb*hbcm*p7p10+ffmcfmb*
     . fmc*hbcm*p7p10+ffmcfmb*hbcm2*p7p10+fmb*fmc*p7p10)+32*(ffmcfmb*
     . p1p3*p3p6*p7p10+ffmcfmb*p3p4*p3p6*p7p10-2*p1p10*p2p3*p6p7+2*
     . p1p10*p3p4*p6p7+3*p1p10*p3p5*p6p7+p1p10*p3p6*p4p7+p1p10*p3p6*
     . p5p7+4*p1p10*p3p7*p4p6+2*p1p3*p4p6*p7p10-p1p3*p5p7*p6p10-p1p5*
     . p3p6*p7p10))+(16*p1p3*(p6p12+3*p7p13)+16*p3p6*(-p1p12+p7p10)+
     . 16*p3p7*(ffmcfmb*p3p13+3*p1p13+p2p13-p3p13-p4p13-2*p5p13)+8*(-
     . ffmcfmb*hbcm2*p6p12+fmc*hbcm*p6p12-5*hbcm2*p7p13+6*p2p3*p7p13+
     . 4*p3p11*p6p7+6*p3p13*p4p7-4*p3p4*p7p13-6*p3p5*p7p13))
      ans1=w2*(32*(p6p7*p1p3)*(-3*p1p9-p2p9+2*p5p9)+32*(p3p7*p3p6)*(-
     . 2*ffmcfmb*p2p9+2*ffmcfmb*p3p9+2*ffmcfmb*p4p9+ffmcfmb*p5p9+2*
     . p1p9)+32*(p3p7*p1p3)*(ffmcfmb*p6p9-p6p9)+16*p5p7*(ffmcfmb*
     . hbcm2*p6p9-fmc*hbcm*p6p9)+16*p4p6*(ffmcfmb*hbcm2*p7p9-fmc*hbcm
     . *p7p9)+16*p6p7*(ffmcfmb*hbcm2*p2p9-ffmcfmb*hbcm2*p5p9+fmb*hbcm
     . *p1p9-fmc*hbcm*p1p9-fmc*hbcm*p2p9+fmc*hbcm*p5p9+5*hbcm2*p1p9)+
     . 16*p3p6*(ffmcfmb*fmb*hbcm*p7p9-ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*
     . hbcm2*p7p9)+16*p3p7*(-ffmcfmb**2*hbcm2*p6p9+ffmcfmb*fmc*hbcm*
     . p6p9+ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+32*(ffmcfmb*p2p3*p3p6*
     . p7p9-ffmcfmb*p3p4*p3p6*p7p9-p1p3*p4p6*p7p9-p1p3*p5p7*p6p9-2*
     . p1p9*p2p3*p6p7+2*p1p9*p3p4*p6p7+3*p1p9*p3p5*p6p7+p1p9*p3p6*
     . p4p7+p1p9*p3p6*p5p7+4*p1p9*p3p7*p4p6))+ans2+ans5
      ans=ccc*ans1
      b(30)=ans
      b(32)=w2*ccc*(24*p4p6*(-ffmcfmb*hbcm+fmc)+8*p3p6*(2*ffmcfmb*fmb
     . +2*ffmcfmb*fmc-2*ffmcfmb*hbcm+fmc))
      ans2=w12*(16*p4p7*(ffmcfmb*hbcm*p6p10-fmc*p6p10)+32*p4p6*(
     . ffmcfmb*hbcm*p7p10-fmc*p7p10)+16*p6p7*(-2*ffmcfmb*hbcm*p1p10-
     . ffmcfmb*hbcm*p2p10+2*ffmcfmb*hbcm*p5p10-2*fmb*p1p10+fmc*p2p10-
     . 2*fmc*p5p10+hbcm*p1p10)+16*p3p6*(-ffmcfmb*fmb*p7p10-ffmcfmb*
     . fmc*p7p10+ffmcfmb*hbcm*p7p10-fmc*p7p10)+16*p3p7*(ffmcfmb**2*
     . hbcm*p6p10-ffmcfmb*fmc*p6p10))+8*(ffmcfmb*hbcm*p6p12-fmb*p7p13
     . -fmc*p6p12-fmc*p7p13)
      ans1=w2*(16*p4p7*(ffmcfmb*hbcm*p6p9-fmc*p6p9)+32*p4p6*(ffmcfmb*
     . hbcm*p7p9-fmc*p7p9)+16*p6p7*(-ffmcfmb*hbcm*p2p9+2*ffmcfmb*hbcm
     . *p5p9+fmb*p1p9+fmc*p1p9+fmc*p2p9-2*fmc*p5p9-hbcm*p1p9)+16*p3p6
     . *(-ffmcfmb*fmb*p7p9-ffmcfmb*fmc*p7p9+ffmcfmb*hbcm*p7p9-fmc*
     . p7p9)+16*p3p7*(ffmcfmb**2*hbcm*p6p9-ffmcfmb*fmc*p6p9))+w17*(16
     . *p4p7*(ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p11-fmc*p6p10-fmc*
     . p6p11)+32*p4p6*(ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p11-fmc*
     . p7p10-fmc*p7p11)+16*p6p7*(-2*ffmcfmb*hbcm*p1p10-2*ffmcfmb*hbcm
     . *p1p11-ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p2p11+2*ffmcfmb*hbcm*
     . p5p10+2*ffmcfmb*hbcm*p5p11-2*fmb*p1p10-2*fmb*p1p11+fmc*p2p10+
     . fmc*p2p11-2*fmc*p5p10-2*fmc*p5p11+hbcm*p1p10+hbcm*p1p11)+16*
     . p3p6*(-ffmcfmb*fmb*p7p10-ffmcfmb*fmb*p7p11-ffmcfmb*fmc*p7p10-
     . ffmcfmb*fmc*p7p11+ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p11-fmc*
     . p7p10-fmc*p7p11)+16*p3p7*(ffmcfmb**2*hbcm*p6p10+ffmcfmb**2*
     . hbcm*p6p11-ffmcfmb*fmc*p6p10-ffmcfmb*fmc*p6p11))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(34)=4*ccc*w2*(3*ffmcfmb*hbcm+4*fmb+fmc-3*hbcm)
      b(35)=ccc*(8*w2*(-2*ffmcfmb*hbcm*p7p9-3*fmb*p7p9-fmc*p7p9+2*
     . hbcm*p7p9)+8*w17*(-2*ffmcfmb*hbcm*p7p10-2*ffmcfmb*hbcm*p7p11-3
     . *fmb*p7p10-3*fmb*p7p11-fmc*p7p10-fmc*p7p11+2*hbcm*p7p10+2*hbcm
     . *p7p11)+8*w12*(-2*ffmcfmb*hbcm*p7p10-3*fmb*p7p10-fmc*p7p10+2*
     . hbcm*p7p10))
      b(36)=4*ccc*w2*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+6*
     . p1p3+6*p2p3-8*p3p4)
      b(37)=w2*ccc*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)
      b(38)=ccc*(w2*(16*p3p7*(-p1p9-p2p9+3*p5p9)+8*(ffmcfmb*hbcm2*
     . p7p9-2*fmb*hbcm*p7p9+fmc*hbcm*p7p9+3*hbcm2*p7p9-4*p1p3*p7p9-4*
     . p2p3*p7p9+6*p3p4*p7p9))+w17*(48*p3p4*(p7p10+p7p11)+32*p2p3*(-
     . p7p10-p7p11)+32*p1p3*(-p7p10-p7p11)+16*p3p7*(-p1p10-p1p11-
     . p2p10-p2p11+3*p5p10+3*p5p11)+8*(ffmcfmb*hbcm2*p7p10+ffmcfmb*
     . hbcm2*p7p11-2*fmb*hbcm*p7p10-2*fmb*hbcm*p7p11+fmc*hbcm*p7p10+
     . fmc*hbcm*p7p11+3*hbcm2*p7p10+3*hbcm2*p7p11))+w12*(16*p3p7*(-
     . p1p10-p2p10+3*p5p10)+8*(ffmcfmb*hbcm2*p7p10-2*fmb*hbcm*p7p10+
     . fmc*hbcm*p7p10+3*hbcm2*p7p10-4*p1p3*p7p10-4*p2p3*p7p10+6*p3p4*
     . p7p10)))
      b(39)=w17*ccc*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+16*(p1p3*p6p7+p2p3*p6p7-3*
     . p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))
      b(40)=w17*ccc*(32*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6
     . )*(2*ffmcfmb*hbcm-fmc)+32*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(
     . p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+32*(p6p7*p1p3)*(ffmcfmb*hbcm+fmb
     . )+16*(p3p7*p3p6)*(ffmcfmb*fmb+2*ffmcfmb*fmc)+8*p6p7*(-ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm+2*fmc
     . *hbcm2+fmc2*hbcm)+16*(-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7))
      b(41)=w17*ccc*((p6p7*p1p3)*(-16*ffmcfmb+32)+8*p6p7*(ffmcfmb**2*
     . hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*
     . fmb*fmc+fmc*hbcm+fmc2)+16*(ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*
     . p5p7-p1p2*p6p7+3*p1p5*p6p7))
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+16*(p1p3*p6p7+p2p3*p6p7-3*
     . p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))
      b(43)=ccc*(w2*(16*p6p7*(p1p9+p2p9-3*p5p9)+16*p3p6*(-ffmcfmb*
     . p7p9-p7p9)+16*(-ffmcfmb*p3p7*p6p9-3*p4p6*p7p9-p4p7*p6p9))+w17*
     . (16*p4p7*(-p6p10-p6p11)+48*p4p6*(-p7p10-p7p11)+16*p6p7*(p1p10+
     . p1p11+p2p10+p2p11-3*p5p10-3*p5p11)+16*p3p7*(-ffmcfmb*p6p10-
     . ffmcfmb*p6p11)+16*p3p6*(-ffmcfmb*p7p10-ffmcfmb*p7p11-p7p10-
     . p7p11))+w12*(16*p6p7*(p1p10+p2p10-3*p5p10)+16*p3p6*(-ffmcfmb*
     . p7p10-p7p10)+16*(-ffmcfmb*p3p7*p6p10-3*p4p6*p7p10-p4p7*p6p10))
     . +8*(-p6p12-p7p13))
      b(44)=w17*ccc*(16*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb
     . **2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*ffmcfmb*
     . hbcm2-fmb*fmc)+16*(-ffmcfmb*p1p3*p3p6+ffmcfmb*p2p3*p3p6-2*
     . ffmcfmb*p3p4*p3p6-3*p1p3*p4p6+p1p5*p3p6))
      b(45)=w17*ccc*(p3p6*(-16*ffmcfmb-8)-32*p4p6)
      b(46)=w17*ccc*(24*p4p6*(-ffmcfmb*hbcm+fmc)+8*p3p6*(2*ffmcfmb*
     . fmb+2*ffmcfmb*fmc-2*ffmcfmb*hbcm+fmc))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb-8)-32*p4p6)
      b(49)=w17*ccc*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)
      b(50)=w17*ccc*(p3p7*(-16*ffmcfmb+24)+32*p5p7)
      b(51)=w17*ccc*(16*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*fmb*
     . fmc-fmc2)+16*(2*ffmcfmb*p1p3*p3p7-2*p1p2*p3p7-p1p3*p5p7+3*p1p4
     . *p3p7))
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+24)+32*p5p7)
      b(53)=ccc*(8*w2*(-3*p1p9-3*p2p9+4*p3p9+4*p4p9)+8*w17*(-3*p1p10-
     . 3*p1p11-3*p2p10-3*p2p11+4*p3p10+4*p3p11+4*p4p10+4*p4p11)+8*w12
     . *(-3*p1p10-3*p2p10+4*p3p10+4*p4p10))
      b(54)=4*ccc*w17*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+6
     . *p1p3+6*p2p3-8*p3p4)
      b(55)=4*ccc*w17*(3*ffmcfmb*hbcm+4*fmb+fmc-3*hbcm)
      b(56)=w17*ccc*(p1p3*(16*ffmcfmb+8)+4*(-2*ffmcfmb**2*hbcm2-3*
     . ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+3*fmb*fmc-
     . 2*fmc*hbcm-fmc2-6*p1p2+8*p1p4))
      b(57)=-8*ccc
      b(58)=w17*ccc*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)
      b(59)=w17*ccc*(24*p3p4*(ffmcfmb*hbcm-fmc)+16*p2p3*(-ffmcfmb*
     . hbcm+fmc)+8*p1p3*(-4*ffmcfmb*hbcm-fmb+2*fmc)+4*(ffmcfmb**2*
     . hbcm3-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+
     . 2*fmb*fmc*hbcm-3*fmc*hbcm2+fmc2*hbcm+4*hbcm*p1p2-6*hbcm*p1p4))
      b(60)=w2*ccc*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)
      b(61)=-4*ccc*hbcm
      b(62)=8*ccc*ffmcfmb*hbcm*p3p6
      b(63)=ccc*(w17*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+16*(p1p3*p6p7+p2p3*p6p7-3*
     . p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))+w12*((p3p7*p3p6)*(-32*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)
     . +16*(p1p3*p6p7+p2p3*p6p7-3*p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6)))
      b(64)=ccc*(w17*((p6p7*p1p3)*(-16*ffmcfmb+32)+8*p6p7*(ffmcfmb**2
     . *hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*
     . fmb*fmc+fmc*hbcm+fmc2)+16*(ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*
     . p5p7-p1p2*p6p7+3*p1p5*p6p7))+w12*((p6p7*p1p3)*(-16*ffmcfmb+32)
     . +8*p6p7*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2-2*fmb*fmc+fmc*hbcm+fmc2)+16*(ffmcfmb*p3p6*
     . p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*p6p7+3*p1p5*p6p7))-8*p6p7)
      b(66)=ccc*(w2*(16*p6p7*(hbcm*p1p9+hbcm*p2p9-2*hbcm*p5p9)+16*
     . p3p6*(-ffmcfmb*hbcm*p7p9-fmb*p7p9)+16*(-ffmcfmb*hbcm*p3p7*p6p9
     . -2*hbcm*p4p6*p7p9-hbcm*p4p7*p6p9))+w17*(16*p4p7*(-hbcm*p6p10-
     . hbcm*p6p11)+32*p4p6*(-hbcm*p7p10-hbcm*p7p11)+16*p6p7*(hbcm*
     . p1p10+hbcm*p1p11+hbcm*p2p10+hbcm*p2p11-2*hbcm*p5p10-2*hbcm*
     . p5p11)+16*p3p7*(-ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p11)+16*
     . p3p6*(-ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p11-fmb*p7p10-fmb*
     . p7p11))+w12*(16*p6p7*(hbcm*p1p10+hbcm*p2p10-2*hbcm*p5p10)+16*
     . p3p6*(-ffmcfmb*hbcm*p7p10-fmb*p7p10)+16*(-ffmcfmb*hbcm*p3p7*
     . p6p10-2*hbcm*p4p6*p7p10-hbcm*p4p7*p6p10))-8*hbcm*p6p12)
      b(67)=ccc*(w17*(p3p6*(-16*ffmcfmb-8)-32*p4p6)+w12*(p3p6*(-16*
     . ffmcfmb-8)-32*p4p6))
      b(69)=ccc*(w17*(p3p7*(-16*ffmcfmb+24)+32*p5p7)+w12*(p3p7*(-16*
     . ffmcfmb+24)+32*p5p7))
      b(70)=ccc*(w17*(16*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*fmb*
     . fmc-fmc2)+16*(2*ffmcfmb*p1p3*p3p7-2*p1p2*p3p7-p1p3*p5p7+3*p1p4
     . *p3p7))+w12*(16*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*fmb*
     . fmc-fmc2)+16*(2*ffmcfmb*p1p3*p3p7-2*p1p2*p3p7-p1p3*p5p7+3*p1p4
     . *p3p7))+8*p3p7)
      b(72)=ccc*(w17*(p1p3*(16*ffmcfmb+8)+4*(-2*ffmcfmb**2*hbcm2-3*
     . ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+3*fmb*fmc-
     . 2*fmc*hbcm-fmc2-6*p1p2+8*p1p4))+w12*(p1p3*(16*ffmcfmb+8)+4*(-2
     . *ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+2*
     . ffmcfmb*hbcm2+3*fmb*fmc-2*fmc*hbcm-fmc2-6*p1p2+8*p1p4))+8)
      b(73)=8*ccc*w17*p6p7*(2*ffmcfmb*hbcm+3*fmb+fmc-2*hbcm)
      b(75)=ccc*(8*w2*(3*ffmcfmb*hbcm*p1p9+2*ffmcfmb*hbcm*p2p9-3*
     . ffmcfmb*hbcm*p3p9-3*ffmcfmb*hbcm*p4p9+fmb*p1p9-2*fmc*p1p9-2*
     . fmc*p2p9+3*fmc*p3p9+3*fmc*p4p9-hbcm*p1p9)+8*w17*(3*ffmcfmb*
     . hbcm*p1p10+3*ffmcfmb*hbcm*p1p11+2*ffmcfmb*hbcm*p2p10+2*ffmcfmb
     . *hbcm*p2p11-3*ffmcfmb*hbcm*p3p10-3*ffmcfmb*hbcm*p3p11-3*
     . ffmcfmb*hbcm*p4p10-3*ffmcfmb*hbcm*p4p11+fmb*p1p10+fmb*p1p11-2*
     . fmc*p1p10-2*fmc*p1p11-2*fmc*p2p10-2*fmc*p2p11+3*fmc*p3p10+3*
     . fmc*p3p11+3*fmc*p4p10+3*fmc*p4p11-hbcm*p1p10-hbcm*p1p11)+8*w12
     . *(3*ffmcfmb*hbcm*p1p10+2*ffmcfmb*hbcm*p2p10-3*ffmcfmb*hbcm*
     . p3p10-3*ffmcfmb*hbcm*p4p10+fmb*p1p10-2*fmc*p1p10-2*fmc*p2p10+3
     . *fmc*p3p10+3*fmc*p4p10-hbcm*p1p10))
      ans=ccc*(w2*(16*p1p3*(p1p9+p2p9-p3p9-p4p9)+8*(2*ffmcfmb*hbcm2*
     . p1p9+ffmcfmb*hbcm2*p2p9-2*ffmcfmb*hbcm2*p3p9-2*ffmcfmb*hbcm2*
     . p4p9-fmb*hbcm*p1p9-fmc*hbcm*p1p9-fmc*hbcm*p2p9+2*fmc*hbcm*p3p9
     . +2*fmc*hbcm*p4p9-hbcm2*p1p9-2*p1p9*p3p5))+w17*(16*p3p5*(-p1p10
     . -p1p11)+16*p1p3*(p1p10+p1p11+p2p10+p2p11-p3p10-p3p11-p4p10-
     . p4p11)+8*(2*ffmcfmb*hbcm2*p1p10+2*ffmcfmb*hbcm2*p1p11+ffmcfmb*
     . hbcm2*p2p10+ffmcfmb*hbcm2*p2p11-2*ffmcfmb*hbcm2*p3p10-2*
     . ffmcfmb*hbcm2*p3p11-2*ffmcfmb*hbcm2*p4p10-2*ffmcfmb*hbcm2*
     . p4p11-fmb*hbcm*p1p10-fmb*hbcm*p1p11-fmc*hbcm*p1p10-fmc*hbcm*
     . p1p11-fmc*hbcm*p2p10-fmc*hbcm*p2p11+2*fmc*hbcm*p3p10+2*fmc*
     . hbcm*p3p11+2*fmc*hbcm*p4p10+2*fmc*hbcm*p4p11-hbcm2*p1p10-hbcm2
     . *p1p11))+w12*(16*p1p3*(p1p10+p2p10-p3p10-p4p10)+8*(2*ffmcfmb*
     . hbcm2*p1p10+ffmcfmb*hbcm2*p2p10-2*ffmcfmb*hbcm2*p3p10-2*
     . ffmcfmb*hbcm2*p4p10-fmb*hbcm*p1p10-fmc*hbcm*p1p10-fmc*hbcm*
     . p2p10+2*fmc*hbcm*p3p10+2*fmc*hbcm*p4p10-hbcm2*p1p10-2*p1p10*
     . p3p5)))
      b(76)=ans
      b(77)=ccc*(8*w2*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9-hbcm2*p6p9-2*
     . p3p5*p6p9+2*p3p6*p5p9)+w17*(16*p3p6*(p5p10+p5p11)+16*p3p5*(-
     . p6p10-p6p11)+8*(ffmcfmb*hbcm2*p6p10+ffmcfmb*hbcm2*p6p11-fmb*
     . hbcm*p6p10-fmb*hbcm*p6p11-hbcm2*p6p10-hbcm2*p6p11))+8*w12*(
     . ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10-hbcm2*p6p10-2*p3p5*p6p10+2*
     . p3p6*p5p10)-24*p3p13)
      b(78)=ccc*(8*w2*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+8*w17*(
     . ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p11+fmb*p6p10+fmb*p6p11-hbcm
     . *p6p10-hbcm*p6p11)+8*w12*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*
     . p6p10))
      b(79)=8*ccc*w2*p6p7*(2*ffmcfmb*hbcm+3*fmb+fmc-2*hbcm)
      b(80)=w2*ccc*(24*p5p7*(-ffmcfmb*hbcm+fmc)+8*p3p7*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+2*fmc))
      b(81)=ccc*(8*w17*p6p7*(2*ffmcfmb*hbcm+3*fmb+fmc-2*hbcm)+8*w12*
     . p6p7*(2*ffmcfmb*hbcm+3*fmb+fmc-2*hbcm))
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc-2*hbcm)+8*(-2*hbcm*p4p7
     . -hbcm*p5p7))
      b(84)=8*ccc*(-2*p3p7-3*p4p7-p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm+4*hbcm2-6*p1p3-4*p2p3+2*p3p4+6*
     . p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(2*fmb+2*fmc-hbcm)
      b(88)=ccc*(w17*(24*p5p7*(-ffmcfmb*hbcm+fmc)+8*p3p7*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+2*fmc))+w12*(24*p5p7*(-ffmcfmb
     . *hbcm+fmc)+8*p3p7*(ffmcfmb**2*hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+
     . 2*fmc)))
      b(89)=-4*ccc*hbcm
      b(90)=w17*ccc*(24*p5p7*(-ffmcfmb*hbcm+fmc)+8*p3p7*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+2*fmc))
      b(92)=8*ccc*(hbcm2-p2p3+p3p4+p3p5)
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(-ffmcfmb*hbcm+fmc)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp16_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(2)=w2*ccc*(8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(ffmcfmb*p2p3*
     . p6p7+ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*p6p7
     . +p1p5*p6p7))
      b(3)=ccc*(8*w2*(-ffmcfmb**2*hbcm2*p7p9-2*ffmcfmb*p1p9*p3p7+2*
     . ffmcfmb*p2p3*p7p9+fmc2*p7p9+2*p1p2*p7p9-2*p1p3*p7p9-6*p1p4*
     . p7p9-2*p1p9*p4p7)+8*w7*(-ffmcfmb**2*hbcm2*p7p11-2*ffmcfmb*
     . p1p11*p3p7+2*ffmcfmb*p2p3*p7p11+fmc2*p7p11-2*p1p11*p4p7+2*p1p2
     . *p7p11-2*p1p3*p7p11-6*p1p4*p7p11)+w17*(48*p1p4*(-p7p10-p7p11)+
     . 16*p1p3*(-p7p10-p7p11)+16*p1p2*(p7p10+p7p11)+16*p4p7*(-p1p10-
     . p1p11)+16*p2p3*(ffmcfmb*p7p10+ffmcfmb*p7p11)+16*p3p7*(-ffmcfmb
     . *p1p10-ffmcfmb*p1p11)+8*(-ffmcfmb**2*hbcm2*p7p10-ffmcfmb**2*
     . hbcm2*p7p11+fmc2*p7p10+fmc2*p7p11))+8*(-p1p12+p7p11))
      b(5)=w2*ccc*(8*p3p7*(ffmcfmb**2*hbcm2-fmc2)+16*(ffmcfmb*p1p3*
     . p3p7-ffmcfmb*p2p3*p3p7-p1p2*p3p7-p1p3*p5p7+3*p1p4*p3p7))
      b(6)=ccc*(w2*(16*p1p3*(ffmcfmb*hbcm*p7p9+fmb*p7p9)+16*(-fmb*
     . p1p9*p3p7-hbcm*p1p2*p7p9+2*hbcm*p1p4*p7p9-hbcm*p1p9*p5p7))+w7*
     . (16*p1p3*(ffmcfmb*hbcm*p7p11+fmb*p7p11)+16*(-fmb*p1p11*p3p7-
     . hbcm*p1p11*p5p7-hbcm*p1p2*p7p11+2*hbcm*p1p4*p7p11))+w17*(32*
     . p1p4*(hbcm*p7p10+hbcm*p7p11)+16*p1p2*(-hbcm*p7p10-hbcm*p7p11)+
     . 16*p5p7*(-hbcm*p1p10-hbcm*p1p11)+16*p3p7*(-fmb*p1p10-fmb*p1p11
     . )+16*p1p3*(ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p11+fmb*p7p10+fmb
     . *p7p11))+8*hbcm*p7p10)
      ans=ccc*(w2*(16*p3p6*(ffmcfmb*p2p9-2*ffmcfmb*p3p9-2*ffmcfmb*
     . p4p9-p1p9)+8*(-ffmcfmb**2*hbcm2*p6p9+2*ffmcfmb*p2p3*p6p9+fmc2*
     . p6p9-2*p1p2*p6p9+2*p1p3*p6p9+2*p1p5*p6p9-6*p1p9*p4p6))+w7*(16*
     . p3p6*(ffmcfmb*p2p11-2*ffmcfmb*p3p11-2*ffmcfmb*p4p11-p1p11)+8*(
     . -ffmcfmb**2*hbcm2*p6p11+2*ffmcfmb*p2p3*p6p11+fmc2*p6p11-6*
     . p1p11*p4p6-2*p1p2*p6p11+2*p1p3*p6p11+2*p1p5*p6p11))+w17*(16*
     . p1p5*(p6p10+p6p11)+16*p1p3*(p6p10+p6p11)+16*p1p2*(-p6p10-p6p11
     . )+48*p4p6*(-p1p10-p1p11)+16*p2p3*(ffmcfmb*p6p10+ffmcfmb*p6p11)
     . +16*p3p6*(ffmcfmb*p2p10+ffmcfmb*p2p11-2*ffmcfmb*p3p10-2*
     . ffmcfmb*p3p11-2*ffmcfmb*p4p10-2*ffmcfmb*p4p11-p1p10-p1p11)+8*(
     . -ffmcfmb**2*hbcm2*p6p10-ffmcfmb**2*hbcm2*p6p11+fmc2*p6p10+fmc2
     . *p6p11))+8*(-p1p13+p2p13-2*p3p13-2*p4p13+p6p10+p6p11))
      b(7)=ans
      b(9)=w2*ccc*(8*p3p6*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm+2*ffmcfmb*hbcm2-fmc2)+16*(-2*ffmcfmb*p2p3*p3p6+2*
     . ffmcfmb*p3p4*p3p6+p1p2*p3p6+3*p1p3*p4p6-p1p5*p3p6))
      ans1=(w2*(16*p3p6*(-ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*p3p9+ffmcfmb
     . *hbcm*p4p9)+16*p1p3*(-ffmcfmb*hbcm*p6p9-fmb*p6p9)+8*(ffmcfmb**
     . 2*hbcm3*p6p9-2*ffmcfmb*hbcm*p2p3*p6p9-fmc2*hbcm*p6p9+2*hbcm*
     . p1p2*p6p9-2*hbcm*p1p5*p6p9-2*hbcm*p1p9*p4p6))+w7*(16*p1p3*(-
     . ffmcfmb*hbcm*p6p11-fmb*p6p11)+16*p3p6*(-ffmcfmb*hbcm*p1p11+
     . ffmcfmb*hbcm*p3p11+ffmcfmb*hbcm*p4p11)+8*(ffmcfmb**2*hbcm3*
     . p6p11-2*ffmcfmb*hbcm*p2p3*p6p11-fmc2*hbcm*p6p11-2*hbcm*p1p11*
     . p4p6+2*hbcm*p1p2*p6p11-2*hbcm*p1p5*p6p11))+w17*(16*p1p5*(-hbcm
     . *p6p10-hbcm*p6p11)+16*p1p2*(hbcm*p6p10+hbcm*p6p11)+16*p4p6*(-
     . hbcm*p1p10-hbcm*p1p11)+16*p2p3*(-ffmcfmb*hbcm*p6p10-ffmcfmb*
     . hbcm*p6p11)+16*p1p3*(-ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p11-
     . fmb*p6p10-fmb*p6p11)+16*p3p6*(ffmcfmb**2*hbcm*p3p10+2*ffmcfmb*
     . fmb*p3p10+ffmcfmb*fmc*p3p10-ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*
     . p1p11+ffmcfmb*hbcm*p3p11+ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*p4p11
     . )+8*(ffmcfmb**2*hbcm3*p6p10+ffmcfmb**2*hbcm3*p6p11-fmc2*hbcm*
     . p6p10-fmc2*hbcm*p6p11))+8*(hbcm*p1p13+hbcm*p3p13+hbcm*p4p13-
     . hbcm*p6p10-hbcm*p6p11))
      ans=ccc*ans1
      b(10)=ans
      b(12)=w2*ccc*(8*p6p7*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p3p7*
     . p3p6)*(ffmcfmb**2*hbcm+ffmcfmb*fmb+ffmcfmb*fmc)+16*(ffmcfmb*
     . hbcm*p2p3*p6p7+ffmcfmb*hbcm*p3p6*p5p7-hbcm*p1p2*p6p7+hbcm*p1p3
     . *p6p7+hbcm*p1p5*p6p7))
      b(14)=8*ccc*w2*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2-p1p2+p1p3+
     . 4*p1p4)
      b(16)=w2*ccc*(8*p1p3*(-ffmcfmb*hbcm-fmb)+4*(-ffmcfmb**2*hbcm3+2
     . *ffmcfmb*hbcm*p2p3+fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4))
      b(17)=w17*ccc*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)
      b(18)=4*ccc*w17*(-3*ffmcfmb*hbcm-4*fmb-fmc+3*hbcm)
      b(19)=4*ccc*w17*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+6
     . *p1p3+6*p2p3-8*p3p4)
      b(20)=w17*ccc*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)
      b(21)=ccc*(8*w2*(-2*hbcm*p1p9-2*hbcm*p2p9+3*hbcm*p3p9+3*hbcm*
     . p4p9)+8*w7*(-2*hbcm*p1p11-2*hbcm*p2p11+3*hbcm*p3p11+3*hbcm*
     . p4p11)+8*w17*(3*ffmcfmb*hbcm*p3p10+4*fmb*p3p10+fmc*p3p10-2*
     . hbcm*p1p10-2*hbcm*p1p11-2*hbcm*p2p10-2*hbcm*p2p11+3*hbcm*p3p11
     . +3*hbcm*p4p10+3*hbcm*p4p11))
      b(22)=w17*ccc*(8*p1p3*(-ffmcfmb*hbcm-fmb)+4*(-ffmcfmb**2*hbcm3+
     . 2*ffmcfmb*hbcm*p2p3+fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4))
      b(27)=8*ccc*w17*p3p6*(-ffmcfmb**2*hbcm-2*ffmcfmb*fmb-ffmcfmb*
     . fmc+ffmcfmb*hbcm)
      b(28)=w17*ccc*(8*p3p6*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmc2)+16*(2*ffmcfmb*p2p3*p3p6
     . -2*ffmcfmb*p3p4*p3p6-p1p2*p3p6-3*p1p3*p4p6+p1p5*p3p6))
      b(29)=w17*ccc*(16*(p6p7*p1p3)*(ffmcfmb*hbcm+fmb)+8*p6p7*(-
     . ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+ffmcfmb*fmc)+16*(ffmcfmb*hbcm*p2p3*p6p7+ffmcfmb*
     . hbcm*p3p6*p5p7-hbcm*p1p2*p6p7+hbcm*p1p5*p6p7))
      ans3=w2*(32*(p6p7*p1p3)*(-p1p9+p5p9)+32*(p3p7*p3p6)*(-2*ffmcfmb
     . *p2p9+2*ffmcfmb*p3p9+2*ffmcfmb*p4p9+ffmcfmb*p5p9+2*p1p9)+32*(
     . p3p7*p1p3)*(ffmcfmb*p6p9-p6p9)+16*p3p6*(-ffmcfmb**2*hbcm2*p7p9
     . +ffmcfmb*fmb*hbcm*p7p9-ffmcfmb*hbcm2*p7p9)+32*(ffmcfmb*p2p3*
     . p3p6*p7p9-ffmcfmb*p3p4*p3p6*p7p9+hbcm2*p1p9*p6p7-p1p3*p4p6*
     . p7p9-p1p3*p5p7*p6p9-p1p9*p2p3*p6p7+p1p9*p3p4*p6p7+p1p9*p3p5*
     . p6p7+p1p9*p3p6*p4p7+p1p9*p3p6*p5p7+4*p1p9*p3p7*p4p6))+(16*p3p7
     . *(-2*p2p13+2*p3p13+2*p4p13+p5p13)+16*p3p6*(ffmcfmb*p3p12-p7p10
     . -p7p11)+8*(-ffmcfmb*hbcm2*p7p13+fmc*hbcm*p7p13+hbcm2*p7p13+2*
     . p1p3*p7p13+2*p3p10*p6p7+2*p3p4*p7p13))
      ans2=w17*(32*(p3p6*p1p5)*(-p7p10-p7p11)+32*(p5p7*p1p3)*(-p6p10-
     . p6p11)+64*(p4p6*p1p3)*(p7p10+p7p11)+32*(p3p6*p1p2)*(p7p10+
     . p7p11)+128*(p4p6*p3p7)*(p1p10+p1p11)+32*(p5p7*p3p6)*(p1p10+
     . p1p11)+32*(p4p7*p3p6)*(p1p10+p1p11)+32*(p6p7*p3p5)*(p1p10+
     . p1p11)+32*(p6p7*p3p4)*(p1p10+p1p11)+32*(p6p7*p2p3)*(-p1p10-
     . p1p11)+32*(p6p7*p1p3)*(-p1p10-p1p11+p5p10+p5p11)+32*p6p7*(
     . hbcm2*p1p10+hbcm2*p1p11)+32*(p3p6*p3p4)*(ffmcfmb*p7p10+ffmcfmb
     . *p7p11)+32*(p3p6*p2p3)*(-ffmcfmb*p7p10-ffmcfmb*p7p11)+32*(p3p7
     . *p3p6)*(-2*ffmcfmb*p2p10-2*ffmcfmb*p2p11+2*ffmcfmb*p3p10+2*
     . ffmcfmb*p3p11+2*ffmcfmb*p4p10+2*ffmcfmb*p4p11+ffmcfmb*p5p10+
     . ffmcfmb*p5p11+2*p1p10+2*p1p11)+32*(p3p7*p1p3)*(ffmcfmb*p6p10+
     . ffmcfmb*p6p11-p6p10-p6p11)+16*p3p6*(ffmcfmb*fmc*hbcm*p7p10+
     . ffmcfmb*fmc*hbcm*p7p11+ffmcfmb*hbcm2*p7p10+ffmcfmb*hbcm2*p7p11
     . -fmc2*p7p10-fmc2*p7p11))+ans3
      ans1=w7*(32*(p6p7*p1p3)*(-p1p11+p5p11)+32*(p3p7*p3p6)*(-2*
     . ffmcfmb*p2p11+2*ffmcfmb*p3p11+2*ffmcfmb*p4p11+ffmcfmb*p5p11+2*
     . p1p11)+32*(p3p7*p1p3)*(ffmcfmb*p6p11-p6p11)+16*p3p6*(ffmcfmb*
     . fmc*hbcm*p7p11+ffmcfmb*hbcm2*p7p11-fmc2*p7p11)+32*(-ffmcfmb*
     . p2p3*p3p6*p7p11+ffmcfmb*p3p4*p3p6*p7p11+hbcm2*p1p11*p6p7-p1p11
     . *p2p3*p6p7+p1p11*p3p4*p6p7+p1p11*p3p5*p6p7+p1p11*p3p6*p4p7+
     . p1p11*p3p6*p5p7+4*p1p11*p3p7*p4p6+p1p2*p3p6*p7p11+2*p1p3*p4p6*
     . p7p11-p1p3*p5p7*p6p11-p1p5*p3p6*p7p11))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(32)=8*ccc*w2*p3p6*(ffmcfmb**2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-
     . ffmcfmb*hbcm)
      b(33)=ccc*(16*w2*p3p6*(-ffmcfmb*fmb*p7p9-ffmcfmb*fmc*p7p9)+w7*(
     . 16*p6p7*(-ffmcfmb*hbcm*p1p11-fmb*p1p11+hbcm*p1p11)+16*p3p6*(-
     . ffmcfmb*fmb*p7p11-ffmcfmb*fmc*p7p11))+w17*(16*p6p7*(-ffmcfmb*
     . hbcm*p1p10-ffmcfmb*hbcm*p1p11-fmb*p1p10-fmb*p1p11+hbcm*p1p10+
     . hbcm*p1p11)+16*p3p6*(-ffmcfmb*fmb*p7p10-ffmcfmb*fmb*p7p11-
     . ffmcfmb*fmc*p7p10-ffmcfmb*fmc*p7p11))+8*(-fmb*p7p13-fmc*p7p13)
     . )
      b(34)=4*ccc*w2*(3*ffmcfmb*hbcm+4*fmb+fmc-3*hbcm)
      b(35)=ccc*(8*w2*(-2*ffmcfmb*hbcm*p7p9-3*fmb*p7p9-fmc*p7p9+2*
     . hbcm*p7p9)+8*w7*(-2*ffmcfmb*hbcm*p7p11-3*fmb*p7p11-fmc*p7p11+2
     . *hbcm*p7p11)+8*w17*(-2*ffmcfmb*hbcm*p7p10-2*ffmcfmb*hbcm*p7p11
     . -3*fmb*p7p10-3*fmb*p7p11-fmc*p7p10-fmc*p7p11+2*hbcm*p7p10+2*
     . hbcm*p7p11))
      b(36)=4*ccc*w2*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+6*
     . p1p3+6*p2p3-8*p3p4)
      b(37)=w2*ccc*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)
      b(38)=ccc*(w2*(16*p3p7*(-p1p9-p2p9+3*p5p9)+8*(ffmcfmb*hbcm2*
     . p7p9-2*fmb*hbcm*p7p9+fmc*hbcm*p7p9+3*hbcm2*p7p9-4*p1p3*p7p9-4*
     . p2p3*p7p9+6*p3p4*p7p9))+w7*(16*p3p7*(-p1p11-p2p11+3*p5p11)+8*(
     . ffmcfmb*hbcm2*p7p11-2*fmb*hbcm*p7p11+fmc*hbcm*p7p11+3*hbcm2*
     . p7p11-4*p1p3*p7p11-4*p2p3*p7p11+6*p3p4*p7p11))+w17*(48*p3p4*(
     . p7p10+p7p11)+32*p2p3*(-p7p10-p7p11)+32*p1p3*(-p7p10-p7p11)+16*
     . p3p7*(-p1p10-p1p11-p2p10-p2p11+3*p5p10+3*p5p11)+8*(ffmcfmb*
     . hbcm2*p7p10+ffmcfmb*hbcm2*p7p11-2*fmb*hbcm*p7p10-2*fmb*hbcm*
     . p7p11+fmc*hbcm*p7p10+fmc*hbcm*p7p11+3*hbcm2*p7p10+3*hbcm2*
     . p7p11))+24*p3p12)
      b(39)=ccc*(w7*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+16*(-p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))+
     . w17*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*hbcm2-fmb*
     . hbcm-hbcm2)+16*(-p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6)))
      b(40)=ccc*(w7*(16*(p6p7*p1p3)*(ffmcfmb*hbcm+fmb)+8*p6p7*(-
     . ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+ffmcfmb*fmc)+16*(ffmcfmb*hbcm*p2p3*p6p7+ffmcfmb*
     . hbcm*p3p6*p5p7-hbcm*p1p2*p6p7+hbcm*p1p5*p6p7))+w17*(16*(p6p7*
     . p1p3)*(ffmcfmb*hbcm+fmb)+8*p6p7*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+
     . 16*(p3p7*p3p6)*(ffmcfmb**2*hbcm+ffmcfmb*fmb+ffmcfmb*fmc)+16*(
     . ffmcfmb*hbcm*p2p3*p6p7+ffmcfmb*hbcm*p3p6*p5p7-hbcm*p1p2*p6p7+
     . hbcm*p1p5*p6p7))+8*hbcm*p6p7)
      b(41)=ccc*(w7*(8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(ffmcfmb*p2p3
     . *p6p7+ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*
     . p6p7+p1p5*p6p7))+w17*(8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(
     . ffmcfmb*p2p3*p6p7+ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*
     . p6p7+p1p3*p6p7+p1p5*p6p7))+8*p6p7)
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+16*(-p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))
      b(43)=ccc*(w2*(16*p3p6*(-ffmcfmb*p7p9-p7p9)+16*(-ffmcfmb*p3p7*
     . p6p9-3*p4p6*p7p9-p4p7*p6p9-p5p9*p6p7))+w7*(16*p3p6*(-ffmcfmb*
     . p7p11-p7p11)+16*(-ffmcfmb*p3p7*p6p11-3*p4p6*p7p11-p4p7*p6p11-
     . p5p11*p6p7))+w17*(16*p6p7*(-p5p10-p5p11)+16*p4p7*(-p6p10-p6p11
     . )+48*p4p6*(-p7p10-p7p11)+16*p3p7*(-ffmcfmb*p6p10-ffmcfmb*p6p11
     . )+16*p3p6*(-ffmcfmb*p7p10-ffmcfmb*p7p11-p7p10-p7p11))+8*(-
     . p6p12-p7p13))
      b(44)=ccc*(w7*(8*p3p6*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmc2)+16*(2*ffmcfmb*p2p3*p3p6
     . -2*ffmcfmb*p3p4*p3p6-p1p2*p3p6-3*p1p3*p4p6+p1p5*p3p6))+w17*(8*
     . p3p6*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-2*
     . ffmcfmb*hbcm2+fmc2)+16*(2*ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*
     . p3p6-p1p2*p3p6-3*p1p3*p4p6+p1p5*p3p6))+8*p3p6)
      b(45)=ccc*(w7*(p3p6*(-16*ffmcfmb-8)-32*p4p6)+w17*(p3p6*(-16*
     . ffmcfmb-8)-32*p4p6))
      b(46)=ccc*(8*w7*p3p6*(ffmcfmb**2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc
     . -ffmcfmb*hbcm)+8*w17*p3p6*(ffmcfmb**2*hbcm+2*ffmcfmb*fmb+
     . ffmcfmb*fmc-ffmcfmb*hbcm))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb-8)-32*p4p6)
      b(49)=ccc*(w7*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7)+
     . w17*(8*p3p7*(ffmcfmb*hbcm+3*fmb+fmc)+24*hbcm*p5p7))
      b(50)=ccc*(w7*(p3p7*(-16*ffmcfmb+24)+32*p5p7)+w17*(p3p7*(-16*
     . ffmcfmb+24)+32*p5p7))
      b(51)=ccc*(w7*(8*p3p7*(ffmcfmb**2*hbcm2-fmc2)+16*(ffmcfmb*p1p3*
     . p3p7-ffmcfmb*p2p3*p3p7-p1p2*p3p7-p1p3*p5p7+3*p1p4*p3p7))+w17*(
     . 8*p3p7*(ffmcfmb**2*hbcm2-fmc2)+16*(ffmcfmb*p1p3*p3p7-ffmcfmb*
     . p2p3*p3p7-p1p2*p3p7-p1p3*p5p7+3*p1p4*p3p7))-8*p3p7)
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+24)+32*p5p7)
      b(53)=ccc*(8*w2*(-3*p1p9-3*p2p9+4*p3p9+4*p4p9)+8*w7*(-3*p1p11-3
     . *p2p11+4*p3p11+4*p4p11)+8*w17*(-3*p1p10-3*p1p11-3*p2p10-3*
     . p2p11+4*p3p10+4*p3p11+4*p4p10+4*p4p11))
      b(54)=ccc*(4*w7*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm-4*hbcm2+6
     . *p1p3+6*p2p3-8*p3p4)+4*w17*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*
     . hbcm-4*hbcm2+6*p1p3+6*p2p3-8*p3p4))
      b(55)=ccc*(4*w7*(3*ffmcfmb*hbcm+4*fmb+fmc-3*hbcm)+4*w17*(3*
     . ffmcfmb*hbcm+4*fmb+fmc-3*hbcm))
      b(56)=ccc*(8*w7*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2-p1p2+p1p3
     . +4*p1p4)+8*w17*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2-p1p2+p1p3
     . +4*p1p4)-8)
      b(57)=-8*ccc
      b(58)=ccc*(w7*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)+w17*(8
     . *p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6))
      b(59)=ccc*(w7*(8*p1p3*(-ffmcfmb*hbcm-fmb)+4*(-ffmcfmb**2*hbcm3+
     . 2*ffmcfmb*hbcm*p2p3+fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4))+w17*(8
     . *p1p3*(-ffmcfmb*hbcm-fmb)+4*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm*
     . p2p3+fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4))+4*hbcm)
      b(60)=w2*ccc*(8*p3p6*(2*ffmcfmb*hbcm+fmb)+24*hbcm*p4p6)
      b(61)=-4*ccc*hbcm
      b(62)=8*ccc*ffmcfmb*hbcm*p3p6
      b(63)=w17*ccc*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+16*(-p3p5*p6p7-p3p6*p4p7-3*p3p7*p4p6))
      b(64)=w17*ccc*(8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(ffmcfmb*p2p3
     . *p6p7+ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*
     . p6p7+p1p5*p6p7))
      b(66)=ccc*(w2*(16*p3p6*(-ffmcfmb*hbcm*p7p9-fmb*p7p9)+16*(-
     . ffmcfmb*hbcm*p3p7*p6p9-2*hbcm*p4p6*p7p9-hbcm*p4p7*p6p9-hbcm*
     . p5p9*p6p7))+w7*(16*p3p6*(-ffmcfmb*hbcm*p7p11-fmb*p7p11)+16*(-
     . ffmcfmb*hbcm*p3p7*p6p11-2*hbcm*p4p6*p7p11-hbcm*p4p7*p6p11-hbcm
     . *p5p11*p6p7))+w17*(16*p6p7*(-hbcm*p5p10-hbcm*p5p11)+16*p4p7*(-
     . hbcm*p6p10-hbcm*p6p11)+32*p4p6*(-hbcm*p7p10-hbcm*p7p11)+16*
     . p3p7*(-ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p11)+16*p3p6*(-
     . ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p11-fmb*p7p10-fmb*p7p11))-8*
     . hbcm*p6p12)
      b(67)=w17*ccc*(p3p6*(-16*ffmcfmb-8)-32*p4p6)
      b(69)=w17*ccc*(p3p7*(-16*ffmcfmb+24)+32*p5p7)
      b(70)=w17*ccc*(8*p3p7*(ffmcfmb**2*hbcm2-fmc2)+16*(ffmcfmb*p1p3*
     . p3p7-ffmcfmb*p2p3*p3p7-p1p2*p3p7-p1p3*p5p7+3*p1p4*p3p7))
      b(72)=8*ccc*w17*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2-p1p2+p1p3
     . +4*p1p4)
      b(73)=ccc*(8*w7*p6p7*(ffmcfmb*hbcm+fmb-hbcm)+8*w17*p6p7*(
     . ffmcfmb*hbcm+fmb-hbcm))
      b(75)=ccc*(8*w2*(ffmcfmb*hbcm*p1p9+fmb*p1p9-hbcm*p1p9)+8*w7*(
     . ffmcfmb*hbcm*p1p11+fmb*p1p11-hbcm*p1p11)+8*w17*(ffmcfmb*hbcm*
     . p1p10+ffmcfmb*hbcm*p1p11+fmb*p1p10+fmb*p1p11-hbcm*p1p10-hbcm*
     . p1p11))
      b(76)=ccc*(w2*(16*p1p3*(p1p9+p2p9-p3p9-p4p9)+8*(ffmcfmb*hbcm2*
     . p1p9-fmb*hbcm*p1p9-hbcm2*p1p9-2*p1p9*p3p5))+w7*(16*p1p3*(p1p11
     . +p2p11-p3p11-p4p11)+8*(ffmcfmb*hbcm2*p1p11-fmb*hbcm*p1p11-
     . hbcm2*p1p11-2*p1p11*p3p5))+w17*(16*p3p5*(-p1p10-p1p11)+16*p1p3
     . *(p1p10+p1p11+p2p10+p2p11-p3p10-p3p11-p4p10-p4p11)+8*(ffmcfmb*
     . hbcm2*p1p10+ffmcfmb*hbcm2*p1p11-fmb*hbcm*p1p10-fmb*hbcm*p1p11-
     . hbcm2*p1p10-hbcm2*p1p11)))
      b(77)=ccc*(8*w2*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9-hbcm2*p6p9-2*
     . p3p5*p6p9+2*p3p6*p5p9)+8*w7*(ffmcfmb*hbcm2*p6p11-fmb*hbcm*
     . p6p11-hbcm2*p6p11-2*p3p5*p6p11+2*p3p6*p5p11)+w17*(16*p3p6*(
     . p5p10+p5p11)+16*p3p5*(-p6p10-p6p11)+8*(ffmcfmb*hbcm2*p6p10+
     . ffmcfmb*hbcm2*p6p11-fmb*hbcm*p6p10-fmb*hbcm*p6p11-hbcm2*p6p10-
     . hbcm2*p6p11)))
      b(78)=ccc*(8*w2*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+8*w7*(
     . ffmcfmb*hbcm*p6p11+fmb*p6p11-hbcm*p6p11)+8*w17*(ffmcfmb*hbcm*
     . p6p10+ffmcfmb*hbcm*p6p11+fmb*p6p10+fmb*p6p11-hbcm*p6p10-hbcm*
     . p6p11))
      b(79)=8*ccc*w2*p6p7*(ffmcfmb*hbcm+fmb-hbcm)
      b(81)=8*ccc*w17*p6p7*(ffmcfmb*hbcm+fmb-hbcm)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(84)=8*ccc*(p3p7+2*p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p2p3-4*p3p4)
      b(86)=-8*ccc
      b(87)=4*ccc*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(89)=-4*ccc*hbcm
      b(92)=8*ccc*p1p3
      b(93)=8*ccc*p3p6
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp15_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(3)=ccc*(w16*(16*p5p7*(p1p11-p1p9)+16*p4p7*(p1p11-p1p9)+16*
     . p3p7*(p1p11-p1p9))+16*w12*(-p1p10*p3p7-p1p10*p4p7-p1p10*p5p7)+
     . 16*(p1p12+p2p12-p3p12-p4p12-p5p12))
      b(6)=8*ccc*(-hbcm*p1p12+hbcm*p3p12+hbcm*p4p12+hbcm*p5p12-hbcm*
     . p7p11)
      b(7)=ccc*(w16*(16*p1p5*(-p6p11+p6p9)+16*p1p3*(-p6p11+p6p9)+16*
     . p1p2*(p6p11-p6p9)+16*p4p6*(p1p11-p1p9)+16*p3p6*(p1p11-p1p9)+8*
     . (-ffmcfmb*fmb*hbcm*p6p11+ffmcfmb*fmb*hbcm*p6p9+ffmcfmb*fmc*
     . hbcm*p6p11-ffmcfmb*fmc*hbcm*p6p9+ffmcfmb*hbcm2*p6p11-ffmcfmb*
     . hbcm2*p6p9+fmb*fmc*p6p11-fmb*fmc*p6p9-fmc*hbcm*p6p11+fmc*hbcm*
     . p6p9-fmc2*p6p11+fmc2*p6p9))+8*w12*(ffmcfmb*fmb*hbcm*p6p10-
     . ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*hbcm2*p6p10-fmb*fmc*p6p10+fmc*
     . hbcm*p6p10+fmc2*p6p10-2*p1p10*p3p6-2*p1p10*p4p6-2*p1p2*p6p10+2
     . *p1p3*p6p10+2*p1p5*p6p10)+8*(-p1p13-2*p2p13+2*p3p13+p4p13+
     . p5p13+p6p11))
      b(9)=w16*ccc*(8*p3p6*(fmb*fmc-fmc2)+16*(p1p2*p3p6+p1p3*p4p6-
     . p1p5*p3p6))
      ans2=8*(-fmb*p3p13-fmc*p3p13+hbcm*p2p13-hbcm*p3p13-hbcm*p4p13-
     . hbcm*p6p11)
      ans1=w16*(16*p1p5*(hbcm*p6p11-hbcm*p6p9)+16*p1p2*(-hbcm*p6p11+
     . hbcm*p6p9)+16*p4p6*(hbcm*p1p11-hbcm*p1p9)+16*p3p6*(-fmc*p1p11+
     . fmc*p1p9+fmc*p3p11-fmc*p3p9+fmc*p4p11-fmc*p4p9)+16*p3p5*(-
     . ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p9+fmc*p6p11-fmc*p6p9)+16*
     . p2p3*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9-fmc*p6p11+fmc*p6p9)
     . +16*p1p3*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9+fmb*p6p11-fmb*
     . p6p9)+8*(-ffmcfmb*fmb*hbcm2*p6p11+ffmcfmb*fmb*hbcm2*p6p9-
     . ffmcfmb*fmc*hbcm2*p6p11+ffmcfmb*fmc*hbcm2*p6p9-ffmcfmb*hbcm3*
     . p6p11+ffmcfmb*hbcm3*p6p9+fmb*fmc*hbcm*p6p11-fmb*fmc*hbcm*p6p9+
     . fmc*hbcm2*p6p11-fmc*hbcm2*p6p9+fmc2*hbcm*p6p11-fmc2*hbcm*p6p9)
     . )+w12*(16*p3p6*(fmc*p1p10-fmc*p4p10)+16*p3p5*(ffmcfmb*hbcm*
     . p6p10-fmc*p6p10)+16*p4p6*(-ffmcfmb*hbcm*p3p10+fmc*p3p10-hbcm*
     . p1p10)+16*p2p3*(-ffmcfmb*hbcm*p6p10+fmc*p6p10)+16*p1p3*(-
     . ffmcfmb*hbcm*p6p10-fmb*p6p10)+8*(ffmcfmb*fmb*hbcm2*p6p10+
     . ffmcfmb*fmc*hbcm2*p6p10+ffmcfmb*hbcm3*p6p10-fmb*fmc*hbcm*p6p10
     . -fmc*hbcm2*p6p10-fmc2*hbcm*p6p10+2*hbcm*p1p2*p6p10-2*hbcm*p1p5
     . *p6p10))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(14)=4*ccc*w16*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-
     . fmc2+2*p1p3+4*p1p4)
      b(16)=w16*ccc*(8*p3p4*(ffmcfmb*hbcm-fmc)+8*p1p3*(-ffmcfmb*hbcm-
     . fmb)+4*(ffmcfmb*hbcm3-fmc*hbcm2-2*hbcm*p1p4))
      b(17)=8*ccc*w12*(fmb*p3p6+hbcm*p4p6)
      b(18)=4*ccc*w12*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      b(19)=4*ccc*w12*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p1p3-4*p3p4)
      b(21)=ccc*(8*w16*(-hbcm*p3p11+hbcm*p3p9-hbcm*p4p11+hbcm*p4p9)+8
     . *w12*(ffmcfmb*hbcm*p3p10+2*fmb*p3p10+fmc*p3p10+hbcm*p4p10))
      b(22)=w12*ccc*(8*p3p4*(ffmcfmb*hbcm-fmc)+8*p1p3*(-ffmcfmb*hbcm-
     . fmb)+4*(ffmcfmb*hbcm3-fmc*hbcm2-2*hbcm*p1p4))
      b(27)=w12*ccc*(8*p4p6*(ffmcfmb*hbcm-fmc)-8*fmc*p3p6)
      b(28)=w12*ccc*(8*p3p6*(-fmb*fmc+fmc2)+16*(-p1p2*p3p6-p1p3*p4p6+
     . p1p5*p3p6))
      ans2=(16*p3p4*(p6p12-2*p7p13)+16*p2p3*(-p6p12+2*p7p13)+32*p3p7*
     . (p1p13+p2p13-p3p13-p4p13-p5p13)+16*p3p6*(-p1p12+p3p12+p4p12+
     . p5p12-p7p11)+8*(-fmb*hbcm*p6p12+fmb*hbcm*p7p13+fmc*hbcm*p6p12-
     . fmc*hbcm*p7p13+hbcm2*p6p12-3*hbcm2*p7p13+2*p1p3*p7p13+4*p3p11*
     . p6p7+2*p3p13*p4p7+2*p3p13*p5p7-2*p3p5*p7p13))
      ans1=w16*(32*(p3p7*p1p5)*(p6p11-p6p9)+32*(p3p7*p1p3)*(p6p11-
     . p6p9)+32*(p3p7*p1p2)*(-p6p11+p6p9)+32*(p4p6*p3p7)*(-p1p11+p1p9
     . )+32*(p5p7*p3p6)*(-p1p11+p1p9)+32*(p4p7*p3p6)*(-p1p11+p1p9)+64
     . *(p3p7*p3p6)*(-p1p11+p1p9)+16*p3p6*(-fmb*fmc*p7p9+fmc2*p7p9)+
     . 16*p3p7*(ffmcfmb*fmb*hbcm*p6p11-ffmcfmb*fmb*hbcm*p6p9-ffmcfmb*
     . fmc*hbcm*p6p11+ffmcfmb*fmc*hbcm*p6p9-ffmcfmb*hbcm2*p6p11+
     . ffmcfmb*hbcm2*p6p9-fmb*fmc*p6p11+fmb*fmc*p6p9+fmc*hbcm*p6p11-
     . fmc*hbcm*p6p9+fmc2*p6p11-fmc2*p6p9)+32*(-p1p2*p3p6*p7p9-p1p3*
     . p4p6*p7p9+p1p5*p3p6*p7p9))+w12*(16*p3p7*(-ffmcfmb*fmb*hbcm*
     . p6p10+ffmcfmb*fmc*hbcm*p6p10+ffmcfmb*hbcm2*p6p10+fmb*fmc*p6p10
     . -fmc*hbcm*p6p10-fmc2*p6p10)+32*(2*p1p10*p3p6*p3p7+p1p10*p3p6*
     . p4p7+p1p10*p3p6*p5p7+p1p10*p3p7*p4p6+p1p2*p3p7*p6p10-p1p3*p3p7
     . *p6p10-p1p5*p3p7*p6p10))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(32)=w16*ccc*(8*p4p6*(-ffmcfmb*hbcm+fmc)+8*fmc*p3p6)
      b(33)=ccc*(w16*(16*p5p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p9+
     . fmc*p6p11-fmc*p6p9)+16*p4p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*
     . p6p9+fmc*p6p11-fmc*p6p9)+16*p3p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*
     . hbcm*p6p9+fmc*p6p11-fmc*p6p9))+w12*(16*p5p7*(ffmcfmb*hbcm*
     . p6p10-fmc*p6p10)+16*p4p7*(ffmcfmb*hbcm*p6p10-fmc*p6p10)+16*
     . p3p7*(ffmcfmb*hbcm*p6p10-fmc*p6p10))+8*(-fmb*p6p12+fmb*p7p13-
     . fmc*p6p12+fmc*p7p13+hbcm*p6p12-hbcm*p7p13))
      b(34)=4*ccc*w16*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(36)=4*ccc*w16*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p1p3-4*p3p4)
      b(39)=16*ccc*w16*(p3p6*p3p7+p3p6*p4p7+p3p6*p5p7)
      b(42)=16*ccc*w16*(-p3p6*p3p7-p3p6*p4p7-p3p6*p5p7)
      b(43)=ccc*(w16*(16*p5p7*(p6p11-p6p9)+16*p4p7*(p6p11-p6p9)+16*
     . p3p7*(p6p11-p6p9))+16*w12*(-p3p7*p6p10-p4p7*p6p10-p5p7*p6p10)+
     . 8*(-p6p12+p7p13))
      b(44)=w16*ccc*(8*p3p6*(fmb*fmc-fmc2)+16*(p1p2*p3p6+p1p3*p4p6-
     . p1p5*p3p6))
      b(45)=8*ccc*w16*(p3p6+2*p4p6)
      b(46)=w16*ccc*(8*p4p6*(ffmcfmb*hbcm-fmc)-8*fmc*p3p6)
      b(47)=8*ccc*w16*(-p3p6-2*p4p6)
      b(53)=ccc*(8*w16*(p1p11-p1p9-2*p3p11+2*p3p9-2*p4p11+2*p4p9)+8*
     . w12*(-p1p10+2*p3p10+2*p4p10))
      b(54)=4*ccc*w16*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3+4*p3p4)
      b(55)=4*ccc*w16*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      b(56)=4*ccc*w16*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc+fmc2
     . -2*p1p3-4*p1p4)
      b(58)=8*ccc*w16*(-fmb*p3p6-hbcm*p4p6)
      b(59)=w16*ccc*(8*p3p4*(-ffmcfmb*hbcm+fmc)+8*p1p3*(ffmcfmb*hbcm+
     . fmb)+4*(-ffmcfmb*hbcm3+fmc*hbcm2+2*hbcm*p1p4))
      b(60)=8*ccc*w16*(fmb*p3p6+hbcm*p4p6)
      b(63)=16*ccc*w12*(-p3p6*p3p7-p3p6*p4p7-p3p6*p5p7)
      b(66)=ccc*(w16*(16*p5p7*(hbcm*p6p11-hbcm*p6p9)+16*p4p7*(hbcm*
     . p6p11-hbcm*p6p9)+16*p3p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9
     . +fmb*p6p11-fmb*p6p9))+w12*(16*p3p7*(-ffmcfmb*hbcm*p6p10-fmb*
     . p6p10)+16*(-hbcm*p4p7*p6p10-hbcm*p5p7*p6p10))+8*(-hbcm*p6p12+
     . hbcm*p7p13))
      b(67)=8*ccc*w12*(-p3p6-2*p4p6)
      b(72)=4*ccc*w12*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-
     . fmc2+2*p1p3+4*p1p4)
      b(75)=ccc*(8*w16*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p1p9+ffmcfmb
     . *hbcm*p3p11-ffmcfmb*hbcm*p3p9+ffmcfmb*hbcm*p4p11-ffmcfmb*hbcm*
     . p4p9-fmb*p1p11+fmb*p1p9-fmc*p3p11+fmc*p3p9-fmc*p4p11+fmc*p4p9+
     . hbcm*p1p11-hbcm*p1p9)+8*w12*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*
     . p3p10-ffmcfmb*hbcm*p4p10+fmb*p1p10+fmc*p3p10+fmc*p4p10-hbcm*
     . p1p10))
      b(76)=ccc*(w16*(16*p3p5*(p1p11-p1p9)+16*p2p3*(-p1p11+p1p9)+16*
     . p1p3*(-p1p11+p1p9+p3p11-p3p9+p4p11-p4p9)+8*(fmb*hbcm*p1p11-fmb
     . *hbcm*p1p9-fmc*hbcm*p1p11+fmc*hbcm*p1p9+hbcm2*p1p11-hbcm2*p1p9
     . ))+w12*(16*p1p3*(p1p10-p3p10-p4p10)+8*(-fmb*hbcm*p1p10+fmc*
     . hbcm*p1p10-hbcm2*p1p10+2*p1p10*p2p3-2*p1p10*p3p5)))
      b(77)=ccc*(w16*(16*p3p5*(p6p11-p6p9)+16*p2p3*(-p6p11+p6p9)+16*
     . p3p6*(p2p11-p2p9-p5p11+p5p9)+8*(-ffmcfmb*hbcm2*p6p11+ffmcfmb*
     . hbcm2*p6p9+fmb*hbcm*p6p11-fmb*hbcm*p6p9+hbcm2*p6p11-hbcm2*p6p9
     . ))+w12*(16*p3p6*(-p2p10+p5p10)+8*(ffmcfmb*hbcm2*p6p10-fmb*hbcm
     . *p6p10-hbcm2*p6p10+2*p2p3*p6p10-2*p3p5*p6p10))-8*p3p13)
      b(78)=ccc*(8*w16*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p9-fmb*
     . p6p11+fmb*p6p9+hbcm*p6p11-hbcm*p6p9)+8*w12*(ffmcfmb*hbcm*p6p10
     . +fmb*p6p10-hbcm*p6p10))
      b(84)=8*ccc*(-p3p7-p4p7-p5p7)
      b(85)=4*ccc*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3-2*p2p3+2*p3p4+2*
     . p3p5)
      b(87)=4*ccc*hbcm
      b(89)=4*ccc*hbcm
      b(92)=16*ccc*(hbcm2-p1p3-p2p3+p3p4+p3p5)
      do 200 n=1,94 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp14_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*
     . p1p3))
      b(3)=ccc*(16*w1*(p1p8*p3p7+p1p8*p4p7+p1p8*p5p7)+16*w13*(p1p10*
     . p3p7+p1p10*p4p7+p1p10*p5p7)+16*(-p1p12-p2p12+p3p12+p4p12+p5p12
     . ))
      b(6)=8*ccc*(hbcm*p1p12-hbcm*p3p12-hbcm*p4p12-hbcm*p5p12+hbcm*
     . p7p11)
      b(7)=ccc*(8*w1*(-ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p8+hbcm2*p6p8+2*
     . p1p2*p6p8-2*p1p3*p6p8-2*p1p5*p6p8+2*p1p8*p3p6+2*p1p8*p4p6)+8*
     . w13*(-ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10+hbcm2*p6p10+2*p1p10*
     . p3p6+2*p1p10*p4p6+2*p1p2*p6p10-2*p1p3*p6p10-2*p1p5*p6p10)+8*(
     . p1p13+2*p2p13-2*p3p13-p4p13-p5p13-p6p11))
      b(8)=w1*ccc*(8*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . hbcm-fmc*hbcm)+16*(-p1p2*p3p6-p1p3*p4p6+p1p5*p3p6))
      b(10)=ccc*(w1*(16*p1p3*(fmb*p6p8+fmc*p6p8)+16*p4p6*(-ffmcfmb*
     . hbcm*p3p8-fmb*p3p8+hbcm*p3p8)+16*p3p6*(-ffmcfmb*hbcm*p1p8+
     . ffmcfmb*hbcm*p4p8+hbcm*p1p8-hbcm*p4p8)+8*(ffmcfmb*fmb*hbcm2*
     . p6p8+ffmcfmb*fmc*hbcm2*p6p8+fmb*fmc*hbcm*p6p8-fmb*hbcm2*p6p8+
     . fmb2*hbcm*p6p8-fmc*hbcm2*p6p8))+w13*(16*p1p3*(fmb*p6p10+fmc*
     . p6p10)+16*p4p6*(-ffmcfmb*hbcm*p3p10-fmb*p3p10+hbcm*p3p10)+16*
     . p3p6*(-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p4p10+hbcm*p1p10-hbcm*
     . p4p10)+8*(ffmcfmb*fmb*hbcm2*p6p10+ffmcfmb*fmc*hbcm2*p6p10+fmb*
     . fmc*hbcm*p6p10-fmb*hbcm2*p6p10+fmb2*hbcm*p6p10-fmc*hbcm2*p6p10
     . ))+8*(fmb*p3p13+fmc*p3p13-hbcm*p1p13-hbcm*p2p13+hbcm*p5p13))
      b(13)=4*ccc*w1*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc+fmb*
     . hbcm-fmb2-fmc*hbcm-2*p1p3-4*p1p4)
      b(15)=w1*ccc*(8*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*
     . hbcm+fmc+hbcm)+4*(ffmcfmb*hbcm3+fmb*hbcm2+2*hbcm*p1p4-hbcm3))
      b(17)=w13*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-fmc-hbcm)-8*hbcm*p4p6)
      b(18)=4*ccc*w13*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(19)=4*ccc*w13*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3+4*p3p4)
      b(21)=ccc*(8*w1*(ffmcfmb*hbcm*p3p8-fmb*p3p8-2*fmc*p3p8+hbcm*
     . p1p8-hbcm*p3p8-hbcm*p4p8)+8*w13*(ffmcfmb*hbcm*p3p10-fmb*p3p10-
     . 2*fmc*p3p10+hbcm*p1p10-hbcm*p3p10-hbcm*p4p10))
      b(22)=w13*ccc*(8*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*
     . hbcm+fmc+hbcm)+4*(ffmcfmb*hbcm3+fmb*hbcm2+2*hbcm*p1p4-hbcm3))
      b(23)=w1*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-fmc-hbcm)-8*hbcm*p4p6)
      b(24)=4*ccc*w1*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(25)=4*ccc*w1*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3+4*p3p4)
      b(27)=w13*ccc*(8*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb*
     . hbcm-hbcm))
      b(28)=w13*ccc*(8*p3p6*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . hbcm+fmc*hbcm)+16*(p1p2*p3p6+p1p3*p4p6-p1p5*p3p6))
      b(30)=ccc*(w13*(16*p3p7*(ffmcfmb*hbcm2*p6p10+fmb*hbcm*p6p10-
     . hbcm2*p6p10)+32*(-2*p1p10*p3p6*p3p7-p1p10*p3p6*p4p7-p1p10*p3p6
     . *p5p7-p1p10*p3p7*p4p6-p1p2*p3p7*p6p10+p1p3*p3p7*p6p10+p1p5*
     . p3p7*p6p10))+w1*(16*p3p7*(ffmcfmb*hbcm2*p6p8+fmb*hbcm*p6p8-
     . hbcm2*p6p8)+16*p3p6*(ffmcfmb*fmb*hbcm*p7p8-ffmcfmb*fmc*hbcm*
     . p7p8-fmb*hbcm*p7p8+fmc*hbcm*p7p8)+32*(p1p2*p3p6*p7p8-p1p2*p3p7
     . *p6p8+p1p3*p3p7*p6p8+p1p3*p4p6*p7p8-p1p5*p3p6*p7p8+p1p5*p3p7*
     . p6p8-2*p1p8*p3p6*p3p7-p1p8*p3p6*p4p7-p1p8*p3p6*p5p7-p1p8*p3p7*
     . p4p6))+(16*p3p4*(-p6p12+2*p7p13)+16*p2p3*(p6p12-2*p7p13)+32*
     . p3p7*(-p1p13-p2p13+p3p13+p4p13+p5p13)+16*p3p6*(p1p12-p3p12-
     . p4p12-p5p12+p7p11)+8*(-hbcm2*p6p12+3*hbcm2*p7p13-2*p1p3*p7p13-
     . 4*p3p11*p6p7-2*p3p13*p4p7-2*p3p13*p5p7+2*p3p5*p7p13)))
      b(31)=w1*ccc*(8*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb*
     . hbcm-hbcm))
      b(33)=8*ccc*(fmb*p6p12-fmb*p7p13+fmc*p6p12-fmc*p7p13)
      b(43)=ccc*(16*w1*(p3p7*p6p8+p4p7*p6p8+p5p7*p6p8)+16*w13*(p3p7*
     . p6p10+p4p7*p6p10+p5p7*p6p10)+8*(p6p12-p7p13))
      b(53)=ccc*(8*w1*(p1p8-2*p3p8-2*p4p8)+8*w13*(p1p10-2*p3p10-2*
     . p4p10))
      b(63)=16*ccc*w13*(p3p6*p3p7+p3p6*p4p7+p3p6*p5p7)
      b(65)=16*ccc*w1*(p3p6*p3p7+p3p6*p4p7+p3p6*p5p7)
      b(66)=ccc*(16*w1*p3p7*(fmb*p6p8+fmc*p6p8)+16*w13*p3p7*(fmb*
     . p6p10+fmc*p6p10))
      b(67)=8*ccc*w13*(p3p6+2*p4p6)
      b(68)=8*ccc*w1*(p3p6+2*p4p6)
      b(72)=4*ccc*w13*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc+fmb
     . *hbcm-fmb2-fmc*hbcm-2*p1p3-4*p1p4)
      b(75)=ccc*(8*w1*(ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p3p8-ffmcfmb*
     . hbcm*p4p8-fmb*p3p8-fmb*p4p8-fmc*p1p8-hbcm*p1p8+hbcm*p3p8+hbcm*
     . p4p8)+8*w13*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p3p10-ffmcfmb*
     . hbcm*p4p10-fmb*p3p10-fmb*p4p10-fmc*p1p10-hbcm*p1p10+hbcm*p3p10
     . +hbcm*p4p10))
      b(76)=ccc*(w1*(16*p1p3*(-p1p8+p3p8+p4p8)+8*(hbcm2*p1p8-2*p1p8*
     . p2p3+2*p1p8*p3p5))+w13*(16*p1p3*(-p1p10+p3p10+p4p10)+8*(hbcm2*
     . p1p10-2*p1p10*p2p3+2*p1p10*p3p5)))
      b(77)=ccc*(w1*(16*p3p6*(p2p8-p5p8)+8*(ffmcfmb*hbcm2*p6p8+fmb*
     . hbcm*p6p8-2*p2p3*p6p8+2*p3p5*p6p8))+w13*(16*p3p6*(p2p10-p5p10)
     . +8*(ffmcfmb*hbcm2*p6p10+fmb*hbcm*p6p10-2*p2p3*p6p10+2*p3p5*
     . p6p10))+8*p3p13)
      b(78)=ccc*(8*w1*(-fmb*p6p8-fmc*p6p8)+8*w13*(-fmb*p6p10-fmc*
     . p6p10))
      b(83)=8*ccc*(hbcm*p3p7+hbcm*p4p7+hbcm*p5p7)
      b(84)=8*ccc*(p3p7+p4p7+p5p7)
      b(85)=4*ccc*(-fmb*hbcm+fmc*hbcm-2*hbcm2+2*p1p3+2*p2p3-2*p3p4-2*
     . p3p5)
      b(87)=4*ccc*hbcm
      b(89)=4*ccc*hbcm
      b(92)=16*ccc*(-hbcm2+p1p3+p2p3-p3p4-p3p5)
      do 200 n=1,94 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp13_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(93) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,93 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(8)=w1*ccc*(8*p4p6*(fmb*hbcm-fmc*hbcm)+8*p3p6*(fmb*hbcm-fmc*
     . hbcm))
      b(10)=w1*ccc*(16*p4p6*(hbcm*p3p8-hbcm*p4p8)+16*p3p6*(hbcm*p1p8-
     . hbcm*p4p8)+8*(-fmb2*hbcm*p6p8+fmc2*hbcm*p6p8+2*hbcm*p1p3*p6p8+
     . 2*hbcm*p2p3*p6p8-2*hbcm*p3p5*p6p8-hbcm3*p6p8))
      b(15)=4*ccc*w1*(fmb2*hbcm-fmc2*hbcm+2*hbcm*p1p3-2*hbcm*p3p4-
     . hbcm3)
      b(21)=8*ccc*w1*(-hbcm*p3p8+hbcm*p4p8)
      b(23)=-8*ccc*hbcm*p3p6*w1
      b(24)=4*ccc*hbcm*w1
      b(25)=4*ccc*w1*(fmb*hbcm-fmc*hbcm)
      b(30)=w1*ccc*(16*p4p6*(-fmb*hbcm*p7p8+fmc*hbcm*p7p8)+16*p3p6*(-
     . fmb*hbcm*p7p8+fmc*hbcm*p7p8))
      b(31)=8*ccc*w1*(-hbcm*p3p6-hbcm*p4p6)
      b(33)=16*ccc*w1*(-hbcm*p3p7*p6p8-hbcm*p4p7*p6p8-hbcm*p5p7*p6p8)
      b(66)=16*ccc*hbcm*p3p7*p6p8*w1
      b(75)=8*ccc*w1*(-hbcm*p1p8+hbcm*p3p8)
      b(76)=8*ccc*w1*(-fmb*hbcm*p1p8+fmb*hbcm*p3p8+fmc*hbcm*p1p8-fmc*
     . hbcm*p3p8)
      b(77)=8*ccc*w1*(-fmb*hbcm*p6p8+fmc*hbcm*p6p8)
      b(78)=-8*ccc*hbcm*p6p8*w1
      do 200 n=1,93 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp11_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(93) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,93 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*
     . p2p3))
      b(1)=16*ccc*w1*(-2*p1p5*p6p7-p3p7*p5p6-2*p5p6*p5p7)
      b(3)=ccc*(16*w1*(-p1p2*p7p8+p1p3*p7p8+p1p4*p7p8+p1p8*p3p7+2*
     . p1p8*p4p7)+16*w8*(p1p11*p3p7+2*p1p11*p4p7-p1p2*p7p11+p1p3*
     . p7p11+p1p4*p7p11)+8*(p1p12-p7p10))
      b(4)=16*ccc*w1*(p1p2*p3p7+2*p1p3*p5p7-p1p4*p3p7)
      b(6)=ccc*(w1*(16*p1p3*(ffmcfmb*hbcm*p7p8-fmc*p7p8-hbcm*p7p8)+16
     . *(fmc*p1p8*p3p7+hbcm*p1p2*p7p8-hbcm*p1p4*p7p8+hbcm*p1p8*p5p7))
     . +w8*(16*p1p3*(ffmcfmb*hbcm*p7p11-fmc*p7p11-hbcm*p7p11)+16*(fmc
     . *p1p11*p3p7+hbcm*p1p11*p5p7+hbcm*p1p2*p7p11-hbcm*p1p4*p7p11))-
     . 8*hbcm*p7p10)
      b(7)=ccc*(w1*(16*p5p6*(-p2p8+2*p3p8+2*p4p8)-32*p1p5*p6p8)+w8*(
     . 16*p5p6*(-p2p11+2*p3p11+2*p4p11)-32*p1p5*p6p11))
      b(8)=w1*ccc*(8*p5p6*(fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(2*p1p5*p3p6
     . +p2p3*p5p6-2*p3p4*p5p6))
      b(10)=ccc*(w1*(16*p5p6*(ffmcfmb*hbcm*p3p8-fmb*p3p8-2*fmc*p3p8+
     . hbcm*p1p8+hbcm*p2p8-hbcm*p3p8-hbcm*p4p8)+16*p3p6*(-ffmcfmb*
     . hbcm*p1p8+fmb*p1p8+2*fmc*p1p8+hbcm*p1p8)+16*(hbcm*p1p5*p6p8+
     . hbcm*p1p8*p4p6))+w8*(16*p5p6*(hbcm*p1p11+hbcm*p2p11-hbcm*p3p11
     . -hbcm*p4p11)+16*p3p6*(-ffmcfmb*hbcm*p1p11+fmb*p1p11+2*fmc*
     . p1p11+hbcm*p1p11)+16*(hbcm*p1p11*p4p6+hbcm*p1p5*p6p11)))
      b(11)=16*ccc*w1*(-fmc*p3p7*p5p6-hbcm*p1p5*p6p7-hbcm*p5p6*p5p7)
      b(13)=16*ccc*w1*(p1p2-p1p3-p1p4)
      b(15)=w1*ccc*(8*p1p3*(-ffmcfmb*hbcm+fmb+2*fmc+hbcm)+8*(-hbcm*
     . p1p2+hbcm*p1p4))
      b(21)=ccc*(8*w8*(hbcm*p2p11-hbcm*p3p11-hbcm*p4p11)+8*w1*(
     . ffmcfmb*hbcm*p3p8-fmb*p3p8-2*fmc*p3p8+hbcm*p2p8-hbcm*p3p8-hbcm
     . *p4p8))
      b(23)=w1*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+8*(-hbcm*
     . p4p6-hbcm*p5p6))
      b(24)=4*ccc*w1*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(25)=4*ccc*w1*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p2p3+4*p3p4)
      b(26)=8*ccc*w1*(-fmc*p3p7-hbcm*p5p7)
      b(30)=ccc*(w1*(32*(p5p6*p3p7)*(2*p2p8-2*p3p8-2*p4p8-p5p8)+16*
     . p6p7*(-fmb*hbcm*p1p8+fmc*hbcm*p1p8-2*hbcm2*p1p8)+16*p5p6*(-
     . ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p8+2*hbcm2*p7p8)+32*(2*p1p3*p1p8
     . *p6p7-p1p5*p3p6*p7p8+p1p5*p3p7*p6p8+p1p8*p2p3*p6p7-2*p1p8*p3p5
     . *p6p7-2*p1p8*p3p6*p3p7-2*p1p8*p3p6*p4p7-p1p8*p3p7*p4p6-p2p3*
     . p5p6*p7p8+p3p4*p5p6*p7p8))+w8*(32*(p5p6*p3p7)*(2*p2p11-2*p3p11
     . -2*p4p11-p5p11)+16*p6p7*(-fmb*hbcm*p1p11+fmc*hbcm*p1p11-2*
     . hbcm2*p1p11)+16*p5p6*(-ffmcfmb*hbcm2*p7p11-fmc*hbcm*p7p11)+32*
     . (2*p1p11*p1p3*p6p7+p1p11*p2p3*p6p7-2*p1p11*p3p5*p6p7-2*p1p11*
     . p3p6*p3p7-2*p1p11*p3p6*p4p7-p1p11*p3p7*p4p6+p1p5*p3p6*p7p11+
     . p1p5*p3p7*p6p11-p3p4*p5p6*p7p11))-16*p3p12*p5p6)
      b(31)=8*ccc*w1*p5p6*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(33)=ccc*(16*w1*p5p6*(-ffmcfmb*hbcm*p7p8+fmc*p7p8)+16*w8*p5p6*
     . (-ffmcfmb*hbcm*p7p11+fmc*p7p11))
      b(35)=ccc*(8*w1*(-ffmcfmb*hbcm*p7p8+fmc*p7p8)+8*w8*(-ffmcfmb*
     . hbcm*p7p11+fmc*p7p11))
      b(38)=ccc*(w1*(16*p3p7*(p1p8+p2p8-p5p8)+8*(-ffmcfmb*hbcm2*p7p8-
     . fmc*hbcm*p7p8-2*p3p4*p7p8))+w8*(16*p3p7*(p1p11+p2p11-p5p11)+8*
     . (-ffmcfmb*hbcm2*p7p11-fmc*hbcm*p7p11-2*p3p4*p7p11))-8*p3p12)
      b(39)=w8*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p6p7-p2p3*p6p7+2*p3p5*p6p7+2*p3p6*p3p7+2*p3p6*p4p7+p3p7*p4p6+
     . p3p7*p5p6))
      b(40)=16*ccc*w8*(-fmc*p3p7*p5p6-hbcm*p1p5*p6p7-hbcm*p5p6*p5p7)
      b(41)=16*ccc*w8*(-2*p1p5*p6p7-p3p7*p5p6-2*p5p6*p5p7)
      b(43)=ccc*(w1*(16*p6p7*(-2*p1p8-p2p8+2*p5p8)+16*(p3p6*p7p8+p3p7
     . *p6p8+p4p6*p7p8+2*p4p7*p6p8+p5p6*p7p8))+w8*(16*p6p7*(-2*p1p11-
     . p2p11+2*p5p11)+16*(p3p6*p7p11+p3p7*p6p11+p4p6*p7p11+2*p4p7*
     . p6p11+p5p6*p7p11))+8*(p6p12-p7p13))
      b(44)=w8*ccc*(8*p5p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p5*
     . p3p6-p2p3*p5p6+2*p3p4*p5p6))
      b(45)=16*ccc*w8*(p3p6+p4p6+p5p6)
      b(46)=8*ccc*w8*p5p6*(ffmcfmb*hbcm-fmb-2*fmc)
      b(49)=8*ccc*w8*(-fmc*p3p7-hbcm*p5p7)
      b(50)=8*ccc*w8*(-p3p7-2*p5p7)
      b(51)=16*ccc*w8*(p1p2*p3p7+2*p1p3*p5p7-p1p4*p3p7)
      b(53)=ccc*(8*w1*(p2p8-2*p3p8-2*p4p8)+8*w8*(p2p11-2*p3p11-2*
     . p4p11))
      b(54)=4*ccc*w8*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p2p3+4*p3p4)
      b(55)=4*ccc*w8*(ffmcfmb*hbcm-fmb-2*fmc)
      b(56)=16*ccc*w8*(p1p2-p1p3-p1p4)
      b(58)=w8*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+8*(-hbcm*
     . p4p6-hbcm*p5p6))
      b(59)=w8*ccc*(8*p1p3*(-ffmcfmb*hbcm+fmb+2*fmc+hbcm)+8*(-hbcm*
     . p1p2+hbcm*p1p4))
      b(61)=4*ccc*hbcm
      b(62)=8*ccc*hbcm*p5p6
      b(65)=w1*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p6p7-p2p3*p6p7+2*p3p5*p6p7+2*p3p6*p3p7+2*p3p6*p4p7+p3p7*p4p6+
     . p3p7*p5p6))
      b(66)=ccc*(w1*(16*p6p7*(-hbcm*p1p8+hbcm*p5p8)+16*p3p7*(-ffmcfmb
     . *hbcm*p6p8+fmb*p6p8+fmc*p6p8+hbcm*p6p8)+16*p3p6*(-ffmcfmb*hbcm
     . *p7p8+fmc*p7p8+hbcm*p7p8)+16*(hbcm*p4p6*p7p8+hbcm*p4p7*p6p8+
     . hbcm*p5p6*p7p8))+w8*(16*p6p7*(-hbcm*p1p11+hbcm*p5p11)+16*p3p7*
     . (-ffmcfmb*hbcm*p6p11+fmb*p6p11+fmc*p6p11+hbcm*p6p11)+16*p3p6*(
     . -ffmcfmb*hbcm*p7p11+fmc*p7p11+hbcm*p7p11)+16*(hbcm*p4p6*p7p11+
     . hbcm*p4p7*p6p11+hbcm*p5p6*p7p11))+8*hbcm*p6p12)
      b(68)=16*ccc*w1*(p3p6+p4p6+p5p6)
      b(71)=8*ccc*w1*(-p3p7-2*p5p7)
      b(73)=8*ccc*w8*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(74)=8*ccc*w1*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(75)=ccc*(8*w1*(ffmcfmb*hbcm*p1p8-fmb*p1p8-2*fmc*p1p8)+8*w8*(
     . ffmcfmb*hbcm*p1p11-fmb*p1p11-2*fmc*p1p11))
      b(76)=ccc*(w1*(16*p1p3*(-2*p1p8-p2p8+2*p3p8+2*p4p8)+8*(fmb*hbcm
     . *p1p8-fmc*hbcm*p1p8+2*hbcm2*p1p8-2*p1p8*p2p3+4*p1p8*p3p5))+w8*
     . (16*p1p3*(-2*p1p11-p2p11+2*p3p11+2*p4p11)+8*(fmb*hbcm*p1p11-
     . fmc*hbcm*p1p11+2*hbcm2*p1p11-2*p1p11*p2p3+4*p1p11*p3p5)))
      b(77)=ccc*(w1*(16*p3p6*(2*p1p8+p2p8-2*p5p8)+8*(fmb*hbcm*p6p8-
     . fmc*hbcm*p6p8+2*hbcm2*p6p8-4*p1p3*p6p8-2*p2p3*p6p8+4*p3p5*p6p8
     . ))+w8*(16*p3p6*(2*p1p11+p2p11-2*p5p11)+8*(fmb*hbcm*p6p11-fmc*
     . hbcm*p6p11+2*hbcm2*p6p11-4*p1p3*p6p11-2*p2p3*p6p11+4*p3p5*
     . p6p11)))
      b(78)=ccc*(8*w1*(ffmcfmb*hbcm*p6p8-fmb*p6p8-2*fmc*p6p8)+8*w8*(
     . ffmcfmb*hbcm*p6p11-fmb*p6p11-2*fmc*p6p11))
      do 200 n=1,93 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp12_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(93) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,93 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(2)=16*ccc*w15*(2*p1p5*p6p7+p3p7*p5p6+2*p5p6*p5p7)
      b(3)=ccc*(16*w7*(-p1p11*p3p7-2*p1p11*p4p7+p1p2*p7p11-p1p3*p7p11
     . -p1p4*p7p11)+w15*(16*p1p4*(p7p10-p7p9)+16*p1p3*(p7p10-p7p9)+16
     . *p1p2*(-p7p10+p7p9)+32*p4p7*(p1p10-p1p9)+16*p3p7*(p1p10-p1p9))
     . +8*(-p1p12+p7p10))
      b(5)=16*ccc*w15*(-p1p2*p3p7-2*p1p3*p5p7+p1p4*p3p7)
      b(6)=ccc*(w7*(16*p1p3*(fmb*p7p11+fmc*p7p11)+16*p3p7*(-ffmcfmb*
     . hbcm*p1p11-fmb*p1p11-fmc*p1p11)-16*hbcm*p1p11*p5p7)+w15*(16*
     . p5p7*(hbcm*p1p10-hbcm*p1p9)+16*p1p3*(-fmb*p7p10+fmb*p7p9-fmc*
     . p7p10+fmc*p7p9)+16*p3p7*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p9+
     . fmb*p1p10-fmb*p1p9+fmc*p1p10-fmc*p1p9))-8*hbcm*p1p12)
      b(7)=ccc*(w7*(16*p5p6*(p2p11-2*p3p11-2*p4p11)+32*p1p5*p6p11)+
     . w15*(16*p5p6*(-p2p10+p2p9+2*p3p10-2*p3p9+2*p4p10-2*p4p9)+32*
     . p1p5*(-p6p10+p6p9)))
      b(9)=w15*ccc*(8*p5p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p5*
     . p3p6-p2p3*p5p6+2*p3p4*p5p6))
      b(10)=ccc*(w7*(16*p5p6*(-hbcm*p1p11+hbcm*p3p11+hbcm*p4p11)+16*
     . p3p6*(-ffmcfmb*hbcm*p1p11-2*fmb*p1p11-fmc*p1p11)+16*(-hbcm*
     . p1p11*p4p6-hbcm*p1p5*p6p11))+w15*(16*p1p5*(hbcm*p6p10-hbcm*
     . p6p9)+16*p4p6*(hbcm*p1p10-hbcm*p1p9)+16*p5p6*(-ffmcfmb*hbcm*
     . p3p10-2*fmb*p3p10-fmc*p3p10+hbcm*p1p10-hbcm*p1p9+hbcm*p3p9-
     . hbcm*p4p10+hbcm*p4p9)+16*p3p6*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm
     . *p1p9+2*fmb*p1p10-2*fmb*p1p9+fmc*p1p10-fmc*p1p9)))
      b(12)=w15*ccc*(16*(p5p6*p3p7)*(ffmcfmb*hbcm+fmb+fmc)+16*(p6p7*
     . p1p3)*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)+16*(hbcm*p1p5*p6p7+hbcm*
     . p5p6*p5p7))
      b(14)=16*ccc*w15*(-p1p2+p1p3+p1p4)
      b(16)=w15*ccc*(8*p1p3*(-ffmcfmb*hbcm-2*fmb-fmc)+8*(hbcm*p1p2-
     . hbcm*p1p4))
      b(17)=w15*ccc*(8*p3p6*(-ffmcfmb*hbcm-2*fmb-fmc)+8*(-hbcm*p4p6-
     . hbcm*p5p6))
      b(18)=4*ccc*w15*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(19)=4*ccc*w15*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p2p3+4*p3p4)
      b(20)=w15*ccc*(8*p3p7*(-ffmcfmb*hbcm-fmb-fmc)-8*hbcm*p5p7)
      b(21)=ccc*(8*w7*(hbcm*p3p11+hbcm*p4p11)+8*w15*(-ffmcfmb*hbcm*
     . p3p10-2*fmb*p3p10-fmc*p3p10+hbcm*p3p9-hbcm*p4p10+hbcm*p4p9))
      b(22)=w15*ccc*(8*p1p3*(ffmcfmb*hbcm+2*fmb+fmc)+8*(-hbcm*p1p2+
     . hbcm*p1p4))
      b(27)=8*ccc*w15*p5p6*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(28)=w15*ccc*(8*p5p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p5*
     . p3p6-p2p3*p5p6+2*p3p4*p5p6))
      b(29)=w15*ccc*(16*(p5p6*p3p7)*(-ffmcfmb*hbcm-fmb-fmc)+16*(-hbcm
     . *p1p5*p6p7-hbcm*p5p6*p5p7))
      ans=ccc*(w7*(32*(p5p6*p3p7)*(-2*p2p11+2*p3p11+2*p4p11+p5p11)+16
     . *p6p7*(fmb*hbcm*p1p11-fmc*hbcm*p1p11+2*hbcm2*p1p11)+16*p5p6*(-
     . ffmcfmb*hbcm2*p7p11+fmc*hbcm*p7p11+hbcm2*p7p11)+32*(-2*p1p11*
     . p1p3*p6p7-p1p11*p2p3*p6p7+2*p1p11*p3p5*p6p7+2*p1p11*p3p6*p3p7+
     . 2*p1p11*p3p6*p4p7+p1p11*p3p7*p4p6-p1p5*p3p6*p7p11-p1p5*p3p7*
     . p6p11+p3p4*p5p6*p7p11))+w15*(32*(p5p6*p3p4)*(-p7p10-p7p9)+32*(
     . p5p6*p3p7)*(2*p2p10-2*p2p9-2*p3p10+2*p3p9-2*p4p10+2*p4p9-p5p10
     . +p5p9)+32*(p3p7*p1p5)*(p6p10-p6p9)+32*(p3p6*p1p5)*(p7p10+p7p9)
     . +32*(p4p6*p3p7)*(-p1p10+p1p9)+64*(p4p7*p3p6)*(-p1p10+p1p9)+64*
     . (p3p7*p3p6)*(-p1p10+p1p9)+64*(p6p7*p3p5)*(-p1p10+p1p9)+32*(
     . p6p7*p2p3)*(p1p10-p1p9)+64*(p6p7*p1p3)*(p1p10-p1p9)+16*p6p7*(-
     . fmb*hbcm*p1p10+fmb*hbcm*p1p9+fmc*hbcm*p1p10-fmc*hbcm*p1p9-2*
     . hbcm2*p1p10+2*hbcm2*p1p9)+16*p5p6*(ffmcfmb*hbcm2*p7p10-ffmcfmb
     . *hbcm2*p7p9+fmb*hbcm*p7p9-fmc*hbcm*p7p10-hbcm2*p7p10-hbcm2*
     . p7p9)+32*p2p3*p5p6*p7p9)+16*p3p12*p5p6)
      b(30)=ans
      b(32)=8*ccc*w15*p5p6*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(33)=ccc*(16*w7*p5p6*(-fmb*p7p11-fmc*p7p11)+w15*(16*p5p6*(fmb*
     . p7p10-fmb*p7p9+fmc*p7p10-fmc*p7p9)+16*p6p7*(ffmcfmb*hbcm*p1p9+
     . 2*fmb*p1p9+fmc*p1p9-hbcm*p1p9)))
      b(34)=4*ccc*w15*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(35)=ccc*(8*w7*(-fmb*p7p11-fmc*p7p11)+8*w15*(fmb*p7p10-fmb*
     . p7p9+fmc*p7p10-fmc*p7p9))
      b(36)=4*ccc*w15*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p2p3-4*p3p4)
      b(37)=w15*ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(38)=ccc*(w7*(16*p3p7*(-p1p11-p2p11+p5p11)+8*(-ffmcfmb*hbcm2*
     . p7p11+fmc*hbcm*p7p11+hbcm2*p7p11+2*p3p4*p7p11))+w15*(16*p3p4*(
     . -p7p10+p7p9)+16*p3p7*(p1p10-p1p9+p2p10-p2p9-p5p10+p5p9)+8*(
     . ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p9-fmc*hbcm*p7p10+fmc*hbcm
     . *p7p9-hbcm2*p7p10+hbcm2*p7p9))+8*p3p12)
      b(39)=w7*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(2*p1p3*
     . p6p7+p2p3*p6p7-2*p3p5*p6p7-2*p3p6*p3p7-2*p3p6*p4p7-p3p7*p4p6-
     . p3p7*p5p6))
      b(40)=w7*ccc*(16*(p5p6*p3p7)*(ffmcfmb*hbcm+fmb+fmc)+16*(hbcm*
     . p1p5*p6p7+hbcm*p5p6*p5p7))
      b(41)=16*ccc*w7*(2*p1p5*p6p7+p3p7*p5p6+2*p5p6*p5p7)
      b(42)=w15*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(2*p1p3*
     . p6p7+p2p3*p6p7-2*p3p5*p6p7-2*p3p6*p3p7-2*p3p6*p4p7-p3p7*p4p6-
     . p3p7*p5p6))
      b(43)=ccc*(w7*(16*p6p7*(2*p1p11+p2p11-2*p5p11)+16*(-p3p6*p7p11-
     . p3p7*p6p11-p4p6*p7p11-2*p4p7*p6p11-p5p6*p7p11))+w15*(16*p5p6*(
     . p7p10-p7p9)+32*p4p7*(p6p10-p6p9)+16*p4p6*(p7p10-p7p9)+16*p3p7*
     . (p6p10-p6p9)+16*p3p6*(p7p10-p7p9)+16*p6p7*(-2*p1p10+2*p1p9-
     . p2p10+p2p9+2*p5p10-2*p5p9))+8*(-p6p12+p7p13))
      b(44)=w7*ccc*(8*p5p6*(fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(2*p1p5*
     . p3p6+p2p3*p5p6-2*p3p4*p5p6))
      b(45)=16*ccc*w7*(-p3p6-p4p6-p5p6)
      b(46)=8*ccc*w7*p5p6*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(47)=16*ccc*w15*(-p3p6-p4p6-p5p6)
      b(49)=w7*ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(50)=8*ccc*w7*(p3p7+2*p5p7)
      b(51)=16*ccc*w7*(-p1p2*p3p7-2*p1p3*p5p7+p1p4*p3p7)
      b(52)=8*ccc*w15*(p3p7+2*p5p7)
      b(53)=ccc*(8*w7*(-p2p11+2*p3p11+2*p4p11)+8*w15*(p2p10-p2p9-2*
     . p3p10+2*p3p9-2*p4p10+2*p4p9))
      b(54)=4*ccc*w7*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p2p3-4*p3p4)
      b(55)=4*ccc*w7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(56)=16*ccc*w7*(-p1p2+p1p3+p1p4)
      b(58)=w7*ccc*(8*p3p6*(ffmcfmb*hbcm+2*fmb+fmc)+8*(hbcm*p4p6+hbcm
     . *p5p6))
      b(59)=w7*ccc*(8*p1p3*(-ffmcfmb*hbcm-2*fmb-fmc)+8*(hbcm*p1p2-
     . hbcm*p1p4))
      b(60)=w15*ccc*(8*p3p6*(ffmcfmb*hbcm+2*fmb+fmc)+8*(hbcm*p4p6+
     . hbcm*p5p6))
      b(61)=4*ccc*hbcm
      b(62)=8*ccc*hbcm*p5p6
      b(63)=w15*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p6p7-p2p3*p6p7+2*p3p5*p6p7+2*p3p6*p3p7+2*p3p6*p4p7+p3p7*p4p6+
     . p3p7*p5p6))
      b(64)=16*ccc*w15*(-2*p1p5*p6p7-p3p7*p5p6-2*p5p6*p5p7)
      b(66)=ccc*(w7*(16*p6p7*(hbcm*p1p11+hbcm*p2p11-hbcm*p5p11)+16*
     . p3p6*(-fmb*p7p11-fmc*p7p11)+16*(-fmb*p3p7*p6p11-hbcm*p4p7*
     . p6p11))+w15*(16*p4p7*(hbcm*p6p10-hbcm*p6p9)+16*p6p7*(-hbcm*
     . p1p10+hbcm*p1p9-hbcm*p2p10+hbcm*p2p9+hbcm*p5p10-hbcm*p5p9)+16*
     . p3p7*(fmb*p6p10-fmb*p6p9)+16*p3p6*(fmb*p7p10-fmb*p7p9+fmc*
     . p7p10-fmc*p7p9))+8*hbcm*p7p13)
      b(67)=16*ccc*w15*(p3p6+p4p6+p5p6)
      b(69)=8*ccc*w15*(-p3p7-2*p5p7)
      b(70)=16*ccc*w15*(p1p2*p3p7+2*p1p3*p5p7-p1p4*p3p7)
      b(72)=16*ccc*w15*(p1p2-p1p3-p1p4)
      b(73)=8*ccc*w7*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(75)=ccc*(8*w7*(ffmcfmb*hbcm*p1p11+2*fmb*p1p11+fmc*p1p11-hbcm*
     . p1p11)+8*w15*(-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p1p9-2*fmb*
     . p1p10+2*fmb*p1p9-fmc*p1p10+fmc*p1p9+hbcm*p1p10-hbcm*p1p9))
      b(76)=ccc*(w7*(16*p1p3*(2*p1p11+p2p11-2*p3p11-2*p4p11)+8*(-fmb*
     . hbcm*p1p11+fmc*hbcm*p1p11-2*hbcm2*p1p11+2*p1p11*p2p3-4*p1p11*
     . p3p5))+w15*(32*p3p5*(p1p10-p1p9)+16*p2p3*(-p1p10+p1p9)+16*p1p3
     . *(-2*p1p10+2*p1p9-p2p10+p2p9+2*p3p10-2*p3p9+2*p4p10-2*p4p9)+8*
     . (fmb*hbcm*p1p10-fmb*hbcm*p1p9-fmc*hbcm*p1p10+fmc*hbcm*p1p9+2*
     . hbcm2*p1p10-2*hbcm2*p1p9)))
      b(77)=ccc*(w7*(16*p3p6*(-2*p1p11-p2p11+2*p5p11)+8*(-fmb*hbcm*
     . p6p11+fmc*hbcm*p6p11-2*hbcm2*p6p11+4*p1p3*p6p11+2*p2p3*p6p11-4
     . *p3p5*p6p11))+w15*(32*p3p5*(p6p10-p6p9)+16*p2p3*(-p6p10+p6p9)+
     . 32*p1p3*(-p6p10+p6p9)+16*p3p6*(2*p1p10-2*p1p9+p2p10-p2p9-2*
     . p5p10+2*p5p9)+8*(fmb*hbcm*p6p10-fmb*hbcm*p6p9-fmc*hbcm*p6p10+
     . fmc*hbcm*p6p9+2*hbcm2*p6p10-2*hbcm2*p6p9)))
      b(78)=ccc*(8*w7*(ffmcfmb*hbcm*p6p11+2*fmb*p6p11+fmc*p6p11-hbcm*
     . p6p11)+8*w15*(-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p9-2*fmb*
     . p6p10+2*fmb*p6p9-fmc*p6p10+fmc*p6p9+hbcm*p6p10-hbcm*p6p9))
      b(79)=8*ccc*w15*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(81)=8*ccc*w15*p6p7*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      do 200 n=1,93 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp19_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=w11*ccc*((p5p7*p3p6)*(-32*ffmcfmb+32)+(p3p7*p3p6)*(-16*
     . ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb
     . *fmc*hbcm+2*ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+2*fmb2+fmc*hbcm-
     . hbcm2)+16*(p1p2*p6p7-p1p3*p6p7-3*p1p5*p6p7))
      ans=ccc*(w7*(16*p3p7*(-ffmcfmb*p1p11-p1p11)+8*(ffmcfmb**2*hbcm2
     . *p7p11-ffmcfmb*fmb*hbcm*p7p11+ffmcfmb*fmc*hbcm*p7p11-2*ffmcfmb
     . *hbcm2*p7p11+fmb*fmc*p7p11+fmb*hbcm*p7p11-2*fmb2*p7p11-fmc*
     . hbcm*p7p11+hbcm2*p7p11-6*p1p11*p4p7+4*p1p2*p7p11-4*p1p3*p7p11-
     . 6*p1p4*p7p11))+w11*(48*p1p4*(-p7p11+p7p8)+32*p1p3*(-p7p11+p7p8
     . )+32*p1p2*(p7p11-p7p8)+48*p4p7*(-p1p11+p1p8)+16*p3p7*(-ffmcfmb
     . *p1p11+ffmcfmb*p1p8-p1p11+p1p8)+8*(ffmcfmb**2*hbcm2*p7p11-
     . ffmcfmb**2*hbcm2*p7p8-ffmcfmb*fmb*hbcm*p7p11+ffmcfmb*fmb*hbcm*
     . p7p8+ffmcfmb*fmc*hbcm*p7p11-ffmcfmb*fmc*hbcm*p7p8-2*ffmcfmb*
     . hbcm2*p7p11+2*ffmcfmb*hbcm2*p7p8+fmb*fmc*p7p11-fmb*fmc*p7p8+
     . fmb*hbcm*p7p11-fmb*hbcm*p7p8-2*fmb2*p7p11+2*fmb2*p7p8-fmc*hbcm
     . *p7p11+fmc*hbcm*p7p8+hbcm2*p7p11-hbcm2*p7p8))+w13*(16*p3p7*(
     . ffmcfmb*p1p10+p1p10)+8*(-ffmcfmb**2*hbcm2*p7p10+ffmcfmb*fmb*
     . hbcm*p7p10-ffmcfmb*fmc*hbcm*p7p10+2*ffmcfmb*hbcm2*p7p10-fmb*
     . fmc*p7p10-fmb*hbcm*p7p10+2*fmb2*p7p10+fmc*hbcm*p7p10-hbcm2*
     . p7p10+6*p1p10*p4p7-4*p1p2*p7p10+4*p1p3*p7p10+6*p1p4*p7p10))+8*
     . (-3*p1p12+p3p12+p4p12+p5p12+p7p10-p7p11))
      b(3)=ans
      b(4)=w11*ccc*(16*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p7*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2+fmb*fmc+2*fmb*hbcm-2*fmb2-fmc*hbcm)+16*(-ffmcfmb*p1p3*
     . p3p7+2*p1p2*p3p7+3*p1p3*p5p7-3*p1p4*p3p7))
      ans3=w13*(32*p3p4*(-ffmcfmb*hbcm*p7p10-fmb*p7p10+hbcm*p7p10)+16
     . *p2p3*(ffmcfmb*hbcm*p7p10+fmb*p7p10-hbcm*p7p10)+16*p1p3*(
     . ffmcfmb*hbcm*p7p10-fmc*p7p10-2*hbcm*p7p10)+16*p3p7*(ffmcfmb*
     . hbcm*p1p10+ffmcfmb*hbcm*p2p10-2*ffmcfmb*hbcm*p5p10+2*fmb*p1p10
     . +fmb*p2p10-2*fmb*p5p10+fmc*p1p10-hbcm*p2p10+2*hbcm*p5p10)+8*(-
     . ffmcfmb*fmb*hbcm2*p7p10-ffmcfmb*fmc*hbcm2*p7p10-2*ffmcfmb*
     . hbcm3*p7p10-fmb*fmc*hbcm*p7p10-fmb*hbcm2*p7p10-fmb2*hbcm*p7p10
     . +fmc*hbcm2*p7p10+4*hbcm*p1p10*p5p7+2*hbcm*p1p2*p7p10-4*hbcm*
     . p1p4*p7p10+2*hbcm3*p7p10))+8*(2*fmb*p3p12+2*fmc*p3p12-hbcm*
     . p2p12-3*hbcm*p3p12-hbcm*p4p12-hbcm*p5p12+hbcm*p7p10+2*hbcm*
     . p7p11)
      ans2=w11*(32*p1p4*(hbcm*p7p11-hbcm*p7p8)+16*p1p2*(-hbcm*p7p11+
     . hbcm*p7p8)+32*p5p7*(-hbcm*p1p11+hbcm*p1p8)+32*p3p4*(ffmcfmb*
     . hbcm*p7p11-ffmcfmb*hbcm*p7p8+fmb*p7p11-fmb*p7p8-hbcm*p7p11+
     . hbcm*p7p8)+16*p2p3*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8-fmb*
     . p7p11+fmb*p7p8+hbcm*p7p11-hbcm*p7p8)+16*p1p3*(-ffmcfmb*hbcm*
     . p7p11+ffmcfmb*hbcm*p7p8+fmc*p7p11-fmc*p7p8+2*hbcm*p7p11-2*hbcm
     . *p7p8)+16*p3p7*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p1p8-ffmcfmb*
     . hbcm*p2p11+ffmcfmb*hbcm*p2p8+2*ffmcfmb*hbcm*p5p11-2*ffmcfmb*
     . hbcm*p5p8-2*fmb*p1p11+2*fmb*p1p8-fmb*p2p11+fmb*p2p8+2*fmb*
     . p5p11-2*fmb*p5p8-fmc*p1p11+fmc*p1p8+hbcm*p2p11-hbcm*p2p8-2*
     . hbcm*p5p11+2*hbcm*p5p8)+8*(ffmcfmb*fmb*hbcm2*p7p11-ffmcfmb*fmb
     . *hbcm2*p7p8+ffmcfmb*fmc*hbcm2*p7p11-ffmcfmb*fmc*hbcm2*p7p8+2*
     . ffmcfmb*hbcm3*p7p11-2*ffmcfmb*hbcm3*p7p8+fmb*fmc*hbcm*p7p11-
     . fmb*fmc*hbcm*p7p8+fmb*hbcm2*p7p11-fmb*hbcm2*p7p8+fmb2*hbcm*
     . p7p11-fmb2*hbcm*p7p8-fmc*hbcm2*p7p11+fmc*hbcm2*p7p8-2*hbcm3*
     . p7p11+2*hbcm3*p7p8))+ans3
      ans1=w7*(32*p3p4*(ffmcfmb*hbcm*p7p11+fmb*p7p11-hbcm*p7p11)+16*
     . p2p3*(-ffmcfmb*hbcm*p7p11-fmb*p7p11+hbcm*p7p11)+16*p1p3*(-
     . ffmcfmb*hbcm*p7p11+fmc*p7p11+2*hbcm*p7p11)+16*p3p7*(-ffmcfmb*
     . hbcm*p1p11-ffmcfmb*hbcm*p2p11+2*ffmcfmb*hbcm*p5p11-2*fmb*p1p11
     . -fmb*p2p11+2*fmb*p5p11-fmc*p1p11+hbcm*p2p11-2*hbcm*p5p11)+8*(
     . ffmcfmb*fmb*hbcm2*p7p11+ffmcfmb*fmc*hbcm2*p7p11+2*ffmcfmb*
     . hbcm3*p7p11+fmb*fmc*hbcm*p7p11+fmb*hbcm2*p7p11+fmb2*hbcm*p7p11
     . -fmc*hbcm2*p7p11-4*hbcm*p1p11*p5p7-2*hbcm*p1p2*p7p11+4*hbcm*
     . p1p4*p7p11-2*hbcm3*p7p11))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(16*p3p6*(-ffmcfmb*p2p10+2*ffmcfmb*p3p10+2*ffmcfmb*
     . p4p10+p1p10+p2p10-2*p3p10-2*p4p10)+8*(-ffmcfmb**2*hbcm2*p6p10+
     . ffmcfmb*fmb*hbcm*p6p10-ffmcfmb*fmc*hbcm*p6p10+2*ffmcfmb*hbcm2*
     . p6p10-fmb*fmc*p6p10-fmb*hbcm*p6p10+2*fmb2*p6p10+fmc*hbcm*p6p10
     . -hbcm2*p6p10+2*p1p10*p4p6+2*p1p2*p6p10-2*p1p3*p6p10-6*p1p5*
     . p6p10))+8*(3*p2p13-4*p3p13-3*p4p13-p5p13+p6p10-p6p11)
      ans1=w7*(16*p3p6*(ffmcfmb*p2p11-2*ffmcfmb*p3p11-2*ffmcfmb*p4p11
     . -p1p11-p2p11+2*p3p11+2*p4p11)+8*(ffmcfmb**2*hbcm2*p6p11-
     . ffmcfmb*fmb*hbcm*p6p11+ffmcfmb*fmc*hbcm*p6p11-2*ffmcfmb*hbcm2*
     . p6p11+fmb*fmc*p6p11+fmb*hbcm*p6p11-2*fmb2*p6p11-fmc*hbcm*p6p11
     . +hbcm2*p6p11-2*p1p11*p4p6-2*p1p2*p6p11+2*p1p3*p6p11+6*p1p5*
     . p6p11))+w11*(48*p1p5*(p6p11-p6p8)+16*p1p3*(p6p11-p6p8)+16*p1p2
     . *(-p6p11+p6p8)+16*p4p6*(-p1p11+p1p8)+16*p3p6*(ffmcfmb*p2p11-
     . ffmcfmb*p2p8-2*ffmcfmb*p3p11+2*ffmcfmb*p3p8-2*ffmcfmb*p4p11+2*
     . ffmcfmb*p4p8-p1p11+p1p8-p2p11+p2p8+2*p3p11-2*p3p8+2*p4p11-2*
     . p4p8)+8*(ffmcfmb**2*hbcm2*p6p11-ffmcfmb**2*hbcm2*p6p8-ffmcfmb*
     . fmb*hbcm*p6p11+ffmcfmb*fmb*hbcm*p6p8+ffmcfmb*fmc*hbcm*p6p11-
     . ffmcfmb*fmc*hbcm*p6p8-2*ffmcfmb*hbcm2*p6p11+2*ffmcfmb*hbcm2*
     . p6p8+fmb*fmc*p6p11-fmb*fmc*p6p8+fmb*hbcm*p6p11-fmb*hbcm*p6p8-2
     . *fmb2*p6p11+2*fmb2*p6p8-fmc*hbcm*p6p11+fmc*hbcm*p6p8+hbcm2*
     . p6p11-hbcm2*p6p8))+ans2
      ans=ccc*ans1
      b(7)=ans
      b(8)=w11*ccc*((p3p6*p3p4)*(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*
     . ffmcfmb-16)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(
     . ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm-2*fmb2+2*
     . hbcm2)+16*(-p1p2*p3p6-p1p3*p4p6+3*p1p5*p3p6))
      ans4=8*(ffmcfmb*fmb*hbcm2*p6p11-ffmcfmb*fmb*hbcm2*p6p8+ffmcfmb*
     . fmc*hbcm2*p6p11-ffmcfmb*fmc*hbcm2*p6p8-2*ffmcfmb*hbcm3*p6p11+2
     . *ffmcfmb*hbcm3*p6p8+fmb*fmc*hbcm*p6p11-fmb*fmc*hbcm*p6p8-3*fmb
     . *hbcm2*p6p11+3*fmb*hbcm2*p6p8+fmb2*hbcm*p6p11-fmb2*hbcm*p6p8-
     . fmc*hbcm2*p6p11+fmc*hbcm2*p6p8+2*hbcm3*p6p11-2*hbcm3*p6p8)
      ans3=32*p1p5*(-hbcm*p6p11+hbcm*p6p8)+16*p1p2*(hbcm*p6p11-hbcm*
     . p6p8)+16*p4p6*(-3*ffmcfmb*hbcm*p3p8-3*fmb*p3p8-2*hbcm*p1p11+2*
     . hbcm*p1p8+3*hbcm*p3p8)+32*p3p5*(-ffmcfmb*hbcm*p6p11+ffmcfmb*
     . hbcm*p6p8-fmb*p6p11+fmb*p6p8+hbcm*p6p11-hbcm*p6p8)+16*p2p3*(
     . ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-hbcm*
     . p6p11+hbcm*p6p8)+16*p1p3*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8
     . +fmb*p6p11-fmb*p6p8-2*hbcm*p6p11+2*hbcm*p6p8)+16*p3p6*(-
     . ffmcfmb**2*hbcm*p3p8-2*ffmcfmb*fmb*p3p8-ffmcfmb*fmc*p3p8+
     . ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p3p11+
     . ffmcfmb*hbcm*p3p8-ffmcfmb*hbcm*p4p11+ffmcfmb*hbcm*p4p8+fmb*
     . p2p11-fmb*p2p8-2*fmb*p3p11+2*fmb*p3p8-2*fmb*p4p11+2*fmb*p4p8-
     . fmc*p1p11+fmc*p1p8+fmc*p3p8-hbcm*p1p11+hbcm*p1p8-hbcm*p2p11+
     . hbcm*p2p8+hbcm*p3p11+hbcm*p4p11-hbcm*p4p8)+ans4
      ans2=w11*ans3
      ans5=w13*(32*p3p5*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+16*
     . p4p6*(-3*ffmcfmb*hbcm*p3p10-3*fmb*p3p10+2*hbcm*p1p10+3*hbcm*
     . p3p10)+16*p2p3*(-ffmcfmb*hbcm*p6p10-fmb*p6p10+hbcm*p6p10)+16*
     . p1p3*(-ffmcfmb*hbcm*p6p10-fmb*p6p10+2*hbcm*p6p10)+16*p3p6*(-
     . ffmcfmb**2*hbcm*p3p10-2*ffmcfmb*fmb*p3p10-ffmcfmb*fmc*p3p10-
     . ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p3p10+ffmcfmb*hbcm*p4p10-fmb*
     . p2p10+2*fmb*p3p10+2*fmb*p4p10+fmc*p1p10+fmc*p3p10+hbcm*p1p10+
     . hbcm*p2p10-hbcm*p4p10)+8*(-ffmcfmb*fmb*hbcm2*p6p10-ffmcfmb*fmc
     . *hbcm2*p6p10+2*ffmcfmb*hbcm3*p6p10-fmb*fmc*hbcm*p6p10+3*fmb*
     . hbcm2*p6p10-fmb2*hbcm*p6p10+fmc*hbcm2*p6p10-2*hbcm*p1p2*p6p10+
     . 4*hbcm*p1p5*p6p10-2*hbcm3*p6p10))+8*(3*fmb*p3p13+3*fmc*p3p13-2
     . *hbcm*p2p13+2*hbcm*p4p13+hbcm*p5p13-hbcm*p6p10+hbcm*p6p11)
      ans1=w7*(32*p3p5*(-ffmcfmb*hbcm*p6p11-fmb*p6p11+hbcm*p6p11)+16*
     . p2p3*(ffmcfmb*hbcm*p6p11+fmb*p6p11-hbcm*p6p11)+16*p3p6*(
     . ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p3p11-ffmcfmb*hbcm*p4p11+fmb*
     . p2p11-2*fmb*p3p11-2*fmb*p4p11-fmc*p1p11-hbcm*p1p11-hbcm*p2p11+
     . hbcm*p3p11+hbcm*p4p11)+16*p1p3*(ffmcfmb*hbcm*p6p11+fmb*p6p11-2
     . *hbcm*p6p11)+8*(ffmcfmb*fmb*hbcm2*p6p11+ffmcfmb*fmc*hbcm2*
     . p6p11-2*ffmcfmb*hbcm3*p6p11+fmb*fmc*hbcm*p6p11-3*fmb*hbcm2*
     . p6p11+fmb2*hbcm*p6p11-fmc*hbcm2*p6p11-4*hbcm*p1p11*p4p6+2*hbcm
     . *p1p2*p6p11-4*hbcm*p1p5*p6p11+2*hbcm3*p6p11))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      b(11)=w11*ccc*(32*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p7
     . *p3p6)*(ffmcfmb*hbcm+2*fmb-hbcm)+32*(p6p7*p3p5)*(-ffmcfmb*hbcm
     . -fmb+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*
     . p1p3)*(ffmcfmb*hbcm+fmb-2*hbcm)+8*p6p7*(ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm-3*fmb*hbcm2+
     . fmb2*hbcm-fmc*hbcm2+2*hbcm3)+16*(p3p7*p3p6)*(-ffmcfmb**2*hbcm-
     . ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm+fmb+fmc)+16*(hbcm*p1p2*
     . p6p7-2*hbcm*p1p5*p6p7))
      b(13)=4*ccc*w11*(2*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm-4*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm-3*fmb2-fmc*hbcm+2*
     . hbcm2+6*p1p2-6*p1p3-8*p1p4)
      b(15)=w11*ccc*(24*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+16*p2p3*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmc+3*hbcm)+4*(
     . ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm-fmc*hbcm2-4*
     . hbcm*p1p2+6*hbcm*p1p4-3*hbcm3))
      b(17)=w13*ccc*(8*p3p6*(-ffmcfmb*hbcm-fmb-fmc-hbcm)-24*hbcm*p4p6
     . )
      b(18)=4*ccc*w13*(fmb+fmc)
      b(19)=4*ccc*w13*(4*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+2*hbcm2-2*
     . p1p3-6*p2p3+8*p3p4)
      b(20)=w13*ccc*(8*p3p7*(ffmcfmb*hbcm-fmb-fmc-2*hbcm)-24*hbcm*
     . p5p7)
      b(21)=ccc*(8*w7*(-hbcm*p1p11-2*hbcm*p2p11+3*hbcm*p3p11+3*hbcm*
     . p4p11)+8*w11*(-fmb*p3p8-fmc*p3p8-hbcm*p1p11+hbcm*p1p8-2*hbcm*
     . p2p11+2*hbcm*p2p8+3*hbcm*p3p11-3*hbcm*p3p8+3*hbcm*p4p11-3*hbcm
     . *p4p8)+8*w13*(-fmb*p3p10-fmc*p3p10+hbcm*p1p10+2*hbcm*p2p10-3*
     . hbcm*p3p10-3*hbcm*p4p10))
      b(22)=w13*ccc*(24*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+16*p2p3*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmc+3*hbcm)+4*(
     . ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm-fmc*hbcm2-4*
     . hbcm*p1p2+6*hbcm*p1p4-3*hbcm3))
      b(23)=w11*ccc*(8*p3p6*(-ffmcfmb*hbcm-fmb-fmc-hbcm)-24*hbcm*p4p6
     . )
      b(24)=4*ccc*w11*(fmb+fmc)
      b(25)=4*ccc*w11*(4*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+2*hbcm2-2*
     . p1p3-6*p2p3+8*p3p4)
      b(26)=w11*ccc*(8*p3p7*(ffmcfmb*hbcm-fmb-fmc-2*hbcm)-24*hbcm*
     . p5p7)
      b(27)=w13*ccc*(24*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb
     . **2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-fmc-hbcm))
      b(28)=w13*ccc*((p3p6*p3p4)*(32*ffmcfmb-32)+(p3p6*p2p3)*(-16*
     . ffmcfmb+16)+16*p4p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p6*(-
     . ffmcfmb**2*hbcm2+3*ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+2*fmb2-2*
     . hbcm2)+16*(p1p2*p3p6+p1p3*p4p6-3*p1p5*p3p6))
      b(29)=w13*ccc*(32*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p7
     . *p3p6)*(ffmcfmb*hbcm+2*fmb-hbcm)+32*(p6p7*p3p5)*(-ffmcfmb*hbcm
     . -fmb+hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*
     . p1p3)*(ffmcfmb*hbcm+fmb-2*hbcm)+8*p6p7*(ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm-3*fmb*hbcm2+
     . fmb2*hbcm-fmc*hbcm2+2*hbcm3)+16*(p3p7*p3p6)*(-ffmcfmb**2*hbcm-
     . ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm+fmb+fmc)+16*(hbcm*p1p2*
     . p6p7-2*hbcm*p1p5*p6p7))
      ans4=16*p6p7*(ffmcfmb*hbcm2*p2p11-ffmcfmb*hbcm2*p2p8-ffmcfmb*
     . hbcm2*p5p11+ffmcfmb*hbcm2*p5p8+fmb*hbcm*p1p11-fmb*hbcm*p1p8+
     . fmb*hbcm*p2p11-fmb*hbcm*p2p8-fmb*hbcm*p5p11+fmb*hbcm*p5p8-fmc*
     . hbcm*p1p11+fmc*hbcm*p1p8+4*hbcm2*p1p11-4*hbcm2*p1p8-hbcm2*
     . p2p11+hbcm2*p2p8+hbcm2*p5p11-hbcm2*p5p8)+16*p3p7*(ffmcfmb*fmb*
     . hbcm*p6p11-ffmcfmb*fmb*hbcm*p6p8+ffmcfmb*hbcm2*p6p11-ffmcfmb*
     . hbcm2*p6p8+fmb2*p6p11-fmb2*p6p8-hbcm2*p6p11+hbcm2*p6p8)+16*
     . p3p6*(-ffmcfmb**2*hbcm2*p7p11+ffmcfmb*fmb*hbcm*p7p11-ffmcfmb*
     . fmb*hbcm*p7p8+2*ffmcfmb*hbcm2*p7p11+ffmcfmb*hbcm2*p7p8-fmb*fmc
     . *p7p11-fmb*hbcm*p7p11+fmb2*p7p11+fmb2*p7p8-hbcm2*p7p11-hbcm2*
     . p7p8)+32*(p1p2*p3p6*p7p11+p1p3*p4p6*p7p11)
      ans3=32*(p3p7*p1p5)*(-p6p11+p6p8)+32*(p3p6*p1p5)*(-2*p7p11-p7p8
     . )+32*(p5p7*p1p3)*(-p6p11+p6p8)+96*(p4p6*p3p7)*(p1p11-p1p8)+32*
     . (p5p7*p3p6)*(p1p11-p1p8)+96*(p4p7*p3p6)*(p1p11-p1p8)+96*(p6p7*
     . p3p5)*(p1p11-p1p8)+32*(p6p7*p3p4)*(p1p11-p1p8)+64*(p6p7*p2p3)*
     . (-p1p11+p1p8)+32*(p6p7*p1p3)*(-3*p1p11+3*p1p8+p5p11-p5p8)+32*(
     . p3p6*p3p4)*(ffmcfmb*p7p11+ffmcfmb*p7p8-p7p11-p7p8)+32*(p3p6*
     . p2p3)*(-ffmcfmb*p7p8+p7p8)+32*(p3p7*p3p6)*(-2*ffmcfmb*p2p11+2*
     . ffmcfmb*p2p8+2*ffmcfmb*p3p11-2*ffmcfmb*p3p8+2*ffmcfmb*p4p11-2*
     . ffmcfmb*p4p8+ffmcfmb*p5p11-ffmcfmb*p5p8+4*p1p11-4*p1p8+2*p2p11
     . -2*p2p8-2*p3p11+2*p3p8-2*p4p11+2*p4p8-p5p11+p5p8)+32*(p3p7*
     . p1p3)*(ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+16*p5p7*(ffmcfmb
     . *hbcm2*p6p11-ffmcfmb*hbcm2*p6p8+fmb*hbcm*p6p11-fmb*hbcm*p6p8-
     . hbcm2*p6p11+hbcm2*p6p8)+16*p4p6*(-ffmcfmb*hbcm2*p7p11-ffmcfmb*
     . hbcm2*p7p8-fmb*hbcm*p7p11-fmb*hbcm*p7p8+hbcm2*p7p11+hbcm2*p7p8
     . )+ans4
      ans2=w11*ans3
      ans6=(16*p3p4*(-p6p12+3*p7p13)+16*p6p7*(p3p10-2*p3p11)+16*p2p3*
     . (2*p6p12-3*p7p13)+16*p3p7*(-ffmcfmb*p3p13-2*p1p13-3*p2p13+4*
     . p3p13+4*p4p13+3*p5p13-p6p11)+16*p3p6*(ffmcfmb*p3p12+p1p12-
     . p2p12-2*p3p12-p4p12-p5p12-p7p10+2*p7p11)+8*(ffmcfmb*hbcm2*
     . p6p12-ffmcfmb*hbcm2*p7p13+2*fmb*hbcm*p6p12-fmb*hbcm*p7p13-fmc*
     . hbcm*p6p12+2*fmc*hbcm*p7p13-2*hbcm2*p6p12+4*hbcm2*p7p13-2*
     . p3p13*p4p7+2*p3p5*p7p13))
      ans5=w13*(32*(p6p7*p1p3)*(3*p1p10-p5p10)+32*(p3p6*p3p4)*(-
     . ffmcfmb*p7p10+p7p10)+32*(p3p7*p3p6)*(2*ffmcfmb*p2p10-2*ffmcfmb
     . *p3p10-2*ffmcfmb*p4p10-ffmcfmb*p5p10-4*p1p10-2*p2p10+2*p3p10+2
     . *p4p10+p5p10)+32*(p3p7*p1p3)*(-ffmcfmb*p6p10+p6p10)+16*p5p7*(-
     . ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10+hbcm2*p6p10)+16*p4p6*(
     . ffmcfmb*hbcm2*p7p10+fmb*hbcm*p7p10-hbcm2*p7p10)+16*p6p7*(-
     . ffmcfmb*hbcm2*p2p10+ffmcfmb*hbcm2*p5p10-fmb*hbcm*p1p10-fmb*
     . hbcm*p2p10+fmb*hbcm*p5p10+fmc*hbcm*p1p10-4*hbcm2*p1p10+hbcm2*
     . p2p10-hbcm2*p5p10)+16*p3p7*(-ffmcfmb*fmb*hbcm*p6p10-ffmcfmb*
     . hbcm2*p6p10-fmb2*p6p10+hbcm2*p6p10)+16*p3p6*(ffmcfmb**2*hbcm2*
     . p7p10-ffmcfmb*fmb*hbcm*p7p10-2*ffmcfmb*hbcm2*p7p10+fmb*fmc*
     . p7p10+fmb*hbcm*p7p10-fmb2*p7p10+hbcm2*p7p10)+32*(2*p1p10*p2p3*
     . p6p7-p1p10*p3p4*p6p7-3*p1p10*p3p5*p6p7-3*p1p10*p3p6*p4p7-p1p10
     . *p3p6*p5p7-3*p1p10*p3p7*p4p6-p1p2*p3p6*p7p10-p1p3*p4p6*p7p10+
     . p1p3*p5p7*p6p10+2*p1p5*p3p6*p7p10+p1p5*p3p7*p6p10))+ans6
      ans1=w7*(32*(p6p7*p1p3)*(-3*p1p11+p5p11)+32*(p3p6*p3p4)*(
     . ffmcfmb*p7p11-p7p11)+32*(p3p7*p3p6)*(-2*ffmcfmb*p2p11+2*
     . ffmcfmb*p3p11+2*ffmcfmb*p4p11+ffmcfmb*p5p11+4*p1p11+2*p2p11-2*
     . p3p11-2*p4p11-p5p11)+32*(p3p7*p1p3)*(ffmcfmb*p6p11-p6p11)+16*
     . p5p7*(ffmcfmb*hbcm2*p6p11+fmb*hbcm*p6p11-hbcm2*p6p11)+16*p4p6*
     . (-ffmcfmb*hbcm2*p7p11-fmb*hbcm*p7p11+hbcm2*p7p11)+16*p6p7*(
     . ffmcfmb*hbcm2*p2p11-ffmcfmb*hbcm2*p5p11+fmb*hbcm*p1p11+fmb*
     . hbcm*p2p11-fmb*hbcm*p5p11-fmc*hbcm*p1p11+4*hbcm2*p1p11-hbcm2*
     . p2p11+hbcm2*p5p11)+16*p3p7*(ffmcfmb*fmb*hbcm*p6p11+ffmcfmb*
     . hbcm2*p6p11+fmb2*p6p11-hbcm2*p6p11)+16*p3p6*(-ffmcfmb**2*hbcm2
     . *p7p11+ffmcfmb*fmb*hbcm*p7p11+2*ffmcfmb*hbcm2*p7p11-fmb*fmc*
     . p7p11-fmb*hbcm*p7p11+fmb2*p7p11-hbcm2*p7p11)+32*(-2*p1p11*p2p3
     . *p6p7+p1p11*p3p4*p6p7+3*p1p11*p3p5*p6p7+3*p1p11*p3p6*p4p7+
     . p1p11*p3p6*p5p7+3*p1p11*p3p7*p4p6+p1p2*p3p6*p7p11+p1p3*p4p6*
     . p7p11-p1p3*p5p7*p6p11-2*p1p5*p3p6*p7p11-p1p5*p3p7*p6p11))+ans2
     . +ans5
      ans=ccc*ans1
      b(30)=ans
      b(31)=w11*ccc*(24*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb
     . **2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-fmc-hbcm))
      ans3=w13*(32*p4p7*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+32*
     . p4p6*(ffmcfmb*hbcm*p7p10+fmb*p7p10-hbcm*p7p10)+16*p3p7*(
     . ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+16*p6p7*(-ffmcfmb*
     . hbcm*p1p10-ffmcfmb*hbcm*p2p10+2*ffmcfmb*hbcm*p5p10-fmb*p1p10-
     . fmb*p2p10+2*fmb*p5p10+hbcm*p1p10+hbcm*p2p10-2*hbcm*p5p10)+16*
     . p3p6*(ffmcfmb*fmb*p7p10+ffmcfmb*fmc*p7p10+ffmcfmb*hbcm*p7p10-
     . fmc*p7p10-hbcm*p7p10))+8*(-fmb*p7p13-fmc*p7p13)
      ans2=w11*(32*p4p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8-fmb*
     . p6p11+fmb*p6p8+hbcm*p6p11-hbcm*p6p8)+32*p4p6*(-ffmcfmb*hbcm*
     . p7p11+ffmcfmb*hbcm*p7p8-fmb*p7p11+fmb*p7p8+hbcm*p7p11-hbcm*
     . p7p8)+16*p3p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8-fmb*p6p11
     . +fmb*p6p8+hbcm*p6p11-hbcm*p6p8)+16*p6p7*(ffmcfmb*hbcm*p1p11-
     . ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p2p8-2*
     . ffmcfmb*hbcm*p5p11+2*ffmcfmb*hbcm*p5p8+fmb*p1p11-fmb*p1p8+fmb*
     . p2p11-fmb*p2p8-2*fmb*p5p11+2*fmb*p5p8-hbcm*p1p11+hbcm*p1p8-
     . hbcm*p2p11+hbcm*p2p8+2*hbcm*p5p11-2*hbcm*p5p8)+16*p3p6*(-
     . ffmcfmb*fmb*p7p11+ffmcfmb*fmb*p7p8-ffmcfmb*fmc*p7p11+ffmcfmb*
     . fmc*p7p8-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8+fmc*p7p11-fmc*
     . p7p8+hbcm*p7p11-hbcm*p7p8))+ans3
      ans1=w7*(32*p4p7*(-ffmcfmb*hbcm*p6p11-fmb*p6p11+hbcm*p6p11)+32*
     . p4p6*(-ffmcfmb*hbcm*p7p11-fmb*p7p11+hbcm*p7p11)+16*p3p7*(-
     . ffmcfmb*hbcm*p6p11-fmb*p6p11+hbcm*p6p11)+16*p6p7*(ffmcfmb*hbcm
     . *p1p11+ffmcfmb*hbcm*p2p11-2*ffmcfmb*hbcm*p5p11+fmb*p1p11+fmb*
     . p2p11-2*fmb*p5p11-hbcm*p1p11-hbcm*p2p11+2*hbcm*p5p11)+16*p3p6*
     . (-ffmcfmb*fmb*p7p11-ffmcfmb*fmc*p7p11-ffmcfmb*hbcm*p7p11+fmc*
     . p7p11+hbcm*p7p11))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(35)=ccc*(8*w7*(-fmb*p7p11-fmc*p7p11)+8*w11*(-fmb*p7p11+fmb*
     . p7p8-fmc*p7p11+fmc*p7p8)+8*w13*(fmb*p7p10+fmc*p7p10))
      b(38)=ccc*(w7*(16*p3p7*(-2*p1p11-p2p11+3*p5p11)+8*(2*ffmcfmb*
     . hbcm2*p7p11-fmb*hbcm*p7p11+fmc*hbcm*p7p11+2*hbcm2*p7p11-2*p1p3
     . *p7p11-4*p2p3*p7p11+6*p3p4*p7p11))+w11*(48*p3p4*(p7p11-p7p8)+
     . 32*p2p3*(-p7p11+p7p8)+16*p1p3*(-p7p11+p7p8)+16*p3p7*(-2*p1p11+
     . 2*p1p8-p2p11+p2p8+3*p5p11-3*p5p8)+8*(2*ffmcfmb*hbcm2*p7p11-2*
     . ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p11+fmb*hbcm*p7p8+fmc*hbcm*p7p11
     . -fmc*hbcm*p7p8+2*hbcm2*p7p11-2*hbcm2*p7p8))+w13*(16*p3p7*(2*
     . p1p10+p2p10-3*p5p10)+8*(-2*ffmcfmb*hbcm2*p7p10+fmb*hbcm*p7p10-
     . fmc*hbcm*p7p10-2*hbcm2*p7p10+2*p1p3*p7p10+4*p2p3*p7p10-6*p3p4*
     . p7p10))+24*p3p12)
      b(39)=ccc*(w7*((p3p7*p3p6)*(-32*ffmcfmb-32)+8*p6p7*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-4*hbcm2)+16*(2*p1p3*p6p7+p2p3*p6p7-3*
     . p3p5*p6p7-3*p3p6*p4p7-3*p3p7*p4p6))+w11*((p3p7*p3p6)*(-32*
     . ffmcfmb-32)+8*p6p7*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-4*hbcm2)
     . +16*(2*p1p3*p6p7+p2p3*p6p7-3*p3p5*p6p7-3*p3p6*p4p7-3*p3p7*p4p6
     . )))
      ans=ccc*(w7*(32*(p4p6*p3p7)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p5p7*
     . p3p6)*(-ffmcfmb*hbcm-2*fmb+hbcm)+32*(p6p7*p3p5)*(ffmcfmb*hbcm+
     . fmb-hbcm)+16*(p6p7*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*
     . p1p3)*(-ffmcfmb*hbcm-fmb+2*hbcm)+8*p6p7*(-ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*hbcm+3*fmb*hbcm2-
     . fmb2*hbcm+fmc*hbcm2-2*hbcm3)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmb-fmc)+16*(-hbcm*p1p2*
     . p6p7+2*hbcm*p1p5*p6p7))+w11*(32*(p4p6*p3p7)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm-2*fmb+hbcm)+32*(p6p7*p3p5)
     . *(ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*p2p3)*(-ffmcfmb*hbcm-fmb+
     . hbcm)+16*(p6p7*p1p3)*(-ffmcfmb*hbcm-fmb+2*hbcm)+8*p6p7*(-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+3*fmb*hbcm2-fmb2*hbcm+fmc*hbcm2-2*hbcm3)+16*(p3p7*p3p6)*(
     . ffmcfmb**2*hbcm+ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmb-fmc)+
     . 16*(-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7)))
      b(40)=ans
      b(41)=ccc*(w7*((p5p7*p3p6)*(32*ffmcfmb-32)+(p3p7*p3p6)*(16*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm-2*fmb2-fmc*hbcm+
     . hbcm2)+16*(-p1p2*p6p7+p1p3*p6p7+3*p1p5*p6p7))+w11*((p5p7*p3p6)
     . *(32*ffmcfmb-32)+(p3p7*p3p6)*(16*ffmcfmb-16)+8*p6p7*(ffmcfmb**
     . 2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmb*
     . fmc+fmb*hbcm-2*fmb2-fmc*hbcm+hbcm2)+16*(-p1p2*p6p7+p1p3*p6p7+3
     . *p1p5*p6p7)))
      b(43)=ccc*(w7*(16*p6p7*(2*p1p11+p2p11-3*p5p11)+16*p3p7*(-
     . ffmcfmb*p6p11-p6p11)+16*p3p6*(-ffmcfmb*p7p11-p7p11)+48*(-p4p6*
     . p7p11-p4p7*p6p11))+w11*(48*p4p7*(-p6p11+p6p8)+48*p4p6*(-p7p11+
     . p7p8)+16*p6p7*(2*p1p11-2*p1p8+p2p11-p2p8-3*p5p11+3*p5p8)+16*
     . p3p7*(-ffmcfmb*p6p11+ffmcfmb*p6p8-p6p11+p6p8)+16*p3p6*(-
     . ffmcfmb*p7p11+ffmcfmb*p7p8-p7p11+p7p8))+w13*(16*p6p7*(-2*p1p10
     . -p2p10+3*p5p10)+16*p3p7*(ffmcfmb*p6p10+p6p10)+16*p3p6*(ffmcfmb
     . *p7p10+p7p10)+48*(p4p6*p7p10+p4p7*p6p10))+8*(-p6p12-p7p13))
      b(44)=ccc*(w7*((p3p6*p3p4)*(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*
     . ffmcfmb-16)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(
     . ffmcfmb**2*hbcm2-3*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm-2*fmb2+2*
     . hbcm2)+16*(-p1p2*p3p6-p1p3*p4p6+3*p1p5*p3p6))+w11*((p3p6*p3p4)
     . *(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*ffmcfmb-16)+16*p4p6*(ffmcfmb
     . *hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2-3*ffmcfmb*
     . hbcm2+fmb*fmc+fmb*hbcm-2*fmb2+2*hbcm2)+16*(-p1p2*p3p6-p1p3*
     . p4p6+3*p1p5*p3p6)))
      b(45)=ccc*(w7*(p3p6*(-16*ffmcfmb-8)-32*p4p6)+w11*(p3p6*(-16*
     . ffmcfmb-8)-32*p4p6))
      b(46)=ccc*(w7*(24*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb
     . **2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-fmc-hbcm))+w11*(24*p4p6*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb**2*hbcm+2*ffmcfmb*fmb+
     . ffmcfmb*fmc-fmc-hbcm)))
      b(49)=ccc*(w7*(8*p3p7*(-ffmcfmb*hbcm+fmb+fmc+2*hbcm)+24*hbcm*
     . p5p7)+w11*(8*p3p7*(-ffmcfmb*hbcm+fmb+fmc+2*hbcm)+24*hbcm*p5p7)
     . )
      b(50)=ccc*(w7*(p3p7*(-16*ffmcfmb+24)+32*p5p7)+w11*(p3p7*(-16*
     . ffmcfmb+24)+32*p5p7))
      b(51)=ccc*(w7*(16*p5p7*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(
     . -ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-fmb*fmc-2*fmb*hbcm+2*fmb2+fmc*hbcm)+16*(ffmcfmb*p1p3*
     . p3p7-2*p1p2*p3p7-3*p1p3*p5p7+3*p1p4*p3p7))+w11*(16*p5p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2+
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-2*fmb*
     . hbcm+2*fmb2+fmc*hbcm)+16*(ffmcfmb*p1p3*p3p7-2*p1p2*p3p7-3*p1p3
     . *p5p7+3*p1p4*p3p7)))
      b(53)=ccc*(8*w7*(-p1p11-3*p2p11+4*p3p11+4*p4p11)+8*w11*(-p1p11+
     . p1p8-3*p2p11+3*p2p8+4*p3p11-4*p3p8+4*p4p11-4*p4p8)+8*w13*(
     . p1p10+3*p2p10-4*p3p10-4*p4p10))
      b(54)=ccc*(4*w7*(-4*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-2*hbcm2+2*
     . p1p3+6*p2p3-8*p3p4)+4*w11*(-4*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-
     . 2*hbcm2+2*p1p3+6*p2p3-8*p3p4))
      b(55)=ccc*(4*w7*(fmb+fmc)+4*w11*(fmb+fmc))
      b(56)=ccc*(4*w7*(-2*ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm+4*ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+3*fmb2+fmc*hbcm-2*
     . hbcm2-6*p1p2+6*p1p3+8*p1p4)+4*w11*(-2*ffmcfmb**2*hbcm2+ffmcfmb
     . *fmb*hbcm-ffmcfmb*fmc*hbcm+4*ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+3*
     . fmb2+fmc*hbcm-2*hbcm2-6*p1p2+6*p1p3+8*p1p4))
      b(57)=-8*ccc
      b(58)=ccc*(w7*(8*p3p6*(ffmcfmb*hbcm+fmb+fmc+hbcm)+24*hbcm*p4p6)
     . +w11*(8*p3p6*(ffmcfmb*hbcm+fmb+fmc+hbcm)+24*hbcm*p4p6))
      b(59)=ccc*(w7*(24*p3p4*(-ffmcfmb*hbcm-fmb+hbcm)+16*p2p3*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(ffmcfmb*hbcm-fmc-3*hbcm)+4*(-
     . ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm+fmc*hbcm2+4*
     . hbcm*p1p2-6*hbcm*p1p4+3*hbcm3))+w11*(24*p3p4*(-ffmcfmb*hbcm-
     . fmb+hbcm)+16*p2p3*(ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(ffmcfmb*hbcm
     . -fmc-3*hbcm)+4*(-ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm+
     . fmc*hbcm2+4*hbcm*p1p2-6*hbcm*p1p4+3*hbcm3)))
      b(61)=-4*ccc*hbcm
      b(62)=-8*ccc*fmb*p3p6
      b(63)=w13*ccc*((p3p7*p3p6)*(32*ffmcfmb+32)+8*p6p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+4*hbcm2)+16*(-2*p1p3*p6p7-p2p3*p6p7+3*
     . p3p5*p6p7+3*p3p6*p4p7+3*p3p7*p4p6))
      b(64)=w13*ccc*((p5p7*p3p6)*(-32*ffmcfmb+32)+(p3p7*p3p6)*(-16*
     . ffmcfmb+16)+8*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb
     . *fmc*hbcm+2*ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+2*fmb2+fmc*hbcm-
     . hbcm2)+16*(p1p2*p6p7-p1p3*p6p7-3*p1p5*p6p7))
      b(65)=w11*ccc*((p3p7*p3p6)*(32*ffmcfmb+32)+8*p6p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+4*hbcm2)+16*(-2*p1p3*p6p7-p2p3*p6p7+3*
     . p3p5*p6p7+3*p3p6*p4p7+3*p3p7*p4p6))
      b(66)=ccc*(w7*(16*p6p7*(hbcm*p1p11+hbcm*p2p11-2*hbcm*p5p11)+16*
     . p3p6*(-fmb*p7p11-fmc*p7p11-hbcm*p7p11)+16*(-hbcm*p3p7*p6p11-2*
     . hbcm*p4p6*p7p11-2*hbcm*p4p7*p6p11))+w11*(32*p4p7*(-hbcm*p6p11+
     . hbcm*p6p8)+32*p4p6*(-hbcm*p7p11+hbcm*p7p8)+16*p3p7*(-hbcm*
     . p6p11+hbcm*p6p8)+16*p6p7*(hbcm*p1p11-hbcm*p1p8+hbcm*p2p11-hbcm
     . *p2p8-2*hbcm*p5p11+2*hbcm*p5p8)+16*p3p6*(-fmb*p7p11+fmb*p7p8-
     . fmc*p7p11+fmc*p7p8-hbcm*p7p11+hbcm*p7p8))+w13*(16*p6p7*(-hbcm*
     . p1p10-hbcm*p2p10+2*hbcm*p5p10)+16*p3p6*(fmb*p7p10+fmc*p7p10+
     . hbcm*p7p10)+16*(hbcm*p3p7*p6p10+2*hbcm*p4p6*p7p10+2*hbcm*p4p7*
     . p6p10)))
      b(67)=w13*ccc*(p3p6*(16*ffmcfmb+8)+32*p4p6)
      b(68)=w11*ccc*(p3p6*(16*ffmcfmb+8)+32*p4p6)
      b(69)=w13*ccc*(p3p7*(16*ffmcfmb-24)-32*p5p7)
      b(70)=w13*ccc*(16*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p7*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2+fmb*fmc+2*fmb*hbcm-2*fmb2-fmc*hbcm)+16*(-ffmcfmb*p1p3*
     . p3p7+2*p1p2*p3p7+3*p1p3*p5p7-3*p1p4*p3p7))
      b(71)=w11*ccc*(p3p7*(16*ffmcfmb-24)-32*p5p7)
      b(72)=4*ccc*w13*(2*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm-4*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm-3*fmb2-fmc*hbcm+2*
     . hbcm2+6*p1p2-6*p1p3-8*p1p4)
      b(73)=ccc*(8*w7*p6p7*(fmb+fmc)+8*w11*p6p7*(fmb+fmc))
      b(74)=8*ccc*w11*p6p7*(-fmb-fmc)
      b(75)=ccc*(8*w7*(-ffmcfmb*hbcm*p1p11-2*ffmcfmb*hbcm*p2p11+3*
     . ffmcfmb*hbcm*p3p11+3*ffmcfmb*hbcm*p4p11-2*fmb*p2p11+3*fmb*
     . p3p11+3*fmb*p4p11+fmc*p1p11+hbcm*p1p11+2*hbcm*p2p11-3*hbcm*
     . p3p11-3*hbcm*p4p11)+8*w11*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*
     . p1p8-2*ffmcfmb*hbcm*p2p11+2*ffmcfmb*hbcm*p2p8+3*ffmcfmb*hbcm*
     . p3p11-3*ffmcfmb*hbcm*p3p8+3*ffmcfmb*hbcm*p4p11-3*ffmcfmb*hbcm*
     . p4p8-2*fmb*p2p11+2*fmb*p2p8+3*fmb*p3p11-3*fmb*p3p8+3*fmb*p4p11
     . -3*fmb*p4p8+fmc*p1p11-fmc*p1p8+hbcm*p1p11-hbcm*p1p8+2*hbcm*
     . p2p11-2*hbcm*p2p8-3*hbcm*p3p11+3*hbcm*p3p8-3*hbcm*p4p11+3*hbcm
     . *p4p8)+8*w13*(ffmcfmb*hbcm*p1p10+2*ffmcfmb*hbcm*p2p10-3*
     . ffmcfmb*hbcm*p3p10-3*ffmcfmb*hbcm*p4p10+2*fmb*p2p10-3*fmb*
     . p3p10-3*fmb*p4p10-fmc*p1p10-hbcm*p1p10-2*hbcm*p2p10+3*hbcm*
     . p3p10+3*hbcm*p4p10))
      ans2=w13*(16*p1p3*(-3*p1p10-2*p2p10+3*p3p10+3*p4p10)+8*(-3*
     . ffmcfmb*hbcm2*p1p10-ffmcfmb*hbcm2*p2p10+2*ffmcfmb*hbcm2*p3p10+
     . 2*ffmcfmb*hbcm2*p4p10-fmb*hbcm*p2p10+2*fmb*hbcm*p3p10+2*fmb*
     . hbcm*p4p10-fmc*hbcm*p1p10+5*hbcm2*p1p10+hbcm2*p2p10-2*hbcm2*
     . p3p10-2*hbcm2*p4p10-2*p1p10*p2p3+6*p1p10*p3p5))
      ans1=w7*(16*p1p3*(3*p1p11+2*p2p11-3*p3p11-3*p4p11)+8*(3*ffmcfmb
     . *hbcm2*p1p11+ffmcfmb*hbcm2*p2p11-2*ffmcfmb*hbcm2*p3p11-2*
     . ffmcfmb*hbcm2*p4p11+fmb*hbcm*p2p11-2*fmb*hbcm*p3p11-2*fmb*hbcm
     . *p4p11+fmc*hbcm*p1p11-5*hbcm2*p1p11-hbcm2*p2p11+2*hbcm2*p3p11+
     . 2*hbcm2*p4p11+2*p1p11*p2p3-6*p1p11*p3p5))+w11*(48*p3p5*(-p1p11
     . +p1p8)+16*p2p3*(p1p11-p1p8)+16*p1p3*(3*p1p11-3*p1p8+2*p2p11-2*
     . p2p8-3*p3p11+3*p3p8-3*p4p11+3*p4p8)+8*(3*ffmcfmb*hbcm2*p1p11-3
     . *ffmcfmb*hbcm2*p1p8+ffmcfmb*hbcm2*p2p11-ffmcfmb*hbcm2*p2p8-2*
     . ffmcfmb*hbcm2*p3p11+2*ffmcfmb*hbcm2*p3p8-2*ffmcfmb*hbcm2*p4p11
     . +2*ffmcfmb*hbcm2*p4p8+fmb*hbcm*p2p11-fmb*hbcm*p2p8-2*fmb*hbcm*
     . p3p11+2*fmb*hbcm*p3p8-2*fmb*hbcm*p4p11+2*fmb*hbcm*p4p8+fmc*
     . hbcm*p1p11-fmc*hbcm*p1p8-5*hbcm2*p1p11+5*hbcm2*p1p8-hbcm2*
     . p2p11+hbcm2*p2p8+2*hbcm2*p3p11-2*hbcm2*p3p8+2*hbcm2*p4p11-2*
     . hbcm2*p4p8))+ans2
      ans=ccc*ans1
      b(76)=ans
      b(77)=ccc*(w7*(16*p3p6*(-2*p1p11-p2p11+3*p5p11)+8*(2*ffmcfmb*
     . hbcm2*p6p11-fmb*hbcm*p6p11+fmc*hbcm*p6p11-4*hbcm2*p6p11+4*p1p3
     . *p6p11+2*p2p3*p6p11-6*p3p5*p6p11))+w11*(48*p3p5*(-p6p11+p6p8)+
     . 16*p2p3*(p6p11-p6p8)+32*p1p3*(p6p11-p6p8)+16*p3p6*(-2*p1p11+2*
     . p1p8-p2p11+p2p8+3*p5p11-3*p5p8)+8*(2*ffmcfmb*hbcm2*p6p11-2*
     . ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm*p6p8+fmc*hbcm*p6p11
     . -fmc*hbcm*p6p8-4*hbcm2*p6p11+4*hbcm2*p6p8))+w13*(16*p3p6*(2*
     . p1p10+p2p10-3*p5p10)+8*(-2*ffmcfmb*hbcm2*p6p10+fmb*hbcm*p6p10-
     . fmc*hbcm*p6p10+4*hbcm2*p6p10-4*p1p3*p6p10-2*p2p3*p6p10+6*p3p5*
     . p6p10))+8*p3p13)
      b(78)=ccc*(8*w7*(fmb*p6p11+fmc*p6p11)+8*w11*(fmb*p6p11-fmb*p6p8
     . +fmc*p6p11-fmc*p6p8)+8*w13*(-fmb*p6p10-fmc*p6p10))
      b(81)=8*ccc*w13*p6p7*(-fmb-fmc)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc+hbcm)+8*(hbcm*p4p7+2*
     . hbcm*p5p7))
      b(84)=8*ccc*(2*p3p7+p4p7+3*p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-4*hbcm2+2*p1p3+4*p2p3-6*p3p4-2*
     . p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(88)=w13*ccc*(24*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+8*p3p7*(ffmcfmb
     . **2*hbcm+ffmcfmb*fmb-3*ffmcfmb*hbcm-2*fmb+2*hbcm))
      b(89)=-4*ccc*hbcm
      b(90)=ccc*(w7*(24*p5p7*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p7*(-ffmcfmb
     . **2*hbcm-ffmcfmb*fmb+3*ffmcfmb*hbcm+2*fmb-2*hbcm))+w11*(24*
     . p5p7*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p7*(-ffmcfmb**2*hbcm-ffmcfmb*
     . fmb+3*ffmcfmb*hbcm+2*fmb-2*hbcm)))
      b(91)=w11*ccc*(24*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+8*p3p7*(ffmcfmb
     . **2*hbcm+ffmcfmb*fmb-3*ffmcfmb*hbcm-2*fmb+2*hbcm))
      b(92)=8*ccc*(-hbcm2+2*p1p3+p2p3-p3p4-p3p5)
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(ffmcfmb*hbcm+fmb-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp1s1_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(12) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,12 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=-4*ccc*p6p7*w1
      b(2)=4*ccc*p6p7*w2
      b(3)=4*ccc*(-p7p8*w1+p7p9*w2)
      b(4)=4*ccc*p3p7*w1
      b(5)=-4*ccc*p3p7*w2
      b(6)=4*ccc*(hbcm*p7p8*w1-hbcm*p7p9*w2)
      b(7)=4*ccc*(-p6p8*w1+p6p9*w2)
      b(8)=4*ccc*p3p6*w1
      b(9)=-4*ccc*p3p6*w2
      b(10)=4*ccc*(hbcm*p6p8*w1-hbcm*p6p9*w2)
      b(11)=4*ccc*hbcm*p6p7*w1
      b(12)=-4*ccc*hbcm*p6p7*w2
      do 200 n=1,12 
         c(n,1)=c(n,1)+1d0*b(n)
200   continue
      return
      end 
 
      subroutine amp27_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5))
      b(3)=ccc*(w1*(8*p5p7*(-p1p8-p2p8-p4p8+p5p8)+8*p4p7*(-p1p8-p2p8-
     . p4p8+p5p8)+8*p3p7*(-p1p8-p2p8-p4p8+p5p8))+w16*(8*p5p7*(-p1p11+
     . p1p9-p2p11+p2p9-p4p11+p4p9+p5p11-p5p9)+8*p4p7*(-p1p11+p1p9-
     . p2p11+p2p9-p4p11+p4p9+p5p11-p5p9)+8*p3p7*(-p1p11+p1p9-p2p11+
     . p2p9-p4p11+p4p9+p5p11-p5p9)))
      ans=ccc*(w1*(8*p5p6*(p3p8+p4p8)+8*p4p6*(-p1p8+p2p8-p3p8-p5p8)+8
     . *p3p6*(-p1p8+p2p8-p5p8)+4*(ffmcfmb*fmb*hbcm*p6p8-ffmcfmb*fmc*
     . hbcm*p6p8-ffmcfmb*hbcm2*p6p8-fmb*fmc*p6p8+fmb2*p6p8+fmc*hbcm*
     . p6p8+hbcm2*p6p8-2*p1p2*p6p8+2*p1p3*p6p8+2*p1p5*p6p8-2*p2p3*
     . p6p8-2*p2p5*p6p8+2*p3p5*p6p8))+w16*(8*p3p5*(p6p11-p6p9)+8*p5p6
     . *(p3p11-p3p9+p4p11-p4p9)+8*p2p5*(-p6p11+p6p9)+8*p2p3*(-p6p11+
     . p6p9)+8*p1p5*(p6p11-p6p9)+8*p1p3*(p6p11-p6p9)+8*p1p2*(-p6p11+
     . p6p9)+8*p4p6*(-p1p11+p1p9+p2p11-p2p9-p3p11+p3p9-p5p11+p5p9)+8*
     . p3p6*(-p1p11+p1p9+p2p11-p2p9-p5p11+p5p9)+4*(ffmcfmb*fmb*hbcm*
     . p6p11-ffmcfmb*fmb*hbcm*p6p9-ffmcfmb*fmc*hbcm*p6p11+ffmcfmb*fmc
     . *hbcm*p6p9-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p9-fmb*fmc*
     . p6p11+fmb*fmc*p6p9+fmb2*p6p11-fmb2*p6p9+fmc*hbcm*p6p11-fmc*
     . hbcm*p6p9+hbcm2*p6p11-hbcm2*p6p9))+4*(-p1p13+2*p3p13+p4p13+
     . p5p13-p6p11))
      b(7)=ans
      b(8)=w1*ccc*(4*p4p6*(-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm+2*hbcm2
     . )+4*p3p6*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+fmb*fmc-fmb2)+4*(-
     . hbcm2*p5p6+2*p1p2*p3p6+2*p1p3*p4p6-2*p1p5*p3p6-2*p2p3*p4p6+2*
     . p2p5*p3p6-2*p3p4*p5p6+2*p3p5*p4p6))
      b(9)=w16*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-2*hbcm2
     . )+4*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmb*fmc+fmb2)+4*(
     . hbcm2*p5p6-2*p1p2*p3p6-2*p1p3*p4p6+2*p1p5*p3p6+2*p2p3*p4p6-2*
     . p2p5*p3p6+2*p3p4*p5p6-2*p3p5*p4p6))
      ans3=4*(-fmb*p3p13-fmc*p3p13+hbcm*p1p13+hbcm*p3p13+hbcm*p4p13-
     . hbcm*p5p13+hbcm*p6p11)
      ans2=w16*(8*p2p5*(hbcm*p6p11-hbcm*p6p9)+8*p1p5*(-hbcm*p6p11+
     . hbcm*p6p9)+8*p1p2*(hbcm*p6p11-hbcm*p6p9)+8*p4p6*(-hbcm*p1p11+
     . hbcm*p1p9-hbcm*p2p11+hbcm*p2p9-hbcm*p4p11+hbcm*p4p9+hbcm*p5p11
     . -hbcm*p5p9)+8*p3p5*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9-fmc*
     . p6p11+fmc*p6p9-hbcm*p6p11+hbcm*p6p9)+8*p2p3*(-ffmcfmb*hbcm*
     . p6p11+ffmcfmb*hbcm*p6p9+fmc*p6p11-fmc*p6p9+hbcm*p6p11-hbcm*
     . p6p9)+8*p1p3*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p9-2*fmb*
     . p6p11+2*fmb*p6p9-fmc*p6p11+fmc*p6p9+hbcm*p6p11-hbcm*p6p9)+8*
     . p3p6*(ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p3p11+
     . ffmcfmb*hbcm*p3p9-ffmcfmb*hbcm*p4p11+ffmcfmb*hbcm*p4p9-fmb*
     . p1p11+fmb*p1p9-fmb*p2p11+fmb*p2p9+fmb*p3p11-fmb*p3p9+fmb*p4p11
     . -fmb*p4p9+fmb*p5p11-fmb*p5p9-fmc*p2p11+fmc*p2p9+fmc*p5p11-fmc*
     . p5p9)+4*(ffmcfmb*fmb*hbcm2*p6p11-ffmcfmb*fmb*hbcm2*p6p9+
     . ffmcfmb*fmc*hbcm2*p6p11-ffmcfmb*fmc*hbcm2*p6p9+ffmcfmb*hbcm3*
     . p6p11-ffmcfmb*hbcm3*p6p9-fmb*fmc*hbcm*p6p11+fmb*fmc*hbcm*p6p9-
     . fmb2*hbcm*p6p11+fmb2*hbcm*p6p9-fmc*hbcm2*p6p11+fmc*hbcm2*p6p9-
     . hbcm3*p6p11+hbcm3*p6p9))+ans3
      ans1=w1*(8*p4p6*(-hbcm*p1p8-hbcm*p2p8+hbcm*p3p8-hbcm*p4p8+hbcm*
     . p5p8)+8*p5p6*(-fmb*p3p8-fmc*p3p8)+8*p3p5*(ffmcfmb*hbcm*p6p8-
     . fmc*p6p8-hbcm*p6p8)+8*p2p3*(-ffmcfmb*hbcm*p6p8+fmc*p6p8+hbcm*
     . p6p8)+8*p3p6*(ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p4p8-fmb*p1p8-fmb
     . *p2p8+fmb*p4p8+fmb*p5p8-fmc*p2p8+fmc*p5p8)+8*p1p3*(-ffmcfmb*
     . hbcm*p6p8-2*fmb*p6p8-fmc*p6p8+hbcm*p6p8)+4*(ffmcfmb*fmb*hbcm2*
     . p6p8+ffmcfmb*fmc*hbcm2*p6p8+ffmcfmb*hbcm3*p6p8-fmb*fmc*hbcm*
     . p6p8-fmb2*hbcm*p6p8-fmc*hbcm2*p6p8+2*hbcm*p1p2*p6p8-2*hbcm*
     . p1p5*p6p8+2*hbcm*p2p5*p6p8-hbcm3*p6p8))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(13)=2*ccc*w1*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2+2*p1p3+4*p1p4+2
     . *p2p3+2*p2p5+2*p3p4-2*p3p5)
      b(14)=2*ccc*w16*(-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2-2*p1p3-4*p1p4
     . -2*p2p3-2*p2p5-2*p3p4+2*p3p5)
      b(15)=w1*ccc*(4*p3p5*(fmb+fmc)+4*p2p3*(-fmb-fmc)+4*p1p3*(-2*fmb
     . -2*fmc+hbcm)+2*(-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+2*fmb
     . *hbcm2+fmb2*hbcm+2*fmc*hbcm2-fmc2*hbcm-4*hbcm*p1p4-2*hbcm*p2p5
     . -2*hbcm*p3p4-hbcm3))
      b(16)=w16*ccc*(4*p3p5*(-fmb-fmc)+4*p2p3*(fmb+fmc)+4*p1p3*(2*fmb
     . +2*fmc-hbcm)+2*(2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-2*fmb*
     . hbcm2-fmb2*hbcm-2*fmc*hbcm2+fmc2*hbcm+4*hbcm*p1p4+2*hbcm*p2p5+
     . 2*hbcm*p3p4+hbcm3))
      b(21)=ccc*(4*w16*(-hbcm*p1p11+hbcm*p1p9+hbcm*p2p11-hbcm*p2p9+2*
     . hbcm*p3p11-2*hbcm*p3p9+3*hbcm*p4p11-3*hbcm*p4p9-hbcm*p5p11+
     . hbcm*p5p9)+4*w1*(3*fmb*p3p8+3*fmc*p3p8-hbcm*p1p8+hbcm*p2p8-
     . hbcm*p3p8+3*hbcm*p4p8-hbcm*p5p8))
      b(23)=w1*ccc*(4*p3p6*(-ffmcfmb*hbcm+2*fmb+fmc)+8*hbcm*p4p6)
      b(24)=6*ccc*w1*(-fmb-fmc+hbcm)
      b(25)=2*ccc*w1*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+2*
     . p1p3-2*p2p3-6*p3p4+2*p3p5)
      ans2=w1*(16*(p5p6*p3p7)*(-p3p8-p4p8)+16*(p4p6*p3p7)*(p1p8-p2p8+
     . p3p8+p5p8)+16*(p3p7*p3p6)*(2*p1p8-p2p8+p5p8)+8*p4p6*(2*ffmcfmb
     . *hbcm2*p7p8-fmb*hbcm*p7p8+fmc*hbcm*p7p8-2*hbcm2*p7p8)+8*p3p7*(
     . -ffmcfmb*fmb*hbcm*p6p8+ffmcfmb*fmc*hbcm*p6p8+ffmcfmb*hbcm2*
     . p6p8+fmb*fmc*p6p8-fmb2*p6p8-fmc*hbcm*p6p8-hbcm2*p6p8)+8*p3p6*(
     . -ffmcfmb*fmb*hbcm*p7p8+ffmcfmb*fmc*hbcm*p7p8-fmb*fmc*p7p8+fmb2
     . *p7p8)+8*(hbcm2*p5p6*p7p8-2*p1p2*p3p6*p7p8+2*p1p2*p3p7*p6p8-2*
     . p1p3*p3p7*p6p8-2*p1p3*p4p6*p7p8+2*p1p5*p3p6*p7p8-2*p1p5*p3p7*
     . p6p8+2*p1p8*p3p6*p4p7+2*p1p8*p3p6*p5p7+2*p2p3*p3p7*p6p8+2*p2p3
     . *p4p6*p7p8-2*p2p5*p3p6*p7p8+2*p2p5*p3p7*p6p8+2*p3p4*p5p6*p7p8-
     . 2*p3p5*p3p7*p6p8-2*p3p5*p4p6*p7p8))+(8*p3p6*(p1p12-p3p12-p4p12
     . -p5p12+p7p11)+4*(fmb*hbcm*p6p12-fmb*hbcm*p7p13-fmc*hbcm*p6p12+
     . fmc*hbcm*p7p13-hbcm2*p6p12-hbcm2*p7p13+2*p1p3*p7p13+2*p2p3*
     . p6p12+2*p3p13*p4p7+2*p3p13*p5p7-2*p3p4*p6p12-2*p3p5*p7p13))
      ans1=w16*(16*(p3p7*p3p5)*(-p6p11+p6p9)+16*(p5p6*p3p7)*(-p3p11+
     . p3p9-p4p11+p4p9)+16*(p3p7*p2p5)*(p6p11-p6p9)+16*(p3p7*p2p3)*(
     . p6p11-p6p9)+16*(p3p7*p1p5)*(-p6p11+p6p9)+16*(p3p7*p1p3)*(-
     . p6p11+p6p9)+16*(p3p7*p1p2)*(p6p11-p6p9)+16*(p4p6*p3p7)*(p1p11-
     . p1p9-p2p11+p2p9+p3p11-p3p9+p5p11-p5p9)+16*(p5p7*p3p6)*(p1p11-
     . p1p9)+16*(p4p7*p3p6)*(p1p11-p1p9)+16*(p3p7*p3p6)*(2*p1p11-2*
     . p1p9-p2p11+p2p9+p5p11-p5p9)+8*p4p6*(-2*ffmcfmb*hbcm2*p7p9+fmb*
     . hbcm*p7p9-fmc*hbcm*p7p9+2*hbcm2*p7p9)+8*p3p7*(-ffmcfmb*fmb*
     . hbcm*p6p11+ffmcfmb*fmb*hbcm*p6p9+ffmcfmb*fmc*hbcm*p6p11-
     . ffmcfmb*fmc*hbcm*p6p9+ffmcfmb*hbcm2*p6p11-ffmcfmb*hbcm2*p6p9+
     . fmb*fmc*p6p11-fmb*fmc*p6p9-fmb2*p6p11+fmb2*p6p9-fmc*hbcm*p6p11
     . +fmc*hbcm*p6p9-hbcm2*p6p11+hbcm2*p6p9)+8*p3p6*(ffmcfmb*fmb*
     . hbcm*p7p9-ffmcfmb*fmc*hbcm*p7p9+fmb*fmc*p7p9-fmb2*p7p9)+8*(-
     . hbcm2*p5p6*p7p9+2*p1p2*p3p6*p7p9+2*p1p3*p4p6*p7p9-2*p1p5*p3p6*
     . p7p9-2*p2p3*p4p6*p7p9+2*p2p5*p3p6*p7p9-2*p3p4*p5p6*p7p9+2*p3p5
     . *p4p6*p7p9))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(31)=w1*ccc*(4*p5p6*(fmb+fmc)+4*p3p6*(-ffmcfmb*hbcm+fmb)-4*
     . hbcm*p4p6)
      b(32)=w16*ccc*(4*p5p6*(fmb+fmc)+4*p3p6*(-ffmcfmb*hbcm+fmb)-4*
     . hbcm*p4p6)
      b(33)=ccc*(w1*(8*p5p7*(ffmcfmb*hbcm*p6p8+fmb*p6p8-hbcm*p6p8)+8*
     . p4p7*(ffmcfmb*hbcm*p6p8+fmb*p6p8-hbcm*p6p8)+8*p3p7*(ffmcfmb*
     . hbcm*p6p8+fmb*p6p8-hbcm*p6p8))+w16*(8*p5p7*(ffmcfmb*hbcm*p6p11
     . -ffmcfmb*hbcm*p6p9+fmb*p6p11-fmb*p6p9-hbcm*p6p11+hbcm*p6p9)+8*
     . p4p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9+fmb*p6p11-fmb*p6p9-
     . hbcm*p6p11+hbcm*p6p9)+8*p3p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*
     . p6p9+fmb*p6p11-fmb*p6p9-hbcm*p6p11+hbcm*p6p9))+4*(fmb*p6p12-
     . fmb*p7p13+fmc*p6p12-fmc*p7p13-hbcm*p6p12+hbcm*p7p13))
      b(34)=6*ccc*w16*(-fmb-fmc+hbcm)
      b(36)=2*ccc*w16*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+hbcm2-2*
     . p1p3+2*p2p3+6*p3p4-2*p3p5)
      b(39)=8*ccc*w16*(-p3p6*p3p7-p3p6*p4p7-p3p6*p5p7)
      b(42)=8*ccc*w16*(p3p6*p3p7+p3p6*p4p7+p3p6*p5p7)
      b(43)=ccc*(8*w1*(-p3p7*p6p8-p4p7*p6p8-p5p7*p6p8)+w16*(8*p5p7*(-
     . p6p11+p6p9)+8*p4p7*(-p6p11+p6p9)+8*p3p7*(-p6p11+p6p9))+4*(
     . p6p12-p7p13))
      b(44)=w16*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-2*
     . hbcm2)+4*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmb*fmc+fmb2
     . )+4*(hbcm2*p5p6-2*p1p2*p3p6-2*p1p3*p4p6+2*p1p5*p3p6+2*p2p3*
     . p4p6-2*p2p5*p3p6+2*p3p4*p5p6-2*p3p5*p4p6))
      b(45)=4*ccc*w16*(-p3p6-3*p4p6+p5p6)
      b(46)=w16*ccc*(4*p5p6*(-fmb-fmc)+4*p3p6*(ffmcfmb*hbcm-fmb)+4*
     . hbcm*p4p6)
      b(47)=4*ccc*w16*(p3p6+3*p4p6-p5p6)
      b(53)=ccc*(4*w1*(-p1p8+p2p8+2*p3p8+3*p4p8-p5p8)+4*w16*(-p1p11+
     . p1p9+p2p11-p2p9+2*p3p11-2*p3p9+3*p4p11-3*p4p9-p5p11+p5p9))
      b(54)=2*ccc*w16*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+2
     . *p1p3-2*p2p3-6*p3p4+2*p3p5)
      b(55)=6*ccc*w16*(fmb+fmc-hbcm)
      b(56)=2*ccc*w16*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2+2*p1p3+4*p1p4+
     . 2*p2p3+2*p2p5+2*p3p4-2*p3p5)
      b(58)=w16*ccc*(4*p3p6*(-ffmcfmb*hbcm+2*fmb+fmc)+8*hbcm*p4p6)
      b(59)=w16*ccc*(4*p3p5*(fmb+fmc)+4*p2p3*(-fmb-fmc)+4*p1p3*(-2*
     . fmb-2*fmc+hbcm)+2*(-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+2*
     . fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2-fmc2*hbcm-4*hbcm*p1p4-2*hbcm*
     . p2p5-2*hbcm*p3p4-hbcm3))
      b(60)=w16*ccc*(4*p3p6*(ffmcfmb*hbcm-2*fmb-fmc)-8*hbcm*p4p6)
      b(65)=8*ccc*w1*(-p3p6*p3p7-p3p6*p4p7-p3p6*p5p7)
      b(66)=ccc*(w1*(8*p3p7*(-ffmcfmb*hbcm*p6p8-2*fmb*p6p8-fmc*p6p8+
     . hbcm*p6p8)+8*(-hbcm*p4p7*p6p8-hbcm*p5p7*p6p8))+w16*(8*p5p7*(-
     . hbcm*p6p11+hbcm*p6p9)+8*p4p7*(-hbcm*p6p11+hbcm*p6p9)+8*p3p7*(-
     . ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p9-2*fmb*p6p11+2*fmb*p6p9-
     . fmc*p6p11+fmc*p6p9+hbcm*p6p11-hbcm*p6p9))+4*(hbcm*p6p12-hbcm*
     . p7p13))
      b(68)=4*ccc*w1*(-p3p6-3*p4p6+p5p6)
      b(75)=ccc*(4*w1*(2*fmb*p1p8+fmb*p2p8-fmb*p3p8-fmb*p5p8+2*fmc*
     . p1p8+fmc*p2p8-fmc*p3p8-fmc*p5p8-2*hbcm*p1p8-hbcm*p2p8+hbcm*
     . p3p8+hbcm*p5p8)+4*w16*(2*fmb*p1p11-2*fmb*p1p9+fmb*p2p11-fmb*
     . p2p9-fmb*p3p11+fmb*p3p9-fmb*p5p11+fmb*p5p9+2*fmc*p1p11-2*fmc*
     . p1p9+fmc*p2p11-fmc*p2p9-fmc*p3p11+fmc*p3p9-fmc*p5p11+fmc*p5p9-
     . 2*hbcm*p1p11+2*hbcm*p1p9-hbcm*p2p11+hbcm*p2p9+hbcm*p3p11-hbcm*
     . p3p9+hbcm*p5p11-hbcm*p5p9))
      ans2=w16*(8*p3p5*(-2*p1p11+2*p1p9-p2p11+p2p9+p3p11-p3p9+p5p11-
     . p5p9)+8*p3p4*(p1p11-p1p9-p3p11+p3p9-p4p11+p4p9)+8*p2p3*(2*
     . p1p11-2*p1p9+p2p11-p2p9-p3p11+p3p9-p5p11+p5p9)+8*p1p3*(p1p11-
     . p1p9-p3p11+p3p9-p4p11+p4p9)+4*(2*ffmcfmb*hbcm2*p1p11-2*ffmcfmb
     . *hbcm2*p1p9-2*ffmcfmb*hbcm2*p3p11+2*ffmcfmb*hbcm2*p3p9-2*
     . ffmcfmb*hbcm2*p4p11+2*ffmcfmb*hbcm2*p4p9-2*fmb*hbcm*p1p11+2*
     . fmb*hbcm*p1p9-fmb*hbcm*p2p11+fmb*hbcm*p2p9+fmb*hbcm*p3p11-fmb*
     . hbcm*p3p9+fmb*hbcm*p5p11-fmb*hbcm*p5p9+2*fmc*hbcm*p1p11-2*fmc*
     . hbcm*p1p9+fmc*hbcm*p2p11-fmc*hbcm*p2p9-fmc*hbcm*p3p11+fmc*hbcm
     . *p3p9-fmc*hbcm*p5p11+fmc*hbcm*p5p9-2*hbcm2*p1p11+2*hbcm2*p1p9-
     . hbcm2*p2p11+hbcm2*p2p9+hbcm2*p3p11-hbcm2*p3p9+hbcm2*p5p11-
     . hbcm2*p5p9))
      ans1=w1*(8*p3p5*(-2*p1p8-p2p8+p3p8+p5p8)+8*p3p4*(p1p8-p3p8-p4p8
     . )+8*p2p3*(2*p1p8+p2p8-p3p8-p5p8)+8*p1p3*(p1p8-p3p8-p4p8)+4*(2*
     . ffmcfmb*hbcm2*p1p8-2*ffmcfmb*hbcm2*p3p8-2*ffmcfmb*hbcm2*p4p8-2
     . *fmb*hbcm*p1p8-fmb*hbcm*p2p8+fmb*hbcm*p3p8+fmb*hbcm*p5p8+2*fmc
     . *hbcm*p1p8+fmc*hbcm*p2p8-fmc*hbcm*p3p8-fmc*hbcm*p5p8-2*hbcm2*
     . p1p8-hbcm2*p2p8+hbcm2*p3p8+hbcm2*p5p8))+ans2
      ans=ccc*ans1
      b(76)=ans
      b(77)=ccc*(w1*(16*p3p6*(-p2p8+p5p8)+4*(ffmcfmb*hbcm2*p6p8-2*fmb
     . *hbcm*p6p8+fmc*hbcm*p6p8-2*hbcm2*p6p8+4*p2p3*p6p8-4*p3p5*p6p8)
     . )+w16*(16*p3p5*(-p6p11+p6p9)+16*p2p3*(p6p11-p6p9)+16*p3p6*(-
     . p2p11+p2p9+p5p11-p5p9)+4*(ffmcfmb*hbcm2*p6p11-ffmcfmb*hbcm2*
     . p6p9-2*fmb*hbcm*p6p11+2*fmb*hbcm*p6p9+fmc*hbcm*p6p11-fmc*hbcm*
     . p6p9-2*hbcm2*p6p11+2*hbcm2*p6p9))-4*p3p13)
      b(78)=ccc*(4*w1*(ffmcfmb*hbcm*p6p8+2*fmb*p6p8+fmc*p6p8-2*hbcm*
     . p6p8)+4*w16*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p9+2*fmb*p6p11-
     . 2*fmb*p6p9+fmc*p6p11-fmc*p6p9-2*hbcm*p6p11+2*hbcm*p6p9))
      b(83)=4*ccc*(-hbcm*p3p7-hbcm*p4p7-hbcm*p5p7)
      b(84)=4*ccc*(-p3p7-p4p7-p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm+hbcm2-p1p3-p2p3+p3p4+p3p5)
      b(87)=-4*ccc*hbcm
      b(89)=-4*ccc*hbcm
      do 200 n=1,94 
         c(n,2)=c(n,2)-0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp26_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      b(1)=w11*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2)+8*(-p1p3
     . *p6p7-2*p1p5*p6p7-p2p3*p6p7-p2p4*p6p7+p3p4*p6p7-p3p5*p6p7-p3p7
     . *p4p6+p3p7*p5p6-p4p6*p4p7-p4p6*p5p7+p4p7*p5p6+p5p6*p5p7))
      b(2)=w2*ccc*(4*p6p7*(-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2)+8*(p1p3*
     . p6p7+2*p1p5*p6p7+p2p3*p6p7+p2p4*p6p7-p3p4*p6p7+p3p5*p6p7+p3p7*
     . p4p6-p3p7*p5p6+p4p6*p4p7+p4p6*p5p7-p4p7*p5p6-p5p6*p5p7))
      b(3)=ccc*(4*w2*(-2*ffmcfmb*hbcm2*p7p9+fmb2*p7p9-fmc2*p7p9+hbcm2
     . *p7p9+4*p1p2*p7p9-2*p1p3*p7p9-4*p1p4*p7p9+2*p2p3*p7p9+2*p2p4*
     . p7p9-2*p3p4*p7p9+2*p3p5*p7p9)+w11*(8*p3p5*(p7p11-p7p8)+8*p3p4*
     . (-p7p11+p7p8)+8*p2p4*(p7p11-p7p8)+8*p2p3*(p7p11-p7p8)+16*p1p4*
     . (-p7p11+p7p8)+8*p1p3*(-p7p11+p7p8)+16*p1p2*(p7p11-p7p8)+4*(-2*
     . ffmcfmb*hbcm2*p7p11+2*ffmcfmb*hbcm2*p7p8+fmb2*p7p11-fmb2*p7p8-
     . fmc2*p7p11+fmc2*p7p8+hbcm2*p7p11-hbcm2*p7p8)))
      b(4)=w11*ccc*(4*p3p7*(-fmb*hbcm+fmb2+fmc*hbcm-fmc2)+4*p5p7*(2*
     . ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+4*p4p7*(2*ffmcfmb*hbcm2
     . -fmb*hbcm+fmc*hbcm-hbcm2)+8*(2*p1p2*p3p7+p1p3*p4p7+p1p3*p5p7-2
     . *p1p4*p3p7-p2p3*p4p7-p2p3*p5p7+p2p4*p3p7+p3p4*p4p7+p3p4*p5p7-
     . p3p5*p4p7-p3p5*p5p7))
      b(5)=w2*ccc*(4*p3p7*(fmb*hbcm-fmb2-fmc*hbcm+fmc2)+4*p5p7*(-2*
     . ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+4*p4p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+8*(-2*p1p2*p3p7-p1p3*p4p7-p1p3*
     . p5p7+2*p1p4*p3p7+p2p3*p4p7+p2p3*p5p7-p2p4*p3p7-p3p4*p4p7-p3p4*
     . p5p7+p3p5*p4p7+p3p5*p5p7))
      ans2=w11*(8*p3p4*(hbcm*p7p11-hbcm*p7p8)+8*p2p4*(-hbcm*p7p11+
     . hbcm*p7p8)+8*p2p3*(-hbcm*p7p11+hbcm*p7p8)+16*p1p4*(hbcm*p7p11-
     . hbcm*p7p8)+16*p1p2*(-hbcm*p7p11+hbcm*p7p8)+8*p5p7*(-hbcm*p1p11
     . +hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*p4p11+hbcm*p4p8+hbcm*
     . p5p11-hbcm*p5p8)+8*p4p7*(-hbcm*p1p11+hbcm*p1p8+hbcm*p2p11-hbcm
     . *p2p8-hbcm*p4p11+hbcm*p4p8+hbcm*p5p11-hbcm*p5p8)+8*p3p5*(-fmb*
     . p7p11+fmb*p7p8-fmc*p7p11+fmc*p7p8)+8*p1p3*(2*fmb*p7p11-2*fmb*
     . p7p8+2*fmc*p7p11-2*fmc*p7p8-hbcm*p7p11+hbcm*p7p8)+8*p3p7*(-fmb
     . *p1p11+fmb*p1p8+fmb*p2p11-fmb*p2p8-fmb*p4p11+fmb*p4p8-fmc*
     . p1p11+fmc*p1p8+fmc*p2p11-fmc*p2p8-fmc*p4p11+fmc*p4p8+hbcm*
     . p5p11-hbcm*p5p8)+4*(2*ffmcfmb*fmb*hbcm2*p7p11-2*ffmcfmb*fmb*
     . hbcm2*p7p8+2*ffmcfmb*fmc*hbcm2*p7p11-2*ffmcfmb*fmc*hbcm2*p7p8-
     . 2*fmb*hbcm2*p7p11+2*fmb*hbcm2*p7p8-fmb2*hbcm*p7p11+fmb2*hbcm*
     . p7p8-2*fmc*hbcm2*p7p11+2*fmc*hbcm2*p7p8+fmc2*hbcm*p7p11-fmc2*
     . hbcm*p7p8+hbcm3*p7p11-hbcm3*p7p8))
      ans1=w2*(8*p5p7*(-hbcm*p1p9+hbcm*p2p9-hbcm*p4p9+hbcm*p5p9)+8*
     . p4p7*(-hbcm*p1p9+hbcm*p2p9-hbcm*p4p9+hbcm*p5p9)+8*p3p5*(-fmb*
     . p7p9-fmc*p7p9)+8*p3p7*(-fmb*p1p9+fmb*p2p9-fmb*p4p9-fmc*p1p9+
     . fmc*p2p9-fmc*p4p9+hbcm*p5p9)+8*p1p3*(2*fmb*p7p9+2*fmc*p7p9-
     . hbcm*p7p9)+4*(2*ffmcfmb*fmb*hbcm2*p7p9+2*ffmcfmb*fmc*hbcm2*
     . p7p9-2*fmb*hbcm2*p7p9-fmb2*hbcm*p7p9-2*fmc*hbcm2*p7p9+fmc2*
     . hbcm*p7p9-4*hbcm*p1p2*p7p9+4*hbcm*p1p4*p7p9-2*hbcm*p2p3*p7p9-2
     . *hbcm*p2p4*p7p9+2*hbcm*p3p4*p7p9+hbcm3*p7p9))+ans2
      ans=ccc*ans1
      b(6)=ans
      b(7)=ccc*(w2*(8*p5p6*(-p2p9+p3p9+p4p9)+8*p4p6*(-p1p9-p3p9-p5p9)
     . +8*p3p6*(-p1p9-p5p9)+4*(ffmcfmb*fmb*hbcm*p6p9-ffmcfmb*fmc*hbcm
     . *p6p9-ffmcfmb*hbcm2*p6p9-fmb*fmc*p6p9+fmb2*p6p9+fmc*hbcm*p6p9+
     . hbcm2*p6p9+2*p1p3*p6p9+2*p1p5*p6p9+2*p3p5*p6p9))+w11*(8*p3p5*(
     . p6p11-p6p8)+8*p5p6*(-p2p11+p2p8+p3p11-p3p8+p4p11-p4p8)+8*p1p5*
     . (p6p11-p6p8)+8*p1p3*(p6p11-p6p8)+8*p4p6*(-p1p11+p1p8-p3p11+
     . p3p8-p5p11+p5p8)+8*p3p6*(-p1p11+p1p8-p5p11+p5p8)+4*(ffmcfmb*
     . fmb*hbcm*p6p11-ffmcfmb*fmb*hbcm*p6p8-ffmcfmb*fmc*hbcm*p6p11+
     . ffmcfmb*fmc*hbcm*p6p8-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8-
     . fmb*fmc*p6p11+fmb*fmc*p6p8+fmb2*p6p11-fmb2*p6p8+fmc*hbcm*p6p11
     . -fmc*hbcm*p6p8+hbcm2*p6p11-hbcm2*p6p8))+4*(p1p13+p2p13-2*p3p13
     . -p4p13-p5p13))
      b(8)=w11*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-2*hbcm2
     . )+4*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmb*fmc+fmb2)+4*(
     . hbcm2*p5p6-2*p1p3*p4p6+2*p1p5*p3p6-2*p2p3*p5p6+2*p3p4*p5p6-2*
     . p3p5*p4p6))
      b(9)=w2*ccc*(4*p4p6*(-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm+2*hbcm2
     . )+4*p3p6*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+fmb*fmc-fmb2)+4*(-
     . hbcm2*p5p6+2*p1p3*p4p6-2*p1p5*p3p6+2*p2p3*p5p6-2*p3p4*p5p6+2*
     . p3p5*p4p6))
      ans2=w11*(8*p1p5*(-hbcm*p6p11+hbcm*p6p8)+8*p4p6*(-hbcm*p1p11+
     . hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*p3p8-hbcm*p4p11+hbcm*p4p8+
     . hbcm*p5p11-hbcm*p5p8)+8*p5p6*(fmb*p3p8+fmc*p3p8)+8*p3p5*(
     . ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8-fmc*p6p11+fmc*p6p8-hbcm*
     . p6p11+hbcm*p6p8)+8*p1p3*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8
     . -2*fmb*p6p11+2*fmb*p6p8-fmc*p6p11+fmc*p6p8+hbcm*p6p11-hbcm*
     . p6p8)+8*p3p6*(ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p1p8+ffmcfmb*
     . hbcm*p2p11-ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p3p11-ffmcfmb*hbcm*
     . p4p11+ffmcfmb*hbcm*p4p8-fmb*p1p11+fmb*p1p8-fmb*p2p11+fmb*p2p8+
     . fmb*p3p11+fmb*p4p11-fmb*p4p8+fmb*p5p11-fmb*p5p8+fmc*p5p11-fmc*
     . p5p8)+4*(ffmcfmb*fmb*hbcm2*p6p11-ffmcfmb*fmb*hbcm2*p6p8+
     . ffmcfmb*fmc*hbcm2*p6p11-ffmcfmb*fmc*hbcm2*p6p8+ffmcfmb*hbcm3*
     . p6p11-ffmcfmb*hbcm3*p6p8-fmb*fmc*hbcm*p6p11+fmb*fmc*hbcm*p6p8-
     . fmb2*hbcm*p6p11+fmb2*hbcm*p6p8-fmc*hbcm2*p6p11+fmc*hbcm2*p6p8-
     . hbcm3*p6p11+hbcm3*p6p8))+4*(fmb*p3p13+fmc*p3p13-hbcm*p1p13+
     . hbcm*p2p13-hbcm*p3p13-hbcm*p4p13+hbcm*p5p13)
      ans1=w2*(8*p4p6*(-hbcm*p1p9+hbcm*p2p9-hbcm*p4p9+hbcm*p5p9)+8*
     . p3p5*(ffmcfmb*hbcm*p6p9-fmc*p6p9-hbcm*p6p9)+8*p3p6*(ffmcfmb*
     . hbcm*p1p9+ffmcfmb*hbcm*p2p9-ffmcfmb*hbcm*p3p9-ffmcfmb*hbcm*
     . p4p9-fmb*p1p9-fmb*p2p9+fmb*p3p9+fmb*p4p9+fmb*p5p9+fmc*p5p9)+8*
     . p1p3*(-ffmcfmb*hbcm*p6p9-2*fmb*p6p9-fmc*p6p9+hbcm*p6p9)+4*(
     . ffmcfmb*fmb*hbcm2*p6p9+ffmcfmb*fmc*hbcm2*p6p9+ffmcfmb*hbcm3*
     . p6p9-fmb*fmc*hbcm*p6p9-fmb2*hbcm*p6p9-fmc*hbcm2*p6p9-2*hbcm*
     . p1p5*p6p9-hbcm3*p6p9))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=w11*ccc*(8*(p5p6*p3p7)*(fmb+fmc)+8*(p6p7*p3p4)*(fmb+fmc)+
     . 8*(p6p7*p2p3)*(-fmb-fmc)+8*(p6p7*p1p3)*(-2*fmb-2*fmc+hbcm)+8*(
     . p5p7*p3p6)*(ffmcfmb*hbcm-fmb)+8*(p4p7*p3p6)*(ffmcfmb*hbcm-fmb)
     . +4*p6p7*(2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-fmb2*hbcm+
     . fmc2*hbcm-hbcm3)+8*(-2*hbcm*p1p5*p6p7-hbcm*p2p4*p6p7-hbcm*p3p5
     . *p6p7-hbcm*p3p7*p4p6))
      b(12)=w2*ccc*(8*(p5p6*p3p7)*(-fmb-fmc)+8*(p6p7*p3p4)*(-fmb-fmc)
     . +8*(p6p7*p2p3)*(fmb+fmc)+8*(p6p7*p1p3)*(-fmb-fmc+2*hbcm)+8*(
     . p5p7*p3p6)*(-ffmcfmb*hbcm+fmb)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm+
     . fmb)+4*p6p7*(-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+fmb2*
     . hbcm-fmc2*hbcm+hbcm3)+8*(2*hbcm*p1p5*p6p7+hbcm*p2p4*p6p7+hbcm*
     . p3p5*p6p7+hbcm*p3p7*p4p6))
      b(13)=2*ccc*w11*(-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2+4*p1p2-2*p1p3
     . -4*p1p4+2*p2p3+2*p2p4-2*p3p4+2*p3p5)
      b(14)=2*ccc*w2*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2-4*p1p2+2*p1p3+4
     . *p1p4-2*p2p3-2*p2p4+2*p3p4-2*p3p5)
      b(15)=w11*ccc*(4*p3p5*(-fmb-fmc)+4*p1p3*(2*fmb+2*fmc-hbcm)+2*(2
     . *ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-2*fmb*hbcm2-fmb2*hbcm-2
     . *fmc*hbcm2+fmc2*hbcm-4*hbcm*p1p2+4*hbcm*p1p4-2*hbcm*p2p3-2*
     . hbcm*p2p4+2*hbcm*p3p4+hbcm3))
      b(16)=w2*ccc*(4*p3p5*(fmb+fmc)+4*p1p3*(-2*fmb-2*fmc+hbcm)+2*(-2
     . *ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+2*fmb*hbcm2+fmb2*hbcm+2
     . *fmc*hbcm2-fmc2*hbcm+4*hbcm*p1p2-4*hbcm*p1p4+2*hbcm*p2p3+2*
     . hbcm*p2p4-2*hbcm*p3p4-hbcm3))
      b(21)=ccc*(4*w2*(-hbcm*p1p9-3*hbcm*p2p9+2*hbcm*p3p9+3*hbcm*p4p9
     . -hbcm*p5p9)+4*w11*(-3*fmb*p3p8-3*fmc*p3p8-hbcm*p1p11+hbcm*p1p8
     . -3*hbcm*p2p11+3*hbcm*p2p8+2*hbcm*p3p11+hbcm*p3p8+3*hbcm*p4p11-
     . 3*hbcm*p4p8-hbcm*p5p11+hbcm*p5p8))
      b(23)=w11*ccc*(4*p3p6*(ffmcfmb*hbcm-2*fmb-fmc)-8*hbcm*p4p6)
      b(24)=6*ccc*w11*(fmb+fmc-hbcm)
      b(25)=2*ccc*w11*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+hbcm2-2*
     . p1p3-6*p2p3+6*p3p4-2*p3p5)
      b(26)=w11*ccc*(4*p3p7*(-3*fmb-3*fmc+hbcm)+8*(-hbcm*p4p7-hbcm*
     . p5p7))
      ans4=8*p6p7*(-2*ffmcfmb*hbcm2*p5p11+2*ffmcfmb*hbcm2*p5p8+fmb*
     . hbcm*p1p11-fmb*hbcm*p1p8-fmb*hbcm*p2p11+fmb*hbcm*p2p8+fmb*hbcm
     . *p4p11-fmb*hbcm*p4p8-fmc*hbcm*p1p11+fmc*hbcm*p1p8+fmc*hbcm*
     . p2p11-fmc*hbcm*p2p8-fmc*hbcm*p4p11+fmc*hbcm*p4p8+3*hbcm2*p1p11
     . -3*hbcm2*p1p8+hbcm2*p2p11-hbcm2*p2p8-hbcm2*p4p11+hbcm2*p4p8+2*
     . hbcm2*p5p11-2*hbcm2*p5p8)+8*p4p7*(ffmcfmb*hbcm2*p6p11-ffmcfmb*
     . hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm*p6p8-hbcm2*p6p11+hbcm2*p6p8
     . )+8*p4p6*(-2*ffmcfmb*hbcm2*p7p11+fmb*hbcm*p7p11-fmc*hbcm*p7p11
     . +2*hbcm2*p7p11)+8*p3p7*(ffmcfmb*hbcm2*p6p11-ffmcfmb*hbcm2*p6p8
     . -fmb*hbcm*p6p11+fmb*hbcm*p6p8-hbcm2*p6p11+hbcm2*p6p8)+8*p3p6*(
     . ffmcfmb*fmb*hbcm*p7p11-ffmcfmb*fmc*hbcm*p7p11+fmb*fmc*p7p11-
     . fmb2*p7p11)+8*(-hbcm2*p5p6*p7p11+2*p1p3*p4p6*p7p11-2*p1p5*p3p6
     . *p7p11+2*p2p3*p5p6*p7p11-2*p3p4*p5p6*p7p11+2*p3p5*p4p6*p7p11)
      ans3=16*(p5p7*p3p5)*(-p6p11+p6p8)+16*(p4p7*p3p5)*(-p6p11+p6p8)+
     . 16*(p3p7*p3p5)*(-p6p11+p6p8)+16*(p5p6*p3p7)*(p2p11-p2p8-p3p11+
     . p3p8-p4p11+p4p8-p5p11+p5p8)+16*(p5p7*p1p3)*(-p6p11+p6p8)+16*(
     . p4p7*p1p3)*(-p6p11+p6p8)+16*(p3p7*p1p3)*(-p6p11+p6p8)+16*(p4p6
     . *p3p7)*(2*p1p11-2*p1p8-p2p11+p2p8+p3p11-p3p8+p4p11-p4p8+p5p11-
     . p5p8)+16*(p5p7*p3p6)*(p1p11-p1p8+p5p11-p5p8)+16*(p4p7*p3p6)*(
     . p1p11-p1p8+p5p11-p5p8)+16*(p3p7*p3p6)*(2*p1p11-2*p1p8+p5p11-
     . p5p8)+16*(p6p7*p3p5)*(2*p1p11-2*p1p8+p5p11-p5p8)+16*(p6p7*p3p4
     . )*(p1p11-p1p8+p2p11-p2p8-p4p11+p4p8)+16*(p6p7*p2p3)*(-p1p11+
     . p1p8-p2p11+p2p8+p4p11-p4p8)+16*(p6p7*p1p3)*(-2*p1p11+2*p1p8+
     . p5p11-p5p8)+8*p5p7*(ffmcfmb*hbcm2*p6p11-ffmcfmb*hbcm2*p6p8-fmb
     . *hbcm*p6p11+fmb*hbcm*p6p8-hbcm2*p6p11+hbcm2*p6p8)+ans4
      ans2=w11*ans3
      ans5=(8*p3p7*(-p1p13-p2p13+p3p13+p4p13+p5p13)+8*(fmb*hbcm*p7p13
     . -fmc*hbcm*p7p13+hbcm2*p7p13-p1p3*p7p13-p2p3*p7p13+p3p4*p7p13+
     . p3p5*p7p13))
      ans1=w2*(16*(p5p6*p3p7)*(p2p9-p3p9-p4p9-p5p9)+16*(p4p6*p3p7)*(2
     . *p1p9-p2p9+p3p9+p4p9+p5p9)+16*(p5p7*p3p6)*(p1p9+p5p9)+16*(p4p7
     . *p3p6)*(p1p9+p5p9)+16*(p3p7*p3p6)*(2*p1p9+p5p9)+16*(p6p7*p3p5)
     . *(2*p1p9+p5p9)+16*(p6p7*p3p4)*(p1p9+p2p9-p4p9)+16*(p6p7*p2p3)*
     . (-p1p9-p2p9+p4p9)+16*(p6p7*p1p3)*(-2*p1p9+p5p9)+8*p6p7*(-2*
     . ffmcfmb*hbcm2*p5p9+fmb*hbcm*p1p9-fmb*hbcm*p2p9+fmb*hbcm*p4p9-
     . fmc*hbcm*p1p9+fmc*hbcm*p2p9-fmc*hbcm*p4p9+3*hbcm2*p1p9+hbcm2*
     . p2p9-hbcm2*p4p9+2*hbcm2*p5p9)+8*p5p7*(ffmcfmb*hbcm2*p6p9-fmb*
     . hbcm*p6p9-hbcm2*p6p9)+8*p4p7*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9
     . -hbcm2*p6p9)+8*p3p7*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9-hbcm2*
     . p6p9)+16*(-p1p3*p3p7*p6p9-p1p3*p4p7*p6p9-p1p3*p5p7*p6p9-p3p5*
     . p3p7*p6p9-p3p5*p4p7*p6p9-p3p5*p5p7*p6p9))+ans2+ans5
      ans=ccc*ans1
      b(30)=ans
      b(31)=w11*ccc*(4*p5p6*(-fmb-fmc)+4*p3p6*(ffmcfmb*hbcm-fmb)+4*
     . hbcm*p4p6)
      b(32)=w2*ccc*(4*p5p6*(-fmb-fmc)+4*p3p6*(ffmcfmb*hbcm-fmb)+4*
     . hbcm*p4p6)
      b(33)=ccc*(w2*(8*p5p6*(fmb*p7p9+fmc*p7p9)+8*p6p7*(fmb*p1p9-fmb*
     . p2p9+fmb*p4p9+fmc*p1p9-fmc*p2p9+fmc*p4p9-hbcm*p1p9+hbcm*p2p9-
     . hbcm*p4p9)+8*p3p6*(-ffmcfmb*hbcm*p7p9+fmb*p7p9)-8*hbcm*p4p6*
     . p7p9)+w11*(8*p4p6*(-hbcm*p7p11+hbcm*p7p8)+8*p5p6*(fmb*p7p11-
     . fmb*p7p8+fmc*p7p11-fmc*p7p8)+8*p6p7*(-2*fmb*p1p11+2*fmb*p1p8-
     . fmb*p2p11+fmb*p2p8+fmb*p4p11-fmb*p4p8-2*fmc*p1p11+2*fmc*p1p8-
     . fmc*p2p11+fmc*p2p8+fmc*p4p11-fmc*p4p8+2*hbcm*p1p11-2*hbcm*p1p8
     . +hbcm*p2p11-hbcm*p2p8-hbcm*p4p11+hbcm*p4p8)+8*p3p6*(-ffmcfmb*
     . hbcm*p7p11+ffmcfmb*hbcm*p7p8+fmb*p7p11-fmb*p7p8))-8*hbcm*p7p13
     . )
      b(34)=6*ccc*w2*(fmb+fmc-hbcm)
      b(35)=ccc*(12*w2*(-fmb*p7p9-fmc*p7p9+hbcm*p7p9)+12*w11*(-fmb*
     . p7p11+fmb*p7p8-fmc*p7p11+fmc*p7p8+hbcm*p7p11-hbcm*p7p8))
      b(36)=2*ccc*w2*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+2*
     . p1p3+6*p2p3-6*p3p4+2*p3p5)
      b(37)=w2*ccc*(4*p3p7*(3*fmb+3*fmc-hbcm)+8*(hbcm*p4p7+hbcm*p5p7)
     . )
      b(38)=ccc*(w2*(8*p3p7*(-p1p9+p2p9-p4p9+3*p5p9)+4*(2*ffmcfmb*
     . hbcm2*p7p9-3*fmb*hbcm*p7p9+3*fmc*hbcm*p7p9+hbcm2*p7p9-2*p1p3*
     . p7p9-6*p2p3*p7p9+6*p3p4*p7p9-2*p3p5*p7p9))+w11*(8*p3p5*(-p7p11
     . +p7p8)+24*p3p4*(p7p11-p7p8)+24*p2p3*(-p7p11+p7p8)+8*p1p3*(-
     . p7p11+p7p8)+8*p3p7*(-p1p11+p1p8+p2p11-p2p8-p4p11+p4p8+3*p5p11-
     . 3*p5p8)+4*(2*ffmcfmb*hbcm2*p7p11-2*ffmcfmb*hbcm2*p7p8-3*fmb*
     . hbcm*p7p11+3*fmb*hbcm*p7p8+3*fmc*hbcm*p7p11-3*fmc*hbcm*p7p8+
     . hbcm2*p7p11-hbcm2*p7p8)))
      b(39)=w11*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm-3*
     . hbcm2)+8*(p1p3*p6p7-p2p3*p6p7+p3p4*p6p7-3*p3p5*p6p7-p3p6*p3p7-
     . 3*p3p7*p4p6+p3p7*p5p6))
      b(40)=w11*ccc*(8*(p5p6*p3p7)*(-fmb-fmc)+8*(p6p7*p3p4)*(-fmb-fmc
     . )+8*(p6p7*p2p3)*(fmb+fmc)+8*(p6p7*p1p3)*(2*fmb+2*fmc-hbcm)+8*(
     . p5p7*p3p6)*(-ffmcfmb*hbcm+fmb)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm+
     . fmb)+4*p6p7*(-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+fmb2*
     . hbcm-fmc2*hbcm+hbcm3)+8*(2*hbcm*p1p5*p6p7+hbcm*p2p4*p6p7+hbcm*
     . p3p5*p6p7+hbcm*p3p7*p4p6))
      b(41)=w11*ccc*(4*p6p7*(-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2)+8*(
     . p1p3*p6p7+2*p1p5*p6p7+p2p3*p6p7+p2p4*p6p7-p3p4*p6p7+p3p5*p6p7+
     . p3p7*p4p6-p3p7*p5p6+p4p6*p4p7+p4p6*p5p7-p4p7*p5p6-p5p6*p5p7))
      b(42)=w2*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm-3*
     . hbcm2)+8*(p1p3*p6p7-p2p3*p6p7+p3p4*p6p7-3*p3p5*p6p7-p3p6*p3p7-
     . 3*p3p7*p4p6+p3p7*p5p6))
      b(43)=ccc*(w2*(8*p6p7*(p1p9-p2p9+p4p9-3*p5p9)+8*(-p3p6*p7p9-3*
     . p4p6*p7p9+p5p6*p7p9))+w11*(8*p5p6*(p7p11-p7p8)+24*p4p6*(-p7p11
     . +p7p8)+8*p3p6*(-p7p11+p7p8)+8*p6p7*(p1p11-p1p8-p2p11+p2p8+
     . p4p11-p4p8-3*p5p11+3*p5p8)))
      b(44)=w11*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-2*
     . hbcm2)+4*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmb*fmc+fmb2
     . )+4*(hbcm2*p5p6-2*p1p3*p4p6+2*p1p5*p3p6-2*p2p3*p5p6+2*p3p4*
     . p5p6-2*p3p5*p4p6))
      b(45)=4*ccc*w11*(-p3p6-3*p4p6+p5p6)
      b(46)=w11*ccc*(4*p5p6*(-fmb-fmc)+4*p3p6*(ffmcfmb*hbcm-fmb)+4*
     . hbcm*p4p6)
      b(47)=4*ccc*w2*(-p3p6-3*p4p6+p5p6)
      b(49)=w11*ccc*(4*p3p7*(3*fmb+3*fmc-hbcm)+8*(hbcm*p4p7+hbcm*p5p7
     . ))
      b(50)=8*ccc*w11*(p3p7+p4p7+p5p7)
      b(51)=w11*ccc*(4*p3p7*(fmb*hbcm-fmb2-fmc*hbcm+fmc2)+4*p5p7*(-2*
     . ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+4*p4p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+8*(-2*p1p2*p3p7-p1p3*p4p7-p1p3*
     . p5p7+2*p1p4*p3p7+p2p3*p4p7+p2p3*p5p7-p2p4*p3p7-p3p4*p4p7-p3p4*
     . p5p7+p3p5*p4p7+p3p5*p5p7))
      b(52)=8*ccc*w2*(p3p7+p4p7+p5p7)
      b(53)=ccc*(4*w2*(-p1p9-3*p2p9+2*p3p9+3*p4p9-p5p9)+4*w11*(-p1p11
     . +p1p8-3*p2p11+3*p2p8+2*p3p11-2*p3p8+3*p4p11-3*p4p8-p5p11+p5p8)
     . )
      b(54)=2*ccc*w11*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+2
     . *p1p3+6*p2p3-6*p3p4+2*p3p5)
      b(55)=6*ccc*w11*(fmb+fmc-hbcm)
      b(56)=2*ccc*w11*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2-4*p1p2+2*p1p3+
     . 4*p1p4-2*p2p3-2*p2p4+2*p3p4-2*p3p5)
      b(58)=w11*ccc*(4*p3p6*(-ffmcfmb*hbcm+2*fmb+fmc)+8*hbcm*p4p6)
      b(59)=w11*ccc*(4*p3p5*(fmb+fmc)+4*p1p3*(-2*fmb-2*fmc+hbcm)+2*(-
     . 2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+2*fmb*hbcm2+fmb2*hbcm+
     . 2*fmc*hbcm2-fmc2*hbcm+4*hbcm*p1p2-4*hbcm*p1p4+2*hbcm*p2p3+2*
     . hbcm*p2p4-2*hbcm*p3p4-hbcm3))
      b(60)=w2*ccc*(4*p3p6*(-ffmcfmb*hbcm+2*fmb+fmc)+8*hbcm*p4p6)
      b(65)=w11*ccc*(4*p6p7*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm+3
     . *hbcm2)+8*(-p1p3*p6p7+p2p3*p6p7-p3p4*p6p7+3*p3p5*p6p7+p3p6*
     . p3p7+3*p3p7*p4p6-p3p7*p5p6))
      b(66)=ccc*(w2*(8*p6p7*(hbcm*p1p9-hbcm*p2p9+hbcm*p4p9-3*hbcm*
     . p5p9)+8*p3p6*(ffmcfmb*hbcm*p7p9-2*fmb*p7p9-fmc*p7p9)-16*hbcm*
     . p4p6*p7p9)+w11*(16*p4p6*(-hbcm*p7p11+hbcm*p7p8)+8*p6p7*(hbcm*
     . p1p11-hbcm*p1p8-hbcm*p2p11+hbcm*p2p8+hbcm*p4p11-hbcm*p4p8-3*
     . hbcm*p5p11+3*hbcm*p5p8)+8*p3p6*(ffmcfmb*hbcm*p7p11-ffmcfmb*
     . hbcm*p7p8-2*fmb*p7p11+2*fmb*p7p8-fmc*p7p11+fmc*p7p8))+8*hbcm*
     . p7p13)
      b(68)=4*ccc*w11*(p3p6+3*p4p6-p5p6)
      b(71)=8*ccc*w11*(-p3p7-p4p7-p5p7)
      b(73)=12*ccc*w11*p6p7*(fmb+fmc-hbcm)
      b(74)=12*ccc*w11*p6p7*(-fmb-fmc+hbcm)
      b(75)=ccc*(4*w2*(2*fmb*p1p9-fmb*p3p9-fmb*p5p9+2*fmc*p1p9-fmc*
     . p3p9-fmc*p5p9-2*hbcm*p1p9+hbcm*p3p9+hbcm*p5p9)+4*w11*(2*fmb*
     . p1p11-2*fmb*p1p8-fmb*p3p11+fmb*p3p8-fmb*p5p11+fmb*p5p8+2*fmc*
     . p1p11-2*fmc*p1p8-fmc*p3p11+fmc*p3p8-fmc*p5p11+fmc*p5p8-2*hbcm*
     . p1p11+2*hbcm*p1p8+hbcm*p3p11-hbcm*p3p8+hbcm*p5p11-hbcm*p5p8))
      ans=ccc*(w2*(8*p3p5*(-2*p1p9+p3p9+p5p9)+8*p3p4*(p1p9+p2p9-p3p9-
     . p4p9)+8*p2p3*(-p1p9-p2p9+p3p9+p4p9)+8*p1p3*(p1p9+p2p9-p3p9-
     . p4p9)+4*(2*ffmcfmb*hbcm2*p1p9+2*ffmcfmb*hbcm2*p2p9-2*ffmcfmb*
     . hbcm2*p3p9-2*ffmcfmb*hbcm2*p4p9-2*fmb*hbcm*p1p9+fmb*hbcm*p3p9+
     . fmb*hbcm*p5p9+2*fmc*hbcm*p1p9-fmc*hbcm*p3p9-fmc*hbcm*p5p9-2*
     . hbcm2*p1p9+hbcm2*p3p9+hbcm2*p5p9))+w11*(8*p3p5*(-2*p1p11+2*
     . p1p8+p3p11-p3p8+p5p11-p5p8)+8*p3p4*(p1p11-p1p8+p2p11-p2p8-
     . p3p11+p3p8-p4p11+p4p8)+8*p2p3*(-p1p11+p1p8-p2p11+p2p8+p3p11-
     . p3p8+p4p11-p4p8)+8*p1p3*(p1p11-p1p8+p2p11-p2p8-p3p11+p3p8-
     . p4p11+p4p8)+4*(2*ffmcfmb*hbcm2*p1p11-2*ffmcfmb*hbcm2*p1p8+2*
     . ffmcfmb*hbcm2*p2p11-2*ffmcfmb*hbcm2*p2p8-2*ffmcfmb*hbcm2*p3p11
     . +2*ffmcfmb*hbcm2*p3p8-2*ffmcfmb*hbcm2*p4p11+2*ffmcfmb*hbcm2*
     . p4p8-2*fmb*hbcm*p1p11+2*fmb*hbcm*p1p8+fmb*hbcm*p3p11-fmb*hbcm*
     . p3p8+fmb*hbcm*p5p11-fmb*hbcm*p5p8+2*fmc*hbcm*p1p11-2*fmc*hbcm*
     . p1p8-fmc*hbcm*p3p11+fmc*hbcm*p3p8-fmc*hbcm*p5p11+fmc*hbcm*p5p8
     . -2*hbcm2*p1p11+2*hbcm2*p1p8+hbcm2*p3p11-hbcm2*p3p8+hbcm2*p5p11
     . -hbcm2*p5p8)))
      b(76)=ans
      b(77)=ccc*(4*w2*(ffmcfmb*hbcm2*p6p9-2*fmb*hbcm*p6p9+fmc*hbcm*
     . p6p9-2*hbcm2*p6p9-4*p3p5*p6p9+4*p3p6*p5p9)+w11*(16*p3p6*(p5p11
     . -p5p8)+16*p3p5*(-p6p11+p6p8)+4*(ffmcfmb*hbcm2*p6p11-ffmcfmb*
     . hbcm2*p6p8-2*fmb*hbcm*p6p11+2*fmb*hbcm*p6p8+fmc*hbcm*p6p11-fmc
     . *hbcm*p6p8-2*hbcm2*p6p11+2*hbcm2*p6p8))+4*p3p13)
      b(78)=ccc*(4*w2*(ffmcfmb*hbcm*p6p9+2*fmb*p6p9+fmc*p6p9-2*hbcm*
     . p6p9)+4*w11*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+2*fmb*p6p11-
     . 2*fmb*p6p8+fmc*p6p11-fmc*p6p8-2*hbcm*p6p11+2*hbcm*p6p8))
      b(79)=12*ccc*w2*p6p7*(fmb+fmc-hbcm)
      b(80)=w2*ccc*(4*p5p7*(-fmb-fmc+hbcm)+4*p4p7*(-fmb-fmc+hbcm)+4*
     . p3p7*(-fmb-fmc+hbcm))
      b(83)=4*ccc*(hbcm*p3p7-hbcm*p4p7-hbcm*p5p7)
      b(84)=4*ccc*(p3p7+p4p7+p5p7)
      b(85)=4*ccc*(-fmb*hbcm+fmc*hbcm-hbcm2+p1p3+p2p3-p3p4-p3p5)
      b(87)=4*ccc*hbcm
      b(89)=4*ccc*hbcm
      b(90)=w11*ccc*(4*p5p7*(-fmb-fmc+hbcm)+4*p4p7*(-fmb-fmc+hbcm)+4*
     . p3p7*(-fmb-fmc+hbcm))
      b(91)=w11*ccc*(4*p5p7*(fmb+fmc-hbcm)+4*p4p7*(fmb+fmc-hbcm)+4*
     . p3p7*(fmb+fmc-hbcm))
      do 200 n=1,94 
         c(n,1)=c(n,1)+0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp25_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3+2*ffmcfmb*p2p3
     . -fmb2+hbcm2+2*p1p2-2*p1p3-2*p2p3))
      b(1)=w1*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc-
     . fmb*hbcm+fmb2+fmc*hbcm)+16*(p1p2*p6p7-p1p3*p6p7-2*p1p5*p6p7))
      b(11)=w1*ccc*(4*p6p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*
     . p3p5)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p1p3)*(ffmcfmb*hbcm-2*fmc-hbcm)+8*(hbcm*p1p2*
     . p6p7-2*hbcm*p1p5*p6p7))
      b(29)=ccc*(w20*(4*p6p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7
     . *p3p5)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p1p3)*(ffmcfmb*hbcm-2*fmc-hbcm)+8*(hbcm*p1p2*
     . p6p7-2*hbcm*p1p5*p6p7))+4*hbcm*p6p7)
      b(30)=ccc*(w1*(32*(p6p7*p1p3)*(2*p1p8+p2p8-p5p8)+8*p6p7*(-
     . ffmcfmb*hbcm2*p1p8+ffmcfmb*hbcm2*p2p8-fmb*hbcm*p1p8+fmb*hbcm*
     . p2p8-5*hbcm2*p1p8-hbcm2*p2p8)+32*(p1p8*p2p3*p6p7-p1p8*p3p4*
     . p6p7-2*p1p8*p3p5*p6p7))+w20*(64*(p6p7*p3p5)*(-p1p10-p1p11)+32*
     . (p6p7*p3p4)*(-p1p10-p1p11)+32*(p6p7*p2p3)*(p1p10+p1p11)+32*(
     . p6p7*p1p3)*(2*p1p10+2*p1p11+p2p10+p2p11-p5p10-p5p11)+8*p6p7*(-
     . ffmcfmb*hbcm2*p1p10-ffmcfmb*hbcm2*p1p11+ffmcfmb*hbcm2*p2p10+
     . ffmcfmb*hbcm2*p2p11-fmb*hbcm*p1p10-fmb*hbcm*p1p11+fmb*hbcm*
     . p2p10+fmb*hbcm*p2p11-5*hbcm2*p1p10-5*hbcm2*p1p11-hbcm2*p2p10-
     . hbcm2*p2p11))+8*p6p7*(-p3p10+p3p11))
      b(33)=ccc*(8*w1*p6p7*(-2*ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p5p8+
     . fmb*p5p8+2*fmc*p1p8-hbcm*p5p8)+8*w20*p6p7*(-2*ffmcfmb*hbcm*
     . p1p10-2*ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p5p10+ffmcfmb*hbcm*
     . p5p11+fmb*p5p10+fmb*p5p11+2*fmc*p1p10+2*fmc*p1p11-hbcm*p5p10-
     . hbcm*p5p11))
      b(39)=w20*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-p1p3*
     . p6p7-p2p3*p6p7+2*p3p5*p6p7))
      b(40)=ccc*(w20*(4*p6p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7
     . *p3p5)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p1p3)*(ffmcfmb*hbcm-2*fmc-hbcm)+8*(hbcm*p1p2*
     . p6p7-2*hbcm*p1p5*p6p7))-4*hbcm*p6p7)
      b(41)=w20*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc-fmb*hbcm+fmb2+fmc*hbcm)+16*(p1p2*p6p7-p1p3*p6p7-2*p1p5*
     . p6p7))
      b(43)=ccc*(16*w1*p6p7*(-p1p8-p2p8+2*p5p8)+16*w20*p6p7*(-p1p10-
     . p1p11-p2p10-p2p11+2*p5p10+2*p5p11))
      b(63)=w20*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-p1p3*
     . p6p7-p2p3*p6p7+2*p3p5*p6p7))
      b(64)=w20*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc-fmb*hbcm+fmb2+fmc*hbcm)+16*(p1p2*p6p7-p1p3*p6p7-2*p1p5*
     . p6p7))
      b(65)=w1*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-p1p3*p6p7
     . -p2p3*p6p7+2*p3p5*p6p7))
      b(66)=ccc*(16*w20*p6p7*(hbcm*p5p10+hbcm*p5p11)+16*hbcm*p5p8*
     . p6p7*w1)
      b(73)=8*ccc*w20*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(74)=8*ccc*w1*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(81)=8*ccc*w20*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      do 200 n=1,94 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
      end 
 
      subroutine amp24_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2+
     . fmc2+hbcm2+2*p3p4))
      b(11)=w1*ccc*(4*p6p7*(fmb2*hbcm-fmc2*hbcm+hbcm3)+8*(-hbcm*p1p3*
     . p6p7-hbcm*p2p3*p6p7+hbcm*p3p5*p6p7))
      b(30)=8*ccc*w1*p6p7*(fmb*hbcm*p1p8-fmb*hbcm*p2p8+fmb*hbcm*p4p8+
     . fmb*hbcm*p5p8-fmc*hbcm*p1p8+fmc*hbcm*p2p8-fmc*hbcm*p4p8-fmc*
     . hbcm*p5p8)
      b(33)=8*ccc*w1*p6p7*(hbcm*p1p8+hbcm*p2p8-hbcm*p4p8-hbcm*p5p8)
      b(65)=8*ccc*w1*p6p7*(-fmb*hbcm+fmc*hbcm)
      b(66)=16*ccc*hbcm*p4p8*p6p7*w1
      b(74)=-8*ccc*hbcm*p6p7*w1
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
      end 
 
      subroutine amp23_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(12)=w2*ccc*(4*p6p7*(-fmb2*hbcm+fmc2*hbcm-hbcm3)-8*hbcm*p3p5*
     . p6p7)
      b(42)=8*ccc*w2*p6p7*(fmb*hbcm-fmc*hbcm)
      b(66)=16*ccc*hbcm*p5p9*p6p7*w2
      b(79)=8*ccc*hbcm*p6p7*w2
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
      end 
 
      subroutine amp22_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*
     . ffmcfmb*p2p3-fmc2+2*p1p2))
      b(2)=w2*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc+
     . fmc2)+16*(-p1p2*p6p7+p1p3*p6p7+2*p1p5*p6p7))
      b(12)=w2*ccc*(8*(p6p7*p1p3)*(-2*fmb-fmc+2*hbcm)+4*p6p7*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+8*(
     . -fmc*p2p3*p6p7-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7))
      b(29)=ccc*(w17*(4*p6p7*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p3p5)
     . *(-ffmcfmb*hbcm+fmc)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm+2*fmb+fmc)+
     . 8*(-fmc*p2p3*p6p7-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7))+4*hbcm*
     . p6p7)
      b(30)=ccc*(w2*(32*(p6p7*p1p3)*(-2*p1p9-p2p9+p5p9)+8*p6p7*(-
     . ffmcfmb*hbcm2*p1p9+ffmcfmb*hbcm2*p2p9+2*fmb*hbcm*p1p9-fmc*hbcm
     . *p1p9-fmc*hbcm*p2p9+6*hbcm2*p1p9)+32*(-p1p9*p2p3*p6p7+p1p9*
     . p3p4*p6p7+2*p1p9*p3p5*p6p7))+w17*(64*(p6p7*p3p5)*(p1p10+p1p11)
     . +32*(p6p7*p3p4)*(p1p10+p1p11)+32*(p6p7*p2p3)*(-p1p10-p1p11)+32
     . *(p6p7*p1p3)*(-2*p1p10-2*p1p11-p2p10-p2p11+p5p10+p5p11)+8*p6p7
     . *(-ffmcfmb*hbcm2*p1p10-ffmcfmb*hbcm2*p1p11+ffmcfmb*hbcm2*p2p10
     . +ffmcfmb*hbcm2*p2p11+2*fmb*hbcm*p1p10+2*fmb*hbcm*p1p11-fmc*
     . hbcm*p1p10-fmc*hbcm*p1p11-fmc*hbcm*p2p10-fmc*hbcm*p2p11+6*
     . hbcm2*p1p10+6*hbcm2*p1p11))+8*p6p7*(p3p10-p3p11))
      b(33)=ccc*(8*w2*p6p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+
     . ffmcfmb*hbcm*p5p9+2*fmb*p1p9+fmc*p1p9+fmc*p2p9-fmc*p5p9-2*hbcm
     . *p1p9)+8*w17*p6p7*(-ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p11-
     . ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p2p11+ffmcfmb*hbcm*p5p10+
     . ffmcfmb*hbcm*p5p11-2*fmb*p1p10-2*fmb*p1p11-fmc*p1p10-fmc*p1p11
     . +fmc*p2p10+fmc*p2p11-fmc*p5p10-fmc*p5p11))
      b(39)=w17*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*
     . p6p7+p2p3*p6p7-2*p3p5*p6p7))
      b(40)=ccc*(w17*(4*p6p7*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p3p5)
     . *(-ffmcfmb*hbcm+fmc)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm+2*fmb+fmc)+
     . 8*(-fmc*p2p3*p6p7-hbcm*p1p2*p6p7+2*hbcm*p1p5*p6p7))-4*hbcm*
     . p6p7)
      b(41)=w17*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc+fmc2)+16*(-p1p2*p6p7+p1p3*p6p7+2*p1p5*p6p7))
      b(42)=w2*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*p6p7
     . +p2p3*p6p7-2*p3p5*p6p7))
      b(43)=ccc*(16*w2*p6p7*(p1p9+p2p9-2*p5p9)+16*w17*p6p7*(p1p10+
     . p1p11+p2p10+p2p11-2*p5p10-2*p5p11))
      b(63)=w17*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*
     . p6p7+p2p3*p6p7-2*p3p5*p6p7))
      b(64)=w17*ccc*(4*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc+fmc2)+16*(-p1p2*p6p7+p1p3*p6p7+2*p1p5*p6p7))
      b(66)=ccc*(16*w2*p6p7*(hbcm*p1p9+hbcm*p2p9-hbcm*p5p9)+16*w17*
     . p6p7*(hbcm*p1p10+hbcm*p1p11+hbcm*p2p10+hbcm*p2p11-hbcm*p5p10-
     . hbcm*p5p11))
      b(73)=8*ccc*w17*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(79)=8*ccc*w2*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(81)=8*ccc*w17*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
      end 
 
      subroutine amp21_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=w5*ccc*(8*p6p7*(ffmcfmb**2*hbcm2-fmc2)+16*(-ffmcfmb*p3p6*
     . p3p7-2*ffmcfmb*p3p6*p5p7+p1p2*p6p7-p1p3*p6p7-p1p5*p6p7))
      b(3)=ccc*(8*w8*(ffmcfmb**2*hbcm2*p7p11+2*ffmcfmb*p1p11*p3p7-
     . fmc2*p7p11+2*p1p11*p4p7+2*p1p4*p7p11)+w5*(16*p1p4*(-p7p10+p7p8
     . )+16*p4p7*(-p1p10+p1p8)+16*p3p7*(-ffmcfmb*p1p10+ffmcfmb*p1p8)+
     . 8*(-ffmcfmb**2*hbcm2*p7p10+ffmcfmb**2*hbcm2*p7p8+fmc2*p7p10-
     . fmc2*p7p8))+8*w12*(-ffmcfmb**2*hbcm2*p7p10-2*ffmcfmb*p1p10*
     . p3p7+fmc2*p7p10-2*p1p10*p4p7-2*p1p4*p7p10)+8*(3*p1p12+2*p2p12-
     . 3*p3p12-3*p4p12-3*p5p12-p7p10+p7p11))
      b(4)=w5*ccc*((p3p7*p1p3)*(-16*ffmcfmb+16)+16*p5p7*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2-fmc*
     . hbcm+fmc2)+16*(p1p3*p5p7-p1p4*p3p7))
      ans2=w12*(16*p3p4*(-ffmcfmb*hbcm*p7p10+fmc*p7p10)+16*p1p3*(
     . ffmcfmb*hbcm*p7p10-fmc*p7p10)+16*p3p7*(ffmcfmb*hbcm*p1p10+
     . ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p5p10-fmc*p2p10+fmc*p5p10-hbcm
     . *p1p10)+8*(-ffmcfmb**2*hbcm3*p7p10+2*ffmcfmb*fmc*hbcm2*p7p10-
     . fmc2*hbcm*p7p10-2*hbcm*p1p10*p5p7+2*hbcm*p1p4*p7p10))+8*(-fmb*
     . p3p12-fmc*p3p12-hbcm*p1p12+hbcm*p2p12+3*hbcm*p3p12+2*hbcm*
     . p4p12+2*hbcm*p5p12-3*hbcm*p7p11)
      ans1=w8*(16*p3p4*(ffmcfmb*hbcm*p7p11-fmc*p7p11)+16*p1p3*(-
     . ffmcfmb*hbcm*p7p11+fmc*p7p11)+16*p3p7*(-ffmcfmb*hbcm*p1p11-
     . ffmcfmb*hbcm*p2p11+ffmcfmb*hbcm*p5p11+fmc*p2p11-fmc*p5p11+hbcm
     . *p1p11)+8*(ffmcfmb**2*hbcm3*p7p11-2*ffmcfmb*fmc*hbcm2*p7p11+
     . fmc2*hbcm*p7p11+2*hbcm*p1p11*p5p7-2*hbcm*p1p4*p7p11))+w5*(16*
     . p1p4*(hbcm*p7p10-hbcm*p7p8)+16*p5p7*(-hbcm*p1p10+hbcm*p1p8)+16
     . *p3p4*(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8+fmc*p7p10-fmc*
     . p7p8)+16*p1p3*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p8-fmc*p7p10+
     . fmc*p7p8)+16*p3p7*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p8+
     . ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p5p10+
     . ffmcfmb*hbcm*p5p8-fmc*p2p10+fmc*p2p8+fmc*p5p10-fmc*p5p8-hbcm*
     . p1p10+hbcm*p1p8)+8*(-ffmcfmb**2*hbcm3*p7p10+ffmcfmb**2*hbcm3*
     . p7p8+2*ffmcfmb*fmc*hbcm2*p7p10-2*ffmcfmb*fmc*hbcm2*p7p8-fmc2*
     . hbcm*p7p10+fmc2*hbcm*p7p8))+ans2
      ans=ccc*ans1
      b(6)=ans
      b(7)=ccc*(w8*(16*p3p6*(-ffmcfmb*p2p11+2*ffmcfmb*p3p11+2*ffmcfmb
     . *p4p11+p1p11)+8*(ffmcfmb**2*hbcm2*p6p11-fmc2*p6p11+6*p1p11*
     . p4p6+2*p1p2*p6p11-2*p1p3*p6p11-2*p1p5*p6p11))+w5*(16*p1p5*(
     . p6p10-p6p8)+16*p1p3*(p6p10-p6p8)+16*p1p2*(-p6p10+p6p8)+48*p4p6
     . *(-p1p10+p1p8)+16*p3p6*(ffmcfmb*p2p10-ffmcfmb*p2p8-2*ffmcfmb*
     . p3p10+2*ffmcfmb*p3p8-2*ffmcfmb*p4p10+2*ffmcfmb*p4p8-p1p10+p1p8
     . )+8*(-ffmcfmb**2*hbcm2*p6p10+ffmcfmb**2*hbcm2*p6p8+fmc2*p6p10-
     . fmc2*p6p8))+w12*(16*p3p6*(ffmcfmb*p2p10-2*ffmcfmb*p3p10-2*
     . ffmcfmb*p4p10-p1p10)+8*(-ffmcfmb**2*hbcm2*p6p10+fmc2*p6p10-6*
     . p1p10*p4p6-2*p1p2*p6p10+2*p1p3*p6p10+2*p1p5*p6p10))+8*(-2*
     . p1p13-3*p2p13+4*p3p13+p4p13+3*p5p13-p6p10+p6p11))
      b(8)=w5*ccc*(16*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(-ffmcfmb
     . **2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*
     . hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6-p1p2*p3p6
     . -3*p1p3*p4p6+p1p5*p3p6))
      ans3=w12*(16*p3p5*(ffmcfmb*hbcm*p6p10-fmc*p6p10)+16*p4p6*(-3*
     . ffmcfmb*hbcm*p3p10+3*fmc*p3p10-hbcm*p1p10)+16*p2p3*(-ffmcfmb*
     . hbcm*p6p10+fmc*p6p10)+16*p3p6*(-ffmcfmb**2*hbcm*p3p10+ffmcfmb*
     . fmb*p3p10+2*ffmcfmb*fmc*p3p10-2*ffmcfmb*hbcm*p1p10-ffmcfmb*
     . hbcm*p2p10+ffmcfmb*hbcm*p3p10+2*ffmcfmb*hbcm*p4p10+fmc*p1p10-
     . fmc*p4p10)+8*(-ffmcfmb**2*hbcm3*p6p10+2*ffmcfmb*fmc*hbcm2*
     . p6p10+2*ffmcfmb*hbcm3*p6p10-2*fmc*hbcm2*p6p10-fmc2*hbcm*p6p10+
     . 2*hbcm*p1p2*p6p10-2*hbcm*p1p3*p6p10-2*hbcm*p1p5*p6p10))+8*(-4*
     . fmb*p3p13-4*fmc*p3p13+hbcm*p1p13+2*hbcm*p2p13-hbcm*p4p13-2*
     . hbcm*p5p13+hbcm*p6p10-hbcm*p6p11)
      ans2=w5*(16*p1p5*(-hbcm*p6p10+hbcm*p6p8)+16*p1p3*(-hbcm*p6p10+
     . hbcm*p6p8)+16*p1p2*(hbcm*p6p10-hbcm*p6p8)+16*p3p5*(ffmcfmb*
     . hbcm*p6p10-ffmcfmb*hbcm*p6p8-fmc*p6p10+fmc*p6p8)+16*p4p6*(-3*
     . ffmcfmb*hbcm*p3p10+3*ffmcfmb*hbcm*p3p8+3*fmc*p3p10-3*fmc*p3p8-
     . hbcm*p1p10+hbcm*p1p8)+16*p2p3*(-ffmcfmb*hbcm*p6p10+ffmcfmb*
     . hbcm*p6p8+fmc*p6p10-fmc*p6p8)+16*p3p6*(-ffmcfmb**2*hbcm*p3p10+
     . ffmcfmb**2*hbcm*p3p8+ffmcfmb*fmb*p3p10-ffmcfmb*fmb*p3p8+2*
     . ffmcfmb*fmc*p3p10-2*ffmcfmb*fmc*p3p8-2*ffmcfmb*hbcm*p1p10+2*
     . ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p2p8+ffmcfmb
     . *hbcm*p3p10-ffmcfmb*hbcm*p3p8+2*ffmcfmb*hbcm*p4p10-2*ffmcfmb*
     . hbcm*p4p8+fmc*p1p10-fmc*p1p8-fmc*p4p10+fmc*p4p8)+8*(-ffmcfmb**
     . 2*hbcm3*p6p10+ffmcfmb**2*hbcm3*p6p8+2*ffmcfmb*fmc*hbcm2*p6p10-
     . 2*ffmcfmb*fmc*hbcm2*p6p8+2*ffmcfmb*hbcm3*p6p10-2*ffmcfmb*hbcm3
     . *p6p8-2*fmc*hbcm2*p6p10+2*fmc*hbcm2*p6p8-fmc2*hbcm*p6p10+fmc2*
     . hbcm*p6p8))+ans3
      ans1=w8*(16*p3p5*(-ffmcfmb*hbcm*p6p11+fmc*p6p11)+16*p2p3*(
     . ffmcfmb*hbcm*p6p11-fmc*p6p11)+16*p3p6*(2*ffmcfmb*hbcm*p1p11+
     . ffmcfmb*hbcm*p2p11-2*ffmcfmb*hbcm*p3p11-2*ffmcfmb*hbcm*p4p11-
     . fmc*p1p11+fmc*p3p11+fmc*p4p11)+8*(ffmcfmb**2*hbcm3*p6p11-2*
     . ffmcfmb*fmc*hbcm2*p6p11-2*ffmcfmb*hbcm3*p6p11+2*fmc*hbcm2*
     . p6p11+fmc2*hbcm*p6p11+2*hbcm*p1p11*p4p6-2*hbcm*p1p2*p6p11+2*
     . hbcm*p1p3*p6p11+2*hbcm*p1p5*p6p11))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=w5*ccc*(16*(p4p6*p3p7)*(ffmcfmb*hbcm-fmc)+16*(p5p7*p3p6)*
     . (-2*ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+16*(
     . p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+8*p6p7*(-ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-2*fmc*hbcm2-fmc2*hbcm)+16*(
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-2*ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+16
     . *(hbcm*p1p2*p6p7-hbcm*p1p3*p6p7-hbcm*p1p5*p6p7))
      b(13)=4*ccc*w5*(-2*ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-fmb*fmc+3*fmc2+2*p1p2-2*p1p3-8*p1p4)
      b(15)=w5*ccc*(24*p3p4*(-ffmcfmb*hbcm+fmc)+8*p2p3*(ffmcfmb*hbcm-
     . fmc)+8*p1p3*(2*ffmcfmb*hbcm-2*fmc+hbcm)+4*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+2*fmc*hbcm2-2*fmc2*hbcm-2*hbcm*p1p2+6*hbcm*p1p4))
      b(17)=ccc*(w5*(8*p3p6*(ffmcfmb*hbcm+hbcm)+24*hbcm*p4p6)+w12*(8*
     . p3p6*(ffmcfmb*hbcm+hbcm)+24*hbcm*p4p6))
      b(18)=ccc*(4*w5*(-fmb-fmc)+4*w12*(-fmb-fmc))
      b(19)=ccc*(4*w5*(-4*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-2*hbcm2+6*
     . p1p3+2*p2p3-8*p3p4)+4*w12*(-4*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-
     . 2*hbcm2+6*p1p3+2*p2p3-8*p3p4))
      b(20)=ccc*(w5*(8*p3p7*(-ffmcfmb*hbcm+2*hbcm)+24*hbcm*p5p7)+w12*
     . (8*p3p7*(-ffmcfmb*hbcm+2*hbcm)+24*hbcm*p5p7))
      b(21)=ccc*(8*w8*(2*hbcm*p1p11+hbcm*p2p11-3*hbcm*p3p11-3*hbcm*
     . p4p11)+8*w5*(fmb*p3p10-fmb*p3p8+fmc*p3p10-fmc*p3p8-2*hbcm*
     . p1p10+2*hbcm*p1p8-hbcm*p2p10+hbcm*p2p8+3*hbcm*p3p10-3*hbcm*
     . p3p8+3*hbcm*p4p10-3*hbcm*p4p8)+8*w12*(fmb*p3p10+fmc*p3p10-2*
     . hbcm*p1p10-hbcm*p2p10+3*hbcm*p3p10+3*hbcm*p4p10))
      b(22)=ccc*(w5*(24*p3p4*(ffmcfmb*hbcm-fmc)+8*p2p3*(-ffmcfmb*hbcm
     . +fmc)+8*p1p3*(-2*ffmcfmb*hbcm+2*fmc-hbcm)+4*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-2*fmc*hbcm2+2*fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4))+w12*(24
     . *p3p4*(ffmcfmb*hbcm-fmc)+8*p2p3*(-ffmcfmb*hbcm+fmc)+8*p1p3*(-2
     . *ffmcfmb*hbcm+2*fmc-hbcm)+4*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*
     . hbcm2-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmc*
     . hbcm2+2*fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4)))
      b(23)=w5*ccc*(8*p3p6*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p4p6)
      b(24)=4*ccc*w5*(fmb+fmc)
      b(25)=4*ccc*w5*(4*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+2*hbcm2-6*
     . p1p3-2*p2p3+8*p3p4)
      b(26)=w5*ccc*(8*p3p7*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p5p7)
      b(27)=ccc*(w5*(24*p4p6*(ffmcfmb*hbcm-fmc)+8*p3p6*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmc))+w12*(24*p4p6
     . *(ffmcfmb*hbcm-fmc)+8*p3p6*(ffmcfmb**2*hbcm-ffmcfmb*fmb-2*
     . ffmcfmb*fmc+ffmcfmb*hbcm-fmc)))
      b(28)=ccc*(w5*(16*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmc*hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-2*ffmcfmb*p3p4*p3p6
     . -p1p2*p3p6-3*p1p3*p4p6+p1p5*p3p6))+w12*(16*p4p6*(ffmcfmb*hbcm2
     . -fmc*hbcm)+8*p3p6*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-2*
     . ffmcfmb*p3p4*p3p6-p1p2*p3p6-3*p1p3*p4p6+p1p5*p3p6)))
      b(29)=ccc*(w5*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6
     . )*(2*ffmcfmb*hbcm-fmc)+16*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(
     . p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+8*p6p7*(ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmc*hbcm2+fmc2*hbcm)+16*(
     . p3p7*p3p6)*(-ffmcfmb**2*hbcm+2*ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+
     . 16*(-hbcm*p1p2*p6p7+hbcm*p1p3*p6p7+hbcm*p1p5*p6p7))+w12*(16*(
     . p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6)*(2*ffmcfmb*hbcm-
     . fmc)+16*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p2p3)*(
     . ffmcfmb*hbcm-fmc)+8*p6p7*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2
     . -2*ffmcfmb*hbcm3+2*fmc*hbcm2+fmc2*hbcm)+16*(p3p7*p3p6)*(-
     . ffmcfmb**2*hbcm+2*ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+16*(-hbcm*p1p2
     . *p6p7+hbcm*p1p3*p6p7+hbcm*p1p5*p6p7)))
      ans4=16*p3p7*(ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*fmc*hbcm*p6p8+
     . ffmcfmb*hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm
     . *p6p8-fmc2*p6p10+fmc2*p6p8)+16*p3p6*(ffmcfmb**2*hbcm2*p7p10-
     . ffmcfmb*fmb*hbcm*p7p8+ffmcfmb*fmc*hbcm*p7p10+ffmcfmb*hbcm2*
     . p7p8+fmc*hbcm*p7p8-fmc2*p7p8)+32*(-ffmcfmb*p2p3*p3p6*p7p8+p1p2
     . *p3p6*p7p8-p1p5*p3p6*p7p8)
      ans3=32*(p3p7*p1p5)*(-p6p10+p6p8)+32*(p5p7*p1p3)*(-p6p10+p6p8)+
     . 32*(p4p6*p1p3)*(p7p10+2*p7p8)+32*(p3p7*p1p2)*(p6p10-p6p8)+96*(
     . p4p6*p3p7)*(p1p10-p1p8)+32*(p5p7*p3p6)*(p1p10-p1p8)+32*(p4p7*
     . p3p6)*(p1p10-p1p8)+32*(p6p7*p3p5)*(p1p10-p1p8)+32*(p6p7*p3p4)*
     . (p1p10-p1p8)+32*(p6p7*p2p3)*(-p1p10+p1p8)+32*(p6p7*p1p3)*(-
     . p1p10+p1p8-p2p10+p2p8+p5p10-p5p8)+32*(p3p6*p3p4)*(ffmcfmb*
     . p7p10+ffmcfmb*p7p8)+32*(p3p7*p3p6)*(-2*ffmcfmb*p2p10+2*ffmcfmb
     . *p2p8+2*ffmcfmb*p3p10-2*ffmcfmb*p3p8+2*ffmcfmb*p4p10-2*ffmcfmb
     . *p4p8+ffmcfmb*p5p10-ffmcfmb*p5p8+2*p1p10-2*p1p8)+32*(p3p7*p1p3
     . )*(ffmcfmb*p6p10-ffmcfmb*p6p8-2*p6p10+2*p6p8)+16*p5p7*(ffmcfmb
     . *hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm*p6p8)+
     . 16*p4p6*(-ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p8+fmc*hbcm*
     . p7p10+fmc*hbcm*p7p8)+16*p6p7*(ffmcfmb*hbcm2*p2p10-ffmcfmb*
     . hbcm2*p2p8-ffmcfmb*hbcm2*p5p10+ffmcfmb*hbcm2*p5p8-fmc*hbcm*
     . p2p10+fmc*hbcm*p2p8+fmc*hbcm*p5p10-fmc*hbcm*p5p8+2*hbcm2*p1p10
     . -2*hbcm2*p1p8)+ans4
      ans2=w5*ans3
      ans6=(16*p3p4*(p6p12-3*p7p13)+16*p6p7*(-p3p10+4*p3p11)+16*p2p3*
     . (-p6p12+4*p7p13)+16*p1p3*(p6p12+3*p7p13)+16*p3p7*(ffmcfmb*
     . p3p13+3*p1p13+2*p2p13-3*p3p13-2*p4p13-3*p5p13+p6p10)+16*p3p6*(
     . -ffmcfmb*p3p12-2*p1p12+p3p12+p4p12+p5p12-p7p11)+8*(-ffmcfmb*
     . hbcm2*p6p12+ffmcfmb*hbcm2*p7p13-fmb*hbcm*p6p12+fmb*hbcm*p7p13+
     . 2*fmc*hbcm*p6p12+hbcm2*p6p12-7*hbcm2*p7p13+6*p3p13*p4p7-6*p3p5
     . *p7p13))
      ans5=w12*(32*(p6p7*p1p3)*(-p1p10-p2p10+p5p10)+32*(p3p7*p3p6)*(-
     . 2*ffmcfmb*p2p10+2*ffmcfmb*p3p10+2*ffmcfmb*p4p10+ffmcfmb*p5p10+
     . 2*p1p10)+32*(p3p7*p1p3)*(ffmcfmb*p6p10-2*p6p10)+16*p5p7*(
     . ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+16*p4p6*(-ffmcfmb*hbcm2*
     . p7p10+fmc*hbcm*p7p10)+16*p6p7*(ffmcfmb*hbcm2*p2p10-ffmcfmb*
     . hbcm2*p5p10-fmc*hbcm*p2p10+fmc*hbcm*p5p10+2*hbcm2*p1p10)+16*
     . p3p7*(ffmcfmb*fmc*hbcm*p6p10+ffmcfmb*hbcm2*p6p10-fmc*hbcm*
     . p6p10-fmc2*p6p10)+16*p3p6*(ffmcfmb**2*hbcm2*p7p10+ffmcfmb*fmc*
     . hbcm*p7p10)+32*(ffmcfmb*p3p4*p3p6*p7p10-p1p10*p2p3*p6p7+p1p10*
     . p3p4*p6p7+p1p10*p3p5*p6p7+p1p10*p3p6*p4p7+p1p10*p3p6*p5p7+3*
     . p1p10*p3p7*p4p6+p1p2*p3p7*p6p10+p1p3*p4p6*p7p10-p1p3*p5p7*
     . p6p10-p1p5*p3p7*p6p10))+ans6
      ans1=w8*(32*(p6p7*p1p3)*(p1p11+p2p11-p5p11)+32*(p3p7*p3p6)*(2*
     . ffmcfmb*p2p11-2*ffmcfmb*p3p11-2*ffmcfmb*p4p11-ffmcfmb*p5p11-2*
     . p1p11)+32*(p3p7*p1p3)*(-ffmcfmb*p6p11+2*p6p11)+16*p5p7*(-
     . ffmcfmb*hbcm2*p6p11+fmc*hbcm*p6p11)+16*p4p6*(ffmcfmb*hbcm2*
     . p7p11-fmc*hbcm*p7p11)+16*p6p7*(-ffmcfmb*hbcm2*p2p11+ffmcfmb*
     . hbcm2*p5p11+fmc*hbcm*p2p11-fmc*hbcm*p5p11-2*hbcm2*p1p11)+16*
     . p3p7*(-ffmcfmb*fmc*hbcm*p6p11-ffmcfmb*hbcm2*p6p11+fmc*hbcm*
     . p6p11+fmc2*p6p11)+16*p3p6*(-ffmcfmb**2*hbcm2*p7p11-ffmcfmb*fmc
     . *hbcm*p7p11)+32*(-ffmcfmb*p3p4*p3p6*p7p11+p1p11*p2p3*p6p7-
     . p1p11*p3p4*p6p7-p1p11*p3p5*p6p7-p1p11*p3p6*p4p7-p1p11*p3p6*
     . p5p7-3*p1p11*p3p7*p4p6-p1p2*p3p7*p6p11-p1p3*p4p6*p7p11+p1p3*
     . p5p7*p6p11+p1p5*p3p7*p6p11))+ans2+ans5
      ans=ccc*ans1
      b(30)=ans
      b(31)=w5*ccc*(24*p4p6*(-ffmcfmb*hbcm+fmc)+8*p3p6*(-ffmcfmb**2*
     . hbcm+ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm+fmc))
      ans2=8*(ffmcfmb*hbcm*p6p12+ffmcfmb*hbcm*p7p13-fmc*p6p12-fmc*
     . p7p13)
      ans1=w8*(16*p4p7*(-ffmcfmb*hbcm*p6p11+fmc*p6p11)+16*p4p6*(-
     . ffmcfmb*hbcm*p7p11+fmc*p7p11)+16*p6p7*(ffmcfmb*hbcm*p2p11-
     . ffmcfmb*hbcm*p5p11-fmc*p2p11+fmc*p5p11)+16*p3p7*(-ffmcfmb**2*
     . hbcm*p6p11+ffmcfmb*fmc*p6p11)+16*p3p6*(-ffmcfmb**2*hbcm*p7p11+
     . ffmcfmb*fmc*p7p11))+w5*(16*p4p7*(ffmcfmb*hbcm*p6p10-ffmcfmb*
     . hbcm*p6p8-fmc*p6p10+fmc*p6p8)+16*p4p6*(ffmcfmb*hbcm*p7p10-
     . ffmcfmb*hbcm*p7p8-fmc*p7p10+fmc*p7p8)+16*p6p7*(-ffmcfmb*hbcm*
     . p2p10+ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p5p10-ffmcfmb*hbcm*p5p8+
     . fmc*p2p10-fmc*p2p8-fmc*p5p10+fmc*p5p8)+16*p3p7*(ffmcfmb**2*
     . hbcm*p6p10-ffmcfmb**2*hbcm*p6p8-ffmcfmb*fmc*p6p10+ffmcfmb*fmc*
     . p6p8)+16*p3p6*(ffmcfmb**2*hbcm*p7p10-ffmcfmb**2*hbcm*p7p8-
     . ffmcfmb*fmc*p7p10+ffmcfmb*fmc*p7p8))+w12*(16*p4p7*(ffmcfmb*
     . hbcm*p6p10-fmc*p6p10)+16*p4p6*(ffmcfmb*hbcm*p7p10-fmc*p7p10)+
     . 16*p6p7*(-ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p5p10+fmc*p2p10-fmc*
     . p5p10)+16*p3p7*(ffmcfmb**2*hbcm*p6p10-ffmcfmb*fmc*p6p10)+16*
     . p3p6*(ffmcfmb**2*hbcm*p7p10-ffmcfmb*fmc*p7p10))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(38)=ccc*(w8*(16*p3p7*(p2p11-p5p11)+16*(-ffmcfmb*hbcm2*p7p11+
     . p1p3*p7p11-p3p4*p7p11))+w5*(16*p3p4*(p7p10-p7p8)+16*p3p7*(-
     . p2p10+p2p8+p5p10-p5p8)+16*p1p3*(-p7p10+p7p8)+16*(ffmcfmb*hbcm2
     . *p7p10-ffmcfmb*hbcm2*p7p8))+w12*(16*p3p7*(-p2p10+p5p10)+16*(
     . ffmcfmb*hbcm2*p7p10-p1p3*p7p10+p3p4*p7p10))-8*p3p12)
      b(39)=w8*ccc*(16*p6p7*(-ffmcfmb*hbcm2+hbcm2)+16*(2*ffmcfmb*p3p6
     . *p3p7-p2p3*p6p7+p3p5*p6p7+p3p6*p4p7+p3p7*p4p6))
      b(40)=w8*ccc*(16*(p4p6*p3p7)*(ffmcfmb*hbcm-fmc)+16*(p5p7*p3p6)*
     . (-2*ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+16*(
     . p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+8*p6p7*(-ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-2*fmc*hbcm2-fmc2*hbcm)+16*(
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-2*ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+16
     . *(hbcm*p1p2*p6p7-hbcm*p1p3*p6p7-hbcm*p1p5*p6p7))
      b(41)=w8*ccc*(8*p6p7*(ffmcfmb**2*hbcm2-fmc2)+16*(-ffmcfmb*p3p6*
     . p3p7-2*ffmcfmb*p3p6*p5p7+p1p2*p6p7-p1p3*p6p7-p1p5*p6p7))
      b(43)=ccc*(w8*(16*p6p7*(-p2p11+p5p11)+16*(ffmcfmb*p3p6*p7p11+
     . ffmcfmb*p3p7*p6p11+p4p6*p7p11+p4p7*p6p11))+w5*(16*p4p7*(-p6p10
     . +p6p8)+16*p4p6*(-p7p10+p7p8)+16*p6p7*(p2p10-p2p8-p5p10+p5p8)+
     . 16*p3p7*(-ffmcfmb*p6p10+ffmcfmb*p6p8)+16*p3p6*(-ffmcfmb*p7p10+
     . ffmcfmb*p7p8))+w12*(16*p6p7*(p2p10-p5p10)+16*(-ffmcfmb*p3p6*
     . p7p10-ffmcfmb*p3p7*p6p10-p4p6*p7p10-p4p7*p6p10))+8*(-p6p12-
     . p7p13))
      b(44)=w8*ccc*(16*p4p6*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p6*(ffmcfmb
     . **2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*
     . hbcm-fmc2)+16*(-ffmcfmb*p2p3*p3p6+2*ffmcfmb*p3p4*p3p6+p1p2*
     . p3p6+3*p1p3*p4p6-p1p5*p3p6))
      b(45)=w8*ccc*(p3p6*(16*ffmcfmb+8)+32*p4p6)
      b(46)=w8*ccc*(24*p4p6*(ffmcfmb*hbcm-fmc)+8*p3p6*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmc))
      b(49)=w8*ccc*(8*p3p7*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p5p7)
      b(50)=w8*ccc*(p3p7*(16*ffmcfmb-24)-32*p5p7)
      b(51)=w8*ccc*((p3p7*p1p3)*(-16*ffmcfmb+16)+16*p5p7*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2-fmc*
     . hbcm+fmc2)+16*(p1p3*p5p7-p1p4*p3p7))
      b(53)=ccc*(8*w8*(3*p1p11+p2p11-4*p3p11-4*p4p11)+8*w5*(-3*p1p10+
     . 3*p1p8-p2p10+p2p8+4*p3p10-4*p3p8+4*p4p10-4*p4p8)+8*w12*(-3*
     . p1p10-p2p10+4*p3p10+4*p4p10))
      b(54)=4*ccc*w8*(4*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+2*hbcm2-6*
     . p1p3-2*p2p3+8*p3p4)
      b(55)=4*ccc*w8*(-fmb-fmc)
      b(56)=4*ccc*w8*(-2*ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-fmb*fmc+3*fmc2+2*p1p2-2*p1p3-8*p1p4)
      b(57)=-8*ccc
      b(58)=w8*ccc*(8*p3p6*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p4p6)
      b(59)=w8*ccc*(24*p3p4*(-ffmcfmb*hbcm+fmc)+8*p2p3*(ffmcfmb*hbcm-
     . fmc)+8*p1p3*(2*ffmcfmb*hbcm-2*fmc+hbcm)+4*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+2*fmc*hbcm2-2*fmc2*hbcm-2*hbcm*p1p2+6*hbcm*p1p4))
      b(61)=-4*ccc*hbcm
      b(62)=8*ccc*ffmcfmb*hbcm*p3p6
      b(63)=ccc*(w5*(16*p6p7*(ffmcfmb*hbcm2-hbcm2)+16*(-2*ffmcfmb*
     . p3p6*p3p7+p2p3*p6p7-p3p5*p6p7-p3p6*p4p7-p3p7*p4p6))+w12*(16*
     . p6p7*(ffmcfmb*hbcm2-hbcm2)+16*(-2*ffmcfmb*p3p6*p3p7+p2p3*p6p7-
     . p3p5*p6p7-p3p6*p4p7-p3p7*p4p6)))
      b(64)=ccc*(w5*(8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(ffmcfmb*p3p6
     . *p3p7+2*ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*p6p7+p1p5*p6p7))+w12*
     . (8*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+16*(ffmcfmb*p3p6*p3p7+2*
     . ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*p6p7+p1p5*p6p7)))
      b(65)=w5*ccc*(16*p6p7*(-ffmcfmb*hbcm2+hbcm2)+16*(2*ffmcfmb*p3p6
     . *p3p7-p2p3*p6p7+p3p5*p6p7+p3p6*p4p7+p3p7*p4p6))
      b(66)=ccc*(w8*(16*p6p7*(-hbcm*p2p11+hbcm*p5p11)+16*(ffmcfmb*
     . hbcm*p3p6*p7p11+ffmcfmb*hbcm*p3p7*p6p11+hbcm*p4p6*p7p11+hbcm*
     . p4p7*p6p11))+w5*(16*p4p7*(-hbcm*p6p10+hbcm*p6p8)+16*p4p6*(-
     . hbcm*p7p10+hbcm*p7p8)+16*p6p7*(hbcm*p2p10-hbcm*p2p8-hbcm*p5p10
     . +hbcm*p5p8)+16*p3p7*(-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p8)+16
     . *p3p6*(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8))+w12*(16*p6p7*(
     . hbcm*p2p10-hbcm*p5p10)+16*(-ffmcfmb*hbcm*p3p6*p7p10-ffmcfmb*
     . hbcm*p3p7*p6p10-hbcm*p4p6*p7p10-hbcm*p4p7*p6p10))+8*(-hbcm*
     . p6p12-hbcm*p7p13))
      b(67)=ccc*(w5*(p3p6*(-16*ffmcfmb-8)-32*p4p6)+w12*(p3p6*(-16*
     . ffmcfmb-8)-32*p4p6))
      b(68)=w5*ccc*(p3p6*(16*ffmcfmb+8)+32*p4p6)
      b(69)=ccc*(w5*(p3p7*(-16*ffmcfmb+24)+32*p5p7)+w12*(p3p7*(-16*
     . ffmcfmb+24)+32*p5p7))
      b(70)=ccc*(w5*((p3p7*p1p3)*(16*ffmcfmb-16)+16*p5p7*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*p3p7*(ffmcfmb**2*hbcm2-ffmcfmb*hbcm2+fmc*
     . hbcm-fmc2)+16*(-p1p3*p5p7+p1p4*p3p7))+w12*((p3p7*p1p3)*(16*
     . ffmcfmb-16)+16*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(ffmcfmb
     . **2*hbcm2-ffmcfmb*hbcm2+fmc*hbcm-fmc2)+16*(-p1p3*p5p7+p1p4*
     . p3p7)))
      b(71)=w5*ccc*(p3p7*(16*ffmcfmb-24)-32*p5p7)
      b(72)=ccc*(4*w5*(2*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm+fmb*fmc-3*fmc2-2*p1p2+2*p1p3+8*p1p4)+4*w12*(2*ffmcfmb
     . **2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-3*fmc2-2*
     . p1p2+2*p1p3+8*p1p4))
      b(75)=ccc*(8*w8*(-2*ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p2p11+3*
     . ffmcfmb*hbcm*p3p11+3*ffmcfmb*hbcm*p4p11+2*fmc*p1p11+fmc*p2p11-
     . 3*fmc*p3p11-3*fmc*p4p11)+8*w5*(2*ffmcfmb*hbcm*p1p10-2*ffmcfmb*
     . hbcm*p1p8+ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p2p8-3*ffmcfmb*hbcm*
     . p3p10+3*ffmcfmb*hbcm*p3p8-3*ffmcfmb*hbcm*p4p10+3*ffmcfmb*hbcm*
     . p4p8-2*fmc*p1p10+2*fmc*p1p8-fmc*p2p10+fmc*p2p8+3*fmc*p3p10-3*
     . fmc*p3p8+3*fmc*p4p10-3*fmc*p4p8)+8*w12*(2*ffmcfmb*hbcm*p1p10+
     . ffmcfmb*hbcm*p2p10-3*ffmcfmb*hbcm*p3p10-3*ffmcfmb*hbcm*p4p10-2
     . *fmc*p1p10-fmc*p2p10+3*fmc*p3p10+3*fmc*p4p10))
      ans=ccc*(w8*(16*p1p3*(-p1p11+p3p11+p4p11)+8*(-3*ffmcfmb*hbcm2*
     . p1p11-ffmcfmb*hbcm2*p2p11+2*ffmcfmb*hbcm2*p3p11+2*ffmcfmb*
     . hbcm2*p4p11+fmc*hbcm*p1p11+fmc*hbcm*p2p11-2*fmc*hbcm*p3p11-2*
     . fmc*hbcm*p4p11+2*hbcm2*p1p11-2*p1p11*p2p3+2*p1p11*p3p5))+w5*(
     . 16*p3p5*(-p1p10+p1p8)+16*p2p3*(p1p10-p1p8)+16*p1p3*(p1p10-p1p8
     . -p3p10+p3p8-p4p10+p4p8)+8*(3*ffmcfmb*hbcm2*p1p10-3*ffmcfmb*
     . hbcm2*p1p8+ffmcfmb*hbcm2*p2p10-ffmcfmb*hbcm2*p2p8-2*ffmcfmb*
     . hbcm2*p3p10+2*ffmcfmb*hbcm2*p3p8-2*ffmcfmb*hbcm2*p4p10+2*
     . ffmcfmb*hbcm2*p4p8-fmc*hbcm*p1p10+fmc*hbcm*p1p8-fmc*hbcm*p2p10
     . +fmc*hbcm*p2p8+2*fmc*hbcm*p3p10-2*fmc*hbcm*p3p8+2*fmc*hbcm*
     . p4p10-2*fmc*hbcm*p4p8-2*hbcm2*p1p10+2*hbcm2*p1p8))+w12*(16*
     . p1p3*(p1p10-p3p10-p4p10)+8*(3*ffmcfmb*hbcm2*p1p10+ffmcfmb*
     . hbcm2*p2p10-2*ffmcfmb*hbcm2*p3p10-2*ffmcfmb*hbcm2*p4p10-fmc*
     . hbcm*p1p10-fmc*hbcm*p2p10+2*fmc*hbcm*p3p10+2*fmc*hbcm*p4p10-2*
     . hbcm2*p1p10+2*p1p10*p2p3-2*p1p10*p3p5)))
      b(76)=ans
      b(77)=ccc*(w8*(16*p3p6*(p2p11-p5p11)+16*(-ffmcfmb*hbcm2*p6p11+
     . hbcm2*p6p11-p2p3*p6p11+p3p5*p6p11))+w5*(16*p3p5*(-p6p10+p6p8)+
     . 16*p2p3*(p6p10-p6p8)+16*p3p6*(-p2p10+p2p8+p5p10-p5p8)+16*(
     . ffmcfmb*hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-hbcm2*p6p10+hbcm2*p6p8)
     . )+w12*(16*p3p6*(-p2p10+p5p10)+16*(ffmcfmb*hbcm2*p6p10-hbcm2*
     . p6p10+p2p3*p6p10-p3p5*p6p10))-24*p3p13)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(fmc-2*hbcm)+8*(-2*hbcm*p4p7-hbcm*p5p7))
      b(84)=8*ccc*(-2*p3p7-3*p4p7-p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm+4*hbcm2-6*p1p3-4*p2p3+2*p3p4+6*
     . p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(88)=ccc*(w5*(24*p5p7*(-ffmcfmb*hbcm+fmc)+8*p3p7*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+2*fmc))+w12*(24*p5p7*(-ffmcfmb
     . *hbcm+fmc)+8*p3p7*(ffmcfmb**2*hbcm-ffmcfmb*fmc-2*ffmcfmb*hbcm+
     . 2*fmc)))
      b(89)=-4*ccc*hbcm
      b(90)=w8*ccc*(24*p5p7*(ffmcfmb*hbcm-fmc)+8*p3p7*(-ffmcfmb**2*
     . hbcm+ffmcfmb*fmc+2*ffmcfmb*hbcm-2*fmc))
      b(91)=w5*ccc*(24*p5p7*(ffmcfmb*hbcm-fmc)+8*p3p7*(-ffmcfmb**2*
     . hbcm+ffmcfmb*fmc+2*ffmcfmb*hbcm-2*fmc))
      b(92)=8*ccc*(3*hbcm2-2*p1p3-3*p2p3+3*p3p4+3*p3p5)
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(-ffmcfmb*hbcm+fmc)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp20_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      b(1)=ccc*(w18*((p5p7*p3p6)*(-16*ffmcfmb+16)+(p4p7*p3p6)*(16*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm)+16*(ffmcfmb*p1p3*p6p7
     . +p1p4*p6p7))+w1*((p5p7*p3p6)*(-16*ffmcfmb+16)+(p4p7*p3p6)*(16*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm)+16*(ffmcfmb*p1p3*p6p7
     . +p1p4*p6p7)))
      b(2)=w18*ccc*((p5p7*p3p6)*(-16*ffmcfmb+16)+(p4p7*p3p6)*(16*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm)+16*(ffmcfmb*p1p3*p6p7
     . +p1p4*p6p7))
      ans=ccc*(w18*(16*p5p7*(-p1p8-p1p9)+32*p4p7*(p1p8+p1p9)+16*p1p4*
     . (p7p8+p7p9)+16*p3p7*(ffmcfmb*p1p8+ffmcfmb*p1p9)+16*p1p3*(
     . ffmcfmb*p7p8+ffmcfmb*p7p9)+8*(ffmcfmb**2*hbcm2*p7p8+ffmcfmb**2
     . *hbcm2*p7p9+ffmcfmb*fmb*hbcm*p7p8+ffmcfmb*fmb*hbcm*p7p9-
     . ffmcfmb*fmc*hbcm*p7p8-ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*hbcm2*p7p8
     . -ffmcfmb*hbcm2*p7p9-fmb*fmc*p7p8-fmb*fmc*p7p9+fmc*hbcm*p7p8+
     . fmc*hbcm*p7p9))+8*w1*(ffmcfmb**2*hbcm2*p7p8+ffmcfmb*fmb*hbcm*
     . p7p8-ffmcfmb*fmc*hbcm*p7p8-ffmcfmb*hbcm2*p7p8+2*ffmcfmb*p1p3*
     . p7p8+2*ffmcfmb*p1p8*p3p7-fmb*fmc*p7p8+fmc*hbcm*p7p8+2*p1p4*
     . p7p8+4*p1p8*p4p7-2*p1p8*p5p7)+8*w13*(ffmcfmb**2*hbcm2*p7p10+
     . ffmcfmb*fmb*hbcm*p7p10-ffmcfmb*fmc*hbcm*p7p10-ffmcfmb*hbcm2*
     . p7p10+2*ffmcfmb*p1p10*p3p7+2*ffmcfmb*p1p3*p7p10-fmb*fmc*p7p10+
     . fmc*hbcm*p7p10+4*p1p10*p4p7-2*p1p10*p5p7+2*p1p4*p7p10)+8*(
     . p1p12-2*p4p12+p5p12-3*p7p10+2*p7p8-p7p9))
      b(3)=ans
      b(4)=ccc*(w18*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*
     . hbcm)+16*(-2*ffmcfmb*p1p3*p3p7-2*p1p3*p4p7+p1p3*p5p7-p1p4*p3p7
     . ))+w1*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm)+16*(-2*
     . ffmcfmb*p1p3*p3p7-2*p1p3*p4p7+p1p3*p5p7-p1p4*p3p7)))
      b(5)=w18*ccc*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*
     . hbcm)+16*(-2*ffmcfmb*p1p3*p3p7-2*p1p3*p4p7+p1p3*p5p7-p1p4*p3p7
     . ))
      ans2=w13*(16*p3p4*(-ffmcfmb*hbcm*p7p10-fmb*p7p10+hbcm*p7p10)+16
     . *p3p7*(-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p4p10+fmb*p4p10+fmc*
     . p1p10-hbcm*p4p10)+8*(-ffmcfmb**2*hbcm3*p7p10-ffmcfmb*fmb*hbcm2
     . *p7p10-ffmcfmb*fmc*hbcm2*p7p10+ffmcfmb*hbcm3*p7p10-fmb*fmc*
     . hbcm*p7p10+fmc*hbcm2*p7p10-2*fmc*p1p3*p7p10-2*hbcm*p1p10*p4p7+
     . 2*hbcm*p1p10*p5p7-2*hbcm*p1p4*p7p10))+8*(-fmb*p3p12-fmc*p3p12+
     . hbcm*p1p12+hbcm*p3p12-hbcm*p4p12+hbcm*p7p8)
      ans1=w18*(16*p5p7*(hbcm*p1p8+hbcm*p1p9)+16*p4p7*(-hbcm*p1p8-
     . hbcm*p1p9)+16*p1p4*(-hbcm*p7p8-hbcm*p7p9)+16*p1p3*(-fmc*p7p8-
     . fmc*p7p9)+16*p3p4*(-ffmcfmb*hbcm*p7p8-ffmcfmb*hbcm*p7p9-fmb*
     . p7p8-fmb*p7p9+hbcm*p7p8+hbcm*p7p9)+16*p3p7*(-ffmcfmb*hbcm*p1p8
     . -ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p4p9+fmb*
     . p4p8+fmb*p4p9+fmc*p1p8+fmc*p1p9-hbcm*p4p8-hbcm*p4p9)+8*(-
     . ffmcfmb**2*hbcm3*p7p8-ffmcfmb**2*hbcm3*p7p9-ffmcfmb*fmb*hbcm2*
     . p7p8-ffmcfmb*fmb*hbcm2*p7p9-ffmcfmb*fmc*hbcm2*p7p8-ffmcfmb*fmc
     . *hbcm2*p7p9+ffmcfmb*hbcm3*p7p8+ffmcfmb*hbcm3*p7p9-fmb*fmc*hbcm
     . *p7p8-fmb*fmc*hbcm*p7p9+fmc*hbcm2*p7p8+fmc*hbcm2*p7p9))+w1*(16
     . *p3p4*(-ffmcfmb*hbcm*p7p8-fmb*p7p8+hbcm*p7p8)+16*p3p7*(-
     . ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p4p8+fmb*p4p8+fmc*p1p8-hbcm*
     . p4p8)+8*(-ffmcfmb**2*hbcm3*p7p8-ffmcfmb*fmb*hbcm2*p7p8-ffmcfmb
     . *fmc*hbcm2*p7p8+ffmcfmb*hbcm3*p7p8-fmb*fmc*hbcm*p7p8+fmc*hbcm2
     . *p7p8-2*fmc*p1p3*p7p8-2*hbcm*p1p4*p7p8-2*hbcm*p1p8*p4p7+2*hbcm
     . *p1p8*p5p7))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(16*p3p6*(ffmcfmb*p1p10+ffmcfmb*p4p10-ffmcfmb*p5p10-
     . p4p10+p5p10)+8*(ffmcfmb**2*hbcm2*p6p10+2*ffmcfmb*fmb*hbcm*
     . p6p10-2*ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*hbcm2*p6p10+2*ffmcfmb*
     . p1p3*p6p10-2*fmb*fmc*p6p10-fmb*hbcm*p6p10+fmb2*p6p10+2*fmc*
     . hbcm*p6p10+2*p1p10*p4p6+4*p1p4*p6p10-2*p1p5*p6p10))+8*(p4p13-2
     . *p6p10-2*p6p8+p6p9)
      ans1=w18*(16*p4p6*(p1p8+p1p9)+16*p1p5*(-p6p8-p6p9)+32*p1p4*(
     . p6p8+p6p9)+16*p3p6*(ffmcfmb*p1p8+ffmcfmb*p1p9+ffmcfmb*p4p8+
     . ffmcfmb*p4p9-ffmcfmb*p5p8-ffmcfmb*p5p9-p4p8-p4p9+p5p8+p5p9)+16
     . *p1p3*(ffmcfmb*p6p8+ffmcfmb*p6p9)+8*(ffmcfmb**2*hbcm2*p6p8+
     . ffmcfmb**2*hbcm2*p6p9+2*ffmcfmb*fmb*hbcm*p6p8+2*ffmcfmb*fmb*
     . hbcm*p6p9-2*ffmcfmb*fmc*hbcm*p6p8-2*ffmcfmb*fmc*hbcm*p6p9-
     . ffmcfmb*hbcm2*p6p8-ffmcfmb*hbcm2*p6p9-2*fmb*fmc*p6p8-2*fmb*fmc
     . *p6p9-fmb*hbcm*p6p8-fmb*hbcm*p6p9+fmb2*p6p8+fmb2*p6p9+2*fmc*
     . hbcm*p6p8+2*fmc*hbcm*p6p9))+w1*(16*p3p6*(ffmcfmb*p1p8+ffmcfmb*
     . p4p8-ffmcfmb*p5p8-p4p8+p5p8)+8*(ffmcfmb**2*hbcm2*p6p8+2*
     . ffmcfmb*fmb*hbcm*p6p8-2*ffmcfmb*fmc*hbcm*p6p8-ffmcfmb*hbcm2*
     . p6p8+2*ffmcfmb*p1p3*p6p8-2*fmb*fmc*p6p8-fmb*hbcm*p6p8+fmb2*
     . p6p8+2*fmc*hbcm*p6p8+4*p1p4*p6p8-2*p1p5*p6p8+2*p1p8*p4p6))+
     . ans2
      ans=ccc*ans1
      b(7)=ans
      b(8)=ccc*(w18*((p3p6*p3p5)*(16*ffmcfmb-16)+(p3p6*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p4p6*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb*fmc-fmb2
     . -fmc*hbcm)+16*(-2*ffmcfmb*p1p3*p3p6-p1p3*p4p6-2*p1p4*p3p6+p1p5
     . *p3p6))+w1*((p3p6*p3p5)*(16*ffmcfmb-16)+(p3p6*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p4p6*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb*fmc-fmb2
     . -fmc*hbcm)+16*(-2*ffmcfmb*p1p3*p3p6-p1p3*p4p6-2*p1p4*p3p6+p1p5
     . *p3p6)))
      b(9)=w18*ccc*((p3p6*p3p5)*(16*ffmcfmb-16)+(p3p6*p3p4)*(-16*
     . ffmcfmb+16)+8*p5p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p4p6*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb*fmc-fmb2
     . -fmc*hbcm)+16*(-2*ffmcfmb*p1p3*p3p6-p1p3*p4p6-2*p1p4*p3p6+p1p5
     . *p3p6))
      ans3=w13*(16*p3p5*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+16*
     . p3p4*(-ffmcfmb*hbcm*p6p10-fmb*p6p10+hbcm*p6p10)+16*p5p6*(
     . ffmcfmb*hbcm*p3p10+fmb*p3p10-hbcm*p1p10-hbcm*p3p10)+16*p4p6*(-
     . 2*ffmcfmb*hbcm*p3p10-2*fmb*p3p10+hbcm*p1p10+2*hbcm*p3p10)+16*
     . p1p3*(-ffmcfmb*hbcm*p6p10+fmc*p6p10)+16*p3p6*(-2*ffmcfmb*fmb*
     . p3p10-2*ffmcfmb*fmc*p3p10-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*
     . p4p10+fmb*p1p10+fmb*p3p10+fmb*p4p10-fmb*p5p10+2*fmc*p1p10+2*
     . fmc*p3p10-hbcm*p4p10)+8*(-ffmcfmb*fmb*hbcm2*p6p10-ffmcfmb*fmc*
     . hbcm2*p6p10-fmb*fmc*hbcm*p6p10+fmb*hbcm2*p6p10-fmb2*hbcm*p6p10
     . +fmc*hbcm2*p6p10-2*hbcm*p1p4*p6p10+2*hbcm*p1p5*p6p10))+8*(-3*
     . fmb*p3p13-3*fmc*p3p13+2*hbcm*p3p13-hbcm*p4p13+hbcm*p6p10+hbcm*
     . p6p8-hbcm*p6p9)
      ans2=w1*(16*p5p6*(ffmcfmb*hbcm*p3p8+fmb*p3p8-hbcm*p1p8-hbcm*
     . p3p8)+16*p4p6*(-2*ffmcfmb*hbcm*p3p8-2*fmb*p3p8+hbcm*p1p8+2*
     . hbcm*p3p8)+16*p3p5*(ffmcfmb*hbcm*p6p8+fmb*p6p8-hbcm*p6p8)+16*
     . p3p4*(-ffmcfmb*hbcm*p6p8-fmb*p6p8+hbcm*p6p8)+16*p1p3*(-ffmcfmb
     . *hbcm*p6p8+fmc*p6p8)+16*p3p6*(-2*ffmcfmb*fmb*p3p8-2*ffmcfmb*
     . fmc*p3p8-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p4p8+fmb*p1p8+fmb*p3p8
     . +fmb*p4p8-fmb*p5p8+2*fmc*p1p8+2*fmc*p3p8-hbcm*p4p8)+8*(-
     . ffmcfmb*fmb*hbcm2*p6p8-ffmcfmb*fmc*hbcm2*p6p8-fmb*fmc*hbcm*
     . p6p8+fmb*hbcm2*p6p8-fmb2*hbcm*p6p8+fmc*hbcm2*p6p8-2*hbcm*p1p4*
     . p6p8+2*hbcm*p1p5*p6p8))+ans3
      ans1=w18*(16*p1p5*(hbcm*p6p8+hbcm*p6p9)+16*p1p4*(-hbcm*p6p8-
     . hbcm*p6p9)+16*p5p6*(ffmcfmb*hbcm*p3p8+fmb*p3p8-hbcm*p1p8-hbcm*
     . p1p9-hbcm*p3p8)+16*p4p6*(-2*ffmcfmb*hbcm*p3p8-2*fmb*p3p8+hbcm*
     . p1p8+hbcm*p1p9+2*hbcm*p3p8)+16*p3p5*(ffmcfmb*hbcm*p6p8+ffmcfmb
     . *hbcm*p6p9+fmb*p6p8+fmb*p6p9-hbcm*p6p8-hbcm*p6p9)+16*p3p4*(-
     . ffmcfmb*hbcm*p6p8-ffmcfmb*hbcm*p6p9-fmb*p6p8-fmb*p6p9+hbcm*
     . p6p8+hbcm*p6p9)+16*p1p3*(-ffmcfmb*hbcm*p6p8-ffmcfmb*hbcm*p6p9+
     . fmc*p6p8+fmc*p6p9)+16*p3p6*(-2*ffmcfmb*fmb*p3p8-2*ffmcfmb*fmc*
     . p3p8-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*p4p8+
     . ffmcfmb*hbcm*p4p9+fmb*p1p8+fmb*p1p9+fmb*p3p8+fmb*p4p8+fmb*p4p9
     . -fmb*p5p8-fmb*p5p9+2*fmc*p1p8+2*fmc*p1p9+2*fmc*p3p8-hbcm*p4p8-
     . hbcm*p4p9)+8*(-ffmcfmb*fmb*hbcm2*p6p8-ffmcfmb*fmb*hbcm2*p6p9-
     . ffmcfmb*fmc*hbcm2*p6p8-ffmcfmb*fmc*hbcm2*p6p9-fmb*fmc*hbcm*
     . p6p8-fmb*fmc*hbcm*p6p9+fmb*hbcm2*p6p8+fmb*hbcm2*p6p9-fmb2*hbcm
     . *p6p8-fmb2*hbcm*p6p9+fmc*hbcm2*p6p8+fmc*hbcm2*p6p9))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(w18*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(
     . p4p7*p3p6)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p4)*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(
     . p3p7*p3p6)*(-ffmcfmb**2*hbcm-ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*
     . hbcm+fmc)+16*(ffmcfmb*hbcm*p1p3*p6p7+fmb*p3p6*p5p7+hbcm*p1p4*
     . p6p7))+w1*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p7*
     . p3p6)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p4)*(ffmcfmb*hbcm+
     . fmb-hbcm)+8*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(p3p7*p3p6)
     . *(-ffmcfmb**2*hbcm-ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+
     . 16*(ffmcfmb*hbcm*p1p3*p6p7+fmb*p3p6*p5p7+hbcm*p1p4*p6p7)))
      b(12)=w18*ccc*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p7
     . *p3p6)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p4)*(ffmcfmb*hbcm+
     . fmb-hbcm)+8*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(p3p7*p3p6)
     . *(-ffmcfmb**2*hbcm-ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+
     . 16*(fmb*p3p6*p5p7+fmc*p1p3*p6p7+hbcm*p1p4*p6p7))
      b(13)=ccc*(4*w18*(-2*ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-4*ffmcfmb*p1p3+3*fmb*fmc+fmb*
     . hbcm-fmb2-3*fmc*hbcm-6*p1p4+2*p1p5)+4*w1*(-2*ffmcfmb**2*hbcm2-
     . 3*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-4*
     . ffmcfmb*p1p3+3*fmb*fmc+fmb*hbcm-fmb2-3*fmc*hbcm-6*p1p4+2*p1p5)
     . )
      b(14)=4*ccc*w18*(-2*ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-4*ffmcfmb*p1p3+3*fmb*fmc+fmb*
     . hbcm-fmb2-3*fmc*hbcm-6*p1p4+2*p1p5)
      b(15)=ccc*(w18*(8*p3p5*(-ffmcfmb*hbcm-fmb+hbcm)+16*p3p4*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmb+3*fmc)+4*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-2*fmc*hbcm2+4
     . *hbcm*p1p4-2*hbcm*p1p5))+w1*(8*p3p5*(-ffmcfmb*hbcm-fmb+hbcm)+
     . 16*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmb+3*
     . fmc)+4*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-2*fmc*
     . hbcm2+4*hbcm*p1p4-2*hbcm*p1p5)))
      b(16)=w18*ccc*(8*p3p5*(-ffmcfmb*hbcm-fmb+hbcm)+16*p3p4*(ffmcfmb
     . *hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmb+3*fmc)+4*(ffmcfmb**2
     . *hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+2
     . *fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-2*fmc*hbcm2+4*hbcm*p1p4-2*
     . hbcm*p1p5))
      b(17)=w13*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6+
     . hbcm*p5p6))
      b(18)=4*ccc*w13*(-3*ffmcfmb*hbcm+fmb+4*fmc)
      b(19)=4*ccc*w13*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)
      b(20)=w13*ccc*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7))
      b(21)=ccc*(8*w18*(3*ffmcfmb*hbcm*p3p8-fmb*p3p8-4*fmc*p3p8-2*
     . hbcm*p4p8-2*hbcm*p4p9+hbcm*p5p8+hbcm*p5p9)+8*w1*(3*ffmcfmb*
     . hbcm*p3p8-fmb*p3p8-4*fmc*p3p8-2*hbcm*p4p8+hbcm*p5p8)+8*w13*(3*
     . ffmcfmb*hbcm*p3p10-fmb*p3p10-4*fmc*p3p10-2*hbcm*p4p10+hbcm*
     . p5p10))
      b(22)=ccc*(w13*(8*p3p5*(-ffmcfmb*hbcm-fmb+hbcm)+16*p3p4*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmb+3*fmc)+4*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-2*fmc*hbcm2+4
     . *hbcm*p1p4-2*hbcm*p1p5))-4*hbcm)
      b(23)=ccc*(w18*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6
     . +hbcm*p5p6))+w1*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*
     . p4p6+hbcm*p5p6)))
      b(24)=ccc*(4*w18*(-3*ffmcfmb*hbcm+fmb+4*fmc)+4*w1*(-3*ffmcfmb*
     . hbcm+fmb+4*fmc))
      b(25)=ccc*(4*w18*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)+4*w1*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*p3p5)
     . )
      b(26)=ccc*(w18*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm
     . *p5p7))+w1*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7)))
      b(27)=w13*ccc*(8*p5p6*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p6*(ffmcfmb
     . *hbcm+fmb-hbcm)+8*p3p6*(2*ffmcfmb*fmb+2*ffmcfmb*fmc-fmb-2*fmc)
     . )
      b(28)=ccc*(w13*((p3p6*p3p5)*(-16*ffmcfmb+16)+(p3p6*p3p4)*(16*
     . ffmcfmb-16)+8*p5p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p6*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb
     . *fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmb*fmc+fmb2+fmc*
     . hbcm)+16*(2*ffmcfmb*p1p3*p3p6+p1p3*p4p6+2*p1p4*p3p6-p1p5*p3p6)
     . )-8*p3p6)
      b(29)=ccc*(w13*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(
     . p4p7*p3p6)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p4)*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(
     . p3p7*p3p6)*(-ffmcfmb**2*hbcm-ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*
     . hbcm+fmc)+16*(ffmcfmb*hbcm*p1p3*p6p7+fmb*p3p6*p5p7+hbcm*p1p4*
     . p6p7))-8*hbcm*p6p7)
      ans4=16*p3p6*(ffmcfmb*fmb*hbcm*p7p8+ffmcfmb*fmb*hbcm*p7p9-
     . ffmcfmb*fmc*hbcm*p7p8-ffmcfmb*fmc*hbcm*p7p9-fmb*fmc*p7p8-fmb*
     . fmc*p7p9+fmb2*p7p8+fmb2*p7p9+fmc*hbcm*p7p8+fmc*hbcm*p7p9)+16*
     . p3p7*(-ffmcfmb**2*hbcm2*p6p8-ffmcfmb**2*hbcm2*p6p9-2*ffmcfmb*
     . fmb*hbcm*p6p8-2*ffmcfmb*fmb*hbcm*p6p9+ffmcfmb*fmc*hbcm*p6p8+
     . ffmcfmb*fmc*hbcm*p6p9+ffmcfmb*hbcm2*p6p8+ffmcfmb*hbcm2*p6p9+
     . fmb*fmc*p6p8+fmb*fmc*p6p9+fmb*hbcm*p6p8+fmb*hbcm*p6p9-fmb2*
     . p6p8-fmb2*p6p9-fmc*hbcm*p6p8-fmc*hbcm*p6p9)
      ans3=64*(p4p6*p3p7)*(-p1p8-p1p9)+32*(p5p7*p3p6)*(p1p8+p1p9)+32*
     . (p4p7*p3p6)*(-p1p8-p1p9)+32*(p6p7*p3p4)*(p1p8+p1p9)+32*(p3p7*
     . p1p5)*(p6p8+p6p9)+32*(p3p6*p1p5)*(-p7p8-p7p9)+32*(p3p7*p1p4)*(
     . -p6p8-p6p9)+32*(p3p6*p1p4)*(p7p8+p7p9)+32*(p4p7*p1p3)*(-p6p8-
     . p6p9)+32*(p4p6*p1p3)*(p7p8+p7p9)+32*(p3p6*p3p5)*(-ffmcfmb*p7p8
     . -ffmcfmb*p7p9+p7p8+p7p9)+32*(p3p7*p3p6)*(-2*ffmcfmb*p1p8-2*
     . ffmcfmb*p1p9+ffmcfmb*p5p8+ffmcfmb*p5p9-p5p8-p5p9)+32*(p3p7*
     . p1p3)*(-ffmcfmb*p6p8-ffmcfmb*p6p9)+32*(p3p6*p1p3)*(ffmcfmb*
     . p7p8+ffmcfmb*p7p9)+16*p5p7*(-ffmcfmb*hbcm2*p6p8-ffmcfmb*hbcm2*
     . p6p9-fmb*hbcm*p6p8-fmb*hbcm*p6p9+hbcm2*p6p8+hbcm2*p6p9)+16*
     . p5p6*(ffmcfmb*hbcm2*p7p8+ffmcfmb*hbcm2*p7p9+fmb*hbcm*p7p8+fmb*
     . hbcm*p7p9-hbcm2*p7p8-hbcm2*p7p9)+16*p6p7*(ffmcfmb*hbcm2*p1p8+
     . ffmcfmb*hbcm2*p1p9-ffmcfmb*hbcm2*p4p8-ffmcfmb*hbcm2*p4p9-fmb*
     . hbcm*p4p8-fmb*hbcm*p4p9+fmc*hbcm*p1p8+fmc*hbcm*p1p9+hbcm2*p4p8
     . +hbcm2*p4p9)+ans4
      ans2=w18*ans3
      ans6=w13*(32*(p3p6*p3p4)*(-ffmcfmb*p7p10+p7p10)+32*(p3p7*p3p6)*
     . (-2*ffmcfmb*p1p10+ffmcfmb*p5p10-p5p10)+16*p5p7*(-ffmcfmb*hbcm2
     . *p6p10-fmb*hbcm*p6p10+hbcm2*p6p10)+16*p4p6*(ffmcfmb*hbcm2*
     . p7p10+fmb*hbcm*p7p10-hbcm2*p7p10)+16*p6p7*(ffmcfmb*hbcm2*p1p10
     . -ffmcfmb*hbcm2*p4p10-fmb*hbcm*p4p10+fmc*hbcm*p1p10+hbcm2*p4p10
     . )+16*p3p7*(-ffmcfmb**2*hbcm2*p6p10-2*ffmcfmb*fmb*hbcm*p6p10+
     . ffmcfmb*fmc*hbcm*p6p10+ffmcfmb*hbcm2*p6p10+fmb*fmc*p6p10+fmb*
     . hbcm*p6p10-fmb2*p6p10-fmc*hbcm*p6p10)+16*p3p6*(-ffmcfmb**2*
     . hbcm2*p7p10+ffmcfmb*hbcm2*p7p10+fmb*fmc*p7p10)+32*(-ffmcfmb*
     . p1p3*p3p6*p7p10-ffmcfmb*p1p3*p3p7*p6p10+p1p10*p3p4*p6p7-p1p10*
     . p3p6*p4p7+p1p10*p3p6*p5p7-2*p1p10*p3p7*p4p6-p1p3*p4p7*p6p10-
     . p1p4*p3p6*p7p10-p1p4*p3p7*p6p10+p1p5*p3p7*p6p10))+(16*p3p6*(-
     . p1p12+p3p12-p4p12+p7p10+p7p8)+16*p3p7*(-ffmcfmb*p3p13+p1p13-
     . p4p13+p6p10+p6p8-p6p9)+8*(ffmcfmb*hbcm2*p6p12+fmc*hbcm*p6p12+2
     . *p1p3*p6p12+2*p3p12*p4p6-2*p3p13*p4p7+2*p3p4*p6p12-2*p3p8*p6p7
     . ))
      ans5=w1*(32*(p3p6*p3p5)*(-ffmcfmb*p7p8+p7p8)+32*(p3p7*p3p6)*(-2
     . *ffmcfmb*p1p8+ffmcfmb*p5p8-p5p8)+16*p5p7*(-ffmcfmb*hbcm2*p6p8-
     . fmb*hbcm*p6p8+hbcm2*p6p8)+16*p5p6*(ffmcfmb*hbcm2*p7p8+fmb*hbcm
     . *p7p8-hbcm2*p7p8)+16*p6p7*(ffmcfmb*hbcm2*p1p8-ffmcfmb*hbcm2*
     . p4p8-fmb*hbcm*p4p8+fmc*hbcm*p1p8+hbcm2*p4p8)+16*p3p6*(ffmcfmb*
     . fmb*hbcm*p7p8-ffmcfmb*fmc*hbcm*p7p8-fmb*fmc*p7p8+fmb2*p7p8+fmc
     . *hbcm*p7p8)+16*p3p7*(-ffmcfmb**2*hbcm2*p6p8-2*ffmcfmb*fmb*hbcm
     . *p6p8+ffmcfmb*fmc*hbcm*p6p8+ffmcfmb*hbcm2*p6p8+fmb*fmc*p6p8+
     . fmb*hbcm*p6p8-fmb2*p6p8-fmc*hbcm*p6p8)+32*(ffmcfmb*p1p3*p3p6*
     . p7p8-ffmcfmb*p1p3*p3p7*p6p8+p1p3*p4p6*p7p8-p1p3*p4p7*p6p8+p1p4
     . *p3p6*p7p8-p1p4*p3p7*p6p8-p1p5*p3p6*p7p8+p1p5*p3p7*p6p8+p1p8*
     . p3p4*p6p7-p1p8*p3p6*p4p7+p1p8*p3p6*p5p7-2*p1p8*p3p7*p4p6))+
     . ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w18*(8*p5p6*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p6*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(2*ffmcfmb*fmb+2*ffmcfmb*fmc-fmb
     . -2*fmc))+w1*(8*p5p6*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p6*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*p3p6*(2*ffmcfmb*fmb+2*ffmcfmb*fmc-fmb-2*fmc))
     . )
      b(32)=w18*ccc*(8*p5p6*(ffmcfmb*hbcm+fmb-hbcm)+16*p4p6*(-ffmcfmb
     . *hbcm-fmb+hbcm)+8*p3p6*(-2*ffmcfmb*fmb-2*ffmcfmb*fmc+fmb+2*fmc
     . ))
      ans2=8*(2*fmb*p6p12-3*fmb*p7p13+2*fmc*p6p12-3*fmc*p7p13-hbcm*
     . p6p12+2*hbcm*p7p13)
      ans1=w18*(16*p5p7*(-ffmcfmb*hbcm*p6p8-ffmcfmb*hbcm*p6p9-fmb*
     . p6p8-fmb*p6p9+hbcm*p6p8+hbcm*p6p9)+16*p4p7*(ffmcfmb*hbcm*p6p8+
     . ffmcfmb*hbcm*p6p9+fmb*p6p8+fmb*p6p9-hbcm*p6p8-hbcm*p6p9)+16*
     . p4p6*(ffmcfmb*hbcm*p7p8+ffmcfmb*hbcm*p7p9+fmb*p7p8+fmb*p7p9-
     . hbcm*p7p8-hbcm*p7p9)+16*p6p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*
     . p4p8-ffmcfmb*hbcm*p4p9-fmb*p4p8-fmb*p4p9-fmc*p1p9+hbcm*p4p8+
     . hbcm*p4p9)+16*p3p6*(ffmcfmb*fmb*p7p8+ffmcfmb*fmb*p7p9+ffmcfmb*
     . fmc*p7p8+ffmcfmb*fmc*p7p9-fmc*p7p8-fmc*p7p9))+w1*(16*p5p7*(-
     . ffmcfmb*hbcm*p6p8-fmb*p6p8+hbcm*p6p8)+16*p6p7*(-ffmcfmb*hbcm*
     . p4p8-fmb*p4p8+hbcm*p4p8)+16*p4p7*(ffmcfmb*hbcm*p6p8+fmb*p6p8-
     . hbcm*p6p8)+16*p4p6*(ffmcfmb*hbcm*p7p8+fmb*p7p8-hbcm*p7p8)+16*
     . p3p6*(ffmcfmb*fmb*p7p8+ffmcfmb*fmc*p7p8-fmc*p7p8))+w13*(16*
     . p5p7*(-ffmcfmb*hbcm*p6p10-fmb*p6p10+hbcm*p6p10)+16*p4p7*(
     . ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+16*p4p6*(ffmcfmb*hbcm
     . *p7p10+fmb*p7p10-hbcm*p7p10)+16*p6p7*(-ffmcfmb*hbcm*p4p10-fmb*
     . p4p10+hbcm*p4p10)+16*p3p6*(ffmcfmb*fmb*p7p10+ffmcfmb*fmc*p7p10
     . -fmc*p7p10))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(34)=4*ccc*w18*(3*ffmcfmb*hbcm-fmb-4*fmc)
      b(35)=ccc*(8*w18*(-ffmcfmb*hbcm*p7p8-ffmcfmb*hbcm*p7p9+fmc*p7p8
     . +fmc*p7p9)+8*w1*(-ffmcfmb*hbcm*p7p8+fmc*p7p8)+8*w13*(-ffmcfmb*
     . hbcm*p7p10+fmc*p7p10))
      b(36)=4*ccc*w18*(2*ffmcfmb*hbcm2-fmb*hbcm+3*fmc*hbcm+6*p3p4-2*
     . p3p5)
      b(37)=w18*ccc*(8*p3p7*(2*ffmcfmb*hbcm-fmc)+8*(2*hbcm*p4p7-hbcm*
     . p5p7))
      b(38)=ccc*(w18*(16*p3p7*(p4p8+p4p9)+16*p3p4*(-p7p8-p7p9)+8*(-
     . ffmcfmb*hbcm2*p7p8-ffmcfmb*hbcm2*p7p9-fmc*hbcm*p7p8-fmc*hbcm*
     . p7p9))+8*w1*(-ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8-2*p3p4*p7p8+2*
     . p3p7*p4p8)+8*w13*(-ffmcfmb*hbcm2*p7p10-fmc*hbcm*p7p10-2*p3p4*
     . p7p10+2*p3p7*p4p10)-8*p3p12)
      b(42)=w18*ccc*(8*p6p7*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(2*ffmcfmb*
     . p3p6*p3p7-p3p4*p6p7+2*p3p6*p4p7-p3p6*p5p7+p3p7*p4p6))
      b(43)=ccc*(w18*(16*p5p7*(-p6p8-p6p9)+16*p6p7*(-p4p8-p4p9)+32*
     . p4p7*(p6p8+p6p9)+16*p4p6*(p7p8+p7p9)+16*p3p7*(ffmcfmb*p6p8+
     . ffmcfmb*p6p9)+16*p3p6*(ffmcfmb*p7p8+ffmcfmb*p7p9))+16*w1*(
     . ffmcfmb*p3p6*p7p8+ffmcfmb*p3p7*p6p8+p4p6*p7p8+2*p4p7*p6p8-p4p8
     . *p6p7-p5p7*p6p8)+16*w13*(ffmcfmb*p3p6*p7p10+ffmcfmb*p3p7*p6p10
     . -p4p10*p6p7+p4p6*p7p10+2*p4p7*p6p10-p5p7*p6p10)+8*(p6p12-3*
     . p7p13))
      b(47)=8*ccc*w18*(2*ffmcfmb*p3p6+3*p4p6-p5p6)
      b(52)=8*ccc*w18*(2*ffmcfmb*p3p7+3*p4p7-p5p7)
      b(53)=ccc*(8*w18*(-3*p4p8-3*p4p9+p5p8+p5p9)+8*w1*(-3*p4p8+p5p8)
     . +8*w13*(-3*p4p10+p5p10))
      b(57)=-8*ccc
      b(60)=w18*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-3*fmc)+8*(-2*hbcm*p4p6+
     . hbcm*p5p6))
      b(61)=-4*ccc*hbcm
      b(62)=-8*ccc*fmb*p3p6
      b(63)=w13*ccc*(8*p6p7*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(2*ffmcfmb*
     . p3p6*p3p7-p3p4*p6p7+2*p3p6*p4p7-p3p6*p5p7+p3p7*p4p6))
      b(64)=ccc*(w13*((p5p7*p3p6)*(-16*ffmcfmb+16)+(p4p7*p3p6)*(16*
     . ffmcfmb-16)+8*p6p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm)+16*(ffmcfmb*p1p3*p6p7
     . +p1p4*p6p7))-8*p6p7)
      b(65)=ccc*(w18*(8*p6p7*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(2*ffmcfmb*
     . p3p6*p3p7-p3p4*p6p7+2*p3p6*p4p7-p3p6*p5p7+p3p7*p4p6))+w1*(8*
     . p6p7*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(2*ffmcfmb*p3p6*p3p7-p3p4*
     . p6p7+2*p3p6*p4p7-p3p6*p5p7+p3p7*p4p6)))
      b(66)=ccc*(w18*(16*p5p7*(-hbcm*p6p8-hbcm*p6p9)+16*p6p7*(-hbcm*
     . p4p8-hbcm*p4p9)+16*p4p7*(hbcm*p6p8+hbcm*p6p9)+16*p4p6*(hbcm*
     . p7p8+hbcm*p7p9)+16*p3p6*(fmc*p7p8+fmc*p7p9)+16*p3p7*(-ffmcfmb*
     . hbcm*p6p8-ffmcfmb*hbcm*p6p9+fmb*p6p8+fmb*p6p9+2*fmc*p6p8+2*fmc
     . *p6p9))+w1*(16*p3p7*(-ffmcfmb*hbcm*p6p8+fmb*p6p8+2*fmc*p6p8)+
     . 16*(fmc*p3p6*p7p8+hbcm*p4p6*p7p8+hbcm*p4p7*p6p8-hbcm*p4p8*p6p7
     . -hbcm*p5p7*p6p8))+w13*(16*p3p7*(-ffmcfmb*hbcm*p6p10+fmb*p6p10+
     . 2*fmc*p6p10)+16*(fmc*p3p6*p7p10-hbcm*p4p10*p6p7+hbcm*p4p6*
     . p7p10+hbcm*p4p7*p6p10-hbcm*p5p7*p6p10))+8*(hbcm*p6p12-2*hbcm*
     . p7p13))
      b(67)=8*ccc*w13*(2*ffmcfmb*p3p6+3*p4p6-p5p6)
      b(68)=ccc*(8*w18*(2*ffmcfmb*p3p6+3*p4p6-p5p6)+8*w1*(2*ffmcfmb*
     . p3p6+3*p4p6-p5p6))
      b(69)=8*ccc*w13*(2*ffmcfmb*p3p7+3*p4p7-p5p7)
      b(70)=ccc*(w13*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*
     . hbcm)+16*(-2*ffmcfmb*p1p3*p3p7-2*p1p3*p4p7+p1p3*p5p7-p1p4*p3p7
     . ))+8*p3p7)
      b(71)=ccc*(8*w18*(2*ffmcfmb*p3p7+3*p4p7-p5p7)+8*w1*(2*ffmcfmb*
     . p3p7+3*p4p7-p5p7))
      b(72)=ccc*(4*w13*(-2*ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-4*ffmcfmb*p1p3+3*fmb*fmc+fmb*
     . hbcm-fmb2-3*fmc*hbcm-6*p1p4+2*p1p5)+8)
      b(74)=ccc*(8*w18*p6p7*(ffmcfmb*hbcm-fmc)+8*w1*p6p7*(ffmcfmb*
     . hbcm-fmc))
      b(75)=ccc*(8*w18*(2*ffmcfmb*hbcm*p1p8+2*ffmcfmb*hbcm*p1p9-2*
     . ffmcfmb*hbcm*p4p8-2*ffmcfmb*hbcm*p4p9+ffmcfmb*hbcm*p5p8+
     . ffmcfmb*hbcm*p5p9-fmb*p1p8-fmb*p1p9-2*fmb*p4p8-2*fmb*p4p9+fmb*
     . p5p8+fmb*p5p9-3*fmc*p1p8-3*fmc*p1p9+2*hbcm*p4p8+2*hbcm*p4p9-
     . hbcm*p5p8-hbcm*p5p9)+8*w1*(2*ffmcfmb*hbcm*p1p8-2*ffmcfmb*hbcm*
     . p4p8+ffmcfmb*hbcm*p5p8-fmb*p1p8-2*fmb*p4p8+fmb*p5p8-3*fmc*p1p8
     . +2*hbcm*p4p8-hbcm*p5p8)+8*w13*(2*ffmcfmb*hbcm*p1p10-2*ffmcfmb*
     . hbcm*p4p10+ffmcfmb*hbcm*p5p10-fmb*p1p10-2*fmb*p4p10+fmb*p5p10-
     . 3*fmc*p1p10+2*hbcm*p4p10-hbcm*p5p10))
      ans=ccc*(w18*(16*p3p5*(p1p8+p1p9)+32*p3p4*(-p1p8-p1p9)+16*p1p3*
     . (2*p4p8+2*p4p9-p5p8-p5p9)+8*(-ffmcfmb*hbcm2*p1p8-ffmcfmb*hbcm2
     . *p1p9+ffmcfmb*hbcm2*p4p8+ffmcfmb*hbcm2*p4p9-ffmcfmb*hbcm2*p5p8
     . -ffmcfmb*hbcm2*p5p9+fmb*hbcm*p1p8+fmb*hbcm*p1p9+fmb*hbcm*p4p8+
     . fmb*hbcm*p4p9-fmb*hbcm*p5p8-fmb*hbcm*p5p9-2*fmc*hbcm*p1p8-2*
     . fmc*hbcm*p1p9-hbcm2*p4p8-hbcm2*p4p9+hbcm2*p5p8+hbcm2*p5p9))+w1
     . *(16*p1p3*(2*p4p8-p5p8)+8*(-ffmcfmb*hbcm2*p1p8+ffmcfmb*hbcm2*
     . p4p8-ffmcfmb*hbcm2*p5p8+fmb*hbcm*p1p8+fmb*hbcm*p4p8-fmb*hbcm*
     . p5p8-2*fmc*hbcm*p1p8-hbcm2*p4p8+hbcm2*p5p8-4*p1p8*p3p4+2*p1p8*
     . p3p5))+w13*(16*p1p3*(2*p4p10-p5p10)+8*(-ffmcfmb*hbcm2*p1p10+
     . ffmcfmb*hbcm2*p4p10-ffmcfmb*hbcm2*p5p10+fmb*hbcm*p1p10+fmb*
     . hbcm*p4p10-fmb*hbcm*p5p10-2*fmc*hbcm*p1p10-hbcm2*p4p10+hbcm2*
     . p5p10-4*p1p10*p3p4+2*p1p10*p3p5)))
      b(76)=ans
      b(77)=ccc*(w18*(16*p3p6*(2*p4p8+2*p4p9-p5p8-p5p9)+16*p3p5*(p6p8
     . +p6p9)+32*p3p4*(-p6p8-p6p9)+8*(-ffmcfmb*hbcm2*p6p8-ffmcfmb*
     . hbcm2*p6p9+fmb*hbcm*p6p8+fmb*hbcm*p6p9-2*fmc*hbcm*p6p8-2*fmc*
     . hbcm*p6p9))+w1*(16*p3p6*(2*p4p8-p5p8)+8*(-ffmcfmb*hbcm2*p6p8+
     . fmb*hbcm*p6p8-2*fmc*hbcm*p6p8-4*p3p4*p6p8+2*p3p5*p6p8))+w13*(
     . 16*p3p6*(2*p4p10-p5p10)+8*(-ffmcfmb*hbcm2*p6p10+fmb*hbcm*p6p10
     . -2*fmc*hbcm*p6p10-4*p3p4*p6p10+2*p3p5*p6p10)))
      b(78)=ccc*(8*w18*(2*ffmcfmb*hbcm*p6p8+2*ffmcfmb*hbcm*p6p9-fmb*
     . p6p8-fmb*p6p9-3*fmc*p6p8-3*fmc*p6p9)+8*w1*(2*ffmcfmb*hbcm*p6p8
     . -fmb*p6p8-3*fmc*p6p8)+8*w13*(2*ffmcfmb*hbcm*p6p10-fmb*p6p10-3*
     . fmc*p6p10))
      b(79)=8*ccc*w18*p6p7*(ffmcfmb*hbcm-fmc)
      b(80)=w18*ccc*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p7*(ffmcfmb
     . *hbcm+fmb-hbcm)+8*p3p7*(ffmcfmb**2*hbcm+ffmcfmb*fmb-ffmcfmb*
     . hbcm))
      b(81)=8*ccc*w13*p6p7*(ffmcfmb*hbcm-fmc)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc-hbcm)+8*hbcm*p5p7)
      b(84)=8*ccc*(-p4p7+p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-2*p1p3-2*p3p4+2*p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(2*fmb+2*fmc-hbcm)
      b(88)=w13*ccc*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p7*(ffmcfmb
     . *hbcm+fmb-hbcm)+8*p3p7*(ffmcfmb**2*hbcm+ffmcfmb*fmb-ffmcfmb*
     . hbcm))
      b(89)=-4*ccc*hbcm
      b(91)=ccc*(w18*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p7*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p3p7*(ffmcfmb**2*hbcm+ffmcfmb*fmb-
     . ffmcfmb*hbcm))+w1*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+16*p4p7*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p3p7*(ffmcfmb**2*hbcm+ffmcfmb*fmb-
     . ffmcfmb*hbcm)))
      b(92)=8*ccc*p1p3
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(ffmcfmb*hbcm+fmb-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp2_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(31) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,31 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p4*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      b(8)=8*ccc*w5*p4p6*(fmb*hbcm-fmc*hbcm)
      b(10)=16*ccc*w5*p4p6*(-hbcm*p1p10+hbcm*p1p8+hbcm*p4p10-hbcm*
     . p4p8)
      b(15)=-8*ccc*hbcm*p1p4*w5
      b(18)=-4*ccc*hbcm*w5
      b(19)=4*ccc*w5*(-fmb*hbcm+fmc*hbcm)
      b(21)=8*ccc*w5*(hbcm*p1p10-hbcm*p1p8-hbcm*p4p10+hbcm*p4p8)
      b(22)=8*ccc*hbcm*p1p4*w5
      b(24)=4*ccc*hbcm*w5
      b(25)=4*ccc*w5*(fmb*hbcm-fmc*hbcm)
      b(27)=8*ccc*hbcm*p4p6*w5
      b(28)=8*ccc*w5*p4p6*(fmb*hbcm-fmc*hbcm)
      b(30)=16*ccc*w5*p4p6*(-fmb*hbcm*p7p8+fmc*hbcm*p7p8)
      b(31)=-8*ccc*hbcm*p4p6*w5
      do 200 n=1,31 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp1s2_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=-8*ccc*p6p7*w1
      b(2)=8*ccc*p6p7*w2
      b(3)=8*ccc*(-p7p8*w1+p7p9*w2)
      b(4)=8*ccc*p3p7*w1
      b(5)=-8*ccc*p3p7*w2
      b(6)=8*ccc*(hbcm*p7p8*w1-hbcm*p7p9*w2)
      b(7)=8*ccc*(-p6p8*w1+p6p9*w2)
      b(8)=8*ccc*p3p6*w1
      b(9)=-8*ccc*p3p6*w2
      b(10)=8*ccc*(hbcm*p6p8*w1-hbcm*p6p9*w2)
      b(11)=-4*ccc*hbcm*p6p7*w1
      b(12)=4*ccc*hbcm*p6p7*w2
      b(13)=6*ccc*w1
      b(14)=-6*ccc*w2
      b(15)=-6*ccc*hbcm*w1
      b(16)=6*ccc*hbcm*w2
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp28_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(1)=w11*ccc*((p5p6*p3p7)*(-8*ffmcfmb+16)+(p4p6*p3p7)*(-8*
     . ffmcfmb-8)+(p5p7*p3p6)*(-32*ffmcfmb+24)+(p3p7*p3p6)*(-24*
     . ffmcfmb+16)+(p6p7*p2p3)*(-8*ffmcfmb+8)+4*p6p7*(ffmcfmb**2*
     . hbcm2-fmb2-hbcm2)+8*(p1p2*p6p7-p1p3*p6p7-p1p5*p6p7+p2p5*p6p7-
     . p3p5*p6p7-p4p6*p4p7-2*p4p6*p5p7+3*p5p6*p5p7))
      b(2)=w2*ccc*((p5p6*p3p7)*(8*ffmcfmb-16)+(p4p6*p3p7)*(8*ffmcfmb+
     . 8)+(p5p7*p3p6)*(32*ffmcfmb-24)+(p3p7*p3p6)*(24*ffmcfmb-16)+(
     . p6p7*p2p3)*(8*ffmcfmb-8)+4*p6p7*(-ffmcfmb**2*hbcm2+fmb2+hbcm2)
     . +8*(-p1p2*p6p7+p1p3*p6p7+p1p5*p6p7-p2p5*p6p7+p3p5*p6p7+p4p6*
     . p4p7+2*p4p6*p5p7-3*p5p6*p5p7))
      ans2=w11*(8*p5p7*(p5p11-p5p8)+8*p3p5*(p7p11-p7p8)+8*p3p4*(-
     . p7p11+p7p8)+8*p2p5*(-p7p11+p7p8)+8*p2p4*(p7p11-p7p8)+16*p1p4*(
     . -p7p11+p7p8)+8*p1p3*(-p7p11+p7p8)+8*p1p2*(p7p11-p7p8)+8*p4p7*(
     . -p1p11+p1p8+p2p11-p2p8-p4p11+p4p8)+8*p2p3*(ffmcfmb*p7p11-
     . ffmcfmb*p7p8)+8*p3p7*(-ffmcfmb*p1p11+ffmcfmb*p1p8+ffmcfmb*
     . p2p11-ffmcfmb*p2p8-ffmcfmb*p4p11+ffmcfmb*p4p8-ffmcfmb*p5p11+
     . ffmcfmb*p5p8+p5p11-p5p8)+4*(-2*ffmcfmb*hbcm2*p7p11+2*ffmcfmb*
     . hbcm2*p7p8+fmb2*p7p11-fmb2*p7p8-fmc2*p7p11+fmc2*p7p8+hbcm2*
     . p7p11-hbcm2*p7p8))+4*(-p1p12+p2p12-p4p12-p5p12+p7p11)
      ans1=w2*(8*p4p7*(-p1p9+p2p9-p4p9)+8*p3p7*(-ffmcfmb*p1p9+ffmcfmb
     . *p2p9-ffmcfmb*p4p9-ffmcfmb*p5p9+p5p9)+4*(-2*ffmcfmb*hbcm2*p7p9
     . +2*ffmcfmb*p2p3*p7p9+fmb2*p7p9-fmc2*p7p9+hbcm2*p7p9+2*p1p2*
     . p7p9-2*p1p3*p7p9-4*p1p4*p7p9+2*p2p4*p7p9-2*p2p5*p7p9-2*p3p4*
     . p7p9+2*p3p5*p7p9+2*p5p7*p5p9))+w7*(8*p4p7*(-p1p11+p2p11-p4p11)
     . +8*p3p7*(-ffmcfmb*p1p11+ffmcfmb*p2p11-ffmcfmb*p4p11-ffmcfmb*
     . p5p11+p5p11)+4*(-2*ffmcfmb*hbcm2*p7p11+2*ffmcfmb*p2p3*p7p11+
     . fmb2*p7p11-fmc2*p7p11+hbcm2*p7p11+2*p1p2*p7p11-2*p1p3*p7p11-4*
     . p1p4*p7p11+2*p2p4*p7p11-2*p2p5*p7p11-2*p3p4*p7p11+2*p3p5*p7p11
     . +2*p5p11*p5p7))+ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=w11*ccc*((p3p7*p3p5)*(-8*ffmcfmb+8)+(p3p7*p2p3)*(16*
     . ffmcfmb-8)+4*p4p7*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+4*p3p7*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2+fmb2-fmc2+
     . hbcm2)+8*(ffmcfmb*hbcm2*p5p7-ffmcfmb*p1p3*p3p7-ffmcfmb*p3p4*
     . p3p7+p1p2*p3p7+p1p3*p5p7-2*p1p4*p3p7-p2p3*p5p7+p2p4*p3p7-p2p5*
     . p3p7+p3p4*p5p7-p3p5*p4p7))
      b(5)=w2*ccc*((p3p7*p3p5)*(8*ffmcfmb-8)+(p3p7*p2p3)*(-16*ffmcfmb
     . +8)+4*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p3p7*(ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb2+fmc2-hbcm2)+8*(-
     . ffmcfmb*hbcm2*p5p7+ffmcfmb*p1p3*p3p7+ffmcfmb*p3p4*p3p7-p1p2*
     . p3p7-p1p3*p5p7+2*p1p4*p3p7+p2p3*p5p7-p2p4*p3p7+p2p5*p3p7-p3p4*
     . p5p7+p3p5*p4p7))
      ans2=w11*(8*p4p7*(hbcm*p5p11-hbcm*p5p8)+8*p3p4*(hbcm*p7p11-hbcm
     . *p7p8)+8*p2p5*(hbcm*p7p11-hbcm*p7p8)+8*p2p4*(-hbcm*p7p11+hbcm*
     . p7p8)+16*p1p4*(hbcm*p7p11-hbcm*p7p8)+8*p1p2*(-hbcm*p7p11+hbcm*
     . p7p8)+8*p5p7*(-hbcm*p1p11+hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*
     . p4p11+hbcm*p4p8)+8*p3p7*(ffmcfmb*hbcm*p5p11-ffmcfmb*hbcm*p5p8-
     . fmb*p1p11+fmb*p1p8+fmb*p2p11-fmb*p2p8-fmb*p4p11+fmb*p4p8)+8*
     . p3p5*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8-fmb*p7p11+fmb*p7p8
     . )+8*p2p3*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8)+8*p1p3*(2*
     . ffmcfmb*hbcm*p7p11-2*ffmcfmb*hbcm*p7p8+2*fmb*p7p11-2*fmb*p7p8-
     . hbcm*p7p11+hbcm*p7p8)+4*(2*ffmcfmb**2*hbcm3*p7p11-2*ffmcfmb**2
     . *hbcm3*p7p8+2*ffmcfmb*fmb*hbcm2*p7p11-2*ffmcfmb*fmb*hbcm2*p7p8
     . -2*ffmcfmb*hbcm3*p7p11+2*ffmcfmb*hbcm3*p7p8-2*fmb*hbcm2*p7p11+
     . 2*fmb*hbcm2*p7p8-fmb2*hbcm*p7p11+fmb2*hbcm*p7p8+fmc2*hbcm*
     . p7p11-fmc2*hbcm*p7p8+hbcm3*p7p11-hbcm3*p7p8))+4*(-fmb*p3p12-
     . fmc*p3p12+hbcm*p3p12+2*hbcm*p5p12+hbcm*p7p10-2*hbcm*p7p11+hbcm
     . *p7p8-hbcm*p7p9)
      ans1=w2*(8*p5p7*(-hbcm*p1p9+hbcm*p2p9-hbcm*p4p9)+8*p3p7*(
     . ffmcfmb*hbcm*p5p9-fmb*p1p9+fmb*p2p9-fmb*p4p9)+8*p3p5*(-ffmcfmb
     . *hbcm*p7p9-fmb*p7p9)+8*p1p3*(2*ffmcfmb*hbcm*p7p9+2*fmb*p7p9-
     . hbcm*p7p9)+4*(2*ffmcfmb**2*hbcm3*p7p9+2*ffmcfmb*fmb*hbcm2*p7p9
     . -2*ffmcfmb*hbcm*p2p3*p7p9-2*ffmcfmb*hbcm3*p7p9-2*fmb*hbcm2*
     . p7p9-fmb2*hbcm*p7p9+fmc2*hbcm*p7p9-2*hbcm*p1p2*p7p9+4*hbcm*
     . p1p4*p7p9-2*hbcm*p2p4*p7p9+2*hbcm*p2p5*p7p9+2*hbcm*p3p4*p7p9+2
     . *hbcm*p4p7*p5p9+hbcm3*p7p9))+w7*(8*p5p7*(-hbcm*p1p11+hbcm*
     . p2p11-hbcm*p4p11)+8*p3p7*(ffmcfmb*hbcm*p5p11-fmb*p1p11+fmb*
     . p2p11-fmb*p4p11)+8*p3p5*(-ffmcfmb*hbcm*p7p11-fmb*p7p11)+8*p1p3
     . *(2*ffmcfmb*hbcm*p7p11+2*fmb*p7p11-hbcm*p7p11)+4*(2*ffmcfmb**2
     . *hbcm3*p7p11+2*ffmcfmb*fmb*hbcm2*p7p11-2*ffmcfmb*hbcm*p2p3*
     . p7p11-2*ffmcfmb*hbcm3*p7p11-2*fmb*hbcm2*p7p11-fmb2*hbcm*p7p11+
     . fmc2*hbcm*p7p11-2*hbcm*p1p2*p7p11+4*hbcm*p1p4*p7p11-2*hbcm*
     . p2p4*p7p11+2*hbcm*p2p5*p7p11+2*hbcm*p3p4*p7p11+2*hbcm*p4p7*
     . p5p11+hbcm3*p7p11))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w11*(8*p3p5*(p6p11-p6p8)+8*p2p5*(-p6p11+p6p8)+8*p5p6*(-2*
     . p2p11+2*p2p8+3*p3p11-3*p3p8+3*p4p11-3*p4p8)+8*p1p5*(p6p11-p6p8
     . )+8*p1p3*(p6p11-p6p8)+8*p1p2*(-p6p11+p6p8)+8*p4p6*(-p1p11+p1p8
     . +p2p11-p2p8-3*p3p11+3*p3p8-2*p4p11+2*p4p8-p5p11+p5p8)+8*p2p3*(
     . ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+8*p3p6*(2*ffmcfmb*p2p11
     . -2*ffmcfmb*p2p8-4*ffmcfmb*p3p11+4*ffmcfmb*p3p8-4*ffmcfmb*p4p11
     . +4*ffmcfmb*p4p8-p1p11+p1p8-p2p11+p2p8+2*p3p11-2*p3p8+2*p4p11-2
     . *p4p8-p5p11+p5p8)+4*(-ffmcfmb**2*hbcm2*p6p11+ffmcfmb**2*hbcm2*
     . p6p8+fmb2*p6p11-fmb2*p6p8+hbcm2*p6p11-hbcm2*p6p8))+4*(-p1p13+2
     . *p2p13-4*p3p13-4*p4p13-p5p13+p6p10+p6p11+p6p9)
      ans1=w2*(8*p5p6*(-2*p2p9+3*p3p9+3*p4p9)+8*p4p6*(-p1p9+p2p9-3*
     . p3p9-2*p4p9-p5p9)+8*p3p6*(2*ffmcfmb*p2p9-4*ffmcfmb*p3p9-4*
     . ffmcfmb*p4p9-p1p9-p2p9+2*p3p9+2*p4p9-p5p9)+8*p2p3*(ffmcfmb*
     . p6p9-p6p9)+4*(-ffmcfmb**2*hbcm2*p6p9+fmb2*p6p9+hbcm2*p6p9-2*
     . p1p2*p6p9+2*p1p3*p6p9+2*p1p5*p6p9-2*p2p5*p6p9+2*p3p5*p6p9))+w7
     . *(8*p5p6*(-2*p2p11+3*p3p11+3*p4p11)+8*p4p6*(-p1p11+p2p11-3*
     . p3p11-2*p4p11-p5p11)+8*p2p3*(ffmcfmb*p6p11-p6p11)+8*p3p6*(2*
     . ffmcfmb*p2p11-4*ffmcfmb*p3p11-4*ffmcfmb*p4p11-p1p11-p2p11+2*
     . p3p11+2*p4p11-p5p11)+4*(-ffmcfmb**2*hbcm2*p6p11+fmb2*p6p11+
     . hbcm2*p6p11-2*p1p2*p6p11+2*p1p3*p6p11+2*p1p5*p6p11-2*p2p5*
     . p6p11+2*p3p5*p6p11))+ans2
      ans=ccc*ans1
      b(7)=ans
      b(8)=w11*ccc*((p3p6*p3p4)*(-32*ffmcfmb+16)+(p3p6*p2p3)*(24*
     . ffmcfmb-16)+4*p5p6*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm+3*hbcm2)
     . +4*p4p6*(2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-4*hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-3*
     . ffmcfmb*hbcm2-2*fmb*hbcm+fmb2+fmc*hbcm+2*hbcm2)+8*(-p1p2*p3p6-
     . p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*p2p3*p5p6-p2p5*p3p6-2*p3p4*
     . p4p6+3*p3p4*p5p6-p3p5*p4p6))
      b(9)=w2*ccc*((p3p6*p3p4)*(32*ffmcfmb-16)+(p3p6*p2p3)*(-24*
     . ffmcfmb+16)+4*p5p6*(-ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm-3*hbcm2
     . )+4*p4p6*(-2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+4*hbcm2)+4*p3p6*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+3*
     . ffmcfmb*hbcm2+2*fmb*hbcm-fmb2-fmc*hbcm-2*hbcm2)+8*(p1p2*p3p6+
     . p1p3*p4p6-p1p5*p3p6-p2p3*p4p6+2*p2p3*p5p6+p2p5*p3p6+2*p3p4*
     . p4p6-3*p3p4*p5p6+p3p5*p4p6))
      ans5=8*p3p6*(-2*ffmcfmb**2*hbcm*p3p8-4*ffmcfmb*fmb*p3p8-2*
     . ffmcfmb*fmc*p3p8-3*ffmcfmb*hbcm*p1p11+3*ffmcfmb*hbcm*p1p8-
     . ffmcfmb*hbcm*p2p11+ffmcfmb*hbcm*p2p8+3*ffmcfmb*hbcm*p3p11+
     . ffmcfmb*hbcm*p3p8+3*ffmcfmb*hbcm*p4p11-3*ffmcfmb*hbcm*p4p8+
     . ffmcfmb*hbcm*p5p11-ffmcfmb*hbcm*p5p8-fmb*p1p11+fmb*p1p8-fmb*
     . p2p11+fmb*p2p8+fmb*p3p11+2*fmb*p3p8+fmb*p4p11-fmb*p4p8+fmb*
     . p5p11-fmb*p5p8+fmc*p3p8+2*hbcm*p1p11-2*hbcm*p1p8+hbcm*p2p11-
     . hbcm*p2p8-2*hbcm*p3p11-2*hbcm*p4p11+2*hbcm*p4p8)+4*(ffmcfmb**2
     . *hbcm3*p6p11-ffmcfmb**2*hbcm3*p6p8-fmb2*hbcm*p6p11+fmb2*hbcm*
     . p6p8-hbcm3*p6p11+hbcm3*p6p8)
      ans4=8*p3p5*(-hbcm*p6p11+hbcm*p6p8)+8*p2p5*(hbcm*p6p11-hbcm*
     . p6p8)+8*p1p5*(-hbcm*p6p11+hbcm*p6p8)+8*p1p2*(hbcm*p6p11-hbcm*
     . p6p8)+8*p5p6*(2*ffmcfmb*hbcm*p3p8+3*fmb*p3p8+fmc*p3p8+2*hbcm*
     . p1p11-2*hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-2*hbcm*p3p11-2*hbcm*
     . p4p11+2*hbcm*p4p8)+8*p4p6*(-ffmcfmb*hbcm*p3p8-2*fmb*p3p8-fmc*
     . p3p8-3*hbcm*p1p11+3*hbcm*p1p8+2*hbcm*p3p11-hbcm*p3p8+hbcm*
     . p4p11-hbcm*p4p8+hbcm*p5p11-hbcm*p5p8)+8*p2p3*(-ffmcfmb*hbcm*
     . p6p11+ffmcfmb*hbcm*p6p8+hbcm*p6p11-hbcm*p6p8)+8*p1p3*(-2*
     . ffmcfmb*hbcm*p6p11+2*ffmcfmb*hbcm*p6p8-2*fmb*p6p11+2*fmb*p6p8+
     . hbcm*p6p11-hbcm*p6p8)+ans5
      ans3=w11*ans4
      ans6=4*(-fmb*p3p13-fmc*p3p13+hbcm*p1p13+3*hbcm*p3p13+2*hbcm*
     . p4p13+hbcm*p5p13-hbcm*p6p10-hbcm*p6p11-hbcm*p6p9)
      ans2=w7*(8*p5p6*(2*hbcm*p1p11+hbcm*p2p11-2*hbcm*p3p11-2*hbcm*
     . p4p11)+8*p4p6*(-3*hbcm*p1p11+2*hbcm*p3p11+hbcm*p4p11+hbcm*
     . p5p11)+8*p2p3*(-ffmcfmb*hbcm*p6p11+hbcm*p6p11)+8*p1p3*(-2*
     . ffmcfmb*hbcm*p6p11-2*fmb*p6p11+hbcm*p6p11)+8*p3p6*(-3*ffmcfmb*
     . hbcm*p1p11-ffmcfmb*hbcm*p2p11+3*ffmcfmb*hbcm*p3p11+3*ffmcfmb*
     . hbcm*p4p11+ffmcfmb*hbcm*p5p11-fmb*p1p11-fmb*p2p11+fmb*p3p11+
     . fmb*p4p11+fmb*p5p11+2*hbcm*p1p11+hbcm*p2p11-2*hbcm*p3p11-2*
     . hbcm*p4p11)+4*(ffmcfmb**2*hbcm3*p6p11-fmb2*hbcm*p6p11+2*hbcm*
     . p1p2*p6p11-2*hbcm*p1p5*p6p11+2*hbcm*p2p5*p6p11-2*hbcm*p3p5*
     . p6p11-hbcm3*p6p11))+ans3+ans6
      ans1=w2*(8*p5p6*(2*hbcm*p1p9+hbcm*p2p9-2*hbcm*p3p9-2*hbcm*p4p9)
     . +8*p4p6*(-3*hbcm*p1p9+2*hbcm*p3p9+hbcm*p4p9+hbcm*p5p9)+8*p2p3*
     . (-ffmcfmb*hbcm*p6p9+hbcm*p6p9)+8*p3p6*(-3*ffmcfmb*hbcm*p1p9-
     . ffmcfmb*hbcm*p2p9+3*ffmcfmb*hbcm*p3p9+3*ffmcfmb*hbcm*p4p9+
     . ffmcfmb*hbcm*p5p9-fmb*p1p9-fmb*p2p9+fmb*p3p9+fmb*p4p9+fmb*p5p9
     . +2*hbcm*p1p9+hbcm*p2p9-2*hbcm*p3p9-2*hbcm*p4p9)+8*p1p3*(-2*
     . ffmcfmb*hbcm*p6p9-2*fmb*p6p9+hbcm*p6p9)+4*(ffmcfmb**2*hbcm3*
     . p6p9-fmb2*hbcm*p6p9+2*hbcm*p1p2*p6p9-2*hbcm*p1p5*p6p9+2*hbcm*
     . p2p5*p6p9-2*hbcm*p3p5*p6p9-hbcm3*p6p9))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=w11*ccc*(8*(p5p6*p3p7)*(ffmcfmb*hbcm+2*fmb+fmc)+8*(p4p6*
     . p3p7)*(-2*ffmcfmb*hbcm-fmb-fmc)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm+
     . hbcm)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm+hbcm)+8*(p6p7*p1p3)*(-2*
     . ffmcfmb*hbcm-2*fmb+hbcm)+4*p6p7*(ffmcfmb**2*hbcm3-fmb2*hbcm-
     . hbcm3)+8*(p3p7*p3p6)*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*fmb-2*
     . ffmcfmb*fmc+ffmcfmb*hbcm+2*fmb+fmc)+8*(hbcm*p1p2*p6p7-hbcm*
     . p1p5*p6p7+hbcm*p2p5*p6p7-hbcm*p3p5*p6p7-hbcm*p4p6*p4p7-hbcm*
     . p4p6*p5p7+2*hbcm*p5p6*p5p7))
      b(12)=w2*ccc*(8*(p5p6*p3p7)*(-ffmcfmb*hbcm-2*fmb-fmc)+8*(p4p6*
     . p3p7)*(2*ffmcfmb*hbcm+fmb+fmc)+16*(p5p7*p3p6)*(ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-hbcm)+4*p6p7*(-ffmcfmb**2*
     . hbcm3+fmb2*hbcm+hbcm3)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm+2*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm-2*fmb-fmc)+8*(-hbcm*
     . p1p2*p6p7+hbcm*p1p3*p6p7+hbcm*p1p5*p6p7-hbcm*p2p5*p6p7+hbcm*
     . p3p5*p6p7+hbcm*p4p6*p4p7+hbcm*p4p6*p5p7-2*hbcm*p5p6*p5p7))
      b(13)=2*ccc*w11*(-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3+fmb2-fmc2+
     . hbcm2+2*p1p2-2*p1p3-4*p1p4+2*p2p4-2*p2p5-2*p3p4+2*p3p5)
      b(14)=2*ccc*w2*(2*ffmcfmb*hbcm2-2*ffmcfmb*p2p3-fmb2+fmc2-hbcm2-
     . 2*p1p2+2*p1p3+4*p1p4-2*p2p4+2*p2p5+2*p3p4-2*p3p5)
      b(15)=w11*ccc*(4*p3p5*(-ffmcfmb*hbcm-fmb)+4*p1p3*(2*ffmcfmb*
     . hbcm+2*fmb-hbcm)+2*(2*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*hbcm*p2p3-2*ffmcfmb*hbcm3-2*fmb*hbcm2-fmb2*hbcm+fmc2*
     . hbcm-2*hbcm*p1p2+4*hbcm*p1p4-2*hbcm*p2p4+2*hbcm*p2p5+2*hbcm*
     . p3p4+hbcm3))
      b(16)=w2*ccc*(4*p3p5*(ffmcfmb*hbcm+fmb)+4*p1p3*(-2*ffmcfmb*hbcm
     . -2*fmb+hbcm)+2*(-2*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*
     . ffmcfmb*hbcm*p2p3+2*ffmcfmb*hbcm3+2*fmb*hbcm2+fmb2*hbcm-fmc2*
     . hbcm+2*hbcm*p1p2-4*hbcm*p1p4+2*hbcm*p2p4-2*hbcm*p2p5-2*hbcm*
     . p3p4-hbcm3))
      b(21)=ccc*(4*w2*(-hbcm*p1p9-3*hbcm*p2p9+2*hbcm*p3p9+3*hbcm*p4p9
     . -hbcm*p5p9)+4*w7*(-hbcm*p1p11-3*hbcm*p2p11+2*hbcm*p3p11+3*hbcm
     . *p4p11-hbcm*p5p11)+4*w11*(-3*ffmcfmb*hbcm*p3p8-3*fmb*p3p8-hbcm
     . *p1p11+hbcm*p1p8-3*hbcm*p2p11+3*hbcm*p2p8+2*hbcm*p3p11+hbcm*
     . p3p8+3*hbcm*p4p11-3*hbcm*p4p8-hbcm*p5p11+hbcm*p5p8))
      b(23)=w11*ccc*(8*p3p6*(-2*ffmcfmb*hbcm-fmb+hbcm)+8*(-2*hbcm*
     . p4p6+hbcm*p5p6))
      b(24)=6*ccc*w11*(ffmcfmb*hbcm+fmb-hbcm)
      b(25)=2*ccc*w11*(5*ffmcfmb*hbcm2-3*fmb*hbcm+hbcm2-2*p1p3-6*p2p3
     . +6*p3p4-2*p3p5)
      b(26)=w11*ccc*(4*p3p7*(ffmcfmb*hbcm-3*fmb)+4*(hbcm*p4p7-3*hbcm*
     . p5p7))
      ans3=(16*(p5p6*p3p7)*(3*p2p9-3*p3p9-3*p4p9-2*p5p9)+16*(p4p6*
     . p3p7)*(2*p1p9-3*p2p9+3*p3p9+3*p4p9+2*p5p9)+16*(p6p7*p3p5)*(
     . p1p9+p5p9)+16*(p6p7*p3p4)*(p1p9+p5p9)+16*(p6p7*p2p3)*(-p1p9-
     . p5p9)+16*(p6p7*p1p3)*(-p1p9+p5p9)+16*p6p7*(hbcm2*p1p9+hbcm2*
     . p5p9)+16*(p3p7*p3p5)*(ffmcfmb*p6p9-p6p9)+16*(p3p7*p3p4)*(
     . ffmcfmb*p6p9-p6p9)+16*(p3p6*p3p4)*(-2*ffmcfmb*p7p9+p7p9)+16*(
     . p3p7*p3p6)*(-4*ffmcfmb*p2p9+4*ffmcfmb*p3p9+4*ffmcfmb*p4p9+2*
     . ffmcfmb*p5p9+2*p1p9+2*p2p9-2*p3p9-2*p4p9-p5p9)+16*(p3p7*p2p3)*
     . (-ffmcfmb*p6p9+p6p9)+16*(p3p6*p2p3)*(2*ffmcfmb*p7p9-p7p9)+16*(
     . p3p7*p1p3)*(ffmcfmb*p6p9-p6p9)+8*p5p6*(ffmcfmb*hbcm2*p7p9-fmb*
     . hbcm*p7p9+hbcm2*p7p9)+8*p4p6*(-ffmcfmb*hbcm2*p7p9+fmb*hbcm*
     . p7p9-hbcm2*p7p9)+16*p3p7*(ffmcfmb*hbcm2*p6p9-hbcm2*p6p9)+8*
     . p3p6*(-2*ffmcfmb**2*hbcm2*p7p9+2*ffmcfmb*fmb*hbcm*p7p9-ffmcfmb
     . *hbcm2*p7p9-fmb*hbcm*p7p9+hbcm2*p7p9)+16*(-hbcm2*p5p7*p6p9-
     . p1p3*p5p7*p6p9+p1p9*p3p6*p4p7+p1p9*p3p6*p5p7+p2p3*p4p6*p7p9-
     . p2p3*p5p6*p7p9+p2p3*p5p7*p6p9-p3p4*p4p6*p7p9+p3p4*p5p6*p7p9-
     . p3p4*p5p7*p6p9-p3p5*p5p7*p6p9))
      ans2=w2*ans3
      ans6=16*(-hbcm2*p5p7*p6p11+p1p11*p3p6*p4p7+p1p11*p3p6*p5p7+p1p2
     . *p3p6*p7p11+p1p3*p4p6*p7p11-p1p3*p5p7*p6p11-p1p5*p3p6*p7p11+
     . p2p3*p5p6*p7p11+p2p3*p5p7*p6p11+p2p5*p3p6*p7p11+p3p4*p4p6*
     . p7p11-2*p3p4*p5p6*p7p11-p3p4*p5p7*p6p11+p3p5*p4p6*p7p11-p3p5*
     . p5p7*p6p11)
      ans5=16*(p5p6*p3p7)*(3*p2p11-3*p3p11-3*p4p11-2*p5p11)+16*(p4p6*
     . p3p7)*(2*p1p11-3*p2p11+3*p3p11+3*p4p11+2*p5p11)+16*(p6p7*p3p5)
     . *(p1p11+p5p11)+16*(p6p7*p3p4)*(p1p11+p5p11)+16*(p6p7*p2p3)*(-
     . p1p11-p5p11)+16*(p6p7*p1p3)*(-p1p11+p5p11)+16*p6p7*(hbcm2*
     . p1p11+hbcm2*p5p11)+8*p5p6*(fmb*hbcm*p7p11-fmc*hbcm*p7p11-2*
     . hbcm2*p7p11)+16*(p3p7*p3p5)*(ffmcfmb*p6p11-p6p11)+16*(p3p7*
     . p3p4)*(ffmcfmb*p6p11-p6p11)+16*(p3p6*p3p4)*(2*ffmcfmb*p7p11-
     . p7p11)+16*(p3p7*p2p3)*(-ffmcfmb*p6p11+p6p11)+16*(p3p6*p2p3)*(-
     . ffmcfmb*p7p11+p7p11)+16*(p3p7*p3p6)*(-4*ffmcfmb*p2p11+4*
     . ffmcfmb*p3p11+4*ffmcfmb*p4p11+2*ffmcfmb*p5p11+2*p1p11+2*p2p11-
     . 2*p3p11-2*p4p11-p5p11)+16*(p3p7*p1p3)*(ffmcfmb*p6p11-p6p11)+8*
     . p4p6*(-3*ffmcfmb*hbcm2*p7p11+fmc*hbcm*p7p11+3*hbcm2*p7p11)+16*
     . p3p7*(ffmcfmb*hbcm2*p6p11-hbcm2*p6p11)+8*p3p6*(-ffmcfmb**2*
     . hbcm2*p7p11+2*ffmcfmb*fmc*hbcm*p7p11+2*ffmcfmb*hbcm2*p7p11+fmb
     . *hbcm*p7p11-fmb2*p7p11-fmc*hbcm*p7p11-hbcm2*p7p11)+ans6
      ans4=w7*ans5
      ans9=16*(p3p6*p2p3)*(-ffmcfmb*p7p11-2*ffmcfmb*p7p8+p7p11+p7p8)+
     . 16*(p3p7*p3p6)*(-4*ffmcfmb*p2p11+4*ffmcfmb*p2p8+4*ffmcfmb*
     . p3p11-4*ffmcfmb*p3p8+4*ffmcfmb*p4p11-4*ffmcfmb*p4p8+2*ffmcfmb*
     . p5p11-2*ffmcfmb*p5p8+2*p1p11-2*p1p8+2*p2p11-2*p2p8-2*p3p11+2*
     . p3p8-2*p4p11+2*p4p8-p5p11+p5p8)+16*(p3p7*p1p3)*(ffmcfmb*p6p11-
     . ffmcfmb*p6p8-p6p11+p6p8)+8*p5p6*(-ffmcfmb*hbcm2*p7p8+fmb*hbcm*
     . p7p11+fmb*hbcm*p7p8-fmc*hbcm*p7p11-2*hbcm2*p7p11-hbcm2*p7p8)+8
     . *p4p6*(-3*ffmcfmb*hbcm2*p7p11+ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p8
     . +fmc*hbcm*p7p11+3*hbcm2*p7p11+hbcm2*p7p8)+16*p3p7*(ffmcfmb*
     . hbcm2*p6p11-ffmcfmb*hbcm2*p6p8-hbcm2*p6p11+hbcm2*p6p8)+8*p3p6*
     . (-ffmcfmb**2*hbcm2*p7p11+2*ffmcfmb**2*hbcm2*p7p8-2*ffmcfmb*fmb
     . *hbcm*p7p8+2*ffmcfmb*fmc*hbcm*p7p11+2*ffmcfmb*hbcm2*p7p11+
     . ffmcfmb*hbcm2*p7p8+fmb*hbcm*p7p11+fmb*hbcm*p7p8-fmb2*p7p11-fmc
     . *hbcm*p7p11-hbcm2*p7p11-hbcm2*p7p8)+16*(p1p2*p3p6*p7p11+p1p3*
     . p4p6*p7p11-p1p5*p3p6*p7p11-p2p3*p4p6*p7p8+p2p5*p3p6*p7p11+p3p5
     . *p4p6*p7p11)
      ans8=16*(p5p7*p3p5)*(-p6p11+p6p8)+16*(p5p7*p3p4)*(-p6p11+p6p8)+
     . 16*(p5p6*p3p4)*(-2*p7p11-p7p8)+16*(p4p6*p3p4)*(p7p11+p7p8)+16*
     . (p5p7*p2p3)*(p6p11-p6p8)+16*(p5p6*p2p3)*(p7p11+p7p8)+16*(p5p6*
     . p3p7)*(3*p2p11-3*p2p8-3*p3p11+3*p3p8-3*p4p11+3*p4p8-2*p5p11+2*
     . p5p8)+16*(p5p7*p1p3)*(-p6p11+p6p8)+16*(p4p6*p3p7)*(2*p1p11-2*
     . p1p8-3*p2p11+3*p2p8+3*p3p11-3*p3p8+3*p4p11-3*p4p8+2*p5p11-2*
     . p5p8)+16*(p5p7*p3p6)*(p1p11-p1p8)+16*(p4p7*p3p6)*(p1p11-p1p8)+
     . 16*(p6p7*p3p5)*(p1p11-p1p8+p5p11-p5p8)+16*(p6p7*p3p4)*(p1p11-
     . p1p8+p5p11-p5p8)+16*(p6p7*p2p3)*(-p1p11+p1p8-p5p11+p5p8)+16*(
     . p6p7*p1p3)*(-p1p11+p1p8+p5p11-p5p8)+16*p5p7*(-hbcm2*p6p11+
     . hbcm2*p6p8)+16*p6p7*(hbcm2*p1p11-hbcm2*p1p8+hbcm2*p5p11-hbcm2*
     . p5p8)+16*(p3p7*p3p5)*(ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+
     . 16*(p3p7*p3p4)*(ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+16*(
     . p3p6*p3p4)*(2*ffmcfmb*p7p11+2*ffmcfmb*p7p8-p7p11-p7p8)+16*(
     . p3p7*p2p3)*(-ffmcfmb*p6p11+ffmcfmb*p6p8+p6p11-p6p8)+ans9
      ans7=w11*ans8
      ans10=(8*p3p4*(p6p12+2*p7p13)+8*p6p7*(p3p10+p3p9)+8*p3p7*(-
     . ffmcfmb*p3p13-4*p2p13+4*p3p13+4*p4p13+2*p5p13)+8*p3p6*(2*
     . ffmcfmb*p3p12-p1p12-p3p12+p5p12-p7p10-p7p11-p7p9)+4*(-2*
     . ffmcfmb*hbcm2*p7p13-fmb*hbcm*p6p12+fmb*hbcm*p7p13+fmc*hbcm*
     . p6p12+fmc*hbcm*p7p13+hbcm2*p6p12+3*hbcm2*p7p13+2*p1p3*p7p13-2*
     . p2p3*p6p12+4*p3p12*p4p6-4*p3p12*p5p6-2*p3p13*p4p7+2*p3p5*p7p13
     . ))
      ans1=ans2+ans4+ans7+ans10
      ans=ccc*ans1
      b(30)=ans
      b(31)=w11*ccc*(4*p5p6*(-2*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm)+4*p4p6
     . *(ffmcfmb*hbcm+2*fmb+fmc-hbcm)+4*p3p6*(2*ffmcfmb**2*hbcm+4*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-4*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm))
      b(32)=w2*ccc*(4*p5p6*(-2*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm)+4*p4p6*
     . (ffmcfmb*hbcm+2*fmb+fmc-hbcm)+4*p3p6*(2*ffmcfmb**2*hbcm+4*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-4*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm))
      ans2=w11*(8*p4p6*(-fmb*p7p11+fmb*p7p8-fmc*p7p11+fmc*p7p8)+8*
     . p5p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-
     . hbcm*p6p11+hbcm*p6p8)+8*p5p6*(ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*
     . p7p8+2*fmb*p7p11-2*fmb*p7p8+fmc*p7p11-fmc*p7p8-hbcm*p7p11+hbcm
     . *p7p8)+8*p4p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-
     . fmb*p6p8-hbcm*p6p11+hbcm*p6p8)+8*p3p7*(ffmcfmb*hbcm*p6p11-
     . ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-hbcm*p6p11+hbcm*p6p8)+16*
     . p6p7*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p1p8-fmb*p1p11+fmb*p1p8
     . +hbcm*p1p11-hbcm*p1p8)+8*p3p6*(-2*ffmcfmb*fmb*p7p11+2*ffmcfmb*
     . fmb*p7p8-2*ffmcfmb*fmc*p7p11+2*ffmcfmb*fmc*p7p8+ffmcfmb*hbcm*
     . p7p11-ffmcfmb*hbcm*p7p8+2*fmb*p7p11-2*fmb*p7p8+fmc*p7p11-fmc*
     . p7p8-hbcm*p7p11+hbcm*p7p8))+8*(-fmb*p7p13-fmc*p7p13)
      ans1=w2*(8*p4p6*(-fmb*p7p9-fmc*p7p9)+8*p5p7*(ffmcfmb*hbcm*p6p9+
     . fmb*p6p9-hbcm*p6p9)+8*p5p6*(ffmcfmb*hbcm*p7p9+2*fmb*p7p9+fmc*
     . p7p9-hbcm*p7p9)+8*p4p7*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+
     . 8*p3p7*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+8*p3p6*(-2*
     . ffmcfmb*fmb*p7p9-2*ffmcfmb*fmc*p7p9+ffmcfmb*hbcm*p7p9+2*fmb*
     . p7p9+fmc*p7p9-hbcm*p7p9))+w7*(8*p4p6*(-fmb*p7p11-fmc*p7p11)+8*
     . p5p7*(ffmcfmb*hbcm*p6p11+fmb*p6p11-hbcm*p6p11)+8*p5p6*(ffmcfmb
     . *hbcm*p7p11+2*fmb*p7p11+fmc*p7p11-hbcm*p7p11)+8*p4p7*(ffmcfmb*
     . hbcm*p6p11+fmb*p6p11-hbcm*p6p11)+8*p3p7*(ffmcfmb*hbcm*p6p11+
     . fmb*p6p11-hbcm*p6p11)+16*p6p7*(-ffmcfmb*hbcm*p1p11-fmb*p1p11+
     . hbcm*p1p11)+8*p3p6*(-2*ffmcfmb*fmb*p7p11-2*ffmcfmb*fmc*p7p11+
     . ffmcfmb*hbcm*p7p11+2*fmb*p7p11+fmc*p7p11-hbcm*p7p11))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(34)=6*ccc*w2*(ffmcfmb*hbcm+fmb-hbcm)
      b(35)=ccc*(12*w2*(-ffmcfmb*hbcm*p7p9-fmb*p7p9+hbcm*p7p9)+12*w7*
     . (-ffmcfmb*hbcm*p7p11-fmb*p7p11+hbcm*p7p11)+12*w11*(-ffmcfmb*
     . hbcm*p7p11+ffmcfmb*hbcm*p7p8-fmb*p7p11+fmb*p7p8+hbcm*p7p11-
     . hbcm*p7p8))
      b(36)=2*ccc*w2*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+2*p1p3+6*p2p3
     . -6*p3p4+2*p3p5)
      b(37)=w2*ccc*(4*p3p7*(-ffmcfmb*hbcm+3*fmb)+4*(-hbcm*p4p7+3*hbcm
     . *p5p7))
      b(38)=ccc*(w2*(8*p3p7*(-p1p9+p2p9-p4p9+3*p5p9)+4*(5*ffmcfmb*
     . hbcm2*p7p9-3*fmb*hbcm*p7p9+hbcm2*p7p9-2*p1p3*p7p9-6*p2p3*p7p9+
     . 6*p3p4*p7p9-2*p3p5*p7p9))+w7*(8*p3p7*(-p1p11+p2p11-p4p11+3*
     . p5p11)+4*(5*ffmcfmb*hbcm2*p7p11-3*fmb*hbcm*p7p11+hbcm2*p7p11-2
     . *p1p3*p7p11-6*p2p3*p7p11+6*p3p4*p7p11-2*p3p5*p7p11))+w11*(8*
     . p3p5*(-p7p11+p7p8)+24*p3p4*(p7p11-p7p8)+24*p2p3*(-p7p11+p7p8)+
     . 8*p1p3*(-p7p11+p7p8)+8*p3p7*(-p1p11+p1p8+p2p11-p2p8-p4p11+p4p8
     . +3*p5p11-3*p5p8)+4*(5*ffmcfmb*hbcm2*p7p11-5*ffmcfmb*hbcm2*p7p8
     . -3*fmb*hbcm*p7p11+3*fmb*hbcm*p7p8+hbcm2*p7p11-hbcm2*p7p8))+8*
     . p3p12)
      b(39)=ccc*(w7*((p3p7*p3p6)*(-32*ffmcfmb+8)+8*p6p7*(ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+8*(-2*p3p5*p6p7-p3p6*p4p7+p3p6*p5p7-4*
     . p3p7*p4p6+2*p3p7*p5p6))+w11*((p3p7*p3p6)*(-32*ffmcfmb+8)+8*
     . p6p7*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(-2*p3p5*p6p7-p3p6*p4p7+
     . p3p6*p5p7-4*p3p7*p4p6+2*p3p7*p5p6)))
      ans=ccc*(w7*(8*(p5p6*p3p7)*(-ffmcfmb*hbcm-2*fmb-fmc)+8*(p4p6*
     . p3p7)*(2*ffmcfmb*hbcm+fmb+fmc)+16*(p5p7*p3p6)*(ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p3)*(2*
     . ffmcfmb*hbcm+2*fmb-hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3+fmb2*hbcm+
     . hbcm3)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm+2*ffmcfmb*fmb+2*
     . ffmcfmb*fmc-ffmcfmb*hbcm-2*fmb-fmc)+8*(-hbcm*p1p2*p6p7+hbcm*
     . p1p5*p6p7-hbcm*p2p5*p6p7+hbcm*p3p5*p6p7+hbcm*p4p6*p4p7+hbcm*
     . p4p6*p5p7-2*hbcm*p5p6*p5p7))+w11*(8*(p5p6*p3p7)*(-ffmcfmb*hbcm
     . -2*fmb-fmc)+8*(p4p6*p3p7)*(2*ffmcfmb*hbcm+fmb+fmc)+16*(p5p7*
     . p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-hbcm)+8*
     . (p6p7*p1p3)*(2*ffmcfmb*hbcm+2*fmb-hbcm)+4*p6p7*(-ffmcfmb**2*
     . hbcm3+fmb2*hbcm+hbcm3)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm+2*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm-2*fmb-fmc)+8*(-hbcm*
     . p1p2*p6p7+hbcm*p1p5*p6p7-hbcm*p2p5*p6p7+hbcm*p3p5*p6p7+hbcm*
     . p4p6*p4p7+hbcm*p4p6*p5p7-2*hbcm*p5p6*p5p7))+4*hbcm*p6p7)
      b(40)=ans
      b(41)=ccc*(w7*((p5p6*p3p7)*(8*ffmcfmb-16)+(p4p6*p3p7)*(8*
     . ffmcfmb+8)+(p5p7*p3p6)*(32*ffmcfmb-24)+(p3p7*p3p6)*(24*ffmcfmb
     . -16)+(p6p7*p2p3)*(8*ffmcfmb-8)+4*p6p7*(-ffmcfmb**2*hbcm2+fmb2+
     . hbcm2)+8*(-p1p2*p6p7+p1p3*p6p7+p1p5*p6p7-p2p5*p6p7+p3p5*p6p7+
     . p4p6*p4p7+2*p4p6*p5p7-3*p5p6*p5p7))+w11*((p5p6*p3p7)*(8*
     . ffmcfmb-16)+(p4p6*p3p7)*(8*ffmcfmb+8)+(p5p7*p3p6)*(32*ffmcfmb-
     . 24)+(p3p7*p3p6)*(24*ffmcfmb-16)+(p6p7*p2p3)*(8*ffmcfmb-8)+4*
     . p6p7*(-ffmcfmb**2*hbcm2+fmb2+hbcm2)+8*(-p1p2*p6p7+p1p3*p6p7+
     . p1p5*p6p7-p2p5*p6p7+p3p5*p6p7+p4p6*p4p7+2*p4p6*p5p7-3*p5p6*
     . p5p7))+4*p6p7)
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb+8)+8*p6p7*(ffmcfmb*hbcm2
     . -fmb*hbcm-hbcm2)+8*(-2*p3p5*p6p7-p3p6*p4p7+p3p6*p5p7-4*p3p7*
     . p4p6+2*p3p7*p5p6))
      b(43)=ccc*(w2*(8*p3p7*(-2*ffmcfmb*p6p9+p6p9)+8*(-2*ffmcfmb*p3p6
     . *p7p9-4*p4p6*p7p9-p4p7*p6p9+2*p5p6*p7p9+p5p7*p6p9-2*p5p9*p6p7)
     . )+w7*(8*p3p7*(-2*ffmcfmb*p6p11+p6p11)+8*(-2*ffmcfmb*p3p6*p7p11
     . -4*p4p6*p7p11-p4p7*p6p11-2*p5p11*p6p7+2*p5p6*p7p11+p5p7*p6p11)
     . )+w11*(8*p5p7*(p6p11-p6p8)+16*p5p6*(p7p11-p7p8)+16*p6p7*(-
     . p5p11+p5p8)+8*p4p7*(-p6p11+p6p8)+32*p4p6*(-p7p11+p7p8)+8*p3p7*
     . (-2*ffmcfmb*p6p11+2*ffmcfmb*p6p8+p6p11-p6p8)+16*p3p6*(-ffmcfmb
     . *p7p11+ffmcfmb*p7p8))+8*(-p6p12-p7p13))
      ans=ccc*(w7*((p3p6*p3p4)*(-32*ffmcfmb+16)+(p3p6*p2p3)*(24*
     . ffmcfmb-16)+4*p5p6*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm+3*hbcm2)
     . +4*p4p6*(2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-4*hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-3*
     . ffmcfmb*hbcm2-2*fmb*hbcm+fmb2+fmc*hbcm+2*hbcm2)+8*(-p1p2*p3p6-
     . p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*p2p3*p5p6-p2p5*p3p6-2*p3p4*
     . p4p6+3*p3p4*p5p6-p3p5*p4p6))+w11*((p3p6*p3p4)*(-32*ffmcfmb+16)
     . +(p3p6*p2p3)*(24*ffmcfmb-16)+4*p5p6*(ffmcfmb*hbcm2-2*fmb*hbcm+
     . fmc*hbcm+3*hbcm2)+4*p4p6*(2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-4*
     . hbcm2)+4*p3p6*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2-2*fmb*hbcm+fmb2+fmc*hbcm+2*hbcm2)+8*(
     . -p1p2*p3p6-p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*p2p3*p5p6-p2p5*p3p6
     . -2*p3p4*p4p6+3*p3p4*p5p6-p3p5*p4p6))+4*p3p6)
      b(44)=ans
      b(45)=ccc*(w7*(p3p6*(-16*ffmcfmb+4)+4*(-5*p4p6+3*p5p6))+w11*(
     . p3p6*(-16*ffmcfmb+4)+4*(-5*p4p6+3*p5p6)))
      b(46)=ccc*(w7*(4*p5p6*(-2*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm)+4*p4p6
     . *(ffmcfmb*hbcm+2*fmb+fmc-hbcm)+4*p3p6*(2*ffmcfmb**2*hbcm+4*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-4*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm))+
     . w11*(4*p5p6*(-2*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm)+4*p4p6*(ffmcfmb
     . *hbcm+2*fmb+fmc-hbcm)+4*p3p6*(2*ffmcfmb**2*hbcm+4*ffmcfmb*fmb+
     . 2*ffmcfmb*fmc-4*ffmcfmb*hbcm-3*fmb-fmc+2*hbcm)))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb+4)+4*(-5*p4p6+3*p5p6))
      b(48)=4*ccc*(p3p6+p4p6+p5p6)
      b(49)=ccc*(w7*(4*p3p7*(-ffmcfmb*hbcm+3*fmb)+4*(-hbcm*p4p7+3*
     . hbcm*p5p7))+w11*(4*p3p7*(-ffmcfmb*hbcm+3*fmb)+4*(-hbcm*p4p7+3*
     . hbcm*p5p7)))
      b(50)=ccc*(w7*(p3p7*(-16*ffmcfmb+12)+4*(-p4p7+3*p5p7))+w11*(
     . p3p7*(-16*ffmcfmb+12)+4*(-p4p7+3*p5p7)))
      b(51)=ccc*(w7*((p3p7*p3p5)*(8*ffmcfmb-8)+(p3p7*p2p3)*(-16*
     . ffmcfmb+8)+4*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p3p7*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb2+fmc2-
     . hbcm2)+8*(-ffmcfmb*hbcm2*p5p7+ffmcfmb*p1p3*p3p7+ffmcfmb*p3p4*
     . p3p7-p1p2*p3p7-p1p3*p5p7+2*p1p4*p3p7+p2p3*p5p7-p2p4*p3p7+p2p5*
     . p3p7-p3p4*p5p7+p3p5*p4p7))+w11*((p3p7*p3p5)*(8*ffmcfmb-8)+(
     . p3p7*p2p3)*(-16*ffmcfmb+8)+4*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm+
     . hbcm2)+4*p3p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2
     . -fmb2+fmc2-hbcm2)+8*(-ffmcfmb*hbcm2*p5p7+ffmcfmb*p1p3*p3p7+
     . ffmcfmb*p3p4*p3p7-p1p2*p3p7-p1p3*p5p7+2*p1p4*p3p7+p2p3*p5p7-
     . p2p4*p3p7+p2p5*p3p7-p3p4*p5p7+p3p5*p4p7))-4*p3p7)
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+12)+4*(-p4p7+3*p5p7))
      b(53)=ccc*(4*w2*(-p1p9-3*p2p9+2*p3p9+3*p4p9-p5p9)+4*w7*(-p1p11-
     . 3*p2p11+2*p3p11+3*p4p11-p5p11)+4*w11*(-p1p11+p1p8-3*p2p11+3*
     . p2p8+2*p3p11-2*p3p8+3*p4p11-3*p4p8-p5p11+p5p8))
      b(54)=ccc*(2*w7*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+2*p1p3+6*
     . p2p3-6*p3p4+2*p3p5)+2*w11*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+2
     . *p1p3+6*p2p3-6*p3p4+2*p3p5))
      b(55)=ccc*(6*w7*(ffmcfmb*hbcm+fmb-hbcm)+6*w11*(ffmcfmb*hbcm+fmb
     . -hbcm))
      b(56)=ccc*(2*w7*(2*ffmcfmb*hbcm2-2*ffmcfmb*p2p3-fmb2+fmc2-hbcm2
     . -2*p1p2+2*p1p3+4*p1p4-2*p2p4+2*p2p5+2*p3p4-2*p3p5)+2*w11*(2*
     . ffmcfmb*hbcm2-2*ffmcfmb*p2p3-fmb2+fmc2-hbcm2-2*p1p2+2*p1p3+4*
     . p1p4-2*p2p4+2*p2p5+2*p3p4-2*p3p5)-2)
      b(57)=-8*ccc
      b(58)=ccc*(w7*(8*p3p6*(2*ffmcfmb*hbcm+fmb-hbcm)+8*(2*hbcm*p4p6-
     . hbcm*p5p6))+w11*(8*p3p6*(2*ffmcfmb*hbcm+fmb-hbcm)+8*(2*hbcm*
     . p4p6-hbcm*p5p6)))
      b(59)=ccc*(w7*(4*p3p5*(ffmcfmb*hbcm+fmb)+4*p1p3*(-2*ffmcfmb*
     . hbcm-2*fmb+hbcm)+2*(-2*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*
     . ffmcfmb*hbcm*p2p3+2*ffmcfmb*hbcm3+2*fmb*hbcm2+fmb2*hbcm-fmc2*
     . hbcm+2*hbcm*p1p2-4*hbcm*p1p4+2*hbcm*p2p4-2*hbcm*p2p5-2*hbcm*
     . p3p4-hbcm3))+w11*(4*p3p5*(ffmcfmb*hbcm+fmb)+4*p1p3*(-2*ffmcfmb
     . *hbcm-2*fmb+hbcm)+2*(-2*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2
     . *ffmcfmb*hbcm*p2p3+2*ffmcfmb*hbcm3+2*fmb*hbcm2+fmb2*hbcm-fmc2*
     . hbcm+2*hbcm*p1p2-4*hbcm*p1p4+2*hbcm*p2p4-2*hbcm*p2p5-2*hbcm*
     . p3p4-hbcm3))+2*hbcm)
      b(60)=w2*ccc*(8*p3p6*(2*ffmcfmb*hbcm+fmb-hbcm)+8*(2*hbcm*p4p6-
     . hbcm*p5p6))
      b(61)=-8*ccc*hbcm
      b(62)=ccc*(4*p3p6*(ffmcfmb*hbcm-fmb+hbcm)+8*hbcm*p4p6)
      b(65)=w11*ccc*((p3p7*p3p6)*(32*ffmcfmb-8)+8*p6p7*(-ffmcfmb*
     . hbcm2+fmb*hbcm+hbcm2)+8*(2*p3p5*p6p7+p3p6*p4p7-p3p6*p5p7+4*
     . p3p7*p4p6-2*p3p7*p5p6))
      ans=ccc*(w2*(8*p3p7*(-2*ffmcfmb*hbcm*p6p9+hbcm*p6p9)+8*p3p6*(-2
     . *ffmcfmb*hbcm*p7p9-2*fmb*p7p9+hbcm*p7p9)+8*(-3*hbcm*p4p6*p7p9-
     . hbcm*p4p7*p6p9+hbcm*p5p6*p7p9+hbcm*p5p7*p6p9-2*hbcm*p5p9*p6p7)
     . )+w7*(8*p3p7*(-2*ffmcfmb*hbcm*p6p11+hbcm*p6p11)+8*p3p6*(-2*
     . ffmcfmb*hbcm*p7p11-2*fmb*p7p11+hbcm*p7p11)+8*(-3*hbcm*p4p6*
     . p7p11-hbcm*p4p7*p6p11-2*hbcm*p5p11*p6p7+hbcm*p5p6*p7p11+hbcm*
     . p5p7*p6p11))+w11*(8*p5p7*(hbcm*p6p11-hbcm*p6p8)+8*p5p6*(hbcm*
     . p7p11-hbcm*p7p8)+16*p6p7*(-hbcm*p5p11+hbcm*p5p8)+8*p4p7*(-hbcm
     . *p6p11+hbcm*p6p8)+24*p4p6*(-hbcm*p7p11+hbcm*p7p8)+8*p3p7*(-2*
     . ffmcfmb*hbcm*p6p11+2*ffmcfmb*hbcm*p6p8+hbcm*p6p11-hbcm*p6p8)+8
     . *p3p6*(-2*ffmcfmb*hbcm*p7p11+2*ffmcfmb*hbcm*p7p8-2*fmb*p7p11+2
     . *fmb*p7p8+hbcm*p7p11-hbcm*p7p8))-8*hbcm*p6p12)
      b(66)=ans
      b(68)=w11*ccc*(p3p6*(16*ffmcfmb-4)+4*(5*p4p6-3*p5p6))
      b(71)=w11*ccc*(p3p7*(16*ffmcfmb-12)+4*(p4p7-3*p5p7))
      b(73)=ccc*(8*w7*p6p7*(ffmcfmb*hbcm+fmb-hbcm)+8*w11*p6p7*(
     . ffmcfmb*hbcm+fmb-hbcm))
      b(74)=8*ccc*w11*p6p7*(-ffmcfmb*hbcm-fmb+hbcm)
      b(75)=ccc*(4*w2*(2*ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p3p9-ffmcfmb*
     . hbcm*p5p9+2*fmb*p1p9-fmb*p3p9-fmb*p5p9-2*hbcm*p1p9+hbcm*p3p9+
     . hbcm*p5p9)+4*w7*(2*ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p3p11-
     . ffmcfmb*hbcm*p5p11+2*fmb*p1p11-fmb*p3p11-fmb*p5p11-2*hbcm*
     . p1p11+hbcm*p3p11+hbcm*p5p11)+4*w11*(2*ffmcfmb*hbcm*p1p11-2*
     . ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p3p11+ffmcfmb*hbcm*p3p8-ffmcfmb
     . *hbcm*p5p11+ffmcfmb*hbcm*p5p8+2*fmb*p1p11-2*fmb*p1p8-fmb*p3p11
     . +fmb*p3p8-fmb*p5p11+fmb*p5p8-2*hbcm*p1p11+2*hbcm*p1p8+hbcm*
     . p3p11-hbcm*p3p8+hbcm*p5p11-hbcm*p5p8))
      ans2=w11*(8*p3p5*(-2*p1p11+2*p1p8+p3p11-p3p8+p5p11-p5p8)+8*p3p4
     . *(p1p11-p1p8+p2p11-p2p8-p3p11+p3p8-p4p11+p4p8)+8*p2p3*(-p1p11+
     . p1p8-p2p11+p2p8+p3p11-p3p8+p4p11-p4p8)+8*p1p3*(p1p11-p1p8+
     . p2p11-p2p8-p3p11+p3p8-p4p11+p4p8)+4*(4*ffmcfmb*hbcm2*p1p11-4*
     . ffmcfmb*hbcm2*p1p8+2*ffmcfmb*hbcm2*p2p11-2*ffmcfmb*hbcm2*p2p8-
     . 3*ffmcfmb*hbcm2*p3p11+3*ffmcfmb*hbcm2*p3p8-2*ffmcfmb*hbcm2*
     . p4p11+2*ffmcfmb*hbcm2*p4p8-ffmcfmb*hbcm2*p5p11+ffmcfmb*hbcm2*
     . p5p8-2*fmb*hbcm*p1p11+2*fmb*hbcm*p1p8+fmb*hbcm*p3p11-fmb*hbcm*
     . p3p8+fmb*hbcm*p5p11-fmb*hbcm*p5p8-2*hbcm2*p1p11+2*hbcm2*p1p8+
     . hbcm2*p3p11-hbcm2*p3p8+hbcm2*p5p11-hbcm2*p5p8))
      ans1=w2*(8*p3p5*(-2*p1p9+p3p9+p5p9)+8*p3p4*(p1p9+p2p9-p3p9-p4p9
     . )+8*p2p3*(-p1p9-p2p9+p3p9+p4p9)+8*p1p3*(p1p9+p2p9-p3p9-p4p9)+4
     . *(4*ffmcfmb*hbcm2*p1p9+2*ffmcfmb*hbcm2*p2p9-3*ffmcfmb*hbcm2*
     . p3p9-2*ffmcfmb*hbcm2*p4p9-ffmcfmb*hbcm2*p5p9-2*fmb*hbcm*p1p9+
     . fmb*hbcm*p3p9+fmb*hbcm*p5p9-2*hbcm2*p1p9+hbcm2*p3p9+hbcm2*p5p9
     . ))+w7*(8*p3p5*(-2*p1p11+p3p11+p5p11)+8*p3p4*(p1p11+p2p11-p3p11
     . -p4p11)+8*p2p3*(-p1p11-p2p11+p3p11+p4p11)+8*p1p3*(p1p11+p2p11-
     . p3p11-p4p11)+4*(4*ffmcfmb*hbcm2*p1p11+2*ffmcfmb*hbcm2*p2p11-3*
     . ffmcfmb*hbcm2*p3p11-2*ffmcfmb*hbcm2*p4p11-ffmcfmb*hbcm2*p5p11-
     . 2*fmb*hbcm*p1p11+fmb*hbcm*p3p11+fmb*hbcm*p5p11-2*hbcm2*p1p11+
     . hbcm2*p3p11+hbcm2*p5p11))+ans2
      ans=ccc*ans1
      b(76)=ans
      b(77)=ccc*(8*w2*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9-hbcm2*p6p9-2*
     . p3p5*p6p9+2*p3p6*p5p9)+8*w7*(ffmcfmb*hbcm2*p6p11-fmb*hbcm*
     . p6p11-hbcm2*p6p11-2*p3p5*p6p11+2*p3p6*p5p11)+w11*(16*p3p6*(
     . p5p11-p5p8)+16*p3p5*(-p6p11+p6p8)+8*(ffmcfmb*hbcm2*p6p11-
     . ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm*p6p8-hbcm2*p6p11+
     . hbcm2*p6p8)))
      b(78)=ccc*(8*w2*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+8*w7*(
     . ffmcfmb*hbcm*p6p11+fmb*p6p11-hbcm*p6p11)+8*w11*(ffmcfmb*hbcm*
     . p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-hbcm*p6p11+hbcm*
     . p6p8))
      b(79)=8*ccc*w2*p6p7*(ffmcfmb*hbcm+fmb-hbcm)
      b(80)=w2*ccc*(4*p4p7*(-ffmcfmb*hbcm-fmb+hbcm)+4*p3p7*(-ffmcfmb
     . **2*hbcm-ffmcfmb*fmb+ffmcfmb*hbcm))
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(84)=8*ccc*(p3p7+2*p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p2p3-4*p3p4)
      b(86)=-8*ccc
      b(87)=4*ccc*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(89)=-4*ccc*hbcm
      b(90)=ccc*(w7*(4*p4p7*(-ffmcfmb*hbcm-fmb+hbcm)+4*p3p7*(-ffmcfmb
     . **2*hbcm-ffmcfmb*fmb+ffmcfmb*hbcm))+w11*(4*p4p7*(-ffmcfmb*hbcm
     . -fmb+hbcm)+4*p3p7*(-ffmcfmb**2*hbcm-ffmcfmb*fmb+ffmcfmb*hbcm))
     . )
      b(91)=w11*ccc*(4*p4p7*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p7*(ffmcfmb**
     . 2*hbcm+ffmcfmb*fmb-ffmcfmb*hbcm))
      b(92)=2*ccc*(ffmcfmb*hbcm2+fmb*hbcm+hbcm2+2*p1p3-2*p2p3+2*p3p4+
     . 2*p3p5)
      b(93)=8*ccc*p3p6
      b(94)=2*ccc*(ffmcfmb*hbcm+fmb-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp29_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=ccc*(w5*(4*p6p7*(ffmcfmb**2*hbcm2-fmc2)+8*(-2*ffmcfmb*p3p6
     . *p3p7-4*ffmcfmb*p3p6*p5p7+2*p1p2*p6p7-2*p1p3*p6p7-2*p1p5*p6p7-
     . 3*p3p7*p4p6-p4p6*p4p7-5*p4p6*p5p7))+w1*(4*p6p7*(ffmcfmb**2*
     . hbcm2-fmc2)+8*(-2*ffmcfmb*p3p6*p3p7-4*ffmcfmb*p3p6*p5p7+2*p1p2
     . *p6p7-2*p1p3*p6p7-2*p1p5*p6p7-3*p3p7*p4p6-p4p6*p4p7-5*p4p6*
     . p5p7)))
      b(3)=ccc*(w1*(8*p4p7*(p1p8-p2p8+p4p8+p5p8)+8*p3p7*(ffmcfmb*p1p8
     . -ffmcfmb*p2p8+ffmcfmb*p4p8+ffmcfmb*p5p8))+w8*(8*p4p7*(p1p11-
     . p2p11+p4p11+p5p11)+8*p3p7*(ffmcfmb*p1p11-ffmcfmb*p2p11+ffmcfmb
     . *p4p11+ffmcfmb*p5p11))+w5*(8*p4p7*(-p1p10+p1p8+p2p10-p2p8-
     . p4p10+p4p8-p5p10+p5p8)+8*p3p7*(-ffmcfmb*p1p10+ffmcfmb*p1p8+
     . ffmcfmb*p2p10-ffmcfmb*p2p8-ffmcfmb*p4p10+ffmcfmb*p4p8-ffmcfmb*
     . p5p10+ffmcfmb*p5p8))+4*(p2p12-p5p12-p7p10-p7p8))
      b(4)=ccc*(w5*((p3p7*p3p4)*(-8*ffmcfmb+8)+(p3p7*p1p3)*(-8*
     . ffmcfmb+8)+4*p4p7*(ffmcfmb*hbcm2+fmc*hbcm-2*hbcm2)+4*p3p7*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+8*(ffmcfmb*hbcm2*p5p7+
     . ffmcfmb*p2p3*p3p7-ffmcfmb*p3p5*p3p7+p1p3*p5p7+p2p3*p4p7+p3p4*
     . p5p7-p3p5*p4p7))+w1*((p3p7*p3p4)*(-8*ffmcfmb+8)+(p3p7*p1p3)*(-
     . 8*ffmcfmb+8)+4*p4p7*(ffmcfmb*hbcm2+fmc*hbcm-2*hbcm2)+4*p3p7*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+8*(ffmcfmb*hbcm2*p5p7+
     . ffmcfmb*p2p3*p3p7-ffmcfmb*p3p5*p3p7+p1p3*p5p7+p2p3*p4p7+p3p4*
     . p5p7-p3p5*p4p7)))
      b(6)=ccc*(w1*(8*p4p7*(hbcm*p2p8-hbcm*p5p8)+8*p5p7*(hbcm*p1p8+
     . hbcm*p4p8)+8*p3p7*(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p8-
     . ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p8+hbcm*p1p8+hbcm*p4p8))+w8*(
     . 8*p4p7*(hbcm*p2p11-hbcm*p5p11)+8*p5p7*(hbcm*p1p11+hbcm*p4p11)+
     . 8*p3p7*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*
     . p4p11-ffmcfmb*hbcm*p5p11+hbcm*p1p11+hbcm*p4p11))+w5*(8*p4p7*(-
     . hbcm*p2p10+hbcm*p2p8+hbcm*p5p10-hbcm*p5p8)+8*p5p7*(-hbcm*p1p10
     . +hbcm*p1p8-hbcm*p4p10+hbcm*p4p8)+8*p3p7*(ffmcfmb*hbcm*p1p10-
     . ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p2p8+ffmcfmb
     . *hbcm*p4p10-ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p10-ffmcfmb*hbcm*
     . p5p8-hbcm*p1p10+hbcm*p1p8-hbcm*p4p10+hbcm*p4p8))+4*(hbcm*p1p12
     . -hbcm*p2p12+hbcm*p4p12+hbcm*p5p12))
      ans=ccc*(w1*(8*p4p6*(2*p1p8-2*p2p8+5*p3p8+4*p4p8)+16*p3p6*(-
     . ffmcfmb*p2p8+2*ffmcfmb*p3p8+2*ffmcfmb*p4p8+p1p8)+4*(ffmcfmb**2
     . *hbcm2*p6p8-fmc2*p6p8+4*p1p2*p6p8-4*p1p3*p6p8-4*p1p5*p6p8))+w8
     . *(8*p4p6*(2*p1p11-2*p2p11+5*p3p11+4*p4p11)+16*p3p6*(-ffmcfmb*
     . p2p11+2*ffmcfmb*p3p11+2*ffmcfmb*p4p11+p1p11)+4*(ffmcfmb**2*
     . hbcm2*p6p11-fmc2*p6p11+4*p1p2*p6p11-4*p1p3*p6p11-4*p1p5*p6p11)
     . )+w5*(16*p1p5*(p6p10-p6p8)+16*p1p3*(p6p10-p6p8)+16*p1p2*(-
     . p6p10+p6p8)+8*p4p6*(-2*p1p10+2*p1p8+2*p2p10-2*p2p8-5*p3p10+5*
     . p3p8-4*p4p10+4*p4p8)+16*p3p6*(ffmcfmb*p2p10-ffmcfmb*p2p8-2*
     . ffmcfmb*p3p10+2*ffmcfmb*p3p8-2*ffmcfmb*p4p10+2*ffmcfmb*p4p8-
     . p1p10+p1p8)+4*(-ffmcfmb**2*hbcm2*p6p10+ffmcfmb**2*hbcm2*p6p8+
     . fmc2*p6p10-fmc2*p6p8))+4*(3*p1p13+2*p2p13-6*p3p13-5*p4p13-
     . p5p13-2*p6p10+p6p11))
      b(7)=ans
      b(8)=ccc*(w5*(4*p4p6*(ffmcfmb*hbcm2+2*fmb*hbcm-3*fmc*hbcm-4*
     . hbcm2)+4*p3p6*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2-fmc*hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-
     . 2*ffmcfmb*p3p4*p3p6-p1p2*p3p6-p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*
     . p3p4*p4p6))+w1*(4*p4p6*(ffmcfmb*hbcm2+2*fmb*hbcm-3*fmc*hbcm-4*
     . hbcm2)+4*p3p6*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2-fmc*hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-
     . 2*ffmcfmb*p3p4*p3p6-p1p2*p3p6-p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*
     . p3p4*p4p6)))
      ans4=4*(-ffmcfmb**2*hbcm3*p6p10+ffmcfmb**2*hbcm3*p6p8+2*ffmcfmb
     . *fmc*hbcm2*p6p10-2*ffmcfmb*fmc*hbcm2*p6p8+2*ffmcfmb*hbcm3*
     . p6p10-2*ffmcfmb*hbcm3*p6p8-2*fmc*hbcm2*p6p10+2*fmc*hbcm2*p6p8-
     . fmc2*hbcm*p6p10+fmc2*hbcm*p6p8)
      ans3=16*p1p5*(-hbcm*p6p10+hbcm*p6p8)+16*p1p2*(hbcm*p6p10-hbcm*
     . p6p8)+8*p3p5*(ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p8-fmc*p6p10+
     . fmc*p6p8)+8*p4p6*(-3*ffmcfmb*hbcm*p3p10+3*ffmcfmb*hbcm*p3p8+2*
     . fmb*p3p10-2*fmb*p3p8+5*fmc*p3p10-5*fmc*p3p8-4*hbcm*p1p10+4*
     . hbcm*p1p8-2*hbcm*p2p10+2*hbcm*p2p8+3*hbcm*p3p10-3*hbcm*p3p8+2*
     . hbcm*p4p10-2*hbcm*p4p8)+8*p2p3*(-ffmcfmb*hbcm*p6p10+ffmcfmb*
     . hbcm*p6p8+fmc*p6p10-fmc*p6p8)+8*p1p3*(ffmcfmb*hbcm*p6p10-
     . ffmcfmb*hbcm*p6p8-fmc*p6p10+fmc*p6p8-2*hbcm*p6p10+2*hbcm*p6p8)
     . +8*p3p6*(-2*ffmcfmb**2*hbcm*p3p10+2*ffmcfmb**2*hbcm*p3p8+2*
     . ffmcfmb*fmb*p3p10-2*ffmcfmb*fmb*p3p8+4*ffmcfmb*fmc*p3p10-4*
     . ffmcfmb*fmc*p3p8-3*ffmcfmb*hbcm*p1p10+3*ffmcfmb*hbcm*p1p8-2*
     . ffmcfmb*hbcm*p2p10+2*ffmcfmb*hbcm*p2p8+2*ffmcfmb*hbcm*p3p10-2*
     . ffmcfmb*hbcm*p3p8+3*ffmcfmb*hbcm*p4p10-3*ffmcfmb*hbcm*p4p8+fmc
     . *p1p10-fmc*p1p8-fmc*p4p10+fmc*p4p8)+ans4
      ans2=w5*ans3
      ans5=4*(-3*hbcm*p1p13-2*hbcm*p2p13+4*hbcm*p3p13+3*hbcm*p4p13+
     . hbcm*p5p13+2*hbcm*p6p10-hbcm*p6p11)
      ans1=w8*(8*p4p6*(4*hbcm*p1p11+2*hbcm*p2p11-3*hbcm*p3p11-2*hbcm*
     . p4p11)+8*p3p5*(-ffmcfmb*hbcm*p6p11+fmc*p6p11)+8*p2p3*(ffmcfmb*
     . hbcm*p6p11-fmc*p6p11)+8*p1p3*(-ffmcfmb*hbcm*p6p11+fmc*p6p11+2*
     . hbcm*p6p11)+8*p3p6*(3*ffmcfmb*hbcm*p1p11+2*ffmcfmb*hbcm*p2p11-
     . 3*ffmcfmb*hbcm*p3p11-3*ffmcfmb*hbcm*p4p11-fmc*p1p11+fmc*p3p11+
     . fmc*p4p11)+4*(ffmcfmb**2*hbcm3*p6p11-2*ffmcfmb*fmc*hbcm2*p6p11
     . -2*ffmcfmb*hbcm3*p6p11+2*fmc*hbcm2*p6p11+fmc2*hbcm*p6p11-4*
     . hbcm*p1p2*p6p11+4*hbcm*p1p5*p6p11))+w1*(8*p4p6*(3*ffmcfmb*hbcm
     . *p3p8-2*fmb*p3p8-5*fmc*p3p8+4*hbcm*p1p8+2*hbcm*p2p8-3*hbcm*
     . p3p8-2*hbcm*p4p8)+8*p3p5*(-ffmcfmb*hbcm*p6p8+fmc*p6p8)+8*p2p3*
     . (ffmcfmb*hbcm*p6p8-fmc*p6p8)+8*p1p3*(-ffmcfmb*hbcm*p6p8+fmc*
     . p6p8+2*hbcm*p6p8)+8*p3p6*(2*ffmcfmb**2*hbcm*p3p8-2*ffmcfmb*fmb
     . *p3p8-4*ffmcfmb*fmc*p3p8+3*ffmcfmb*hbcm*p1p8+2*ffmcfmb*hbcm*
     . p2p8-2*ffmcfmb*hbcm*p3p8-3*ffmcfmb*hbcm*p4p8-fmc*p1p8+fmc*p4p8
     . )+4*(ffmcfmb**2*hbcm3*p6p8-2*ffmcfmb*fmc*hbcm2*p6p8-2*ffmcfmb*
     . hbcm3*p6p8+2*fmc*hbcm2*p6p8+fmc2*hbcm*p6p8-4*hbcm*p1p2*p6p8+4*
     . hbcm*p1p5*p6p8))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w5*(8*(p4p6*p3p7)*(-2*fmc-hbcm)+8*(p5p7*p3p6)*(-3*
     . ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p6p7*
     . p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p1p3)*(ffmcfmb*hbcm-fmc-2*
     . hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmc*hbcm2+2*ffmcfmb*
     . hbcm3-2*fmc*hbcm2-fmc2*hbcm)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm-3*
     . ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+8*(2*hbcm*p1p2*p6p7-2*hbcm*p1p5*
     . p6p7-hbcm*p4p6*p4p7-3*hbcm*p4p6*p5p7))+w1*(8*(p4p6*p3p7)*(-2*
     . fmc-hbcm)+8*(p5p7*p3p6)*(-3*ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5)*(
     . ffmcfmb*hbcm-fmc)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*
     . p1p3)*(ffmcfmb*hbcm-fmc-2*hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-2*fmc*hbcm2-fmc2*hbcm)+8*(
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-3*ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+8*
     . (2*hbcm*p1p2*p6p7-2*hbcm*p1p5*p6p7-hbcm*p4p6*p4p7-3*hbcm*p4p6*
     . p5p7)))
      b(11)=ans
      b(13)=ccc*(4*w5*(-ffmcfmb*hbcm2+ffmcfmb*p2p3+p1p2-p1p3-3*p1p4+
     . p2p4-p3p4)+4*w1*(-ffmcfmb*hbcm2+ffmcfmb*p2p3+p1p2-p1p3-3*p1p4+
     . p2p4-p3p4))
      b(15)=ccc*(w5*(4*p3p4*(-ffmcfmb*hbcm+fmc+hbcm)+4*p1p3*(-ffmcfmb
     . *hbcm+fmc+hbcm)+4*(-ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-ffmcfmb
     . *hbcm*p2p3+ffmcfmb*hbcm3-hbcm*p1p2+3*hbcm*p1p4-hbcm*p2p4))+w1*
     . (4*p3p4*(-ffmcfmb*hbcm+fmc+hbcm)+4*p1p3*(-ffmcfmb*hbcm+fmc+
     . hbcm)+4*(-ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm*p2p3
     . +ffmcfmb*hbcm3-hbcm*p1p2+3*hbcm*p1p4-hbcm*p2p4)))
      b(17)=w5*ccc*(4*p3p6*(ffmcfmb*hbcm+fmc+2*hbcm)+24*hbcm*p4p6)
      b(18)=6*ccc*w5*(ffmcfmb*hbcm-fmc)
      b(19)=2*ccc*w5*(-5*ffmcfmb*hbcm2-3*fmc*hbcm+4*p1p3-8*p3p4)
      b(20)=w5*ccc*(4*p3p7*(-4*ffmcfmb*hbcm+3*hbcm)+4*(-hbcm*p4p7+3*
     . hbcm*p5p7))
      b(21)=ccc*(4*w8*(2*hbcm*p1p11-3*hbcm*p3p11-4*hbcm*p4p11)+4*w1*(
     . 3*ffmcfmb*hbcm*p3p8-3*fmc*p3p8+2*hbcm*p1p8-3*hbcm*p3p8-4*hbcm*
     . p4p8)+4*w5*(-3*ffmcfmb*hbcm*p3p10+3*ffmcfmb*hbcm*p3p8+3*fmc*
     . p3p10-3*fmc*p3p8-2*hbcm*p1p10+2*hbcm*p1p8+3*hbcm*p3p10-3*hbcm*
     . p3p8+4*hbcm*p4p10-4*hbcm*p4p8))
      b(22)=w5*ccc*(4*p3p4*(ffmcfmb*hbcm-fmc-hbcm)+4*p1p3*(ffmcfmb*
     . hbcm-fmc-hbcm)+4*(ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm*p2p3-ffmcfmb*hbcm3+hbcm*p1p2-3*hbcm*p1p4+hbcm*p2p4))
      b(23)=ccc*(w5*(4*p3p6*(-ffmcfmb*hbcm-fmc-2*hbcm)-24*hbcm*p4p6)+
     . w1*(4*p3p6*(-ffmcfmb*hbcm-fmc-2*hbcm)-24*hbcm*p4p6))
      b(24)=ccc*(6*w5*(-ffmcfmb*hbcm+fmc)+6*w1*(-ffmcfmb*hbcm+fmc))
      b(25)=ccc*(2*w5*(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p1p3+8*p3p4)+2*w1
     . *(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p1p3+8*p3p4))
      b(26)=ccc*(w5*(4*p3p7*(4*ffmcfmb*hbcm-3*hbcm)+4*(hbcm*p4p7-3*
     . hbcm*p5p7))+w1*(4*p3p7*(4*ffmcfmb*hbcm-3*hbcm)+4*(hbcm*p4p7-3*
     . hbcm*p5p7)))
      b(27)=w5*ccc*(4*p4p6*(3*ffmcfmb*hbcm-2*fmb-5*fmc)+4*p3p6*(2*
     . ffmcfmb**2*hbcm-2*ffmcfmb*fmb-4*ffmcfmb*fmc+ffmcfmb*hbcm-fmc))
      b(28)=w5*ccc*(4*p4p6*(ffmcfmb*hbcm2+2*fmb*hbcm-3*fmc*hbcm-4*
     . hbcm2)+4*p3p6*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2-fmc*hbcm+fmc2)+16*(ffmcfmb*p2p3*p3p6-
     . 2*ffmcfmb*p3p4*p3p6-p1p2*p3p6-p1p3*p4p6+p1p5*p3p6+p2p3*p4p6-2*
     . p3p4*p4p6))
      b(29)=w5*ccc*(8*(p4p6*p3p7)*(2*fmc+hbcm)+8*(p5p7*p3p6)*(3*
     . ffmcfmb*hbcm-fmc)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*
     . p2p3)*(ffmcfmb*hbcm-fmc)+8*(p6p7*p1p3)*(-ffmcfmb*hbcm+fmc+2*
     . hbcm)+4*p6p7*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*
     . hbcm3+2*fmc*hbcm2+fmc2*hbcm)+8*(p3p7*p3p6)*(-ffmcfmb**2*hbcm+3
     . *ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+8*(-2*hbcm*p1p2*p6p7+2*hbcm*
     . p1p5*p6p7+hbcm*p4p6*p4p7+3*hbcm*p4p6*p5p7))
      ans5=8*p6p7*(ffmcfmb*hbcm2*p2p10-ffmcfmb*hbcm2*p2p8-ffmcfmb*
     . hbcm2*p5p10+ffmcfmb*hbcm2*p5p8-fmc*hbcm*p2p10+fmc*hbcm*p2p8+
     . fmc*hbcm*p5p10-fmc*hbcm*p5p8+4*hbcm2*p1p10-4*hbcm2*p1p8)+8*
     . p3p7*(ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*fmc*hbcm*p6p8+ffmcfmb*
     . hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm*p6p8-
     . fmc2*p6p10+fmc2*p6p8)+8*p3p6*(2*ffmcfmb**2*hbcm2*p7p10-ffmcfmb
     . **2*hbcm2*p7p8-2*ffmcfmb*fmb*hbcm*p7p8+2*ffmcfmb*fmc*hbcm*
     . p7p10+3*ffmcfmb*hbcm2*p7p8+fmc*hbcm*p7p8-fmc2*p7p8)+32*(-
     . ffmcfmb*p2p3*p3p6*p7p8+p1p2*p3p6*p7p8+p1p3*p4p6*p7p8-p1p5*p3p6
     . *p7p8-p2p3*p4p6*p7p8)
      ans4=32*(p4p6*p3p4)*(p7p10+p7p8)+32*(p3p7*p1p5)*(-p6p10+p6p8)+
     . 32*(p5p7*p1p3)*(-p6p10+p6p8)+32*(p3p7*p1p2)*(p6p10-p6p8)+16*(
     . p4p6*p3p7)*(2*p1p10-2*p1p8-4*p2p10+4*p2p8+5*p3p10-5*p3p8+4*
     . p4p10-4*p4p8+2*p5p10-2*p5p8)+32*(p5p7*p3p6)*(p1p10-p1p8)+32*(
     . p4p7*p3p6)*(p1p10-p1p8)+32*(p6p7*p3p5)*(p1p10-p1p8)+32*(p6p7*
     . p3p4)*(p1p10-p1p8)+32*(p6p7*p2p3)*(-p1p10+p1p8)+32*(p6p7*p1p3)
     . *(-p1p10+p1p8-p2p10+p2p8+p5p10-p5p8)+32*(p3p6*p3p4)*(ffmcfmb*
     . p7p10+ffmcfmb*p7p8)+32*(p3p7*p3p6)*(-2*ffmcfmb*p2p10+2*ffmcfmb
     . *p2p8+2*ffmcfmb*p3p10-2*ffmcfmb*p3p8+2*ffmcfmb*p4p10-2*ffmcfmb
     . *p4p8+ffmcfmb*p5p10-ffmcfmb*p5p8+2*p1p10-2*p1p8)+32*(p3p7*p1p3
     . )*(ffmcfmb*p6p10-ffmcfmb*p6p8-2*p6p10+2*p6p8)+8*p5p7*(ffmcfmb*
     . hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm*p6p8)+8
     . *p4p6*(2*ffmcfmb*hbcm2*p7p10-3*ffmcfmb*hbcm2*p7p8-2*fmb*hbcm*
     . p7p8+2*fmc*hbcm*p7p10+fmc*hbcm*p7p8+4*hbcm2*p7p8)+ans5
      ans3=w5*ans4
      ans6=(8*p3p4*(p6p12+2*p7p13)+8*p1p3*(2*p6p12-p7p13)+8*p3p7*(-2*
     . p1p13-4*p2p13+5*p3p13+4*p4p13+2*p5p13+2*p6p10)+8*p3p6*(-2*
     . ffmcfmb*p3p12-3*p1p12+p3p12+p4p12+p5p12-p7p11)+4*(-ffmcfmb*
     . hbcm2*p6p12+2*ffmcfmb*hbcm2*p7p13-fmb*hbcm*p6p12+fmb*hbcm*
     . p7p13+2*fmc*hbcm*p6p12+fmc*hbcm*p7p13+hbcm2*p6p12+hbcm2*p7p13-
     . 2*p2p3*p6p12-4*p3p10*p6p7-4*p3p12*p4p6+2*p3p5*p7p13))
      ans2=w8*(16*(p4p6*p3p7)*(-2*p1p11+4*p2p11-5*p3p11-4*p4p11-2*
     . p5p11)+32*(p6p7*p1p3)*(p1p11+p2p11-p5p11)+32*(p3p7*p3p6)*(2*
     . ffmcfmb*p2p11-2*ffmcfmb*p3p11-2*ffmcfmb*p4p11-ffmcfmb*p5p11-2*
     . p1p11)+32*(p3p7*p1p3)*(-ffmcfmb*p6p11+2*p6p11)+8*p5p7*(-
     . ffmcfmb*hbcm2*p6p11+fmc*hbcm*p6p11)+16*p4p6*(-ffmcfmb*hbcm2*
     . p7p11-fmc*hbcm*p7p11)+8*p6p7*(-ffmcfmb*hbcm2*p2p11+ffmcfmb*
     . hbcm2*p5p11+fmc*hbcm*p2p11-fmc*hbcm*p5p11-4*hbcm2*p1p11)+8*
     . p3p7*(-ffmcfmb*fmc*hbcm*p6p11-ffmcfmb*hbcm2*p6p11+fmc*hbcm*
     . p6p11+fmc2*p6p11)+16*p3p6*(-ffmcfmb**2*hbcm2*p7p11-ffmcfmb*fmc
     . *hbcm*p7p11)+32*(-ffmcfmb*p3p4*p3p6*p7p11+p1p11*p2p3*p6p7-
     . p1p11*p3p4*p6p7-p1p11*p3p5*p6p7-p1p11*p3p6*p4p7-p1p11*p3p6*
     . p5p7-p1p2*p3p7*p6p11+p1p3*p5p7*p6p11+p1p5*p3p7*p6p11-p3p4*p4p6
     . *p7p11))+ans3+ans6
      ans1=w1*(16*(p4p6*p3p7)*(-2*p1p8+4*p2p8-5*p3p8-4*p4p8-2*p5p8)+
     . 32*(p6p7*p1p3)*(p1p8+p2p8-p5p8)+32*(p3p7*p3p6)*(2*ffmcfmb*p2p8
     . -2*ffmcfmb*p3p8-2*ffmcfmb*p4p8-ffmcfmb*p5p8-2*p1p8)+32*(p3p7*
     . p1p3)*(-ffmcfmb*p6p8+2*p6p8)+8*p5p7*(-ffmcfmb*hbcm2*p6p8+fmc*
     . hbcm*p6p8)+8*p4p6*(-3*ffmcfmb*hbcm2*p7p8-2*fmb*hbcm*p7p8+fmc*
     . hbcm*p7p8+4*hbcm2*p7p8)+8*p6p7*(-ffmcfmb*hbcm2*p2p8+ffmcfmb*
     . hbcm2*p5p8+fmc*hbcm*p2p8-fmc*hbcm*p5p8-4*hbcm2*p1p8)+8*p3p7*(-
     . ffmcfmb*fmc*hbcm*p6p8-ffmcfmb*hbcm2*p6p8+fmc*hbcm*p6p8+fmc2*
     . p6p8)+8*p3p6*(-ffmcfmb**2*hbcm2*p7p8-2*ffmcfmb*fmb*hbcm*p7p8+3
     . *ffmcfmb*hbcm2*p7p8+fmc*hbcm*p7p8-fmc2*p7p8)+32*(-ffmcfmb*p2p3
     . *p3p6*p7p8+ffmcfmb*p3p4*p3p6*p7p8+p1p2*p3p6*p7p8-p1p2*p3p7*
     . p6p8+p1p3*p4p6*p7p8+p1p3*p5p7*p6p8-p1p5*p3p6*p7p8+p1p5*p3p7*
     . p6p8+p1p8*p2p3*p6p7-p1p8*p3p4*p6p7-p1p8*p3p5*p6p7-p1p8*p3p6*
     . p4p7-p1p8*p3p6*p5p7-p2p3*p4p6*p7p8+p3p4*p4p6*p7p8))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w5*(4*p4p6*(-3*ffmcfmb*hbcm+2*fmb+5*fmc)+4*p3p6*(-2*
     . ffmcfmb**2*hbcm+2*ffmcfmb*fmb+4*ffmcfmb*fmc-ffmcfmb*hbcm+fmc))
     . +w1*(4*p4p6*(-3*ffmcfmb*hbcm+2*fmb+5*fmc)+4*p3p6*(-2*ffmcfmb**
     . 2*hbcm+2*ffmcfmb*fmb+4*ffmcfmb*fmc-ffmcfmb*hbcm+fmc)))
      ans2=w5*(8*p4p7*(ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p8-fmc*p6p10
     . +fmc*p6p8)+16*p4p6*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p8-fmc*
     . p7p10+fmc*p7p8)+8*p6p7*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p8-
     . ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p5p10-
     . ffmcfmb*hbcm*p5p8-fmc*p1p10+fmc*p1p8+fmc*p2p10-fmc*p2p8-fmc*
     . p5p10+fmc*p5p8)+8*p3p7*(ffmcfmb**2*hbcm*p6p10-ffmcfmb**2*hbcm*
     . p6p8-ffmcfmb*fmc*p6p10+ffmcfmb*fmc*p6p8)+16*p3p6*(ffmcfmb**2*
     . hbcm*p7p10-ffmcfmb**2*hbcm*p7p8-ffmcfmb*fmc*p7p10+ffmcfmb*fmc*
     . p7p8))+4*(ffmcfmb*hbcm*p6p12+2*ffmcfmb*hbcm*p7p13-fmc*p6p12-2*
     . fmc*p7p13)
      ans1=w1*(8*p4p7*(-ffmcfmb*hbcm*p6p8+fmc*p6p8)+16*p4p6*(-ffmcfmb
     . *hbcm*p7p8+fmc*p7p8)+8*p6p7*(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*
     . p2p8-ffmcfmb*hbcm*p5p8+fmc*p1p8-fmc*p2p8+fmc*p5p8)+8*p3p7*(-
     . ffmcfmb**2*hbcm*p6p8+ffmcfmb*fmc*p6p8)+16*p3p6*(-ffmcfmb**2*
     . hbcm*p7p8+ffmcfmb*fmc*p7p8))+w8*(8*p4p7*(-ffmcfmb*hbcm*p6p11+
     . fmc*p6p11)+16*p4p6*(-ffmcfmb*hbcm*p7p11+fmc*p7p11)+8*p6p7*(-
     . ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p5p11+fmc*
     . p1p11-fmc*p2p11+fmc*p5p11)+8*p3p7*(-ffmcfmb**2*hbcm*p6p11+
     . ffmcfmb*fmc*p6p11)+16*p3p6*(-ffmcfmb**2*hbcm*p7p11+ffmcfmb*fmc
     . *p7p11))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(39)=w8*ccc*(4*p6p7*(-3*ffmcfmb*hbcm2-fmc*hbcm+4*hbcm2)+16*(2*
     . ffmcfmb*p3p6*p3p7-p2p3*p6p7+p3p5*p6p7+p3p6*p4p7+p3p7*p4p6))
      b(40)=w8*ccc*(8*(p4p6*p3p7)*(-2*fmc-hbcm)+8*(p5p7*p3p6)*(-3*
     . ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p6p7*
     . p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p1p3)*(ffmcfmb*hbcm-fmc-2*
     . hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmc*hbcm2+2*ffmcfmb*
     . hbcm3-2*fmc*hbcm2-fmc2*hbcm)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm-3*
     . ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+8*(2*hbcm*p1p2*p6p7-2*hbcm*p1p5*
     . p6p7-hbcm*p4p6*p4p7-3*hbcm*p4p6*p5p7))
      b(41)=w8*ccc*(4*p6p7*(ffmcfmb**2*hbcm2-fmc2)+8*(-2*ffmcfmb*p3p6
     . *p3p7-4*ffmcfmb*p3p6*p5p7+2*p1p2*p6p7-2*p1p3*p6p7-2*p1p5*p6p7-
     . 3*p3p7*p4p6-p4p6*p4p7-5*p4p6*p5p7))
      b(43)=ccc*(w1*(16*p6p7*(-p2p8+p5p8)+16*(ffmcfmb*p3p6*p7p8+
     . ffmcfmb*p3p7*p6p8+p4p6*p7p8+p4p7*p6p8))+w8*(16*p6p7*(-p2p11+
     . p5p11)+16*(ffmcfmb*p3p6*p7p11+ffmcfmb*p3p7*p6p11+p4p6*p7p11+
     . p4p7*p6p11))+w5*(16*p4p7*(-p6p10+p6p8)+16*p4p6*(-p7p10+p7p8)+
     . 16*p6p7*(p2p10-p2p8-p5p10+p5p8)+16*p3p7*(-ffmcfmb*p6p10+
     . ffmcfmb*p6p8)+16*p3p6*(-ffmcfmb*p7p10+ffmcfmb*p7p8))+4*(-p6p12
     . -3*p7p13))
      b(44)=w8*ccc*(4*p4p6*(-ffmcfmb*hbcm2-2*fmb*hbcm+3*fmc*hbcm+4*
     . hbcm2)+4*p3p6*(ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*
     . fmc*hbcm+3*ffmcfmb*hbcm2+fmc*hbcm-fmc2)+16*(-ffmcfmb*p2p3*p3p6
     . +2*ffmcfmb*p3p4*p3p6+p1p2*p3p6+p1p3*p4p6-p1p5*p3p6-p2p3*p4p6+2
     . *p3p4*p4p6))
      b(45)=w8*ccc*(p3p6*(16*ffmcfmb+8)+32*p4p6)
      b(46)=w8*ccc*(4*p4p6*(3*ffmcfmb*hbcm-2*fmb-5*fmc)+4*p3p6*(2*
     . ffmcfmb**2*hbcm-2*ffmcfmb*fmb-4*ffmcfmb*fmc+ffmcfmb*hbcm-fmc))
      b(49)=w8*ccc*(4*p3p7*(4*ffmcfmb*hbcm-3*hbcm)+4*(hbcm*p4p7-3*
     . hbcm*p5p7))
      b(50)=w8*ccc*(p3p7*(16*ffmcfmb-12)+4*(p4p7-3*p5p7))
      b(51)=w8*ccc*((p3p7*p3p4)*(-8*ffmcfmb+8)+(p3p7*p1p3)*(-8*
     . ffmcfmb+8)+4*p4p7*(ffmcfmb*hbcm2+fmc*hbcm-2*hbcm2)+4*p3p7*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+8*(ffmcfmb*hbcm2*p5p7+
     . ffmcfmb*p2p3*p3p7-ffmcfmb*p3p5*p3p7+p1p3*p5p7+p2p3*p4p7+p3p4*
     . p5p7-p3p5*p4p7))
      b(53)=ccc*(4*w1*(2*p1p8-3*p3p8-4*p4p8)+4*w8*(2*p1p11-3*p3p11-4*
     . p4p11)+4*w5*(-2*p1p10+2*p1p8+3*p3p10-3*p3p8+4*p4p10-4*p4p8))
      b(54)=2*ccc*w8*(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p1p3+8*p3p4)
      b(55)=6*ccc*w8*(ffmcfmb*hbcm-fmc)
      b(56)=ccc*(4*w8*(-ffmcfmb*hbcm2+ffmcfmb*p2p3+p1p2-p1p3-3*p1p4+
     . p2p4-p3p4)-2)
      b(57)=-8*ccc
      b(58)=w8*ccc*(4*p3p6*(-ffmcfmb*hbcm-fmc-2*hbcm)-24*hbcm*p4p6)
      b(59)=ccc*(w8*(4*p3p4*(-ffmcfmb*hbcm+fmc+hbcm)+4*p1p3*(-ffmcfmb
     . *hbcm+fmc+hbcm)+4*(-ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-ffmcfmb
     . *hbcm*p2p3+ffmcfmb*hbcm3-hbcm*p1p2+3*hbcm*p1p4-hbcm*p2p4))+2*
     . hbcm)
      b(61)=-8*ccc*hbcm
      b(62)=8*ccc*(ffmcfmb*hbcm*p3p6+hbcm*p4p6)
      b(63)=w5*ccc*(4*p6p7*(3*ffmcfmb*hbcm2+fmc*hbcm-4*hbcm2)+16*(-2*
     . ffmcfmb*p3p6*p3p7+p2p3*p6p7-p3p5*p6p7-p3p6*p4p7-p3p7*p4p6))
      b(64)=w5*ccc*(4*p6p7*(-ffmcfmb**2*hbcm2+fmc2)+8*(2*ffmcfmb*p3p6
     . *p3p7+4*ffmcfmb*p3p6*p5p7-2*p1p2*p6p7+2*p1p3*p6p7+2*p1p5*p6p7+
     . 3*p3p7*p4p6+p4p6*p4p7+5*p4p6*p5p7))
      b(65)=ccc*(w5*(4*p6p7*(-3*ffmcfmb*hbcm2-fmc*hbcm+4*hbcm2)+16*(2
     . *ffmcfmb*p3p6*p3p7-p2p3*p6p7+p3p5*p6p7+p3p6*p4p7+p3p7*p4p6))+
     . w1*(4*p6p7*(-3*ffmcfmb*hbcm2-fmc*hbcm+4*hbcm2)+16*(2*ffmcfmb*
     . p3p6*p3p7-p2p3*p6p7+p3p5*p6p7+p3p6*p4p7+p3p7*p4p6)))
      b(66)=ccc*(w1*(16*p6p7*(-hbcm*p2p8+hbcm*p5p8)+8*p3p7*(ffmcfmb*
     . hbcm*p6p8+fmc*p6p8)+16*(ffmcfmb*hbcm*p3p6*p7p8+hbcm*p4p6*p7p8+
     . hbcm*p4p7*p6p8))+w8*(16*p6p7*(-hbcm*p2p11+hbcm*p5p11)+8*p3p7*(
     . ffmcfmb*hbcm*p6p11+fmc*p6p11)+16*(ffmcfmb*hbcm*p3p6*p7p11+hbcm
     . *p4p6*p7p11+hbcm*p4p7*p6p11))+w5*(16*p4p7*(-hbcm*p6p10+hbcm*
     . p6p8)+16*p4p6*(-hbcm*p7p10+hbcm*p7p8)+16*p6p7*(hbcm*p2p10-hbcm
     . *p2p8-hbcm*p5p10+hbcm*p5p8)+8*p3p7*(-ffmcfmb*hbcm*p6p10+
     . ffmcfmb*hbcm*p6p8-fmc*p6p10+fmc*p6p8)+16*p3p6*(-ffmcfmb*hbcm*
     . p7p10+ffmcfmb*hbcm*p7p8))+4*(-hbcm*p6p12-3*hbcm*p7p13))
      b(67)=w5*ccc*(p3p6*(-16*ffmcfmb-8)-32*p4p6)
      b(68)=ccc*(w5*(p3p6*(16*ffmcfmb+8)+32*p4p6)+w1*(p3p6*(16*
     . ffmcfmb+8)+32*p4p6))
      b(69)=w5*ccc*(p3p7*(-16*ffmcfmb+12)+4*(-p4p7+3*p5p7))
      b(70)=w5*ccc*((p3p7*p3p4)*(8*ffmcfmb-8)+(p3p7*p1p3)*(8*ffmcfmb-
     . 8)+4*p4p7*(-ffmcfmb*hbcm2-fmc*hbcm+2*hbcm2)+4*p3p7*(ffmcfmb**2
     . *hbcm2-ffmcfmb*fmc*hbcm)+8*(-ffmcfmb*hbcm2*p5p7-ffmcfmb*p2p3*
     . p3p7+ffmcfmb*p3p5*p3p7-p1p3*p5p7-p2p3*p4p7-p3p4*p5p7+p3p5*p4p7
     . ))
      b(71)=ccc*(w5*(p3p7*(16*ffmcfmb-12)+4*(p4p7-3*p5p7))+w1*(p3p7*(
     . 16*ffmcfmb-12)+4*(p4p7-3*p5p7)))
      b(72)=4*ccc*w5*(ffmcfmb*hbcm2-ffmcfmb*p2p3-p1p2+p1p3+3*p1p4-
     . p2p4+p3p4)
      b(73)=4*ccc*w8*p6p7*(ffmcfmb*hbcm-fmc)
      b(74)=ccc*(4*w5*p6p7*(ffmcfmb*hbcm-fmc)+4*w1*p6p7*(ffmcfmb*hbcm
     . -fmc))
      b(75)=ccc*(4*w1*(ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p4p8-fmc*p1p8-
     . fmc*p4p8)+4*w8*(ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p4p11-fmc*
     . p1p11-fmc*p4p11)+4*w5*(-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p1p8-
     . ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*p4p8+fmc*p1p10-fmc*p1p8+fmc*
     . p4p10-fmc*p4p8))
      ans=ccc*(w1*(8*p3p5*(p1p8+p4p8)+8*p3p4*(-p1p8+p3p8+p4p8)+8*p2p3
     . *(-p1p8-p4p8)+8*p1p3*(-p1p8+p3p8+p4p8)+4*(-3*ffmcfmb*hbcm2*
     . p1p8+2*ffmcfmb*hbcm2*p3p8+ffmcfmb*hbcm2*p4p8-fmc*hbcm*p1p8-fmc
     . *hbcm*p4p8+2*hbcm2*p1p8+2*hbcm2*p4p8))+w8*(8*p3p5*(p1p11+p4p11
     . )+8*p3p4*(-p1p11+p3p11+p4p11)+8*p2p3*(-p1p11-p4p11)+8*p1p3*(-
     . p1p11+p3p11+p4p11)+4*(-3*ffmcfmb*hbcm2*p1p11+2*ffmcfmb*hbcm2*
     . p3p11+ffmcfmb*hbcm2*p4p11-fmc*hbcm*p1p11-fmc*hbcm*p4p11+2*
     . hbcm2*p1p11+2*hbcm2*p4p11))+w5*(8*p3p5*(-p1p10+p1p8-p4p10+p4p8
     . )+8*p3p4*(p1p10-p1p8-p3p10+p3p8-p4p10+p4p8)+8*p2p3*(p1p10-p1p8
     . +p4p10-p4p8)+8*p1p3*(p1p10-p1p8-p3p10+p3p8-p4p10+p4p8)+4*(3*
     . ffmcfmb*hbcm2*p1p10-3*ffmcfmb*hbcm2*p1p8-2*ffmcfmb*hbcm2*p3p10
     . +2*ffmcfmb*hbcm2*p3p8-ffmcfmb*hbcm2*p4p10+ffmcfmb*hbcm2*p4p8+
     . fmc*hbcm*p1p10-fmc*hbcm*p1p8+fmc*hbcm*p4p10-fmc*hbcm*p4p8-2*
     . hbcm2*p1p10+2*hbcm2*p1p8-2*hbcm2*p4p10+2*hbcm2*p4p8)))
      b(76)=ans
      b(77)=ccc*(w1*(16*p3p6*(p2p8-p5p8)+4*(-3*ffmcfmb*hbcm2*p6p8-fmc
     . *hbcm*p6p8+4*hbcm2*p6p8-4*p2p3*p6p8+4*p3p5*p6p8))+w8*(16*p3p6*
     . (p2p11-p5p11)+4*(-3*ffmcfmb*hbcm2*p6p11-fmc*hbcm*p6p11+4*hbcm2
     . *p6p11-4*p2p3*p6p11+4*p3p5*p6p11))+w5*(16*p3p5*(-p6p10+p6p8)+
     . 16*p2p3*(p6p10-p6p8)+16*p3p6*(-p2p10+p2p8+p5p10-p5p8)+4*(3*
     . ffmcfmb*hbcm2*p6p10-3*ffmcfmb*hbcm2*p6p8+fmc*hbcm*p6p10-fmc*
     . hbcm*p6p8-4*hbcm2*p6p10+4*hbcm2*p6p8))+4*p3p13)
      b(78)=ccc*(4*w1*(ffmcfmb*hbcm*p6p8-fmc*p6p8)+4*w8*(ffmcfmb*hbcm
     . *p6p11-fmc*p6p11)+4*w5*(-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p8+
     . fmc*p6p10-fmc*p6p8))
      b(81)=4*ccc*w5*p6p7*(-ffmcfmb*hbcm+fmc)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(4*p3p7*(2*fmc+hbcm)+4*(hbcm*p4p7+3*hbcm*p5p7))
      b(84)=4*ccc*(3*p3p7+p4p7+5*p5p7)
      b(85)=4*ccc*(fmb*hbcm-fmc*hbcm-3*hbcm2+p1p3+3*p2p3-5*p3p4-p3p5)
      b(86)=-8*ccc
      b(87)=4*ccc*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(88)=w5*ccc*(4*p4p7*(ffmcfmb*hbcm-fmc)+4*p3p7*(ffmcfmb**2*hbcm
     . -ffmcfmb*fmc))
      b(89)=-4*ccc*hbcm
      b(90)=w8*ccc*(4*p4p7*(-ffmcfmb*hbcm+fmc)+4*p3p7*(-ffmcfmb**2*
     . hbcm+ffmcfmb*fmc))
      b(91)=ccc*(w5*(4*p4p7*(-ffmcfmb*hbcm+fmc)+4*p3p7*(-ffmcfmb**2*
     . hbcm+ffmcfmb*fmc))+w1*(4*p4p7*(-ffmcfmb*hbcm+fmc)+4*p3p7*(-
     . ffmcfmb**2*hbcm+ffmcfmb*fmc)))
      b(92)=2*ccc*(ffmcfmb*hbcm2-fmc*hbcm+2*hbcm2+2*p1p3-2*p2p3+2*
     . p3p4+2*p3p5)
      b(93)=8*ccc*p3p6
      b(94)=2*ccc*(-ffmcfmb*hbcm+fmc)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp8_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(89) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,89 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)
     . )
      b(2)=w2*ccc*(8*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc+
     . fmc2)+16*(p1p3*p6p7+2*p1p5*p6p7))
      b(3)=ccc*(8*w2*(ffmcfmb*fmb*hbcm*p7p9-ffmcfmb*fmc*hbcm*p7p9-fmb
     . *fmc*p7p9+fmc2*p7p9+4*p1p2*p7p9-2*p1p3*p7p9-4*p1p4*p7p9)+8*w12
     . *(ffmcfmb*fmb*hbcm*p7p10-ffmcfmb*fmc*hbcm*p7p10-fmb*fmc*p7p10+
     . fmc2*p7p10+4*p1p2*p7p10-2*p1p3*p7p10-4*p1p4*p7p10))
      b(5)=w2*ccc*(8*p3p7*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc
     . -fmc2)+16*(-2*p1p2*p3p7-p1p3*p4p7-p1p3*p5p7+2*p1p4*p3p7))
      b(6)=ccc*(w2*(16*p3p7*(-ffmcfmb*hbcm*p5p9-fmb*p1p9-fmc*p1p9+fmc
     . *p5p9)+16*p3p4*(-ffmcfmb*hbcm*p7p9+fmc*p7p9)+16*p2p3*(ffmcfmb*
     . hbcm*p7p9-fmc*p7p9)+16*p1p3*(ffmcfmb*hbcm*p7p9+fmb*p7p9)+8*(-
     . ffmcfmb*hbcm3*p7p9+fmc*hbcm2*p7p9-2*hbcm*p1p2*p7p9+2*hbcm*p1p4
     . *p7p9-2*hbcm*p1p9*p4p7-2*hbcm*p1p9*p5p7))+w12*(16*p3p7*(-
     . ffmcfmb*hbcm*p5p10-fmb*p1p10-fmc*p1p10+fmc*p5p10)+16*p3p4*(-
     . ffmcfmb*hbcm*p7p10+fmc*p7p10)+16*p2p3*(ffmcfmb*hbcm*p7p10-fmc*
     . p7p10)+16*p1p3*(ffmcfmb*hbcm*p7p10+fmb*p7p10)+8*(-ffmcfmb*
     . hbcm3*p7p10+fmc*hbcm2*p7p10-2*hbcm*p1p10*p4p7-2*hbcm*p1p10*
     . p5p7-2*hbcm*p1p2*p7p10+2*hbcm*p1p4*p7p10)))
      b(7)=ccc*(8*w2*(ffmcfmb*fmb*hbcm*p6p9-ffmcfmb*fmc*hbcm*p6p9-
     . ffmcfmb*hbcm2*p6p9-fmb*fmc*p6p9+fmc*hbcm*p6p9+fmc2*p6p9+2*p1p3
     . *p6p9+2*p1p5*p6p9-2*p1p9*p3p6-2*p1p9*p4p6)+8*w12*(ffmcfmb*fmb*
     . hbcm*p6p10-ffmcfmb*fmc*hbcm*p6p10-ffmcfmb*hbcm2*p6p10-fmb*fmc*
     . p6p10+fmc*hbcm*p6p10+fmc2*p6p10-2*p1p10*p3p6-2*p1p10*p4p6+2*
     . p1p3*p6p10+2*p1p5*p6p10)+8*(-p1p13-p2p13+2*p3p13+p4p13+p5p13))
      b(9)=w2*ccc*(8*p3p6*(fmb*fmc-fmc2)+16*(p1p3*p4p6-p1p5*p3p6))
      b(10)=ccc*(w2*(16*p3p6*(fmc*p1p9+fmc*p2p9-fmc*p3p9-fmc*p4p9)+16
     . *p3p5*(ffmcfmb*hbcm*p6p9-fmc*p6p9)+16*p1p3*(-ffmcfmb*hbcm*p6p9
     . -fmb*p6p9)+8*(ffmcfmb*fmb*hbcm2*p6p9+ffmcfmb*fmc*hbcm2*p6p9+
     . ffmcfmb*hbcm3*p6p9-fmb*fmc*hbcm*p6p9-fmc*hbcm2*p6p9-fmc2*hbcm*
     . p6p9-2*hbcm*p1p5*p6p9-2*hbcm*p1p9*p4p6))+w12*(16*p3p6*(fmc*
     . p1p10+fmc*p2p10-fmc*p4p10)+16*p3p5*(ffmcfmb*hbcm*p6p10-fmc*
     . p6p10)+16*p4p6*(-ffmcfmb*hbcm*p3p10+fmc*p3p10-hbcm*p1p10)+16*
     . p1p3*(-ffmcfmb*hbcm*p6p10-fmb*p6p10)+8*(ffmcfmb*fmb*hbcm2*
     . p6p10+ffmcfmb*fmc*hbcm2*p6p10+ffmcfmb*hbcm3*p6p10-fmb*fmc*hbcm
     . *p6p10-fmc*hbcm2*p6p10-fmc2*hbcm*p6p10-2*hbcm*p1p5*p6p10))+8*(
     . -fmb*p3p13-fmc*p3p13+hbcm*p2p13-hbcm*p3p13-hbcm*p4p13))
      b(12)=w2*ccc*(16*(p6p7*p1p3)*(-fmb-fmc+hbcm)+8*p6p7*(-ffmcfmb*
     . hbcm3+fmc*hbcm2)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*
     . p3p5)*(-ffmcfmb*hbcm+fmc)+16*(-fmc*p3p6*p4p7-fmc*p3p6*p5p7+
     . hbcm*p1p5*p6p7))
      b(14)=4*ccc*w2*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-fmc2
     . -4*p1p2+2*p1p3+4*p1p4)
      b(16)=w2*ccc*(8*p3p4*(ffmcfmb*hbcm-fmc)+8*p2p3*(-ffmcfmb*hbcm+
     . fmc)+8*p1p3*(-ffmcfmb*hbcm-fmb)+4*(ffmcfmb*hbcm3-fmc*hbcm2+2*
     . hbcm*p1p2-2*hbcm*p1p4))
      b(17)=8*ccc*w12*(fmb*p3p6+hbcm*p4p6)
      b(18)=4*ccc*w12*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      b(19)=4*ccc*w12*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p1p3+4*p2p3-4*p3p4
     . )
      b(20)=w12*ccc*(8*p3p7*(ffmcfmb*hbcm+2*fmb+fmc)+8*(hbcm*p4p7+
     . hbcm*p5p7))
      b(21)=ccc*(8*w2*(-hbcm*p2p9+hbcm*p3p9+hbcm*p4p9)+8*w12*(ffmcfmb
     . *hbcm*p3p10+2*fmb*p3p10+fmc*p3p10-hbcm*p2p10+hbcm*p4p10))
      b(22)=w12*ccc*(8*p3p4*(ffmcfmb*hbcm-fmc)+8*p2p3*(-ffmcfmb*hbcm+
     . fmc)+8*p1p3*(-ffmcfmb*hbcm-fmb)+4*(ffmcfmb*hbcm3-fmc*hbcm2+2*
     . hbcm*p1p2-2*hbcm*p1p4))
      b(27)=w12*ccc*(8*p4p6*(ffmcfmb*hbcm-fmc)-8*fmc*p3p6)
      b(28)=w12*ccc*(8*p3p6*(-fmb*fmc+fmc2)+16*(-p1p3*p4p6+p1p5*p3p6)
     . )
      b(29)=w12*ccc*(8*p6p7*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p4p6*p3p7)
     . *(-ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+16*(
     . p6p7*p1p3)*(ffmcfmb*hbcm+fmb)+16*(-fmc*p3p6*p4p7-fmc*p3p6*p5p7
     . +hbcm*p1p5*p6p7))
      ans=ccc*(w2*(32*(p6p7*p1p3)*(-2*p1p9+p5p9)+16*p6p7*(fmb*hbcm*
     . p1p9-fmc*hbcm*p1p9+3*hbcm2*p1p9)+16*p5p7*(ffmcfmb*hbcm2*p6p9-
     . fmc*hbcm*p6p9)+16*p4p7*(ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+16*
     . p3p7*(ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+32*(-p1p3*p3p7*p6p9-
     . p1p3*p4p7*p6p9-p1p3*p5p7*p6p9-p1p9*p2p3*p6p7+p1p9*p3p4*p6p7+2*
     . p1p9*p3p5*p6p7+2*p1p9*p3p6*p3p7+p1p9*p3p6*p4p7+p1p9*p3p6*p5p7+
     . 2*p1p9*p3p7*p4p6))+w12*(32*(p6p7*p1p3)*(-2*p1p10+p5p10)+16*
     . p6p7*(fmb*hbcm*p1p10-fmc*hbcm*p1p10+3*hbcm2*p1p10)+16*p3p6*(
     . fmb*fmc*p7p10-fmc2*p7p10)+16*p5p7*(ffmcfmb*hbcm2*p6p10-fmc*
     . hbcm*p6p10)+16*p4p7*(ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+16*
     . p3p7*(ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+32*(-p1p10*p2p3*p6p7
     . +p1p10*p3p4*p6p7+2*p1p10*p3p5*p6p7+2*p1p10*p3p6*p3p7+p1p10*
     . p3p6*p4p7+p1p10*p3p6*p5p7+2*p1p10*p3p7*p4p6-p1p3*p3p7*p6p10+
     . p1p3*p4p6*p7p10-p1p3*p4p7*p6p10-p1p3*p5p7*p6p10-p1p5*p3p6*
     . p7p10))+(16*p3p7*(p1p13+p2p13-p3p13-p4p13-p5p13)+8*(fmb*hbcm*
     . p7p13-fmc*hbcm*p7p13-2*hbcm2*p7p13+2*p1p3*p7p13+2*p2p3*p7p13-2
     . *p3p4*p7p13-2*p3p5*p7p13)))
      b(30)=ans
      b(32)=w2*ccc*(8*p4p6*(-ffmcfmb*hbcm+fmc)+8*fmc*p3p6)
      b(33)=ccc*(w2*(16*p6p7*(ffmcfmb*hbcm*p5p9+fmb*p1p9+fmc*p1p9-fmc
     . *p5p9-hbcm*p1p9)+16*p4p6*(ffmcfmb*hbcm*p7p9-fmc*p7p9)-16*fmc*
     . p3p6*p7p9)+w12*(16*p4p6*(ffmcfmb*hbcm*p7p10-fmc*p7p10)+16*p6p7
     . *(-ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p5p10-fmb*p1p10-fmc*p5p10)-
     . 16*fmc*p3p6*p7p10)-8*hbcm*p7p13)
      b(34)=4*ccc*w2*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(35)=ccc*(8*w2*(-ffmcfmb*hbcm*p7p9-2*fmb*p7p9-fmc*p7p9+hbcm*
     . p7p9)+8*w12*(-ffmcfmb*hbcm*p7p10-2*fmb*p7p10-fmc*p7p10+hbcm*
     . p7p10))
      b(36)=4*ccc*w2*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p1p3+4*p2p3-4*p3p4)
      b(37)=w2*ccc*(8*p3p7*(ffmcfmb*hbcm+2*fmb+fmc)+8*(hbcm*p4p7+hbcm
     . *p5p7))
      b(38)=ccc*(w2*(16*p3p7*(-p1p9+2*p5p9)+8*(-fmb*hbcm*p7p9+fmc*
     . hbcm*p7p9+2*hbcm2*p7p9-2*p1p3*p7p9-4*p2p3*p7p9+4*p3p4*p7p9))+
     . w12*(16*p3p7*(-p1p10+2*p5p10)+8*(-fmb*hbcm*p7p10+fmc*hbcm*
     . p7p10+2*hbcm2*p7p10-2*p1p3*p7p10-4*p2p3*p7p10+4*p3p4*p7p10)))
      b(42)=w2*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*p6p7
     . -2*p3p5*p6p7-p3p6*p3p7-2*p3p7*p4p6))
      b(43)=ccc*(w2*(16*p6p7*(p1p9-2*p5p9)+16*(-p3p6*p7p9-2*p4p6*p7p9
     . ))+w12*(16*p6p7*(p1p10-2*p5p10)+16*(-p3p6*p7p10-2*p4p6*p7p10))
     . )
      b(47)=8*ccc*w2*(-p3p6-2*p4p6)
      b(52)=16*ccc*w2*(p3p7+p4p7+p5p7)
      b(53)=ccc*(8*w2*(-p1p9-2*p2p9+2*p3p9+2*p4p9)+8*w12*(-p1p10-2*
     . p2p10+2*p3p10+2*p4p10))
      b(60)=8*ccc*w2*(fmb*p3p6+hbcm*p4p6)
      b(63)=w12*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*
     . p6p7-2*p3p5*p6p7-p3p6*p3p7-2*p3p7*p4p6))
      b(64)=w12*ccc*(8*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc+fmc2)+16*(p1p3*p6p7+2*p1p5*p6p7))
      b(66)=ccc*(w2*(16*p6p7*(hbcm*p1p9-hbcm*p5p9)+16*(-fmb*p3p6*p7p9
     . -hbcm*p4p6*p7p9))+w12*(16*p6p7*(hbcm*p1p10-hbcm*p5p10)+16*(-
     . fmb*p3p6*p7p10-hbcm*p4p6*p7p10))+8*hbcm*p7p13)
      b(67)=8*ccc*w12*(-p3p6-2*p4p6)
      b(69)=16*ccc*w12*(p3p7+p4p7+p5p7)
      b(70)=w12*ccc*(8*p3p7*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . fmc-fmc2)+16*(-2*p1p2*p3p7-p1p3*p4p7-p1p3*p5p7+2*p1p4*p3p7))
      b(72)=4*ccc*w12*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-
     . fmc2-4*p1p2+2*p1p3+4*p1p4)
      b(75)=ccc*(8*w2*(ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*p2p9-ffmcfmb*
     . hbcm*p3p9-ffmcfmb*hbcm*p4p9+fmb*p1p9-fmc*p2p9+fmc*p3p9+fmc*
     . p4p9-hbcm*p1p9)+8*w12*(ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p2p10-
     . ffmcfmb*hbcm*p3p10-ffmcfmb*hbcm*p4p10+fmb*p1p10-fmc*p2p10+fmc*
     . p3p10+fmc*p4p10-hbcm*p1p10))
      b(76)=ccc*(w2*(16*p1p3*(p1p9+p2p9-p3p9-p4p9)+8*(-fmb*hbcm*p1p9+
     . fmc*hbcm*p1p9-hbcm2*p1p9-2*p1p9*p3p5))+w12*(16*p1p3*(p1p10+
     . p2p10-p3p10-p4p10)+8*(-fmb*hbcm*p1p10+fmc*hbcm*p1p10-hbcm2*
     . p1p10-2*p1p10*p3p5)))
      b(77)=ccc*(8*w2*(ffmcfmb*hbcm2*p6p9-fmb*hbcm*p6p9-hbcm2*p6p9-2*
     . p3p5*p6p9+2*p3p6*p5p9)+8*w12*(ffmcfmb*hbcm2*p6p10-fmb*hbcm*
     . p6p10-hbcm2*p6p10-2*p3p5*p6p10+2*p3p6*p5p10)-8*p3p13)
      b(78)=ccc*(8*w2*(ffmcfmb*hbcm*p6p9+fmb*p6p9-hbcm*p6p9)+8*w12*(
     . ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10))
      b(79)=8*ccc*w2*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(80)=w2*ccc*(8*p5p7*(-ffmcfmb*hbcm+fmc)+8*p4p7*(-ffmcfmb*hbcm+
     . fmc)+8*p3p7*(-ffmcfmb*hbcm+fmc))
      b(81)=8*ccc*w12*p6p7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(83)=8*ccc*(-hbcm*p4p7-hbcm*p5p7)
      b(84)=8*ccc*(-p3p7-p4p7-p5p7)
      b(85)=4*ccc*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3-2*p2p3+2*p3p4+2*
     . p3p5)
      b(87)=4*ccc*hbcm
      b(88)=w12*ccc*(8*p5p7*(-ffmcfmb*hbcm+fmc)+8*p4p7*(-ffmcfmb*hbcm
     . +fmc)+8*p3p7*(-ffmcfmb*hbcm+fmc))
      b(89)=4*ccc*hbcm
      do 200 n=1,89 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp7_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(80) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,80 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(5)=w2*ccc*(8*p5p7*(-fmb*hbcm+fmc*hbcm)+8*p4p7*(-fmb*hbcm+fmc*
     . hbcm)+8*p3p7*(-fmb*hbcm+fmc*hbcm))
      b(6)=8*ccc*w2*(fmb2*hbcm*p7p9-fmc2*hbcm*p7p9+2*hbcm*p1p3*p7p9+2
     . *hbcm*p2p3*p7p9-2*hbcm*p3p4*p7p9-2*hbcm*p3p7*p5p9-2*hbcm*p4p7*
     . p5p9-2*hbcm*p5p7*p5p9-hbcm3*p7p9)
      b(9)=8*ccc*w2*p4p6*(-fmb*hbcm+fmc*hbcm)
      b(10)=-16*ccc*hbcm*p4p6*p5p9*w2
      b(12)=w2*ccc*(8*p6p7*(-fmb2*hbcm+fmc2*hbcm-hbcm3)+16*(-hbcm*
     . p3p5*p6p7-hbcm*p3p7*p4p6))
      b(16)=4*ccc*w2*(-fmb2*hbcm+fmc2*hbcm-2*hbcm*p1p3-2*hbcm*p2p3+2*
     . hbcm*p3p4+hbcm3)
      b(21)=8*ccc*hbcm*p5p9*w2
      b(32)=-8*ccc*hbcm*p4p6*w2
      b(33)=16*ccc*hbcm*p4p6*p7p9*w2
      b(34)=4*ccc*hbcm*w2
      b(35)=-8*ccc*hbcm*p7p9*w2
      b(36)=4*ccc*w2*(-fmb*hbcm+fmc*hbcm)
      b(37)=8*ccc*hbcm*p3p7*w2
      b(38)=8*ccc*w2*(fmb*hbcm*p7p9-fmc*hbcm*p7p9)
      b(42)=8*ccc*w2*p6p7*(fmb*hbcm-fmc*hbcm)
      b(66)=16*ccc*hbcm*p5p9*p6p7*w2
      b(75)=8*ccc*w2*(hbcm*p1p9+hbcm*p2p9-hbcm*p3p9-hbcm*p4p9-hbcm*
     . p5p9)
      b(76)=8*ccc*w2*(fmb*hbcm*p1p9+fmb*hbcm*p2p9-fmb*hbcm*p3p9-fmb*
     . hbcm*p4p9-fmb*hbcm*p5p9-fmc*hbcm*p1p9-fmc*hbcm*p2p9+fmc*hbcm*
     . p3p9+fmc*hbcm*p4p9+fmc*hbcm*p5p9)
      b(79)=8*ccc*hbcm*p6p7*w2
      b(80)=8*ccc*w2*(-hbcm*p3p7-hbcm*p4p7-hbcm*p5p7)
      do 200 n=1,80 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp6_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(78) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,78 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p5*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      b(6)=w11*ccc*(16*p1p4*(-hbcm*p7p11+hbcm*p7p8)+16*p1p2*(hbcm*
     . p7p11-hbcm*p7p8)+16*p5p7*(hbcm*p1p11-hbcm*p1p8)+16*p4p7*(hbcm*
     . p1p11-hbcm*p1p8))
      b(8)=8*ccc*w11*p5p6*(-fmb*hbcm+fmc*hbcm)
      b(10)=w11*ccc*(16*p1p5*(hbcm*p6p11-hbcm*p6p8)+16*p5p6*(hbcm*
     . p1p11-hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*p3p11-hbcm*p4p11+
     . hbcm*p4p8)+16*p4p6*(hbcm*p1p11-hbcm*p1p8))
      b(11)=16*ccc*w11*(hbcm*p1p5*p6p7+hbcm*p4p7*p5p6+hbcm*p5p6*p5p7)
      b(15)=8*ccc*w11*(hbcm*p1p2-hbcm*p1p4)
      b(21)=8*ccc*w11*(hbcm*p2p11-hbcm*p2p8-hbcm*p3p11-hbcm*p4p11+
     . hbcm*p4p8)
      b(23)=8*ccc*w11*(hbcm*p4p6+hbcm*p5p6)
      b(24)=4*ccc*hbcm*w11
      b(25)=4*ccc*w11*(fmb*hbcm-fmc*hbcm)
      b(26)=8*ccc*w11*(hbcm*p4p7+hbcm*p5p7)
      b(30)=w11*ccc*(16*p5p6*(fmb*hbcm*p7p11-fmc*hbcm*p7p11)+16*p6p7*
     . (-fmb*hbcm*p1p11+fmb*hbcm*p1p8+fmc*hbcm*p1p11-fmc*hbcm*p1p8))
      b(31)=8*ccc*hbcm*p5p6*w11
      b(33)=16*ccc*w11*p5p6*(-hbcm*p7p11+hbcm*p7p8)
      b(35)=8*ccc*w11*(-hbcm*p7p11+hbcm*p7p8)
      b(38)=8*ccc*w11*(fmb*hbcm*p7p11-fmb*hbcm*p7p8-fmc*hbcm*p7p11+
     . fmc*hbcm*p7p8)
      b(39)=8*ccc*w11*p6p7*(fmb*hbcm-fmc*hbcm)
      b(40)=16*ccc*w11*(-hbcm*p1p5*p6p7-hbcm*p4p7*p5p6-hbcm*p5p6*p5p7
     . )
      b(44)=8*ccc*w11*p5p6*(-fmb*hbcm+fmc*hbcm)
      b(46)=8*ccc*hbcm*p5p6*w11
      b(49)=8*ccc*w11*(-hbcm*p4p7-hbcm*p5p7)
      b(54)=4*ccc*w11*(-fmb*hbcm+fmc*hbcm)
      b(55)=4*ccc*hbcm*w11
      b(58)=8*ccc*w11*(-hbcm*p4p6-hbcm*p5p6)
      b(59)=8*ccc*w11*(-hbcm*p1p2+hbcm*p1p4)
      b(65)=8*ccc*w11*p6p7*(-fmb*hbcm+fmc*hbcm)
      b(66)=w11*ccc*(16*p5p6*(hbcm*p7p11-hbcm*p7p8)+16*p4p6*(hbcm*
     . p7p11-hbcm*p7p8)+16*p6p7*(-hbcm*p1p11+hbcm*p1p8+hbcm*p5p11-
     . hbcm*p5p8))
      b(73)=8*ccc*hbcm*p6p7*w11
      b(74)=-8*ccc*hbcm*p6p7*w11
      b(75)=8*ccc*w11*(hbcm*p1p11-hbcm*p1p8)
      b(76)=8*ccc*w11*(fmb*hbcm*p1p11-fmb*hbcm*p1p8-fmc*hbcm*p1p11+
     . fmc*hbcm*p1p8)
      b(77)=8*ccc*w11*(fmb*hbcm*p6p11-fmb*hbcm*p6p8-fmc*hbcm*p6p11+
     . fmc*hbcm*p6p8)
      b(78)=8*ccc*w11*(hbcm*p6p11-hbcm*p6p8)
      do 200 n=1,78 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp5_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(72) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,72 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*
     . p3p4+fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=16*ccc*w5*(p3p7*p4p6+2*p4p6*p5p7)
      b(3)=ccc*(16*w5*p1p4*(-p7p10+p7p8)+16*p1p4*p7p11*w8)
      b(4)=-16*ccc*p1p4*p3p7*w5
      b(6)=ccc*(16*w5*p1p4*(hbcm*p7p10-hbcm*p7p8)-16*hbcm*p1p4*p7p11*
     . w8)
      b(7)=ccc*(16*w8*p4p6*(2*p1p11+p2p11-2*p3p11-2*p4p11)+16*w5*p4p6
     . *(-2*p1p10+2*p1p8-p2p10+p2p8+2*p3p10-2*p3p8+2*p4p10-2*p4p8))
      b(8)=w5*ccc*(8*p4p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p4p6-p2p3*p4p6+2*p3p4*p4p6))
      b(10)=ccc*(16*w8*p4p6*(-hbcm*p1p11-hbcm*p2p11+hbcm*p3p11+hbcm*
     . p4p11)+16*w5*p4p6*(ffmcfmb*hbcm*p3p10-ffmcfmb*hbcm*p3p8-fmb*
     . p3p10+fmb*p3p8-2*fmc*p3p10+2*fmc*p3p8+hbcm*p1p10-hbcm*p1p8+
     . hbcm*p2p10-hbcm*p2p8-hbcm*p3p10+hbcm*p3p8-hbcm*p4p10+hbcm*p4p8
     . ))
      b(11)=16*ccc*w5*(fmc*p3p7*p4p6+hbcm*p4p6*p5p7)
      b(13)=-16*ccc*p1p4*w5
      b(15)=8*ccc*hbcm*p1p4*w5
      b(18)=4*ccc*w5*(ffmcfmb*hbcm-fmb-2*fmc)
      b(19)=4*ccc*w5*(fmb*hbcm-fmc*hbcm-2*hbcm2+4*p1p3+2*p2p3-4*p3p4)
      b(20)=8*ccc*w5*(fmc*p3p7+hbcm*p5p7)
      b(21)=ccc*(8*w8*(hbcm*p1p11+hbcm*p2p11-hbcm*p3p11-hbcm*p4p11)+8
     . *w5*(-ffmcfmb*hbcm*p3p10+ffmcfmb*hbcm*p3p8+fmb*p3p10-fmb*p3p8+
     . 2*fmc*p3p10-2*fmc*p3p8-hbcm*p1p10+hbcm*p1p8-hbcm*p2p10+hbcm*
     . p2p8+hbcm*p3p10-hbcm*p3p8+hbcm*p4p10-hbcm*p4p8))
      b(22)=-8*ccc*hbcm*p1p4*w5
      b(24)=4*ccc*w5*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(25)=4*ccc*w5*(-fmb*hbcm+fmc*hbcm+2*hbcm2-4*p1p3-2*p2p3+4*p3p4
     . )
      b(26)=8*ccc*w5*(-fmc*p3p7-hbcm*p5p7)
      b(27)=8*ccc*w5*p4p6*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(28)=w5*ccc*(8*p4p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p4p6-p2p3*p4p6+2*p3p4*p4p6))
      b(29)=16*ccc*w5*(-fmc*p3p7*p4p6-hbcm*p4p6*p5p7)
      b(30)=ccc*(w8*(32*(p4p6*p3p7)*(-2*p1p11-2*p2p11+2*p3p11+2*p4p11
     . +p5p11)+16*p4p6*(ffmcfmb*hbcm2*p7p11+fmc*hbcm*p7p11)+32*(-p1p3
     . *p4p6*p7p11+p3p4*p4p6*p7p11))+w5*(32*(p4p6*p3p4)*(-p7p10-p7p8)
     . +32*(p4p6*p1p3)*(p7p10+p7p8)+32*(p4p6*p3p7)*(2*p1p10-2*p1p8+2*
     . p2p10-2*p2p8-2*p3p10+2*p3p8-2*p4p10+2*p4p8-p5p10+p5p8)+16*p4p6
     . *(-ffmcfmb*hbcm2*p7p10+ffmcfmb*hbcm2*p7p8+fmb*hbcm*p7p8-fmc*
     . hbcm*p7p10-2*hbcm2*p7p8)+32*p2p3*p4p6*p7p8)+16*p3p12*p4p6)
      b(31)=8*ccc*w5*p4p6*(ffmcfmb*hbcm-fmb-2*fmc)
      b(33)=ccc*(16*w8*p4p6*(ffmcfmb*hbcm*p7p11-fmc*p7p11)+16*w5*p4p6
     . *(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8+fmc*p7p10-fmc*p7p8))
      b(35)=ccc*(8*w8*(-ffmcfmb*hbcm*p7p11+fmc*p7p11)+8*w5*(ffmcfmb*
     . hbcm*p7p10-ffmcfmb*hbcm*p7p8-fmc*p7p10+fmc*p7p8))
      b(38)=ccc*(w8*(16*p3p7*(p2p11-p5p11)+8*(-ffmcfmb*hbcm2*p7p11-
     . fmc*hbcm*p7p11+2*p1p3*p7p11-2*p3p4*p7p11))+w5*(16*p3p4*(p7p10-
     . p7p8)+16*p3p7*(-p2p10+p2p8+p5p10-p5p8)+16*p1p3*(-p7p10+p7p8)+8
     . *(ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p8+fmc*hbcm*p7p10-fmc*
     . hbcm*p7p8))-8*p3p12)
      b(40)=16*ccc*w8*(fmc*p3p7*p4p6+hbcm*p4p6*p5p7)
      b(41)=16*ccc*w8*(p3p7*p4p6+2*p4p6*p5p7)
      b(44)=w8*ccc*(8*p4p6*(fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(2*p1p3*
     . p4p6+p2p3*p4p6-2*p3p4*p4p6))
      b(46)=8*ccc*w8*p4p6*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(49)=8*ccc*w8*(-fmc*p3p7-hbcm*p5p7)
      b(50)=8*ccc*w8*(-p3p7-2*p5p7)
      b(51)=-16*ccc*p1p4*p3p7*w8
      b(53)=ccc*(8*w8*(2*p1p11+p2p11-2*p3p11-2*p4p11)+8*w5*(-2*p1p10+
     . 2*p1p8-p2p10+p2p8+2*p3p10-2*p3p8+2*p4p10-2*p4p8))
      b(54)=4*ccc*w8*(-fmb*hbcm+fmc*hbcm+2*hbcm2-4*p1p3-2*p2p3+4*p3p4
     . )
      b(55)=4*ccc*w8*(ffmcfmb*hbcm-fmb-2*fmc)
      b(56)=-16*ccc*p1p4*w8
      b(59)=8*ccc*hbcm*p1p4*w8
      b(61)=4*ccc*hbcm
      b(62)=-8*ccc*hbcm*p4p6
      b(64)=16*ccc*w5*(-p3p7*p4p6-2*p4p6*p5p7)
      b(69)=8*ccc*w5*(p3p7+2*p5p7)
      b(70)=16*ccc*p1p4*p3p7*w5
      b(71)=8*ccc*w5*(-p3p7-2*p5p7)
      b(72)=16*ccc*p1p4*w5
      do 200 n=1,72 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp4_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(62) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,62 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)
     . )
      b(2)=16*ccc*w2*(-p3p7*p4p6-2*p4p6*p5p7)
      b(3)=16*ccc*p1p4*(-p7p11*w7-p7p9*w2)
      b(5)=16*ccc*p1p4*p3p7*w2
      b(7)=ccc*(16*w2*p4p6*(-2*p1p9-p2p9+2*p3p9+2*p4p9)+16*w7*p4p6*(-
     . 2*p1p11-p2p11+2*p3p11+2*p4p11))
      b(9)=w2*ccc*(8*p4p6*(fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(2*p1p3*p4p6
     . +p2p3*p4p6-2*p3p4*p4p6))
      b(10)=ccc*(16*w2*p4p6*(hbcm*p1p9-hbcm*p3p9-hbcm*p4p9)+16*w7*
     . p4p6*(hbcm*p1p11-hbcm*p3p11-hbcm*p4p11))
      b(12)=w2*ccc*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb-fmc)-16*hbcm*
     . p4p6*p5p7)
      b(14)=16*ccc*p1p4*w2
      b(16)=-8*ccc*hbcm*p1p4*w2
      b(21)=ccc*(8*w2*(-hbcm*p1p9+hbcm*p3p9+hbcm*p4p9)+8*w7*(-hbcm*
     . p1p11+hbcm*p3p11+hbcm*p4p11))
      b(30)=ccc*(w2*(32*(p4p6*p3p7)*(2*p1p9+2*p2p9-2*p3p9-2*p4p9-p5p9
     . )+16*p4p6*(ffmcfmb*hbcm2*p7p9-fmb*hbcm*p7p9+hbcm2*p7p9)+32*(-
     . p1p3*p4p6*p7p9-p2p3*p4p6*p7p9+p3p4*p4p6*p7p9))+w7*(32*(p4p6*
     . p3p7)*(2*p1p11+2*p2p11-2*p3p11-2*p4p11-p5p11)+16*p4p6*(ffmcfmb
     . *hbcm2*p7p11-fmc*hbcm*p7p11-hbcm2*p7p11)+32*(p1p3*p4p6*p7p11-
     . p3p4*p4p6*p7p11))-16*p3p12*p4p6)
      b(32)=8*ccc*w2*p4p6*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      b(33)=ccc*(16*w2*p4p6*(fmb*p7p9+fmc*p7p9)+16*w7*p4p6*(fmb*p7p11
     . +fmc*p7p11))
      b(34)=4*ccc*w2*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(35)=ccc*(8*w2*(-fmb*p7p9-fmc*p7p9)+8*w7*(-fmb*p7p11-fmc*p7p11
     . ))
      b(36)=4*ccc*w2*(fmb*hbcm-fmc*hbcm-2*hbcm2+4*p1p3+2*p2p3-4*p3p4)
      b(37)=w2*ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(38)=ccc*(w2*(16*p3p7*(-p2p9+p5p9)+8*(-ffmcfmb*hbcm2*p7p9+fmc*
     . hbcm*p7p9+hbcm2*p7p9-2*p1p3*p7p9+2*p3p4*p7p9))+w7*(16*p3p7*(-
     . p2p11+p5p11)+8*(-ffmcfmb*hbcm2*p7p11+fmc*hbcm*p7p11+hbcm2*
     . p7p11-2*p1p3*p7p11+2*p3p4*p7p11))+8*p3p12)
      b(40)=w7*ccc*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb-fmc)-16*hbcm*
     . p4p6*p5p7)
      b(41)=16*ccc*w7*(-p3p7*p4p6-2*p4p6*p5p7)
      b(44)=w7*ccc*(8*p4p6*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(-2*p1p3*
     . p4p6-p2p3*p4p6+2*p3p4*p4p6))
      b(46)=8*ccc*w7*p4p6*(-ffmcfmb*hbcm-2*fmb-fmc+hbcm)
      b(49)=w7*ccc*(8*p3p7*(ffmcfmb*hbcm+fmb+fmc)+8*hbcm*p5p7)
      b(50)=8*ccc*w7*(p3p7+2*p5p7)
      b(51)=16*ccc*p1p4*p3p7*w7
      b(52)=8*ccc*w2*(p3p7+2*p5p7)
      b(53)=ccc*(8*w2*(-2*p1p9-p2p9+2*p3p9+2*p4p9)+8*w7*(-2*p1p11-
     . p2p11+2*p3p11+2*p4p11))
      b(54)=4*ccc*w7*(fmb*hbcm-fmc*hbcm-2*hbcm2+4*p1p3+2*p2p3-4*p3p4)
      b(55)=4*ccc*w7*(ffmcfmb*hbcm+2*fmb+fmc-hbcm)
      b(56)=16*ccc*p1p4*w7
      b(59)=-8*ccc*hbcm*p1p4*w7
      b(61)=4*ccc*hbcm
      b(62)=-8*ccc*hbcm*p4p6
      do 200 n=1,62 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp36_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=ccc*(w5*(2*p6p7*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb
     . *fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm-2*fmc2)+4*(-2*ffmcfmb
     . *p3p6*p3p7+2*ffmcfmb*p3p6*p4p7-6*ffmcfmb*p3p6*p5p7+2*p1p2*p6p7
     . -2*p1p3*p6p7+2*p1p4*p6p7-2*p1p5*p6p7-3*p3p7*p4p6+p4p6*p4p7-7*
     . p4p6*p5p7))+w1*(2*p6p7*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm-2*fmc2)+4*(-2*
     . ffmcfmb*p3p6*p3p7+2*ffmcfmb*p3p6*p4p7-6*ffmcfmb*p3p6*p5p7+2*
     . p1p2*p6p7-2*p1p3*p6p7+2*p1p4*p6p7-2*p1p5*p6p7-3*p3p7*p4p6+p4p6
     . *p4p7-7*p4p6*p5p7)))
      b(2)=w2*ccc*(2*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb
     . *fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+4*(2*ffmcfmb*
     . p3p6*p3p7-2*ffmcfmb*p3p6*p4p7+6*ffmcfmb*p3p6*p5p7-2*p1p2*p6p7+
     . 2*p1p3*p6p7-2*p1p4*p6p7+2*p1p5*p6p7+3*p3p7*p4p6-p4p6*p4p7+7*
     . p4p6*p5p7))
      b(3)=ccc*(w2*(4*p5p7*(p1p9+p4p9)+4*p4p7*(-2*p1p9+p2p9-p5p9)+4*
     . p3p7*(-3*ffmcfmb*p1p9+ffmcfmb*p2p9-ffmcfmb*p4p9-ffmcfmb*p5p9+
     . p1p9+p4p9))+w1*(4*p5p7*(-p1p8-p4p8)+4*p4p7*(2*p1p8-p2p8+p5p8)+
     . 4*p3p7*(3*ffmcfmb*p1p8-ffmcfmb*p2p8+ffmcfmb*p4p8+ffmcfmb*p5p8-
     . p1p8-p4p8))+w5*(4*p5p7*(p1p10-p1p8+p4p10-p4p8)+4*p4p7*(-2*
     . p1p10+2*p1p8+p2p10-p2p8-p5p10+p5p8)+4*p3p7*(-3*ffmcfmb*p1p10+3
     . *ffmcfmb*p1p8+ffmcfmb*p2p10-ffmcfmb*p2p8-ffmcfmb*p4p10+ffmcfmb
     . *p4p8-ffmcfmb*p5p10+ffmcfmb*p5p8+p1p10-p1p8+p4p10-p4p8))+2*(-3
     . *p1p12+p2p12-p4p12-p5p12))
      b(4)=ccc*(w5*((p3p7*p3p4)*(-4*ffmcfmb+4)+(p3p7*p1p3)*(-12*
     . ffmcfmb+4)+2*p4p7*(-ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-hbcm2)+2
     . *p3p7*(-3*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm
     . +ffmcfmb*hbcm2)+4*(ffmcfmb*hbcm2*p5p7+ffmcfmb*p2p3*p3p7-
     . ffmcfmb*p3p5*p3p7-2*p1p3*p4p7+p1p3*p5p7+p2p3*p4p7+p3p4*p5p7-
     . p3p5*p4p7))+w1*((p3p7*p3p4)*(-4*ffmcfmb+4)+(p3p7*p1p3)*(-12*
     . ffmcfmb+4)+2*p4p7*(-ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm-hbcm2)+2
     . *p3p7*(-3*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm
     . +ffmcfmb*hbcm2)+4*(ffmcfmb*hbcm2*p5p7+ffmcfmb*p2p3*p3p7-
     . ffmcfmb*p3p5*p3p7-2*p1p3*p4p7+p1p3*p5p7+p2p3*p4p7+p3p4*p5p7-
     . p3p5*p4p7)))
      b(5)=w2*ccc*((p3p7*p3p4)*(4*ffmcfmb-4)+(p3p7*p1p3)*(12*ffmcfmb-
     . 4)+2*p4p7*(ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm+hbcm2)+2*p3p7*(3*
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2)+4*(-ffmcfmb*hbcm2*p5p7-ffmcfmb*p2p3*p3p7+ffmcfmb*p3p5*
     . p3p7+2*p1p3*p4p7-p1p3*p5p7-p2p3*p4p7-p3p4*p5p7+p3p5*p4p7))
      ans=ccc*(w2*(4*p5p7*(-hbcm*p1p9-hbcm*p4p9)+4*p4p7*(2*hbcm*p1p9-
     . hbcm*p2p9+hbcm*p5p9)+4*p3p7*(3*ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*
     . p2p9+ffmcfmb*hbcm*p4p9+ffmcfmb*hbcm*p5p9-hbcm*p1p9-hbcm*p4p9))
     . +w1*(4*p5p7*(hbcm*p1p8+hbcm*p4p8)+4*p4p7*(-2*hbcm*p1p8+hbcm*
     . p2p8-hbcm*p5p8)+4*p3p7*(-3*ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p8
     . -ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p8+hbcm*p1p8+hbcm*p4p8))+w5*
     . (4*p5p7*(-hbcm*p1p10+hbcm*p1p8-hbcm*p4p10+hbcm*p4p8)+4*p4p7*(2
     . *hbcm*p1p10-2*hbcm*p1p8-hbcm*p2p10+hbcm*p2p8+hbcm*p5p10-hbcm*
     . p5p8)+4*p3p7*(3*ffmcfmb*hbcm*p1p10-3*ffmcfmb*hbcm*p1p8-ffmcfmb
     . *hbcm*p2p10+ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p4p10-ffmcfmb*hbcm*
     . p4p8+ffmcfmb*hbcm*p5p10-ffmcfmb*hbcm*p5p8-hbcm*p1p10+hbcm*p1p8
     . -hbcm*p4p10+hbcm*p4p8))+2*(3*hbcm*p1p12-hbcm*p2p12+hbcm*p4p12+
     . hbcm*p5p12))
      b(6)=ans
      b(7)=ccc*(w2*(8*p4p6*(p2p9-2*p3p9-3*p4p9+p5p9)+8*p3p6*(-ffmcfmb
     . *p1p9+ffmcfmb*p2p9-2*ffmcfmb*p3p9-3*ffmcfmb*p4p9+ffmcfmb*p5p9+
     . p1p9)+8*p1p9*p5p6)+w1*(8*p4p6*(-p2p8+2*p3p8+3*p4p8-p5p8)+8*
     . p3p6*(ffmcfmb*p1p8-ffmcfmb*p2p8+2*ffmcfmb*p3p8+3*ffmcfmb*p4p8-
     . ffmcfmb*p5p8-p1p8)-8*p1p8*p5p6)+w5*(8*p4p6*(p2p10-p2p8-2*p3p10
     . +2*p3p8-3*p4p10+3*p4p8+p5p10-p5p8)+8*p5p6*(p1p10-p1p8)+8*p3p6*
     . (-ffmcfmb*p1p10+ffmcfmb*p1p8+ffmcfmb*p2p10-ffmcfmb*p2p8-2*
     . ffmcfmb*p3p10+2*ffmcfmb*p3p8-3*ffmcfmb*p4p10+3*ffmcfmb*p4p8+
     . ffmcfmb*p5p10-ffmcfmb*p5p8+p1p10-p1p8))+4*(-p1p13+p2p13-2*
     . p3p13-3*p4p13+p5p13))
      b(8)=ccc*(w5*((p3p6*p1p3)*(-8*ffmcfmb+8)+2*p5p6*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+4*p4p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-
     . hbcm2)+2*p3p6*(-3*ffmcfmb**2*hbcm2+6*ffmcfmb*fmb*hbcm-7*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*p2p3*
     . p3p6-3*ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6+p1p3*p5p6+p2p3*p4p6
     . -3*p3p4*p4p6+p3p5*p4p6))+w1*((p3p6*p1p3)*(-8*ffmcfmb+8)+2*p5p6
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+4*p4p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm
     . -3*fmc*hbcm-hbcm2)+2*p3p6*(-3*ffmcfmb**2*hbcm2+6*ffmcfmb*fmb*
     . hbcm-7*ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*
     . p2p3*p3p6-3*ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6+p1p3*p5p6+p2p3
     . *p4p6-3*p3p4*p4p6+p3p5*p4p6)))
      b(9)=w2*ccc*((p3p6*p1p3)*(8*ffmcfmb-8)+2*p5p6*(ffmcfmb*hbcm2-
     . fmc*hbcm)+4*p4p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+hbcm2)
     . +2*p3p6*(3*ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm+7*ffmcfmb*fmc*
     . hbcm+3*ffmcfmb*hbcm2-fmc*hbcm)+8*(-ffmcfmb*p2p3*p3p6+3*ffmcfmb
     . *p3p4*p3p6-ffmcfmb*p3p5*p3p6-p1p3*p5p6-p2p3*p4p6+3*p3p4*p4p6-
     . p3p5*p4p6))
      ans2=w5*(8*p4p6*(3*fmb*p3p10-3*fmb*p3p8+3*fmc*p3p10-3*fmc*p3p8-
     . 3*hbcm*p1p10+3*hbcm*p1p8-hbcm*p2p10+hbcm*p2p8-hbcm*p3p10+hbcm*
     . p3p8+3*hbcm*p4p10-3*hbcm*p4p8-hbcm*p5p10+hbcm*p5p8)+4*p5p6*(
     . ffmcfmb*hbcm*p3p10-ffmcfmb*hbcm*p3p8-fmc*p3p10+fmc*p3p8)+4*
     . p3p6*(-ffmcfmb**2*hbcm*p3p10+ffmcfmb**2*hbcm*p3p8+6*ffmcfmb*
     . fmb*p3p10-6*ffmcfmb*fmb*p3p8+7*ffmcfmb*fmc*p3p10-7*ffmcfmb*fmc
     . *p3p8-6*ffmcfmb*hbcm*p1p10+6*ffmcfmb*hbcm*p1p8-2*ffmcfmb*hbcm*
     . p2p10+2*ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p3p10+ffmcfmb*hbcm*p3p8
     . +6*ffmcfmb*hbcm*p4p10-6*ffmcfmb*hbcm*p4p8-2*ffmcfmb*hbcm*p5p10
     . +2*ffmcfmb*hbcm*p5p8-fmc*p3p10+fmc*p3p8))+4*(hbcm*p1p13-hbcm*
     . p2p13+2*hbcm*p3p13+3*hbcm*p4p13-hbcm*p5p13)
      ans1=w2*(8*p4p6*(-3*hbcm*p1p9-hbcm*p2p9+2*hbcm*p3p9+3*hbcm*p4p9
     . -hbcm*p5p9)+8*p3p6*(-3*ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+2*
     . ffmcfmb*hbcm*p3p9+3*ffmcfmb*hbcm*p4p9-ffmcfmb*hbcm*p5p9))+w1*(
     . 8*p4p6*(-3*fmb*p3p8-3*fmc*p3p8+3*hbcm*p1p8+hbcm*p2p8+hbcm*p3p8
     . -3*hbcm*p4p8+hbcm*p5p8)+4*p5p6*(-ffmcfmb*hbcm*p3p8+fmc*p3p8)+4
     . *p3p6*(ffmcfmb**2*hbcm*p3p8-6*ffmcfmb*fmb*p3p8-7*ffmcfmb*fmc*
     . p3p8+6*ffmcfmb*hbcm*p1p8+2*ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p3p8
     . -6*ffmcfmb*hbcm*p4p8+2*ffmcfmb*hbcm*p5p8+fmc*p3p8))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans2=w1*(4*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-2*fmb-4*fmc+hbcm)+4*(
     . p5p7*p3p6)*(-5*ffmcfmb*hbcm+fmc)+4*(p4p7*p3p6)*(ffmcfmb*hbcm-
     . fmc)+4*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+4*(p6p7*p3p4)*(-ffmcfmb*
     . hbcm+fmc)+4*(p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+4*(p6p7*p1p3)*(
     . ffmcfmb*hbcm-2*fmb-3*fmc)+4*(p3p7*p3p6)*(-2*ffmcfmb*fmb-6*
     . ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+2*p6p7*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*
     . hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*(2*hbcm*p1p2*p6p7+2*hbcm*p1p4*
     . p6p7-2*hbcm*p1p5*p6p7-hbcm*p4p6*p4p7-5*hbcm*p4p6*p5p7))
      ans1=w5*(4*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-2*fmb-4*fmc+hbcm)+4*(
     . p5p7*p3p6)*(-5*ffmcfmb*hbcm+fmc)+4*(p4p7*p3p6)*(ffmcfmb*hbcm-
     . fmc)+4*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+4*(p6p7*p3p4)*(-ffmcfmb*
     . hbcm+fmc)+4*(p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+4*(p6p7*p1p3)*(
     . ffmcfmb*hbcm-2*fmb-3*fmc)+4*(p3p7*p3p6)*(-2*ffmcfmb*fmb-6*
     . ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+2*p6p7*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*
     . hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*(2*hbcm*p1p2*p6p7+2*hbcm*p1p4*
     . p6p7-2*hbcm*p1p5*p6p7-hbcm*p4p6*p4p7-5*hbcm*p4p6*p5p7))+ans2
      ans=ccc*ans1
      b(11)=ans
      b(12)=w2*ccc*(4*(p4p6*p3p7)*(2*ffmcfmb*hbcm+2*fmb+4*fmc-hbcm)+4
     . *(p5p7*p3p6)*(5*ffmcfmb*hbcm-fmc)+4*(p4p7*p3p6)*(-ffmcfmb*hbcm
     . +fmc)+4*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+4*(p6p7*p3p4)*(ffmcfmb
     . *hbcm-fmc)+4*(p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+4*(p6p7*p1p3)*(-
     . ffmcfmb*hbcm+fmc+2*hbcm)+4*(p3p7*p3p6)*(2*ffmcfmb*fmb+6*
     . ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+2*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*
     . hbcm2+2*fmc2*hbcm)+4*(-2*hbcm*p1p2*p6p7-2*hbcm*p1p4*p6p7+2*
     . hbcm*p1p5*p6p7+hbcm*p4p6*p4p7+5*hbcm*p4p6*p5p7))
      b(13)=ccc*(w5*(p3p4*(-2*ffmcfmb+2)+p1p3*(-2*ffmcfmb+2)+2*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*hbcm2+ffmcfmb*p2p3+ffmcfmb*p3p5+p1p2+
     . p1p5+p2p4+p4p5))+w1*(p3p4*(-2*ffmcfmb+2)+p1p3*(-2*ffmcfmb+2)+2
     . *(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2+ffmcfmb*p2p3+ffmcfmb*p3p5+
     . p1p2+p1p5+p2p4+p4p5)))+ccc
      b(14)=ccc*(w2*(p3p4*(2*ffmcfmb-2)+p1p3*(2*ffmcfmb-2)+2*(ffmcfmb
     . **2*hbcm2-ffmcfmb*hbcm2-ffmcfmb*p2p3-ffmcfmb*p3p5-p1p2-p1p5-
     . p2p4-p4p5))-1)
      b(15)=ccc*(w5*(2*p3p4*(ffmcfmb*hbcm-hbcm)+2*p1p3*(ffmcfmb*hbcm-
     . hbcm)+2*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm*p2p3-ffmcfmb*hbcm*p3p5-
     . ffmcfmb*hbcm3-hbcm*p1p2-hbcm*p1p5-hbcm*p2p4-hbcm*p4p5))+w1*(2*
     . p3p4*(ffmcfmb*hbcm-hbcm)+2*p1p3*(ffmcfmb*hbcm-hbcm)+2*(ffmcfmb
     . **2*hbcm3-ffmcfmb*hbcm*p2p3-ffmcfmb*hbcm*p3p5-ffmcfmb*hbcm3-
     . hbcm*p1p2-hbcm*p1p5-hbcm*p2p4-hbcm*p4p5))-hbcm)
      b(16)=ccc*(w2*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p1p3*(-ffmcfmb*
     . hbcm+hbcm)+2*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p2p3+ffmcfmb*hbcm
     . *p3p5+ffmcfmb*hbcm3+hbcm*p1p2+hbcm*p1p5+hbcm*p2p4+hbcm*p4p5))+
     . hbcm)
      b(17)=w5*ccc*(4*p3p6*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p6-hbcm*
     . p5p6))
      b(20)=w5*ccc*(2*p3p7*(-8*ffmcfmb*hbcm+3*hbcm)+2*(-5*hbcm*p4p7+3
     . *hbcm*p5p7))
      b(22)=ccc*(w5*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p1p3*(-ffmcfmb*
     . hbcm+hbcm)+2*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p2p3+ffmcfmb*hbcm
     . *p3p5+ffmcfmb*hbcm3+hbcm*p1p2+hbcm*p1p5+hbcm*p2p4+hbcm*p4p5))-
     . hbcm)
      b(23)=ccc*(w5*(4*p3p6*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p4p6+
     . hbcm*p5p6))+w1*(4*p3p6*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p4p6+
     . hbcm*p5p6)))
      b(26)=ccc*(w5*(2*p3p7*(8*ffmcfmb*hbcm-3*hbcm)+2*(5*hbcm*p4p7-3*
     . hbcm*p5p7))+w1*(2*p3p7*(8*ffmcfmb*hbcm-3*hbcm)+2*(5*hbcm*p4p7-
     . 3*hbcm*p5p7)))
      b(27)=w5*ccc*(12*p4p6*(-fmb-fmc+hbcm)+2*p5p6*(-ffmcfmb*hbcm+fmc
     . )+2*p3p6*(ffmcfmb**2*hbcm-6*ffmcfmb*fmb-7*ffmcfmb*fmc+5*
     . ffmcfmb*hbcm+fmc))
      b(28)=w5*ccc*((p3p6*p1p3)*(-8*ffmcfmb+8)+2*p5p6*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+4*p4p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-
     . hbcm2)+2*p3p6*(-3*ffmcfmb**2*hbcm2+6*ffmcfmb*fmb*hbcm-7*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*p2p3*
     . p3p6-3*ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6+p1p3*p5p6+p2p3*p4p6
     . -3*p3p4*p4p6+p3p5*p4p6))
      b(29)=w5*ccc*(4*(p4p6*p3p7)*(2*ffmcfmb*hbcm+2*fmb+4*fmc-hbcm)+4
     . *(p5p7*p3p6)*(5*ffmcfmb*hbcm-fmc)+4*(p4p7*p3p6)*(-ffmcfmb*hbcm
     . +fmc)+4*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+4*(p6p7*p3p4)*(ffmcfmb
     . *hbcm-fmc)+4*(p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+4*(p6p7*p1p3)*(-
     . ffmcfmb*hbcm+2*fmb+3*fmc)+4*(p3p7*p3p6)*(2*ffmcfmb*fmb+6*
     . ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+2*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*
     . hbcm2+2*fmc2*hbcm)+4*(-2*hbcm*p1p2*p6p7-2*hbcm*p1p4*p6p7+2*
     . hbcm*p1p5*p6p7+hbcm*p4p6*p4p7+5*hbcm*p4p6*p5p7))
      ans5=4*p6p7*(-ffmcfmb*hbcm2*p1p10+ffmcfmb*hbcm2*p1p8+ffmcfmb*
     . hbcm2*p2p10-ffmcfmb*hbcm2*p2p8+ffmcfmb*hbcm2*p4p10-ffmcfmb*
     . hbcm2*p4p8-ffmcfmb*hbcm2*p5p10+ffmcfmb*hbcm2*p5p8+fmc*hbcm*
     . p1p10-fmc*hbcm*p1p8-fmc*hbcm*p2p10+fmc*hbcm*p2p8-fmc*hbcm*
     . p4p10+fmc*hbcm*p4p8+fmc*hbcm*p5p10-fmc*hbcm*p5p8+4*hbcm2*p1p10
     . -4*hbcm2*p1p8)+4*p3p7*(-2*ffmcfmb**2*hbcm2*p6p10+2*ffmcfmb**2*
     . hbcm2*p6p8+2*ffmcfmb*fmc*hbcm*p6p10-2*ffmcfmb*fmc*hbcm*p6p8+
     . ffmcfmb*hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm
     . *p6p8)+4*p3p6*(2*ffmcfmb**2*hbcm2*p7p10+ffmcfmb**2*hbcm2*p7p8-
     . 2*ffmcfmb*fmb*hbcm*p7p10-4*ffmcfmb*fmb*hbcm*p7p8+4*ffmcfmb*fmc
     . *hbcm*p7p10+3*ffmcfmb*fmc*hbcm*p7p8+2*ffmcfmb*hbcm2*p7p10+
     . ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8)+16*(-ffmcfmb*p2p3*p3p6*p7p8-
     . ffmcfmb*p3p5*p3p6*p7p8-p1p3*p5p6*p7p8-p2p3*p4p6*p7p8-p3p5*p4p6
     . *p7p8)
      ans4=16*(p4p6*p3p4)*(2*p7p10+p7p8)+16*(p5p7*p1p3)*(-p6p10+p6p8)
     . +16*(p4p7*p1p3)*(p6p10-p6p8)+16*(p6p7*p1p3)*(-p2p10+p2p8-p4p10
     . +p4p8+p5p10-p5p8)+16*(p5p6*p3p7)*(-p1p10+p1p8)+16*(p4p6*p3p7)*
     . (p1p10-p1p8-2*p2p10+2*p2p8+2*p3p10-2*p3p8+2*p4p10-2*p4p8)+16*(
     . p6p7*p3p5)*(p1p10-p1p8)+16*(p6p7*p3p4)*(p1p10-p1p8)+16*(p6p7*
     . p2p3)*(-p1p10+p1p8)+16*(p3p6*p3p4)*(2*ffmcfmb*p7p10+ffmcfmb*
     . p7p8)+16*(p3p7*p1p3)*(2*ffmcfmb*p6p10-2*ffmcfmb*p6p8-p6p10+
     . p6p8)+16*(p3p6*p1p3)*(ffmcfmb*p7p8-p7p8)+16*(p3p7*p3p6)*(2*
     . ffmcfmb*p1p10-2*ffmcfmb*p1p8-2*ffmcfmb*p2p10+2*ffmcfmb*p2p8+2*
     . ffmcfmb*p3p10-2*ffmcfmb*p3p8+2*ffmcfmb*p4p10-2*ffmcfmb*p4p8-
     . p1p10+p1p8)+4*p5p7*(ffmcfmb*hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc
     . *hbcm*p6p10+fmc*hbcm*p6p8)+4*p5p6*(ffmcfmb*hbcm2*p7p8-fmc*hbcm
     . *p7p8)+4*p4p7*(-ffmcfmb*hbcm2*p6p10+ffmcfmb*hbcm2*p6p8+fmc*
     . hbcm*p6p10-fmc*hbcm*p6p8)+8*p4p6*(ffmcfmb*hbcm2*p7p10+ffmcfmb*
     . hbcm2*p7p8-fmb*hbcm*p7p10-2*fmb*hbcm*p7p8+2*fmc*hbcm*p7p10+fmc
     . *hbcm*p7p8+hbcm2*p7p10)+ans5
      ans3=w5*ans4
      ans6=(4*p6p7*(-2*p3p10+p3p8)+4*p3p7*(-ffmcfmb*p3p13+4*p1p13-4*
     . p2p13+5*p3p13+4*p4p13)+8*p3p6*(-ffmcfmb*p3p12-2*p1p12)+4*(-
     . ffmcfmb*hbcm2*p6p12+ffmcfmb*hbcm2*p7p13-fmb*hbcm*p7p13+fmc*
     . hbcm*p6p12+2*fmc*hbcm*p7p13+hbcm2*p7p13+4*p1p3*p6p12-2*p3p12*
     . p4p6+p3p13*p5p7+4*p3p4*p7p13))
      ans2=w1*(16*(p4p6*p3p7)*(-p1p8+2*p2p8-2*p3p8-2*p4p8)+16*(p6p7*
     . p1p3)*(p2p8+p4p8-p5p8)+16*(p3p7*p3p6)*(-2*ffmcfmb*p1p8+2*
     . ffmcfmb*p2p8-2*ffmcfmb*p3p8-2*ffmcfmb*p4p8+p1p8)+16*(p3p7*p1p3
     . )*(-2*ffmcfmb*p6p8+p6p8)+16*(p3p6*p1p3)*(ffmcfmb*p7p8-p7p8)+4*
     . p5p7*(-ffmcfmb*hbcm2*p6p8+fmc*hbcm*p6p8)+4*p5p6*(ffmcfmb*hbcm2
     . *p7p8-fmc*hbcm*p7p8)+4*p4p7*(ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p8)
     . +8*p4p6*(ffmcfmb*hbcm2*p7p8-2*fmb*hbcm*p7p8+fmc*hbcm*p7p8)+4*
     . p6p7*(ffmcfmb*hbcm2*p1p8-ffmcfmb*hbcm2*p2p8-ffmcfmb*hbcm2*p4p8
     . +ffmcfmb*hbcm2*p5p8-fmc*hbcm*p1p8+fmc*hbcm*p2p8+fmc*hbcm*p4p8-
     . fmc*hbcm*p5p8-4*hbcm2*p1p8)+4*p3p7*(2*ffmcfmb**2*hbcm2*p6p8-2*
     . ffmcfmb*fmc*hbcm*p6p8-ffmcfmb*hbcm2*p6p8+fmc*hbcm*p6p8)+4*p3p6
     . *(ffmcfmb**2*hbcm2*p7p8-4*ffmcfmb*fmb*hbcm*p7p8+3*ffmcfmb*fmc*
     . hbcm*p7p8+ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8)+16*(-ffmcfmb*p2p3*
     . p3p6*p7p8+ffmcfmb*p3p4*p3p6*p7p8-ffmcfmb*p3p5*p3p6*p7p8-p1p3*
     . p4p7*p6p8-p1p3*p5p6*p7p8+p1p3*p5p7*p6p8+p1p8*p2p3*p6p7-p1p8*
     . p3p4*p6p7-p1p8*p3p5*p6p7+p1p8*p3p7*p5p6-p2p3*p4p6*p7p8+p3p4*
     . p4p6*p7p8-p3p5*p4p6*p7p8))+ans3+ans6
      ans1=w2*(16*(p4p6*p3p7)*(p1p9-2*p2p9+2*p3p9+2*p4p9)+16*(p6p7*
     . p1p3)*(-p2p9-p4p9+p5p9)+16*(p3p7*p3p6)*(2*ffmcfmb*p1p9-2*
     . ffmcfmb*p2p9+2*ffmcfmb*p3p9+2*ffmcfmb*p4p9-p1p9)+16*(p3p7*p1p3
     . )*(2*ffmcfmb*p6p9-p6p9)+16*(p3p6*p1p3)*(-ffmcfmb*p7p9+p7p9)+4*
     . p5p7*(ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+4*p5p6*(-ffmcfmb*hbcm2
     . *p7p9+fmc*hbcm*p7p9)+4*p4p7*(-ffmcfmb*hbcm2*p6p9+fmc*hbcm*p6p9
     . )+8*p4p6*(-ffmcfmb*hbcm2*p7p9+2*fmb*hbcm*p7p9-fmc*hbcm*p7p9)+4
     . *p6p7*(-ffmcfmb*hbcm2*p1p9+ffmcfmb*hbcm2*p2p9+ffmcfmb*hbcm2*
     . p4p9-ffmcfmb*hbcm2*p5p9+fmc*hbcm*p1p9-fmc*hbcm*p2p9-fmc*hbcm*
     . p4p9+fmc*hbcm*p5p9+4*hbcm2*p1p9)+4*p3p7*(-2*ffmcfmb**2*hbcm2*
     . p6p9+2*ffmcfmb*fmc*hbcm*p6p9+ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)
     . +4*p3p6*(-ffmcfmb**2*hbcm2*p7p9+4*ffmcfmb*fmb*hbcm*p7p9-3*
     . ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*hbcm2*p7p9+fmc*hbcm*p7p9)+16*(
     . ffmcfmb*p2p3*p3p6*p7p9-ffmcfmb*p3p4*p3p6*p7p9+ffmcfmb*p3p5*
     . p3p6*p7p9+p1p3*p4p7*p6p9+p1p3*p5p6*p7p9-p1p3*p5p7*p6p9-p1p9*
     . p2p3*p6p7+p1p9*p3p4*p6p7+p1p9*p3p5*p6p7-p1p9*p3p7*p5p6+p2p3*
     . p4p6*p7p9-p3p4*p4p6*p7p9+p3p5*p4p6*p7p9))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w5*(12*p4p6*(fmb+fmc-hbcm)+2*p5p6*(ffmcfmb*hbcm-fmc)
     . +2*p3p6*(-ffmcfmb**2*hbcm+6*ffmcfmb*fmb+7*ffmcfmb*fmc-5*
     . ffmcfmb*hbcm-fmc))+w1*(12*p4p6*(fmb+fmc-hbcm)+2*p5p6*(ffmcfmb*
     . hbcm-fmc)+2*p3p6*(-ffmcfmb**2*hbcm+6*ffmcfmb*fmb+7*ffmcfmb*fmc
     . -5*ffmcfmb*hbcm-fmc)))
      b(32)=w2*ccc*(12*p4p6*(fmb+fmc-hbcm)+2*p5p6*(ffmcfmb*hbcm-fmc)+
     . 2*p3p6*(-ffmcfmb**2*hbcm+6*ffmcfmb*fmb+7*ffmcfmb*fmc-5*ffmcfmb
     . *hbcm-fmc))
      ans2=w5*(4*p5p7*(-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p8+fmc*
     . p6p10-fmc*p6p8)+4*p4p7*(ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p8-
     . fmc*p6p10+fmc*p6p8)+8*p4p6*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*
     . p7p8-fmb*p7p10+fmb*p7p8-2*fmc*p7p10+2*fmc*p7p8+hbcm*p7p10-hbcm
     . *p7p8)+4*p6p7*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p8-ffmcfmb*
     . hbcm*p2p10+ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*
     . p4p8+ffmcfmb*hbcm*p5p10-ffmcfmb*hbcm*p5p8-2*fmb*p1p10+2*fmb*
     . p1p8-3*fmc*p1p10+3*fmc*p1p8+fmc*p2p10-fmc*p2p8+fmc*p4p10-fmc*
     . p4p8-fmc*p5p10+fmc*p5p8+2*hbcm*p1p10-2*hbcm*p1p8)+4*p3p7*(2*
     . ffmcfmb**2*hbcm*p6p10-2*ffmcfmb**2*hbcm*p6p8-2*ffmcfmb*fmc*
     . p6p10+2*ffmcfmb*fmc*p6p8-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p8+
     . fmc*p6p10-fmc*p6p8)+8*p3p6*(ffmcfmb**2*hbcm*p7p10-ffmcfmb**2*
     . hbcm*p7p8-ffmcfmb*fmb*p7p10+ffmcfmb*fmb*p7p8-2*ffmcfmb*fmc*
     . p7p10+2*ffmcfmb*fmc*p7p8+ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p8)
     . )+4*(ffmcfmb*hbcm*p6p12+ffmcfmb*hbcm*p7p13-fmb*p7p13-fmc*p6p12
     . -2*fmc*p7p13+hbcm*p7p13)
      ans1=w2*(4*p5p7*(-ffmcfmb*hbcm*p6p9+fmc*p6p9)+4*p4p7*(ffmcfmb*
     . hbcm*p6p9-fmc*p6p9)+8*p4p6*(ffmcfmb*hbcm*p7p9-fmb*p7p9-2*fmc*
     . p7p9+hbcm*p7p9)+4*p6p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9-
     . ffmcfmb*hbcm*p4p9+ffmcfmb*hbcm*p5p9-fmc*p1p9+fmc*p2p9+fmc*p4p9
     . -fmc*p5p9)+4*p3p7*(2*ffmcfmb**2*hbcm*p6p9-2*ffmcfmb*fmc*p6p9-
     . ffmcfmb*hbcm*p6p9+fmc*p6p9)+8*p3p6*(ffmcfmb**2*hbcm*p7p9-
     . ffmcfmb*fmb*p7p9-2*ffmcfmb*fmc*p7p9+ffmcfmb*hbcm*p7p9))+w1*(4*
     . p5p7*(ffmcfmb*hbcm*p6p8-fmc*p6p8)+4*p4p7*(-ffmcfmb*hbcm*p6p8+
     . fmc*p6p8)+8*p4p6*(-ffmcfmb*hbcm*p7p8+fmb*p7p8+2*fmc*p7p8-hbcm*
     . p7p8)+4*p6p7*(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p8+ffmcfmb*
     . hbcm*p4p8-ffmcfmb*hbcm*p5p8+2*fmb*p1p8+3*fmc*p1p8-fmc*p2p8-fmc
     . *p4p8+fmc*p5p8-2*hbcm*p1p8)+4*p3p7*(-2*ffmcfmb**2*hbcm*p6p8+2*
     . ffmcfmb*fmc*p6p8+ffmcfmb*hbcm*p6p8-fmc*p6p8)+8*p3p6*(-ffmcfmb
     . **2*hbcm*p7p8+ffmcfmb*fmb*p7p8+2*ffmcfmb*fmc*p7p8-ffmcfmb*hbcm
     . *p7p8))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(37)=w2*ccc*(2*p3p7*(-8*ffmcfmb*hbcm+3*hbcm)+2*(-5*hbcm*p4p7+3
     . *hbcm*p5p7))
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb+8)+4*p6p7*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+8*(-p1p3*p6p7+p2p3*p6p7+p3p4*
     . p6p7-p3p5*p6p7-p3p6*p4p7+p3p6*p5p7-2*p3p7*p4p6))
      b(43)=ccc*(w2*(8*p6p7*(-p1p9+p2p9+p4p9-p5p9)+8*p3p7*(-2*ffmcfmb
     . *p6p9+p6p9)+8*(-2*ffmcfmb*p3p6*p7p9-2*p4p6*p7p9-p4p7*p6p9+p5p7
     . *p6p9))+w1*(8*p6p7*(p1p8-p2p8-p4p8+p5p8)+8*p3p7*(2*ffmcfmb*
     . p6p8-p6p8)+8*(2*ffmcfmb*p3p6*p7p8+2*p4p6*p7p8+p4p7*p6p8-p5p7*
     . p6p8))+w5*(8*p5p7*(p6p10-p6p8)+8*p4p7*(-p6p10+p6p8)+16*p4p6*(-
     . p7p10+p7p8)+8*p6p7*(-p1p10+p1p8+p2p10-p2p8+p4p10-p4p8-p5p10+
     . p5p8)+8*p3p7*(-2*ffmcfmb*p6p10+2*ffmcfmb*p6p8+p6p10-p6p8)+16*
     . p3p6*(-ffmcfmb*p7p10+ffmcfmb*p7p8))+8*(-p6p12-p7p13))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb+4)+4*(-3*p4p6+p5p6))
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+6)+2*(-5*p4p7+3*p5p7))
      b(56)=-ccc
      b(57)=-8*ccc
      b(59)=ccc*hbcm
      b(60)=w2*ccc*(4*p3p6*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p6-hbcm*
     . p5p6))
      b(61)=-8*ccc*hbcm
      b(62)=ccc*(2*p3p6*(3*ffmcfmb*hbcm-fmb+hbcm)+8*hbcm*p4p6)
      b(63)=w5*ccc*((p3p7*p3p6)*(-32*ffmcfmb+8)+4*p6p7*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+8*(-p1p3*p6p7+p2p3*p6p7+p3p4*
     . p6p7-p3p5*p6p7-p3p6*p4p7+p3p6*p5p7-2*p3p7*p4p6))
      b(64)=w5*ccc*(2*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+4*(2*
     . ffmcfmb*p3p6*p3p7-2*ffmcfmb*p3p6*p4p7+6*ffmcfmb*p3p6*p5p7-2*
     . p1p2*p6p7+2*p1p3*p6p7-2*p1p4*p6p7+2*p1p5*p6p7+3*p3p7*p4p6-p4p6
     . *p4p7+7*p4p6*p5p7))
      b(65)=ccc*(w5*((p3p7*p3p6)*(32*ffmcfmb-8)+4*p6p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+8*(p1p3*p6p7-p2p3*p6p7-p3p4*
     . p6p7+p3p5*p6p7+p3p6*p4p7-p3p6*p5p7+2*p3p7*p4p6))+w1*((p3p7*
     . p3p6)*(32*ffmcfmb-8)+4*p6p7*(-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*
     . hbcm+hbcm2)+8*(p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+p3p5*p6p7+p3p6*
     . p4p7-p3p6*p5p7+2*p3p7*p4p6)))
      ans=ccc*(w2*(8*p6p7*(-hbcm*p1p9+hbcm*p2p9+hbcm*p4p9-hbcm*p5p9)+
     . 8*p3p7*(-2*ffmcfmb*hbcm*p6p9+hbcm*p6p9)+8*(-2*ffmcfmb*hbcm*
     . p3p6*p7p9-2*hbcm*p4p6*p7p9-hbcm*p4p7*p6p9+hbcm*p5p7*p6p9))+w1*
     . (8*p6p7*(hbcm*p1p8-hbcm*p2p8-hbcm*p4p8+hbcm*p5p8)+8*p3p7*(2*
     . ffmcfmb*hbcm*p6p8-hbcm*p6p8)+8*(2*ffmcfmb*hbcm*p3p6*p7p8+2*
     . hbcm*p4p6*p7p8+hbcm*p4p7*p6p8-hbcm*p5p7*p6p8))+w5*(8*p5p7*(
     . hbcm*p6p10-hbcm*p6p8)+8*p4p7*(-hbcm*p6p10+hbcm*p6p8)+16*p4p6*(
     . -hbcm*p7p10+hbcm*p7p8)+8*p6p7*(-hbcm*p1p10+hbcm*p1p8+hbcm*
     . p2p10-hbcm*p2p8+hbcm*p4p10-hbcm*p4p8-hbcm*p5p10+hbcm*p5p8)+8*
     . p3p7*(-2*ffmcfmb*hbcm*p6p10+2*ffmcfmb*hbcm*p6p8+hbcm*p6p10-
     . hbcm*p6p8)+16*p3p6*(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8))+8*
     . (-hbcm*p6p12-hbcm*p7p13))
      b(66)=ans
      b(67)=w5*ccc*(p3p6*(-16*ffmcfmb+4)+4*(-3*p4p6+p5p6))
      b(68)=ccc*(w5*(p3p6*(16*ffmcfmb-4)+4*(3*p4p6-p5p6))+w1*(p3p6*(
     . 16*ffmcfmb-4)+4*(3*p4p6-p5p6)))
      b(69)=w5*ccc*(p3p7*(-16*ffmcfmb+6)+2*(-5*p4p7+3*p5p7))
      b(70)=w5*ccc*((p3p7*p3p4)*(4*ffmcfmb-4)+(p3p7*p1p3)*(12*ffmcfmb
     . -4)+2*p4p7*(ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm+hbcm2)+2*p3p7*(3
     . *ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2)+4*(-ffmcfmb*hbcm2*p5p7-ffmcfmb*p2p3*p3p7+ffmcfmb*p3p5*
     . p3p7+2*p1p3*p4p7-p1p3*p5p7-p2p3*p4p7-p3p4*p5p7+p3p5*p4p7))
      b(71)=ccc*(w5*(p3p7*(16*ffmcfmb-6)+2*(5*p4p7-3*p5p7))+w1*(p3p7*
     . (16*ffmcfmb-6)+2*(5*p4p7-3*p5p7)))
      b(72)=w5*ccc*(p3p4*(2*ffmcfmb-2)+p1p3*(2*ffmcfmb-2)+2*(ffmcfmb
     . **2*hbcm2-ffmcfmb*hbcm2-ffmcfmb*p2p3-ffmcfmb*p3p5-p1p2-p1p5-
     . p2p4-p4p5))+ccc
      b(74)=ccc*(4*w5*p6p7*(-fmb-fmc+hbcm)+4*w1*p6p7*(-fmb-fmc+hbcm))
      b(79)=4*ccc*w2*p6p7*(fmb+fmc-hbcm)
      b(80)=w2*ccc*(2*p4p7*(ffmcfmb*hbcm-fmb-2*fmc+hbcm)+2*p3p7*(
     . ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm))
      b(81)=4*ccc*w5*p6p7*(fmb+fmc-hbcm)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*hbcm*p5p7)
      b(84)=4*ccc*(p3p7-p4p7+3*p5p7)
      b(85)=ccc*(-3*ffmcfmb*hbcm2+6*fmb*hbcm-7*fmc*hbcm-2*hbcm2-4*
     . p1p3+4*p2p3-12*p3p4+4*p3p5)
      b(86)=-8*ccc
      b(87)=ccc*(-ffmcfmb*hbcm+6*fmb+7*fmc-6*hbcm)
      b(88)=w5*ccc*(2*p4p7*(ffmcfmb*hbcm-fmb-2*fmc+hbcm)+2*p3p7*(
     . ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm))
      b(89)=-8*ccc*hbcm
      b(91)=ccc*(w5*(2*p4p7*(-ffmcfmb*hbcm+fmb+2*fmc-hbcm)+2*p3p7*(-
     . ffmcfmb**2*hbcm+ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm))+w1*(2
     . *p4p7*(-ffmcfmb*hbcm+fmb+2*fmc-hbcm)+2*p3p7*(-ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm)))
      b(92)=ccc*(3*ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm+hbcm2+6*p1p3-2*
     . p2p3+2*p3p4+2*p3p5)
      b(93)=8*ccc*p3p6
      b(94)=ccc*(-ffmcfmb*hbcm+fmb+2*fmc-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)+0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp35_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      ans=ccc*(w11*((p5p6*p3p7)*(-8*ffmcfmb+4)+(p5p7*p3p6)*(-28*
     . ffmcfmb+20)+(p4p7*p3p6)*(4*ffmcfmb-4)+(p3p7*p3p6)*(-20*ffmcfmb
     . +12)+(p6p7*p3p4)*(-4*ffmcfmb-4)+(p6p7*p2p3)*(-4*ffmcfmb+12)+2*
     . p6p7*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm-2*fmc2)+4*(ffmcfmb*p1p3*p6p7+
     . ffmcfmb*p3p5*p6p7-2*ffmcfmb*p3p7*p4p6+p1p2*p6p7+p1p4*p6p7+2*
     . p1p5*p6p7+2*p2p4*p6p7+p2p5*p6p7+p4p5*p6p7-2*p4p6*p5p7-3*p4p7*
     . p5p6+3*p5p6*p5p7))+w1*((p5p6*p3p7)*(-8*ffmcfmb+4)+(p5p7*p3p6)*
     . (-28*ffmcfmb+20)+(p4p7*p3p6)*(4*ffmcfmb-4)+(p3p7*p3p6)*(-20*
     . ffmcfmb+12)+(p6p7*p3p4)*(-4*ffmcfmb-4)+(p6p7*p2p3)*(-4*ffmcfmb
     . +12)+2*p6p7*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm-2*fmc2)+4*(ffmcfmb*p1p3*
     . p6p7+ffmcfmb*p3p5*p6p7-2*ffmcfmb*p3p7*p4p6+p1p2*p6p7+p1p4*p6p7
     . +2*p1p5*p6p7+2*p2p4*p6p7+p2p5*p6p7+p4p5*p6p7-2*p4p6*p5p7-3*
     . p4p7*p5p6+3*p5p6*p5p7))+2*p6p7)
      b(1)=ans
      b(2)=ccc*(w2*((p5p6*p3p7)*(8*ffmcfmb-4)+(p5p7*p3p6)*(28*ffmcfmb
     . -20)+(p4p7*p3p6)*(-4*ffmcfmb+4)+(p3p7*p3p6)*(20*ffmcfmb-12)+(
     . p6p7*p3p4)*(4*ffmcfmb+4)+(p6p7*p2p3)*(4*ffmcfmb-12)+2*p6p7*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+4*(-ffmcfmb*p1p3*p6p7-ffmcfmb*
     . p3p5*p6p7+2*ffmcfmb*p3p7*p4p6-p1p2*p6p7-p1p4*p6p7-2*p1p5*p6p7-
     . 2*p2p4*p6p7-p2p5*p6p7-p4p5*p6p7+2*p4p6*p5p7+3*p4p7*p5p6-3*p5p6
     . *p5p7))-2*p6p7)
      ans2=w11*(8*p5p7*(p5p11-p5p8)+4*p4p5*(-p7p11+p7p8)+4*p2p5*(-
     . p7p11+p7p8)+4*p1p4*(-p7p11+p7p8)+4*p1p2*(-p7p11+p7p8)+8*p4p7*(
     . -p1p11+p1p8+p2p11-p2p8-p4p11+p4p8)+4*p3p5*(-ffmcfmb*p7p11+
     . ffmcfmb*p7p8)+4*p3p4*(ffmcfmb*p7p11-ffmcfmb*p7p8-p7p11+p7p8)+4
     . *p2p3*(ffmcfmb*p7p11-ffmcfmb*p7p8-p7p11+p7p8)+4*p1p3*(-ffmcfmb
     . *p7p11+ffmcfmb*p7p8)+8*p3p7*(-ffmcfmb*p1p11+ffmcfmb*p1p8+
     . ffmcfmb*p2p11-ffmcfmb*p2p8-ffmcfmb*p4p11+ffmcfmb*p4p8-ffmcfmb*
     . p5p11+ffmcfmb*p5p8+p5p11-p5p8)+4*(ffmcfmb**2*hbcm2*p7p11-
     . ffmcfmb**2*hbcm2*p7p8-ffmcfmb*hbcm2*p7p11+ffmcfmb*hbcm2*p7p8))
     . +2*(-2*p1p12+2*p2p12-2*p4p12-2*p5p12-p7p10+p7p11+p7p8-p7p9)
      ans1=w2*(8*p4p7*(-p1p9+p2p9-p4p9)+4*p3p4*(ffmcfmb*p7p9-p7p9)+4*
     . p2p3*(ffmcfmb*p7p9-p7p9)+8*p3p7*(-ffmcfmb*p1p9+ffmcfmb*p2p9-
     . ffmcfmb*p4p9-ffmcfmb*p5p9+p5p9)+4*(ffmcfmb**2*hbcm2*p7p9-
     . ffmcfmb*hbcm2*p7p9-ffmcfmb*p1p3*p7p9-ffmcfmb*p3p5*p7p9-p1p2*
     . p7p9-p1p4*p7p9-p2p5*p7p9-p4p5*p7p9+2*p5p7*p5p9))+w1*(8*p4p7*(
     . p1p8-p2p8+p4p8)+4*p3p4*(-ffmcfmb*p7p8+p7p8)+4*p2p3*(-ffmcfmb*
     . p7p8+p7p8)+8*p3p7*(ffmcfmb*p1p8-ffmcfmb*p2p8+ffmcfmb*p4p8+
     . ffmcfmb*p5p8-p5p8)+4*(-ffmcfmb**2*hbcm2*p7p8+ffmcfmb*hbcm2*
     . p7p8+ffmcfmb*p1p3*p7p8+ffmcfmb*p3p5*p7p8+p1p2*p7p8+p1p4*p7p8+
     . p2p5*p7p8+p4p5*p7p8-2*p5p7*p5p8))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans=ccc*(w11*((p3p7*p3p5)*(-12*ffmcfmb+8)+(p3p7*p3p4)*(-4*
     . ffmcfmb-4)+(p3p7*p2p3)*(12*ffmcfmb-4)+2*p5p7*(-ffmcfmb*hbcm2+2
     . *fmb*hbcm-fmc*hbcm+2*hbcm2)+2*p3p7*(-ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2+2*fmb*hbcm-
     . fmc*hbcm+2*hbcm2)+4*(-2*ffmcfmb*hbcm2*p4p7-3*ffmcfmb*p1p3*p3p7
     . -p1p2*p3p7-2*p1p3*p4p7-p1p4*p3p7+2*p2p3*p4p7-p2p5*p3p7-2*p3p4*
     . p4p7+2*p3p5*p5p7-p3p7*p4p5))+w1*((p3p7*p3p5)*(-12*ffmcfmb+8)+(
     . p3p7*p3p4)*(-4*ffmcfmb-4)+(p3p7*p2p3)*(12*ffmcfmb-4)+2*p5p7*(-
     . ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm+2*hbcm2)+2*p3p7*(-ffmcfmb**2
     . *hbcm2-2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2+2*
     . fmb*hbcm-fmc*hbcm+2*hbcm2)+4*(-2*ffmcfmb*hbcm2*p4p7-3*ffmcfmb*
     . p1p3*p3p7-p1p2*p3p7-2*p1p3*p4p7-p1p4*p3p7+2*p2p3*p4p7-p2p5*
     . p3p7-2*p3p4*p4p7+2*p3p5*p5p7-p3p7*p4p5))-2*p3p7)
      b(4)=ans
      b(5)=ccc*(w2*((p3p7*p3p5)*(12*ffmcfmb-8)+(p3p7*p3p4)*(4*ffmcfmb
     . +4)+(p3p7*p2p3)*(-12*ffmcfmb+4)+2*p5p7*(ffmcfmb*hbcm2-2*fmb*
     . hbcm+fmc*hbcm-2*hbcm2)+2*p3p7*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm+5*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-2*
     . hbcm2)+4*(2*ffmcfmb*hbcm2*p4p7+3*ffmcfmb*p1p3*p3p7+p1p2*p3p7+2
     . *p1p3*p4p7+p1p4*p3p7-2*p2p3*p4p7+p2p5*p3p7+2*p3p4*p4p7-2*p3p5*
     . p5p7+p3p7*p4p5))+2*p3p7)
      ans2=w11*(8*p5p7*(-hbcm*p5p11+hbcm*p5p8)+4*p4p5*(hbcm*p7p11-
     . hbcm*p7p8)+4*p2p5*(hbcm*p7p11-hbcm*p7p8)+4*p1p4*(hbcm*p7p11-
     . hbcm*p7p8)+4*p1p2*(hbcm*p7p11-hbcm*p7p8)+8*p4p7*(hbcm*p1p11-
     . hbcm*p1p8-hbcm*p2p11+hbcm*p2p8+hbcm*p4p11-hbcm*p4p8)+4*p3p5*(
     . ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*p7p8)+4*p3p4*(-ffmcfmb*hbcm*
     . p7p11+ffmcfmb*hbcm*p7p8+hbcm*p7p11-hbcm*p7p8)+4*p2p3*(-ffmcfmb
     . *hbcm*p7p11+ffmcfmb*hbcm*p7p8+hbcm*p7p11-hbcm*p7p8)+4*p1p3*(
     . ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*p7p8)+8*p3p7*(ffmcfmb*hbcm*
     . p1p11-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p2p11+ffmcfmb*hbcm*p2p8+
     . ffmcfmb*hbcm*p4p11-ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p11-
     . ffmcfmb*hbcm*p5p8-hbcm*p5p11+hbcm*p5p8)+4*(-ffmcfmb**2*hbcm3*
     . p7p11+ffmcfmb**2*hbcm3*p7p8+ffmcfmb*hbcm3*p7p11-ffmcfmb*hbcm3*
     . p7p8))+2*(2*hbcm*p1p12-2*hbcm*p2p12+2*hbcm*p4p12+2*hbcm*p5p12+
     . hbcm*p7p10-hbcm*p7p11-hbcm*p7p8+hbcm*p7p9)
      ans1=w2*(8*p4p7*(hbcm*p1p9-hbcm*p2p9+hbcm*p4p9)+4*p3p4*(-
     . ffmcfmb*hbcm*p7p9+hbcm*p7p9)+4*p2p3*(-ffmcfmb*hbcm*p7p9+hbcm*
     . p7p9)+8*p3p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+ffmcfmb*hbcm
     . *p4p9+ffmcfmb*hbcm*p5p9-hbcm*p5p9)+4*(-ffmcfmb**2*hbcm3*p7p9+
     . ffmcfmb*hbcm*p1p3*p7p9+ffmcfmb*hbcm*p3p5*p7p9+ffmcfmb*hbcm3*
     . p7p9+hbcm*p1p2*p7p9+hbcm*p1p4*p7p9+hbcm*p2p5*p7p9+hbcm*p4p5*
     . p7p9-2*hbcm*p5p7*p5p9))+w1*(8*p4p7*(-hbcm*p1p8+hbcm*p2p8-hbcm*
     . p4p8)+4*p3p4*(ffmcfmb*hbcm*p7p8-hbcm*p7p8)+4*p2p3*(ffmcfmb*
     . hbcm*p7p8-hbcm*p7p8)+8*p3p7*(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*
     . p2p8-ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p8+hbcm*p5p8)+4*(ffmcfmb
     . **2*hbcm3*p7p8-ffmcfmb*hbcm*p1p3*p7p8-ffmcfmb*hbcm*p3p5*p7p8-
     . ffmcfmb*hbcm3*p7p8-hbcm*p1p2*p7p8-hbcm*p1p4*p7p8-hbcm*p2p5*
     . p7p8-hbcm*p4p5*p7p8+2*hbcm*p5p7*p5p8))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w11*(4*p4p5*(-p6p11+p6p8)+4*p2p5*(-p6p11+p6p8)+4*p5p6*(-2*
     . p2p11+2*p2p8+3*p3p11-3*p3p8+4*p4p11-4*p4p8)+8*p4p6*(p2p11-p2p8
     . -2*p3p11+2*p3p8-2*p4p11+2*p4p8)+4*p1p4*(-p6p11+p6p8)+4*p1p2*(-
     . p6p11+p6p8)+4*p3p5*(-ffmcfmb*p6p11+ffmcfmb*p6p8)+4*p3p4*(
     . ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+4*p2p3*(ffmcfmb*p6p11-
     . ffmcfmb*p6p8-p6p11+p6p8)+4*p3p6*(4*ffmcfmb*p2p11-4*ffmcfmb*
     . p2p8-7*ffmcfmb*p3p11+7*ffmcfmb*p3p8-8*ffmcfmb*p4p11+8*ffmcfmb*
     . p4p8-2*p2p11+2*p2p8+3*p3p11-3*p3p8+4*p4p11-4*p4p8)+4*p1p3*(-
     . ffmcfmb*p6p11+ffmcfmb*p6p8)+4*(ffmcfmb**2*hbcm2*p6p11-ffmcfmb
     . **2*hbcm2*p6p8-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8))+2*(4*
     . p2p13-7*p3p13-8*p4p13-p6p10+p6p11+p6p8-p6p9)
      ans1=w2*(4*p5p6*(-2*p2p9+3*p3p9+4*p4p9)+8*p4p6*(p2p9-2*p3p9-2*
     . p4p9)+4*p3p4*(ffmcfmb*p6p9-p6p9)+4*p3p6*(4*ffmcfmb*p2p9-7*
     . ffmcfmb*p3p9-8*ffmcfmb*p4p9-2*p2p9+3*p3p9+4*p4p9)+4*p2p3*(
     . ffmcfmb*p6p9-p6p9)+4*(ffmcfmb**2*hbcm2*p6p9-ffmcfmb*hbcm2*p6p9
     . -ffmcfmb*p1p3*p6p9-ffmcfmb*p3p5*p6p9-p1p2*p6p9-p1p4*p6p9-p2p5*
     . p6p9-p4p5*p6p9))+w1*(4*p5p6*(2*p2p8-3*p3p8-4*p4p8)+8*p4p6*(-
     . p2p8+2*p3p8+2*p4p8)+4*p3p4*(-ffmcfmb*p6p8+p6p8)+4*p3p6*(-4*
     . ffmcfmb*p2p8+7*ffmcfmb*p3p8+8*ffmcfmb*p4p8+2*p2p8-3*p3p8-4*
     . p4p8)+4*p2p3*(-ffmcfmb*p6p8+p6p8)+4*(-ffmcfmb**2*hbcm2*p6p8+
     . ffmcfmb*hbcm2*p6p8+ffmcfmb*p1p3*p6p8+ffmcfmb*p3p5*p6p8+p1p2*
     . p6p8+p1p4*p6p8+p2p5*p6p8+p4p5*p6p8))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans=ccc*(w11*((p3p6*p3p4)*(-28*ffmcfmb+12)+(p3p6*p2p3)*(20*
     . ffmcfmb-12)+2*p5p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*
     . hbcm2)+2*p4p6*(-ffmcfmb*hbcm2+4*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+2
     . *p3p6*(-ffmcfmb**2*hbcm2+7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm
     . -7*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*hbcm2)+4*(-ffmcfmb*
     . p1p3*p3p6-ffmcfmb*p3p5*p3p6-p1p2*p3p6-p1p4*p3p6+2*p2p3*p4p6-2*
     . p2p3*p5p6-p2p5*p3p6-4*p3p4*p4p6+4*p3p4*p5p6-p3p6*p4p5))+w1*((
     . p3p6*p3p4)*(-28*ffmcfmb+12)+(p3p6*p2p3)*(20*ffmcfmb-12)+2*p5p6
     . *(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*hbcm2)+2*p4p6*(-
     . ffmcfmb*hbcm2+4*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+2*p3p6*(-ffmcfmb
     . **2*hbcm2+7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm-7*ffmcfmb*
     . hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*hbcm2)+4*(-ffmcfmb*p1p3*p3p6-
     . ffmcfmb*p3p5*p3p6-p1p2*p3p6-p1p4*p3p6+2*p2p3*p4p6-2*p2p3*p5p6-
     . p2p5*p3p6-4*p3p4*p4p6+4*p3p4*p5p6-p3p6*p4p5))-2*p3p6)
      b(8)=ans
      b(9)=ccc*(w2*((p3p6*p3p4)*(28*ffmcfmb-12)+(p3p6*p2p3)*(-20*
     . ffmcfmb+12)+2*p5p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-3*
     . hbcm2)+2*p4p6*(ffmcfmb*hbcm2-4*fmb*hbcm+3*fmc*hbcm+4*hbcm2)+2*
     . p3p6*(ffmcfmb**2*hbcm2-7*ffmcfmb*fmb*hbcm+6*ffmcfmb*fmc*hbcm+7
     . *ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-3*hbcm2)+4*(ffmcfmb*p1p3*
     . p3p6+ffmcfmb*p3p5*p3p6+p1p2*p3p6+p1p4*p3p6-2*p2p3*p4p6+2*p2p3*
     . p5p6+p2p5*p3p6+4*p3p4*p4p6-4*p3p4*p5p6+p3p6*p4p5))+2*p3p6)
      ans4=4*(-ffmcfmb**2*hbcm3*p6p11+ffmcfmb**2*hbcm3*p6p8+ffmcfmb*
     . hbcm3*p6p11-ffmcfmb*hbcm3*p6p8)
      ans3=4*p4p5*(hbcm*p6p11-hbcm*p6p8)+4*p2p5*(hbcm*p6p11-hbcm*p6p8
     . )+4*p1p4*(hbcm*p6p11-hbcm*p6p8)+4*p1p2*(hbcm*p6p11-hbcm*p6p8)+
     . 4*p5p6*(3*fmb*p3p8+3*fmc*p3p8+3*hbcm*p1p11-3*hbcm*p1p8+2*hbcm*
     . p2p11-2*hbcm*p2p8-3*hbcm*p3p11-4*hbcm*p4p11+4*hbcm*p4p8)+4*
     . p4p6*(-ffmcfmb*hbcm*p3p8-4*fmb*p3p8-3*fmc*p3p8-5*hbcm*p1p11+5*
     . hbcm*p1p8-2*hbcm*p2p11+2*hbcm*p2p8+4*hbcm*p3p11+4*hbcm*p4p11-4
     . *hbcm*p4p8)+4*p3p5*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8)+4*
     . p3p4*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8+hbcm*p6p11-hbcm*
     . p6p8)+4*p2p3*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8+hbcm*p6p11
     . -hbcm*p6p8)+4*p1p3*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8)+4*
     . p3p6*(-ffmcfmb**2*hbcm*p3p8-7*ffmcfmb*fmb*p3p8-6*ffmcfmb*fmc*
     . p3p8-8*ffmcfmb*hbcm*p1p11+8*ffmcfmb*hbcm*p1p8-4*ffmcfmb*hbcm*
     . p2p11+4*ffmcfmb*hbcm*p2p8+7*ffmcfmb*hbcm*p3p11+8*ffmcfmb*hbcm*
     . p4p11-8*ffmcfmb*hbcm*p4p8+3*fmb*p3p8+3*fmc*p3p8+3*hbcm*p1p11-3
     . *hbcm*p1p8+2*hbcm*p2p11-2*hbcm*p2p8-3*hbcm*p3p11-4*hbcm*p4p11+
     . 4*hbcm*p4p8)+ans4
      ans2=w11*ans3
      ans5=w1*(4*p5p6*(3*fmb*p3p8+3*fmc*p3p8-3*hbcm*p1p8-2*hbcm*p2p8+
     . 4*hbcm*p4p8)+4*p4p6*(-ffmcfmb*hbcm*p3p8-4*fmb*p3p8-3*fmc*p3p8+
     . 5*hbcm*p1p8+2*hbcm*p2p8-4*hbcm*p4p8)+4*p3p4*(ffmcfmb*hbcm*p6p8
     . -hbcm*p6p8)+4*p2p3*(ffmcfmb*hbcm*p6p8-hbcm*p6p8)+4*p3p6*(-
     . ffmcfmb**2*hbcm*p3p8-7*ffmcfmb*fmb*p3p8-6*ffmcfmb*fmc*p3p8+8*
     . ffmcfmb*hbcm*p1p8+4*ffmcfmb*hbcm*p2p8-8*ffmcfmb*hbcm*p4p8+3*
     . fmb*p3p8+3*fmc*p3p8-3*hbcm*p1p8-2*hbcm*p2p8+4*hbcm*p4p8)+4*(
     . ffmcfmb**2*hbcm3*p6p8-ffmcfmb*hbcm*p1p3*p6p8-ffmcfmb*hbcm*p3p5
     . *p6p8-ffmcfmb*hbcm3*p6p8-hbcm*p1p2*p6p8-hbcm*p1p4*p6p8-hbcm*
     . p2p5*p6p8-hbcm*p4p5*p6p8))+2*(-4*hbcm*p2p13+7*hbcm*p3p13+8*
     . hbcm*p4p13+hbcm*p6p10-hbcm*p6p11-hbcm*p6p8+hbcm*p6p9)
      ans1=w2*(4*p5p6*(3*hbcm*p1p9+2*hbcm*p2p9-3*hbcm*p3p9-4*hbcm*
     . p4p9)+4*p4p6*(-5*hbcm*p1p9-2*hbcm*p2p9+4*hbcm*p3p9+4*hbcm*p4p9
     . )+4*p3p4*(-ffmcfmb*hbcm*p6p9+hbcm*p6p9)+4*p2p3*(-ffmcfmb*hbcm*
     . p6p9+hbcm*p6p9)+4*p3p6*(-8*ffmcfmb*hbcm*p1p9-4*ffmcfmb*hbcm*
     . p2p9+7*ffmcfmb*hbcm*p3p9+8*ffmcfmb*hbcm*p4p9+3*hbcm*p1p9+2*
     . hbcm*p2p9-3*hbcm*p3p9-4*hbcm*p4p9)+4*(-ffmcfmb**2*hbcm3*p6p9+
     . ffmcfmb*hbcm*p1p3*p6p9+ffmcfmb*hbcm*p3p5*p6p9+ffmcfmb*hbcm3*
     . p6p9+hbcm*p1p2*p6p9+hbcm*p1p4*p6p9+hbcm*p2p5*p6p9+hbcm*p4p5*
     . p6p9))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      ans2=w1*(4*(p6p7*p2p3)*(2*fmb+fmc+hbcm)+4*(p5p6*p3p7)*(-ffmcfmb
     . *hbcm+fmb+2*fmc)+8*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-fmb-fmc+hbcm)+
     . 4*(p5p7*p3p6)*(-6*ffmcfmb*hbcm+2*fmb+fmc+3*hbcm)+4*(p4p7*p3p6)
     . *(-ffmcfmb*hbcm+2*fmb-hbcm)+4*(p6p7*p3p5)*(2*ffmcfmb*hbcm-fmc)
     . +4*(p6p7*p3p4)*(-2*ffmcfmb*hbcm-2*fmb-fmc+hbcm)+2*p6p7*(-3*
     . ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*
     . ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*(p3p7*p3p6
     . )*(-2*ffmcfmb**2*hbcm-3*ffmcfmb*fmb-5*ffmcfmb*fmc+3*fmb+3*fmc)
     . +4*(fmc*p1p3*p6p7+hbcm*p1p2*p6p7+hbcm*p1p4*p6p7+2*hbcm*p1p5*
     . p6p7+2*hbcm*p2p4*p6p7+hbcm*p2p5*p6p7+hbcm*p4p5*p6p7-2*hbcm*
     . p4p6*p4p7-2*hbcm*p4p6*p5p7-hbcm*p4p7*p5p6+3*hbcm*p5p6*p5p7))+2
     . *hbcm*p6p7
      ans1=w11*(4*(p6p7*p2p3)*(2*fmb+fmc+hbcm)+4*(p5p6*p3p7)*(-
     . ffmcfmb*hbcm+fmb+2*fmc)+8*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-fmb-fmc
     . +hbcm)+4*(p5p7*p3p6)*(-6*ffmcfmb*hbcm+2*fmb+fmc+3*hbcm)+4*(
     . p4p7*p3p6)*(-ffmcfmb*hbcm+2*fmb-hbcm)+4*(p6p7*p3p5)*(2*ffmcfmb
     . *hbcm-fmc)+4*(p6p7*p3p4)*(-2*ffmcfmb*hbcm-2*fmb-fmc+hbcm)+2*
     . p6p7*(-3*ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2+3*ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*(
     . p3p7*p3p6)*(-2*ffmcfmb**2*hbcm-3*ffmcfmb*fmb-5*ffmcfmb*fmc+3*
     . fmb+3*fmc)+4*(fmc*p1p3*p6p7+hbcm*p1p2*p6p7+hbcm*p1p4*p6p7+2*
     . hbcm*p1p5*p6p7+2*hbcm*p2p4*p6p7+hbcm*p2p5*p6p7+hbcm*p4p5*p6p7-
     . 2*hbcm*p4p6*p4p7-2*hbcm*p4p6*p5p7-hbcm*p4p7*p5p6+3*hbcm*p5p6*
     . p5p7))+ans2
      ans=ccc*ans1
      b(11)=ans
      ans=ccc*(w2*(4*(p6p7*p2p3)*(-2*fmb-fmc-hbcm)+4*(p6p7*p1p3)*(2*
     . fmb+fmc-2*hbcm)+4*(p5p6*p3p7)*(ffmcfmb*hbcm-fmb-2*fmc)+8*(p4p6
     . *p3p7)*(2*ffmcfmb*hbcm+fmb+fmc-hbcm)+4*(p5p7*p3p6)*(6*ffmcfmb*
     . hbcm-2*fmb-fmc-3*hbcm)+4*(p4p7*p3p6)*(ffmcfmb*hbcm-2*fmb+hbcm)
     . +4*(p6p7*p3p5)*(-2*ffmcfmb*hbcm+fmc)+4*(p6p7*p3p4)*(2*ffmcfmb*
     . hbcm+2*fmb+fmc-hbcm)+2*p6p7*(3*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2
     . +2*fmc2*hbcm)+4*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm+3*ffmcfmb*fmb+5
     . *ffmcfmb*fmc-3*fmb-3*fmc)+4*(-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7-2*
     . hbcm*p1p5*p6p7-2*hbcm*p2p4*p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7+
     . 2*hbcm*p4p6*p4p7+2*hbcm*p4p6*p5p7+hbcm*p4p7*p5p6-3*hbcm*p5p6*
     . p5p7))-2*hbcm*p6p7)
      b(12)=ans
      b(13)=ccc*(w11*(p3p4*(2*ffmcfmb-2)+p2p3*(2*ffmcfmb-2)+2*(
     . ffmcfmb**2*hbcm2-ffmcfmb*hbcm2-ffmcfmb*p1p3-ffmcfmb*p3p5-p1p2-
     . p1p4-p2p5-p4p5))+w1*(p3p4*(2*ffmcfmb-2)+p2p3*(2*ffmcfmb-2)+2*(
     . ffmcfmb**2*hbcm2-ffmcfmb*hbcm2-ffmcfmb*p1p3-ffmcfmb*p3p5-p1p2-
     . p1p4-p2p5-p4p5))-1)
      b(14)=w2*ccc*(p3p4*(-2*ffmcfmb+2)+p2p3*(-2*ffmcfmb+2)+2*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*hbcm2+ffmcfmb*p1p3+ffmcfmb*p3p5+p1p2+
     . p1p4+p2p5+p4p5))+ccc
      b(15)=ccc*(w11*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p2p3*(-ffmcfmb*
     . hbcm+hbcm)+2*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p1p3+ffmcfmb*hbcm
     . *p3p5+ffmcfmb*hbcm3+hbcm*p1p2+hbcm*p1p4+hbcm*p2p5+hbcm*p4p5))+
     . w1*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p2p3*(-ffmcfmb*hbcm+hbcm)+2*
     . (-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p1p3+ffmcfmb*hbcm*p3p5+ffmcfmb
     . *hbcm3+hbcm*p1p2+hbcm*p1p4+hbcm*p2p5+hbcm*p4p5))+hbcm)
      b(16)=ccc*(w2*(2*p3p4*(ffmcfmb*hbcm-hbcm)+2*p2p3*(ffmcfmb*hbcm-
     . hbcm)+2*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm*p1p3-ffmcfmb*hbcm*p3p5-
     . ffmcfmb*hbcm3-hbcm*p1p2-hbcm*p1p4-hbcm*p2p5-hbcm*p4p5))-hbcm)
      b(22)=-ccc*hbcm
      b(23)=ccc*(w11*(2*p3p6*(-8*ffmcfmb*hbcm+3*hbcm)+2*(-5*hbcm*p4p6
     . +3*hbcm*p5p6))+w1*(2*p3p6*(-8*ffmcfmb*hbcm+3*hbcm)+2*(-5*hbcm*
     . p4p6+3*hbcm*p5p6)))
      b(26)=ccc*(w11*(4*p3p7*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p7-
     . hbcm*p5p7))+w1*(4*p3p7*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p7-
     . hbcm*p5p7)))
      b(28)=-2*ccc*p3p6
      b(29)=-2*ccc*hbcm*p6p7
      ans4=4*p3p7*(-ffmcfmb**2*hbcm2*p6p9+ffmcfmb*fmc*hbcm*p6p9+3*
     . ffmcfmb*hbcm2*p6p9+2*fmb*hbcm*p6p9-fmc*hbcm*p6p9-2*hbcm2*p6p9)
     . +4*p3p6*(-2*ffmcfmb**2*hbcm2*p7p9+4*ffmcfmb*fmb*hbcm*p7p9-2*
     . ffmcfmb*fmc*hbcm*p7p9-3*ffmcfmb*hbcm2*p7p9-2*fmb*hbcm*p7p9+fmc
     . *hbcm*p7p9+2*hbcm2*p7p9)+16*(ffmcfmb*p1p3*p3p7*p6p9+ffmcfmb*
     . p3p5*p3p7*p6p9+p1p3*p1p9*p6p7+p1p3*p4p7*p6p9-p1p9*p3p5*p6p7+
     . p2p3*p4p6*p7p9-p2p3*p5p6*p7p9+p2p3*p5p7*p6p9-p3p4*p4p6*p7p9+
     . p3p4*p5p6*p7p9-p3p4*p5p7*p6p9+p3p5*p4p7*p6p9-p3p6*p4p7*p5p9-
     . p3p6*p5p7*p5p9)
      ans3=8*(p5p6*p3p7)*(3*p2p9-3*p3p9-3*p4p9-p5p9)+16*(p6p7*p3p4)*(
     . -p2p9+p4p9+p5p9)+16*(p6p7*p2p3)*(p2p9-p4p9-p5p9)+8*(p4p6*p3p7)
     . *(p1p9-4*p2p9+4*p3p9+4*p4p9+2*p5p9)+16*(p3p7*p3p4)*(ffmcfmb*
     . p6p9-p6p9)+16*(p3p6*p3p4)*(-2*ffmcfmb*p7p9+p7p9)+16*(p3p7*p2p3
     . )*(-ffmcfmb*p6p9+p6p9)+16*(p3p6*p2p3)*(2*ffmcfmb*p7p9-p7p9)+8*
     . (p3p7*p3p6)*(ffmcfmb*p1p9-7*ffmcfmb*p2p9+7*ffmcfmb*p3p9+7*
     . ffmcfmb*p4p9+3*ffmcfmb*p5p9+3*p2p9-3*p3p9-3*p4p9-3*p5p9)+4*
     . p5p7*(-ffmcfmb*hbcm2*p6p9+2*fmb*hbcm*p6p9-fmc*hbcm*p6p9-2*
     . hbcm2*p6p9)+4*p5p6*(ffmcfmb*hbcm2*p7p9-2*fmb*hbcm*p7p9+fmc*
     . hbcm*p7p9+2*hbcm2*p7p9)+8*p4p7*(-ffmcfmb*hbcm2*p6p9+fmb*hbcm*
     . p6p9+hbcm2*p6p9)+4*p4p6*(-ffmcfmb*hbcm2*p7p9+2*fmb*hbcm*p7p9-
     . fmc*hbcm*p7p9-2*hbcm2*p7p9)+4*p6p7*(ffmcfmb*hbcm2*p1p9-ffmcfmb
     . *hbcm2*p2p9+ffmcfmb*hbcm2*p4p9+3*ffmcfmb*hbcm2*p5p9-2*fmb*hbcm
     . *p1p9+2*fmb*hbcm*p2p9-2*fmb*hbcm*p4p9+fmc*hbcm*p1p9-fmc*hbcm*
     . p2p9+fmc*hbcm*p4p9+fmc*hbcm*p5p9-2*hbcm2*p1p9-2*hbcm2*p2p9+2*
     . hbcm2*p4p9)+ans4
      ans2=w2*ans3
      ans7=4*p3p7*(ffmcfmb**2*hbcm2*p6p8-ffmcfmb*fmc*hbcm*p6p8-3*
     . ffmcfmb*hbcm2*p6p8-2*fmb*hbcm*p6p8+fmc*hbcm*p6p8+2*hbcm2*p6p8)
     . +4*p3p6*(2*ffmcfmb**2*hbcm2*p7p8-4*ffmcfmb*fmb*hbcm*p7p8+2*
     . ffmcfmb*fmc*hbcm*p7p8+3*ffmcfmb*hbcm2*p7p8+2*fmb*hbcm*p7p8-fmc
     . *hbcm*p7p8-2*hbcm2*p7p8)+16*(-ffmcfmb*p1p3*p3p7*p6p8-ffmcfmb*
     . p3p5*p3p7*p6p8-p1p3*p1p8*p6p7-p1p3*p4p7*p6p8+p1p8*p3p5*p6p7-
     . p2p3*p4p6*p7p8+p2p3*p5p6*p7p8-p2p3*p5p7*p6p8+p3p4*p4p6*p7p8-
     . p3p4*p5p6*p7p8+p3p4*p5p7*p6p8-p3p5*p4p7*p6p8+p3p6*p4p7*p5p8+
     . p3p6*p5p7*p5p8)
      ans6=8*(p5p6*p3p7)*(-3*p2p8+3*p3p8+3*p4p8+p5p8)+16*(p6p7*p3p4)*
     . (p2p8-p4p8-p5p8)+16*(p6p7*p2p3)*(-p2p8+p4p8+p5p8)+8*(p4p6*p3p7
     . )*(-p1p8+4*p2p8-4*p3p8-4*p4p8-2*p5p8)+16*(p3p7*p3p4)*(-ffmcfmb
     . *p6p8+p6p8)+16*(p3p6*p3p4)*(2*ffmcfmb*p7p8-p7p8)+16*(p3p7*p2p3
     . )*(ffmcfmb*p6p8-p6p8)+16*(p3p6*p2p3)*(-2*ffmcfmb*p7p8+p7p8)+8*
     . (p3p7*p3p6)*(-ffmcfmb*p1p8+7*ffmcfmb*p2p8-7*ffmcfmb*p3p8-7*
     . ffmcfmb*p4p8-3*ffmcfmb*p5p8-3*p2p8+3*p3p8+3*p4p8+3*p5p8)+4*
     . p5p7*(ffmcfmb*hbcm2*p6p8-2*fmb*hbcm*p6p8+fmc*hbcm*p6p8+2*hbcm2
     . *p6p8)+4*p5p6*(-ffmcfmb*hbcm2*p7p8+2*fmb*hbcm*p7p8-fmc*hbcm*
     . p7p8-2*hbcm2*p7p8)+8*p4p7*(ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p8-
     . hbcm2*p6p8)+4*p4p6*(ffmcfmb*hbcm2*p7p8-2*fmb*hbcm*p7p8+fmc*
     . hbcm*p7p8+2*hbcm2*p7p8)+4*p6p7*(-ffmcfmb*hbcm2*p1p8+ffmcfmb*
     . hbcm2*p2p8-ffmcfmb*hbcm2*p4p8-3*ffmcfmb*hbcm2*p5p8+2*fmb*hbcm*
     . p1p8-2*fmb*hbcm*p2p8+2*fmb*hbcm*p4p8-fmc*hbcm*p1p8+fmc*hbcm*
     . p2p8-fmc*hbcm*p4p8-fmc*hbcm*p5p8+2*hbcm2*p1p8+2*hbcm2*p2p8-2*
     . hbcm2*p4p8)+ans7
      ans5=w1*ans6
      ans12=8*(ffmcfmb*p1p3*p3p6*p7p11+ffmcfmb*p3p5*p3p6*p7p11+p1p2*
     . p3p6*p7p11+p1p4*p3p6*p7p11-2*p2p3*p4p6*p7p8+2*p2p3*p5p6*p7p8+
     . p2p5*p3p6*p7p11+p3p6*p4p5*p7p11)
      ans11=4*p6p7*(ffmcfmb*hbcm2*p1p11-ffmcfmb*hbcm2*p1p8-ffmcfmb*
     . hbcm2*p2p11+ffmcfmb*hbcm2*p2p8+ffmcfmb*hbcm2*p4p11-ffmcfmb*
     . hbcm2*p4p8+3*ffmcfmb*hbcm2*p5p11-3*ffmcfmb*hbcm2*p5p8-2*fmb*
     . hbcm*p1p11+2*fmb*hbcm*p1p8+2*fmb*hbcm*p2p11-2*fmb*hbcm*p2p8-2*
     . fmb*hbcm*p4p11+2*fmb*hbcm*p4p8+fmc*hbcm*p1p11-fmc*hbcm*p1p8-
     . fmc*hbcm*p2p11+fmc*hbcm*p2p8+fmc*hbcm*p4p11-fmc*hbcm*p4p8+fmc*
     . hbcm*p5p11-fmc*hbcm*p5p8-2*hbcm2*p1p11+2*hbcm2*p1p8-2*hbcm2*
     . p2p11+2*hbcm2*p2p8+2*hbcm2*p4p11-2*hbcm2*p4p8)+4*p3p7*(-
     . ffmcfmb**2*hbcm2*p6p11+ffmcfmb**2*hbcm2*p6p8+ffmcfmb*fmc*hbcm*
     . p6p11-ffmcfmb*fmc*hbcm*p6p8+3*ffmcfmb*hbcm2*p6p11-3*ffmcfmb*
     . hbcm2*p6p8+2*fmb*hbcm*p6p11-2*fmb*hbcm*p6p8-fmc*hbcm*p6p11+fmc
     . *hbcm*p6p8-2*hbcm2*p6p11+2*hbcm2*p6p8)+4*p3p6*(-ffmcfmb**2*
     . hbcm2*p7p11+2*ffmcfmb**2*hbcm2*p7p8-3*ffmcfmb*fmb*hbcm*p7p11-4
     . *ffmcfmb*fmb*hbcm*p7p8+4*ffmcfmb*fmc*hbcm*p7p11+2*ffmcfmb*fmc*
     . hbcm*p7p8+4*ffmcfmb*hbcm2*p7p11+3*ffmcfmb*hbcm2*p7p8+fmb*hbcm*
     . p7p11+2*fmb*hbcm*p7p8-2*fmc*hbcm*p7p11-fmc*hbcm*p7p8-hbcm2*
     . p7p11-2*hbcm2*p7p8)+ans12
      ans10=8*(p3p7*p3p6)*(ffmcfmb*p1p11-ffmcfmb*p1p8-7*ffmcfmb*p2p11
     . +7*ffmcfmb*p2p8+7*ffmcfmb*p3p11-7*ffmcfmb*p3p8+7*ffmcfmb*p4p11
     . -7*ffmcfmb*p4p8+3*ffmcfmb*p5p11-3*ffmcfmb*p5p8+3*p2p11-3*p2p8-
     . 3*p3p11+3*p3p8-3*p4p11+3*p4p8-3*p5p11+3*p5p8)+4*p5p7*(-ffmcfmb
     . *hbcm2*p6p11+ffmcfmb*hbcm2*p6p8+2*fmb*hbcm*p6p11-2*fmb*hbcm*
     . p6p8-fmc*hbcm*p6p11+fmc*hbcm*p6p8-2*hbcm2*p6p11+2*hbcm2*p6p8)+
     . 4*p5p6*(-ffmcfmb*hbcm2*p7p11-ffmcfmb*hbcm2*p7p8+fmb*hbcm*p7p11
     . +2*fmb*hbcm*p7p8-2*fmc*hbcm*p7p11-fmc*hbcm*p7p8-hbcm2*p7p11-2*
     . hbcm2*p7p8)+8*p4p7*(-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8+
     . fmb*hbcm*p6p11-fmb*hbcm*p6p8+hbcm2*p6p11-hbcm2*p6p8)+4*p4p6*(
     . ffmcfmb*hbcm2*p7p8-2*fmb*hbcm*p7p11-2*fmb*hbcm*p7p8+2*fmc*hbcm
     . *p7p11+fmc*hbcm*p7p8+2*hbcm2*p7p11+2*hbcm2*p7p8)+ans11
      ans9=16*(p5p7*p3p6)*(-p5p11+p5p8)+16*(p4p7*p3p6)*(-p5p11+p5p8)+
     . 16*(p4p7*p3p5)*(p6p11-p6p8)+16*(p5p7*p3p4)*(-p6p11+p6p8)+16*(
     . p5p6*p3p4)*(-p7p11-p7p8)+16*(p4p6*p3p4)*(p7p11+p7p8)+16*(p5p7*
     . p2p3)*(p6p11-p6p8)+8*(p5p6*p3p7)*(3*p2p11-3*p2p8-3*p3p11+3*
     . p3p8-3*p4p11+3*p4p8-p5p11+p5p8)+16*(p6p7*p3p4)*(-p2p11+p2p8+
     . p4p11-p4p8+p5p11-p5p8)+16*(p6p7*p2p3)*(p2p11-p2p8-p4p11+p4p8-
     . p5p11+p5p8)+16*(p4p7*p1p3)*(p6p11-p6p8)+8*(p4p6*p3p7)*(p1p11-
     . p1p8-4*p2p11+4*p2p8+4*p3p11-4*p3p8+4*p4p11-4*p4p8+2*p5p11-2*
     . p5p8)+16*(p6p7*p3p5)*(-p1p11+p1p8)+16*(p6p7*p1p3)*(p1p11-p1p8)
     . +16*(p3p7*p3p5)*(ffmcfmb*p6p11-ffmcfmb*p6p8)+16*(p3p7*p3p4)*(
     . ffmcfmb*p6p11-ffmcfmb*p6p8-p6p11+p6p8)+8*(p3p6*p3p4)*(3*
     . ffmcfmb*p7p11+4*ffmcfmb*p7p8-p7p11-2*p7p8)+16*(p3p7*p2p3)*(-
     . ffmcfmb*p6p11+ffmcfmb*p6p8+p6p11-p6p8)+8*(p3p6*p2p3)*(-ffmcfmb
     . *p7p11-4*ffmcfmb*p7p8+p7p11+2*p7p8)+16*(p3p7*p1p3)*(ffmcfmb*
     . p6p11-ffmcfmb*p6p8)+ans10
      ans8=w11*ans9
      ans13=(8*p3p4*(p6p12+2*p7p13)+4*p6p7*(-2*p3p11+p3p8)+4*p3p7*(-2
     . *ffmcfmb*p3p13+p1p13-7*p2p13+7*p3p13+7*p4p13+3*p5p13)+4*p3p6*(
     . -ffmcfmb*p3p12-4*p1p12+p3p12+p7p10-p7p11)+2*(-ffmcfmb*hbcm2*
     . p6p12+ffmcfmb*hbcm2*p7p13-3*fmb*hbcm*p7p13+fmc*hbcm*p6p12+4*
     . fmc*hbcm*p7p13+4*hbcm2*p6p12+3*hbcm2*p7p13+4*p1p3*p6p12-4*p2p3
     . *p6p12+2*p3p12*p5p6-4*p3p13*p4p7+4*p3p5*p6p12))
      ans1=ans2+ans5+ans8+ans13
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w11*(6*p5p6*(-fmb-fmc+hbcm)+2*p4p6*(ffmcfmb*hbcm+4*
     . fmb+3*fmc-4*hbcm)+2*p3p6*(ffmcfmb**2*hbcm+7*ffmcfmb*fmb+6*
     . ffmcfmb*fmc-7*ffmcfmb*hbcm-3*fmb-3*fmc+3*hbcm))+w1*(6*p5p6*(-
     . fmb-fmc+hbcm)+2*p4p6*(ffmcfmb*hbcm+4*fmb+3*fmc-4*hbcm)+2*p3p6*
     . (ffmcfmb**2*hbcm+7*ffmcfmb*fmb+6*ffmcfmb*fmc-7*ffmcfmb*hbcm-3*
     . fmb-3*fmc+3*hbcm)))
      b(32)=w2*ccc*(6*p5p6*(-fmb-fmc+hbcm)+2*p4p6*(ffmcfmb*hbcm+4*fmb
     . +3*fmc-4*hbcm)+2*p3p6*(ffmcfmb**2*hbcm+7*ffmcfmb*fmb+6*ffmcfmb
     . *fmc-7*ffmcfmb*hbcm-3*fmb-3*fmc+3*hbcm))
      ans5=4*p3p6*(ffmcfmb**2*hbcm*p7p11-ffmcfmb**2*hbcm*p7p8-3*
     . ffmcfmb*fmb*p7p11+3*ffmcfmb*fmb*p7p8-4*ffmcfmb*fmc*p7p11+4*
     . ffmcfmb*fmc*p7p8+2*ffmcfmb*hbcm*p7p11-2*ffmcfmb*hbcm*p7p8+fmb*
     . p7p11-fmb*p7p8+2*fmc*p7p11-2*fmc*p7p8-hbcm*p7p11+hbcm*p7p8)
      ans4=8*p4p6*(-fmb*p7p11+fmb*p7p8-fmc*p7p11+fmc*p7p8+hbcm*p7p11-
     . hbcm*p7p8)+4*p5p7*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+2*fmb*
     . p6p11-2*fmb*p6p8+fmc*p6p11-fmc*p6p8-2*hbcm*p6p11+2*hbcm*p6p8)+
     . 4*p5p6*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8+fmb*p7p11-fmb*
     . p7p8+2*fmc*p7p11-2*fmc*p7p8-hbcm*p7p11+hbcm*p7p8)+8*p4p7*(
     . ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-hbcm*
     . p6p11+hbcm*p6p8)+4*p6p7*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p1p8
     . +ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p4p11+
     . ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p11-ffmcfmb*hbcm*p5p8+2*fmb*
     . p2p11-2*fmb*p2p8-2*fmb*p4p11+2*fmb*p4p8+fmc*p1p11-fmc*p1p8+fmc
     . *p2p11-fmc*p2p8-fmc*p4p11+fmc*p4p8-fmc*p5p11+fmc*p5p8-2*hbcm*
     . p2p11+2*hbcm*p2p8+2*hbcm*p4p11-2*hbcm*p4p8)+4*p3p7*(ffmcfmb**2
     . *hbcm*p6p11-ffmcfmb**2*hbcm*p6p8-ffmcfmb*fmc*p6p11+ffmcfmb*fmc
     . *p6p8+ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+2*fmb*p6p11-2*fmb*
     . p6p8+fmc*p6p11-fmc*p6p8-2*hbcm*p6p11+2*hbcm*p6p8)+ans5
      ans3=w11*ans4
      ans6=2*(ffmcfmb*hbcm*p6p12+ffmcfmb*hbcm*p7p13-3*fmb*p7p13-fmc*
     . p6p12-4*fmc*p7p13+3*hbcm*p7p13)
      ans2=w1*(8*p4p6*(fmb*p7p8+fmc*p7p8-hbcm*p7p8)+4*p5p7*(-ffmcfmb*
     . hbcm*p6p8-2*fmb*p6p8-fmc*p6p8+2*hbcm*p6p8)+4*p5p6*(ffmcfmb*
     . hbcm*p7p8-fmb*p7p8-2*fmc*p7p8+hbcm*p7p8)+8*p4p7*(-ffmcfmb*hbcm
     . *p6p8-fmb*p6p8+hbcm*p6p8)+4*p6p7*(ffmcfmb*hbcm*p1p8-ffmcfmb*
     . hbcm*p2p8+ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p8-2*fmb*p2p8+2*fmb
     . *p4p8-fmc*p1p8-fmc*p2p8+fmc*p4p8+fmc*p5p8+2*hbcm*p2p8-2*hbcm*
     . p4p8)+4*p3p7*(-ffmcfmb**2*hbcm*p6p8+ffmcfmb*fmc*p6p8-ffmcfmb*
     . hbcm*p6p8-2*fmb*p6p8-fmc*p6p8+2*hbcm*p6p8)+4*p3p6*(-ffmcfmb**2
     . *hbcm*p7p8+3*ffmcfmb*fmb*p7p8+4*ffmcfmb*fmc*p7p8-2*ffmcfmb*
     . hbcm*p7p8-fmb*p7p8-2*fmc*p7p8+hbcm*p7p8))+ans3+ans6
      ans1=w2*(8*p4p6*(-fmb*p7p9-fmc*p7p9+hbcm*p7p9)+4*p5p7*(ffmcfmb*
     . hbcm*p6p9+2*fmb*p6p9+fmc*p6p9-2*hbcm*p6p9)+4*p5p6*(-ffmcfmb*
     . hbcm*p7p9+fmb*p7p9+2*fmc*p7p9-hbcm*p7p9)+8*p4p7*(ffmcfmb*hbcm*
     . p6p9+fmb*p6p9-hbcm*p6p9)+4*p6p7*(-ffmcfmb*hbcm*p1p9+ffmcfmb*
     . hbcm*p2p9-ffmcfmb*hbcm*p4p9+ffmcfmb*hbcm*p5p9-2*fmb*p1p9+2*fmb
     . *p2p9-2*fmb*p4p9-fmc*p1p9+fmc*p2p9-fmc*p4p9-fmc*p5p9+2*hbcm*
     . p1p9-2*hbcm*p2p9+2*hbcm*p4p9)+4*p3p7*(ffmcfmb**2*hbcm*p6p9-
     . ffmcfmb*fmc*p6p9+ffmcfmb*hbcm*p6p9+2*fmb*p6p9+fmc*p6p9-2*hbcm*
     . p6p9)+4*p3p6*(ffmcfmb**2*hbcm*p7p9-3*ffmcfmb*fmb*p7p9-4*
     . ffmcfmb*fmc*p7p9+2*ffmcfmb*hbcm*p7p9+fmb*p7p9+2*fmc*p7p9-hbcm*
     . p7p9))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(37)=w2*ccc*(4*p3p7*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p4p7+
     . hbcm*p5p7))
      b(39)=w11*ccc*((p3p7*p3p6)*(-32*ffmcfmb+12)+4*p6p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+4*(-2*p1p3*p6p7+2*p2p3*p6p7-2*
     . p3p4*p6p7+2*p3p5*p6p7-2*p3p6*p4p7+2*p3p6*p5p7-3*p3p7*p4p6+p3p7
     . *p5p6))
      b(40)=ccc*(w11*(4*(p6p7*p2p3)*(-2*fmb-fmc-hbcm)+4*(p5p6*p3p7)*(
     . ffmcfmb*hbcm-fmb-2*fmc)+8*(p4p6*p3p7)*(2*ffmcfmb*hbcm+fmb+fmc-
     . hbcm)+4*(p5p7*p3p6)*(6*ffmcfmb*hbcm-2*fmb-fmc-3*hbcm)+4*(p4p7*
     . p3p6)*(ffmcfmb*hbcm-2*fmb+hbcm)+4*(p6p7*p3p5)*(-2*ffmcfmb*hbcm
     . +fmc)+4*(p6p7*p3p4)*(2*ffmcfmb*hbcm+2*fmb+fmc-hbcm)+2*p6p7*(3*
     . ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-3*
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+4*(p3p7*p3p6
     . )*(2*ffmcfmb**2*hbcm+3*ffmcfmb*fmb+5*ffmcfmb*fmc-3*fmb-3*fmc)+
     . 4*(-fmc*p1p3*p6p7-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7-2*hbcm*p1p5*
     . p6p7-2*hbcm*p2p4*p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7+2*hbcm*
     . p4p6*p4p7+2*hbcm*p4p6*p5p7+hbcm*p4p7*p5p6-3*hbcm*p5p6*p5p7))+2
     . *hbcm*p6p7)
      b(41)=ccc*(w11*((p5p6*p3p7)*(8*ffmcfmb-4)+(p5p7*p3p6)*(28*
     . ffmcfmb-20)+(p4p7*p3p6)*(-4*ffmcfmb+4)+(p3p7*p3p6)*(20*ffmcfmb
     . -12)+(p6p7*p3p4)*(4*ffmcfmb+4)+(p6p7*p2p3)*(4*ffmcfmb-12)+2*
     . p6p7*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+4*(-ffmcfmb*p1p3*p6p7-
     . ffmcfmb*p3p5*p6p7+2*ffmcfmb*p3p7*p4p6-p1p2*p6p7-p1p4*p6p7-2*
     . p1p5*p6p7-2*p2p4*p6p7-p2p5*p6p7-p4p5*p6p7+2*p4p6*p5p7+3*p4p7*
     . p5p6-3*p5p6*p5p7))+2*p6p7)
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb+12)+4*p6p7*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+hbcm2)+4*(-2*p1p3*p6p7+2*p2p3*p6p7-2*
     . p3p4*p6p7+2*p3p5*p6p7-2*p3p6*p4p7+2*p3p6*p5p7-3*p3p7*p4p6+p3p7
     . *p5p6))
      b(43)=ccc*(w2*(8*p6p7*(-p1p9+p2p9-p4p9+p5p9)+8*p3p7*(-2*ffmcfmb
     . *p6p9+p6p9)+4*p3p6*(-4*ffmcfmb*p7p9+p7p9)+4*(-3*p4p6*p7p9-2*
     . p4p7*p6p9+p5p6*p7p9+2*p5p7*p6p9))+w1*(8*p6p7*(p1p8-p2p8+p4p8-
     . p5p8)+8*p3p7*(2*ffmcfmb*p6p8-p6p8)+4*p3p6*(4*ffmcfmb*p7p8-p7p8
     . )+4*(3*p4p6*p7p8+2*p4p7*p6p8-p5p6*p7p8-2*p5p7*p6p8))+w11*(8*
     . p5p7*(p6p11-p6p8)+4*p5p6*(p7p11-p7p8)+8*p4p7*(-p6p11+p6p8)+12*
     . p4p6*(-p7p11+p7p8)+8*p6p7*(-p1p11+p1p8+p2p11-p2p8-p4p11+p4p8+
     . p5p11-p5p8)+8*p3p7*(-2*ffmcfmb*p6p11+2*ffmcfmb*p6p8+p6p11-p6p8
     . )+4*p3p6*(-4*ffmcfmb*p7p11+4*ffmcfmb*p7p8+p7p11-p7p8))+8*(-
     . p6p12-p7p13))
      b(44)=ccc*(w11*((p3p6*p3p4)*(-28*ffmcfmb+12)+(p3p6*p2p3)*(20*
     . ffmcfmb-12)+2*p5p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*
     . hbcm2)+2*p4p6*(-ffmcfmb*hbcm2+4*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+2
     . *p3p6*(-ffmcfmb**2*hbcm2+7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm
     . -7*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+3*hbcm2)+4*(-ffmcfmb*
     . p1p3*p3p6-ffmcfmb*p3p5*p3p6-p1p2*p3p6-p1p4*p3p6+2*p2p3*p4p6-2*
     . p2p3*p5p6-p2p5*p3p6-4*p3p4*p4p6+4*p3p4*p5p6-p3p6*p4p5))+2*p3p6
     . )
      b(45)=w11*ccc*(p3p6*(-16*ffmcfmb+6)+2*(-5*p4p6+3*p5p6))
      b(46)=w11*ccc*(6*p5p6*(-fmb-fmc+hbcm)+2*p4p6*(ffmcfmb*hbcm+4*
     . fmb+3*fmc-4*hbcm)+2*p3p6*(ffmcfmb**2*hbcm+7*ffmcfmb*fmb+6*
     . ffmcfmb*fmc-7*ffmcfmb*hbcm-3*fmb-3*fmc+3*hbcm))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb+6)+2*(-5*p4p6+3*p5p6))
      b(48)=4*ccc*(p3p6+p4p6+p5p6)
      b(49)=w11*ccc*(4*p3p7*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p4p7+
     . hbcm*p5p7))
      b(50)=w11*ccc*(p3p7*(-16*ffmcfmb+4)+4*(-3*p4p7+p5p7))
      b(51)=ccc*(w11*((p3p7*p3p5)*(12*ffmcfmb-8)+(p3p7*p3p4)*(4*
     . ffmcfmb+4)+(p3p7*p2p3)*(-12*ffmcfmb+4)+2*p5p7*(ffmcfmb*hbcm2-2
     . *fmb*hbcm+fmc*hbcm-2*hbcm2)+2*p3p7*(ffmcfmb**2*hbcm2+2*ffmcfmb
     . *fmb*hbcm-ffmcfmb*fmc*hbcm+5*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm
     . -2*hbcm2)+4*(2*ffmcfmb*hbcm2*p4p7+3*ffmcfmb*p1p3*p3p7+p1p2*
     . p3p7+2*p1p3*p4p7+p1p4*p3p7-2*p2p3*p4p7+p2p5*p3p7+2*p3p4*p4p7-2
     . *p3p5*p5p7+p3p7*p4p5))-2*p3p7)
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+4)+4*(-3*p4p7+p5p7))
      b(56)=ccc*(w11*(p3p4*(-2*ffmcfmb+2)+p2p3*(-2*ffmcfmb+2)+2*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*hbcm2+ffmcfmb*p1p3+ffmcfmb*p3p5+p1p2+
     . p1p4+p2p5+p4p5))-1)
      b(57)=-8*ccc
      b(58)=w11*ccc*(2*p3p6*(8*ffmcfmb*hbcm-3*hbcm)+2*(5*hbcm*p4p6-3*
     . hbcm*p5p6))
      b(59)=ccc*(w11*(2*p3p4*(ffmcfmb*hbcm-hbcm)+2*p2p3*(ffmcfmb*hbcm
     . -hbcm)+2*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm*p1p3-ffmcfmb*hbcm*p3p5
     . -ffmcfmb*hbcm3-hbcm*p1p2-hbcm*p1p4-hbcm*p2p5-hbcm*p4p5))+hbcm)
      b(60)=w2*ccc*(2*p3p6*(8*ffmcfmb*hbcm-3*hbcm)+2*(5*hbcm*p4p6-3*
     . hbcm*p5p6))
      b(61)=-8*ccc*hbcm
      b(62)=ccc*(4*p3p6*(ffmcfmb*hbcm-fmb+hbcm)+8*hbcm*p4p6)
      b(64)=-2*ccc*p6p7
      b(65)=ccc*(w11*((p3p7*p3p6)*(32*ffmcfmb-12)+4*p6p7*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+4*(2*p1p3*p6p7-2*p2p3*p6p7+2*
     . p3p4*p6p7-2*p3p5*p6p7+2*p3p6*p4p7-2*p3p6*p5p7+3*p3p7*p4p6-p3p7
     . *p5p6))+w1*((p3p7*p3p6)*(32*ffmcfmb-12)+4*p6p7*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+4*(2*p1p3*p6p7-2*p2p3*p6p7+2*
     . p3p4*p6p7-2*p3p5*p6p7+2*p3p6*p4p7-2*p3p6*p5p7+3*p3p7*p4p6-p3p7
     . *p5p6)))
      ans=ccc*(w2*(8*p6p7*(-hbcm*p1p9+hbcm*p2p9-hbcm*p4p9+hbcm*p5p9)+
     . 8*p3p7*(-2*ffmcfmb*hbcm*p6p9+hbcm*p6p9)+4*p3p6*(-4*ffmcfmb*
     . hbcm*p7p9+hbcm*p7p9)+4*(-3*hbcm*p4p6*p7p9-2*hbcm*p4p7*p6p9+
     . hbcm*p5p6*p7p9+2*hbcm*p5p7*p6p9))+w1*(8*p6p7*(hbcm*p1p8-hbcm*
     . p2p8+hbcm*p4p8-hbcm*p5p8)+8*p3p7*(2*ffmcfmb*hbcm*p6p8-hbcm*
     . p6p8)+4*p3p6*(4*ffmcfmb*hbcm*p7p8-hbcm*p7p8)+4*(3*hbcm*p4p6*
     . p7p8+2*hbcm*p4p7*p6p8-hbcm*p5p6*p7p8-2*hbcm*p5p7*p6p8))+w11*(8
     . *p5p7*(hbcm*p6p11-hbcm*p6p8)+4*p5p6*(hbcm*p7p11-hbcm*p7p8)+8*
     . p4p7*(-hbcm*p6p11+hbcm*p6p8)+12*p4p6*(-hbcm*p7p11+hbcm*p7p8)+8
     . *p6p7*(-hbcm*p1p11+hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*p4p11+
     . hbcm*p4p8+hbcm*p5p11-hbcm*p5p8)+8*p3p7*(-2*ffmcfmb*hbcm*p6p11+
     . 2*ffmcfmb*hbcm*p6p8+hbcm*p6p11-hbcm*p6p8)+4*p3p6*(-4*ffmcfmb*
     . hbcm*p7p11+4*ffmcfmb*hbcm*p7p8+hbcm*p7p11-hbcm*p7p8))+8*(-hbcm
     . *p6p12-hbcm*p7p13))
      b(66)=ans
      b(68)=ccc*(w11*(p3p6*(16*ffmcfmb-6)+2*(5*p4p6-3*p5p6))+w1*(p3p6
     . *(16*ffmcfmb-6)+2*(5*p4p6-3*p5p6)))
      b(70)=2*ccc*p3p7
      b(71)=ccc*(w11*(p3p7*(16*ffmcfmb-4)+4*(3*p4p7-p5p7))+w1*(p3p7*(
     . 16*ffmcfmb-4)+4*(3*p4p7-p5p7)))
      b(72)=ccc
      b(73)=4*ccc*w11*p6p7*(-fmb-fmc+hbcm)
      b(74)=ccc*(4*w11*p6p7*(fmb+fmc-hbcm)+4*w1*p6p7*(fmb+fmc-hbcm))
      b(79)=4*ccc*w2*p6p7*(-fmb-fmc+hbcm)
      b(80)=w2*ccc*(2*p5p7*(ffmcfmb*hbcm+2*fmb+fmc-2*hbcm)+2*p3p7*(-
     . ffmcfmb**2*hbcm-2*ffmcfmb*fmb-ffmcfmb*fmc+3*ffmcfmb*hbcm+2*fmb
     . +fmc-2*hbcm))
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(2*p3p7*(3*ffmcfmb*hbcm+3*fmb+4*fmc)+2*(3*hbcm*p4p7+7
     . *hbcm*p5p7))
      b(84)=2*ccc*(3*p3p7-p4p7+7*p5p7)
      b(85)=ccc*(-3*ffmcfmb*hbcm2+7*fmb*hbcm-6*fmc*hbcm-7*hbcm2+8*
     . p2p3-16*p3p4)
      b(86)=-8*ccc
      b(87)=ccc*(ffmcfmb*hbcm+7*fmb+6*fmc-7*hbcm)
      b(89)=-8*ccc*hbcm
      b(90)=w11*ccc*(2*p5p7*(ffmcfmb*hbcm+2*fmb+fmc-2*hbcm)+2*p3p7*(-
     . ffmcfmb**2*hbcm-2*ffmcfmb*fmb-ffmcfmb*fmc+3*ffmcfmb*hbcm+2*fmb
     . +fmc-2*hbcm))
      b(91)=ccc*(w11*(2*p5p7*(-ffmcfmb*hbcm-2*fmb-fmc+2*hbcm)+2*p3p7*
     . (ffmcfmb**2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-3*ffmcfmb*hbcm-2*
     . fmb-fmc+2*hbcm))+w1*(2*p5p7*(-ffmcfmb*hbcm-2*fmb-fmc+2*hbcm)+2
     . *p3p7*(ffmcfmb**2*hbcm+2*ffmcfmb*fmb+ffmcfmb*fmc-3*ffmcfmb*
     . hbcm-2*fmb-fmc+2*hbcm)))
      b(92)=ccc*(3*ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm+2*hbcm2+4*p1p3-4
     . *p2p3+4*p3p4+4*p3p5)
      b(93)=8*ccc*p3p6
      b(94)=ccc*(ffmcfmb*hbcm+2*fmb+fmc-2*hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)+1d0*b(n)
200   continue
      return
      end 
 
      subroutine amp34_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)
     . )
      b(1)=w1*ccc*(2*p6p7*(2*ffmcfmb*hbcm2-fmb2+fmc2-hbcm2)+4*(2*p1p2
     . *p6p7-2*p1p3*p6p7+p1p4*p6p7-5*p1p5*p6p7-p2p4*p6p7+p2p5*p6p7+
     . p3p4*p6p7-p3p5*p6p7))
      b(2)=w2*ccc*(2*p6p7*(-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2)+4*(-2*
     . p1p2*p6p7+2*p1p3*p6p7-p1p4*p6p7+5*p1p5*p6p7+p2p4*p6p7-p2p5*
     . p6p7-p3p4*p6p7+p3p5*p6p7))
      b(11)=w1*ccc*(4*(p6p7*p3p4)*(fmb+fmc)+4*(p6p7*p2p3)*(ffmcfmb*
     . hbcm-fmb)+4*(p6p7*p1p3)*(-ffmcfmb*hbcm-3*fmb-4*fmc+2*hbcm)+2*
     . p6p7*(2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-fmb2*hbcm+fmc2*
     . hbcm-hbcm3)+4*(2*hbcm*p1p2*p6p7-4*hbcm*p1p5*p6p7-hbcm*p3p5*
     . p6p7))
      b(12)=w2*ccc*(4*(p6p7*p3p4)*(-fmb-fmc)+4*(p6p7*p2p3)*(-ffmcfmb*
     . hbcm+fmb)+4*(p6p7*p1p3)*(ffmcfmb*hbcm-3*fmb-2*fmc+4*hbcm)+2*
     . p6p7*(-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+fmb2*hbcm-fmc2*
     . hbcm+hbcm3)+4*(-2*hbcm*p1p2*p6p7+4*hbcm*p1p5*p6p7+hbcm*p3p5*
     . p6p7))
      b(29)=4*ccc*hbcm*p6p7
      b(30)=ccc*(w2*(8*(p6p7*p3p5)*(5*p1p9-p2p9+p5p9)+8*(p6p7*p3p4)*(
     . 2*p1p9-p4p9)+8*(p6p7*p2p3)*(-2*p1p9+p4p9)+8*(p6p7*p1p3)*(-4*
     . p1p9-2*p2p9-p4p9+2*p5p9)+4*p6p7*(-ffmcfmb*hbcm2*p1p9+ffmcfmb*
     . hbcm2*p2p9-2*ffmcfmb*hbcm2*p5p9+3*fmb*hbcm*p1p9-fmb*hbcm*p2p9+
     . fmb*hbcm*p4p9-2*fmc*hbcm*p1p9-fmc*hbcm*p4p9+7*hbcm2*p1p9-hbcm2
     . *p2p9-hbcm2*p4p9+2*hbcm2*p5p9))+w1*(8*(p6p7*p3p5)*(-5*p1p8+
     . p2p8-p5p8)+8*(p6p7*p3p4)*(-2*p1p8+p4p8)+8*(p6p7*p2p3)*(2*p1p8-
     . p4p8)+8*(p6p7*p1p3)*(4*p1p8+2*p2p8+p4p8-2*p5p8)+4*p6p7*(
     . ffmcfmb*hbcm2*p1p8-ffmcfmb*hbcm2*p2p8+2*ffmcfmb*hbcm2*p5p8-3*
     . fmb*hbcm*p1p8+fmb*hbcm*p2p8-fmb*hbcm*p4p8+2*fmc*hbcm*p1p8+fmc*
     . hbcm*p4p8-7*hbcm2*p1p8+hbcm2*p2p8+hbcm2*p4p8-2*hbcm2*p5p8))+4*
     . p6p7*(-p3p10+p3p11))
      b(33)=ccc*(4*w2*p6p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+3*fmb
     . *p1p9-fmb*p2p9+fmb*p4p9+2*fmc*p1p9+fmc*p4p9-3*hbcm*p1p9+hbcm*
     . p2p9-hbcm*p4p9)+4*w1*p6p7*(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*
     . p2p8+3*fmb*p1p8+fmb*p2p8-fmb*p4p8+4*fmc*p1p8-fmc*p4p8-3*hbcm*
     . p1p8-hbcm*p2p8+hbcm*p4p8))
      b(40)=-4*ccc*hbcm*p6p7
      b(42)=w2*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm-3*
     . hbcm2)+8*(p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-3*p3p5*p6p7))
      b(43)=ccc*(8*w2*p6p7*(p1p9+p2p9+p4p9-3*p5p9)+8*w1*p6p7*(-p1p8-
     . p2p8-p4p8+3*p5p8))
      b(65)=w1*ccc*(4*p6p7*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm+3*
     . hbcm2)+8*(-p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+3*p3p5*p6p7))
      b(66)=ccc*(8*w2*p6p7*(hbcm*p1p9+hbcm*p2p9+hbcm*p4p9-3*hbcm*p5p9
     . )+8*w1*p6p7*(-hbcm*p1p8-hbcm*p2p8-hbcm*p4p8+3*hbcm*p5p8))
      b(74)=12*ccc*w1*p6p7*(-fmb-fmc+hbcm)
      b(79)=12*ccc*w2*p6p7*(fmb+fmc-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp33_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=ccc*(w11*((p5p7*p3p6)*(-24*ffmcfmb+24)+(p4p7*p3p6)*(8*
     . ffmcfmb-8)+(p3p7*p3p6)*(-8*ffmcfmb+8)+4*p6p7*(ffmcfmb**2*hbcm2
     . +2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*fmb*fmc+fmb2+2*fmc*
     . hbcm-hbcm2)+8*(ffmcfmb*p1p3*p6p7+p1p2*p6p7+p1p4*p6p7-p1p5*p6p7
     . ))+w1*((p5p7*p3p6)*(-24*ffmcfmb+24)+(p4p7*p3p6)*(8*ffmcfmb-8)+
     . (p3p7*p3p6)*(-8*ffmcfmb+8)+4*p6p7*(ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*fmb*fmc+fmb2+2*fmc*hbcm-hbcm2)+8
     . *(ffmcfmb*p1p3*p6p7+p1p2*p6p7+p1p4*p6p7-p1p5*p6p7)))
      ans=ccc*(w1*(16*p3p7*(ffmcfmb*p1p8+p1p8)+8*p1p3*(ffmcfmb*p7p8+
     . p7p8)+4*(ffmcfmb**2*hbcm2*p7p8+2*ffmcfmb*fmb*hbcm*p7p8-2*
     . ffmcfmb*fmc*hbcm*p7p8-2*fmb*fmc*p7p8+fmb2*p7p8+2*fmc*hbcm*p7p8
     . -hbcm2*p7p8+4*p1p4*p7p8+12*p1p8*p4p7))+w11*(16*p1p4*(-p7p11+
     . p7p8)+48*p4p7*(-p1p11+p1p8)+8*p1p3*(-ffmcfmb*p7p11+ffmcfmb*
     . p7p8-p7p11+p7p8)+16*p3p7*(-ffmcfmb*p1p11+ffmcfmb*p1p8-p1p11+
     . p1p8)+4*(-ffmcfmb**2*hbcm2*p7p11+ffmcfmb**2*hbcm2*p7p8-2*
     . ffmcfmb*fmb*hbcm*p7p11+2*ffmcfmb*fmb*hbcm*p7p8+2*ffmcfmb*fmc*
     . hbcm*p7p11-2*ffmcfmb*fmc*hbcm*p7p8+2*fmb*fmc*p7p11-2*fmb*fmc*
     . p7p8-fmb2*p7p11+fmb2*p7p8-2*fmc*hbcm*p7p11+2*fmc*hbcm*p7p8+
     . hbcm2*p7p11-hbcm2*p7p8))+w13*(8*p1p3*(ffmcfmb*p7p10+p7p10)+16*
     . p3p7*(ffmcfmb*p1p10+p1p10)+4*(ffmcfmb**2*hbcm2*p7p10+2*ffmcfmb
     . *fmb*hbcm*p7p10-2*ffmcfmb*fmc*hbcm*p7p10-2*fmb*fmc*p7p10+fmb2*
     . p7p10+2*fmc*hbcm*p7p10-hbcm2*p7p10+12*p1p10*p4p7+4*p1p4*p7p10)
     . )+4*(-5*p1p12+2*p3p12+2*p4p12+2*p5p12-2*p7p11))
      b(3)=ans
      b(4)=ccc*(w11*(12*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p4p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p3p7*(-ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb*fmc+
     . fmb*hbcm-fmb2-2*fmc*hbcm)+8*(-3*ffmcfmb*p1p3*p3p7-3*p1p3*p4p7+
     . 3*p1p3*p5p7-2*p1p4*p3p7))+w1*(12*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+4*p4p7*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p3p7*(-ffmcfmb
     . **2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+
     . 2*fmb*fmc+fmb*hbcm-fmb2-2*fmc*hbcm)+8*(-3*ffmcfmb*p1p3*p3p7-3*
     . p1p3*p4p7+3*p1p3*p5p7-2*p1p4*p3p7)))
      ans3=w13*(16*p3p4*(-ffmcfmb*hbcm*p7p10-fmb*p7p10+hbcm*p7p10)+8*
     . p1p3*(ffmcfmb*hbcm*p7p10-2*fmc*p7p10-hbcm*p7p10)+8*p3p7*(-
     . ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p4p10-
     . ffmcfmb*hbcm*p5p10+fmb*p1p10+fmb*p2p10+fmb*p4p10-fmb*p5p10+2*
     . fmc*p1p10-hbcm*p2p10-hbcm*p4p10+hbcm*p5p10)+4*(-ffmcfmb**2*
     . hbcm3*p7p10-2*ffmcfmb*fmb*hbcm2*p7p10-2*ffmcfmb*fmc*hbcm2*
     . p7p10-2*fmb*fmc*hbcm*p7p10-fmb2*hbcm*p7p10+2*fmc*hbcm2*p7p10-2
     . *hbcm*p1p10*p4p7+6*hbcm*p1p10*p5p7-4*hbcm*p1p4*p7p10+hbcm3*
     . p7p10))+4*(-fmb*p3p12-fmc*p3p12+3*hbcm*p1p12-hbcm*p2p12-hbcm*
     . p3p12-3*hbcm*p4p12-2*hbcm*p5p12+3*hbcm*p7p11+hbcm*p7p8)
      ans2=w11*(16*p1p4*(hbcm*p7p11-hbcm*p7p8)+24*p5p7*(-hbcm*p1p11+
     . hbcm*p1p8)+8*p4p7*(hbcm*p1p11-hbcm*p1p8)+16*p3p4*(ffmcfmb*hbcm
     . *p7p11-ffmcfmb*hbcm*p7p8+fmb*p7p11-fmb*p7p8-hbcm*p7p11+hbcm*
     . p7p8)+8*p1p3*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8+2*fmc*
     . p7p11-2*fmc*p7p8+hbcm*p7p11-hbcm*p7p8)+8*p3p7*(ffmcfmb*hbcm*
     . p1p11-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p2p11+ffmcfmb*hbcm*p2p8-
     . ffmcfmb*hbcm*p4p11+ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p11-
     . ffmcfmb*hbcm*p5p8-fmb*p1p11+fmb*p1p8-fmb*p2p11+fmb*p2p8-fmb*
     . p4p11+fmb*p4p8+fmb*p5p11-fmb*p5p8-2*fmc*p1p11+2*fmc*p1p8+hbcm*
     . p2p11-hbcm*p2p8+hbcm*p4p11-hbcm*p4p8-hbcm*p5p11+hbcm*p5p8)+4*(
     . ffmcfmb**2*hbcm3*p7p11-ffmcfmb**2*hbcm3*p7p8+2*ffmcfmb*fmb*
     . hbcm2*p7p11-2*ffmcfmb*fmb*hbcm2*p7p8+2*ffmcfmb*fmc*hbcm2*p7p11
     . -2*ffmcfmb*fmc*hbcm2*p7p8+2*fmb*fmc*hbcm*p7p11-2*fmb*fmc*hbcm*
     . p7p8+fmb2*hbcm*p7p11-fmb2*hbcm*p7p8-2*fmc*hbcm2*p7p11+2*fmc*
     . hbcm2*p7p8-hbcm3*p7p11+hbcm3*p7p8))+ans3
      ans1=w1*(16*p3p4*(-ffmcfmb*hbcm*p7p8-fmb*p7p8+hbcm*p7p8)+8*p3p7
     . *(-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p4p8-
     . ffmcfmb*hbcm*p5p8+fmb*p1p8+fmb*p2p8+fmb*p4p8-fmb*p5p8+2*fmc*
     . p1p8-hbcm*p2p8-hbcm*p4p8+hbcm*p5p8)+8*p1p3*(ffmcfmb*hbcm*p7p8-
     . 2*fmc*p7p8-hbcm*p7p8)+4*(-ffmcfmb**2*hbcm3*p7p8-2*ffmcfmb*fmb*
     . hbcm2*p7p8-2*ffmcfmb*fmc*hbcm2*p7p8-2*fmb*fmc*hbcm*p7p8-fmb2*
     . hbcm*p7p8+2*fmc*hbcm2*p7p8-4*hbcm*p1p4*p7p8-2*hbcm*p1p8*p4p7+6
     . *hbcm*p1p8*p5p7+hbcm3*p7p8))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(8*p3p6*(-2*ffmcfmb*p2p10+3*ffmcfmb*p3p10+4*ffmcfmb*
     . p4p10+2*p2p10-3*p3p10-4*p4p10)+4*(3*ffmcfmb*fmb*hbcm*p6p10-3*
     . ffmcfmb*fmc*hbcm*p6p10+3*ffmcfmb*hbcm2*p6p10+2*ffmcfmb*p1p3*
     . p6p10-3*fmb*fmc*p6p10+3*fmb2*p6p10+3*fmc*hbcm*p6p10-3*hbcm2*
     . p6p10+2*p1p2*p6p10+2*p1p4*p6p10-6*p1p5*p6p10))+4*(3*p2p13-3*
     . p3p13-3*p4p13-p6p10-p6p11-p6p8)
      ans1=w1*(8*p3p6*(-2*ffmcfmb*p2p8+3*ffmcfmb*p3p8+4*ffmcfmb*p4p8+
     . 2*p2p8-3*p3p8-4*p4p8)+4*(3*ffmcfmb*fmb*hbcm*p6p8-3*ffmcfmb*fmc
     . *hbcm*p6p8+3*ffmcfmb*hbcm2*p6p8+2*ffmcfmb*p1p3*p6p8-3*fmb*fmc*
     . p6p8+3*fmb2*p6p8+3*fmc*hbcm*p6p8-3*hbcm2*p6p8+2*p1p2*p6p8+2*
     . p1p4*p6p8-6*p1p5*p6p8))+w11*(24*p1p5*(p6p11-p6p8)+8*p1p4*(-
     . p6p11+p6p8)+8*p1p2*(-p6p11+p6p8)+8*p3p6*(2*ffmcfmb*p2p11-2*
     . ffmcfmb*p2p8-3*ffmcfmb*p3p11+3*ffmcfmb*p3p8-4*ffmcfmb*p4p11+4*
     . ffmcfmb*p4p8-2*p2p11+2*p2p8+3*p3p11-3*p3p8+4*p4p11-4*p4p8)+8*
     . p1p3*(-ffmcfmb*p6p11+ffmcfmb*p6p8)+12*(-ffmcfmb*fmb*hbcm*p6p11
     . +ffmcfmb*fmb*hbcm*p6p8+ffmcfmb*fmc*hbcm*p6p11-ffmcfmb*fmc*hbcm
     . *p6p8-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8+fmb*fmc*p6p11-fmb
     . *fmc*p6p8-fmb2*p6p11+fmb2*p6p8-fmc*hbcm*p6p11+fmc*hbcm*p6p8+
     . hbcm2*p6p11-hbcm2*p6p8))+ans2
      ans=ccc*ans1
      b(7)=ans
      b(8)=ccc*(w11*((p3p6*p3p4)*(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*
     . ffmcfmb-16)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2+3*fmb*fmc-3*
     . fmb2+3*hbcm2)+8*(-ffmcfmb*p1p3*p3p6-p1p2*p3p6-p1p4*p3p6+3*p1p5
     . *p3p6))+w1*((p3p6*p3p4)*(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*
     . ffmcfmb-16)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2+3*fmb*fmc-3*
     . fmb2+3*hbcm2)+8*(-ffmcfmb*p1p3*p3p6-p1p2*p3p6-p1p4*p3p6+3*p1p5
     . *p3p6)))
      ans4=4*(2*ffmcfmb**2*hbcm3*p6p11-2*ffmcfmb**2*hbcm3*p6p8+5*
     . ffmcfmb*fmb*hbcm2*p6p11-5*ffmcfmb*fmb*hbcm2*p6p8+3*ffmcfmb*fmc
     . *hbcm2*p6p11-3*ffmcfmb*fmc*hbcm2*p6p8-5*ffmcfmb*hbcm3*p6p11+5*
     . ffmcfmb*hbcm3*p6p8+3*fmb*fmc*hbcm*p6p11-3*fmb*fmc*hbcm*p6p8-6*
     . fmb*hbcm2*p6p11+6*fmb*hbcm2*p6p8+3*fmb2*hbcm*p6p11-3*fmb2*hbcm
     . *p6p8-3*fmc*hbcm2*p6p11+3*fmc*hbcm2*p6p8+3*hbcm3*p6p11-3*hbcm3
     . *p6p8)
      ans3=24*p1p5*(-hbcm*p6p11+hbcm*p6p8)+8*p1p4*(hbcm*p6p11-hbcm*
     . p6p8)+8*p1p2*(hbcm*p6p11-hbcm*p6p8)+32*p4p6*(-ffmcfmb*hbcm*
     . p3p8-fmb*p3p8-hbcm*p1p11+hbcm*p1p8+hbcm*p3p8)+24*p3p5*(-
     . ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8-fmb*p6p11+fmb*p6p8+hbcm*
     . p6p11-hbcm*p6p8)+8*p3p4*(ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8+
     . fmb*p6p11-fmb*p6p8-hbcm*p6p11+hbcm*p6p8)+8*p2p3*(ffmcfmb*hbcm*
     . p6p11-ffmcfmb*hbcm*p6p8+fmb*p6p11-fmb*p6p8-hbcm*p6p11+hbcm*
     . p6p8)+8*p1p3*(4*ffmcfmb*hbcm*p6p11-4*ffmcfmb*hbcm*p6p8+3*fmb*
     . p6p11-3*fmb*p6p8-3*hbcm*p6p11+3*hbcm*p6p8)+8*p3p6*(-ffmcfmb**2
     . *hbcm*p3p8-4*ffmcfmb*fmb*p3p8-3*ffmcfmb*fmc*p3p8-ffmcfmb*hbcm*
     . p1p11+ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p3p8+2*fmb*p2p11-2*fmb*
     . p2p8-3*fmb*p3p11+3*fmb*p3p8-4*fmb*p4p11+4*fmb*p4p8-3*fmc*p1p11
     . +3*fmc*p1p8+3*fmc*p3p8)+ans4
      ans2=w11*ans3
      ans6=w13*(24*p3p5*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+8*
     . p3p4*(-ffmcfmb*hbcm*p6p10-fmb*p6p10+hbcm*p6p10)+32*p4p6*(-
     . ffmcfmb*hbcm*p3p10-fmb*p3p10+hbcm*p1p10+hbcm*p3p10)+8*p2p3*(-
     . ffmcfmb*hbcm*p6p10-fmb*p6p10+hbcm*p6p10)+8*p1p3*(-4*ffmcfmb*
     . hbcm*p6p10-3*fmb*p6p10+3*hbcm*p6p10)+8*p3p6*(-ffmcfmb**2*hbcm*
     . p3p10-4*ffmcfmb*fmb*p3p10-3*ffmcfmb*fmc*p3p10+ffmcfmb*hbcm*
     . p1p10+ffmcfmb*hbcm*p3p10-2*fmb*p2p10+3*fmb*p3p10+4*fmb*p4p10+3
     . *fmc*p1p10+3*fmc*p3p10)+4*(-2*ffmcfmb**2*hbcm3*p6p10-5*ffmcfmb
     . *fmb*hbcm2*p6p10-3*ffmcfmb*fmc*hbcm2*p6p10+5*ffmcfmb*hbcm3*
     . p6p10-3*fmb*fmc*hbcm*p6p10+6*fmb*hbcm2*p6p10-3*fmb2*hbcm*p6p10
     . +3*fmc*hbcm2*p6p10-2*hbcm*p1p2*p6p10-2*hbcm*p1p4*p6p10+6*hbcm*
     . p1p5*p6p10-3*hbcm3*p6p10))+4*(-3*hbcm*p2p13+3*hbcm*p3p13+3*
     . hbcm*p4p13+hbcm*p6p10+hbcm*p6p11+hbcm*p6p8)
      ans5=w1*(32*p4p6*(-ffmcfmb*hbcm*p3p8-fmb*p3p8+hbcm*p1p8+hbcm*
     . p3p8)+24*p3p5*(ffmcfmb*hbcm*p6p8+fmb*p6p8-hbcm*p6p8)+8*p3p4*(-
     . ffmcfmb*hbcm*p6p8-fmb*p6p8+hbcm*p6p8)+8*p2p3*(-ffmcfmb*hbcm*
     . p6p8-fmb*p6p8+hbcm*p6p8)+8*p1p3*(-4*ffmcfmb*hbcm*p6p8-3*fmb*
     . p6p8+3*hbcm*p6p8)+8*p3p6*(-ffmcfmb**2*hbcm*p3p8-4*ffmcfmb*fmb*
     . p3p8-3*ffmcfmb*fmc*p3p8+ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p3p8-2*
     . fmb*p2p8+3*fmb*p3p8+4*fmb*p4p8+3*fmc*p1p8+3*fmc*p3p8)+4*(-2*
     . ffmcfmb**2*hbcm3*p6p8-5*ffmcfmb*fmb*hbcm2*p6p8-3*ffmcfmb*fmc*
     . hbcm2*p6p8+5*ffmcfmb*hbcm3*p6p8-3*fmb*fmc*hbcm*p6p8+6*fmb*
     . hbcm2*p6p8-3*fmb2*hbcm*p6p8+3*fmc*hbcm2*p6p8-2*hbcm*p1p2*p6p8-
     . 2*hbcm*p1p4*p6p8+6*hbcm*p1p5*p6p8-3*hbcm3*p6p8))+ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      ans2=w1*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*p3p6)*
     . (-2*ffmcfmb*hbcm-fmb+2*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm-fmb+
     . hbcm)+8*(p6p7*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p2p3)*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm+fmb-hbcm)
     . +4*p6p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-2*fmb*hbcm2+fmb2*hbcm-2*
     . fmc*hbcm2+hbcm3)+8*(p3p7*p3p6)*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*
     . fmb-2*ffmcfmb*fmc+2*ffmcfmb*hbcm+fmb+2*fmc)+8*(3*fmb*p3p6*p5p7
     . +hbcm*p1p2*p6p7+hbcm*p1p4*p6p7-hbcm*p1p5*p6p7))
      ans1=w11*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*p3p6)
     . *(-2*ffmcfmb*hbcm-fmb+2*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm-fmb
     . +hbcm)+8*(p6p7*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p2p3)*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm+fmb-hbcm)
     . +4*p6p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-2*fmb*hbcm2+fmb2*hbcm-2*
     . fmc*hbcm2+hbcm3)+8*(p3p7*p3p6)*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*
     . fmb-2*ffmcfmb*fmc+2*ffmcfmb*hbcm+fmb+2*fmc)+8*(3*fmb*p3p6*p5p7
     . +hbcm*p1p2*p6p7+hbcm*p1p4*p6p7-hbcm*p1p5*p6p7))+ans2
      ans=ccc*ans1
      b(11)=ans
      b(13)=ccc*(w11*(p1p3*(-4*ffmcfmb-12)+2*(-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmb2-3*fmc*hbcm+3
     . *hbcm2+4*p1p2-8*p1p4))+w1*(p1p3*(-4*ffmcfmb-12)+2*(-3*ffmcfmb*
     . fmb*hbcm+3*ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmb2-3
     . *fmc*hbcm+3*hbcm2+4*p1p2-8*p1p4)))
      b(15)=ccc*(w11*(16*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p2p3*(-
     . ffmcfmb*hbcm-fmb+hbcm)+4*p1p3*(-2*ffmcfmb*hbcm+3*fmc+3*hbcm)+2
     . *(2*ffmcfmb**2*hbcm3+5*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm+3*fmb2*hbcm-3*fmc*hbcm2-4*hbcm*
     . p1p2+8*hbcm*p1p4-3*hbcm3))+w1*(16*p3p4*(ffmcfmb*hbcm+fmb-hbcm)
     . +8*p2p3*(-ffmcfmb*hbcm-fmb+hbcm)+4*p1p3*(-2*ffmcfmb*hbcm+3*fmc
     . +3*hbcm)+2*(2*ffmcfmb**2*hbcm3+5*ffmcfmb*fmb*hbcm2+3*ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3+3*fmb*fmc*hbcm+3*fmb2*hbcm-3*fmc*hbcm2
     . -4*hbcm*p1p2+8*hbcm*p1p4-3*hbcm3)))
      b(17)=w13*ccc*(4*p3p6*(-ffmcfmb*hbcm-3*fmc)-16*hbcm*p4p6)
      b(18)=6*ccc*w13*(-ffmcfmb*hbcm+fmc)
      b(19)=2*ccc*w13*(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p2p3+8*p3p4)
      b(20)=w13*ccc*(4*p3p7*(4*ffmcfmb*hbcm-2*fmc-hbcm)+12*(hbcm*p4p7
     . -hbcm*p5p7))
      b(21)=ccc*(4*w11*(3*ffmcfmb*hbcm*p3p8-3*fmc*p3p8-2*hbcm*p2p11+2
     . *hbcm*p2p8+3*hbcm*p3p11-3*hbcm*p3p8+4*hbcm*p4p11-4*hbcm*p4p8)+
     . 4*w1*(3*ffmcfmb*hbcm*p3p8-3*fmc*p3p8+2*hbcm*p2p8-3*hbcm*p3p8-4
     . *hbcm*p4p8)+4*w13*(3*ffmcfmb*hbcm*p3p10-3*fmc*p3p10+2*hbcm*
     . p2p10-3*hbcm*p3p10-4*hbcm*p4p10))
      b(22)=ccc*(w13*(16*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p2p3*(-
     . ffmcfmb*hbcm-fmb+hbcm)+4*p1p3*(-2*ffmcfmb*hbcm+3*fmc+3*hbcm)+2
     . *(2*ffmcfmb**2*hbcm3+5*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm+3*fmb2*hbcm-3*fmc*hbcm2-4*hbcm*
     . p1p2+8*hbcm*p1p4-3*hbcm3))-2*hbcm)
      b(23)=ccc*(w11*(4*p3p6*(-ffmcfmb*hbcm-3*fmc)-16*hbcm*p4p6)+w1*(
     . 4*p3p6*(-ffmcfmb*hbcm-3*fmc)-16*hbcm*p4p6))
      b(24)=ccc*(6*w11*(-ffmcfmb*hbcm+fmc)+6*w1*(-ffmcfmb*hbcm+fmc))
      b(25)=ccc*(2*w11*(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p2p3+8*p3p4)+2*
     . w1*(5*ffmcfmb*hbcm2+3*fmc*hbcm-4*p2p3+8*p3p4))
      b(26)=ccc*(w11*(4*p3p7*(4*ffmcfmb*hbcm-2*fmc-hbcm)+12*(hbcm*
     . p4p7-hbcm*p5p7))+w1*(4*p3p7*(4*ffmcfmb*hbcm-2*fmc-hbcm)+12*(
     . hbcm*p4p7-hbcm*p5p7)))
      b(27)=w13*ccc*(16*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p6*(ffmcfmb
     . **2*hbcm+4*ffmcfmb*fmb+3*ffmcfmb*fmc-ffmcfmb*hbcm-3*fmc))
      b(28)=ccc*(w13*((p3p6*p3p4)*(32*ffmcfmb-32)+(p3p6*p2p3)*(-16*
     . ffmcfmb+16)+16*p4p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p3p6*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2-3*fmb*fmc+3*
     . fmb2-3*hbcm2)+8*(ffmcfmb*p1p3*p3p6+p1p2*p3p6+p1p4*p3p6-3*p1p5*
     . p3p6))-4*p3p6)
      b(29)=ccc*(w13*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7
     . *p3p6)*(-2*ffmcfmb*hbcm-fmb+2*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb*
     . hbcm-fmb+hbcm)+8*(p6p7*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*
     . p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm+
     . fmb-hbcm)+4*p6p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-2*fmb*hbcm2+
     . fmb2*hbcm-2*fmc*hbcm2+hbcm3)+8*(p3p7*p3p6)*(-2*ffmcfmb**2*hbcm
     . -2*ffmcfmb*fmb-2*ffmcfmb*fmc+2*ffmcfmb*hbcm+fmb+2*fmc)+8*(3*
     . fmb*p3p6*p5p7+hbcm*p1p2*p6p7+hbcm*p1p4*p6p7-hbcm*p1p5*p6p7))-4
     . *hbcm*p6p7)
      ans4=8*p6p7*(-ffmcfmb*hbcm2*p1p11+ffmcfmb*hbcm2*p1p8+ffmcfmb*
     . hbcm2*p2p11-ffmcfmb*hbcm2*p2p8+ffmcfmb*hbcm2*p4p11-ffmcfmb*
     . hbcm2*p4p8-ffmcfmb*hbcm2*p5p11+ffmcfmb*hbcm2*p5p8+fmb*hbcm*
     . p1p11-fmb*hbcm*p1p8+fmb*hbcm*p2p11-fmb*hbcm*p2p8+fmb*hbcm*
     . p4p11-fmb*hbcm*p4p8-fmb*hbcm*p5p11+fmb*hbcm*p5p8-2*fmc*hbcm*
     . p1p11+2*fmc*hbcm*p1p8+hbcm2*p1p11-hbcm2*p1p8-hbcm2*p2p11+hbcm2
     . *p2p8-hbcm2*p4p11+hbcm2*p4p8+hbcm2*p5p11-hbcm2*p5p8)+8*p3p7*(
     . ffmcfmb**2*hbcm2*p6p11-ffmcfmb**2*hbcm2*p6p8+3*ffmcfmb*fmb*
     . hbcm*p6p11-3*ffmcfmb*fmb*hbcm*p6p8-ffmcfmb*fmc*hbcm*p6p11+
     . ffmcfmb*fmc*hbcm*p6p8+2*ffmcfmb*hbcm2*p6p11-2*ffmcfmb*hbcm2*
     . p6p8-fmb*fmc*p6p11+fmb*fmc*p6p8+fmb*hbcm*p6p11-fmb*hbcm*p6p8+2
     . *fmb2*p6p11-2*fmb2*p6p8+fmc*hbcm*p6p11-fmc*hbcm*p6p8-3*hbcm2*
     . p6p11+3*hbcm2*p6p8)+8*p3p6*(ffmcfmb**2*hbcm2*p7p11-ffmcfmb*fmb
     . *hbcm*p7p8+2*ffmcfmb*hbcm2*p7p8-2*fmb*fmc*p7p11-fmb*fmc*p7p8+
     . fmb2*p7p11+2*fmb2*p7p8-hbcm2*p7p11-2*hbcm2*p7p8)+16*(ffmcfmb*
     . p1p3*p3p6*p7p11+p1p2*p3p6*p7p11+p1p4*p3p6*p7p11)
      ans3=32*(p3p7*p1p5)*(-p6p11+p6p8)+16*(p3p6*p1p5)*(-p7p11-2*p7p8
     . )+32*(p4p7*p1p3)*(p6p11-p6p8)+32*(p4p6*p3p7)*(p1p11-p1p8)+64*(
     . p4p7*p3p6)*(p1p11-p1p8)+16*(p6p7*p3p5)*(p1p11-p1p8)+16*(p6p7*
     . p3p4)*(-p1p11+p1p8)+16*(p6p7*p2p3)*(-p1p11+p1p8)+16*(p6p7*p1p3
     . )*(-p1p11+p1p8)+32*(p3p6*p3p4)*(ffmcfmb*p7p11+ffmcfmb*p7p8-
     . p7p11-p7p8)+32*(p3p6*p2p3)*(-ffmcfmb*p7p8+p7p8)+32*(p3p7*p1p3)
     . *(ffmcfmb*p6p11-ffmcfmb*p6p8)+16*(p3p7*p3p6)*(ffmcfmb*p1p11-
     . ffmcfmb*p1p8-3*ffmcfmb*p2p11+3*ffmcfmb*p2p8+3*ffmcfmb*p3p11-3*
     . ffmcfmb*p3p8+3*ffmcfmb*p4p11-3*ffmcfmb*p4p8+ffmcfmb*p5p11-
     . ffmcfmb*p5p8+3*p1p11-3*p1p8+3*p2p11-3*p2p8-3*p3p11+3*p3p8-3*
     . p4p11+3*p4p8-p5p11+p5p8)+24*p5p7*(ffmcfmb*hbcm2*p6p11-ffmcfmb*
     . hbcm2*p6p8+fmb*hbcm*p6p11-fmb*hbcm*p6p8-hbcm2*p6p11+hbcm2*p6p8
     . )+8*p4p7*(ffmcfmb*hbcm2*p6p11-ffmcfmb*hbcm2*p6p8+fmb*hbcm*
     . p6p11-fmb*hbcm*p6p8-hbcm2*p6p11+hbcm2*p6p8)+16*p4p6*(-ffmcfmb*
     . hbcm2*p7p11-ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p11-fmb*hbcm*p7p8+
     . hbcm2*p7p11+hbcm2*p7p8)+ans4
      ans2=w11*ans3
      ans7=16*(-ffmcfmb*p1p3*p3p6*p7p10-2*ffmcfmb*p1p3*p3p7*p6p10+
     . p1p10*p1p3*p6p7+p1p10*p2p3*p6p7+p1p10*p3p4*p6p7-p1p10*p3p5*
     . p6p7-4*p1p10*p3p6*p4p7-2*p1p10*p3p7*p4p6-p1p2*p3p6*p7p10-2*
     . p1p3*p4p7*p6p10-p1p4*p3p6*p7p10+p1p5*p3p6*p7p10+2*p1p5*p3p7*
     . p6p10)
      ans6=32*(p3p6*p3p4)*(-ffmcfmb*p7p10+p7p10)+16*(p3p7*p3p6)*(-
     . ffmcfmb*p1p10+3*ffmcfmb*p2p10-3*ffmcfmb*p3p10-3*ffmcfmb*p4p10-
     . ffmcfmb*p5p10-3*p1p10-3*p2p10+3*p3p10+3*p4p10+p5p10)+24*p5p7*(
     . -ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10+hbcm2*p6p10)+8*p4p7*(-
     . ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10+hbcm2*p6p10)+16*p4p6*(
     . ffmcfmb*hbcm2*p7p10+fmb*hbcm*p7p10-hbcm2*p7p10)+8*p6p7*(
     . ffmcfmb*hbcm2*p1p10-ffmcfmb*hbcm2*p2p10-ffmcfmb*hbcm2*p4p10+
     . ffmcfmb*hbcm2*p5p10-fmb*hbcm*p1p10-fmb*hbcm*p2p10-fmb*hbcm*
     . p4p10+fmb*hbcm*p5p10+2*fmc*hbcm*p1p10-hbcm2*p1p10+hbcm2*p2p10+
     . hbcm2*p4p10-hbcm2*p5p10)+8*p3p7*(-ffmcfmb**2*hbcm2*p6p10-3*
     . ffmcfmb*fmb*hbcm*p6p10+ffmcfmb*fmc*hbcm*p6p10-2*ffmcfmb*hbcm2*
     . p6p10+fmb*fmc*p6p10-fmb*hbcm*p6p10-2*fmb2*p6p10-fmc*hbcm*p6p10
     . +3*hbcm2*p6p10)+8*p3p6*(-ffmcfmb**2*hbcm2*p7p10+2*fmb*fmc*
     . p7p10-fmb2*p7p10+hbcm2*p7p10)+ans7
      ans5=w13*ans6
      ans8=(8*p3p4*(-p6p12+3*p7p13)+8*p6p7*(-3*p3p11-p3p8)+24*p2p3*(
     . p6p12-p7p13)+8*p3p6*(-p2p12-p3p12-3*p4p12-2*p5p12+p7p10+3*
     . p7p11+p7p8)+8*p3p7*(-2*ffmcfmb*p3p13-p1p13-3*p2p13+5*p3p13+5*
     . p4p13+3*p5p13-2*p6p11)+4*(2*ffmcfmb*hbcm2*p6p12+3*fmb*hbcm*
     . p6p12-3*fmb*hbcm*p7p13-fmc*hbcm*p6p12+3*fmc*hbcm*p7p13-3*hbcm2
     . *p6p12+3*hbcm2*p7p13+4*p1p3*p6p12+2*p3p12*p4p6-4*p3p13*p4p7))
      ans1=w1*(32*(p3p6*p3p4)*(ffmcfmb*p7p8-p7p8)+32*(p3p6*p2p3)*(-
     . ffmcfmb*p7p8+p7p8)+16*(p3p7*p3p6)*(-ffmcfmb*p1p8+3*ffmcfmb*
     . p2p8-3*ffmcfmb*p3p8-3*ffmcfmb*p4p8-ffmcfmb*p5p8-3*p1p8-3*p2p8+
     . 3*p3p8+3*p4p8+p5p8)+24*p5p7*(-ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p8
     . +hbcm2*p6p8)+8*p4p7*(-ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p8+hbcm2*
     . p6p8)+16*p4p6*(-ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p8+hbcm2*p7p8)+8
     . *p6p7*(ffmcfmb*hbcm2*p1p8-ffmcfmb*hbcm2*p2p8-ffmcfmb*hbcm2*
     . p4p8+ffmcfmb*hbcm2*p5p8-fmb*hbcm*p1p8-fmb*hbcm*p2p8-fmb*hbcm*
     . p4p8+fmb*hbcm*p5p8+2*fmc*hbcm*p1p8-hbcm2*p1p8+hbcm2*p2p8+hbcm2
     . *p4p8-hbcm2*p5p8)+8*p3p6*(-ffmcfmb*fmb*hbcm*p7p8+2*ffmcfmb*
     . hbcm2*p7p8-fmb*fmc*p7p8+2*fmb2*p7p8-2*hbcm2*p7p8)+8*p3p7*(-
     . ffmcfmb**2*hbcm2*p6p8-3*ffmcfmb*fmb*hbcm*p6p8+ffmcfmb*fmc*hbcm
     . *p6p8-2*ffmcfmb*hbcm2*p6p8+fmb*fmc*p6p8-fmb*hbcm*p6p8-2*fmb2*
     . p6p8-fmc*hbcm*p6p8+3*hbcm2*p6p8)+16*(-2*ffmcfmb*p1p3*p3p7*p6p8
     . +p1p3*p1p8*p6p7-2*p1p3*p4p7*p6p8-2*p1p5*p3p6*p7p8+2*p1p5*p3p7*
     . p6p8+p1p8*p2p3*p6p7+p1p8*p3p4*p6p7-p1p8*p3p5*p6p7-4*p1p8*p3p6*
     . p4p7-2*p1p8*p3p7*p4p6))+ans2+ans5+ans8
      ans=ccc*ans1
      b(30)=ans
      b(31)=ccc*(w11*(16*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p6*(ffmcfmb
     . **2*hbcm+4*ffmcfmb*fmb+3*ffmcfmb*fmc-ffmcfmb*hbcm-3*fmc))+w1*(
     . 16*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p6*(ffmcfmb**2*hbcm+4*
     . ffmcfmb*fmb+3*ffmcfmb*fmc-ffmcfmb*hbcm-3*fmc)))
      ans3=w13*(32*p4p7*(ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+16*
     . p4p6*(ffmcfmb*hbcm*p7p10+fmb*p7p10-hbcm*p7p10)+16*p3p7*(
     . ffmcfmb*hbcm*p6p10+fmb*p6p10-hbcm*p6p10)+8*p6p7*(-ffmcfmb*hbcm
     . *p1p10-ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*
     . p5p10-fmb*p1p10-fmb*p2p10-fmb*p4p10+fmb*p5p10+hbcm*p1p10+hbcm*
     . p2p10+hbcm*p4p10-hbcm*p5p10)+16*p3p6*(ffmcfmb*fmb*p7p10+
     . ffmcfmb*fmc*p7p10-fmc*p7p10))+4*(fmb*p6p12-3*fmb*p7p13+fmc*
     . p6p12-3*fmc*p7p13-hbcm*p6p12+3*hbcm*p7p13)
      ans2=w11*(32*p4p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8-fmb*
     . p6p11+fmb*p6p8+hbcm*p6p11-hbcm*p6p8)+16*p4p6*(-ffmcfmb*hbcm*
     . p7p11+ffmcfmb*hbcm*p7p8-fmb*p7p11+fmb*p7p8+hbcm*p7p11-hbcm*
     . p7p8)+16*p3p7*(-ffmcfmb*hbcm*p6p11+ffmcfmb*hbcm*p6p8-fmb*p6p11
     . +fmb*p6p8+hbcm*p6p11-hbcm*p6p8)+8*p6p7*(ffmcfmb*hbcm*p1p11-
     . ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p11-ffmcfmb*hbcm*p2p8+ffmcfmb
     . *hbcm*p4p11-ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p11+ffmcfmb*hbcm*
     . p5p8+fmb*p1p11-fmb*p1p8+fmb*p2p11-fmb*p2p8+fmb*p4p11-fmb*p4p8-
     . fmb*p5p11+fmb*p5p8-hbcm*p1p11+hbcm*p1p8-hbcm*p2p11+hbcm*p2p8-
     . hbcm*p4p11+hbcm*p4p8+hbcm*p5p11-hbcm*p5p8)+16*p3p6*(-ffmcfmb*
     . fmb*p7p11+ffmcfmb*fmb*p7p8-ffmcfmb*fmc*p7p11+ffmcfmb*fmc*p7p8+
     . fmc*p7p11-fmc*p7p8))+ans3
      ans1=w1*(32*p4p7*(ffmcfmb*hbcm*p6p8+fmb*p6p8-hbcm*p6p8)+16*p4p6
     . *(ffmcfmb*hbcm*p7p8+fmb*p7p8-hbcm*p7p8)+16*p3p7*(ffmcfmb*hbcm*
     . p6p8+fmb*p6p8-hbcm*p6p8)+8*p6p7*(-ffmcfmb*hbcm*p1p8-ffmcfmb*
     . hbcm*p2p8-ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p8-fmb*p1p8-fmb*
     . p2p8-fmb*p4p8+fmb*p5p8+hbcm*p1p8+hbcm*p2p8+hbcm*p4p8-hbcm*p5p8
     . )+16*p3p6*(ffmcfmb*fmb*p7p8+ffmcfmb*fmc*p7p8-fmc*p7p8))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(35)=ccc*(8*w1*(-ffmcfmb*hbcm*p7p8+fmc*p7p8)+8*w11*(ffmcfmb*
     . hbcm*p7p11-ffmcfmb*hbcm*p7p8-fmc*p7p11+fmc*p7p8)+8*w13*(-
     . ffmcfmb*hbcm*p7p10+fmc*p7p10))
      b(38)=ccc*(w1*(8*p3p7*(p1p8+p2p8+p4p8-p5p8)+8*(-ffmcfmb*hbcm2*
     . p7p8-fmc*hbcm*p7p8-2*p3p4*p7p8))+w11*(16*p3p4*(p7p11-p7p8)+8*
     . p3p7*(-p1p11+p1p8-p2p11+p2p8-p4p11+p4p8+p5p11-p5p8)+8*(ffmcfmb
     . *hbcm2*p7p11-ffmcfmb*hbcm2*p7p8+fmc*hbcm*p7p11-fmc*hbcm*p7p8))
     . +w13*(8*p3p7*(p1p10+p2p10+p4p10-p5p10)+8*(-ffmcfmb*hbcm2*p7p10
     . -fmc*hbcm*p7p10-2*p3p4*p7p10))-4*p3p12)
      b(39)=w11*ccc*((p3p7*p3p6)*(-32*ffmcfmb-16)+8*p6p7*(ffmcfmb*
     . hbcm2+fmc*hbcm-hbcm2)+8*(p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-p3p5*
     . p6p7-6*p3p6*p4p7-2*p3p7*p4p6))
      b(40)=w11*ccc*(16*(p4p6*p3p7)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p7*
     . p3p6)*(2*ffmcfmb*hbcm+fmb-2*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*hbcm+
     . fmb-hbcm)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p6p7*p2p3)
     . *(-ffmcfmb*hbcm-fmb+hbcm)+8*(p6p7*p1p3)*(-2*ffmcfmb*hbcm-fmb+
     . hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*
     . fmc*hbcm2+2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmb*hbcm2-fmb2*hbcm
     . +2*fmc*hbcm2-hbcm3)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm+2*ffmcfmb
     . *fmb+2*ffmcfmb*fmc-2*ffmcfmb*hbcm-fmb-2*fmc)+8*(-3*fmb*p3p6*
     . p5p7-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7+hbcm*p1p5*p6p7))
      b(41)=w11*ccc*((p5p7*p3p6)*(24*ffmcfmb-24)+(p4p7*p3p6)*(-8*
     . ffmcfmb+8)+(p3p7*p3p6)*(8*ffmcfmb-8)+4*p6p7*(-ffmcfmb**2*hbcm2
     . -2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+2*fmb*fmc-fmb2-2*fmc*
     . hbcm+hbcm2)+8*(-ffmcfmb*p1p3*p6p7-p1p2*p6p7-p1p4*p6p7+p1p5*
     . p6p7))
      b(43)=ccc*(w1*(8*p6p7*(-p1p8-p2p8-p4p8+p5p8)+16*p3p7*(ffmcfmb*
     . p6p8+p6p8)+16*(ffmcfmb*p3p6*p7p8+p4p6*p7p8+3*p4p7*p6p8))+w11*(
     . 48*p4p7*(-p6p11+p6p8)+16*p4p6*(-p7p11+p7p8)+8*p6p7*(p1p11-p1p8
     . +p2p11-p2p8+p4p11-p4p8-p5p11+p5p8)+16*p3p7*(-ffmcfmb*p6p11+
     . ffmcfmb*p6p8-p6p11+p6p8)+16*p3p6*(-ffmcfmb*p7p11+ffmcfmb*p7p8)
     . )+w13*(8*p6p7*(-p1p10-p2p10-p4p10+p5p10)+16*p3p7*(ffmcfmb*
     . p6p10+p6p10)+16*(ffmcfmb*p3p6*p7p10+p4p6*p7p10+3*p4p7*p6p10))+
     . 4*(-p6p12-3*p7p13))
      b(44)=w11*ccc*((p3p6*p3p4)*(-32*ffmcfmb+32)+(p3p6*p2p3)*(16*
     . ffmcfmb-16)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2+3*fmb*fmc-3*
     . fmb2+3*hbcm2)+8*(-ffmcfmb*p1p3*p3p6-p1p2*p3p6-p1p4*p3p6+3*p1p5
     . *p3p6))
      b(45)=16*ccc*w11*(-ffmcfmb*p3p6-p4p6)
      b(46)=w11*ccc*(16*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p6*(ffmcfmb
     . **2*hbcm+4*ffmcfmb*fmb+3*ffmcfmb*fmc-ffmcfmb*hbcm-3*fmc))
      b(49)=w11*ccc*(4*p3p7*(-4*ffmcfmb*hbcm+2*fmc+hbcm)+12*(-hbcm*
     . p4p7+hbcm*p5p7))
      b(50)=w11*ccc*(p3p7*(-16*ffmcfmb+4)+4*(-5*p4p7+3*p5p7))
      b(51)=w11*ccc*(12*p5p7*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p4p7*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p3p7*(ffmcfmb**2*hbcm2+2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmb*fmc-
     . fmb*hbcm+fmb2+2*fmc*hbcm)+8*(3*ffmcfmb*p1p3*p3p7+3*p1p3*p4p7-3
     . *p1p3*p5p7+2*p1p4*p3p7))
      b(53)=ccc*(4*w1*(2*p2p8-3*p3p8-4*p4p8)+4*w11*(-2*p2p11+2*p2p8+3
     . *p3p11-3*p3p8+4*p4p11-4*p4p8)+4*w13*(2*p2p10-3*p3p10-4*p4p10))
      b(54)=2*ccc*w11*(-5*ffmcfmb*hbcm2-3*fmc*hbcm+4*p2p3-8*p3p4)
      b(55)=6*ccc*w11*(-ffmcfmb*hbcm+fmc)
      b(56)=w11*ccc*(p1p3*(4*ffmcfmb+12)+2*(3*ffmcfmb*fmb*hbcm-3*
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2-3*fmb*fmc+3*fmb2+3*fmc*hbcm-3
     . *hbcm2-4*p1p2+8*p1p4))
      b(57)=-8*ccc
      b(58)=w11*ccc*(4*p3p6*(ffmcfmb*hbcm+3*fmc)+16*hbcm*p4p6)
      b(59)=w11*ccc*(16*p3p4*(-ffmcfmb*hbcm-fmb+hbcm)+8*p2p3*(ffmcfmb
     . *hbcm+fmb-hbcm)+4*p1p3*(2*ffmcfmb*hbcm-3*fmc-3*hbcm)+2*(-2*
     . ffmcfmb**2*hbcm3-5*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-3*fmb*fmc*hbcm-3*fmb2*hbcm+3*fmc*hbcm2+4*hbcm*
     . p1p2-8*hbcm*p1p4+3*hbcm3))
      b(61)=-4*ccc*hbcm
      b(62)=-8*ccc*fmb*p3p6
      b(63)=w13*ccc*((p3p7*p3p6)*(32*ffmcfmb+16)+8*p6p7*(-ffmcfmb*
     . hbcm2-fmc*hbcm+hbcm2)+8*(-p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+p3p5*
     . p6p7+6*p3p6*p4p7+2*p3p7*p4p6))
      b(64)=ccc*(w13*((p5p7*p3p6)*(-24*ffmcfmb+24)+(p4p7*p3p6)*(8*
     . ffmcfmb-8)+(p3p7*p3p6)*(-8*ffmcfmb+8)+4*p6p7*(ffmcfmb**2*hbcm2
     . +2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*fmb*fmc+fmb2+2*fmc*
     . hbcm-hbcm2)+8*(ffmcfmb*p1p3*p6p7+p1p2*p6p7+p1p4*p6p7-p1p5*p6p7
     . ))-4*p6p7)
      b(65)=ccc*(w11*((p3p7*p3p6)*(32*ffmcfmb+16)+8*p6p7*(-ffmcfmb*
     . hbcm2-fmc*hbcm+hbcm2)+8*(-p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+p3p5*
     . p6p7+6*p3p6*p4p7+2*p3p7*p4p6))+w1*((p3p7*p3p6)*(32*ffmcfmb+16)
     . +8*p6p7*(-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+8*(-p1p3*p6p7-p2p3*
     . p6p7-p3p4*p6p7+p3p5*p6p7+6*p3p6*p4p7+2*p3p7*p4p6)))
      b(66)=ccc*(w1*(8*p6p7*(-hbcm*p1p8-hbcm*p2p8-hbcm*p4p8+hbcm*p5p8
     . )+8*p3p7*(-ffmcfmb*hbcm*p6p8+fmc*p6p8+2*hbcm*p6p8)+16*(fmc*
     . p3p6*p7p8+hbcm*p4p6*p7p8+2*hbcm*p4p7*p6p8))+w11*(32*p4p7*(-
     . hbcm*p6p11+hbcm*p6p8)+16*p4p6*(-hbcm*p7p11+hbcm*p7p8)+8*p6p7*(
     . hbcm*p1p11-hbcm*p1p8+hbcm*p2p11-hbcm*p2p8+hbcm*p4p11-hbcm*p4p8
     . -hbcm*p5p11+hbcm*p5p8)+16*p3p6*(-fmc*p7p11+fmc*p7p8)+8*p3p7*(
     . ffmcfmb*hbcm*p6p11-ffmcfmb*hbcm*p6p8-fmc*p6p11+fmc*p6p8-2*hbcm
     . *p6p11+2*hbcm*p6p8))+w13*(8*p6p7*(-hbcm*p1p10-hbcm*p2p10-hbcm*
     . p4p10+hbcm*p5p10)+8*p3p7*(-ffmcfmb*hbcm*p6p10+fmc*p6p10+2*hbcm
     . *p6p10)+16*(fmc*p3p6*p7p10+hbcm*p4p6*p7p10+2*hbcm*p4p7*p6p10))
     . +4*(hbcm*p6p12-3*hbcm*p7p13))
      b(67)=16*ccc*w13*(ffmcfmb*p3p6+p4p6)
      b(68)=ccc*(16*w11*(ffmcfmb*p3p6+p4p6)+16*w1*(ffmcfmb*p3p6+p4p6)
     . )
      b(69)=w13*ccc*(p3p7*(16*ffmcfmb-4)+4*(5*p4p7-3*p5p7))
      b(70)=ccc*(w13*(12*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p4p7*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p3p7*(-ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb*fmc+
     . fmb*hbcm-fmb2-2*fmc*hbcm)+8*(-3*ffmcfmb*p1p3*p3p7-3*p1p3*p4p7+
     . 3*p1p3*p5p7-2*p1p4*p3p7))+4*p3p7)
      b(71)=ccc*(w11*(p3p7*(16*ffmcfmb-4)+4*(5*p4p7-3*p5p7))+w1*(p3p7
     . *(16*ffmcfmb-4)+4*(5*p4p7-3*p5p7)))
      b(72)=ccc*(w13*(p1p3*(-4*ffmcfmb-12)+2*(-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmb2-3*fmc*hbcm+3
     . *hbcm2+4*p1p2-8*p1p4))+2)
      b(73)=8*ccc*w11*p6p7*(-ffmcfmb*hbcm+fmc)
      b(74)=ccc*(8*w11*p6p7*(ffmcfmb*hbcm-fmc)+8*w1*p6p7*(ffmcfmb*
     . hbcm-fmc))
      b(75)=ccc*(4*w1*(3*ffmcfmb*hbcm*p1p8+2*ffmcfmb*hbcm*p2p8-3*
     . ffmcfmb*hbcm*p3p8-4*ffmcfmb*hbcm*p4p8+2*fmb*p2p8-3*fmb*p3p8-4*
     . fmb*p4p8-3*fmc*p1p8-2*hbcm*p2p8+3*hbcm*p3p8+4*hbcm*p4p8)+4*w11
     . *(-3*ffmcfmb*hbcm*p1p11+3*ffmcfmb*hbcm*p1p8-2*ffmcfmb*hbcm*
     . p2p11+2*ffmcfmb*hbcm*p2p8+3*ffmcfmb*hbcm*p3p11-3*ffmcfmb*hbcm*
     . p3p8+4*ffmcfmb*hbcm*p4p11-4*ffmcfmb*hbcm*p4p8-2*fmb*p2p11+2*
     . fmb*p2p8+3*fmb*p3p11-3*fmb*p3p8+4*fmb*p4p11-4*fmb*p4p8+3*fmc*
     . p1p11-3*fmc*p1p8+2*hbcm*p2p11-2*hbcm*p2p8-3*hbcm*p3p11+3*hbcm*
     . p3p8-4*hbcm*p4p11+4*hbcm*p4p8)+4*w13*(3*ffmcfmb*hbcm*p1p10+2*
     . ffmcfmb*hbcm*p2p10-3*ffmcfmb*hbcm*p3p10-4*ffmcfmb*hbcm*p4p10+2
     . *fmb*p2p10-3*fmb*p3p10-4*fmb*p4p10-3*fmc*p1p10-2*hbcm*p2p10+3*
     . hbcm*p3p10+4*hbcm*p4p10))
      ans2=w13*(8*p1p3*(-3*p1p10-2*p2p10+3*p3p10+4*p4p10)+4*(-5*
     . ffmcfmb*hbcm2*p1p10-2*ffmcfmb*hbcm2*p2p10+3*ffmcfmb*hbcm2*
     . p3p10+4*ffmcfmb*hbcm2*p4p10-2*fmb*hbcm*p2p10+3*fmb*hbcm*p3p10+
     . 4*fmb*hbcm*p4p10-3*fmc*hbcm*p1p10+6*hbcm2*p1p10+2*hbcm2*p2p10-
     . 3*hbcm2*p3p10-4*hbcm2*p4p10-2*p1p10*p2p3-2*p1p10*p3p4+6*p1p10*
     . p3p5))
      ans1=w1*(8*p1p3*(-3*p1p8-2*p2p8+3*p3p8+4*p4p8)+4*(-5*ffmcfmb*
     . hbcm2*p1p8-2*ffmcfmb*hbcm2*p2p8+3*ffmcfmb*hbcm2*p3p8+4*ffmcfmb
     . *hbcm2*p4p8-2*fmb*hbcm*p2p8+3*fmb*hbcm*p3p8+4*fmb*hbcm*p4p8-3*
     . fmc*hbcm*p1p8+6*hbcm2*p1p8+2*hbcm2*p2p8-3*hbcm2*p3p8-4*hbcm2*
     . p4p8-2*p1p8*p2p3-2*p1p8*p3p4+6*p1p8*p3p5))+w11*(24*p3p5*(-
     . p1p11+p1p8)+8*p3p4*(p1p11-p1p8)+8*p2p3*(p1p11-p1p8)+8*p1p3*(3*
     . p1p11-3*p1p8+2*p2p11-2*p2p8-3*p3p11+3*p3p8-4*p4p11+4*p4p8)+4*(
     . 5*ffmcfmb*hbcm2*p1p11-5*ffmcfmb*hbcm2*p1p8+2*ffmcfmb*hbcm2*
     . p2p11-2*ffmcfmb*hbcm2*p2p8-3*ffmcfmb*hbcm2*p3p11+3*ffmcfmb*
     . hbcm2*p3p8-4*ffmcfmb*hbcm2*p4p11+4*ffmcfmb*hbcm2*p4p8+2*fmb*
     . hbcm*p2p11-2*fmb*hbcm*p2p8-3*fmb*hbcm*p3p11+3*fmb*hbcm*p3p8-4*
     . fmb*hbcm*p4p11+4*fmb*hbcm*p4p8+3*fmc*hbcm*p1p11-3*fmc*hbcm*
     . p1p8-6*hbcm2*p1p11+6*hbcm2*p1p8-2*hbcm2*p2p11+2*hbcm2*p2p8+3*
     . hbcm2*p3p11-3*hbcm2*p3p8+4*hbcm2*p4p11-4*hbcm2*p4p8))+ans2
      ans=ccc*ans1
      b(76)=ans
      b(77)=ccc*(w1*(8*p3p6*(3*p1p8+p2p8+p4p8-3*p5p8)+4*(-5*ffmcfmb*
     . hbcm2*p6p8-3*fmc*hbcm*p6p8+6*hbcm2*p6p8-6*p1p3*p6p8-2*p2p3*
     . p6p8-2*p3p4*p6p8+6*p3p5*p6p8))+w11*(24*p3p5*(-p6p11+p6p8)+8*
     . p3p4*(p6p11-p6p8)+8*p2p3*(p6p11-p6p8)+24*p1p3*(p6p11-p6p8)+8*
     . p3p6*(-3*p1p11+3*p1p8-p2p11+p2p8-p4p11+p4p8+3*p5p11-3*p5p8)+4*
     . (5*ffmcfmb*hbcm2*p6p11-5*ffmcfmb*hbcm2*p6p8+3*fmc*hbcm*p6p11-3
     . *fmc*hbcm*p6p8-6*hbcm2*p6p11+6*hbcm2*p6p8))+w13*(8*p3p6*(3*
     . p1p10+p2p10+p4p10-3*p5p10)+4*(-5*ffmcfmb*hbcm2*p6p10-3*fmc*
     . hbcm*p6p10+6*hbcm2*p6p10-6*p1p3*p6p10-2*p2p3*p6p10-2*p3p4*
     . p6p10+6*p3p5*p6p10)))
      b(78)=ccc*(12*w1*(ffmcfmb*hbcm*p6p8-fmc*p6p8)+12*w11*(-ffmcfmb*
     . hbcm*p6p11+ffmcfmb*hbcm*p6p8+fmc*p6p11-fmc*p6p8)+12*w13*(
     . ffmcfmb*hbcm*p6p10-fmc*p6p10))
      b(81)=8*ccc*w13*p6p7*(ffmcfmb*hbcm-fmc)
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(4*p3p7*(2*ffmcfmb*hbcm+2*fmb+2*fmc-hbcm)+4*(hbcm*
     . p4p7+3*hbcm*p5p7))
      b(84)=4*ccc*(p3p7-p4p7+3*p5p7)
      b(85)=2*ccc*(-ffmcfmb*hbcm2+4*fmb*hbcm-3*fmc*hbcm-4*hbcm2+4*
     . p2p3-8*p3p4)
      b(86)=-8*ccc
      b(87)=2*ccc*(ffmcfmb*hbcm+4*fmb+3*fmc-4*hbcm)
      b(88)=w13*ccc*(12*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+12*p4p7*(
     . ffmcfmb*hbcm+fmb-hbcm)+4*p3p7*(2*ffmcfmb**2*hbcm+2*ffmcfmb*fmb
     . -3*ffmcfmb*hbcm-fmb+hbcm))
      b(89)=-8*ccc*hbcm
      b(90)=w11*ccc*(12*p5p7*(ffmcfmb*hbcm+fmb-hbcm)+12*p4p7*(-
     . ffmcfmb*hbcm-fmb+hbcm)+4*p3p7*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*
     . fmb+3*ffmcfmb*hbcm+fmb-hbcm))
      b(91)=ccc*(w11*(12*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+12*p4p7*(
     . ffmcfmb*hbcm+fmb-hbcm)+4*p3p7*(2*ffmcfmb**2*hbcm+2*ffmcfmb*fmb
     . -3*ffmcfmb*hbcm-fmb+hbcm))+w1*(12*p5p7*(-ffmcfmb*hbcm-fmb+hbcm
     . )+12*p4p7*(ffmcfmb*hbcm+fmb-hbcm)+4*p3p7*(2*ffmcfmb**2*hbcm+2*
     . ffmcfmb*fmb-3*ffmcfmb*hbcm-fmb+hbcm)))
      b(92)=8*ccc*(-hbcm2+2*p1p3+p2p3-p3p4-p3p5)
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(ffmcfmb*hbcm+fmb-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp32_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(1)=w5*ccc*(4*p6p7*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm-2*fmc2)+8*(-ffmcfmb*
     . p3p6*p3p7+ffmcfmb*p3p6*p4p7-3*ffmcfmb*p3p6*p5p7+p1p2*p6p7-p1p3
     . *p6p7+p1p4*p6p7-p1p5*p6p7))
      b(2)=w2*ccc*(4*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb
     . *fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+8*(ffmcfmb*
     . p3p6*p3p7-ffmcfmb*p3p6*p4p7+3*ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3
     . *p6p7-p1p4*p6p7+p1p5*p6p7))
      ans=ccc*(w2*(8*p3p7*(-2*ffmcfmb*p1p9+p1p9)+4*(-ffmcfmb**2*hbcm2
     . *p7p9+ffmcfmb*fmb*hbcm*p7p9-ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*
     . hbcm2*p7p9-fmb*fmc*p7p9+fmc*hbcm*p7p9+2*fmc2*p7p9-4*p1p4*p7p9-
     . 2*p1p9*p4p7+2*p1p9*p5p7))+w5*(16*p1p4*(-p7p10+p7p8)+8*p5p7*(
     . p1p10-p1p8)+8*p4p7*(-p1p10+p1p8)+8*p3p7*(-2*ffmcfmb*p1p10+2*
     . ffmcfmb*p1p8+p1p10-p1p8)+4*(-ffmcfmb**2*hbcm2*p7p10+ffmcfmb**2
     . *hbcm2*p7p8+ffmcfmb*fmb*hbcm*p7p10-ffmcfmb*fmb*hbcm*p7p8-
     . ffmcfmb*fmc*hbcm*p7p10+ffmcfmb*fmc*hbcm*p7p8-ffmcfmb*hbcm2*
     . p7p10+ffmcfmb*hbcm2*p7p8-fmb*fmc*p7p10+fmb*fmc*p7p8+fmc*hbcm*
     . p7p10-fmc*hbcm*p7p8+2*fmc2*p7p10-2*fmc2*p7p8))+w12*(8*p3p7*(-2
     . *ffmcfmb*p1p10+p1p10)+4*(-ffmcfmb**2*hbcm2*p7p10+ffmcfmb*fmb*
     . hbcm*p7p10-ffmcfmb*fmc*hbcm*p7p10-ffmcfmb*hbcm2*p7p10-fmb*fmc*
     . p7p10+fmc*hbcm*p7p10+2*fmc2*p7p10-2*p1p10*p4p7+2*p1p10*p5p7-4*
     . p1p4*p7p10))+4*(-p1p12-p3p12-p4p12-2*p5p12+p7p11+p7p9))
      b(3)=ans
      b(4)=w5*ccc*((p3p7*p1p3)*(-16*ffmcfmb+8)+12*p5p7*(ffmcfmb*hbcm2
     . -fmc*hbcm)+4*p4p7*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p3p7*(-ffmcfmb**
     . 2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc+2*fmc2)+8*(-
     . p1p3*p4p7+p1p3*p5p7-2*p1p4*p3p7))
      b(5)=w2*ccc*((p3p7*p1p3)*(16*ffmcfmb-8)+12*p5p7*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+4*p4p7*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p7*(ffmcfmb**2*
     . hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-2*fmc2)+8*(
     . p1p3*p4p7-p1p3*p5p7+2*p1p4*p3p7))
      ans3=w12*(16*p3p4*(-ffmcfmb*hbcm*p7p10+fmc*p7p10)+16*p1p3*(
     . ffmcfmb*hbcm*p7p10-fmc*p7p10)+8*p3p7*(ffmcfmb*hbcm*p1p10+
     . ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p4p10-ffmcfmb*hbcm*p5p10+fmc*
     . p1p10-fmc*p2p10-fmc*p4p10+fmc*p5p10-hbcm*p1p10)+4*(-ffmcfmb**2
     . *hbcm3*p7p10+ffmcfmb*fmb*hbcm2*p7p10+3*ffmcfmb*fmc*hbcm2*p7p10
     . -ffmcfmb*hbcm3*p7p10-fmb*fmc*hbcm*p7p10+fmc*hbcm2*p7p10-2*fmc2
     . *hbcm*p7p10+2*hbcm*p1p10*p4p7-2*hbcm*p1p10*p5p7+4*hbcm*p1p4*
     . p7p10))+4*(hbcm*p2p12+hbcm*p3p12+2*hbcm*p4p12+hbcm*p5p12+hbcm*
     . p7p10-2*hbcm*p7p11-hbcm*p7p8)
      ans2=w5*(16*p1p4*(hbcm*p7p10-hbcm*p7p8)+8*p5p7*(-hbcm*p1p10+
     . hbcm*p1p8)+8*p4p7*(hbcm*p1p10-hbcm*p1p8)+16*p3p4*(-ffmcfmb*
     . hbcm*p7p10+ffmcfmb*hbcm*p7p8+fmc*p7p10-fmc*p7p8)+16*p1p3*(
     . ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p8-fmc*p7p10+fmc*p7p8)+8*
     . p3p7*(ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p10-
     . ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p4p10-ffmcfmb*hbcm*p4p8-ffmcfmb
     . *hbcm*p5p10+ffmcfmb*hbcm*p5p8+fmc*p1p10-fmc*p1p8-fmc*p2p10+fmc
     . *p2p8-fmc*p4p10+fmc*p4p8+fmc*p5p10-fmc*p5p8-hbcm*p1p10+hbcm*
     . p1p8)+4*(-ffmcfmb**2*hbcm3*p7p10+ffmcfmb**2*hbcm3*p7p8+ffmcfmb
     . *fmb*hbcm2*p7p10-ffmcfmb*fmb*hbcm2*p7p8+3*ffmcfmb*fmc*hbcm2*
     . p7p10-3*ffmcfmb*fmc*hbcm2*p7p8-ffmcfmb*hbcm3*p7p10+ffmcfmb*
     . hbcm3*p7p8-fmb*fmc*hbcm*p7p10+fmb*fmc*hbcm*p7p8+fmc*hbcm2*
     . p7p10-fmc*hbcm2*p7p8-2*fmc2*hbcm*p7p10+2*fmc2*hbcm*p7p8))+ans3
      ans1=w2*(16*p3p4*(-ffmcfmb*hbcm*p7p9+fmc*p7p9)+8*p3p7*(ffmcfmb*
     . hbcm*p1p9+ffmcfmb*hbcm*p2p9+ffmcfmb*hbcm*p4p9-ffmcfmb*hbcm*
     . p5p9+fmc*p1p9-fmc*p2p9-fmc*p4p9+fmc*p5p9-hbcm*p1p9)+16*p1p3*(
     . ffmcfmb*hbcm*p7p9-fmc*p7p9)+4*(-ffmcfmb**2*hbcm3*p7p9+ffmcfmb*
     . fmb*hbcm2*p7p9+3*ffmcfmb*fmc*hbcm2*p7p9-ffmcfmb*hbcm3*p7p9-fmb
     . *fmc*hbcm*p7p9+fmc*hbcm2*p7p9-2*fmc2*hbcm*p7p9+4*hbcm*p1p4*
     . p7p9+2*hbcm*p1p9*p4p7-2*hbcm*p1p9*p5p7))+ans2
      ans=ccc*ans1
      b(6)=ans
      b(7)=ccc*(w2*(8*p3p6*(-ffmcfmb*p1p9+ffmcfmb*p2p9-2*ffmcfmb*p3p9
     . -3*ffmcfmb*p4p9+ffmcfmb*p5p9+p1p9)+8*(-3*p1p9*p4p6+p1p9*p5p6))
     . +w5*(8*p5p6*(p1p10-p1p8)+24*p4p6*(-p1p10+p1p8)+8*p3p6*(-
     . ffmcfmb*p1p10+ffmcfmb*p1p8+ffmcfmb*p2p10-ffmcfmb*p2p8-2*
     . ffmcfmb*p3p10+2*ffmcfmb*p3p8-3*ffmcfmb*p4p10+3*ffmcfmb*p4p8+
     . ffmcfmb*p5p10-ffmcfmb*p5p8+p1p10-p1p8))+w12*(8*p3p6*(-ffmcfmb*
     . p1p10+ffmcfmb*p2p10-2*ffmcfmb*p3p10-3*ffmcfmb*p4p10+ffmcfmb*
     . p5p10+p1p10)+8*(-3*p1p10*p4p6+p1p10*p5p6))+4*(-3*p1p13-p2p13+2
     . *p3p13-p4p13+3*p5p13))
      b(8)=w5*ccc*((p3p6*p1p3)*(-8*ffmcfmb+8)+4*p5p6*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+12*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p6*(-ffmcfmb**2
     . *hbcm2+3*ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+
     . fmc*hbcm)+8*(ffmcfmb*p2p3*p3p6-3*ffmcfmb*p3p4*p3p6+ffmcfmb*
     . p3p5*p3p6-3*p1p3*p4p6+p1p3*p5p6))
      b(9)=w2*ccc*((p3p6*p1p3)*(8*ffmcfmb-8)+4*p5p6*(ffmcfmb*hbcm2-
     . fmc*hbcm)+12*p4p6*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p3p6*(ffmcfmb**2
     . *hbcm2-3*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-
     . fmc*hbcm)+8*(-ffmcfmb*p2p3*p3p6+3*ffmcfmb*p3p4*p3p6-ffmcfmb*
     . p3p5*p3p6+3*p1p3*p4p6-p1p3*p5p6))
      ans=ccc*(8*w2*p3p6*(-3*ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9+2*
     . ffmcfmb*hbcm*p3p9+3*ffmcfmb*hbcm*p4p9-ffmcfmb*hbcm*p5p9)+w5*(8
     . *p5p6*(ffmcfmb*hbcm*p3p10-ffmcfmb*hbcm*p3p8-fmc*p3p10+fmc*p3p8
     . )+24*p4p6*(-ffmcfmb*hbcm*p3p10+ffmcfmb*hbcm*p3p8+fmc*p3p10-fmc
     . *p3p8)+8*p3p6*(-ffmcfmb**2*hbcm*p3p10+ffmcfmb**2*hbcm*p3p8+3*
     . ffmcfmb*fmb*p3p10-3*ffmcfmb*fmb*p3p8+4*ffmcfmb*fmc*p3p10-4*
     . ffmcfmb*fmc*p3p8-3*ffmcfmb*hbcm*p1p10+3*ffmcfmb*hbcm*p1p8-
     . ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p2p8+3*ffmcfmb*hbcm*p4p10-3*
     . ffmcfmb*hbcm*p4p8-ffmcfmb*hbcm*p5p10+ffmcfmb*hbcm*p5p8-fmc*
     . p3p10+fmc*p3p8))+w12*(8*p5p6*(ffmcfmb*hbcm*p3p10-fmc*p3p10)+24
     . *p4p6*(-ffmcfmb*hbcm*p3p10+fmc*p3p10)+8*p3p6*(-ffmcfmb**2*hbcm
     . *p3p10+3*ffmcfmb*fmb*p3p10+4*ffmcfmb*fmc*p3p10-3*ffmcfmb*hbcm*
     . p1p10-ffmcfmb*hbcm*p2p10+3*ffmcfmb*hbcm*p4p10-ffmcfmb*hbcm*
     . p5p10-fmc*p3p10))+4*(-2*fmb*p3p13-2*fmc*p3p13+3*hbcm*p1p13+
     . hbcm*p2p13+hbcm*p4p13-3*hbcm*p5p13))
      b(10)=ans
      b(11)=w5*ccc*(8*(p6p7*p1p3)*(-fmb-fmc)+16*(p4p6*p3p7)*(ffmcfmb*
     . hbcm-fmc)+8*(p5p7*p3p6)*(-3*ffmcfmb*hbcm+fmc)+8*(p4p7*p3p6)*(
     . ffmcfmb*hbcm-fmc)+8*(p6p7*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p6p7*
     . p3p4)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm+fmc)+4*
     . p6p7*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+8*(p3p7*p3p6
     . )*(ffmcfmb**2*hbcm-ffmcfmb*fmb-4*ffmcfmb*fmc+fmc)+8*(hbcm*p1p2
     . *p6p7+hbcm*p1p4*p6p7-hbcm*p1p5*p6p7))
      b(12)=w2*ccc*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+8*(p5p7*p3p6)*
     . (3*ffmcfmb*hbcm-fmc)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm+fmc)+8*(p6p7
     . *p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p3p4)*(ffmcfmb*hbcm-fmc)+8*
     . (p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p6p7*p1p3)*(-ffmcfmb*hbcm+
     . fmc+hbcm)+4*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-3*ffmcfmb
     . *fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+8
     . *(p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*fmb+4*ffmcfmb*fmc-fmc)+
     . 8*(-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7+hbcm*p1p5*p6p7))
      b(13)=w5*ccc*(p1p3*(-4*ffmcfmb+4)+2*(3*ffmcfmb*fmb*hbcm-3*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-3*fmb*fmc+3*fmc*hbcm+3*fmc2+2
     . *p1p2-6*p1p4+2*p1p5))
      b(14)=w2*ccc*(p1p3*(4*ffmcfmb-4)+2*(-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmc*hbcm-3*fmc2-2
     . *p1p2+6*p1p4-2*p1p5))
      b(15)=w5*ccc*(4*p3p5*(ffmcfmb*hbcm-fmc)+12*p3p4*(-ffmcfmb*hbcm+
     . fmc)+4*p2p3*(ffmcfmb*hbcm-fmc)+4*p1p3*(4*ffmcfmb*hbcm-3*fmc-
     . hbcm)+2*(-2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+5*ffmcfmb*fmc
     . *hbcm2-ffmcfmb*hbcm3-3*fmb*fmc*hbcm+fmc*hbcm2-3*fmc2*hbcm-2*
     . hbcm*p1p2+6*hbcm*p1p4-2*hbcm*p1p5))
      b(16)=w2*ccc*(4*p3p5*(-ffmcfmb*hbcm+fmc)+12*p3p4*(ffmcfmb*hbcm-
     . fmc)+4*p2p3*(-ffmcfmb*hbcm+fmc)+4*p1p3*(-4*ffmcfmb*hbcm+3*fmc+
     . hbcm)+2*(2*ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-5*ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+3*fmb*fmc*hbcm-fmc*hbcm2+3*fmc2*hbcm+2*
     . hbcm*p1p2-6*hbcm*p1p4+2*hbcm*p1p5))
      b(17)=ccc*(w5*(4*p3p6*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p6-hbcm
     . *p5p6))+w12*(4*p3p6*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p6-hbcm*
     . p5p6)))
      b(18)=ccc*(6*w5*(-ffmcfmb*hbcm-fmb+hbcm)+6*w12*(-ffmcfmb*hbcm-
     . fmb+hbcm))
      b(19)=ccc*(2*w5*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+6*p1p3+2*
     . p2p3-6*p3p4+2*p3p5)+2*w12*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+6
     . *p1p3+2*p2p3-6*p3p4+2*p3p5))
      b(20)=ccc*(w5*(4*p3p7*(-ffmcfmb*hbcm+fmb+hbcm)+8*(-hbcm*p4p7+2*
     . hbcm*p5p7))+w12*(4*p3p7*(-ffmcfmb*hbcm+fmb+hbcm)+8*(-hbcm*p4p7
     . +2*hbcm*p5p7)))
      b(21)=ccc*(4*w2*(-3*hbcm*p1p9-hbcm*p2p9+2*hbcm*p3p9+3*hbcm*p4p9
     . -hbcm*p5p9)+4*w5*(3*ffmcfmb*hbcm*p3p10-3*ffmcfmb*hbcm*p3p8+3*
     . fmb*p3p10-3*fmb*p3p8-3*hbcm*p1p10+3*hbcm*p1p8-hbcm*p2p10+hbcm*
     . p2p8-hbcm*p3p10+hbcm*p3p8+3*hbcm*p4p10-3*hbcm*p4p8-hbcm*p5p10+
     . hbcm*p5p8)+4*w12*(3*ffmcfmb*hbcm*p3p10+3*fmb*p3p10-3*hbcm*
     . p1p10-hbcm*p2p10-hbcm*p3p10+3*hbcm*p4p10-hbcm*p5p10))
      b(22)=ccc*(w5*(4*p3p5*(-ffmcfmb*hbcm+fmc)+12*p3p4*(ffmcfmb*hbcm
     . -fmc)+4*p2p3*(-ffmcfmb*hbcm+fmc)+4*p1p3*(-4*ffmcfmb*hbcm+3*fmc
     . +hbcm)+2*(2*ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2-5*ffmcfmb*fmc
     . *hbcm2+ffmcfmb*hbcm3+3*fmb*fmc*hbcm-fmc*hbcm2+3*fmc2*hbcm+2*
     . hbcm*p1p2-6*hbcm*p1p4+2*hbcm*p1p5))+w12*(4*p3p5*(-ffmcfmb*hbcm
     . +fmc)+12*p3p4*(ffmcfmb*hbcm-fmc)+4*p2p3*(-ffmcfmb*hbcm+fmc)+4*
     . p1p3*(-4*ffmcfmb*hbcm+3*fmc+hbcm)+2*(2*ffmcfmb**2*hbcm3-3*
     . ffmcfmb*fmb*hbcm2-5*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+3*fmb*fmc*
     . hbcm-fmc*hbcm2+3*fmc2*hbcm+2*hbcm*p1p2-6*hbcm*p1p4+2*hbcm*p1p5
     . ))-2*hbcm)
      b(23)=w5*ccc*(4*p3p6*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p4p6+
     . hbcm*p5p6))
      b(24)=6*ccc*w5*(ffmcfmb*hbcm+fmb-hbcm)
      b(25)=2*ccc*w5*(5*ffmcfmb*hbcm2-3*fmb*hbcm+hbcm2-6*p1p3-2*p2p3+
     . 6*p3p4-2*p3p5)
      b(26)=w5*ccc*(4*p3p7*(ffmcfmb*hbcm-fmb-hbcm)+8*(hbcm*p4p7-2*
     . hbcm*p5p7))
      b(27)=ccc*(w5*(4*p5p6*(-ffmcfmb*hbcm+fmc)+12*p4p6*(ffmcfmb*hbcm
     . -fmc)+4*p3p6*(ffmcfmb**2*hbcm-3*ffmcfmb*fmb-4*ffmcfmb*fmc+2*
     . ffmcfmb*hbcm+fmc))+w12*(4*p5p6*(-ffmcfmb*hbcm+fmc)+12*p4p6*(
     . ffmcfmb*hbcm-fmc)+4*p3p6*(ffmcfmb**2*hbcm-3*ffmcfmb*fmb-4*
     . ffmcfmb*fmc+2*ffmcfmb*hbcm+fmc)))
      b(28)=ccc*(w5*((p3p6*p1p3)*(-8*ffmcfmb+8)+4*p5p6*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+12*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p6*(-
     . ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm-2*
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*p2p3*p3p6-3*ffmcfmb*p3p4*
     . p3p6+ffmcfmb*p3p5*p3p6-3*p1p3*p4p6+p1p3*p5p6))+w12*((p3p6*p1p3
     . )*(-8*ffmcfmb+8)+4*p5p6*(-ffmcfmb*hbcm2+fmc*hbcm)+12*p4p6*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p3p6*(-ffmcfmb**2*hbcm2+3*ffmcfmb*
     . fmb*hbcm-4*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . ffmcfmb*p2p3*p3p6-3*ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6-3*p1p3
     . *p4p6+p1p3*p5p6)))
      ans=ccc*(w5*(8*(p6p7*p1p3)*(fmb+fmc)+16*(p4p6*p3p7)*(-ffmcfmb*
     . hbcm+fmc)+8*(p5p7*p3p6)*(3*ffmcfmb*hbcm-fmc)+8*(p4p7*p3p6)*(-
     . ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*
     . p3p4)*(ffmcfmb*hbcm-fmc)+8*(p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+4*
     . p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+8*(p3p7*p3p6
     . )*(-ffmcfmb**2*hbcm+ffmcfmb*fmb+4*ffmcfmb*fmc-fmc)+8*(-hbcm*
     . p1p2*p6p7-hbcm*p1p4*p6p7+hbcm*p1p5*p6p7))+w12*(8*(p6p7*p1p3)*(
     . fmb+fmc)+16*(p4p6*p3p7)*(-ffmcfmb*hbcm+fmc)+8*(p5p7*p3p6)*(3*
     . ffmcfmb*hbcm-fmc)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*
     . p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p3p4)*(ffmcfmb*hbcm-fmc)+8*(
     . p6p7*p2p3)*(ffmcfmb*hbcm-fmc)+4*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*
     . hbcm2+2*fmc2*hbcm)+8*(p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*fmb
     . +4*ffmcfmb*fmc-fmc)+8*(-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7+hbcm*
     . p1p5*p6p7)))
      b(29)=ans
      ans4=8*p6p7*(-ffmcfmb*hbcm2*p1p10+ffmcfmb*hbcm2*p1p8+ffmcfmb*
     . hbcm2*p2p10-ffmcfmb*hbcm2*p2p8+ffmcfmb*hbcm2*p4p10-ffmcfmb*
     . hbcm2*p4p8-ffmcfmb*hbcm2*p5p10+ffmcfmb*hbcm2*p5p8+fmc*hbcm*
     . p1p10-fmc*hbcm*p1p8-fmc*hbcm*p2p10+fmc*hbcm*p2p8-fmc*hbcm*
     . p4p10+fmc*hbcm*p4p8+fmc*hbcm*p5p10-fmc*hbcm*p5p8+2*hbcm2*p1p10
     . -2*hbcm2*p1p8)+8*p3p7*(-2*ffmcfmb**2*hbcm2*p6p10+2*ffmcfmb**2*
     . hbcm2*p6p8+2*ffmcfmb*fmc*hbcm*p6p10-2*ffmcfmb*fmc*hbcm*p6p8+
     . ffmcfmb*hbcm2*p6p10-ffmcfmb*hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm
     . *p6p8)+8*p3p6*(ffmcfmb**2*hbcm2*p7p10-ffmcfmb*fmb*hbcm*p7p10-2
     . *ffmcfmb*fmb*hbcm*p7p8+2*ffmcfmb*fmc*hbcm*p7p10+2*ffmcfmb*fmc*
     . hbcm*p7p8+ffmcfmb*hbcm2*p7p10+ffmcfmb*hbcm2*p7p8-fmc*hbcm*p7p8
     . )+16*(-ffmcfmb*p2p3*p3p6*p7p8-ffmcfmb*p3p5*p3p6*p7p8-p1p3*p5p6
     . *p7p8)
      ans3=16*(p5p7*p1p3)*(-p6p10+p6p8)+16*(p4p7*p1p3)*(p6p10-p6p8)+
     . 16*(p4p6*p1p3)*(2*p7p10+p7p8)+16*(p6p7*p1p3)*(-p2p10+p2p8-
     . p4p10+p4p8+p5p10-p5p8)+16*(p5p6*p3p7)*(-p1p10+p1p8)+48*(p4p6*
     . p3p7)*(p1p10-p1p8)+16*(p6p7*p3p5)*(p1p10-p1p8)+16*(p6p7*p3p4)*
     . (p1p10-p1p8)+16*(p6p7*p2p3)*(-p1p10+p1p8)+16*(p3p6*p3p4)*(2*
     . ffmcfmb*p7p10+ffmcfmb*p7p8)+16*(p3p7*p1p3)*(2*ffmcfmb*p6p10-2*
     . ffmcfmb*p6p8-p6p10+p6p8)+16*(p3p6*p1p3)*(ffmcfmb*p7p8-p7p8)+16
     . *(p3p7*p3p6)*(2*ffmcfmb*p1p10-2*ffmcfmb*p1p8-2*ffmcfmb*p2p10+2
     . *ffmcfmb*p2p8+2*ffmcfmb*p3p10-2*ffmcfmb*p3p8+2*ffmcfmb*p4p10-2
     . *ffmcfmb*p4p8-p1p10+p1p8)+8*p5p7*(ffmcfmb*hbcm2*p6p10-ffmcfmb*
     . hbcm2*p6p8-fmc*hbcm*p6p10+fmc*hbcm*p6p8)+8*p5p6*(ffmcfmb*hbcm2
     . *p7p8-fmc*hbcm*p7p8)+8*p4p7*(-ffmcfmb*hbcm2*p6p10+ffmcfmb*
     . hbcm2*p6p8+fmc*hbcm*p6p10-fmc*hbcm*p6p8)+8*p4p6*(-2*ffmcfmb*
     . hbcm2*p7p10-ffmcfmb*hbcm2*p7p8+2*fmc*hbcm*p7p10+fmc*hbcm*p7p8)
     . +ans4
      ans2=w5*ans3
      ans6=(8*p6p7*(-p3p10+2*p3p11+p3p8)+16*p1p3*(p6p12+p7p13)+8*p3p7
     . *(2*ffmcfmb*p3p13+4*p1p13-p3p13-2*p5p13)+8*p3p6*(-ffmcfmb*
     . p3p12-2*p1p12)+4*(-2*ffmcfmb*hbcm2*p6p12+ffmcfmb*hbcm2*p7p13-
     . fmb*hbcm*p7p13+2*fmc*hbcm*p6p12+2*fmc*hbcm*p7p13-3*hbcm2*p7p13
     . +4*p2p3*p7p13-2*p3p12*p4p6+8*p3p13*p4p7-4*p3p13*p5p7-4*p3p5*
     . p7p13))
      ans5=w12*(16*(p6p7*p1p3)*(-p2p10-p4p10+p5p10)+16*(p3p7*p1p3)*(2
     . *ffmcfmb*p6p10-p6p10)+16*(p3p7*p3p6)*(2*ffmcfmb*p1p10-2*
     . ffmcfmb*p2p10+2*ffmcfmb*p3p10+2*ffmcfmb*p4p10-p1p10)+8*p5p7*(
     . ffmcfmb*hbcm2*p6p10-fmc*hbcm*p6p10)+8*p4p7*(-ffmcfmb*hbcm2*
     . p6p10+fmc*hbcm*p6p10)+16*p4p6*(-ffmcfmb*hbcm2*p7p10+fmc*hbcm*
     . p7p10)+8*p6p7*(-ffmcfmb*hbcm2*p1p10+ffmcfmb*hbcm2*p2p10+
     . ffmcfmb*hbcm2*p4p10-ffmcfmb*hbcm2*p5p10+fmc*hbcm*p1p10-fmc*
     . hbcm*p2p10-fmc*hbcm*p4p10+fmc*hbcm*p5p10+2*hbcm2*p1p10)+8*p3p7
     . *(-2*ffmcfmb**2*hbcm2*p6p10+2*ffmcfmb*fmc*hbcm*p6p10+ffmcfmb*
     . hbcm2*p6p10-fmc*hbcm*p6p10)+8*p3p6*(ffmcfmb**2*hbcm2*p7p10-
     . ffmcfmb*fmb*hbcm*p7p10+2*ffmcfmb*fmc*hbcm*p7p10+ffmcfmb*hbcm2*
     . p7p10)+16*(2*ffmcfmb*p3p4*p3p6*p7p10-p1p10*p2p3*p6p7+p1p10*
     . p3p4*p6p7+p1p10*p3p5*p6p7+3*p1p10*p3p7*p4p6-p1p10*p3p7*p5p6+2*
     . p1p3*p4p6*p7p10+p1p3*p4p7*p6p10-p1p3*p5p7*p6p10))+ans6
      ans1=w2*(16*(p6p7*p1p3)*(-p2p9-p4p9+p5p9)+16*(p3p7*p3p6)*(2*
     . ffmcfmb*p1p9-2*ffmcfmb*p2p9+2*ffmcfmb*p3p9+2*ffmcfmb*p4p9-p1p9
     . )+16*(p3p7*p1p3)*(2*ffmcfmb*p6p9-p6p9)+16*(p3p6*p1p3)*(-
     . ffmcfmb*p7p9+p7p9)+8*p5p7*(ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+8
     . *p5p6*(-ffmcfmb*hbcm2*p7p9+fmc*hbcm*p7p9)+8*p4p7*(-ffmcfmb*
     . hbcm2*p6p9+fmc*hbcm*p6p9)+8*p4p6*(ffmcfmb*hbcm2*p7p9-fmc*hbcm*
     . p7p9)+8*p6p7*(-ffmcfmb*hbcm2*p1p9+ffmcfmb*hbcm2*p2p9+ffmcfmb*
     . hbcm2*p4p9-ffmcfmb*hbcm2*p5p9+fmc*hbcm*p1p9-fmc*hbcm*p2p9-fmc*
     . hbcm*p4p9+fmc*hbcm*p5p9+2*hbcm2*p1p9)+8*p3p6*(2*ffmcfmb*fmb*
     . hbcm*p7p9-2*ffmcfmb*fmc*hbcm*p7p9-ffmcfmb*hbcm2*p7p9+fmc*hbcm*
     . p7p9)+8*p3p7*(-2*ffmcfmb**2*hbcm2*p6p9+2*ffmcfmb*fmc*hbcm*p6p9
     . +ffmcfmb*hbcm2*p6p9-fmc*hbcm*p6p9)+16*(ffmcfmb*p2p3*p3p6*p7p9-
     . ffmcfmb*p3p4*p3p6*p7p9+ffmcfmb*p3p5*p3p6*p7p9-p1p3*p4p6*p7p9+
     . p1p3*p4p7*p6p9+p1p3*p5p6*p7p9-p1p3*p5p7*p6p9-p1p9*p2p3*p6p7+
     . p1p9*p3p4*p6p7+p1p9*p3p5*p6p7+3*p1p9*p3p7*p4p6-p1p9*p3p7*p5p6)
     . )+ans2+ans5
      ans=ccc*ans1
      b(30)=ans
      b(31)=w5*ccc*(4*p5p6*(ffmcfmb*hbcm-fmc)+12*p4p6*(-ffmcfmb*hbcm+
     . fmc)+4*p3p6*(-ffmcfmb**2*hbcm+3*ffmcfmb*fmb+4*ffmcfmb*fmc-2*
     . ffmcfmb*hbcm-fmc))
      b(32)=w2*ccc*(4*p5p6*(ffmcfmb*hbcm-fmc)+12*p4p6*(-ffmcfmb*hbcm+
     . fmc)+4*p3p6*(-ffmcfmb**2*hbcm+3*ffmcfmb*fmb+4*ffmcfmb*fmc-2*
     . ffmcfmb*hbcm-fmc))
      ans3=w12*(8*p5p7*(-ffmcfmb*hbcm*p6p10+fmc*p6p10)+8*p4p7*(
     . ffmcfmb*hbcm*p6p10-fmc*p6p10)+16*p4p6*(ffmcfmb*hbcm*p7p10-fmc*
     . p7p10)+8*p6p7*(-ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm*p4p10+ffmcfmb*
     . hbcm*p5p10-fmb*p1p10-fmc*p1p10+fmc*p2p10+fmc*p4p10-fmc*p5p10+
     . hbcm*p1p10)+8*p3p7*(2*ffmcfmb**2*hbcm*p6p10-2*ffmcfmb*fmc*
     . p6p10-ffmcfmb*hbcm*p6p10+fmc*p6p10)+8*p3p6*(ffmcfmb**2*hbcm*
     . p7p10-ffmcfmb*fmb*p7p10-2*ffmcfmb*fmc*p7p10+ffmcfmb*hbcm*p7p10
     . ))+4*(2*ffmcfmb*hbcm*p6p12+ffmcfmb*hbcm*p7p13-fmb*p7p13-2*fmc*
     . p6p12-2*fmc*p7p13+hbcm*p7p13)
      ans2=w5*(8*p5p7*(-ffmcfmb*hbcm*p6p10+ffmcfmb*hbcm*p6p8+fmc*
     . p6p10-fmc*p6p8)+8*p4p7*(ffmcfmb*hbcm*p6p10-ffmcfmb*hbcm*p6p8-
     . fmc*p6p10+fmc*p6p8)+16*p4p6*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*
     . p7p8-fmc*p7p10+fmc*p7p8)+8*p6p7*(-ffmcfmb*hbcm*p2p10+ffmcfmb*
     . hbcm*p2p8-ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*
     . p5p10-ffmcfmb*hbcm*p5p8-fmb*p1p10+fmb*p1p8-fmc*p1p10+fmc*p1p8+
     . fmc*p2p10-fmc*p2p8+fmc*p4p10-fmc*p4p8-fmc*p5p10+fmc*p5p8+hbcm*
     . p1p10-hbcm*p1p8)+8*p3p7*(2*ffmcfmb**2*hbcm*p6p10-2*ffmcfmb**2*
     . hbcm*p6p8-2*ffmcfmb*fmc*p6p10+2*ffmcfmb*fmc*p6p8-ffmcfmb*hbcm*
     . p6p10+ffmcfmb*hbcm*p6p8+fmc*p6p10-fmc*p6p8)+8*p3p6*(ffmcfmb**2
     . *hbcm*p7p10-ffmcfmb**2*hbcm*p7p8-ffmcfmb*fmb*p7p10+ffmcfmb*fmb
     . *p7p8-2*ffmcfmb*fmc*p7p10+2*ffmcfmb*fmc*p7p8+ffmcfmb*hbcm*
     . p7p10-ffmcfmb*hbcm*p7p8))+ans3
      ans1=w2*(8*p5p7*(-ffmcfmb*hbcm*p6p9+fmc*p6p9)+8*p4p7*(ffmcfmb*
     . hbcm*p6p9-fmc*p6p9)+16*p4p6*(ffmcfmb*hbcm*p7p9-fmc*p7p9)+8*
     . p6p7*(ffmcfmb*hbcm*p1p9-ffmcfmb*hbcm*p2p9-ffmcfmb*hbcm*p4p9+
     . ffmcfmb*hbcm*p5p9-fmc*p1p9+fmc*p2p9+fmc*p4p9-fmc*p5p9)+8*p3p7*
     . (2*ffmcfmb**2*hbcm*p6p9-2*ffmcfmb*fmc*p6p9-ffmcfmb*hbcm*p6p9+
     . fmc*p6p9)+8*p3p6*(ffmcfmb**2*hbcm*p7p9-ffmcfmb*fmb*p7p9-2*
     . ffmcfmb*fmc*p7p9+ffmcfmb*hbcm*p7p9))+ans2
      ans=ccc*ans1
      b(33)=ans
      b(34)=6*ccc*w2*(ffmcfmb*hbcm+fmb-hbcm)
      b(35)=ccc*(4*w2*(-ffmcfmb*hbcm*p7p9-fmb*p7p9+hbcm*p7p9)+4*w5*(-
     . ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8-fmb*p7p10+fmb*p7p8+hbcm*
     . p7p10-hbcm*p7p8)+4*w12*(-ffmcfmb*hbcm*p7p10-fmb*p7p10+hbcm*
     . p7p10))
      b(36)=2*ccc*w2*(-5*ffmcfmb*hbcm2+3*fmb*hbcm-hbcm2+6*p1p3+2*p2p3
     . -6*p3p4+2*p3p5)
      b(37)=w2*ccc*(4*p3p7*(-ffmcfmb*hbcm+fmb+hbcm)+8*(-hbcm*p4p7+2*
     . hbcm*p5p7))
      b(38)=ccc*(w2*(8*p3p7*(p1p9-p2p9-p4p9+p5p9)+4*(3*ffmcfmb*hbcm2*
     . p7p9-fmb*hbcm*p7p9+hbcm2*p7p9-4*p1p3*p7p9+4*p3p4*p7p9))+w5*(16
     . *p3p4*(p7p10-p7p8)+16*p1p3*(-p7p10+p7p8)+8*p3p7*(p1p10-p1p8-
     . p2p10+p2p8-p4p10+p4p8+p5p10-p5p8)+4*(3*ffmcfmb*hbcm2*p7p10-3*
     . ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p10+fmb*hbcm*p7p8+hbcm2*p7p10-
     . hbcm2*p7p8))+w12*(8*p3p7*(p1p10-p2p10-p4p10+p5p10)+4*(3*
     . ffmcfmb*hbcm2*p7p10-fmb*hbcm*p7p10+hbcm2*p7p10-4*p1p3*p7p10+4*
     . p3p4*p7p10)))
      b(42)=w2*ccc*((p3p7*p3p6)*(-32*ffmcfmb+8)+4*p6p7*(3*ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+8*(-p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-p3p5*
     . p6p7-p3p6*p4p7+p3p6*p5p7-2*p3p7*p4p6))
      b(43)=ccc*(w2*(8*p6p7*(-p1p9+p2p9+p4p9-p5p9)+8*p3p7*(-2*ffmcfmb
     . *p6p9+p6p9)+8*(-2*ffmcfmb*p3p6*p7p9-2*p4p6*p7p9-p4p7*p6p9+p5p7
     . *p6p9))+w5*(8*p5p7*(p6p10-p6p8)+8*p4p7*(-p6p10+p6p8)+16*p4p6*(
     . -p7p10+p7p8)+8*p6p7*(-p1p10+p1p8+p2p10-p2p8+p4p10-p4p8-p5p10+
     . p5p8)+8*p3p7*(-2*ffmcfmb*p6p10+2*ffmcfmb*p6p8+p6p10-p6p8)+16*
     . p3p6*(-ffmcfmb*p7p10+ffmcfmb*p7p8))+w12*(8*p6p7*(-p1p10+p2p10+
     . p4p10-p5p10)+8*p3p7*(-2*ffmcfmb*p6p10+p6p10)+8*(-2*ffmcfmb*
     . p3p6*p7p10-2*p4p6*p7p10-p4p7*p6p10+p5p7*p6p10))+8*(-p6p12-
     . p7p13))
      b(47)=w2*ccc*(p3p6*(-16*ffmcfmb+4)+4*(-3*p4p6+p5p6))
      b(52)=w2*ccc*(p3p7*(-16*ffmcfmb+12)+4*(-3*p4p7+5*p5p7))
      b(53)=ccc*(4*w2*(-3*p1p9-p2p9+2*p3p9+3*p4p9-p5p9)+4*w5*(-3*
     . p1p10+3*p1p8-p2p10+p2p8+2*p3p10-2*p3p8+3*p4p10-3*p4p8-p5p10+
     . p5p8)+4*w12*(-3*p1p10-p2p10+2*p3p10+3*p4p10-p5p10))
      b(57)=-8*ccc
      b(60)=w2*ccc*(4*p3p6*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*p4p6-hbcm*
     . p5p6))
      b(61)=-4*ccc*hbcm
      b(62)=8*ccc*ffmcfmb*hbcm*p3p6
      b(63)=ccc*(w5*((p3p7*p3p6)*(-32*ffmcfmb+8)+4*p6p7*(3*ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+8*(-p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-p3p5*
     . p6p7-p3p6*p4p7+p3p6*p5p7-2*p3p7*p4p6))+w12*((p3p7*p3p6)*(-32*
     . ffmcfmb+8)+4*p6p7*(3*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(-p1p3*
     . p6p7+p2p3*p6p7+p3p4*p6p7-p3p5*p6p7-p3p6*p4p7+p3p6*p5p7-2*p3p7*
     . p4p6)))
      b(64)=ccc*(w5*(4*p6p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+fmc*hbcm+2*fmc2)+8*(
     . ffmcfmb*p3p6*p3p7-ffmcfmb*p3p6*p4p7+3*ffmcfmb*p3p6*p5p7-p1p2*
     . p6p7+p1p3*p6p7-p1p4*p6p7+p1p5*p6p7))+w12*(4*p6p7*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+
     . fmc*hbcm+2*fmc2)+8*(ffmcfmb*p3p6*p3p7-ffmcfmb*p3p6*p4p7+3*
     . ffmcfmb*p3p6*p5p7-p1p2*p6p7+p1p3*p6p7-p1p4*p6p7+p1p5*p6p7)))
      b(65)=w5*ccc*((p3p7*p3p6)*(32*ffmcfmb-8)+4*p6p7*(-3*ffmcfmb*
     . hbcm2+fmb*hbcm+hbcm2)+8*(p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+p3p5*
     . p6p7+p3p6*p4p7-p3p6*p5p7+2*p3p7*p4p6))
      ans=ccc*(w2*(8*p6p7*(-hbcm*p1p9+hbcm*p2p9+hbcm*p4p9-hbcm*p5p9)+
     . 8*p3p7*(-2*ffmcfmb*hbcm*p6p9+hbcm*p6p9)+8*(-2*ffmcfmb*hbcm*
     . p3p6*p7p9-2*hbcm*p4p6*p7p9-hbcm*p4p7*p6p9+hbcm*p5p7*p6p9))+w5*
     . (8*p5p7*(hbcm*p6p10-hbcm*p6p8)+8*p4p7*(-hbcm*p6p10+hbcm*p6p8)+
     . 16*p4p6*(-hbcm*p7p10+hbcm*p7p8)+8*p6p7*(-hbcm*p1p10+hbcm*p1p8+
     . hbcm*p2p10-hbcm*p2p8+hbcm*p4p10-hbcm*p4p8-hbcm*p5p10+hbcm*p5p8
     . )+8*p3p7*(-2*ffmcfmb*hbcm*p6p10+2*ffmcfmb*hbcm*p6p8+hbcm*p6p10
     . -hbcm*p6p8)+16*p3p6*(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8))+
     . w12*(8*p6p7*(-hbcm*p1p10+hbcm*p2p10+hbcm*p4p10-hbcm*p5p10)+8*
     . p3p7*(-2*ffmcfmb*hbcm*p6p10+hbcm*p6p10)+8*(-2*ffmcfmb*hbcm*
     . p3p6*p7p10-2*hbcm*p4p6*p7p10-hbcm*p4p7*p6p10+hbcm*p5p7*p6p10))
     . +8*(-hbcm*p6p12-hbcm*p7p13))
      b(66)=ans
      b(67)=ccc*(w5*(p3p6*(-16*ffmcfmb+4)+4*(-3*p4p6+p5p6))+w12*(p3p6
     . *(-16*ffmcfmb+4)+4*(-3*p4p6+p5p6)))
      b(68)=w5*ccc*(p3p6*(16*ffmcfmb-4)+4*(3*p4p6-p5p6))
      b(69)=ccc*(w5*(p3p7*(-16*ffmcfmb+12)+4*(-3*p4p7+5*p5p7))+w12*(
     . p3p7*(-16*ffmcfmb+12)+4*(-3*p4p7+5*p5p7)))
      b(70)=ccc*(w5*((p3p7*p1p3)*(16*ffmcfmb-8)+12*p5p7*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+4*p4p7*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p7*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-2*
     . fmc2)+8*(p1p3*p4p7-p1p3*p5p7+2*p1p4*p3p7))+w12*((p3p7*p1p3)*(
     . 16*ffmcfmb-8)+12*p5p7*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p4p7*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p3p7*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-2*fmc2)+8*(p1p3*p4p7-p1p3*p5p7+2
     . *p1p4*p3p7)))
      b(71)=w5*ccc*(p3p7*(16*ffmcfmb-12)+4*(3*p4p7-5*p5p7))
      b(72)=ccc*(w5*(p1p3*(4*ffmcfmb-4)+2*(-3*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmc*hbcm-3*fmc2-2
     . *p1p2+6*p1p4-2*p1p5))+w12*(p1p3*(4*ffmcfmb-4)+2*(-3*ffmcfmb*
     . fmb*hbcm+3*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmc*
     . hbcm-3*fmc2-2*p1p2+6*p1p4-2*p1p5))+2)
      b(74)=4*ccc*w5*p6p7*(-ffmcfmb*hbcm-fmb+hbcm)
      b(75)=ccc*(4*w2*(3*ffmcfmb*hbcm*p1p9+ffmcfmb*hbcm*p2p9-2*
     . ffmcfmb*hbcm*p3p9-3*ffmcfmb*hbcm*p4p9+ffmcfmb*hbcm*p5p9-3*fmc*
     . p1p9-fmc*p2p9+2*fmc*p3p9+3*fmc*p4p9-fmc*p5p9)+4*w5*(3*ffmcfmb*
     . hbcm*p1p10-3*ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p10-ffmcfmb*hbcm
     . *p2p8-2*ffmcfmb*hbcm*p3p10+2*ffmcfmb*hbcm*p3p8-3*ffmcfmb*hbcm*
     . p4p10+3*ffmcfmb*hbcm*p4p8+ffmcfmb*hbcm*p5p10-ffmcfmb*hbcm*p5p8
     . -3*fmc*p1p10+3*fmc*p1p8-fmc*p2p10+fmc*p2p8+2*fmc*p3p10-2*fmc*
     . p3p8+3*fmc*p4p10-3*fmc*p4p8-fmc*p5p10+fmc*p5p8)+4*w12*(3*
     . ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p2p10-2*ffmcfmb*hbcm*p3p10-3*
     . ffmcfmb*hbcm*p4p10+ffmcfmb*hbcm*p5p10-3*fmc*p1p10-fmc*p2p10+2*
     . fmc*p3p10+3*fmc*p4p10-fmc*p5p10))
      b(76)=ccc*(4*w2*(3*ffmcfmb*hbcm2*p1p9+ffmcfmb*hbcm2*p2p9-2*
     . ffmcfmb*hbcm2*p3p9-3*ffmcfmb*hbcm2*p4p9+ffmcfmb*hbcm2*p5p9-3*
     . fmc*hbcm*p1p9-fmc*hbcm*p2p9+2*fmc*hbcm*p3p9+3*fmc*hbcm*p4p9-
     . fmc*hbcm*p5p9)+4*w5*(3*ffmcfmb*hbcm2*p1p10-3*ffmcfmb*hbcm2*
     . p1p8+ffmcfmb*hbcm2*p2p10-ffmcfmb*hbcm2*p2p8-2*ffmcfmb*hbcm2*
     . p3p10+2*ffmcfmb*hbcm2*p3p8-3*ffmcfmb*hbcm2*p4p10+3*ffmcfmb*
     . hbcm2*p4p8+ffmcfmb*hbcm2*p5p10-ffmcfmb*hbcm2*p5p8-3*fmc*hbcm*
     . p1p10+3*fmc*hbcm*p1p8-fmc*hbcm*p2p10+fmc*hbcm*p2p8+2*fmc*hbcm*
     . p3p10-2*fmc*hbcm*p3p8+3*fmc*hbcm*p4p10-3*fmc*hbcm*p4p8-fmc*
     . hbcm*p5p10+fmc*hbcm*p5p8)+4*w12*(3*ffmcfmb*hbcm2*p1p10+ffmcfmb
     . *hbcm2*p2p10-2*ffmcfmb*hbcm2*p3p10-3*ffmcfmb*hbcm2*p4p10+
     . ffmcfmb*hbcm2*p5p10-3*fmc*hbcm*p1p10-fmc*hbcm*p2p10+2*fmc*hbcm
     . *p3p10+3*fmc*hbcm*p4p10-fmc*hbcm*p5p10))
      b(77)=-8*ccc*p3p13
      b(79)=4*ccc*w2*p6p7*(ffmcfmb*hbcm+fmb-hbcm)
      b(80)=w2*ccc*(16*p5p7*(-ffmcfmb*hbcm+fmc)+8*p4p7*(ffmcfmb*hbcm-
     . fmc)+8*p3p7*(ffmcfmb**2*hbcm-ffmcfmb*fmc-ffmcfmb*hbcm+fmc))
      b(81)=ccc*(4*w5*p6p7*(ffmcfmb*hbcm+fmb-hbcm)+4*w12*p6p7*(
     . ffmcfmb*hbcm+fmb-hbcm))
      b(82)=-8*ccc*p3p7
      b(83)=ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-3*hbcm)-8*hbcm*p4p7)
      b(84)=4*ccc*(-p3p7-3*p4p7+p5p7)
      b(85)=2*ccc*(-ffmcfmb*hbcm2+3*fmb*hbcm-4*fmc*hbcm+3*hbcm2-6*
     . p1p3-2*p2p3-2*p3p4+6*p3p5)
      b(86)=-8*ccc
      b(87)=2*ccc*(-ffmcfmb*hbcm+3*fmb+4*fmc-3*hbcm)
      b(88)=ccc*(w5*(16*p5p7*(-ffmcfmb*hbcm+fmc)+8*p4p7*(ffmcfmb*hbcm
     . -fmc)+8*p3p7*(ffmcfmb**2*hbcm-ffmcfmb*fmc-ffmcfmb*hbcm+fmc))+
     . w12*(16*p5p7*(-ffmcfmb*hbcm+fmc)+8*p4p7*(ffmcfmb*hbcm-fmc)+8*
     . p3p7*(ffmcfmb**2*hbcm-ffmcfmb*fmc-ffmcfmb*hbcm+fmc)))
      b(89)=-8*ccc*hbcm
      b(91)=w5*ccc*(16*p5p7*(ffmcfmb*hbcm-fmc)+8*p4p7*(-ffmcfmb*hbcm+
     . fmc)+8*p3p7*(-ffmcfmb**2*hbcm+ffmcfmb*fmc+ffmcfmb*hbcm-fmc))
      b(92)=4*ccc*(hbcm2+p1p3-p2p3+p3p4+p3p5)
      b(93)=8*ccc*p3p6
      b(94)=4*ccc*(-ffmcfmb*hbcm+fmc)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp31_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5))
      b(1)=8*ccc*w1*(3*p1p5*p6p7+p3p7*p5p6-p4p7*p5p6+3*p5p6*p5p7)
      b(2)=8*ccc*w15*(-3*p1p5*p6p7-p3p7*p5p6+p4p7*p5p6-3*p5p6*p5p7)
      b(3)=ccc*(8*w1*(p1p2*p7p8-p1p3*p7p8-p1p4*p7p8-p1p8*p3p7-3*p1p8*
     . p4p7+p1p8*p5p7)+w15*(8*p1p4*(-p7p10+p7p9)+8*p1p3*(-p7p10+p7p9)
     . +8*p1p2*(p7p10-p7p9)+8*p5p7*(p1p10-p1p9)+24*p4p7*(-p1p10+p1p9)
     . +8*p3p7*(-p1p10+p1p9))+4*(p1p12-p7p10))
      b(4)=8*ccc*w1*(-p1p2*p3p7+p1p3*p4p7-3*p1p3*p5p7+p1p4*p3p7)
      b(5)=8*ccc*w15*(p1p2*p3p7-p1p3*p4p7+3*p1p3*p5p7-p1p4*p3p7)
      b(6)=ccc*(w1*(8*p3p7*(-ffmcfmb*hbcm*p1p8-fmb*p1p8-2*fmc*p1p8+
     . hbcm*p1p8)+8*p1p3*(-ffmcfmb*hbcm*p7p8+fmb*p7p8+2*fmc*p7p8)+8*(
     . -hbcm*p1p2*p7p8+hbcm*p1p4*p7p8-2*hbcm*p1p8*p5p7))+w15*(8*p1p4*
     . (hbcm*p7p10-hbcm*p7p9)+8*p1p2*(-hbcm*p7p10+hbcm*p7p9)+16*p5p7*
     . (-hbcm*p1p10+hbcm*p1p9)+8*p1p3*(-ffmcfmb*hbcm*p7p10+ffmcfmb*
     . hbcm*p7p9+fmb*p7p10-fmb*p7p9+2*fmc*p7p10-2*fmc*p7p9)+8*p3p7*(-
     . ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p1p9-fmb*p1p10+fmb*p1p9-2*fmc*
     . p1p10+2*fmc*p1p9+hbcm*p1p10-hbcm*p1p9))+4*(hbcm*p1p12+hbcm*
     . p7p10))
      b(7)=ccc*(w1*(8*p5p6*(-p1p8+p2p8-2*p3p8-3*p4p8+p5p8)+24*p1p5*
     . p6p8)+w15*(24*p1p5*(p6p10-p6p9)+8*p5p6*(-p1p10+p1p9+p2p10-p2p9
     . -2*p3p10+2*p3p9-3*p4p10+3*p4p9+p5p10-p5p9)))
      b(8)=w1*ccc*(4*p5p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+
     . hbcm2)+8*(p1p3*p5p6-3*p1p5*p3p6-p2p3*p5p6+3*p3p4*p5p6-p3p5*
     . p5p6))
      b(9)=w15*ccc*(4*p5p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-
     . hbcm2)+8*(-p1p3*p5p6+3*p1p5*p3p6+p2p3*p5p6-3*p3p4*p5p6+p3p5*
     . p5p6))
      b(10)=ccc*(w1*(8*p5p6*(3*fmb*p3p8+3*fmc*p3p8-hbcm*p1p8-hbcm*
     . p2p8-hbcm*p3p8+3*hbcm*p4p8-hbcm*p5p8)+8*p3p6*(-3*fmb*p1p8-3*
     . fmc*p1p8+hbcm*p1p8)+8*(-3*hbcm*p1p5*p6p8-2*hbcm*p1p8*p4p6))+
     . w15*(24*p1p5*(-hbcm*p6p10+hbcm*p6p9)+16*p4p6*(-hbcm*p1p10+hbcm
     . *p1p9)+8*p5p6*(3*fmb*p3p10+3*fmc*p3p10-hbcm*p1p10+hbcm*p1p9-
     . hbcm*p2p10+hbcm*p2p9-hbcm*p3p10-2*hbcm*p3p9+3*hbcm*p4p10-3*
     . hbcm*p4p9-hbcm*p5p10+hbcm*p5p9)+8*p3p6*(-3*fmb*p1p10+3*fmb*
     . p1p9-3*fmc*p1p10+3*fmc*p1p9+hbcm*p1p10-hbcm*p1p9)))
      b(11)=w1*ccc*(8*(p5p6*p3p7)*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*(3*
     . hbcm*p1p5*p6p7+2*hbcm*p5p6*p5p7))
      b(12)=w15*ccc*(24*(p6p7*p1p3)*(fmb+fmc-hbcm)+8*(p5p6*p3p7)*(-
     . ffmcfmb*hbcm-fmb-2*fmc+hbcm)+8*(-3*hbcm*p1p5*p6p7-2*hbcm*p5p6*
     . p5p7))
      b(13)=4*ccc*w1*(-2*p1p2+2*p1p3+2*p1p4-p1p5)
      b(14)=4*ccc*w15*(2*p1p2-2*p1p3-2*p1p4+p1p5)
      b(15)=w1*ccc*(4*p1p3*(-3*fmb-3*fmc+hbcm)+4*(2*hbcm*p1p2-2*hbcm*
     . p1p4+hbcm*p1p5))
      b(16)=w15*ccc*(4*p1p3*(3*fmb+3*fmc-hbcm)+4*(-2*hbcm*p1p2+2*hbcm
     . *p1p4-hbcm*p1p5))
      b(17)=w15*ccc*(4*p3p6*(3*fmb+3*fmc-hbcm)+8*(hbcm*p4p6+hbcm*p5p6
     . ))
      b(18)=6*ccc*w15*(-fmb-fmc+hbcm)
      b(19)=2*ccc*w15*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2-2
     . *p1p3+2*p2p3-6*p3p4+2*p3p5)
      b(20)=w15*ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*hbcm*p5p7
     . )
      b(21)=ccc*(4*w1*(3*fmb*p3p8+3*fmc*p3p8+hbcm*p1p8-hbcm*p2p8-hbcm
     . *p3p8+3*hbcm*p4p8-hbcm*p5p8)+4*w15*(3*fmb*p3p10+3*fmc*p3p10+
     . hbcm*p1p10-hbcm*p1p9-hbcm*p2p10+hbcm*p2p9-hbcm*p3p10-2*hbcm*
     . p3p9+3*hbcm*p4p10-3*hbcm*p4p9-hbcm*p5p10+hbcm*p5p9))
      b(22)=w15*ccc*(4*p1p3*(-3*fmb-3*fmc+hbcm)+4*(2*hbcm*p1p2-2*hbcm
     . *p1p4+hbcm*p1p5))
      b(23)=w1*ccc*(4*p3p6*(3*fmb+3*fmc-hbcm)+8*(hbcm*p4p6+hbcm*p5p6)
     . )
      b(24)=6*ccc*w1*(-fmb-fmc+hbcm)
      b(25)=2*ccc*w1*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2-2*
     . p1p3+2*p2p3-6*p3p4+2*p3p5)
      b(26)=w1*ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*hbcm*p5p7)
      b(27)=12*ccc*w15*p5p6*(-fmb-fmc+hbcm)
      b(28)=w15*ccc*(4*p5p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-
     . hbcm2)+8*(-p1p3*p5p6+3*p1p5*p3p6+p2p3*p5p6-3*p3p4*p5p6+p3p5*
     . p5p6))
      b(29)=w15*ccc*(8*(p5p6*p3p7)*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*(3
     . *hbcm*p1p5*p6p7+2*hbcm*p5p6*p5p7))
      ans2=w15*(16*(p5p6*p3p4)*(2*p7p10+p7p9)+32*(p3p7*p1p5)*(-p6p10+
     . p6p9)+16*(p3p6*p1p5)*(-p7p10-2*p7p9)+16*(p5p6*p3p7)*(p1p10-
     . p1p9-2*p2p10+2*p2p9+2*p3p10-2*p3p9+2*p4p10-2*p4p9)+16*(p4p6*
     . p3p7)*(p1p10-p1p9)+16*(p5p7*p3p6)*(-p1p10+p1p9)+48*(p4p7*p3p6)
     . *(p1p10-p1p9)+32*(p3p7*p3p6)*(p1p10-p1p9)+48*(p6p7*p3p5)*(
     . p1p10-p1p9)+16*(p6p7*p3p4)*(-p1p10+p1p9)+16*(p6p7*p2p3)*(-
     . p1p10+p1p9)+48*(p6p7*p1p3)*(-p1p10+p1p9)+8*p5p6*(ffmcfmb*hbcm2
     . *p7p10+ffmcfmb*hbcm2*p7p9-fmb*hbcm*p7p10-2*fmb*hbcm*p7p9+2*fmc
     . *hbcm*p7p10+fmc*hbcm*p7p9+hbcm2*p7p10)+8*p6p7*(-2*ffmcfmb*
     . hbcm2*p1p10+2*ffmcfmb*hbcm2*p1p9+3*fmb*hbcm*p1p10-3*fmb*hbcm*
     . p1p9-3*fmc*hbcm*p1p10+3*fmc*hbcm*p1p9+3*hbcm2*p1p10-3*hbcm2*
     . p1p9)+16*(p1p3*p5p6*p7p9-p2p3*p5p6*p7p9-p3p5*p5p6*p7p9))+8*
     . p3p12*p5p6
      ans1=w1*(16*(p5p6*p3p7)*(p1p8-2*p2p8+2*p3p8+2*p4p8)+8*p5p6*(-
     . ffmcfmb*hbcm2*p7p8+2*fmb*hbcm*p7p8-fmc*hbcm*p7p8)+8*p6p7*(-2*
     . ffmcfmb*hbcm2*p1p8+3*fmb*hbcm*p1p8-3*fmc*hbcm*p1p8+3*hbcm2*
     . p1p8)+16*(-3*p1p3*p1p8*p6p7-p1p3*p5p6*p7p8+2*p1p5*p3p6*p7p8-2*
     . p1p5*p3p7*p6p8-p1p8*p2p3*p6p7-p1p8*p3p4*p6p7+3*p1p8*p3p5*p6p7+
     . 2*p1p8*p3p6*p3p7+3*p1p8*p3p6*p4p7-p1p8*p3p6*p5p7+p1p8*p3p7*
     . p4p6+p2p3*p5p6*p7p8-p3p4*p5p6*p7p8+p3p5*p5p6*p7p8))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(31)=12*ccc*w1*p5p6*(-fmb-fmc+hbcm)
      b(32)=12*ccc*w15*p5p6*(-fmb-fmc+hbcm)
      b(33)=ccc*(8*w1*p5p6*(ffmcfmb*hbcm*p7p8-fmb*p7p8-2*fmc*p7p8+
     . hbcm*p7p8)+w15*(24*p6p7*(-fmb*p1p9-fmc*p1p9+hbcm*p1p9)+8*p5p6*
     . (ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p9-fmb*p7p10+fmb*p7p9-2*fmc
     . *p7p10+2*fmc*p7p9+hbcm*p7p10-hbcm*p7p9)))
      b(34)=6*ccc*w15*(-fmb-fmc+hbcm)
      b(35)=ccc*(4*w1*(ffmcfmb*hbcm*p7p8-fmb*p7p8-2*fmc*p7p8+hbcm*
     . p7p8)+4*w15*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p9-fmb*p7p10+
     . fmb*p7p9-2*fmc*p7p10+2*fmc*p7p9+hbcm*p7p10-hbcm*p7p9))
      b(36)=2*ccc*w15*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+hbcm2+2*
     . p1p3-2*p2p3+6*p3p4-2*p3p5)
      b(37)=w15*ccc*(4*p3p7*(-ffmcfmb*hbcm-fmb-2*fmc+hbcm)-8*hbcm*
     . p5p7)
      b(38)=ccc*(w1*(8*p3p7*(-p1p8-p2p8-p4p8+p5p8)+4*(ffmcfmb*hbcm2*
     . p7p8-fmb*hbcm*p7p8+2*fmc*hbcm*p7p8+hbcm2*p7p8+4*p3p4*p7p8))+
     . w15*(16*p3p4*(p7p10-p7p9)+8*p3p7*(-p1p10+p1p9-p2p10+p2p9-p4p10
     . +p4p9+p5p10-p5p9)+4*(ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p9-
     . fmb*hbcm*p7p10+fmb*hbcm*p7p9+2*fmc*hbcm*p7p10-2*fmc*hbcm*p7p9+
     . hbcm2*p7p10-hbcm2*p7p9))+4*p3p12)
      b(42)=w15*ccc*(4*p6p7*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm+3
     . *hbcm2)+8*(-3*p1p3*p6p7-p2p3*p6p7-p3p4*p6p7+3*p3p5*p6p7+2*p3p6
     . *p3p7+3*p3p6*p4p7-p3p6*p5p7+p3p7*p4p6+p3p7*p5p6))
      b(43)=ccc*(w1*(8*p6p7*(3*p1p8+p2p8+p4p8-3*p5p8)+8*(-p3p6*p7p8-
     . p3p7*p6p8-p4p6*p7p8-3*p4p7*p6p8-p5p6*p7p8+p5p7*p6p8))+w15*(8*
     . p5p7*(p6p10-p6p9)+8*p5p6*(-p7p10+p7p9)+24*p4p7*(-p6p10+p6p9)+8
     . *p4p6*(-p7p10+p7p9)+8*p3p7*(-p6p10+p6p9)+8*p3p6*(-p7p10+p7p9)+
     . 8*p6p7*(3*p1p10-3*p1p9+p2p10-p2p9+p4p10-p4p9-3*p5p10+3*p5p9))+
     . 4*(p6p12-p7p13))
      b(47)=8*ccc*w15*(p3p6+p4p6+p5p6)
      b(52)=4*ccc*w15*(-p3p7+p4p7-3*p5p7)
      b(53)=ccc*(4*w1*(p1p8-p2p8+2*p3p8+3*p4p8-p5p8)+4*w15*(p1p10-
     . p1p9-p2p10+p2p9+2*p3p10-2*p3p9+3*p4p10-3*p4p9-p5p10+p5p9))
      b(60)=w15*ccc*(4*p3p6*(-3*fmb-3*fmc+hbcm)+8*(-hbcm*p4p6-hbcm*
     . p5p6))
      b(61)=-4*ccc*hbcm
      b(62)=-8*ccc*hbcm*p5p6
      b(63)=w15*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm-3*
     . hbcm2)+8*(3*p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-3*p3p5*p6p7-2*p3p6*
     . p3p7-3*p3p6*p4p7+p3p6*p5p7-p3p7*p4p6-p3p7*p5p6))
      b(64)=8*ccc*w15*(3*p1p5*p6p7+p3p7*p5p6-p4p7*p5p6+3*p5p6*p5p7)
      b(65)=w1*ccc*(4*p6p7*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm-3*
     . hbcm2)+8*(3*p1p3*p6p7+p2p3*p6p7+p3p4*p6p7-3*p3p5*p6p7-2*p3p6*
     . p3p7-3*p3p6*p4p7+p3p6*p5p7-p3p7*p4p6-p3p7*p5p6))
      b(66)=ccc*(w1*(8*p6p7*(3*hbcm*p1p8+hbcm*p2p8+hbcm*p4p8-3*hbcm*
     . p5p8)+8*p3p7*(ffmcfmb*hbcm*p6p8-2*fmb*p6p8-fmc*p6p8)+8*p3p6*(
     . ffmcfmb*hbcm*p7p8-fmb*p7p8-2*fmc*p7p8)+8*(-hbcm*p4p6*p7p8-2*
     . hbcm*p4p7*p6p8-hbcm*p5p6*p7p8))+w15*(8*p5p6*(-hbcm*p7p10+hbcm*
     . p7p9)+16*p4p7*(-hbcm*p6p10+hbcm*p6p9)+8*p4p6*(-hbcm*p7p10+hbcm
     . *p7p9)+8*p6p7*(3*hbcm*p1p10-3*hbcm*p1p9+hbcm*p2p10-hbcm*p2p9+
     . hbcm*p4p10-hbcm*p4p9-3*hbcm*p5p10+3*hbcm*p5p9)+8*p3p7*(ffmcfmb
     . *hbcm*p6p10-ffmcfmb*hbcm*p6p9-2*fmb*p6p10+2*fmb*p6p9-fmc*p6p10
     . +fmc*p6p9)+8*p3p6*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p9-fmb*
     . p7p10+fmb*p7p9-2*fmc*p7p10+2*fmc*p7p9))+4*(-hbcm*p6p12-hbcm*
     . p7p13))
      b(67)=8*ccc*w15*(-p3p6-p4p6-p5p6)
      b(68)=8*ccc*w1*(-p3p6-p4p6-p5p6)
      b(69)=4*ccc*w15*(p3p7-p4p7+3*p5p7)
      b(70)=8*ccc*w15*(-p1p2*p3p7+p1p3*p4p7-3*p1p3*p5p7+p1p4*p3p7)
      b(71)=4*ccc*w1*(p3p7-p4p7+3*p5p7)
      b(72)=4*ccc*w15*(-2*p1p2+2*p1p3+2*p1p4-p1p5)
      b(74)=12*ccc*w1*p6p7*(fmb+fmc-hbcm)
      b(75)=ccc*(12*w1*(fmb*p1p8+fmc*p1p8-hbcm*p1p8)+12*w15*(fmb*
     . p1p10-fmb*p1p9+fmc*p1p10-fmc*p1p9-hbcm*p1p10+hbcm*p1p9))
      b(76)=ccc*(w1*(8*p1p3*(2*p1p8+p2p8-2*p3p8-3*p4p8+p5p8)+4*(2*
     . ffmcfmb*hbcm2*p1p8-3*fmb*hbcm*p1p8+3*fmc*hbcm*p1p8-3*hbcm2*
     . p1p8+2*p1p8*p2p3+2*p1p8*p3p4-6*p1p8*p3p5))+w15*(24*p3p5*(-
     . p1p10+p1p9)+8*p3p4*(p1p10-p1p9)+8*p2p3*(p1p10-p1p9)+8*p1p3*(2*
     . p1p10-2*p1p9+p2p10-p2p9-2*p3p10+2*p3p9-3*p4p10+3*p4p9+p5p10-
     . p5p9)+4*(2*ffmcfmb*hbcm2*p1p10-2*ffmcfmb*hbcm2*p1p9-3*fmb*hbcm
     . *p1p10+3*fmb*hbcm*p1p9+3*fmc*hbcm*p1p10-3*fmc*hbcm*p1p9-3*
     . hbcm2*p1p10+3*hbcm2*p1p9)))
      b(77)=ccc*(w1*(8*p3p6*(-3*p1p8-p2p8-p4p8+3*p5p8)+4*(2*ffmcfmb*
     . hbcm2*p6p8-3*fmb*hbcm*p6p8+3*fmc*hbcm*p6p8-3*hbcm2*p6p8+6*p1p3
     . *p6p8+2*p2p3*p6p8+2*p3p4*p6p8-6*p3p5*p6p8))+w15*(24*p3p5*(-
     . p6p10+p6p9)+8*p3p4*(p6p10-p6p9)+8*p2p3*(p6p10-p6p9)+24*p1p3*(
     . p6p10-p6p9)+8*p3p6*(-3*p1p10+3*p1p9-p2p10+p2p9-p4p10+p4p9+3*
     . p5p10-3*p5p9)+4*(2*ffmcfmb*hbcm2*p6p10-2*ffmcfmb*hbcm2*p6p9-3*
     . fmb*hbcm*p6p10+3*fmb*hbcm*p6p9+3*fmc*hbcm*p6p10-3*fmc*hbcm*
     . p6p9-3*hbcm2*p6p10+3*hbcm2*p6p9)))
      b(78)=ccc*(12*w1*(fmb*p6p8+fmc*p6p8-hbcm*p6p8)+12*w15*(fmb*
     . p6p10-fmb*p6p9+fmc*p6p10-fmc*p6p9-hbcm*p6p10+hbcm*p6p9))
      b(79)=12*ccc*w15*p6p7*(-fmb-fmc+hbcm)
      b(81)=12*ccc*w15*p6p7*(fmb+fmc-hbcm)
      do 200 n=1,94 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp30_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(94) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,94 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      b(1)=8*ccc*w5*(p3p7*p4p6-p4p6*p4p7+3*p4p6*p5p7)
      b(2)=8*ccc*w2*(-p3p7*p4p6+p4p6*p4p7-3*p4p6*p5p7)
      b(3)=ccc*(16*w5*p1p4*(-p7p10+p7p8)-16*p1p4*p7p9*w2)
      b(4)=-16*ccc*p1p4*p3p7*w5
      b(5)=16*ccc*p1p4*p3p7*w2
      b(6)=ccc*(16*w5*p1p4*(hbcm*p7p10-hbcm*p7p8)+16*hbcm*p1p4*p7p9*
     . w2)
      b(7)=ccc*(8*w2*p4p6*(-3*p1p9-p2p9+2*p3p9+3*p4p9-p5p9)+8*w5*p4p6
     . *(-3*p1p10+3*p1p8-p2p10+p2p8+2*p3p10-2*p3p8+3*p4p10-3*p4p8-
     . p5p10+p5p8))
      b(8)=w5*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+
     . hbcm2)+8*(-3*p1p3*p4p6-p2p3*p4p6+3*p3p4*p4p6-p3p5*p4p6))
      b(9)=w2*ccc*(4*p4p6*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-
     . hbcm2)+8*(3*p1p3*p4p6+p2p3*p4p6-3*p3p4*p4p6+p3p5*p4p6))
      b(10)=ccc*(8*w2*p4p6*(3*hbcm*p1p9+hbcm*p2p9-2*hbcm*p3p9-3*hbcm*
     . p4p9+hbcm*p5p9)+8*w5*p4p6*(-3*fmb*p3p10+3*fmb*p3p8-3*fmc*p3p10
     . +3*fmc*p3p8+3*hbcm*p1p10-3*hbcm*p1p8+hbcm*p2p10-hbcm*p2p8+hbcm
     . *p3p10-hbcm*p3p8-3*hbcm*p4p10+3*hbcm*p4p8+hbcm*p5p10-hbcm*p5p8
     . ))
      b(11)=w5*ccc*(8*(p4p6*p3p7)*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+16*
     . hbcm*p4p6*p5p7)
      b(12)=w2*ccc*(8*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb-2*fmc+hbcm)-16*
     . hbcm*p4p6*p5p7)
      b(13)=-12*ccc*p1p4*w5
      b(14)=12*ccc*p1p4*w2
      b(15)=12*ccc*hbcm*p1p4*w5
      b(16)=-12*ccc*hbcm*p1p4*w2
      b(18)=6*ccc*w5*(-fmb-fmc+hbcm)
      b(19)=2*ccc*w5*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+6*
     . p1p3+2*p2p3-6*p3p4+2*p3p5)
      b(20)=w5*ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*hbcm*p5p7)
      b(21)=ccc*(4*w2*(-3*hbcm*p1p9-hbcm*p2p9+2*hbcm*p3p9+3*hbcm*p4p9
     . -hbcm*p5p9)+4*w5*(3*fmb*p3p10-3*fmb*p3p8+3*fmc*p3p10-3*fmc*
     . p3p8-3*hbcm*p1p10+3*hbcm*p1p8-hbcm*p2p10+hbcm*p2p8-hbcm*p3p10+
     . hbcm*p3p8+3*hbcm*p4p10-3*hbcm*p4p8-hbcm*p5p10+hbcm*p5p8))
      b(22)=-12*ccc*hbcm*p1p4*w5
      b(24)=6*ccc*w5*(fmb+fmc-hbcm)
      b(25)=2*ccc*w5*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+hbcm2-6*
     . p1p3-2*p2p3+6*p3p4-2*p3p5)
      b(26)=w5*ccc*(4*p3p7*(-ffmcfmb*hbcm-fmb-2*fmc+hbcm)-8*hbcm*p5p7
     . )
      b(27)=12*ccc*w5*p4p6*(fmb+fmc-hbcm)
      b(28)=w5*ccc*(4*p4p6*(2*ffmcfmb*hbcm2-3*fmb*hbcm+3*fmc*hbcm+
     . hbcm2)+8*(-3*p1p3*p4p6-p2p3*p4p6+3*p3p4*p4p6-p3p5*p4p6))
      b(29)=w5*ccc*(8*(p4p6*p3p7)*(-ffmcfmb*hbcm-fmb-2*fmc+hbcm)-16*
     . hbcm*p4p6*p5p7)
      b(30)=ccc*(w2*(32*(p4p6*p3p7)*(p1p9+p2p9-p3p9-p4p9)+8*p4p6*(
     . ffmcfmb*hbcm2*p7p9-2*fmb*hbcm*p7p9+fmc*hbcm*p7p9)+16*(-p1p3*
     . p4p6*p7p9-p2p3*p4p6*p7p9+p3p4*p4p6*p7p9-p3p5*p4p6*p7p9))+w5*(
     . 16*(p4p6*p3p4)*(-2*p7p10-p7p8)+16*(p4p6*p1p3)*(2*p7p10+p7p8)+
     . 32*(p4p6*p3p7)*(p1p10-p1p8+p2p10-p2p8-p3p10+p3p8-p4p10+p4p8)+8
     . *p4p6*(-ffmcfmb*hbcm2*p7p10-ffmcfmb*hbcm2*p7p8+fmb*hbcm*p7p10+
     . 2*fmb*hbcm*p7p8-2*fmc*hbcm*p7p10-fmc*hbcm*p7p8-hbcm2*p7p10)+16
     . *(p2p3*p4p6*p7p8+p3p5*p4p6*p7p8))+8*p3p12*p4p6)
      b(31)=12*ccc*w5*p4p6*(-fmb-fmc+hbcm)
      b(32)=12*ccc*w2*p4p6*(-fmb-fmc+hbcm)
      b(33)=ccc*(8*w2*p4p6*(-ffmcfmb*hbcm*p7p9+fmb*p7p9+2*fmc*p7p9-
     . hbcm*p7p9)+8*w5*p4p6*(-ffmcfmb*hbcm*p7p10+ffmcfmb*hbcm*p7p8+
     . fmb*p7p10-fmb*p7p8+2*fmc*p7p10-2*fmc*p7p8-hbcm*p7p10+hbcm*p7p8
     . ))
      b(34)=6*ccc*w2*(fmb+fmc-hbcm)
      b(35)=ccc*(4*w2*(ffmcfmb*hbcm*p7p9-fmb*p7p9-2*fmc*p7p9+hbcm*
     . p7p9)+4*w5*(ffmcfmb*hbcm*p7p10-ffmcfmb*hbcm*p7p8-fmb*p7p10+fmb
     . *p7p8-2*fmc*p7p10+2*fmc*p7p8+hbcm*p7p10-hbcm*p7p8))
      b(36)=2*ccc*w2*(-2*ffmcfmb*hbcm2+3*fmb*hbcm-3*fmc*hbcm-hbcm2+6*
     . p1p3+2*p2p3-6*p3p4+2*p3p5)
      b(37)=w2*ccc*(4*p3p7*(ffmcfmb*hbcm+fmb+2*fmc-hbcm)+8*hbcm*p5p7)
      b(38)=ccc*(w2*(8*p3p7*(p1p9-p2p9-p4p9+p5p9)+4*(ffmcfmb*hbcm2*
     . p7p9-fmb*hbcm*p7p9+2*fmc*hbcm*p7p9+hbcm2*p7p9-4*p1p3*p7p9+4*
     . p3p4*p7p9))+w5*(16*p3p4*(p7p10-p7p8)+16*p1p3*(-p7p10+p7p8)+8*
     . p3p7*(p1p10-p1p8-p2p10+p2p8-p4p10+p4p8+p5p10-p5p8)+4*(ffmcfmb*
     . hbcm2*p7p10-ffmcfmb*hbcm2*p7p8-fmb*hbcm*p7p10+fmb*hbcm*p7p8+2*
     . fmc*hbcm*p7p10-2*fmc*hbcm*p7p8+hbcm2*p7p10-hbcm2*p7p8))-4*
     . p3p12)
      b(52)=4*ccc*w2*(p3p7-p4p7+3*p5p7)
      b(53)=ccc*(4*w2*(-3*p1p9-p2p9+2*p3p9+3*p4p9-p5p9)+4*w5*(-3*
     . p1p10+3*p1p8-p2p10+p2p8+2*p3p10-2*p3p8+3*p4p10-3*p4p8-p5p10+
     . p5p8))
      b(61)=4*ccc*hbcm
      b(62)=-8*ccc*hbcm*p4p6
      b(64)=8*ccc*w5*(-p3p7*p4p6+p4p6*p4p7-3*p4p6*p5p7)
      b(69)=4*ccc*w5*(p3p7-p4p7+3*p5p7)
      b(70)=16*ccc*p1p4*p3p7*w5
      b(71)=4*ccc*w5*(-p3p7+p4p7-3*p5p7)
      b(72)=12*ccc*p1p4*w5
      do 200 n=1,94 
         c(n,2)=c(n,2)+0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp3_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(38) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,38 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(9)=8*ccc*w2*p4p6*(-fmb*hbcm+fmc*hbcm)
      b(10)=-16*ccc*hbcm*p4p6*p5p9*w2
      b(12)=-16*ccc*hbcm*p3p7*p4p6*w2
      b(21)=8*ccc*hbcm*p5p9*w2
      b(32)=-8*ccc*hbcm*p4p6*w2
      b(33)=16*ccc*hbcm*p4p6*p7p9*w2
      b(34)=4*ccc*hbcm*w2
      b(35)=-8*ccc*hbcm*p7p9*w2
      b(36)=4*ccc*w2*(-fmb*hbcm+fmc*hbcm)
      b(37)=8*ccc*hbcm*p3p7*w2
      b(38)=8*ccc*w2*(fmb*hbcm*p7p9-fmc*hbcm*p7p9)
      do 200 n=1,38 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp9_3p2(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(91) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,91 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*
     . p3p4+fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=w11*ccc*(8*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc
     . -fmb*hbcm+fmb2+fmc*hbcm)+16*(-p1p3*p6p7-2*p1p5*p6p7))
      b(3)=ccc*(w11*(32*p1p4*(-p7p11+p7p8)+16*p1p3*(-p7p11+p7p8)+32*
     . p1p2*(p7p11-p7p8)+8*(-ffmcfmb*fmb*hbcm*p7p11+ffmcfmb*fmb*hbcm*
     . p7p8+ffmcfmb*fmc*hbcm*p7p11-ffmcfmb*fmc*hbcm*p7p8+fmb*fmc*
     . p7p11-fmb*fmc*p7p8+fmb*hbcm*p7p11-fmb*hbcm*p7p8-fmb2*p7p11+
     . fmb2*p7p8-fmc*hbcm*p7p11+fmc*hbcm*p7p8))+8*w13*(ffmcfmb*fmb*
     . hbcm*p7p10-ffmcfmb*fmc*hbcm*p7p10-fmb*fmc*p7p10-fmb*hbcm*p7p10
     . +fmb2*p7p10+fmc*hbcm*p7p10-4*p1p2*p7p10+2*p1p3*p7p10+4*p1p4*
     . p7p10))
      b(4)=w11*ccc*(8*p3p7*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . fmc+fmb*hbcm-fmb2-fmc*hbcm)+16*(2*p1p2*p3p7+p1p3*p4p7+p1p3*
     . p5p7-2*p1p4*p3p7))
      ans=ccc*(w11*(16*p1p4*(hbcm*p7p11-hbcm*p7p8)+16*p1p2*(-hbcm*
     . p7p11+hbcm*p7p8)+16*p3p7*(ffmcfmb*hbcm*p5p11-ffmcfmb*hbcm*p5p8
     . -fmb*p1p11+fmb*p1p8+fmb*p5p11-fmb*p5p8-fmc*p1p11+fmc*p1p8-hbcm
     . *p5p11+hbcm*p5p8)+16*p3p4*(ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*
     . p7p8+fmb*p7p11-fmb*p7p8-hbcm*p7p11+hbcm*p7p8)+16*p2p3*(-
     . ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8-fmb*p7p11+fmb*p7p8+hbcm*
     . p7p11-hbcm*p7p8)+16*p1p3*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*
     . p7p8+fmc*p7p11-fmc*p7p8+hbcm*p7p11-hbcm*p7p8)+8*(ffmcfmb*hbcm3
     . *p7p11-ffmcfmb*hbcm3*p7p8+fmb*hbcm2*p7p11-fmb*hbcm2*p7p8-hbcm3
     . *p7p11+hbcm3*p7p8))+w13*(16*p3p7*(-ffmcfmb*hbcm*p5p10+fmb*
     . p1p10-fmb*p5p10+fmc*p1p10+hbcm*p5p10)+16*p3p4*(-ffmcfmb*hbcm*
     . p7p10-fmb*p7p10+hbcm*p7p10)+16*p2p3*(ffmcfmb*hbcm*p7p10+fmb*
     . p7p10-hbcm*p7p10)+16*p1p3*(ffmcfmb*hbcm*p7p10-fmc*p7p10-hbcm*
     . p7p10)+8*(-ffmcfmb*hbcm3*p7p10-fmb*hbcm2*p7p10+2*hbcm*p1p2*
     . p7p10-2*hbcm*p1p4*p7p10+hbcm3*p7p10)))
      b(6)=ans
      b(7)=ccc*(w11*(16*p1p5*(p6p11-p6p8)+16*p1p3*(p6p11-p6p8)+16*
     . p4p6*(-p1p11+p1p8)+16*p3p6*(-p1p11+p1p8)+8*(ffmcfmb*hbcm2*
     . p6p11-ffmcfmb*hbcm2*p6p8+fmb*hbcm*p6p11-fmb*hbcm*p6p8-hbcm2*
     . p6p11+hbcm2*p6p8))+8*w13*(-ffmcfmb*hbcm2*p6p10-fmb*hbcm*p6p10+
     . hbcm2*p6p10+2*p1p10*p3p6+2*p1p10*p4p6-2*p1p3*p6p10-2*p1p5*
     . p6p10)+8*(p1p13+p2p13-2*p3p13-p4p13-p5p13))
      b(8)=w11*ccc*(8*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . hbcm-fmc*hbcm)+16*(-p1p3*p4p6+p1p5*p3p6))
      ans=ccc*(w11*(16*p1p3*(-fmb*p6p11+fmb*p6p8-fmc*p6p11+fmc*p6p8)+
     . 16*p4p6*(-ffmcfmb*hbcm*p3p8-fmb*p3p8+hbcm*p3p8)+16*p3p6*(
     . ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p1p8+ffmcfmb*hbcm*p2p11-
     . ffmcfmb*hbcm*p2p8-ffmcfmb*hbcm*p3p11-ffmcfmb*hbcm*p4p11+
     . ffmcfmb*hbcm*p4p8-hbcm*p1p11+hbcm*p1p8-hbcm*p2p11+hbcm*p2p8+
     . hbcm*p3p11+hbcm*p4p11-hbcm*p4p8)+8*(-ffmcfmb*fmb*hbcm2*p6p11+
     . ffmcfmb*fmb*hbcm2*p6p8-ffmcfmb*fmc*hbcm2*p6p11+ffmcfmb*fmc*
     . hbcm2*p6p8-fmb*fmc*hbcm*p6p11+fmb*fmc*hbcm*p6p8+fmb*hbcm2*
     . p6p11-fmb*hbcm2*p6p8-fmb2*hbcm*p6p11+fmb2*hbcm*p6p8+fmc*hbcm2*
     . p6p11-fmc*hbcm2*p6p8))+w13*(16*p1p3*(fmb*p6p10+fmc*p6p10)+16*
     . p4p6*(-ffmcfmb*hbcm*p3p10-fmb*p3p10+hbcm*p3p10)+16*p3p6*(-
     . ffmcfmb*hbcm*p1p10-ffmcfmb*hbcm*p2p10+ffmcfmb*hbcm*p4p10+hbcm*
     . p1p10+hbcm*p2p10-hbcm*p4p10)+8*(ffmcfmb*fmb*hbcm2*p6p10+
     . ffmcfmb*fmc*hbcm2*p6p10+fmb*fmc*hbcm*p6p10-fmb*hbcm2*p6p10+
     . fmb2*hbcm*p6p10-fmc*hbcm2*p6p10))+8*(fmb*p3p13+fmc*p3p13-hbcm*
     . p1p13+hbcm*p5p13))
      b(10)=ans
      b(11)=w11*ccc*(8*p6p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p6
     . *p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p7*p3p6)*(ffmcfmb*hbcm-
     . hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p6p7*p3p5)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p1p3)*(ffmcfmb*hbcm-fmc-hbcm)-
     . 16*hbcm*p1p5*p6p7)
      b(13)=4*ccc*w11*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc+fmb
     . *hbcm-fmb2-fmc*hbcm+4*p1p2-2*p1p3-4*p1p4)
      b(15)=w11*ccc*(8*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p2p3*(-ffmcfmb*
     . hbcm-fmb+hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmc+hbcm)+4*(ffmcfmb*
     . hbcm3+fmb*hbcm2-2*hbcm*p1p2+2*hbcm*p1p4-hbcm3))
      b(17)=w13*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-fmc-hbcm)-8*hbcm*p4p6)
      b(18)=4*ccc*w13*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(19)=4*ccc*w13*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3-4*p2p3+4*
     . p3p4)
      b(20)=w13*ccc*(8*p3p7*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+8*(-hbcm*
     . p4p7-hbcm*p5p7))
      b(21)=ccc*(8*w11*(ffmcfmb*hbcm*p3p8-fmb*p3p8-2*fmc*p3p8-hbcm*
     . p1p11+hbcm*p1p8-hbcm*p2p11+hbcm*p2p8+hbcm*p3p11-hbcm*p3p8+hbcm
     . *p4p11-hbcm*p4p8)+8*w13*(ffmcfmb*hbcm*p3p10-fmb*p3p10-2*fmc*
     . p3p10+hbcm*p1p10+hbcm*p2p10-hbcm*p3p10-hbcm*p4p10))
      b(22)=w13*ccc*(8*p3p4*(ffmcfmb*hbcm+fmb-hbcm)+8*p2p3*(-ffmcfmb*
     . hbcm-fmb+hbcm)+8*p1p3*(-ffmcfmb*hbcm+fmc+hbcm)+4*(ffmcfmb*
     . hbcm3+fmb*hbcm2-2*hbcm*p1p2+2*hbcm*p1p4-hbcm3))
      b(23)=w11*ccc*(8*p3p6*(ffmcfmb*hbcm-fmb-fmc-hbcm)-8*hbcm*p4p6)
      b(24)=4*ccc*w11*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(25)=4*ccc*w11*(-fmb*hbcm+fmc*hbcm+2*hbcm2-2*p1p3-4*p2p3+4*
     . p3p4)
      b(26)=w11*ccc*(8*p3p7*(ffmcfmb*hbcm-fmb-2*fmc-hbcm)+8*(-hbcm*
     . p4p7-hbcm*p5p7))
      b(27)=w13*ccc*(8*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb*
     . hbcm-hbcm))
      b(28)=w13*ccc*(8*p3p6*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . hbcm+fmc*hbcm)+16*(p1p3*p4p6-p1p5*p3p6))
      b(29)=w13*ccc*(8*p6p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p6
     . *p3p7)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p7*p3p6)*(ffmcfmb*hbcm-
     . hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p6p7*p3p5)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p1p3)*(ffmcfmb*hbcm-fmc-hbcm)-
     . 16*hbcm*p1p5*p6p7)
      ans2=w13*(32*(p6p7*p1p3)*(2*p1p10-p5p10)+16*p5p7*(ffmcfmb*hbcm2
     . *p6p10+fmb*hbcm*p6p10-hbcm2*p6p10)+16*p4p7*(ffmcfmb*hbcm2*
     . p6p10+fmb*hbcm*p6p10-hbcm2*p6p10)+16*p3p7*(ffmcfmb*hbcm2*p6p10
     . +fmb*hbcm*p6p10-hbcm2*p6p10)+16*p3p6*(-ffmcfmb*fmb*hbcm*p7p10+
     . ffmcfmb*fmc*hbcm*p7p10+fmb*hbcm*p7p10-fmc*hbcm*p7p10)+16*(-3*
     . hbcm2*p1p10*p6p7+2*p1p10*p2p3*p6p7-2*p1p10*p3p4*p6p7-4*p1p10*
     . p3p5*p6p7-4*p1p10*p3p6*p3p7-2*p1p10*p3p6*p4p7-2*p1p10*p3p6*
     . p5p7-4*p1p10*p3p7*p4p6+2*p1p3*p3p7*p6p10-2*p1p3*p4p6*p7p10+2*
     . p1p3*p4p7*p6p10+2*p1p3*p5p7*p6p10+2*p1p5*p3p6*p7p10))+(16*p3p7
     . *(-p1p13-p2p13+p3p13+p4p13+p5p13)+8*(fmb*hbcm*p7p13-fmc*hbcm*
     . p7p13+2*hbcm2*p7p13-2*p1p3*p7p13-2*p2p3*p7p13+2*p3p4*p7p13+2*
     . p3p5*p7p13))
      ans1=w11*(32*(p5p7*p1p3)*(-p6p11+p6p8)+32*(p4p7*p1p3)*(-p6p11+
     . p6p8)+32*(p3p7*p1p3)*(-p6p11+p6p8)+64*(p4p6*p3p7)*(p1p11-p1p8)
     . +32*(p5p7*p3p6)*(p1p11-p1p8)+32*(p4p7*p3p6)*(p1p11-p1p8)+64*(
     . p3p7*p3p6)*(p1p11-p1p8)+64*(p6p7*p3p5)*(p1p11-p1p8)+32*(p6p7*
     . p3p4)*(p1p11-p1p8)+32*(p6p7*p2p3)*(-p1p11+p1p8)+32*(p6p7*p1p3)
     . *(-2*p1p11+2*p1p8+p5p11-p5p8)+48*p6p7*(hbcm2*p1p11-hbcm2*p1p8)
     . +16*p5p7*(-ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8-fmb*hbcm*
     . p6p11+fmb*hbcm*p6p8+hbcm2*p6p11-hbcm2*p6p8)+16*p4p7*(-ffmcfmb*
     . hbcm2*p6p11+ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm*p6p8+
     . hbcm2*p6p11-hbcm2*p6p8)+16*p3p7*(-ffmcfmb*hbcm2*p6p11+ffmcfmb*
     . hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm*p6p8+hbcm2*p6p11-hbcm2*p6p8
     . )+16*p3p6*(ffmcfmb*fmb*hbcm*p7p11-ffmcfmb*fmc*hbcm*p7p11-fmb*
     . hbcm*p7p11+fmc*hbcm*p7p11)+32*(p1p3*p4p6*p7p11-p1p5*p3p6*p7p11
     . ))+ans2
      ans=ccc*ans1
      b(30)=ans
      b(31)=w11*ccc*(8*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb*
     . hbcm-hbcm))
      b(33)=ccc*(w11*(16*p4p6*(-ffmcfmb*hbcm*p7p11+ffmcfmb*hbcm*p7p8-
     . fmb*p7p11+fmb*p7p8+hbcm*p7p11-hbcm*p7p8)+16*p3p6*(-ffmcfmb*
     . hbcm*p7p11+ffmcfmb*hbcm*p7p8+hbcm*p7p11-hbcm*p7p8)+16*p6p7*(
     . ffmcfmb*hbcm*p1p11-ffmcfmb*hbcm*p1p8-ffmcfmb*hbcm*p5p11+
     . ffmcfmb*hbcm*p5p8-fmb*p5p11+fmb*p5p8-fmc*p1p11+fmc*p1p8+hbcm*
     . p5p11-hbcm*p5p8))+w13*(16*p4p6*(ffmcfmb*hbcm*p7p10+fmb*p7p10-
     . hbcm*p7p10)+16*p3p6*(ffmcfmb*hbcm*p7p10-hbcm*p7p10)+16*p6p7*(-
     . ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p5p10+fmb*p5p10+fmc*p1p10-hbcm
     . *p5p10))-8*hbcm*p7p13)
      b(35)=ccc*(8*w11*(ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*p7p8-fmb*
     . p7p11+fmb*p7p8-2*fmc*p7p11+2*fmc*p7p8)+8*w13*(-ffmcfmb*hbcm*
     . p7p10+fmb*p7p10+2*fmc*p7p10))
      b(38)=ccc*(w11*(32*p3p4*(p7p11-p7p8)+32*p2p3*(-p7p11+p7p8)+16*
     . p1p3*(-p7p11+p7p8)+16*p3p7*(-p1p11+p1p8+2*p5p11-2*p5p8)+8*(-
     . fmb*hbcm*p7p11+fmb*hbcm*p7p8+fmc*hbcm*p7p11-fmc*hbcm*p7p8+2*
     . hbcm2*p7p11-2*hbcm2*p7p8))+w13*(16*p3p7*(p1p10-2*p5p10)+8*(fmb
     . *hbcm*p7p10-fmc*hbcm*p7p10-2*hbcm2*p7p10+2*p1p3*p7p10+4*p2p3*
     . p7p10-4*p3p4*p7p10)))
      b(39)=w11*ccc*(8*p6p7*(-fmb*hbcm+fmc*hbcm-2*hbcm2)+16*(p1p3*
     . p6p7-2*p3p5*p6p7-p3p6*p3p7-2*p3p7*p4p6))
      b(40)=w11*ccc*(8*p6p7*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p4p6*
     . p3p7)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm+
     . hbcm)+16*(p4p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(p6p7*p3p5)*(
     . ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*p1p3)*(-ffmcfmb*hbcm+fmc+hbcm)
     . +16*hbcm*p1p5*p6p7)
      b(41)=w11*ccc*(8*p6p7*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . fmc+fmb*hbcm-fmb2-fmc*hbcm)+16*(p1p3*p6p7+2*p1p5*p6p7))
      b(43)=ccc*(w11*(32*p4p6*(-p7p11+p7p8)+16*p3p6*(-p7p11+p7p8)+16*
     . p6p7*(p1p11-p1p8-2*p5p11+2*p5p8))+w13*(16*p6p7*(-p1p10+2*p5p10
     . )+16*(p3p6*p7p10+2*p4p6*p7p10)))
      b(44)=w11*ccc*(8*p3p6*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . hbcm-fmc*hbcm)+16*(-p1p3*p4p6+p1p5*p3p6))
      b(45)=8*ccc*w11*(-p3p6-2*p4p6)
      b(46)=w11*ccc*(8*p4p6*(ffmcfmb*hbcm+fmb-hbcm)+8*p3p6*(ffmcfmb*
     . hbcm-hbcm))
      b(49)=w11*ccc*(8*p3p7*(-ffmcfmb*hbcm+fmb+2*fmc+hbcm)+8*(hbcm*
     . p4p7+hbcm*p5p7))
      b(50)=16*ccc*w11*(p3p7+p4p7+p5p7)
      b(51)=w11*ccc*(8*p3p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc-fmb*hbcm+fmb2+fmc*hbcm)+16*(-2*p1p2*p3p7-p1p3*p4p7-p1p3*
     . p5p7+2*p1p4*p3p7))
      b(53)=ccc*(8*w11*(-p1p11+p1p8-2*p2p11+2*p2p8+2*p3p11-2*p3p8+2*
     . p4p11-2*p4p8)+8*w13*(p1p10+2*p2p10-2*p3p10-2*p4p10))
      b(54)=4*ccc*w11*(fmb*hbcm-fmc*hbcm-2*hbcm2+2*p1p3+4*p2p3-4*p3p4
     . )
      b(55)=4*ccc*w11*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(56)=4*ccc*w11*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc-fmb*
     . hbcm+fmb2+fmc*hbcm-4*p1p2+2*p1p3+4*p1p4)
      b(58)=w11*ccc*(8*p3p6*(-ffmcfmb*hbcm+fmb+fmc+hbcm)+8*hbcm*p4p6)
      b(59)=w11*ccc*(8*p3p4*(-ffmcfmb*hbcm-fmb+hbcm)+8*p2p3*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*p1p3*(ffmcfmb*hbcm-fmc-hbcm)+4*(-ffmcfmb*
     . hbcm3-fmb*hbcm2+2*hbcm*p1p2-2*hbcm*p1p4+hbcm3))
      b(63)=w13*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-p1p3*
     . p6p7+2*p3p5*p6p7+p3p6*p3p7+2*p3p7*p4p6))
      b(64)=w13*ccc*(8*p6p7*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*
     . fmc-fmb*hbcm+fmb2+fmc*hbcm)+16*(-p1p3*p6p7-2*p1p5*p6p7))
      b(65)=w11*ccc*(8*p6p7*(fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(-p1p3*
     . p6p7+2*p3p5*p6p7+p3p6*p3p7+2*p3p7*p4p6))
      b(66)=ccc*(w11*(16*p6p7*(-hbcm*p5p11+hbcm*p5p8)+16*p4p6*(-hbcm*
     . p7p11+hbcm*p7p8)+16*p3p6*(ffmcfmb*hbcm*p7p11-ffmcfmb*hbcm*p7p8
     . -fmb*p7p11+fmb*p7p8-fmc*p7p11+fmc*p7p8-hbcm*p7p11+hbcm*p7p8))+
     . w13*(16*p3p6*(-ffmcfmb*hbcm*p7p10+fmb*p7p10+fmc*p7p10+hbcm*
     . p7p10)+16*(hbcm*p4p6*p7p10+hbcm*p5p10*p6p7))+8*hbcm*p7p13)
      b(67)=8*ccc*w13*(p3p6+2*p4p6)
      b(68)=8*ccc*w11*(p3p6+2*p4p6)
      b(69)=16*ccc*w13*(-p3p7-p4p7-p5p7)
      b(70)=w13*ccc*(8*p3p7*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*
     . fmc+fmb*hbcm-fmb2-fmc*hbcm)+16*(2*p1p2*p3p7+p1p3*p4p7+p1p3*
     . p5p7-2*p1p4*p3p7))
      b(71)=16*ccc*w11*(-p3p7-p4p7-p5p7)
      b(72)=4*ccc*w13*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc+fmb
     . *hbcm-fmb2-fmc*hbcm+4*p1p2-2*p1p3-4*p1p4)
      b(73)=8*ccc*w11*p6p7*(-ffmcfmb*hbcm+fmb+2*fmc)
      b(74)=8*ccc*w11*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(75)=ccc*(8*w11*(-ffmcfmb*hbcm*p1p11+ffmcfmb*hbcm*p1p8-ffmcfmb
     . *hbcm*p2p11+ffmcfmb*hbcm*p2p8+ffmcfmb*hbcm*p3p11-ffmcfmb*hbcm*
     . p3p8+ffmcfmb*hbcm*p4p11-ffmcfmb*hbcm*p4p8-fmb*p2p11+fmb*p2p8+
     . fmb*p3p11-fmb*p3p8+fmb*p4p11-fmb*p4p8+fmc*p1p11-fmc*p1p8+hbcm*
     . p1p11-hbcm*p1p8+hbcm*p2p11-hbcm*p2p8-hbcm*p3p11+hbcm*p3p8-hbcm
     . *p4p11+hbcm*p4p8)+8*w13*(ffmcfmb*hbcm*p1p10+ffmcfmb*hbcm*p2p10
     . -ffmcfmb*hbcm*p3p10-ffmcfmb*hbcm*p4p10+fmb*p2p10-fmb*p3p10-fmb
     . *p4p10-fmc*p1p10-hbcm*p1p10-hbcm*p2p10+hbcm*p3p10+hbcm*p4p10))
      b(76)=ccc*(w11*(16*p3p5*(-p1p11+p1p8)+16*p1p3*(p1p11-p1p8+p2p11
     . -p2p8-p3p11+p3p8-p4p11+p4p8)+8*(-hbcm2*p1p11+hbcm2*p1p8))+w13*
     . (16*p1p3*(-p1p10-p2p10+p3p10+p4p10)+8*(hbcm2*p1p10+2*p1p10*
     . p3p5)))
      b(77)=ccc*(w11*(16*p3p6*(p5p11-p5p8)+16*p3p5*(-p6p11+p6p8)+8*(-
     . ffmcfmb*hbcm2*p6p11+ffmcfmb*hbcm2*p6p8-fmb*hbcm*p6p11+fmb*hbcm
     . *p6p8))+8*w13*(ffmcfmb*hbcm2*p6p10+fmb*hbcm*p6p10+2*p3p5*p6p10
     . -2*p3p6*p5p10)+8*p3p13)
      b(78)=ccc*(8*w11*(fmb*p6p11-fmb*p6p8+fmc*p6p11-fmc*p6p8)+8*w13*
     . (-fmb*p6p10-fmc*p6p10))
      b(81)=8*ccc*w13*p6p7*(ffmcfmb*hbcm-fmb-2*fmc)
      b(83)=8*ccc*hbcm*p3p7
      b(84)=8*ccc*(p3p7+p4p7+p5p7)
      b(85)=4*ccc*(-fmb*hbcm+fmc*hbcm-2*hbcm2+2*p1p3+2*p2p3-2*p3p4-2*
     . p3p5)
      b(87)=4*ccc*hbcm
      b(88)=w13*ccc*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+8*p4p7*(-ffmcfmb
     . *hbcm-fmb+hbcm)+8*p3p7*(-ffmcfmb*hbcm-fmb+hbcm))
      b(89)=4*ccc*hbcm
      b(90)=w11*ccc*(8*p5p7*(ffmcfmb*hbcm+fmb-hbcm)+8*p4p7*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*p3p7*(ffmcfmb*hbcm+fmb-hbcm))
      b(91)=w11*ccc*(8*p5p7*(-ffmcfmb*hbcm-fmb+hbcm)+8*p4p7*(-ffmcfmb
     . *hbcm-fmb+hbcm)+8*p3p7*(-ffmcfmb*hbcm-fmb+hbcm))
      do 200 n=1,91 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      function ams1_3p2(n) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
#include "inclppp.h"
#include "inclcon.h"
#include "inclamp.h"
      b1=-p15p15
      b2=-p14p14
      aa1=c(30,n)**2+b1+b2
      b1=p15p15
      b2=-p14p14
      b3=p17p17
      b4=-p16p16
      b5=b3+b4
      b6=-2*c(30,n)*p0p16
      b7=-c(30,n)**2+b1+b2+b5+b6
      b8=2*p0p15*p5p17
      b9=b8
      b10=-2*p0p17*p5p15
      b11=b10
      b12=-2*p15p17
      b13=2*p14p16
      b14=b12+b13
      b15=-2*epspn2
      b16=-2*epspn4
      b17=-2*p0p14*p5p16
      b18=b17
      b19=2*p0p16*p5p14
      b20=b19
      b21=2*c(30,n)*p5p14
      aa2=b11+b14*p0p5+b15+b16+b18+b20+b21+b7*fmb+b9
      b1=2*p4p15*p5p15
      aa3=b1
      aa4=-2*epspn0
      b1=-2*p0p15*p4p17
      b2=b1
      b3=-2*p0p17*p4p15
      b4=b3
      b5=2*p15p17
      b6=2*p14p16
      b7=b5+b6
      b8=2*epspn1
      b9=-2*epspn3
      b10=-2*p0p14*p4p16
      b11=b10
      b12=-2*p0p16*p4p14
      b13=b12
      b14=-2*c(30,n)*p4p14
      aa5=b11+b13+b14+b2+b4+b7*p0p4+b8+b9
      b1=2*p4p14*p5p14
      aa6=b1
      b1=p17p17
      b2=p16p16
      b3=b1+b2
      b4=-2*p4p17*p5p17
      b5=b4
      b6=2*epspn5
      b7=-2*p4p16*p5p16
      b8=b7
      aa7=b3*p4p5+b5+b6+b8
      b1=4*p0p17*p4p17
      b2=b1
      b3=4*epspn6
      b4=4*p0p16*p4p16
      b5=b4
      b6=2*c(30,n)*p4p16
      aa8=b2+b3+b5+b6
      b1=-2*p17p17
      b2=-2*p16p16
      b3=b1+b2
      b4=-2*c(30,n)*p5p16
      aa9=b3*p0p5+b4
      aa10=2*c(30,n)*epspn7
      b1=2*c(30,n)*p0p16*p4p5
      aa11=b1
      ams1_3p2=aa1*p4p5+aa10+aa11+aa2*fmc+aa3+aa4+aa5*fmb+aa6+aa7+aa8*
     . p0p5+aa9*p0p4
      return
      end
      
