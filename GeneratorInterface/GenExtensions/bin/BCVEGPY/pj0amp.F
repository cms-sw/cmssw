c...this is the 36 amplitude for (3p0) state.
c...obtained from the program fdc.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) z.x. zhang, j.x. wang and x.g. wu                    c
c reference: hep-ph/0309120                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine amp10_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(1)=ccc*(w1*(8*p6p7*(2*fmb*p3p4s2-fmb2*fmc*hbcm2-fmc*hbcm4-2*
     . fmc*p3p4s2+fmc3*hbcm2)+16*(p6p7*p3p5*p3p4)*(fmb-fmc)+16*(p6p7*
     . p3p4*p2p3)*(-fmb+fmc)+16*(p6p7*p3p4*p1p3)*(-fmb+fmc)+16*(p6p7*
     . p3p4)*(fmb*hbcm2-fmc*hbcm2)+16*(fmc*hbcm2*p1p3*p6p7+fmc*hbcm2*
     . p2p3*p6p7-fmc*hbcm2*p3p5*p6p7))+(32*(p6p7*p3p4)*(-fb2*fmb+fb2*
     . fmc)+32*(p6p7*p3p5)*(2*fb1*hbcm-fb2*fmb+fb2*fmc)+32*(p6p7*p2p3
     . )*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+32*(p6p7*p1p3)*(-2*fb1*hbcm+
     . fb2*fmb-fb2*fmc)+32*p6p7*(fb1*fmb2*hbcm-fb1*fmc2*hbcm+fb1*
     . hbcm3-fb2*fmb*hbcm2+fb2*fmc*hbcm2)))
      b(2)=ccc*(w1*(8*(p5p7*p3p4)*(-fmb+fmc)+8*fmc*hbcm2*p5p7)+16*
     . p5p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(3)=ccc*(w1*(8*(p5p6*p3p4)*(-fmb+fmc)+8*(p4p6*p3p4)*(-fmb+fmc)
     . +8*(p3p6*p3p4)*(-fmb+fmc)+8*(fmc*hbcm2*p3p6+fmc*hbcm2*p4p6+fmc
     . *hbcm2*p5p6))+(16*p5p6*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+16*p4p6*(
     . -2*fb1*hbcm+fb2*fmb-fb2*fmc)+16*p3p6*(-2*fb1*hbcm+fb2*fmb-fb2*
     . fmc)))
      b(4)=ccc*(w1*(8*(p6p7*p3p4)*(fmb2-fmc2-hbcm2)-16*p3p4s2*p6p7)+(
     . 16*p6p7*(-fb2*fmb2+fb2*fmc2+fb2*hbcm2)+32*fb2*p3p4*p6p7))
      b(5)=ccc*(w1*(8*p5p7*(fmb*fmc*hbcm2+fmc2*hbcm2-2*p3p4s2)-8*
     . hbcm2*p3p4*p5p7)+(16*p5p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*
     . hbcm2)+32*fb2*p3p4*p5p7))
      b(6)=ccc*(w1*(8*(p3p6*p3p4)*(-fmb2+fmc2)+8*p5p6*(fmb*fmc*hbcm2+
     . fmc2*hbcm2-2*p3p4s2)+8*p4p6*(fmb*fmc*hbcm2+fmc2*hbcm2-2*p3p4s2
     . )+8*p3p6*(fmb*fmc*hbcm2+fmc2*hbcm2)+8*(-hbcm2*p3p4*p4p6-hbcm2*
     . p3p4*p5p6))+(16*p5p6*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*hbcm2
     . )+16*p4p6*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*hbcm2)+16*p3p6*(
     . -2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*fmb2-fb2*fmc2)+32*(fb2*p3p4
     . *p4p6+fb2*p3p4*p5p6)))
      b(7)=ccc*(w1*(4*p3p4*(fmb2-fmc2-hbcm2)-8*p3p4s2)+8*(-fb2*fmb2+
     . fb2*fmc2+fb2*hbcm2+2*fb2*p3p4))
      b(8)=ccc*(w1*(8*(p3p5*p3p4)*(-fmb+fmc)+8*(p3p4*p2p3)*(fmb-fmc)+
     . 8*(p3p4*p1p3)*(fmb-fmc)+8*p3p4*(-fmb*hbcm2+fmc*hbcm2)+4*(-2*
     . fmb*p3p4s2+fmb2*fmc*hbcm2-2*fmc*hbcm2*p1p3-2*fmc*hbcm2*p2p3+2*
     . fmc*hbcm2*p3p5+fmc*hbcm4+2*fmc*p3p4s2-fmc3*hbcm2))+(16*p3p4*(
     . fb2*fmb-fb2*fmc)+16*p3p5*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+16*p2p3
     . *(2*fb1*hbcm-fb2*fmb+fb2*fmc)+16*p1p3*(2*fb1*hbcm-fb2*fmb+fb2*
     . fmc)+16*(-fb1*fmb2*hbcm+fb1*fmc2*hbcm-fb1*hbcm3+fb2*fmb*hbcm2-
     . fb2*fmc*hbcm2)))
      b(9)=ccc*(w1*(8*(p3p6*p3p4)*(fmb-fmc)-8*fmc*hbcm2*p3p6)+16*p3p6
     . *(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      b(11)=ccc*(4*w1*(-fmb*fmc*hbcm2-fmc2*hbcm2+hbcm2*p3p4+2*p3p4s2)
     . +8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm-fb2*hbcm2-2*fb2*p3p4))
      b(13)=ccc*(w1*(4*p3p4*(-fmb+fmc)+4*fmc*hbcm2)+8*(-2*fb1*hbcm+
     . fb2*fmb-fb2*fmc))
      b(15)=ccc*(w1*(8*p6p7*(-fmb*fmc*hbcm2-fmc2*hbcm2+2*p3p4s2)+8*
     . hbcm2*p3p4*p6p7)+(16*p6p7*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm-fb2*
     . hbcm2)-32*fb2*p3p4*p6p7))
      b(16)=ccc*(w1*(8*(p6p7*p3p4)*(fmb-fmc)-8*fmc*hbcm2*p6p7)+16*
     . p6p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp18_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      ans4=8*(p6p7*p3p5)*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm3+fmb*hbcm2-
     . fmb2*hbcm)+8*(p6p7*p3p4)*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3-fmb*
     . hbcm2+fmb2*hbcm)+16*(p3p7*p3p6*p3p5)*(ffmcfmb**2*hbcm+ffmcfmb*
     . fmc-ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2*hbcm+
     . ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+16*(fmb*p3p4*p3p7*p4p6+fmb*p3p5*
     . p3p7*p4p6-fmc*p2p3*p3p5*p6p7-2*hbcm*p2p3*p4p5*p6p7-2*hbcm*p2p3
     . *p4p6*p5p7+2*hbcm*p2p3*p4p7*p5p6+2*hbcm*p2p4*p3p5*p6p7+2*hbcm*
     . p2p4*p3p6*p5p7-2*hbcm*p2p4*p3p7*p5p6-hbcm*p2p5*p3p4*p6p7+hbcm*
     . p2p5*p3p5*p6p7-2*hbcm*p2p5*p3p6*p4p7+2*hbcm*p2p5*p3p7*p4p6)
      ans3=16*(p4p6*p3p7)*(fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm)+16*(p6p7
     . *p4p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p6p7*p2p4)*(
     . ffmcfmb*hbcm3+fmb*hbcm2)+32*(p4p5*p3p7*p3p6)*(ffmcfmb*hbcm-
     . hbcm)+16*(p5p6*p3p7*p3p5)*(-ffmcfmb*hbcm+hbcm)+16*(p4p7*p3p6*
     . p3p5)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p6*p3p7*p3p4)*(-ffmcfmb*
     . hbcm+hbcm)+16*(p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(
     . p6p7*p3p5*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*p3p4*p2p3)*(-
     . 2*ffmcfmb*hbcm-fmc)+16*(p5p7*p3p6)*(-ffmcfmb*fmc*hbcm2-fmb*fmc
     . *hbcm+fmc*hbcm2)+8*p6p7*(-ffmcfmb*fmb2*hbcm3-ffmcfmb*fmc2*
     . hbcm3+2*ffmcfmb*hbcm*p3p5s2-fmb*fmc2*hbcm2+2*fmb*p3p5s2+fmb2*
     . hbcm3-fmb3*hbcm2+fmc2*hbcm3-2*hbcm*p3p5s2)+16*(p6p7*p2p3)*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmc2*hbcm)+16
     . *(p3p7*p3p6)*(ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc2*hbcm-fmb*fmc*
     . hbcm-fmc2*hbcm)+16*(p5p6*p3p7)*(-ffmcfmb**2*hbcm3+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm)+16*(p4p7*p3p6)*(-ffmcfmb**2*hbcm3-ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*
     . hbcm2)+ans4
      ans2=w18*ans3
      ans6=w8*(16*(p6p7*p2p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(
     . p3p7*p3p6*p2p4)*(ffmcfmb*hbcm-hbcm)+16*(p5p6*p3p7*p2p3)*(
     . ffmcfmb*hbcm-hbcm)+16*(p4p7*p3p6*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm
     . )+16*(p6p7*p3p5*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p2p3)*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2)+
     . 16*(p3p7*p3p6*p2p3)*(ffmcfmb**2*hbcm+ffmcfmb*fmc-ffmcfmb*hbcm-
     . fmc)+16*(fmb*p2p3*p3p7*p4p6-fmc*p2p3s2*p6p7-hbcm*p2p3*p2p5*
     . p6p7))+(16*(p4p6*p3p7)*(-2*fb2*fmb-hbcm)+8*(p5p6*p3p7)*(4*fb1*
     . ffmcfmb*hbcm-4*fb1*hbcm-hbcm)+8*(p4p7*p3p6)*(4*fb1*ffmcfmb*
     . hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+8*(p6p7*p3p5)*(-4*fb1*ffmcfmb*
     . hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+4*p6p7*(-4*fb1*ffmcfmb**2*
     . hbcm3+4*fb1*ffmcfmb*hbcm3+4*fb1*fmb2*hbcm-4*fb2*fmb*hbcm2+2*
     . ffmcfmb*hbcm3+2*fmb*hbcm2-3*hbcm3)+8*(p3p7*p3p6)*(4*fb1*
     . ffmcfmb**2*hbcm-4*fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmc+4*fb2*fmc
     . -4*ffmcfmb*hbcm-fmc+3*hbcm)+32*(-fb1*hbcm*p2p5*p6p7+fb2*fmc*
     . p2p3*p6p7))
      ans5=w1*(16*(p6p7*p4p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p5p6*p3p7*p3p4)*(-ffmcfmb*hbcm+hbcm)+16*(p4p7*p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p5*p3p4)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p6p7*p3p4*p2p3)*(-2*ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6)
     . *(ffmcfmb*fmc2*hbcm-fmc2*hbcm)+16*(p5p6*p3p7)*(-ffmcfmb*fmc*
     . hbcm2+fmc*hbcm2)+16*(p6p7*p3p5)*(ffmcfmb*fmc*hbcm2+fmb*fmc*
     . hbcm-fmc*hbcm2)+16*(p6p7*p2p3)*(-ffmcfmb*fmc*hbcm2-fmc2*hbcm)+
     . 16*(p4p6*p3p7)*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+8*p6p7*(-
     . ffmcfmb*fmb*fmc*hbcm3-ffmcfmb*fmc2*hbcm3+fmb*fmc*hbcm3-fmb*
     . fmc2*hbcm2-fmb2*fmc*hbcm2+fmc2*hbcm3)+16*(p4p7*p3p6)*(-ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-
     . fmb*fmc*hbcm+fmc*hbcm2)+8*(p6p7*p3p4)*(-ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3+fmb*hbcm2-fmb2*hbcm)+16*(p3p7*
     . p3p6*p3p4)*(ffmcfmb**2*hbcm+ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+16*(
     . ffmcfmb*hbcm3*p2p4*p6p7+fmb*p3p4*p3p7*p4p6+fmc*hbcm2*p2p5*p6p7
     . +hbcm*p2p5*p3p4*p6p7))+ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(1)=ans
      ans2=w1*(16*(p4p5*p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p5p7*p3p4)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8
     . *p5p7*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+8*(p3p7*p3p4
     . )*(ffmcfmb*fmb-ffmcfmb*fmc-fmb+fmc)+8*p3p7*(2*ffmcfmb*fmb*fmc*
     . hbcm-fmb*fmc*hbcm)+8*p4p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(
     . ffmcfmb*hbcm*p2p3*p4p7-ffmcfmb*hbcm*p2p4*p3p7))+(4*p3p7*(-4*
     . fb2*ffmcfmb*fmb+4*fb2*ffmcfmb*fmc+4*fb2*fmb-4*fb2*fmc-fmb+fmc)
     . +4*p5p7*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*p4p7*
     . (-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm))
      ans1=w8*(16*(p3p7*p2p5)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p2p4)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p5p7*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(
     . p4p7*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p3p7*p2p3)*(ffmcfmb*fmb-
     . ffmcfmb*fmc-fmb+fmc))+w18*(8*p3p7*(fmb*fmc*hbcm-fmb2*hbcm)+8*(
     . p5p7*p3p5)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*p3p5)*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*(p5p7*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*
     . p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p3p7*p3p5)*(ffmcfmb*fmb-
     . ffmcfmb*fmc-fmb+fmc)+8*(p3p7*p3p4)*(ffmcfmb*fmb-ffmcfmb*fmc-
     . fmb+fmc)+8*p5p7*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+8*p4p7*(
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3+fmb*fmc*hbcm+2*fmb*hbcm2-2*fmb2*hbcm-fmc*hbcm2)+16*(
     . ffmcfmb*hbcm*p2p3*p4p7-ffmcfmb*hbcm*p2p3*p5p7-ffmcfmb*hbcm*
     . p2p4*p3p7+ffmcfmb*hbcm*p2p5*p3p7-2*hbcm*p2p4*p5p7+2*hbcm*p2p5*
     . p4p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w18*(8*(p5p6*p3p5)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p6*p3p5)*(
     . -ffmcfmb*hbcm-fmb+hbcm)+8*(p5p6*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+
     . 8*(p4p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*p5p6*(ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc*hbcm2)+8*p4p6*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm-2*fmb*hbcm2
     . +2*fmb2*hbcm+fmc*hbcm2)+8*p3p6*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**
     . 2*fmc*hbcm2-ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmb2*hbcm+ffmcfmb*fmc*hbcm2)+16*(-ffmcfmb*hbcm*p2p3*p4p6+
     . ffmcfmb*hbcm*p2p3*p5p6+ffmcfmb*hbcm*p2p4*p3p6-ffmcfmb*hbcm*
     . p2p5*p3p6+2*hbcm*p2p4*p5p6-2*hbcm*p2p5*p4p6))+(4*p5p6*(-4*fb1*
     . ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+4*p4p6*(4*fb1*ffmcfmb*
     . hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm))
      ans1=w8*(8*(p5p6*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p6*p2p3)*(-
     . ffmcfmb*hbcm-fmb+hbcm))+w1*(8*(p5p6*p3p4)*(ffmcfmb*hbcm+fmb-
     . hbcm)+8*(p4p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*p5p6*(ffmcfmb*
     . fmc*hbcm2+fmb*fmc*hbcm-fmc*hbcm2)+8*p4p6*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2)+8*p3p6*(-ffmcfmb**2*fmc*hbcm2-ffmcfmb*fmb*fmc*hbcm
     . +ffmcfmb*fmc*hbcm2)+16*(-ffmcfmb*hbcm*p2p3*p4p6+ffmcfmb*hbcm*
     . p2p4*p3p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans3=((p5p6*p3p7)*(-32*fb2*ffmcfmb+32*fb2-8)+(p3p7*p3p6)*(-32*
     . fb2*ffmcfmb**2+32*fb2*ffmcfmb-16*ffmcfmb+8)+4*p6p7*(-4*fb1*
     . ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmb*hbcm+4*fb1*
     . fmc*hbcm-4*fb2*fmb*fmc-4*fb2*fmb2+2*ffmcfmb*hbcm2+fmb*hbcm-fmc
     . *hbcm-2*hbcm2)+32*(-fb2*p2p4*p6p7+fb2*p2p5*p6p7))
      ans2=w18*((p5p6*p3p7*p3p5)*(16*ffmcfmb-16)+(p5p6*p3p7*p3p4)*(16
     . *ffmcfmb-16)+16*(p5p6*p4p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16
     . *(p5p7*p4p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p2p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2+2*fmb*
     . hbcm+fmc*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*
     . hbcm+fmb*fmc-fmb*hbcm+fmb2-fmc*hbcm)+8*(p6p7*p3p4)*(ffmcfmb*
     . fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-fmb*hbcm+fmb2-fmc*hbcm)+16*(
     . p6p7*p2p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+16*(p3p7*p3p6*
     . p3p5)*(ffmcfmb**2-ffmcfmb)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2-
     . ffmcfmb)+16*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . ffmcfmb*hbcm2)+8*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*
     . hbcm3-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb2*
     . fmc*hbcm+fmb2*hbcm2-fmb3*hbcm)+16*(fmb*hbcm*p3p7*p4p6+p2p4*
     . p3p4*p6p7+p2p4*p3p5*p6p7-p2p5*p3p4*p6p7-p2p5*p3p5*p6p7))+ans3
      ans1=w8*((p5p6*p3p7*p2p3)*(16*ffmcfmb-16)+16*(p6p7*p2p5)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*
     . hbcm+fmb*fmc-fmb*hbcm+fmb2-fmc*hbcm)+16*(p3p7*p3p6*p2p3)*(
     . ffmcfmb**2-ffmcfmb)+16*(p2p3*p2p4*p6p7-p2p3*p2p5*p6p7))+w1*((
     . p5p6*p3p7*p3p4)*(16*ffmcfmb-16)+16*(p6p7*p4p5)*(-ffmcfmb*hbcm2
     . -fmb*hbcm+hbcm2)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p5p6*p3p7)*(ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(p6p7*p3p4)*(ffmcfmb*
     . fmb*hbcm+ffmcfmb*fmc*hbcm+fmb*fmc-fmb*hbcm+fmb2-fmc*hbcm)+16*(
     . p3p7*p3p6*p3p4)*(ffmcfmb**2-ffmcfmb)+16*(p4p7*p3p6)*(-ffmcfmb
     . **2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+8*p6p7*(ffmcfmb**2*
     . fmc*hbcm3+2*ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmc*hbcm3-fmb*fmc*
     . hbcm2+fmb2*fmc*hbcm)+16*(ffmcfmb*fmb*hbcm*p3p7*p4p6+ffmcfmb*
     . fmc*hbcm*p2p3*p6p7-fmc*hbcm*p2p5*p6p7+p2p4*p3p4*p6p7-p2p5*p3p4
     . *p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans3=((p3p7*p3p5)*(32*fb2*ffmcfmb-32*fb2+8)+4*p5p7*(-4*fb1*fmb*
     . hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+4*p3p7*(4*fb1*
     . ffmcfmb*fmb*hbcm+4*fb1*ffmcfmb*fmc*hbcm-4*fb1*fmb*hbcm-4*fb1*
     . fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2+4*fb2*fmb*
     . fmc-fmb*hbcm+fmc*hbcm+hbcm2)+32*(-fb2*p2p3*p4p7+fb2*p2p4*p3p7)
     . )
      ans2=w18*((p3p7*p3p5*p3p4)*(-16*ffmcfmb+16)+8*(p5p7*p3p5)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p3p5)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p5p7*p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+
     . 16*(p3p7*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p7*p2p4)*(-
     . ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+16*(p5p7*p2p3)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm+fmc*
     . hbcm)+8*p4p7*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*
     . hbcm2)+8*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm+fmc*hbcm)+8*(
     . p3p7*p3p4)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2-fmb*fmc-fmb*hbcm+fmc*hbcm)+8*p3p7*(-ffmcfmb
     . **2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*hbcm3+ffmcfmb*
     . fmc*hbcm3-2*ffmcfmb*p3p5s2-fmb*fmc*hbcm2-fmb*fmc2*hbcm+fmb2*
     . fmc*hbcm-fmb2*hbcm2+2*p3p5s2)+16*(-fmb*hbcm*p3p7*p4p5-hbcm2*
     . p2p4*p5p7+hbcm2*p2p5*p4p7+p2p3*p3p4*p4p7+p2p3*p3p5*p4p7-p2p4*
     . p3p4*p3p7-p2p4*p3p5*p3p7))+ans3
      ans1=w8*((p3p7*p3p5*p2p3)*(-16*ffmcfmb+16)+16*(p3p7*p2p5)*(
     . ffmcfmb*hbcm2-hbcm2)+16*(p3p7*p2p4)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+8*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p7*
     . p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p3p7*p2p3)*(-ffmcfmb**
     . 2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*
     . fmc-fmb*hbcm-fmc*hbcm)+16*(-p2p3*p2p4*p3p7+p2p3s2*p4p7))+w1*((
     . p3p7*p3p5*p3p4)*(-16*ffmcfmb+16)+16*(p4p5*p3p7)*(ffmcfmb*hbcm2
     . -hbcm2)+8*(p5p7*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p3p7*
     . p2p4)*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2+
     . fmc*hbcm)+8*p5p7*(ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmc*hbcm3)+
     . 16*(p3p7*p3p5)*(-ffmcfmb*fmc*hbcm+fmc*hbcm)+8*p4p7*(ffmcfmb**2
     . *hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4)+8*(p3p7*p3p4)*(-
     . ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-fmb*fmc+fmb*hbcm+fmc*hbcm)+8*p3p7*(-ffmcfmb**2*fmc*hbcm3
     . -2*ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb*
     . fmc2*hbcm)+16*(p2p3*p3p4*p4p7-p2p4*p3p4*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans3=((p3p6*p2p3)*(32*fb2*ffmcfmb+8)+4*p5p6*(4*fb1*fmb*hbcm+4*
     . fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2)+4*p3p6*(8*fb1*ffmcfmb*fmb
     . *hbcm+4*fb1*ffmcfmb*fmc*hbcm-4*fb1*fmb*hbcm-4*fb1*fmc*hbcm+4*
     . fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2+4*fb2*fmb*fmc+4*fb2*
     . fmb2-2*ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+32*(fb2*p2p3*p4p6+fb2*
     . p2p4*p3p6-fb2*p2p5*p3p6))
      ans2=w18*(16*(p3p6*p2p4)*(-2*fmb*hbcm-fmc*hbcm)+8*(p5p6*p3p5)*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p6*p3p5)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+8*(p5p6*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+
     . 16*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p6*p2p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-fmb*
     . hbcm-fmc*hbcm)+32*(p3p6*p2p3)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm)+8*p4p6*(-ffmcfmb**2*hbcm4+ffmcfmb*hbcm4-fmb*hbcm3+fmb2*
     . hbcm2)+8*(p3p6*p3p5)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm)
     . +8*(p3p6*p3p4)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm)+8*p3p6*(-ffmcfmb**2*fmb*
     . hbcm3-ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3
     . -fmb*fmc*hbcm2+fmb2*fmc*hbcm-fmb2*hbcm2+fmb3*hbcm)+16*(-
     . ffmcfmb*p2p3*p3p4*p3p6-ffmcfmb*p2p3*p3p5*p3p6+hbcm2*p2p4*p5p6-
     . hbcm2*p2p5*p4p6-p2p3*p3p4*p4p6-p2p3*p3p5*p4p6-p2p4*p3p4*p3p6-
     . p2p4*p3p5*p3p6+p2p5*p3p4*p3p6+p2p5*p3p5*p3p6))+ans3
      ans1=w8*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p3p6
     . *p2p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p6*p2p3)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+8*(p3p6*p2p3)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm)
     . +16*(-ffmcfmb*p2p3s2*p3p6-p2p3*p2p4*p3p6+p2p3*p2p5*p3p6-p2p3s2
     . *p4p6))+w1*(16*(p4p5*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p5p6*p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p6*p2p3)*(-
     . ffmcfmb*hbcm2-fmc*hbcm)+8*p5p6*(-ffmcfmb*fmc*hbcm3-fmb*fmc*
     . hbcm2+fmc*hbcm3)+8*p4p6*(-ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3+
     . ffmcfmb*hbcm4)+8*(p3p6*p3p4)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm)+8*p3p6*(-
     . ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc*hbcm3
     . +fmb*fmc*hbcm2-fmb2*fmc*hbcm)+16*(-2*ffmcfmb*fmc*hbcm*p2p3*
     . p3p6-ffmcfmb*p2p3*p3p4*p3p6-fmc*hbcm*p2p4*p3p6+fmc*hbcm*p2p5*
     . p3p6-p2p3*p3p4*p4p6-p2p4*p3p4*p3p6+p2p5*p3p4*p3p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w18*(8*p4p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*p2p5*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*p2p4*(ffmcfmb*hbcm2+3*fmb*hbcm+2*fmc
     . *hbcm)+16*p2p3*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+4*p3p5*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-ffmcfmb
     . *hbcm2+2*fmb*fmc-fmb*hbcm+fmb2-2*fmc*hbcm)+4*p3p4*(ffmcfmb**2*
     . hbcm2+2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+2*
     . fmb*fmc-fmb*hbcm+fmb2-2*fmc*hbcm)+4*(2*ffmcfmb**2*fmb*hbcm3+2*
     . ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-2*
     . ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2+2*ffmcfmb*p2p3*p3p4+2*
     . ffmcfmb*p2p3*p3p5+2*fmb*fmc*hbcm2+fmb*fmc2*hbcm-2*fmb2*fmc*
     . hbcm+fmb2*hbcm2-fmb3*hbcm-fmc2*hbcm2+4*p2p4*p3p4+4*p2p4*p3p5-2
     . *p2p5*p3p4-2*p2p5*p3p5))+(p2p3*(-16*fb2*ffmcfmb-4)+2*(-8*fb1*
     . ffmcfmb*fmb*hbcm-8*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmb*hbcm+8*fb1*
     . fmc*hbcm-4*fb2*ffmcfmb**2*hbcm2+4*fb2*ffmcfmb*hbcm2-8*fb2*fmb*
     . fmc-4*fb2*fmb2-16*fb2*p2p4+8*fb2*p2p5+2*ffmcfmb*hbcm2-2*fmc*
     . hbcm-hbcm2))
      ans1=w8*(8*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p2p4*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p2p3*(ffmcfmb**2*hbcm2+2*
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+2*fmb*fmc-
     . fmb*hbcm+fmb2-2*fmc*hbcm)+8*(ffmcfmb*p2p3s2+2*p2p3*p2p4-p2p3*
     . p2p5))+w1*(8*p4p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p2p4*(
     . ffmcfmb*hbcm2+2*fmc*hbcm)+4*p3p4*(ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm+2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+2*fmb*fmc-fmb*hbcm+
     . fmb2-2*fmc*hbcm)+4*(2*ffmcfmb**2*fmc*hbcm3+3*ffmcfmb*fmb*fmc*
     . hbcm2+4*ffmcfmb*fmc*hbcm*p2p3-2*ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2
     . *hbcm2+2*ffmcfmb*p2p3*p3p4-fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmb2*
     . fmc*hbcm-2*fmc*hbcm*p2p5-fmc2*hbcm2+4*p2p4*p3p4-2*p2p5*p3p4))+
     . ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=(4*p3p5*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*
     . p3p4*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+4*p2p3*(-
     . 4*fb1*ffmcfmb*hbcm-4*fb2*fmc+3*hbcm)+2*(4*fb1*fmb*fmc*hbcm-4*
     . fb1*fmb2*hbcm-8*fb1*hbcm*p2p4+8*fb1*hbcm*p2p5-4*fb2*ffmcfmb*
     . fmb*hbcm2+4*fb2*ffmcfmb*fmc*hbcm2+4*fb2*fmb*hbcm2-4*fb2*fmc*
     . hbcm2+fmb*hbcm2-fmc*hbcm2))
      ans2=w18*(8*p2p5*(ffmcfmb*hbcm3+fmc*hbcm2)+8*p2p4*(-ffmcfmb*
     . hbcm3-2*fmb*hbcm2+fmc*hbcm2)+8*(p3p5*p2p3)*(-ffmcfmb*hbcm+fmc)
     . +8*(p3p4*p2p3)*(3*ffmcfmb*hbcm+fmc)+8*p2p3*(-2*ffmcfmb*fmb*
     . hbcm2+2*ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmc2*hbcm)+4*p3p5*(-2*
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*ffmcfmb
     . *hbcm3+fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-fmc*hbcm2)+4*p3p4*(2*
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb
     . *hbcm3+fmb*fmc*hbcm+3*fmb*hbcm2-3*fmb2*hbcm-fmc*hbcm2)+4*(-
     . ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*fmc*hbcm4+ffmcfmb*fmb*hbcm4-
     . ffmcfmb*fmc*hbcm4+2*ffmcfmb*hbcm*p3p4s2-2*ffmcfmb*hbcm*p3p5s2+
     . fmb*fmc*hbcm3+2*fmb*p3p4s2-2*fmb*p3p5s2-fmb2*fmc*hbcm2-fmb2*
     . hbcm3+fmb3*hbcm2+4*hbcm*p2p3*p4p5+2*hbcm*p2p4*p3p4-6*hbcm*p2p4
     . *p3p5+6*hbcm*p2p5*p3p4-2*hbcm*p2p5*p3p5-2*hbcm*p3p4s2+2*hbcm*
     . p3p5s2))+ans3
      ans1=w8*(8*p2p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*p2p4*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p3p5*p2p3)*(-ffmcfmb*hbcm-
     . fmb+hbcm)+8*(p3p4*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+4*p2p3*(-2*
     . ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-3*fmc*hbcm2)+
     . 8*(-ffmcfmb*hbcm*p2p3s2+fmc*p2p3s2-hbcm*p2p3*p2p4+hbcm*p2p3*
     . p2p5))+w1*(8*p4p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*p2p4*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p5*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm
     . )+8*(p3p4*p2p3)*(3*ffmcfmb*hbcm+fmc)+8*p3p5*(-ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+8*p2p3*(2*ffmcfmb*fmc*hbcm2+fmc2
     . *hbcm)+4*p3p4*(2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm-fmc
     . *hbcm2)+4*(ffmcfmb**2*fmc*hbcm4+2*ffmcfmb*fmb*fmc*hbcm3-
     . ffmcfmb*fmc*hbcm4+2*ffmcfmb*hbcm*p3p4s2-fmb*fmc*hbcm3+2*fmb*
     . p3p4s2+fmb2*fmc*hbcm2-2*fmc*hbcm2*p2p5+2*hbcm*p2p4*p3p4-2*hbcm
     . *p2p5*p3p4-2*hbcm*p3p4s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+2*fmc)+8*p3p6*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm
     . +fmc2*hbcm)+8*(fmc*hbcm2*p5p6-hbcm*p3p4*p4p6+hbcm*p3p4*p5p6-2*
     . hbcm*p3p6*p4p5))+w18*(8*p5p6*(-ffmcfmb*hbcm3-fmc*hbcm2)+8*p4p6
     . *(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2)+8*(p3p6*p3p5)*(3*
     . ffmcfmb*hbcm-fmb+2*fmc)+8*(p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+2*fmc
     . )+8*p3p6*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-
     . fmb2*hbcm+fmc2*hbcm)+8*(-hbcm*p3p4*p4p6-3*hbcm*p3p4*p5p6+3*
     . hbcm*p3p5*p4p6+hbcm*p3p5*p5p6+2*hbcm*p3p6*p4p5))+w8*(8*p5p6*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*p4p6*(ffmcfmb*hbcm3+fmb*hbcm2
     . -hbcm3)+8*(p3p6*p2p3)*(-ffmcfmb*hbcm-fmb+2*fmc)+8*p3p6*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-2*fmc*hbcm2)+
     . 8*(hbcm*p2p3*p4p6-hbcm*p2p3*p5p6))+(4*p3p6*(-4*fb1*ffmcfmb*
     . hbcm+4*fb2*fmb-8*fb2*fmc+3*hbcm)+16*(fb1*hbcm*p4p6-fb1*hbcm*
     . p5p6)))
      b(9)=ans
      b(10)=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm2+2*fmc*hbcm)+8*(2*ffmcfmb*
     . fmc*hbcm*p3p6+ffmcfmb*p3p4*p3p6-fmc*hbcm*p5p6+2*p3p4*p4p6-p3p4
     . *p5p6))+w18*(16*p5p6*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb
     . *hbcm2+3*fmb*hbcm+2*fmc*hbcm)+16*p3p6*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm)+8*(ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6+2*
     . p3p4*p4p6-p3p4*p5p6+2*p3p5*p4p6-p3p5*p5p6))+w8*(8*p5p6*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2
     . )+8*(ffmcfmb*p2p3*p3p6+2*p2p3*p4p6-p2p3*p5p6))+(p3p6*(-16*fb2*
     . ffmcfmb-4)+16*(-2*fb2*p4p6+fb2*p5p6)))
      ans=ccc*(w1*(4*p3p4*(-3*ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+4*(-
     . 2*ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+2*fmc*hbcm*p3p5+fmc2*hbcm2-2
     . *hbcm2*p4p5+2*p3p4*p3p5-4*p3p4s2))+w18*(4*p3p5*(3*ffmcfmb*
     . hbcm2+fmb*hbcm-2*fmc*hbcm)+4*p3p4*(-3*ffmcfmb*hbcm2-5*fmb*hbcm
     . -2*fmc*hbcm)+4*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3-2*fmb
     . *fmc*hbcm2-fmb2*hbcm2+fmc2*hbcm2+2*hbcm2*p4p5-2*p3p4*p3p5-4*
     . p3p4s2+2*p3p5s2))+w8*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16
     . *p3p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p3*(-ffmcfmb*hbcm2-
     . fmb*hbcm-2*fmc*hbcm)+4*(-ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3-
     . 2*ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-2*fmb*fmc*hbcm2+fmb*hbcm3-
     . fmb2*hbcm2+2*fmc*hbcm3-4*p2p3*p3p4+2*p2p3*p3p5))+2*(-4*fb1*fmb
     . *hbcm-8*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2+16*fb2*p3p4-8*fb2*
     . p3p5+3*hbcm2))
      b(11)=ans
      ans=ccc*(w1*(8*p4p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p3p4)*(-3
     . *ffmcfmb*hbcm-fmc)+8*p3p7*(-2*ffmcfmb*fmc*hbcm2-fmc2*hbcm)+8*(
     . fmc*hbcm2*p5p7-hbcm*p3p4*p4p7+hbcm*p3p4*p5p7))+w18*(8*p5p7*(-
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*p4p7*(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc
     . *hbcm2)+8*(p3p7*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p3p7*p3p4)*(-3*
     . ffmcfmb*hbcm-fmc)+8*p3p7*(2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm-fmc2*hbcm)+8*(-hbcm*p3p4*p4p7-3*hbcm*p3p4*
     . p5p7+3*hbcm*p3p5*p4p7+hbcm*p3p5*p5p7-2*hbcm*p3p7*p4p5))+w8*(8*
     . p5p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*p4p7*(ffmcfmb*hbcm3+
     . fmb*hbcm2-hbcm3)+16*(p3p7*p3p5)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*
     . p3p4)*(-ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p3)*(ffmcfmb*hbcm-fmc)+8*
     . p3p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2
     . -ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2+2*fmc*hbcm2)+8*(hbcm*
     . p2p3*p4p7-hbcm*p2p3*p5p7))+(4*p3p7*(4*fb1*ffmcfmb*hbcm+4*fb2*
     . fmc-3*hbcm)+16*(fb1*hbcm*p4p7-fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w1*(4*p3p4*(-2*ffmcfmb*hbcm+fmb-3*fmc)+4*(-3*ffmcfmb
     . *fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm+2*hbcm*p4p5))+w18*(4*p3p5*
     . (-2*ffmcfmb*hbcm+fmb-3*fmc)+4*p3p4*(-2*ffmcfmb*hbcm+fmb-3*fmc)
     . +4*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-3*fmb*fmc*hbcm+
     . fmb2*hbcm-2*fmc2*hbcm-4*hbcm*p4p5))+w8*(4*p2p3*(2*ffmcfmb*hbcm
     . +fmb-3*fmc)+4*(2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2-3*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+3*fmc*hbcm2))+4*(4*fb1*ffmcfmb*hbcm-2*fb2*fmb+6*fb2*
     . fmc-3*hbcm))
      b(14)=ccc*(w1*(8*p4p7*(-ffmcfmb*hbcm2-2*fmc*hbcm)+8*(-2*ffmcfmb
     . *fmc*hbcm*p3p7-ffmcfmb*p3p4*p3p7+fmc*hbcm*p5p7-2*p3p4*p4p7+
     . p3p4*p5p7))+w8*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*p4p7*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p3p7*(-2*ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm+fmb*hbcm+2*fmc*hbcm)+8*(-ffmcfmb*p2p3*p3p7-2*
     . p2p3*p4p7+p2p3*p5p7))+w18*(16*p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*
     . p4p7*(-ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm)+16*p3p7*(-ffmcfmb*
     . fmb*hbcm-ffmcfmb*fmc*hbcm)+8*(-ffmcfmb*p3p4*p3p7-ffmcfmb*p3p5*
     . p3p7-2*p3p4*p4p7+p3p4*p5p7-2*p3p5*p4p7+p3p5*p5p7))+(p3p7*(16*
     . fb2*ffmcfmb+4)+16*(2*fb2*p4p7-fb2*p5p7)))
      ans2=w18*(16*(p4p7*p3p6)*(2*fmb*hbcm+fmc*hbcm)+16*(p5p6*p3p7)*(
     . -ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*
     . hbcm+fmc*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p6p7*p3p5)*(2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+8*(p6p7*p3p4)*(
     . -2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm)+8*p6p7*(-ffmcfmb*fmb*
     . hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2-fmb2*hbcm2-2*p3p4s2+2*
     . p3p5s2)+32*(p3p7*p3p6)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+16*
     . (ffmcfmb*p3p4*p3p6*p3p7+ffmcfmb*p3p5*p3p6*p3p7+hbcm2*p4p6*p5p7
     . -hbcm2*p4p7*p5p6+p3p4*p3p6*p4p7-p3p4*p3p6*p5p7+p3p4*p3p7*p4p6+
     . p3p5*p3p6*p4p7-p3p5*p3p6*p5p7+p3p5*p3p7*p4p6))+((p3p7*p3p6)*(-
     . 32*fb2*ffmcfmb-8)+8*p6p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+hbcm2
     . )+32*(fb2*p3p4*p6p7-fb2*p3p5*p6p7-fb2*p3p6*p4p7+fb2*p3p6*p5p7-
     . fb2*p3p7*p4p6))
      ans1=w1*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p4)*
     . (-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+8*p6p7*(-ffmcfmb*fmc*
     . hbcm3+fmb*fmc*hbcm2-2*p3p4s2)+16*(2*ffmcfmb*fmc*hbcm*p3p6*p3p7
     . +ffmcfmb*p3p4*p3p6*p3p7+fmc*hbcm*p3p5*p6p7+fmc*hbcm*p3p6*p4p7-
     . fmc*hbcm*p3p6*p5p7-hbcm2*p4p5*p6p7+p3p4*p3p5*p6p7+p3p4*p3p6*
     . p4p7-p3p4*p3p6*p5p7+p3p4*p3p7*p4p6))+w8*(8*(p6p7*p2p3)*(-fmb*
     . hbcm-fmc*hbcm)+16*(p5p6*p3p7)*(-ffmcfmb*hbcm2+hbcm2)+16*(p4p6*
     . p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p4p7*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+16*(p6p7*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*
     . p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p6p7*(-ffmcfmb*fmb*
     . hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmb*hbcm3-fmb2*hbcm2+fmc
     . *hbcm3)+16*(p3p7*p3p6)*(2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-
     . fmb*hbcm-fmc*hbcm)+16*(ffmcfmb*p2p3*p3p6*p3p7-p2p3*p3p4*p6p7+
     . p2p3*p3p5*p6p7+p2p3*p3p6*p4p7-p2p3*p3p6*p5p7+p2p3*p3p7*p4p6))+
     . ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w1*(8*(p6p7*p3p4)*(ffmcfmb*hbcm-fmb+2*fmc)+8*p6p7*(2*
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm)+16*(-ffmcfmb*hbcm*
     . p3p6*p4p7+ffmcfmb*hbcm*p3p7*p4p6-hbcm*p4p5*p6p7))+w18*(8*(p6p7
     . *p3p5)*(ffmcfmb*hbcm-fmb+2*fmc)+8*(p6p7*p3p4)*(ffmcfmb*hbcm-
     . fmb+2*fmc)+8*p6p7*(-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+2*
     . fmb*fmc*hbcm-fmb2*hbcm+fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p4p7+
     . ffmcfmb*hbcm*p3p6*p5p7+ffmcfmb*hbcm*p3p7*p4p6-ffmcfmb*hbcm*
     . p3p7*p5p6+hbcm*p4p5*p6p7+2*hbcm*p4p6*p5p7-2*hbcm*p4p7*p5p6))+
     . w8*(16*(p5p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm-fmb+2*fmc)+8*
     . p6p7*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+2*fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-2*fmc*
     . hbcm2))+4*p6p7*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmb-8*fb2*fmc+3*hbcm
     . ))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp17_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      ans4=(32*(p6p7*p2p3)*(-fb2*fmb+fb2*fmc-hbcm)+8*(p6p7*p1p3)*(4*
     . fb1*ffmcfmb*hbcm-hbcm)+8*p6p7*(-2*fb1*ffmcfmb**2*hbcm3+2*fb1*
     . fmc2*hbcm-ffmcfmb*hbcm3+fmb*hbcm2+2*hbcm3)+8*(p3p7*p3p6)*(4*
     . fb1*ffmcfmb**2*hbcm-4*fb2*ffmcfmb*fmc+4*ffmcfmb*hbcm-fmc+4*
     . hbcm)+32*(-fb1*hbcm*p2p5*p6p7+hbcm*p3p5*p6p7+hbcm*p3p6*p5p7+
     . hbcm*p3p7*p4p6))
      ans3=w12*(16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(p6p7*
     . p2p3*p1p3)*(-2*ffmcfmb*hbcm-fmb+fmc)+8*p6p7*(-ffmcfmb**2*hbcm5
     . -2*ffmcfmb*hbcm*p1p3s2+fmc2*hbcm3)+16*(p4p6*p3p7)*(-ffmcfmb**2
     . *hbcm3+fmc2*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3+fmc2*hbcm)
     . +16*(p6p7*p3p5)*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p6p7*p2p3)*(
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*
     . hbcm)+8*(p6p7*p1p3)*(ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-fmc2*
     . hbcm)+16*(p3p7*p3p6*p1p3)*(ffmcfmb**2*hbcm-ffmcfmb*fmc+2*
     . ffmcfmb*hbcm)+16*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2
     . *fmc*hbcm2-ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(2*ffmcfmb*hbcm*p1p3
     . *p3p5*p6p7+2*ffmcfmb*hbcm*p1p3*p3p6*p5p7+2*ffmcfmb*hbcm*p1p3*
     . p3p7*p4p6+2*ffmcfmb*hbcm*p2p3*p3p6*p5p7-2*ffmcfmb*hbcm*p2p5*
     . p3p6*p3p7-hbcm*p1p3*p2p5*p6p7))+ans4
      ans2=w17*(16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(p6p7*
     . p2p3*p1p3)*(-ffmcfmb*hbcm-fmb+fmc)+8*p6p7*(-ffmcfmb**2*hbcm5-2
     . *ffmcfmb*hbcm*p1p3s2-2*fmb*p2p3s2+2*fmc*p2p3s2+fmc2*hbcm3)+16*
     . (p4p6*p3p7)*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(p6p7*p3p5)*(-ffmcfmb**2*hbcm3+
     . fmc2*hbcm)+8*(p6p7*p2p3)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2
     . -2*ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm+fmc2*hbcm)+8*(p6p7*p1p3)*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-fmc2*hbcm)+16*(p3p7*p3p6*p2p3
     . )*(ffmcfmb**2*hbcm-ffmcfmb*fmc+2*ffmcfmb*hbcm)+16*(p3p7*p3p6*
     . p1p3)*(ffmcfmb**2*hbcm-ffmcfmb*fmc+2*ffmcfmb*hbcm)+16*(p3p7*
     . p3p6)*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*fmc*hbcm2-ffmcfmb**2*
     . hbcm3+fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*p1p2*p3p6*p3p7+2*ffmcfmb*
     . hbcm*p1p3*p3p5*p6p7+2*ffmcfmb*hbcm*p1p3*p3p6*p5p7+2*ffmcfmb*
     . hbcm*p1p3*p3p7*p4p6+2*ffmcfmb*hbcm*p2p3*p3p6*p5p7+2*ffmcfmb*
     . hbcm*p2p3*p3p7*p4p6-hbcm*p1p3*p2p5*p6p7-hbcm*p2p3*p2p5*p6p7))+
     . ans3
      ans1=w2*(16*(p6p7*p3p5*p2p3)*(-fmb+fmc)+16*(p6p7*p2p5)*(ffmcfmb
     . *hbcm3+fmb*hbcm2-hbcm3)+32*(p3p7*p3p6*p2p5)*(-ffmcfmb*hbcm-
     . hbcm)+32*(p5p7*p3p6*p2p3)*(ffmcfmb*hbcm+hbcm)+16*(p5p6*p3p7)*(
     . ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3-fmc2*hbcm)+16*(p6p7*p2p3)*(
     . ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb2*hbcm)+16*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm3+fmc2*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb**2*hbcm3+
     . fmc2*hbcm)+16*(p3p7*p3p6*p3p5)*(ffmcfmb**2*hbcm-ffmcfmb*fmc+2*
     . ffmcfmb*hbcm)+16*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2-ffmcfmb*
     . fmb*fmc*hbcm+ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm)+16*(ffmcfmb*
     . fmb*hbcm2*p3p7*p4p6+ffmcfmb*hbcm*p1p3*p3p5*p6p7+2*ffmcfmb*hbcm
     . *p1p3*p3p6*p5p7-2*ffmcfmb*hbcm*p1p3*p3p7*p5p6-2*ffmcfmb*hbcm*
     . p1p5*p3p6*p3p7-2*ffmcfmb*hbcm*p2p3*p3p7*p5p6+2*ffmcfmb*hbcm*
     . p3p5*p3p7*p4p6+2*ffmcfmb*hbcm*p3p5*p3p7*p5p6-fmb*hbcm2*p1p2*
     . p6p7-2*hbcm*p1p2*p3p5*p6p7-2*hbcm*p1p2*p3p6*p5p7+2*hbcm*p1p2*
     . p3p7*p5p6+2*hbcm*p1p5*p2p3*p6p7+2*hbcm*p2p3*p2p5*p6p7+2*hbcm*
     . p2p3*p4p6*p5p7+2*hbcm*p2p3*p5p6*p5p7+hbcm*p2p5*p3p5*p6p7-2*
     . hbcm*p2p5*p3p7*p4p6-2*hbcm*p2p5*p3p7*p5p6))+ans2
      ans=ccc*ans1
      b(1)=ans
      ans=ccc*(w2*(8*(p3p7*p3p5)*(-ffmcfmb*fmb+ffmcfmb*fmc-2*ffmcfmb*
     . hbcm)+8*p5p7*(ffmcfmb**2*hbcm3-fmc2*hbcm)+8*p3p7*(ffmcfmb**2*
     . fmb*hbcm2+ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb*hbcm2-2*ffmcfmb*
     . fmb2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p5p7+ffmcfmb*hbcm*p1p5*p3p7+
     . hbcm*p1p2*p5p7-hbcm*p2p3*p5p7+hbcm*p2p5*p3p7))+w17*(8*(p3p7*
     . p2p3)*(-ffmcfmb*fmb+ffmcfmb*fmc-2*ffmcfmb*hbcm)+8*(p3p7*p1p3)*
     . (-ffmcfmb*fmb+ffmcfmb*fmc-2*ffmcfmb*hbcm)+16*p5p7*(ffmcfmb**2*
     . hbcm3-fmc2*hbcm)+8*p3p7*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*fmc*
     . hbcm2+ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc2*hbcm-
     . fmc2*hbcm)+16*(ffmcfmb*hbcm*p1p2*p3p7-2*ffmcfmb*hbcm*p1p3*p5p7
     . -2*ffmcfmb*hbcm*p2p5*p3p7))+w12*(8*(p3p7*p1p3)*(-ffmcfmb*fmb+
     . ffmcfmb*fmc-2*ffmcfmb*hbcm)+16*p5p7*(ffmcfmb**2*hbcm3-fmc2*
     . hbcm)+8*p3p7*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*fmc*hbcm2+
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc2*hbcm-fmc2*
     . hbcm)-32*ffmcfmb*hbcm*p1p3*p5p7)+(4*p3p7*(-4*fb2*ffmcfmb*fmb+4
     . *fb2*ffmcfmb*fmc-fmb+fmc-4*hbcm)-32*hbcm*p5p7))
      b(2)=ans
      b(3)=ccc*(w2*(8*p5p6*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+16*(ffmcfmb*
     . hbcm*p1p3*p5p6+ffmcfmb*hbcm*p2p5*p3p6-hbcm*p1p2*p5p6+hbcm*p2p3
     . *p5p6+hbcm*p2p5*p4p6+hbcm*p2p5*p5p6))+w17*(16*p4p6*(ffmcfmb**2
     . *hbcm3-fmc2*hbcm)+8*p3p6*(ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(-
     . ffmcfmb*hbcm*p1p2*p3p6-ffmcfmb*hbcm*p1p3*p3p6-2*ffmcfmb*hbcm*
     . p1p3*p4p6+ffmcfmb*hbcm*p2p3*p3p6+2*ffmcfmb*hbcm*p2p5*p3p6))+
     . w12*(16*p4p6*(ffmcfmb**2*hbcm3-fmc2*hbcm)+8*p3p6*(ffmcfmb**2*
     . hbcm3-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p2*p3p6-ffmcfmb*hbcm*p1p3
     . *p3p6-2*ffmcfmb*hbcm*p1p3*p4p6+ffmcfmb*hbcm*p2p3*p3p6+2*
     . ffmcfmb*hbcm*p2p5*p3p6))+16*(-hbcm*p3p6-2*hbcm*p4p6))
      ans2=w12*(32*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p2p3
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p2)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p6p7*p1p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+16*(
     . p3p7*p3p6*p1p3)*(-ffmcfmb**2+ffmcfmb)+16*(p4p6*p3p7)*(-ffmcfmb
     . **2*hbcm2+ffmcfmb*fmc*hbcm)+8*p6p7*(-ffmcfmb**2*fmb*hbcm3-
     . ffmcfmb**2*fmc*hbcm3+fmb*fmc2*hbcm+fmc3*hbcm)+16*(ffmcfmb*fmc*
     . hbcm*p3p6*p3p7+ffmcfmb*p1p3*p3p7*p4p6+p1p2*p1p3*p6p7-p1p3*p2p3
     . *p6p7-2*p1p3*p2p4*p6p7))+(8*p6p7*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc
     . *hbcm-hbcm2)+(p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(p3p7*p3p6)*(-32*
     . fb2*ffmcfmb**2+32*fb2*ffmcfmb-16*ffmcfmb+8)+32*(fb2*p1p2*p6p7-
     . fb2*p2p3*p6p7-2*fb2*p2p4*p6p7))
      ans1=w2*(16*(p6p7*p2p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(
     . p3p7*p3p6*p3p5)*(-ffmcfmb**2+ffmcfmb)+16*(p5p6*p3p7)*(ffmcfmb
     . **2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+8*p6p7*(-ffmcfmb**2*
     . fmb*hbcm3+fmb*fmc2*hbcm)+16*(ffmcfmb*fmb*hbcm*p1p3*p6p7+
     . ffmcfmb*fmb*hbcm*p3p6*p3p7+2*ffmcfmb*fmb*hbcm*p3p7*p4p6+
     . ffmcfmb*p3p5*p3p7*p4p6+fmb*hbcm*p1p2*p6p7-fmb*hbcm*p2p3*p6p7-3
     . *fmb*hbcm*p2p4*p6p7+p1p2*p3p5*p6p7-p2p3*p3p5*p6p7-2*p2p4*p3p5*
     . p6p7))+w17*(32*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*
     . p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p2)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p1p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+
     . 16*(p3p7*p3p6*p2p3)*(-ffmcfmb**2+ffmcfmb)+16*(p3p7*p3p6*p1p3)*
     . (-ffmcfmb**2+ffmcfmb)+16*(p4p6*p3p7)*(-ffmcfmb**2*hbcm2+
     . ffmcfmb*fmc*hbcm)+8*p6p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc
     . *hbcm3+fmb*fmc2*hbcm+fmc3*hbcm-2*p2p3s2)+16*(ffmcfmb*fmc*hbcm*
     . p3p6*p3p7+ffmcfmb*p1p3*p3p7*p4p6+ffmcfmb*p2p3*p3p7*p4p6+p1p2*
     . p1p3*p6p7+p1p2*p2p3*p6p7-p1p3*p2p3*p6p7-2*p1p3*p2p4*p6p7-2*
     . p2p3*p2p4*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans3=w12*(32*(p3p7*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+32*(p5p7*p2p3
     . )*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2+fmc
     . *hbcm)+8*p5p7*(ffmcfmb**2*hbcm4-fmc2*hbcm2)+16*(p3p7*p3p5)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+16*(p3p7*p2p3)*(ffmcfmb**2*
     . hbcm2-ffmcfmb*fmc*hbcm)+8*(p3p7*p1p3)*(ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*p3p7*(
     . ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-ffmcfmb**2*hbcm4+
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*p1p3s2-fmb*fmc2*hbcm-fmc3*hbcm)+16
     . *(-ffmcfmb*hbcm2*p1p3*p5p7-ffmcfmb*p1p3*p2p3*p3p7+ffmcfmb*p1p3
     . *p3p5*p3p7+p1p2*p1p3*p3p7+2*p1p3*p2p3*p5p7-2*p1p3*p2p5*p3p7))+
     . ((p3p7*p3p5)*(32*fb2*ffmcfmb+8)+(p3p7*p2p3)*(-32*fb2*ffmcfmb-8
     . )+(p3p7*p1p3)*(-32*fb2*ffmcfmb-8)+4*p3p7*(4*fb1*ffmcfmb*fmc*
     . hbcm+4*fb2*ffmcfmb**2*hbcm2+4*fb2*ffmcfmb*hbcm2-2*ffmcfmb*
     . hbcm2-4*fmb*hbcm-3*fmc*hbcm+hbcm2)+16*(2*fb2*p1p2*p3p7+4*fb2*
     . p2p3*p5p7-4*fb2*p2p5*p3p7-hbcm2*p5p7))
      ans2=w17*(16*(p3p7*p2p5)*(ffmcfmb*hbcm2-2*fmc*hbcm)+32*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p5p7*(ffmcfmb**2*hbcm4-fmc2*
     . hbcm2+4*p2p3s2)+16*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*
     . hbcm)+8*(p3p7*p2p3)*(ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-3*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*(p3p7*p1p3)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*
     . p3p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-ffmcfmb**2*
     . hbcm4+ffmcfmb*fmc*hbcm3-2*ffmcfmb*p1p3s2-2*ffmcfmb*p2p3s2-fmb*
     . fmc2*hbcm-fmc3*hbcm)+16*(-ffmcfmb*hbcm2*p1p3*p5p7-2*ffmcfmb*
     . p1p3*p2p3*p3p7+ffmcfmb*p1p3*p3p5*p3p7+ffmcfmb*p2p3*p3p5*p3p7+
     . fmc*hbcm*p1p2*p3p7+p1p2*p1p3*p3p7+p1p2*p2p3*p3p7+2*p1p3*p2p3*
     . p5p7-2*p1p3*p2p5*p3p7-2*p2p3*p2p5*p3p7))+ans3
      ans1=w2*(16*(p3p7*p2p5)*(ffmcfmb*hbcm2-2*fmb*hbcm)+16*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2+2*fmb*hbcm)+8*p5p7*(ffmcfmb**2*hbcm4-
     . fmc2*hbcm2)+8*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*p3p7*(ffmcfmb**2*fmb*
     . hbcm3-ffmcfmb*fmb*fmc*hbcm2+2*ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*
     . hbcm2+2*ffmcfmb*p3p5s2-fmb*fmc2*hbcm)+16*(-2*ffmcfmb*fmb*hbcm*
     . p1p3*p3p7-2*ffmcfmb*fmb*hbcm*p2p3*p3p7-ffmcfmb*hbcm2*p1p3*p5p7
     . +ffmcfmb*hbcm2*p1p5*p3p7-ffmcfmb*p1p3*p3p5*p3p7-ffmcfmb*p2p3*
     . p3p5*p3p7+2*fmb*hbcm*p1p2*p3p7+hbcm2*p1p2*p5p7+p1p2*p3p5*p3p7+
     . 2*p2p3*p3p5*p5p7-2*p2p5*p3p5*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans2=w12*(16*(p3p6*p2p5)*(2*ffmcfmb*hbcm2-fmc*hbcm)+32*(p4p6*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+16*(p3p6*p2p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+
     . 8*p4p6*(ffmcfmb**2*hbcm4-fmc2*hbcm2)+8*(p3p6*p1p3)*(ffmcfmb**2
     . *hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-fmc2)+8*p3p6*(
     . ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*p1p3s2-fmb
     . *fmc2*hbcm-fmc3*hbcm)+16*(-ffmcfmb*hbcm2*p1p3*p4p6+p1p2*p1p3*
     . p3p6+2*p1p3*p2p3*p4p6-p1p3*p2p5*p3p6))+((p3p6*p1p3)*(-32*fb2*
     . ffmcfmb-8)+8*p3p6*(2*fb2*ffmcfmb**2*hbcm2-2*fb2*fmc2-2*fmb*
     . hbcm-fmc*hbcm)+16*(2*fb2*p1p2*p3p6+4*fb2*p2p3*p4p6-2*fb2*p2p5*
     . p3p6-hbcm2*p4p6))
      ans1=w2*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmb*hbcm)+16*(p5p6*p2p3)
     . *(ffmcfmb*hbcm2+fmb*hbcm)+8*p5p6*(-ffmcfmb**2*hbcm4+fmc2*hbcm2
     . )+8*(p3p6*p3p5)*(ffmcfmb**2*hbcm2-fmc2)+8*p3p6*(ffmcfmb**2*fmb
     . *hbcm3-fmb*fmc2*hbcm)+16*(-ffmcfmb*fmb*hbcm*p1p3*p3p6+ffmcfmb*
     . fmb*hbcm*p2p3*p3p6+ffmcfmb*hbcm2*p1p3*p5p6-ffmcfmb*p1p3*p3p5*
     . p3p6+fmb*hbcm*p1p2*p3p6+3*fmb*hbcm*p2p3*p4p6-hbcm2*p1p2*p5p6+
     . hbcm2*p2p5*p4p6+hbcm2*p2p5*p5p6+p1p2*p3p5*p3p6+2*p2p3*p3p5*
     . p4p6-p2p5*p3p5*p3p6))+w17*(16*(p3p6*p2p5)*(2*ffmcfmb*hbcm2-fmc
     . *hbcm)+32*(p4p6*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p6*p1p2)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb**2*hbcm4-fmc2*hbcm2
     . +4*p2p3s2)+8*(p3p6*p2p3)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+
     . 2*ffmcfmb*fmc*hbcm-fmc2)+8*(p3p6*p1p3)*(ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-fmc2)+8*p3p6*(ffmcfmb**2*
     . fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*p1p3s2-fmb*fmc2*hbcm-
     . fmc3*hbcm)+16*(-ffmcfmb*hbcm2*p1p3*p4p6-ffmcfmb*p1p3*p2p3*p3p6
     . +p1p2*p1p3*p3p6+p1p2*p2p3*p3p6+2*p1p3*p2p3*p4p6-p1p3*p2p5*p3p6
     . -p2p3*p2p5*p3p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w12*(24*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+16*p2p3*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+16*p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+4*p1p3*(-
     . ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-2*
     . ffmcfmb*hbcm2+fmc2)+4*(-2*ffmcfmb**2*fmb*hbcm3-2*ffmcfmb**2*
     . fmc*hbcm3+ffmcfmb**2*hbcm4+2*ffmcfmb*p1p3s2+2*fmb*fmc2*hbcm-
     . fmc2*hbcm2+2*fmc3*hbcm-4*p1p2*p1p3+4*p1p3*p2p3+6*p1p3*p2p5))+(
     . p1p3*(16*fb2*ffmcfmb+4)+4*(-2*fb2*ffmcfmb**2*hbcm2+2*fb2*fmc2-
     . 8*fb2*p1p2+8*fb2*p2p3+12*fb2*p2p5+ffmcfmb*hbcm2+4*fmb*hbcm+3*
     . fmc*hbcm-2*hbcm2))
      ans1=w2*(8*p2p5*(-ffmcfmb*hbcm2+3*fmb*hbcm+hbcm2)+4*p3p5*(-
     . ffmcfmb**2*hbcm2+fmc2)+8*(-ffmcfmb**2*fmb*hbcm3+2*ffmcfmb*fmb*
     . hbcm*p1p3+ffmcfmb*p1p3*p3p5+fmb*fmc2*hbcm-3*fmb*hbcm*p1p2+3*
     . fmb*hbcm*p2p3-2*p1p2*p3p5+2*p2p3*p3p5+3*p2p5*p3p5))+w17*((p2p3
     . *p1p3)*(8*ffmcfmb+16)+24*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+16*
     . p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+4*p2p3*(-ffmcfmb**2*hbcm2-4*
     . ffmcfmb*hbcm2+4*fmc*hbcm+fmc2)+4*p1p3*(-ffmcfmb**2*hbcm2+4*
     . ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmc2)+4*(-
     . 2*ffmcfmb**2*fmb*hbcm3-2*ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4
     . +2*ffmcfmb*p1p3s2+2*fmb*fmc2*hbcm-fmc2*hbcm2+2*fmc3*hbcm-4*
     . p1p2*p1p3-4*p1p2*p2p3+6*p1p3*p2p5+6*p2p3*p2p5+4*p2p3s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans2=w12*(16*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+8*(p2p3*p1p3)*(3*ffmcfmb*hbcm+2*fmb-fmc)+16*
     . p3p5*(ffmcfmb**2*hbcm3-fmc2*hbcm)+16*p2p3*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm)+8*p1p3*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb
     . *hbcm3+fmc2*hbcm)+4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4
     . +2*ffmcfmb**2*hbcm5-8*ffmcfmb*hbcm*p1p3*p3p5+4*ffmcfmb*hbcm*
     . p1p3s2-fmb*fmc2*hbcm2-2*fmc2*hbcm3+fmc3*hbcm2-2*hbcm*p1p2*p1p3
     . +4*hbcm*p1p3*p2p5))+(4*p2p3*(4*fb1*ffmcfmb*hbcm+8*fb2*fmb-4*
     . fb2*fmc+7*hbcm)+8*(-2*fb1*hbcm*p1p2+4*fb1*hbcm*p2p5-fmb*hbcm2+
     . fmc*hbcm2+2*hbcm*p1p3-4*hbcm*p3p5-2*hbcm3))
      ans1=w2*(8*p2p5*(-ffmcfmb*hbcm3-2*fmb*hbcm2+hbcm3)+8*(p3p5*p2p3
     . )*(-ffmcfmb*hbcm+2*fmb-fmc)+8*p2p3*(-2*ffmcfmb*fmb*hbcm2-fmb*
     . fmc*hbcm+3*fmb2*hbcm)+8*p3p5*(ffmcfmb**2*hbcm3-fmc2*hbcm)+4*(
     . ffmcfmb**2*fmb*hbcm4-2*ffmcfmb*fmb*hbcm2*p1p3-4*ffmcfmb*hbcm*
     . p1p3*p3p5-fmb*fmc2*hbcm2+4*fmb*hbcm2*p1p2+6*hbcm*p1p2*p3p5-4*
     . hbcm*p1p5*p2p3-4*hbcm*p2p3*p2p5-4*hbcm*p2p5*p3p5))+w17*(16*
     . p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*hbcm3-fmc*
     . hbcm2)+8*(p2p3*p1p3)*(ffmcfmb*hbcm+2*fmb-fmc)+16*p3p5*(ffmcfmb
     . **2*hbcm3-fmc2*hbcm)+8*p2p3*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*
     . hbcm2+2*ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-fmc2*hbcm)+8*p1p3*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb
     . *hbcm3+fmc2*hbcm)+4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4
     . +2*ffmcfmb**2*hbcm5-8*ffmcfmb*hbcm*p1p3*p3p5+4*ffmcfmb*hbcm*
     . p1p3s2-2*ffmcfmb*hbcm*p2p3s2-fmb*fmc2*hbcm2+4*fmb*p2p3s2-2*fmc
     . *p2p3s2-2*fmc2*hbcm3+fmc3*hbcm2-2*hbcm*p1p2*p1p3+2*hbcm*p1p2*
     . p2p3+4*hbcm*p1p3*p2p5+4*hbcm*p2p3*p2p5))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w2*(8*p5p6*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p3p6*
     . p3p5)*(-3*ffmcfmb*hbcm+fmb)+8*p3p6*(-2*ffmcfmb*fmb*hbcm2+fmb2*
     . hbcm)+8*(-3*fmb*hbcm2*p4p6-4*hbcm*p3p5*p4p6-2*hbcm*p3p5*p5p6))
     . +w17*(16*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p6*p2p3)*(
     . ffmcfmb*hbcm+fmb)+8*(p3p6*p1p3)*(ffmcfmb*hbcm+fmb)+8*p3p6*(-2*
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*p3p5*p3p6+hbcm*p1p2*p3p6+
     . hbcm*p1p3*p4p6+hbcm*p2p3*p4p6))+w12*(16*p4p6*(-ffmcfmb*hbcm3+
     . fmc*hbcm2)+8*(p3p6*p1p3)*(3*ffmcfmb*hbcm+fmb)+8*p3p6*(-ffmcfmb
     . **2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*
     . hbcm3+fmb*fmc*hbcm)+16*(ffmcfmb*hbcm*p2p3*p3p6-2*ffmcfmb*hbcm*
     . p3p5*p3p6+hbcm*p1p3*p4p6))+(4*p3p6*(4*fb1*ffmcfmb*hbcm+4*fb2*
     . fmb+3*hbcm)+32*fb1*hbcm*p4p6))
      b(9)=ans
      b(10)=ccc*(w2*((p3p6*p3p5)*(-8*ffmcfmb-8)+8*p5p6*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+8*p3p6*(-2*ffmcfmb*fmb*hbcm-fmb*hbcm)+8*
     . (-4*fmb*hbcm*p4p6-3*p3p5*p4p6))+w17*((p3p6*p2p3)*(-8*ffmcfmb-8
     . )+(p3p6*p1p3)*(-8*ffmcfmb-8)+24*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+
     . 8*p3p6*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2
     . -fmc*hbcm)+24*(-p1p3*p4p6-p2p3*p4p6))+w12*((p3p6*p1p3)*(-8*
     . ffmcfmb-8)+24*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(-2*ffmcfmb
     . *fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-fmc*hbcm)-24*p1p3
     . *p4p6)+(p3p6*(-16*fb2*ffmcfmb-16*fb2-4)-48*fb2*p4p6))
      ans=ccc*(w2*(4*p3p5*(3*ffmcfmb*hbcm2-4*fmb*hbcm+fmc*hbcm-3*
     . hbcm2)+4*(2*ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2+2*fmb*hbcm*p1p3+2*
     . fmb*hbcm*p2p3-4*fmb*hbcm3+3*fmb2*hbcm2-2*hbcm2*p1p5-2*hbcm2*
     . p2p5+2*p1p3*p3p5+2*p2p3*p3p5-6*p3p5s2))+w17*(24*p3p5*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*p2p3*(-3*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3
     . *hbcm2)+4*p1p3*(-3*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+
     . 4*(2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3+3*ffmcfmb*hbcm4-2*
     . fmb*fmc*hbcm2-3*fmc*hbcm3-2*fmc2*hbcm2+2*hbcm2*p1p2+4*p1p3*
     . p2p3-6*p1p3*p3p5+2*p1p3s2-6*p2p3*p3p5+2*p2p3s2))+w12*(24*p3p5*
     . (ffmcfmb*hbcm2-fmc*hbcm)+8*p2p3*(-ffmcfmb*hbcm2+fmc*hbcm)+4*
     . p1p3*(-ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+4*(-ffmcfmb
     . **2*hbcm4+2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3+3*ffmcfmb*
     . hbcm4-2*fmb*fmc*hbcm2-3*fmc*hbcm3-fmc2*hbcm2+2*p1p3*p2p3-6*
     . p1p3*p3p5+2*p1p3s2))+2*(-8*fb1*fmb*hbcm-4*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2-12*fb2*hbcm2+8*fb2*p1p3+8*fb2*p2p3-24*fb2*p3p5+3
     . *hbcm2))
      b(11)=ans
      ans=ccc*(w2*(8*p5p7*(ffmcfmb*hbcm3+2*fmb*hbcm2-hbcm3)+8*(p3p7*
     . p3p5)*(-ffmcfmb*hbcm-2*fmb+fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2+fmb*
     . fmc*hbcm-3*fmb2*hbcm)+16*(hbcm*p1p5*p3p7+hbcm*p2p5*p3p7+hbcm*
     . p3p5*p5p7))+w17*(16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*
     . p2p3)*(3*ffmcfmb*hbcm-2*fmb+fmc)+8*(p3p7*p1p3)*(3*ffmcfmb*hbcm
     . -2*fmb+fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*
     . p3p5*p3p7-hbcm*p1p2*p3p7-hbcm*p1p3*p5p7-hbcm*p2p3*p5p7))+w12*(
     . 16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p1p3)*(-ffmcfmb*hbcm
     . -2*fmb+fmc)+8*p3p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm+fmc2*hbcm)-16*hbcm*p1p3*p5p7)
     . +(4*p3p7*(-4*fb1*ffmcfmb*hbcm-8*fb2*fmb+4*fb2*fmc-3*hbcm)-32*
     . fb1*hbcm*p5p7))
      b(12)=ans
      b(13)=ccc*(w2*(4*p3p5*(2*ffmcfmb*hbcm-3*fmb+fmc-4*hbcm)+4*(3*
     . ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-3*fmb*hbcm2-4*fmb2*hbcm+2*hbcm*
     . p1p5+2*hbcm*p2p5))+w17*(4*p2p3*(2*ffmcfmb*hbcm-3*fmb+fmc+2*
     . hbcm)+4*p1p3*(2*ffmcfmb*hbcm-3*fmb+fmc+2*hbcm)+4*(3*ffmcfmb*
     . fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+2
     . *fmc*hbcm2+3*fmc2*hbcm-4*hbcm*p1p2))+w12*(4*p1p3*(-2*ffmcfmb*
     . hbcm-3*fmb+fmc+2*hbcm)+4*(2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*
     . hbcm2-3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+2*fmc
     . *hbcm2+fmc2*hbcm))+4*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-6*fb2*fmb
     . +2*fb2*fmc-3*hbcm))
      b(14)=ccc*(w2*((p3p7*p3p5)*(8*ffmcfmb-16)+8*p5p7*(ffmcfmb*hbcm2
     . -3*fmb*hbcm-hbcm2)+8*p3p7*(2*ffmcfmb*fmb*hbcm-3*fmb*hbcm)-24*
     . p3p5*p5p7)+w17*((p3p7*p2p3)*(8*ffmcfmb-16)+(p3p7*p1p3)*(8*
     . ffmcfmb-16)+24*p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p7*(2*ffmcfmb
     . *fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-2*fmc*hbcm)+24*(-
     . p1p3*p5p7-p2p3*p5p7))+w12*((p3p7*p1p3)*(8*ffmcfmb-16)+24*p5p7*
     . (ffmcfmb*hbcm2-fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc
     . *hbcm+2*ffmcfmb*hbcm2-2*fmc*hbcm)-24*p1p3*p5p7)+(p3p7*(16*fb2*
     . ffmcfmb-32*fb2+4)-48*fb2*p5p7))
      ans3=w12*(32*(p4p6*p3p7)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*p3p6
     . )*(-2*ffmcfmb*hbcm2+fmc*hbcm)+32*(p6p7*p3p4)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p1p3
     . )*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm+2*hbcm2)+8*p6p7*(
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2+2*fmc*hbcm3-fmc2*hbcm2-2*p1p3s2)+16*(p3p7*p3p6)*(-
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm)+16*(-ffmcfmb*p1p3*p3p6*
     . p3p7-p1p3*p2p3*p6p7+2*p1p3*p3p4*p6p7+p1p3*p3p6*p5p7-2*p1p3*
     . p3p7*p4p6))+((p3p7*p3p6)*(-32*fb2*ffmcfmb-8)+8*p6p7*(-2*fb1*
     . fmb*hbcm-2*fb1*fmc*hbcm+4*fb2*hbcm2+hbcm2)+32*(-fb2*p1p3*p6p7-
     . fb2*p2p3*p6p7+2*fb2*p3p4*p6p7+fb2*p3p6*p5p7-2*fb2*p3p7*p4p6))
      ans2=w17*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2-2*fmc*hbcm)+16*(p5p7*
     . p3p6)*(-2*ffmcfmb*hbcm2+fmc*hbcm)+32*(p6p7*p3p4)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(p6p7*p2p3)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*
     . hbcm+2*hbcm2)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*
     . hbcm+2*hbcm2)+8*p6p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2+2*fmc*hbcm3-fmc2*hbcm2-2*p1p3s2-2*
     . p2p3s2)+32*(p3p7*p3p6)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm)+16
     . *(-ffmcfmb*p1p3*p3p6*p3p7-ffmcfmb*p2p3*p3p6*p3p7-2*p1p3*p2p3*
     . p6p7+2*p1p3*p3p4*p6p7+p1p3*p3p6*p5p7-2*p1p3*p3p7*p4p6+2*p2p3*
     . p3p4*p6p7+p2p3*p3p6*p5p7-2*p2p3*p3p7*p4p6))+ans3
      ans1=w2*(16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p3p5)
     . *(2*ffmcfmb*hbcm2+3*fmb*hbcm+fmc*hbcm+2*hbcm2)+8*p6p7*(ffmcfmb
     . *fmb*hbcm3+fmb*fmc*hbcm2+3*fmb*hbcm3+2*fmb2*hbcm2)+16*(-2*
     . ffmcfmb*fmb*hbcm*p3p6*p3p7-ffmcfmb*p3p5*p3p6*p3p7-2*fmb*hbcm*
     . p1p3*p6p7-2*fmb*hbcm*p2p3*p6p7+3*fmb*hbcm*p3p4*p6p7-3*fmb*hbcm
     . *p3p7*p4p6-fmb*hbcm*p3p7*p5p6-hbcm2*p1p5*p6p7-hbcm2*p2p5*p6p7-
     . hbcm2*p4p6*p5p7-hbcm2*p5p6*p5p7-p1p3*p3p5*p6p7-p2p3*p3p5*p6p7+
     . 2*p3p4*p3p5*p6p7+p3p5*p3p6*p5p7-2*p3p5*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w2*(16*(p5p6*p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p5)*(
     . -ffmcfmb*hbcm+2*fmb-fmc+3*hbcm)+8*p6p7*(-2*ffmcfmb*fmb*hbcm2-
     . fmb*fmc*hbcm+2*fmb*hbcm2+3*fmb2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*
     . p5p7-hbcm*p1p5*p6p7-hbcm*p2p5*p6p7-hbcm*p4p6*p5p7-hbcm*p5p6*
     . p5p7))+w17*(8*(p6p7*p2p3)*(-ffmcfmb*hbcm+2*fmb-fmc-hbcm)+8*(
     . p6p7*p1p3)*(-ffmcfmb*hbcm+2*fmb-fmc-hbcm)+8*p6p7*(-2*ffmcfmb*
     . fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmc
     . *hbcm2-2*fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*
     . hbcm*p3p6*p5p7-2*ffmcfmb*hbcm*p3p7*p4p6+hbcm*p1p2*p6p7))+w12*(
     . 8*(p6p7*p1p3)*(ffmcfmb*hbcm+2*fmb-fmc-hbcm)+8*p6p7*(-ffmcfmb**
     . 2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+
     . 2*fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*
     . p3p7-2*ffmcfmb*hbcm*p3p6*p5p7))+4*p6p7*(4*fb1*ffmcfmb*hbcm-4*
     . fb1*hbcm+8*fb2*fmb-4*fb2*fmc+3*hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp16_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      ans4=16*(-2*ffmcfmb*hbcm*p1p5*p3p6*p3p7+2*ffmcfmb*hbcm*p3p5*
     . p3p7*p5p6+fmb*hbcm2*p2p5*p6p7-fmb*p2p3*p3p5*p6p7-fmb2*hbcm*
     . p2p3*p6p7+2*hbcm*p2p3*p3p6*p5p7+2*hbcm*p2p3*p4p6*p5p7+2*hbcm*
     . p2p3*p5p6*p5p7+hbcm*p2p5*p3p5*p6p7-2*hbcm*p2p5*p3p7*p4p6-2*
     . hbcm*p2p5*p3p7*p5p6)
      ans3=16*(p5p7*p5p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p5p7*p4p6)*(
     . -ffmcfmb*hbcm3+fmc*hbcm2)+16*(p4p6*p3p7*p3p5)*(ffmcfmb*hbcm+
     . fmc)+16*(p5p7*p3p6*p3p5)*(-ffmcfmb*hbcm+fmc)+32*(p3p7*p3p6*
     . p2p5)*(-ffmcfmb*hbcm-hbcm)+16*(p4p6*p3p7)*(-ffmcfmb*fmb*hbcm2+
     . 2*fmb*fmc*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*
     . hbcm3+fmb*fmc*hbcm+fmc*hbcm2)+16*(p5p6*p3p7)*(-ffmcfmb**2*
     . hbcm3+ffmcfmb*hbcm3+fmb*fmc*hbcm)+8*(p6p7*p3p5)*(ffmcfmb**2*
     . hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*
     . fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6*p3p5)*(ffmcfmb**2*hbcm-
     . ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+8*p6p7*(ffmcfmb**2*fmb*hbcm4-
     . ffmcfmb*fmb*fmc*hbcm3-ffmcfmb*fmb*hbcm4+ffmcfmb*fmb2*hbcm3-2*
     . ffmcfmb*hbcm*p3p5s2+fmb*fmc*hbcm3-fmb2*fmc*hbcm2+2*fmc*p3p5s2)
     . +16*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2-ffmcfmb*fmb*fmc*hbcm-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm+2*fmb*fmc*hbcm)+ans4
      ans2=w2*ans3
      ans7=8*p6p7*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4+ffmcfmb
     . **2*hbcm5+4*ffmcfmb*hbcm*p2p3s2-fmb*fmc2*hbcm2-2*fmb*p2p3s2-
     . fmc2*hbcm3+fmc3*hbcm2)+16*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2+
     . ffmcfmb**2*fmc*hbcm2+ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(-2*hbcm*
     . p1p2*p2p3*p6p7+2*hbcm*p1p2*p3p5*p6p7+2*hbcm*p1p2*p3p6*p5p7+2*
     . hbcm*p1p2*p3p7*p4p6-hbcm*p1p3*p2p5*p6p7-hbcm*p2p3*p2p5*p6p7)
      ans6=16*(p6p7*p1p2)*(fmb*hbcm2-fmc*hbcm2+hbcm3)+16*(p6p7*p2p5)*
     . (ffmcfmb*hbcm3-fmc*hbcm2)+16*(p4p6*p3p7*p2p3)*(-ffmcfmb*hbcm+
     . fmc)+16*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5*
     . p2p3)*(-3*ffmcfmb*hbcm+fmc)+16*(p4p6*p3p7*p1p3)*(-ffmcfmb*hbcm
     . +fmc)+16*(p5p7*p3p6*p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5*
     . p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p2p3*p1p3)*(2*ffmcfmb*hbcm-
     . fmb)+32*(p3p7*p3p6*p1p2)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7)*(
     . ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(p5p7*p3p6)*(ffmcfmb**2*hbcm3-
     . fmc2*hbcm)+16*(p6p7*p3p5)*(ffmcfmb**2*hbcm3-fmc2*hbcm)+8*(p6p7
     . *p2p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-
     . 3*ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+8*(p6p7*
     . p1p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6*p2p3)*(
     . ffmcfmb**2*hbcm-ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+16*(p3p7*p3p6*
     . p1p3)*(ffmcfmb**2*hbcm-ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+ans7
      ans5=w17*ans6
      ans8=(8*(p4p6*p3p7)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p5p7
     . *p3p6)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p6p7*p3p5)*(4*
     . fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+4*p6p7*(-4*fb1*ffmcfmb**2*
     . hbcm3+4*fb1*ffmcfmb*hbcm3+4*fb1*fmb*fmc*hbcm+4*fb2*ffmcfmb*fmb
     . *hbcm2-4*fb2*ffmcfmb*fmc*hbcm2+4*fb2*fmc*hbcm2-2*ffmcfmb*hbcm3
     . -fmb*hbcm2+3*fmc*hbcm2-hbcm3)+8*(p3p7*p3p6)*(4*fb1*ffmcfmb**2*
     . hbcm+4*fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmc+4*fb2*fmc+4*ffmcfmb*
     . hbcm-fmc-hbcm)+32*(-fb1*hbcm*p2p5*p6p7-fb2*fmb*p2p3*p6p7))
      ans1=w7*(16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(p4p6*p3p7
     . *p2p3)*(ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+
     . fmc)+16*(p6p7*p3p5*p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p2p3)*(
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6*p2p3)*(ffmcfmb**2*
     . hbcm-ffmcfmb*fmc+ffmcfmb*hbcm+fmc)+16*(-2*ffmcfmb*hbcm*p1p2*
     . p3p6*p3p7+2*ffmcfmb*hbcm*p2p5*p3p6*p3p7-fmb*p2p3s2*p6p7-hbcm*
     . p2p3*p2p5*p6p7))+ans2+ans5+ans8
      ans=ccc*ans1
      b(1)=ans
      ans2=(8*p5p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+4*p3p7*(-4*
     . fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmb+4*fb2*ffmcfmb*fmc-4*fb2*fmc
     . -fmb+fmc+hbcm))
      ans1=w7*(16*(p5p7*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p3p7*p2p3)*(-
     . ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+16*(ffmcfmb*hbcm*
     . p1p2*p3p7-2*ffmcfmb*hbcm*p2p5*p3p7))+w2*(16*(p5p7*p3p5)*(
     . ffmcfmb*hbcm-fmc)+16*(p5p7*p2p3)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*
     . p3p5)*(-ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+8*p3p7*(2*
     . ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmb2*hbcm-2*
     . fmb*fmc*hbcm)+8*p5p7*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmc*hbcm2)+16*(
     . ffmcfmb*hbcm*p1p5*p3p7+hbcm*p2p5*p3p7))+w17*(16*(p5p7*p2p3)*(3
     . *ffmcfmb*hbcm-fmc)+16*(p5p7*p1p3)*(ffmcfmb*hbcm-fmc)+16*(p3p7*
     . p1p2)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*p2p3)*(-ffmcfmb*fmb+ffmcfmb*
     . fmc+ffmcfmb*hbcm-fmc)+8*(p3p7*p1p3)*(-ffmcfmb*fmb+ffmcfmb*fmc+
     . ffmcfmb*hbcm-fmc)+16*p5p7*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+8*p3p7
     . *(-ffmcfmb**2*hbcm3+fmc2*hbcm)+32*(-ffmcfmb*hbcm*p2p5*p3p7-
     . hbcm*p1p2*p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w7*(16*(p4p6*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p3p6*p2p3)*(
     . ffmcfmb*hbcm-fmc))+w2*(16*(p4p6*p3p5)*(ffmcfmb*hbcm-fmc)+8*(
     . p3p6*p3p5)*(ffmcfmb*hbcm-fmc)+16*(p5p6*p2p3)*(-ffmcfmb*hbcm+
     . hbcm)+24*p4p6*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+8*p5p6*(ffmcfmb
     . **2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-
     . fmb*fmc*hbcm+fmc*hbcm2)+8*p3p6*(ffmcfmb**2*fmb*hbcm2-ffmcfmb*
     . fmb*fmc*hbcm+ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+16*(ffmcfmb*hbcm*
     . p2p5*p3p6+hbcm*p2p5*p4p6+hbcm*p2p5*p5p6))+w17*(16*(p4p6*p2p3)*
     . (3*ffmcfmb*hbcm-fmc)+8*(p3p6*p2p3)*(5*ffmcfmb*hbcm-fmc)+16*(
     . p4p6*p1p3)*(ffmcfmb*hbcm-fmc)+8*(p3p6*p1p3)*(ffmcfmb*hbcm-fmc)
     . +16*(p3p6*p1p2)*(-ffmcfmb*hbcm-hbcm)+16*p4p6*(-ffmcfmb**2*
     . hbcm3+fmc2*hbcm)+8*p3p6*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*fmc*
     . hbcm2-ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc2*hbcm+
     . fmc2*hbcm)+32*(ffmcfmb*hbcm*p2p5*p3p6-hbcm*p1p2*p4p6))+(8*p4p6
     . *(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+4*p3p6*(-4*fb1*ffmcfmb*
     . hbcm-4*fb2*fmc+hbcm)))
      b(3)=ans
      ans3=((p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(p6p7*p2p3)*(-32*fb2*
     . ffmcfmb-8)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+32*fb2*ffmcfmb-16*
     . ffmcfmb+8)+4*p6p7*(4*fb1*ffmcfmb*fmb*hbcm+4*fb1*ffmcfmb*fmc*
     . hbcm-4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2
     . +4*fb2*fmb*fmc-fmb*hbcm-fmc*hbcm+hbcm2)+32*(fb2*p1p2*p6p7-fb2*
     . p2p4*p6p7))
      ans2=w17*(16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p2
     . )*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+16*(p3p7*p3p6*
     . p2p3)*(-ffmcfmb**2+ffmcfmb)+16*(p3p7*p3p6*p1p3)*(-ffmcfmb**2+
     . ffmcfmb)+16*(p5p7*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm)+8*
     . (p6p7*p2p3)*(ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc
     . *hbcm+3*ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm)+8*(p6p7*p1p3)*(ffmcfmb
     . **2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*
     . fmc-fmc*hbcm)+8*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*
     . hbcm3-ffmcfmb**2*hbcm4-2*ffmcfmb*p2p3s2-fmb*fmc2*hbcm+fmc2*
     . hbcm2-fmc3*hbcm)+16*(ffmcfmb**2*hbcm2*p3p6*p3p7-ffmcfmb*p1p3*
     . p2p3*p6p7+ffmcfmb*p1p3*p3p7*p4p6+ffmcfmb*p2p3*p3p7*p4p6+p1p2*
     . p1p3*p6p7+p1p2*p2p3*p6p7-p1p3*p2p4*p6p7-p2p3*p2p4*p6p7))+ans3
      ans1=w7*(16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p2)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p3p6*p2p3)*(-ffmcfmb**2+
     . ffmcfmb)+8*(p6p7*p2p3)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm)+16*(ffmcfmb*
     . p2p3*p3p7*p4p6-ffmcfmb*p2p3s2*p6p7+p1p2*p2p3*p6p7-p2p3*p2p4*
     . p6p7))+w2*(16*(p5p7*p5p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*
     . p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p6*p3p7)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-fmc*hbcm)+16*(p3p7*p3p6*p3p5)*(-ffmcfmb
     . **2+ffmcfmb)+16*(p5p7*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm
     . )+8*(p6p7*p3p5)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc
     . *hbcm-ffmcfmb*hbcm2+fmb*fmc+fmc*hbcm)+8*p6p7*(ffmcfmb**2*fmb*
     . hbcm3-ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*
     . hbcm2+fmb*fmc*hbcm2+fmb2*fmc*hbcm)+16*(-ffmcfmb*fmb*hbcm*p2p3*
     . p6p7+ffmcfmb*fmb*hbcm*p3p6*p3p7+2*ffmcfmb*fmb*hbcm*p3p7*p4p6-
     . ffmcfmb*p2p3*p3p5*p6p7+ffmcfmb*p3p5*p3p7*p4p6+fmb*hbcm*p1p2*
     . p6p7-fmb*hbcm*p2p4*p6p7+p1p2*p3p5*p6p7-p2p4*p3p5*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans2=w17*(16*(p3p7*p2p5)*(ffmcfmb*hbcm2-2*fmc*hbcm)+8*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2+3*fmc*hbcm)+8*(p5p7*p1p3)*(ffmcfmb*hbcm2
     . -fmc*hbcm)+8*(p3p7*p2p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-fmb*fmc-fmc2)+8*(p3p7*p1p3)*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc2)+8*p5p7*(-ffmcfmb
     . **2*hbcm4+fmc2*hbcm2+4*p2p3s2)+8*p3p7*(-ffmcfmb**2*fmb*hbcm3-
     . ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*p1p3s2-2*ffmcfmb*p2p3s2+fmb*
     . fmc2*hbcm+fmc3*hbcm)+16*(-2*ffmcfmb*p1p3*p2p3*p3p7+ffmcfmb*
     . p1p3*p3p5*p3p7+ffmcfmb*p2p3*p3p5*p3p7-fmb*hbcm*p1p2*p3p7-hbcm2
     . *p1p2*p5p7+p1p2*p1p3*p3p7+p1p2*p2p3*p3p7+2*p1p3*p2p3*p5p7-2*
     . p1p3*p2p5*p3p7-2*p2p3*p2p5*p3p7))+((p3p7*p3p5)*(32*fb2*ffmcfmb
     . +8)+(p3p7*p2p3)*(-32*fb2*ffmcfmb-8)+(p3p7*p1p3)*(-32*fb2*
     . ffmcfmb-8)+4*p5p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+
     . 4*p3p7*(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2-4*fb2*fmb*fmc-4*fb2*fmc2+2*ffmcfmb*hbcm2+fmb*
     . hbcm+fmc*hbcm+hbcm2)+32*(fb2*p1p2*p3p7+2*fb2*p2p3*p5p7-2*fb2*
     . p2p5*p3p7))
      ans1=w7*(16*(p3p7*p2p5)*(ffmcfmb*hbcm2-2*fmc*hbcm)+24*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p2p3)*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc2)+16*(-ffmcfmb
     . *p1p3*p2p3*p3p7+ffmcfmb*p2p3*p3p5*p3p7-ffmcfmb*p2p3s2*p3p7+fmc
     . *hbcm*p1p2*p3p7+p1p2*p2p3*p3p7-2*p2p3*p2p5*p3p7+2*p2p3s2*p5p7)
     . )+w2*(16*(p3p7*p2p5)*(-2*fmb*hbcm+fmc*hbcm)+8*(p5p7*p3p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p3p5)*(3*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc2)+8*p5p7*(-ffmcfmb
     . **2*hbcm4-ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4+
     . fmb*fmc*hbcm2-fmc*hbcm3)+8*p3p7*(-ffmcfmb**2*fmb*hbcm3+ffmcfmb
     . *fmb*fmc*hbcm2+2*ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2+2*
     . ffmcfmb*p3p5s2-fmb*fmc2*hbcm-2*fmb2*fmc*hbcm)+16*(-ffmcfmb*fmb
     . *hbcm*p1p3*p3p7-ffmcfmb*fmb*hbcm*p2p3*p3p7-ffmcfmb*p1p3*p3p5*
     . p3p7-ffmcfmb*p2p3*p3p5*p3p7+fmb*hbcm*p1p2*p3p7+2*fmb*hbcm*p2p3
     . *p5p7+fmc*hbcm*p1p5*p3p7+p1p2*p3p5*p3p7+2*p2p3*p3p5*p5p7-2*
     . p2p5*p3p5*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans2=w17*(16*(p3p6*p1p2)*(-fmb*hbcm-fmc*hbcm)+16*(p3p6*p2p5)*(2
     . *ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*p2p3)*(-ffmcfmb*hbcm2+3*fmc*
     . hbcm)+8*(p4p6*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p6*p2p3)*(5*
     . ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-fmb*fmc)+8*(p3p6*p1p3)*(
     . ffmcfmb*fmb*hbcm-fmb*fmc)+8*p4p6*(-ffmcfmb**2*hbcm4+fmc2*hbcm2
     . +4*p2p3s2)+16*(p3p6*p3p5)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)
     . +8*p3p6*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3-ffmcfmb**2
     . *hbcm4+ffmcfmb*fmc*hbcm3+2*ffmcfmb*p2p3s2+fmb*fmc2*hbcm+fmc3*
     . hbcm)+16*(ffmcfmb*p1p3*p2p3*p3p6-hbcm2*p1p2*p4p6+2*p1p3*p2p3*
     . p4p6-p1p3*p2p5*p3p6-p2p3*p2p5*p3p6))+((p3p6*p2p3)*(32*fb2*
     . ffmcfmb+8)+4*p4p6*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+
     . 4*p3p6*(-4*fb1*ffmcfmb*fmb*hbcm-4*fb2*fmb*fmc+fmb*hbcm)+32*(2*
     . fb2*p2p3*p4p6-fb2*p2p5*p3p6))
      ans1=w7*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+24*(p4p6*p2p3)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p6*p2p3)*(-2*ffmcfmb**2*hbcm2+
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*fmc)+16*(ffmcfmb*
     . p2p3s2*p3p6-p2p3*p2p5*p3p6+2*p2p3s2*p4p6))+w2*(16*(p5p6*p3p5)*
     . (-ffmcfmb*hbcm2+fmc*hbcm)+24*(p4p6*p3p5)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmb*hbcm)+16*p4p6*(-
     . ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2)+8*p5p6*(ffmcfmb**2*hbcm4-
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2+fmc*hbcm3)+8*(p3p6*p3p5)*(-2*ffmcfmb**2*hbcm2+ffmcfmb*
     . fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*fmc)+8*p3p6*(-ffmcfmb**2*fmb*
     . hbcm3+ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmb2*hbcm2-fmb2*fmc*hbcm)+
     . 16*(2*ffmcfmb*fmb*hbcm*p2p3*p3p6+ffmcfmb*p2p3*p3p5*p3p6+3*fmb*
     . hbcm*p2p3*p4p6+fmb*hbcm*p2p3*p5p6+hbcm2*p2p5*p4p6+hbcm2*p2p5*
     . p5p6+2*p2p3*p3p5*p4p6-p2p5*p3p5*p3p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w17*((p2p3*p1p3)*(-8*ffmcfmb+16)+24*p2p5*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+8*p1p2*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+4*
     . p2p3*(ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2+2*fmb*fmc+3*fmc*hbcm+fmc2)+4*p1p3*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*
     . fmb*fmc-fmc*hbcm+fmc2)+4*(2*ffmcfmb**2*fmb*hbcm3+2*ffmcfmb**2*
     . fmc*hbcm3-ffmcfmb**2*hbcm4-2*ffmcfmb*p2p3s2-2*fmb*fmc2*hbcm+
     . fmc2*hbcm2-2*fmc3*hbcm-2*p1p2*p1p3-2*p1p2*p2p3+6*p1p3*p2p5+6*
     . p2p3*p2p5+4*p2p3s2))+(p2p3*(-16*fb2*ffmcfmb+32*fb2-4)+2*(8*fb1
     . *ffmcfmb*fmb*hbcm+8*fb1*ffmcfmb*fmc*hbcm-4*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2+8*fb2*fmb*fmc+4*fb2*fmc2-
     . 8*fb2*p1p2+24*fb2*p2p5-2*ffmcfmb*hbcm2-2*fmb*hbcm+hbcm2))
      ans1=w7*(24*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p1p2*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*p2p3*(ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+2*fmb*fmc+3*fmc*hbcm+fmc2)+8*
     . (-ffmcfmb*p2p3s2-p1p2*p2p3+3*p2p3*p2p5+2*p2p3s2))+w2*(8*p2p5*(
     . 3*fmb*hbcm-fmc*hbcm+hbcm2)+(p3p5*p2p3)*(-8*ffmcfmb+16)+8*p1p5*
     . (ffmcfmb*hbcm2-fmc*hbcm)+8*p2p3*(-2*ffmcfmb*fmb*hbcm+3*fmb*
     . hbcm)+4*p3p5*(ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2+2*fmb*fmc+3*fmc*hbcm+fmc2)+4*(2*
     . ffmcfmb**2*fmb*hbcm3-3*ffmcfmb*fmb*fmc*hbcm2-2*ffmcfmb*fmb*
     . hbcm3-3*ffmcfmb*fmb2*hbcm2+2*fmb*fmc*hbcm2+fmb*fmc2*hbcm-2*fmb
     . *hbcm*p1p2+3*fmb2*fmc*hbcm-2*p1p2*p3p5+6*p2p5*p3p5))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=(8*p3p5*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+8*p2p3*(4*fb1
     . *ffmcfmb*hbcm+4*fb2*fmb+hbcm)+4*p1p3*(4*fb1*ffmcfmb*hbcm+4*fb2
     . *fmc-hbcm)+2*(-8*fb1*ffmcfmb*hbcm3-4*fb1*fmb*fmc*hbcm+4*fb1*
     . fmc2*hbcm-8*fb1*hbcm*p1p2+16*fb1*hbcm*p2p5-4*fb2*ffmcfmb*fmb*
     . hbcm2+4*fb2*ffmcfmb*fmc*hbcm2-8*fb2*fmc*hbcm2+fmb*hbcm2-fmc*
     . hbcm2+2*hbcm3))
      ans2=w17*(16*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm3-fmb*hbcm2-2*hbcm3)+16*(p3p5*p2p3)*(3*ffmcfmb*hbcm-fmc)+
     . 16*(p3p5*p1p3)*(ffmcfmb*hbcm-fmc)+8*(p2p3*p1p3)*(-5*ffmcfmb*
     . hbcm+2*fmb+fmc)+16*p3p5*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+4*p2p3*(
     . 2*ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+6*
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm-2*fmc*hbcm2-5*fmc2*hbcm)+4*p1p3*(
     . 2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3-fmb*fmc*hbcm-2*fmc*hbcm2-fmc2*hbcm)+4*(-ffmcfmb
     . **2*fmb*hbcm4+ffmcfmb**2*fmc*hbcm4-2*ffmcfmb**2*hbcm5-2*
     . ffmcfmb*hbcm*p1p3s2-8*ffmcfmb*hbcm*p2p3s2+fmb*fmc2*hbcm2+4*fmb
     . *p2p3s2+2*fmc*p1p3s2+2*fmc2*hbcm3-fmc3*hbcm2+2*hbcm*p1p2*p1p3+
     . 6*hbcm*p1p2*p2p3-8*hbcm*p1p2*p3p5+4*hbcm*p1p3*p2p5+4*hbcm*p2p3
     . *p2p5))+ans3
      ans1=w7*(16*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+16*(p3p5*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p2p3*
     . p1p3)*(-ffmcfmb*hbcm+fmc)+4*p2p3*(-2*ffmcfmb**2*hbcm3-3*
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+3*fmb*
     . fmc*hbcm-2*fmc*hbcm2-fmc2*hbcm)+8*(2*fmb*p2p3s2-hbcm*p1p2*p2p3
     . +2*hbcm*p2p3*p2p5))+w2*(8*p2p5*(-2*fmb*hbcm2-fmc*hbcm2+hbcm3)+
     . 8*p1p5*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p5*p1p3)*(-ffmcfmb*hbcm+
     . fmc)+8*p2p3*(-2*ffmcfmb*fmb*hbcm2+3*fmb2*hbcm)+8*p1p3*(-
     . ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+4*p3p5*(-2*ffmcfmb**2*hbcm3+3*
     . ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-3*fmb*
     . fmc*hbcm-2*fmc*hbcm2-fmc2*hbcm)+4*(-ffmcfmb**2*fmb*hbcm4+2*
     . ffmcfmb*fmb*fmc*hbcm3+3*ffmcfmb*fmb*hbcm4-2*ffmcfmb*fmb2*hbcm3
     . +4*ffmcfmb*hbcm*p3p5s2-3*fmb*fmc*hbcm3-fmb*fmc2*hbcm2+2*fmb*
     . hbcm2*p1p2+4*fmb*p2p3*p3p5+2*fmb2*fmc*hbcm2-4*fmc*p3p5s2+2*
     . hbcm*p1p2*p3p5-4*hbcm*p1p5*p2p3-4*hbcm*p2p3*p2p5-4*hbcm*p2p5*
     . p3p5))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(w2*(8*p5p6*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p3p6*
     . p3p5)*(-3*ffmcfmb*hbcm+fmb)+8*p3p6*(-2*ffmcfmb*fmb*hbcm2+fmb2*
     . hbcm)+8*(-3*fmb*hbcm2*p4p6-4*hbcm*p3p5*p4p6-2*hbcm*p3p5*p5p6))
     . +w17*(16*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p6*p2p3)*(
     . ffmcfmb*hbcm+fmb)+8*(p3p6*p1p3)*(ffmcfmb*hbcm+fmb)+8*p3p6*(-2*
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*p3p5*p3p6+hbcm*p1p2*p3p6+
     . hbcm*p1p3*p4p6+hbcm*p2p3*p4p6))+w7*(16*p4p6*(-ffmcfmb*hbcm3+
     . fmc*hbcm2)+8*(p3p6*p2p3)*(ffmcfmb*hbcm+fmb)+8*p3p6*(-ffmcfmb**
     . 2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm)+16*
     . hbcm*p2p3*p4p6)+(4*p3p6*(4*fb1*ffmcfmb*hbcm+4*fb2*fmb+3*hbcm)+
     . 32*fb1*hbcm*p4p6))
      b(10)=ccc*(w2*((p3p6*p3p5)*(-8*ffmcfmb-8)+8*p5p6*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+8*p3p6*(-2*ffmcfmb*fmb*hbcm-fmb*hbcm)+8*
     . (-4*fmb*hbcm*p4p6-3*p3p5*p4p6))+w17*((p3p6*p2p3)*(-8*ffmcfmb-8
     . )+(p3p6*p1p3)*(-8*ffmcfmb-8)+24*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+
     . 8*p3p6*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2
     . -fmc*hbcm)+24*(-p1p3*p4p6-p2p3*p4p6))+w7*((p3p6*p2p3)*(-8*
     . ffmcfmb-8)+24*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb**2
     . *hbcm2-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)-24*p2p3*p4p6)+
     . (p3p6*(-16*fb2*ffmcfmb-16*fb2-4)-48*fb2*p4p6))
      ans=ccc*(w2*(4*p3p5*(3*ffmcfmb*hbcm2-4*fmb*hbcm+fmc*hbcm-3*
     . hbcm2)+4*(2*ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2+2*fmb*hbcm*p1p3+2*
     . fmb*hbcm*p2p3-4*fmb*hbcm3+3*fmb2*hbcm2-2*hbcm2*p1p5-2*hbcm2*
     . p2p5+2*p1p3*p3p5+2*p2p3*p3p5-6*p3p5s2))+w17*(24*p3p5*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*p2p3*(-3*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3
     . *hbcm2)+4*p1p3*(-3*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+
     . 4*(2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3+3*ffmcfmb*hbcm4-2*
     . fmb*fmc*hbcm2-3*fmc*hbcm3-2*fmc2*hbcm2+2*hbcm2*p1p2+4*p1p3*
     . p2p3-6*p1p3*p3p5+2*p1p3s2-6*p2p3*p3p5+2*p2p3s2))+w7*(24*p3p5*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p2p3*(-ffmcfmb*hbcm2-2*fmb*hbcm+fmc*
     . hbcm-3*hbcm2)+8*p1p3*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(-ffmcfmb**2*
     . hbcm4+2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3+3*ffmcfmb*hbcm4-
     . 2*fmb*fmc*hbcm2-3*fmc*hbcm3-fmc2*hbcm2+2*p1p3*p2p3-6*p2p3*p3p5
     . +2*p2p3s2))+2*(-8*fb1*fmb*hbcm-4*fb1*fmc*hbcm+4*fb2*ffmcfmb*
     . hbcm2-12*fb2*hbcm2+8*fb2*p1p3+8*fb2*p2p3-24*fb2*p3p5+3*hbcm2))
      b(11)=ans
      ans=ccc*(w2*(8*p5p7*(ffmcfmb*hbcm3+2*fmb*hbcm2-hbcm3)+8*(p3p7*
     . p3p5)*(-ffmcfmb*hbcm-2*fmb+fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2+fmb*
     . fmc*hbcm-3*fmb2*hbcm)+16*(hbcm*p1p5*p3p7+hbcm*p2p5*p3p7+hbcm*
     . p3p5*p5p7))+w17*(16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*
     . p2p3)*(3*ffmcfmb*hbcm-2*fmb+fmc)+8*(p3p7*p1p3)*(3*ffmcfmb*hbcm
     . -2*fmb+fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*
     . p3p5*p3p7-hbcm*p1p2*p3p7-hbcm*p1p3*p5p7-hbcm*p2p3*p5p7))+w7*(
     . 16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p2p3)*(ffmcfmb*hbcm-
     . 2*fmb+fmc)+8*p3p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc2*hbcm)+16*(
     . ffmcfmb*hbcm*p1p3*p3p7-2*ffmcfmb*hbcm*p3p5*p3p7-hbcm*p2p3*p5p7
     . ))+(4*p3p7*(-4*fb1*ffmcfmb*hbcm-8*fb2*fmb+4*fb2*fmc-3*hbcm)-32
     . *fb1*hbcm*p5p7))
      b(12)=ans
      b(13)=ccc*(w2*(4*p3p5*(2*ffmcfmb*hbcm-3*fmb+fmc-4*hbcm)+4*(3*
     . ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-3*fmb*hbcm2-4*fmb2*hbcm+2*hbcm*
     . p1p5+2*hbcm*p2p5))+w17*(4*p2p3*(2*ffmcfmb*hbcm-3*fmb+fmc+2*
     . hbcm)+4*p1p3*(2*ffmcfmb*hbcm-3*fmb+fmc+2*hbcm)+4*(3*ffmcfmb*
     . fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+2
     . *fmc*hbcm2+3*fmc2*hbcm-4*hbcm*p1p2))+w7*(4*p2p3*(-2*ffmcfmb*
     . hbcm-3*fmb+fmc+2*hbcm)+4*(2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*
     . hbcm2-3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+2*fmc
     . *hbcm2+fmc2*hbcm))+4*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-6*fb2*fmb
     . +2*fb2*fmc-3*hbcm))
      b(14)=ccc*(w7*((p3p7*p2p3)*(8*ffmcfmb-16)+24*p5p7*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p7*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-2*fmc*hbcm)-24*p2p3*p5p7)+w2*((p3p7*p3p5)*(8*
     . ffmcfmb-16)+8*p5p7*(ffmcfmb*hbcm2-3*fmb*hbcm-hbcm2)+8*p3p7*(2*
     . ffmcfmb*fmb*hbcm-3*fmb*hbcm)-24*p3p5*p5p7)+w17*((p3p7*p2p3)*(8
     . *ffmcfmb-16)+(p3p7*p1p3)*(8*ffmcfmb-16)+24*p5p7*(ffmcfmb*hbcm2
     . -fmc*hbcm)+8*p3p7*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-2*fmc*hbcm)+24*(-p1p3*p5p7-p2p3*p5p7))+(p3p7*(16
     . *fb2*ffmcfmb-32*fb2+4)-48*fb2*p5p7))
      ans2=w7*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2-2*fmc*hbcm)+16*(p5p7*
     . p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+8*(p6p7*p2p3)*(3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+
     . hbcm2)+16*(p6p7*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*p6p7*(-
     . ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb*
     . hbcm4-fmb*fmc*hbcm2+fmc*hbcm3-2*p2p3s2)+16*(p3p7*p3p6)*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm)+16*(-
     . ffmcfmb*p2p3*p3p6*p3p7-p1p3*p2p3*p6p7+p2p3*p3p4*p6p7+p2p3*p3p6
     . *p5p7-2*p2p3*p3p7*p4p6))+((p3p7*p3p6)*(-32*fb2*ffmcfmb-8)+4*
     . p6p7*(-4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+
     . 32*(-fb2*p1p3*p6p7-fb2*p2p3*p6p7+fb2*p3p4*p6p7+fb2*p3p6*p5p7-2
     . *fb2*p3p7*p4p6))
      ans1=w2*(16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p3p5)
     . *(ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*p6p7*(ffmcfmb*fmb*hbcm3+fmb*
     . hbcm3+fmb2*hbcm2)+16*(-2*ffmcfmb*fmb*hbcm*p3p6*p3p7-ffmcfmb*
     . p3p5*p3p6*p3p7-fmb*hbcm*p1p3*p6p7-fmb*hbcm*p2p3*p6p7+fmb*hbcm*
     . p3p4*p6p7-3*fmb*hbcm*p3p7*p4p6-fmb*hbcm*p3p7*p5p6-hbcm2*p4p6*
     . p5p7-hbcm2*p5p6*p5p7-p1p3*p3p5*p6p7-p2p3*p3p5*p6p7+p3p4*p3p5*
     . p6p7+p3p5*p3p6*p5p7-2*p3p5*p3p7*p4p6))+w17*(16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2-2*fmc*hbcm)+16*(p5p7*p3p6)*(-2*ffmcfmb*hbcm2+fmc
     . *hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p3)*
     . (ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+hbcm2)+8*(p6p7*p1p3)*(
     . ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+hbcm2)+8*p6p7*(ffmcfmb*fmb*
     . hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3-
     . fmc2*hbcm2-2*p1p3s2-2*p2p3s2)+32*(p3p7*p3p6)*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm)+16*(-ffmcfmb*p1p3*p3p6*p3p7-ffmcfmb*
     . p2p3*p3p6*p3p7+hbcm2*p1p2*p6p7-2*p1p3*p2p3*p6p7+p1p3*p3p4*p6p7
     . +p1p3*p3p6*p5p7-2*p1p3*p3p7*p4p6+p2p3*p3p4*p6p7+p2p3*p3p6*p5p7
     . -2*p2p3*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w2*(16*(p5p6*p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p5)
     . *(-ffmcfmb*hbcm+fmb+hbcm)+8*p6p7*(-ffmcfmb*fmb*hbcm2+fmb*hbcm2
     . +fmb2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p5p7-hbcm*p4p6*p5p7-hbcm*
     . p5p6*p5p7))+w17*(8*(p6p7*p2p3)*(-ffmcfmb*hbcm+fmb-hbcm)+8*(
     . p6p7*p1p3)*(-ffmcfmb*hbcm+fmb-hbcm)+8*p6p7*(-ffmcfmb*fmb*hbcm2
     . +ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*
     . hbcm)+16*(-2*ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*hbcm*p3p6*p5p7-2
     . *ffmcfmb*hbcm*p3p7*p4p6+hbcm*p1p2*p6p7))+w7*(8*(p6p7*p2p3)*(
     . ffmcfmb*hbcm+fmb-hbcm)+8*p6p7*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+
     . 16*(-ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*hbcm*p3p7*p4p6))+4*p6p7*
     . (4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+3*hbcm))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp15_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(1)=ccc*(w16*(16*(p5p7*p5p6)*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+
     . 16*(p5p7*p3p6*p3p5)*(ffmcfmb*hbcm-fmc+2*hbcm)+16*(p5p7*p3p6*
     . p2p3)*(-ffmcfmb*hbcm+fmc-2*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*fmb*
     . hbcm2-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+16*(fmb*hbcm2*p4p6*
     . p5p7+2*hbcm*p1p2*p3p6*p5p7-2*hbcm*p1p5*p3p6*p5p7-2*hbcm*p2p3*
     . p4p6*p5p7-2*hbcm*p2p3*p5p6*p5p7-2*hbcm*p2p5*p3p6*p5p7+2*hbcm*
     . p3p5*p4p6*p5p7+2*hbcm*p3p5*p5p6*p5p7))+w12*(16*(p5p7*p3p6*p1p3
     . )*(ffmcfmb*hbcm-fmc)+16*(p5p7*p3p6)*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2))+8*(p5p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb2*
     . fmc+3*hbcm))
      b(2)=ccc*(w16*(8*(p5p7*p3p5)*(-fmb+fmc-2*hbcm)+8*(p5p7*p2p3)*(
     . fmb-fmc+2*hbcm)+8*p5p7*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*
     . hbcm2-2*fmb2*hbcm)+16*(-hbcm*p1p2*p5p7+hbcm*p1p5*p5p7+2*hbcm*
     . p2p5*p5p7))+w12*(8*(p5p7*p1p3)*(-fmb+fmc)+8*p5p7*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+16*p5p7*(-fb2
     . *fmb+fb2*fmc))
      b(3)=16*ccc*w16*(hbcm*p2p5*p3p6+2*hbcm*p2p5*p4p6+hbcm*p2p5*p5p6
     . )
      b(4)=ccc*(w16*((p5p7*p3p6*p3p5)*(-16*ffmcfmb+16)+(p5p7*p3p6*
     . p2p3)*(16*ffmcfmb-16)+16*(p5p7*p5p6)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+16*(fmb*hbcm*p3p6*p5p7+2*fmb*hbcm*p4p6*p5p7-p2p3*p4p6*
     . p5p7+p3p5*p4p6*p5p7))+w12*((p5p7*p3p6*p1p3)*(-16*ffmcfmb+16)+
     . 16*(p5p7*p4p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(fmc*hbcm*p3p6*
     . p5p7+p1p3*p4p6*p5p7))+((p5p7*p3p6)*(-32*fb2*ffmcfmb+32*fb2-8)+
     . 32*fb2*p4p6*p5p7))
      b(5)=ccc*(w16*(8*p5p7*(-fmb*fmc*hbcm2+2*fmb*hbcm3-fmb2*hbcm2+2*
     . p2p3s2+2*p3p5s2)+8*(p5p7*p3p5)*(-ffmcfmb*hbcm2+2*fmb*hbcm-fmc*
     . hbcm+hbcm2)+8*(p5p7*p2p3)*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-
     . hbcm2)+16*(-fmb*hbcm*p1p3*p5p7-hbcm2*p1p2*p5p7+hbcm2*p1p5*p5p7
     . +hbcm2*p2p5*p5p7+p1p3*p2p3*p5p7-p1p3*p3p5*p5p7-2*p2p3*p3p5*
     . p5p7))+w12*(8*p5p7*(-ffmcfmb*hbcm4+fmc*hbcm3-2*p1p3s2)+16*(
     . p5p7*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p7*p2p3)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(p5p7*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+
     . 16*(-p1p3*p2p3*p5p7+p1p3*p3p5*p5p7))+(4*p5p7*(4*fb1*fmc*hbcm+4
     . *fb2*ffmcfmb*hbcm2+4*fb2*hbcm2-hbcm2)+32*(-fb2*p1p3*p5p7-fb2*
     . p2p3*p5p7+fb2*p3p5*p5p7)))
      b(6)=ccc*(16*w16*(hbcm2*p2p5*p4p6+hbcm2*p2p5*p5p6+p2p3*p2p5*
     . p3p6-p2p5*p3p5*p3p6)+16*w12*(-fmc*hbcm*p2p5*p3p6-p1p3*p2p5*
     . p3p6)-32*fb2*p2p5*p3p6)
      b(7)=ccc*(w16*(8*p2p5*(-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+8*(-p2p3*
     . p2p5+p2p5*p3p5))+w12*(8*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p1p3*
     . p2p5)+16*fb2*p2p5)
      b(8)=w16*ccc*(8*p2p5*(-fmc*hbcm2+2*hbcm3)+16*(-hbcm*p1p3*p2p5-
     . hbcm*p2p3*p2p5+hbcm*p2p5*p3p5))
      b(9)=ccc*(w16*(8*p5p6*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p3p6*
     . p3p5)*(-ffmcfmb*hbcm+fmb)+8*(p3p6*p2p3)*(ffmcfmb*hbcm-fmb)+8*(
     . -fmb*hbcm2*p4p6+fmb2*hbcm*p3p6+2*hbcm*p2p3*p4p6+2*hbcm*p2p3*
     . p5p6-2*hbcm*p2p5*p3p6-2*hbcm*p3p5*p4p6-2*hbcm*p3p5*p5p6))+w12*
     . (8*p3p6*(fmb*fmc*hbcm-fmc2*hbcm)+8*(p3p6*p1p3)*(-ffmcfmb*hbcm+
     . fmb))+4*p3p6*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmb-3*hbcm))
      b(10)=ccc*(w16*((p3p6*p3p5)*(8*ffmcfmb-8)+(p3p6*p2p3)*(-8*
     . ffmcfmb+8)+8*p5p6*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(-fmb*hbcm
     . *p3p6-2*fmb*hbcm*p4p6+p2p3*p4p6-p3p5*p4p6))+w12*((p3p6*p1p3)*(
     . 8*ffmcfmb-8)+8*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*(-fmc*hbcm*p3p6
     . -p1p3*p4p6))+(p3p6*(16*fb2*ffmcfmb-16*fb2+4)-16*fb2*p4p6))
      b(11)=ccc*(w16*(4*p3p5*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-hbcm2
     . )+4*p2p3*(-ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm+hbcm2)+4*(fmb*fmc
     . *hbcm2+2*fmb*hbcm*p1p3-2*fmb*hbcm3+fmb2*hbcm2+2*hbcm2*p1p2-2*
     . hbcm2*p1p5-2*hbcm2*p2p5-2*p1p3*p2p3+2*p1p3*p3p5+4*p2p3*p3p5-2*
     . p2p3s2-2*p3p5s2))+w12*(8*p3p5*(ffmcfmb*hbcm2-fmc*hbcm)+8*p2p3*
     . (-ffmcfmb*hbcm2+fmc*hbcm)+4*p1p3*(-ffmcfmb*hbcm2+fmc*hbcm-
     . hbcm2)+4*(ffmcfmb*hbcm4-fmc*hbcm3+2*p1p3*p2p3-2*p1p3*p3p5+2*
     . p1p3s2))+2*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2+8*
     . fb2*p1p3+8*fb2*p2p3-8*fb2*p3p5+hbcm2))
      b(13)=ccc*(w16*(4*p3p5*(-fmb+fmc-2*hbcm)+4*p2p3*(fmb-fmc+2*hbcm
     . )+4*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm-2*
     . hbcm*p1p2+2*hbcm*p1p5+4*hbcm*p2p5))+w12*(4*p1p3*(-fmb+fmc)+4*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+8
     . *(-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp14_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*
     . p1p3))
      b(1)=ccc*(w1*(16*(p5p7*p3p6)*(fmc*hbcm2+fmc2*hbcm)+16*(p5p7*
     . p4p6)*(-ffmcfmb*hbcm3+2*hbcm3)+16*(p5p7*p3p6*p3p4)*(ffmcfmb*
     . hbcm+fmc+hbcm)+32*(-hbcm*p1p3*p4p6*p5p7-hbcm*p2p3*p4p6*p5p7+
     . hbcm*p3p4*p4p6*p5p7+hbcm*p3p5*p4p6*p5p7))+w13*(16*(p5p7*p3p6)*
     . (fmb*fmc*hbcm-fmb2*hbcm)+16*(p5p7*p3p6*p1p3)*(ffmcfmb*hbcm+fmc
     . -hbcm))+8*(p5p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*fmc
     . -3*hbcm))
      b(2)=ccc*(w1*(8*(p5p7*p3p4)*(fmb-fmc-2*hbcm)+8*p5p7*(-ffmcfmb*
     . fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm)+16*hbcm*p1p4*p5p7)+w13*(8*
     . (p5p7*p1p3)*(fmb-fmc)+8*p5p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2))+16*p5p7*(-
     . fb2*fmb+fb2*fmc))
      b(3)=16*ccc*hbcm*p2p5*p4p6*w1
      b(4)=ccc*(w1*((p5p7*p3p6*p3p4)*(16*ffmcfmb-16)+16*(p5p7*p4p6)*(
     . -ffmcfmb*hbcm2-fmc*hbcm)+16*(-fmc*hbcm*p3p6*p5p7-p3p4*p4p6*
     . p5p7))+w13*((p5p7*p3p6*p1p3)*(16*ffmcfmb-16)+16*(p5p7*p4p6)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+
     . hbcm2)-16*p1p3*p4p6*p5p7)+((p5p7*p3p6)*(-32*fb2*ffmcfmb+32*fb2
     . -8)+32*fb2*p4p6*p5p7))
      b(5)=ccc*(w1*(8*p5p7*(-fmb*fmc*hbcm2-2*fmc*hbcm3-fmc2*hbcm2)+8*
     . (p5p7*p3p4)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm-2*hbcm2)+16*(
     . fmc*hbcm*p1p3*p5p7+2*fmc*hbcm*p2p3*p5p7-2*fmc*hbcm*p3p5*p5p7+
     . hbcm2*p1p4*p5p7+p2p3*p3p4*p5p7-p3p4*p3p5*p5p7))+w13*(8*p5p7*(-
     . ffmcfmb*hbcm4-fmb*hbcm3+hbcm4)+16*(p5p7*p3p5)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+16*(p5p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+
     . 8*(p5p7*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+16*(p1p3*p2p3*
     . p5p7-p1p3*p3p5*p5p7))+(4*p5p7*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*
     . hbcm2-hbcm2)+32*(-fb2*p2p3*p5p7+fb2*p3p5*p5p7)))
      b(6)=ccc*(16*w1*(fmc*hbcm*p2p5*p3p6+hbcm2*p2p5*p4p6)+16*w13*(
     . p3p6*p2p5)*(ffmcfmb*hbcm2-hbcm2))
      b(7)=ccc*(8*w1*(-2*fmc*hbcm*p2p5-p2p5*p3p4)+w13*(8*p2p5*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)-8*p1p3*p2p5)+16*fb2*p2p5)
      b(8)=8*ccc*w1*(-fmc*hbcm2*p2p5-2*hbcm*p2p5*p3p4)
      b(9)=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm3-2*hbcm3)+8*(p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*p3p6*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-
     . fmc*hbcm2+fmc2*hbcm)+16*(hbcm*p1p3*p4p6-hbcm*p1p4*p3p6+hbcm*
     . p2p3*p4p6-hbcm*p3p4*p4p6-hbcm*p3p5*p4p6))+w13*(8*(p3p6*p1p3)*(
     . -ffmcfmb*hbcm-fmb+hbcm)+8*p3p6*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc
     . *hbcm2+fmb*hbcm2-fmc*hbcm2))+4*p3p6*(-4*fb1*ffmcfmb*hbcm+4*fb1
     . *hbcm+4*fb2*fmb+3*hbcm))
      b(10)=ccc*(w1*((p3p6*p3p4)*(-8*ffmcfmb+8)+8*p4p6*(ffmcfmb*hbcm2
     . +fmc*hbcm)+8*(fmc*hbcm*p3p6+p3p4*p4p6))+w13*((p3p6*p1p3)*(-8*
     . ffmcfmb+8)+8*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(
     . ffmcfmb*hbcm2-hbcm2)+8*p1p3*p4p6)+(p3p6*(16*fb2*ffmcfmb-16*fb2
     . +4)-16*fb2*p4p6))
      b(11)=ccc*(w1*(4*p3p4*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+2*
     . hbcm2)+4*(fmb*fmc*hbcm2-2*fmc*hbcm*p1p3-4*fmc*hbcm*p2p3+4*fmc*
     . hbcm*p3p5+2*fmc*hbcm3+fmc2*hbcm2-2*hbcm2*p1p4-2*p2p3*p3p4+2*
     . p3p4*p3p5))+w13*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p2p3*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p1p3*(-ffmcfmb*hbcm2-fmb*
     . hbcm+2*hbcm2)+4*(ffmcfmb*hbcm4+fmb*hbcm3-hbcm4-2*p1p3*p2p3+2*
     . p1p3*p3p5))+2*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+8*fb2*p2p3-
     . 8*fb2*p3p5+hbcm2))
      b(13)=ccc*(w1*(4*p3p4*(fmb-fmc-2*hbcm)+4*(-ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-2*fmc2*hbcm+2*hbcm*p1p4))+w13*(4*p1p3*(fmb-fmc)+4
     . *(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+fmc*hbcm2))+8*(-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp13_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(2)=ccc*(w1*(8*(p5p7*p3p4)*(-fmb+fmc)+8*fmc*hbcm2*p5p7)+16*
     . p5p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(5)=ccc*(w1*(8*p5p7*(fmb*fmc*hbcm2+fmc2*hbcm2-2*p3p4s2)-8*
     . hbcm2*p3p4*p5p7)+(16*p5p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*
     . hbcm2)+32*fb2*p3p4*p5p7))
      b(9)=ccc*(w1*(8*(p3p6*p3p4)*(fmb-fmc)-8*fmc*hbcm2*p3p6)+16*p3p6
     . *(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      b(11)=ccc*(4*w1*(-fmb*fmc*hbcm2-fmc2*hbcm2+hbcm2*p3p4+2*p3p4s2)
     . +8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm-fb2*hbcm2-2*fb2*p3p4))
      b(13)=ccc*(w1*(4*p3p4*(-fmb+fmc)+4*fmc*hbcm2)+8*(-2*fb1*hbcm+
     . fb2*fmb-fb2*fmc))
      do 200 n=1,16 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp12_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      ans3=(4*p6p7*(-4*fb1*fmb*fmc*hbcm+4*fb1*fmc2*hbcm-4*fb2*ffmcfmb
     . *fmb*hbcm2+4*fb2*ffmcfmb*fmc*hbcm2+fmb*hbcm2-fmc*hbcm2)+8*(
     . p5p6*p3p7)*(4*fb1*ffmcfmb*hbcm-4*fb2*fmc+3*hbcm)+8*(p4p6*p3p7)
     . *(4*fb1*ffmcfmb*hbcm-4*fb2*fmc+3*hbcm)+8*(p3p7*p3p6)*(4*fb1*
     . ffmcfmb*hbcm-4*fb2*fmc+3*hbcm))
      ans2=w15*(16*(p6p7*p2p5)*(fmc*hbcm2-hbcm3)+16*(p6p7*p1p2)*(-fmc
     . *hbcm2+hbcm3)+16*(p5p7*p5p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(
     . p5p7*p4p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb
     . *hbcm3+fmc*hbcm2)+16*(p5p6*p3p7*p3p5)*(ffmcfmb*hbcm-fmc)+16*(
     . p4p6*p3p7*p3p5)*(ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6*p3p5)*(
     . ffmcfmb*hbcm-fmc)+16*(p5p6*p3p7*p1p3)*(-ffmcfmb*hbcm+fmc)+16*(
     . p4p6*p3p7*p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p3p7*p3p6*p1p3)*(-
     . ffmcfmb*hbcm+fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm4+fmb*fmc*hbcm3)+8*
     . (p6p7*p3p5)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm
     . +fmc2*hbcm)+16*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2-fmb2*hbcm)+8*(
     . p6p7*p1p3)*(ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-
     . fmc2*hbcm)+16*(-ffmcfmb*fmb*hbcm2*p3p6*p3p7-ffmcfmb*fmb*hbcm2*
     . p3p7*p4p6-ffmcfmb*fmb*hbcm2*p3p7*p5p6-2*hbcm*p1p2*p2p3*p6p7+2*
     . hbcm*p1p2*p3p6*p3p7+2*hbcm*p1p2*p3p7*p4p6+2*hbcm*p1p2*p3p7*
     . p5p6+2*hbcm*p1p5*p2p3*p6p7+2*hbcm*p2p3*p2p5*p6p7+2*hbcm*p2p3*
     . p3p6*p5p7+2*hbcm*p2p3*p4p6*p5p7+2*hbcm*p2p3*p5p6*p5p7-2*hbcm*
     . p2p5*p3p6*p3p7-2*hbcm*p2p5*p3p7*p4p6-2*hbcm*p2p5*p3p7*p5p6))+
     . ans3
      ans1=w7*(16*(p5p6*p3p7*p2p3)*(ffmcfmb*hbcm-fmc)+16*(p4p6*p3p7*
     . p2p3)*(ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6*p2p3)*(ffmcfmb*hbcm-fmc
     . )+8*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*
     . hbcm+fmc2*hbcm))+ans2
      ans=ccc*ans1
      b(1)=ans
      b(2)=ccc*(8*w7*(p3p7*p2p3)*(ffmcfmb*fmb-ffmcfmb*fmc-ffmcfmb*
     . hbcm+fmc)+w15*(8*p5p7*(ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmc*
     . hbcm2-fmc2*hbcm)+8*(p3p7*p3p5)*(ffmcfmb*fmb-ffmcfmb*fmc-
     . ffmcfmb*hbcm+fmc)+8*(p3p7*p1p3)*(-ffmcfmb*fmb+ffmcfmb*fmc+
     . ffmcfmb*hbcm-fmc)+8*(ffmcfmb*fmb*hbcm2*p3p7-2*hbcm*p1p2*p3p7-2
     . *hbcm*p2p3*p5p7+2*hbcm*p2p5*p3p7))+4*p3p7*(-4*fb1*ffmcfmb*hbcm
     . +4*fb2*ffmcfmb*fmb-4*fb2*ffmcfmb*fmc+4*fb2*fmc+fmb-fmc-3*hbcm)
     . )
      b(3)=w15*ccc*(8*p5p6*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+8*p4p6*(
     . ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+8*p3p6*(ffmcfmb*fmb*hbcm2-fmb*
     . fmc*hbcm)+16*(-hbcm*p1p2*p3p6-hbcm*p1p2*p4p6-hbcm*p1p2*p5p6+
     . hbcm*p2p5*p3p6+hbcm*p2p5*p4p6+hbcm*p2p5*p5p6))
      ans2=((p5p6*p3p7)*(-32*fb2*ffmcfmb-8)+(p4p6*p3p7)*(-32*fb2*
     . ffmcfmb-8)+(p3p7*p3p6)*(-32*fb2*ffmcfmb-8)+4*p6p7*(4*fb1*fmc*
     . hbcm+4*fb2*ffmcfmb*hbcm2-hbcm2)+32*(-fb2*p2p3*p6p7-fb2*p2p4*
     . p6p7))
      ans1=w7*(16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p2p3)*
     . (ffmcfmb*hbcm2-fmc*hbcm)+16*(-ffmcfmb*p2p3*p3p6*p3p7-ffmcfmb*
     . p2p3*p3p7*p4p6-ffmcfmb*p2p3*p3p7*p5p6-p2p3*p2p4*p6p7-p2p3s2*
     . p6p7))+w15*(16*(p6p7*p2p5)*(-fmb*hbcm-fmc*hbcm+hbcm2)+16*(p6p7
     . *p1p2)*(fmb*hbcm+fmc*hbcm-hbcm2)+16*(p5p7*p5p6)*(ffmcfmb*hbcm2
     . -fmc*hbcm)+16*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*
     . p3p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p1p5)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p1p3
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+8*p6p7*(-ffmcfmb*fmb*fmc*hbcm2-
     . ffmcfmb*fmb2*hbcm2+fmb*fmc2*hbcm+fmb2*fmc*hbcm)+16*(ffmcfmb*
     . p1p3*p3p6*p3p7+ffmcfmb*p1p3*p3p7*p4p6+ffmcfmb*p1p3*p3p7*p5p6-
     . ffmcfmb*p3p5*p3p6*p3p7-ffmcfmb*p3p5*p3p7*p4p6-ffmcfmb*p3p5*
     . p3p7*p5p6-fmb*hbcm*p2p3*p6p7-2*fmb*hbcm*p2p4*p6p7+p1p3*p2p3*
     . p6p7+p1p3*p2p4*p6p7-p2p3*p3p5*p6p7-p2p4*p3p5*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans2=((p3p7*p3p5)*(-32*fb2*ffmcfmb-8)+(p3p7*p2p3)*(32*fb2*
     . ffmcfmb+8)+(p3p7*p1p3)*(32*fb2*ffmcfmb+8)+4*p5p7*(4*fb1*fmc*
     . hbcm+4*fb2*ffmcfmb*hbcm2-hbcm2)+4*p3p7*(-4*fb1*ffmcfmb*fmb*
     . hbcm-4*fb1*ffmcfmb*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*fmb*fmc+
     . 4*fb2*fmc2-3*fmb*hbcm-3*fmc*hbcm-hbcm2))
      ans1=w7*(8*(p5p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p2p3)*
     . (-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmb*fmc+fmc2
     . )+16*(-ffmcfmb*hbcm2*p1p2*p3p7+ffmcfmb*hbcm2*p2p5*p3p7+ffmcfmb
     . *p1p3*p2p3*p3p7-ffmcfmb*p2p3*p3p5*p3p7+ffmcfmb*p2p3s2*p3p7))+
     . w15*(8*(p5p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p7*p1p3)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*p5p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3+ffmcfmb*hbcm4-fmb*fmc*hbcm2-fmc*hbcm3-fmc2*hbcm2)+8*(
     . p3p7*p3p5)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+
     . fmb*fmc+fmc2)+8*(p3p7*p1p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm
     . +ffmcfmb*hbcm2-fmb*fmc-fmc2)+8*p3p7*(ffmcfmb*fmb*fmc*hbcm2+
     . ffmcfmb*fmb2*hbcm2-2*ffmcfmb*p1p3s2-2*ffmcfmb*p3p5s2)+16*(-
     . ffmcfmb*hbcm2*p1p5*p3p7-ffmcfmb*hbcm2*p2p3*p5p7-ffmcfmb*p1p3*
     . p2p3*p3p7+2*ffmcfmb*p1p3*p3p5*p3p7+ffmcfmb*p2p3*p3p5*p3p7-fmc*
     . hbcm*p1p2*p3p7+fmc*hbcm*p2p5*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w7*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p6*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*p2p3)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p1p2*
     . p2p3*p3p6-p2p3*p2p5*p3p6))+w15*(16*(p3p6*p2p5)*(-fmb*hbcm+fmc*
     . hbcm)+16*(p3p6*p1p2)*(fmb*hbcm-fmc*hbcm)+8*(p5p6*p3p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm
     . )+16*(p3p6*p1p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p6*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +8*p3p6*(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmb2*hbcm2-fmb*fmc2*
     . hbcm-fmb2*fmc*hbcm)+16*(fmb*hbcm*p2p3*p4p6+fmb*hbcm*p2p3*p5p6-
     . hbcm2*p1p2*p4p6-hbcm2*p1p2*p5p6+hbcm2*p2p5*p4p6+hbcm2*p2p5*
     . p5p6-p1p2*p1p3*p3p6+p1p2*p3p5*p3p6+p1p3*p2p5*p3p6-p2p5*p3p5*
     . p3p6))+(4*p5p6*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-hbcm2)+4*
     . p4p6*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-hbcm2)+32*(fb2*p1p2*
     . p3p6-fb2*p2p5*p3p6)))
      b(6)=ans
      b(7)=ccc*(w7*(8*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p2p3*(-ffmcfmb
     . *hbcm2+fmc*hbcm)+8*p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+8*(-p1p2*p2p3
     . +p2p3*p2p5))+w15*(8*p2p5*(fmb*hbcm-fmc*hbcm+hbcm2)+8*p1p2*(-
     . fmb*hbcm+fmc*hbcm-hbcm2)+4*p3p5*(-ffmcfmb*hbcm2+fmc*hbcm)+8*
     . p1p5*(ffmcfmb*hbcm2-fmc*hbcm)+4*p1p3*(ffmcfmb*hbcm2-fmc*hbcm)+
     . 4*(-ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmb2*hbcm2+fmb*fmc2*hbcm+2*
     . fmb*hbcm*p2p3+fmb2*fmc*hbcm+2*p1p2*p1p3-2*p1p2*p3p5-2*p1p3*
     . p2p5+2*p2p5*p3p5))+2*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-8*fb2
     . *p1p2+8*fb2*p2p5-hbcm2))
      b(8)=ccc*(4*w7*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*
     . fmc*hbcm-fmc2*hbcm)+w15*(8*p2p5*(-fmc*hbcm2+hbcm3)+8*p1p2*(fmc
     . *hbcm2-hbcm3)+4*p3p5*(ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-fmb*
     . fmc*hbcm-fmc2*hbcm)+8*p2p3*(-ffmcfmb*fmb*hbcm2+fmb2*hbcm)+4*
     . p1p3*(-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmc2*
     . hbcm)+4*(ffmcfmb*fmb*hbcm4-fmb*fmc*hbcm3+4*hbcm*p1p2*p2p3-4*
     . hbcm*p1p5*p2p3-4*hbcm*p2p3*p2p5))+2*(4*fb1*fmb*fmc*hbcm-4*fb1*
     . fmc2*hbcm+4*fb2*ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-fmb*
     . hbcm2+fmc*hbcm2))
      b(9)=ccc*(w7*(8*(p3p6*p2p3)*(fmb-fmc)+8*p3p6*(-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm))+w15*(8*(p3p6*
     . p3p5)*(fmb-fmc)+8*(p3p6*p1p3)*(-fmb+fmc)+8*p3p6*(-ffmcfmb*fmb*
     . hbcm2-fmb*fmc*hbcm+2*fmb2*hbcm)+8*(-fmb*hbcm2*p4p6-fmb*hbcm2*
     . p5p6+2*hbcm*p1p2*p3p6+2*hbcm*p1p3*p4p6+2*hbcm*p1p3*p5p6-4*hbcm
     . *p1p5*p3p6-2*hbcm*p2p5*p3p6-2*hbcm*p3p5*p4p6-2*hbcm*p3p5*p5p6)
     . )+16*p3p6*(fb2*fmb-fb2*fmc))
      b(10)=ccc*(8*w15*(-2*fmb*hbcm*p3p6-2*fmb*hbcm*p4p6-2*fmb*hbcm*
     . p5p6+p1p3*p3p6+p1p3*p4p6+p1p3*p5p6-p3p5*p3p6-p3p5*p4p6-p3p5*
     . p5p6)+w7*(8*p5p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p4p6*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*(-p2p3*p3p6-
     . p2p3*p4p6-p2p3*p5p6))+16*(-fb2*p3p6-fb2*p4p6-fb2*p5p6))
      b(11)=ccc*(w7*(8*p3p5*(ffmcfmb*hbcm2-fmc*hbcm)+4*p2p3*(-ffmcfmb
     . *hbcm2+fmc*hbcm-hbcm2)+8*p1p3*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(
     . ffmcfmb*hbcm4-fmc*hbcm3+2*p1p3*p2p3-2*p2p3*p3p5+2*p2p3s2))+w15
     . *(4*p3p5*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-hbcm2)+4*p1p3*(-
     . ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm+hbcm2)+4*(fmb*fmc*hbcm2+2*
     . fmb*hbcm*p2p3-2*fmb*hbcm3+fmb2*hbcm2+2*hbcm2*p1p2-2*hbcm2*p1p5
     . -2*hbcm2*p2p5-2*p1p3*p2p3+4*p1p3*p3p5-2*p1p3s2+2*p2p3*p3p5-2*
     . p3p5s2))+2*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2+8*
     . fb2*p1p3+8*fb2*p2p3-8*fb2*p3p5+hbcm2))
      b(12)=ccc*(w7*(8*(p3p7*p2p3)*(-ffmcfmb*hbcm+fmc)+8*p3p7*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2))+w15*(8*p5p7*(ffmcfmb*
     . hbcm3-hbcm3)+8*(p3p7*p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p3p7*p1p3)*(
     . ffmcfmb*hbcm-fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb2*
     . hbcm)+16*(-hbcm*p1p2*p3p7+hbcm*p1p5*p3p7+hbcm*p2p5*p3p7))+4*
     . p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm))
      b(13)=ccc*(w7*(4*p2p3*(-fmb+fmc)+4*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+w15*(4*p3p5*(-fmb+fmc-2*
     . hbcm)+4*p1p3*(fmb-fmc+2*hbcm)+4*(ffmcfmb*fmb*hbcm2+fmb*fmc*
     . hbcm-fmb*hbcm2-2*fmb2*hbcm-2*hbcm*p1p2+4*hbcm*p1p5+2*hbcm*p2p5
     . ))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(w15*(8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(
     . ffmcfmb*p1p3*p3p7-ffmcfmb*p3p5*p3p7-fmb*hbcm*p3p7+p1p3*p5p7-
     . p3p5*p5p7))+w7*(8*p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*(ffmcfmb*
     . hbcm2*p3p7-ffmcfmb*p2p3*p3p7-p2p3*p5p7))+(p3p7*(-16*fb2*
     . ffmcfmb-4)-16*fb2*p5p7))
      ans=ccc*(w7*(8*p6p7*(-ffmcfmb*hbcm4+fmc*hbcm3)+16*(p5p7*p3p6)*(
     . -ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+16*(ffmcfmb
     . *hbcm2*p3p7*p4p6+ffmcfmb*hbcm2*p3p7*p5p6+ffmcfmb*p2p3*p3p6*
     . p3p7+p2p3*p3p4*p6p7+p2p3*p3p6*p5p7))+w15*(8*p6p7*(fmb*fmc*
     . hbcm2+2*fmb*hbcm3+fmb2*hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+
     . fmb*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+
     . hbcm2)+8*(p6p7*p1p3)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-hbcm2
     . )+16*(-ffmcfmb*p1p3*p3p6*p3p7+ffmcfmb*p3p5*p3p6*p3p7-fmb*hbcm*
     . p2p3*p6p7+2*fmb*hbcm*p3p4*p6p7-fmb*hbcm*p3p7*p4p6-fmb*hbcm*
     . p3p7*p5p6+hbcm2*p1p2*p6p7-hbcm2*p1p5*p6p7-hbcm2*p2p5*p6p7-
     . hbcm2*p4p6*p5p7-hbcm2*p5p6*p5p7-p1p3*p3p4*p6p7-p1p3*p3p6*p5p7+
     . p3p4*p3p5*p6p7+p3p5*p3p6*p5p7))+((p3p7*p3p6)*(32*fb2*ffmcfmb+8
     . )+4*p6p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+
     . hbcm2)+32*(fb2*p3p4*p6p7+fb2*p3p6*p5p7)))
      b(15)=ans
      b(16)=ccc*(w7*(8*(p6p7*p2p3)*(fmb-fmc)+8*p6p7*(-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm))+w15*(8*(p6p7*
     . p3p5)*(fmb-fmc+2*hbcm)+8*(p6p7*p1p3)*(-fmb+fmc-2*hbcm)+8*p6p7*
     . (-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)+16*(
     . hbcm*p1p2*p6p7-2*hbcm*p1p5*p6p7-hbcm*p2p5*p6p7-hbcm*p3p6*p5p7-
     . hbcm*p4p6*p5p7-hbcm*p5p6*p5p7))+16*p6p7*(fb2*fmb-fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp11_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*
     . p2p3))
      ans2=w8*(16*p6p7*(fmb*p2p3s2-fmc*p2p3s2)+16*(p5p6*p3p7*p2p3)*(
     . ffmcfmb*hbcm+fmb-hbcm)+16*(p4p6*p3p7*p2p3)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p3p7*p3p6*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p6p7*p2p3
     . )*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+fmc*hbcm2))+(32*(p6p7*p2p3)*(-fb2*fmb+fb2*fmc)+4*
     . p6p7*(4*fb1*fmb*fmc*hbcm-4*fb1*fmb2*hbcm-4*fb2*ffmcfmb*fmb*
     . hbcm2+4*fb2*ffmcfmb*fmc*hbcm2+4*fb2*fmb*hbcm2-4*fb2*fmc*hbcm2+
     . fmb*hbcm2-fmc*hbcm2)+8*(p5p6*p3p7)*(4*fb1*ffmcfmb*hbcm-4*fb1*
     . hbcm-4*fb2*fmb-3*hbcm)+8*(p4p6*p3p7)*(4*fb1*ffmcfmb*hbcm-4*fb1
     . *hbcm-4*fb2*fmb-3*hbcm)+8*(p3p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*
     . fb1*hbcm-4*fb2*fmb-3*hbcm))
      ans1=w1*(16*(p6p7*p3p4*p2p3)*(fmb-fmc)+16*(p6p7*p2p4)*(-fmb*
     . hbcm2+fmc*hbcm2-hbcm3)+16*(p5p6*p4p7)*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+16*(p4p7*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+
     . 16*(p4p7*p3p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p5p6*p3p7*
     . p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p4p6*p3p7*p3p4)*(ffmcfmb*
     . hbcm+fmb-hbcm)+16*(p3p7*p3p6*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*
     . p6p7*(ffmcfmb*fmc*hbcm4+fmb*fmc*hbcm3-fmc*hbcm4)+16*(p6p7*p3p5
     . )*(ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc*hbcm2)+16*(p6p7*p2p3)*(-
     . ffmcfmb*fmc*hbcm2+fmc*hbcm2-fmc2*hbcm)+16*(p6p7*p1p3)*(-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+8*(p6p7*p3p4)*(-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-
     . fmb2*hbcm-fmc*hbcm2)+16*(fmb*fmc*hbcm*p3p6*p3p7+fmb*fmc*hbcm*
     . p3p7*p4p6+fmb*fmc*hbcm*p3p7*p5p6-fmc*hbcm2*p1p2*p6p7+fmc*hbcm2
     . *p2p5*p6p7-2*hbcm*p1p2*p3p4*p6p7+2*hbcm*p1p3*p2p4*p6p7+2*hbcm*
     . p2p3*p2p4*p6p7-2*hbcm*p2p4*p3p5*p6p7+2*hbcm*p2p5*p3p4*p6p7))+
     . ans2
      ans=ccc*ans1
      b(1)=ans
      b(2)=ccc*(8*w8*(p3p7*p2p3)*(-ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*
     . hbcm-fmc+hbcm)+w1*(8*p5p7*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc
     . *hbcm2)+8*(p3p7*p3p4)*(-ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-
     . fmc+hbcm)+8*p4p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+8*(-fmb*fmc*hbcm
     . *p3p7+2*hbcm*p1p2*p4p7+2*hbcm*p2p4*p5p7-2*hbcm*p2p5*p4p7))+4*
     . p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm+4*fb2*ffmcfmb*fmb-4*fb2*
     . ffmcfmb*fmc+4*fb2*fmc+fmb-fmc+3*hbcm))
      b(3)=w1*ccc*(8*p5p6*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)
     . +8*p4p6*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+8*p3p6*(-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+16*(hbcm*p2p4*p3p6+
     . hbcm*p2p4*p4p6+hbcm*p2p4*p5p6))
      ans=ccc*(w8*((p5p6*p3p7*p2p3)*(16*ffmcfmb-16)+(p4p6*p3p7*p2p3)*
     . (16*ffmcfmb-16)+(p3p7*p3p6*p2p3)*(16*ffmcfmb-16)+16*(p6p7*p2p4
     . )*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+16*p2p3*p2p4*p6p7)+w1*(16*(p6p7*p2p4)*(-fmb*
     . hbcm+fmc*hbcm+hbcm2)+(p5p6*p3p7*p3p4)*(16*ffmcfmb-16)+(p4p6*
     . p3p7*p3p4)*(16*ffmcfmb-16)+(p3p7*p3p6*p3p4)*(16*ffmcfmb-16)+16
     . *(p5p6*p4p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p7*p4p6)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p7*p3p6)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+8*(p6p7*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8
     . *p6p7*(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm2-fmb*fmc*hbcm2+
     . fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc2*hbcm2)+16*(fmc*hbcm*p2p3*p6p7
     . +p2p4*p3p4*p6p7))+((p5p6*p3p7)*(-32*fb2*ffmcfmb+32*fb2-8)+(
     . p4p6*p3p7)*(-32*fb2*ffmcfmb+32*fb2-8)+(p3p7*p3p6)*(-32*fb2*
     . ffmcfmb+32*fb2-8)+4*p6p7*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4
     . *fb2*hbcm2-hbcm2)-32*fb2*p2p4*p6p7))
      b(4)=ans
      ans2=w8*(16*p3p7*(-ffmcfmb*p2p3s2+p2p3s2)+(p3p7*p3p5*p2p3)*(16*
     . ffmcfmb-16)+(p3p7*p2p3*p1p3)*(-16*ffmcfmb+16)+8*(p5p7*p2p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p7*p2p3)*(-ffmcfmb*fmb*hbcm
     . -ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm
     . -hbcm2)+16*(-fmb*hbcm*p1p2*p3p7+fmb*hbcm*p2p5*p3p7-p1p2*p2p3*
     . p3p7+p2p3*p2p5*p3p7-p2p3s2*p5p7))+((p3p7*p3p5)*(-32*fb2*
     . ffmcfmb+32*fb2-8)+(p3p7*p2p3)*(32*fb2*ffmcfmb-32*fb2+8)+(p3p7*
     . p1p3)*(32*fb2*ffmcfmb-32*fb2+8)+4*p5p7*(4*fb1*fmb*hbcm+4*fb2*
     . ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2)+4*p3p7*(-4*fb1*ffmcfmb*fmb*
     . hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmb*hbcm+4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2+4*fb2*fmb*fmc+4*fb2*fmb2+4*fb2*hbcm2+3*fmb*
     . hbcm+3*fmc*hbcm-hbcm2)+32*(fb2*p1p2*p3p7+fb2*p2p3*p5p7-fb2*
     . p2p5*p3p7))
      ans1=w1*(8*p3p7*(-fmb*fmc2*hbcm-fmb2*fmc*hbcm)+(p3p7*p3p5*p3p4)
     . *(16*ffmcfmb-16)+(p3p7*p3p4*p2p3)*(-16*ffmcfmb+16)+(p3p7*p3p4*
     . p1p3)*(-16*ffmcfmb+16)+16*(p4p7*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm
     . +hbcm2)+8*(p5p7*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*
     . p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p1p3)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmb*hbcm3+fmb2*hbcm2-fmc*
     . hbcm3+hbcm4)+8*(p3p7*p3p4)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm
     . +ffmcfmb*hbcm2-fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm-hbcm2)+16*(-fmc*
     . hbcm*p1p2*p3p7-fmc*hbcm*p2p3*p5p7+fmc*hbcm*p2p5*p3p7+hbcm2*
     . p1p2*p4p7+hbcm2*p2p4*p5p7-hbcm2*p2p5*p4p7-p1p2*p3p4*p3p7-p2p3*
     . p3p4*p5p7+p2p5*p3p4*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w8*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p5p6*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p6*p2p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+16*(-p1p2*p2p3*p3p6+p2p3*p2p5*p3p6-p2p3s2*p4p6
     . -p2p3s2*p5p6))+w1*(16*(p3p6*p2p4)*(fmb*hbcm+fmc*hbcm)+8*(p5p6*
     . p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p6*p3p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(-ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*
     . fmc2*hbcm2+fmb*fmc*hbcm2-fmb*fmc2*hbcm-fmb2*fmc*hbcm+fmc2*
     . hbcm2)+16*(-2*fmc*hbcm*p1p2*p3p6-fmc*hbcm*p2p3*p4p6-fmc*hbcm*
     . p2p3*p5p6+2*fmc*hbcm*p2p5*p3p6+hbcm2*p2p4*p4p6+hbcm2*p2p4*p5p6
     . -p1p2*p3p4*p3p6-p2p3*p3p4*p4p6-p2p3*p3p4*p5p6+p2p5*p3p4*p3p6))
     . +(4*p5p6*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2
     . )+4*p4p6*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2
     . )+32*(fb2*p1p2*p3p6+fb2*p2p3*p4p6+fb2*p2p3*p5p6-fb2*p2p5*p3p6)
     . ))
      b(6)=ans
      b(7)=ccc*(w8*(8*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p3*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p1p2*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*(p1p2*p2p3-p2p3*p2p5-p2p3s2))+w1*(8*p2p4*(-fmb*hbcm-
     . fmc*hbcm+hbcm2)+4*p3p4*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(
     . ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm2-fmb*fmc*hbcm2+fmb*
     . fmc2*hbcm+fmb2*fmc*hbcm+4*fmc*hbcm*p1p2-2*fmc*hbcm*p2p3-4*fmc*
     . hbcm*p2p5-fmc2*hbcm2+2*p1p2*p3p4-2*p2p3*p3p4-2*p2p5*p3p4))+2*(
     . 4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-8*fb2*p1p2+8*
     . fb2*p2p3+8*fb2*p2p5-hbcm2))
      ans=ccc*(w1*(8*(p3p4*p2p3)*(-fmb+fmc)+8*p2p4*(fmb*hbcm2-fmc*
     . hbcm2+hbcm3)+8*p3p5*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2
     . )+8*p2p3*(ffmcfmb*fmc*hbcm2-fmc*hbcm2+fmc2*hbcm)+8*p1p3*(
     . ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc*hbcm2)+4*p3p4*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*
     . hbcm2)+4*(-ffmcfmb*fmc*hbcm4-fmb*fmc*hbcm3+2*fmc*hbcm2*p1p2-2*
     . fmc*hbcm2*p2p5+fmc*hbcm4+4*hbcm*p1p2*p3p4-4*hbcm*p1p3*p2p4-4*
     . hbcm*p2p3*p2p4+4*hbcm*p2p4*p3p5-4*hbcm*p2p5*p3p4))+w8*(4*p2p3*
     . (-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-
     . fmb2*hbcm-fmc*hbcm2)+8*(-fmb*p2p3s2+fmc*p2p3s2))+(16*p2p3*(fb2
     . *fmb-fb2*fmc)+2*(-4*fb1*fmb*fmc*hbcm+4*fb1*fmb2*hbcm+4*fb2*
     . ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-4*fb2*fmb*hbcm2+4*
     . fb2*fmc*hbcm2-fmb*hbcm2+fmc*hbcm2)))
      b(8)=ans
      b(9)=ccc*(w1*(8*(p3p6*p3p4)*(-fmb+fmc)+8*p3p6*(ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm-fmc*hbcm2+2*fmc2*hbcm)+8*(-fmc*hbcm2*p4p6-
     . fmc*hbcm2*p5p6-2*hbcm*p2p4*p3p6-2*hbcm*p3p4*p4p6-2*hbcm*p3p4*
     . p5p6))+w8*(8*(p3p6*p2p3)*(-fmb+fmc)+8*p3p6*(-ffmcfmb*fmb*hbcm2
     . +ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)
     . )+16*p3p6*(fb2*fmb-fb2*fmc))
      b(10)=ccc*(8*w1*(2*fmc*hbcm*p3p6+2*fmc*hbcm*p4p6+2*fmc*hbcm*
     . p5p6+p3p4*p3p6+p3p4*p4p6+p3p4*p5p6)+w8*(8*p5p6*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p2p3*p3p6+p2p3*p4p6+p2p3*p5p6
     . ))+16*(-fb2*p3p6-fb2*p4p6-fb2*p5p6))
      b(11)=ccc*(w1*(4*p3p4*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+2*
     . hbcm2)+4*(fmb*fmc*hbcm2-4*fmc*hbcm*p1p3-2*fmc*hbcm*p2p3+4*fmc*
     . hbcm*p3p5+2*fmc*hbcm3+fmc2*hbcm2-2*hbcm2*p2p4-2*p1p3*p3p4+2*
     . p3p4*p3p5))+w8*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p2p3*(
     . -ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2)+8*p1p3*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+4*(ffmcfmb*hbcm4+fmb*hbcm3-hbcm4-2*p1p3*p2p3+2*
     . p2p3*p3p5))+2*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+8*fb2*p1p3-
     . 8*fb2*p3p5+hbcm2))
      b(12)=ccc*(w1*(8*p4p7*(ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)+8*(p3p7
     . *p3p4)*(-ffmcfmb*hbcm-fmc+hbcm)+8*(fmc*hbcm2*p5p7-fmc2*hbcm*
     . p3p7+2*hbcm*p1p3*p4p7+2*hbcm*p2p3*p4p7+2*hbcm*p3p4*p5p7-2*hbcm
     . *p3p5*p4p7))+w8*(8*p3p7*(-fmb*fmc*hbcm+fmb2*hbcm)+8*(p3p7*p2p3
     . )*(-ffmcfmb*hbcm-fmc+hbcm))+4*p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb1*
     . hbcm+4*fb2*fmc+3*hbcm))
      b(13)=ccc*(w1*(4*p3p4*(fmb-fmc-2*hbcm)+4*(-ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-2*fmc2*hbcm+2*hbcm*p2p4))+w8*(4*p2p3*(fmb-fmc)+4*
     . (ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+fmc*hbcm2))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(w8*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(ffmcfmb
     . *p2p3*p3p7+fmb*hbcm*p3p7+p2p3*p5p7))+w1*(8*p4p7*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(ffmcfmb*p3p4*p3p7+fmc*hbcm*p3p7+2*fmc*hbcm*
     . p5p7+p3p4*p5p7))+(p3p7*(-16*fb2*ffmcfmb-4)-16*fb2*p5p7))
      ans=ccc*(w1*(8*p6p7*(fmb*fmc*hbcm2-2*fmc*hbcm3+fmc2*hbcm2-2*
     . p3p4s2)+(p3p7*p3p6*p3p4)*(-16*ffmcfmb+16)+16*(p4p7*p3p6)*(
     . ffmcfmb*hbcm2-fmc*hbcm-hbcm2)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm2+
     . fmb*hbcm-2*fmc*hbcm)+16*(fmc*hbcm*p2p3*p6p7-2*fmc*hbcm*p3p6*
     . p5p7+fmc*hbcm*p3p7*p4p6+fmc*hbcm*p3p7*p5p6-hbcm2*p2p4*p6p7-
     . hbcm2*p4p6*p4p7-hbcm2*p4p7*p5p6+p2p3*p3p4*p6p7-p3p4*p3p6*p5p7+
     . p3p4*p3p7*p4p6+p3p4*p3p7*p5p6))+w8*((p3p7*p3p6*p2p3)*(-16*
     . ffmcfmb+16)+8*p6p7*(-ffmcfmb*hbcm4-fmb*hbcm3+hbcm4+2*p2p3s2)+
     . 16*(p5p7*p3p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p3p4)*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-2*hbcm2)+16*(fmb*hbcm*p3p7*p4p6+fmb*hbcm*p3p7*p5p6-
     . p2p3*p3p4*p6p7-p2p3*p3p6*p5p7+p2p3*p3p7*p4p6+p2p3*p3p7*p5p6))+
     . ((p3p7*p3p6)*(32*fb2*ffmcfmb-32*fb2+8)+4*p6p7*(-4*fb1*fmb*hbcm
     . -4*fb2*ffmcfmb*hbcm2+8*fb2*hbcm2+hbcm2)+32*(-fb2*p2p3*p6p7+fb2
     . *p3p4*p6p7+fb2*p3p6*p5p7-fb2*p3p7*p4p6-fb2*p3p7*p5p6)))
      b(15)=ans
      b(16)=ccc*(w1*(8*(p6p7*p3p4)*(-fmb+fmc+2*hbcm)+8*p6p7*(ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+16*(-hbcm*p2p4*p6p7-hbcm*
     . p3p6*p4p7-hbcm*p4p6*p4p7-hbcm*p4p7*p5p6))+w8*(8*(p6p7*p2p3)*(-
     . fmb+fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*
     . hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2))+16*p6p7*(fb2*fmb-fb2*fmc)
     . )
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp19_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      ans4=8*p6p7*(-ffmcfmb*fmb*fmc*hbcm3-2*ffmcfmb*fmc*hbcm4+ffmcfmb
     . *fmc2*hbcm3+2*ffmcfmb*hbcm*p2p3s2+fmb*fmc2*hbcm2-2*fmb*p2p3s2+
     . 2*fmc2*hbcm3-fmc3*hbcm2)+8*(p6p7*p3p4)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm-2*fmc*hbcm2+2*fmc2*hbcm)+8*(p6p7*p2p3)*(-ffmcfmb**2*hbcm3
     . +ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+2*fmc*hbcm2-
     . 6*fmc2*hbcm)+16*(-2*ffmcfmb*hbcm*p1p2*p3p6*p3p7-2*ffmcfmb*hbcm
     . *p1p3*p3p7*p4p6+2*ffmcfmb*hbcm*p2p5*p3p6*p3p7+2*ffmcfmb*hbcm*
     . p3p5*p3p7*p4p6-fmc*hbcm2*p1p2*p6p7-2*hbcm*p1p2*p3p4*p6p7-2*
     . hbcm*p1p2*p3p6*p4p7+2*hbcm*p1p2*p3p7*p4p6+2*hbcm*p1p3*p2p4*
     . p6p7+2*hbcm*p1p4*p2p3*p6p7+4*hbcm*p2p3*p2p4*p6p7-hbcm*p2p3*
     . p2p5*p6p7+4*hbcm*p2p3*p3p6*p4p7+4*hbcm*p2p3*p4p6*p4p7+2*hbcm*
     . p2p3*p4p6*p5p7-4*hbcm*p2p4*p3p5*p6p7-4*hbcm*p2p4*p3p6*p3p7-2*
     . hbcm*p2p4*p3p6*p5p7-4*hbcm*p2p4*p3p7*p4p6+3*hbcm*p2p5*p3p4*
     . p6p7+2*hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*p4p6)
      ans3=16*(p6p7*p2p4)*(-fmb*hbcm2+fmc*hbcm2-2*hbcm3)+16*(p5p7*
     . p4p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p4p7*p4p6)*(-ffmcfmb*
     . hbcm3+fmc*hbcm2)+16*(p6p7*p2p5)*(ffmcfmb*hbcm3+fmc*hbcm2)+16*(
     . p6p7*p1p4)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p4p6*p3p7*p3p4)*(3*
     . ffmcfmb*hbcm-fmc)+16*(p5p7*p3p6*p3p4)*(ffmcfmb*hbcm-fmc)+16*(
     . p6p7*p3p5*p3p4)*(ffmcfmb*hbcm-fmc)+16*(p4p6*p3p7*p2p3)*(-
     . ffmcfmb*hbcm+fmc)+16*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+fmc)+16*(
     . p6p7*p3p5*p2p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p3p4*p2p3)*(-
     . ffmcfmb*hbcm+fmb)+16*(p3p7*p3p6*p3p4)*(2*ffmcfmb*fmc+3*ffmcfmb
     . *hbcm-2*fmc)+16*(p3p7*p3p6*p2p3)*(-2*ffmcfmb*fmc+ffmcfmb*hbcm+
     . 2*fmc)+16*(p4p6*p3p7)*(-2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmc*
     . hbcm2+fmc2*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*fmc*hbcm2+fmc2*hbcm)
     . +16*(p4p7*p3p6)*(-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmc*hbcm2+
     . fmc2*hbcm)+16*(p3p7*p3p6)*(-ffmcfmb*fmc*hbcm2+ffmcfmb*fmc2*
     . hbcm+2*fmc2*hbcm)+32*(p6p7*p3p5)*(-ffmcfmb*fmc*hbcm2+fmc2*hbcm
     . )+16*(p6p7*p1p3)*(ffmcfmb*fmc*hbcm2-fmc2*hbcm)+ans4
      ans2=w11*ans3
      ans7=8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb
     . *fmc*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmc*hbcm2)-16*hbcm*
     . p1p3*p2p5*p6p7
      ans6=16*(p6p7*p2p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(p3p7*
     . p3p6*p2p5)*(ffmcfmb*hbcm-hbcm)+32*(p5p7*p3p6*p2p3)*(-ffmcfmb*
     . hbcm+hbcm)+16*(p4p6*p3p7*p1p3)*(ffmcfmb*hbcm-fmc)+16*(p5p7*
     . p3p6*p1p3)*(ffmcfmb*hbcm-fmc)+16*(p6p7*p3p5*p1p3)*(ffmcfmb*
     . hbcm-fmc)+16*(p6p7*p2p3*p1p3)*(-ffmcfmb*hbcm+fmb)+16*(p3p7*
     . p3p6*p1p3)*(2*ffmcfmb*fmc+ffmcfmb*hbcm-2*fmc)+8*p6p7*(ffmcfmb
     . **2*hbcm5+ffmcfmb*fmb*hbcm4-ffmcfmb*fmc*hbcm4-ffmcfmb*hbcm5-
     . fmb*fmc*hbcm3+fmc*hbcm4)+16*(p4p6*p3p7)*(ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2)+16*(p5p7*p3p6)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2
     . -ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(
     . p3p7*p3p6)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmb2*hbcm-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc*hbcm2)+16*(p6p7*p3p5)*(ffmcfmb**2*hbcm3+ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*
     . hbcm2)+16*(p6p7*p2p3)*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3-fmb*
     . hbcm2+fmb2*hbcm)+ans7
      ans5=w13*ans6
      ans8=(8*(p4p6*p3p7)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p5p7
     . *p3p6)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p3p7*p3p6)*(4*
     . fb1*ffmcfmb*hbcm-8*fb2*ffmcfmb*fmc+8*fb2*fmc-2*fmc-3*hbcm)+8*(
     . p6p7*p3p5)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p6p7*p2p3)*(
     . -4*fb1*ffmcfmb*hbcm-4*fb2*fmb+hbcm)+4*p6p7*(4*fb1*ffmcfmb**2*
     . hbcm3+4*fb1*fmb*fmc*hbcm+4*fb2*ffmcfmb*fmb*hbcm2+4*fb2*ffmcfmb
     . *fmc*hbcm2-fmb*hbcm2-fmc*hbcm2)-32*fb1*hbcm*p2p5*p6p7)
      ans1=w7*(16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*p6p7*(
     . ffmcfmb*hbcm*p2p3s2-fmb*p2p3s2)+16*(p4p6*p3p7*p2p3)*(ffmcfmb*
     . hbcm+fmc)+16*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*
     . p3p5*p2p3)*(-ffmcfmb*hbcm+fmc)+16*(p3p7*p3p6*p2p3)*(-2*ffmcfmb
     . *fmc+ffmcfmb*hbcm+2*fmc)+8*(p6p7*p2p3)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+
     . 16*(-2*ffmcfmb*hbcm*p1p2*p3p6*p3p7+2*ffmcfmb*hbcm*p2p5*p3p6*
     . p3p7-hbcm*p2p3*p2p5*p6p7))+ans2+ans5+ans8
      ans=ccc*ans1
      b(1)=ans
      ans2=w13*(16*(p5p7*p1p3)*(-ffmcfmb*hbcm+fmc)+8*(p3p7*p1p3)*(
     . ffmcfmb*fmb-ffmcfmb*fmc-ffmcfmb*hbcm+fmc)+16*p5p7*(-ffmcfmb**2
     . *hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-fmc*hbcm2)+8*p3p7*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*
     . fmc*hbcm2-ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm-2*ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmb2*hbcm+2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-fmc*hbcm2))+(8*p5p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+
     . hbcm)+4*p3p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmb+4*fb2*
     . ffmcfmb*fmc-4*fb2*fmc-fmb+fmc+hbcm))
      ans1=w7*(16*(p5p7*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p3p7*p2p3)*(-
     . ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+16*(ffmcfmb*hbcm*
     . p1p2*p3p7-2*ffmcfmb*hbcm*p2p5*p3p7))+w11*(16*(p5p7*p3p4)*(-
     . ffmcfmb*hbcm+fmc)+16*(p5p7*p2p3)*(ffmcfmb*hbcm-fmc)+24*p5p7*(
     . ffmcfmb*fmc*hbcm2-fmc2*hbcm)+8*(p3p7*p3p4)*(ffmcfmb*fmb-
     . ffmcfmb*fmc-3*ffmcfmb*hbcm+fmc)+8*(p3p7*p2p3)*(-ffmcfmb*fmb+
     . ffmcfmb*fmc-ffmcfmb*hbcm-fmc)+8*p4p7*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm)+8*p3p7*
     . (-ffmcfmb**2*fmc*hbcm2+ffmcfmb*fmb*fmc*hbcm+2*ffmcfmb*fmc*
     . hbcm2-2*ffmcfmb*fmc2*hbcm-2*fmc2*hbcm)+16*(ffmcfmb*hbcm*p1p2*
     . p3p7+ffmcfmb*hbcm*p1p4*p3p7-2*ffmcfmb*hbcm*p2p5*p3p7+2*hbcm*
     . p1p2*p4p7-2*hbcm*p2p3*p4p7+2*hbcm*p2p4*p3p7+3*hbcm*p2p4*p5p7-3
     . *hbcm*p2p5*p4p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w7*(16*(p4p6*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p3p6*p2p3)*(
     . ffmcfmb*hbcm-fmc))+w11*(16*(p4p6*p3p4)*(-ffmcfmb*hbcm+fmc)+8*(
     . p3p6*p3p4)*(-ffmcfmb*hbcm+fmc)+16*(p3p6*p2p4)*(ffmcfmb*hbcm+
     . hbcm)+16*(p4p6*p2p3)*(ffmcfmb*hbcm-fmc+hbcm)+8*(p3p6*p2p3)*(
     . ffmcfmb*hbcm-fmc)+8*p4p6*(-ffmcfmb**2*hbcm3+3*ffmcfmb*fmc*
     . hbcm2-2*fmc2*hbcm)+8*p3p6*(ffmcfmb**2*fmc*hbcm2+ffmcfmb*fmc*
     . hbcm2-ffmcfmb*fmc2*hbcm-fmc2*hbcm)+16*(-hbcm*p1p2*p4p6+3*hbcm*
     . p2p4*p4p6+hbcm*p2p5*p4p6))+w13*(32*(p3p6*p2p5)*(-ffmcfmb*hbcm+
     . hbcm)+16*(p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p1p3)*(-
     . ffmcfmb*hbcm+fmc)+8*(p3p6*p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p3p6*
     . p1p2)*(ffmcfmb*hbcm-hbcm)+16*p4p6*(-ffmcfmb**2*hbcm3-ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*
     . hbcm2)+8*p3p6*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*fmc*hbcm2-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmc2*hbcm+
     . ffmcfmb*hbcm3-fmc*hbcm2+fmc2*hbcm))+(8*p4p6*(-4*fb1*ffmcfmb*
     . hbcm-4*fb2*fmc+hbcm)+4*p3p6*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+
     . hbcm)))
      b(3)=ans
      ans3=w13*(32*(p6p7*p2p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(
     . p6p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p1p2)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p1p3)*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm-fmb*fmc-fmc2)+16*(p3p7*p3p6*p1p3)*(ffmcfmb**2
     . -ffmcfmb)+16*(p4p6*p3p7)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . ffmcfmb*hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p7*p3p6)*(-ffmcfmb**2*hbcm2
     . +ffmcfmb*hbcm2)+8*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*
     . hbcm3-ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-ffmcfmb*fmc*hbcm3-
     . ffmcfmb*fmc2*hbcm2+fmb*fmc*hbcm2-fmb*fmc2*hbcm-fmb2*fmc*hbcm+
     . fmc2*hbcm2)+16*(-ffmcfmb*p1p3*p3p7*p4p6-p1p2*p1p3*p6p7+p1p3*
     . p2p3*p6p7+2*p1p3*p2p4*p6p7))+((p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(
     . p3p7*p3p6)*(-32*fb2*ffmcfmb**2+32*fb2*ffmcfmb-16*ffmcfmb+8)+4*
     . p6p7*(4*fb1*ffmcfmb*fmb*hbcm+4*fb1*ffmcfmb*fmc*hbcm+4*fb2*fmb*
     . fmc+4*fb2*fmc2-fmb*hbcm-fmc*hbcm)+32*(fb2*p1p2*p6p7-fb2*p2p3*
     . p6p7-2*fb2*p2p4*p6p7))
      ans2=w11*(16*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+32*(p4p7*p4p6
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+16*(p6p7*p2p4)*(3*ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+16*(
     . p6p7*p1p4)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p4)*(ffmcfmb*fmb
     . *hbcm+ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-fmb*fmc+2*fmc*hbcm-fmc2
     . )+8*(p6p7*p2p3)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+2*ffmcfmb*
     . hbcm2+fmb*fmc+fmc2)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2-ffmcfmb)+
     . 16*(p3p7*p3p6*p2p3)*(-ffmcfmb**2+ffmcfmb)+16*(p4p6*p3p7)*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+8*
     . p6p7*(ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmb*fmc*hbcm2-3*ffmcfmb*
     . fmc2*hbcm2+fmb*fmc2*hbcm+2*fmc3*hbcm-2*p2p3s2)+16*(-ffmcfmb*
     . fmc*hbcm*p3p6*p3p7-ffmcfmb*hbcm2*p1p2*p6p7+ffmcfmb*p2p3*p3p7*
     . p4p6-ffmcfmb*p3p4*p3p7*p4p6+p1p2*p2p3*p6p7-p1p2*p3p4*p6p7-2*
     . p2p3*p2p4*p6p7+p2p3*p3p4*p6p7+2*p2p4*p3p4*p6p7))+ans3
      ans1=w7*(32*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p2)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p3)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+fmb*fmc-2*fmc*hbcm+fmc2)+16*(
     . p3p7*p3p6*p2p3)*(-ffmcfmb**2+ffmcfmb)+16*(ffmcfmb*p2p3*p3p7*
     . p4p6+p1p2*p2p3*p6p7-2*p2p3*p2p4*p6p7-p2p3s2*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans4=16*(ffmcfmb*fmc*hbcm*p1p3*p3p7-2*ffmcfmb*fmc*hbcm*p3p5*
     . p3p7+ffmcfmb*p2p3*p3p4*p3p7+ffmcfmb*p2p3*p3p5*p3p7-ffmcfmb*
     . p3p4*p3p5*p3p7+fmc*hbcm*p1p4*p3p7+hbcm2*p1p2*p4p7+2*hbcm2*p2p4
     . *p5p7-2*hbcm2*p2p5*p4p7+p1p2*p2p3*p3p7-p1p2*p3p4*p3p7-2*p2p3*
     . p2p5*p3p7-2*p2p3*p3p4*p5p7+2*p2p5*p3p4*p3p7)
      ans3=32*(p4p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+24*(p5p7*p3p4)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p2p5)*(ffmcfmb*hbcm2+fmc*hbcm
     . )+16*(p3p7*p2p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+hbcm2)+24
     . *(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p2p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+16*(p4p7*p1p3)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2-fmc*hbcm)+16*p5p7*(
     . ffmcfmb*fmc*hbcm3-fmc2*hbcm2+2*p2p3s2)+8*p4p7*(ffmcfmb*fmb*
     . hbcm3+ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4-fmb*fmc*hbcm2-2*fmc*
     . hbcm3-fmc2*hbcm2)+8*(p3p7*p3p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*
     . fmb*hbcm-3*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmc2)+8*(
     . p3p7*p2p3)*(ffmcfmb**2*hbcm2+3*ffmcfmb*fmc*hbcm-fmb*fmc-fmc2)+
     . 8*p3p7*(-ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmc*hbcm3+2*ffmcfmb*fmc2
     . *hbcm2-2*ffmcfmb*p2p3s2-fmb*fmc2*hbcm-fmc2*hbcm2-2*fmc3*hbcm)+
     . ans4
      ans2=w11*ans3
      ans6=((p3p7*p3p5)*(32*fb2*ffmcfmb+8)+(p3p7*p2p3)*(-32*fb2*
     . ffmcfmb-8)+4*p5p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+
     . 4*p3p7*(-4*fb1*ffmcfmb*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2-4*fb2*
     . fmb*fmc-4*fb2*fmc2+2*fmb*hbcm+fmc*hbcm)+32*(fb2*p1p2*p3p7+2*
     . fb2*p2p3*p5p7-2*fb2*p2p5*p3p7))
      ans5=w13*(32*(p3p7*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+32*(
     . p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p7*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+8*p5p7*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb
     . *fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p7*
     . p3p5)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+16*(
     . p3p7*p2p3)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+8
     . *(p3p7*p1p3)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm-2*ffmcfmb*
     . hbcm2+fmb*fmc+fmc2)+8*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*
     . fmc*hbcm3-ffmcfmb**2*hbcm4-ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*
     . hbcm3+ffmcfmb*fmc2*hbcm2+ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmb*fmc2*
     . hbcm+fmb2*fmc*hbcm-fmc2*hbcm2)+16*(ffmcfmb*p1p3*p2p3*p3p7-
     . ffmcfmb*p1p3*p3p5*p3p7-p1p2*p1p3*p3p7-2*p1p3*p2p3*p5p7+2*p1p3*
     . p2p5*p3p7))+ans6
      ans1=w7*(16*(p3p7*p2p5)*(ffmcfmb*hbcm2-2*fmc*hbcm)+24*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+8*(p3p7*p2p3)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm-
     . fmb*fmc-fmc2)+16*(ffmcfmb*p2p3*p3p5*p3p7-ffmcfmb*p2p3s2*p3p7+
     . p1p2*p2p3*p3p7-2*p2p3*p2p5*p3p7+2*p2p3s2*p5p7))+ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans4=(4*p4p6*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+4*p3p6
     . *(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*
     . hbcm2-4*fb2*ffmcfmb*hbcm2-4*fb2*fmb*fmc-4*fb2*fmc2+fmb*hbcm+
     . hbcm2)+32*(fb2*p1p2*p3p6+2*fb2*p2p3*p4p6-2*fb2*p2p5*p3p6))
      ans3=w13*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)+32*(
     . p4p6*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p6*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+8*p4p6*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb
     . *fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p6*
     . p3p5)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p3p6*p2p3)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . ffmcfmb*hbcm2+fmb*hbcm)+8*(p3p6*p1p3)*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm+fmc2)+8*p3p6*(
     . -ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4+
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc2*hbcm2-
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmb2*fmc*hbcm+fmc*
     . hbcm3-fmc2*hbcm2)+16*(-p1p2*p1p3*p3p6-2*p1p3*p2p3*p4p6+2*p1p3*
     . p2p5*p3p6))+ans4
      ans2=w11*(16*(p4p6*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+24*(p4p6*
     . p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p6*p2p5)*(2*ffmcfmb*
     . hbcm2+fmc*hbcm)+16*(p3p6*p2p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*
     . hbcm+hbcm2)+8*(p4p6*p2p3)*(-3*ffmcfmb*hbcm2-3*fmc*hbcm+2*hbcm2
     . )+16*(p3p6*p1p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p6*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2-fmc*
     . hbcm)+8*p4p6*(ffmcfmb**2*hbcm4-2*ffmcfmb*hbcm4+2*fmc*hbcm3-
     . fmc2*hbcm2+4*p2p3s2)+8*(p3p6*p3p4)*(-ffmcfmb**2*hbcm2-ffmcfmb*
     . fmb*hbcm+ffmcfmb*hbcm2+fmb*fmc-fmc*hbcm+fmc2)+8*(p3p6*p2p3)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*fmc+fmc*
     . hbcm-fmc2)+8*p3p6*(-ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*fmc*hbcm2
     . +ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*hbcm2-fmb*fmc2*hbcm-fmc2*
     . hbcm2-2*fmc3*hbcm)+16*(-hbcm2*p1p2*p4p6+2*hbcm2*p2p4*p4p6+
     . hbcm2*p2p5*p4p6+p1p2*p2p3*p3p6-p1p2*p3p4*p3p6-2*p2p3*p2p5*p3p6
     . -2*p2p3*p3p4*p4p6+2*p2p5*p3p4*p3p6))+ans3
      ans1=w7*(32*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+24*(p4p6*p2p3)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p6*p1p2)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+8*(p3p6*p2p3)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+2*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc*hbcm-fmc2)+16*(p1p2
     . *p2p3*p3p6-2*p2p3*p2p5*p3p6+2*p2p3s2*p4p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(24*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p2p3*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p1p2*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+4*p1p3*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+3*ffmcfmb*
     . fmc*hbcm+2*ffmcfmb*hbcm2-fmb*fmc-2*fmc*hbcm-2*fmc2)+4*(2*
     . ffmcfmb**2*fmb*hbcm3+2*ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4-
     . ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmb2*hbcm2-3*ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmb*fmc2*hbcm
     . -2*fmb2*fmc*hbcm+fmc*hbcm3+2*fmc2*hbcm2+4*p1p2*p1p3-4*p1p3*
     . p2p3-6*p1p3*p2p5))+2*(4*fb1*ffmcfmb*fmb*hbcm+4*fb1*ffmcfmb*fmc
     . *hbcm-4*fb2*ffmcfmb**2*hbcm2+4*fb2*fmb*fmc+8*fb2*fmc2-16*fb2*
     . p1p2+16*fb2*p2p3+24*fb2*p2p5+2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*
     . hbcm)
      ans1=w7*(24*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+16*p1p2*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*p2p3*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-3*
     . ffmcfmb*fmc*hbcm-4*ffmcfmb*hbcm2+fmb*fmc+4*fmc*hbcm+2*fmc2)+8*
     . (-2*p1p2*p2p3+3*p2p3*p2p5+2*p2p3s2))+w11*(8*p2p5*(-3*ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p2p4*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm)+8
     . *p1p4*(ffmcfmb*hbcm2-fmc*hbcm)+8*p1p2*(2*ffmcfmb*hbcm2+fmc*
     . hbcm)+4*p3p4*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc
     . *hbcm-2*ffmcfmb*hbcm2-fmb*fmc+2*fmc*hbcm-2*fmc2)+4*p2p3*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc*hbcm-4*ffmcfmb
     . *hbcm2+fmb*fmc-2*fmc*hbcm+2*fmc2)+4*(2*ffmcfmb**2*fmc*hbcm3-
     . ffmcfmb*fmb*fmc*hbcm2-5*ffmcfmb*fmc2*hbcm2+fmb*fmc2*hbcm+3*
     . fmc3*hbcm-4*p1p2*p2p3+4*p1p2*p3p4+6*p2p3*p2p5-4*p2p3*p3p4+4*
     . p2p3s2-6*p2p5*p3p4))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=(8*p3p5*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+4*p2p3*(4*fb1
     . *ffmcfmb*hbcm+4*fb1*hbcm+4*fb2*fmb-3*hbcm)+4*p1p3*(4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmc-hbcm)+2*(-8*fb1*ffmcfmb*hbcm3-4*fb1*fmb
     . *fmc*hbcm+4*fb1*fmc2*hbcm-8*fb1*hbcm*p1p2+16*fb1*hbcm*p2p5-4*
     . fb2*ffmcfmb*fmb*hbcm2+4*fb2*ffmcfmb*fmc*hbcm2-8*fb2*fmc*hbcm2+
     . fmb*hbcm2-fmc*hbcm2+2*hbcm3))
      ans3=w13*(16*p2p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*p1p2*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p3p5*p1p3)*(-ffmcfmb*hbcm+
     . fmc)+8*(p2p3*p1p3)*(3*ffmcfmb*hbcm-fmb-hbcm)+16*p3p5*(-ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+
     . fmb*fmc*hbcm-fmc*hbcm2)+16*p2p3*(ffmcfmb**2*hbcm3-ffmcfmb*
     . hbcm3+fmb*hbcm2-fmb2*hbcm)+4*p1p3*(2*ffmcfmb**2*hbcm3+3*
     . ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-4*ffmcfmb*hbcm3-3*fmb*
     . fmc*hbcm+4*fmc*hbcm2+fmc2*hbcm)+4*(ffmcfmb**2*fmb*hbcm4-
     . ffmcfmb**2*fmc*hbcm4-2*ffmcfmb**2*hbcm5-2*ffmcfmb*fmb*fmc*
     . hbcm3-3*ffmcfmb*fmb*hbcm4+ffmcfmb*fmb2*hbcm3+3*ffmcfmb*fmc*
     . hbcm4+ffmcfmb*fmc2*hbcm3+2*ffmcfmb*hbcm*p1p3s2+2*ffmcfmb*hbcm5
     . +3*fmb*fmc*hbcm3+fmb*fmc2*hbcm2-fmb2*fmc*hbcm2-2*fmc*hbcm4-2*
     . fmc*p1p3s2-fmc2*hbcm3-2*hbcm*p1p2*p1p3+4*hbcm*p1p3*p2p5))+ans4
      ans2=w11*(8*p2p5*(-2*ffmcfmb*hbcm3-fmc*hbcm2)+8*p2p4*(-ffmcfmb*
     . hbcm3+fmb*hbcm2-2*fmc*hbcm2+4*hbcm3)+8*p1p4*(ffmcfmb*hbcm3-fmc
     . *hbcm2)+8*p1p2*(ffmcfmb*hbcm3+fmc*hbcm2)+16*(p3p5*p3p4)*(-
     . ffmcfmb*hbcm+fmc)+16*(p3p5*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p3p4*
     . p2p3)*(3*ffmcfmb*hbcm-fmb-hbcm)+8*(p3p4*p1p3)*(ffmcfmb*hbcm-
     . fmc)+8*(p2p3*p1p3)*(-ffmcfmb*hbcm+fmc)+24*p3p5*(ffmcfmb*fmc*
     . hbcm2-fmc2*hbcm)+16*p1p3*(-ffmcfmb*fmc*hbcm2+fmc2*hbcm)+4*p3p4
     . *(2*ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-4*
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+4*fmc*hbcm2-3*fmc2*hbcm)+4*p2p3*(2*
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*
     . hbcm-4*fmc*hbcm2+9*fmc2*hbcm)+4*(-ffmcfmb**2*fmc*hbcm4+ffmcfmb
     . *fmb*fmc*hbcm3+4*ffmcfmb*fmc*hbcm4-ffmcfmb*fmc2*hbcm3-6*
     . ffmcfmb*hbcm*p2p3s2-fmb*fmc2*hbcm2+2*fmb*p2p3s2-4*fmc2*hbcm3+2
     . *fmc3*hbcm2-2*hbcm*p1p2*p2p3+6*hbcm*p1p2*p3p4-8*hbcm*p1p3*p2p4
     . -4*hbcm*p1p4*p2p3-12*hbcm*p2p3*p2p4+4*hbcm*p2p3*p2p5+2*hbcm*
     . p2p3s2+12*hbcm*p2p4*p3p5-8*hbcm*p2p5*p3p4))+ans3
      ans1=w7*(16*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+16*(p3p5*p2p3)*(ffmcfmb*hbcm-fmc)+8*(p2p3*
     . p1p3)*(-ffmcfmb*hbcm+fmc)+4*p2p3*(2*ffmcfmb**2*hbcm3-ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-3*fmc2*hbcm)+8*(-3*
     . ffmcfmb*hbcm*p2p3s2+fmb*p2p3s2-hbcm*p1p2*p2p3+2*hbcm*p2p3*p2p5
     . +hbcm*p2p3s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w7*(16*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p6*p2p3)*(
     . -2*ffmcfmb*hbcm+fmb-fmc+hbcm)+8*p3p6*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*
     . hbcm2-2*fmc2*hbcm)+16*hbcm*p2p3*p4p6)+w13*(8*(p3p6*p1p3)*(-fmb
     . +fmc+hbcm)+16*p4p6*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+32*(p3p6*
     . p3p5)*(ffmcfmb*hbcm-hbcm)+16*(p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+
     . 8*p3p6*(ffmcfmb**2*hbcm3+ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-
     . 2*fmb2*hbcm-2*hbcm3)+16*hbcm*p1p3*p4p6)+w11*(8*p4p6*(-fmc*
     . hbcm2-2*hbcm3)+8*(p3p6*p3p4)*(-fmb+fmc+hbcm)+8*(p3p6*p2p3)*(-2
     . *ffmcfmb*hbcm+fmb-fmc+hbcm)+8*p3p6*(ffmcfmb**2*hbcm3-ffmcfmb*
     . fmb*hbcm2-ffmcfmb*hbcm3+3*fmc2*hbcm)+16*(hbcm*p1p3*p4p6-hbcm*
     . p1p4*p3p6+hbcm*p2p3*p4p6-2*hbcm*p2p4*p3p6-2*hbcm*p3p4*p4p6-
     . hbcm*p3p5*p4p6))+(8*p3p6*(2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-hbcm)
     . +32*fb1*hbcm*p4p6))
      b(9)=ans
      b(10)=ccc*(w7*((p3p6*p2p3)*(-8*ffmcfmb-8)+24*p4p6*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-fmc*hbcm)-24*p2p3*p4p6)+w13*((p3p6*p1p3)*(8*
     . ffmcfmb+8)+24*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmc*hbcm-
     . hbcm2)+24*p1p3*p4p6)+w11*((p3p6*p3p4)*(8*ffmcfmb+8)+(p3p6*p2p3
     . )*(-8*ffmcfmb-8)+8*p4p6*(3*ffmcfmb*hbcm2+fmc*hbcm)+8*p3p6*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+24*(-p2p3*
     . p4p6+p3p4*p4p6))+(p3p6*(-16*fb2*ffmcfmb-16*fb2-4)-48*fb2*p4p6)
     . )
      ans2=4*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-8*
     . fb2*hbcm2+8*fb2*p1p3+4*fb2*p2p3-12*fb2*p3p5-hbcm2)
      ans1=w7*(24*p3p5*(ffmcfmb*hbcm2-fmc*hbcm)+4*p2p3*(-2*ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm-4*hbcm2)+16*p1p3*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+4*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*
     . hbcm3+4*ffmcfmb*hbcm4-fmb*fmc*hbcm2-4*fmc*hbcm3-2*fmc2*hbcm2+4
     . *p1p3*p2p3-6*p2p3*p3p5+2*p2p3s2))+w13*(24*p3p5*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*p2p3*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p1p3*
     . (-4*ffmcfmb*hbcm2-5*fmb*hbcm-fmc*hbcm+6*hbcm2)+4*(-ffmcfmb**2*
     . hbcm4-3*ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+4*ffmcfmb*hbcm4-
     . fmb*fmc*hbcm2+5*fmb*hbcm3-2*fmb2*hbcm2+fmc*hbcm3-3*hbcm4-2*
     . p1p3*p2p3+6*p1p3*p3p5-4*p1p3s2))+w11*(4*p3p4*(fmb*hbcm+fmc*
     . hbcm+6*hbcm2)+8*p3p5*(3*ffmcfmb*hbcm2+fmc*hbcm)+4*p2p3*(-2*
     . ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm-4*hbcm2)+8*p1p3*(-2*ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb
     . *fmc*hbcm3+4*ffmcfmb*hbcm4+2*fmc*hbcm3+3*fmc2*hbcm2-2*hbcm2*
     . p1p4-4*hbcm2*p2p4+4*p1p3*p2p3-4*p1p3*p3p4-2*p2p3*p3p4-6*p2p3*
     . p3p5+2*p2p3s2+6*p3p4*p3p5))+ans2
      ans=ccc*ans1
      b(11)=ans
      ans=ccc*(w7*(16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p2p3)*(4
     . *ffmcfmb*hbcm-fmb+fmc-hbcm)+8*p3p7*(-ffmcfmb**2*hbcm3-ffmcfmb*
     . hbcm3-fmb*fmc*hbcm-fmc*hbcm2+2*fmc2*hbcm)+16*(ffmcfmb*hbcm*
     . p1p3*p3p7-2*ffmcfmb*hbcm*p3p5*p3p7-hbcm*p2p3*p5p7))+w13*(16*
     . p5p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p3p7*p1p3)*(-2*
     . ffmcfmb*hbcm+fmb-fmc+hbcm)+8*p3p7*(-ffmcfmb**2*hbcm3+ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm-2*fmb*
     . hbcm2+2*fmb2*hbcm+fmc*hbcm2)-16*hbcm*p1p3*p5p7)+w11*(8*p5p7*(2
     . *ffmcfmb*hbcm3+fmc*hbcm2)+8*p4p7*(2*ffmcfmb*hbcm3-fmb*hbcm2+
     . fmc*hbcm2-4*hbcm3)+8*(p3p7*p3p4)*(-4*ffmcfmb*hbcm+fmb-fmc+hbcm
     . )+8*(p3p7*p2p3)*(4*ffmcfmb*hbcm-fmb+fmc-hbcm)+8*p3p7*(-ffmcfmb
     . **2*hbcm3-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmc*hbcm2-3*fmc2*
     . hbcm)+16*(ffmcfmb*hbcm*p1p3*p3p7-2*ffmcfmb*hbcm*p3p5*p3p7+2*
     . hbcm*p1p3*p4p7+hbcm*p1p4*p3p7+hbcm*p2p3*p4p7-hbcm*p2p3*p5p7+2*
     . hbcm*p2p4*p3p7+2*hbcm*p3p4*p5p7-3*hbcm*p3p5*p4p7))+(8*p3p7*(-2
     . *fb1*hbcm-2*fb2*fmb+2*fb2*fmc+hbcm)-32*fb1*hbcm*p5p7))
      b(12)=ans
      b(13)=ccc*(w7*(4*p2p3*(4*ffmcfmb*hbcm-fmb+fmc)+4*(-2*ffmcfmb**2
     . *hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+3*fmc2
     . *hbcm))+w13*(4*p1p3*(-4*ffmcfmb*hbcm+fmb-fmc+4*hbcm)+4*(-2*
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+4*ffmcfmb
     . *hbcm3-fmb*fmc*hbcm-fmb*hbcm2+3*fmb2*hbcm+fmc*hbcm2-2*hbcm3))+
     . w11*(4*p3p4*(-4*ffmcfmb*hbcm+fmb-fmc-2*hbcm)+4*p2p3*(4*ffmcfmb
     . *hbcm-fmb+fmc)+4*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-4*fmc2*hbcm+2*hbcm*p1p4+6*hbcm*p2p4))+8*(-
     . fb2*fmb+fb2*fmc+hbcm))
      b(14)=ccc*(w7*((p3p7*p2p3)*(8*ffmcfmb-16)+24*p5p7*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+2*
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-2*fmc*hbcm)-24*p2p3*p5p7)+w13
     . *((p3p7*p1p3)*(-8*ffmcfmb+16)+24*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+3*ffmcfmb*
     . hbcm2+2*fmb*hbcm-2*hbcm2)+24*p1p3*p5p7)+w11*(8*p4p7*(fmb*hbcm+
     . fmc*hbcm)+(p3p7*p3p4)*(-8*ffmcfmb+16)+(p3p7*p2p3)*(8*ffmcfmb-
     . 16)+8*p5p7*(3*ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2+fmc*hbcm)+24*(-p2p3*
     . p5p7+p3p4*p5p7))+(p3p7*(16*fb2*ffmcfmb-32*fb2+4)-48*fb2*p5p7))
      ans3=w11*(16*(p4p7*p3p6)*(-fmb*hbcm-fmc*hbcm-hbcm2)+16*(p4p6*
     . p3p7)*(2*ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+16*(p5p7*p3p6)*(-2*
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p4)*(-4*ffmcfmb*hbcm2+fmb*
     . hbcm-fmc*hbcm)+8*(p6p7*p2p3)*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*
     . hbcm+2*hbcm2)+8*p6p7*(ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*hbcm4+2*fmc2*hbcm2-2*p2p3s2-4*p3p4s2)+16*(p3p7*p3p6)*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmc*hbcm)+16*(ffmcfmb*hbcm2*p1p3*p6p7-hbcm2*p1p4*p6p7-
     . hbcm2*p2p4*p6p7-2*hbcm2*p4p6*p4p7-hbcm2*p4p6*p5p7-p1p3*p2p3*
     . p6p7+p1p3*p3p4*p6p7+3*p2p3*p3p4*p6p7+2*p2p3*p3p6*p5p7-2*p2p3*
     . p3p7*p4p6-2*p3p4*p3p6*p5p7+2*p3p4*p3p7*p4p6))+(16*p6p7*(-fb1*
     . fmb*hbcm-fb1*fmc*hbcm+2*fb2*hbcm2)+32*(-fb2*p1p3*p6p7-fb2*p2p3
     . *p6p7+2*fb2*p3p4*p6p7+2*fb2*p3p6*p5p7-2*fb2*p3p7*p4p6))
      ans2=w13*(32*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(
     . p5p7*p3p6)*(-ffmcfmb*hbcm2-2*fmb*hbcm+hbcm2)+32*(p6p7*p3p4)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm2+fmb*hbcm-fmc*
     . hbcm-4*hbcm2)+8*p6p7*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2-fmb*hbcm3-fmb2*hbcm2+fmc*hbcm3+2*
     . hbcm4+2*p1p3s2)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+16*(
     . p1p3*p2p3*p6p7-2*p1p3*p3p4*p6p7-2*p1p3*p3p6*p5p7+2*p1p3*p3p7*
     . p4p6))+ans3
      ans1=w7*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2-2*fmc*hbcm)+32*(p5p7*
     . p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+32*(p6p7*p3p4)*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+8*(p6p7*p2p3)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm+
     . 2*hbcm2)+16*(p6p7*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*p6p7*(
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2+2*fmc*hbcm3-fmc2*hbcm2-2*p2p3s2)+16*(p3p7*p3p6)*(ffmcfmb
     . **2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2)+16*
     . (-p1p3*p2p3*p6p7+2*p2p3*p3p4*p6p7+2*p2p3*p3p6*p5p7-2*p2p3*p3p7
     . *p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w7*(8*(p6p7*p2p3)*(-2*ffmcfmb*hbcm+fmb-fmc)+8*p6p7*(
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*
     . hbcm-2*fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*hbcm*
     . p3p7*p4p6))+w13*(32*(p5p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*
     . p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p3)*(2*ffmcfmb*hbcm-fmb+
     . fmc-2*hbcm)+8*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb
     . *fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-2*fmb2*hbcm-
     . fmc*hbcm2+hbcm3))+w11*(16*(p4p6*p3p7)*(-ffmcfmb*hbcm-hbcm)+16*
     . (p4p7*p3p6)*(-ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p4)*(2*ffmcfmb*hbcm
     . -fmb+fmc+2*hbcm)+8*(p6p7*p2p3)*(-2*ffmcfmb*hbcm+fmb-fmc)+8*
     . p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+3*
     . fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p3p7-hbcm*p1p4*p6p7-2*hbcm*
     . p2p4*p6p7-3*hbcm*p4p6*p4p7-hbcm*p4p6*p5p7))+8*p6p7*(2*fb2*fmb-
     . 2*fb2*fmc-hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp1s1_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(6) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,6 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=ccc*(4*w1*(fmc*hbcm2*p6p7-hbcm*p3p4*p6p7)+4*w2*(fmb*hbcm2*
     . p6p7-hbcm*p3p5*p6p7+2*hbcm*p3p6*p5p7+2*hbcm*p3p7*p5p6)-24*fb1*
     . hbcm*p6p7)
      b(2)=4*ccc*(hbcm*p4p7*w1-hbcm*p5p7*w2)
      b(3)=4*ccc*(hbcm*p4p6*w1-hbcm*p5p6*w2)
      b(4)=ccc*(4*w1*(fmc*hbcm*p6p7-p3p4*p6p7)+4*w2*(-fmb*hbcm*p6p7+
     . p3p5*p6p7)+8*fb2*p6p7)
      b(5)=ccc*(4*w1*(-fmc*hbcm*p3p7+hbcm2*p4p7-p3p4*p3p7)+4*w2*(fmb*
     . hbcm*p3p7-hbcm2*p5p7+p3p5*p3p7)+8*fb2*p3p7)
      b(6)=ccc*(4*w1*(-fmc*hbcm*p3p6+hbcm2*p4p6-p3p4*p3p6)+4*w2*(fmb*
     . hbcm*p3p6-hbcm2*p5p6+p3p5*p3p6)+8*fb2*p3p6)
      do 200 n=1,6 
         c(n,1)=c(n,1)+1.0d0*b(n)
200   continue
      return
      end 
 
      subroutine amp27_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5))
      ans=ccc*(w1*(8*(p5p7*p4p6)*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(
     . p5p7*p3p6*p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc-2*hbcm)+8*(p5p7*p3p6)
     . *(ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-3*fmc*hbcm2-2*fmc2*hbcm)+8*(2
     . *hbcm*p1p3*p4p6*p5p7+2*hbcm*p2p3*p4p6*p5p7-hbcm*p3p4*p4p6*p5p7
     . +hbcm*p3p4*p5p6*p5p7-2*hbcm*p3p5*p4p6*p5p7))+w16*(8*(p5p7*p5p6
     . )*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*hbcm3)+8*(p5p7*p3p6*
     . p3p5)*(ffmcfmb*hbcm-fmb+2*fmc-2*hbcm)+8*(p5p7*p3p6*p2p3)*(-
     . ffmcfmb*hbcm+fmb-2*fmc+2*hbcm)+8*(p5p7*p3p6)*(ffmcfmb*fmb*
     . hbcm2+2*fmb*fmc*hbcm-3*fmb*hbcm2-fmb2*hbcm)+8*(-2*fmb*hbcm2*
     . p4p6*p5p7-2*hbcm*p1p2*p3p6*p5p7+2*hbcm*p1p5*p3p6*p5p7+3*hbcm*
     . p2p3*p4p6*p5p7+3*hbcm*p2p3*p5p6*p5p7-2*hbcm*p2p4*p3p6*p5p7+2*
     . hbcm*p2p5*p3p6*p5p7-3*hbcm*p3p5*p4p6*p5p7-3*hbcm*p3p5*p5p6*
     . p5p7+2*hbcm*p3p6*p4p5*p5p7))+(4*(p5p7*p3p6)*(-12*fb1*ffmcfmb*
     . hbcm+16*fb1*hbcm-4*fb2*fmb+8*fb2*fmc+3*hbcm)+16*(fb1*hbcm*p4p6
     . *p5p7+fb1*hbcm*p5p6*p5p7)))
      b(1)=ans
      b(2)=ccc*(w1*(12*p5p7*(-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*(
     . p5p7*p3p4)*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+8*(-hbcm*p1p4*
     . p5p7+hbcm*p2p4*p5p7-hbcm*p4p5*p5p7))+w16*(12*p5p7*(-fmb*fmc*
     . hbcm+fmb*hbcm2+fmb2*hbcm)+4*(p5p7*p3p5)*(-2*ffmcfmb*hbcm+3*fmb
     . -3*fmc+3*hbcm)+4*(p5p7*p2p3)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*
     . hbcm)+8*(hbcm*p1p2*p5p7-hbcm*p1p5*p5p7+hbcm*p2p4*p5p7-3*hbcm*
     . p2p5*p5p7-hbcm*p4p5*p5p7))+12*p5p7*(-2*fb1*hbcm+2*fb2*fmb-2*
     . fb2*fmc-hbcm))
      b(3)=ccc*(8*w16*(-hbcm*p2p5*p3p6-3*hbcm*p2p5*p4p6-hbcm*p2p5*
     . p5p6)-8*hbcm*p2p5*p4p6*w1)
      b(4)=ccc*(w16*((p5p7*p3p6*p3p5)*(16*ffmcfmb-16)+(p5p7*p3p6*p2p3
     . )*(-16*ffmcfmb+16)+8*(p5p7*p5p6)*(-ffmcfmb*hbcm2-fmb*hbcm-fmc*
     . hbcm+2*hbcm2)+8*(-fmb*hbcm*p3p6*p5p7-3*fmb*hbcm*p4p6*p5p7+2*
     . p2p3*p4p6*p5p7-2*p3p5*p4p6*p5p7))+w1*((p5p7*p3p6*p3p4)*(-16*
     . ffmcfmb+16)+8*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+
     . hbcm2)+8*(fmc*hbcm*p3p6*p5p7-fmc*hbcm*p5p6*p5p7+2*p3p4*p4p6*
     . p5p7))+((p5p7*p3p6)*(32*fb2*ffmcfmb-32*fb2+8)-32*fb2*p4p6*p5p7
     . ))
      ans=ccc*(w1*(4*(p5p7*p3p4)*(3*fmb*hbcm+fmc*hbcm+hbcm2)+4*p5p7*(
     . -2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*hbcm2+3*fmc*hbcm3+3*fmc2*hbcm2-
     . 2*p3p4s2)+8*(-fmc*hbcm*p1p3*p5p7-3*fmc*hbcm*p2p3*p5p7+3*fmc*
     . hbcm*p3p5*p5p7-hbcm2*p1p4*p5p7+hbcm2*p2p4*p5p7-hbcm2*p4p5*p5p7
     . -p1p3*p3p4*p5p7-3*p2p3*p3p4*p5p7+3*p3p4*p3p5*p5p7))+w16*(12*(
     . p5p7*p3p5)*(-fmb*hbcm+fmc*hbcm-hbcm2)+12*(p5p7*p2p3)*(fmb*hbcm
     . -fmc*hbcm+hbcm2)+4*p5p7*(2*ffmcfmb*fmb*hbcm3+3*fmb*fmc*hbcm2-3
     . *fmb*hbcm3+3*fmb2*hbcm2-6*p2p3s2-6*p3p5s2)+8*(fmb*hbcm*p1p3*
     . p5p7+fmb*hbcm*p3p4*p5p7+hbcm2*p1p2*p5p7-hbcm2*p1p5*p5p7+hbcm2*
     . p2p4*p5p7-3*hbcm2*p2p5*p5p7-hbcm2*p4p5*p5p7-p1p3*p2p3*p5p7+
     . p1p3*p3p5*p5p7-p2p3*p3p4*p5p7+6*p2p3*p3p5*p5p7+p3p4*p3p5*p5p7)
     . )+(4*p5p7*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-
     . 6*fb2*hbcm2-5*hbcm2)+16*(fb2*p1p3*p5p7+3*fb2*p2p3*p5p7+fb2*
     . p3p4*p5p7-3*fb2*p3p5*p5p7)))
      b(5)=ans
      b(6)=ccc*(8*w1*(-2*fmc*hbcm*p2p5*p3p6-hbcm2*p2p5*p4p6-2*p2p5*
     . p3p4*p3p6)+w16*(8*(p3p6*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(-
     . hbcm2*p2p5*p4p6-hbcm2*p2p5*p5p6-p2p3*p2p5*p3p6+p2p5*p3p5*p3p6)
     . )+32*fb2*p2p5*p3p6)
      b(7)=ccc*(w16*(12*p2p5*(fmc*hbcm-hbcm2)+12*(p2p3*p2p5-p2p5*p3p5
     . ))+12*w1*(fmc*hbcm*p2p5+p2p5*p3p4)-24*fb2*p2p5)
      b(8)=ccc*(12*w1*(fmc*hbcm2*p2p5+hbcm*p2p5*p3p4)+w16*(4*p2p5*(2*
     . ffmcfmb*hbcm3+3*fmc*hbcm2-3*hbcm3)+4*(2*hbcm*p1p3*p2p5+3*hbcm*
     . p2p3*p2p5+2*hbcm*p2p5*p3p4-3*hbcm*p2p5*p3p5))-24*fb1*hbcm*p2p5
     . )
      b(9)=ccc*(w1*(4*p4p6*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+4*(p3p6*
     . p3p4)*(-ffmcfmb*hbcm+2*fmb-fmc+hbcm)+4*p3p6*(-ffmcfmb*fmc*
     . hbcm2+2*fmb*fmc*hbcm-fmc2*hbcm)+4*(-2*hbcm*p1p3*p4p6+2*hbcm*
     . p1p4*p3p6-2*hbcm*p2p3*p4p6-2*hbcm*p2p4*p3p6+hbcm*p3p4*p4p6-
     . hbcm*p3p4*p5p6+2*hbcm*p3p5*p4p6+2*hbcm*p3p6*p4p5))+w16*(4*p5p6
     . *(-ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2+2*hbcm3)+4*(p3p6*p3p5)*
     . (ffmcfmb*hbcm-2*fmb+fmc-hbcm)+4*(p3p6*p2p3)*(-ffmcfmb*hbcm+2*
     . fmb-fmc+hbcm)+4*p3p6*(-ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb2*
     . hbcm)+4*(2*fmb*hbcm2*p4p6-3*hbcm*p2p3*p4p6-3*hbcm*p2p3*p5p6+4*
     . hbcm*p2p5*p3p6+3*hbcm*p3p5*p4p6+3*hbcm*p3p5*p5p6))+(2*p3p6*(12
     . *fb1*ffmcfmb*hbcm-4*fb1*hbcm-8*fb2*fmb+4*fb2*fmc+3*hbcm)+8*(-
     . fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(10)=ccc*(w16*((p3p6*p3p5)*(-8*ffmcfmb+8)+(p3p6*p2p3)*(8*
     . ffmcfmb-8)+4*p5p6*(ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm-2*hbcm2)+4*
     . (fmb*hbcm*p3p6+3*fmb*hbcm*p4p6-2*p2p3*p4p6+2*p3p5*p4p6))+w1*((
     . p3p6*p3p4)*(8*ffmcfmb-8)+4*p4p6*(-ffmcfmb*hbcm2+fmb*hbcm-fmc*
     . hbcm-hbcm2)+4*(-fmc*hbcm*p3p6+fmc*hbcm*p5p6-2*p3p4*p4p6))+(
     . p3p6*(-16*fb2*ffmcfmb+16*fb2-4)+16*fb2*p4p6))
      b(11)=ccc*(w1*(2*p3p4*(-3*fmb*hbcm-fmc*hbcm-hbcm2)+2*(2*ffmcfmb
     . *fmc*hbcm3-3*fmb*fmc*hbcm2+2*fmc*hbcm*p1p3+6*fmc*hbcm*p2p3-6*
     . fmc*hbcm*p3p5-3*fmc*hbcm3-3*fmc2*hbcm2+2*hbcm2*p1p4-2*hbcm2*
     . p2p4+2*hbcm2*p4p5+2*p1p3*p3p4+6*p2p3*p3p4-6*p3p4*p3p5+2*p3p4s2
     . ))+w16*(6*p3p5*(fmb*hbcm-fmc*hbcm+hbcm2)+6*p2p3*(-fmb*hbcm+fmc
     . *hbcm-hbcm2)+2*(-2*ffmcfmb*fmb*hbcm3-3*fmb*fmc*hbcm2-2*fmb*
     . hbcm*p1p3-2*fmb*hbcm*p3p4+3*fmb*hbcm3-3*fmb2*hbcm2-2*hbcm2*
     . p1p2+2*hbcm2*p1p5-2*hbcm2*p2p4+6*hbcm2*p2p5+2*hbcm2*p4p5+2*
     . p1p3*p2p3-2*p1p3*p3p5+2*p2p3*p3p4-12*p2p3*p3p5+6*p2p3s2-2*p3p4
     . *p3p5+6*p3p5s2))+2*(6*fb1*fmb*hbcm+6*fb1*fmc*hbcm-4*fb2*
     . ffmcfmb*hbcm2+6*fb2*hbcm2-4*fb2*p1p3-12*fb2*p2p3-4*fb2*p3p4+12
     . *fb2*p3p5+5*hbcm2))
      b(13)=ccc*(w1*(2*p3p4*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+2*(-3*
     . fmb*fmc*hbcm+3*fmc*hbcm2+3*fmc2*hbcm-2*hbcm*p1p4+2*hbcm*p2p4-2
     . *hbcm*p4p5))+w16*(2*p3p5*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+
     . 2*p2p3*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm)+2*(-3*fmb*fmc*hbcm+
     . 3*fmb*hbcm2+3*fmb2*hbcm+2*hbcm*p1p2-2*hbcm*p1p5+2*hbcm*p2p4-6*
     . hbcm*p2p5-2*hbcm*p4p5))+6*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-
     . hbcm))
      do 200 n=1,16 
         c(n,2)=c(n,2)-0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp26_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      ans2=w2*(8*(p6p7*p2p3)*(3*fmb*fmc*hbcm-fmb*hbcm2-3*fmb2*hbcm)+8
     . *(p5p6*p4p7)*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*hbcm3)+8*(
     . p4p7*p3p6*p3p5)*(ffmcfmb*hbcm-fmb+2*fmc-2*hbcm)+8*(p6p7*p3p5*
     . p2p3)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-hbcm)+8*(p4p7*p3p6)*(ffmcfmb
     . *fmb*hbcm2+2*fmb*fmc*hbcm-3*fmb*hbcm2-fmb2*hbcm)+8*(-2*fmb*
     . hbcm2*p1p2*p6p7-fmb*hbcm2*p2p4*p6p7+2*fmb*hbcm2*p2p5*p6p7-2*
     . fmb*hbcm2*p4p6*p4p7-2*hbcm*p1p2*p3p5*p6p7+2*hbcm*p1p2*p3p7*
     . p5p6+2*hbcm*p1p5*p2p3*p6p7+2*hbcm*p1p5*p3p6*p4p7-2*hbcm*p2p3*
     . p2p5*p6p7+2*hbcm*p2p3*p4p5*p6p7+2*hbcm*p2p3*p4p7*p5p6+2*hbcm*
     . p2p3*p5p6*p5p7-hbcm*p2p4*p3p5*p6p7+2*hbcm*p2p5*p3p5*p6p7-2*
     . hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*p5p6-3*hbcm*p3p5*p4p6*
     . p4p7-3*hbcm*p3p5*p4p7*p5p6+2*hbcm*p3p6*p4p5*p4p7))+(8*(p6p7*
     . p2p3)*(2*fb1*hbcm-6*fb2*fmb+6*fb2*fmc+3*hbcm)+4*(p4p7*p3p6)*(-
     . 12*fb1*ffmcfmb*hbcm+16*fb1*hbcm-4*fb2*fmb+8*fb2*fmc+3*hbcm)+16
     . *(2*fb1*hbcm*p1p2*p6p7+fb1*hbcm*p2p4*p6p7-2*fb1*hbcm*p2p5*p6p7
     . +fb1*hbcm*p4p6*p4p7+fb1*hbcm*p4p7*p5p6))
      ans1=w11*(8*(p6p7*p2p3)*(3*fmb*fmc*hbcm-fmc*hbcm2-3*fmc2*hbcm)+
     . 8*(p4p7*p4p6)*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p6p7*p2p4)*(2
     . *ffmcfmb*hbcm3-3*fmb*hbcm2+2*fmc*hbcm2-3*hbcm3)+8*(p4p7*p3p6*
     . p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc-2*hbcm)+8*p6p7*(2*ffmcfmb*hbcm*
     . p2p3s2-3*fmb*p2p3s2+3*fmc*p2p3s2-hbcm*p2p3s2)+8*(p4p7*p3p6*
     . p2p3)*(ffmcfmb*hbcm-fmb+2*fmc+2*hbcm)+8*(p6p7*p3p4*p2p3)*(-2*
     . ffmcfmb*hbcm+3*fmb-3*fmc+hbcm)+8*(p4p7*p3p6)*(ffmcfmb*fmc*
     . hbcm2+fmb*fmc*hbcm-3*fmc*hbcm2-2*fmc2*hbcm)+8*(-2*fmc*hbcm2*
     . p1p2*p6p7+2*fmc*hbcm2*p2p5*p6p7-2*hbcm*p1p2*p3p4*p6p7+2*hbcm*
     . p1p2*p3p7*p4p6+2*hbcm*p1p3*p2p4*p6p7+2*hbcm*p1p3*p4p6*p4p7+2*
     . hbcm*p1p4*p2p3*p6p7+5*hbcm*p2p3*p2p4*p6p7-4*hbcm*p2p3*p2p5*
     . p6p7+2*hbcm*p2p3*p4p5*p6p7+3*hbcm*p2p3*p4p6*p4p7+2*hbcm*p2p3*
     . p4p6*p5p7-hbcm*p2p3*p4p7*p5p6+hbcm*p2p4*p3p4*p6p7-6*hbcm*p2p4*
     . p3p5*p6p7-4*hbcm*p2p4*p3p6*p3p7+2*hbcm*p2p4*p3p6*p4p7-2*hbcm*
     . p2p4*p3p6*p5p7-4*hbcm*p2p4*p3p7*p4p6+2*hbcm*p2p4*p3p7*p5p6+2*
     . hbcm*p2p5*p3p4*p6p7-2*hbcm*p2p5*p3p7*p4p6-hbcm*p3p4*p4p6*p4p7+
     . hbcm*p3p4*p4p7*p5p6-2*hbcm*p3p5*p4p6*p4p7))+ans2
      ans=ccc*ans1
      b(1)=ans
      b(2)=ccc*(w2*(12*p4p7*(-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+4*(
     . p4p7*p3p5)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+8*(-hbcm*p1p5*
     . p4p7+hbcm*p2p5*p4p7-hbcm*p4p5*p4p7))+w11*(12*p4p7*(-fmb*fmc*
     . hbcm+fmc*hbcm2+fmc2*hbcm)+4*(p4p7*p3p4)*(2*ffmcfmb*hbcm-3*fmb+
     . 3*fmc+hbcm)+4*(p4p7*p2p3)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+8
     . *(hbcm*p1p2*p4p7-hbcm*p1p4*p4p7+2*hbcm*p2p4*p3p7-hbcm*p2p4*
     . p4p7+2*hbcm*p2p4*p5p7+hbcm*p2p5*p4p7-hbcm*p4p5*p4p7))+12*p4p7*
     . (-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-hbcm))
      b(3)=ccc*(8*w2*(-hbcm*p1p2*p5p6+hbcm*p2p3*p5p6+hbcm*p2p5*p5p6)+
     . 8*w11*(-hbcm*p1p2*p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*p3p6+2*hbcm*
     . p2p4*p4p6-hbcm*p2p4*p5p6+hbcm*p2p5*p4p6))
      b(4)=ccc*(w2*((p4p7*p3p6*p3p5)*(16*ffmcfmb-16)+8*(p5p6*p4p7)*(-
     . ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(-3*fmb*hbcm*p2p4*
     . p6p7-fmb*hbcm*p3p6*p4p7-3*fmb*hbcm*p4p6*p4p7-3*p2p4*p3p5*p6p7-
     . 2*p3p5*p4p6*p4p7))+w11*(24*(p6p7*p2p4)*(-fmb*hbcm+hbcm2)+(p4p7
     . *p3p6*p3p4)*(-16*ffmcfmb+16)+(p4p7*p3p6*p2p3)*(16*ffmcfmb-16)+
     . 8*(p4p7*p4p6)*(ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+hbcm2)+8*(fmc*
     . hbcm*p3p6*p4p7-fmc*hbcm*p4p7*p5p6-3*p2p3*p2p4*p6p7-2*p2p3*p4p6
     . *p4p7+3*p2p4*p3p4*p6p7+2*p3p4*p4p6*p4p7))+((p4p7*p3p6)*(32*fb2
     . *ffmcfmb-32*fb2+8)+16*(-3*fb2*p2p4*p6p7-2*fb2*p4p6*p4p7)))
      ans2=w2*(12*(p4p7*p3p5)*(-fmb*hbcm+fmc*hbcm-hbcm2)+4*p4p7*(2*
     . ffmcfmb*fmb*hbcm3+3*fmb*fmc*hbcm2-3*fmb*hbcm3+3*fmb2*hbcm2-6*
     . p3p5s2)+8*(2*fmb*hbcm*p1p2*p3p7+fmb*hbcm*p1p3*p4p7+fmb*hbcm*
     . p2p3*p4p7+2*fmb*hbcm*p2p3*p5p7+fmb*hbcm*p2p4*p3p7-2*fmb*hbcm*
     . p2p5*p3p7+fmb*hbcm*p3p4*p4p7-hbcm2*p1p5*p4p7+hbcm2*p2p5*p4p7-
     . hbcm2*p4p5*p4p7+2*p1p2*p3p5*p3p7+p1p3*p3p5*p4p7+p2p3*p3p5*p4p7
     . +2*p2p3*p3p5*p5p7+p2p4*p3p5*p3p7-2*p2p5*p3p5*p3p7+p3p4*p3p5*
     . p4p7))+(4*p4p7*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*fb2*ffmcfmb*
     . hbcm2-6*fb2*hbcm2-5*hbcm2)+16*(2*fb2*p1p2*p3p7+fb2*p1p3*p4p7+
     . fb2*p2p3*p4p7+2*fb2*p2p3*p5p7+fb2*p2p4*p3p7-2*fb2*p2p5*p3p7+
     . fb2*p3p4*p4p7-3*fb2*p3p5*p4p7))
      ans1=w11*(4*(p4p7*p3p4)*(3*fmb*hbcm+fmc*hbcm+hbcm2)+8*(p3p7*
     . p2p4)*(3*fmb*hbcm+2*fmc*hbcm-hbcm2)+4*(p4p7*p2p3)*(-3*fmb*hbcm
     . -5*fmc*hbcm-hbcm2)+4*p4p7*(-2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*
     . hbcm2+3*fmc*hbcm3+3*fmc2*hbcm2+2*p2p3s2-2*p3p4s2)+8*(-2*fmc*
     . hbcm*p1p2*p3p7-fmc*hbcm*p1p3*p4p7-2*fmc*hbcm*p2p3*p5p7+2*fmc*
     . hbcm*p2p5*p3p7+3*fmc*hbcm*p3p5*p4p7+hbcm2*p1p2*p4p7-hbcm2*p1p4
     . *p4p7-hbcm2*p2p4*p4p7+2*hbcm2*p2p4*p5p7+hbcm2*p2p5*p4p7-hbcm2*
     . p4p5*p4p7+2*p1p2*p2p3*p3p7-2*p1p2*p3p4*p3p7+p1p3*p2p3*p4p7-
     . p1p3*p3p4*p4p7+p2p3*p2p4*p3p7-2*p2p3*p2p5*p3p7-2*p2p3*p3p4*
     . p5p7-3*p2p3*p3p5*p4p7+2*p2p3s2*p5p7-p2p4*p3p4*p3p7+2*p2p5*p3p4
     . *p3p7+3*p3p4*p3p5*p4p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w2*((p3p6*p3p5*p2p3)*(-16*ffmcfmb+8)+8*(p5p6*p2p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm-hbcm2)+8*(fmb*hbcm*p1p2*p3p6+3
     . *fmb*hbcm*p2p3*p4p6-fmb*hbcm*p2p5*p3p6-hbcm2*p1p2*p5p6+hbcm2*
     . p2p5*p5p6+p1p2*p3p5*p3p6+2*p2p3*p3p5*p4p6-p2p5*p3p5*p3p6))+w11
     . *(8*p3p6*(-2*ffmcfmb*p2p3s2+p2p3s2)+(p3p6*p3p4*p2p3)*(16*
     . ffmcfmb-8)+8*(p3p6*p2p4)*(-ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+
     . 8*(p4p6*p2p3)*(-ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+8*(-fmc*hbcm*
     . p1p2*p3p6+fmc*hbcm*p2p3*p5p6+fmc*hbcm*p2p5*p3p6-hbcm2*p1p2*
     . p4p6+hbcm2*p2p4*p4p6+hbcm2*p2p5*p4p6+p1p2*p2p3*p3p6-p1p2*p3p4*
     . p3p6-p2p3*p2p5*p3p6-2*p2p3*p3p4*p4p6+2*p2p3s2*p4p6+p2p5*p3p4*
     . p3p6))+((p3p6*p2p3)*(-32*fb2*ffmcfmb+16*fb2-8)+16*(fb2*p1p2*
     . p3p6+2*fb2*p2p3*p4p6-fb2*p2p5*p3p6)))
      b(7)=ccc*(w11*(4*p2p4*(-3*fmb*hbcm-2*fmc*hbcm+3*hbcm2)+4*(2*fmc
     . *hbcm*p1p2-2*fmc*hbcm*p2p3-2*fmc*hbcm*p2p5-2*p1p2*p2p3+2*p1p2*
     . p3p4-p2p3*p2p4+2*p2p3*p2p5-2*p2p3*p3p4+2*p2p3s2+p2p4*p3p4-2*
     . p2p5*p3p4))+4*w2*(-2*fmb*hbcm*p1p2+2*fmb*hbcm*p2p3-fmb*hbcm*
     . p2p4+2*fmb*hbcm*p2p5-2*p1p2*p3p5+2*p2p3*p3p5-p2p4*p3p5+2*p2p5*
     . p3p5)+8*(-2*fb2*p1p2+2*fb2*p2p3-fb2*p2p4+2*fb2*p2p5))
      ans=ccc*(w2*(4*p2p3*(-3*fmb*fmc*hbcm+fmb*hbcm2+3*fmb2*hbcm)+4*(
     . p3p5*p2p3)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+hbcm)+4*(2*fmb*hbcm2*
     . p1p2+fmb*hbcm2*p2p4-2*fmb*hbcm2*p2p5+2*hbcm*p1p2*p3p5-2*hbcm*
     . p1p5*p2p3+2*hbcm*p2p3*p2p5-2*hbcm*p2p3*p4p5+hbcm*p2p4*p3p5-2*
     . hbcm*p2p5*p3p5))+w11*(4*p2p3*(-3*fmb*fmc*hbcm+fmc*hbcm2+3*fmc2
     . *hbcm)+4*p2p4*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-2*fmc*hbcm2+3*
     . hbcm3)+4*(p3p4*p2p3)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-hbcm)+4*(-2*
     . ffmcfmb*hbcm*p2p3s2+3*fmb*p2p3s2+2*fmc*hbcm2*p1p2-2*fmc*hbcm2*
     . p2p5-3*fmc*p2p3s2+2*hbcm*p1p2*p3p4-2*hbcm*p1p3*p2p4-2*hbcm*
     . p1p4*p2p3-5*hbcm*p2p3*p2p4+4*hbcm*p2p3*p2p5-2*hbcm*p2p3*p4p5+
     . hbcm*p2p3s2-hbcm*p2p4*p3p4+6*hbcm*p2p4*p3p5-2*hbcm*p2p5*p3p4))
     . +(4*p2p3*(-2*fb1*hbcm+6*fb2*fmb-6*fb2*fmc-3*hbcm)+8*(-2*fb1*
     . hbcm*p1p2-fb1*hbcm*p2p4+2*fb1*hbcm*p2p5)))
      b(8)=ans
      b(9)=ccc*(w11*(4*p4p6*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+4*(p3p6*
     . p3p4)*(ffmcfmb*hbcm-2*fmb+fmc-hbcm)+4*(p3p6*p2p3)*(-ffmcfmb*
     . hbcm+2*fmb-fmc+hbcm)+4*p3p6*(ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm+
     . fmc2*hbcm)+4*(2*hbcm*p1p2*p3p6+2*hbcm*p1p3*p4p6-2*hbcm*p1p4*
     . p3p6+hbcm*p2p3*p4p6-hbcm*p2p3*p5p6-2*hbcm*p2p4*p3p6+2*hbcm*
     . p2p5*p3p6-hbcm*p3p4*p4p6+hbcm*p3p4*p5p6-2*hbcm*p3p5*p4p6-2*
     . hbcm*p3p6*p4p5))+w2*(4*p5p6*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*
     . hbcm2-2*hbcm3)+4*(p3p6*p3p5)*(-ffmcfmb*hbcm+2*fmb-fmc+hbcm)+4*
     . p3p6*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+2*fmb2*hbcm)+4*(-2*fmb*
     . hbcm2*p4p6-3*hbcm*p3p5*p4p6-3*hbcm*p3p5*p5p6))+(2*p3p6*(-12*
     . fb1*ffmcfmb*hbcm+4*fb1*hbcm+8*fb2*fmb-4*fb2*fmc-3*hbcm)+8*(fb1
     . *hbcm*p4p6+fb1*hbcm*p5p6)))
      b(10)=ccc*(w2*((p3p6*p3p5)*(8*ffmcfmb-8)+4*p5p6*(-ffmcfmb*hbcm2
     . -fmb*hbcm-fmc*hbcm+2*hbcm2)+4*(-fmb*hbcm*p3p6-3*fmb*hbcm*p4p6-
     . 2*p3p5*p4p6))+w11*((p3p6*p3p4)*(-8*ffmcfmb+8)+(p3p6*p2p3)*(8*
     . ffmcfmb-8)+4*p4p6*(ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+hbcm2)+4*(
     . fmc*hbcm*p3p6-fmc*hbcm*p5p6-2*p2p3*p4p6+2*p3p4*p4p6))+(p3p6*(
     . 16*fb2*ffmcfmb-16*fb2+4)-16*fb2*p4p6))
      b(11)=ccc*(w11*(2*p3p4*(3*fmb*hbcm+fmc*hbcm+hbcm2)+2*p2p3*(-3*
     . fmb*hbcm-fmc*hbcm-hbcm2)+2*(-2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*
     . hbcm2-2*fmc*hbcm*p1p3+6*fmc*hbcm*p3p5+3*fmc*hbcm3+3*fmc2*hbcm2
     . +2*hbcm2*p1p2-2*hbcm2*p1p4-6*hbcm2*p2p4+2*hbcm2*p2p5-2*hbcm2*
     . p4p5+2*p1p3*p2p3-2*p1p3*p3p4+4*p2p3*p3p4-6*p2p3*p3p5-2*p2p3s2+
     . 6*p3p4*p3p5-2*p3p4s2))+w2*(6*p3p5*(-fmb*hbcm+fmc*hbcm-hbcm2)+2
     . *(2*ffmcfmb*fmb*hbcm3+3*fmb*fmc*hbcm2+2*fmb*hbcm*p1p3-2*fmb*
     . hbcm*p2p3+2*fmb*hbcm*p3p4-3*fmb*hbcm3+3*fmb2*hbcm2-2*hbcm2*
     . p1p5+2*hbcm2*p2p5-2*hbcm2*p4p5+2*p1p3*p3p5-2*p2p3*p3p5+2*p3p4*
     . p3p5-6*p3p5s2))+2*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2-6*fb2*hbcm2+4*fb2*p1p3-4*fb2*p2p3+4*fb2*p3p4-12*
     . fb2*p3p5-5*hbcm2))
      b(12)=ccc*(w2*(4*p3p7*(3*fmb*fmc*hbcm-fmb*hbcm2-3*fmb2*hbcm)+4*
     . (p3p7*p3p5)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-hbcm)+8*(fmb*hbcm2*
     . p4p7+fmb*hbcm2*p5p7+hbcm*p1p5*p3p7-hbcm*p2p5*p3p7+hbcm*p3p5*
     . p4p7+hbcm*p3p5*p5p7+hbcm*p3p7*p4p5))+w11*(4*p3p7*(3*fmb*fmc*
     . hbcm-fmc*hbcm2-3*fmc2*hbcm)+4*(p3p7*p3p4)*(-2*ffmcfmb*hbcm+3*
     . fmb-3*fmc+hbcm)+4*(p3p7*p2p3)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-hbcm
     . )+8*(fmc*hbcm2*p4p7+fmc*hbcm2*p5p7-hbcm*p1p2*p3p7+hbcm*p1p4*
     . p3p7-hbcm*p2p3*p4p7-hbcm*p2p3*p5p7+3*hbcm*p2p4*p3p7-hbcm*p2p5*
     . p3p7+hbcm*p3p4*p4p7+hbcm*p3p4*p5p7+hbcm*p3p7*p4p5))+(4*p3p7*(2
     . *fb1*hbcm-6*fb2*fmb+6*fb2*fmc+3*hbcm)+16*(-fb1*hbcm*p4p7-fb1*
     . hbcm*p5p7)))
      b(13)=ccc*(w2*(2*p3p5*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm)+2*(3*
     . fmb*fmc*hbcm-3*fmb*hbcm2-3*fmb2*hbcm+2*hbcm*p1p5-2*hbcm*p2p5+2
     . *hbcm*p4p5))+w11*(2*p3p4*(-2*ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+2*
     . p2p3*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+2*(3*fmb*fmc*hbcm-3*fmc
     . *hbcm2-3*fmc2*hbcm-2*hbcm*p1p2+2*hbcm*p1p4+6*hbcm*p2p4-2*hbcm*
     . p2p5+2*hbcm*p4p5))+6*(2*fb1*hbcm-2*fb2*fmb+2*fb2*fmc+hbcm))
      b(14)=ccc*(8*w11*(fmc*hbcm*p3p7+fmc*hbcm*p4p7+fmc*hbcm*p5p7-
     . p2p3*p3p7-p2p3*p4p7-p2p3*p5p7+p3p4*p3p7+p3p4*p4p7+p3p4*p5p7)+8
     . *w2*(-fmb*hbcm*p3p7-fmb*hbcm*p4p7-fmb*hbcm*p5p7-p3p5*p3p7-p3p5
     . *p4p7-p3p5*p5p7)+16*(-fb2*p3p7-fb2*p4p7-fb2*p5p7))
      ans2=w2*(4*(p6p7*p3p5)*(fmb*hbcm+3*fmc*hbcm+hbcm2)+(p3p7*p3p6*
     . p3p5)*(16*ffmcfmb-8)+8*(p5p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm-
     . fmc*hbcm+hbcm2)+4*p6p7*(2*ffmcfmb*fmb*hbcm3+3*fmb*fmc*hbcm2+
     . fmb*hbcm3+3*fmb2*hbcm2-2*p3p5s2)+8*(-fmb*hbcm*p1p3*p6p7-3*fmb*
     . hbcm*p2p3*p6p7+3*fmb*hbcm*p3p4*p6p7+fmb*hbcm*p3p6*p4p7+fmb*
     . hbcm*p3p6*p5p7-3*fmb*hbcm*p3p7*p4p6-hbcm2*p1p5*p6p7+hbcm2*p2p5
     . *p6p7-hbcm2*p4p5*p6p7-hbcm2*p4p7*p5p6-hbcm2*p5p6*p5p7-p1p3*
     . p3p5*p6p7-3*p2p3*p3p5*p6p7+3*p3p4*p3p5*p6p7+p3p5*p3p6*p4p7+
     . p3p5*p3p6*p5p7-2*p3p5*p3p7*p4p6))+((p3p7*p3p6)*(32*fb2*ffmcfmb
     . -16*fb2+8)+4*p6p7*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2+2*fb2*hbcm2-5*hbcm2)+16*(-fb2*p1p3*p6p7-3*fb2*
     . p2p3*p6p7+3*fb2*p3p4*p6p7-fb2*p3p5*p6p7+fb2*p3p6*p4p7+fb2*p3p6
     . *p5p7-2*fb2*p3p7*p4p6))
      ans1=w11*(12*(p6p7*p3p4)*(fmb*hbcm-fmc*hbcm-hbcm2)+12*(p6p7*
     . p2p3)*(-fmb*hbcm+fmc*hbcm+hbcm2)+(p3p7*p3p6*p3p4)*(-16*ffmcfmb
     . +8)+(p3p7*p3p6*p2p3)*(16*ffmcfmb-8)+8*(p4p6*p3p7)*(ffmcfmb*
     . hbcm2-fmb*hbcm+fmc*hbcm)+4*p6p7*(-2*ffmcfmb*fmc*hbcm3+3*fmb*
     . fmc*hbcm2-fmc*hbcm3+3*fmc2*hbcm2-6*p2p3s2-6*p3p4s2)+8*(fmc*
     . hbcm*p1p3*p6p7+fmc*hbcm*p3p5*p6p7-fmc*hbcm*p3p6*p4p7-fmc*hbcm*
     . p3p6*p5p7-fmc*hbcm*p3p7*p5p6+hbcm2*p1p2*p6p7-hbcm2*p1p4*p6p7-3
     . *hbcm2*p2p4*p6p7+hbcm2*p2p5*p6p7-hbcm2*p4p5*p6p7-hbcm2*p4p6*
     . p4p7-hbcm2*p4p6*p5p7-p1p3*p2p3*p6p7+p1p3*p3p4*p6p7+6*p2p3*p3p4
     . *p6p7-p2p3*p3p5*p6p7+p2p3*p3p6*p4p7+p2p3*p3p6*p5p7-2*p2p3*p3p7
     . *p4p6+p3p4*p3p5*p6p7-p3p4*p3p6*p4p7-p3p4*p3p6*p5p7+2*p3p4*p3p7
     . *p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w2*(12*p6p7*(-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+4*(
     . p6p7*p3p5)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+8*(-hbcm*p1p5*
     . p6p7+hbcm*p2p5*p6p7-hbcm*p3p7*p5p6-hbcm*p4p5*p6p7-hbcm*p4p7*
     . p5p6-hbcm*p5p6*p5p7))+w11*(12*p6p7*(-fmb*fmc*hbcm+fmc*hbcm2+
     . fmc2*hbcm)+4*(p6p7*p3p4)*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+4*(
     . p6p7*p2p3)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+8*(hbcm*p1p2*
     . p6p7-hbcm*p1p4*p6p7-3*hbcm*p2p4*p6p7+hbcm*p2p5*p6p7-hbcm*p3p7*
     . p4p6-hbcm*p4p5*p6p7-hbcm*p4p6*p4p7-hbcm*p4p6*p5p7))+12*p6p7*(-
     . 2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-hbcm))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp25_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3+2*ffmcfmb*p2p3
     . -fmb2+hbcm2+2*p1p2-2*p1p3-2*p2p3))
      ans2=w20*(16*(p6p7*p2p3*p1p3)*(fmb-fmc)+8*(p6p7*p1p2)*(-fmb*
     . hbcm2+fmc*hbcm2)+8*p6p7*(ffmcfmb*hbcm*p1p3s2-ffmcfmb*hbcm*
     . p2p3s2+fmb*p1p3s2+fmb*p2p3s2-2*fmc*p2p3s2-hbcm*p1p3s2+hbcm*
     . p2p3s2)+4*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-3*
     . fmb*fmc*hbcm-fmb*hbcm2+3*fmb2*hbcm+fmc*hbcm2)+4*(p6p7*p1p3)*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmb*hbcm2-
     . fmb2*hbcm+fmc*hbcm2)+16*(hbcm*p1p2*p1p3*p6p7-hbcm*p1p2*p2p3*
     . p6p7))+(2*p6p7*(4*fb1*fmb*fmc*hbcm-4*fb1*fmb2*hbcm-4*fb2*
     . ffmcfmb*fmb*hbcm2+4*fb2*ffmcfmb*fmc*hbcm2+4*fb2*fmb*hbcm2-4*
     . fb2*fmc*hbcm2+fmb*hbcm2-fmc*hbcm2)+4*(p6p7*p2p3)*(-4*fb1*
     . ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb+8*fb2*fmc+3*hbcm)+4*(p6p7*
     . p1p3)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*fmb-3*hbcm))
      ans1=w1*(8*(p6p7*p2p4)*(ffmcfmb*hbcm3-fmb*hbcm2+2*fmc*hbcm2-3*
     . hbcm3)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*
     . p3p4*p2p3)*(-ffmcfmb*hbcm+fmb-2*fmc+hbcm)+8*(p6p7*p3p4*p1p3)*(
     . ffmcfmb*hbcm+fmb-hbcm)+4*p6p7*(ffmcfmb*fmc*hbcm4+fmb*fmc*hbcm3
     . -fmc*hbcm4)+8*(p6p7*p3p5)*(ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc*
     . hbcm2)+8*(p6p7*p2p3)*(-ffmcfmb*fmc*hbcm2+fmc*hbcm2-2*fmc2*hbcm
     . )+8*(p6p7*p1p3)*(-ffmcfmb*fmc*hbcm2+fmc*hbcm2)+4*(p6p7*p3p4)*(
     . -ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-
     . fmb2*hbcm-fmc*hbcm2)+8*(-fmc*hbcm2*p1p2*p6p7+2*fmc*hbcm2*p2p5*
     . p6p7-2*hbcm*p1p2*p3p4*p6p7+4*hbcm*p1p3*p2p4*p6p7+4*hbcm*p2p3*
     . p2p4*p6p7-4*hbcm*p2p4*p3p5*p6p7+4*hbcm*p2p5*p3p4*p6p7))+ans2
      ans=ccc*ans1
      b(1)=ans
      ans=ccc*(w20*(8*p6p7*(ffmcfmb*p1p3s2-ffmcfmb*p2p3s2-p1p3s2+
     . p2p3s2)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p6p7*
     . p2p3)*(3*ffmcfmb*hbcm2+fmb*hbcm-3*hbcm2)+4*(p6p7*p1p3)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p6p7*p1p2)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+2*hbcm2)+8*(-p1p2*p1p3*p6p7-p1p2*p2p3*p6p7+2*p1p3*
     . p2p4*p6p7+2*p2p3*p2p4*p6p7))+w1*((p6p7*p3p4*p2p3)*(-8*ffmcfmb+
     . 8)+(p6p7*p3p4*p1p3)*(8*ffmcfmb-8)+4*(p6p7*p3p4)*(ffmcfmb*hbcm2
     . +fmb*hbcm-hbcm2)+8*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*
     . hbcm+hbcm2)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*
     . p6p7*(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm2-fmb*fmc*hbcm2+
     . fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc2*hbcm2)+8*(-2*fmc*hbcm*p1p2*
     . p6p7+2*fmc*hbcm*p2p3*p6p7-p1p2*p3p4*p6p7+2*p2p4*p3p4*p6p7))+((
     . p6p7*p2p3)*(16*fb2*ffmcfmb-16*fb2+4)+(p6p7*p1p3)*(-16*fb2*
     . ffmcfmb+16*fb2-4)+2*p6p7*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4
     . *fb2*hbcm2-hbcm2)+16*(fb2*p1p2*p6p7-2*fb2*p2p4*p6p7)))
      b(4)=ans
      b(15)=ccc*(w1*(8*p6p7*(fmb*fmc*hbcm2-2*fmc*hbcm3+fmc2*hbcm2-2*
     . p3p4s2)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+16*
     . (fmc*hbcm*p1p3*p6p7+fmc*hbcm*p2p3*p6p7-hbcm2*p1p4*p6p7-hbcm2*
     . p2p4*p6p7+p1p3*p3p4*p6p7+p2p3*p3p4*p6p7))+w20*(8*p6p7*(-
     . ffmcfmb*hbcm4-fmb*hbcm3+hbcm4+2*p1p3s2+2*p2p3s2)+16*(p6p7*p3p4
     . )*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2
     . +fmb*hbcm-2*hbcm2)+8*(p6p7*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-2*
     . hbcm2)+16*(-hbcm2*p1p2*p6p7+2*p1p3*p2p3*p6p7-p1p3*p3p4*p6p7-
     . p2p3*p3p4*p6p7))+(4*p6p7*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+
     . 8*fb2*hbcm2+hbcm2)+32*(-fb2*p1p3*p6p7-fb2*p2p3*p6p7+fb2*p3p4*
     . p6p7)))
      b(16)=ccc*(w1*(8*(p6p7*p3p4)*(-fmb+fmc+2*hbcm)+8*p6p7*(ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+16*(-hbcm*p1p4*p6p7-hbcm*
     . p2p4*p6p7))+w20*(8*(p6p7*p2p3)*(-fmb+fmc)+8*(p6p7*p1p3)*(-fmb+
     . fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm
     . +fmb*hbcm2-fmb2*hbcm-fmc*hbcm2))+16*p6p7*(fb2*fmb-fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
      end 
 
      subroutine amp24_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2+
     . fmc2+hbcm2+2*p3p4))
      b(1)=ccc*(w1*(4*p6p7*(2*fmb*p3p4s2-fmb2*fmc*hbcm2-fmc*hbcm4-2*
     . fmc*p3p4s2+fmc3*hbcm2)+8*(p6p7*p3p5*p3p4)*(fmb-fmc)+8*(p6p7*
     . p3p4*p2p3)*(-fmb+fmc)+8*(p6p7*p3p4*p1p3)*(-fmb+fmc)+8*(p6p7*
     . p3p4)*(fmb*hbcm2-fmc*hbcm2)+8*(fmc*hbcm2*p1p3*p6p7+fmc*hbcm2*
     . p2p3*p6p7-fmc*hbcm2*p3p5*p6p7))+(16*(p6p7*p3p4)*(-fb2*fmb+fb2*
     . fmc)+16*(p6p7*p3p5)*(2*fb1*hbcm-fb2*fmb+fb2*fmc)+16*(p6p7*p2p3
     . )*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+16*(p6p7*p1p3)*(-2*fb1*hbcm+
     . fb2*fmb-fb2*fmc)+16*p6p7*(fb1*fmb2*hbcm-fb1*fmc2*hbcm+fb1*
     . hbcm3-fb2*fmb*hbcm2+fb2*fmc*hbcm2)))
      b(4)=ccc*(w1*(4*(p6p7*p3p4)*(fmb2-fmc2-hbcm2)-8*p3p4s2*p6p7)+(8
     . *p6p7*(-fb2*fmb2+fb2*fmc2+fb2*hbcm2)+16*fb2*p3p4*p6p7))
      b(15)=ccc*(w1*(8*p6p7*(-fmb*fmc*hbcm2-fmc2*hbcm2+2*p3p4s2)+8*
     . hbcm2*p3p4*p6p7)+(16*p6p7*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm-fb2*
     . hbcm2)-32*fb2*p3p4*p6p7))
      b(16)=ccc*(w1*(8*(p6p7*p3p4)*(fmb-fmc)-8*fmc*hbcm2*p6p7)+16*
     . p6p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
      end 
 
      subroutine amp23_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(1)=ccc*(w2*(16*(p6p7*p3p5*p2p3)*(fmb-fmc)+4*p6p7*(fmb*fmc2*
     . hbcm2-fmb*hbcm4-fmb3*hbcm2)+8*(2*fmb*hbcm2*p2p3*p6p7-fmb*hbcm2
     . *p3p5*p6p7))+(32*(p6p7*p2p3)*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+16*
     . p6p7*(fb1*fmb2*hbcm-fb1*fmc2*hbcm+fb1*hbcm3)+32*fb1*hbcm*p3p5*
     . p6p7))
      b(4)=ccc*(w2*(4*(p6p7*p3p5)*(-fmb2+fmc2-hbcm2)-8*p3p5s2*p6p7)+(
     . 8*p6p7*(-fb2*fmb2+fb2*fmc2-fb2*hbcm2)-16*fb2*p3p5*p6p7))
      b(15)=ccc*(w2*(8*p6p7*(-fmb*fmc*hbcm2-fmb2*hbcm2+2*p3p5s2)+8*
     . hbcm2*p3p5*p6p7)+(16*p6p7*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*
     . hbcm2)+32*fb2*p3p5*p6p7))
      b(16)=ccc*(w2*(8*(p6p7*p3p5)*(-fmb+fmc)-8*fmb*hbcm2*p6p7)+16*
     . p6p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
      end 
 
      subroutine amp22_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*
     . ffmcfmb*p2p3-fmc2+2*p1p2))
      ans=ccc*(w2*(8*(p6p7*p2p5)*(ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)+8*
     . (p6p7*p1p5)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p3p5*p2p3)*(-
     . ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5*p1p3)*(ffmcfmb*hbcm-fmc)+4*p6p7
     . *(-ffmcfmb*fmb*hbcm4+fmb*fmc*hbcm3)+4*(p6p7*p3p5)*(-ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmc2*hbcm)+8*(p6p7*
     . p2p3)*(2*ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb2*hbcm)+8*(-fmb*
     . fmc*hbcm*p1p3*p6p7-fmb*hbcm2*p1p2*p6p7-2*hbcm*p1p2*p3p5*p6p7+4
     . *hbcm*p1p5*p2p3*p6p7+4*hbcm*p2p3*p2p5*p6p7))+w17*(8*(p6p7*p1p2
     . )*(-fmb*hbcm2+fmc*hbcm2)+8*p6p7*(ffmcfmb*hbcm*p1p3s2-ffmcfmb*
     . hbcm*p2p3s2-fmc*p1p3s2+fmc*p2p3s2)+4*(p6p7*p2p3)*(3*ffmcfmb*
     . fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm)+4*(p6p7*
     . p1p3)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*
     . hbcm)+16*(-hbcm*p1p2*p1p3*p6p7+hbcm*p1p2*p2p3*p6p7))+(2*p6p7*(
     . -4*fb1*fmb*fmc*hbcm+4*fb1*fmc2*hbcm-4*fb2*ffmcfmb*fmb*hbcm2+4*
     . fb2*ffmcfmb*fmc*hbcm2+fmb*hbcm2-fmc*hbcm2)+4*(p6p7*p2p3)*(-4*
     . fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm)+4*(p6p7*p1p3)*(4*fb1*
     . ffmcfmb*hbcm-4*fb2*fmc+3*hbcm)))
      b(1)=ans
      ans=ccc*(w17*(8*p6p7*(-ffmcfmb*p1p3s2+ffmcfmb*p2p3s2-2*p2p3s2)+
     . 16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p6p7*p2p3)*(ffmcfmb
     . *hbcm2-3*fmc*hbcm)+4*(p6p7*p1p3)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p6p7*p1p2)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+8*(p1p2*p1p3*p6p7+
     . p1p2*p2p3*p6p7-2*p1p3*p2p3*p6p7-2*p1p3*p2p4*p6p7-2*p2p3*p2p4*
     . p6p7))+w2*((p6p7*p3p5*p2p3)*(8*ffmcfmb-16)+4*(p6p7*p3p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p5)*(-ffmcfmb*hbcm2-2*fmb*
     . hbcm-fmc*hbcm+2*hbcm2)+8*(p6p7*p1p5)*(ffmcfmb*hbcm2-fmc*hbcm)+
     . 4*p6p7*(-ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmb2*hbcm2+fmb*fmc2*
     . hbcm+fmb2*fmc*hbcm)+8*(-ffmcfmb*p1p3*p3p5*p6p7+2*fmb*hbcm*p1p2
     . *p6p7-2*fmb*hbcm*p2p3*p6p7-4*fmb*hbcm*p2p4*p6p7+p1p2*p3p5*p6p7
     . -2*p2p4*p3p5*p6p7))+((p6p7*p2p3)*(16*fb2*ffmcfmb-32*fb2+4)+(
     . p6p7*p1p3)*(-16*fb2*ffmcfmb-4)+2*p6p7*(4*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2-hbcm2)+16*(fb2*p1p2*p6p7-2*fb2*p2p4*p6p7)))
      b(4)=ans
      b(15)=ccc*(w2*(8*p6p7*(fmb*fmc*hbcm2+2*fmb*hbcm3+fmb2*hbcm2)+8*
     . (p6p7*p3p5)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+16*(-fmb
     . *hbcm*p1p3*p6p7-fmb*hbcm*p2p3*p6p7+2*fmb*hbcm*p3p4*p6p7-hbcm2*
     . p1p5*p6p7-hbcm2*p2p5*p6p7+p3p4*p3p5*p6p7))+w17*(8*p6p7*(-
     . ffmcfmb*hbcm4+fmc*hbcm3)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+8*(p6p7*
     . p1p3)*(ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+16*(-hbcm2*p1p2*p6p7+p1p3
     . *p3p4*p6p7+p2p3*p3p4*p6p7))+(4*p6p7*(-4*fb1*fmc*hbcm-4*fb2*
     . ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+32*fb2*p3p4*p6p7))
      b(16)=ccc*(w2*(8*(p6p7*p3p5)*(fmb-fmc+2*hbcm)+8*p6p7*(-ffmcfmb*
     . fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)+16*(-hbcm*p1p5*
     . p6p7-hbcm*p2p5*p6p7))+w17*(8*(p6p7*p2p3)*(fmb-fmc)+8*(p6p7*
     . p1p3)*(fmb-fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-fmc2*hbcm))+16*p6p7*(fb2*fmb-fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
      end 
 
      subroutine amp21_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      ans4=16*(p3p7*p3p6)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . fmc2*hbcm-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+16*(
     . p6p7*p3p5)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3)+
     . 8*(p6p7*p3p4)*(ffmcfmb**2*hbcm3+2*fmb*hbcm2-fmb2*hbcm-hbcm3)+
     . 16*(p6p7*p2p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . hbcm3)+8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm3-2*fmb*hbcm2+fmb2*hbcm+
     . hbcm3)+8*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*hbcm5+2*
     . ffmcfmb*fmb*hbcm4-ffmcfmb*fmb2*hbcm3+ffmcfmb*fmc2*hbcm3-
     . ffmcfmb*hbcm5+fmb*fmc2*hbcm2-fmc2*hbcm3)+16*(2*ffmcfmb*hbcm*
     . p2p3*p3p6*p5p7-2*ffmcfmb*hbcm*p2p5*p3p6*p3p7+ffmcfmb*hbcm3*
     . p2p5*p6p7-hbcm*p1p3*p2p5*p6p7+2*hbcm*p2p3*p4p6*p5p7-2*hbcm*
     . p2p4*p3p6*p5p7+hbcm*p2p5*p3p4*p6p7+2*hbcm*p2p5*p3p6*p4p7-2*
     . hbcm*p2p5*p3p7*p4p6)
      ans3=16*(p5p7*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p6p7*p1p4)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(p4p6*p3p7*p3p5)*(ffmcfmb*
     . hbcm-hbcm)+16*(p4p6*p3p7*p3p4)*(3*ffmcfmb*hbcm+fmb-3*hbcm)+16*
     . (p5p7*p3p6*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p6p7*p3p5*p3p4)*(
     . ffmcfmb*hbcm+fmb-hbcm)+32*(p4p6*p3p7*p2p3)*(-ffmcfmb*hbcm+hbcm
     . )+16*(p6p7*p3p4*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p6*p3p7*
     . p1p3)*(-3*ffmcfmb*hbcm-fmb+3*hbcm)+16*(p5p7*p3p6*p1p3)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p5*p1p3)*(-ffmcfmb*hbcm-fmb+
     . hbcm)+16*(p6p7*p2p3*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p3p7*
     . p3p6*p3p4)*(ffmcfmb*fmb+ffmcfmb*fmc+3*ffmcfmb*hbcm+fmb-fmc-3*
     . hbcm)+16*(p3p7*p3p6*p1p3)*(-ffmcfmb*fmb-ffmcfmb*fmc-ffmcfmb*
     . hbcm-fmb+fmc+hbcm)+16*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*
     . hbcm3-fmb2*hbcm+hbcm3)+16*(p4p6*p3p7)*(ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-fmb*hbcm2+fmb2*hbcm-hbcm3)+16*(p5p7*p3p6)*(
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+ans4
      ans2=w5*ans3
      ans7=8*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*fmc*hbcm4+ffmcfmb
     . **2*hbcm5+2*ffmcfmb*fmb*fmc*hbcm3+2*ffmcfmb*fmb*hbcm4-ffmcfmb*
     . fmb2*hbcm3-2*ffmcfmb*fmc*hbcm4-ffmcfmb*fmc2*hbcm3-ffmcfmb*
     . hbcm5-2*fmb*fmc*hbcm3-fmb*fmc2*hbcm2+fmb2*fmc*hbcm2+fmc*hbcm4+
     . fmc2*hbcm3)+16*(2*ffmcfmb*hbcm*p2p3*p3p6*p5p7-2*ffmcfmb*hbcm*
     . p2p5*p3p6*p3p7-hbcm*p1p3*p2p5*p6p7)
      ans6=16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(p4p6*p3p7*
     . p1p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p5p7*p3p6*p1p3)*(-ffmcfmb*
     . hbcm-fmb+hbcm)+16*(p6p7*p3p5*p1p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16
     . *(p6p7*p2p3*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*(p3p7*p3p6*p1p3)*
     . (-ffmcfmb*fmb-ffmcfmb*fmc-ffmcfmb*hbcm-fmb+fmc+hbcm)+16*(p4p6*
     . p3p7)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(p5p7*p3p6)*(ffmcfmb
     . **2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-
     . fmb*fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm-
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(
     . p6p7*p3p5)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(p6p7*p2p3)*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm3
     . -2*fmb*hbcm2+fmb2*hbcm+hbcm3)+ans7
      ans5=w12*ans6
      ans8=(8*(p4p6*p3p7)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+8*(p5p7*p3p6)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+8*(p3p7*p3p6)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*
     . ffmcfmb*fmb-4*fb2*ffmcfmb*fmc-4*fb2*fmb+4*fb2*fmc-fmb-fmc-3*
     . hbcm)+8*(p6p7*p3p5)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+8*(p6p7*p2p3)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+
     . hbcm)+8*p6p7*(2*fb1*ffmcfmb**2*hbcm3-4*fb1*ffmcfmb*hbcm3+2*fb1
     . *fmb2*hbcm+2*fb1*hbcm3+4*fb2*ffmcfmb*fmb*hbcm2-4*fb2*fmb*hbcm2
     . -fmb*hbcm2)-32*fb1*hbcm*p2p5*p6p7)
      ans1=w8*(16*(p6p7*p2p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(
     . p3p7*p3p6*p2p5)*(-ffmcfmb*hbcm+hbcm)+16*p6p7*(-ffmcfmb*hbcm*
     . p2p3s2-fmb*p2p3s2+hbcm*p2p3s2)+16*(p4p6*p3p7*p2p3)*(-ffmcfmb*
     . hbcm+fmb+hbcm)+16*(p5p7*p3p6*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*
     . (p6p7*p3p5*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+32*(p3p7*p3p6*p1p2)*(
     . ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p2p3)*(ffmcfmb*fmb+ffmcfmb*
     . fmc-ffmcfmb*hbcm+fmb-fmc+hbcm)+8*(p6p7*p2p3)*(-ffmcfmb**2*
     . hbcm3+2*ffmcfmb*hbcm3+fmb2*hbcm-hbcm3)-16*hbcm*p2p3*p2p5*p6p7)
     . +ans2+ans5+ans8
      ans=ccc*ans1
      b(1)=ans
      ans2=w12*(16*(p5p7*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p3p7*p1p3)*
     . (-ffmcfmb*fmb+ffmcfmb*fmc+ffmcfmb*hbcm+2*fmb-fmc-hbcm)+16*p5p7
     . *(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+8*p3p7*(ffmcfmb**2*fmb*
     . hbcm2-ffmcfmb**2*fmc*hbcm2-ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*
     . hbcm-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm
     . +ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm))+(8*p5p7*(4
     . *fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*p3p7*(4*fb1*
     . ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*ffmcfmb*fmb+4*fb2*ffmcfmb*fmc+8*
     . fb2*fmb-4*fb2*fmc-fmb+fmc+hbcm))
      ans1=w8*(32*(p3p7*p2p5)*(ffmcfmb*hbcm-hbcm)+16*(p5p7*p2p3)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm+hbcm)+8*(
     . p3p7*p2p3)*(ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm-2*fmb+fmc-
     . hbcm))+w5*(16*(p5p7*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p3p7*
     . p1p4)*(ffmcfmb*hbcm-hbcm)+16*(p5p7*p1p3)*(ffmcfmb*hbcm+fmb-
     . hbcm)+8*(p3p7*p3p4)*(ffmcfmb*fmb-ffmcfmb*fmc-3*ffmcfmb*hbcm-2*
     . fmb+fmc+3*hbcm)+8*(p3p7*p1p3)*(-ffmcfmb*fmb+ffmcfmb*fmc+
     . ffmcfmb*hbcm+2*fmb-fmc-hbcm)+8*p5p7*(-2*ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+fmc*hbcm2)+8*p4p7*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmb2
     . *hbcm-hbcm3)+8*p3p7*(ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*fmc*hbcm-2*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*fmc2*hbcm+ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*
     . hbcm)+16*(hbcm*p2p4*p5p7-hbcm*p2p5*p4p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(8*p4p6*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*
     . p3p6*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm))
      ans1=w8*(16*(p4p6*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p3p6*p2p3)*
     . (-ffmcfmb*hbcm-fmb+hbcm))+w5*(16*(p4p6*p3p4)*(-ffmcfmb*hbcm-
     . fmb+hbcm)+8*(p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p6*p1p3
     . )*(ffmcfmb*hbcm+fmb-hbcm)+8*(p3p6*p1p3)*(ffmcfmb*hbcm+fmb-hbcm
     . )+8*p4p6*(-ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+fmb*hbcm2-2*
     . fmb2*hbcm+hbcm3)+8*p3p6*(-ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*
     . hbcm3-ffmcfmb*fmb2*hbcm+ffmcfmb*hbcm3)+16*(-ffmcfmb*hbcm*p1p2*
     . p3p6+ffmcfmb*hbcm*p2p3*p3p6+ffmcfmb*hbcm*p2p4*p3p6+2*ffmcfmb*
     . hbcm*p2p5*p3p6-hbcm*p1p2*p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*p4p6+3*
     . hbcm*p2p5*p4p6))+w12*(16*(p4p6*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8
     . *(p3p6*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+16*p4p6*(-ffmcfmb**2*
     . hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-fmc*hbcm2)+8*p3p6*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*
     . fmc*hbcm2-ffmcfmb**2*hbcm3+ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb2*
     . hbcm+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+16*(-ffmcfmb*hbcm*
     . p1p2*p3p6+ffmcfmb*hbcm*p2p3*p3p6+2*ffmcfmb*hbcm*p2p5*p3p6))+
     . ans2
      ans=ccc*ans1
      b(3)=ans
      ans3=(p4p6*p3p7*p3p4)*(-16*ffmcfmb+16)+(p4p6*p3p7*p1p3)*(16*
     . ffmcfmb-16)+32*(p5p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*
     . (p4p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p1p4)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+(p3p7*p3p6*p3p4)*(16*ffmcfmb**2-
     . 32*ffmcfmb+16)+(p3p7*p3p6*p1p3)*(-16*ffmcfmb**2+32*ffmcfmb-16)
     . +16*(p4p6*p3p7)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*fmb*hbcm
     . +hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . ffmcfmb*hbcm2)+16*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm+ffmcfmb*hbcm2)+16*(p3p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*
     . fmb*hbcm+ffmcfmb*hbcm2)+8*(p6p7*p3p4)*(ffmcfmb**2*hbcm2+2*
     . ffmcfmb*fmb*hbcm+fmb2-hbcm2)+8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm2-
     . 2*ffmcfmb*fmb*hbcm-fmb2+hbcm2)+8*p6p7*(ffmcfmb**2*fmb*hbcm3+
     . ffmcfmb**2*hbcm4+ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc2*hbcm2-ffmcfmb
     . *hbcm4+fmb*fmc2*hbcm-fmc2*hbcm2)+16*(-ffmcfmb*hbcm2*p1p2*p6p7+
     . ffmcfmb*hbcm2*p2p3*p6p7+ffmcfmb*hbcm2*p2p4*p6p7+p1p2*p1p3*p6p7
     . -p1p2*p3p4*p6p7-p1p3*p2p3*p6p7-p1p3*p2p4*p6p7+p2p3*p3p4*p6p7+
     . p2p4*p3p4*p6p7)
      ans2=w5*ans3
      ans5=((p4p6*p3p7)*(32*fb2*ffmcfmb-32*fb2+8)+(p3p7*p3p6)*(-32*
     . fb2*ffmcfmb**2+64*fb2*ffmcfmb-32*fb2-16*ffmcfmb+16)+8*p6p7*(2*
     . fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2-2*fb2*fmb2+2*fb2*
     . hbcm2-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+32*(fb2*p1p2*p6p7-fb2*p2p3
     . *p6p7-fb2*p2p4*p6p7))
      ans4=w12*((p4p6*p3p7*p1p3)*(16*ffmcfmb-16)+16*(p6p7*p2p4)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm
     . )+16*(p6p7*p1p2)*(-ffmcfmb*hbcm2+fmc*hbcm)+(p3p7*p3p6*p1p3)*(-
     . 16*ffmcfmb**2+32*ffmcfmb-16)+16*(p4p6*p3p7)*(-ffmcfmb**2*hbcm2
     . +ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+16*(p3p7*p3p6
     . )*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb
     . *hbcm2-fmc*hbcm)+8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*
     . fmb*hbcm-fmb2+hbcm2)+8*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*
     . fmc*hbcm3+ffmcfmb**2*hbcm4+ffmcfmb*fmb2*hbcm2-2*ffmcfmb*fmc*
     . hbcm3-ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4-fmb*fmc2*hbcm-fmb2*fmc*
     . hbcm+fmc*hbcm3+fmc2*hbcm2)+16*(p1p2*p1p3*p6p7-p1p3*p2p3*p6p7-
     . p1p3*p2p4*p6p7))+ans5
      ans1=w8*((p4p6*p3p7*p2p3)*(-16*ffmcfmb+16)+16*(p6p7*p2p4)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p1p2)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+(p3p7*p3p6*p2p3)*(16*ffmcfmb**2-32*ffmcfmb+16)
     . +8*(p6p7*p2p3)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+fmb2-hbcm2
     . )+16*(-p1p2*p2p3*p6p7+p2p3*p2p4*p6p7+p2p3s2*p6p7))+ans2+ans4
      ans=ccc*ans1
      b(4)=ans
      ans4=8*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*hbcm4-ffmcfmb*fmb
     . *fmc*hbcm2+ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2-ffmcfmb*fmc*
     . hbcm3-2*ffmcfmb*fmc2*hbcm2+ffmcfmb*hbcm4-2*ffmcfmb*p1p3s2+fmb*
     . fmc*hbcm2-fmb*fmc2*hbcm+fmc*hbcm3+2*fmc2*hbcm2+2*p1p3s2)+16*(-
     . ffmcfmb*hbcm2*p2p3*p5p7+ffmcfmb*hbcm2*p2p5*p3p7+hbcm2*p2p4*
     . p5p7-hbcm2*p2p5*p4p7+p1p3*p2p3*p5p7-p1p3*p2p5*p3p7-p2p3*p3p4*
     . p5p7+p2p5*p3p4*p3p7)
      ans3=(p3p7*p3p5*p3p4)*(-16*ffmcfmb+16)+(p3p7*p3p4*p2p3)*(16*
     . ffmcfmb-16)+(p3p7*p3p5*p1p3)*(16*ffmcfmb-16)+(p3p7*p3p4*p1p3)*
     . (16*ffmcfmb-16)+(p3p7*p2p3*p1p3)*(-16*ffmcfmb+16)+16*(p4p7*
     . p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+24*(p5p7*p3p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+16*(p3p7*p1p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p5p7*
     . p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p5p7*(ffmcfmb**2*hbcm4
     . +ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2-fmc*hbcm3)+8*p4p7*(ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3-
     . 2*ffmcfmb*hbcm4-2*fmb*hbcm3+fmb2*hbcm2+hbcm4)+16*(p3p7*p3p5)*(
     . -ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p3p7*p3p4)*(-2*ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-ffmcfmb*fmc
     . *hbcm+ffmcfmb*hbcm2+3*fmb*hbcm-fmb2+fmc*hbcm+hbcm2)+16*(p3p7*
     . p2p3)*(ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*
     . hbcm)+8*(p3p7*p1p3)*(2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*hbcm+fmb2-fmc*hbcm-hbcm2)+
     . ans4
      ans2=w5*ans3
      ans6=((p3p7*p3p5)*(32*fb2*ffmcfmb-32*fb2+8)+(p3p7*p2p3)*(-32*
     . fb2*ffmcfmb+32*fb2-8)+(p3p7*p1p3)*(-32*fb2*ffmcfmb+32*fb2-8)+4
     . *p5p7*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+
     . 4*p3p7*(4*fb1*ffmcfmb*fmc*hbcm-4*fb1*fmb*hbcm-4*fb1*fmc*hbcm+4
     . *fb2*ffmcfmb*hbcm2+4*fb2*fmb2-4*fb2*hbcm2+2*ffmcfmb*hbcm2+2*
     . fmb*hbcm-fmc*hbcm+hbcm2)+32*(fb2*p2p3*p5p7-fb2*p2p5*p3p7))
      ans5=w12*((p3p7*p3p5*p1p3)*(16*ffmcfmb-16)+(p3p7*p2p3*p1p3)*(-
     . 16*ffmcfmb+16)+16*(p3p7*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p5p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p7*p1p3)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+8*p5p7*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*
     . hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)
     . +16*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-fmc*hbcm)+16*(p3p7*p2p3)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p1p3)*(2*ffmcfmb**2*hbcm2
     . +2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*hbcm+
     . fmb2+fmc*hbcm-hbcm2)+8*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*
     . fmc*hbcm3-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*
     . hbcm2+2*ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2+ffmcfmb*hbcm4-2*
     . ffmcfmb*p1p3s2-fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc*
     . hbcm3-fmc2*hbcm2+2*p1p3s2)+16*(p1p3*p2p3*p5p7-p1p3*p2p5*p3p7))
     . +ans6
      ans1=w8*(16*p3p7*(ffmcfmb*p2p3s2-p2p3s2)+(p3p7*p3p5*p2p3)*(-16*
     . ffmcfmb+16)+(p3p7*p2p3*p1p3)*(16*ffmcfmb-16)+16*(p3p7*p2p5)*(2
     . *ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+24*(p5p7*p2p3)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2+hbcm2)+8*
     . (p3p7*p2p3)*(ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*hbcm-fmb2-fmc*
     . hbcm+hbcm2)+16*(p2p3*p2p5*p3p7-p2p3s2*p5p7))+ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans3=w12*(16*(p3p6*p2p5)*(2*ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p6*
     . p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*p1p3)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+8*(p3p6*p1p3)*(ffmcfmb*fmb*hbcm-fmb*hbcm+fmb2)
     . +8*p4p6*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p6*p3p5)*(ffmcfmb
     . **2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+16*(p3p6*p2p3)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*p3p6*(-
     . ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*hbcm3+ffmcfmb
     . *fmc2*hbcm2-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmb2*fmc
     . *hbcm-fmc2*hbcm2)+16*(-ffmcfmb*hbcm2*p1p2*p3p6+ffmcfmb*p1p3*
     . p2p3*p3p6+p1p3*p2p3*p4p6-p1p3*p2p5*p3p6))+((p3p6*p2p3)*(32*fb2
     . *ffmcfmb+8)+4*p4p6*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*
     . hbcm2+hbcm2)+4*p3p6*(4*fb1*ffmcfmb*fmb*hbcm-4*fb1*fmb*hbcm+4*
     . fb2*fmb2+2*ffmcfmb*hbcm2+3*fmb*hbcm-2*hbcm2)+32*(fb2*p2p3*p4p6
     . -fb2*p2p5*p3p6))
      ans2=w5*(32*(p4p6*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6
     . *p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p6*p2p3)*(-2*
     . ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2)+16*(p3p6*p1p4)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+24*(p4p6*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +8*(p3p6*p3p4)*(-ffmcfmb*fmb*hbcm+fmb*hbcm-fmb2)+8*(p3p6*p1p3)
     . *(ffmcfmb*fmb*hbcm-fmb*hbcm+fmb2)+8*p4p6*(ffmcfmb**2*hbcm4+
     . ffmcfmb*hbcm4+3*fmb*hbcm3-fmb2*hbcm2-2*hbcm4)+16*(p3p6*p3p5)*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+16*(p3p6*p2p3
     . )*(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2)+8*p3p6*(-ffmcfmb**2*fmb*
     . hbcm3+ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2-
     . ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4-fmb*fmc2*hbcm+fmc2*hbcm2)+16*
     . (-ffmcfmb*hbcm2*p1p2*p3p6+ffmcfmb*hbcm2*p2p4*p3p6+2*ffmcfmb*
     . hbcm2*p2p5*p3p6+ffmcfmb*p1p3*p2p3*p3p6-ffmcfmb*p2p3*p3p4*p3p6-
     . hbcm2*p1p2*p4p6+hbcm2*p2p4*p4p6+2*hbcm2*p2p5*p4p6+p1p3*p2p3*
     . p4p6-p1p3*p2p5*p3p6-p2p3*p3p4*p4p6+p2p5*p3p4*p3p6))+ans3
      ans1=w8*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6
     . *p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p6*p2p3)*(-2*
     . ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2+fmb*hbcm-
     . fmb2)+16*(-ffmcfmb*p2p3s2*p3p6+p2p3*p2p5*p3p6-p2p3s2*p4p6))+
     . ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w12*(24*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p2p3*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+4*p1p3*(-
     . ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*fmc+
     . fmb*hbcm-2*fmb2+fmc*hbcm+hbcm2)+4*(2*ffmcfmb**2*fmb*hbcm3+2*
     . ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3+2*
     . ffmcfmb*fmb2*hbcm2-3*ffmcfmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm2-
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmb*fmc2*hbcm-2*fmb2*fmc*hbcm+
     . fmc*hbcm3+2*fmc2*hbcm2-2*p1p2*p1p3+2*p1p3*p2p3+6*p1p3*p2p5))+2
     . *(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmb*
     . hbcm+4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2-8*fb2*ffmcfmb*hbcm2
     . -4*fb2*fmb*fmc-8*fb2*fmb2+4*fb2*hbcm2-8*fb2*p1p2+8*fb2*p2p3+24
     . *fb2*p2p5-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2)
      ans1=w8*(24*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p1p2*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p2p3*(ffmcfmb**2*hbcm2+3*
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-4*ffmcfmb*hbcm2+fmb*fmc-5*
     . fmb*hbcm+2*fmb2-fmc*hbcm+3*hbcm2)+8*(p1p2*p2p3-3*p2p3*p2p5-
     . p2p3s2))+w5*(8*p2p5*(-3*ffmcfmb*hbcm2-fmc*hbcm)+8*p2p4*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p1p4*(-ffmcfmb*hbcm2-fmb*hbcm
     . +hbcm2)+4*p3p4*(ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm+2*ffmcfmb*hbcm2+fmb*fmc+fmb*hbcm+2*fmb2-fmc*hbcm-3*
     . hbcm2)+4*p1p3*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-fmb*fmc+fmb*hbcm-2*fmb2+fmc*hbcm+hbcm2)+4*(2*ffmcfmb
     . **2*fmb*hbcm3+ffmcfmb**2*hbcm4+ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*
     . fmb*hbcm3+2*ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*hbcm3+3*ffmcfmb*
     . fmc2*hbcm2+2*ffmcfmb*hbcm2*p1p2-2*ffmcfmb*hbcm2*p2p3-ffmcfmb*
     . hbcm4+3*fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc*hbcm3-3*fmc2*hbcm2-2*
     . p1p2*p1p3+2*p1p2*p3p4+2*p1p3*p2p3+6*p1p3*p2p5-2*p2p3*p3p4-6*
     . p2p5*p3p4))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=4*(ffmcfmb**2*fmb*hbcm4-2*ffmcfmb**2*hbcm5+ffmcfmb*fmb*fmc
     . *hbcm3-3*ffmcfmb*fmb*hbcm4+ffmcfmb*fmb2*hbcm3-ffmcfmb*fmc*
     . hbcm4-2*ffmcfmb*fmc2*hbcm3-2*ffmcfmb*hbcm*p1p3s2+2*ffmcfmb*
     . hbcm3*p1p2+2*ffmcfmb*hbcm5-2*fmb*fmc*hbcm3-2*fmb*fmc2*hbcm2-2*
     . fmb*p1p3s2+fmb2*fmc*hbcm2+fmc*hbcm4+2*fmc2*hbcm3-2*hbcm*p1p2*
     . p1p3+2*hbcm*p1p2*p3p4+4*hbcm*p1p3*p2p5+2*hbcm*p1p3s2-4*hbcm*
     . p1p4*p2p3-4*hbcm*p2p3*p2p4+4*hbcm*p2p4*p3p5-8*hbcm*p2p5*p3p4)
      ans3=8*p2p5*(-2*ffmcfmb*hbcm3-fmc*hbcm2)+8*p2p4*(-ffmcfmb*hbcm3
     . +fmb*hbcm2+hbcm3)+8*p1p4*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(
     . p3p5*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p3p4*p2p3)*(3*ffmcfmb*
     . hbcm+fmb-2*hbcm)+16*(p3p5*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(
     . p3p4*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p2p3*p1p3)*(-3*ffmcfmb*
     . hbcm-fmb+2*hbcm)+8*p3p5*(-2*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2
     . )+4*p3p4*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm-5*fmb*hbcm2+3*fmb2*hbcm+fmc*hbcm2+2*hbcm3)+
     . 8*p2p3*(2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2
     . -2*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2+fmc2*hbcm)+4*p1p3*(2*
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*
     . hbcm+3*fmb*hbcm2-fmb2*hbcm-fmc*hbcm2-2*hbcm3)+ans4
      ans2=w5*ans3
      ans6=(8*p3p5*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*
     . p2p3*(-4*fb1*ffmcfmb*hbcm+8*fb1*hbcm-4*fb2*fmb-3*hbcm)+4*p1p3*
     . (-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+2*(8*fb1*
     . ffmcfmb*hbcm3+4*fb1*fmb*fmc*hbcm-4*fb1*fmb2*hbcm-8*fb1*hbcm*
     . p1p2+16*fb1*hbcm*p2p5-8*fb1*hbcm3-4*fb2*ffmcfmb*fmb*hbcm2+4*
     . fb2*ffmcfmb*fmc*hbcm2+12*fb2*fmb*hbcm2-4*fb2*fmc*hbcm2+fmb*
     . hbcm2-fmc*hbcm2+2*hbcm3))
      ans5=w12*(16*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+16*(p3p5*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(
     . p2p3*p1p3)*(-3*ffmcfmb*hbcm-fmb+2*hbcm)+16*p3p5*(-ffmcfmb**2*
     . hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-fmc*hbcm2)+8*p2p3*(2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm+2*fmc*
     . hbcm2-fmc2*hbcm)+4*p1p3*(2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+3*fmb*hbcm2-fmb2*hbcm+fmc*hbcm2
     . -2*hbcm3)+4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4-2*
     . ffmcfmb**2*hbcm5-2*ffmcfmb*fmb*fmc*hbcm3-3*ffmcfmb*fmb*hbcm4+
     . ffmcfmb*fmb2*hbcm3+3*ffmcfmb*fmc*hbcm4+ffmcfmb*fmc2*hbcm3-2*
     . ffmcfmb*hbcm*p1p3s2+2*ffmcfmb*hbcm5+3*fmb*fmc*hbcm3+fmb*fmc2*
     . hbcm2-2*fmb*p1p3s2-fmb2*fmc*hbcm2-2*fmc*hbcm4-fmc2*hbcm3-2*
     . hbcm*p1p2*p1p3+4*hbcm*p1p3*p2p5+2*hbcm*p1p3s2))+ans6
      ans1=w8*(16*p2p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*p1p2*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p3p5*p2p3)*(-ffmcfmb*hbcm-
     . fmb+hbcm)+8*(p2p3*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+4*p2p3*(2*
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-4*ffmcfmb
     . *hbcm3+fmb*fmc*hbcm+fmb*hbcm2-3*fmb2*hbcm-fmc*hbcm2+2*hbcm3)+8
     . *(3*ffmcfmb*hbcm*p2p3s2+fmb*p2p3s2-hbcm*p1p2*p2p3+2*hbcm*p2p3*
     . p2p5-2*hbcm*p2p3s2))+ans2+ans5
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(w8*(16*p4p6*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(p3p6*
     . p2p3)*(2*ffmcfmb*hbcm-hbcm)+8*p3p6*(ffmcfmb**2*hbcm3-ffmcfmb*
     . hbcm3+fmb*hbcm2-fmb2*hbcm)+16*hbcm*p2p3*p4p6)+w5*(8*p4p6*(-fmb
     . *hbcm2-4*hbcm3)+8*p3p6*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-3*
     . ffmcfmb*hbcm3+fmc2*hbcm)+8*(2*ffmcfmb*hbcm*p2p3*p3p6-4*ffmcfmb
     . *hbcm*p3p5*p3p6+hbcm*p1p3*p3p6+4*hbcm*p1p3*p4p6-2*hbcm*p1p4*
     . p3p6+4*hbcm*p2p3*p4p6-hbcm*p3p4*p3p6-4*hbcm*p3p4*p4p6-6*hbcm*
     . p3p5*p4p6))+w12*(16*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p3p6*(
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-3*ffmcfmb
     . *hbcm3+fmc*hbcm2-fmc2*hbcm)+8*(2*ffmcfmb*hbcm*p2p3*p3p6-4*
     . ffmcfmb*hbcm*p3p5*p3p6+hbcm*p1p3*p3p6+2*hbcm*p1p3*p4p6))+(8*
     . p3p6*(2*fb1*hbcm-hbcm)+32*fb1*hbcm*p4p6))
      b(10)=ccc*(w8*((p3p6*p2p3)*(8*ffmcfmb+8)+24*p4p6*(ffmcfmb*hbcm2
     . +fmb*hbcm-hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+fmb
     . *hbcm-hbcm2)+24*p2p3*p4p6)+w5*((p3p6*p3p4)*(8*ffmcfmb+8)+(p3p6
     . *p1p3)*(-8*ffmcfmb-8)+8*p4p6*(3*ffmcfmb*hbcm2-fmb*hbcm)+8*p3p6
     . *(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+24*(-p1p3*
     . p4p6+p3p4*p4p6))+w12*((p3p6*p1p3)*(-8*ffmcfmb-8)+24*p4p6*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm-2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)-24*p1p3*p4p6)+
     . (p3p6*(-16*fb2*ffmcfmb-16*fb2-4)-48*fb2*p4p6))
      ans2=4*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-8*
     . fb2*hbcm2+4*fb2*p1p3+8*fb2*p2p3-12*fb2*p3p5-hbcm2)
      ans1=w8*(24*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p2p3*(-4*
     . ffmcfmb*hbcm2-5*fmb*hbcm-fmc*hbcm+6*hbcm2)+8*p1p3*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+4*(-ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*hbcm3
     . -ffmcfmb*fmc*hbcm3+4*ffmcfmb*hbcm4-fmb*fmc*hbcm2+5*fmb*hbcm3-2
     . *fmb2*hbcm2+fmc*hbcm3-3*hbcm4-2*p1p3*p2p3+6*p2p3*p3p5-4*p2p3s2
     . ))+w5*(4*p3p4*(fmb*hbcm+fmc*hbcm+6*hbcm2)+8*p3p5*(3*ffmcfmb*
     . hbcm2+fmc*hbcm)+8*p2p3*(-2*ffmcfmb*hbcm2-fmc*hbcm)+4*p1p3*(-2*
     . ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm-4*hbcm2)+4*(-ffmcfmb**2*hbcm4+
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+4*ffmcfmb*hbcm4+2*fmc*
     . hbcm3+3*fmc2*hbcm2-4*hbcm2*p1p4-2*hbcm2*p2p4+4*p1p3*p2p3-2*
     . p1p3*p3p4-6*p1p3*p3p5+2*p1p3s2-4*p2p3*p3p4+6*p3p4*p3p5))+w12*(
     . 24*p3p5*(ffmcfmb*hbcm2-fmc*hbcm)+16*p2p3*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+4*p1p3*(-2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-4*hbcm2)+4*(-
     . ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3+4*
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2-4*fmc*hbcm3-2*fmc2*hbcm2+4*p1p3*
     . p2p3-6*p1p3*p3p5+2*p1p3s2))+ans2
      ans=ccc*ans1
      b(11)=ans
      ans=ccc*(w8*(16*p5p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(p3p7*
     . p3p5)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*p2p3)*(-4*ffmcfmb*hbcm+3*
     . hbcm)+16*(p3p7*p1p3)*(-ffmcfmb*hbcm+hbcm)+8*p3p7*(-ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-2*
     . fmb*hbcm2+fmb2*hbcm+fmc*hbcm2-2*hbcm3)-16*hbcm*p2p3*p5p7)+w5*(
     . 8*p5p7*(2*ffmcfmb*hbcm3+fmc*hbcm2)+16*p4p7*(ffmcfmb*hbcm3-
     . hbcm3)+8*(p3p7*p3p4)*(-4*ffmcfmb*hbcm+3*hbcm)+8*(p3p7*p1p3)*(2
     . *ffmcfmb*hbcm-hbcm)+8*p3p7*(-ffmcfmb**2*hbcm3-ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+16*(-hbcm*p1p3*p5p7+
     . hbcm*p1p4*p3p7+hbcm*p2p3*p4p7+2*hbcm*p3p4*p5p7-hbcm*p3p5*p4p7)
     . )+w12*(16*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p1p3)*(2*
     . ffmcfmb*hbcm-hbcm)+8*p3p7*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3-fmc
     . *hbcm2+fmc2*hbcm)-16*hbcm*p1p3*p5p7)+(8*p3p7*(-2*fb1*hbcm+hbcm
     . )-32*fb1*hbcm*p5p7))
      b(12)=ans
      b(13)=ccc*(w8*(4*p2p3*(-4*ffmcfmb*hbcm+fmb-fmc+4*hbcm)+4*(-2*
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+4*ffmcfmb
     . *hbcm3-fmb*fmc*hbcm-fmb*hbcm2+3*fmb2*hbcm+fmc*hbcm2-2*hbcm3))+
     . w5*(4*p3p4*(-4*ffmcfmb*hbcm+fmb-fmc-2*hbcm)+4*p1p3*(4*ffmcfmb*
     . hbcm-fmb+fmc)+4*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb
     . *fmc*hbcm2-4*fmc2*hbcm+6*hbcm*p1p4+2*hbcm*p2p4))+w12*(4*p1p3*(
     . 4*ffmcfmb*hbcm-fmb+fmc)+4*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+3*fmc2*hbcm))+8*(-fb2*fmb
     . +fb2*fmc+hbcm))
      b(14)=ccc*(w8*((p3p7*p2p3)*(-8*ffmcfmb+16)+24*p5p7*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+3*fmb*hbcm+fmc*hbcm-2*
     . hbcm2)+24*p2p3*p5p7)+w5*((p3p7*p3p4)*(-8*ffmcfmb+16)+(p3p7*
     . p1p3)*(8*ffmcfmb-16)+8*p5p7*(3*ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*
     . (-ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+fmc*hbcm)+
     . 24*(-p1p3*p5p7+p3p4*p5p7))+w12*((p3p7*p1p3)*(8*ffmcfmb-16)+24*
     . p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-2*fmc*hbcm)-24*p1p3*p5p7)+(
     . p3p7*(16*fb2*ffmcfmb-32*fb2+4)-48*fb2*p5p7))
      ans2=w12*((p3p7*p3p6*p1p3)*(-32*ffmcfmb+16)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*p3p6)*(-2*ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p1p3)*
     . (ffmcfmb*hbcm2-fmc*hbcm)+8*p6p7*(-ffmcfmb**2*hbcm4+2*ffmcfmb*
     . fmc*hbcm3-fmc2*hbcm2-2*p1p3s2)+16*(p3p7*p3p6)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+16*(p1p3*p3p4
     . *p6p7+p1p3*p3p6*p5p7-p1p3*p3p7*p4p6))+((p3p7*p3p6)*(-64*fb2*
     . ffmcfmb+32*fb2-16)+8*p6p7*(4*fb2*ffmcfmb*hbcm2-hbcm2)+32*(-fb2
     . *p1p3*p6p7+fb2*p3p4*p6p7+fb2*p3p6*p5p7-fb2*p3p7*p4p6))
      ans1=w8*((p3p7*p3p6*p2p3)*(32*ffmcfmb-16)+16*(p4p6*p3p7)*(2*
     . ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2
     . -fmb*hbcm+hbcm2)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2
     . )+16*(p6p7*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p6p7*(-
     . ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3-fmb2*hbcm2+hbcm4)+16*(
     . p3p7*p3p6)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2-
     . fmb*hbcm)+16*(-hbcm2*p2p3*p6p7+p1p3*p2p3*p6p7-p2p3*p3p4*p6p7-
     . p2p3*p3p6*p5p7+p2p3*p3p7*p4p6))+w5*((p3p7*p3p6*p3p4)*(32*
     . ffmcfmb-16)+(p3p7*p3p6*p1p3)*(-32*ffmcfmb+16)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2-hbcm2)+8*p6p7*(-ffmcfmb**2*hbcm4+fmc2*hbcm2-2*
     . p1p3s2-2*p3p4s2)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*
     . hbcm2)+16*(ffmcfmb*hbcm2*p1p3*p6p7-ffmcfmb*hbcm2*p3p4*p6p7-
     . ffmcfmb*hbcm2*p3p6*p4p7-2*ffmcfmb*hbcm2*p3p6*p5p7-hbcm2*p1p4*
     . p6p7-hbcm2*p4p6*p4p7-2*hbcm2*p4p6*p5p7+2*p1p3*p3p4*p6p7+p1p3*
     . p3p6*p5p7-p1p3*p3p7*p4p6-p3p4*p3p6*p5p7+p3p4*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w8*(32*(p4p6*p3p7)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6
     . )*(ffmcfmb*hbcm-hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm-hbcm)+8*
     . p6p7*(ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-fmb2*hbcm+hbcm3))+w5*(
     . 16*(p4p6*p3p7)*(ffmcfmb*hbcm-2*hbcm)+8*p6p7*(ffmcfmb**2*hbcm3+
     . fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p6p7+ffmcfmb*hbcm*p3p4*p6p7-
     . ffmcfmb*hbcm*p3p6*p3p7-ffmcfmb*hbcm*p3p6*p4p7-2*ffmcfmb*hbcm*
     . p3p6*p5p7-hbcm*p1p4*p6p7-hbcm*p4p6*p4p7-3*hbcm*p4p6*p5p7))+w12
     . *(8*p6p7*(ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*
     . p6p7-ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*hbcm*p3p6*p5p7))-8*hbcm*
     . p6p7)
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp20_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      ans4=16*(2*ffmcfmb*hbcm*p1p3*p3p6*p4p7-2*ffmcfmb*hbcm*p1p3*p3p7
     . *p4p6+2*ffmcfmb*hbcm*p2p3*p3p6*p4p7-2*ffmcfmb*hbcm*p2p3*p3p7*
     . p4p6-2*ffmcfmb*hbcm*p3p5*p3p6*p4p7+2*ffmcfmb*hbcm*p3p5*p3p7*
     . p4p6-ffmcfmb*hbcm3*p4p6*p5p7+fmc*hbcm2*p1p4*p6p7-fmc*hbcm2*
     . p5p6*p5p7-fmc*p1p3*p3p4*p6p7-fmc2*hbcm*p1p3*p6p7+hbcm*p1p4*
     . p3p4*p6p7+hbcm*p3p4*p4p6*p4p7-hbcm*p3p4*p5p6*p5p7)
      ans3=16*(p4p7*p3p6*p3p4)*(fmc+hbcm)+16*(p4p7*p4p6)*(-ffmcfmb*
     . hbcm3+fmc*hbcm2)+16*(p4p6*p3p7*p3p4)*(2*ffmcfmb*hbcm+hbcm)+16*
     . (p5p7*p3p6*p3p4)*(2*ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5*p3p4)*(
     . ffmcfmb*hbcm+fmc)+16*(p6p7*p3p4*p2p3)*(-ffmcfmb*hbcm-fmc)+16*(
     . p5p7*p3p6)*(ffmcfmb*fmc*hbcm2+fmc2*hbcm)+16*(p4p7*p3p6)*(
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmc*hbcm2+fmc2*hbcm)+16*(
     . p3p7*p3p6)*(2*ffmcfmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm)+16*(p6p7*
     . p3p5)*(ffmcfmb*fmc*hbcm2+fmc2*hbcm)+16*(p6p7*p2p3)*(-ffmcfmb*
     . fmc*hbcm2-fmc2*hbcm)+16*(p4p6*p3p7)*(-ffmcfmb**2*hbcm3+2*
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p3p4)*(ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmc*hbcm2+fmc2*hbcm)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2*
     . hbcm+ffmcfmb*fmc+2*ffmcfmb*hbcm)+8*p6p7*(ffmcfmb**2*fmc*hbcm4+
     . 2*ffmcfmb*hbcm*p3p4s2+2*fmc*p3p4s2+2*fmc2*hbcm3-fmc3*hbcm2-2*
     . hbcm*p3p4s2)+ans4
      ans2=w1*ans3
      ans8=16*(2*ffmcfmb*hbcm*p1p3*p3p6*p4p7-2*ffmcfmb*hbcm*p1p3*p3p6
     . *p5p7-2*ffmcfmb*hbcm*p1p3*p3p7*p4p6+2*ffmcfmb*hbcm*p1p3*p3p7*
     . p5p6+2*ffmcfmb*hbcm*p2p3*p3p6*p4p7-2*ffmcfmb*hbcm*p2p3*p3p6*
     . p5p7-2*ffmcfmb*hbcm*p2p3*p3p7*p4p6+2*ffmcfmb*hbcm*p2p3*p3p7*
     . p5p6-2*ffmcfmb*hbcm*p3p5*p3p7*p5p6+2*ffmcfmb*hbcm*p3p6*p3p7*
     . p4p5-fmc*p1p3*p3p4*p6p7+fmc*p3p5*p3p6*p5p7-2*hbcm*p1p3*p4p5*
     . p6p7-2*hbcm*p1p3*p4p6*p5p7+2*hbcm*p1p3*p4p7*p5p6+hbcm*p1p4*
     . p3p4*p6p7-hbcm*p1p4*p3p5*p6p7-2*hbcm*p1p4*p3p6*p5p7+2*hbcm*
     . p1p4*p3p7*p5p6+2*hbcm*p1p5*p3p4*p6p7+2*hbcm*p1p5*p3p6*p4p7-2*
     . hbcm*p1p5*p3p7*p4p6-2*hbcm*p2p3*p4p5*p6p7-4*hbcm*p2p3*p4p6*
     . p5p7+4*hbcm*p2p3*p4p7*p5p6+hbcm*p3p4*p4p6*p4p7+4*hbcm*p3p4*
     . p4p6*p5p7+3*hbcm*p3p4*p5p6*p5p7+hbcm*p3p5*p3p7*p4p6+2*hbcm*
     . p3p5*p4p5*p6p7-3*hbcm*p3p5*p4p6*p4p7-4*hbcm*p3p5*p4p7*p5p6-
     . hbcm*p3p5*p5p6*p5p7+2*hbcm*p3p6*p4p5*p5p7+2*hbcm*p3p7*p4p5*
     . p4p6)
      ans7=16*(p3p7*p3p6)*(ffmcfmb*fmb*fmc*hbcm-2*ffmcfmb*fmb*hbcm2+2
     . *ffmcfmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm)+16*(p4p6*p3p7)*(-ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3-fmb*hbcm2+2*fmb2*
     . hbcm+fmc*hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2
     . +2*fmc2*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*
     . hbcm2+2*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm+3*
     . fmc2*hbcm)+8*(p6p7*p3p4)*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2
     . +2*ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm+2*fmb*hbcm2-4*fmb2*hbcm+
     . fmc2*hbcm)+16*(p3p7*p3p6*p3p5)*(ffmcfmb**2*hbcm+ffmcfmb*fmc-2*
     . ffmcfmb*hbcm)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2*hbcm+ffmcfmb*fmc
     . +2*ffmcfmb*hbcm)+8*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*fmc*
     . hbcm4-2*ffmcfmb*fmb2*hbcm3+2*ffmcfmb*hbcm*p3p4s2+2*ffmcfmb*
     . hbcm*p3p5s2+2*fmb*fmc*hbcm3+fmb*fmc2*hbcm2-2*fmb2*fmc*hbcm2+2*
     . fmc*p3p4s2+2*fmc*p3p5s2+2*fmc2*hbcm3-fmc3*hbcm2-2*hbcm*p3p4s2)
     . +ans8
      ans6=16*(p4p7*p3p6*p3p4)*(fmc+hbcm)+16*(p5p6*p4p7)*(-fmb*hbcm2+
     . fmc*hbcm2-2*hbcm3)+16*(p5p7*p4p6)*(-fmb*hbcm2+fmc*hbcm2+2*
     . hbcm3)+16*(p6p7*p1p4)*(-fmb*hbcm2+fmc*hbcm2)+16*(p6p7*p1p3)*(-
     . fmb*fmc*hbcm-fmc2*hbcm)+16*(p5p7*p5p6)*(ffmcfmb*hbcm3+fmc*
     . hbcm2)+16*(p4p7*p4p6)*(-ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2)+
     . 16*(p6p7*p4p5)*(-ffmcfmb*hbcm3-fmc*hbcm2+2*hbcm3)+16*(p6p7*
     . p1p5)*(ffmcfmb*hbcm3+fmc*hbcm2)+16*(p4p7*p3p6*p3p5)*(-2*
     . ffmcfmb*hbcm+fmc-3*hbcm)+16*(p4p6*p3p7*p3p4)*(2*ffmcfmb*hbcm+
     . hbcm)+16*(p5p7*p3p6*p3p4)*(2*ffmcfmb*hbcm+fmc+4*hbcm)+16*(p6p7
     . *p3p5*p3p4)*(2*ffmcfmb*hbcm+2*fmc-hbcm)+16*(p6p7*p3p5*p2p3)*(-
     . ffmcfmb*hbcm-fmc)+16*(p6p7*p3p4*p2p3)*(-ffmcfmb*hbcm-fmc)+16*(
     . p6p7*p3p5*p1p3)*(-2*ffmcfmb*hbcm-fmc)+16*(p5p6*p3p7)*(-ffmcfmb
     . *fmb*hbcm2-ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+16*(p4p7*p3p6)*(
     . -ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmb*hbcm2-2*fmb2*hbcm+fmc*hbcm2+fmc2*hbcm)+16*(p6p7*p2p3)
     . *(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmc2*hbcm)+
     . ans7
      ans5=w18*ans6
      ans11=16*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmc*
     . hbcm2-2*hbcm3)+16*(p3p7*p3p6)*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . fmc*hbcm-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*fmc2*hbcm
     . +ffmcfmb*hbcm3-fmc*hbcm2+fmc2*hbcm)+16*(p6p7*p3p5)*(-ffmcfmb**
     . 2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-fmc*hbcm2)+16*(p6p7*p2p3)*(ffmcfmb**2*hbcm3+ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*
     . hbcm2)+8*(p6p7*p1p3)*(ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-2*fmb*
     . fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(p3p7*p3p6*p1p3)*(ffmcfmb**
     . 2*hbcm+ffmcfmb*fmc-2*ffmcfmb*hbcm)+8*p6p7*(-ffmcfmb**2*fmb*
     . hbcm4+ffmcfmb**2*fmc*hbcm4-ffmcfmb**2*hbcm5-ffmcfmb*fmb2*hbcm3
     . +ffmcfmb*fmc2*hbcm3+ffmcfmb*hbcm5+2*fmb*fmc*hbcm3+fmb*fmc2*
     . hbcm2-fmb2*fmc*hbcm2-fmc*hbcm4-2*fmc*p1p3s2-fmc2*hbcm3)+16*(-
     . hbcm*p1p3*p1p4*p6p7-hbcm*p1p3*p4p6*p4p7+hbcm*p1p3*p5p6*p5p7)
      ans10=16*(p5p7*p5p6)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p6p7*p1p4)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(p4p7*p3p6*p3p5)*(-ffmcfmb*
     . hbcm+hbcm)+32*(p5p7*p3p6*p3p4)*(ffmcfmb*hbcm-hbcm)+32*(p5p7*
     . p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+32*(p4p7*p3p6*p2p3)*(ffmcfmb*
     . hbcm-hbcm)+32*(p3p7*p3p6*p1p4)*(ffmcfmb*hbcm-hbcm)+16*(p4p6*
     . p3p7*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+16*(p5p7*p3p6*p1p3)*(-2*
     . ffmcfmb*hbcm+fmc+2*hbcm)+16*(p4p7*p3p6*p1p3)*(2*ffmcfmb*hbcm+
     . fmc-3*hbcm)+16*(p6p7*p3p5*p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*
     . p3p4*p1p3)*(ffmcfmb*hbcm+fmc-hbcm)+16*(p6p7*p2p3*p1p3)*(
     . ffmcfmb*hbcm-fmc)+16*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb2*hbcm-fmc*hbcm2+2*
     . hbcm3)+16*(p6p7*p3p4)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+16*(p4p6*p3p7)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*hbcm3-fmb*hbcm2+fmb2*hbcm)+ans11
      ans9=w13*ans10
      ans12=(16*(p5p7*p3p6)*(-2*fb2*fmc+hbcm)+32*(p4p6*p3p7)*(-fb1*
     . hbcm+hbcm)+16*(p4p7*p3p6)*(-2*fb1*hbcm-2*fb2*fmc-hbcm)+8*(p6p7
     . *p3p5)*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+3*hbcm)+8*(p6p7*p3p4)*(-
     . 4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmc-hbcm)+8*(p6p7*p2p3)*(4
     . *fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm)+8*p6p7*(-2*fb1*ffmcfmb**2*
     . hbcm3+2*fb1*fmc2*hbcm-4*fb2*fmc*hbcm2+ffmcfmb*hbcm3+2*fmb*
     . hbcm2-fmc*hbcm2)+8*(p3p7*p3p6)*(4*fb1*ffmcfmb**2*hbcm-8*fb1*
     . ffmcfmb*hbcm-4*fb2*ffmcfmb*fmc-4*ffmcfmb*hbcm-fmc+6*hbcm)+32*(
     . -fb1*hbcm*p1p4*p6p7-fb1*hbcm*p4p6*p4p7+fb1*hbcm*p5p6*p5p7+fb2*
     . fmc*p1p3*p6p7))
      ans1=ans2+ans5+ans9+ans12
      ans=ccc*ans1
      b(1)=ans
      ans3=w13*(8*(p5p7*p1p3)*(3*ffmcfmb*hbcm+fmb-2*fmc-2*hbcm)+8*(
     . p4p7*p1p3)*(-ffmcfmb*hbcm+fmb-2*fmc+2*hbcm)+8*(p3p7*p1p3)*(
     . ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+8*p4p7*(2*ffmcfmb*
     . fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb
     . *hbcm2+2*fmb2*hbcm+2*fmc*hbcm2-hbcm3)+8*p5p7*(2*ffmcfmb**2*
     . hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-
     . 2*fmb*fmc*hbcm-fmb*hbcm2+2*fmc*hbcm2+hbcm3)+8*p3p7*(ffmcfmb**2
     . *fmb*hbcm2-ffmcfmb**2*fmc*hbcm2+ffmcfmb**2*hbcm3-ffmcfmb*fmb*
     . fmc*hbcm+ffmcfmb*fmb2*hbcm-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*
     . hbcm2))+(4*p5p7*(4*fb1*ffmcfmb*hbcm-4*fb2*fmb+8*fb2*fmc-7*hbcm
     . )+4*p4p7*(4*fb1*ffmcfmb*hbcm-4*fb2*fmb+8*fb2*fmc+hbcm)+4*p3p7*
     . (4*fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmb+4*fb2*ffmcfmb*fmc+4*fb2*
     . fmc-fmb+fmc-3*hbcm))
      ans2=w18*(8*(p5p7*p3p5)*(-ffmcfmb*hbcm+fmb-2*fmc)+8*(p4p7*p3p5)
     . *(-ffmcfmb*hbcm+fmb-2*fmc+4*hbcm)+8*(p5p7*p3p4)*(-ffmcfmb*hbcm
     . +fmb-2*fmc-4*hbcm)+8*(p4p7*p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc)+8*(
     . p3p7*p3p5)*(ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+8*(p3p7*
     . p3p4)*(ffmcfmb*fmb-ffmcfmb*fmc-3*ffmcfmb*hbcm-fmc)+8*p5p7*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmb2*hbcm-4*fmc2*hbcm)+8*p4p7*(-
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+5*fmb2*hbcm-2*fmc2*hbcm)+8*p3p7*(
     . ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*fmc*hbcm2-ffmcfmb*fmb*fmc*hbcm
     . +ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmb2*hbcm-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*fmc2*hbcm-fmb*fmc*hbcm-fmc2*hbcm)+16*(-ffmcfmb*hbcm*
     . p1p3*p4p7+ffmcfmb*hbcm*p1p3*p5p7+ffmcfmb*hbcm*p1p4*p3p7-
     . ffmcfmb*hbcm*p1p5*p3p7+2*hbcm*p1p4*p5p7-2*hbcm*p1p5*p4p7-hbcm*
     . p3p7*p4p5-2*hbcm*p4p5*p5p7))+ans3
      ans1=w1*(8*(p5p7*p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc)+8*(p4p7*p3p4)*
     . (-ffmcfmb*hbcm+fmb-2*fmc)+8*p5p7*(-2*ffmcfmb*fmc*hbcm2+fmb*fmc
     . *hbcm-fmc2*hbcm)+8*(p3p7*p3p4)*(ffmcfmb*fmb-ffmcfmb*fmc-3*
     . ffmcfmb*hbcm-fmc)+8*p4p7*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*
     . hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmc2*hbcm)+8*p3p7*(-
     . ffmcfmb**2*fmc*hbcm2+ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*fmc2*hbcm-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p4p7+
     . ffmcfmb*hbcm*p1p4*p3p7+hbcm*p4p5*p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans3=w13*(16*(p3p6*p3p5)*(-ffmcfmb*hbcm+hbcm)+16*(p3p6*p3p4)*(
     . ffmcfmb*hbcm-hbcm)+16*(p3p6*p1p5)*(ffmcfmb*hbcm-hbcm)+16*(p3p6
     . *p1p4)*(-ffmcfmb*hbcm+hbcm)+8*(p5p6*p1p3)*(-ffmcfmb*hbcm-fmc+2
     . *hbcm)+8*(p4p6*p1p3)*(3*ffmcfmb*hbcm-fmc-2*hbcm)+8*(p3p6*p1p3)
     . *(ffmcfmb*hbcm-fmc)+8*p5p6*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+8*
     . p4p6*(2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-3
     . *ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb2*hbcm+fmc*hbcm2+hbcm3)+8*p3p6*
     . (ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmb2*hbcm-
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*fmc2*hbcm-ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+2*fmb2*hbcm+fmc*hbcm2-2*fmc2*hbcm))+(4*p5p6*(4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmc+hbcm)+4*p4p6*(4*fb1*ffmcfmb*hbcm+4*fb2*
     . fmc-7*hbcm)+4*p3p6*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm))
      ans2=w1*(8*(p5p6*p3p4)*(-ffmcfmb*hbcm-fmc)+8*(p4p6*p3p4)*(-
     . ffmcfmb*hbcm-fmc)+8*(p3p6*p3p4)*(ffmcfmb*hbcm-fmc)+8*p5p6*(-
     . ffmcfmb*fmc*hbcm2-fmc2*hbcm)+8*p3p6*(-ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*fmc2*hbcm-fmc2*hbcm)+8*p4p6*(ffmcfmb**2*hbcm3-ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3)+16*(ffmcfmb*hbcm*p1p3*p4p6-ffmcfmb*
     . hbcm*p1p4*p3p6+ffmcfmb*hbcm*p3p6*p4p5+hbcm*p4p5*p4p6))+ans3
      ans1=w18*(8*(p5p6*p3p5)*(-ffmcfmb*hbcm-fmc)+8*(p4p6*p3p5)*(-
     . ffmcfmb*hbcm-fmc-4*hbcm)+8*(p3p6*p3p5)*(-3*ffmcfmb*hbcm-fmc)+8
     . *(p5p6*p3p4)*(-ffmcfmb*hbcm-fmc+4*hbcm)+8*(p4p6*p3p4)*(-
     . ffmcfmb*hbcm-fmc)+8*(p3p6*p3p4)*(ffmcfmb*hbcm-fmc)+8*p3p6*(
     . ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmb2*hbcm-ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*fmc2*hbcm-fmb*fmc*hbcm-fmc2*hbcm)+8*p5p6*(-ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+2*fmc2*hbcm)+8*p4p6*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm-4*fmb2*
     . hbcm)+16*(ffmcfmb*hbcm*p1p3*p4p6-ffmcfmb*hbcm*p1p3*p5p6-
     . ffmcfmb*hbcm*p1p4*p3p6+ffmcfmb*hbcm*p1p5*p3p6-2*hbcm*p1p4*p5p6
     . +2*hbcm*p1p5*p4p6-hbcm*p3p6*p4p5-2*hbcm*p4p5*p4p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans4=8*(p6p7*p3p4)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*hbcm2+2*fmb*
     . hbcm+2*fmc*hbcm+fmc2)+8*p6p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2
     . *fmc*hbcm3+2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmb2*hbcm2+2*ffmcfmb*
     . fmc*hbcm3+fmb*fmc2*hbcm+2*fmb2*fmc*hbcm+fmc3*hbcm+2*p3p4s2)+16
     . *(-ffmcfmb*p1p3*p3p4*p6p7-ffmcfmb*p1p3*p3p5*p6p7-p1p4*p3p4*
     . p6p7-p1p4*p3p5*p6p7+p3p4*p3p5*p6p7-p3p4*p3p6*p4p7-2*p3p4*p4p6*
     . p4p7-p3p4*p4p6*p5p7+p3p4*p5p6*p5p7-p3p5*p3p6*p4p7-2*p3p5*p4p6*
     . p4p7-p3p5*p4p6*p5p7+p3p5*p5p6*p5p7)
      ans3=32*(p5p7*p4p6)*(-fmb*hbcm-fmc*hbcm)+16*(p6p7*p1p4)*(-fmb*
     . hbcm-fmc*hbcm)+(p4p6*p3p7*p3p5)*(-16*ffmcfmb-16)+(p4p6*p3p7*
     . p3p4)*(-16*ffmcfmb-16)+32*(p5p7*p5p6)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +16*(p5p6*p4p7)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+16*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm)+16*(p6p7*p4p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p1p5)*(ffmcfmb*hbcm2-fmc*hbcm
     . )+16*(p5p6*p3p7)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-fmc*hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm
     . -ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+32*(p3p7*p3p6)*(-ffmcfmb*
     . fmb*hbcm-ffmcfmb*fmc*hbcm)+16*(p6p7*p1p3)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm)+16*(p3p7*p3p6*p3p5)*(ffmcfmb**2-2*ffmcfmb)+
     . 16*(p3p7*p3p6*p3p4)*(ffmcfmb**2-2*ffmcfmb)+16*(p4p6*p3p7)*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmb*hbcm-
     . fmc*hbcm)+8*(p6p7*p3p5)*(-ffmcfmb**2*hbcm2+2*fmc*hbcm+fmc2)+
     . ans4
      ans2=w18*ans3
      ans7=16*(-p1p3*p1p4*p6p7+p1p3*p3p4*p6p7-p1p3*p3p6*p4p7-2*p1p3*
     . p4p6*p4p7-p1p3*p4p6*p5p7+p1p3*p5p6*p5p7)
      ans6=(p4p6*p3p7*p1p3)*(-16*ffmcfmb-16)+16*(p5p7*p5p6)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+16*(p5p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+32*(p4p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7
     . *p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p1p4)*(-ffmcfmb
     . *hbcm2-fmb*hbcm+hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm)+16*(p4p7*p3p6)*(-ffmcfmb*
     . fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+16*(
     . p3p7*p3p6*p1p3)*(ffmcfmb**2-2*ffmcfmb)+16*(p4p6*p3p7)*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-fmb*hbcm+hbcm2)+16*(p3p7*
     . p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p1p3)*(-ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+2*fmc*hbcm
     . +fmc2)+8*p6p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+
     . ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2+2*
     . ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4-2*ffmcfmb*
     . p1p3s2+fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc*hbcm3-fmc2*hbcm2)+ans7
      ans5=w13*ans6
      ans8=((p4p6*p3p7)*(32*fb2*ffmcfmb+32*fb2+8)+(p6p7*p1p3)*(32*fb2
     . *ffmcfmb+8)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+64*fb2*ffmcfmb-16*
     . ffmcfmb+16)+8*p6p7*(2*fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2
     . -2*fb2*fmc2+2*fmb*hbcm+2*fmc*hbcm-hbcm2)+32*(fb2*p1p4*p6p7-fb2
     . *p3p4*p6p7+fb2*p3p6*p4p7+2*fb2*p4p6*p4p7+fb2*p4p6*p5p7-fb2*
     . p5p6*p5p7))
      ans1=w1*((p4p6*p3p7*p3p4)*(-16*ffmcfmb-16)+16*(p5p7*p4p6)*(-
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p4p6)*(-ffmcfmb*hbcm2-2*fmc*
     . hbcm)+16*(p4p7*p3p6)*(-ffmcfmb*fmc*hbcm-fmc*hbcm)+16*(p3p7*
     . p3p6*p3p4)*(ffmcfmb**2-2*ffmcfmb)+16*(p4p6*p3p7)*(-ffmcfmb**2*
     . hbcm2-ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(p6p7*p3p4)*(-ffmcfmb**2*
     . hbcm2+2*ffmcfmb*hbcm2+2*fmc*hbcm+fmc2)+8*p6p7*(-ffmcfmb**2*fmc
     . *hbcm3+2*ffmcfmb*fmc*hbcm3+fmc3*hbcm+2*p3p4s2)+16*(-ffmcfmb*
     . fmc*hbcm*p1p3*p6p7-2*ffmcfmb*fmc*hbcm*p3p6*p3p7-ffmcfmb*fmc*
     . hbcm*p3p6*p5p7-ffmcfmb*p1p3*p3p4*p6p7-fmc*hbcm*p1p4*p6p7+fmc*
     . hbcm*p5p6*p5p7-p1p4*p3p4*p6p7-p3p4*p3p6*p4p7-2*p3p4*p4p6*p4p7-
     . p3p4*p4p6*p5p7+p3p4*p5p6*p5p7))+ans2+ans5+ans8
      ans=ccc*ans1
      b(4)=ans
      ans4=8*(p3p7*p3p4)*(2*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-fmc*hbcm-fmc2)+8*p3p7*(
     . ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-3*ffmcfmb*fmb*hbcm3+
     . 3*ffmcfmb*fmb2*hbcm2-3*ffmcfmb*fmc*hbcm3-ffmcfmb*fmc2*hbcm2-2*
     . ffmcfmb*p3p5s2+fmb*fmc*hbcm2-fmb*fmc2*hbcm-2*fmb2*fmc*hbcm-
     . fmc2*hbcm2-fmc3*hbcm)+16*(ffmcfmb*p1p3*p3p4*p3p7+ffmcfmb*p1p3*
     . p3p5*p3p7+ffmcfmb*p2p3*p3p4*p3p7+ffmcfmb*p2p3*p3p5*p3p7-
     . ffmcfmb*p3p4*p3p5*p3p7+hbcm2*p1p4*p5p7-hbcm2*p1p5*p4p7-hbcm2*
     . p4p5*p5p7+p1p3*p3p4*p4p7-p1p3*p3p4*p5p7+p1p3*p3p5*p4p7-p1p3*
     . p3p5*p5p7+p1p4*p3p4*p3p7+p1p4*p3p5*p3p7+2*p2p3*p3p4*p4p7-p2p3*
     . p3p4*p5p7+2*p2p3*p3p5*p4p7-p2p3*p3p5*p5p7-2*p3p4*p3p5*p4p7+2*
     . p3p4*p3p5*p5p7)
      ans3=8*(p5p7*p3p5)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(p4p7*p3p4)*(
     . -fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(p4p7*p1p3)*(2*fmb*hbcm+fmc*
     . hbcm)+16*(p4p5*p3p7)*(ffmcfmb*hbcm2-fmc*hbcm-hbcm2)+8*(p4p7*
     . p3p5)*(-4*ffmcfmb*hbcm2-5*fmb*hbcm-fmc*hbcm)+8*(p5p7*p3p4)*(4*
     . ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm)+32*(p5p7*p2p3)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2+3*fmb*hbcm+2*fmc
     . *hbcm)+16*(p3p7*p1p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p1p4)
     . *(ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm)+16*(p5p7*p1p3)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*p5p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+2*
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2+fmb2*hbcm2-2*fmc*hbcm3-2*fmc2*
     . hbcm2+4*p3p4s2)+8*(p3p7*p3p5)*(-3*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm-fmc2)+32*(p3p7*p2p3)*(ffmcfmb*
     . fmb*hbcm+ffmcfmb*fmc*hbcm)+32*(p3p7*p1p3)*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm)+8*p4p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-4*fmb*hbcm3+3*fmb2*hbcm2-2*fmc
     . *hbcm3-fmc2*hbcm2-4*p3p5s2)+ans4
      ans2=w18*ans3
      ans7=8*p3p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3-ffmcfmb
     . **2*hbcm4-2*ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-ffmcfmb*fmc*
     . hbcm3-ffmcfmb*fmc2*hbcm2+ffmcfmb*hbcm4+2*ffmcfmb*p1p3s2+fmb*
     . fmc*hbcm2-fmb*fmc2*hbcm-fmb2*fmc*hbcm+fmc2*hbcm2)+16*(ffmcfmb*
     . p1p3*p2p3*p3p7-ffmcfmb*p1p3*p3p5*p3p7+p1p3*p1p4*p3p7+2*p1p3*
     . p2p3*p4p7-p1p3*p2p3*p5p7+2*p1p3*p3p4*p5p7-2*p1p3*p3p5*p4p7)
      ans6=8*(p5p7*p1p3)*(-fmb*hbcm+fmc*hbcm+2*hbcm2)+32*(p4p7*p3p5)*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+32*(p5p7*p3p4)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+16*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +32*(p4p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p3p7*p1p4)*
     . (ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p7*p1p3)*(2*ffmcfmb*hbcm2+
     . 3*fmb*hbcm+fmc*hbcm-4*hbcm2)+8*p4p7*(ffmcfmb*fmb*hbcm3+ffmcfmb
     . *fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2-3*fmb*hbcm3+fmb2*
     . hbcm2-fmc*hbcm3+2*hbcm4+2*p1p3s2)+8*p5p7*(ffmcfmb**2*hbcm4+
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmb*hbcm3-
     . fmc*hbcm3-hbcm4-2*p1p3s2)+16*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+16*(p3p7*p2p3)*(ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+8*(p3p7*p1p3)*(2*ffmcfmb
     . **2*hbcm2+3*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-3*ffmcfmb*
     . hbcm2-fmc*hbcm-fmc2)+ans7
      ans5=w13*ans6
      ans8=((p3p7*p3p5)*(32*fb2*ffmcfmb+8)+(p3p7*p2p3)*(-32*fb2*
     . ffmcfmb-8)+(p3p7*p1p3)*(-32*fb2*ffmcfmb-8)+16*p5p7*(fb1*fmb*
     . hbcm+fb1*fmc*hbcm-2*fb2*hbcm2-hbcm2)+16*p4p7*(fb1*fmb*hbcm+fb1
     . *fmc*hbcm+2*fb2*hbcm2)+4*p3p7*(4*fb1*ffmcfmb*fmb*hbcm+4*fb1*
     . fmc*hbcm+4*fb2*ffmcfmb*hbcm2+4*fb2*fmc2-3*fmb*hbcm-4*fmc*hbcm+
     . hbcm2)+32*(-fb2*p1p3*p4p7+fb2*p1p3*p5p7-fb2*p1p4*p3p7-2*fb2*
     . p2p3*p4p7+fb2*p2p3*p5p7-2*fb2*p3p4*p5p7+2*fb2*p3p5*p4p7))
      ans1=w1*(8*(p4p7*p3p4)*(-fmb*hbcm-fmc*hbcm-2*hbcm2)+16*(p4p7*
     . p3p5)*(-ffmcfmb*hbcm2-2*fmc*hbcm)+8*(p5p7*p3p4)*(2*ffmcfmb*
     . hbcm2-fmb*hbcm+3*fmc*hbcm+2*hbcm2)+16*(p4p7*p2p3)*(ffmcfmb*
     . hbcm2+2*fmc*hbcm)+16*(p3p7*p1p4)*(ffmcfmb*hbcm2+fmc*hbcm)+8*
     . p5p7*(ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+2*fmc*hbcm3+4*p3p4s2)+8*
     . p4p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2-2*fmc*
     . hbcm3-fmc2*hbcm2)+8*(p3p7*p3p4)*(2*ffmcfmb**2*hbcm2-ffmcfmb*
     . fmb*hbcm-2*ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-fmc*hbcm-fmc2)+8*
     . p3p7*(ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmb*fmc*hbcm2-3*ffmcfmb*fmc
     . *hbcm3-ffmcfmb*fmc2*hbcm2-fmc2*hbcm2-fmc3*hbcm)+16*(2*ffmcfmb*
     . fmc*hbcm*p1p3*p3p7+2*ffmcfmb*fmc*hbcm*p2p3*p3p7-2*ffmcfmb*fmc*
     . hbcm*p3p5*p3p7+ffmcfmb*p1p3*p3p4*p3p7+ffmcfmb*p2p3*p3p4*p3p7-
     . ffmcfmb*p3p4*p3p5*p3p7+fmc*hbcm*p1p3*p4p7-fmc*hbcm*p1p3*p5p7-
     . fmc*hbcm*p2p3*p5p7+hbcm2*p4p5*p5p7+p1p3*p3p4*p4p7-p1p3*p3p4*
     . p5p7+p1p4*p3p4*p3p7+2*p2p3*p3p4*p4p7-p2p3*p3p4*p5p7-2*p3p4*
     . p3p5*p4p7))+ans2+ans5+ans8
      ans=ccc*ans1
      b(5)=ans
      ans4=8*p4p6*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3+fmb*fmc*hbcm2+4*fmb*hbcm3-2*fmb2*hbcm2+2*fmc*hbcm3+4*
     . p3p4s2+2*p3p5s2)+8*p3p6*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*
     . hbcm3+ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*hbcm3
     . +fmb*fmc*hbcm2-3*fmb*fmc2*hbcm-2*fmb2*fmc*hbcm-fmc2*hbcm2-fmc3
     . *hbcm+2*p3p5s2)+16*(-ffmcfmb*p2p3*p3p4*p3p6-ffmcfmb*p2p3*p3p5*
     . p3p6-hbcm2*p1p4*p5p6+hbcm2*p1p5*p4p6-hbcm2*p4p5*p4p6-p1p3*p3p4
     . *p4p6+p1p3*p3p4*p5p6-p1p3*p3p5*p4p6+p1p3*p3p5*p5p6+p1p4*p3p4*
     . p3p6+p1p4*p3p5*p3p6-p1p5*p3p4*p3p6-p1p5*p3p5*p3p6-2*p2p3*p3p4*
     . p4p6+p2p3*p3p4*p5p6-2*p2p3*p3p5*p4p6+p2p3*p3p5*p5p6+p3p4*p3p5*
     . p3p6+3*p3p4*p3p5*p4p6-p3p4*p3p5*p5p6)
      ans3=16*(p3p6*p1p4)*(2*fmb*hbcm+fmc*hbcm)+16*(p4p6*p1p3)*(-2*
     . fmb*hbcm-fmc*hbcm)+16*(p4p5*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-fmc*
     . hbcm-hbcm2)+8*(p5p6*p3p5)*(-3*ffmcfmb*hbcm2+3*fmc*hbcm-2*hbcm2
     . )+8*(p4p6*p3p5)*(ffmcfmb*hbcm2+4*fmb*hbcm+3*fmc*hbcm)+8*(p5p6*
     . p3p4)*(-ffmcfmb*hbcm2+2*fmb*hbcm+3*fmc*hbcm)+8*(p4p6*p3p4)*(3*
     . ffmcfmb*hbcm2+6*fmb*hbcm+3*fmc*hbcm+2*hbcm2)+32*(p5p6*p2p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-3*fmb*
     . hbcm-2*fmc*hbcm)+16*(p3p6*p1p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p5p6*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*p5p6*(ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+
     . fmc2*hbcm2-2*p3p5s2)+8*(p3p6*p3p5)*(2*ffmcfmb*fmb*hbcm+2*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb2-fmc*hbcm-2*fmc2)+8*(p3p6
     . *p3p4)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-2*
     . fmb*hbcm+2*fmb2-fmc*hbcm-2*fmc2)+32*(p3p6*p2p3)*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm)+ans4
      ans2=w18*ans3
      ans7=8*p3p6*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3+ffmcfmb*
     . fmb*hbcm3+ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*hbcm3-ffmcfmb*fmc2*
     . hbcm2+fmb*fmc*hbcm2-2*fmb*fmc2*hbcm-fmb*hbcm3-fmb2*fmc*hbcm+
     . fmb3*hbcm-2*fmc*hbcm3+fmc2*hbcm2)+16*(-ffmcfmb*p1p3*p2p3*p3p6-
     . fmb*hbcm*p1p5*p3p6+p1p3*p1p4*p3p6-p1p3*p1p5*p3p6-2*p1p3*p2p3*
     . p4p6+p1p3*p2p3*p5p6+2*p1p3*p3p4*p4p6+p1p3*p3p5*p3p6+p1p3*p3p5*
     . p4p6-p1p3*p3p5*p5p6)
      ans6=16*(p5p6*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p6*
     . p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+32*(p4p6*p3p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+16*(p5p6*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+32*(p4p6*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p3p6
     . *p1p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p5p6*p1p3)*(ffmcfmb*
     . hbcm2+2*fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p4p6*p1p3)*(-ffmcfmb*
     . hbcm2-2*fmb*hbcm+fmc*hbcm+4*hbcm2)+8*(p3p6*p1p3)*(ffmcfmb*
     . hbcm2+2*fmb2+fmc*hbcm-2*fmc2)+8*p5p6*(ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmb*hbcm3+fmb2
     . *hbcm2-fmc*hbcm3+hbcm4+2*p1p3s2)+16*(p3p6*p3p5)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-fmc*hbcm)+16*(p3p6*p3p4)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+16*(p3p6*p2p3)*(-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+fmb*hbcm+2*fmc*hbcm)+8*
     . p4p6*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmb*hbcm3-fmc*hbcm3-2*hbcm4-2*
     . p1p3s2)+ans7
      ans5=w13*ans6
      ans8=((p3p6*p2p3)*(32*fb2*ffmcfmb+8)+4*p5p6*(4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2+8*fb2*hbcm2+hbcm2)+4*p4p6*(4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2-8*fb2*hbcm2-3*hbcm2)+4*p3p6*(4*fb1*fmc*hbcm-
     . 4*fb2*ffmcfmb*hbcm2-8*fb2*fmb2+8*fb2*fmc2+2*ffmcfmb*hbcm2-4*
     . fmb*hbcm-4*fmc*hbcm-hbcm2)+32*(fb2*p1p3*p4p6-fb2*p1p3*p5p6-fb2
     . *p1p4*p3p6+fb2*p1p5*p3p6+2*fb2*p2p3*p4p6-fb2*p2p3*p5p6-2*fb2*
     . p3p4*p4p6-fb2*p3p5*p3p6-fb2*p3p5*p4p6+fb2*p3p5*p5p6))
      ans1=w1*(16*(p4p6*p3p5)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p6*p3p4)*
     . (ffmcfmb*hbcm2-fmc*hbcm-2*hbcm2)+8*(p4p6*p3p4)*(3*ffmcfmb*
     . hbcm2+3*fmc*hbcm+2*hbcm2)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-2*fmc
     . *hbcm)+8*p5p6*(ffmcfmb*fmc*hbcm3-2*fmc*hbcm3-fmc2*hbcm2)+16*(
     . p3p6*p3p5)*(ffmcfmb*fmc*hbcm+fmc*hbcm)+8*(p3p6*p3p4)*(2*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2*fmb2-fmc*hbcm-2*fmc2)+8*p4p6*
     . (ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3+2*fmc*hbcm3+4*p3p4s2)+8*
     . p3p6*(ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmc*hbcm3+2*fmb2*fmc*hbcm-
     . fmc2*hbcm2-fmc3*hbcm)+16*(-2*ffmcfmb*fmc*hbcm*p2p3*p3p6-
     . ffmcfmb*p2p3*p3p4*p3p6-fmc*hbcm*p1p3*p4p6+fmc*hbcm*p1p3*p5p6+
     . fmc*hbcm*p1p4*p3p6-fmc*hbcm*p1p5*p3p6+fmc*hbcm*p2p3*p5p6-fmc*
     . hbcm*p3p5*p5p6+fmc*hbcm*p3p6*p4p5+hbcm2*p4p5*p4p6-p1p3*p3p4*
     . p4p6+p1p3*p3p4*p5p6+p1p4*p3p4*p3p6-p1p5*p3p4*p3p6-2*p2p3*p3p4*
     . p4p6+p2p3*p3p4*p5p6+p3p4*p3p5*p3p6+p3p4*p3p5*p4p6-p3p4*p3p5*
     . p5p6))+ans2+ans5+ans8
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(8*p3p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p3p4*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p1p5*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+16*p1p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p1p3*(-
     . ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm+2*
     . ffmcfmb*hbcm2+2*fmb*hbcm-2*fmb2+4*fmc*hbcm+3*fmc2)+4*(-2*
     . ffmcfmb**2*fmb*hbcm3-2*ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4+4
     . *ffmcfmb*fmb*hbcm3-3*ffmcfmb*fmb2*hbcm2+4*ffmcfmb*fmc*hbcm3+3*
     . ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4-2*ffmcfmb*p1p3s2+3*fmb*fmc2*
     . hbcm-fmb*hbcm3+2*fmb2*fmc*hbcm+2*fmb2*hbcm2-fmb3*hbcm-2*fmc*
     . hbcm3-3*fmc2*hbcm2-4*p1p3*p1p4+2*p1p3*p1p5+4*p1p3*p3p4-2*p1p3*
     . p3p5))+(p1p3*(16*fb2*ffmcfmb+4)+4*(2*fb2*ffmcfmb**2*hbcm2-4*
     . fb2*ffmcfmb*hbcm2+4*fb2*fmb2-6*fb2*fmc2+8*fb2*p1p4-4*fb2*p1p5-
     . 8*fb2*p3p4+4*fb2*p3p5-ffmcfmb*hbcm2+3*fmb*hbcm+4*fmc*hbcm-
     . hbcm2))
      ans1=w1*(8*p4p5*(ffmcfmb*hbcm2-fmc*hbcm)+8*p1p4*(-ffmcfmb*hbcm2
     . -2*fmc*hbcm)+4*p3p4*(-ffmcfmb**2*hbcm2+4*ffmcfmb*hbcm2-2*fmb2+
     . 4*fmc*hbcm+3*fmc2)+8*(-ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*fmc*hbcm
     . *p1p3+2*ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2-ffmcfmb*p1p3*p3p4
     . -fmb2*fmc*hbcm+fmc*hbcm*p1p5-fmc*hbcm*p3p5+fmc3*hbcm-2*p1p4*
     . p3p4+p1p5*p3p4-p3p4*p3p5+2*p3p4s2))+w18*(8*p4p5*(-ffmcfmb*
     . hbcm2+fmb*hbcm+2*fmc*hbcm)+16*p1p5*(ffmcfmb*hbcm2-fmc*hbcm)+8*
     . p1p4*(-ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm)+16*p1p3*(-ffmcfmb*
     . fmb*hbcm-ffmcfmb*fmc*hbcm)+4*p3p5*(-ffmcfmb**2*hbcm2-2*ffmcfmb
     . *hbcm2-2*fmb2+4*fmc*hbcm+3*fmc2)+4*p3p4*(-ffmcfmb**2*hbcm2+4*
     . ffmcfmb*hbcm2+6*fmb*hbcm-2*fmb2+4*fmc*hbcm+3*fmc2)+8*(-ffmcfmb
     . **2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+2*ffmcfmb*fmb*hbcm3-2*
     . ffmcfmb*fmb2*hbcm2+2*ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2-
     . ffmcfmb*p1p3*p3p4-ffmcfmb*p1p3*p3p5+2*fmb*fmc2*hbcm+2*fmb2*fmc
     . *hbcm+fmc3*hbcm-2*p1p4*p3p4-2*p1p4*p3p5+p1p5*p3p4+p1p5*p3p5+
     . p3p4*p3p5+2*p3p4s2-p3p5s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4+2*ffmcfmb*fmb
     . *hbcm4+2*ffmcfmb*fmb2*hbcm3-2*ffmcfmb*fmc*hbcm4-2*ffmcfmb*fmc2
     . *hbcm3-2*ffmcfmb*hbcm*p3p4s2-2*ffmcfmb*hbcm*p3p5s2-4*fmb*fmc*
     . hbcm3-3*fmb*fmc2*hbcm2+2*fmb*p3p4s2+2*fmb*p3p5s2+2*fmb2*fmc*
     . hbcm2+2*fmb2*hbcm3-4*fmc*p3p4s2-4*fmc*p3p5s2-2*fmc2*hbcm3+fmc3
     . *hbcm2+4*hbcm*p1p3*p4p5-2*hbcm*p1p4*p3p4+6*hbcm*p1p4*p3p5-6*
     . hbcm*p1p5*p3p4+2*hbcm*p1p5*p3p5+8*hbcm*p2p3*p4p5+2*hbcm*p3p4s2
     . -8*hbcm*p3p5*p4p5-2*hbcm*p3p5s2)
      ans3=8*p4p5*(-fmb*hbcm2+fmc*hbcm2-2*hbcm3)+8*p1p5*(-ffmcfmb*
     . hbcm3-fmc*hbcm2)+8*p1p4*(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2)+
     . 16*(p3p5*p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc)+8*(p3p5*p2p3)*(2*
     . ffmcfmb*hbcm-fmb+3*fmc)+8*(p3p4*p2p3)*(2*ffmcfmb*hbcm-fmb+3*
     . fmc)+8*(p3p5*p1p3)*(3*ffmcfmb*hbcm-fmb+2*fmc)+8*(p3p4*p1p3)*(-
     . ffmcfmb*hbcm-fmb+2*fmc)+8*p2p3*(-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb
     . *fmc*hbcm2+3*fmb*fmc*hbcm-fmb2*hbcm+2*fmc2*hbcm)+8*p1p3*(-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-fmb2*hbcm+
     . fmc2*hbcm)+8*p3p5*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmb*hbcm2-fmc
     . *hbcm2-3*fmc2*hbcm)+8*p3p4*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*
     . hbcm2-2*ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm-fmb*hbcm2+4*fmb2*hbcm
     . -fmc*hbcm2-fmc2*hbcm)+ans4
      ans2=w18*ans3
      ans7=4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4+2*ffmcfmb**2*
     . hbcm5+2*ffmcfmb*fmb*hbcm4+2*ffmcfmb*fmb2*hbcm3-2*ffmcfmb*fmc*
     . hbcm4-2*ffmcfmb*fmc2*hbcm3-2*ffmcfmb*hbcm*p1p3s2-2*ffmcfmb*
     . hbcm5-4*fmb*fmc*hbcm3-2*fmb*fmc2*hbcm2-fmb*hbcm4-2*fmb*p1p3s2+
     . fmb2*fmc*hbcm2+fmb3*hbcm2+3*fmc*hbcm4+4*fmc*p1p3s2+2*fmc2*
     . hbcm3+2*hbcm*p1p3*p1p4-2*hbcm*p1p3*p1p5)
      ans6=8*p1p5*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*p1p4*(ffmcfmb*
     . hbcm3+fmb*hbcm2-hbcm3)+8*(p3p5*p1p3)*(3*ffmcfmb*hbcm+fmb-2*fmc
     . -hbcm)+8*(p3p4*p1p3)*(-ffmcfmb*hbcm+fmb-2*fmc+hbcm)+8*(p2p3*
     . p1p3)*(-2*ffmcfmb*hbcm-fmb+3*fmc)+16*p3p4*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+
     . 16*p3p5*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+8*p2p3*(-2*ffmcfmb**2*
     . hbcm3-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+
     . 3*fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-3*fmc*hbcm2)+8*p1p3*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*ffmcfmb
     . *hbcm3+2*fmb*fmc*hbcm+fmb*hbcm2-3*fmc*hbcm2-fmc2*hbcm)+ans7
      ans5=w13*ans6
      ans8=(4*p3p5*(4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb+8*fb2*fmc
     . -7*hbcm)+4*p3p4*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*fmb+8*fb2
     . *fmc+hbcm)+8*p2p3*(-4*fb1*ffmcfmb*hbcm+2*fb2*fmb-6*fb2*fmc+3*
     . hbcm)+4*p1p3*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmb-8*fb2*fmc+3*hbcm)+
     . 4*(4*fb1*ffmcfmb*hbcm3+4*fb1*fmb2*hbcm-4*fb1*fmc2*hbcm+4*fb1*
     . hbcm*p1p4-4*fb1*hbcm*p1p5-4*fb2*fmb*hbcm2+8*fb2*fmc*hbcm2-2*
     . fmb*hbcm2+2*fmc*hbcm2-3*hbcm3))
      ans1=w1*(8*p4p5*(-ffmcfmb*hbcm3-fmc*hbcm2+2*hbcm3)+8*p1p4*(
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p5*p3p4)*(-ffmcfmb*hbcm+fmb-2*
     . fmc-hbcm)+8*(p3p4*p2p3)*(2*ffmcfmb*hbcm-fmb+3*fmc)+8*(p3p4*
     . p1p3)*(-ffmcfmb*hbcm-fmb+2*fmc)+8*p3p5*(-2*ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+8*p2p3*(3*ffmcfmb*fmc*hbcm2-
     . fmb*fmc*hbcm+2*fmc2*hbcm)+8*p1p3*(ffmcfmb*fmc*hbcm2-fmb*fmc*
     . hbcm+fmc2*hbcm)+8*p3p4*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2-fmc2*hbcm)+4*(-
     . ffmcfmb**2*fmc*hbcm4-2*ffmcfmb*fmc*hbcm4-2*ffmcfmb*fmc2*hbcm3-
     . 2*ffmcfmb*hbcm*p3p4s2+2*fmb*fmc*hbcm3+2*fmb*p3p4s2-2*fmb2*fmc*
     . hbcm2+2*fmc*hbcm2*p1p5-4*fmc*p3p4s2-2*fmc2*hbcm3+fmc3*hbcm2-4*
     . hbcm*p1p3*p4p5-2*hbcm*p1p4*p3p4+2*hbcm*p1p5*p3p4-4*hbcm*p2p3*
     . p4p5+2*hbcm*p3p4s2+4*hbcm*p3p5*p4p5))+ans2+ans5+ans8
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+2*fmc)+8*p3p6*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm
     . +fmc2*hbcm)+8*(fmc*hbcm2*p5p6-hbcm*p3p4*p4p6+hbcm*p3p4*p5p6-2*
     . hbcm*p3p6*p4p5))+w18*(8*p5p6*(-ffmcfmb*hbcm3-fmc*hbcm2)+8*p4p6
     . *(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2)+8*(p3p6*p3p5)*(3*
     . ffmcfmb*hbcm-fmb+2*fmc)+8*(p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+2*fmc
     . )+8*p3p6*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-
     . fmb2*hbcm+fmc2*hbcm)+8*(-hbcm*p3p4*p4p6-3*hbcm*p3p4*p5p6+3*
     . hbcm*p3p5*p4p6+hbcm*p3p5*p5p6+2*hbcm*p3p6*p4p5))+w13*(8*p5p6*(
     . -ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*p4p6*(ffmcfmb*hbcm3+fmb*
     . hbcm2-hbcm3)+16*(p3p6*p3p5)*(ffmcfmb*hbcm-hbcm)+16*(p3p6*p3p4)
     . *(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p3)*(-ffmcfmb*hbcm-fmb+2*fmc)+
     . 8*p3p6*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb2*hbcm-fmc*hbcm2)+8*(hbcm*p1p3
     . *p4p6-hbcm*p1p3*p5p6))+(4*p3p6*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmb-
     . 8*fb2*fmc+3*hbcm)+16*(fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(9)=ans
      b(10)=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm2+2*fmc*hbcm)+8*(2*ffmcfmb*
     . fmc*hbcm*p3p6+ffmcfmb*p3p4*p3p6-fmc*hbcm*p5p6+2*p3p4*p4p6-p3p4
     . *p5p6))+w18*(16*p5p6*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p4p6*(ffmcfmb
     . *hbcm2+3*fmb*hbcm+2*fmc*hbcm)+16*p3p6*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm)+8*(ffmcfmb*p3p4*p3p6+ffmcfmb*p3p5*p3p6+2*
     . p3p4*p4p6-p3p4*p5p6+2*p3p5*p4p6-p3p5*p5p6))+w13*(8*p5p6*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*p3p6*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*hbcm-
     . 2*fmc*hbcm)+8*(ffmcfmb*p1p3*p3p6+2*p1p3*p4p6-p1p3*p5p6))+(p3p6
     . *(-16*fb2*ffmcfmb-4)+16*(-2*fb2*p4p6+fb2*p5p6)))
      ans=ccc*(w1*(4*p3p4*(-3*ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+4*(-
     . 2*ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+2*fmc*hbcm*p3p5+fmc2*hbcm2-2
     . *hbcm2*p4p5+2*p3p4*p3p5-4*p3p4s2))+w18*(4*p3p5*(3*ffmcfmb*
     . hbcm2+fmb*hbcm-2*fmc*hbcm)+4*p3p4*(-3*ffmcfmb*hbcm2-5*fmb*hbcm
     . -2*fmc*hbcm)+4*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3-2*fmb
     . *fmc*hbcm2-fmb2*hbcm2+fmc2*hbcm2+2*hbcm2*p4p5-2*p3p4*p3p5-4*
     . p3p4s2+2*p3p5s2))+w13*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+
     . 16*p3p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p1p3*(-ffmcfmb*hbcm2
     . -fmb*hbcm-2*fmc*hbcm)+4*(-ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3
     . -2*ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-2*fmb*fmc*hbcm2+fmb*hbcm3-
     . fmb2*hbcm2+2*fmc*hbcm3-4*p1p3*p3p4+2*p1p3*p3p5))+2*(-4*fb1*fmb
     . *hbcm-8*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2+16*fb2*p3p4-8*fb2*
     . p3p5+3*hbcm2))
      b(11)=ans
      ans=ccc*(w1*(8*p4p7*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*p3p4)*(-3
     . *ffmcfmb*hbcm-fmc)+8*p3p7*(-2*ffmcfmb*fmc*hbcm2-fmc2*hbcm)+8*(
     . fmc*hbcm2*p5p7-hbcm*p3p4*p4p7+hbcm*p3p4*p5p7))+w18*(8*p5p7*(-
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*p4p7*(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc
     . *hbcm2)+8*(p3p7*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p3p7*p3p4)*(-3*
     . ffmcfmb*hbcm-fmc)+8*p3p7*(2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm-fmc2*hbcm)+8*(-hbcm*p3p4*p4p7-3*hbcm*p3p4*
     . p5p7+3*hbcm*p3p5*p4p7+hbcm*p3p5*p5p7-2*hbcm*p3p7*p4p5))+w13*(8
     . *p5p7*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*p4p7*(ffmcfmb*hbcm3+
     . fmb*hbcm2-hbcm3)+8*(p3p7*p1p3)*(ffmcfmb*hbcm-fmc)+8*p3p7*(
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+8*(hbcm*p1p3*p4p7-hbcm*p1p3*p5p7
     . ))+(4*p3p7*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm)+16*(fb1*hbcm*
     . p4p7-fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w1*(4*p3p4*(-2*ffmcfmb*hbcm+fmb-3*fmc)+4*(-3*ffmcfmb
     . *fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm+2*hbcm*p4p5))+w18*(4*p3p5*
     . (-2*ffmcfmb*hbcm+fmb-3*fmc)+4*p3p4*(-2*ffmcfmb*hbcm+fmb-3*fmc)
     . +4*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-3*fmb*fmc*hbcm+
     . fmb2*hbcm-2*fmc2*hbcm-4*hbcm*p4p5))+w13*(4*p1p3*(2*ffmcfmb*
     . hbcm+fmb-3*fmc)+4*(2*ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2-3*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+3*fmc*hbcm2))+4*(4*fb1*ffmcfmb*hbcm-2*fb2*fmb+6*fb2*
     . fmc-3*hbcm))
      b(14)=ccc*(w1*(8*p4p7*(-ffmcfmb*hbcm2-2*fmc*hbcm)+8*(-2*ffmcfmb
     . *fmc*hbcm*p3p7-ffmcfmb*p3p4*p3p7+fmc*hbcm*p5p7-2*p3p4*p4p7+
     . p3p4*p5p7))+w18*(16*p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*p4p7*(-
     . ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm)+16*p3p7*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm)+8*(-ffmcfmb*p3p4*p3p7-ffmcfmb*p3p5*p3p7
     . -2*p3p4*p4p7+p3p4*p5p7-2*p3p5*p4p7+p3p5*p5p7))+w13*(8*p5p7*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*p4p7*(-ffmcfmb*hbcm2-fmb*hbcm
     . +hbcm2)+8*p3p7*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2)+8*(-ffmcfmb*p1p3*p3p7-2*p1p3*p4p7+p1p3*p5p7))+(p3p7*(16
     . *fb2*ffmcfmb+4)+16*(2*fb2*p4p7-fb2*p5p7)))
      ans2=w13*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(
     . p4p7*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p3p4)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p1p3)*(-ffmcfmb*hbcm2-
     . fmc*hbcm)+8*p6p7*(-ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3-ffmcfmb*
     . fmc*hbcm3+ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p7*p3p6
     . )*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmc*hbcm)+16*(ffmcfmb*p1p3*p3p6*p3p7-fmb*hbcm*p3p6*p5p7-
     . p1p3*p3p4*p6p7+p1p3*p3p6*p4p7-p1p3*p3p6*p5p7+p1p3*p3p7*p4p6))+
     . ((p3p7*p3p6)*(-32*fb2*ffmcfmb-8)+4*p6p7*(-4*fb1*fmc*hbcm+4*fb2
     . *ffmcfmb*hbcm2+hbcm2)+32*(fb2*p3p4*p6p7-fb2*p3p6*p4p7+fb2*p3p6
     . *p5p7-fb2*p3p7*p4p6))
      ans1=w1*(16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p4)*
     . (-ffmcfmb*hbcm2-fmc*hbcm)+8*p6p7*(-ffmcfmb*fmc*hbcm3+fmc2*
     . hbcm2-2*p3p4s2)+16*(2*ffmcfmb*fmc*hbcm*p3p6*p3p7+ffmcfmb*p3p4*
     . p3p6*p3p7+fmc*hbcm*p3p6*p4p7-fmc*hbcm*p3p6*p5p7+p3p4*p3p6*p4p7
     . -p3p4*p3p6*p5p7+p3p4*p3p7*p4p6))+w18*(16*(p4p7*p3p6)*(2*fmb*
     . hbcm+fmc*hbcm)+16*(p5p6*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm)+16*(p5p7*p3p6)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +8*(p6p7*p3p4)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+8*p6p7*(-
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmc2*hbcm2-2
     . *p3p4s2)+32*(p3p7*p3p6)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+16
     . *(ffmcfmb*p3p4*p3p6*p3p7+ffmcfmb*p3p5*p3p6*p3p7+hbcm2*p4p5*
     . p6p7+hbcm2*p4p6*p5p7-hbcm2*p4p7*p5p6-p3p4*p3p5*p6p7+p3p4*p3p6*
     . p4p7-p3p4*p3p6*p5p7+p3p4*p3p7*p4p6+p3p5*p3p6*p4p7-p3p5*p3p6*
     . p5p7+p3p5*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w1*(8*(p6p7*p3p4)*(ffmcfmb*hbcm+fmc)+8*p6p7*(ffmcfmb
     . *fmc*hbcm2+fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p4p7+ffmcfmb*hbcm
     . *p3p7*p4p6))+w18*(8*(p6p7*p3p5)*(ffmcfmb*hbcm+fmc)+8*(p6p7*
     . p3p4)*(ffmcfmb*hbcm+fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2+fmb*fmc*hbcm+fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p4p7+
     . ffmcfmb*hbcm*p3p6*p5p7+ffmcfmb*hbcm*p3p7*p4p6-ffmcfmb*hbcm*
     . p3p7*p5p6+hbcm*p4p5*p6p7+2*hbcm*p4p6*p5p7-2*hbcm*p4p7*p5p6))+
     . w13*(16*(p5p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p4p7*p3p6)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p6p7*p1p3)*(-ffmcfmb*hbcm+fmc)+8*p6p7*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2))+4*p6p7*(-4*fb1*ffmcfmb*hbcm-4*
     . fb2*fmc+3*hbcm))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp2_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(14) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,14 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p4*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      b(1)=ccc*(w5*(16*(p5p7*p3p6*p3p4)*(fmb-fmc)+16*(p5p7*p3p6*p1p3)
     . *(-fmb+fmc)-16*fmc*hbcm2*p3p6*p5p7)+32*(p5p7*p3p6)*(2*fb1*hbcm
     . -fb2*fmb+fb2*fmc))
      b(2)=ccc*(w5*(8*(p5p7*p3p4)*(-fmb+fmc)+8*(p5p7*p1p3)*(fmb-fmc)+
     . 8*fmc*hbcm2*p5p7)+16*p5p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(5)=ccc*(w5*(8*p5p7*(fmb*fmc*hbcm2+fmc2*hbcm2)+8*(-hbcm2*p1p3*
     . p5p7-2*hbcm2*p1p4*p5p7+hbcm2*p3p4*p5p7+2*p1p3*p2p3*p5p7-2*p1p3
     . *p3p5*p5p7-2*p2p3*p3p4*p5p7+2*p3p4*p3p5*p5p7))+(16*p5p7*(-2*
     . fb1*fmb*hbcm-2*fb1*fmc*hbcm-fb2*hbcm2)+32*(fb2*p2p3*p5p7-fb2*
     . p3p5*p5p7)))
      b(7)=ccc*(8*w5*(-p1p3*p2p5+p2p5*p3p4)-16*fb2*p2p5)
      b(8)=8*ccc*p2p5*(-4*fb1*hbcm+fmc*hbcm2*w5)
      b(11)=ccc*(4*w5*(-fmb*fmc*hbcm2-fmc2*hbcm2+hbcm2*p1p3+2*hbcm2*
     . p1p4-hbcm2*p3p4-2*p1p3*p2p3+2*p1p3*p3p5+2*p2p3*p3p4-2*p3p4*
     . p3p5)+8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*hbcm2-2*fb2*p2p3+2*
     . fb2*p3p5))
      b(13)=ccc*(w5*(4*p3p4*(-fmb+fmc)+4*p1p3*(fmb-fmc)+4*fmc*hbcm2)+
     . 8*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      do 200 n=1,14 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp1s2_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(8) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,8 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=ccc*(4*w1*(-fmc*hbcm2*p6p7-2*hbcm*p3p4*p6p7)+4*w2*(-fmb*
     . hbcm2*p6p7-2*hbcm*p3p5*p6p7-2*hbcm*p3p6*p5p7+4*hbcm*p3p7*p5p6)
     . )
      b(2)=4*ccc*(2*hbcm*p4p7*w1+hbcm*p5p7*w2)
      b(3)=4*ccc*(-hbcm*p4p6*w1-2*hbcm*p5p6*w2)
      b(4)=ccc*(4*w1*(2*fmc*hbcm*p6p7+p3p4*p6p7)+4*w2*(-2*fmb*hbcm*
     . p6p7-p3p5*p6p7)-8*fb2*p6p7)
      b(5)=ccc*(8*w1*(-fmc*hbcm*p3p7+hbcm2*p4p7-p3p4*p3p7)+4*w2*(2*
     . fmb*hbcm*p3p7+hbcm2*p5p7+2*p3p5*p3p7)+16*fb2*p3p7)
      b(6)=ccc*(4*w1*(-2*fmc*hbcm*p3p6-hbcm2*p4p6-2*p3p4*p3p6)+8*w2*(
     . fmb*hbcm*p3p6-hbcm2*p5p6+p3p5*p3p6)+16*fb2*p3p6)
      b(7)=ccc*(6*w1*(fmc*hbcm+p3p4)+6*w2*(-fmb*hbcm-p3p5)-12*fb2)
      b(8)=ccc*(6*w1*(fmc*hbcm2+hbcm*p3p4)+6*w2*(fmb*hbcm2+hbcm*p3p5)
     . -12*fb1*hbcm)
      do 200 n=1,8 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp28_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      ans5=8*(-2*ffmcfmb*hbcm*p1p2*p3p6*p3p7-2*ffmcfmb*hbcm*p1p3*p3p7
     . *p4p6-ffmcfmb*hbcm*p2p3*p3p7*p5p6+2*ffmcfmb*hbcm*p2p5*p3p6*
     . p3p7-ffmcfmb*hbcm*p3p4*p3p7*p5p6+2*ffmcfmb*hbcm*p3p5*p3p7*p4p6
     . +2*ffmcfmb*hbcm3*p2p5*p6p7-2*hbcm*p1p2*p3p6*p4p7+2*hbcm*p1p2*
     . p3p7*p4p6+2*hbcm*p1p4*p2p3*p6p7+2*hbcm*p2p3*p2p4*p6p7-2*hbcm*
     . p2p3*p2p5*p6p7+2*hbcm*p2p3*p4p5*p6p7+4*hbcm*p2p3*p4p6*p4p7+2*
     . hbcm*p2p3*p4p6*p5p7-2*hbcm*p2p3*p4p7*p5p6-4*hbcm*p2p4*p3p5*
     . p6p7-2*hbcm*p2p4*p3p6*p5p7-4*hbcm*p2p4*p3p7*p4p6+2*hbcm*p2p4*
     . p3p7*p5p6+2*hbcm*p2p5*p3p4*p6p7+2*hbcm*p2p5*p3p6*p4p7-2*hbcm*
     . p2p5*p3p7*p4p6)
      ans4=4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm+fmc2*hbcm)+4*(p6p7*p2p3)*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc
     . *hbcm+2*fmc*hbcm2-3*fmc2*hbcm)+8*(p3p7*p3p6*p3p4)*(-ffmcfmb**2
     . *hbcm-ffmcfmb*fmb+4*ffmcfmb*fmc+5*ffmcfmb*hbcm-3*fmc)+8*(p3p7*
     . p3p6*p2p3)*(ffmcfmb**2*hbcm+ffmcfmb*fmb-4*ffmcfmb*fmc-ffmcfmb*
     . hbcm+3*fmc)+4*p6p7*(ffmcfmb**2*fmc*hbcm4-2*ffmcfmb*fmb*fmc*
     . hbcm3-2*ffmcfmb*fmc*hbcm4+2*ffmcfmb*hbcm*p2p3s2+2*fmb*fmc2*
     . hbcm2-4*fmb*p2p3s2+2*fmc2*hbcm3-fmc3*hbcm2)+8*(p3p7*p3p6)*(-
     . ffmcfmb**2*fmc*hbcm2-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*fmc2*hbcm+2*fmc2*hbcm)+ans5
      ans3=8*(p5p7*p4p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p7*p4p6)*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p4p5)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+8*(p6p7*p2p4)*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*
     . hbcm3)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p6*p3p7*
     . p3p4)*(3*ffmcfmb*hbcm-2*fmc)+8*(p5p7*p3p6*p3p4)*(ffmcfmb*hbcm-
     . fmc)+8*(p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm+fmc)+16*(p6p7*p3p5*p3p4
     . )*(ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6*p2p4)*(-ffmcfmb*hbcm-2*hbcm
     . )+8*(p4p6*p3p7*p2p3)*(-ffmcfmb*hbcm+2*fmc)+8*(p5p7*p3p6*p2p3)*
     . (-ffmcfmb*hbcm+fmc)+8*(p4p7*p3p6*p2p3)*(ffmcfmb*hbcm-fmc+4*
     . hbcm)+16*(p6p7*p3p5*p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p6p7*p3p4*
     . p2p3)*(-3*ffmcfmb*hbcm+2*fmb+2*hbcm)+8*(p5p6*p3p7)*(ffmcfmb*
     . fmc*hbcm2-fmc2*hbcm)+8*(p4p6*p3p7)*(-2*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc2*hbcm)+8*(p5p7*p3p6)*(-ffmcfmb*
     . fmc*hbcm2+fmc2*hbcm)+16*(p6p7*p3p5)*(-ffmcfmb*fmc*hbcm2+fmc2*
     . hbcm)+8*(p4p7*p3p6)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmb*fmc*hbcm+3*fmc*hbcm2+
     . fmc2*hbcm)+ans4
      ans2=w11*ans3
      ans8=4*p6p7*(ffmcfmb**2*fmb*hbcm4-2*ffmcfmb*fmb*fmc*hbcm3-2*
     . ffmcfmb*fmb*hbcm4+2*ffmcfmb*fmb2*hbcm3-4*ffmcfmb*hbcm*p3p5s2+2
     . *fmb*fmc*hbcm3+fmb*fmc2*hbcm2-2*fmb2*fmc*hbcm2+4*fmc*p3p5s2)+8
     . *(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2-2*ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm+2*fmb*fmc*hbcm)+8*(-2*
     . ffmcfmb*hbcm*p1p5*p3p6*p3p7-2*ffmcfmb*hbcm*p2p3*p3p7*p5p6+3*
     . ffmcfmb*hbcm*p3p5*p3p7*p5p6-2*ffmcfmb*hbcm*p3p6*p3p7*p4p5+2*
     . fmb*hbcm2*p2p5*p6p7+2*hbcm*p2p3*p4p6*p5p7+2*hbcm*p2p3*p5p6*
     . p5p7+2*hbcm*p2p5*p3p5*p6p7-2*hbcm*p2p5*p3p7*p4p6-2*hbcm*p2p5*
     . p3p7*p5p6)
      ans7=8*(p5p7*p5p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p5p7*p4p6)*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p6*p3p7*p3p5)*(ffmcfmb*hbcm+2*
     . fmc)+8*(p5p7*p3p6*p3p5)*(-ffmcfmb*hbcm+fmc)+8*(p4p7*p3p6*p3p5)
     . *(ffmcfmb*hbcm-fmc)+16*(p3p7*p3p6*p2p5)*(-ffmcfmb*hbcm+hbcm)+
     . 16*(p5p7*p3p6*p2p3)*(2*ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p5*p2p3)*(
     . ffmcfmb*hbcm-2*fmb)+8*(p4p6*p3p7)*(-ffmcfmb*fmb*hbcm2+3*fmb*
     . fmc*hbcm)+8*(p4p7*p3p6)*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+8*(
     . p6p7*p2p3)*(ffmcfmb*fmb*hbcm2-2*fmb2*hbcm)+8*(p5p6*p3p7)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmc2*hbcm)+8*(p5p7*p3p6)*(-2*ffmcfmb**2*hbcm3-ffmcfmb*fmb
     . *hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*
     . hbcm2)+4*(p6p7*p3p5)*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm+2*fmc*hbcm2+
     . fmc2*hbcm)+8*(p3p7*p3p6*p3p5)*(ffmcfmb**2*hbcm+ffmcfmb*fmb-4*
     . ffmcfmb*fmc-ffmcfmb*hbcm+3*fmc)+ans8
      ans6=w2*ans7
      ans9=(4*(p5p6*p3p7)*(-4*fb1*ffmcfmb*hbcm-hbcm)+4*(p4p6*p3p7)*(4
     . *fb1*ffmcfmb*hbcm+8*fb2*fmc-3*hbcm)+4*(p5p7*p3p6)*(4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmc-hbcm)+4*(p4p7*p3p6)*(-4*fb1*ffmcfmb*
     . hbcm-4*fb2*fmc+hbcm)+8*(p6p7*p3p5)*(4*fb1*ffmcfmb*hbcm+4*fb2*
     . fmc-hbcm)+4*(p6p7*p2p3)*(-4*fb1*ffmcfmb*hbcm-8*fb2*fmb+hbcm)+4
     . *p6p7*(-2*fb1*ffmcfmb**2*hbcm3+4*fb1*ffmcfmb*hbcm3+4*fb1*fmb*
     . fmc*hbcm-2*fb1*fmc2*hbcm+4*fb2*ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb
     . *fmc*hbcm2+4*fb2*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*hbcm2+3*fmc*
     . hbcm2-hbcm3)+4*(p3p7*p3p6)*(4*fb1*ffmcfmb**2*hbcm-4*fb1*
     . ffmcfmb*hbcm+4*fb2*ffmcfmb*fmb-16*fb2*ffmcfmb*fmc+12*fb2*fmc+4
     . *ffmcfmb*hbcm+fmb-4*fmc-7*hbcm)-32*fb1*hbcm*p2p5*p6p7)
      ans1=w7*(16*(p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*p6p7*(
     . ffmcfmb*hbcm*p2p3s2-2*fmb*p2p3s2)+8*(p4p6*p3p7*p2p3)*(ffmcfmb*
     . hbcm+2*fmc)+8*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+fmc)+8*(p4p7*
     . p3p6*p2p3)*(ffmcfmb*hbcm-fmc)+16*(p6p7*p3p5*p2p3)*(-ffmcfmb*
     . hbcm+fmc)+4*(p6p7*p2p3)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-
     . 2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmc*hbcm2
     . +fmc2*hbcm)+8*(p3p7*p3p6*p2p3)*(ffmcfmb**2*hbcm+ffmcfmb*fmb-4*
     . ffmcfmb*fmc-ffmcfmb*hbcm+3*fmc)+8*(-2*ffmcfmb*hbcm*p1p2*p3p6*
     . p3p7-ffmcfmb*hbcm*p2p3*p3p7*p5p6-2*ffmcfmb*hbcm*p2p4*p3p6*p3p7
     . +2*ffmcfmb*hbcm*p2p5*p3p6*p3p7-2*hbcm*p2p3*p2p5*p6p7))+ans2+
     . ans6+ans9
      ans=ccc*ans1
      b(1)=ans
      ans2=w2*(12*(p5p7*p3p5)*(ffmcfmb*hbcm-fmc)+4*(p4p7*p3p5)*(-
     . ffmcfmb*hbcm+fmc)+12*p5p7*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+4*
     . p4p7*(-ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+4*(p3p7*p3p5)*(ffmcfmb
     . **2*hbcm-3*ffmcfmb*fmb+4*ffmcfmb*fmc-3*fmc)+4*p3p7*(-ffmcfmb**
     . 2*fmb*hbcm2+4*ffmcfmb*fmb*fmc*hbcm-3*ffmcfmb*fmb2*hbcm-3*fmb*
     . fmc*hbcm)+8*(ffmcfmb*hbcm*p1p5*p3p7-ffmcfmb*hbcm*p2p5*p3p7+
     . ffmcfmb*hbcm*p3p7*p4p5))+(6*p5p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*
     . fmc+hbcm)+2*p4p7*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm)+2*p3p7*(4
     . *fb1*ffmcfmb**2*hbcm-12*fb2*ffmcfmb*fmb+16*fb2*ffmcfmb*fmc-12*
     . fb2*fmc+4*ffmcfmb*hbcm-3*fmb+4*fmc+6*hbcm))
      ans1=w7*(12*(p5p7*p2p3)*(ffmcfmb*hbcm-fmc)+4*(p4p7*p2p3)*(-
     . ffmcfmb*hbcm+fmc)+4*(p3p7*p2p3)*(ffmcfmb**2*hbcm-3*ffmcfmb*fmb
     . +4*ffmcfmb*fmc-3*fmc)+8*(ffmcfmb*hbcm*p1p2*p3p7+ffmcfmb*hbcm*
     . p2p4*p3p7-3*ffmcfmb*hbcm*p2p5*p3p7))+w11*(12*(p5p7*p3p4)*(-
     . ffmcfmb*hbcm+fmc)+4*(p4p7*p3p4)*(ffmcfmb*hbcm-fmc)+12*(p5p7*
     . p2p3)*(ffmcfmb*hbcm-fmc)+4*(p4p7*p2p3)*(ffmcfmb*hbcm+fmc-6*
     . hbcm)+12*p5p7*(ffmcfmb*fmc*hbcm2-fmc2*hbcm)+4*p4p7*(-3*ffmcfmb
     . *fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+3*fmb*fmc*hbcm-
     . 3*fmc*hbcm2-2*fmc2*hbcm)+4*(p3p7*p3p4)*(-ffmcfmb**2*hbcm+3*
     . ffmcfmb*fmb-4*ffmcfmb*fmc-4*ffmcfmb*hbcm+3*fmc)+4*(p3p7*p2p3)*
     . (ffmcfmb**2*hbcm-3*ffmcfmb*fmb+4*ffmcfmb*fmc-3*fmc)+4*p3p7*(-
     . ffmcfmb**2*fmc*hbcm2+3*ffmcfmb*fmb*fmc*hbcm-2*ffmcfmb*fmc2*
     . hbcm-3*fmc2*hbcm)+8*(ffmcfmb*hbcm*p1p2*p3p7+ffmcfmb*hbcm*p1p4*
     . p3p7-3*ffmcfmb*hbcm*p2p5*p3p7+ffmcfmb*hbcm*p3p7*p4p5+hbcm*p1p2
     . *p4p7+3*hbcm*p2p4*p3p7+3*hbcm*p2p4*p5p7-3*hbcm*p2p5*p4p7))+
     . ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w2*(4*(p5p6*p3p5)*(-ffmcfmb*hbcm+fmc)+12*(p4p6*p3p5)*(
     . ffmcfmb*hbcm-fmc)+4*(p3p6*p3p5)*(ffmcfmb*hbcm-fmc)+8*(p3p6*
     . p2p5)*(2*ffmcfmb*hbcm-hbcm)+8*(p5p6*p2p3)*(-ffmcfmb*hbcm+2*
     . hbcm)+16*p4p6*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+4*p5p6*(ffmcfmb
     . **2*hbcm3-2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmc*hbcm2+fmc2
     . *hbcm)+8*p3p6*(ffmcfmb**2*fmb*hbcm2-ffmcfmb*fmb*fmc*hbcm)+8*(
     . hbcm*p2p5*p4p6+hbcm*p2p5*p5p6))+(2*p5p6*(4*fb1*ffmcfmb*hbcm+4*
     . fb2*fmc-hbcm)+6*p4p6*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+2*
     . p3p6*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm))
      ans1=w7*(4*(p5p6*p2p3)*(-ffmcfmb*hbcm+fmc)+12*(p4p6*p2p3)*(
     . ffmcfmb*hbcm-fmc)+4*(p3p6*p2p3)*(ffmcfmb*hbcm-fmc))+w11*(4*(
     . p5p6*p3p4)*(ffmcfmb*hbcm-fmc)+12*(p4p6*p3p4)*(-ffmcfmb*hbcm+
     . fmc)+4*(p3p6*p3p4)*(-ffmcfmb*hbcm+fmc)+4*(p5p6*p2p3)*(-ffmcfmb
     . *hbcm+fmc)+4*(p4p6*p2p3)*(3*ffmcfmb*hbcm-3*fmc+2*hbcm)+4*(p3p6
     . *p2p3)*(ffmcfmb*hbcm-fmc)+8*p5p6*(-ffmcfmb*fmc*hbcm2+fmc2*hbcm
     . )+4*p4p6*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2-2*fmc2*hbcm)+8*p3p6
     . *(ffmcfmb**2*fmc*hbcm2-ffmcfmb*fmc2*hbcm)+8*(2*ffmcfmb*hbcm*
     . p2p4*p3p6-hbcm*p1p2*p4p6+3*hbcm*p2p4*p4p6-2*hbcm*p2p4*p5p6+
     . hbcm*p2p5*p4p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans4=8*(-ffmcfmb*fmc*hbcm*p3p6*p3p7+ffmcfmb*fmc*hbcm*p3p7*p5p6-
     . ffmcfmb*hbcm2*p1p2*p6p7-ffmcfmb*hbcm2*p2p5*p6p7+2*ffmcfmb*p2p3
     . *p3p7*p4p6-2*ffmcfmb*p3p4*p3p7*p4p6+p1p2*p2p3*p6p7-p1p2*p3p4*
     . p6p7-p2p3*p2p4*p6p7+p2p3*p2p5*p6p7+p2p4*p3p4*p6p7-p2p5*p3p4*
     . p6p7)
      ans3=8*(p6p7*p2p4)*(-2*fmb*hbcm-fmc*hbcm+2*hbcm2)+(p6p7*p3p4*
     . p2p3)*(8*ffmcfmb-8)+8*(p5p6*p4p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*
     . (p5p7*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p4p6)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(p6p7*p4p5)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p7
     . *p3p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p1p4)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+16*(p3p7*p3p6*p3p4)*(ffmcfmb**2-ffmcfmb)+16*(p3p7*
     . p3p6*p2p3)*(-ffmcfmb**2+ffmcfmb)+8*(p4p6*p3p7)*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p6p7*p3p4
     . )*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2*
     . fmb*fmc-fmc2)+4*(p6p7*p2p3)*(ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*
     . hbcm+2*fmb*fmc-2*fmc*hbcm+fmc2)+4*p6p7*(-ffmcfmb**2*fmc*hbcm3-
     . 2*ffmcfmb*fmb*fmc*hbcm2+2*ffmcfmb*fmc*hbcm3-2*ffmcfmb*p2p3s2+2
     . *fmb*fmc2*hbcm-2*fmc2*hbcm2+fmc3*hbcm+2*p2p3s2)+ans4
      ans2=w11*ans3
      ans6=((p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(p6p7*p2p3)*(-16*fb2*
     . ffmcfmb+16*fb2-4)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+32*fb2*
     . ffmcfmb-16*ffmcfmb+8)+4*p6p7*(4*fb1*ffmcfmb*fmb*hbcm+4*fb1*
     . ffmcfmb*fmc*hbcm-4*fb1*fmc*hbcm+2*fb2*ffmcfmb**2*hbcm2-4*fb2*
     . ffmcfmb*hbcm2+4*fb2*fmb*fmc+2*fb2*fmc2-fmb*hbcm-fmc*hbcm+hbcm2
     . )+16*(fb2*p1p2*p6p7-fb2*p2p4*p6p7+fb2*p2p5*p6p7))
      ans5=w2*((p6p7*p3p5*p2p3)*(-8*ffmcfmb+8)+8*(p5p7*p5p6)*(ffmcfmb
     . *hbcm2-fmc*hbcm)+8*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(
     . p5p6*p3p7)*(ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2*fmc*hbcm)+8*
     . (p6p7*p2p3)*(-ffmcfmb*fmb*hbcm+fmb*hbcm)+16*(p3p7*p3p6*p3p5)*(
     . -ffmcfmb**2+ffmcfmb)+8*(p5p7*p3p6)*(2*ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7*p3p5)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*
     . ffmcfmb*hbcm2+2*fmb*fmc+2*fmc*hbcm+fmc2)+4*p6p7*(ffmcfmb**2*
     . fmb*hbcm3-2*ffmcfmb*fmb*fmc*hbcm2-2*ffmcfmb*fmb*hbcm3-2*
     . ffmcfmb*fmb2*hbcm2+2*fmb*fmc*hbcm2+fmb*fmc2*hbcm+2*fmb2*fmc*
     . hbcm)+8*(ffmcfmb*fmb*hbcm*p3p6*p3p7+3*ffmcfmb*fmb*hbcm*p3p7*
     . p4p6+2*ffmcfmb*p3p5*p3p7*p4p6+fmb*hbcm*p1p2*p6p7-fmb*hbcm*p2p4
     . *p6p7+fmb*hbcm*p2p5*p6p7+p1p2*p3p5*p6p7-p2p4*p3p5*p6p7+p2p5*
     . p3p5*p6p7))+ans6
      ans1=w7*(8*p6p7*(-ffmcfmb*p2p3s2+p2p3s2)+8*(p6p7*p2p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +8*(p6p7*p1p2)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p3p7*p3p6*p2p3)*(
     . -ffmcfmb**2+ffmcfmb)+4*(p6p7*p2p3)*(ffmcfmb**2*hbcm2-2*ffmcfmb
     . *fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*fmc+fmc2)+8*(2*ffmcfmb*p2p3
     . *p3p7*p4p6+p1p2*p2p3*p6p7-p2p3*p2p4*p6p7+p2p3*p2p5*p6p7))+ans2
     . +ans5
      ans=ccc*ans1
      b(4)=ans
      ans4=8*(ffmcfmb*fmc*hbcm*p1p3*p3p7-3*ffmcfmb*fmc*hbcm*p3p5*p3p7
     . -ffmcfmb*p1p3*p2p3*p3p7+ffmcfmb*p1p3*p3p4*p3p7+ffmcfmb*p2p3*
     . p3p4*p3p7+3*ffmcfmb*p2p3*p3p5*p3p7-3*ffmcfmb*p3p4*p3p5*p3p7+
     . fmc*hbcm*p1p4*p3p7+fmc*hbcm*p3p7*p4p5+hbcm2*p1p2*p4p7+3*hbcm2*
     . p2p4*p5p7-3*hbcm2*p2p5*p4p7+p1p2*p2p3*p3p7-p1p2*p3p4*p3p7+p2p3
     . *p2p4*p3p7-3*p2p3*p2p5*p3p7+p2p3*p3p4*p4p7-3*p2p3*p3p4*p5p7-
     . p2p4*p3p4*p3p7+3*p2p5*p3p4*p3p7)
      ans3=24*(p4p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+12*(p5p7*p3p4)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm
     . )+8*(p3p7*p2p4)*(-ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*hbcm)+12*(
     . p5p7*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p7*p2p3)*(-ffmcfmb*
     . hbcm2-6*fmb*hbcm-fmc*hbcm)+8*(p4p7*p1p3)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+12*p5p7*(ffmcfmb*fmc*hbcm3-fmc2*hbcm2+2*p2p3s2)+4*p4p7*(
     . -2*ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*hbcm3+4*ffmcfmb*fmc*hbcm3+3*
     . ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*fmc*hbcm3-2*fmc2*hbcm2-2*
     . p2p3s2)+4*(p3p7*p3p4)*(3*ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-7
     . *ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc+2*fmc*hbcm+3*fmc2)
     . +4*(p3p7*p2p3)*(-ffmcfmb**2*hbcm2+5*ffmcfmb*fmc*hbcm+3*ffmcfmb
     . *hbcm2-3*fmb*fmc-3*fmc2)+4*p3p7*(ffmcfmb**2*fmc*hbcm3-3*
     . ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2-4*ffmcfmb*p2p3s2+2*
     . ffmcfmb*p3p4s2-3*fmb*fmc2*hbcm-3*fmc3*hbcm)+ans4
      ans2=w11*ans3
      ans6=((p3p7*p3p5)*(48*fb2*ffmcfmb+12)+(p3p7*p3p4)*(-16*fb2*
     . ffmcfmb-4)+(p3p7*p2p3)*(-32*fb2*ffmcfmb-8)+(p3p7*p1p3)*(-16*
     . fb2*ffmcfmb-4)+6*p5p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+
     . hbcm2)+2*p4p7*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-hbcm2)+2*
     . p3p7*(4*fb1*ffmcfmb*fmc*hbcm-4*fb2*ffmcfmb**2*hbcm2+12*fb2*
     . ffmcfmb*hbcm2-12*fb2*fmb*fmc-12*fb2*fmc2+2*ffmcfmb*hbcm2+6*fmb
     . *hbcm+11*fmc*hbcm+3*hbcm2)+16*(fb2*p1p2*p3p7-fb2*p2p3*p4p7+3*
     . fb2*p2p3*p5p7+fb2*p2p4*p3p7-3*fb2*p2p5*p3p7))
      ans5=w2*(8*(p3p7*p2p5)*(-3*fmb*hbcm-fmc*hbcm)+12*(p5p7*p3p5)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +12*p5p7*(-ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2)+4*p4p7*(ffmcfmb*
     . fmb*hbcm3-fmb*fmc*hbcm2)+4*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2+6*
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2-3*fmb*fmc-3*
     . fmc2)+4*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb*fmb*fmc*hbcm2+3*
     . ffmcfmb*fmb*hbcm3+6*ffmcfmb*p3p5s2-3*fmb*fmc2*hbcm-3*fmb2*fmc*
     . hbcm)+8*(-ffmcfmb*fmb*hbcm*p1p3*p3p7-2*ffmcfmb*fmb*hbcm*p2p3*
     . p3p7-ffmcfmb*fmb*hbcm*p3p4*p3p7-ffmcfmb*p1p3*p3p5*p3p7-2*
     . ffmcfmb*p2p3*p3p5*p3p7-ffmcfmb*p3p4*p3p5*p3p7+fmb*hbcm*p1p2*
     . p3p7-fmb*hbcm*p2p3*p4p7+3*fmb*hbcm*p2p3*p5p7+fmb*hbcm*p2p4*
     . p3p7+fmc*hbcm*p1p5*p3p7+fmc*hbcm*p3p7*p4p5+p1p2*p3p5*p3p7-p2p3
     . *p3p5*p4p7+3*p2p3*p3p5*p5p7+p2p4*p3p5*p3p7-3*p2p5*p3p5*p3p7))+
     . ans6
      ans1=w7*(12*(p5p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p2p3)
     . *(ffmcfmb*hbcm2-fmc*hbcm)+4*(p3p7*p2p3)*(-ffmcfmb**2*hbcm2+
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2-3*fmb*fmc-3*fmc2)+8*(-ffmcfmb
     . *p1p3*p2p3*p3p7-ffmcfmb*p2p3*p3p4*p3p7+3*ffmcfmb*p2p3*p3p5*
     . p3p7-2*ffmcfmb*p2p3s2*p3p7+fmc*hbcm*p1p2*p3p7+fmc*hbcm*p2p4*
     . p3p7-3*fmc*hbcm*p2p5*p3p7+p1p2*p2p3*p3p7+p2p3*p2p4*p3p7-3*p2p3
     . *p2p5*p3p7-p2p3s2*p4p7+3*p2p3s2*p5p7))+ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans4=8*(2*ffmcfmb*hbcm2*p2p5*p3p6-hbcm2*p1p2*p4p6+2*hbcm2*p2p4*
     . p4p6-hbcm2*p2p4*p5p6+hbcm2*p2p5*p4p6-2*p2p3*p2p5*p3p6-3*p2p3*
     . p3p4*p4p6+p2p3*p3p4*p5p6+2*p2p5*p3p4*p3p6)
      ans3=(p3p6*p3p4*p2p3)*(-8*ffmcfmb+8)+8*(p4p5*p3p6)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(p4p6*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p5p6*p3p4)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*p3p4)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(p3p6*p2p4)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*
     . hbcm-hbcm2)+8*(p5p6*p2p3)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*
     . p2p3)*(-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+8*(p3p6*p1p4)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)
     . +4*p5p6*(-ffmcfmb*fmc*hbcm3+fmc2*hbcm2-2*p2p3s2)+4*p4p6*(
     . ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2
     . +fmc*hbcm3-fmc2*hbcm2+6*p2p3s2)+4*(p3p6*p3p4)*(-ffmcfmb**2*
     . hbcm2-2*ffmcfmb*fmb*hbcm+2*fmb*fmc+fmc2)+4*(p3p6*p2p3)*(-3*
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*
     . ffmcfmb*hbcm2-2*fmb*fmc+2*fmc*hbcm-fmc2)+4*p3p6*(ffmcfmb**2*
     . fmc*hbcm3+2*ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmc*hbcm3+2*ffmcfmb*
     . p2p3s2-2*fmb*fmc2*hbcm+fmc2*hbcm2-fmc3*hbcm-2*p2p3s2)+ans4
      ans2=w11*ans3
      ans5=w2*((p3p6*p3p5*p2p3)*(8*ffmcfmb-8)+8*(p5p6*p3p5)*(-ffmcfmb
     . *hbcm2+fmc*hbcm)+16*(p4p6*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p3p6*p2p5)*(2*ffmcfmb*hbcm2-2*fmb*hbcm-hbcm2)+12*p4p6*(-
     . ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2)+8*(p3p6*p2p3)*(3*ffmcfmb*fmb*
     . hbcm-2*fmb*hbcm)+4*p5p6*(ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3-2*
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+
     . fmc2*hbcm2)+4*(p3p6*p3p5)*(-3*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*
     . hbcm+4*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-2*fmb*fmc-2*fmc*hbcm-
     . fmc2)+4*p3p6*(-ffmcfmb**2*fmb*hbcm3+2*ffmcfmb*fmb*fmc*hbcm2+
     . ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmb2*hbcm2-fmb*fmc*hbcm2-fmb*fmc2*
     . hbcm-2*fmb2*fmc*hbcm)+8*(ffmcfmb*hbcm2*p2p3*p5p6+4*fmb*hbcm*
     . p2p3*p4p6+hbcm2*p2p5*p4p6+hbcm2*p2p5*p5p6+3*p2p3*p3p5*p4p6-
     . p2p3*p3p5*p5p6-2*p2p5*p3p5*p3p6))+((p3p6*p2p3)*(16*fb2*ffmcfmb
     . -16*fb2+4)+4*p4p6*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+
     . 4*p3p6*(-4*fb1*ffmcfmb*fmb*hbcm+2*fb2*ffmcfmb**2*hbcm2-4*fb2*
     . fmb*fmc-2*fb2*fmc2+fmb*hbcm)+16*(3*fb2*p2p3*p4p6-fb2*p2p3*p5p6
     . -2*fb2*p2p5*p3p6))
      ans1=w7*(8*p3p6*(ffmcfmb*p2p3s2-p2p3s2)+16*(p3p6*p2p5)*(ffmcfmb
     . *hbcm2-fmc*hbcm)+8*(p5p6*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p4p6*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p3p6*p2p3)*(-3*ffmcfmb
     . **2*hbcm2+2*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm+2*ffmcfmb*
     . hbcm2-2*fmb*fmc-2*fmc*hbcm-fmc2)+8*(-2*p2p3*p2p5*p3p6+3*p2p3s2
     . *p4p6-p2p3s2*p5p6))+ans2+ans5
      ans=ccc*ans1
      b(6)=ans
      ans2=w11*((p3p4*p2p3)*(4*ffmcfmb-12)+4*p4p5*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*p2p4*(ffmcfmb*hbcm2-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2)+4*
     . p1p4*(ffmcfmb*hbcm2-fmc*hbcm)+2*p3p4*(-2*ffmcfmb**2*hbcm2+3*
     . ffmcfmb*fmb*hbcm+5*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-3*fmb*fmc+
     . fmc*hbcm-3*fmc2)+2*p2p3*(2*ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm
     . -3*ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc-3*fmc*hbcm+3*
     . fmc2)+2*(-3*ffmcfmb*fmb*fmc*hbcm2+3*ffmcfmb*fmc*hbcm3-3*
     . ffmcfmb*fmc2*hbcm2+2*ffmcfmb*hbcm2*p1p2-6*ffmcfmb*hbcm2*p2p5-2
     . *ffmcfmb*p2p3s2+3*fmb*fmc2*hbcm-3*fmc2*hbcm2+3*fmc3*hbcm-2*
     . p1p2*p2p3+2*p1p2*p3p4-2*p2p3*p2p4+6*p2p3*p2p5+6*p2p3s2+2*p2p4*
     . p3p4-6*p2p5*p3p4))+(p2p3*(-8*fb2*ffmcfmb+24*fb2-2)+(12*fb1*
     . ffmcfmb*fmb*hbcm+12*fb1*ffmcfmb*fmc*hbcm-12*fb1*fmc*hbcm-12*
     . fb2*ffmcfmb*hbcm2+12*fb2*fmb*fmc+12*fb2*fmc2-8*fb2*p1p2-8*fb2*
     . p2p4+24*fb2*p2p5-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2))
      ans1=w2*((p3p5*p2p3)*(-4*ffmcfmb+12)+4*p4p5*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*p2p5*(-ffmcfmb*hbcm2+3*fmb*hbcm+fmc*hbcm)+4*p1p5*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p2p3*(-ffmcfmb*fmb*hbcm+3*fmb*hbcm)+
     . 2*p3p5*(2*ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-5*ffmcfmb*fmc*
     . hbcm-3*ffmcfmb*hbcm2+3*fmb*fmc+3*fmc*hbcm+3*fmc2)+2*(-3*
     . ffmcfmb*fmb*fmc*hbcm2-3*ffmcfmb*fmb*hbcm3-3*ffmcfmb*fmb2*hbcm2
     . +3*fmb*fmc*hbcm2+3*fmb*fmc2*hbcm-2*fmb*hbcm*p1p2-2*fmb*hbcm*
     . p2p4+3*fmb2*fmc*hbcm-2*p1p2*p3p5-2*p2p4*p3p5+6*p2p5*p3p5))+w7*
     . (12*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p2p4*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+2*p2p3*(2*ffmcfmb**2*
     . hbcm2-3*ffmcfmb*fmb*hbcm-5*ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2+3*
     . fmb*fmc+3*fmc*hbcm+3*fmc2)+4*(-ffmcfmb*p2p3s2-p1p2*p2p3-p2p3*
     . p2p4+3*p2p3*p2p5+3*p2p3s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=2*(-2*ffmcfmb**2*fmc*hbcm4+3*ffmcfmb*fmb*fmc*hbcm3+3*
     . ffmcfmb*fmc*hbcm4-ffmcfmb*fmc2*hbcm3-6*ffmcfmb*hbcm*p2p3s2+2*
     . ffmcfmb*hbcm*p3p4s2+2*ffmcfmb*hbcm3*p1p2-6*ffmcfmb*hbcm3*p2p5-
     . 3*fmb*fmc2*hbcm2+6*fmb*p2p3s2-2*fmc*p2p3s2-2*fmc*p3p4s2-3*fmc2
     . *hbcm3+3*fmc3*hbcm2-2*hbcm*p1p2*p2p3+2*hbcm*p1p2*p3p4-4*hbcm*
     . p1p3*p2p4-4*hbcm*p1p4*p2p3-10*hbcm*p2p3*p2p4+6*hbcm*p2p3*p2p5-
     . 4*hbcm*p2p3*p4p5-2*hbcm*p2p4*p3p4+12*hbcm*p2p4*p3p5-6*hbcm*
     . p2p5*p3p4)
      ans3=4*p4p5*(ffmcfmb*hbcm3-fmc*hbcm2)+4*p2p4*(-ffmcfmb*hbcm3+3*
     . fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+4*p1p4*(ffmcfmb*hbcm3-fmc*hbcm2
     . )+12*(p3p5*p3p4)*(-ffmcfmb*hbcm+fmc)+12*(p3p5*p2p3)*(ffmcfmb*
     . hbcm-fmc)+4*(p3p4*p2p3)*(4*ffmcfmb*hbcm-3*fmb+2*fmc-2*hbcm)+4*
     . (p3p4*p1p3)*(ffmcfmb*hbcm-fmc)+4*(p2p3*p1p3)*(-ffmcfmb*hbcm+
     . fmc)+12*p3p5*(ffmcfmb*fmc*hbcm2-fmc2*hbcm)+4*p1p3*(-ffmcfmb*
     . fmc*hbcm2+fmc2*hbcm)+2*p3p4*(-3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3+3*fmb*fmc*hbcm+fmc*hbcm2-fmc2*hbcm)+2*p2p3
     . *(-3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-3*fmb
     . *fmc*hbcm-3*fmc*hbcm2+7*fmc2*hbcm)+ans4
      ans2=w11*ans3
      ans6=(6*p3p5*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+hbcm)+2*p3p4*(4*fb1
     . *ffmcfmb*hbcm+4*fb2*fmc-hbcm)+2*p2p3*(4*fb1*ffmcfmb*hbcm+12*
     . fb2*fmb-4*fb2*fmc-hbcm)+2*p1p3*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-
     . hbcm)+(8*fb1*ffmcfmb**2*hbcm3-12*fb1*ffmcfmb*hbcm3-12*fb1*fmb*
     . fmc*hbcm+12*fb1*fmc2*hbcm-8*fb1*hbcm*p1p2-8*fb1*hbcm*p2p4+24*
     . fb1*hbcm*p2p5-12*fb2*ffmcfmb*fmb*hbcm2+20*fb2*ffmcfmb*fmc*
     . hbcm2-12*fb2*fmc*hbcm2+2*ffmcfmb*hbcm3+3*fmb*hbcm2-7*fmc*hbcm2
     . +3*hbcm3))
      ans5=w2*(4*p4p5*(ffmcfmb*hbcm3-fmc*hbcm2)+4*p2p5*(-ffmcfmb*
     . hbcm3-3*fmb*hbcm2+fmc*hbcm2)+4*p1p5*(ffmcfmb*hbcm3-fmc*hbcm2)+
     . 4*(p3p5*p3p4)*(-ffmcfmb*hbcm+fmc)+4*(p3p5*p2p3)*(-3*ffmcfmb*
     . hbcm+3*fmb-fmc)+4*(p3p5*p1p3)*(-ffmcfmb*hbcm+fmc)+6*p3p5*(
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm
     . -fmc*hbcm2-fmc2*hbcm)+4*p3p4*(-ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)
     . +4*p2p3*(-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+3*fmb2*hbcm)+4*p1p3*(
     . -ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+2*(-2*ffmcfmb**2*fmb*hbcm4+5*
     . ffmcfmb*fmb*fmc*hbcm3+3*ffmcfmb*fmb*hbcm4-3*ffmcfmb*fmb2*hbcm3
     . +6*ffmcfmb*hbcm*p3p5s2-3*fmb*fmc*hbcm3-3*fmb*fmc2*hbcm2+2*fmb*
     . hbcm2*p1p2+2*fmb*hbcm2*p2p4+3*fmb2*fmc*hbcm2-6*fmc*p3p5s2+2*
     . hbcm*p1p2*p3p5-4*hbcm*p1p5*p2p3+4*hbcm*p2p3*p2p5-4*hbcm*p2p3*
     . p4p5+2*hbcm*p2p4*p3p5-6*hbcm*p2p5*p3p5))+ans6
      ans1=w7*(12*p2p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*p2p4*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+4*p1p2*(ffmcfmb*hbcm3-fmc*hbcm2)+12*(p3p5*
     . p2p3)*(ffmcfmb*hbcm-fmc)+4*(p3p4*p2p3)*(-ffmcfmb*hbcm+fmc)+4*(
     . p2p3*p1p3)*(-ffmcfmb*hbcm+fmc)+6*p2p3*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*
     . hbcm)+4*(-3*ffmcfmb*hbcm*p2p3s2+3*fmb*p2p3s2-fmc*p2p3s2-hbcm*
     . p1p2*p2p3-hbcm*p2p3*p2p4+3*hbcm*p2p3*p2p5))+ans2+ans5
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w2*(8*p5p6*(ffmcfmb*hbcm3-hbcm3)+4*(p3p6*p3p5)*(-6*
     . ffmcfmb*hbcm+2*fmb+3*hbcm)+8*p3p6*(-2*ffmcfmb*fmb*hbcm2+fmb*
     . hbcm2+fmb2*hbcm)+4*(-4*fmb*hbcm2*p4p6-5*hbcm*p3p5*p4p6-hbcm*
     . p3p5*p5p6))+w7*(4*(p3p6*p2p3)*(2*fmb-hbcm)+4*p5p6*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+12*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*p3p6*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+4*(3*hbcm*
     . p2p3*p4p6-hbcm*p2p3*p5p6))+w11*(4*(p3p6*p2p3)*(2*fmb-hbcm)+4*
     . p4p6*(-fmb*hbcm2-fmc*hbcm2-hbcm3)+4*p5p6*(ffmcfmb*hbcm3+fmc*
     . hbcm2)+4*(p3p6*p3p4)*(-2*ffmcfmb*hbcm-2*fmb+hbcm)+4*p3p6*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmc*hbcm2+fmc2*hbcm)+4*(2*hbcm*p1p3*p4p6-2*hbcm*
     . p1p4*p3p6+hbcm*p2p3*p4p6-hbcm*p2p3*p5p6-2*hbcm*p2p4*p3p6-3*
     . hbcm*p3p4*p4p6+3*hbcm*p3p4*p5p6-2*hbcm*p3p5*p4p6-2*hbcm*p3p6*
     . p4p5))+(8*p3p6*(2*fb1*ffmcfmb*hbcm-fb1*hbcm+2*fb2*fmb+hbcm)+8*
     . (3*fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(9)=ans
      b(10)=ccc*(w2*(4*p5p6*(-2*ffmcfmb*hbcm2+fmb*hbcm+2*hbcm2)+4*
     . p3p6*(-4*ffmcfmb*fmb*hbcm+fmb*hbcm)+4*(-2*ffmcfmb*p3p5*p3p6-5*
     . fmb*hbcm*p4p6-4*p3p5*p4p6+2*p3p5*p5p6))+w7*(8*p5p6*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+16*p4p6*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p6*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm)+8*(-ffmcfmb*p2p3*p3p6-2*
     . p2p3*p4p6+p2p3*p5p6))+w11*(4*p5p6*(-2*ffmcfmb*hbcm2-fmc*hbcm)+
     . 4*p4p6*(3*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+hbcm2)+4*p3p6*(2*
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(-ffmcfmb*p2p3
     . *p3p6+ffmcfmb*p3p4*p3p6-2*p2p3*p4p6+p2p3*p5p6+2*p3p4*p4p6-p3p4
     . *p5p6))+(p3p6*(-16*fb2*ffmcfmb-4)+16*(-2*fb2*p4p6+fb2*p5p6)))
      ans2=(-12*fb1*fmb*hbcm+20*fb2*ffmcfmb*hbcm2-12*fb2*hbcm2+8*fb2*
     . p1p3-8*fb2*p2p3+8*fb2*p3p4-24*fb2*p3p5-hbcm2)
      ans1=w2*(6*p3p5*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+2*(5*ffmcfmb*fmb
     . *hbcm3+2*fmb*hbcm*p1p3-2*fmb*hbcm*p2p3+2*fmb*hbcm*p3p4-3*fmb*
     . hbcm3+3*fmb2*hbcm2-2*hbcm2*p1p5+2*hbcm2*p2p5-2*hbcm2*p4p5+2*
     . p1p3*p3p5-2*p2p3*p3p5+2*p3p4*p3p5-6*p3p5s2))+w7*(12*p3p5*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p3p4*(-ffmcfmb*hbcm2+fmc*hbcm)+2*
     . p2p3*(ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm-3*hbcm2)+4*p1p3*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+2*(-2*ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*
     . hbcm3+5*ffmcfmb*fmc*hbcm3+3*ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*
     . fmc*hbcm3-3*fmc2*hbcm2+2*p1p3*p2p3+2*p2p3*p3p4-6*p2p3*p3p5-2*
     . p2p3s2))+w11*(2*p3p4*(ffmcfmb*hbcm2+3*fmb*hbcm+hbcm2)+2*p2p3*(
     . ffmcfmb*hbcm2-3*fmb*hbcm-3*hbcm2)+2*(-2*ffmcfmb**2*hbcm4+3*
     . ffmcfmb*fmb*hbcm3-2*ffmcfmb*hbcm2*p1p3+6*ffmcfmb*hbcm2*p3p5+3*
     . ffmcfmb*hbcm4+3*fmc2*hbcm2-2*hbcm2*p1p4-6*hbcm2*p2p4-2*hbcm2*
     . p4p5+2*p1p3*p2p3-2*p1p3*p3p4+4*p2p3*p3p4-6*p2p3*p3p5-2*p2p3s2+
     . 6*p3p4*p3p5-2*p3p4s2))+ans2
      ans=ccc*ans1
      b(11)=ans
      ans2=(2*p3p7*(4*fb1*ffmcfmb*hbcm-12*fb2*fmb+5*hbcm)+8*(fb1*hbcm
     . *p4p7-3*fb1*hbcm*p5p7))
      ans1=w2*(4*(p3p7*p3p5)*(ffmcfmb*hbcm-3*fmb)+4*p3p7*(-ffmcfmb*
     . fmb*hbcm2-3*fmb2*hbcm)+4*(-fmb*hbcm2*p4p7+3*fmb*hbcm2*p5p7+2*
     . hbcm*p1p5*p3p7-2*hbcm*p2p5*p3p7-hbcm*p3p5*p4p7+3*hbcm*p3p5*
     . p5p7+2*hbcm*p3p7*p4p5))+w7*(12*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)+
     . 4*p4p7*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p3p7*p2p3)*(5*ffmcfmb*
     . hbcm-3*fmb)+4*p3p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-3*
     . ffmcfmb*hbcm3-3*fmb*fmc*hbcm+3*fmc2*hbcm)+4*(2*ffmcfmb*hbcm*
     . p1p3*p3p7+2*ffmcfmb*hbcm*p3p4*p3p7-6*ffmcfmb*hbcm*p3p5*p3p7+
     . hbcm*p2p3*p4p7-3*hbcm*p2p3*p5p7))+w11*(4*p4p7*(4*ffmcfmb*hbcm3
     . -3*fmb*hbcm2-3*hbcm3)+4*(p3p7*p3p4)*(-7*ffmcfmb*hbcm+3*fmb+2*
     . hbcm)+4*(p3p7*p2p3)*(5*ffmcfmb*hbcm-3*fmb)+4*p3p7*(ffmcfmb**2*
     . hbcm3-3*ffmcfmb*hbcm3-3*fmc2*hbcm)+4*(2*ffmcfmb*hbcm*p1p3*p3p7
     . -6*ffmcfmb*hbcm*p3p5*p3p7+3*ffmcfmb*hbcm3*p5p7+2*hbcm*p1p3*
     . p4p7+2*hbcm*p1p4*p3p7-hbcm*p2p3*p4p7-3*hbcm*p2p3*p5p7+6*hbcm*
     . p2p4*p3p7+hbcm*p3p4*p4p7+3*hbcm*p3p4*p5p7-6*hbcm*p3p5*p4p7+2*
     . hbcm*p3p7*p4p5))+ans2
      ans=ccc*ans1
      b(12)=ans
      b(13)=ccc*(w7*(6*p2p3*(ffmcfmb*hbcm-fmb+hbcm)+6*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+
     . fmc2*hbcm))+w2*(2*p3p5*(5*ffmcfmb*hbcm-3*fmb-3*hbcm)+2*(3*
     . ffmcfmb*fmb*hbcm2-3*fmb*hbcm2-3*fmb2*hbcm+2*hbcm*p1p5-2*hbcm*
     . p2p5+2*hbcm*p4p5))+w11*(2*p3p4*(-5*ffmcfmb*hbcm+3*fmb-hbcm)+6*
     . p2p3*(ffmcfmb*hbcm-fmb+hbcm)+2*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*
     . hbcm3-3*fmc2*hbcm+2*hbcm*p1p4+6*hbcm*p2p4+2*hbcm*p4p5))+3*(-4*
     . fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm))
      b(14)=ccc*(w2*((p3p7*p3p5)*(16*ffmcfmb-12)+4*p3p7*(4*ffmcfmb*
     . fmb*hbcm-3*fmb*hbcm)+4*(fmb*hbcm*p4p7-3*fmb*hbcm*p5p7+p3p5*
     . p4p7-3*p3p5*p5p7))+w7*((p3p7*p2p3)*(16*ffmcfmb-12)+12*p5p7*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p4p7*(-ffmcfmb*hbcm2+fmc*hbcm)+4*
     . p3p7*(-ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-
     . 3*fmc*hbcm)+4*(p2p3*p4p7-3*p2p3*p5p7))+w11*((p3p7*p3p4)*(-16*
     . ffmcfmb+12)+(p3p7*p2p3)*(16*ffmcfmb-12)+4*p4p7*(2*ffmcfmb*
     . hbcm2+3*fmb*hbcm-3*hbcm2)+4*p3p7*(-ffmcfmb**2*hbcm2+3*ffmcfmb*
     . fmb*hbcm)+4*(3*ffmcfmb*hbcm2*p5p7+p2p3*p4p7-3*p2p3*p5p7-p3p4*
     . p4p7+3*p3p4*p5p7))+(p3p7*(32*fb2*ffmcfmb-24*fb2+8)+8*(fb2*p4p7
     . -3*fb2*p5p7)))
      ans3=w11*(8*(p6p7*p3p4)*(fmb*hbcm-hbcm2)+(p3p7*p3p6*p3p4)*(16*
     . ffmcfmb-8)+(p3p7*p3p6*p2p3)*(-16*ffmcfmb+8)+8*(p5p6*p3p7)*(-
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*p3p7)*(2*ffmcfmb*hbcm2-fmb*
     . hbcm+fmc*hbcm)+8*(p4p7*p3p6)*(-2*ffmcfmb*hbcm2-2*fmb*hbcm+
     . hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmb*hbcm)+4*p6p7*(-ffmcfmb
     . **2*hbcm4+2*ffmcfmb*fmb*hbcm3+fmc2*hbcm2-2*p2p3s2-2*p3p4s2)+8*
     . (p3p7*p3p6)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm)+8*(ffmcfmb*hbcm2*p1p3*p6p7+
     . ffmcfmb*hbcm2*p3p5*p6p7-2*ffmcfmb*hbcm2*p3p6*p5p7-hbcm2*p1p4*
     . p6p7-hbcm2*p2p4*p6p7-hbcm2*p4p5*p6p7-2*hbcm2*p4p6*p4p7-hbcm2*
     . p4p6*p5p7+hbcm2*p4p7*p5p6-p1p3*p2p3*p6p7+p1p3*p3p4*p6p7+2*p2p3
     . *p3p4*p6p7-p2p3*p3p5*p6p7+2*p2p3*p3p6*p5p7-3*p2p3*p3p7*p4p6+
     . p2p3*p3p7*p5p6+p3p4*p3p5*p6p7-2*p3p4*p3p6*p5p7+3*p3p4*p3p7*
     . p4p6-p3p4*p3p7*p5p6))+((p3p7*p3p6)*(-32*fb2*ffmcfmb+16*fb2-8)+
     . 16*p6p7*(-fb1*fmb*hbcm+fb2*ffmcfmb*hbcm2)+16*(-fb2*p1p3*p6p7-
     . fb2*p2p3*p6p7+fb2*p3p4*p6p7-fb2*p3p5*p6p7+2*fb2*p3p6*p5p7-3*
     . fb2*p3p7*p4p6+fb2*p3p7*p5p6))
      ans2=w7*((p3p7*p3p6*p2p3)*(-16*ffmcfmb+8)+8*(p5p6*p3p7)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*p3p7)*(ffmcfmb*hbcm2-3*fmc*
     . hbcm)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p5)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm
     . )+8*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+8*(p6p7*p1p3
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+4*p6p7*(-ffmcfmb**2*hbcm4+2*ffmcfmb
     . *fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-fmc2*hbcm2-2*
     . p2p3s2)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2
     . *ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+8*(-p1p3*p2p3*p6p7+
     . p2p3*p3p4*p6p7-p2p3*p3p5*p6p7+2*p2p3*p3p6*p5p7-3*p2p3*p3p7*
     . p4p6+p2p3*p3p7*p5p6))+ans3
      ans1=w2*((p3p7*p3p6*p3p5)*(-16*ffmcfmb+8)+8*(p5p7*p3p6)*(-2*
     . ffmcfmb*hbcm2+2*fmb*hbcm+hbcm2)+8*p6p7*(ffmcfmb*fmb*hbcm3+fmb2
     . *hbcm2-p3p5s2)+16*(p3p7*p3p6)*(-2*ffmcfmb*fmb*hbcm+fmb*hbcm)+8
     . *(ffmcfmb*hbcm2*p3p5*p6p7-fmb*hbcm*p1p3*p6p7-fmb*hbcm*p2p3*
     . p6p7+fmb*hbcm*p3p4*p6p7-4*fmb*hbcm*p3p7*p4p6-hbcm2*p4p6*p5p7-
     . hbcm2*p5p6*p5p7-p1p3*p3p5*p6p7-p2p3*p3p5*p6p7+p3p4*p3p5*p6p7+2
     . *p3p5*p3p6*p5p7-3*p3p5*p3p7*p4p6+p3p5*p3p7*p5p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w2*(16*(p5p6*p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p5p7*p3p6)*(
     . -2*ffmcfmb*hbcm+hbcm)+8*(p6p7*p3p5)*(-ffmcfmb*hbcm+fmb+hbcm)+8
     . *p6p7*(-ffmcfmb*fmb*hbcm2+fmb*hbcm2+fmb2*hbcm)+8*(-hbcm*p4p6*
     . p5p7-hbcm*p5p6*p5p7))+w7*(8*(p6p7*p2p3)*(fmb-hbcm)+4*p6p7*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-2*fmc*hbcm2-fmc2*hbcm)+8*(-
     . ffmcfmb*hbcm*p3p6*p3p7-3*ffmcfmb*hbcm*p3p7*p4p6+ffmcfmb*hbcm*
     . p3p7*p5p6))+w11*(8*(p6p7*p2p3)*(fmb-hbcm)+8*(p4p6*p3p7)*(-
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p4)*(ffmcfmb*hbcm-fmb)+4*p6p7*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmc2*hbcm
     . )+8*(-ffmcfmb*hbcm*p3p6*p3p7-2*ffmcfmb*hbcm*p3p6*p4p7+ffmcfmb*
     . hbcm*p3p7*p5p6-hbcm*p1p4*p6p7-hbcm*p2p4*p6p7-hbcm*p4p5*p6p7-3*
     . hbcm*p4p6*p4p7-hbcm*p4p6*p5p7+2*hbcm*p4p7*p5p6))+8*p6p7*(2*fb1
     . *ffmcfmb*hbcm-2*fb1*hbcm+2*fb2*fmb+hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp9_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*
     . p3p4+fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-fmb2+hbcm2-2*p1p3))
      ans2=(32*(p6p7*p2p3)*(-fb2*fmb+fb2*fmc)+8*(p4p7*p3p6)*(-4*fb1*
     . ffmcfmb*hbcm+4*fb1*hbcm+4*fb2*fmc+3*hbcm))
      ans1=w11*(16*(p4p7*p3p6)*(-fmc*hbcm2-fmc2*hbcm)+16*p6p7*(-fmb*
     . p2p3s2+fmc*p2p3s2)+16*(p6p7*p3p4*p2p3)*(fmb-fmc)+16*(p6p7*p2p4
     . )*(-fmb*hbcm2+fmc*hbcm2-2*hbcm3)+16*(p4p7*p4p6)*(ffmcfmb*hbcm3
     . -2*hbcm3)+16*(p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm-fmc-hbcm)+16*(
     . p4p7*p3p6*p2p3)*(ffmcfmb*hbcm+fmc+hbcm)+16*(p6p7*p2p3)*(-
     . ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmc*hbcm2-2*fmc2*hbcm)+16*(-fmc
     . *hbcm2*p1p2*p6p7+fmc*hbcm2*p2p5*p6p7-2*hbcm*p1p2*p3p4*p6p7+2*
     . hbcm*p1p2*p3p7*p4p6+2*hbcm*p1p3*p2p4*p6p7+2*hbcm*p1p3*p4p6*
     . p4p7+2*hbcm*p1p4*p2p3*p6p7+4*hbcm*p2p3*p2p4*p6p7-2*hbcm*p2p3*
     . p2p5*p6p7+4*hbcm*p2p3*p4p6*p4p7+2*hbcm*p2p3*p4p6*p5p7-4*hbcm*
     . p2p4*p3p5*p6p7-4*hbcm*p2p4*p3p6*p3p7-2*hbcm*p2p4*p3p6*p5p7-4*
     . hbcm*p2p4*p3p7*p4p6+2*hbcm*p2p5*p3p4*p6p7-2*hbcm*p2p5*p3p7*
     . p4p6-2*hbcm*p3p4*p4p6*p4p7-2*hbcm*p3p5*p4p6*p4p7))+w13*(16*(
     . p6p7*p2p3*p1p3)*(fmb-fmc)+16*(p4p7*p3p6)*(-fmb*fmc*hbcm+fmb2*
     . hbcm)+16*(p4p7*p3p6*p1p3)*(-ffmcfmb*hbcm-fmc+hbcm)+16*(p6p7*
     . p2p3)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*
     . hbcm2+fmb2*hbcm+fmc*hbcm2))+ans2
      ans=ccc*ans1
      b(1)=ans
      b(2)=ccc*(w11*(8*(p4p7*p3p4)*(-fmb+fmc+2*hbcm)+8*(p4p7*p2p3)*(
     . fmb-fmc-2*hbcm)+8*p4p7*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*
     . hbcm)+16*(hbcm*p1p2*p4p7-hbcm*p1p4*p4p7+2*hbcm*p2p4*p3p7+2*
     . hbcm*p2p4*p5p7))+w13*(8*(p4p7*p1p3)*(-fmb+fmc)+8*p4p7*(-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-
     . fmb2*hbcm-fmc*hbcm2))+16*p4p7*(fb2*fmb-fb2*fmc))
      b(3)=16*ccc*w11*(-hbcm*p1p2*p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*p3p6+
     . 2*hbcm*p2p4*p4p6+hbcm*p2p5*p4p6)
      b(4)=ccc*(w13*((p4p7*p3p6*p1p3)*(-16*ffmcfmb+16)+16*(p4p7*p4p6)
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p3p6)*(ffmcfmb*hbcm2-
     . hbcm2)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p1p3*
     . p2p4*p6p7+p1p3*p4p6*p4p7))+w11*((p4p7*p3p6*p3p4)*(-16*ffmcfmb+
     . 16)+(p4p7*p3p6*p2p3)*(16*ffmcfmb-16)+16*(p4p7*p4p6)*(ffmcfmb*
     . hbcm2+fmc*hbcm)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmb*hbcm)+16*(
     . fmc*hbcm*p3p6*p4p7-p2p3*p2p4*p6p7-p2p3*p4p6*p4p7+p2p4*p3p4*
     . p6p7+p3p4*p4p6*p4p7))+((p4p7*p3p6)*(32*fb2*ffmcfmb-32*fb2+8)+
     . 32*(-fb2*p2p4*p6p7-fb2*p4p6*p4p7)))
      ans2=(4*p4p7*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+hbcm2)+32*(
     . fb2*p1p2*p3p7+fb2*p2p3*p4p7+fb2*p2p3*p5p7-fb2*p2p5*p3p7-fb2*
     . p3p5*p4p7))
      ans1=w11*(8*p4p7*(fmb*fmc*hbcm2+2*fmc*hbcm3+fmc2*hbcm2+2*p2p3s2
     . )+8*(p4p7*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+2*hbcm2)+
     . 16*(p3p7*p2p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+hbcm2)+8*(
     . p4p7*p2p3)*(ffmcfmb*hbcm2-fmb*hbcm-6*fmc*hbcm-2*hbcm2)+16*(-2*
     . fmc*hbcm*p1p2*p3p7-fmc*hbcm*p1p3*p4p7-2*fmc*hbcm*p2p3*p5p7+2*
     . fmc*hbcm*p2p5*p3p7+2*fmc*hbcm*p3p5*p4p7+hbcm2*p1p2*p4p7-hbcm2*
     . p1p4*p4p7+hbcm2*p2p4*p5p7+p1p2*p2p3*p3p7-p1p2*p3p4*p3p7-p2p3*
     . p2p5*p3p7-p2p3*p3p4*p4p7-p2p3*p3p4*p5p7-p2p3*p3p5*p4p7+p2p3s2*
     . p5p7+p2p5*p3p4*p3p7+p3p4*p3p5*p4p7))+w13*(8*p4p7*(ffmcfmb*
     . hbcm4+fmb*hbcm3-hbcm4)+16*(p4p7*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+16*(p3p7*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p5p7*
     . p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p7*p2p3)*(-ffmcfmb
     . *hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . 2*hbcm2)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(-
     . p1p2*p1p3*p3p7-p1p3*p2p3*p4p7-p1p3*p2p3*p5p7+p1p3*p2p5*p3p7+
     . p1p3*p3p5*p4p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w11*(16*p3p6*(-ffmcfmb*p2p3s2+p2p3s2)+(p3p6*p3p4*p2p3
     . )*(16*ffmcfmb-16)+16*(p3p6*p2p4)*(-ffmcfmb*hbcm2+fmb*hbcm+fmc*
     . hbcm+hbcm2)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+16*
     . (-fmc*hbcm*p1p2*p3p6+fmc*hbcm*p2p5*p3p6-hbcm2*p1p2*p4p6+hbcm2*
     . p2p4*p4p6+hbcm2*p2p5*p4p6-p2p3*p3p4*p4p6+p2p3s2*p4p6))+w13*((
     . p3p6*p2p3*p1p3)*(16*ffmcfmb-16)+16*(p3p6*p2p5)*(ffmcfmb*hbcm2-
     . hbcm2)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p3p6
     . *p1p2)*(-ffmcfmb*hbcm2+hbcm2)-16*p1p3*p2p3*p4p6)+((p3p6*p2p3)*
     . (-32*fb2*ffmcfmb+32*fb2-8)+32*fb2*p2p3*p4p6))
      b(7)=ccc*(w11*(8*p2p4*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(2*
     . fmc*hbcm*p1p2-2*fmc*hbcm*p2p3-2*fmc*hbcm*p2p5-p1p2*p2p3+p1p2*
     . p3p4+p2p3*p2p5-p2p3*p3p4+p2p3s2-p2p5*p3p4))+w13*(8*p2p5*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p2p3*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+8*p1p2*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p1p2*p1p3-p1p3
     . *p2p3-p1p3*p2p5))+16*(-fb2*p1p2+fb2*p2p3+fb2*p2p5))
      b(8)=ccc*(w11*(8*(p3p4*p2p3)*(-fmb+fmc)+8*p2p4*(fmb*hbcm2-fmc*
     . hbcm2+2*hbcm3)+8*p2p3*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmc*
     . hbcm2+2*fmc2*hbcm)+8*(fmb*p2p3s2+fmc*hbcm2*p1p2-fmc*hbcm2*p2p5
     . -fmc*p2p3s2+2*hbcm*p1p2*p3p4-2*hbcm*p1p3*p2p4-2*hbcm*p1p4*p2p3
     . -4*hbcm*p2p3*p2p4+2*hbcm*p2p3*p2p5+4*hbcm*p2p4*p3p5-2*hbcm*
     . p2p5*p3p4))+w13*(8*(p2p3*p1p3)*(-fmb+fmc)+8*p2p3*(-ffmcfmb*fmb
     . *hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*
     . hbcm2))+16*p2p3*(fb2*fmb-fb2*fmc))
      b(9)=ccc*(w11*(8*p4p6*(ffmcfmb*hbcm3-2*hbcm3)+8*(p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*(p3p6*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8
     . *p3p6*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmc*hbcm2+fmc2*hbcm)+16*
     . (hbcm*p1p2*p3p6+hbcm*p1p3*p4p6-hbcm*p1p4*p3p6+hbcm*p2p3*p4p6-
     . hbcm*p2p4*p3p6-hbcm*p3p4*p4p6-hbcm*p3p5*p4p6))+w13*(8*(p3p6*
     . p1p3)*(-ffmcfmb*hbcm-fmb+hbcm)+8*p3p6*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+fmb*hbcm2-fmc*hbcm2))+4*p3p6*(-4*fb1*ffmcfmb
     . *hbcm+4*fb1*hbcm+4*fb2*fmb+3*hbcm))
      b(10)=ccc*(w11*((p3p6*p3p4)*(-8*ffmcfmb+8)+(p3p6*p2p3)*(8*
     . ffmcfmb-8)+8*p4p6*(ffmcfmb*hbcm2+fmc*hbcm)+8*(fmc*hbcm*p3p6-
     . p2p3*p4p6+p3p4*p4p6))+w13*((p3p6*p1p3)*(-8*ffmcfmb+8)+8*p4p6*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p6*(ffmcfmb*hbcm2-hbcm2)+8*
     . p1p3*p4p6)+(p3p6*(16*fb2*ffmcfmb-16*fb2+4)-16*fb2*p4p6))
      b(11)=ccc*(w11*(4*p3p4*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+2*
     . hbcm2)+4*p2p3*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm-2*hbcm2)+4*(
     . fmb*fmc*hbcm2-2*fmc*hbcm*p1p3+4*fmc*hbcm*p3p5+2*fmc*hbcm3+fmc2
     . *hbcm2+2*hbcm2*p1p2-2*hbcm2*p1p4-2*hbcm2*p2p4-2*p2p3*p3p5+2*
     . p3p4*p3p5))+w13*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p1p3*
     . (-ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2)+4*(ffmcfmb*hbcm4+fmb*hbcm3-
     . hbcm4+2*p1p3*p3p5))+2*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2-8*
     . fb2*p3p5+hbcm2))
      b(12)=ccc*(w11*(8*(p3p7*p3p4)*(fmb-fmc)+8*(p3p7*p2p3)*(-fmb+fmc
     . )+8*p3p7*(-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmc*hbcm2-2*fmc2*
     . hbcm)+8*(fmc*hbcm2*p4p7+fmc*hbcm2*p5p7-2*hbcm*p1p2*p3p7+2*hbcm
     . *p1p4*p3p7-2*hbcm*p2p3*p4p7-2*hbcm*p2p3*p5p7+4*hbcm*p2p4*p3p7+
     . 2*hbcm*p3p4*p4p7+2*hbcm*p3p4*p5p7))+w13*(8*(p3p7*p1p3)*(fmb-
     . fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-
     . fmb*hbcm2+fmb2*hbcm+fmc*hbcm2))+16*p3p7*(-fb2*fmb+fb2*fmc))
      b(13)=ccc*(w11*(4*p3p4*(fmb-fmc-2*hbcm)+4*p2p3*(-fmb+fmc+2*hbcm
     . )+4*(-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm-2*hbcm*p1p2+2
     . *hbcm*p1p4+4*hbcm*p2p4))+w13*(4*p1p3*(fmb-fmc)+4*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*
     . hbcm2))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(8*w11*(2*fmc*hbcm*p3p7+2*fmc*hbcm*p4p7+2*fmc*hbcm*
     . p5p7-p2p3*p3p7-p2p3*p4p7-p2p3*p5p7+p3p4*p3p7+p3p4*p4p7+p3p4*
     . p5p7)+w13*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p4p7*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p3p7*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*(p1p3*p3p7+p1p3*p4p7+p1p3*p5p7))+16*(-fb2*p3p7-fb2*
     . p4p7-fb2*p5p7))
      ans2=((p3p7*p3p6)*(32*fb2*ffmcfmb-32*fb2+8)+4*p6p7*(-4*fb1*fmb*
     . hbcm-4*fb2*ffmcfmb*hbcm2+8*fb2*hbcm2+hbcm2)+32*(-fb2*p1p3*p6p7
     . -fb2*p2p3*p6p7+fb2*p3p4*p6p7-fb2*p3p7*p4p6))
      ans1=w11*(8*p6p7*(fmb*fmc*hbcm2-2*fmc*hbcm3+fmc2*hbcm2-2*p2p3s2
     . -2*p3p4s2)+(p3p7*p3p6*p3p4)*(-16*ffmcfmb+16)+(p3p7*p3p6*p2p3)*
     . (16*ffmcfmb-16)+16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+
     . 8*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm-2*fmc*hbcm)+8*(p6p7*
     . p2p3)*(ffmcfmb*hbcm2-fmb*hbcm+2*fmc*hbcm)+16*(fmc*hbcm*p1p3*
     . p6p7-fmc*hbcm*p3p6*p4p7-fmc*hbcm*p3p6*p5p7+hbcm2*p1p2*p6p7-
     . hbcm2*p1p4*p6p7-hbcm2*p2p4*p6p7-hbcm2*p4p6*p4p7-hbcm2*p4p6*
     . p5p7-p1p3*p2p3*p6p7+p1p3*p3p4*p6p7+2*p2p3*p3p4*p6p7-p2p3*p3p7*
     . p4p6+p3p4*p3p7*p4p6))+w13*((p3p7*p3p6*p1p3)*(-16*ffmcfmb+16)+8
     . *p6p7*(-ffmcfmb*hbcm4-fmb*hbcm3+hbcm4+2*p1p3s2)+16*(p4p6*p3p7)
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2
     . +hbcm2)+16*(p4p7*p3p6)*(-ffmcfmb*hbcm2+hbcm2)+16*(p6p7*p3p4)*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p6p7*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)
     . +16*(p1p3*p2p3*p6p7-p1p3*p3p4*p6p7+p1p3*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w11*(8*(p6p7*p3p4)*(-fmb+fmc+2*hbcm)+8*(p6p7*p2p3)*(
     . fmb-fmc-2*hbcm)+8*p6p7*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*
     . hbcm)+16*(hbcm*p1p2*p6p7-hbcm*p1p4*p6p7-2*hbcm*p2p4*p6p7-hbcm*
     . p3p7*p4p6-hbcm*p4p6*p4p7-hbcm*p4p6*p5p7))+w13*(8*(p6p7*p1p3)*(
     . -fmb+fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc
     . *hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2))+16*p6p7*(fb2*fmb-fb2*fmc
     . ))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp8_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)
     . )
      ans=ccc*(w2*(16*(p6p7*p3p5*p2p3)*(-fmb+fmc)+16*(p5p6*p4p7)*(
     . ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+16*(p4p7*p3p6*p3p5)*(-ffmcfmb*
     . hbcm+fmc-2*hbcm)+16*(p4p7*p3p6)*(ffmcfmb*fmb*hbcm2+fmb*fmc*
     . hbcm-fmb*hbcm2-fmb2*hbcm)+16*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2+
     . fmb*fmc*hbcm-2*fmb2*hbcm)+16*(-fmb*hbcm2*p1p2*p6p7+fmb*hbcm2*
     . p2p5*p6p7-fmb*hbcm2*p4p6*p4p7-2*hbcm*p1p2*p3p5*p6p7+2*hbcm*
     . p1p2*p3p7*p5p6+2*hbcm*p1p5*p2p3*p6p7+2*hbcm*p1p5*p3p6*p4p7+2*
     . hbcm*p2p3*p4p7*p5p6+2*hbcm*p2p3*p5p6*p5p7+2*hbcm*p2p5*p3p5*
     . p6p7-2*hbcm*p2p5*p3p7*p5p6-2*hbcm*p3p5*p4p6*p4p7-2*hbcm*p3p5*
     . p4p7*p5p6))+w12*(16*(p6p7*p2p3*p1p3)*(-fmb+fmc)+16*(p4p7*p3p6*
     . p1p3)*(-ffmcfmb*hbcm+fmc)+16*(p4p7*p3p6)*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2)+16*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+(32*(p6p7*p2p3)*(-fb2*fmb+
     . fb2*fmc)+8*(p4p7*p3p6)*(-4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm))
     . )
      b(1)=ans
      b(2)=ccc*(w2*(8*(p4p7*p3p5)*(fmb-fmc+2*hbcm)+8*p4p7*(-ffmcfmb*
     . fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)-16*hbcm*p1p5*
     . p4p7)+w12*(8*(p4p7*p1p3)*(fmb-fmc)+8*p4p7*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm))+16*p4p7*(fb2*fmb-
     . fb2*fmc))
      b(3)=16*ccc*w2*(-hbcm*p1p2*p5p6+hbcm*p2p3*p5p6+hbcm*p2p5*p5p6)
      b(4)=ccc*(w2*((p4p7*p3p6*p3p5)*(16*ffmcfmb-16)+16*(p5p6*p4p7)*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(-2*fmb*hbcm*p2p4*p6p7-fmb*
     . hbcm*p3p6*p4p7-2*fmb*hbcm*p4p6*p4p7-p2p4*p3p5*p6p7-p3p5*p4p6*
     . p4p7))+w12*((p4p7*p3p6*p1p3)*(16*ffmcfmb-16)+16*(p4p7*p4p6)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm
     . )+16*(-fmc*hbcm*p3p6*p4p7-p1p3*p2p4*p6p7-p1p3*p4p6*p4p7))+((
     . p4p7*p3p6)*(32*fb2*ffmcfmb-32*fb2+8)+32*(-fb2*p2p4*p6p7-fb2*
     . p4p6*p4p7)))
      ans=ccc*(w2*(8*p4p7*(fmb*fmc*hbcm2-2*fmb*hbcm3+fmb2*hbcm2-2*
     . p3p5s2)+8*(p4p7*p3p5)*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-hbcm2
     . )+16*(2*fmb*hbcm*p1p2*p3p7+fmb*hbcm*p1p3*p4p7+2*fmb*hbcm*p2p3*
     . p4p7+2*fmb*hbcm*p2p3*p5p7-2*fmb*hbcm*p2p5*p3p7-hbcm2*p1p5*p4p7
     . +p1p2*p3p5*p3p7+p1p3*p3p5*p4p7+p2p3*p3p5*p4p7+p2p3*p3p5*p5p7-
     . p2p5*p3p5*p3p7))+w12*(8*p4p7*(ffmcfmb*hbcm4-fmc*hbcm3+2*p1p3s2
     . )+16*(p4p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p3p7*p2p5)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*p2p3)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p4p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7*p1p3)*(
     . -ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+16*(p3p7*p1p2)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p1p2*p1p3*p3p7+p1p3*p2p3*p4p7+p1p3*p2p3*p5p7-
     . p1p3*p2p5*p3p7-p1p3*p3p5*p4p7))+(4*p4p7*(-4*fb1*fmc*hbcm-4*fb2
     . *ffmcfmb*hbcm2-4*fb2*hbcm2+hbcm2)+32*(fb2*p1p2*p3p7+fb2*p1p3*
     . p4p7+fb2*p2p3*p4p7+fb2*p2p3*p5p7-fb2*p2p5*p3p7-fb2*p3p5*p4p7))
     . )
      b(5)=ans
      b(6)=ccc*(w2*(16*(p5p6*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm)+16*(-
     . ffmcfmb*p2p3*p3p5*p3p6+fmb*hbcm*p1p2*p3p6+2*fmb*hbcm*p2p3*p4p6
     . -fmb*hbcm*p2p5*p3p6-hbcm2*p1p2*p5p6+hbcm2*p2p5*p5p6+p1p2*p3p5*
     . p3p6+p2p3*p3p5*p4p6-p2p5*p3p5*p3p6))+w12*(16*(p4p6*p2p3)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(-ffmcfmb*p1p3*p2p3*p3p6+fmc*hbcm*
     . p1p2*p3p6-fmc*hbcm*p2p5*p3p6+p1p2*p1p3*p3p6+p1p3*p2p3*p4p6-
     . p1p3*p2p5*p3p6))+((p3p6*p2p3)*(-32*fb2*ffmcfmb-8)+32*(fb2*p1p2
     . *p3p6+fb2*p2p3*p4p6-fb2*p2p5*p3p6)))
      b(7)=ccc*(8*w2*(-2*fmb*hbcm*p1p2+2*fmb*hbcm*p2p3+2*fmb*hbcm*
     . p2p5-p1p2*p3p5+p2p3*p3p5+p2p5*p3p5)+w12*(8*p2p5*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*p2p3*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p1p2*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(-p1p2*p1p3+p1p3*p2p3+p1p3*p2p5))+16
     . *(-fb2*p1p2+fb2*p2p3+fb2*p2p5))
      b(8)=ccc*(w2*(8*(p3p5*p2p3)*(fmb-fmc)+8*p2p3*(-ffmcfmb*fmb*
     . hbcm2-fmb*fmc*hbcm+2*fmb2*hbcm)+8*(fmb*hbcm2*p1p2-fmb*hbcm2*
     . p2p5+2*hbcm*p1p2*p3p5-2*hbcm*p1p5*p2p3-2*hbcm*p2p5*p3p5))+w12*
     . (8*(p2p3*p1p3)*(fmb-fmc)+8*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm))+16*p2p3*(fb2*fmb-fb2*fmc))
      b(9)=ccc*(w2*(8*p5p6*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p3p6*
     . p3p5)*(-ffmcfmb*hbcm+fmb)+8*(-fmb*hbcm2*p4p6+fmb2*hbcm*p3p6-2*
     . hbcm*p3p5*p4p6-2*hbcm*p3p5*p5p6))+w12*(8*p3p6*(fmb*fmc*hbcm-
     . fmc2*hbcm)+8*(p3p6*p1p3)*(-ffmcfmb*hbcm+fmb))+4*p3p6*(-4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmb-3*hbcm))
      b(10)=ccc*(w2*((p3p6*p3p5)*(8*ffmcfmb-8)+8*p5p6*(-ffmcfmb*hbcm2
     . -fmb*hbcm+hbcm2)+8*(-fmb*hbcm*p3p6-2*fmb*hbcm*p4p6-p3p5*p4p6))
     . +w12*((p3p6*p1p3)*(8*ffmcfmb-8)+8*p4p6*(ffmcfmb*hbcm2-fmc*hbcm
     . )+8*(-fmc*hbcm*p3p6-p1p3*p4p6))+(p3p6*(16*fb2*ffmcfmb-16*fb2+4
     . )-16*fb2*p4p6))
      b(11)=ccc*(w2*(4*p3p5*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-hbcm2)
     . +4*(fmb*fmc*hbcm2+2*fmb*hbcm*p1p3-2*fmb*hbcm3+fmb2*hbcm2-2*
     . hbcm2*p1p5+2*p1p3*p3p5-2*p3p5s2))+w12*(8*p3p5*(ffmcfmb*hbcm2-
     . fmc*hbcm)+4*p1p3*(-ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+4*(ffmcfmb*
     . hbcm4-fmc*hbcm3-2*p1p3*p3p5+2*p1p3s2))+2*(-4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2-4*fb2*hbcm2+8*fb2*p1p3-8*fb2*p3p5+hbcm2))
      b(12)=ccc*(w2*(8*(p3p7*p3p5)*(-fmb+fmc)+8*p3p7*(ffmcfmb*fmb*
     . hbcm2+fmb*fmc*hbcm-2*fmb2*hbcm)+8*(fmb*hbcm2*p4p7+fmb*hbcm2*
     . p5p7+2*hbcm*p1p5*p3p7+2*hbcm*p3p5*p4p7+2*hbcm*p3p5*p5p7))+w12*
     . (8*(p3p7*p1p3)*(-fmb+fmc)+8*p3p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+16*p3p7*(-fb2*fmb+fb2*fmc))
      b(13)=ccc*(w2*(4*p3p5*(-fmb+fmc-2*hbcm)+4*(ffmcfmb*fmb*hbcm2+
     . fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm+2*hbcm*p1p5))+w12*(4*p1p3*(
     . -fmb+fmc)+4*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+
     . fmc2*hbcm))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(8*w2*(-2*fmb*hbcm*p3p7-2*fmb*hbcm*p4p7-2*fmb*hbcm*
     . p5p7-p3p5*p3p7-p3p5*p4p7-p3p5*p5p7)+w12*(8*p5p7*(ffmcfmb*hbcm2
     . -fmc*hbcm)+8*p4p7*(ffmcfmb*hbcm2-fmc*hbcm)+8*p3p7*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(-p1p3*p3p7-p1p3*p4p7-p1p3*p5p7))+16*(-fb2*
     . p3p7-fb2*p4p7-fb2*p5p7))
      ans=ccc*(w2*(8*p6p7*(fmb*fmc*hbcm2+2*fmb*hbcm3+fmb2*hbcm2)+16*(
     . p5p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*
     . hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+16*(ffmcfmb*p3p5*p3p6*p3p7-
     . fmb*hbcm*p1p3*p6p7-2*fmb*hbcm*p2p3*p6p7+2*fmb*hbcm*p3p4*p6p7+
     . fmb*hbcm*p3p6*p4p7+fmb*hbcm*p3p6*p5p7-2*fmb*hbcm*p3p7*p4p6-
     . hbcm2*p1p5*p6p7-hbcm2*p4p7*p5p6-hbcm2*p5p6*p5p7-p2p3*p3p5*p6p7
     . +p3p4*p3p5*p6p7+p3p5*p3p6*p4p7+p3p5*p3p6*p5p7-p3p5*p3p7*p4p6))
     . +w12*(8*p6p7*(-ffmcfmb*hbcm4+fmc*hbcm3)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p3p4)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+16*(p6p7*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+16*(ffmcfmb*p1p3*p3p6*p3p7+fmc*
     . hbcm*p3p6*p4p7+fmc*hbcm*p3p6*p5p7-p1p3*p2p3*p6p7+p1p3*p3p4*
     . p6p7+p1p3*p3p6*p4p7+p1p3*p3p6*p5p7-p1p3*p3p7*p4p6))+((p3p7*
     . p3p6)*(32*fb2*ffmcfmb+8)+4*p6p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb
     . *hbcm2+4*fb2*hbcm2+hbcm2)+32*(-fb2*p2p3*p6p7+fb2*p3p4*p6p7+fb2
     . *p3p6*p4p7+fb2*p3p6*p5p7-fb2*p3p7*p4p6)))
      b(15)=ans
      b(16)=ccc*(w2*(8*(p6p7*p3p5)*(fmb-fmc+2*hbcm)+8*p6p7*(-ffmcfmb*
     . fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)+16*(-hbcm*p1p5*
     . p6p7-hbcm*p3p7*p5p6-hbcm*p4p7*p5p6-hbcm*p5p6*p5p7))+w12*(8*(
     . p6p7*p1p3)*(fmb-fmc)+8*p6p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+fmb*fmc*hbcm-fmc2*hbcm))+16*p6p7*(fb2*fmb-fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp7_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(1)=ccc*(w2*(16*(p4p7*p3p6*p3p5)*(fmb-fmc)+16*(p6p7*p3p5*p2p3)
     . *(fmb-fmc)+16*(fmb*hbcm2*p2p3*p6p7+fmb*hbcm2*p3p6*p4p7))+(32*(
     . p4p7*p3p6)*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)+32*(p6p7*p2p3)*(-2*
     . fb1*hbcm+fb2*fmb-fb2*fmc)))
      b(2)=ccc*(w2*(8*(p4p7*p3p5)*(-fmb+fmc)-8*fmb*hbcm2*p4p7)+16*
     . p4p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      b(5)=ccc*(w2*(8*p4p7*(-fmb*fmc*hbcm2-fmb2*hbcm2+2*p3p5s2)+8*
     . hbcm2*p3p5*p4p7)+(16*p4p7*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*
     . hbcm2)+32*fb2*p3p5*p4p7))
      b(8)=ccc*(w2*(8*(p3p5*p2p3)*(-fmb+fmc)-8*fmb*hbcm2*p2p3)+16*
     . p2p3*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      b(11)=ccc*(4*w2*(-fmb*fmc*hbcm2-fmb2*hbcm2+hbcm2*p3p5+2*p3p5s2)
     . +8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*hbcm2+2*fb2*p3p5))
      b(12)=ccc*(w2*(8*(p3p7*p3p5)*(fmb-fmc)+8*fmb*hbcm2*p3p7)+16*
     . p3p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(13)=ccc*(w2*(4*p3p5*(fmb-fmc)+4*fmb*hbcm2)+8*(-2*fb1*hbcm+fb2
     . *fmb-fb2*fmc))
      b(15)=ccc*(w2*(8*p6p7*(-fmb*fmc*hbcm2-fmb2*hbcm2+2*p3p5s2)+8*
     . hbcm2*p3p5*p6p7)+(16*p6p7*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*
     . hbcm2)+32*fb2*p3p5*p6p7))
      b(16)=ccc*(w2*(8*(p6p7*p3p5)*(-fmb+fmc)-8*fmb*hbcm2*p6p7)+16*
     . p6p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp6_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p5*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      b(1)=ccc*(w11*(16*p6p7*(fmb*p2p3s2-fmc*p2p3s2)+16*(p6p7*p3p4*
     . p2p3)*(-fmb+fmc)+16*(p6p7*p2p4)*(fmb*hbcm2-fmc*hbcm2)+16*(fmc*
     . hbcm2*p1p2*p6p7-fmc*hbcm2*p2p5*p6p7+fmc*hbcm2*p3p6*p4p7+fmc*
     . hbcm2*p4p6*p4p7+fmc*hbcm2*p4p7*p5p6))+(32*(p6p7*p2p3)*(fb2*fmb
     . -fb2*fmc)+64*(-fb1*hbcm*p1p2*p6p7+fb1*hbcm*p2p5*p6p7-fb1*hbcm*
     . p3p6*p4p7-fb1*hbcm*p4p6*p4p7-fb1*hbcm*p4p7*p5p6)))
      b(2)=ccc*(w11*(8*(p4p7*p3p4)*(fmb-fmc)+8*(p4p7*p2p3)*(-fmb+fmc)
     . -8*fmc*hbcm2*p4p7)+16*p4p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      b(4)=ccc*(16*w11*(-hbcm2*p2p4*p6p7+p2p3*p2p4*p6p7+p2p3*p3p6*
     . p4p7+p2p3*p4p6*p4p7+p2p3*p4p7*p5p6-p2p4*p3p4*p6p7-p3p4*p3p6*
     . p4p7-p3p4*p4p6*p4p7-p3p4*p4p7*p5p6)+32*(fb2*p2p4*p6p7+fb2*p3p6
     . *p4p7+fb2*p4p6*p4p7+fb2*p4p7*p5p6))
      b(5)=ccc*(w11*(8*p4p7*(-fmb*fmc*hbcm2-fmc2*hbcm2-2*p2p3s2)+8*(
     . hbcm2*p2p3*p4p7-2*hbcm2*p2p4*p5p7-hbcm2*p3p4*p4p7-2*p1p2*p2p3*
     . p3p7+2*p1p2*p3p4*p3p7-2*p1p3*p2p3*p4p7+2*p1p3*p3p4*p4p7+2*p2p3
     . *p2p5*p3p7+2*p2p3*p3p4*p4p7+2*p2p3*p3p4*p5p7+2*p2p3*p3p5*p4p7-
     . 2*p2p3s2*p5p7-2*p2p5*p3p4*p3p7-2*p3p4*p3p5*p4p7))+(16*p4p7*(2*
     . fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*hbcm2)+32*(-fb2*p1p2*p3p7-fb2*
     . p1p3*p4p7-fb2*p2p3*p4p7-fb2*p2p3*p5p7+fb2*p2p5*p3p7+fb2*p3p5*
     . p4p7)))
      b(6)=ccc*(16*w11*(-hbcm2*p2p4*p4p6-hbcm2*p2p4*p5p6-p1p2*p2p3*
     . p3p6+p1p2*p3p4*p3p6+p2p3*p2p5*p3p6+p2p3*p3p4*p4p6+p2p3*p3p4*
     . p5p6-p2p3s2*p4p6-p2p3s2*p5p6-p2p5*p3p4*p3p6)+32*(-fb2*p1p2*
     . p3p6-fb2*p2p3*p4p6-fb2*p2p3*p5p6+fb2*p2p5*p3p6))
      b(7)=ccc*(8*w11*(-hbcm2*p2p4+p1p2*p2p3-p1p2*p3p4-p2p3*p2p5+p2p3
     . *p3p4-p2p3s2+p2p5*p3p4)+16*(fb2*p1p2-fb2*p2p3-fb2*p2p5))
      b(8)=ccc*(w11*(8*(p3p4*p2p3)*(fmb-fmc)+8*p2p4*(-fmb*hbcm2+fmc*
     . hbcm2)+8*(-fmb*p2p3s2-fmc*hbcm2*p1p2+fmc*hbcm2*p2p5+fmc*p2p3s2
     . ))+(16*p2p3*(-fb2*fmb+fb2*fmc)+32*(fb1*hbcm*p1p2-fb1*hbcm*p2p5
     . )))
      b(9)=ccc*(w11*(8*(p3p6*p3p4)*(fmb-fmc)+8*(p3p6*p2p3)*(-fmb+fmc)
     . +8*(fmc*hbcm2*p4p6+fmc*hbcm2*p5p6))+(16*p3p6*(-fb2*fmb+fb2*fmc
     . )+32*(-fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(10)=ccc*(8*w11*(p2p3*p3p6+p2p3*p4p6+p2p3*p5p6-p3p4*p3p6-p3p4*
     . p4p6-p3p4*p5p6)+16*(fb2*p3p6+fb2*p4p6+fb2*p5p6))
      b(11)=ccc*(4*w11*(-fmb*fmc*hbcm2-fmc2*hbcm2+hbcm2*p2p3+2*hbcm2*
     . p2p4-hbcm2*p3p4-2*p1p3*p2p3+2*p1p3*p3p4+2*p2p3*p3p5-2*p3p4*
     . p3p5)+8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*hbcm2-2*fb2*p1p3+2*
     . fb2*p3p5))
      b(12)=ccc*(w11*(8*(p3p7*p3p4)*(-fmb+fmc)+8*(p3p7*p2p3)*(fmb-fmc
     . )+8*(-fmc*hbcm2*p4p7-fmc*hbcm2*p5p7))+(16*p3p7*(fb2*fmb-fb2*
     . fmc)+32*(fb1*hbcm*p4p7+fb1*hbcm*p5p7)))
      b(13)=ccc*(w11*(4*p3p4*(-fmb+fmc)+4*p2p3*(fmb-fmc)+4*fmc*hbcm2)
     . +8*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(14)=ccc*(8*w11*(p2p3*p3p7+p2p3*p4p7+p2p3*p5p7-p3p4*p3p7-p3p4*
     . p4p7-p3p4*p5p7)+16*(fb2*p3p7+fb2*p4p7+fb2*p5p7))
      b(15)=ccc*(w11*(8*p6p7*(-fmb*fmc*hbcm2-fmc2*hbcm2+2*p2p3s2+2*
     . p3p4s2)+8*(-hbcm2*p2p3*p6p7+2*hbcm2*p2p4*p6p7+hbcm2*p3p4*p6p7-
     . 4*p2p3*p3p4*p6p7-2*p2p3*p3p6*p4p7-2*p2p3*p3p6*p5p7+2*p2p3*p3p7
     . *p4p6+2*p2p3*p3p7*p5p6+2*p3p4*p3p6*p4p7+2*p3p4*p3p6*p5p7-2*
     . p3p4*p3p7*p4p6-2*p3p4*p3p7*p5p6))+(16*p6p7*(2*fb1*fmb*hbcm+2*
     . fb1*fmc*hbcm-fb2*hbcm2)+32*(fb2*p2p3*p6p7-fb2*p3p4*p6p7-fb2*
     . p3p6*p4p7-fb2*p3p6*p5p7+fb2*p3p7*p4p6+fb2*p3p7*p5p6)))
      b(16)=ccc*(w11*(8*(p6p7*p3p4)*(fmb-fmc)+8*(p6p7*p2p3)*(-fmb+fmc
     . )-8*fmc*hbcm2*p6p7)+16*p6p7*(2*fb1*hbcm-fb2*fmb+fb2*fmc))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp5_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*
     . p3p4+fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=ccc*(16*w8*(p3p7*p3p6*p2p3)*(ffmcfmb*fmb-ffmcfmb*fmc+
     . ffmcfmb*hbcm+fmc-hbcm)+w5*(16*(p5p7*p3p6)*(ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-fmc*hbcm2)+16*(p3p7*p3p6*p3p4)*(ffmcfmb*fmb-
     . ffmcfmb*fmc+ffmcfmb*hbcm+fmc-hbcm)+16*(p3p7*p3p6*p1p3)*(-
     . ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-fmc+hbcm)+16*(p4p7*p3p6)*
     . (-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmb2*hbcm-fmc*hbcm2+hbcm3)+16*(fmb*fmc*hbcm*p3p6*p3p7+2*
     . hbcm*p1p2*p3p6*p5p7-2*hbcm*p2p4*p3p6*p5p7+2*hbcm*p2p5*p3p6*
     . p4p7))+8*(p3p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*
     . ffmcfmb*fmb+4*fb2*ffmcfmb*fmc-4*fb2*fmc-fmb+fmc-3*hbcm))
      b(2)=ccc*(8*w8*(p3p7*p2p3)*(-ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*
     . hbcm-fmc+hbcm)+w5*(8*p5p7*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc
     . *hbcm2)+8*(p3p7*p3p4)*(-ffmcfmb*fmb+ffmcfmb*fmc-ffmcfmb*hbcm-
     . fmc+hbcm)+8*(p3p7*p1p3)*(ffmcfmb*fmb-ffmcfmb*fmc+ffmcfmb*hbcm+
     . fmc-hbcm)+8*p4p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+8*(-fmb*fmc*hbcm
     . *p3p7-2*hbcm*p1p2*p5p7+2*hbcm*p2p4*p5p7-2*hbcm*p2p5*p4p7))+4*
     . p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm+4*fb2*ffmcfmb*fmb-4*fb2*
     . ffmcfmb*fmc+4*fb2*fmc+fmb-fmc+3*hbcm))
      ans3=((p3p7*p3p5)*(-32*fb2*ffmcfmb+32*fb2-8)+(p3p7*p2p3)*(32*
     . fb2*ffmcfmb-32*fb2+8)+4*p5p7*(4*fb1*fmb*hbcm+4*fb2*ffmcfmb*
     . hbcm2-4*fb2*hbcm2-hbcm2)+4*p3p7*(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1
     . *ffmcfmb*fmc*hbcm+4*fb1*fmb*hbcm+4*fb1*fmc*hbcm-4*fb2*ffmcfmb*
     . hbcm2+4*fb2*fmb*fmc+4*fb2*fmb2+4*fb2*hbcm2+3*fmb*hbcm+3*fmc*
     . hbcm-hbcm2)+32*(fb2*p2p3*p5p7-fb2*p2p5*p3p7))
      ans2=w5*(8*p3p7*(-fmb*fmc2*hbcm-fmb2*fmc*hbcm)+(p3p7*p3p5*p3p4)
     . *(16*ffmcfmb-16)+(p3p7*p3p4*p2p3)*(-16*ffmcfmb+16)+(p3p7*p3p5*
     . p1p3)*(-16*ffmcfmb+16)+(p3p7*p2p3*p1p3)*(16*ffmcfmb-16)+16*(
     . p4p7*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p7*p3p4)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p5p7*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+
     . 8*p4p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*
     . fmc*hbcm2-2*fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3+hbcm4)+8*(p3p7*p3p4
     . )*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc+
     . fmb*hbcm-fmb2+fmc*hbcm-hbcm2)+8*(p3p7*p1p3)*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmb*fmc-fmb*hbcm+fmb2-fmc*hbcm+
     . hbcm2)+16*(fmb*hbcm*p1p4*p3p7-fmc*hbcm*p2p3*p5p7+fmc*hbcm*p2p5
     . *p3p7-hbcm2*p1p2*p5p7+hbcm2*p2p4*p5p7-hbcm2*p2p5*p4p7+p1p3*
     . p2p3*p5p7-p1p3*p2p5*p3p7-p2p3*p3p4*p5p7+p2p5*p3p4*p3p7))+ans3
      ans1=w8*(16*p3p7*(-ffmcfmb*p2p3s2+p2p3s2)+(p3p7*p3p5*p2p3)*(16*
     . ffmcfmb-16)+8*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(
     . p3p7*p2p3)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-
     . fmb*fmc+fmb*hbcm-fmb2+fmc*hbcm-hbcm2)+16*(fmb*hbcm*p2p5*p3p7+
     . p2p3*p2p5*p3p7-p2p3s2*p5p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(7)=ccc*(w8*(8*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p3*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(-p2p3*p2p5-p2p3s2))+w5*(8*
     . p2p4*(-fmb*hbcm-fmc*hbcm+hbcm2)+8*p1p2*(fmb*hbcm+fmc*hbcm-
     . hbcm2)+4*p3p4*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*p1p4*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+4*p1p3*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4
     . *(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm2-fmb*fmc*hbcm2+fmb*
     . fmc2*hbcm+fmb2*fmc*hbcm-2*fmc*hbcm*p2p3-4*fmc*hbcm*p2p5-fmc2*
     . hbcm2+2*p1p3*p2p3+2*p1p3*p2p5-2*p2p3*p3p4-2*p2p5*p3p4))+2*(4*
     . fb1*fmb*hbcm+4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2+8*fb2*p2p3+8*fb2*
     . p2p5-hbcm2))
      ans=ccc*(w8*(4*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*
     . fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+8*(-fmb*p2p3s2+fmc*
     . p2p3s2))+w5*(8*(p3p4*p2p3)*(-fmb+fmc)+8*(p2p3*p1p3)*(fmb-fmc)+
     . 8*p2p4*(fmb*hbcm2-fmc*hbcm2+hbcm3)+8*p1p2*(-fmb*hbcm2+fmc*
     . hbcm2-hbcm3)+8*p3p5*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2
     . )+8*p2p3*(ffmcfmb*fmc*hbcm2-fmc*hbcm2+fmc2*hbcm)+4*p3p4*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+fmc*hbcm2)+4*p1p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+4*(-ffmcfmb*
     . fmc*hbcm4-fmb*fmc*hbcm3-2*fmc*hbcm2*p2p5+fmc*hbcm4+4*hbcm*p1p2
     . *p2p3-4*hbcm*p1p2*p3p5+4*hbcm*p1p3*p2p5-4*hbcm*p1p4*p2p3-4*
     . hbcm*p2p3*p2p4+4*hbcm*p2p4*p3p5-4*hbcm*p2p5*p3p4))+(16*p2p3*(
     . fb2*fmb-fb2*fmc)+2*(-4*fb1*fmb*fmc*hbcm+4*fb1*fmb2*hbcm+4*fb2*
     . ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-4*fb2*fmb*hbcm2+4*
     . fb2*fmc*hbcm2-fmb*hbcm2+fmc*hbcm2)))
      b(8)=ans
      b(11)=ccc*(w8*(8*p3p5*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p2p3*(-
     . ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2)+4*(ffmcfmb*hbcm4+fmb*hbcm3-
     . hbcm4+2*p2p3*p3p5))+w5*(4*p3p4*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*
     . hbcm+2*hbcm2)+4*p1p3*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm-2*
     . hbcm2)+4*(fmb*fmc*hbcm2-2*fmc*hbcm*p2p3+4*fmc*hbcm*p3p5+2*fmc*
     . hbcm3+fmc2*hbcm2+2*hbcm2*p1p2-2*hbcm2*p1p4-2*hbcm2*p2p4-2*p1p3
     . *p3p5+2*p3p4*p3p5))+2*(-4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2-8*
     . fb2*p3p5+hbcm2))
      b(12)=ccc*(w8*(8*p3p7*(-fmb*fmc*hbcm+fmb2*hbcm)+8*(p3p7*p2p3)*(
     . -ffmcfmb*hbcm-fmc+hbcm))+w5*(8*p4p7*(ffmcfmb*hbcm3+fmc*hbcm2-2
     . *hbcm3)+8*(p3p7*p3p4)*(-ffmcfmb*hbcm-fmc+hbcm)+8*(p3p7*p1p3)*(
     . ffmcfmb*hbcm+fmc-hbcm)+8*(fmc*hbcm2*p5p7-fmc2*hbcm*p3p7-2*hbcm
     . *p1p3*p5p7+2*hbcm*p1p4*p3p7+2*hbcm*p2p3*p4p7+2*hbcm*p3p4*p5p7-
     . 2*hbcm*p3p5*p4p7))+4*p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm+4*
     . fb2*fmc+3*hbcm))
      b(13)=ccc*(w5*(4*p3p4*(fmb-fmc-2*hbcm)+4*p1p3*(-fmb+fmc+2*hbcm)
     . +4*(-ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm-2*hbcm*p1p2+4*
     . hbcm*p1p4+2*hbcm*p2p4))+w8*(4*p2p3*(fmb-fmc)+4*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*
     . hbcm2))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(w8*(8*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(ffmcfmb
     . *p2p3*p3p7+fmb*hbcm*p3p7+p2p3*p5p7))+w5*(8*p4p7*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(-ffmcfmb*p1p3*p3p7+ffmcfmb*p3p4*p3p7+fmc*
     . hbcm*p3p7+2*fmc*hbcm*p5p7-p1p3*p5p7+p3p4*p5p7))+(p3p7*(-16*fb2
     . *ffmcfmb-4)-16*fb2*p5p7))
      do 200 n=1,16 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp4_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)
     . )
      b(1)=ccc*(16*w7*(p3p7*p3p6*p2p3)*(-ffmcfmb*fmb+ffmcfmb*fmc+
     . ffmcfmb*hbcm-fmc)+w2*(16*(p5p7*p3p6)*(-ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3+fmc*hbcm2+fmc2*hbcm)+16*(p3p7*p3p6*p3p5)*(-
     . ffmcfmb*fmb+ffmcfmb*fmc+ffmcfmb*hbcm-fmc)+16*(-ffmcfmb*fmb*
     . hbcm2*p3p6*p3p7+2*hbcm*p2p3*p3p6*p5p7-2*hbcm*p2p5*p3p6*p3p7))+
     . 8*(p3p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb2*ffmcfmb*fmb+4*fb2*
     . ffmcfmb*fmc-4*fb2*fmc-fmb+fmc+3*hbcm))
      b(2)=ccc*(8*w7*(p3p7*p2p3)*(ffmcfmb*fmb-ffmcfmb*fmc-ffmcfmb*
     . hbcm+fmc)+w2*(8*p5p7*(ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmc*
     . hbcm2-fmc2*hbcm)+8*(p3p7*p3p5)*(ffmcfmb*fmb-ffmcfmb*fmc-
     . ffmcfmb*hbcm+fmc)+8*(ffmcfmb*fmb*hbcm2*p3p7-2*hbcm*p2p3*p5p7+2
     . *hbcm*p2p5*p3p7))+4*p3p7*(-4*fb1*ffmcfmb*hbcm+4*fb2*ffmcfmb*
     . fmb-4*fb2*ffmcfmb*fmc+4*fb2*fmc+fmb-fmc-3*hbcm))
      ans=ccc*(w7*(8*(p5p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*
     . p2p3)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmb*
     . fmc+fmc2)+16*(ffmcfmb*hbcm2*p2p5*p3p7-ffmcfmb*p2p3*p3p5*p3p7+
     . ffmcfmb*p2p3s2*p3p7))+w2*(8*(p5p7*p3p5)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+8*p5p7*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+ffmcfmb*
     . hbcm4-fmb*fmc*hbcm2-fmc*hbcm3-fmc2*hbcm2)+8*(p3p7*p3p5)*(-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmb*fmc+fmc2)+
     . 8*p3p7*(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmb2*hbcm2-2*ffmcfmb*
     . p3p5s2)+16*(-ffmcfmb*hbcm2*p2p3*p5p7+ffmcfmb*p2p3*p3p5*p3p7+
     . fmc*hbcm*p2p5*p3p7))+((p3p7*p3p5)*(-32*fb2*ffmcfmb-8)+(p3p7*
     . p2p3)*(32*fb2*ffmcfmb+8)+4*p5p7*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*
     . hbcm2-hbcm2)+4*p3p7*(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc
     . *hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*fmb*fmc+4*fb2*fmc2-3*fmb*hbcm-
     . 3*fmc*hbcm-hbcm2)))
      b(5)=ans
      b(7)=ccc*(w7*(8*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p2p3*(-ffmcfmb
     . *hbcm2+fmc*hbcm)+8*p2p3*p2p5)+w2*(8*p2p5*(fmb*hbcm-fmc*hbcm+
     . hbcm2)+4*p3p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(-ffmcfmb*fmb*fmc*
     . hbcm2-ffmcfmb*fmb2*hbcm2+fmb*fmc2*hbcm+2*fmb*hbcm*p2p3+fmb2*
     . fmc*hbcm+2*p2p5*p3p5))+2*(4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2+8
     . *fb2*p2p5-hbcm2))
      b(8)=ccc*(4*w7*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*
     . fmc*hbcm-fmc2*hbcm)+w2*(8*p2p5*(-fmc*hbcm2+hbcm3)+4*p3p5*(
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmc2*hbcm)+8*
     . p2p3*(-ffmcfmb*fmb*hbcm2+fmb2*hbcm)+4*(ffmcfmb*fmb*hbcm4-fmb*
     . fmc*hbcm3-4*hbcm*p2p3*p2p5))+2*(4*fb1*fmb*fmc*hbcm-4*fb1*fmc2*
     . hbcm+4*fb2*ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-fmb*hbcm2
     . +fmc*hbcm2))
      b(11)=ccc*(w2*(4*p3p5*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-hbcm2)
     . +4*(fmb*fmc*hbcm2+2*fmb*hbcm*p2p3-2*fmb*hbcm3+fmb2*hbcm2-2*
     . hbcm2*p2p5+2*p2p3*p3p5-2*p3p5s2))+w7*(8*p3p5*(ffmcfmb*hbcm2-
     . fmc*hbcm)+4*p2p3*(-ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+4*(ffmcfmb*
     . hbcm4-fmc*hbcm3-2*p2p3*p3p5+2*p2p3s2))+2*(-4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2-4*fb2*hbcm2+8*fb2*p2p3-8*fb2*p3p5+hbcm2))
      b(12)=ccc*(w7*(8*(p3p7*p2p3)*(-ffmcfmb*hbcm+fmc)+8*p3p7*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2))+w2*(8*p5p7*(ffmcfmb*
     . hbcm3-hbcm3)+8*(p3p7*p3p5)*(-ffmcfmb*hbcm+fmc)+8*p3p7*(ffmcfmb
     . *fmb*hbcm2+fmb*fmc*hbcm-fmb2*hbcm)+16*hbcm*p2p5*p3p7)+4*p3p7*(
     . -4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm))
      b(13)=ccc*(w7*(4*p2p3*(-fmb+fmc)+4*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm))+w2*(4*p3p5*(-fmb+fmc-2*hbcm
     . )+4*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm+2*
     . hbcm*p2p5))+8*(-fb2*fmb+fb2*fmc))
      b(14)=ccc*(w2*(8*p5p7*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(-
     . ffmcfmb*p3p5*p3p7-fmb*hbcm*p3p7-p3p5*p5p7))+w7*(8*p5p7*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(ffmcfmb*hbcm2*p3p7-ffmcfmb*p2p3*
     . p3p7-p2p3*p5p7))+(p3p7*(-16*fb2*ffmcfmb-4)-16*fb2*p5p7))
      do 200 n=1,16 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
      end 
 
      subroutine amp36_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      ans4=8*(p4p7*p3p6)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb
     . *hbcm3-fmb*fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm)+2*(p6p7*p3p4)*(
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb
     . *hbcm3-3*fmb*fmc*hbcm+fmc*hbcm2+4*fmc2*hbcm)+4*(p3p7*p3p6*p3p4
     . )*(3*ffmcfmb**2*hbcm-ffmcfmb*fmb+4*ffmcfmb*fmc+6*ffmcfmb*hbcm+
     . fmb-2*fmc-3*hbcm)+2*p6p7*(-ffmcfmb**2*fmc*hbcm4-3*ffmcfmb*fmb*
     . fmc*hbcm3+ffmcfmb*fmc*hbcm4+3*ffmcfmb*fmc2*hbcm3+2*ffmcfmb*
     . hbcm*p3p4s2+3*fmb*fmc2*hbcm2-4*fmb*p3p4s2+2*fmc*p3p4s2+3*fmc2*
     . hbcm3-2*fmc3*hbcm2)+4*(p3p7*p3p6)*(-ffmcfmb**2*fmc*hbcm2-
     . ffmcfmb*fmb*fmc*hbcm+8*ffmcfmb*fmc*hbcm2+4*ffmcfmb*fmc2*hbcm+
     . fmb*fmc*hbcm-3*fmc*hbcm2-2*fmc2*hbcm)+8*(2*ffmcfmb*hbcm*p1p3*
     . p3p6*p4p7+2*ffmcfmb*hbcm*p2p3*p3p6*p4p7-2*ffmcfmb*hbcm*p3p5*
     . p3p6*p4p7+fmc*hbcm2*p2p5*p6p7-fmc*hbcm2*p5p6*p5p7+2*hbcm*p1p3*
     . p1p4*p6p7+hbcm*p1p3*p4p6*p5p7+2*hbcm*p1p4*p2p3*p6p7-hbcm*p1p4*
     . p3p4*p6p7-2*hbcm*p1p4*p3p5*p6p7+hbcm*p2p3*p4p6*p5p7+hbcm*p2p5*
     . p3p4*p6p7+hbcm*p3p4*p4p6*p4p7-hbcm*p3p4*p4p6*p5p7-hbcm*p3p4*
     . p5p6*p5p7-hbcm*p3p5*p4p6*p5p7)
      ans3=4*(p5p7*p4p6)*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p4p7*p4p6
     . )*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm3-fmb
     . *hbcm2+fmc*hbcm2-hbcm3)+8*(p4p6*p3p7*p3p5)*(3*ffmcfmb*hbcm-
     . hbcm)+8*(p4p6*p3p7*p3p4)*(4*ffmcfmb*hbcm+fmb-hbcm)+4*(p5p7*
     . p3p6*p3p4)*(ffmcfmb*hbcm+fmb-2*fmc-5*hbcm)+8*(p4p7*p3p6*p3p4)*
     . (-ffmcfmb*hbcm-fmb+fmc+2*hbcm)+4*(p6p7*p3p5*p3p4)*(3*ffmcfmb*
     . hbcm+fmc)+8*(p4p6*p3p7*p2p3)*(-3*ffmcfmb*hbcm+hbcm)+4*(p6p7*
     . p3p4*p2p3)*(-3*ffmcfmb*hbcm-fmc)+8*(p4p6*p3p7*p1p3)*(-3*
     . ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p4*p1p3)*(-ffmcfmb*hbcm+2*fmb-fmc
     . )+4*(p5p7*p3p6)*(3*ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-5*fmc*hbcm2-
     . 2*fmc2*hbcm)+4*(p6p7*p3p5)*(ffmcfmb*fmc*hbcm2+3*fmc2*hbcm)+4*(
     . p6p7*p2p3)*(-ffmcfmb*fmc*hbcm2-3*fmc2*hbcm)+4*(p6p7*p1p3)*(
     . ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-3*fmc2*hbcm)+4*(p4p6*p3p7)*(-
     . ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+4*
     . ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb*hbcm2-hbcm3)+ans4
      ans2=w1*ans3
      ans8=2*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*hbcm5-ffmcfmb*fmb
     . *fmc*hbcm3+ffmcfmb*fmc*hbcm4+ffmcfmb*fmc2*hbcm3+2*ffmcfmb*hbcm
     . *p1p3s2+2*ffmcfmb*hbcm*p3p4s2+2*fmb*fmc2*hbcm2-4*fmb*p1p3s2-4*
     . fmb*p3p4s2+2*fmc*p1p3s2+2*fmc*p3p4s2+2*fmc2*hbcm3-fmc3*hbcm2)+
     . 4*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2+3*ffmcfmb**2*hbcm3-4*
     . ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmb*hbcm2+6*ffmcfmb*fmc*hbcm2+7*
     . ffmcfmb*fmc2*hbcm-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-3*fmc*hbcm2-3
     . *fmc2*hbcm)+8*(3*ffmcfmb*hbcm*p2p3*p3p6*p5p7-ffmcfmb*hbcm*p2p5
     . *p3p6*p3p7-ffmcfmb*hbcm*p3p5*p3p6*p5p7-hbcm*p1p2*p3p6*p5p7+
     . hbcm*p1p3*p1p4*p6p7-hbcm*p1p3*p2p5*p6p7-hbcm*p1p3*p4p6*p4p7+4*
     . hbcm*p1p3*p4p6*p5p7+hbcm*p1p3*p5p6*p5p7+2*hbcm*p1p4*p2p3*p6p7-
     . hbcm*p1p4*p3p4*p6p7-2*hbcm*p1p4*p3p5*p6p7+hbcm*p1p4*p3p6*p5p7-
     . 2*hbcm*p1p4*p3p7*p4p6-hbcm*p1p5*p3p6*p5p7-2*hbcm*p2p3*p4p6*
     . p4p7+6*hbcm*p2p3*p4p6*p5p7-hbcm*p2p4*p3p6*p5p7+hbcm*p2p5*p3p4*
     . p6p7+hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*p4p6+hbcm*p3p4*p4p6*
     . p4p7-4*hbcm*p3p4*p4p6*p5p7-hbcm*p3p4*p5p6*p5p7+2*hbcm*p3p5*
     . p4p6*p4p7-2*hbcm*p3p5*p4p6*p5p7-hbcm*p3p6*p4p5*p5p7)
      ans7=4*(p4p6*p3p7)*(2*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2*
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-fmc*hbcm2+2*fmc2*hbcm-2*
     . hbcm3)+4*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-5*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-4*fmc*hbcm2-3
     . *fmc2*hbcm)+4*(p4p7*p3p6)*(-2*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*
     . hbcm2+ffmcfmb*hbcm3-2*fmb*fmc*hbcm+3*fmc*hbcm2+2*fmc2*hbcm)+4*
     . (p6p7*p3p5)*(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2+2*fmc2*hbcm)+2
     . *(p6p7*p3p4)*(ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc
     . *hbcm2+3*ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+4*(
     . p6p7*p2p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2-2*fmc2*hbcm)+2
     . *(p6p7*p1p3)*(-ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+
     . 4*(p3p7*p3p6*p3p4)*(3*ffmcfmb**2*hbcm-ffmcfmb*fmb+4*ffmcfmb*
     . fmc+8*ffmcfmb*hbcm+fmb-2*fmc-5*hbcm)+4*(p3p7*p3p6*p1p3)*(-3*
     . ffmcfmb**2*hbcm+ffmcfmb*fmb-4*ffmcfmb*fmc-fmb+2*fmc+hbcm)+ans8
      ans6=4*(p5p7*p5p6)*(-ffmcfmb*hbcm3-fmc*hbcm2)+4*(p5p7*p4p6)*(-5
     . *ffmcfmb*hbcm3-fmc*hbcm2-2*hbcm3)+4*(p4p7*p4p6)*(-ffmcfmb*
     . hbcm3+fmc*hbcm2+2*hbcm3)+4*(p6p7*p2p5)*(ffmcfmb*hbcm3+fmc*
     . hbcm2)+4*(p6p7*p1p4)*(-ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*
     . hbcm3)+16*(p4p6*p3p7*p3p5)*(2*ffmcfmb*hbcm-hbcm)+8*(p4p6*p3p7*
     . p3p4)*(5*ffmcfmb*hbcm+fmb-2*hbcm)+4*(p5p7*p3p6*p3p4)*(-ffmcfmb
     . *hbcm+fmb-2*fmc-9*hbcm)+8*(p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+
     . fmc+2*hbcm)+4*(p6p7*p3p5*p3p4)*(3*ffmcfmb*hbcm+fmc)+16*(p4p6*
     . p3p7*p2p3)*(-2*ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p4*p2p3)*(-3*
     . ffmcfmb*hbcm-fmc)+8*(p3p7*p3p6*p1p4)*(-3*ffmcfmb*hbcm+hbcm)+8*
     . (p4p6*p3p7*p1p3)*(-5*ffmcfmb*hbcm-fmb+2*hbcm)+4*(p5p7*p3p6*
     . p1p3)*(5*ffmcfmb*hbcm-fmb+2*fmc+hbcm)+8*(p4p7*p3p6*p1p3)*(
     . ffmcfmb*hbcm+fmb-fmc)+4*(p6p7*p3p5*p1p3)*(-3*ffmcfmb*hbcm-fmc)
     . +8*(p6p7*p3p4*p1p3)*(-ffmcfmb*hbcm+2*fmb-fmc)+4*(p6p7*p2p3*
     . p1p3)*(3*ffmcfmb*hbcm+fmc)+ans7
      ans5=w5*ans6
      ans12=8*(-ffmcfmb*hbcm*p2p5*p3p6*p3p7+fmb*hbcm2*p1p4*p6p7+2*fmb
     . *hbcm2*p4p6*p4p7-fmb*hbcm2*p5p6*p5p7-hbcm*p1p2*p3p6*p5p7+hbcm*
     . p1p4*p3p5*p6p7-2*hbcm*p1p5*p2p3*p6p7+2*hbcm*p1p5*p3p5*p6p7-2*
     . hbcm*p1p5*p3p6*p4p7+2*hbcm*p1p5*p3p7*p4p6+2*hbcm*p2p3*p2p5*
     . p6p7+2*hbcm*p2p3*p4p5*p6p7+2*hbcm*p2p3*p4p6*p5p7-2*hbcm*p2p3*
     . p4p7*p5p6-hbcm*p2p4*p3p6*p5p7-hbcm*p2p5*p3p5*p6p7+hbcm*p2p5*
     . p3p6*p4p7-2*hbcm*p2p5*p3p7*p4p6-2*hbcm*p3p5*p4p5*p6p7+3*hbcm*
     . p3p5*p4p6*p4p7+2*hbcm*p3p5*p4p7*p5p6-hbcm*p3p5*p5p6*p5p7-2*
     . hbcm*p3p7*p4p5*p4p6)
      ans11=4*(p6p7*p3p4)*(-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+2*fmb2*
     . hbcm)+4*(p6p7*p2p3)*(-ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+4*(p6p7*
     . p1p3)*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb2*hbcm)+4*(p5p6*
     . p3p7)*(-2*ffmcfmb**2*hbcm3+4*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*
     . hbcm2+5*ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*hbcm3)+4*(p5p7*
     . p3p6)*(ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-
     . 3*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-3*fmb*hbcm2-fmb2*hbcm-fmc*hbcm2
     . +2*hbcm3)+2*(p6p7*p3p5)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2-2*
     . fmc2*hbcm)+4*(p3p7*p3p6*p3p5)*(-3*ffmcfmb**2*hbcm+ffmcfmb*fmb-
     . 4*ffmcfmb*fmc+8*ffmcfmb*hbcm-fmb+2*fmc-3*hbcm)+2*p6p7*(-
     . ffmcfmb**2*fmb*hbcm4-ffmcfmb*fmb*fmc*hbcm3+ffmcfmb*fmb*hbcm4+3
     . *ffmcfmb*fmb2*hbcm3-6*ffmcfmb*hbcm*p3p5s2-fmb*fmc*hbcm3-2*fmb*
     . fmc2*hbcm2+fmb2*fmc*hbcm2-2*fmc*p3p5s2)+4*(p3p7*p3p6)*(-
     . ffmcfmb**2*fmb*hbcm2-4*ffmcfmb*fmb*fmc*hbcm+8*ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmb2*hbcm+2*fmb*fmc*hbcm-3*fmb*hbcm2-fmb2*hbcm)+
     . ans12
      ans10=8*(p4p7*p3p6)*(-fmb*fmc*hbcm+2*fmb*hbcm2+fmb2*hbcm)+4*(
     . p5p6*p4p7)*(-ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2+2*hbcm3)+8*(
     . p5p7*p4p6)*(-ffmcfmb*hbcm3+hbcm3)+4*(p6p7*p4p5)*(ffmcfmb*hbcm3
     . -2*fmb*hbcm2+fmc*hbcm2-2*hbcm3)+4*(p6p7*p2p5)*(ffmcfmb*hbcm3+
     . fmc*hbcm2-2*hbcm3)+4*(p6p7*p1p5)*(-ffmcfmb*hbcm3+2*fmb*hbcm2-
     . fmc*hbcm2+2*hbcm3)+8*(p4p5*p3p7*p3p6)*(-3*ffmcfmb*hbcm+hbcm)+
     . 16*(p5p6*p3p7*p3p5)*(2*ffmcfmb*hbcm-hbcm)+8*(p4p6*p3p7*p3p5)*(
     . 3*ffmcfmb*hbcm-fmb-2*hbcm)+4*(p5p7*p3p6*p3p5)*(ffmcfmb*hbcm-
     . fmb+2*fmc-hbcm)+8*(p4p7*p3p6*p3p5)*(ffmcfmb*hbcm+fmb-fmc+2*
     . hbcm)+8*(p5p7*p3p6*p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p5*p3p4
     . )*(-ffmcfmb*hbcm+2*fmb-fmc)+16*(p5p6*p3p7*p2p3)*(-2*ffmcfmb*
     . hbcm+hbcm)+8*(p5p7*p3p6*p2p3)*(3*ffmcfmb*hbcm-2*hbcm)+4*(p6p7*
     . p3p5*p2p3)*(3*ffmcfmb*hbcm+fmc)+8*(p3p7*p3p6*p1p5)*(-ffmcfmb*
     . hbcm+hbcm)+8*(p5p7*p3p6*p1p3)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p5
     . *p1p3)*(ffmcfmb*hbcm-2*fmb+fmc)+8*(p4p6*p3p7)*(3*ffmcfmb*fmb*
     . hbcm2-fmb*hbcm2-fmb2*hbcm)+ans11
      ans9=w2*ans10
      ans13=(4*(p4p6*p3p7)*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmb+hbcm)+2*(
     . p5p7*p3p6)*(-20*fb1*ffmcfmb*hbcm+20*fb1*hbcm-4*fb2*fmb+8*fb2*
     . fmc+11*hbcm)+16*(p4p7*p3p6)*(fb1*ffmcfmb*hbcm-2*fb1*hbcm+fb2*
     . fmb-fb2*fmc)+2*(p6p7*p3p5)*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc+3*
     . hbcm)+2*(p6p7*p3p4)*(4*fb1*ffmcfmb*hbcm+8*fb2*fmb-4*fb2*fmc+
     . hbcm)+2*(p6p7*p2p3)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-3*hbcm)+2*(
     . p6p7*p1p3)*(-4*fb1*ffmcfmb*hbcm-8*fb2*fmb+4*fb2*fmc-hbcm)+p6p7
     . *(4*fb1*ffmcfmb**2*hbcm3-4*fb1*ffmcfmb*hbcm3-4*fb1*fmb*fmc*
     . hbcm+8*fb1*fmc2*hbcm+12*fb2*ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*
     . fmc*hbcm2-4*fb2*fmc*hbcm2+3*fmb*hbcm2-fmc*hbcm2+3*hbcm3)+2*(
     . p3p7*p3p6)*(20*fb1*ffmcfmb**2*hbcm-40*fb1*ffmcfmb*hbcm+12*fb1*
     . hbcm+4*fb2*ffmcfmb*fmb-16*fb2*ffmcfmb*fmc-4*fb2*fmb+8*fb2*fmc-
     . 12*ffmcfmb*hbcm+fmb-4*fmc+14*hbcm)+16*(-fb1*hbcm*p1p4*p6p7-fb1
     . *hbcm*p2p5*p6p7-fb1*hbcm*p4p6*p4p7+fb1*hbcm*p5p6*p5p7))
      ans1=ans2+ans5+ans9+ans13
      ans=ccc*ans1
      b(1)=ans
      ans4=(12*p5p7*(-fb1*hbcm+fb2*fmb-fb2*fmc-2*hbcm)+p4p7*(-4*fb1*
     . ffmcfmb*hbcm+16*fb1*hbcm-16*fb2*fmb+12*fb2*fmc+hbcm)+p3p7*(-4*
     . fb1*ffmcfmb**2*hbcm+28*fb1*ffmcfmb*hbcm-12*fb1*hbcm-28*fb2*
     . ffmcfmb*fmb+24*fb2*ffmcfmb*fmc+12*fb2*fmb-12*fb2*fmc+12*
     . ffmcfmb*hbcm-7*fmb+6*fmc-17*hbcm))
      ans3=w2*(6*(p5p7*p3p5)*(fmb-fmc+hbcm)+4*(p4p5*p3p7)*(ffmcfmb*
     . hbcm-hbcm)+2*(p4p7*p3p5)*(ffmcfmb*hbcm-4*fmb+3*fmc-4*hbcm)+4*(
     . p5p7*p3p4)*(-ffmcfmb*hbcm+hbcm)+4*(p3p7*p1p5)*(3*ffmcfmb*hbcm-
     . hbcm)+4*(p5p7*p1p3)*(-ffmcfmb*hbcm+hbcm)+2*p4p7*(ffmcfmb*fmb*
     . hbcm2+3*fmb*fmc*hbcm-4*fmb*hbcm2-4*fmb2*hbcm)+2*p5p7*(-2*
     . ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+3*fmb*hbcm2+3*
     . fmb2*hbcm)+2*(p3p7*p3p5)*(3*ffmcfmb**2*hbcm-7*ffmcfmb*fmb+6*
     . ffmcfmb*fmc-9*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+2*p3p7*(ffmcfmb
     . **2*fmb*hbcm2+6*ffmcfmb*fmb*fmc*hbcm-7*ffmcfmb*fmb*hbcm2-7*
     . ffmcfmb*fmb2*hbcm-3*fmb*fmc*hbcm+3*fmb*hbcm2+3*fmb2*hbcm)+4*(
     . ffmcfmb*hbcm*p2p3*p5p7-ffmcfmb*hbcm*p2p5*p3p7+hbcm*p1p2*p5p7+2
     . *hbcm*p1p5*p4p7+hbcm*p2p4*p5p7-hbcm*p2p5*p4p7))+ans4
      ans2=w5*(2*(p5p7*p3p4)*(-3*fmb+3*fmc+7*hbcm)+2*(p4p7*p3p4)*(-
     . ffmcfmb*hbcm+4*fmb-3*fmc-4*hbcm)+4*(p3p7*p1p4)*(5*ffmcfmb*hbcm
     . -3*hbcm)+2*(p5p7*p1p3)*(-4*ffmcfmb*hbcm+3*fmb-3*fmc+hbcm)+2*(
     . p4p7*p1p3)*(ffmcfmb*hbcm-4*fmb+3*fmc)+2*p4p7*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*fmc*hbcm2-4*fmc2*hbcm)+2*
     . p5p7*(-2*ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+3*fmc
     . *hbcm2+5*fmc2*hbcm)+2*(p3p7*p3p4)*(-3*ffmcfmb**2*hbcm+7*
     . ffmcfmb*fmb-6*ffmcfmb*fmc-7*ffmcfmb*hbcm-3*fmb+3*fmc+5*hbcm)+2
     . *(p3p7*p1p3)*(3*ffmcfmb**2*hbcm-7*ffmcfmb*fmb+6*ffmcfmb*fmc-
     . ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+2*p3p7*(ffmcfmb**2*fmb*hbcm2-3*
     . ffmcfmb**2*hbcm3+6*ffmcfmb*fmb*fmc*hbcm-6*ffmcfmb*fmc*hbcm2-9*
     . ffmcfmb*fmc2*hbcm+2*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+3*fmc*hbcm2+5
     . *fmc2*hbcm)+4*(ffmcfmb*hbcm*p2p3*p5p7-ffmcfmb*hbcm*p2p5*p3p7+
     . ffmcfmb*hbcm*p3p5*p5p7+hbcm*p1p2*p5p7+2*hbcm*p1p4*p4p7-3*hbcm*
     . p1p4*p5p7+hbcm*p1p5*p5p7+hbcm*p2p4*p5p7-hbcm*p2p5*p4p7+hbcm*
     . p4p5*p5p7))+ans3
      ans1=w1*(2*p5p7*(-3*fmb*fmc*hbcm+3*fmc*hbcm2+5*fmc2*hbcm)+2*(
     . p5p7*p3p4)*(2*ffmcfmb*hbcm-3*fmb+3*fmc+3*hbcm)+2*(p4p7*p3p4)*(
     . -ffmcfmb*hbcm+4*fmb-3*fmc-4*hbcm)+8*(p3p7*p1p4)*(2*ffmcfmb*
     . hbcm-hbcm)+2*p4p7*(ffmcfmb*fmc*hbcm2+4*fmb*fmc*hbcm-4*fmc*
     . hbcm2-5*fmc2*hbcm)+2*(p3p7*p3p4)*(-3*ffmcfmb**2*hbcm+7*ffmcfmb
     . *fmb-6*ffmcfmb*fmc-5*ffmcfmb*hbcm-3*fmb+3*fmc+3*hbcm)+2*p3p7*(
     . ffmcfmb**2*fmc*hbcm2+7*ffmcfmb*fmb*fmc*hbcm-7*ffmcfmb*fmc*
     . hbcm2-10*ffmcfmb*fmc2*hbcm-3*fmb*fmc*hbcm+3*fmc*hbcm2+5*fmc2*
     . hbcm)+8*(hbcm*p1p4*p4p7-hbcm*p1p4*p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans3=w2*(2*(p5p6*p3p5)*(ffmcfmb*hbcm-2*fmb+fmc-2*hbcm)+2*p5p6*(
     . ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb*hbcm2-2*fmb2*hbcm)+2*(
     . p3p6*p3p5)*(3*ffmcfmb**2*hbcm+2*ffmcfmb*fmb-ffmcfmb*fmc+3*
     . ffmcfmb*hbcm-2*fmb+fmc-2*hbcm)+2*p3p6*(-ffmcfmb**2*fmb*hbcm2-
     . ffmcfmb*fmb*fmc*hbcm+3*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmb2*hbcm+
     . fmb*fmc*hbcm-2*fmb*hbcm2-2*fmb2*hbcm)+8*(-ffmcfmb*hbcm*p1p5*
     . p3p6+ffmcfmb*hbcm*p2p5*p3p6+ffmcfmb*hbcm*p3p5*p4p6+ffmcfmb*
     . hbcm*p3p6*p4p5-hbcm*p1p5*p4p6+hbcm*p2p5*p4p6+hbcm*p4p5*p4p6))+
     . (p5p6*(-4*fb1*ffmcfmb*hbcm+8*fb1*hbcm-8*fb2*fmb+4*fb2*fmc-hbcm
     . )+p3p6*(4*fb1*ffmcfmb**2*hbcm-12*fb1*ffmcfmb*hbcm+8*fb1*hbcm+8
     . *fb2*ffmcfmb*fmb-4*fb2*ffmcfmb*fmc-8*fb2*fmb+4*fb2*fmc+12*
     . ffmcfmb*hbcm+2*fmb-fmc-11*hbcm)-12*hbcm*p4p6)
      ans2=w5*(4*p4p6*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3)+2*
     . (p5p6*p3p4)*(-ffmcfmb*hbcm+2*fmb-fmc-2*hbcm)+2*(p5p6*p1p3)*(
     . ffmcfmb*hbcm-2*fmb+fmc+2*hbcm)+2*p5p6*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+2*(p3p6*p3p4
     . )*(-3*ffmcfmb**2*hbcm-2*ffmcfmb*fmb+ffmcfmb*fmc+ffmcfmb*hbcm+2
     . *fmb-fmc-2*hbcm)+2*(p3p6*p1p3)*(3*ffmcfmb**2*hbcm+2*ffmcfmb*
     . fmb-ffmcfmb*fmc-ffmcfmb*hbcm-2*fmb+fmc+2*hbcm)+2*p3p6*(-
     . ffmcfmb**2*fmb*hbcm2-3*ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmb2*hbcm+ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+8*(-ffmcfmb*
     . hbcm*p1p2*p3p6+ffmcfmb*hbcm*p1p3*p4p6-ffmcfmb*hbcm*p1p5*p3p6+
     . ffmcfmb*hbcm*p2p3*p3p6+ffmcfmb*hbcm*p2p4*p3p6+2*ffmcfmb*hbcm*
     . p2p5*p3p6-ffmcfmb*hbcm*p3p4*p4p6+ffmcfmb*hbcm*p3p5*p3p6+
     . ffmcfmb*hbcm*p3p6*p4p5-hbcm*p1p2*p4p6+hbcm*p1p4*p3p6+hbcm*p1p4
     . *p4p6+hbcm*p1p4*p5p6-hbcm*p1p5*p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*
     . p4p6+2*hbcm*p2p5*p4p6+hbcm*p3p5*p4p6+hbcm*p4p5*p4p6))+ans3
      ans1=w1*(2*(p5p6*p3p4)*(-ffmcfmb*hbcm+2*fmb-fmc-2*hbcm)+4*(p4p6
     . *p3p4)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p4)*(-ffmcfmb*hbcm+hbcm)
     . +4*(p4p6*p1p3)*(ffmcfmb*hbcm-hbcm)+2*p5p6*(ffmcfmb*fmc*hbcm2+2
     . *fmb*fmc*hbcm-2*fmc*hbcm2-3*fmc2*hbcm)+4*p4p6*(ffmcfmb**2*
     . hbcm3-ffmcfmb*hbcm3)+2*(p3p6*p3p4)*(-3*ffmcfmb**2*hbcm-2*
     . ffmcfmb*fmb+ffmcfmb*fmc+5*ffmcfmb*hbcm+2*fmb-fmc-2*hbcm)+2*
     . p3p6*(-ffmcfmb**2*fmc*hbcm2-2*ffmcfmb*fmb*fmc*hbcm+3*ffmcfmb*
     . fmc*hbcm2+3*ffmcfmb*fmc2*hbcm+2*fmb*fmc*hbcm-2*fmc*hbcm2-3*
     . fmc2*hbcm)+4*(-ffmcfmb*hbcm*p2p3*p4p6+2*ffmcfmb*hbcm*p2p4*p3p6
     . -ffmcfmb*hbcm*p3p5*p4p6+2*ffmcfmb*hbcm*p3p6*p4p5-hbcm*p1p2*
     . p4p6+2*hbcm*p1p4*p5p6-hbcm*p1p5*p4p6+hbcm*p2p4*p4p6+hbcm*p4p5*
     . p4p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans4=8*(-2*ffmcfmb*p3p4*p3p7*p4p6-fmc*hbcm*p1p2*p6p7-fmc*hbcm*
     . p1p3*p6p7+fmc*hbcm*p2p3*p6p7+fmc*hbcm*p2p4*p6p7+fmc*hbcm*p5p6*
     . p5p7-p1p2*p3p4*p6p7-p1p3*p3p4*p6p7-p1p4*p3p4*p6p7+p2p3*p3p4*
     . p6p7+p2p4*p3p4*p6p7-p3p4*p3p6*p4p7-2*p3p4*p4p6*p4p7+p3p4*p4p6*
     . p5p7+p3p4*p5p6*p5p7)
      ans3=(p5p7*p3p6*p3p4)*(-16*ffmcfmb+16)+4*(p5p7*p4p6)*(ffmcfmb*
     . hbcm2-fmb*hbcm+2*fmc*hbcm+hbcm2)+8*(p4p7*p4p6)*(-ffmcfmb*hbcm2
     . -2*fmc*hbcm)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm+
     . hbcm2)+8*(p5p7*p3p6)*(-ffmcfmb*fmc*hbcm+2*fmc*hbcm)+8*(p3p7*
     . p3p6)*(-3*ffmcfmb*fmc*hbcm+fmc*hbcm)+(p3p7*p3p6*p3p4)*(16*
     . ffmcfmb**2-32*ffmcfmb+8)+4*(p4p6*p3p7)*(-ffmcfmb**2*hbcm2+3*
     . ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+8*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+2*(p6p7*p3p4)*(
     . ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2+fmb*fmc+3*fmc*hbcm+2*fmc2)+2*p6p7*(-ffmcfmb**2*fmc*hbcm3
     . +ffmcfmb*fmb*fmc*hbcm2+3*ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*
     . hbcm2+3*fmb*fmc2*hbcm-3*fmc2*hbcm2+2*fmc3*hbcm+4*p3p4s2)+ans4
      ans2=w1*ans3
      ans7=2*(p6p7*p1p3)*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-3*ffmcfmb*hbcm2-fmb*fmc-fmc*hbcm-2*fmc2)+2*
     . p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*hbcm4+ffmcfmb*fmb*fmc*
     . hbcm2+ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*hbcm2+2*fmb*fmc2*hbcm-2
     . *fmc2*hbcm2+fmc3*hbcm+4*p1p3s2+4*p3p4s2)+8*(2*ffmcfmb*p1p3*
     . p3p7*p4p6-2*ffmcfmb*p3p4*p3p7*p4p6+p1p2*p1p3*p6p7-p1p2*p3p4*
     . p6p7+p1p3*p1p4*p6p7-p1p3*p2p3*p6p7-p1p3*p2p4*p6p7-2*p1p3*p3p4*
     . p6p7+p1p3*p3p6*p4p7+2*p1p3*p4p6*p4p7-p1p3*p4p6*p5p7-p1p3*p5p6*
     . p5p7-p1p4*p3p4*p6p7+p2p3*p3p4*p6p7+p2p4*p3p4*p6p7-p3p4*p3p6*
     . p4p7-2*p3p4*p4p6*p4p7+p3p4*p4p6*p5p7+p3p4*p5p6*p5p7)
      ans6=(p5p7*p3p6*p3p4)*(-16*ffmcfmb+16)+(p5p7*p3p6*p1p3)*(16*
     . ffmcfmb-16)+4*(p5p7*p5p6)*(ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p7*
     . p4p6)*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm)+4*(p4p7*p4p6)*(-3*
     . ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm)+4*(p6p7*p2p4)*(ffmcfmb*
     . hbcm2+fmc*hbcm)+4*(p6p7*p2p3)*(ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7
     . *p1p4)*(-3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*hbcm2)+4*(p6p7*
     . p1p2)*(-ffmcfmb*hbcm2-fmc*hbcm)+4*(p3p7*p3p6)*(-2*ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2+fmc*hbcm)+(p3p7*p3p6*p3p4)*(16*ffmcfmb**2-
     . 32*ffmcfmb+8)+(p3p7*p3p6*p1p3)*(-16*ffmcfmb**2+32*ffmcfmb-8)+4
     . *(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm-2*ffmcfmb
     . *fmc*hbcm-ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm)+4*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+2*fmc*hbcm)+4*(p4p7*p3p6)*(-
     . 2*ffmcfmb**2*hbcm2+ffmcfmb*hbcm2-fmc*hbcm)+2*(p6p7*p3p4)*(
     . ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+3*ffmcfmb
     . *hbcm2+fmb*fmc+fmc*hbcm+2*fmc2)+ans7
      ans5=w5*ans6
      ans10=2*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb*fmb*fmc*hbcm2-3*
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-fmb*fmc*hbcm2-2*fmb*fmc2*
     . hbcm-fmb2*fmc*hbcm)+8*(2*ffmcfmb*p3p5*p3p7*p4p6+fmb*hbcm*p1p2*
     . p6p7+fmb*hbcm*p1p3*p6p7+fmb*hbcm*p1p4*p6p7-fmb*hbcm*p2p3*p6p7-
     . fmb*hbcm*p2p4*p6p7-fmb*hbcm*p3p4*p6p7+3*fmb*hbcm*p4p6*p4p7-fmb
     . *hbcm*p5p6*p5p7+p1p2*p3p5*p6p7+p1p3*p3p5*p6p7+p1p4*p3p5*p6p7-
     . p2p3*p3p5*p6p7-p2p4*p3p5*p6p7-p3p4*p3p5*p6p7+p3p5*p3p6*p4p7+2*
     . p3p5*p4p6*p4p7-p3p5*p4p6*p5p7-p3p5*p5p6*p5p7)
      ans9=(p5p7*p3p6*p3p5)*(16*ffmcfmb-16)+4*(p5p6*p4p7)*(ffmcfmb*
     . hbcm2+2*fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p5p7*p4p6)*(ffmcfmb*
     . hbcm2-fmb*hbcm-hbcm2)+4*(p6p7*p4p5)*(-ffmcfmb*hbcm2-2*fmb*hbcm
     . -fmc*hbcm+2*hbcm2)+4*(p6p7*p2p5)*(-ffmcfmb*hbcm2-2*fmb*hbcm-
     . fmc*hbcm+2*hbcm2)+4*(p6p7*p1p5)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*
     . hbcm-2*hbcm2)+8*(p4p6*p3p7)*(4*ffmcfmb*fmb*hbcm-fmb*hbcm)+8*(
     . p4p7*p3p6)*(ffmcfmb*fmb*hbcm+fmb*hbcm)+8*(p3p7*p3p6)*(3*
     . ffmcfmb*fmb*hbcm-fmb*hbcm)+(p3p7*p3p6*p3p5)*(-16*ffmcfmb**2+32
     . *ffmcfmb-8)+4*(p5p6*p3p7)*(2*ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*
     . hbcm+2*ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*
     . hbcm2)+4*(p5p7*p3p6)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-2*hbcm2)+2*
     . (p6p7*p3p5)*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc*hbcm-2*fmc2)+ans10
      ans8=w2*ans9
      ans11=((p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(p5p7*p3p6)*(32*fb2*
     . ffmcfmb-32*fb2+8)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+64*fb2*
     . ffmcfmb-16*fb2-16*ffmcfmb+16)+p6p7*(-4*fb1*ffmcfmb*fmb*hbcm-4*
     . fb1*ffmcfmb*fmc*hbcm+4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2-12*
     . fb2*ffmcfmb*hbcm2-4*fb2*fmb*fmc-8*fb2*fmc2+2*ffmcfmb*hbcm2+3*
     . fmb*hbcm+fmc*hbcm-3*hbcm2)+16*(fb2*p1p2*p6p7+fb2*p1p3*p6p7+fb2
     . *p1p4*p6p7-fb2*p2p3*p6p7-fb2*p2p4*p6p7-fb2*p3p4*p6p7+fb2*p3p6*
     . p4p7+2*fb2*p4p6*p4p7-fb2*p4p6*p5p7-fb2*p5p6*p5p7))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(4)=ans
      ans4=4*(2*fmc*hbcm*p1p3*p4p7-fmc*hbcm*p1p3*p5p7+4*fmc*hbcm*p2p3
     . *p4p7-3*fmc*hbcm*p2p3*p5p7-4*fmc*hbcm*p3p5*p4p7+3*fmc*hbcm*
     . p3p5*p5p7+2*hbcm2*p1p4*p4p7-2*hbcm2*p1p4*p5p7+2*p1p3*p3p4*p4p7
     . -p1p3*p3p4*p5p7+4*p2p3*p3p4*p4p7-3*p2p3*p3p4*p5p7-4*p3p4*p3p5*
     . p4p7+3*p3p4*p3p5*p5p7)
      ans3=2*(p5p7*p3p4)*(3*fmb*hbcm+fmc*hbcm+3*hbcm2)+(p3p7*p3p5*
     . p3p4)*(-28*ffmcfmb+12)+(p3p7*p3p4*p2p3)*(28*ffmcfmb-12)+(p3p7*
     . p3p4*p1p3)*(12*ffmcfmb-4)+2*(p4p7*p3p4)*(-ffmcfmb*hbcm2-4*fmb*
     . hbcm-3*fmc*hbcm-4*hbcm2)+8*(p3p7*p1p4)*(2*ffmcfmb*hbcm2-hbcm2)
     . +2*p5p7*(-2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*hbcm2+3*fmc*hbcm3+5*
     . fmc2*hbcm2-2*p3p4s2)+2*p4p7*(ffmcfmb*fmc*hbcm3-4*fmb*fmc*hbcm2
     . -4*fmc*hbcm3-5*fmc2*hbcm2)+4*(p3p7*p3p5)*(-7*ffmcfmb*fmc*hbcm+
     . 3*fmc*hbcm)+4*(p3p7*p2p3)*(7*ffmcfmb*fmc*hbcm-3*fmc*hbcm)+4*(
     . p3p7*p1p3)*(3*ffmcfmb*fmc*hbcm-fmc*hbcm)+2*(p3p7*p3p4)*(-
     . ffmcfmb**2*hbcm2-7*ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm-7*
     . ffmcfmb*hbcm2+3*fmb*hbcm+fmc*hbcm+3*hbcm2)+2*p3p7*(3*ffmcfmb**
     . 2*fmc*hbcm3-7*ffmcfmb*fmb*fmc*hbcm2-9*ffmcfmb*fmc*hbcm3-10*
     . ffmcfmb*fmc2*hbcm2+2*ffmcfmb*p3p4s2+3*fmb*fmc*hbcm2+3*fmc*
     . hbcm3+5*fmc2*hbcm2-2*p3p4s2)+ans4
      ans2=w1*ans3
      ans7=2*(p3p7*p3p4)*(-ffmcfmb**2*hbcm2-9*ffmcfmb*fmb*hbcm-6*
     . ffmcfmb*fmc*hbcm-7*ffmcfmb*hbcm2+5*fmb*hbcm+3*fmc*hbcm+3*hbcm2
     . )+4*(p3p7*p2p3)*(ffmcfmb**2*hbcm2+6*ffmcfmb*fmc*hbcm-3*fmc*
     . hbcm)+2*(p3p7*p1p3)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+6*
     . ffmcfmb*fmc*hbcm+7*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm-3*hbcm2)+
     . 2*p3p7*(-3*ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*hbcm4-6*ffmcfmb*fmb
     . *fmc*hbcm2+2*ffmcfmb*fmb*hbcm3-6*ffmcfmb*fmc*hbcm3-9*ffmcfmb*
     . fmc2*hbcm2-6*ffmcfmb*p1p3s2+2*ffmcfmb*p3p4s2+3*fmb*fmc*hbcm2+3
     . *fmc*hbcm3+5*fmc2*hbcm2+2*p1p3s2-2*p3p4s2)+4*(-ffmcfmb*hbcm2*
     . p2p5*p3p7+2*ffmcfmb*p1p3*p3p4*p3p7+hbcm2*p1p2*p5p7+2*hbcm2*
     . p1p4*p4p7-3*hbcm2*p1p4*p5p7+hbcm2*p1p5*p5p7+hbcm2*p2p4*p5p7-
     . hbcm2*p2p5*p4p7+hbcm2*p4p5*p5p7-4*p1p3*p2p3*p4p7+3*p1p3*p2p3*
     . p5p7+2*p1p3*p3p4*p4p7+4*p1p3*p3p5*p4p7-3*p1p3*p3p5*p5p7+4*p2p3
     . *p3p4*p4p7-3*p2p3*p3p4*p5p7-4*p3p4*p3p5*p4p7+3*p3p4*p3p5*p5p7)
      ans6=(p3p7*p3p5*p3p4)*(-28*ffmcfmb+12)+(p3p7*p3p4*p2p3)*(28*
     . ffmcfmb-12)+(p3p7*p3p5*p1p3)*(28*ffmcfmb-12)+(p3p7*p2p3*p1p3)*
     . (-28*ffmcfmb+12)+4*(p5p7*p3p5)*(ffmcfmb*hbcm2+3*fmc*hbcm)+4*(
     . p4p7*p3p5)*(-ffmcfmb*hbcm2-3*fmc*hbcm)+2*(p5p7*p3p4)*(-2*
     . ffmcfmb*hbcm2+5*fmb*hbcm+3*fmc*hbcm+5*hbcm2)+2*(p4p7*p3p4)*(-
     . ffmcfmb*hbcm2-4*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+4*(p5p7*p2p3)*(
     . ffmcfmb*hbcm2-3*fmc*hbcm)+4*(p4p7*p2p3)*(ffmcfmb*hbcm2+3*fmc*
     . hbcm)+4*(p3p7*p1p4)*(5*ffmcfmb*hbcm2-3*hbcm2)+2*(p5p7*p1p3)*(-
     . 2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm-hbcm2)+2*(p4p7*p1p3)*(
     . ffmcfmb*hbcm2+3*fmc*hbcm+4*hbcm2)+2*p4p7*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*fmc*hbcm3-4*fmc2*hbcm2-4*
     . p1p3s2)+2*p5p7*(-2*ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3+2*
     . ffmcfmb*hbcm4+3*fmb*fmc*hbcm2+3*fmc*hbcm3+5*fmc2*hbcm2+2*
     . p1p3s2-2*p3p4s2)+4*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2-6*ffmcfmb*
     . fmc*hbcm+3*fmc*hbcm)+ans7
      ans5=w5*ans6
      ans10=2*p3p7*(-3*ffmcfmb**2*fmb*hbcm3-6*ffmcfmb*fmb*fmc*hbcm2+9
     . *ffmcfmb*fmb*hbcm3-7*ffmcfmb*fmb2*hbcm2+14*ffmcfmb*p3p5s2+3*
     . fmb*fmc*hbcm2-3*fmb*hbcm3+3*fmb2*hbcm2-6*p3p5s2)+4*(-ffmcfmb*
     . hbcm2*p2p5*p3p7-2*fmb*hbcm*p1p3*p4p7-4*fmb*hbcm*p2p3*p4p7+
     . hbcm2*p1p2*p5p7+2*hbcm2*p1p5*p4p7+hbcm2*p2p4*p5p7-hbcm2*p2p5*
     . p4p7-2*p1p3*p3p5*p4p7+p1p3*p3p5*p5p7-4*p2p3*p3p5*p4p7+3*p2p3*
     . p3p5*p5p7+p3p4*p3p5*p5p7)
      ans9=(p3p7*p3p5*p3p4)*(-4*ffmcfmb+4)+(p3p7*p3p5*p2p3)*(-28*
     . ffmcfmb+12)+(p3p7*p3p5*p1p3)*(-12*ffmcfmb+4)+4*(p4p5*p3p7)*(
     . ffmcfmb*hbcm2-hbcm2)+2*(p5p7*p3p5)*(2*ffmcfmb*hbcm2-3*fmb*hbcm
     . +3*fmc*hbcm-3*hbcm2)+2*(p4p7*p3p5)*(-ffmcfmb*hbcm2+4*fmb*hbcm-
     . 3*fmc*hbcm+4*hbcm2)+4*(p5p7*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm+
     . hbcm2)+4*(p5p7*p2p3)*(ffmcfmb*hbcm2+3*fmb*hbcm)+4*(p3p7*p1p5)*
     . (3*ffmcfmb*hbcm2-hbcm2)+4*(p5p7*p1p3)*(-ffmcfmb*hbcm2+fmb*hbcm
     . +hbcm2)+2*p4p7*(-ffmcfmb*fmb*hbcm3-3*fmb*fmc*hbcm2+4*fmb*hbcm3
     . -4*fmb2*hbcm2+8*p3p5s2)+4*(p3p7*p3p4)*(-ffmcfmb*fmb*hbcm+fmb*
     . hbcm)+4*(p3p7*p2p3)*(-7*ffmcfmb*fmb*hbcm+3*fmb*hbcm)+4*(p3p7*
     . p1p3)*(-3*ffmcfmb*fmb*hbcm+fmb*hbcm)+2*p5p7*(-2*ffmcfmb**2*
     . hbcm4+2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*hbcm4+3*fmb*fmc*hbcm2-3*
     . fmb*hbcm3+3*fmb2*hbcm2-6*p3p5s2)+2*(p3p7*p3p5)*(-ffmcfmb**2*
     . hbcm2+7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm+7*ffmcfmb*hbcm2-3*
     . fmb*hbcm+3*fmc*hbcm-3*hbcm2)+ans10
      ans8=w2*ans9
      ans11=((p3p7*p3p5)*(56*fb2*ffmcfmb-24*fb2+14)+(p3p7*p3p4)*(-8*
     . fb2*ffmcfmb+8*fb2-2)+(p3p7*p2p3)*(-56*fb2*ffmcfmb+24*fb2-14)+(
     . p3p7*p1p3)*(-24*fb2*ffmcfmb+8*fb2-6)+4*p5p7*(-3*fb1*fmb*hbcm-3
     . *fb1*fmc*hbcm+2*fb2*ffmcfmb*hbcm2-3*fb2*hbcm2-7*hbcm2)+p4p7*(
     . 16*fb1*fmb*hbcm+12*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+16*fb2*
     . hbcm2+5*hbcm2)+p3p7*(28*fb1*ffmcfmb*fmb*hbcm+24*fb1*ffmcfmb*
     . fmc*hbcm-12*fb1*fmb*hbcm-12*fb1*fmc*hbcm-12*fb2*ffmcfmb**2*
     . hbcm2+36*fb2*ffmcfmb*hbcm2-12*fb2*hbcm2+12*ffmcfmb*hbcm2-11*
     . fmb*hbcm-6*fmc*hbcm-3*hbcm2)+8*(-2*fb2*p1p3*p4p7+fb2*p1p3*p5p7
     . -4*fb2*p2p3*p4p7+3*fb2*p2p3*p5p7+fb2*p3p4*p5p7+4*fb2*p3p5*p4p7
     . -3*fb2*p3p5*p5p7))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(5)=ans
      ans4=4*(-2*ffmcfmb*fmc*hbcm*p1p3*p3p6+2*ffmcfmb*hbcm2*p2p4*p3p6
     . -ffmcfmb*hbcm2*p3p5*p4p6+2*ffmcfmb*hbcm2*p3p6*p4p5-2*ffmcfmb*
     . p1p3*p3p4*p3p6+2*fmc*hbcm*p2p3*p5p6-2*fmc*hbcm*p3p5*p5p6-hbcm2
     . *p1p2*p4p6+2*hbcm2*p1p4*p5p6-hbcm2*p1p5*p4p6+hbcm2*p2p4*p4p6+
     . hbcm2*p4p5*p4p6-2*p1p3*p3p4*p4p6-4*p2p3*p3p4*p4p6+2*p2p3*p3p4*
     . p5p6-2*p3p4*p3p5*p5p6)
      ans3=(p3p6*p3p5*p3p4)*(8*ffmcfmb-8)+(p3p6*p3p4*p2p3)*(-24*
     . ffmcfmb+8)+2*(p5p6*p3p4)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-2
     . *hbcm2)+4*(p4p6*p3p4)*(ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2)+4*(p4p6
     . *p2p3)*(-ffmcfmb*hbcm2-4*fmc*hbcm)+8*(p3p6*p1p4)*(-ffmcfmb*
     . hbcm2+hbcm2)+4*(p4p6*p1p3)*(ffmcfmb*hbcm2-2*fmc*hbcm-hbcm2)+2*
     . p5p6*(ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-2*fmc*hbcm3-3*fmc2*
     . hbcm2)+8*(p3p6*p3p5)*(ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(p3p6*p2p3)
     . *(-3*ffmcfmb*fmc*hbcm+fmc*hbcm)+4*p4p6*(ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+2*p3p4s2)+2*(p3p6*p3p4)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+5*ffmcfmb*fmc*hbcm+5*
     . ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-2*hbcm2)+2*p3p6*(3*ffmcfmb**
     . 2*fmc*hbcm3+2*ffmcfmb*fmb*fmc*hbcm2+3*ffmcfmb*fmc*hbcm3+3*
     . ffmcfmb*fmc2*hbcm2+4*ffmcfmb*p3p4s2-2*fmb*fmc*hbcm2-2*fmc*
     . hbcm3-3*fmc2*hbcm2)+ans4
      ans2=w1*ans3
      ans7=2*(p3p6*p1p3)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+
     . 2*p3p6*(-3*ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*hbcm4+ffmcfmb*fmb*
     . fmc*hbcm2-ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*
     . hbcm3+ffmcfmb*hbcm4+4*ffmcfmb*p1p3s2+4*ffmcfmb*p3p4s2-fmb*fmc*
     . hbcm2-fmc*hbcm3-2*fmc2*hbcm2)+8*(-ffmcfmb*hbcm2*p1p2*p3p6-
     . ffmcfmb*hbcm2*p1p5*p3p6+ffmcfmb*hbcm2*p2p4*p3p6+2*ffmcfmb*
     . hbcm2*p2p5*p3p6+ffmcfmb*hbcm2*p3p6*p4p5-2*ffmcfmb*p1p3*p3p4*
     . p3p6-hbcm2*p1p2*p4p6+hbcm2*p1p4*p3p6+hbcm2*p1p4*p4p6+hbcm2*
     . p1p4*p5p6-hbcm2*p1p5*p4p6+hbcm2*p2p4*p4p6+2*hbcm2*p2p5*p4p6+
     . hbcm2*p3p5*p4p6+hbcm2*p4p5*p4p6+2*p1p3*p2p3*p4p6-p1p3*p2p3*
     . p5p6-2*p1p3*p3p4*p4p6+p1p3*p3p5*p5p6-2*p2p3*p3p4*p4p6+p2p3*
     . p3p4*p5p6-p3p4*p3p5*p5p6)
      ans6=8*(p4p6*p3p4)*(-fmb*hbcm+hbcm2)+8*(p4p6*p2p3)*(2*fmb*hbcm-
     . hbcm2)+8*(p4p6*p1p3)*(fmb*hbcm-hbcm2)+(p3p6*p3p5*p3p4)*(8*
     . ffmcfmb-8)+(p3p6*p3p4*p2p3)*(-24*ffmcfmb+8)+(p3p6*p3p5*p1p3)*(
     . -8*ffmcfmb+8)+(p3p6*p2p3*p1p3)*(24*ffmcfmb-8)+4*(p5p6*p3p5)*(-
     . ffmcfmb*hbcm2-fmc*hbcm)+2*(p5p6*p3p4)*(-ffmcfmb*hbcm2-2*fmb*
     . hbcm-fmc*hbcm-2*hbcm2)+4*(p5p6*p2p3)*(ffmcfmb*hbcm2+fmc*hbcm)+
     . 2*(p5p6*p1p3)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+2*
     . p5p6*(-ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2-fmc*hbcm3
     . -2*fmc2*hbcm2)+4*p4p6*(-2*ffmcfmb*fmb*hbcm3+fmb2*hbcm2-fmc2*
     . hbcm2+hbcm4+2*p1p3s2+2*p3p4s2)+4*(p3p6*p3p5)*(ffmcfmb**2*hbcm2
     . +ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+2*(p3p6*p3p4)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+5*ffmcfmb
     . *hbcm2-2*fmb*hbcm-fmc*hbcm-2*hbcm2)+4*(p3p6*p2p3)*(-ffmcfmb**2
     . *hbcm2+4*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*
     . hbcm)+ans7
      ans5=w5*ans6
      ans9=((p3p6*p3p5)*(-16*fb2*ffmcfmb+16*fb2-4)+(p3p6*p3p4)*(-16*
     . fb2*ffmcfmb-4)+(p3p6*p2p3)*(48*fb2*ffmcfmb-16*fb2+12)+(p3p6*
     . p1p3)*(16*fb2*ffmcfmb+4)+4*p4p6*(-4*fb2*ffmcfmb*hbcm2-hbcm2)+
     . p5p6*(8*fb1*fmb*hbcm+4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2+8*fb2*
     . hbcm2-hbcm2)+p3p6*(-8*fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*
     . hbcm+8*fb1*fmb*hbcm+4*fb1*fmc*hbcm-12*fb2*ffmcfmb**2*hbcm2-12*
     . fb2*ffmcfmb*hbcm2+8*fb2*hbcm2+12*ffmcfmb*hbcm2-10*fmb*hbcm-5*
     . fmc*hbcm-3*hbcm2)+16*(fb2*p1p3*p4p6+2*fb2*p2p3*p4p6-fb2*p2p3*
     . p5p6-fb2*p3p4*p4p6+fb2*p3p5*p5p6))
      ans8=w2*((p3p6*p3p5*p2p3)*(24*ffmcfmb-8)+2*(p5p6*p3p5)*(-
     . ffmcfmb*hbcm2+2*fmb*hbcm-fmc*hbcm+2*hbcm2)+2*p5p6*(-ffmcfmb*
     . fmb*hbcm3-fmb*fmc*hbcm2+2*fmb*hbcm3-2*fmb2*hbcm2+4*p3p5s2)+8*(
     . p3p6*p2p3)*(3*ffmcfmb*fmb*hbcm-fmb*hbcm)+2*(p3p6*p3p5)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-3*ffmcfmb
     . *hbcm2+2*fmb*hbcm-fmc*hbcm+2*hbcm2)+2*p3p6*(-3*ffmcfmb**2*fmb*
     . hbcm3+ffmcfmb*fmb*fmc*hbcm2-3*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmb2
     . *hbcm2-4*ffmcfmb*p3p5s2-fmb*fmc*hbcm2+2*fmb*hbcm3-2*fmb2*hbcm2
     . +4*p3p5s2)+8*(ffmcfmb*fmb*hbcm*p1p3*p3p6-ffmcfmb*fmb*hbcm*p3p4
     . *p3p6-ffmcfmb*fmb*hbcm3*p4p6-ffmcfmb*hbcm2*p1p5*p3p6+ffmcfmb*
     . hbcm2*p2p5*p3p6+ffmcfmb*hbcm2*p3p6*p4p5+ffmcfmb*p1p3*p3p5*p3p6
     . -ffmcfmb*p3p4*p3p5*p3p6+fmb*hbcm*p1p3*p4p6+2*fmb*hbcm*p2p3*
     . p4p6-fmb*hbcm*p2p3*p5p6-fmb*hbcm*p3p4*p4p6-hbcm2*p1p5*p4p6+
     . hbcm2*p2p5*p4p6+hbcm2*p4p5*p4p6+p1p3*p3p5*p4p6+2*p2p3*p3p5*
     . p4p6-p2p3*p3p5*p5p6-p3p4*p3p5*p4p6))+ans9
      ans1=ans2+ans5+ans8
      ans=ccc*ans1
      b(6)=ans
      ans2=w2*((p3p5*p3p4)*(2*ffmcfmb-2)+(p3p5*p1p3)*(2*ffmcfmb-2)+2*
     . p3p4*(ffmcfmb*fmb*hbcm-fmb*hbcm)+2*p1p3*(ffmcfmb*fmb*hbcm-fmb*
     . hbcm)+2*p3p5*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)
     . +2*(ffmcfmb**2*fmb*hbcm3-ffmcfmb*fmb*hbcm*p2p3-ffmcfmb*fmb*
     . hbcm3-ffmcfmb*p2p3*p3p5-ffmcfmb*p3p5s2-fmb*hbcm*p1p2-fmb*hbcm*
     . p1p5-fmb*hbcm*p2p4-fmb*hbcm*p4p5-p1p2*p3p5-p1p5*p3p5-p2p4*p3p5
     . -p3p5*p4p5))+(p3p5*(-4*fb2*ffmcfmb-1)+4*p3p4*(fb2*ffmcfmb-fb2)
     . +p2p3*(-4*fb2*ffmcfmb-1)+4*p1p3*(fb2*ffmcfmb-fb2)+(4*fb2*
     . ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2-4*fb2*p1p2-4*fb2*p1p5-4*
     . fb2*p2p4-4*fb2*p4p5+2*ffmcfmb*hbcm2+9*fmb*hbcm+9*fmc*hbcm-10*
     . hbcm2+p1p3+p3p4))
      ans1=w1*((p3p4*p1p3)*(-2*ffmcfmb+2)+2*p1p3*(-ffmcfmb*fmc*hbcm+
     . fmc*hbcm)+2*p3p4*(-ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2+fmc*hbcm)+2*(-ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmc*hbcm*p2p3
     . +ffmcfmb*fmc*hbcm*p3p5+ffmcfmb*fmc*hbcm3+ffmcfmb*p2p3*p3p4+
     . ffmcfmb*p3p4*p3p5-ffmcfmb*p3p4s2+fmc*hbcm*p1p2+fmc*hbcm*p1p5+
     . fmc*hbcm*p2p4+fmc*hbcm*p4p5+p1p2*p3p4+p1p5*p3p4+p2p4*p3p4+p3p4
     . *p4p5+p3p4s2))+w5*(2*p4p5*(-fmb*hbcm+hbcm2)+2*p2p4*(-fmb*hbcm+
     . hbcm2)+2*p1p5*(-fmb*hbcm+hbcm2)+2*p1p2*(-fmb*hbcm+hbcm2)+2*
     . p3p5*(-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+2*p2p3*(-ffmcfmb*fmb*
     . hbcm+ffmcfmb*hbcm2)+2*p3p4*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm
     . -fmb*hbcm+hbcm2)+2*p1p3*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+2*(ffmcfmb**2*fmb*hbcm3-ffmcfmb
     . **2*hbcm4-ffmcfmb*fmb*hbcm3+ffmcfmb*hbcm4-ffmcfmb*p1p3*p2p3-
     . ffmcfmb*p1p3*p3p5+ffmcfmb*p1p3s2+ffmcfmb*p2p3*p3p4+ffmcfmb*
     . p3p4*p3p5-ffmcfmb*p3p4s2-p1p2*p1p3+p1p2*p3p4-p1p3*p1p5-p1p3*
     . p2p4-p1p3*p4p5-p1p3s2+p1p5*p3p4+p2p4*p3p4+p3p4*p4p5+p3p4s2))+
     . ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=w2*(2*(p3p5*p3p4)*(-ffmcfmb*hbcm+hbcm)+2*(p3p5*p1p3)*(-
     . ffmcfmb*hbcm+hbcm)+2*p3p4*(-ffmcfmb*fmb*hbcm2+fmb*hbcm2)+2*
     . p1p3*(-ffmcfmb*fmb*hbcm2+fmb*hbcm2)+2*p3p5*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3)+2*(-ffmcfmb**2*fmb*hbcm4+
     . ffmcfmb*fmb*hbcm2*p2p3+ffmcfmb*fmb*hbcm4+ffmcfmb*hbcm*p2p3*
     . p3p5+ffmcfmb*hbcm*p3p5s2+fmb*hbcm2*p1p2+fmb*hbcm2*p1p5+fmb*
     . hbcm2*p2p4+fmb*hbcm2*p4p5+hbcm*p1p2*p3p5+hbcm*p1p5*p3p5+hbcm*
     . p2p4*p3p5+hbcm*p3p5*p4p5))+(p3p5*(-4*fb1*ffmcfmb*hbcm-19*hbcm)
     . +p3p4*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm-hbcm)+p2p3*(-4*fb1*
     . ffmcfmb*hbcm+29*hbcm)+p1p3*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+11*
     . hbcm)+(4*fb1*ffmcfmb**2*hbcm3-4*fb1*ffmcfmb*hbcm3-4*fb1*hbcm*
     . p1p2-4*fb1*hbcm*p1p5-4*fb1*hbcm*p2p4-4*fb1*hbcm*p4p5-9*fmb*
     . hbcm2+9*fmc*hbcm2-10*hbcm3))
      ans2=w5*(2*p4p5*(fmb*hbcm2+hbcm3)+2*p2p4*(fmb*hbcm2+hbcm3)+2*
     . p1p5*(fmb*hbcm2+hbcm3)+2*p1p2*(fmb*hbcm2+hbcm3)+2*(p3p5*p3p4)*
     . (-ffmcfmb*hbcm+2*hbcm)+2*(p3p4*p2p3)*(3*ffmcfmb*hbcm-2*hbcm)+2
     . *(p3p5*p1p3)*(-3*ffmcfmb*hbcm+2*hbcm)+2*(p2p3*p1p3)*(ffmcfmb*
     . hbcm-2*hbcm)+2*p3p5*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+3*
     . ffmcfmb*hbcm3)+2*p3p4*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+fmb
     . *hbcm2+hbcm3)+2*p2p3*(2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3)+2*p1p3*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+2*(-ffmcfmb**2*fmb*hbcm4-
     . ffmcfmb**2*hbcm5+ffmcfmb*fmb*hbcm4+ffmcfmb*hbcm*p1p3s2-2*
     . ffmcfmb*hbcm*p2p3s2-ffmcfmb*hbcm*p3p4s2+2*ffmcfmb*hbcm*p3p5s2+
     . ffmcfmb*hbcm5-hbcm*p1p2*p1p3-2*hbcm*p1p2*p2p3+hbcm*p1p2*p3p4+2
     . *hbcm*p1p2*p3p5-hbcm*p1p3*p1p5-hbcm*p1p3*p2p4-hbcm*p1p3*p4p5-
     . hbcm*p1p3s2-2*hbcm*p1p5*p2p3+hbcm*p1p5*p3p4+2*hbcm*p1p5*p3p5-2
     . *hbcm*p2p3*p2p4-2*hbcm*p2p3*p4p5+hbcm*p2p4*p3p4+2*hbcm*p2p4*
     . p3p5+hbcm*p3p4*p4p5+hbcm*p3p4s2+2*hbcm*p3p5*p4p5))+ans3
      ans1=w1*(2*(p3p4*p1p3)*(-ffmcfmb*hbcm+hbcm)+2*p1p3*(-ffmcfmb*
     . fmc*hbcm2+fmc*hbcm2)+2*p3p4*(-ffmcfmb**2*hbcm3-ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmc*hbcm2)+2*(-ffmcfmb**2*fmc*hbcm4+
     . ffmcfmb*fmc*hbcm2*p2p3+ffmcfmb*fmc*hbcm2*p3p5+ffmcfmb*fmc*
     . hbcm4+ffmcfmb*hbcm*p2p3*p3p4+ffmcfmb*hbcm*p3p4*p3p5-ffmcfmb*
     . hbcm*p3p4s2+fmc*hbcm2*p1p2+fmc*hbcm2*p1p5+fmc*hbcm2*p2p4+fmc*
     . hbcm2*p4p5+hbcm*p1p2*p3p4+hbcm*p1p5*p3p4+hbcm*p2p4*p3p4+hbcm*
     . p3p4*p4p5+hbcm*p3p4s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(4*(p3p6*p3p4)*(-4*ffmcfmb*hbcm+hbcm)+4*p3p6*(-4*
     . ffmcfmb*fmc*hbcm2+fmc*hbcm2)+4*(-3*fmc*hbcm2*p4p6+fmc*hbcm2*
     . p5p6-3*hbcm*p3p4*p4p6+hbcm*p3p4*p5p6))+w2*(4*(p3p6*p3p5)*(-4*
     . ffmcfmb*hbcm+hbcm)+4*p3p6*(-4*ffmcfmb*fmb*hbcm2+fmb*hbcm2)+4*(
     . -3*fmb*hbcm2*p4p6+fmb*hbcm2*p5p6-3*hbcm*p3p5*p4p6+hbcm*p3p5*
     . p5p6))+w5*(2*p5p6*(ffmcfmb*hbcm3+fmc*hbcm2)+4*p4p6*(2*ffmcfmb*
     . hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(p3p6*p3p4)*(-2*ffmcfmb*hbcm+hbcm
     . )+4*(p3p6*p1p3)*(2*ffmcfmb*hbcm-hbcm)+2*p3p6*(3*ffmcfmb**2*
     . hbcm3-6*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-5*ffmcfmb*hbcm3+
     . fmc*hbcm2)+4*(2*ffmcfmb*hbcm*p2p3*p3p6-6*ffmcfmb*hbcm*p3p5*
     . p3p6+hbcm*p1p3*p4p6-hbcm*p1p3*p5p6+2*hbcm*p2p3*p4p6-hbcm*p3p4*
     . p4p6+hbcm*p3p4*p5p6-6*hbcm*p3p5*p4p6))+(8*p3p6*(4*fb1*ffmcfmb*
     . hbcm-fb1*hbcm+2*hbcm)+8*(3*fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(9)=ans
      b(10)=ccc*(w1*((p3p6*p3p4)*(16*ffmcfmb-4)+4*p3p6*(4*ffmcfmb*fmc
     . *hbcm-fmc*hbcm)+4*(3*fmc*hbcm*p4p6-fmc*hbcm*p5p6+3*p3p4*p4p6-
     . p3p4*p5p6))+w2*((p3p6*p3p5)*(-16*ffmcfmb+4)+4*p3p6*(-4*ffmcfmb
     . *fmb*hbcm+fmb*hbcm)+4*(-3*fmb*hbcm*p4p6+fmb*hbcm*p5p6-3*p3p5*
     . p4p6+p3p5*p5p6))+w5*(12*p4p6*(-fmb*hbcm+hbcm2)+(p3p6*p3p4)*(16
     . *ffmcfmb-4)+(p3p6*p1p3)*(-16*ffmcfmb+4)+2*p5p6*(-ffmcfmb*hbcm2
     . -fmc*hbcm)+2*p3p6*(ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm+ffmcfmb
     . *fmc*hbcm+5*ffmcfmb*hbcm2-fmc*hbcm)+4*(-3*p1p3*p4p6+p1p3*p5p6+
     . 3*p3p4*p4p6-p3p4*p5p6))+(p3p6*(-32*fb2*ffmcfmb+8*fb2-8)+8*(-3*
     . fb2*p4p6+fb2*p5p6)))
      b(11)=ccc*(w5*(2*p3p4*(ffmcfmb*hbcm2-hbcm2)+2*p1p3*(ffmcfmb*
     . hbcm2-hbcm2)+2*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm2*p2p3-ffmcfmb*
     . hbcm2*p3p5-ffmcfmb*hbcm4-hbcm2*p1p2-hbcm2*p1p5-hbcm2*p2p4-
     . hbcm2*p4p5))+18*hbcm2)
      ans=ccc*(w1*(2*(p3p7*p3p4)*(-8*ffmcfmb*hbcm+3*hbcm)+2*p3p7*(-8*
     . ffmcfmb*fmc*hbcm2+3*fmc*hbcm2)+2*(-5*fmc*hbcm2*p4p7+3*fmc*
     . hbcm2*p5p7-5*hbcm*p3p4*p4p7+3*hbcm*p3p4*p5p7))+w2*(2*(p3p7*
     . p3p5)*(-8*ffmcfmb*hbcm+3*hbcm)+2*p3p7*(-8*ffmcfmb*fmb*hbcm2+3*
     . fmb*hbcm2)+2*(-5*fmb*hbcm2*p4p7+3*fmb*hbcm2*p5p7-5*hbcm*p3p5*
     . p4p7+3*hbcm*p3p5*p5p7))+w5*(2*p5p7*(2*ffmcfmb*hbcm3+3*fmc*
     . hbcm2)+2*p4p7*(-ffmcfmb*hbcm3-fmb*hbcm2-3*fmc*hbcm2-hbcm3)+10*
     . (p3p7*p3p4)*(-2*ffmcfmb*hbcm+hbcm)+2*(p3p7*p1p3)*(2*ffmcfmb*
     . hbcm-hbcm)+2*p3p7*(-3*ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-6*
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+3*fmc*hbcm2)+2*(2*ffmcfmb*hbcm
     . *p2p3*p3p7-2*ffmcfmb*hbcm*p3p5*p3p7+hbcm*p1p3*p4p7-hbcm*p1p3*
     . p5p7+2*hbcm*p2p3*p4p7-5*hbcm*p3p4*p4p7+5*hbcm*p3p4*p5p7-2*hbcm
     . *p3p5*p4p7))+(4*p3p7*(8*fb1*ffmcfmb*hbcm-3*fb1*hbcm-4*hbcm)+4*
     . (5*fb1*hbcm*p4p7-3*fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w5*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p1p3*(-ffmcfmb*
     . hbcm+hbcm)+2*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p2p3+ffmcfmb*hbcm
     . *p3p5+ffmcfmb*hbcm3+hbcm*p1p2+hbcm*p1p5+hbcm*p2p4+hbcm*p4p5))-
     . 18*hbcm)
      b(14)=ccc*(w1*((p3p7*p3p4)*(-16*ffmcfmb+6)+2*p3p7*(-8*ffmcfmb*
     . fmc*hbcm+3*fmc*hbcm)+2*(-5*fmc*hbcm*p4p7+3*fmc*hbcm*p5p7-5*
     . p3p4*p4p7+3*p3p4*p5p7))+w2*((p3p7*p3p5)*(16*ffmcfmb-6)+2*p3p7*
     . (8*ffmcfmb*fmb*hbcm-3*fmb*hbcm)+2*(5*fmb*hbcm*p4p7-3*fmb*hbcm*
     . p5p7+5*p3p5*p4p7-3*p3p5*p5p7))+w5*((p3p7*p3p4)*(-16*ffmcfmb+6)
     . +(p3p7*p1p3)*(16*ffmcfmb-6)+2*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm-3*
     . fmc*hbcm-hbcm2)+2*p3p7*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-6*
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+3*fmc*hbcm)+2*(3*fmc*hbcm*p5p7+
     . 5*p1p3*p4p7-3*p1p3*p5p7-5*p3p4*p4p7+3*p3p4*p5p7))+(p3p7*(32*
     . fb2*ffmcfmb-12*fb2+8)+4*(5*fb2*p4p7-3*fb2*p5p7)))
      ans2=w5*((p3p7*p3p6*p3p4)*(32*ffmcfmb-8)+(p3p7*p3p6*p1p3)*(-32*
     . ffmcfmb+8)+8*(p4p6*p3p7)*(ffmcfmb*hbcm2-fmb*hbcm)+4*(p5p7*p3p6
     . )*(-5*ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p7*p3p6)*(-3*ffmcfmb*hbcm2+
     . fmc*hbcm)+4*(p6p7*p3p4)*(-2*ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-
     . hbcm2)+4*(p6p7*p1p3)*(2*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm+hbcm2)
     . +4*(p3p7*p3p6)*(-2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb
     . *hbcm2-fmc*hbcm)+2*p6p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2-fmc*hbcm3+4*fmc2
     . *hbcm2-8*p1p3s2-8*p3p4s2)+8*(-2*hbcm2*p1p4*p6p7-hbcm2*p4p6*
     . p4p7-3*hbcm2*p4p6*p5p7+4*p1p3*p3p4*p6p7-p1p3*p3p6*p4p7+p1p3*
     . p3p6*p5p7-2*p1p3*p3p7*p4p6+p3p4*p3p6*p4p7-p3p4*p3p6*p5p7+2*
     . p3p4*p3p7*p4p6))+((p3p7*p3p6)*(-64*fb2*ffmcfmb+16*fb2-16)+4*
     . p6p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2+2*fb2
     . *hbcm2+hbcm2)+16*(-2*fb2*p1p3*p6p7+2*fb2*p3p4*p6p7-fb2*p3p6*
     . p4p7+fb2*p3p6*p5p7-2*fb2*p3p7*p4p6))
      ans1=w1*(4*(p6p7*p3p4)*(fmb*hbcm-3*fmc*hbcm-hbcm2)+(p3p7*p3p6*
     . p3p4)*(32*ffmcfmb-8)+4*(p4p6*p3p7)*(4*ffmcfmb*hbcm2+4*fmc*hbcm
     . -hbcm2)+8*(p4p7*p3p6)*(-2*ffmcfmb*hbcm2+fmc*hbcm)+4*p6p7*(-2*
     . ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmc*hbcm3+3*fmc2*hbcm2-4*
     . p3p4s2)+8*(p3p7*p3p6)*(4*ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(4*fmc*
     . hbcm*p1p3*p6p7-2*fmc*hbcm*p3p6*p5p7-4*hbcm2*p1p4*p6p7-hbcm2*
     . p4p6*p4p7-hbcm2*p4p6*p5p7+4*p1p3*p3p4*p6p7+2*p3p4*p3p6*p4p7-2*
     . p3p4*p3p6*p5p7+4*p3p4*p3p7*p4p6))+w2*(4*(p6p7*p3p5)*(fmb*hbcm+
     . fmc*hbcm+hbcm2)+(p3p7*p3p6*p3p5)*(-32*ffmcfmb+8)+8*(p5p6*p3p7)
     . *(2*ffmcfmb*hbcm2-hbcm2)+8*(p5p7*p3p6)*(-2*ffmcfmb*hbcm2+fmb*
     . hbcm+hbcm2)+4*p6p7*(2*ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2+fmb*
     . hbcm3+fmb2*hbcm2)+8*(p3p7*p3p6)*(-4*ffmcfmb*fmb*hbcm+fmb*hbcm)
     . +8*(-2*fmb*hbcm*p1p3*p6p7+2*fmb*hbcm*p3p4*p6p7-fmb*hbcm*p3p6*
     . p4p7-2*fmb*hbcm*p3p7*p4p6+hbcm2*p1p5*p6p7-hbcm2*p2p5*p6p7-
     . hbcm2*p4p5*p6p7-hbcm2*p4p6*p5p7+hbcm2*p4p7*p5p6-2*p1p3*p3p5*
     . p6p7+2*p3p4*p3p5*p6p7-p3p5*p3p6*p4p7+p3p5*p3p6*p5p7-2*p3p5*
     . p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w2*(4*p6p7*(-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+8*(p5p6
     . *p3p7)*(2*ffmcfmb*hbcm-hbcm)+8*(p5p7*p3p6)*(-2*ffmcfmb*hbcm+
     . hbcm)+4*(p6p7*p3p5)*(-2*ffmcfmb*hbcm+fmb-fmc+hbcm)+8*(hbcm*
     . p1p5*p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7-hbcm*p4p6*p5p7+hbcm*
     . p4p7*p5p6))+w1*(4*p6p7*(-fmb*fmc*hbcm+fmc*hbcm2+3*fmc2*hbcm)+4
     . *(p4p6*p3p7)*(4*ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4)*(2*ffmcfmb*
     . hbcm-fmb+fmc+hbcm)+4*(-4*ffmcfmb*hbcm*p3p6*p4p7-4*hbcm*p1p4*
     . p6p7-hbcm*p4p6*p4p7-hbcm*p4p6*p5p7))+w5*(16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4)*(2*ffmcfmb*hbcm-fmb+fmc+hbcm)
     . +4*(p6p7*p1p3)*(-2*ffmcfmb*hbcm+fmb-fmc-hbcm)+2*p6p7*(ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-
     . fmb*fmc*hbcm+fmc*hbcm2+4*fmc2*hbcm)+8*(-ffmcfmb*hbcm*p3p6*p3p7
     . -ffmcfmb*hbcm*p3p6*p4p7-3*ffmcfmb*hbcm*p3p6*p5p7-2*hbcm*p1p4*
     . p6p7-4*hbcm*p4p6*p5p7))+4*p6p7*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*
     . fmc+hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)+0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp35_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      ans5=4*(ffmcfmb*fmc*hbcm2*p3p6*p5p7-2*ffmcfmb*hbcm*p1p2*p3p6*
     . p3p7+2*ffmcfmb*hbcm*p1p3*p3p6*p4p7-4*ffmcfmb*hbcm*p1p3*p3p7*
     . p4p6+ffmcfmb*hbcm*p2p3*p3p6*p5p7+2*ffmcfmb*hbcm*p2p5*p3p6*p3p7
     . +ffmcfmb*hbcm*p3p4*p3p6*p5p7-2*ffmcfmb*hbcm*p3p5*p3p6*p4p7+4*
     . ffmcfmb*hbcm*p3p5*p3p7*p4p6-ffmcfmb*hbcm3*p2p5*p6p7+ffmcfmb*
     . hbcm3*p4p7*p5p6+fmc*hbcm2*p1p4*p6p7+fmc*hbcm2*p4p5*p6p7+fmc*
     . hbcm2*p4p6*p5p7+fmc*p1p3*p2p3*p6p7+fmc*p3p4*p3p5*p6p7+fmc2*
     . hbcm*p1p3*p6p7-hbcm*p1p2*p2p3*p6p7+3*hbcm*p1p2*p3p4*p6p7-4*
     . hbcm*p1p2*p3p6*p4p7+2*hbcm*p1p2*p3p7*p4p6-4*hbcm*p1p3*p2p4*
     . p6p7-2*hbcm*p1p3*p4p6*p4p7+hbcm*p1p4*p2p3*p6p7+hbcm*p1p4*p3p4*
     . p6p7-6*hbcm*p2p3*p2p4*p6p7+3*hbcm*p2p3*p2p5*p6p7+hbcm*p2p3*
     . p4p5*p6p7+4*hbcm*p2p3*p4p6*p4p7+hbcm*p2p3*p4p6*p5p7-3*hbcm*
     . p2p3*p4p7*p5p6-2*hbcm*p2p4*p3p4*p6p7+4*hbcm*p2p4*p3p5*p6p7-4*
     . hbcm*p2p4*p3p6*p4p7+2*hbcm*p2p4*p3p7*p5p6-hbcm*p2p5*p3p4*p6p7+
     . 4*hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*p4p6+hbcm*p3p4*p4p5*
     . p6p7+2*hbcm*p3p4*p4p6*p4p7+hbcm*p3p4*p4p6*p5p7-hbcm*p3p4*p4p7*
     . p5p6+2*hbcm*p3p5*p4p6*p4p7)
      ans4=2*(p6p7*p3p4)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-3*
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+3*fmc*hbcm2)+2*(
     . p6p7*p2p3)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+3*ffmcfmb*hbcm3-3*fmb*fmc*hbcm+fmc*hbcm2+8*fmc2*hbcm)+4*
     . (p3p7*p3p6*p3p4)*(3*ffmcfmb**2*hbcm-ffmcfmb*fmb+4*ffmcfmb*fmc+
     . 3*ffmcfmb*hbcm-fmc)+4*(p3p7*p3p6*p2p3)*(-3*ffmcfmb**2*hbcm+
     . ffmcfmb*fmb-4*ffmcfmb*fmc+5*ffmcfmb*hbcm+fmc)+2*p6p7*(-3*
     . ffmcfmb**2*fmc*hbcm4+ffmcfmb*fmb*fmc*hbcm3+3*ffmcfmb*fmc*hbcm4
     . -ffmcfmb*fmc2*hbcm3-8*ffmcfmb*hbcm*p2p3s2-fmb*fmc2*hbcm2+4*fmb
     . *p2p3s2-6*fmc*p2p3s2-2*fmc*p3p4s2-fmc2*hbcm3+2*fmc3*hbcm2+2*
     . hbcm*p2p3s2+2*hbcm*p3p4s2)+4*(p3p7*p3p6)*(-ffmcfmb**2*fmc*
     . hbcm2-ffmcfmb*fmb*fmc*hbcm+5*ffmcfmb*fmc*hbcm2+4*ffmcfmb*fmc2*
     . hbcm-fmc2*hbcm)+ans5
      ans3=8*(p6p7*p3p4*p2p3)*(-fmb+2*fmc+hbcm)+4*(p4p7*p4p6)*(-2*
     . ffmcfmb*hbcm3+fmb*hbcm2+fmc*hbcm2+hbcm3)+4*(p6p7*p2p4)*(-
     . ffmcfmb*hbcm3+2*fmb*hbcm2-3*fmc*hbcm2+2*hbcm3)+4*(p6p7*p1p2)*(
     . ffmcfmb*hbcm3+2*fmc*hbcm2)+4*(p5p6*p3p7*p3p4)*(-3*ffmcfmb*hbcm
     . -fmc)+4*(p4p6*p3p7*p3p4)*(2*ffmcfmb*hbcm+fmb-fmc)+4*(p4p7*p3p6
     . *p3p4)*(ffmcfmb*hbcm-2*fmb+4*fmc+5*hbcm)+8*(p3p7*p3p6*p2p4)*(-
     . 3*ffmcfmb*hbcm+hbcm)+4*(p5p6*p3p7*p2p3)*(-ffmcfmb*hbcm+fmc)+4*
     . (p4p6*p3p7*p2p3)*(2*ffmcfmb*hbcm-fmb+fmc)+4*(p4p7*p3p6*p2p3)*(
     . ffmcfmb*hbcm+2*fmb-4*fmc+3*hbcm)+4*(p6p7*p3p5*p2p3)*(2*ffmcfmb
     . *hbcm-fmc)+4*(p6p7*p3p4*p1p3)*(2*ffmcfmb*hbcm-fmc)+4*(p5p6*
     . p3p7)*(-ffmcfmb*fmc*hbcm2-fmc2*hbcm)+4*(p6p7*p3p5)*(2*ffmcfmb*
     . fmc*hbcm2-fmc2*hbcm)+4*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc2*hbcm)+4*(
     . p4p7*p3p6)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2-2*fmb*fmc*hbcm+6*fmc*hbcm2+4*fmc2*hbcm)+ans4
      ans2=w11*ans3
      ans8=2*p6p7*(-3*ffmcfmb**2*fmc*hbcm4+ffmcfmb*fmb*fmc*hbcm3+3*
     . ffmcfmb*fmc*hbcm4-ffmcfmb*fmc2*hbcm3-fmb*fmc2*hbcm2-2*fmc*
     . p3p4s2-fmc2*hbcm3+2*fmc3*hbcm2+2*hbcm*p3p4s2)+4*(p3p7*p3p6)*(-
     . ffmcfmb**2*fmc*hbcm2-ffmcfmb*fmb*fmc*hbcm+5*ffmcfmb*fmc*hbcm2+
     . 4*ffmcfmb*fmc2*hbcm-fmc2*hbcm)+4*(ffmcfmb*fmc*hbcm2*p3p6*p5p7+
     . 2*ffmcfmb*hbcm*p1p3*p3p6*p4p7-4*ffmcfmb*hbcm*p1p3*p3p7*p4p6+2*
     . ffmcfmb*hbcm*p2p3*p3p6*p4p7-4*ffmcfmb*hbcm*p2p3*p3p7*p4p6+
     . ffmcfmb*hbcm*p3p4*p3p6*p5p7-2*ffmcfmb*hbcm*p3p5*p3p6*p4p7+4*
     . ffmcfmb*hbcm*p3p5*p3p7*p4p6+ffmcfmb*hbcm3*p4p7*p5p6+3*fmc*
     . hbcm2*p1p2*p6p7+fmc*hbcm2*p1p4*p6p7-fmc*hbcm2*p2p5*p6p7+fmc*
     . hbcm2*p4p5*p6p7+fmc*hbcm2*p4p6*p5p7+fmc*p3p4*p3p5*p6p7+fmc2*
     . hbcm*p1p3*p6p7+3*hbcm*p1p2*p3p4*p6p7-4*hbcm*p1p3*p2p4*p6p7-2*
     . hbcm*p1p3*p4p6*p4p7+hbcm*p1p4*p3p4*p6p7-4*hbcm*p2p3*p2p4*p6p7-
     . 2*hbcm*p2p3*p4p6*p4p7-2*hbcm*p2p4*p3p4*p6p7+4*hbcm*p2p4*p3p5*
     . p6p7-hbcm*p2p5*p3p4*p6p7+hbcm*p3p4*p4p5*p6p7+2*hbcm*p3p4*p4p6*
     . p4p7+hbcm*p3p4*p4p6*p5p7-hbcm*p3p4*p4p7*p5p6+2*hbcm*p3p5*p4p6*
     . p4p7)
      ans7=4*(p6p7*p2p3)*(-2*fmb*fmc*hbcm+fmc*hbcm2+5*fmc2*hbcm)+4*(
     . p4p7*p4p6)*(-2*ffmcfmb*hbcm3+fmb*hbcm2+fmc*hbcm2+hbcm3)+8*(
     . p6p7*p2p4)*(-ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+hbcm3)+4*(p5p6*
     . p3p7*p3p4)*(-3*ffmcfmb*hbcm-fmc)+4*(p4p6*p3p7*p3p4)*(2*ffmcfmb
     . *hbcm+fmb-fmc)+4*(p4p7*p3p6*p3p4)*(ffmcfmb*hbcm-2*fmb+4*fmc+5*
     . hbcm)+4*(p6p7*p3p4*p2p3)*(2*ffmcfmb*hbcm-2*fmb+3*fmc+hbcm)+4*(
     . p6p7*p3p4*p1p3)*(2*ffmcfmb*hbcm-fmc)+4*(p5p6*p3p7)*(-ffmcfmb*
     . fmc*hbcm2-fmc2*hbcm)+4*(p6p7*p3p5)*(2*ffmcfmb*fmc*hbcm2-fmc2*
     . hbcm)+4*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+2
     . *ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc2*hbcm)+4*(p4p7*p3p6)*(-ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-2*fmb*fmc*hbcm+6
     . *fmc*hbcm2+4*fmc2*hbcm)+2*(p6p7*p3p4)*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*
     . hbcm+3*fmc*hbcm2)+4*(p3p7*p3p6*p3p4)*(3*ffmcfmb**2*hbcm-
     . ffmcfmb*fmb+4*ffmcfmb*fmc+3*ffmcfmb*hbcm-fmc)+ans8
      ans6=w1*ans7
      ans12=4*(-2*ffmcfmb*hbcm*p1p3*p3p7*p5p6-2*ffmcfmb*hbcm*p1p5*
     . p3p6*p3p7+ffmcfmb*hbcm*p3p5*p3p6*p5p7+fmb*fmc*hbcm*p1p3*p6p7+3
     . *fmb*hbcm2*p1p2*p6p7+fmb*hbcm2*p1p4*p6p7+2*fmb*hbcm2*p2p4*p6p7
     . +4*fmb*hbcm2*p4p6*p4p7+fmc*p1p3*p3p5*p6p7+3*hbcm*p1p2*p3p5*
     . p6p7-2*hbcm*p1p2*p3p7*p5p6+hbcm*p1p4*p3p5*p6p7-2*hbcm*p1p4*
     . p3p7*p5p6-4*hbcm*p1p5*p2p3*p6p7-4*hbcm*p1p5*p3p6*p4p7+2*hbcm*
     . p1p5*p3p7*p4p6+4*hbcm*p2p3*p2p5*p6p7-4*hbcm*p2p3*p4p5*p6p7+4*
     . hbcm*p2p3*p4p6*p5p7-4*hbcm*p2p3*p4p7*p5p6+2*hbcm*p2p4*p3p5*
     . p6p7-hbcm*p2p5*p3p5*p6p7+4*hbcm*p2p5*p3p6*p4p7-4*hbcm*p2p5*
     . p3p7*p4p6+hbcm*p3p5*p4p5*p6p7+6*hbcm*p3p5*p4p6*p4p7+hbcm*p3p5*
     . p4p6*p5p7+5*hbcm*p3p5*p4p7*p5p6-4*hbcm*p3p6*p4p5*p4p7)
      ans11=4*(p5p6*p3p7)*(ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2-3*
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2)+4*(
     . p5p7*p3p6)*(-2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+2*ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3-fmc*hbcm2)+2*(p6p7*p3p5)*(-ffmcfmb**2*
     . hbcm3+3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-
     . fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*(p3p7*p3p6*p3p5)*(-3*
     . ffmcfmb**2*hbcm+ffmcfmb*fmb-4*ffmcfmb*fmc+5*ffmcfmb*hbcm+fmc)+
     . 2*p6p7*(-3*ffmcfmb**2*fmb*hbcm4+3*ffmcfmb*fmb*fmc*hbcm3+3*
     . ffmcfmb*fmb*hbcm4-ffmcfmb*fmb2*hbcm3+4*ffmcfmb*hbcm*p3p5s2-fmb
     . *fmc*hbcm3-2*fmb*fmc2*hbcm2+fmb2*fmc*hbcm2-2*fmc*p3p5s2)+4*(
     . p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2-4*ffmcfmb*fmb*fmc*hbcm+5*
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm+fmb*fmc*hbcm)+ans12
      ans10=4*(p6p7*p2p3)*(-3*fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)+4*(
     . p5p6*p4p7)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-2*fmc*hbcm2+4*hbcm3)+
     . 4*(p5p7*p4p6)*(-ffmcfmb*hbcm3+fmb*hbcm2+fmc*hbcm2)+4*(p6p7*
     . p4p5)*(ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2)+4*(p6p7*p2p5)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2)+4*(p6p7*p1p5)*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+8*(p4p5*p3p7*p3p6)*(-3*ffmcfmb*hbcm+hbcm)+4*(
     . p5p6*p3p7*p3p5)*(5*ffmcfmb*hbcm+fmc)+4*(p4p6*p3p7*p3p5)*(6*
     . ffmcfmb*hbcm-fmb+fmc)+4*(p4p7*p3p6*p3p5)*(-ffmcfmb*hbcm+2*fmb-
     . 4*fmc+3*hbcm)+8*(p5p6*p3p7*p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*
     . p3p5*p3p4)*(-2*ffmcfmb*hbcm+fmc+hbcm)+8*(p3p7*p3p6*p2p5)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p5p6*p3p7*p2p3)*(-3*ffmcfmb*hbcm+hbcm)+
     . 16*(p5p7*p3p6*p2p3)*(2*ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p5*p2p3)*(
     . -4*ffmcfmb*hbcm+2*fmb-3*fmc+hbcm)+4*(p4p6*p3p7)*(4*ffmcfmb*fmb
     . *hbcm2+fmb*fmc*hbcm-fmb2*hbcm)+4*(p4p7*p3p6)*(-ffmcfmb*fmb*
     . hbcm2-4*fmb*fmc*hbcm+5*fmb*hbcm2+2*fmb2*hbcm)+4*(p6p7*p3p4)*(-
     . 2*ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm+fmb*hbcm2)+ans11
      ans9=w2*ans10
      ans13=(4*(p6p7*p1p3)*(2*fb2*fmc-3*hbcm)+4*(p6p7*p2p3)*(-2*fb1*
     . hbcm+4*fb2*fmb-6*fb2*fmc-11*hbcm)+2*(p5p6*p3p7)*(-4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmc-hbcm)+8*(p4p6*p3p7)*(-2*fb1*ffmcfmb*
     . hbcm-fb2*fmb+fb2*fmc+2*hbcm)+2*(p5p7*p3p6)*(-4*fb1*ffmcfmb*
     . hbcm+5*hbcm)+2*(p4p7*p3p6)*(20*fb1*ffmcfmb*hbcm-28*fb1*hbcm+8*
     . fb2*fmb-16*fb2*fmc-9*hbcm)+8*(p6p7*p3p5)*(-2*fb1*ffmcfmb*hbcm-
     . fb2*fmc+4*hbcm)+8*(p6p7*p3p4)*(2*fb1*ffmcfmb*hbcm-fb1*hbcm+fb2
     . *fmc)+p6p7*(12*fb1*ffmcfmb**2*hbcm3-12*fb1*ffmcfmb*hbcm3-4*fb1
     . *fmb*fmc*hbcm+8*fb1*fmc2*hbcm-4*fb2*ffmcfmb*fmb*hbcm2+12*fb2*
     . ffmcfmb*fmc*hbcm2-4*fb2*fmc*hbcm2+17*fmb*hbcm2-15*fmc*hbcm2+15
     . *hbcm3)+2*(p3p7*p3p6)*(20*fb1*ffmcfmb**2*hbcm-28*fb1*ffmcfmb*
     . hbcm+4*fb2*ffmcfmb*fmb-16*fb2*ffmcfmb*fmc+4*fb2*fmc-12*ffmcfmb
     . *hbcm+fmb-4*fmc+11*hbcm)+8*(-3*fb1*hbcm*p1p2*p6p7-fb1*hbcm*
     . p1p4*p6p7-2*fb1*hbcm*p2p4*p6p7+fb1*hbcm*p2p5*p6p7-fb1*hbcm*
     . p4p5*p6p7-2*fb1*hbcm*p4p6*p4p7-fb1*hbcm*p4p6*p5p7-fb1*hbcm*
     . p4p7*p5p6))
      ans1=ans2+ans6+ans9+ans13
      ans=ccc*ans1
      b(1)=ans
      ans3=w2*(12*p4p7*(fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm)+2*(p5p7*
     . p3p5)*(ffmcfmb*hbcm-fmc)+4*(p4p7*p3p5)*(2*ffmcfmb*hbcm-3*fmb+3
     . *fmc-3*hbcm)+2*p5p7*(ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm)+2*(p3p7*
     . p3p5)*(3*ffmcfmb**2*hbcm-6*ffmcfmb*fmb+7*ffmcfmb*fmc-5*ffmcfmb
     . *hbcm-fmc)+2*p3p7*(-ffmcfmb**2*fmb*hbcm2+7*ffmcfmb*fmb*fmc*
     . hbcm-5*ffmcfmb*fmb*hbcm2-6*ffmcfmb*fmb2*hbcm-fmb*fmc*hbcm)+8*(
     . ffmcfmb*hbcm*p1p5*p3p7-ffmcfmb*hbcm*p2p5*p3p7+ffmcfmb*hbcm*
     . p3p7*p4p5+hbcm*p1p5*p4p7-hbcm*p2p5*p4p7+hbcm*p4p5*p4p7))+(12*
     . p4p7*(2*fb1*hbcm-2*fb2*fmb+2*fb2*fmc+hbcm)+p5p7*(-4*fb1*
     . ffmcfmb*hbcm-4*fb2*fmc-13*hbcm)+p3p7*(4*fb1*ffmcfmb**2*hbcm+20
     . *fb1*ffmcfmb*hbcm-24*fb2*ffmcfmb*fmb+28*fb2*ffmcfmb*fmc-4*fb2*
     . fmc+12*ffmcfmb*hbcm-6*fmb+7*fmc-7*hbcm))
      ans2=w1*(2*(p5p7*p3p4)*(-ffmcfmb*hbcm+fmc)+4*(p4p7*p3p4)*(-
     . ffmcfmb*hbcm+3*fmb-3*fmc-2*hbcm)+8*(p3p7*p2p4)*(-ffmcfmb*hbcm+
     . hbcm)+4*(p4p7*p2p3)*(ffmcfmb*hbcm-hbcm)+2*p5p7*(ffmcfmb*fmc*
     . hbcm2-fmc2*hbcm)+4*p4p7*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm3+3*fmb*
     . fmc*hbcm-3*fmc*hbcm2-3*fmc2*hbcm)+2*(p3p7*p3p4)*(-3*ffmcfmb**2
     . *hbcm+6*ffmcfmb*fmb-7*ffmcfmb*fmc-3*ffmcfmb*hbcm+fmc)+2*p3p7*(
     . -ffmcfmb**2*fmc*hbcm2+6*ffmcfmb*fmb*fmc*hbcm-5*ffmcfmb*fmc*
     . hbcm2-5*ffmcfmb*fmc2*hbcm-fmc2*hbcm)+4*(-ffmcfmb*hbcm*p1p3*
     . p4p7+2*ffmcfmb*hbcm*p1p4*p3p7-ffmcfmb*hbcm*p3p5*p4p7+2*ffmcfmb
     . *hbcm*p3p7*p4p5-hbcm*p1p2*p4p7+hbcm*p1p4*p4p7+2*hbcm*p2p4*p5p7
     . -hbcm*p2p5*p4p7+hbcm*p4p5*p4p7))+ans3
      ans1=w11*(2*(p5p7*p3p4)*(-ffmcfmb*hbcm+fmc)+4*(p4p7*p3p4)*(-
     . ffmcfmb*hbcm+3*fmb-3*fmc-2*hbcm)+2*(p5p7*p2p3)*(ffmcfmb*hbcm-
     . fmc)+4*(p4p7*p2p3)*(3*ffmcfmb*hbcm-3*fmb+3*fmc-4*hbcm)+2*p5p7*
     . (ffmcfmb*fmc*hbcm2-fmc2*hbcm)+4*p4p7*(ffmcfmb**2*hbcm3-ffmcfmb
     . *hbcm3+3*fmb*fmc*hbcm-3*fmc*hbcm2-3*fmc2*hbcm)+2*(p3p7*p3p4)*(
     . -3*ffmcfmb**2*hbcm+6*ffmcfmb*fmb-7*ffmcfmb*fmc-3*ffmcfmb*hbcm+
     . fmc)+2*(p3p7*p2p3)*(3*ffmcfmb**2*hbcm-6*ffmcfmb*fmb+7*ffmcfmb*
     . fmc-5*ffmcfmb*hbcm-fmc)+2*p3p7*(-ffmcfmb**2*fmc*hbcm2+6*
     . ffmcfmb*fmb*fmc*hbcm-5*ffmcfmb*fmc*hbcm2-5*ffmcfmb*fmc2*hbcm-
     . fmc2*hbcm)+4*(2*ffmcfmb*hbcm*p1p2*p3p7-ffmcfmb*hbcm*p1p3*p4p7+
     . 2*ffmcfmb*hbcm*p1p4*p3p7-6*ffmcfmb*hbcm*p2p5*p3p7-ffmcfmb*hbcm
     . *p3p5*p4p7+2*ffmcfmb*hbcm*p3p7*p4p5+hbcm*p1p2*p4p7+hbcm*p1p4*
     . p4p7+2*hbcm*p2p4*p3p7+2*hbcm*p2p4*p4p7+2*hbcm*p2p4*p5p7-7*hbcm
     . *p2p5*p4p7+hbcm*p4p5*p4p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w2*(4*(p4p5*p3p6)*(ffmcfmb*hbcm-hbcm)+2*(p4p6*p3p5)*(
     . ffmcfmb*hbcm+fmb-2*fmc+hbcm)+4*(p5p6*p3p4)*(-ffmcfmb*hbcm+hbcm
     . )+4*(p3p6*p2p5)*(3*ffmcfmb*hbcm-hbcm)+4*(p5p6*p2p3)*(-ffmcfmb*
     . hbcm+hbcm)+2*p4p6*(ffmcfmb*fmb*hbcm2-2*fmb*fmc*hbcm+fmb*hbcm2+
     . fmb2*hbcm)+4*p5p6*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3)+2*(p3p6*
     . p3p5)*(3*ffmcfmb**2*hbcm+ffmcfmb*fmb-2*ffmcfmb*fmc-ffmcfmb*
     . hbcm)+2*p3p6*(ffmcfmb**2*fmb*hbcm2-2*ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm)+4*(ffmcfmb*hbcm*p1p3*p5p6
     . -ffmcfmb*hbcm*p1p5*p3p6+hbcm*p1p2*p5p6+hbcm*p1p4*p5p6-hbcm*
     . p1p5*p4p6+2*hbcm*p2p5*p4p6))+(p4p6*(-4*fb1*ffmcfmb*hbcm-4*fb1*
     . hbcm+4*fb2*fmb-8*fb2*fmc-11*hbcm)+p3p6*(-4*fb1*ffmcfmb**2*hbcm
     . -4*fb1*ffmcfmb*hbcm+4*fb2*ffmcfmb*fmb-8*fb2*ffmcfmb*fmc+12*
     . ffmcfmb*hbcm+fmb-2*fmc-13*hbcm))
      ans1=w11*(2*(p4p6*p3p4)*(-ffmcfmb*hbcm-fmb+2*fmc+hbcm)+4*(p3p6*
     . p2p4)*(5*ffmcfmb*hbcm-3*hbcm)+2*(p4p6*p2p3)*(ffmcfmb*hbcm+fmb-
     . 2*fmc+hbcm)+2*p4p6*(ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+
     . 2*(p3p6*p3p4)*(-3*ffmcfmb**2*hbcm-ffmcfmb*fmb+2*ffmcfmb*fmc+3*
     . ffmcfmb*hbcm)+2*(p3p6*p2p3)*(3*ffmcfmb**2*hbcm+ffmcfmb*fmb-2*
     . ffmcfmb*fmc-ffmcfmb*hbcm)+2*p3p6*(ffmcfmb**2*fmc*hbcm2-ffmcfmb
     . *fmb*fmc*hbcm+ffmcfmb*fmc*hbcm2-2*ffmcfmb*fmc2*hbcm+2*fmc2*
     . hbcm)+4*(-ffmcfmb*hbcm*p1p2*p3p6-ffmcfmb*hbcm*p2p3*p5p6+
     . ffmcfmb*hbcm*p2p5*p3p6+ffmcfmb*hbcm*p3p4*p5p6+fmc2*hbcm*p5p6-
     . hbcm*p1p2*p4p6+2*hbcm*p2p4*p4p6-3*hbcm*p2p4*p5p6+hbcm*p2p5*
     . p4p6))+w1*(2*(p4p6*p3p4)*(-ffmcfmb*hbcm-fmb+2*fmc+hbcm)+8*(
     . p3p6*p2p4)*(2*ffmcfmb*hbcm-hbcm)+2*p4p6*(ffmcfmb*fmc*hbcm2-fmb
     . *fmc*hbcm+fmc*hbcm2)+2*(p3p6*p3p4)*(-3*ffmcfmb**2*hbcm-ffmcfmb
     . *fmb+2*ffmcfmb*fmc+3*ffmcfmb*hbcm)+2*p3p6*(ffmcfmb**2*fmc*
     . hbcm2-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmc*hbcm2-2*ffmcfmb*fmc2*
     . hbcm+2*fmc2*hbcm)+4*(ffmcfmb*hbcm*p3p4*p5p6+fmc2*hbcm*p5p6+2*
     . hbcm*p2p4*p4p6-2*hbcm*p2p4*p5p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans4=2*p6p7*(ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*fmc*hbcm2-3*
     . ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*hbcm2-2*ffmcfmb*p2p3s2+2*
     . ffmcfmb*p3p4s2-fmb*fmc2*hbcm+fmc2*hbcm2-2*fmc3*hbcm+2*p2p3s2-2
     . *p3p4s2)+4*(-ffmcfmb*fmc*hbcm*p1p3*p6p7-ffmcfmb*fmc*hbcm*p3p5*
     . p6p7-3*ffmcfmb*fmc*hbcm*p3p6*p3p7-ffmcfmb*fmc*hbcm*p3p6*p5p7+2
     . *ffmcfmb*fmc*hbcm*p3p7*p5p6+ffmcfmb*p1p3*p2p3*p6p7-ffmcfmb*
     . p1p3*p3p4*p6p7+ffmcfmb*p2p3*p3p5*p6p7+ffmcfmb*p2p3*p3p6*p5p7-
     . ffmcfmb*p3p4*p3p5*p6p7-ffmcfmb*p3p4*p3p6*p5p7-fmc*hbcm*p1p2*
     . p6p7-fmc*hbcm*p1p4*p6p7-fmc*hbcm*p2p5*p6p7-fmc*hbcm*p4p5*p6p7-
     . fmc*hbcm*p4p6*p5p7+p1p2*p2p3*p6p7-p1p2*p3p4*p6p7+p1p4*p2p3*
     . p6p7-p1p4*p3p4*p6p7+4*p2p3*p2p4*p6p7+p2p3*p2p5*p6p7+p2p3*p4p5*
     . p6p7+4*p2p3*p4p6*p4p7+p2p3*p4p6*p5p7-p2p3*p4p7*p5p6-4*p2p4*
     . p3p4*p6p7-p2p5*p3p4*p6p7-p3p4*p4p5*p6p7-4*p3p4*p4p6*p4p7-p3p4*
     . p4p6*p5p7+p3p4*p4p7*p5p6)
      ans3=(p4p6*p3p7*p3p4)*(-16*ffmcfmb-4)+(p4p7*p3p6*p3p4)*(12*
     . ffmcfmb-12)+(p4p6*p3p7*p2p3)*(16*ffmcfmb+4)+(p4p7*p3p6*p2p3)*(
     . -12*ffmcfmb+12)+4*(p5p6*p4p7)*(ffmcfmb*hbcm2+2*fmc*hbcm)+4*(
     . p4p7*p4p6)*(-2*ffmcfmb*hbcm2+fmb*hbcm-3*fmc*hbcm-hbcm2)+8*(
     . p6p7*p2p4)*(-2*ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm-hbcm2)+4*(p3p7*
     . p3p6*p3p4)*(4*ffmcfmb**2-5*ffmcfmb)+4*(p3p7*p3p6*p2p3)*(-4*
     . ffmcfmb**2+5*ffmcfmb)+4*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm2+2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-fmc*hbcm)+
     . 4*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+2*ffmcfmb*hbcm2-2*fmc*hbcm)+2*(p6p7*p3p4)*(3*ffmcfmb**2*
     . hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmb*fmc-
     . 3*fmc*hbcm+2*fmc2)+2*(p6p7*p2p3)*(-3*ffmcfmb**2*hbcm2+ffmcfmb*
     . fmb*hbcm+5*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*fmc-fmc*hbcm-2*
     . fmc2)+ans4
      ans2=w11*ans3
      ans7=4*(-ffmcfmb*fmc*hbcm*p1p3*p6p7-ffmcfmb*fmc*hbcm*p3p5*p6p7-
     . 3*ffmcfmb*fmc*hbcm*p3p6*p3p7-ffmcfmb*fmc*hbcm*p3p6*p5p7+2*
     . ffmcfmb*fmc*hbcm*p3p7*p5p6-ffmcfmb*p1p3*p3p4*p6p7-ffmcfmb*p3p4
     . *p3p5*p6p7-ffmcfmb*p3p4*p3p6*p5p7-fmc*hbcm*p1p2*p6p7-fmc*hbcm*
     . p1p4*p6p7-fmc*hbcm*p2p5*p6p7-fmc*hbcm*p4p5*p6p7-fmc*hbcm*p4p6*
     . p5p7-p1p2*p3p4*p6p7-p1p4*p3p4*p6p7-4*p2p4*p3p4*p6p7-p2p5*p3p4*
     . p6p7-p3p4*p4p5*p6p7-4*p3p4*p4p6*p4p7-p3p4*p4p6*p5p7+p3p4*p4p7*
     . p5p6)
      ans6=(p4p6*p3p7*p3p4)*(-16*ffmcfmb-4)+(p4p7*p3p6*p3p4)*(12*
     . ffmcfmb-12)+(p6p7*p3p4*p2p3)*(4*ffmcfmb-4)+4*(p5p6*p4p7)*(
     . ffmcfmb*hbcm2+2*fmc*hbcm)+4*(p4p7*p4p6)*(-2*ffmcfmb*hbcm2+fmb*
     . hbcm-3*fmc*hbcm-hbcm2)+8*(p6p7*p2p4)*(-ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+4*(p6p7*p2p3)*(ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p3p7*p3p6*
     . p3p4)*(4*ffmcfmb**2-5*ffmcfmb)+4*(p4p6*p3p7)*(-2*ffmcfmb**2*
     . hbcm2+2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-
     . fmc*hbcm)+4*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-2*fmc*hbcm)+2*(p6p7*p3p4)*(3*
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2+fmb*fmc-3*fmc*hbcm+2*fmc2)+2*p6p7*(ffmcfmb**2*fmc*hbcm3+
     . ffmcfmb*fmb*fmc*hbcm2-3*ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*hbcm2
     . +2*ffmcfmb*p3p4s2-fmb*fmc2*hbcm+fmc2*hbcm2-2*fmc3*hbcm-2*
     . p3p4s2)+ans7
      ans5=w1*ans6
      ans10=2*p6p7*(-ffmcfmb**2*fmb*hbcm3+ffmcfmb*fmb*fmc*hbcm2+3*
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2+2*ffmcfmb*p3p5s2-fmb*fmc*
     . hbcm2-2*fmb*fmc2*hbcm-fmb2*fmc*hbcm)+4*(ffmcfmb*fmb*hbcm*p1p3*
     . p6p7+3*ffmcfmb*fmb*hbcm*p3p6*p3p7+ffmcfmb*p1p3*p3p5*p6p7+
     . ffmcfmb*p3p5*p3p6*p5p7+fmb*hbcm*p1p2*p6p7+fmb*hbcm*p1p4*p6p7+4
     . *fmb*hbcm*p2p4*p6p7+6*fmb*hbcm*p4p6*p4p7+p1p2*p3p5*p6p7+p1p4*
     . p3p5*p6p7+4*p2p4*p3p5*p6p7+p2p5*p3p5*p6p7+p3p5*p4p5*p6p7+4*
     . p3p5*p4p6*p4p7+p3p5*p4p6*p5p7-p3p5*p4p7*p5p6)
      ans9=(p4p6*p3p7*p3p5)*(16*ffmcfmb+4)+(p4p7*p3p6*p3p5)*(-12*
     . ffmcfmb+12)+(p6p7*p3p5*p3p4)*(-4*ffmcfmb+4)+(p6p7*p3p5*p2p3)*(
     . -4*ffmcfmb+4)+4*(p5p6*p4p7)*(2*ffmcfmb*hbcm2+fmb*hbcm+2*fmc*
     . hbcm-4*hbcm2)+4*(p5p7*p4p6)*(ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+
     . 4*(p6p7*p4p5)*(-ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm)+4*(p6p7*p2p5)
     . *(ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm)+4*(p6p7*p1p5)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+4*(p4p6*p3p7)*(6*ffmcfmb*fmb*hbcm+fmb*hbcm)+4*
     . (p4p7*p3p6)*(ffmcfmb*fmb*hbcm+fmb*hbcm)+4*(p6p7*p3p4)*(-
     . ffmcfmb*fmb*hbcm+fmb*hbcm)+4*(p6p7*p2p3)*(-ffmcfmb*fmb*hbcm+
     . fmb*hbcm)+4*(p3p7*p3p6*p3p5)*(-4*ffmcfmb**2+5*ffmcfmb)+4*(p5p6
     . *p3p7)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm
     . -3*ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p7*p3p6)*(2*ffmcfmb**2*hbcm2+
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+2*
     . (p6p7*p3p5)*(-3*ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+3*ffmcfmb*
     . fmc*hbcm+3*ffmcfmb*hbcm2-fmb*fmc-fmc*hbcm-2*fmc2)+ans10
      ans8=w2*ans9
      ans11=((p4p6*p3p7)*(32*fb2*ffmcfmb+8*fb2+8)+(p5p7*p3p6)*(8*fb2*
     . ffmcfmb+2)+(p4p7*p3p6)*(-24*fb2*ffmcfmb+24*fb2-6)+(p6p7*p3p5)*
     . (8*fb2*ffmcfmb+2)+(p6p7*p3p4)*(-8*fb2*ffmcfmb+8*fb2-2)+(p6p7*
     . p2p3)*(-8*fb2*ffmcfmb+8*fb2-2)+(p6p7*p1p3)*(8*fb2*ffmcfmb+2)+(
     . p3p7*p3p6)*(-32*fb2*ffmcfmb**2+40*fb2*ffmcfmb-16*ffmcfmb+10)+
     . p6p7*(-4*fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmc
     . *hbcm-4*fb2*ffmcfmb**2*hbcm2+12*fb2*ffmcfmb*hbcm2-4*fb2*fmb*
     . fmc-8*fb2*fmc2-2*ffmcfmb*hbcm2+17*fmb*hbcm+15*fmc*hbcm-15*
     . hbcm2)+8*(fb2*p1p2*p6p7+fb2*p1p4*p6p7+4*fb2*p2p4*p6p7+fb2*p2p5
     . *p6p7+fb2*p4p5*p6p7+4*fb2*p4p6*p4p7+fb2*p4p6*p5p7-fb2*p4p7*
     . p5p6))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(4)=ans
      ans4=2*p3p7*(ffmcfmb**2*fmc*hbcm3-6*ffmcfmb*fmb*fmc*hbcm2-3*
     . ffmcfmb*fmc*hbcm3-5*ffmcfmb*fmc2*hbcm2-10*ffmcfmb*p2p3s2+2*
     . ffmcfmb*p3p4s2-fmc2*hbcm2+2*p2p3s2+2*p3p4s2)+4*(3*ffmcfmb*fmc*
     . hbcm*p1p3*p3p7-5*ffmcfmb*fmc*hbcm*p3p5*p3p7-3*ffmcfmb*p1p3*
     . p2p3*p3p7+3*ffmcfmb*p1p3*p3p4*p3p7+5*ffmcfmb*p2p3*p3p5*p3p7-5*
     . ffmcfmb*p3p4*p3p5*p3p7+hbcm2*p1p2*p4p7+hbcm2*p1p4*p4p7+2*hbcm2
     . *p2p4*p3p7+2*hbcm2*p2p4*p4p7+2*hbcm2*p2p4*p5p7-7*hbcm2*p2p5*
     . p4p7+hbcm2*p4p5*p4p7-p1p2*p2p3*p3p7+p1p2*p3p4*p3p7-2*p1p3*p2p3
     . *p4p7+2*p1p3*p3p4*p4p7-p1p4*p2p3*p3p7+p1p4*p3p4*p3p7-p2p3*p2p5
     . *p3p7+2*p2p3*p3p4*p4p7-2*p2p3*p3p4*p5p7+6*p2p3*p3p5*p4p7-p2p3*
     . p3p7*p4p5+p2p5*p3p4*p3p7-6*p3p4*p3p5*p4p7+p3p4*p3p7*p4p5)
      ans3=(p3p7*p3p4*p2p3)*(16*ffmcfmb-8)+4*(p4p5*p3p7)*(2*ffmcfmb*
     . hbcm2+fmc*hbcm)+4*(p4p7*p3p5)*(-ffmcfmb*hbcm2-6*fmc*hbcm)+2*(
     . p5p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p3p4)*(ffmcfmb*
     . hbcm2-3*fmb*hbcm-fmc*hbcm-2*hbcm2)+4*(p3p7*p2p5)*(-6*ffmcfmb*
     . hbcm2+fmc*hbcm)+2*(p5p7*p2p3)*(-ffmcfmb*hbcm2-3*fmc*hbcm)+4*(
     . p4p7*p2p3)*(ffmcfmb*hbcm2-3*fmb*hbcm+fmc*hbcm+2*hbcm2)+4*(p3p7
     . *p1p4)*(2*ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p1p3)*(-ffmcfmb*
     . hbcm2+2*fmc*hbcm)+4*(p3p7*p1p2)*(2*ffmcfmb*hbcm2+fmc*hbcm)+2*
     . p5p7*(ffmcfmb*fmc*hbcm3-fmc2*hbcm2+4*p2p3s2)+4*p4p7*(ffmcfmb**
     . 2*hbcm4+2*ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*
     . fmc*hbcm3-3*fmc2*hbcm2-4*p2p3s2+2*p3p4s2)+2*(p3p7*p3p4)*(-
     . ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-5*ffmcfmb*fmc*hbcm-ffmcfmb
     . *hbcm2+3*fmc*hbcm)+2*(p3p7*p2p3)*(3*ffmcfmb**2*hbcm2-6*ffmcfmb
     . *fmb*hbcm+3*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2-fmc*hbcm)+ans4
      ans2=w11*ans3
      ans7=4*(3*ffmcfmb*fmc*hbcm*p1p3*p3p7-5*ffmcfmb*fmc*hbcm*p3p5*
     . p3p7+3*ffmcfmb*p1p3*p3p4*p3p7-5*ffmcfmb*p3p4*p3p5*p3p7+fmc*
     . hbcm*p1p2*p3p7-2*fmc*hbcm*p2p3*p5p7+fmc*hbcm*p2p5*p3p7-hbcm2*
     . p1p2*p4p7+hbcm2*p1p4*p4p7+2*hbcm2*p2p4*p5p7-hbcm2*p2p5*p4p7+
     . hbcm2*p4p5*p4p7+p1p2*p3p4*p3p7+2*p1p3*p3p4*p4p7+p1p4*p3p4*p3p7
     . +4*p2p3*p3p4*p4p7-2*p2p3*p3p4*p5p7+p2p5*p3p4*p3p7-6*p3p4*p3p5*
     . p4p7+p3p4*p3p7*p4p5)
      ans6=(p3p7*p3p4*p2p3)*(20*ffmcfmb-4)+4*(p4p5*p3p7)*(2*ffmcfmb*
     . hbcm2+fmc*hbcm)+4*(p4p7*p3p5)*(-ffmcfmb*hbcm2-6*fmc*hbcm)+2*(
     . p5p7*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p3p4)*(ffmcfmb*
     . hbcm2-3*fmb*hbcm-fmc*hbcm-2*hbcm2)+8*(p3p7*p2p4)*(-ffmcfmb*
     . hbcm2+hbcm2)+4*(p4p7*p2p3)*(ffmcfmb*hbcm2+4*fmc*hbcm-hbcm2)+4*
     . (p3p7*p1p4)*(2*ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p1p3)*(-ffmcfmb
     . *hbcm2+2*fmc*hbcm)+2*p5p7*(ffmcfmb*fmc*hbcm3-fmc2*hbcm2)+4*(
     . p3p7*p2p3)*(5*ffmcfmb*fmc*hbcm-fmc*hbcm)+4*p4p7*(ffmcfmb**2*
     . hbcm4+2*ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*fmc*
     . hbcm3-3*fmc2*hbcm2+2*p3p4s2)+2*(p3p7*p3p4)*(-ffmcfmb**2*hbcm2-
     . 6*ffmcfmb*fmb*hbcm-5*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+3*fmc*hbcm
     . )+2*p3p7*(ffmcfmb**2*fmc*hbcm3-6*ffmcfmb*fmb*fmc*hbcm2-3*
     . ffmcfmb*fmc*hbcm3-5*ffmcfmb*fmc2*hbcm2+2*ffmcfmb*p3p4s2-fmc2*
     . hbcm2+2*p3p4s2)+ans7
      ans5=w1*ans6
      ans10=4*(-3*ffmcfmb*fmb*hbcm*p1p3*p3p7+2*ffmcfmb*hbcm2*p1p5*
     . p3p7-3*ffmcfmb*p1p3*p3p5*p3p7-fmb*hbcm*p1p2*p3p7-2*fmb*hbcm*
     . p1p3*p4p7-fmb*hbcm*p1p4*p3p7-4*fmb*hbcm*p2p3*p4p7+2*fmb*hbcm*
     . p2p3*p5p7-2*fmb*hbcm*p3p4*p4p7+2*hbcm2*p1p5*p4p7-2*hbcm2*p2p5*
     . p4p7+2*hbcm2*p4p5*p4p7-p1p2*p3p5*p3p7-2*p1p3*p3p5*p4p7-p1p4*
     . p3p5*p3p7-4*p2p3*p3p5*p4p7+2*p2p3*p3p5*p5p7-p2p5*p3p5*p3p7-2*
     . p3p4*p3p5*p4p7-p3p5*p3p7*p4p5)
      ans9=12*(p4p7*p3p5)*(fmb*hbcm-fmc*hbcm+hbcm2)+(p3p7*p3p5*p3p4)*
     . (-4*ffmcfmb-4)+(p3p7*p3p5*p2p3)*(-20*ffmcfmb+4)+4*(p4p5*p3p7)*
     . (2*ffmcfmb*hbcm2-fmb*hbcm)+2*(p5p7*p3p5)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+4*(p3p7*p2p5)*(-2*ffmcfmb*hbcm2-fmb*hbcm)+2*p5p7*(-
     . ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2)+4*p4p7*(-2*ffmcfmb*fmb*hbcm3-
     . 3*fmb*fmc*hbcm2+3*fmb*hbcm3-3*fmb2*hbcm2+6*p3p5s2)+4*(p3p7*
     . p3p4)*(-ffmcfmb*fmb*hbcm-fmb*hbcm)+4*(p3p7*p2p3)*(-5*ffmcfmb*
     . fmb*hbcm+fmb*hbcm)+2*(p3p7*p3p5)*(3*ffmcfmb**2*hbcm2+4*ffmcfmb
     . *fmb*hbcm-7*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+fmc*hbcm)+2*p3p7*
     . (-ffmcfmb**2*fmb*hbcm3-7*ffmcfmb*fmb*fmc*hbcm2+3*ffmcfmb*fmb*
     . hbcm3-6*ffmcfmb*fmb2*hbcm2+10*ffmcfmb*p3p5s2+fmb*fmc*hbcm2)+
     . ans10
      ans8=w2*ans9
      ans11=((p3p7*p3p5)*(40*fb2*ffmcfmb+10)+(p3p7*p3p4)*(-8*fb2*
     . ffmcfmb-8*fb2-2)+(p3p7*p2p3)*(-40*fb2*ffmcfmb+8*fb2-10)+(p3p7*
     . p1p3)*(-24*fb2*ffmcfmb-6)+p5p7*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*
     . hbcm2-13*hbcm2)+4*p4p7*(6*fb1*fmb*hbcm+6*fb1*fmc*hbcm-4*fb2*
     . ffmcfmb*hbcm2+6*fb2*hbcm2+5*hbcm2)+p3p7*(24*fb1*ffmcfmb*fmb*
     . hbcm+28*fb1*ffmcfmb*fmc*hbcm-4*fb1*fmc*hbcm-4*fb2*ffmcfmb**2*
     . hbcm2+12*fb2*ffmcfmb*hbcm2+16*ffmcfmb*hbcm2-12*fmb*hbcm-7*fmc*
     . hbcm+9*hbcm2)+8*(-fb2*p1p2*p3p7-2*fb2*p1p3*p4p7-fb2*p1p4*p3p7-
     . 4*fb2*p2p3*p4p7+2*fb2*p2p3*p5p7-fb2*p2p5*p3p7-2*fb2*p3p4*p4p7+
     . 6*fb2*p3p5*p4p7-fb2*p3p7*p4p5))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(5)=ans
      ans4=4*(2*ffmcfmb*fmc*hbcm*p3p5*p3p6-2*ffmcfmb*p2p3*p3p5*p3p6+2
     . *ffmcfmb*p3p4*p3p5*p3p6-fmc*hbcm*p1p3*p4p6+fmc*hbcm*p1p4*p3p6+
     . 2*fmc*hbcm*p2p3*p5p6-fmc*hbcm*p3p4*p5p6+fmc*hbcm*p3p5*p4p6+fmc
     . *hbcm*p3p6*p4p5-hbcm2*p1p2*p4p6+2*hbcm2*p2p4*p4p6-3*hbcm2*p2p4
     . *p5p6+hbcm2*p2p5*p4p6-p1p2*p2p3*p3p6+p1p2*p3p4*p3p6+p1p3*p2p3*
     . p4p6-p1p3*p3p4*p4p6-p1p4*p2p3*p3p6+p1p4*p3p4*p3p6-p2p3*p2p5*
     . p3p6-3*p2p3*p3p4*p4p6+3*p2p3*p3p4*p5p6-p2p3*p3p5*p4p6-p2p3*
     . p3p6*p4p5+p2p5*p3p4*p3p6+p3p4*p3p5*p4p6+p3p4*p3p6*p4p5)
      ans3=(p3p6*p3p4*p2p3)*(-24*ffmcfmb+12)+2*(p4p6*p3p4)*(-ffmcfmb*
     . hbcm2+fmb*hbcm+2*fmc*hbcm+hbcm2)+4*(p3p6*p2p5)*(ffmcfmb*hbcm2+
     . fmc*hbcm)+4*(p3p6*p2p4)*(5*ffmcfmb*hbcm2-3*hbcm2)+2*(p4p6*p2p3
     . )*(-ffmcfmb*hbcm2+fmb*hbcm-4*fmc*hbcm-hbcm2)+4*(p3p6*p1p2)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*p5p6*(-ffmcfmb*fmc*hbcm3+fmc2*hbcm2-
     . 2*p2p3s2-p3p4s2)+2*p4p6*(ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc*
     . hbcm3+6*p2p3s2)+2*(p3p6*p3p4)*(-3*ffmcfmb**2*hbcm2+ffmcfmb*fmb
     . *hbcm+2*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2)+2*(p3p6*p2p3)*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-10*ffmcfmb*fmc*hbcm-3*
     . ffmcfmb*hbcm2+6*fmc*hbcm)+2*p3p6*(ffmcfmb**2*fmc*hbcm3+ffmcfmb
     . *fmb*fmc*hbcm2+ffmcfmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm2+12*
     . ffmcfmb*p2p3s2+2*fmc2*hbcm2-6*p2p3s2)+ans4
      ans2=w11*ans3
      ans7=((p3p6*p3p5)*(-16*fb2*ffmcfmb-4)+(p3p6*p2p3)*(48*fb2*
     . ffmcfmb-24*fb2+12)+4*p5p6*(2*fb2*ffmcfmb*hbcm2-hbcm2)+p4p6*(-4
     . *fb1*fmb*hbcm-8*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-7
     . *hbcm2)+p3p6*(-4*fb1*ffmcfmb*fmb*hbcm-8*fb1*ffmcfmb*fmc*hbcm-4
     . *fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2+16*ffmcfmb*hbcm2-13*
     . fmb*hbcm-8*fmc*hbcm-7*hbcm2)+8*(-fb2*p1p2*p3p6+fb2*p1p3*p4p6-
     . fb2*p1p4*p3p6+3*fb2*p2p3*p4p6-2*fb2*p2p3*p5p6-fb2*p2p5*p3p6+
     . fb2*p3p4*p5p6-fb2*p3p5*p4p6-fb2*p3p6*p4p5))
      ans6=w2*((p3p6*p3p5*p2p3)*(24*ffmcfmb-12)+4*(p4p5*p3p6)*(
     . ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+2*(p4p6*p3p5)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+2*fmc*hbcm-hbcm2)+4*(p5p6*p3p4)*(-ffmcfmb*hbcm2+fmb*
     . hbcm+hbcm2)+4*(p3p6*p2p5)*(3*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+4*(
     . p5p6*p2p3)*(-ffmcfmb*hbcm2-2*fmb*hbcm+hbcm2)+2*p4p6*(-ffmcfmb*
     . fmb*hbcm3+2*fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-2*p3p5s2)+12*(
     . p3p6*p2p3)*(2*ffmcfmb*fmb*hbcm-fmb*hbcm)+4*p5p6*(-ffmcfmb**2*
     . hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*hbcm4)+2*(p3p6*p3p5)*(ffmcfmb
     . **2*hbcm2-3*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-3*ffmcfmb*
     . hbcm2)+2*p3p6*(-ffmcfmb**2*fmb*hbcm3+2*ffmcfmb*fmb*fmc*hbcm2-
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-4*ffmcfmb*p3p5s2)+4*(
     . ffmcfmb*hbcm2*p1p3*p5p6-ffmcfmb*hbcm2*p1p5*p3p6+ffmcfmb*hbcm2*
     . p3p5*p5p6-fmb*hbcm*p1p2*p3p6+fmb*hbcm*p1p3*p4p6-fmb*hbcm*p1p4*
     . p3p6+3*fmb*hbcm*p2p3*p4p6+hbcm2*p1p2*p5p6+hbcm2*p1p4*p5p6-
     . hbcm2*p1p5*p4p6+2*hbcm2*p2p5*p4p6-p1p2*p3p5*p3p6+p1p3*p3p5*
     . p4p6-p1p4*p3p5*p3p6+3*p2p3*p3p5*p4p6-2*p2p3*p3p5*p5p6-p2p5*
     . p3p5*p3p6+p3p4*p3p5*p5p6-p3p5*p3p6*p4p5))+ans7
      ans5=w1*((p3p6*p3p4*p2p3)*(-24*ffmcfmb+12)+2*(p4p6*p3p4)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+hbcm2)+8*(p3p6*p2p4)*(2*
     . ffmcfmb*hbcm2-hbcm2)+4*p5p6*(-ffmcfmb*fmc*hbcm3+fmc2*hbcm2-
     . p3p4s2)+2*p4p6*(ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc*hbcm3)+12*
     . (p3p6*p2p3)*(-2*ffmcfmb*fmc*hbcm+fmc*hbcm)+2*(p3p6*p3p4)*(-3*
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm+3*ffmcfmb
     . *hbcm2)+2*p3p6*(ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*fmc*hbcm2+
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm2+2*fmc2*hbcm2)+4*(2*
     . ffmcfmb*fmc*hbcm*p3p5*p3p6+2*ffmcfmb*p3p4*p3p5*p3p6+fmc*hbcm*
     . p1p2*p3p6-fmc*hbcm*p1p3*p4p6+fmc*hbcm*p1p4*p3p6-3*fmc*hbcm*
     . p2p3*p4p6+2*fmc*hbcm*p2p3*p5p6+fmc*hbcm*p2p5*p3p6-fmc*hbcm*
     . p3p4*p5p6+fmc*hbcm*p3p5*p4p6+fmc*hbcm*p3p6*p4p5+2*hbcm2*p2p4*
     . p4p6-2*hbcm2*p2p4*p5p6+p1p2*p3p4*p3p6-p1p3*p3p4*p4p6+p1p4*p3p4
     . *p3p6-3*p2p3*p3p4*p4p6+2*p2p3*p3p4*p5p6+p2p5*p3p4*p3p6+p3p4*
     . p3p5*p4p6+p3p4*p3p6*p4p5))+ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(6)=ans
      ans2=w2*((p3p5*p3p4)*(-2*ffmcfmb+2)+(p3p5*p2p3)*(-2*ffmcfmb+2)+
     . 2*p3p4*(-ffmcfmb*fmb*hbcm+fmb*hbcm)+2*p2p3*(-ffmcfmb*fmb*hbcm+
     . fmb*hbcm)+2*p3p5*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2)+2*(-ffmcfmb**2*fmb*hbcm3+ffmcfmb*fmb*hbcm*p1p3+ffmcfmb*
     . fmb*hbcm3+ffmcfmb*p1p3*p3p5+ffmcfmb*p3p5s2+fmb*hbcm*p1p2+fmb*
     . hbcm*p1p4+fmb*hbcm*p2p5+fmb*hbcm*p4p5+p1p2*p3p5+p1p4*p3p5+p2p5
     . *p3p5+p3p5*p4p5))+(p3p4*(-4*fb2*ffmcfmb+4*fb2-1)+p2p3*(-4*fb2*
     . ffmcfmb+4*fb2-1)+(-4*fb2*ffmcfmb**2*hbcm2+4*fb2*ffmcfmb*hbcm2+
     . 4*fb2*ffmcfmb*p1p3+4*fb2*ffmcfmb*p3p5+4*fb2*p1p2+4*fb2*p1p4+4*
     . fb2*p2p5+4*fb2*p4p5-2*ffmcfmb*hbcm2+9*fmb*hbcm+9*fmc*hbcm-8*
     . hbcm2+p1p3+p3p5))
      ans1=w11*(2*p3p4*(ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmc*hbcm)+2*p2p3*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-fmc*hbcm)+2*(ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmc*
     . hbcm*p1p3-ffmcfmb*fmc*hbcm*p3p5-ffmcfmb*fmc*hbcm3+ffmcfmb*p1p3
     . *p2p3-ffmcfmb*p1p3*p3p4+ffmcfmb*p2p3*p3p5-ffmcfmb*p2p3s2-
     . ffmcfmb*p3p4*p3p5+ffmcfmb*p3p4s2-fmc*hbcm*p1p2-fmc*hbcm*p1p4-
     . fmc*hbcm*p2p5-fmc*hbcm*p4p5+p1p2*p2p3-p1p2*p3p4+p1p4*p2p3-p1p4
     . *p3p4+p2p3*p2p5+p2p3*p4p5+p2p3s2-p2p5*p3p4-p3p4*p4p5-p3p4s2))+
     . w1*((p3p4*p2p3)*(2*ffmcfmb-2)+2*p2p3*(ffmcfmb*fmc*hbcm-fmc*
     . hbcm)+2*p3p4*(ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-
     . fmc*hbcm)+2*(ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmc*hbcm*p1p3-
     . ffmcfmb*fmc*hbcm*p3p5-ffmcfmb*fmc*hbcm3-ffmcfmb*p1p3*p3p4-
     . ffmcfmb*p3p4*p3p5+ffmcfmb*p3p4s2-fmc*hbcm*p1p2-fmc*hbcm*p1p4-
     . fmc*hbcm*p2p5-fmc*hbcm*p4p5-p1p2*p3p4-p1p4*p3p4-p2p5*p3p4-p3p4
     . *p4p5-p3p4s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=(p3p5*(4*fb1*ffmcfmb*hbcm-17*hbcm)+p3p4*(-4*fb1*ffmcfmb*
     . hbcm+4*fb1*hbcm+hbcm)+p2p3*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm+25*
     . hbcm)+p1p3*(4*fb1*ffmcfmb*hbcm+7*hbcm)+(-4*fb1*ffmcfmb**2*
     . hbcm3+4*fb1*ffmcfmb*hbcm3+4*fb1*hbcm*p1p2+4*fb1*hbcm*p1p4+4*
     . fb1*hbcm*p2p5+4*fb1*hbcm*p4p5-9*fmb*hbcm2+9*fmc*hbcm2-8*hbcm3)
     . )
      ans2=w1*(2*(p3p4*p2p3)*(ffmcfmb*hbcm-hbcm)+2*p2p3*(ffmcfmb*fmc*
     . hbcm2-fmc*hbcm2)+2*p3p4*(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmc*hbcm2)+2*(ffmcfmb**2*fmc*hbcm4-ffmcfmb*fmc*
     . hbcm2*p1p3-ffmcfmb*fmc*hbcm2*p3p5-ffmcfmb*fmc*hbcm4-ffmcfmb*
     . hbcm*p1p3*p3p4-ffmcfmb*hbcm*p3p4*p3p5+ffmcfmb*hbcm*p3p4s2-fmc*
     . hbcm2*p1p2-fmc*hbcm2*p1p4-fmc*hbcm2*p2p5-fmc*hbcm2*p4p5-hbcm*
     . p1p2*p3p4-hbcm*p1p4*p3p4-hbcm*p2p5*p3p4-hbcm*p3p4*p4p5-hbcm*
     . p3p4s2))+w2*(2*(p3p5*p3p4)*(ffmcfmb*hbcm-hbcm)+2*(p3p5*p2p3)*(
     . ffmcfmb*hbcm-hbcm)+2*p3p4*(ffmcfmb*fmb*hbcm2-fmb*hbcm2)+2*p2p3
     . *(ffmcfmb*fmb*hbcm2-fmb*hbcm2)+2*p3p5*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3)+2*(ffmcfmb**2*fmb*hbcm4-
     . ffmcfmb*fmb*hbcm2*p1p3-ffmcfmb*fmb*hbcm4-ffmcfmb*hbcm*p1p3*
     . p3p5-ffmcfmb*hbcm*p3p5s2-fmb*hbcm2*p1p2-fmb*hbcm2*p1p4-fmb*
     . hbcm2*p2p5-fmb*hbcm2*p4p5-hbcm*p1p2*p3p5-hbcm*p1p4*p3p5-hbcm*
     . p2p5*p3p5-hbcm*p3p5*p4p5))+ans3
      ans1=w11*(4*(p3p4*p2p3)*(ffmcfmb*hbcm-hbcm)+2*p3p4*(ffmcfmb**2*
     . hbcm3+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmc*hbcm2)+2*p2p3*(
     . ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmc*hbcm2)+2*
     . (ffmcfmb**2*fmc*hbcm4-ffmcfmb*fmc*hbcm2*p1p3-ffmcfmb*fmc*hbcm2
     . *p3p5-ffmcfmb*fmc*hbcm4-ffmcfmb*hbcm*p1p3*p2p3-ffmcfmb*hbcm*
     . p1p3*p3p4-ffmcfmb*hbcm*p2p3*p3p5+ffmcfmb*hbcm*p2p3s2-ffmcfmb*
     . hbcm*p3p4*p3p5+ffmcfmb*hbcm*p3p4s2-fmc*hbcm2*p1p2-fmc*hbcm2*
     . p1p4-fmc*hbcm2*p2p5-fmc*hbcm2*p4p5-hbcm*p1p2*p2p3-hbcm*p1p2*
     . p3p4-hbcm*p1p4*p2p3-hbcm*p1p4*p3p4-hbcm*p2p3*p2p5-hbcm*p2p3*
     . p4p5-hbcm*p2p3s2-hbcm*p2p5*p3p4-hbcm*p3p4*p4p5-hbcm*p3p4s2))+
     . ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(2*(p3p6*p3p4)*(-8*ffmcfmb*hbcm+3*hbcm)+2*p3p6*(-8*
     . ffmcfmb*fmc*hbcm2+3*fmc*hbcm2)+2*(-5*fmc*hbcm2*p4p6+3*fmc*
     . hbcm2*p5p6-5*hbcm*p3p4*p4p6+3*hbcm*p3p4*p5p6))+w2*(2*(p3p6*
     . p3p5)*(-8*ffmcfmb*hbcm+3*hbcm)+2*p3p6*(-8*ffmcfmb*fmb*hbcm2+3*
     . fmb*hbcm2)+2*(-5*fmb*hbcm2*p4p6+3*fmb*hbcm2*p5p6-5*hbcm*p3p5*
     . p4p6+3*hbcm*p3p5*p5p6))+w11*(2*p5p6*(2*ffmcfmb*hbcm3+3*fmc*
     . hbcm2)+2*p4p6*(-ffmcfmb*hbcm3-fmb*hbcm2-3*fmc*hbcm2-hbcm3)+2*(
     . p3p6*p3p4)*(-8*ffmcfmb*hbcm+3*hbcm)+2*(p3p6*p2p3)*(4*ffmcfmb*
     . hbcm-3*hbcm)+2*p3p6*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-6*
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+3*fmc*hbcm2)+2*(-4*ffmcfmb*
     . hbcm*p3p5*p3p6-2*hbcm*p1p2*p3p6+2*hbcm*p1p3*p4p6-2*hbcm*p1p4*
     . p3p6+hbcm*p2p3*p4p6-hbcm*p2p3*p5p6-2*hbcm*p2p5*p3p6-5*hbcm*
     . p3p4*p4p6+5*hbcm*p3p4*p5p6-2*hbcm*p3p5*p4p6-2*hbcm*p3p6*p4p5))
     . +(4*p3p6*(8*fb1*ffmcfmb*hbcm-3*fb1*hbcm+5*hbcm)+4*(5*fb1*hbcm*
     . p4p6-3*fb1*hbcm*p5p6)))
      b(9)=ans
      b(10)=ccc*(w1*((p3p6*p3p4)*(16*ffmcfmb-6)+2*p3p6*(8*ffmcfmb*fmc
     . *hbcm-3*fmc*hbcm)+2*(5*fmc*hbcm*p4p6-3*fmc*hbcm*p5p6+5*p3p4*
     . p4p6-3*p3p4*p5p6))+w2*((p3p6*p3p5)*(-16*ffmcfmb+6)+2*p3p6*(-8*
     . ffmcfmb*fmb*hbcm+3*fmb*hbcm)+2*(-5*fmb*hbcm*p4p6+3*fmb*hbcm*
     . p5p6-5*p3p5*p4p6+3*p3p5*p5p6))+w11*((p3p6*p3p4)*(16*ffmcfmb-6)
     . +(p3p6*p2p3)*(-16*ffmcfmb+6)+2*p4p6*(ffmcfmb*hbcm2-fmb*hbcm+3*
     . fmc*hbcm+hbcm2)+2*p3p6*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+6*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-3*fmc*hbcm)+2*(-3*fmc*hbcm*p5p6
     . -5*p2p3*p4p6+3*p2p3*p5p6+5*p3p4*p4p6-3*p3p4*p5p6))+(p3p6*(-32*
     . fb2*ffmcfmb+12*fb2-8)+4*(-5*fb2*p4p6+3*fb2*p5p6)))
      b(11)=ccc*(w11*(2*p3p4*(ffmcfmb*hbcm2-hbcm2)+2*p2p3*(ffmcfmb*
     . hbcm2-hbcm2)+2*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm2*p1p3-ffmcfmb*
     . hbcm2*p3p5-ffmcfmb*hbcm4-hbcm2*p1p2-hbcm2*p1p4-hbcm2*p2p5-
     . hbcm2*p4p5))+18*hbcm2)
      ans=ccc*(w1*(4*(p3p7*p3p4)*(-4*ffmcfmb*hbcm+hbcm)+4*p3p7*(-4*
     . ffmcfmb*fmc*hbcm2+fmc*hbcm2)+4*(-3*fmc*hbcm2*p4p7+fmc*hbcm2*
     . p5p7-3*hbcm*p3p4*p4p7+hbcm*p3p4*p5p7))+w2*(4*(p3p7*p3p5)*(-4*
     . ffmcfmb*hbcm+hbcm)+4*p3p7*(-4*ffmcfmb*fmb*hbcm2+fmb*hbcm2)+4*(
     . -3*fmb*hbcm2*p4p7+fmb*hbcm2*p5p7-3*hbcm*p3p5*p4p7+hbcm*p3p5*
     . p5p7))+w11*(2*p5p7*(ffmcfmb*hbcm3+fmc*hbcm2)+4*p4p7*(2*ffmcfmb
     . *hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(p3p7*p3p4)*(-3*ffmcfmb*hbcm+2*
     . hbcm)+2*p3p7*(ffmcfmb**2*hbcm3-6*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc
     . *hbcm2-3*ffmcfmb*hbcm3+fmc*hbcm2)+4*(3*ffmcfmb*hbcm*p1p3*p3p7+
     . ffmcfmb*hbcm*p2p3*p3p7-5*ffmcfmb*hbcm*p3p5*p3p7+hbcm*p1p2*p3p7
     . +2*hbcm*p1p3*p4p7+hbcm*p1p4*p3p7+hbcm*p2p3*p4p7-hbcm*p2p3*p5p7
     . +hbcm*p2p5*p3p7-hbcm*p3p4*p4p7+hbcm*p3p4*p5p7-6*hbcm*p3p5*p4p7
     . +hbcm*p3p7*p4p5))+(4*p3p7*(8*fb1*ffmcfmb*hbcm-2*fb1*hbcm-5*
     . hbcm)+8*(3*fb1*hbcm*p4p7-fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w11*(2*p3p4*(-ffmcfmb*hbcm+hbcm)+2*p2p3*(-ffmcfmb*
     . hbcm+hbcm)+2*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm*p1p3+ffmcfmb*hbcm
     . *p3p5+ffmcfmb*hbcm3+hbcm*p1p2+hbcm*p1p4+hbcm*p2p5+hbcm*p4p5))-
     . 18*hbcm)
      b(14)=ccc*(w1*((p3p7*p3p4)*(-16*ffmcfmb+4)+4*p3p7*(-4*ffmcfmb*
     . fmc*hbcm+fmc*hbcm)+4*(-3*fmc*hbcm*p4p7+fmc*hbcm*p5p7-3*p3p4*
     . p4p7+p3p4*p5p7))+w2*((p3p7*p3p5)*(16*ffmcfmb-4)+4*p3p7*(4*
     . ffmcfmb*fmb*hbcm-fmb*hbcm)+4*(3*fmb*hbcm*p4p7-fmb*hbcm*p5p7+3*
     . p3p5*p4p7-p3p5*p5p7))+w11*(12*p4p7*(fmb*hbcm-hbcm2)+(p3p7*p3p4
     . )*(-16*ffmcfmb+4)+(p3p7*p2p3)*(16*ffmcfmb-4)+2*p5p7*(ffmcfmb*
     . hbcm2+fmc*hbcm)+2*p3p7*(-ffmcfmb**2*hbcm2+6*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2+fmc*hbcm)+4*(3*p2p3*p4p7-p2p3
     . *p5p7-3*p3p4*p4p7+p3p4*p5p7))+(p3p7*(32*fb2*ffmcfmb-8*fb2+8)+8
     . *(3*fb2*p4p7-fb2*p5p7)))
      ans4=((p3p7*p3p6)*(-64*fb2*ffmcfmb+24*fb2-16)+8*p6p7*(fb1*fmb*
     . hbcm+fb1*fmc*hbcm-2*fb2*ffmcfmb*hbcm2-fb2*hbcm2+4*hbcm2)+8*(4*
     . fb2*p2p3*p6p7-4*fb2*p3p4*p6p7-2*fb2*p3p6*p4p7+2*fb2*p3p6*p5p7-
     . 3*fb2*p3p7*p4p6+fb2*p3p7*p5p6))
      ans3=w11*(4*(p4p6*p3p7)*(-fmb*hbcm+2*fmc*hbcm)+(p3p7*p3p6*p3p4)
     . *(32*ffmcfmb-12)+(p3p7*p3p6*p2p3)*(-32*ffmcfmb+12)+4*(p5p6*
     . p3p7)*(ffmcfmb*hbcm2-2*fmc*hbcm)+4*(p5p7*p3p6)*(-2*ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(p4p7*p3p6)*(-3*ffmcfmb*hbcm2-4*fmb*hbcm+2*
     . hbcm2)+4*(p6p7*p3p4)*(3*ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm)+4*(
     . p6p7*p2p3)*(-ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-2*hbcm2)+2*p6p7*(
     . 3*ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb
     . *hbcm4-fmb*fmc*hbcm2+fmc*hbcm3-4*fmc2*hbcm2+8*p2p3s2+8*p3p4s2)
     . +4*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm2-5*ffmcfmb*fmb*hbcm+5*
     . ffmcfmb*fmc*hbcm-3*fmc*hbcm)+4*(-ffmcfmb*hbcm2*p1p3*p6p7-
     . ffmcfmb*hbcm2*p3p5*p6p7-hbcm2*p1p2*p6p7-hbcm2*p1p4*p6p7+4*
     . hbcm2*p2p4*p6p7-hbcm2*p2p5*p6p7-hbcm2*p4p5*p6p7-3*hbcm2*p4p6*
     . p4p7-hbcm2*p4p6*p5p7+2*hbcm2*p4p7*p5p6-8*p2p3*p3p4*p6p7-2*p2p3
     . *p3p6*p4p7+2*p2p3*p3p6*p5p7-3*p2p3*p3p7*p4p6+p2p3*p3p7*p5p6+2*
     . p3p4*p3p6*p4p7-2*p3p4*p3p6*p5p7+3*p3p4*p3p7*p4p6-p3p4*p3p7*
     . p5p6))+ans4
      ans2=w2*(4*(p6p7*p3p5)*(-fmb*hbcm-fmc*hbcm-hbcm2)+(p3p7*p3p6*
     . p3p5)*(-32*ffmcfmb+12)+4*(p5p6*p3p7)*(4*ffmcfmb*hbcm2+fmb*hbcm
     . -2*hbcm2)+8*(p5p7*p3p6)*(-2*ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*
     . p6p7*(-2*ffmcfmb*fmb*hbcm3-fmb*fmc*hbcm2-fmb*hbcm3-fmb2*hbcm2)
     . +4*(p3p7*p3p6)*(-8*ffmcfmb*fmb*hbcm+3*fmb*hbcm)+4*(4*fmb*hbcm*
     . p2p3*p6p7-4*fmb*hbcm*p3p4*p6p7-2*fmb*hbcm*p3p6*p4p7-3*fmb*hbcm
     . *p3p7*p4p6+2*hbcm2*p1p5*p6p7-2*hbcm2*p2p5*p6p7+2*hbcm2*p4p5*
     . p6p7-2*hbcm2*p4p6*p5p7+2*hbcm2*p4p7*p5p6+4*p2p3*p3p5*p6p7-4*
     . p3p4*p3p5*p6p7-2*p3p5*p3p6*p4p7+2*p3p5*p3p6*p5p7-3*p3p5*p3p7*
     . p4p6+p3p5*p3p7*p5p6))+ans3
      ans1=w1*(4*(p6p7*p3p4)*(-fmb*hbcm+3*fmc*hbcm+hbcm2)+(p3p7*p3p6*
     . p3p4)*(32*ffmcfmb-12)+4*(p4p6*p3p7)*(4*ffmcfmb*hbcm2+3*fmc*
     . hbcm)+4*(p4p7*p3p6)*(-4*ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2)+4*p6p7
     . *(2*ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmc*hbcm3-3*fmc2*hbcm2+4*
     . p3p4s2)+4*(p3p7*p3p6)*(8*ffmcfmb*fmc*hbcm-3*fmc*hbcm)+4*(-4*
     . fmc*hbcm*p2p3*p6p7-2*fmc*hbcm*p3p6*p5p7-fmc*hbcm*p3p7*p5p6+4*
     . hbcm2*p2p4*p6p7+hbcm2*p4p6*p4p7+hbcm2*p4p7*p5p6-4*p2p3*p3p4*
     . p6p7+2*p3p4*p3p6*p4p7-2*p3p4*p3p6*p5p7+3*p3p4*p3p7*p4p6-p3p4*
     . p3p7*p5p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans2=8*p6p7*(fb1*hbcm-fb2*fmb+fb2*fmc+4*hbcm)
      ans1=w2*(4*p6p7*(fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm)+8*(p5p6*p3p7
     . )*(2*ffmcfmb*hbcm-hbcm)+8*(p5p7*p3p6)*(-2*ffmcfmb*hbcm+hbcm)+4
     . *(p6p7*p3p5)*(2*ffmcfmb*hbcm-fmb+fmc-hbcm)+8*(hbcm*p1p5*p6p7-
     . hbcm*p2p5*p6p7+hbcm*p4p5*p6p7-hbcm*p4p6*p5p7+hbcm*p4p7*p5p6))+
     . w1*(4*p6p7*(fmb*fmc*hbcm-fmc*hbcm2-3*fmc2*hbcm)+4*(p4p7*p3p6)*
     . (-4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p4)*(-2*ffmcfmb*hbcm+fmb-fmc-
     . hbcm)+4*(4*ffmcfmb*hbcm*p3p7*p4p6+4*hbcm*p2p4*p6p7+hbcm*p4p6*
     . p4p7+hbcm*p4p7*p5p6))+w11*(4*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-hbcm
     . )+4*(p6p7*p3p4)*(-ffmcfmb*hbcm+fmb-fmc-2*hbcm)+4*(p6p7*p2p3)*(
     . 3*ffmcfmb*hbcm-fmb+fmc)+2*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2
     . -4*fmc2*hbcm)+4*(-ffmcfmb*hbcm*p1p3*p6p7-ffmcfmb*hbcm*p3p5*
     . p6p7-3*ffmcfmb*hbcm*p3p6*p3p7-5*ffmcfmb*hbcm*p3p6*p4p7-ffmcfmb
     . *hbcm*p3p6*p5p7+2*ffmcfmb*hbcm*p3p7*p5p6-hbcm*p1p2*p6p7-hbcm*
     . p1p4*p6p7+4*hbcm*p2p4*p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7-5*
     . hbcm*p4p6*p4p7-hbcm*p4p6*p5p7+4*hbcm*p4p7*p5p6))+ans2
      ans=ccc*ans1
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)+1.0d0*b(n)
200   continue
      return
      end 
 
      subroutine amp34_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)
     . )
      ans3=(8*(p6p7*p3p4)*(fb2*fmb-fb2*fmc)+4*(p6p7*p3p5)*(-2*fb1*
     . hbcm-3*hbcm)+2*p6p7*(-2*fb1*fmb2*hbcm+2*fb1*fmc2*hbcm-2*fb1*
     . hbcm3+4*fb2*ffmcfmb*fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-5*fmb*
     . hbcm2+5*fmc*hbcm2-3*hbcm3)+2*(p6p7*p2p3)*(-12*fb1*ffmcfmb*hbcm
     . +12*fb1*hbcm-12*fb2*fmb+16*fb2*fmc+9*hbcm)+2*(p6p7*p1p3)*(12*
     . fb1*ffmcfmb*hbcm-4*fb1*hbcm-4*fb2*fmb+3*hbcm)+8*(2*fb1*hbcm*
     . p1p2*p6p7+fb1*hbcm*p1p4*p6p7-fb1*hbcm*p1p5*p6p7-fb1*hbcm*p2p4*
     . p6p7-3*fb1*hbcm*p2p5*p6p7))
      ans2=w1*(4*(p6p7*p4p5)*(-fmb*hbcm2+fmc*hbcm2-hbcm3)+2*(p6p7*
     . p3p4)*(2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm-2*fmb*hbcm2+fmb2*hbcm+2*
     . fmc*hbcm2+fmc2*hbcm-hbcm3)+4*(p6p7*p2p4)*(ffmcfmb*hbcm3-2*fmb*
     . hbcm2+3*fmc*hbcm2-2*hbcm3)+4*(p6p7*p1p4)*(-ffmcfmb*hbcm3-fmc*
     . hbcm2)+4*(p6p7*p3p5*p3p4)*(2*ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4*
     . p2p3)*(-3*ffmcfmb*hbcm+3*fmb-4*fmc+hbcm)+4*(p6p7*p3p4*p1p3)*(-
     . ffmcfmb*hbcm+fmb+hbcm)+4*(p6p7*p2p3)*(ffmcfmb*fmc*hbcm2+3*fmb*
     . fmc*hbcm-2*fmc*hbcm2-4*fmc2*hbcm)+4*(p6p7*p1p3)*(-ffmcfmb*fmc*
     . hbcm2+fmb*fmc*hbcm)+2*p6p7*(-2*ffmcfmb*fmb*fmc*hbcm3+2*ffmcfmb
     . *fmc2*hbcm3-2*fmb*p3p4s2+fmb2*fmc*hbcm2+fmc*hbcm4+2*fmc*p3p4s2
     . -fmc3*hbcm2)+4*(-2*fmc*hbcm2*p1p2*p6p7+4*fmc*hbcm2*p2p5*p6p7+
     . fmc*hbcm2*p3p5*p6p7-2*hbcm*p1p2*p3p4*p6p7+4*hbcm*p1p3*p2p4*
     . p6p7+2*hbcm*p1p3*p4p5*p6p7-hbcm*p1p4*p3p4*p6p7-hbcm*p1p5*p3p4*
     . p6p7+4*hbcm*p2p3*p2p4*p6p7+2*hbcm*p2p3*p4p5*p6p7+hbcm*p2p4*
     . p3p4*p6p7-4*hbcm*p2p4*p3p5*p6p7+5*hbcm*p2p5*p3p4*p6p7-2*hbcm*
     . p3p5*p4p5*p6p7))+ans3
      ans1=w2*(4*(p6p7*p3p5*p3p4)*(fmb-fmc)+4*(p6p7*p4p5)*(-fmb*hbcm2
     . +fmc*hbcm2-hbcm3)+4*(p6p7*p3p4)*(-fmb*fmc*hbcm+fmb2*hbcm)+2*(
     . p6p7*p3p5)*(-2*ffmcfmb*hbcm3+2*fmb*hbcm2+fmb2*hbcm-fmc2*hbcm+
     . hbcm3)+4*(p6p7*p2p5)*(ffmcfmb*hbcm3+fmb*hbcm2+2*fmc*hbcm2-3*
     . hbcm3)+4*(p6p7*p1p5)*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+4*(p6p7*
     . p3p5*p2p3)*(3*ffmcfmb*hbcm-3*fmb+4*fmc-hbcm)+4*(p6p7*p3p5*p1p3
     . )*(ffmcfmb*hbcm-fmb-hbcm)+4*(p6p7*p2p3)*(ffmcfmb*fmb*hbcm2+4*
     . fmb*fmc*hbcm-2*fmb*hbcm2-3*fmb2*hbcm)+4*(p6p7*p1p3)*(-ffmcfmb*
     . fmb*hbcm2-fmb2*hbcm)+2*p6p7*(-2*ffmcfmb*fmb*fmc*hbcm3+2*
     . ffmcfmb*fmb2*hbcm3-4*ffmcfmb*hbcm*p3p5s2-fmb*fmc2*hbcm2+fmb*
     . hbcm4+fmb3*hbcm2+2*hbcm*p3p5s2)+4*(-2*fmb*hbcm2*p1p2*p6p7-2*
     . hbcm*p1p2*p3p5*p6p7+hbcm*p1p4*p3p5*p6p7+4*hbcm*p1p5*p2p3*p6p7+
     . hbcm*p1p5*p3p5*p6p7+4*hbcm*p2p3*p2p5*p6p7+4*hbcm*p2p3*p4p5*
     . p6p7-hbcm*p2p4*p3p5*p6p7-hbcm*p2p5*p3p5*p6p7-2*hbcm*p3p5*p4p5*
     . p6p7))+ans2
      ans=ccc*ans1
      b(1)=ans
      ans2=w1*(4*(p6p7*p4p5)*(-fmb*hbcm-fmc*hbcm+hbcm2)+(p6p7*p3p4*
     . p2p3)*(-8*ffmcfmb+12)+(p6p7*p3p4*p1p3)*(8*ffmcfmb-4)+4*(p6p7*
     . p2p4)*(ffmcfmb*hbcm2-2*fmb*hbcm+2*fmc*hbcm+2*hbcm2)+2*p6p7*(2*
     . ffmcfmb*fmc*hbcm3-fmb2*fmc*hbcm-fmc*hbcm3+fmc3*hbcm+2*p3p4s2)+
     . 2*(p6p7*p3p4)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2*fmb*
     . hbcm-fmb2+fmc2+hbcm2)+4*(-ffmcfmb*hbcm2*p1p4*p6p7-2*fmc*hbcm*
     . p1p2*p6p7+fmc*hbcm*p1p5*p6p7+2*fmc*hbcm*p2p3*p6p7-fmc*hbcm*
     . p2p5*p6p7-fmc*hbcm*p3p5*p6p7-2*p1p2*p3p4*p6p7+4*p2p4*p3p4*p6p7
     . -p3p4*p3p5*p6p7))+((p6p7*p2p3)*(16*fb2*ffmcfmb-24*fb2+4)+(p6p7
     . *p1p3)*(-16*fb2*ffmcfmb+8*fb2-4)+2*p6p7*(-4*fb2*ffmcfmb*hbcm2+
     . 2*fb2*fmb2-2*fb2*fmc2+2*fb2*hbcm2-3*fmb*hbcm-3*fmc*hbcm+5*
     . hbcm2)+8*(2*fb2*p1p2*p6p7-4*fb2*p2p4*p6p7-fb2*p3p4*p6p7+fb2*
     . p3p5*p6p7))
      ans1=w2*(4*(p6p7*p4p5)*(-fmb*hbcm-fmc*hbcm+hbcm2)+(p6p7*p3p5*
     . p2p3)*(8*ffmcfmb-12)+(p6p7*p3p5*p1p3)*(-8*ffmcfmb+4)+4*(p6p7*
     . p2p5)*(-ffmcfmb*hbcm2-2*fmb*hbcm-2*fmc*hbcm+3*hbcm2)+4*(p6p7*
     . p1p5)*(ffmcfmb*hbcm2-hbcm2)+2*p6p7*(-2*ffmcfmb*fmb*hbcm3-fmb*
     . fmc2*hbcm+fmb*hbcm3+fmb3*hbcm+2*p3p5s2)+2*(p6p7*p3p5)*(-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2+hbcm2
     . )+4*(2*fmb*hbcm*p1p2*p6p7+fmb*hbcm*p1p4*p6p7-2*fmb*hbcm*p2p3*
     . p6p7-5*fmb*hbcm*p2p4*p6p7-fmb*hbcm*p3p4*p6p7+2*p1p2*p3p5*p6p7-
     . 4*p2p4*p3p5*p6p7-p3p4*p3p5*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      b(15)=ccc*(w1*(12*(p6p7*p3p4)*(fmb*hbcm-fmc*hbcm-hbcm2)+4*p6p7*
     . (-2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*hbcm2-fmc*hbcm3+3*fmc2*hbcm2-6
     . *p3p4s2)+8*(fmc*hbcm*p1p3*p6p7+fmc*hbcm*p2p3*p6p7+fmc*hbcm*
     . p3p5*p6p7-hbcm2*p1p4*p6p7-hbcm2*p2p4*p6p7-hbcm2*p4p5*p6p7+p1p3
     . *p3p4*p6p7+p2p3*p3p4*p6p7+p3p4*p3p5*p6p7))+w2*(4*(p6p7*p3p5)*(
     . fmb*hbcm+3*fmc*hbcm+hbcm2)+4*p6p7*(2*ffmcfmb*fmb*hbcm3+3*fmb*
     . fmc*hbcm2+fmb*hbcm3+3*fmb2*hbcm2-2*p3p5s2)+8*(-fmb*hbcm*p1p3*
     . p6p7-fmb*hbcm*p2p3*p6p7+3*fmb*hbcm*p3p4*p6p7-hbcm2*p1p5*p6p7-
     . hbcm2*p2p5*p6p7-hbcm2*p4p5*p6p7-p1p3*p3p5*p6p7-p2p3*p3p5*p6p7+
     . 3*p3p4*p3p5*p6p7))+(4*p6p7*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*
     . fb2*ffmcfmb*hbcm2+2*fb2*hbcm2-5*hbcm2)+16*(-fb2*p1p3*p6p7-fb2*
     . p2p3*p6p7+3*fb2*p3p4*p6p7-fb2*p3p5*p6p7)))
      b(16)=ccc*(w2*(12*p6p7*(-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+4*(
     . p6p7*p3p5)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+8*(-hbcm*p1p5*
     . p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7))+w1*(12*p6p7*(-fmb*fmc*
     . hbcm+fmc*hbcm2+fmc2*hbcm)+4*(p6p7*p3p4)*(2*ffmcfmb*hbcm-3*fmb+
     . 3*fmc+hbcm)+8*(-hbcm*p1p4*p6p7-hbcm*p2p4*p6p7-hbcm*p4p5*p6p7))
     . +12*p6p7*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-hbcm))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
      end 
 
      subroutine amp33_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      ans4=8*(p3p7*p3p6*p2p3)*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*fmc+5*
     . ffmcfmb*hbcm)+4*p6p7*(ffmcfmb**2*fmc*hbcm4+2*ffmcfmb*hbcm*
     . p2p3s2+2*ffmcfmb*hbcm*p3p4s2+2*fmc*p2p3s2+2*fmc*p3p4s2+2*fmc2*
     . hbcm3-fmc3*hbcm2-2*hbcm*p3p4s2)+8*(ffmcfmb*fmc*hbcm2*p3p6*p5p7
     . -2*ffmcfmb*hbcm*p1p2*p3p6*p3p7+2*ffmcfmb*hbcm*p1p3*p3p6*p4p7-4
     . *ffmcfmb*hbcm*p1p3*p3p7*p4p6+ffmcfmb*hbcm*p2p3*p3p6*p5p7-2*
     . ffmcfmb*hbcm*p2p4*p3p6*p3p7+2*ffmcfmb*hbcm*p2p5*p3p6*p3p7+
     . ffmcfmb*hbcm*p3p4*p3p6*p5p7-2*ffmcfmb*hbcm*p3p5*p3p6*p4p7+4*
     . ffmcfmb*hbcm*p3p5*p3p7*p4p6+fmc*hbcm2*p1p4*p6p7+fmc*hbcm2*p4p6
     . *p5p7-fmc*p1p3*p3p4*p6p7-fmc2*hbcm*p1p3*p6p7-2*hbcm*p1p2*p3p4*
     . p6p7-4*hbcm*p1p2*p3p6*p4p7+2*hbcm*p1p2*p3p7*p4p6+2*hbcm*p1p3*
     . p2p4*p6p7-2*hbcm*p1p3*p4p6*p4p7+hbcm*p1p4*p2p3*p6p7+hbcm*p1p4*
     . p3p4*p6p7+2*hbcm*p2p3*p2p4*p6p7+hbcm*p2p3*p2p5*p6p7-hbcm*p2p3*
     . p3p7*p4p6+3*hbcm*p2p3*p4p6*p4p7+hbcm*p2p3*p4p6*p5p7-2*hbcm*
     . p2p4*p3p5*p6p7-2*hbcm*p2p4*p3p7*p4p6+3*hbcm*p2p5*p3p4*p6p7+4*
     . hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*p4p6+3*hbcm*p3p4*p4p6*
     . p4p7+hbcm*p3p4*p4p6*p5p7+2*hbcm*p3p5*p4p6*p4p7)
      ans3=8*(p4p7*p4p6)*(-2*ffmcfmb*hbcm3+fmc*hbcm2+2*hbcm3)+8*(p6p7
     . *p2p5)*(ffmcfmb*hbcm3+2*fmc*hbcm2)+8*(p6p7*p2p4)*(ffmcfmb*
     . hbcm3+fmc*hbcm2-2*hbcm3)+8*(p6p7*p1p2)*(-ffmcfmb*hbcm3-fmc*
     . hbcm2)+8*(p4p6*p3p7*p3p4)*(4*ffmcfmb*hbcm+hbcm)+8*(p4p7*p3p6*
     . p3p4)*(ffmcfmb*hbcm+2*fmc+2*hbcm)+8*(p6p7*p3p5*p3p4)*(ffmcfmb*
     . hbcm+fmc)+8*(p4p7*p3p6*p2p3)*(ffmcfmb*hbcm-2*fmc+6*hbcm)+8*(
     . p6p7*p3p5*p2p3)*(-ffmcfmb*hbcm-fmc)+8*(p6p7*p3p4*p2p3)*(-2*
     . ffmcfmb*hbcm-2*fmc+hbcm)+8*(p6p7*p2p3*p1p3)*(2*ffmcfmb*hbcm+
     . fmc)+8*(p4p7*p3p6)*(ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmc*
     . hbcm2+2*fmc2*hbcm)+8*(p3p7*p3p6)*(3*ffmcfmb*fmc*hbcm2+2*
     . ffmcfmb*fmc2*hbcm)+8*(p6p7*p3p5)*(ffmcfmb*fmc*hbcm2+fmc2*hbcm)
     . +8*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3+fmc*hbcm2)
     . +4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmc*hbcm2+fmc2*hbcm
     . )+4*(p6p7*p2p3)*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-3*fmc2*hbcm)+8*(p3p7*p3p6*p3p4)*(2*ffmcfmb**2*
     . hbcm+2*ffmcfmb*fmc+3*ffmcfmb*hbcm)+ans4
      ans2=w11*ans3
      ans7=8*(ffmcfmb*fmc*hbcm2*p3p6*p5p7+2*ffmcfmb*hbcm*p1p3*p3p6*
     . p4p7-4*ffmcfmb*hbcm*p1p3*p3p7*p4p6+2*ffmcfmb*hbcm*p2p3*p3p6*
     . p4p7-4*ffmcfmb*hbcm*p2p3*p3p7*p4p6+ffmcfmb*hbcm*p3p4*p3p6*p5p7
     . -2*ffmcfmb*hbcm*p3p5*p3p6*p4p7+4*ffmcfmb*hbcm*p3p5*p3p7*p4p6+
     . fmc*hbcm2*p1p4*p6p7+fmc*hbcm2*p2p5*p6p7+fmc*hbcm2*p4p6*p5p7-
     . fmc*p1p3*p3p4*p6p7-fmc2*hbcm*p1p3*p6p7-2*hbcm*p1p3*p4p6*p4p7+
     . hbcm*p1p4*p3p4*p6p7-2*hbcm*p2p3*p4p6*p4p7+hbcm*p2p5*p3p4*p6p7+
     . 3*hbcm*p3p4*p4p6*p4p7+hbcm*p3p4*p4p6*p5p7+2*hbcm*p3p5*p4p6*
     . p4p7)
      ans6=8*(p4p7*p4p6)*(-2*ffmcfmb*hbcm3+fmc*hbcm2+2*hbcm3)+8*(p4p6
     . *p3p7*p3p4)*(4*ffmcfmb*hbcm+hbcm)+8*(p4p7*p3p6*p3p4)*(ffmcfmb*
     . hbcm+2*fmc+2*hbcm)+8*(p6p7*p3p5*p3p4)*(ffmcfmb*hbcm+fmc)+8*(
     . p6p7*p3p4*p2p3)*(-ffmcfmb*hbcm-fmc)+8*(p4p7*p3p6)*(ffmcfmb*fmc
     . *hbcm2-2*ffmcfmb*hbcm3+2*fmc*hbcm2+2*fmc2*hbcm)+8*(p3p7*p3p6)*
     . (3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*fmc2*hbcm)+8*(p6p7*p3p5)*(
     . ffmcfmb*fmc*hbcm2+fmc2*hbcm)+8*(p6p7*p2p3)*(-ffmcfmb*fmc*hbcm2
     . -fmc2*hbcm)+8*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm3+4*ffmcfmb*hbcm3
     . +fmc*hbcm2)+4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmc*
     . hbcm2+fmc2*hbcm)+8*(p3p7*p3p6*p3p4)*(2*ffmcfmb**2*hbcm+2*
     . ffmcfmb*fmc+3*ffmcfmb*hbcm)+4*p6p7*(ffmcfmb**2*fmc*hbcm4+2*
     . ffmcfmb*hbcm*p3p4s2+2*fmc*p3p4s2+2*fmc2*hbcm3-fmc3*hbcm2-2*
     . hbcm*p3p4s2)+ans7
      ans5=w1*ans6
      ans10=8*(p3p7*p3p6)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*fmc*hbcm-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmb2*hbcm+ffmcfmb*fmc*hbcm2-ffmcfmb*
     . fmc2*hbcm+ffmcfmb*hbcm3-fmc*hbcm2+fmc2*hbcm)+8*(p6p7*p3p5)*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2)+8*(p6p7*p2p3)*(ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2)+4*(p6p7*p1p3)*(ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-2*
     . fmb*fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm)+8*(p3p7*p3p6*p1p3)*(2*
     . ffmcfmb**2*hbcm+2*ffmcfmb*fmc-3*ffmcfmb*hbcm)+4*p6p7*(-ffmcfmb
     . **2*fmb*hbcm4+ffmcfmb**2*fmc*hbcm4-ffmcfmb**2*hbcm5-ffmcfmb*
     . fmb2*hbcm3+ffmcfmb*fmc2*hbcm3+ffmcfmb*hbcm5+2*fmb*fmc*hbcm3+
     . fmb*fmc2*hbcm2-fmb2*fmc*hbcm2-fmc*hbcm4-2*fmc*p1p3s2-fmc2*
     . hbcm3)+8*(-ffmcfmb*hbcm*p1p3*p3p6*p5p7-hbcm*p1p3*p1p4*p6p7-
     . hbcm*p1p3*p2p5*p6p7-hbcm*p1p3*p4p6*p4p7-hbcm*p1p3*p4p6*p5p7)
      ans9=8*(p5p7*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*p2p5)*(-ffmcfmb
     . *hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+16*(p4p7*p3p6*p3p5)*(-ffmcfmb*hbcm+hbcm)+16*(p5p7
     . *p3p6*p3p4)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p2p5)*(ffmcfmb*
     . hbcm-hbcm)+32*(p5p7*p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(p4p7*
     . p3p6*p2p3)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p1p4)*(ffmcfmb*
     . hbcm-hbcm)+8*(p4p6*p3p7*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+8*(p4p7*
     . p3p6*p1p3)*(3*ffmcfmb*hbcm+2*fmc-4*hbcm)+8*(p6p7*p3p5*p1p3)*(-
     . ffmcfmb*hbcm+fmc)+8*(p6p7*p3p4*p1p3)*(ffmcfmb*hbcm+fmc-hbcm)+8
     . *(p6p7*p2p3*p1p3)*(ffmcfmb*hbcm-fmc)+8*(p4p7*p3p6)*(-ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm-2*
     . fmb2*hbcm-fmc*hbcm2+2*hbcm3)+8*(p6p7*p3p4)*(-ffmcfmb*fmb*hbcm2
     . +ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)
     . +8*(p4p6*p3p7)*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3-fmb*hbcm2+fmb2
     . *hbcm)+8*(p5p7*p3p6)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3)+ans10
      ans8=w13*ans9
      ans11=(8*(p4p6*p3p7)*(-2*fb1*hbcm+3*hbcm)+4*(p5p7*p3p6)*(-4*fb1
     . *ffmcfmb*hbcm+5*hbcm)+4*(p4p7*p3p6)*(4*fb1*ffmcfmb*hbcm-8*fb1*
     . hbcm-8*fb2*fmc-5*hbcm)+4*(p6p7*p3p5)*(-4*fb1*ffmcfmb*hbcm-4*
     . fb2*fmc+5*hbcm)+4*(p6p7*p3p4)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-
     . 4*fb2*fmc-hbcm)+4*(p6p7*p2p3)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmc-5*
     . hbcm)+4*p6p7*(-2*fb1*ffmcfmb**2*hbcm3+2*fb1*fmc2*hbcm-4*fb2*
     . fmc*hbcm2+2*ffmcfmb*hbcm3+3*fmb*hbcm2-fmc*hbcm2)+4*(p3p7*p3p6)
     . *(8*fb1*ffmcfmb**2*hbcm-12*fb1*ffmcfmb*hbcm-8*fb2*ffmcfmb*fmc-
     . 8*ffmcfmb*hbcm-2*fmc+9*hbcm)+16*(-fb1*hbcm*p1p4*p6p7-fb1*hbcm*
     . p2p5*p6p7-fb1*hbcm*p4p6*p4p7-fb1*hbcm*p4p6*p5p7+fb2*fmc*p1p3*
     . p6p7))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(1)=ans
      ans3=(2*p4p7*(4*fb1*ffmcfmb*hbcm-8*fb2*fmb+12*fb2*fmc+hbcm)+2*
     . p3p7*(4*fb1*ffmcfmb*hbcm-8*fb2*ffmcfmb*fmb+8*fb2*ffmcfmb*fmc+4
     . *fb2*fmc-2*fmb+2*fmc-5*hbcm)-12*hbcm*p5p7)
      ans2=w1*(4*(p4p7*p3p4)*(-ffmcfmb*hbcm+2*fmb-3*fmc-2*hbcm)+4*(
     . p3p7*p3p4)*(2*ffmcfmb*fmb-2*ffmcfmb*fmc-5*ffmcfmb*hbcm-fmc)+4*
     . p4p7*(-ffmcfmb**2*hbcm3-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+2*
     . fmb*fmc*hbcm-4*fmc2*hbcm)+4*p3p7*(-2*ffmcfmb**2*fmc*hbcm2+2*
     . ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmc*hbcm2-4*ffmcfmb*fmc2*hbcm-
     . fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p4p7+2*ffmcfmb*hbcm*p1p4*p3p7
     . -ffmcfmb*hbcm*p3p4*p5p7-fmc2*hbcm*p5p7+hbcm*p1p4*p4p7+2*hbcm*
     . p2p4*p5p7-hbcm*p2p5*p4p7))+w13*(4*(p4p7*p1p3)*(-ffmcfmb*hbcm+2
     . *fmb-3*fmc+2*hbcm)+4*(p3p7*p1p3)*(2*ffmcfmb*fmb-2*ffmcfmb*fmc+
     . ffmcfmb*hbcm-fmc)+4*p4p7*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3-3*fmb*fmc*hbcm-2*fmb*hbcm2+3*fmb2*hbcm+3*
     . fmc*hbcm2-hbcm3)+4*p3p7*(2*ffmcfmb**2*fmb*hbcm2-2*ffmcfmb**2*
     . fmc*hbcm2+ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb*
     . hbcm2+2*ffmcfmb*fmb2*hbcm+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc*hbcm2))+ans3
      ans1=w11*(4*(p4p7*p3p4)*(-ffmcfmb*hbcm+2*fmb-3*fmc-2*hbcm)+4*(
     . p4p7*p2p3)*(ffmcfmb*hbcm-2*fmb+3*fmc-6*hbcm)+4*(p3p7*p3p4)*(2*
     . ffmcfmb*fmb-2*ffmcfmb*fmc-5*ffmcfmb*hbcm-fmc)+4*(p3p7*p2p3)*(-
     . 2*ffmcfmb*fmb+2*ffmcfmb*fmc-3*ffmcfmb*hbcm+fmc)+4*p4p7*(-
     . ffmcfmb**2*hbcm3-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+2*fmb*fmc
     . *hbcm-4*fmc2*hbcm)+4*p3p7*(-2*ffmcfmb**2*fmc*hbcm2+2*ffmcfmb*
     . fmb*fmc*hbcm-ffmcfmb*fmc*hbcm2-4*ffmcfmb*fmc2*hbcm-fmc2*hbcm)+
     . 8*(2*ffmcfmb*hbcm*p1p2*p3p7-ffmcfmb*hbcm*p1p3*p4p7+2*ffmcfmb*
     . hbcm*p1p4*p3p7+ffmcfmb*hbcm*p2p3*p5p7-4*ffmcfmb*hbcm*p2p5*p3p7
     . -ffmcfmb*hbcm*p3p4*p5p7-fmc2*hbcm*p5p7+3*hbcm*p1p2*p4p7+hbcm*
     . p1p4*p4p7+hbcm*p2p4*p3p7+3*hbcm*p2p4*p5p7-6*hbcm*p2p5*p4p7))+
     . ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w13*(8*(p3p6*p3p4)*(ffmcfmb*hbcm-hbcm)+24*(p3p6*p2p5)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6
     . *p1p4)*(-ffmcfmb*hbcm+hbcm)+4*(p4p6*p1p3)*(ffmcfmb*hbcm+fmc-2*
     . hbcm)+16*(p3p6*p1p2)*(ffmcfmb*hbcm-hbcm)+4*p4p6*(-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb2*hbcm-
     . fmc*hbcm2+hbcm3)+4*(p3p6*p1p3)*(-ffmcfmb**2*hbcm+ffmcfmb*fmc)+
     . 4*p3p6*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*fmc*hbcm2+ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb2*hbcm-ffmcfmb*fmc*hbcm2
     . -ffmcfmb*hbcm3))+(2*p4p6*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc-7*hbcm
     . )+2*p3p6*(-4*fb1*ffmcfmb**2*hbcm-4*fb2*ffmcfmb*fmc+4*ffmcfmb*
     . hbcm-fmc-6*hbcm))
      ans1=w11*(4*(p4p6*p3p4)*(-ffmcfmb*hbcm+fmc)+4*(p4p6*p2p3)*(
     . ffmcfmb*hbcm-fmc+2*hbcm)+4*p4p6*(ffmcfmb*fmc*hbcm2-fmc2*hbcm)+
     . 4*(p3p6*p3p4)*(-ffmcfmb**2*hbcm+ffmcfmb*fmc)+4*(p3p6*p2p3)*(
     . ffmcfmb**2*hbcm-ffmcfmb*fmc+2*ffmcfmb*hbcm)+4*p3p6*(ffmcfmb**2
     . *fmc*hbcm2-ffmcfmb*fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p2*p3p6+3*
     . ffmcfmb*hbcm*p2p4*p3p6+ffmcfmb*hbcm*p2p5*p3p6-hbcm*p1p2*p4p6+3
     . *hbcm*p2p4*p4p6+hbcm*p2p5*p4p6))+w1*(4*(p4p6*p3p4)*(-ffmcfmb*
     . hbcm+fmc)+4*p4p6*(ffmcfmb*fmc*hbcm2-fmc2*hbcm)+4*(p3p6*p3p4)*(
     . -ffmcfmb**2*hbcm+ffmcfmb*fmc)+4*p3p6*(ffmcfmb**2*fmc*hbcm2-
     . ffmcfmb*fmc2*hbcm)+16*(ffmcfmb*hbcm*p2p4*p3p6+hbcm*p2p4*p4p6))
     . +ans2
      ans=ccc*ans1
      b(3)=ans
      ans5=4*p6p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3+ffmcfmb
     . **2*hbcm4+2*ffmcfmb*fmb*hbcm3-ffmcfmb*fmb2*hbcm2+2*ffmcfmb*fmc
     . *hbcm3+ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4-2*ffmcfmb*p1p3s2+fmb*
     . fmc2*hbcm+fmb2*fmc*hbcm-fmc*hbcm3-fmc2*hbcm2)+8*(-ffmcfmb*p1p3
     . *p3p6*p5p7-p1p2*p1p3*p6p7-p1p3*p1p4*p6p7+p1p3*p2p3*p6p7+p1p3*
     . p2p4*p6p7+p1p3*p3p4*p6p7-3*p1p3*p4p6*p4p7-p1p3*p4p6*p5p7)
      ans4=(p4p6*p3p7*p1p3)*(-16*ffmcfmb-8)+(p4p7*p3p6*p1p3)*(8*
     . ffmcfmb-16)+8*(p5p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+24*(
     . p4p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p3p4)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p6p7*p2p4)*(ffmcfmb*hbcm2+fmb
     . *hbcm-hbcm2)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p6p7*p1p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p1p2)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm
     . -ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmc*hbcm+2*hbcm2)+8*(p3p7*
     . p3p6*p1p3)*(2*ffmcfmb**2-3*ffmcfmb)+8*(p4p6*p3p7)*(-2*ffmcfmb
     . **2*hbcm2-2*ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(
     . p5p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+
     . 8*(p3p7*p3p6)*(-2*ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm+2*ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7*p1p3)*(-ffmcfmb**2*
     . hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+2*
     . fmc*hbcm+fmc2)+ans5
      ans3=w13*ans4
      ans6=((p4p6*p3p7)*(32*fb2*ffmcfmb+16*fb2+8)+(p5p7*p3p6)*(16*fb2
     . *ffmcfmb+4)+(p4p7*p3p6)*(-16*fb2*ffmcfmb+32*fb2-4)+(p6p7*p1p3)
     . *(16*fb2*ffmcfmb+4)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+48*fb2*
     . ffmcfmb-16*ffmcfmb+12)+4*p6p7*(2*fb2*ffmcfmb**2*hbcm2-4*fb2*
     . ffmcfmb*hbcm2-2*fb2*fmc2+3*fmb*hbcm+3*fmc*hbcm-hbcm2)+16*(fb2*
     . p1p2*p6p7+fb2*p1p4*p6p7-fb2*p2p3*p6p7-fb2*p2p4*p6p7-fb2*p3p4*
     . p6p7+3*fb2*p4p6*p4p7+fb2*p4p6*p5p7))
      ans2=w1*((p4p6*p3p7*p3p4)*(-16*ffmcfmb-8)+(p4p7*p3p6*p3p4)*(8*
     . ffmcfmb-16)+8*(p4p7*p4p6)*(-2*ffmcfmb*hbcm2-3*fmc*hbcm)+8*(
     . p4p7*p3p6)*(-ffmcfmb*fmc*hbcm-2*fmc*hbcm)+8*(p3p7*p3p6*p3p4)*(
     . 2*ffmcfmb**2-3*ffmcfmb)+8*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p6p7*p3p4)*(-ffmcfmb**2*hbcm2+2*
     . ffmcfmb*hbcm2+2*fmc*hbcm+fmc2)+4*p6p7*(-ffmcfmb**2*fmc*hbcm3+2
     . *ffmcfmb*fmc*hbcm3+fmc3*hbcm+2*p3p4s2)+8*(-ffmcfmb*fmc*hbcm*
     . p1p3*p6p7-3*ffmcfmb*fmc*hbcm*p3p6*p3p7-ffmcfmb*fmc*hbcm*p3p6*
     . p5p7-ffmcfmb*p1p3*p3p4*p6p7-ffmcfmb*p3p4*p3p6*p5p7-fmc*hbcm*
     . p1p2*p6p7-fmc*hbcm*p1p4*p6p7+fmc*hbcm*p2p3*p6p7+fmc*hbcm*p2p4*
     . p6p7-fmc*hbcm*p4p6*p5p7-p1p2*p3p4*p6p7-p1p4*p3p4*p6p7+p2p3*
     . p3p4*p6p7+p2p4*p3p4*p6p7-3*p3p4*p4p6*p4p7-p3p4*p4p6*p5p7))+
     . ans3+ans6
      ans1=w11*((p4p6*p3p7*p3p4)*(-16*ffmcfmb-8)+(p4p7*p3p6*p3p4)*(8*
     . ffmcfmb-16)+(p4p6*p3p7*p2p3)*(16*ffmcfmb+8)+(p4p7*p3p6*p2p3)*(
     . -8*ffmcfmb+16)+8*(p4p7*p4p6)*(-2*ffmcfmb*hbcm2-3*fmc*hbcm)+8*(
     . p4p7*p3p6)*(-ffmcfmb*fmc*hbcm-2*fmc*hbcm)+8*(p3p7*p3p6*p3p4)*(
     . 2*ffmcfmb**2-3*ffmcfmb)+8*(p3p7*p3p6*p2p3)*(-2*ffmcfmb**2+3*
     . ffmcfmb)+8*(p4p6*p3p7)*(-2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmc*hbcm
     . -fmc*hbcm)+4*(p6p7*p3p4)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*hbcm2+2*
     . fmc*hbcm+fmc2)+4*(p6p7*p2p3)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2
     . +2*fmc*hbcm-fmc2)+4*p6p7*(-ffmcfmb**2*fmc*hbcm3+2*ffmcfmb*fmc*
     . hbcm3+fmc3*hbcm-2*p2p3s2+2*p3p4s2)+8*(-ffmcfmb*fmc*hbcm*p1p3*
     . p6p7-3*ffmcfmb*fmc*hbcm*p3p6*p3p7-ffmcfmb*fmc*hbcm*p3p6*p5p7+
     . ffmcfmb*p1p3*p2p3*p6p7-ffmcfmb*p1p3*p3p4*p6p7+ffmcfmb*p2p3*
     . p3p6*p5p7-ffmcfmb*p3p4*p3p6*p5p7-fmc*hbcm*p1p2*p6p7-fmc*hbcm*
     . p1p4*p6p7+fmc*hbcm*p2p4*p6p7-fmc*hbcm*p4p6*p5p7+p1p2*p2p3*p6p7
     . -p1p2*p3p4*p6p7+p1p4*p2p3*p6p7-p1p4*p3p4*p6p7-p2p3*p2p4*p6p7+3
     . *p2p3*p4p6*p4p7+p2p3*p4p6*p5p7+p2p4*p3p4*p6p7-3*p3p4*p4p6*p4p7
     . -p3p4*p4p6*p5p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans4=8*(3*ffmcfmb*fmc*hbcm*p1p3*p3p7-4*ffmcfmb*fmc*hbcm*p3p5*
     . p3p7-ffmcfmb*p1p3*p2p3*p3p7+ffmcfmb*p1p3*p3p4*p3p7+2*ffmcfmb*
     . p2p3*p3p4*p3p7+2*ffmcfmb*p2p3*p3p5*p3p7-2*ffmcfmb*p3p4*p3p5*
     . p3p7+2*fmc*hbcm*p1p3*p4p7-2*fmc*hbcm*p2p3*p5p7+fmc*hbcm*p3p4*
     . p5p7+hbcm2*p1p2*p4p7+hbcm2*p1p4*p4p7+3*hbcm2*p2p4*p5p7-4*hbcm2
     . *p2p5*p4p7-p1p3*p2p3*p4p7+p1p3*p3p4*p4p7-p1p4*p2p3*p3p7+p1p4*
     . p3p4*p3p7-p2p3*p2p5*p3p7+3*p2p3*p3p4*p4p7-3*p2p3*p3p4*p5p7+3*
     . p2p3*p3p5*p4p7+p2p5*p3p4*p3p7-3*p3p4*p3p5*p4p7)
      ans3=8*(p4p7*p3p5)*(-ffmcfmb*hbcm2-4*fmc*hbcm)+4*(p4p7*p3p4)*(
     . ffmcfmb*hbcm2-2*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+8*(p3p7*p2p5)*(-3
     . *ffmcfmb*hbcm2+2*fmc*hbcm)+8*(p3p7*p2p4)*(-ffmcfmb*hbcm2+fmc*
     . hbcm+hbcm2)+4*(p4p7*p2p3)*(5*ffmcfmb*hbcm2-2*fmb*hbcm+3*fmc*
     . hbcm)+8*(p3p7*p1p4)*(2*ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p1p2)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*p5p7*(ffmcfmb*fmc*hbcm3-fmc2*hbcm2+2
     . *p2p3s2+p3p4s2)+4*p4p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3-2*
     . fmb*fmc*hbcm2-4*fmc*hbcm3-2*fmc2*hbcm2-6*p2p3s2)+4*(p3p7*p3p4)
     . *(3*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm-5*
     . ffmcfmb*hbcm2-fmc*hbcm-fmc2)+4*(p3p7*p2p3)*(ffmcfmb**2*hbcm2-2
     . *ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm+
     . fmc2)+4*p3p7*(ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*fmb*fmc*hbcm2-5*
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm2-4*ffmcfmb*p2p3s2-fmc2*
     . hbcm2-fmc3*hbcm)+ans4
      ans2=w11*ans3
      ans8=8*(ffmcfmb*hbcm2*p1p3*p5p7+2*ffmcfmb*p1p3*p2p3*p3p7-2*
     . ffmcfmb*p1p3*p3p5*p3p7+p1p3*p1p4*p3p7+3*p1p3*p2p3*p4p7-2*p1p3*
     . p2p3*p5p7+p1p3*p2p5*p3p7+p1p3*p3p4*p5p7-3*p1p3*p3p5*p4p7)
      ans7=24*(p4p7*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p7*
     . p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p3p7*p2p5)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+16*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+24*(p4p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p3p7*
     . p1p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p7*p1p3)*(3*ffmcfmb*
     . hbcm2+4*fmb*hbcm+fmc*hbcm-6*hbcm2)+4*p4p7*(ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-3*ffmcfmb*hbcm4+fmb*fmc*hbcm2-4*fmb*hbcm3+
     . fmb2*hbcm2-fmc*hbcm3+3*hbcm4+2*p1p3s2)+8*p5p7*(ffmcfmb**2*
     . hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4)+16*(p3p7*p3p5)*(-
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+16*(p3p7*p2p3
     . )*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+4*(p3p7*
     . p1p3)*(3*ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*
     . hbcm-5*ffmcfmb*hbcm2-fmc*hbcm-fmc2)+4*p3p7*(ffmcfmb**2*fmb*
     . hbcm3+ffmcfmb**2*fmc*hbcm3-2*ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*
     . hbcm3+ffmcfmb*fmb2*hbcm2-ffmcfmb*fmc*hbcm3-ffmcfmb*fmc2*hbcm2+
     . 2*ffmcfmb*hbcm4+2*ffmcfmb*p1p3s2+fmb*fmc*hbcm2-fmb*fmc2*hbcm-
     . fmb2*fmc*hbcm+fmc2*hbcm2)+ans8
      ans6=w13*ans7
      ans9=((p3p7*p3p5)*(32*fb2*ffmcfmb+8)+(p3p7*p2p3)*(-32*fb2*
     . ffmcfmb-8)+(p3p7*p1p3)*(-16*fb2*ffmcfmb-4)+4*p5p7*(-4*fb2*
     . ffmcfmb*hbcm2-hbcm2)+2*p4p7*(8*fb1*fmb*hbcm+4*fb1*fmc*hbcm+4*
     . fb2*ffmcfmb*hbcm2+8*fb2*hbcm2-hbcm2)+2*p3p7*(8*fb1*ffmcfmb*fmb
     . *hbcm+4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2+4*fb2*ffmcfmb*
     . hbcm2+4*fb2*fmc2-4*fmb*hbcm-6*fmc*hbcm+hbcm2)+16*(-fb2*p1p3*
     . p4p7-fb2*p1p4*p3p7-3*fb2*p2p3*p4p7+2*fb2*p2p3*p5p7-fb2*p2p5*
     . p3p7-fb2*p3p4*p5p7+3*fb2*p3p5*p4p7))
      ans5=w1*(8*(p4p7*p3p5)*(-ffmcfmb*hbcm2-4*fmc*hbcm)+4*(p4p7*p3p4
     . )*(ffmcfmb*hbcm2-2*fmb*hbcm-3*fmc*hbcm-4*hbcm2)+8*(p4p7*p2p3)*
     . (ffmcfmb*hbcm2+4*fmc*hbcm)+8*(p3p7*p1p4)*(2*ffmcfmb*hbcm2+fmc*
     . hbcm)+8*p5p7*(ffmcfmb*fmc*hbcm3-fmc2*hbcm2+p3p4s2)+4*p4p7*(-
     . ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-4*fmc*hbcm3
     . -2*fmc2*hbcm2)+4*(p3p7*p3p4)*(3*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb
     . *hbcm-4*ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2-fmc*hbcm-fmc2)+4*p3p7
     . *(ffmcfmb**2*fmc*hbcm3-2*ffmcfmb*fmb*fmc*hbcm2-5*ffmcfmb*fmc*
     . hbcm3-2*ffmcfmb*fmc2*hbcm2-fmc2*hbcm2-fmc3*hbcm)+8*(3*ffmcfmb*
     . fmc*hbcm*p1p3*p3p7+4*ffmcfmb*fmc*hbcm*p2p3*p3p7-4*ffmcfmb*fmc*
     . hbcm*p3p5*p3p7+ffmcfmb*p1p3*p3p4*p3p7+2*ffmcfmb*p2p3*p3p4*p3p7
     . -2*ffmcfmb*p3p4*p3p5*p3p7+2*fmc*hbcm*p1p3*p4p7-2*fmc*hbcm*p2p3
     . *p5p7+fmc*hbcm*p2p5*p3p7+fmc*hbcm*p3p4*p5p7+hbcm2*p1p4*p4p7+2*
     . hbcm2*p2p4*p5p7-hbcm2*p2p5*p4p7+p1p3*p3p4*p4p7+p1p4*p3p4*p3p7+
     . 3*p2p3*p3p4*p4p7-2*p2p3*p3p4*p5p7+p2p5*p3p4*p3p7-3*p3p4*p3p5*
     . p4p7))+ans6+ans9
      ans1=ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans5=8*(ffmcfmb*p1p3*p3p5*p3p6-2*fmb*hbcm*p1p2*p3p6+fmb*hbcm*
     . p1p4*p3p6+3*fmb*hbcm*p2p5*p3p6-2*p1p2*p1p3*p3p6+p1p3*p1p4*p3p6
     . -3*p1p3*p2p3*p4p6+3*p1p3*p2p5*p3p6+p1p3*p3p4*p4p6+p1p3*p3p5*
     . p4p6)
      ans4=(p3p6*p3p4*p1p3)*(8*ffmcfmb-8)+(p3p6*p2p3*p1p3)*(-24*
     . ffmcfmb+16)+8*(p4p6*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p4p6*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6*p2p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p6*p1p3)*(ffmcfmb*hbcm2-2*
     . fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(p3p6*p3p4)*(ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+4*p4p6*(2*ffmcfmb**2*hbcm4
     . +ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2+2*fmb*hbcm3-fmb2*hbcm2+fmc*hbcm3-hbcm4-2*p1p3s2)+8*(p3p6
     . *p3p5)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+8*(
     . p3p6*p2p3)*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc
     . *hbcm+ffmcfmb*hbcm2+2*fmb*hbcm+2*fmc*hbcm)+4*(p3p6*p1p3)*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm)+4*p3p6*(ffmcfmb**2*fmb*
     . hbcm3+ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4-ffmcfmb*fmb*fmc*
     . hbcm2-ffmcfmb*fmb2*hbcm2-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4)+ans5
      ans3=w13*ans4
      ans6=((p3p6*p3p5)*(-16*fb2*ffmcfmb-4)+(p3p6*p3p4)*(-16*fb2*
     . ffmcfmb+16*fb2-4)+(p3p6*p2p3)*(48*fb2*ffmcfmb-32*fb2+12)+2*
     . p4p6*(-4*fb1*fmc*hbcm-4*fb2*ffmcfmb*hbcm2-8*fb2*hbcm2-3*hbcm2)
     . +2*p3p6*(-4*fb1*ffmcfmb*fmc*hbcm-4*fb2*ffmcfmb**2*hbcm2+2*
     . ffmcfmb*hbcm2-6*fmb*hbcm-fmc*hbcm-6*hbcm2)+16*(2*fb2*p1p2*p3p6
     . +fb2*p1p3*p4p6-fb2*p1p4*p3p6+3*fb2*p2p3*p4p6-3*fb2*p2p5*p3p6-
     . fb2*p3p4*p4p6-fb2*p3p5*p4p6))
      ans2=w1*((p3p6*p3p4*p2p3)*(-24*ffmcfmb+16)+4*(p4p6*p3p4)*(-
     . ffmcfmb*hbcm2+3*fmc*hbcm+2*hbcm2)+4*p4p6*(ffmcfmb*fmc*hbcm3+2*
     . fmc*hbcm3-fmc2*hbcm2+2*p3p4s2)+8*(p3p6*p2p3)*(-3*ffmcfmb*fmc*
     . hbcm+2*fmc*hbcm)+4*(p3p6*p3p4)*(ffmcfmb**2*hbcm2+ffmcfmb*fmc*
     . hbcm-2*fmc*hbcm)+4*p3p6*(ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmc2*
     . hbcm2+2*ffmcfmb*p3p4s2-2*fmc3*hbcm-2*p3p4s2)+8*(ffmcfmb*fmc*
     . hbcm*p3p5*p3p6+ffmcfmb*p3p4*p3p5*p3p6-2*fmc*hbcm*p1p2*p3p6-fmc
     . *hbcm*p1p3*p4p6+fmc*hbcm*p1p4*p3p6-3*fmc*hbcm*p2p3*p4p6+2*fmc*
     . hbcm*p2p4*p3p6+3*fmc*hbcm*p2p5*p3p6+fmc*hbcm*p3p5*p4p6+2*hbcm2
     . *p2p4*p4p6-2*p1p2*p3p4*p3p6-p1p3*p3p4*p4p6+p1p4*p3p4*p3p6-3*
     . p2p3*p3p4*p4p6+3*p2p5*p3p4*p3p6+p3p4*p3p5*p4p6))+ans3+ans6
      ans1=w11*((p3p6*p3p4*p2p3)*(-32*ffmcfmb+24)+4*(p4p6*p3p4)*(-
     . ffmcfmb*hbcm2+3*fmc*hbcm+2*hbcm2)+8*(p3p6*p2p5)*(ffmcfmb*hbcm2
     . +3*fmc*hbcm)+4*(p4p6*p2p3)*(-ffmcfmb*hbcm2-5*fmc*hbcm)+8*(p3p6
     . *p1p2)*(-ffmcfmb*hbcm2-2*fmc*hbcm)+4*p4p6*(ffmcfmb*fmc*hbcm3+2
     . *fmc*hbcm3-fmc2*hbcm2+6*p2p3s2+2*p3p4s2)+4*(p3p6*p3p4)*(
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm-2*fmc*hbcm)+4*(p3p6*p2p3)*(-
     . 3*ffmcfmb**2*hbcm2-3*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+4*fmc*
     . hbcm)+4*p3p6*(ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmc2*hbcm2+6*
     . ffmcfmb*p2p3s2+2*ffmcfmb*p3p4s2-2*fmc3*hbcm-4*p2p3s2-2*p3p4s2)
     . +8*(ffmcfmb*fmc*hbcm*p3p5*p3p6-ffmcfmb*p2p3*p3p5*p3p6+ffmcfmb*
     . p3p4*p3p5*p3p6-fmc*hbcm*p1p3*p4p6+fmc*hbcm*p1p4*p3p6+3*fmc*
     . hbcm*p2p4*p3p6+fmc*hbcm*p3p5*p4p6-hbcm2*p1p2*p4p6+3*hbcm2*p2p4
     . *p4p6+hbcm2*p2p5*p4p6+2*p1p2*p2p3*p3p6-2*p1p2*p3p4*p3p6+p1p3*
     . p2p3*p4p6-p1p3*p3p4*p4p6-p1p4*p2p3*p3p6+p1p4*p3p4*p3p6-3*p2p3*
     . p2p5*p3p6-4*p2p3*p3p4*p4p6-p2p3*p3p5*p4p6+3*p2p5*p3p4*p3p6+
     . p3p4*p3p5*p4p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w13*(4*p3p4*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+12*p2p5*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p2p3*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+4*p1p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*p1p2*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+4*p1p3*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm+2*ffmcfmb*hbcm2)+4*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-
     . ffmcfmb*hbcm4-ffmcfmb*p1p3s2+2*p1p2*p1p3-p1p3*p1p4-2*p1p3*p2p3
     . -3*p1p3*p2p5+p1p3*p3p4))+(p1p3*(8*fb2*ffmcfmb+2)+2*(-4*fb2*
     . ffmcfmb*hbcm2-8*fb2*p1p2+4*fb2*p1p4+8*fb2*p2p3+12*fb2*p2p5-4*
     . fb2*p3p4+3*fmb*hbcm+3*fmc*hbcm-hbcm2))
      ans1=w1*(8*p2p4*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p4*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+4*(-ffmcfmb*fmc
     . *hbcm*p1p3+ffmcfmb*fmc*hbcm3-ffmcfmb*fmc2*hbcm2-ffmcfmb*p1p3*
     . p3p4+2*fmc*hbcm*p1p2-fmc*hbcm*p1p4-2*fmc*hbcm*p2p3-3*fmc*hbcm*
     . p2p5+fmc3*hbcm+2*p1p2*p3p4-p1p4*p3p4-2*p2p3*p3p4-3*p2p5*p3p4+
     . p3p4s2))+w11*(12*p2p4*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p4*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+4*
     . p2p3*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmc*
     . hbcm)+4*(-ffmcfmb*fmc*hbcm*p1p3+ffmcfmb*fmc*hbcm3-ffmcfmb*fmc2
     . *hbcm2+ffmcfmb*p1p3*p2p3-ffmcfmb*p1p3*p3p4+2*fmc*hbcm*p1p2-fmc
     . *hbcm*p1p4-3*fmc*hbcm*p2p5+fmc3*hbcm-2*p1p2*p2p3+2*p1p2*p3p4+
     . p1p4*p2p3-p1p4*p3p4+3*p2p3*p2p5-3*p2p3*p3p4+2*p2p3s2-3*p2p5*
     . p3p4+p3p4s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=w13*(12*p2p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*p1p4*(
     . ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*p1p2*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+4*(p3p4*p1p3)*(-ffmcfmb*hbcm-fmc+hbcm)+8*(p2p3*
     . p1p3)*(ffmcfmb*hbcm+fmc-hbcm)+4*p3p4*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+
     . 8*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm+fmb*
     . hbcm2-fmb2*hbcm-fmc*hbcm2)+4*p1p3*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2)+4*(ffmcfmb**2*fmb*hbcm4-ffmcfmb**2*fmc*hbcm4-
     . ffmcfmb*fmb*fmc*hbcm3-ffmcfmb*fmb*hbcm4+ffmcfmb*fmb2*hbcm3+
     . ffmcfmb*fmc*hbcm4+ffmcfmb*hbcm*p1p3s2-2*hbcm*p1p2*p1p3+hbcm*
     . p1p3*p1p4+3*hbcm*p1p3*p2p5))+(2*p3p4*(4*fb1*ffmcfmb*hbcm-4*fb1
     . *hbcm+4*fb2*fmc+hbcm)+8*p2p3*(-2*fb1*ffmcfmb*hbcm+2*fb1*hbcm-2
     . *fb2*fmc+hbcm)+2*p1p3*(4*fb1*ffmcfmb*hbcm+hbcm)+2*(4*fb1*
     . ffmcfmb**2*hbcm3-4*fb1*ffmcfmb*hbcm3-8*fb1*hbcm*p1p2+4*fb1*
     . hbcm*p1p4+12*fb1*hbcm*p2p5+4*fb2*ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3-3*fmb*hbcm2+fmc*hbcm2-6*hbcm*p3p5-hbcm3))
      ans2=w1*(8*p2p4*(-ffmcfmb*hbcm3-fmc*hbcm2+2*hbcm3)+8*(p3p4*p2p3
     . )*(2*ffmcfmb*hbcm+fmc-hbcm)+4*p3p4*(-ffmcfmb*fmc*hbcm2-ffmcfmb
     . *hbcm3+fmc*hbcm2-fmc2*hbcm)+8*p2p3*(ffmcfmb*fmc*hbcm2-fmc*
     . hbcm2+2*fmc2*hbcm)+4*p1p3*(-ffmcfmb*fmc*hbcm2+2*fmc2*hbcm)+4*(
     . -ffmcfmb**2*fmc*hbcm4+ffmcfmb*fmc*hbcm4+ffmcfmb*hbcm*p1p3*p3p4
     . -2*ffmcfmb*hbcm*p3p4*p3p5-ffmcfmb*hbcm*p3p4s2+2*fmc*hbcm2*p1p2
     . -fmc*hbcm2*p1p4-3*fmc*hbcm2*p2p5-fmc*p3p4s2-2*fmc2*hbcm*p3p5-2
     . *fmc2*hbcm3+fmc3*hbcm2+2*hbcm*p1p2*p3p4-4*hbcm*p1p3*p2p4-hbcm*
     . p1p4*p3p4-4*hbcm*p2p3*p2p4+4*hbcm*p2p4*p3p5-3*hbcm*p2p5*p3p4+
     . hbcm*p3p4s2))+ans3
      ans1=w11*(4*p2p5*(-2*ffmcfmb*hbcm3-3*fmc*hbcm2)+12*p2p4*(-
     . ffmcfmb*hbcm3-fmc*hbcm2+2*hbcm3)+8*p1p2*(ffmcfmb*hbcm3+fmc*
     . hbcm2)+4*(p3p4*p2p3)*(5*ffmcfmb*hbcm+3*fmc-3*hbcm)+4*p3p4*(-
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+4*p2p3*(2
     . *ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-2*fmc*hbcm2+4*fmc2*hbcm)+4*
     . p1p3*(-ffmcfmb*fmc*hbcm2+2*fmc2*hbcm)+4*(-ffmcfmb**2*fmc*hbcm4
     . +ffmcfmb*fmc*hbcm4-3*ffmcfmb*hbcm*p1p3*p2p3+ffmcfmb*hbcm*p1p3*
     . p3p4+2*ffmcfmb*hbcm*p2p3*p3p5-4*ffmcfmb*hbcm*p2p3s2-2*ffmcfmb*
     . hbcm*p3p4*p3p5-ffmcfmb*hbcm*p3p4s2-fmc*hbcm2*p1p4-2*fmc*p2p3s2
     . -fmc*p3p4s2-2*fmc2*hbcm*p3p5-2*fmc2*hbcm3+fmc3*hbcm2-2*hbcm*
     . p1p2*p2p3+4*hbcm*p1p2*p3p4-6*hbcm*p1p3*p2p4-hbcm*p1p4*p2p3-
     . hbcm*p1p4*p3p4-6*hbcm*p2p3*p2p4+hbcm*p2p3*p2p5+2*hbcm*p2p3s2+6
     . *hbcm*p2p4*p3p5-5*hbcm*p2p5*p3p4+hbcm*p3p4s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(4*(p3p6*p3p4)*(ffmcfmb*hbcm+3*fmc)+4*p3p6*(-
     . ffmcfmb*fmc*hbcm2+5*fmc2*hbcm)+16*(-fmc*hbcm2*p4p6-hbcm*p2p4*
     . p3p6-hbcm*p3p4*p4p6))+w13*(16*p4p6*(ffmcfmb*hbcm3+fmb*hbcm2-
     . hbcm3)+24*(p3p6*p3p5)*(ffmcfmb*hbcm-hbcm)+8*(p3p6*p3p4)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+4*(p3p6*
     . p1p3)*(ffmcfmb*hbcm+3*fmc)+4*p3p6*(-ffmcfmb**2*hbcm3+ffmcfmb*
     . fmb*hbcm2+4*ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*fmb2*hbcm-3*hbcm3)+
     . 16*hbcm*p1p3*p4p6)+w11*(4*p4p6*(-ffmcfmb*hbcm3-3*fmc*hbcm2-2*
     . hbcm3)+4*(p3p6*p3p4)*(-ffmcfmb*hbcm+3*fmc+2*hbcm)+4*(p3p6*p2p3
     . )*(-3*ffmcfmb*hbcm-3*fmc+2*hbcm)+4*p3p6*(-ffmcfmb**2*hbcm3+5*
     . fmc2*hbcm)+8*(-ffmcfmb*hbcm*p3p5*p3p6-hbcm*p1p2*p3p6+hbcm*p1p3
     . *p4p6-hbcm*p1p4*p3p6+hbcm*p2p3*p4p6-3*hbcm*p2p4*p3p6-3*hbcm*
     . p3p4*p4p6-hbcm*p3p5*p4p6))+(2*p3p6*(4*fb1*ffmcfmb*hbcm-12*fb2*
     . fmc-5*hbcm)+32*fb1*hbcm*p4p6))
      b(9)=ans
      b(10)=ccc*(16*w1*(ffmcfmb*fmc*hbcm*p3p6+ffmcfmb*p3p4*p3p6+fmc*
     . hbcm*p4p6+p3p4*p4p6)+w13*(16*p4p6*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+4*p3p6*(ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+3*ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2-3*fmc*hbcm)+16*(ffmcfmb*p1p3*p3p6+p1p3*
     . p4p6))+w11*(4*p4p6*(ffmcfmb*hbcm2+3*fmc*hbcm)+4*p3p6*(ffmcfmb
     . **2*hbcm2+3*ffmcfmb*fmc*hbcm)+16*(-ffmcfmb*p2p3*p3p6+ffmcfmb*
     . p3p4*p3p6-p2p3*p4p6+p3p4*p4p6))+(p3p6*(-32*fb2*ffmcfmb-8)-32*
     . fb2*p4p6))
      ans2=(-12*fb1*fmc*hbcm+20*fb2*ffmcfmb*hbcm2-24*fb2*hbcm2+24*fb2
     . *p1p3+8*fb2*p2p3+8*fb2*p3p4-24*fb2*p3p5-hbcm2)
      ans1=w1*(2*p3p4*(-3*ffmcfmb*hbcm2+fmc*hbcm+6*hbcm2)+2*(-5*
     . ffmcfmb*fmc*hbcm3-6*fmc*hbcm*p1p3-2*fmc*hbcm*p2p3+6*fmc*hbcm*
     . p3p5+6*fmc*hbcm3+5*fmc2*hbcm2-4*hbcm2*p2p4-6*p1p3*p3p4-2*p2p3*
     . p3p4+6*p3p4*p3p5-2*p3p4s2))+w13*(12*p3p5*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+4*p3p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p3*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+2*p1p3*(-5*ffmcfmb*hbcm2-6*fmb*
     . hbcm-3*fmc*hbcm+6*hbcm2)+2*(-2*ffmcfmb**2*hbcm4-5*ffmcfmb*fmb*
     . hbcm3-3*ffmcfmb*fmc*hbcm3+5*ffmcfmb*hbcm4-3*fmb*fmc*hbcm2+6*
     . fmb*hbcm3-3*fmb2*hbcm2+3*fmc*hbcm3-3*hbcm4-2*p1p3*p2p3-2*p1p3*
     . p3p4+6*p1p3*p3p5-6*p1p3s2))+w11*(2*p3p4*(-5*ffmcfmb*hbcm2+3*
     . fmc*hbcm+8*hbcm2)+2*p2p3*(ffmcfmb*hbcm2-3*fmc*hbcm-4*hbcm2)+4*
     . p1p3*(-ffmcfmb*hbcm2-3*fmc*hbcm)+2*(-2*ffmcfmb**2*hbcm4-3*
     . ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4+6*fmc*hbcm*p3p5+6*fmc*hbcm3+
     . 5*fmc2*hbcm2-2*hbcm2*p1p2-2*hbcm2*p1p4-6*hbcm2*p2p4+6*p1p3*
     . p2p3-6*p1p3*p3p4-6*p2p3*p3p5+2*p2p3s2+6*p3p4*p3p5-2*p3p4s2))+
     . ans2
      ans=ccc*ans1
      b(11)=ans
      ans2=w11*(4*p5p7*(2*ffmcfmb*hbcm3+3*fmc*hbcm2)+4*p4p7*(3*
     . ffmcfmb*hbcm3-2*fmb*hbcm2+2*fmc*hbcm2-8*hbcm3)+4*(p3p7*p3p4)*(
     . -6*ffmcfmb*hbcm-2*fmc+hbcm)+4*(p3p7*p2p3)*(6*ffmcfmb*hbcm+2*
     . fmc-hbcm)+4*p3p7*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-5*ffmcfmb*hbcm3+2*fmc*hbcm2-3*fmc2*hbcm)+4*(
     . 6*ffmcfmb*hbcm*p1p3*p3p7-8*ffmcfmb*hbcm*p3p5*p3p7+2*hbcm*p1p2*
     . p3p7+8*hbcm*p1p3*p4p7+2*hbcm*p1p4*p3p7+7*hbcm*p2p3*p4p7-hbcm*
     . p2p3*p5p7+2*hbcm*p2p4*p3p7-3*hbcm*p3p4*p4p7+5*hbcm*p3p4*p5p7-
     . 12*hbcm*p3p5*p4p7))+(8*p3p7*(2*fb1*ffmcfmb*hbcm-fb1*hbcm+2*fb2
     . *fmc-hbcm)+8*(fb1*hbcm*p4p7-3*fb1*hbcm*p5p7))
      ans1=w1*(4*p4p7*(2*ffmcfmb*hbcm3-fmc*hbcm2-2*hbcm3)+4*(p3p7*
     . p3p4)*(-6*ffmcfmb*hbcm-2*fmc+hbcm)+4*p3p7*(-4*ffmcfmb*fmc*
     . hbcm2+fmc*hbcm2-2*fmc2*hbcm)+4*(3*fmc*hbcm2*p5p7+2*hbcm*p1p3*
     . p4p7+2*hbcm*p2p3*p4p7-3*hbcm*p3p4*p4p7+3*hbcm*p3p4*p5p7-2*hbcm
     . *p3p5*p4p7))+w13*(4*(p3p7*p1p3)*(-2*fmc+hbcm)+12*p5p7*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*p4p7*(ffmcfmb*hbcm3+fmb*hbcm2
     . -hbcm3)+4*p3p7*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb
     . *fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+2*
     . fmc*hbcm2)+4*(hbcm*p1p3*p4p7-3*hbcm*p1p3*p5p7))+ans2
      ans=ccc*ans1
      b(12)=ans
      b(13)=ccc*(w11*(2*p3p4*(-5*ffmcfmb*hbcm-3*fmc-2*hbcm)+2*p2p3*(5
     . *ffmcfmb*hbcm+3*fmc-2*hbcm)+2*(-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*
     . hbcm*p1p3-2*ffmcfmb*hbcm3-5*fmc2*hbcm+2*hbcm*p1p2+2*hbcm*p1p4+
     . 6*hbcm*p2p4))+w1*(2*p3p4*(-5*ffmcfmb*hbcm-3*fmc)+2*(-3*ffmcfmb
     . *fmc*hbcm2-5*fmc2*hbcm+4*hbcm*p2p4))+w13*(6*p1p3*(-ffmcfmb*
     . hbcm-fmc+2*hbcm)+6*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3))+3*(4*
     . fb1*ffmcfmb*hbcm+4*fb2*fmc-hbcm))
      ans=ccc*(w1*((p3p7*p3p4)*(-8*ffmcfmb+4)+4*p4p7*(-2*ffmcfmb*
     . hbcm2-3*fmc*hbcm)+4*p3p7*(-4*ffmcfmb*fmc*hbcm+fmc*hbcm)+12*(
     . fmc*hbcm*p5p7-p3p4*p4p7+p3p4*p5p7))+w13*((p3p7*p1p3)*(-8*
     . ffmcfmb+4)+12*p5p7*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+12*p4p7*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p3p7*(-2*ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm+3*ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+12*(-p1p3*
     . p4p7+p1p3*p5p7))+w11*((p3p7*p3p4)*(-8*ffmcfmb+4)+(p3p7*p2p3)*(
     . 8*ffmcfmb-4)+4*p4p7*(-5*ffmcfmb*hbcm2+2*fmb*hbcm+2*fmc*hbcm)+4
     . *p3p7*(-2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2+2*
     . fmc*hbcm)+12*(fmc*hbcm*p5p7+p2p3*p4p7-p2p3*p5p7-p3p4*p4p7+p3p4
     . *p5p7))+(p3p7*(16*fb2*ffmcfmb-8*fb2+4)+24*(fb2*p4p7-fb2*p5p7))
     . )
      b(14)=ans
      ans2=w11*(8*(p4p6*p3p7)*(2*fmc*hbcm-hbcm2)+(p3p7*p3p6*p3p4)*(16
     . *ffmcfmb-8)+(p3p7*p3p6*p2p3)*(-16*ffmcfmb+8)+8*(p5p7*p3p6)*(-
     . ffmcfmb*hbcm2-3*fmc*hbcm)+16*(p4p7*p3p6)*(ffmcfmb*hbcm2-fmb*
     . hbcm-fmc*hbcm-hbcm2)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm2-fmc*hbcm+
     . hbcm2)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+4*p6p7*(-
     . ffmcfmb**2*hbcm4-2*ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4+3*fmc2*
     . hbcm2-4*p3p4s2)+16*(p3p7*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm)+8*(-ffmcfmb*
     . hbcm2*p1p3*p6p7-hbcm2*p1p2*p6p7-hbcm2*p1p4*p6p7-hbcm2*p2p4*
     . p6p7-3*hbcm2*p4p6*p4p7-hbcm2*p4p6*p5p7+2*p2p3*p3p4*p6p7-p2p3*
     . p3p6*p4p7+3*p2p3*p3p6*p5p7-2*p2p3*p3p7*p4p6+p3p4*p3p6*p4p7-3*
     . p3p4*p3p6*p5p7+2*p3p4*p3p7*p4p6))+((p3p7*p3p6)*(-32*fb2*
     . ffmcfmb+16*fb2-8)+16*p6p7*(-fb1*fmc*hbcm+fb2*ffmcfmb*hbcm2)+16
     . *(2*fb2*p3p4*p6p7-fb2*p3p6*p4p7+3*fb2*p3p6*p5p7-2*fb2*p3p7*
     . p4p6))
      ans1=w1*((p3p7*p3p6*p3p4)*(16*ffmcfmb-8)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p4)*(-ffmcfmb*hbcm2-fmc*hbcm
     . )+8*p6p7*(-ffmcfmb*fmc*hbcm3+fmc2*hbcm2-2*p3p4s2)+8*(p3p7*p3p6
     . )*(4*ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(fmc*hbcm*p3p6*p4p7-3*fmc*
     . hbcm*p3p6*p5p7+p3p4*p3p6*p4p7-3*p3p4*p3p6*p5p7+2*p3p4*p3p7*
     . p4p6))+w13*(8*(p6p7*p1p3)*(-fmc*hbcm-hbcm2)+(p3p7*p3p6*p1p3)*(
     . 16*ffmcfmb-8)+16*(p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*
     . (p4p7*p3p6)*(2*ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+16*(p6p7*p3p4)*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p6p7*(-ffmcfmb**2*hbcm4-2*
     . ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-fmb2*
     . hbcm2+2*fmc*hbcm3+hbcm4)+8*(p3p7*p3p6)*(2*ffmcfmb**2*hbcm2+2*
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2-fmb*hbcm-2
     . *fmc*hbcm)+8*(-3*fmb*hbcm*p3p6*p5p7-2*p1p3*p3p4*p6p7+p1p3*p3p6
     . *p4p7-3*p1p3*p3p6*p5p7+2*p1p3*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w1*(8*(p6p7*p3p4)*(ffmcfmb*hbcm+fmc)+8*p6p7*(ffmcfmb*
     . fmc*hbcm2+fmc2*hbcm)+16*(-ffmcfmb*hbcm*p3p6*p4p7+ffmcfmb*hbcm*
     . p3p7*p4p6))+w13*(8*(p6p7*p1p3)*(fmc-hbcm)+24*(p5p7*p3p6)*(
     . ffmcfmb*hbcm-hbcm)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*(p3p7*
     . p3p6)*(ffmcfmb*hbcm-hbcm)+4*p6p7*(-ffmcfmb**2*hbcm3-2*ffmcfmb*
     . fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+2*fmb*fmc*hbcm-fmb2*hbcm-2*fmc*
     . hbcm2+hbcm3))+w11*(8*(p4p6*p3p7)*(-2*ffmcfmb*hbcm-hbcm)+8*(
     . p4p7*p3p6)*(-3*ffmcfmb*hbcm-2*hbcm)+8*(p6p7*p3p4)*(ffmcfmb*
     . hbcm+fmc+hbcm)+8*(p6p7*p2p3)*(-ffmcfmb*hbcm-fmc+hbcm)+4*p6p7*(
     . -ffmcfmb**2*hbcm3+2*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+3*fmc2*
     . hbcm)+8*(-ffmcfmb*hbcm*p1p3*p6p7-3*ffmcfmb*hbcm*p3p6*p3p7-
     . ffmcfmb*hbcm*p3p6*p5p7-hbcm*p1p2*p6p7-hbcm*p1p4*p6p7-hbcm*p2p4
     . *p6p7-5*hbcm*p4p6*p4p7-hbcm*p4p6*p5p7))+8*p6p7*(-2*fb1*ffmcfmb
     . *hbcm-2*fb2*fmc+hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp32_3p0(cc) 
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
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      ans5=8*(2*ffmcfmb*hbcm*p1p3*p2p3*p6p7-2*ffmcfmb*hbcm*p1p3*p3p5*
     . p6p7-2*ffmcfmb*hbcm*p2p3*p3p4*p6p7+4*ffmcfmb*hbcm*p2p3*p3p6*
     . p5p7-2*ffmcfmb*hbcm*p2p5*p3p6*p3p7+2*ffmcfmb*hbcm*p3p4*p3p5*
     . p6p7+ffmcfmb*hbcm3*p2p5*p6p7-ffmcfmb*hbcm3*p5p6*p5p7+hbcm*p1p3
     . *p1p4*p6p7-hbcm*p1p3*p2p5*p6p7-hbcm*p1p3*p4p6*p4p7+4*hbcm*p1p3
     . *p4p6*p5p7+hbcm*p1p3*p5p6*p5p7+2*hbcm*p1p4*p2p3*p6p7-hbcm*p1p4
     . *p3p4*p6p7-2*hbcm*p1p4*p3p5*p6p7-2*hbcm*p1p4*p3p7*p4p6-2*hbcm*
     . p2p3*p4p6*p4p7+6*hbcm*p2p3*p4p6*p5p7-2*hbcm*p2p4*p3p6*p5p7+
     . hbcm*p2p5*p3p4*p6p7+2*hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*p3p7*
     . p4p6+hbcm*p3p4*p4p6*p4p7-4*hbcm*p3p4*p4p6*p5p7-hbcm*p3p4*p5p6*
     . p5p7+2*hbcm*p3p5*p4p6*p4p7-2*hbcm*p3p5*p4p6*p5p7-2*hbcm*p3p6*
     . p4p5*p5p7+hbcm3*p4p6*p4p7)
      ans4=4*(p6p7*p1p3)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*hbcm3-fmc2*hbcm)+8*(p3p7*p3p6*p3p4)*(-ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+2*ffmcfmb*fmc+7*ffmcfmb*hbcm-fmc-2*hbcm)+8*(p3p7*
     . p3p6*p1p3)*(ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc-3*
     . ffmcfmb*hbcm+fmc)+4*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*
     . hbcm5+2*ffmcfmb*hbcm*p1p3s2+2*ffmcfmb*hbcm*p3p4s2+fmb*fmc2*
     . hbcm2-2*fmb*p1p3s2-2*fmb*p3p4s2+fmc2*hbcm3)+8*(p3p7*p3p6)*(-
     . ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+3*ffmcfmb*fmc2*hbcm+fmb*
     . fmc*hbcm-fmc*hbcm2)+ans5
      ans3=8*(p5p7*p3p6*p1p3)*(fmc+hbcm)+8*(p6p7*p1p4)*(-fmb*hbcm2-
     . hbcm3)+8*(p5p7*p4p6)*(-3*ffmcfmb*hbcm3-hbcm3)+16*(p4p6*p3p7*
     . p3p5)*(2*ffmcfmb*hbcm-hbcm)+8*(p4p6*p3p7*p3p4)*(5*ffmcfmb*hbcm
     . +fmb-2*hbcm)+8*(p5p7*p3p6*p3p4)*(2*ffmcfmb*hbcm-fmc-5*hbcm)+8*
     . (p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+fmc+hbcm)+16*(p4p6*p3p7*
     . p2p3)*(-2*ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p1p4)*(-ffmcfmb*
     . hbcm-hbcm)+8*(p4p6*p3p7*p1p3)*(-5*ffmcfmb*hbcm-fmb+2*hbcm)+8*(
     . p4p7*p3p6*p1p3)*(ffmcfmb*hbcm+fmb-fmc+hbcm)+16*(p6p7*p3p4*p1p3
     . )*(-ffmcfmb*hbcm+fmb)+8*(p4p7*p3p6)*(-2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+8*(p4p6*p3p7)*
     . (2*ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3+fmc2*hbcm-
     . hbcm3)+8*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+8*(p6p7*p3p5)*
     . (ffmcfmb**2*hbcm3+fmc2*hbcm)+4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3-2
     . *ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmc2*hbcm)+8*(p6p7*p2p3)*(-
     . ffmcfmb**2*hbcm3-fmc2*hbcm)+ans4
      ans2=w5*ans3
      ans8=8*(p5p7*p3p6)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb*hbcm2+2*hbcm3)+4*(p6p7*p3p5)*(
     . ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-fmc2*hbcm)+8*(p3p7*p3p6*p3p5)
     . *(ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm+fmc-2
     . *hbcm)+4*p6p7*(-ffmcfmb**2*fmb*hbcm4+2*ffmcfmb*fmb2*hbcm3-4*
     . ffmcfmb*hbcm*p3p5s2-fmb*fmc2*hbcm2)+8*(p3p7*p3p6)*(-ffmcfmb**2
     . *fmb*hbcm2-2*ffmcfmb*fmb*fmc*hbcm+3*ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmb2*hbcm+fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm)+8*(2*ffmcfmb*hbcm*
     . p2p3*p3p5*p6p7+4*ffmcfmb*hbcm*p2p3*p3p6*p5p7+fmb*hbcm2*p1p4*
     . p6p7+2*fmb*hbcm2*p4p6*p4p7-fmb*hbcm2*p5p6*p5p7-2*hbcm*p1p2*
     . p3p6*p5p7+hbcm*p1p4*p3p5*p6p7-2*hbcm*p1p5*p2p3*p6p7+2*hbcm*
     . p1p5*p3p5*p6p7-2*hbcm*p1p5*p3p6*p4p7+2*hbcm*p1p5*p3p6*p5p7+2*
     . hbcm*p1p5*p3p7*p4p6+2*hbcm*p2p3*p2p5*p6p7+2*hbcm*p2p3*p4p5*
     . p6p7+2*hbcm*p2p3*p4p6*p5p7-2*hbcm*p2p3*p4p7*p5p6-hbcm*p2p5*
     . p3p5*p6p7-2*hbcm*p2p5*p3p7*p4p6-2*hbcm*p3p5*p4p5*p6p7+3*hbcm*
     . p3p5*p4p6*p4p7+2*hbcm*p3p5*p4p7*p5p6-hbcm*p3p5*p5p6*p5p7-2*
     . hbcm*p3p6*p4p5*p5p7-2*hbcm*p3p7*p4p5*p4p6)
      ans7=8*(p5p6*p4p7)*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+8*(p5p7*
     . p4p6)*(-ffmcfmb*hbcm3+hbcm3)+8*(p6p7*p4p5)*(ffmcfmb*hbcm3-fmb*
     . hbcm2-hbcm3)+8*(p6p7*p2p5)*(ffmcfmb*hbcm3-hbcm3)+8*(p6p7*p1p5)
     . *(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+16*(p4p5*p3p7*p3p6)*(-
     . ffmcfmb*hbcm-hbcm)+16*(p5p6*p3p7*p3p5)*(2*ffmcfmb*hbcm-hbcm)+8
     . *(p4p6*p3p7*p3p5)*(3*ffmcfmb*hbcm-fmb-2*hbcm)+8*(p5p7*p3p6*
     . p3p5)*(-2*ffmcfmb*hbcm+fmc+hbcm)+8*(p4p7*p3p6*p3p5)*(ffmcfmb*
     . hbcm+fmb-fmc+3*hbcm)+8*(p6p7*p3p5*p3p4)*(-ffmcfmb*hbcm+fmb)+16
     . *(p3p7*p3p6*p2p5)*(-ffmcfmb*hbcm-hbcm)+16*(p5p6*p3p7*p2p3)*(-2
     . *ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p1p5)*(-ffmcfmb*hbcm+2*hbcm)
     . +16*(p5p7*p3p6*p1p3)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p3p5*p1p3)*(
     . ffmcfmb*hbcm-fmb)+8*(p4p6*p3p7)*(3*ffmcfmb*fmb*hbcm2-fmb*hbcm2
     . -fmb2*hbcm)+8*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+2*
     . fmb*hbcm2+2*fmb2*hbcm)+8*(p6p7*p3p4)*(-ffmcfmb*fmb*hbcm2+fmb2*
     . hbcm)+8*(p6p7*p1p3)*(ffmcfmb*fmb*hbcm2-fmb2*hbcm)+8*(p5p6*p3p7
     . )*(-2*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+3*ffmcfmb*hbcm3-fmb
     . *hbcm2-hbcm3)+ans8
      ans6=w2*ans7
      ans11=4*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*fmc*hbcm4+
     . ffmcfmb**2*hbcm5+2*ffmcfmb*fmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm3+
     . 2*ffmcfmb*hbcm*p1p3s2-fmb*fmc2*hbcm2-2*fmb*p1p3s2-fmc2*hbcm3+
     . fmc3*hbcm2)+8*(p3p7*p3p6)*(-ffmcfmb**2*fmb*hbcm2+ffmcfmb**2*
     . fmc*hbcm2+ffmcfmb**2*hbcm3-ffmcfmb*fmb*fmc*hbcm+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm-fmc2*hbcm)+8*(2*
     . ffmcfmb*hbcm*p1p3*p2p3*p6p7-2*ffmcfmb*hbcm*p1p3*p3p5*p6p7-2*
     . ffmcfmb*hbcm*p1p4*p3p6*p3p7-2*ffmcfmb*hbcm*p2p3*p3p6*p4p7+4*
     . ffmcfmb*hbcm*p2p3*p3p6*p5p7-2*ffmcfmb*hbcm*p2p5*p3p6*p3p7-2*
     . ffmcfmb*hbcm*p3p4*p3p6*p5p7+2*ffmcfmb*hbcm*p3p5*p3p6*p4p7-hbcm
     . *p1p3*p1p4*p6p7-hbcm*p1p3*p2p5*p6p7-hbcm*p1p3*p4p6*p4p7+hbcm*
     . p1p3*p5p6*p5p7)
      ans10=8*(p5p7*p3p6*p1p3)*(fmc+hbcm)+8*(p5p7*p5p6)*(-ffmcfmb*
     . hbcm3+fmc*hbcm2)+8*(p4p7*p4p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(
     . p6p7*p2p5)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p6p7*p1p4)*(ffmcfmb*
     . hbcm3-fmc*hbcm2)+8*(p4p6*p3p7*p1p3)*(-ffmcfmb*hbcm-fmb)+8*(
     . p4p7*p3p6*p1p3)*(-ffmcfmb*hbcm+fmb-fmc-hbcm)+8*(p6p7*p3p4*p1p3
     . )*(-ffmcfmb*hbcm+fmb)+8*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*
     . hbcm)+8*(p6p7*p3p4)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*
     . fmc*hbcm-fmc2*hbcm)+8*(p4p6*p3p7)*(ffmcfmb**2*hbcm3+ffmcfmb*
     . fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm)+8*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-ffmcfmb
     . *hbcm3+fmc*hbcm2)+8*(p6p7*p3p5)*(ffmcfmb**2*hbcm3-fmc2*hbcm)+8
     . *(p6p7*p2p3)*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+4*(p6p7*p1p3)*(-
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+3*fmc2*hbcm)+8*(p3p7*p3p6*p1p3)*(
     . ffmcfmb**2*hbcm-ffmcfmb*fmb-2*ffmcfmb*fmc-3*ffmcfmb*hbcm+fmc)+
     . ans11
      ans9=w12*ans10
      ans12=(4*(p4p6*p3p7)*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmb+hbcm)+8*(
     . p5p7*p3p6)*(-4*fb1*ffmcfmb*hbcm+2*fb1*hbcm+2*fb2*fmc-hbcm)+4*(
     . p4p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb-4*fb2*fmc+
     . hbcm)+4*(p6p7*p3p4)*(4*fb1*ffmcfmb*hbcm+4*fb2*fmb-hbcm)+4*(
     . p6p7*p1p3)*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmb+hbcm)+4*p6p7*(2*fb1*
     . ffmcfmb**2*hbcm3+2*fb1*fmc2*hbcm+4*fb2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3)+4*(p3p7*p3p6)*(4*fb1*ffmcfmb**2*hbcm-4*fb1*
     . ffmcfmb*hbcm-4*fb2*ffmcfmb*fmb-8*fb2*ffmcfmb*fmc+4*fb2*fmc+4*
     . ffmcfmb*hbcm-fmb-2*fmc-3*hbcm)+16*(-fb1*hbcm*p1p4*p6p7-fb1*
     . hbcm*p2p5*p6p7-fb1*hbcm*p4p6*p4p7+fb1*hbcm*p5p6*p5p7))
      ans1=ans2+ans6+ans9+ans12
      ans=ccc*ans1
      b(1)=ans
      ans3=w12*(4*(p5p7*p1p3)*(fmb-fmc)+4*(p4p7*p1p3)*(ffmcfmb*hbcm-2
     . *fmb+fmc+hbcm)+4*(p3p7*p1p3)*(-2*ffmcfmb*fmb+2*ffmcfmb*fmc+2*
     . ffmcfmb*hbcm+fmb-fmc)+4*p5p7*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+fmb*fmc*hbcm-fmc2*hbcm)+4*p4p7*(2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc*hbcm2+2*
     . fmc2*hbcm)+4*p3p7*(2*ffmcfmb**2*fmb*hbcm2-2*ffmcfmb**2*fmc*
     . hbcm2-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+2*ffmcfmb*fmc2*hbcm+fmb*fmc*hbcm))+(4*
     . p5p7*(2*fb2*fmb-2*fb2*fmc-3*hbcm)+4*p3p7*(-4*fb2*ffmcfmb*fmb+4
     . *fb2*ffmcfmb*fmc+2*fb2*fmb-2*fb2*fmc-fmb+fmc)+2*p4p7*(-4*fb1*
     . ffmcfmb*hbcm+4*fb1*hbcm-8*fb2*fmb+4*fb2*fmc+hbcm))
      ans2=w2*(4*(p5p7*p3p5)*(2*ffmcfmb*hbcm+fmb-fmc)+4*(p4p7*p3p5)*(
     . ffmcfmb*hbcm-2*fmb+fmc-3*hbcm)+16*(p3p7*p1p5)*(ffmcfmb*hbcm-
     . hbcm)+8*(p5p7*p1p3)*(-ffmcfmb*hbcm+hbcm)+4*(p3p7*p3p5)*(-2*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-2*ffmcfmb*hbcm+fmb-fmc+2*hbcm)+4*
     . p4p7*(2*ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-2*fmb*hbcm2-3*fmb2*hbcm
     . )+4*p5p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+fmb*
     . hbcm2+fmb2*hbcm-hbcm3)+4*p3p7*(2*ffmcfmb**2*fmb*hbcm2+2*
     . ffmcfmb*fmb*fmc*hbcm-3*ffmcfmb*fmb*hbcm2-4*ffmcfmb*fmb2*hbcm-
     . fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)+8*(hbcm*p1p2*p5p7+hbcm*
     . p1p5*p4p7-hbcm*p1p5*p5p7-hbcm*p2p3*p5p7+hbcm*p2p5*p3p7+hbcm*
     . p3p7*p4p5+hbcm*p4p5*p5p7))+ans3
      ans1=w5*(4*(p5p7*p1p3)*(fmb-fmc)+4*(p5p7*p3p4)*(-2*ffmcfmb*hbcm
     . -fmb+fmc+4*hbcm)+4*(p4p7*p3p4)*(-ffmcfmb*hbcm+2*fmb-fmc-hbcm)+
     . 4*(p4p7*p1p3)*(ffmcfmb*hbcm-2*fmb+fmc-hbcm)+4*(p3p7*p3p4)*(2*
     . ffmcfmb*fmb-2*ffmcfmb*fmc-6*ffmcfmb*hbcm-fmb+fmc+2*hbcm)+4*(
     . p3p7*p1p3)*(-2*ffmcfmb*fmb+2*ffmcfmb*fmc+2*ffmcfmb*hbcm+fmb-
     . fmc)+4*p5p7*(-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*
     . hbcm)+4*p4p7*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+4*p3p7*(2*ffmcfmb**2
     . *fmb*hbcm2-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb
     . *hbcm2-2*ffmcfmb*fmc*hbcm2-4*ffmcfmb*fmc2*hbcm-fmb*fmc*hbcm+
     . fmc*hbcm2+fmc2*hbcm)+8*(2*ffmcfmb*hbcm*p1p4*p3p7+hbcm*p1p4*
     . p4p7-hbcm*p1p4*p5p7+hbcm*p2p4*p5p7-hbcm*p2p5*p4p7+hbcm*p4p5*
     . p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans3=(2*p5p6*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb+hbcm)+2*
     . p3p6*(4*fb1*ffmcfmb**2*hbcm-8*fb1*ffmcfmb*hbcm+4*fb1*hbcm+4*
     . fb2*ffmcfmb*fmb-4*fb2*fmb+4*ffmcfmb*hbcm+fmb-4*hbcm)-12*hbcm*
     . p4p6)
      ans2=w2*(4*(p5p6*p3p5)*(ffmcfmb*hbcm-fmb-hbcm)+4*p5p6*(ffmcfmb*
     . fmb*hbcm2-fmb*hbcm2-fmb2*hbcm)+4*(p3p6*p3p5)*(ffmcfmb**2*hbcm+
     . ffmcfmb*fmb+2*ffmcfmb*hbcm-fmb-hbcm)+4*p3p6*(-ffmcfmb**2*fmb*
     . hbcm2+2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm-fmb*hbcm2-fmb2*
     . hbcm)+8*(-ffmcfmb*hbcm*p1p5*p3p6+ffmcfmb*hbcm*p2p5*p3p6+
     . ffmcfmb*hbcm*p3p5*p4p6+ffmcfmb*hbcm*p3p6*p4p5-hbcm*p1p5*p4p6+
     . hbcm*p2p5*p4p6+hbcm*p4p5*p4p6))+w12*(4*(p5p6*p1p3)*(ffmcfmb*
     . hbcm-fmb+hbcm)+4*p5p6*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*(p3p6*p1p3)*
     . (ffmcfmb**2*hbcm+ffmcfmb*fmb-fmb+hbcm)+4*p3p6*(-ffmcfmb**2*fmb
     . *hbcm2+ffmcfmb**2*fmc*hbcm2-ffmcfmb**2*hbcm3+ffmcfmb*fmb*fmc*
     . hbcm+ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm-2*ffmcfmb*fmc*hbcm2-2
     . *ffmcfmb*fmc2*hbcm-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+8*(-
     . ffmcfmb*hbcm*p1p2*p3p6+ffmcfmb*hbcm*p1p4*p3p6-ffmcfmb*hbcm*
     . p1p5*p3p6+ffmcfmb*hbcm*p2p3*p3p6+2*ffmcfmb*hbcm*p2p5*p3p6-
     . ffmcfmb*hbcm*p3p4*p3p6+ffmcfmb*hbcm*p3p5*p3p6))+ans3
      ans1=w5*(4*p4p6*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3)+4*
     . (p5p6*p3p4)*(-ffmcfmb*hbcm+fmb-hbcm)+4*(p5p6*p1p3)*(ffmcfmb*
     . hbcm-fmb+hbcm)+4*p5p6*(ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3-fmc2*
     . hbcm)+4*(p3p6*p3p4)*(-ffmcfmb**2*hbcm-ffmcfmb*fmb+fmb-hbcm)+4*
     . (p3p6*p1p3)*(ffmcfmb**2*hbcm+ffmcfmb*fmb-fmb+hbcm)+4*p3p6*(-
     . ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmb2*hbcm-fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p2*p3p6+
     . ffmcfmb*hbcm*p1p3*p4p6-ffmcfmb*hbcm*p1p5*p3p6+ffmcfmb*hbcm*
     . p2p3*p3p6+ffmcfmb*hbcm*p2p4*p3p6+2*ffmcfmb*hbcm*p2p5*p3p6-
     . ffmcfmb*hbcm*p3p4*p4p6+ffmcfmb*hbcm*p3p5*p3p6+ffmcfmb*hbcm*
     . p3p6*p4p5-hbcm*p1p2*p4p6+hbcm*p1p4*p3p6+hbcm*p1p4*p4p6+hbcm*
     . p1p4*p5p6-hbcm*p1p5*p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*p4p6+2*hbcm*
     . p2p5*p4p6+hbcm*p3p5*p4p6+hbcm*p4p5*p4p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans4=8*(-ffmcfmb**2*hbcm2*p3p6*p3p7-ffmcfmb**2*hbcm2*p3p6*p4p7-
     . ffmcfmb*hbcm2*p1p2*p6p7+ffmcfmb*hbcm2*p2p3*p6p7+ffmcfmb*hbcm2*
     . p2p4*p6p7+ffmcfmb*hbcm2*p5p6*p5p7+2*ffmcfmb*p1p3*p3p7*p4p6-2*
     . ffmcfmb*p3p4*p3p7*p4p6+p1p2*p1p3*p6p7-p1p2*p3p4*p6p7+p1p3*p1p4
     . *p6p7-p1p3*p2p3*p6p7-p1p3*p2p4*p6p7-2*p1p3*p3p4*p6p7+p1p3*p3p6
     . *p4p7+2*p1p3*p4p6*p4p7-p1p3*p4p6*p5p7-p1p3*p5p6*p5p7-p1p4*p3p4
     . *p6p7+p2p3*p3p4*p6p7+p2p4*p3p4*p6p7-p3p4*p3p6*p4p7-2*p3p4*p4p6
     . *p4p7+p3p4*p4p6*p5p7+p3p4*p5p6*p5p7)
      ans3=(p5p7*p3p6*p3p4)*(-16*ffmcfmb+16)+(p5p7*p3p6*p1p3)*(16*
     . ffmcfmb-16)+8*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmb*hbcm)+8*(p4p7*
     . p4p6)*(-2*ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p1p4)*(-2*ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+(p3p7*p3p6*p3p4)*(16*ffmcfmb**2-32*
     . ffmcfmb+8)+(p3p7*p3p6*p1p3)*(-16*ffmcfmb**2+32*ffmcfmb-8)+8*(
     . p4p6*p3p7)*(-2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-fmb*hbcm)+8
     . *(p5p7*p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2)+4*(p6p7*p3p4)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2+fmc2)+4*(
     . p6p7*p1p3)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . hbcm2-fmc2)+4*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmc2*hbcm2+fmb*fmc2*hbcm-fmc2*hbcm2+2*p1p3s2+2*p3p4s2)
     . +ans4
      ans2=w5*ans3
      ans7=8*(2*ffmcfmb*p3p5*p3p7*p4p6+fmb*hbcm*p1p2*p6p7+fmb*hbcm*
     . p1p3*p6p7+fmb*hbcm*p1p4*p6p7-fmb*hbcm*p2p3*p6p7-fmb*hbcm*p2p4*
     . p6p7-fmb*hbcm*p3p4*p6p7+3*fmb*hbcm*p4p6*p4p7-fmb*hbcm*p5p6*
     . p5p7+p1p2*p3p5*p6p7+p1p3*p3p5*p6p7+p1p4*p3p5*p6p7-p2p3*p3p5*
     . p6p7-p2p4*p3p5*p6p7-p3p4*p3p5*p6p7+p3p5*p3p6*p4p7+2*p3p5*p4p6*
     . p4p7-p3p5*p4p6*p5p7-p3p5*p5p6*p5p7)
      ans6=(p5p7*p3p6*p3p5)*(16*ffmcfmb-16)+8*(p5p6*p4p7)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmb*hbcm-
     . hbcm2)+8*(p6p7*p4p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*
     . p2p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p1p5)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*(p4p6*p3p7)*(4*ffmcfmb*fmb*hbcm-fmb*
     . hbcm)+8*(p5p7*p3p6)*(ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm-
     . hbcm2)+8*(p4p7*p3p6)*(ffmcfmb*fmb*hbcm+fmb*hbcm)+8*(p3p7*p3p6)
     . *(3*ffmcfmb*fmb*hbcm-fmb*hbcm)+(p3p7*p3p6*p3p5)*(-16*ffmcfmb**
     . 2+32*ffmcfmb-8)+8*(p5p6*p3p7)*(2*ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm-3*ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p6p7*p3p5)*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-fmc2)+4*p6p7*(ffmcfmb**2*
     . fmb*hbcm3-2*ffmcfmb*fmb*hbcm3-fmb*fmc2*hbcm)+ans7
      ans5=w2*ans6
      ans10=8*(2*ffmcfmb*p1p3*p3p7*p4p6+p1p2*p1p3*p6p7+p1p3*p1p4*p6p7
     . -p1p3*p2p3*p6p7-p1p3*p2p4*p6p7-p1p3*p3p4*p6p7+p1p3*p3p6*p4p7+2
     . *p1p3*p4p6*p4p7-p1p3*p4p6*p5p7-p1p3*p5p6*p5p7)
      ans9=(p5p7*p3p6*p1p3)*(16*ffmcfmb-16)+8*(p5p7*p5p6)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(p5p7*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p4p7*p4p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p3p4)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(p6p7*p2p4)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7
     . *p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p1p4)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+8*(p6p7*p1p2)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7*p3p6
     . )*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+(
     . p3p7*p3p6*p1p3)*(-16*ffmcfmb**2+32*ffmcfmb-8)+16*(p4p6*p3p7)*(
     . -ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+8*(p5p7*p3p6)*(-ffmcfmb**2
     . *hbcm2+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-2*fmc*hbcm)+8*(p3p7*p3p6
     . )*(-ffmcfmb**2*hbcm2+3*ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p6p7*p1p3
     . )*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2+2*fmc*
     . hbcm-fmc2)+4*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*fmc*hbcm3+
     . ffmcfmb**2*hbcm4-2*ffmcfmb*fmc*hbcm3-fmb*fmc2*hbcm+fmc2*hbcm2-
     . fmc3*hbcm+2*p1p3s2)+ans10
      ans8=w12*ans9
      ans11=((p4p6*p3p7)*(32*fb2*ffmcfmb+8)+(p5p7*p3p6)*(32*fb2*
     . ffmcfmb-32*fb2+8)+(p3p7*p3p6)*(-32*fb2*ffmcfmb**2+64*fb2*
     . ffmcfmb-16*fb2-16*ffmcfmb+16)+8*p6p7*(fb2*ffmcfmb**2*hbcm2-2*
     . fb2*ffmcfmb*hbcm2-fb2*fmc2)+16*(fb2*p1p2*p6p7+fb2*p1p3*p6p7+
     . fb2*p1p4*p6p7-fb2*p2p3*p6p7-fb2*p2p4*p6p7-fb2*p3p4*p6p7+fb2*
     . p3p6*p4p7+2*fb2*p4p6*p4p7-fb2*p4p6*p5p7-fb2*p5p6*p5p7))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(4)=ans
      ans4=4*(p3p7*p3p4)*(-3*ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-fmc2+hbcm2)
     . +8*(p3p7*p2p3)*(2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(p3p7*p1p3)*(3*ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm+fmc2-hbcm2)
     . +4*p3p7*(-ffmcfmb**2*fmb*hbcm3-2*ffmcfmb**2*hbcm4-2*ffmcfmb*
     . fmb*fmc*hbcm2-2*ffmcfmb*fmc*hbcm3-4*ffmcfmb*fmc2*hbcm2+ffmcfmb
     . *hbcm4-4*ffmcfmb*p1p3s2+fmb*fmc*hbcm2-fmb*fmc2*hbcm+fmc*hbcm3+
     . fmc2*hbcm2+2*p1p3s2)+8*(ffmcfmb*hbcm2*p2p5*p3p7+hbcm2*p2p4*
     . p5p7-hbcm2*p2p5*p4p7+hbcm2*p4p5*p5p7-p1p3*p1p4*p3p7-2*p1p3*
     . p2p3*p4p7+3*p1p3*p2p3*p5p7-p1p3*p2p5*p3p7+p1p3*p3p4*p4p7-3*
     . p1p3*p3p4*p5p7+2*p1p3*p3p5*p4p7-p1p3*p3p5*p5p7+p1p4*p3p4*p3p7+
     . 2*p2p3*p3p4*p4p7-3*p2p3*p3p4*p5p7+p2p5*p3p4*p3p7-2*p3p4*p3p5*
     . p4p7+p3p4*p3p5*p5p7)
      ans3=4*(p4p7*p3p4)*(-fmb*hbcm-fmc*hbcm-2*hbcm2)+4*(p4p7*p1p3)*(
     . -fmb*hbcm+fmc*hbcm+2*hbcm2)+(p3p7*p3p5*p3p4)*(-16*ffmcfmb+8)+(
     . p3p7*p3p4*p2p3)*(16*ffmcfmb-8)+(p3p7*p3p5*p1p3)*(16*ffmcfmb-8)
     . +(p3p7*p3p4*p1p3)*(16*ffmcfmb-8)+(p3p7*p2p3*p1p3)*(-16*ffmcfmb
     . +8)+8*(p5p7*p3p5)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7*p3p5)*(-2*
     . ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p7*p3p4)*(3*ffmcfmb*hbcm2+2*fmb*
     . hbcm+fmc*hbcm+3*hbcm2)+8*(p5p7*p2p3)*(-3*ffmcfmb*hbcm2-fmc*
     . hbcm)+8*(p4p7*p2p3)*(2*ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p1p4)*(
     . 2*ffmcfmb*hbcm2+fmb*hbcm)+4*(p5p7*p1p3)*(-5*ffmcfmb*hbcm2-fmc*
     . hbcm-hbcm2)+4*p5p7*(2*ffmcfmb**2*hbcm4+ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2+fmc*hbcm3+fmc2*hbcm2+4*p1p3s2+2*p3p4s2)+4*p4p7*(ffmcfmb
     . **2*hbcm4+ffmcfmb*fmb*hbcm3-2*ffmcfmb*hbcm4-fmb*fmc*hbcm2-fmc*
     . hbcm3-fmc2*hbcm2-2*p1p3s2)+8*(p3p7*p3p5)*(-2*ffmcfmb**2*hbcm2-
     . 2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+ans4
      ans2=w5*ans3
      ans7=8*(ffmcfmb*hbcm2*p2p5*p3p7-fmb*hbcm*p1p3*p4p7-fmb*hbcm*
     . p1p4*p3p7-3*fmb*hbcm*p2p3*p4p7-fmb*hbcm*p3p4*p5p7+hbcm2*p1p2*
     . p5p7+hbcm2*p1p5*p4p7-hbcm2*p1p5*p5p7+hbcm2*p4p5*p5p7-p1p3*p3p5
     . *p4p7+2*p1p3*p3p5*p5p7-p1p4*p3p5*p3p7-2*p2p3*p3p5*p4p7+3*p2p3*
     . p3p5*p5p7-p2p5*p3p5*p3p7-p3p4*p3p5*p5p7)
      ans6=(p3p7*p3p5*p2p3)*(-16*ffmcfmb+8)+(p3p7*p3p5*p1p3)*(-16*
     . ffmcfmb+8)+8*(p4p5*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm)+4*(p5p7*p3p5
     . )*(ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm-3*hbcm2)+4*(p4p7*p3p5)*(-
     . 2*ffmcfmb*hbcm2+3*fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(p5p7*p2p3)*(-2
     . *ffmcfmb*hbcm2+2*fmb*hbcm+hbcm2)+8*(p3p7*p1p5)*(ffmcfmb*hbcm2-
     . fmb*hbcm-hbcm2)+8*(p5p7*p1p3)*(-ffmcfmb*hbcm2+2*fmb*hbcm+hbcm2
     . )+4*p4p7*(-ffmcfmb*fmb*hbcm3-fmb*fmc*hbcm2+3*fmb*hbcm3-2*fmb2*
     . hbcm2+4*p3p5s2)+16*(p3p7*p2p3)*(-2*ffmcfmb*fmb*hbcm+fmb*hbcm)+
     . 8*(p3p7*p1p3)*(-2*ffmcfmb*fmb*hbcm+fmb*hbcm)+4*p5p7*(ffmcfmb**
     . 2*hbcm4-2*ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2-2*fmb*hbcm3-hbcm4-2*
     . p3p5s2)+4*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2+6*ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm+fmc2-
     . hbcm2)+4*p3p7*(-ffmcfmb**2*fmb*hbcm3-2*ffmcfmb*fmb*fmc*hbcm2+4
     . *ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmb2*hbcm2+4*ffmcfmb*p3p5s2+fmb*
     . fmc*hbcm2+fmb*fmc2*hbcm-2*fmb*hbcm3+fmb2*hbcm2-2*p3p5s2)+ans7
      ans5=w2*ans6
      ans10=4*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*fmc*hbcm3-2*
     . ffmcfmb**2*hbcm4+2*ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-4*ffmcfmb*
     . p1p3s2+fmb*fmc2*hbcm-fmc*hbcm3+fmc3*hbcm+2*p1p3s2)+8*(-p1p3*
     . p1p4*p3p7-2*p1p3*p2p3*p4p7+3*p1p3*p2p3*p5p7-p1p3*p2p5*p3p7-
     . p1p3*p3p4*p5p7+2*p1p3*p3p5*p4p7-p1p3*p3p5*p5p7)
      ans9=(p3p7*p3p5*p1p3)*(16*ffmcfmb-8)+(p3p7*p2p3*p1p3)*(-16*
     . ffmcfmb+8)+8*(p5p7*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*
     . p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p7*p3p4)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+8*(p3p7*p2p5)*(ffmcfmb*hbcm2-fmc*hbcm)+24*(p5p7*p2p3
     . )*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p7*p2p3)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+8*(p3p7*p1p4)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p7*p1p3)*(-5
     . *ffmcfmb*hbcm2+3*fmc*hbcm-hbcm2)+4*(p4p7*p1p3)*(2*ffmcfmb*
     . hbcm2+fmb*hbcm-fmc*hbcm+2*hbcm2)+4*p4p7*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+
     . fmc2*hbcm2-2*p1p3s2)+4*p5p7*(2*ffmcfmb**2*hbcm4-2*ffmcfmb*fmc*
     . hbcm3+ffmcfmb*hbcm4-fmc*hbcm3+4*p1p3s2)+8*(p3p7*p3p5)*(-2*
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+8*
     . (p3p7*p2p3)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmc*hbcm-ffmcfmb*
     . hbcm2+fmc*hbcm)+4*(p3p7*p1p3)*(3*ffmcfmb**2*hbcm2+2*ffmcfmb*
     . fmb*hbcm-2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm+fmc2-hbcm2)
     . +ans10
      ans8=w12*ans9
      ans11=((p3p7*p3p5)*(32*fb2*ffmcfmb-16*fb2+8)+(p3p7*p2p3)*(-32*
     . fb2*ffmcfmb+16*fb2-8)+(p3p7*p1p3)*(-32*fb2*ffmcfmb+16*fb2-8)+2
     . *p5p7*(-4*fb1*fmc*hbcm-12*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2)
     . +8*p4p7*(fb1*fmb*hbcm+fb1*fmc*hbcm+2*fb2*hbcm2)+2*p3p7*(8*fb1*
     . ffmcfmb*fmc*hbcm-4*fb1*fmc*hbcm+4*fb2*ffmcfmb**2*hbcm2+4*fb2*
     . ffmcfmb*hbcm2+4*fb2*fmc2-4*fb2*hbcm2-2*fmc*hbcm+3*hbcm2)+16*(-
     . fb2*p1p3*p4p7+2*fb2*p1p3*p5p7-fb2*p1p4*p3p7-2*fb2*p2p3*p4p7+3*
     . fb2*p2p3*p5p7-fb2*p2p5*p3p7-fb2*p3p4*p5p7+2*fb2*p3p5*p4p7-fb2*
     . p3p5*p5p7))
      ans1=ans2+ans5+ans8+ans11
      ans=ccc*ans1
      b(5)=ans
      ans4=8*(ffmcfmb**2*hbcm2*p3p5*p3p6-ffmcfmb*hbcm2*p1p2*p3p6-
     . ffmcfmb*hbcm2*p1p5*p3p6+ffmcfmb*hbcm2*p2p3*p5p6+ffmcfmb*hbcm2*
     . p2p4*p3p6+2*ffmcfmb*hbcm2*p2p5*p3p6-ffmcfmb*hbcm2*p3p5*p5p6+
     . ffmcfmb*hbcm2*p3p6*p4p5-2*ffmcfmb*p1p3*p3p4*p3p6-hbcm2*p1p2*
     . p4p6+hbcm2*p1p4*p3p6+hbcm2*p1p4*p4p6+hbcm2*p1p4*p5p6-hbcm2*
     . p1p5*p4p6+hbcm2*p2p4*p4p6+2*hbcm2*p2p5*p4p6+hbcm2*p3p5*p4p6+
     . hbcm2*p4p5*p4p6+2*p1p3*p2p3*p4p6-p1p3*p2p3*p5p6-2*p1p3*p3p4*
     . p4p6+p1p3*p3p5*p5p6-2*p2p3*p3p4*p4p6+p2p3*p3p4*p5p6-p3p4*p3p5*
     . p5p6)
      ans3=8*(p4p6*p3p4)*(-fmb*hbcm+hbcm2)+8*(p4p6*p2p3)*(2*fmb*hbcm-
     . hbcm2)+8*(p4p6*p1p3)*(fmb*hbcm-hbcm2)+(p3p6*p3p5*p3p4)*(8*
     . ffmcfmb-8)+(p3p6*p3p4*p2p3)*(-24*ffmcfmb+8)+(p3p6*p3p5*p1p3)*(
     . -8*ffmcfmb+8)+(p3p6*p2p3*p1p3)*(24*ffmcfmb-8)+4*(p5p6*p3p4)*(-
     . ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+4*(p5p6*p1p3)*(ffmcfmb*hbcm2+fmb
     . *hbcm+hbcm2)+4*p5p6*(-ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4-fmc2*
     . hbcm2)+4*p4p6*(-2*ffmcfmb*fmb*hbcm3+fmb2*hbcm2-fmc2*hbcm2+
     . hbcm4+2*p1p3s2+2*p3p4s2)+4*(p3p6*p3p4)*(ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p3p6*p2p3)
     . *(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm)+4*(p3p6*p1p3)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2+fmb*hbcm+
     . hbcm2)+4*p3p6*(-ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*hbcm4-ffmcfmb*
     . fmb*hbcm3+ffmcfmb*fmb2*hbcm2+2*ffmcfmb*p1p3s2+2*ffmcfmb*p3p4s2
     . -fmc2*hbcm2)+ans4
      ans2=w5*ans3
      ans8=8*(-ffmcfmb*hbcm2*p1p2*p3p6+ffmcfmb*hbcm2*p1p4*p3p6-
     . ffmcfmb*hbcm2*p1p5*p3p6+2*ffmcfmb*hbcm2*p2p5*p3p6-ffmcfmb*p1p3
     . *p3p4*p3p6+2*p1p3*p2p3*p4p6-p1p3*p2p3*p5p6-p1p3*p3p4*p4p6+p1p3
     . *p3p5*p5p6)
      ans7=(p3p6*p3p5*p1p3)*(-8*ffmcfmb+8)+(p3p6*p2p3*p1p3)*(24*
     . ffmcfmb-8)+8*(p5p6*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p6*
     . p3p4)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p6*p2p3)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+16*(p4p6*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p6*
     . p1p3)*(ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p4p6*p1p3)*(-2*ffmcfmb
     . *hbcm2+fmc*hbcm)+4*p5p6*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2+fmc*hbcm3+fmc2*hbcm2)+8*(p3p6*p3p4
     . )*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm)+8*p4p6*(ffmcfmb**2*
     . hbcm4-ffmcfmb*fmc*hbcm3+p1p3s2)+8*(p3p6*p3p5)*(ffmcfmb**2*
     . hbcm2-ffmcfmb*fmc*hbcm+fmc*hbcm)+8*(p3p6*p2p3)*(-ffmcfmb**2*
     . hbcm2+2*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p3p6*
     . p1p3)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2
     . *ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p3p6*(-ffmcfmb**2*fmb*hbcm3-
     . ffmcfmb**2*fmc*hbcm3+ffmcfmb**2*hbcm4-ffmcfmb*fmb*fmc*hbcm2-
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-2*ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*fmc2*hbcm2+2*ffmcfmb*p1p3s2+fmb*fmc*hbcm2+fmc*hbcm3+
     . fmc2*hbcm2)+ans8
      ans6=w12*ans7
      ans9=((p3p6*p3p5)*(-16*fb2*ffmcfmb+16*fb2-4)+(p3p6*p3p4)*(-16*
     . fb2*ffmcfmb-4)+(p3p6*p2p3)*(48*fb2*ffmcfmb-16*fb2+12)+(p3p6*
     . p1p3)*(16*fb2*ffmcfmb+4)+4*p4p6*(-4*fb2*ffmcfmb*hbcm2-hbcm2)+2
     . *p5p6*(4*fb1*fmb*hbcm-4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+2
     . *p3p6*(-4*fb1*ffmcfmb*fmb*hbcm+4*fb1*fmb*hbcm-4*fb2*ffmcfmb**2
     . *hbcm2-8*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+2*ffmcfmb*hbcm2-5*fmb*
     . hbcm)+16*(fb2*p1p3*p4p6+2*fb2*p2p3*p4p6-fb2*p2p3*p5p6-fb2*p3p4
     . *p4p6+fb2*p3p5*p5p6))
      ans5=w2*((p3p6*p3p5*p2p3)*(24*ffmcfmb-8)+4*(p5p6*p3p5)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p5p6*(-ffmcfmb*fmb*hbcm3+fmb*
     . hbcm3-fmb2*hbcm2+2*p3p5s2)+8*(p3p6*p2p3)*(3*ffmcfmb*fmb*hbcm-
     . fmb*hbcm)+4*(p3p6*p3p5)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p3p6*(-ffmcfmb**2*fmb*hbcm3-2*
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmb2*hbcm2-2*ffmcfmb*p3p5s2+fmb*
     . hbcm3-fmb2*hbcm2+2*p3p5s2)+8*(ffmcfmb*fmb*hbcm*p1p3*p3p6-
     . ffmcfmb*fmb*hbcm*p3p4*p3p6-ffmcfmb*fmb*hbcm3*p4p6-ffmcfmb*
     . hbcm2*p1p5*p3p6+ffmcfmb*hbcm2*p2p5*p3p6+ffmcfmb*hbcm2*p3p6*
     . p4p5+ffmcfmb*p1p3*p3p5*p3p6-ffmcfmb*p3p4*p3p5*p3p6+fmb*hbcm*
     . p1p3*p4p6+2*fmb*hbcm*p2p3*p4p6-fmb*hbcm*p2p3*p5p6-fmb*hbcm*
     . p3p4*p4p6-hbcm2*p1p5*p4p6+hbcm2*p2p5*p4p6+hbcm2*p4p5*p4p6+p1p3
     . *p3p5*p4p6+2*p2p3*p3p5*p4p6-p2p3*p3p5*p5p6-p3p4*p3p5*p4p6))+
     . ans6+ans9
      ans1=ans2+ans5
      ans=ccc*ans1
      b(6)=ans
      ans3=w12*(4*p3p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p3p4*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*p2p5*(-ffmcfmb*hbcm2+fmc*hbcm)+4*p2p3*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*p1p5*(ffmcfmb*hbcm2-fmc*hbcm)+4*p1p4
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+4*p1p2*(ffmcfmb*hbcm2-fmc*hbcm)+2*
     . p1p3*(-2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+
     . fmb2-fmc2+hbcm2)+2*(2*ffmcfmb**2*hbcm4-ffmcfmb*fmb2*hbcm2-2*
     . ffmcfmb*fmc*hbcm3+ffmcfmb*fmc2*hbcm2-ffmcfmb*hbcm4+2*ffmcfmb*
     . p1p3s2+fmb2*fmc*hbcm+fmc*hbcm3-fmc3*hbcm-2*p1p2*p1p3+2*p1p3*
     . p1p4-2*p1p3*p1p5+2*p1p3*p2p3+4*p1p3*p2p5-2*p1p3*p3p4+2*p1p3*
     . p3p5))+(p1p3*(8*fb2*ffmcfmb+2)+2*(-4*fb2*ffmcfmb*hbcm2+2*fb2*
     . fmb2-2*fb2*fmc2+2*fb2*hbcm2-4*fb2*p1p2+4*fb2*p1p4-4*fb2*p1p5+4
     . *fb2*p2p3+8*fb2*p2p5-4*fb2*p3p4+4*fb2*p3p5+3*fmb*hbcm+3*fmc*
     . hbcm-hbcm2))
      ans2=w5*((p3p4*p1p3)*(-4*ffmcfmb-4)+4*p4p5*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+4*p2p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+2*p3p4*(2*
     . ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-2*fmb*hbcm-fmb2+fmc2+hbcm2
     . )+2*p1p3*(-2*ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+fmb2-fmc2+hbcm2)
     . +2*(2*ffmcfmb**2*hbcm4-ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc2*hbcm2+2
     . *ffmcfmb*hbcm2*p1p2-2*ffmcfmb*hbcm2*p1p4+2*ffmcfmb*hbcm2*p1p5-
     . 2*ffmcfmb*hbcm2*p2p3-4*ffmcfmb*hbcm2*p2p5-2*ffmcfmb*hbcm2*p3p5
     . -ffmcfmb*hbcm4+2*ffmcfmb*p1p3s2-2*p1p2*p1p3+2*p1p2*p3p4+2*p1p3
     . *p1p4-2*p1p3*p1p5+2*p1p3*p2p3+4*p1p3*p2p5+2*p1p3*p3p5-2*p1p4*
     . p3p4+2*p1p5*p3p4-2*p2p3*p3p4-4*p2p5*p3p4-2*p3p4*p3p5+2*p3p4s2)
     . )+ans3
      ans1=w2*(4*p4p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p5*(-
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p1p5*(ffmcfmb*hbcm2-hbcm2)+2*
     . p3p5*(-2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+2*fmb*hbcm+fmb2-
     . fmc2+hbcm2)+2*(2*ffmcfmb*fmb*hbcm*p1p3-2*ffmcfmb*fmb*hbcm3+2*
     . ffmcfmb*p1p3*p3p5-fmb*fmc2*hbcm-2*fmb*hbcm*p1p2+2*fmb*hbcm*
     . p1p4+2*fmb*hbcm*p2p3-2*fmb*hbcm*p3p4+fmb*hbcm3+fmb3*hbcm-2*
     . p1p2*p3p5+2*p1p4*p3p5-2*p1p5*p3p5+2*p2p3*p3p5+4*p2p5*p3p5-2*
     . p3p4*p3p5+2*p3p5s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=(4*p3p5*(2*fb1*hbcm-3*hbcm)+2*p3p4*(-4*fb1*ffmcfmb*hbcm-4*
     . fb2*fmb+hbcm)+8*p2p3*(2*fb1*ffmcfmb*hbcm-fb1*hbcm+2*fb2*fmb+
     . hbcm)+8*p1p3*(2*fb1*ffmcfmb*hbcm-fb1*hbcm+fb2*fmb+hbcm)+2*(-4*
     . fb1*ffmcfmb**2*hbcm3+2*fb1*fmb2*hbcm-2*fb1*fmc2*hbcm-4*fb1*
     . hbcm*p1p2+4*fb1*hbcm*p1p4-4*fb1*hbcm*p1p5+8*fb1*hbcm*p2p5+2*
     . fb1*hbcm3-4*fb2*ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3-fmb*hbcm2+3*
     . fmc*hbcm2-3*hbcm3))
      ans3=w12*(4*p3p5*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*p2p5*(-ffmcfmb*
     . hbcm3+fmc*hbcm2)+4*p1p5*(ffmcfmb*hbcm3-fmc*hbcm2)+4*p1p4*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+4*p1p2*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(
     . p3p4*p1p3)*(ffmcfmb*hbcm-fmb)+4*(p2p3*p1p3)*(-2*ffmcfmb*hbcm+2
     . *fmb-hbcm)+4*p3p4*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc
     . *hbcm+fmc2*hbcm)+4*p2p3*(-2*ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+2*
     . p1p3*(-4*ffmcfmb*fmb*hbcm2+4*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3
     . +2*fmb*fmc*hbcm+fmb2*hbcm-2*fmc*hbcm2-3*fmc2*hbcm+hbcm3)+2*(2*
     . ffmcfmb**2*fmb*hbcm4-2*ffmcfmb**2*fmc*hbcm4-2*ffmcfmb*fmb*fmc*
     . hbcm3-ffmcfmb*fmb2*hbcm3+3*ffmcfmb*fmc2*hbcm3-ffmcfmb*hbcm5+2*
     . fmb*p1p3s2+fmb2*fmc*hbcm2+fmc*hbcm4-fmc3*hbcm2-2*hbcm*p1p2*
     . p1p3+2*hbcm*p1p3*p1p4-2*hbcm*p1p3*p1p5+4*hbcm*p1p3*p2p5+2*hbcm
     . *p1p3*p3p5-2*hbcm*p1p3s2))+ans4
      ans2=w2*(4*p4p5*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+2*p3p5*(2*
     . ffmcfmb*hbcm3-2*fmb*hbcm2-fmb2*hbcm+fmc2*hbcm-hbcm3)+4*p2p5*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*p1p5*(ffmcfmb*hbcm3-hbcm3)+4*
     . (p3p5*p3p4)*(ffmcfmb*hbcm-fmb)+4*(p3p5*p2p3)*(-4*ffmcfmb*hbcm+
     . 2*fmb+hbcm)+4*(p3p5*p1p3)*(-2*ffmcfmb*hbcm+fmb+hbcm)+4*p3p4*(
     . ffmcfmb*fmb*hbcm2-fmb2*hbcm)+4*p2p3*(-2*ffmcfmb*fmb*hbcm2+fmb*
     . hbcm2+2*fmb2*hbcm)+4*p1p3*(-2*ffmcfmb*fmb*hbcm2+fmb*hbcm2+fmb2
     . *hbcm)+2*(2*ffmcfmb**2*fmb*hbcm4-2*ffmcfmb*fmb2*hbcm3+4*
     . ffmcfmb*hbcm*p3p5s2+fmb*fmc2*hbcm2+2*fmb*hbcm2*p1p2-2*fmb*
     . hbcm2*p1p4-fmb*hbcm4-fmb3*hbcm2+2*hbcm*p1p2*p3p5-2*hbcm*p1p4*
     . p3p5+4*hbcm*p1p5*p2p3-2*hbcm*p1p5*p3p5-4*hbcm*p2p3*p2p5-4*hbcm
     . *p2p3*p4p5+4*hbcm*p3p5*p4p5-2*hbcm*p3p5s2))+ans3
      ans1=w5*(4*p4p5*(-ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+4*p2p4*(-
     . ffmcfmb*hbcm3+fmb*hbcm2+hbcm3)+4*(p3p5*p3p4)*(-2*ffmcfmb*hbcm+
     . hbcm)+4*(p3p4*p2p3)*(4*ffmcfmb*hbcm-2*fmb-hbcm)+4*(p3p4*p1p3)*
     . (ffmcfmb*hbcm-2*fmb+hbcm)+4*(p2p3*p1p3)*(-2*ffmcfmb*hbcm+2*fmb
     . -hbcm)+2*p3p4*(2*ffmcfmb*fmb*hbcm2-4*ffmcfmb*hbcm3+2*fmb*hbcm2
     . -fmb2*hbcm-fmc2*hbcm+hbcm3)+4*p2p3*(-2*ffmcfmb*fmb*hbcm2+
     . ffmcfmb*hbcm3+2*fmc2*hbcm)+2*p1p3*(-4*ffmcfmb*fmb*hbcm2+2*
     . ffmcfmb*hbcm3+fmb2*hbcm+fmc2*hbcm+hbcm3)+2*(2*ffmcfmb**2*fmb*
     . hbcm4-ffmcfmb*fmb2*hbcm3-ffmcfmb*fmc2*hbcm3-2*ffmcfmb*hbcm*
     . p3p4s2+2*ffmcfmb*hbcm3*p1p2+2*ffmcfmb*hbcm3*p1p4+2*ffmcfmb*
     . hbcm3*p1p5-4*ffmcfmb*hbcm3*p2p5-2*ffmcfmb*hbcm3*p3p5-ffmcfmb*
     . hbcm5+2*fmb*p1p3s2+2*fmb*p3p4s2-2*hbcm*p1p2*p1p3+2*hbcm*p1p2*
     . p3p4-2*hbcm*p1p3*p1p4-2*hbcm*p1p3*p1p5+4*hbcm*p1p3*p2p5+2*hbcm
     . *p1p3*p3p5-2*hbcm*p1p3s2-8*hbcm*p1p4*p2p3+2*hbcm*p1p4*p3p4+2*
     . hbcm*p1p5*p3p4-4*hbcm*p2p3*p2p4-4*hbcm*p2p3*p4p5+4*hbcm*p2p4*
     . p3p5-4*hbcm*p2p5*p3p4+4*hbcm*p3p5*p4p5))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w2*(4*(p3p6*p3p5)*(-4*ffmcfmb*hbcm+hbcm)+4*p3p6*(-4*
     . ffmcfmb*fmb*hbcm2+fmb*hbcm2)+4*(-3*fmb*hbcm2*p4p6+fmb*hbcm2*
     . p5p6-3*hbcm*p3p5*p4p6+hbcm*p3p5*p5p6))+w5*(4*p4p6*(2*ffmcfmb*
     . hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(p3p6*p3p4)*(-2*ffmcfmb*hbcm+hbcm
     . )+4*(p3p6*p1p3)*(2*ffmcfmb*hbcm-hbcm)+4*p3p6*(ffmcfmb**2*hbcm3
     . -3*ffmcfmb*fmb*hbcm2-2*ffmcfmb*hbcm3)+4*(2*ffmcfmb*hbcm*p2p3*
     . p3p6-6*ffmcfmb*hbcm*p3p5*p3p6+ffmcfmb*hbcm3*p5p6+hbcm*p1p3*
     . p4p6-hbcm*p1p3*p5p6+2*hbcm*p2p3*p4p6-hbcm*p3p4*p4p6+hbcm*p3p4*
     . p5p6-6*hbcm*p3p5*p4p6))+w12*(4*p5p6*(ffmcfmb*hbcm3-fmc*hbcm2)+
     . 12*p4p6*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p3p6*p1p3)*(2*ffmcfmb*
     . hbcm-hbcm)+4*p3p6*(ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+4*
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmc*hbcm2)+4*(2*ffmcfmb*hbcm
     . *p2p3*p3p6+2*ffmcfmb*hbcm*p3p4*p3p6-6*ffmcfmb*hbcm*p3p5*p3p6+3
     . *hbcm*p1p3*p4p6-hbcm*p1p3*p5p6))+(8*p3p6*(4*fb1*ffmcfmb*hbcm-
     . fb1*hbcm+2*hbcm)+8*(3*fb1*hbcm*p4p6-fb1*hbcm*p5p6)))
      b(9)=ans
      b(10)=ccc*(w2*((p3p6*p3p5)*(-16*ffmcfmb+4)+4*p3p6*(-4*ffmcfmb*
     . fmb*hbcm+fmb*hbcm)+4*(-3*fmb*hbcm*p4p6+fmb*hbcm*p5p6-3*p3p5*
     . p4p6+p3p5*p5p6))+w5*(12*p4p6*(-fmb*hbcm+hbcm2)+(p3p6*p3p4)*(16
     . *ffmcfmb-4)+(p3p6*p1p3)*(-16*ffmcfmb+4)+4*p3p6*(ffmcfmb**2*
     . hbcm2-3*ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2)+4*(-ffmcfmb*hbcm2*
     . p5p6-3*p1p3*p4p6+p1p3*p5p6+3*p3p4*p4p6-p3p4*p5p6))+w12*((p3p6*
     . p1p3)*(-16*ffmcfmb+4)+4*p5p6*(-ffmcfmb*hbcm2+fmc*hbcm)+12*p4p6
     . *(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p6*(ffmcfmb**2*hbcm2-3*ffmcfmb*
     . fmb*hbcm-4*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+fmc*hbcm)+4*(-3*
     . p1p3*p4p6+p1p3*p5p6))+(p3p6*(-32*fb2*ffmcfmb+8*fb2-8)+8*(-3*
     . fb2*p4p6+fb2*p5p6)))
      ans2=(-12*fb1*fmb*hbcm+20*fb2*ffmcfmb*hbcm2-12*fb2*hbcm2-8*fb2*
     . p1p3+8*fb2*p2p3+8*fb2*p3p4-24*fb2*p3p5-hbcm2)
      ans1=w2*(6*p3p5*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+2*(5*ffmcfmb*fmb
     . *hbcm3-2*fmb*hbcm*p1p3+2*fmb*hbcm*p2p3+2*fmb*hbcm*p3p4-3*fmb*
     . hbcm3+3*fmb2*hbcm2+2*hbcm2*p1p5-2*hbcm2*p2p5-2*hbcm2*p4p5-2*
     . p1p3*p3p5+2*p2p3*p3p5+2*p3p4*p3p5-6*p3p5s2))+w5*(2*p3p4*(
     . ffmcfmb*hbcm2+3*fmb*hbcm+hbcm2)+2*p1p3*(ffmcfmb*hbcm2-3*fmb*
     . hbcm-3*hbcm2)+2*(-2*ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*hbcm3-2*
     . ffmcfmb*hbcm2*p2p3+6*ffmcfmb*hbcm2*p3p5+3*ffmcfmb*hbcm4+3*fmc2
     . *hbcm2-6*hbcm2*p1p4-2*hbcm2*p2p4-2*hbcm2*p4p5+2*p1p3*p2p3+4*
     . p1p3*p3p4-6*p1p3*p3p5-2*p1p3s2-2*p2p3*p3p4+6*p3p4*p3p5-2*
     . p3p4s2))+w12*(12*p3p5*(ffmcfmb*hbcm2-fmc*hbcm)+4*p3p4*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*p2p3*(-ffmcfmb*hbcm2+fmc*hbcm)+2*
     . p1p3*(ffmcfmb*hbcm2-3*fmb*hbcm-2*fmc*hbcm-3*hbcm2)+2*(-2*
     . ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*hbcm3+5*ffmcfmb*fmc*hbcm3+3*
     . ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*fmc*hbcm3-3*fmc2*hbcm2+2*p1p3*
     . p2p3+2*p1p3*p3p4-6*p1p3*p3p5-2*p1p3s2))+ans2
      ans=ccc*ans1
      b(11)=ans
      ans=ccc*(w2*(8*p5p7*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*(p3p7*
     . p3p5)*(-ffmcfmb*hbcm-fmb+2*hbcm)+4*p3p7*(-ffmcfmb*fmb*hbcm2+
     . fmb*hbcm2-fmb2*hbcm)+4*(-2*fmb*hbcm2*p4p7-2*hbcm*p1p5*p3p7+2*
     . hbcm*p2p5*p3p7-3*hbcm*p3p5*p4p7+hbcm*p3p5*p5p7+2*hbcm*p3p7*
     . p4p5))+w5*(4*p5p7*(3*ffmcfmb*hbcm3+fmc*hbcm2)+4*p4p7*(2*
     . ffmcfmb*hbcm3-fmb*hbcm2-fmc*hbcm2-hbcm3)+4*(p3p7*p3p4)*(-7*
     . ffmcfmb*hbcm+fmb+2*hbcm)+4*(p3p7*p1p3)*(3*ffmcfmb*hbcm-fmb)+4*
     . p3p7*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2+
     . fmc*hbcm2-2*fmc2*hbcm)+4*(-hbcm*p1p3*p4p7-3*hbcm*p1p3*p5p7+4*
     . hbcm*p1p4*p3p7+2*hbcm*p2p3*p4p7-hbcm*p3p4*p4p7+5*hbcm*p3p4*
     . p5p7-2*hbcm*p3p5*p4p7))+w12*(12*p5p7*(ffmcfmb*hbcm3-fmc*hbcm2)
     . +4*p4p7*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p3p7*p1p3)*(3*ffmcfmb*
     . hbcm-fmb)+4*p3p7*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+4*(hbcm*p1p3*p4p7-3*hbcm*
     . p1p3*p5p7))+(2*p3p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmb+hbcm)+8*(
     . fb1*hbcm*p4p7-3*fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w5*(2*p3p4*(-5*ffmcfmb*hbcm+3*fmb-hbcm)+6*p1p3*(
     . ffmcfmb*hbcm-fmb+hbcm)+2*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*hbcm3-
     . 3*fmc2*hbcm+6*hbcm*p1p4+2*hbcm*p2p4+2*hbcm*p4p5))+w2*(2*p3p5*(
     . 5*ffmcfmb*hbcm-3*fmb-3*hbcm)+2*(3*ffmcfmb*fmb*hbcm2-3*fmb*
     . hbcm2-3*fmb2*hbcm-2*hbcm*p1p5+2*hbcm*p2p5+2*hbcm*p4p5))+w12*(6
     . *p1p3*(ffmcfmb*hbcm-fmb+hbcm)+6*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc
     . *hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm))+3*(-4*
     . fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm))
      ans=ccc*(w2*((p3p7*p3p5)*(8*ffmcfmb-8)+4*p5p7*(2*ffmcfmb*hbcm2-
     . 3*fmb*hbcm-2*hbcm2)+4*p3p7*(4*ffmcfmb*fmb*hbcm-3*fmb*hbcm)+4*(
     . 3*fmb*hbcm*p4p7+2*p3p5*p4p7-4*p3p5*p5p7))+w5*((p3p7*p3p4)*(-8*
     . ffmcfmb+8)+(p3p7*p1p3)*(8*ffmcfmb-8)+4*p5p7*(4*ffmcfmb*hbcm2+
     . fmc*hbcm)+4*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm-hbcm2)+4*
     . p3p7*(-2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2+
     . fmc*hbcm)+8*(p1p3*p4p7-2*p1p3*p5p7-p3p4*p4p7+2*p3p4*p5p7))+w12
     . *((p3p7*p1p3)*(8*ffmcfmb-8)+16*p5p7*(ffmcfmb*hbcm2-fmc*hbcm)+8
     . *p4p7*(-ffmcfmb*hbcm2+fmc*hbcm)+8*p3p7*(-ffmcfmb**2*hbcm2+
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+8*(p1p3*p4p7-2*p1p3*
     . p5p7))+(p3p7*(16*fb2*ffmcfmb-16*fb2+4)+16*(fb2*p4p7-2*fb2*p5p7
     . )))
      b(14)=ans
      ans3=w12*((p3p7*p3p6*p1p3)*(-32*ffmcfmb+8)+16*(p4p6*p3p7)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p7*p3p6)*(-3*ffmcfmb*hbcm2+fmc*
     . hbcm)+8*(p4p7*p3p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p3p4)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7*p1p3)*(3*ffmcfmb*hbcm2-fmb*
     . hbcm-4*fmc*hbcm+hbcm2)+4*p6p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*
     . hbcm3+3*ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*
     . hbcm3-2*fmc2*hbcm2-4*p1p3s2)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm+fmc*hbcm)+8*(2*p1p3*p3p4*
     . p6p7-p1p3*p3p6*p4p7+p1p3*p3p6*p5p7-2*p1p3*p3p7*p4p6))+((p3p7*
     . p3p6)*(-64*fb2*ffmcfmb+16*fb2-16)+2*p6p7*(-4*fb1*fmb*hbcm+12*
     . fb2*ffmcfmb*hbcm2+4*fb2*hbcm2-hbcm2)+16*(-2*fb2*p1p3*p6p7+2*
     . fb2*p3p4*p6p7-fb2*p3p6*p4p7+fb2*p3p6*p5p7-2*fb2*p3p7*p4p6))
      ans2=w5*((p3p7*p3p6*p3p4)*(32*ffmcfmb-8)+(p3p7*p3p6*p1p3)*(-32*
     . ffmcfmb+8)+8*(p4p6*p3p7)*(ffmcfmb*hbcm2-fmb*hbcm)+4*(p6p7*p3p4
     . )*(-3*ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p6p7*p1p3)*(3*ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+4*p6p7*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*
     . hbcm3-ffmcfmb*hbcm4+2*fmc2*hbcm2-4*p1p3s2-4*p3p4s2)+8*(p3p7*
     . p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+8*(-ffmcfmb*hbcm2*
     . p3p6*p4p7-3*ffmcfmb*hbcm2*p3p6*p5p7-2*hbcm2*p1p4*p6p7-hbcm2*
     . p4p6*p4p7-3*hbcm2*p4p6*p5p7+4*p1p3*p3p4*p6p7-p1p3*p3p6*p4p7+
     . p1p3*p3p6*p5p7-2*p1p3*p3p7*p4p6+p3p4*p3p6*p4p7-p3p4*p3p6*p5p7+
     . 2*p3p4*p3p7*p4p6))+ans3
      ans1=w2*((p3p7*p3p6*p3p5)*(-32*ffmcfmb+8)+8*(p5p6*p3p7)*(2*
     . ffmcfmb*hbcm2-hbcm2)+8*(p5p7*p3p6)*(-2*ffmcfmb*hbcm2+fmb*hbcm+
     . hbcm2)+4*(p6p7*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+4*p6p7*(3*
     . ffmcfmb*fmb*hbcm3+fmb*hbcm3+fmb2*hbcm2)+8*(p3p7*p3p6)*(-4*
     . ffmcfmb*fmb*hbcm+fmb*hbcm)+8*(-2*fmb*hbcm*p1p3*p6p7+2*fmb*hbcm
     . *p3p4*p6p7-fmb*hbcm*p3p6*p4p7-2*fmb*hbcm*p3p7*p4p6+hbcm2*p1p5*
     . p6p7-hbcm2*p2p5*p6p7-hbcm2*p4p5*p6p7-hbcm2*p4p6*p5p7+hbcm2*
     . p4p7*p5p6-2*p1p3*p3p5*p6p7+2*p3p4*p3p5*p6p7-p3p5*p3p6*p4p7+
     . p3p5*p3p6*p5p7-2*p3p5*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w2*(8*(p5p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+8*(p5p7*p3p6)*
     . (-2*ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p5)*(-3*ffmcfmb*hbcm+fmb+hbcm
     . )+4*p6p7*(-ffmcfmb*fmb*hbcm2+fmb*hbcm2+fmb2*hbcm)+8*(hbcm*p1p5
     . *p6p7-hbcm*p2p5*p6p7-hbcm*p4p5*p6p7-hbcm*p4p6*p5p7+hbcm*p4p7*
     . p5p6))+w5*(16*(p4p6*p3p7)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4)*(3
     . *ffmcfmb*hbcm-fmb+hbcm)+4*(p6p7*p1p3)*(-3*ffmcfmb*hbcm+fmb-
     . hbcm)+4*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3
     . +2*fmc2*hbcm)+8*(-ffmcfmb*hbcm*p3p6*p3p7-ffmcfmb*hbcm*p3p6*
     . p4p7-3*ffmcfmb*hbcm*p3p6*p5p7-2*hbcm*p1p4*p6p7-4*hbcm*p4p6*
     . p5p7))+w12*(4*(p6p7*p1p3)*(-3*ffmcfmb*hbcm+fmb-hbcm)+4*p6p7*(
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+8*(-ffmcfmb*hbcm*
     . p3p6*p3p7+ffmcfmb*hbcm*p3p6*p4p7-3*ffmcfmb*hbcm*p3p6*p5p7))+2*
     . p6p7*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb-hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp31_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5))
      ans5=8*(2*hbcm*p1p2*p2p3*p6p7-2*hbcm*p1p2*p3p6*p3p7-2*hbcm*p1p2
     . *p3p7*p4p6-2*hbcm*p1p2*p3p7*p5p6+2*hbcm*p1p3*p1p4*p6p7-hbcm*
     . p1p3*p1p5*p6p7+hbcm*p1p3*p3p6*p4p7+hbcm*p1p3*p3p6*p5p7-2*hbcm*
     . p1p3*p4p5*p6p7+hbcm*p1p3*p4p6*p4p7+hbcm*p1p3*p4p6*p5p7+hbcm*
     . p1p3*p4p7*p5p6+hbcm*p1p3*p5p6*p5p7+2*hbcm*p1p4*p2p3*p6p7-2*
     . hbcm*p1p4*p3p5*p6p7-2*hbcm*p1p4*p3p6*p3p7-2*hbcm*p1p4*p3p7*
     . p4p6-2*hbcm*p1p4*p3p7*p5p6-4*hbcm*p1p5*p2p3*p6p7+2*hbcm*p1p5*
     . p3p4*p6p7+hbcm*p1p5*p3p5*p6p7-2*hbcm*p2p3*p2p5*p6p7-2*hbcm*
     . p2p3*p3p6*p5p7-2*hbcm*p2p3*p4p5*p6p7-2*hbcm*p2p3*p4p6*p5p7-2*
     . hbcm*p2p3*p5p6*p5p7+2*hbcm*p2p5*p3p6*p3p7+2*hbcm*p2p5*p3p7*
     . p4p6+2*hbcm*p2p5*p3p7*p5p6-hbcm*p3p5*p3p6*p4p7-hbcm*p3p5*p3p6*
     . p5p7+2*hbcm*p3p5*p4p5*p6p7-hbcm*p3p5*p4p6*p4p7-hbcm*p3p5*p4p6*
     . p5p7-hbcm*p3p5*p4p7*p5p6-hbcm*p3p5*p5p6*p5p7+2*hbcm*p3p6*p3p7*
     . p4p5+2*hbcm*p3p7*p4p5*p4p6+2*hbcm*p3p7*p4p5*p5p6)
      ans4=8*(p6p7*p2p3*p1p3)*(2*ffmcfmb*hbcm-2*fmb+2*fmc-hbcm)+8*(
     . p5p6*p3p7)*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*hbcm2)+8*(p4p6*
     . p3p7)*(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*hbcm2)+8*(p3p7*p3p6)
     . *(ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm-fmb*hbcm2)+4*p6p7*(2*ffmcfmb*
     . fmb*fmc*hbcm3-2*ffmcfmb*fmb2*hbcm3+4*ffmcfmb*hbcm*p1p3s2+4*
     . ffmcfmb*hbcm*p3p5s2+fmb*fmc2*hbcm2-fmb*hbcm4-fmb3*hbcm2-2*hbcm
     . *p1p3s2-2*hbcm*p3p5s2)+ans5
      ans3=8*(p6p7*p3p5*p3p4)*(-fmb+fmc)+8*(p6p7*p3p4*p1p3)*(fmb-fmc)
     . +8*(p6p7*p4p5)*(fmb*hbcm2-fmc*hbcm2+hbcm3)+8*(p6p7*p2p5)*(-fmb
     . *hbcm2-fmc*hbcm2+hbcm3)+8*(p6p7*p1p4)*(-fmb*hbcm2+fmc*hbcm2-
     . hbcm3)+8*(p6p7*p1p2)*(fmb*hbcm2+fmc*hbcm2-hbcm3)+8*(p6p7*p3p4)
     . *(fmb*fmc*hbcm-fmb2*hbcm)+8*(p6p7*p2p3)*(-2*fmb*fmc*hbcm+fmb*
     . hbcm2+2*fmb2*hbcm)+8*(p5p7*p5p6)*(ffmcfmb*hbcm3-fmb*hbcm2-
     . hbcm3)+8*(p5p7*p4p6)*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+8*(p5p7*
     . p3p6)*(ffmcfmb*hbcm3-fmb*hbcm2-hbcm3)+4*(p6p7*p3p5)*(2*ffmcfmb
     . *hbcm3-2*fmb*hbcm2-fmb2*hbcm+fmc2*hbcm-hbcm3)+8*(p6p7*p1p5)*(2
     . *ffmcfmb*hbcm3+fmb*hbcm2)+4*(p6p7*p1p3)*(-2*ffmcfmb*hbcm3+2*
     . fmb*hbcm2+fmb2*hbcm-fmc2*hbcm+hbcm3)+8*(p5p6*p3p7*p3p5)*(
     . ffmcfmb*hbcm+fmc)+8*(p4p6*p3p7*p3p5)*(ffmcfmb*hbcm+fmc)+8*(
     . p3p7*p3p6*p3p5)*(ffmcfmb*hbcm+fmc)+8*(p6p7*p3p5*p2p3)*(-2*
     . ffmcfmb*hbcm+2*fmb-2*fmc+hbcm)+8*(p5p6*p3p7*p1p3)*(-ffmcfmb*
     . hbcm-fmc)+8*(p4p6*p3p7*p1p3)*(-ffmcfmb*hbcm-fmc)+8*(p3p7*p3p6*
     . p1p3)*(-ffmcfmb*hbcm-fmc)+16*(p6p7*p3p5*p1p3)*(-2*ffmcfmb*hbcm
     . +hbcm)+ans4
      ans2=w15*ans3
      ans8=8*(2*fmc*hbcm2*p1p2*p6p7+fmc*hbcm2*p1p3*p6p7+fmc*hbcm2*
     . p1p5*p6p7-2*fmc*hbcm2*p2p5*p6p7-fmc*hbcm2*p3p5*p6p7+2*hbcm*
     . p1p2*p3p4*p6p7+2*hbcm*p1p3*p1p4*p6p7-2*hbcm*p1p3*p2p4*p6p7-2*
     . hbcm*p1p3*p4p5*p6p7+2*hbcm*p1p4*p2p3*p6p7-2*hbcm*p1p4*p3p5*
     . p6p7+hbcm*p1p5*p3p4*p6p7-2*hbcm*p2p3*p2p4*p6p7-2*hbcm*p2p3*
     . p4p5*p6p7+2*hbcm*p2p4*p3p5*p6p7-2*hbcm*p2p5*p3p4*p6p7+hbcm*
     . p3p4*p3p6*p4p7+hbcm*p3p4*p3p6*p5p7+hbcm*p3p4*p4p6*p4p7+hbcm*
     . p3p4*p4p6*p5p7+hbcm*p3p4*p4p7*p5p6+hbcm*p3p4*p5p6*p5p7+2*hbcm*
     . p3p5*p4p5*p6p7)
      ans7=8*(p6p7*p4p5)*(fmb*hbcm2-fmc*hbcm2+hbcm3)+8*(p6p7*p2p4)*(
     . fmb*hbcm2-fmc*hbcm2+hbcm3)+8*(p6p7*p1p4)*(-fmb*hbcm2+fmc*hbcm2
     . -hbcm3)+8*(p6p7*p2p3)*(-2*fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+
     . 8*(p5p6*p4p7)*(ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p7*p4p6)*(ffmcfmb
     . *hbcm3+fmc*hbcm2)+8*(p4p7*p3p6)*(ffmcfmb*hbcm3+fmc*hbcm2)+4*(
     . p6p7*p3p4)*(-2*ffmcfmb*hbcm3+2*fmb*fmc*hbcm+2*fmb*hbcm2-fmb2*
     . hbcm-2*fmc*hbcm2-fmc2*hbcm+hbcm3)+8*(p5p6*p3p7*p3p4)*(-ffmcfmb
     . *hbcm-fmc)+8*(p4p6*p3p7*p3p4)*(-ffmcfmb*hbcm-fmc)+8*(p3p7*p3p6
     . *p3p4)*(-ffmcfmb*hbcm-fmc)+8*(p6p7*p3p5*p3p4)*(-2*ffmcfmb*hbcm
     . +hbcm)+8*(p6p7*p3p4*p2p3)*(2*ffmcfmb*hbcm-2*fmb+2*fmc-hbcm)+8*
     . (p6p7*p3p4*p1p3)*(2*ffmcfmb*hbcm-hbcm)+8*(p5p6*p3p7)*(ffmcfmb*
     . fmc*hbcm2-fmc*hbcm2-fmc2*hbcm)+8*(p4p6*p3p7)*(ffmcfmb*fmc*
     . hbcm2-fmc*hbcm2-fmc2*hbcm)+8*(p3p7*p3p6)*(ffmcfmb*fmc*hbcm2-
     . fmc*hbcm2-fmc2*hbcm)+4*p6p7*(2*ffmcfmb*fmb*fmc*hbcm3-2*ffmcfmb
     . *fmc2*hbcm3+2*fmb*p3p4s2-fmb2*fmc*hbcm2-fmc*hbcm4-2*fmc*p3p4s2
     . +fmc3*hbcm2)+ans8
      ans6=w1*ans7
      ans9=(16*(p6p7*p3p4)*(-fb2*fmb+fb2*fmc)+8*(p6p7*p3p5)*(2*fb1*
     . hbcm+3*hbcm)+8*(p6p7*p2p3)*(-2*fb1*hbcm+4*fb2*fmb-4*fb2*fmc-3*
     . hbcm)+8*(p6p7*p1p3)*(-2*fb1*hbcm-3*hbcm)+4*p6p7*(2*fb1*fmb2*
     . hbcm-2*fb1*fmc2*hbcm+2*fb1*hbcm3-4*fb2*ffmcfmb*fmb*hbcm2+4*fb2
     . *ffmcfmb*fmc*hbcm2+5*fmb*hbcm2-5*fmc*hbcm2+3*hbcm3)+4*(p5p6*
     . p3p7)*(-12*fb1*ffmcfmb*hbcm+8*fb1*hbcm+4*fb2*fmc+3*hbcm)+4*(
     . p4p6*p3p7)*(-12*fb1*ffmcfmb*hbcm+8*fb1*hbcm+4*fb2*fmc+3*hbcm)+
     . 4*(p3p7*p3p6)*(-12*fb1*ffmcfmb*hbcm+8*fb1*hbcm+4*fb2*fmc+3*
     . hbcm)+16*(-2*fb1*hbcm*p1p2*p6p7-fb1*hbcm*p1p5*p6p7+2*fb1*hbcm*
     . p2p5*p6p7-fb1*hbcm*p3p6*p4p7+fb1*hbcm*p3p6*p5p7-fb1*hbcm*p4p6*
     . p4p7+fb1*hbcm*p4p6*p5p7-fb1*hbcm*p4p7*p5p6+fb1*hbcm*p5p6*p5p7)
     . )
      ans1=ans2+ans6+ans9
      ans=ccc*ans1
      b(1)=ans
      ans2=(4*p5p7*(-4*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-3*hbcm)+2*p3p7*(
     . 12*fb1*ffmcfmb*hbcm-8*fb1*hbcm-8*fb2*ffmcfmb*fmb+8*fb2*ffmcfmb
     . *fmc+4*fb2*fmb-8*fb2*fmc-2*fmb+2*fmc-3*hbcm)+8*fb1*hbcm*p4p7)
      ans1=w1*(4*(p5p7*p3p4)*(2*ffmcfmb*hbcm-fmb+fmc-2*hbcm)+4*p3p7*(
     . -ffmcfmb*fmc*hbcm2+fmc*hbcm2+fmc2*hbcm)+4*(p3p7*p3p4)*(2*
     . ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmb+2*fmc)+4*p4p7*(-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmc*hbcm2)+4
     . *(fmc*hbcm2*p5p7-2*hbcm*p1p2*p4p7+2*hbcm*p1p4*p5p7-2*hbcm*p2p4
     . *p5p7+2*hbcm*p2p5*p4p7-hbcm*p3p4*p4p7-2*hbcm*p4p5*p5p7))+w15*(
     . 4*p4p7*(-fmb*fmc*hbcm+fmb2*hbcm)+4*(p5p7*p3p5)*(-2*ffmcfmb*
     . hbcm+fmb-fmc+2*hbcm)+4*(p5p7*p1p3)*(2*ffmcfmb*hbcm-fmb+fmc-2*
     . hbcm)+4*(p3p7*p3p5)*(-2*ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm
     . +fmb-2*fmc)+4*(p3p7*p1p3)*(2*ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb
     . *hbcm-fmb+2*fmc)+4*p5p7*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm+fmc*hbcm2+hbcm3
     . )+4*p3p7*(-ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2)+4*(2*hbcm
     . *p1p2*p3p7-hbcm*p1p3*p4p7+2*hbcm*p1p4*p3p7+2*hbcm*p1p4*p5p7-2*
     . hbcm*p1p5*p4p7-2*hbcm*p1p5*p5p7+2*hbcm*p2p3*p5p7-2*hbcm*p2p5*
     . p3p7+hbcm*p3p5*p4p7-2*hbcm*p3p7*p4p5-2*hbcm*p4p5*p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w15*(4*p5p6*(-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+4*p4p6*(-
     . fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm)+4*p3p6*(-fmb*fmc*hbcm+fmb*
     . hbcm2+fmb2*hbcm)+4*(p5p6*p3p5)*(-2*ffmcfmb*hbcm+fmb-fmc+hbcm)+
     . 4*(p4p6*p3p5)*(-2*ffmcfmb*hbcm+fmb-fmc+hbcm)+4*(p3p6*p3p5)*(-2
     . *ffmcfmb*hbcm+fmb-fmc+hbcm)+4*(p5p6*p1p3)*(2*ffmcfmb*hbcm-fmb+
     . fmc-hbcm)+4*(p4p6*p1p3)*(2*ffmcfmb*hbcm-fmb+fmc-hbcm)+4*(p3p6*
     . p1p3)*(2*ffmcfmb*hbcm-fmb+fmc-hbcm)+8*(hbcm*p1p2*p3p6+hbcm*
     . p1p2*p4p6+hbcm*p1p2*p5p6+hbcm*p1p4*p3p6+hbcm*p1p4*p4p6+hbcm*
     . p1p4*p5p6-hbcm*p1p5*p3p6-hbcm*p1p5*p4p6-hbcm*p1p5*p5p6-hbcm*
     . p2p5*p3p6-hbcm*p2p5*p4p6-hbcm*p2p5*p5p6-hbcm*p3p6*p4p5-hbcm*
     . p4p5*p4p6-hbcm*p4p5*p5p6))+(4*p5p6*(-2*fb1*hbcm+2*fb2*fmb-2*
     . fb2*fmc-3*hbcm)+4*p4p6*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-3*hbcm
     . )+4*p3p6*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-3*hbcm))
      ans1=w1*(4*p5p6*(-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*p4p6*(-
     . fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*p3p6*(-fmb*fmc*hbcm+fmc*
     . hbcm2+fmc2*hbcm)+4*(p5p6*p3p4)*(2*ffmcfmb*hbcm-fmb+fmc-hbcm)+4
     . *(p4p6*p3p4)*(2*ffmcfmb*hbcm-fmb+fmc-hbcm)+4*(p3p6*p3p4)*(2*
     . ffmcfmb*hbcm-fmb+fmc-hbcm)+8*(hbcm*p1p4*p3p6+hbcm*p1p4*p4p6+
     . hbcm*p1p4*p5p6-hbcm*p2p4*p3p6-hbcm*p2p4*p4p6-hbcm*p2p4*p5p6-
     . hbcm*p3p6*p4p5-hbcm*p4p5*p4p6-hbcm*p4p5*p5p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans3=((p5p6*p3p7)*(32*fb2*ffmcfmb-16*fb2+8)+(p4p6*p3p7)*(32*fb2
     . *ffmcfmb-16*fb2+8)+(p3p7*p3p6)*(32*fb2*ffmcfmb-16*fb2+8)+4*
     . p6p7*(4*fb2*ffmcfmb*hbcm2-2*fb2*fmb2+2*fb2*fmc2-2*fb2*hbcm2+3*
     . fmb*hbcm+3*fmc*hbcm-5*hbcm2)+16*(fb2*p1p3*p6p7+fb2*p1p5*p6p7+
     . fb2*p2p3*p6p7+2*fb2*p2p4*p6p7+fb2*p3p4*p6p7-fb2*p3p5*p6p7))
      ans2=w15*(8*(p6p7*p4p5)*(fmb*hbcm+fmc*hbcm-hbcm2)+8*(p6p7*p2p5)
     . *(fmb*hbcm+fmc*hbcm-hbcm2)+8*(p6p7*p1p4)*(-fmb*hbcm-fmc*hbcm+
     . hbcm2)+8*(p6p7*p1p2)*(-fmb*hbcm-fmc*hbcm+hbcm2)+(p5p6*p3p7*
     . p3p5)*(16*ffmcfmb-8)+(p4p6*p3p7*p3p5)*(16*ffmcfmb-8)+(p3p7*
     . p3p6*p3p5)*(16*ffmcfmb-8)+(p5p6*p3p7*p1p3)*(-16*ffmcfmb+8)+(
     . p4p6*p3p7*p1p3)*(-16*ffmcfmb+8)+(p3p7*p3p6*p1p3)*(-16*ffmcfmb+
     . 8)+8*(p5p7*p5p6)*(-ffmcfmb*hbcm2+hbcm2)+8*(p5p7*p4p6)*(-
     . ffmcfmb*hbcm2+hbcm2)+8*(p5p7*p3p6)*(-ffmcfmb*hbcm2+hbcm2)+4*
     . p6p7*(2*ffmcfmb*fmb*hbcm3+fmb*fmc2*hbcm-fmb*hbcm3-fmb3*hbcm-2*
     . p1p3s2-2*p3p5s2)+4*(p6p7*p3p5)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*
     . fmc*hbcm-2*fmb*hbcm-fmb2+fmc2-hbcm2)+4*(p6p7*p1p3)*(-2*ffmcfmb
     . *fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2+hbcm2)+8*(
     . fmb*hbcm*p1p5*p6p7+fmb*hbcm*p2p3*p6p7+2*fmb*hbcm*p2p4*p6p7+fmb
     . *hbcm*p3p4*p6p7-fmb*hbcm*p3p6*p4p7-fmb*hbcm*p4p6*p4p7-fmb*hbcm
     . *p4p7*p5p6-p1p3*p1p5*p6p7-p1p3*p2p3*p6p7-2*p1p3*p2p4*p6p7-p1p3
     . *p3p4*p6p7+2*p1p3*p3p5*p6p7+p1p5*p3p5*p6p7+p2p3*p3p5*p6p7+2*
     . p2p4*p3p5*p6p7+p3p4*p3p5*p6p7))+ans3
      ans1=w1*(8*(p6p7*p4p5)*(fmb*hbcm+fmc*hbcm-hbcm2)+8*(p6p7*p2p4)*
     . (fmb*hbcm-fmc*hbcm-hbcm2)+8*(p6p7*p1p4)*(-fmb*hbcm-fmc*hbcm+
     . hbcm2)+(p5p6*p3p7*p3p4)*(-16*ffmcfmb+8)+(p4p6*p3p7*p3p4)*(-16*
     . ffmcfmb+8)+(p3p7*p3p6*p3p4)*(-16*ffmcfmb+8)+4*p6p7*(-2*ffmcfmb
     . *fmc*hbcm3+fmb2*fmc*hbcm+fmc*hbcm3-fmc3*hbcm-2*p3p4s2)+4*(p6p7
     . *p3p4)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2
     . -fmc2-hbcm2)+8*(ffmcfmb*hbcm2*p3p6*p4p7+ffmcfmb*hbcm2*p4p6*
     . p4p7+ffmcfmb*hbcm2*p4p7*p5p6-fmc*hbcm*p1p3*p6p7-fmc*hbcm*p1p5*
     . p6p7-fmc*hbcm*p2p3*p6p7+fmc*hbcm*p3p5*p6p7-fmc*hbcm*p3p6*p5p7-
     . fmc*hbcm*p4p6*p5p7-fmc*hbcm*p5p6*p5p7-p1p3*p3p4*p6p7-p1p5*p3p4
     . *p6p7-p2p3*p3p4*p6p7-2*p2p4*p3p4*p6p7+p3p4*p3p5*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans4=4*(2*fmb*hbcm*p2p3*p4p7+2*fmb*hbcm*p3p4*p5p7-fmb*hbcm3*
     . p4p7+2*fmc*hbcm*p1p2*p3p7-2*fmc*hbcm*p2p5*p3p7+2*hbcm2*p1p4*
     . p5p7-4*hbcm2*p1p5*p5p7-2*hbcm2*p4p5*p5p7+2*p1p2*p1p3*p3p7-2*
     . p1p2*p3p5*p3p7+2*p1p3*p2p3*p5p7-2*p1p3*p2p5*p3p7-2*p1p3*p3p4*
     . p5p7+4*p1p3*p3p5*p5p7-2*p2p3*p3p5*p5p7+2*p2p5*p3p5*p3p7+2*p3p4
     . *p3p5*p5p7)
      ans3=8*(p4p5*p3p7)*(-fmb*hbcm-fmc*hbcm)+4*(p4p7*p3p5)*(-fmb*
     . hbcm+fmc*hbcm)+8*(p3p7*p1p4)*(fmb*hbcm+fmc*hbcm)+4*(p4p7*p1p3)
     . *(fmb*hbcm-fmc*hbcm)+(p3p7*p3p5*p2p3)*(-16*ffmcfmb+8)+(p3p7*
     . p3p5*p1p3)*(-32*ffmcfmb+16)+(p3p7*p2p3*p1p3)*(16*ffmcfmb-8)+4*
     . (p5p7*p3p5)*(-2*ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2)+8*(p5p7*p2p3)*
     . (ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+8*(p3p7*p1p5)*(ffmcfmb*hbcm2+
     . fmc*hbcm-hbcm2)+4*(p5p7*p1p3)*(2*ffmcfmb*hbcm2-2*fmc*hbcm-
     . hbcm2)+4*p5p7*(ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*
     . hbcm4+2*fmb*fmc*hbcm2+2*fmb2*hbcm2+fmc*hbcm3+hbcm4-2*p1p3s2-2*
     . p3p5s2)+4*(p3p7*p3p5)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+2*
     . ffmcfmb*hbcm2-fmb*fmc-fmc2-hbcm2)+4*(p3p7*p1p3)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmc2+hbcm2)+4*
     . p3p7*(-ffmcfmb*fmb*fmc*hbcm2-ffmcfmb*fmb2*hbcm2+4*ffmcfmb*
     . p1p3s2+4*ffmcfmb*p3p5s2+fmb*fmc*hbcm2-fmb*fmc2*hbcm-fmb2*fmc*
     . hbcm+fmb2*hbcm2-2*p1p3s2-2*p3p5s2)+ans4
      ans2=w15*ans3
      ans6=((p3p7*p3p5)*(32*fb2*ffmcfmb-16*fb2+8)+(p3p7*p2p3)*(-32*
     . fb2*ffmcfmb+16*fb2-8)+(p3p7*p1p3)*(-32*fb2*ffmcfmb+16*fb2-8)+4
     . *p5p7*(-4*fb1*fmb*hbcm-4*fb1*fmc*hbcm+4*fb2*ffmcfmb*hbcm2-2*
     . fb2*hbcm2-5*hbcm2)+8*p4p7*(fb1*fmb*hbcm+fb1*fmc*hbcm)+2*p3p7*(
     . 12*fb1*ffmcfmb*fmb*hbcm+12*fb1*ffmcfmb*fmc*hbcm-8*fb1*fmb*hbcm
     . -8*fb1*fmc*hbcm+8*fb2*ffmcfmb*hbcm2-4*fb2*fmb*fmc-4*fb2*fmc2-4
     . *fb2*hbcm2-3*fmb*hbcm-3*fmc*hbcm+2*hbcm2)+16*(-fb2*p1p2*p3p7+
     . fb2*p1p3*p5p7-fb2*p2p3*p5p7+fb2*p2p5*p3p7+fb2*p3p4*p5p7-fb2*
     . p3p5*p5p7))
      ans5=w1*(4*(p5p7*p3p4)*(-2*fmc*hbcm-hbcm2)+4*(p4p7*p3p4)*(-fmb*
     . hbcm-fmc*hbcm)+(p3p7*p3p5*p3p4)*(-16*ffmcfmb+8)+(p3p7*p3p4*
     . p2p3)*(16*ffmcfmb-8)+(p3p7*p3p4*p1p3)*(16*ffmcfmb-8)+4*p5p7*(-
     . 2*ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc2*hbcm2-2*p3p4s2)+4*p4p7*
     . (-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2-fmc2*hbcm2)+4*(p3p7*p3p4)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*
     . hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmc2+hbcm2)+4*p3p7*(-ffmcfmb*fmb*
     . fmc*hbcm2-ffmcfmb*fmc2*hbcm2+fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmc2*
     . hbcm2+fmc3*hbcm)+8*(-ffmcfmb*hbcm2*p1p3*p4p7-ffmcfmb*hbcm2*
     . p2p3*p4p7+ffmcfmb*hbcm2*p3p5*p4p7+fmc*hbcm*p1p2*p3p7+2*fmc*
     . hbcm*p2p3*p5p7-fmc*hbcm*p2p5*p3p7-hbcm2*p1p2*p4p7+hbcm2*p1p4*
     . p5p7-hbcm2*p2p4*p5p7+hbcm2*p2p5*p4p7-hbcm2*p4p5*p5p7+p1p2*p3p4
     . *p3p7-p1p3*p3p4*p5p7+p2p3*p3p4*p5p7-p2p5*p3p4*p3p7+p3p4*p3p5*
     . p5p7))+ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans4=8*(-fmb*hbcm*p2p3*p4p6-fmb*hbcm*p2p3*p5p6+fmb*hbcm*p3p4*
     . p4p6+fmb*hbcm*p3p4*p5p6+hbcm2*p1p2*p4p6+hbcm2*p1p2*p5p6+hbcm2*
     . p1p4*p4p6+hbcm2*p1p4*p5p6-hbcm2*p1p5*p4p6-hbcm2*p1p5*p5p6-
     . hbcm2*p2p5*p4p6-hbcm2*p2p5*p5p6-hbcm2*p4p5*p4p6-hbcm2*p4p5*
     . p5p6+2*p1p2*p1p3*p3p6-2*p1p2*p3p5*p3p6+p1p3*p1p5*p3p6+p1p3*
     . p2p3*p4p6+p1p3*p2p3*p5p6-2*p1p3*p2p5*p3p6-p1p3*p3p4*p4p6-p1p3*
     . p3p4*p5p6+2*p1p3*p3p5*p4p6+2*p1p3*p3p5*p5p6-p1p5*p3p5*p3p6-
     . p2p3*p3p5*p4p6-p2p3*p3p5*p5p6+2*p2p5*p3p5*p3p6+p3p4*p3p5*p4p6+
     . p3p4*p3p5*p5p6)
      ans3=8*(p4p5*p3p6)*(-fmb*hbcm-fmc*hbcm)+4*(p5p6*p3p5)*(-fmb*
     . hbcm+fmc*hbcm-hbcm2)+4*(p4p6*p3p5)*(-fmb*hbcm+fmc*hbcm-hbcm2)+
     . 8*(p3p6*p2p5)*(fmb*hbcm-fmc*hbcm)+8*(p3p6*p1p5)*(-fmb*hbcm-
     . hbcm2)+8*(p3p6*p1p4)*(fmb*hbcm+fmc*hbcm)+4*(p5p6*p1p3)*(fmb*
     . hbcm-fmc*hbcm+hbcm2)+4*(p4p6*p1p3)*(fmb*hbcm-fmc*hbcm+hbcm2)+8
     . *(p3p6*p1p2)*(-fmb*hbcm+fmc*hbcm)+4*p3p6*(fmb*fmc*hbcm2-fmb*
     . fmc2*hbcm+fmb2*hbcm2+fmb3*hbcm)+4*p5p6*(2*ffmcfmb*fmb*hbcm3+
     . fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-2*p1p3s2-2*p3p5s2)+4*p4p6*(
     . 2*ffmcfmb*fmb*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-2*
     . p1p3s2-2*p3p5s2)+4*(p3p6*p3p5)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*
     . fmc*hbcm+fmb*hbcm+fmb2+fmc*hbcm-fmc2)+4*(p3p6*p1p3)*(2*ffmcfmb
     . *fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*hbcm-fmb2-fmc*hbcm+fmc2)+ans4
      ans2=w15*ans3
      ans5=(4*p5p6*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+4*fb2*ffmcfmb*
     . hbcm2-2*fb2*hbcm2-5*hbcm2)+4*p4p6*(-2*fb1*fmb*hbcm-2*fb1*fmc*
     . hbcm+4*fb2*ffmcfmb*hbcm2-2*fb2*hbcm2-5*hbcm2)+4*p3p6*(-2*fb1*
     . fmb*hbcm-2*fb1*fmc*hbcm+2*fb2*fmb2-2*fb2*fmc2-3*fmb*hbcm-3*fmc
     . *hbcm)+16*(-2*fb2*p1p2*p3p6+fb2*p1p3*p4p6+fb2*p1p3*p5p6-fb2*
     . p1p5*p3p6-fb2*p2p3*p4p6-fb2*p2p3*p5p6+2*fb2*p2p5*p3p6+fb2*p3p4
     . *p4p6+fb2*p3p4*p5p6-fb2*p3p5*p4p6-fb2*p3p5*p5p6))
      ans1=w1*(8*(p4p5*p3p6)*(-fmb*hbcm-fmc*hbcm)+4*(p5p6*p3p4)*(fmb*
     . hbcm-fmc*hbcm-hbcm2)+4*(p4p6*p3p4)*(fmb*hbcm-fmc*hbcm-hbcm2)+8
     . *(p3p6*p2p4)*(-fmb*hbcm-fmc*hbcm)+8*(p3p6*p1p4)*(fmb*hbcm+fmc*
     . hbcm)+4*p3p6*(fmb*fmc*hbcm2-fmb2*fmc*hbcm+fmc2*hbcm2+fmc3*hbcm
     . )+4*p5p6*(-2*ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc*hbcm3+fmc2*
     . hbcm2-2*p3p4s2)+4*p4p6*(-2*ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc
     . *hbcm3+fmc2*hbcm2-2*p3p4s2)+4*(p3p6*p3p4)*(2*ffmcfmb*fmb*hbcm+
     . 2*ffmcfmb*fmc*hbcm-fmb*hbcm-fmb2-fmc*hbcm+fmc2)+8*(2*fmc*hbcm*
     . p1p2*p3p6-fmc*hbcm*p1p3*p4p6-fmc*hbcm*p1p3*p5p6+fmc*hbcm*p1p5*
     . p3p6+fmc*hbcm*p2p3*p4p6+fmc*hbcm*p2p3*p5p6-2*fmc*hbcm*p2p5*
     . p3p6+fmc*hbcm*p3p5*p4p6+fmc*hbcm*p3p5*p5p6+hbcm2*p1p4*p4p6+
     . hbcm2*p1p4*p5p6-hbcm2*p2p4*p4p6-hbcm2*p2p4*p5p6-hbcm2*p4p5*
     . p4p6-hbcm2*p4p5*p5p6+2*p1p2*p3p4*p3p6-p1p3*p3p4*p4p6-p1p3*p3p4
     . *p5p6+p1p5*p3p4*p3p6+p2p3*p3p4*p4p6+p2p3*p3p4*p5p6-2*p2p5*p3p4
     . *p3p6+p3p4*p3p5*p4p6+p3p4*p3p5*p5p6))+ans2+ans5
      ans=ccc*ans1
      b(6)=ans
      ans2=2*(4*fb2*ffmcfmb*hbcm2-2*fb2*fmb2+2*fb2*fmc2-2*fb2*hbcm2+8
     . *fb2*p1p2+4*fb2*p1p3+4*fb2*p1p5-4*fb2*p2p3-8*fb2*p2p5+4*fb2*
     . p3p4-4*fb2*p3p5+3*fmb*hbcm+3*fmc*hbcm-5*hbcm2)
      ans1=w1*(4*p4p5*(fmb*hbcm+fmc*hbcm-hbcm2)+4*p2p4*(fmb*hbcm+fmc*
     . hbcm-hbcm2)+4*p1p4*(-fmb*hbcm-fmc*hbcm+hbcm2)+2*p3p4*(-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2-hbcm2
     . )+2*(-2*ffmcfmb*fmc*hbcm3+fmb2*fmc*hbcm-4*fmc*hbcm*p1p2-2*fmc*
     . hbcm*p1p3-2*fmc*hbcm*p1p5+2*fmc*hbcm*p2p3+4*fmc*hbcm*p2p5+2*
     . fmc*hbcm*p3p5+fmc*hbcm3-fmc3*hbcm-4*p1p2*p3p4-2*p1p3*p3p4-2*
     . p1p5*p3p4+2*p2p3*p3p4+4*p2p5*p3p4+2*p3p4*p3p5-2*p3p4s2))+w15*(
     . 4*p4p5*(fmb*hbcm+fmc*hbcm-hbcm2)+4*p2p5*(-fmb*hbcm+fmc*hbcm-
     . hbcm2)+4*p1p4*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*p1p2*(fmb*hbcm-fmc*
     . hbcm+hbcm2)+2*p3p5*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-2*
     . fmb*hbcm-fmb2+fmc2-hbcm2)+2*p1p3*(-2*ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2+hbcm2)+2*(2*ffmcfmb*fmb*
     . hbcm3+fmb*fmc2*hbcm+2*fmb*hbcm*p1p5-2*fmb*hbcm*p2p3+2*fmb*hbcm
     . *p3p4-fmb*hbcm3-fmb3*hbcm-4*p1p2*p1p3+4*p1p2*p3p5-2*p1p3*p1p5+
     . 2*p1p3*p2p3+4*p1p3*p2p5-2*p1p3*p3p4+4*p1p3*p3p5-2*p1p3s2+2*
     . p1p5*p3p5-2*p2p3*p3p5-4*p2p5*p3p5+2*p3p4*p3p5-2*p3p5s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans3=(8*p3p4*(fb2*fmb-fb2*fmc)+4*p3p5*(-2*fb1*hbcm-3*hbcm)+4*
     . p2p3*(2*fb1*hbcm-4*fb2*fmb+4*fb2*fmc+3*hbcm)+4*p1p3*(2*fb1*
     . hbcm+3*hbcm)+2*(-2*fb1*fmb2*hbcm+2*fb1*fmc2*hbcm+8*fb1*hbcm*
     . p1p2+4*fb1*hbcm*p1p5-8*fb1*hbcm*p2p5-2*fb1*hbcm3+4*fb2*ffmcfmb
     . *fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-5*fmb*hbcm2+5*fmc*hbcm2-3*
     . hbcm3))
      ans2=w1*(4*p4p5*(-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*p2p4*(-fmb*hbcm2
     . +fmc*hbcm2-hbcm3)+4*p1p4*(fmb*hbcm2-fmc*hbcm2+hbcm3)+4*p2p3*(2
     . *fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+2*p3p4*(2*ffmcfmb*hbcm3-2
     . *fmb*fmc*hbcm-2*fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2+fmc2*hbcm-
     . hbcm3)+4*(p3p5*p3p4)*(2*ffmcfmb*hbcm-hbcm)+4*(p3p4*p2p3)*(-2*
     . ffmcfmb*hbcm+2*fmb-2*fmc+hbcm)+4*(p3p4*p1p3)*(-2*ffmcfmb*hbcm+
     . hbcm)+2*(-2*ffmcfmb*fmb*fmc*hbcm3+2*ffmcfmb*fmc2*hbcm3-2*fmb*
     . p3p4s2+fmb2*fmc*hbcm2-4*fmc*hbcm2*p1p2-2*fmc*hbcm2*p1p3-2*fmc*
     . hbcm2*p1p5+4*fmc*hbcm2*p2p5+2*fmc*hbcm2*p3p5+fmc*hbcm4+2*fmc*
     . p3p4s2-fmc3*hbcm2-4*hbcm*p1p2*p3p4-4*hbcm*p1p3*p1p4+4*hbcm*
     . p1p3*p2p4+4*hbcm*p1p3*p4p5-4*hbcm*p1p4*p2p3+4*hbcm*p1p4*p3p5-2
     . *hbcm*p1p5*p3p4+4*hbcm*p2p3*p2p4+4*hbcm*p2p3*p4p5-4*hbcm*p2p4*
     . p3p5+4*hbcm*p2p5*p3p4-4*hbcm*p3p5*p4p5))+ans3
      ans1=w15*(4*(p3p5*p3p4)*(fmb-fmc)+4*(p3p4*p1p3)*(-fmb+fmc)+4*
     . p4p5*(-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*p2p5*(fmb*hbcm2+fmc*hbcm2-
     . hbcm3)+4*p1p4*(fmb*hbcm2-fmc*hbcm2+hbcm3)+4*p1p2*(-fmb*hbcm2-
     . fmc*hbcm2+hbcm3)+4*p3p4*(-fmb*fmc*hbcm+fmb2*hbcm)+4*p2p3*(2*
     . fmb*fmc*hbcm-fmb*hbcm2-2*fmb2*hbcm)+2*p3p5*(-2*ffmcfmb*hbcm3+2
     . *fmb*hbcm2+fmb2*hbcm-fmc2*hbcm+hbcm3)+4*p1p5*(-2*ffmcfmb*hbcm3
     . -fmb*hbcm2)+2*p1p3*(2*ffmcfmb*hbcm3-2*fmb*hbcm2-fmb2*hbcm+fmc2
     . *hbcm-hbcm3)+4*(p3p5*p2p3)*(2*ffmcfmb*hbcm-2*fmb+2*fmc-hbcm)+8
     . *(p3p5*p1p3)*(2*ffmcfmb*hbcm-hbcm)+4*(p2p3*p1p3)*(-2*ffmcfmb*
     . hbcm+2*fmb-2*fmc+hbcm)+2*(-2*ffmcfmb*fmb*fmc*hbcm3+2*ffmcfmb*
     . fmb2*hbcm3-4*ffmcfmb*hbcm*p1p3s2-4*ffmcfmb*hbcm*p3p5s2-fmb*
     . fmc2*hbcm2+fmb*hbcm4+fmb3*hbcm2-4*hbcm*p1p2*p2p3-4*hbcm*p1p3*
     . p1p4+2*hbcm*p1p3*p1p5+4*hbcm*p1p3*p4p5+2*hbcm*p1p3s2-4*hbcm*
     . p1p4*p2p3+4*hbcm*p1p4*p3p5+8*hbcm*p1p5*p2p3-4*hbcm*p1p5*p3p4-2
     . *hbcm*p1p5*p3p5+4*hbcm*p2p3*p2p5+4*hbcm*p2p3*p4p5-4*hbcm*p3p5*
     . p4p5+2*hbcm*p3p5s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(w1*(4*p3p6*(3*fmb*fmc*hbcm-fmc*hbcm2-3*fmc2*hbcm)+4*(
     . p3p6*p3p4)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+hbcm)+8*(fmc*hbcm2*
     . p4p6+fmc*hbcm2*p5p6-hbcm*p1p4*p3p6+hbcm*p2p4*p3p6+hbcm*p3p4*
     . p4p6+hbcm*p3p4*p5p6+hbcm*p3p6*p4p5))+w15*(4*p3p6*(3*fmb*fmc*
     . hbcm-fmb*hbcm2-3*fmb2*hbcm)+4*(p3p6*p3p5)*(2*ffmcfmb*hbcm-3*
     . fmb+3*fmc-hbcm)+4*(p3p6*p1p3)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+
     . hbcm)+8*(fmb*hbcm2*p4p6+fmb*hbcm2*p5p6-hbcm*p1p2*p3p6-hbcm*
     . p1p3*p4p6-hbcm*p1p3*p5p6-hbcm*p1p4*p3p6+3*hbcm*p1p5*p3p6+hbcm*
     . p2p5*p3p6+hbcm*p3p5*p4p6+hbcm*p3p5*p5p6+hbcm*p3p6*p4p5))+(4*
     . p3p6*(2*fb1*hbcm-6*fb2*fmb+6*fb2*fmc+3*hbcm)+16*(-fb1*hbcm*
     . p4p6-fb1*hbcm*p5p6)))
      b(10)=ccc*(8*w1*(-fmc*hbcm*p3p6-fmc*hbcm*p4p6-fmc*hbcm*p5p6-
     . p3p4*p3p6-p3p4*p4p6-p3p4*p5p6)+8*w15*(fmb*hbcm*p3p6+fmb*hbcm*
     . p4p6+fmb*hbcm*p5p6-p1p3*p3p6-p1p3*p4p6-p1p3*p5p6+p3p5*p3p6+
     . p3p5*p4p6+p3p5*p5p6)+16*(fb2*p3p6+fb2*p4p6+fb2*p5p6))
      b(11)=ccc*(w1*(2*p3p4*(-3*fmb*hbcm-fmc*hbcm-hbcm2)+2*(2*ffmcfmb
     . *fmc*hbcm3-3*fmb*fmc*hbcm2+6*fmc*hbcm*p1p3+2*fmc*hbcm*p2p3-6*
     . fmc*hbcm*p3p5-3*fmc*hbcm3-3*fmc2*hbcm2-2*hbcm2*p1p4+2*hbcm2*
     . p2p4+2*hbcm2*p4p5+6*p1p3*p3p4+2*p2p3*p3p4-6*p3p4*p3p5+2*p3p4s2
     . ))+w15*(6*p3p5*(fmb*hbcm-fmc*hbcm+hbcm2)+6*p1p3*(-fmb*hbcm+fmc
     . *hbcm-hbcm2)+2*(-2*ffmcfmb*fmb*hbcm3-3*fmb*fmc*hbcm2-2*fmb*
     . hbcm*p2p3-2*fmb*hbcm*p3p4+3*fmb*hbcm3-3*fmb2*hbcm2-2*hbcm2*
     . p1p2-2*hbcm2*p1p4+6*hbcm2*p1p5+2*hbcm2*p2p5+2*hbcm2*p4p5+2*
     . p1p3*p2p3+2*p1p3*p3p4-12*p1p3*p3p5+6*p1p3s2-2*p2p3*p3p5-2*p3p4
     . *p3p5+6*p3p5s2))+2*(6*fb1*fmb*hbcm+6*fb1*fmc*hbcm-4*fb2*
     . ffmcfmb*hbcm2+6*fb2*hbcm2-12*fb2*p1p3-4*fb2*p2p3-4*fb2*p3p4+12
     . *fb2*p3p5+5*hbcm2))
      ans=ccc*(w1*(4*p4p7*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2+hbcm3
     . )+4*(p3p7*p3p4)*(ffmcfmb*hbcm-fmb+2*fmc)+4*p3p7*(-ffmcfmb*fmc*
     . hbcm2-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+4*(-2*fmc*hbcm2*p5p7
     . -2*hbcm*p1p3*p4p7-2*hbcm*p2p3*p4p7-hbcm*p3p4*p4p7-3*hbcm*p3p4*
     . p5p7+2*hbcm*p3p5*p4p7))+w15*(4*p5p7*(-ffmcfmb*hbcm3-fmc*hbcm2+
     . 2*hbcm3)+4*(p3p7*p3p5)*(-ffmcfmb*hbcm+fmb-2*fmc)+4*(p3p7*p1p3)
     . *(ffmcfmb*hbcm-fmb+2*fmc)+4*p3p7*(-ffmcfmb*fmb*hbcm2-2*fmb*fmc
     . *hbcm+fmb*hbcm2+fmb2*hbcm)+4*(2*hbcm*p1p2*p3p7-hbcm*p1p3*p4p7-
     . hbcm*p1p3*p5p7+2*hbcm*p1p4*p3p7-2*hbcm*p1p5*p3p7-2*hbcm*p2p5*
     . p3p7+hbcm*p3p5*p4p7+hbcm*p3p5*p5p7-2*hbcm*p3p7*p4p5))+(2*p3p7*
     . (12*fb1*ffmcfmb*hbcm-8*fb1*hbcm+4*fb2*fmb-8*fb2*fmc-3*hbcm)+8*
     . (fb1*hbcm*p4p7+fb1*hbcm*p5p7)))
      b(12)=ans
      b(13)=ccc*(w1*(2*p3p4*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+2*(-3*
     . fmb*fmc*hbcm+3*fmc*hbcm2+3*fmc2*hbcm+2*hbcm*p1p4-2*hbcm*p2p4-2
     . *hbcm*p4p5))+w15*(2*p3p5*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+
     . 2*p1p3*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm)+2*(-3*fmb*fmc*hbcm+
     . 3*fmb*hbcm2+3*fmb2*hbcm+2*hbcm*p1p2+2*hbcm*p1p4-6*hbcm*p1p5-2*
     . hbcm*p2p5-2*hbcm*p4p5))+6*(-2*fb1*hbcm+2*fb2*fmb-2*fb2*fmc-
     . hbcm))
      b(14)=ccc*(w15*(4*p5p7*(-ffmcfmb*hbcm2+fmb*hbcm-fmc*hbcm+2*
     . hbcm2)+4*(-2*ffmcfmb*p1p3*p3p7+2*ffmcfmb*p3p5*p3p7+fmb*hbcm*
     . p3p7-fmb*hbcm*p4p7-2*p1p3*p5p7+2*p3p5*p5p7))+w1*(4*p4p7*(
     . ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm+hbcm2)+4*(-2*ffmcfmb*p3p4*p3p7
     . -fmc*hbcm*p3p7-3*fmc*hbcm*p5p7-2*p3p4*p5p7))+(p3p7*(16*fb2*
     . ffmcfmb+4)+16*fb2*p5p7))
      ans2=w15*(4*(p6p7*p3p5)*(-fmb*hbcm-3*fmc*hbcm-hbcm2)+4*(p6p7*
     . p1p3)*(fmb*hbcm+3*fmc*hbcm+hbcm2)+(p3p7*p3p6*p3p5)*(-16*
     . ffmcfmb+8)+(p3p7*p3p6*p1p3)*(16*ffmcfmb-8)+8*(p5p7*p3p6)*(
     . ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-hbcm2)+4*p6p7*(-2*ffmcfmb*fmb*
     . hbcm3-3*fmb*fmc*hbcm2-fmb*hbcm3-3*fmb2*hbcm2+2*p1p3s2+2*p3p5s2
     . )+8*(fmb*hbcm*p2p3*p6p7-3*fmb*hbcm*p3p4*p6p7+fmb*hbcm*p3p6*
     . p4p7+fmb*hbcm*p3p7*p4p6+fmb*hbcm*p3p7*p5p6-hbcm2*p1p2*p6p7-
     . hbcm2*p1p4*p6p7+3*hbcm2*p1p5*p6p7+hbcm2*p2p5*p6p7+hbcm2*p4p5*
     . p6p7+hbcm2*p4p6*p5p7+hbcm2*p5p6*p5p7-p1p3*p2p3*p6p7+3*p1p3*
     . p3p4*p6p7-2*p1p3*p3p5*p6p7+2*p1p3*p3p6*p5p7-p1p3*p3p7*p4p6-
     . p1p3*p3p7*p5p6+p2p3*p3p5*p6p7-3*p3p4*p3p5*p6p7-2*p3p5*p3p6*
     . p5p7+p3p5*p3p7*p4p6+p3p5*p3p7*p5p6))+((p3p7*p3p6)*(-32*fb2*
     . ffmcfmb+16*fb2-8)+4*p6p7*(6*fb1*fmb*hbcm+6*fb1*fmc*hbcm-4*fb2*
     . ffmcfmb*hbcm2-2*fb2*hbcm2+5*hbcm2)+16*(-fb2*p1p3*p6p7+fb2*p2p3
     . *p6p7-3*fb2*p3p4*p6p7+fb2*p3p5*p6p7-2*fb2*p3p6*p5p7+fb2*p3p7*
     . p4p6+fb2*p3p7*p5p6))
      ans1=w1*(12*(p6p7*p3p4)*(-fmb*hbcm+fmc*hbcm+hbcm2)+(p3p7*p3p6*
     . p3p4)*(16*ffmcfmb-8)+8*(p4p7*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm+
     . fmc*hbcm)+4*p6p7*(2*ffmcfmb*fmc*hbcm3-3*fmb*fmc*hbcm2+fmc*
     . hbcm3-3*fmc2*hbcm2+6*p3p4s2)+8*(fmc*hbcm*p1p3*p6p7-fmc*hbcm*
     . p2p3*p6p7-fmc*hbcm*p3p5*p6p7+3*fmc*hbcm*p3p6*p5p7-fmc*hbcm*
     . p3p7*p4p6-fmc*hbcm*p3p7*p5p6-hbcm2*p1p4*p6p7+hbcm2*p2p4*p6p7+
     . hbcm2*p4p5*p6p7+hbcm2*p4p6*p4p7+hbcm2*p4p7*p5p6+p1p3*p3p4*p6p7
     . -p2p3*p3p4*p6p7-p3p4*p3p5*p6p7+2*p3p4*p3p6*p5p7-p3p4*p3p7*p4p6
     . -p3p4*p3p7*p5p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      b(16)=ccc*(w1*(12*p6p7*(fmb*fmc*hbcm-fmc*hbcm2-fmc2*hbcm)+4*(
     . p6p7*p3p4)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+8*(-hbcm*p1p4*
     . p6p7+hbcm*p2p4*p6p7+hbcm*p3p6*p4p7+hbcm*p4p5*p6p7+hbcm*p4p6*
     . p4p7+hbcm*p4p7*p5p6))+w15*(12*p6p7*(fmb*fmc*hbcm-fmb*hbcm2-
     . fmb2*hbcm)+4*(p6p7*p3p5)*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm)+4
     . *(p6p7*p1p3)*(-2*ffmcfmb*hbcm+3*fmb-3*fmc+3*hbcm)+8*(-hbcm*
     . p1p2*p6p7-hbcm*p1p4*p6p7+3*hbcm*p1p5*p6p7+hbcm*p2p5*p6p7+hbcm*
     . p3p6*p5p7+hbcm*p4p5*p6p7+hbcm*p4p6*p5p7+hbcm*p5p6*p5p7))+12*
     . p6p7*(2*fb1*hbcm-2*fb2*fmb+2*fb2*fmc+hbcm))
      do 200 n=1,16 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp30_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      ans2=w2*(8*(p4p7*p3p6)*(-fmb*fmc*hbcm+fmb2*hbcm)+8*(p5p7*p3p6*
     . p3p5)*(-2*ffmcfmb*hbcm+fmb-fmc+2*hbcm)+8*(p3p7*p3p6*p3p5)*(-2*
     . ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm+fmb-2*fmc)+8*(p5p7*p3p6
     . )*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+fmb*hbcm2+fmb2*hbcm+fmc*hbcm2+hbcm3)+8*(p3p7*p3p6)*(-
     . ffmcfmb*fmb*hbcm2-fmb*fmc*hbcm+fmb*hbcm2)+8*(2*hbcm*p1p5*p3p6*
     . p3p7+2*hbcm*p1p5*p3p6*p5p7+2*hbcm*p2p3*p3p6*p5p7-2*hbcm*p2p5*
     . p3p6*p3p7+hbcm*p3p5*p3p6*p4p7-2*hbcm*p3p6*p3p7*p4p5-2*hbcm*
     . p3p6*p4p5*p5p7))+(8*(p5p7*p3p6)*(-4*fb1*hbcm+2*fb2*fmb-2*fb2*
     . fmc-3*hbcm)+4*(p3p7*p3p6)*(12*fb1*ffmcfmb*hbcm-8*fb1*hbcm-8*
     . fb2*ffmcfmb*fmb+8*fb2*ffmcfmb*fmc+4*fb2*fmb-8*fb2*fmc-2*fmb+2*
     . fmc-3*hbcm)+16*fb1*hbcm*p3p6*p4p7)
      ans1=w5*(8*(p5p7*p3p6*p3p4)*(2*ffmcfmb*hbcm-fmb+fmc-2*hbcm)+8*(
     . p5p7*p3p6*p1p3)*(-2*ffmcfmb*hbcm+fmb-fmc+2*hbcm)+8*(p3p7*p3p6)
     . *(-ffmcfmb*fmc*hbcm2+fmc*hbcm2+fmc2*hbcm)+8*(p3p7*p3p6*p3p4)*(
     . 2*ffmcfmb*fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmb+2*fmc)+8*(p3p7*
     . p3p6*p1p3)*(-2*ffmcfmb*fmb+2*ffmcfmb*fmc-ffmcfmb*hbcm+fmb-2*
     . fmc)+8*(p4p7*p3p6)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*(fmc*hbcm2*p3p6*p5p7+2*hbcm*p1p2*
     . p3p6*p5p7+hbcm*p1p3*p3p6*p4p7-2*hbcm*p1p4*p3p6*p3p7+2*hbcm*
     . p1p5*p3p6*p5p7-2*hbcm*p2p4*p3p6*p5p7+2*hbcm*p2p5*p3p6*p4p7-
     . hbcm*p3p4*p3p6*p4p7-2*hbcm*p3p6*p4p5*p5p7))+ans2
      ans=ccc*ans1
      b(1)=ans
      ans2=(4*p5p7*(4*fb1*hbcm-2*fb2*fmb+2*fb2*fmc+3*hbcm)+2*p3p7*(-
     . 12*fb1*ffmcfmb*hbcm+8*fb1*hbcm+8*fb2*ffmcfmb*fmb-8*fb2*ffmcfmb
     . *fmc-4*fb2*fmb+8*fb2*fmc+2*fmb-2*fmc+3*hbcm)-8*fb1*hbcm*p4p7)
      ans1=w5*(4*(p5p7*p3p4)*(-2*ffmcfmb*hbcm+fmb-fmc+2*hbcm)+4*(p5p7
     . *p1p3)*(2*ffmcfmb*hbcm-fmb+fmc-2*hbcm)+4*p3p7*(ffmcfmb*fmc*
     . hbcm2-fmc*hbcm2-fmc2*hbcm)+4*(p3p7*p3p4)*(-2*ffmcfmb*fmb+2*
     . ffmcfmb*fmc-ffmcfmb*hbcm+fmb-2*fmc)+4*(p3p7*p1p3)*(2*ffmcfmb*
     . fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmb+2*fmc)+4*p4p7*(ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmc*hbcm2)+4*(-fmc*hbcm2
     . *p5p7-2*hbcm*p1p2*p5p7-hbcm*p1p3*p4p7+2*hbcm*p1p4*p3p7-2*hbcm*
     . p1p5*p5p7+2*hbcm*p2p4*p5p7-2*hbcm*p2p5*p4p7+hbcm*p3p4*p4p7+2*
     . hbcm*p4p5*p5p7))+w2*(4*p4p7*(fmb*fmc*hbcm-fmb2*hbcm)+4*(p5p7*
     . p3p5)*(2*ffmcfmb*hbcm-fmb+fmc-2*hbcm)+4*(p3p7*p3p5)*(2*ffmcfmb
     . *fmb-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmb+2*fmc)+4*p5p7*(-ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb*
     . hbcm2-fmb2*hbcm-fmc*hbcm2-hbcm3)+4*p3p7*(ffmcfmb*fmb*hbcm2+fmb
     . *fmc*hbcm-fmb*hbcm2)+4*(-2*hbcm*p1p5*p3p7-2*hbcm*p1p5*p5p7-2*
     . hbcm*p2p3*p5p7+2*hbcm*p2p5*p3p7-hbcm*p3p5*p4p7+2*hbcm*p3p7*
     . p4p5+2*hbcm*p4p5*p5p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans4=8*(ffmcfmb*hbcm2*p2p3*p4p7-ffmcfmb*hbcm2*p3p5*p4p7-2*fmc*
     . hbcm*p2p3*p5p7+fmc*hbcm*p2p5*p3p7-hbcm2*p1p2*p5p7-hbcm2*p1p4*
     . p4p7+hbcm2*p1p4*p5p7-hbcm2*p1p5*p5p7+hbcm2*p2p4*p5p7-hbcm2*
     . p2p5*p4p7+hbcm2*p4p5*p5p7-p1p3*p1p4*p3p7+p1p3*p2p3*p5p7-p1p3*
     . p2p5*p3p7-2*p1p3*p3p4*p5p7+p1p3*p3p5*p5p7+p1p4*p3p4*p3p7-p2p3*
     . p3p4*p5p7+p2p5*p3p4*p3p7-p3p4*p3p5*p5p7)
      ans3=4*(p5p7*p3p4)*(2*fmc*hbcm+hbcm2)+4*(p5p7*p1p3)*(-2*fmc*
     . hbcm-hbcm2)+4*(p4p7*p3p4)*(fmb*hbcm+fmc*hbcm)+4*(p4p7*p1p3)*(-
     . fmb*hbcm-fmc*hbcm)+(p3p7*p3p5*p3p4)*(16*ffmcfmb-8)+(p3p7*p3p4*
     . p2p3)*(-16*ffmcfmb+8)+(p3p7*p3p5*p1p3)*(-16*ffmcfmb+8)+(p3p7*
     . p2p3*p1p3)*(16*ffmcfmb-8)+8*(p3p7*p1p4)*(-ffmcfmb*hbcm2+fmb*
     . hbcm+fmc*hbcm+hbcm2)+4*p5p7*(2*ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2
     . -fmc2*hbcm2+2*p1p3s2+2*p3p4s2)+4*p4p7*(ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2+fmc2*hbcm2)+4*(
     . p3p7*p3p4)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2
     . -fmb*fmc-fmc2-hbcm2)+4*(p3p7*p1p3)*(ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmc2+hbcm2)+4*p3p7*(ffmcfmb*
     . fmb*fmc*hbcm2+ffmcfmb*fmc2*hbcm2-fmb*fmc*hbcm2-fmb*fmc2*hbcm-
     . fmc2*hbcm2-fmc3*hbcm)+ans4
      ans2=w5*ans3
      ans6=((p3p7*p3p5)*(-32*fb2*ffmcfmb+16*fb2-8)+(p3p7*p2p3)*(32*
     . fb2*ffmcfmb-16*fb2+8)+4*p5p7*(4*fb1*fmb*hbcm+4*fb1*fmc*hbcm-4*
     . fb2*ffmcfmb*hbcm2+2*fb2*hbcm2+5*hbcm2)+8*p4p7*(-fb1*fmb*hbcm-
     . fb1*fmc*hbcm)+2*p3p7*(-12*fb1*ffmcfmb*fmb*hbcm-12*fb1*ffmcfmb*
     . fmc*hbcm+8*fb1*fmb*hbcm+8*fb1*fmc*hbcm-8*fb2*ffmcfmb*hbcm2+4*
     . fb2*fmb*fmc+4*fb2*fmc2+4*fb2*hbcm2+3*fmb*hbcm+3*fmc*hbcm-2*
     . hbcm2)+16*(fb2*p1p3*p5p7-fb2*p1p4*p3p7+fb2*p2p3*p5p7-fb2*p2p5*
     . p3p7-fb2*p3p4*p5p7+fb2*p3p5*p5p7))
      ans5=w2*(8*(p4p5*p3p7)*(fmb*hbcm+fmc*hbcm)+4*(p4p7*p3p5)*(fmb*
     . hbcm-fmc*hbcm)+8*(p3p7*p1p5)*(-fmb*hbcm-fmc*hbcm)+(p3p7*p3p5*
     . p2p3)*(16*ffmcfmb-8)+4*(p5p7*p3p5)*(2*ffmcfmb*hbcm2-2*fmc*hbcm
     . -hbcm2)+8*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+4*p5p7*(
     . -ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-2*fmb*fmc*
     . hbcm2-2*fmb2*hbcm2-fmc*hbcm3-hbcm4+2*p3p5s2)+4*(p3p7*p3p5)*(
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2+fmb*fmc+fmc2
     . +hbcm2)+4*p3p7*(ffmcfmb*fmb*fmc*hbcm2+ffmcfmb*fmb2*hbcm2-4*
     . ffmcfmb*p3p5s2-fmb*fmc*hbcm2+fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmb2*
     . hbcm2+2*p3p5s2)+4*(2*fmb*hbcm*p1p3*p5p7-2*fmb*hbcm*p1p4*p3p7-2
     . *fmb*hbcm*p2p3*p4p7-2*fmb*hbcm*p3p4*p5p7+fmb*hbcm3*p4p7+2*fmc*
     . hbcm*p2p5*p3p7-2*hbcm2*p1p5*p5p7+2*hbcm2*p4p5*p5p7+2*p1p3*p3p5
     . *p5p7-2*p1p4*p3p5*p3p7+2*p2p3*p3p5*p5p7-2*p2p5*p3p5*p3p7-2*
     . p3p4*p3p5*p5p7))+ans6
      ans1=ans2+ans5
      ans=ccc*ans1
      b(5)=ans
      ans2=2*(-4*fb2*ffmcfmb*hbcm2+2*fb2*fmb2-2*fb2*fmc2+2*fb2*hbcm2+
     . 4*fb2*p1p3+4*fb2*p1p4+4*fb2*p2p3+8*fb2*p2p5-4*fb2*p3p4+4*fb2*
     . p3p5-3*fmb*hbcm-3*fmc*hbcm+5*hbcm2)
      ans1=w2*(4*p4p5*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*p2p5*(fmb*hbcm-fmc
     . *hbcm+hbcm2)+4*p1p5*(fmb*hbcm+fmc*hbcm-hbcm2)+2*p3p5*(-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2+hbcm2
     . )+2*(-2*ffmcfmb*fmb*hbcm3-fmb*fmc2*hbcm+2*fmb*hbcm*p1p3+2*fmb*
     . hbcm*p1p4+2*fmb*hbcm*p2p3-2*fmb*hbcm*p3p4+fmb*hbcm3+fmb3*hbcm+
     . 2*p1p3*p3p5+2*p1p4*p3p5+2*p2p3*p3p5+4*p2p5*p3p5-2*p3p4*p3p5+2*
     . p3p5s2))+w5*(4*p4p5*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*p2p4*(-fmb*
     . hbcm-fmc*hbcm+hbcm2)+4*p1p5*(fmb*hbcm+fmc*hbcm-hbcm2)+4*p1p2*(
     . fmb*hbcm+fmc*hbcm-hbcm2)+2*p3p4*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*
     . fmc*hbcm-2*fmb*hbcm-fmb2+fmc2+hbcm2)+2*p1p3*(-2*ffmcfmb*fmb*
     . hbcm-2*ffmcfmb*fmc*hbcm+2*fmb*hbcm+fmb2-fmc2-hbcm2)+2*(2*
     . ffmcfmb*fmc*hbcm3-fmb2*fmc*hbcm-2*fmc*hbcm*p1p4-2*fmc*hbcm*
     . p2p3-4*fmc*hbcm*p2p5-2*fmc*hbcm*p3p5-fmc*hbcm3+fmc3*hbcm+2*
     . p1p3*p1p4+2*p1p3*p2p3+4*p1p3*p2p5-4*p1p3*p3p4+2*p1p3*p3p5+2*
     . p1p3s2-2*p1p4*p3p4-2*p2p3*p3p4-4*p2p5*p3p4-2*p3p4*p3p5+2*
     . p3p4s2))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans4=2*(2*ffmcfmb*fmb*fmc*hbcm3-2*ffmcfmb*fmc2*hbcm3+2*fmb*
     . p1p3s2+2*fmb*p3p4s2-fmb2*fmc*hbcm2-4*fmc*hbcm2*p2p5-2*fmc*
     . hbcm2*p3p5-fmc*hbcm4-2*fmc*p1p3s2-2*fmc*p3p4s2+fmc3*hbcm2+4*
     . hbcm*p1p2*p2p3-4*hbcm*p1p2*p3p5-2*hbcm*p1p3*p1p4+4*hbcm*p1p3*
     . p2p5-8*hbcm*p1p4*p2p3+2*hbcm*p1p4*p3p4+4*hbcm*p1p5*p2p3-4*hbcm
     . *p1p5*p3p5-4*hbcm*p2p3*p2p4-4*hbcm*p2p3*p4p5+4*hbcm*p2p4*p3p5-
     . 4*hbcm*p2p5*p3p4+4*hbcm*p3p5*p4p5)
      ans3=8*(p3p4*p1p3)*(-fmb+fmc)+4*p4p5*(fmb*hbcm2-fmc*hbcm2+hbcm3
     . )+4*p2p4*(fmb*hbcm2-fmc*hbcm2+hbcm3)+4*p1p5*(-fmb*hbcm2+fmc*
     . hbcm2-hbcm3)+4*p1p2*(-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*p2p3*(-2*
     . fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+2*p3p4*(-2*ffmcfmb*hbcm3+2
     . *fmb*fmc*hbcm+2*fmb*hbcm2-fmb2*hbcm-2*fmc*hbcm2-fmc2*hbcm+
     . hbcm3)+4*p1p4*(2*ffmcfmb*hbcm3-fmc*hbcm2)+2*p1p3*(2*ffmcfmb*
     . hbcm3-2*fmb*fmc*hbcm-2*fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2+fmc2*
     . hbcm-hbcm3)+4*(p3p5*p3p4)*(-2*ffmcfmb*hbcm+hbcm)+4*(p3p4*p2p3)
     . *(2*ffmcfmb*hbcm-2*fmb+2*fmc-hbcm)+4*(p3p5*p1p3)*(2*ffmcfmb*
     . hbcm-hbcm)+4*(p2p3*p1p3)*(-2*ffmcfmb*hbcm+2*fmb-2*fmc+hbcm)+
     . ans4
      ans2=w5*ans3
      ans5=w2*(4*(p3p5*p3p4)*(-fmb+fmc)+4*(p3p5*p1p3)*(fmb-fmc)+4*
     . p4p5*(fmb*hbcm2-fmc*hbcm2+hbcm3)+4*p2p5*(-fmb*hbcm2-fmc*hbcm2+
     . hbcm3)+4*p1p5*(-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*p3p4*(fmb*fmc*
     . hbcm-fmb2*hbcm)+4*p2p3*(-2*fmb*fmc*hbcm+fmb*hbcm2+2*fmb2*hbcm)
     . +4*p1p3*(-fmb*fmc*hbcm+fmb2*hbcm)+2*p3p5*(2*ffmcfmb*hbcm3-2*
     . fmb*hbcm2-fmb2*hbcm+fmc2*hbcm-hbcm3)+4*(p3p5*p2p3)*(-2*ffmcfmb
     . *hbcm+2*fmb-2*fmc+hbcm)+2*(2*ffmcfmb*fmb*fmc*hbcm3-2*ffmcfmb*
     . fmb2*hbcm3+4*ffmcfmb*hbcm*p3p5s2+fmb*fmc2*hbcm2-2*fmb*hbcm2*
     . p1p4-fmb*hbcm4-fmb3*hbcm2-2*hbcm*p1p4*p3p5+4*hbcm*p1p5*p2p3-4*
     . hbcm*p1p5*p3p5-4*hbcm*p2p3*p2p5-4*hbcm*p2p3*p4p5+4*hbcm*p3p5*
     . p4p5-2*hbcm*p3p5s2))+(8*p3p4*(-fb2*fmb+fb2*fmc)+8*p1p3*(fb2*
     . fmb-fb2*fmc)+4*p3p5*(2*fb1*hbcm+3*hbcm)+4*p2p3*(-2*fb1*hbcm+4*
     . fb2*fmb-4*fb2*fmc-3*hbcm)+2*(2*fb1*fmb2*hbcm-2*fb1*fmc2*hbcm+4
     . *fb1*hbcm*p1p4+8*fb1*hbcm*p2p5+2*fb1*hbcm3-4*fb2*ffmcfmb*fmb*
     . hbcm2+4*fb2*ffmcfmb*fmc*hbcm2+5*fmb*hbcm2-5*fmc*hbcm2+3*hbcm3)
     . )
      ans1=ans2+ans5
      ans=ccc*ans1
      b(8)=ans
      b(11)=ccc*(w5*(2*p3p4*(3*fmb*hbcm+fmc*hbcm+hbcm2)+2*p1p3*(-3*
     . fmb*hbcm-fmc*hbcm-hbcm2)+2*(-2*ffmcfmb*fmc*hbcm3+3*fmb*fmc*
     . hbcm2-2*fmc*hbcm*p2p3+6*fmc*hbcm*p3p5+3*fmc*hbcm3+3*fmc2*hbcm2
     . +2*hbcm2*p1p2-6*hbcm2*p1p4+2*hbcm2*p1p5-2*hbcm2*p2p4-2*hbcm2*
     . p4p5+2*p1p3*p2p3+4*p1p3*p3p4-6*p1p3*p3p5-2*p1p3s2-2*p2p3*p3p4+
     . 6*p3p4*p3p5-2*p3p4s2))+w2*(6*p3p5*(-fmb*hbcm+fmc*hbcm-hbcm2)+2
     . *(2*ffmcfmb*fmb*hbcm3+3*fmb*fmc*hbcm2-2*fmb*hbcm*p1p3+2*fmb*
     . hbcm*p2p3+2*fmb*hbcm*p3p4-3*fmb*hbcm3+3*fmb2*hbcm2+2*hbcm2*
     . p1p5-2*hbcm2*p2p5-2*hbcm2*p4p5-2*p1p3*p3p5+2*p2p3*p3p5+2*p3p4*
     . p3p5-6*p3p5s2))+2*(-6*fb1*fmb*hbcm-6*fb1*fmc*hbcm+4*fb2*
     . ffmcfmb*hbcm2-6*fb2*hbcm2-4*fb2*p1p3+4*fb2*p2p3+4*fb2*p3p4-12*
     . fb2*p3p5-5*hbcm2))
      b(12)=ccc*(w5*(4*p4p7*(ffmcfmb*hbcm3-fmb*hbcm2+2*fmc*hbcm2-
     . hbcm3)+4*(p3p7*p3p4)*(-ffmcfmb*hbcm+fmb-2*fmc)+4*(p3p7*p1p3)*(
     . ffmcfmb*hbcm-fmb+2*fmc)+4*p3p7*(ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm
     . -fmc*hbcm2-2*fmc2*hbcm)+4*(2*fmc*hbcm2*p5p7-hbcm*p1p3*p4p7-3*
     . hbcm*p1p3*p5p7+4*hbcm*p1p4*p3p7+2*hbcm*p2p3*p4p7+hbcm*p3p4*
     . p4p7+3*hbcm*p3p4*p5p7-2*hbcm*p3p5*p4p7))+w2*(4*p5p7*(ffmcfmb*
     . hbcm3+fmc*hbcm2-2*hbcm3)+4*(p3p7*p3p5)*(ffmcfmb*hbcm-fmb+2*fmc
     . )+4*p3p7*(ffmcfmb*fmb*hbcm2+2*fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm
     . )+4*(-2*hbcm*p1p5*p3p7+2*hbcm*p2p5*p3p7-hbcm*p3p5*p4p7-hbcm*
     . p3p5*p5p7+2*hbcm*p3p7*p4p5))+(2*p3p7*(-12*fb1*ffmcfmb*hbcm+8*
     . fb1*hbcm-4*fb2*fmb+8*fb2*fmc+3*hbcm)+8*(-fb1*hbcm*p4p7-fb1*
     . hbcm*p5p7)))
      b(13)=ccc*(w2*(2*p3p5*(2*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm)+2*(3*
     . fmb*fmc*hbcm-3*fmb*hbcm2-3*fmb2*hbcm-2*hbcm*p1p5+2*hbcm*p2p5+2
     . *hbcm*p4p5))+w5*(2*p3p4*(-2*ffmcfmb*hbcm+3*fmb-3*fmc-hbcm)+2*
     . p1p3*(2*ffmcfmb*hbcm-3*fmb+3*fmc+hbcm)+2*(3*fmb*fmc*hbcm-3*fmc
     . *hbcm2-3*fmc2*hbcm-2*hbcm*p1p2+6*hbcm*p1p4-2*hbcm*p1p5+2*hbcm*
     . p2p4+2*hbcm*p4p5))+6*(2*fb1*hbcm-2*fb2*fmb+2*fb2*fmc+hbcm))
      b(14)=ccc*(w2*(4*p5p7*(ffmcfmb*hbcm2-fmb*hbcm+fmc*hbcm-2*hbcm2)
     . +4*(-2*ffmcfmb*p3p5*p3p7-fmb*hbcm*p3p7+fmb*hbcm*p4p7-2*p3p5*
     . p5p7))+w5*(4*p4p7*(-ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm-hbcm2)+4*(
     . -2*ffmcfmb*p1p3*p3p7+2*ffmcfmb*p3p4*p3p7+fmc*hbcm*p3p7+3*fmc*
     . hbcm*p5p7-2*p1p3*p5p7+2*p3p4*p5p7))+(p3p7*(-16*fb2*ffmcfmb-4)-
     . 16*fb2*p5p7))
      do 200 n=1,16 
         c(n,2)=c(n,2)+0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      subroutine amp3_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(14) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,14 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(1)=ccc*(w2*(16*(p5p7*p3p6*p3p5)*(-fmb+fmc)+16*(p3p7*p3p6*p3p5
     . )*(-fmb+fmc)+16*(-fmb*hbcm2*p3p6*p3p7-fmb*hbcm2*p3p6*p5p7))+(
     . 32*(p5p7*p3p6)*(2*fb1*hbcm-fb2*fmb+fb2*fmc)+32*(p3p7*p3p6)*(2*
     . fb1*hbcm-fb2*fmb+fb2*fmc)))
      b(2)=ccc*(w2*(8*(p5p7*p3p5)*(fmb-fmc)+8*(p3p7*p3p5)*(fmb-fmc)+8
     . *(fmb*hbcm2*p3p7+fmb*hbcm2*p5p7))+(16*p5p7*(-2*fb1*hbcm+fb2*
     . fmb-fb2*fmc)+16*p3p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc)))
      b(5)=ccc*(w2*(8*(p3p7*p3p5)*(fmb2-fmc2)+8*p5p7*(fmb*fmc*hbcm2+
     . fmb2*hbcm2-2*p3p5s2)+8*p3p7*(fmb*fmc*hbcm2+fmb2*hbcm2)-8*hbcm2
     . *p3p5*p5p7)+(16*p5p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm-fb2*hbcm2
     . )+16*p3p7*(-2*fb1*fmb*hbcm-2*fb1*fmc*hbcm+fb2*fmb2-fb2*fmc2)-
     . 32*fb2*p3p5*p5p7))
      b(7)=ccc*(w2*(4*p3p5*(-fmb2+fmc2-hbcm2)-8*p3p5s2)+8*(-fb2*fmb2+
     . fb2*fmc2-fb2*hbcm2-2*fb2*p3p5))
      b(8)=ccc*(w2*(8*(p3p5*p2p3)*(-fmb+fmc)+4*(-fmb*fmc2*hbcm2-2*fmb
     . *hbcm2*p2p3+2*fmb*hbcm2*p3p5+fmb*hbcm4+fmb3*hbcm2))+(16*p2p3*(
     . 2*fb1*hbcm-fb2*fmb+fb2*fmc)+16*(-fb1*fmb2*hbcm+fb1*fmc2*hbcm-2
     . *fb1*hbcm*p3p5-fb1*hbcm3)))
      b(11)=ccc*(4*w2*(-fmb*fmc*hbcm2-fmb2*hbcm2+hbcm2*p3p5+2*p3p5s2)
     . +8*(2*fb1*fmb*hbcm+2*fb1*fmc*hbcm+fb2*hbcm2+2*fb2*p3p5))
      b(12)=ccc*(w2*(8*(p3p7*p3p5)*(fmb-fmc)+8*fmb*hbcm2*p3p7)+16*
     . p3p7*(-2*fb1*hbcm+fb2*fmb-fb2*fmc))
      b(13)=ccc*(w2*(4*p3p5*(fmb-fmc)+4*fmb*hbcm2)+8*(-2*fb1*hbcm+fb2
     . *fmb-fb2*fmc))
      do 200 n=1,14 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
      end 
 
      subroutine amp29_3p0(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(16) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,16 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      ans4=4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+4*fmb*hbcm2-2*fmb2*hbcm-
     . fmc*hbcm2-2*hbcm3)+8*(p3p7*p3p6*p3p4)*(2*ffmcfmb**2*hbcm+2*
     . ffmcfmb*fmc+ffmcfmb*hbcm+fmb-2*fmc-3*hbcm)+4*p6p7*(-ffmcfmb**2
     . *fmc*hbcm4-3*ffmcfmb*fmb*fmc*hbcm3+3*ffmcfmb*fmc*hbcm4+3*
     . ffmcfmb*fmc2*hbcm3+4*fmb*fmc*hbcm3+3*fmb*fmc2*hbcm2-2*fmb2*fmc
     . *hbcm2-2*fmc*hbcm4-3*fmc2*hbcm3)+16*(fmc*hbcm2*p2p5*p6p7+hbcm*
     . p2p5*p3p4*p6p7)
      ans3=8*(p5p7*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p6p7*p1p4)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p6*p3p7*p3p5)*(ffmcfmb*
     . hbcm-hbcm)+16*(p4p6*p3p7*p3p4)*(2*ffmcfmb*hbcm+fmb-2*hbcm)+8*(
     . p5p7*p3p6*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p7*p3p6*p3p4)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p5*p3p4)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p4p6*p3p7*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(p6p7*p3p4*
     . p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p6*p3p7*p1p3)*(-ffmcfmb*
     . hbcm+hbcm)+8*(p5p7*p3p6)*(ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc*
     . hbcm2)+8*(p3p7*p3p6)*(3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*fmc2*hbcm+
     . fmb*fmc*hbcm-3*fmc*hbcm2-2*fmc2*hbcm)+16*(p6p7*p3p5)*(ffmcfmb*
     . fmc*hbcm2+fmb*fmc*hbcm-fmc*hbcm2)+16*(p6p7*p2p3)*(-ffmcfmb*fmc
     . *hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+8*(p4p6*p3p7)*(2*ffmcfmb*fmb*
     . hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+2*fmb*fmc*hbcm-fmb*
     . hbcm2-2*fmc*hbcm2-hbcm3)+8*(p4p7*p3p6)*(-2*ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+fmc*hbcm2)+ans4
      ans2=w1*ans3
      ans8=4*p6p7*(-ffmcfmb**2*fmb*hbcm4+ffmcfmb**2*hbcm5-ffmcfmb*fmb
     . *fmc*hbcm3+2*ffmcfmb*fmb*hbcm4-ffmcfmb*fmb2*hbcm3+ffmcfmb*fmc*
     . hbcm4+2*ffmcfmb*fmc2*hbcm3-ffmcfmb*hbcm5+2*fmb*fmc*hbcm3+2*fmb
     . *fmc2*hbcm2-fmb2*fmc*hbcm2-fmc*hbcm4-2*fmc2*hbcm3)+16*(ffmcfmb
     . *hbcm*p2p3*p3p6*p5p7-ffmcfmb*hbcm*p2p5*p3p6*p3p7-hbcm*p1p2*
     . p3p6*p5p7-hbcm*p1p3*p2p5*p6p7+2*hbcm*p2p3*p4p6*p5p7-hbcm*p2p4*
     . p3p6*p5p7+hbcm*p2p5*p3p4*p6p7+hbcm*p2p5*p3p6*p4p7-2*hbcm*p2p5*
     . p3p7*p4p6)
      ans7=8*(p4p7*p3p6)*(-2*ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+
     . ffmcfmb*hbcm3-fmb2*hbcm+hbcm3)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm3
     . -2*ffmcfmb*fmb*fmc*hbcm-ffmcfmb*fmb*hbcm2+ffmcfmb*fmb2*hbcm+3*
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*fmc2*hbcm-ffmcfmb*hbcm3+2*fmb*fmc*
     . hbcm-3*fmc*hbcm2-3*fmc2*hbcm)+8*(p6p7*p3p5)*(ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm
     . -fmc*hbcm2)+4*(p6p7*p3p4)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+4*fmb*hbcm2-2*
     . fmb2*hbcm-fmc*hbcm2-2*hbcm3)+8*(p6p7*p2p3)*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2)+4*(p6p7*p1p3)*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2
     . -ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm-4*fmb*hbcm2+2*
     . fmb2*hbcm+fmc*hbcm2+2*hbcm3)+8*(p3p7*p3p6*p3p4)*(2*ffmcfmb**2*
     . hbcm+2*ffmcfmb*fmc+3*ffmcfmb*hbcm+fmb-2*fmc-5*hbcm)+8*(p3p7*
     . p3p6*p1p3)*(-2*ffmcfmb**2*hbcm-2*ffmcfmb*fmc+ffmcfmb*hbcm-fmb+
     . 2*fmc+hbcm)+ans8
      ans6=16*(p5p7*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p7*
     . p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*p2p5)*(ffmcfmb*
     . hbcm3+fmc*hbcm2)+16*(p6p7*p1p4)*(-ffmcfmb*hbcm3-fmb*hbcm2+
     . hbcm3)+32*(p4p6*p3p7*p3p5)*(ffmcfmb*hbcm-hbcm)+16*(p4p6*p3p7*
     . p3p4)*(3*ffmcfmb*hbcm+fmb-3*hbcm)+8*(p5p7*p3p6*p3p4)*(ffmcfmb*
     . hbcm+fmb-hbcm)+8*(p4p7*p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+16*
     . (p6p7*p3p5*p3p4)*(ffmcfmb*hbcm+fmb-hbcm)+32*(p4p6*p3p7*p2p3)*(
     . -ffmcfmb*hbcm+hbcm)+16*(p6p7*p3p4*p2p3)*(-ffmcfmb*hbcm-fmb+
     . hbcm)+16*(p3p7*p3p6*p1p4)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7*
     . p1p3)*(-3*ffmcfmb*hbcm-fmb+3*hbcm)+8*(p5p7*p3p6*p1p3)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+8*(p4p7*p3p6*p1p3)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p6p7*p3p5*p1p3)*(-ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*
     . p2p3*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p6*p3p7)*(ffmcfmb**2*
     . hbcm3+2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-2*
     . fmb*hbcm2+2*fmb2*hbcm-fmc*hbcm2-2*hbcm3)+8*(p5p7*p3p6)*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+ffmcfmb
     . *hbcm3+2*fmb*fmc*hbcm-2*fmc*hbcm2)+ans7
      ans5=w5*ans6
      ans9=(8*(p4p6*p3p7)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+4*(p5p7*p3p6)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+4*(p4p7*p3p6)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+
     . hbcm)+8*(p6p7*p3p5)*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-
     . hbcm)+8*(p6p7*p2p3)*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+
     . hbcm)+2*p6p7*(4*fb1*ffmcfmb**2*hbcm3-12*fb1*ffmcfmb*hbcm3-4*
     . fb1*fmb*fmc*hbcm+8*fb1*fmb2*hbcm+8*fb1*hbcm3+12*fb2*ffmcfmb*
     . fmb*hbcm2-4*fb2*ffmcfmb*fmc*hbcm2-16*fb2*fmb*hbcm2+4*fb2*fmc*
     . hbcm2+2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-3*hbcm3)+4*(p3p7*
     . p3p6)*(8*fb1*ffmcfmb**2*hbcm-20*fb1*ffmcfmb*hbcm+12*fb1*hbcm-8
     . *fb2*ffmcfmb*fmc-4*fb2*fmb+8*fb2*fmc-8*ffmcfmb*hbcm-2*fmc+7*
     . hbcm)-32*fb1*hbcm*p2p5*p6p7)
      ans1=w8*(16*(p6p7*p2p5)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p3p7*p3p6*p2p5)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p2p4)*(
     . ffmcfmb*hbcm-hbcm)+16*p6p7*(-ffmcfmb*hbcm*p2p3s2-fmb*p2p3s2+
     . hbcm*p2p3s2)+16*(p4p6*p3p7*p2p3)*(-ffmcfmb*hbcm+fmb+hbcm)+8*(
     . p5p7*p3p6*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p4p7*p3p6*p2p3)*(-
     . ffmcfmb*hbcm-fmb+hbcm)+16*(p6p7*p3p5*p2p3)*(ffmcfmb*hbcm+fmb-
     . hbcm)+16*(p3p7*p3p6*p1p2)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p2p3)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb
     . *hbcm3-fmb*fmc*hbcm+2*fmb2*hbcm+fmc*hbcm2-2*hbcm3)+8*(p3p7*
     . p3p6*p2p3)*(2*ffmcfmb**2*hbcm+2*ffmcfmb*fmc-5*ffmcfmb*hbcm+fmb
     . -2*fmc+3*hbcm)-16*hbcm*p2p3*p2p5*p6p7)+ans2+ans5+ans9
      ans=ccc*ans1
      b(1)=ans
      ans3=(6*p5p7*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+2*
     . p4p7*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+2*p3p7*(-
     . 4*fb1*ffmcfmb**2*hbcm+16*fb1*ffmcfmb*hbcm-12*fb1*hbcm-16*fb2*
     . ffmcfmb*fmb+12*fb2*ffmcfmb*fmc+12*fb2*fmb-12*fb2*fmc+4*ffmcfmb
     . *hbcm-4*fmb+3*fmc-8*hbcm))
      ans2=w5*(12*(p5p7*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+4*(p4p7*p3p4)*
     . (ffmcfmb*hbcm+fmb-hbcm)+24*(p3p7*p1p4)*(ffmcfmb*hbcm-hbcm)+12*
     . (p5p7*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+4*(p4p7*p1p3)*(-ffmcfmb*
     . hbcm-fmb+hbcm)+12*p5p7*(-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*
     . hbcm2)+4*p4p7*(ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3+fmb2*hbcm-hbcm3
     . )+4*(p3p7*p3p4)*(-ffmcfmb**2*hbcm+4*ffmcfmb*fmb-3*ffmcfmb*fmc-
     . 4*ffmcfmb*hbcm-3*fmb+3*fmc+5*hbcm)+4*(p3p7*p1p3)*(ffmcfmb**2*
     . hbcm-4*ffmcfmb*fmb+3*ffmcfmb*fmc+3*fmb-3*fmc-hbcm)+4*p3p7*(
     . ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*fmc*hbcm+
     . ffmcfmb*fmb2*hbcm-3*ffmcfmb*fmc*hbcm2-5*ffmcfmb*fmc2*hbcm+
     . ffmcfmb*hbcm3-3*fmb*fmc*hbcm+3*fmc*hbcm2+5*fmc2*hbcm)+8*(
     . ffmcfmb*hbcm*p2p3*p5p7-ffmcfmb*hbcm*p2p5*p3p7+hbcm*p1p2*p5p7+
     . hbcm*p2p4*p5p7-hbcm*p2p5*p4p7))+ans3
      ans1=w8*(24*(p3p7*p2p5)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*p2p4)*(-
     . ffmcfmb*hbcm+hbcm)+12*(p5p7*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+4*(
     . p4p7*p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+8*(p3p7*p1p2)*(-ffmcfmb*
     . hbcm+hbcm)+4*(p3p7*p2p3)*(-ffmcfmb**2*hbcm+4*ffmcfmb*fmb-3*
     . ffmcfmb*fmc+4*ffmcfmb*hbcm-3*fmb+3*fmc-3*hbcm))+w1*(12*(p5p7*
     . p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+4*(p4p7*p3p4)*(ffmcfmb*hbcm+fmb
     . -hbcm)+16*(p3p7*p1p4)*(ffmcfmb*hbcm-hbcm)+12*p5p7*(-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+4*p4p7*(ffmcfmb*fmc*hbcm2+
     . fmb*fmc*hbcm-fmc*hbcm2)+4*(p3p7*p3p4)*(-ffmcfmb**2*hbcm+4*
     . ffmcfmb*fmb-3*ffmcfmb*fmc-2*ffmcfmb*hbcm-3*fmb+3*fmc+3*hbcm)+4
     . *p3p7*(ffmcfmb**2*fmc*hbcm2+4*ffmcfmb*fmb*fmc*hbcm-4*ffmcfmb*
     . fmc*hbcm2-5*ffmcfmb*fmc2*hbcm-3*fmb*fmc*hbcm+3*fmc*hbcm2+5*
     . fmc2*hbcm))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=w5*(16*(p4p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p3p6*p3p4)*
     . (-ffmcfmb*hbcm-fmb+hbcm)+16*(p4p6*p1p3)*(ffmcfmb*hbcm+fmb-hbcm
     . )+8*(p3p6*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+4*p4p6*(-ffmcfmb**2*
     . hbcm3-5*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+2*fmb*hbcm2-4*fmb2*hbcm+fmc*hbcm2+2*hbcm3)+4*p3p6*(-2
     . *ffmcfmb**2*fmb*hbcm2-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmb2*hbcm-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2)+16*(-ffmcfmb*hbcm*p1p2*p3p6+ffmcfmb*hbcm*p2p3*p3p6
     . +ffmcfmb*hbcm*p2p4*p3p6+2*ffmcfmb*hbcm*p2p5*p3p6-hbcm*p1p2*
     . p4p6+hbcm*p2p3*p4p6+hbcm*p2p4*p4p6+3*hbcm*p2p5*p4p6))+(8*p4p6*
     . (4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+4*p3p6*(4*fb1*
     . ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm))
      ans1=w8*(16*(p4p6*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+8*(p3p6*p2p3)*
     . (-ffmcfmb*hbcm-fmb+hbcm))+w1*(16*(p4p6*p3p4)*(-ffmcfmb*hbcm-
     . fmb+hbcm)+8*(p3p6*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+4*p4p6*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-4*ffmcfmb*fmc*hbcm2-4*fmb
     . *fmc*hbcm-fmb2*hbcm+4*fmc*hbcm2+hbcm3)+8*p3p6*(-ffmcfmb**2*fmc
     . *hbcm2-ffmcfmb*fmb*fmc*hbcm-fmb*fmc*hbcm+fmc*hbcm2)+8*(-
     . ffmcfmb*hbcm*p2p3*p4p6+2*ffmcfmb*hbcm*p2p4*p3p6-hbcm*p1p2*p4p6
     . +hbcm*p2p4*p4p6+hbcm*p2p5*p4p6))+ans2
      ans=ccc*ans1
      b(3)=ans
      ans5=4*(p6p7*p1p3)*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc-2*fmb2+fmc*hbcm+2*hbcm2
     . )+4*p6p7*(ffmcfmb**2*fmb*hbcm3+ffmcfmb**2*hbcm4+ffmcfmb*fmb*
     . fmc*hbcm2+ffmcfmb*fmb2*hbcm2+ffmcfmb*fmc*hbcm3+2*ffmcfmb*fmc2*
     . hbcm2-ffmcfmb*hbcm4+2*fmb*fmc2*hbcm+fmb2*fmc*hbcm-fmc*hbcm3-2*
     . fmc2*hbcm2)+16*(p1p2*p1p3*p6p7-p1p2*p3p4*p6p7-p1p3*p2p3*p6p7-
     . p1p3*p2p4*p6p7+p2p3*p3p4*p6p7+p2p4*p3p4*p6p7)
      ans4=(p4p6*p3p7*p3p4)*(-16*ffmcfmb+16)+(p4p6*p3p7*p1p3)*(16*
     . ffmcfmb-16)+32*(p5p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*
     . (p4p7*p4p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p2p4)*(
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p3)*(ffmcfmb*hbcm2+fmc*hbcm)
     . +16*(p6p7*p1p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p1p2)*
     . (-ffmcfmb*hbcm2-fmc*hbcm)+(p3p7*p3p6*p3p4)*(16*ffmcfmb**2-32*
     . ffmcfmb+16)+(p3p7*p3p6*p1p3)*(-16*ffmcfmb**2+32*ffmcfmb-16)+8*
     . (p4p6*p3p7)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2-4*fmb*hbcm+fmc*hbcm+2*hbcm2)+16*(p5p7*p3p6)
     . *(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+16*(p4p7*
     . p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2)+8*(
     . p3p7*p3p6)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7*p3p4)*(ffmcfmb**2*hbcm2+3
     . *ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc+2*
     . fmb2-fmc*hbcm-2*hbcm2)+ans5
      ans3=w5*ans4
      ans6=((p4p6*p3p7)*(32*fb2*ffmcfmb-32*fb2+8)+(p3p7*p3p6)*(-32*
     . fb2*ffmcfmb**2+64*fb2*ffmcfmb-32*fb2-16*ffmcfmb+16)+2*p6p7*(-4
     . *fb1*ffmcfmb*fmb*hbcm-4*fb1*ffmcfmb*fmc*hbcm+4*fb1*fmc*hbcm+4*
     . fb2*ffmcfmb**2*hbcm2-12*fb2*ffmcfmb*hbcm2-4*fb2*fmb*fmc-8*fb2*
     . fmb2+8*fb2*hbcm2-fmb*hbcm-fmc*hbcm+hbcm2)+32*(fb2*p1p2*p6p7-
     . fb2*p2p3*p6p7-fb2*p2p4*p6p7))
      ans2=w1*((p4p6*p3p7*p3p4)*(-16*ffmcfmb+16)+8*(p5p7*p4p6)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p4p6)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+16*(p6p7*p1p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +16*(p3p7*p3p6)*(-ffmcfmb*fmc*hbcm+fmc*hbcm)+8*(p4p6*p3p7)*(2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*hbcm+2*
     . fmc*hbcm+hbcm2)+(p3p7*p3p6*p3p4)*(16*ffmcfmb**2-32*ffmcfmb+16)
     . +16*(p4p7*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2)+4*(p6p7*p3p4)*(ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc+2*fmb2-fmc*hbcm-2*hbcm2
     . )+4*p6p7*(-ffmcfmb**2*fmc*hbcm3+ffmcfmb*fmb*fmc*hbcm2+3*
     . ffmcfmb*fmc*hbcm3+3*ffmcfmb*fmc2*hbcm2+3*fmb*fmc2*hbcm+2*fmb2*
     . fmc*hbcm-2*fmc*hbcm3-3*fmc2*hbcm2)+16*(-fmc*hbcm*p1p2*p6p7+fmc
     . *hbcm*p2p3*p6p7+fmc*hbcm*p2p4*p6p7-p1p2*p3p4*p6p7+p2p3*p3p4*
     . p6p7+p2p4*p3p4*p6p7))+ans3+ans6
      ans1=w8*((p4p6*p3p7*p2p3)*(-16*ffmcfmb+16)+16*(p6p7*p2p4)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p1p2)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+(p3p7*p3p6*p2p3)*(16*ffmcfmb**2-32*ffmcfmb+16)
     . +4*(p6p7*p2p3)*(ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm+ffmcfmb*hbcm2+fmb*fmc+2*fmb2-fmc*hbcm-2*hbcm2)+16*(-
     . p1p2*p2p3*p6p7+p2p3*p2p4*p6p7+p2p3s2*p6p7))+ans2
      ans=ccc*ans1
      b(4)=ans
      ans5=4*(p3p7*p1p3)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm-3*hbcm2)+
     . 4*p3p7*(-ffmcfmb**2*fmb*hbcm3-ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*
     . fmc*hbcm2+ffmcfmb*fmb2*hbcm2-3*ffmcfmb*fmc*hbcm3-5*ffmcfmb*
     . fmc2*hbcm2+ffmcfmb*hbcm4-2*ffmcfmb*p1p3s2+2*ffmcfmb*p3p4s2+3*
     . fmb*fmc*hbcm2+3*fmc*hbcm3+5*fmc2*hbcm2+2*p1p3s2-2*p3p4s2)+8*(
     . ffmcfmb*hbcm2*p2p3*p5p7-ffmcfmb*hbcm2*p2p5*p3p7+hbcm2*p1p2*
     . p5p7+hbcm2*p2p4*p5p7-hbcm2*p2p5*p4p7)
      ans4=(p3p7*p3p5*p3p4)*(-24*ffmcfmb+24)+(p3p7*p3p4*p2p3)*(24*
     . ffmcfmb-24)+(p3p7*p3p5*p1p3)*(24*ffmcfmb-24)+(p3p7*p2p3*p1p3)*
     . (-24*ffmcfmb+24)+8*(p4p7*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +20*(p5p7*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p7*p3p4)*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p2p3)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+24*(p3p7*p1p4)*(ffmcfmb*hbcm2-hbcm2)+4*(p5p7*
     . p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p7*p1p3)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+4*p4p7*(ffmcfmb*fmb*hbcm3-ffmcfmb*hbcm4-
     . 2*fmb*hbcm3+fmb2*hbcm2+hbcm4)+4*p5p7*(2*ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+3*fmb*
     . fmc*hbcm2-3*fmc*hbcm3)+8*(p3p7*p3p5)*(-ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+3*fmc*hbcm)+
     . 4*(p3p7*p3p4)*(-ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-3*ffmcfmb*
     . fmc*hbcm-2*ffmcfmb*hbcm2+5*fmb*hbcm+3*fmc*hbcm+3*hbcm2)+8*(
     . p3p7*p2p3)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*
     . hbcm-ffmcfmb*hbcm2-3*fmc*hbcm)+ans5
      ans3=w5*ans4
      ans6=((p3p7*p3p5)*(48*fb2*ffmcfmb-48*fb2+12)+(p3p7*p3p4)*(-16*
     . fb2*ffmcfmb+16*fb2-4)+(p3p7*p2p3)*(-48*fb2*ffmcfmb+48*fb2-12)+
     . (p3p7*p1p3)*(-16*fb2*ffmcfmb+16*fb2-4)+6*p5p7*(-4*fb1*fmb*hbcm
     . -4*fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+2*p4p7*(4*fb1*fmb*hbcm
     . +4*fb2*ffmcfmb*hbcm2-4*fb2*hbcm2-hbcm2)+2*p3p7*(16*fb1*ffmcfmb
     . *fmb*hbcm+12*fb1*ffmcfmb*fmc*hbcm-12*fb1*fmb*hbcm-12*fb1*fmc*
     . hbcm-4*fb2*ffmcfmb**2*hbcm2+16*fb2*ffmcfmb*hbcm2-12*fb2*hbcm2+
     . 2*ffmcfmb*hbcm2-8*fmb*hbcm-3*fmc*hbcm-2*hbcm2))
      ans2=w1*((p3p7*p3p5*p3p4)*(-24*ffmcfmb+24)+(p3p7*p3p4*p2p3)*(24
     . *ffmcfmb-24)+(p3p7*p3p4*p1p3)*(8*ffmcfmb-8)+12*(p5p7*p3p4)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p7*p3p4)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+16*(p3p7*p1p4)*(ffmcfmb*hbcm2-hbcm2)+12*p5p7*(
     . ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmc*hbcm3)+4*p4p7*(-ffmcfmb*
     . fmc*hbcm3-fmb*fmc*hbcm2+fmc*hbcm3)+24*(p3p7*p3p5)*(-ffmcfmb*
     . fmc*hbcm+fmc*hbcm)+24*(p3p7*p2p3)*(ffmcfmb*fmc*hbcm-fmc*hbcm)+
     . 8*(p3p7*p1p3)*(ffmcfmb*fmc*hbcm-fmc*hbcm)+4*(p3p7*p3p4)*(-
     . ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-2*ffmcfmb
     . *hbcm2+3*fmb*hbcm+fmc*hbcm+3*hbcm2)+4*p3p7*(ffmcfmb**2*fmc*
     . hbcm3-4*ffmcfmb*fmb*fmc*hbcm2-4*ffmcfmb*fmc*hbcm3-5*ffmcfmb*
     . fmc2*hbcm2+2*ffmcfmb*p3p4s2+3*fmb*fmc*hbcm2+3*fmc*hbcm3+5*fmc2
     . *hbcm2-2*p3p4s2))+ans3+ans6
      ans1=w8*(24*p3p7*(ffmcfmb*p2p3s2-p2p3s2)+(p3p7*p3p5*p2p3)*(-24*
     . ffmcfmb+24)+(p3p7*p3p4*p2p3)*(8*ffmcfmb-8)+(p3p7*p2p3*p1p3)*(8
     . *ffmcfmb-8)+24*(p3p7*p2p5)*(ffmcfmb*hbcm2-hbcm2)+8*(p3p7*p2p4)
     . *(-ffmcfmb*hbcm2+hbcm2)+12*(p5p7*p2p3)*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+4*(p4p7*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p3p7*p1p2)*(-ffmcfmb*hbcm2+hbcm2)+4*(p3p7*p2p3)*(ffmcfmb**2*
     . hbcm2+4*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm-4*ffmcfmb*hbcm2-3*
     . fmb*hbcm-3*fmc*hbcm+3*hbcm2))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans5=16*(-ffmcfmb*hbcm2*p1p2*p3p6+ffmcfmb*hbcm2*p2p4*p3p6+
     . ffmcfmb*p1p3*p2p3*p3p6-ffmcfmb*p2p3*p3p4*p3p6-hbcm2*p1p2*p4p6+
     . hbcm2*p2p4*p4p6+2*hbcm2*p2p5*p4p6+p1p3*p2p3*p4p6-p1p3*p2p5*
     . p3p6-p2p3*p3p4*p4p6+p2p5*p3p4*p3p6)
      ans4=32*(p4p6*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6*
     . p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p3p6*p2p5)*(3*ffmcfmb*
     . hbcm2+fmc*hbcm)+16*(p4p6*p2p3)*(-2*ffmcfmb*hbcm2-fmb*hbcm+2*
     . hbcm2)+16*(p3p6*p1p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6*
     . p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p4p6*(ffmcfmb**2*hbcm4
     . -ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+3*ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2+6*fmb*hbcm3-2*fmb2*hbcm2-fmc*hbcm3-4*hbcm4)+16*(p3p6*
     . p3p5)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+4*(
     . p3p6*p3p4)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm
     . -ffmcfmb*hbcm2-fmb*fmc+2*fmb*hbcm-2*fmb2+fmc*hbcm)+16*(p3p6*
     . p2p3)*(-ffmcfmb**2*hbcm2+ffmcfmb*hbcm2)+4*(p3p6*p1p3)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2+fmb*fmc-2*fmb*hbcm+2*fmb2-fmc*hbcm)+4*p3p6*(-ffmcfmb**2*
     . fmb*hbcm3+2*ffmcfmb**2*hbcm4-ffmcfmb*fmb*fmc*hbcm2+3*ffmcfmb*
     . fmb*hbcm3-ffmcfmb*fmb2*hbcm2-2*ffmcfmb*fmc2*hbcm2-2*ffmcfmb*
     . hbcm4+fmb*fmc*hbcm2-2*fmb*fmc2*hbcm-fmb2*fmc*hbcm+2*fmc2*hbcm2
     . )+ans5
      ans3=w5*ans4
      ans6=((p3p6*p2p3)*(32*fb2*ffmcfmb+8)+4*p4p6*(-4*fb1*fmb*hbcm-4*
     . fb2*ffmcfmb*hbcm2+4*fb2*hbcm2+hbcm2)+2*p3p6*(12*fb1*ffmcfmb*
     . fmb*hbcm+4*fb1*ffmcfmb*fmc*hbcm-8*fb1*fmb*hbcm-4*fb1*fmc*hbcm+
     . 4*fb2*ffmcfmb**2*hbcm2-4*fb2*ffmcfmb*hbcm2+4*fb2*fmb*fmc+8*fb2
     . *fmb2+3*fmb*hbcm+fmc*hbcm-hbcm2)+32*(fb2*p2p3*p4p6-fb2*p2p5*
     . p3p6))
      ans2=w1*(8*(p4p6*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p6*
     . p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p6*p2p3)*(-2*ffmcfmb
     . *hbcm2-fmb*hbcm-2*fmc*hbcm+hbcm2)+16*(p3p6*p1p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*(p4p6*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+4*p4p6*(-ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3+2*ffmcfmb
     . *fmc*hbcm3+2*ffmcfmb*hbcm4+2*fmb*fmc*hbcm2+2*fmb*hbcm3-fmb2*
     . hbcm2-2*fmc*hbcm3-hbcm4)+4*(p3p6*p3p4)*(ffmcfmb**2*hbcm2-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmb*fmc+2*fmb*
     . hbcm-2*fmb2+fmc*hbcm)+4*p3p6*(ffmcfmb**2*fmc*hbcm3-ffmcfmb*fmb
     . *fmc*hbcm2-ffmcfmb*fmc*hbcm3-3*ffmcfmb*fmc2*hbcm2+2*fmb*fmc*
     . hbcm2-3*fmb*fmc2*hbcm-2*fmb2*fmc*hbcm+3*fmc2*hbcm2)+8*(-2*
     . ffmcfmb*fmc*hbcm*p2p3*p3p6+2*ffmcfmb*hbcm2*p2p4*p3p6-2*ffmcfmb
     . *p2p3*p3p4*p3p6+2*fmc*hbcm*p2p5*p3p6-hbcm2*p1p2*p4p6+hbcm2*
     . p2p4*p4p6+hbcm2*p2p5*p4p6-2*p2p3*p3p4*p4p6+2*p2p5*p3p4*p3p6))+
     . ans3+ans6
      ans1=w8*(16*(p3p6*p2p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+24*(p4p6
     . *p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p3p6*p2p3)*(-3*
     . ffmcfmb**2*hbcm2-5*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+3*ffmcfmb
     . *hbcm2-fmb*fmc+2*fmb*hbcm-2*fmb2+fmc*hbcm)+16*(-ffmcfmb*p2p3s2
     . *p3p6+p2p3*p2p5*p3p6-p2p3s2*p4p6))+ans2
      ans=ccc*ans1
      b(6)=ans
      ans2=w5*(4*p2p4*(-fmb*hbcm+hbcm2)+4*p1p2*(-fmb*hbcm+hbcm2)+12*
     . p1p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p3*(-ffmcfmb*fmb*hbcm
     . +ffmcfmb*hbcm2)+2*p3p4*(2*ffmcfmb**2*hbcm2+5*ffmcfmb*fmb*hbcm+
     . 3*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+3*fmb*fmc+2*fmb*hbcm+3*fmb2
     . -3*fmc*hbcm-5*hbcm2)+2*p1p3*(-2*ffmcfmb**2*hbcm2-5*ffmcfmb*fmb
     . *hbcm-3*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-3*fmb*fmc+2*fmb*hbcm-3*
     . fmb2+3*fmc*hbcm+hbcm2)+2*(2*ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*fmc
     . *hbcm2+2*ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3+5*ffmcfmb*fmc2*
     . hbcm2-2*ffmcfmb*hbcm4-2*ffmcfmb*p1p3*p2p3+2*ffmcfmb*p2p3*p3p4+
     . 5*fmb*fmc2*hbcm+3*fmb2*fmc*hbcm-6*fmc*hbcm*p2p5-3*fmc*hbcm3-5*
     . fmc2*hbcm2-2*p1p2*p1p3+2*p1p2*p3p4-2*p1p3*p2p4+6*p1p3*p2p5+2*
     . p2p4*p3p4-6*p2p5*p3p4))+(p2p3*(-8*fb2*ffmcfmb-2)+(-12*fb1*
     . ffmcfmb*fmb*hbcm-12*fb1*ffmcfmb*fmc*hbcm+12*fb1*fmc*hbcm-12*
     . fb2*ffmcfmb*hbcm2-12*fb2*fmb*fmc-12*fb2*fmb2+12*fb2*hbcm2-8*
     . fb2*p1p2-8*fb2*p2p4+24*fb2*p2p5-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2)
     . )
      ans1=w1*(8*p1p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+2*p3p4*(2*
     . ffmcfmb**2*hbcm2+5*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm+ffmcfmb
     . *hbcm2+3*fmb*fmc+3*fmb2-3*fmc*hbcm-3*hbcm2)+2*(3*ffmcfmb*fmb*
     . fmc*hbcm2+2*ffmcfmb*fmc*hbcm*p2p3+3*ffmcfmb*fmc*hbcm3+5*
     . ffmcfmb*fmc2*hbcm2+2*ffmcfmb*p2p3*p3p4+5*fmb*fmc2*hbcm+3*fmb2*
     . fmc*hbcm+2*fmc*hbcm*p1p2+2*fmc*hbcm*p2p4-6*fmc*hbcm*p2p5-3*fmc
     . *hbcm3-5*fmc2*hbcm2+2*p1p2*p3p4+2*p2p4*p3p4-6*p2p5*p3p4))+w8*(
     . 12*p2p5*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p2p4*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+4*p1p2*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+2*p2p3*(
     . 2*ffmcfmb**2*hbcm2+5*ffmcfmb*fmb*hbcm+3*ffmcfmb*fmc*hbcm-5*
     . ffmcfmb*hbcm2+3*fmb*fmc-6*fmb*hbcm+3*fmb2-3*fmc*hbcm+3*hbcm2)+
     . 4*(ffmcfmb*p2p3s2+p1p2*p2p3+p2p3*p2p4-3*p2p3*p2p5))+ans2
      ans=ccc*ans1
      b(7)=ans
      ans5=2*(2*ffmcfmb**2*fmb*hbcm4+3*ffmcfmb*fmb*fmc*hbcm3-2*
     . ffmcfmb*fmb*hbcm4+2*ffmcfmb*fmb2*hbcm3-3*ffmcfmb*fmc*hbcm4-5*
     . ffmcfmb*fmc2*hbcm3-2*ffmcfmb*hbcm*p1p3s2+4*ffmcfmb*hbcm*p2p3*
     . p3p5-4*ffmcfmb*hbcm*p2p3s2+2*ffmcfmb*hbcm*p3p4s2-6*fmb*fmc*
     . hbcm3-5*fmb*fmc2*hbcm2-2*fmb*p1p3s2+2*fmb*p3p4s2+3*fmb2*fmc*
     . hbcm2+3*fmc*hbcm4+5*fmc2*hbcm3-2*hbcm*p1p2*p1p3-4*hbcm*p1p2*
     . p2p3+2*hbcm*p1p2*p3p4+4*hbcm*p1p2*p3p5-2*hbcm*p1p3*p2p4+2*hbcm
     . *p1p3*p2p5+2*hbcm*p1p3s2-4*hbcm*p2p3*p2p4+2*hbcm*p2p4*p3p4+4*
     . hbcm*p2p4*p3p5-10*hbcm*p2p5*p3p4-2*hbcm*p3p4s2)
      ans4=4*p2p4*(fmb*hbcm2+hbcm3)+4*p1p2*(fmb*hbcm2+hbcm3)+4*p2p5*(
     . -2*ffmcfmb*hbcm3-3*fmc*hbcm2)+12*p1p4*(ffmcfmb*hbcm3+fmb*hbcm2
     . -hbcm3)+12*(p3p5*p3p4)*(-ffmcfmb*hbcm-fmb+hbcm)+4*(p3p4*p2p3)*
     . (4*ffmcfmb*hbcm+3*fmb-3*hbcm)+12*(p3p5*p1p3)*(ffmcfmb*hbcm+fmb
     . -hbcm)+4*(p2p3*p1p3)*(-4*ffmcfmb*hbcm-3*fmb+3*hbcm)+12*p3p5*(-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+2*p3p4*(5*ffmcfmb*
     . fmb*hbcm2-3*ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-3*fmb*fmc*hbcm-8
     . *fmb*hbcm2+5*fmb2*hbcm+3*fmc*hbcm2+3*hbcm3)+4*p2p3*(ffmcfmb*
     . fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*
     . fmc*hbcm2)+2*p1p3*(-ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2+3*
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm+4*fmb*hbcm2-fmb2*hbcm-3*fmc*hbcm2
     . -3*hbcm3)+ans5
      ans3=w5*ans4
      ans6=(6*p3p5*(4*fb1*ffmcfmb*hbcm-4*fb1*hbcm+4*fb2*fmb+hbcm)+2*
     . p3p4*(-4*fb1*ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+4*p2p3*(-
     . 8*fb1*ffmcfmb*hbcm+6*fb1*hbcm-6*fb2*fmb+hbcm)+2*p1p3*(-4*fb1*
     . ffmcfmb*hbcm+4*fb1*hbcm-4*fb2*fmb-hbcm)+(-8*fb1*ffmcfmb**2*
     . hbcm3+20*fb1*ffmcfmb*hbcm3+12*fb1*fmb*fmc*hbcm-12*fb1*fmb2*
     . hbcm-8*fb1*hbcm*p1p2-8*fb1*hbcm*p2p4+24*fb1*hbcm*p2p5-12*fb1*
     . hbcm3-20*fb2*ffmcfmb*fmb*hbcm2+12*fb2*ffmcfmb*fmc*hbcm2+24*fb2
     . *fmb*hbcm2-12*fb2*fmc*hbcm2+2*ffmcfmb*hbcm3+7*fmb*hbcm2-3*fmc*
     . hbcm2-hbcm3))
      ans2=w1*(8*p1p4*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+12*(p3p5*p3p4)*
     . (-ffmcfmb*hbcm-fmb+hbcm)+4*(p3p4*p2p3)*(4*ffmcfmb*hbcm+3*fmb-3
     . *hbcm)+4*(p3p4*p1p3)*(ffmcfmb*hbcm+fmb-hbcm)+12*p3p5*(-ffmcfmb
     . *fmc*hbcm2-fmb*fmc*hbcm+fmc*hbcm2)+4*p2p3*(4*ffmcfmb*fmc*hbcm2
     . +3*fmb*fmc*hbcm-3*fmc*hbcm2)+4*p1p3*(ffmcfmb*fmc*hbcm2+fmb*fmc
     . *hbcm-fmc*hbcm2)+2*p3p4*(3*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2
     . -3*ffmcfmb*hbcm3-fmb*fmc*hbcm-6*fmb*hbcm2+3*fmb2*hbcm+fmc*
     . hbcm2+3*hbcm3)+2*(2*ffmcfmb**2*fmc*hbcm4+5*ffmcfmb*fmb*fmc*
     . hbcm3-5*ffmcfmb*fmc*hbcm4-5*ffmcfmb*fmc2*hbcm3+2*ffmcfmb*hbcm*
     . p3p4s2-6*fmb*fmc*hbcm3-5*fmb*fmc2*hbcm2+2*fmb*p3p4s2+3*fmb2*
     . fmc*hbcm2+2*fmc*hbcm2*p1p2+2*fmc*hbcm2*p2p4-6*fmc*hbcm2*p2p5+3
     . *fmc*hbcm4+5*fmc2*hbcm3+2*hbcm*p1p2*p3p4+2*hbcm*p2p4*p3p4-6*
     . hbcm*p2p5*p3p4-2*hbcm*p3p4s2))+ans3+ans6
      ans1=w8*(12*p2p5*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*p2p4*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*p1p2*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+12*(p3p5*p2p3)*(-ffmcfmb*hbcm-fmb+hbcm)+4*(p3p4*
     . p2p3)*(ffmcfmb*hbcm+fmb-hbcm)+4*(p2p3*p1p3)*(ffmcfmb*hbcm+fmb-
     . hbcm)+6*p2p3*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3+fmb*fmc*hbcm-fmb2*hbcm-fmc*hbcm2+hbcm3)+4*(2*ffmcfmb*
     . hbcm*p2p3s2+3*fmb*p2p3s2-hbcm*p1p2*p2p3-hbcm*p2p3*p2p4+3*hbcm*
     . p2p3*p2p5-3*hbcm*p2p3s2))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(w1*(8*p4p6*(ffmcfmb*hbcm3-2*fmc*hbcm2-hbcm3)+4*(p3p6*
     . p3p4)*(-ffmcfmb*hbcm+fmc-2*hbcm)+4*p3p6*(-ffmcfmb*fmc*hbcm2-2*
     . fmc*hbcm2+3*fmc2*hbcm)+8*(hbcm*p1p3*p4p6-2*hbcm*p1p4*p3p6+hbcm
     . *p2p3*p4p6-3*hbcm*p3p4*p4p6-hbcm*p3p5*p4p6))+w8*(16*p4p6*(
     . ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*(p3p6*p2p3)*(3*ffmcfmb*hbcm+
     . fmc-2*hbcm)+4*p3p6*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb
     . *fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*hbcm+2*fmb*hbcm2-2*fmb2*hbcm-
     . fmc*hbcm2)+16*hbcm*p2p3*p4p6)+w5*(4*p4p6*(ffmcfmb*hbcm3-2*fmb*
     . hbcm2-fmc*hbcm2-8*hbcm3)+4*(p3p6*p3p4)*(-ffmcfmb*hbcm+fmc-2*
     . hbcm)+4*(p3p6*p1p3)*(ffmcfmb*hbcm-fmc+2*hbcm)+4*p3p6*(ffmcfmb
     . **2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-5*ffmcfmb*
     . hbcm3-fmc*hbcm2+2*fmc2*hbcm)+16*(ffmcfmb*hbcm*p2p3*p3p6-2*
     . ffmcfmb*hbcm*p3p5*p3p6+2*hbcm*p1p3*p4p6-hbcm*p1p4*p3p6+2*hbcm*
     . p2p3*p4p6-2*hbcm*p3p4*p4p6-3*hbcm*p3p5*p4p6))+(2*p3p6*(-4*fb1*
     . ffmcfmb*hbcm+8*fb1*hbcm-4*fb2*fmc-hbcm)+32*fb1*hbcm*p4p6))
      b(9)=ans
      b(10)=ccc*(w1*((p3p6*p3p4)*(8*ffmcfmb+8)+8*p4p6*(ffmcfmb*hbcm2+
     . 3*fmc*hbcm)+8*p3p6*(2*ffmcfmb*fmc*hbcm+fmc*hbcm)+24*p3p4*p4p6)
     . +w8*((p3p6*p2p3)*(8*ffmcfmb+8)+24*p4p6*(ffmcfmb*hbcm2+fmb*hbcm
     . -hbcm2)+8*p3p6*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+fmb*hbcm-
     . hbcm2)+24*p2p3*p4p6)+w5*((p3p6*p3p4)*(8*ffmcfmb+8)+(p3p6*p1p3)
     . *(-8*ffmcfmb-8)+4*p4p6*(5*ffmcfmb*hbcm2-2*fmb*hbcm+fmc*hbcm)+4
     . *p3p6*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2+fmc
     . *hbcm)+24*(-p1p3*p4p6+p3p4*p4p6))+(p3p6*(-16*fb2*ffmcfmb-16*
     . fb2-4)-48*fb2*p4p6))
      ans2=(-12*fb1*fmc*hbcm+20*fb2*ffmcfmb*hbcm2-24*fb2*hbcm2+8*fb2*
     . p1p3+24*fb2*p2p3+8*fb2*p3p4-24*fb2*p3p5-hbcm2)
      ans1=w1*(2*p3p4*(-3*ffmcfmb*hbcm2+fmc*hbcm+6*hbcm2)+2*(-5*
     . ffmcfmb*fmc*hbcm3-2*fmc*hbcm*p1p3-6*fmc*hbcm*p2p3+6*fmc*hbcm*
     . p3p5+6*fmc*hbcm3+5*fmc2*hbcm2-4*hbcm2*p1p4-2*p1p3*p3p4-6*p2p3*
     . p3p4+6*p3p4*p3p5-2*p3p4s2))+w8*(12*p3p5*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+4*p3p4*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+2*p2p3*(-5*
     . ffmcfmb*hbcm2-6*fmb*hbcm-3*fmc*hbcm+6*hbcm2)+4*p1p3*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+2*(-2*ffmcfmb**2*hbcm4-5*ffmcfmb*fmb*
     . hbcm3-3*ffmcfmb*fmc*hbcm3+5*ffmcfmb*hbcm4-3*fmb*fmc*hbcm2+6*
     . fmb*hbcm3-3*fmb2*hbcm2+3*fmc*hbcm3-3*hbcm4-2*p1p3*p2p3-2*p2p3*
     . p3p4+6*p2p3*p3p5-6*p2p3s2))+w5*(2*p3p4*(-5*ffmcfmb*hbcm2+3*fmc
     . *hbcm+8*hbcm2)+4*p2p3*(-ffmcfmb*hbcm2-3*fmc*hbcm)+2*p1p3*(
     . ffmcfmb*hbcm2-3*fmc*hbcm-4*hbcm2)+2*(-2*ffmcfmb**2*hbcm4-3*
     . ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4+6*fmc*hbcm*p3p5+6*fmc*hbcm3+
     . 5*fmc2*hbcm2-2*hbcm2*p1p2-6*hbcm2*p1p4-2*hbcm2*p2p4+6*p1p3*
     . p2p3-6*p1p3*p3p5+2*p1p3s2-6*p2p3*p3p4+6*p3p4*p3p5-2*p3p4s2))+
     . ans2
      ans=ccc*ans1
      b(11)=ans
      ans=ccc*(w1*(4*(p3p7*p3p4)*(-4*ffmcfmb*hbcm+3*hbcm)+4*p3p7*(-4*
     . ffmcfmb*fmc*hbcm2+3*fmc*hbcm2)+4*(-fmc*hbcm2*p4p7+3*fmc*hbcm2*
     . p5p7-hbcm*p3p4*p4p7+3*hbcm*p3p4*p5p7))+w8*(12*p5p7*(-ffmcfmb*
     . hbcm3-fmb*hbcm2+hbcm3)+4*p4p7*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+
     . 24*(p3p7*p3p5)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*p3p4)*(-ffmcfmb*
     . hbcm+hbcm)+4*(p3p7*p2p3)*(-2*ffmcfmb*hbcm+3*hbcm)+8*(p3p7*p1p3
     . )*(-ffmcfmb*hbcm+hbcm)+4*p3p7*(-ffmcfmb**2*hbcm3+4*ffmcfmb*fmb
     . *hbcm2-3*ffmcfmb*fmc*hbcm2+4*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+4*(hbcm*p2p3*p4p7-3*hbcm*p2p3*p5p7))+w5*(4*p5p7
     . *(2*ffmcfmb*hbcm3+3*fmc*hbcm2)+4*p4p7*(ffmcfmb*hbcm3-2*hbcm3)+
     . 4*(p3p7*p3p4)*(-6*ffmcfmb*hbcm+5*hbcm)+4*(p3p7*p1p3)*(2*
     . ffmcfmb*hbcm-hbcm)+4*p3p7*(-ffmcfmb**2*hbcm3-3*ffmcfmb*fmc*
     . hbcm2+3*fmc*hbcm2)+4*(2*ffmcfmb*hbcm*p2p3*p3p7-2*ffmcfmb*hbcm*
     . p3p5*p3p7+hbcm*p1p3*p4p7-hbcm*p1p3*p5p7+2*hbcm*p2p3*p4p7-hbcm*
     . p3p4*p4p7+5*hbcm*p3p4*p5p7-2*hbcm*p3p5*p4p7))+(8*p3p7*(4*fb1*
     . ffmcfmb*hbcm-3*fb1*hbcm-2*hbcm)+8*(fb1*hbcm*p4p7-3*fb1*hbcm*
     . p5p7)))
      b(12)=ans
      b(13)=ccc*(w5*(2*p3p4*(-5*ffmcfmb*hbcm-3*fmc-2*hbcm)+2*p1p3*(5*
     . ffmcfmb*hbcm+3*fmc-2*hbcm)+2*(-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*
     . hbcm*p2p3-2*ffmcfmb*hbcm3-5*fmc2*hbcm+2*hbcm*p1p2+6*hbcm*p1p4+
     . 2*hbcm*p2p4))+w1*(2*p3p4*(-5*ffmcfmb*hbcm-3*fmc)+2*(-3*ffmcfmb
     . *fmc*hbcm2-5*fmc2*hbcm+4*hbcm*p1p4))+w8*(6*p2p3*(-ffmcfmb*hbcm
     . -fmc+2*hbcm)+6*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3))+3*(4*fb1*
     . ffmcfmb*hbcm+4*fb2*fmc-hbcm))
      b(14)=ccc*(w1*((p3p7*p3p4)*(-16*ffmcfmb+12)+4*p3p7*(-4*ffmcfmb*
     . fmc*hbcm+3*fmc*hbcm)+4*(-fmc*hbcm*p4p7+3*fmc*hbcm*p5p7-p3p4*
     . p4p7+3*p3p4*p5p7))+w8*((p3p7*p2p3)*(-16*ffmcfmb+12)+12*p5p7*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*p4p7*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+4*p3p7*(-ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-3*ffmcfmb*
     . fmc*hbcm+4*ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*hbcm-3*hbcm2)+4*(-
     . p2p3*p4p7+3*p2p3*p5p7))+w5*((p3p7*p3p4)*(-16*ffmcfmb+12)+(p3p7
     . *p1p3)*(16*ffmcfmb-12)+4*p3p7*(-ffmcfmb**2*hbcm2-3*ffmcfmb*fmc
     . *hbcm+3*fmc*hbcm)+4*(-ffmcfmb*hbcm2*p4p7+3*fmc*hbcm*p5p7+p1p3*
     . p4p7-3*p1p3*p5p7-p3p4*p4p7+3*p3p4*p5p7))+(p3p7*(32*fb2*ffmcfmb
     . -24*fb2+8)+8*(fb2*p4p7-3*fb2*p5p7)))
      ans2=w5*((p3p7*p3p6*p3p4)*(32*ffmcfmb-16)+(p3p7*p3p6*p1p3)*(-32
     . *ffmcfmb+16)+16*(p4p6*p3p7)*(ffmcfmb*hbcm2-hbcm2)+8*(p5p7*p3p6
     . )*(-3*ffmcfmb*hbcm2-fmc*hbcm)+4*(p6p7*p3p4)*(-3*ffmcfmb*hbcm2-
     . fmc*hbcm)+4*(p6p7*p1p3)*(3*ffmcfmb*hbcm2+fmc*hbcm)+4*p6p7*(-
     . ffmcfmb**2*hbcm4-ffmcfmb*fmc*hbcm3+2*fmc2*hbcm2-4*p1p3s2-4*
     . p3p4s2)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(-ffmcfmb*hbcm2*p3p6*p4p7-hbcm2*
     . p1p4*p6p7-hbcm2*p4p6*p4p7-2*hbcm2*p4p6*p5p7+2*p1p3*p3p4*p6p7+
     . p1p3*p3p6*p5p7-p1p3*p3p7*p4p6-p3p4*p3p6*p5p7+p3p4*p3p7*p4p6))+
     . ((p3p7*p3p6)*(-64*fb2*ffmcfmb+32*fb2-16)+2*p6p7*(-4*fb1*fmc*
     . hbcm+12*fb2*ffmcfmb*hbcm2-hbcm2)+32*(-fb2*p1p3*p6p7+fb2*p3p4*
     . p6p7+fb2*p3p6*p5p7-fb2*p3p7*p4p6))
      ans1=w1*((p3p7*p3p6*p3p4)*(32*ffmcfmb-16)+8*(p4p6*p3p7)*(2*
     . ffmcfmb*hbcm2+2*fmc*hbcm-hbcm2)+4*(p6p7*p3p4)*(-ffmcfmb*hbcm2-
     . 3*fmc*hbcm)+4*p6p7*(-3*ffmcfmb*fmc*hbcm3+3*fmc2*hbcm2-4*p3p4s2
     . )+16*(p3p7*p3p6)*(2*ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(-2*ffmcfmb*
     . hbcm2*p3p6*p4p7+2*fmc*hbcm*p1p3*p6p7-2*fmc*hbcm*p3p6*p5p7-2*
     . hbcm2*p1p4*p6p7-hbcm2*p4p6*p4p7-hbcm2*p4p6*p5p7+2*p1p3*p3p4*
     . p6p7-2*p3p4*p3p6*p5p7+2*p3p4*p3p7*p4p6))+w8*((p3p7*p3p6*p2p3)*
     . (32*ffmcfmb-16)+16*(p4p6*p3p7)*(2*ffmcfmb*hbcm2+fmb*hbcm-2*
     . hbcm2)+16*(p5p7*p3p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7
     . *p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p6p7*p2p3)*(ffmcfmb*
     . hbcm2-fmc*hbcm-4*hbcm2)+16*(p6p7*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm
     . -hbcm2)+4*p6p7*(-ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*hbcm3-ffmcfmb*
     . fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2-2*fmb2*hbcm2+fmc*hbcm3+2
     . *hbcm4)+8*(p3p7*p3p6)*(ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+16*(p1p3*
     . p2p3*p6p7-p2p3*p3p4*p6p7-p2p3*p3p6*p5p7+p2p3*p3p7*p4p6))+ans2
      ans=ccc*ans1
      b(15)=ans
      ans=ccc*(w1*(8*(p4p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4)*
     . (3*ffmcfmb*hbcm+fmc)+4*p6p7*(ffmcfmb*fmc*hbcm2+3*fmc2*hbcm)+8*
     . (-2*ffmcfmb*hbcm*p3p6*p4p7-2*hbcm*p1p4*p6p7-hbcm*p4p6*p4p7-
     . hbcm*p4p6*p5p7))+w8*(32*(p4p6*p3p7)*(ffmcfmb*hbcm-hbcm)+16*(
     . p3p7*p3p6)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p2p3)*(3*ffmcfmb*hbcm+
     . fmc-4*hbcm)+4*p6p7*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb
     . *fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmb2*hbcm-fmc*hbcm2+
     . 2*hbcm3))+w5*(16*(p4p6*p3p7)*(ffmcfmb*hbcm-2*hbcm)+4*(p6p7*
     . p3p4)*(3*ffmcfmb*hbcm+fmc)+4*(p6p7*p1p3)*(-3*ffmcfmb*hbcm-fmc)
     . +4*p6p7*(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2+2*fmc2*hbcm)+16*(-
     . ffmcfmb*hbcm*p3p6*p3p7-ffmcfmb*hbcm*p3p6*p4p7-2*ffmcfmb*hbcm*
     . p3p6*p5p7-hbcm*p1p4*p6p7-hbcm*p4p6*p4p7-3*hbcm*p4p6*p5p7))+2*
     . p6p7*(-4*fb1*ffmcfmb*hbcm-4*fb2*fmc-hbcm))
      b(16)=ans
      do 200 n=1,16
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
      end 
 
      double precision function ams1_3p0(n)
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
#include "inclppp.h"
#include "inclcon.h"
#include "inclamp.h"
      b1=-4*c(15,n)*c(11,n)+4*c(12,n)*c(9,n)
      b2=8*c(16,n)*c(13,n)-8*c(14,n)*c(10,n)
      b3=-4*c(16,n)*c(12,n)+4*c(15,n)*c(14,n)+4*c(13,n)*c(9,n)-4*c(11
     . ,n)*c(10,n)+b2*p3p7
      b4=4*c(16,n)*c(9,n)-4*c(15,n)*c(10,n)+4*c(14,n)*c(11,n)-4*c(13,
     . n)*c(12,n)
      b5=2*c(15,n)**2+2*c(12,n)**2+2*c(11,n)**2+2*c(9,n)**2+b1*p6p7+
     . b3*p3p6+b4*p3p7
      b6=2*c(15,n)*c(5,n)-2*c(12,n)*c(1,n)-2*c(11,n)*c(6,n)+2*c(9,n)*
     . c(8,n)
      b7=-4*c(11,n)*c(6,n)+4*c(9,n)*c(8,n)
      b8=2*c(15,n)*c(6,n)+2*c(12,n)*c(8,n)+2*c(11,n)*c(5,n)+2*c(9,n)*
     . c(1,n)+b7*p6p7
      b9=-2*c(16,n)*c(8,n)-2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)-2*c(13,n
     . )*c(1,n)+2*c(12,n)*c(3,n)-2*c(11,n)*c(4,n)+2*c(10,n)*c(5,n)-2*
     . c(9,n)*c(2,n)
      b10=-4*c(13,n)*c(2,n)*p6p7
      b11=4*c(14,n)*c(4,n)+4*c(13,n)*c(3,n)-4*c(10,n)*c(7,n)+b10
      b12=-2*c(16,n)*c(1,n)+2*c(15,n)*c(4,n)-2*c(14,n)*c(5,n)-2*c(13,
     . n)*c(8,n)-2*c(12,n)*c(2,n)+2*c(11,n)*c(7,n)-2*c(10,n)*c(6,n)+2
     . *c(9,n)*c(3,n)+b11*p3p6+b9*p6p7
      b13=-4*c(13,n)*c(8,n)+4*c(11,n)*c(7,n)-4*c(10,n)*c(6,n)+4*c(9,n
     . )*c(3,n)
      b14=2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)-2*c(14,n)*c(6,n)+2*c(13,n
     . )*c(1,n)-2*c(12,n)*c(3,n)+2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)+2*
     . c(9,n)*c(2,n)
      b15=b13*p5p7+b14*p5p6
      b16=-2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)+2*c(13,
     . n)*c(1,n)-2*c(12,n)*c(3,n)-2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)+2
     . *c(9,n)*c(2,n)
      b17=-4*c(12,n)*c(2,n)*p5p6
      b18=8*c(13,n)*c(3,n)-8*c(10,n)*c(7,n)
      b19=4*c(13,n)*c(2,n)*p5p6
      b20=b18*p5p7+b19
      b21=b16*p5p7+b17+b20*p3p7
      b22=4*c(13,n)*c(2,n)*p3p6s2*p5p7
      b23=b22
      aa1=b12*p3p5+b15*p3p7+b21*p3p6+b23+b5*p2p5+b6*p5p6+b8*p5p7
      b1=4*c(11,n)*c(2,n)*p6p7
      b2=4*c(16,n)*c(7,n)-4*c(14,n)*c(3,n)-4*c(13,n)*c(4,n)
      b3=2*c(16,n)*c(5,n)-2*c(15,n)*c(2,n)-2*c(14,n)*c(1,n)+2*c(13,n)
     . *c(6,n)+2*c(12,n)*c(4,n)-2*c(11,n)*c(3,n)-2*c(10,n)*c(8,n)+2*c
     . (9,n)*c(7,n)+b1+b2*p3p7
      b4=-2*c(15,n)*c(8,n)+2*c(12,n)*c(6,n)+2*c(11,n)*c(1,n)-2*c(9,n)
     . *c(5,n)
      b5=2*c(16,n)*c(6,n)-2*c(15,n)*c(3,n)+2*c(14,n)*c(8,n)+2*c(13,n)
     . *c(5,n)-2*c(12,n)*c(7,n)+2*c(11,n)*c(2,n)+2*c(10,n)*c(1,n)-2*c
     . (9,n)*c(4,n)
      b6=-4*c(14,n)*c(2,n)*p3p6s2
      b7=-2*c(15,n)*c(1,n)-2*c(12,n)*c(5,n)+2*c(11,n)*c(8,n)+2*c(9,n)
     . *c(6,n)+b3*p3p6+b4*p6p7+b5*p3p7+b6
      b8=-4*c(14,n)*c(8,n)-4*c(13,n)*c(5,n)+4*c(12,n)*c(7,n)-4*c(11,n
     . )*c(2,n)
      b9=-2*c(16,n)*c(5,n)+2*c(15,n)*c(2,n)-2*c(14,n)*c(1,n)+2*c(13,n
     . )*c(6,n)-2*c(12,n)*c(4,n)+2*c(11,n)*c(3,n)+2*c(10,n)*c(8,n)+2*
     . c(9,n)*c(7,n)+b8*p6p7
      b10=-2*c(16,n)*c(6,n)+2*c(15,n)*c(3,n)-2*c(14,n)*c(8,n)-2*c(13,
     . n)*c(5,n)+2*c(12,n)*c(7,n)-2*c(11,n)*c(2,n)-2*c(10,n)*c(1,n)+2
     . *c(9,n)*c(4,n)
      b11=4*c(16,n)*c(7,n)-4*c(14,n)*c(3,n)-4*c(13,n)*c(4,n)-4*c(10,n
     . )*c(2,n)
      b12=-4*c(16,n)*c(4,n)+4*c(14,n)*c(2,n)+4*c(13,n)*c(7,n)+4*c(10,
     . n)*c(3,n)+b11*p6p7
      b13=4*c(14,n)*c(2,n)+8*c(13,n)*c(7,n)
      b14=-4*c(16,n)*c(7,n)+4*c(14,n)*c(3,n)+4*c(13,n)*c(4,n)
      b15=b13*p4p6+b14*p4p7
      b16=b10*p4p7+b12*p3p4+b15*p3p6+b9*p4p6
      b17=-4*c(8,n)*c(5,n)*p6p7
      b18=2*c(8,n)*c(6,n)-2*c(5,n)*c(1,n)+b17
      b19=-2*c(8,n)*c(4,n)+2*c(7,n)*c(1,n)-2*c(6,n)*c(2,n)-2*c(5,n)*c
     . (3,n)
      b20=4*c(7,n)*c(2,n)*p6p7
      b21=-4*c(4,n)*c(2,n)+b20
      b22=4*c(7,n)*c(2,n)-4*c(4,n)*c(3,n)
      b23=2*c(8,n)*c(7,n)+2*c(6,n)*c(3,n)+2*c(5,n)*c(2,n)-2*c(4,n)*c(
     . 1,n)+b19*p6p7+b21*p3p6+b22*p3p7
      b24=2*c(8,n)*c(4,n)-2*c(7,n)*c(1,n)-2*c(6,n)*c(2,n)+2*c(5,n)*c(
     . 3,n)
      b25=4*c(8,n)*c(7,n)*p4p6
      b26=4*c(7,n)*c(2,n)*p3p7*p4p6
      b27=b26
      b28=b24*p4p7+b25+b27
      b29=-2*c(8,n)*c(4,n)-2*c(7,n)*c(1,n)+2*c(6,n)*c(2,n)-2*c(5,n)*c
     . (3,n)
      b30=b29*p3p7*p4p6
      b31=-2*c(8,n)*c(5,n)-2*c(6,n)*c(1,n)
      b32=-2*c(16,n)*c(7,n)+2*c(14,n)*c(3,n)+2*c(13,n)*c(4,n)+2*c(10,
     . n)*c(2,n)
      b33=2*c(16,n)*c(4,n)-2*c(14,n)*c(2,n)-2*c(13,n)*c(7,n)-2*c(10,n
     . )*c(3,n)+b32*p6p7
      b34=-4*c(7,n)*c(2,n)*p6p7
      b35=2*c(7,n)*c(3,n)+2*c(4,n)*c(2,n)+b34
      b36=-2*c(7,n)*c(2,n)+2*c(4,n)*c(3,n)
      b37=b33*p2p4+b35*p4p6+b36*p4p7
      b38=-4*c(7,n)*c(2,n)*p3p6s2*p4p7
      b39=b38
      aa2=b16*p2p3+b18*p4p6+b23*p3p4+b28*p3p6+b30+b31*p4p7+b37*hbcm2+
     . b39+b7*p2p4
      b1=-4*c(11,n)*c(2,n)*p6p7
      b2=4*c(16,n)*c(7,n)+4*c(14,n)*c(3,n)-4*c(13,n)*c(4,n)
      b3=2*c(16,n)*c(5,n)+2*c(15,n)*c(2,n)+2*c(14,n)*c(1,n)+2*c(13,n)
     . *c(6,n)+2*c(12,n)*c(4,n)+2*c(11,n)*c(3,n)+2*c(10,n)*c(8,n)+2*c
     . (9,n)*c(7,n)+b1+b2*p3p7
      b4=2*c(15,n)*c(8,n)+2*c(12,n)*c(6,n)-2*c(11,n)*c(1,n)-2*c(9,n)*
     . c(5,n)
      b5=2*c(16,n)*c(6,n)+2*c(15,n)*c(3,n)-2*c(14,n)*c(8,n)+2*c(13,n)
     . *c(5,n)-2*c(12,n)*c(7,n)-2*c(11,n)*c(2,n)-2*c(10,n)*c(1,n)-2*c
     . (9,n)*c(4,n)
      b6=4*c(14,n)*c(2,n)*p3p6s2
      b7=2*c(15,n)*c(1,n)-2*c(12,n)*c(5,n)-2*c(11,n)*c(8,n)+2*c(9,n)*
     . c(6,n)+b3*p3p6+b4*p6p7+b5*p3p7+b6
      b8=-4*c(16,n)*c(11,n)+4*c(15,n)*c(13,n)-4*c(14,n)*c(9,n)+4*c(12
     . ,n)*c(10,n)
      b9=8*c(16,n)*c(14,n)+8*c(13,n)*c(10,n)
      b10=4*c(16,n)*c(15,n)-4*c(14,n)*c(12,n)-4*c(13,n)*c(11,n)+4*c(
     . 10,n)*c(9,n)+b8*p6p7+b9*p3p6
      b11=2*c(16,n)*c(5,n)+2*c(15,n)*c(2,n)+2*c(14,n)*c(1,n)+2*c(13,n
     . )*c(6,n)-2*c(12,n)*c(4,n)-2*c(11,n)*c(3,n)+2*c(10,n)*c(8,n)-2*
     . c(9,n)*c(7,n)
      b12=4*c(13,n)*c(6,n)-4*c(11,n)*c(3,n)+4*c(10,n)*c(8,n)-4*c(9,n)
     . *c(7,n)
      b13=2*c(16,n)*c(6,n)+2*c(15,n)*c(3,n)-2*c(14,n)*c(8,n)-2*c(13,n
     . )*c(5,n)-2*c(12,n)*c(7,n)+2*c(11,n)*c(2,n)+2*c(10,n)*c(1,n)+2*
     . c(9,n)*c(4,n)+b12*p6p7
      b14=4*c(16,n)*c(7,n)+4*c(14,n)*c(3,n)+4*c(13,n)*c(4,n)
      b15=4*c(14,n)*c(2,n)*p5p6
      b16=b14*p5p7+b15
      b17=b10*p2p5+b11*p5p6+b13*p5p7+b16*p3p6
      b18=-2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)-2*c(14,n)*c(6,n)-2*c(13,
     . n)*c(1,n)+2*c(12,n)*c(3,n)+2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)-2
     . *c(9,n)*c(2,n)
      b19=-4*c(13,n)*c(2,n)*p6p7
      b20=-4*c(14,n)*c(4,n)+4*c(13,n)*c(3,n)+4*c(10,n)*c(7,n)+b19
      b21=-2*c(16,n)*c(1,n)-2*c(15,n)*c(4,n)+2*c(14,n)*c(5,n)-2*c(13,
     . n)*c(8,n)-2*c(12,n)*c(2,n)-2*c(11,n)*c(7,n)+2*c(10,n)*c(6,n)+2
     . *c(9,n)*c(3,n)+b18*p6p7+b20*p3p6
      b22=-4*c(8,n)*c(2,n)*p6p7
      b23=-4*c(7,n)*c(4,n)*p3p7
      b24=2*c(8,n)*c(3,n)+2*c(7,n)*c(6,n)-2*c(5,n)*c(4,n)-2*c(2,n)*c(
     . 1,n)+b22+b23
      b25=-2*c(8,n)*c(1,n)-2*c(6,n)*c(5,n)
      b26=-2*c(8,n)*c(2,n)+2*c(7,n)*c(5,n)-2*c(6,n)*c(4,n)-2*c(3,n)*c
     . (1,n)
      b27=2*c(7,n)*c(4,n)-2*c(3,n)*c(2,n)
      b28=-c(7,n)**2-c(4,n)**2+c(3,n)**2+c(2,n)**2+b27*p6p7
      b29=-c(8,n)**2+c(6,n)**2+c(5,n)**2-c(1,n)**2+b21*p2p3+b24*p3p6+
     . b25*p6p7+b26*p3p7+b28*hbcm2
      b30=2*c(8,n)*c(6,n)+2*c(5,n)*c(1,n)
      b31=4*c(8,n)*c(6,n)*p6p7
      b32=-2*c(8,n)*c(5,n)+2*c(6,n)*c(1,n)+b31
      b33=2*c(8,n)*c(4,n)-2*c(7,n)*c(1,n)-2*c(6,n)*c(2,n)-2*c(5,n)*c(
     . 3,n)
      b34=-4*c(7,n)*c(2,n)*p6p7
      b35=4*c(7,n)*c(3,n)+b34
      b36=-2*c(8,n)*c(7,n)+2*c(6,n)*c(3,n)+2*c(5,n)*c(2,n)+2*c(4,n)*c
     . (1,n)+b33*p6p7+b35*p3p6
      b37=-4*c(8,n)*c(7,n)+4*c(6,n)*c(3,n)
      b38=-2*c(8,n)*c(4,n)+2*c(7,n)*c(1,n)+2*c(6,n)*c(2,n)+2*c(5,n)*c
     . (3,n)
      b39=b37*p5p7+b38*p5p6
      b40=2*c(8,n)*c(4,n)+2*c(7,n)*c(1,n)+2*c(6,n)*c(2,n)+2*c(5,n)*c(
     . 3,n)
      b41=8*c(7,n)*c(3,n)*p5p7
      b42=4*c(7,n)*c(2,n)*p5p6
      b43=b41+b42
      b44=4*c(5,n)*c(2,n)*p5p6
      b45=b40*p5p7+b43*p3p7+b44
      b46=-2*c(16,n)*c(7,n)-2*c(14,n)*c(3,n)+2*c(13,n)*c(4,n)-2*c(10,
     . n)*c(2,n)
      b47=2*c(16,n)*c(4,n)+2*c(14,n)*c(2,n)-2*c(13,n)*c(7,n)+2*c(10,n
     . )*c(3,n)+b46*p6p7
      b48=-2*c(7,n)*c(3,n)+2*c(4,n)*c(2,n)
      b49=-4*c(7,n)*c(3,n)*p6p7
      b50=2*c(7,n)*c(2,n)+2*c(4,n)*c(3,n)+b49
      b51=b47*p2p5+b48*p5p6+b50*p5p7
      b52=4*c(7,n)*c(2,n)*p3p6s2*p5p7
      b53=b52
      aa3=b17*p2p3+b29*fmb+b30*p5p6+b32*p5p7+b36*p3p5+b39*p3p7+b45*
     . p3p6+b51*hbcm2+b53+b7*p2p5
      b1=4*c(12,n)*c(8,n)-4*c(11,n)*c(5,n)
      b2=2*c(15,n)*c(5,n)+2*c(12,n)*c(1,n)+2*c(11,n)*c(6,n)+2*c(9,n)*
     . c(8,n)+b1*p6p7
      b3=2*c(16,n)*c(8,n)-2*c(15,n)*c(7,n)-2*c(14,n)*c(6,n)+2*c(13,n)
     . *c(1,n)+2*c(12,n)*c(3,n)-2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)-2*c
     . (9,n)*c(2,n)
      b4=4*c(13,n)*c(2,n)*p6p7
      b5=4*c(16,n)*c(2,n)+b4
      b6=4*c(16,n)*c(3,n)+4*c(14,n)*c(7,n)+4*c(13,n)*c(2,n)-4*c(10,n)
     . *c(4,n)
      b7=2*c(16,n)*c(1,n)+2*c(15,n)*c(4,n)+2*c(14,n)*c(5,n)+2*c(13,n)
     . *c(8,n)-2*c(12,n)*c(2,n)+2*c(11,n)*c(7,n)+2*c(10,n)*c(6,n)+2*c
     . (9,n)*c(3,n)+b3*p6p7+b5*p3p6+b6*p3p7
      b8=-2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)-2*c(13,n
     . )*c(1,n)-2*c(12,n)*c(3,n)+2*c(11,n)*c(4,n)+2*c(10,n)*c(5,n)-2*
     . c(9,n)*c(2,n)
      b9=4*c(14,n)*c(5,n)+4*c(13,n)*c(8,n)+4*c(11,n)*c(7,n)
      b10=8*c(14,n)*c(7,n)+4*c(13,n)*c(2,n)
      b11=b10*p3p7*p4p6
      b12=b11+b8*p4p7+b9*p4p6
      b13=2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)-2*c(13,n
     . )*c(1,n)+2*c(12,n)*c(3,n)-2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)+2*
     . c(9,n)*c(2,n)
      b14=b13*p3p7*p4p6
      b15=2*c(15,n)*c(6,n)+2*c(12,n)*c(8,n)-2*c(11,n)*c(5,n)-2*c(9,n)
     . *c(1,n)
      b16=-4*c(13,n)*c(2,n)*p3p6s2*p4p7
      b17=b16
      aa4=b12*p3p6+b14+b15*p4p7+b17+b2*p4p6+b7*p3p4
      b1=-8*c(14,n)*c(11,n)+8*c(13,n)*c(12,n)
      b2=4*c(16,n)*c(12,n)+4*c(15,n)*c(14,n)+4*c(13,n)*c(9,n)+4*c(11,
     . n)*c(10,n)+b1*p6p7
      b3=-4*c(16,n)*c(9,n)+4*c(15,n)*c(10,n)-4*c(14,n)*c(11,n)+4*c(13
     . ,n)*c(12,n)
      b4=8*c(16,n)*c(13,n)-8*c(14,n)*c(10,n)
      b5=4*c(16,n)**2+4*c(14,n)**2+4*c(13,n)**2+4*c(10,n)**2+b4*p6p7
      b6=8*c(14,n)**2+8*c(13,n)**2
      b7=-8*c(16,n)*c(13,n)+8*c(14,n)*c(10,n)
      b8=b6*p4p6+b7*p4p7
      b9=b2*p4p6+b3*p4p7+b5*p3p4+b8*p3p6
      b10=2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)-2*c(14,n)*c(6,n)+2*c(13,n
     . )*c(1,n)-2*c(12,n)*c(3,n)+2*c(11,n)*c(4,n)-2*c(10,n)*c(5,n)+2*
     . c(9,n)*c(2,n)
      b11=2*c(16,n)*c(1,n)-2*c(15,n)*c(4,n)+2*c(14,n)*c(5,n)+2*c(13,n
     . )*c(8,n)+2*c(12,n)*c(2,n)-2*c(11,n)*c(7,n)+2*c(10,n)*c(6,n)-2*
     . c(9,n)*c(3,n)+b10*p6p7
      b12=4*c(13,n)*c(8,n)-4*c(11,n)*c(7,n)+4*c(10,n)*c(6,n)-4*c(9,n)
     . *c(3,n)
      b13=-2*c(16,n)*c(8,n)-2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)-2*c(13,
     . n)*c(1,n)+2*c(12,n)*c(3,n)-2*c(11,n)*c(4,n)+2*c(10,n)*c(5,n)-2
     . *c(9,n)*c(2,n)
      b14=b12*p5p7+b13*p5p6
      b15=8*c(13,n)*c(8,n)-8*c(11,n)*c(7,n)
      b16=2*c(16,n)*c(8,n)+2*c(15,n)*c(7,n)+2*c(14,n)*c(6,n)+2*c(13,n
     . )*c(1,n)+2*c(12,n)*c(3,n)+2*c(11,n)*c(4,n)+2*c(10,n)*c(5,n)-2*
     . c(9,n)*c(2,n)+b15*p6p7
      b17=4*c(14,n)*c(5,n)+4*c(13,n)*c(8,n)+4*c(12,n)*c(2,n)-4*c(11,n
     . )*c(7,n)
      b18=b16*p5p7+b17*p5p6
      b19=4*c(16,n)*c(2,n)+4*c(14,n)*c(4,n)+4*c(13,n)*c(3,n)-4*c(10,n
     . )*c(7,n)
      b20=8*c(13,n)*c(3,n)-8*c(10,n)*c(7,n)
      b21=4*c(16,n)*c(3,n)+4*c(14,n)*c(7,n)-4*c(13,n)*c(2,n)+4*c(10,n
     . )*c(4,n)+b20*p6p7
      b22=b19*p5p6+b21*p5p7
      b23=4*c(13,n)*c(2,n)*p6p7
      b24=-4*c(14,n)*c(4,n)-4*c(13,n)*c(3,n)+4*c(10,n)*c(7,n)+b23
      b25=-8*c(13,n)*c(3,n)+8*c(10,n)*c(7,n)
      b26=-4*c(13,n)*c(2,n)*p5p6
      b27=b25*p5p7+b26
      b28=8*c(14,n)*c(7,n)-4*c(13,n)*c(2,n)
      b29=b28*p4p6*p5p7
      b30=b24*p4p5+b27*p4p7+b29
      aa5=b11*p4p5+b14*p4p7+b18*p4p6+b22*p3p4+b30*p3p6+b9*p2p5
      b1=2*c(8,n)*c(1,n)-2*c(6,n)*c(5,n)
      aa6=c(8,n)**2+c(6,n)**2+c(5,n)**2+c(1,n)**2+b1*p6p7
      b1=2*c(8,n)**2+2*c(6,n)**2
      b2=-2*c(8,n)*c(1,n)+2*c(6,n)*c(5,n)
      aa7=b1*p5p7+b2*p5p6
      b1=2*c(8,n)*c(3,n)-2*c(7,n)*c(6,n)+2*c(5,n)*c(4,n)+2*c(2,n)*c(1
     . ,n)
      b2=4*c(8,n)*c(3,n)-4*c(7,n)*c(6,n)
      b3=-2*c(8,n)*c(2,n)+2*c(7,n)*c(5,n)+2*c(6,n)*c(4,n)+2*c(3,n)*c(
     . 1,n)+b2*p6p7
      b4=-4*c(7,n)*c(4,n)-4*c(3,n)*c(2,n)
      b5=2*c(7,n)**2+2*c(4,n)**2+2*c(3,n)**2+2*c(2,n)**2+b4*p6p7
      b6=4*c(7,n)**2+4*c(3,n)**2
      b7=4*c(7,n)*c(4,n)+4*c(3,n)*c(2,n)
      b8=b6*p5p7+b7*p5p6
      b9=4*c(3,n)*c(2,n)*p5p7
      b10=4*c(2,n)**2*p5p6
      b11=b10+b9
      aa8=b1*p5p6+b11*p3p6+b3*p5p7+b5*p3p5+b8*p3p7
      b1=-4*c(8,n)*c(2,n)-4*c(7,n)*c(5,n)
      b2=2*c(8,n)*c(3,n)+2*c(7,n)*c(6,n)+2*c(5,n)*c(4,n)-2*c(2,n)*c(1
     . ,n)+b1*p6p7
      b3=-2*c(8,n)*c(2,n)-2*c(7,n)*c(5,n)+2*c(6,n)*c(4,n)-2*c(3,n)*c(
     . 1,n)
      b4=4*c(7,n)**2*p4p6
      b5=4*c(7,n)*c(4,n)*p4p7
      b6=b4+b5
      aa9=b2*p4p6+b3*p4p7+b6*p3p6
      b1=4*c(8,n)*c(2,n)*p6p7
      b2=-2*c(8,n)*c(3,n)+2*c(7,n)*c(6,n)-2*c(5,n)*c(4,n)+2*c(2,n)*c(
     . 1,n)+b1
      b3=-4*c(8,n)*c(3,n)+4*c(7,n)*c(6,n)
      b4=-4*c(8,n)*c(2,n)*p5p6
      b5=b3*p5p7+b4
      b6=8*c(7,n)**2*p4p6*p5p7
      b7=b6
      b8=-4*c(7,n)*c(4,n)*p4p5
      b9=b7+b8
      b10=4*c(7,n)*c(5,n)*p4p6*p5p7
      b11=b10
      aa10=b11+b2*p4p5+b5*p4p7+b9*p3p7
      b1=4*c(8,n)*c(3,n)+4*c(7,n)*c(6,n)
      b2=4*c(8,n)*c(2,n)+4*c(7,n)*c(5,n)
      b3=b1*p5p7+b2*p5p6
      b4=2*c(8,n)*c(2,n)+2*c(7,n)*c(5,n)-2*c(6,n)*c(4,n)+2*c(3,n)*c(1
     . ,n)
      aa11=b3*p4p6+b4*p4p5
      b1=4*c(8,n)**2*p6p7
      b2=2*c(8,n)*c(1,n)+2*c(6,n)*c(5,n)+b1
      b3=2*c(8,n)**2+2*c(5,n)**2
      aa12=b2*p5p7+b3*p5p6
      b1=-4*c(16,n)*c(13,n)+4*c(14,n)*c(10,n)
      b2=-2*c(16,n)**2-2*c(14,n)**2-2*c(13,n)**2-2*c(10,n)**2+b1*p6p7
      b3=-2*c(16,n)*c(2,n)-2*c(14,n)*c(4,n)-2*c(13,n)*c(3,n)+2*c(10,n
     . )*c(7,n)
      b4=-4*c(13,n)*c(3,n)+4*c(10,n)*c(7,n)
      b5=-2*c(16,n)*c(3,n)-2*c(14,n)*c(7,n)+2*c(13,n)*c(2,n)-2*c(10,n
     . )*c(4,n)+b4*p6p7
      b6=b2*p2p5+b3*p5p6+b5*p5p7
      b7=-4*c(14,n)*c(7,n)-4*c(13,n)*c(2,n)
      b8=-2*c(16,n)*c(2,n)+2*c(14,n)*c(4,n)+2*c(13,n)*c(3,n)+2*c(10,n
     . )*c(7,n)+b7*p6p7
      b9=-2*c(16,n)*c(3,n)-2*c(14,n)*c(7,n)-2*c(13,n)*c(2,n)+2*c(10,n
     . )*c(4,n)
      b10=b8*p4p6+b9*p4p7
      b11=2*c(7,n)*c(4,n)+2*c(3,n)*c(2,n)
      b12=-c(7,n)**2-c(4,n)**2-c(3,n)**2-c(2,n)**2+b11*p6p7
      b13=-2*c(7,n)**2-2*c(2,n)**2
      b14=-4*c(7,n)**2*p6p7
      b15=2*c(7,n)*c(4,n)-2*c(3,n)*c(2,n)+b14
      b16=b13*p5p6+b15*p5p7
      b17=-2*c(7,n)**2-2*c(3,n)**2
      b18=-2*c(7,n)*c(4,n)-2*c(3,n)*c(2,n)
      b19=b17*p5p7+b18*p5p6
      aa13=b10*p2p5+b12*p4p5+b16*p4p6+b19*p4p7+b6*p2p4
      ams1_3p0=aa1*p2p4+aa10*p3p6+aa11*p3p7+aa12*p4p6+aa13*hbcm2+aa2*fmb
     . +aa3*fmc+aa4*p2p5+aa5*p2p3+aa6*p4p5+aa7*p4p7+aa8*p3p4+aa9*p3p5
      return
      end
      
