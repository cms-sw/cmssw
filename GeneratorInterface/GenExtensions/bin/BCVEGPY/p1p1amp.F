c...this is the 36 amplitude for (1p1) state.
c...obtained from the program fdc.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) z.x. zhang, j.x. wang and x.g. wu                    c
c reference: hep-ph/0309120                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine amp10_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(2)=ccc*(w1*(8*(p6p7*p4p8)*(fmb2*hbcm-fmc2*hbcm-hbcm3)-16*hbcm
     . *p3p4*p4p8*p6p7)+32*(fb4*hbcm*p2p8*p6p7+fb4*hbcm*p3p7*p6p8+fb4
     . *hbcm*p4p7*p6p8))
      b(3)=8*ccc*w1*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(4)=16*ccc*p5p7*(-fb4*fmb*hbcm-fb4*fmc*hbcm)
      b(5)=ccc*(w1*(8*(p5p7*p4p8)*(2*fmb*hbcm2-2*fmc*hbcm2-hbcm3)-16*
     . hbcm*p3p4*p4p8*p5p7)+32*fb4*hbcm*p4p8*p5p7)
      b(6)=w1*ccc*(8*(p5p6*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p4p8*
     . p4p6)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p4p8*p3p6)*(fmb*hbcm+fmc*
     . hbcm-2*hbcm2))
      b(7)=ccc*(16*p5p6*(-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*p4p6*(-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)+16*p3p6*(-fb4*fmb*hbcm-fb4*fmc*hbcm))
      b(8)=ccc*(w1*(8*(p5p6*p4p8)*(2*fmb*hbcm2-2*fmc*hbcm2-hbcm3)+8*(
     . p4p8*p4p6)*(2*fmb*hbcm2-2*fmc*hbcm2-hbcm3)+8*(p4p8*p3p6)*(2*
     . fmb*hbcm2-fmb2*hbcm-2*fmc*hbcm2+fmc2*hbcm)+16*(-hbcm*p3p4*p4p6
     . *p4p8-hbcm*p3p4*p4p8*p5p6))+(16*p6p8*(fb4*fmb2*hbcm-fb4*fmc2*
     . hbcm-fb4*hbcm3)+32*(-fb4*hbcm*p3p4*p6p8+fb4*hbcm*p3p6*p4p8+fb4
     . *hbcm*p4p6*p4p8+fb4*hbcm*p4p8*p5p6)))
      b(9)=ccc*(16*p6p7*(-fb4*fmb2*hbcm+fb4*fmc2*hbcm+fb4*hbcm3)+32*(
     . -fb4*hbcm*p2p3*p6p7+fb4*hbcm*p3p4*p6p7-fb4*hbcm*p3p6*p3p7-fb4*
     . hbcm*p3p6*p4p7))
      b(10)=ccc*(w1*(16*(p6p7*p4p8)*(-fmb2*hbcm2+fmc2*hbcm2+hbcm4)+16
     . *(p5p6*p4p8*p3p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(p4p8*p4p6*
     . p3p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(p4p8*p4p7*p3p6)*(fmb*
     . hbcm+fmc*hbcm-2*hbcm2)+32*hbcm2*p3p4*p4p8*p6p7)+(32*(p7p8*p5p6
     . )*(fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(p6p8*p4p7)*(-fb4*fmb*hbcm-
     . fb4*fmc*hbcm)+32*(p7p8*p4p6)*(fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(
     . p6p8*p3p7)*(-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(p7p8*p3p6)*(fb4*
     . fmb*hbcm+fb4*fmc*hbcm)+32*(p6p7*p2p8)*(-fb4*fmb*hbcm-fb4*fmc*
     . hbcm)))
      b(11)=ccc*(w1*(4*p4p8*(fmb2*hbcm-fmc2*hbcm-hbcm3)-8*hbcm*p3p4*
     . p4p8)+16*(-fb4*hbcm*p1p8-fb4*hbcm*p2p8+fb4*hbcm*p4p8+fb4*hbcm*
     . p5p8))
      b(13)=ccc*(w1*(8*p4p8*(-fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3-fmc2*
     . hbcm2+hbcm4)+8*(p4p8*p3p5)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(
     . p4p8*p3p4)*(-fmb*hbcm-fmc*hbcm)+8*(p4p8*p2p3)*(fmb*hbcm+fmc*
     . hbcm-2*hbcm2)+8*(p4p8*p1p3)*(fmb*hbcm+fmc*hbcm-2*hbcm2))+(16*
     . p5p8*(fb4*fmb*hbcm+fb4*fmc*hbcm)+16*p4p8*(fb4*fmb*hbcm+fb4*fmc
     . *hbcm)+16*p2p8*(-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*p1p8*(-fb4*fmb*
     . hbcm-fb4*fmc*hbcm)))
      b(14)=8*ccc*(-fb4*fmb2*hbcm+fb4*fmc2*hbcm+2*fb4*hbcm*p1p3+2*fb4
     . *hbcm*p2p3-2*fb4*hbcm*p3p5-fb4*hbcm3)
      b(15)=ccc*(8*w1*(p4p8*p3p6)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+16*p6p8
     . *(-fb4*fmb*hbcm-fb4*fmc*hbcm))
      b(16)=16*ccc*fb4*hbcm*p6p8
      b(17)=ccc*(w1*(4*p4p8*(-2*fmb*hbcm2+2*fmc*hbcm2+hbcm3)+8*hbcm*
     . p3p4*p4p8)-16*fb4*hbcm*p4p8)
      b(19)=4*ccc*w1*p4p8*(-fmb*hbcm-fmc*hbcm+2*hbcm2)
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(24)=-16*ccc*fb4*hbcm*p3p6
      b(25)=-16*ccc*fb4*hbcm*p5p7
      b(26)=16*ccc*(-fb4*hbcm*p3p6-fb4*hbcm*p4p6-fb4*hbcm*p5p6)
      b(27)=16*ccc*p6p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(28)=8*ccc*w1*(p6p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(31)=ccc*(w1*(8*(p6p7*p4p8)*(-2*fmb*hbcm2+2*fmc*hbcm2+hbcm3)+
     . 16*hbcm*p3p4*p4p8*p6p7)-32*fb4*hbcm*p4p8*p6p7)
      b(32)=16*ccc*fb4*hbcm*p6p7
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp18_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      b(1)=ccc*(32*(p5p6*p3p7)*(-fb3*ffmcfmb+fb3)+32*(p4p6*p3p7)*(fb3
     . *ffmcfmb-fb3)+4*p6p7*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*
     . hbcm2-8*fb3*fmb*fmc-4*fb3*fmb2-8*fb4*ffmcfmb*fmb*hbcm-8*fb4*
     . ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+8*fb4*fmc*hbcm+2*fmb*hbcm+2*
     . fmc*hbcm-hbcm2)+32*(-fb3*ffmcfmb*p2p3*p6p7-2*fb3*p2p4*p6p7+fb3
     . *p2p5*p6p7))
      ans2=(8*(p6p7*p5p8)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+8*(p6p8*p5p7)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+8*(p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+8*(p6p8*p4p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+8*(p7p8*p4p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+8*(p6p7*p2p8)*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+
     . hbcm)+8*(p6p8*p3p7)*(-4*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc+4*
     . fb3*fmb-4*fb3*fmc-ffmcfmb*hbcm+hbcm)+32*(p7p8*p3p6)*(-fb3*
     . ffmcfmb*fmb-fb4*ffmcfmb**2*hbcm+fb4*ffmcfmb*hbcm)-8*hbcm*p5p6*
     . p7p8)
      ans1=w18*(16*(p5p8*p5p6*p3p7)*(ffmcfmb*hbcm-hbcm)+16*(p5p6*p4p8
     . *p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p5p8)*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+
     . 8*(p6p7*p4p8)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*fmc*
     . hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+16*(p5p8*p3p7*p3p6)*(
     . ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*
     . hbcm-ffmcfmb*hbcm)+16*(hbcm*p2p4*p4p8*p6p7+hbcm*p2p4*p5p8*p6p7
     . -hbcm*p2p5*p4p8*p6p7-hbcm*p2p5*p5p8*p6p7))+w1*(16*(p5p6*p4p8*
     . p3p7)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p4p8)*(ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+
     . 16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(hbcm*
     . p2p4*p4p8*p6p7-hbcm*p2p5*p4p8*p6p7))+w8*(16*(p5p6*p3p7*p2p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p8)*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2-fmb*fmc*hbcm-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2)+16*(p3p7
     . *p3p6*p2p8)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(hbcm*p2p4*p2p8*
     . p6p7-hbcm*p2p5*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w18*(8*(p5p8*p5p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p5p7*p4p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p5p8*p4p7)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p8*p4p7)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+8*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+fmb*hbcm+fmc*hbcm)+8*(p4p8*p3p7)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm))+w1*(8*(p5p7*p4p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p8*p4p7)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+8*(p4p8*p3p7)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*
     . hbcm+fmb*hbcm+fmc*hbcm))+w8*(8*(p5p7*p2p8)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+8*(p4p7*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(
     . p3p7*p2p8)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*
     . hbcm))+(32*(p5p8*p3p7)*(-fb3*ffmcfmb+fb3)+32*(p4p8*p3p7)*(fb3*
     . ffmcfmb-fb3)+4*p7p8*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*
     . hbcm2-4*fb3*fmb*fmc-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*
     . hbcm+4*fb4*fmc*hbcm+2*fmb*hbcm+2*fmc*hbcm-hbcm2)+32*(-fb3*
     . ffmcfmb*p2p3*p7p8-fb3*ffmcfmb*p2p8*p3p7-fb3*p2p4*p7p8-2*fb3*
     . p2p8*p4p7+fb3*p2p8*p5p7)))
      b(3)=ans
      b(4)=ccc*(32*(p3p7*p3p5)*(fb3*ffmcfmb-fb3)+32*(p3p7*p3p4)*(-fb3
     . *ffmcfmb+fb3)+4*p5p7*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*
     . fmb*hbcm-hbcm2)+4*p4p7*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*
     . fmb*hbcm+hbcm2)+4*p3p7*(4*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*
     . hbcm2+4*fb3*fmb*fmc+8*fb4*ffmcfmb*fmb*hbcm+8*fb4*ffmcfmb*fmc*
     . hbcm-4*fb4*fmb*hbcm-8*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+hbcm2)+32
     . *(-fb3*p2p3*p4p7+fb3*p2p4*p3p7))
      ans2=(32*(p5p8*p3p7)*(fb4*ffmcfmb*hbcm-fb4*hbcm)+8*(p3p7*p2p8)*
     . (-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p7p8*p2p3)*(4*fb4*ffmcfmb*hbcm-
     . hbcm)+8*(p4p8*p3p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+8*(p7p8*p3p4)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+4*p7p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2+
     . 4*fb3*fmc*hbcm2+4*fb4*ffmcfmb**2*hbcm3-4*fb4*ffmcfmb*hbcm3-4*
     . fb4*fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2)+8*(4*fb4*hbcm*p2p4*p7p8-
     . 4*fb4*hbcm*p2p8*p4p7+hbcm*p3p5*p7p8))
      ans1=w18*(8*(p5p8*p5p7)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(p5p7
     . *p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p5p8*p3p7*p3p5)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p4p8*p3p7*p3p5)*(-ffmcfmb*hbcm+hbcm)+8*
     . (p5p8*p3p7)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2)+8*(p4p8*
     . p3p7)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2)+16*(hbcm*p2p3*
     . p4p7*p4p8+hbcm*p2p3*p4p7*p5p8-hbcm*p2p4*p3p7*p4p8-hbcm*p2p4*
     . p3p7*p5p8))+w1*(8*(p5p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+
     . 16*(p4p8*p3p7*p3p5)*(-ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7)*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*
     . hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2)+16*(hbcm*p2p3*p4p7*
     . p4p8-hbcm*p2p4*p3p7*p4p8))+w8*(8*(p5p7*p2p8)*(ffmcfmb*hbcm3+
     . fmb*hbcm2-hbcm3)+16*(p3p7*p3p5*p2p8)*(-ffmcfmb*hbcm+hbcm)+8*(
     . p3p7*p2p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2)+16*(hbcm
     . *p2p3*p2p8*p4p7-hbcm*p2p4*p2p8*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w18*(8*(p5p8*p5p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*
     . (p5p6*p4p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p8*p4p6)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p8*p4p6)*(ffmcfmb*hbcm2+fmb
     . *hbcm-hbcm2))+w1*(8*(p5p6*p4p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2
     . )+8*(p4p8*p4p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+w8*(8*(p5p6*
     . p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p6*p2p8)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2))+(4*p6p8*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*
     . ffmcfmb*hbcm2-8*fb3*fmb*fmc-4*fb3*fmb2-8*fb4*ffmcfmb*fmb*hbcm-
     . 8*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+8*fb4*fmc*hbcm+2*fmb*
     . hbcm+2*fmc*hbcm-hbcm2)+32*(-fb3*ffmcfmb*p2p3*p6p8-fb3*ffmcfmb*
     . p2p8*p3p6-2*fb3*p2p4*p6p8+fb3*p2p5*p6p8-fb3*p2p8*p4p6)))
      b(7)=ccc*(4*p5p6*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb*
     . hbcm+hbcm2)+4*p4p6*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb
     . *hbcm-hbcm2)+4*p3p6*(4*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*
     . hbcm2+8*fb3*fmb*fmc+4*fb3*fmb2+8*fb4*ffmcfmb*fmb*hbcm+8*fb4*
     . ffmcfmb*fmc*hbcm-4*fb4*fmb*hbcm-8*fb4*fmc*hbcm-2*fmb*hbcm-2*
     . fmc*hbcm+hbcm2)+32*(2*fb3*ffmcfmb*p2p3*p3p6+fb3*p2p3*p4p6+2*
     . fb3*p2p4*p3p6-fb3*p2p5*p3p6))
      ans2=w8*(8*(p5p6*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p3p6
     . *p2p8)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2
     . )+16*(-ffmcfmb*hbcm*p2p3*p2p8*p3p6-hbcm*p2p3*p2p8*p4p6-hbcm*
     . p2p4*p2p8*p3p6+hbcm*p2p5*p2p8*p3p6))+(8*(p3p6*p2p8)*(-4*fb3*
     . fmc+hbcm)+32*(p6p8*p2p3)*(fb3*fmc+fb4*ffmcfmb*hbcm)+8*(p5p8*
     . p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p4p8*
     . p3p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p8*
     . p3p5)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p8*
     . p3p4)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*p6p8*(4
     . *fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-4*fb3*fmb*hbcm2
     . +4*fb3*fmc*hbcm2-4*fb4*fmb*fmc*hbcm+4*fb4*fmb2*hbcm+ffmcfmb*
     . hbcm3+2*fmb*hbcm2-fmc*hbcm2-hbcm3)+32*(fb4*hbcm*p2p4*p6p8-fb4*
     . hbcm*p2p5*p6p8+fb4*hbcm*p2p8*p4p6))
      ans1=w18*(8*(p5p8*p5p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(
     . p5p6*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p5p8*p3p6)*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb
     . *hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+8*(p4p8*
     . p3p6)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2
     . +ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+16*
     . (-ffmcfmb*hbcm*p2p3*p3p6*p4p8-ffmcfmb*hbcm*p2p3*p3p6*p5p8-hbcm
     . *p2p3*p4p6*p4p8-hbcm*p2p3*p4p6*p5p8-hbcm*p2p4*p3p6*p4p8-hbcm*
     . p2p4*p3p6*p5p8+hbcm*p2p5*p3p6*p4p8+hbcm*p2p5*p3p6*p5p8))+w1*(8
     . *(p5p6*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p4p8*p3p6)*(-
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb
     . *hbcm3+fmb*fmc*hbcm+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2)+16*(-
     . ffmcfmb*hbcm*p2p3*p3p6*p4p8-hbcm*p2p3*p4p6*p4p8-hbcm*p2p4*p3p6
     . *p4p8+hbcm*p2p5*p3p6*p4p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(32*(p5p6*p3p7)*(-fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(p4p6*
     . p3p7)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p5p7*
     . p3p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p4p7*
     . p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p6p7*
     . p3p5)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p6p7*
     . p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p7*
     . p2p3)*(-4*fb3*fmb+8*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p3p7*
     . p3p6)*(8*fb3*ffmcfmb*fmb-4*fb3*ffmcfmb*fmc-4*fb3*fmb+4*fb3*fmc
     . -ffmcfmb*hbcm)+4*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*
     . fmc*hbcm2+4*fb3*fmb*hbcm2-4*fb3*fmc*hbcm2+4*fb4*fmb*fmc*hbcm-4
     . *fb4*fmb2*hbcm-fmb*hbcm2+fmc*hbcm2)+32*(-fb4*hbcm*p2p4*p6p7+
     . fb4*hbcm*p2p5*p6p7))
      ans3=(8*(p6p7*p2p8)*(4*fb4*fmb*hbcm+4*fb4*fmc*hbcm-hbcm2)+8*(
     . p6p8*p3p7)*(4*fb3*fmb*fmc+4*fb3*fmb2+4*fb4*ffmcfmb*fmb*hbcm+4*
     . fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc*hbcm+ffmcfmb*hbcm2-2*fmb*hbcm-
     . fmc*hbcm)+64*(p5p8*p3p7*p3p6)*(fb3*ffmcfmb-fb3)+64*(p6p8*p3p7*
     . p3p5)*(-fb3*ffmcfmb+fb3)+8*(p6p7*p5p8)*(-4*fb3*ffmcfmb*hbcm2+4
     . *fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+8*(p6p8*p5p7)*(4*fb3*ffmcfmb*
     . hbcm2-4*fb3*hbcm2+4*fb4*fmb*hbcm+hbcm2)+32*(p7p8*p5p6)*(-fb3*
     . ffmcfmb*hbcm2+fb3*hbcm2-fb4*fmb*hbcm)+8*(p7p8*p3p6)*(-4*fb3*
     . ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-4*fb3*fmb2-
     . 8*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+4
     . *fb4*fmc*hbcm+ffmcfmb*hbcm2+fmb*hbcm+fmc*hbcm-hbcm2)+8*(-8*fb3
     . *ffmcfmb*p2p3*p3p6*p7p8+8*fb3*ffmcfmb*p2p3*p3p7*p6p8+16*fb3*
     . ffmcfmb*p2p8*p3p6*p3p7-8*fb3*p2p3*p4p6*p7p8+8*fb3*p2p3*p4p7*
     . p6p8-8*fb3*p2p3*p4p8*p6p7-8*fb3*p2p4*p3p6*p7p8+8*fb3*p2p4*p3p7
     . *p6p8+8*fb3*p2p5*p3p6*p7p8-8*fb3*p2p5*p3p7*p6p8-8*fb3*p2p8*
     . p3p4*p6p7+8*fb3*p2p8*p3p5*p6p7+8*fb3*p2p8*p3p6*p4p7-8*fb3*p2p8
     . *p3p6*p5p7+16*fb3*p2p8*p3p7*p4p6+hbcm2*p4p6*p7p8))
      ans2=w1*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm2+hbcm2)+16*(p5p7*
     . p4p8*p3p6)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p4p8*p3p5)
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p6p7*p4p8)*(ffmcfmb**2*
     . hbcm4-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*hbcm2)+16*(p4p8*p3p7*p3p6)*
     . (-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16
     . *(fmb*hbcm*p3p7*p4p8*p5p6+fmc*hbcm*p2p3*p4p8*p6p7+hbcm2*p2p5*
     . p4p8*p6p7))+w8*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+hbcm2)+16*
     . (p5p7*p3p6*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p3p5
     . *p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p6p7*p2p8)*(ffmcfmb**
     . 2*hbcm4-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*hbcm2)+16*(p3p7*p3p6*p2p8
     . )*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+
     . 16*(fmb*hbcm*p2p8*p3p7*p5p6+fmc*hbcm*p2p3*p2p8*p6p7+hbcm2*p2p5
     . *p2p8*p6p7))+ans3
      ans1=w18*(16*(p5p8*p4p6*p3p7)*(-ffmcfmb*hbcm2+hbcm2)+16*(p4p8*
     . p4p6*p3p7)*(-ffmcfmb*hbcm2+hbcm2)+16*(p5p8*p5p7*p3p6)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p5p7*p4p8*p3p6)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p5p8*p3p5)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+16*(p6p7*p4p8*p3p5)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)
     . +8*(p6p7*p5p8)*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*
     . hbcm2)+8*(p6p7*p4p8)*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4+fmb*hbcm3
     . -fmb2*hbcm2)+16*(p5p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*
     . fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16*(p4p8*p3p7*p3p6)*(-ffmcfmb
     . **2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16*(fmb*
     . hbcm*p3p7*p4p8*p5p6+fmb*hbcm*p3p7*p5p6*p5p8+fmc*hbcm*p2p3*p4p8
     . *p6p7+fmc*hbcm*p2p3*p5p8*p6p7+hbcm2*p2p5*p4p8*p6p7+hbcm2*p2p5*
     . p5p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w18*(4*p5p8*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb*hbcm2+fmb2*
     . hbcm+2*fmc*hbcm2)+4*p4p8*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2
     . -2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb*hbcm2+
     . fmb2*hbcm+2*fmc*hbcm2)+8*(ffmcfmb*hbcm*p2p3*p4p8+ffmcfmb*hbcm*
     . p2p3*p5p8+2*hbcm*p2p4*p4p8+2*hbcm*p2p4*p5p8-hbcm*p2p5*p4p8-
     . hbcm*p2p5*p5p8))+w1*(4*p4p8*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*
     . hbcm2-2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm-fmb*
     . hbcm2+fmb2*hbcm+2*fmc*hbcm2)+8*(ffmcfmb*hbcm*p2p3*p4p8+2*hbcm*
     . p2p4*p4p8-hbcm*p2p5*p4p8))+w8*(4*p2p8*(ffmcfmb**2*hbcm3+2*
     . ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*
     . hbcm-fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2)+8*(ffmcfmb*hbcm*p2p3*
     . p2p8+2*hbcm*p2p4*p2p8-hbcm*p2p5*p2p8))+(8*p2p8*(2*fb3*fmc+2*
     . fb4*ffmcfmb*hbcm-hbcm)+4*p5p8*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4
     . *fb4*hbcm+hbcm)+8*p4p8*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)))
      b(11)=ans
      b(12)=2*ccc*(8*fb3*ffmcfmb**2*hbcm2-8*fb3*ffmcfmb*hbcm2+16*fb3*
     . ffmcfmb*p2p3+12*fb3*fmb*fmc+4*fb3*fmb2+24*fb3*p2p4-8*fb3*p2p5+
     . 12*fb4*ffmcfmb*fmb*hbcm+12*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmb*hbcm
     . -12*fb4*fmc*hbcm-3*fmb*hbcm-3*fmc*hbcm+2*hbcm2)
      ans2=w8*(8*(p3p5*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p4*
     . p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p2p8*p2p3)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*p2p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+
     . fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3)+8*(hbcm2*p2p4*
     . p2p8-hbcm2*p2p5*p2p8))+(4*p5p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*
     . hbcm2+4*fb4*fmb*hbcm+hbcm2)+4*p4p8*(-4*fb3*ffmcfmb*hbcm2+4*fb3
     . *hbcm2-4*fb4*fmb*hbcm-hbcm2)+4*p2p8*(-4*fb3*ffmcfmb*hbcm2+4*
     . fb4*fmc*hbcm-hbcm2)+32*(fb3*p2p3*p4p8-fb3*p2p8*p3p4))
      ans1=w18*(8*(p5p8*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p8
     . *p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p8*p3p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+8*(p4p8*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*(p5p8*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p2p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*p5p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3)+4*p4p8*(
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+
     . fmb2*hbcm2-fmc*hbcm3)+8*(hbcm2*p2p4*p4p8+hbcm2*p2p4*p5p8-hbcm2
     . *p2p5*p4p8-hbcm2*p2p5*p5p8))+w1*(8*(p4p8*p3p5)*(-ffmcfmb*hbcm2
     . -fmb*hbcm+hbcm2)+8*(p4p8*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+
     . 8*(p4p8*p2p3)*(ffmcfmb*hbcm2-fmc*hbcm)+4*p4p8*(ffmcfmb*fmb*
     . hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc
     . *hbcm3)+8*(hbcm2*p2p4*p4p8-hbcm2*p2p5*p4p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(4*p2p3*(-4*fb3*fmc-8*fb4*ffmcfmb*hbcm+hbcm)+4*p3p5*(
     . 4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*p3p4*(-4*fb3*
     . fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*(-8*fb3*ffmcfmb*fmb*
     . hbcm2+8*fb3*ffmcfmb*fmc*hbcm2+4*fb3*fmb*hbcm2-8*fb3*fmc*hbcm2-
     . 4*fb4*ffmcfmb**2*hbcm3+4*fb4*ffmcfmb*hbcm3+8*fb4*fmb*fmc*hbcm-
     . 4*fb4*fmb2*hbcm-16*fb4*hbcm*p2p4+8*fb4*hbcm*p2p5-2*fmb*hbcm2+2
     . *fmc*hbcm2+hbcm3))
      b(15)=ccc*(w18*(8*(p5p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*
     . hbcm)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-
     . hbcm2*p4p6*p4p8-hbcm2*p4p6*p5p8+hbcm2*p4p8*p5p6+hbcm2*p5p6*
     . p5p8))+w1*(8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8
     . *(-hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+w8*(8*(p3p6*p2p8)*(
     . ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-hbcm2*p2p8*p4p6+hbcm2*
     . p2p8*p5p6))+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmb*hbcm+8*fb4
     . *fmc*hbcm-hbcm2)+32*(-2*fb3*p3p4*p6p8+fb3*p3p5*p6p8+2*fb3*p3p6
     . *p4p8-fb3*p3p6*p5p8)))
      b(16)=ccc*(8*w18*(ffmcfmb*hbcm*p3p6*p4p8+ffmcfmb*hbcm*p3p6*p5p8
     . +2*hbcm*p4p6*p4p8+2*hbcm*p4p6*p5p8-hbcm*p4p8*p5p6-hbcm*p5p6*
     . p5p8)+8*w1*(ffmcfmb*hbcm*p3p6*p4p8+2*hbcm*p4p6*p4p8-hbcm*p4p8*
     . p5p6)+8*w8*(ffmcfmb*hbcm*p2p8*p3p6+2*hbcm*p2p8*p4p6-hbcm*p2p8*
     . p5p6)+4*p6p8*(4*fb3*fmb-12*fb3*fmc-8*fb4*ffmcfmb*hbcm+hbcm))
      b(17)=ccc*(w18*(4*p5p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+4
     . *p4p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+8*(-2*hbcm*p3p4*
     . p4p8-2*hbcm*p3p4*p5p8+hbcm*p3p5*p4p8+hbcm*p3p5*p5p8))+w1*(4*
     . p4p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+8*(-2*hbcm*p3p4*
     . p4p8+hbcm*p3p5*p4p8))+w8*(4*p2p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*
     . fmc*hbcm2)+8*(-2*hbcm*p2p8*p3p4+hbcm*p2p8*p3p5))+16*(2*fb4*
     . hbcm*p4p8-fb4*hbcm*p5p8))
      b(18)=ccc*(w18*(8*(p5p8*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p8
     . *p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p4p7*p4p8-hbcm2*
     . p4p7*p5p8+hbcm2*p4p8*p5p7+hbcm2*p5p7*p5p8))+w1*(8*(p4p8*p3p7)*
     . (-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p4p7*p4p8+hbcm2*p4p8*p5p7)
     . )+w8*(8*(p3p7*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p2p8*
     . p4p7+hbcm2*p2p8*p5p7))+(4*p7p8*(4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*
     . hbcm+hbcm2)+32*(fb3*p3p4*p7p8-fb3*p3p7*p4p8)))
      b(19)=ccc*(w18*(4*p5p8*(-2*ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm)+4
     . *p4p8*(-2*ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm))+4*w1*p4p8*(-2*
     . ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm)+4*w8*p2p8*(-2*ffmcfmb*hbcm2
     . +fmb*hbcm+3*fmc*hbcm)+16*(-3*fb3*p4p8+fb3*p5p8))
      b(20)=ccc*(8*w18*(-ffmcfmb*hbcm*p3p7*p4p8-ffmcfmb*hbcm*p3p7*
     . p5p8-2*hbcm*p4p7*p4p8-2*hbcm*p4p7*p5p8+hbcm*p4p8*p5p7+hbcm*
     . p5p7*p5p8)+8*w1*(-ffmcfmb*hbcm*p3p7*p4p8-2*hbcm*p4p7*p4p8+hbcm
     . *p4p8*p5p7)+8*w8*(-ffmcfmb*hbcm*p2p8*p3p7-2*hbcm*p2p8*p4p7+
     . hbcm*p2p8*p5p7)+8*p7p8*(-2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm))
      b(21)=2*ccc*(4*fb3*fmb-16*fb3*fmc-12*fb4*ffmcfmb*hbcm+3*hbcm)
      b(22)=4*ccc*(4*fb3*ffmcfmb*hbcm2+12*fb3*p3p4-4*fb3*p3p5-2*fb4*
     . fmb*hbcm-6*fb4*fmc*hbcm+hbcm2)
      b(23)=ccc*(4*p3p7*(4*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+16*(2*fb4
     . *hbcm*p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(8*p3p6*(-2*fb3*fmb+6*fb3*fmc+2*fb4*ffmcfmb*hbcm-hbcm
     . )+16*(-2*fb4*hbcm*p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(4*p5p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+8*p4p7*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*
     . p3p7*(-8*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*fmc+4*fb3*fmb-8*fb3*fmc
     . +hbcm))
      b(26)=ccc*(4*p5p6*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+8*p4p6*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*
     . p3p6*(4*fb3*ffmcfmb*fmb+4*fb4*ffmcfmb**2*hbcm-4*fb4*ffmcfmb*
     . hbcm-ffmcfmb*hbcm))
      b(27)=ccc*(4*p6p7*(4*fb3*ffmcfmb*hbcm2-4*fb4*fmb*hbcm-8*fb4*fmc
     . *hbcm+hbcm2)+32*(-2*fb3*ffmcfmb*p3p6*p3p7+2*fb3*p3p4*p6p7-fb3*
     . p3p5*p6p7-2*fb3*p3p6*p4p7+fb3*p3p6*p5p7-fb3*p3p7*p4p6))
      b(28)=ccc*(w18*(8*(p6p7*p5p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*
     . hbcm)+8*(p6p7*p4p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm))+8*w1*
     . (p6p7*p4p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*w8*(p6p7*
     . p2p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+32*(-fb3*ffmcfmb*
     . p3p6*p7p8-fb3*ffmcfmb*p3p7*p6p8-fb3*p4p6*p7p8-2*fb3*p4p7*p6p8+
     . 2*fb3*p4p8*p6p7+fb3*p5p7*p6p8-fb3*p5p8*p6p7))
      b(29)=16*ccc*(-2*fb3*ffmcfmb*p3p6-3*fb3*p4p6+fb3*p5p6)
      b(30)=16*ccc*(2*fb3*ffmcfmb*p3p7+3*fb3*p4p7-fb3*p5p7)
      ans=ccc*(w18*(8*(p6p7*p5p8)*(fmb*hbcm2-fmc*hbcm2)+8*(p6p7*p4p8)
     . *(fmb*hbcm2-fmc*hbcm2)+16*(ffmcfmb*hbcm*p3p6*p3p7*p4p8+ffmcfmb
     . *hbcm*p3p6*p3p7*p5p8-hbcm*p3p4*p4p8*p6p7-hbcm*p3p4*p5p8*p6p7+
     . hbcm*p3p5*p4p8*p6p7+hbcm*p3p5*p5p8*p6p7+hbcm*p3p6*p4p7*p4p8+
     . hbcm*p3p6*p4p7*p5p8-hbcm*p3p6*p4p8*p5p7-hbcm*p3p6*p5p7*p5p8+
     . hbcm*p3p7*p4p6*p4p8+hbcm*p3p7*p4p6*p5p8))+w1*(8*(p6p7*p4p8)*(
     . fmb*hbcm2-fmc*hbcm2)+16*(ffmcfmb*hbcm*p3p6*p3p7*p4p8-hbcm*p3p4
     . *p4p8*p6p7+hbcm*p3p5*p4p8*p6p7+hbcm*p3p6*p4p7*p4p8-hbcm*p3p6*
     . p4p8*p5p7+hbcm*p3p7*p4p6*p4p8))+w8*(8*(p6p7*p2p8)*(fmb*hbcm2-
     . fmc*hbcm2)+16*(ffmcfmb*hbcm*p2p8*p3p6*p3p7-hbcm*p2p8*p3p4*p6p7
     . +hbcm*p2p8*p3p5*p6p7+hbcm*p2p8*p3p6*p4p7-hbcm*p2p8*p3p6*p5p7+
     . hbcm*p2p8*p3p7*p4p6))+(32*(p6p8*p3p7)*(-fb3*fmc-fb4*ffmcfmb*
     . hbcm)+8*(p7p8*p3p6)*(4*fb3*fmc-hbcm)+32*(-fb4*hbcm*p4p6*p7p8-
     . fb4*hbcm*p4p7*p6p8+fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p7*p6p8-fb4*
     . hbcm*p5p8*p6p7)))
      b(31)=ans
      b(32)=8*ccc*p6p7*(2*fb3*fmb-6*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp17_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(1)=ccc*(16*p6p7*(-fb3*ffmcfmb**2*hbcm2+fb3*fmc2)+32*(fb3*
     . ffmcfmb*p1p3*p6p7+fb3*ffmcfmb*p3p6*p3p7+2*fb3*ffmcfmb*p3p7*
     . p4p6+fb3*p1p2*p6p7-fb3*p2p3*p6p7-3*fb3*p2p4*p6p7))
      ans=ccc*(w2*(16*(p5p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm
     . )+16*(ffmcfmb*hbcm*p3p7*p4p6*p5p8+hbcm*p1p2*p5p8*p6p7-hbcm*
     . p2p3*p5p8*p6p7-2*hbcm*p2p4*p5p8*p6p7))+w17*(16*(p3p7*p3p6*p2p8
     . )*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(p3p7*p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(ffmcfmb*hbcm*p1p8*p3p7*p4p6+
     . ffmcfmb*hbcm*p2p8*p3p7*p4p6+hbcm*p1p2*p1p8*p6p7+hbcm*p1p2*p2p8
     . *p6p7-hbcm*p1p8*p2p3*p6p7-2*hbcm*p1p8*p2p4*p6p7-hbcm*p2p3*p2p8
     . *p6p7-2*hbcm*p2p4*p2p8*p6p7))+w12*(16*(p3p7*p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(ffmcfmb*hbcm*p1p8*p3p7*p4p6+
     . hbcm*p1p2*p1p8*p6p7-hbcm*p1p8*p2p3*p6p7-2*hbcm*p1p8*p2p4*p6p7)
     . )+(8*(p7p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*(p6p7*p2p8)*(4*fb3*fmb
     . +4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+32*(p6p8*p3p7)*(-fb3*
     . ffmcfmb*fmb-fb4*ffmcfmb**2*hbcm+fb4*ffmcfmb*hbcm)+8*hbcm*p4p6*
     . p7p8))
      b(2)=ans
      b(3)=ccc*(8*w2*(p5p8*p3p7)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+
     . w17*(8*(p3p7*p2p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+8*(p3p7
     . *p1p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm))+8*w12*(p3p7*p1p8)*
     . (ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm)+(4*p7p8*(-4*fb3*ffmcfmb**2
     . *hbcm2+4*fb3*fmc2+fmb*hbcm+fmc*hbcm)+32*(fb3*ffmcfmb*p1p3*p7p8
     . +fb3*ffmcfmb*p1p8*p3p7-2*fb3*ffmcfmb*p3p7*p5p8-2*fb3*p1p2*p7p8
     . +2*fb3*p2p3*p7p8+3*fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*p5p7))
     . )
      b(4)=ccc*(16*p3p7*(fb3*ffmcfmb**2*hbcm2+2*fb3*ffmcfmb*hbcm2-fb3
     . *fmc2+fb4*ffmcfmb*fmb*hbcm+fb4*ffmcfmb*fmc*hbcm)+32*(-2*fb3*
     . ffmcfmb*p1p3*p3p7-2*fb3*ffmcfmb*p2p3*p3p7+2*fb3*ffmcfmb*p3p5*
     . p3p7+2*fb3*p1p2*p3p7+3*fb3*p2p3*p5p7-3*fb3*p2p5*p3p7))
      ans2=(4*p7p8*(ffmcfmb*hbcm3-fmc*hbcm2+hbcm3)+8*(p3p7*p2p8)*(8*
     . fb3*fmb-4*fb3*fmc-hbcm)+32*(p7p8*p2p3)*(-2*fb3*fmb+fb3*fmc-fb4
     . *ffmcfmb*hbcm)+8*(-4*fb4*ffmcfmb*hbcm*p1p8*p3p7+4*fb4*ffmcfmb*
     . hbcm*p3p7*p5p8+4*fb4*hbcm*p1p2*p7p8-8*fb4*hbcm*p2p5*p7p8+8*fb4
     . *hbcm*p2p8*p5p7-hbcm*p1p3*p7p8+hbcm*p3p5*p7p8))
      ans1=w2*(8*(p5p8*p3p7)*(ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3)+16*(-ffmcfmb*hbcm*p1p3*p3p7*p5p8-ffmcfmb*hbcm*
     . p2p3*p3p7*p5p8+ffmcfmb*hbcm*p3p5*p3p7*p5p8+hbcm*p1p2*p3p7*p5p8
     . +2*hbcm*p2p3*p5p7*p5p8-2*hbcm*p2p5*p3p7*p5p8))+w17*(8*(p3p7*
     . p2p8)*(ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3)+8*(
     . p3p7*p1p8)*(ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3)+
     . 16*(-ffmcfmb*hbcm*p1p3*p1p8*p3p7-ffmcfmb*hbcm*p1p3*p2p8*p3p7-
     . ffmcfmb*hbcm*p1p8*p2p3*p3p7+ffmcfmb*hbcm*p1p8*p3p5*p3p7-
     . ffmcfmb*hbcm*p2p3*p2p8*p3p7+ffmcfmb*hbcm*p2p8*p3p5*p3p7+hbcm*
     . p1p2*p1p8*p3p7+hbcm*p1p2*p2p8*p3p7+2*hbcm*p1p8*p2p3*p5p7-2*
     . hbcm*p1p8*p2p5*p3p7+2*hbcm*p2p3*p2p8*p5p7-2*hbcm*p2p5*p2p8*
     . p3p7))+w12*(8*(p3p7*p1p8)*(ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3)+16*(-ffmcfmb*hbcm*p1p3*p1p8*p3p7-ffmcfmb*hbcm*
     . p1p8*p2p3*p3p7+ffmcfmb*hbcm*p1p8*p3p5*p3p7+hbcm*p1p2*p1p8*p3p7
     . +2*hbcm*p1p8*p2p3*p5p7-2*hbcm*p1p8*p2p5*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(32*(p3p6*p2p8)*(-fb3*ffmcfmb-fb3)+16*p6p8*(-fb3*
     . ffmcfmb**2*hbcm2+fb3*fmc2)+32*(fb3*ffmcfmb*p1p3*p6p8-fb3*p2p4*
     . p6p8-3*fb3*p2p8*p4p6))
      b(7)=ccc*(32*(p3p6*p2p3)*(fb3*ffmcfmb+fb3)+16*p3p6*(fb3*ffmcfmb
     . **2*hbcm2-fb3*fmc2)+32*(-fb3*ffmcfmb*p1p3*p3p6+3*fb3*p2p3*p4p6
     . +fb3*p2p4*p3p6))
      ans=ccc*(w2*(8*(p5p8*p3p6)*(ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(-
     . ffmcfmb*hbcm*p1p3*p3p6*p5p8+hbcm*p2p3*p3p6*p5p8+2*hbcm*p2p3*
     . p4p6*p5p8+hbcm*p2p4*p3p6*p5p8))+w17*(8*(p3p6*p2p8)*(ffmcfmb**2
     . *hbcm3-fmc2*hbcm)+8*(p3p6*p1p8)*(ffmcfmb**2*hbcm3-fmc2*hbcm)+
     . 16*(-ffmcfmb*hbcm*p1p3*p1p8*p3p6-ffmcfmb*hbcm*p1p3*p2p8*p3p6+
     . hbcm*p1p8*p2p3*p3p6+2*hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p4*p3p6+
     . hbcm*p2p3*p2p8*p3p6+2*hbcm*p2p3*p2p8*p4p6+hbcm*p2p4*p2p8*p3p6)
     . )+w12*(8*(p3p6*p1p8)*(ffmcfmb**2*hbcm3-fmc2*hbcm)+16*(-ffmcfmb
     . *hbcm*p1p3*p1p8*p3p6+hbcm*p1p8*p2p3*p3p6+2*hbcm*p1p8*p2p3*p4p6
     . +hbcm*p1p8*p2p4*p3p6))+(16*p6p8*(fb4*ffmcfmb**2*hbcm3-fb4*fmc2
     . *hbcm)+8*(p3p6*p2p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-hbcm)+8*(
     . p6p8*p2p3)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(
     . -4*fb4*ffmcfmb*hbcm*p1p3*p6p8+4*fb4*hbcm*p2p4*p6p8+8*fb4*hbcm*
     . p2p8*p4p6-hbcm*p1p8*p3p6)))
      b(8)=ans
      b(9)=ccc*(8*(p6p7*p2p3)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)+8*
     . (p3p7*p3p6)*(4*fb3*ffmcfmb*fmb-ffmcfmb*hbcm)+32*(fb4*ffmcfmb*
     . hbcm*p3p7*p4p6+fb4*hbcm*p1p2*p6p7-2*fb4*hbcm*p2p4*p6p7))
      ans2=(8*(p7p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm)+64*(p3p7*p3p6*p2p8
     . )*(3*fb3*ffmcfmb+fb3)+8*(p6p7*p2p8)*(-4*fb3*ffmcfmb*hbcm2-4*
     . fb3*hbcm2+4*fb4*fmb*hbcm+hbcm2)+8*(p6p8*p3p7)*(4*fb3*ffmcfmb**
     . 2*hbcm2+4*fb3*ffmcfmb*hbcm2-4*fb4*ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2)+8*(-8*fb3*ffmcfmb*p1p3*p3p7*p6p8+8*fb3*ffmcfmb*p1p8*
     . p3p6*p3p7-8*fb3*ffmcfmb*p2p3*p3p6*p7p8+8*fb3*ffmcfmb*p3p4*p3p7
     . *p6p8-8*fb3*ffmcfmb*p3p6*p3p7*p4p8+8*fb3*p1p3*p2p8*p6p7+8*fb3*
     . p1p8*p2p3*p6p7+16*fb3*p2p3*p2p8*p6p7-8*fb3*p2p3*p4p6*p7p8+8*
     . fb3*p2p3*p4p7*p6p8-16*fb3*p2p3*p4p8*p6p7-8*fb3*p2p8*p3p4*p6p7+
     . 32*fb3*p2p8*p3p7*p4p6+hbcm2*p1p8*p6p7))
      ans1=w2*(16*(p6p7*p5p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+8*(p6p7
     . *p5p8)*(-ffmcfmb**2*hbcm4+fmc2*hbcm2)+16*(p5p8*p3p7*p3p6)*(
     . ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(ffmcfmb*hbcm2*p1p3*p5p8
     . *p6p7-hbcm2*p1p2*p5p8*p6p7+hbcm2*p2p4*p5p8*p6p7))+w17*(16*(
     . p6p7*p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+16*(p6p7*p2p3*p1p8)
     . *(-fmb*hbcm-fmc*hbcm+hbcm2)+8*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4+
     . fmc2*hbcm2)+8*(p6p7*p1p8)*(-ffmcfmb**2*hbcm4+fmc2*hbcm2)+16*(
     . p3p7*p3p6*p2p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(p3p7*
     . p3p6*p1p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(ffmcfmb*
     . hbcm2*p1p3*p1p8*p6p7+ffmcfmb*hbcm2*p1p3*p2p8*p6p7-hbcm2*p1p2*
     . p1p8*p6p7-hbcm2*p1p2*p2p8*p6p7+hbcm2*p1p8*p2p4*p6p7+hbcm2*p2p4
     . *p2p8*p6p7))+w12*(16*(p6p7*p2p3*p1p8)*(-fmb*hbcm-fmc*hbcm+
     . hbcm2)+8*(p6p7*p1p8)*(-ffmcfmb**2*hbcm4+fmc2*hbcm2)+16*(p3p7*
     . p3p6*p1p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(ffmcfmb*
     . hbcm2*p1p3*p1p8*p6p7-hbcm2*p1p2*p1p8*p6p7+hbcm2*p1p8*p2p4*p6p7
     . ))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(w2*(4*p5p8*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+8*(ffmcfmb*
     . hbcm*p1p3*p5p8-2*hbcm*p1p2*p5p8+2*hbcm*p2p3*p5p8+3*hbcm*p2p5*
     . p5p8))+w17*(4*p2p8*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+4*p1p8*(-
     . ffmcfmb**2*hbcm3+fmc2*hbcm)+8*(ffmcfmb*hbcm*p1p3*p1p8+ffmcfmb*
     . hbcm*p1p3*p2p8-2*hbcm*p1p2*p1p8-2*hbcm*p1p2*p2p8+2*hbcm*p1p8*
     . p2p3+3*hbcm*p1p8*p2p5+2*hbcm*p2p3*p2p8+3*hbcm*p2p5*p2p8))+w12*
     . (4*p1p8*(-ffmcfmb**2*hbcm3+fmc2*hbcm)+8*(ffmcfmb*hbcm*p1p3*
     . p1p8-2*hbcm*p1p2*p1p8+2*hbcm*p1p8*p2p3+3*hbcm*p1p8*p2p5))+(8*
     . p2p8*(-6*fb3*fmb+2*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)
     . +4*hbcm*p1p8))
      b(12)=16*ccc*(fb3*ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*p1p3-fb3*fmc2+
     . 3*fb3*p1p2-3*fb3*p2p3-4*fb3*p2p5)
      b(13)=ccc*(w2*(8*(p5p8*p2p3)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm
     . )+8*(-hbcm2*p1p2*p5p8+2*hbcm2*p2p5*p5p8))+w17*(8*(p2p8*p2p3)*(
     . ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+8*(p2p3*p1p8)*(ffmcfmb*
     . hbcm2+2*fmb*hbcm+fmc*hbcm)+8*(-hbcm2*p1p2*p1p8-hbcm2*p1p2*p2p8
     . +2*hbcm2*p1p8*p2p5+2*hbcm2*p2p5*p2p8))+w12*(8*(p2p3*p1p8)*(
     . ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+8*(-hbcm2*p1p2*p1p8+2*hbcm2
     . *p1p8*p2p5))+(4*p2p8*(-4*fb3*ffmcfmb*hbcm2+12*fb3*hbcm2+8*fb4*
     . fmb*hbcm+4*fb4*fmc*hbcm+hbcm2)+32*(-fb3*p1p3*p2p8+fb3*p1p8*
     . p2p3-3*fb3*p2p3*p5p8+3*fb3*p2p8*p3p5)))
      b(14)=ccc*(8*p2p3*(6*fb3*fmb-2*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)
     . +8*(-fb4*ffmcfmb**2*hbcm3+2*fb4*ffmcfmb*hbcm*p1p3+fb4*fmc2*
     . hbcm-4*fb4*hbcm*p1p2+6*fb4*hbcm*p2p5))
      b(15)=ccc*(w2*(8*(p5p8*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm)+16*hbcm2*
     . p4p6*p5p8)+w17*(8*(p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p3p6
     . *p1p8)*(ffmcfmb*hbcm2+fmb*hbcm)+16*(hbcm2*p1p8*p4p6+hbcm2*p2p8
     . *p4p6))+w12*(8*(p3p6*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm)+16*hbcm2*
     . p1p8*p4p6)+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb
     . *hbcm+hbcm2)+32*(fb3*p1p3*p6p8-fb3*p1p8*p3p6+fb3*p2p3*p6p8-fb3
     . *p2p8*p3p6-fb3*p3p4*p6p8+fb3*p3p6*p4p8)))
      b(16)=ccc*(w2*(8*(p5p8*p3p6)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p4p6*
     . p5p8)+w17*(8*(p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)+8*(p3p6*p1p8)*(-
     . ffmcfmb*hbcm-hbcm)+24*(-hbcm*p1p8*p4p6-hbcm*p2p8*p4p6))+w12*(8
     . *(p3p6*p1p8)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p1p8*p4p6)+8*p6p8*(2
     . *fb3*fmb+2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm))
      b(17)=ccc*(w2*(4*p5p8*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*
     . hbcm3)+8*(hbcm*p1p3*p5p8+hbcm*p2p3*p5p8-3*hbcm*p3p5*p5p8))+w17
     . *(4*p2p8*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*hbcm3)+4*p1p8*
     . (ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*hbcm3)+8*(hbcm*p1p3*
     . p1p8+hbcm*p1p3*p2p8+hbcm*p1p8*p2p3-3*hbcm*p1p8*p3p5+hbcm*p2p3*
     . p2p8-3*hbcm*p2p8*p3p5))+w12*(4*p1p8*(ffmcfmb*hbcm3-2*fmb*hbcm2
     . +fmc*hbcm2-3*hbcm3)+8*(hbcm*p1p3*p1p8+hbcm*p1p8*p2p3-3*hbcm*
     . p1p8*p3p5))+16*(fb4*hbcm*p1p8+fb4*hbcm*p2p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w2*(8*(p5p8*p3p7)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*
     . hbcm)-16*hbcm2*p5p7*p5p8)+w17*(8*(p3p7*p2p8)*(-ffmcfmb*hbcm2-2
     . *fmb*hbcm-fmc*hbcm)+8*(p3p7*p1p8)*(-ffmcfmb*hbcm2-2*fmb*hbcm-
     . fmc*hbcm)+16*(-hbcm2*p1p8*p5p7-hbcm2*p2p8*p5p7))+w12*(8*(p3p7*
     . p1p8)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)-16*hbcm2*p1p8*p5p7)
     . +(4*p7p8*(4*fb3*ffmcfmb*hbcm2-12*fb3*hbcm2-8*fb4*fmb*hbcm-4*
     . fb4*fmc*hbcm-hbcm2)+32*(fb3*p1p3*p7p8-fb3*p1p8*p3p7+fb3*p2p3*
     . p7p8-fb3*p2p8*p3p7-3*fb3*p3p5*p7p8+3*fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w2*p5p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*
     . hbcm2)+w17*(4*p2p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*
     . hbcm2)+4*p1p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2))+
     . 4*w12*p1p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(
     . -fb3*p1p8-fb3*p2p8+4*fb3*p5p8))
      b(20)=ccc*(w2*(8*(p5p8*p3p7)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p5p7
     . *p5p8)+w17*(8*(p3p7*p2p8)*(ffmcfmb*hbcm-2*hbcm)+8*(p3p7*p1p8)*
     . (ffmcfmb*hbcm-2*hbcm)+24*(-hbcm*p1p8*p5p7-hbcm*p2p8*p5p7))+w12
     . *(8*(p3p7*p1p8)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p1p8*p5p7)+4*
     . p7p8*(12*fb3*fmb-4*fb3*fmc+8*fb4*ffmcfmb*hbcm-8*fb4*hbcm-hbcm)
     . )
      b(21)=2*ccc*(16*fb3*fmb-4*fb3*fmc+12*fb4*ffmcfmb*hbcm-12*fb4*
     . hbcm-3*hbcm)
      b(22)=4*ccc*(4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+4*fb3*p1p3+4*fb3*
     . p2p3-16*fb3*p3p5-6*fb4*fmb*hbcm-2*fb4*fmc*hbcm-hbcm2)
      b(23)=ccc*(8*p3p7*(-6*fb3*fmb+2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm
     . )-48*fb4*hbcm*p5p7)
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb-8*fb4*ffmcfmb*hbcm+hbcm)-48*fb4*
     . hbcm*p4p6)
      b(25)=4*ccc*p3p7*(-8*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc-4*fb4*
     . ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm+ffmcfmb*hbcm)
      b(27)=ccc*(32*(p3p7*p3p6)*(-2*fb3*ffmcfmb-fb3)+4*p6p7*(4*fb3*
     . ffmcfmb*hbcm2+12*fb3*hbcm2-8*fb4*fmb*hbcm-4*fb4*fmc*hbcm-hbcm2
     . )+32*(-2*fb3*p1p3*p6p7-2*fb3*p2p3*p6p7+3*fb3*p3p4*p6p7-fb3*
     . p3p6*p4p7-3*fb3*p3p7*p4p6))
      b(28)=ccc*(8*w2*(p6p7*p5p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-
     . hbcm2)+w17*(8*(p6p7*p2p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-
     . hbcm2)+8*(p6p7*p1p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)
     . )+8*w12*(p6p7*p1p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+
     . (32*(p7p8*p3p6)*(-fb3*ffmcfmb-fb3)+32*(-fb3*ffmcfmb*p3p7*p6p8-
     . 2*fb3*p1p8*p6p7-2*fb3*p2p8*p6p7-3*fb3*p4p6*p7p8-fb3*p4p7*p6p8+
     . 3*fb3*p4p8*p6p7)))
      b(29)=ccc*(16*p3p6*(-2*fb3*ffmcfmb-fb3)-64*fb3*p4p6)
      b(30)=ccc*(16*p3p7*(2*fb3*ffmcfmb-3*fb3)-64*fb3*p5p7)
      ans=ccc*(w2*(8*(p6p7*p5p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+16*(
     . p5p8*p3p7*p3p6)*(-ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p3*p5p8*p6p7-
     . hbcm*p2p3*p5p8*p6p7+2*hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8-
     . 2*hbcm*p3p7*p4p6*p5p8))+w17*(8*(p6p7*p2p8)*(-fmb*hbcm2+fmc*
     . hbcm2+2*hbcm3)+8*(p6p7*p1p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+16
     . *(p3p7*p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p1p8)*(-
     . ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p3*p1p8*p6p7-hbcm*p1p3*p2p8*
     . p6p7-hbcm*p1p8*p2p3*p6p7+2*hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*
     . p4p7-2*hbcm*p1p8*p3p7*p4p6-hbcm*p2p3*p2p8*p6p7+2*hbcm*p2p8*
     . p3p4*p6p7-hbcm*p2p8*p3p6*p4p7-2*hbcm*p2p8*p3p7*p4p6))+w12*(8*(
     . p6p7*p1p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+16*(p3p7*p3p6*p1p8)*
     . (-ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p3*p1p8*p6p7-hbcm*p1p8*p2p3*
     . p6p7+2*hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p4p7-2*hbcm*p1p8*
     . p3p7*p4p6))+(8*(p6p8*p3p7)*(4*fb3*fmb-4*fb4*hbcm-hbcm)+32*(
     . p7p8*p3p6)*(-fb3*fmb-fb4*ffmcfmb*hbcm)+32*(-fb4*hbcm*p1p8*p6p7
     . -fb4*hbcm*p2p8*p6p7-2*fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*p6p8+2*
     . fb4*hbcm*p4p8*p6p7)))
      b(31)=ans
      b(32)=8*ccc*p6p7*(6*fb3*fmb-2*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp16_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*ffmcfmb*p2p3-
     . fmc2+2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(1)=ccc*(4*p6p7*(4*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*hbcm2+4*
     . fb3*fmb*fmc+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*
     . fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+hbcm2)+32*(-fb3*ffmcfmb*p2p3*
     . p6p7+fb3*ffmcfmb*p3p6*p3p7+2*fb3*ffmcfmb*p3p7*p4p6+fb3*p1p2*
     . p6p7-fb3*p2p4*p6p7))
      ans2=w17*(8*(p6p7*p2p8)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+8*(
     . p6p7*p1p8)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6*p2p8
     . )*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(p3p7*p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(-ffmcfmb*hbcm*p1p8*p2p3*p6p7
     . +ffmcfmb*hbcm*p1p8*p3p7*p4p6-ffmcfmb*hbcm*p2p3*p2p8*p6p7+
     . ffmcfmb*hbcm*p2p8*p3p7*p4p6+hbcm*p1p2*p1p8*p6p7+hbcm*p1p2*p2p8
     . *p6p7-hbcm*p1p8*p2p4*p6p7-hbcm*p2p4*p2p8*p6p7))+(8*(p6p7*p4p8)
     . *(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p8*p4p7)*(4*fb3*fmc
     . +4*fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p4p6)*(8*fb3*fmc+8*fb4*
     . ffmcfmb*hbcm-hbcm)+8*(p7p8*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm
     . -ffmcfmb*hbcm)+16*(p6p7*p2p8)*(2*fb3*fmc+2*fb4*ffmcfmb*hbcm-
     . hbcm)+8*(p6p7*p1p8)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(
     . p6p8*p3p7)*(-4*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc+4*fb4*ffmcfmb
     . *hbcm-ffmcfmb*hbcm))
      ans1=w2*(8*(p6p7*p5p8)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2)+16*(
     . p5p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+16*(-ffmcfmb*
     . hbcm*p2p3*p5p8*p6p7+ffmcfmb*hbcm*p3p7*p4p6*p5p8+hbcm*p1p2*p5p8
     . *p6p7-hbcm*p2p4*p5p8*p6p7))+w7*(8*(p6p7*p2p8)*(ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc*hbcm2)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+
     . ffmcfmb*hbcm)+16*(-ffmcfmb*hbcm*p2p3*p2p8*p6p7+ffmcfmb*hbcm*
     . p2p8*p3p7*p4p6+hbcm*p1p2*p2p8*p6p7-hbcm*p2p4*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w2*(16*(p5p8*p5p7)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p8*
     . p3p7)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*
     . hbcm))+w7*(16*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p7*
     . p2p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*
     . hbcm))+w17*(16*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p5p7*
     . p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p7*p2p8)*(ffmcfmb*fmb*hbcm
     . +ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p7*p1p8)*(
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+(32
     . *(p7p8*p2p3)*(-fb3*ffmcfmb+2*fb3)+4*p7p8*(4*fb3*ffmcfmb**2*
     . hbcm2-4*fb3*ffmcfmb*hbcm2+8*fb3*fmb*fmc+4*fb3*fmc2+8*fb4*
     . ffmcfmb*fmb*hbcm+8*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc*hbcm-fmb*
     . hbcm-fmc*hbcm+hbcm2)+32*(fb3*ffmcfmb*p1p8*p3p7-2*fb3*ffmcfmb*
     . p3p7*p5p8-fb3*p1p2*p7p8+3*fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8
     . *p5p7)))
      b(3)=ans
      b(4)=ccc*(8*p5p7*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+8*
     . p3p7*(-2*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*
     . fmc-2*fb3*fmc2-2*fb4*ffmcfmb*fmb*hbcm-2*fb4*ffmcfmb*fmc*hbcm+
     . fmb*hbcm+fmc*hbcm)+32*(-fb3*ffmcfmb*p1p3*p3p7-fb3*ffmcfmb*p2p3
     . *p3p7+2*fb3*ffmcfmb*p3p5*p3p7+fb3*p1p2*p3p7+3*fb3*p2p3*p5p7-3*
     . fb3*p2p5*p3p7))
      ans3=(16*(p5p8*p3p7)*(-4*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm)+8*(
     . p7p8*p3p5)*(8*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+8*(p3p7*p1p8)*(
     . 4*fb3*fmc-hbcm)+32*(p7p8*p1p3)*(-fb3*fmc-fb4*ffmcfmb*hbcm)+16*
     . (p3p7*p2p8)*(4*fb3*fmb+2*fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p2p3)*
     . (-8*fb3*fmb-8*fb4*ffmcfmb*hbcm+hbcm)+4*p7p8*(4*fb3*ffmcfmb*fmb
     . *hbcm2-4*fb3*ffmcfmb*fmc*hbcm2+8*fb3*fmc*hbcm2+8*fb4*ffmcfmb*
     . hbcm3+4*fb4*fmb*fmc*hbcm-4*fb4*fmc2*hbcm-fmb*hbcm2+fmc*hbcm2-
     . hbcm3)+32*(fb4*hbcm*p1p2*p7p8-2*fb4*hbcm*p2p5*p7p8+2*fb4*hbcm*
     . p2p8*p5p7))
      ans2=w17*(8*(p5p7*p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p5p7*p1p8
     . )*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p7*p2p8)*(-ffmcfmb*fmb*hbcm2
     . +ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc2*hbcm)+8*(
     . p3p7*p1p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3
     . +fmb*fmc*hbcm-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p1p8*p3p7-
     . ffmcfmb*hbcm*p1p3*p2p8*p3p7-ffmcfmb*hbcm*p1p8*p2p3*p3p7+
     . ffmcfmb*hbcm*p1p8*p3p5*p3p7-ffmcfmb*hbcm*p2p3*p2p8*p3p7+
     . ffmcfmb*hbcm*p2p8*p3p5*p3p7+hbcm*p1p2*p1p8*p3p7+hbcm*p1p2*p2p8
     . *p3p7+2*hbcm*p1p8*p2p3*p5p7-2*hbcm*p1p8*p2p5*p3p7+2*hbcm*p2p3*
     . p2p8*p5p7-2*hbcm*p2p5*p2p8*p3p7))+ans3
      ans1=w2*(8*(p5p8*p5p7)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p5p8*p3p7)
     . *(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p3p7*p5p8-ffmcfmb*hbcm*
     . p2p3*p3p7*p5p8+ffmcfmb*hbcm*p3p5*p3p7*p5p8+hbcm*p1p2*p3p7*p5p8
     . +2*hbcm*p2p3*p5p7*p5p8-2*hbcm*p2p5*p3p7*p5p8))+w7*(8*(p5p7*
     . p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p7*p2p8)*(-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc2*hbcm)+
     . 16*(-ffmcfmb*hbcm*p1p3*p2p8*p3p7-ffmcfmb*hbcm*p2p3*p2p8*p3p7+
     . ffmcfmb*hbcm*p2p8*p3p5*p3p7+hbcm*p1p2*p2p8*p3p7+2*hbcm*p2p3*
     . p2p8*p5p7-2*hbcm*p2p5*p2p8*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w2*(16*(p5p8*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p8*
     . p3p6)*(ffmcfmb*hbcm2-fmc*hbcm))+w7*(16*(p4p6*p2p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*(p3p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm))+w17*(
     . 16*(p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p6*p2p8)*(ffmcfmb
     . *hbcm2-fmc*hbcm)+16*(p4p6*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(
     . p3p6*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm))+(32*(p3p6*p2p8)*(-fb3*
     . ffmcfmb-fb3)+4*p6p8*(4*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*
     . hbcm2+4*fb3*fmb*fmc+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*
     . hbcm-4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+hbcm2)+32*(-fb3*ffmcfmb*
     . p2p3*p6p8+fb3*p1p2*p6p8-fb3*p2p4*p6p8-3*fb3*p2p8*p4p6)))
      b(7)=ccc*(32*(p3p6*p2p3)*(2*fb3*ffmcfmb+fb3)+8*p4p6*(-4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+4*p3p6*(-4*fb3*ffmcfmb**2*
     . hbcm2-4*fb3*fmb*fmc-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*
     . hbcm+fmb*hbcm+fmc*hbcm)+32*(-fb3*p1p2*p3p6+3*fb3*p2p3*p4p6+fb3
     . *p2p4*p3p6))
      ans2=(8*(p4p8*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p6p8
     . *p3p4)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p8)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p8*p1p3)*(4*fb3*fmc+4*
     . fb4*ffmcfmb*hbcm-hbcm)+8*(p3p6*p2p8)*(4*fb3*fmb-4*fb3*fmc+hbcm
     . )+32*(p6p8*p2p3)*(-fb3*fmb+fb3*fmc+fb4*ffmcfmb*hbcm+fb4*hbcm)+
     . 4*p6p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-4*fb3*
     . fmc*hbcm2-4*fb4*ffmcfmb**2*hbcm3-4*fb4*ffmcfmb*hbcm3+4*fb4*fmb
     . *fmc*hbcm+ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+32*(-fb4*hbcm*p1p2*
     . p6p8+fb4*hbcm*p2p4*p6p8+2*fb4*hbcm*p2p8*p4p6))
      ans1=w2*(8*(p5p8*p4p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(p5p8*p3p6
     . *p2p3)*(ffmcfmb*hbcm+hbcm)+8*(p5p8*p3p6)*(-ffmcfmb*fmb*hbcm2+
     . fmb*fmc*hbcm)+16*(-hbcm*p1p2*p3p6*p5p8+2*hbcm*p2p3*p4p6*p5p8+
     . hbcm*p2p4*p3p6*p5p8))+w7*(8*(p4p6*p2p8)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+16*(p3p6*p2p8*p2p3)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8)*(
     . -ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+16*(-hbcm*p1p2*p2p8*p3p6+2*
     . hbcm*p2p3*p2p8*p4p6+hbcm*p2p4*p2p8*p3p6))+w17*(8*(p4p6*p2p8)*(
     . -ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p6*p1p8)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+16*(p3p6*p2p8*p2p3)*(ffmcfmb*hbcm+hbcm)+16*(p3p6*p2p3*
     . p1p8)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8)*(-ffmcfmb*fmb*hbcm2+
     . fmb*fmc*hbcm)+8*(p3p6*p1p8)*(-ffmcfmb*fmb*hbcm2+fmb*fmc*hbcm)+
     . 16*(-hbcm*p1p2*p1p8*p3p6-hbcm*p1p2*p2p8*p3p6+2*hbcm*p1p8*p2p3*
     . p4p6+hbcm*p1p8*p2p4*p3p6+2*hbcm*p2p3*p2p8*p4p6+hbcm*p2p4*p2p8*
     . p3p6))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(16*(p4p6*p3p7)*(-4*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm)+8
     . *(p4p7*p3p6)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p7*p3p4
     . )*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p3)*(-4*fb3*
     . fmc-8*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p7*p1p3)*(-4*fb3*fmc-4*fb4*
     . ffmcfmb*hbcm+hbcm)+8*(p3p7*p3p6)*(4*fb3*ffmcfmb*fmb-4*fb3*
     . ffmcfmb*fmc-4*fb3*fmc-4*fb4*ffmcfmb**2*hbcm-4*fb4*ffmcfmb*hbcm
     . +hbcm)+4*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*
     . hbcm2+4*fb3*fmc*hbcm2+4*fb4*ffmcfmb**2*hbcm3+4*fb4*ffmcfmb*
     . hbcm3-4*fb4*fmb*fmc*hbcm+fmb*hbcm2-fmc*hbcm2-hbcm3)+32*(fb4*
     . hbcm*p1p2*p6p7-fb4*hbcm*p2p4*p6p7))
      ans3=(64*(p3p7*p3p6*p2p8)*(3*fb3*ffmcfmb+fb3)+8*(p6p7*p4p8)*(4*
     . fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p8*p4p7)*(-4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+8*(p7p8*p4p6)*(4*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p8*p3p7)*(4*fb3*
     . ffmcfmb*hbcm2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p2p8)*(-2*fb3*ffmcfmb*hbcm2-2
     . *fb4*fmc*hbcm+hbcm2)+8*(p6p7*p1p8)*(-4*fb3*ffmcfmb*hbcm2-4*fb4
     . *fmc*hbcm+hbcm2)+8*(p7p8*p3p6)*(4*fb3*ffmcfmb**2*hbcm2+4*fb4*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+64*(-fb3*
     . ffmcfmb*p1p3*p3p7*p6p8+fb3*ffmcfmb*p1p8*p3p6*p3p7-fb3*ffmcfmb*
     . p2p3*p3p6*p7p8+fb3*ffmcfmb*p3p4*p3p7*p6p8-fb3*ffmcfmb*p3p6*
     . p3p7*p4p8+fb3*p1p8*p2p3*p6p7+fb3*p2p3*p2p8*p6p7-fb3*p2p3*p4p6*
     . p7p8+fb3*p2p3*p4p7*p6p8-fb3*p2p3*p4p8*p6p7+4*fb3*p2p8*p3p7*
     . p4p6))
      ans2=w17*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p4p7*p3p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p3p4*p2p8)*(
     . -ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p2p8*p2p3)*(ffmcfmb*hbcm2-
     . fmb*hbcm-fmc*hbcm+hbcm2)+16*(p4p6*p3p7*p1p8)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p4p7*p3p6*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p6p7*p3p4*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p2p3*p1p8)*
     . (ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm+hbcm2)+16*(p6p7*p2p8*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p1p8*p1p3)*(ffmcfmb*hbcm2-fmc
     . *hbcm)+8*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+8*(
     . p6p7*p1p8)*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p7*p3p6*
     . p2p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(p3p7*p3p6*p1p8)*
     . (ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(-hbcm2*p1p2*p1p8*p6p7-
     . hbcm2*p1p2*p2p8*p6p7+hbcm2*p1p8*p2p4*p6p7+hbcm2*p2p4*p2p8*p6p7
     . ))+ans3
      ans1=w2*(16*(p5p8*p4p6*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p8
     . *p4p7*p3p6)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p5p8*p3p4)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p5p8*p2p3)*(ffmcfmb*hbcm2-fmb
     . *hbcm-fmc*hbcm+hbcm2)+16*(p6p7*p5p8*p1p3)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+8*(p6p7*p5p8)*(-ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(
     . p5p8*p3p7*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm)+16*(-hbcm2
     . *p1p2*p5p8*p6p7+hbcm2*p2p4*p5p8*p6p7))+w7*(16*(p4p6*p3p7*p2p8)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p7*p3p6*p2p8)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+16*(p6p7*p3p4*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p6p7*p2p8*p2p3)*(ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm+hbcm2)+16*(
     . p6p7*p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p2p8)*(-
     . ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-ffmcfmb*
     . hbcm4-fmb*fmc*hbcm2+fmc*hbcm3)+16*(p3p7*p3p6*p2p8)*(ffmcfmb**2
     . *hbcm2-ffmcfmb*fmb*hbcm)+16*(-hbcm2*p1p2*p2p8*p6p7+hbcm2*p2p4*
     . p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w2*(8*(p5p8*p2p3)*(-ffmcfmb*hbcm+2*hbcm)+4*p5p8*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+8*(-hbcm*
     . p1p2*p5p8+3*hbcm*p2p5*p5p8))+w7*(8*(p2p8*p2p3)*(-ffmcfmb*hbcm+
     . 2*hbcm)+4*p2p8*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb
     . *fmc*hbcm2-ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+8
     . *(-hbcm*p1p2*p2p8+3*hbcm*p2p5*p2p8))+w17*(8*(p2p8*p2p3)*(-
     . ffmcfmb*hbcm+2*hbcm)+8*(p2p3*p1p8)*(-ffmcfmb*hbcm+2*hbcm)+4*
     . p2p8*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2
     . -ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*p1p8*(
     . ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+8*(-hbcm*
     . p1p2*p1p8-hbcm*p1p2*p2p8+3*hbcm*p1p8*p2p5+3*hbcm*p2p5*p2p8))+(
     . 12*p5p8*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p1p8*(-4*fb3*fmc
     . -4*fb4*ffmcfmb*hbcm+hbcm)+8*p2p8*(-6*fb3*fmb-6*fb4*ffmcfmb*
     . hbcm+4*fb4*hbcm+hbcm)))
      b(11)=ans
      b(12)=ccc*(16*p2p3*(2*fb3*ffmcfmb-3*fb3)+2*(-8*fb3*ffmcfmb**2*
     . hbcm2+8*fb3*ffmcfmb*hbcm2-12*fb3*fmb*fmc-4*fb3*fmc2+8*fb3*p1p2
     . -32*fb3*p2p5-12*fb4*ffmcfmb*fmb*hbcm-12*fb4*ffmcfmb*fmc*hbcm+8
     . *fb4*fmc*hbcm+3*fmb*hbcm+3*fmc*hbcm-2*hbcm2))
      ans2=w17*(16*(p3p5*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p2p8*
     . p2p3)*(ffmcfmb*hbcm2+fmb*hbcm)+16*(p3p5*p1p8)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p2p3*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p2p8*p1p3
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p1p8*p1p3)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*p2p8*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*ffmcfmb*
     . hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+fmc2*hbcm2)+4*p1p8*(-ffmcfmb*
     . fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*
     . fmc*hbcm3+fmc2*hbcm2)+8*(-hbcm2*p1p2*p1p8-hbcm2*p1p2*p2p8+2*
     . hbcm2*p1p8*p2p5+2*hbcm2*p2p5*p2p8))+(8*p5p8*(4*fb3*ffmcfmb*
     . hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*p2p8*(-4*fb3*ffmcfmb*hbcm2+6*fb3
     . *hbcm2+4*fb4*fmb*hbcm+hbcm2)+4*p1p8*(-4*fb3*ffmcfmb*hbcm2-4*
     . fb4*fmc*hbcm+hbcm2)+32*(-fb3*p1p3*p2p8+fb3*p1p8*p2p3-3*fb3*
     . p2p3*p5p8+3*fb3*p2p8*p3p5))
      ans1=w2*(16*(p5p8*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p8*p2p3
     . )*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p5p8*p1p3)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*p5p8*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*ffmcfmb*
     . hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+fmc2*hbcm2)+8*(-hbcm2*p1p2*
     . p5p8+2*hbcm2*p2p5*p5p8))+w7*(16*(p3p5*p2p8)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p2p8*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p2p8*p1p3
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+4*p2p8*(-ffmcfmb*fmb*hbcm3-ffmcfmb*
     . fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+fmc2*hbcm2
     . )+8*(-hbcm2*p1p2*p2p8+2*hbcm2*p2p5*p2p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(12*p3p5*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p1p3*
     . (4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p2p3*(12*fb3*fmb+8*fb4*
     . ffmcfmb*hbcm-3*hbcm)+2*(-8*fb3*ffmcfmb*fmb*hbcm2+8*fb3*ffmcfmb
     . *fmc*hbcm2-12*fb3*fmc*hbcm2+4*fb4*ffmcfmb**2*hbcm3-12*fb4*
     . ffmcfmb*hbcm3-8*fb4*fmb*fmc*hbcm+4*fb4*fmc2*hbcm-8*fb4*hbcm*
     . p1p2+24*fb4*hbcm*p2p5+2*fmb*hbcm2-2*fmc*hbcm2+3*hbcm3))
      b(15)=ccc*(w2*(8*(p5p8*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm)+16*hbcm2*
     . p4p6*p5p8)+w7*(8*(p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm)+16*hbcm2
     . *p2p8*p4p6)+w17*(8*(p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(
     . p3p6*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm)+16*(hbcm2*p1p8*p4p6+hbcm2*
     . p2p8*p4p6))+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*
     . fmb*hbcm+hbcm2)+32*(fb3*p1p3*p6p8-fb3*p1p8*p3p6+fb3*p2p3*p6p8-
     . fb3*p2p8*p3p6-fb3*p3p4*p6p8+fb3*p3p6*p4p8)))
      b(16)=ccc*(w2*(8*(p5p8*p3p6)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p4p6*
     . p5p8)+w7*(8*(p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p2p8*p4p6
     . )+w17*(8*(p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)+8*(p3p6*p1p8)*(-
     . ffmcfmb*hbcm-hbcm)+24*(-hbcm*p1p8*p4p6-hbcm*p2p8*p4p6))+8*p6p8
     . *(2*fb3*fmb+2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm))
      b(17)=ccc*(w2*(4*p5p8*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*
     . hbcm3)+8*(hbcm*p1p3*p5p8+hbcm*p2p3*p5p8-3*hbcm*p3p5*p5p8))+w7*
     . (4*p2p8*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*hbcm3)+8*(hbcm*
     . p1p3*p2p8+hbcm*p2p3*p2p8-3*hbcm*p2p8*p3p5))+w17*(4*p2p8*(
     . ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*hbcm3)+4*p1p8*(ffmcfmb*
     . hbcm3-2*fmb*hbcm2+fmc*hbcm2-3*hbcm3)+8*(hbcm*p1p3*p1p8+hbcm*
     . p1p3*p2p8+hbcm*p1p8*p2p3-3*hbcm*p1p8*p3p5+hbcm*p2p3*p2p8-3*
     . hbcm*p2p8*p3p5))+16*(fb4*hbcm*p1p8+fb4*hbcm*p2p8-3*fb4*hbcm*
     . p5p8))
      b(18)=ccc*(w2*(8*(p5p8*p3p7)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*
     . hbcm)-16*hbcm2*p5p7*p5p8)+w7*(8*(p3p7*p2p8)*(-ffmcfmb*hbcm2-2*
     . fmb*hbcm-fmc*hbcm)-16*hbcm2*p2p8*p5p7)+w17*(8*(p3p7*p2p8)*(-
     . ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+8*(p3p7*p1p8)*(-ffmcfmb*
     . hbcm2-2*fmb*hbcm-fmc*hbcm)+16*(-hbcm2*p1p8*p5p7-hbcm2*p2p8*
     . p5p7))+(4*p7p8*(4*fb3*ffmcfmb*hbcm2-12*fb3*hbcm2-8*fb4*fmb*
     . hbcm-4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p1p3*p7p8-fb3*p1p8*p3p7+fb3
     . *p2p3*p7p8-fb3*p2p8*p3p7-3*fb3*p3p5*p7p8+3*fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w2*p5p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*
     . hbcm2)+4*w7*p2p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2
     . )+w17*(4*p2p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2)+4
     . *p1p8*(-2*ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm+2*hbcm2))+16*(-fb3
     . *p1p8-fb3*p2p8+4*fb3*p5p8))
      b(20)=ccc*(w2*(8*(p5p8*p3p7)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p5p7
     . *p5p8)+w7*(8*(p3p7*p2p8)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p2p8*
     . p5p7)+w17*(8*(p3p7*p2p8)*(ffmcfmb*hbcm-2*hbcm)+8*(p3p7*p1p8)*(
     . ffmcfmb*hbcm-2*hbcm)+24*(-hbcm*p1p8*p5p7-hbcm*p2p8*p5p7))+4*
     . p7p8*(12*fb3*fmb-4*fb3*fmc+8*fb4*ffmcfmb*hbcm-8*fb4*hbcm-hbcm)
     . )
      b(21)=2*ccc*(16*fb3*fmb-4*fb3*fmc+12*fb4*ffmcfmb*hbcm-12*fb4*
     . hbcm-3*hbcm)
      b(22)=4*ccc*(4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+4*fb3*p1p3+4*fb3*
     . p2p3-16*fb3*p3p5-6*fb4*fmb*hbcm-2*fb4*fmc*hbcm-hbcm2)
      b(23)=ccc*(8*p3p7*(-6*fb3*fmb+2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm
     . )-48*fb4*hbcm*p5p7)
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb-8*fb4*ffmcfmb*hbcm+hbcm)-48*fb4*
     . hbcm*p4p6)
      b(25)=ccc*(12*p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*p3p7*
     . (-4*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc-4*fb3*fmc-2*fb4*ffmcfmb*
     . hbcm+hbcm))
      b(26)=ccc*(12*p4p6*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p3p6*
     . (-4*fb3*ffmcfmb*fmc-4*fb3*fmc-4*fb4*ffmcfmb**2*hbcm-4*fb4*
     . ffmcfmb*hbcm+ffmcfmb*hbcm+hbcm))
      b(27)=ccc*(32*(p3p7*p3p6)*(-2*fb3*ffmcfmb-fb3)+4*p6p7*(4*fb3*
     . ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+32*(-fb3*p1p3*
     . p6p7-fb3*p2p3*p6p7+fb3*p3p4*p6p7-fb3*p3p6*p4p7-3*fb3*p3p7*p4p6
     . ))
      b(28)=ccc*(8*w2*(p6p7*p5p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*w7
     . *(p6p7*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+w17*(8*(p6p7*p2p8)
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p6p7*p1p8)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2))+(32*(p7p8*p3p6)*(-fb3*ffmcfmb-fb3)+32*(-fb3*
     . ffmcfmb*p3p7*p6p8-fb3*p1p8*p6p7-fb3*p2p8*p6p7-3*fb3*p4p6*p7p8-
     . fb3*p4p7*p6p8+fb3*p4p8*p6p7)))
      b(29)=ccc*(16*p3p6*(-2*fb3*ffmcfmb-fb3)-64*fb3*p4p6)
      b(30)=ccc*(16*p3p7*(2*fb3*ffmcfmb-3*fb3)-64*fb3*p5p7)
      ans=ccc*(w2*(8*(p6p7*p5p8)*(ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p5p8*p3p7*p3p6)*(-ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p3*p5p8*p6p7-
     . hbcm*p2p3*p5p8*p6p7+hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8-2*
     . hbcm*p3p7*p4p6*p5p8))+w7*(8*(p6p7*p2p8)*(ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)+16*(-
     . hbcm*p1p3*p2p8*p6p7-hbcm*p2p3*p2p8*p6p7+hbcm*p2p8*p3p4*p6p7-
     . hbcm*p2p8*p3p6*p4p7-2*hbcm*p2p8*p3p7*p4p6))+w17*(8*(p6p7*p2p8)
     . *(ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*p1p8)*(ffmcfmb*hbcm3-
     . fmb*hbcm2+hbcm3)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)+16*(
     . p3p7*p3p6*p1p8)*(-ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p3*p1p8*p6p7-
     . hbcm*p1p3*p2p8*p6p7-hbcm*p1p8*p2p3*p6p7+hbcm*p1p8*p3p4*p6p7-
     . hbcm*p1p8*p3p6*p4p7-2*hbcm*p1p8*p3p7*p4p6-hbcm*p2p3*p2p8*p6p7+
     . hbcm*p2p8*p3p4*p6p7-hbcm*p2p8*p3p6*p4p7-2*hbcm*p2p8*p3p7*p4p6)
     . )+(8*(p6p8*p3p7)*(4*fb3*fmb-4*fb4*hbcm-hbcm)+32*(p7p8*p3p6)*(-
     . fb3*fmb-fb4*ffmcfmb*hbcm)+32*(-fb4*hbcm*p1p8*p6p7-fb4*hbcm*
     . p2p8*p6p7-2*fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*p6p8+fb4*hbcm*
     . p4p8*p6p7)))
      b(31)=ans
      b(32)=4*ccc*p6p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp15_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(1)=32*ccc*(fb3*p3p6*p5p7+2*fb3*p4p6*p5p7)
      b(2)=ccc*(w16*(16*(p5p8*p5p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(
     . p5p7*p3p6*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(-hbcm*p2p8*p4p6*p5p7+
     . hbcm*p4p6*p5p7*p5p8))+w12*(16*(p5p7*p3p6*p1p8)*(-ffmcfmb*hbcm+
     . hbcm)+16*hbcm*p1p8*p4p6*p5p7)+32*(p6p8*p5p7)*(-fb3*fmb-fb4*
     . ffmcfmb*hbcm+fb4*hbcm))
      b(3)=ccc*(w16*(8*(p5p8*p5p7)*(fmb*hbcm+fmc*hbcm)+8*(p5p7*p2p8)*
     . (-fmb*hbcm-fmc*hbcm))+8*w12*(p5p7*p1p8)*(fmb*hbcm+fmc*hbcm)+32
     . *(fb3*p1p8*p5p7+2*fb3*p2p8*p5p7-2*fb3*p5p7*p5p8))
      b(4)=ccc*(16*p5p7*(2*fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(-
     . fb3*p1p3*p5p7-2*fb3*p2p3*p5p7+2*fb3*p3p5*p5p7))
      b(5)=ccc*(w16*(8*(p5p8*p5p7)*(ffmcfmb*hbcm3-fmc*hbcm2+hbcm3)+8*
     . (p5p7*p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+16*(hbcm*p1p3*
     . p2p8*p5p7-hbcm*p1p3*p5p7*p5p8+hbcm*p2p3*p2p8*p5p7-hbcm*p2p3*
     . p5p7*p5p8-hbcm*p2p8*p3p5*p5p7+hbcm*p3p5*p5p7*p5p8))+w12*(8*(
     . p5p7*p1p8)*(ffmcfmb*hbcm3-fmc*hbcm2+hbcm3)+16*(-hbcm*p1p3*p1p8
     . *p5p7-hbcm*p1p8*p2p3*p5p7+hbcm*p1p8*p3p5*p5p7))+32*(-fb4*hbcm*
     . p1p8*p5p7-fb4*hbcm*p2p8*p5p7+fb4*hbcm*p5p7*p5p8))
      b(6)=32*ccc*(fb3*p1p2*p6p8-fb3*p2p3*p6p8-fb3*p2p4*p6p8)
      b(7)=32*ccc*(-fb3*p1p2*p3p6+fb3*p2p3*p3p6+fb3*p2p4*p3p6)
      b(8)=ccc*(16*w16*(hbcm*p1p2*p2p8*p3p6-hbcm*p1p2*p3p6*p5p8-hbcm*
     . p2p3*p2p8*p3p6+hbcm*p2p3*p3p6*p5p8-hbcm*p2p4*p2p8*p3p6+hbcm*
     . p2p4*p3p6*p5p8)+16*w12*(-hbcm*p1p2*p1p8*p3p6+hbcm*p1p8*p2p3*
     . p3p6+hbcm*p1p8*p2p4*p3p6)+32*(-fb4*hbcm*p1p2*p6p8+fb4*hbcm*
     . p2p3*p6p8+fb4*hbcm*p2p4*p6p8))
      b(9)=ccc*(8*(p5p7*p3p6)*(4*fb3*fmb-hbcm)+32*fb4*hbcm*p4p6*p5p7)
      b(10)=ccc*(w16*(16*(p5p8*p5p7*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm)+16
     . *(p5p7*p3p6*p2p8)*(-ffmcfmb*hbcm2+fmb*hbcm))+16*w12*(p5p7*p3p6
     . *p1p8)*(ffmcfmb*hbcm2-fmb*hbcm)+(8*(p6p8*p5p7)*(4*fb3*ffmcfmb*
     . hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm+hbcm2)+64*(fb3*p1p2*p3p6*p7p8
     . -fb3*p1p2*p3p7*p6p8-fb3*p1p3*p5p7*p6p8+fb3*p1p8*p3p6*p5p7-fb3*
     . p2p3*p3p6*p7p8+fb3*p2p3*p3p7*p6p8-fb3*p2p4*p3p6*p7p8+fb3*p2p4*
     . p3p7*p6p8+fb3*p2p8*p3p6*p3p7+fb3*p2p8*p3p6*p4p7+fb3*p2p8*p3p6*
     . p5p7+fb3*p3p4*p5p7*p6p8-fb3*p3p6*p4p8*p5p7)))
      b(11)=ccc*(8*w16*(-hbcm*p2p5*p2p8+hbcm*p2p5*p5p8)+8*hbcm*p1p8*
     . p2p5*w12)
      b(12)=-32*ccc*fb3*p2p5
      b(14)=16*ccc*fb4*hbcm*p2p5
      b(15)=ccc*(w16*(8*(p5p8*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm)+8*(p3p6
     . *p2p8)*(ffmcfmb*hbcm2-fmb*hbcm))+8*w12*(p3p6*p1p8)*(-ffmcfmb*
     . hbcm2+fmb*hbcm)+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*
     . fb4*fmb*hbcm-hbcm2)+32*(fb3*p1p3*p6p8-fb3*p1p8*p3p6-fb3*p3p4*
     . p6p8+fb3*p3p6*p4p8)))
      b(16)=ccc*(w16*(8*(p5p8*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p3p6*p2p8)
     . *(-ffmcfmb*hbcm+hbcm)+8*(hbcm*p2p8*p4p6-hbcm*p4p6*p5p8))+w12*(
     . 8*(p3p6*p1p8)*(ffmcfmb*hbcm-hbcm)-8*hbcm*p1p8*p4p6)+16*p6p8*(
     . fb3*fmb+fb4*ffmcfmb*hbcm-fb4*hbcm))
      b(17)=ccc*(w16*(4*p5p8*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+4*p2p8*
     . (ffmcfmb*hbcm3-fmc*hbcm2+hbcm3)+8*(-hbcm*p1p3*p2p8+hbcm*p1p3*
     . p5p8-hbcm*p2p3*p2p8+hbcm*p2p3*p5p8+hbcm*p2p8*p3p5-hbcm*p3p5*
     . p5p8))+w12*(4*p1p8*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(hbcm*
     . p1p3*p1p8+hbcm*p1p8*p2p3-hbcm*p1p8*p3p5))+16*(fb4*hbcm*p1p8+
     . fb4*hbcm*p2p8-fb4*hbcm*p5p8))
      b(19)=ccc*(w16*(4*p5p8*(-fmb*hbcm-fmc*hbcm)+4*p2p8*(fmb*hbcm+
     . fmc*hbcm))+4*w12*p1p8*(-fmb*hbcm-fmc*hbcm)+16*(-fb3*p1p8-2*fb3
     . *p2p8+2*fb3*p5p8))
      b(21)=2*ccc*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p1p3+4*fb3*p2p3-4*fb3*p3p5-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb+hbcm)-16*fb4*hbcm*p4p6)
      b(25)=4*ccc*p5p7*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4
     . *hbcm+hbcm)
      b(27)=32*ccc*(-fb3*p3p6*p3p7-fb3*p3p6*p4p7-fb3*p3p6*p5p7)
      b(28)=32*ccc*(-fb3*p3p7*p6p8-fb3*p4p7*p6p8-fb3*p5p7*p6p8)
      b(29)=16*ccc*(-fb3*p3p6-2*fb3*p4p6)
      b(31)=ccc*(16*w16*(hbcm*p2p8*p3p6*p3p7+hbcm*p2p8*p3p6*p4p7+hbcm
     . *p2p8*p3p6*p5p7-hbcm*p3p6*p3p7*p5p8-hbcm*p3p6*p4p7*p5p8-hbcm*
     . p3p6*p5p7*p5p8)+16*w12*(-hbcm*p1p8*p3p6*p3p7-hbcm*p1p8*p3p6*
     . p4p7-hbcm*p1p8*p3p6*p5p7)+32*(-fb4*hbcm*p3p7*p6p8-fb4*hbcm*
     . p4p7*p6p8-fb4*hbcm*p5p7*p6p8))
      do 200 n=1,32 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp14_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*
     . p1p3))
      b(1)=32*ccc*(fb3*p3p6*p5p7+2*fb3*p4p6*p5p7)
      b(2)=ccc*(w1*(16*(p5p7*p4p8*p3p6)*(ffmcfmb*hbcm-hbcm)-16*hbcm*
     . p4p6*p4p8*p5p7)+w13*(16*(p5p7*p3p6*p1p8)*(ffmcfmb*hbcm-hbcm)-
     . 16*hbcm*p1p8*p4p6*p5p7)+8*(p6p8*p5p7)*(-4*fb3*fmb+4*fb3*fmc-
     . hbcm))
      b(3)=ccc*(8*w1*(p5p7*p4p8)*(-fmb*hbcm-fmc*hbcm)+8*w13*(p5p7*
     . p1p8)*(-fmb*hbcm-fmc*hbcm)+32*(fb3*p1p8*p5p7+2*fb3*p2p8*p5p7-2
     . *fb3*p5p7*p5p8))
      b(4)=ccc*(16*p5p7*(2*fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(-
     . fb3*p1p3*p5p7-2*fb3*p2p3*p5p7+2*fb3*p3p5*p5p7))
      b(5)=ccc*(w1*(8*(p5p7*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2)+16*(hbcm
     . *p2p3*p4p8*p5p7-hbcm*p3p5*p4p8*p5p7))+w13*(8*(p5p7*p1p8)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2)+16*(hbcm*p1p8*p2p3*p5p7-hbcm*p1p8*
     . p3p5*p5p7))+32*(-fb4*hbcm*p2p8*p5p7+fb4*hbcm*p5p7*p5p8))
      b(6)=32*ccc*(fb3*p1p2*p6p8-fb3*p2p3*p6p8-fb3*p2p4*p6p8)
      b(7)=32*ccc*(-fb3*p1p2*p3p6+fb3*p2p3*p3p6+fb3*p2p4*p3p6)
      b(9)=ccc*(32*(p5p7*p3p6)*(fb3*fmb-fb3*fmc-fb4*ffmcfmb*hbcm+fb4*
     . hbcm)+32*fb4*hbcm*p4p6*p5p7)
      b(10)=ccc*(16*w1*(p5p7*p4p8*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm+
     . hbcm2)+16*w13*(p5p7*p3p6*p1p8)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)
     . +(8*(p6p8*p5p7)*(-4*fb3*ffmcfmb*hbcm2+8*fb3*hbcm2-4*fb4*fmb*
     . hbcm+hbcm2)+64*(fb3*p1p2*p3p6*p7p8-fb3*p1p2*p3p7*p6p8-fb3*p1p3
     . *p5p7*p6p8+fb3*p1p8*p3p6*p5p7-fb3*p2p3*p3p6*p7p8+fb3*p2p3*p3p7
     . *p6p8-fb3*p2p4*p3p6*p7p8+fb3*p2p4*p3p7*p6p8+fb3*p2p8*p3p6*p3p7
     . +fb3*p2p8*p3p6*p4p7+fb3*p2p8*p3p6*p5p7+fb3*p3p4*p5p7*p6p8-fb3*
     . p3p6*p4p8*p5p7)))
      b(11)=8*ccc*(-hbcm*p1p8*p2p5*w13-hbcm*p2p5*p4p8*w1)
      b(12)=-32*ccc*fb3*p2p5
      b(14)=16*ccc*fb4*hbcm*p2p5
      b(15)=ccc*(8*w1*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*
     . w13*(p3p6*p1p8)*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+(4*p6p8*(4*fb3*
     . ffmcfmb*hbcm2-8*fb3*hbcm2+4*fb4*fmb*hbcm-hbcm2)+32*(fb3*p1p3*
     . p6p8-fb3*p1p8*p3p6-fb3*p3p4*p6p8+fb3*p3p6*p4p8)))
      b(16)=ccc*(w1*(8*(p4p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*hbcm*p4p6*
     . p4p8)+w13*(8*(p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+8*hbcm*p1p8*p4p6
     . )+4*p6p8*(4*fb3*fmb-4*fb3*fmc+hbcm))
      b(17)=ccc*(w1*(4*p4p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*(-hbcm*p2p3*
     . p4p8+hbcm*p3p5*p4p8))+w13*(4*p1p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*
     . (-hbcm*p1p8*p2p3+hbcm*p1p8*p3p5))+16*(fb4*hbcm*p2p8-fb4*hbcm*
     . p5p8))
      b(19)=ccc*(4*w1*p4p8*(fmb*hbcm+fmc*hbcm)+4*w13*p1p8*(fmb*hbcm+
     . fmc*hbcm)+16*(-fb3*p1p8-2*fb3*p2p8+2*fb3*p5p8))
      b(21)=2*ccc*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p1p3+4*fb3*p2p3-4*fb3*p3p5-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)
      b(24)=ccc*(16*p3p6*(-fb3*fmb+fb3*fmc+fb4*ffmcfmb*hbcm-fb4*hbcm)
     . -16*fb4*hbcm*p4p6)
      b(25)=4*ccc*p5p7*(-4*fb3*fmb+8*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)
      b(27)=32*ccc*(-fb3*p3p6*p3p7-fb3*p3p6*p4p7-fb3*p3p6*p5p7)
      b(28)=32*ccc*(-fb3*p3p7*p6p8-fb3*p4p7*p6p8-fb3*p5p7*p6p8)
      b(29)=16*ccc*(-fb3*p3p6-2*fb3*p4p6)
      do 200 n=1,32 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp13_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2
     . +fmc2+hbcm2+2*p3p4))
      b(2)=-32*ccc*fb4*hbcm*p5p7*p6p8
      b(3)=8*ccc*w1*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(4)=16*ccc*p5p7*(-fb4*fmb*hbcm-fb4*fmc*hbcm)
      b(5)=ccc*(w1*(8*(p5p7*p4p8)*(2*fmb*hbcm2-2*fmc*hbcm2-hbcm3)-16*
     . hbcm*p3p4*p4p8*p5p7)+32*fb4*hbcm*p4p8*p5p7)
      b(9)=32*ccc*fb4*hbcm*p3p6*p5p7
      b(10)=ccc*(16*w1*(p5p7*p4p8*p3p6)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+
     . 32*(p6p8*p5p7)*(fb4*fmb*hbcm+fb4*fmc*hbcm))
      b(15)=ccc*(8*w1*(p4p8*p3p6)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+16*p6p8
     . *(-fb4*fmb*hbcm-fb4*fmc*hbcm))
      b(16)=16*ccc*fb4*hbcm*p6p8
      b(17)=ccc*(w1*(4*p4p8*(-2*fmb*hbcm2+2*fmc*hbcm2+hbcm3)+8*hbcm*
     . p3p4*p4p8)-16*fb4*hbcm*p4p8)
      b(19)=4*ccc*w1*p4p8*(-fmb*hbcm-fmc*hbcm+2*hbcm2)
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(24)=-16*ccc*fb4*hbcm*p3p6
      b(25)=-16*ccc*fb4*hbcm*p5p7
      do 200 n=1,32 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp12_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-2*ffmcfmb*p3p5+fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5)*(ffmcfmb
     . **2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(1)=ccc*(4*p6p7*(4*fb3*fmb*fmc+4*fb3*fmc2+4*fb4*ffmcfmb*fmb*
     . hbcm+4*fb4*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+32*(-fb3*p2p3*
     . p6p7-2*fb3*p2p4*p6p7))
      ans=ccc*(w7*(8*(p6p7*p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(-
     . ffmcfmb*hbcm*p2p8*p3p6*p3p7-ffmcfmb*hbcm*p2p8*p3p7*p4p6-
     . ffmcfmb*hbcm*p2p8*p3p7*p5p6-hbcm*p2p3*p2p8*p6p7-hbcm*p2p4*p2p8
     . *p6p7))+w15*(8*(p6p7*p5p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p6p7*
     . p1p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+16*(ffmcfmb*hbcm*p1p8*p3p6*
     . p3p7+ffmcfmb*hbcm*p1p8*p3p7*p4p6+ffmcfmb*hbcm*p1p8*p3p7*p5p6-
     . ffmcfmb*hbcm*p3p6*p3p7*p5p8-ffmcfmb*hbcm*p3p7*p4p6*p5p8-
     . ffmcfmb*hbcm*p3p7*p5p6*p5p8+hbcm*p1p8*p2p3*p6p7+hbcm*p1p8*p2p4
     . *p6p7-hbcm*p2p3*p5p8*p6p7-hbcm*p2p4*p5p8*p6p7))+(8*(p6p7*p4p8)
     . *(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p8*p4p7)*(4*fb3*fmc
     . +4*fb4*ffmcfmb*hbcm-hbcm)+8*(p6p8*p3p7)*(4*fb3*fmc-hbcm)+8*(
     . p6p7*p2p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(-
     . hbcm*p3p6*p7p8-hbcm*p4p6*p7p8-hbcm*p5p6*p7p8)))
      b(2)=ans
      b(3)=ccc*(8*w7*(p3p7*p2p8)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2+fmc*hbcm)+w15*(8*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p1p8)*(
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm))+(4*
     . p7p8*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+
     . hbcm2)+32*(-fb3*p1p2*p7p8+fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8
     . *p5p7)))
      b(4)=ccc*(16*p3p7*(-fb4*ffmcfmb*fmb*hbcm-fb4*ffmcfmb*fmc*hbcm)+
     . 32*(fb3*p1p2*p3p7+fb3*p2p3*p5p7-fb3*p2p5*p3p7))
      ans=ccc*(w7*(8*(p5p7*p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*
     . p2p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc2*hbcm)+16*(ffmcfmb*hbcm*p1p3*p2p8*p3p7+ffmcfmb*
     . hbcm*p2p3*p2p8*p3p7-ffmcfmb*hbcm*p2p8*p3p5*p3p7))+w15*(8*(p5p8
     . *p5p7)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p5p7*p1p8)*(-ffmcfmb*hbcm3
     . +fmc*hbcm2)+8*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc2*hbcm)+8*(p3p7*p1p8)*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm
     . -fmc2*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p1p8*p3p7+ffmcfmb*hbcm*p1p3
     . *p3p7*p5p8-ffmcfmb*hbcm*p1p8*p2p3*p3p7+ffmcfmb*hbcm*p1p8*p3p5*
     . p3p7+ffmcfmb*hbcm*p2p3*p3p7*p5p8-ffmcfmb*hbcm*p3p5*p3p7*p5p8))
     . +(4*p7p8*(-4*fb3*ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*hbcm2-4*
     . fb4*fmb*fmc*hbcm+4*fb4*fmc2*hbcm-fmb*hbcm2+fmc*hbcm2-hbcm3)+8*
     . (4*fb4*ffmcfmb*hbcm*p1p8*p3p7+4*fb4*ffmcfmb*hbcm*p2p8*p3p7-4*
     . fb4*ffmcfmb*hbcm*p3p7*p5p8+hbcm*p1p3*p7p8+hbcm*p2p3*p7p8-hbcm*
     . p3p5*p7p8)))
      b(5)=ans
      b(6)=ccc*(4*p6p8*(4*fb3*fmb*fmc+4*fb3*fmc2+4*fb4*ffmcfmb*fmb*
     . hbcm+4*fb4*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+32*(-fb3*p2p3*
     . p6p8-2*fb3*p2p4*p6p8-fb3*p2p8*p3p6-fb3*p2p8*p4p6-fb3*p2p8*p5p6
     . ))
      b(7)=ccc*(4*p3p6*(-4*fb3*fmb*fmc-4*fb3*fmc2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm)+32*(2*fb3*p2p3*
     . p3p6+fb3*p2p3*p4p6+fb3*p2p3*p5p6+2*fb3*p2p4*p3p6))
      b(8)=ccc*(w7*(8*(p5p6*p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p4p6*
     . p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+16*(hbcm*p2p3*p2p8*p3p6+hbcm*
     . p2p4*p2p8*p3p6))+w15*(8*(p5p8*p5p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+
     . 8*(p5p8*p4p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p5p6*p1p8)*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p6*p1p8)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+16*(-hbcm*p1p8*p2p3*p3p6-hbcm*p1p8*p2p4*p3p6+hbcm*p2p3*
     . p3p6*p5p8+hbcm*p2p4*p3p6*p5p8))+(8*(p4p8*p3p6)*(4*fb3*fmc+4*
     . fb4*ffmcfmb*hbcm-hbcm)+8*(p6p8*p3p4)*(-4*fb3*fmc-4*fb4*ffmcfmb
     . *hbcm+hbcm)+4*p6p8*(-4*fb3*fmc*hbcm2-4*fb4*ffmcfmb*hbcm3+hbcm3
     . )+32*(p3p6*p2p8)*(fb3*fmb-fb3*fmc)+32*(p6p8*p2p3)*(-fb3*fmb+
     . fb3*fmc+fb4*hbcm)+32*fb4*hbcm*p2p4*p6p8))
      b(9)=ccc*(8*(p4p7*p3p6)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*
     . (p3p7*p3p6)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p7*p3p4)
     . *(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p6p7*(4*fb3*fmc*hbcm2+4
     . *fb4*ffmcfmb*hbcm3-hbcm3)+8*(p6p7*p2p3)*(-4*fb3*fmb-4*fb4*
     . ffmcfmb*hbcm+hbcm)+32*(-fb4*ffmcfmb*hbcm*p3p7*p4p6-fb4*ffmcfmb
     . *hbcm*p3p7*p5p6-fb4*hbcm*p2p4*p6p7))
      ans2=(32*(p6p7*p2p8)*(-fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+8*(
     . p6p8*p3p7)*(-4*fb3*fmb*fmc-4*fb3*fmc2+fmb*hbcm+fmc*hbcm)+32*(
     . p7p8*p3p6)*(fb3*fmb*fmc+fb3*fmc2+fb4*ffmcfmb*fmb*hbcm+fb4*
     . ffmcfmb*fmc*hbcm)+8*(p7p8*p5p6)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc
     . *hbcm+hbcm2)+8*(p7p8*p4p6)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm
     . +hbcm2)+64*(fb3*p2p3*p2p8*p6p7-fb3*p2p3*p3p6*p7p8+fb3*p2p3*
     . p3p7*p6p8-fb3*p2p3*p4p6*p7p8+fb3*p2p3*p4p7*p6p8-fb3*p2p3*p4p8*
     . p6p7-fb3*p2p3*p5p6*p7p8-fb3*p2p4*p3p6*p7p8+fb3*p2p4*p3p7*p6p8-
     . fb3*p2p8*p3p4*p6p7+2*fb3*p2p8*p3p6*p3p7+fb3*p2p8*p3p6*p4p7+2*
     . fb3*p2p8*p3p7*p4p6+2*fb3*p2p8*p3p7*p5p6))
      ans1=w7*(16*(p5p6*p3p7*p2p8)*(ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p6*
     . p3p7*p2p8)*(ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p8)*(-ffmcfmb*
     . fmb*hbcm3-ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc2*hbcm2)+16*(p3p7
     . *p3p6*p2p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm))+w15*(16*(p5p8
     . *p5p6*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p8*p4p6*p3p7)*(
     . ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-
     . fmc*hbcm)+16*(p4p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*(
     . p6p7*p5p8)*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2
     . +fmc2*hbcm2)+8*(p6p7*p1p8)*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3-fmb*fmc*hbcm2-fmc2*hbcm2)+16*(p5p8*p3p7*p3p6)*(ffmcfmb*
     . fmb*hbcm+ffmcfmb*fmc*hbcm)+16*(p3p7*p3p6*p1p8)*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(w7*(4*p2p8*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(-hbcm*p1p2*
     . p2p8+hbcm*p2p5*p2p8))+w15*(4*p5p8*(ffmcfmb*hbcm3-fmc*hbcm2)+4*
     . p1p8*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(hbcm*p1p2*p1p8-hbcm*p1p2*
     . p5p8-hbcm*p1p8*p2p5+hbcm*p2p5*p5p8))+(4*p5p8*(4*fb3*fmc+4*fb4*
     . ffmcfmb*hbcm-hbcm)+4*p1p8*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
     . +4*p2p8*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)))
      b(12)=2*ccc*(-4*fb3*fmb*fmc-4*fb3*fmc2+16*fb3*p1p2-8*fb3*p2p3-
     . 16*fb3*p2p5-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm)
      b(13)=ccc*(4*w7*p2p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-fmb*
     . fmc*hbcm2-fmc2*hbcm2)+w15*(4*p5p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*
     . fmc*hbcm3-fmb*fmc*hbcm2-fmc2*hbcm2)+4*p1p8*(-ffmcfmb*fmb*hbcm3
     . -ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc2*hbcm2))+16*(fb3*hbcm2*
     . p2p8-2*fb3*p1p3*p2p8+2*fb3*p1p8*p2p3-2*fb3*p2p3*p5p8+2*fb3*
     . p2p8*p3p5))
      b(14)=ccc*(4*p3p5*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p1p3*(
     . 4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p2p3*(4*fb3*fmb+4*fb4*
     . ffmcfmb*hbcm-hbcm)+2*(-4*fb3*fmc*hbcm2-4*fb4*ffmcfmb*hbcm3-8*
     . fb4*hbcm*p1p2+8*fb4*hbcm*p2p5+hbcm3))
      b(15)=ccc*(8*w7*(p3p6*p2p8)*(fmb*hbcm+fmc*hbcm)+w15*(8*(p5p8*
     . p3p6)*(fmb*hbcm+fmc*hbcm)+8*(p3p6*p1p8)*(-fmb*hbcm-fmc*hbcm))+
     . (16*p6p8*(-2*fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(fb3*p2p3
     . *p6p8-fb3*p2p8*p3p6-2*fb3*p3p4*p6p8+2*fb3*p3p6*p4p8)))
      b(16)=ccc*(8*w7*(-hbcm*p2p8*p3p6-hbcm*p2p8*p4p6-hbcm*p2p8*p5p6)
     . +8*w15*(hbcm*p1p8*p3p6+hbcm*p1p8*p4p6+hbcm*p1p8*p5p6-hbcm*p3p6
     . *p5p8-hbcm*p4p6*p5p8-hbcm*p5p6*p5p8)+4*p6p8*(8*fb3*fmb-4*fb3*
     . fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm))
      b(17)=ccc*(w7*(4*p2p8*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(hbcm*
     . p1p3*p2p8+hbcm*p2p3*p2p8-hbcm*p2p8*p3p5))+w15*(4*p5p8*(-
     . ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+4*p1p8*(ffmcfmb*hbcm3-fmc*hbcm2
     . +hbcm3)+8*(-hbcm*p1p3*p1p8+hbcm*p1p3*p5p8-hbcm*p1p8*p2p3+hbcm*
     . p1p8*p3p5+hbcm*p2p3*p5p8-hbcm*p3p5*p5p8))+16*(fb4*hbcm*p1p8+
     . fb4*hbcm*p2p8-fb4*hbcm*p5p8))
      b(18)=ccc*(8*w7*(p3p7*p2p8)*(-ffmcfmb*hbcm2-fmc*hbcm)+w15*(8*(
     . p5p8*p3p7)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p7*p1p8)*(ffmcfmb*
     . hbcm2+fmc*hbcm))+(4*p7p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2-4*
     . fb4*fmc*hbcm-hbcm2)+32*(fb3*p1p3*p7p8-fb3*p1p8*p3p7+fb3*p2p3*
     . p7p8-fb3*p2p8*p3p7-fb3*p3p5*p7p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w7*p2p8*(-fmb*hbcm-fmc*hbcm)+w15*(4*p5p8*(-fmb*
     . hbcm-fmc*hbcm)+4*p1p8*(fmb*hbcm+fmc*hbcm))+16*(-2*fb3*p1p8-fb3
     . *p2p8+2*fb3*p5p8))
      b(20)=ccc*(8*w7*(-ffmcfmb*hbcm*p2p8*p3p7-hbcm*p2p8*p5p7)+8*w15*
     . (ffmcfmb*hbcm*p1p8*p3p7-ffmcfmb*hbcm*p3p7*p5p8+hbcm*p1p8*p5p7-
     . hbcm*p5p7*p5p8)+4*p7p8*(4*fb3*fmb-4*fb3*fmc-hbcm))
      b(21)=2*ccc*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+4*fb3*p1p3+2*fb3*p2p3-4*fb3*p3p5-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)
      b(23)=ccc*(16*p3p7*(-fb3*fmb+fb3*fmc-fb4*ffmcfmb*hbcm)-16*fb4*
     . hbcm*p5p7)
      b(24)=ccc*(4*p3p6*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm
     . )+16*(-fb4*hbcm*p4p6-fb4*hbcm*p5p6))
      b(25)=ccc*(4*p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)-16*fb4*
     . ffmcfmb*hbcm*p3p7)
      b(26)=ccc*(4*p5p6*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p4p6*(
     . -4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p3p6*(-4*fb3*fmc-4*fb4*
     . ffmcfmb*hbcm+hbcm))
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p2p3*p6p7+2*fb3*p3p4*p6p7-2*fb3*p3p6*p3p7-2*fb3*p3p6*p4p7
     . -fb3*p3p7*p4p6-fb3*p3p7*p5p6))
      b(28)=ccc*(8*w7*(p6p7*p2p8)*(fmb*hbcm+fmc*hbcm)+w15*(8*(p6p7*
     . p5p8)*(fmb*hbcm+fmc*hbcm)+8*(p6p7*p1p8)*(-fmb*hbcm-fmc*hbcm))+
     . 32*(-fb3*p2p8*p6p7-fb3*p3p6*p7p8-fb3*p3p7*p6p8-fb3*p4p6*p7p8-2
     . *fb3*p4p7*p6p8+2*fb3*p4p8*p6p7-fb3*p5p6*p7p8))
      b(29)=32*ccc*(-fb3*p3p6-fb3*p4p6-fb3*p5p6)
      b(30)=16*ccc*(-fb3*p3p7-2*fb3*p5p7)
      b(31)=ccc*(w7*(8*(p6p7*p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2+hbcm3)+
     . 16*(p3p7*p3p6*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(hbcm*p2p8*p3p4*
     . p6p7-hbcm*p2p8*p3p6*p4p7))+w15*(8*(p6p7*p5p8)*(-ffmcfmb*hbcm3+
     . fmc*hbcm2+hbcm3)+8*(p6p7*p1p8)*(ffmcfmb*hbcm3-fmc*hbcm2-hbcm3)
     . +16*(p5p8*p3p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p1p8)*(
     . -ffmcfmb*hbcm+hbcm)+16*(-hbcm*p1p8*p3p4*p6p7+hbcm*p1p8*p3p6*
     . p4p7+hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8))+(32*(p6p8*p3p7)
     . *(fb3*fmb-fb3*fmc+fb4*ffmcfmb*hbcm-fb4*hbcm)+8*(p7p8*p3p6)*(-4
     . *fb3*fmb+4*fb3*fmc+hbcm)+32*(-fb4*hbcm*p4p7*p6p8+fb4*hbcm*p4p8
     . *p6p7)))
      b(32)=4*ccc*p6p7*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp11_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*
     . p2p3))
      b(1)=ccc*(4*p6p7*(-4*fb3*fmb*fmc-4*fb3*fmb2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm)+32*(-fb3*p2p3*p6p7-2*fb3*p2p4*p6p7))
      ans=ccc*(w1*(8*(p6p7*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*
     . (p5p6*p4p8*p3p7)*(ffmcfmb*hbcm-hbcm)+16*(p4p8*p4p6*p3p7)*(
     . ffmcfmb*hbcm-hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*
     . hbcm*p2p4*p4p8*p6p7)+w8*(8*(p6p7*p2p8)*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+16*(p5p6*p3p7*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(p4p6*
     . p3p7*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p3p6*p2p8)*(ffmcfmb*
     . hbcm-hbcm)+16*hbcm*p2p4*p2p8*p6p7)+(32*(p6p8*p3p7)*(-fb4*
     . ffmcfmb*hbcm+fb4*hbcm)+8*(p6p7*p2p8)*(-4*fb3*fmc-4*fb4*ffmcfmb
     . *hbcm+4*fb4*hbcm+hbcm)+32*(p7p8*p5p6)*(-fb3*fmb-fb4*ffmcfmb*
     . hbcm+fb4*hbcm)+8*(p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+8*(p6p8*p4p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*
     . fb4*hbcm+hbcm)+32*(p7p8*p4p6)*(-fb3*fmb-fb4*ffmcfmb*hbcm+fb4*
     . hbcm)+32*(p7p8*p3p6)*(-fb3*fmb-fb4*ffmcfmb*hbcm+fb4*hbcm)))
      b(2)=ans
      b(3)=ccc*(8*w1*(p4p8*p3p7)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+8*w8*(p3p7*p2p8)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+(4*p7p8*(-
     . 4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-4*fb3*fmb2+4*fb3*hbcm2-4*fb4
     . *ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+hbcm2)
     . +32*(-fb3*p1p2*p7p8+fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*p5p7)
     . ))
      b(4)=ccc*(4*p3p7*(4*fb3*fmb*fmc+4*fb3*fmb2-fmb*hbcm-fmc*hbcm)+
     . 32*(fb3*p1p2*p3p7+fb3*p2p3*p5p7-fb3*p2p5*p3p7))
      ans2=(8*(p5p8*p3p7)*(4*fb3*fmb-hbcm)+32*(p7p8*p3p5)*(-fb3*fmb-
     . fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(p3p7*p2p8)*(-4*fb3*fmb+hbcm)+32*
     . (p7p8*p2p3)*(fb3*fmb+fb4*ffmcfmb*hbcm-fb4*hbcm)+8*(p3p7*p1p8)*
     . (-4*fb3*fmb+hbcm)+32*(p7p8*p1p3)*(fb3*fmb+fb4*ffmcfmb*hbcm-fb4
     . *hbcm)+4*p7p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2
     . -8*fb3*fmb*hbcm2+4*fb3*fmc*hbcm2-4*fb4*ffmcfmb*hbcm3-4*fb4*fmb
     . *fmc*hbcm+4*fb4*fmb2*hbcm+4*fb4*hbcm3-fmb*hbcm2+fmc*hbcm2)+32*
     . (fb4*hbcm*p1p2*p7p8-fb4*hbcm*p2p5*p7p8+fb4*hbcm*p2p8*p5p7))
      ans1=w1*(8*(p5p7*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p4p8*p3p7*p3p5)*(ffmcfmb*hbcm-hbcm)+16*(p4p8*p3p7*p2p3)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p4p8*p3p7*p1p3)*(-ffmcfmb*hbcm+hbcm)+8*
     . (p4p8*p3p7)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3
     . +fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm+fmc*hbcm2-hbcm3)+16*(-hbcm*
     . p1p2*p3p7*p4p8-hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p3p7*p4p8))+w8*(8
     . *(p5p7*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p3p7*p3p5*
     . p2p8)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p2p8*p2p3)*(-ffmcfmb*hbcm+
     . hbcm)+16*(p3p7*p2p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p8)*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm
     . -fmb*hbcm2-fmb2*hbcm+fmc*hbcm2-hbcm3)+16*(-hbcm*p1p2*p2p8*p3p7
     . -hbcm*p2p3*p2p8*p5p7+hbcm*p2p5*p2p8*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(4*p6p8*(-4*fb3*fmb*fmc-4*fb3*fmb2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm)+32*(-fb3*p2p3*p6p8-2*fb3*p2p4*p6p8-fb3*p2p8*
     . p3p6-fb3*p2p8*p4p6-fb3*p2p8*p5p6))
      b(7)=ccc*(4*p3p6*(4*fb3*fmb*fmc+4*fb3*fmb2+4*fb4*ffmcfmb*fmb*
     . hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmb*hbcm-4*fb4*fmc*hbcm-fmb*
     . hbcm-fmc*hbcm)+32*(2*fb3*p2p3*p3p6+fb3*p2p3*p4p6+fb3*p2p3*p5p6
     . +2*fb3*p2p4*p3p6))
      b(8)=ccc*(w1*(8*(p5p6*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*
     . (p4p8*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(-hbcm*p2p3*
     . p3p6*p4p8-hbcm*p2p3*p4p6*p4p8-hbcm*p2p3*p4p8*p5p6-hbcm*p2p4*
     . p3p6*p4p8))+w8*(8*(p5p6*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)
     . +8*(p4p6*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(-hbcm*p2p3
     . *p2p8*p3p6-hbcm*p2p3*p2p8*p4p6-hbcm*p2p3*p2p8*p5p6-hbcm*p2p4*
     . p2p8*p3p6))+(8*(p4p8*p3p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*
     . fb4*hbcm+hbcm)+8*(p6p8*p3p4)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+32*(p3p6*p2p8)*(fb3*fmb-fb3*fmc)+32*(p6p8*p2p3)
     . *(-fb3*fmb+fb3*fmc+fb4*hbcm)+4*p6p8*(4*fb3*fmb*hbcm2+4*fb4*
     . ffmcfmb*hbcm3-4*fb4*hbcm3-hbcm3)+32*(fb4*hbcm*p2p4*p6p8+fb4*
     . hbcm*p2p8*p4p6+fb4*hbcm*p2p8*p5p6)))
      b(9)=ccc*(8*(p6p7*p2p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+8*(p5p6*p3p7)*(4*fb3*fmb-hbcm)+8*(p4p6*p3p7)*(4*fb3
     . *fmb-hbcm)+8*(p4p7*p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+8*(p3p7*p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+8*(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+4*p6p7*(-4*fb3*fmb*hbcm2-4*fb4*ffmcfmb*hbcm3+4*fb4*
     . hbcm3+hbcm3)-32*fb4*hbcm*p2p4*p6p7)
      ans2=(8*(p7p8*p3p6)*(fmb*hbcm+fmc*hbcm)+32*(p6p8*p3p7)*(fb4*
     . ffmcfmb*fmb*hbcm+fb4*ffmcfmb*fmc*hbcm-fb4*fmb*hbcm-fb4*fmc*
     . hbcm)+8*(p7p8*p5p6)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*
     . fmb*hbcm+hbcm2)+8*(p7p8*p4p6)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*
     . hbcm2-4*fb4*fmb*hbcm+hbcm2)+32*(-fb3*hbcm2*p2p8*p6p7+2*fb3*
     . p2p3*p2p8*p6p7-2*fb3*p2p3*p3p6*p7p8+2*fb3*p2p3*p3p7*p6p8-2*fb3
     . *p2p3*p4p6*p7p8+2*fb3*p2p3*p4p7*p6p8-2*fb3*p2p3*p4p8*p6p7-2*
     . fb3*p2p3*p5p6*p7p8-2*fb3*p2p4*p3p6*p7p8+2*fb3*p2p4*p3p7*p6p8-2
     . *fb3*p2p8*p3p4*p6p7+4*fb3*p2p8*p3p6*p3p7+2*fb3*p2p8*p3p6*p4p7+
     . 4*fb3*p2p8*p3p7*p4p6+4*fb3*p2p8*p3p7*p5p6))
      ans1=w1*(16*(p6p7*p4p8*p2p3)*(fmb*hbcm+fmc*hbcm)+16*(p5p6*p4p8*
     . p3p7)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+16*(p4p8*p4p6*p3p7)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p6p7*p4p8)*(ffmcfmb*fmb*hbcm3
     . +ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc*
     . hbcm3)+16*(p4p8*p3p7*p3p6)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm
     . +fmb*hbcm+fmc*hbcm))+w8*(16*(p6p7*p2p8*p2p3)*(fmb*hbcm+fmc*
     . hbcm)+16*(p5p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+16*(
     . p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p6p7*p2p8)*
     . (ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+
     . fmb2*hbcm2-fmc*hbcm3)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(w1*(4*p4p8*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(hbcm*
     . p1p2*p4p8-hbcm*p2p3*p4p8-hbcm*p2p5*p4p8))+w8*(4*p2p8*(-ffmcfmb
     . *hbcm3-fmb*hbcm2+hbcm3)+8*(hbcm*p1p2*p2p8-hbcm*p2p3*p2p8-hbcm*
     . p2p5*p2p8))+(4*p2p8*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p5p8
     . *(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*p1p8*(4*fb3
     . *fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)))
      b(12)=2*ccc*(4*fb3*fmb*fmc+4*fb3*fmb2+16*fb3*p1p2-8*fb3*p2p3-16
     . *fb3*p2p5+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*
     . fmb*hbcm-4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm)
      b(13)=ccc*(w1*(8*(p4p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+4*p4p8*(-
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmb*hbcm3-
     . fmb2*hbcm2+fmc*hbcm3))+w8*(8*(p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+
     . 4*p2p8*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmb
     . *hbcm3-fmb2*hbcm2+fmc*hbcm3))+(16*p2p8*(fb3*hbcm2+fb4*fmb*hbcm
     . +fb4*fmc*hbcm)+32*(-fb3*p1p3*p2p8+fb3*p1p8*p2p3-fb3*p2p3*p5p8+
     . fb3*p2p8*p3p5)))
      b(14)=ccc*(4*p2p3*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+4*p3p5*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*
     . p1p3*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*(4*fb3*
     . fmb*hbcm2+4*fb4*ffmcfmb*hbcm3-8*fb4*hbcm*p1p2+8*fb4*hbcm*p2p5-
     . 4*fb4*hbcm3-hbcm3))
      b(15)=ccc*(8*w1*(p4p8*p3p6)*(-fmb*hbcm-fmc*hbcm)+8*w8*(p3p6*
     . p2p8)*(-fmb*hbcm-fmc*hbcm)+(16*p6p8*(-2*fb3*hbcm2+fb4*fmb*hbcm
     . +fb4*fmc*hbcm)+32*(fb3*p2p3*p6p8-fb3*p2p8*p3p6-2*fb3*p3p4*p6p8
     . +2*fb3*p3p6*p4p8)))
      b(16)=ccc*(8*w1*(hbcm*p3p6*p4p8+hbcm*p4p6*p4p8+hbcm*p4p8*p5p6)+
     . 8*w8*(hbcm*p2p8*p3p6+hbcm*p2p8*p4p6+hbcm*p2p8*p5p6)+4*p6p8*(4*
     . fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm))
      b(17)=ccc*(w1*(4*p4p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*(-hbcm*p1p3*
     . p4p8+hbcm*p3p5*p4p8))+w8*(4*p2p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*(
     . -hbcm*p1p3*p2p8+hbcm*p2p8*p3p5))+16*(fb4*hbcm*p1p8-fb4*hbcm*
     . p5p8))
      b(18)=ccc*(8*w1*(p4p8*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+8*w8
     . *(p3p7*p2p8)*(ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+(4*p7p8*(4*fb3*
     . ffmcfmb*hbcm2-8*fb3*hbcm2-4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p1p3*
     . p7p8-fb3*p1p8*p3p7+fb3*p2p3*p7p8-fb3*p2p8*p3p7-fb3*p3p5*p7p8+
     . fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w1*p4p8*(fmb*hbcm+fmc*hbcm)+4*w8*p2p8*(fmb*hbcm+
     . fmc*hbcm)+16*(-2*fb3*p1p8-fb3*p2p8+2*fb3*p5p8))
      b(20)=ccc*(8*w1*(ffmcfmb*hbcm*p3p7*p4p8+hbcm*p4p8*p5p7)+8*w8*(
     . ffmcfmb*hbcm*p2p8*p3p7+hbcm*p2p8*p5p7)+16*p7p8*(-fb3*fmc-fb4*
     . ffmcfmb*hbcm))
      b(21)=2*ccc*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+4*fb3*p1p3+2*fb3*p2p3-4*fb3*p3p5-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)
      b(23)=ccc*(4*p3p7*(4*fb3*fmc-hbcm)-16*fb4*hbcm*p5p7)
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb+8*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+16*(-fb4*hbcm*p4p6-fb4*hbcm*p5p6))
      b(25)=ccc*(4*p5p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+4*p3p7*(4*fb3*fmb-hbcm))
      b(26)=ccc*(4*p5p6*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+4*p4p6*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*p3p6
     . *(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm))
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p2p3*p6p7+2*fb3*p3p4*p6p7-2*fb3*p3p6*p3p7-2*fb3*p3p6*p4p7
     . -fb3*p3p7*p4p6-fb3*p3p7*p5p6))
      b(28)=ccc*(8*w1*(p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm)+8*w8*(p6p7*
     . p2p8)*(-fmb*hbcm-fmc*hbcm)+32*(-fb3*p2p8*p6p7-fb3*p3p6*p7p8-
     . fb3*p3p7*p6p8-fb3*p4p6*p7p8-2*fb3*p4p7*p6p8+2*fb3*p4p8*p6p7-
     . fb3*p5p6*p7p8))
      b(29)=32*ccc*(-fb3*p3p6-fb3*p4p6-fb3*p5p6)
      b(30)=16*ccc*(-fb3*p3p7-2*fb3*p5p7)
      b(31)=ccc*(w1*(8*(p6p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)+
     . 16*(p4p8*p3p7*p3p6)*(-ffmcfmb*hbcm+2*hbcm)+16*(hbcm*p2p3*p4p8*
     . p6p7-hbcm*p3p4*p4p8*p6p7+hbcm*p3p6*p4p7*p4p8+hbcm*p3p7*p4p6*
     . p4p8+hbcm*p3p7*p4p8*p5p6))+w8*(8*(p6p7*p2p8)*(ffmcfmb*hbcm3+
     . fmb*hbcm2-2*hbcm3)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb*hbcm+2*hbcm)+
     . 16*(hbcm*p2p3*p2p8*p6p7-hbcm*p2p8*p3p4*p6p7+hbcm*p2p8*p3p6*
     . p4p7+hbcm*p2p8*p3p7*p4p6+hbcm*p2p8*p3p7*p5p6))+(8*(p6p8*p3p7)*
     . (-4*fb3*fmc-4*fb4*hbcm+hbcm)+32*(p7p8*p3p6)*(fb3*fmc+fb4*
     . ffmcfmb*hbcm-fb4*hbcm)+32*(-fb4*hbcm*p2p8*p6p7-fb4*hbcm*p4p6*
     . p7p8-fb4*hbcm*p4p7*p6p8+fb4*hbcm*p4p8*p6p7-fb4*hbcm*p5p6*p7p8)
     . ))
      b(32)=4*ccc*p6p7*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp19_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=ccc*(4*p6p7*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*fmb*fmc+8*fb3*
     . fmc2+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-fmb*hbcm-
     . fmc*hbcm)+32*(fb3*ffmcfmb*p3p6*p3p7+2*fb3*ffmcfmb*p3p7*p4p6+
     . fb3*p1p2*p6p7-fb3*p2p3*p6p7-3*fb3*p2p4*p6p7))
      ans2=(16*(p6p7*p4p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+16*(
     . p6p8*p4p7)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p4p6)*(
     . 8*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p3p6)*(4*fb3*fmc+4*
     . fb4*ffmcfmb*hbcm-ffmcfmb*hbcm)+8*(p6p7*p2p8)*(4*fb3*fmc+4*fb4*
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p8)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm
     . -hbcm)+8*(p6p8*p3p7)*(-4*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc+4*
     . fb3*fmc+4*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm-hbcm))
      ans1=w7*(8*(p6p7*p2p8)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb
     . *fmc*hbcm+fmc2*hbcm)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+
     . ffmcfmb*hbcm)+16*(ffmcfmb*hbcm*p2p8*p3p7*p4p6+hbcm*p1p2*p2p8*
     . p6p7-hbcm*p2p3*p2p8*p6p7-2*hbcm*p2p4*p2p8*p6p7))+w11*(8*(p6p7*
     . p4p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*
     . hbcm)+8*(p6p7*p2p8)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-fmb*
     . fmc*hbcm+fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-
     . ffmcfmb*hbcm)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+ffmcfmb*
     . hbcm)+16*(ffmcfmb*hbcm*p2p8*p3p7*p4p6-ffmcfmb*hbcm*p3p7*p4p6*
     . p4p8+hbcm*p1p2*p2p8*p6p7-hbcm*p1p2*p4p8*p6p7-hbcm*p2p3*p2p8*
     . p6p7+hbcm*p2p3*p4p8*p6p7-2*hbcm*p2p4*p2p8*p6p7+2*hbcm*p2p4*
     . p4p8*p6p7))+w13*(8*(p6p7*p1p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc
     . *hbcm2+fmb*fmc*hbcm-fmc2*hbcm)+16*(p3p7*p3p6*p1p8)*(ffmcfmb**2
     . *hbcm-ffmcfmb*hbcm)+16*(-ffmcfmb*hbcm*p1p8*p3p7*p4p6-hbcm*p1p2
     . *p1p8*p6p7+hbcm*p1p8*p2p3*p6p7+2*hbcm*p1p8*p2p4*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      b(3)=ccc*(w7*(16*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p7*
     . p2p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*
     . hbcm))+w11*(16*(p5p7*p4p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p5p7*
     . p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p3p7)*(-ffmcfmb*fmb*
     . hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p2p8)*(
     . ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+w13
     . *(16*(p5p7*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p1p8)*(-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm))+(16
     . *p7p8*(-fb3*ffmcfmb**2*hbcm2+fb3*fmb*fmc+2*fb3*fmc2+fb4*
     . ffmcfmb*fmb*hbcm+fb4*ffmcfmb*fmc*hbcm)+32*(fb3*ffmcfmb*p1p8*
     . p3p7-2*fb3*ffmcfmb*p3p7*p5p8-2*fb3*p1p2*p7p8+2*fb3*p2p3*p7p8+3
     . *fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*p5p7)))
      b(4)=ccc*(8*p5p7*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+4*
     . p3p7*(4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc
     . -8*fb3*fmc2-4*fb4*fmc*hbcm+fmb*hbcm+fmc*hbcm+hbcm2)+32*(-fb3*
     . ffmcfmb*p1p3*p3p7-2*fb3*ffmcfmb*p2p3*p3p7+2*fb3*ffmcfmb*p3p5*
     . p3p7+2*fb3*p1p2*p3p7+3*fb3*p2p3*p5p7-3*fb3*p2p5*p3p7))
      ans2=(16*(p5p8*p3p7)*(-4*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm)+8*(
     . p7p8*p3p5)*(8*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+8*(p3p7*p1p8)*(
     . 4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p1p3)*(-4*fb3*fmc-4
     . *fb4*ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p8)*(4*fb3*fmb+4*fb4*hbcm-
     . hbcm)+32*(p7p8*p2p3)*(-fb3*fmb-fb4*ffmcfmb*hbcm-fb4*hbcm)+4*
     . p7p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2+8*fb3*
     . fmc*hbcm2+8*fb4*ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-4*fb4*fmc2*
     . hbcm+ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)+32*(fb4*hbcm*p1p2*p7p8-2
     . *fb4*hbcm*p2p5*p7p8+2*fb4*hbcm*p2p8*p5p7))
      ans1=w7*(8*(p5p7*p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p7*p2p8)
     . *(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm)+
     . 16*(-ffmcfmb*hbcm*p2p3*p2p8*p3p7+ffmcfmb*hbcm*p2p8*p3p5*p3p7+
     . hbcm*p1p2*p2p8*p3p7+2*hbcm*p2p3*p2p8*p5p7-2*hbcm*p2p5*p2p8*
     . p3p7))+w11*(8*(p5p7*p4p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p5p7*
     . p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p8*p3p7)*(-ffmcfmb**2*
     . hbcm3-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*hbcm)+8*(p3p7*p2p8)*
     . (ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-fmc2*hbcm)+16
     . *(-ffmcfmb*hbcm*p2p3*p2p8*p3p7+ffmcfmb*hbcm*p2p3*p3p7*p4p8+
     . ffmcfmb*hbcm*p2p8*p3p5*p3p7-ffmcfmb*hbcm*p3p5*p3p7*p4p8+hbcm*
     . p1p2*p2p8*p3p7-hbcm*p1p2*p3p7*p4p8+2*hbcm*p2p3*p2p8*p5p7-2*
     . hbcm*p2p3*p4p8*p5p7-2*hbcm*p2p5*p2p8*p3p7+2*hbcm*p2p5*p3p7*
     . p4p8))+w13*(8*(p5p7*p1p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p7*
     . p1p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+fmc2*
     . hbcm)+16*(ffmcfmb*hbcm*p1p8*p2p3*p3p7-ffmcfmb*hbcm*p1p8*p3p5*
     . p3p7-hbcm*p1p2*p1p8*p3p7-2*hbcm*p1p8*p2p3*p5p7+2*hbcm*p1p8*
     . p2p5*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w7*(16*(p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p6*
     . p2p8)*(ffmcfmb*hbcm2-fmc*hbcm))+w11*(16*(p4p8*p4p6)*(-ffmcfmb*
     . hbcm2+fmc*hbcm)+8*(p4p8*p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p6*p2p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm))+w13*(16*(p4p6*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm)
     . +8*(p3p6*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm))+(32*(p3p6*p2p8)*(-
     . fb3*ffmcfmb-fb3)+4*p6p8*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*fmb*fmc
     . +8*fb3*fmc2+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-fmb*
     . hbcm-fmc*hbcm)+32*(fb3*p1p2*p6p8-fb3*p2p3*p6p8-3*fb3*p2p4*p6p8
     . -3*fb3*p2p8*p4p6)))
      b(7)=ccc*(32*(p3p6*p2p3)*(fb3*ffmcfmb+2*fb3)+8*p4p6*(-4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+4*p3p6*(4*fb3*ffmcfmb**2*
     . hbcm2-4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-8*fb3*fmc2-4*fb4*
     . ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm+hbcm2)+32*(-fb3*p1p2*p3p6+3*fb3*p2p3*p4p6+3*fb3*
     . p2p4*p3p6))
      ans2=(16*(p4p8*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+16*(
     . p6p8*p3p4)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p8)*
     . (-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(p6p8*p1p3)*(4*fb3*fmc+
     . 4*fb4*ffmcfmb*hbcm-hbcm)+8*(p3p6*p2p8)*(4*fb3*fmb-8*fb3*fmc-4*
     . fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p8*p2p3)*(-4*fb3*fmb+8*
     . fb3*fmc+4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-hbcm)+4*p6p8*(4*fb3*
     . ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-8*fb3*fmc*hbcm2-8*
     . fb4*ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-4*fb4*fmc2*hbcm+ffmcfmb*
     . hbcm3-fmb*hbcm2+2*hbcm3)+32*(-fb4*hbcm*p1p2*p6p8+2*fb4*hbcm*
     . p2p4*p6p8+2*fb4*hbcm*p2p8*p4p6))
      ans1=w7*(8*(p4p6*p2p8)*(-ffmcfmb*hbcm3+fmc*hbcm2)+8*(p3p6*p2p8)
     . *(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3+fmb*fmc*
     . hbcm+fmc*hbcm2-fmc2*hbcm)+16*(-hbcm*p1p2*p2p8*p3p6+2*hbcm*p2p3
     . *p2p8*p3p6+2*hbcm*p2p3*p2p8*p4p6+2*hbcm*p2p4*p2p8*p3p6))+w11*(
     . 8*(p4p8*p4p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p4p6*p2p8)*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p8*p3p6)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2+fmc2*
     . hbcm)+8*(p3p6*p2p8)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmb*fmc*hbcm+fmc*hbcm2-fmc2*hbcm)+16*(-hbcm*p1p2
     . *p2p8*p3p6+hbcm*p1p2*p3p6*p4p8+2*hbcm*p2p3*p2p8*p3p6+2*hbcm*
     . p2p3*p2p8*p4p6-2*hbcm*p2p3*p3p6*p4p8-2*hbcm*p2p3*p4p6*p4p8+2*
     . hbcm*p2p4*p2p8*p3p6-2*hbcm*p2p4*p3p6*p4p8))+w13*(8*(p4p6*p1p8)
     . *(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p6*p1p8)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3-fmb*fmc*hbcm-fmc*hbcm2+fmc2*
     . hbcm)+16*(hbcm*p1p2*p1p8*p3p6-2*hbcm*p1p8*p2p3*p3p6-2*hbcm*
     . p1p8*p2p3*p4p6-2*hbcm*p1p8*p2p4*p3p6))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(16*(p4p6*p3p7)*(-4*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm)+
     . 16*(p4p7*p3p6)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+16*(p6p7*
     . p3p4)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p3)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+hbcm)+8*(p6p7*p1p3)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6)*(2*fb3*ffmcfmb
     . *fmb-2*fb3*ffmcfmb*fmc-4*fb3*fmc-2*fb4*ffmcfmb**2*hbcm-2*fb4*
     . ffmcfmb*hbcm+hbcm)+4*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2+4*fb3*
     . ffmcfmb*fmc*hbcm2+8*fb3*fmc*hbcm2+8*fb4*ffmcfmb*hbcm3-4*fb4*
     . fmb*fmc*hbcm+4*fb4*fmc2*hbcm+fmb*hbcm2-fmc*hbcm2-2*hbcm3)+32*(
     . fb4*hbcm*p1p2*p6p7-2*fb4*hbcm*p2p4*p6p7))
      ans4=(8*(p7p8*p3p6)*(4*fb3*fmc2+4*fb4*ffmcfmb*fmc*hbcm-fmb*hbcm
     . -hbcm2)+64*(p3p7*p3p6*p2p8)*(3*fb3*ffmcfmb+fb3)+8*(p6p7*p4p8)*
     . (4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p8*p4p7)*(-4*
     . fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+8*(p7p8*p4p6)*(4*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p8*p3p7)*(4*fb3*
     . ffmcfmb*hbcm2-4*fb3*fmc2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*fmc*hbcm
     . +fmc*hbcm+hbcm2)+8*(p6p7*p1p8)*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc
     . *hbcm+hbcm2)+64*(-fb3*ffmcfmb*hbcm2*p2p8*p6p7-fb3*ffmcfmb*p1p3
     . *p3p7*p6p8+fb3*ffmcfmb*p1p8*p3p6*p3p7-fb3*ffmcfmb*p2p3*p3p6*
     . p7p8+fb3*ffmcfmb*p3p4*p3p7*p6p8-fb3*ffmcfmb*p3p6*p3p7*p4p8+fb3
     . *p1p8*p2p3*p6p7+2*fb3*p2p3*p2p8*p6p7+fb3*p2p3*p3p7*p6p8-fb3*
     . p2p3*p4p6*p7p8+2*fb3*p2p3*p4p7*p6p8-2*fb3*p2p3*p4p8*p6p7-fb3*
     . p2p4*p3p6*p7p8+fb3*p2p4*p3p7*p6p8-fb3*p2p8*p3p4*p6p7+fb3*p2p8*
     . p3p6*p4p7+4*fb3*p2p8*p3p7*p4p6))
      ans3=w13*(16*(p6p7*p2p3*p1p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+16*(
     . p4p6*p3p7*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p7*p3p6*p1p8)*(
     . -ffmcfmb*hbcm2+fmc*hbcm)+16*(p6p7*p3p4*p1p8)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+16*(p6p7*p1p8*p1p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p3p7*p3p6*p1p8)*(ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+fmc*hbcm)+8
     . *(p6p7*p1p8)*(-ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3+2*ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmc*hbcm3)+16*(hbcm2*
     . p1p2*p1p8*p6p7-hbcm2*p1p8*p2p4*p6p7))+ans4
      ans2=w11*(16*(p6p7*p4p8*p2p3)*(fmb*hbcm+fmc*hbcm-hbcm2)+16*(
     . p6p7*p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+16*(p4p8*p4p6*p3p7)
     . *(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p8*p4p7*p3p6)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p4p8*p3p4)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p7*p3p6*p2p8)*
     . (ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p3p4*p2p8)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p4p8*p1p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(
     . p6p7*p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(p4p8*p3p7*p3p6)*(
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+fmc*hbcm)+16*(p3p7*p3p6*p2p8
     . )*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmc*hbcm)+8*(p6p7*p4p8)*
     . (-ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+2*
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2-2*fmc*hbcm3)+8*(p6p7*p2p8)*(
     . ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*ffmcfmb
     . *hbcm4-fmb*fmc*hbcm2+2*fmc*hbcm3)+16*(-hbcm2*p1p2*p2p8*p6p7+
     . hbcm2*p1p2*p4p8*p6p7+hbcm2*p2p4*p2p8*p6p7-hbcm2*p2p4*p4p8*p6p7
     . ))+ans3
      ans1=w7*(16*(p6p7*p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+16*(
     . p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p7*p3p6*p2p8)*
     . (ffmcfmb*hbcm2-fmc*hbcm)+16*(p6p7*p3p4*p2p8)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+16*(p6p7*p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+16*(
     . p3p7*p3p6*p2p8)*(-ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-fmc*hbcm)+
     . 8*(p6p7*p2p8)*(ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*
     . hbcm3-2*ffmcfmb*hbcm4-fmb*fmc*hbcm2+2*fmc*hbcm3)+16*(-hbcm2*
     . p1p2*p2p8*p6p7+hbcm2*p2p4*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w7*(4*p2p8*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+8*(-2*hbcm*p1p2*
     . p2p8+2*hbcm*p2p3*p2p8+3*hbcm*p2p5*p2p8))+w11*(4*p4p8*(ffmcfmb
     . **2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-2*
     . fmc2*hbcm)+4*p2p8*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb
     . *fmc*hbcm2-fmb*fmc*hbcm+2*fmc2*hbcm)+8*(-2*hbcm*p1p2*p2p8+2*
     . hbcm*p1p2*p4p8+2*hbcm*p2p3*p2p8-2*hbcm*p2p3*p4p8+3*hbcm*p2p5*
     . p2p8-3*hbcm*p2p5*p4p8))+w13*(4*p1p8*(ffmcfmb**2*hbcm3-ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2+fmb*fmc*hbcm-2*fmc2*hbcm)+8*(2*
     . hbcm*p1p2*p1p8-2*hbcm*p1p8*p2p3-3*hbcm*p1p8*p2p5))+(12*p5p8*(4
     . *fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*p1p8*(-4*fb3*fmc-4*fb4*
     . ffmcfmb*hbcm+hbcm)+4*p2p8*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)
     . ))
      b(11)=ans
      b(12)=2*ccc*(8*fb3*ffmcfmb**2*hbcm2-4*fb3*fmb*fmc-12*fb3*fmc2+
     . 24*fb3*p1p2-24*fb3*p2p3-32*fb3*p2p5-4*fb4*ffmcfmb*fmb*hbcm-4*
     . fb4*ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm)
      ans2=w13*(16*(p3p5*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p2p3*p1p8)
     . *(-ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p1p8*p1p3)*(-ffmcfmb*hbcm2
     . +fmc*hbcm)+4*p1p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+2*
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2-2*fmc*hbcm3-fmc2*hbcm2)+8*(hbcm2*
     . p1p2*p1p8-2*hbcm2*p1p8*p2p5))+(8*p5p8*(4*fb3*ffmcfmb*hbcm2+4*
     . fb4*fmc*hbcm-hbcm2)+4*p2p8*(-12*fb3*ffmcfmb*hbcm2+16*fb3*hbcm2
     . +4*fb4*fmb*hbcm+hbcm2)+4*p1p8*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*
     . hbcm+hbcm2)+32*(-2*fb3*p1p3*p2p8+2*fb3*p1p8*p2p3-3*fb3*p2p3*
     . p5p8+3*fb3*p2p8*p3p5))
      ans1=w7*(16*(p3p5*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p2p8*p2p3)
     . *(ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p2p8*p1p3)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+4*p2p8*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-2*
     . ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+fmc2*hbcm2)+8*(-hbcm2*
     . p1p2*p2p8+2*hbcm2*p2p5*p2p8))+w11*(16*(p4p8*p3p5)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+16*(p3p5*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(
     . p4p8*p2p3)*(-ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p2p8*p2p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p4p8*p1p3)*(-ffmcfmb*hbcm2+
     . fmc*hbcm)+8*(p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+4*p4p8*(
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2-2*fmc*hbcm3-fmc2*hbcm2)+4*p2p8*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmc*hbcm3+
     . fmc2*hbcm2)+8*(-hbcm2*p1p2*p2p8+hbcm2*p1p2*p4p8+2*hbcm2*p2p5*
     . p2p8-2*hbcm2*p2p5*p4p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(12*p3p5*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*p1p3*
     . (4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p2p3*(4*fb3*fmb+4*fb4*
     . ffmcfmb*hbcm+8*fb4*hbcm-hbcm)+2*(-4*fb3*ffmcfmb*fmb*hbcm2+12*
     . fb3*ffmcfmb*fmc*hbcm2-16*fb3*fmc*hbcm2+4*fb4*ffmcfmb**2*hbcm3-
     . 16*fb4*ffmcfmb*hbcm3-4*fb4*fmb*fmc*hbcm+8*fb4*fmc2*hbcm-16*fb4
     . *hbcm*p1p2+24*fb4*hbcm*p2p5-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*
     . hbcm2+4*hbcm3))
      b(15)=ccc*(w7*(8*(p3p6*p2p8)*(fmb*hbcm+fmc*hbcm+hbcm2)+16*hbcm2
     . *p2p8*p4p6)+w11*(8*(p4p8*p3p6)*(-fmb*hbcm-fmc*hbcm-hbcm2)+8*(
     . p3p6*p2p8)*(fmb*hbcm+fmc*hbcm+hbcm2)+16*(hbcm2*p2p8*p4p6-hbcm2
     . *p4p6*p4p8))+w13*(8*(p3p6*p1p8)*(-fmb*hbcm-fmc*hbcm-hbcm2)-16*
     . hbcm2*p1p8*p4p6)+(16*p6p8*(-2*fb3*ffmcfmb*hbcm2-2*fb3*hbcm2+
     . fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(fb3*p1p3*p6p8-fb3*p1p8*p3p6+2*
     . fb3*p2p3*p6p8-2*fb3*p2p8*p3p6-3*fb3*p3p4*p6p8+3*fb3*p3p6*p4p8)
     . ))
      b(16)=ccc*(w7*(8*(p3p6*p2p8)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p2p8*
     . p4p6)+w11*(8*(p4p8*p3p6)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8)*(-
     . ffmcfmb*hbcm-hbcm)+24*(-hbcm*p2p8*p4p6+hbcm*p4p6*p4p8))+w13*(8
     . *(p3p6*p1p8)*(ffmcfmb*hbcm+hbcm)+24*hbcm*p1p8*p4p6)+4*p6p8*(4*
     . fb3*fmb-4*fb3*fmc-hbcm))
      b(17)=ccc*(w7*(4*p2p8*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-4*
     . hbcm3)+8*(2*hbcm*p1p3*p2p8+hbcm*p2p3*p2p8-3*hbcm*p2p8*p3p5))+
     . w11*(4*p4p8*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+4*hbcm3)+4*
     . p2p8*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-4*hbcm3)+8*(2*hbcm*
     . p1p3*p2p8-2*hbcm*p1p3*p4p8+hbcm*p2p3*p2p8-hbcm*p2p3*p4p8-3*
     . hbcm*p2p8*p3p5+3*hbcm*p3p5*p4p8))+w13*(4*p1p8*(-2*ffmcfmb*
     . hbcm3+fmb*hbcm2-fmc*hbcm2+4*hbcm3)+8*(-2*hbcm*p1p3*p1p8-hbcm*
     . p1p8*p2p3+3*hbcm*p1p8*p3p5))+16*(2*fb4*hbcm*p1p8+fb4*hbcm*p2p8
     . -3*fb4*hbcm*p5p8))
      b(18)=ccc*(w7*(8*(p3p7*p2p8)*(-fmb*hbcm-fmc*hbcm-hbcm2)-16*
     . hbcm2*p2p8*p5p7)+w11*(8*(p4p8*p3p7)*(fmb*hbcm+fmc*hbcm+hbcm2)+
     . 8*(p3p7*p2p8)*(-fmb*hbcm-fmc*hbcm-hbcm2)+16*(-hbcm2*p2p8*p5p7+
     . hbcm2*p4p8*p5p7))+w13*(8*(p3p7*p1p8)*(fmb*hbcm+fmc*hbcm+hbcm2)
     . +16*hbcm2*p1p8*p5p7)+(16*p7p8*(2*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2
     . -fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(2*fb3*p1p3*p7p8-2*fb3*p1p8*
     . p3p7+fb3*p2p3*p7p8-fb3*p2p8*p3p7-3*fb3*p3p5*p7p8+3*fb3*p3p7*
     . p5p8)))
      b(19)=ccc*(4*w7*p2p8*(-fmb*hbcm-fmc*hbcm)+w11*(4*p4p8*(fmb*hbcm
     . +fmc*hbcm)+4*p2p8*(-fmb*hbcm-fmc*hbcm))+4*w13*p1p8*(fmb*hbcm+
     . fmc*hbcm)+16*(-3*fb3*p1p8-fb3*p2p8+4*fb3*p5p8))
      b(20)=ccc*(w7*(8*(p3p7*p2p8)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p2p8
     . *p5p7)+w11*(8*(p4p8*p3p7)*(-ffmcfmb*hbcm+2*hbcm)+8*(p3p7*p2p8)
     . *(ffmcfmb*hbcm-2*hbcm)+24*(-hbcm*p2p8*p5p7+hbcm*p4p8*p5p7))+
     . w13*(8*(p3p7*p1p8)*(-ffmcfmb*hbcm+2*hbcm)+24*hbcm*p1p8*p5p7)+4
     . *p7p8*(4*fb3*fmb-4*fb3*fmc+hbcm))
      b(21)=8*ccc*(fb3*fmb-fb3*fmc)
      b(22)=8*ccc*(4*fb3*ffmcfmb*hbcm2-6*fb3*hbcm2+6*fb3*p1p3+2*fb3*
     . p2p3-8*fb3*p3p5-fb4*fmb*hbcm-fb4*fmc*hbcm)
      b(23)=ccc*(16*p3p7*(-fb3*fmb+fb3*fmc+fb4*ffmcfmb*hbcm-2*fb4*
     . hbcm)-48*fb4*hbcm*p5p7)
      b(24)=ccc*(16*p3p6*(-fb3*fmb+fb3*fmc-fb4*ffmcfmb*hbcm-fb4*hbcm)
     . -48*fb4*hbcm*p4p6)
      b(25)=ccc*(12*p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p3p7*
     . (-4*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*fmc-8*fb3*fmc+4*fb4*ffmcfmb
     . **2*hbcm-8*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm+2*hbcm))
      b(26)=ccc*(12*p4p6*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p3p6*
     . (-4*fb3*ffmcfmb*fmc-4*fb3*fmc-4*fb4*ffmcfmb**2*hbcm-4*fb4*
     . ffmcfmb*hbcm+ffmcfmb*hbcm+hbcm))
      b(27)=ccc*(64*(p3p7*p3p6)*(-fb3*ffmcfmb-fb3)+16*p6p7*(2*fb3*
     . ffmcfmb*hbcm2+2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(-fb3*
     . p1p3*p6p7-2*fb3*p2p3*p6p7+3*fb3*p3p4*p6p7-3*fb3*p3p6*p4p7-3*
     . fb3*p3p7*p4p6))
      b(28)=ccc*(8*w7*(p6p7*p2p8)*(fmb*hbcm+fmc*hbcm)+w11*(8*(p6p7*
     . p4p8)*(-fmb*hbcm-fmc*hbcm)+8*(p6p7*p2p8)*(fmb*hbcm+fmc*hbcm))+
     . 8*w13*(p6p7*p1p8)*(-fmb*hbcm-fmc*hbcm)+(32*(p6p8*p3p7)*(-fb3*
     . ffmcfmb-fb3)+32*(p7p8*p3p6)*(-fb3*ffmcfmb-fb3)+32*(-fb3*p1p8*
     . p6p7-2*fb3*p2p8*p6p7-3*fb3*p4p6*p7p8-3*fb3*p4p7*p6p8+3*fb3*
     . p4p8*p6p7)))
      b(29)=ccc*(16*p3p6*(-2*fb3*ffmcfmb-fb3)-64*fb3*p4p6)
      b(30)=ccc*(16*p3p7*(2*fb3*ffmcfmb-3*fb3)-64*fb3*p5p7)
      ans=ccc*(w7*(8*(p6p7*p2p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+16*(-
     . hbcm*p1p3*p2p8*p6p7-hbcm*p2p3*p2p8*p6p7+2*hbcm*p2p8*p3p4*p6p7-
     . 2*hbcm*p2p8*p3p6*p3p7-2*hbcm*p2p8*p3p6*p4p7-2*hbcm*p2p8*p3p7*
     . p4p6))+w11*(8*(p6p7*p4p8)*(fmb*hbcm2-fmc*hbcm2-2*hbcm3)+8*(
     . p6p7*p2p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+16*(-hbcm*p1p3*p2p8*
     . p6p7+hbcm*p1p3*p4p8*p6p7-hbcm*p2p3*p2p8*p6p7+hbcm*p2p3*p4p8*
     . p6p7+2*hbcm*p2p8*p3p4*p6p7-2*hbcm*p2p8*p3p6*p3p7-2*hbcm*p2p8*
     . p3p6*p4p7-2*hbcm*p2p8*p3p7*p4p6-2*hbcm*p3p4*p4p8*p6p7+2*hbcm*
     . p3p6*p3p7*p4p8+2*hbcm*p3p6*p4p7*p4p8+2*hbcm*p3p7*p4p6*p4p8))+
     . w13*(8*(p6p7*p1p8)*(fmb*hbcm2-fmc*hbcm2-2*hbcm3)+16*(hbcm*p1p3
     . *p1p8*p6p7+hbcm*p1p8*p2p3*p6p7-2*hbcm*p1p8*p3p4*p6p7+2*hbcm*
     . p1p8*p3p6*p3p7+2*hbcm*p1p8*p3p6*p4p7+2*hbcm*p1p8*p3p7*p4p6))+(
     . 32*(p6p8*p3p7)*(fb3*fmb-fb3*fmc-fb4*hbcm)+32*(p7p8*p3p6)*(-fb3
     . *fmb+fb3*fmc-fb4*hbcm)+32*(-fb4*hbcm*p1p8*p6p7-fb4*hbcm*p2p8*
     . p6p7-2*fb4*hbcm*p4p6*p7p8-2*fb4*hbcm*p4p7*p6p8+2*fb4*hbcm*p4p8
     . *p6p7)))
      b(31)=ans
      b(32)=16*ccc*p6p7*(fb3*fmb-fb3*fmc)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp1s1_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(10) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,10 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=-8*ccc*fb3*p6p7
      b(2)=4*ccc*(-hbcm*p4p8*p6p7*w1+hbcm*p5p8*p6p7*w2)
      b(3)=-8*ccc*fb3*p7p8
      b(4)=8*ccc*fb3*p3p7
      b(5)=4*ccc*(2*fb4*hbcm*p7p8-hbcm*p3p7*p4p8*w1+hbcm*p3p7*p5p8*w2
     . )
      b(6)=-8*ccc*fb3*p6p8
      b(7)=8*ccc*fb3*p3p6
      b(8)=4*ccc*(2*fb4*hbcm*p6p8-hbcm*p3p6*p4p8*w1+hbcm*p3p6*p5p8*w2
     . )
      b(9)=8*ccc*fb4*hbcm*p6p7
      b(10)=12*ccc*(hbcm2*p4p8*p6p7*w1-hbcm2*p5p8*p6p7*w2)
      do 200 n=1,10 
         c(n,1)=c(n,1)+1.0d0*b(n)
200   continue
      return
	end 
 
      subroutine amp27_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p2p3-2*p2p5+2*p3p5))
      b(1)=16*ccc*(-fb3*p3p6*p5p7-3*fb3*p4p6*p5p7+fb3*p5p6*p5p7)
      b(2)=ccc*(w1*(16*(p5p7*p4p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*hbcm*
     . p4p6*p4p8*p5p7)+w16*(16*(p5p8*p5p7*p3p6)*(ffmcfmb*hbcm-hbcm)+
     . 16*(p5p7*p3p6*p2p8)*(-ffmcfmb*hbcm+hbcm)+16*(hbcm*p2p8*p4p6*
     . p5p7-hbcm*p4p6*p5p7*p5p8))+4*(p6p8*p5p7)*(8*fb3*fmb-4*fb3*fmc+
     . 4*fb4*ffmcfmb*hbcm-8*fb4*hbcm+hbcm))
      b(3)=ccc*(12*w1*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+w16*(12*(
     . p5p8*p5p7)*(-fmb*hbcm-fmc*hbcm+hbcm2)+12*(p5p7*p2p8)*(fmb*hbcm
     . +fmc*hbcm-hbcm2))+16*(-fb3*p1p8*p5p7-3*fb3*p2p8*p5p7-fb3*p4p8*
     . p5p7+3*fb3*p5p7*p5p8))
      b(4)=ccc*(8*p5p7*(2*fb3*ffmcfmb*hbcm2-3*fb3*hbcm2-3*fb4*fmb*
     . hbcm-3*fb4*fmc*hbcm)+16*(fb3*p1p3*p5p7+3*fb3*p2p3*p5p7+fb3*
     . p3p4*p5p7-3*fb3*p3p5*p5p7))
      b(5)=ccc*(w1*(4*(p5p7*p4p8)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc
     . *hbcm2+3*hbcm3)+8*(-hbcm*p1p3*p4p8*p5p7-3*hbcm*p2p3*p4p8*p5p7-
     . hbcm*p3p4*p4p8*p5p7+3*hbcm*p3p5*p4p8*p5p7))+w16*(4*(p5p8*p5p7)
     . *(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-3*hbcm3)+4*(p5p7*
     . p2p8)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+8*(-
     . hbcm*p1p3*p2p8*p5p7+hbcm*p1p3*p5p7*p5p8-3*hbcm*p2p3*p2p8*p5p7+
     . 3*hbcm*p2p3*p5p7*p5p8-hbcm*p2p8*p3p4*p5p7+3*hbcm*p2p8*p3p5*
     . p5p7+hbcm*p3p4*p5p7*p5p8-3*hbcm*p3p5*p5p7*p5p8))+16*(fb4*hbcm*
     . p1p8*p5p7+3*fb4*hbcm*p2p8*p5p7+fb4*hbcm*p4p8*p5p7-3*fb4*hbcm*
     . p5p7*p5p8))
      b(6)=16*ccc*(-fb3*p1p2*p6p8+fb3*p2p3*p6p8+fb3*p2p4*p6p8-fb3*
     . p2p5*p6p8)
      b(7)=16*ccc*(fb3*p1p2*p3p6-fb3*p2p3*p3p6-fb3*p2p4*p3p6+fb3*p2p5
     . *p3p6)
      b(8)=ccc*(8*w1*(-hbcm*p1p2*p3p6*p4p8+hbcm*p2p3*p3p6*p4p8+hbcm*
     . p2p4*p3p6*p4p8-hbcm*p2p5*p3p6*p4p8)+8*w16*(-hbcm*p1p2*p2p8*
     . p3p6+hbcm*p1p2*p3p6*p5p8+hbcm*p2p3*p2p8*p3p6-hbcm*p2p3*p3p6*
     . p5p8+hbcm*p2p4*p2p8*p3p6-hbcm*p2p4*p3p6*p5p8-hbcm*p2p5*p2p8*
     . p3p6+hbcm*p2p5*p3p6*p5p8)+16*(fb4*hbcm*p1p2*p6p8-fb4*hbcm*p2p3
     . *p6p8-fb4*hbcm*p2p4*p6p8+fb4*hbcm*p2p5*p6p8))
      b(9)=ccc*(4*(p5p7*p3p6)*(-8*fb3*fmb+4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm+hbcm)-32*fb4*hbcm*p4p6*p5p7)
      b(10)=ccc*(w1*(8*(p5p7*p4p8*p3p6)*(3*ffmcfmb*hbcm2-2*fmb*hbcm-
     . fmc*hbcm-hbcm2)+8*(-hbcm2*p4p6*p4p8*p5p7-hbcm2*p4p8*p5p6*p5p7)
     . )+w16*(8*(p5p8*p5p7*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*
     . hbcm+hbcm2)+8*(p5p7*p3p6*p2p8)*(3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc
     . *hbcm-hbcm2)+8*(-hbcm2*p2p8*p4p6*p5p7-hbcm2*p2p8*p5p6*p5p7+
     . hbcm2*p4p6*p5p7*p5p8+hbcm2*p5p6*p5p7*p5p8))+(4*(p6p8*p5p7)*(-4
     . *fb3*ffmcfmb*hbcm2+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm-3*hbcm2)+32*(
     . -fb3*p1p2*p3p6*p7p8+fb3*p1p2*p3p7*p6p8+fb3*p1p3*p5p7*p6p8-fb3*
     . p1p8*p3p6*p5p7+fb3*p2p3*p3p6*p7p8-fb3*p2p3*p3p7*p6p8-fb3*p2p3*
     . p5p7*p6p8+fb3*p2p4*p3p6*p7p8-fb3*p2p4*p3p7*p6p8-fb3*p2p5*p3p6*
     . p7p8+fb3*p2p5*p3p7*p6p8-fb3*p2p8*p3p6*p3p7-fb3*p2p8*p3p6*p4p7-
     . fb3*p3p4*p5p7*p6p8+fb3*p3p5*p5p7*p6p8+fb3*p3p6*p4p8*p5p7-fb3*
     . p3p6*p5p7*p5p8)))
      b(11)=ccc*(12*w16*(hbcm*p2p5*p2p8-hbcm*p2p5*p5p8)+12*hbcm*p2p5*
     . p4p8*w1)
      b(12)=24*ccc*fb3*p2p5
      b(13)=ccc*(12*w16*(hbcm2*p2p5*p2p8-hbcm2*p2p5*p5p8)+12*hbcm2*
     . p2p5*p4p8*w1)
      b(14)=-24*ccc*fb4*hbcm*p2p5
      b(15)=ccc*(w1*(4*(p4p8*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*
     . hbcm+hbcm2)+4*(hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+w16*(4*(p5p8*
     . p3p6)*(3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-hbcm2)+4*(p3p6*p2p8
     . )*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+4*(hbcm2*p2p8*
     . p4p6+hbcm2*p2p8*p5p6-hbcm2*p4p6*p5p8-hbcm2*p5p6*p5p8))+(2*p6p8
     . *(4*fb3*ffmcfmb*hbcm2-8*fb4*fmb*hbcm-4*fb4*fmc*hbcm+3*hbcm2)+
     . 16*(-fb3*p1p3*p6p8+fb3*p1p8*p3p6+fb3*p2p3*p6p8-fb3*p2p8*p3p6+
     . fb3*p3p4*p6p8-fb3*p3p5*p6p8-fb3*p3p6*p4p8+fb3*p3p6*p5p8)))
      b(16)=ccc*(w1*(8*(p4p8*p3p6)*(ffmcfmb*hbcm-hbcm)-8*hbcm*p4p6*
     . p4p8)+w16*(8*(p5p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(-hbcm*p2p8*p4p6+hbcm*p4p6*p5p8))+2*p6p8*
     . (-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+8*fb4*hbcm-hbcm))
      b(17)=ccc*(w1*(2*p4p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-
     . 3*hbcm3)+4*(hbcm*p1p3*p4p8+3*hbcm*p2p3*p4p8+hbcm*p3p4*p4p8-3*
     . hbcm*p3p5*p4p8))+w16*(2*p5p8*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*
     . fmc*hbcm2+3*hbcm3)+2*p2p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+4*(hbcm*p1p3*p2p8-hbcm*p1p3*p5p8+3*hbcm*p2p3*
     . p2p8-3*hbcm*p2p3*p5p8+hbcm*p2p8*p3p4-3*hbcm*p2p8*p3p5-hbcm*
     . p3p4*p5p8+3*hbcm*p3p5*p5p8))+8*(-fb4*hbcm*p1p8-3*fb4*hbcm*p2p8
     . -fb4*hbcm*p4p8+3*fb4*hbcm*p5p8))
      b(19)=ccc*(6*w1*p4p8*(-fmb*hbcm-fmc*hbcm+hbcm2)+w16*(6*p5p8*(
     . fmb*hbcm+fmc*hbcm-hbcm2)+6*p2p8*(-fmb*hbcm-fmc*hbcm+hbcm2))+8*
     . (fb3*p1p8+3*fb3*p2p8+fb3*p4p8-3*fb3*p5p8))
      b(21)=12*ccc*(-fb3*fmb+fb3*fmc+fb4*hbcm)
      b(22)=4*ccc*(-2*fb3*ffmcfmb*hbcm2+3*fb3*hbcm2-2*fb3*p1p3-6*fb3*
     . p2p3-2*fb3*p3p4+6*fb3*p3p5+3*fb4*fmb*hbcm+3*fb4*fmc*hbcm)
      b(24)=ccc*(2*p3p6*(8*fb3*fmb-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)
     . +16*fb4*hbcm*p4p6)
      b(25)=24*ccc*p5p7*(fb3*fmb-fb3*fmc-fb4*hbcm)
      b(27)=16*ccc*(fb3*p3p6*p3p7+fb3*p3p6*p4p7+fb3*p3p6*p5p7)
      b(28)=16*ccc*(fb3*p3p7*p6p8+fb3*p4p7*p6p8+fb3*p5p7*p6p8)
      b(29)=8*ccc*(fb3*p3p6+3*fb3*p4p6-fb3*p5p6)
      b(31)=ccc*(8*w1*(-hbcm*p3p6*p3p7*p4p8-hbcm*p3p6*p4p7*p4p8-hbcm*
     . p3p6*p4p8*p5p7)+8*w16*(-hbcm*p2p8*p3p6*p3p7-hbcm*p2p8*p3p6*
     . p4p7-hbcm*p2p8*p3p6*p5p7+hbcm*p3p6*p3p7*p5p8+hbcm*p3p6*p4p7*
     . p5p8+hbcm*p3p6*p5p7*p5p8)+16*(fb4*hbcm*p3p7*p6p8+fb4*hbcm*p4p7
     . *p6p8+fb4*hbcm*p5p7*p6p8))
      do 200 n=1,32 
         c(n,2)=c(n,2)-0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp26_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      b(1)=16*ccc*(-3*fb3*p2p4*p6p7-fb3*p3p6*p4p7-3*fb3*p4p6*p4p7+fb3
     . *p4p7*p5p6)
      b(2)=ccc*(w2*(16*(p5p8*p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(-3*
     . hbcm*p2p4*p5p8*p6p7-2*hbcm*p4p6*p4p7*p5p8))+w11*(16*(p4p8*p4p7
     . *p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(p4p7*p3p6*p2p8)*(ffmcfmb*hbcm-
     . hbcm)+8*(-3*hbcm*p2p4*p2p8*p6p7+3*hbcm*p2p4*p4p8*p6p7-2*hbcm*
     . p2p8*p4p6*p4p7+2*hbcm*p4p6*p4p7*p4p8))+4*(p6p8*p4p7)*(8*fb3*
     . fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-8*fb4*hbcm+hbcm))
      b(3)=ccc*(12*w2*(p5p8*p4p7)*(-fmb*hbcm-fmc*hbcm+hbcm2)+w11*(12*
     . (p4p8*p4p7)*(fmb*hbcm+fmc*hbcm-hbcm2)+12*(p4p7*p2p8)*(-fmb*
     . hbcm-fmc*hbcm+hbcm2))+16*(-2*fb3*p1p2*p7p8-fb3*p1p8*p4p7+2*fb3
     . *p2p3*p7p8-fb3*p2p4*p7p8+2*fb3*p2p5*p7p8+fb3*p2p8*p4p7-fb3*
     . p4p7*p4p8+3*fb3*p4p7*p5p8))
      b(4)=ccc*(8*p4p7*(2*fb3*ffmcfmb*hbcm2-3*fb3*hbcm2-3*fb4*fmb*
     . hbcm-3*fb4*fmc*hbcm)+16*(2*fb3*p1p2*p3p7+fb3*p1p3*p4p7+fb3*
     . p2p3*p4p7+2*fb3*p2p3*p5p7+fb3*p2p4*p3p7-2*fb3*p2p5*p3p7+fb3*
     . p3p4*p4p7-3*fb3*p3p5*p4p7))
      ans=ccc*(w2*(4*(p5p8*p4p7)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+8*(2*hbcm*p1p2*p3p7*p5p8+hbcm*p1p3*p4p7*p5p8+
     . hbcm*p2p3*p4p7*p5p8+2*hbcm*p2p3*p5p7*p5p8+hbcm*p2p4*p3p7*p5p8-
     . 2*hbcm*p2p5*p3p7*p5p8+hbcm*p3p4*p4p7*p5p8-3*hbcm*p3p5*p4p7*
     . p5p8))+w11*(4*(p4p8*p4p7)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*
     . hbcm2+3*hbcm3)+4*(p4p7*p2p8)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*
     . fmc*hbcm2-3*hbcm3)+8*(2*hbcm*p1p2*p2p8*p3p7-2*hbcm*p1p2*p3p7*
     . p4p8+hbcm*p1p3*p2p8*p4p7-hbcm*p1p3*p4p7*p4p8+hbcm*p2p3*p2p8*
     . p4p7+2*hbcm*p2p3*p2p8*p5p7-hbcm*p2p3*p4p7*p4p8-2*hbcm*p2p3*
     . p4p8*p5p7+hbcm*p2p4*p2p8*p3p7-hbcm*p2p4*p3p7*p4p8-2*hbcm*p2p5*
     . p2p8*p3p7+2*hbcm*p2p5*p3p7*p4p8+hbcm*p2p8*p3p4*p4p7-3*hbcm*
     . p2p8*p3p5*p4p7-hbcm*p3p4*p4p7*p4p8+3*hbcm*p3p5*p4p7*p4p8))+(16
     . *(p3p7*p2p8)*(3*fb3*fmb-3*fb3*fmc-fb4*hbcm)+16*(p7p8*p2p3)*(-3
     . *fb3*fmb+3*fb3*fmc+fb4*hbcm)+16*(2*fb4*hbcm*p1p2*p7p8+fb4*hbcm
     . *p1p8*p4p7+fb4*hbcm*p2p4*p7p8-2*fb4*hbcm*p2p5*p7p8+fb4*hbcm*
     . p2p8*p4p7+2*fb4*hbcm*p2p8*p5p7+fb4*hbcm*p4p7*p4p8-3*fb4*hbcm*
     . p4p7*p5p8)))
      b(5)=ans
      b(6)=16*ccc*(-fb3*p2p4*p6p8-fb3*p2p8*p3p6-3*fb3*p2p8*p4p6+fb3*
     . p2p8*p5p6)
      b(7)=16*ccc*(fb3*p2p3*p3p6+3*fb3*p2p3*p4p6-fb3*p2p3*p5p6+fb3*
     . p2p4*p3p6)
      b(8)=ccc*(w2*(16*(p5p8*p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+8*(2*
     . hbcm*p2p3*p4p6*p5p8+hbcm*p2p4*p3p6*p5p8))+w11*(16*(p4p8*p3p6*
     . p2p3)*(ffmcfmb*hbcm-hbcm)+16*(p3p6*p2p8*p2p3)*(-ffmcfmb*hbcm+
     . hbcm)+8*(2*hbcm*p2p3*p2p8*p4p6-2*hbcm*p2p3*p4p6*p4p8+hbcm*p2p4
     . *p2p8*p3p6-hbcm*p2p4*p3p6*p4p8))+(4*(p3p6*p2p8)*(8*fb3*fmb-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8*p2p3)*(-8*fb3*fmb+4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+8*fb4*hbcm-hbcm)+16*(fb4*hbcm*p2p4*
     . p6p8+2*fb4*hbcm*p2p8*p4p6)))
      b(9)=ccc*(4*(p4p7*p3p6)*(-8*fb3*fmb+4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm+hbcm)+16*(-3*fb4*hbcm*p2p4*p6p7-2*fb4*hbcm*p4p6*p4p7))
      ans=ccc*(w2*(24*(p6p7*p5p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+8*(
     . p5p8*p4p7*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+8
     . *(3*hbcm2*p2p4*p5p8*p6p7+hbcm2*p4p6*p4p7*p5p8+hbcm2*p4p7*p5p6*
     . p5p8))+w11*(24*(p6p7*p4p8*p2p3)*(fmb*hbcm+fmc*hbcm-hbcm2)+24*(
     . p6p7*p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm+hbcm2)+8*(p4p8*p4p7*p3p6)*
     . (3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-hbcm2)+8*(p4p7*p3p6*p2p8)
     . *(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+8*(3*hbcm2*p2p4*
     . p2p8*p6p7-3*hbcm2*p2p4*p4p8*p6p7+hbcm2*p2p8*p4p6*p4p7+hbcm2*
     . p2p8*p4p7*p5p6-hbcm2*p4p6*p4p7*p4p8-hbcm2*p4p7*p4p8*p5p6))+(4*
     . (p6p8*p4p7)*(-4*fb3*ffmcfmb*hbcm2+8*fb4*fmb*hbcm+4*fb4*fmc*
     . hbcm-3*hbcm2)+32*(fb3*p1p3*p4p7*p6p8+fb3*p1p8*p2p3*p6p7-fb3*
     . p1p8*p3p6*p4p7+3*fb3*p2p3*p2p8*p6p7+fb3*p2p3*p4p7*p6p8-3*fb3*
     . p2p3*p4p8*p6p7+fb3*p2p3*p5p8*p6p7+fb3*p2p8*p3p6*p3p7-fb3*p2p8*
     . p3p6*p4p7+3*fb3*p2p8*p3p7*p4p6-fb3*p2p8*p3p7*p5p6-fb3*p3p4*
     . p4p7*p6p8+fb3*p3p5*p4p7*p6p8+fb3*p3p6*p4p7*p4p8-fb3*p3p6*p4p7*
     . p5p8)))
      b(10)=ans
      b(11)=ccc*(4*w2*(-2*hbcm*p1p2*p5p8+2*hbcm*p2p3*p5p8-hbcm*p2p4*
     . p5p8+2*hbcm*p2p5*p5p8)+4*w11*(-2*hbcm*p1p2*p2p8+2*hbcm*p1p2*
     . p4p8+2*hbcm*p2p3*p2p8-2*hbcm*p2p3*p4p8-hbcm*p2p4*p2p8+hbcm*
     . p2p4*p4p8+2*hbcm*p2p5*p2p8-2*hbcm*p2p5*p4p8)+24*p2p8*(-fb3*fmb
     . +fb3*fmc+fb4*hbcm))
      b(12)=8*ccc*(2*fb3*p1p2-2*fb3*p2p3+fb3*p2p4-2*fb3*p2p5)
      b(13)=ccc*(w2*(4*(p5p8*p2p3)*(3*fmb*hbcm+3*fmc*hbcm-hbcm2)+4*(-
     . 2*hbcm2*p1p2*p5p8-hbcm2*p2p4*p5p8+2*hbcm2*p2p5*p5p8))+w11*(4*(
     . p4p8*p2p3)*(-3*fmb*hbcm-3*fmc*hbcm+hbcm2)+4*(p2p8*p2p3)*(3*fmb
     . *hbcm+3*fmc*hbcm-hbcm2)+4*(-2*hbcm2*p1p2*p2p8+2*hbcm2*p1p2*
     . p4p8-hbcm2*p2p4*p2p8+hbcm2*p2p4*p4p8+2*hbcm2*p2p5*p2p8-2*hbcm2
     . *p2p5*p4p8))+(8*p2p8*(-2*fb3*ffmcfmb*hbcm2+3*fb3*hbcm2+3*fb4*
     . fmb*hbcm+3*fb4*fmc*hbcm)+16*(-fb3*p1p3*p2p8+fb3*p1p8*p2p3+fb3*
     . p2p3*p4p8-3*fb3*p2p3*p5p8-fb3*p2p8*p3p4+3*fb3*p2p8*p3p5)))
      b(14)=ccc*(8*p2p3*(3*fb3*fmb-3*fb3*fmc-fb4*hbcm)+8*(-2*fb4*hbcm
     . *p1p2-fb4*hbcm*p2p4+2*fb4*hbcm*p2p5))
      b(15)=ccc*(w2*(4*(p5p8*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*
     . hbcm+hbcm2)+4*(hbcm2*p4p6*p5p8+hbcm2*p5p6*p5p8))+w11*(4*(p4p8*
     . p3p6)*(3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-hbcm2)+4*(p3p6*p2p8
     . )*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+hbcm2)+4*(hbcm2*p2p8*
     . p4p6+hbcm2*p2p8*p5p6-hbcm2*p4p6*p4p8-hbcm2*p4p8*p5p6))+(2*p6p8
     . *(-4*fb3*ffmcfmb*hbcm2+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm-3*hbcm2)+
     . 16*(fb3*p1p3*p6p8-fb3*p1p8*p3p6+fb3*p2p3*p6p8-fb3*p2p8*p3p6-
     . fb3*p3p4*p6p8+fb3*p3p5*p6p8+fb3*p3p6*p4p8-fb3*p3p6*p5p8)))
      b(16)=ccc*(w2*(8*(p5p8*p3p6)*(ffmcfmb*hbcm-hbcm)-8*hbcm*p4p6*
     . p5p8)+w11*(8*(p4p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(-hbcm*p2p8*p4p6+hbcm*p4p6*p4p8))+2*p6p8*
     . (8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-8*fb4*hbcm+hbcm))
      b(17)=ccc*(w2*(2*p5p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-
     . 3*hbcm3)+4*(hbcm*p1p3*p5p8-hbcm*p2p3*p5p8+hbcm*p3p4*p5p8-3*
     . hbcm*p3p5*p5p8))+w11*(2*p4p8*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*
     . fmc*hbcm2+3*hbcm3)+2*p2p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+4*(hbcm*p1p3*p2p8-hbcm*p1p3*p4p8-hbcm*p2p3*p2p8
     . +hbcm*p2p3*p4p8+hbcm*p2p8*p3p4-3*hbcm*p2p8*p3p5-hbcm*p3p4*p4p8
     . +3*hbcm*p3p5*p4p8))+8*(fb4*hbcm*p1p8-fb4*hbcm*p2p8+fb4*hbcm*
     . p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w2*(4*(p5p8*p3p7)*(-3*fmb*hbcm-3*fmc*hbcm+hbcm2)+8*(
     . -hbcm2*p4p7*p5p8-hbcm2*p5p7*p5p8))+w11*(4*(p4p8*p3p7)*(3*fmb*
     . hbcm+3*fmc*hbcm-hbcm2)+4*(p3p7*p2p8)*(-3*fmb*hbcm-3*fmc*hbcm+
     . hbcm2)+8*(-hbcm2*p2p8*p4p7-hbcm2*p2p8*p5p7+hbcm2*p4p7*p4p8+
     . hbcm2*p4p8*p5p7))+(8*p7p8*(2*fb3*ffmcfmb*hbcm2-3*fb3*hbcm2-3*
     . fb4*fmb*hbcm-3*fb4*fmc*hbcm)+16*(fb3*p1p3*p7p8-fb3*p1p8*p3p7-
     . fb3*p2p3*p7p8+fb3*p2p8*p3p7+fb3*p3p4*p7p8-3*fb3*p3p5*p7p8-fb3*
     . p3p7*p4p8+3*fb3*p3p7*p5p8)))
      b(19)=ccc*(6*w2*p5p8*(-fmb*hbcm-fmc*hbcm+hbcm2)+w11*(6*p4p8*(
     . fmb*hbcm+fmc*hbcm-hbcm2)+6*p2p8*(-fmb*hbcm-fmc*hbcm+hbcm2))+8*
     . (-fb3*p1p8+fb3*p2p8-fb3*p4p8+3*fb3*p5p8))
      b(20)=ccc*(8*w2*(-hbcm*p3p7*p5p8-hbcm*p4p7*p5p8-hbcm*p5p7*p5p8)
     . +8*w11*(-hbcm*p2p8*p3p7-hbcm*p2p8*p4p7-hbcm*p2p8*p5p7+hbcm*
     . p3p7*p4p8+hbcm*p4p7*p4p8+hbcm*p4p8*p5p7)+24*p7p8*(fb3*fmb-fb3*
     . fmc-fb4*hbcm))
      b(21)=12*ccc*(fb3*fmb-fb3*fmc-fb4*hbcm)
      b(22)=4*ccc*(2*fb3*ffmcfmb*hbcm2-3*fb3*hbcm2+2*fb3*p1p3-2*fb3*
     . p2p3+2*fb3*p3p4-6*fb3*p3p5-3*fb4*fmb*hbcm-3*fb4*fmc*hbcm)
      b(23)=ccc*(8*p3p7*(-3*fb3*fmb+3*fb3*fmc+fb4*hbcm)+16*(-fb4*hbcm
     . *p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(2*p3p6*(-8*fb3*fmb+4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm
     . )-16*fb4*hbcm*p4p6)
      b(25)=24*ccc*p4p7*(fb3*fmb-fb3*fmc-fb4*hbcm)
      b(27)=ccc*(8*p6p7*(2*fb3*ffmcfmb*hbcm2+fb3*hbcm2-3*fb4*fmb*hbcm
     . -3*fb4*fmc*hbcm)+16*(-fb3*p1p3*p6p7-3*fb3*p2p3*p6p7+3*fb3*p3p4
     . *p6p7-fb3*p3p5*p6p7-fb3*p3p6*p3p7-3*fb3*p3p7*p4p6+fb3*p3p7*
     . p5p6))
      b(28)=ccc*(12*w2*(p6p7*p5p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+w11*(12*
     . (p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+12*(p6p7*p2p8)*(fmb*
     . hbcm+fmc*hbcm-hbcm2))+16*(-fb3*p1p8*p6p7-3*fb3*p2p8*p6p7-fb3*
     . p3p6*p7p8-3*fb3*p4p6*p7p8+3*fb3*p4p8*p6p7+fb3*p5p6*p7p8-fb3*
     . p5p8*p6p7))
      b(29)=8*ccc*(-fb3*p3p6-3*fb3*p4p6+fb3*p5p6)
      b(30)=16*ccc*(-fb3*p3p7-fb3*p4p7-fb3*p5p7)
      ans=ccc*(w2*(4*(p6p7*p5p8)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2+hbcm3)+16*(p5p8*p3p7*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(-hbcm*
     . p1p3*p5p8*p6p7-3*hbcm*p2p3*p5p8*p6p7+3*hbcm*p3p4*p5p8*p6p7-
     . hbcm*p3p5*p5p8*p6p7-2*hbcm*p3p7*p4p6*p5p8))+w11*(4*(p6p7*p4p8)
     . *(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2-hbcm3)+4*(p6p7*p2p8
     . )*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2+hbcm3)+16*(p4p8*
     . p3p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p2p8)*(ffmcfmb*
     . hbcm-hbcm)+8*(-hbcm*p1p3*p2p8*p6p7+hbcm*p1p3*p4p8*p6p7-3*hbcm*
     . p2p3*p2p8*p6p7+3*hbcm*p2p3*p4p8*p6p7+3*hbcm*p2p8*p3p4*p6p7-
     . hbcm*p2p8*p3p5*p6p7-2*hbcm*p2p8*p3p7*p4p6-3*hbcm*p3p4*p4p8*
     . p6p7+hbcm*p3p5*p4p8*p6p7+2*hbcm*p3p7*p4p6*p4p8))+(4*(p6p8*p3p7
     . )*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-8*fb4*hbcm+hbcm)+4*(
     . p7p8*p3p6)*(-8*fb3*fmb+4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+16*(
     . -fb4*hbcm*p1p8*p6p7-3*fb4*hbcm*p2p8*p6p7-2*fb4*hbcm*p4p6*p7p8+
     . 3*fb4*hbcm*p4p8*p6p7-fb4*hbcm*p5p8*p6p7)))
      b(31)=ans
      b(32)=24*ccc*p6p7*(fb3*fmb-fb3*fmc-fb4*hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp25_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3+2*ffmcfmb*p2p3
     . -fmb2+hbcm2+2*p1p2-2*p1p3-2*p2p3))
      b(1)=ccc*(2*p6p7*(-4*fb3*fmb*fmc-4*fb3*fmb2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm)+32*(fb3*p1p2*p6p7-fb3*p2p3*p6p7-2*fb3*p2p4*p6p7
     . ))
      b(2)=ccc*(w1*(4*(p6p7*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*
     . (p6p7*p4p8*p2p3)*(-ffmcfmb*hbcm+hbcm)+8*(p6p7*p4p8*p1p3)*(
     . ffmcfmb*hbcm-hbcm)+8*(-hbcm*p1p2*p4p8*p6p7+2*hbcm*p2p4*p4p8*
     . p6p7))+w20*(4*(p6p7*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*(
     . p6p7*p1p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p6p7*p2p8*p2p3)
     . *(-ffmcfmb*hbcm+hbcm)+8*(p6p7*p2p3*p1p8)*(-ffmcfmb*hbcm+hbcm)+
     . 8*(p6p7*p2p8*p1p3)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p8*p1p3)*(
     . ffmcfmb*hbcm-hbcm)+8*(-hbcm*p1p2*p1p8*p6p7-hbcm*p1p2*p2p8*p6p7
     . +2*hbcm*p1p8*p2p4*p6p7+2*hbcm*p2p4*p2p8*p6p7))+(4*(p6p7*p4p8)*
     . (4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p6p7*p2p8)*(
     . 2*fb3*fmb-4*fb3*fmc-2*fb4*ffmcfmb*hbcm+2*fb4*hbcm+hbcm)+16*(
     . p6p7*p1p8)*(-fb3*fmb-fb4*ffmcfmb*hbcm+fb4*hbcm)))
      b(9)=ccc*(4*(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+4*(p6p7*p2p3)*(-4*fb3*fmb+8*fb3*fmc+8*fb4*ffmcfmb*
     . hbcm-8*fb4*hbcm-hbcm)+4*(p6p7*p1p3)*(4*fb3*fmb-hbcm)+2*p6p7*(-
     . 4*fb3*fmb*hbcm2-4*fb4*ffmcfmb*hbcm3+4*fb4*hbcm3+hbcm3)+16*(fb4
     . *hbcm*p1p2*p6p7-2*fb4*hbcm*p2p4*p6p7))
      ans=ccc*(w1*(8*(p6p7*p4p8*p2p3)*(ffmcfmb*hbcm2+fmb*hbcm+2*fmc*
     . hbcm-hbcm2)+8*(p6p7*p4p8*p1p3)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2)
     . +4*(p6p7*p4p8)*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*
     . hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3))+w20*(8*(p6p7*p2p8*p2p3)
     . *(ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+8*(p6p7*p2p3*p1p8)*
     . (ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+8*(p6p7*p2p8*p1p3)*(
     . -ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p6p7*p1p8*p1p3)*(-ffmcfmb*
     . hbcm2+fmb*hbcm+hbcm2)+4*(p6p7*p2p8)*(ffmcfmb*fmb*hbcm3+ffmcfmb
     . *fmc*hbcm3+fmb*fmc*hbcm2-fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3)+4*(
     . p6p7*p1p8)*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-
     . fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3))+(4*(p6p7*p2p8)*(4*fb3*ffmcfmb
     . *hbcm2-12*fb3*hbcm2+4*fb4*fmb*hbcm-hbcm2)+4*(p6p7*p1p8)*(-4*
     . fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm+hbcm2)+64*(fb3*
     . p1p3*p2p8*p6p7+fb3*p2p3*p2p8*p6p7-fb3*p2p3*p4p8*p6p7-fb3*p2p8*
     . p3p4*p6p7)))
      b(10)=ans
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p1p3*p6p7-fb3*p2p3*p6p7+2*fb3*p3p4*p6p7))
      b(28)=ccc*(8*w1*(p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm)+w20*(8*(p6p7*
     . p2p8)*(-fmb*hbcm-fmc*hbcm)+8*(p6p7*p1p8)*(-fmb*hbcm-fmc*hbcm))
     . +32*(-fb3*p1p8*p6p7-fb3*p2p8*p6p7+2*fb3*p4p8*p6p7))
      b(31)=ccc*(w1*(8*(p6p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)+
     . 16*(hbcm*p1p3*p4p8*p6p7+hbcm*p2p3*p4p8*p6p7-hbcm*p3p4*p4p8*
     . p6p7))+w20*(8*(p6p7*p2p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)+8*
     . (p6p7*p1p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)+16*(hbcm*p1p3*
     . p1p8*p6p7+hbcm*p1p3*p2p8*p6p7+hbcm*p1p8*p2p3*p6p7-hbcm*p1p8*
     . p3p4*p6p7+hbcm*p2p3*p2p8*p6p7-hbcm*p2p8*p3p4*p6p7))+32*(-fb4*
     . hbcm*p1p8*p6p7-fb4*hbcm*p2p8*p6p7+fb4*hbcm*p4p8*p6p7))
      b(32)=4*ccc*p6p7*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
	end 
 
      subroutine amp24_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(-fmb2+
     . fmc2+hbcm2+2*p3p4))
      b(2)=ccc*(w1*(4*(p6p7*p4p8)*(fmb2*hbcm-fmc2*hbcm-hbcm3)-8*hbcm*
     . p3p4*p4p8*p6p7)+32*fb4*hbcm*p2p8*p6p7)
      b(9)=ccc*(8*p6p7*(-fb4*fmb2*hbcm+fb4*fmc2*hbcm+fb4*hbcm3)+16*(-
     . 2*fb4*hbcm*p2p3*p6p7+fb4*hbcm*p3p4*p6p7))
      b(10)=ccc*(w1*(8*(p6p7*p4p8)*(-fmb2*hbcm2+fmc2*hbcm2+hbcm4)+16*
     . hbcm2*p3p4*p4p8*p6p7)+32*(p6p7*p2p8)*(-fb4*fmb*hbcm-fb4*fmc*
     . hbcm))
      b(27)=16*ccc*p6p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(28)=8*ccc*w1*(p6p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(31)=ccc*(w1*(8*(p6p7*p4p8)*(-2*fmb*hbcm2+2*fmc*hbcm2+hbcm3)+
     . 16*hbcm*p3p4*p4p8*p6p7)-32*fb4*hbcm*p4p8*p6p7)
      b(32)=16*ccc*fb4*hbcm*p6p7
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
	end 
 
      subroutine amp23_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(2)=ccc*(w2*(4*(p6p7*p5p8)*(-fmb2*hbcm+fmc2*hbcm-hbcm3)-8*hbcm
     . *p3p5*p5p8*p6p7)+16*(fb4*hbcm*p1p8*p6p7+fb4*hbcm*p2p8*p6p7-fb4
     . *hbcm*p4p8*p6p7-fb4*hbcm*p5p8*p6p7))
      b(9)=ccc*(8*p6p7*(-fb4*fmb2*hbcm+fb4*fmc2*hbcm+fb4*hbcm3)+16*(-
     . fb4*hbcm*p1p3*p6p7-fb4*hbcm*p2p3*p6p7+fb4*hbcm*p3p4*p6p7))
      b(10)=ccc*(w2*(8*(p6p7*p5p8)*(fmb*hbcm3+fmb2*hbcm2+fmc*hbcm3-
     . fmc2*hbcm2-hbcm4)+8*(p6p7*p5p8*p3p5)*(fmb*hbcm+fmc*hbcm)+8*(
     . p6p7*p5p8*p3p4)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p6p7*p5p8*p2p3)
     . *(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p6p7*p5p8*p1p3)*(-fmb*hbcm-fmc
     . *hbcm+2*hbcm2))+(16*(p6p7*p5p8)*(fb4*fmb*hbcm+fb4*fmc*hbcm)+16
     . *(p6p7*p4p8)*(fb4*fmb*hbcm+fb4*fmc*hbcm)+16*(p6p7*p2p8)*(-fb4*
     . fmb*hbcm-fb4*fmc*hbcm)+16*(p6p7*p1p8)*(-fb4*fmb*hbcm-fb4*fmc*
     . hbcm)))
      b(27)=16*ccc*p6p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(28)=8*ccc*w2*(p6p7*p5p8)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)
      b(31)=ccc*(w2*(8*(p6p7*p5p8)*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+16
     . *hbcm*p3p5*p5p8*p6p7)+32*fb4*hbcm*p5p8*p6p7)
      b(32)=16*ccc*fb4*hbcm*p6p7
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.8740730364761186d0*b(n)
200   continue
      return
	end 
 
      subroutine amp22_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-2*
     . ffmcfmb*p2p3-fmc2+2*p1p2))
      b(1)=ccc*(2*p6p7*(4*fb3*fmb*fmc+4*fb3*fmc2+4*fb4*ffmcfmb*fmb*
     . hbcm+4*fb4*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+32*(fb3*p1p2*
     . p6p7-fb3*p2p3*p6p7-2*fb3*p2p4*p6p7))
      b(2)=ccc*(w2*(4*(p6p7*p5p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p6p7*
     . p5p8*p2p3)*(ffmcfmb*hbcm-2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p5p8*
     . p6p7+hbcm*p1p2*p5p8*p6p7-2*hbcm*p2p4*p5p8*p6p7))+w17*(4*(p6p7*
     . p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p6p7*p1p8)*(ffmcfmb*hbcm3-
     . fmc*hbcm2)+8*(p6p7*p2p8*p2p3)*(ffmcfmb*hbcm-2*hbcm)+8*(p6p7*
     . p2p3*p1p8)*(ffmcfmb*hbcm-2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p1p8*
     . p6p7-ffmcfmb*hbcm*p1p3*p2p8*p6p7+hbcm*p1p2*p1p8*p6p7+hbcm*p1p2
     . *p2p8*p6p7-2*hbcm*p1p8*p2p4*p6p7-2*hbcm*p2p4*p2p8*p6p7))+(4*(
     . p6p7*p4p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p2p8)*
     . (8*fb3*fmb+8*fb4*ffmcfmb*hbcm-8*fb4*hbcm-hbcm)-4*hbcm*p1p8*
     . p6p7))
      b(9)=ccc*(4*(p6p7*p3p4)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+2*
     . p6p7*(4*fb3*fmc*hbcm2+4*fb4*ffmcfmb*hbcm3-hbcm3)+8*(p6p7*p2p3)
     . *(-4*fb3*fmb-2*fb4*ffmcfmb*hbcm+hbcm)+16*(-fb4*ffmcfmb*hbcm*
     . p1p3*p6p7+fb4*hbcm*p1p2*p6p7-2*fb4*hbcm*p2p4*p6p7))
      ans=ccc*(w2*(8*(p6p7*p5p8*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*(
     . p6p7*p5p8*p1p3)*(ffmcfmb*hbcm2+fmc*hbcm)+4*(p6p7*p5p8)*(-
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc2*hbcm2))
     . +w17*(8*(p6p7*p2p8*p2p3)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*
     . p2p3*p1p8)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p2p8*p1p3)*(
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p1p8*p1p3)*(ffmcfmb*hbcm2+fmc*
     . hbcm)+4*(p6p7*p2p8)*(-ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3+fmb*
     . fmc*hbcm2+fmc2*hbcm2)+4*(p6p7*p1p8)*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2+fmc2*hbcm2))+(4*(p6p7*p2p8)*(-
     . 4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm-
     . hbcm2)+4*(p6p7*p1p8)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm+hbcm2
     . )+64*(fb3*p1p3*p2p8*p6p7+fb3*p2p3*p2p8*p6p7-fb3*p2p3*p4p8*p6p7
     . -fb3*p2p8*p3p4*p6p7)))
      b(10)=ans
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p1p3*p6p7-fb3*p2p3*p6p7+2*fb3*p3p4*p6p7))
      b(28)=ccc*(8*w2*(p6p7*p5p8)*(fmb*hbcm+fmc*hbcm)+w17*(8*(p6p7*
     . p2p8)*(fmb*hbcm+fmc*hbcm)+8*(p6p7*p1p8)*(fmb*hbcm+fmc*hbcm))+
     . 32*(-fb3*p1p8*p6p7-fb3*p2p8*p6p7+2*fb3*p4p8*p6p7))
      b(31)=ccc*(w2*(8*(p6p7*p5p8)*(-ffmcfmb*hbcm3+fmc*hbcm2+hbcm3)+
     . 16*hbcm*p3p4*p5p8*p6p7)+w17*(8*(p6p7*p2p8)*(-ffmcfmb*hbcm3+fmc
     . *hbcm2+hbcm3)+8*(p6p7*p1p8)*(-ffmcfmb*hbcm3+fmc*hbcm2+hbcm3)+
     . 16*(hbcm*p1p8*p3p4*p6p7+hbcm*p2p8*p3p4*p6p7))+32*fb4*hbcm*p4p8
     . *p6p7)
      b(32)=4*ccc*p6p7*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.1092591295595148d0*b(n)
200   continue
      return
	end 
 
      subroutine amp21_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=ccc*(64*(p4p6*p3p7)*(fb3*ffmcfmb-fb3)+32*(p3p7*p3p6)*(fb3*
     . ffmcfmb-fb3)+16*p6p7*(fb3*ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*hbcm2
     . -fb3*fmb2+fb3*hbcm2)+32*(fb3*p1p2*p6p7-fb3*p2p3*p6p7-fb3*p2p4*
     . p6p7))
      ans2=(8*(p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+8*(p6p8*p4p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+16*(p7p8*p4p6)*(-2*fb3*fmb-2*fb4*ffmcfmb*hbcm+2*fb4*hbcm
     . +hbcm)+8*(p6p7*p1p8)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm
     . +hbcm)+8*(p6p8*p3p7)*(-4*fb3*ffmcfmb*fmb-4*fb4*ffmcfmb**2*hbcm
     . +4*fb4*ffmcfmb*hbcm+hbcm)+8*(p7p8*p3p6)*(-4*fb3*ffmcfmb*fmb-4*
     . fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm+hbcm))
      ans1=w8*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm+hbcm)+8*(p6p7*p2p8)
     . *(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmb2*hbcm-hbcm3)+16*(p3p7*
     . p3p6*p2p8)*(ffmcfmb**2*hbcm-2*ffmcfmb*hbcm+hbcm)+16*(-hbcm*
     . p1p2*p2p8*p6p7+hbcm*p2p3*p2p8*p6p7+hbcm*p2p4*p2p8*p6p7))+w5*(
     . 16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7*p1p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*
     . hbcm3+fmb2*hbcm-hbcm3)+8*(p6p7*p1p8)*(ffmcfmb**2*hbcm3-2*
     . ffmcfmb*hbcm3-fmb2*hbcm+hbcm3)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2
     . *hbcm-2*ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p1p8)*(-ffmcfmb**2*
     . hbcm+2*ffmcfmb*hbcm-hbcm)+16*(hbcm*p1p2*p1p8*p6p7-hbcm*p1p2*
     . p4p8*p6p7-hbcm*p1p8*p2p3*p6p7-hbcm*p1p8*p2p4*p6p7+hbcm*p2p3*
     . p4p8*p6p7+hbcm*p2p4*p4p8*p6p7))+w12*(16*(p4p6*p3p7*p1p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p8)*(ffmcfmb**2*hbcm3-2*ffmcfmb*
     . hbcm3-fmb2*hbcm+hbcm3)+16*(p3p7*p3p6*p1p8)*(-ffmcfmb**2*hbcm+2
     . *ffmcfmb*hbcm-hbcm)+16*(hbcm*p1p2*p1p8*p6p7-hbcm*p1p8*p2p3*
     . p6p7-hbcm*p1p8*p2p4*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w8*(16*(p5p7*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p3p7*p2p8)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+2
     . *fmb*hbcm+fmc*hbcm-hbcm2))+w5*(16*(p5p7*p4p8)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+16*(p5p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +8*(p4p8*p3p7)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+8*(p3p7*p1p8)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+hbcm2)
     . )+w12*(16*(p5p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p7*
     . p1p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-2*fmb*
     . hbcm-fmc*hbcm+hbcm2))+(64*(p5p8*p3p7)*(-fb3*ffmcfmb+fb3)+32*(
     . p3p7*p1p8)*(fb3*ffmcfmb-fb3)+4*p7p8*(4*fb3*ffmcfmb**2*hbcm2-8*
     . fb3*ffmcfmb*hbcm2-4*fb3*fmb2+4*fb3*hbcm2+fmb*hbcm+fmc*hbcm)+32
     . *(fb3*p2p5*p7p8+fb3*p2p8*p3p7+3*fb3*p2p8*p5p7)))
      b(3)=ans
      b(4)=ccc*(64*(p3p7*p3p5)*(fb3*ffmcfmb-fb3)+64*(p3p7*p2p3)*(-fb3
     . *ffmcfmb+fb3)+32*(p3p7*p1p3)*(-fb3*ffmcfmb+fb3)+8*p5p7*(-4*fb3
     . *ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+4*p3p7*(-4*
     . fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*hbcm2+4*fb3*fmb2-8*fb3*
     . hbcm2+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-8*fb4*fmb*
     . hbcm-4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p2p3*p5p7-fb3*p2p5*p3p7))
      ans2=w12*(8*(p5p7*p1p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p3p7*p3p5*p1p8)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p2p3*p1p8)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p3p7*p1p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+8*
     . (p3p7*p1p8)*(-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*hbcm2+fmb2*
     . hbcm+fmc*hbcm2-hbcm3)+16*(hbcm*p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*
     . p3p7))+(32*(p3p7*p1p8)*(-fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(p5p8*
     . p3p7)*(4*fb3*fmb+8*fb4*ffmcfmb*hbcm-8*fb4*hbcm-hbcm)+16*(p7p8*
     . p3p5)*(-2*fb3*fmb-2*fb4*ffmcfmb*hbcm+2*fb4*hbcm+hbcm)+8*(p3p7*
     . p2p8)*(-4*fb3*fmb-8*fb4*ffmcfmb*hbcm+8*fb4*hbcm+hbcm)+16*(p7p8
     . *p2p3)*(2*fb3*fmb+2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm)+4*p7p8*(
     . 8*fb3*ffmcfmb*fmb*hbcm2-8*fb3*fmb*hbcm2+4*fb4*ffmcfmb**2*hbcm3
     . -8*fb4*ffmcfmb*hbcm3+4*fb4*fmb2*hbcm+4*fb4*hbcm3-ffmcfmb*hbcm3
     . -fmc*hbcm2+3*hbcm3)+8*(-4*fb4*hbcm*p2p5*p7p8+4*fb4*hbcm*p2p8*
     . p5p7-hbcm*p1p3*p7p8))
      ans1=w8*(8*(p5p7*p2p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p3p7
     . *p3p5*p2p8)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*p2p8*p2p3)*(ffmcfmb*
     . hbcm-hbcm)+16*(p3p7*p2p8*p1p3)*(ffmcfmb*hbcm-hbcm)+8*(p3p7*
     . p2p8)*(ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*hbcm2-fmb2*hbcm-fmc
     . *hbcm2+hbcm3)+16*(-hbcm*p2p3*p2p8*p5p7+hbcm*p2p5*p2p8*p3p7))+
     . w5*(8*(p5p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(p5p7*p1p8
     . )*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p8*p3p7*p3p5)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p4p8*p3p7*p2p3)*(ffmcfmb*hbcm-hbcm)+16*
     . (p3p7*p3p5*p1p8)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p2p3*p1p8)*(-
     . ffmcfmb*hbcm+hbcm)+16*(p4p8*p3p7*p1p3)*(ffmcfmb*hbcm-hbcm)+16*
     . (p3p7*p1p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7)*(ffmcfmb*
     . fmc*hbcm2-ffmcfmb*hbcm3+fmb*hbcm2-fmb2*hbcm-fmc*hbcm2+hbcm3)+8
     . *(p3p7*p1p8)*(-ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3-fmb*hbcm2+fmb2*
     . hbcm+fmc*hbcm2-hbcm3)+16*(hbcm*p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*
     . p3p7-hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p3p7*p4p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w8*(16*(p4p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+w5*(16*(p4p8*p4p6)*
     . (ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+16*(p4p6*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)
     . +8*(p3p6*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2))+w12*(16*(p4p6*
     . p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p6*p1p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2))+(16*p6p8*(fb3*ffmcfmb**2*hbcm2-2*fb3*
     . ffmcfmb*hbcm2-fb3*fmb2+fb3*hbcm2)+32*(-fb3*ffmcfmb*p2p8*p3p6+
     . fb3*p1p2*p6p8-fb3*p2p3*p6p8-fb3*p2p4*p6p8-fb3*p2p8*p4p6)))
      b(7)=ccc*(32*(p3p6*p2p3)*(fb3*ffmcfmb+fb3)+8*p4p6*(-4*fb3*
     . ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+4*p3p6*(-4*fb3
     . *ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2+4*fb3*fmb2-4*fb4*fmb*
     . hbcm-hbcm2)+32*(-fb3*p1p2*p3p6+fb3*p2p3*p4p6+fb3*p2p4*p3p6))
      ans2=(8*(p6p8*p2p3)*(4*fb4*hbcm+hbcm)+8*(p4p8*p3p6)*(-4*fb3*fmb
     . -4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p8*p3p4)*(4*fb3*fmb+
     . 4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p3p6*p1p8)*(4*fb3*fmb+4
     . *fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p6p8*p1p3)*(-4*fb3*fmb-4
     . *fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*p6p8*(8*fb3*ffmcfmb*fmb*
     . hbcm2+4*fb4*ffmcfmb**2*hbcm3+4*fb4*fmb2*hbcm-4*fb4*hbcm3-
     . ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+32*(fb4*ffmcfmb*hbcm*p2p8*p3p6-
     . fb4*hbcm*p1p2*p6p8+fb4*hbcm*p2p4*p6p8+fb4*hbcm*p2p8*p4p6))
      ans1=w8*(8*(p4p6*p2p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p3p6
     . *p2p8*p2p3)*(-ffmcfmb*hbcm-hbcm)+8*(p3p6*p2p8)*(-ffmcfmb*fmb*
     . hbcm2+fmb*hbcm2-fmb2*hbcm)+16*(hbcm*p1p2*p2p8*p3p6-hbcm*p2p3*
     . p2p8*p4p6-hbcm*p2p4*p2p8*p3p6))+w5*(8*(p4p8*p4p6)*(ffmcfmb*
     . hbcm3+fmb*hbcm2-hbcm3)+8*(p4p6*p1p8)*(-ffmcfmb*hbcm3-fmb*hbcm2
     . +hbcm3)+16*(p4p8*p3p6*p2p3)*(-ffmcfmb*hbcm-hbcm)+16*(p3p6*p2p3
     . *p1p8)*(ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p6)*(-ffmcfmb*fmb*hbcm2+
     . fmb*hbcm2-fmb2*hbcm)+8*(p3p6*p1p8)*(ffmcfmb*fmb*hbcm2-fmb*
     . hbcm2+fmb2*hbcm)+16*(-hbcm*p1p2*p1p8*p3p6+hbcm*p1p2*p3p6*p4p8+
     . hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p4*p3p6-hbcm*p2p3*p4p6*p4p8-
     . hbcm*p2p4*p3p6*p4p8))+w12*(8*(p4p6*p1p8)*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+16*(p3p6*p2p3*p1p8)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*
     . p1p8)*(ffmcfmb*fmb*hbcm2-fmb*hbcm2+fmb2*hbcm)+16*(-hbcm*p1p2*
     . p1p8*p3p6+hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p4*p3p6))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(8*(p4p6*p3p7)*(4*fb3*fmb+8*fb4*ffmcfmb*hbcm-8*fb4*
     . hbcm-hbcm)+8*(p4p7*p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+8*(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+8*(p6p7*p1p3)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+16*(p3p7*p3p6)*(4*fb3*ffmcfmb*fmb+2*fb4*ffmcfmb**2*
     . hbcm-2*fb4*hbcm-ffmcfmb*hbcm)+8*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2
     . -2*fb4*ffmcfmb**2*hbcm3-2*fb4*fmb2*hbcm+2*fb4*hbcm3+ffmcfmb*
     . hbcm3)+32*(fb4*hbcm*p1p2*p6p7-fb4*hbcm*p2p3*p6p7-fb4*hbcm*p2p4
     . *p6p7))
      ans4=(8*(p7p8*p3p6)*(-4*fb3*fmb2-4*fb4*ffmcfmb*fmb*hbcm+4*fb4*
     . fmb*hbcm-fmb*hbcm+hbcm2)+64*(p4p8*p3p7*p3p6)*(-fb3*ffmcfmb+fb3
     . )+64*(p6p8*p3p7*p3p4)*(fb3*ffmcfmb-fb3)+64*(p7p8*p3p6*p2p3)*(-
     . fb3*ffmcfmb-fb3)+64*(p3p7*p3p6*p1p8)*(fb3*ffmcfmb-fb3)+64*(
     . p6p8*p3p7*p1p3)*(-fb3*ffmcfmb+fb3)+8*(p6p7*p4p8)*(4*fb3*
     . ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb*hbcm+hbcm2)+8*(p6p8*p4p7)*
     . (-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+8*(
     . p7p8*p4p6)*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb*hbcm+
     . hbcm2)+8*(p6p8*p3p7)*(4*fb3*ffmcfmb*hbcm2+4*fb3*fmb2-4*fb3*
     . hbcm2-4*fb4*ffmcfmb*fmb*hbcm-fmb*hbcm-hbcm2)+8*(p6p7*p1p8)*(-4
     . *fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+64*(-fb3*
     . ffmcfmb*hbcm2*p2p8*p6p7+3*fb3*ffmcfmb*p2p8*p3p6*p3p7+fb3*p1p2*
     . p3p6*p7p8-fb3*p1p2*p3p7*p6p8+fb3*p1p3*p2p8*p6p7+fb3*p2p3*p3p7*
     . p6p8-fb3*p2p3*p4p6*p7p8-fb3*p2p4*p3p6*p7p8+fb3*p2p4*p3p7*p6p8-
     . fb3*p2p8*p3p4*p6p7+fb3*p2p8*p3p6*p4p7+2*fb3*p2p8*p3p7*p4p6))
      ans3=w12*(16*(p4p6*p3p7*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16
     . *(p4p7*p3p6*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*
     . p3p4*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p1p8*p1p3)*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p3p7*p3p6*p1p8)*(-2*
     . ffmcfmb*fmb*hbcm+fmb*hbcm)+8*(p6p7*p1p8)*(ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmb*hbcm3+fmb2*hbcm2-hbcm4)+16*(-hbcm2*p1p2*p1p8*p6p7+
     . hbcm2*p1p8*p2p3*p6p7+hbcm2*p1p8*p2p4*p6p7))+ans4
      ans2=w5*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16
     . *(p4p8*p4p7*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p4p8
     . *p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p6*p3p7*p1p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p3p6*p1p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p3p4*p1p8)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+16*(p6p7*p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)
     . +16*(p6p7*p1p8*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p8*
     . p3p7*p3p6)*(2*ffmcfmb*fmb*hbcm-fmb*hbcm)+16*(p3p7*p3p6*p1p8)*(
     . -2*ffmcfmb*fmb*hbcm+fmb*hbcm)+8*(p6p7*p4p8)*(-ffmcfmb**2*hbcm4
     . -2*ffmcfmb*fmb*hbcm3-fmb2*hbcm2+hbcm4)+8*(p6p7*p1p8)*(ffmcfmb
     . **2*hbcm4+2*ffmcfmb*fmb*hbcm3+fmb2*hbcm2-hbcm4)+16*(-hbcm2*
     . p1p2*p1p8*p6p7+hbcm2*p1p2*p4p8*p6p7+hbcm2*p1p8*p2p3*p6p7+hbcm2
     . *p1p8*p2p4*p6p7-hbcm2*p2p3*p4p8*p6p7-hbcm2*p2p4*p4p8*p6p7))+
     . ans3
      ans1=w8*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16
     . *(p4p7*p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p3p4
     . *p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p2p8*p1p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p3p7*p3p6*p2p8)*(2*ffmcfmb*
     . fmb*hbcm-fmb*hbcm)+8*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4-2*ffmcfmb*
     . fmb*hbcm3-fmb2*hbcm2+hbcm4)+16*(hbcm2*p1p2*p2p8*p6p7-hbcm2*
     . p2p3*p2p8*p6p7-hbcm2*p2p4*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w8*(4*p2p8*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2+2*
     . fmb2*hbcm+fmc*hbcm2-hbcm3)+8*(hbcm*p1p2*p2p8-hbcm*p2p3*p2p8-3*
     . hbcm*p2p5*p2p8))+w5*(4*p4p8*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3-fmb*fmc*hbcm-fmb*hbcm2
     . +2*fmb2*hbcm+fmc*hbcm2-hbcm3)+4*p1p8*(ffmcfmb**2*hbcm3-ffmcfmb
     . *fmb*hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*
     . hbcm2-2*fmb2*hbcm-fmc*hbcm2+hbcm3)+8*(-hbcm*p1p2*p1p8+hbcm*
     . p1p2*p4p8+hbcm*p1p8*p2p3+3*hbcm*p1p8*p2p5-hbcm*p2p3*p4p8-3*
     . hbcm*p2p5*p4p8))+w12*(4*p1p8*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*
     . hbcm2+ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3+fmb*fmc*hbcm+fmb*hbcm2
     . -2*fmb2*hbcm-fmc*hbcm2+hbcm3)+8*(-hbcm*p1p2*p1p8+hbcm*p1p8*
     . p2p3+3*hbcm*p1p8*p2p5))+(12*p5p8*(-4*fb3*fmb-4*fb4*ffmcfmb*
     . hbcm+4*fb4*hbcm+hbcm)+8*p2p8*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+4*p1p8*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm
     . -hbcm)))
      b(11)=ans
      b(12)=2*ccc*(-8*fb3*ffmcfmb**2*hbcm2+16*fb3*ffmcfmb*hbcm2+4*fb3
     . *fmb*fmc+12*fb3*fmb2-8*fb3*hbcm2+8*fb3*p1p2-8*fb3*p2p3-32*fb3*
     . p2p5+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmb*
     . hbcm-4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm)
      ans2=w12*(16*(p3p5*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p2p3
     . *p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2)+8*(p1p8*p1p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p1p8*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4-fmb*fmc*hbcm2+3*fmb*hbcm3-
     . fmb2*hbcm2+fmc*hbcm3-2*hbcm4)+8*(-hbcm2*p1p2*p1p8+2*hbcm2*p1p8
     . *p2p5))+(8*p5p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb*
     . hbcm+hbcm2)+4*p2p8*(-12*fb3*ffmcfmb*hbcm2+12*fb3*hbcm2-4*fb4*
     . fmb*hbcm-hbcm2)+4*p1p8*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4
     . *fmb*hbcm-hbcm2)+32*(-fb3*p2p3*p5p8+fb3*p2p8*p3p5))
      ans1=w8*(16*(p3p5*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p2p8
     . *p2p3)*(ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+8*(p2p8*p1p3)*(ffmcfmb
     . *hbcm2+fmb*hbcm-hbcm2)+4*p2p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*
     . hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2-3*fmb*hbcm3+fmb2*hbcm2-fmc
     . *hbcm3+2*hbcm4)+8*(hbcm2*p1p2*p2p8-2*hbcm2*p2p5*p2p8))+w5*(16*
     . (p4p8*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p8*p2p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm-2*hbcm2)+16*(p3p5*p1p8)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2)+8*(p2p3*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+2*hbcm2
     . )+8*(p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p1p8*p1p3)*(
     . -ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*p4p8*(ffmcfmb*fmb*hbcm3+
     . ffmcfmb*fmc*hbcm3-2*ffmcfmb*hbcm4+fmb*fmc*hbcm2-3*fmb*hbcm3+
     . fmb2*hbcm2-fmc*hbcm3+2*hbcm4)+4*p1p8*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3+2*ffmcfmb*hbcm4-fmb*fmc*hbcm2+3*fmb*hbcm3-
     . fmb2*hbcm2+fmc*hbcm3-2*hbcm4)+8*(-hbcm2*p1p2*p1p8+hbcm2*p1p2*
     . p4p8+2*hbcm2*p1p8*p2p5-2*hbcm2*p2p5*p4p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(12*p3p5*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+8*p2p3*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+6*fb4*hbcm+hbcm)+4
     . *p1p3*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*(-12*
     . fb3*ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*hbcm2+20*fb3*fmb*hbcm2
     . -4*fb3*fmc*hbcm2-4*fb4*ffmcfmb**2*hbcm3+16*fb4*ffmcfmb*hbcm3+4
     . *fb4*fmb*fmc*hbcm-8*fb4*fmb2*hbcm-8*fb4*hbcm*p1p2+24*fb4*hbcm*
     . p2p5-12*fb4*hbcm3+2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-4*hbcm3)
     . )
      b(15)=ccc*(8*w8*(-hbcm2*p2p8*p3p6-2*hbcm2*p2p8*p4p6)+8*w5*(
     . hbcm2*p1p8*p3p6+2*hbcm2*p1p8*p4p6-hbcm2*p3p6*p4p8-2*hbcm2*p4p6
     . *p4p8)+8*w12*(hbcm2*p1p8*p3p6+2*hbcm2*p1p8*p4p6)+32*(-fb3*
     . ffmcfmb*hbcm2*p6p8+fb3*p1p3*p6p8-fb3*p1p8*p3p6-fb3*p3p4*p6p8+
     . fb3*p3p6*p4p8))
      b(16)=ccc*(w8*(8*(p3p6*p2p8)*(ffmcfmb*hbcm+hbcm)+24*hbcm*p2p8*
     . p4p6)+w5*(8*(p4p8*p3p6)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p8)*(-
     . ffmcfmb*hbcm-hbcm)+24*(-hbcm*p1p8*p4p6+hbcm*p4p6*p4p8))+w12*(8
     . *(p3p6*p1p8)*(-ffmcfmb*hbcm-hbcm)-24*hbcm*p1p8*p4p6)-4*hbcm*
     . p6p8)
      b(17)=ccc*(w8*(4*p2p8*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+4*
     . hbcm3)+8*(-hbcm*p1p3*p2p8-2*hbcm*p2p3*p2p8+3*hbcm*p2p8*p3p5))+
     . w5*(4*p4p8*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+4*hbcm3)+4*
     . p1p8*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-4*hbcm3)+8*(hbcm*
     . p1p3*p1p8-hbcm*p1p3*p4p8+2*hbcm*p1p8*p2p3-3*hbcm*p1p8*p3p5-2*
     . hbcm*p2p3*p4p8+3*hbcm*p3p5*p4p8))+w12*(4*p1p8*(2*ffmcfmb*hbcm3
     . -fmb*hbcm2+fmc*hbcm2-4*hbcm3)+8*(hbcm*p1p3*p1p8+2*hbcm*p1p8*
     . p2p3-3*hbcm*p1p8*p3p5))+16*(fb4*hbcm*p1p8+2*fb4*hbcm*p2p8-3*
     . fb4*hbcm*p5p8))
      b(18)=ccc*(8*w8*(hbcm2*p2p8*p3p7+2*hbcm2*p2p8*p5p7)+8*w5*(-
     . hbcm2*p1p8*p3p7-2*hbcm2*p1p8*p5p7+hbcm2*p3p7*p4p8+2*hbcm2*p4p8
     . *p5p7)+8*w12*(-hbcm2*p1p8*p3p7-2*hbcm2*p1p8*p5p7)+(32*p7p8*(
     . fb3*ffmcfmb*hbcm2-fb3*hbcm2)+32*(fb3*p2p3*p7p8-fb3*p2p8*p3p7-
     . fb3*p3p5*p7p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w8*p2p8*(fmb*hbcm+fmc*hbcm)+w5*(4*p4p8*(fmb*hbcm+
     . fmc*hbcm)+4*p1p8*(-fmb*hbcm-fmc*hbcm))+4*w12*p1p8*(-fmb*hbcm-
     . fmc*hbcm)+16*(-fb3*p1p8-3*fb3*p2p8+4*fb3*p5p8))
      b(20)=ccc*(w8*(8*(p3p7*p2p8)*(-ffmcfmb*hbcm+2*hbcm)+24*hbcm*
     . p2p8*p5p7)+w5*(8*(p4p8*p3p7)*(-ffmcfmb*hbcm+2*hbcm)+8*(p3p7*
     . p1p8)*(ffmcfmb*hbcm-2*hbcm)+24*(-hbcm*p1p8*p5p7+hbcm*p4p8*p5p7
     . ))+w12*(8*(p3p7*p1p8)*(ffmcfmb*hbcm-2*hbcm)-24*hbcm*p1p8*p5p7)
     . +4*hbcm*p7p8)
      b(21)=8*ccc*(fb3*fmb-fb3*fmc)
      b(22)=8*ccc*(4*fb3*ffmcfmb*hbcm2-6*fb3*hbcm2+2*fb3*p1p3+6*fb3*
     . p2p3-8*fb3*p3p5-fb4*fmb*hbcm-fb4*fmc*hbcm)
      b(23)=ccc*(16*p3p7*(fb4*ffmcfmb*hbcm-2*fb4*hbcm)-48*fb4*hbcm*
     . p5p7)
      b(24)=ccc*(16*p3p6*(-fb4*ffmcfmb*hbcm-fb4*hbcm)-48*fb4*hbcm*
     . p4p6)
      b(25)=ccc*(12*p5p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+4*p3p7*(-8*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc+12*fb3*fmb-
     . 4*fb3*fmc-4*fb4*ffmcfmb**2*hbcm+12*fb4*ffmcfmb*hbcm-8*fb4*hbcm
     . +ffmcfmb*hbcm-2*hbcm))
      b(26)=ccc*(12*p4p6*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+4*p3p6*(4*fb3*ffmcfmb*fmb+4*fb3*fmb+4*fb4*ffmcfmb**2*
     . hbcm-4*fb4*hbcm-ffmcfmb*hbcm-hbcm))
      b(27)=32*ccc*(fb3*ffmcfmb*hbcm2*p6p7-2*fb3*ffmcfmb*p3p6*p3p7-
     . fb3*p1p3*p6p7+fb3*p3p4*p6p7-fb3*p3p6*p4p7-fb3*p3p7*p4p6)
      b(28)=32*ccc*(-fb3*ffmcfmb*p3p6*p7p8-fb3*ffmcfmb*p3p7*p6p8-fb3*
     . p1p8*p6p7-fb3*p4p6*p7p8-fb3*p4p7*p6p8+fb3*p4p8*p6p7)
      b(29)=ccc*(16*p3p6*(-2*fb3*ffmcfmb-fb3)-64*fb3*p4p6)
      b(30)=ccc*(16*p3p7*(2*fb3*ffmcfmb-3*fb3)-64*fb3*p5p7)
      b(31)=ccc*(16*w8*(2*ffmcfmb*hbcm*p2p8*p3p6*p3p7-ffmcfmb*hbcm3*
     . p2p8*p6p7+hbcm*p1p3*p2p8*p6p7-hbcm*p2p8*p3p4*p6p7+hbcm*p2p8*
     . p3p6*p4p7+hbcm*p2p8*p3p7*p4p6)+16*w5*(-2*ffmcfmb*hbcm*p1p8*
     . p3p6*p3p7+2*ffmcfmb*hbcm*p3p6*p3p7*p4p8+ffmcfmb*hbcm3*p1p8*
     . p6p7-ffmcfmb*hbcm3*p4p8*p6p7-hbcm*p1p3*p1p8*p6p7+hbcm*p1p3*
     . p4p8*p6p7+hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p4p7-hbcm*p1p8*
     . p3p7*p4p6-hbcm*p3p4*p4p8*p6p7+hbcm*p3p6*p4p7*p4p8+hbcm*p3p7*
     . p4p6*p4p8)+16*w12*(-2*ffmcfmb*hbcm*p1p8*p3p6*p3p7+ffmcfmb*
     . hbcm3*p1p8*p6p7-hbcm*p1p3*p1p8*p6p7+hbcm*p1p8*p3p4*p6p7-hbcm*
     . p1p8*p3p6*p4p7-hbcm*p1p8*p3p7*p4p6)+(8*(p6p8*p3p7)*(-4*fb4*
     . ffmcfmb*hbcm-hbcm)+8*(p7p8*p3p6)*(-4*fb4*ffmcfmb*hbcm-hbcm)+32
     . *(-fb4*hbcm*p1p8*p6p7-fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*p6p8+
     . fb4*hbcm*p4p8*p6p7)))
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp20_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*
     . hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3)*(
     . ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+2*ffmcfmb*p3p5+fmc2+2*p4p5))
      b(1)=ccc*(64*(p4p6*p3p7)*(fb3*ffmcfmb-fb3)+32*(p4p7*p3p6)*(-fb3
     . *ffmcfmb+fb3)+16*p6p7*(fb3*ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*
     . hbcm2-fb3*fmc2)+32*(fb3*ffmcfmb*p1p3*p6p7-fb3*ffmcfmb*p3p6*
     . p5p7+fb3*p1p4*p6p7-fb3*p3p4*p6p7+fb3*p3p7*p5p6-fb3*p4p6*p5p7+
     . fb3*p4p7*p5p6))
      ans3=(8*(p7p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+32*(p6p8*p5p7)*(-fb3*
     . fmc-fb4*ffmcfmb*hbcm)+32*(p6p8*p4p7)*(-fb3*fmc-fb4*ffmcfmb*
     . hbcm)+8*(p6p7*p2p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(
     . p6p8*p3p7)*(-4*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc+4*fb3*fmb-8*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm+hbcm)+8*(hbcm*p1p8*
     . p6p7+hbcm*p4p6*p7p8))
      ans2=w1*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p8*p4p7
     . *p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3+2*
     . ffmcfmb*hbcm3+fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-
     . ffmcfmb*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p4p8*p6p7+ffmcfmb*hbcm*
     . p3p6*p4p8*p5p7-hbcm*p1p4*p4p8*p6p7+hbcm*p3p4*p4p8*p6p7-hbcm*
     . p3p7*p4p8*p5p6+hbcm*p4p6*p4p8*p5p7-hbcm*p4p7*p4p8*p5p6))+w13*(
     . 16*(p4p6*p3p7*p1p8)*(-ffmcfmb*hbcm+hbcm)+16*(p4p7*p3p6*p1p8)*(
     . ffmcfmb*hbcm-hbcm)+8*(p6p7*p1p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*
     . hbcm3+fmc2*hbcm)+16*(p3p7*p3p6*p1p8)*(ffmcfmb**2*hbcm-ffmcfmb*
     . hbcm)+16*(-ffmcfmb*hbcm*p1p3*p1p8*p6p7+ffmcfmb*hbcm*p1p8*p3p6*
     . p5p7-hbcm*p1p4*p1p8*p6p7+hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p7*
     . p5p6+hbcm*p1p8*p4p6*p5p7-hbcm*p1p8*p4p7*p5p6))+ans3
      ans1=w18*(16*(p5p8*p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p8*
     . p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p5p8*p4p7*p3p6)*(ffmcfmb*
     . hbcm-hbcm)+16*(p4p8*p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*
     . p5p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*hbcm)+8*(p6p7*
     . p4p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*hbcm)+16*(p5p8*
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(p4p8*p3p7*p3p6)*
     . (ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(-ffmcfmb*hbcm*p1p3*p4p8*
     . p6p7-ffmcfmb*hbcm*p1p3*p5p8*p6p7+ffmcfmb*hbcm*p3p6*p4p8*p5p7+
     . ffmcfmb*hbcm*p3p6*p5p7*p5p8-hbcm*p1p4*p4p8*p6p7-hbcm*p1p4*p5p8
     . *p6p7+hbcm*p3p4*p4p8*p6p7+hbcm*p3p4*p5p8*p6p7-hbcm*p3p7*p4p8*
     . p5p6-hbcm*p3p7*p5p6*p5p8+hbcm*p4p6*p4p8*p5p7+hbcm*p4p6*p5p7*
     . p5p8-hbcm*p4p7*p4p8*p5p6-hbcm*p4p7*p5p6*p5p8))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(4*p7p8*(4*fb3*ffmcfmb**2*hbcm2-8*fb3*ffmcfmb*hbcm2-4*fb3*
     . fmc2+fmb*hbcm+fmc*hbcm)+32*(fb3*ffmcfmb*p1p3*p7p8+fb3*ffmcfmb*
     . p1p8*p3p7-2*fb3*ffmcfmb*p3p7*p5p8+fb3*p1p4*p7p8+2*fb3*p1p8*
     . p4p7-fb3*p1p8*p5p7-fb3*p3p4*p7p8+fb3*p3p7*p4p8-3*fb3*p4p7*p5p8
     . +3*fb3*p4p8*p5p7))
      ans1=w18*(8*(p5p8*p5p7)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(
     . p5p7*p4p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p5p8*p4p7)*(
     . ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p4p8*p4p7)*(ffmcfmb*
     . hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p3p7)*(-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+w1*
     . (8*(p5p7*p4p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p4p8*
     . p4p7)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p4p8*p3p7)*(-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+w13
     . *(8*(p5p7*p1p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p4p7*
     . p1p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p3p7*p1p8)*(-
     . ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+
     . ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=ccc*(4*p5p7*(-4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+4*fb4*fmb*
     . hbcm+8*fb4*fmc*hbcm-hbcm2)+4*p4p7*(-4*fb3*ffmcfmb*hbcm2+16*fb3
     . *hbcm2+4*fb4*fmb*hbcm+8*fb4*fmc*hbcm-hbcm2)+4*p3p7*(-4*fb3*
     . ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*hbcm2+4*fb3*fmc2+4*fb4*ffmcfmb
     . *fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm-hbcm2)+32*(-2*
     . fb3*ffmcfmb*p1p3*p3p7-2*fb3*ffmcfmb*p2p3*p3p7+2*fb3*ffmcfmb*
     . p3p5*p3p7-2*fb3*p1p3*p4p7+fb3*p1p3*p5p7-fb3*p1p4*p3p7-3*fb3*
     . p2p3*p4p7+fb3*p2p3*p5p7-3*fb3*p3p4*p5p7+3*fb3*p3p5*p4p7))
      ans3=(8*(p5p8*p3p7)*(4*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+8*(p4p8
     . *p3p7)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+16*(p7p8
     . *p3p5)*(-2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm)+8*(p7p8*p3p4)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p3p7*p2p8)*(-4*
     . fb3*fmc-8*fb4*ffmcfmb*hbcm+hbcm)+16*(p7p8*p2p3)*(2*fb3*fmc+2*
     . fb4*ffmcfmb*hbcm-hbcm)+32*(p3p7*p1p8)*(-fb3*fmc-fb4*ffmcfmb*
     . hbcm)+8*(p7p8*p1p3)*(4*fb3*fmc-hbcm)+4*p7p8*(-8*fb3*fmc*hbcm2-
     . 4*fb4*ffmcfmb**2*hbcm3+4*fb4*fmc2*hbcm+ffmcfmb*hbcm3+fmb*hbcm2
     . +2*hbcm3)+32*(-fb4*hbcm*p1p4*p7p8-fb4*hbcm*p1p8*p4p7+fb4*hbcm*
     . p1p8*p5p7-2*fb4*hbcm*p2p8*p4p7+fb4*hbcm*p2p8*p5p7+2*fb4*hbcm*
     . p4p7*p5p8-2*fb4*hbcm*p4p8*p5p7))
      ans2=w1*(8*(p5p7*p4p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+8*(p4p8*
     . p4p7)*(-fmb*hbcm2+fmc*hbcm2-2*hbcm3)+8*(p4p8*p3p7)*(-ffmcfmb*
     . fmb*hbcm2-ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+16*(ffmcfmb*hbcm*
     . p1p3*p3p7*p4p8+ffmcfmb*hbcm*p2p3*p3p7*p4p8-ffmcfmb*hbcm*p3p5*
     . p3p7*p4p8+hbcm*p1p3*p4p7*p4p8-hbcm*p1p3*p4p8*p5p7+hbcm*p1p4*
     . p3p7*p4p8+2*hbcm*p2p3*p4p7*p4p8-hbcm*p2p3*p4p8*p5p7+2*hbcm*
     . p3p4*p4p8*p5p7-2*hbcm*p3p5*p4p7*p4p8))+w13*(8*(p5p7*p1p8)*(-
     . fmb*hbcm2+fmc*hbcm2+2*hbcm3)+8*(p4p7*p1p8)*(-fmb*hbcm2+fmc*
     . hbcm2-2*hbcm3)+8*(p3p7*p1p8)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3
     . +fmc*hbcm2-fmc2*hbcm)+16*(ffmcfmb*hbcm*p1p3*p1p8*p3p7+ffmcfmb*
     . hbcm*p1p8*p2p3*p3p7-ffmcfmb*hbcm*p1p8*p3p5*p3p7+hbcm*p1p3*p1p8
     . *p4p7-hbcm*p1p3*p1p8*p5p7+hbcm*p1p4*p1p8*p3p7+2*hbcm*p1p8*p2p3
     . *p4p7-hbcm*p1p8*p2p3*p5p7+2*hbcm*p1p8*p3p4*p5p7-2*hbcm*p1p8*
     . p3p5*p4p7))+ans3
      ans1=w18*(8*(p5p8*p5p7)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+8*(p5p7*
     . p4p8)*(-fmb*hbcm2+fmc*hbcm2+2*hbcm3)+8*(p5p8*p4p7)*(-fmb*hbcm2
     . +fmc*hbcm2-2*hbcm3)+8*(p4p8*p4p7)*(-fmb*hbcm2+fmc*hbcm2-2*
     . hbcm3)+8*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*hbcm3+fmc*
     . hbcm2-fmc2*hbcm)+8*(p4p8*p3p7)*(-ffmcfmb*fmb*hbcm2-ffmcfmb*
     . hbcm3+fmc*hbcm2-fmc2*hbcm)+16*(ffmcfmb*hbcm*p1p3*p3p7*p4p8+
     . ffmcfmb*hbcm*p1p3*p3p7*p5p8+ffmcfmb*hbcm*p2p3*p3p7*p4p8+
     . ffmcfmb*hbcm*p2p3*p3p7*p5p8-ffmcfmb*hbcm*p3p5*p3p7*p4p8-
     . ffmcfmb*hbcm*p3p5*p3p7*p5p8+hbcm*p1p3*p4p7*p4p8+hbcm*p1p3*p4p7
     . *p5p8-hbcm*p1p3*p4p8*p5p7-hbcm*p1p3*p5p7*p5p8+hbcm*p1p4*p3p7*
     . p4p8+hbcm*p1p4*p3p7*p5p8+2*hbcm*p2p3*p4p7*p4p8+2*hbcm*p2p3*
     . p4p7*p5p8-hbcm*p2p3*p4p8*p5p7-hbcm*p2p3*p5p7*p5p8+2*hbcm*p3p4*
     . p4p8*p5p7+2*hbcm*p3p4*p5p7*p5p8-2*hbcm*p3p5*p4p7*p4p8-2*hbcm*
     . p3p5*p4p7*p5p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w18*(8*(p5p8*p5p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p6*
     . p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p8*p4p6)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+8*(p4p8*p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p8*p3p6)
     . *(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmc*
     . hbcm))+w1*(8*(p5p6*p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p4p6
     . )*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmc*
     . hbcm))+w13*(8*(p5p6*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*
     . p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p3p6*p1p8)*(ffmcfmb*hbcm2-
     . fmc*hbcm))+(32*(p4p8*p3p6)*(-fb3*ffmcfmb+fb3)+16*p6p8*(fb3*
     . ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*hbcm2+2*fb3*fmb2-3*fb3*fmc2)+32
     . *(fb3*ffmcfmb*p1p3*p6p8+fb3*ffmcfmb*p1p8*p3p6-fb3*ffmcfmb*p3p6
     . *p5p8+2*fb3*p1p4*p6p8-fb3*p1p5*p6p8+fb3*p1p8*p4p6-2*fb3*p3p4*
     . p6p8+fb3*p3p5*p6p8-fb3*p4p6*p5p8+fb3*p4p8*p5p6)))
      b(6)=ans
      b(7)=ccc*(32*(p3p6*p3p5)*(fb3*ffmcfmb-fb3)+32*(p3p6*p3p4)*(fb3*
     . ffmcfmb+fb3)+4*p5p6*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2
     . )+4*p4p6*(-4*fb3*ffmcfmb*hbcm2+8*fb3*hbcm2+4*fb4*fmc*hbcm-
     . hbcm2)+4*p3p6*(-4*fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*hbcm2-8*
     . fb3*fmb2+12*fb3*fmc2+4*fb4*fmc*hbcm-hbcm2)+32*(-2*fb3*ffmcfmb*
     . p1p3*p3p6-fb3*p1p3*p4p6-2*fb3*p1p4*p3p6+fb3*p1p5*p3p6-fb3*p3p4
     . *p5p6+fb3*p3p5*p4p6))
      ans3=(8*(p5p8*p3p6)*(-4*fb3*fmc+hbcm)+8*(p4p8*p3p6)*(-4*fb3*fmc
     . -4*fb4*hbcm+hbcm)+32*(p6p8*p3p5)*(fb3*fmc+fb4*ffmcfmb*hbcm-fb4
     . *hbcm)+32*(p6p8*p3p4)*(fb3*fmc+fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(
     . p3p6*p1p8)*(4*fb3*fmc-hbcm)+32*(p6p8*p1p3)*(-fb3*fmc-fb4*
     . ffmcfmb*hbcm)+32*p6p8*(fb3*fmc*hbcm2+fb4*ffmcfmb*hbcm3-fb4*
     . fmb2*hbcm+fb4*fmc2*hbcm)+32*(-fb4*hbcm*p1p4*p6p8+fb4*hbcm*p1p5
     . *p6p8-fb4*hbcm*p1p8*p4p6+fb4*hbcm*p4p6*p5p8-fb4*hbcm*p4p8*p5p6
     . ))
      ans2=w1*(8*(p5p6*p4p8)*(ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p8*p4p6)*
     . (ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)+8*(p4p8*p3p6)*(-ffmcfmb*
     . hbcm3+2*fmb2*hbcm+fmc*hbcm2-2*fmc2*hbcm)+16*(p4p8*p3p6*p3p5)*(
     . -ffmcfmb*hbcm+hbcm)+16*(ffmcfmb*hbcm*p1p3*p3p6*p4p8-ffmcfmb*
     . hbcm*p3p4*p3p6*p4p8+hbcm*p1p3*p4p6*p4p8+hbcm*p1p4*p3p6*p4p8-
     . hbcm*p1p5*p3p6*p4p8+hbcm*p3p4*p4p8*p5p6-hbcm*p3p5*p4p6*p4p8))+
     . w13*(8*(p5p6*p1p8)*(ffmcfmb*hbcm3+fmc*hbcm2)+8*(p4p6*p1p8)*(
     . ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)+8*(p3p6*p1p8)*(-ffmcfmb*hbcm3
     . +2*fmb2*hbcm+fmc*hbcm2-2*fmc2*hbcm)+16*(p3p6*p3p5*p1p8)*(-
     . ffmcfmb*hbcm+hbcm)+16*(ffmcfmb*hbcm*p1p3*p1p8*p3p6-ffmcfmb*
     . hbcm*p1p8*p3p4*p3p6+hbcm*p1p3*p1p8*p4p6+hbcm*p1p4*p1p8*p3p6-
     . hbcm*p1p5*p1p8*p3p6+hbcm*p1p8*p3p4*p5p6-hbcm*p1p8*p3p5*p4p6))+
     . ans3
      ans1=w18*(8*(p5p8*p5p6)*(ffmcfmb*hbcm3+fmc*hbcm2)+8*(p5p6*p4p8)
     . *(ffmcfmb*hbcm3+fmc*hbcm2)+8*(p5p8*p4p6)*(ffmcfmb*hbcm3+fmc*
     . hbcm2-2*hbcm3)+8*(p4p8*p4p6)*(ffmcfmb*hbcm3+fmc*hbcm2-2*hbcm3)
     . +8*(p5p8*p3p6)*(-ffmcfmb*hbcm3+2*fmb2*hbcm+fmc*hbcm2-2*fmc2*
     . hbcm)+8*(p4p8*p3p6)*(-ffmcfmb*hbcm3+2*fmb2*hbcm+fmc*hbcm2-2*
     . fmc2*hbcm)+16*(p5p8*p3p6*p3p5)*(-ffmcfmb*hbcm+hbcm)+16*(p4p8*
     . p3p6*p3p5)*(-ffmcfmb*hbcm+hbcm)+16*(ffmcfmb*hbcm*p1p3*p3p6*
     . p4p8+ffmcfmb*hbcm*p1p3*p3p6*p5p8-ffmcfmb*hbcm*p3p4*p3p6*p4p8-
     . ffmcfmb*hbcm*p3p4*p3p6*p5p8+hbcm*p1p3*p4p6*p4p8+hbcm*p1p3*p4p6
     . *p5p8+hbcm*p1p4*p3p6*p4p8+hbcm*p1p4*p3p6*p5p8-hbcm*p1p5*p3p6*
     . p4p8-hbcm*p1p5*p3p6*p5p8+hbcm*p3p4*p4p8*p5p6+hbcm*p3p4*p5p6*
     . p5p8-hbcm*p3p5*p4p6*p4p8-hbcm*p3p5*p4p6*p5p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(32*(p4p6*p3p7)*(fb4*ffmcfmb*hbcm-fb4*hbcm)+16*p6p7*(
     . fb4*ffmcfmb**2*hbcm3-2*fb4*ffmcfmb*hbcm3-fb4*fmc2*hbcm)+8*(
     . p5p7*p3p6)*(4*fb3*fmc-hbcm)+8*(p4p7*p3p6)*(4*fb3*fmc+4*fb4*
     . hbcm-hbcm)+8*(p6p7*p2p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8
     . *(p3p7*p3p6)*(4*fb3*ffmcfmb*fmb-4*fb3*ffmcfmb*fmc-4*fb3*fmb+8*
     . fb3*fmc-4*fb4*ffmcfmb**2*hbcm+8*fb4*ffmcfmb*hbcm-hbcm)+32*(fb4
     . *ffmcfmb*hbcm*p1p3*p6p7+fb4*hbcm*p1p4*p6p7-fb4*hbcm*p3p4*p6p7+
     . fb4*hbcm*p3p7*p5p6-fb4*hbcm*p4p6*p5p7+fb4*hbcm*p4p7*p5p6))
      ans4=8*(8*fb3*ffmcfmb*p1p3*p3p6*p7p8-8*fb3*ffmcfmb*p1p3*p3p7*
     . p6p8+16*fb3*ffmcfmb*p2p8*p3p6*p3p7-8*fb3*ffmcfmb*p3p4*p3p6*
     . p7p8+8*fb3*ffmcfmb*p3p4*p3p7*p6p8+8*fb3*p1p3*p4p6*p7p8-8*fb3*
     . p1p3*p4p7*p6p8+8*fb3*p1p4*p3p6*p7p8-8*fb3*p1p4*p3p7*p6p8-8*fb3
     . *p1p5*p3p6*p7p8+8*fb3*p1p5*p3p7*p6p8+8*fb3*p1p8*p3p6*p4p7-8*
     . fb3*p1p8*p3p7*p4p6-8*fb3*p2p8*p3p4*p6p7+16*fb3*p2p8*p3p6*p4p7-
     . 8*fb3*p2p8*p3p6*p5p7+8*fb3*p2p8*p3p7*p4p6+8*fb3*p3p4*p5p6*p7p8
     . -8*fb3*p3p4*p5p7*p6p8-8*fb3*p3p5*p4p6*p7p8+8*fb3*p3p5*p4p7*
     . p6p8-8*fb3*p3p6*p4p7*p5p8+8*fb3*p3p6*p4p8*p5p7+8*fb3*p3p7*p4p6
     . *p5p8-8*fb3*p3p7*p4p8*p5p6-hbcm2*p1p8*p6p7)
      ans3=64*(p5p8*p3p7*p3p6)*(fb3*ffmcfmb-fb3)+64*(p7p8*p3p6*p3p5)*
     . (-fb3*ffmcfmb+fb3)+32*(p6p8*p5p7)*(-fb3*ffmcfmb*hbcm2+fb4*fmc*
     . hbcm)+8*(p7p8*p5p6)*(4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)
     . +32*(p6p8*p4p7)*(-fb3*ffmcfmb*hbcm2+2*fb3*hbcm2+fb4*fmc*hbcm)+
     . 8*(p7p8*p4p6)*(4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2-4*fb4*fmc*hbcm+
     . hbcm2)+8*(p7p8*p3p6)*(-4*fb3*ffmcfmb*hbcm2+8*fb3*fmb2-8*fb3*
     . fmc2-4*fb4*fmc*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+8*(p6p7*p2p8)*(-4*
     . fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p8*p3p7)*(-4*fb3*
     . ffmcfmb**2*hbcm2+8*fb3*ffmcfmb*hbcm2-8*fb3*fmb2+8*fb3*fmc2-4*
     . fb4*ffmcfmb*fmb*hbcm+4*fb4*fmb*hbcm+4*fb4*fmc*hbcm+ffmcfmb*
     . hbcm2-hbcm2)+ans4
      ans2=w1*(16*(p4p8*p4p7*p3p6)*(-fmc*hbcm+hbcm2)+16*(p5p6*p4p8*
     . p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+16*(p4p8*p4p6*p3p7)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p4p8)*(ffmcfmb**2*hbcm4-2*
     . ffmcfmb*hbcm4-fmc2*hbcm2)+16*(p4p8*p3p7*p3p6)*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16*(ffmcfmb*
     . hbcm2*p1p3*p4p8*p6p7-fmc*hbcm*p3p6*p4p8*p5p7+hbcm2*p1p4*p4p8*
     . p6p7-hbcm2*p3p4*p4p8*p6p7-hbcm2*p4p6*p4p8*p5p7+hbcm2*p4p7*p4p8
     . *p5p6))+w13*(16*(p4p7*p3p6*p1p8)*(-fmc*hbcm+hbcm2)+16*(p5p6*
     . p3p7*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+16*(p4p6*p3p7*p1p8)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p1p8)*(ffmcfmb**2*hbcm4-2*
     . ffmcfmb*hbcm4-fmc2*hbcm2)+16*(p3p7*p3p6*p1p8)*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16*(ffmcfmb*
     . hbcm2*p1p3*p1p8*p6p7-fmc*hbcm*p1p8*p3p6*p5p7+hbcm2*p1p4*p1p8*
     . p6p7-hbcm2*p1p8*p3p4*p6p7-hbcm2*p1p8*p4p6*p5p7+hbcm2*p1p8*p4p7
     . *p5p6))+ans3
      ans1=w18*(16*(p5p8*p4p7*p3p6)*(-fmc*hbcm+hbcm2)+16*(p4p8*p4p7*
     . p3p6)*(-fmc*hbcm+hbcm2)+16*(p5p8*p5p6*p3p7)*(-ffmcfmb*hbcm2+
     . fmc*hbcm+hbcm2)+16*(p5p6*p4p8*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm+
     . hbcm2)+16*(p5p8*p4p6*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+16*(p4p8*
     . p4p6*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p5p8)*(ffmcfmb**2
     . *hbcm4-2*ffmcfmb*hbcm4-fmc2*hbcm2)+8*(p6p7*p4p8)*(ffmcfmb**2*
     . hbcm4-2*ffmcfmb*hbcm4-fmc2*hbcm2)+16*(p5p8*p3p7*p3p6)*(-
     . ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-fmb*hbcm)+16*(
     . p4p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2-fmb*hbcm)+16*(ffmcfmb*hbcm2*p1p3*p4p8*p6p7+ffmcfmb*hbcm2
     . *p1p3*p5p8*p6p7-fmc*hbcm*p3p6*p4p8*p5p7-fmc*hbcm*p3p6*p5p7*
     . p5p8+hbcm2*p1p4*p4p8*p6p7+hbcm2*p1p4*p5p8*p6p7-hbcm2*p3p4*p4p8
     . *p6p7-hbcm2*p3p4*p5p8*p6p7-hbcm2*p4p6*p4p8*p5p7-hbcm2*p4p6*
     . p5p7*p5p8+hbcm2*p4p7*p4p8*p5p6+hbcm2*p4p7*p5p6*p5p8))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w18*(4*p5p8*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-2*fmb2*
     . hbcm+3*fmc2*hbcm)+4*p4p8*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-2*
     . fmb2*hbcm+3*fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p4p8-ffmcfmb*hbcm
     . *p1p3*p5p8-2*hbcm*p1p4*p4p8-2*hbcm*p1p4*p5p8+hbcm*p1p5*p4p8+
     . hbcm*p1p5*p5p8+2*hbcm*p3p4*p4p8+2*hbcm*p3p4*p5p8-hbcm*p3p5*
     . p4p8-hbcm*p3p5*p5p8))+w1*(4*p4p8*(-ffmcfmb**2*hbcm3+2*ffmcfmb*
     . hbcm3-2*fmb2*hbcm+3*fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p4p8-2*
     . hbcm*p1p4*p4p8+hbcm*p1p5*p4p8+2*hbcm*p3p4*p4p8-hbcm*p3p5*p4p8)
     . )+w13*(4*p1p8*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3-2*fmb2*hbcm+3
     . *fmc2*hbcm)+8*(-ffmcfmb*hbcm*p1p3*p1p8-2*hbcm*p1p4*p1p8+hbcm*
     . p1p5*p1p8+2*hbcm*p1p8*p3p4-hbcm*p1p8*p3p5))+(8*p5p8*(2*fb3*fmb
     . -6*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*p4p8*(2*fb3*fmb-6*fb3*
     . fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p2p8*(-4*fb3*fmb+16*fb3*fmc+12*
     . fb4*ffmcfmb*hbcm-3*hbcm)+4*p1p8*(-4*fb3*fmb+12*fb3*fmc+8*fb4*
     . ffmcfmb*hbcm-hbcm)))
      b(11)=ans
      b(12)=16*ccc*(-fb3*ffmcfmb**2*hbcm2+2*fb3*ffmcfmb*hbcm2-2*fb3*
     . ffmcfmb*p1p3-fb3*fmb2+2*fb3*fmc2-3*fb3*p1p4+fb3*p1p5+3*fb3*
     . p3p4-fb3*p3p5)
      ans2=w13*(8*p1p8*(-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*hbcm2+2*fmc*
     . hbcm3+fmc2*hbcm2)+8*(p3p5*p1p8)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc
     . *hbcm-hbcm2)+8*(p3p4*p1p8)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm
     . +hbcm2)+8*(p2p3*p1p8)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm)+8*
     . (p1p8*p1p3)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-hbcm2*p1p4
     . *p1p8+hbcm2*p1p5*p1p8))+(4*p5p8*(4*fb3*ffmcfmb*hbcm2+8*fb3*
     . hbcm2-4*fb4*fmb*hbcm-8*fb4*fmc*hbcm+hbcm2)+4*p4p8*(4*fb3*
     . ffmcfmb*hbcm2-16*fb3*hbcm2-4*fb4*fmb*hbcm-8*fb4*fmc*hbcm+hbcm2
     . )+8*p2p8*(-4*fb3*ffmcfmb*hbcm2+2*fb4*fmb*hbcm+6*fb4*fmc*hbcm-
     . hbcm2)+4*p1p8*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmb*hbcm+8*fb4*fmc*
     . hbcm-hbcm2)+32*(2*fb3*p1p3*p4p8-fb3*p1p3*p5p8-2*fb3*p1p8*p3p4+
     . fb3*p1p8*p3p5+3*fb3*p2p3*p4p8-fb3*p2p3*p5p8-3*fb3*p2p8*p3p4+
     . fb3*p2p8*p3p5+3*fb3*p3p4*p5p8-3*fb3*p3p5*p4p8))
      ans1=w18*(8*p5p8*(-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*hbcm2+2*fmc*
     . hbcm3+fmc2*hbcm2)+8*p4p8*(-ffmcfmb*hbcm4+fmb*hbcm3-fmb2*hbcm2+
     . 2*fmc*hbcm3+fmc2*hbcm2)+8*(p5p8*p3p5)*(-ffmcfmb*hbcm2+fmb*hbcm
     . +2*fmc*hbcm-hbcm2)+8*(p4p8*p3p5)*(-ffmcfmb*hbcm2+fmb*hbcm+2*
     . fmc*hbcm-hbcm2)+8*(p5p8*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*
     . hbcm+hbcm2)+8*(p4p8*p3p4)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+
     . hbcm2)+8*(p5p8*p2p3)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm)+8*(
     . p4p8*p2p3)*(2*ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm)+8*(p5p8*p1p3)
     . *(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(p4p8*p1p3)*(ffmcfmb*
     . hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-hbcm2*p1p4*p4p8-hbcm2*p1p4*p5p8
     . +hbcm2*p1p5*p4p8+hbcm2*p1p5*p5p8))+w1*(8*p4p8*(-ffmcfmb*hbcm4+
     . fmb*hbcm3-fmb2*hbcm2+2*fmc*hbcm3+fmc2*hbcm2)+8*(p4p8*p3p5)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+8*(p4p8*p3p4)*(-
     . ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm+hbcm2)+8*(p4p8*p2p3)*(2*
     . ffmcfmb*hbcm2-fmb*hbcm-3*fmc*hbcm)+8*(p4p8*p1p3)*(ffmcfmb*
     . hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-hbcm2*p1p4*p4p8+hbcm2*p1p5*p4p8
     . ))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(8*p3p5*(-2*fb3*fmb+6*fb3*fmc+4*fb4*ffmcfmb*hbcm+2*
     . fb4*hbcm-hbcm)+8*p3p4*(-2*fb3*fmb+6*fb3*fmc+4*fb4*ffmcfmb*hbcm
     . -4*fb4*hbcm-hbcm)+4*p2p3*(4*fb3*fmb-16*fb3*fmc-12*fb4*ffmcfmb*
     . hbcm+3*hbcm)+8*p1p3*(2*fb3*fmb-6*fb3*fmc-2*fb4*ffmcfmb*hbcm+
     . hbcm)+8*(-2*fb3*fmb*hbcm2+6*fb3*fmc*hbcm2+fb4*ffmcfmb**2*hbcm3
     . +2*fb4*ffmcfmb*hbcm3+2*fb4*fmb2*hbcm-3*fb4*fmc2*hbcm+4*fb4*
     . hbcm*p1p4-2*fb4*hbcm*p1p5-hbcm3))
      b(15)=ccc*(w18*(8*(p5p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*
     . hbcm)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-
     . hbcm2*p4p6*p4p8-hbcm2*p4p6*p5p8+hbcm2*p4p8*p5p6+hbcm2*p5p6*
     . p5p8))+w1*(8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8
     . *(-hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+w13*(8*(p3p6*p1p8)*(
     . ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm)+8*(-hbcm2*p1p8*p4p6+hbcm2*
     . p1p8*p5p6))+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmb*hbcm+8*fb4
     . *fmc*hbcm-hbcm2)+32*(-2*fb3*p3p4*p6p8+fb3*p3p5*p6p8+2*fb3*p3p6
     . *p4p8-fb3*p3p6*p5p8)))
      b(16)=ccc*(8*w18*(ffmcfmb*hbcm*p3p6*p4p8+ffmcfmb*hbcm*p3p6*p5p8
     . +2*hbcm*p4p6*p4p8+2*hbcm*p4p6*p5p8-hbcm*p4p8*p5p6-hbcm*p5p6*
     . p5p8)+8*w1*(ffmcfmb*hbcm*p3p6*p4p8+2*hbcm*p4p6*p4p8-hbcm*p4p8*
     . p5p6)+8*w13*(ffmcfmb*hbcm*p1p8*p3p6+2*hbcm*p1p8*p4p6-hbcm*p1p8
     . *p5p6)+4*p6p8*(4*fb3*fmb-12*fb3*fmc-8*fb4*ffmcfmb*hbcm+hbcm))
      b(17)=ccc*(w18*(4*p5p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+4
     . *p4p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+8*(-2*hbcm*p3p4*
     . p4p8-2*hbcm*p3p4*p5p8+hbcm*p3p5*p4p8+hbcm*p3p5*p5p8))+w1*(4*
     . p4p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2)+8*(-2*hbcm*p3p4*
     . p4p8+hbcm*p3p5*p4p8))+w13*(4*p1p8*(-ffmcfmb*hbcm3+fmb*hbcm2-2*
     . fmc*hbcm2)+8*(-2*hbcm*p1p8*p3p4+hbcm*p1p8*p3p5))+16*(2*fb4*
     . hbcm*p4p8-fb4*hbcm*p5p8))
      b(18)=ccc*(w18*(8*(p5p8*p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p8
     . *p3p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p4p7*p4p8-hbcm2*
     . p4p7*p5p8+hbcm2*p4p8*p5p7+hbcm2*p5p7*p5p8))+w1*(8*(p4p8*p3p7)*
     . (-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p4p7*p4p8+hbcm2*p4p8*p5p7)
     . )+w13*(8*(p3p7*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(-hbcm2*p1p8*
     . p4p7+hbcm2*p1p8*p5p7))+(4*p7p8*(4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*
     . hbcm+hbcm2)+32*(fb3*p3p4*p7p8-fb3*p3p7*p4p8)))
      b(19)=ccc*(w18*(4*p5p8*(-2*ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm)+4
     . *p4p8*(-2*ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm))+4*w1*p4p8*(-2*
     . ffmcfmb*hbcm2+fmb*hbcm+3*fmc*hbcm)+4*w13*p1p8*(-2*ffmcfmb*
     . hbcm2+fmb*hbcm+3*fmc*hbcm)+16*(-3*fb3*p4p8+fb3*p5p8))
      b(20)=ccc*(8*w18*(-ffmcfmb*hbcm*p3p7*p4p8-ffmcfmb*hbcm*p3p7*
     . p5p8-2*hbcm*p4p7*p4p8-2*hbcm*p4p7*p5p8+hbcm*p4p8*p5p7+hbcm*
     . p5p7*p5p8)+8*w1*(-ffmcfmb*hbcm*p3p7*p4p8-2*hbcm*p4p7*p4p8+hbcm
     . *p4p8*p5p7)+8*w13*(-ffmcfmb*hbcm*p1p8*p3p7-2*hbcm*p1p8*p4p7+
     . hbcm*p1p8*p5p7)+8*p7p8*(-2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm))
      b(21)=2*ccc*(4*fb3*fmb-16*fb3*fmc-12*fb4*ffmcfmb*hbcm+3*hbcm)
      b(22)=4*ccc*(4*fb3*ffmcfmb*hbcm2+12*fb3*p3p4-4*fb3*p3p5-2*fb4*
     . fmb*hbcm-6*fb4*fmc*hbcm+hbcm2)
      b(23)=ccc*(4*p3p7*(4*fb3*fmc+8*fb4*ffmcfmb*hbcm-hbcm)+16*(2*fb4
     . *hbcm*p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(8*p3p6*(-2*fb3*fmb+6*fb3*fmc+2*fb4*ffmcfmb*hbcm-hbcm
     . )+16*(-2*fb4*hbcm*p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(8*p5p7*(-2*fb3*fmb+6*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm
     . )+8*p4p7*(-2*fb3*fmb+6*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p3p7
     . *(-4*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*fmc+4*fb3*fmc+4*fb4*ffmcfmb
     . **2*hbcm+4*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm-hbcm))
      b(26)=ccc*(4*p5p6*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p4p6*(4
     . *fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p3p6*(4*fb3*fmc+4*fb4*
     . ffmcfmb*hbcm-hbcm))
      b(27)=ccc*(4*p6p7*(4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+32
     . *(-2*fb3*ffmcfmb*p3p6*p3p7+fb3*p3p4*p6p7-2*fb3*p3p6*p4p7+fb3*
     . p3p6*p5p7-fb3*p3p7*p4p6))
      b(28)=ccc*(w18*(8*(p6p7*p5p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*
     . p4p8)*(ffmcfmb*hbcm2-fmc*hbcm))+8*w1*(p6p7*p4p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+8*w13*(p6p7*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+32*
     . (-fb3*ffmcfmb*p3p6*p7p8-fb3*ffmcfmb*p3p7*p6p8-fb3*p4p6*p7p8-2*
     . fb3*p4p7*p6p8+fb3*p4p8*p6p7+fb3*p5p7*p6p8))
      b(29)=16*ccc*(-2*fb3*ffmcfmb*p3p6-3*fb3*p4p6+fb3*p5p6)
      b(30)=16*ccc*(2*fb3*ffmcfmb*p3p7+3*fb3*p4p7-fb3*p5p7)
      ans=ccc*(w18*(8*(p6p7*p5p8)*(-ffmcfmb*hbcm3-fmc*hbcm2)+8*(p6p7*
     . p4p8)*(-ffmcfmb*hbcm3-fmc*hbcm2)+16*(ffmcfmb*hbcm*p3p6*p3p7*
     . p4p8+ffmcfmb*hbcm*p3p6*p3p7*p5p8-hbcm*p3p4*p4p8*p6p7-hbcm*p3p4
     . *p5p8*p6p7+hbcm*p3p6*p4p7*p4p8+hbcm*p3p6*p4p7*p5p8-hbcm*p3p6*
     . p4p8*p5p7-hbcm*p3p6*p5p7*p5p8+hbcm*p3p7*p4p6*p4p8+hbcm*p3p7*
     . p4p6*p5p8))+w1*(8*(p6p7*p4p8)*(-ffmcfmb*hbcm3-fmc*hbcm2)+16*(
     . ffmcfmb*hbcm*p3p6*p3p7*p4p8-hbcm*p3p4*p4p8*p6p7+hbcm*p3p6*p4p7
     . *p4p8-hbcm*p3p6*p4p8*p5p7+hbcm*p3p7*p4p6*p4p8))+w13*(8*(p6p7*
     . p1p8)*(-ffmcfmb*hbcm3-fmc*hbcm2)+16*(ffmcfmb*hbcm*p1p8*p3p6*
     . p3p7-hbcm*p1p8*p3p4*p6p7+hbcm*p1p8*p3p6*p4p7-hbcm*p1p8*p3p6*
     . p5p7+hbcm*p1p8*p3p7*p4p6))+(32*(p6p8*p3p7)*(-fb3*fmc-fb4*
     . ffmcfmb*hbcm)+8*(p7p8*p3p6)*(4*fb3*fmc-hbcm)+32*(-fb4*hbcm*
     . p4p6*p7p8-fb4*hbcm*p4p7*p6p8+fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p7*
     . p6p8)))
      b(31)=ans
      b(32)=4*ccc*p6p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.05883183899358491d0*b(n)
         c(n,3)=c(n,3)-0.2077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp2_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(25) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,25 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p4*p2p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      b(3)=w5*ccc*(8*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(p5p7*
     . p1p8)*(-fmb*hbcm-fmc*hbcm+2*hbcm2))
      b(4)=16*ccc*p5p7*(-fb4*fmb*hbcm-fb4*fmc*hbcm)
      b(5)=ccc*(w5*(8*(p5p7*p4p8)*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*(
     . p5p7*p1p8)*(-2*fmb*hbcm2+2*fmc*hbcm2-hbcm3)+16*(hbcm*p1p8*p2p3
     . *p5p7-hbcm*p1p8*p3p5*p5p7-hbcm*p2p3*p4p8*p5p7+hbcm*p3p5*p4p8*
     . p5p7))+32*(fb4*hbcm*p2p8*p5p7-fb4*hbcm*p5p7*p5p8))
      b(11)=8*ccc*w5*(-hbcm*p1p8*p2p5+hbcm*p2p5*p4p8)
      b(13)=16*ccc*w5*(-hbcm2*p1p8*p2p5+hbcm2*p2p5*p4p8)
      b(14)=-16*ccc*fb4*hbcm*p2p5
      b(17)=ccc*(w5*(4*p4p8*(-2*fmb*hbcm2+2*fmc*hbcm2-hbcm3)+4*p1p8*(
     . 2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*(-hbcm*p1p8*p2p3+hbcm*p1p8*
     . p3p5+hbcm*p2p3*p4p8-hbcm*p3p5*p4p8))+16*(-fb4*hbcm*p2p8+fb4*
     . hbcm*p5p8))
      b(19)=w5*ccc*(4*p4p8*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+4*p1p8*(fmb*
     . hbcm+fmc*hbcm-2*hbcm2))
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(25)=-16*ccc*fb4*hbcm*p5p7
      do 200 n=1,25 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp1s2_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(14) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,14 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=-16*ccc*fb3*p6p7
      b(2)=4*ccc*(hbcm*p4p8*p6p7*w1-hbcm*p5p8*p6p7*w2)
      b(3)=-16*ccc*fb3*p7p8
      b(4)=16*ccc*fb3*p3p7
      b(5)=8*ccc*(2*fb4*hbcm*p7p8-hbcm*p3p7*p4p8*w1+hbcm*p3p7*p5p8*w2
     . )
      b(6)=-16*ccc*fb3*p6p8
      b(7)=16*ccc*fb3*p3p6
      b(8)=8*ccc*(2*fb4*hbcm*p6p8-hbcm*p3p6*p4p8*w1+hbcm*p3p6*p5p8*w2
     . )
      b(9)=-8*ccc*fb4*hbcm*p6p7
      b(11)=6*ccc*(hbcm*p4p8*w1-hbcm*p5p8*w2)
      b(12)=12*ccc*fb3
      b(13)=6*ccc*(hbcm2*p4p8*w1-hbcm2*p5p8*w2)
      b(14)=-12*ccc*fb4*hbcm
      do 200 n=1,14 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
	end 
 
      subroutine amp28_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2))
      b(1)=ccc*(16*(p6p7*p2p3)*(-fb3*ffmcfmb+fb3)+4*p6p7*(2*fb3*
     . ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*hbcm2+4*fb3*fmb*fmc+2*fb3*fmc2+
     . 4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc*hbcm-
     . fmb*hbcm-fmc*hbcm+hbcm2)+16*(fb3*ffmcfmb*p3p6*p3p7+3*fb3*
     . ffmcfmb*p3p7*p4p6-fb3*ffmcfmb*p3p7*p5p6+fb3*p1p2*p6p7-fb3*p2p4
     . *p6p7+fb3*p2p5*p6p7))
      ans3=(4*(p6p7*p5p8)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8
     . *p5p7)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8*p5p6)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p4p8)*(-4*fb3*fmc-4*
     . fb4*ffmcfmb*hbcm+hbcm)+4*(p6p8*p4p7)*(4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm-hbcm)+4*(p7p8*p4p6)*(12*fb3*fmc+12*fb4*ffmcfmb*hbcm-hbcm)
     . +4*(p7p8*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-2*ffmcfmb*hbcm+
     . hbcm)+8*(p6p7*p2p8)*(2*fb3*fmc+2*fb4*ffmcfmb*hbcm-hbcm)+4*(
     . p6p7*p1p8)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8*p3p7)*(
     . -8*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*fmc-4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm-2*ffmcfmb*hbcm+hbcm))
      ans2=w11*(8*(p6p7*p4p8*p2p3)*(ffmcfmb*hbcm-hbcm)+8*(p6p7*p2p8*
     . p2p3)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+2*fmb*
     . fmc*hbcm-2*fmc*hbcm2-fmc2*hbcm)+4*(p6p7*p2p8)*(ffmcfmb**2*
     . hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-
     . 2*fmb*fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(
     . ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2
     . *hbcm+ffmcfmb*hbcm)+8*(2*ffmcfmb*hbcm*p2p8*p3p7*p4p6-2*ffmcfmb
     . *hbcm*p3p7*p4p6*p4p8+hbcm*p1p2*p2p8*p6p7-hbcm*p1p2*p4p8*p6p7-
     . hbcm*p2p4*p2p8*p6p7+hbcm*p2p4*p4p8*p6p7+hbcm*p2p5*p2p8*p6p7-
     . hbcm*p2p5*p4p8*p6p7))+ans3
      ans1=w2*(8*(p6p7*p5p8*p2p3)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p5p8)*
     . (ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-2*
     . ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm)+16*(p5p8*
     . p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+8*(2*ffmcfmb*hbcm*
     . p3p7*p4p6*p5p8+hbcm*p1p2*p5p8*p6p7-hbcm*p2p4*p5p8*p6p7+hbcm*
     . p2p5*p5p8*p6p7))+w7*(8*(p6p7*p2p8*p2p3)*(-ffmcfmb*hbcm+hbcm)+4
     . *(p6p7*p2p8)*(ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*ffmcfmb*
     . fmc*hbcm2-2*ffmcfmb*hbcm3-2*fmb*fmc*hbcm+2*fmc*hbcm2+fmc2*hbcm
     . )+16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+8*(2*
     . ffmcfmb*hbcm*p2p8*p3p7*p4p6+hbcm*p1p2*p2p8*p6p7-hbcm*p2p4*p2p8
     . *p6p7+hbcm*p2p5*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w2*(12*(p5p8*p5p7)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p8*
     . p4p7)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p8*p3p7)*(-ffmcfmb**2*
     . hbcm2+3*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-3*fmc*hbcm))+w7*(
     . 12*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p7*p2p8)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p3p7*p2p8)*(-ffmcfmb**2*hbcm2+3*
     . ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-3*fmc*hbcm))+w11*(12*(p5p7
     . *p4p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*p4p7)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+12*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p7*p2p8
     . )*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*p3p7)*(ffmcfmb**2*hbcm2-3*
     . ffmcfmb*fmb*hbcm-4*ffmcfmb*fmc*hbcm+3*fmc*hbcm)+4*(p3p7*p2p8)*
     . (-ffmcfmb**2*hbcm2+3*ffmcfmb*fmb*hbcm+4*ffmcfmb*fmc*hbcm-3*fmc
     . *hbcm))+(16*(p7p8*p2p3)*(-fb3*ffmcfmb+3*fb3)+2*p7p8*(-12*fb3*
     . ffmcfmb*hbcm2+12*fb3*fmb*fmc+12*fb3*fmc2+12*fb4*ffmcfmb*fmb*
     . hbcm+12*fb4*ffmcfmb*fmc*hbcm-12*fb4*fmc*hbcm-ffmcfmb*hbcm2+fmc
     . *hbcm)+16*(fb3*ffmcfmb*p1p8*p3p7-fb3*ffmcfmb*p2p8*p3p7+fb3*
     . ffmcfmb*p3p7*p4p8-3*fb3*ffmcfmb*p3p7*p5p8-fb3*p1p2*p7p8-fb3*
     . p2p4*p7p8+3*fb3*p2p5*p7p8)))
      b(3)=ans
      b(4)=ccc*(6*p5p7*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+2*
     . p4p7*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+2*p3p7*(-4*fb3
     . *ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*hbcm2-12*fb3*fmb*fmc-12*fb3*
     . fmc2+4*fb4*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*
     . hbcm)+16*(-fb3*ffmcfmb*p1p3*p3p7-2*fb3*ffmcfmb*p2p3*p3p7-fb3*
     . ffmcfmb*p3p4*p3p7+3*fb3*ffmcfmb*p3p5*p3p7+fb3*p1p2*p3p7-fb3*
     . p2p3*p4p7+3*fb3*p2p3*p5p7+fb3*p2p4*p3p7-3*fb3*p2p5*p3p7))
      ans3=(12*(p5p8*p3p7)*(-4*fb3*fmc+hbcm)+4*(p4p8*p3p7)*(4*fb3*fmc
     . -hbcm)+48*(p7p8*p3p5)*(fb3*fmc+fb4*ffmcfmb*hbcm)+16*(p7p8*p3p4
     . )*(-fb3*fmc-fb4*ffmcfmb*hbcm)+4*(p3p7*p1p8)*(4*fb3*fmc-hbcm)+
     . 16*(p7p8*p1p3)*(-fb3*fmc-fb4*ffmcfmb*hbcm)+4*(p3p7*p2p8)*(12*
     . fb3*fmb-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+4*(p7p8*p2p3)*(-12*
     . fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+2*p7p8*(12*fb3*
     . ffmcfmb*fmb*hbcm2-20*fb3*ffmcfmb*fmc*hbcm2+12*fb3*fmc*hbcm2-8*
     . fb4*ffmcfmb**2*hbcm3+12*fb4*ffmcfmb*hbcm3+12*fb4*fmb*fmc*hbcm-
     . 12*fb4*fmc2*hbcm+ffmcfmb*hbcm3-fmc*hbcm2)+16*(fb4*hbcm*p1p2*
     . p7p8+fb4*hbcm*p2p4*p7p8-3*fb4*hbcm*p2p5*p7p8-fb4*hbcm*p2p8*
     . p4p7+3*fb4*hbcm*p2p8*p5p7))
      ans2=w11*(12*(p5p7*p4p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p4p8*p4p7
     . )*(-ffmcfmb*hbcm3+fmc*hbcm2)+12*(p5p7*p2p8)*(-ffmcfmb*hbcm3+
     . fmc*hbcm2)+4*(p4p7*p2p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p4p8*
     . p3p7)*(ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-3*
     . fmb*fmc*hbcm+3*fmc2*hbcm)+4*(p3p7*p2p8)*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*fmc2*hbcm)+
     . 8*(-ffmcfmb*hbcm*p1p3*p2p8*p3p7+ffmcfmb*hbcm*p1p3*p3p7*p4p8-2*
     . ffmcfmb*hbcm*p2p3*p2p8*p3p7+2*ffmcfmb*hbcm*p2p3*p3p7*p4p8-
     . ffmcfmb*hbcm*p2p8*p3p4*p3p7+3*ffmcfmb*hbcm*p2p8*p3p5*p3p7+
     . ffmcfmb*hbcm*p3p4*p3p7*p4p8-3*ffmcfmb*hbcm*p3p5*p3p7*p4p8+hbcm
     . *p1p2*p2p8*p3p7-hbcm*p1p2*p3p7*p4p8-hbcm*p2p3*p2p8*p4p7+3*hbcm
     . *p2p3*p2p8*p5p7+hbcm*p2p3*p4p7*p4p8-3*hbcm*p2p3*p4p8*p5p7+hbcm
     . *p2p4*p2p8*p3p7-hbcm*p2p4*p3p7*p4p8-3*hbcm*p2p5*p2p8*p3p7+3*
     . hbcm*p2p5*p3p7*p4p8))+ans3
      ans1=w2*(12*(p5p8*p5p7)*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p5p8*p4p7
     . )*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p5p8*p3p7)*(-ffmcfmb**2*hbcm3-
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*fmc2*hbcm)+
     . 8*(-ffmcfmb*hbcm*p1p3*p3p7*p5p8-2*ffmcfmb*hbcm*p2p3*p3p7*p5p8-
     . ffmcfmb*hbcm*p3p4*p3p7*p5p8+3*ffmcfmb*hbcm*p3p5*p3p7*p5p8+hbcm
     . *p1p2*p3p7*p5p8-hbcm*p2p3*p4p7*p5p8+3*hbcm*p2p3*p5p7*p5p8+hbcm
     . *p2p4*p3p7*p5p8-3*hbcm*p2p5*p3p7*p5p8))+w7*(12*(p5p7*p2p8)*(-
     . ffmcfmb*hbcm3+fmc*hbcm2)+4*(p4p7*p2p8)*(ffmcfmb*hbcm3-fmc*
     . hbcm2)+4*(p3p7*p2p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmc*hbcm2+3*
     . ffmcfmb*hbcm3+3*fmb*fmc*hbcm-3*fmc2*hbcm)+8*(-ffmcfmb*hbcm*
     . p1p3*p2p8*p3p7-2*ffmcfmb*hbcm*p2p3*p2p8*p3p7-ffmcfmb*hbcm*p2p8
     . *p3p4*p3p7+3*ffmcfmb*hbcm*p2p8*p3p5*p3p7+hbcm*p1p2*p2p8*p3p7-
     . hbcm*p2p3*p2p8*p4p7+3*hbcm*p2p3*p2p8*p5p7+hbcm*p2p4*p2p8*p3p7-
     . 3*hbcm*p2p5*p2p8*p3p7))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w2*(4*(p5p8*p5p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+12*(p5p8*
     . p4p6)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p8*p3p6)*(ffmcfmb*hbcm2-
     . fmc*hbcm))+w7*(4*(p5p6*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+12*(
     . p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p3p6*p2p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm))+w11*(4*(p5p6*p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+
     . 12*(p4p8*p4p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*p3p6)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p6*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm
     . )+12*(p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p3p6*p2p8)*(
     . ffmcfmb*hbcm2-fmc*hbcm))+(16*(p6p8*p2p3)*(-fb3*ffmcfmb+fb3)+4*
     . p6p8*(2*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*hbcm2+4*fb3*fmb*fmc
     . +2*fb3*fmc2+4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*
     . fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+hbcm2)+16*(-2*fb3*ffmcfmb*p2p8*
     . p3p6+fb3*p1p2*p6p8-fb3*p2p4*p6p8+fb3*p2p5*p6p8-4*fb3*p2p8*p4p6
     . +2*fb3*p2p8*p5p6)))
      b(6)=ans
      b(7)=ccc*(16*(p3p6*p2p3)*(3*fb3*ffmcfmb-fb3)+2*p5p6*(4*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+6*p4p6*(-4*fb3*ffmcfmb*
     . hbcm2-4*fb4*fmc*hbcm+hbcm2)+2*p3p6*(-4*fb3*ffmcfmb**2*hbcm2+4*
     . fb3*ffmcfmb*hbcm2-8*fb3*fmb*fmc-4*fb3*fmc2-8*fb4*ffmcfmb*fmb*
     . hbcm-8*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+2*fmb*hbcm+2*fmc*
     . hbcm-hbcm2)+16*(-fb3*p1p2*p3p6+4*fb3*p2p3*p4p6-2*fb3*p2p3*p5p6
     . +fb3*p2p4*p3p6-fb3*p2p5*p3p6))
      ans2=(4*(p5p8*p3p6)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*(
     . p4p8*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8*p3p5)*(
     . 4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8*p3p4)*(-4*fb3*fmc-4
     . *fb4*ffmcfmb*hbcm+hbcm)+4*(p3p6*p1p8)*(-4*fb3*fmc-4*fb4*
     . ffmcfmb*hbcm+hbcm)+4*(p6p8*p1p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm
     . -hbcm)+16*(p3p6*p2p8)*(2*fb3*fmb-fb3*fmc+fb4*ffmcfmb*hbcm-fb4*
     . hbcm)+4*(p6p8*p2p3)*(-8*fb3*fmb+4*fb3*fmc+4*fb4*hbcm+hbcm)+4*
     . p6p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-2*fb4*
     . ffmcfmb**2*hbcm3+4*fb4*fmb*fmc*hbcm-2*fb4*fmc2*hbcm+ffmcfmb*
     . hbcm3-fmb*hbcm2)+16*(-fb4*hbcm*p1p2*p6p8+fb4*hbcm*p2p4*p6p8-
     . fb4*hbcm*p2p5*p6p8+3*fb4*hbcm*p2p8*p4p6-fb4*hbcm*p2p8*p5p6))
      ans1=w2*(8*(p5p8*p4p6)*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p5p8*p3p6)
     . *(ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*fmb*fmc*hbcm-fmc2*
     . hbcm)+8*(ffmcfmb*hbcm*p2p3*p3p6*p5p8-hbcm*p1p2*p3p6*p5p8+3*
     . hbcm*p2p3*p4p6*p5p8-hbcm*p2p3*p5p6*p5p8+hbcm*p2p4*p3p6*p5p8-
     . hbcm*p2p5*p3p6*p5p8))+w7*(8*(p4p6*p2p8)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+4*(p3p6*p2p8)*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+2*
     . fmb*fmc*hbcm-fmc2*hbcm)+8*(ffmcfmb*hbcm*p2p3*p2p8*p3p6-hbcm*
     . p1p2*p2p8*p3p6+3*hbcm*p2p3*p2p8*p4p6-hbcm*p2p3*p2p8*p5p6+hbcm*
     . p2p4*p2p8*p3p6-hbcm*p2p5*p2p8*p3p6))+w11*(8*(p4p8*p4p6)*(
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*(p4p6*p2p8)*(-ffmcfmb*hbcm3+fmc*
     . hbcm2)+4*(p4p8*p3p6)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-2*
     . fmb*fmc*hbcm+fmc2*hbcm)+4*(p3p6*p2p8)*(ffmcfmb**2*hbcm3-2*
     . ffmcfmb*fmb*hbcm2+2*fmb*fmc*hbcm-fmc2*hbcm)+8*(ffmcfmb*hbcm*
     . p2p3*p2p8*p3p6-ffmcfmb*hbcm*p2p3*p3p6*p4p8-hbcm*p1p2*p2p8*p3p6
     . +hbcm*p1p2*p3p6*p4p8+3*hbcm*p2p3*p2p8*p4p6-hbcm*p2p3*p2p8*p5p6
     . -3*hbcm*p2p3*p4p6*p4p8+hbcm*p2p3*p4p8*p5p6+hbcm*p2p4*p2p8*p3p6
     . -hbcm*p2p4*p3p6*p4p8-hbcm*p2p5*p2p8*p3p6+hbcm*p2p5*p3p6*p4p8))
     . +ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(4*(p5p6*p3p7)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(
     . p4p6*p3p7)*(-12*fb3*fmc-4*fb4*ffmcfmb*hbcm+3*hbcm)+4*(p5p7*
     . p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p4p7*p3p6)*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p5)*(-4*fb3*fmc-4*
     . fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p3p4)*(4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm-hbcm)+4*(p6p7*p2p3)*(-4*fb3*fmc-8*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+4*(p6p7*p1p3)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+
     . 32*(p3p7*p3p6)*(fb3*ffmcfmb*fmb-fb3*ffmcfmb*fmc-fb4*ffmcfmb**2
     . *hbcm)+4*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*
     . hbcm2+2*fb4*ffmcfmb**2*hbcm3-4*fb4*fmb*fmc*hbcm+2*fb4*fmc2*
     . hbcm+fmb*hbcm2-fmc*hbcm2)+16*(fb4*hbcm*p1p2*p6p7-fb4*hbcm*p2p4
     . *p6p7+fb4*hbcm*p2p5*p6p7))
      ans5=8*(-fmc*hbcm*p2p8*p3p7*p5p6+fmc*hbcm*p3p7*p4p8*p5p6-hbcm2*
     . p1p2*p2p8*p6p7+hbcm2*p1p2*p4p8*p6p7+hbcm2*p2p4*p2p8*p6p7-hbcm2
     . *p2p4*p4p8*p6p7-hbcm2*p2p5*p2p8*p6p7+hbcm2*p2p5*p4p8*p6p7)
      ans4=8*(p6p7*p4p8*p2p3)*(2*fmb*hbcm+fmc*hbcm-hbcm2)+8*(p6p7*
     . p2p8*p2p3)*(-2*fmb*hbcm-fmc*hbcm+hbcm2)+8*(p4p8*p4p6*p3p7)*(2*
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p5p7*p4p8*p3p6)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+8*(p4p8*p4p7*p3p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*
     . p4p8*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p4p8*p3p4)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p6*p3p7*p2p8)*(-2*ffmcfmb*hbcm2+
     . fmc*hbcm)+8*(p5p7*p3p6*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7
     . *p3p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p5*p2p8)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p3p4*p2p8)*(-ffmcfmb*hbcm2+fmc
     . *hbcm)+8*(p6p7*p4p8*p1p3)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*
     . p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p6p7*p4p8)*(ffmcfmb**2*
     . hbcm4-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3+2*fmb*fmc*hbcm2+
     . fmc2*hbcm2)+4*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*
     . hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-fmc2*hbcm2)+8*(p4p8*
     . p3p7*p3p6)*(-2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+ffmcfmb*
     . hbcm2)+8*(p3p7*p3p6*p2p8)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*hbcm2)+ans5
      ans3=w11*ans4
      ans7=32*(-fb3*ffmcfmb*p1p3*p3p7*p6p8+fb3*ffmcfmb*p1p8*p3p6*p3p7
     . +fb3*ffmcfmb*p3p4*p3p7*p6p8-fb3*ffmcfmb*p3p5*p3p7*p6p8-fb3*
     . ffmcfmb*p3p6*p3p7*p4p8+fb3*ffmcfmb*p3p6*p3p7*p5p8+fb3*p1p8*
     . p2p3*p6p7+fb3*p2p3*p2p8*p6p7-fb3*p2p3*p4p6*p7p8+fb3*p2p3*p4p7*
     . p6p8-fb3*p2p3*p4p8*p6p7+fb3*p2p3*p5p6*p7p8-fb3*p2p3*p5p7*p6p8+
     . fb3*p2p3*p5p8*p6p7+5*fb3*p2p8*p3p7*p4p6-3*fb3*p2p8*p3p7*p5p6)
      ans6=32*(p3p7*p3p6*p2p8)*(5*fb3*ffmcfmb-fb3)+32*(p6p8*p3p7*p2p3
     . )*(fb3*ffmcfmb-fb3)+32*(p7p8*p3p6*p2p3)*(-2*fb3*ffmcfmb+fb3)+4
     . *(p6p7*p5p8)*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+4*(
     . p6p8*p5p7)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+16*(p7p8
     . *p5p6)*(-fb3*ffmcfmb*hbcm2-fb4*fmc*hbcm)+4*(p6p7*p4p8)*(4*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+4*(p6p8*p4p7)*(-4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+8*(p7p8*p4p6)*(2*fb3*
     . ffmcfmb*hbcm2+2*fb4*fmc*hbcm-hbcm2)+4*(p6p8*p3p7)*(4*fb3*
     . ffmcfmb*hbcm2-8*fb4*ffmcfmb*fmb*hbcm-8*fb4*ffmcfmb*fmc*hbcm+4*
     . fb4*fmc*hbcm+2*ffmcfmb*hbcm2+2*fmc*hbcm-hbcm2)+8*(p6p7*p2p8)*(
     . -2*fb3*ffmcfmb*hbcm2-2*fb4*fmc*hbcm+hbcm2)+4*(p6p7*p1p8)*(-4*
     . fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+8*(p7p8*p3p6)*(4*fb3*
     . ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*hbcm2+4*fb4*ffmcfmb*fmc*hbcm-2*
     . fb4*fmc*hbcm+ffmcfmb*hbcm2-fmb*hbcm-fmc*hbcm)+ans7
      ans2=w7*(8*(p6p7*p2p8*p2p3)*(-2*fmb*hbcm-fmc*hbcm+hbcm2)+8*(
     . p4p6*p3p7*p2p8)*(-2*ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p7*p3p6*p2p8)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7*p3p6*p2p8)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+8*(p6p7*p3p5*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*
     . p3p4*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p2p8*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-fmc2*
     . hbcm2)+8*(p3p7*p3p6*p2p8)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*hbcm2)+8*(-fmc*hbcm*p2p8*p3p7*p5p6-hbcm2*p1p2*
     . p2p8*p6p7+hbcm2*p2p4*p2p8*p6p7-hbcm2*p2p5*p2p8*p6p7))+ans3+
     . ans6
      ans1=w2*(8*(p6p7*p5p8*p2p3)*(-2*fmb*hbcm-fmc*hbcm+hbcm2)+8*(
     . p5p8*p4p6*p3p7)*(-2*ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p8*p5p7*p3p6)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p5p8*p4p7*p3p6)*(ffmcfmb*hbcm2-
     . fmc*hbcm)+8*(p6p7*p5p8*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*
     . p5p8*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p6p7*p5p8*p1p3)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*(p6p7*p5p8)*(-ffmcfmb**2*hbcm4+2*
     . ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*fmc*hbcm2-fmc2*
     . hbcm2)+8*(p5p8*p3p7*p3p6)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*
     . hbcm-ffmcfmb*hbcm2)+8*(-fmc*hbcm*p3p7*p5p6*p5p8-hbcm2*p1p2*
     . p5p8*p6p7+hbcm2*p2p4*p5p8*p6p7-hbcm2*p2p5*p5p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w2*(4*(p5p8*p2p3)*(-ffmcfmb*hbcm+3*hbcm)+6*p5p8*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2+fmc2*hbcm)+4*(-hbcm*p1p2*p5p8-hbcm*p2p4*p5p8+3*hbcm
     . *p2p5*p5p8))+w7*(4*(p2p8*p2p3)*(-ffmcfmb*hbcm+3*hbcm)+6*p2p8*(
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm
     . +fmc*hbcm2+fmc2*hbcm)+4*(-hbcm*p1p2*p2p8-hbcm*p2p4*p2p8+3*hbcm
     . *p2p5*p2p8))+w11*(4*(p4p8*p2p3)*(ffmcfmb*hbcm-3*hbcm)+4*(p2p8*
     . p2p3)*(-ffmcfmb*hbcm+3*hbcm)+6*p4p8*(-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-fmc2*
     . hbcm)+6*p2p8*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmc*hbcm2+fmc2*hbcm)+4*(-hbcm*p1p2*p2p8+
     . hbcm*p1p2*p4p8-hbcm*p2p4*p2p8+hbcm*p2p4*p4p8+3*hbcm*p2p5*p2p8-
     . 3*hbcm*p2p5*p4p8))+(6*p5p8*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)
     . +2*p4p8*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*p1p8*(-4*fb3*
     . fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*p2p8*(-12*fb3*fmb+4*fb3*fmc-8*
     . fb4*ffmcfmb*hbcm+12*fb4*hbcm+hbcm)))
      b(11)=ans
      b(12)=ccc*(8*p2p3*(fb3*ffmcfmb-3*fb3)+(12*fb3*ffmcfmb*hbcm2-12*
     . fb3*fmb*fmc-12*fb3*fmc2+8*fb3*p1p2+8*fb3*p2p4-24*fb3*p2p5-12*
     . fb4*ffmcfmb*fmb*hbcm-12*fb4*ffmcfmb*fmc*hbcm+12*fb4*fmc*hbcm+3
     . *fmb*hbcm+3*fmc*hbcm-3*hbcm2))
      ans3=(6*p5p8*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+2*p4p8*
     . (-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+2*p2p8*(-16*fb3*
     . ffmcfmb*hbcm2+12*fb3*hbcm2+12*fb4*fmb*hbcm+4*fb4*fmc*hbcm+
     . hbcm2)+2*p1p8*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+16*(
     . -fb3*p1p3*p2p8+fb3*p1p8*p2p3+fb3*p2p3*p4p8-3*fb3*p2p3*p5p8-fb3
     . *p2p8*p3p4+3*fb3*p2p8*p3p5))
      ans2=w11*(12*(p4p8*p3p5)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p8*p3p4)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+12*(p3p5*p2p8)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+4*(p3p4*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p8*p2p3)*(-
     . ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm)+4*(p2p8*p2p3)*(ffmcfmb*
     . hbcm2+3*fmb*hbcm+fmc*hbcm)+4*(p4p8*p1p3)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+4*(p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+2*p4p8*(-2*
     . ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*hbcm3+5*ffmcfmb*fmc*hbcm3+3*
     . ffmcfmb*hbcm4-3*fmb*fmc*hbcm2-3*fmc*hbcm3-3*fmc2*hbcm2)+2*p2p8
     . *(2*ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*hbcm3-5*ffmcfmb*fmc*hbcm3-3
     . *ffmcfmb*hbcm4+3*fmb*fmc*hbcm2+3*fmc*hbcm3+3*fmc2*hbcm2)+4*(-
     . hbcm2*p1p2*p2p8+hbcm2*p1p2*p4p8-hbcm2*p2p4*p2p8+hbcm2*p2p4*
     . p4p8+3*hbcm2*p2p5*p2p8-3*hbcm2*p2p5*p4p8))+ans3
      ans1=w2*(12*(p5p8*p3p5)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p8*p3p4)
     . *(ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p8*p2p3)*(ffmcfmb*hbcm2+3*fmb*
     . hbcm+fmc*hbcm)+4*(p5p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+2*p5p8*(
     . 2*ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*hbcm3-5*ffmcfmb*fmc*hbcm3-3*
     . ffmcfmb*hbcm4+3*fmb*fmc*hbcm2+3*fmc*hbcm3+3*fmc2*hbcm2)+4*(-
     . hbcm2*p1p2*p5p8-hbcm2*p2p4*p5p8+3*hbcm2*p2p5*p5p8))+w7*(12*(
     . p3p5*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p3p4*p2p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(p2p8*p2p3)*(ffmcfmb*hbcm2+3*fmb*hbcm+fmc*
     . hbcm)+4*(p2p8*p1p3)*(ffmcfmb*hbcm2-fmc*hbcm)+2*p2p8*(2*ffmcfmb
     . **2*hbcm4-3*ffmcfmb*fmb*hbcm3-5*ffmcfmb*fmc*hbcm3-3*ffmcfmb*
     . hbcm4+3*fmb*fmc*hbcm2+3*fmc*hbcm3+3*fmc2*hbcm2)+4*(-hbcm2*p1p2
     . *p2p8-hbcm2*p2p4*p2p8+3*hbcm2*p2p5*p2p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(6*p3p5*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*p3p4*(
     . 4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+2*p1p3*(4*fb3*fmc+4*fb4*
     . ffmcfmb*hbcm-hbcm)+4*p2p3*(6*fb3*fmb-2*fb3*fmc+2*fb4*ffmcfmb*
     . hbcm-hbcm)+(-12*fb3*ffmcfmb*fmb*hbcm2+20*fb3*ffmcfmb*fmc*hbcm2
     . -12*fb3*fmc*hbcm2+8*fb4*ffmcfmb**2*hbcm3-12*fb4*ffmcfmb*hbcm3-
     . 12*fb4*fmb*fmc*hbcm+12*fb4*fmc2*hbcm-8*fb4*hbcm*p1p2-8*fb4*
     . hbcm*p2p4+24*fb4*hbcm*p2p5-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*
     . hbcm2+3*hbcm3))
      b(15)=ccc*(w2*(4*(p5p8*p3p6)*(2*ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)
     . +4*(3*hbcm2*p4p6*p5p8-hbcm2*p5p6*p5p8))+w7*(4*(p3p6*p2p8)*(2*
     . ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)+4*(3*hbcm2*p2p8*p4p6-hbcm2*
     . p2p8*p5p6))+w11*(4*(p4p8*p3p6)*(-2*ffmcfmb*hbcm2-2*fmb*hbcm+
     . hbcm2)+4*(p3p6*p2p8)*(2*ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)+4*(3*
     . hbcm2*p2p8*p4p6-hbcm2*p2p8*p5p6-3*hbcm2*p4p6*p4p8+hbcm2*p4p8*
     . p5p6))+(4*p6p8*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmb*hbcm+hbcm2)+16*
     . (fb3*p1p3*p6p8-fb3*p1p8*p3p6+fb3*p2p3*p6p8-fb3*p2p8*p3p6-fb3*
     . p3p4*p6p8+fb3*p3p5*p6p8+fb3*p3p6*p4p8-fb3*p3p6*p5p8)))
      b(16)=ccc*(8*w2*(-ffmcfmb*hbcm*p3p6*p5p8-2*hbcm*p4p6*p5p8+hbcm*
     . p5p6*p5p8)+8*w7*(-ffmcfmb*hbcm*p2p8*p3p6-2*hbcm*p2p8*p4p6+hbcm
     . *p2p8*p5p6)+8*w11*(-ffmcfmb*hbcm*p2p8*p3p6+ffmcfmb*hbcm*p3p6*
     . p4p8-2*hbcm*p2p8*p4p6+hbcm*p2p8*p5p6+2*hbcm*p4p6*p4p8-hbcm*
     . p4p8*p5p6)+8*p6p8*(2*fb3*fmb+2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-
     . hbcm))
      b(17)=ccc*(w2*(2*p5p8*(5*ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(
     . hbcm*p1p3*p5p8-hbcm*p2p3*p5p8+hbcm*p3p4*p5p8-3*hbcm*p3p5*p5p8)
     . )+w7*(2*p2p8*(5*ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(hbcm*
     . p1p3*p2p8-hbcm*p2p3*p2p8+hbcm*p2p8*p3p4-3*hbcm*p2p8*p3p5))+w11
     . *(2*p4p8*(-5*ffmcfmb*hbcm3+3*fmb*hbcm2+3*hbcm3)+2*p2p8*(5*
     . ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(hbcm*p1p3*p2p8-hbcm*p1p3
     . *p4p8-hbcm*p2p3*p2p8+hbcm*p2p3*p4p8+hbcm*p2p8*p3p4-3*hbcm*p2p8
     . *p3p5-hbcm*p3p4*p4p8+3*hbcm*p3p5*p4p8))+8*(fb4*hbcm*p1p8-fb4*
     . hbcm*p2p8+fb4*hbcm*p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w2*(4*(p5p8*p3p7)*(ffmcfmb*hbcm2-3*fmb*hbcm)+4*(
     . hbcm2*p4p7*p5p8-3*hbcm2*p5p7*p5p8))+w7*(4*(p3p7*p2p8)*(ffmcfmb
     . *hbcm2-3*fmb*hbcm)+4*(hbcm2*p2p8*p4p7-3*hbcm2*p2p8*p5p7))+w11*
     . (4*(p4p8*p3p7)*(-ffmcfmb*hbcm2+3*fmb*hbcm)+4*(p3p7*p2p8)*(
     . ffmcfmb*hbcm2-3*fmb*hbcm)+4*(hbcm2*p2p8*p4p7-3*hbcm2*p2p8*p5p7
     . -hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+(2*p7p8*(20*fb3*ffmcfmb*
     . hbcm2-12*fb3*hbcm2-12*fb4*fmb*hbcm+hbcm2)+16*(fb3*p1p3*p7p8-
     . fb3*p1p8*p3p7-fb3*p2p3*p7p8+fb3*p2p8*p3p7+fb3*p3p4*p7p8-3*fb3*
     . p3p5*p7p8-fb3*p3p7*p4p8+3*fb3*p3p7*p5p8)))
      b(19)=ccc*(6*w2*p5p8*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+6*w7*p2p8*
     . (-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+w11*(6*p4p8*(ffmcfmb*hbcm2+fmb
     . *hbcm-hbcm2)+6*p2p8*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2))+8*(-fb3*
     . p1p8+fb3*p2p8-fb3*p4p8+3*fb3*p5p8))
      b(20)=ccc*(w2*(4*(p5p8*p3p7)*(4*ffmcfmb*hbcm-3*hbcm)+4*(hbcm*
     . p4p7*p5p8-3*hbcm*p5p7*p5p8))+w7*(4*(p3p7*p2p8)*(4*ffmcfmb*hbcm
     . -3*hbcm)+4*(hbcm*p2p8*p4p7-3*hbcm*p2p8*p5p7))+w11*(4*(p4p8*
     . p3p7)*(-4*ffmcfmb*hbcm+3*hbcm)+4*(p3p7*p2p8)*(4*ffmcfmb*hbcm-3
     . *hbcm)+4*(hbcm*p2p8*p4p7-3*hbcm*p2p8*p5p7-hbcm*p4p7*p4p8+3*
     . hbcm*p4p8*p5p7))+2*p7p8*(12*fb3*fmb+12*fb4*ffmcfmb*hbcm-12*fb4
     . *hbcm+hbcm))
      b(21)=3*ccc*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)
      b(22)=ccc*(20*fb3*ffmcfmb*hbcm2-12*fb3*hbcm2+8*fb3*p1p3-8*fb3*
     . p2p3+8*fb3*p3p4-24*fb3*p3p5-12*fb4*fmb*hbcm-3*hbcm2)
      b(23)=ccc*(2*p3p7*(-12*fb3*fmb+4*fb4*ffmcfmb*hbcm+3*hbcm)+8*(
     . fb4*hbcm*p4p7-3*fb4*hbcm*p5p7))
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb-8*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+16*(-2*fb4*hbcm*p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(6*p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*p4p7*(
     . 4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+2*p3p7*(-12*fb3*ffmcfmb*fmb
     . +16*fb3*ffmcfmb*fmc-12*fb3*fmc+4*fb4*ffmcfmb**2*hbcm-ffmcfmb*
     . hbcm+3*hbcm))
      b(26)=ccc*(4*p5p6*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*p4p6*(-
     . 4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p3p6*(-4*fb3*ffmcfmb*fmc-
     . 4*fb4*ffmcfmb**2*hbcm+ffmcfmb*hbcm))
      b(27)=ccc*(16*(p3p7*p3p6)*(-4*fb3*ffmcfmb+fb3)+4*p6p7*(4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmb*hbcm-hbcm2)+16*(-fb3*p1p3*p6p7-fb3*
     . p2p3*p6p7+fb3*p3p4*p6p7-fb3*p3p5*p6p7-fb3*p3p6*p4p7+fb3*p3p6*
     . p5p7-4*fb3*p3p7*p4p6+2*fb3*p3p7*p5p6))
      b(28)=ccc*(8*w2*(p6p7*p5p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*w7
     . *(p6p7*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+w11*(8*(p6p7*p4p8)
     . *(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p6p7*p2p8)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2))+(16*(p6p8*p3p7)*(-2*fb3*ffmcfmb+fb3)+16*(-2*
     . fb3*ffmcfmb*p3p6*p7p8-fb3*p1p8*p6p7-fb3*p2p8*p6p7-4*fb3*p4p6*
     . p7p8-fb3*p4p7*p6p8+fb3*p4p8*p6p7+2*fb3*p5p6*p7p8+fb3*p5p7*p6p8
     . -fb3*p5p8*p6p7)))
      b(29)=ccc*(8*p3p6*(-4*fb3*ffmcfmb+fb3)+8*(-5*fb3*p4p6+3*fb3*
     . p5p6))
      b(30)=ccc*(8*p3p7*(4*fb3*ffmcfmb-3*fb3)+8*(fb3*p4p7-3*fb3*p5p7)
     . )
      ans2=(8*(p6p8*p3p7)*(4*fb3*fmb-2*fb4*hbcm-hbcm)+16*(p7p8*p3p6)*
     . (-2*fb3*fmb-2*fb4*ffmcfmb*hbcm+fb4*hbcm)+16*(-fb4*hbcm*p1p8*
     . p6p7-fb4*hbcm*p2p8*p6p7-3*fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*
     . p6p8+fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p6*p7p8+fb4*hbcm*p5p7*p6p8-
     . fb4*hbcm*p5p8*p6p7))
      ans1=w2*(8*(p6p7*p5p8)*(ffmcfmb*hbcm3-fmb*hbcm2)+8*(-2*ffmcfmb*
     . hbcm*p3p6*p3p7*p5p8-hbcm*p1p3*p5p8*p6p7-hbcm*p2p3*p5p8*p6p7+
     . hbcm*p3p4*p5p8*p6p7-hbcm*p3p5*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8+
     . hbcm*p3p6*p5p7*p5p8-3*hbcm*p3p7*p4p6*p5p8+hbcm*p3p7*p5p6*p5p8)
     . )+w7*(8*(p6p7*p2p8)*(ffmcfmb*hbcm3-fmb*hbcm2)+8*(-2*ffmcfmb*
     . hbcm*p2p8*p3p6*p3p7-hbcm*p1p3*p2p8*p6p7-hbcm*p2p3*p2p8*p6p7+
     . hbcm*p2p8*p3p4*p6p7-hbcm*p2p8*p3p5*p6p7-hbcm*p2p8*p3p6*p4p7+
     . hbcm*p2p8*p3p6*p5p7-3*hbcm*p2p8*p3p7*p4p6+hbcm*p2p8*p3p7*p5p6)
     . )+w11*(8*(p6p7*p4p8)*(-ffmcfmb*hbcm3+fmb*hbcm2)+8*(p6p7*p2p8)*
     . (ffmcfmb*hbcm3-fmb*hbcm2)+8*(-2*ffmcfmb*hbcm*p2p8*p3p6*p3p7+2*
     . ffmcfmb*hbcm*p3p6*p3p7*p4p8-hbcm*p1p3*p2p8*p6p7+hbcm*p1p3*p4p8
     . *p6p7-hbcm*p2p3*p2p8*p6p7+hbcm*p2p3*p4p8*p6p7+hbcm*p2p8*p3p4*
     . p6p7-hbcm*p2p8*p3p5*p6p7-hbcm*p2p8*p3p6*p4p7+hbcm*p2p8*p3p6*
     . p5p7-3*hbcm*p2p8*p3p7*p4p6+hbcm*p2p8*p3p7*p5p6-hbcm*p3p4*p4p8*
     . p6p7+hbcm*p3p5*p4p8*p6p7+hbcm*p3p6*p4p7*p4p8-hbcm*p3p6*p4p8*
     . p5p7+3*hbcm*p3p7*p4p6*p4p8-hbcm*p3p7*p4p8*p5p6))+ans2
      ans=ccc*ans1
      b(31)=ans
      b(32)=4*ccc*p6p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp29_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+fmc2-2*
     . p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p2p3-fmb2+hbcm2-2*p2p3))
      b(1)=ccc*(64*(p4p6*p3p7)*(fb3*ffmcfmb-fb3)+32*(p3p7*p3p6)*(fb3*
     . ffmcfmb-fb3)+2*p6p7*(4*fb3*ffmcfmb**2*hbcm2-12*fb3*ffmcfmb*
     . hbcm2-4*fb3*fmb*fmc-8*fb3*fmb2+8*fb3*hbcm2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+fmb*hbcm+fmc*hbcm-
     . hbcm2)+32*(fb3*p1p2*p6p7-fb3*p2p3*p6p7-fb3*p2p4*p6p7))
      ans3=(4*(p6p7*p2p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+8*(
     . p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(
     . p6p8*p4p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+16*
     . (p7p8*p4p6)*(-2*fb3*fmb-2*fb4*ffmcfmb*hbcm+2*fb4*hbcm+hbcm)+8*
     . (p6p7*p1p8)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*
     . (p6p8*p3p7)*(-8*fb3*ffmcfmb*fmb+4*fb3*ffmcfmb*fmc-4*fb3*fmc-4*
     . fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm+3*hbcm)+8*
     . (p7p8*p3p6)*(-4*fb3*ffmcfmb*fmb-4*fb4*ffmcfmb**2*hbcm+4*fb4*
     . ffmcfmb*hbcm+hbcm))
      ans2=w5*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+16*(p4p6*p3p7
     . *p1p8)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+2*fmb2*hbcm+fmc*hbcm2-2*hbcm3)+4*(p6p7*p1p8)*(ffmcfmb**2*
     . hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*
     . fmc*hbcm-2*fmb2*hbcm-fmc*hbcm2+2*hbcm3)+16*(p4p8*p3p7*p3p6)*(
     . ffmcfmb**2*hbcm-2*ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm+2*ffmcfmb*hbcm-hbcm)+16*(hbcm*p1p2*p1p8*p6p7-
     . hbcm*p1p2*p4p8*p6p7-hbcm*p1p8*p2p3*p6p7-hbcm*p1p8*p2p4*p6p7+
     . hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7))+ans3
      ans1=w1*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p4p8)
     . *(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+2*fmb2*hbcm+fmc*hbcm2-2*hbcm3)+16*(
     . p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-2*ffmcfmb*hbcm+hbcm)+16*(-
     . hbcm*p1p2*p4p8*p6p7+hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7))+
     . w8*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p2p8)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb
     . *hbcm3-fmb*fmc*hbcm+2*fmb2*hbcm+fmc*hbcm2-2*hbcm3)+16*(p3p7*
     . p3p6*p2p8)*(ffmcfmb**2*hbcm-2*ffmcfmb*hbcm+hbcm)+16*(-hbcm*
     . p1p2*p2p8*p6p7+hbcm*p2p3*p2p8*p6p7+hbcm*p2p4*p2p8*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(2*p7p8*(ffmcfmb*hbcm2+4*fmb*hbcm+3*fmc*hbcm-4*hbcm2)+48*(
     . p5p8*p3p7)*(-fb3*ffmcfmb+fb3)+16*(p4p8*p3p7)*(fb3*ffmcfmb-fb3)
     . +16*(p3p7*p1p8)*(fb3*ffmcfmb-fb3)+16*(-fb3*ffmcfmb*p2p8*p3p7-
     . fb3*p2p8*p4p7+3*fb3*p2p8*p5p7))
      ans1=w1*(12*(p5p7*p4p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p8*
     . p4p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p8*p3p7)*(-ffmcfmb
     . **2*hbcm2-4*ffmcfmb*fmb*hbcm-3*ffmcfmb*fmc*hbcm+4*ffmcfmb*
     . hbcm2+3*fmb*hbcm+3*fmc*hbcm-3*hbcm2))+w8*(12*(p5p7*p2p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p7*p2p8)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+4*(p3p7*p2p8)*(-ffmcfmb**2*hbcm2-4*ffmcfmb*fmb
     . *hbcm-3*ffmcfmb*fmc*hbcm+4*ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*hbcm
     . -3*hbcm2))+w5*(12*(p5p7*p4p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4
     . *(p4p8*p4p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+12*(p5p7*p1p8)*(-
     . ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p7*p1p8)*(ffmcfmb*hbcm2+fmb
     . *hbcm-hbcm2)+4*(p4p8*p3p7)*(-ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*
     . hbcm-3*ffmcfmb*fmc*hbcm+4*ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*hbcm-
     . 3*hbcm2)+4*(p3p7*p1p8)*(ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+3*
     . ffmcfmb*fmc*hbcm-4*ffmcfmb*hbcm2-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2
     . ))+ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=ccc*(48*(p3p7*p3p5)*(fb3*ffmcfmb-fb3)+16*(p3p7*p3p4)*(-fb3
     . *ffmcfmb+fb3)+48*(p3p7*p2p3)*(-fb3*ffmcfmb+fb3)+16*(p3p7*p1p3)
     . *(-fb3*ffmcfmb+fb3)+6*p5p7*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4
     . *fb4*fmb*hbcm-hbcm2)+2*p4p7*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4
     . *fb4*fmb*hbcm+hbcm2)+2*p3p7*(-4*fb3*ffmcfmb**2*hbcm2+16*fb3*
     . ffmcfmb*hbcm2-12*fb3*hbcm2+16*fb4*ffmcfmb*fmb*hbcm+12*fb4*
     . ffmcfmb*fmc*hbcm-12*fb4*fmb*hbcm-12*fb4*fmc*hbcm+ffmcfmb*hbcm2
     . ))
      ans3=(2*p7p8*(-ffmcfmb*hbcm3+4*fmb*hbcm2-3*fmc*hbcm2+2*hbcm3)+
     . 48*(p5p8*p3p7)*(fb4*ffmcfmb*hbcm-fb4*hbcm)+16*(p4p8*p3p7)*(-
     . fb4*ffmcfmb*hbcm+fb4*hbcm)+48*(p3p7*p2p8)*(-fb4*ffmcfmb*hbcm+
     . fb4*hbcm)+16*(p3p7*p1p8)*(-fb4*ffmcfmb*hbcm+fb4*hbcm)+4*(-hbcm
     . *p1p3*p7p8-3*hbcm*p2p3*p7p8-hbcm*p3p4*p7p8+3*hbcm*p3p5*p7p8))
      ans2=w5*(12*(p5p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*(p4p8
     . *p4p7)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+12*(p5p7*p1p8)*(-
     . ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*(p4p7*p1p8)*(ffmcfmb*hbcm3+
     . fmb*hbcm2-hbcm3)+24*(p4p8*p3p7*p3p5)*(-ffmcfmb*hbcm+hbcm)+8*(
     . p4p8*p3p7*p3p4)*(ffmcfmb*hbcm-hbcm)+24*(p4p8*p3p7*p2p3)*(
     . ffmcfmb*hbcm-hbcm)+24*(p3p7*p3p5*p1p8)*(ffmcfmb*hbcm-hbcm)+8*(
     . p3p7*p3p4*p1p8)*(-ffmcfmb*hbcm+hbcm)+24*(p3p7*p2p3*p1p8)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7*p1p3)*(ffmcfmb*hbcm-hbcm)+8*(
     . p3p7*p1p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+4*(p4p8*p3p7)*(ffmcfmb**2
     . *hbcm3-4*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-4*ffmcfmb*hbcm3
     . +3*fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+4*(p3p7*p1p8)*(-ffmcfmb**2*
     . hbcm3+4*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*hbcm2+4*ffmcfmb*hbcm3-
     . 3*fmb*hbcm2+3*fmc*hbcm2-3*hbcm3))+ans3
      ans1=w1*(12*(p5p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+4*(p4p8
     . *p4p7)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+24*(p4p8*p3p7*p3p5)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7*p3p4)*(ffmcfmb*hbcm-hbcm)+24*(
     . p4p8*p3p7*p2p3)*(ffmcfmb*hbcm-hbcm)+8*(p4p8*p3p7*p1p3)*(
     . ffmcfmb*hbcm-hbcm)+4*(p4p8*p3p7)*(ffmcfmb**2*hbcm3-4*ffmcfmb*
     . fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-4*ffmcfmb*hbcm3+3*fmb*hbcm2-3*
     . fmc*hbcm2+3*hbcm3))+w8*(12*(p5p7*p2p8)*(ffmcfmb*hbcm3+fmb*
     . hbcm2-hbcm3)+4*(p4p7*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+24
     . *(p3p7*p3p5*p2p8)*(-ffmcfmb*hbcm+hbcm)+8*(p3p7*p3p4*p2p8)*(
     . ffmcfmb*hbcm-hbcm)+24*(p3p7*p2p8*p2p3)*(ffmcfmb*hbcm-hbcm)+8*(
     . p3p7*p2p8*p1p3)*(ffmcfmb*hbcm-hbcm)+4*(p3p7*p2p8)*(ffmcfmb**2*
     . hbcm3-4*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-4*ffmcfmb*hbcm3+
     . 3*fmb*hbcm2-3*fmc*hbcm2+3*hbcm3))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w1*(16*(p4p8*p4p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(
     . p4p8*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+w8*(16*(p4p6*p2p8)*
     . (ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p3p6*p2p8)*(ffmcfmb*hbcm2+
     . fmb*hbcm-hbcm2))+w5*(16*(p4p8*p4p6)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+8*(p4p8*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p6*
     . p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p3p6*p1p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2))+(2*p6p8*(4*fb3*ffmcfmb**2*hbcm2-12*fb3*
     . ffmcfmb*hbcm2-4*fb3*fmb*fmc-8*fb3*fmb2+8*fb3*hbcm2-4*fb4*
     . ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm-hbcm2)+32*(-fb3*ffmcfmb*p2p8*p3p6+fb3*p1p2*p6p8-
     . fb3*p2p3*p6p8-fb3*p2p4*p6p8-fb3*p2p8*p4p6)))
      b(7)=ccc*(32*(p3p6*p2p3)*(fb3*ffmcfmb+fb3)+8*p4p6*(-4*fb3*
     . ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+2*p3p6*(-4*fb3
     . *ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2+4*fb3*fmb*fmc+8*fb3*fmb2
     . +4*fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-8*fb4*fmb*hbcm-
     . 4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm-hbcm2)+32*(-fb3*p1p2*p3p6+fb3
     . *p2p3*p4p6+fb3*p2p4*p3p6))
      ans3=(8*(p6p8*p2p3)*(4*fb4*hbcm+hbcm)+8*(p4p8*p3p6)*(-4*fb3*fmb
     . -4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p6p8*p3p4)*(4*fb3*fmb+
     . 4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p3p6*p1p8)*(4*fb3*fmb+4
     . *fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*(p6p8*p1p3)*(-4*fb3*fmb-4
     . *fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*p6p8*(12*fb3*ffmcfmb*fmb*
     . hbcm2-4*fb3*ffmcfmb*fmc*hbcm2+4*fb3*fmc*hbcm2+4*fb4*ffmcfmb**2
     . *hbcm3+4*fb4*ffmcfmb*hbcm3-4*fb4*fmb*fmc*hbcm+8*fb4*fmb2*hbcm-
     . 8*fb4*hbcm3+3*fmb*hbcm2-fmc*hbcm2-3*hbcm3)+32*(fb4*ffmcfmb*
     . hbcm*p2p8*p3p6-fb4*hbcm*p1p2*p6p8+fb4*hbcm*p2p4*p6p8+fb4*hbcm*
     . p2p8*p4p6))
      ans2=w5*(8*(p4p8*p4p6)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(p4p6*
     . p1p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(p4p8*p3p6*p2p3)*(-
     . ffmcfmb*hbcm-hbcm)+16*(p3p6*p2p3*p1p8)*(ffmcfmb*hbcm+hbcm)+4*(
     . p4p8*p3p6)*(-ffmcfmb**2*hbcm3-3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+2*fmb*hbcm2-2*fmb2*hbcm-fmc*
     . hbcm2)+4*(p3p6*p1p8)*(ffmcfmb**2*hbcm3+3*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*fmc*hbcm-2*fmb*hbcm2+2*
     . fmb2*hbcm+fmc*hbcm2)+16*(-hbcm*p1p2*p1p8*p3p6+hbcm*p1p2*p3p6*
     . p4p8+hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p4*p3p6-hbcm*p2p3*p4p6*
     . p4p8-hbcm*p2p4*p3p6*p4p8))+ans3
      ans1=w1*(8*(p4p8*p4p6)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p4p8
     . *p3p6*p2p3)*(-ffmcfmb*hbcm-hbcm)+4*(p4p8*p3p6)*(-ffmcfmb**2*
     . hbcm3-3*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*
     . fmc*hbcm+2*fmb*hbcm2-2*fmb2*hbcm-fmc*hbcm2)+16*(hbcm*p1p2*p3p6
     . *p4p8-hbcm*p2p3*p4p6*p4p8-hbcm*p2p4*p3p6*p4p8))+w8*(8*(p4p6*
     . p2p8)*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p3p6*p2p8*p2p3)*(-
     . ffmcfmb*hbcm-hbcm)+4*(p3p6*p2p8)*(-ffmcfmb**2*hbcm3-3*ffmcfmb*
     . fmb*hbcm2+ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm+2*fmb*
     . hbcm2-2*fmb2*hbcm-fmc*hbcm2)+16*(hbcm*p1p2*p2p8*p3p6-hbcm*p2p3
     . *p2p8*p4p6-hbcm*p2p4*p2p8*p3p6))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(4*(p6p7*p2p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-8*fb4*
     . hbcm-hbcm)+8*(p4p6*p3p7)*(4*fb3*fmb+8*fb4*ffmcfmb*hbcm-8*fb4*
     . hbcm-hbcm)+8*(p4p7*p3p6)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+8*(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+8*(p6p7*p1p3)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)+4*(p3p7*p3p6)*(16*fb3*ffmcfmb*fmb-4*fb3*ffmcfmb*fmc
     . +4*fb3*fmc+4*fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm-8*fb4*hbcm
     . -3*ffmcfmb*hbcm-hbcm)+2*p6p7*(-12*fb3*ffmcfmb*fmb*hbcm2+4*fb3*
     . ffmcfmb*fmc*hbcm2-4*fb3*fmc*hbcm2-4*fb4*ffmcfmb**2*hbcm3-4*fb4
     . *ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-8*fb4*fmb2*hbcm+8*fb4*hbcm3+
     . 2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+hbcm3)+32*(fb4*hbcm*p1p2*
     . p6p7-fb4*hbcm*p2p4*p6p7))
      ans5=16*(-hbcm2*p1p2*p1p8*p6p7+hbcm2*p1p2*p4p8*p6p7+hbcm2*p1p8*
     . p2p3*p6p7+hbcm2*p1p8*p2p4*p6p7-hbcm2*p2p3*p4p8*p6p7-hbcm2*p2p4
     . *p4p8*p6p7)
      ans4=16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(
     . p4p8*p4p7*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p4p8*
     . p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p4p6*p3p7*p1p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p4p7*p3p6*p1p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p3p4*p1p8)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+16*(p6p7*p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)
     . +16*(p6p7*p1p8*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p6p7*
     . p4p8)*(-ffmcfmb**2*hbcm4-3*ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3
     . -ffmcfmb*hbcm4-fmb*fmc*hbcm2-2*fmb2*hbcm2+fmc*hbcm3+2*hbcm4)+4
     . *(p6p7*p1p8)*(ffmcfmb**2*hbcm4+3*ffmcfmb*fmb*hbcm3+ffmcfmb*fmc
     . *hbcm3+ffmcfmb*hbcm4+fmb*fmc*hbcm2+2*fmb2*hbcm2-fmc*hbcm3-2*
     . hbcm4)+8*(p4p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+8*(
     . p3p7*p3p6*p1p8)*(ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm-ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+ans5
      ans3=w5*ans4
      ans7=64*(3*fb3*ffmcfmb*p2p8*p3p6*p3p7+fb3*p1p2*p3p6*p7p8-fb3*
     . p1p2*p3p7*p6p8+fb3*p1p3*p2p8*p6p7+fb3*p2p3*p3p7*p6p8-fb3*p2p3*
     . p4p6*p7p8-fb3*p2p4*p3p6*p7p8+fb3*p2p4*p3p7*p6p8-fb3*p2p8*p3p4*
     . p6p7+fb3*p2p8*p3p6*p4p7+2*fb3*p2p8*p3p7*p4p6)
      ans6=64*(p4p8*p3p7*p3p6)*(-fb3*ffmcfmb+fb3)+64*(p6p8*p3p7*p3p4)
     . *(fb3*ffmcfmb-fb3)+64*(p7p8*p3p6*p2p3)*(-fb3*ffmcfmb-fb3)+64*(
     . p3p7*p3p6*p1p8)*(fb3*ffmcfmb-fb3)+64*(p6p8*p3p7*p1p3)*(-fb3*
     . ffmcfmb+fb3)+8*(p6p7*p4p8)*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*
     . fb4*fmb*hbcm+hbcm2)+8*(p6p8*p4p7)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*
     . hbcm2-4*fb4*fmb*hbcm-hbcm2)+8*(p7p8*p4p6)*(4*fb3*ffmcfmb*hbcm2
     . -4*fb3*hbcm2+4*fb4*fmb*hbcm+hbcm2)+4*(p6p8*p3p7)*(8*fb3*
     . ffmcfmb*hbcm2+4*fb3*fmb*fmc+8*fb3*fmb2-8*fb3*hbcm2-4*fb4*
     . ffmcfmb*fmb*hbcm+ffmcfmb*hbcm2-3*fmb*hbcm-fmc*hbcm-2*hbcm2)+4*
     . (p6p7*p2p8)*(-12*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(
     . p6p7*p1p8)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-
     . hbcm2)+4*(p7p8*p3p6)*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*
     . hbcm2-4*fb3*fmb*fmc-8*fb3*fmb2-12*fb4*ffmcfmb*fmb*hbcm-4*fb4*
     . ffmcfmb*fmc*hbcm+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm+ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+ans7
      ans2=w8*(16*(p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16
     . *(p4p7*p3p6*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p3p4
     . *p2p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p2p8*p1p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p6p7*p2p8)*(-ffmcfmb**2*hbcm4
     . -3*ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2-2*fmb2*hbcm2+fmc*hbcm3+2*hbcm4)+8*(p3p7*p3p6*p2p8)*(-
     . ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-2*fmb*hbcm-fmc*hbcm)+16*(hbcm2*p1p2*p2p8*p6p7-hbcm2*p2p3
     . *p2p8*p6p7-hbcm2*p2p4*p2p8*p6p7))+ans3+ans6
      ans1=w1*(16*(p4p8*p4p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16
     . *(p4p8*p4p7*p3p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+16*(p6p7*p4p8
     . *p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+16*(p6p7*p4p8*p1p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p6p7*p4p8)*(-ffmcfmb**2*hbcm4
     . -3*ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2-2*fmb2*hbcm2+fmc*hbcm3+2*hbcm4)+8*(p4p8*p3p7*p3p6)*(-
     . ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-2*fmb*hbcm-fmc*hbcm)+16*(hbcm2*p1p2*p4p8*p6p7-hbcm2*p2p3
     . *p4p8*p6p7-hbcm2*p2p4*p4p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w1*(6*p4p8*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+4*(
     . ffmcfmb*hbcm*p2p3*p4p8+hbcm*p1p2*p4p8+hbcm*p2p4*p4p8-3*hbcm*
     . p2p5*p4p8))+w8*(6*p2p8*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+4*(
     . ffmcfmb*hbcm*p2p3*p2p8+hbcm*p1p2*p2p8+hbcm*p2p4*p2p8-3*hbcm*
     . p2p5*p2p8))+w5*(6*p4p8*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmb2*hbcm+fmc*hbcm2-hbcm3)+6*p1p8*(
     . -ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmb2*hbcm-fmc*hbcm2+hbcm3)+4*(-ffmcfmb*hbcm*p1p8*p2p3+
     . ffmcfmb*hbcm*p2p3*p4p8-hbcm*p1p2*p1p8+hbcm*p1p2*p4p8-hbcm*p1p8
     . *p2p4+3*hbcm*p1p8*p2p5+hbcm*p2p4*p4p8-3*hbcm*p2p5*p4p8))+(6*
     . p5p8*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*p4p8*(4
     . *fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+8*p2p8*(3*fb3*fmb
     . +3*fb4*ffmcfmb*hbcm-3*fb4*hbcm-hbcm)+2*p1p8*(4*fb3*fmb+4*fb4*
     . ffmcfmb*hbcm-4*fb4*hbcm-hbcm)))
      b(11)=ans
      b(12)=ccc*(12*fb3*ffmcfmb*hbcm2+8*fb3*ffmcfmb*p2p3+12*fb3*fmb*
     . fmc+12*fb3*fmb2-12*fb3*hbcm2+8*fb3*p1p2+8*fb3*p2p4-24*fb3*p2p5
     . +12*fb4*ffmcfmb*fmb*hbcm+12*fb4*ffmcfmb*fmc*hbcm-12*fb4*fmc*
     . hbcm-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2)
      ans3=(6*p5p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*fmb*hbcm+
     . hbcm2)+2*p4p8*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm
     . -hbcm2)+8*p2p8*(-3*fb3*ffmcfmb*hbcm2+3*fb3*hbcm2-3*fb4*fmb*
     . hbcm-hbcm2)+2*p1p8*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb
     . *hbcm-hbcm2))
      ans2=w5*(12*(p4p8*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p8
     . *p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p8*p2p3)*(4*ffmcfmb
     . *hbcm2+3*fmb*hbcm-3*hbcm2)+12*(p3p5*p1p8)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+4*(p3p4*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(
     . p2p3*p1p8)*(-4*ffmcfmb*hbcm2-3*fmb*hbcm+3*hbcm2)+4*(p4p8*p1p3)
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p1p8*p1p3)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+2*p4p8*(2*ffmcfmb**2*hbcm4+5*ffmcfmb*fmb*hbcm3
     . +3*ffmcfmb*fmc*hbcm3-5*ffmcfmb*hbcm4+3*fmb*fmc*hbcm2-6*fmb*
     . hbcm3+3*fmb2*hbcm2-3*fmc*hbcm3+3*hbcm4)+2*p1p8*(-2*ffmcfmb**2*
     . hbcm4-5*ffmcfmb*fmb*hbcm3-3*ffmcfmb*fmc*hbcm3+5*ffmcfmb*hbcm4-
     . 3*fmb*fmc*hbcm2+6*fmb*hbcm3-3*fmb2*hbcm2+3*fmc*hbcm3-3*hbcm4)+
     . 4*(-hbcm2*p1p2*p1p8+hbcm2*p1p2*p4p8-hbcm2*p1p8*p2p4+3*hbcm2*
     . p1p8*p2p5+hbcm2*p2p4*p4p8-3*hbcm2*p2p5*p4p8))+ans3
      ans1=w1*(12*(p4p8*p3p5)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p4p8
     . *p3p4)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p8*p2p3)*(4*ffmcfmb
     . *hbcm2+3*fmb*hbcm-3*hbcm2)+4*(p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*
     . hbcm-hbcm2)+2*p4p8*(2*ffmcfmb**2*hbcm4+5*ffmcfmb*fmb*hbcm3+3*
     . ffmcfmb*fmc*hbcm3-5*ffmcfmb*hbcm4+3*fmb*fmc*hbcm2-6*fmb*hbcm3+
     . 3*fmb2*hbcm2-3*fmc*hbcm3+3*hbcm4)+4*(hbcm2*p1p2*p4p8+hbcm2*
     . p2p4*p4p8-3*hbcm2*p2p5*p4p8))+w8*(12*(p3p5*p2p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+4*(p3p4*p2p8)*(ffmcfmb*hbcm2+fmb*hbcm-
     . hbcm2)+4*(p2p8*p2p3)*(4*ffmcfmb*hbcm2+3*fmb*hbcm-3*hbcm2)+4*(
     . p2p8*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+2*p2p8*(2*ffmcfmb**2
     . *hbcm4+5*ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3-5*ffmcfmb*hbcm4
     . +3*fmb*fmc*hbcm2-6*fmb*hbcm3+3*fmb2*hbcm2-3*fmc*hbcm3+3*hbcm4)
     . +4*(hbcm2*p1p2*p2p8+hbcm2*p2p4*p2p8-3*hbcm2*p2p5*p2p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(6*p3p5*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+2*p3p4*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*
     . p2p3*(-12*fb3*fmb-16*fb4*ffmcfmb*hbcm+12*fb4*hbcm+3*hbcm)+2*
     . p1p3*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+(-20*fb3*
     . ffmcfmb*fmb*hbcm2+12*fb3*ffmcfmb*fmc*hbcm2+24*fb3*fmb*hbcm2-12
     . *fb3*fmc*hbcm2-8*fb4*ffmcfmb**2*hbcm3+20*fb4*ffmcfmb*hbcm3+12*
     . fb4*fmb*fmc*hbcm-12*fb4*fmb2*hbcm-8*fb4*hbcm*p1p2-8*fb4*hbcm*
     . p2p4+24*fb4*hbcm*p2p5-12*fb4*hbcm3+2*ffmcfmb*hbcm3-3*fmb*hbcm2
     . +3*fmc*hbcm2-3*hbcm3))
      b(15)=ccc*(w1*(4*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmc*hbcm-2*hbcm2)-
     . 16*hbcm2*p4p6*p4p8)+w8*(4*(p3p6*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm-
     . 2*hbcm2)-16*hbcm2*p2p8*p4p6)+w5*(4*(p4p8*p3p6)*(ffmcfmb*hbcm2-
     . fmc*hbcm-2*hbcm2)+4*(p3p6*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm+2*
     . hbcm2)+16*(hbcm2*p1p8*p4p6-hbcm2*p4p6*p4p8))+(2*p6p8*(-12*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p1p3*p6p8-fb3*p1p8
     . *p3p6-fb3*p3p4*p6p8+fb3*p3p6*p4p8)))
      b(16)=ccc*(w1*(8*(p4p8*p3p6)*(ffmcfmb*hbcm+hbcm)+24*hbcm*p4p6*
     . p4p8)+w8*(8*(p3p6*p2p8)*(ffmcfmb*hbcm+hbcm)+24*hbcm*p2p8*p4p6)
     . +w5*(8*(p4p8*p3p6)*(ffmcfmb*hbcm+hbcm)+8*(p3p6*p1p8)*(-ffmcfmb
     . *hbcm-hbcm)+24*(-hbcm*p1p8*p4p6+hbcm*p4p6*p4p8))+2*p6p8*(-4*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm))
      b(17)=ccc*(w1*(2*p4p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+4*
     . (-hbcm*p1p3*p4p8-3*hbcm*p2p3*p4p8-hbcm*p3p4*p4p8+3*hbcm*p3p5*
     . p4p8))+w8*(2*p2p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+4*(-
     . hbcm*p1p3*p2p8-3*hbcm*p2p3*p2p8-hbcm*p2p8*p3p4+3*hbcm*p2p8*
     . p3p5))+w5*(2*p4p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+2*
     . p1p8*(5*ffmcfmb*hbcm3+3*fmc*hbcm2-6*hbcm3)+4*(hbcm*p1p3*p1p8-
     . hbcm*p1p3*p4p8+3*hbcm*p1p8*p2p3+hbcm*p1p8*p3p4-3*hbcm*p1p8*
     . p3p5-3*hbcm*p2p3*p4p8-hbcm*p3p4*p4p8+3*hbcm*p3p5*p4p8))+8*(fb4
     . *hbcm*p1p8+3*fb4*hbcm*p2p8+fb4*hbcm*p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w1*(4*(p4p8*p3p7)*(-4*ffmcfmb*hbcm2+3*hbcm2)+4*(-
     . hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+w8*(4*(p3p7*p2p8)*(-4*
     . ffmcfmb*hbcm2+3*hbcm2)+4*(-hbcm2*p2p8*p4p7+3*hbcm2*p2p8*p5p7))
     . +w5*(4*(p4p8*p3p7)*(-4*ffmcfmb*hbcm2+3*hbcm2)+4*(p3p7*p1p8)*(4
     . *ffmcfmb*hbcm2-3*hbcm2)+4*(hbcm2*p1p8*p4p7-3*hbcm2*p1p8*p5p7-
     . hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+8*hbcm2*p7p8)
      b(19)=ccc*(6*w1*p4p8*(-ffmcfmb*hbcm2+fmc*hbcm)+6*w8*p2p8*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+w5*(6*p4p8*(-ffmcfmb*hbcm2+fmc*hbcm)+6
     . *p1p8*(ffmcfmb*hbcm2-fmc*hbcm))+8*(-fb3*p1p8-3*fb3*p2p8-fb3*
     . p4p8+3*fb3*p5p8))
      b(20)=ccc*(w1*(4*(p4p8*p3p7)*(-4*ffmcfmb*hbcm+3*hbcm)+4*(-hbcm*
     . p4p7*p4p8+3*hbcm*p4p8*p5p7))+w8*(4*(p3p7*p2p8)*(-4*ffmcfmb*
     . hbcm+3*hbcm)+4*(-hbcm*p2p8*p4p7+3*hbcm*p2p8*p5p7))+w5*(4*(p4p8
     . *p3p7)*(-4*ffmcfmb*hbcm+3*hbcm)+4*(p3p7*p1p8)*(4*ffmcfmb*hbcm-
     . 3*hbcm)+4*(hbcm*p1p8*p4p7-3*hbcm*p1p8*p5p7-hbcm*p4p7*p4p8+3*
     . hbcm*p4p8*p5p7))+8*hbcm*p7p8)
      b(21)=3*ccc*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=ccc*(20*fb3*ffmcfmb*hbcm2-24*fb3*hbcm2+8*fb3*p1p3+24*fb3*
     . p2p3+8*fb3*p3p4-24*fb3*p3p5-12*fb4*fmc*hbcm+3*hbcm2)
      b(23)=ccc*(8*p3p7*(4*fb4*ffmcfmb*hbcm-3*fb4*hbcm)+8*(fb4*hbcm*
     . p4p7-3*fb4*hbcm*p5p7))
      b(24)=ccc*(2*p3p6*(4*fb3*fmc-4*fb4*ffmcfmb*hbcm-8*fb4*hbcm-hbcm
     . )-48*fb4*hbcm*p4p6)
      b(25)=ccc*(6*p5p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+2*p4p7*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*
     . p3p7*(-16*fb3*ffmcfmb*fmb+12*fb3*ffmcfmb*fmc+12*fb3*fmb-12*fb3
     . *fmc-4*fb4*ffmcfmb**2*hbcm+16*fb4*ffmcfmb*hbcm-12*fb4*hbcm+
     . ffmcfmb*hbcm))
      b(26)=ccc*(12*p4p6*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+4*p3p6*(4*fb3*ffmcfmb*fmb+4*fb3*fmb+4*fb4*ffmcfmb**2*
     . hbcm-4*fb4*hbcm-ffmcfmb*hbcm-hbcm))
      b(27)=ccc*(2*p6p7*(12*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+
     . 32*(-2*fb3*ffmcfmb*p3p6*p3p7-fb3*p1p3*p6p7+fb3*p3p4*p6p7-fb3*
     . p3p6*p4p7-fb3*p3p7*p4p6))
      b(28)=ccc*(4*w1*(p6p7*p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+4*w8*(p6p7
     . *p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+w5*(4*(p6p7*p4p8)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(p6p7*p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm))+32*(-
     . fb3*ffmcfmb*p3p6*p7p8-fb3*ffmcfmb*p3p7*p6p8-fb3*p1p8*p6p7-fb3*
     . p4p6*p7p8-fb3*p4p7*p6p8+fb3*p4p8*p6p7))
      b(29)=ccc*(16*p3p6*(-2*fb3*ffmcfmb-fb3)-64*fb3*p4p6)
      b(30)=ccc*(8*p3p7*(4*fb3*ffmcfmb-3*fb3)+8*(fb3*p4p7-3*fb3*p5p7)
     . )
      ans=ccc*(w1*(4*(p6p7*p4p8)*(-3*ffmcfmb*hbcm3-fmc*hbcm2)+16*(2*
     . ffmcfmb*hbcm*p3p6*p3p7*p4p8+hbcm*p1p3*p4p8*p6p7-hbcm*p3p4*p4p8
     . *p6p7+hbcm*p3p6*p4p7*p4p8+hbcm*p3p7*p4p6*p4p8))+w8*(4*(p6p7*
     . p2p8)*(-3*ffmcfmb*hbcm3-fmc*hbcm2)+16*(2*ffmcfmb*hbcm*p2p8*
     . p3p6*p3p7+hbcm*p1p3*p2p8*p6p7-hbcm*p2p8*p3p4*p6p7+hbcm*p2p8*
     . p3p6*p4p7+hbcm*p2p8*p3p7*p4p6))+w5*(4*(p6p7*p4p8)*(-3*ffmcfmb*
     . hbcm3-fmc*hbcm2)+4*(p6p7*p1p8)*(3*ffmcfmb*hbcm3+fmc*hbcm2)+16*
     . (-2*ffmcfmb*hbcm*p1p8*p3p6*p3p7+2*ffmcfmb*hbcm*p3p6*p3p7*p4p8-
     . hbcm*p1p3*p1p8*p6p7+hbcm*p1p3*p4p8*p6p7+hbcm*p1p8*p3p4*p6p7-
     . hbcm*p1p8*p3p6*p4p7-hbcm*p1p8*p3p7*p4p6-hbcm*p3p4*p4p8*p6p7+
     . hbcm*p3p6*p4p7*p4p8+hbcm*p3p7*p4p6*p4p8))+(8*(p6p8*p3p7)*(-4*
     . fb4*ffmcfmb*hbcm-hbcm)+8*(p7p8*p3p6)*(-4*fb4*ffmcfmb*hbcm-hbcm
     . )+32*(-fb4*hbcm*p1p8*p6p7-fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*
     . p6p8+fb4*hbcm*p4p8*p6p7)))
      b(31)=ans
      b(32)=2*ccc*p6p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp5_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(31) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,31 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*
     . p3p4+fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p2p3-fmb2+hbcm2-2*p2p3))
      b(3)=ccc*(8*w8*(p3p7*p2p8)*(ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+w5*(8*(p4p8*p3p7)*(ffmcfmb*fmb*
     . hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm+hbcm2)+8*(p3p7*
     . p1p8)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*
     . hbcm-hbcm2))+(4*p7p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-4*fb3
     . *fmb2+4*fb3*hbcm2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*
     . hbcm+4*fb4*fmc*hbcm+hbcm2)+32*(fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3
     . *p2p8*p5p7)))
      b(4)=ccc*(4*p3p7*(4*fb3*fmb*fmc+4*fb3*fmb2-fmb*hbcm-fmc*hbcm)+
     . 32*(fb3*p2p3*p5p7-fb3*p2p5*p3p7))
      ans2=(8*(p5p8*p3p7)*(4*fb3*fmb-hbcm)+32*(p7p8*p3p5)*(-fb3*fmb-
     . fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(p3p7*p2p8)*(-4*fb3*fmb+hbcm)+32*
     . (p7p8*p2p3)*(fb3*fmb+fb4*ffmcfmb*hbcm-fb4*hbcm)+4*p7p8*(4*fb3*
     . ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-8*fb3*fmb*hbcm2+4*
     . fb3*fmc*hbcm2-4*fb4*ffmcfmb*hbcm3-4*fb4*fmb*fmc*hbcm+4*fb4*
     . fmb2*hbcm+4*fb4*hbcm3-fmb*hbcm2+fmc*hbcm2)+32*(-fb4*hbcm*p2p5*
     . p7p8+fb4*hbcm*p2p8*p5p7))
      ans1=w8*(8*(p5p7*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+16*(
     . p3p7*p3p5*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(p3p7*p2p8*p2p3)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p8)*(ffmcfmb*fmb*hbcm2-ffmcfmb*
     . fmc*hbcm2+ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm+fmc*
     . hbcm2-hbcm3)+16*(-hbcm*p2p3*p2p8*p5p7+hbcm*p2p5*p2p8*p3p7))+w5
     . *(8*(p5p7*p4p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p5p7*p1p8)
     . *(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+16*(p4p8*p3p7*p3p5)*(ffmcfmb*
     . hbcm-hbcm)+16*(p4p8*p3p7*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*
     . p3p5*p1p8)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*p2p3*p1p8)*(ffmcfmb*
     . hbcm-hbcm)+8*(p4p8*p3p7)*(ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmb*hbcm2-fmb2*hbcm+fmc*hbcm2-hbcm3
     . )+8*(p3p7*p1p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+fmb*hbcm2+fmb2*hbcm-fmc*hbcm2+hbcm3)+16*(
     . hbcm*p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p2p3*p4p8*p5p7+
     . hbcm*p2p5*p3p7*p4p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(11)=ccc*(w8*(4*p2p8*(-ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(-hbcm
     . *p2p3*p2p8-hbcm*p2p5*p2p8))+w5*(4*p4p8*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+4*p1p8*(ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(hbcm*
     . p1p8*p2p3+hbcm*p1p8*p2p5-hbcm*p2p3*p4p8-hbcm*p2p5*p4p8))+(4*
     . p2p8*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p5p8*(-4*fb3*fmb-4*
     . fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)))
      b(12)=2*ccc*(4*fb3*fmb*fmc+4*fb3*fmb2-8*fb3*p2p3-16*fb3*p2p5+4*
     . fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmb*hbcm-4*
     . fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm)
      b(13)=ccc*(w8*(8*(p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+4*p2p8*(-
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmb*hbcm3-
     . fmb2*hbcm2+fmc*hbcm3))+w5*(8*(p4p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+
     . 8*(p2p3*p1p8)*(fmb*hbcm+fmc*hbcm)+4*p4p8*(-ffmcfmb*fmb*hbcm3-
     . ffmcfmb*fmc*hbcm3-fmb*fmc*hbcm2+fmb*hbcm3-fmb2*hbcm2+fmc*hbcm3
     . )+4*p1p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+fmb*fmc*hbcm2-
     . fmb*hbcm3+fmb2*hbcm2-fmc*hbcm3))+(16*p2p8*(fb3*hbcm2+fb4*fmb*
     . hbcm+fb4*fmc*hbcm)+32*(-fb3*p2p3*p5p8+fb3*p2p8*p3p5)))
      b(14)=ccc*(4*p2p3*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+4*p3p5*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+2*
     . (4*fb3*fmb*hbcm2+4*fb4*ffmcfmb*hbcm3+8*fb4*hbcm*p2p5-4*fb4*
     . hbcm3-hbcm3))
      b(17)=ccc*(w8*(4*p2p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*hbcm*p2p8*
     . p3p5)+w5*(4*p4p8*(ffmcfmb*hbcm3+fmb*hbcm2)+4*p1p8*(-ffmcfmb*
     . hbcm3-fmb*hbcm2)+8*(-hbcm*p1p8*p3p5+hbcm*p3p5*p4p8))-16*fb4*
     . hbcm*p5p8)
      b(18)=ccc*(8*w8*(p3p7*p2p8)*(ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+w5*(
     . 8*(p4p8*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm-hbcm2)+8*(p3p7*p1p8)*(-
     . ffmcfmb*hbcm2-fmc*hbcm+hbcm2))+(4*p7p8*(4*fb3*ffmcfmb*hbcm2-8*
     . fb3*hbcm2-4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p2p3*p7p8-fb3*p2p8*
     . p3p7-fb3*p3p5*p7p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w8*p2p8*(fmb*hbcm+fmc*hbcm)+w5*(4*p4p8*(fmb*hbcm+
     . fmc*hbcm)+4*p1p8*(-fmb*hbcm-fmc*hbcm))+16*(-fb3*p2p8+2*fb3*
     . p5p8))
      b(20)=ccc*(8*w8*(ffmcfmb*hbcm*p2p8*p3p7+hbcm*p2p8*p5p7)+8*w5*(-
     . ffmcfmb*hbcm*p1p8*p3p7+ffmcfmb*hbcm*p3p7*p4p8-hbcm*p1p8*p5p7+
     . hbcm*p4p8*p5p7)+16*p7p8*(-fb3*fmc-fb4*ffmcfmb*hbcm))
      b(21)=2*ccc*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p2p3-4*fb3*p3p5-fb4*fmb*hbcm-
     . fb4*fmc*hbcm)
      b(23)=ccc*(4*p3p7*(4*fb3*fmc-hbcm)-16*fb4*hbcm*p5p7)
      b(25)=ccc*(4*p5p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+4*p3p7*(4*fb3*fmb-hbcm))
      b(30)=16*ccc*(-fb3*p3p7-2*fb3*p5p7)
      do 200 n=1,31 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp4_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(30) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,30 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3-fmc2)
     . )
      b(3)=ccc*(8*w2*(p5p8*p3p7)*(-ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2+fmc*hbcm)+8*w7*(p3p7*p2p8)*(-ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+(4*p7p8*(4*fb3*
     . ffmcfmb*hbcm2+4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm+hbcm2)+32*(fb3*
     . p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*p5p7)))
      b(4)=ccc*(16*p3p7*(-fb4*ffmcfmb*fmb*hbcm-fb4*ffmcfmb*fmc*hbcm)+
     . 32*(fb3*p2p3*p5p7-fb3*p2p5*p3p7))
      b(5)=ccc*(w2*(8*(p5p8*p5p7)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p5p8*
     . p3p7)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3-fmb*
     . fmc*hbcm+fmc2*hbcm)+16*(ffmcfmb*hbcm*p2p3*p3p7*p5p8-ffmcfmb*
     . hbcm*p3p5*p3p7*p5p8))+w7*(8*(p5p7*p2p8)*(ffmcfmb*hbcm3-fmc*
     . hbcm2)+8*(p3p7*p2p8)*(-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-
     . ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc2*hbcm)+16*(ffmcfmb*hbcm*p2p3*
     . p2p8*p3p7-ffmcfmb*hbcm*p2p8*p3p5*p3p7))+(4*p7p8*(-4*fb3*
     . ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*hbcm2-4*fb4*fmb*fmc*hbcm+4
     . *fb4*fmc2*hbcm-fmb*hbcm2+fmc*hbcm2-hbcm3)+8*(4*fb4*ffmcfmb*
     . hbcm*p2p8*p3p7-4*fb4*ffmcfmb*hbcm*p3p7*p5p8+hbcm*p2p3*p7p8-
     . hbcm*p3p5*p7p8)))
      b(11)=ccc*(w2*(4*p5p8*(ffmcfmb*hbcm3-fmc*hbcm2)+8*hbcm*p2p5*
     . p5p8)+w7*(4*p2p8*(ffmcfmb*hbcm3-fmc*hbcm2)+8*hbcm*p2p5*p2p8)+(
     . 4*p5p8*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*p2p8*(-4*fb3*fmb-
     . 4*fb4*ffmcfmb*hbcm+hbcm)))
      b(12)=2*ccc*(-4*fb3*fmb*fmc-4*fb3*fmc2-8*fb3*p2p3-16*fb3*p2p5-4
     . *fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm
     . )
      b(13)=ccc*(4*w2*p5p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3-fmb*
     . fmc*hbcm2-fmc2*hbcm2)+4*w7*p2p8*(ffmcfmb*fmb*hbcm3+ffmcfmb*fmc
     . *hbcm3-fmb*fmc*hbcm2-fmc2*hbcm2)+16*(fb3*hbcm2*p2p8-2*fb3*p2p3
     . *p5p8+2*fb3*p2p8*p3p5))
      b(14)=ccc*(4*p3p5*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p2p3*(
     . 4*fb3*fmb+4*fb4*ffmcfmb*hbcm-hbcm)+2*(-4*fb3*fmc*hbcm2-4*fb4*
     . ffmcfmb*hbcm3+8*fb4*hbcm*p2p5+hbcm3))
      b(17)=ccc*(w2*(4*p5p8*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(hbcm*
     . p2p3*p5p8-hbcm*p3p5*p5p8))+w7*(4*p2p8*(-ffmcfmb*hbcm3+fmc*
     . hbcm2-hbcm3)+8*(hbcm*p2p3*p2p8-hbcm*p2p8*p3p5))+16*(fb4*hbcm*
     . p2p8-fb4*hbcm*p5p8))
      b(18)=ccc*(8*w2*(p5p8*p3p7)*(-ffmcfmb*hbcm2-fmc*hbcm)+8*w7*(
     . p3p7*p2p8)*(-ffmcfmb*hbcm2-fmc*hbcm)+(4*p7p8*(-4*fb3*ffmcfmb*
     . hbcm2-4*fb3*hbcm2-4*fb4*fmc*hbcm-hbcm2)+32*(fb3*p2p3*p7p8-fb3*
     . p2p8*p3p7-fb3*p3p5*p7p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w2*p5p8*(-fmb*hbcm-fmc*hbcm)+4*w7*p2p8*(-fmb*hbcm-
     . fmc*hbcm)+16*(-fb3*p2p8+2*fb3*p5p8))
      b(20)=ccc*(8*w2*(-ffmcfmb*hbcm*p3p7*p5p8-hbcm*p5p7*p5p8)+8*w7*(
     . -ffmcfmb*hbcm*p2p8*p3p7-hbcm*p2p8*p5p7)+4*p7p8*(4*fb3*fmb-4*
     . fb3*fmc-hbcm))
      b(21)=2*ccc*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p2p3-4*fb3*p3p5-fb4*fmb*hbcm-
     . fb4*fmc*hbcm)
      b(23)=ccc*(16*p3p7*(-fb3*fmb+fb3*fmc-fb4*ffmcfmb*hbcm)-16*fb4*
     . hbcm*p5p7)
      b(25)=ccc*(4*p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)-16*fb4*
     . ffmcfmb*hbcm*p3p7)
      b(30)=16*ccc*(-fb3*p3p7-2*fb3*p5p7)
      do 200 n=1,30 
         c(n,2)=c(n,2)+0.09245003270420485d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp36_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=ccc*(32*(p4p6*p3p7)*(2*fb3*ffmcfmb-fb3)+16*(p5p7*p3p6)*(-
     . fb3*ffmcfmb-fb3)+16*(p4p7*p3p6)*(-fb3*ffmcfmb+2*fb3)+p6p7*(4*
     . fb3*ffmcfmb**2*hbcm2-12*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-8*fb3*
     . fmc2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*
     . hbcm+fmb*hbcm+fmc*hbcm-hbcm2)+16*(fb3*ffmcfmb*p3p6*p3p7+fb3*
     . p1p2*p6p7+fb3*p1p3*p6p7+fb3*p1p4*p6p7-fb3*p2p3*p6p7-fb3*p2p4*
     . p6p7-fb3*p3p4*p6p7+fb3*p3p7*p5p6+2*fb3*p4p6*p4p7-3*fb3*p4p6*
     . p5p7+fb3*p4p7*p5p6))
      ans3=(2*(p6p8*p5p7)*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-8*
     . fb4*hbcm-hbcm)+4*(p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+2*(p6p8*p4p7)*(-8*fb3*fmb+4*fb3*fmc-4*fb4*
     . ffmcfmb*hbcm+8*fb4*hbcm-3*hbcm)+4*(p7p8*p4p6)*(-4*fb3*fmb-4*
     . fb4*ffmcfmb*hbcm+4*fb4*hbcm+3*hbcm)+16*(p6p7*p2p8)*(fb3*fmb-
     . fb3*fmc-fb4*hbcm)+4*(p6p7*p1p8)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm
     . +4*fb4*hbcm+hbcm)+2*(p6p8*p3p7)*(-16*fb3*ffmcfmb*fmb+8*fb3*
     . ffmcfmb*fmc+8*fb3*fmb-4*fb3*fmc-8*fb4*ffmcfmb**2*hbcm+20*fb4*
     . ffmcfmb*hbcm-8*fb4*hbcm-2*ffmcfmb*hbcm-hbcm)+4*(p7p8*p3p6)*(-4
     . *fb3*ffmcfmb*fmb-4*fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm-
     . ffmcfmb*hbcm+2*hbcm))
      ans2=w5*(8*(p4p8*p4p6*p3p7)*(-2*ffmcfmb*hbcm+hbcm)+16*(p4p8*
     . p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p4p6*p3p7*p1p8)*(2*ffmcfmb*
     . hbcm-hbcm)+16*(p4p7*p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+2*(p6p7*
     . p4p8)*(-ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3
     . *ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*fmc2*hbcm)+2*(p6p7*
     . p1p8)*(ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-3*
     . ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*fmc2*hbcm)+16*(p4p8*
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(p3p7*p3p6*p1p8)*
     . (-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+8*(hbcm*p1p2*p1p8*p6p7-hbcm*
     . p1p2*p4p8*p6p7+hbcm*p1p3*p1p8*p6p7-hbcm*p1p3*p4p8*p6p7+hbcm*
     . p1p4*p1p8*p6p7-hbcm*p1p4*p4p8*p6p7-hbcm*p1p8*p2p3*p6p7-hbcm*
     . p1p8*p2p4*p6p7-hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p5p7+hbcm*
     . p1p8*p3p7*p5p6+hbcm*p1p8*p4p6*p4p7-2*hbcm*p1p8*p4p6*p5p7+hbcm*
     . p1p8*p4p7*p5p6+hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7+hbcm*
     . p3p4*p4p8*p6p7+hbcm*p3p6*p4p8*p5p7-hbcm*p3p7*p4p8*p5p6-hbcm*
     . p4p6*p4p7*p4p8+2*hbcm*p4p6*p4p8*p5p7-hbcm*p4p7*p4p8*p5p6))+
     . ans3
      ans1=w2*(8*(p5p8*p4p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+16*(p5p8*p4p7
     . *p3p6)*(-ffmcfmb*hbcm+hbcm)+2*(p6p7*p5p8)*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3+fmb*fmc*
     . hbcm-fmc*hbcm2-2*fmc2*hbcm)+16*(p5p8*p3p7*p3p6)*(-ffmcfmb**2*
     . hbcm+ffmcfmb*hbcm)+8*(hbcm*p1p2*p5p8*p6p7+hbcm*p1p3*p5p8*p6p7+
     . hbcm*p1p4*p5p8*p6p7-hbcm*p2p3*p5p8*p6p7-hbcm*p2p4*p5p8*p6p7-
     . hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p5p7*p5p8+hbcm*p3p7*p5p6*p5p8+
     . hbcm*p4p6*p4p7*p5p8-2*hbcm*p4p6*p5p7*p5p8+hbcm*p4p7*p5p6*p5p8)
     . )+w1*(8*(p4p8*p4p6*p3p7)*(-2*ffmcfmb*hbcm+hbcm)+16*(p4p8*p4p7*
     . p3p6)*(ffmcfmb*hbcm-hbcm)+2*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*fmb*hbcm2-ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3-fmb*fmc*
     . hbcm+fmc*hbcm2+2*fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*
     . hbcm-ffmcfmb*hbcm)+8*(-hbcm*p1p2*p4p8*p6p7-hbcm*p1p3*p4p8*p6p7
     . -hbcm*p1p4*p4p8*p6p7+hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7+
     . hbcm*p3p4*p4p8*p6p7+hbcm*p3p6*p4p8*p5p7-hbcm*p3p7*p4p8*p5p6-
     . hbcm*p4p6*p4p7*p4p8+2*hbcm*p4p6*p4p8*p5p7-hbcm*p4p7*p4p8*p5p6)
     . )+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(p7p8*(ffmcfmb*hbcm2+7*fmb*hbcm+6*fmc*hbcm-7*hbcm2)+8*(
     . p5p8*p3p7)*(-7*fb3*ffmcfmb+3*fb3)+8*(p4p8*p3p7)*(fb3*ffmcfmb-
     . fb3)+8*(p3p7*p1p8)*(3*fb3*ffmcfmb-fb3)+8*(-fb3*ffmcfmb*p2p8*
     . p3p7+2*fb3*p1p8*p4p7-fb3*p1p8*p5p7-fb3*p2p8*p4p7-4*fb3*p4p7*
     . p5p8-fb3*p4p8*p5p7+3*fb3*p5p7*p5p8))
      ans1=w2*(6*(p5p8*p5p7)*(-fmb*hbcm-fmc*hbcm+hbcm2)+2*(p5p8*p4p7)
     . *(ffmcfmb*hbcm2+4*fmb*hbcm+3*fmc*hbcm-4*hbcm2)+2*(p5p8*p3p7)*(
     . ffmcfmb**2*hbcm2+7*ffmcfmb*fmb*hbcm+6*ffmcfmb*fmc*hbcm-7*
     . ffmcfmb*hbcm2-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2))+w1*(6*(p5p7*p4p8
     . )*(fmb*hbcm+fmc*hbcm-hbcm2)+2*(p4p8*p4p7)*(-ffmcfmb*hbcm2-4*
     . fmb*hbcm-3*fmc*hbcm+4*hbcm2)+2*(p4p8*p3p7)*(-ffmcfmb**2*hbcm2-
     . 7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm+7*ffmcfmb*hbcm2+3*fmb*
     . hbcm+3*fmc*hbcm-3*hbcm2))+w5*(6*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm
     . -hbcm2)+6*(p5p7*p1p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+2*(p4p8*p4p7)
     . *(-ffmcfmb*hbcm2-4*fmb*hbcm-3*fmc*hbcm+4*hbcm2)+2*(p4p7*p1p8)*
     . (ffmcfmb*hbcm2+4*fmb*hbcm+3*fmc*hbcm-4*hbcm2)+2*(p4p8*p3p7)*(-
     . ffmcfmb**2*hbcm2-7*ffmcfmb*fmb*hbcm-6*ffmcfmb*fmc*hbcm+7*
     . ffmcfmb*hbcm2+3*fmb*hbcm+3*fmc*hbcm-3*hbcm2)+2*(p3p7*p1p8)*(
     . ffmcfmb**2*hbcm2+7*ffmcfmb*fmb*hbcm+6*ffmcfmb*fmc*hbcm-7*
     . ffmcfmb*hbcm2-3*fmb*hbcm-3*fmc*hbcm+3*hbcm2))+ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=ccc*(8*(p3p7*p3p5)*(7*fb3*ffmcfmb-3*fb3)+8*(p3p7*p3p4)*(-
     . fb3*ffmcfmb+fb3)+8*(p3p7*p2p3)*(-7*fb3*ffmcfmb+3*fb3)+8*(p3p7*
     . p1p3)*(-3*fb3*ffmcfmb+fb3)+4*p5p7*(2*fb3*ffmcfmb*hbcm2-3*fb3*
     . hbcm2-3*fb4*fmb*hbcm-3*fb4*fmc*hbcm)+p4p7*(-4*fb3*ffmcfmb*
     . hbcm2+16*fb3*hbcm2+16*fb4*fmb*hbcm+12*fb4*fmc*hbcm+hbcm2)+p3p7
     . *(-12*fb3*ffmcfmb**2*hbcm2+36*fb3*ffmcfmb*hbcm2-12*fb3*hbcm2+
     . 28*fb4*ffmcfmb*fmb*hbcm+24*fb4*ffmcfmb*fmc*hbcm-12*fb4*fmb*
     . hbcm-12*fb4*fmc*hbcm+ffmcfmb*hbcm2)+8*(-2*fb3*p1p3*p4p7+fb3*
     . p1p3*p5p7-4*fb3*p2p3*p4p7+3*fb3*p2p3*p5p7+fb3*p3p4*p5p7+4*fb3*
     . p3p5*p4p7-3*fb3*p3p5*p5p7))
      ans5=4*(-2*hbcm*p1p3*p1p8*p4p7+hbcm*p1p3*p1p8*p5p7+2*hbcm*p1p3*
     . p4p7*p4p8-hbcm*p1p3*p4p8*p5p7-4*hbcm*p1p8*p2p3*p4p7+3*hbcm*
     . p1p8*p2p3*p5p7+hbcm*p1p8*p3p4*p5p7+4*hbcm*p1p8*p3p5*p4p7-3*
     . hbcm*p1p8*p3p5*p5p7+4*hbcm*p2p3*p4p7*p4p8-3*hbcm*p2p3*p4p8*
     . p5p7-hbcm*p3p4*p4p8*p5p7-4*hbcm*p3p5*p4p7*p4p8+3*hbcm*p3p5*
     . p4p8*p5p7)
      ans4=2*(p5p7*p4p8)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2+3*
     . hbcm3)+2*(p4p8*p4p7)*(ffmcfmb*hbcm3-4*fmb*hbcm2+3*fmc*hbcm2-4*
     . hbcm3)+2*(p5p7*p1p8)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-
     . 3*hbcm3)+2*(p4p7*p1p8)*(-ffmcfmb*hbcm3+4*fmb*hbcm2-3*fmc*hbcm2
     . +4*hbcm3)+4*(p4p8*p3p7*p3p5)*(-7*ffmcfmb*hbcm+3*hbcm)+4*(p4p8*
     . p3p7*p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p4p8*p3p7*p2p3)*(7*ffmcfmb*
     . hbcm-3*hbcm)+4*(p3p7*p3p5*p1p8)*(7*ffmcfmb*hbcm-3*hbcm)+4*(
     . p3p7*p3p4*p1p8)*(-ffmcfmb*hbcm+hbcm)+4*(p3p7*p2p3*p1p8)*(-7*
     . ffmcfmb*hbcm+3*hbcm)+4*(p4p8*p3p7*p1p3)*(3*ffmcfmb*hbcm-hbcm)+
     . 4*(p3p7*p1p8*p1p3)*(-3*ffmcfmb*hbcm+hbcm)+2*(p4p8*p3p7)*(3*
     . ffmcfmb**2*hbcm3-7*ffmcfmb*fmb*hbcm2+6*ffmcfmb*fmc*hbcm2-9*
     . ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+2*(p3p7*p1p8)*(
     . -3*ffmcfmb**2*hbcm3+7*ffmcfmb*fmb*hbcm2-6*ffmcfmb*fmc*hbcm2+9*
     . ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-3*hbcm3)+ans5
      ans3=w5*ans4
      ans6=(p7p8*(-3*ffmcfmb*hbcm3+7*fmb*hbcm2-6*fmc*hbcm2+7*hbcm3)+8
     . *(p5p8*p3p7)*(7*fb4*ffmcfmb*hbcm-3*fb4*hbcm)+8*(p4p8*p3p7)*(-
     . fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(p3p7*p2p8)*(-7*fb4*ffmcfmb*hbcm+
     . 3*fb4*hbcm)+8*(p3p7*p1p8)*(-3*fb4*ffmcfmb*hbcm+fb4*hbcm)+2*(-8
     . *fb4*hbcm*p1p8*p4p7+4*fb4*hbcm*p1p8*p5p7-16*fb4*hbcm*p2p8*p4p7
     . +12*fb4*hbcm*p2p8*p5p7+16*fb4*hbcm*p4p7*p5p8+4*fb4*hbcm*p4p8*
     . p5p7-12*fb4*hbcm*p5p7*p5p8-3*hbcm*p1p3*p7p8-7*hbcm*p2p3*p7p8-
     . hbcm*p3p4*p7p8+7*hbcm*p3p5*p7p8))
      ans2=w1*(2*(p5p7*p4p8)*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*
     . hbcm2+3*hbcm3)+2*(p4p8*p4p7)*(ffmcfmb*hbcm3-4*fmb*hbcm2+3*fmc*
     . hbcm2-4*hbcm3)+4*(p4p8*p3p7*p3p5)*(-7*ffmcfmb*hbcm+3*hbcm)+4*(
     . p4p8*p3p7*p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p4p8*p3p7*p2p3)*(7*
     . ffmcfmb*hbcm-3*hbcm)+4*(p4p8*p3p7*p1p3)*(3*ffmcfmb*hbcm-hbcm)+
     . 2*(p4p8*p3p7)*(3*ffmcfmb**2*hbcm3-7*ffmcfmb*fmb*hbcm2+6*
     . ffmcfmb*fmc*hbcm2-9*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2+3*
     . hbcm3)+4*(2*hbcm*p1p3*p4p7*p4p8-hbcm*p1p3*p4p8*p5p7+4*hbcm*
     . p2p3*p4p7*p4p8-3*hbcm*p2p3*p4p8*p5p7-hbcm*p3p4*p4p8*p5p7-4*
     . hbcm*p3p5*p4p7*p4p8+3*hbcm*p3p5*p4p8*p5p7))+ans3+ans6
      ans1=w2*(2*(p5p8*p5p7)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2
     . -3*hbcm3)+2*(p5p8*p4p7)*(-ffmcfmb*hbcm3+4*fmb*hbcm2-3*fmc*
     . hbcm2+4*hbcm3)+4*(p5p8*p3p7*p3p5)*(7*ffmcfmb*hbcm-3*hbcm)+4*(
     . p5p8*p3p7*p3p4)*(-ffmcfmb*hbcm+hbcm)+4*(p5p8*p3p7*p2p3)*(-7*
     . ffmcfmb*hbcm+3*hbcm)+4*(p5p8*p3p7*p1p3)*(-3*ffmcfmb*hbcm+hbcm)
     . +2*(p5p8*p3p7)*(-3*ffmcfmb**2*hbcm3+7*ffmcfmb*fmb*hbcm2-6*
     . ffmcfmb*fmc*hbcm2+9*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-3*
     . hbcm3)+4*(-2*hbcm*p1p3*p4p7*p5p8+hbcm*p1p3*p5p7*p5p8-4*hbcm*
     . p2p3*p4p7*p5p8+3*hbcm*p2p3*p5p7*p5p8+hbcm*p3p4*p5p7*p5p8+4*
     . hbcm*p3p5*p4p7*p5p8-3*hbcm*p3p5*p5p7*p5p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w2*(2*(p5p8*p5p6)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-2
     . *hbcm2)+2*(p5p8*p3p6)*(-ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-2*hbcm2))
     . +w1*(2*(p5p6*p4p8)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*hbcm2
     . )+2*(p4p8*p3p6)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+ffmcfmb*
     . fmc*hbcm-3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*hbcm2))+w5*(2*(
     . p5p6*p4p8)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*hbcm2)+2*(
     . p5p6*p1p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-2*hbcm2)+2*(p4p8
     . *p3p6)*(ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-3
     . *ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+2*hbcm2)+2*(p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+3*ffmcfmb
     . *hbcm2+2*fmb*hbcm+fmc*hbcm-2*hbcm2))+(p6p8*(-ffmcfmb*hbcm2-2*
     . fmb*hbcm-fmc*hbcm+2*hbcm2)+16*(p4p8*p3p6)*(-fb3*ffmcfmb+fb3)+
     . 16*(p3p6*p1p8)*(fb3*ffmcfmb-fb3)+16*(-fb3*ffmcfmb*p2p8*p3p6-
     . fb3*ffmcfmb*p3p6*p5p8-fb3*p1p8*p5p6-fb3*p2p8*p4p6-fb3*p4p6*
     . p5p8+fb3*p4p8*p5p6)))
      b(6)=ans
      b(7)=ccc*(16*(p3p6*p3p4)*(fb3*ffmcfmb-fb3)+16*(p3p6*p1p3)*(-fb3
     . *ffmcfmb+fb3)+p5p6*(-4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+8*fb4*fmb
     . *hbcm+4*fb4*fmc*hbcm+hbcm2)+16*p4p6*(-fb3*ffmcfmb*hbcm2+fb3*
     . hbcm2)+p3p6*(-12*fb3*ffmcfmb**2*hbcm2+20*fb3*ffmcfmb*hbcm2-8*
     . fb3*hbcm2-8*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+8*fb4*
     . fmb*hbcm+4*fb4*fmc*hbcm-ffmcfmb*hbcm2+hbcm2)+16*(fb3*ffmcfmb*
     . p2p3*p3p6+fb3*ffmcfmb*p3p5*p3p6+fb3*p1p3*p5p6+fb3*p2p3*p4p6-
     . fb3*p3p4*p5p6+fb3*p3p5*p4p6))
      ans3=(p6p8*(-3*ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2+6*hbcm3)+16*
     . (p4p8*p3p6)*(fb4*ffmcfmb*hbcm-fb4*hbcm)+16*(p3p6*p1p8)*(-fb4*
     . ffmcfmb*hbcm+fb4*hbcm)+4*(4*fb4*ffmcfmb*hbcm*p2p8*p3p6+4*fb4*
     . ffmcfmb*hbcm*p3p6*p5p8+4*fb4*hbcm*p1p8*p5p6+4*fb4*hbcm*p2p8*
     . p4p6+4*fb4*hbcm*p4p6*p5p8-4*fb4*hbcm*p4p8*p5p6-hbcm*p1p3*p6p8+
     . hbcm*p2p3*p6p8+hbcm*p3p4*p6p8+hbcm*p3p5*p6p8))
      ans2=w5*(2*(p5p6*p4p8)*(ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2+2*
     . hbcm3)+8*(p4p8*p4p6)*(ffmcfmb*hbcm3-hbcm3)+2*(p5p6*p1p8)*(-
     . ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2-2*hbcm3)+8*(p4p6*p1p8)*(-
     . ffmcfmb*hbcm3+hbcm3)+8*(p4p8*p3p6*p3p4)*(-ffmcfmb*hbcm+hbcm)+8
     . *(p3p6*p3p4*p1p8)*(ffmcfmb*hbcm-hbcm)+8*(p4p8*p3p6*p1p3)*(
     . ffmcfmb*hbcm-hbcm)+8*(p3p6*p1p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+2*(
     . p4p8*p3p6)*(3*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*hbcm2-ffmcfmb*fmc
     . *hbcm2-5*ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2+2*hbcm3)+2*(p3p6*
     . p1p8)*(-3*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*fmc*
     . hbcm2+5*ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2-2*hbcm3)+8*(
     . ffmcfmb*hbcm*p1p8*p2p3*p3p6+ffmcfmb*hbcm*p1p8*p3p5*p3p6-
     . ffmcfmb*hbcm*p2p3*p3p6*p4p8-ffmcfmb*hbcm*p3p5*p3p6*p4p8+hbcm*
     . p1p3*p1p8*p5p6-hbcm*p1p3*p4p8*p5p6+hbcm*p1p8*p2p3*p4p6-hbcm*
     . p1p8*p3p4*p5p6+hbcm*p1p8*p3p5*p4p6-hbcm*p2p3*p4p6*p4p8+hbcm*
     . p3p4*p4p8*p5p6-hbcm*p3p5*p4p6*p4p8))+ans3
      ans1=w2*(2*(p5p8*p5p6)*(-ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2-2*
     . hbcm3)+8*(p5p8*p4p6)*(-ffmcfmb*hbcm3+hbcm3)+8*(p5p8*p3p6*p3p4)
     . *(ffmcfmb*hbcm-hbcm)+8*(p5p8*p3p6*p1p3)*(-ffmcfmb*hbcm+hbcm)+2
     . *(p5p8*p3p6)*(-3*ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2+ffmcfmb*
     . fmc*hbcm2+5*ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2-2*hbcm3)+8*(
     . ffmcfmb*hbcm*p2p3*p3p6*p5p8+ffmcfmb*hbcm*p3p5*p3p6*p5p8+hbcm*
     . p1p3*p5p6*p5p8+hbcm*p2p3*p4p6*p5p8-hbcm*p3p4*p5p6*p5p8+hbcm*
     . p3p5*p4p6*p5p8))+w1*(2*(p5p6*p4p8)*(ffmcfmb*hbcm3-2*fmb*hbcm2+
     . fmc*hbcm2+2*hbcm3)+8*(p4p8*p4p6)*(ffmcfmb*hbcm3-hbcm3)+8*(p4p8
     . *p3p6*p3p4)*(-ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p6*p1p3)*(ffmcfmb*
     . hbcm-hbcm)+2*(p4p8*p3p6)*(3*ffmcfmb**2*hbcm3+2*ffmcfmb*fmb*
     . hbcm2-ffmcfmb*fmc*hbcm2-5*ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2+
     . 2*hbcm3)+8*(-ffmcfmb*hbcm*p2p3*p3p6*p4p8-ffmcfmb*hbcm*p3p5*
     . p3p6*p4p8-hbcm*p1p3*p4p8*p5p6-hbcm*p2p3*p4p6*p4p8+hbcm*p3p4*
     . p4p8*p5p6-hbcm*p3p5*p4p6*p4p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(4*(p4p6*p3p7)*(4*fb3*fmb+12*fb4*ffmcfmb*hbcm-8*fb4*
     . hbcm-hbcm)+2*(p5p7*p3p6)*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*
     . hbcm+hbcm)+2*(p4p7*p3p6)*(8*fb3*fmb-4*fb3*fmc-12*fb4*ffmcfmb*
     . hbcm+8*fb4*hbcm-hbcm)+4*(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*
     . hbcm+hbcm)+16*(p6p7*p2p3)*(-fb3*fmb+fb3*fmc)+4*(p6p7*p1p3)*(4*
     . fb3*fmb+4*fb4*ffmcfmb*hbcm-hbcm)+2*(p3p7*p3p6)*(24*fb3*ffmcfmb
     . *fmb-8*fb3*ffmcfmb*fmc-8*fb3*fmb+4*fb3*fmc-12*fb4*ffmcfmb*hbcm
     . +8*fb4*hbcm-4*ffmcfmb*hbcm+hbcm)+p6p7*(-12*fb3*ffmcfmb*fmb*
     . hbcm2+4*fb3*ffmcfmb*fmc*hbcm2-4*fb3*fmc*hbcm2-4*fb4*ffmcfmb**2
     . *hbcm3-4*fb4*ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-8*fb4*fmc2*hbcm+
     . 2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+hbcm3)+16*(fb4*hbcm*p1p2*
     . p6p7+fb4*hbcm*p1p4*p6p7-fb4*hbcm*p2p4*p6p7+fb4*hbcm*p3p7*p5p6+
     . fb4*hbcm*p4p6*p4p7-2*fb4*hbcm*p4p6*p5p7+fb4*hbcm*p4p7*p5p6))
      b(9)=ans
      ans5=4*(p3p7*p3p6*p1p8)*(5*ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2)+8*(-hbcm2*p1p2*p1p8*p6p7+
     . hbcm2*p1p2*p4p8*p6p7-hbcm2*p1p4*p1p8*p6p7+hbcm2*p1p4*p4p8*p6p7
     . +hbcm2*p1p8*p2p3*p6p7+hbcm2*p1p8*p2p4*p6p7+hbcm2*p1p8*p4p6*
     . p5p7-hbcm2*p1p8*p4p7*p5p6-hbcm2*p2p3*p4p8*p6p7-hbcm2*p2p4*p4p8
     . *p6p7-hbcm2*p4p6*p4p8*p5p7+hbcm2*p4p7*p4p8*p5p6)
      ans4=4*(p5p6*p4p8*p3p7)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+8*(
     . p4p8*p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p5p7*p4p8*
     . p3p6)*(ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+4*(p4p8*p4p7*p3p6)*(
     . -5*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+8*(p6p7*p4p8*
     . p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(p5p6*p3p7*p1p8)*(-ffmcfmb*
     . hbcm2-2*fmb*hbcm-fmc*hbcm)+8*(p4p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-
     . fmb*hbcm+hbcm2)+4*(p5p7*p3p6*p1p8)*(-ffmcfmb*hbcm2+2*fmb*hbcm+
     . fmc*hbcm)+4*(p4p7*p3p6*p1p8)*(5*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*
     . hbcm-2*hbcm2)+8*(p6p7*p3p4*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(
     . p6p7*p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p1p8*p1p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm)+2*(p6p7*p4p8)*(-ffmcfmb**2*hbcm4-3*
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2+fmc*hbcm3-2*fmc2*hbcm2)+2*(p6p7*p1p8)*(ffmcfmb**2*hbcm4+
     . 3*ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2-fmc*hbcm3+2*fmc2*hbcm2)+4*(p4p8*p3p7*p3p6)*(-5*ffmcfmb**
     . 2*hbcm2+4*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2)+
     . ans5
      ans3=w5*ans4
      ans7=2*(p7p8*p3p6)*(4*fb3*ffmcfmb**2*hbcm2-12*fb3*ffmcfmb*hbcm2
     . +8*fb3*hbcm2+4*fb4*ffmcfmb*fmc*hbcm-8*fb4*fmb*hbcm-4*fb4*fmc*
     . hbcm+5*ffmcfmb*hbcm2-4*fmb*hbcm-2*fmc*hbcm-hbcm2)+32*(fb3*
     . ffmcfmb*p1p8*p3p6*p3p7-fb3*ffmcfmb*p2p3*p3p6*p7p8-fb3*ffmcfmb*
     . p3p5*p3p6*p7p8-fb3*ffmcfmb*p3p6*p3p7*p4p8+fb3*ffmcfmb*p3p6*
     . p3p7*p5p8+2*fb3*p1p3*p2p8*p6p7-fb3*p1p3*p4p7*p6p8-fb3*p1p3*
     . p5p6*p7p8+fb3*p1p3*p5p7*p6p8+fb3*p1p8*p3p6*p4p7-fb3*p1p8*p3p6*
     . p5p7+fb3*p1p8*p3p7*p5p6-fb3*p2p3*p4p6*p7p8-2*fb3*p2p8*p3p4*
     . p6p7+fb3*p2p8*p3p6*p4p7-fb3*p2p8*p3p6*p5p7+3*fb3*p2p8*p3p7*
     . p4p6+fb3*p3p4*p4p7*p6p8+fb3*p3p4*p5p6*p7p8-fb3*p3p4*p5p7*p6p8-
     . fb3*p3p5*p4p6*p7p8-fb3*p3p6*p4p7*p4p8+fb3*p3p6*p4p8*p5p7+fb3*
     . p3p7*p4p6*p5p8-fb3*p3p7*p4p8*p5p6)
      ans6=32*(p6p8*p3p7*p3p4)*(2*fb3*ffmcfmb-fb3)+32*(p7p8*p3p6*p3p4
     . )*(-fb3*ffmcfmb+fb3)+32*(p3p7*p3p6*p2p8)*(5*fb3*ffmcfmb-fb3)+
     . 32*(p6p8*p3p7*p1p3)*(-2*fb3*ffmcfmb+fb3)+32*(p7p8*p3p6*p1p3)*(
     . fb3*ffmcfmb-fb3)+2*(p6p8*p5p7)*(-4*fb3*ffmcfmb*hbcm2-8*fb3*
     . hbcm2+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm-hbcm2)+2*(p7p8*p5p6)*(4*
     . fb3*ffmcfmb*hbcm2+8*fb3*hbcm2-8*fb4*fmb*hbcm-4*fb4*fmc*hbcm-
     . hbcm2)+4*(p6p7*p4p8)*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*
     . fmb*hbcm+hbcm2)+2*(p6p8*p4p7)*(4*fb3*ffmcfmb*hbcm2+8*fb3*hbcm2
     . -8*fb4*fmb*hbcm-4*fb4*fmc*hbcm+5*hbcm2)+4*(p7p8*p4p6)*(4*fb3*
     . ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+16*(p6p7*p2p8)
     . *(-2*fb3*ffmcfmb*hbcm2-fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+4*
     . (p6p7*p1p8)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-
     . hbcm2)+2*(p6p8*p3p7)*(8*fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*
     . hbcm2-8*fb3*hbcm2-16*fb4*ffmcfmb*fmb*hbcm-8*fb4*ffmcfmb*fmc*
     . hbcm+8*fb4*fmb*hbcm+4*fb4*fmc*hbcm+5*ffmcfmb*hbcm2+fmc*hbcm-
     . hbcm2)+ans7
      ans2=w1*(4*(p5p6*p4p8*p3p7)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)
     . +8*(p4p8*p4p6*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p5p7*
     . p4p8*p3p6)*(ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm)+4*(p4p8*p4p7*
     . p3p6)*(-5*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+8*(p6p7*
     . p4p8*p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm)+8*(p6p7*p4p8*p1p3)*(
     . ffmcfmb*hbcm2+fmb*hbcm)+2*(p6p7*p4p8)*(-ffmcfmb**2*hbcm4-3*
     . ffmcfmb*fmb*hbcm3-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4-fmb*fmc*
     . hbcm2+fmc*hbcm3-2*fmc2*hbcm2)+4*(p4p8*p3p7*p3p6)*(-5*ffmcfmb**
     . 2*hbcm2+4*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2)+8
     . *(hbcm2*p1p2*p4p8*p6p7+hbcm2*p1p4*p4p8*p6p7-hbcm2*p2p3*p4p8*
     . p6p7-hbcm2*p2p4*p4p8*p6p7-hbcm2*p4p6*p4p8*p5p7+hbcm2*p4p7*p4p8
     . *p5p6))+ans3+ans6
      ans1=w2*(4*(p5p8*p5p6*p3p7)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm
     . )+8*(p5p8*p4p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p5p8*
     . p5p7*p3p6)*(-ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm)+4*(p5p8*p4p7*
     . p3p6)*(5*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-2*hbcm2)+8*(p6p7*
     . p5p8*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p5p8*p1p3)*(-
     . ffmcfmb*hbcm2-fmb*hbcm)+2*(p6p7*p5p8)*(ffmcfmb**2*hbcm4+3*
     . ffmcfmb*fmb*hbcm3+ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2-fmc*hbcm3+2*fmc2*hbcm2)+4*(p5p8*p3p7*p3p6)*(5*ffmcfmb**2
     . *hbcm2-4*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2)+8*
     . (-hbcm2*p1p2*p5p8*p6p7-hbcm2*p1p4*p5p8*p6p7+hbcm2*p2p3*p5p8*
     . p6p7+hbcm2*p2p4*p5p8*p6p7+hbcm2*p4p6*p5p7*p5p8-hbcm2*p4p7*p5p6
     . *p5p8))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w2*(2*(p5p8*p3p4)*(ffmcfmb*hbcm-hbcm)+2*(p5p8*p1p3)*(
     . ffmcfmb*hbcm-hbcm)+2*p5p8*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm3)+2*(
     . -ffmcfmb*hbcm*p2p3*p5p8-ffmcfmb*hbcm*p3p5*p5p8-hbcm*p1p2*p5p8-
     . hbcm*p1p5*p5p8-hbcm*p2p4*p5p8-hbcm*p4p5*p5p8))+w1*(2*(p4p8*
     . p3p4)*(-ffmcfmb*hbcm+hbcm)+2*(p4p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+
     . 2*p4p8*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3)+2*(ffmcfmb*hbcm*p2p3*
     . p4p8+ffmcfmb*hbcm*p3p5*p4p8+hbcm*p1p2*p4p8+hbcm*p1p5*p4p8+hbcm
     . *p2p4*p4p8+hbcm*p4p5*p4p8))+w5*(2*(p4p8*p3p4)*(-ffmcfmb*hbcm+
     . hbcm)+2*(p3p4*p1p8)*(ffmcfmb*hbcm-hbcm)+2*(p4p8*p1p3)*(-
     . ffmcfmb*hbcm+hbcm)+2*(p1p8*p1p3)*(ffmcfmb*hbcm-hbcm)+2*p4p8*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*hbcm3)+2*p1p8*(ffmcfmb**2*hbcm3-
     . ffmcfmb*hbcm3)+2*(-ffmcfmb*hbcm*p1p8*p2p3-ffmcfmb*hbcm*p1p8*
     . p3p5+ffmcfmb*hbcm*p2p3*p4p8+ffmcfmb*hbcm*p3p5*p4p8-hbcm*p1p2*
     . p1p8+hbcm*p1p2*p4p8-hbcm*p1p5*p1p8+hbcm*p1p5*p4p8-hbcm*p1p8*
     . p2p4-hbcm*p1p8*p4p5+hbcm*p2p4*p4p8+hbcm*p4p5*p4p8))+(hbcm*p1p8
     . -hbcm*p2p8+hbcm*p4p8-hbcm*p5p8))
      b(11)=ans
      b(12)=ccc*(4*p3p4*(-fb3*ffmcfmb+fb3)+4*p1p3*(-fb3*ffmcfmb+fb3)+
     . 4*(-fb3*ffmcfmb**2*hbcm2+fb3*ffmcfmb*hbcm2+fb3*ffmcfmb*p2p3+
     . fb3*ffmcfmb*p3p5+fb3*p1p2+fb3*p1p5+fb3*p2p4+fb3*p4p5))
      ans=ccc*(w2*(2*(p5p8*p3p4)*(ffmcfmb*hbcm2-hbcm2)+2*(p5p8*p1p3)*
     . (ffmcfmb*hbcm2-hbcm2)+2*p5p8*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4)+
     . 2*(-ffmcfmb*hbcm2*p2p3*p5p8-ffmcfmb*hbcm2*p3p5*p5p8-hbcm2*p1p2
     . *p5p8-hbcm2*p1p5*p5p8-hbcm2*p2p4*p5p8-hbcm2*p4p5*p5p8))+w1*(2*
     . (p4p8*p3p4)*(-ffmcfmb*hbcm2+hbcm2)+2*(p4p8*p1p3)*(-ffmcfmb*
     . hbcm2+hbcm2)+2*p4p8*(-ffmcfmb**2*hbcm4+ffmcfmb*hbcm4)+2*(
     . ffmcfmb*hbcm2*p2p3*p4p8+ffmcfmb*hbcm2*p3p5*p4p8+hbcm2*p1p2*
     . p4p8+hbcm2*p1p5*p4p8+hbcm2*p2p4*p4p8+hbcm2*p4p5*p4p8))+w5*(2*(
     . p4p8*p3p4)*(-ffmcfmb*hbcm2+hbcm2)+2*(p3p4*p1p8)*(ffmcfmb*hbcm2
     . -hbcm2)+2*(p4p8*p1p3)*(-ffmcfmb*hbcm2+hbcm2)+2*(p1p8*p1p3)*(
     . ffmcfmb*hbcm2-hbcm2)+2*p4p8*(-ffmcfmb**2*hbcm4+ffmcfmb*hbcm4)+
     . 2*p1p8*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4)+2*(-ffmcfmb*hbcm2*p1p8
     . *p2p3-ffmcfmb*hbcm2*p1p8*p3p5+ffmcfmb*hbcm2*p2p3*p4p8+ffmcfmb*
     . hbcm2*p3p5*p4p8-hbcm2*p1p2*p1p8+hbcm2*p1p2*p4p8-hbcm2*p1p5*
     . p1p8+hbcm2*p1p5*p4p8-hbcm2*p1p8*p2p4-hbcm2*p1p8*p4p5+hbcm2*
     . p2p4*p4p8+hbcm2*p4p5*p4p8))+(hbcm2*p1p8-hbcm2*p2p8+hbcm2*p4p8-
     . hbcm2*p5p8))
      b(13)=ans
      b(14)=ccc*(4*p3p4*(fb4*ffmcfmb*hbcm-fb4*hbcm)+4*p1p3*(fb4*
     . ffmcfmb*hbcm-fb4*hbcm)+4*(fb4*ffmcfmb**2*hbcm3-fb4*ffmcfmb*
     . hbcm*p2p3-fb4*ffmcfmb*hbcm*p3p5-fb4*ffmcfmb*hbcm3-fb4*hbcm*
     . p1p2-fb4*hbcm*p1p5-fb4*hbcm*p2p4-fb4*hbcm*p4p5))
      b(15)=ccc*(w2*(4*(p5p8*p3p6)*(4*ffmcfmb*hbcm2-hbcm2)+4*(3*hbcm2
     . *p4p6*p5p8-hbcm2*p5p6*p5p8))+w1*(4*(p4p8*p3p6)*(-4*ffmcfmb*
     . hbcm2+hbcm2)+4*(-3*hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+w5*(4*(
     . p4p8*p3p6)*(-4*ffmcfmb*hbcm2+hbcm2)+4*(p3p6*p1p8)*(4*ffmcfmb*
     . hbcm2-hbcm2)+4*(3*hbcm2*p1p8*p4p6-hbcm2*p1p8*p5p6-3*hbcm2*p4p6
     . *p4p8+hbcm2*p4p8*p5p6))+8*hbcm2*p6p8)
      b(16)=ccc*(w2*(4*(p5p8*p3p6)*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*
     . p4p6*p5p8+hbcm*p5p6*p5p8))+w1*(4*(p4p8*p3p6)*(4*ffmcfmb*hbcm-
     . hbcm)+4*(3*hbcm*p4p6*p4p8-hbcm*p4p8*p5p6))+w5*(4*(p4p8*p3p6)*(
     . 4*ffmcfmb*hbcm-hbcm)+4*(p3p6*p1p8)*(-4*ffmcfmb*hbcm+hbcm)+4*(-
     . 3*hbcm*p1p8*p4p6+hbcm*p1p8*p5p6+3*hbcm*p4p6*p4p8-hbcm*p4p8*
     . p5p6))-8*hbcm*p6p8)
      b(18)=ccc*(w2*(2*(p5p8*p3p7)*(8*ffmcfmb*hbcm2-3*hbcm2)+2*(5*
     . hbcm2*p4p7*p5p8-3*hbcm2*p5p7*p5p8))+w1*(2*(p4p8*p3p7)*(-8*
     . ffmcfmb*hbcm2+3*hbcm2)+2*(-5*hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7
     . ))+w5*(2*(p4p8*p3p7)*(-8*ffmcfmb*hbcm2+3*hbcm2)+2*(p3p7*p1p8)*
     . (8*ffmcfmb*hbcm2-3*hbcm2)+2*(5*hbcm2*p1p8*p4p7-3*hbcm2*p1p8*
     . p5p7-5*hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+8*hbcm2*p7p8)
      b(20)=ccc*(w2*(2*(p5p8*p3p7)*(8*ffmcfmb*hbcm-3*hbcm)+2*(5*hbcm*
     . p4p7*p5p8-3*hbcm*p5p7*p5p8))+w1*(2*(p4p8*p3p7)*(-8*ffmcfmb*
     . hbcm+3*hbcm)+2*(-5*hbcm*p4p7*p4p8+3*hbcm*p4p8*p5p7))+w5*(2*(
     . p4p8*p3p7)*(-8*ffmcfmb*hbcm+3*hbcm)+2*(p3p7*p1p8)*(8*ffmcfmb*
     . hbcm-3*hbcm)+2*(5*hbcm*p1p8*p4p7-3*hbcm*p1p8*p5p7-5*hbcm*p4p7*
     . p4p8+3*hbcm*p4p8*p5p7))+8*hbcm*p7p8)
      b(23)=ccc*(4*p3p7*(8*fb4*ffmcfmb*hbcm-3*fb4*hbcm)+4*(5*fb4*hbcm
     . *p4p7-3*fb4*hbcm*p5p7))
      b(24)=ccc*(8*p3p6*(-4*fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(-3*fb4*hbcm
     . *p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(12*p5p7*(fb3*fmb-fb3*fmc-fb4*hbcm)+p4p7*(-16*fb3*fmb
     . +12*fb3*fmc-4*fb4*ffmcfmb*hbcm+16*fb4*hbcm+hbcm)+p3p7*(-28*fb3
     . *ffmcfmb*fmb+24*fb3*ffmcfmb*fmc+12*fb3*fmb-12*fb3*fmc-4*fb4*
     . ffmcfmb**2*hbcm+28*fb4*ffmcfmb*hbcm-12*fb4*hbcm+ffmcfmb*hbcm))
      b(26)=ccc*(p5p6*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+8*fb4*
     . hbcm+hbcm)+p3p6*(8*fb3*ffmcfmb*fmb-4*fb3*ffmcfmb*fmc-8*fb3*fmb
     . +4*fb3*fmc+4*fb4*ffmcfmb**2*hbcm-12*fb4*ffmcfmb*hbcm+8*fb4*
     . hbcm-ffmcfmb*hbcm+hbcm))
      b(27)=ccc*(16*(p3p7*p3p6)*(-4*fb3*ffmcfmb+fb3)+8*p6p7*(2*fb3*
     . ffmcfmb*hbcm2+fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*(-2*fb3*
     . p1p3*p6p7+2*fb3*p3p4*p6p7-fb3*p3p6*p4p7+fb3*p3p6*p5p7-2*fb3*
     . p3p7*p4p6))
      b(28)=ccc*(4*w2*(p6p7*p5p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+4*w1*(
     . p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+w5*(4*(p6p7*p4p8)*(-fmb*
     . hbcm-fmc*hbcm+hbcm2)+4*(p6p7*p1p8)*(fmb*hbcm+fmc*hbcm-hbcm2))+
     . (16*(p6p8*p3p7)*(-2*fb3*ffmcfmb+fb3)+16*(-2*fb3*ffmcfmb*p3p6*
     . p7p8-2*fb3*p1p8*p6p7-2*fb3*p4p6*p7p8-fb3*p4p7*p6p8+2*fb3*p4p8*
     . p6p7+fb3*p5p7*p6p8)))
      b(29)=ccc*(8*p3p6*(-4*fb3*ffmcfmb+fb3)+8*(-3*fb3*p4p6+fb3*p5p6)
     . )
      b(30)=ccc*(4*p3p7*(8*fb3*ffmcfmb-3*fb3)+4*(5*fb3*p4p7-3*fb3*
     . p5p7))
      ans2=(8*(p6p8*p3p7)*(-4*fb4*ffmcfmb*hbcm+2*fb4*hbcm-hbcm)+8*(
     . p7p8*p3p6)*(-4*fb4*ffmcfmb*hbcm-hbcm)+16*(-2*fb4*hbcm*p1p8*
     . p6p7-2*fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*p6p8+2*fb4*hbcm*p4p8*
     . p6p7+fb4*hbcm*p5p7*p6p8))
      ans1=w2*(4*(p6p7*p5p8)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+
     . hbcm3)+8*(p5p8*p3p7*p3p6)*(-4*ffmcfmb*hbcm+hbcm)+8*(-2*hbcm*
     . p1p3*p5p8*p6p7+2*hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8+hbcm*
     . p3p6*p5p7*p5p8-2*hbcm*p3p7*p4p6*p5p8))+w1*(4*(p6p7*p4p8)*(-2*
     . ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2-hbcm3)+8*(p4p8*p3p7*p3p6)*(4
     . *ffmcfmb*hbcm-hbcm)+8*(2*hbcm*p1p3*p4p8*p6p7-2*hbcm*p3p4*p4p8*
     . p6p7+hbcm*p3p6*p4p7*p4p8-hbcm*p3p6*p4p8*p5p7+2*hbcm*p3p7*p4p6*
     . p4p8))+w5*(4*(p6p7*p4p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2
     . -hbcm3)+4*(p6p7*p1p8)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+
     . hbcm3)+8*(p4p8*p3p7*p3p6)*(4*ffmcfmb*hbcm-hbcm)+8*(p3p7*p3p6*
     . p1p8)*(-4*ffmcfmb*hbcm+hbcm)+8*(-2*hbcm*p1p3*p1p8*p6p7+2*hbcm*
     . p1p3*p4p8*p6p7+2*hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p4p7+hbcm*
     . p1p8*p3p6*p5p7-2*hbcm*p1p8*p3p7*p4p6-2*hbcm*p3p4*p4p8*p6p7+
     . hbcm*p3p6*p4p7*p4p8-hbcm*p3p6*p4p8*p5p7+2*hbcm*p3p7*p4p6*p4p8)
     . )+ans2
      ans=ccc*ans1
      b(31)=ans
      b(32)=8*ccc*p6p7*(fb3*fmb-fb3*fmc-fb4*hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
	end 
 
      subroutine amp35_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+
     . fmc2-2*p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2))
      b(1)=ccc*(8*(p6p7*p3p4)*(-fb3*ffmcfmb+fb3)+8*(p6p7*p2p3)*(-fb3*
     . ffmcfmb+fb3)+p6p7*(-4*fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*
     . hbcm2-4*fb3*fmb*fmc-8*fb3*fmc2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*
     . ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+fmb*hbcm+fmc*hbcm-hbcm2)+8*(
     . fb3*ffmcfmb*p1p3*p6p7+fb3*ffmcfmb*p3p5*p6p7+2*fb3*ffmcfmb*p3p6
     . *p3p7+6*fb3*ffmcfmb*p3p7*p4p6-2*fb3*ffmcfmb*p3p7*p5p6+fb3*p1p2
     . *p6p7+fb3*p1p4*p6p7+4*fb3*p2p4*p6p7+fb3*p2p5*p6p7+fb3*p3p6*
     . p4p7+fb3*p4p5*p6p7+5*fb3*p4p6*p4p7-3*fb3*p4p7*p5p6))
      ans4=(4*(p4p8*p4p7*p3p6)*(4*ffmcfmb*hbcm-3*hbcm)+4*(p6p7*p4p8*
     . p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p4p7*p3p6*p2p8)*(-4*ffmcfmb*hbcm+
     . 3*hbcm)+4*(p6p7*p3p4*p2p8)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p4p8*
     . p2p3)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p2p8*p2p3)*(-ffmcfmb*hbcm+
     . hbcm)+2*(p6p7*p4p8)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*
     . fmc2*hbcm)+2*(p6p7*p2p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*
     . fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+
     . 16*(p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+4*(ffmcfmb
     . *hbcm*p1p3*p2p8*p6p7-ffmcfmb*hbcm*p1p3*p4p8*p6p7+ffmcfmb*hbcm*
     . p2p8*p3p5*p6p7+4*ffmcfmb*hbcm*p2p8*p3p7*p4p6-ffmcfmb*hbcm*p3p5
     . *p4p8*p6p7-4*ffmcfmb*hbcm*p3p7*p4p6*p4p8+hbcm*p1p2*p2p8*p6p7-
     . hbcm*p1p2*p4p8*p6p7+hbcm*p1p4*p2p8*p6p7-hbcm*p1p4*p4p8*p6p7+4*
     . hbcm*p2p4*p2p8*p6p7-4*hbcm*p2p4*p4p8*p6p7+hbcm*p2p5*p2p8*p6p7-
     . hbcm*p2p5*p4p8*p6p7+hbcm*p2p8*p4p5*p6p7+3*hbcm*p2p8*p4p6*p4p7-
     . hbcm*p2p8*p4p7*p5p6-hbcm*p4p5*p4p8*p6p7-3*hbcm*p4p6*p4p7*p4p8+
     . hbcm*p4p7*p4p8*p5p6))
      ans3=w11*ans4
      ans5=(2*(p6p8*p5p7)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*(
     . p7p8*p5p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+2*(p6p7*p4p8)*(
     . 8*fb3*fmc+8*fb4*ffmcfmb*hbcm-3*hbcm)+2*(p6p7*p2p8)*(-8*fb3*fmc
     . -8*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p8*p4p7)*(-8*fb3*fmb+4*fb3*fmc-
     . 4*fb4*ffmcfmb*hbcm+8*fb4*hbcm-hbcm)+8*(p7p8*p4p6)*(-fb3*fmb+
     . fb3*fmc+fb4*hbcm+hbcm)+2*(p6p8*p3p7)*(-16*fb3*ffmcfmb*fmb+12*
     . fb3*ffmcfmb*fmc-4*fb3*fmc-4*fb4*ffmcfmb**2*hbcm+12*fb4*ffmcfmb
     . *hbcm-3*ffmcfmb*hbcm+hbcm)+2*(p7p8*p3p6)*(-4*fb3*ffmcfmb*fmb+4
     . *fb3*fmc-4*fb4*ffmcfmb**2*hbcm+8*fb4*ffmcfmb*hbcm-3*ffmcfmb*
     . hbcm+3*hbcm)+2*(hbcm*p1p8*p6p7+hbcm*p5p8*p6p7))
      ans2=w1*(4*(p4p8*p4p7*p3p6)*(4*ffmcfmb*hbcm-3*hbcm)+4*(p6p7*
     . p4p8*p3p4)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p4p8*p2p3)*(ffmcfmb*
     . hbcm-hbcm)+2*(p6p7*p4p8)*(ffmcfmb**2*hbcm3+ffmcfmb*fmb*hbcm2-
     . ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmb*fmc*hbcm+fmc*hbcm2+2*
     . fmc2*hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+
     . 4*(-ffmcfmb*hbcm*p1p3*p4p8*p6p7-ffmcfmb*hbcm*p3p5*p4p8*p6p7-4*
     . ffmcfmb*hbcm*p3p7*p4p6*p4p8-hbcm*p1p2*p4p8*p6p7-hbcm*p1p4*p4p8
     . *p6p7-4*hbcm*p2p4*p4p8*p6p7-hbcm*p2p5*p4p8*p6p7-hbcm*p4p5*p4p8
     . *p6p7-3*hbcm*p4p6*p4p7*p4p8+hbcm*p4p7*p4p8*p5p6))+ans3+ans5
      ans1=w2*(4*(p5p8*p4p7*p3p6)*(-4*ffmcfmb*hbcm+3*hbcm)+4*(p6p7*
     . p5p8*p3p4)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p5p8*p2p3)*(-ffmcfmb*
     . hbcm+hbcm)+2*(p6p7*p5p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+
     . ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+fmb*fmc*hbcm-fmc*hbcm2-2*
     . fmc2*hbcm)+16*(p5p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)
     . +4*(ffmcfmb*hbcm*p1p3*p5p8*p6p7+ffmcfmb*hbcm*p3p5*p5p8*p6p7+4*
     . ffmcfmb*hbcm*p3p7*p4p6*p5p8+hbcm*p1p2*p5p8*p6p7+hbcm*p1p4*p5p8
     . *p6p7+4*hbcm*p2p4*p5p8*p6p7+hbcm*p2p5*p5p8*p6p7+hbcm*p4p5*p5p8
     . *p6p7+3*hbcm*p4p6*p4p7*p5p8-hbcm*p4p7*p5p6*p5p8))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(8*(p7p8*p3p4)*(-fb3*ffmcfmb+fb3)+8*(p7p8*p2p3)*(-fb3*
     . ffmcfmb+fb3)+p7p8*(-8*fb3*ffmcfmb**2*hbcm2+8*fb3*ffmcfmb*hbcm2
     . -ffmcfmb*hbcm2+6*fmb*hbcm+7*fmc*hbcm-6*hbcm2)+8*(fb3*ffmcfmb*
     . p1p3*p7p8+2*fb3*ffmcfmb*p1p8*p3p7-2*fb3*ffmcfmb*p2p8*p3p7+fb3*
     . ffmcfmb*p3p5*p7p8+2*fb3*ffmcfmb*p3p7*p4p8-6*fb3*ffmcfmb*p3p7*
     . p5p8+fb3*p1p2*p7p8+fb3*p1p4*p7p8+2*fb3*p1p8*p4p7+fb3*p2p5*p7p8
     . -2*fb3*p2p8*p4p7+fb3*p4p5*p7p8+2*fb3*p4p7*p4p8-6*fb3*p4p7*p5p8
     . ))
      ans1=w2*(12*(p5p8*p4p7)*(fmb*hbcm+fmc*hbcm-hbcm2)+2*(p5p8*p5p7)
     . *(ffmcfmb*hbcm2-fmc*hbcm)+2*(p5p8*p3p7)*(-ffmcfmb**2*hbcm2+6*
     . ffmcfmb*fmb*hbcm+7*ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2-fmc*hbcm))
     . +w1*(12*(p4p8*p4p7)*(-fmb*hbcm-fmc*hbcm+hbcm2)+2*(p5p7*p4p8)*(
     . -ffmcfmb*hbcm2+fmc*hbcm)+2*(p4p8*p3p7)*(ffmcfmb**2*hbcm2-6*
     . ffmcfmb*fmb*hbcm-7*ffmcfmb*fmc*hbcm+5*ffmcfmb*hbcm2+fmc*hbcm))
     . +w11*(12*(p4p8*p4p7)*(-fmb*hbcm-fmc*hbcm+hbcm2)+12*(p4p7*p2p8)
     . *(fmb*hbcm+fmc*hbcm-hbcm2)+2*(p5p7*p4p8)*(-ffmcfmb*hbcm2+fmc*
     . hbcm)+2*(p5p7*p2p8)*(ffmcfmb*hbcm2-fmc*hbcm)+2*(p4p8*p3p7)*(
     . ffmcfmb**2*hbcm2-6*ffmcfmb*fmb*hbcm-7*ffmcfmb*fmc*hbcm+5*
     . ffmcfmb*hbcm2+fmc*hbcm)+2*(p3p7*p2p8)*(-ffmcfmb**2*hbcm2+6*
     . ffmcfmb*fmb*hbcm+7*ffmcfmb*fmc*hbcm-5*ffmcfmb*hbcm2-fmc*hbcm))
     . +ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=ccc*(8*(p3p7*p3p4)*(-fb3*ffmcfmb-fb3)+8*(p3p7*p2p3)*(-5*
     . fb3*ffmcfmb+fb3)+p5p7*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm+
     . hbcm2)+8*p4p7*(-2*fb3*ffmcfmb*hbcm2+3*fb3*hbcm2+3*fb4*fmb*hbcm
     . +3*fb4*fmc*hbcm)+p3p7*(-4*fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*
     . hbcm2+24*fb4*ffmcfmb*fmb*hbcm+28*fb4*ffmcfmb*fmc*hbcm-4*fb4*
     . fmc*hbcm-ffmcfmb*hbcm2+hbcm2)+8*(-3*fb3*ffmcfmb*p1p3*p3p7+5*
     . fb3*ffmcfmb*p3p5*p3p7-fb3*p1p2*p3p7-2*fb3*p1p3*p4p7-fb3*p1p4*
     . p3p7-4*fb3*p2p3*p4p7+2*fb3*p2p3*p5p7-fb3*p2p5*p3p7-2*fb3*p3p4*
     . p4p7+6*fb3*p3p5*p4p7-fb3*p3p7*p4p5))
      ans5=4*(-3*ffmcfmb*hbcm*p1p3*p2p8*p3p7+3*ffmcfmb*hbcm*p1p3*p3p7
     . *p4p8+5*ffmcfmb*hbcm*p2p8*p3p5*p3p7-5*ffmcfmb*hbcm*p3p5*p3p7*
     . p4p8-hbcm*p1p2*p2p8*p3p7+hbcm*p1p2*p3p7*p4p8-2*hbcm*p1p3*p2p8*
     . p4p7+2*hbcm*p1p3*p4p7*p4p8-hbcm*p1p4*p2p8*p3p7+hbcm*p1p4*p3p7*
     . p4p8-4*hbcm*p2p3*p2p8*p4p7+2*hbcm*p2p3*p2p8*p5p7+4*hbcm*p2p3*
     . p4p7*p4p8-2*hbcm*p2p3*p4p8*p5p7-hbcm*p2p5*p2p8*p3p7+hbcm*p2p5*
     . p3p7*p4p8-2*hbcm*p2p8*p3p4*p4p7+6*hbcm*p2p8*p3p5*p4p7-hbcm*
     . p2p8*p3p7*p4p5+2*hbcm*p3p4*p4p7*p4p8-6*hbcm*p3p5*p4p7*p4p8+
     . hbcm*p3p7*p4p5*p4p8)
      ans4=2*(p5p7*p4p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p4p8*p4p7)*(2*
     . ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-3*hbcm3)+2*(p5p7*p2p8)*(
     . -ffmcfmb*hbcm3+fmc*hbcm2)+4*(p4p7*p2p8)*(-2*ffmcfmb*hbcm3+3*
     . fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+4*(p4p8*p3p7*p3p4)*(ffmcfmb*
     . hbcm+hbcm)+4*(p3p7*p3p4*p2p8)*(-ffmcfmb*hbcm-hbcm)+4*(p4p8*
     . p3p7*p2p3)*(5*ffmcfmb*hbcm-hbcm)+4*(p3p7*p2p8*p2p3)*(-5*
     . ffmcfmb*hbcm+hbcm)+2*(p4p8*p3p7)*(ffmcfmb**2*hbcm3-6*ffmcfmb*
     . fmb*hbcm2+7*ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmc*hbcm2)+2*(
     . p3p7*p2p8)*(-ffmcfmb**2*hbcm3+6*ffmcfmb*fmb*hbcm2-7*ffmcfmb*
     . fmc*hbcm2+3*ffmcfmb*hbcm3+fmc*hbcm2)+ans5
      ans3=w11*ans4
      ans6=(2*(p5p8*p3p7)*(24*fb4*ffmcfmb*hbcm-hbcm)+2*(p4p8*p3p7)*(-
     . 8*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8*p3p5)*(-2*fb4*ffmcfmb*hbcm+3*
     . hbcm)+4*(p7p8*p3p4)*(2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm)+2*(
     . p3p7*p2p8)*(-24*fb4*ffmcfmb*hbcm+8*fb4*hbcm+hbcm)+4*(p7p8*p2p3
     . )*(2*fb4*ffmcfmb*hbcm-2*fb4*hbcm-3*hbcm)+2*(p3p7*p1p8)*(-8*fb4
     . *ffmcfmb*hbcm-hbcm)+4*(p7p8*p1p3)*(-2*fb4*ffmcfmb*hbcm-hbcm)+
     . p7p8*(8*fb4*ffmcfmb**2*hbcm3-8*fb4*ffmcfmb*hbcm3-3*ffmcfmb*
     . hbcm3+6*fmb*hbcm2-7*fmc*hbcm2+6*hbcm3)+8*(-fb4*hbcm*p1p2*p7p8-
     . fb4*hbcm*p1p4*p7p8-2*fb4*hbcm*p1p8*p4p7-fb4*hbcm*p2p5*p7p8-4*
     . fb4*hbcm*p2p8*p4p7+2*fb4*hbcm*p2p8*p5p7-fb4*hbcm*p4p5*p7p8-2*
     . fb4*hbcm*p4p7*p4p8+6*fb4*hbcm*p4p7*p5p8))
      ans2=w1*(2*(p5p7*p4p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p4p8*p4p7)*
     . (2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-3*hbcm3)+4*(p4p8*p3p7
     . *p3p4)*(ffmcfmb*hbcm+hbcm)+4*(p4p8*p3p7*p2p3)*(5*ffmcfmb*hbcm-
     . hbcm)+2*(p4p8*p3p7)*(ffmcfmb**2*hbcm3-6*ffmcfmb*fmb*hbcm2+7*
     . ffmcfmb*fmc*hbcm2-3*ffmcfmb*hbcm3-fmc*hbcm2)+4*(3*ffmcfmb*hbcm
     . *p1p3*p3p7*p4p8-5*ffmcfmb*hbcm*p3p5*p3p7*p4p8+hbcm*p1p2*p3p7*
     . p4p8+2*hbcm*p1p3*p4p7*p4p8+hbcm*p1p4*p3p7*p4p8+4*hbcm*p2p3*
     . p4p7*p4p8-2*hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p3p7*p4p8+2*hbcm*
     . p3p4*p4p7*p4p8-6*hbcm*p3p5*p4p7*p4p8+hbcm*p3p7*p4p5*p4p8))+
     . ans3+ans6
      ans1=w2*(2*(p5p8*p5p7)*(-ffmcfmb*hbcm3+fmc*hbcm2)+4*(p5p8*p4p7)
     . *(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2+3*hbcm3)+4*(p5p8*
     . p3p7*p3p4)*(-ffmcfmb*hbcm-hbcm)+4*(p5p8*p3p7*p2p3)*(-5*ffmcfmb
     . *hbcm+hbcm)+2*(p5p8*p3p7)*(-ffmcfmb**2*hbcm3+6*ffmcfmb*fmb*
     . hbcm2-7*ffmcfmb*fmc*hbcm2+3*ffmcfmb*hbcm3+fmc*hbcm2)+4*(-3*
     . ffmcfmb*hbcm*p1p3*p3p7*p5p8+5*ffmcfmb*hbcm*p3p5*p3p7*p5p8-hbcm
     . *p1p2*p3p7*p5p8-2*hbcm*p1p3*p4p7*p5p8-hbcm*p1p4*p3p7*p5p8-4*
     . hbcm*p2p3*p4p7*p5p8+2*hbcm*p2p3*p5p7*p5p8-hbcm*p2p5*p3p7*p5p8-
     . 2*hbcm*p3p4*p4p7*p5p8+6*hbcm*p3p5*p4p7*p5p8-hbcm*p3p7*p4p5*
     . p5p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w2*(2*(p5p8*p4p6)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+
     . hbcm2)+2*(p5p8*p3p6)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm-2*
     . ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2))+w1*(2*(p4p8*p4p6)*(-ffmcfmb*
     . hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+2*(p4p8*p3p6)*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2))+w11*
     . (2*(p4p8*p4p6)*(-ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-hbcm2)+2*(
     . p4p6*p2p8)*(ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+hbcm2)+2*(p4p8*
     . p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-
     . ffmcfmb*hbcm2)+2*(p3p6*p2p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmb*
     . hbcm-2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2))+(8*(p6p8*p3p4)*(-fb3*
     . ffmcfmb+fb3)+16*(p3p6*p2p8)*(-2*fb3*ffmcfmb+fb3)+8*(p6p8*p2p3)
     . *(-fb3*ffmcfmb+fb3)+p6p8*(-8*fb3*ffmcfmb**2*hbcm2+8*fb3*
     . ffmcfmb*hbcm2+ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+hbcm2)+8*(fb3*
     . ffmcfmb*p1p3*p6p8+fb3*ffmcfmb*p3p5*p6p8+fb3*p1p2*p6p8+fb3*p1p4
     . *p6p8+fb3*p2p5*p6p8-2*fb3*p2p8*p4p6+2*fb3*p2p8*p5p6-fb3*p3p6*
     . p4p8+fb3*p4p5*p6p8-fb3*p4p6*p4p8-fb3*p4p8*p5p6)))
      b(6)=ans
      b(7)=ccc*(8*(p3p6*p2p3)*(5*fb3*ffmcfmb-3*fb3)+p4p6*(-4*fb3*
     . ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-8*fb4*fmc*hbcm+hbcm2)
     . +p3p6*(-4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2-4*fb4*
     . ffmcfmb*fmb*hbcm-8*fb4*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2)+8*(fb3*
     . ffmcfmb*hbcm2*p5p6-fb3*ffmcfmb*p1p3*p3p6+fb3*ffmcfmb*p3p4*p3p6
     . -fb3*ffmcfmb*p3p5*p3p6-fb3*p1p2*p3p6-fb3*p1p4*p3p6+2*fb3*p2p3*
     . p4p6-2*fb3*p2p3*p5p6-fb3*p2p5*p3p6+fb3*p3p4*p4p6+fb3*p3p4*p5p6
     . -fb3*p3p6*p4p5))
      ans3=(2*(p4p8*p3p6)*(4*fb4*hbcm+hbcm)+8*(p6p8*p3p4)*(fb4*
     . ffmcfmb*hbcm-fb4*hbcm)+2*(p3p6*p2p8)*(16*fb4*ffmcfmb*hbcm-8*
     . fb4*hbcm+hbcm)+8*(p6p8*p2p3)*(fb4*ffmcfmb*hbcm-fb4*hbcm+hbcm)+
     . p6p8*(8*fb4*ffmcfmb**2*hbcm3-8*fb4*ffmcfmb*hbcm3-3*ffmcfmb*
     . hbcm3-fmb*hbcm2+2*fmc*hbcm2+hbcm3)+2*(-4*fb4*ffmcfmb*hbcm*p1p3
     . *p6p8-4*fb4*ffmcfmb*hbcm*p3p5*p6p8-4*fb4*hbcm*p1p2*p6p8-4*fb4*
     . hbcm*p1p4*p6p8-4*fb4*hbcm*p2p5*p6p8+8*fb4*hbcm*p2p8*p4p6-8*fb4
     . *hbcm*p2p8*p5p6-4*fb4*hbcm*p4p5*p6p8+4*fb4*hbcm*p4p6*p4p8+4*
     . fb4*hbcm*p4p8*p5p6-hbcm*p1p8*p3p6-hbcm*p3p6*p5p8))
      ans2=w11*(2*(p4p8*p4p6)*(ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2-
     . hbcm3)+2*(p4p6*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+2*fmc*hbcm2+
     . hbcm3)+4*(p4p8*p3p6*p2p3)*(-5*ffmcfmb*hbcm+3*hbcm)+4*(p3p6*
     . p2p8*p2p3)*(5*ffmcfmb*hbcm-3*hbcm)+2*(p4p8*p3p6)*(ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3)+2*(
     . p3p6*p2p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*
     . hbcm2+ffmcfmb*hbcm3)+4*(-ffmcfmb*hbcm*p1p3*p2p8*p3p6+ffmcfmb*
     . hbcm*p1p3*p3p6*p4p8+ffmcfmb*hbcm*p2p8*p3p4*p3p6-ffmcfmb*hbcm*
     . p2p8*p3p5*p3p6-ffmcfmb*hbcm*p3p4*p3p6*p4p8+ffmcfmb*hbcm*p3p5*
     . p3p6*p4p8+ffmcfmb*hbcm3*p2p8*p5p6-ffmcfmb*hbcm3*p4p8*p5p6-hbcm
     . *p1p2*p2p8*p3p6+hbcm*p1p2*p3p6*p4p8-hbcm*p1p4*p2p8*p3p6+hbcm*
     . p1p4*p3p6*p4p8+2*hbcm*p2p3*p2p8*p4p6-2*hbcm*p2p3*p2p8*p5p6-2*
     . hbcm*p2p3*p4p6*p4p8+2*hbcm*p2p3*p4p8*p5p6-hbcm*p2p5*p2p8*p3p6+
     . hbcm*p2p5*p3p6*p4p8+hbcm*p2p8*p3p4*p4p6+hbcm*p2p8*p3p4*p5p6-
     . hbcm*p2p8*p3p6*p4p5-hbcm*p3p4*p4p6*p4p8-hbcm*p3p4*p4p8*p5p6+
     . hbcm*p3p6*p4p5*p4p8))+ans3
      ans1=w2*(2*(p5p8*p4p6)*(-ffmcfmb*hbcm3-fmb*hbcm2+2*fmc*hbcm2+
     . hbcm3)+4*(p5p8*p3p6*p2p3)*(5*ffmcfmb*hbcm-3*hbcm)+2*(p5p8*p3p6
     . )*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*fmc*hbcm2+
     . ffmcfmb*hbcm3)+4*(-ffmcfmb*hbcm*p1p3*p3p6*p5p8+ffmcfmb*hbcm*
     . p3p4*p3p6*p5p8-ffmcfmb*hbcm*p3p5*p3p6*p5p8+ffmcfmb*hbcm3*p5p6*
     . p5p8-hbcm*p1p2*p3p6*p5p8-hbcm*p1p4*p3p6*p5p8+2*hbcm*p2p3*p4p6*
     . p5p8-2*hbcm*p2p3*p5p6*p5p8-hbcm*p2p5*p3p6*p5p8+hbcm*p3p4*p4p6*
     . p5p8+hbcm*p3p4*p5p6*p5p8-hbcm*p3p6*p4p5*p5p8))+w1*(2*(p4p8*
     . p4p6)*(ffmcfmb*hbcm3+fmb*hbcm2-2*fmc*hbcm2-hbcm3)+4*(p4p8*p3p6
     . *p2p3)*(-5*ffmcfmb*hbcm+3*hbcm)+2*(p4p8*p3p6)*(ffmcfmb**2*
     . hbcm3+ffmcfmb*fmb*hbcm2-2*ffmcfmb*fmc*hbcm2-ffmcfmb*hbcm3)+4*(
     . ffmcfmb*hbcm*p1p3*p3p6*p4p8-ffmcfmb*hbcm*p3p4*p3p6*p4p8+
     . ffmcfmb*hbcm*p3p5*p3p6*p4p8-ffmcfmb*hbcm3*p4p8*p5p6+hbcm*p1p2*
     . p3p6*p4p8+hbcm*p1p4*p3p6*p4p8-2*hbcm*p2p3*p4p6*p4p8+2*hbcm*
     . p2p3*p4p8*p5p6+hbcm*p2p5*p3p6*p4p8-hbcm*p3p4*p4p6*p4p8-hbcm*
     . p3p4*p4p8*p5p6+hbcm*p3p6*p4p5*p4p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      ans=ccc*(2*(p5p6*p3p7)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*(
     . p5p7*p3p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p7*p3p4)*(
     . -4*fb3*fmc-6*fb4*ffmcfmb*hbcm+2*fb4*hbcm+hbcm)+4*(p6p7*p2p3)*(
     . 4*fb3*fmc+2*fb4*ffmcfmb*hbcm+2*fb4*hbcm-hbcm)+8*(p4p6*p3p7)*(
     . fb3*fmb-fb3*fmc+4*fb4*ffmcfmb*hbcm-fb4*hbcm)+4*(p4p7*p3p6)*(8*
     . fb3*fmb-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm)+4*(p3p7*
     . p3p6)*(10*fb3*ffmcfmb*fmb-6*fb3*ffmcfmb*fmc-4*fb4*ffmcfmb**2*
     . hbcm-2*fb4*ffmcfmb*hbcm-ffmcfmb*hbcm)+p6p7*(4*fb3*ffmcfmb*fmb*
     . hbcm2-12*fb3*ffmcfmb*fmc*hbcm2-4*fb3*fmc*hbcm2-12*fb4*ffmcfmb
     . **2*hbcm3+4*fb4*ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-8*fb4*fmc2*
     . hbcm+2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+hbcm3)+8*(fb4*ffmcfmb
     . *hbcm*p1p3*p6p7+fb4*ffmcfmb*hbcm*p3p5*p6p7+fb4*hbcm*p1p2*p6p7+
     . fb4*hbcm*p1p4*p6p7+4*fb4*hbcm*p2p4*p6p7+fb4*hbcm*p2p5*p6p7+fb4
     . *hbcm*p4p5*p6p7+3*fb4*hbcm*p4p6*p4p7-fb4*hbcm*p4p7*p5p6))
      b(9)=ans
      ans5=4*(p3p7*p3p6*p2p8)*(5*ffmcfmb**2*hbcm2-4*ffmcfmb*fmb*hbcm-
     . ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2)+4*(-ffmcfmb*hbcm2*p1p3*p2p8*
     . p6p7+ffmcfmb*hbcm2*p1p3*p4p8*p6p7-ffmcfmb*hbcm2*p2p8*p3p5*p6p7
     . +ffmcfmb*hbcm2*p3p5*p4p8*p6p7-hbcm2*p1p2*p2p8*p6p7+hbcm2*p1p2*
     . p4p8*p6p7-hbcm2*p1p4*p2p8*p6p7+hbcm2*p1p4*p4p8*p6p7-4*hbcm2*
     . p2p4*p2p8*p6p7+4*hbcm2*p2p4*p4p8*p6p7-hbcm2*p2p5*p2p8*p6p7+
     . hbcm2*p2p5*p4p8*p6p7-hbcm2*p2p8*p4p5*p6p7-hbcm2*p2p8*p4p6*p4p7
     . -hbcm2*p2p8*p4p7*p5p6+hbcm2*p4p5*p4p8*p6p7+hbcm2*p4p6*p4p7*
     . p4p8+hbcm2*p4p7*p4p8*p5p6)
      ans4=4*(p5p6*p4p8*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*p4p6*
     . p3p7)*(3*ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p7*p4p8*p3p6)*(ffmcfmb*
     . hbcm2-fmc*hbcm)+4*(p4p8*p4p7*p3p6)*(-6*ffmcfmb*hbcm2+4*fmb*
     . hbcm+2*fmc*hbcm+hbcm2)+4*(p6p7*p4p8*p3p4)*(-3*ffmcfmb*hbcm2+2*
     . fmc*hbcm+hbcm2)+4*(p5p6*p3p7*p2p8)*(-ffmcfmb*hbcm2-fmc*hbcm)+4
     . *(p4p6*p3p7*p2p8)*(-3*ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p7*p3p6*
     . p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p7*p3p6*p2p8)*(6*ffmcfmb*
     . hbcm2-4*fmb*hbcm-2*fmc*hbcm-hbcm2)+4*(p6p7*p3p4*p2p8)*(3*
     . ffmcfmb*hbcm2-2*fmc*hbcm-hbcm2)+4*(p6p7*p4p8*p2p3)*(ffmcfmb*
     . hbcm2-2*fmb*hbcm-4*fmc*hbcm+3*hbcm2)+4*(p6p7*p2p8*p2p3)*(-
     . ffmcfmb*hbcm2+2*fmb*hbcm+4*fmc*hbcm-3*hbcm2)+2*(p6p7*p4p8)*(-3
     . *ffmcfmb**2*hbcm4+ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3+
     . ffmcfmb*hbcm4-fmb*fmc*hbcm2+fmc*hbcm3-2*fmc2*hbcm2)+2*(p6p7*
     . p2p8)*(3*ffmcfmb**2*hbcm4-ffmcfmb*fmb*hbcm3-3*ffmcfmb*fmc*
     . hbcm3-ffmcfmb*hbcm4+fmb*fmc*hbcm2-fmc*hbcm3+2*fmc2*hbcm2)+4*(
     . p4p8*p3p7*p3p6)*(-5*ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+
     . ffmcfmb*fmc*hbcm+2*ffmcfmb*hbcm2)+ans5
      ans3=w11*ans4
      ans7=2*(-16*fb3*ffmcfmb*p1p3*p3p7*p6p8+16*fb3*ffmcfmb*p1p8*p3p6
     . *p3p7+16*fb3*ffmcfmb*p3p4*p3p7*p6p8-16*fb3*ffmcfmb*p3p5*p3p7*
     . p6p8-16*fb3*ffmcfmb*p3p6*p3p7*p4p8+16*fb3*ffmcfmb*p3p6*p3p7*
     . p5p8-16*fb3*p1p3*p4p7*p6p8+16*fb3*p1p8*p3p6*p4p7-32*fb3*p2p3*
     . p2p8*p6p7-16*fb3*p2p3*p4p6*p7p8+32*fb3*p2p3*p4p8*p6p7+16*fb3*
     . p2p3*p5p6*p7p8-16*fb3*p2p3*p5p7*p6p8+16*fb3*p2p8*p3p6*p4p7+40*
     . fb3*p2p8*p3p7*p4p6-24*fb3*p2p8*p3p7*p5p6+16*fb3*p3p4*p4p7*p6p8
     . -16*fb3*p3p5*p4p7*p6p8-16*fb3*p3p6*p4p7*p4p8+16*fb3*p3p6*p4p7*
     . p5p8-hbcm2*p1p8*p6p7-hbcm2*p5p8*p6p7)
      ans6=16*(p3p7*p3p6*p2p8)*(10*fb3*ffmcfmb-3*fb3)+32*(p6p8*p3p7*
     . p2p3)*(fb3*ffmcfmb-fb3)+32*(p7p8*p3p6*p2p3)*(-2*fb3*ffmcfmb+
     . fb3)+2*(p6p8*p5p7)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+
     . 2*(p7p8*p5p6)*(-4*fb3*ffmcfmb*hbcm2-4*fb4*fmc*hbcm-hbcm2)+2*(
     . p6p7*p4p8)*(-8*fb3*ffmcfmb*hbcm2-8*fb4*fmc*hbcm+3*hbcm2)+4*(
     . p6p8*p4p7)*(4*fb3*ffmcfmb*hbcm2-8*fb4*fmb*hbcm-4*fb4*fmc*hbcm+
     . 3*hbcm2)+2*(p7p8*p4p6)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-3*
     . hbcm2)+2*(p6p7*p2p8)*(8*fb3*ffmcfmb*hbcm2+8*fb4*fmc*hbcm-hbcm2
     . )+2*(p6p8*p3p7)*(4*fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*hbcm2-16
     . *fb4*ffmcfmb*fmb*hbcm-12*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm+5
     . *ffmcfmb*hbcm2+2*fmc*hbcm-hbcm2)+2*(p7p8*p3p6)*(8*fb3*ffmcfmb
     . **2*hbcm2-4*fb3*ffmcfmb*hbcm2+8*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc
     . *hbcm+5*ffmcfmb*hbcm2-4*fmb*hbcm-3*fmc*hbcm-hbcm2)+ans7
      ans2=w1*(4*(p5p6*p4p8*p3p7)*(ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*
     . p4p6*p3p7)*(3*ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p7*p4p8*p3p6)*(
     . ffmcfmb*hbcm2-fmc*hbcm)+4*(p4p8*p4p7*p3p6)*(-6*ffmcfmb*hbcm2+4
     . *fmb*hbcm+2*fmc*hbcm+hbcm2)+4*(p6p7*p4p8*p3p4)*(-3*ffmcfmb*
     . hbcm2+2*fmc*hbcm+hbcm2)+4*(p6p7*p4p8*p2p3)*(ffmcfmb*hbcm2-2*
     . fmb*hbcm-4*fmc*hbcm+3*hbcm2)+2*(p6p7*p4p8)*(-3*ffmcfmb**2*
     . hbcm4+ffmcfmb*fmb*hbcm3+3*ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4-fmb*
     . fmc*hbcm2+fmc*hbcm3-2*fmc2*hbcm2)+4*(p4p8*p3p7*p3p6)*(-5*
     . ffmcfmb**2*hbcm2+4*ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm+2*ffmcfmb
     . *hbcm2)+4*(ffmcfmb*hbcm2*p1p3*p4p8*p6p7+ffmcfmb*hbcm2*p3p5*
     . p4p8*p6p7+hbcm2*p1p2*p4p8*p6p7+hbcm2*p1p4*p4p8*p6p7+4*hbcm2*
     . p2p4*p4p8*p6p7+hbcm2*p2p5*p4p8*p6p7+hbcm2*p4p5*p4p8*p6p7+hbcm2
     . *p4p6*p4p7*p4p8+hbcm2*p4p7*p4p8*p5p6))+ans3+ans6
      ans1=w2*(4*(p5p8*p5p6*p3p7)*(-ffmcfmb*hbcm2-fmc*hbcm)+4*(p5p8*
     . p4p6*p3p7)*(-3*ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p8*p5p7*p3p6)*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+4*(p5p8*p4p7*p3p6)*(6*ffmcfmb*hbcm2-4*
     . fmb*hbcm-2*fmc*hbcm-hbcm2)+4*(p6p7*p5p8*p3p4)*(3*ffmcfmb*hbcm2
     . -2*fmc*hbcm-hbcm2)+4*(p6p7*p5p8*p2p3)*(-ffmcfmb*hbcm2+2*fmb*
     . hbcm+4*fmc*hbcm-3*hbcm2)+2*(p6p7*p5p8)*(3*ffmcfmb**2*hbcm4-
     . ffmcfmb*fmb*hbcm3-3*ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4+fmb*fmc*
     . hbcm2-fmc*hbcm3+2*fmc2*hbcm2)+4*(p5p8*p3p7*p3p6)*(5*ffmcfmb**2
     . *hbcm2-4*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm-2*ffmcfmb*hbcm2)+4*
     . (-ffmcfmb*hbcm2*p1p3*p5p8*p6p7-ffmcfmb*hbcm2*p3p5*p5p8*p6p7-
     . hbcm2*p1p2*p5p8*p6p7-hbcm2*p1p4*p5p8*p6p7-4*hbcm2*p2p4*p5p8*
     . p6p7-hbcm2*p2p5*p5p8*p6p7-hbcm2*p4p5*p5p8*p6p7-hbcm2*p4p6*p4p7
     . *p5p8-hbcm2*p4p7*p5p6*p5p8))+ans2
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w2*(2*(p5p8*p3p4)*(-ffmcfmb*hbcm+hbcm)+2*(p5p8*p2p3)*(
     . -ffmcfmb*hbcm+hbcm)+2*p5p8*(-ffmcfmb**2*hbcm3+ffmcfmb*hbcm3)+2
     . *(ffmcfmb*hbcm*p1p3*p5p8+ffmcfmb*hbcm*p3p5*p5p8+hbcm*p1p2*p5p8
     . +hbcm*p1p4*p5p8+hbcm*p2p5*p5p8+hbcm*p4p5*p5p8))+w1*(2*(p4p8*
     . p3p4)*(ffmcfmb*hbcm-hbcm)+2*(p4p8*p2p3)*(ffmcfmb*hbcm-hbcm)+2*
     . p4p8*(ffmcfmb**2*hbcm3-ffmcfmb*hbcm3)+2*(-ffmcfmb*hbcm*p1p3*
     . p4p8-ffmcfmb*hbcm*p3p5*p4p8-hbcm*p1p2*p4p8-hbcm*p1p4*p4p8-hbcm
     . *p2p5*p4p8-hbcm*p4p5*p4p8))+w11*(2*(p4p8*p3p4)*(ffmcfmb*hbcm-
     . hbcm)+2*(p3p4*p2p8)*(-ffmcfmb*hbcm+hbcm)+2*(p4p8*p2p3)*(
     . ffmcfmb*hbcm-hbcm)+2*(p2p8*p2p3)*(-ffmcfmb*hbcm+hbcm)+2*p4p8*(
     . ffmcfmb**2*hbcm3-ffmcfmb*hbcm3)+2*p2p8*(-ffmcfmb**2*hbcm3+
     . ffmcfmb*hbcm3)+2*(ffmcfmb*hbcm*p1p3*p2p8-ffmcfmb*hbcm*p1p3*
     . p4p8+ffmcfmb*hbcm*p2p8*p3p5-ffmcfmb*hbcm*p3p5*p4p8+hbcm*p1p2*
     . p2p8-hbcm*p1p2*p4p8+hbcm*p1p4*p2p8-hbcm*p1p4*p4p8+hbcm*p2p5*
     . p2p8-hbcm*p2p5*p4p8+hbcm*p2p8*p4p5-hbcm*p4p5*p4p8))+(hbcm*p1p8
     . -hbcm*p2p8-hbcm*p4p8+hbcm*p5p8))
      b(11)=ans
      b(12)=ccc*(4*p3p4*(fb3*ffmcfmb-fb3)+4*p2p3*(fb3*ffmcfmb-fb3)+4*
     . (fb3*ffmcfmb**2*hbcm2-fb3*ffmcfmb*hbcm2-fb3*ffmcfmb*p1p3-fb3*
     . ffmcfmb*p3p5-fb3*p1p2-fb3*p1p4-fb3*p2p5-fb3*p4p5))
      ans=ccc*(w2*(2*(p5p8*p3p4)*(-ffmcfmb*hbcm2+hbcm2)+2*(p5p8*p2p3)
     . *(-ffmcfmb*hbcm2+hbcm2)+2*p5p8*(-ffmcfmb**2*hbcm4+ffmcfmb*
     . hbcm4)+2*(ffmcfmb*hbcm2*p1p3*p5p8+ffmcfmb*hbcm2*p3p5*p5p8+
     . hbcm2*p1p2*p5p8+hbcm2*p1p4*p5p8+hbcm2*p2p5*p5p8+hbcm2*p4p5*
     . p5p8))+w1*(2*(p4p8*p3p4)*(ffmcfmb*hbcm2-hbcm2)+2*(p4p8*p2p3)*(
     . ffmcfmb*hbcm2-hbcm2)+2*p4p8*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4)+2
     . *(-ffmcfmb*hbcm2*p1p3*p4p8-ffmcfmb*hbcm2*p3p5*p4p8-hbcm2*p1p2*
     . p4p8-hbcm2*p1p4*p4p8-hbcm2*p2p5*p4p8-hbcm2*p4p5*p4p8))+w11*(2*
     . (p4p8*p3p4)*(ffmcfmb*hbcm2-hbcm2)+2*(p3p4*p2p8)*(-ffmcfmb*
     . hbcm2+hbcm2)+2*(p4p8*p2p3)*(ffmcfmb*hbcm2-hbcm2)+2*(p2p8*p2p3)
     . *(-ffmcfmb*hbcm2+hbcm2)+2*p4p8*(ffmcfmb**2*hbcm4-ffmcfmb*hbcm4
     . )+2*p2p8*(-ffmcfmb**2*hbcm4+ffmcfmb*hbcm4)+2*(ffmcfmb*hbcm2*
     . p1p3*p2p8-ffmcfmb*hbcm2*p1p3*p4p8+ffmcfmb*hbcm2*p2p8*p3p5-
     . ffmcfmb*hbcm2*p3p5*p4p8+hbcm2*p1p2*p2p8-hbcm2*p1p2*p4p8+hbcm2*
     . p1p4*p2p8-hbcm2*p1p4*p4p8+hbcm2*p2p5*p2p8-hbcm2*p2p5*p4p8+
     . hbcm2*p2p8*p4p5-hbcm2*p4p5*p4p8))+(hbcm2*p1p8-hbcm2*p2p8-hbcm2
     . *p4p8+hbcm2*p5p8))
      b(13)=ans
      b(14)=ccc*(4*p3p4*(-fb4*ffmcfmb*hbcm+fb4*hbcm)+4*p2p3*(-fb4*
     . ffmcfmb*hbcm+fb4*hbcm)+4*(-fb4*ffmcfmb**2*hbcm3+fb4*ffmcfmb*
     . hbcm*p1p3+fb4*ffmcfmb*hbcm*p3p5+fb4*ffmcfmb*hbcm3+fb4*hbcm*
     . p1p2+fb4*hbcm*p1p4+fb4*hbcm*p2p5+fb4*hbcm*p4p5))
      b(15)=ccc*(w2*(2*(p5p8*p3p6)*(8*ffmcfmb*hbcm2-3*hbcm2)+2*(5*
     . hbcm2*p4p6*p5p8-3*hbcm2*p5p6*p5p8))+w1*(2*(p4p8*p3p6)*(-8*
     . ffmcfmb*hbcm2+3*hbcm2)+2*(-5*hbcm2*p4p6*p4p8+3*hbcm2*p4p8*p5p6
     . ))+w11*(2*(p4p8*p3p6)*(-8*ffmcfmb*hbcm2+3*hbcm2)+2*(p3p6*p2p8)
     . *(8*ffmcfmb*hbcm2-3*hbcm2)+2*(5*hbcm2*p2p8*p4p6-3*hbcm2*p2p8*
     . p5p6-5*hbcm2*p4p6*p4p8+3*hbcm2*p4p8*p5p6))+8*hbcm2*p6p8)
      b(16)=ccc*(w2*(2*(p5p8*p3p6)*(-8*ffmcfmb*hbcm+3*hbcm)+2*(-5*
     . hbcm*p4p6*p5p8+3*hbcm*p5p6*p5p8))+w1*(2*(p4p8*p3p6)*(8*ffmcfmb
     . *hbcm-3*hbcm)+2*(5*hbcm*p4p6*p4p8-3*hbcm*p4p8*p5p6))+w11*(2*(
     . p4p8*p3p6)*(8*ffmcfmb*hbcm-3*hbcm)+2*(p3p6*p2p8)*(-8*ffmcfmb*
     . hbcm+3*hbcm)+2*(-5*hbcm*p2p8*p4p6+3*hbcm*p2p8*p5p6+5*hbcm*p4p6
     . *p4p8-3*hbcm*p4p8*p5p6))-8*hbcm*p6p8)
      b(18)=ccc*(w2*(4*(p5p8*p3p7)*(4*ffmcfmb*hbcm2-hbcm2)+4*(3*hbcm2
     . *p4p7*p5p8-hbcm2*p5p7*p5p8))+w1*(4*(p4p8*p3p7)*(-4*ffmcfmb*
     . hbcm2+hbcm2)+4*(-3*hbcm2*p4p7*p4p8+hbcm2*p4p8*p5p7))+w11*(4*(
     . p4p8*p3p7)*(-4*ffmcfmb*hbcm2+hbcm2)+4*(p3p7*p2p8)*(4*ffmcfmb*
     . hbcm2-hbcm2)+4*(3*hbcm2*p2p8*p4p7-hbcm2*p2p8*p5p7-3*hbcm2*p4p7
     . *p4p8+hbcm2*p4p8*p5p7))+8*hbcm2*p7p8)
      b(20)=ccc*(w2*(4*(p5p8*p3p7)*(4*ffmcfmb*hbcm-hbcm)+4*(3*hbcm*
     . p4p7*p5p8-hbcm*p5p7*p5p8))+w1*(4*(p4p8*p3p7)*(-4*ffmcfmb*hbcm+
     . hbcm)+4*(-3*hbcm*p4p7*p4p8+hbcm*p4p8*p5p7))+w11*(4*(p4p8*p3p7)
     . *(-4*ffmcfmb*hbcm+hbcm)+4*(p3p7*p2p8)*(4*ffmcfmb*hbcm-hbcm)+4*
     . (3*hbcm*p2p8*p4p7-hbcm*p2p8*p5p7-3*hbcm*p4p7*p4p8+hbcm*p4p8*
     . p5p7))+8*hbcm*p7p8)
      b(23)=ccc*(8*p3p7*(4*fb4*ffmcfmb*hbcm-fb4*hbcm)+8*(3*fb4*hbcm*
     . p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(4*p3p6*(-8*fb4*ffmcfmb*hbcm+3*fb4*hbcm)+4*(-5*fb4*
     . hbcm*p4p6+3*fb4*hbcm*p5p6))
      b(25)=ccc*(p5p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+24*p4p7*(-
     . fb3*fmb+fb3*fmc+fb4*hbcm)+p3p7*(-24*fb3*ffmcfmb*fmb+28*fb3*
     . ffmcfmb*fmc-4*fb3*fmc+4*fb4*ffmcfmb**2*hbcm+20*fb4*ffmcfmb*
     . hbcm-ffmcfmb*hbcm+hbcm))
      b(26)=ccc*(p4p6*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm+hbcm)+p3p6*(4*fb3*ffmcfmb*fmb-8*fb3*ffmcfmb*fmc-4*fb4*
     . ffmcfmb**2*hbcm-4*fb4*ffmcfmb*hbcm+ffmcfmb*hbcm))
      b(27)=ccc*(8*(p3p7*p3p6)*(-8*fb3*ffmcfmb+3*fb3)+8*p6p7*(-2*fb3*
     . ffmcfmb*hbcm2-fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+8*(4*fb3*
     . p2p3*p6p7-4*fb3*p3p4*p6p7-2*fb3*p3p6*p4p7+2*fb3*p3p6*p5p7-3*
     . fb3*p3p7*p4p6+fb3*p3p7*p5p6))
      b(28)=ccc*(4*w2*(p6p7*p5p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*w1*(
     . p6p7*p4p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+w11*(4*(p6p7*p4p8)*(fmb*
     . hbcm+fmc*hbcm-hbcm2)+4*(p6p7*p2p8)*(-fmb*hbcm-fmc*hbcm+hbcm2))
     . +(16*(p6p8*p3p7)*(-2*fb3*ffmcfmb+fb3)+8*(p7p8*p3p6)*(-4*fb3*
     . ffmcfmb+fb3)+8*(4*fb3*p2p8*p6p7-3*fb3*p4p6*p7p8-2*fb3*p4p7*
     . p6p8-4*fb3*p4p8*p6p7+fb3*p5p6*p7p8+2*fb3*p5p7*p6p8)))
      b(29)=ccc*(4*p3p6*(-8*fb3*ffmcfmb+3*fb3)+4*(-5*fb3*p4p6+3*fb3*
     . p5p6))
      b(30)=ccc*(8*p3p7*(4*fb3*ffmcfmb-fb3)+8*(3*fb3*p4p7-fb3*p5p7))
      ans2=(8*(p6p8*p3p7)*(-4*fb4*ffmcfmb*hbcm+2*fb4*hbcm-hbcm)+8*(
     . p7p8*p3p6)*(-4*fb4*ffmcfmb*hbcm+fb4*hbcm-hbcm)+8*(4*fb4*hbcm*
     . p2p8*p6p7-3*fb4*hbcm*p4p6*p7p8-2*fb4*hbcm*p4p7*p6p8-4*fb4*hbcm
     . *p4p8*p6p7+fb4*hbcm*p5p6*p7p8+2*fb4*hbcm*p5p7*p6p8))
      ans1=w2*(4*(p6p7*p5p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2-
     . hbcm3)+4*(p5p8*p3p7*p3p6)*(-8*ffmcfmb*hbcm+3*hbcm)+4*(4*hbcm*
     . p2p3*p5p8*p6p7-4*hbcm*p3p4*p5p8*p6p7-2*hbcm*p3p6*p4p7*p5p8+2*
     . hbcm*p3p6*p5p7*p5p8-3*hbcm*p3p7*p4p6*p5p8+hbcm*p3p7*p5p6*p5p8)
     . )+w1*(4*(p6p7*p4p8)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+hbcm3
     . )+4*(p4p8*p3p7*p3p6)*(8*ffmcfmb*hbcm-3*hbcm)+4*(-4*hbcm*p2p3*
     . p4p8*p6p7+4*hbcm*p3p4*p4p8*p6p7+2*hbcm*p3p6*p4p7*p4p8-2*hbcm*
     . p3p6*p4p8*p5p7+3*hbcm*p3p7*p4p6*p4p8-hbcm*p3p7*p4p8*p5p6))+w11
     . *(4*(p6p7*p4p8)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2+hbcm3)+4*
     . (p6p7*p2p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2-hbcm3)+4*(
     . p4p8*p3p7*p3p6)*(8*ffmcfmb*hbcm-3*hbcm)+4*(p3p7*p3p6*p2p8)*(-8
     . *ffmcfmb*hbcm+3*hbcm)+4*(4*hbcm*p2p3*p2p8*p6p7-4*hbcm*p2p3*
     . p4p8*p6p7-4*hbcm*p2p8*p3p4*p6p7-2*hbcm*p2p8*p3p6*p4p7+2*hbcm*
     . p2p8*p3p6*p5p7-3*hbcm*p2p8*p3p7*p4p6+hbcm*p2p8*p3p7*p5p6+4*
     . hbcm*p3p4*p4p8*p6p7+2*hbcm*p3p6*p4p7*p4p8-2*hbcm*p3p6*p4p8*
     . p5p7+3*hbcm*p3p7*p4p6*p4p8-hbcm*p3p7*p4p8*p5p6))+ans2
      ans=ccc*ans1
      b(31)=ans
      b(32)=8*ccc*p6p7*(-fb3*fmb+fb3*fmc+fb4*hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+1d0*b(n)
200   continue
      return
	end 
 
      subroutine amp34_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((2*p1p2)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)
     . )
      b(1)=ccc*(4*p6p7*(-2*fb3*ffmcfmb*hbcm2+fb3*fmb2-fb3*fmc2+fb3*
     . hbcm2)+8*(2*fb3*p1p2*p6p7+fb3*p1p4*p6p7-fb3*p1p5*p6p7-2*fb3*
     . p2p3*p6p7-5*fb3*p2p4*p6p7+fb3*p2p5*p6p7-fb3*p3p4*p6p7+fb3*p3p5
     . *p6p7))
      b(2)=ccc*(w2*(2*(p6p7*p5p8)*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*
     . hbcm+hbcm3)+4*(p6p7*p5p8*p2p3)*(2*ffmcfmb*hbcm-3*hbcm)+4*(p6p7
     . *p5p8*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+4*(2*hbcm*p1p2*p5p8*p6p7-4*
     . hbcm*p2p4*p5p8*p6p7-hbcm*p3p4*p5p8*p6p7+hbcm*p3p5*p5p8*p6p7))+
     . w1*(2*(p6p7*p4p8)*(2*ffmcfmb*hbcm3-fmb2*hbcm+fmc2*hbcm-hbcm3)+
     . 4*(p6p7*p4p8*p2p3)*(-2*ffmcfmb*hbcm+3*hbcm)+4*(p6p7*p4p8*p1p3)
     . *(2*ffmcfmb*hbcm-hbcm)+4*(-2*hbcm*p1p2*p4p8*p6p7+4*hbcm*p2p4*
     . p4p8*p6p7+hbcm*p3p4*p4p8*p6p7-hbcm*p3p5*p4p8*p6p7))+(2*(p6p7*
     . p1p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+8*(p6p7*p5p8)*(-fb3
     . *fmb+fb3*fmc+fb4*hbcm)+2*(p6p7*p2p8)*(16*fb3*fmb-12*fb3*fmc+4*
     . fb4*ffmcfmb*hbcm-16*fb4*hbcm+hbcm)))
      b(9)=ccc*(2*(p6p7*p1p3)*(4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm-hbcm)+8*(p6p7*p3p5)*(fb3*fmb-fb3*fmc)+2*(p6p7*p2p3)*(-16*
     . fb3*fmb+12*fb3*fmc+4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*p6p7*
     . (-2*fb3*ffmcfmb*fmb*hbcm2+2*fb3*ffmcfmb*fmc*hbcm2+2*fb3*fmb*
     . hbcm2-2*fb3*fmc*hbcm2+fb4*fmb2*hbcm-fb4*fmc2*hbcm-fb4*hbcm3)+8
     . *(2*fb4*hbcm*p1p2*p6p7-4*fb4*hbcm*p2p4*p6p7-fb4*hbcm*p3p4*p6p7
     . ))
      ans2=(8*(p6p7*p5p8)*(fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*(
     . p6p7*p4p8)*(fb3*ffmcfmb*hbcm2-fb3*hbcm2)+2*(p6p7*p2p8)*(-12*
     . fb3*ffmcfmb*hbcm2+16*fb4*fmb*hbcm+12*fb4*fmc*hbcm-3*hbcm2)+2*(
     . p6p7*p1p8)*(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm+3*hbcm2)+16*(2
     . *fb3*p1p3*p2p8*p6p7+fb3*p1p3*p4p8*p6p7-fb3*p1p8*p3p4*p6p7+2*
     . fb3*p2p3*p2p8*p6p7-3*fb3*p2p3*p4p8*p6p7-3*fb3*p2p8*p3p4*p6p7+2
     . *fb3*p2p8*p3p5*p6p7+fb3*p3p4*p5p8*p6p7-fb3*p3p5*p4p8*p6p7))
      ans1=w2*(4*(p6p7*p5p8*p3p5)*(-fmb*hbcm-fmc*hbcm)+4*(p6p7*p5p8*
     . p2p3)*(-3*ffmcfmb*hbcm2-2*fmb*hbcm-3*fmc*hbcm+6*hbcm2)+4*(p6p7
     . *p5p8*p1p3)*(3*ffmcfmb*hbcm2+fmc*hbcm-2*hbcm2)+2*(p6p7*p5p8)*(
     . 2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*hbcm3-fmb2*hbcm2
     . -2*fmc*hbcm3+fmc2*hbcm2+hbcm4)+4*(-2*hbcm2*p1p2*p5p8*p6p7+
     . hbcm2*p1p4*p5p8*p6p7-hbcm2*p1p5*p5p8*p6p7+3*hbcm2*p2p4*p5p8*
     . p6p7+hbcm2*p2p5*p5p8*p6p7+hbcm2*p3p4*p5p8*p6p7))+w1*(4*(p6p7*
     . p4p8*p3p5)*(fmb*hbcm+fmc*hbcm)+4*(p6p7*p4p8*p2p3)*(3*ffmcfmb*
     . hbcm2+2*fmb*hbcm+3*fmc*hbcm-6*hbcm2)+4*(p6p7*p4p8*p1p3)*(-3*
     . ffmcfmb*hbcm2-fmc*hbcm+2*hbcm2)+2*(p6p7*p4p8)*(-2*ffmcfmb*fmb*
     . hbcm3-2*ffmcfmb*fmc*hbcm3+2*fmb*hbcm3+fmb2*hbcm2+2*fmc*hbcm3-
     . fmc2*hbcm2-hbcm4)+4*(2*hbcm2*p1p2*p4p8*p6p7-hbcm2*p1p4*p4p8*
     . p6p7+hbcm2*p1p5*p4p8*p6p7-3*hbcm2*p2p4*p4p8*p6p7-hbcm2*p2p5*
     . p4p8*p6p7-hbcm2*p3p4*p4p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(27)=ccc*(8*p6p7*(2*fb3*ffmcfmb*hbcm2+fb3*hbcm2-3*fb4*fmb*hbcm
     . -3*fb4*fmc*hbcm)+16*(-fb3*p1p3*p6p7-fb3*p2p3*p6p7+3*fb3*p3p4*
     . p6p7-fb3*p3p5*p6p7))
      b(28)=ccc*(12*w2*(p6p7*p5p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+12*w1*(
     . p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+16*(-fb3*p1p8*p6p7-fb3*
     . p2p8*p6p7+3*fb3*p4p8*p6p7-fb3*p5p8*p6p7))
      b(31)=ccc*(w2*(4*(p6p7*p5p8)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc
     . *hbcm2+hbcm3)+8*(-hbcm*p1p3*p5p8*p6p7-hbcm*p2p3*p5p8*p6p7+3*
     . hbcm*p3p4*p5p8*p6p7-hbcm*p3p5*p5p8*p6p7))+w1*(4*(p6p7*p4p8)*(-
     . 2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*fmc*hbcm2-hbcm3)+8*(hbcm*p1p3*
     . p4p8*p6p7+hbcm*p2p3*p4p8*p6p7-3*hbcm*p3p4*p4p8*p6p7+hbcm*p3p5*
     . p4p8*p6p7))+16*(-fb4*hbcm*p1p8*p6p7-fb4*hbcm*p2p8*p6p7+3*fb4*
     . hbcm*p4p8*p6p7-fb4*hbcm*p5p8*p6p7))
      b(32)=24*ccc*p6p7*(fb3*fmb-fb3*fmc-fb4*hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.9833321660356334d0*b(n)
200   continue
      return
	end 
 
      subroutine amp33_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*p3p4+fmc2-2*
     . p2p4)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(ffmcfmb**2*hbcm2
     . -2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=ccc*(8*p6p7*(fb3*ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*hbcm2-fb3*
     . fmc2)+16*(fb3*ffmcfmb*p1p3*p6p7+2*fb3*ffmcfmb*p3p6*p3p7+4*fb3*
     . ffmcfmb*p3p7*p4p6+fb3*p1p2*p6p7+fb3*p1p4*p6p7-fb3*p2p3*p6p7-
     . fb3*p2p4*p6p7-fb3*p3p4*p6p7+2*fb3*p3p6*p4p7+4*fb3*p4p6*p4p7))
      ans2=w13*(16*(p4p7*p3p6*p1p8)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p1p8)
     . *(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*hbcm)+16*(p3p7*p3p6*
     . p1p8)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+8*(-ffmcfmb*hbcm*p1p3*
     . p1p8*p6p7-2*ffmcfmb*hbcm*p1p8*p3p7*p4p6-hbcm*p1p2*p1p8*p6p7-
     . hbcm*p1p4*p1p8*p6p7+hbcm*p1p8*p2p3*p6p7+hbcm*p1p8*p2p4*p6p7+
     . hbcm*p1p8*p3p4*p6p7-2*hbcm*p1p8*p4p6*p4p7))+(8*(p7p8*p3p6)*(-
     . ffmcfmb*hbcm+hbcm)+8*(p6p7*p2p8)*(-4*fb3*fmc-4*fb4*ffmcfmb*
     . hbcm+hbcm)+4*(p6p8*p4p7)*(-8*fb3*fmb+12*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm-3*hbcm)+4*(p6p8*p3p7)*(-8*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*
     . fmc+4*fb3*fmc+4*fb4*ffmcfmb*hbcm-2*ffmcfmb*hbcm-hbcm)+4*(hbcm*
     . p1p8*p6p7+2*hbcm*p4p6*p7p8))
      ans1=w1*(16*(p4p8*p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+4*(p6p7*p4p8)*
     . (-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*hbcm)+16*(p4p8*p3p7*
     . p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+8*(-ffmcfmb*hbcm*p1p3*
     . p4p8*p6p7-2*ffmcfmb*hbcm*p3p7*p4p6*p4p8-hbcm*p1p2*p4p8*p6p7-
     . hbcm*p1p4*p4p8*p6p7+hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7+
     . hbcm*p3p4*p4p8*p6p7-2*hbcm*p4p6*p4p7*p4p8))+w11*(16*(p4p8*p4p7
     . *p3p6)*(ffmcfmb*hbcm-hbcm)+16*(p4p7*p3p6*p2p8)*(-ffmcfmb*hbcm+
     . hbcm)+4*(p6p7*p4p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*
     . hbcm)+4*(p6p7*p2p8)*(ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-fmc2*
     . hbcm)+16*(p4p8*p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(
     . p3p7*p3p6*p2p8)*(-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+8*(ffmcfmb*
     . hbcm*p1p3*p2p8*p6p7-ffmcfmb*hbcm*p1p3*p4p8*p6p7+2*ffmcfmb*hbcm
     . *p2p8*p3p7*p4p6-2*ffmcfmb*hbcm*p3p7*p4p6*p4p8+hbcm*p1p2*p2p8*
     . p6p7-hbcm*p1p2*p4p8*p6p7+hbcm*p1p4*p2p8*p6p7-hbcm*p1p4*p4p8*
     . p6p7-hbcm*p2p3*p2p8*p6p7+hbcm*p2p3*p4p8*p6p7-hbcm*p2p4*p2p8*
     . p6p7+hbcm*p2p4*p4p8*p6p7-hbcm*p2p8*p3p4*p6p7+2*hbcm*p2p8*p4p6*
     . p4p7+hbcm*p3p4*p4p8*p6p7-2*hbcm*p4p6*p4p7*p4p8))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w1*(4*(p4p8*p4p7)*(ffmcfmb*hbcm2-2*fmb*hbcm-3*fmc*hbcm
     . )+4*(p4p8*p3p7)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2-fmc*hbcm))+w11*(4*(p4p8*p4p7)*(ffmcfmb*hbcm2-2*
     . fmb*hbcm-3*fmc*hbcm)+4*(p4p7*p2p8)*(-ffmcfmb*hbcm2+2*fmb*hbcm+
     . 3*fmc*hbcm)+4*(p4p8*p3p7)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*
     . hbcm+ffmcfmb*hbcm2-fmc*hbcm)+4*(p3p7*p2p8)*(2*ffmcfmb*fmb*hbcm
     . +2*ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2+fmc*hbcm))+w13*(4*(p4p7*p1p8
     . )*(ffmcfmb*hbcm2-2*fmb*hbcm-3*fmc*hbcm)+4*(p3p7*p1p8)*(-2*
     . ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2-fmc*hbcm))+(
     . 4*p7p8*(2*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*hbcm2-2*fb3*fmc2+
     . fmb*hbcm+fmc*hbcm)+16*(fb3*ffmcfmb*p1p3*p7p8+2*fb3*ffmcfmb*
     . p1p8*p3p7-4*fb3*ffmcfmb*p3p7*p5p8+fb3*p1p4*p7p8+3*fb3*p1p8*
     . p4p7+fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*p5p7-fb3*p3p4*p7p8+
     . fb3*p3p7*p4p8-5*fb3*p4p7*p5p8+fb3*p4p8*p5p7)))
      b(3)=ans
      b(4)=ccc*(2*p4p7*(-4*fb3*ffmcfmb*hbcm2+24*fb3*hbcm2+8*fb4*fmb*
     . hbcm+12*fb4*fmc*hbcm-hbcm2)+2*p3p7*(-4*fb3*ffmcfmb**2*hbcm2+20
     . *fb3*ffmcfmb*hbcm2+4*fb3*fmc2+8*fb4*ffmcfmb*fmb*hbcm+8*fb4*
     . ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm-hbcm2)+16*(-fb3*ffmcfmb*hbcm2*
     . p5p7-3*fb3*ffmcfmb*p1p3*p3p7-4*fb3*ffmcfmb*p2p3*p3p7+4*fb3*
     . ffmcfmb*p3p5*p3p7-3*fb3*p1p3*p4p7-fb3*p1p4*p3p7-5*fb3*p2p3*
     . p4p7+2*fb3*p2p3*p5p7-fb3*p2p5*p3p7-fb3*p3p4*p5p7+5*fb3*p3p5*
     . p4p7))
      ans4=(4*(p5p8*p3p7)*(4*fb3*fmc+12*fb4*ffmcfmb*hbcm-hbcm)+4*(
     . p4p8*p3p7)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*(
     . p7p8*p3p5)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+3*hbcm)+4*(p7p8*p3p4
     . )*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+4*(p3p7*p2p8
     . )*(-4*fb3*fmc-12*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8*p2p3)*(4*fb3*
     . fmc+4*fb4*ffmcfmb*hbcm-3*hbcm)+16*(p3p7*p1p8)*(-fb3*fmc-fb4*
     . ffmcfmb*hbcm)+4*(p7p8*p1p3)*(4*fb3*fmc-hbcm)+4*p7p8*(-4*fb3*
     . fmc*hbcm2-2*fb4*ffmcfmb**2*hbcm3+2*fb4*fmc2*hbcm+ffmcfmb*hbcm3
     . +fmb*hbcm2+hbcm3)+16*(-fb4*hbcm*p1p4*p7p8-fb4*hbcm*p1p8*p4p7-
     . fb4*hbcm*p2p5*p7p8-3*fb4*hbcm*p2p8*p4p7+2*fb4*hbcm*p2p8*p5p7+3
     . *fb4*hbcm*p4p7*p5p8-fb4*hbcm*p4p8*p5p7))
      ans3=w13*(4*(p4p7*p1p8)*(-ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2
     . *hbcm3)+4*(p3p7*p1p8)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+8*(ffmcfmb*hbcm*p1p3*p1p8*
     . p3p7+2*ffmcfmb*hbcm*p1p8*p2p3*p3p7-2*ffmcfmb*hbcm*p1p8*p3p5*
     . p3p7+ffmcfmb*hbcm3*p1p8*p5p7+hbcm*p1p3*p1p8*p4p7+hbcm*p1p4*
     . p1p8*p3p7+3*hbcm*p1p8*p2p3*p4p7-2*hbcm*p1p8*p2p3*p5p7+hbcm*
     . p1p8*p2p5*p3p7+hbcm*p1p8*p3p4*p5p7-3*hbcm*p1p8*p3p5*p4p7))+
     . ans4
      ans2=w11*(4*(p4p8*p4p7)*(-ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2
     . *hbcm3)+4*(p4p7*p2p8)*(ffmcfmb*hbcm3+2*fmb*hbcm2-fmc*hbcm2+2*
     . hbcm3)+4*(p4p8*p3p7)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+4*(p3p7*p2p8)*(ffmcfmb**2*
     . hbcm3+2*ffmcfmb*fmb*hbcm2+ffmcfmb*hbcm3-fmc*hbcm2+fmc2*hbcm)+8
     . *(-ffmcfmb*hbcm*p1p3*p2p8*p3p7+ffmcfmb*hbcm*p1p3*p3p7*p4p8-2*
     . ffmcfmb*hbcm*p2p3*p2p8*p3p7+2*ffmcfmb*hbcm*p2p3*p3p7*p4p8+2*
     . ffmcfmb*hbcm*p2p8*p3p5*p3p7-2*ffmcfmb*hbcm*p3p5*p3p7*p4p8-
     . ffmcfmb*hbcm3*p2p8*p5p7+ffmcfmb*hbcm3*p4p8*p5p7-hbcm*p1p3*p2p8
     . *p4p7+hbcm*p1p3*p4p7*p4p8-hbcm*p1p4*p2p8*p3p7+hbcm*p1p4*p3p7*
     . p4p8-3*hbcm*p2p3*p2p8*p4p7+2*hbcm*p2p3*p2p8*p5p7+3*hbcm*p2p3*
     . p4p7*p4p8-2*hbcm*p2p3*p4p8*p5p7-hbcm*p2p5*p2p8*p3p7+hbcm*p2p5*
     . p3p7*p4p8-hbcm*p2p8*p3p4*p5p7+3*hbcm*p2p8*p3p5*p4p7+hbcm*p3p4*
     . p4p8*p5p7-3*hbcm*p3p5*p4p7*p4p8))+ans3
      ans1=w1*(4*(p4p8*p4p7)*(-ffmcfmb*hbcm3-2*fmb*hbcm2+fmc*hbcm2-2*
     . hbcm3)+4*(p4p8*p3p7)*(-ffmcfmb**2*hbcm3-2*ffmcfmb*fmb*hbcm2-
     . ffmcfmb*hbcm3+fmc*hbcm2-fmc2*hbcm)+8*(ffmcfmb*hbcm*p1p3*p3p7*
     . p4p8+2*ffmcfmb*hbcm*p2p3*p3p7*p4p8-2*ffmcfmb*hbcm*p3p5*p3p7*
     . p4p8+ffmcfmb*hbcm3*p4p8*p5p7+hbcm*p1p3*p4p7*p4p8+hbcm*p1p4*
     . p3p7*p4p8+3*hbcm*p2p3*p4p7*p4p8-2*hbcm*p2p3*p4p8*p5p7+hbcm*
     . p2p5*p3p7*p4p8+hbcm*p3p4*p4p8*p5p7-3*hbcm*p3p5*p4p7*p4p8))+
     . ans2
      ans=ccc*ans1
      b(5)=ans
      b(6)=ccc*(w1*(4*(p4p8*p4p6)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p8*
     . p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm))+w11*(4*(p4p8*p4p6)
     . *(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p4p6*p2p8)*(ffmcfmb*hbcm2-fmc*
     . hbcm)+4*(p4p8*p3p6)*(-ffmcfmb**2*hbcm2+ffmcfmb*fmc*hbcm)+4*(
     . p3p6*p2p8)*(ffmcfmb**2*hbcm2-ffmcfmb*fmc*hbcm))+w13*(4*(p4p6*
     . p1p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+4*(p3p6*p1p8)*(-ffmcfmb**2*
     . hbcm2+ffmcfmb*fmc*hbcm))+(2*p6p8*(-8*fb3*ffmcfmb*hbcm2+ffmcfmb
     . *hbcm2-fmc*hbcm)+16*(fb3*ffmcfmb*p1p3*p6p8-2*fb3*ffmcfmb*p2p8*
     . p3p6+fb3*p1p2*p6p8+fb3*p1p4*p6p8-fb3*p2p3*p6p8-3*fb3*p2p4*p6p8
     . -2*fb3*p2p8*p4p6-fb3*p3p4*p6p8)))
      b(7)=ccc*(16*(p3p6*p2p3)*(2*fb3*ffmcfmb+fb3)+2*p4p6*(-4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+2*p3p6*(-4*fb3*ffmcfmb**2*
     . hbcm2+8*fb3*ffmcfmb*hbcm2-4*fb4*ffmcfmb*fmc*hbcm+ffmcfmb*hbcm2
     . )+16*(-fb3*ffmcfmb*p1p3*p3p6-fb3*p1p2*p3p6-fb3*p1p4*p3p6+2*fb3
     . *p2p3*p4p6+3*fb3*p2p4*p3p6+fb3*p3p4*p3p6))
      ans2=w13*(4*(p4p6*p1p8)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p3p6*p2p3*
     . p1p8)*(-2*ffmcfmb*hbcm-hbcm)+4*(p3p6*p1p8)*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3)+8*(ffmcfmb*hbcm*p1p3*p1p8*
     . p3p6+hbcm*p1p2*p1p8*p3p6+hbcm*p1p4*p1p8*p3p6-2*hbcm*p1p8*p2p3*
     . p4p6-3*hbcm*p1p8*p2p4*p3p6-hbcm*p1p8*p3p4*p3p6))+(4*(p4p8*p3p6
     . )*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p6p8*p3p4)*(-4*fb3*
     . fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p3p6*p2p8)*(-4*fb3*
     . fmc+hbcm)+16*(p6p8*p2p3)*(2*fb3*fmc+2*fb4*ffmcfmb*hbcm+fb4*
     . hbcm)+2*p6p8*(-8*fb3*ffmcfmb*fmc*hbcm2-8*fb4*ffmcfmb**2*hbcm3+
     . 8*fb4*ffmcfmb*hbcm3+ffmcfmb*hbcm3+fmc*hbcm2)+4*(-4*fb4*ffmcfmb
     . *hbcm*p1p3*p6p8-4*fb4*hbcm*p1p2*p6p8-4*fb4*hbcm*p1p4*p6p8+12*
     . fb4*hbcm*p2p4*p6p8+8*fb4*hbcm*p2p8*p4p6-hbcm*p1p8*p3p6))
      ans1=w1*(4*(p4p8*p4p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+8*(p4p8*p3p6*
     . p2p3)*(-2*ffmcfmb*hbcm-hbcm)+4*(p4p8*p3p6)*(ffmcfmb**2*hbcm3-
     . ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3)+8*(ffmcfmb*hbcm*p1p3*p3p6*
     . p4p8+hbcm*p1p2*p3p6*p4p8+hbcm*p1p4*p3p6*p4p8-2*hbcm*p2p3*p4p6*
     . p4p8-3*hbcm*p2p4*p3p6*p4p8-hbcm*p3p4*p3p6*p4p8))+w11*(4*(p4p8*
     . p4p6)*(ffmcfmb*hbcm3-fmc*hbcm2)+4*(p4p6*p2p8)*(-ffmcfmb*hbcm3+
     . fmc*hbcm2)+8*(p4p8*p3p6*p2p3)*(-2*ffmcfmb*hbcm-hbcm)+8*(p3p6*
     . p2p8*p2p3)*(2*ffmcfmb*hbcm+hbcm)+4*(p4p8*p3p6)*(ffmcfmb**2*
     . hbcm3-ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3)+4*(p3p6*p2p8)*(-
     . ffmcfmb**2*hbcm3+ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3)+8*(-
     . ffmcfmb*hbcm*p1p3*p2p8*p3p6+ffmcfmb*hbcm*p1p3*p3p6*p4p8-hbcm*
     . p1p2*p2p8*p3p6+hbcm*p1p2*p3p6*p4p8-hbcm*p1p4*p2p8*p3p6+hbcm*
     . p1p4*p3p6*p4p8+2*hbcm*p2p3*p2p8*p4p6-2*hbcm*p2p3*p4p6*p4p8+3*
     . hbcm*p2p4*p2p8*p3p6-3*hbcm*p2p4*p3p6*p4p8+hbcm*p2p8*p3p4*p3p6-
     . hbcm*p3p4*p3p6*p4p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(8*p6p7*(fb4*ffmcfmb**2*hbcm3-2*fb4*ffmcfmb*hbcm3-fb4*
     . fmc2*hbcm)+8*(p6p7*p2p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-2*fb4*
     . hbcm-hbcm)+4*(p4p7*p3p6)*(8*fb3*fmb-12*fb3*fmc-12*fb4*ffmcfmb*
     . hbcm+8*fb4*hbcm+hbcm)+4*(p3p7*p3p6)*(8*fb3*ffmcfmb*fmb-8*fb3*
     . ffmcfmb*fmc-4*fb3*fmc-8*fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm
     . +hbcm)+16*(fb4*ffmcfmb*hbcm*p1p3*p6p7+2*fb4*ffmcfmb*hbcm*p3p7*
     . p4p6+fb4*hbcm*p1p2*p6p7+fb4*hbcm*p1p4*p6p7-fb4*hbcm*p2p4*p6p7-
     . fb4*hbcm*p3p4*p6p7+2*fb4*hbcm*p4p6*p4p7))
      ans4=(4*(p7p8*p3p6)*(4*fb3*fmc2+4*fb4*ffmcfmb*fmc*hbcm+ffmcfmb*
     . hbcm2-2*fmb*hbcm-2*hbcm2)+4*(p6p8*p4p7)*(-12*fb3*ffmcfmb*hbcm2
     . +16*fb3*hbcm2-8*fb4*fmb*hbcm-4*fb4*fmc*hbcm+3*hbcm2)+4*(p7p8*
     . p4p6)*(4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+8*(p6p7*p2p8)
     . *(-4*fb3*ffmcfmb*hbcm2+4*fb4*fmc*hbcm-hbcm2)+4*(p6p8*p3p7)*(-4
     . *fb3*ffmcfmb**2*hbcm2+12*fb3*ffmcfmb*hbcm2-4*fb3*fmc2-8*fb4*
     . ffmcfmb*fmb*hbcm-4*fb4*fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+
     . 4*(-16*fb3*ffmcfmb*p1p3*p3p7*p6p8+16*fb3*ffmcfmb*p1p8*p3p6*
     . p3p7-16*fb3*ffmcfmb*p2p3*p3p6*p7p8+48*fb3*ffmcfmb*p2p8*p3p6*
     . p3p7+16*fb3*ffmcfmb*p3p4*p3p7*p6p8-16*fb3*ffmcfmb*p3p6*p3p7*
     . p4p8-16*fb3*p1p3*p4p7*p6p8+16*fb3*p1p8*p3p6*p4p7+16*fb3*p2p3*
     . p3p7*p6p8-16*fb3*p2p3*p4p6*p7p8+16*fb3*p2p3*p4p7*p6p8-16*fb3*
     . p2p4*p3p6*p7p8+16*fb3*p2p4*p3p7*p6p8-16*fb3*p2p8*p3p4*p6p7+32*
     . fb3*p2p8*p3p6*p4p7+32*fb3*p2p8*p3p7*p4p6+16*fb3*p3p4*p4p7*p6p8
     . -16*fb3*p3p6*p4p7*p4p8-hbcm2*p1p8*p6p7))
      ans3=w13*(8*(p4p6*p3p7*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p7*
     . p3p6*p1p8)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+4*(
     . p6p7*p1p8)*(ffmcfmb**2*hbcm4-2*ffmcfmb*hbcm4-fmc2*hbcm2)+8*(
     . p3p7*p3p6*p1p8)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*hbcm2*p1p3*p1p8*
     . p6p7+hbcm2*p1p2*p1p8*p6p7+hbcm2*p1p4*p1p8*p6p7-hbcm2*p1p8*p2p3
     . *p6p7-hbcm2*p1p8*p2p4*p6p7-hbcm2*p1p8*p3p4*p6p7))+ans4
      ans2=w11*(8*(p4p8*p4p6*p3p7)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*
     . p4p7*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+8*(
     . p4p6*p3p7*p2p8)*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(p4p7*p3p6*p2p8)*(
     . 3*ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm-2*hbcm2)+4*(p6p7*p4p8)*(
     . ffmcfmb**2*hbcm4-2*ffmcfmb*hbcm4-fmc2*hbcm2)+4*(p6p7*p2p8)*(-
     . ffmcfmb**2*hbcm4+2*ffmcfmb*hbcm4+fmc2*hbcm2)+8*(p4p8*p3p7*p3p6
     . )*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*fmc*hbcm+
     . ffmcfmb*hbcm2+fmc*hbcm)+8*(p3p7*p3p6*p2p8)*(ffmcfmb**2*hbcm2-2
     . *ffmcfmb*fmb*hbcm+ffmcfmb*fmc*hbcm-ffmcfmb*hbcm2-fmc*hbcm)+8*(
     . -ffmcfmb*hbcm2*p1p3*p2p8*p6p7+ffmcfmb*hbcm2*p1p3*p4p8*p6p7-
     . hbcm2*p1p2*p2p8*p6p7+hbcm2*p1p2*p4p8*p6p7-hbcm2*p1p4*p2p8*p6p7
     . +hbcm2*p1p4*p4p8*p6p7+hbcm2*p2p3*p2p8*p6p7-hbcm2*p2p3*p4p8*
     . p6p7+hbcm2*p2p4*p2p8*p6p7-hbcm2*p2p4*p4p8*p6p7+hbcm2*p2p8*p3p4
     . *p6p7-hbcm2*p3p4*p4p8*p6p7))+ans3
      ans1=w1*(8*(p4p8*p4p6*p3p7)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p4p8*
     . p4p7*p3p6)*(-3*ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm+2*hbcm2)+4*(
     . p6p7*p4p8)*(ffmcfmb**2*hbcm4-2*ffmcfmb*hbcm4-fmc2*hbcm2)+8*(
     . p4p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm-ffmcfmb*
     . fmc*hbcm+ffmcfmb*hbcm2+fmc*hbcm)+8*(ffmcfmb*hbcm2*p1p3*p4p8*
     . p6p7+hbcm2*p1p2*p4p8*p6p7+hbcm2*p1p4*p4p8*p6p7-hbcm2*p2p3*p4p8
     . *p6p7-hbcm2*p2p4*p4p8*p6p7-hbcm2*p3p4*p4p8*p6p7))+ans2
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(4*w1*(-ffmcfmb*hbcm*p1p3*p4p8+ffmcfmb*hbcm3*p4p8+2*
     . hbcm*p1p2*p4p8-hbcm*p1p4*p4p8-2*hbcm*p2p3*p4p8-3*hbcm*p2p5*
     . p4p8+hbcm*p3p4*p4p8)+4*w11*(ffmcfmb*hbcm*p1p3*p2p8-ffmcfmb*
     . hbcm*p1p3*p4p8-ffmcfmb*hbcm3*p2p8+ffmcfmb*hbcm3*p4p8-2*hbcm*
     . p1p2*p2p8+2*hbcm*p1p2*p4p8+hbcm*p1p4*p2p8-hbcm*p1p4*p4p8+2*
     . hbcm*p2p3*p2p8-2*hbcm*p2p3*p4p8+3*hbcm*p2p5*p2p8-3*hbcm*p2p5*
     . p4p8-hbcm*p2p8*p3p4+hbcm*p3p4*p4p8)+4*w13*(-ffmcfmb*hbcm*p1p3*
     . p1p8+ffmcfmb*hbcm3*p1p8+2*hbcm*p1p2*p1p8-hbcm*p1p4*p1p8-2*hbcm
     . *p1p8*p2p3-3*hbcm*p1p8*p2p5+hbcm*p1p8*p3p4)+(2*p4p8*(-4*fb3*
     . fmc-4*fb4*ffmcfmb*hbcm+hbcm)+4*p2p8*(4*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm-hbcm)+2*hbcm*p1p8))
      b(12)=8*ccc*(fb3*ffmcfmb*hbcm2-fb3*ffmcfmb*p1p3+2*fb3*p1p2-fb3*
     . p1p4-2*fb3*p2p3-3*fb3*p2p5+fb3*p3p4)
      ans2=(2*p4p8*(4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2-4*fb4*fmc*hbcm+
     . hbcm2)+4*p2p8*(-8*fb3*ffmcfmb*hbcm2+8*fb3*hbcm2+4*fb4*fmc*hbcm
     . -hbcm2)+2*p1p8*(-8*fb3*ffmcfmb*hbcm2+hbcm2)+16*(fb3*ffmcfmb*
     . hbcm2*p5p8-2*fb3*p1p3*p2p8+fb3*p1p3*p4p8+2*fb3*p1p8*p2p3-fb3*
     . p1p8*p3p4+fb3*p2p3*p4p8-2*fb3*p2p3*p5p8-fb3*p2p8*p3p4+2*fb3*
     . p2p8*p3p5+fb3*p3p4*p5p8-fb3*p3p5*p4p8))
      ans1=w1*(4*(p4p8*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+8*(p4p8*
     . p2p3)*(ffmcfmb*hbcm2-fmc*hbcm-hbcm2)+4*p4p8*(-ffmcfmb**2*hbcm4
     . +ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4)+4*(-ffmcfmb*hbcm2*p1p3*p4p8+
     . 2*hbcm2*p1p2*p4p8-hbcm2*p1p4*p4p8-3*hbcm2*p2p5*p4p8))+w11*(4*(
     . p4p8*p3p4)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+4*(p3p4*p2p8)*(
     . ffmcfmb*hbcm2-fmc*hbcm-hbcm2)+8*(p4p8*p2p3)*(ffmcfmb*hbcm2-fmc
     . *hbcm-hbcm2)+8*(p2p8*p2p3)*(-ffmcfmb*hbcm2+fmc*hbcm+hbcm2)+4*
     . p4p8*(-ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3+ffmcfmb*hbcm4)+4*
     . p2p8*(ffmcfmb**2*hbcm4-ffmcfmb*fmc*hbcm3-ffmcfmb*hbcm4)+4*(
     . ffmcfmb*hbcm2*p1p3*p2p8-ffmcfmb*hbcm2*p1p3*p4p8-2*hbcm2*p1p2*
     . p2p8+2*hbcm2*p1p2*p4p8+hbcm2*p1p4*p2p8-hbcm2*p1p4*p4p8+3*hbcm2
     . *p2p5*p2p8-3*hbcm2*p2p5*p4p8))+w13*(4*(p3p4*p1p8)*(-ffmcfmb*
     . hbcm2+fmc*hbcm+hbcm2)+8*(p2p3*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm-
     . hbcm2)+4*p1p8*(-ffmcfmb**2*hbcm4+ffmcfmb*fmc*hbcm3+ffmcfmb*
     . hbcm4)+4*(-ffmcfmb*hbcm2*p1p3*p1p8+2*hbcm2*p1p2*p1p8-hbcm2*
     . p1p4*p1p8-3*hbcm2*p1p8*p2p5))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(2*p3p4*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm
     . )+4*p2p3*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+2*(4*
     . fb3*ffmcfmb*fmc*hbcm2+4*fb4*ffmcfmb**2*hbcm3+4*fb4*ffmcfmb*
     . hbcm*p1p3-4*fb4*ffmcfmb*hbcm3-8*fb4*hbcm*p1p2+4*fb4*hbcm*p1p4+
     . 12*fb4*hbcm*p2p5-ffmcfmb*hbcm3))
      b(15)=ccc*(w1*(4*(p4p8*p3p6)*(-ffmcfmb*hbcm2-3*fmc*hbcm)-16*
     . hbcm2*p4p6*p4p8)+w11*(4*(p4p8*p3p6)*(-ffmcfmb*hbcm2-3*fmc*hbcm
     . )+4*(p3p6*p2p8)*(ffmcfmb*hbcm2+3*fmc*hbcm)+16*(hbcm2*p2p8*p4p6
     . -hbcm2*p4p6*p4p8))+w13*(4*(p3p6*p1p8)*(-ffmcfmb*hbcm2-3*fmc*
     . hbcm)-16*hbcm2*p1p8*p4p6)+(2*p6p8*(-20*fb3*ffmcfmb*hbcm2+12*
     . fb4*fmc*hbcm+hbcm2)+32*(fb3*p2p3*p6p8-fb3*p2p8*p3p6-2*fb3*p3p4
     . *p6p8+2*fb3*p3p6*p4p8)))
      b(16)=ccc*(16*w1*(ffmcfmb*hbcm*p3p6*p4p8+hbcm*p4p6*p4p8)+16*w11
     . *(-ffmcfmb*hbcm*p2p8*p3p6+ffmcfmb*hbcm*p3p6*p4p8-hbcm*p2p8*
     . p4p6+hbcm*p4p6*p4p8)+16*w13*(ffmcfmb*hbcm*p1p8*p3p6+hbcm*p1p8*
     . p4p6)+2*p6p8*(-12*fb3*fmc-12*fb4*ffmcfmb*hbcm-hbcm))
      b(17)=ccc*(w1*(2*p4p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+4*
     . (-3*hbcm*p1p3*p4p8-hbcm*p2p3*p4p8-hbcm*p3p4*p4p8+3*hbcm*p3p5*
     . p4p8))+w11*(2*p4p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+2*
     . p2p8*(5*ffmcfmb*hbcm3+3*fmc*hbcm2-6*hbcm3)+4*(3*hbcm*p1p3*p2p8
     . -3*hbcm*p1p3*p4p8+hbcm*p2p3*p2p8-hbcm*p2p3*p4p8+hbcm*p2p8*p3p4
     . -3*hbcm*p2p8*p3p5-hbcm*p3p4*p4p8+3*hbcm*p3p5*p4p8))+w13*(2*
     . p1p8*(-5*ffmcfmb*hbcm3-3*fmc*hbcm2+6*hbcm3)+4*(-3*hbcm*p1p3*
     . p1p8-hbcm*p1p8*p2p3-hbcm*p1p8*p3p4+3*hbcm*p1p8*p3p5))+8*(3*fb4
     . *hbcm*p1p8+fb4*hbcm*p2p8+fb4*hbcm*p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w1*(4*(p4p8*p3p7)*(-2*ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2
     . )+4*(-hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+w11*(4*(p4p8*p3p7)*(
     . -2*ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2)+4*(p3p7*p2p8)*(2*ffmcfmb*
     . hbcm2-2*fmc*hbcm-hbcm2)+4*(hbcm2*p2p8*p4p7-3*hbcm2*p2p8*p5p7-
     . hbcm2*p4p7*p4p8+3*hbcm2*p4p8*p5p7))+w13*(4*(p3p7*p1p8)*(-2*
     . ffmcfmb*hbcm2+2*fmc*hbcm+hbcm2)+4*(-hbcm2*p1p8*p4p7+3*hbcm2*
     . p1p8*p5p7))+(4*p7p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4*fmc
     . *hbcm+hbcm2)+16*(fb3*p1p3*p7p8-fb3*p1p8*p3p7+fb3*p2p3*p7p8-fb3
     . *p2p8*p3p7+fb3*p3p4*p7p8-fb3*p3p5*p7p8-fb3*p3p7*p4p8+fb3*p3p7*
     . p5p8)))
      b(19)=ccc*(6*w1*p4p8*(-ffmcfmb*hbcm2+fmc*hbcm)+w11*(6*p4p8*(-
     . ffmcfmb*hbcm2+fmc*hbcm)+6*p2p8*(ffmcfmb*hbcm2-fmc*hbcm))+6*w13
     . *p1p8*(-ffmcfmb*hbcm2+fmc*hbcm)+8*(-3*fb3*p1p8-fb3*p2p8-fb3*
     . p4p8+3*fb3*p5p8))
      b(20)=ccc*(w1*(4*(p4p8*p3p7)*(-2*ffmcfmb*hbcm+hbcm)+12*(-hbcm*
     . p4p7*p4p8+hbcm*p4p8*p5p7))+w11*(4*(p4p8*p3p7)*(-2*ffmcfmb*hbcm
     . +hbcm)+4*(p3p7*p2p8)*(2*ffmcfmb*hbcm-hbcm)+12*(hbcm*p2p8*p4p7-
     . hbcm*p2p8*p5p7-hbcm*p4p7*p4p8+hbcm*p4p8*p5p7))+w13*(4*(p3p7*
     . p1p8)*(-2*ffmcfmb*hbcm+hbcm)+12*(-hbcm*p1p8*p4p7+hbcm*p1p8*
     . p5p7))+8*p7p8*(-2*fb3*fmc-2*fb4*ffmcfmb*hbcm+hbcm))
      b(21)=3*ccc*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=ccc*(20*fb3*ffmcfmb*hbcm2-24*fb3*hbcm2+24*fb3*p1p3+8*fb3*
     . p2p3+8*fb3*p3p4-24*fb3*p3p5-12*fb4*fmc*hbcm+3*hbcm2)
      b(23)=ccc*(4*p3p7*(4*fb3*fmc+8*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm
     . )+24*(fb4*hbcm*p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(2*p3p6*(12*fb3*fmc-4*fb4*ffmcfmb*hbcm-3*hbcm)-32*fb4
     . *hbcm*p4p6)
      b(25)=ccc*(2*p4p7*(-8*fb3*fmb+20*fb3*fmc+12*fb4*ffmcfmb*hbcm-3*
     . hbcm)+2*p3p7*(-8*fb3*ffmcfmb*fmb+16*fb3*ffmcfmb*fmc+4*fb3*fmc+
     . 8*fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*hbcm-2*ffmcfmb*hbcm-hbcm))
      b(26)=ccc*(2*p4p6*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+2*p3p6*(
     . -4*fb3*ffmcfmb*fmc-4*fb4*ffmcfmb**2*hbcm+ffmcfmb*hbcm))
      b(27)=ccc*(32*(p3p7*p3p6)*(-2*fb3*ffmcfmb-fb3)+4*p6p7*(4*fb3*
     . ffmcfmb*hbcm2-4*fb4*fmc*hbcm+hbcm2)+32*(fb3*p3p4*p6p7-3*fb3*
     . p3p6*p4p7-fb3*p3p7*p4p6))
      b(28)=ccc*(8*w1*(p6p7*p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+w11*(8*(
     . p6p7*p4p8)*(ffmcfmb*hbcm2-fmc*hbcm)+8*(p6p7*p2p8)*(-ffmcfmb*
     . hbcm2+fmc*hbcm))+8*w13*(p6p7*p1p8)*(ffmcfmb*hbcm2-fmc*hbcm)+(
     . 32*(p6p8*p3p7)*(-fb3*ffmcfmb-fb3)+32*(-fb3*ffmcfmb*p3p6*p7p8-
     . fb3*p4p6*p7p8-3*fb3*p4p7*p6p8+fb3*p4p8*p6p7)))
      b(29)=32*ccc*(-fb3*ffmcfmb*p3p6-fb3*p4p6)
      b(30)=ccc*(8*p3p7*(4*fb3*ffmcfmb-fb3)+8*(5*fb3*p4p7-3*fb3*p5p7)
     . )
      ans=ccc*(w1*(8*(p6p7*p4p8)*(-ffmcfmb*hbcm3-fmc*hbcm2)+16*(p4p8*
     . p3p7*p3p6)*(ffmcfmb*hbcm+hbcm)+16*(-hbcm*p3p4*p4p8*p6p7+2*hbcm
     . *p3p6*p4p7*p4p8+hbcm*p3p7*p4p6*p4p8))+w11*(8*(p6p7*p4p8)*(-
     . ffmcfmb*hbcm3-fmc*hbcm2)+8*(p6p7*p2p8)*(ffmcfmb*hbcm3+fmc*
     . hbcm2)+16*(p4p8*p3p7*p3p6)*(ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*
     . p2p8)*(-ffmcfmb*hbcm-hbcm)+16*(hbcm*p2p8*p3p4*p6p7-2*hbcm*p2p8
     . *p3p6*p4p7-hbcm*p2p8*p3p7*p4p6-hbcm*p3p4*p4p8*p6p7+2*hbcm*p3p6
     . *p4p7*p4p8+hbcm*p3p7*p4p6*p4p8))+w13*(8*(p6p7*p1p8)*(-ffmcfmb*
     . hbcm3-fmc*hbcm2)+16*(p3p7*p3p6*p1p8)*(ffmcfmb*hbcm+hbcm)+16*(-
     . hbcm*p1p8*p3p4*p6p7+2*hbcm*p1p8*p3p6*p4p7+hbcm*p1p8*p3p7*p4p6)
     . )+(32*(p6p8*p3p7)*(-fb3*fmc-fb4*ffmcfmb*hbcm-fb4*hbcm)+8*(p7p8
     . *p3p6)*(4*fb3*fmc-hbcm)+32*(-fb4*hbcm*p4p6*p7p8-2*fb4*hbcm*
     . p4p7*p6p8+fb4*hbcm*p4p8*p6p7)))
      b(31)=ans
      b(32)=4*ccc*p6p7*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp32_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*p3p5+fmb2+
     . hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*ffmcfmb*p3p4+
     . fmc2-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2))
      b(1)=ccc*(32*(p4p6*p3p7)*(2*fb3*ffmcfmb-fb3)+16*(p5p7*p3p6)*(-
     . fb3*ffmcfmb-fb3)+16*(p4p7*p3p6)*(-fb3*ffmcfmb+2*fb3)+8*p6p7*(
     . fb3*ffmcfmb**2*hbcm2-2*fb3*ffmcfmb*hbcm2-fb3*fmc2)+16*(fb3*
     . ffmcfmb*p3p6*p3p7+fb3*p1p2*p6p7+fb3*p1p3*p6p7+fb3*p1p4*p6p7-
     . fb3*p2p3*p6p7-fb3*p2p4*p6p7-fb3*p3p4*p6p7+fb3*p3p7*p5p6+2*fb3*
     . p4p6*p4p7-3*fb3*p4p6*p5p7+fb3*p4p7*p5p6))
      ans3=w12*(8*(p4p6*p3p7*p1p8)*(2*ffmcfmb*hbcm-hbcm)+16*(p4p7*
     . p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p1p8)*(ffmcfmb**2*
     . hbcm3-2*ffmcfmb*hbcm3-fmc2*hbcm)+16*(p3p7*p3p6*p1p8)*(-ffmcfmb
     . **2*hbcm+ffmcfmb*hbcm)+8*(hbcm*p1p2*p1p8*p6p7+hbcm*p1p3*p1p8*
     . p6p7+hbcm*p1p4*p1p8*p6p7-hbcm*p1p8*p2p3*p6p7-hbcm*p1p8*p2p4*
     . p6p7-hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p5p7+hbcm*p1p8*p3p7*
     . p5p6+hbcm*p1p8*p4p6*p4p7-2*hbcm*p1p8*p4p6*p5p7+hbcm*p1p8*p4p7*
     . p5p6))+(4*(p6p8*p5p7)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm
     . -hbcm)+4*(p6p7*p4p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)+4*(p6p8*p4p7)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-
     . hbcm)+4*(p7p8*p4p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . 3*hbcm)+4*(p6p7*p2p8)*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm
     . -hbcm)+4*(p6p7*p1p8)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm
     . +hbcm)+4*(p6p8*p3p7)*(-8*fb3*ffmcfmb*fmb+4*fb3*fmb-8*fb4*
     . ffmcfmb**2*hbcm+12*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+4*(p7p8*
     . p3p6)*(-4*fb3*ffmcfmb*fmb-4*fb4*ffmcfmb**2*hbcm+4*fb4*ffmcfmb*
     . hbcm-ffmcfmb*hbcm+2*hbcm))
      ans2=w5*(8*(p4p8*p4p6*p3p7)*(-2*ffmcfmb*hbcm+hbcm)+16*(p4p8*
     . p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+8*(p4p6*p3p7*p1p8)*(2*ffmcfmb*
     . hbcm-hbcm)+16*(p4p7*p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*
     . p4p8)*(-ffmcfmb**2*hbcm3+2*ffmcfmb*hbcm3+fmc2*hbcm)+4*(p6p7*
     . p1p8)*(ffmcfmb**2*hbcm3-2*ffmcfmb*hbcm3-fmc2*hbcm)+16*(p4p8*
     . p3p7*p3p6)*(ffmcfmb**2*hbcm-ffmcfmb*hbcm)+16*(p3p7*p3p6*p1p8)*
     . (-ffmcfmb**2*hbcm+ffmcfmb*hbcm)+8*(hbcm*p1p2*p1p8*p6p7-hbcm*
     . p1p2*p4p8*p6p7+hbcm*p1p3*p1p8*p6p7-hbcm*p1p3*p4p8*p6p7+hbcm*
     . p1p4*p1p8*p6p7-hbcm*p1p4*p4p8*p6p7-hbcm*p1p8*p2p3*p6p7-hbcm*
     . p1p8*p2p4*p6p7-hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p5p7+hbcm*
     . p1p8*p3p7*p5p6+hbcm*p1p8*p4p6*p4p7-2*hbcm*p1p8*p4p6*p5p7+hbcm*
     . p1p8*p4p7*p5p6+hbcm*p2p3*p4p8*p6p7+hbcm*p2p4*p4p8*p6p7+hbcm*
     . p3p4*p4p8*p6p7+hbcm*p3p6*p4p8*p5p7-hbcm*p3p7*p4p8*p5p6-hbcm*
     . p4p6*p4p7*p4p8+2*hbcm*p4p6*p4p8*p5p7-hbcm*p4p7*p4p8*p5p6))+
     . ans3
      ans1=w2*(8*(p5p8*p4p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+16*(p5p8*p4p7
     . *p3p6)*(-ffmcfmb*hbcm+hbcm)+4*(p6p7*p5p8)*(ffmcfmb**2*hbcm3-2*
     . ffmcfmb*hbcm3-fmc2*hbcm)+16*(p5p8*p3p7*p3p6)*(-ffmcfmb**2*hbcm
     . +ffmcfmb*hbcm)+8*(hbcm*p1p2*p5p8*p6p7+hbcm*p1p3*p5p8*p6p7+hbcm
     . *p1p4*p5p8*p6p7-hbcm*p2p3*p5p8*p6p7-hbcm*p2p4*p5p8*p6p7-hbcm*
     . p3p4*p5p8*p6p7-hbcm*p3p6*p5p7*p5p8+hbcm*p3p7*p5p6*p5p8+hbcm*
     . p4p6*p4p7*p5p8-2*hbcm*p4p6*p5p7*p5p8+hbcm*p4p7*p5p6*p5p8))+
     . ans2
      ans=ccc*ans1
      b(2)=ans
      ans2=(32*(p5p8*p3p7)*(-2*fb3*ffmcfmb+fb3)+32*(p3p7*p1p8)*(fb3*
     . ffmcfmb-fb3)+4*p7p8*(2*fb3*ffmcfmb**2*hbcm2-4*fb3*ffmcfmb*
     . hbcm2-2*fb3*fmc2+fmb*hbcm+fmc*hbcm)+16*(fb3*p1p3*p7p8+fb3*p1p4
     . *p7p8+fb3*p1p8*p4p7-2*fb3*p1p8*p5p7+fb3*p2p5*p7p8+fb3*p2p8*
     . p3p7+fb3*p2p8*p5p7-fb3*p3p4*p7p8+fb3*p3p7*p4p8-3*fb3*p4p7*p5p8
     . +fb3*p4p8*p5p7+2*fb3*p5p7*p5p8))
      ans1=w2*(4*(p5p8*p5p7)*(-fmb*hbcm-fmc*hbcm)+4*(p5p8*p4p7)*(
     . ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+4*(p5p8*p3p7)*(2*
     . ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm))+w5*(4*
     . (p5p7*p4p8)*(fmb*hbcm+fmc*hbcm)+4*(p5p7*p1p8)*(-fmb*hbcm-fmc*
     . hbcm)+4*(p4p8*p4p7)*(-ffmcfmb*hbcm2-2*fmb*hbcm-fmc*hbcm+hbcm2)
     . +4*(p4p7*p1p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+4*(
     . p4p8*p3p7)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+fmb*hbcm+
     . fmc*hbcm)+4*(p3p7*p1p8)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm
     . -fmb*hbcm-fmc*hbcm))+w12*(4*(p5p7*p1p8)*(-fmb*hbcm-fmc*hbcm)+4
     . *(p4p7*p1p8)*(ffmcfmb*hbcm2+2*fmb*hbcm+fmc*hbcm-hbcm2)+4*(p3p7
     . *p1p8)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*
     . hbcm))+ans2
      ans=ccc*ans1
      b(3)=ans
      b(4)=ccc*(32*(p3p7*p3p5)*(2*fb3*ffmcfmb-fb3)+32*(p3p7*p2p3)*(-2
     . *fb3*ffmcfmb+fb3)+16*(p3p7*p1p3)*(-2*fb3*ffmcfmb+fb3)+8*p5p7*(
     . -2*fb3*ffmcfmb*hbcm2-2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+2*
     . p4p7*(-4*fb3*ffmcfmb*hbcm2+12*fb3*hbcm2+8*fb4*fmb*hbcm+4*fb4*
     . fmc*hbcm+hbcm2)+8*p3p7*(-fb3*ffmcfmb**2*hbcm2+4*fb3*ffmcfmb*
     . hbcm2+fb3*fmc2-2*fb3*hbcm2+2*fb4*ffmcfmb*fmb*hbcm+2*fb4*
     . ffmcfmb*fmc*hbcm-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*(-fb3*p1p3*p4p7
     . +2*fb3*p1p3*p5p7-fb3*p1p4*p3p7-3*fb3*p2p3*p4p7+4*fb3*p2p3*p5p7
     . -fb3*p2p5*p3p7-fb3*p3p4*p5p7+3*fb3*p3p5*p4p7-2*fb3*p3p5*p5p7))
      ans4=(16*(p5p8*p3p7)*(2*fb4*ffmcfmb*hbcm-fb4*hbcm)+16*(p3p7*
     . p2p8)*(-2*fb4*ffmcfmb*hbcm+fb4*hbcm)+4*(p4p8*p3p7)*(-4*fb3*fmb
     . -4*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8*p3p4)*(4*fb3*fmb+4*fb4*
     . ffmcfmb*hbcm-hbcm)+4*(p3p7*p1p8)*(4*fb3*fmb-4*fb4*ffmcfmb*hbcm
     . +4*fb4*hbcm-hbcm)+4*(p7p8*p1p3)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm
     . -hbcm)+4*p7p8*(4*fb3*ffmcfmb*fmb*hbcm2+2*fb4*ffmcfmb**2*hbcm3+
     . 2*fb4*fmc2*hbcm-fmc*hbcm2+hbcm3)+8*(-2*fb4*hbcm*p1p4*p7p8-2*
     . fb4*hbcm*p1p8*p4p7+4*fb4*hbcm*p1p8*p5p7-2*fb4*hbcm*p2p5*p7p8-4
     . *fb4*hbcm*p2p8*p4p7+6*fb4*hbcm*p2p8*p5p7+4*fb4*hbcm*p4p7*p5p8-
     . 2*fb4*hbcm*p4p8*p5p7-2*fb4*hbcm*p5p7*p5p8-hbcm*p2p3*p7p8+hbcm*
     . p3p5*p7p8))
      ans3=w12*(4*(p4p7*p1p8)*(fmb*hbcm2-fmc*hbcm2+2*hbcm3)+4*(p5p7*
     . p1p8)*(-3*ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(p3p7*p3p5*p1p8)*(2
     . *ffmcfmb*hbcm-hbcm)+8*(p3p7*p2p3*p1p8)*(-2*ffmcfmb*hbcm+hbcm)+
     . 8*(p3p7*p1p8*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+4*(p3p7*p1p8)*(
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmc*hbcm2+
     . fmc2*hbcm-hbcm3)+8*(-hbcm*p1p3*p1p8*p4p7+2*hbcm*p1p3*p1p8*p5p7
     . -hbcm*p1p4*p1p8*p3p7-2*hbcm*p1p8*p2p3*p4p7+3*hbcm*p1p8*p2p3*
     . p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p1p8*p3p4*p5p7+2*hbcm*p1p8*p3p5*
     . p4p7-hbcm*p1p8*p3p5*p5p7))+ans4
      ans2=w5*(4*(p4p8*p4p7)*(-fmb*hbcm2+fmc*hbcm2-2*hbcm3)+4*(p4p7*
     . p1p8)*(fmb*hbcm2-fmc*hbcm2+2*hbcm3)+4*(p5p7*p4p8)*(3*ffmcfmb*
     . hbcm3-fmc*hbcm2+hbcm3)+4*(p5p7*p1p8)*(-3*ffmcfmb*hbcm3+fmc*
     . hbcm2-hbcm3)+8*(p4p8*p3p7*p3p5)*(-2*ffmcfmb*hbcm+hbcm)+8*(p4p8
     . *p3p7*p2p3)*(2*ffmcfmb*hbcm-hbcm)+8*(p3p7*p3p5*p1p8)*(2*
     . ffmcfmb*hbcm-hbcm)+8*(p3p7*p2p3*p1p8)*(-2*ffmcfmb*hbcm+hbcm)+8
     . *(p4p8*p3p7*p1p3)*(2*ffmcfmb*hbcm-hbcm)+8*(p3p7*p1p8*p1p3)*(-2
     . *ffmcfmb*hbcm+hbcm)+4*(p4p8*p3p7)*(-ffmcfmb**2*hbcm3+2*ffmcfmb
     . *fmc*hbcm2-ffmcfmb*hbcm3-fmc*hbcm2-fmc2*hbcm+hbcm3)+4*(p3p7*
     . p1p8)*(ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmc*
     . hbcm2+fmc2*hbcm-hbcm3)+8*(-hbcm*p1p3*p1p8*p4p7+2*hbcm*p1p3*
     . p1p8*p5p7+hbcm*p1p3*p4p7*p4p8-2*hbcm*p1p3*p4p8*p5p7-hbcm*p1p4*
     . p1p8*p3p7+hbcm*p1p4*p3p7*p4p8-2*hbcm*p1p8*p2p3*p4p7+3*hbcm*
     . p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p1p8*p3p4*p5p7+2*hbcm*
     . p1p8*p3p5*p4p7-hbcm*p1p8*p3p5*p5p7+2*hbcm*p2p3*p4p7*p4p8-3*
     . hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p3p7*p4p8+hbcm*p3p4*p4p8*p5p7-2*
     . hbcm*p3p5*p4p7*p4p8+hbcm*p3p5*p4p8*p5p7))+ans3
      ans1=w2*(4*(p5p8*p4p7)*(fmb*hbcm2-fmc*hbcm2+2*hbcm3)+4*(p5p8*
     . p5p7)*(-3*ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(p5p8*p3p7*p3p5)*(2
     . *ffmcfmb*hbcm-hbcm)+8*(p5p8*p3p7*p2p3)*(-2*ffmcfmb*hbcm+hbcm)+
     . 8*(p5p8*p3p7*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+4*(p5p8*p3p7)*(
     . ffmcfmb**2*hbcm3-2*ffmcfmb*fmc*hbcm2+ffmcfmb*hbcm3+fmc*hbcm2+
     . fmc2*hbcm-hbcm3)+8*(-hbcm*p1p3*p4p7*p5p8+2*hbcm*p1p3*p5p7*p5p8
     . -hbcm*p1p4*p3p7*p5p8-2*hbcm*p2p3*p4p7*p5p8+3*hbcm*p2p3*p5p7*
     . p5p8-hbcm*p2p5*p3p7*p5p8-hbcm*p3p4*p5p7*p5p8+2*hbcm*p3p5*p4p7*
     . p5p8-hbcm*p3p5*p5p7*p5p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w2*(4*(p5p8*p5p6)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(
     . p5p8*p3p6)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+2*ffmcfmb*hbcm2
     . +fmb*hbcm-hbcm2))+w5*(4*(p5p6*p4p8)*(-ffmcfmb*hbcm2-fmb*hbcm+
     . hbcm2)+4*(p5p6*p1p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+4*(p4p8*
     . p3p6)*(ffmcfmb**2*hbcm2+ffmcfmb*fmb*hbcm-2*ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2)+4*(p3p6*p1p8)*(-ffmcfmb**2*hbcm2-ffmcfmb*fmb*hbcm+
     . 2*ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+w12*(4*(p5p6*p1p8)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+4*(p3p6*p1p8)*(-ffmcfmb**2*hbcm2-ffmcfmb
     . *fmb*hbcm+2*ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+(2*p6p8*(-ffmcfmb*
     . hbcm2-fmb*hbcm+hbcm2)+16*(p4p8*p3p6)*(-fb3*ffmcfmb+fb3)+16*(
     . p3p6*p1p8)*(fb3*ffmcfmb-fb3)+16*(-fb3*ffmcfmb*p2p8*p3p6-fb3*
     . ffmcfmb*p3p6*p5p8-fb3*p1p8*p5p6-fb3*p2p8*p4p6-fb3*p4p6*p5p8+
     . fb3*p4p8*p5p6)))
      b(6)=ans
      b(7)=ccc*(16*(p3p6*p3p4)*(fb3*ffmcfmb-fb3)+16*(p3p6*p1p3)*(-fb3
     . *ffmcfmb+fb3)+2*p5p6*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4*fb4*
     . fmb*hbcm+hbcm2)+16*p4p6*(-fb3*ffmcfmb*hbcm2+fb3*hbcm2)+2*p3p6*
     . (-4*fb3*ffmcfmb**2*hbcm2+8*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4
     . *ffmcfmb*fmb*hbcm+4*fb4*fmb*hbcm-ffmcfmb*hbcm2+hbcm2)+16*(fb3*
     . ffmcfmb*p2p3*p3p6+fb3*ffmcfmb*p3p5*p3p6+fb3*p1p3*p5p6+fb3*p2p3
     . *p4p6-fb3*p3p4*p5p6+fb3*p3p5*p4p6))
      ans3=w12*(4*(p5p6*p1p8)*(-ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(
     . p4p6*p1p8)*(-ffmcfmb*hbcm3+hbcm3)+8*(p3p6*p3p4*p1p8)*(ffmcfmb*
     . hbcm-hbcm)+8*(p3p6*p1p8*p1p3)*(-ffmcfmb*hbcm+hbcm)+4*(p3p6*
     . p1p8)*(-ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmb
     . *hbcm2-hbcm3)+8*(ffmcfmb*hbcm*p1p8*p2p3*p3p6+ffmcfmb*hbcm*p1p8
     . *p3p5*p3p6+hbcm*p1p3*p1p8*p5p6+hbcm*p1p8*p2p3*p4p6-hbcm*p1p8*
     . p3p4*p5p6+hbcm*p1p8*p3p5*p4p6))+(2*p6p8*(-ffmcfmb*hbcm3-fmb*
     . hbcm2+3*hbcm3)+16*(p4p8*p3p6)*(fb4*ffmcfmb*hbcm-fb4*hbcm)+16*(
     . p3p6*p1p8)*(-fb4*ffmcfmb*hbcm+fb4*hbcm)+4*(4*fb4*ffmcfmb*hbcm*
     . p2p8*p3p6+4*fb4*ffmcfmb*hbcm*p3p6*p5p8+4*fb4*hbcm*p1p8*p5p6+4*
     . fb4*hbcm*p2p8*p4p6+4*fb4*hbcm*p4p6*p5p8-4*fb4*hbcm*p4p8*p5p6-
     . hbcm*p1p3*p6p8+hbcm*p2p3*p6p8+hbcm*p3p4*p6p8+hbcm*p3p5*p6p8))
      ans2=w5*(4*(p5p6*p4p8)*(ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*(p4p8*
     . p4p6)*(ffmcfmb*hbcm3-hbcm3)+4*(p5p6*p1p8)*(-ffmcfmb*hbcm3+fmb*
     . hbcm2-hbcm3)+8*(p4p6*p1p8)*(-ffmcfmb*hbcm3+hbcm3)+8*(p4p8*p3p6
     . *p3p4)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p3p4*p1p8)*(ffmcfmb*hbcm-
     . hbcm)+8*(p4p8*p3p6*p1p3)*(ffmcfmb*hbcm-hbcm)+8*(p3p6*p1p8*p1p3
     . )*(-ffmcfmb*hbcm+hbcm)+4*(p4p8*p3p6)*(ffmcfmb**2*hbcm3+ffmcfmb
     . *fmb*hbcm2-2*ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+4*(p3p6*p1p8)*(-
     . ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmb*hbcm2-
     . hbcm3)+8*(ffmcfmb*hbcm*p1p8*p2p3*p3p6+ffmcfmb*hbcm*p1p8*p3p5*
     . p3p6-ffmcfmb*hbcm*p2p3*p3p6*p4p8-ffmcfmb*hbcm*p3p5*p3p6*p4p8+
     . hbcm*p1p3*p1p8*p5p6-hbcm*p1p3*p4p8*p5p6+hbcm*p1p8*p2p3*p4p6-
     . hbcm*p1p8*p3p4*p5p6+hbcm*p1p8*p3p5*p4p6-hbcm*p2p3*p4p6*p4p8+
     . hbcm*p3p4*p4p8*p5p6-hbcm*p3p5*p4p6*p4p8))+ans3
      ans1=w2*(4*(p5p8*p5p6)*(-ffmcfmb*hbcm3+fmb*hbcm2-hbcm3)+8*(p5p8
     . *p4p6)*(-ffmcfmb*hbcm3+hbcm3)+8*(p5p8*p3p6*p3p4)*(ffmcfmb*hbcm
     . -hbcm)+8*(p5p8*p3p6*p1p3)*(-ffmcfmb*hbcm+hbcm)+4*(p5p8*p3p6)*(
     . -ffmcfmb**2*hbcm3-ffmcfmb*fmb*hbcm2+2*ffmcfmb*hbcm3+fmb*hbcm2-
     . hbcm3)+8*(ffmcfmb*hbcm*p2p3*p3p6*p5p8+ffmcfmb*hbcm*p3p5*p3p6*
     . p5p8+hbcm*p1p3*p5p6*p5p8+hbcm*p2p3*p4p6*p5p8-hbcm*p3p4*p5p6*
     . p5p8+hbcm*p3p5*p4p6*p5p8))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(4*(p4p6*p3p7)*(4*fb3*fmb+12*fb4*ffmcfmb*hbcm-8*fb4*
     . hbcm-hbcm)+4*(p5p7*p3p6)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)+
     . 4*(p4p7*p3p6)*(4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-hbcm)+4
     . *(p6p7*p3p4)*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p2p3
     . )*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)+4*(p6p7*p1p3)*(4*fb3*
     . fmb+4*fb4*ffmcfmb*hbcm-hbcm)+4*(p3p7*p3p6)*(12*fb3*ffmcfmb*fmb
     . -4*fb3*fmb+4*fb4*ffmcfmb**2*hbcm-8*fb4*ffmcfmb*hbcm+4*fb4*hbcm
     . -3*ffmcfmb*hbcm+hbcm)+4*p6p7*(-4*fb3*ffmcfmb*fmb*hbcm2-2*fb4*
     . ffmcfmb**2*hbcm3-2*fb4*fmc2*hbcm+ffmcfmb*hbcm3)+16*(fb4*hbcm*
     . p1p2*p6p7+fb4*hbcm*p1p4*p6p7-fb4*hbcm*p2p4*p6p7+fb4*hbcm*p3p7*
     . p5p6+fb4*hbcm*p4p6*p4p7-2*fb4*hbcm*p4p6*p5p7+fb4*hbcm*p4p7*
     . p5p6))
      ans4=8*(fmb*hbcm*p1p8*p3p6*p5p7-fmb*hbcm*p3p6*p4p8*p5p7-hbcm2*
     . p1p2*p1p8*p6p7+hbcm2*p1p2*p4p8*p6p7-hbcm2*p1p4*p1p8*p6p7+hbcm2
     . *p1p4*p4p8*p6p7+hbcm2*p1p8*p2p3*p6p7+hbcm2*p1p8*p2p4*p6p7+
     . hbcm2*p1p8*p4p6*p5p7-hbcm2*p1p8*p4p7*p5p6-hbcm2*p2p3*p4p8*p6p7
     . -hbcm2*p2p4*p4p8*p6p7-hbcm2*p4p6*p4p8*p5p7+hbcm2*p4p7*p4p8*
     . p5p6)
      ans3=8*(p5p6*p4p8*p3p7)*(ffmcfmb*hbcm2+fmb*hbcm)+8*(p4p8*p4p6*
     . p3p7)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+8*(p4p8*p4p7*p3p6)*(-2*
     . ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p6p7*p4p8*p3p4)*(-ffmcfmb*
     . hbcm2-fmb*hbcm)+8*(p5p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm)+8
     . *(p4p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p3p6
     . *p1p8)*(2*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p6p7*p3p4*p1p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm)+8*(p6p7*p4p8*p1p3)*(ffmcfmb*hbcm2+fmb*
     . hbcm)+8*(p6p7*p1p8*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(p6p7*
     . p4p8)*(-ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3-fmc2*hbcm2)+4*(
     . p6p7*p1p8)*(ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3+fmc2*hbcm2)+8
     . *(p4p8*p3p7*p3p6)*(-2*ffmcfmb**2*hbcm2+2*ffmcfmb*fmb*hbcm+
     . ffmcfmb*hbcm2)+8*(p3p7*p3p6*p1p8)*(2*ffmcfmb**2*hbcm2-2*
     . ffmcfmb*fmb*hbcm-ffmcfmb*hbcm2)+ans4
      ans2=w5*ans3
      ans7=32*(fb3*ffmcfmb*p1p8*p3p6*p3p7-fb3*ffmcfmb*p2p3*p3p6*p7p8-
     . fb3*ffmcfmb*p3p5*p3p6*p7p8-fb3*ffmcfmb*p3p6*p3p7*p4p8+fb3*
     . ffmcfmb*p3p6*p3p7*p5p8+2*fb3*p1p3*p2p8*p6p7-fb3*p1p3*p4p7*p6p8
     . -fb3*p1p3*p5p6*p7p8+fb3*p1p3*p5p7*p6p8+fb3*p1p8*p3p6*p4p7-fb3*
     . p1p8*p3p6*p5p7+fb3*p1p8*p3p7*p5p6-fb3*p2p3*p4p6*p7p8-2*fb3*
     . p2p8*p3p4*p6p7+fb3*p2p8*p3p6*p4p7-fb3*p2p8*p3p6*p5p7+3*fb3*
     . p2p8*p3p7*p4p6+fb3*p3p4*p4p7*p6p8+fb3*p3p4*p5p6*p7p8-fb3*p3p4*
     . p5p7*p6p8-fb3*p3p5*p4p6*p7p8-fb3*p3p6*p4p7*p4p8+fb3*p3p6*p4p8*
     . p5p7+fb3*p3p7*p4p6*p5p8-fb3*p3p7*p4p8*p5p6)
      ans6=32*(p6p8*p3p7*p3p4)*(2*fb3*ffmcfmb-fb3)+32*(p7p8*p3p6*p3p4
     . )*(-fb3*ffmcfmb+fb3)+32*(p3p7*p3p6*p2p8)*(5*fb3*ffmcfmb-fb3)+
     . 32*(p6p8*p3p7*p1p3)*(-2*fb3*ffmcfmb+fb3)+32*(p7p8*p3p6*p1p3)*(
     . fb3*ffmcfmb-fb3)+16*(p6p8*p5p7)*(-fb3*ffmcfmb*hbcm2-fb3*hbcm2+
     . fb4*fmb*hbcm)+4*(p7p8*p5p6)*(4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4
     . *fb4*fmb*hbcm-hbcm2)+4*(p6p7*p4p8)*(4*fb3*ffmcfmb*hbcm2-4*fb3*
     . hbcm2+4*fb4*fmb*hbcm+hbcm2)+8*(p6p8*p4p7)*(2*fb3*ffmcfmb*hbcm2
     . +2*fb3*hbcm2-2*fb4*fmb*hbcm+hbcm2)+4*(p7p8*p4p6)*(4*fb3*
     . ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+4*(p7p8*p3p6)*
     . (-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm+2*ffmcfmb*
     . hbcm2-2*fmb*hbcm-hbcm2)+4*(p6p7*p2p8)*(-12*fb3*ffmcfmb*hbcm2-4
     . *fb3*hbcm2+4*fb4*fmb*hbcm+hbcm2)+4*(p6p7*p1p8)*(-4*fb3*ffmcfmb
     . *hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+8*(p6p8*p3p7)*(4*fb3*
     . ffmcfmb**2*hbcm2+2*fb3*ffmcfmb*hbcm2-2*fb3*hbcm2-4*fb4*ffmcfmb
     . *fmb*hbcm+2*fb4*fmb*hbcm+ffmcfmb*hbcm2)+ans7
      ans5=w12*(8*(p5p6*p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm)+8*(p4p6*
     . p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p4p7*p3p6*p1p8)*
     . (2*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p6p7*p3p4*p1p8)*(ffmcfmb*
     . hbcm2+fmb*hbcm)+8*(p6p7*p1p8*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm)+4
     . *(p6p7*p1p8)*(ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3+fmc2*hbcm2)
     . +8*(p3p7*p3p6*p1p8)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*hbcm2)+8*(fmb*hbcm*p1p8*p3p6*p5p7-hbcm2*p1p2*p1p8*p6p7
     . -hbcm2*p1p4*p1p8*p6p7+hbcm2*p1p8*p2p3*p6p7+hbcm2*p1p8*p2p4*
     . p6p7+hbcm2*p1p8*p4p6*p5p7-hbcm2*p1p8*p4p7*p5p6))+ans6
      ans1=w2*(8*(p5p8*p5p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm)+8*(p5p8*
     . p4p6*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(p5p8*p4p7*p3p6)*
     . (2*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p6p7*p5p8*p3p4)*(ffmcfmb*
     . hbcm2+fmb*hbcm)+8*(p6p7*p5p8*p1p3)*(-ffmcfmb*hbcm2-fmb*hbcm)+4
     . *(p6p7*p5p8)*(ffmcfmb**2*hbcm4+2*ffmcfmb*fmb*hbcm3+fmc2*hbcm2)
     . +8*(p5p8*p3p7*p3p6)*(2*ffmcfmb**2*hbcm2-2*ffmcfmb*fmb*hbcm-
     . ffmcfmb*hbcm2)+8*(fmb*hbcm*p3p6*p5p7*p5p8-hbcm2*p1p2*p5p8*p6p7
     . -hbcm2*p1p4*p5p8*p6p7+hbcm2*p2p3*p5p8*p6p7+hbcm2*p2p4*p5p8*
     . p6p7+hbcm2*p4p6*p5p7*p5p8-hbcm2*p4p7*p5p6*p5p8))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      ans=ccc*(w2*(2*p5p8*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3
     . )+4*(ffmcfmb*hbcm*p1p3*p5p8-hbcm*p1p2*p5p8+hbcm*p1p4*p5p8-hbcm
     . *p1p5*p5p8+hbcm*p2p3*p5p8+2*hbcm*p2p5*p5p8-hbcm*p3p4*p5p8+hbcm
     . *p3p5*p5p8))+w5*(2*p4p8*(2*ffmcfmb*hbcm3-fmb2*hbcm+fmc2*hbcm-
     . hbcm3)+2*p1p8*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3)+4*(
     . ffmcfmb*hbcm*p1p3*p1p8-ffmcfmb*hbcm*p1p3*p4p8-hbcm*p1p2*p1p8+
     . hbcm*p1p2*p4p8+hbcm*p1p4*p1p8-hbcm*p1p4*p4p8-hbcm*p1p5*p1p8+
     . hbcm*p1p5*p4p8+hbcm*p1p8*p2p3+2*hbcm*p1p8*p2p5-hbcm*p1p8*p3p4+
     . hbcm*p1p8*p3p5-hbcm*p2p3*p4p8-2*hbcm*p2p5*p4p8+hbcm*p3p4*p4p8-
     . hbcm*p3p5*p4p8))+w12*(2*p1p8*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*
     . hbcm+hbcm3)+4*(ffmcfmb*hbcm*p1p3*p1p8-hbcm*p1p2*p1p8+hbcm*p1p4
     . *p1p8-hbcm*p1p5*p1p8+hbcm*p1p8*p2p3+2*hbcm*p1p8*p2p5-hbcm*p1p8
     . *p3p4+hbcm*p1p8*p3p5))+(2*p4p8*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4
     . *fb4*hbcm-hbcm)+4*p2p8*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*
     . hbcm+hbcm)+4*p1p8*(-2*fb3*fmb-2*fb4*ffmcfmb*hbcm+2*fb4*hbcm+
     . hbcm)))
      b(11)=ans
      b(12)=4*ccc*(2*fb3*ffmcfmb*hbcm2-2*fb3*ffmcfmb*p1p3-fb3*fmb2+
     . fb3*fmc2-fb3*hbcm2+2*fb3*p1p2-2*fb3*p1p4+2*fb3*p1p5-2*fb3*p2p3
     . -4*fb3*p2p5+2*fb3*p3p4-2*fb3*p3p5)
      ans2=w12*(4*(p3p4*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(p2p3*p1p8)
     . *(2*ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)+4*(p1p8*p1p3)*(2*ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+2*p1p8*(-2*ffmcfmb**2*hbcm4-2*ffmcfmb*
     . fmb*hbcm3+fmb2*hbcm2-fmc2*hbcm2+hbcm4)+4*(-hbcm2*p1p2*p1p8+
     . hbcm2*p1p4*p1p8-hbcm2*p1p5*p1p8+2*hbcm2*p1p8*p2p5+hbcm2*p1p8*
     . p3p5))+(2*p4p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4*fmb*hbcm
     . -hbcm2)+4*p2p8*(-8*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2+4*fb4*fmb*
     . hbcm+hbcm2)+4*p1p8*(-2*fb3*ffmcfmb*hbcm2+2*fb3*hbcm2+2*fb4*fmb
     . *hbcm+hbcm2)+16*(fb3*ffmcfmb*hbcm2*p5p8+fb3*p1p3*p2p8-fb3*p1p3
     . *p5p8-fb3*p1p8*p2p3+fb3*p1p8*p3p5+fb3*p2p3*p4p8-2*fb3*p2p3*
     . p5p8-fb3*p2p8*p3p4+2*fb3*p2p8*p3p5+fb3*p3p4*p5p8-fb3*p3p5*p4p8
     . ))
      ans1=w2*(4*(p5p8*p3p4)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(p5p8*p2p3)*
     . (2*ffmcfmb*hbcm2+2*fmb*hbcm-hbcm2)+4*(p5p8*p1p3)*(2*ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+2*p5p8*(-2*ffmcfmb**2*hbcm4-2*ffmcfmb*
     . fmb*hbcm3+fmb2*hbcm2-fmc2*hbcm2+hbcm4)+4*(-hbcm2*p1p2*p5p8+
     . hbcm2*p1p4*p5p8-hbcm2*p1p5*p5p8+2*hbcm2*p2p5*p5p8+hbcm2*p3p5*
     . p5p8))+w5*(4*(p4p8*p3p4)*(ffmcfmb*hbcm2+fmb*hbcm)+4*(p4p8*p2p3
     . )*(-2*ffmcfmb*hbcm2-2*fmb*hbcm+hbcm2)+4*(p3p4*p1p8)*(-ffmcfmb*
     . hbcm2-fmb*hbcm)+4*(p2p3*p1p8)*(2*ffmcfmb*hbcm2+2*fmb*hbcm-
     . hbcm2)+4*(p4p8*p1p3)*(-2*ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p1p8
     . *p1p3)*(2*ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+2*p4p8*(2*ffmcfmb**2*
     . hbcm4+2*ffmcfmb*fmb*hbcm3-fmb2*hbcm2+fmc2*hbcm2-hbcm4)+2*p1p8*
     . (-2*ffmcfmb**2*hbcm4-2*ffmcfmb*fmb*hbcm3+fmb2*hbcm2-fmc2*hbcm2
     . +hbcm4)+4*(-hbcm2*p1p2*p1p8+hbcm2*p1p2*p4p8+hbcm2*p1p4*p1p8-
     . hbcm2*p1p4*p4p8-hbcm2*p1p5*p1p8+hbcm2*p1p5*p4p8+2*hbcm2*p1p8*
     . p2p5+hbcm2*p1p8*p3p5-2*hbcm2*p2p5*p4p8-hbcm2*p3p5*p4p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(2*p3p4*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+hbcm)+4*p2p3*(
     . 4*fb3*fmb+4*fb4*ffmcfmb*hbcm-2*fb4*hbcm-hbcm)+2*p1p3*(4*fb3*
     . fmb+8*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+2*(-4*fb3*ffmcfmb*fmb*
     . hbcm2-4*fb4*ffmcfmb**2*hbcm3+2*fb4*fmb2*hbcm-2*fb4*fmc2*hbcm-4
     . *fb4*hbcm*p1p2+4*fb4*hbcm*p1p4-4*fb4*hbcm*p1p5+8*fb4*hbcm*p2p5
     . +4*fb4*hbcm*p3p5+2*fb4*hbcm3+ffmcfmb*hbcm3))
      b(15)=ccc*(w2*(4*(p5p8*p3p6)*(4*ffmcfmb*hbcm2-hbcm2)+4*(3*hbcm2
     . *p4p6*p5p8-hbcm2*p5p6*p5p8))+w5*(4*(p4p8*p3p6)*(-4*ffmcfmb*
     . hbcm2+hbcm2)+4*(p3p6*p1p8)*(4*ffmcfmb*hbcm2-hbcm2)+4*(3*hbcm2*
     . p1p8*p4p6-hbcm2*p1p8*p5p6-3*hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+
     . w12*(4*(p3p6*p1p8)*(4*ffmcfmb*hbcm2-hbcm2)+4*(3*hbcm2*p1p8*
     . p4p6-hbcm2*p1p8*p5p6))+8*hbcm2*p6p8)
      b(16)=ccc*(w2*(4*(p5p8*p3p6)*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*
     . p4p6*p5p8+hbcm*p5p6*p5p8))+w5*(4*(p4p8*p3p6)*(4*ffmcfmb*hbcm-
     . hbcm)+4*(p3p6*p1p8)*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p1p8*
     . p4p6+hbcm*p1p8*p5p6+3*hbcm*p4p6*p4p8-hbcm*p4p8*p5p6))+w12*(4*(
     . p3p6*p1p8)*(-4*ffmcfmb*hbcm+hbcm)+4*(-3*hbcm*p1p8*p4p6+hbcm*
     . p1p8*p5p6))-8*hbcm*p6p8)
      b(17)=ccc*(w2*(2*p5p8*(5*ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(
     . -hbcm*p1p3*p5p8+hbcm*p2p3*p5p8+hbcm*p3p4*p5p8-3*hbcm*p3p5*p5p8
     . ))+w5*(2*p4p8*(-5*ffmcfmb*hbcm3+3*fmb*hbcm2+3*hbcm3)+2*p1p8*(5
     . *ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(-hbcm*p1p3*p1p8+hbcm*
     . p1p3*p4p8+hbcm*p1p8*p2p3+hbcm*p1p8*p3p4-3*hbcm*p1p8*p3p5-hbcm*
     . p2p3*p4p8-hbcm*p3p4*p4p8+3*hbcm*p3p5*p4p8))+w12*(2*p1p8*(5*
     . ffmcfmb*hbcm3-3*fmb*hbcm2-3*hbcm3)+4*(-hbcm*p1p3*p1p8+hbcm*
     . p1p8*p2p3+hbcm*p1p8*p3p4-3*hbcm*p1p8*p3p5))+8*(-fb4*hbcm*p1p8+
     . fb4*hbcm*p2p8+fb4*hbcm*p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w2*(4*(p5p8*p3p7)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(hbcm2
     . *p4p7*p5p8-3*hbcm2*p5p7*p5p8))+w5*(4*(p4p8*p3p7)*(ffmcfmb*
     . hbcm2+fmb*hbcm)+4*(p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(
     . hbcm2*p1p8*p4p7-3*hbcm2*p1p8*p5p7-hbcm2*p4p7*p4p8+3*hbcm2*p4p8
     . *p5p7))+w12*(4*(p3p7*p1p8)*(-ffmcfmb*hbcm2-fmb*hbcm)+4*(hbcm2*
     . p1p8*p4p7-3*hbcm2*p1p8*p5p7))+(2*p7p8*(12*fb3*ffmcfmb*hbcm2-4*
     . fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+16*(-fb3*p1p3*p7p8+fb3*p1p8*
     . p3p7+fb3*p2p3*p7p8-fb3*p2p8*p3p7+fb3*p3p4*p7p8-fb3*p3p5*p7p8-
     . fb3*p3p7*p4p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(6*w2*p5p8*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+w5*(6*p4p8
     . *(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+6*p1p8*(-ffmcfmb*hbcm2-fmb*
     . hbcm+hbcm2))+6*w12*p1p8*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+8*(fb3
     . *p1p8-fb3*p2p8-fb3*p4p8+3*fb3*p5p8))
      b(20)=ccc*(w2*(8*(p5p8*p3p7)*(ffmcfmb*hbcm-hbcm)+8*(hbcm*p4p7*
     . p5p8-2*hbcm*p5p7*p5p8))+w5*(8*(p4p8*p3p7)*(-ffmcfmb*hbcm+hbcm)
     . +8*(p3p7*p1p8)*(ffmcfmb*hbcm-hbcm)+8*(hbcm*p1p8*p4p7-2*hbcm*
     . p1p8*p5p7-hbcm*p4p7*p4p8+2*hbcm*p4p8*p5p7))+w12*(8*(p3p7*p1p8)
     . *(ffmcfmb*hbcm-hbcm)+8*(hbcm*p1p8*p4p7-2*hbcm*p1p8*p5p7))+2*
     . p7p8*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+hbcm))
      b(21)=3*ccc*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)
      b(22)=ccc*(20*fb3*ffmcfmb*hbcm2-12*fb3*hbcm2-8*fb3*p1p3+8*fb3*
     . p2p3+8*fb3*p3p4-24*fb3*p3p5-12*fb4*fmb*hbcm-3*hbcm2)
      b(23)=ccc*(2*p3p7*(-4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+
     . hbcm)+16*(fb4*hbcm*p4p7-2*fb4*hbcm*p5p7))
      b(24)=ccc*(8*p3p6*(-4*fb4*ffmcfmb*hbcm+fb4*hbcm)+8*(-3*fb4*hbcm
     . *p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(2*p5p7*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4
     . *hbcm-hbcm)+4*p4p7*(-6*fb3*fmb+2*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*
     . fb4*hbcm+hbcm)+2*p3p7*(-16*fb3*ffmcfmb*fmb+8*fb3*ffmcfmb*fmc+8
     . *fb3*fmb-4*fb3*fmc-8*fb4*ffmcfmb**2*hbcm+12*fb4*ffmcfmb*hbcm-4
     . *fb4*hbcm+2*ffmcfmb*hbcm-hbcm))
      b(26)=ccc*(2*p5p6*(-4*fb3*fmb-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+
     . hbcm)+2*p3p6*(4*fb3*ffmcfmb*fmb-4*fb3*fmb+4*fb4*ffmcfmb**2*
     . hbcm-8*fb4*ffmcfmb*hbcm+4*fb4*hbcm-ffmcfmb*hbcm+hbcm))
      b(27)=ccc*(16*(p3p7*p3p6)*(-4*fb3*ffmcfmb+fb3)+2*p6p7*(12*fb3*
     . ffmcfmb*hbcm2+4*fb3*hbcm2-4*fb4*fmb*hbcm-hbcm2)+16*(-2*fb3*
     . p1p3*p6p7+2*fb3*p3p4*p6p7-fb3*p3p6*p4p7+fb3*p3p6*p5p7-2*fb3*
     . p3p7*p4p6))
      b(28)=ccc*(4*w2*(p6p7*p5p8)*(ffmcfmb*hbcm2+fmb*hbcm-hbcm2)+w5*(
     . 4*(p6p7*p4p8)*(-ffmcfmb*hbcm2-fmb*hbcm+hbcm2)+4*(p6p7*p1p8)*(
     . ffmcfmb*hbcm2+fmb*hbcm-hbcm2))+4*w12*(p6p7*p1p8)*(ffmcfmb*
     . hbcm2+fmb*hbcm-hbcm2)+(16*(p6p8*p3p7)*(-2*fb3*ffmcfmb+fb3)+16*
     . (-2*fb3*ffmcfmb*p3p6*p7p8-2*fb3*p1p8*p6p7-2*fb3*p4p6*p7p8-fb3*
     . p4p7*p6p8+2*fb3*p4p8*p6p7+fb3*p5p7*p6p8)))
      b(29)=ccc*(8*p3p6*(-4*fb3*ffmcfmb+fb3)+8*(-3*fb3*p4p6+fb3*p5p6)
     . )
      b(30)=ccc*(8*p3p7*(4*fb3*ffmcfmb-3*fb3)+8*(3*fb3*p4p7-5*fb3*
     . p5p7))
      ans=ccc*(w2*(4*(p6p7*p5p8)*(3*ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*
     . (p5p8*p3p7*p3p6)*(-4*ffmcfmb*hbcm+hbcm)+8*(-2*hbcm*p1p3*p5p8*
     . p6p7+2*hbcm*p3p4*p5p8*p6p7-hbcm*p3p6*p4p7*p5p8+hbcm*p3p6*p5p7*
     . p5p8-2*hbcm*p3p7*p4p6*p5p8))+w5*(4*(p6p7*p4p8)*(-3*ffmcfmb*
     . hbcm3+fmb*hbcm2-hbcm3)+4*(p6p7*p1p8)*(3*ffmcfmb*hbcm3-fmb*
     . hbcm2+hbcm3)+8*(p4p8*p3p7*p3p6)*(4*ffmcfmb*hbcm-hbcm)+8*(p3p7*
     . p3p6*p1p8)*(-4*ffmcfmb*hbcm+hbcm)+8*(-2*hbcm*p1p3*p1p8*p6p7+2*
     . hbcm*p1p3*p4p8*p6p7+2*hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p4p7+
     . hbcm*p1p8*p3p6*p5p7-2*hbcm*p1p8*p3p7*p4p6-2*hbcm*p3p4*p4p8*
     . p6p7+hbcm*p3p6*p4p7*p4p8-hbcm*p3p6*p4p8*p5p7+2*hbcm*p3p7*p4p6*
     . p4p8))+w12*(4*(p6p7*p1p8)*(3*ffmcfmb*hbcm3-fmb*hbcm2+hbcm3)+8*
     . (p3p7*p3p6*p1p8)*(-4*ffmcfmb*hbcm+hbcm)+8*(-2*hbcm*p1p3*p1p8*
     . p6p7+2*hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p6*p4p7+hbcm*p1p8*p3p6*
     . p5p7-2*hbcm*p1p8*p3p7*p4p6))+(8*(p6p8*p3p7)*(-4*fb4*ffmcfmb*
     . hbcm+2*fb4*hbcm-hbcm)+8*(p7p8*p3p6)*(-4*fb4*ffmcfmb*hbcm-hbcm)
     . +16*(-2*fb4*hbcm*p1p8*p6p7-2*fb4*hbcm*p4p6*p7p8-fb4*hbcm*p4p7*
     . p6p8+2*fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p7*p6p8)))
      b(31)=ans
      b(32)=2*ccc*p6p7*(4*fb3*fmb+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.1818181818181818d0*b(n)
         c(n,2)=c(n,2)-0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp31_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p5)*(ffmcfmb**2*hbcm2+2*ffmcfmb*p3p4+fmc2)*(
     . ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*p1p3-2*ffmcfmb*p3p5
     . +fmb2+hbcm2-2*p1p3-2*p1p5+2*p3p5))
      b(1)=ccc*(8*p6p7*(2*fb3*ffmcfmb*hbcm2-fb3*fmb2+fb3*fmc2-fb3*
     . hbcm2)+16*(fb3*p1p3*p6p7+fb3*p1p5*p6p7+fb3*p2p3*p6p7+2*fb3*
     . p2p4*p6p7+fb3*p3p4*p6p7-fb3*p3p5*p6p7-fb3*p3p6*p4p7+fb3*p3p6*
     . p5p7-fb3*p4p6*p4p7+fb3*p4p6*p5p7-fb3*p4p7*p5p6+fb3*p5p6*p5p7))
      ans2=(4*(p7p8*p5p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8
     . *p4p6)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+4*(p7p8*p3p6)*(4*
     . fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+16*(p6p7*p5p8)*(fb3*fmb-fb3*
     . fmc-fb4*hbcm)+16*(p6p8*p5p7)*(-fb3*fmb+fb3*fmc)+4*(p6p8*p3p7)*
     . (-4*fb3*fmb+4*fb4*ffmcfmb*hbcm+hbcm)+32*(p6p7*p2p8)*(-fb3*fmb+
     . fb3*fmc+fb4*hbcm)+16*(p6p7*p1p8)*(-fb3*fmb+fb3*fmc+fb4*hbcm)+
     . 16*fb4*hbcm*p4p7*p6p8)
      ans1=w1*(4*(p6p7*p4p8)*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+
     . hbcm3)+8*(p5p6*p4p8*p3p7)*(-2*ffmcfmb*hbcm+hbcm)+8*(p4p8*p4p6*
     . p3p7)*(-2*ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7*p3p6)*(-2*ffmcfmb*
     . hbcm+hbcm)+8*(-hbcm*p1p3*p4p8*p6p7-hbcm*p1p5*p4p8*p6p7-hbcm*
     . p2p3*p4p8*p6p7-2*hbcm*p2p4*p4p8*p6p7-hbcm*p3p4*p4p8*p6p7+hbcm*
     . p3p5*p4p8*p6p7))+w15*(4*(p6p7*p5p8)*(2*ffmcfmb*hbcm3-fmb2*hbcm
     . +fmc2*hbcm-hbcm3)+4*(p6p7*p1p8)*(-2*ffmcfmb*hbcm3+fmb2*hbcm-
     . fmc2*hbcm+hbcm3)+8*(p5p8*p5p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+8*(
     . p5p8*p4p6*p3p7)*(2*ffmcfmb*hbcm-hbcm)+8*(p5p8*p3p7*p3p6)*(2*
     . ffmcfmb*hbcm-hbcm)+8*(p5p6*p3p7*p1p8)*(-2*ffmcfmb*hbcm+hbcm)+8
     . *(p4p6*p3p7*p1p8)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*p3p6*p1p8)*(-
     . 2*ffmcfmb*hbcm+hbcm)+8*(-hbcm*p1p3*p1p8*p6p7+hbcm*p1p3*p5p8*
     . p6p7-hbcm*p1p5*p1p8*p6p7+hbcm*p1p5*p5p8*p6p7-hbcm*p1p8*p2p3*
     . p6p7-2*hbcm*p1p8*p2p4*p6p7-hbcm*p1p8*p3p4*p6p7+hbcm*p1p8*p3p5*
     . p6p7+hbcm*p2p3*p5p8*p6p7+2*hbcm*p2p4*p5p8*p6p7+hbcm*p3p4*p5p8*
     . p6p7-hbcm*p3p5*p5p8*p6p7))+ans2
      ans=ccc*ans1
      b(2)=ans
      ans=ccc*(w1*(4*(p5p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+4*(p4p8*
     . p3p7)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+
     . fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*hbcm2*p4p7*p4p8)+w15*(4*(p5p8*
     . p5p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+4*(p5p7*p1p8)*(fmb*hbcm+fmc
     . *hbcm-2*hbcm2)+4*(p5p8*p3p7)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc
     . *hbcm-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+2*hbcm2)+4*(p3p7*
     . p1p8)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+
     . fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*(hbcm2*p1p8*p4p7-hbcm2*p4p7*
     . p5p8))+(2*p7p8*(4*fb3*ffmcfmb*hbcm2+4*fb3*fmb*fmc+4*fb3*fmc2+4
     . *fb4*ffmcfmb*fmb*hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmc*hbcm+
     . fmb*hbcm+fmc*hbcm-3*hbcm2)+16*(fb3*p1p2*p7p8-fb3*p1p8*p4p7-fb3
     . *p2p5*p7p8-fb3*p2p8*p3p7-fb3*p2p8*p5p7+fb3*p3p4*p7p8-fb3*p3p7*
     . p4p8+fb3*p4p7*p5p8-fb3*p4p8*p5p7)))
      b(3)=ans
      b(4)=ccc*(2*p3p7*(-4*fb3*fmb*fmc-4*fb3*fmc2+4*fb4*ffmcfmb*fmb*
     . hbcm+4*fb4*ffmcfmb*fmc*hbcm-4*fb4*fmb*hbcm-4*fb4*fmc*hbcm+fmb*
     . hbcm+fmc*hbcm)+8*p5p7*(2*fb3*ffmcfmb*hbcm2-fb4*fmb*hbcm-fb4*
     . fmc*hbcm)+8*(-fb3*hbcm2*p4p7-2*fb3*p1p2*p3p7+2*fb3*p1p3*p4p7+2
     . *fb3*p2p3*p4p7-4*fb3*p2p3*p5p7+2*fb3*p2p5*p3p7+2*fb3*p3p4*p5p7
     . -2*fb3*p3p5*p4p7))
      ans3=(4*(p5p8*p3p7)*(-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+
     . hbcm)+4*(p7p8*p3p5)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+4*(
     . p3p7*p2p8)*(4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-hbcm)+4*(
     . p7p8*p2p3)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+4*(p3p7*p1p8)*
     . (4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-hbcm)+4*(p7p8*p1p3)*(
     . -4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+16*(p4p8*p3p7)*(fb3*fmb-
     . fb3*fmc)+16*(p7p8*p3p4)*(-fb3*fmb+fb3*fmc)+2*p7p8*(-4*fb3*
     . ffmcfmb*fmb*hbcm2+4*fb3*ffmcfmb*fmc*hbcm2+4*fb3*fmc*hbcm2+4*
     . fb4*ffmcfmb*hbcm3+4*fb4*fmb*fmc*hbcm-4*fb4*fmc2*hbcm+3*fmb*
     . hbcm2-3*fmc*hbcm2+hbcm3)+16*(-fb4*hbcm*p1p2*p7p8+fb4*hbcm*p1p8
     . *p5p7+fb4*hbcm*p2p5*p7p8-fb4*hbcm*p2p8*p5p7+fb4*hbcm*p4p8*p5p7
     . -fb4*hbcm*p5p7*p5p8))
      ans2=w15*(4*(p5p8*p4p7)*(fmb*hbcm2-fmc*hbcm2)+4*(p4p7*p1p8)*(-
     . fmb*hbcm2+fmc*hbcm2)+4*(p5p8*p5p7)*(2*ffmcfmb*hbcm3-2*fmb*
     . hbcm2+2*fmc*hbcm2-hbcm3)+4*(p5p7*p1p8)*(-2*ffmcfmb*hbcm3+2*fmb
     . *hbcm2-2*fmc*hbcm2+hbcm3)+8*(p5p8*p3p7*p3p5)*(2*ffmcfmb*hbcm-
     . hbcm)+8*(p5p8*p3p7*p2p3)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*p3p5*
     . p1p8)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p3*p1p8)*(2*ffmcfmb*
     . hbcm-hbcm)+8*(p5p8*p3p7*p1p3)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*
     . p1p8*p1p3)*(2*ffmcfmb*hbcm-hbcm)+4*(p5p8*p3p7)*(3*ffmcfmb*fmb*
     . hbcm2-3*ffmcfmb*fmc*hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmb*
     . hbcm2+2*fmc*hbcm2-fmc2*hbcm-hbcm3)+4*(p3p7*p1p8)*(-3*ffmcfmb*
     . fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm+2*
     . fmb*hbcm2-2*fmc*hbcm2+fmc2*hbcm+hbcm3)+8*(hbcm*p1p2*p1p8*p3p7-
     . hbcm*p1p2*p3p7*p5p8-hbcm*p1p3*p1p8*p5p7+hbcm*p1p3*p5p7*p5p8+
     . hbcm*p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p1p8*p3p4*p5p7+
     . hbcm*p1p8*p3p5*p5p7-hbcm*p2p3*p5p7*p5p8+hbcm*p2p5*p3p7*p5p8+
     . hbcm*p3p4*p5p7*p5p8-hbcm*p3p5*p5p7*p5p8))+ans3
      ans1=w1*(4*(p4p8*p4p7)*(-fmb*hbcm2+fmc*hbcm2)+4*(p5p7*p4p8)*(-2
     . *ffmcfmb*hbcm3+2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*(p4p8*p3p7*
     . p3p5)*(-2*ffmcfmb*hbcm+hbcm)+8*(p4p8*p3p7*p2p3)*(2*ffmcfmb*
     . hbcm-hbcm)+8*(p4p8*p3p7*p1p3)*(2*ffmcfmb*hbcm-hbcm)+4*(p4p8*
     . p3p7)*(-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*hbcm2-2*ffmcfmb*
     . hbcm3-fmb*fmc*hbcm+2*fmb*hbcm2-2*fmc*hbcm2+fmc2*hbcm+hbcm3)+8*
     . (hbcm*p1p2*p3p7*p4p8-hbcm*p1p3*p4p8*p5p7+hbcm*p2p3*p4p8*p5p7-
     . hbcm*p2p5*p3p7*p4p8-hbcm*p3p4*p4p8*p5p7+hbcm*p3p5*p4p8*p5p7))+
     . ans2
      ans=ccc*ans1
      b(5)=ans
      ans=ccc*(w1*(4*(p5p6*p4p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+4*(p4p8*
     . p4p6)*(fmb*hbcm+fmc*hbcm-hbcm2)+4*(p4p8*p3p6)*(fmb*hbcm+fmc*
     . hbcm-hbcm2))+w15*(4*(p5p8*p5p6)*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*(
     . p5p8*p4p6)*(-fmb*hbcm-fmc*hbcm+hbcm2)+4*(p5p8*p3p6)*(-fmb*hbcm
     . -fmc*hbcm+hbcm2)+4*(p5p6*p1p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+4*(
     . p4p6*p1p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+4*(p3p6*p1p8)*(fmb*hbcm+
     . fmc*hbcm-hbcm2))+(8*p6p8*(2*fb3*ffmcfmb*hbcm2-fb3*fmb2+fb3*
     . fmc2-fb3*hbcm2)+16*(fb3*p1p3*p6p8+fb3*p1p5*p6p8-fb3*p1p8*p3p6-
     . fb3*p1p8*p4p6-fb3*p1p8*p5p6+fb3*p2p3*p6p8+2*fb3*p2p4*p6p8+fb3*
     . p2p8*p3p6+fb3*p2p8*p4p6+fb3*p2p8*p5p6+fb3*p3p4*p6p8-fb3*p3p5*
     . p6p8-fb3*p3p6*p4p8+fb3*p3p6*p5p8-fb3*p4p6*p4p8+fb3*p4p6*p5p8-
     . fb3*p4p8*p5p6+fb3*p5p6*p5p8)))
      b(6)=ans
      b(7)=ccc*(8*p3p6*(fb3*fmb2-fb3*fmc2-fb4*fmb*hbcm-fb4*fmc*hbcm)+
     . 8*p5p6*(2*fb3*ffmcfmb*hbcm2-fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*
     . hbcm)+8*p4p6*(2*fb3*ffmcfmb*hbcm2-fb3*hbcm2-fb4*fmb*hbcm-fb4*
     . fmc*hbcm)+16*(fb3*p1p3*p4p6+fb3*p1p3*p5p6-fb3*p1p5*p3p6-2*fb3*
     . p2p3*p3p6-fb3*p2p3*p4p6-fb3*p2p3*p5p6-2*fb3*p2p4*p3p6+fb3*p3p4
     . *p4p6+fb3*p3p4*p5p6-fb3*p3p5*p4p6-fb3*p3p5*p5p6))
      ans3=(16*(p5p8*p3p6)*(-fb3*fmb+fb3*fmc)+16*(p6p8*p3p5)*(fb3*fmb
     . -fb3*fmc)+16*(p3p6*p2p8)*(-fb3*fmb+fb3*fmc)+16*(p6p8*p2p3)*(
     . fb3*fmb-fb3*fmc-2*fb4*hbcm)+16*(p3p6*p1p8)*(fb3*fmb-fb3*fmc)+
     . 16*(p6p8*p1p3)*(-fb3*fmb+fb3*fmc)+8*p6p8*(-2*fb3*ffmcfmb*fmb*
     . hbcm2+2*fb3*ffmcfmb*fmc*hbcm2+2*fb3*fmb*hbcm2-2*fb3*fmc*hbcm2+
     . fb4*fmb2*hbcm-fb4*fmc2*hbcm-fb4*hbcm3)+16*(-fb4*hbcm*p1p5*p6p8
     . +fb4*hbcm*p1p8*p4p6+fb4*hbcm*p1p8*p5p6-2*fb4*hbcm*p2p4*p6p8-
     . fb4*hbcm*p2p8*p4p6-fb4*hbcm*p2p8*p5p6-fb4*hbcm*p3p4*p6p8+fb4*
     . hbcm*p3p6*p4p8+fb4*hbcm*p4p6*p4p8-fb4*hbcm*p4p6*p5p8+fb4*hbcm*
     . p4p8*p5p6-fb4*hbcm*p5p6*p5p8))
      ans2=w15*(4*(p5p8*p3p6)*(-fmb*hbcm2+fmb2*hbcm+fmc*hbcm2-fmc2*
     . hbcm)+4*(p3p6*p1p8)*(fmb*hbcm2-fmb2*hbcm-fmc*hbcm2+fmc2*hbcm)+
     . 4*(p5p8*p5p6)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*(
     . p5p8*p4p6)*(2*ffmcfmb*hbcm3-fmb*hbcm2+fmc*hbcm2-hbcm3)+4*(p5p6
     . *p1p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+hbcm3)+4*(p4p6*
     . p1p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+hbcm3)+8*(-hbcm*
     . p1p3*p1p8*p4p6-hbcm*p1p3*p1p8*p5p6+hbcm*p1p3*p4p6*p5p8+hbcm*
     . p1p3*p5p6*p5p8+hbcm*p1p5*p1p8*p3p6-hbcm*p1p5*p3p6*p5p8+2*hbcm*
     . p1p8*p2p3*p3p6+hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p3*p5p6+2*hbcm*
     . p1p8*p2p4*p3p6-hbcm*p1p8*p3p4*p4p6-hbcm*p1p8*p3p4*p5p6+hbcm*
     . p1p8*p3p5*p4p6+hbcm*p1p8*p3p5*p5p6-2*hbcm*p2p3*p3p6*p5p8-hbcm*
     . p2p3*p4p6*p5p8-hbcm*p2p3*p5p6*p5p8-2*hbcm*p2p4*p3p6*p5p8+hbcm*
     . p3p4*p4p6*p5p8+hbcm*p3p4*p5p6*p5p8-hbcm*p3p5*p4p6*p5p8-hbcm*
     . p3p5*p5p6*p5p8))+ans3
      ans1=w1*(4*(p4p8*p3p6)*(fmb*hbcm2-fmb2*hbcm-fmc*hbcm2+fmc2*hbcm
     . )+4*(p5p6*p4p8)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+hbcm3)+4
     . *(p4p8*p4p6)*(-2*ffmcfmb*hbcm3+fmb*hbcm2-fmc*hbcm2+hbcm3)+8*(-
     . hbcm*p1p3*p4p6*p4p8-hbcm*p1p3*p4p8*p5p6+hbcm*p1p5*p3p6*p4p8+2*
     . hbcm*p2p3*p3p6*p4p8+hbcm*p2p3*p4p6*p4p8+hbcm*p2p3*p4p8*p5p6+2*
     . hbcm*p2p4*p3p6*p4p8-hbcm*p3p4*p4p6*p4p8-hbcm*p3p4*p4p8*p5p6+
     . hbcm*p3p5*p4p6*p4p8+hbcm*p3p5*p4p8*p5p6))+ans2
      ans=ccc*ans1
      b(8)=ans
      b(9)=ccc*(4*(p5p6*p3p7)*(-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm+hbcm)+4*(p4p6*p3p7)*(-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm+hbcm)+16*(p5p7*p3p6)*(fb3*fmb-fb3*fmc)+16*(p3p7*p3p6)*(
     . fb3*fmb-fb3*fmc-fb4*hbcm)+16*(p6p7*p3p5)*(-fb3*fmb+fb3*fmc)+16
     . *(p6p7*p2p3)*(2*fb3*fmb-2*fb3*fmc-fb4*hbcm)+16*(p6p7*p1p3)*(
     . fb3*fmb-fb3*fmc)+8*p6p7*(2*fb3*ffmcfmb*fmb*hbcm2-2*fb3*ffmcfmb
     . *fmc*hbcm2-2*fb3*fmb*hbcm2+2*fb3*fmc*hbcm2-fb4*fmb2*hbcm+fb4*
     . fmc2*hbcm+fb4*hbcm3)+16*(fb4*hbcm*p1p5*p6p7+2*fb4*hbcm*p2p4*
     . p6p7+fb4*hbcm*p3p4*p6p7-fb4*hbcm*p3p6*p4p7))
      ans4=8*(hbcm2*p1p5*p1p8*p6p7-hbcm2*p1p5*p5p8*p6p7+2*hbcm2*p1p8*
     . p2p4*p6p7+hbcm2*p1p8*p3p4*p6p7+hbcm2*p1p8*p4p6*p4p7-hbcm2*p1p8
     . *p4p6*p5p7+hbcm2*p1p8*p4p7*p5p6-hbcm2*p1p8*p5p6*p5p7-2*hbcm2*
     . p2p4*p5p8*p6p7-hbcm2*p3p4*p5p8*p6p7-hbcm2*p4p6*p4p7*p5p8+hbcm2
     . *p4p6*p5p7*p5p8-hbcm2*p4p7*p5p6*p5p8+hbcm2*p5p6*p5p7*p5p8)
      ans3=8*(p5p8*p4p7*p3p6)*(-fmb*hbcm-fmc*hbcm+hbcm2)+8*(p6p7*p5p8
     . *p3p5)*(fmb*hbcm+fmc*hbcm)+8*(p6p7*p5p8*p2p3)*(fmb*hbcm+fmc*
     . hbcm-2*hbcm2)+8*(p4p7*p3p6*p1p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+8*(
     . p6p7*p3p5*p1p8)*(-fmb*hbcm-fmc*hbcm)+8*(p6p7*p2p3*p1p8)*(-fmb*
     . hbcm-fmc*hbcm+2*hbcm2)+8*(p6p7*p5p8*p1p3)*(-fmb*hbcm-fmc*hbcm)
     . +8*(p6p7*p1p8*p1p3)*(fmb*hbcm+fmc*hbcm)+8*(p5p8*p5p6*p3p7)*(-3
     . *ffmcfmb*hbcm2+fmb*hbcm+hbcm2)+8*(p5p8*p4p6*p3p7)*(-3*ffmcfmb*
     . hbcm2+fmb*hbcm+hbcm2)+8*(p5p6*p3p7*p1p8)*(3*ffmcfmb*hbcm2-fmb*
     . hbcm-hbcm2)+8*(p4p6*p3p7*p1p8)*(3*ffmcfmb*hbcm2-fmb*hbcm-hbcm2
     . )+4*(p6p7*p5p8)*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3+2*
     . fmb*hbcm3+fmb2*hbcm2+2*fmc*hbcm3-fmc2*hbcm2-hbcm4)+4*(p6p7*
     . p1p8)*(2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*hbcm3-
     . fmb2*hbcm2-2*fmc*hbcm3+fmc2*hbcm2+hbcm4)+8*(p5p8*p3p7*p3p6)*(-
     . 2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+fmb*hbcm+fmc*hbcm)+8*(
     . p3p7*p3p6*p1p8)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc*hbcm-fmb*
     . hbcm-fmc*hbcm)+ans4
      ans2=w15*ans3
      ans6=16*(fb3*hbcm2*p5p7*p6p8+2*fb3*p1p3*p2p8*p6p7-2*fb3*p1p3*
     . p4p6*p7p8+2*fb3*p1p3*p4p7*p6p8-2*fb3*p1p3*p4p8*p6p7-2*fb3*p1p3
     . *p5p6*p7p8+2*fb3*p1p5*p3p6*p7p8-2*fb3*p1p5*p3p7*p6p8+2*fb3*
     . p1p8*p3p4*p6p7-2*fb3*p1p8*p3p6*p4p7+2*fb3*p1p8*p3p7*p4p6+2*fb3
     . *p1p8*p3p7*p5p6-2*fb3*p2p3*p2p8*p6p7+2*fb3*p2p3*p3p6*p7p8-2*
     . fb3*p2p3*p3p7*p6p8+2*fb3*p2p3*p4p6*p7p8-2*fb3*p2p3*p4p7*p6p8+2
     . *fb3*p2p3*p4p8*p6p7+2*fb3*p2p3*p5p6*p7p8+2*fb3*p2p4*p3p6*p7p8-
     . 2*fb3*p2p4*p3p7*p6p8+4*fb3*p2p8*p3p4*p6p7-2*fb3*p2p8*p3p5*p6p7
     . -4*fb3*p2p8*p3p6*p3p7-4*fb3*p2p8*p3p6*p4p7+2*fb3*p2p8*p3p6*
     . p5p7-4*fb3*p2p8*p3p7*p4p6-4*fb3*p2p8*p3p7*p5p6+2*fb3*p3p4*p5p7
     . *p6p8-2*fb3*p3p4*p5p8*p6p7+2*fb3*p3p5*p4p6*p7p8-2*fb3*p3p5*
     . p4p7*p6p8+2*fb3*p3p5*p4p8*p6p7+2*fb3*p3p5*p5p6*p7p8+2*fb3*p3p6
     . *p4p7*p5p8-2*fb3*p3p6*p4p8*p5p7-2*fb3*p3p7*p4p6*p5p8-2*fb3*
     . p3p7*p5p6*p5p8)
      ans5=16*(p6p7*p5p8)*(-fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+16*(
     . p6p7*p1p8)*(fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+4*(p6p8*p3p7)
     . *(4*fb3*fmb*fmc+4*fb3*fmb2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*
     . ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+4*(p7p8*p3p6)*(-4*fb3*fmb*
     . fmc-4*fb3*fmb2-4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4
     . *fb4*fmb*hbcm+4*fb4*fmc*hbcm-fmb*hbcm-fmc*hbcm)+4*(p7p8*p5p6)*
     . (-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2+4*fb4*fmb*hbcm-3*hbcm2)+32*(
     . p6p7*p4p8)*(-fb3*ffmcfmb*hbcm2+fb3*hbcm2)+16*(p6p8*p4p7)*(2*
     . fb3*ffmcfmb*hbcm2-2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+4*(
     . p7p8*p4p6)*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2+4*fb4*fmb*hbcm-3*
     . hbcm2)+32*(p6p7*p2p8)*(fb3*ffmcfmb*hbcm2-fb4*fmb*hbcm-fb4*fmc*
     . hbcm)+ans6
      ans1=w1*(8*(p4p8*p4p7*p3p6)*(fmb*hbcm+fmc*hbcm-hbcm2)+8*(p6p7*
     . p4p8*p3p5)*(-fmb*hbcm-fmc*hbcm)+8*(p6p7*p4p8*p2p3)*(-fmb*hbcm-
     . fmc*hbcm+2*hbcm2)+8*(p6p7*p4p8*p1p3)*(fmb*hbcm+fmc*hbcm)+8*(
     . p5p6*p4p8*p3p7)*(3*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*(p4p8*p4p6*
     . p3p7)*(3*ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+4*(p6p7*p4p8)*(2*
     . ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-2*fmb*hbcm3-fmb2*hbcm2-2
     . *fmc*hbcm3+fmc2*hbcm2+hbcm4)+8*(p4p8*p3p7*p3p6)*(2*ffmcfmb*fmb
     . *hbcm+2*ffmcfmb*fmc*hbcm-fmb*hbcm-fmc*hbcm)+8*(hbcm2*p1p5*p4p8
     . *p6p7+2*hbcm2*p2p4*p4p8*p6p7+hbcm2*p3p4*p4p8*p6p7+hbcm2*p4p6*
     . p4p7*p4p8-hbcm2*p4p6*p4p8*p5p7+hbcm2*p4p7*p4p8*p5p6-hbcm2*p4p8
     . *p5p6*p5p7))+ans2+ans5
      ans=ccc*ans1
      b(10)=ans
      b(11)=ccc*(w1*(2*p4p8*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+
     . hbcm3)+4*(-2*hbcm*p1p2*p4p8-hbcm*p1p3*p4p8-hbcm*p1p5*p4p8+hbcm
     . *p2p3*p4p8+2*hbcm*p2p5*p4p8-hbcm*p3p4*p4p8+hbcm*p3p5*p4p8))+
     . w15*(2*p5p8*(2*ffmcfmb*hbcm3-fmb2*hbcm+fmc2*hbcm-hbcm3)+2*p1p8
     . *(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3)+4*(-2*hbcm*p1p2*
     . p1p8+2*hbcm*p1p2*p5p8-hbcm*p1p3*p1p8+hbcm*p1p3*p5p8-hbcm*p1p5*
     . p1p8+hbcm*p1p5*p5p8+hbcm*p1p8*p2p3+2*hbcm*p1p8*p2p5-hbcm*p1p8*
     . p3p4+hbcm*p1p8*p3p5-hbcm*p2p3*p5p8-2*hbcm*p2p5*p5p8+hbcm*p3p4*
     . p5p8-hbcm*p3p5*p5p8))+(8*p4p8*(-fb3*fmb+fb3*fmc+fb4*hbcm)+16*
     . p2p8*(fb3*fmb-fb3*fmc-fb4*hbcm)))
      b(12)=4*ccc*(-2*fb3*ffmcfmb*hbcm2+fb3*fmb2-fb3*fmc2+fb3*hbcm2-4
     . *fb3*p1p2-2*fb3*p1p3-2*fb3*p1p5+2*fb3*p2p3+4*fb3*p2p5-2*fb3*
     . p3p4+2*fb3*p3p5)
      ans2=(8*p4p8*(fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+16*p2p8*(fb3
     . *ffmcfmb*hbcm2-fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*(fb3*
     . ffmcfmb*hbcm2*p1p8-fb3*ffmcfmb*hbcm2*p5p8+2*fb3*p1p3*p2p8-fb3*
     . p1p3*p4p8-2*fb3*p1p8*p2p3+fb3*p1p8*p3p4-fb3*p2p3*p4p8+2*fb3*
     . p2p3*p5p8+fb3*p2p8*p3p4-2*fb3*p2p8*p3p5-fb3*p3p4*p5p8+fb3*p3p5
     . *p4p8))
      ans1=w1*(4*(p4p8*p3p4)*(-fmb*hbcm-fmc*hbcm)+4*(p4p8*p2p3)*(2*
     . fmb*hbcm+2*fmc*hbcm-hbcm2)+2*p4p8*(-2*ffmcfmb*fmb*hbcm3-2*
     . ffmcfmb*fmc*hbcm3+fmb2*hbcm2-fmc2*hbcm2+hbcm4)+4*(-2*hbcm2*
     . p1p2*p4p8-hbcm2*p1p3*p4p8-hbcm2*p1p5*p4p8+2*hbcm2*p2p5*p4p8+
     . hbcm2*p3p5*p4p8))+w15*(4*(p5p8*p3p4)*(fmb*hbcm+fmc*hbcm)+4*(
     . p5p8*p2p3)*(-2*fmb*hbcm-2*fmc*hbcm+hbcm2)+4*(p3p4*p1p8)*(-fmb*
     . hbcm-fmc*hbcm)+4*(p2p3*p1p8)*(2*fmb*hbcm+2*fmc*hbcm-hbcm2)+2*
     . p5p8*(2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-fmb2*hbcm2+fmc2*
     . hbcm2-hbcm4)+2*p1p8*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3+
     . fmb2*hbcm2-fmc2*hbcm2+hbcm4)+4*(-2*hbcm2*p1p2*p1p8+2*hbcm2*
     . p1p2*p5p8-hbcm2*p1p3*p1p8+hbcm2*p1p3*p5p8-hbcm2*p1p5*p1p8+
     . hbcm2*p1p5*p5p8+2*hbcm2*p1p8*p2p5+hbcm2*p1p8*p3p5-2*hbcm2*p2p5
     . *p5p8-hbcm2*p3p5*p5p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(8*p3p4*(fb3*fmb-fb3*fmc)+8*p2p3*(-2*fb3*fmb+2*fb3*
     . fmc+fb4*hbcm)+4*(2*fb3*ffmcfmb*fmb*hbcm2-2*fb3*ffmcfmb*fmc*
     . hbcm2-fb4*fmb2*hbcm+fb4*fmc2*hbcm+4*fb4*hbcm*p1p2+2*fb4*hbcm*
     . p1p3+2*fb4*hbcm*p1p5-4*fb4*hbcm*p2p5-2*fb4*hbcm*p3p5-fb4*hbcm3
     . ))
      b(15)=ccc*(w1*(4*(p4p8*p3p6)*(3*fmb*hbcm+3*fmc*hbcm-hbcm2)+8*(
     . hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+w15*(4*(p5p8*p3p6)*(-3*fmb*
     . hbcm-3*fmc*hbcm+hbcm2)+4*(p3p6*p1p8)*(3*fmb*hbcm+3*fmc*hbcm-
     . hbcm2)+8*(hbcm2*p1p8*p4p6+hbcm2*p1p8*p5p6-hbcm2*p4p6*p5p8-
     . hbcm2*p5p6*p5p8))+(8*p6p8*(2*fb3*ffmcfmb*hbcm2+fb3*hbcm2-3*fb4
     . *fmb*hbcm-3*fb4*fmc*hbcm)+16*(fb3*p1p3*p6p8-fb3*p1p8*p3p6-fb3*
     . p2p3*p6p8+fb3*p2p8*p3p6+3*fb3*p3p4*p6p8-fb3*p3p5*p6p8-3*fb3*
     . p3p6*p4p8+fb3*p3p6*p5p8)))
      b(16)=ccc*(8*w1*(-hbcm*p3p6*p4p8-hbcm*p4p6*p4p8-hbcm*p4p8*p5p6)
     . +8*w15*(-hbcm*p1p8*p3p6-hbcm*p1p8*p4p6-hbcm*p1p8*p5p6+hbcm*
     . p3p6*p5p8+hbcm*p4p6*p5p8+hbcm*p5p6*p5p8)+24*p6p8*(-fb3*fmb+fb3
     . *fmc+fb4*hbcm))
      b(17)=ccc*(w1*(2*p4p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-
     . 3*hbcm3)+4*(3*hbcm*p1p3*p4p8+hbcm*p2p3*p4p8+hbcm*p3p4*p4p8-3*
     . hbcm*p3p5*p4p8))+w15*(2*p5p8*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*
     . fmc*hbcm2+3*hbcm3)+2*p1p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+4*(3*hbcm*p1p3*p1p8-3*hbcm*p1p3*p5p8+hbcm*p1p8*
     . p2p3+hbcm*p1p8*p3p4-3*hbcm*p1p8*p3p5-hbcm*p2p3*p5p8-hbcm*p3p4*
     . p5p8+3*hbcm*p3p5*p5p8))+8*(-3*fb4*hbcm*p1p8-fb4*hbcm*p2p8-fb4*
     . hbcm*p4p8+3*fb4*hbcm*p5p8))
      b(18)=ccc*(w1*(4*(p4p8*p3p7)*(-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*
     . hbcm+2*hbcm2)+4*(-hbcm2*p4p7*p4p8-hbcm2*p4p8*p5p7))+w15*(4*(
     . p5p8*p3p7)*(3*ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*(
     . p3p7*p1p8)*(-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+2*hbcm2)+4*(-
     . hbcm2*p1p8*p4p7-hbcm2*p1p8*p5p7+hbcm2*p4p7*p5p8+hbcm2*p5p7*
     . p5p8))+(2*p7p8*(-4*fb3*ffmcfmb*hbcm2+4*fb3*hbcm2+4*fb4*fmb*
     . hbcm+8*fb4*fmc*hbcm+3*hbcm2)+16*(-fb3*p1p3*p7p8+fb3*p1p8*p3p7-
     . fb3*p2p3*p7p8+fb3*p2p8*p3p7-fb3*p3p4*p7p8+fb3*p3p5*p7p8+fb3*
     . p3p7*p4p8-fb3*p3p7*p5p8)))
      b(19)=ccc*(6*w1*p4p8*(-fmb*hbcm-fmc*hbcm+hbcm2)+w15*(6*p5p8*(
     . fmb*hbcm+fmc*hbcm-hbcm2)+6*p1p8*(-fmb*hbcm-fmc*hbcm+hbcm2))+8*
     . (3*fb3*p1p8+fb3*p2p8+fb3*p4p8-3*fb3*p5p8))
      b(20)=ccc*(8*w1*(-ffmcfmb*hbcm*p3p7*p4p8-hbcm*p4p8*p5p7)+8*w15*
     . (-ffmcfmb*hbcm*p1p8*p3p7+ffmcfmb*hbcm*p3p7*p5p8-hbcm*p1p8*p5p7
     . +hbcm*p5p7*p5p8)+2*p7p8*(-4*fb3*fmb+8*fb3*fmc+4*fb4*ffmcfmb*
     . hbcm+4*fb4*hbcm+hbcm))
      b(21)=12*ccc*(-fb3*fmb+fb3*fmc+fb4*hbcm)
      b(22)=4*ccc*(-2*fb3*ffmcfmb*hbcm2+3*fb3*hbcm2-6*fb3*p1p3-2*fb3*
     . p2p3-2*fb3*p3p4+6*fb3*p3p5+3*fb4*fmb*hbcm+3*fb4*fmc*hbcm)
      b(23)=ccc*(2*p3p7*(4*fb3*fmb-8*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4
     . *hbcm+hbcm)+16*fb4*hbcm*p5p7)
      b(24)=ccc*(8*p3p6*(3*fb3*fmb-3*fb3*fmc-fb4*hbcm)+16*(fb4*hbcm*
     . p4p6+fb4*hbcm*p5p6))
      b(25)=ccc*(2*p3p7*(-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+
     . hbcm)+8*p4p7*(fb3*fmb-fb3*fmc)-8*fb4*hbcm*p5p7)
      b(26)=ccc*(8*p5p6*(fb3*fmb-fb3*fmc-fb4*hbcm)+8*p4p6*(fb3*fmb-
     . fb3*fmc-fb4*hbcm)+8*p3p6*(fb3*fmb-fb3*fmc-fb4*hbcm))
      b(27)=ccc*(8*p6p7*(-2*fb3*ffmcfmb*hbcm2-fb3*hbcm2+3*fb4*fmb*
     . hbcm+3*fb4*fmc*hbcm)+16*(-fb3*p1p3*p6p7+fb3*p2p3*p6p7-3*fb3*
     . p3p4*p6p7+fb3*p3p5*p6p7+2*fb3*p3p6*p3p7+3*fb3*p3p6*p4p7-fb3*
     . p3p6*p5p7+fb3*p3p7*p4p6+fb3*p3p7*p5p6))
      b(28)=ccc*(12*w1*(p6p7*p4p8)*(fmb*hbcm+fmc*hbcm-hbcm2)+w15*(12*
     . (p6p7*p5p8)*(-fmb*hbcm-fmc*hbcm+hbcm2)+12*(p6p7*p1p8)*(fmb*
     . hbcm+fmc*hbcm-hbcm2))+16*(-fb3*p1p8*p6p7+fb3*p2p8*p6p7+fb3*
     . p3p6*p7p8+fb3*p3p7*p6p8+fb3*p4p6*p7p8+3*fb3*p4p7*p6p8-3*fb3*
     . p4p8*p6p7+fb3*p5p6*p7p8-fb3*p5p7*p6p8+fb3*p5p8*p6p7))
      b(29)=16*ccc*(fb3*p3p6+fb3*p4p6+fb3*p5p6)
      b(30)=8*ccc*(fb3*p3p7-fb3*p4p7+3*fb3*p5p7)
      ans1=(w1*(4*(p6p7*p4p8)*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2+hbcm3)+8*(p4p8*p3p7*p3p6)*(2*ffmcfmb*hbcm-3*hbcm)+8*(
     . hbcm*p1p3*p4p8*p6p7-hbcm*p2p3*p4p8*p6p7+3*hbcm*p3p4*p4p8*p6p7-
     . hbcm*p3p5*p4p8*p6p7-2*hbcm*p3p6*p4p7*p4p8-hbcm*p3p7*p4p6*p4p8-
     . hbcm*p3p7*p4p8*p5p6))+w15*(4*(p6p7*p5p8)*(-2*ffmcfmb*hbcm3+3*
     . fmb*hbcm2-3*fmc*hbcm2-hbcm3)+4*(p6p7*p1p8)*(2*ffmcfmb*hbcm3-3*
     . fmb*hbcm2+3*fmc*hbcm2+hbcm3)+8*(p5p8*p3p7*p3p6)*(-2*ffmcfmb*
     . hbcm+3*hbcm)+8*(p3p7*p3p6*p1p8)*(2*ffmcfmb*hbcm-3*hbcm)+8*(
     . hbcm*p1p3*p1p8*p6p7-hbcm*p1p3*p5p8*p6p7-hbcm*p1p8*p2p3*p6p7+3*
     . hbcm*p1p8*p3p4*p6p7-hbcm*p1p8*p3p5*p6p7-2*hbcm*p1p8*p3p6*p4p7-
     . hbcm*p1p8*p3p7*p4p6-hbcm*p1p8*p3p7*p5p6+hbcm*p2p3*p5p8*p6p7-3*
     . hbcm*p3p4*p5p8*p6p7+hbcm*p3p5*p5p8*p6p7+2*hbcm*p3p6*p4p7*p5p8+
     . hbcm*p3p7*p4p6*p5p8+hbcm*p3p7*p5p6*p5p8))+(4*(p6p8*p3p7)*(-4*
     . fb3*fmb+8*fb3*fmc-4*fb4*ffmcfmb*hbcm+12*fb4*hbcm-hbcm)+4*(p7p8
     . *p3p6)*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+16*(-fb4*
     . hbcm*p1p8*p6p7+fb4*hbcm*p2p8*p6p7+fb4*hbcm*p4p6*p7p8+2*fb4*
     . hbcm*p4p7*p6p8-3*fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p6*p7p8+fb4*
     . hbcm*p5p8*p6p7)))
      ans=ccc*ans1
      b(31)=ans
      b(32)=24*ccc*p6p7*(-fb3*fmb+fb3*fmc+fb4*hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.8181818181818182d0*b(n)
         c(n,2)=c(n,2)+0.1512818716977898d0*b(n)
         c(n,3)=c(n,3)+0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp30_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3+2*
     . ffmcfmb*p3p4+fmc2-2*p1p4))
      ans=ccc*(w2*(4*(p5p8*p5p7)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+4*(p5p8*
     . p3p7)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+
     . fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*hbcm2*p4p7*p5p8)+w5*(4*(p5p7*
     . p4p8)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+4*(p5p7*p1p8)*(fmb*hbcm+fmc
     . *hbcm-2*hbcm2)+4*(p4p8*p3p7)*(2*ffmcfmb*fmb*hbcm+2*ffmcfmb*fmc
     . *hbcm-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+2*hbcm2)+4*(p3p7*
     . p1p8)*(-2*ffmcfmb*fmb*hbcm-2*ffmcfmb*fmc*hbcm+3*ffmcfmb*hbcm2+
     . fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*(hbcm2*p1p8*p4p7-hbcm2*p4p7*
     . p4p8))+(2*p7p8*(-4*fb3*ffmcfmb*hbcm2-4*fb3*fmb*fmc-4*fb3*fmc2-
     . 4*fb4*ffmcfmb*fmb*hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmc*hbcm-
     . fmb*hbcm-fmc*hbcm+3*hbcm2)+16*(fb3*p1p3*p7p8+fb3*p1p4*p7p8-fb3
     . *p1p8*p3p7-fb3*p1p8*p5p7+fb3*p2p5*p7p8+fb3*p2p8*p3p7+fb3*p2p8*
     . p5p7-fb3*p3p4*p7p8+fb3*p3p7*p4p8-fb3*p4p7*p5p8+fb3*p4p8*p5p7))
     . )
      b(3)=ans
      b(4)=ccc*(2*p3p7*(4*fb3*fmb*fmc+4*fb3*fmc2-4*fb4*ffmcfmb*fmb*
     . hbcm-4*fb4*ffmcfmb*fmc*hbcm+4*fb4*fmb*hbcm+4*fb4*fmc*hbcm-fmb*
     . hbcm-fmc*hbcm)+8*p5p7*(-2*fb3*ffmcfmb*hbcm2+fb4*fmb*hbcm+fb4*
     . fmc*hbcm)+8*(fb3*hbcm2*p4p7+2*fb3*p1p3*p5p7-2*fb3*p1p4*p3p7-2*
     . fb3*p2p3*p4p7+4*fb3*p2p3*p5p7-2*fb3*p2p5*p3p7-2*fb3*p3p4*p5p7+
     . 2*fb3*p3p5*p4p7))
      ans3=(4*(p5p8*p3p7)*(4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-
     . hbcm)+4*(p7p8*p3p5)*(-4*fb3*fmc-4*fb4*ffmcfmb*hbcm-hbcm)+4*(
     . p3p7*p2p8)*(-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm+hbcm)+4*(
     . p7p8*p2p3)*(4*fb3*fmc+4*fb4*ffmcfmb*hbcm+hbcm)+16*(p4p8*p3p7)*
     . (-fb3*fmb+fb3*fmc)+16*(p7p8*p3p4)*(fb3*fmb-fb3*fmc)+16*(p3p7*
     . p1p8)*(fb3*fmb-fb3*fmc)+16*(p7p8*p1p3)*(-fb3*fmb+fb3*fmc)+2*
     . p7p8*(4*fb3*ffmcfmb*fmb*hbcm2-4*fb3*ffmcfmb*fmc*hbcm2-4*fb3*
     . fmc*hbcm2-4*fb4*ffmcfmb*hbcm3-4*fb4*fmb*fmc*hbcm+4*fb4*fmc2*
     . hbcm-3*fmb*hbcm2+3*fmc*hbcm2-hbcm3)+16*(-fb4*hbcm*p1p4*p7p8+
     . fb4*hbcm*p1p8*p5p7-fb4*hbcm*p2p5*p7p8+fb4*hbcm*p2p8*p5p7-fb4*
     . hbcm*p4p8*p5p7+fb4*hbcm*p5p7*p5p8))
      ans2=w5*(4*(p4p8*p4p7)*(fmb*hbcm2-fmc*hbcm2)+4*(p4p7*p1p8)*(-
     . fmb*hbcm2+fmc*hbcm2)+4*(p5p7*p4p8)*(2*ffmcfmb*hbcm3-2*fmb*
     . hbcm2+2*fmc*hbcm2-hbcm3)+4*(p5p7*p1p8)*(-2*ffmcfmb*hbcm3+2*fmb
     . *hbcm2-2*fmc*hbcm2+hbcm3)+8*(p4p8*p3p7*p3p5)*(2*ffmcfmb*hbcm-
     . hbcm)+8*(p4p8*p3p7*p2p3)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*p3p5*
     . p1p8)*(-2*ffmcfmb*hbcm+hbcm)+8*(p3p7*p2p3*p1p8)*(2*ffmcfmb*
     . hbcm-hbcm)+4*(p4p8*p3p7)*(3*ffmcfmb*fmb*hbcm2-3*ffmcfmb*fmc*
     . hbcm2+2*ffmcfmb*hbcm3+fmb*fmc*hbcm-2*fmb*hbcm2+2*fmc*hbcm2-
     . fmc2*hbcm-hbcm3)+4*(p3p7*p1p8)*(-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb
     . *fmc*hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm+2*fmb*hbcm2-2*fmc*
     . hbcm2+fmc2*hbcm+hbcm3)+8*(hbcm*p1p3*p1p8*p5p7-hbcm*p1p3*p4p8*
     . p5p7-hbcm*p1p4*p1p8*p3p7+hbcm*p1p4*p3p7*p4p8+hbcm*p1p8*p2p3*
     . p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p1p8*p3p4*p5p7+hbcm*p1p8*p3p5*
     . p5p7-hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p3p7*p4p8+hbcm*p3p4*p4p8*
     . p5p7-hbcm*p3p5*p4p8*p5p7))+ans3
      ans1=w2*(4*(p5p8*p4p7)*(-fmb*hbcm2+fmc*hbcm2)+4*(p5p8*p5p7)*(-2
     . *ffmcfmb*hbcm3+2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*(p5p8*p3p7*
     . p3p5)*(-2*ffmcfmb*hbcm+hbcm)+8*(p5p8*p3p7*p2p3)*(2*ffmcfmb*
     . hbcm-hbcm)+4*(p5p8*p3p7)*(-3*ffmcfmb*fmb*hbcm2+3*ffmcfmb*fmc*
     . hbcm2-2*ffmcfmb*hbcm3-fmb*fmc*hbcm+2*fmb*hbcm2-2*fmc*hbcm2+
     . fmc2*hbcm+hbcm3)+8*(hbcm*p1p3*p5p7*p5p8-hbcm*p1p4*p3p7*p5p8+
     . hbcm*p2p3*p5p7*p5p8-hbcm*p2p5*p3p7*p5p8-hbcm*p3p4*p5p7*p5p8+
     . hbcm*p3p5*p5p7*p5p8))+ans2
      ans=ccc*ans1
      b(5)=ans
      b(11)=ccc*(w2*(2*p5p8*(-2*ffmcfmb*hbcm3+fmb2*hbcm-fmc2*hbcm+
     . hbcm3)+4*(hbcm*p1p3*p5p8+hbcm*p1p4*p5p8+hbcm*p2p3*p5p8+2*hbcm*
     . p2p5*p5p8-hbcm*p3p4*p5p8+hbcm*p3p5*p5p8))+w5*(2*p4p8*(2*
     . ffmcfmb*hbcm3-fmb2*hbcm+fmc2*hbcm-hbcm3)+2*p1p8*(-2*ffmcfmb*
     . hbcm3+fmb2*hbcm-fmc2*hbcm+hbcm3)+4*(hbcm*p1p3*p1p8-hbcm*p1p3*
     . p4p8+hbcm*p1p4*p1p8-hbcm*p1p4*p4p8+hbcm*p1p8*p2p3+2*hbcm*p1p8*
     . p2p5-hbcm*p1p8*p3p4+hbcm*p1p8*p3p5-hbcm*p2p3*p4p8-2*hbcm*p2p5*
     . p4p8+hbcm*p3p4*p4p8-hbcm*p3p5*p4p8))+(8*p4p8*(fb3*fmb-fb3*fmc-
     . fb4*hbcm)+16*p2p8*(-fb3*fmb+fb3*fmc+fb4*hbcm)+8*p1p8*(-fb3*fmb
     . +fb3*fmc+fb4*hbcm)))
      b(12)=4*ccc*(2*fb3*ffmcfmb*hbcm2-fb3*fmb2+fb3*fmc2-fb3*hbcm2-2*
     . fb3*p1p3-2*fb3*p1p4-2*fb3*p2p3-4*fb3*p2p5+2*fb3*p3p4-2*fb3*
     . p3p5)
      ans2=(8*p4p8*(-fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+8*p1p8*(fb3
     . *hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+16*p2p8*(-fb3*ffmcfmb*hbcm2+
     . fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+16*(fb3*ffmcfmb*hbcm2*
     . p5p8+fb3*p1p3*p2p8-fb3*p1p3*p5p8-fb3*p1p8*p2p3+fb3*p1p8*p3p5+
     . fb3*p2p3*p4p8-2*fb3*p2p3*p5p8-fb3*p2p8*p3p4+2*fb3*p2p8*p3p5+
     . fb3*p3p4*p5p8-fb3*p3p5*p4p8))
      ans1=w2*(4*(p5p8*p3p4)*(-fmb*hbcm-fmc*hbcm)+4*(p5p8*p2p3)*(2*
     . fmb*hbcm+2*fmc*hbcm-hbcm2)+4*(p5p8*p1p3)*(fmb*hbcm+fmc*hbcm)+2
     . *p5p8*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3+fmb2*hbcm2-
     . fmc2*hbcm2+hbcm4)+4*(hbcm2*p1p4*p5p8+2*hbcm2*p2p5*p5p8+hbcm2*
     . p3p5*p5p8))+w5*(4*(p4p8*p3p4)*(fmb*hbcm+fmc*hbcm)+4*(p4p8*p2p3
     . )*(-2*fmb*hbcm-2*fmc*hbcm+hbcm2)+4*(p3p4*p1p8)*(-fmb*hbcm-fmc*
     . hbcm)+4*(p2p3*p1p8)*(2*fmb*hbcm+2*fmc*hbcm-hbcm2)+4*(p4p8*p1p3
     . )*(-fmb*hbcm-fmc*hbcm)+4*(p1p8*p1p3)*(fmb*hbcm+fmc*hbcm)+2*
     . p4p8*(2*ffmcfmb*fmb*hbcm3+2*ffmcfmb*fmc*hbcm3-fmb2*hbcm2+fmc2*
     . hbcm2-hbcm4)+2*p1p8*(-2*ffmcfmb*fmb*hbcm3-2*ffmcfmb*fmc*hbcm3+
     . fmb2*hbcm2-fmc2*hbcm2+hbcm4)+4*(hbcm2*p1p4*p1p8-hbcm2*p1p4*
     . p4p8+2*hbcm2*p1p8*p2p5+hbcm2*p1p8*p3p5-2*hbcm2*p2p5*p4p8-hbcm2
     . *p3p5*p4p8))+ans2
      ans=ccc*ans1
      b(13)=ans
      b(14)=ccc*(8*p3p4*(-fb3*fmb+fb3*fmc)+8*p2p3*(2*fb3*fmb-2*fb3*
     . fmc-fb4*hbcm)+8*p1p3*(fb3*fmb-fb3*fmc)+4*(-2*fb3*ffmcfmb*fmb*
     . hbcm2+2*fb3*ffmcfmb*fmc*hbcm2+fb4*fmb2*hbcm-fb4*fmc2*hbcm+2*
     . fb4*hbcm*p1p4+4*fb4*hbcm*p2p5+2*fb4*hbcm*p3p5+fb4*hbcm3))
      b(17)=ccc*(w2*(2*p5p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*hbcm2-
     . 3*hbcm3)+4*(-hbcm*p1p3*p5p8+hbcm*p2p3*p5p8+hbcm*p3p4*p5p8-3*
     . hbcm*p3p5*p5p8))+w5*(2*p4p8*(-2*ffmcfmb*hbcm3+3*fmb*hbcm2-3*
     . fmc*hbcm2+3*hbcm3)+2*p1p8*(2*ffmcfmb*hbcm3-3*fmb*hbcm2+3*fmc*
     . hbcm2-3*hbcm3)+4*(-hbcm*p1p3*p1p8+hbcm*p1p3*p4p8+hbcm*p1p8*
     . p2p3+hbcm*p1p8*p3p4-3*hbcm*p1p8*p3p5-hbcm*p2p3*p4p8-hbcm*p3p4*
     . p4p8+3*hbcm*p3p5*p4p8))+8*(-fb4*hbcm*p1p8+fb4*hbcm*p2p8+fb4*
     . hbcm*p4p8-3*fb4*hbcm*p5p8))
      b(18)=ccc*(w2*(4*(p5p8*p3p7)*(-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*
     . hbcm+2*hbcm2)+4*(-hbcm2*p4p7*p5p8-hbcm2*p5p7*p5p8))+w5*(4*(
     . p4p8*p3p7)*(3*ffmcfmb*hbcm2+fmb*hbcm+2*fmc*hbcm-2*hbcm2)+4*(
     . p3p7*p1p8)*(-3*ffmcfmb*hbcm2-fmb*hbcm-2*fmc*hbcm+2*hbcm2)+4*(-
     . hbcm2*p1p8*p4p7-hbcm2*p1p8*p5p7+hbcm2*p4p7*p4p8+hbcm2*p4p8*
     . p5p7))+(2*p7p8*(4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2-4*fb4*fmb*hbcm
     . -8*fb4*fmc*hbcm-3*hbcm2)+16*(-fb3*p1p3*p7p8+fb3*p1p8*p3p7+fb3*
     . p2p3*p7p8-fb3*p2p8*p3p7+fb3*p3p4*p7p8-fb3*p3p5*p7p8-fb3*p3p7*
     . p4p8+fb3*p3p7*p5p8)))
      b(19)=ccc*(6*w2*p5p8*(-fmb*hbcm-fmc*hbcm+hbcm2)+w5*(6*p4p8*(fmb
     . *hbcm+fmc*hbcm-hbcm2)+6*p1p8*(-fmb*hbcm-fmc*hbcm+hbcm2))+8*(
     . fb3*p1p8-fb3*p2p8-fb3*p4p8+3*fb3*p5p8))
      b(20)=ccc*(8*w2*(-ffmcfmb*hbcm*p3p7*p5p8-hbcm*p5p7*p5p8)+8*w5*(
     . -ffmcfmb*hbcm*p1p8*p3p7+ffmcfmb*hbcm*p3p7*p4p8-hbcm*p1p8*p5p7+
     . hbcm*p4p8*p5p7)+2*p7p8*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm
     . -4*fb4*hbcm-hbcm))
      b(21)=12*ccc*(fb3*fmb-fb3*fmc-fb4*hbcm)
      b(22)=4*ccc*(2*fb3*ffmcfmb*hbcm2-3*fb3*hbcm2-2*fb3*p1p3+2*fb3*
     . p2p3+2*fb3*p3p4-6*fb3*p3p5-3*fb4*fmb*hbcm-3*fb4*fmc*hbcm)
      b(23)=ccc*(2*p3p7*(-4*fb3*fmb+8*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*
     . fb4*hbcm-hbcm)-16*fb4*hbcm*p5p7)
      b(25)=ccc*(2*p3p7*(4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm-hbcm
     . )+8*p4p7*(-fb3*fmb+fb3*fmc)+8*fb4*hbcm*p5p7)
      b(30)=8*ccc*(-fb3*p3p7+fb3*p4p7-3*fb3*p5p7)
      do 200 n=1,32 
         c(n,2)=c(n,2)+0.8320502943378437d0*b(n)
         c(n,3)=c(n,3)-0.1869893980016914d0*b(n)
200   continue
      return
	end 
 
      subroutine amp3_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(26) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,26 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p1p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(3)=w2*ccc*(8*(p5p8*p5p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(p5p8
     . *p3p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2))
      b(4)=ccc*(16*p5p7*(-fb4*fmb*hbcm-fb4*fmc*hbcm)+16*p3p7*(-fb4*
     . fmb*hbcm-fb4*fmc*hbcm))
      b(5)=ccc*(w2*(8*(p5p8*p5p7)*(-2*fmb*hbcm2+2*fmc*hbcm2-hbcm3)+8*
     . (p5p8*p3p7)*(-2*fmb*hbcm2+fmb2*hbcm+2*fmc*hbcm2-fmc2*hbcm)-16*
     . hbcm*p3p5*p5p7*p5p8)+(16*p7p8*(fb4*fmb2*hbcm-fb4*fmc2*hbcm+fb4
     . *hbcm3)+32*(-fb4*hbcm*p2p3*p7p8+fb4*hbcm*p2p8*p3p7+fb4*hbcm*
     . p3p5*p7p8-fb4*hbcm*p3p7*p5p8-fb4*hbcm*p5p7*p5p8)))
      b(11)=ccc*(w2*(4*p5p8*(-fmb2*hbcm+fmc2*hbcm-hbcm3)-8*hbcm*p3p5*
     . p5p8)-16*fb4*hbcm*p2p8)
      b(13)=ccc*(w2*(8*p5p8*(-fmb2*hbcm2+fmc2*hbcm2-hbcm4)+8*(p5p8*
     . p2p3)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)-16*hbcm2*p3p5*p5p8)+16*p2p8
     . *(-fb4*fmb*hbcm-fb4*fmc*hbcm))
      b(14)=8*ccc*(-fb4*fmb2*hbcm+fb4*fmc2*hbcm+2*fb4*hbcm*p2p3-2*fb4
     . *hbcm*p3p5-fb4*hbcm3)
      b(17)=ccc*(w2*(4*p5p8*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*hbcm*
     . p3p5*p5p8)+16*fb4*hbcm*p5p8)
      b(18)=ccc*(8*w2*(p5p8*p3p7)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+16*p7p8
     . *(fb4*fmb*hbcm+fb4*fmc*hbcm))
      b(19)=4*ccc*w2*p5p8*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(20)=16*ccc*fb4*hbcm*p7p8
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(23)=-16*ccc*fb4*hbcm*p3p7
      b(25)=16*ccc*(-fb4*hbcm*p3p7-fb4*hbcm*p5p7)
      do 200 n=1,26 
         c(n,2)=c(n,2)-0.7396002616336388d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp6_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((4*p1p5*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*
     . ffmcfmb*p3p4+fmc2-2*p2p4))
      b(2)=ccc*(16*w11*(hbcm*p2p4*p2p8*p6p7-hbcm*p2p4*p4p8*p6p7+hbcm*
     . p2p8*p3p6*p4p7+hbcm*p2p8*p4p6*p4p7+hbcm*p2p8*p4p7*p5p6-hbcm*
     . p3p6*p4p7*p4p8-hbcm*p4p6*p4p7*p4p8-hbcm*p4p7*p4p8*p5p6)+32*fb4
     . *hbcm*p4p7*p6p8)
      b(3)=w11*ccc*(8*(p4p8*p4p7)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+8*(
     . p4p7*p2p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2))
      b(4)=16*ccc*p4p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(5)=ccc*(w11*(8*(p4p8*p4p7)*(-2*fmb*hbcm2+2*fmc*hbcm2-hbcm3)+8
     . *(p4p7*p2p8)*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+16*(-hbcm*p1p2*
     . p2p8*p3p7+hbcm*p1p2*p3p7*p4p8-hbcm*p1p3*p2p8*p4p7+hbcm*p1p3*
     . p4p7*p4p8-hbcm*p2p3*p2p8*p4p7-hbcm*p2p3*p2p8*p5p7+hbcm*p2p3*
     . p4p7*p4p8+hbcm*p2p3*p4p8*p5p7+hbcm*p2p5*p2p8*p3p7-hbcm*p2p5*
     . p3p7*p4p8+hbcm*p2p8*p3p5*p4p7-hbcm*p3p5*p4p7*p4p8))+32*(-fb4*
     . hbcm*p1p2*p7p8-fb4*hbcm*p1p8*p4p7+fb4*hbcm*p2p5*p7p8-fb4*hbcm*
     . p2p8*p4p7-fb4*hbcm*p2p8*p5p7+fb4*hbcm*p4p7*p5p8))
      b(8)=ccc*(16*w11*(-hbcm*p2p3*p2p8*p3p6-hbcm*p2p3*p2p8*p4p6-hbcm
     . *p2p3*p2p8*p5p6+hbcm*p2p3*p3p6*p4p8+hbcm*p2p3*p4p6*p4p8+hbcm*
     . p2p3*p4p8*p5p6-hbcm*p2p4*p2p8*p3p6+hbcm*p2p4*p3p6*p4p8)+32*(-
     . fb4*hbcm*p2p3*p6p8-fb4*hbcm*p2p4*p6p8-fb4*hbcm*p2p8*p4p6-fb4*
     . hbcm*p2p8*p5p6))
      b(9)=32*ccc*(fb4*hbcm*p2p4*p6p7+fb4*hbcm*p4p6*p4p7+fb4*hbcm*
     . p4p7*p5p6)
      b(10)=ccc*(w11*(16*(p4p8*p4p7*p3p6)*(fmb*hbcm+fmc*hbcm)+16*(
     . p4p7*p3p6*p2p8)*(-fmb*hbcm-fmc*hbcm)+16*(p6p7*p4p8*p2p3)*(-fmb
     . *hbcm-fmc*hbcm+2*hbcm2)+16*(p6p7*p2p8*p2p3)*(fmb*hbcm+fmc*hbcm
     . -2*hbcm2)+32*(-hbcm2*p2p4*p2p8*p6p7+hbcm2*p2p4*p4p8*p6p7-hbcm2
     . *p2p8*p4p6*p4p7-hbcm2*p2p8*p4p7*p5p6+hbcm2*p4p6*p4p7*p4p8+
     . hbcm2*p4p7*p4p8*p5p6))+32*(p6p8*p4p7)*(-fb4*fmb*hbcm-fb4*fmc*
     . hbcm))
      b(11)=ccc*(8*w11*(hbcm*p1p2*p2p8-hbcm*p1p2*p4p8-hbcm*p2p3*p2p8+
     . hbcm*p2p3*p4p8-hbcm*p2p5*p2p8+hbcm*p2p5*p4p8)-16*fb4*hbcm*p2p8
     . )
      b(13)=ccc*(w11*(8*(p4p8*p2p3)*(fmb*hbcm+fmc*hbcm)+8*(p2p8*p2p3)
     . *(-fmb*hbcm-fmc*hbcm)+16*(hbcm2*p1p2*p2p8-hbcm2*p1p2*p4p8-
     . hbcm2*p2p5*p2p8+hbcm2*p2p5*p4p8))+16*p2p8*(-fb4*fmb*hbcm-fb4*
     . fmc*hbcm))
      b(14)=16*ccc*(fb4*hbcm*p1p2-fb4*hbcm*p2p5)
      b(15)=ccc*(w11*(8*(p4p8*p3p6)*(fmb*hbcm+fmc*hbcm)+8*(p3p6*p2p8)
     . *(-fmb*hbcm-fmc*hbcm)+16*(-hbcm2*p2p8*p4p6-hbcm2*p2p8*p5p6+
     . hbcm2*p4p6*p4p8+hbcm2*p4p8*p5p6))+16*p6p8*(-fb4*fmb*hbcm-fb4*
     . fmc*hbcm))
      b(16)=ccc*(8*w11*(hbcm*p2p8*p3p6+hbcm*p2p8*p4p6+hbcm*p2p8*p5p6-
     . hbcm*p3p6*p4p8-hbcm*p4p6*p4p8-hbcm*p4p8*p5p6)+16*fb4*hbcm*p6p8
     . )
      b(17)=ccc*(w11*(4*p4p8*(-2*fmb*hbcm2+2*fmc*hbcm2-hbcm3)+4*p2p8*
     . (2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*(-hbcm*p1p3*p2p8+hbcm*p1p3*
     . p4p8+hbcm*p2p8*p3p5-hbcm*p3p5*p4p8))+16*(-fb4*hbcm*p1p8+fb4*
     . hbcm*p5p8))
      b(18)=ccc*(w11*(8*(p4p8*p3p7)*(-fmb*hbcm-fmc*hbcm)+8*(p3p7*p2p8
     . )*(fmb*hbcm+fmc*hbcm)+16*(hbcm2*p2p8*p4p7+hbcm2*p2p8*p5p7-
     . hbcm2*p4p7*p4p8-hbcm2*p4p8*p5p7))+16*p7p8*(fb4*fmb*hbcm+fb4*
     . fmc*hbcm))
      b(19)=w11*ccc*(4*p4p8*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+4*p2p8*(fmb*
     . hbcm+fmc*hbcm-2*hbcm2))
      b(20)=ccc*(8*w11*(hbcm*p2p8*p3p7+hbcm*p2p8*p4p7+hbcm*p2p8*p5p7-
     . hbcm*p3p7*p4p8-hbcm*p4p7*p4p8-hbcm*p4p8*p5p7)+16*fb4*hbcm*p7p8
     . )
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(23)=16*ccc*(fb4*hbcm*p4p7+fb4*hbcm*p5p7)
      b(24)=16*ccc*(fb4*hbcm*p4p6+fb4*hbcm*p5p6)
      b(25)=16*ccc*fb4*hbcm*p4p7
      b(27)=16*ccc*p6p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(28)=w11*ccc*(8*(p6p7*p4p8)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+8*(
     . p6p7*p2p8)*(-fmb*hbcm-fmc*hbcm+2*hbcm2))
      b(31)=ccc*(w11*(8*(p6p7*p4p8)*(-2*fmb*hbcm2+2*fmc*hbcm2+hbcm3)+
     . 8*(p6p7*p2p8)*(2*fmb*hbcm2-2*fmc*hbcm2-hbcm3)+16*(hbcm*p2p3*
     . p2p8*p6p7-hbcm*p2p3*p4p8*p6p7-hbcm*p2p8*p3p4*p6p7+hbcm*p2p8*
     . p3p6*p3p7+hbcm*p2p8*p3p7*p4p6+hbcm*p2p8*p3p7*p5p6+hbcm*p3p4*
     . p4p8*p6p7-hbcm*p3p6*p3p7*p4p8-hbcm*p3p7*p4p6*p4p8-hbcm*p3p7*
     . p4p8*p5p6))+32*(fb4*hbcm*p2p8*p6p7+fb4*hbcm*p3p7*p6p8+fb4*hbcm
     . *p4p6*p7p8-fb4*hbcm*p4p8*p6p7+fb4*hbcm*p5p6*p7p8))
      b(32)=16*ccc*fb4*hbcm*p6p7
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp7_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(fmb2-fmc2+hbcm2+2*p3p5))
      b(3)=8*ccc*w2*(p5p8*p4p7)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(4)=16*ccc*p4p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(5)=ccc*(w2*(8*(p5p8*p4p7)*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+16*
     . hbcm*p3p5*p4p7*p5p8)+32*(-fb4*hbcm*p2p3*p7p8+fb4*hbcm*p2p8*
     . p3p7+fb4*hbcm*p4p7*p5p8))
      b(10)=16*ccc*w2*(p6p7*p5p8*p2p3)*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(11)=-16*ccc*fb4*hbcm*p2p8
      b(13)=ccc*(8*w2*(p5p8*p2p3)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)+16*
     . p2p8*(-fb4*fmb*hbcm-fb4*fmc*hbcm))
      b(14)=16*ccc*fb4*hbcm*p2p3
      b(17)=ccc*(w2*(4*p5p8*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+8*hbcm*
     . p3p5*p5p8)+16*fb4*hbcm*p5p8)
      b(18)=ccc*(8*w2*(p5p8*p3p7)*(fmb*hbcm+fmc*hbcm-2*hbcm2)+16*p7p8
     . *(fb4*fmb*hbcm+fb4*fmc*hbcm))
      b(19)=4*ccc*w2*p5p8*(fmb*hbcm+fmc*hbcm-2*hbcm2)
      b(20)=16*ccc*fb4*hbcm*p7p8
      b(21)=8*ccc*fb4*hbcm
      b(22)=8*ccc*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(23)=-16*ccc*fb4*hbcm*p3p7
      b(25)=16*ccc*fb4*hbcm*p4p7
      b(27)=16*ccc*p6p7*(fb4*fmb*hbcm+fb4*fmc*hbcm)
      b(28)=8*ccc*w2*(p6p7*p5p8)*(-fmb*hbcm-fmc*hbcm+2*hbcm2)
      b(31)=ccc*(w2*(8*(p6p7*p5p8)*(2*fmb*hbcm2-2*fmc*hbcm2+hbcm3)+16
     . *hbcm*p3p5*p5p8*p6p7)+32*fb4*hbcm*p5p8*p6p7)
      b(32)=16*ccc*fb4*hbcm*p6p7
      do 200 n=1,32 
         c(n,1)=c(n,1)-0.7272727272727273d0*b(n)
         c(n,2)=c(n,2)+0.1344727748424798d0*b(n)
         c(n,3)=c(n,3)+0.1662127982237257d0*b(n)
200   continue
      return
	end 
 
      subroutine amp9_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p2p3+2*ffmcfmb*
     . p3p4+fmc2-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2+2*ffmcfmb*
     . p1p3-fmb2+hbcm2-2*p1p3))
      b(1)=32*ccc*(-2*fb3*p2p4*p6p7-fb3*p3p6*p4p7-2*fb3*p4p6*p4p7)
      b(2)=ccc*(w11*(16*(p4p8*p4p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(
     . p4p7*p3p6*p2p8)*(ffmcfmb*hbcm-hbcm)+16*(-hbcm*p2p4*p2p8*p6p7+
     . hbcm*p2p4*p4p8*p6p7-hbcm*p2p8*p4p6*p4p7+hbcm*p4p6*p4p7*p4p8))+
     . w13*(16*(p4p7*p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+16*(hbcm*p1p8*
     . p2p4*p6p7+hbcm*p1p8*p4p6*p4p7))+8*(p6p8*p4p7)*(4*fb3*fmb-4*fb3
     . *fmc+hbcm))
      b(3)=ccc*(w11*(8*(p4p8*p4p7)*(fmb*hbcm+fmc*hbcm)+8*(p4p7*p2p8)*
     . (-fmb*hbcm-fmc*hbcm))+8*w13*(p4p7*p1p8)*(fmb*hbcm+fmc*hbcm)+32
     . *(-2*fb3*p1p2*p7p8-fb3*p1p8*p4p7+2*fb3*p2p3*p7p8+2*fb3*p2p5*
     . p7p8+2*fb3*p4p7*p5p8))
      b(4)=ccc*(16*p4p7*(-2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . 2*fb3*p1p2*p3p7+fb3*p1p3*p4p7+2*fb3*p2p3*p4p7+2*fb3*p2p3*p5p7-
     . 2*fb3*p2p5*p3p7-2*fb3*p3p5*p4p7))
      b(5)=ccc*(w11*(8*(p4p8*p4p7)*(ffmcfmb*hbcm3+fmb*hbcm2)+8*(p4p7*
     . p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2)+16*(hbcm*p1p2*p2p8*p3p7-hbcm*
     . p1p2*p3p7*p4p8+hbcm*p2p3*p2p8*p4p7+hbcm*p2p3*p2p8*p5p7-hbcm*
     . p2p3*p4p7*p4p8-hbcm*p2p3*p4p8*p5p7-hbcm*p2p5*p2p8*p3p7+hbcm*
     . p2p5*p3p7*p4p8-hbcm*p2p8*p3p5*p4p7+hbcm*p3p5*p4p7*p4p8))+w13*(
     . 8*(p4p7*p1p8)*(ffmcfmb*hbcm3+fmb*hbcm2)+16*(-hbcm*p1p2*p1p8*
     . p3p7-hbcm*p1p8*p2p3*p4p7-hbcm*p1p8*p2p3*p5p7+hbcm*p1p8*p2p5*
     . p3p7+hbcm*p1p8*p3p5*p4p7))+(8*(p3p7*p2p8)*(4*fb3*fmb-8*fb3*fmc
     . -4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm)+8*(p7p8*p2p3)*(-4*fb3*fmb
     . +8*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm)+32*(fb4*hbcm*
     . p1p2*p7p8-fb4*hbcm*p2p5*p7p8+fb4*hbcm*p2p8*p4p7+fb4*hbcm*p2p8*
     . p5p7-fb4*hbcm*p4p7*p5p8)))
      b(6)=32*ccc*(-fb3*p2p4*p6p8-fb3*p2p8*p3p6-2*fb3*p2p8*p4p6)
      b(7)=32*ccc*(fb3*p2p3*p3p6+2*fb3*p2p3*p4p6+fb3*p2p4*p3p6)
      b(8)=ccc*(w11*(16*(p4p8*p3p6*p2p3)*(ffmcfmb*hbcm-hbcm)+16*(p3p6
     . *p2p8*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(hbcm*p2p3*p2p8*p4p6-hbcm*
     . p2p3*p4p6*p4p8))+w13*(16*(p3p6*p2p3*p1p8)*(ffmcfmb*hbcm-hbcm)-
     . 16*hbcm*p1p8*p2p3*p4p6)+(32*(p3p6*p2p8)*(fb3*fmb-fb3*fmc-fb4*
     . ffmcfmb*hbcm+fb4*hbcm)+8*(p6p8*p2p3)*(-4*fb3*fmb+4*fb3*fmc-
     . hbcm)+32*fb4*hbcm*p2p8*p4p6))
      b(9)=ccc*(32*(p4p7*p3p6)*(-fb3*fmb+fb3*fmc+fb4*ffmcfmb*hbcm-fb4
     . *hbcm)+32*(-fb4*hbcm*p2p4*p6p7-fb4*hbcm*p4p6*p4p7))
      b(10)=ccc*(w11*(16*(p6p7*p4p8*p2p3)*(fmb*hbcm+fmc*hbcm)+16*(
     . p6p7*p2p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+16*(p4p8*p4p7*p3p6)*(
     . ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+16*(p4p7*p3p6*p2p8)*(-ffmcfmb*
     . hbcm2+fmb*hbcm+hbcm2))+w13*(16*(p6p7*p2p3*p1p8)*(fmb*hbcm+fmc*
     . hbcm)+16*(p4p7*p3p6*p1p8)*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2))+(8*(
     . p6p8*p4p7)*(4*fb3*ffmcfmb*hbcm2-8*fb3*hbcm2+4*fb4*fmb*hbcm-
     . hbcm2)+64*(fb3*p1p3*p4p7*p6p8+fb3*p1p8*p2p3*p6p7-fb3*p1p8*p3p6
     . *p4p7+2*fb3*p2p3*p2p8*p6p7+fb3*p2p3*p4p7*p6p8-2*fb3*p2p3*p4p8*
     . p6p7+fb3*p2p8*p3p6*p3p7-fb3*p2p8*p3p6*p4p7+2*fb3*p2p8*p3p7*
     . p4p6-fb3*p3p4*p4p7*p6p8+fb3*p3p6*p4p7*p4p8)))
      b(11)=ccc*(8*w11*(-hbcm*p1p2*p2p8+hbcm*p1p2*p4p8+hbcm*p2p3*p2p8
     . -hbcm*p2p3*p4p8+hbcm*p2p5*p2p8-hbcm*p2p5*p4p8)+8*w13*(hbcm*
     . p1p2*p1p8-hbcm*p1p8*p2p3-hbcm*p1p8*p2p5)+4*p2p8*(-4*fb3*fmb+8*
     . fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm))
      b(12)=32*ccc*(fb3*p1p2-fb3*p2p3-fb3*p2p5)
      b(13)=ccc*(w11*(8*(p4p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+8*(p2p8*p2p3
     . )*(fmb*hbcm+fmc*hbcm))+8*w13*(p2p3*p1p8)*(-fmb*hbcm-fmc*hbcm)+
     . (16*p2p8*(2*fb3*hbcm2+fb4*fmb*hbcm+fb4*fmc*hbcm)+32*(-fb3*p1p3
     . *p2p8+fb3*p1p8*p2p3-2*fb3*p2p3*p5p8+2*fb3*p2p8*p3p5)))
      b(14)=ccc*(4*p2p3*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4
     . *hbcm+hbcm)+16*(-fb4*hbcm*p1p2+fb4*hbcm*p2p5))
      b(15)=ccc*(w11*(8*(p4p8*p3p6)*(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+8*
     . (p3p6*p2p8)*(-ffmcfmb*hbcm2+fmb*hbcm+hbcm2))+8*w13*(p3p6*p1p8)
     . *(ffmcfmb*hbcm2-fmb*hbcm-hbcm2)+(4*p6p8*(4*fb3*ffmcfmb*hbcm2-8
     . *fb3*hbcm2+4*fb4*fmb*hbcm-hbcm2)+32*(fb3*p1p3*p6p8-fb3*p1p8*
     . p3p6+fb3*p2p3*p6p8-fb3*p2p8*p3p6-fb3*p3p4*p6p8+fb3*p3p6*p4p8))
     . )
      b(16)=ccc*(w11*(8*(p4p8*p3p6)*(-ffmcfmb*hbcm+hbcm)+8*(p3p6*p2p8
     . )*(ffmcfmb*hbcm-hbcm)+8*(-hbcm*p2p8*p4p6+hbcm*p4p6*p4p8))+w13*
     . (8*(p3p6*p1p8)*(-ffmcfmb*hbcm+hbcm)+8*hbcm*p1p8*p4p6)+4*p6p8*(
     . 4*fb3*fmb-4*fb3*fmc+hbcm))
      b(17)=ccc*(w11*(4*p4p8*(ffmcfmb*hbcm3+fmb*hbcm2)+4*p2p8*(-
     . ffmcfmb*hbcm3-fmb*hbcm2)+8*(-hbcm*p2p8*p3p5+hbcm*p3p5*p4p8))+
     . w13*(4*p1p8*(ffmcfmb*hbcm3+fmb*hbcm2)+8*hbcm*p1p8*p3p5)-16*fb4
     . *hbcm*p5p8)
      b(18)=ccc*(w11*(8*(p4p8*p3p7)*(fmb*hbcm+fmc*hbcm)+8*(p3p7*p2p8)
     . *(-fmb*hbcm-fmc*hbcm))+8*w13*(p3p7*p1p8)*(fmb*hbcm+fmc*hbcm)+(
     . 16*p7p8*(-2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(fb3*p1p3*
     . p7p8-fb3*p1p8*p3p7-2*fb3*p3p5*p7p8+2*fb3*p3p7*p5p8)))
      b(19)=ccc*(w11*(4*p4p8*(fmb*hbcm+fmc*hbcm)+4*p2p8*(-fmb*hbcm-
     . fmc*hbcm))+4*w13*p1p8*(fmb*hbcm+fmc*hbcm)+16*(-fb3*p1p8+2*fb3*
     . p5p8))
      b(20)=ccc*(8*w11*(-hbcm*p2p8*p3p7-hbcm*p2p8*p4p7-hbcm*p2p8*p5p7
     . +hbcm*p3p7*p4p8+hbcm*p4p7*p4p8+hbcm*p4p8*p5p7)+8*w13*(hbcm*
     . p1p8*p3p7+hbcm*p1p8*p4p7+hbcm*p1p8*p5p7)+4*p7p8*(4*fb3*fmb-8*
     . fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm))
      b(21)=2*ccc*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p1p3-4*fb3*p3p5-fb4*fmb*hbcm-
     . fb4*fmc*hbcm)
      b(23)=ccc*(4*p3p7*(-4*fb3*fmb+8*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*
     . fb4*hbcm-hbcm)+16*(-fb4*hbcm*p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(16*p3p6*(-fb3*fmb+fb3*fmc+fb4*ffmcfmb*hbcm-fb4*hbcm)
     . -16*fb4*hbcm*p4p6)
      b(25)=4*ccc*p4p7*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p1p3*p6p7-2*fb3*p2p3*p6p7+2*fb3*p3p4*p6p7-fb3*p3p6*p3p7-2
     . *fb3*p3p7*p4p6))
      b(28)=ccc*(w11*(8*(p6p7*p4p8)*(-fmb*hbcm-fmc*hbcm)+8*(p6p7*p2p8
     . )*(fmb*hbcm+fmc*hbcm))+8*w13*(p6p7*p1p8)*(-fmb*hbcm-fmc*hbcm)+
     . 32*(-fb3*p1p8*p6p7-2*fb3*p2p8*p6p7-fb3*p3p6*p7p8-2*fb3*p4p6*
     . p7p8+2*fb3*p4p8*p6p7))
      b(29)=16*ccc*(-fb3*p3p6-2*fb3*p4p6)
      b(30)=32*ccc*(-fb3*p3p7-fb3*p4p7-fb3*p5p7)
      b(31)=ccc*(w11*(8*(p6p7*p4p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)
     . +8*(p6p7*p2p8)*(-ffmcfmb*hbcm3-fmb*hbcm2+2*hbcm3)+16*(p4p8*
     . p3p7*p3p6)*(-ffmcfmb*hbcm+hbcm)+16*(p3p7*p3p6*p2p8)*(ffmcfmb*
     . hbcm-hbcm)+16*(-hbcm*p1p3*p2p8*p6p7+hbcm*p1p3*p4p8*p6p7-hbcm*
     . p2p3*p2p8*p6p7+hbcm*p2p3*p4p8*p6p7+hbcm*p2p8*p3p4*p6p7-hbcm*
     . p2p8*p3p7*p4p6-hbcm*p3p4*p4p8*p6p7+hbcm*p3p7*p4p6*p4p8))+w13*(
     . 8*(p6p7*p1p8)*(ffmcfmb*hbcm3+fmb*hbcm2-2*hbcm3)+16*(p3p7*p3p6*
     . p1p8)*(-ffmcfmb*hbcm+hbcm)+16*(hbcm*p1p3*p1p8*p6p7+hbcm*p1p8*
     . p2p3*p6p7-hbcm*p1p8*p3p4*p6p7+hbcm*p1p8*p3p7*p4p6))+(8*(p6p8*
     . p3p7)*(4*fb3*fmb-4*fb3*fmc+hbcm)+32*(p7p8*p3p6)*(-fb3*fmb+fb3*
     . fmc+fb4*ffmcfmb*hbcm-fb4*hbcm)+32*(-fb4*hbcm*p1p8*p6p7-fb4*
     . hbcm*p2p8*p6p7-fb4*hbcm*p4p6*p7p8+fb4*hbcm*p4p8*p6p7)))
      b(32)=4*ccc*p6p7*(4*fb3*fmb-8*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      subroutine amp8_1p1(cc) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
      dimension b(32) 
#include "inclcon.h"
#include "inclppp.h"
#include "inclamp.h"
      do 100 n=1,32 
         b(n)=0.0d0 
100   continue 
      ccc=cc/((-2*p2p4)*(ffmcfmb**2*hbcm2-2*ffmcfmb*hbcm2-2*ffmcfmb*
     . p3p5+fmb2+hbcm2+2*p3p5)*(ffmcfmb**2*hbcm2-2*ffmcfmb*p1p3-fmc2)
     . )
      b(1)=32*ccc*(-2*fb3*p2p4*p6p7-fb3*p3p6*p4p7-2*fb3*p4p6*p4p7)
      b(2)=ccc*(w2*(16*(p5p8*p4p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(-hbcm
     . *p2p4*p5p8*p6p7-hbcm*p4p6*p4p7*p5p8))+w12*(16*(p4p7*p3p6*p1p8)
     . *(ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p8*p2p4*p6p7-hbcm*p1p8*p4p6*
     . p4p7))+32*(p6p8*p4p7)*(fb3*fmb+fb4*ffmcfmb*hbcm-fb4*hbcm))
      b(3)=ccc*(8*w2*(p5p8*p4p7)*(-fmb*hbcm-fmc*hbcm)+8*w12*(p4p7*
     . p1p8)*(-fmb*hbcm-fmc*hbcm)+32*(-2*fb3*p1p2*p7p8-fb3*p1p8*p4p7+
     . 2*fb3*p2p3*p7p8+2*fb3*p2p5*p7p8+2*fb3*p4p7*p5p8))
      b(4)=ccc*(16*p4p7*(-2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . 2*fb3*p1p2*p3p7+fb3*p1p3*p4p7+2*fb3*p2p3*p4p7+2*fb3*p2p3*p5p7-
     . 2*fb3*p2p5*p3p7-2*fb3*p3p5*p4p7))
      b(5)=ccc*(w2*(8*(p5p8*p4p7)*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+16
     . *(hbcm*p1p2*p3p7*p5p8+hbcm*p1p3*p4p7*p5p8+hbcm*p2p3*p4p7*p5p8+
     . hbcm*p2p3*p5p7*p5p8-hbcm*p2p5*p3p7*p5p8-hbcm*p3p5*p4p7*p5p8))+
     . w12*(8*(p4p7*p1p8)*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+16*(hbcm*
     . p1p2*p1p8*p3p7+hbcm*p1p3*p1p8*p4p7+hbcm*p1p8*p2p3*p4p7+hbcm*
     . p1p8*p2p3*p5p7-hbcm*p1p8*p2p5*p3p7-hbcm*p1p8*p3p5*p4p7))+(8*(
     . p3p7*p2p8)*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)+8*(
     . p7p8*p2p3)*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm)+32*(
     . fb4*hbcm*p1p2*p7p8+fb4*hbcm*p1p8*p4p7-fb4*hbcm*p2p5*p7p8+fb4*
     . hbcm*p2p8*p4p7+fb4*hbcm*p2p8*p5p7-fb4*hbcm*p4p7*p5p8)))
      b(6)=32*ccc*(-fb3*p2p4*p6p8-fb3*p2p8*p3p6-2*fb3*p2p8*p4p6)
      b(7)=32*ccc*(fb3*p2p3*p3p6+2*fb3*p2p3*p4p6+fb3*p2p4*p3p6)
      b(8)=ccc*(w2*(16*(p5p8*p3p6*p2p3)*(-ffmcfmb*hbcm+hbcm)+16*(hbcm
     . *p2p3*p4p6*p5p8+hbcm*p2p4*p3p6*p5p8))+w12*(16*(p3p6*p2p3*p1p8)
     . *(-ffmcfmb*hbcm+hbcm)+16*(hbcm*p1p8*p2p3*p4p6+hbcm*p1p8*p2p4*
     . p3p6))+(8*(p3p6*p2p8)*(4*fb3*fmb-hbcm)+32*(p6p8*p2p3)*(-fb3*
     . fmb-fb4*ffmcfmb*hbcm+fb4*hbcm)+32*(fb4*hbcm*p2p4*p6p8+fb4*hbcm
     . *p2p8*p4p6)))
      b(9)=ccc*(8*(p4p7*p3p6)*(-4*fb3*fmb+hbcm)+32*(-fb4*hbcm*p2p4*
     . p6p7-fb4*hbcm*p4p6*p4p7))
      b(10)=ccc*(w2*(16*(p6p7*p5p8*p2p3)*(-fmb*hbcm-fmc*hbcm)+16*(
     . p5p8*p4p7*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm))+w12*(16*(p6p7*p2p3*
     . p1p8)*(-fmb*hbcm-fmc*hbcm)+16*(p4p7*p3p6*p1p8)*(-ffmcfmb*hbcm2
     . +fmb*hbcm))+(8*(p6p8*p4p7)*(-4*fb3*ffmcfmb*hbcm2-4*fb3*hbcm2+4
     . *fb4*fmb*hbcm-hbcm2)+64*(fb3*p1p3*p4p7*p6p8+fb3*p1p8*p2p3*p6p7
     . -fb3*p1p8*p3p6*p4p7+2*fb3*p2p3*p2p8*p6p7+fb3*p2p3*p4p7*p6p8-2*
     . fb3*p2p3*p4p8*p6p7+fb3*p2p8*p3p6*p3p7-fb3*p2p8*p3p6*p4p7+2*fb3
     . *p2p8*p3p7*p4p6-fb3*p3p4*p4p7*p6p8+fb3*p3p6*p4p7*p4p8)))
      b(11)=ccc*(8*w2*(-hbcm*p1p2*p5p8+hbcm*p2p3*p5p8+hbcm*p2p5*p5p8)
     . +8*w12*(-hbcm*p1p2*p1p8+hbcm*p1p8*p2p3+hbcm*p1p8*p2p5)+4*p2p8*
     . (-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+4*fb4*hbcm+hbcm))
      b(12)=32*ccc*(fb3*p1p2-fb3*p2p3-fb3*p2p5)
      b(13)=ccc*(8*w2*(p5p8*p2p3)*(fmb*hbcm+fmc*hbcm)+8*w12*(p2p3*
     . p1p8)*(fmb*hbcm+fmc*hbcm)+(16*p2p8*(2*fb3*hbcm2+fb4*fmb*hbcm+
     . fb4*fmc*hbcm)+32*(-fb3*p1p3*p2p8+fb3*p1p8*p2p3-2*fb3*p2p3*p5p8
     . +2*fb3*p2p8*p3p5)))
      b(14)=ccc*(4*p2p3*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-hbcm)
     . +16*(-fb4*hbcm*p1p2+fb4*hbcm*p2p5))
      b(15)=ccc*(8*w2*(p5p8*p3p6)*(-ffmcfmb*hbcm2+fmb*hbcm)+8*w12*(
     . p3p6*p1p8)*(-ffmcfmb*hbcm2+fmb*hbcm)+(4*p6p8*(-4*fb3*ffmcfmb*
     . hbcm2-4*fb3*hbcm2+4*fb4*fmb*hbcm-hbcm2)+32*(fb3*p1p3*p6p8-fb3*
     . p1p8*p3p6+fb3*p2p3*p6p8-fb3*p2p8*p3p6-fb3*p3p4*p6p8+fb3*p3p6*
     . p4p8)))
      b(16)=ccc*(w2*(8*(p5p8*p3p6)*(ffmcfmb*hbcm-hbcm)-8*hbcm*p4p6*
     . p5p8)+w12*(8*(p3p6*p1p8)*(ffmcfmb*hbcm-hbcm)-8*hbcm*p1p8*p4p6)
     . +16*p6p8*(fb3*fmb+fb4*ffmcfmb*hbcm-fb4*hbcm))
      b(17)=ccc*(w2*(4*p5p8*(-ffmcfmb*hbcm3+fmc*hbcm2-hbcm3)+8*(hbcm*
     . p1p3*p5p8-hbcm*p3p5*p5p8))+w12*(4*p1p8*(-ffmcfmb*hbcm3+fmc*
     . hbcm2-hbcm3)+8*(hbcm*p1p3*p1p8-hbcm*p1p8*p3p5))+16*(fb4*hbcm*
     . p1p8-fb4*hbcm*p5p8))
      b(18)=ccc*(8*w2*(p5p8*p3p7)*(-fmb*hbcm-fmc*hbcm)+8*w12*(p3p7*
     . p1p8)*(-fmb*hbcm-fmc*hbcm)+(16*p7p8*(-2*fb3*hbcm2-fb4*fmb*hbcm
     . -fb4*fmc*hbcm)+32*(fb3*p1p3*p7p8-fb3*p1p8*p3p7-2*fb3*p3p5*p7p8
     . +2*fb3*p3p7*p5p8)))
      b(19)=ccc*(4*w2*p5p8*(-fmb*hbcm-fmc*hbcm)+4*w12*p1p8*(-fmb*hbcm
     . -fmc*hbcm)+16*(-fb3*p1p8+2*fb3*p5p8))
      b(20)=ccc*(8*w2*(-hbcm*p3p7*p5p8-hbcm*p4p7*p5p8-hbcm*p5p7*p5p8)
     . +8*w12*(-hbcm*p1p8*p3p7-hbcm*p1p8*p4p7-hbcm*p1p8*p5p7)+4*p7p8*
     . (8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-hbcm))
      b(21)=2*ccc*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*hbcm-
     . hbcm)
      b(22)=8*ccc*(-2*fb3*hbcm2+2*fb3*p1p3-4*fb3*p3p5-fb4*fmb*hbcm-
     . fb4*fmc*hbcm)
      b(23)=ccc*(4*p3p7*(-8*fb3*fmb+4*fb3*fmc-4*fb4*ffmcfmb*hbcm+hbcm
     . )+16*(-fb4*hbcm*p4p7-fb4*hbcm*p5p7))
      b(24)=ccc*(4*p3p6*(-4*fb3*fmb+hbcm)-16*fb4*hbcm*p4p6)
      b(25)=4*ccc*p4p7*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)
      b(27)=ccc*(16*p6p7*(2*fb3*hbcm2-fb4*fmb*hbcm-fb4*fmc*hbcm)+32*(
     . -fb3*p1p3*p6p7-2*fb3*p2p3*p6p7+2*fb3*p3p4*p6p7-fb3*p3p6*p3p7-2
     . *fb3*p3p7*p4p6))
      b(28)=ccc*(8*w2*(p6p7*p5p8)*(fmb*hbcm+fmc*hbcm)+8*w12*(p6p7*
     . p1p8)*(fmb*hbcm+fmc*hbcm)+32*(-fb3*p1p8*p6p7-2*fb3*p2p8*p6p7-
     . fb3*p3p6*p7p8-2*fb3*p4p6*p7p8+2*fb3*p4p8*p6p7))
      b(29)=16*ccc*(-fb3*p3p6-2*fb3*p4p6)
      b(30)=32*ccc*(-fb3*p3p7-fb3*p4p7-fb3*p5p7)
      b(31)=ccc*(w2*(8*(p6p7*p5p8)*(-ffmcfmb*hbcm3+fmc*hbcm2+hbcm3)+
     . 16*(p5p8*p3p7*p3p6)*(ffmcfmb*hbcm-hbcm)+16*(-hbcm*p2p3*p5p8*
     . p6p7+hbcm*p3p4*p5p8*p6p7-hbcm*p3p7*p4p6*p5p8))+w12*(8*(p6p7*
     . p1p8)*(-ffmcfmb*hbcm3+fmc*hbcm2+hbcm3)+16*(p3p7*p3p6*p1p8)*(
     . ffmcfmb*hbcm-hbcm)+16*(-hbcm*p1p8*p2p3*p6p7+hbcm*p1p8*p3p4*
     . p6p7-hbcm*p1p8*p3p7*p4p6))+(32*(p6p8*p3p7)*(fb3*fmb+fb4*
     . ffmcfmb*hbcm-fb4*hbcm)+8*(p7p8*p3p6)*(-4*fb3*fmb+hbcm)+32*(-
     . fb4*hbcm*p2p8*p6p7-fb4*hbcm*p4p6*p7p8+fb4*hbcm*p4p8*p6p7)))
      b(32)=4*ccc*p6p7*(8*fb3*fmb-4*fb3*fmc+4*fb4*ffmcfmb*hbcm-4*fb4*
     . hbcm-hbcm)
      do 200 n=1,32 
         c(n,1)=c(n,1)+0.09090909090909091d0*b(n)
         c(n,2)=c(n,2)-0.01680909685530997d0*b(n)
         c(n,3)=c(n,3)-0.02077659977796572d0*b(n)
200   continue
      return
	end 
 
      double precision function ams1_1p1(n) 
      implicit double precision(a-h,o-z) 
      implicit integer (i-n) 
#include "inclppp.h"
#include "inclcon.h"
#include "inclamp.h"
      b1=-p10p10
      b2=-p9p9
      aa1=c(10,n)**2+b1+b2
      b1=-p10p10
      b2=p9p9
      b3=-p12p12
      b4=p11p11
      b5=b3+b4
      b6=2*c(10,n)*p0p11
      b7=c(10,n)**2+b1+b2+b5+b6
      b8=-2*p0p10*p5p12
      b9=b8
      b10=2*p0p12*p5p10
      b11=b10
      b12=2*p10p12
      b13=-2*p9p11
      b14=b12+b13
      b15=2*epspn2
      b16=2*epspn4
      b17=-2*p0p11*p5p9
      b18=b17
      b19=2*p0p9*p5p11
      b20=b19
      b21=-2*c(10,n)*p5p9
      aa2=b11+b14*p0p5+b15+b16+b18+b20+b21+b7*fmb+b9
      b1=2*p4p10*p5p10
      aa3=b1
      aa4=2*epspn0
      b1=-2*p0p10*p4p12
      b2=b1
      b3=-2*p0p12*p4p10
      b4=b3
      b5=2*p10p12
      b6=2*p9p11
      b7=b5+b6
      b8=2*epspn1
      b9=-2*epspn3
      b10=-2*p0p11*p4p9
      b11=b10
      b12=-2*p0p9*p4p11
      b13=b12
      b14=-2*c(10,n)*p4p9
      aa5=b11+b13+b14+b2+b4+b7*p0p4+b8+b9
      b1=2*p4p9*p5p9
      aa6=b1
      b1=p12p12
      b2=p11p11
      b3=b1+b2
      b4=-2*p4p12*p5p12
      b5=b4
      b6=2*epspn5
      b7=-2*p4p11*p5p11
      b8=b7
      aa7=b3*p4p5+b5+b6+b8
      b1=4*p0p12*p4p12
      b2=b1
      b3=4*epspn6
      b4=4*p0p11*p4p11
      b5=b4
      b6=2*c(10,n)*p4p11
      aa8=b2+b3+b5+b6
      b1=-2*p12p12
      b2=-2*p11p11
      b3=b1+b2
      b4=-2*c(10,n)*p5p11
      aa9=b3*p0p5+b4
      aa10=2*c(10,n)*epspn7
      b1=2*c(10,n)*p0p11*p4p5
      aa11=b1
      ams1_1p1=aa1*p4p5+aa10+aa11+aa2*fmc+aa3+aa4+aa5*fmb+aa6+aa7+aa8*
     . p0p5+aa9*p0p4
      return
	end
