cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...subroutines for gg ->bc+b+\bar{c}:  bc in p-wave states.         c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...       bc should be in color-singlet and color-octet states.     c
c copyright (c) z.x zhang, chafik driouich, paula eerola and x.g. wu c
c reference: phys.rev. d70,114019(2004);                             c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c**********************************************************
c******** to get square amplitude for 1p1   ***************
c**********************************************************
      double precision function amps20_1p1()
      implicit double precision (a-h,o-z) 
      implicit integer (i-n)
      common/cccc/c36,c35,c34,c33,c32,c31,c30,c29 
     . ,c28,c27,c26,c25,c24,c23,c22,c21,c20,c19,c18,c17 
     . ,c16,c15,c14,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3
     . ,c2,c1s2,c1s1
#include "inclamp.h"
c...color flow.
 	common/colflow/amp2cf(10),smatval
	dimension tempamp(3)

	do ii=1,10
	   amp2cf(ii)=0.0d0
	end do

      do 200 m=1,32 
        do 400 n=1,3 
          c(m,n)=0.0d0
400     continue 
200   continue

      call genppp()

      call amp1s1_1p1(c1s1) 
      call amp1s2_1p1(c1s2) 
      call amp2_1p1(c2)
      call amp3_1p1(c3)
      call amp4_1p1(c4) 
      call amp5_1p1(c5) 
      call amp6_1p1(c6) 
      call amp7_1p1(c7) 
      call amp8_1p1(c8) 
      call amp9_1p1(c9) 
      call amp10_1p1(c10) 
      call amp11_1p1(c11) 
      call amp12_1p1(c12) 
      call amp13_1p1(c13) 
      call amp14_1p1(c14) 
      call amp15_1p1(c15) 
      call amp16_1p1(c16) 
      call amp17_1p1(c17) 
      call amp18_1p1(c18) 
      call amp19_1p1(c19) 
      call amp20_1p1(c20) 
      call amp21_1p1(c21) 
      call amp22_1p1(c22) 
      call amp23_1p1(c23) 
      call amp24_1p1(c24) 
      call amp25_1p1(c25) 
      call amp26_1p1(c26) 
      call amp27_1p1(c27) 
      call amp28_1p1(c28) 
      call amp29_1p1(c29) 
      call amp30_1p1(c30) 
      call amp31_1p1(c31) 
      call amp32_1p1(c32) 
      call amp33_1p1(c33) 
      call amp34_1p1(c34) 
      call amp35_1p1(c35) 
      call amp36_1p1(c36) 
      
	amps20_1p1=0.0d0

      do 600 n=1,3
         call genpppn(n)
         tempamp(n)=ams1_1p1(n)
         amps20_1p1=amps20_1p1+tempamp(n)
600   continue

	nreduce=4
      amps20_1p1=nreduce*amps20_1p1

c... color-flow.
c... note here: since the cross-terms in the color-flow are neglected,
c... the total square of amplitude does not equal to the sum of the
c... square of the partial amplitude (pertain to a certain color-flow).

	 amp2cf(1)=6.0d0/65.0d0*tempamp(1)+11.0d0/78.0d0*tempamp(2)
       amp2cf(2)=6.0d0/65.0d0*tempamp(1)+2.0d0/429.0d0*tempamp(2)
     &           +3.0d0/22.0d0*tempamp(3)
	 amp2cf(3)=27.0d0/130.0d0*tempamp(1)+3.0d0/286.0d0*tempamp(2)
     &           +1.0d0/66.0d0*tempamp(3)
c...smatval need not to be normalized to the whole amplitude, 
c...since we only need the relative probability for each color-flow.
	 smatval=amp2cf(1)+amp2cf(2)+amp2cf(3)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases.<--------> double precision
	if(amps20_1p1.lt.1.0d-16) amps20_1p1=1.0d-16
	do ii=1,3
	   if(amp2cf(ii).lt.1.0d-16) amp2cf(ii)=1.0d-16
	end do
	if(smatval.lt.1.0d-16) smatval=1.0d-16
      
	return
      end

c**********************************************************************
      double precision function amps2_1p1() 
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/ppp/pp(4,40),guv(4)
#include "inclcon.h"
      dimension ep1(4,2:3), ep2(4,2:3), ep3(4,2:4)

      amps2_1p1=0.d0

      call genpolar(pp(1,1),ep1)
      call genpolar(pp(1,2),ep2)
      call genpolar3(pp(1,3),ep3,hbcm)

c...the same for subprocess and tevatron and lhc
      do 400 nb1=2,2 
        do 401 nb2=2,3 
         do 402 nc3=2,4
          do i=1,4
            pp(i,6)=ep1(i,nb1)
            pp(i,7)=ep2(i,nb2)
            pp(i,8)=ep3(i,nc3)
          enddo
          amps2_1p1=amps2_1p1+2.0d0*amps20_1p1()
402      continue
401     continue
400   continue

      return
      end

c**********************************************************
c******** to get square amplitude for 3p0   ***************
c**********************************************************
      double precision function amps20_3p0()
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/cccc/c36,c35,c34,c33,c32,c31,c30,c29 
     . ,c28,c27,c26,c25,c24,c23,c22,c21,c20,c19,c18,c17 
     . ,c16,c15,c14,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3 
     . ,c2,c1s2,c1s1 
#include "inclamp.h"
c...color flow.
 	common/colflow/amp2cf(10),smatval
	dimension tempamp(3)

	do ii=1,10
	   amp2cf(ii)=0.0d0
	end do

       do 200 m=1,16 
        do 400 n=1,3 
           c(m,n)=0.d0 
400     continue 
200    continue
      
	 call genppp()

	 call amp1s1_3p0(c1s1) 
       call amp1s2_3p0(c1s2)
       call amp2_3p0(c2)
       call amp3_3p0(c3) 
       call amp4_3p0(c4) 
       call amp5_3p0(c5) 
       call amp6_3p0(c6) 
       call amp7_3p0(c7) 
       call amp8_3p0(c8) 
       call amp9_3p0(c9) 
       call amp10_3p0(c10) 
       call amp11_3p0(c11) 
       call amp12_3p0(c12) 
       call amp13_3p0(c13) 
       call amp14_3p0(c14) 
       call amp15_3p0(c15) 
       call amp16_3p0(c16) 
       call amp17_3p0(c17) 
       call amp18_3p0(c18) 
       call amp19_3p0(c19) 
       call amp20_3p0(c20) 
       call amp21_3p0(c21) 
       call amp22_3p0(c22) 
       call amp23_3p0(c23) 
       call amp24_3p0(c24) 
       call amp25_3p0(c25) 
       call amp26_3p0(c26) 
       call amp27_3p0(c27) 
       call amp28_3p0(c28) 
       call amp29_3p0(c29) 
       call amp30_3p0(c30) 
       call amp31_3p0(c31) 
       call amp32_3p0(c32) 
       call amp33_3p0(c33) 
       call amp34_3p0(c34) 
       call amp35_3p0(c35) 
       call amp36_3p0(c36)
      
	 amps20_3p0=0.0d0

       do n=1,3
         tempamp(n)=ams1_3p0(n)
	   amps20_3p0=amps20_3p0+tempamp(n)
       enddo

       nreduce=4
       amps20_3p0=nreduce*amps20_3p0

c... color-flow.
c... note here: since the cross-terms in the color-flow are neglected,
c... the total square of amplitude does not equal to the sum of the
c... square of the partial amplitude (pertain to a certain color-flow).

	 amp2cf(1)=6.0d0/65.0d0*tempamp(1)+11.0d0/78.0d0*tempamp(2)
       amp2cf(2)=6.0d0/65.0d0*tempamp(1)+2.0d0/429.0d0*tempamp(2)
     &           +3.0d0/22.0d0*tempamp(3)
	 amp2cf(3)=27.0d0/130.0d0*tempamp(1)+3.0d0/286.0d0*tempamp(2)
     &           +1.0d0/66.0d0*tempamp(3)
c...smatval need not to be normalized to the whole amplitude, 
c...since we only need the relative probability for each color-flow.
	 smatval=amp2cf(1)+amp2cf(2)+amp2cf(3)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases. <------> double precision
	if(amps20_3p0.lt.1.0d-16) amps20_3p0=1.0d-16
	do ii=1,3
	   if(amp2cf(ii).lt.1.0d-16) amp2cf(ii)=1.0d-16
	end do
	if(smatval.lt.1.0d-16) smatval=1.0d-16
      
	return
      end

c********************************** 
      double precision function amps2_3p0()
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/ppp/pp(4,40),guv(4)
#include "inclcon.h"
      dimension ep1(4,2:3), ep2(4,2:3)
      
	amps2_3p0=0.d0

      call genpolar(pp(1,1),ep1)
      call genpolar(pp(1,2),ep2)

c...the same for subprocess and tevatron and lhc
      do 400 nb1=2,2
        do 401 nb2=2,3
         do i=1,4
           pp(i,6)=ep1(i,nb1)
           pp(i,7)=ep2(i,nb2)
         enddo
         amps2_3p0=amps2_3p0+2.0d0*amps20_3p0()
401     continue
400   continue

      return
      end
c**********************************************************
c******** to get square amplitude for 3p1   ***************
c**********************************************************
      double precision function amps20_3p1()
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/cccc/c36,c35,c34,c33,c32,c31,c30,c29 
     . ,c28,c27,c26,c25,c24,c23,c22,c21,c20,c19,c18,c17 
     . ,c16,c15,c14,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3 
     . ,c2,c1s2,c1s1 
#include "inclamp.h"
c...color flow.
 	common/colflow/amp2cf(10),smatval
	dimension tempamp(3)

	do ii=1,10
	   amp2cf(ii)=0.0d0
	end do

      do 200 m=1,48 
       do 400 n=1,3 
         c(m,n)=0.0d0 
400    continue 
200   continue 
      
	call genppp()

      call amp1s1_3p1(c1s1) 
      call amp1s2_3p1(c1s2) 
      call amp2_3p1(c2) 
      call amp3_3p1(c3) 
      call amp4_3p1(c4) 
      call amp5_3p1(c5) 
      call amp6_3p1(c6) 
      call amp7_3p1(c7) 
      call amp8_3p1(c8) 
      call amp9_3p1(c9) 
      call amp10_3p1(c10) 
      call amp11_3p1(c11) 
      call amp12_3p1(c12) 
      call amp13_3p1(c13) 
      call amp14_3p1(c14) 
      call amp15_3p1(c15) 
      call amp16_3p1(c16) 
      call amp17_3p1(c17) 
      call amp18_3p1(c18) 
      call amp19_3p1(c19) 
      call amp20_3p1(c20) 
      call amp21_3p1(c21) 
      call amp22_3p1(c22) 
      call amp23_3p1(c23) 
      call amp24_3p1(c24) 
      call amp25_3p1(c25) 
      call amp26_3p1(c26) 
      call amp27_3p1(c27) 
      call amp28_3p1(c28) 
      call amp29_3p1(c29) 
      call amp30_3p1(c30) 
      call amp31_3p1(c31) 
      call amp32_3p1(c32) 
      call amp33_3p1(c33) 
      call amp34_3p1(c34) 
      call amp35_3p1(c35) 
      call amp36_3p1(c36) 
      
	amps20_3p1=0.0d0

      do 600 n=1,3
         call genpppn(n)
         tempamp(n)=ams1_3p1(n)
	   amps20_3p1=amps20_3p1+tempamp(n)
  600 continue

      nreduce=4
      amps20_3p1=nreduce*amps20_3p1

c... color-flow.
c... note here: since the cross-terms in the color-flow are neglected,
c... the total square of amplitude does not equal to the sum of the
c... square of the partial amplitude (pertain to a certain color-flow).

	 amp2cf(1)=6.0d0/65.0d0*tempamp(1)+11.0d0/78.0d0*tempamp(2)
       amp2cf(2)=6.0d0/65.0d0*tempamp(1)+2.0d0/429.0d0*tempamp(2)
     &           +3.0d0/22.0d0*tempamp(3)
	 amp2cf(3)=27.0d0/130.0d0*tempamp(1)+3.0d0/286.0d0*tempamp(2)
     &           +1.0d0/66.0d0*tempamp(3)
c...smatval need not to be normalized to the whole amplitude, 
c...since we only need the relative probability for each color-flow.
	 smatval=amp2cf(1)+amp2cf(2)+amp2cf(3)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases. <----->double precision
	if(amps20_3p1.lt.1.0d-16) amps20_3p1=1.0d-16
	do ii=1,3
	   if(amp2cf(ii).lt.1.0d-16) amp2cf(ii)=1.0d-16
	end do
	if(smatval.lt.1.0d-16) smatval=1.0d-16

      return
      end

c******************************** 
      double precision function amps2_3p1() 
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/ppp/pp(4,40),guv(4)
#include "inclcon.h"
      dimension ep1(4,2:3), ep2(4,2:3), ep3(4,2:4) 

      amps2_3p1=0.0d0

      call genpolar(pp(1,1),ep1)
      call genpolar(pp(1,2),ep2)
      call genpolar3(pp(1,3),ep3,hbcm)

c...the same for subprocess and tevatron and lhc
	do 400 nb1=2,2 
        do 401 nb2=2,3 
         do 402 nc3=2,4 
           do i=1,4
             pp(i,6)=ep1(i,nb1)
             pp(i,7)=ep2(i,nb2)
             pp(i,8)=ep3(i,nc3)
           enddo
           amps2_3p1=amps2_3p1+2.0d0*amps20_3p1()
402      continue
401     continue
400   continue

      return
      end

c**********************************************************
c******** to get square amplitude for 3p2   ***************
c**********************************************************
      double precision function amps20_3p2() 
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/cccc/c36,c35,c34,c33,c32,c31,c30,c29 
     . ,c28,c27,c26,c25,c24,c23,c22,c21,c20,c19,c18,c17 
     . ,c16,c15,c14,c13,c12,c11,c10,c9,c8,c7,c6,c5,c4,c3 
     . ,c2,c1s2,c1s1 
#include "inclamp.h"
c...color flow.
 	common/colflow/amp2cf(10),smatval
	dimension tempamp(3)

	do ii=1,10
	   amp2cf(ii)=0.0d0
	end do

      do 200 m=1,94
        do 400 n=1,3
          c(m,n)=0.0d0
400     continue 
200   continue 

      call genppp() 
      call amp1s1_3p2(c1s1) 
      call amp1s2_3p2(c1s2) 
      call amp2_3p2(c2) 
      call amp3_3p2(c3) 
      call amp4_3p2(c4) 
      call amp5_3p2(c5) 
      call amp6_3p2(c6) 
      call amp7_3p2(c7) 
      call amp8_3p2(c8) 
      call amp9_3p2(c9) 
      call amp10_3p2(c10) 
      call amp11_3p2(c11) 
      call amp12_3p2(c12) 
      call amp13_3p2(c13) 
      call amp14_3p2(c14) 
      call amp15_3p2(c15) 
      call amp16_3p2(c16) 
      call amp17_3p2(c17) 
      call amp18_3p2(c18) 
      call amp19_3p2(c19) 
      call amp20_3p2(c20) 
      call amp21_3p2(c21) 
      call amp22_3p2(c22) 
      call amp23_3p2(c23) 
      call amp24_3p2(c24) 
      call amp25_3p2(c25) 
      call amp26_3p2(c26) 
      call amp27_3p2(c27) 
      call amp28_3p2(c28) 
      call amp29_3p2(c29) 
      call amp30_3p2(c30) 
      call amp31_3p2(c31) 
      call amp32_3p2(c32) 
      call amp33_3p2(c33) 
      call amp34_3p2(c34) 
      call amp35_3p2(c35) 
      call amp36_3p2(c36)

	amps20_3p2=0.0d0

      do n=1,3
         call genpppn(n)
	   tempamp(n)=ams1_3p2(n)
	   amps20_3p2=amps20_3p2+tempamp(n)
      enddo

	nreduce=4
      amps20_3p2=nreduce*amps20_3p2

c... color-flow.
c... note here: since the cross-terms in the color-flow are neglected,
c... the total square of amplitude does not equal to the sum of the
c... square of the partial amplitude (pertain to a certain color-flow).

	 amp2cf(1)=6.0d0/65.0d0*tempamp(1)+11.0d0/78.0d0*tempamp(2)
       amp2cf(2)=6.0d0/65.0d0*tempamp(1)+2.0d0/429.0d0*tempamp(2)
     &           +3.0d0/22.0d0*tempamp(3)
	 amp2cf(3)=27.0d0/130.0d0*tempamp(1)+3.0d0/286.0d0*tempamp(2)
     &           +1.0d0/66.0d0*tempamp(3)
c...smatval need not to be normalized to the whole amplitude, 
c...since we only need the relative probability for each color-flow.
	 smatval=amp2cf(1)+amp2cf(2)+amp2cf(3)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases. <------>double precision
	if(amps20_3p2.lt.1.0d-16) amps20_3p2=1.0d-16
	do ii=1,3
	   if(amp2cf(ii).lt.1.0d-16) amp2cf(ii)=1.0d-16
	end do
	if(smatval.lt.1.0d-16) smatval=1.0d-16

      return
      end
c******************************************* 
      double precision function amps2_3p2() 
      implicit double precision (a-h,o-z) 
      implicit integer (i-n) 
      common/ppp/pp(4,40),guv(4)
#include "inclcon.h"
      dimension ep1(4,2:3), ep2(4,2:3), ept3(4,4,2:6) 
      
	amps2_3p2=0.0d0

      call genpolar(pp(1,1),ep1)
      call genpolar(pp(1,2),ep2)
      call gentensor(pp(1,3),ept3,hbcm)

c...the same for subprocess and tevatron and lhc
	do 400 nb1=2,2
        do 401 nb2=2,3 
         do 402 nt3=2,6 
          do i=1,4
           pp(i,6)=ep1(i,nb1)
           pp(i,7)=ep2(i,nb2)
          enddo
          call def_p_eq_p_dot_tp(8,4,ept3,nt3)
          call def_p_eq_p_dot_tp(9,5,ept3,nt3)
          call def_p_eq_p_dot_tp(10,1,ept3,nt3)
          call def_p_eq_p_dot_tp(11,2,ept3,nt3)
          call def_p_eq_p_dot_tp(12,7,ept3,nt3)
          call def_p_eq_p_dot_tp(13,6,ept3,nt3)
          amps2_3p2=amps2_3p2+2.0d0*amps20_3p2()
402      continue
401     continue
400   continue

      return
	end
	
