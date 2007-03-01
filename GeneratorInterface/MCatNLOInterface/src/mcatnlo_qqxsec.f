c This file contains all the relevant cross section formulae for heavy
c quark production, to be used in conjunction with the heavy quark
c package.
c Common blocks :
c    /process/prc           (character * 2)
c    /nl/nl
c    /scheme/schhad1,schhad2  (character * 2)
c    /betfac/betfac,delta
c Functions called:
c    ddilog  (cernlib)
c
      function ppsv(s,t,xm2,xmur2,xmuf1h1,xmuf2h2)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      parameter (vca=3.d0)
      parameter (vtf=0.5d0)
      parameter (vcf=4/3.d0)
      character * 2 prc
      common/process/prc
      common/nl/nl
      common/betfac/betfac,delta
      b = sqrt(1-4*xm2/s)
      xlb = log(b*betfac)
      b0 = (11*vca-4*vtf*nl)/(12*pi)
      csih1h2  = log(xmuf1h1/xmuf2h2)
      csirh = log(xmur2/xmuf2h2)
      if(prc.eq.'gg')then
        bbb = ggqq2(s,t,xm2,xmuf2h2,nl)+(8*pi*b0*csirh
     #       -(4*pi*b0+8*vca*xlb)*csih1h2)*ggborn(s,t,xm2)
      elseif(prc.eq.'qq')then
        bbb = qqqq2(s,t,xm2,xmuf2h2,nl)+(8*pi*b0*csirh
     #       -vcf*(3.d0+8*xlb)*csih1h2)*qqborn(s,t,xm2)
      elseif(prc.eq.'qg') then
        bbb = qgqq2(s,t,xm2,nl)
      else
        write(*,*)'PPSV: non existent process ',prc
        stop
      endif
      ppsv = bbb
      end

      function ppcolp(y,s,q1q,x,xm2,xlmude)
      implicit real * 8 (a-h,o-z)
      character * 2 prc,schhad1,schhad2,schtmp
      common/nl/nl
      common/process/prc
      common/scheme/schhad1,schhad2
      common/schtmp/schtmp
      if(prc.eq.'gg') then
          if( y .eq. 1) then
             schtmp=schhad1
          elseif( y .eq. -1 ) then
             schtmp=schhad2
          else
             write(*,*) 'error in ppcolp: y=',y
             stop
          endif
          ppcolp = ggcolp(s,q1q,x,xm2,xlmude,nl)
      elseif(prc.eq.'qq') then
          if( y .eq. 1) then
             schtmp=schhad1
          elseif( y .eq. -1 ) then
             schtmp=schhad2
          else
             write(*,*) 'error in ppcolp: y=',y
             stop
          endif
          ppcolp = qqcolp(s,q1q,x,xm2,xlmude,nl)
      elseif(prc.eq.'qg') then
          if( y .eq. 1 ) then
             ppcolp = qgcolp1(s,q1q,x,xm2,xlmude,nl)
          elseif( y .eq. -1 ) then
             ppcolp = qgcolp2(s,q1q,x,xm2,xlmude,nl)
          else
             write(*,*) 'error in ppcolp: y=',y
             stop
          endif
      else
          write(*,*) 'error in ppcolp: prc=',prc
          stop
      endif
      end

      function ppcoll(y,s,q1q,x,xm2)
      implicit real * 8 (a-h,o-z)
      character * 2 prc,schhad1,schhad2,schtmp
      common/nl/nl
      common/process/prc
      common/scheme/schhad1,schhad2
      common/schtmp/schtmp
      if(prc.eq.'gg') then
          if( y .eq. 1) then
             schtmp=schhad1
          elseif( y .eq. -1 ) then
             schtmp=schhad2
          else
             write(*,*) 'error in ppcoll: y=',y
             stop
          endif
          ppcoll = ggcoll(s,q1q,x,xm2,nl)
      elseif(prc.eq.'qq') then
          if( y .eq. 1) then
             schtmp=schhad1
          elseif( y .eq. -1 ) then
             schtmp=schhad2
          else
             write(*,*) 'error in ppcoll: y=',y
             stop
          endif
          ppcoll = qqcoll(s,q1q,x,xm2,nl)
      elseif(prc.eq.'qg') then
          if( y .eq. 1 ) then
             ppcoll = qgcoll1(s,q1q,x,xm2,nl)
          elseif( y .eq. -1 ) then
             ppcoll = qgcoll2(s,q1q,x,xm2,nl)
          else
             write(*,*) 'error in ppcoll: y=',y
             stop
          endif
      else
          write(*,*) 'error in ppcoll: prc=',prc
          stop
      endif
      end

      function fpp(s0,x,y,xm20,q1q0,q2q0,w1h,w2h,cth2)
      implicit real * 8 (a-h,o-z)
      character * 2 prc
      common/process/prc
      s=1
      xm2=xm20/s0
      q1q=q1q0/s0
      q2q=q2q0/s0
      if(prc.eq.'gg') then
         fpp = fgg(s,x,y,xm2,q1q,q2q,w1h,w2h,cth2)
      elseif(prc.eq.'qg') then
         fpp = fqg(s,x,y,xm2,q1q,q2q,w1h,w2h,cth2)
      elseif(prc.eq.'qq') then
         fpp = fqq(s,x,y,xm2,q1q,q2q,w1h,w2h,cth2)
      else
         write(*,*) 'FPP: non existent subprocess',prc
         stop
      endif
      end

C Correction terms for change of subtraction scheme.
C Assume that the change of structure functions is given by:
C xkdij(), xkpij(x), xklij(x).
C
      function cthdgg1(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd = xkdgg(nl)
      xkp = xkpgg(one,nl)
      xkl = xklgg(one,nl)
      t2   = 1 - t1
      xlg2 = log(t2)
      cthdgg1 = -( xkd-xlg2*xkp+xlg2**2*xkl/2 )*hqh0gg(t1,ro)
      return
      end

      function cthdgg2(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd = xkdgg(nl)
      xkp = xkpgg(one,nl)
      xkl = xklgg(one,nl)
      xlg1 = log(t1)
      cthdgg2 = -( xkd-xlg1*xkp+xlg1**2*xkl/2 )*hqh0gg(t1,ro)
      return
      end

      function cthdqa1(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd = xkdqq(nl)
      xkp = xkpqq(one,nl)
      xkl = xklqq(one,nl)
      t2   = 1 - t1
      xlg2 = log(t2)
      cthdqa1 = -( xkd-xlg2*xkp+xlg2**2*xkl/2 )*hqh0qa(t1,ro)
      return
      end

      function cthdqa2(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd = xkdqq(nl)
      xkp = xkpqq(one,nl)
      xkl = xklqq(one,nl)
      xlg1 = log(t1)
      cthdqa2 = -( xkd-xlg1*xkp+xlg1**2*xkl/2 )*hqh0qa(t1,ro)
      return
      end

      function cthdqg1(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd2 = xkdgq(nl)
      xkp2 = xkpgq(one,nl)
      xkl2 = xklgq(one,nl)
      t2   = 1 - t1
      xlg2 = log(t2)
      cthdqg1 = -( xkd2-xlg2*xkp2+xlg2**2*xkl2/2 )*hqh0gg(t1,ro)
      return
      end

      function cthdqg2(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      data one/1.d0/
      xkd1 = xkdqg(nl)
      xkp1 = xkpqg(one,nl)
      xkl1 = xklqg(one,nl)
      xlg1 = log(t1)
      cthdqg2 = -( xkd1-xlg1*xkp1+xlg1**2*xkl1/2 )*hqh0qa(t1,ro)
      return
      end

      function cthpgg1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkp2 = xkpgg(r2,nl)
      xkl2 = xklgg(r2,nl)
      cthpgg1 = -(xkp2-log(1-t1)*xkl2)*hqh0gg(t1,ro/r2)/r2
      return
      end

      function cthpgg2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkp1 = xkpgg(r1,nl)
      xkl1 = xklgg(r1,nl)
      cthpgg2 = -( xkp1-log(1-t2)*xkl1 )*hqh0gg(t2,ro/r1)/r1
      return
      end

      function cthpqa1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkp2 = xkpqq(r2,nl)
      xkl2 = xklqq(r2,nl)
      cthpqa1 = -( xkp2-log(1-t1)*xkl2 )*hqh0qa(t1,ro/r2)/r2
      return
      end

      function cthpqa2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkp1 = xkpqq(r1,nl)
      xkl1 = xklqq(r1,nl)
      cthpqa2 = -( xkp1-log(1-t2)*xkl1 )*hqh0qa(t2,ro/r1)/r1
      return
      end

      function cthpqg1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkp2 = xkpgq(r2,nl)
      xkl2 = xklgq(r2,nl)
      cthpqg1 = -( xkp2-log(1-t1)*xkl2 )*hqh0gg(t1,ro/r2)/r2
      return
      end

      function cthpqg2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkp1 = xkpqg(r1,nl)
      xkl1 = xklqg(r1,nl)
      cthpqg2 = -(xkp1-log(1-t2)*xkl1)*hqh0qa(t2,ro/r1)/r1
      return
      end

      function cthlgg1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkl2 = xklgg(r2,nl)
      cthlgg1 = -xkl2*hqh0gg(t1,ro/r2)/r2
      return
      end

      function cthlgg2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkl1 = xklgg(r1,nl)
      cthlgg2 = -xkl1*hqh0gg(t2,ro/r1)/r1
      return
      end

      function cthlqa1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkl2 = xklqq(r2,nl)
      cthlqa1 = -xkl2*hqh0qa(t1,ro/r2)/r2
      return
      end

      function cthlqa2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkl1 = xklqq(r1,nl)
      cthlqa2 = -xkl1*hqh0qa(t2,ro/r1)/r1
      return
      end

      function cthlqg1(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r2 = t2/(1-t1)
      xkl2 = xklgq(r2,nl)
      cthlqg1 = -xkl2*hqh0gg(t1,ro/r2)/r2
      return
      end

      function cthlqg2(tx,t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      t2 = 1 - t1 - tx
      r1 = t1/(1-t2)
      xkl1 = xklqg(r1,nl)
      cthlqg2 = -xkl1*hqh0qa(t2,ro/r1)/r1
      return
      end

      FUNCTION ASHDQA(T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      DATA PI/3.141 592 653 589 793/
      T2 = 1-T1
      B = DSQRT(1-RO)
      LP = (B+1)/ 2.D0
      LM = (1-B)/ 2.D0
      AT = T1
      AW = T2
      VLTM = DLOG(4*AT/RO)
      VLPM = DLOG(LP/LM)
      VLSM = DLOG(4/RO)
      VLWM = DLOG(4*AW/RO)
      VDW = DDILOG((AW-RO/ 4.D0)/AW)-VLWM**2/ 2.D0
      VDT = DDILOG((AT-RO/ 4.D0)/AT)-VLTM**2/ 2.D0
      VDMP = DDILOG(-LM/LP)
      AUINV = 1/(RO/ 4.D0-AW)
      ATINV = 1/(RO/ 4.D0-AT)
      SRLGPR = DLOG((1-B)*T2/((B+1)*T1))*DLOG((B+1)*T2/((1-B)*T1))
      SRL21P = DDILOG(1-(B+1)*T1/((1-B)*T2))
      SRL21M = DDILOG(1-(1-B)*T1/((B+1)*T2))
      SRL22P = DDILOG(1-(B+1)*T2/((1-B)*T1))
      SRL22M = DDILOG(1-(1-B)*T2/((B+1)*T1))
      SRLG1 = DLOG((B+1)/(1-B))/B
      SL3515 = SRLGPR+B**2*SRLG1**2+SRL21P+SRL21M
      SL3525 = SRLGPR+B**2*SRLG1**2+SRL22P+SRL22M
      DD = -5*(2*(4*T1**2-2*T1+1)+3*RO)*VLSM*VLWM/ 27.D0
      DD = DD-5*AUINV*(2*(T1-1)*T1+RO)*VLWM/ 27.D0
      DD = 5*(2*(4*T1**2-6*T1+3)+3*RO)*VLSM*VLTM/ 27.D0+DD
      DD = 5*ATINV*(2*(T1-1)*T1+RO)*VLTM/ 27.D0+DD
      DD = 5*(2*T1-1)*VLSM**2/ 27.D0+DD
      DD = 10*(2*T1-1)*VLSM/( 27.D0*B**2)+DD
      DD = DD-5*(RO*(2*T1-1)-2*(3*T1-2))*VLPM**2/( 54.D0*B)
      DD = DD-5*(2*(T1-1)+RO)*VLPM**2/( 54.D0*B**3)
      DD = DD-5*(2*(2*T1-1)+RO)*VDW/ 27.D0
      DD = 5*(RO-2*(2*T1-1))*VDT/ 27.D0+DD
      DD = DD-10*(RO*(2*T1-1)-2*(3*T1-2))*VDMP/( 27.D0*B)
      DD = DD-10*(2*(T1-1)+RO)*VDMP/( 27.D0*B**3)
      DD = DD-5*SL3525*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0
      DD = 5*SL3515*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0+DD
      DD = DD-5*PI**2*(RO*(2*T1-1)-2*(3*T1-2))/( 162.D0*B)
      DD = DD-5*PI**2*(2*(T1-1)+RO)/( 162.D0*B**3)
      DD = 5*PI**2*(2*T1-1)/ 81.D0+DD
      ASHDQA = DD
      RETURN
      END
      FUNCTION ASHPQA(TX,T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(TX.EQ.0)THEN
      T2 = 1-T1
      SRL12 = DLOG(T1/T2)
      PP = 20*SRL12*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0
      ASHPQA = PP
      RETURN
      ELSE
      T2 = -TX-T1+1
      T11 = 1/(1-T1)
      T22 = 1/(TX+T1)
      B = DSQRT((1-TX)**2-RO)
      DLAM2 = 1/(TX**2-2*TX-RO+1)
      RLGXRO = DLOG((4*TX+RO)/RO)/TX
      RLG12 = DLOG(T1/(1-T2))/TX
      RLG21 = DLOG(T2/(1-T1))/TX
      RLG11 = DLOG(T1/(1-T1))
      RLG22 = DLOG(T2/(1-T2))
      RL34 = DLOG((2*TX**2+2*B*TX-2*TX-RO)/(2*TX**2-2*B*TX-2*TX-RO))/(B*
     1   TX)
      RL35 = DLOG((2*TX+RO-2*B-2)/(2*TX+RO+2*B-2))/B
      PP = -5*RLGXRO*(T2-1)*(T2+T1-1)*(2*(T2-T1)+RO)*T22/ 54.D0
      PP = 5*T22*(4*(T2**2+(-T1-5)*T2-3*T1+4)*TX+RO*(T2**2+(T1-7)*T2-5*T
     1   1+6))/( 54.D0*(4*TX+RO))+PP
      PP = 5*RLGXRO*(T1-1)*T11*(T2+T1-1)*(RO-2*(T2-T1))/ 54.D0+PP
      PP = PP-5*T11*(RO*((T1-5)*T2+T1**2-7*T1+6)-4*((T1+3)*T2-T1**2+5*T1
     1   -4)*TX)/( 54.D0*(4*TX+RO))
      PP = PP-5*RLG22*(2*(3*T2**2-2*T2+T1**2+1)+RO*(T2+T1+1))/( 27.D0*(T
     1   2+T1))
      PP = 5*RLG21*(T2+T1-1)*(2*(2*T2**2+2*T1**2-2*T1+1)+RO)/( 27.D0*(T2
     1   +T1))+PP
      PP = PP-5*RLG12*(T2+T1-1)*(2*(2*T2**2-2*T2+2*T1**2+1)+RO)/( 27.D0*
     1   (T2+T1))
      PP = 5*RLG11*(2*(T2**2+3*T1**2-2*T1+1)+RO*(T2+T1+1))/( 27.D0*(T2+T
     1   1))+PP
      TMP0 = RO-2*T1*(T2+T1)
      TMP0 = -5*DLAM2*RL35*(T2+T1-1)**2*(T2+T1+RO)*TMP0/( 27.D0*(T2+T1))
      PP = TMP0+PP
      PP = PP-5*RL35*(T2+T1-1)*(RO*(2*T1-1)-2*(T2**2-T1*T2-2*T1**2+T1))/
     1   ( 27.D0*(T2+T1))
      PP = 5*DLAM2*RL34*(T2+T1-1)*(RO-2*T2*(T2+T1))*(-RO*TX+T2+T1)/( 27.
     1   D0*(T2+T1))+PP
      PP = 5*RL34*(T2+T1-1)*(2*(T2**2+T2-T1**2)+RO*(2*T1-1))/( 27.D0*(T2
     1   +T1))+PP
      TMP0 = RO-2*((T1+1)*T2+T1**2-T1)
      TMP0 = -20*DLAM2*(T2+T1-2)*(T2+T1-1)*TMP0/( 27.D0*(4*TX+RO))
      PP = TMP0+PP
      PP = PP-20*(T2+T1-2)*(T2+T1-1)/( 27.D0*(4*TX+RO))
      ASHPQA = PP
      RETURN
      ENDIF
      END
      FUNCTION ASHPQG(TX,T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(TX.EQ.0)THEN
      PP = 0
      ASHPQG = PP
      RETURN
      ELSE
      T2 = -TX-T1+1
      T11 = 1/(1-T1)
      T22 = 1/(TX+T1)
      B = DSQRT((1-TX)**2-RO)
      DLAM2 = 1/(TX**2-2*TX-RO+1)
      RLGXRO = DLOG((4*TX+RO)/RO)/TX
      RLG12 = DLOG(T1/(1-T2))/TX
      RLG21 = DLOG(T2/(1-T1))/TX
      RLGRO = DLOG(4*(1-T1)*(1-T2)/RO)
      RR = T2*(T2*TX**2+RO*(1-T2))
      RR = DSQRT(RR)
      RL3424 = 2*DLOG((T2*TX+RR)/(RR-T2*TX))/(RR*TX)
      RL34 = DLOG((2*TX**2+2*B*TX-2*TX-RO)/(2*TX**2-2*B*TX-2*TX-RO))/(B*
     1   TX)
      PP = RO*T2-2*(T2-1)*(T1*T2+T1-1)
      PP = 5*PP*RLGXRO*(T2+T1-1)**2*T22**2/( 72.D0*T2)
      TMP0 = RO-2*((T1+1)*T2+T1-1)
      TMP0 = -5*(T2+1)*(T2+T1-1)**2*T22**2*TMP0/( 18.D0*T2*(4*TX+RO))
      PP = TMP0+PP
      PP = 5*RLGXRO*(T2+T1-1)**2*(RO*(T2+1)-2*(T2**2+3*T1*T2+3*T1-2))*T2
     1   2/( 72.D0*T2)+PP
      PP = PP-5*(T2+T1-1)*(RO-2*(T2+2*T1-1))*T22/( 72.D0*T2)
      PP = PP-5*RLGXRO*(T1-1)*T11**2*(T2+T1-1)**3/( 72.D0*T2)
      TMP0 = 3*RO-4*((2*T1+1)*T2+T1-1)
      TMP0 = 5*(T1-1)*T11**2*(T2+T1-1)**2*TMP0/( 144.D0*T2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = RO*(T1-1)-2*((T1-2)*T2+2*T1**2-4*T1+2)
      TMP0 = -5*RLGXRO*T11*(T2+T1-1)**2*TMP0/( 144.D0*T2)
      PP = TMP0+PP
      PP = 5*T11*(T2+T1-1)*(-T1*T2-4*T1**2+6*T1+RO*(T1-1)-2)/( 144.D0*T2
     1   )+PP
      PP = PP-5*RLGRO*(T2+T1-1)*(2*(2*T2**2+(4*T1-4)*T2+4*T1**2-6*T1+3)+
     1   RO)/( 72.D0*(T2-1)*T2)
      TMP0 = 2*(T2**2+(2*T1-2)*T2+4*T1**2-4*T1+2)+RO*T2
      TMP0 = 5*RLG21*(T2+T1-1)**2*TMP0/( 72.D0*(T2-1)*T2)
      PP = TMP0+PP
      TMP0 = 2*T2*(3*T2**2+(6*T1-4)*T2+8*T1**2-6*T1+3)+RO*(T2-4)*(T2-1)
      TMP0 = 5*RLG12*(T2+T1-1)**2*TMP0/( 72.D0*(T2-1)*T2**2)
      PP = TMP0+PP
      TMP0 = 2*T2*(3*T2**2+(6*T1-4)*T2+4*T1**2-4*T1+2)*TX+RO*(T2**3+(T1+
     1   1)*T2**2+(4*T1-2)*T2-4*T1)
      TMP0 = -5*RL3424*(T2+T1-1)**2*TMP0/( 144.D0*(T2-1)*T2)
      PP = TMP0+PP
      PP = 5*DLAM2*RL34*(T2+T1-1)**2*(T2+T1)*(RO-2*T2*(T2+T1))/( 144.D0*
     1   T2)+PP
      TMP0 = RO*(T2+T1+2)-2*(T2**2+(3*T1-2)*T2+2*T1**2-T1)
      TMP0 = 5*RL34*(T2+T1-1)**2*TMP0/( 144.D0*T2)
      PP = TMP0+PP
      TMP0 = RO-2*((T1+1)*T2+T1**2-T1)
      TMP0 = 5*DLAM2*(T2+T1-1)**2*(T2+T1)*TMP0/( 36.D0*T2*(4*TX+RO))
      PP = TMP0+PP
      PP = 5*(T2+T1-1)*(-4*(3*T2**2+(2*T1-6)*T2-10*T1+3)*TX+RO*(-5*T2**2
     1   -(8*T1-24)*T2+24*T1-19)+RO**2*(T2-3))/( 144.D0*(T2-1)*T2*(4*TX+
     2   RO))+PP
      ASHPQG = PP
      RETURN
      ENDIF
      END
      function hqh0qa(t1,ro)
      implicit double precision (a-z)
      t2=1-t1
      hqh0qa=2*(2*t1**2+2*t2**2+ro)/9
      return
      end

      function hqbdqa(t1,ro,nl)
      implicit double precision (a-z)
      parameter(c1=2.d0/3,c2=8.d0/3)
      integer nl
      t2 = 1-t1
      hqbdqa = ( 7 - nl * c1 + c2*log(t1*t2) )*hqh0qa(t1,ro)
      return
      end

      function hqadqa(t1,ro,nl)
      implicit double precision (a-z)
      parameter (vcf=4/3.d0)
      integer nl
      t2 = 1-t1
      hqadqa = ( 2*vcf*log(t2)-3*vcf/2.d0 )*hqh0qa(t1,ro)
      return
      end

      function hqbpqa(tx,t1,ro)
      implicit double precision (a-z)
      parameter (cf=4.d0/3)
      pqq(x) = cf*(1+x**2)
      t2 = 1-tx-t1
      hqbpqa = -(1-t1)/t2*hqh0qa(t1  ,ro*(1-t1)/t2)*pqq(t2/(1-t1))
     #         -(1-t2)/t1*hqh0qa(1-t2,ro*(1-t2)/t1)*pqq(t1/(1-t2))
      return
      end

      function hqaaqa(tx,t1,ro)
      implicit double precision (a-z)
      parameter (cf=4.d0/3)
      pqq(x) = cf*(1+x**2)
      t2 = 1-tx-t1
      hqaaqa = -(1-t1)/t2*hqh0qa(t1  ,ro*(1-t1)/t2)*pqq(t2/(1-t1))
      return
      end

      function hqhlqa(tx,t1,ro)
      implicit double precision (a-z)
      hqhlqa = - 2 * hqbpqa(tx,t1,ro)
      return
      end

      function hqh0gg(t1,ro)
      implicit double precision (a-z)
      parameter (cf=4.d0/3)
      tt=cf
      t2 = 1-t1
      hqh0gg = (cf/t1/t2-3)*(t1**2+t2**2+ro*(1-ro/t1/t2/4))/8
      return
      end

      function hqbdgg(t1,ro,nl)
      implicit double precision (a-z)
      t2 = 1-t1
      hqbdgg = 6*log(t1*t2)*hqh0gg(t1,ro)
      return
      end

      function hqadgg(t1,ro,nl)
      implicit double precision (a-z)
      integer nl
      parameter (pi=3.14159265358979312D0)
      parameter (vca=3.d0)
      parameter (vtf=0.5d0)
      t2 = 1-t1
      b0 = (11*vca-4*vtf*nl)/(12*pi)
      hqadgg = ( 2*vca*log(t2)-2*pi*b0 )*hqh0gg(t1,ro)
      return
      end

      function hqbpgg(tx,t1,ro)
      implicit double precision (a-z)
      pgg(x) = 6*(x+(1-x)**2*(1/x+x))
      t2 = 1-t1-tx
      hqbpgg = -(1-t1)/t2*hqh0gg(t1  ,ro*(1-t1)/t2)*pgg(t2/(1-t1))
     #         -(1-t2)/t1*hqh0gg(1-t2,ro*(1-t2)/t1)*pgg(t1/(1-t2))
      return
      end

      function hqaagg(tx,t1,ro)
      implicit double precision (a-z)
      parameter (vca=3.d0)
      pgg(x) = 2*vca*(x+(1-x)**2*(1/x+x))
      t2 = 1-t1-tx
      hqaagg = -(1-t1)/t2*hqh0gg(t1  ,ro*(1-t1)/t2)*pgg(t2/(1-t1))
      return
      end

      function hqhlgg(tx,t1,ro)
      implicit double precision (a-z)
      hqhlgg = - 2 * hqbpgg(tx,t1,ro)
      return
      end

      function hqbpqg(tx,t1,ro)
      implicit double precision (a-z)
      parameter (cf=4.d0/3)
      t2 = 1-t1-tx
      x = t2/(1-t1)
      pgq = cf*(1+(1-x)**2)/x
      x = t1/(1-t2)
      pqg = (x**2+(1-x)**2)/2
      hqbpqg = -1/t2*hqh0gg(t1  ,ro*(1-t1)/t2)*tx*pgq
     #         -1/t1*hqh0qa(1-t2,ro*(1-t2)/t1)*tx*pqg
      return
      end

      function hqaaqg(tx,t1,ro)
      implicit double precision (a-z)
      parameter (cf=4.d0/3)
      t2 = 1-t1-tx
      x = t2/(1-t1)
      pgq = cf*(1+(1-x)**2)/x
      hqaaqg = -1/t2*hqh0gg(t1  ,ro*(1-t1)/t2)*tx*pgq
      return
      end

      function hqhlqg(tx,t1,ro)
      implicit double precision (a-z)
      hqhlqg = - 2 * hqbpqg(tx,t1,ro)
      return
      end
      FUNCTION HQHDGG(T1,RO,NL)
      IMPLICIT DOUBLE PRECISION (A-Z)
      INTEGER NL
      DATA PI/3.141 592 653 589 793/
      NLF = NL
      T2 = 1-T1
      B = DSQRT(1-RO)
      LP = (B+1)/ 2.D0
      LM = (1-B)/ 2.D0
      AT = T1
      AW = T2
      VLTM = DLOG(4*AT/RO)
      VLPM = DLOG(LP/LM)
      VLSM = DLOG(4/RO)
      VLWM = DLOG(4*AW/RO)
      VLBL = DLOG(B/LM)
      VDW = DDILOG((AW-RO/ 4.D0)/AW)-VLWM**2/ 2.D0
      VDT = DDILOG((AT-RO/ 4.D0)/AT)-VLTM**2/ 2.D0
      VDMP = DDILOG(-LM/LP)
      VDMB = VLBL**2/ 2.D0+DDILOG(-LM/B)
      AUINV = 1/(RO/ 4.D0-AW)
      ATINV = 1/(RO/ 4.D0-AT)
      SRLGPR = DLOG((1-B)*T2/((B+1)*T1))*DLOG((B+1)*T2/((1-B)*T1))
      SRL21P = DDILOG(1-(B+1)*T1/((1-B)*T2))
      SRL21M = DDILOG(1-(1-B)*T1/((B+1)*T2))
      SRL22P = DDILOG(1-(B+1)*T2/((1-B)*T1))
      SRL22M = DDILOG(1-(1-B)*T2/((B+1)*T1))
      SRLG1 = DLOG((B+1)/(1-B))/B
      SRL2L = (DDILOG(-4*B/(1-B)**2)-DDILOG(4*B/(B+1)**2))/B
      SRL212 = DLOG(4*T1*T2/RO)**2/ 2.D0+DDILOG(1-RO/( 4.D0*T1*T2))
      SL3515 = SRLGPR+B**2*SRLG1**2+SRL21P+SRL21M
      SL3525 = SRLGPR+B**2*SRLG1**2+SRL22P+SRL22M
      DD = -(RO**2*(36*T1**3-112*T1**2+107*T1-39)-8*(T1-1)**2*(9*T1**3-1
     1   0*T1**2+20*T1-11)+4*RO*(T1-1)**2*(18*T1**2-29*T1+19))*VLWM**2/(
     2    1152.D0*(T1-1)**3*T1)
      DD = DD-(2*(T1-1)*T1*(2*T1**2-2*T1+1)+RO**2*(T1**2-T1+1)+2*RO*(T1-
     1   1)*T1)*VLTM*VLWM/( 16.D0*(T1-1)**2*T1**2)
      DD = 9*(2*(T1-1)*T1*(12*T1**4-32*T1**3+35*T1**2-18*T1+4)+2*RO*(T1-
     1   1)*T1*(5*T1**2-8*T1+4)+RO**2*(3*T1**2-4*T1+2))*VLSM*VLWM/( 32.D
     2   0*(T1-1)**2*T1**2)+DD
      DD = (9*T1-10)*(-4*(T1-1)*(T1**2-2*T1+2)-2*RO**2*(T1**2-3*T1+3)-2*
     1   RO*(T1-1)*T1+RO**3)*VLPM*VLWM/( 576.D0*B*(T1-1)**2*T1)+DD
      TMP0 = -32*(T1-1)**3*(3*T1**2-23*T1+8)+8*RO*(T1-1)**2*(T1**2+17*T1
     1   +12)+12*RO**2*(T1-1)*(T1+3)+3*RO**3
      TMP0 = -AUINV**2*(9*T1-1)*TMP0*VLWM/( 4608.D0*(T1-1)**2*T1)
      DD = TMP0+DD
      DD = DD-AUINV*(9*T1-1)*(RO**2*(8*T1**2+8*T1-17)+4*(T1-1)**3*(25*T1
     1   -8)+2*RO*(T1-1)**2*(16*T1+27)+4*RO**3)*VLWM/( 576.D0*(T1-1)**2*
     2   T1)
      DD = (RO**2*(36*T1**3+4*T1**2-9*T1+8)-8*T1**2*(9*T1**3-17*T1**2+27
     1   *T1-8)-4*RO*T1**2*(18*T1**2-7*T1+8))*VLTM**2/( 1152.D0*(T1-1)*T
     2   1**3)+DD
      DD = 9*(2*(T1-1)*T1*(12*T1**4-16*T1**3+11*T1**2-4*T1+1)+2*RO*(T1-1
     1   )*T1*(5*T1**2-2*T1+1)+RO**2*(3*T1**2-2*T1+1))*VLSM*VLTM/( 32.D0
     2   *(T1-1)**2*T1**2)+DD
      DD = (9*T1+1)*(-2*RO**2*(T1**2+T1+1)+4*T1*(T1**2+1)-2*RO*(T1-1)*T1
     1   +RO**3)*VLPM*VLTM/( 576.D0*B*(T1-1)*T1**2)+DD
      DD = DD-ATINV**2*(9*T1-8)*(32*T1**3*(3*T1**2+17*T1-12)+8*RO*T1**2*
     1   (T1**2-19*T1+30)+12*RO**2*(T1-4)*T1+3*RO**3)*VLTM/( 4608.D0*(T1
     2   -1)*T1**2)
      DD = DD-ATINV*(9*T1-8)*(RO**2*(8*T1**2-24*T1-1)+4*T1**3*(25*T1-17)
     1   -2*RO*T1**2*(16*T1-43)+4*RO**3)*VLTM/( 576.D0*(T1-1)*T1**2)
      DD = DD-((T1-1)*T1*(2*T1**2-2*T1+1)*(432*T1**2-432*T1+191)+RO*(T1-
     1   1)*T1*(414*T1**2-414*T1+191)+2*RO**2*(54*T1**2-54*T1+25))*VLSM*
     2   *2/( 64.D0*(T1-1)**2*T1**2)
      TMP0 = 2*RO*(T1-1)*T1*(8*T1**2-8*T1-5)*(18*T1**2-18*T1-1)
      TMP0 = TMP0+2*RO**2*(54*T1**4-108*T1**3-25*T1**2+79*T1+3)+3*RO**3*
     1   (18*T1**2-18*T1-1)-4*(T1-1)*T1*(4*T1-5)*(4*T1+1)*(9*T1**2-9*T1+
     2   5)
      TMP0 = -TMP0*VLPM*VLSM/( 1152.D0*B*(T1-1)**2*T1**2)
      DD = TMP0+DD
      DD = DD-9*(2*T1-1)**2*(4*(3*T1**2-3*T1+1)-RO*(2*T1-1)**2+RO**2)*VL
     1   SM/( 64.D0*B**2*(T1-1)*T1)
      DD = DD-27*(2*T1-1)**2*(RO-2*(2*T1**2-2*T1+1))*VLSM/( 128.D0*B**4*
     1   (T1-1)*T1)
      DD = ((T1-1)*T1*(2304*T1**4-4608*T1**3+5128*T1**2-2824*T1+593)+14*
     1   RO*(T1-1)*T1*(36*T1**2-36*T1+25)+32*RO**2*(9*T1**2-9*T1+4))*VLS
     2   M/( 576.D0*(T1-1)**2*T1**2)+DD
      DD = DD-(2*(2448*T1**4-4896*T1**3+4513*T1**2-2065*T1+286)+RO*(144*
     1   T1**4-288*T1**3+181*T1**2-37*T1+2)+2*RO**2*(54*T1**2-54*T1+19))
     2   *VLPM**2/( 2304.D0*B*(T1-1)*T1)
      DD = DD-9*(-2*(24*T1**4-48*T1**3+49*T1**2-25*T1+4)+RO*(16*T1**4-32
     1   *T1**3+49*T1**2-33*T1+6)+RO**2*(-7*T1**2+7*T1-3)+RO**3)*VLPM**2
     2   /( 256.D0*B**3*(T1-1)*T1)
      DD = DD-9*(2*T1-1)**2*(8*(T1**2-T1+1)+4*RO*(T1-2)*(T1+1)+3*RO**2)*
     1   VLPM**2/( 512.D0*B**5*(T1-1)*T1)
      DD = DD-(2*(29*T1**2-29*T1+21)+2*RO**2*(27*T1**2-27*T1+4)+9*RO*(2*
     1   T1-1)**2)*VLPM**2/( 1152.D0*(T1-1)*T1)
      DD = B*(27*RO**2*(2*T1-1)**2+18*RO*(2*T1-1)**2+72*T1**2-72*T1+29)*
     1   VLPM/( 576.D0*(T1-1)*T1)+DD
      DD = (-2*(T1-1)*T1*(936*T1**4-1872*T1**3+2086*T1**2-1150*T1+285)+R
     1   O*(T1-1)*T1*(864*T1**4-1728*T1**3+836*T1**2+28*T1-229)+2*RO**2*
     2   (324*T1**4-648*T1**3+317*T1**2+7*T1-64)+16*RO**3*(9*T1**2-9*T1+
     3   4))*VLPM/( 1152.D0*B*(T1-1)**2*T1**2)+DD
      DD = DD-(8*(T1-1)**2*(315*T1**3-413*T1**2+160*T1+2)+4*RO*(T1-1)**2
     1   *(9*T1-1)*(20*T1-1)+RO**2*(2*T1-3)*(2*T1-1)*(9*T1-1))*VDW/( 115
     2   2.D0*(T1-1)**3*T1)
      DD = (8*T1**2*(315*T1**3-532*T1**2+279*T1-64)-4*RO*T1**2*(9*T1-8)*
     1   (20*T1-19)+RO**2*(2*T1-1)*(2*T1+1)*(9*T1-8))*VDT/( 1152.D0*(T1-
     2   1)*T1**3)+DD
      DD = (-2*(T1-1)*T1*(2592*T1**4-5184*T1**3+4721*T1**2-2129*T1+282)-
     1   2*RO**2*(18*T1**4-36*T1**3+57*T1**2-39*T1-1)-RO*(T1-1)*T1*(117*
     2   T1**2-117*T1-2)+RO**3*(18*T1**2-18*T1-1))*VDMP/( 576.D0*B*(T1-1
     3   )**2*T1**2)+DD
      DD = DD-9*(-2*(24*T1**4-48*T1**3+49*T1**2-25*T1+4)+RO*(16*T1**4-32
     1   *T1**3+49*T1**2-33*T1+6)+RO**2*(-7*T1**2+7*T1-3)+RO**3)*VDMP/(
     2   64.D0*B**3*(T1-1)*T1)
      DD = DD-9*(2*T1-1)**2*(8*(T1**2-T1+1)+4*RO*(T1-2)*(T1+1)+3*RO**2)*
     1   VDMP/( 128.D0*B**5*(T1-1)*T1)
      TMP0 = 4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T1+RO**2
      TMP0 = (RO-2)*(18*T1**2-18*T1-1)*TMP0*VDMB/( 576.D0*B*(T1-1)**2*T1
     1   **2)
      DD = TMP0+DD
      TMP0 = 4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T1+RO**2
      TMP0 = -(RO-2)*SRL2L*(18*T1**2-18*T1-1)*TMP0/( 2304.D0*(T1-1)**2*T
     1   1**2)
      DD = TMP0+DD
      DD = DD-9*SRL212*(2*T1**2-2*T1+1)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*
     1   RO*(T1-1)*T1+RO**2)/( 64.D0*(T1-1)**2*T1**2)
      DD = DD-SL3525*(3*T1-1)*(3*T1+1)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*R
     1   O*(T1-1)*T1+RO**2)/( 64.D0*(T1-1)**2*T1**2)
      DD = DD-SL3515*(3*T1-4)*(3*T1-2)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*R
     1   O*(T1-1)*T1+RO**2)/( 64.D0*(T1-1)**2*T1**2)
      DD = DD-PI**2*(2*(T1-1)*T1*(2016*T1**4-4032*T1**3+3889*T1**2-1873*
     1   T1+298)+RO*(T1-1)*T1*(576*T1**4-1152*T1**3+373*T1**2+203*T1+14)
     2   +2*RO**2*(162*T1**4-324*T1**3+121*T1**2+41*T1+3)+3*RO**3*(18*T1
     3   **2-18*T1-1))/( 6912.D0*B*(T1-1)**2*T1**2)
      DD = DD-9*(2*T1-1)**4/( 64.D0*B**2*(T1-1)*T1)
      DD = DD-3*PI**2*(-2*(24*T1**4-48*T1**3+49*T1**2-25*T1+4)+RO*(16*T1
     1   **4-32*T1**3+49*T1**2-33*T1+6)+RO**2*(-7*T1**2+7*T1-3)+RO**3)/(
     2    256.D0*B**3*(T1-1)*T1)
      DD = DD-3*PI**2*(2*T1-1)**2*(8*(T1**2-T1+1)+4*RO*(T1-2)*(T1+1)+3*R
     1   O**2)/( 512.D0*B**5*(T1-1)*T1)
      TMP0 = 8*RO*(T1-1)**2*(PI**2*(T1**2+6*T1)+204*T1-60)+32*(T1-1)**3*
     1   (PI**2*T1**2-24*T1**2+228*T1-96)+12*RO**2*(T1-1)*(PI**2*(T1+1)+
     2   8*T1-2)+3*PI**2*RO**3
      TMP0 = AUINV*(9*T1-1)*TMP0/( 27648.D0*(T1-1)**3*T1)
      DD = TMP0+DD
      DD = DD-ATINV*(9*T1-8)*(-32*T1**3*(PI**2*(T1**2-2*T1+1)-24*T1**2-1
     1   80*T1+108)+8*RO*T1**2*(PI**2*(T1**2-8*T1+7)-204*T1+144)+12*RO**
     2   2*T1*(8*T1+PI**2*(T1-2)-6)+3*PI**2*RO**3)/( 27648.D0*(T1-1)*T1*
     3   *3)
      TMP0 = 2*RO**2*(PI**2*(81*T1**6-243*T1**5+221*T1**4-37*T1**3-57*T1
     1   **2+35*T1-4)-648*T1**6+1944*T1**5-1674*T1**4+108*T1**3+462*T1**
     2   2-192*T1)
      TMP0 = TMP0-2*(T1-1)**2*T1**2*(PI**2*(648*T1**4-1296*T1**3+1083*T1
     1   **2-435*T1+40)-3456*T1**4+6912*T1**3+516*T1**2-3972*T1+1293)
      TMP0 = TMP0-3*RO*(T1-1)*T1*(PI**2*(96*T1**4-192*T1**3+209*T1**2-11
     1   3*T1+16)+(72*NLF-1296)*T1**4+(2592-144*NLF)*T1**3+(90*NLF-1728)
     2   *T1**2+(432-18*NLF)*T1+64)
      TMP0 = TMP0/( 3456.D0*(T1-1)**3*T1**3)
      DD = TMP0+DD
      HQHDGG = DD
      RETURN
      END
      FUNCTION HQHDQA(T1,RO,NL)
      IMPLICIT DOUBLE PRECISION (A-Z)
      INTEGER NL
      DATA PI/3.141 592 653 589 793/
      NLF = NL
      T2 = 1-T1
      B = DSQRT(1-RO)
      LP = (B+1)/ 2.D0
      LM = (1-B)/ 2.D0
      AT = T1
      AW = T2
      VLTM = DLOG(4*AT/RO)
      VLPM = DLOG(LP/LM)
      VLSM = DLOG(4/RO)
      VLWM = DLOG(4*AW/RO)
      VLBL = DLOG(B/LM)
      VDW = DDILOG((AW-RO/ 4.D0)/AW)-VLWM**2/ 2.D0
      VDT = DDILOG((AT-RO/ 4.D0)/AT)-VLTM**2/ 2.D0
      VDMP = DDILOG(-LM/LP)
      VDMB = VLBL**2/ 2.D0+DDILOG(-LM/B)
      AUINV = 1/(RO/ 4.D0-AW)
      ATINV = 1/(RO/ 4.D0-AT)
      SRLGPR = DLOG((1-B)*T2/((B+1)*T1))*DLOG((B+1)*T2/((1-B)*T1))
      SRL21P = DDILOG(1-(B+1)*T1/((1-B)*T2))
      SRL21M = DDILOG(1-(1-B)*T1/((B+1)*T2))
      SRL22P = DDILOG(1-(B+1)*T2/((1-B)*T1))
      SRL22M = DDILOG(1-(1-B)*T2/((B+1)*T1))
      SRLG1 = DLOG((B+1)/(1-B))/B
      SRL2L = (DDILOG(-4*B/(1-B)**2)-DDILOG(4*B/(B+1)**2))/B
      SRL212 = DLOG(4*T1*T2/RO)**2/ 2.D0+DDILOG(1-RO/( 4.D0*T1*T2))
      SL3515 = SRLGPR+B**2*SRLG1**2+SRL21P+SRL21M
      SL3525 = SRLGPR+B**2*SRLG1**2+SRL22P+SRL22M
      DD = -(2*(28*T1**2-46*T1+23)+5*RO)*VLSM*VLWM/ 27.D0
      DD = AUINV*(2*(T1-1)*T1+RO)*VLWM/ 3.D0+DD
      DD = DD-(2*(28*T1**2-10*T1+5)+5*RO)*VLSM*VLTM/ 27.D0
      DD = ATINV*(2*(T1-1)*T1+RO)*VLTM/ 3.D0+DD
      DD = (152*(2*T1**2-2*T1+1)+67*RO)*VLSM**2/ 54.D0+DD
      DD = 2*(RO-2)*(2*(2*T1**2-2*T1+1)+RO)*VLPM*VLSM/( 27.D0*B)+DD
      DD = DD-2*(2*T1-1)*(RO*(2*T1-1)-6*T1+5)*VLSM/( 3.D0*B**2)
      DD = DD-(2*(6*T1**2-10*T1+3)+RO*(8*T1-3))*VLSM/( 3.D0*B**4)
      DD = 4*((2*NLF-37)*(2*T1**2-2*T1+1)+(NLF-14)*RO)*VLSM/ 27.D0+DD
      DD = (2*(68*T1**2-95*T1+43)+2*RO*T1*(2*T1+7)+RO**2)*VLPM**2/( 54.D
     1   0*B)+DD
      DD = (2*(T1-1)+RO)*(RO*(2*T1-1)-6*T1+5)*VLPM**2/( 6.D0*B**3)+DD
      DD = (4*RO*(T1-1)*(T1+2)+8*(T1-1)**2+3*RO**2)*VLPM**2/( 12.D0*B**5
     1   )+DD
      DD = 2*B*(RO+2)*((2*T1-1)**2+RO)*VLPM/ 27.D0+DD
      DD = DD-(-2*(26*T1**2-26*T1+15)+RO*(24*T1**2-24*T1-1)+8*RO**2)*VLP
     1   M/( 27.D0*B)
      DD = (2*(2*T1-1)+RO)*VDW/ 3.D0+DD
      DD = (RO-2*(2*T1-1))*VDT/ 3.D0+DD
      DD = 4*(8*T1**2+RO*T1-11*T1+5)*VDMP/( 3.D0*B)+DD
      DD = 2*(2*(T1-1)+RO)*(RO*(2*T1-1)-6*T1+5)*VDMP/( 3.D0*B**3)+DD
      DD = (4*RO*(T1-1)*(T1+2)+8*(T1-1)**2+3*RO**2)*VDMP/( 3.D0*B**5)+DD
      DD = DD-2*(RO-2)*(2*(2*T1**2-2*T1+1)+RO)*VDMB/( 27.D0*B)
      DD = (RO-2)*SRL2L*(2*(2*T1**2-2*T1+1)+RO)/ 54.D0+DD
      DD = DD-2*SRL212*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0
      DD = SL3525*(2*(2*T1**2-2*T1+1)+RO)/ 3.D0+DD
      DD = SL3515*(2*(2*T1**2-2*T1+1)+RO)/ 3.D0+DD
      DD = PI**2*(56*T1**2+RO*T1*(8*T1+1)-83*T1+2*RO**2+37)/( 81.D0*B)+D
     1   D
      DD = 2*(2*T1-1)**2/( 3.D0*B**2)+DD
      DD = PI**2*(2*(T1-1)+RO)*(RO*(2*T1-1)-6*T1+5)/( 18.D0*B**3)+DD
      DD = PI**2*(4*RO*(T1-1)*(T1+2)+8*(T1-1)**2+3*RO**2)/( 36.D0*B**5)+
     1   DD
      DD = DD-(4*(PI**2*(22*T1**2-22*T1+11)+(40*NLF-392)*T1**2+(392-40*N
     1   LF)*T1+20*NLF-223)+RO*(96*T1**2-96*T1+31*PI**2+40*NLF-344)+24*R
     2   O**2)/ 162.D0
      HQHDQA = DD
      RETURN
      END
      FUNCTION HQHPGG(TX,T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(TX.EQ.0)THEN
      T2 = 1-T1
      B = DSQRT(1-RO)
      VLSM = DLOG(4/RO)
      SRL12 = DLOG(T1/T2)
      SRLGRO = DLOG(4*T1*T2/RO)
      SRLG1 = DLOG((B+1)/(1-B))/B
      PP = -(9*T1**2-9*T1+4)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T
     1   1+RO**2)*VLSM/( 4.D0*(T1-1)**2*T1**2)
      PP = 9*SRLGRO*(2*T1**2-2*T1+1)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*
     1   (T1-1)*T1+RO**2)/( 32.D0*(T1-1)**2*T1**2)+PP
      TMP0 = 4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T1+RO**2
      TMP0 = -(RO-2)*SRLG1*(18*T1**2-18*T1-1)*TMP0/( 576.D0*(T1-1)**2*T1
     1   **2)
      PP = TMP0+PP
      PP = 9*SRL12*(2*T1-1)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T1
     1   +RO**2)/( 32.D0*(T1-1)**2*T1**2)+PP
      PP = (9*T1**2-9*T1+4)*(4*(T1-1)*T1*(2*T1**2-2*T1+1)+4*RO*(T1-1)*T1
     1   +RO**2)/( 18.D0*(T1-1)**2*T1**2)+PP
      HQHPGG = PP
      RETURN
      ELSE
      T2 = -TX-T1+1
      T11 = 1/(1-T1)
      ITX = 1/(1-TX)
      T22 = 1/(TX+T1)
      DRO = 1/(4*TX+RO)
      B = DSQRT((1-TX)**2-RO)
      DLAM2 = 1/(TX**2-2*TX-RO+1)
      VLSM = DLOG(4/RO)
      RLGXRO = DLOG((4*TX+RO)/RO)/TX
      RLG12 = DLOG(T1/(1-T2))/TX
      RLG21 = DLOG(T2/(1-T1))/TX
      RLG11 = DLOG(T1/(1-T1))
      RLG22 = DLOG(T2/(1-T2))
      RLGRO = DLOG(4*(1-T1)*(1-T2)/RO)
      RR = T1*(T1-RO*(1-T2))
      D2435 = 1/RR
      RR = DSQRT(RR)
      RL3524 = 2*DLOG((T1+RR)/(T1-RR))/RR
      RR = T2*(T2-RO*(1-T1))
      D1435 = 1/RR
      RR = DSQRT(RR)
      RL3514 = 2*DLOG((T2+RR)/(T2-RR))/RR
      RR = T1*(T1*TX**2+RO*(1-T1))
      RR = DSQRT(RR)
      RL3414 = 2*DLOG((T1*TX+RR)/(RR-T1*TX))/(RR*TX)
      RR = T2*(T2*TX**2+RO*(1-T2))
      RR = DSQRT(RR)
      RL3424 = 2*DLOG((T2*TX+RR)/(RR-T2*TX))/(RR*TX)
      RR = TX**2+RO*(1-T1)*(1-T2)
      RR = DSQRT(RR)
      RL1424 = 2*DLOG((TX+RR)/(RR-TX))/(RR*TX)
      RL34 = DLOG((2*TX**2+2*B*TX-2*TX-RO)/(2*TX**2-2*B*TX-2*TX-RO))/(B*
     1   TX)
      RL35 = DLOG((2*TX+RO-2*B-2)/(2*TX+RO+2*B-2))/B
      TMP0 = T1*(-T2-T1+1)*T22**2+(-T2-T1+1)/T1+T1/(-T2-T1+1)
      TMP0 = -3*(3*T2**2-3*T2+4/ 3.D0)*(4*RO*T2**2/(T1*T22)-4*RO*T2/(T1*
     1   T22)+RO**2/(T1**2*T22**2)+8*T2**4-16*T2**3+12*T2**2-4*T2)*T22**
     2   2*TMP0/( 8.D0*T1*T2**2)
      PP = T11**2*(-T2-T1+1)*T2+T2/(-T2-T1+1)+(-T2-T1+1)/T2
      PP = -3*PP*(3*T1**2-3*T1+4/ 3.D0)*T11**2*(4*RO*T1**2/(T11*T2)-4*RO
     1   *T1/(T11*T2)+RO**2/(T11**2*T2**2)+8*T1**4-16*T1**3+12*T1**2-4*T
     2   1)/( 8.D0*T1**2*T2)
      PP = TMP0+PP
      PP = PP*(-T2-T1+1)*VLSM
      TMP0 = -4*(486*T2**5+(-225*T1**2+1622*T1-1530)*T2**4+(-207*T1**3+2
     1   866*T1**2-3399*T1+1692)*T2**3+(1539*T1**3-2997*T1**2+1761*T1-75
     2   6)*T2**2+(-293*T1**3-40*T1**2+187*T1+126)*T2-207*T1**3+396*T1**
     3   2-171*T1-18)
      TMP0 = TMP0+2*RO*((36*T1+324)*T2**4+(72*T1**2+749*T1-716)*T2**3+(4
     1   21*T1**2-906*T1+476)*T2**2+(-13*T1**2+125*T1-100)*T2-32*T1**2-4
     2   *T1+16)+RO**2*(-(36*T1+360)*T2**3-(356*T1-673)*T2**2-(354-358*T
     3   1)*T2-46*T1+41)+2*RO**3*(T2-1)*(9*T2-1)
      TMP0 = -RLGXRO*(T2+T1-1)*T22**3*TMP0/( 2304.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -32*((3*T1**2-20*T1-13)*T2**2+(-20*T1**2-6*T1+26)*T2-13*T1*
     1   *2+26*T1-13)*TX+8*RO*((T1**2+18*T1+47)*T2**2+(18*T1**2+76*T1-94
     2   )*T2+47*T1**2-94*T1+47)-12*RO**2*((T1+6)*T2+6*T1-6)+3*RO**3
      TMP0 = DRO*(T2+T1-1)*(T2+2*T1-1)*(9*T2-1)*T22**3*TMP0/( 576.D0*T1*
     1   T2*(4*TX+RO))
      PP = TMP0+PP
      PP = PP-(T2+T1-1)**2*(4*RO-T1*(81*T2**3-162*T2**2+125*T2-36))*T22*
     1   *3/( 18.D0*T1)
      TMP0 = RO*(27*T2-4)-2*(27*T1*T2**2+(-4*T1-27)*T2-27*T1+27)
      TMP0 = RLGXRO*(T2+T1-1)**2*T22**2*TMP0/( 128.D0*T2)
      PP = TMP0+PP
      TMP0 = 4*T2*(5184*T2**7+(10368*T1-25920)*T2**6+(10368*T1**2-41472*
     1   T1+54144)*T2**5+(-32076*T1**2+67131*T1-61056)*T2**4+(-1872*T1**
     2   3+33152*T1**2-56453*T1+39744)*T2**3+(-1800*T1**4-8139*T1**3-113
     3   60*T1**2+25485*T1-14400)*T2**2+(-5632*T1**4+11315*T1**3-656*T1*
     4   *2-4983*T1+2304)*T2+424*T1**4-920*T1**3+572*T1**2-76*T1)*TX
      TMP0 = RO**2*((5184*T1+7264)*T2**4+(4608*T1**2-5703*T1-19712)*T2**
     1   3+(-7584*T1**2-7229*T1+19936)*T2**2+(-2392*T1**2+12324*T1-9792)
     2   *T2+2304*T1**2-4608*T1+2304)*TX+TMP0
      TMP0 = TMP0+RO*T2*(5184*T2**7+(10368*T1-25920)*T2**6+(10368*T1**2-
     1   20736*T1+59904)*T2**5+(29952*T1**2-10733*T1-84096)*T2**4+(58752
     2   *T1**3-175028*T1**2+62123*T1+74304)*T2**3+(18432*T1**4-168599*T
     3   1**3+248724*T1**2-70771*T1-37440)*T2**2+(-42320*T1**4+141423*T1
     4   **3-144396*T1**2+39577*T1+8064)*T2+10768*T1**4-31320*T1**3+3038
     5   0*T1**2-9828*T1)+4*RO**3*(364*T2**4+(359*T1-1052)*T2**3+(67*T1*
     6   *2-990*T1+1156)*T2**2+(-307*T1**2+919*T1-612)*T2+144*T1**2-288*
     7   T1+144)
      TMP0 = -T22**2*TMP0/( 1152.D0*T1**2*T2**2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = -154*T2**3+2*RO*(T2**2-T2+T1**2-T1)-(66*T1-311)*T2**2-(66*T
     1   1**2-132*T1+157)*T2-154*T1**3+311*T1**2-157*T1
      TMP0 = -RLGXRO*T11*(T2+T1-1)**2*T22*TMP0/( 128.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = 9*T11*(T2+T1-1)*(-18*T2**3+RO*(T2**2-T2+T1**2-T1)-(12*T1-41
     1   )*T2**2-(12*T1**2-24*T1+23)*T2-18*T1**3+41*T1**2-23*T1)*T22/( 1
     2   28.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -T1**2*(3240*T2**6+(9720*T1-9720)*T2**5+(9072*T1**2-23040*T
     1   1+12960)*T2**4+(1944*T1**3-13716*T1**2+23548*T1-9720)*T2**3+(-6
     2   48*T1**4+2772*T1**3+8414*T1**2-13300*T1+3240)*T2**2+(4392*T1**4
     3   -1904*T1**3-4723*T1**2+3044*T1)*T2+1610*T1**4-3063*T1**3+1433*T
     4   1**2+28*T1)
      TMP0 = TMP0+2*RO*T1*(2430*T2**5+(5670*T1-6480)*T2**4+(4068*T1**2-1
     1   1393*T1+6480)*T2**3+(990*T1**3-5450*T1**2+7396*T1-3240)*T2**2+(
     2   324*T1**4-849*T1**3+1383*T1**2-1673*T1+810)*T2-33*T1**4+51*T1**
     3   3-9*T1**2)-2*RO**2*(810*T2**4+(1215*T1-2025)*T2**3+(9*T1**3+485
     4   *T1**2-2025*T1+1620)*T2**2+(9*T1**4+76*T1**3-485*T1**2+810*T1-4
     5   05)*T2-13*T1**4+11*T1**3)+RO**3*T1**2*(8*T2-T1)
      TMP0 = -RLGXRO*(T2+T1-1)*T22*TMP0/( 1152.D0*T1**4*T2)
      PP = TMP0+PP
      PP = 9*RLGRO*(2*(8*T2**3+(14*T1-21)*T2**2+(16*T1**3-27*T1+22)*T2+1
     1   6*T1**4-32*T1**3+12*T1**2+9*T1-9)+RO*(2*T2**2+3*T2+8*T1**2-8*T1
     2   -5)+4*RO**2)*T22/( 128.D0*T1)+PP
      PP = RLG22*(2*T2*(2*T2**3+(-T1-2)*T2**2-T1**2*T2-2*T1**3+T1**2-T1)
     1   +RO*(-(8*T1-2)*T2**2-(6*T1**2-5*T1+2)*T2+2*T1**2-T1)+RO**2*(T2*
     2   *2+(3*T1+1)*T2+2*T1**2-T1))*T22/( 64.D0*T1*T2**2)+PP
      TMP0 = -4*T1*(31*T2**3+(-72*T1**3+27*T1**2-10*T1-11)*T2**2+(-72*T1
     1   **4+144*T1**3-25*T1**2-44*T1)*T2+72*T1**4-144*T1**3+72*T1**2)+R
     2   O*(26*T2**3+(-27*T1**2-39*T1-8)*T2**2+(-36*T1**3+149*T1**2-45*T
     3   1-18)*T2+36*T1**3-58*T1**2+20*T1)+2*RO**2*(9*T2**2+(T1+7)*T2-2*
     4   T1**2+T1)
      TMP0 = -RLG21*(T2+T1-1)**2*T22*TMP0/( 128.D0*T1**2*T2**2)
      PP = TMP0+PP
      TMP0 = 4*T1**2*(180*T2**6+(360*T1-360)*T2**5+(144*T1**2-666*T1+360
     1   )*T2**4+(-108*T1**3-306*T1**2+517*T1-180)*T2**3+(-72*T1**4+36*T
     2   1**3+89*T1**2-209*T1)*T2**2+(2*T1**3-35*T1**2)*T2-72*T1**4+72*T
     3   1**3)
      TMP0 = TMP0-RO*T1*(1080*T2**5+(1440*T1-1800)*T2**4+(243*T1**2-2026
     1   *T1+1080)*T2**3+(-72*T1**3-354*T1**2+604*T1-360)*T2**2+(104*T1*
     2   *3-45*T1**2-18*T1)*T2-40*T1**3+20*T1**2)+2*RO**2*(180*T2**4+(90
     3   *T1-270)*T2**3+(-27*T1**2-90*T1+90)*T2**2+(-T1**3-7*T1**2)*T2+2
     4   *T1**4-T1**3)
      TMP0 = RLG12*(T2+T1-1)**2*T22*TMP0/( 128.D0*T1**4*T2**2)
      PP = TMP0+PP
      TMP0 = 2*RO*((T1**2-3*T1+1)*T2-3*T1**2+T1-1)+RO**2*((T1+1)*T2+2*T1
     1   **2+T1+1)-4*T1*((T1+1)*T2+T1**2-T1)
      TMP0 = -RL3524*(T2+T1-1)*T22*TMP0/( 128.D0*T1)
      PP = TMP0+PP
      TMP0 = -2*(T2**3+(5*T1+1)*T2**2+(8*T1**2+T1)*T2+8*T1**3)*TX+RO*(T2
     1   **3-4*T1*T2**2+(-2*T1**2-T1-1)*T2-2*T1**2+T1)+RO**2*T2*(T2+1)
      TMP0 = -9*RL3424*(T2+T1-1)**2*T22*TMP0/( 128.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -4*((T1-1)*T2+T1**2-T1+2)*TX+RO**2*((T1-1)*T2+2*T1**2-3*T1+
     1   3)+2*RO*(T2+1)*(T2+T1**2-T1-1)
      TMP0 = RL1424*(T2+T1-1)**2*T22*TMP0/( 128.D0*T1)
      PP = TMP0+PP
      PP = PP-D2435*(RO-2*T1)*(16*T1**2+16*RO*T1+RO**2)*(T2+T1-1)**2*T22
     1   /( 72.D0*T1)
      TMP0 = 4*T1**3*((36*T1-72)*T2**5+(-144*T1**2-468*T1-25)*T2**4+(-10
     1   8*T1**3-414*T1**2+424*T1+270)*T2**3+(-144*T1**4-270*T1**3+711*T
     2   1**2-139*T1-177)*T2**2+(-144*T1**5+576*T1**4-520*T1**3-59*T1**2
     3   +147*T1+4)*T2+144*T1**5-432*T1**4+432*T1**3-144*T1**2)*TX
      TMP0 = TMP0+RO*T1*(6912*T2**7+(20736*T1-27648)*T2**6+(36*T1**3+223
     1   56*T1**2-64732*T1+44544)*T2**5+(-144*T1**4+6984*T1**3-53223*T1*
     2   *2+79032*T1-39936)*T2**4+(-108*T1**5-3258*T1**4-10806*T1**3+495
     3   30*T1**2-56064*T1+25344)*T2**3+(-144*T1**6-2106*T1**5+5755*T1**
     4   4+6643*T1**3-29251*T1**2+30280*T1-12288)*T2**2+(-144*T1**7+432*
     5   T1**6+1080*T1**5-2183*T1**4-4021*T1**3+11024*T1**2-9252*T1+3072
     6   )*T2+144*T1**7-288*T1**6+440*T1**5-1024*T1**4+1164*T1**3-436*T1
     7   **2)
      TMP0 = TMP0+RO**2*(-(1728*T1+1152)*T2**6-(3456*T1**2-1728*T1-4032)
     1   *T2**5-(2151*T1**3-5135*T1**2-2688*T1+5120)*T2**4-(-288*T1**4-3
     2   673*T1**3+3375*T1**2+2688*T1-3200)*T2**3-(-423*T1**5+278*T1**4+
     3   3738*T1**3-3889*T1**2-576*T1+1920)*T2**2-(-36*T1**6+296*T1**5+4
     4   79*T1**4-2573*T1**3+913*T1**2+2112*T1-1472)*T2-36*T1**6-278*T1*
     5   *5+926*T1**4-357*T1**3-1280*T1**2+1536*T1-512)
      TMP0 = TMP0+2*RO**3*(144*T2**5+(288*T1-360)*T2**4+(17*T1**2-432*T1
     1   +280)*T2**3+(-25*T1**3+80*T1**2+128*T1-120)*T2**2+(-4*T1**4+56*
     2   T1**3-65*T1**2-112*T1+120)*T2+30*T1**4-63*T1**3-32*T1**2+128*T1
     3   -64)
      TMP0 = T22*TMP0/( 128.D0*T1**4*T2**2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = 4*((207*T1**3-1539*T1**2+293*T1+207)*T2**3+(225*T1**4-2866*
     1   T1**3+2997*T1**2+40*T1-396)*T2**2+(-1622*T1**4+3399*T1**3-1761*
     2   T1**2-187*T1+171)*T2-486*T1**5+1530*T1**4-1692*T1**3+756*T1**2-
     3   126*T1+18)
      TMP0 = TMP0+2*RO*((72*T1**3+421*T1**2-13*T1-32)*T2**2+(36*T1**4+74
     1   9*T1**3-906*T1**2+125*T1-4)*T2+324*T1**4-716*T1**3+476*T1**2-10
     2   0*T1+16)+RO**2*(-(36*T1**3+356*T1**2-358*T1+46)*T2-360*T1**3+67
     3   3*T1**2-354*T1+41)+2*RO**3*(T1-1)*(9*T1-1)
      TMP0 = -RLGXRO*T11**3*(T2+T1-1)*TMP0/( 2304.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -32*((3*T1**2-20*T1-13)*T2**2+(-20*T1**2-6*T1+26)*T2-13*T1*
     1   *2+26*T1-13)*TX+8*RO*((T1**2+18*T1+47)*T2**2+(18*T1**2+76*T1-94
     2   )*T2+47*T1**2-94*T1+47)-12*RO**2*((T1+6)*T2+6*T1-6)+3*RO**3
      TMP0 = DRO*(9*T1-1)*T11**3*(T2+T1-1)*(2*T2+T1-1)*TMP0/( 576.D0*T1*
     1   T2*(4*TX+RO))
      PP = TMP0+PP
      PP = PP-T11**3*(T2+T1-1)**2*(4*RO-(81*T1**3-162*T1**2+125*T1-36)*T
     1   2)/( 18.D0*T2)
      TMP0 = RO*(27*T1-4)-2*((27*T1**2-4*T1-27)*T2-27*T1+27)
      TMP0 = RLGXRO*T11**2*(T2+T1-1)**2*TMP0/( 128.D0*T1)
      PP = TMP0+PP
      TMP0 = -4*T1*((1800*T1**2+5632*T1-424)*T2**4+(1872*T1**3+8139*T1**
     1   2-11315*T1+920)*T2**3+(-10368*T1**5+32076*T1**4-33152*T1**3+113
     2   60*T1**2+656*T1-572)*T2**2+(-10368*T1**6+41472*T1**5-67131*T1**
     3   4+56453*T1**3-25485*T1**2+4983*T1+76)*T2-5184*T1**7+25920*T1**6
     4   -54144*T1**5+61056*T1**4-39744*T1**3+14400*T1**2-2304*T1)*TX
      TMP0 = RO**2*((4608*T1**3-7584*T1**2-2392*T1+2304)*T2**2+(5184*T1*
     1   *4-5703*T1**3-7229*T1**2+12324*T1-4608)*T2+7264*T1**4-19712*T1*
     2   *3+19936*T1**2-9792*T1+2304)*TX+TMP0
      TMP0 = TMP0+RO*T1*((18432*T1**2-42320*T1+10768)*T2**4+(58752*T1**3
     1   -168599*T1**2+141423*T1-31320)*T2**3+(10368*T1**5+29952*T1**4-1
     2   75028*T1**3+248724*T1**2-144396*T1+30380)*T2**2+(10368*T1**6-20
     3   736*T1**5-10733*T1**4+62123*T1**3-70771*T1**2+39577*T1-9828)*T2
     4   +5184*T1**7-25920*T1**6+59904*T1**5-84096*T1**4+74304*T1**3-374
     5   40*T1**2+8064*T1)+4*RO**3*((67*T1**2-307*T1+144)*T2**2+(359*T1*
     6   *3-990*T1**2+919*T1-288)*T2+364*T1**4-1052*T1**3+1156*T1**2-612
     7   *T1+144)
      TMP0 = -T11**2*TMP0/( 1152.D0*T1**2*T2**2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = -T2**2*((648*T1**2-4392*T1-1610)*T2**4+(-1944*T1**3-2772*T1
     1   **2+1904*T1+3063)*T2**3+(-9072*T1**4+13716*T1**3-8414*T1**2+472
     2   3*T1-1433)*T2**2+(-9720*T1**5+23040*T1**4-23548*T1**3+13300*T1*
     3   *2-3044*T1-28)*T2-3240*T1**6+9720*T1**5-12960*T1**4+9720*T1**3-
     4   3240*T1**2)
      TMP0 = TMP0-2*RO*T2*((324*T1-33)*T2**4+(990*T1**2-849*T1+51)*T2**3
     1   +(4068*T1**3-5450*T1**2+1383*T1-9)*T2**2+(5670*T1**4-11393*T1**
     2   3+7396*T1**2-1673*T1)*T2+2430*T1**5-6480*T1**4+6480*T1**3-3240*
     3   T1**2+810*T1)+2*RO**2*((9*T1-13)*T2**4+(9*T1**2+76*T1+11)*T2**3
     4   +(485*T1**2-485*T1)*T2**2+(1215*T1**3-2025*T1**2+810*T1)*T2+810
     5   *T1**4-2025*T1**3+1620*T1**2-405*T1)+RO**3*T2**2*(T2-8*T1)
      TMP0 = RLGXRO*T11*(T2+T1-1)*TMP0/( 1152.D0*T1*T2**4)
      PP = TMP0+PP
      PP = 9*RLGRO*T11*(2*(16*T2**4+(16*T1-32)*T2**3+12*T2**2+(14*T1**2-
     1   27*T1+9)*T2+8*T1**3-21*T1**2+22*T1-9)+RO*(8*T2**2-8*T2+2*T1**2+
     2   3*T1-5)+4*RO**2)/( 128.D0*T2)+PP
      TMP0 = -4*T2**2*((72*T1**2+72)*T2**4+(108*T1**3-36*T1**2-2*T1-72)*
     1   T2**3+(-144*T1**4+306*T1**3-89*T1**2+35*T1)*T2**2+(-360*T1**5+6
     2   66*T1**4-517*T1**3+209*T1**2)*T2-180*T1**6+360*T1**5-360*T1**4+
     3   180*T1**3)
      TMP0 = TMP0+2*RO**2*(2*T2**4+(-T1-1)*T2**3+(-27*T1**2-7*T1)*T2**2+
     1   (90*T1**3-90*T1**2)*T2+180*T1**4-270*T1**3+90*T1**2)+RO*T2*((72
     2   *T1**2-104*T1+40)*T2**3+(-243*T1**3+354*T1**2+45*T1-20)*T2**2+(
     3   -1440*T1**4+2026*T1**3-604*T1**2+18*T1)*T2-1080*T1**5+1800*T1**
     4   4-1080*T1**3+360*T1**2)
      TMP0 = RLG21*T11*(T2+T1-1)**2*TMP0/( 128.D0*T1**2*T2**4)
      PP = TMP0+PP
      TMP0 = -4*T2*((72*T1-72)*T2**4+(72*T1**2-144*T1+144)*T2**3+(-27*T1
     1   **2+25*T1-72)*T2**2+(10*T1**2+44*T1)*T2-31*T1**3+11*T1**2)+RO*(
     2   (36*T1-36)*T2**3+(27*T1**2-149*T1+58)*T2**2+(39*T1**2+45*T1-20)
     3   *T2-26*T1**3+8*T1**2+18*T1)+2*RO**2*(2*T2**2+(-T1-1)*T2-9*T1**2
     4   -7*T1)
      TMP0 = RLG12*T11*(T2+T1-1)**2*TMP0/( 128.D0*T1**2*T2**2)
      PP = TMP0+PP
      PP = RLG11*T11*(-2*T1*(2*T2**3+(T1-1)*T2**2+(T1**2+1)*T2-2*T1**3+2
     1   *T1**2)+RO*(-(6*T1-2)*T2**2-(8*T1**2-5*T1+1)*T2+2*T1**2-2*T1)+R
     2   O**2*(2*T2**2+(3*T1-1)*T2+T1**2+T1))/( 64.D0*T1**2*T2)+PP
      TMP0 = 2*RO*((T1-3)*T2**2+(1-3*T1)*T2+T1-1)+RO**2*(2*T2**2+(T1+1)*
     1   T2+T1+1)-4*T2*(T2**2+(T1-1)*T2+T1)
      TMP0 = -RL3514*T11*(T2+T1-1)*TMP0/( 128.D0*T2)
      PP = TMP0+PP
      TMP0 = -2*(8*T2**3+8*T1*T2**2+(5*T1**2+T1)*T2+T1**3+T1**2)*TX+RO*(
     1   -(2*T1+2)*T2**2-(4*T1**2+T1-1)*T2+T1**3-T1)+RO**2*T1*(T1+1)
      TMP0 = -9*RL3414*T11*(T2+T1-1)**2*TMP0/( 128.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -4*(T2**2+(T1-1)*T2-T1+2)*TX+RO**2*(2*T2**2+(T1-3)*T2-T1+3)
     1   +2*RO*(T1+1)*(T2**2-T2+T1-1)
      TMP0 = RL1424*T11*(T2+T1-1)**2*TMP0/( 128.D0*T2)
      PP = TMP0+PP
      PP = PP-D1435*T11*(RO-2*T2)*(T2+T1-1)**2*(16*T2**2+16*RO*T2+RO**2)
     1   /( 72.D0*T2)
      TMP0 = 4*T2**3*((144*T1-144)*T2**5+(144*T1**2-576*T1+432)*T2**4+(1
     1   08*T1**3+270*T1**2+520*T1-432)*T2**3+(144*T1**4+414*T1**3-711*T
     2   1**2+59*T1+144)*T2**2+(-36*T1**5+468*T1**4-424*T1**3+139*T1**2-
     3   147*T1)*T2+72*T1**5+25*T1**4-270*T1**3+177*T1**2-4*T1)*TX
      TMP0 = TMP0+RO*T2*((144*T1-144)*T2**7+(144*T1**2-432*T1+288)*T2**6
     1   +(108*T1**3+2106*T1**2-1080*T1-440)*T2**5+(144*T1**4+3258*T1**3
     2   -5755*T1**2+2183*T1+1024)*T2**4+(-36*T1**5-6984*T1**4+10806*T1*
     3   *3-6643*T1**2+4021*T1-1164)*T2**3+(-22356*T1**5+53223*T1**4-495
     4   30*T1**3+29251*T1**2-11024*T1+436)*T2**2+(-20736*T1**6+64732*T1
     5   **5-79032*T1**4+56064*T1**3-30280*T1**2+9252*T1)*T2-6912*T1**7+
     6   27648*T1**6-44544*T1**5+39936*T1**4-25344*T1**3+12288*T1**2-307
     7   2*T1)
      TMP0 = TMP0+RO**2*(-(36*T1-36)*T2**6-(423*T1**2-296*T1-278)*T2**5-
     1   (288*T1**3-278*T1**2-479*T1+926)*T2**4-(-2151*T1**4+3673*T1**3-
     2   3738*T1**2+2573*T1-357)*T2**3-(-3456*T1**5+5135*T1**4-3375*T1**
     3   3+3889*T1**2-913*T1-1280)*T2**2-(-1728*T1**6+1728*T1**5+2688*T1
     4   **4-2688*T1**3+576*T1**2-2112*T1+1536)*T2+1152*T1**6-4032*T1**5
     5   +5120*T1**4-3200*T1**3+1920*T1**2-1472*T1+512)
      TMP0 = TMP0+2*RO**3*((4*T1-30)*T2**4+(25*T1**2-56*T1+63)*T2**3+(-1
     1   7*T1**3-80*T1**2+65*T1+32)*T2**2+(-288*T1**4+432*T1**3-128*T1**
     2   2+112*T1-128)*T2-144*T1**5+360*T1**4-280*T1**3+120*T1**2-120*T1
     3   +64)
      TMP0 = -T11*TMP0/( 128.D0*T1**2*T2**4*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = -RO*T1*T2*(30*T2**5+(40*T1-20)*T2**4+(8*T1**2-15*T1+10)*T2*
     1   *3+8*T1**3*T2**2+(40*T1**4-15*T1**3)*T2+30*T1**5-20*T1**4+10*T1
     2   **3)
      TMP0 = TMP0+5*RO**2*(2*T2**5+(T1-1)*T2**4+T1**4*T2+2*T1**5-T1**4)+
     1   2*T1**2*T2**2*(T2+T1)*(10*T2**4+(10*T1-10)*T2**3+(-4*T1**2-5*T1
     2   +10)*T2**2+(10*T1**3-5*T1**2)*T2+10*T1**4-10*T1**3+10*T1**2)
      TMP0 = 9*RLGXRO*(T2+T1-1)**2*TMP0/( 64.D0*T1**4*T2**4)
      PP = TMP0+PP
      PP = PP-9*RLGRO*(2*(16*T2**4+(48*T1-40)*T2**3+(96*T1**2-126*T1+41)
     1   *T2**2+(48*T1**3-126*T1**2+90*T1-17)*T2+16*T1**4-40*T1**3+41*T1
     2   **2-17*T1)+RO*(6*T2**2-5*T2+6*T1**2-5*T1))/( 128.D0*T1*T2)
      PP = 9*ITX*RLG22*(-2*T2*(17*T2**3+(10*T1-21)*T2**2+(9*T1**2-7*T1+1
     1   4)*T2-2*T1**2+2*T1-4)+RO*(-5*T2**3-(4*T1+7)*T2**2-(-T1**2-3*T1-
     2   4)*T2+2*T1**2-4*T1)+2*RO**2*(T2+T1-2))/( 128.D0*T2**2)+PP
      PP = PP-RLG22*(-2*T2*((81*T1-4)*T2**2+(45*T1**2-43*T1+8)*T2+36*T1*
     1   *3-20*T1**2+20*T1-4)+RO*(-45*T1*T2**2-(63*T1**2-17*T1-4)*T2+22*
     2   T1**2-38*T1)+2*RO**2*((18*T1+1)*T2+18*T1**2-8*T1-2))/( 128.D0*T
     3   1*T2**2)
      TMP0 = RO*(2*T2**3+(3*T1**2+2*T1+4)*T2**2+(-5*T1**3+2*T1**2+8*T1)*
     1   T2-6*T1**3+4*T1**2)-4*T1*T2*((8*T1**2-4*T1-1)*T2**2+(-4*T1-4)*T
     2   2-4*T1**3+3*T1**2-6*T1)+2*RO**2*(T2**2+(2*T1-4)*T2+3*T1**2-4*T1
     3   )
      TMP0 = 9*ITX*RLG21*(T2+T1-1)**2*TMP0/( 128.D0*T1**2*T2**2)
      PP = TMP0+PP
      TMP0 = 2*RO**2*(18*T1**2*T2**5+(18*T1**3-73*T1**2+72*T1-104)*T2**4
     1   +(18*T1**4-132*T1**3+223*T1**2-208*T1+136)*T2**3+(18*T1**5-271*
     2   T1**4+241*T1**3-156*T1**2+64*T1-32)*T2**2+(-414*T1**5+468*T1**4
     3   -208*T1**3+64*T1**2)*T2-252*T1**6+486*T1**5-338*T1**4+136*T1**3
     4   -32*T1**2)
      TMP0 = TMP0-4*T1**2*T2**2*((72*T1-72)*T2**5+(396*T1**2-306*T1+272)
     1   *T2**4+(1116*T1**3-1080*T1**2+637*T1-400)*T2**3+(1584*T1**4-229
     2   5*T1**3+1505*T1**2-653*T1+264)*T2**2+(1116*T1**5-2169*T1**4+209
     3   5*T1**3-1058*T1**2+310*T1-64)*T2+396*T1**6-1008*T1**5+1280*T1**
     4   4-940*T1**3+336*T1**2-64*T1)
      TMP0 = TMP0+RO*T1*T2*((216*T1**2-188*T1+832)*T2**4+(1323*T1**3-121
     1   0*T1**2+1310*T1-1088)*T2**3+(3159*T1**4-3199*T1**3+1370*T1**2-5
     2   46*T1+256)*T2**2+(3636*T1**5-5138*T1**4+2768*T1**3-546*T1**2)*T
     3   2+1656*T1**6-3528*T1**5+3064*T1**4-1448*T1**3+256*T1**2)
      TMP0 = -RLG21*(T2+T1-1)*TMP0/( 128.D0*T1**4*T2**4)
      PP = TMP0+PP
      TMP0 = RO*(-(5*T1+6)*T2**3-(-3*T1**2-2*T1-4)*T2**2-(-2*T1**2-8*T1)
     1   *T2+2*T1**3+4*T1**2)+4*T1*T2*(4*T2**3+(-8*T1**2-3)*T2**2+(4*T1*
     2   *2+4*T1+6)*T2+T1**2+4*T1)+2*RO**2*(3*T2**2+(2*T1-4)*T2+T1**2-4*
     3   T1)
      TMP0 = 9*ITX*RLG12*(T2+T1-1)**2*TMP0/( 128.D0*T1**2*T2**2)
      PP = TMP0+PP
      TMP0 = -RO*T1*T2*(1656*T2**6+(3636*T1-3528)*T2**5+(3159*T1**2-5138
     1   *T1+3064)*T2**4+(1323*T1**3-3199*T1**2+2768*T1-1448)*T2**3+(216
     2   *T1**4-1210*T1**3+1370*T1**2-546*T1+256)*T2**2+(-188*T1**4+1310
     3   *T1**3-546*T1**2)*T2+832*T1**4-1088*T1**3+256*T1**2)
      TMP0 = TMP0+4*T1**2*T2**2*(396*T2**6+(1116*T1-1008)*T2**5+(1584*T1
     1   **2-2169*T1+1280)*T2**4+(1116*T1**3-2295*T1**2+2095*T1-940)*T2*
     2   *3+(396*T1**4-1080*T1**3+1505*T1**2-1058*T1+336)*T2**2+(72*T1**
     3   5-306*T1**4+637*T1**3-653*T1**2+310*T1-64)*T2-72*T1**5+272*T1**
     4   4-400*T1**3+264*T1**2-64*T1)
      TMP0 = TMP0+2*RO**2*(252*T2**6+(-18*T1**2+414*T1-486)*T2**5+(-18*T
     1   1**3+271*T1**2-468*T1+338)*T2**4+(-18*T1**4+132*T1**3-241*T1**2
     2   +208*T1-136)*T2**3+(-18*T1**5+73*T1**4-223*T1**3+156*T1**2-64*T
     3   1+32)*T2**2+(-72*T1**4+208*T1**3-64*T1**2)*T2+104*T1**4-136*T1*
     4   *3+32*T1**2)
      TMP0 = RLG12*(T2+T1-1)*TMP0/( 128.D0*T1**4*T2**4)
      PP = TMP0+PP
      PP = 9*ITX*RLG11*(-2*T1*((9*T1-2)*T2**2+(10*T1**2-7*T1+2)*T2+17*T1
     1   **3-21*T1**2+14*T1-4)+RO*((T1+2)*T2**2+(-4*T1**2+3*T1-4)*T2-5*T
     2   1**3-7*T1**2+4*T1)+2*RO**2*(T2+T1-2))/( 128.D0*T1**2)+PP
      PP = PP-RLG11*(-2*T1*(36*T2**3+(45*T1-20)*T2**2+(81*T1**2-43*T1+20
     1   )*T2-4*T1**2+8*T1-4)+RO*(-(63*T1-22)*T2**2-(45*T1**2-17*T1+38)*
     2   T2+4*T1)+2*RO**2*(18*T2**2+(18*T1-8)*T2+T1-2))/( 128.D0*T1**2*T
     3   2)
      TMP0 = RO*((T1+1)*T2**2+(2*T1-2)*T2-T1**3+3*T1**2)+2*T1*((T1+1)*T2
     1   +3*T1**2+T1+2)+RO**2*((T1+1)*T2-T1**2-T1-2)
      TMP0 = 9*ITX*RL3524*(T2+T1-1)*TMP0/( 128.D0*T1)
      PP = TMP0+PP
      TMP0 = -2*RO*T1*(2*T2**3+(2*T1-4)*T2**2+(-6*T1-1)*T2+3*T1+3)-8*T1*
     1   *2*(2*T2**2+(2*T1-2)*T2-T1)+RO**2*(T2-1)*(3*T2-T1-3)
      TMP0 = -D2435**2*RL3524*RO**3*(T2+T1-1)*TMP0/ 144.D0
      PP = TMP0+PP
      TMP0 = D2435*RL3524*RO*(T2+T1-1)*(32*T1**2*(T1+1)*TX-16*RO*T1*((5*
     1   T1-1)*T2+T1**2-3*T1+1)-2*RO**2*T1*((T1+1)*T2-8*T1)+RO**3*((T1-1
     2   )*T2+1))/( 288.D0*T1)
      PP = TMP0+PP
      PP = RL3524*(RO*(-(65*T1+65)*T2**3-(-14*T1**2-421*T1-209)*T2**2-(-
     1   79*T1**3-362*T1**2+257*T1+144)*T2+2*T1**3+16*T1**2-18*T1)-2*RO*
     2   *2*(64*T2**3+(93*T1-99)*T2**2+(21*T1**2-72*T1+27)*T2+T1**3+10*T
     3   1**2+10*T1+9)-2*T1*((63*T1+63)*T2**2+(79*T1**2-47)*T2-2*T1**3-1
     4   8*T1**2-2*T1-18)+RO**3*((8*T1+1)*T2**2+(8*T1**2-6*T1)*T2+T1**2+
     5   9*T1))/( 1152.D0*T1*T2)+PP
      TMP0 = RO*(T2**3-3*T2**2+(-T1**2-2*T1)*T2-T1**2+2*T1)-2*T2*(3*T2**
     1   2+(T1+1)*T2+T1+2)+RO**2*(T2**2+(1-T1)*T2-T1+2)
      TMP0 = -9*ITX*RL3514*(T2+T1-1)*TMP0/( 128.D0*T2)
      PP = TMP0+PP
      TMP0 = 2*RO*T2*((2*T1**2-6*T1+3)*T2+2*T1**3-4*T1**2-T1+3)+8*T2**2*
     1   ((2*T1-1)*T2+2*T1**2-2*T1)+RO**2*(T1-1)*(T2-3*T1+3)
      TMP0 = D1435**2*RL3514*RO**3*(T2+T1-1)*TMP0/ 144.D0
      PP = TMP0+PP
      TMP0 = D1435*RL3514*RO*(T2+T1-1)*(32*T2**2*(T2+1)*TX-16*RO*T2*(T2*
     1   *2+(5*T1-3)*T2-T1+1)+RO**3*(T1*T2-T1+1)-2*RO**2*T2*((T1-8)*T2+T
     2   1))/( 288.D0*T2)
      PP = TMP0+PP
      PP = RL3514*(RO*((79*T1+2)*T2**3+(14*T1**2+362*T1+16)*T2**2+(-65*T
     1   1**3+421*T1**2-257*T1-18)*T2-65*T1**3+209*T1**2-144*T1)+2*T2*(2
     2   *T2**3+(18-79*T1)*T2**2+(2-63*T1**2)*T2-63*T1**2+47*T1+18)-2*RO
     3   **2*(T2**3+(21*T1+10)*T2**2+(93*T1**2-72*T1+10)*T2+64*T1**3-99*
     4   T1**2+27*T1+9)+RO**3*((8*T1+1)*T2**2+(8*T1**2-6*T1+9)*T2+T1**2)
     5   )/( 1152.D0*T1*T2)+PP
      TMP0 = -4*RO*(T2**2+3*T1*T2+T1**2)+4*(T2+T1)**2*(T2**2+T1**2)+3*RO
     1   **2
      TMP0 = -9*DLAM2**2*ITX*RL35*(T2+T1-1)**3*(T2+T1+RO)*TMP0/ 64.D0
      PP = TMP0+PP
      TMP0 = -4*(T2+T1)*(3*T2**3+(5*T1-2)*T2**2+5*T1**2*T2+3*T1**3-2*T1*
     1   *2)-RO*(T2+T1)*(12*T2**2+(8*T1-13)*T2+12*T1**2-13*T1+4)+RO**2*(
     2   8*T2+8*T1-5)
      TMP0 = -9*DLAM2*ITX*RL35*(T2+T1-1)**2*TMP0/ 128.D0
      PP = TMP0+PP
      TMP0 = -2*(2*T2**3+(6*T1-11)*T2**2+(6*T1**2-6*T1+12)*T2+2*T1**3-11
     1   *T1**2+12*T1)+RO*(-4*T2**2-(8*T1-1)*T2-4*T1**2+T1-1)+4*RO**2
      TMP0 = 9*ITX*RL35*(T2+T1-1)*TMP0/ 128.D0
      PP = TMP0+PP
      TMP0 = 4*RO*T1*T2*(32*T2**4+(64*T1-68)*T2**3+(64*T1**2-48*T1+37)*T
     1   2**2+(64*T1**3-48*T1**2+106*T1)*T2+32*T1**4-68*T1**3+37*T1**2)
      TMP0 = TMP0+RO**2*(-128*T1*T2**3-(384*T1**2-44*T1+3)*T2**2-(128*T1
     1   **3-44*T1**2+102*T1)*T2-3*T1**2)+8*T1*T2*(T2+T1)**2*(18*T2**3+(
     2   18*T1-17)*T2**2+(18*T1**2-2*T1)*T2+18*T1**3-17*T1**2)+96*RO**3*
     3   T1*T2
      TMP0 = DLAM2**2*RL35*(T2+T1-1)**2*TMP0/( 256.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = -4*RO*(207*T1*T2**4+(369*T1**2-387*T1)*T2**3+(369*T1**3-198
     1   *T1**2+207*T1+5)*T2**2+(207*T1**4-387*T1**3+207*T1**2+T1)*T2+5*
     2   T1**2)-4*T1*T2*(T2+T1)*(243*T2**3+(405*T1-243)*T2**2+(405*T1**2
     3   -234*T1-11)*T2+243*T1**3-243*T1**2-11*T1)+9*RO**2*((60*T1-2)*T2
     4   **2+(60*T1**2-32*T1+3)*T2-2*T1**2+3*T1)+9*RO**3*(T2+T1)
      TMP0 = DLAM2*RL35*(T2+T1-1)*TMP0/( 1152.D0*T1*T2)
      PP = TMP0+PP
      PP = RL35*((162*T1-18)*T2**4+RO*((126*T1-18)*T2**3+(324*T1**2-594*
     1   T1+16)*T2**2+(126*T1**3-594*T1**2+468*T1+3)*T2-18*T1**3+16*T1**
     2   2+3*T1)+(486*T1**2-567*T1+18)*T2**3+RO**2*(9*T2**2+(10-36*T1)*T
     3   2+9*T1**2+10*T1)+(486*T1**3-1242*T1**2+725*T1-38)*T2**2+(162*T1
     4   **4-567*T1**3+725*T1**2-292*T1)*T2-18*T1**4+18*T1**3-38*T1**2)/
     5   ( 576.D0*T1*T2)+PP
      TMP0 = 2*T2*(9*T2**2+(13*T1-8)*T2+8*T1**2-8*T1+4)*TX+RO*(5*T2**3+(
     1   3*T1-2)*T2**2+(-2*T1**2+2*T1-4)*T2-8*T1)+4*RO**2*(T2**2-T2+1)
      TMP0 = -9*ITX*RL3424*(T2+T1-1)**2*TMP0/( 256.D0*T2)
      PP = TMP0+PP
      TMP0 = 2*T2*((17*T1-6)*T2**2+(24*T1**2-34*T1+4)*T2+16*T1**3-32*T1*
     1   *2+16*T1)*TX+RO*((5*T1-2)*T2**3+(6*T1**2+2*T1-6)*T2**2+(12*T1**
     2   2-22*T1)*T2-32*T1**2)+2*RO**2*(T2-2)*T2
      TMP0 = 9*RL3424*(T2+T1-1)**2*TMP0/( 256.D0*T1*T2**2)
      PP = TMP0+PP
      TMP0 = 2*T1*(8*T2**2+(13*T1-8)*T2+9*T1**2-8*T1+4)*TX+RO*(-2*T1*T2*
     1   *2-(-3*T1**2-2*T1+8)*T2+5*T1**3-2*T1**2-4*T1)+4*RO**2*(T1**2-T1
     2   +1)
      TMP0 = -9*ITX*RL3414*(T2+T1-1)**2*TMP0/( 256.D0*T1)
      PP = TMP0+PP
      TMP0 = 2*T1*(16*T2**3+(24*T1-32)*T2**2+(17*T1**2-34*T1+16)*T2-6*T1
     1   **2+4*T1)*TX+RO*((6*T1**2+12*T1-32)*T2**2+(5*T1**3+2*T1**2-22*T
     2   1)*T2-2*T1**3-6*T1**2)+2*RO**2*(T1-2)*T1
      TMP0 = 9*RL3414*(T2+T1-1)**2*TMP0/( 256.D0*T1**2*T2)
      PP = TMP0+PP
      TMP0 = -4*RO*(T2**2+3*T1*T2+T1**2)+4*(T2+T1)**2*(T2**2+T1**2)+3*RO
     1   **2
      TMP0 = -9*DLAM2**2*ITX*RL34*(-T2-T1+RO)*(T2+T1-1)**2*TMP0/ 64.D0
      PP = TMP0+PP
      TMP0 = 4*(T2+T1)*(T2**3+(2-T1)*T2**2-T1**2*T2+T1**3+2*T1**2)-RO*(T
     1   2+T1)*(4*T2**2+(5-8*T1)*T2+4*T1**2+5*T1+4)+5*RO**2
      TMP0 = -9*DLAM2*ITX*RL34*(T2+T1-1)**2*TMP0/ 128.D0
      PP = TMP0+PP
      TMP0 = -2*(2*T2**3+(14*T1+3)*T2**2+(14*T1**2+22*T1-12)*T2+2*T1**3+
     1   3*T1**2-12*T1)+RO*(2*T2+2*T1-1)+4*RO**2
      TMP0 = 9*ITX*RL34*(T2+T1-1)**2*TMP0/ 128.D0
      PP = TMP0+PP
      TMP0 = 4*RO*((32*T1**2+8*T1)*T2**5+(96*T1**3-48*T1**2-4*T1+2)*T2**
     1   4+(96*T1**4-112*T1**3+40*T1**2-3*T1)*T2**3+(32*T1**5-48*T1**4+4
     2   0*T1**3-18*T1**2)*T2**2+(8*T1**5-4*T1**4-3*T1**3)*T2+2*T1**4)
      TMP0 = -24*RO**3*T1*T2*TX+TMP0-8*(T2+T1)**2*((2*T1+1)*T2**4+(2*T1*
     1   *2-3*T1)*T2**3+2*T1**3*T2**2+(2*T1**4-3*T1**3)*T2+T1**4)+RO**2*
     2   (-48*T1*T2**4-(176*T1**2-64*T1)*T2**3-(176*T1**3-160*T1**2+28*T
     3   1+3)*T2**2-(48*T1**4-64*T1**3+28*T1**2-18*T1)*T2-3*T1**2)
      TMP0 = 9*DLAM2**2*RL34*(T2+T1-1)*TMP0/( 256.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = RO*((26*T1+8)*T2**4+(110*T1**2-49*T1+8)*T2**3+(110*T1**3-11
     1   4*T1**2+57*T1-4)*T2**2+(26*T1**4-49*T1**3+57*T1**2-52*T1)*T2+8*
     2   T1**4+8*T1**3-4*T1**2)
      TMP0 = TMP0+(T2+T1)*((17*T1-24)*T2**4+(83*T1**2-41*T1+8)*T2**3+(83
     1   *T1**3-162*T1**2+52*T1)*T2**2+(17*T1**4-41*T1**3+52*T1**2)*T2-2
     2   4*T1**4+8*T1**3)-2*RO**2*(2*T2**3+(19*T1+2)*T2**2+(19*T1**2-16*
     3   T1-1)*T2+2*T1**3+2*T1**2-T1)+2*RO**3*(T2+T1)
      TMP0 = -9*DLAM2*RL34*(T2+T1-1)*TMP0/( 256.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = (8*T1-32)*T2**4+2*RO*(4*T2**3+(4-21*T1)*T2**2+(-21*T1**2+42
     1   *T1-8)*T2+4*T1**3+4*T1**2-8*T1)+(56*T1**2+101*T1+32)*T2**3+(56*
     2   T1**3+106*T1**2-197*T1-16)*T2**2+(8*T1**4+101*T1**3-197*T1**2+1
     3   32*T1)*T2-32*T1**4+32*T1**3-16*T1**2
      TMP0 = 9*RL34*(T2+T1-1)*TMP0/( 256.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = 4*(T2**2+(2*T1-20)*T2+T1**2-20*T1+20)*TX-2*RO*((T1+10)*T2**
     1   2+(T1**2-20*T1-1)*T2+10*T1**2-T1)+RO**2*(-2*T2**2-(-14*T1-33)*T
     2   2-2*T1**2+33*T1-60)+RO**3*((8*T1+1)*T2+T1-10)
      TMP0 = -RL1424*(T2+T1-1)**2*TMP0/( 1152.D0*T1*T2)
      PP = TMP0+PP
      TMP0 = 3*T1*T2**3+(6*T1**2+14*T1+13)*T2**2+(3*T1**3+14*T1**2-4*T1)
     1   *T2+13*T1**2
      TMP0 = 32*(T2-T1)**2*(T2+T1)*TMP0*TX
      TMP0 = TMP0-8*RO*(72*T1*T2**7+(288*T1**2-287*T1)*T2**6+(504*T1**3-
     1   287*T1**2+464*T1-47)*T2**5+(576*T1**4+574*T1**3-720*T1**2-415*T
     2   1+16)*T2**4+(504*T1**5+574*T1**4-2368*T1**3+1326*T1**2+128*T1+2
     3   2)*T2**3+(288*T1**6-287*T1**5-720*T1**4+1326*T1**3-576*T1**2-22
     4   *T1)*T2**2+(72*T1**7-287*T1**6+464*T1**5-415*T1**4+128*T1**3-22
     5   *T1**2)*T2-47*T1**5+16*T1**4+22*T1**3)
      TMP0 = TMP0+4*RO**2*(36*T1*T2**6+(108*T1**2+36*T1)*T2**5+(144*T1**
     1   3+792*T1**2-T1-18)*T2**4+(144*T1**4+1512*T1**3-1151*T1**2-368*T
     2   1-40)*T2**3+(108*T1**5+792*T1**4-1151*T1**3-92*T1**2+472*T1+40)
     3   *T2**2+(36*T1**6+36*T1**5-T1**4-368*T1**3+472*T1**2-224*T1)*T2-
     4   18*T1**4-40*T1**3+40*T1**2)
      TMP0 = -108*RO**4*T1*T2*TX+TMP0-3*RO**3*(48*T1*T2**4+(192*T1**2+14
     1   4*T1-1)*T2**3+(192*T1**3+240*T1**2-383*T1-8)*T2**2+(48*T1**4+14
     2   4*T1**3-383*T1**2+208*T1)*T2-T1**3-8*T1**2)
      TMP0 = -DLAM2**2*DRO*(T2+T1-1)*TMP0/( 64.D0*T1*T2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = RO*(8*T2**3+(-T1-8)*T2**2+(2*T1-T1**2)*T2+8*T1**3-8*T1**2)-
     1   6*T1*T2*(6*T2**3+(12*T1-13)*T2**2+(12*T1**2-32*T1+12)*T2+6*T1**
     2   3-13*T1**2+12*T1-2)
      TMP0 = DRO*(T2+T1-1)**2*TMP0/( 9.D0*T1**2*T2**2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = 8*(T2+T1)**2*(T2**2+T1**2)*TX+4*RO*(T2**4+(2*T1+1)*T2**3+(2
     1   *T1**2+11*T1-1)*T2**2+(2*T1**3+11*T1**2-10*T1)*T2+T1**4+T1**3-T
     2   1**2)-4*RO**2*(T2**2+(3*T1+2)*T2+T1**2+2*T1-2)+3*RO**3
      TMP0 = 13*DLAM2**2*(T2+T1-1)**2*TMP0/( 8.D0*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = RO*(3762*T1*T2**4+(6678*T1**2+3663*T1+1548)*T2**3+(6678*T1*
     1   *3+11934*T1**2-11187*T1-500)*T2**2+(3762*T1**4+3663*T1**3-11187
     2   *T1**2+3356*T1)*T2+1548*T1**3-500*T1**2)
      TMP0 = TMP0-8415*T1*T2**5-(24444*T1**2-9711*T1+1944)*T2**4-(32058*
     1   T1**3-27117*T1**2-1244*T1-568)*T2**3-18*RO**2*((177*T1+32)*T2**
     2   2+(177*T1**2-136*T1-6)*T2+32*T1**2-6*T1)-(24444*T1**4-27117*T1*
     3   *3+8024*T1**2+568*T1)*T2**2+144*RO**3*(T2+T1)-(8415*T1**5-9711*
     4   T1**4-1244*T1**3+568*T1**2)*T2-1944*T1**4+568*T1**3
      TMP0 = -DLAM2*(T2+T1-1)*TMP0/( 576.D0*T1*T2*(4*TX+RO))
      PP = TMP0+PP
      PP = D2435**2*RO**2*(-4*RO*T1*(T2**3+(2*T1-2)*T2**2+T1**2*T2+2*T1*
     1   *2-T1+1)+RO**2*(3*T2**2+(2*T1-6)*T2+3*T1**2-2*T1+3)+4*T1**2*(2*
     2   T2**2+(4*T1-2)*T2+2*T1**2-2*T1+1))/ 36.D0+PP
      PP = PP-D2435*(RO**2+16*RO+16)*(2*T1*(T2+T1)+RO*(T2-T1-1))/ 72.D0
      PP = D1435**2*RO**2*(-4*RO*T2*((T1+2)*T2**2+(2*T1**2-1)*T2+T1**3-2
     1   *T1**2+1)+RO**2*(3*T2**2+(2*T1-2)*T2+3*T1**2-6*T1+3)+4*T2**2*(2
     2   *T2**2+(4*T1-2)*T2+2*T1**2-2*T1+1))/ 36.D0+PP
      PP = D1435*(RO**2+16*RO+16)*(RO*(T2-T1+1)-2*T2*(T2+T1))/ 72.D0+PP
      TMP0 = 2*T1**2*T2**2*(18144*T2**6+(49248*T1-44064)*T2**5+(103311*T
     1   1**2-76932*T1+42912)*T2**4+(116478*T1**3-124803*T1**2+46060*T1-
     2   26208)*T2**3+(103311*T1**4-124803*T1**3+56836*T1**2-17288*T1+92
     3   16)*T2**2+(49248*T1**5-76932*T1**4+46060*T1**3-17288*T1**2-432*
     4   T1)*T2+18144*T1**6-44064*T1**5+42912*T1**4-26208*T1**3+9216*T1*
     5   *2)*TX
      TMP0 = TMP0+RO**2*(-(23328*T1+18144)*T2**7-(48924*T1**2+5184*T1-51
     1   840)*T2**6-(39456*T1**3-6157*T1**2-62208*T1+54720)*T2**5-(19584
     2   *T1**4+743*T1**3-24722*T1**2+39744*T1-34560)*T2**4-(39456*T1**5
     3   +743*T1**4+1856*T1**3+18810*T1**2-30240*T1+21600)*T2**3-(48924*
     4   T1**6-6157*T1**5-24722*T1**4+18810*T1**3-52992*T1**2+24192*T1-8
     5   064)*T2**2-(23328*T1**7+5184*T1**6-62208*T1**5+39744*T1**4-3024
     6   0*T1**3+24192*T1**2)*T2-18144*T1**7+51840*T1**6-54720*T1**5+345
     7   60*T1**4-21600*T1**3+8064*T1**2)
      TMP0 = TMP0+2*RO*T1*T2*((4536*T1+46656)*T2**7+(12312*T1**2+133488*
     1   T1-145152)*T2**6+(26910*T1**3+156591*T1**2-305690*T1+176256)*T2
     2   **5+(30132*T1**4+83799*T1**3-244768*T1**2+254782*T1-124416)*T2*
     3   *4+(26910*T1**5+83799*T1**4-138222*T1**3+175977*T1**2-149144*T1
     4   +67392)*T2**3+(12312*T1**6+156591*T1**5-244768*T1**4+175977*T1*
     5   *3-141696*T1**2+62028*T1-20736)*T2**2+(4536*T1**7+133488*T1**6-
     6   305690*T1**5+254782*T1**4-149144*T1**3+62028*T1**2)*T2+46656*T1
     7   **7-145152*T1**6+176256*T1**5-124416*T1**4+67392*T1**3-20736*T1
     8   **2)
      TMP0 = TMP0+2*RO**3*(2268*T2**6+(4536*T1-4212)*T2**5+(2287*T1**2-3
     1   888*T1+2628)*T2**4+(1514*T1**3-972*T1**2+1368*T1-1692)*T2**3+(2
     2   287*T1**4-972*T1**3+2592*T1**2-2016*T1+1008)*T2**2+(4536*T1**5-
     3   3888*T1**4+1368*T1**3-2016*T1**2)*T2+2268*T1**6-4212*T1**5+2628
     4   *T1**4-1692*T1**3+1008*T1**2)
      TMP0 = TMP0/( 1152.D0*T1**4*T2**4*(4*TX+RO))
      PP = TMP0+PP
      HQHPGG = PP
      RETURN
      ENDIF
      END
      FUNCTION HQHPQA(TX,T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(TX.EQ.0)THEN
      T2 = 1-T1
      B = DSQRT(1-RO)
      VLSM = DLOG(4/RO)
      SRL12 = DLOG(T1/T2)
      SRLGRO = DLOG(4*T1*T2/RO)
      SRLG1 = DLOG((B+1)/(1-B))/B
      PP = 64*(2*(2*T1**2-2*T1+1)+RO)*VLSM/ 27.D0
      PP = 4*SRLGRO*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0+PP
      PP = 2*(RO-2)*SRLG1*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0+PP
      PP = PP
      PP = PP-32*(2*(2*T1**2-2*T1+1)+RO)/ 27.D0
      HQHPQA = PP
      RETURN
      ELSE
      T2 = -TX-T1+1
      T11 = 1/(1-T1)
      T22 = 1/(TX+T1)
      DRO = 1/(4*TX+RO)
      B = DSQRT((1-TX)**2-RO)
      DLAM2 = 1/(TX**2-2*TX-RO+1)
      VLSM = DLOG(4/RO)
      RLGXRO = DLOG((4*TX+RO)/RO)/TX
      RLG12 = DLOG(T1/(1-T2))/TX
      RLG21 = DLOG(T2/(1-T1))/TX
      RLG11 = DLOG(T1/(1-T1))
      RLG22 = DLOG(T2/(1-T2))
      RLGRO = DLOG(4*(1-T1)*(1-T2)/RO)
      RL34 = DLOG((2*TX**2+2*B*TX-2*TX-RO)/(2*TX**2-2*B*TX-2*TX-RO))/(B*
     1   TX)
      RL35 = DLOG((2*TX+RO-2*B-2)/(2*TX+RO+2*B-2))/B
      TMP0 = RO/(T1*T22)+4*T2**2-4*T2+2
      TMP0 = 16*((-T2-T1+1)**2+2*T1*(-T2-T1+1)+2*T1**2)*T22*TMP0/( 27.D0
     1   *T1*(-T2-T1+1))
      PP = RO/(T11*T2)+4*T1**2-4*T1+2
      PP = 16*PP*T11*((-T2-T1+1)**2+2*T1*(-T2-T1+1)-2*(-T2-T1+1)+2*T1**2
     1   -4*T1+2)/( 27.D0*(-T2-T1+1)*T2)
      PP = TMP0+PP
      PP = PP*(-T2-T1+1)*VLSM
      PP = PP-RLGXRO*(T2+T1-1)*(9*RO*(T2-1)-2*(9*T2**2+(27*T1-11)*T2-11*
     1   T1+2))*T22/ 54.D0
      PP = T22*(4*((63*T1-32)*T2**2+(45*T1**2-89*T1+64)*T2-49*T1**2+26*T
     1   1-32)*TX+RO*((63*T1-32)*T2**2+(63*T1**2-107*T1+64)*T2-67*T1**2+
     2   44*T1-32))/( 54.D0*T1*(4*TX+RO))+PP
      PP = PP-RLGXRO*T11*(T2+T1-1)*(9*RO*(T1-1)-2*((27*T1-11)*T2+9*T1**2
     1   -11*T1+2))/ 54.D0
      PP = T11*(4*((45*T1-49)*T2**2+(63*T1**2-89*T1+26)*T2-32*T1**2+64*T
     1   1-32)*TX+RO*((63*T1-67)*T2**2+(63*T1**2-107*T1+44)*T2-32*T1**2+
     2   64*T1-32))/( 54.D0*T2*(4*TX+RO))+PP
      PP = 4*RLGRO*(2*(T2**2-T2+T1**2-T1+1)+RO)/ 27.D0+PP
      PP = RLG22*(2*(3*T2**2-2*T2+T1**2+1)+RO*(T2+T1+1))/( 3.D0*(T2+T1))
     1   +PP
      TMP0 = RO*(2*T2**3+(2*T1-9)*T2**2+(16*T1**2-32*T1+16)*T2+16*T1**3-
     1   32*T1**2+16*T1)+2*T2*(2*T2**2+2*T1**2-2*T1+1)*(2*T2**2+(7-14*T1
     2   )*T2-16*T1**2+16*T1)
      TMP0 = -RLG21*(T2+T1-1)*TMP0/( 27.D0*T2**2*(T2+T1))
      PP = TMP0+PP
      TMP0 = RO*(16*T2**3+(16*T1-32)*T2**2+(2*T1**2-32*T1+16)*T2+2*T1**3
     1   -9*T1**2+16*T1)-2*T1*(2*T2**2-2*T2+2*T1**2+1)*(16*T2**2+(14*T1-
     2   16)*T2-2*T1**2-7*T1)
      TMP0 = -RLG12*(T2+T1-1)*TMP0/( 27.D0*T1**2*(T2+T1))
      PP = TMP0+PP
      PP = RLG11*(2*(T2**2+3*T1**2-2*T1+1)+RO*(T2+T1+1))/( 3.D0*(T2+T1))
     1   +PP
      TMP0 = RO**2*(-(12*T1-48)*T2**2-(20*T1**2-140*T1+3)*T2-8*T1**3+56*
     1   T1**2-3*T1)+4*RO*T1*(T2+T1)*((2*T1-32)*T2**2+(4*T1**2-34*T1+3)*
     2   T2+2*T1**3-2*T1**2+2*T1)-8*T1**2*(T2+T1)**3+3*RO**3*(T2+T1-9)
      TMP0 = DLAM2**2*RL35*(T2+T1-1)**2*TMP0/( 54.D0*(T2+T1))
      PP = TMP0+PP
      TMP0 = RO*(-(8*T1**2-54*T1+7)*T2**2-(16*T1**3-80*T1**2+40*T1+7)*T2
     1   -8*T1**4+26*T1**3-33*T1**2+11*T1)-2*T1*(T2+T1)*(9*T2**2+(-4*T1-
     2   7)*T2-13*T1**2+11*T1)+RO**2*((4*T1-11)*T2+4*T1**2-29*T1+18)
      TMP0 = DLAM2*RL35*(T2+T1-1)*TMP0/( 27.D0*(T2+T1))
      PP = TMP0+PP
      PP = RL35*(-2*(10*T2**3+(T1-9)*T2**2+(12*T1**2-2*T1+1)*T2+21*T1**3
     1   -29*T1**2+10*T1)+RO*(7*T2**2+(4*T1**2-8*T1+2)*T2+4*T1**3-15*T1*
     2   *2+20*T1-9)+RO**2*(T2+T1))/( 27.D0*(T2+T1))+PP
      TMP0 = RO**2*(-12*T2**4-(56*T1-16)*T2**3-(88*T1**2-68*T1+4)*T2**2-
     1   (56*T1**3-76*T1**2+20*T1-3)*T2-12*T1**4+24*T1**3-12*T1**2+3*T1)
      TMP0 = -3*RO**3*(2*T2+2*T1-1)*TX+TMP0+4*RO*T2*(T2+T1)*((8*T1+2)*T2
     1   **3+(24*T1**2-12*T1-2)*T2**2+(24*T1**3-30*T1**2+6*T1-2)*T2+8*T1
     2   **4-16*T1**3+8*T1**2-3*T1)+8*T2**2*(T2+T1)**3
      TMP0 = -DLAM2**2*RL34*(T2+T1-1)*TMP0/( 6.D0*(T2+T1))
      PP = TMP0+PP
      TMP0 = RO*(-(12*T1-2)*T2**3-(28*T1**2-16*T1-4)*T2**2-(20*T1**3-22*
     1   T1**2+4*T1-1)*T2-4*T1**4+8*T1**3-8*T1**2+3*T1)-2*T2*(T2+T1)*(4*
     2   T2**2+T2-4*T1**2+3*T1)+RO**2*(T2+3*T1-2)*(2*T2+2*T1-1)
      TMP0 = -DLAM2*RL34*(T2+T1-1)*TMP0/( 3.D0*(T2+T1))
      PP = TMP0+PP
      PP = RL34*(T2+T1-1)*(9*RO*(2*T2**2-T2-2*T1**2+T1-1)-2*(36*T2**3-7*
     1   T2**2+(16-14*T1)*T2+36*T1**3-43*T1**2+25*T1))/( 27.D0*(T2+T1))+
     2   PP
      TMP0 = 32*(T2+T1)*((3*T1**2-20*T1-13)*T2**2+(6*T1**3-26*T1**2+20*T
     1   1)*T2+3*T1**4-6*T1**3+3*T1**2)*TX
      TMP0 = TMP0-8*RO*((36*T1**2+144*T1+36)*T2**4+(144*T1**3+289*T1**2-
     1   342*T1-25)*T2**3+(216*T1**4+3*T1**3-598*T1**2+473*T1+20)*T2**2+
     2   (144*T1**5-285*T1**4-26*T1**3+437*T1**2-248*T1-22)*T2+36*T1**6-
     3   143*T1**5+194*T1**4-61*T1**3-68*T1**2+42*T1)
      TMP0 = -27*RO**4*TX+TMP0+4*RO**2*((18*T1**2+180*T1+126)*T2**3+(54*
     1   T1**3+378*T1**2-159*T1-252)*T2**2+(54*T1**4+216*T1**3-409*T1**2
     2   -12*T1+220)*T2+18*T1**5+18*T1**4-124*T1**3+88*T1**2+76*T1-76)-3
     3   *RO**3*((36*T1+72)*T2**2+(60*T1**2+60*T1-119)*T2+24*T1**3-71*T1
     4   +56)
      TMP0 = 2*DLAM2**2*DRO*(T2+T1-1)*TMP0/( 27.D0*(4*TX+RO))
      PP = TMP0+PP
      PP = PP-128*DRO*(T2-T1)*(T2+T1-1)**2/( 27.D0*(4*TX+RO))
      TMP0 = 16*(T2+T1)**2*(9*T2**2+4*T1**2)*TX+4*RO*(9*T2**4+(18*T1+36)
     1   *T2**3+(26*T1**2+166*T1-23)*T2**2+(34*T1**3+120*T1**2-130*T1)*T
     2   2+17*T1**4-10*T1**3-3*T1**2)-4*RO**2*(9*T2**2+(39*T1+39)*T2+17*
     3   T1**2+13*T1-26)+39*RO**3
      TMP0 = -4*DLAM2**2*(T2+T1-1)**2*TMP0/( 27.D0*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = 2*(72*T2**4+(216*T1-144)*T2**3+(280*T1**2-347*T1+45)*T2**2+
     1   (200*T1**3-276*T1**2+138*T1-16)*T2+64*T1**4-73*T1**3-7*T1**2+16
     2   *T1)+RO*(-36*T2**3-(72*T1+36)*T2**2-(100*T1**2+172*T1-219)*T2-6
     3   4*T1**3-72*T1**2+201*T1-88)+2*RO**2*(9*T2+25*T1-17)
      TMP0 = -4*DLAM2*(T2+T1-1)*TMP0/( 27.D0*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = 8*T1*T2*(36*T1*T2**3+(72*T1**2-125*T1+16)*T2**2+(68*T1**3-1
     1   32*T1**2+96*T1-16)*T2+16*T1**2-16*T1)*TX
      TMP0 = TMP0+RO*(-32*T2**5-(-72*T1**2+96*T1-96)*T2**4-(-144*T1**3+4
     1   63*T1**2-224*T1+96)*T2**3-(-200*T1**4+591*T1**3-528*T1**2+128*T
     2   1-32)*T2**2-(96*T1**4-224*T1**3+128*T1**2)*T2-32*T1**5+96*T1**4
     3   -96*T1**3+32*T1**2)+8*RO**2*(T2**4+(2*T1-2)*T2**3+(6*T1**2-2*T1
     4   +1)*T2**2+(2*T1**3-2*T1**2)*T2+T1**4-2*T1**3+T1**2)
      TMP0 = -TMP0/( 27.D0*T1**2*T2**2*(4*TX+RO))
      PP = TMP0+PP
      HQHPQA = PP
      RETURN
      ENDIF
      END
      FUNCTION HQHPQG(TX,T1,RO)
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(TX.EQ.0)THEN
      PP = 0
      HQHPQG = PP
      RETURN
      ELSE
      T2 = -TX-T1+1
      T11 = 1/(1-T1)
      T22 = 1/(TX+T1)
      B = DSQRT((1-TX)**2-RO)
      DLAM2 = 1/(TX**2-2*TX-RO+1)
      VLSM = DLOG(4/RO)
      RLGXRO = DLOG((4*TX+RO)/RO)/TX
      RLG12 = DLOG(T1/(1-T2))/TX
      RLG21 = DLOG(T2/(1-T1))/TX
      RLGRO = DLOG(4*(1-T1)*(1-T2)/RO)
      RR = T2*(T2*TX**2+RO*(1-T2))
      RR = DSQRT(RR)
      RL3424 = 2*DLOG((T2*TX+RR)/(RR-T2*TX))/(RR*TX)
      RL34 = DLOG((2*TX**2+2*B*TX-2*TX-RO)/(2*TX**2-2*B*TX-2*TX-RO))/(B*
     1   TX)
      PP = T11*T2-2*(T1-1)/T2-2
      PP = -PP*(3*T1**2-3*T1+4/ 3.D0)*T11**2*(4*RO*(1-T1)*T1**2/T2-4*RO*
     1   (1-T1)*T1/T2+RO**2*(1-T1)**2/T2**2+8*T1**4-16*T1**3+12*T1**2-4*
     2   T1)/( 12.D0*T1**2*T2)
      PP = 2*(4*T2**2-4*T2+RO*(1-T2)/T1+2)*(2*T1**2*T22**2-2*T1*T22+1)/(
     1    9.D0*T1)+PP
      PP = PP*(-T2-T1+1)*VLSM
      TMP0 = RO*(9*T2-2)-2*(T2-1)*((9*T1+2)*T2+9*T1-9)
      TMP0 = -RLGXRO*(T2+T1-1)**2*T22**2*TMP0/( 72.D0*T2)
      PP = TMP0+PP
      PP = (T2+T1-1)*T22**2*(2*((7*T1+23)*T2**2+(46*T1-16)*T2+7*T1-7)*TX
     1   +RO*(15*T2**2+(23*T1-8)*T2+7*T1-7))/( 18.D0*T2*(4*TX+RO))+PP
      TMP0 = RO*(7*T2+9)-2*(7*T2**2+(23*T1+2)*T2+23*T1-16)
      TMP0 = -RLGXRO*(T2+T1-1)**2*T22*TMP0/( 72.D0*T2)
      PP = TMP0+PP
      PP = (T2+T1-1)*T22*(-8*RO**2*TX-RO*T1*((2*T1+16)*T2**2+(7*T1+16)*T
     1   2+32*T1-32)+2*T1**2*T2*(2*T2**2+(4*T1+5)*T2+14*T1-23))/( 72.D0*
     2   T1**2*T2**2)+PP
      PP = PP-RLGXRO*(9*T1+7)*T11**2*(T2+T1-1)**3/( 72.D0*T2)
      TMP0 = (9*T1**3-5*T1+4)*T2**3+(18*T1**4-T1**3-36*T1**2+31*T1-12)*T
     1   2**2+(18*T1**5-18*T1**4-46*T1**3+90*T1**2-60*T1+16)*T2+18*T1**5
     2   -72*T1**4+116*T1**3-96*T1**2+42*T1-8
      TMP0 = 16*RO**2*(T1-1)*TMP0*TX
      TMP0 = TMP0-4*T1**2*T2**2*((144*T1**3-306*T1**2+185*T1-55)*T2**2+(
     1   288*T1**4-864*T1**3+983*T1**2-526*T1+119)*T2+288*T1**5-1152*T1*
     2   *4+1856*T1**3-1536*T1**2+672*T1-128)*TX**2
      TMP0 = TMP0-RO*T1*T2*((144*T1**4+288*T1**3-971*T1**2+763*T1-256)*T
     1   2**3+(288*T1**5+864*T1**4-3963*T1**3+4918*T1**2-2875*T1+768)*T2
     2   **2+(288*T1**6+1152*T1**5-7104*T1**4+12544*T1**3-10848*T1**2+49
     3   92*T1-1024)*T2+1152*T1**6-5760*T1**5+12032*T1**4-13568*T1**3+88
     4   32*T1**2-3200*T1+512)*TX
      TMP0 = TMP0+4*RO**3*(T1-1)*((9*T1**2-9*T1+4)*T2**3+(31*T1**3-58*T1
     1   **2+39*T1-12)*T2**2+(36*T1**4-108*T1**3+124*T1**2-68*T1+16)*T2+
     2   18*T1**5-72*T1**4+116*T1**3-96*T1**2+42*T1-8)
      TMP0 = -T11**2*TMP0/( 144.D0*T1**2*T2**4*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = 2*T2*((27*T1+16)*T2+18*T1**2+14*T1-16)+RO*(-(9*T1+23)*T2-32
     1   *T1+32)+8*RO**2
      TMP0 = RLGXRO*T11*(T2+T1-1)**2*TMP0/( 144.D0*T2**2)
      PP = TMP0+PP
      PP = PP-T11*(RO*T1*T2*((9*T1**2+23*T1)*T2**2+(-576*T1**3+592*T1**2
     1   -272*T1+256)*T2-576*T1**4+864*T1**3-256*T1**2+224*T1-256)*TX+T1
     2   **2*T2**3*((9*T1+22)*T2-36*T1**2+8*T1-4)*TX+4*RO**2*(2*T1**2*T2
     3   **3+(-17*T1**3-T1**2+2*T1+16)*T2**2+(-72*T1**4+108*T1**3-32*T1*
     4   *2+28*T1-32)*T2-36*T1**5+90*T1**4-70*T1**3+30*T1**2-30*T1+16))/
     5   ( 144.D0*T1**2*T2**4)
      TMP0 = 2*(2*T2**2+(4*T1-4)*T2+4*T1**2-6*T1+3)+RO
      TMP0 = -RLGRO*(T2+T1-1)*(2*T2-1)*TMP0/( 8.D0*(T2-1)*T2)
      PP = TMP0+PP
      TMP0 = 2*T1*T2**2*(36*T1*T2**4+(144*T1**2-117*T1+16)*T2**3+(216*T1
     1   **3-378*T1**2+212*T1-48)*T2**2+(144*T1**4-468*T1**3+496*T1**2-2
     2   58*T1+64)*T2-144*T1**4+288*T1**3-280*T1**2+136*T1-32)
      TMP0 = TMP0-RO*T1*T2*((45*T1-32)*T2**3+(144*T1**2-230*T1+96)*T2**2
     1   +(144*T1**3-432*T1**2+384*T1-128)*T2-144*T1**3+288*T1**2-208*T1
     2   +64)+2*RO**2*(T2-1)*((5*T1-4)*T2**2+(18*T1**2-18*T1+8)*T2+18*T1
     3   **3-36*T1**2+26*T1-8)
      TMP0 = RLG21*(T2+T1-1)**2*TMP0/( 72.D0*T1**2*(T2-1)*T2**4)
      PP = TMP0+PP
      TMP0 = 2*T1*T2**2*(32*T2**5+(100*T1-64)*T2**4+(136*T1**2-159*T1+64
     1   )*T2**3+(72*T1**3-154*T1**2+136*T1-64)*T2**2+(-72*T1**3+154*T1*
     2   *2-159*T1+64)*T2+72*T1**3-136*T1**2+100*T1-32)
      TMP0 = TMP0-RO*(T2-1)*T2*(16*T2**4+(32*T1-16)*T2**3+(27*T1**2-32*T
     1   1)*T2**2+(64*T1-68*T1**2)*T2+208*T1**2-64*T1)+2*RO**2*(T2-1)*((
     2   5*T1-4)*T2**2+(8-18*T1)*T2+26*T1-8)
      TMP0 = RLG12*(T2+T1-1)**2*TMP0/( 72.D0*T1**2*(T2-1)*T2**4)
      PP = TMP0+PP
      TMP0 = 2*T2*(3*T2**2+(6*T1-4)*T2+4*T1**2-4*T1+2)*TX+RO*(T2**3+(T1+
     1   1)*T2**2+(4*T1-2)*T2-4*T1)
      TMP0 = RL3424*(T2-2)*(T2+T1-1)**2*TMP0/( 16.D0*(T2-1)*T2**2)
      PP = TMP0+PP
      TMP0 = RO*(-(16*T1+16)*T2**2-(16*T1**2-16*T1+7)*T2+9*T1)+2*T2*(T2+
     1   T1)*(7*T2-9*T1)+8*RO**2*T2
      TMP0 = DLAM2*RL34*(T2+T1-1)**2*TMP0/( 144.D0*T2)
      PP = TMP0+PP
      TMP0 = RO*(7*T2-9*T1-18)-2*(9*T2**2+(23*T1-16)*T2-18*T1**2+27*T1)
      TMP0 = -RL34*(T2+T1-1)**2*TMP0/( 144.D0*T2)
      PP = TMP0+PP
      TMP0 = 2*(32*T2**4+(64*T1-32)*T2**3+(32*T1**2-25*T1+7)*T2**2+(-2*T
     1   1**2-16*T1)*T2-9*T1**3+9*T1**2)+RO*(-16*T2**3-(16*T1+32)*T2**2-
     2   (32*T1-25)*T2+9*T1)+8*RO**2*T2
      TMP0 = DLAM2*(T2+T1-1)**2*TMP0/( 36.D0*T2*(4*TX+RO))
      PP = TMP0+PP
      TMP0 = -4*T1*T2**2*(64*T2**5+(464*T1-128)*T2**4+(688*T1**2-705*T1+
     1   32)*T2**3+(864*T1**3-1138*T1**2+470*T1+32)*T2**2+(576*T1**4-172
     2   8*T1**3+1066*T1**2-485*T1)*T2-576*T1**4+864*T1**3-544*T1**2+256
     3   *T1)*TX**2
      TMP0 = TMP0-RO*T2*((64*T1+128)*T2**6+(464*T1**2+256*T1-384)*T2**5+
     1   (688*T1**3+509*T1**2-864*T1+384)*T2**4+(864*T1**4+5576*T1**3-38
     2   48*T1**2+2976*T1-128)*T2**3+(576*T1**5+9216*T1**4-19144*T1**3+9
     3   851*T1**2-7040*T1)*T2**2+(4608*T1**5-21024*T1**4+21664*T1**3-12
     4   160*T1**2+6912*T1)*T2-5184*T1**5+10944*T1**4-8640*T1**3+5184*T1
     5   **2-2304*T1)*TX
      TMP0 = RO**2*(32*T2**6+(64*T1-64)*T2**5+(243*T1**2-56*T1)*T2**4+(1
     1   440*T1**3+375*T1**2+352*T1+640)*T2**3+(1296*T1**4-288*T1**3-276
     2   0*T1**2+504*T1-1760)*T2**2+(-3888*T1**3+4320*T1**2-2160*T1+1728
     3   )*T2-1296*T1**4+2736*T1**3-2160*T1**2+1296*T1-576)*TX+TMP0
      TMP0 = TMP0+2*RO**3*(T2-1)*((5*T1-4)*T2**3+(141*T1**2-27*T1+76)*T2
     1   **2+(324*T1**3-360*T1**2+180*T1-144)*T2+162*T1**4-342*T1**3+270
     2   *T1**2-162*T1+72)
      TMP0 = TMP0/( 144.D0*T1**2*(T2-1)*T2**4*(4*TX+RO))
      PP = TMP0+PP
      HQHPQG = PP
      RETURN
      ENDIF
      END

       function ggborn(s,t,m2)
c
c      d sigma (born_gg) = g^4 ggborn d phi2
c
       implicit real * 8 (a-z)
       v = 8
       n = 3
       p12 = s/2
       p13 = -t/2
       p23 = p12 - p13
          ggborn = 1/(2*v*n)*(v/(p13*p23)-2*n**2/p12**2)*
     #           (p13**2+p23**2+2*m2*p12-(m2*p12)**2/(p13*p23))
       ggborn = ggborn/(2*s)
       return
       end

       function qqborn(s,t,m2)
c
c      d sigma (born_qq) = g^4 qqborn d phi2
c
       implicit real * 8 (a-z)
       v = 8
       n2 = 3**2
       p12 = s/2
       p13 = -t/2
       p23 = p12 - p13
       qqborn = v/(2*n2)*( (p13**2+p23**2)/p12**2+m2/p12)
       qqborn = qqborn/(2*s)
       return
       end

       function ggcolp(s,q1q,x,m2,xlmude,nl)
c
c      d sigma_gg (c+-) = g^6 [ 1/(1-x)_rho ggcolp
c                    +       [log(1-x)/(1-x)]_rho * ggcoll ] d phi2x
c
c      xlmude = log(s/xmu2) + log(delta/2)
c
       implicit real * 8 (a-z)
       integer nl
       character * 2 schtmp
       common/schtmp/schtmp
       data pi/3.141 592 653 589 793/
       n = 3
       ggcolp = n/(4*pi**2) * xlmude *
     #           ( x + (1-x)**2/x + x*(1-x)**2 )
       if(schtmp.eq.'DI')then
           ggcolp = ggcolp - xkpgg(x,nl)/(8*pi**2)
       elseif(schtmp.ne.'MS')then
           write(6,*)'scheme ',schtmp,'not known'
           stop
       endif
       ggcolp = ggcolp * ggborn(s*x,q1q,m2)
       return
       end

       function ggcoll(s,q1q,x,m2,nl)
       implicit real * 8 (a-z)
       integer nl
       character * 2 schtmp
       common/schtmp/schtmp
       data pi/3.141 592 653 589 793/
       n = 3
       ggcoll = n/(4*pi**2) * 2 *
     #          ( x + (1-x)**2/x + x*(1-x)**2 )
       if(schtmp.eq.'DI')then
           ggcoll = ggcoll - xklgg(x,nl)/(8*pi**2)
       elseif(schtmp.ne.'MS')then
           write(6,*)'scheme ',schtmp,'not known'
           stop
       endif
       ggcoll = ggcoll * ggborn(s*x,q1q,m2)
       return
       end



       function qqcolp(s,q1q,x,m2,xlmude,nl)
c
c      d sigma_qq (c+-) = g^6 [ 1/(1-x)_rho qqcolp
c                    +       [log(1-x)/(1-x)]_rho * qqcoll ] d phi2x
c
c      xlmude = log(s/xmu2) + log(delta/2)
c
       implicit real * 8 (a-z)
       integer nl
       character * 2 schtmp
       common/schtmp/schtmp
       data pi/3.141 592 653 589 793/
       cf = 4.d0/3.d0
       qqcolp = cf/(8*pi**2) * (xlmude * (1+x**2) + (1-x)**2)
       if(schtmp.eq.'DI')then
           qqcolp = qqcolp - xkpqq(x,nl)/(8*pi**2)
       elseif(schtmp.ne.'MS')then
           write(6,*)'scheme ',schtmp,'not known'
           stop
       endif
       qqcolp = qqcolp * qqborn(s*x,q1q,m2)
       return
       end

       function qqcoll(s,q1q,x,m2,nl)
       implicit real * 8 (a-z)
       integer nl
       character * 2 schtmp
       common/schtmp/schtmp
       data pi/3.141 592 653 589 793/
       cf = 4.d0/3.d0
       qqcoll = cf/(8*pi**2) * 2 * (1+x**2)
       if(schtmp.eq.'DI')then
           qqcoll = qqcoll - xklqq(x,nl)/(8*pi**2)
       elseif(schtmp.ne.'MS')then
           write(6,*)'scheme ',schtmp,'not known'
           stop
       endif
       qqcoll = qqcoll * qqborn(s*x,q1q,m2)
       return
       end

       function qgcolp1(s,q2q,x,m2,xlmude,nl)
c
c      d sigma_qg (c+) = g^6 [ 1/(1-x)_rho qgcolp1
c                    +       [log(1-x)/(1-x)]_rho * qgcoll1 ] d phi2x
c
c      xlmude = log(s/xmu2) + log(delta/2)
c
       implicit real * 8 (a-z)
       integer nl
       character * 2 schhad1,schhad2
       common/scheme/schhad1,schhad2
       data pi/3.141 592 653 589 793/
       cf = 4.d0/3.d0
       qgcolp1 = cf/(8*pi**2) *
     #   ( xlmude*(1 + (1-x)**2)/x + x )*(1-x)
       if(schhad1.eq.'DI')then
           qgcolp1 = qgcolp1 - xkpgq(x,nl)/(8*pi**2)
       elseif(schhad1.ne.'MS')then
           write(6,*)'scheme ',schhad1,'not known'
           stop
       endif
       qgcolp1 = qgcolp1 * ggborn(s*x,q2q,m2)
       return
       end

       function qgcoll1(s,q2q,x,m2,nl)
       implicit real * 8 (a-z)
       integer nl
       character * 2 schhad1,schhad2
       common/scheme/schhad1,schhad2
       data pi/3.141 592 653 589 793/
       cf = 4.d0/3.d0
       qgcoll1 = cf/(8*pi**2) *
     #   2*(1 + (1-x)**2)/x * (1-x)
       if(schhad1.eq.'DI')then
           qgcoll1 = qgcoll1 - xklgq(x,nl)/(8*pi**2)
       elseif(schhad1.ne.'MS')then
           write(6,*)'scheme ',schhad1,'not known'
           stop
       endif
       qgcoll1 = qgcoll1 * ggborn(s*x,q2q,m2)
       return
       end


       function qgcolp2(s,q1q,x,m2,xlmude,nl)
c
c      d sigma_qg (c-) = g^6 [ 1/(1-x)_rho qgcolp2
c                    +       [log(1-x)/(1-x)]_rho * qgcoll2 ] d phi2x
c
c      xlmude = log(s/xmu2) + log(delta/2)
c
       implicit real * 8 (a-z)
       integer nl
       character * 2 schhad1,schhad2
       common/scheme/schhad1,schhad2
       data pi/3.141 592 653 589 793/
       tf = .5d0
       qgcolp2 =
     #  tf/(8*pi**2)*(xlmude*(x**2+(1-x)**2)+2*x*(1-x))*(1-x)
       if(schhad2.eq.'DI')then
           qgcolp2 = qgcolp2 - xkpqg(x,nl)/(8*pi**2)
       elseif(schhad2.ne.'MS')then
           write(6,*)'scheme ',schhad2,'not known'
           stop
       endif
       qgcolp2 = qgcolp2 * qqborn(s*x,q1q,m2)
       return
       end

       function qgcoll2(s,q1q,x,m2,nl)
       implicit real * 8 (a-z)
       integer nl
       character * 2 schhad1,schhad2
       common/scheme/schhad1,schhad2
       data pi/3.141 592 653 589 793/
       tf = .5d0
       qgcoll2 = tf/(8*pi**2)*2*(x**2+(1-x)**2)*(1-x)
       if(schhad2.eq.'DI')then
           qgcoll2 = qgcoll2 - xklqg(x,nl)/(8*pi**2)
       elseif(schhad2.ne.'MS')then
           write(6,*)'scheme ',schhad2,'not known'
           stop
       endif
       qgcoll2 = qgcoll2 * qqborn(s*x,q1q,m2)
       return
       end

       FUNCTION FGG(S,X,Y,XM2,Q1Q,Q2Q,W1H,W2H,CTH2)
c
c      d sigma_gg (f) = N g^6 / (64 pi^2 s) beta_x d costh1 d th2 dy dx
c                    1/(1-x)_rho ( 1/(1-y)_+ + 1/(1+y)_+ ) fgg
c
c           N = 1 / (4 pi)^2
c
       IMPLICIT REAL* 8(A-H,O-Z)
       REAL* 8 N
       TINY = .1d-6
       V = 8
       N = 3
       N2 = N*N
       CF = 4.D0/3.D0
       CA = 3
       TK = - (1-X)*(1-Y)*S/2
       UK = - (1-X)*(1+Y)*S/2
       IF(1-X.LT.TINY)THEN
          Q2C=-S-Q2Q
          P13 = -Q1Q/2
          P23 = -Q2C/2
          P12 = S/2
          BORN = 1/(2*V*N)*(V/(P13*P23)-2*N**2/P12**2)*
     #           (P13**2+P23**2+2*XM2*P12-(XM2*P12)**2/(P13*P23))
          XXX  = N**2/(4*V)*( (P12+2*XM2)/P23-(P12+2*XM2)/P13
     #           +XM2**2/P13**2-XM2**2/P23**2+2*(P23-P13)/P12 )
          YYY  = 1/(4*V*N**2)*
     #           (P13**2+P23**2+2*XM2*P12-(XM2*P12)**2/(P13*P23))
     #          *(1/(P13*P23)+2*N**2/P12**2)
C         --------------------------------------------------------------
C         Fattori iconali moltiplicati per 4*tk*uk
C         P1.K = -TK/2
C         P2.K = -UK/2
C         P3.K = W1/2
C         P4.K = W2/2
C
          P14 = P23
          P24 = P13
C
          E13 = 16*(1+Y)*P13/W1H
          E14 = 16*(1+Y)*P14/W2H
          E23 = 16*(1-Y)*P23/W1H
          E24 = 16*(1-Y)*P24/W2H
          E12 = 16*P12
          E33 = 16*(1-Y)*(1+Y)* XM2/W1H**2
          E44 = 16*(1-Y)*(1+Y)* XM2/W2H**2
          E34 = 16*(1-Y)*(1+Y)* (S/2-XM2)/(W1H*W2H)
          SUM = BORN*( CF*(E13+E14+E23+E24-2*E12-E33-E44) +2*N*E12 )
     #        +        XXX *(E14+E23-E13-E24)
     #        +        YYY *(2*E12+2*E34-E13-E24-E14-E23)
          FGG = 1/(2*S)*SUM
       ELSEIF(1-Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = - X*Q1Q/2
          P23 = -Q2C/2
          P12 = S*X/2
          BX = SQRT(1-4*XM2/(S*X))
          CTH1 = (P23-P13)/P12/BX
          AZIDEP =
     #    -512*BX**2*(CTH1-1)*(CTH1+1)*V*(2*V+BX**2*CTH1**2*N2-N2)
     #    *(X-1)**2*XM2/((BX*CTH1-1)**2*(BX*CTH1+1)**2*X**2)
          AZIDEP = AZIDEP * (2*CTH2**2-1)/2 /(4*64)
          BORN = 1/(2*V*N)*(V/(P13*P23)-2*N**2/P12**2)*
     #           (P13**2+P23**2+2*XM2*P12-(XM2*P12)**2/(P13*P23))
          SUM = - BORN*(8*UK)*2*CA*(X/(1-X)+(1-X)/X+X*(1-X))
     #    + AZIDEP
          FGG = 1/(2*X*S)*SUM
       ELSEIF(1+Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = -Q1Q/2
          P23 = -X*Q2C/2
          P12 = S*X/2
          BX = SQRT(1-4*XM2/(S*X))
          CTH1 = (P23-P13)/P12/BX
          AZIDEP =
     #    -512*BX**2*(CTH1-1)*(CTH1+1)*V*(2*V+BX**2*CTH1**2*N2-N2)
     #    *(X-1)**2*XM2/((BX*CTH1-1)**2*(BX*CTH1+1)**2*X**2)
          AZIDEP = AZIDEP * (2*CTH2**2-1)/2 /(4*64)
          BORN = 1/(2*V*N)*(V/(P13*P23)-2*N**2/P12**2)*
     #           (P13**2+P23**2+2*XM2*P12-(XM2*P12)**2/(P13*P23))
          SUM = - BORN*(8*TK)*2*CA*(X/(1-X)+(1-X)/X+X*(1-X))
     #          + AZIDEP
          FGG = 1/(2*X*S)*SUM
       ELSE
       S2 = S+TK+UK
       Q1C=-S-TK-Q1Q
       Q2C=-S-UK-Q2Q
       W1 =Q2Q-Q1Q-TK
       W2 =Q1Q-Q2Q-UK
C
       P12 = (S2 - 2*XM2)/2
       P13 = W1/2
       P14 = Q1Q/2
       P15 = Q2C/2
       P23 = W2/2
       P24 = Q1C/2
       P25 = Q2Q/2
       P34 = TK/2
       P35 = UK/2
       P45 = S/2
      ANS = FBB1(XM2,P12,P13,P14,P15,P23,P24,P25,P34,P35,P45)
      ANS = 4*TK*UK*ANS/(2*S*4*64)
      FGG = ANS
      ENDIF
      RETURN
      END

      FUNCTION FBB1(XM2,P12,P13,P14,P15,P23,P24,P25,P34,P35,P45)
      IMPLICIT REAL * 8 (A-Z)
      n = 3
      n2 = n**2
      v = n2-1
      s = 2*xm2+2*p12
      ans = 0
      tmp0 = v**3*xm2*((xm2**2+4*p24*xm2-2*p24*p25)/(p13**2*p24**2)+(p34
     1   *s+2*p15*p25-p14*p24)/(p13*p14*p23*p24))/n2/2+v*xm2*(xm2**2/(
     2   p13*p14*p23*p25)-(5*p13*s+6*p13*p25+8*p13*p23-4*p13**2)/(p13*p1
     3   4*p23*p24)/4)/n2
      tmp0 = v**2*xm2*(-4*xm2**2/(p13*p14*p25**2)+2*s*xm2/(p13*p14*p23*p
     1   25)+(-(p23+p13)*p45+p23**2+p13**2)/(p13*p14*p23*p25))/n2/4+((
     2   -p13*s+p15*p25+p23**2+p15**2+p13**2)/(p13*p15*p24*p25*p34)+(p24
     3   +p15)/(p13*p14*p23*p25))*v*xm2**2-v*((2*(p45+p35)*xm2+p25**2+p1
     4   5**2)/(p14*p23*p34)/2+p13*(2*(p35+p34)*xm2+p23**2+p13**2)/(p1
     5   4*p15*p25*p34))+(n2+1)*p12*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(n
     6   2*p13*p14*p23*p24)/4+v**2*xm2*((p45+p34)*xm2/(p13*p25*p34*p45
     7   )+2*xm2/(p15**2*p34)+(p13/p25-p23/p15)**2/p34**2)+tmp0
      ans = -2*n2*(s**2/4+p45**2+p34**2)*v*xm2**2/(p13*p25*p34*p45*s)+
     1   n2*p14*p24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p25*p34*p45*s
     2   )+2*n2*p23*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(p25*p34*p45*s)-(n
     3   2+1)*v*xm2*(4*p12*xm2+s**2+2*p23*p24+2*p13*p14)/(n2*p13*p14*p23
     4   *p24)/8+n2*v*((p34-2*xm2)/(p13*p24*s)-s/(p15*p25*p34)/4)*xm
     5   2+v*((p12-3*xm2)/(p15*p25*p34)+(5*p13*p25+3*p13*p23-2*p13**2)/(
     6   p13*p14*p23*p24))*xm2+2*n2*(-s*(s+2*p34)/8+4*p15*p25+p14*p24+
     7   p13*p23)*v*xm2/(p15*p25*p34*s)+4*n2*(2*p23**2/(p15*p34**2*s)+(p
     8   45**2+p35**2+p34**2)/(p34**2*s**2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p23*xm2-2*p23*p25)/(p14**2*p23**2)+(p34
     1   *s+2*p15*p25-p13*p23)/(p13*p14*p23*p24))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p14*p24*p25)-(5*p14*s+6*p14*p25+8*p14*p2
     1   4-4*p14**2)/(p13*p14*p23*p24)/4)/n2+v**2*xm2*(-4*xm2**2/(p13*
     2   p14*p25**2)+2*s*xm2/(p13*p14*p24*p25)+(-(p24+p14)*p35+p24**2+p1
     3   4**2)/(p13*p14*p24*p25))/n2/4+((-p14*s+p15*p25+p24**2+p15**2+
     4   p14**2)/(p14*p15*p23*p25*p34)+(p23+p15)/(p13*p14*p24*p25))*v*xm
     5   2**2-v*((2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p24*p34)/2+p14*(
     6   2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p15*p25*p34))+(n2+1)*p12*v*
     7   (2*(p45+p35)*xm2+p25**2+p15**2)/(n2*p13*p14*p23*p24)/4+2*n2*p
     8   24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p25*p34*p35*s)+tmp0
      ans = -2*n2*(s**2/4+p35**2+p34**2)*v*xm2**2/(p14*p25*p34*p35*s)+
     1   v**2*xm2*((p35+p34)*xm2/(p14*p25*p34*p35)+2*xm2/(p15**2*p34)+(p
     2   14/p25-p24/p15)**2/p34**2)+n2*p13*p23*v*(2*(p35+p34)*xm2+p23**2
     3   +p13**2)/(p14*p25*p34*p35*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23
     4   *p24+2*p13*p14)/(n2*p13*p14*p23*p24)/8+n2*v*((p34-2*xm2)/(p14
     5   *p23*s)-s/(p15*p25*p34)/4)*xm2+v*((p12-3*xm2)/(p15*p25*p34)+(
     6   5*p14*p25+3*p14*p24-2*p14**2)/(p13*p14*p23*p24))*xm2+2*n2*(-s*(
     7   s+2*p34)/8+4*p15*p25+p14*p24+p13*p23)*v*xm2/(p15*p25*p34*s)+4
     8   *n2*(2*p24**2/(p15*p34**2*s)+(p45**2+p35**2+p34**2)/(p34**2*s**
     9   2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p25*xm2-2*p23*p25)/(p14**2*p25**2)+(p45
     1   *s-p15*p25+2*p13*p23)/(p14*p15*p24*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p14*p15*p23*p24)-(5*p14*s+8*p14*p24+6*p14*p2
     1   3-4*p14**2)/(p14*p15*p24*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p14*
     2   p15*p23**2)+2*s*xm2/(p14*p15*p23*p24)+(-(p24+p14)*p35+p24**2+p1
     3   4**2)/(p14*p15*p23*p24))/n2/4-2*n2*(s**2/4+p45**2+p35**2)*v
     4   *xm2**2/(p14*p23*p35*p45*s)+((-p14*s+p24**2+p13*p23+p14**2+p13*
     5   *2)/(p13*p14*p23*p25*p45)+(p25+p13)/(p14*p15*p23*p24))*v*xm2**2
     6   -v*(p14*(2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p15*p23*p45)+(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p15*p24*p45)/2)+v**2*xm2*((p45+p3
     8   5)*xm2/(p14*p23*p35*p45)+2*xm2/(p13**2*p45)+(p14/p23-p24/p13)**
     9   2/p45**2)+n2*p15*p25*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(p14*p23
     :   *p35*p45*s)+tmp0
      ans = 2*n2*p24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p23*p35*p45*s)+(
     1   n2+1)*p12*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(n2*p14*p15*p24*p25
     2   )/4-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p24*p25+2*p14*p15)/(n2*p14
     3   *p15*p24*p25)/8+n2*v*((p45-2*xm2)/(p14*p25*s)-s/(p13*p23*p45)
     4   /4)*xm2+v*((p12-3*xm2)/(p13*p23*p45)+(3*p14*p24+5*p14*p23-2*p
     5   14**2)/(p14*p15*p24*p25))*xm2+2*n2*(-s*(s+2*p45)/8+p15*p25+p1
     6   4*p24+4*p13*p23)*v*xm2/(p13*p23*p45*s)+4*n2*(2*p24**2/(p13*p45*
     7   *2*s)+(p45**2+p35**2+p34**2)/(p45**2*s**2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p24*xm2-2*p23*p24)/(p15**2*p24**2)+(p45
     1   *s-p14*p24+2*p13*p23)/(p14*p15*p24*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p14*p15*p23*p25)-(5*p15*s+8*p15*p25+6*p15*p2
     1   3-4*p15**2)/(p14*p15*p24*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p14*
     2   p15*p23**2)+2*s*xm2/(p14*p15*p23*p25)+(-(p25+p15)*p34+p25**2+p1
     3   5**2)/(p14*p15*p23*p25))/n2/4-2*n2*(s**2/4+p45**2+p34**2)*v
     4   *xm2**2/(p15*p23*p34*p45*s)+((-p15*s+p25**2+p13*p23+p15**2+p13*
     5   *2)/(p13*p15*p23*p24*p45)+(p24+p13)/(p14*p15*p23*p25))*v*xm2**2
     6   -v*(p15*(2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p14*p23*p45)+(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p14*p25*p45)/2)+2*n2*p25*v*(2*(p4
     8   5+p35)*xm2+p25**2+p15**2)/(p23*p34*p45*s)+v**2*xm2*((p45+p34)*x
     9   m2/(p15*p23*p34*p45)+2*xm2/(p13**2*p45)+(p15/p23-p25/p13)**2/p4
     :   5**2)+tmp0
      ans = n2*p14*p24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p23*p34*p4
     1   5*s)+(n2+1)*p12*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(n2*p14*p15*p
     2   24*p25)/4-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p24*p25+2*p14*p15)/(
     3   n2*p14*p15*p24*p25)/8+n2*v*((p45-2*xm2)/(p15*p24*s)-s/(p13*p2
     4   3*p45)/4)*xm2+v*((p12-3*xm2)/(p13*p23*p45)+(3*p15*p25+5*p15*p
     5   23-2*p15**2)/(p14*p15*p24*p25))*xm2+2*n2*(-s*(s+2*p45)/8+p15*
     6   p25+p14*p24+4*p13*p23)*v*xm2/(p13*p23*p45*s)+4*n2*(2*p25**2/(p1
     7   3*p45**2*s)+(p45**2+p35**2+p34**2)/(p45**2*s**2))*v*xm2+tmp0+an
     8   s
      tmp0 = v**3*xm2*((xm2**2+4*p23*xm2-2*p23*p24)/(p15**2*p23**2)+(p35
     1   *s+2*p14*p24-p13*p23)/(p13*p15*p23*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p15*p24*p25)-(5*p15*s+8*p15*p25+6*p15*p2
     1   4-4*p15**2)/(p13*p15*p23*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p13*
     2   p15*p24**2)+2*s*xm2/(p13*p15*p24*p25)+(-(p25+p15)*p34+p25**2+p1
     3   5**2)/(p13*p15*p24*p25))/n2/4+((-p15*s+p25**2+p14*p24+p15**2+
     4   p14**2)/(p14*p15*p23*p24*p35)+(p23+p14)/(p13*p15*p24*p25))*v*xm
     5   2**2-v*(p15*(2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p14*p24*p35)+(
     6   2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p25*p35)/2)+2*n2*p25*v*(2
     7   *(p45+p35)*xm2+p25**2+p15**2)/(p24*p34*p35*s)+(n2+1)*p12*v*(2*(
     8   p45+p34)*xm2+p24**2+p14**2)/(n2*p13*p15*p23*p25)/4+tmp0
      ans = -2*n2*(s**2/4+p35**2+p34**2)*v*xm2**2/(p15*p24*p34*p35*s)+
     1   v**2*xm2*((p35+p34)*xm2/(p15*p24*p34*p35)+2*xm2/(p14**2*p35)+(p
     2   15/p24-p25/p14)**2/p35**2)+n2*p13*p23*v*(2*(p35+p34)*xm2+p23**2
     3   +p13**2)/(p15*p24*p34*p35*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23
     4   *p25+2*p13*p15)/(n2*p13*p15*p23*p25)/8+n2*v*((p35-2*xm2)/(p15
     5   *p23*s)-s/(p14*p24*p35)/4)*xm2+v*((p12-3*xm2)/(p14*p24*p35)+(
     6   3*p15*p25+5*p15*p24-2*p15**2)/(p13*p15*p23*p25))*xm2+2*n2*(-s*(
     7   s+2*p35)/8+p15*p25+4*p14*p24+p13*p23)*v*xm2/(p14*p24*p35*s)+4
     8   *n2*(2*p25**2/(p14*p35**2*s)+(p45**2+p35**2+p34**2)/(p35**2*s**
     9   2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p25*xm2-2*p24*p25)/(p13**2*p25**2)+(p35
     1   *s-p15*p25+2*p14*p24)/(p13*p15*p23*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p15*p23*p24)-(5*p13*s+6*p13*p24+8*p13*p2
     1   3-4*p13**2)/(p13*p15*p23*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p13*
     2   p15*p24**2)+2*s*xm2/(p13*p15*p23*p24)+(-(p23+p13)*p45+p23**2+p1
     3   3**2)/(p13*p15*p23*p24))/n2/4-2*n2*(s**2/4+p45**2+p35**2)*v
     4   *xm2**2/(p13*p24*p35*p45*s)+((-p13*s+p14*p24+p23**2+p14**2+p13*
     5   *2)/(p13*p14*p24*p25*p35)+(p25+p14)/(p13*p15*p23*p24))*v*xm2**2
     6   -v*((2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p23*p35)/2+p13*(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p14*p15*p24*p35))+v**2*xm2*((p45+p3
     8   5)*xm2/(p13*p24*p35*p45)+2*xm2/(p14**2*p35)+(p13/p24-p23/p14)**
     9   2/p35**2)+n2*p15*p25*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p24
     :   *p35*p45*s)+tmp0
      ans = (n2+1)*p12*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(n2*p13*p15*p23
     1   *p25)/4+2*n2*p23*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(p24*p35*p
     2   45*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23*p25+2*p13*p15)/(n2*p13
     3   *p15*p23*p25)/8+n2*v*((p35-2*xm2)/(p13*p25*s)-s/(p14*p24*p35)
     4   /4)*xm2+v*((p12-3*xm2)/(p14*p24*p35)+(5*p13*p24+3*p13*p23-2*p
     5   13**2)/(p13*p15*p23*p25))*xm2+2*n2*(-s*(s+2*p35)/8+p15*p25+4*
     6   p14*p24+p13*p23)*v*xm2/(p14*p24*p35*s)+4*n2*(2*p23**2/(p14*p35*
     7   *2*s)+(p45**2+p35**2+p34**2)/(p35**2*s**2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p14*xm2-2*p14*p15)/(p14**2*p23**2)+(p34
     1   *s+2*p15*p25-p14*p24)/(p13*p14*p23*p24))/n2/2+v*xm2*(xm2**2/(
     2   p13*p15*p23*p24)-(5*p23*s-4*p23**2+6*p15*p23+8*p13*p23)/(p13*p1
     3   4*p23*p24)/4)/n2
      tmp0 = v**2*xm2*(-4*xm2**2/(p15**2*p23*p24)+2*s*xm2/(p13*p15*p23*p
     1   24)+(-(p23+p13)*p45+p23**2+p13**2)/(p13*p15*p23*p24))/n2/4+((
     2   -p23*s+p25**2+p15*p25+p23**2+p13**2)/(p14*p15*p23*p25*p34)+(p25
     3   +p14)/(p13*p15*p23*p24))*v*xm2**2-v*((2*(p45+p35)*xm2+p25**2+p1
     4   5**2)/(p13*p24*p34)/2+p23*(2*(p35+p34)*xm2+p23**2+p13**2)/(p1
     5   5*p24*p25*p34))+(n2+1)*p12*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(n
     6   2*p13*p14*p23*p24)/4+v**2*xm2*((p45+p34)*xm2/(p15*p23*p34*p45
     7   )+2*xm2/(p25**2*p34)+(p23/p15-p13/p25)**2/p34**2)+tmp0
      ans = -2*n2*(s**2/4+p45**2+p34**2)*v*xm2**2/(p15*p23*p34*p45*s)+
     1   n2*p14*p24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p23*p34*p45*s
     2   )+2*n2*p13*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(p15*p34*p45*s)-(n
     3   2+1)*v*xm2*(4*p12*xm2+s**2+2*p23*p24+2*p13*p14)/(n2*p13*p14*p23
     4   *p24)/8+n2*v*((p34-2*xm2)/(p14*p23*s)-s/(p15*p25*p34)/4)*xm
     5   2+v*((p12-3*xm2)/(p15*p25*p34)+(-2*p23**2+5*p15*p23+3*p13*p23)/
     6   (p13*p14*p23*p24))*xm2+2*n2*(-s*(s+2*p34)/8+4*p15*p25+p14*p24
     7   +p13*p23)*v*xm2/(p15*p25*p34*s)+4*n2*(2*p13**2/(p25*p34**2*s)+(
     8   p45**2+p35**2+p34**2)/(p34**2*s**2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p13*xm2-2*p13*p15)/(p13**2*p24**2)+(p34
     1   *s+2*p15*p25-p13*p23)/(p13*p14*p23*p24))/n2/2
      tmp0 = v*xm2*(xm2**2/(p14*p15*p23*p24)-(5*p24*s-4*p24**2+6*p15*p24
     1   +8*p14*p24)/(p13*p14*p23*p24)/4)/n2+v**2*xm2*(-4*xm2**2/(p15*
     2   *2*p23*p24)+2*s*xm2/(p14*p15*p23*p24)+(-(p24+p14)*p35+p24**2+p1
     3   4**2)/(p14*p15*p23*p24))/n2/4+((-p24*s+p25**2+p15*p25+p24**2+
     4   p14**2)/(p13*p15*p24*p25*p34)+(p25+p13)/(p14*p15*p23*p24))*v*xm
     5   2**2-v*((2*(p45+p35)*xm2+p25**2+p15**2)/(p14*p23*p34)/2+p24*(
     6   2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p23*p25*p34))+(n2+1)*p12*v*
     7   (2*(p45+p35)*xm2+p25**2+p15**2)/(n2*p13*p14*p23*p24)/4+2*n2*p
     8   14*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p34*p35*s)+tmp0
      ans = -2*n2*(s**2/4+p35**2+p34**2)*v*xm2**2/(p15*p24*p34*p35*s)+
     1   v**2*xm2*((p35+p34)*xm2/(p15*p24*p34*p35)+2*xm2/(p25**2*p34)+(p
     2   24/p15-p14/p25)**2/p34**2)+n2*p13*p23*v*(2*(p35+p34)*xm2+p23**2
     3   +p13**2)/(p15*p24*p34*p35*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23
     4   *p24+2*p13*p14)/(n2*p13*p14*p23*p24)/8+n2*v*((p34-2*xm2)/(p13
     5   *p24*s)-s/(p15*p25*p34)/4)*xm2+v*((p12-3*xm2)/(p15*p25*p34)+(
     6   -2*p24**2+5*p15*p24+3*p14*p24)/(p13*p14*p23*p24))*xm2+2*n2*(-s*
     7   (s+2*p34)/8+4*p15*p25+p14*p24+p13*p23)*v*xm2/(p15*p25*p34*s)+
     8   4*n2*(2*p14**2/(p25*p34**2*s)+(p45**2+p35**2+p34**2)/(p34**2*s*
     9   *2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p15*xm2-2*p13*p15)/(p15**2*p24**2)+(p45
     1   *s-p15*p25+2*p13*p23)/(p14*p15*p24*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p14*p24*p25)-(5*p24*s-4*p24**2+8*p14*p24
     1   +6*p13*p24)/(p14*p15*p24*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p13*
     2   *2*p24*p25)+2*s*xm2/(p13*p14*p24*p25)+(-(p24+p14)*p35+p24**2+p1
     3   4**2)/(p13*p14*p24*p25))/n2/4-2*n2*(s**2/4+p45**2+p35**2)*v
     4   *xm2**2/(p13*p24*p35*p45*s)+((-p24*s+p24**2+p23**2+p13*p23+p14*
     5   *2)/(p13*p15*p23*p24*p45)+(p23+p15)/(p13*p14*p24*p25))*v*xm2**2
     6   -v*(p24*(2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p23*p25*p45)+(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p14*p25*p45)/2)+v**2*xm2*((p45+p3
     8   5)*xm2/(p13*p24*p35*p45)+2*xm2/(p23**2*p45)+(p24/p13-p14/p23)**
     9   2/p45**2)+n2*p15*p25*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p24
     :   *p35*p45*s)+tmp0
      ans = 2*n2*p14*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p35*p45*s)+(
     1   n2+1)*p12*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(n2*p14*p15*p24*p25
     2   )/4-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p24*p25+2*p14*p15)/(n2*p14
     3   *p15*p24*p25)/8+n2*v*((p45-2*xm2)/(p15*p24*s)-s/(p13*p23*p45)
     4   /4)*xm2+v*((p12-3*xm2)/(p13*p23*p45)+(-2*p24**2+3*p14*p24+5*p
     5   13*p24)/(p14*p15*p24*p25))*xm2+2*n2*(-s*(s+2*p45)/8+p15*p25+p
     6   14*p24+4*p13*p23)*v*xm2/(p13*p23*p45*s)+4*n2*(2*p14**2/(p23*p45
     7   **2*s)+(p45**2+p35**2+p34**2)/(p45**2*s**2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p14*xm2-2*p13*p14)/(p14**2*p25**2)+(p45
     1   *s-p14*p24+2*p13*p23)/(p14*p15*p24*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p15*p24*p25)-(5*p25*s-4*p25**2+8*p15*p25
     1   +6*p13*p25)/(p14*p15*p24*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p13*
     2   *2*p24*p25)+2*s*xm2/(p13*p15*p24*p25)+(-(p25+p15)*p34+p25**2+p1
     3   5**2)/(p13*p15*p24*p25))/n2/4-2*n2*(s**2/4+p45**2+p34**2)*v
     4   *xm2**2/(p13*p25*p34*p45*s)+((-p25*s+p25**2+p23**2+p13*p23+p15*
     5   *2)/(p13*p14*p23*p25*p45)+(p23+p14)/(p13*p15*p24*p25))*v*xm2**2
     6   -v*(p25*(2*(p45+p35)*xm2+p25**2+p15**2)/(p13*p23*p24*p45)+(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p15*p24*p45)/2)+2*n2*p15*v*(2*(p4
     8   5+p35)*xm2+p25**2+p15**2)/(p13*p34*p45*s)+v**2*xm2*((p45+p34)*x
     9   m2/(p13*p25*p34*p45)+2*xm2/(p23**2*p45)+(p25/p13-p15/p23)**2/p4
     :   5**2)+tmp0
      ans = n2*p14*p24*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p25*p34*p4
     1   5*s)+(n2+1)*p12*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(n2*p14*p15*p
     2   24*p25)/4-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p24*p25+2*p14*p15)/(
     3   n2*p14*p15*p24*p25)/8+n2*v*((p45-2*xm2)/(p14*p25*s)-s/(p13*p2
     4   3*p45)/4)*xm2+v*((p12-3*xm2)/(p13*p23*p45)+(-2*p25**2+3*p15*p
     5   25+5*p13*p25)/(p14*p15*p24*p25))*xm2+2*n2*(-s*(s+2*p45)/8+p15
     6   *p25+p14*p24+4*p13*p23)*v*xm2/(p13*p23*p45*s)+4*n2*(2*p15**2/(p
     7   23*p45**2*s)+(p45**2+p35**2+p34**2)/(p45**2*s**2))*v*xm2+tmp0+a
     8   ns
      tmp0 = v**3*xm2*((xm2**2+4*p13*xm2-2*p13*p14)/(p13**2*p25**2)+(p35
     1   *s+2*p14*p24-p13*p23)/(p13*p15*p23*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p14*p15*p23*p25)-(5*p25*s-4*p25**2+8*p15*p25
     1   +6*p14*p25)/(p13*p15*p23*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p14*
     2   *2*p23*p25)+2*s*xm2/(p14*p15*p23*p25)+(-(p25+p15)*p34+p25**2+p1
     3   5**2)/(p14*p15*p23*p25))/n2/4+((-p25*s+p25**2+p24**2+p14*p24+
     4   p15**2)/(p13*p14*p24*p25*p35)+(p24+p13)/(p14*p15*p23*p25))*v*xm
     5   2**2-v*(p25*(2*(p45+p35)*xm2+p25**2+p15**2)/(p14*p23*p24*p35)+(
     6   2*(p45+p34)*xm2+p24**2+p14**2)/(p15*p23*p35)/2)+2*n2*p15*v*(2
     7   *(p45+p35)*xm2+p25**2+p15**2)/(p14*p34*p35*s)+(n2+1)*p12*v*(2*(
     8   p45+p34)*xm2+p24**2+p14**2)/(n2*p13*p15*p23*p25)/4+tmp0
      ans = -2*n2*(s**2/4+p35**2+p34**2)*v*xm2**2/(p14*p25*p34*p35*s)+
     1   v**2*xm2*((p35+p34)*xm2/(p14*p25*p34*p35)+2*xm2/(p24**2*p35)+(p
     2   25/p14-p15/p24)**2/p35**2)+n2*p13*p23*v*(2*(p35+p34)*xm2+p23**2
     3   +p13**2)/(p14*p25*p34*p35*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23
     4   *p25+2*p13*p15)/(n2*p13*p15*p23*p25)/8+n2*v*((p35-2*xm2)/(p13
     5   *p25*s)-s/(p14*p24*p35)/4)*xm2+v*((p12-3*xm2)/(p14*p24*p35)+(
     6   -2*p25**2+3*p15*p25+5*p14*p25)/(p13*p15*p23*p25))*xm2+2*n2*(-s*
     7   (s+2*p35)/8+p15*p25+4*p14*p24+p13*p23)*v*xm2/(p14*p24*p35*s)+
     8   4*n2*(2*p15**2/(p24*p35**2*s)+(p45**2+p35**2+p34**2)/(p35**2*s*
     9   *2))*v*xm2+tmp0+ans
      tmp0 = v**3*xm2*((xm2**2+4*p15*xm2-2*p14*p15)/(p15**2*p23**2)+(p35
     1   *s-p15*p25+2*p14*p24)/(p13*p15*p23*p25))/n2/2
      tmp0 = v*xm2*(xm2**2/(p13*p14*p23*p25)-(5*p23*s-4*p23**2+6*p14*p23
     1   +8*p13*p23)/(p13*p15*p23*p25)/4)/n2+v**2*xm2*(-4*xm2**2/(p14*
     2   *2*p23*p25)+2*s*xm2/(p13*p14*p23*p25)+(-(p23+p13)*p45+p23**2+p1
     3   3**2)/(p13*p14*p23*p25))/n2/4-2*n2*(s**2/4+p45**2+p35**2)*v
     4   *xm2**2/(p14*p23*p35*p45*s)+((-p23*s+p24**2+p14*p24+p23**2+p13*
     5   *2)/(p14*p15*p23*p24*p35)+(p24+p15)/(p13*p14*p23*p25))*v*xm2**2
     6   -v*((2*(p45+p34)*xm2+p24**2+p14**2)/(p13*p25*p35)/2+p23*(2*(p
     7   35+p34)*xm2+p23**2+p13**2)/(p14*p24*p25*p35))+v**2*xm2*((p45+p3
     8   5)*xm2/(p14*p23*p35*p45)+2*xm2/(p24**2*p35)+(p23/p14-p13/p24)**
     9   2/p35**2)+n2*p15*p25*v*(2*(p45+p35)*xm2+p25**2+p15**2)/(p14*p23
     :   *p35*p45*s)+tmp0
      ans = (n2+1)*p12*v*(2*(p45+p34)*xm2+p24**2+p14**2)/(n2*p13*p15*p23
     1   *p25)/4+2*n2*p13*v*(2*(p35+p34)*xm2+p23**2+p13**2)/(p14*p35*p
     2   45*s)-(n2+1)*v*xm2*(4*p12*xm2+s**2+2*p23*p25+2*p13*p15)/(n2*p13
     3   *p15*p23*p25)/8+n2*v*((p35-2*xm2)/(p15*p23*s)-s/(p14*p24*p35)
     4   /4)*xm2+v*((p12-3*xm2)/(p14*p24*p35)+(-2*p23**2+5*p14*p23+3*p
     5   13*p23)/(p13*p15*p23*p25))*xm2+2*n2*(-s*(s+2*p35)/8+p15*p25+4
     6   *p14*p24+p13*p23)*v*xm2/(p14*p24*p35*s)+4*n2*(2*p13**2/(p24*p35
     7   **2*s)+(p45**2+p35**2+p34**2)/(p35**2*s**2))*v*xm2+tmp0+ans
      fbb1 = ans
      return
      end

      function ggqq2(s,t,m2,mu2,nl)
c
c     d sigma_gg (sv) = N g^6 ggqq2 d phi2
c        N = 1/(4*pi)**2
c
      implicit double precision (a-z)
      integer nl
      character * 2 schhad1,schhad2
      common/betfac/betfac,delta
      common/scheme/schhad1,schhad2
      data pi/3.141 592 653 589 793/
      ro = 4*m2/s
      t1 = -t/s
      zg = 1
      vca = 3
      vcf = 4
      vcf = vcf/ 3.d0
      vtf = 1
      vtf = vtf/ 2.d0
      vda = 8
      nlf = nl
      t2 = 1-t1
      b = dsqrt(1-ro)
      lp = (b+1)/ 2.d0
      lm = (1-b)/ 2.d0
      at = s*t1
      aw = s*t2
      vlm2 = dlog(m2/mu2)
      vltm = dlog(at/m2)
      vlpm = dlog(lp/lm)
      vlsm = dlog(s/m2)
      vlsmu = dlog(s/mu2)
      vlwm = dlog(aw/m2)
      vlbl = dlog(b/lm)
      vdw = ddilog((aw-m2)/aw)-vlwm**2/ 2.d0
      vdt = ddilog((at-m2)/at)-vltm**2/ 2.d0
      vdmp = ddilog(-lm/lp)
      vdmb = vlbl**2/ 2.d0+ddilog(-lm/b)
      auinv = 1/(m2-aw)
      atinv = 1/(m2-at)
      softt1 = ddilog(1-2*t1/(b+1))+ddilog(1-2*t1/(1-b))+log(2*t1/(1-b))
     1   *log(2*t1/(b+1))
      softt2 = ddilog(1-2*t2/(b+1))+ddilog(1-2*t2/(1-b))+log(2*t2/(1-b))
     1   *log(2*t2/(b+1))
      softb = ddilog(2*b/(b+1))-ddilog(-2*b/(1-b))
      lt1 = log(t1)
      lt2 = log(t2)
      lb = log(b*betfac)
      ss = -4*lb*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**
     1   2)*vca*(vcf+t1**2*vca-t1*vca)*vlsmu*vtf*zg**6/(s*(t1-1)**2*t1**
     2   2*vda)
      tmp0 = 4*nlf*vtf-11*vca
      tmp0 = (8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**2)*t
     1   mp0*(vcf+t1**2*vca-t1*vca)*vlsmu*vtf*zg**6/(s*(t1-1)**2*t1**2*v
     2   da* 6.d0)
      ss = tmp0+ss
      ss = ss-(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**2)*
     1   vca*(6*vcf+4*t1**2*vca-4*t1*vca-vca)*vlsm**2*vtf*zg**6/(s*(t1-1
     2   )**2*t1**2*vda* 4.d0)
      tmp0 = 2*vcf+2*t1**2*vca-2*t1*vca-vca
      tmp0 = (ro-2)*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+r
     1   o**2)*tmp0*(2*vcf-vca)*vlpm*vlsm*vtf*zg**6/(b*s*(t1-1)**2*t1**2
     2   *vda* 8.d0)
      ss = tmp0+ss
      ss = ss-lt2*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro*
     1   *2)*vca*(2*vcf+t1**2*vca-vca)*vlsm*vtf*zg**6/(s*(t1-1)**2*t1**2
     2   *vda* 2.d0)
      ss = ss-lt1*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro*
     1   *2)*vca*(2*vcf+t1**2*vca-2*t1*vca)*vlsm*vtf*zg**6/(s*(t1-1)**2*
     2   t1**2*vda* 2.d0)
      ss = ss-lb*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**
     1   2)*vca*(4*vcf+2*t1**2*vca-2*t1*vca-vca)*vlsm*vtf*zg**6/(s*(t1-1
     2   )**2*t1**2*vda)
      tmp0 = 32*nlf*t1**4*vcf*vtf-64*nlf*t1**3*vcf*vtf+16*nlf*ro*t1**2*v
     1   cf*vtf+48*nlf*t1**2*vcf*vtf-16*nlf*ro*t1*vcf*vtf-16*nlf*t1*vcf*
     2   vtf+4*nlf*ro**2*vcf*vtf+32*nlf*t1**6*vca*vtf-96*nlf*t1**5*vca*v
     3   tf+16*nlf*ro*t1**4*vca*vtf
      tmp0 = 112*nlf*t1**4*vca*vtf-32*nlf*ro*t1**3*vca*vtf-64*nlf*t1**3*
     1   vca*vtf+4*nlf*ro**2*t1**2*vca*vtf+16*nlf*ro*t1**2*vca*vtf+16*nl
     2   f*t1**2*vca*vtf-4*nlf*ro**2*t1*vca*vtf-48*t1**4*vcf**2+96*t1**3
     3   *vcf**2-24*ro*t1**2*vcf**2-72*t1**2*vcf**2+24*ro*t1*vcf**2+24*t
     4   1*vcf**2-6*ro**2*vcf**2-48*t1**6*vca*vcf+144*t1**5*vca*vcf-24*r
     5   o*t1**4*vca*vcf-352*t1**4*vca*vcf+48*ro*t1**3*vca*vcf+464*t1**3
     6   *vca*vcf-6*ro**2*t1**2*vca*vcf-164*ro*t1**2*vca*vcf-252*t1**2*v
     7   ca*vcf+6*ro**2*t1*vca*vcf+140*ro*t1*vca*vcf+44*t1*vca*vcf-35*ro
     8   **2*vca*vcf-160*t1**6*vca**2+480*t1**5*vca**2-116*ro*t1**4*vca*
     9   *2-512*t1**4*vca**2+232*ro*t1**3*vca**2+224*t1**3*vca**2-29*ro*
     :   *2*t1**2*vca**2-104*ro*t1**2*vca**2-32*t1**2*vca**2+29*ro**2*t1
     ;   *vca**2-12*ro*t1*vca**2+3*ro**2*vca**2+tmp0
      tmp0 = -tmp0*vlsm*vtf*zg**6/(s*(t1-1)**2*t1**2*vda* 6.d0)
      ss = tmp0+ss
      ss = ss-(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**2)*
     1   vca*(4*vcf+2*t1**2*vca-2*t1*vca-vca)*vlpm**2*vtf*zg**6/(s*(t1-1
     2   )**2*t1**2*vda* 8.d0)
      tmp0 = 2*vcf+2*t1**2*vca-2*t1*vca-vca
      tmp0 = lb*(ro-2)*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t
     1   1+ro**2)*tmp0*(2*vcf-vca)*vlpm*vtf*zg**6/(b*s*(t1-1)**2*t1**2*v
     2   da* 2.d0)
      ss = tmp0+ss
      tmp0 = 16*ro*t1**4*vcf**2-32*ro*t1**3*vcf**2+16*ro**2*t1**2*vcf**2
      tmp0 = 16*t1**2*vcf**2-16*ro**2*t1*vcf**2+16*ro*t1*vcf**2-16*t1*vc
     1   f**2+4*ro**3*vcf**2-4*ro**2*vcf**2+16*ro*t1**6*vca*vcf-48*ro*t1
     2   **5*vca*vcf+16*ro**2*t1**4*vca*vcf+16*ro*t1**4*vca*vcf+48*t1**4
     3   *vca*vcf-32*ro**2*t1**3*vca*vcf+48*ro*t1**3*vca*vcf-96*t1**3*vc
     4   a*vcf+4*ro**3*t1**2*vca*vcf-4*ro**2*t1**2*vca*vcf+48*t1**2*vca*
     5   vcf-4*ro**3*t1*vca*vcf+20*ro**2*t1*vca*vcf-32*ro*t1*vca*vcf-4*r
     6   o**3*vca*vcf+8*ro**2*vca*vcf-8*ro*t1**6*vca**2+16*t1**6*vca**2+
     7   24*ro*t1**5*vca**2-48*t1**5*vca**2-8*ro**2*t1**4*vca**2-4*ro*t1
     8   **4*vca**2+40*t1**4*vca**2+16*ro**2*t1**3*vca**2-32*ro*t1**3*vc
     9   a**2-2*ro**3*t1**2*vca**2+12*ro*t1**2*vca**2-8*t1**2*vca**2+2*r
     :   o**3*t1*vca**2-8*ro**2*t1*vca**2+8*ro*t1*vca**2+ro**3*vca**2-2*
     ;   ro**2*vca**2+tmp0
      tmp0 = -tmp0*vlpm*vtf*zg**6/(b*s*(t1-1)**2*t1**2*vda* 4.d0)
      ss = tmp0+ss
      ss = ss-softt2*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+
     1   ro**2)*vca*(2*vcf+t1**2*vca-vca)*vtf*zg**6/(s*(t1-1)**2*t1**2*v
     2   da* 2.d0)
      ss = ss-softt1*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+
     1   ro**2)*vca*(2*vcf+t1**2*vca-2*t1*vca)*vtf*zg**6/(s*(t1-1)**2*t1
     2   **2*vda* 2.d0)
      tmp0 = 2*vcf+2*t1**2*vca-2*t1*vca-vca
      tmp0 = -(ro-2)*softb*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1
     1   -4*t1+ro**2)*tmp0*(2*vcf-vca)*vtf*zg**6/(b*s*(t1-1)**2*t1**2*vd
     2   a* 8.d0)
      ss = tmp0+ss
      ss = pi**2*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**
     1   2)*vca*(vcf+t1**2*vca-t1*vca)*vtf*zg**6/(s*(t1-1)**2*t1**2*vda*
     2    3.d0)+ss
      ss = ss-2*lb*lt2*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t
     1   1+ro**2)*vca*(2*vcf+t1**2*vca-vca)*vtf*zg**6/(s*(t1-1)**2*t1**2
     2   *vda)
      ss = lt2*(2*t1**2-2*t1+ro)**2*vca*(2*vcf+t1**2*vca-vca)*vtf*zg**6/
     1   (s*(t1-1)**2*t1**2*vda)+ss
      ss = ss-2*lb*lt1*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t
     1   1+ro**2)*vca*(2*vcf+t1**2*vca-2*t1*vca)*vtf*zg**6/(s*(t1-1)**2*
     2   t1**2*vda)
      ss = lt1*(2*t1**2-2*t1+ro)**2*vca*(2*vcf+t1**2*vca-2*t1*vca)*vtf*z
     1   g**6/(s*(t1-1)**2*t1**2*vda)+ss
      ss = ss-8*lb**2*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1
     1   +ro**2)*vca*(vcf+t1**2*vca-t1*vca)*vtf*zg**6/(s*(t1-1)**2*t1**2
     2   *vda)
      ss = 4*lb*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro**2
     1   )*vcf*(vcf+t1**2*vca-t1*vca)*vtf*zg**6/(s*(t1-1)**2*t1**2*vda)+
     2   ss
      tmp0 = 8*nlf*t1**4*vtf-16*nlf*t1**3*vtf+8*nlf*ro*t1**2*vtf+8*nlf*t
     1   1**2*vtf-8*nlf*ro*t1*vtf+2*nlf*ro**2*vtf-12*t1**4*vcf+24*t1**3*
     2   vcf-12*ro*t1**2*vcf-12*t1**2*vcf+12*ro*t1*vcf-3*ro**2*vcf-34*t1
     3   **4*vca+68*t1**3*vca-40*ro*t1**2*vca-34*t1**2*vca+40*ro*t1*vca-
     4   10*ro**2*vca
      tmp0 = 2*tmp0*(vcf+t1**2*vca-t1*vca)*vtf*zg**6/(s*(t1-1)**2*t1**2*
     1   vda* 3.d0)
      ss = tmp0+ss
      dd = 2*vcf+t1*vca-vca
      dd = -auinv*dd*pi*(4*t1+ro-4)*(8*t1**4-16*t1**3+12*ro*t1**2+8*t1**
     1   2-12*ro*t1+3*ro**2)*(2*vcf-vca)*vlwm**2*vtf*zg**6/((t1-1)**3*t1
     2   *vda* 8.d0)
      tmp0 = 8*ro*t1**3*vcf+16*t1**3*vcf+4*ro**2*t1**2*vcf-8*ro*t1**2*vc
     1   f-48*t1**2*vcf-8*ro**2*t1*vcf+4*ro*t1*vcf+48*t1*vcf+6*ro**2*vcf
     2   -4*ro*vcf-16*vcf+4*ro*t1**4*vca+4*t1**4*vca+2*ro**2*t1**3*vca-1
     3   2*ro*t1**3*vca-24*t1**3*vca-8*ro**2*t1**2*vca+18*ro*t1**2*vca+4
     4   8*t1**2*vca+11*ro**2*t1*vca-16*ro*t1*vca-40*t1*vca-5*ro**2*vca+
     5   6*ro*vca+12*vca
      tmp0 = pi*tmp0*(2*vcf-vca)*vlwm**2*vtf*zg**6/(s*(t1-1)**3*t1*vda)
      dd = tmp0+dd
      dd = 4*pi*(4*t1**4-8*t1**3+ro**2*t1**2+2*ro*t1**2+6*t1**2-ro**2*t1
     1   -2*ro*t1-2*t1+ro**2)*vca*(2*vcf-vca)*vltm*vlwm*vtf*zg**6/(s*(t1
     2   -1)**2*t1**2*vda)+dd
      dd = 2*pi*(8*t1**4-24*t1**3+2*ro*t1**2+30*t1**2-2*ro*t1-18*t1+ro**
     1   2+4)*vca**2*vlsm*vlwm*vtf*zg**6/(s*(t1-1)**2*vda)+dd
      tmp0 = 2*vcf+t1*vca-2*vca
      tmp0 = pi*(4*t1**3+2*ro**2*t1**2+2*ro*t1**2-12*t1**2-6*ro**2*t1-2*
     1   ro*t1+16*t1-ro**3+6*ro**2-8)*tmp0*(2*vcf-vca)*vlpm*vlwm*vtf*zg*
     2   *6/(b*s*(t1-1)**2*t1*vda)
      dd = tmp0+dd
      tmp0 = 96*t1**5*vcf+16*ro*t1**4*vcf-256*t1**4*vcf+96*ro*t1**3*vcf+
     1   256*t1**3*vcf+24*ro**2*t1**2*vcf-192*ro*t1**2*vcf-192*t1**2*vcf
     2   +12*ro**2*t1*vcf+32*ro*t1*vcf+160*t1*vcf+6*ro**3*vcf-36*ro**2*v
     3   cf+48*ro*vcf-64*vcf-32*t1**5*vca-8*ro*t1**4*vca-56*ro*t1**3*vca
     4   +192*t1**3*vca-12*ro**2*t1**2*vca+104*ro*t1**2*vca-256*t1**2*vc
     5   a-8*ro**2*t1*vca-8*ro*t1*vca+96*t1*vca-3*ro**3*vca+20*ro**2*vca
     6   -32*ro*vca
      tmp0 = auinv**2*pi*s*tmp0*(2*vcf+t1*vca-vca)*vlwm*vtf*zg**6/((t1-1
     1   )**2*t1*vda* 8.d0)
      dd = tmp0+dd
      tmp0 = 16*t1**4*vcf+8*ro*t1**3*vcf-20*t1**3*vcf+2*ro**2*t1**2*vcf+
     1   2*ro*t1**2*vcf+2*ro**2*t1*vcf-10*ro*t1*vcf-4*t1*vcf+ro**3*vcf-2
     2   *ro**2*vcf+8*vcf+4*t1**4*vca-28*t1**3*vca-2*ro*t1**2*vca+44*t1*
     3   *2*vca-4*ro*t1*vca-20*t1*vca-ro**2*vca+6*ro*vca
      tmp0 = -auinv*pi*tmp0*(2*vcf+t1*vca-vca)*vlwm*vtf*zg**6/((t1-1)**2
     1   *t1*vda)
      dd = tmp0+dd
      dd = dd-4*pi*(2*t1**2-2*t1+ro)**2*vca*(2*vcf+t1**2*vca-vca)*vlwm*v
     1   tf*zg**6/(s*(t1-1)**2*t1**2*vda)
      tmp0 = 2*vcf-t1*vca
      tmp0 = atinv*pi*(4*t1-ro)*(8*t1**4-16*t1**3+12*ro*t1**2+8*t1**2-12
     1   *ro*t1+3*ro**2)*tmp0*(2*vcf-vca)*vltm**2*vtf*zg**6/((t1-1)*t1**
     2   3*vda* 8.d0)
      dd = tmp0+dd
      dd = dd-pi*(2*vcf-vca)*(8*ro*t1**3*vcf+16*t1**3*vcf-4*ro**2*t1**2*
     1   vcf-16*ro*t1**2*vcf+12*ro*t1*vcf-2*ro**2*vcf-4*ro*t1**4*vca-4*t
     2   1**4*vca+2*ro**2*t1**3*vca+4*ro*t1**3*vca-8*t1**3*vca+2*ro**2*t
     3   1**2*vca-6*ro*t1**2*vca+ro**2*t1*vca)*vltm**2*vtf*zg**6/(s*(t1-
     4   1)*t1**3*vda)
      dd = 2*pi*(8*t1**4-8*t1**3+2*ro*t1**2+6*t1**2-2*ro*t1-2*t1+ro**2)*
     1   vca**2*vlsm*vltm*vtf*zg**6/(s*t1**2*vda)+dd
      tmp0 = 2*vcf-t1*vca-vca
      tmp0 = pi*(4*t1**3-2*ro**2*t1**2-2*ro*t1**2-2*ro**2*t1+2*ro*t1+4*t
     1   1+ro**3-2*ro**2)*tmp0*(2*vcf-vca)*vlpm*vltm*vtf*zg**6/(b*s*(t1-
     2   1)*t1**2*vda)
      dd = tmp0+dd
      dd = atinv**2*pi*s*(2*vcf-t1*vca)*(96*t1**5*vcf-16*ro*t1**4*vcf-22
     1   4*t1**4*vcf+160*ro*t1**3*vcf+192*t1**3*vcf-24*ro**2*t1**2*vcf-1
     2   92*ro*t1**2*vcf+60*ro**2*t1*vcf-6*ro**3*vcf-32*t1**5*vca+8*ro*t
     3   1**4*vca+160*t1**4*vca-88*ro*t1**3*vca-128*t1**3*vca+12*ro**2*t
     4   1**2*vca+112*ro*t1**2*vca-32*ro**2*t1*vca+3*ro**3*vca)*vltm*vtf
     5   *zg**6/((t1-1)*t1**2*vda* 8.d0)+dd
      dd = atinv*pi*(2*vcf-t1*vca)*(16*t1**4*vcf-8*ro*t1**3*vcf-44*t1**3
     1   *vcf+2*ro**2*t1**2*vcf+26*ro*t1**2*vcf+36*t1**2*vcf-6*ro**2*t1*
     2   vcf-18*ro*t1*vcf+ro**3*vcf+2*ro**2*vcf+4*t1**4*vca+12*t1**3*vca
     3   -2*ro*t1**2*vca-16*t1**2*vca+8*ro*t1*vca-ro**2*vca)*vltm*vtf*zg
     4   **6/((t1-1)*t1**2*vda)+dd
      dd = dd-4*pi*(2*t1**2-2*t1+ro)**2*vca*(2*vcf+t1**2*vca-2*t1*vca)*v
     1   ltm*vtf*zg**6/(s*(t1-1)**2*t1**2*vda)
      dd = pi*(ro+1)*(2*t1**2-2*t1+1)*vca**2*vlsm**2*vtf*zg**6/(s*(t1-1)
     1   *t1*vda)+dd
      tmp0 = 16*t1**4*vcf-32*t1**3*vcf-12*ro**2*t1**2*vcf+4*ro*t1**2*vcf
     1   +40*t1**2*vcf+12*ro**2*t1*vcf-4*ro*t1*vcf-24*t1*vcf-2*ro**3*vcf
     2   +4*ro**2*vcf+4*ro**2*t1**4*vca+4*ro*t1**4*vca-20*t1**4*vca-8*ro
     3   **2*t1**3*vca-8*ro*t1**3*vca+40*t1**3*vca-2*ro**3*t1**2*vca+18*
     4   ro**2*t1**2*vca+2*ro*t1**2*vca-40*t1**2*vca+2*ro**3*t1*vca-14*r
     5   o**2*t1*vca+2*ro*t1*vca+20*t1*vca+ro**3*vca-2*ro**2*vca
      tmp0 = -pi*tmp0*(2*vcf-vca)*vlpm*vlsm*vtf*zg**6/(b*s*(t1-1)**2*t1*
     1   *2*vda* 2.d0)
      dd = tmp0+dd
      dd = pi*(2*t1-1)**2*(4*ro*t1**2-12*t1**2-4*ro*t1+12*t1-ro**2+ro-4)
     1   *vca**2*vlsm*vtf*zg**6/(b**2*s*(t1-1)*t1*vda)+dd
      dd = 3*pi*(2*t1-1)**2*(4*t1**2-4*t1-ro+2)*vca**2*vlsm*vtf*zg**6/(b
     1   **4*s*(t1-1)*t1*vda* 2.d0)+dd
      dd = dd-pi*(16*t1**6-48*t1**5+24*ro*t1**4+48*t1**4-48*ro*t1**3-16*
     1   t1**3+4*ro**2*t1**2+34*ro*t1**2-t1**2-4*ro**2*t1-10*ro*t1+t1+2*
     2   ro**2)*vca**2*vlsm*vtf*zg**6/(s*(t1-1)**2*t1**2*vda)
      dd = pi*(32*ro*t1**2*vcf**2-32*t1**2*vcf**2-32*ro*t1*vcf**2+32*t1*
     1   vcf**2-8*ro**2*vcf**2-8*ro*vcf**2+16*vcf**2+32*ro*t1**4*vca*vcf
     2   -64*t1**4*vca*vcf-64*ro*t1**3*vca*vcf+128*t1**3*vca*vcf+24*ro**
     3   2*t1**2*vca*vcf-8*ro*t1**2*vca*vcf-88*t1**2*vca*vcf-24*ro**2*t1
     4   *vca*vcf+40*ro*t1*vca*vcf+24*t1*vca*vcf+16*ro**2*vca*vcf+8*ro*v
     5   ca*vcf-32*vca*vcf-16*ro*t1**4*vca**2-32*t1**4*vca**2+32*ro*t1**
     6   3*vca**2+64*t1**3*vca**2-12*ro**2*t1**2*vca**2-5*ro*t1**2*vca**
     7   2-66*t1**2*vca**2+12*ro**2*t1*vca**2-11*ro*t1*vca**2+34*t1*vca*
     8   *2-6*ro**2*vca**2-2*ro*vca**2+4*vca**2)*vlpm**2*vtf*zg**6/(b*s*
     9   (t1-1)*t1*vda* 4.d0)+dd
      dd = dd-pi*(16*ro*t1**4-48*t1**4-32*ro*t1**3+96*t1**3-7*ro**2*t1**
     1   2+49*ro*t1**2-98*t1**2+7*ro**2*t1-33*ro*t1+50*t1+ro**3-3*ro**2+
     2   6*ro-8)*vca**2*vlpm**2*vtf*zg**6/(b**3*s*(t1-1)*t1*vda* 4.d0)
      dd = dd-pi*(2*t1-1)**2*(4*ro*t1**2+8*t1**2-4*ro*t1-8*t1+3*ro**2-8*
     1   ro+8)*vca**2*vlpm**2*vtf*zg**6/(b**5*s*(t1-1)*t1*vda* 8.d0)
      dd = dd-pi*vlpm**2*vtf*(8*ro**2*t1**2*vca*vtf-8*ro**2*t1*vca*vtf+2
     1   *ro**2*vca*vtf+32*t1**2*vcf**2-32*t1*vcf**2-8*ro**2*vcf**2+48*v
     2   cf**2-16*ro*t1**2*vca*vcf-56*t1**2*vca*vcf+16*ro*t1*vca*vcf+56*
     3   t1*vca*vcf+10*ro**2*vca*vcf-4*ro*vca*vcf-64*vca*vcf+8*ro*t1**2*
     4   vca**2+20*t1**2*vca**2-8*ro*t1*vca**2-20*t1*vca**2-3*ro**2*vca*
     5   *2+2*ro*vca**2+20*vca**2)*zg**6/(s*(t1-1)*t1*vda* 4.d0)
      dd = b*pi*vlpm*vtf*(8*ro**2*t1**2*vca*vtf-8*ro**2*t1*vca*vtf+2*ro*
     1   *2*vca*vtf+8*vcf**2-16*ro*t1**2*vca*vcf-16*t1**2*vca*vcf+16*ro*
     2   t1*vca*vcf+16*t1*vca*vcf-4*ro*vca*vcf-14*vca*vcf+8*ro*t1**2*vca
     3   **2+8*t1**2*vca**2-8*ro*t1*vca**2-8*t1*vca**2+2*ro*vca**2+5*vca
     4   **2)*zg**6/(s*(t1-1)*t1*vda)+dd
      tmp0 = 16*ro*t1**4*vcf-64*t1**4*vcf-32*ro*t1**3*vcf+128*t1**3*vcf+
     1   16*ro**2*t1**2*vcf-16*ro*t1**2*vcf-72*t1**2*vcf-16*ro**2*t1*vcf
     2   +32*ro*t1*vcf+8*t1*vcf+4*ro**3*vcf-8*ro**2*vcf+48*ro*t1**6*vca-
     3   80*t1**6*vca-144*ro*t1**5*vca+240*t1**5*vca+8*ro**2*t1**4*vca+1
     4   00*ro*t1**4*vca-196*t1**4*vca-16*ro**2*t1**3*vca+40*ro*t1**3*vc
     5   a-8*t1**3*vca+4*ro**3*t1**2*vca-10*ro**2*t1**2*vca-31*ro*t1**2*
     6   vca+54*t1**2*vca-4*ro**3*t1*vca+18*ro**2*t1*vca-13*ro*t1*vca-10
     7   *t1*vca-2*ro**3*vca+4*ro**2*vca
      tmp0 = pi*tmp0*(2*vcf-vca)*vlpm*vtf*zg**6/(b*s*(t1-1)**2*t1**2*vda
     1   * 2.d0)
      dd = tmp0+dd
      tmp0 = 4*nlf*vtf-11*vca
      tmp0 = -2*pi*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*t1+ro
     1   **2)*tmp0*(vcf+t1**2*vca-t1*vca)*vlm2*vtf*zg**6/(s*(t1-1)**2*t1
     2   **2*vda* 3.d0)
      dd = tmp0+dd
      tmp0 = 16*t1**4*vcf-32*t1**3*vcf+24*ro*t1**2*vcf+16*t1**2*vcf-24*r
     1   o*t1*vcf+6*ro**2*vcf-8*t1**4*vca+16*t1**3*vca-12*ro*t1**2*vca-4
     2   *t1**2*vca+12*ro*t1*vca-8*t1*vca-3*ro**2*vca+4*vca
      tmp0 = -auinv*pi*(4*t1+ro-4)*tmp0*(2*vcf+t1*vca-vca)*vdw*vtf*zg**6
     1   /((t1-1)**3*t1*vda* 8.d0)
      dd = tmp0+dd
      tmp0 = 16*ro*t1**3*vcf**2
      tmp0 = 32*t1**3*vcf**2+8*ro**2*t1**2*vcf**2-16*ro*t1**2*vcf**2-96*
     1   t1**2*vcf**2-16*ro**2*t1*vcf**2+8*ro*t1*vcf**2+96*t1*vcf**2+12*
     2   ro**2*vcf**2-8*ro*vcf**2-32*vcf**2+8*ro*t1**4*vca*vcf-8*t1**4*v
     3   ca*vcf+4*ro**2*t1**3*vca*vcf-32*ro*t1**3*vca*vcf-16*t1**3*vca*v
     4   cf-16*ro**2*t1**2*vca*vcf+36*ro*t1**2*vca*vcf+92*t1**2*vca*vcf+
     5   22*ro**2*t1*vca*vcf-20*ro*t1*vca*vcf-104*t1*vca*vcf-12*ro**2*vc
     6   a*vcf+8*ro*vca*vcf+36*vca*vcf-16*t1**5*vca**2-8*ro*t1**4*vca**2
     7   +56*t1**4*vca**2-2*ro**2*t1**3*vca**2+20*ro*t1**3*vca**2-62*t1*
     8   *3*vca**2+6*ro**2*t1**2*vca**2-18*ro*t1**2*vca**2+10*t1**2*vca*
     9   *2-7*ro**2*t1*vca**2+8*ro*t1*vca**2+22*t1*vca**2+3*ro**2*vca**2
     :   -2*ro*vca**2-10*vca**2+tmp0
      tmp0 = pi*tmp0*vdw*vtf*zg**6/(s*(t1-1)**3*t1*vda)
      dd = tmp0+dd
      tmp0 = 16*t1**4*vcf-32*t1**3*vcf+24*ro*t1**2*vcf+16*t1**2*vcf-24*r
     1   o*t1*vcf+6*ro**2*vcf-8*t1**4*vca+16*t1**3*vca-12*ro*t1**2*vca-4
     2   *t1**2*vca+12*ro*t1*vca-3*ro**2*vca
      tmp0 = atinv*pi*(4*t1-ro)*tmp0*(2*vcf-t1*vca)*vdt*vtf*zg**6/((t1-1
     1   )*t1**3*vda* 8.d0)
      dd = tmp0+dd
      dd = dd-pi*(16*ro*t1**3*vcf**2+32*t1**3*vcf**2-8*ro**2*t1**2*vcf**
     1   2-32*ro*t1**2*vcf**2+24*ro*t1*vcf**2-4*ro**2*vcf**2-8*ro*t1**4*
     2   vca*vcf+8*t1**4*vca*vcf+4*ro**2*t1**3*vca*vcf-48*t1**3*vca*vcf+
     3   4*ro**2*t1**2*vca*vcf+12*ro*t1**2*vca*vcf+4*t1**2*vca*vcf+2*ro*
     4   *2*t1*vca*vcf-12*ro*t1*vca*vcf+2*ro**2*vca*vcf-16*t1**5*vca**2+
     5   8*ro*t1**4*vca**2+24*t1**4*vca**2-2*ro**2*t1**3*vca**2-12*ro*t1
     6   **3*vca**2+2*t1**3*vca**2+6*ro*t1**2*vca**2-ro**2*t1*vca**2)*vd
     7   t*vtf*zg**6/(s*(t1-1)*t1**3*vda)
      tmp0 = 32*t1**4*vcf**2-64*t1**3*vcf**2-24*ro**2*t1**2*vcf**2+8*ro*
     1   t1**2*vcf**2
      tmp0 = 80*t1**2*vcf**2+24*ro**2*t1*vcf**2-8*ro*t1*vcf**2-48*t1*vcf
     1   **2-4*ro**3*vcf**2+8*ro**2*vcf**2+8*ro**2*t1**4*vca*vcf+8*ro*t1
     2   **4*vca*vcf-56*t1**4*vca*vcf-16*ro**2*t1**3*vca*vcf-16*ro*t1**3
     3   *vca*vcf+112*t1**3*vca*vcf-4*ro**3*t1**2*vca*vcf+48*ro**2*t1**2
     4   *vca*vcf-120*t1**2*vca*vcf+4*ro**3*t1*vca*vcf-40*ro**2*t1*vca*v
     5   cf+8*ro*t1*vca*vcf+64*t1*vca*vcf+4*ro**3*vca*vcf-8*ro**2*vca*vc
     6   f-64*t1**6*vca**2+192*t1**5*vca**2-4*ro**2*t1**4*vca**2-5*ro*t1
     7   **4*vca**2-226*t1**4*vca**2+8*ro**2*t1**3*vca**2+10*ro*t1**3*vc
     8   a**2+132*t1**3*vca**2+2*ro**3*t1**2*vca**2-18*ro**2*t1**2*vca**
     9   2-3*ro*t1**2*vca**2-22*t1**2*vca**2-2*ro**3*t1*vca**2+14*ro**2*
     :   t1*vca**2-2*ro*t1*vca**2-12*t1*vca**2-ro**3*vca**2+2*ro**2*vca*
     ;   *2+tmp0
      tmp0 = pi*tmp0*vdmp*vtf*zg**6/(b*s*(t1-1)**2*t1**2*vda)
      dd = tmp0+dd
      dd = dd-pi*(16*ro*t1**4-48*t1**4-32*ro*t1**3+96*t1**3-7*ro**2*t1**
     1   2+49*ro*t1**2-98*t1**2+7*ro**2*t1-33*ro*t1+50*t1+ro**3-3*ro**2+
     2   6*ro-8)*vca**2*vdmp*vtf*zg**6/(b**3*s*(t1-1)*t1*vda)
      dd = dd-pi*(2*t1-1)**2*(4*ro*t1**2+8*t1**2-4*ro*t1-8*t1+3*ro**2-8*
     1   ro+8)*vca**2*vdmp*vtf*zg**6/(b**5*s*(t1-1)*t1*vda* 2.d0)
      tmp0 = 2*vcf+2*t1**2*vca-2*t1*vca-vca
      tmp0 = -pi*(ro-2)*(8*t1**4-16*t1**3+4*ro*t1**2+12*t1**2-4*ro*t1-4*
     1   t1+ro**2)*tmp0*(2*vcf-vca)*vdmb*vtf*zg**6/(b*s*(t1-1)**2*t1**2*
     2   vda)
      dd = tmp0+dd
      tmp0 = 128*ro*t1**4*vcf**2-224*t1**4*vcf**2-256*ro*t1**3*vcf**2+44
     1   8*t1**3*vcf**2+40*ro**2*t1**2*vcf**2+72*ro*t1**2*vcf**2-304*t1*
     2   *2*vcf**2-40*ro**2*t1*vcf**2+56*ro*t1*vcf**2+80*t1*vcf**2+12*ro
     3   **3*vcf**2-24*ro**2*vcf**2+128*ro*t1**6*vca*vcf
      tmp0 = -256*t1**6*vca*vcf-384*ro*t1**5*vca*vcf+768*t1**5*vca*vcf+7
     1   2*ro**2*t1**4*vca*vcf+200*ro*t1**4*vca*vcf-696*t1**4*vca*vcf-14
     2   4*ro**2*t1**3*vca*vcf+240*ro*t1**3*vca*vcf+112*t1**3*vca*vcf+12
     3   *ro**3*t1**2*vca*vcf+16*ro**2*t1**2*vca*vcf-128*ro*t1**2*vca*vc
     4   f+136*t1**2*vca*vcf-12*ro**3*t1*vca*vcf+56*ro**2*t1*vca*vcf-56*
     5   ro*t1*vca*vcf-64*t1*vca*vcf-12*ro**3*vca*vcf+24*ro**2*vca*vcf-6
     6   4*ro*t1**6*vca**2+64*t1**6*vca**2+192*ro*t1**5*vca**2-192*t1**5
     7   *vca**2-36*ro**2*t1**4*vca**2-133*ro*t1**4*vca**2+158*t1**4*vca
     8   **2+72*ro**2*t1**3*vca**2-54*ro*t1**3*vca**2+4*t1**3*vca**2-6*r
     9   o**3*t1**2*vca**2-18*ro**2*t1**2*vca**2+45*ro*t1**2*vca**2-54*t
     :   1**2*vca**2+6*ro**3*t1*vca**2-18*ro**2*t1*vca**2+14*ro*t1*vca**
     ;   2+20*t1*vca**2+3*ro**3*vca**2-6*ro**2*vca**2+tmp0
      tmp0 = pi**3*tmp0*vtf*zg**6/(b*s*(t1-1)**2*t1**2*vda* 12.d0)
      dd = tmp0+dd
      dd = pi*(ro-1)*(2*t1-1)**2*vca*(2*vcf-vca)*vtf*zg**6/(b**2*s*(t1-1
     1   )*t1*vda)+dd
      dd = dd-pi**3*(16*ro*t1**4-48*t1**4-32*ro*t1**3+96*t1**3-7*ro**2*t
     1   1**2+49*ro*t1**2-98*t1**2+7*ro**2*t1-33*ro*t1+50*t1+ro**3-3*ro*
     2   *2+6*ro-8)*vca**2*vtf*zg**6/(b**3*s*(t1-1)*t1*vda* 12.d0)
      dd = pi*(ro-1)*(2*t1-1)**4*vca**2*vtf*zg**6/(b**4*s*(t1-1)*t1*vda)
     1   +dd
      dd = dd-pi**3*(2*t1-1)**2*(4*ro*t1**2+8*t1**2-4*ro*t1-8*t1+3*ro**2
     1   -8*ro+8)*vca**2*vtf*zg**6/(b**5*s*(t1-1)*t1*vda* 24.d0)
      tmp0 = 64*pi**2*t1**5*vcf+192*t1**5*vcf+16*pi**2*ro*t1**4*vcf-192*
     1   pi**2*t1**4*vcf+192*t1**4*vcf+64*pi**2*ro*t1**3*vcf
      tmp0 = 240*ro*t1**3*vcf+192*pi**2*t1**3*vcf-960*t1**3*vcf+24*pi**2
     1   *ro**2*t1**2*vcf-24*ro**2*t1**2*vcf-176*pi**2*ro*t1**2*vcf+288*
     2   ro*t1**2*vcf-64*pi**2*t1**2*vcf-192*t1**2*vcf+192*ro**2*t1*vcf+
     3   96*pi**2*ro*t1*vcf-1296*ro*t1*vcf+1536*t1*vcf+6*pi**2*ro**3*vcf
     4   -24*pi**2*ro**2*vcf-168*ro**2*vcf+768*ro*vcf-768*vcf-32*pi**2*t
     5   1**5*vca-8*pi**2*ro*t1**4*vca+96*pi**2*t1**4*vca-1152*t1**4*vca
     6   -32*pi**2*ro*t1**3*vca-288*ro*t1**3*vca-96*pi**2*t1**3*vca+3456
     7   *t1**3*vca-12*pi**2*ro**2*t1**2*vca+88*pi**2*ro*t1**2*vca+288*r
     8   o*t1**2*vca+32*pi**2*t1**2*vca-3456*t1**2*vca-72*ro**2*t1*vca-4
     9   8*pi**2*ro*t1*vca+288*ro*t1*vca+1152*t1*vca-3*pi**2*ro**3*vca+1
     :   2*pi**2*ro**2*vca+72*ro**2*vca-288*ro*vca+tmp0
      tmp0 = -auinv*pi*tmp0*(2*vcf+t1*vca-vca)*vtf*zg**6/((t1-1)**3*t1*v
     1   da* 48.d0)
      dd = tmp0+dd
      tmp0 = 64*pi**2*t1**5*vcf+192*t1**5*vcf-16*pi**2*ro*t1**4*vcf-128*
     1   pi**2*t1**4*vcf-1152*t1**4*vcf+128*pi**2*ro*t1**3*vcf+240*ro*t1
     2   **3*vcf+64*pi**2*t1**3*vcf+1728*t1**3*vcf-24*pi**2*ro**2*t1**2*
     3   vcf+24*ro**2*t1**2*vcf-112*pi**2*ro*t1**2*vcf-1008*ro*t1**2*vcf
     4   +48*pi**2*ro**2*t1*vcf+144*ro**2*t1*vcf-6*pi**2*ro**3*vcf-32*pi
     5   **2*t1**5*vca+8*pi**2*ro*t1**4*vca+64*pi**2*t1**4*vca+1152*t1**
     6   4*vca-64*pi**2*ro*t1**3*vca-288*ro*t1**3*vca-32*pi**2*t1**3*vca
     7   -1152*t1**3*vca+12*pi**2*ro**2*t1**2*vca+56*pi**2*ro*t1**2*vca+
     8   576*ro*t1**2*vca-24*pi**2*ro**2*t1*vca-72*ro**2*t1*vca+3*pi**2*
     9   ro**3*vca
      tmp0 = atinv*pi*tmp0*(2*vcf-t1*vca)*vtf*zg**6/((t1-1)*t1**3*vda*
     1   48.d0)
      dd = tmp0+dd
      tmp0 = 256*nlf*t1**6*vcf*vtf-768*nlf*t1**5*vcf*vtf+256*nlf*ro*t1**
     1   4*vcf*vtf+768*nlf*t1**4*vcf*vtf-512*nlf*ro*t1**3*vcf*vtf-256*nl
     2   f*t1**3*vcf*vtf+64*nlf*ro**2*t1**2*vcf*vtf+256*nlf*ro*t1**2*vcf
     3   *vtf
      tmp0 = -64*nlf*ro**2*t1*vcf*vtf+256*nlf*t1**8*vca*vtf-1024*nlf*t1*
     1   *7*vca*vtf-24*pi**2*ro**2*t1**6*vca*vtf+192*ro**2*t1**6*vca*vtf
     2   +288*nlf*ro*t1**6*vca*vtf+32*ro*t1**6*vca*vtf+1536*nlf*t1**6*vc
     3   a*vtf+72*pi**2*ro**2*t1**5*vca*vtf-576*ro**2*t1**5*vca*vtf-864*
     4   nlf*ro*t1**5*vca*vtf-96*ro*t1**5*vca*vtf-1024*nlf*t1**5*vca*vtf
     5   -78*pi**2*ro**2*t1**4*vca*vtf+64*nlf*ro**2*t1**4*vca*vtf+624*ro
     6   **2*t1**4*vca*vtf+872*nlf*ro*t1**4*vca*vtf+104*ro*t1**4*vca*vtf
     7   +256*nlf*t1**4*vca*vtf+36*pi**2*ro**2*t1**3*vca*vtf-128*nlf*ro*
     8   *2*t1**3*vca*vtf-288*ro**2*t1**3*vca*vtf-304*nlf*ro*t1**3*vca*v
     9   tf-48*ro*t1**3*vca*vtf-6*pi**2*ro**2*t1**2*vca*vtf+64*nlf*ro**2
     :   *t1**2*vca*vtf+48*ro**2*t1**2*vca*vtf+8*nlf*ro*t1**2*vca*vtf+8*
     ;   ro*t1**2*vca*vtf-96*pi**2*t1**6*vcf**2-1152*t1**6*vcf**2+288*pi
     <   **2*t1**5*vcf**2+3456*t1**5*vcf**2-8*pi**2*ro**2*t1**4*vcf**2-9
     =   6*pi**2*ro*t1**4*vcf**2-576*ro*t1**4*vcf**2-368*pi**2*t1**4*vcf
     >   **2-4992*t1**4*vcf**2+16*pi**2*ro**2*t1**3*vcf**2+tmp0
      tmp0 = 192*pi**2*ro*t1**3*vcf**2+1152*ro*t1**3*vcf**2+256*pi**2*t1
     1   **3*vcf**2+4224*t1**3*vcf**2-24*pi**2*ro**2*t1**2*vcf**2-192*ro
     2   **2*t1**2*vcf**2-144*pi**2*ro*t1**2*vcf**2-768*ro*t1**2*vcf**2-
     3   80*pi**2*t1**2*vcf**2-1536*t1**2*vcf**2+16*pi**2*ro**2*t1*vcf**
     4   2+192*ro**2*t1*vcf**2+48*pi**2*ro*t1*vcf**2+192*ro*t1*vcf**2-8*
     5   pi**2*ro**2*vcf**2-1152*t1**8*vca*vcf+4608*t1**7*vca*vcf+16*pi*
     6   *2*ro*t1**6*vca*vcf-768*ro*t1**6*vca*vcf+264*pi**2*t1**6*vca*vc
     7   f-8864*t1**6*vca*vcf-48*pi**2*ro*t1**5*vca*vcf+2304*ro*t1**5*vc
     8   a*vcf-792*pi**2*t1**5*vca*vcf+10464*t1**5*vca*vcf+42*pi**2*ro**
     9   2*t1**4*vca*vcf-192*ro**2*t1**4*vca*vcf+124*pi**2*ro*t1**4*vca*
     :   vcf-3920*ro*t1**4*vca*vcf+968*pi**2*t1**4*vca*vcf-6264*t1**4*vc
     ;   a*vcf-84*pi**2*ro**2*t1**3*vca*vcf+384*ro**2*t1**3*vca*vcf-168*
     <   pi**2*ro*t1**3*vca*vcf+4000*ro*t1**3*vca*vcf-616*pi**2*t1**3*vc
     =   a*vcf+464*t1**3*vca*vcf+78*pi**2*ro**2*t1**2*vca*vcf-512*ro**2*
     >   t1**2*vca*vcf+tmp0
      tmp0 = 100*pi**2*ro*t1**2*vca*vcf-1520*ro*t1**2*vca*vcf+176*pi**2*
     1   t1**2*vca*vcf+744*t1**2*vca*vcf-36*pi**2*ro**2*t1*vca*vcf+320*r
     2   o**2*t1*vca*vcf-24*pi**2*ro*t1*vca*vcf-96*ro*t1*vca*vcf+4*pi**2
     3   *ro**2*vca*vcf+160*pi**2*t1**8*vca**2-1088*t1**8*vca**2-640*pi*
     4   *2*t1**7*vca**2+4352*t1**7*vca**2+64*pi**2*ro*t1**6*vca**2-1296
     5   *ro*t1**6*vca**2+1004*pi**2*t1**6*vca**2-5952*t1**6*vca**2-192*
     6   pi**2*ro*t1**5*vca**2+3888*ro*t1**5*vca**2-772*pi**2*t1**5*vca*
     7   *2+2624*t1**5*vca**2+pi**2*ro**2*t1**4*vca**2-320*ro**2*t1**4*v
     8   ca**2+214*pi**2*ro*t1**4*vca**2-3772*ro*t1**4*vca**2+260*pi**2*
     9   t1**4*vca**2+640*t1**4*vca**2-2*pi**2*ro**2*t1**3*vca**2+640*ro
     :   **2*t1**3*vca**2-108*pi**2*ro*t1**3*vca**2+1064*ro*t1**3*vca**2
     ;   +20*pi**2*t1**3*vca**2-576*t1**3*vca**2-3*pi**2*ro**2*t1**2*vca
     <   **2-320*ro**2*t1**2*vca**2+22*pi**2*ro*t1**2*vca**2+116*ro*t1**
     =   2*vca**2-32*pi**2*t1**2*vca**2+4*pi**2*ro**2*t1*vca**2+tmp0
      tmp0 = -pi*tmp0*vtf*zg**6/(s*(t1-1)**3*t1**3*vda* 12.d0)
      dd = tmp0+dd
      ggqq2 = dd/(pi* 4.d0)+ss
      if(schhad1.eq.'DI')then
           one = 1
           xk = xkdgg(nl) + 2*xkpgg(one,nl)*lb + 2*xklgg(one,nl)*lb**2
           xk = xk*16*pi**2
           ggqq2 = ggqq2 - xk*ggborn(s,t,m2)/(8*pi**2)
      elseif(schhad1.ne.'MS')then
           write(6,*)'scheme ',schhad1,'not known'
           stop
      endif
      if(schhad2.eq.'DI')then
           one = 1
           xk = xkdgg(nl) + 2*xkpgg(one,nl)*lb + 2*xklgg(one,nl)*lb**2
           xk = xk*16*pi**2
           ggqq2 = ggqq2 - xk*ggborn(s,t,m2)/(8*pi**2)
      elseif(schhad2.ne.'MS')then
           write(6,*)'scheme ',schhad2,'not known'
           stop
      endif
      return
      end

       FUNCTION FQQ(S,X,Y,XM2,Q1Q,Q2Q,W1H,W2H,CTH2)
c
c      q(p1) + q_barra(p2) --> Q(k1) + Q_barra(k2) + g(k)
c
c      d sigma_qq (f) = N g^6 / (64 pi^2 s) beta_x d costh1 d th2 dy dx
c                       1/(1-x)_rho ( 1/(1-y)_+ + 1/(1+y)_+ ) FQQ
c
c           N = 1 / (4 pi)^2
c
       IMPLICIT REAL* 8(A-H,O-Z)
       REAL* 8 N
       TINY = .1D-10
       V = 8
       N = 3
       N2 = N*N
       CF = 4.D0/3.D0
       CA = 3
       TK = - (1-X)*(1-Y)*S/2
       UK = - (1-X)*(1+Y)*S/2
       IF(1-X.LT.TINY)THEN
          Q2C=-S-Q2Q
          P13 = -Q1Q/2
          P23 = -Q2C/2
          P12 = S/2
          BORN = V/(2*N2)*( (P13**2+P23**2)/P12**2 + XM2/P12 )
C
C         Fattori iconali moltiplicati per 4*tk*uk
C         P1.K = -TK/2
C         P2.K = -UK/2
C         P3.K = W1/2
C         P4.K = W2/2
C
          P14 = P23
          P24 = P13
C
          E13 = 16*(1+Y)*P13/W1H
          E14 = 16*(1+Y)*P14/W2H
          E23 = 16*(1-Y)*P23/W1H
          E24 = 16*(1-Y)*P24/W2H
          E12 = 16*P12
          E33 = 16*(1-Y)*(1+Y)* XM2/W1H**2
          E44 = 16*(1-Y)*(1+Y)* XM2/W2H**2
          E34 = 16*(1-Y)*(1+Y)* (S/2-XM2)/(W1H*W2H)
          SUM = BORN*( CF*(2*E13+2*E24-E33-E44)
     #        +           (2*E14+2*E23-E13-E24-E12-E34)/N )
          FQQ = 1/(2*S)*SUM
       ELSEIF(1-Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = - X*Q1Q/2
          P23 = -Q2C/2
          P12 = S*X/2
          BORN = V/(2*N2)*( (P13**2+P23**2)/P12**2 + XM2/P12 )
          SUM = - BORN*(8*UK)*CF*(1+X**2)/(1-X)
          FQQ = 1/(2*X*S)*SUM
       ELSEIF(1+Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = -Q1Q/2
          P23 = -X*Q2C/2
          P12 = S*X/2
          BORN = V/(2*N2)*( (P13**2+P23**2)/P12**2 + XM2/P12 )
          SUM = - BORN*(8*TK)*CF*(1+X**2)/(1-X)
          FQQ = 1/(2*X*S)*SUM
       ELSE
       S2 = S+TK+UK
       Q1C=-S-TK-Q1Q
       Q2C=-S-UK-Q2Q
       W1 =Q2Q-Q1Q-TK
       W2 =Q1Q-Q2Q-UK
C
       P12 = S/2
       P13 = Q1Q/2
       P14 = Q1C/2
       P15 = TK/2
       P23 = Q2C/2
       P24 = Q2Q/2
       P25 = UK/2
       P34 = (S2-2*XM2)/2
       P35 = W1/2
       P45 = W2/2
      ans = Q4G1M(XM2,P34,P24,P14,P45,P23,P13,P35,P12,P25,P15)
      ANS = 4*TK*UK*ANS/(2*S*4*9)
      FQQ = ANS
      ENDIF
      RETURN
      END

       FUNCTION FQG(S,X,Y,XM2,Q1Q,Q2Q,W1H,W2H,CTH2)
c
c      q(p1) + g(p2) --> Q(k1) + Q_barra(k2) + g(k)
c
c      d sigma_qg (f) = N g^6 / (64 pi^2 s) beta_x d costh1 d th2 dy dx
c                       1/(1-x)_rho ( 1/(1-y)_+ + 1/(1+y)_+ ) FQG
c
c           N = 1 / (4 pi)^2
c
       IMPLICIT REAL* 8(A-H,O-Z)
       REAL* 8 N
       TINY = .1D-10
       V = 8
       N = 3
       N2 = N*N
       TF = .5D0
       CF = 4.D0/3.D0
       CA = 3
       TK = - (1-X)*(1-Y)*S/2
       UK = - (1-X)*(1+Y)*S/2
       IF(1-X.LT.TINY)THEN
          FQG = 0
       ELSEIF(1-Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = - X*Q1Q/2
          P23 = -Q2C/2
          P12 = S*X/2
          BX = SQRT(1-4*XM2/(S*X))
          CTH1 = (P23-P13)/P12/BX
          AZIDEP =
     #    -512*BX**2*(CTH1-1)*(CTH1+1)*V*(2*V+BX**2*CTH1**2*N2-N2)
     #    *(X-1)**2*XM2/((BX*CTH1-1)**2*(BX*CTH1+1)**2*X**2)
          AZIDEP = AZIDEP * (2*CTH2**2-1)/2 /(4*64)
          AZIDEP = AZIDEP * CF / CA
          BORN = 1/(2*V*N)*(V/(P13*P23)-2*N**2/P12**2)*
     #           (P13**2+P23**2+2*XM2*P12-(XM2*P12)**2/(P13*P23))
          SUM = - BORN*(8*UK)*CF*(1+(1-X)**2)/X
     #    + AZIDEP
          FQG = 1/(2*X*S)*SUM
       ELSEIF(1+Y.LT.TINY)THEN
          Q2C = -S-UK-Q2Q
          P13 = -Q1Q/2
          P23 = -X*Q2C/2
          P12 = S*X/2
          BORN = V/(2*N2)*( (P13**2+P23**2)/P12**2 + XM2/P12 )
          SUM = - BORN*(8*TK)*TF*(X**2+(1-X)**2)
          FQG = 1/(2*X*S)*SUM
       ELSE
       S2 = S+TK+UK
       Q1C=-S-TK-Q1Q
       Q2C=-S-UK-Q2Q
       W1 =Q2Q-Q1Q-TK
       W2 =Q1Q-Q2Q-UK
C
       P12 = S/2
       P13 = Q1Q/2
       P14 = Q1C/2
       P15 = TK/2
       P23 = Q2C/2
       P24 = Q2Q/2
       P25 = UK/2
       P34 = (S2-2*XM2)/2
       P35 = W1/2
       P45 = W2/2
      ANS = Q4G1M(XM2,P34,P45,P14,P24,P35,P13,P23,P15,P25,P12)
      ANS = - 4*TK*UK*ANS/(2*S*4*3*8)
      FQG = ANS
      ENDIF
      RETURN
      END


c
c       This function subroutine, Q4G1M, calculates the
c       invariant matrix element squared for the process:
c
c       Q(-p1) + Qbar(-p2) --> q(p3) + qbar(p4) + g(p5)
C
c       summed over initial and final spins and colors including
c       masses for the incoming quark-antiquark pair.
c       The final state quark-antiquark pair are massless.
c       No averaging is performed for initial spins or colors.
c
c
c       1.      p1...p5 are the outgoing four momenta of the
c               partons, satisfying
c                       p1 + p2 + p3 + p4 + p5 = 0
c
c       2.      xm2 is the mass^2 of the heavy Quark and AntiQuark lines
c               with
c                       p1**2 = p2**2 = xm2
c               where
c                       pi**2 = pi(4)**2-pi(1)**2-pi(2)**2-pi(3)**2
c
c Abbiamo:
c             p12 = p1(4)*p2(4)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
c             etc...
c
c Per il processo
c       Q(-p1) + Qbar(-p2) --> q(p3) + qbar(p4) + g(p5)
c chiamare
c       Q4G1M(XM2,P12,P13,P14,P15,P23,P24,P25,P34,P35,P45)*g**6/4/N**2
c
c Per
c       q(-p1) + qbar(-p2) --> Q(p3) + Qbar(p4) + g(p5)
c che equivarrebbe a
c       Q(-p4)+Qbar(-p3) --> q(p2)+qbar(p1)+g(p5)
c chiamare
c       Q4G1M(XM2,P34,P24,P14,P45,P23,P13,P35,P12,P25,P15)*g**6/4/N**2
c
c

      FUNCTION  Q4G1M(XM2,P12,P13,P14,P15,P23,P24,P25,P34,P35,P45)
      REAL*8  DL1, DL2, DL3, DL4
      REAL*8  P12, P13, P14, P15, P23, P24, P25, P34, P35, P45
      REAL*8  Q4G1M, XM2, RES, S, XN, XV
      PARAMETER         (XN = 3.D0)

      XV  = XN**2 - 1
      S = 2*(P12+XM2)

      DL1 = P13/P25-2*P35/S
      DL2 = P14/P25-2*P45/S
      DL3 = P23/P15-2*P35/S
      DL4 = P24/P15-2*P45/S


      RES =
     & +(P13**2+P23**2+P14**2+P24**2+XM2*(P12+P34+XM2))/2/S/P34
     & *(4*XV**2/XN*(P13/P15/P35+P24/P25/P45)
     & +4*XV/XN*(2*P14/P15/P45+2*P23/P25/P35
     & -P13/P15/P35-P24/P25/P45-P12/P15/P25-P34/P35/P45) )

      RES = RES
     & -XV*(XN**2-4)/XN
     & *2*XM2/S/P34*((P13-P14)/P25-(P23-P24)/P15)
     & +4*XV**2/XN*XM2*( (P35**2+P45**2)/P35/P45/S**2
     & -0.5*(1/P15+1/P25+1/P35+1/P45)/S
     & -0.25*(1/P15+1/P25+XM2/P15**2+XM2/P25**2+4/S)/P34
     & -(DL1**2+DL2**2+DL3**2+DL4**2)/4/P34**2)
     & -2*XV/XN
     & *XM2/S/P34*(1+2*P34/S+XM2/P15+XM2/P25+(P35**2+P45**2)/P15/P25
     & +(P13-P14)*(DL1-DL2)/P34+(P23-P24)*(DL3-DL4)/P34 )

      Q4G1M = RES


      RETURN
      END

      function qqqq2(s,t,m2,mu2,nl)
c
c     q(p1) + q_bar(p2) --> Q(k1) + Q_bar(k2)
c
c     t = (p1-k1)^2
c
      implicit double precision (a-z)
      integer nl
      character * 2 schhad1,schhad2
      common/betfac/betfac,delta
      common/scheme/schhad1,schhad2
      data pi/3.141 592 653 589 793/
      ro = 4*m2/s
      t1 = -t/s
      zg = 1
      vn = 3
      vca = 3
      vcf = 4
      vcf = vcf/ 3.d0
      vtf = 1
      vtf = vtf/ 2.d0
      vda = 8
      vbf = -(vn**2-4)*(vn**2-1)/( 16.d0*vn)
      nlf = nl
      t2 = 1-t1
      b = dsqrt(1-ro)
      lp = (b+1)/ 2.d0
      lm = (1-b)/ 2.d0
      at = s*t1
      aw = s*t2
      vlm2 = dlog(m2/mu2)
      vltm = dlog(at/m2)
      vlpm = dlog(lp/lm)
      vlsm = dlog(s/m2)
      vlsmu = dlog(s/mu2)
      vlwm = dlog(aw/m2)
      vlbl = dlog(b/lm)
      vdw = ddilog((aw-m2)/aw)-vlwm**2/ 2.d0
      vdt = ddilog((at-m2)/at)-vltm**2/ 2.d0
      vdmp = ddilog(-lm/lp)
      vdmb = vlbl**2/ 2.d0+ddilog(-lm/b)
      auinv = 1/(m2-aw)
      atinv = 1/(m2-at)
      softt1 = ddilog(1-2*t1/(b+1))+ddilog(1-2*t1/(1-b))+log(2*t1/(1-b))
     1   *log(2*t1/(b+1))
      softt2 = ddilog(1-2*t2/(b+1))+ddilog(1-2*t2/(1-b))+log(2*t2/(1-b))
     1   *log(2*t2/(b+1))
      softb = ddilog(2*b/(b+1))-ddilog(-2*b/(1-b))
      lt1 = log(t1)
      lt2 = log(t2)
      lb = log(b*betfac)
      ss = lb*(4*t1**2-4*t1+ro+2)*vlsmu*(vn-1)**2*(vn+1)**2*zg**6/(s*vn*
     1   *3)
      ss = 3*(4*t1**2-4*t1+ro+2)*vlsmu*(vn-1)**2*(vn+1)**2*zg**6/( 8.d0*
     1   s*vn**3)+ss
      ss = (4*t1**2-4*t1+ro+2)*vlsm**2*(vn-1)*(vn+1)*(3*vn**2-1)*zg**6/(
     1    8.d0*s*vn**3)+ss
      ss = (ro-2)*(4*t1**2-4*t1+ro+2)*vlpm*vlsm*(vn-1)*(vn+1)*zg**6/( 8.
     1   d0*b*s*vn**3)+ss
      ss = lt2*(4*t1**2-4*t1+ro+2)*vlsm*(vn-1)*(vn+1)*zg**6/(s*vn**3)+ss
      ss = lt1*(4*t1**2-4*t1+ro+2)*vlsm*(vn-1)*(vn+1)*(vn**2-2)*zg**6/(
     1   2.d0*s*vn**3)+ss
      ss = lb*(4*t1**2-4*t1+ro+2)*vlsm*(vn-1)*(vn+1)*zg**6/(s*vn)+ss
      ss = ss-vlsm*(vn-1)*(vn+1)*(20*t1**2*vn**2-20*t1*vn**2+5*ro*vn**2+
     1   2*vn**2-20*t1**2+20*t1-5*ro-6)*zg**6/( 8.d0*s*vn**3)
      ss = (4*t1**2-4*t1+ro+2)*vlpm**2*(vn-1)*(vn+1)*zg**6/( 8.d0*s*vn)+
     1   ss
      ss = lb*(ro-2)*(4*t1**2-4*t1+ro+2)*vlpm*(vn-1)*(vn+1)*zg**6/( 2.d0
     1   *b*s*vn**3)+ss
      ss = vlpm*(vn-1)*(vn+1)*(4*t1**2*vn**2-4*t1*vn**2+ro*vn**2+2*vn**2
     1   -4*t1**2+4*t1-4)*zg**6/( 4.d0*b*s*vn**3)+ss
      ss = softt2*(4*t1**2-4*t1+ro+2)*(vn-1)*(vn+1)*zg**6/(s*vn**3)+ss
      ss = softt1*(4*t1**2-4*t1+ro+2)*(vn-1)*(vn+1)*(vn**2-2)*zg**6/( 2.
     1   d0*s*vn**3)+ss
      ss = ss-(ro-2)*softb*(4*t1**2-4*t1+ro+2)*(vn-1)*(vn+1)*zg**6/( 8.d
     1   0*b*s*vn**3)
      ss = ss-pi**2*(4*t1**2-4*t1+ro+2)*(vn-1)**2*(vn+1)**2*zg**6/( 12.d
     1   0*s*vn**3)
      ss = 4*lb*lt2*(4*t1**2-4*t1+ro+2)*(vn-1)*(vn+1)*zg**6/(s*vn**3)+ss
      ss = 2*lt2*(vn-1)*(vn+1)*zg**6/(s*vn**3)+ss
      ss = 2*lb*lt1*(4*t1**2-4*t1+ro+2)*(vn-1)*(vn+1)*(vn**2-2)*zg**6/(s
     1   *vn**3)+ss
      ss = lt1*(vn-1)*(vn+1)*(vn**2-2)*zg**6/(s*vn**3)+ss
      ss = 2*lb**2*(4*t1**2-4*t1+ro+2)*(vn-1)**2*(vn+1)**2*zg**6/(s*vn**
     1   3)+ss
      ss = ss-lb*(4*t1**2-4*t1+ro+2)*(vn-1)**2*(vn+1)**2*zg**6/(s*vn**3)
      ss = ss-5*(vn-1)**2*(vn+1)**2*zg**6/( 4.d0*s*vn**3)
      dd = -2*pi*(8*t1**2-12*t1+ro+6)*vcf**2*vlsm*vlwm*(vca*vda*vtf**2+4
     1   *vbf)*zg**6/(s*vda**2*vtf**2)
      dd = 2*auinv*pi*(2*t1**2-2*t1+ro)*vcf**2*vlwm*(vca*vda*vtf**2+4*vb
     1   f)*zg**6/(vda**2*vtf**2)+dd
      dd = dd-8*pi*vcf**2*vlwm*(vca*vda*vtf**2+4*vbf)*zg**6/(s*vda**2*vt
     1   f**2)
      dd = dd-2*pi*(8*t1**2-4*t1+ro+2)*vcf**2*vlsm*vltm*(vca*vda*vtf**2-
     1   4*vbf)*zg**6/(s*vda**2*vtf**2)
      dd = 2*atinv*pi*(2*t1**2-2*t1+ro)*vcf**2*vltm*(vca*vda*vtf**2-4*vb
     1   f)*zg**6/(vda**2*vtf**2)+dd
      dd = dd-8*pi*vcf**2*vltm*(vca*vda*vtf**2-4*vbf)*zg**6/(s*vda**2*vt
     1   f**2)
      dd = dd-pi*vcf**2*vlsm**2*(16*t1**2*vcf*vda*vtf**2-16*t1*vcf*vda*v
     1   tf**2+4*ro*vcf*vda*vtf**2+8*vcf*vda*vtf**2-16*t1**2*vca*vda*vtf
     2   **2+16*t1*vca*vda*vtf**2-3*ro*vca*vda*vtf**2-8*vca*vda*vtf**2+1
     3   6*t1*vbf-8*vbf)*zg**6/(s*vda**2*vtf**2)
      dd = dd-4*pi*(2*t1-1)*vcf**2*vlsm*(2*ro*t1*vca*vda*vtf**2-6*t1*vca
     1   *vda*vtf**2-ro*vca*vda*vtf**2+5*vca*vda*vtf**2+4*vbf)*zg**6/(b*
     2   *2*s*vda**2*vtf**2)
      dd = dd-2*pi*(12*t1**2+8*ro*t1-20*t1-3*ro+6)*vca*vcf**2*vlsm*zg**6
     1   /(b**4*s*vda)
      dd = 4*pi*vcf**2*vlsm*(16*nlf*t1**2*vtf-16*nlf*t1*vtf+4*nlf*ro*vtf
     1   +8*nlf*vtf+36*t1**2*vcf-36*t1*vcf+9*ro*vcf+6*vcf-44*t1**2*vca+4
     2   4*t1*vca-8*ro*vca-16*vca)*zg**6/( 3.d0*s*vda)+dd
      dd = dd-pi*vcf**2*vlpm**2*(8*ro*t1**2*vcf*vda*vtf**2-16*t1**2*vcf*
     1   vda*vtf**2-8*ro*t1*vcf*vda*vtf**2+16*t1*vcf*vda*vtf**2+2*ro**2*
     2   vcf*vda*vtf**2-8*vcf*vda*vtf**2-4*ro*t1**2*vca*vda*vtf**2-8*t1*
     3   *2*vca*vda*vtf**2+2*ro*t1*vca*vda*vtf**2+14*t1*vca*vda*vtf**2-r
     4   o**2*vca*vda*vtf**2-6*vca*vda*vtf**2-8*ro*t1*vbf+24*t1*vbf+4*ro
     5   *vbf-16*vbf)*zg**6/(b*s*vda**2*vtf**2)
      dd = pi*(2*t1+ro-2)*vcf**2*vlpm**2*(2*ro*t1*vca*vda*vtf**2-6*t1*vc
     1   a*vda*vtf**2-ro*vca*vda*vtf**2+5*vca*vda*vtf**2+4*vbf)*zg**6/(b
     2   **3*s*vda**2*vtf**2)+dd
      dd = pi*(4*ro*t1**2+8*t1**2+4*ro*t1-16*t1+3*ro**2-8*ro+8)*vca*vcf*
     1   *2*vlpm**2*zg**6/( 2.d0*b**5*s*vda)+dd
      dd = 4*b*pi*(ro+2)*vcf**2*vlpm*(8*t1**2*vtf-8*t1*vtf+2*ro*vtf+4*vt
     1   f+6*vcf-3*vca)*zg**6/( 3.d0*s*vda)+dd
      dd = dd-2*pi*(8*ro*t1**2-12*t1**2-8*ro*t1+12*t1-ro+2)*vcf**2*(2*vc
     1   f-vca)*vlpm*zg**6/(b*s*vda)
      dd = 4*pi*(4*t1**2-4*t1+ro+2)*vcf**2*vlm2*(4*nlf*vtf-11*vca)*zg**6
     1   /( 3.d0*s*vda)+dd
      dd = 2*pi*(4*t1+ro-2)*vcf**2*vdw*(vca*vda*vtf**2+4*vbf)*zg**6/(s*v
     1   da**2*vtf**2)+dd
      dd = dd-2*pi*(4*t1-ro-2)*vcf**2*vdt*(vca*vda*vtf**2-4*vbf)*zg**6/(
     1   s*vda**2*vtf**2)
      dd = 8*pi*vcf**2*vdmp*(8*t1**2*vca*vda*vtf**2+ro*t1*vca*vda*vtf**2
     1   -11*t1*vca*vda*vtf**2+5*vca*vda*vtf**2+4*ro*t1*vbf-12*t1*vbf-2*
     2   ro*vbf+8*vbf)*zg**6/(b*s*vda**2*vtf**2)+dd
      dd = 4*pi*(2*t1+ro-2)*vcf**2*vdmp*(2*ro*t1*vca*vda*vtf**2-6*t1*vca
     1   *vda*vtf**2-ro*vca*vda*vtf**2+5*vca*vda*vtf**2+4*vbf)*zg**6/(b*
     2   *3*s*vda**2*vtf**2)+dd
      dd = 2*pi*(4*ro*t1**2+8*t1**2+4*ro*t1-16*t1+3*ro**2-8*ro+8)*vca*vc
     1   f**2*vdmp*zg**6/(b**5*s*vda)+dd
      dd = 4*pi*(ro-2)*(4*t1**2-4*t1+ro+2)*vcf**2*(2*vcf-vca)*vdmb*zg**6
     1   /(b*s*vda)+dd
      dd = dd-2*pi**3*vcf**2*(16*ro*t1**2*vcf*vda*vtf**2-32*t1**2*vcf*vd
     1   a*vtf**2-16*ro*t1*vcf*vda*vtf**2+32*t1*vcf*vda*vtf**2+4*ro**2*v
     2   cf*vda*vtf**2-16*vcf*vda*vtf**2-8*ro*t1**2*vca*vda*vtf**2+8*t1*
     3   *2*vca*vda*vtf**2+7*ro*t1*vca*vda*vtf**2-5*t1*vca*vda*vtf**2-2*
     4   ro**2*vca*vda*vtf**2+3*vca*vda*vtf**2-4*ro*t1*vbf+12*t1*vbf+2*r
     5   o*vbf-8*vbf)*zg**6/( 3.d0*b*s*vda**2*vtf**2)
      dd = dd-4*pi*(ro-1)*vcf**2*(2*vcf-vca)*zg**6/(b**2*s*vda)
      dd = pi**3*(2*t1+ro-2)*vcf**2*(2*ro*t1*vca*vda*vtf**2-6*t1*vca*vda
     1   *vtf**2-ro*vca*vda*vtf**2+5*vca*vda*vtf**2+4*vbf)*zg**6/( 3.d0*
     2   b**3*s*vda**2*vtf**2)+dd
      dd = dd-4*pi*(ro-1)*(2*t1-1)**2*vca*vcf**2*zg**6/(b**4*s*vda)
      dd = pi**3*(4*ro*t1**2+8*t1**2+4*ro*t1-16*t1+3*ro**2-8*ro+8)*vca*v
     1   cf**2*zg**6/( 6.d0*b**5*s*vda)+dd
      dd = dd-pi*vcf**2*(192*ro*t1**2*vda*vtf**3+320*nlf*t1**2*vda*vtf**
     1   3+320*t1**2*vda*vtf**3-192*ro*t1*vda*vtf**3-320*nlf*t1*vda*vtf*
     2   *3-320*t1*vda*vtf**3+48*ro**2*vda*vtf**3+80*nlf*ro*vda*vtf**3+1
     3   76*ro*vda*vtf**3+160*nlf*vda*vtf**3+160*vda*vtf**3-192*pi**2*t1
     4   **2*vcf*vda*vtf**2+1728*t1**2*vcf*vda*vtf**2+192*pi**2*t1*vcf*v
     5   da*vtf**2-1728*t1*vcf*vda*vtf**2-48*pi**2*ro*vcf*vda*vtf**2+432
     6   *ro*vcf*vda*vtf**2-96*pi**2*vcf*vda*vtf**2+576*vcf*vda*vtf**2+7
     7   2*pi**2*t1**2*vca*vda*vtf**2-1216*t1**2*vca*vda*vtf**2-72*pi**2
     8   *t1*vca*vda*vtf**2+1216*t1*vca*vda*vtf**2+21*pi**2*ro*vca*vda*v
     9   tf**2-304*ro*vca*vda*vtf**2+36*pi**2*vca*vda*vtf**2-680*vca*vda
     :   *vtf**2+48*pi**2*t1*vbf-24*pi**2*vbf)*zg**6/( 9.d0*s*vda**2*vtf
     ;   **2)
      qqqq2 = ss+dd/( 4.d0*pi)
      if(schhad1.eq.'DI')then
           one = 1
           xk = xkdqq(nl) + 2*xkpqq(one,nl)*lb + 2*xklqq(one,nl)*lb**2
           xk = xk*16*pi**2
           qqqq2 = qqqq2 - xk*qqborn(s,t,m2)/(8*pi**2)
      elseif(schhad1.ne.'MS')then
           write(6,*)'scheme ',schhad1,'not known'
           stop
      endif
      if(schhad2.eq.'DI')then
           one = 1
           xk = xkdqq(nl) + 2*xkpqq(one,nl)*lb + 2*xklqq(one,nl)*lb**2
           xk = xk*16*pi**2
           qqqq2 = qqqq2 - xk*qqborn(s,t,m2)/(8*pi**2)
      elseif(schhad2.ne.'MS')then
           write(6,*)'scheme ',schhad2,'not known'
           stop
      endif
      return
      end



      function qgqq2(s,t,m2,nl)
c
c     q(p1) + g(p2) --> Q(k1) + Q_bar(k2)
c
c     t = (p1-k1)^2
c
      implicit double precision (a-z)
      integer nl
      character * 2 schhad1,schhad2
      common/betfac/betfac,delta
      common/scheme/schhad1,schhad2
      data pi/3.141 592 653 589 793/
      qgqq2 = 0
      if(schhad1.eq.'DI')then
           one = 1
           ro = 4*m2/s
           b  = sqrt(1-ro)
           lb = log(b*betfac)
           xk = xkdgq(nl) + 2*xkpgq(one,nl)*lb + 2*xklgq(one,nl)*lb**2
           xk = xk*16*pi**2
           qgqq2 = qgqq2 - xk*ggborn(s,t,m2)/(8*pi**2)
      elseif(schhad1.ne.'MS')then
           write(6,*)'scheme ',schhad1,'not known'
           stop
      endif
      if(schhad2.eq.'DI')then
           one = 1
           ro = 4*m2/s
           b  = sqrt(1-ro)
           lb = log(b*betfac)
           xk = xkdqg(nl) + 2*xkpqg(one,nl)*lb + 2*xklqg(one,nl)*lb**2
           xk = xk*16*pi**2
           qgqq2 = qgqq2 - xk*qqborn(s,t,m2)/(8*pi**2)
      elseif(schhad2.ne.'MS')then
           write(6,*)'scheme ',schhad2,'not known'
           stop
      endif
      return
      end

c
c Transformation functions from MS_bar scheme to
c scheme of DFLM.
c
C      function xkpqq(x,nl)
C      implicit double precision (a-z)
C      parameter (fot=4/3.d0)
C      integer nl
C      xkpqq = fot*(-1.5d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
C      return
C      end

C      function xkdqq(nl)
C      implicit double precision (a-z)
C      parameter (fot=4/3.d0)
C      integer nl
C      data pi/3.141 592 653 589 793/
C      xkdqq = -fot*(4.5d0 + pi**2/3)
C      return
C      end

C      function xklqq(x,nl)
C      implicit double precision (a-z)
C      parameter (fot=4/3.d0)
C      integer nl
C      xklqq = fot*(1+x**2)
C      return
C      end

C      function xkpqg(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xkpqg = (1-x)*(-(x**2+(1-x)**2)*log(x)+8*x*(1-x)-1)/2
C      return
C      end

C      function xkdqg(nl)
C      implicit double precision (a-z)
C      integer nl
C      xkdqg = 0
C      return
C      end

C      function xklqg(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xklqg = (1-x)*(x**2+(1-x)**2)/2
C      return
C      end

C      function xkdgg(nl)
C      implicit double precision (a-z)
C      integer nl
C      xkdgg = - 2 * nl * xkdqg(nl)
C      return
C      end

C      function xkpgg(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xkpgg = - 2 * nl * xkpqg(x,nl)
C      return
C      end

C      function xklgg(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xklgg = - 2 * nl * xklqg(x,nl)
C      return
C      end

C      function xkdgq(nl)
C      implicit double precision (a-z)
C      integer nl
C      xkdgq = - xkdqq(nl)
C      return
C      end

C      function xkpgq(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xkpgq = - xkpqq(x,nl)
C      return
C      end

C      function xklgq(x,nl)
C      implicit double precision (a-z)
C      integer nl
C      xklgq = - xklqq(x,nl)
C      return
C      end

c
c Fit to heavy quark partonic total production cross section.
c The cross section is:
c
c  A)  g+g --> Q+Qb
c  sigg(ro) = as^2/M^2*(f0gg(ro)+g^2*(f1gg(ro)+fbgg(ro)*log(mu^2/M^2);
c
c  B)  q+g --> Q+Qb
c  siqg(ro) = as^2/M^2*(g^2*(f1qg(ro)+fbqg(ro)*log(mu^2/M^2);
c
c  C)  q+qb --> Q+Qb
c  siqq(ro) = as^2/M^2*(f0qq(ro)+g^2*(f1qq(ro)+fbqq(ro)*log(mu^2/M^2);
c
c  with g^2 = 4*pi*as, ro = 4*M^2/s, s is the partonic CM energy,
c  M is the heavy quark mass.
c  The symbols are: g for gluon, q for light quark,
c  qb for light antiquark, Q for heavy quark, Qb for heavy antiquark.
c
c  All above is in ms_bar. For the DFLM scheme, use f1qqp,f1ggp and f1qgp
c  instead of f1qq,f1qg,f1gg.
c  The functions cqqq and cqqg are the corrections to apply to f1qq and
c  f1qg due to a redefinition of the quark according to the DIS scheme.
c  cgqg and cggg are the corrections to apply to f1qg and f1gg due to
c  a redefinition of the gluon according to the DFLM group.
c
c  The ratio of the fits to the numerically integrated results is
c  well within 1% of accuracy.
c
      function f1qqp(rho)
      implicit real * 8 (a-h,o-z)
      f1qqp = f1qq(rho)+cqqq(rho)
      end

      function f1qgp(rho)
      implicit real * 8 (a-h,o-z)
      f1qgp = f1qg(rho)+cqqg(rho)+cgqg(rho)
      end

      function f1ggp(rho)
      implicit real * 8 (a-h,o-z)
      f1ggp = f1gg(rho)+cggg(rho)
      end

      function f0qq(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      b = sqrt(1-rho)
      f0qq = (pi*b*rho)/27*(2+rho)
      end

      function f0gg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      b = sqrt(1-rho)
      f0gg = (pi*b*rho)/192*(1/b*(rho**2+16*rho+16)*log((1+b)/(1-b))
     #         -28-31*rho)
      end

      function fbqq(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      b = sqrt(1-rho)
      b2 = b*b
      fbqq = 1/(8*pi2)*( (16*pi)/81 * rho*log((1+b)/(1-b))
     #           + f0qq(rho)/9 * (127-6*nl+48*log(rho/(4*b2))) )
      end

      function fbgg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      h1 = log((1+b)/2)**2-log((1-b)/2)**2
     #     +2*ddilog((1+b)/2)-2*ddilog((1-b)/2)
      h2 = ddilog(2*b/(1+b))-ddilog(-2*b/(1-b))
      fbgg = 1/(8*pi2)*( pi/192 * (
     #   2*rho * (59*rho2+198*rho-288) * log((1+b)/(1-b)) +
     #   12*rho * (rho2+16*rho+16)*h2 - 6*rho * (rho2-16*rho+32)*h1
     #   -(4*b)/15*(7449*rho2-3328*rho+724)
     #   ) + 12*f0gg(rho)*log(rho/(4*b2)) )
      end

      function fbqg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      h1 = log((1+b)/2)**2-log((1-b)/2)**2
     #     +2*ddilog((1+b)/2)-2*ddilog((1-b)/2)
      h2 = ddilog(2*b/(1+b))-ddilog(-2*b/(1-b))
      fbqg = 1/(8*pi2) * pi/192 * (
     #   (4*rho)/9 * (14*rho2+27*rho-136) * log((1+b)/(1-b)) -
     #   (32*rho)/3*(2-rho)*h1 - (8*b)/135 * (1319*rho2-3468*rho+724)
     #  )
      end

      function f1qq(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      data a0,a1,a2,a3,a4,a5,a6,a7
     # / 0.180899d0,  0.101949d0, -0.234371d0, -0.0109950d0,
     #  -0.0185575d0, 0.00907527,  0.0160367,   0.00786727/
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      b4 = b2*b2
      b6 = b4*b2
      xlb = log(8*b2)
      xlrh = log(rho)
      f1qq =
     # rho/(72*pi)*( (16*b)/3*xlb**2-(82*b)/3*xlb-pi2/6)
     # +b*rho*( a0 + b2*(a1*xlb+a2) + b4*(a3*xlb+a4) + a5*b6*xlb
     #         + a6*xlrh + a7*xlrh**2 )
     # + 1/(8*pi2) * (nl-4) * f0qq(rho)*(2.d0/3*log(4/rho)-10.d0/9)
      end

      function f1gg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      data a0,a1,a2,a3,a4,a5,a6,a7
     # / 0.108068d0,  -0.114997d0,  0.0428630d0, 0.131429d0,
     #   0.0438768d0, -0.0760996d0,-0.165878d0, -0.158246d0/
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      b4 = b2*b2
      b6 = b4*b2
      xlb = log(8*b2)
      xlrh = log(rho)
      f1gg =
     # 7/(1536*pi)*( 12*b*xlb**2-(366*b)/7*xlb+(11*pi2)/42)
     # +b*( a0 + b2*(a1*xlb+a2) + b4*a3*xlb + rho2*(a4*xlrh+a5*xlrh**2)
     # + rho*(a6*xlrh+a7*xlrh**2) )
     # + rho2/(1024*pi) * (nl-4) * (log((1+b)/(1-b))-2*b)
      end

      function f1qg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      data a0,a1,a2,a3,a4,a5,a6,a7
     # / 0.0110549d0,  -0.426773d0,  -0.00103876d0, 0.450626d0,
     #  -0.227229d0,    0.0472502d0, -0.197611d0,  -0.0529130d0/
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      b4 = b2*b2
      b6 = b4*b2
      xlb = log(b)
      xlrh = log(rho)
      f1qg =
     #  b*( b2*(a0*xlb+a1) + b4*(a2*xlb+a3) + rho2*(a4*xlrh+a5*xlrh**2)
     # + rho*(a6*xlrh+a7*xlrh**2) )
      end

      function cqqq(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      h2 = ddilog(2*b/(1+b))-ddilog(-2*b/(1-b))
      xlrob2 = log(rho/(4*b2))
      cqqq = 1/(8*pi2) * (8.d0/3.d0*f0qq(rho)*
     # (13.d0/18.d0+2*pi2/3-xlrob2**2-25.d0/6*xlrob2)
     # +8.d0/3*(pi*rho)/27*(7.d0/3*b-2*(4+xlrob2 )
     # *log((1+b)/(1-b))-2*h2) )
      end

      function cqqg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
      rho2 = rho*rho
      b = sqrt(1-rho)
      b2 = b*b
      h2 = ddilog(2*b/(1+b))-ddilog(-2*b/(1-b))
      xlrob2 = log(rho/(4*b2))
      xlbp1  = log((1+b)/(1-b))
      cqqg = 1/(8*pi2)*(pi*rho)/27*(
     #  (1-3*rho2/4)*(h2+xlbp1 *xlrob2 ) +
     # b*(13.d0/6*rho-8.d0/3)*xlrob2
     # -3*rho2*xlbp1 +b*(115*rho/9-70.d0/9) )
      end

      function cggg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
c chisquare=  0.2949500883595510
      data a0,a1,a2,a3,a4,a5,a6,a7
     #/-1.2987168754095587D-03, -9.6976043812505796D-03
     #, 2.7173207835442481D-02, -1.6568156284241256D-02
     #,-1.2852283894558339D-02, -3.8494419116341811D-03
     #,-1.0828640684513157D-03,  1.6428466771704755D-03/
      rho2 = rho*rho
      rho3 = rho*rho2
      b = sqrt(1-rho)
      b2 = b*b
      b4 = b2*b2
      xlrh = log(rho)
      cggg = nl*b2*rho*(a0*rho3+a1*b2*rho3+a2*b4*rho3+a3*b2
     #  +a4*b2*xlrh+a5*b2*xlrh**2+a6*b2*xlrh**3+a7*b2*rho*xlrh**2)
      end

      function cgqg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      common/nl/nl
c chisquare=  0.3084278592622398
      data a0,a1,a2,a3,a4,a5,a6,a7
     #/ 3.8596216074859431D-03, 4.9522213636822912D-02
     #,-9.1583774142403212D-02, 2.9094012063027844D-02
     #,-4.1378041486677086D-03, 2.6131355402790908D-03
     #,-7.5785107186006094D-04,-1.3395653263722391D-02/
      rho2 = rho*rho
      rho3 = rho*rho2
      b = sqrt(1-rho)
      b2 = b*b
      b4 = b2*b2
      xl2b = log(2*b)
      xlrh = log(rho)
      cgqg = -f0gg(rho)/(36*pi2)*(4*pi2-24*xl2b**2+66*xl2b-39)
     #     +b2*rho*(a0*rho3+a1*b2*rho3+a2*b4*rho3+a3*b2
     #     +a4*b2*xlrh+a5*b2*xlrh**2+a6*b2*xlrh**3+a7*b2*rho*xlrh**2)
      end

      function bbgg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      parameter (vca=3.d0)
      parameter (vtf=0.5d0)
      common/nl/nl
      b0 = (11*vca-4*vtf*nl)/(12*pi)
      bbgg = fbgg(rho)-b0/(2*pi)*f0gg(rho)
      bbgg = bbgg/2.d0
      end

      function bbqq(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      parameter (vca=3.d0)
      parameter (vtf=0.5d0)
      common/nl/nl
      b0 = (11*vca-4*vtf*nl)/(12*pi)
      bbqq = fbqq(rho)-b0/(2*pi)*f0qq(rho)
      bbqq = bbqq/2.d0
      end

      function bbqg(rho)
      implicit real * 8 (a-h,o-z)
      parameter(pi=3.14159265358979312D0)
      parameter(pi2=pi*pi)
      parameter (vtf=0.5d0)
      b = sqrt(1-rho)
      vlpm = log((1+b)/(1-b))
      tmp = vtf*rho*( vlpm*(12-9*rho**2)+b*(26*rho-32) )/(1296*pi)
      bbqg = fbqg(rho)+tmp
      end
C
C End total cross section package
C
c
c
c Begin of lepton matrix elements (from MadEvent): use for spin correlations
c
c
      function xmadevttb(iborn,jproc,idr,s,tk,uk,xmom)
c Wrapper for MadEvent functions. Inputs are 
c   iborn = 0(born), 1(real)
c   jproc = 1(gg), 2(qq), 3(qg)
c   idr   = 1..4 for direct, charge-conjugated, reflected, and 
c                    charge-conjugated+reflected events respectively
c   s     = parton cm energy squared
c   tk,uk = FNR invariants
c   xmom  = 4-momenta obtained from invar
c Output is the matrix element squared in GeV^-2, times the flux factor,
c times 4*tk*uk in the case of real matrix elements.
c MadEvent routines use the SM parameters as defined in setmepqq();
c for kinematic-dependent couplings, and if simple factorized
c expressions can't be found (which appears to happen only if a Higgs 
c is involved in the reaction), the call to setmepqq() must be included
c in this routine. In the present case, all the Born (real) formulae 
c are proportional to gs^4*e^4 (gs^6*e^4), and this call can be placed 
c somewhere else (and done only once). This function is called by getspincorr, 
c where only idr=1 is considered. Stop if other options are given in input
      implicit none
      integer iborn,jproc,idr
      real*8 xmadevttb,s,tk,uk,xmom(11,4),xfact,tmp(1)
      real*8 pme0(0:3,8),pme1(0:3,9)
      integer ipart,icomp
c Components: MC@NLO conventions   -> 1=px, 2=py, 3=pz, 4=E
c             MadEvent conventions -> 0=E, 1=px, 2=py, 3=pz
      integer mapcomp(0:3)
      data mapcomp/4,1,2,3/
c Labelling conventions for subprocess: 
c MC@NLO   -> x(1)y(2) -> z(3)t(4)[->l+(6)nu(7)b(8)]tb(5)[->l-(9)nub(10)bb(11)]
c MadEvent -> g(1)g(2) -> l+(3)nu(4)b(5) l-(6)nub(7)bb(8)
c MadEvent -> q(1)qb(2)-> l+(3)nu(4)b(5) l-(6)nub(7)bb(8)
c MadEvent -> g(1)g(2) -> l+(3)nu(4)b(5) l-(6)nub(7)bb(8) g(9)
c MadEvent -> q(1)qb(2)-> l+(3)nu(4)b(5) l-(6)nub(7)bb(8) g(9)
c MadEvent -> q(1)g(2) -> l+(3)nu(4)b(5) l-(6)nub(7)bb(8) q(9)
      integer mapdir(9)
c mapping for direct events
      data mapdir/1,2,6,7,8,9,10,11,3/
c
      if(idr.ne.1)then
        write(*,*)'Error in xmadevttb: use only direct events',idr
        stop
      endif
      if(iborn.eq.0)then
        if(jproc.eq.3)then
          write(*,*)'Error #1 in xmadevttb: iborn, jproc=',iborn,jproc
          stop
        endif
        xfact=1.d0
        do ipart=1,8
          do icomp=0,3
            pme0(icomp,ipart)=xmom(mapdir(ipart),mapcomp(icomp))
          enddo
        enddo
      elseif(iborn.eq.1)then
        xfact=4*tk*uk
        do ipart=1,9
          do icomp=0,3
            pme1(icomp,ipart)=xmom(mapdir(ipart),mapcomp(icomp))
          enddo
        enddo
      else
        write(*,*)'xmadevttb: unknown iborn value',iborn
        stop
      endif
      if(iborn.eq.0.and.jproc.eq.1)then
        call gg_ttb(pme0,tmp)
      elseif(iborn.eq.0.and.jproc.eq.2)then
        call uub_ttb(pme0,tmp)
      elseif(iborn.eq.1.and.jproc.eq.1)then
        call gg_ttbg(pme1,tmp)
      elseif(iborn.eq.1.and.jproc.eq.2)then
        call uub_ttbg(pme1,tmp)
      elseif(iborn.eq.1.and.jproc.eq.3)then
        call ug_ttbu(pme1,tmp)
      else
        write(*,*)'xmadevttb: Error #2'
        stop
      endif
      xmadevttb=xfact*tmp(1)/(2*s)
      return
      end


      subroutine setmepqq(xiwmass,xiwwidth,xizmass,xizwidth,
     #                    xitmass,xitwidth,xibmass,xisin2w,xiee2,xig)
c Fills HELAS common blocks for masses and couplings. The electron charge
c squared and the masses may eventually be passed through a common block
c on a event-by-event basis. This code is mainly taken from coupsm-ORIGINAL.F 
c of the HELAS package. Here, we limit ourselves to setting the following
c parameters:
c
c       real    gw                : weak coupling constant
c       real    gwwa              : dimensionless WWA  coupling
c       real    gwwz              : dimensionless WWZ  coupling
c       complex gal(2)            : coupling with A of charged leptons
c       complex gau(2)            : coupling with A of up-type quarks
c       complex gad(2)            : coupling with A of down-type quarks
c       complex gwf(2)            : coupling with W-,W+ of fermions
c       complex gzn(2)            : coupling with Z of neutrinos
c       complex gzl(2)            : coupling with Z of charged leptons
c       complex gzu(2)            : coupling with Z of up-type quarks
c       complex gzd(2)            : coupling with Z of down-type quarks
c       complex gg(2)             : QCD gqq coupling (L,R)
c
c through the following parameters, given in input
c
c       real    zmass,wmass       : weak boson masses
c       real    zwidth,wwidth     : weak boson width
c       real    tmass,twidth      : top mass and width
c       real    bmass             : bottom mass
c       real    sin2w             : square of sine of the weak angle
c       real    ee2               : positron charge squared
c       real    g                 : QCD 3-,4-gluon coupling
c
      implicit none
      real * 8 xiwmass,xiwwidth,xizmass,xizwidth,xitmass,xitwidth,
     #         xibmass,xisin2w,xiee2,xig
      include "MEcoupl.inc"
      double precision zero,half,one,two,three,pi,ee2,sw,cw,ez,ey,sc2,v
      parameter (zero=0.d0)
      parameter (half=0.5d0)
      parameter (one=1.d0)
      parameter (two=2.d0)
      parameter (three=3.d0)
      parameter (pi=3.14159265358979312D0)
c
      wmass = xiwmass
      wwidth= xiwwidth
      zmass = xizmass
      zwidth= xizwidth
      tmass = xitmass
      twidth= xitwidth
      bmass = xibmass
      sin2w = xisin2w
      ee2   = xiee2
      g     = xig
c
      amass=0.d0
      awidth=1.d-99
c
      ee=sqrt(ee2)
      alpha=ee2/(4*pi)
c
      sw  = sqrt( sin2w )
      cw  = sqrt( One - sin2w )
      ez  = ee/(sw*cw)
      ey  = ee*(sw/cw)
      sc2 = sin2w*( One - sin2w )
      v   = Two*zmass*sqrt(sc2)/ee
c
c vector boson couplings
c
      gw   = ee/sw
      gwwa = ee
      gwwz = ee*cw/sw
c
c fermion-fermion-vector couplings
c
      gal(1) = dcmplx(  ee          , Zero )
      gal(2) = dcmplx(  ee          , Zero )
      gau(1) = dcmplx( -ee*Two/Three, Zero )
      gau(2) = dcmplx( -ee*Two/Three, Zero )
      gad(1) = dcmplx(  ee/Three    , Zero )
      gad(2) = dcmplx(  ee/Three    , Zero )
c
      gwf(1) = dcmplx( -ee/sqrt(Two*sin2w), Zero )
      gwf(2) = dcmplx(  Zero              , Zero )
c
      gzn(1) = dcmplx( -ez*Half                     , Zero )
      gzn(2) = dcmplx(  Zero                        , Zero )
      gzl(1) = dcmplx( -ez*(-Half + sin2w)          , Zero )
      gzl(2) = dcmplx( -ey                          , Zero )
      gzu(1) = dcmplx( -ez*( Half - sin2w*Two/Three), Zero )
      gzu(2) = dcmplx(  ey*Two/Three                , Zero )
      gzd(1) = dcmplx( -ez*(-Half + sin2w/Three)    , Zero )
      gzd(2) = dcmplx( -ey/Three                    , Zero )
c
c QCD coupling
c
      gg(1) = dcmplx( -g, Zero )
      gg(2) = gg(1)
c
      return
      end


C      subroutine switchmom(p1,p,ic,jc,nexternal)
Cc**************************************************************************
Cc     Changes stuff for crossings
Cc**************************************************************************
C      implicit none
C      integer nexternal
C      integer jc(nexternal),ic(nexternal)
C      real*8 p1(0:3,nexternal),p(0:3,nexternal)
C      integer i,j
Cc-----
Cc Begin Code
Cc-----
C      do i=1,nexternal
C         do j=0,3
C            p(j,ic(i))=p1(j,i)
C         enddo
C      enddo
C      do i=1,nexternal
C         jc(i)=1
C      enddo
C      jc(ic(1))=-1
C      jc(ic(2))=-1
C      end
c
c
c Begin of qqbar --> ttbar
c
c
C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UUB_TTB(P1,ANS)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve b e- ve~ b~  
C  
C Crossing   1 is u u~ -> e+ ve b e- ve~ b~  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  8)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 256, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUUB_TTB
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    1/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,8) /-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,8) /-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,8) /-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,8) /-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,8) /-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,8) /-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,8) /-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,8) /-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,8) /-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,8) /-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,8) /-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,8) /-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,8) /-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,8) /-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,8) /-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,8) /-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,8) /-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,8) /-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,8) /-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,8) /-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,8) /-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,8) /-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,8) /-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,8) /-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,8) /-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,8) /-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,8) /-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,8) /-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,8) /-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,8) /-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,8) /-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,8) /-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,8) /-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,8) /-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,8) /-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,8) /-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,8) /-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,8) /-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,8) /-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,8) /-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,8) /-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,8) /-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,8) /-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,8) /-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,8) /-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,8) /-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,8) /-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,8) /-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,8) /-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,8) /-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,8) /-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,8) /-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,8) /-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,8) /-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,8) /-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,8) /-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,8) /-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,8) /-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,8) /-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,8) /-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,8) /-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,8) /-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,8) /-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,8) /-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,8) /-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,8) /-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,8) /-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,8) /-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,8) /-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,8) /-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,8) /-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,8) /-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,8) /-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,8) /-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,8) /-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,8) /-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,8) /-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,8) /-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,8) /-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,8) /-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,8) /-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,8) /-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,8) /-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,8) /-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,8) /-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,8) /-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,8) /-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,8) /-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,8) /-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,8) /-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,8) /-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,8) /-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,8) /-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,8) /-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,8) /-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,8) /-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,8) /-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,8) /-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,8) /-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,8) /-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,8) /-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,8) /-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,8) /-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,8) /-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,8) /-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,8) /-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,8) /-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,8) /-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,8) /-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,8) /-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,8) /-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,8) /-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,8) /-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,8) /-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,8) /-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,8) /-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,8) /-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,8) /-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,8) /-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,8) /-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,8) /-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,8) /-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,8) /-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,8) /-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,8) /-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,8) /-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,8) /-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,8) /-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 129),IHEL=1,8) / 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 130),IHEL=1,8) / 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 131),IHEL=1,8) / 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 132),IHEL=1,8) / 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 133),IHEL=1,8) / 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 134),IHEL=1,8) / 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 135),IHEL=1,8) / 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 136),IHEL=1,8) / 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 137),IHEL=1,8) / 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 138),IHEL=1,8) / 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 139),IHEL=1,8) / 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 140),IHEL=1,8) / 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 141),IHEL=1,8) / 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 142),IHEL=1,8) / 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 143),IHEL=1,8) / 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 144),IHEL=1,8) / 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 145),IHEL=1,8) / 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 146),IHEL=1,8) / 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 147),IHEL=1,8) / 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 148),IHEL=1,8) / 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 149),IHEL=1,8) / 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 150),IHEL=1,8) / 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 151),IHEL=1,8) / 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 152),IHEL=1,8) / 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 153),IHEL=1,8) / 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 154),IHEL=1,8) / 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 155),IHEL=1,8) / 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 156),IHEL=1,8) / 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 157),IHEL=1,8) / 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 158),IHEL=1,8) / 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 159),IHEL=1,8) / 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 160),IHEL=1,8) / 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 161),IHEL=1,8) / 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 162),IHEL=1,8) / 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 163),IHEL=1,8) / 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 164),IHEL=1,8) / 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 165),IHEL=1,8) / 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 166),IHEL=1,8) / 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 167),IHEL=1,8) / 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 168),IHEL=1,8) / 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 169),IHEL=1,8) / 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 170),IHEL=1,8) / 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 171),IHEL=1,8) / 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 172),IHEL=1,8) / 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 173),IHEL=1,8) / 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 174),IHEL=1,8) / 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 175),IHEL=1,8) / 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 176),IHEL=1,8) / 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 177),IHEL=1,8) / 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 178),IHEL=1,8) / 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 179),IHEL=1,8) / 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 180),IHEL=1,8) / 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 181),IHEL=1,8) / 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 182),IHEL=1,8) / 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 183),IHEL=1,8) / 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 184),IHEL=1,8) / 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 185),IHEL=1,8) / 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 186),IHEL=1,8) / 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 187),IHEL=1,8) / 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 188),IHEL=1,8) / 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 189),IHEL=1,8) / 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 190),IHEL=1,8) / 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 191),IHEL=1,8) / 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 192),IHEL=1,8) / 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 193),IHEL=1,8) / 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 194),IHEL=1,8) / 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 195),IHEL=1,8) / 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 196),IHEL=1,8) / 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 197),IHEL=1,8) / 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 198),IHEL=1,8) / 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 199),IHEL=1,8) / 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 200),IHEL=1,8) / 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 201),IHEL=1,8) / 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 202),IHEL=1,8) / 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 203),IHEL=1,8) / 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 204),IHEL=1,8) / 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 205),IHEL=1,8) / 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 206),IHEL=1,8) / 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 207),IHEL=1,8) / 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 208),IHEL=1,8) / 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 209),IHEL=1,8) / 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 210),IHEL=1,8) / 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 211),IHEL=1,8) / 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 212),IHEL=1,8) / 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 213),IHEL=1,8) / 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 214),IHEL=1,8) / 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 215),IHEL=1,8) / 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 216),IHEL=1,8) / 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 217),IHEL=1,8) / 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 218),IHEL=1,8) / 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 219),IHEL=1,8) / 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 220),IHEL=1,8) / 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 221),IHEL=1,8) / 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 222),IHEL=1,8) / 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 223),IHEL=1,8) / 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 224),IHEL=1,8) / 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 225),IHEL=1,8) / 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 226),IHEL=1,8) / 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 227),IHEL=1,8) / 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 228),IHEL=1,8) / 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 229),IHEL=1,8) / 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 230),IHEL=1,8) / 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 231),IHEL=1,8) / 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 232),IHEL=1,8) / 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 233),IHEL=1,8) / 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 234),IHEL=1,8) / 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 235),IHEL=1,8) / 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 236),IHEL=1,8) / 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 237),IHEL=1,8) / 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 238),IHEL=1,8) / 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 239),IHEL=1,8) / 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 240),IHEL=1,8) / 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 241),IHEL=1,8) / 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 242),IHEL=1,8) / 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 243),IHEL=1,8) / 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 244),IHEL=1,8) / 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 245),IHEL=1,8) / 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 246),IHEL=1,8) / 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 247),IHEL=1,8) / 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 248),IHEL=1,8) / 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 249),IHEL=1,8) / 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 250),IHEL=1,8) / 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 251),IHEL=1,8) / 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 252),IHEL=1,8) / 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 253),IHEL=1,8) / 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 254),IHEL=1,8) / 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 255),IHEL=1,8) / 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 256),IHEL=1,8) / 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,8) / 1, 2, 3, 4, 5, 6, 7, 8/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUUB_TTB(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUUB_TTB(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUUB_TTB
      REAL*8 FUNCTION MATUUB_TTB(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve b e- ve~ b~  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   1,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      include "genps.inc"
      integer    nexternal
      parameter (nexternal=  8)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  13, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     2/                                  
C               T[8,2]T[1,5]                                               
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),BMASS ,NHEL(5   ),+1*IC(5   ),W(1,5   ))       
      CALL OXXXXX(P(0,6   ),ZERO ,NHEL(6   ),+1*IC(6   ),W(1,6   ))        
      CALL IXXXXX(P(0,7   ),ZERO ,NHEL(7   ),-1*IC(7   ),W(1,7   ))        
      CALL IXXXXX(P(0,8   ),BMASS ,NHEL(8   ),-1*IC(8   ),W(1,8   ))       
      CALL JIOXXX(W(1,1   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,9   ))     
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,10  ))    
      CALL FVOXXX(W(1,5   ),W(1,10  ),GWF ,TMASS   ,TWIDTH  ,W(1,11  ))    
      CALL FVOXXX(W(1,11  ),W(1,9   ),GG ,TMASS   ,TWIDTH  ,W(1,12  ))     
      CALL JIOXXX(W(1,7   ),W(1,6   ),GWF ,WMASS   ,WWIDTH  ,W(1,13  ))    
      CALL IOVXXX(W(1,8   ),W(1,12  ),W(1,13  ),GWF ,AMP(1   ))            
      JAMP(   1) = +AMP(   1)
      MATUUB_TTB = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUUB_TTB =MATUUB_TTB+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of qqbar --> ttbar
c
c
c
c
c Begin of gg --> ttbar
c
c
C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE GG_TTB(P1,ANS)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : g g -> e+ ve b e- ve~ b~  
C  
C Crossing   1 is g g -> e+ ve b e- ve~ b~  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  8)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 256, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATGG_TTB
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    3/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   2/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,8) /-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,8) /-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,8) /-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,8) /-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,8) /-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,8) /-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,8) /-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,8) /-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,8) /-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,8) /-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,8) /-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,8) /-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,8) /-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,8) /-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,8) /-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,8) /-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,8) /-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,8) /-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,8) /-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,8) /-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,8) /-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,8) /-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,8) /-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,8) /-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,8) /-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,8) /-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,8) /-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,8) /-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,8) /-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,8) /-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,8) /-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,8) /-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,8) /-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,8) /-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,8) /-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,8) /-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,8) /-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,8) /-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,8) /-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,8) /-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,8) /-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,8) /-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,8) /-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,8) /-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,8) /-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,8) /-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,8) /-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,8) /-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,8) /-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,8) /-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,8) /-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,8) /-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,8) /-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,8) /-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,8) /-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,8) /-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,8) /-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,8) /-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,8) /-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,8) /-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,8) /-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,8) /-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,8) /-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,8) /-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,8) /-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,8) /-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,8) /-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,8) /-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,8) /-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,8) /-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,8) /-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,8) /-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,8) /-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,8) /-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,8) /-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,8) /-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,8) /-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,8) /-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,8) /-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,8) /-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,8) /-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,8) /-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,8) /-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,8) /-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,8) /-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,8) /-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,8) /-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,8) /-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,8) /-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,8) /-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,8) /-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,8) /-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,8) /-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,8) /-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,8) /-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,8) /-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,8) /-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,8) /-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,8) /-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,8) /-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,8) /-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,8) /-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,8) /-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,8) /-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,8) /-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,8) /-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,8) /-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,8) /-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,8) /-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,8) /-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,8) /-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,8) /-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,8) /-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,8) /-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,8) /-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,8) /-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,8) /-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,8) /-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,8) /-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,8) /-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,8) /-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,8) /-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,8) /-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,8) /-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,8) /-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,8) /-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,8) /-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,8) /-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 129),IHEL=1,8) / 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 130),IHEL=1,8) / 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 131),IHEL=1,8) / 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 132),IHEL=1,8) / 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 133),IHEL=1,8) / 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 134),IHEL=1,8) / 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 135),IHEL=1,8) / 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 136),IHEL=1,8) / 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 137),IHEL=1,8) / 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 138),IHEL=1,8) / 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 139),IHEL=1,8) / 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 140),IHEL=1,8) / 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 141),IHEL=1,8) / 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 142),IHEL=1,8) / 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 143),IHEL=1,8) / 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 144),IHEL=1,8) / 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 145),IHEL=1,8) / 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 146),IHEL=1,8) / 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 147),IHEL=1,8) / 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 148),IHEL=1,8) / 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 149),IHEL=1,8) / 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 150),IHEL=1,8) / 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 151),IHEL=1,8) / 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 152),IHEL=1,8) / 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 153),IHEL=1,8) / 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 154),IHEL=1,8) / 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 155),IHEL=1,8) / 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 156),IHEL=1,8) / 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 157),IHEL=1,8) / 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 158),IHEL=1,8) / 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 159),IHEL=1,8) / 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 160),IHEL=1,8) / 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 161),IHEL=1,8) / 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 162),IHEL=1,8) / 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 163),IHEL=1,8) / 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 164),IHEL=1,8) / 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 165),IHEL=1,8) / 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 166),IHEL=1,8) / 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 167),IHEL=1,8) / 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 168),IHEL=1,8) / 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 169),IHEL=1,8) / 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 170),IHEL=1,8) / 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 171),IHEL=1,8) / 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 172),IHEL=1,8) / 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 173),IHEL=1,8) / 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 174),IHEL=1,8) / 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 175),IHEL=1,8) / 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 176),IHEL=1,8) / 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 177),IHEL=1,8) / 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 178),IHEL=1,8) / 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 179),IHEL=1,8) / 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 180),IHEL=1,8) / 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 181),IHEL=1,8) / 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 182),IHEL=1,8) / 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 183),IHEL=1,8) / 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 184),IHEL=1,8) / 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 185),IHEL=1,8) / 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 186),IHEL=1,8) / 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 187),IHEL=1,8) / 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 188),IHEL=1,8) / 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 189),IHEL=1,8) / 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 190),IHEL=1,8) / 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 191),IHEL=1,8) / 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 192),IHEL=1,8) / 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 193),IHEL=1,8) / 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 194),IHEL=1,8) / 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 195),IHEL=1,8) / 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 196),IHEL=1,8) / 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 197),IHEL=1,8) / 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 198),IHEL=1,8) / 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 199),IHEL=1,8) / 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 200),IHEL=1,8) / 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 201),IHEL=1,8) / 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 202),IHEL=1,8) / 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 203),IHEL=1,8) / 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 204),IHEL=1,8) / 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 205),IHEL=1,8) / 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 206),IHEL=1,8) / 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 207),IHEL=1,8) / 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 208),IHEL=1,8) / 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 209),IHEL=1,8) / 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 210),IHEL=1,8) / 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 211),IHEL=1,8) / 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 212),IHEL=1,8) / 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 213),IHEL=1,8) / 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 214),IHEL=1,8) / 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 215),IHEL=1,8) / 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 216),IHEL=1,8) / 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 217),IHEL=1,8) / 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 218),IHEL=1,8) / 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 219),IHEL=1,8) / 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 220),IHEL=1,8) / 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 221),IHEL=1,8) / 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 222),IHEL=1,8) / 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 223),IHEL=1,8) / 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 224),IHEL=1,8) / 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 225),IHEL=1,8) / 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 226),IHEL=1,8) / 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 227),IHEL=1,8) / 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 228),IHEL=1,8) / 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 229),IHEL=1,8) / 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 230),IHEL=1,8) / 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 231),IHEL=1,8) / 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 232),IHEL=1,8) / 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 233),IHEL=1,8) / 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 234),IHEL=1,8) / 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 235),IHEL=1,8) / 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 236),IHEL=1,8) / 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 237),IHEL=1,8) / 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 238),IHEL=1,8) / 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 239),IHEL=1,8) / 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 240),IHEL=1,8) / 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 241),IHEL=1,8) / 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 242),IHEL=1,8) / 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 243),IHEL=1,8) / 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 244),IHEL=1,8) / 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 245),IHEL=1,8) / 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 246),IHEL=1,8) / 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 247),IHEL=1,8) / 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 248),IHEL=1,8) / 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 249),IHEL=1,8) / 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 250),IHEL=1,8) / 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 251),IHEL=1,8) / 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 252),IHEL=1,8) / 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 253),IHEL=1,8) / 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 254),IHEL=1,8) / 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 255),IHEL=1,8) / 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 256),IHEL=1,8) / 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,8) / 1, 2, 3, 4, 5, 6, 7, 8/
      DATA (IDEN(IHEL),IHEL=  1,  1) / 256/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATGG_TTB(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATGG_TTB(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATGG_TTB
      REAL*8 FUNCTION MATGG_TTB(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : g g -> e+ ve b e- ve~ b~  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   3,NEIGEN=  2) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      include "genps.inc"
      integer    nexternal
      parameter (nexternal=  8)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  16, NCOLOR=   2) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            3/                                       
      DATA (CF(i,1  ),i=1  ,2  ) /    16,   -2/                            
C               T[8,5,1,2]                                                 
      DATA Denom(2  )/            3/                                       
      DATA (CF(i,2  ),i=1  ,2  ) /    -2,   16/                            
C               T[8,5,2,1]                                                 
C ----------
C BEGIN CODE
C ----------
      CALL VXXXXX(P(0,1   ),ZERO ,NHEL(1   ),-1*IC(1   ),W(1,1   ))        
      CALL VXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),BMASS ,NHEL(5   ),+1*IC(5   ),W(1,5   ))       
      CALL OXXXXX(P(0,6   ),ZERO ,NHEL(6   ),+1*IC(6   ),W(1,6   ))        
      CALL IXXXXX(P(0,7   ),ZERO ,NHEL(7   ),-1*IC(7   ),W(1,7   ))        
      CALL IXXXXX(P(0,8   ),BMASS ,NHEL(8   ),-1*IC(8   ),W(1,8   ))       
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,9   ))    
      CALL FVOXXX(W(1,5   ),W(1,9   ),GWF ,TMASS   ,TWIDTH  ,W(1,10  ))    
      CALL JIOXXX(W(1,7   ),W(1,6   ),GWF ,WMASS   ,WWIDTH  ,W(1,11  ))    
      CALL FVIXXX(W(1,8   ),W(1,11  ),GWF ,TMASS   ,TWIDTH  ,W(1,12  ))    
      CALL FVOXXX(W(1,10  ),W(1,2   ),GG ,TMASS   ,TWIDTH  ,W(1,13  ))     
      CALL IOVXXX(W(1,12  ),W(1,13  ),W(1,1   ),GG ,AMP(1   ))             
      CALL FVOXXX(W(1,10  ),W(1,1   ),GG ,TMASS   ,TWIDTH  ,W(1,14  ))     
      CALL IOVXXX(W(1,12  ),W(1,14  ),W(1,2   ),GG ,AMP(2   ))             
      CALL JVVXXX(W(1,1   ),W(1,2   ),G ,ZERO    ,ZERO    ,W(1,15  ))      
      CALL FVOXXX(W(1,10  ),W(1,15  ),GG ,TMASS   ,TWIDTH  ,W(1,16  ))     
      CALL IOVXXX(W(1,8   ),W(1,16  ),W(1,11  ),GWF ,AMP(3   ))            
      JAMP(   1) = +AMP(   1)-AMP(   3)
      JAMP(   2) = +AMP(   2)+AMP(   3)
      MATGG_TTB = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATGG_TTB =MATGG_TTB+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of gg --> ttbar
c
c
c
c
c Begin of qqbar --> ttbar g
c
c
C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UUB_TTBG(P1,ANS)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve b e- ve~ b~ g  
C  
C Crossing   1 is u u~ -> e+ ve b e- ve~ b~ g  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 512, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUUB_TTBG
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    5/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   4/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 129),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 130),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 131),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 132),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 133),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 134),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 135),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 136),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 137),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 138),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 139),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 140),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 141),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 142),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 143),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 144),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 145),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 146),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 147),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 148),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 149),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 150),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 151),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 152),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 153),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 154),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 155),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 156),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 157),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 158),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 159),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 160),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 161),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 162),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 163),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 164),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 165),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 166),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 167),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 168),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 169),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 170),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 171),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 172),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 173),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 174),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 175),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 176),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 177),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 178),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 179),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 180),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 181),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 182),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 183),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 184),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 185),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 186),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 187),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 188),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 189),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 190),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 191),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 192),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 193),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 194),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 195),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 196),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 197),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 198),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 199),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 200),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 201),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 202),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 203),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 204),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 205),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 206),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 207),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 208),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 209),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 210),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 211),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 212),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 213),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 214),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 215),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 216),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 217),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 218),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 219),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 220),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 221),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 222),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 223),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 224),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 225),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 226),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 227),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 228),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 229),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 230),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 231),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 232),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 233),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 234),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 235),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 236),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 237),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 238),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 239),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 240),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 241),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 242),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 243),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 244),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 245),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 246),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 247),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 248),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 249),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 250),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 251),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 252),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 253),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 254),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 255),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 256),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 257),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 258),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 259),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 260),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 261),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 262),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 263),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 264),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 265),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 266),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 267),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 268),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 269),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 270),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 271),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 272),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 273),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 274),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 275),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 276),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 277),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 278),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 279),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 280),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 281),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 282),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 283),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 284),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 285),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 286),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 287),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 288),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 289),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 290),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 291),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 292),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 293),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 294),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 295),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 296),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 297),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 298),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 299),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 300),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 301),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 302),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 303),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 304),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 305),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 306),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 307),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 308),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 309),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 310),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 311),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 312),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 313),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 314),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 315),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 316),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 317),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 318),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 319),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 320),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 321),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 322),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 323),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 324),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 325),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 326),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 327),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 328),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 329),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 330),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 331),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 332),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 333),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 334),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 335),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 336),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 337),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 338),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 339),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 340),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 341),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 342),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 343),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 344),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 345),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 346),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 347),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 348),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 349),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 350),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 351),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 352),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 353),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 354),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 355),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 356),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 357),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 358),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 359),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 360),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 361),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 362),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 363),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 364),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 365),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 366),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 367),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 368),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 369),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 370),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 371),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 372),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 373),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 374),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 375),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 376),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 377),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 378),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 379),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 380),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 381),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 382),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 383),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 384),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 385),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 386),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 387),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 388),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 389),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 390),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 391),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 392),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 393),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 394),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 395),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 396),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 397),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 398),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 399),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 400),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 401),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 402),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 403),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 404),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 405),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 406),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 407),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 408),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 409),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 410),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 411),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 412),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 413),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 414),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 415),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 416),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 417),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 418),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 419),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 420),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 421),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 422),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 423),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 424),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 425),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 426),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 427),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 428),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 429),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 430),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 431),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 432),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 433),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 434),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 435),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 436),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 437),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 438),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 439),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 440),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 441),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 442),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 443),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 444),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 445),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 446),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 447),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 448),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 449),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 450),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 451),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 452),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 453),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 454),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 455),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 456),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 457),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 458),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 459),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 460),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 461),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 462),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 463),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 464),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 465),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 466),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 467),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 468),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 469),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 470),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 471),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 472),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 473),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 474),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 475),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 476),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 477),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 478),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 479),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 480),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 481),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 482),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 483),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 484),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 485),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 486),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 487),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 488),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 489),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 490),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 491),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 492),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 493),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 494),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 495),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 496),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 497),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 498),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 499),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 500),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 501),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 502),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 503),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 504),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 505),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 506),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 507),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 508),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 509),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 510),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 511),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 512),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,9) / 1, 2, 3, 4, 5, 6, 7, 8, 9/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUUB_TTBG(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUUB_TTBG(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUUB_TTBG
      REAL*8 FUNCTION MATUUB_TTBG(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve b e- ve~ b~ g  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   5,NEIGEN=  4) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  21, NCOLOR=   4) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            3/                                       
      DATA (CF(i,1  ),i=1  ,4  ) /     8,   -1,    7,   -2/                
C               T[8,2]T[1,5,9]                                             
      DATA Denom(2  )/            3/                                       
      DATA (CF(i,2  ),i=1  ,4  ) /    -1,    8,   -2,    7/                
C               T[8,2,9]T[1,5]                                             
      DATA Denom(3  )/            3/                                       
      DATA (CF(i,3  ),i=1  ,4  ) /     7,   -2,    8,   -1/                
C               T[8,2]T[1,5,9]                                             
      DATA Denom(4  )/            3/                                       
      DATA (CF(i,4  ),i=1  ,4  ) /    -2,    7,   -1,    8/                
C               T[1,5]T[8,2,9]                                             
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),BMASS ,NHEL(5   ),+1*IC(5   ),W(1,5   ))       
      CALL OXXXXX(P(0,6   ),ZERO ,NHEL(6   ),+1*IC(6   ),W(1,6   ))        
      CALL IXXXXX(P(0,7   ),ZERO ,NHEL(7   ),-1*IC(7   ),W(1,7   ))        
      CALL IXXXXX(P(0,8   ),BMASS ,NHEL(8   ),-1*IC(8   ),W(1,8   ))       
      CALL VXXXXX(P(0,9   ),ZERO ,NHEL(9   ),+1*IC(9   ),W(1,9   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,10  ))    
      CALL FVOXXX(W(1,5   ),W(1,10  ),GWF ,TMASS   ,TWIDTH  ,W(1,11  ))    
      CALL JIOXXX(W(1,7   ),W(1,6   ),GWF ,WMASS   ,WWIDTH  ,W(1,12  ))    
      CALL FVIXXX(W(1,8   ),W(1,12  ),GWF ,TMASS   ,TWIDTH  ,W(1,13  ))    
      CALL FVIXXX(W(1,1   ),W(1,9   ),GG ,ZERO    ,ZERO    ,W(1,14  ))     
      CALL JIOXXX(W(1,14  ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,15  ))     
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,15  ),GG ,AMP(1   ))             
      CALL FVOXXX(W(1,2   ),W(1,9   ),GG ,ZERO    ,ZERO    ,W(1,16  ))     
      CALL JIOXXX(W(1,1   ),W(1,16  ),GG ,ZERO    ,ZERO    ,W(1,17  ))     
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,17  ),GG ,AMP(2   ))             
      CALL JIOXXX(W(1,1   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,18  ))     
      CALL FVOXXX(W(1,11  ),W(1,9   ),GG ,TMASS   ,TWIDTH  ,W(1,19  ))     
      CALL IOVXXX(W(1,13  ),W(1,19  ),W(1,18  ),GG ,AMP(3   ))             
      CALL JVVXXX(W(1,9   ),W(1,18  ),G ,ZERO    ,ZERO    ,W(1,20  ))      
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,20  ),GG ,AMP(4   ))             
      CALL FVOXXX(W(1,11  ),W(1,18  ),GG ,TMASS   ,TWIDTH  ,W(1,21  ))     
      CALL IOVXXX(W(1,13  ),W(1,21  ),W(1,9   ),GG ,AMP(5   ))             
      JAMP(   1) = +AMP(   1)
      JAMP(   2) = +AMP(   2)
      JAMP(   3) = +AMP(   3)+AMP(   4)
      JAMP(   4) = -AMP(   4)+AMP(   5)
      MATUUB_TTBG = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUUB_TTBG =MATUUB_TTBG+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of qqbar --> ttbar g
c
c
c
c
c Begin of gg --> ttbar g
c
c
C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE GG_TTBG(P1,ANS)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : g g -> e+ ve b e- ve~ b~ g  
C  
C Crossing   1 is g g -> e+ ve b e- ve~ b~ g  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 512, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATGG_TTBG
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /   18/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   6/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 129),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 130),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 131),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 132),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 133),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 134),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 135),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 136),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 137),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 138),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 139),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 140),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 141),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 142),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 143),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 144),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 145),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 146),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 147),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 148),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 149),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 150),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 151),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 152),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 153),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 154),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 155),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 156),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 157),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 158),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 159),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 160),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 161),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 162),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 163),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 164),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 165),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 166),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 167),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 168),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 169),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 170),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 171),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 172),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 173),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 174),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 175),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 176),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 177),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 178),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 179),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 180),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 181),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 182),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 183),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 184),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 185),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 186),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 187),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 188),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 189),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 190),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 191),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 192),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 193),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 194),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 195),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 196),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 197),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 198),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 199),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 200),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 201),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 202),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 203),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 204),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 205),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 206),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 207),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 208),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 209),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 210),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 211),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 212),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 213),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 214),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 215),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 216),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 217),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 218),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 219),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 220),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 221),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 222),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 223),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 224),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 225),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 226),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 227),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 228),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 229),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 230),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 231),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 232),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 233),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 234),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 235),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 236),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 237),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 238),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 239),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 240),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 241),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 242),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 243),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 244),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 245),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 246),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 247),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 248),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 249),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 250),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 251),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 252),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 253),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 254),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 255),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 256),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 257),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 258),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 259),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 260),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 261),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 262),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 263),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 264),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 265),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 266),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 267),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 268),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 269),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 270),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 271),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 272),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 273),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 274),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 275),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 276),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 277),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 278),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 279),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 280),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 281),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 282),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 283),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 284),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 285),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 286),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 287),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 288),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 289),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 290),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 291),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 292),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 293),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 294),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 295),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 296),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 297),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 298),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 299),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 300),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 301),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 302),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 303),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 304),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 305),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 306),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 307),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 308),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 309),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 310),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 311),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 312),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 313),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 314),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 315),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 316),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 317),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 318),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 319),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 320),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 321),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 322),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 323),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 324),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 325),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 326),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 327),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 328),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 329),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 330),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 331),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 332),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 333),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 334),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 335),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 336),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 337),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 338),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 339),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 340),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 341),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 342),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 343),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 344),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 345),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 346),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 347),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 348),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 349),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 350),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 351),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 352),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 353),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 354),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 355),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 356),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 357),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 358),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 359),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 360),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 361),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 362),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 363),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 364),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 365),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 366),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 367),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 368),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 369),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 370),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 371),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 372),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 373),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 374),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 375),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 376),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 377),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 378),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 379),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 380),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 381),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 382),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 383),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 384),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 385),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 386),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 387),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 388),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 389),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 390),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 391),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 392),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 393),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 394),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 395),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 396),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 397),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 398),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 399),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 400),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 401),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 402),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 403),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 404),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 405),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 406),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 407),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 408),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 409),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 410),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 411),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 412),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 413),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 414),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 415),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 416),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 417),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 418),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 419),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 420),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 421),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 422),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 423),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 424),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 425),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 426),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 427),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 428),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 429),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 430),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 431),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 432),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 433),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 434),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 435),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 436),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 437),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 438),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 439),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 440),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 441),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 442),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 443),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 444),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 445),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 446),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 447),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 448),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 449),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 450),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 451),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 452),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 453),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 454),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 455),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 456),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 457),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 458),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 459),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 460),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 461),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 462),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 463),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 464),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 465),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 466),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 467),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 468),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 469),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 470),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 471),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 472),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 473),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 474),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 475),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 476),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 477),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 478),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 479),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 480),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 481),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 482),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 483),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 484),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 485),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 486),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 487),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 488),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 489),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 490),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 491),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 492),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 493),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 494),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 495),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 496),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 497),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 498),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 499),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 500),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 501),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 502),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 503),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 504),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 505),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 506),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 507),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 508),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 509),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 510),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 511),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 512),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,9) / 1, 2, 3, 4, 5, 6, 7, 8, 9/
      DATA (IDEN(IHEL),IHEL=  1,  1) / 256/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATGG_TTBG(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATGG_TTBG(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATGG_TTBG
      REAL*8 FUNCTION MATGG_TTBG(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : g g -> e+ ve b e- ve~ b~ g  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=  18,NEIGEN=  6) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  36, NCOLOR=   6) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            9/                                       
      DATA (CF(i,1  ),i=1  ,6  ) /    64,   -8,    1,    1,   10,   -8/    
C               T[8,5,9,1,2]                                               
      DATA Denom(2  )/            9/                                       
      DATA (CF(i,2  ),i=1  ,6  ) /    -8,   64,   -8,   10,    1,    1/    
C               T[8,5,1,9,2]                                               
      DATA Denom(3  )/            9/                                       
      DATA (CF(i,3  ),i=1  ,6  ) /     1,   -8,   64,    1,   -8,   10/    
C               T[8,5,1,2,9]                                               
      DATA Denom(4  )/            9/                                       
      DATA (CF(i,4  ),i=1  ,6  ) /     1,   10,    1,   64,   -8,   -8/    
C               T[8,5,2,9,1]                                               
      DATA Denom(5  )/            9/                                       
      DATA (CF(i,5  ),i=1  ,6  ) /    10,    1,   -8,   -8,   64,    1/    
C               T[8,5,2,1,9]                                               
      DATA Denom(6  )/            9/                                       
      DATA (CF(i,6  ),i=1  ,6  ) /    -8,    1,   10,   -8,    1,   64/    
C               T[8,5,9,2,1]                                               
C ----------
C BEGIN CODE
C ----------
      CALL VXXXXX(P(0,1   ),ZERO ,NHEL(1   ),-1*IC(1   ),W(1,1   ))        
      CALL VXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),BMASS ,NHEL(5   ),+1*IC(5   ),W(1,5   ))       
      CALL OXXXXX(P(0,6   ),ZERO ,NHEL(6   ),+1*IC(6   ),W(1,6   ))        
      CALL IXXXXX(P(0,7   ),ZERO ,NHEL(7   ),-1*IC(7   ),W(1,7   ))        
      CALL IXXXXX(P(0,8   ),BMASS ,NHEL(8   ),-1*IC(8   ),W(1,8   ))       
      CALL VXXXXX(P(0,9   ),ZERO ,NHEL(9   ),+1*IC(9   ),W(1,9   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,10  ))    
      CALL FVOXXX(W(1,5   ),W(1,10  ),GWF ,TMASS   ,TWIDTH  ,W(1,11  ))    
      CALL JIOXXX(W(1,7   ),W(1,6   ),GWF ,WMASS   ,WWIDTH  ,W(1,12  ))    
      CALL FVIXXX(W(1,8   ),W(1,12  ),GWF ,TMASS   ,TWIDTH  ,W(1,13  ))    
      CALL JVVXXX(W(1,9   ),W(1,1   ),G ,ZERO    ,ZERO    ,W(1,14  ))      
      CALL FVOXXX(W(1,11  ),W(1,2   ),GG ,TMASS   ,TWIDTH  ,W(1,15  ))     
      CALL IOVXXX(W(1,13  ),W(1,15  ),W(1,14  ),GG ,AMP(1   ))             
      CALL JVVXXX(W(1,9   ),W(1,2   ),G ,ZERO    ,ZERO    ,W(1,16  ))      
      CALL FVOXXX(W(1,11  ),W(1,16  ),GG ,TMASS   ,TWIDTH  ,W(1,17  ))     
      CALL IOVXXX(W(1,13  ),W(1,17  ),W(1,1   ),GG ,AMP(2   ))             
      CALL FVOXXX(W(1,11  ),W(1,9   ),GG ,TMASS   ,TWIDTH  ,W(1,18  ))     
      CALL FVOXXX(W(1,18  ),W(1,2   ),GG ,TMASS   ,TWIDTH  ,W(1,19  ))     
      CALL IOVXXX(W(1,13  ),W(1,19  ),W(1,1   ),GG ,AMP(3   ))             
      CALL FVIXXX(W(1,13  ),W(1,1   ),GG ,TMASS   ,TWIDTH  ,W(1,20  ))     
      CALL IOVXXX(W(1,20  ),W(1,15  ),W(1,9   ),GG ,AMP(4   ))             
      CALL FVIXXX(W(1,13  ),W(1,9   ),GG ,TMASS   ,TWIDTH  ,W(1,21  ))     
      CALL IOVXXX(W(1,21  ),W(1,15  ),W(1,1   ),GG ,AMP(5   ))             
      CALL FVOXXX(W(1,11  ),W(1,14  ),GG ,TMASS   ,TWIDTH  ,W(1,22  ))     
      CALL IOVXXX(W(1,13  ),W(1,22  ),W(1,2   ),GG ,AMP(6   ))             
      CALL FVOXXX(W(1,11  ),W(1,1   ),GG ,TMASS   ,TWIDTH  ,W(1,23  ))     
      CALL IOVXXX(W(1,13  ),W(1,23  ),W(1,16  ),GG ,AMP(7   ))             
      CALL FVOXXX(W(1,18  ),W(1,1   ),GG ,TMASS   ,TWIDTH  ,W(1,24  ))     
      CALL IOVXXX(W(1,13  ),W(1,24  ),W(1,2   ),GG ,AMP(8   ))             
      CALL FVIXXX(W(1,13  ),W(1,2   ),GG ,TMASS   ,TWIDTH  ,W(1,25  ))     
      CALL IOVXXX(W(1,25  ),W(1,23  ),W(1,9   ),GG ,AMP(9   ))             
      CALL IOVXXX(W(1,21  ),W(1,23  ),W(1,2   ),GG ,AMP(10  ))             
      CALL JVVXXX(W(1,14  ),W(1,2   ),G ,ZERO    ,ZERO    ,W(1,26  ))      
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,26  ),GG ,AMP(11  ))             
      CALL JVVXXX(W(1,1   ),W(1,16  ),G ,ZERO    ,ZERO    ,W(1,27  ))      
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,27  ),GG ,AMP(12  ))             
      CALL JVVXXX(W(1,1   ),W(1,2   ),G ,ZERO    ,ZERO    ,W(1,28  ))      
      CALL IOVXXX(W(1,13  ),W(1,18  ),W(1,28  ),GG ,AMP(13  ))             
      CALL JVVXXX(W(1,9   ),W(1,28  ),G ,ZERO    ,ZERO    ,W(1,29  ))      
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,29  ),GG ,AMP(14  ))             
      CALL FVOXXX(W(1,11  ),W(1,28  ),GG ,TMASS   ,TWIDTH  ,W(1,30  ))     
      CALL IOVXXX(W(1,13  ),W(1,30  ),W(1,9   ),GG ,AMP(15  ))             
      CALL JGGGXX(W(1,1   ),W(1,2   ),W(1,9   ),G ,W(1,31  ))              
      CALL FVOXXX(W(1,11  ),W(1,31  ),GG ,TMASS   ,TWIDTH  ,W(1,32  ))     
      CALL IOVXXX(W(1,8   ),W(1,32  ),W(1,12  ),GWF ,AMP(16  ))            
      CALL JGGGXX(W(1,9   ),W(1,1   ),W(1,2   ),G ,W(1,33  ))              
      CALL FVOXXX(W(1,11  ),W(1,33  ),GG ,TMASS   ,TWIDTH  ,W(1,34  ))     
      CALL IOVXXX(W(1,8   ),W(1,34  ),W(1,12  ),GWF ,AMP(17  ))            
      CALL JGGGXX(W(1,2   ),W(1,9   ),W(1,1   ),G ,W(1,35  ))              
      CALL FVOXXX(W(1,11  ),W(1,35  ),GG ,TMASS   ,TWIDTH  ,W(1,36  ))     
      CALL IOVXXX(W(1,8   ),W(1,36  ),W(1,12  ),GWF ,AMP(18  ))            
      JAMP(   1) = -AMP(   1)+AMP(   5)+AMP(  11)+AMP(  14)-AMP(  15)
     &             +AMP(  16)-AMP(  17)
      JAMP(   2) = +AMP(   1)-AMP(   2)+AMP(   4)-AMP(  11)+AMP(  12)
     &             +AMP(  17)-AMP(  18)
      JAMP(   3) = +AMP(   2)+AMP(   3)-AMP(  12)-AMP(  13)-AMP(  14)
     &             -AMP(  16)+AMP(  18)
      JAMP(   4) = -AMP(   6)+AMP(   7)+AMP(   9)-AMP(  11)+AMP(  12)
     &             +AMP(  17)-AMP(  18)
      JAMP(   5) = +AMP(   6)+AMP(   8)+AMP(  11)+AMP(  13)+AMP(  14)
     &             +AMP(  16)-AMP(  17)
      JAMP(   6) = -AMP(   7)+AMP(  10)-AMP(  12)-AMP(  14)+AMP(  15)
     &             -AMP(  16)+AMP(  18)
      MATGG_TTBG = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATGG_TTBG =MATGG_TTBG+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of gg --> ttbar g
c
c
c
c
c Begin of qg --> ttbar q
c
c
C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UG_TTBU(P1,ANS)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u g -> e+ ve b e- ve~ b~ u  
C  
C Crossing   1 is u g -> e+ ve b e- ve~ b~ u  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 512, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUG_TTBU
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    5/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   4/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,9) /-1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,9) /-1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,9) /-1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,9) /-1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,9) /-1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,9) /-1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,9) /-1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,9) /-1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,9) /-1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,9) /-1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,9) /-1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,9) /-1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,9) /-1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,9) /-1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,9) /-1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,9) /-1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,9) /-1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,9) /-1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,9) /-1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,9) /-1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,9) /-1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,9) /-1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,9) /-1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,9) /-1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,9) /-1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,9) /-1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,9) /-1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,9) /-1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,9) /-1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,9) /-1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,9) /-1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,9) /-1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 129),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 130),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 131),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 132),IHEL=1,9) /-1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 133),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 134),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 135),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 136),IHEL=1,9) /-1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 137),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 138),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 139),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 140),IHEL=1,9) /-1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 141),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 142),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 143),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 144),IHEL=1,9) /-1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 145),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 146),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 147),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 148),IHEL=1,9) /-1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 149),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 150),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 151),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 152),IHEL=1,9) /-1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 153),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 154),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 155),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 156),IHEL=1,9) /-1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 157),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 158),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 159),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 160),IHEL=1,9) /-1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 161),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 162),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 163),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 164),IHEL=1,9) /-1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 165),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 166),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 167),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 168),IHEL=1,9) /-1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 169),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 170),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 171),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 172),IHEL=1,9) /-1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 173),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 174),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 175),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 176),IHEL=1,9) /-1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 177),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 178),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 179),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 180),IHEL=1,9) /-1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 181),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 182),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 183),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 184),IHEL=1,9) /-1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 185),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 186),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 187),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 188),IHEL=1,9) /-1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 189),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 190),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 191),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 192),IHEL=1,9) /-1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 193),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 194),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 195),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 196),IHEL=1,9) /-1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 197),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 198),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 199),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 200),IHEL=1,9) /-1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 201),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 202),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 203),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 204),IHEL=1,9) /-1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 205),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 206),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 207),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 208),IHEL=1,9) /-1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 209),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 210),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 211),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 212),IHEL=1,9) /-1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 213),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 214),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 215),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 216),IHEL=1,9) /-1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 217),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 218),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 219),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 220),IHEL=1,9) /-1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 221),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 222),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 223),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 224),IHEL=1,9) /-1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 225),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 226),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 227),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 228),IHEL=1,9) /-1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 229),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 230),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 231),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 232),IHEL=1,9) /-1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 233),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 234),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 235),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 236),IHEL=1,9) /-1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 237),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 238),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 239),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 240),IHEL=1,9) /-1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 241),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 242),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 243),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 244),IHEL=1,9) /-1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 245),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 246),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 247),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 248),IHEL=1,9) /-1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 249),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 250),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 251),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 252),IHEL=1,9) /-1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 253),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 254),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 255),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 256),IHEL=1,9) /-1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 257),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 258),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 259),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 260),IHEL=1,9) / 1,-1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 261),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 262),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 263),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 264),IHEL=1,9) / 1,-1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 265),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 266),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 267),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 268),IHEL=1,9) / 1,-1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 269),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 270),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 271),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 272),IHEL=1,9) / 1,-1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 273),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 274),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 275),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 276),IHEL=1,9) / 1,-1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 277),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 278),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 279),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 280),IHEL=1,9) / 1,-1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 281),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 282),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 283),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 284),IHEL=1,9) / 1,-1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 285),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 286),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 287),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 288),IHEL=1,9) / 1,-1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 289),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 290),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 291),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 292),IHEL=1,9) / 1,-1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 293),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 294),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 295),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 296),IHEL=1,9) / 1,-1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 297),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 298),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 299),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 300),IHEL=1,9) / 1,-1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 301),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 302),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 303),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 304),IHEL=1,9) / 1,-1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 305),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 306),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 307),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 308),IHEL=1,9) / 1,-1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 309),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 310),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 311),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 312),IHEL=1,9) / 1,-1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 313),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 314),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 315),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 316),IHEL=1,9) / 1,-1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 317),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 318),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 319),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 320),IHEL=1,9) / 1,-1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 321),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 322),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 323),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 324),IHEL=1,9) / 1,-1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 325),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 326),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 327),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 328),IHEL=1,9) / 1,-1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 329),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 330),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 331),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 332),IHEL=1,9) / 1,-1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 333),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 334),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 335),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 336),IHEL=1,9) / 1,-1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 337),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 338),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 339),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 340),IHEL=1,9) / 1,-1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 341),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 342),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 343),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 344),IHEL=1,9) / 1,-1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 345),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 346),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 347),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 348),IHEL=1,9) / 1,-1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 349),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 350),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 351),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 352),IHEL=1,9) / 1,-1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 353),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 354),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 355),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 356),IHEL=1,9) / 1,-1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 357),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 358),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 359),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 360),IHEL=1,9) / 1,-1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 361),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 362),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 363),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 364),IHEL=1,9) / 1,-1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 365),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 366),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 367),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 368),IHEL=1,9) / 1,-1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 369),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 370),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 371),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 372),IHEL=1,9) / 1,-1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 373),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 374),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 375),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 376),IHEL=1,9) / 1,-1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 377),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 378),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 379),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 380),IHEL=1,9) / 1,-1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 381),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 382),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 383),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 384),IHEL=1,9) / 1,-1, 1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 385),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 386),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 387),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 388),IHEL=1,9) / 1, 1,-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 389),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 390),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 391),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 392),IHEL=1,9) / 1, 1,-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 393),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 394),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 395),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 396),IHEL=1,9) / 1, 1,-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 397),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 398),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 399),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 400),IHEL=1,9) / 1, 1,-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 401),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 402),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 403),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 404),IHEL=1,9) / 1, 1,-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 405),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 406),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 407),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 408),IHEL=1,9) / 1, 1,-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 409),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 410),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 411),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 412),IHEL=1,9) / 1, 1,-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 413),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 414),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 415),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 416),IHEL=1,9) / 1, 1,-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 417),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 418),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 419),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 420),IHEL=1,9) / 1, 1,-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 421),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 422),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 423),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 424),IHEL=1,9) / 1, 1,-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 425),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 426),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 427),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 428),IHEL=1,9) / 1, 1,-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 429),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 430),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 431),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 432),IHEL=1,9) / 1, 1,-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 433),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 434),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 435),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 436),IHEL=1,9) / 1, 1,-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 437),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 438),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 439),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 440),IHEL=1,9) / 1, 1,-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 441),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 442),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 443),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 444),IHEL=1,9) / 1, 1,-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 445),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 446),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 447),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 448),IHEL=1,9) / 1, 1,-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 449),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 450),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 451),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 452),IHEL=1,9) / 1, 1, 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 453),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 454),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 455),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 456),IHEL=1,9) / 1, 1, 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 457),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 458),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 459),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 460),IHEL=1,9) / 1, 1, 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 461),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 462),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 463),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 464),IHEL=1,9) / 1, 1, 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 465),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 466),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 467),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 468),IHEL=1,9) / 1, 1, 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 469),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 470),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 471),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 472),IHEL=1,9) / 1, 1, 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 473),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 474),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 475),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 476),IHEL=1,9) / 1, 1, 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 477),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 478),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 479),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 480),IHEL=1,9) / 1, 1, 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 481),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 482),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 483),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 484),IHEL=1,9) / 1, 1, 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 485),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 486),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 487),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 488),IHEL=1,9) / 1, 1, 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 489),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 490),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 491),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 492),IHEL=1,9) / 1, 1, 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 493),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 494),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 495),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 496),IHEL=1,9) / 1, 1, 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 497),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 498),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 499),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 500),IHEL=1,9) / 1, 1, 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 501),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 502),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 503),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 504),IHEL=1,9) / 1, 1, 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 505),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 506),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 507),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 508),IHEL=1,9) / 1, 1, 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 509),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 510),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 511),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 512),IHEL=1,9) / 1, 1, 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,9) / 1, 2, 3, 4, 5, 6, 7, 8, 9/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  96/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUG_TTBU(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUG_TTBU(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUG_TTBU
      REAL*8 FUNCTION MATUG_TTBU(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.90. Updated 08/26/05               
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u g -> e+ ve b e- ve~ b~ u  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   5,NEIGEN=  4) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      include "genps.inc"
      integer    nexternal
      parameter (nexternal=  9)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  21, NCOLOR=   4) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            3/                                       
      DATA (CF(i,1  ),i=1  ,4  ) /     8,   -1,   -2,    7/                
C               T[8,9]T[1,5,2]                                             
      DATA Denom(2  )/            3/                                       
      DATA (CF(i,2  ),i=1  ,4  ) /    -1,    8,    7,   -2/                
C               T[1,5]T[8,9,2]                                             
      DATA Denom(3  )/            3/                                       
      DATA (CF(i,3  ),i=1  ,4  ) /    -2,    7,    8,   -1/                
C               T[8,9,2]T[1,5]                                             
      DATA Denom(4  )/            3/                                       
      DATA (CF(i,4  ),i=1  ,4  ) /     7,   -2,   -1,    8/                
C               T[8,9]T[1,5,2]                                             
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL VXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),BMASS ,NHEL(5   ),+1*IC(5   ),W(1,5   ))       
      CALL OXXXXX(P(0,6   ),ZERO ,NHEL(6   ),+1*IC(6   ),W(1,6   ))        
      CALL IXXXXX(P(0,7   ),ZERO ,NHEL(7   ),-1*IC(7   ),W(1,7   ))        
      CALL IXXXXX(P(0,8   ),BMASS ,NHEL(8   ),-1*IC(8   ),W(1,8   ))       
      CALL OXXXXX(P(0,9   ),ZERO ,NHEL(9   ),+1*IC(9   ),W(1,9   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,10  ))    
      CALL FVOXXX(W(1,5   ),W(1,10  ),GWF ,TMASS   ,TWIDTH  ,W(1,11  ))    
      CALL JIOXXX(W(1,7   ),W(1,6   ),GWF ,WMASS   ,WWIDTH  ,W(1,12  ))    
      CALL FVIXXX(W(1,8   ),W(1,12  ),GWF ,TMASS   ,TWIDTH  ,W(1,13  ))    
      CALL JIOXXX(W(1,1   ),W(1,9   ),GG ,ZERO    ,ZERO    ,W(1,14  ))     
      CALL FVOXXX(W(1,11  ),W(1,2   ),GG ,TMASS   ,TWIDTH  ,W(1,15  ))     
      CALL IOVXXX(W(1,13  ),W(1,15  ),W(1,14  ),GG ,AMP(1   ))             
      CALL FVOXXX(W(1,11  ),W(1,14  ),GG ,TMASS   ,TWIDTH  ,W(1,16  ))     
      CALL IOVXXX(W(1,13  ),W(1,16  ),W(1,2   ),GG ,AMP(2   ))             
      CALL JVVXXX(W(1,14  ),W(1,2   ),G ,ZERO    ,ZERO    ,W(1,17  ))      
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,17  ),GG ,AMP(3   ))             
      CALL FVOXXX(W(1,9   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,18  ))     
      CALL JIOXXX(W(1,1   ),W(1,18  ),GG ,ZERO    ,ZERO    ,W(1,19  ))     
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,19  ),GG ,AMP(4   ))             
      CALL FVIXXX(W(1,1   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,20  ))     
      CALL JIOXXX(W(1,20  ),W(1,9   ),GG ,ZERO    ,ZERO    ,W(1,21  ))     
      CALL IOVXXX(W(1,13  ),W(1,11  ),W(1,21  ),GG ,AMP(5   ))             
      JAMP(   1) = +AMP(   1)-AMP(   3)
      JAMP(   2) = +AMP(   2)+AMP(   3)
      JAMP(   3) = +AMP(   4)
      JAMP(   4) = +AMP(   5)
      MATUG_TTBU = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUG_TTBU =MATUG_TTBU+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of qg --> ttbar q
c
c
c
c
c End of lepton matrix elements
c
c
