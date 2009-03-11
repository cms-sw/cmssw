cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...first,second,third,fourth: get the matrix elements for four      c
c...              group of basic helicity states.                    c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) z.x zhang, chafik driouich, paula eerola and x.g. wu c
c reference: hep-ph/0309120                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c             nine basic functions from cb and cc sets.              c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	subroutine bfirst
	implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex bfun,colmat,yup,xup,inpup,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/tryup/bfun(9,4,64),yup(100),xup(2),idp,idq0,idk1,idk2,
     &	idb1,idb2,idc1,idc2
      common/counter/ibcstate,nev
c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

c...(l1,l2,l3,l4=+\-) corresponding to the two helicity states of the particle.
       ii=1
c...if (l1.eq.l2 .and. l3.eq.l4) then
c...helicity(-l1,-l2,-l3,-l4,-l5,-l6)=dconjg(helicity(l1,l2,l3,l4,l5,l6))
c...if (l1.eq.l2 .and. l3.ne.l4) then
c...helicity(-l1,-l2,-l3,-l4,-l5,-l6)=-dconjg(helicity(l1,l2,l3,l4,l5,l6))
c...if (l1.ne.l2 .and. l3.eq.l4) then
c...helicity(-l1,-l2,-l3,-l4,-l5,-l6)=-dconjg(helicity(l1,l2,l3,l4,l5,l6))
c...if (l1.ne.l2 .and. l3.ne.l4) then
c...helicity(-l1,-l2,-l3,-l4,-l5,-l6)=dconjg(helicity(l1,l2,l3,l4,l5,l6))

c...for short, we take the notation: helicity(q_b1,q_b2,q_c1,q_c2,k_1,k_2)=
c...helicity(l1,l2,l3,l4,l5,l6)=m(l1,l2,l3,l4,l5,l6). 
c...and helicity(kk)=helicity(kk,1) (kk=1-64).

c...helicity(1)=m(+,+,+,+,+,+)
        kk=1
	  if(iqqbar.eq.0) then
        bfun(1,ii,kk)=(4*yup(1)*(dotup(idb1,idb1)*((yup(11) - yup(12))
     -     *yup(19)*yup(27) +yup(10)*(-(dconjg(yup(5))*yup(5)) + 
     -     dconjg(yup(34))*yup(34))) + yup(10)*((-(dconjg(yup(24))*
     -     dconjg(yup(34))) +dconjg(yup(5))*yup(2))*yup(20) + 
     -     dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/xup(1)
        bfun(2,ii,kk)=(-4*(dotup(idb1,idb1)*yup(9)*(yup(11)*yup(19)*
     -     yup(27) + dconjg(yup(34))*yup(10)*yup(34)) +yup(10)* 
     -     (-(dconjg(yup(24))*dconjg(yup(34))*yup(9)*yup(20)) + 
     -     dotup(idc1,idc1)*(yup(2)*yup(29)*yup(31) + 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/xup(1)
	  bfun(3,ii,kk)=(4*yup(1)*yup(16)*((dconjg(yup(24))*
     -     dconjg(yup(34))+dconjg(yup(5))*(-yup(2)+yup(7)))*yup(20) + 
     -     dconjg(yup(17))*(dotup(idc1,idc1)*yup(17)-yup(11)*yup(32))+ 
     -     dotup(idb1,idb1)*(dconjg(yup(5))*yup(5) -dconjg(yup(34)) 
     -     *yup(34))-dconjg(yup(35))*dotup(idc1,idc1)*yup(35)))/xup(1)
	  bfun(4,ii,kk)=(4*yup(16)*(yup(9)*(dconjg(yup(17))*yup(11)*
     -     yup(32) + dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*((yup(2) -
     -     yup(7))*yup(29)*yup(31) +yup(9)*(-(dconjg(yup(17))*yup(17))+ 
     -     dconjg(yup(35))*yup(35)))))/xup(1)
	  bfun(5,ii,kk)=(2*yup(1)*yup(16)*yup(27)*yup(29))/xup(1)
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(29)*(yup(9)*yup(16)*
     -     yup(19)+(yup(7)*yup(16) + yup(2)*(-yup(12) + yup(18)))*
     -     yup(31)) +4*dconjg(yup(34))*yup(9)*(yup(12) - yup(18))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34))+ 
     -     4*dconjg(yup(35))*(dotup(idc1,idc1)*yup(9)*(yup(12)-yup(18))*
     -     yup(35) + 2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     inpup(idq0,idb1)*yup(27)*yup(36)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(17)=(-,-,-,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c***********************************************************************
	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=-4*(dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)*
     -    yup(30) +dconjg(yup(34))*(-(dconjg(inpup(idq0,idk1))*
     -    dconjg(yup(24))*inpup(idc1,idk2))+dconjg(inpup(idc2,idk1))
     -    *inpup(idb1,idk2)*yup(43) + yup(5)*(yup(44) - yup(45)))) + 
     -    (-(dconjg(yup(24))*dconjg(yup(34))*yup(20)) +dconjg(yup(35)) 
     -    *dotup(idc1,idc1)*yup(35))*yup(46) + dconjg(yup(35))*
     -    (dconjg(yup(13))*dotup(idc1,idc1) -dconjg(yup(18))*yup(36))
     -    *yup(47))
        bfun(2,ii,kk)=4*(dconjg(yup(35))*(-(dconjg(inpup(idc2,idk1))
     -    *dconjg(yup(34))*inpup(idc1,idb1)) + dotup(idc1,idc1)*yup(6))
     -    *yup(48) + dotup(idb1,idb1)*(dconjg(yup(18))*yup(7)*yup(28) + 
     -    dotup(idc1,idc1)*yup(8)*yup(29) - dconjg(yup(34))*
     -    (dconjg(inpup(idq0,idc2))*inpup(idc1,idk2)* yup(2) + 
     -    dconjg(inpup(idc2,idk1))*inpup(idq0,idk2)*yup(20) - 
     -    dconjg(yup(14))*yup(49))) + (-(dconjg(yup(24))*dconjg(yup(34))
     -    *yup(20)) +dconjg(yup(35))*dotup(idc1,idc1)*yup(35))*yup(50))
	bfun(3,ii,kk)=-4*(-((-(dconjg(yup(10))*yup(2))+dotup(idb1,idb1)*
     -    yup(8))*(dconjg(yup(9))*yup(1) +dconjg(yup(18))*yup(11))) + 
     -    dconjg(yup(34))*(dconjg(inpup(idc2,idk1))*inpup(idc1,idk2)*
     -    (-(dconjg(yup(10))*yup(16))-yup(2)*yup(25)+dotup(idb1,idb1)*
     -    (yup(19)+yup(31)))+(-(dconjg(yup(24))*yup(20))+
     -    dotup(idb1,idb1) *yup(34))*(yup(46) + yup(50))) + 
     -    dotup(idc1,idc1)* (yup(2)*yup(8)*yup(25) + dconjg(yup(10))*
     -    (yup(8)*yup(16) - yup(2)*(yup(19) + yup(31)))
     -     + dconjg(yup(35))*yup(35)*(yup(46) + yup(50))))
	bfun(4,ii,kk)=4*(dotup(idb1,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(inpup(idq0,idk1))**2*inpup(idq0,idk1)*inpup(idq0,idk2) 
     -    - dconjg(yup(13))*yup(3) -dconjg(yup(17))*yup(6))) + 
     -    dconjg(yup(34))*inpup(idc1,idk2)*(-(dconjg(inpup(idq0,idk1))*
     -    (dconjg(yup(23))+dconjg(yup(24)))) + dconjg(inpup(idc2,idk1))
     -    *yup(31))) +dconjg(yup(35))*(dotup(idc1,idc1)*
     -    (dconjg(inpup(idb2,idk1))*inpup(idq0,idk2)*yup(21) + 
     -    dconjg(yup(13))*yup(82)) +dconjg(yup(34))*inpup(idc1,idk2)*
     -    (-(dconjg(inpup(idc2,idk1))*yup(51))+dconjg(inpup(idb2,idk1))
     -    *yup(83))))
c...helicity(17)=(-,-,-,-,-,-)
        do mm=1,4
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(2)=m(+,+,+,+,+,-)
        kk=2
	  if(iqqbar.eq.0) then
        bfun(1,ii,kk)=(4*dconjg(yup(10))*yup(1)*((-(dconjg(yup(24))
     -     *dconjg(yup(34))) +dconjg(yup(5))*yup(2))*yup(20) + 
     -     dotup(idb1,idb1)*(-(dconjg(yup(5))*yup(5))+dconjg(yup(34)) 
     -     *yup(34)) + dotup(idc1,idc1)*(-(dconjg(yup(13))*yup(13)) + 
     -     dconjg(yup(35))*yup(35)) + dconjg(yup(18))*yup(13)*
     -     (-dconjg(yup(38)) + yup(36))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*(dotup(idc1,idc1)*
     -     (yup(2)*(-yup(19)+yup(29))*yup(31)+yup(9)*(-(dconjg(yup(13))
     -     *yup(13))+dconjg(yup(35))*yup(35)))+yup(9)*(dconjg(yup(34))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34)) + 
     -     dconjg(yup(18))*yup(13)*yup(36))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*yup(1)*(dotup(idb1,idb1)*(dconjg(yup(18))
     -     *yup(19)*(yup(28) - yup(31)) + dconjg(yup(16))*
     -     (-(dconjg(yup(5))*yup(5)) + dconjg(yup(34))*yup(34))) + 
     -     dconjg(yup(16))*((-(dconjg(yup(24))*dconjg(yup(34))) + 
     -     dconjg(yup(5))*yup(2))*yup(20) + dconjg(yup(35))* 
     -     dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*(dotup(idb1,idb1)*(dotup(idc1,idc1)
     -     *yup(8)*yup(19)*yup(31)+yup(9)*(dconjg(yup(18))*yup(19)
     -     *yup(28) +dconjg(yup(16))*dconjg(yup(34))*yup(34)))
     -     +dconjg(yup(16))*(-(dconjg(yup(24))*dconjg(yup(34))*yup(9)*
     -     yup(20)) +dotup(idc1,idc1)*(yup(2)*yup(29)*yup(31) + 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
	  bfun(5,ii,kk)=(2*dconjg(yup(10))*yup(1)*yup(27)*yup(30))/
     -	 dconjg(yup(8))
	  bfun(8,ii,kk)=(4*((-(dconjg(yup(10))*dconjg(yup(24))) + 
     -     dconjg(yup(14))*dotup(idb1,idb1))*(dconjg(yup(18))*yup(9)
     -     *yup(14)+2*dotup(idc1,idk1)*yup(4)*yup(27))+(dconjg(yup(12)) 
     -     - dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (2*dconjg(yup(35))*dotup(idc1,idk1)*yup(8)*yup(35) + 
     -     (dconjg(yup(12))-dconjg(yup(18)))*(yup(2)*yup(29)*yup(31)+ 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(yup(15))*(dotup(idb1,idb1)*yup(5)
     -     -yup(2)*yup(20))*yup(28)+4*dconjg(yup(35))*dotup(idc1,idc1)*
     -     yup(8)*yup(35))/dconjg(yup(8))
c...helicity(18)=m(-,-,-,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(3)=(+,+,+,+,-,+)
        kk=3
	  if(iqqbar.eq.0) then
        bfun(1,ii,kk)=(4*(dconjg(yup(1))*(dotup(idb1,idb1)*yup(11)*
     -     yup(19)*yup(27)+dconjg(yup(34))*yup(10)*(-(dconjg(yup(24))
     -     *yup(20)) + dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*
     -     (dconjg(yup(8))*dotup(idb1,idb1)*yup(19)*yup(31) + yup(10)*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(1))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
        bfun(2,ii,kk)=(-4*dconjg(yup(9))*(dotup(idb1,idb1)*(yup(4)
     -     *(dconjg(yup(8))*dconjg(yup(33))-dconjg(yup(4))*yup(10))+ 
     -     yup(11)*yup(19)*yup(27) +dconjg(yup(34))*yup(10)*yup(34))+ 
     -     yup(10)*(dconjg(yup(24))*(-(dconjg(yup(34))*yup(20))+yup(4) 
     -     *yup(25))+dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/yup(8)
	  bfun(3,ii,kk)=(-4*yup(16)*(dconjg(yup(1))*(dconjg(yup(17))*
     -     yup(11)*yup(32)+dconjg(yup(34))*(-(dconjg(yup(24))*yup(20))+ 
     -     dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*(yup(25)*
     -     (-yup(19)+yup(30))*yup(31)+dconjg(yup(1))*(-(dconjg(yup(17))
     -     *yup(17))+dconjg(yup(35))*yup(35)))))/yup(8)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*yup(16)*(-(yup(4)*
     -     (dconjg(yup(4))*dotup(idb1,idb1) + (-dconjg(yup(24)) +
     -     dconjg(yup(33)))*yup(25))) +dconjg(yup(17))*
     -     (-(dotup(idc1,idc1)*yup(17)) + yup(11)*yup(32))
     -     + dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34)) + dconjg(yup(35))*dotup(idc1,idc1)
     -     *yup(35)))/yup(8)
	bfun(5,ii,kk)=(2*dconjg(yup(9))*yup(16)*yup(28)*yup(29))/yup(8)
	bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(24))*((yup(12)
     -     - yup(18))*(-(dconjg(yup(34))*yup(20)) +yup(4)*yup(25)) + 
     -     2*dotup(idc1,idk1)*yup(14)*yup(29))+dotup(idb1,idb1)*(yup(12) 
     -     -yup(18))*(-(dconjg(yup(4))*yup(4))+dconjg(yup(34))*yup(34))+ 
     -     dotup(idc1,idc1)*(-(yup(16)*yup(19)*yup(29)) +dconjg(yup(35)) 
     -     *(yup(12) - yup(18))*yup(35))))/yup(8)
	bfun(9,ii,kk)=(4*dconjg(yup(5))*yup(11)*yup(20)*yup(30))/yup(8)
c...helicity(19)=(-,-,-,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(4)=(+,+,+,+,-,-)
        kk=4
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(10))*(dotup(idc1,idc1)*
     -     ((dconjg(yup(3))*dconjg(yup(12)) - dconjg(yup(1))
     -     *dconjg(yup(13)))*yup(13)+yup(25)*yup(30)*yup(31)+ 
     -     dconjg(yup(1))*dconjg(yup(35))*yup(35)) +dconjg(yup(1))*
     -     (dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34)) +dconjg(yup(18))*yup(13) 
     -     *yup(36))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*dconjg(yup(10))*(dconjg(yup(24))
     -     *(dconjg(yup(34))*yup(20)+yup(4)*(dconjg(yup(7))-yup(25)))+ 
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) + dotup(idc1,idc1)*(dconjg(yup(13))*yup(13) - 
     -     dconjg(yup(35))*yup(35)) - dconjg(yup(18))*yup(13)
     -     *yup(36)))/xup(2)
	  bfun(3,ii,kk)=(-4*(dconjg(yup(1))*dconjg(yup(18))*
     -     dotup(idb1,idb1)*yup(19)*yup(28) +dconjg(yup(16))*
     -     (dconjg(yup(1))*dconjg(yup(34))*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(1))*dconjg(yup(35))
     -     *yup(35)))))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*(dotup(idb1,idb1)*
     -     ((dconjg(yup(7))*dconjg(yup(14)) - dconjg(yup(4))
     -     *dconjg(yup(16)))*yup(4) +dconjg(yup(18))*yup(19)*yup(28)+ 
     -     dconjg(yup(16))*dconjg(yup(34))*yup(34)) +dconjg(yup(16))*
     -     (dconjg(yup(24))*(-(dconjg(yup(34))*yup(20))+yup(4)*yup(25))+ 
     -     dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/xup(2)
	bfun(5,ii,kk)=(2*dconjg(yup(9))*dconjg(yup(10))*yup(28)*yup(30))
     -	 /xup(2)
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*(dconjg(yup(24))*(dconjg(yup(10))
     -     *yup(14) -dconjg(yup(34))*yup(20) + yup(4)*yup(25)) - 
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) + dconjg(yup(14))
     -     *yup(14) -dconjg(yup(34))*yup(34)) +dconjg(yup(35))* 
     -     dotup(idc1,idc1)*yup(35)))/xup(2)
        bfun(9,ii,kk)=(-4*dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -    inpup(idk1,idk2)*yup(28)*yup(32))/xup(2)
c...helicity(20)=(-,-,-,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c**********************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*(-(dconjg(yup(24))*dconjg(yup(34))*
     -    (yup(20)*yup(53) - yup(25)*yup(54))) +dotup(idb1,idb1)*
     -    (dconjg(yup(8))*dotup(idc1,idc1)*yup(30) +dconjg(yup(34))*
     -    (-(dconjg(inpup(idq0,idk2))*dconjg(yup(24))*inpup(idc1,idk1))+
     -    dconjg(inpup(idc2,idk2))*inpup(idb1,idk1)*yup(43) + yup(15)*
     -    (yup(55) - yup(56)))) +dconjg(yup(35))*dotup(idc1,idc1)*
     -    (yup(35)*yup(53) + dconjg(yup(3))*yup(57)))
        bfun(2,ii,kk)=-4*(dotup(idb1,idb1)*(-(dconjg(yup(8))
     -    *dotup(idc1,idc1)*yup(29)) +dconjg(yup(34))*
     -    (dconjg(inpup(idq0,idc2))*inpup(idc1,idk1)*yup(16) + 
     -    dconjg(inpup(idc2,idk2))*inpup(idq0,idk1)*yup(20) + 
     -    dconjg(yup(4))*(yup(54) - yup(58)))) +dconjg(yup(35))*
     -    (-(dotup(idc1,idc1)*yup(17)) + yup(11)*yup(32))*yup(59) +
     -    (dconjg(yup(24))*dconjg(yup(34))*yup(20) - dconjg(yup(35))
     -    *dotup(idc1,idc1)*yup(35))*yup(60))
	  bfun(3,ii,kk)=-4*((dconjg(yup(9))*yup(1) + dconjg(yup(18))*
     -    yup(11))*(-(dconjg(yup(8))*dotup(idb1,idb1))+yup(16)*yup(25))+ 
     -    dconjg(yup(34))*(dconjg(inpup(idc2,idk2))*inpup(idc1,idk1)*
     -    (-(dconjg(yup(10))*yup(16))-yup(2)*yup(25)+dotup(idb1,idb1)*
     -    (yup(19) + yup(31))) +(-(dconjg(yup(24))*yup(20)) + 
     -    dotup(idb1,idb1)*yup(34))*(yup(53) + yup(60))) + 
     -    dotup(idc1,idc1)*(dconjg(yup(8))*yup(2)*yup(25) + 
     -    yup(16)*(dconjg(yup(8))*dconjg(yup(10)) -yup(25)*(yup(19) + 
     -    yup(31))) +dconjg(yup(35))*yup(35)*(yup(53) + yup(60))))
	  bfun(4,ii,kk)=-4*(dconjg(inpup(idq0,idk1))*
     -    dconjg(inpup(idq0,idk2))*dotup(idb1,idb1)*dotup(idc1,idc1)*
     -    inpup(idq0,idk1)**2 +dotup(idb1,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(yup(3))*yup(13) +dconjg(yup(6))*yup(17))) + 
     -    dconjg(inpup(idc2,idk2))*dconjg(yup(34))*(dconjg(yup(3))*
     -    inpup(idc1,idb2) -inpup(idc1,idk1)*yup(31))+ dconjg(yup(6))
     -    *yup(11)*yup(32))+dconjg(yup(35))*(-(dconjg(inpup(idq0,idk2))
     -    *dconjg(yup(38))*dotup(idc1,idc1)*inpup(idb1,idk1)) + 
     -    dconjg(inpup(idq0,idk2))*dotup(idc1,idc1)*inpup(idb1,idk1)
     -    *yup(61) - dconjg(inpup(idc2,idk2))*dconjg(yup(34))*
     -    (-(inpup(idc1,idk1)*yup(51)) + inpup(idb1,idk1)*yup(62))))
c...helicity(20)=(-,-,-,-,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(5)=(+,+,+,-,+,+)
        kk=5
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (-(yup(1)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19)) - 
     -     (dconjg(inpup(idc1,idq0))*yup(5)*yup(10) - 
     -     dconjg(inpup(idk2,idk1))*yup(19)*yup(27))*yup(31)) - 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idb2,idk1))
     -     *yup(31)*yup(32) + dconjg(inpup(idb2,idq0))*yup(1)*
     -     (yup(22) + yup(32)))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (yup(9)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19)+dconjg(yup(4))
     -     *yup(4)*(-(dconjg(inpup(idq0,idk1))*yup(10)) - 
     -     dconjg(inpup(idk2,idk1))*yup(19)))+dconjg(yup(35))*yup(10)*
     -     (dconjg(inpup(idb2,idk1))*yup(22)*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+yup(32)))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*yup(16)*
     -     (dconjg(inpup(idc1,idq0))*dotup(idb1,idb1)*yup(5)*
     -     yup(31)+ dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*yup(31)
     -     *yup(32) +dconjg(inpup(idb2,idq0))*yup(1)*(yup(22) +
     -     yup(32))) +dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))
     -     *yup(31)*yup(32)) +dconjg(inpup(idq0,idk2))*yup(1)*
     -     (yup(22) + yup(32)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(16)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*dotup(idb1,idb1)
     -     *yup(4)) +dconjg(yup(35))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31) +dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+
     -     yup(32))) +dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))
     -     *yup(22)*yup(31)) +dconjg(inpup(idq0,idk2))*yup(9)*
     -     (yup(22) + yup(32)))))/xup(1)
	bfun(5,ii,kk)=(2*dconjg(inpup(idk1,idq0))*pmomup(5,idc1)*yup(16)
     -    *(yup(27) + yup(28))*yup(29))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(14))*yup(9)*(dotup(idb1,idb1)*yup(14) - yup(16)
     -     *yup(23))) -dconjg(inpup(idq0,idk1))*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*yup(4)*(yup(12)-yup(18)))-dconjg(yup(14)) 
     -     *yup(12)*(dotup(idb1,idb1)*yup(14) - yup(16)*yup(23)))-
     -     dconjg(yup(35))*((yup(12)-yup(18))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31)+dconjg(inpup(idb2,idq0))*yup(9)*(yup(22) +
     -     yup(32))) +2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     yup(27)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -     pmomup(5,idc1)*yup(28)*yup(35))/xup(1)
c...helicity(21)=(-,-,-,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*******************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))*
     -     dotup(idb1,idb1)*((dconjg(yup(10))-dconjg(inpup(idq0,idk1)
     -     )*inpup(idk2,idk1))*(yup(27) + yup(28))+ (dconjg(yup(11))
     -     + dconjg(yup(18)))*yup(30) +inpup(idb1,idk2)*
     -     (dconjg(inpup(idq0,idc2))*yup(37)-dconjg(inpup(idc1,idq0))
     -     *yup(43)))) +dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     ((yup(22) + yup(32))*yup(46) + (dconjg(yup(11)) + 
     -     dconjg(yup(18)))*yup(47)))
        bfun(2,ii,kk)=4*pmomup(5,idc1)*(-(dotup(idb1,idb1)*yup(8)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2))
     -     *yup(23))) -dconjg(yup(35))*(yup(22) + yup(32))*
     -     (-(dconjg(inpup(idb2,idk1))*dconjg(yup(12))) + 
     -     dconjg(inpup(idb2,idq0))*yup(50)))
	  bfun(3,ii,kk)=4*pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))
     -    *dconjg(yup(1))*dconjg(yup(10))*yup(2) - 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(1))*dotup(idb1,idb1)*
     -    yup(8) -dconjg(inpup(idq0,idk2))*dconjg(yup(11))*
     -    dotup(idb1,idb1)*yup(8) - dconjg(inpup(idq0,idk1))*
     -    dconjg(yup(10))*dconjg(yup(11))*yup(16) - 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(10))*dconjg(yup(18))*
     -    yup(16) +dconjg(inpup(idq0,idk1))*dconjg(yup(11))*
     -    dotup(idb1,idb1)*yup(19) + dconjg(inpup(idq0,idk1))*
     -    dconjg(yup(18))*dotup(idb1,idb1)*yup(19) - 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(11))*yup(2)*yup(25) - 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(18))*yup(2)*yup(25) + 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(11))*dotup(idb1,idb1)
     -    *yup(31) +dconjg(inpup(idq0,idk1))*dconjg(yup(18))*
     -    dotup(idb1,idb1)*yup(31) + dconjg(inpup(idc1,idq0))*
     -    (dconjg(yup(10))*yup(2)-dotup(idb1,idb1)*yup(8))*yup(40)+ 
     -    dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*dconjg(yup(11))*
     -    yup(26) + dconjg(inpup(idb2,idq0))*(yup(22) + yup(32))*
     -    (yup(46) + yup(50))) +dconjg(inpup(idc1,idq0))*
     -    (dconjg(yup(10))*yup(2) -dotup(idb1,idb1)*yup(8))*yup(79))
	  bfun(4,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -    (-(dconjg(inpup(idq0,idk1))**2*(dconjg(yup(5))*
     -    inpup(idc1,idk2) +dconjg(yup(4))*inpup(idc2,idk2))) + 
     -    (dconjg(yup(11))+dconjg(yup(18)))*(dconjg(inpup(idb2,idq0))*
     -    yup(3)+dconjg(inpup(idb1,idq0))*yup(6)))+ (dconjg(yup(11))
     -    + dconjg(yup(18)))*dconjg(yup(35))*(-(dconjg(inpup(idb2,idk1)
     -    )*yup(21)) +dconjg(inpup(idb2,idq0))*yup(82)))
c...helicity(21)=(-,-,-,+,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(6)=(+,+,+,-,+,-)
        kk=6
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (dconjg(inpup(idb2,idq0))*yup(1)*(-((dconjg(yup(11))
     -     + dconjg(yup(18)))*yup(13)) +dconjg(yup(35))*(yup(22) 
     -     + yup(32))) +yup(31)*(dconjg(inpup(idc1,idq0))*
     -     dotup(idb1,idb1)*yup(5) +dconjg(inpup(idb2,idk1))*
     -     (-(dconjg(yup(18))*yup(13)) +dconjg(yup(35))*yup(32) 
     -     ))))/dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*dotup(idb1,idb1)
     -     *yup(4)) +dconjg(inpup(idb2,idk1))*(-(dconjg(yup(11))
     -     *yup(13)) +dconjg(yup(35))*yup(22))*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(-((dconjg(yup(11)) +
     -     dconjg(yup(18)))*yup(13)) + dconjg(yup(35))*(yup(22) + 
     -     yup(32)))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(inpup(idc1,idq0))*dconjg(yup(16))*yup(5) - 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(18))*yup(19))*yup(31)+ 
     -     dconjg(yup(16))*dconjg(yup(35))*(dconjg(inpup(idb2,idk1))
     -     *yup(31)*yup(32) +dconjg(inpup(idb2,idq0))*yup(1)*
     -     (yup(22) + yup(32)))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(11))*dotup(idb1,idb1)*yup(19)*yup(31) - 
     -     dconjg(yup(16))*(-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     dotup(idb1,idb1)*yup(4)) +dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*yup(22)*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+yup(32))))))/
     -     dconjg(yup(8))
	  bfun(5,ii,kk)=(2*dconjg(inpup(idk1,idq0))*dconjg(yup(10))*
     -     pmomup(5,idc1)*(yup(27) + yup(28))*yup(30))/dconjg(yup(8))
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18)))*
     -     dotup(idb1,idb1)*yup(4)) - 2*dotup(idc1,idk1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(yup(35))*yup(8)*yup(22)+ 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(10))*yup(27)*yup(30)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*yup(22)*yup(31) + dconjg(inpup(idb2,idq0))
     -     *yup(9)*(dconjg(yup(12))*(yup(22)+yup(32))-dconjg(yup(18))*
     -     (yup(22) - yup(26) + yup(32))))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(11))*
     -     dconjg(yup(35))*pmomup(5,idc1)*yup(35))/dconjg(yup(8))
c...helicity(22)=(-,-,-,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
        end if

c...helicity(7)=(+,+,+,-,-,+)
        kk=7
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idc1)*
     -     (-(dotup(idb1,idb1)*(-(dconjg(inpup(idq0,idk1))*
     -     dconjg(yup(8))) +dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19))+ yup(10)*
     -     (dconjg(inpup(idq0,idk1))*yup(25)*yup(30) - 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     (yup(22) + yup(32)))))/yup(8)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*(dotup(idb1,idb1)
     -     *yup(19)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15)-dconjg(inpup(idq0,idk2)) 
     -     *yup(31)) + dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     yup(10)*(-yup(21) + yup(22) + yup(32))))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idc1)*yup(16)*
     -     (dconjg(inpup(idq0,idk1))*yup(25)*(yup(19) - yup(30)) + 
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(17)) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35)))*
     -     (yup(22) + yup(32))))/yup(8)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(17))+dconjg(inpup(idb2,idq0))*dconjg(yup(35)))*
     -     pmomup(5,idc1)*yup(16)*(yup(21) - yup(22) - yup(32)))/yup(8)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(14))*(dotup(idb1,idb1)
     -     *yup(14) - yup(16)*yup(23)))+ dconjg(inpup(idb2,idq0))
     -     *dconjg(yup(35))*(-2*dotup(idc1,idk1)*yup(17)+(yup(12)-
     -     yup(18))*(yup(21)- yup(22) - yup(32)))))/yup(8)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idq0))*pmomup(5,idc1)*
     -     (dconjg(yup(5))*yup(20) + yup(25)*yup(28))*yup(30))/yup(8)
c...helicity(23)=(-,-,-,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(8)=(+,+,+,-,-,-)
        kk=8
        if (iqqbar.eq.0) then
	bfun(1,ii,kk)=(-4*dconjg(yup(1))*dconjg(yup(10))*pmomup(5,idc1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))
     -     *inpup(idk2,idk1)*yup(13) - dconjg(inpup(idq0,idk1))*yup(25)*
     -     yup(30) + dconjg(inpup(idb2,idq0))*(-((dconjg(yup(11)) +
     -     dconjg(yup(18)))*yup(13)) +dconjg(yup(35))*(yup(22) 
     -     + yup(32)))))/xup(2)
        bfun(2,ii,kk)=(-4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idc1)*((dconjg(yup(11)) -
     -     dconjg(yup(12)) +dconjg(yup(18)))*yup(13) + 
     -      dconjg(yup(35))*(yup(21) - yup(22) - yup(32))))/xup(2)
	bfun(3,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(16))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*yup(25)*yup(30)) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     (yup(22) + yup(32))))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     dconjg(yup(16))*dconjg(yup(35))*pmomup(5,idc1)*
     -     (yup(21) - yup(22) - yup(32)))/xup(2)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*dconjg(yup(35))
     -     *pmomup(5,idc1)*(yup(21)-yup(22)+yup(26)-yup(32)))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(24)=(-,-,-,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*********************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))
     -     *dotup(idb1,idb1)*((dconjg(inpup(idq0,idk2))*
     -     inpup(idk2,idk1) +yup(25))*(yup(27) + yup(28)) + 
     -     (dconjg(yup(1))+dconjg(yup(9)))*yup(30)+inpup(idb1,idk1)*
     -     (dconjg(inpup(idq0,idc2))*yup(37)-dconjg(inpup(idc1,idq0))
     -     *yup(43)))) +dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     ((yup(22) + yup(32))*yup(53) + (dconjg(yup(1)) + 
     -     dconjg(yup(9)))*yup(57)))
        bfun(2,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(yup(8))*
     -     dotup(idb1,idb1)*(-(dconjg(inpup(idc1,idq0))*yup(20))+ 
     -     dconjg(inpup(idq0,idc2))*yup(23)))-dconjg(yup(35))*(yup(22)
     -     + yup(32))*(-(dconjg(inpup(idb2,idk2))*dconjg(yup(7))) + 
     -     dconjg(inpup(idb2,idq0))*yup(60)))
	  bfun(3,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(1))*dconjg(yup(8))*dotup(idb1,idb1)) - 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(8))*dconjg(yup(11))*
     -     dotup(idb1,idb1) -dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     dconjg(yup(10))*yup(16) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(9))*dconjg(yup(10))*yup(16) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(1))*dotup(idb1,idb1)*
     -     yup(19) +dconjg(inpup(idq0,idk2))*dconjg(yup(9))*
     -     dotup(idb1,idb1)*yup(19) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(1))*yup(2)* yup(25) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(9))*yup(2)*yup(25) + dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(11))*yup(16)*yup(25)+ dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(1))*dotup(idb1,idb1)*yup(31) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(9))*dotup(idb1,idb1)
     -     *yup(31) +dconjg(inpup(idc1,idq0))*(-(dconjg(yup(8))
     -     *dotup(idb1,idb1)) + yup(16)*yup(25))*yup(40) + 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk2))*dconjg(yup(1))*
     -     yup(21) + dconjg(inpup(idb2,idq0))*(yup(22) + yup(32))
     -     *(yup(53) + yup(60))) +dconjg(inpup(idc1,idq0))*
     -     (-(dconjg(yup(8))*dotup(idb1,idb1)) + yup(16)*yup(25))
     -     *yup(79))
	  bfun(4,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idc1)*
     -    (dotup(idb1,idb1)*(-((dconjg(yup(1))+dconjg(yup(9)))*yup(31))+
     -    dconjg(yup(6))*(yup(22) + yup(32)) + dconjg(yup(3))*
     -    (dconjg(yup(36)) + yup(42))) + dconjg(yup(35))*
     -    ((dconjg(yup(1)) + dconjg(yup(9)))*yup(51) +inpup(idb1,idk1)*
     -    (-(dconjg(inpup(idc1,idq0))*yup(62))+dconjg(inpup(idq0,idc2))
     -    *yup(64))))
c...helicity(24)=(-,-,-,+,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
        end if

c...helicity(9)=(+,-,-,+,+,+)
        kk=9
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*yup(1)*
     -     (dconjg(yup(13))*yup(13)*(yup(11)+ yup(18)) +dconjg(yup(8))
     -     *yup(7)*yup(30) +yup(10)*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(13)*(dconjg(yup(3))*yup(7) +dconjg(yup(13))*(yup(11) 
     -     +yup(18)))+yup(10)*(-(dconjg(yup(3))*yup(3))-dconjg(yup(6)) 
     -     *yup(6) +(yup(27) + yup(28))*(yup(29) + yup(30)))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(1)*
     -     (dconjg(yup(6))*yup(7)*yup(17) +yup(11)*yup(19)*yup(29) + 
     -     yup(18)*yup(19)*yup(29) +yup(16)*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/xup(1)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (-((yup(11) - yup(12) + yup(18))*yup(19)*yup(29)) + 
     -     yup(16)*(dconjg(yup(3))*yup(3)- dconjg(inpup(idb1,idq0))
     -     *inpup(idq0,idk1)*yup(6) - (yup(27) + yup(28))*(yup(29)
     -     + yup(30)))))/xup(1)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(yup(29)+yup(30))*(yup(19)-yup(27)
     -     -yup(28)+yup(31)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(25)=(-,+,+,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c**************************************************************
	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dotup(idb1,idb1)*yup(8)*(yup(27) + yup(28)) + 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(22))*inpup(idb1,idk2)
     -     *yup(30) +dconjg(inpup(idq0,idk1))*dconjg(yup(32))*
     -     inpup(idb1,idk2)*yup(30) +dconjg(yup(12))*yup(1)*yup(30) + 
     -     dconjg(yup(12))*yup(9)*yup(30) + dconjg(inpup(idq0,idc2))
     -     *inpup(idb1,idk2)*yup(7)*yup(37)+dconjg(inpup(idc2,idk1))*
     -     inpup(idb1,idk2)*yup(30)*yup(37) +inpup(idb1,idk2)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(7))+dconjg(inpup(idc1,idk1)) 
     -     *yup(30))*yup(43) +yup(46)*(dconjg(yup(37))*yup(37) + 
     -     yup(43)*yup(65)))
        bfun(2,ii,kk)=-4*pmomup(5,idb1)*pmomup(5,idc1)*((yup(27)+ 
     -     yup(28))*(dotup(idb1,idb1)*yup(8) + yup(30)*yup(50) + 
     -     dconjg(yup(13))*yup(66))+yup(43)*(dconjg(inpup(idc1,idk1))*
     -     inpup(idb2,idk2)*yup(35)+yup(6)*(-dconjg(yup(54))+yup(67)))+ 
     -     yup(37)*(dconjg(inpup(idc2,idk1))*inpup(idb2,idk2)*yup(35) + 
     -     yup(6)*(-yup(45) + yup(68))))
	  bfun(3,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(-(
     -     (dconjg(yup(10)) +dconjg(yup(16)))*((dconjg(yup(40))
     -     + dconjg(yup(79)))*yup(5)-yup(8)*yup(11)-yup(1)*yup(31))) + 
     -     inpup(idq0,idk2)*(dconjg(inpup(idc1,idk1))*
     -     (dconjg(yup(2))*yup(5) +dconjg(yup(16))*yup(15)) + 
     -     dconjg(inpup(idc2,idk1))*(dconjg(yup(2))*yup(4) + 
     -     dconjg(yup(16))*yup(14) -(yup(21) + yup(26))*yup(37))) - 
     -     dconjg(inpup(idc1,idk1))*inpup(idq0,idk2)*(yup(21)+ yup(26))
     -     *yup(43) + (yup(27) + yup(28))*(yup(29) + yup(30))*
     -     (yup(46) + yup(50)))
	bfun(4,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(-((yup(1)+yup(9))
     -     *(dconjg(yup(17))*yup(21) +dconjg(yup(13))*yup(38))) - 
     -     yup(8)*(yup(69)*yup(70) + yup(37)*yup(83) + yup(43)*yup(84)
     -     + yup(71)*yup(85)))
c...helicity(25)=(-,+,+,-,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(10)=(+,-,-,+,+,-)
        kk=10
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(1)*(-(dconjg(yup(8))*yup(8))+yup(19)*(yup(27)+yup(28))- 
     -     (yup(29)+yup(30))*(yup(27)+yup(28)-yup(31))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(9)*(dconjg(yup(3))*yup(3)+dconjg(yup(6))*yup(6)+(yup(27) 
     -     + yup(28))*(yup(19) - yup(29) - yup(30)) - yup(19)*yup(31)
     -     ))/dconjg(yup(8))
	bfun(3,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(1)*(yup(19) - yup(29) - yup(30))*(yup(27) + yup(28)
     -     - yup(31)))/dconjg(yup(8))
	bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(9)*(-(dconjg(yup(3))*yup(3)) -dconjg(yup(6))*yup(6) + 
     -     dconjg(inpup(idq0,idk2))*inpup(idq0,idk1)*yup(8) - (yup(27)
     -     + yup(28))*(yup(19) - yup(29) - yup(30))))/dconjg(yup(8))
	bfun(5,ii,kk)=0.0d0
	bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (dconjg(yup(14))*(dconjg(yup(10))+dconjg(yup(16)))*yup(14)- 
     -     (dconjg(yup(12)) - dconjg(yup(18)))*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/dconjg(yup(8))
	bfun(9,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(15))
     -     *yup(5) + yup(8)*yup(28))*(yup(29) + yup(30)))/dconjg(yup(8))
c...helicity(26)=(-,+,+,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(11)=(+,-,-,+,-,+)
        kk=11
        if(iqqbar.eq.0) then
	bfun(1,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(-(dconjg(yup(1))
     -     *dconjg(yup(13))*yup(13)*(yup(11)+yup(18)))-dconjg(yup(2))*
     -     (yup(18)*yup(19) + yup(10)*yup(27))*yup(31) + yup(10)*(
     -     -(dconjg(yup(1))*(yup(27) + yup(28))*(yup(29) + yup(30))) - 
     -     yup(25)*yup(27)*yup(31))))/yup(8)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(yup(4)*
     -     (-(dconjg(inpup(idc2,idk2))*dconjg(yup(8))*dconjg(yup(13))*
     -     inpup(idk1,idb2)) +dconjg(yup(4))*yup(10)*(dconjg(yup(2))
     -     + yup(25))) + dconjg(yup(9))*(dconjg(yup(13))*yup(13)
     -     *(yup(11) + yup(18)) +yup(10)*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))))/yup(8)
	bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(1))*
     -     yup(11)*yup(19)*yup(29) +yup(18)*yup(19)*(dconjg(yup(1))
     -     *yup(29)+yup(25)*yup(31))+yup(16)*(dconjg(yup(1))*(yup(27) 
     -     + yup(28))*(yup(29) + yup(30)) + dconjg(yup(2))*yup(27)*
     -     yup(31) +yup(25)*yup(27)*yup(31))))/yup(8)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idc2,idk2))*inpup(idq0,idk1)*yup(4)*yup(19)
     -     *yup(25) -dconjg(yup(9))*(yup(11) + yup(18))*yup(19)*
     -     yup(29)+yup(16)*(-(dconjg(yup(4))*yup(4)*(dconjg(yup(2))
     -     + yup(25))) -dconjg(yup(9))*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))))/yup(8)
	bfun(5,ii,kk)=(-2*dconjg(yup(8))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (yup(27) + yup(28))*(yup(29) + yup(30)))/yup(8)
	bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(yup(12) - yup(18))*(dconjg(yup(2)) + yup(25)) + 
     -     (dconjg(yup(9))*yup(18)*yup(19) + (2*dconjg(yup(8))
     -     *dotup(idc1,idk1)+dconjg(yup(9))*(yup(12) - yup(18)))*
     -     (yup(27) + yup(28)))*(yup(29) + yup(30))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(27)=(-,+,+,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(12)=(+,-,-,+,-,-)
        kk=12
        if(iqqbar.eq.0) then
	bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(yup(1))*yup(19)*(yup(27)+yup(28))) +dconjg(yup(1)) 
     -     *yup(27)*yup(29) +dconjg(yup(1))*yup(28)*yup(29) + 
     -     dconjg(yup(1))*yup(27)*yup(30)+dconjg(yup(1))*yup(28)*yup(30)
     -     +dconjg(yup(2))*yup(27)*yup(31) - dconjg(yup(7))*yup(27)*
     -     yup(31) +yup(25)*yup(27)*yup(31)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*yup(4)*(dconjg(yup(2))-dconjg(yup(7))
     -     + yup(25)) -dconjg(yup(9))*(yup(27) + yup(28))*(yup(19) -
     -     yup(29) - yup(30))))/xup(2)
	bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(yup(1))*yup(19)*(yup(27)+yup(28)))+dconjg(yup(1))* 
     -     yup(27)*yup(29)+dconjg(yup(1))*yup(28)*yup(29)+dconjg(yup(1))
     -     *yup(27)*yup(30)+dconjg(yup(1))*yup(28)*yup(30)+
     -     dconjg(yup(2))*yup(27)*yup(31)-dconjg(yup(7))*yup(27)*yup(31) 
     -     +yup(25)*yup(27)*yup(31)))/xup(2)
	bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*yup(4)*(dconjg(yup(2)) - dconjg(yup(7))
     -     + yup(25)) - dconjg(yup(9))*(yup(27) + yup(28))*
     -     (yup(19) - yup(29) - yup(30))))/xup(2)
	bfun(5,ii,kk)=0.0d0
	bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *dconjg(yup(10))*dconjg(yup(12))*yup(14) - dconjg(yup(9))
     -     *dconjg(yup(10))*dconjg(yup(14))*yup(14) + (dconjg(yup(4))
     -     *dconjg(yup(12)) -dconjg(yup(9))*dconjg(yup(14)))*
     -     dconjg(yup(16))*yup(14) + dconjg(yup(4))*dconjg(yup(12))
     -     *yup(4)*(dconjg(yup(2)) + yup(25)) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(29) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(30) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(27)*(yup(29)+yup(30)))+4*dconjg(yup(18)) 
     -     *pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(dconjg(yup(2))+yup(25))+dconjg(yup(9))*(yup(27)
     -     + yup(28))*(yup(29) + yup(30))))/xup(2)
	bfun(9,ii,kk)=0.0d0
c...helicity(28)=(-,+,+,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*****************************************************************
	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(8))*dotup(idb1,idb1)*(yup(27) + yup(28)) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(22))*
     -     inpup(idb1,idk1)*yup(30) +dconjg(inpup(idq0,idk2))* 
     -     dconjg(yup(32))*inpup(idb1,idk1)*yup(30) + 
     -     dconjg(yup(7))*yup(11)*yup(30) + dconjg(yup(7))*yup(18)
     -     *yup(30)-dconjg(inpup(idq0,idc2))*inpup(idk1,idb1)*yup(12)*
     -     yup(37) + dconjg(inpup(idc2,idk2))*inpup(idb1,idk1)*yup(30)
     -     *yup(37) + (dconjg(inpup(idc1,idq0))*inpup(idk1,idb1)*
     -     yup(12)+ dconjg(inpup(idc1,idk2))*inpup(idb1,idk1)*yup(30))
     -     *yup(43)+yup(53)*(dconjg(yup(37))*yup(37)+yup(43)*yup(65)))
        bfun(2,ii,kk)=-4*pmomup(5,idb1)*pmomup(5,idc1)*((yup(27) +
     -     yup(28))* (dconjg(yup(8))*dotup(idb1,idb1) - 
     -     dconjg(yup(2))*yup(12) + yup(30)*yup(60)) + yup(43)*(-
     -     (dconjg(inpup(idc1,idk2))*inpup(idk1,idb2)*yup(35)) + 
     -     yup(17)*(dconjg(yup(49)) - yup(72))) + yup(37)*
     -     (-(dconjg(inpup(idc2,idk2))*inpup(idk1,idb2)*yup(35)) + 
     -     yup(17)*(-yup(56) + yup(73))))
	  bfun(3,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(yup(79))*yup(15)*(dconjg(yup(2)) + yup(25))) + 
     -     (dconjg(yup(8))*yup(1) - dconjg(yup(40))*yup(15) +
     -     yup(11)*yup(19))*(dconjg(yup(2))+yup(25))+inpup(idq0,idk1)*
     -     (dconjg(inpup(idc1,idk2))* (dconjg(yup(2))*yup(5) + 
     -     dconjg(yup(16))*yup(15)) + dconjg(inpup(idc2,idk2))*
     -     (dconjg(yup(2))*yup(4) +dconjg(yup(16))*yup(14) - 
     -     (yup(21) + yup(26))*yup(37))) - dconjg(inpup(idc1,idk2))
     -     *inpup(idq0,idk1)*(yup(21) + yup(26))*yup(43) + 
     -     (yup(27) + yup(28))*(yup(29)+yup(30))* (yup(53)+yup(60)))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(2))
     -     + yup(25))*((yup(11) + yup(18))*yup(31) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idq0,idb1)*yup(43) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idq0,idb2)*yup(69) + 
     -     dconjg(inpup(idc2,idk2))* (inpup(idq0,idb1)*yup(37) + 
     -     inpup(idq0,idb2)*yup(71)))
c...helicity(28)=(-,+,+,-,+,+)
        do mm=1,4
	     bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(13)=(+,-,-,-,+,+)
        kk=13
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(10)*yup(31)) - 
     -     dconjg(yup(13))*dotup(idc1,idc1)*yup(13)*
     -     (-(dconjg(inpup(idq0,idk2))*yup(1)) + dconjg(
     -     inpup(idk2,idk1))*yup(31)) -yup(1)*( dconjg(
     -     inpup(idc1,idk2))*yup(19)*yup(24)+dconjg(inpup(idc1,idq0))
     -     *yup(10)*(yup(23)+yup(24)))*yup(34)))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*yup(9)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*dotup(idc1,idc1)
     -     *yup(13)) + dconjg(yup(4))*(dconjg(inpup(idk2,idk1))*yup(19)
     -     *yup(24)+dconjg(inpup(idq0,idk1))*yup(10)*(yup(23)+yup(24)))
     -     +(dconjg(inpup(idc1,idk2))*yup(19)*yup(24) + 
     -     dconjg(inpup(idc1,idq0))*yup(10)*(yup(23)+yup(24)))*yup(34)))
     -     /xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*yup(1)*yup(17)*yup(19) + 
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(16) + 
     -     dconjg(inpup(idk2,idk1))*yup(19)*yup(29))*yup(31)) + 
     -     yup(1)*(dconjg(inpup(idc1,idk2))*yup(19)*yup(23) + 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))
     -     *yup(34)))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(yup(4))*(dconjg(inpup(idk2,idk1))*yup(19)*yup(23) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*(yup(23) + yup(24))) + 
     -     (dconjg(inpup(idc1,idk2))*yup(19)*yup(23) + 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))*
     -     yup(34)))/xup(1)
	bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*yup(9)
     -     *yup(28)*(yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*(yup(12) - yup(18))*
     -     (yup(23)+yup(24))*(dconjg(inpup(idq0,idk1))*dconjg(yup(4))+ 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(14)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*pmomup(5,idb1)*
     -     yup(27)*yup(28)*(yup(29) + yup(30)))/xup(1)
c...helicity(29)=(-,+,+,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*********************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idb2,idq0))*
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(35))*dotup(idc1,idc1)
     -     *inpup(idb1,idk2))+dconjg(inpup(idq0,idk1))*dconjg(yup(10))*
     -     dotup(idc1,idc1)*yup(30) + dconjg(inpup(idq0,idk1))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(10))*dconjg(yup(56))
     -     *yup(34)- dconjg(inpup(idc1,idq0))*dconjg(inpup(idq0,idk1))*
     -     dotup(idb1,idb1)*inpup(idc2,idk2)*yup(34) - 
     -     dconjg(inpup(idc1,idk1))*dconjg(yup(10))*yup(24)*yup(34) - 
     -     dconjg(inpup(idc1,idk1))*dconjg(yup(12))*yup(24)*yup(34) - 
     -     dconjg(inpup(idc1,idq0))*yup(23)*yup(34)*yup(46) + 
     -     dconjg(inpup(idq0,idk1))*yup(24)*yup(34)*yup(74))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(29)-dconjg(yup(16))*
     -     (dotup(idc1,idc1)*(dconjg(inpup(idb1,idq0))*yup(6) + 
     -     dconjg(inpup(idq0,idk1))*yup(29))+ dconjg(inpup(idc1,idk1))
     -     *yup(23)*yup(34)) - yup(34)*(dconjg(inpup(idc1,idk1))*
     -     dconjg(yup(12))*yup(23)+ dconjg(inpup(idc1,idq0))*yup(24)*
     -     yup(50) - dconjg(inpup(idb2,idq0))*dconjg(inpup(idc1,idq0))
     -     *inpup(idc2,idk2)*yup(66) - dconjg(inpup(idq0,idk1))*
     -     (-(dconjg(inpup(idc1,idq0))*dotup(idb1,idb1)*
     -     inpup(idc2,idk2)) + yup(23)*yup(67))))
	  bfun(3,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(8) - dconjg(yup(16))
     -     *dotup(idc1,idc1)*(-(dconjg(inpup(idq0,idk2))*yup(8)) + 
     -     dconjg(inpup(idq0,idk1))*(yup(19) + yup(31)))
     -     - dconjg(inpup(idq0,idk1))*dconjg(yup(10))*(-(dconjg(yup(1))
     -     *yup(9)) -dconjg(yup(11))*yup(18) + dotup(idc1,idc1)
     -     *(yup(19) + yup(31))) + dconjg(inpup(idb1,idq0))*(yup(21)
     -     + yup(26))*(dotup(idc1,idc1)*yup(8)-dconjg(inpup(idc1,idk1)) 
     -     *inpup(idc2,idk2)*yup(34)) - dconjg(yup(16))*
     -     (-(dconjg(inpup(idq0,idk1))*(dconjg(yup(1))*yup(9) + 
     -     dconjg(yup(11))*yup(18))) +dconjg(inpup(idc1,idk1)) 
     -     *yup(33)*yup(34)) + yup(34)*(-(dconjg(inpup(idc1,idk1))*
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(2))*inpup(idc2,idk2)) + 
     -     dconjg(inpup(idc1,idq0))*(yup(23) + yup(24))*
     -     (yup(46) + yup(50))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(-(dconjg(inpup(idc1,idk1))
     -     *inpup(idc2,idk2)*yup(34)*(-(dconjg(inpup(idb1,idq0))*
     -     yup(21)) - dconjg(inpup(idb2,idq0))*yup(38))) - 
     -     dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(34)*yup(70) + dconjg(inpup(idq0,idk1))
     -     *dotup(idc1,idc1)*(dconjg(yup(13))*(dconjg(yup(61)) - 
     -     dconjg(inpup(idq0,idk1))*inpup(idk1,idb2))+dconjg(yup(17))*
     -     (dconjg(inpup(idq0,idk1))*inpup(idb1,idk1) + yup(75))) - 
     -     dconjg(inpup(idb1,idq0))*dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(34)*yup(84))
c...helicity(29)=(-,+,+,+,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(14)=(+,-,-,-,+,-)
        kk=14
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk1)) 
     -     *yup(19))*yup(31) +dconjg(inpup(idc1,idq0))*yup(1)*
     -     (yup(23) + yup(24) - yup(33))*yup(34)))/dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*yup(9)*
     -     (yup(23) + yup(24) - yup(33))*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))+dconjg(inpup(idc1,idq0))*yup(34)))/
     -     dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk2)) 
     -     *yup(8))*yup(31) -dconjg(inpup(idc1,idq0))*yup(1)*
     -     (dconjg(inpup(idq0,idk2))*inpup(idc2,idk2) - yup(23)
     -     - yup(24))*yup(34)))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*yup(9)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(11))*yup(27)) + 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(4))*(yup(23)+yup(24)
     -     - yup(33)) +dconjg(inpup(idc1,idq0))*(yup(23) + yup(24))*
     -     yup(34)))/dconjg(yup(8))
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     pmomup(5,idb1)*yup(9)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(4)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(11))*
     -     pmomup(5,idb1)*yup(5)*(yup(29) + yup(30)))/dconjg(yup(8))
c...helicity(30)=(-,+,+,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
  	  end if

c...helicity(15)=(+,-,-,-,-,+)
        kk=15
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(yup(1))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dotup(idc1,idc1)*yup(13)) +dconjg(inpup(idc1,idk2)) 
     -     *yup(19)*(-(dconjg(yup(2))*yup(5))+yup(24)*yup(34)) + 
     -     dconjg(inpup(idc1,idq0))*yup(10)*(-(yup(5)*(dconjg(yup(2))
     -     + yup(25))) +(yup(23) + yup(24))*yup(34))))/yup(8)
        bfun(2,ii,kk)=(-4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(19)*yup(31) - 
     -     dconjg(yup(9))*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))
     -     *dotup(idc1,idc1)*yup(13)) +(dconjg(inpup(idc1,idk2)) 
     -     *yup(19)*yup(24) +dconjg(inpup(idc1,idq0))*yup(10)*
     -     (yup(23) + yup(24)))*yup(34))))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(inpup(idc1,idk2))*yup(19)*(-(yup(5)*yup(25)) +
     -     yup(23)*yup(34)) +dconjg(inpup(idc1,idq0))*yup(16)*
     -     (-(yup(5)*(dconjg(yup(2)) + yup(25))) +(yup(23) + yup(24)) 
     -     *yup(34))))/yup(8)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dotup(idc1,idc1)*yup(19)*yup(25)*yup(31) - 
     -     dconjg(yup(9))*(dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*
     -     yup(17)*yup(19) + (dconjg(inpup(idc1,idk2))*yup(19)*yup(23)+ 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))
     -     *yup(34))))/yup(8)
	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*yup(27)*(yup(29) + yup(30)))/yup(8)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk2))*(dconjg(yup(9))*dconjg(yup(14))
     -     *yup(18) +2*dconjg(yup(4))*dotup(idc1,idk1)*yup(27)) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(9))*(-yup(12)+yup(18))
     -     *yup(34)))/yup(8)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*pmomup(5,idb1)
     -    *yup(15)*(dconjg(yup(2))+yup(25))*yup(28))/yup(8)
c...helicity(31)=(-,+,+,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(16)=(+,-,-,-,-,-)
        kk=16
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(1))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(5)*(dconjg(yup(2))
     -     - dconjg(yup(7)) +yup(25)) - (yup(23) + yup(24)-yup(33))
     -     *yup(34)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(23) + yup(24) 
     -     - yup(33))*yup(34))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(1))*
     -     dconjg(yup(16))*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))
     -     *inpup(idk2,idk1)*yup(5)) -yup(5)*(dconjg(yup(2)) + 
     -     yup(25)) +(-(dconjg(inpup(idq0,idk2))*inpup(idc2,idk2)) + 
     -     yup(23) + yup(24))*yup(34)))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -     dconjg(yup(16))*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *inpup(idc2,idk2) -yup(23) - yup(24))*yup(34))/xup(2)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -    (dconjg(yup(12))-dconjg(yup(18)))*pmomup(5,idb1)*(yup(23)+
     -    yup(24))*yup(34))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(32)=(-,+,+,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c****************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30)) - yup(34)*(-
     -     (dconjg(inpup(idc1,idq0))*dconjg(yup(45))*yup(25)) + 
     -     dconjg(inpup(idc1,idk2))*yup(24)*(dconjg(yup(7)) + 
     -     yup(25)) +dconjg(inpup(idc1,idq0))*yup(23)*yup(53)) + 
     -     dconjg(inpup(idq0,idk2))*(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk1))
     -     + yup(25)*yup(30)) +yup(34)*(-(dconjg(inpup(idc1,idq0))*
     -     dotup(idb1,idb1)*inpup(idc2,idk1)) +yup(24)*yup(81))))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dconjg(inpup(idb1,idq0))
     -     *dconjg(yup(2))* dotup(idc1,idc1)*yup(17)) - 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(2))*dotup(idc1,idc1)
     -     *yup(29) -dconjg(inpup(idq0,idk2))**2*dotup(idc1,idc1)*
     -     inpup(idk2,idk1)*yup(29) + dconjg(inpup(idc1,idq0))*
     -     dconjg(yup(2))*dconjg(yup(45))*yup(34) - 
     -     dconjg(inpup(idc1,idq0))*dconjg(inpup(idq0,idk2))*
     -     dotup(idb1,idb1)*inpup(idc2,idk1)*yup(34) - 
     -     dconjg(inpup(idc1,idk2))*dconjg(yup(2))*yup(23)*
     -     yup(34) - dconjg(inpup(idc1,idk2))*dconjg(yup(7))*
     -     yup(23)*yup(34) + dconjg(inpup(idq0,idk2))*dconjg(yup(49))
     -     *yup(23)*yup(34) - dconjg(inpup(idc1,idq0))*yup(24)*
     -     yup(34)*yup(60))
	bfun(3,ii,kk)=4*pmomup(5,idb1)*(dconjg(yup(8))*dotup(idc1,idc1)*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(2)) + 
     -     dconjg(inpup(idb1,idq0))*(yup(21) + yup(26)))
     -     + dconjg(inpup(idq0,idk2))*(dconjg(yup(2)) + yup(25))*
     -     (dconjg(yup(1))*yup(9) +dconjg(yup(11))*yup(18) - 
     -     dotup(idc1,idc1)*(yup(19)+yup(31)))-dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(16))*(-(dconjg(yup(8))*dotup(idc1,idc1)) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*yup(34)) + 
     -     yup(34)*(-(dconjg(inpup(idb1,idq0))*
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*(yup(21)+yup(26)))+ 
     -     dconjg(inpup(idc1,idq0))*(yup(23) + yup(24))*(yup(53) + 
     -     yup(60))) -dconjg(inpup(idc1,idk2))*dconjg(yup(2))*yup(34)*
     -     yup(80))
	  bfun(4,ii,kk)=-4*pmomup(5,idb1)*(dconjg(inpup(idb2,idq0))
     -     *dotup(idc1,idc1)*yup(13)*(dconjg(yup(2)) + yup(25)) + 
     -     dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*
     -     (dconjg(yup(2)) + yup(25)) + dconjg(inpup(idq0,idk2))
     -     *dotup(idc1,idc1)*yup(25)*yup(31)-dconjg(inpup(idb1,idq0))*
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*yup(21)*yup(34)+ 
     -     dconjg(inpup(idc1,idk2))*yup(23)*yup(25)*yup(34) +
     -     dconjg(inpup(idc1,idk2))*yup(24)*yup(25)*yup(34) + 
     -     dconjg(yup(2))*(dconjg(inpup(idq0,idk2))*dotup(idc1,idc1)*
     -     yup(31) + dconjg(inpup(idc1,idk2))*(yup(23) + yup(24))
     -     *yup(34))-dconjg(inpup(idb2,idq0))*dconjg(inpup(idc1,idk2))*
     -     inpup(idc2,idk1)*yup(34)*yup(38))
c...helicity(32)=(-,+,+,+,+,+)
        do mm=1,4
	     bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if
      
        if(ibcstate.eq.1) go to 1211

c...helicity(33)=(+,+,-,+,+,+)
        kk=33
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(1)*yup(10)
     -     *(-(dconjg(yup(38))*inpup(idq0,idb1)) + inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*yup(9)*yup(10)*
     -     (inpup(idq0,idk1)*yup(2)*yup(29) + dconjg(yup(35))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36))))/xup(1)
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*yup(1)*yup(16)*
     -     (-(inpup(idq0,idb1)*(-(dconjg(yup(35))*dconjg(yup(38))) + 
     -     dconjg(yup(17))*yup(12))) +inpup(idq0,idb1)*
     -     (dconjg(yup(17))*(yup(11) + yup(18)) -dconjg(yup(35))* 
     -     (dconjg(yup(42)) + yup(36)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*yup(16)*
     -     (inpup(idq0,idk1)*(yup(2)-yup(7))*yup(29)+inpup(idq0,idb1)*
     -     (-(dconjg(yup(17))*(yup(11) + yup(18))) +dconjg(yup(35)) 
     -     *(dconjg(yup(42)) + yup(36)))))/xup(1)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(-((dconjg(yup(38))+dconjg(yup(41)))*
     -     inpup(idq0,idb1)) + inpup(idq0,idb1)*(dconjg(yup(42)) +
     -     yup(36))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(49)=(-,-,+,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*****************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(39))*inpup(idc1,idq0) +dconjg(yup(24))* 
     -     inpup(idc2,idq0))*yup(8)+dconjg(yup(35))*(dconjg(yup(42))
     -     +yup(36))*(inpup(idb1,idk2)*yup(7)+inpup(idq0,idb1)* 
     -     yup(46)))
        bfun(2,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(15))*inpup(idc1,idq0) + dconjg(yup(14))*
     -     inpup(idc2,idq0))*yup(2) + dconjg(inpup(idk2,idk1))*
     -     dotup(idb1,idb1)*inpup(idq0,idk2)**2*(yup(27)+yup(28)) + 
     -     dotup(idb1,idb1)*inpup(idq0,idk2)*(yup(2)*(yup(27) +
     -     yup(28)) -dconjg(inpup(idc2,idk1))*inpup(idq0,idb1)*
     -     yup(37)) -dconjg(inpup(idc1,idk1))*dotup(idb1,idb1)*
     -     inpup(idq0,idb1)*inpup(idq0,idk2)*yup(43)+dconjg(yup(35))
     -     *inpup(idq0,idb1)*((yup(1) + yup(9))*yup(48) + 
     -     (dconjg(yup(42)) + yup(36))*yup(50)))
	  bfun(3,ii,kk)=4*pmomup(5,idc1)*(dconjg(yup(40))*
     -     inpup(idq0,idc1)*(dconjg(yup(10))*yup(2) - dotup(idb1,idb1)
     -     *yup(8))+dconjg(yup(79))*inpup(idq0,idc1)*(dconjg(yup(10))
     -     *yup(2) -dotup(idb1,idb1)*yup(8))+dconjg(inpup(idq0,idk1))
     -     *dotup(idb1,idb1)*inpup(idq0,idk2)**2*yup(11) + 
     -     inpup(idq0,idk2)* (-(dconjg(yup(10))*(yup(2)*yup(11) 
     -     - (yup(1) + yup(9))*yup(16))) + yup(2)*(yup(1) + yup(9))
     -     *yup(25) -dotup(idb1,idb1)*((yup(1) + yup(9)) *yup(19) +
     -     yup(9)*yup(31)))+ dconjg(yup(35))*(-(dconjg(yup(38))*
     -     inpup(idb1,idk2)*yup(1))+inpup(idq0,idb1)*(dconjg(yup(42))
     -     + yup(36))*(yup(46) + yup(50))))
	  bfun(4,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     ((dconjg(yup(20)) + dconjg(yup(39)))*inpup(idc1,idq0) + 
     -     (dconjg(yup(23))+dconjg(yup(24)))*inpup(idc2,idq0))*yup(8)- 
     -     inpup(idq0,idk2)*((yup(1)+yup(9))*(dotup(idb1,idb1)*yup(31)- 
     -     dconjg(yup(35))*yup(51)) + dconjg(inpup(idb2,idk1))*
     -     dconjg(yup(35))*(inpup(idc2,idq0)*yup(83) + 
     -     inpup(idc1,idq0)*yup(84))))
c...helicity(49)=(-,-,+,-,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(34)=(+,+,-,+,+,-)
        kk=34
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*yup(1)*
     -     (-(dconjg(yup(38))*inpup(idq0,idk2)*yup(13)) + 
     -     inpup(idq0,idk2)*yup(13)*(dconjg(yup(42))+yup(36)) - 
     -     dconjg(yup(35))*(-(dconjg(yup(38))*inpup(idq0,idb1)) + 
     -     inpup(idq0,idb1)*(dconjg(yup(42))+yup(36)))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idc1)*yup(9)*
     -     (inpup(idq0,idk1)*yup(2)*(yup(19) - yup(29)) + 
     -     (-(dconjg(yup(35))*inpup(idq0,idb1)) + inpup(idq0,idk2)
     -     *yup(13))*(dconjg(yup(42)) + yup(36))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*yup(1)*(dotup(idb1,idb1)
     -     *yup(19)*(inpup(idq0,idk2)*(yup(27) + yup(28)) - 
     -     inpup(idq0,idk2)*yup(31))-dconjg(yup(16))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) + inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*
     -     (-(dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)*(yup(27) + 
     -     yup(28))) +inpup(idq0,idk1)*(dotup(idb1,idb1)
     -     *yup(8)*yup(19) +dconjg(yup(16))*yup(2)*yup(29)) + 
     -     dconjg(yup(16))*dconjg(yup(35))*inpup(idq0,idb1)
     -     *(dconjg(yup(42)) + yup(36))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*(inpup(idq0,idk2)*
     -     ((dconjg(yup(10))*dconjg(yup(24)) - dconjg(yup(14))
     -     *dotup(idb1,idb1))*yup(14)+2*dconjg(yup(35))*dotup(idc1,idk1)
     -     *yup(35))+(dconjg(yup(12))-dconjg(yup(18)))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) +inpup(idq0,idb1) 
     -     *(dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*inpup(idq0,idk2)*pmomup(5,idc1)*
     -     (dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idq0,idb1)
     -     *yup(5) +yup(2)*yup(28)*yup(29)))/dconjg(yup(8))
c...helicity(50)=(-,-,+,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(35)=(+,+,-,+,-,+)
        kk=35
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*pmomup(5,idc1)*(-(dotup(idb1,idb1)*
     -     inpup(idq0,idk1)*yup(18)*yup(19)*yup(31)) - 
     -     yup(10)*(dconjg(yup(5))*dotup(idb1,idb1)*
     -     inpup(idq0,idc1)*yup(31) + dconjg(yup(35))*
     -     (dconjg(yup(42))*inpup(idk1,idb1)*yup(31) + 
     -     dconjg(yup(1))*inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36))))))/yup(8)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)*
     -     (dotup(idb1,idb1)*(-(dconjg(yup(8))*dconjg(yup(33))) + 
     -     dconjg(yup(4))*yup(10))-dconjg(yup(24))*yup(10)*yup(25))+ 
     -     dconjg(yup(9))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     yup(10)*(dconjg(yup(42)) + yup(36))))/yup(8)
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*yup(16)*
     -     (dconjg(yup(5))*dotup(idb1,idb1)*inpup(idq0,idc1)*
     -     yup(31)+inpup(idk1,idb1)*(dconjg(yup(35))*dconjg(yup(42)) - 
     -     dconjg(yup(17))*yup(18))*yup(31) + dconjg(yup(1))*
     -     inpup(idq0,idb1)*(-(dconjg(yup(17))*(yup(11) + yup(18))) + 
     -     dconjg(yup(35))*(dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(16)*(inpup(idq0,idk1)
     -     *yup(4)*(dconjg(yup(4))*dotup(idb1,idb1) + 
     -     (-dconjg(yup(24))+dconjg(yup(33)))*yup(25))+dconjg(yup(9))
     -     *inpup(idq0,idb1)*(-(dconjg(yup(17))*(yup(11) + yup(18))) + 
     -     dconjg(yup(35))*(dconjg(yup(42)) + yup(36)))))/yup(8)
 	  bfun(5,ii,kk)=(2*inpup(idq0,idk1)*pmomup(5,idc1)*yup(16)*
     -     (yup(27) + yup(28))*yup(29))/yup(8)
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)
     -     *(yup(12) - yup(18))*(dconjg(yup(4))*dotup(idb1,idb1) - 
     -     dconjg(yup(24))*yup(25)) +2*dotup(idc1,idk1)*
     -     (-(inpup(idq0,idk1)*yup(16)*yup(27)*yup(29)) + 
     -     dconjg(yup(8))*dconjg(yup(35))*inpup(idq0,idb1)*yup(36)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*(dconjg(yup(41))*
     -     inpup(idq0,idb1)*yup(18)+inpup(idq0,idb1)*(yup(12)-yup(18))*
     -     (dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(9,ii,kk)=(-4*dconjg(yup(3))*dconjg(yup(35))*
     -    inpup(idq0,idb1)*pmomup(5,idc1)*yup(11))/yup(8)
c...helicity(51)=(-,-,+,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(36)=(+,+,-,+,-,-)
        kk=36
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (-(dconjg(yup(5))*dotup(idb1,idb1)*inpup(idq0,idc1)
     -     *yup(31)) +dconjg(yup(42))*(-(dconjg(yup(35))*
     -     inpup(idk1,idb1)) -inpup(idk2,idk1)*yup(13))*yup(31) - 
     -     dconjg(yup(1))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     (dconjg(yup(42))+yup(36))+dconjg(yup(1))*inpup(idq0,idk2)
     -     *yup(13)*(dconjg(yup(42)) + yup(36))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (inpup(idq0,idk1)*yup(4)*(dconjg(yup(4))*dotup(idb1,idb1)+ 
     -     dconjg(yup(24))*(dconjg(yup(7)) - yup(25)))- dconjg(yup(9))*
     -     (-(dconjg(yup(35))*inpup(idq0,idb1)) +inpup(idq0,idk2) 
     -     *yup(13))*(dconjg(yup(42)) + yup(36))))/xup(2)
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(1))*
     -     dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)*(yup(27)
     -     + yup(28)) -dotup(idb1,idb1)*inpup(idk2,idk1)*yup(19)*
     -     yup(27)*yup(31) - dconjg(yup(16))*(dconjg(yup(5))*
     -     dotup(idb1,idb1)*inpup(idq0,idc1)*yup(31) +dconjg(yup(35))*
     -     (dconjg(yup(42))*inpup(idk1,idb1)*yup(31) + dconjg(yup(1))
     -     *inpup(idq0,idb1)*(dconjg(yup(42))+yup(36))))))/xup(2)
	  bfun(4,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*inpup(idk2,idk1)*yup(4)*yup(19)) + 
     -     dconjg(yup(9))*dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)
     -     *(yup(27)+yup(28))-dconjg(yup(16))*(inpup(idq0,idk1)*yup(4)*
     -     (dconjg(yup(4))*dotup(idb1,idb1) -dconjg(yup(24))*yup(25)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36)))))/xup(2)
 	bfun(5,ii,kk)=(2*dconjg(yup(10))*inpup(idq0,idk1)*pmomup(5,idc1)
     -     *(yup(27) + yup(28))*yup(30))/xup(2)
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(9))*
     -     (dconjg(yup(10))*dconjg(yup(24)) -dconjg(yup(14))* 
     -     dotup(idb1,idb1))*inpup(idq0,idk2)*yup(14) +inpup(idq0,idk1)*
     -     (dotup(idb1,idb1)*(dconjg(yup(4))*(dconjg(yup(12)) -
     -     dconjg(yup(18)))*yup(4) + dconjg(yup(12))*dconjg(yup(14))
     -     *yup(14)) - dconjg(yup(24))*(dconjg(yup(10))*dconjg(yup(12))
     -     *yup(14)+(dconjg(yup(12))-dconjg(yup(18)))*yup(4)*yup(25)))+ 
     -     dconjg(yup(35))*(-2*dotup(idc1,idk1)*inpup(idk2,idk1)*yup(27)
     -     * yup(35) +dconjg(yup(9))*(dconjg(yup(12))-dconjg(yup(18)))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36)))))/xup(2)
	  bfun(9,ii,kk)=(-4*dconjg(yup(35))*inpup(idk2,idk1)
     -     *pmomup(5,idc1)*yup(28)*yup(35))/xup(2)
c...helicity(52)=(-,-,+,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c*********************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*pmomup(5,idc1)*(dconjg(yup(8))*
     -     dotup(idb1,idb1)*(dconjg(yup(39))*inpup(idc1,idq0) + 
     -     dconjg(yup(24))*inpup(idc2,idq0)) + dconjg(yup(35))*
     -     (dconjg(yup(42))+ yup(36))*(-(inpup(idk1,idb1)*yup(12)) + 
     -     inpup(idq0,idb1)*yup(53)))
        bfun(2,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*inpup(idc1,idq0) + dconjg(yup(4))*
     -     inpup(idc2,idq0))*(yup(12) + yup(16)) + 
     -     dotup(idb1,idb1)*inpup(idq0,idk1)*(yup(16)*(yup(27) + 
     -     yup(28)) -dconjg(inpup(idc2,idk2))*inpup(idq0,idb1)*
     -     yup(37)) -dconjg(inpup(idc1,idk2))*dotup(idb1,idb1)*
     -     inpup(idq0,idb1)*inpup(idq0,idk1)*yup(43)+dconjg(yup(35))
     -     *inpup(idq0,idb1)*((yup(11) + yup(18))*yup(59) + 
     -     (dconjg(yup(42)) + yup(36))*yup(60)))
	  bfun(3,ii,kk)=4*pmomup(5,idc1)*(dconjg(inpup(idq0,idk2))
     -     *dotup(idb1,idb1)*inpup(idq0,idk1)**2*yup(1) + 
     -     (dconjg(yup(40)) + dconjg(yup(79)))*inpup(idq0,idc1)*
     -     (-(dconjg(yup(8))*dotup(idb1,idb1)) +yup(16)*yup(25)) + 
     -     inpup(idq0,idk1)*(dconjg(yup(10))*yup(16)*(yup(11)+yup(18))+ 
     -     (-(yup(1)*yup(16)) + yup(2)*(yup(11) + yup(18)))*yup(25) - 
     -     dotup(idb1,idb1)* (yup(18)*yup(19) + (yup(11) + yup(18))*
     -     yup(31))) +dconjg(yup(35))*(-(dconjg(yup(41))*
     -     inpup(idb1,idk1)*yup(11)) + inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36))*(yup(53) + yup(60))))
	  bfun(4,ii,kk)=4*pmomup(5,idc1)*(yup(11) + yup(18))*
     -     (dotup(idb1,idb1)*(-(dconjg(yup(6))*inpup(idq0,idb1)) - 
     -     dconjg(yup(3))*inpup(idq0,idb2) + dconjg(inpup(idq0,idk1))
     -     *inpup(idq0,idk1)**2)+ dconjg(yup(35))*inpup(idb1,idk1)*
     -     (-dconjg(yup(38)) + yup(61)))
c...helicity(52)=(-,-,+,-,+,+)
        do mm=1,4
	     bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(37)=(+,+,-,-,+,+)
        kk=37
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*(yup(1)*(-(dconjg(inpup(idc1,idb2))*
     -     dconjg(yup(35))*inpup(idc2,idb1)*yup(10))+dotup(idb1,idb1)*
     -     (dconjg(yup(34))*yup(10) + dconjg(inpup(idc1,idk2))*
     -     inpup(idc2,idq0)*yup(19)))*yup(34) +dotup(idc1,idc1)*yup(10)*
     -     (yup(2)*yup(29)*yup(31)+dconjg(yup(35))*yup(1)*yup(35))))
     -     /xup(1)
        bfun(2,ii,kk)=(4*yup(9)*(-(dotup(idb1,idb1)*(-(dconjg(yup(4))*
     -     (yup(4)*yup(10) -dconjg(inpup(idk2,idk1))*
     -     inpup(idc2,idq0)*yup(19))) +dconjg(yup(34))*yup(10)*yup(34) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19)*yup(34))) + 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idb2,idk1))*
     -     dconjg(yup(4))*inpup(idc2,idb1) + dconjg(inpup(idc1,idb2))
     -     *inpup(idc2,idb1)*yup(34)-dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(3,ii,kk)=(4*yup(16)*(-((dconjg(yup(34))*dotup(idb1,idb1) + 
     -     (dconjg(inpup(idc1,idk2))*dconjg(yup(17)) - 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35)))*inpup(idc2,idb1))
     -     *yup(1)*yup(34))+dotup(idc1,idc1)*((-yup(2)+yup(7))*yup(29)
     -     *yup(31) +yup(1)*(dconjg(yup(17))*yup(17) - dconjg(yup(35))
     -     *yup(35)))))/xup(1)
	  bfun(4,ii,kk)=(-4*yup(9)*yup(16)*(dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) - dconjg(yup(34))*yup(34)) + 
     -     dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1)) +dotup(idc1,idc1)*yup(17) - 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idb1)*yup(34)) + 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1) +dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(1)
 	  bfun(5,ii,kk)=(2*yup(9)*yup(16)*yup(28)*yup(29))/xup(1)
	  bfun(8,ii,kk)=(4*yup(9)*(yup(12) - yup(18))*
     -     (dconjg(yup(14))*yup(16)*yup(23) -dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) + dconjg(yup(14))*yup(14) - 
     -     dconjg(yup(34))*yup(34)) +dconjg(yup(35))*
     -     (-(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1))- 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) + dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -     dconjg(yup(42))*inpup(idq0,idb1)*yup(28))/xup(1)
c...helicity(53)=(-,-,+,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c****************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*(dconjg(yup(35))*((dconjg(inpup(idc1,idb2))
     -     *inpup(idc2,idb1)*yup(34) - dotup(idc1,idc1)*yup(35))
     -     *yup(46)+ (-(dconjg(yup(13))*dotup(idc1,idc1)) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idk2)*yup(34))*yup(47))- 
     -     dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)*yup(30) + 
     -     yup(34)*(-(dconjg(inpup(idq0,idk1))*dconjg(yup(39))
     -     *inpup(idc2,idk2))+dconjg(inpup(idc1,idk1))*inpup(idb1,idk2)*
     -     yup(37) +yup(4)*(-dconjg(yup(54)) + yup(74)))))
        bfun(2,ii,kk)=4*(dconjg(yup(35))*((dotup(idc1,idc1)*yup(6) - 
     -     dconjg(inpup(idc1,idk1))*inpup(idc2,idb1)*yup(34))*yup(48)+ 
     -     (-(dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))*yup(50)) +dotup(idb1,idb1)*
     -     (dotup(idc1,idc1)*yup(8)*yup(29) + yup(34)*
     -     (inpup(idc2,idk2)*(dconjg(inpup(idc1,idq0))*yup(2) + 
     -     dconjg(inpup(idc1,idq0))*yup(7)) - dconjg(inpup(idc1,idk1))
     -     *inpup(idq0,idk2)*yup(23) + dconjg(yup(15))*yup(86))))
	  bfun(3,ii,kk)=4*(-(dconjg(yup(1))*dconjg(yup(10))*yup(2)*
     -     yup(9)) + dconjg(yup(1))*dotup(idb1,idb1)*yup(8)*yup(9) - 
     -     dconjg(yup(10))*dconjg(yup(11))*yup(2)*yup(18) + 
     -     dconjg(yup(11))*dotup(idb1,idb1)*yup(8)*yup(18) + 
     -     dconjg(inpup(idc1,idk1))*inpup(idc2,idk2)*(dconjg(yup(10))
     -     *yup(16) + yup(2)*yup(25) - dotup(idb1,idb1)*(yup(19) +
     -     yup(31)))*yup(34)-dconjg(yup(34))*dotup(idb1,idb1)*yup(34)*
     -     yup(46) - dconjg(yup(34))*dotup(idb1,idb1)*yup(34)*yup(50) + 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)
     -     *yup(34)*(yup(46)+yup(50))+dotup(idc1,idc1)*(-(yup(2)*yup(8)
     -     *yup(25)) - dconjg(yup(10))*(yup(8)*yup(16)- yup(2)*(yup(19)
     -     + yup(31)))- dconjg(yup(35))*yup(35)*(yup(46) + yup(50))))
	bfun(4,ii,kk)=4*(-(dconjg(inpup(idq0,idk1))**2*dotup(idb1,idb1)*
     -     dotup(idc1,idc1)*inpup(idq0,idk1)*inpup(idq0,idk2)) + 
     -     dotup(idc1,idc1)* (dotup(idb1,idb1)*(dconjg(yup(13))*yup(3)+ 
     -     dconjg(yup(17))*yup(6)) +  dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*inpup(idq0,idk2)*yup(21) + 
     -     dconjg(yup(13))*yup(82))) + inpup(idc2,idk2)*yup(34)*
     -     (-(dconjg(inpup(idq0,idk1))*(dconjg(yup(20)) +
     -     dconjg(yup(39)))*dotup(idb1,idb1))+dconjg(inpup(idc1,idk1))*
     -     (dotup(idb1,idb1)*yup(31) -dconjg(yup(35))*yup(51)) + 
     -     dconjg(inpup(idb2,idk1))*dconjg(yup(35))*yup(84)))
c...helicity(53)=(-,-,+,+,-,-)
        do mm=1,4
	     bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(38)=(+,+,-,-,+,-)
        kk=38
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(10))*(yup(1)*(dconjg(yup(34))
     -     *dotup(idb1,idb1) +dconjg(inpup(idc1,idb2))*
     -     (-(dconjg(yup(35))*inpup(idc2,idb1)) + inpup(idc2,idk2)
     -     *yup(13)))*yup(34) +dotup(idc1,idc1)*(yup(2)*(-yup(19)
     -     + yup(29))*yup(31) +yup(1)*(-(dconjg(yup(13))*yup(13)) + 
     -     dconjg(yup(35))*yup(35)))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*yup(9)*(-((dconjg(yup(13))
     -     *dotup(idc1,idc1)-dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idk2))*yup(13)) +dconjg(inpup(idc1,idb2))* 
     -     inpup(idc2,idk2)*yup(13)*yup(34) + dotup(idb1,idb1)*
     -     (-(dconjg(yup(4))*yup(4)) + dconjg(yup(34))*yup(34)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1) +dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)
     -     *yup(19)*yup(31)+yup(1)*(dconjg(yup(16))*dconjg(yup(34)) + 
     -     dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*yup(19))*yup(34))+ 
     -     4*dconjg(yup(16))*(-(dotup(idc1,idc1)*yup(2)*yup(29)
     -     *yup(31)) +dconjg(yup(35))*yup(1)*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*yup(9)*(dotup(idb1,idb1)*(-(dconjg(yup(4))*
     -     (dconjg(yup(16))*yup(4) -dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(19)))+dconjg(yup(16))*dconjg(yup(34))
     -     *yup(34) +dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*
     -     yup(19)*yup(34)) -dconjg(yup(16))*dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=(2*dconjg(yup(10))*yup(9)*yup(28)*yup(30))
     &	 /dconjg(yup(8))
	  bfun(8,ii,kk)=(-4*yup(9)*((dconjg(yup(12)) - dconjg(yup(18)))*
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(-((dconjg(inpup(idb2,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18))) - 
     -     2*dconjg(inpup(idb2,idq0))*dconjg(yup(14))*dotup(idc1,idk1))*
     -     inpup(idc2,idb1)) +dconjg(inpup(idc1,idb2))*(dconjg(yup(12))
     -     -dconjg(yup(18)))*inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*
     -     (dconjg(yup(10))*yup(19)*yup(30) + (-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dconjg(yup(35))*yup(35))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idb2))*dconjg(yup(11))*
     -     dconjg(yup(35))*inpup(idq0,idb1)*yup(5))/dconjg(yup(8))
c...helicity(54)=(-,-,+,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(39)=(+,+,-,-,-,+)
        kk=39
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(1))* (-(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*yup(5)*yup(10) -dconjg(inpup(idc1,idk2))*
     -     inpup(idq0,idk1)*yup(5)*yup(19) -(dconjg(yup(34))*yup(10)+ 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19))*yup(34)))+ 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idc1,idb2))*
     -     (inpup(idk1,idb1)*yup(5) - inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))))/yup(8)
        bfun(2,ii,kk)=(-4*(dotup(idc1,idc1)*(dconjg(yup(8))*
     -     dotup(idb1,idb1)*yup(19)+yup(10)*yup(25)*yup(30))*yup(31)+ 
     -     dconjg(yup(9))*dotup(idb1,idb1)*(dconjg(yup(34))*yup(10) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19))*yup(34)) + 
     -     4*dconjg(yup(9))*dconjg(yup(35))*yup(10)*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35)))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*yup(16)*(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*yup(5) - dconjg(yup(34))*yup(34)) + 
     -     dconjg(yup(17))*(dotup(idc1,idc1)*yup(17) + 
     -     dconjg(inpup(idc1,idk2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))) - dconjg(yup(35))*
     -     (dconjg(inpup(idc1,idb2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*yup(35))))/yup(8)
	  bfun(4,ii,kk)=(4*yup(16)*(dconjg(yup(9))*(dconjg(yup(34))
     -     *dotup(idb1,idb1)+(dconjg(inpup(idc1,idk2))*dconjg(yup(17))- 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35)))*inpup(idc2,idb1))
     -     *yup(34)+dotup(idc1,idc1)*(yup(25)*(-yup(19)+yup(30))*yup(31) 
     -     +dconjg(yup(9))*(-(dconjg(yup(17))*yup(17)) + 
     -     dconjg(yup(35))*yup(35)))))/yup(8)
 	  bfun(5,ii,kk)=(2*dconjg(yup(1))*yup(16)*yup(27)*yup(29))/yup(8)
	  bfun(8,ii,kk)=(4*(2*dconjg(yup(4))*dotup(idc1,idk1)*
     -     (dotup(idb1,idb1)*yup(14) - yup(16)*yup(23))*yup(27) +
     -     dconjg(yup(9))*(-(dconjg(yup(14))*yup(16)*yup(18)*yup(23)) + 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)*
     -     (-yup(12) + yup(18))*yup(34) + dotup(idb1,idb1)*
     -     (dconjg(yup(14))*yup(14)*yup(18) +dconjg(yup(34))*(yup(12) 
     -     - yup(18))*yup(34))) + dotup(idc1,idc1)*(2*dconjg(yup(8))
     -     *dconjg(yup(35))*dotup(idc1,idk1)*yup(35)+(yup(12)-yup(18))*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
	  bfun(9,ii,kk)=(4*(dconjg(yup(5))*dotup(idb1,idb1)*yup(15)*
     -     yup(28) + dconjg(yup(35))*(-(dconjg(inpup(idc1,idb2))*
     -     inpup(idk1,idb1)*yup(15)*yup(28)) + 
     -     dconjg(yup(8))*dotup(idc1,idc1)*yup(35))))/yup(8)
c...helicity(55)=(-,-,+,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(40)=(+,+,-,-,-,-)
        kk=40
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(10))*
     -     (-(dconjg(yup(13))*dotup(idc1,idc1)*yup(13)) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idk2,idk1)*yup(5)*yup(13) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idk2)*yup(13)*yup(34) - 
     -     dotup(idb1,idb1)*(dconjg(yup(5))*yup(5) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(dconjg(inpup(idc1,idb2))*
     -     (inpup(idk1,idb1)*yup(5) -inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*(-(dconjg(yup(9))*
     -     (dconjg(yup(34))*dotup(idb1,idb1)+dconjg(inpup(idc1,idb2))*
     -     (-(dconjg(yup(35))*inpup(idc2,idb1)) + inpup(idc2,idk2)
     -     *yup(13)))*yup(34)) - dotup(idc1,idc1)*((dconjg(yup(3))*
     -     dconjg(yup(12)) -dconjg(yup(9))*dconjg(yup(13)))*yup(13) + 
     -     yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35))))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*(dotup(idb1,idb1)*
     -     (-(dconjg(inpup(idc1,idq0))*inpup(idk2,idk1)*yup(5)*yup(19))- 
     -     dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*yup(19)*yup(34) + 
     -     dconjg(yup(16))*(dconjg(yup(5))*yup(5) - dconjg(yup(34))*
     -     yup(34))) - dconjg(yup(16))*dconjg(yup(35))*
     -     (dconjg(inpup(idc1,idb2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*yup(35))))/xup(2)
	  bfun(4,ii,kk)=(4*(dconjg(yup(16))*dotup(idc1,idc1)*yup(25)*
     -     yup(30)*yup(31) + dconjg(yup(9))*(dotup(idb1,idb1)*
     -     (dconjg(yup(16))*dconjg(yup(34)) + dconjg(inpup(idc1,idq0))
     -     *inpup(idc2,idk2)*yup(19))*yup(34) - dconjg(yup(16))*
     -     dconjg(yup(35))*(dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35)))))/xup(2)
 	bfun(5,ii,kk)=(2*dconjg(yup(1))*dconjg(yup(10))*yup(27)*yup(30))
     &	  /xup(2)
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(30)*(dconjg(yup(9))*
     -     dconjg(yup(10))*yup(19) + (dconjg(yup(7))*dconjg(yup(10)) + 
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*yup(25))*yup(31)) + 
     -     4*dconjg(yup(9))*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     dconjg(yup(34))*dotup(idb1,idb1)*yup(34)+4*dconjg(yup(35))*
     -     (-2*dconjg(inpup(idb2,idq0))*dotup(idc1,idk1)*
     -     inpup(idk2,idk1)*yup(22)*yup(27) + dconjg(yup(9))*
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(56)=(-,-,+,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c**********************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*(dconjg(yup(8))*dotup(idb1,idb1)*
     -    dotup(idc1,idc1)*yup(30) - dconjg(inpup(idq0,idk2))
     -    *dconjg(yup(39))*dotup(idb1,idb1)*inpup(idc2,idk1)*yup(34)+ 
     -    dconjg(inpup(idc1,idb2))*dconjg(yup(35))*dconjg(yup(45))
     -    *inpup(idk1,idb1)*yup(34) + dconjg(inpup(idc1,idk2))
     -    *dotup(idb1,idb1)*inpup(idb1,idk1)*yup(34)*yup(37) - 
     -    dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)
     -    *yup(34)*yup(53) +dconjg(yup(35))*dotup(idc1,idc1)*yup(35)*
     -    yup(53) + dconjg(yup(3))*dconjg(yup(35))*dotup(idc1,idc1)
     -    *yup(57) -dotup(idb1,idb1)*yup(14)*yup(34)*yup(72) + 
     -    dotup(idb1,idb1)*yup(14)*yup(34)*yup(81))
        bfun(2,ii,kk)=4*dotup(idb1,idb1)*(dconjg(yup(8))
     -    *dotup(idc1,idc1)*yup(29) - (dconjg(yup(5))*
     -    (dconjg(yup(45))- dconjg(yup(68)))- dconjg(inpup(idc1,idq0))
     -    *inpup(idc2,idk1)*yup(16) + dconjg(inpup(idc1,idk2))*
     -    inpup(idq0,idk1)*yup(23))*yup(34)) +4*dconjg(yup(35))*
     -    ((dotup(idc1,idc1)*yup(17) -dconjg(inpup(idc1,idk2))* 
     -    inpup(idc2,idb1)*yup(34))*yup(59)+(-(dconjg(inpup(idc1,idb2)) 
     -    *inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*yup(35))*yup(60))
	bfun(3,ii,kk)=4*(dconjg(yup(1))*dconjg(yup(8))*dotup(idb1,idb1)*
     -    yup(9) + dconjg(yup(8))*dconjg(yup(11))*dotup(idb1,idb1)
     -    *yup(18) -dconjg(yup(1))*yup(9)*yup(16)*yup(25) - 
     -    dconjg(yup(11))*yup(16)*yup(18)*yup(25) + 
     -    dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*(dconjg(yup(10))
     -    *yup(16) + yup(2)*yup(25) - dotup(idb1,idb1)*(yup(19) + 
     -    yup(31)))*yup(34)- dconjg(yup(34))*dotup(idb1,idb1)*yup(34)*
     -    yup(53) - dconjg(yup(34))*dotup(idb1,idb1)*yup(34)*yup(60) + 
     -    dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)*
     -    yup(34)*(yup(53)+yup(60))+dotup(idc1,idc1)*(-(dconjg(yup(8))
     -    *yup(2)*yup(25))+yup(16)*(-(dconjg(yup(8))*dconjg(yup(10)))+ 
     -    yup(25)*(yup(19) + yup(31))) - dconjg(yup(35))*yup(35)
     -    *(yup(53) + yup(60))))
	  bfun(4,ii,kk)=-4*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(35))*
     -    dconjg(yup(38))*dotup(idc1,idc1)*inpup(idb1,idk1)) + 
     -    dconjg(inpup(idq0,idk1))*dconjg(inpup(idq0,idk2))*
     -    dotup(idb1,idb1)*dotup(idc1,idc1)*inpup(idq0,idk1)**2 - 
     -    dconjg(yup(3))*dotup(idb1,idb1)*dotup(idc1,idc1)*
     -    yup(13) - dconjg(yup(6))*dotup(idb1,idb1)*dotup(idc1,idc1)
     -    *yup(17) +dconjg(inpup(idc1,idk2))*dconjg(yup(6))*
     -    dotup(idb1,idb1)*inpup(idc2,idb1)*yup(34) + 
     -    dconjg(inpup(idc1,idk2))*dconjg(yup(3))*
     -    dotup(idb1,idb1)*inpup(idc2,idb2)*yup(34) + 
     -    dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*yup(34)*
     -    (-(dotup(idb1,idb1)*yup(31)) +dconjg(yup(35))*yup(51)) + 
     -    dconjg(inpup(idq0,idk2))*dconjg(yup(35))*
     -    dotup(idc1,idc1)*inpup(idb1,idk1)*yup(61) - 
     -    dconjg(inpup(idc1,idk2))*dconjg(yup(35))*
     -    inpup(idb1,idk1)*yup(34)*yup(64))
c...helicity(56)=(-,-,+,+,+,+)
        do mm=1,4
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(41)=(+,-,+,+,+,+)
        kk=41
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*pmomup(5,idb1)*yup(1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (-(dconjg(inpup(idk2,idk1))*dconjg(yup(5))*inpup(idc1,idb2))- 
     -     dconjg(inpup(idc2,idk2))*dconjg(yup(34))*inpup(idc1,idb2) + 
     -     dotup(idc1,idc1)*yup(13))) +dconjg(inpup(idq0,idk1))* 
     -     dconjg(yup(5))*yup(10)*(yup(20) + yup(39)) - 
     -     dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     yup(10)*(yup(20) + yup(39))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*(dotup(idc1,idc1)*
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3))-dconjg(inpup(idb1,idq0))
     -     *yup(6))*yup(10)*yup(31) + dconjg(yup(13))*yup(13)*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9))+dconjg(inpup(idk2,idk1))
     -     *yup(31)))+dconjg(yup(34))*yup(9)*(dconjg(inpup(idc2,idk2))*
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(13))*inpup(idc1,idb2) - 
     -     dconjg(inpup(idq0,idc2))*yup(10)*(yup(20)+yup(39)))))/xup(1)
	bfun(3,ii,kk)=(4*pmomup(5,idb1)*yup(1)*(dconjg(inpup(idb1,idq0))
     -     *dotup(idc1,idc1)*yup(17)*yup(19) +dconjg(yup(34))*
     -     (dconjg(inpup(idc2,idk2))*yup(19)*yup(20) - 
     -     dconjg(inpup(idq0,idc2))*yup(16)*(yup(20) + yup(39))) + 
     -     dconjg(yup(5))*(dconjg(inpup(idk2,idk1))*yup(19)*yup(20) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*(yup(20)+yup(39)))))/xup(1)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*yup(9)*yup(17)*yup(19) + 
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3))-dconjg(inpup(idb1,idq0))
     -     *yup(6))*yup(16) +dconjg(inpup(idk2,idk1))*yup(19)*yup(29)
     -     )*yup(31)))-dconjg(yup(34))*yup(9)*(dconjg(inpup(idc2,idk2))
     -     *yup(19)*yup(20) - dconjg(inpup(idq0,idc2))*yup(16)*
     -     (yup(20) + yup(39)))))/xup(1)
 	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)
     -     *yup(1)*yup(27)*(yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(2*dconjg(inpup(idk2,idk1))
     -     *dotup(idc1,idk1)*yup(27)*yup(28)*(yup(29) + yup(30)) + 
     -     dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*(yup(9)*yup(13)
     -     *yup(19)+(yup(7)*yup(13)+yup(3)*(-yup(12)+yup(18)))*yup(31))
     -     ) - dconjg(inpup(idb1,idq0))*(yup(9)*yup(17)*yup(19) + 
     -     (yup(7)*yup(17) +yup(6)*(-yup(12) + yup(18)))*yup(31)))
     -     + dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     yup(9)*(yup(12) - yup(18))*(yup(20) + yup(39))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(57)=(-,+,-,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c****************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30) + 
     -     dconjg(inpup(idq0,idk1))* (dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk2))
     -     + dconjg(yup(10))*yup(30)) + dconjg(yup(34))*
     -     (dconjg(inpup(idq0,idc2))*dotup(idb1,idb1)*inpup(idc1,idk2) +
     -     yup(39)*yup(44))) - dconjg(yup(34))*(dconjg(inpup(idq0,idc2))
     -     *dconjg(yup(10))* dconjg(yup(72)) + dconjg(inpup(idc2,idk1))*
     -     (dconjg(yup(10)) + dconjg(yup(12)))*yup(39) - 
     -     dconjg(inpup(idq0,idc2))*yup(20)*yup(46)))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(inpup(idb1,idq0))*dconjg(yup(16))*yup(6) + 
     -    dconjg(inpup(idq0,idk1))*(dconjg(yup(16)) - 
     -    dconjg(inpup(idq0,idk1))*inpup(idk2,idk1))*yup(29))) + 
     -    dconjg(inpup(idb2,idq0))*dconjg(yup(18))*yup(28)*
     -    yup(66) + dconjg(yup(34))*(dconjg(inpup(idq0,idc2))*
     -    dconjg(inpup(idq0,idk1))*dotup(idb1,idb1)*inpup(idc1,idk2) + 
     -    dconjg(inpup(idq0,idc2))*yup(39)*yup(50) +  yup(20)*
     -    (-(dconjg(inpup(idc2,idk1))*(dconjg(yup(12)) +
     -    dconjg(yup(16)))) + dconjg(inpup(idq0,idk1))*yup(68))))
	  bfun(3,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))
     -    *dconjg(yup(9))*dconjg(yup(10))*yup(1) + 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(9))*dconjg(yup(16))
     -    *yup(1) + dconjg(inpup(idq0,idk1))*dconjg(yup(2))*
     -    dotup(idc1,idc1)*yup(8) + dconjg(inpup(idq0,idk2)) 
     -    *dconjg(yup(16))*dotup(idc1,idc1)*yup(8) + 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(10))*dconjg(yup(18))
     -    *yup(11) +dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -    dconjg(yup(18))*yup(11) - dconjg(inpup(idq0,idk1))
     -    *dconjg(yup(10))*dotup(idc1,idc1)*yup(19) - 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -    dotup(idc1,idc1)*yup(19) +dconjg(inpup(idb1,idq0))*
     -    (-(dconjg(inpup(idc2,idk1))*dconjg(yup(34))*
     -    inpup(idc1,idk2))+dotup(idc1,idc1)*yup(8))*(yup(21)+yup(26))- 
     -    dconjg(inpup(idq0,idk1))*dconjg(yup(10))*dotup(idc1,idc1)
     -    *yup(31) -dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -    dotup(idc1,idc1)*yup(31) +dconjg(yup(34))*
     -    (dconjg(inpup(idc2,idk1))*(-(dconjg(inpup(idq0,idk1))
     -    *dconjg(yup(2))*inpup(idc1,idk2)) -dconjg(yup(16)) 
     -    *yup(40)) -dconjg(inpup(idq0,idc2))*(yup(20) + yup(39))*
     -    (yup(46) + yup(50))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))*
     -    dotup(idc1,idc1)*(dconjg(yup(13))*(dconjg(yup(61)) - 
     -    dconjg(inpup(idq0,idk1))*inpup(idk1,idb2))+dconjg(yup(17))*
     -    (dconjg(inpup(idq0,idk1))*inpup(idb1,idk1) +yup(75))) + 
     -    dconjg(yup(34))*inpup(idc1,idk2)*(-(dconjg(inpup(idc2,idk1))*
     -    (-(dconjg(inpup(idb1,idq0))*yup(21))-dconjg(inpup(idb2,idq0))*
     -    yup(38))) -dconjg(inpup(idb1,idq0))*dconjg(inpup(idq0,idk1))
     -    *yup(83) -dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))
     -    *yup(85)))
c...helicity(57)=(-,+,-,-,-,-)
        do mm=1,4
	     bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(42)=(+,-,+,+,+,-)
        kk=42
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(yup(10))*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(5)) -dconjg(inpup(idq0,idc2))*dconjg(yup(34)))*
     -     pmomup(5,idb1)*yup(1)*(yup(20) + yup(39) - yup(40)))
     -     /dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk1))
     -     *yup(19))*yup(31)-dconjg(inpup(idq0,idc2))*dconjg(yup(34))
     -     *yup(9)*(yup(20)+yup(39)-yup(40))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*yup(1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(18))*(-yup(28)+yup(31))+ 
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(5)) - 
     -     dconjg(inpup(idq0,idc2))*dconjg(yup(34)))*
     -     (yup(20) + yup(39))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb2,idq0))*dotup(idc1,idc1)*yup(3)*
     -     yup(31) + dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)
     -     *yup(6)*yup(31) +dconjg(inpup(idq0,idk2))*
     -     (dconjg(yup(18))*yup(9)*yup(28)+dotup(idc1,idc1)*yup(8) 
     -     *yup(31))+dconjg(inpup(idq0,idc2))*dconjg(yup(34))*yup(9)*
     -     (yup(20) + yup(39))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*((-dconjg(yup(12)) 
     -     + dconjg(yup(18)))*dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(31) - 
     -     dconjg(inpup(idq0,idc2))*(-((dconjg(yup(10)) +
     -     dconjg(yup(16)))*(dconjg(yup(18))*yup(9)*yup(14) + 
     -     2*dotup(idc1,idk1)*yup(4)*yup(27))) +(-dconjg(yup(12)) 
     -     + dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(yup(20) 
     -     + yup(39)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(15))*
     -     pmomup(5,idb1)*yup(28)*(yup(20)+yup(39)))/dconjg(yup(8))
c...helicity(58)=(-,+,-,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(43)=(+,-,+,+,-,+)
        kk=43
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(19)*yup(31) + 
     -     dconjg(yup(1))*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (dconjg(inpup(idc2,idk2))*dconjg(yup(34))*inpup(idc1,idb2) - 
     -     dotup(idc1,idc1)*yup(13))) + dconjg(inpup(idq0,idc2))
     -     *dconjg(yup(34))*yup(10)*(yup(20) + yup(39)))))/yup(8)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dotup(idc1,idc1)*yup(13)) - dconjg(inpup(idc2,idk2))*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dconjg(yup(34))*inpup(idc1,idb2)) + dconjg(yup(2))*yup(4)
     -     *yup(19)) -dconjg(inpup(idq0,idc2))*yup(10)*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/yup(8)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*yup(19)*
     -     (-(dconjg(inpup(idb1,idq0))*dconjg(yup(1))*yup(17)) + 
     -     dconjg(inpup(idq0,idk2))*yup(25)*yup(31)))+dconjg(yup(1))
     -     *dconjg(yup(34))*(dconjg(inpup(idc2,idk2))*yup(19)*yup(20) - 
     -     dconjg(inpup(idq0,idc2))*yup(16)*
     -     (yup(20) + yup(39)))))/yup(8)
	  bfun(4,ii,kk)=(-4*dconjg(yup(9))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(inpup(idc2,idk2))*yup(19)*(dconjg(yup(34))*yup(20)
     -     - yup(4)*yup(25)) -dconjg(inpup(idq0,idc2))*yup(16)*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/yup(8)
 	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*dconjg(yup(9))*
     -     pmomup(5,idb1)*yup(28)*(yup(29) + yup(30)))/yup(8)
	bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*(dotup(idc1,idc1)
     -     *(-(dconjg(inpup(idb2,idq0))*yup(13))-dconjg(inpup(idb1,idq0) 
     -     )*yup(17))*yup(19) - dconjg(inpup(idq0,idc2))*
     -     (2*dotup(idc1,idk1)*yup(14)*(yup(29)+yup(30))+(yup(12)-
     -     yup(18))*(yup(4)*(dconjg(yup(2)) + yup(25)) - 
     -     dconjg(yup(34))*(yup(20) + yup(39))))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(59)=(-,+,-,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(44)=(+,-,+,+,-,-)
        kk=44
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(inpup(idq0,idc2))*dconjg(yup(1))*
     -     dconjg(yup(10))*dconjg(yup(34))*pmomup(5,idb1)*
     -     (yup(20) + yup(39) - yup(40)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(4)*(dconjg(yup(2))
     -     - dconjg(yup(7)) +yup(25)) - dconjg(yup(34))*(yup(20) + 
     -     yup(39) - yup(40))))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(16))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(18))*
     -     yup(28)) -  dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     (yup(20) + yup(39))))/xup(2)
	bfun(4,ii,kk)=(-4*dconjg(yup(9))*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dconjg(inpup(idq0,idc2))*dconjg(inpup(idq0,idk2))
     -     *inpup(idk2,idk1)*yup(4) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(18))*yup(28) -dconjg(inpup(idq0,idc2))*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/xup(2)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     ((dconjg(yup(10)) + dconjg(yup(16)))*yup(14) + 
     -     yup(4)*(dconjg(yup(2)) + yup(25)) - 
     -     dconjg(yup(34))*(yup(20) + yup(39))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(60)=(-,+,-,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do
	  end if

c****************************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30))-dconjg(yup(34))*
     -     (dconjg(inpup(idc2,idk2))*(dconjg(yup(7))+yup(25))*yup(39) - 
     -     dconjg(inpup(idq0,idc2))*yup(20)*yup(53) + 
     -     dconjg(inpup(idq0,idc2))*yup(25)*yup(54)) + 
     -     dconjg(inpup(idq0,idk2))* (dotup(idc1,idc1)* (-
     -    (dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk1))+ 
     -    yup(25)*yup(30)) +dconjg(yup(34))*(dconjg(inpup(idq0,idc2))
     -    *dotup(idb1,idb1)*inpup(idc1,idk1) + yup(39)*yup(55))))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(inpup(idb1,idq0))*dconjg(yup(2))*yup(17) + 
     -    dconjg(inpup(idq0,idk2))*(dconjg(yup(2)) + 
     -    dconjg(inpup(idq0,idk2))*inpup(idk2,idk1))*yup(29))) + 
     -    dconjg(yup(34))*(dconjg(inpup(idq0,idc2))*
     -    dconjg(inpup(idq0,idk2))*dotup(idb1,idb1)*inpup(idc1,idk1)- 
     -    dconjg(inpup(idq0,idc2))*(dconjg(yup(2))*yup(54) -
     -    yup(39)*yup(60)) +yup(20)*(-(dconjg(inpup(idc2,idk2))*
     -    (dconjg(yup(2)) + dconjg(yup(7))))+ dconjg(inpup(idq0,idk2))
     -    *yup(73))))
	bfun(3,ii,kk)=4*pmomup(5,idb1)*(dconjg(yup(8))*dotup(idc1,idc1)*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(2)) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(16)) + 
     -     dconjg(inpup(idb1,idq0))*(yup(21) + yup(26)))
     -     + dconjg(inpup(idq0,idk2))*(dconjg(yup(2)) + yup(25))*
     -     (dconjg(yup(9))*yup(1) +dconjg(yup(18))*yup(11) - 
     -     dotup(idc1,idc1)*(yup(19) + yup(31))) + dconjg(yup(34))*
     -     (-(dconjg(inpup(idq0,idc2))*(yup(20) + yup(39))*
     -     (yup(53) + yup(60))) -  dconjg(inpup(idc2,idk2))*
     -     (inpup(idc1,idk1)*(dconjg(inpup(idq0,idk2))*dconjg(yup(16)) + 
     -     dconjg(inpup(idb1,idq0))*(yup(21) + yup(26))) + 
     -     dconjg(yup(2))*yup(79))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(yup(2)) + yup(25))*(dconjg(inpup(idb2,idq0))
     -     *yup(13) +dconjg(inpup(idb1,idq0))*yup(17) + 
     -     dconjg(inpup(idq0,idk2))*yup(31)))-dconjg(inpup(idc2,idk2)) 
     -     *dconjg(yup(34))*(inpup(idc1,idk1)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(21))-dconjg(inpup(idb2,idq0))
     -     *yup(38)) +(dconjg(yup(2))+ yup(25))*(yup(20) + yup(39))))
c...helicity(60)=(-,+,-,-,+,+)
        do mm=1,9
	     bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(45)=(+,-,+,-,+,+)
        kk=45
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk1))*yup(10)*yup(31)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(32)) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)) - 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (-(dconjg(inpup(idk2,idk1))*yup(31)*yup(42)) + 
     -     dconjg(inpup(idq0,idk2))*yup(1)*
     -     (dconjg(yup(36)) + yup(42)))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     (dconjg(inpup(idk2,idk1))*yup(19)*yup(24) + 
     -     dconjg(inpup(idq0,idk1))*yup(10)*(yup(23)+yup(24))))- 
     -     dconjg(inpup(idq0,idk2))**2*dconjg(yup(13))*
     -     yup(9)*(dconjg(yup(36)) + yup(42))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*yup(1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23)) +yup(31)* 
     -     (dconjg(inpup(idc1,idq0))*dconjg(inpup(idk2,idk1))
     -     *yup(19)*yup(20) +dconjg(inpup(idq0,idk1))*yup(16)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(32)) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*yup(9)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23)) - 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     (dconjg(inpup(idk2,idk1))*yup(19)*yup(23) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*
     -     (yup(23) + yup(24)))))/xup(1)
 	  bfun(5,ii,kk)=(2*dconjg(inpup(idq0,idk1))*
     -     dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(14))*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9)) + 
     -     dconjg(inpup(idq0,idk1))*yup(12)) + 
     -     dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (yup(12) - yup(18)))*(yup(23) + yup(24)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(61)=(-,+,-,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c***********************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*dconjg(inpup(idq0,idk1))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))*inpup(idk2,idk1)*
     -     (dconjg(inpup(idq0,idc2))*yup(24)-dconjg(inpup(idc1,idq0)) 
     -     *yup(39)) +dconjg(yup(10))*(dconjg(inpup(idq0,idc2))*
     -     yup(24) -dconjg(inpup(idc1,idq0))*yup(39) - 
     -     dconjg(inpup(idb2,idq0))*(dconjg(yup(36)) + yup(42))))
        bfun(2,ii,kk)=4*dconjg(inpup(idq0,idk1))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))*inpup(idk2,idk1)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2))
     -     *yup(23))) + dconjg(yup(16))*(dconjg(inpup(idc1,idq0))*
     -     yup(20) - dconjg(inpup(idq0,idc2))*yup(23) + 
     -     dconjg(inpup(idb1,idq0))*(yup(22) + yup(32))))
	  bfun(3,ii,kk)=-4*dconjg(inpup(idq0,idk1))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))*(-(dconjg(yup(1))*
     -     (dconjg(yup(10)) + dconjg(yup(16)))) +dconjg(yup(2))*
     -     (dconjg(yup(11))+dconjg(yup(18)))))-dconjg(inpup(idb1,idq0))*
     -     (dconjg(yup(11)) + dconjg(yup(18)))*(yup(21) + yup(26)) + 
     -     dconjg(inpup(idc1,idq0))*(dconjg(yup(10)) + 
     -     dconjg(yup(16)))*yup(40) + dconjg(yup(16))*
     -     (dconjg(inpup(idq0,idc2))*yup(33) -dconjg(inpup(idc1,idq0))* 
     -     yup(40)) - dconjg(yup(11))*(-(dconjg(inpup(idb1,idq0))*
     -     yup(26)) - dconjg(inpup(idb2,idq0))*yup(41)) + 
     -     dconjg(inpup(idc1,idq0))*(dconjg(yup(10)) + 
     -     dconjg(yup(16)))*yup(79))
	  bfun(4,ii,kk)=-4*dconjg(inpup(idq0,idk1))*
     -    (dconjg(yup(11)) + dconjg(yup(18)))*pmomup(5,idb1)*
     -    pmomup(5,idc1)*(-(dconjg(inpup(idb2,idq0))*dconjg(yup(61)))+ 
     -    dconjg(inpup(idq0,idk1))*(dconjg(yup(2)) + yup(25)) - 
     -    dconjg(inpup(idb1,idq0))*yup(75))
c...helicity(61)=(-,+,-,+,-,-)
        do mm=1,4
	     bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(46)=(+,-,+,-,+,-)
        kk=46
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(-4*dconjg(inpup(idq0,idk1))*dconjg(yup(10))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(31)*
     -    (-(dconjg(inpup(idb1,idq0))*yup(32))-dconjg(inpup(idc1,idq0))
     -     *yup(40) -dconjg(inpup(idb2,idq0))*yup(42)))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (yup(23) + yup(24) - yup(33)))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(31)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(18))) - 
     -     dconjg(inpup(idb1,idq0))*yup(32) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(yup(23)+yup(24)))-dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(11))*yup(31)))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(23) + yup(24)))/dconjg(yup(8))
	  bfun(9,ii,kk)=0.0d0
c...helicity(62)=(-,+,-,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(47)=(+,-,+,-,-,+)
        kk=47
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=(4*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(2))*yup(19) + dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(13))*(dconjg(yup(36)) + yup(42))))/yup(8)
        bfun(2,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))*
     -     dconjg(yup(13))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(36)) - yup(38) + yup(42)))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2)) 
     -     *yup(23) - dconjg(inpup(idq0,idk1))*yup(25)))/yup(8)
	  bfun(4,ii,kk)=(-4*dconjg(inpup(idq0,idk2))*dconjg(yup(9))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))-dconjg(inpup(idb1,idq0)) 
     -     *yup(21) +dconjg(inpup(idq0,idc2))*yup(23)))/yup(8)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))*
     -     dconjg(yup(14))*pmomup(5,idb1)*pmomup(5,idc1)*(yup(23) 
     -     + yup(24)))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(63)=(-,+,-,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(48)=(+,-,+,-,-,-)
        kk=48
        if(iqqbar.eq.0) then
	  bfun(1,ii,kk)=0.0d0
        bfun(2,ii,kk)=0.0d0
	  bfun(3,ii,kk)=0.0d0
	  bfun(4,ii,kk)=0.0d0
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=0.0d0
	  bfun(9,ii,kk)=0.0d0
c...helicity(64)=(-,+,-,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do
	  end if

c******************************************************
	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))*
     -     inpup(idk2,idk1)*(dconjg(inpup(idq0,idc2))*yup(24) - 
     -     dconjg(inpup(idc1,idq0))*yup(39))) + 
     -     yup(25)*(dconjg(inpup(idq0,idc2))*yup(24) - 
     -     dconjg(inpup(idc1,idq0))*yup(39)-dconjg(inpup(idb2,idq0))*
     -     (dconjg(yup(36)) + yup(42))))
        bfun(2,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(dconjg(inpup(idq0,idk2))*inpup(idk2,idk1)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2))
     -     *yup(23)) +dconjg(yup(2))*(dconjg(inpup(idc1,idq0))*yup(20) - 
     -     dconjg(inpup(idq0,idc2))*yup(23) + 
     -     dconjg(inpup(idb1,idq0))*(yup(22) + yup(32))))
	  bfun(3,ii,kk)=-4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))*(dconjg(yup(1)) 
     -     + dconjg(yup(9)))*dconjg(yup(16))) +dconjg(inpup(idb1,idq0)) 
     -     *dconjg(yup(1))*yup(21) - dconjg(inpup(idb1,idq0))*
     -     (dconjg(yup(1)) + dconjg(yup(9)))* (yup(21) + yup(26)) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(1))*yup(38) + 
     -     (dconjg(yup(2)) + yup(25))*(dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(11)) + dconjg(inpup(idc1,idq0))*yup(40)) + 
     -     (-(dconjg(inpup(idc1,idq0))*dconjg(yup(2))) + 
     -     dconjg(inpup(idc1,idq0))*(dconjg(yup(2)) + yup(25)))
     -     *yup(79) + dconjg(inpup(idq0,idc2))*dconjg(yup(2))*yup(80))
	  bfun(4,ii,kk)=-4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-((dconjg(yup(1)) + dconjg(yup(9)))*
     -     (-(dconjg(inpup(idb1,idq0))*yup(21)) - 
     -     dconjg(inpup(idb2,idq0))*yup(38))) + 
     -     (dconjg(yup(2)) + yup(25))*(dconjg(inpup(idq0,idc2))*
     -     (yup(23) + yup(24)) - dconjg(inpup(idc1,idq0))*(yup(20)
     -      + yup(39))))
c...helicity(64)=(-,+,-,+,+,+)
        do mm=1,4
	     bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

1211  continue
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c            nine basic functions from first by k1 <-> k2.           c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	subroutine bsecond
	implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex bfun,colmat,yup,xup,inpup,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/tryup/bfun(9,4,64),yup(100),xup(2),idp,idq0,idk1,idk2,
     &	idb1,idb2,idc1,idc2
      common/counter/ibcstate,nev
c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

	 ii=2

c...if (l5.ne.l6) then
c...hel(i,2)(l1,l2,l3,l4,l5,l6)=hel(i,1)(l1,l2,l3,l4,-l5,-l6)
c...(k_1<->k_2), where l1,...,l6 are helicities of q_b1,q_b2,
c...q_c1,q_c2,k_1,k_2;
c...if (l5.eq.l6) then
c...hel(i,2)(l1,l2,l3,l4,l5,l6)=hel(i,1)(l1,l2,l3,l4,l5,l6)(k_1<->k_2)

c...helicity(1,2)=(+,+,+,+,+,+)(k_1,k_2)=helicity(1,1)(k_2,k_1)
        kk=1
        bfun(1,ii,kk)=(4*yup(1)*(dotup(idb1,idb1)*((yup(11) - yup(12))
     -     *yup(19)*yup(27) +yup(10)*(-(dconjg(yup(5))*yup(5)) + 
     -     dconjg(yup(34))*yup(34))) + yup(10)*((-(dconjg(yup(24))*
     -     dconjg(yup(34))) +dconjg(yup(5))*yup(2))*yup(20) + 
     -     dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/xup(1)
        bfun(2,ii,kk)=(-4*(dotup(idb1,idb1)*yup(9)*(yup(11)*yup(19)*
     -     yup(27) + dconjg(yup(34))*yup(10)*yup(34)) +yup(10)* 
     -     (-(dconjg(yup(24))*dconjg(yup(34))*yup(9)*yup(20)) + 
     -     dotup(idc1,idc1)*(yup(2)*yup(29)*yup(31) + 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/xup(1)
	  bfun(3,ii,kk)=(4*yup(1)*yup(16)*((dconjg(yup(24))*
     -     dconjg(yup(34))+dconjg(yup(5))*(-yup(2)+yup(7)))*yup(20) + 
     -     dconjg(yup(17))*(dotup(idc1,idc1)*yup(17)-yup(11)*yup(32))+ 
     -     dotup(idb1,idb1)*(dconjg(yup(5))*yup(5) -dconjg(yup(34)) 
     -     *yup(34))-dconjg(yup(35))*dotup(idc1,idc1)*yup(35)))/xup(1)
	  bfun(4,ii,kk)=(4*yup(16)*(yup(9)*(dconjg(yup(17))*yup(11)*
     -     yup(32) + dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*((yup(2) -
     -     yup(7))*yup(29)*yup(31) +yup(9)*(-(dconjg(yup(17))*yup(17))+ 
     -     dconjg(yup(35))*yup(35)))))/xup(1)
	  bfun(5,ii,kk)=(2*yup(1)*yup(16)*yup(27)*yup(29))/xup(1)
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(29)*(yup(9)*yup(16)*
     -     yup(19)+(yup(7)*yup(16) + yup(2)*(-yup(12) + yup(18)))*
     -     yup(31)) +4*dconjg(yup(34))*yup(9)*(yup(12) - yup(18))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34))+ 
     -     4*dconjg(yup(35))*(dotup(idc1,idc1)*yup(9)*(yup(12)-yup(18))*
     -     yup(35) + 2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     inpup(idq0,idb1)*yup(27)*yup(36)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(17,2)=(-,-,-,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(2,2)=m(+,+,+,+,+,-)=helicity(3,1)(k_2,k_1)
        kk=2
        bfun(1,ii,kk)=(4*(dconjg(yup(1))*(dotup(idb1,idb1)*yup(11)*
     -     yup(19)*yup(27)+dconjg(yup(34))*yup(10)*(-(dconjg(yup(24))
     -     *yup(20)) + dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*
     -     (dconjg(yup(8))*dotup(idb1,idb1)*yup(19)*yup(31) + yup(10)*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(1))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
        bfun(2,ii,kk)=(-4*dconjg(yup(9))*(dotup(idb1,idb1)*(yup(4)
     -     *(dconjg(yup(8))*dconjg(yup(33))-dconjg(yup(4))*yup(10))+ 
     -     yup(11)*yup(19)*yup(27) +dconjg(yup(34))*yup(10)*yup(34))+ 
     -     yup(10)*(dconjg(yup(24))*(-(dconjg(yup(34))*yup(20))+yup(4) 
     -     *yup(25))+dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/yup(8)
	  bfun(3,ii,kk)=(-4*yup(16)*(dconjg(yup(1))*(dconjg(yup(17))*
     -     yup(11)*yup(32)+dconjg(yup(34))*(-(dconjg(yup(24))*yup(20))+ 
     -     dotup(idb1,idb1)*yup(34))) +dotup(idc1,idc1)*(yup(25)*
     -     (-yup(19)+yup(30))*yup(31)+dconjg(yup(1))*(-(dconjg(yup(17))
     -     *yup(17))+dconjg(yup(35))*yup(35)))))/yup(8)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*yup(16)*(-(yup(4)*
     -     (dconjg(yup(4))*dotup(idb1,idb1) + (-dconjg(yup(24)) +
     -     dconjg(yup(33)))*yup(25))) +dconjg(yup(17))*
     -     (-(dotup(idc1,idc1)*yup(17)) + yup(11)*yup(32))
     -     + dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34)) + dconjg(yup(35))*dotup(idc1,idc1)
     -     *yup(35)))/yup(8)
	  bfun(5,ii,kk)=(2*dconjg(yup(9))*yup(16)*yup(28)*yup(29))/yup(8)
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(24))*((yup(12)
     -     - yup(18))*(-(dconjg(yup(34))*yup(20)) +yup(4)*yup(25)) + 
     -     2*dotup(idc1,idk1)*yup(14)*yup(29))+dotup(idb1,idb1)*(yup(12) 
     -     -yup(18))*(-(dconjg(yup(4))*yup(4))+dconjg(yup(34))*yup(34))+ 
     -     dotup(idc1,idc1)*(-(yup(16)*yup(19)*yup(29)) +dconjg(yup(35)) 
     -     *(yup(12) - yup(18))*yup(35))))/yup(8)
	  bfun(9,ii,kk)=(4*dconjg(yup(5))*yup(11)*yup(20)*yup(30))/yup(8)
c...helicity(18,2)=m(-,-,-,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(3,2)=(+,+,+,+,-,+)=helicity(2,1)(k_2,k_1)
        kk=3
        bfun(1,ii,kk)=(4*dconjg(yup(10))*yup(1)*((-(dconjg(yup(24))
     -     *dconjg(yup(34))) +dconjg(yup(5))*yup(2))*yup(20) + 
     -     dotup(idb1,idb1)*(-(dconjg(yup(5))*yup(5))+dconjg(yup(34)) 
     -     *yup(34)) + dotup(idc1,idc1)*(-(dconjg(yup(13))*yup(13)) + 
     -     dconjg(yup(35))*yup(35)) + dconjg(yup(18))*yup(13)*
     -     (-dconjg(yup(38)) + yup(36))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*(dotup(idc1,idc1)*
     -     (yup(2)*(-yup(19)+yup(29))*yup(31)+yup(9)*(-(dconjg(yup(13)) 
     -     *yup(13))+dconjg(yup(35))*yup(35)))+yup(9)*(dconjg(yup(34))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34)) + 
     -     dconjg(yup(18))*yup(13)*yup(36))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*yup(1)*(dotup(idb1,idb1)*(dconjg(yup(18))
     -     *yup(19)*(yup(28) - yup(31)) + dconjg(yup(16))*
     -     (-(dconjg(yup(5))*yup(5)) + dconjg(yup(34))*yup(34))) + 
     -     dconjg(yup(16))*((-(dconjg(yup(24))*dconjg(yup(34))) + 
     -     dconjg(yup(5))*yup(2))*yup(20) + dconjg(yup(35))* 
     -     dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*(dotup(idb1,idb1)*(dotup(idc1,idc1)
     -     *yup(8)*yup(19)*yup(31)+yup(9)*(dconjg(yup(18))*yup(19)
     -     *yup(28) +dconjg(yup(16))*dconjg(yup(34))*yup(34)))
     -     +dconjg(yup(16))*(-(dconjg(yup(24))*dconjg(yup(34))*yup(9)*
     -     yup(20)) +dotup(idc1,idc1)*(yup(2)*yup(29)*yup(31) + 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
	  bfun(5,ii,kk)=(2*dconjg(yup(10))*yup(1)*yup(27)*yup(30))/
     -	 dconjg(yup(8))
	  bfun(8,ii,kk)=(4*((-(dconjg(yup(10))*dconjg(yup(24))) + 
     -     dconjg(yup(14))*dotup(idb1,idb1))*(dconjg(yup(18))*yup(9)
     -     *yup(14)+2*dotup(idc1,idk1)*yup(4)*yup(27))+(dconjg(yup(12)) 
     -     - dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (2*dconjg(yup(35))*dotup(idc1,idk1)*yup(8)*yup(35) + 
     -     (dconjg(yup(12))-dconjg(yup(18)))*(yup(2)*yup(29)*yup(31)+ 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(yup(15))*(dotup(idb1,idb1)*yup(5)
     -     -yup(2)*yup(20))*yup(28)+4*dconjg(yup(35))*dotup(idc1,idc1)*
     -     yup(8)*yup(35))/dconjg(yup(8))
c...helicity(19,2)=(-,-,-,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(4,2)=(+,+,+,+,-,-)=helicity(4,1)(k_2,k_1)
        kk=4
        bfun(1,ii,kk)=(4*dconjg(yup(10))*(dotup(idc1,idc1)*
     -     ((dconjg(yup(3))*dconjg(yup(12)) - dconjg(yup(1))
     -     *dconjg(yup(13)))*yup(13)+yup(25)*yup(30)*yup(31)+ 
     -     dconjg(yup(1))*dconjg(yup(35))*yup(35)) +dconjg(yup(1))*
     -     (dconjg(yup(34))*(-(dconjg(yup(24))*yup(20)) + 
     -     dotup(idb1,idb1)*yup(34)) +dconjg(yup(18))*yup(13) 
     -     *yup(36))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*dconjg(yup(10))*(dconjg(yup(24))
     -     *(dconjg(yup(34))*yup(20)+yup(4)*(dconjg(yup(7))-yup(25)))+ 
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) + dotup(idc1,idc1)*(dconjg(yup(13))*yup(13) - 
     -     dconjg(yup(35))*yup(35)) - dconjg(yup(18))*yup(13)
     -     *yup(36)))/xup(2)
	  bfun(3,ii,kk)=(-4*(dconjg(yup(1))*dconjg(yup(18))*
     -     dotup(idb1,idb1)*yup(19)*yup(28) +dconjg(yup(16))*
     -     (dconjg(yup(1))*dconjg(yup(34))*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(1))*dconjg(yup(35))
     -     *yup(35)))))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*(dotup(idb1,idb1)*
     -     ((dconjg(yup(7))*dconjg(yup(14)) - dconjg(yup(4))
     -     *dconjg(yup(16)))*yup(4) +dconjg(yup(18))*yup(19)*yup(28)+ 
     -     dconjg(yup(16))*dconjg(yup(34))*yup(34)) +dconjg(yup(16))*
     -     (dconjg(yup(24))*(-(dconjg(yup(34))*yup(20))+yup(4)*yup(25))+ 
     -     dconjg(yup(35))*dotup(idc1,idc1)*yup(35))))/xup(2)
	bfun(5,ii,kk)=(2*dconjg(yup(9))*dconjg(yup(10))*yup(28)*yup(30))
     -	 /xup(2)
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*(dconjg(yup(24))*(dconjg(yup(10))
     -     *yup(14) -dconjg(yup(34))*yup(20) + yup(4)*yup(25)) - 
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) + dconjg(yup(14))
     -     *yup(14) -dconjg(yup(34))*yup(34)) +dconjg(yup(35))* 
     -     dotup(idc1,idc1)*yup(35)))/xup(2)
        bfun(9,ii,kk)=(-4*dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -    inpup(idk1,idk2)*yup(28)*yup(32))/xup(2)
c...helicity(20,2)=(-,-,-,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(5,2)=(+,+,+,-,+,+)=helicity(5,1)(k_2,k_1)
        kk=5
        bfun(1,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (-(yup(1)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19)) - 
     -     (dconjg(inpup(idc1,idq0))*yup(5)*yup(10) - 
     -     dconjg(inpup(idk2,idk1))*yup(19)*yup(27))*yup(31)) - 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idb2,idk1))
     -     *yup(31)*yup(32) + dconjg(inpup(idb2,idq0))*yup(1)*
     -     (yup(22) + yup(32)))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (yup(9)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19)+dconjg(yup(4))
     -     *yup(4)*(-(dconjg(inpup(idq0,idk1))*yup(10)) - 
     -     dconjg(inpup(idk2,idk1))*yup(19)))+dconjg(yup(35))*yup(10)*
     -     (dconjg(inpup(idb2,idk1))*yup(22)*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+yup(32)))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*yup(16)*
     -     (dconjg(inpup(idc1,idq0))*dotup(idb1,idb1)*yup(5)*
     -     yup(31)+ dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*yup(31)
     -     *yup(32) +dconjg(inpup(idb2,idq0))*yup(1)*(yup(22) +
     -     yup(32))) +dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))
     -     *yup(31)*yup(32)) +dconjg(inpup(idq0,idk2))*yup(1)*
     -     (yup(22) + yup(32)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(16)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*dotup(idb1,idb1)
     -     *yup(4)) +dconjg(yup(35))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31) +dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+
     -     yup(32))) +dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))
     -     *yup(22)*yup(31)) +dconjg(inpup(idq0,idk2))*yup(9)*
     -     (yup(22) + yup(32)))))/xup(1)
	bfun(5,ii,kk)=(2*dconjg(inpup(idk1,idq0))*pmomup(5,idc1)*yup(16)
     -    *(yup(27) + yup(28))*yup(29))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(14))*yup(9)*(dotup(idb1,idb1)*yup(14) - yup(16)
     -     *yup(23))) -dconjg(inpup(idq0,idk1))*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*yup(4)*(yup(12)-yup(18)))-dconjg(yup(14)) 
     -     *yup(12)*(dotup(idb1,idb1)*yup(14) - yup(16)*yup(23)))-
     -     dconjg(yup(35))*((yup(12)-yup(18))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31)+dconjg(inpup(idb2,idq0))*yup(9)*(yup(22) +
     -     yup(32))) +2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     yup(27)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -    pmomup(5,idc1)*yup(28)*yup(35))/xup(1)
c...helicity(21,2)=(-,-,-,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(6,2)=(+,+,+,-,+,-)=helicity(7,1)(k_2,k_1)
        kk=6
        bfun(1,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idc1)*
     -     (-(dotup(idb1,idb1)*(-(dconjg(inpup(idq0,idk1))*
     -     dconjg(yup(8))) +dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15))*yup(19))+ yup(10)*
     -     (dconjg(inpup(idq0,idk1))*yup(25)*yup(30) - 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     (yup(22) + yup(32)))))/yup(8)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*(dotup(idb1,idb1)
     -     *yup(19)*(dconjg(inpup(idq0,idc2))*yup(14) - 
     -     dconjg(inpup(idc1,idq0))*yup(15)-dconjg(inpup(idq0,idk2)) 
     -     *yup(31)) + dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     yup(10)*(-yup(21) + yup(22) + yup(32))))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idc1)*yup(16)*
     -     (dconjg(inpup(idq0,idk1))*yup(25)*(yup(19) - yup(30)) + 
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(17)) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35)))*
     -     (yup(22) + yup(32))))/yup(8)
	  bfun(4,ii,kk)=(4*dconjg(yup(9))*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(17))+dconjg(inpup(idb2,idq0))*dconjg(yup(35)))*
     -     pmomup(5,idc1)*yup(16)*(yup(21) - yup(22) - yup(32)))/yup(8)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(14))*(dotup(idb1,idb1)
     -     *yup(14) - yup(16)*yup(23)))+ dconjg(inpup(idb2,idq0))
     -     *dconjg(yup(35))*(-2*dotup(idc1,idk1)*yup(17)+(yup(12)-
     -     yup(18))*(yup(21)- yup(22) - yup(32)))))/yup(8)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idq0))*pmomup(5,idc1)*
     -     (dconjg(yup(5))*yup(20) + yup(25)*yup(28))*yup(30))/yup(8)
c...helicity(22,2)=(-,-,-,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(7,2)=(+,+,+,-,-,+)=helicity(6,1)(k_2,k_1)
        kk=7
        bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (dconjg(inpup(idb2,idq0))*yup(1)*(-((dconjg(yup(11))
     -     + dconjg(yup(18)))*yup(13)) +dconjg(yup(35))*(yup(22) 
     -     + yup(32))) +yup(31)*(dconjg(inpup(idc1,idq0))*
     -     dotup(idb1,idb1)*yup(5) +dconjg(inpup(idb2,idk1))*
     -     (-(dconjg(yup(18))*yup(13)) +dconjg(yup(35))*yup(32) 
     -     ))))/dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*dotup(idb1,idb1)
     -     *yup(4)) +dconjg(inpup(idb2,idk1))*(-(dconjg(yup(11))
     -     *yup(13)) +dconjg(yup(35))*yup(22))*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(-((dconjg(yup(11)) +
     -     dconjg(yup(18)))*yup(13)) + dconjg(yup(35))*(yup(22) + 
     -     yup(32)))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(inpup(idc1,idq0))*dconjg(yup(16))*yup(5) - 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(18))*yup(19))*yup(31)+ 
     -     dconjg(yup(16))*dconjg(yup(35))*(dconjg(inpup(idb2,idk1))
     -     *yup(31)*yup(32) +dconjg(inpup(idb2,idq0))*yup(1)*
     -     (yup(22) + yup(32)))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(11))*dotup(idb1,idb1)*yup(19)*yup(31) - 
     -     dconjg(yup(16))*(-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     dotup(idb1,idb1)*yup(4)) +dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*yup(22)*yup(31) + 
     -     dconjg(inpup(idb2,idq0))*yup(9)*(yup(22)+yup(32))))))/
     -     dconjg(yup(8))
	  bfun(5,ii,kk)=(2*dconjg(inpup(idk1,idq0))*dconjg(yup(10))*
     -     pmomup(5,idc1)*(yup(27) + yup(28))*yup(30))/dconjg(yup(8))
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18)))*
     -     dotup(idb1,idb1)*yup(4)) - 2*dotup(idc1,idk1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(yup(35))*yup(8)*yup(22)+ 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(10))*yup(27)*yup(30)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*yup(22)*yup(31) + dconjg(inpup(idb2,idq0))
     -     *yup(9)*(dconjg(yup(12))*(yup(22)+yup(32))-dconjg(yup(18))*
     -     (yup(22) - yup(26) + yup(32))))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(11))*
     -     dconjg(yup(35))*pmomup(5,idc1)*yup(35))/dconjg(yup(8))
c...helicity(23,2)=(-,-,-,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(8,2)=(+,+,+,-,-,-)=helicity(8,1)(k_2,k_1)
        kk=8
        bfun(1,ii,kk)=(-4*dconjg(yup(1))*dconjg(yup(10))*pmomup(5,idc1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))
     -     *inpup(idk2,idk1)*yup(13) - dconjg(inpup(idq0,idk1))*yup(25)*
     -     yup(30) + dconjg(inpup(idb2,idq0))*(-((dconjg(yup(11)) +
     -     dconjg(yup(18)))*yup(13)) +dconjg(yup(35))*(yup(22) 
     -     + yup(32)))))/xup(2)
        bfun(2,ii,kk)=(-4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idc1)*((dconjg(yup(11)) -
     -     dconjg(yup(12)) +dconjg(yup(18)))*yup(13) + 
     -      dconjg(yup(35))*(yup(21) - yup(22) - yup(32))))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(16))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*yup(25)*yup(30)) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     (yup(22) + yup(32))))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     dconjg(yup(16))*dconjg(yup(35))*pmomup(5,idc1)*
     -     (yup(21) - yup(22) - yup(32)))/xup(2)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*dconjg(yup(35))
     -     *pmomup(5,idc1)*(yup(21)-yup(22)+yup(26)-yup(32)))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(24,2)=(-,-,-,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(9,2)=(+,-,-,+,+,+)=helicity(9,1)(k_2,k_1)
        kk=9
        bfun(1,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*yup(1)*
     -     (dconjg(yup(13))*yup(13)*(yup(11)+ yup(18)) +dconjg(yup(8))
     -     *yup(7)*yup(30) +yup(10)*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(13)*(dconjg(yup(3))*yup(7) +dconjg(yup(13))*(yup(11) 
     -     +yup(18)))+yup(10)*(-(dconjg(yup(3))*yup(3))-dconjg(yup(6)) 
     -     *yup(6) +(yup(27) + yup(28))*(yup(29) + yup(30)))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(1)*
     -     (dconjg(yup(6))*yup(7)*yup(17) +yup(11)*yup(19)*yup(29) + 
     -     yup(18)*yup(19)*yup(29) +yup(16)*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/xup(1)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (-((yup(11) - yup(12) + yup(18))*yup(19)*yup(29)) + 
     -     yup(16)*(dconjg(yup(3))*yup(3)- dconjg(inpup(idb1,idq0))
     -     *inpup(idq0,idk1)*yup(6) - (yup(27) + yup(28))*(yup(29)
     -     + yup(30)))))/xup(1)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(yup(29)+yup(30))*(yup(19)-yup(27)
     -     -yup(28)+yup(31)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(25,2)=(-,+,+,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(10,2)=(+,-,-,+,+,-)=helicity(11,1)(k_2,k_1)
        kk=10
        bfun(1,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(-(dconjg(yup(1))
     -     *dconjg(yup(13))*yup(13)*(yup(11)+yup(18)))-dconjg(yup(2))*
     -     (yup(18)*yup(19) + yup(10)*yup(27))*yup(31) + yup(10)*(
     -     -(dconjg(yup(1))*(yup(27) + yup(28))*(yup(29) + yup(30))) - 
     -     yup(25)*yup(27)*yup(31))))/yup(8)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(yup(4)*
     -     (-(dconjg(inpup(idc2,idk2))*dconjg(yup(8))*dconjg(yup(13))*
     -     inpup(idk1,idb2)) +dconjg(yup(4))*yup(10)*(dconjg(yup(2))
     -     + yup(25))) + dconjg(yup(9))*(dconjg(yup(13))*yup(13)
     -     *(yup(11) + yup(18)) +yup(10)*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))))/yup(8)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(1))*
     -     yup(11)*yup(19)*yup(29) +yup(18)*yup(19)*(dconjg(yup(1))
     -     *yup(29)+yup(25)*yup(31))+yup(16)*(dconjg(yup(1))*(yup(27) 
     -     + yup(28))*(yup(29) + yup(30)) + dconjg(yup(2))*yup(27)*
     -     yup(31) +yup(25)*yup(27)*yup(31))))/yup(8)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idc2,idk2))*inpup(idq0,idk1)*yup(4)*yup(19)
     -     *yup(25) -dconjg(yup(9))*(yup(11) + yup(18))*yup(19)*
     -     yup(29)+yup(16)*(-(dconjg(yup(4))*yup(4)*(dconjg(yup(2))
     -     + yup(25))) -dconjg(yup(9))*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))))/yup(8)
	  bfun(5,ii,kk)=(-2*dconjg(yup(8))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (yup(27) + yup(28))*(yup(29) + yup(30)))/yup(8)
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(yup(12) - yup(18))*(dconjg(yup(2)) + yup(25)) + 
     -     (dconjg(yup(9))*yup(18)*yup(19) + (2*dconjg(yup(8))
     -     *dotup(idc1,idk1)+dconjg(yup(9))*(yup(12) - yup(18)))*
     -     (yup(27) + yup(28)))*(yup(29) + yup(30))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(26,2)=(-,+,+,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(11,2)=(+,-,-,+,-,+)=helicity(10,1)(k_2,k_1)
        kk=11
        bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(1)*(-(dconjg(yup(8))*yup(8))+yup(19)*(yup(27)+yup(28))- 
     -     (yup(29)+yup(30))*(yup(27)+yup(28)-yup(31))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(9)*(dconjg(yup(3))*yup(3)+dconjg(yup(6))*yup(6)+(yup(27) 
     -     + yup(28))*(yup(19) - yup(29) - yup(30)) - yup(19)*yup(31)
     -     ))/dconjg(yup(8))
	bfun(3,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(1)*(yup(19) - yup(29) - yup(30))*(yup(27) + yup(28)
     -     - yup(31)))/dconjg(yup(8))
	bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     yup(9)*(-(dconjg(yup(3))*yup(3)) -dconjg(yup(6))*yup(6) + 
     -     dconjg(inpup(idq0,idk2))*inpup(idq0,idk1)*yup(8) - (yup(27)
     -     + yup(28))*(yup(19) - yup(29) - yup(30))))/dconjg(yup(8))
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (dconjg(yup(14))*(dconjg(yup(10))+dconjg(yup(16)))*yup(14)- 
     -     (dconjg(yup(12)) - dconjg(yup(18)))*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/dconjg(yup(8))
	bfun(9,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(15))
     -     *yup(5) + yup(8)*yup(28))*(yup(29) + yup(30)))/dconjg(yup(8))
c...helicity(27,2)=(-,+,+,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(12,2)=(+,-,-,+,-,-)=helicity(12,1)(k_2,k_1)
        kk=12
        bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(yup(1))*yup(19)*(yup(27)+yup(28))) +dconjg(yup(1)) 
     -     *yup(27)*yup(29) +dconjg(yup(1))*yup(28)*yup(29) + 
     -     dconjg(yup(1))*yup(27)*yup(30)+dconjg(yup(1))*yup(28)*yup(30) 
     -     +dconjg(yup(2))*yup(27)*yup(31) - dconjg(yup(7))*yup(27)*
     -     yup(31) +yup(25)*yup(27)*yup(31)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*yup(4)*(dconjg(yup(2))-dconjg(yup(7))
     -     + yup(25)) -dconjg(yup(9))*(yup(27) + yup(28))*(yup(19) -
     -     yup(29) - yup(30))))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(yup(1))*yup(19)*(yup(27)+yup(28)))+dconjg(yup(1))* 
     -     yup(27)*yup(29)+dconjg(yup(1))*yup(28)*yup(29)+dconjg(yup(1))
     -     *yup(27)*yup(30)+dconjg(yup(1))*yup(28)*yup(30)+
     -     dconjg(yup(2))*yup(27)*yup(31)-dconjg(yup(7))*yup(27)*yup(31) 
     -     +yup(25)*yup(27)*yup(31)))/xup(2)
	bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*yup(4)*(dconjg(yup(2)) - dconjg(yup(7))
     -     + yup(25)) - dconjg(yup(9))*(yup(27) + yup(28))*
     -     (yup(19) - yup(29) - yup(30))))/xup(2)
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *dconjg(yup(10))*dconjg(yup(12))*yup(14) - dconjg(yup(9))
     -     *dconjg(yup(10))*dconjg(yup(14))*yup(14) + (dconjg(yup(4))
     -     *dconjg(yup(12)) -dconjg(yup(9))*dconjg(yup(14)))*
     -     dconjg(yup(16))*yup(14) + dconjg(yup(4))*dconjg(yup(12))
     -     *yup(4)*(dconjg(yup(2)) + yup(25)) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(29) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(30) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(27)*(yup(29)+yup(30)))+4*dconjg(yup(18)) 
     -     *pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(dconjg(yup(2))+yup(25))+dconjg(yup(9))*(yup(27)
     -     + yup(28))*(yup(29) + yup(30))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(28,2)=(-,+,+,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(13,2)=(+,-,-,-,+,+)=helicity(13,1)(k_2,k_1)
        kk=13
        bfun(1,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(10)*yup(31)) - 
     -     dconjg(yup(13))*dotup(idc1,idc1)*yup(13)*
     -     (-(dconjg(inpup(idq0,idk2))*yup(1)) + 
     -     dconjg(inpup(idk2,idk1))*yup(31)) -yup(1)*( 
     -     dconjg(inpup(idc1,idk2))*yup(19)*yup(24) + 
     -     dconjg(inpup(idc1,idq0))*yup(10)*
     -     (yup(23) + yup(24)))*yup(34)))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*yup(9)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*dotup(idc1,idc1)
     -     *yup(13)) + dconjg(yup(4))*(dconjg(inpup(idk2,idk1))*yup(19)
     -     *yup(24) +dconjg(inpup(idq0,idk1))*yup(10)*(yup(23) +
     -     yup(24))) +(dconjg(inpup(idc1,idk2))*yup(19)*yup(24) + 
     -     dconjg(inpup(idc1,idq0))*yup(10)*(yup(23) + yup(24)))*
     -     yup(34)))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*yup(1)*yup(17)*yup(19) + 
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(16) + 
     -     dconjg(inpup(idk2,idk1))*yup(19)*yup(29))*yup(31)) + 
     -     yup(1)*(dconjg(inpup(idc1,idk2))*yup(19)*yup(23) + 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))
     -     *yup(34)))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(yup(4))*(dconjg(inpup(idk2,idk1))*yup(19)*yup(23) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*(yup(23) + yup(24))) + 
     -     (dconjg(inpup(idc1,idk2))*yup(19)*yup(23) + 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))*
     -     yup(34)))/xup(1)
	bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*yup(9)
     -     *yup(28)*(yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*(yup(12) - yup(18))*
     -     (yup(23)+yup(24))*(dconjg(inpup(idq0,idk1))*dconjg(yup(4))+ 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(14)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*pmomup(5,idb1)*
     -     yup(27)*yup(28)*(yup(29) + yup(30)))/xup(1)
c...helicity(29,2)=(-,+,+,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(14,2)=(+,-,-,-,+,-)=helicity(15,1)(k_2,k_1)
        kk=14
        bfun(1,ii,kk)=(-4*dconjg(yup(1))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dotup(idc1,idc1)*yup(13)) +dconjg(inpup(idc1,idk2)) 
     -     *yup(19)*(-(dconjg(yup(2))*yup(5))+yup(24)*yup(34)) + 
     -     dconjg(inpup(idc1,idq0))*yup(10)*(-(yup(5)*(dconjg(yup(2))
     -     + yup(25))) +(yup(23) + yup(24))*yup(34))))/yup(8)
        bfun(2,ii,kk)=(-4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(19)*yup(31) - 
     -     dconjg(yup(9))*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))
     -     *dotup(idc1,idc1)*yup(13)) +(dconjg(inpup(idc1,idk2)) 
     -     *yup(19)*yup(24) +dconjg(inpup(idc1,idq0))*yup(10)*
     -     (yup(23) + yup(24)))*yup(34))))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(inpup(idc1,idk2))*yup(19)*(-(yup(5)*yup(25)) +
     -     yup(23)*yup(34)) +dconjg(inpup(idc1,idq0))*yup(16)*
     -     (-(yup(5)*(dconjg(yup(2)) + yup(25))) +(yup(23) + yup(24)) 
     -     *yup(34))))/yup(8)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dotup(idc1,idc1)*yup(19)*yup(25)*yup(31) - 
     -     dconjg(yup(9))*(dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*
     -     yup(17)*yup(19) + (dconjg(inpup(idc1,idk2))*yup(19)*yup(23)+ 
     -     dconjg(inpup(idc1,idq0))*yup(16)*(yup(23) + yup(24)))
     -     *yup(34))))/yup(8)
	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*yup(27)*(yup(29) + yup(30)))/yup(8)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk2))*(dconjg(yup(9))*dconjg(yup(14))
     -     *yup(18) +2*dconjg(yup(4))*dotup(idc1,idk1)*yup(27)) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(9))*(-yup(12)+yup(18))
     -     *yup(34)))/yup(8)
	bfun(9,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*pmomup(5,idb1)
     -   *yup(15)*(dconjg(yup(2)) + yup(25))*yup(28))/yup(8)
c...helicity(30,2)=(-,+,+,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(15,2)=(+,-,-,-,-,+)=helicity(14,1)(k_2,k_1)
        kk=15
        bfun(1,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk1)) 
     -     *yup(19))*yup(31) +dconjg(inpup(idc1,idq0))*yup(1)*
     -     (yup(23) + yup(24) - yup(33))*yup(34)))/dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*yup(9)*
     -     (yup(23) + yup(24) - yup(33))*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))+dconjg(inpup(idc1,idq0))*yup(34)))/
     -     dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk2)) 
     -     *yup(8))*yup(31) -dconjg(inpup(idc1,idq0))*yup(1)*
     -     (dconjg(inpup(idq0,idk2))*inpup(idc2,idk2) - yup(23)
     -     - yup(24))*yup(34)))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*dconjg(yup(16))*pmomup(5,idb1)*yup(9)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(11))*yup(27)) + 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(4))*(yup(23)+yup(24)
     -     - yup(33)) +dconjg(inpup(idc1,idq0))*(yup(23) + yup(24))*
     -     yup(34)))/dconjg(yup(8))
	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     pmomup(5,idb1)*yup(9)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(4)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(11))*
     -     pmomup(5,idb1)*yup(5)*(yup(29) + yup(30)))/dconjg(yup(8))
c...helicity(31,2)=(-,+,+,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(16,2)=(+,-,-,-,-,-)=helicity(16,1)(k_2,k_1)
        kk=16
        bfun(1,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(1))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(5)*(dconjg(yup(2))
     -     - dconjg(yup(7)) +yup(25)) - (yup(23) + yup(24)-yup(33))
     -     *yup(34)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(23) + yup(24) 
     -     - yup(33))*yup(34))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(1))*
     -     dconjg(yup(16))*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))
     -     *inpup(idk2,idk1)*yup(5)) -yup(5)*(dconjg(yup(2)) + 
     -     yup(25)) +(-(dconjg(inpup(idq0,idk2))*inpup(idc2,idk2)) + 
     -     yup(23) + yup(24))*yup(34)))/xup(2)
	  bfun(4,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -     dconjg(yup(16))*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *inpup(idc2,idk2) -yup(23) - yup(24))*yup(34))/xup(2)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -    (dconjg(yup(12)) - dconjg(yup(18)))*
     -    pmomup(5,idb1)*(yup(23) + yup(24))*yup(34))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(32,2)=(-,+,+,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

        if (ibcstate.eq.1) go to 1222

c...helicity(33,2)=(+,+,-,+,+,+)=helicity(33,1)(k_2,k_1)
        kk=33
        bfun(1,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(1)*yup(10)
     -     *(-(dconjg(yup(38))*inpup(idq0,idb1)) + inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*yup(9)*yup(10)*
     -     (inpup(idq0,idk1)*yup(2)*yup(29) + dconjg(yup(35))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36))))/xup(1)
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*yup(1)*yup(16)*
     -     (-(inpup(idq0,idb1)*(-(dconjg(yup(35))*dconjg(yup(38))) + 
     -     dconjg(yup(17))*yup(12))) +inpup(idq0,idb1)*
     -     (dconjg(yup(17))*(yup(11) + yup(18)) -dconjg(yup(35))* 
     -     (dconjg(yup(42)) + yup(36)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*yup(16)*
     -     (inpup(idq0,idk1)*(yup(2)-yup(7))*yup(29)+inpup(idq0,idb1)*
     -     (-(dconjg(yup(17))*(yup(11) + yup(18))) +dconjg(yup(35)) 
     -     *(dconjg(yup(42)) + yup(36)))))/xup(1)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(-((dconjg(yup(38))+dconjg(yup(41)))*
     -     inpup(idq0,idb1)) + inpup(idq0,idb1)*(dconjg(yup(42)) +
     -     yup(36))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(49,2)=(-,-,+,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(34,2)=(+,+,-,+,+,-)=helicity(35,1)(k2,k1)
        kk=34
        bfun(1,ii,kk)=(4*pmomup(5,idc1)*(-(dotup(idb1,idb1)*
     -     inpup(idq0,idk1)*yup(18)*yup(19)*yup(31)) - 
     -     yup(10)*(dconjg(yup(5))*dotup(idb1,idb1)*
     -     inpup(idq0,idc1)*yup(31) + dconjg(yup(35))*
     -     (dconjg(yup(42))*inpup(idk1,idb1)*yup(31) + 
     -     dconjg(yup(1))*inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36))))))/yup(8)
        bfun(2,ii,kk)=(4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)*
     -     (dotup(idb1,idb1)*(-(dconjg(yup(8))*dconjg(yup(33))) + 
     -     dconjg(yup(4))*yup(10))-dconjg(yup(24))*yup(10)*yup(25))+ 
     -     dconjg(yup(9))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     yup(10)*(dconjg(yup(42)) + yup(36))))/yup(8)
	  bfun(3,ii,kk)=(4*pmomup(5,idc1)*yup(16)*
     -     (dconjg(yup(5))*dotup(idb1,idb1)*inpup(idq0,idc1)*
     -     yup(31)+inpup(idk1,idb1)*(dconjg(yup(35))*dconjg(yup(42)) - 
     -     dconjg(yup(17))*yup(18))*yup(31) + dconjg(yup(1))*
     -     inpup(idq0,idb1)*(-(dconjg(yup(17))*(yup(11) + yup(18))) + 
     -     dconjg(yup(35))*(dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(16)*(inpup(idq0,idk1)
     -     *yup(4)*(dconjg(yup(4))*dotup(idb1,idb1) + 
     -     (-dconjg(yup(24))+dconjg(yup(33)))*yup(25))+dconjg(yup(9))
     -     *inpup(idq0,idb1)*(-(dconjg(yup(17))*(yup(11) + yup(18))) + 
     -     dconjg(yup(35))*(dconjg(yup(42)) + yup(36)))))/yup(8)
 	  bfun(5,ii,kk)=(2*inpup(idq0,idk1)*pmomup(5,idc1)*yup(16)*
     -     (yup(27) + yup(28))*yup(29))/yup(8)
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)
     -     *(yup(12) - yup(18))*(dconjg(yup(4))*dotup(idb1,idb1) - 
     -     dconjg(yup(24))*yup(25)) +2*dotup(idc1,idk1)*
     -     (-(inpup(idq0,idk1)*yup(16)*yup(27)*yup(29)) + 
     -     dconjg(yup(8))*dconjg(yup(35))*inpup(idq0,idb1)*yup(36)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*(dconjg(yup(41))*
     -     inpup(idq0,idb1)*yup(18)+inpup(idq0,idb1)*(yup(12)-yup(18))*
     -     (dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(9,ii,kk)=(-4*dconjg(yup(3))*dconjg(yup(35))*
     -    inpup(idq0,idb1)*pmomup(5,idc1)*yup(11))/yup(8)
c...helicity(50,2)=(-,-,+,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(35,2)=(+,+,-,+,-,+)=helicity(34,1)(k2,k1)
        kk=35
        bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*yup(1)*
     -     (-(dconjg(yup(38))*inpup(idq0,idk2)*yup(13)) + 
     -     inpup(idq0,idk2)*yup(13)*(dconjg(yup(42))+yup(36)) - 
     -     dconjg(yup(35))*(-(dconjg(yup(38))*inpup(idq0,idb1)) + 
     -     inpup(idq0,idb1)*(dconjg(yup(42))+yup(36)))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*pmomup(5,idc1)*yup(9)*
     -     (inpup(idq0,idk1)*yup(2)*(yup(19) - yup(29)) + 
     -     (-(dconjg(yup(35))*inpup(idq0,idb1)) + inpup(idq0,idk2)
     -     *yup(13))*(dconjg(yup(42)) + yup(36))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*yup(1)*(dotup(idb1,idb1)
     -     *yup(19)*(inpup(idq0,idk2)*(yup(27) + yup(28)) - 
     -     inpup(idq0,idk2)*yup(31))-dconjg(yup(16))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) + inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*
     -     (-(dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)*(yup(27) + 
     -     yup(28))) +inpup(idq0,idk1)*(dotup(idb1,idb1)
     -     *yup(8)*yup(19) +dconjg(yup(16))*yup(2)*yup(29)) + 
     -     dconjg(yup(16))*dconjg(yup(35))*inpup(idq0,idb1)
     -     *(dconjg(yup(42)) + yup(36))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*(inpup(idq0,idk2)*
     -     ((dconjg(yup(10))*dconjg(yup(24)) - dconjg(yup(14))
     -     *dotup(idb1,idb1))*yup(14)+2*dconjg(yup(35))*dotup(idc1,idk1)
     -     *yup(35))+(dconjg(yup(12))-dconjg(yup(18)))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) +inpup(idq0,idb1) 
     -     *(dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*inpup(idq0,idk2)*pmomup(5,idc1)*
     -     (dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idq0,idb1)
     -     *yup(5) +yup(2)*yup(28)*yup(29)))/dconjg(yup(8))
c...helicity(51,2)=(-,-,+,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(36,2)=(+,+,-,+,-,-)=helicity(36,1)(k2,k1)
        kk=36
        bfun(1,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (-(dconjg(yup(5))*dotup(idb1,idb1)*inpup(idq0,idc1)
     -     *yup(31)) +dconjg(yup(42))*(-(dconjg(yup(35))*
     -     inpup(idk1,idb1)) -inpup(idk2,idk1)*yup(13))*yup(31) - 
     -     dconjg(yup(1))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     (dconjg(yup(42))+yup(36))+dconjg(yup(1))*inpup(idq0,idk2)
     -     *yup(13)*(dconjg(yup(42)) + yup(36))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idc1)*
     -     (inpup(idq0,idk1)*yup(4)*(dconjg(yup(4))*dotup(idb1,idb1)+ 
     -     dconjg(yup(24))*(dconjg(yup(7)) - yup(25)))- dconjg(yup(9))*
     -     (-(dconjg(yup(35))*inpup(idq0,idb1)) +inpup(idq0,idk2) 
     -     *yup(13))*(dconjg(yup(42)) + yup(36))))/xup(2)
	  bfun(3,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(1))*
     -     dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)*(yup(27)
     -     + yup(28)) -dotup(idb1,idb1)*inpup(idk2,idk1)*yup(19)*
     -     yup(27)*yup(31) - dconjg(yup(16))*(dconjg(yup(5))*
     -     dotup(idb1,idb1)*inpup(idq0,idc1)*yup(31) +dconjg(yup(35))*
     -     (dconjg(yup(42))*inpup(idk1,idb1)*yup(31) + dconjg(yup(1))
     -     *inpup(idq0,idb1)*(dconjg(yup(42))+yup(36))))))/xup(2)
	  bfun(4,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*inpup(idk2,idk1)*yup(4)*yup(19)) + 
     -     dconjg(yup(9))*dotup(idb1,idb1)*inpup(idq0,idk2)*yup(19)
     -     *(yup(27)+yup(28))-dconjg(yup(16))*(inpup(idq0,idk1)*yup(4)*
     -     (dconjg(yup(4))*dotup(idb1,idb1) -dconjg(yup(24))*yup(25)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*inpup(idq0,idb1)*
     -     (dconjg(yup(42)) + yup(36)))))/xup(2)
 	bfun(5,ii,kk)=(2*dconjg(yup(10))*inpup(idq0,idk1)*pmomup(5,idc1)
     -     *(yup(27) + yup(28))*yup(30))/xup(2)
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(9))*
     -     (dconjg(yup(10))*dconjg(yup(24)) -dconjg(yup(14))* 
     -     dotup(idb1,idb1))*inpup(idq0,idk2)*yup(14) +inpup(idq0,idk1)*
     -     (dotup(idb1,idb1)*(dconjg(yup(4))*(dconjg(yup(12)) -
     -     dconjg(yup(18)))*yup(4) + dconjg(yup(12))*dconjg(yup(14))
     -     *yup(14)) - dconjg(yup(24))*(dconjg(yup(10))*dconjg(yup(12))
     -     *yup(14)+(dconjg(yup(12))-dconjg(yup(18)))*yup(4)*yup(25)))+ 
     -     dconjg(yup(35))*(-2*dotup(idc1,idk1)*inpup(idk2,idk1)*yup(27)
     -     * yup(35) +dconjg(yup(9))*(dconjg(yup(12))-dconjg(yup(18)))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36)))))/xup(2)
	  bfun(9,ii,kk)=(-4*dconjg(yup(35))*inpup(idk2,idk1)
     -     *pmomup(5,idc1)*yup(28)*yup(35))/xup(2)
c...helicity(52,2)=(-,-,+,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(37,2)=(+,+,-,-,+,+)=helicity(37,1)(k2,k1)
        kk=37
        bfun(1,ii,kk)=(4*(yup(1)*(-(dconjg(inpup(idc1,idb2))*
     -     dconjg(yup(35))*inpup(idc2,idb1)*yup(10))+dotup(idb1,idb1)*
     -     (dconjg(yup(34))*yup(10) + dconjg(inpup(idc1,idk2))*
     -     inpup(idc2,idq0)*yup(19)))*yup(34) +dotup(idc1,idc1)*yup(10)*
     -     (yup(2)*yup(29)*yup(31)+dconjg(yup(35))*yup(1)*yup(35))))
     -     /xup(1)
        bfun(2,ii,kk)=(4*yup(9)*(-(dotup(idb1,idb1)*(-(dconjg(yup(4))*
     -     (yup(4)*yup(10) -dconjg(inpup(idk2,idk1))*
     -     inpup(idc2,idq0)*yup(19))) +dconjg(yup(34))*yup(10)*yup(34) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19)*yup(34))) + 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idb2,idk1))*
     -     dconjg(yup(4))*inpup(idc2,idb1) + dconjg(inpup(idc1,idb2))
     -     *inpup(idc2,idb1)*yup(34)-dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(3,ii,kk)=(4*yup(16)*(-((dconjg(yup(34))*dotup(idb1,idb1) + 
     -     (dconjg(inpup(idc1,idk2))*dconjg(yup(17)) - 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35)))*inpup(idc2,idb1))
     -     *yup(1)*yup(34))+dotup(idc1,idc1)*((-yup(2)+yup(7))*yup(29)
     -     *yup(31) +yup(1)*(dconjg(yup(17))*yup(17) - dconjg(yup(35))
     -     *yup(35)))))/xup(1)
	  bfun(4,ii,kk)=(-4*yup(9)*yup(16)*(dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) - dconjg(yup(34))*yup(34)) + 
     -     dconjg(yup(17))*(-(dconjg(inpup(idk2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1)) +dotup(idc1,idc1)*yup(17) - 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idb1)*yup(34)) + 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1) +dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(1)
 	  bfun(5,ii,kk)=(2*yup(9)*yup(16)*yup(28)*yup(29))/xup(1)
	  bfun(8,ii,kk)=(4*yup(9)*(yup(12) - yup(18))*
     -     (dconjg(yup(14))*yup(16)*yup(23) -dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) + dconjg(yup(14))*yup(14) - 
     -     dconjg(yup(34))*yup(34)) +dconjg(yup(35))*
     -     (-(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1))- 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) + dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -     dconjg(yup(42))*inpup(idq0,idb1)*yup(28))/xup(1)
c...helicity(53,2)=(-,-,+,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(38,2)=(+,+,-,-,+,-)=helicity(39,1)(k2,k1)
        kk=38
        bfun(1,ii,kk)=(4*dconjg(yup(1))* (-(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*yup(5)*yup(10) -dconjg(inpup(idc1,idk2))*
     -     inpup(idq0,idk1)*yup(5)*yup(19) -(dconjg(yup(34))*yup(10)+ 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19))*yup(34)))+ 
     -     dconjg(yup(35))*yup(10)*(dconjg(inpup(idc1,idb2))*
     -     (inpup(idk1,idb1)*yup(5) - inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))))/yup(8)
        bfun(2,ii,kk)=(-4*(dotup(idc1,idc1)*(dconjg(yup(8))*
     -     dotup(idb1,idb1)*yup(19)+yup(10)*yup(25)*yup(30))*yup(31)+ 
     -     dconjg(yup(9))*dotup(idb1,idb1)*(dconjg(yup(34))*yup(10) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idq0)*yup(19))*yup(34)) + 
     -     4*dconjg(yup(9))*dconjg(yup(35))*yup(10)*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35)))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*yup(16)*(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*yup(5) - dconjg(yup(34))*yup(34)) + 
     -     dconjg(yup(17))*(dotup(idc1,idc1)*yup(17) + 
     -     dconjg(inpup(idc1,idk2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))) - dconjg(yup(35))*
     -     (dconjg(inpup(idc1,idb2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*yup(35))))/yup(8)
	  bfun(4,ii,kk)=(4*yup(16)*(dconjg(yup(9))*(dconjg(yup(34))
     -     *dotup(idb1,idb1)+(dconjg(inpup(idc1,idk2))*dconjg(yup(17))- 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35)))*inpup(idc2,idb1))
     -     *yup(34)+dotup(idc1,idc1)*(yup(25)*(-yup(19)+yup(30))*yup(31) 
     -     +dconjg(yup(9))*(-(dconjg(yup(17))*yup(17)) + 
     -     dconjg(yup(35))*yup(35)))))/yup(8)
 	  bfun(5,ii,kk)=(2*dconjg(yup(1))*yup(16)*yup(27)*yup(29))/yup(8)
	  bfun(8,ii,kk)=(4*(2*dconjg(yup(4))*dotup(idc1,idk1)*
     -     (dotup(idb1,idb1)*yup(14) - yup(16)*yup(23))*yup(27) +
     -     dconjg(yup(9))*(-(dconjg(yup(14))*yup(16)*yup(18)*yup(23)) + 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)*
     -     (-yup(12) + yup(18))*yup(34) + dotup(idb1,idb1)*
     -     (dconjg(yup(14))*yup(14)*yup(18) +dconjg(yup(34))*(yup(12) 
     -     - yup(18))*yup(34))) + dotup(idc1,idc1)*(2*dconjg(yup(8))
     -     *dconjg(yup(35))*dotup(idc1,idk1)*yup(35)+(yup(12)-yup(18))*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
	  bfun(9,ii,kk)=(4*(dconjg(yup(5))*dotup(idb1,idb1)*yup(15)*
     -     yup(28) + dconjg(yup(35))*(-(dconjg(inpup(idc1,idb2))*
     -     inpup(idk1,idb1)*yup(15)*yup(28)) + 
     -     dconjg(yup(8))*dotup(idc1,idc1)*yup(35))))/yup(8)
c...helicity(54,2)=(-,-,+,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(39,2)=(+,+,-,-,-,+)=helicity(38,1)(k2,k1)
        kk=39
        bfun(1,ii,kk)=(4*dconjg(yup(10))*(yup(1)*(dconjg(yup(34))
     -     *dotup(idb1,idb1) +dconjg(inpup(idc1,idb2))*
     -     (-(dconjg(yup(35))*inpup(idc2,idb1)) + inpup(idc2,idk2)
     -     *yup(13)))*yup(34) +dotup(idc1,idc1)*(yup(2)*(-yup(19)
     -     + yup(29))*yup(31) +yup(1)*(-(dconjg(yup(13))*yup(13)) + 
     -     dconjg(yup(35))*yup(35)))))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(yup(10))*yup(9)*(-((dconjg(yup(13))
     -     *dotup(idc1,idc1)-dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idk2))*yup(13)) +dconjg(inpup(idc1,idb2))* 
     -     inpup(idc2,idk2)*yup(13)*yup(34) + dotup(idb1,idb1)*
     -     (-(dconjg(yup(4))*yup(4)) + dconjg(yup(34))*yup(34)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*
     -     inpup(idc2,idb1) +dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(-4*dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)
     -     *yup(19)*yup(31)+yup(1)*(dconjg(yup(16))*dconjg(yup(34)) + 
     -     dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*yup(19))*yup(34))+ 
     -     4*dconjg(yup(16))*(-(dotup(idc1,idc1)*yup(2)*yup(29)
     -     *yup(31)) +dconjg(yup(35))*yup(1)*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*yup(9)*(dotup(idb1,idb1)*(-(dconjg(yup(4))*
     -     (dconjg(yup(16))*yup(4) -dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(19)))+dconjg(yup(16))*dconjg(yup(34))
     -     *yup(34) +dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*
     -     yup(19)*yup(34)) -dconjg(yup(16))*dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=(2*dconjg(yup(10))*yup(9)*yup(28)*yup(30))
     &	 /dconjg(yup(8))
	  bfun(8,ii,kk)=(-4*yup(9)*((dconjg(yup(12)) - dconjg(yup(18)))*
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(-((dconjg(inpup(idb2,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18))) - 
     -     2*dconjg(inpup(idb2,idq0))*dconjg(yup(14))*dotup(idc1,idk1))*
     -     inpup(idc2,idb1)) +dconjg(inpup(idc1,idb2))*(dconjg(yup(12))
     -     -dconjg(yup(18)))*inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*
     -     (dconjg(yup(10))*yup(19)*yup(30) + (-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dconjg(yup(35))*yup(35))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idb2))*dconjg(yup(11))*
     -     dconjg(yup(35))*inpup(idq0,idb1)*yup(5))/dconjg(yup(8))
c...helicity(55,2)=(-,-,+,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(40,2)=(+,+,-,-,-,-)=helicity(40,1)(k2,k1)
        kk=40
        bfun(1,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(10))*
     -     (-(dconjg(yup(13))*dotup(idc1,idc1)*yup(13)) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idk2,idk1)*yup(5)*yup(13) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idk2)*yup(13)*yup(34) - 
     -     dotup(idb1,idb1)*(dconjg(yup(5))*yup(5) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(dconjg(inpup(idc1,idb2))*
     -     (inpup(idk1,idb1)*yup(5) -inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(yup(10))*(-(dconjg(yup(9))*
     -     (dconjg(yup(34))*dotup(idb1,idb1)+dconjg(inpup(idc1,idb2))*
     -     (-(dconjg(yup(35))*inpup(idc2,idb1)) + inpup(idc2,idk2)
     -     *yup(13)))*yup(34)) - dotup(idc1,idc1)*((dconjg(yup(3))*
     -     dconjg(yup(12)) -dconjg(yup(9))*dconjg(yup(13)))*yup(13) + 
     -     yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35))))/xup(2)
	  bfun(3,ii,kk)=(4*dconjg(yup(1))*(dotup(idb1,idb1)*
     -     (-(dconjg(inpup(idc1,idq0))*inpup(idk2,idk1)*yup(5)*yup(19))- 
     -     dconjg(inpup(idc1,idq0))*inpup(idc2,idk2)*yup(19)*yup(34) + 
     -     dconjg(yup(16))*(dconjg(yup(5))*yup(5) - dconjg(yup(34))*
     -     yup(34))) - dconjg(yup(16))*dconjg(yup(35))*
     -     (dconjg(inpup(idc1,idb2))*(inpup(idk1,idb1)*yup(5) - 
     -     inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*yup(35))))/xup(2)
	  bfun(4,ii,kk)=(4*(dconjg(yup(16))*dotup(idc1,idc1)*yup(25)*
     -     yup(30)*yup(31) + dconjg(yup(9))*(dotup(idb1,idb1)*
     -     (dconjg(yup(16))*dconjg(yup(34)) + dconjg(inpup(idc1,idq0))
     -     *inpup(idc2,idk2)*yup(19))*yup(34) - dconjg(yup(16))*
     -     dconjg(yup(35))*(dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35)))))/xup(2)
 	bfun(5,ii,kk)=(2*dconjg(yup(1))*dconjg(yup(10))*yup(27)*yup(30))
     &	  /xup(2)
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(30)*(dconjg(yup(9))*
     -     dconjg(yup(10))*yup(19) + (dconjg(yup(7))*dconjg(yup(10)) + 
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*yup(25))*yup(31)) + 
     -     4*dconjg(yup(9))*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     dconjg(yup(34))*dotup(idb1,idb1)*yup(34)+4*dconjg(yup(35))*
     -     (-2*dconjg(inpup(idb2,idq0))*dotup(idc1,idk1)*
     -     inpup(idk2,idk1)*yup(22)*yup(27) + dconjg(yup(9))*
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(56,2)=(-,-,+,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(41,2)=(+,-,+,+,+,+)=helicity(41,1)(k2,k1)
        kk=41
        bfun(1,ii,kk)=(-4*pmomup(5,idb1)*yup(1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (-(dconjg(inpup(idk2,idk1))*dconjg(yup(5))*inpup(idc1,idb2))- 
     -     dconjg(inpup(idc2,idk2))*dconjg(yup(34))*inpup(idc1,idb2) + 
     -     dotup(idc1,idc1)*yup(13))) +dconjg(inpup(idq0,idk1))* 
     -     dconjg(yup(5))*yup(10)*(yup(20) + yup(39)) - 
     -     dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     yup(10)*(yup(20) + yup(39))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*(dotup(idc1,idc1)*
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3))-dconjg(inpup(idb1,idq0))
     -     *yup(6))*yup(10)*yup(31) + dconjg(yup(13))*yup(13)*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9))+dconjg(inpup(idk2,idk1))
     -     *yup(31)))+dconjg(yup(34))*yup(9)*(dconjg(inpup(idc2,idk2))*
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(13))*inpup(idc1,idb2) - 
     -     dconjg(inpup(idq0,idc2))*yup(10)*(yup(20)+yup(39)))))/xup(1)
	bfun(3,ii,kk)=(4*pmomup(5,idb1)*yup(1)*(dconjg(inpup(idb1,idq0))
     -     *dotup(idc1,idc1)*yup(17)*yup(19) +dconjg(yup(34))*
     -     (dconjg(inpup(idc2,idk2))*yup(19)*yup(20) - 
     -     dconjg(inpup(idq0,idc2))*yup(16)*(yup(20) + yup(39))) + 
     -     dconjg(yup(5))*(dconjg(inpup(idk2,idk1))*yup(19)*yup(20) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*(yup(20)+yup(39)))))/xup(1)
	  bfun(4,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*yup(9)*yup(17)*yup(19) + 
     -     ((-(dconjg(inpup(idb2,idq0))*yup(3))-dconjg(inpup(idb1,idq0))
     -     *yup(6))*yup(16) +dconjg(inpup(idk2,idk1))*yup(19)*yup(29)
     -     )*yup(31)))-dconjg(yup(34))*yup(9)*(dconjg(inpup(idc2,idk2))
     -     *yup(19)*yup(20) - dconjg(inpup(idq0,idc2))*yup(16)*
     -     (yup(20) + yup(39)))))/xup(1)
 	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)
     -     *yup(1)*yup(27)*(yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(2*dconjg(inpup(idk2,idk1))
     -     *dotup(idc1,idk1)*yup(27)*yup(28)*(yup(29) + yup(30)) + 
     -     dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*(yup(9)*yup(13)
     -     *yup(19)+(yup(7)*yup(13)+yup(3)*(-yup(12)+yup(18)))*yup(31))
     -     ) - dconjg(inpup(idb1,idq0))*(yup(9)*yup(17)*yup(19) + 
     -     (yup(7)*yup(17) +yup(6)*(-yup(12) + yup(18)))*yup(31)))
     -     + dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     yup(9)*(yup(12) - yup(18))*(yup(20) + yup(39))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(57,2)=(-,+,-,-,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(42,2)=(+,-,+,+,+,-)=helicity(43,1)(k2,k1)
        kk=42
        bfun(1,ii,kk)=(4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(19)*yup(31) + 
     -     dconjg(yup(1))*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (dconjg(inpup(idc2,idk2))*dconjg(yup(34))*inpup(idc1,idb2) - 
     -     dotup(idc1,idc1)*yup(13))) + dconjg(inpup(idq0,idc2))
     -     *dconjg(yup(34))*yup(10)*(yup(20) + yup(39)))))/yup(8)
        bfun(2,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dotup(idc1,idc1)*yup(13)) - dconjg(inpup(idc2,idk2))*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     dconjg(yup(34))*inpup(idc1,idb2)) + dconjg(yup(2))*yup(4)
     -     *yup(19)) -dconjg(inpup(idq0,idc2))*yup(10)*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/yup(8)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*yup(19)*
     -     (-(dconjg(inpup(idb1,idq0))*dconjg(yup(1))*yup(17)) + 
     -     dconjg(inpup(idq0,idk2))*yup(25)*yup(31)))+dconjg(yup(1))
     -     *dconjg(yup(34))*(dconjg(inpup(idc2,idk2))*yup(19)*yup(20) - 
     -     dconjg(inpup(idq0,idc2))*yup(16)*
     -     (yup(20) + yup(39)))))/yup(8)
	  bfun(4,ii,kk)=(-4*dconjg(yup(9))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*yup(19) + 
     -     dconjg(inpup(idc2,idk2))*yup(19)*(dconjg(yup(34))*yup(20)
     -     - yup(4)*yup(25)) -dconjg(inpup(idq0,idc2))*yup(16)*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/yup(8)
 	  bfun(5,ii,kk)=(-2*dconjg(inpup(idq0,idk2))*dconjg(yup(9))*
     -     pmomup(5,idb1)*yup(28)*(yup(29) + yup(30)))/yup(8)
	bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*(dotup(idc1,idc1)
     -     *(-(dconjg(inpup(idb2,idq0))*yup(13))-dconjg(inpup(idb1,idq0) 
     -     )*yup(17))*yup(19) - dconjg(inpup(idq0,idc2))*
     -     (2*dotup(idc1,idk1)*yup(14)*(yup(29)+yup(30))+(yup(12)-
     -     yup(18))*(yup(4)*(dconjg(yup(2)) + yup(25)) - 
     -     dconjg(yup(34))*(yup(20) + yup(39))))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(58,2)=(-,+,-,-,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(43,2)=(+,-,+,+,-,+)=helicity(42,1)(k2,k1)
        kk=43
        bfun(1,ii,kk)=(-4*dconjg(yup(10))*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(5)) -dconjg(inpup(idq0,idc2))*dconjg(yup(34)))*
     -     pmomup(5,idb1)*yup(1)*(yup(20) + yup(39) - yup(40)))
     -     /dconjg(yup(8))
        bfun(2,ii,kk)=(4*dconjg(yup(10))*pmomup(5,idb1)*
     -     (dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6)-dconjg(inpup(idq0,idk1))
     -     *yup(19))*yup(31)-dconjg(inpup(idq0,idc2))*dconjg(yup(34))
     -     *yup(9)*(yup(20)+yup(39)-yup(40))))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*yup(1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(18))*(-yup(28)+yup(31))+ 
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(5)) - 
     -     dconjg(inpup(idq0,idc2))*dconjg(yup(34)))*
     -     (yup(20) + yup(39))))/dconjg(yup(8))
	  bfun(4,ii,kk)=(4*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dconjg(inpup(idb2,idq0))*dotup(idc1,idc1)*yup(3)*
     -     yup(31) + dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)
     -     *yup(6)*yup(31) +dconjg(inpup(idq0,idk2))*
     -     (dconjg(yup(18))*yup(9)*yup(28)+dotup(idc1,idc1)*yup(8) 
     -     *yup(31))+dconjg(inpup(idq0,idc2))*dconjg(yup(34))*yup(9)*
     -     (yup(20) + yup(39))))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*((-dconjg(yup(12)) 
     -     + dconjg(yup(18)))*dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*yup(3)) - 
     -     dconjg(inpup(idb1,idq0))*yup(6))*yup(31) - 
     -     dconjg(inpup(idq0,idc2))*(-((dconjg(yup(10)) +
     -     dconjg(yup(16)))*(dconjg(yup(18))*yup(9)*yup(14) + 
     -     2*dotup(idc1,idk1)*yup(4)*yup(27))) +(-dconjg(yup(12)) 
     -     + dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(yup(20) 
     -     + yup(39)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(15))*
     -     pmomup(5,idb1)*yup(28)*(yup(20)+yup(39)))/dconjg(yup(8))
c...helicity(59,2)=(-,+,-,-,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(44,2)=(+,-,+,+,-,-)=helicity(44,1)(k2,k1)
        kk=44
        bfun(1,ii,kk)=(4*dconjg(inpup(idq0,idc2))*dconjg(yup(1))*
     -     dconjg(yup(10))*dconjg(yup(34))*pmomup(5,idb1)*
     -     (yup(20) + yup(39) - yup(40)))/xup(2)
        bfun(2,ii,kk)=(4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -     dconjg(yup(10))*pmomup(5,idb1)*(yup(4)*(dconjg(yup(2))
     -     - dconjg(yup(7)) +yup(25)) - dconjg(yup(34))*(yup(20) + 
     -     yup(39) - yup(40))))/xup(2)
	bfun(3,ii,kk)=(4*dconjg(yup(1))*dconjg(yup(16))*pmomup(5,idb1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(18))*
     -     yup(28)) -  dconjg(inpup(idq0,idc2))*dconjg(yup(34))*
     -     (yup(20) + yup(39))))/xup(2)
	bfun(4,ii,kk)=(-4*dconjg(yup(9))*dconjg(yup(16))*pmomup(5,idb1)*
     -     (dconjg(inpup(idq0,idc2))*dconjg(inpup(idq0,idk2))
     -     *inpup(idk2,idk1)*yup(4) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(18))*yup(28) -dconjg(inpup(idq0,idc2))*
     -     (-(yup(4)*(dconjg(yup(2)) + yup(25))) + 
     -     dconjg(yup(34))*(yup(20) + yup(39)))))/xup(2)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     ((dconjg(yup(10)) + dconjg(yup(16)))*yup(14) + 
     -     yup(4)*(dconjg(yup(2)) + yup(25)) - 
     -     dconjg(yup(34))*(yup(20) + yup(39))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(60,2)=(-,+,-,-,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   -dconjg(bfun(mm,ii,kk))
        end do

c...helicity(45,2)=(+,-,+,-,+,+)=helicity(45,1)(k2,k1)
        kk=45
        bfun(1,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk1))*yup(10)*yup(31)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(32)) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)) - 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(13))*
     -     (-(dconjg(inpup(idk2,idk1))*yup(31)*yup(42)) + 
     -     dconjg(inpup(idq0,idk2))*yup(1)*
     -     (dconjg(yup(36)) + yup(42)))))/xup(1)
        bfun(2,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     (dconjg(inpup(idk2,idk1))*yup(19)*yup(24) + 
     -     dconjg(inpup(idq0,idk1))*yup(10)*(yup(23)+yup(24))))- 
     -     dconjg(inpup(idq0,idk2))**2*dconjg(yup(13))*
     -     yup(9)*(dconjg(yup(36)) + yup(42))))/xup(1)
	  bfun(3,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*yup(1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23)) +yup(31)* 
     -     (dconjg(inpup(idc1,idq0))*dconjg(inpup(idk2,idk1))
     -     *yup(19)*yup(20) +dconjg(inpup(idq0,idk1))*yup(16)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(32)) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)))))/xup(1)
	  bfun(4,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*yup(9)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23)) - 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(4))*
     -     (dconjg(inpup(idk2,idk1))*yup(19)*yup(23) + 
     -     dconjg(inpup(idq0,idk1))*yup(16)*
     -     (yup(23) + yup(24)))))/xup(1)
 	  bfun(5,ii,kk)=(2*dconjg(inpup(idq0,idk1))*
     -     dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(27) + yup(28))*
     -     (yup(29) + yup(30)))/xup(1)
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(14))*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9)) + 
     -     dconjg(inpup(idq0,idk1))*yup(12)) + 
     -     dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (yup(12) - yup(18)))*(yup(23) + yup(24)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(61,2)=(-,+,-,+,-,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(46,2)=(+,-,+,-,+,-)=helicity(47,1)(k2,k1)
        kk=46
        bfun(1,ii,kk)=(4*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(2))*yup(19) + dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(13))*(dconjg(yup(36)) + yup(42))))/yup(8)
        bfun(2,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))*
     -     dconjg(yup(13))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(36)) - yup(38) + yup(42)))/yup(8)
	  bfun(3,ii,kk)=(4*dconjg(inpup(idq0,idk2))*dconjg(yup(1))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2)) 
     -     *yup(23) - dconjg(inpup(idq0,idk1))*yup(25)))/yup(8)
	  bfun(4,ii,kk)=(-4*dconjg(inpup(idq0,idk2))*dconjg(yup(9))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(19)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))-dconjg(inpup(idb1,idq0)) 
     -     *yup(21) +dconjg(inpup(idq0,idc2))*yup(23)))/yup(8)
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))*
     -     dconjg(yup(14))*pmomup(5,idb1)*pmomup(5,idc1)*(yup(23) 
     -     + yup(24)))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(62,2)=(-,+,-,+,-,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(47,2)=(+,-,+,-,-,+)=helicity(46)(k2,k1)
        kk=47
        bfun(1,ii,kk)=(-4*dconjg(inpup(idq0,idk1))*dconjg(yup(10))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(31)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(32)) - 
     -     dconjg(inpup(idc1,idq0))*yup(40) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)))/dconjg(yup(8))
        bfun(2,ii,kk)=(-4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     dconjg(yup(10))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (yup(23) + yup(24) - yup(33)))/dconjg(yup(8))
	  bfun(3,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*yup(31)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(18))) - 
     -     dconjg(inpup(idb1,idq0))*yup(32) - 
     -     dconjg(inpup(idb2,idq0))*yup(42)))/dconjg(yup(8))
	  bfun(4,ii,kk)=(-4*dconjg(inpup(idq0,idk1))*dconjg(yup(16))*
     -     pmomup(5,idb1)*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(yup(23)+yup(24)))-dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(11))*yup(31)))/dconjg(yup(8))
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=(4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(23) + yup(24)))/dconjg(yup(8))
	  bfun(9,ii,kk)=0.0d0
c...helicity(63,2)=(-,+,-,+,+,-)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

c...helicity(48,2)=(+,-,+,-,-,-)=helicity(48,1)(k2,k1)
        kk=48
        bfun(1,ii,kk)=0.0d0
        bfun(2,ii,kk)=0.0d0
	  bfun(3,ii,kk)=0.0d0
	  bfun(4,ii,kk)=0.0d0
 	  bfun(5,ii,kk)=0.0d0
	  bfun(8,ii,kk)=0.0d0
	  bfun(9,ii,kk)=0.0d0
c...helicity(64,2)=(-,+,-,+,+,+)
        do mm=1,9
	   if (mm.ne.6 .and. mm.ne.7) bfun(mm,ii,kk+16)=
     &	   dconjg(bfun(mm,ii,kk))
        end do

1222  continue	
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c       nine basic functions from first by b quark <-> c quark.      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	subroutine bthird
	implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex bfun,colmat,yup,xup,inpup,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/tryup/bfun(9,4,64),yup(100),xup(2),idp,idq0,idk1,idk2,
     &	idb1,idb2,idc1,idc2
      common/counter/ibcstate,nev
c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

	 ii=3
c...if (l1.ne.l3 .and. l2.ne.l4) then
c...hel(i,3)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,-l2,-l3,-l4,l5,l6)
c...(q_b1<->q_c1,q_b2<->q_c2), where l1,...,l6 are helicities of q_b1,q_b2,
c...q_c1,q_c2,k_1,k_2; (l2=l3);
c...if (l1.ne.l3 .and. l2.eq.l4) then
c...hel(i,3)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,l2,-l3,l4,l5,l6)
c...(q_b1<->q_c1,q_b2<->q_c2)
c...if (l1.eq.l3 .and. l2.ne.l4) then
c...hel(i,3)(l1,l2,l3,l4,l5,l6)=hel(i,1)(l1,-l2,l3,-l4,l5,l6)
c...(q_b1<->q_c1,q_b2<->q_c2)
c...if (l1.eq.l3 .and. l2.eq.l4) then
c...hel(i,3)(l1,l2,l3,l4,l5,l6)=hel(i,1)(l1,l2,l3,l4,l5,l6)
c...(q_b1<->q_c1,q_b2<->q_c2)

       if(iqqbar.eq.0) then
       do kk=1,64
	  bfun(1,ii,kk)=bfun(4,2,kk)
        bfun(2,ii,kk)=bfun(2,2,kk)
	  bfun(3,ii,kk)=bfun(3,2,kk)
	  bfun(4,ii,kk)=bfun(1,2,kk)
	  bfun(5,ii,kk)=bfun(5,2,kk)
	 end do
	 end if

c...helicity(1,3)=(+,+,+,+,+,+)=helicity(1,1)(q_c1,q_c2,q_b1,q_b2)
        kk=1
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(29)*(yup(9)*yup(16)*
     -     yup(19)+(yup(7)*yup(16)+yup(2)*(-yup(12) + yup(18)))*
     -     yup(31)) +4*dconjg(yup(34))*yup(9)*(yup(12) - yup(18))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34)) + 
     -     4*dconjg(yup(35))*(dotup(idc1,idc1)*yup(9)*(yup(12)-yup(18))*
     -     yup(35) + 2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     inpup(idq0,idb1)*yup(27)*yup(36)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(17,3)=(-,-,-,-,-,-)
	  do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=-4*(dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)
     -     *yup(30) +dconjg(yup(34))*(-(dconjg(inpup(idq0,idk1))*
     -     dconjg(yup(24))*inpup(idc1,idk2))+dconjg(inpup(idc2,idk1))
     -     *inpup(idb1,idk2)*yup(43)+ yup(5)*(yup(44) -yup(45)))) + 
     -     (-(dconjg(yup(24))*dconjg(yup(34))*yup(20))+dconjg(yup(35))
     -     *dotup(idc1,idc1)*yup(35))*yup(46) + dconjg(yup(35))*
     -     (dconjg(yup(13))*dotup(idc1,idc1)- dconjg(yup(18))*yup(36))
     -      *yup(47))
        bfun(2,ii,kk)=4*(dconjg(yup(35))*(-(dconjg(inpup(idc2,idk1))*
     -     dconjg(yup(34))*inpup(idc1,idb1))+dotup(idc1,idc1)*yup(6))
     -     *yup(48) + dotup(idb1,idb1)*(dconjg(yup(18))*yup(7)*
     -     yup(28) +dotup(idc1,idc1)*yup(8)*yup(29)-dconjg(yup(34))*
     -     (dconjg(inpup(idq0,idc2))*inpup(idc1,idk2)*yup(2) + 
     -     dconjg(inpup(idc2,idk1))*inpup(idq0,idk2)*yup(20) - 
     -     dconjg(yup(14))*yup(49)))+(-(dconjg(yup(24))*dconjg(yup(34))
     -     *yup(20))+dconjg(yup(35))*dotup(idc1,idc1)*yup(35))*yup(50))
	  bfun(4,ii,kk)=4*(dotup(idb1,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(inpup(idq0,idk1))**2*inpup(idq0,idk1)*inpup(idq0,idk2) 
     -    - dconjg(yup(13))*yup(3) -dconjg(yup(17))*yup(6))) + 
     -    dconjg(yup(34))*inpup(idc1,idk2)*(-(dconjg(inpup(idq0,idk1))*
     -    (dconjg(yup(23))+dconjg(yup(24)))) + dconjg(inpup(idc2,idk1))
     -    *yup(31))) +dconjg(yup(35))*(dotup(idc1,idc1)*
     -    (dconjg(inpup(idb2,idk1))*inpup(idq0,idk2)*yup(21) + 
     -    dconjg(yup(13))*yup(82)) +dconjg(yup(34))*inpup(idc1,idk2)*
     -    (-(dconjg(inpup(idc2,idk1))*yup(51))+dconjg(inpup(idb2,idk1))
     -    *yup(83))))	  
c...helicity(17,3)=(-,-,-,-,-,-)
	  do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(2,3)=m(+,+,+,+,+,-)=helicity(2,1)(q_c1,q_c2,q_b1,q_b2)
        kk=2
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*((-(dconjg(yup(10))*dconjg(yup(24))) + 
     -     dconjg(yup(14))*dotup(idb1,idb1))*(dconjg(yup(18))*yup(9)
     -     *yup(14)+2*dotup(idc1,idk1)*yup(4)*yup(27))+(dconjg(yup(12)) 
     -     - dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (2*dconjg(yup(35))*dotup(idc1,idk1)*yup(8)*yup(35) + 
     -     (dconjg(yup(12))- dconjg(yup(18)))*(yup(2)*yup(29)*yup(31)+ 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(yup(15))*(dotup(idb1,idb1)*yup(5)
     -     -yup(2)*yup(20))*yup(28)+4*dconjg(yup(35))*dotup(idc1,idc1)*
     -     yup(8)*yup(35))/dconjg(yup(8))
c...helicity(18)=m(-,-,-,-,-,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(3,3)=(+,+,+,+,-,+)=helicity(3,1)(q_c1,q_c2,q_b1,q_b2)
        kk=3
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(24))*((yup(12)
     -     - yup(18))*(-(dconjg(yup(34))*yup(20)) +yup(4)*yup(25))+ 
     -     2*dotup(idc1,idk1)*yup(14)*yup(29))+dotup(idb1,idb1)*(yup(12) 
     -     -yup(18))*(-(dconjg(yup(4))*yup(4))+dconjg(yup(34))*yup(34))+ 
     -     dotup(idc1,idc1)*(-(yup(16)*yup(19)*yup(29)) + 
     -     dconjg(yup(35))*(yup(12) - yup(18))*yup(35))))/yup(8)
	  bfun(9,ii,kk)=(4*dconjg(yup(5))*yup(11)*yup(20)*yup(30))/yup(8)
c...helicity(19,3)=(-,-,-,-,+,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
	  end do
	  end if

c...helicity(4,3)=(+,+,+,+,-,-)=helicity(4,1)(q_c1,q_c2,q_b1,q_b2)
        kk=4
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(12))
     -     - dconjg(yup(18)))*(dconjg(yup(24))*(dconjg(yup(10))
     -     *yup(14) -dconjg(yup(34))*yup(20) + yup(4)*yup(25)) - 
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) + dconjg(yup(14))
     -     *yup(14) -dconjg(yup(34))*yup(34)) + dconjg(yup(35))
     -     *dotup(idc1,idc1)*yup(35)))/xup(2)
        bfun(9,ii,kk)=(-4*dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     inpup(idk1,idk2)*yup(28)*yup(32))/xup(2)
c...helicity(20,3)=(-,-,-,-,+,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*(-(dconjg(yup(24))*dconjg(yup(34))*(yup(20)*
     -     yup(53) - yup(25)*yup(54))) + dotup(idb1,idb1)*
     -     (dconjg(yup(8))*dotup(idc1,idc1)*yup(30) + dconjg(yup(34))*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(24))*inpup(idc1,idk1))
     -     +dconjg(inpup(idc2,idk2))*inpup(idb1,idk1)*yup(43) + yup(15)
     -     *(yup(55) - yup(56)))) +dconjg(yup(35))*dotup(idc1,idc1)*
     -     (yup(35)*yup(53) + dconjg(yup(3))*yup(57)))
        bfun(2,ii,kk)=-4*(dotup(idb1,idb1)*(-(dconjg(yup(8))*
     -     dotup(idc1,idc1)*yup(29)) +dconjg(yup(34))*
     -     (dconjg(inpup(idq0,idc2))*inpup(idc1,idk1)*yup(16) + 
     -     dconjg(inpup(idc2,idk2))*inpup(idq0,idk1)*yup(20) + 
     -     dconjg(yup(4))*(yup(54) - yup(58)))) + dconjg(yup(35))*
     -     (-(dotup(idc1,idc1)*yup(17)) + yup(11)*yup(32))*
     -     yup(59) + (dconjg(yup(24))*dconjg(yup(34))*yup(20) - 
     -     dconjg(yup(35))*dotup(idc1,idc1)*yup(35))*yup(60))
	  bfun(4,ii,kk)=-4*(dconjg(inpup(idq0,idk1))*
     -    dconjg(inpup(idq0,idk2))*dotup(idb1,idb1)*dotup(idc1,idc1)*
     -    inpup(idq0,idk1)**2 +dotup(idb1,idb1)*(-(dotup(idc1,idc1)*
     -    (dconjg(yup(3))*yup(13) +dconjg(yup(6))*yup(17))) + 
     -    dconjg(inpup(idc2,idk2))*dconjg(yup(34))*(dconjg(yup(3))*
     -    inpup(idc1,idb2) -inpup(idc1,idk1)*yup(31))+ dconjg(yup(6))
     -    *yup(11)*yup(32))+dconjg(yup(35))*(-(dconjg(inpup(idq0,idk2))
     -    *dconjg(yup(38))*dotup(idc1,idc1)*inpup(idb1,idk1)) + 
     -    dconjg(inpup(idq0,idk2))*dotup(idc1,idc1)*inpup(idb1,idk1)
     -    *yup(61) - dconjg(inpup(idc2,idk2))*dconjg(yup(34))*
     -    (-(inpup(idc1,idk1)*yup(51)) + inpup(idb1,idk1)*yup(62))))
c...helicity(20,3)=(-,-,-,-,+,+)
        do mm=1,4
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(5,3)=(+,+,+,-,+,+)=helicity(41,1)(q_c1,q_c2,q_b1,q_b2)
        kk=5
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(2*dconjg(inpup(idk2,idk1))
     -     *dotup(idc1,idk1)*yup(27)*yup(28)*(yup(29) + yup(30)) + 
     -     dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*(yup(9)*yup(13)
     -     *yup(19)+(yup(7)*yup(13)+yup(3)*(-yup(12)+yup(18)))*yup(31))
     -     ) - dconjg(inpup(idb1,idq0))*(yup(9)*yup(17)*yup(19) + 
     -     (yup(7)*yup(17) +yup(6)*(-yup(12) + yup(18)))*yup(31)))
     -     + dconjg(inpup(idq0,idc2))*dconjg(yup(34))*yup(9)*(yup(12)
     -     - yup(18))*(yup(20) + yup(39))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(21,3)=(-,-,-,+,-,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30) + 
     -     dconjg(inpup(idq0,idk1))*(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk2)) 
     -     + dconjg(yup(10))*yup(30)) +dconjg(yup(34))*
     -     (dconjg(inpup(idq0,idc2))*dotup(idb1,idb1)*inpup(idc1,idk2) +
     -     yup(39)*yup(44))) - dconjg(yup(34))*(dconjg(inpup(idq0,idc2))
     -     *dconjg(yup(10))*dconjg(yup(72)) +dconjg(inpup(idc2,idk1))*
     -     (dconjg(yup(10)) + dconjg(yup(12)))*yup(39) - 
     -     dconjg(inpup(idq0,idc2))*yup(20)*yup(46)))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*dconjg(yup(16))*yup(6) + 
     -     dconjg(inpup(idq0,idk1))*(dconjg(yup(16)) - 
     -     dconjg(inpup(idq0,idk1))*inpup(idk2,idk1))*yup(29))) + 
     -     dconjg(inpup(idb2,idq0))*dconjg(yup(18))*yup(28)*yup(66) +
     -     dconjg(yup(34))*(dconjg(inpup(idq0,idc2))*
     -     dconjg(inpup(idq0,idk1))*dotup(idb1,idb1)*inpup(idc1,idk2)+ 
     -     dconjg(inpup(idq0,idc2))*yup(39)*yup(50) + yup(20)*
     -     (-(dconjg(inpup(idc2,idk1))*(dconjg(yup(12)) + 
     -     dconjg(yup(16))))+dconjg(inpup(idq0,idk1))*yup(68))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))*
     -    dotup(idc1,idc1)*(dconjg(yup(13))*(dconjg(yup(61)) - 
     -    dconjg(inpup(idq0,idk1))*inpup(idk1,idb2))+dconjg(yup(17))*
     -    (dconjg(inpup(idq0,idk1))*inpup(idb1,idk1) +yup(75))) + 
     -    dconjg(yup(34))*inpup(idc1,idk2)*(-(dconjg(inpup(idc2,idk1))*
     -    (-(dconjg(inpup(idb1,idq0))*yup(21))-dconjg(inpup(idb2,idq0))*
     -    yup(38))) -dconjg(inpup(idb1,idq0))*dconjg(inpup(idq0,idk1))
     -    *yup(83) -dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))
     -    *yup(85)))
c...helicity(21,3)=(-,-,-,+,-,-)
        do mm=1,4
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(6,3)=(+,+,+,-,+,-)=helicity(42,1)(q_c1,q_c2,q_b1,q_b2)
        kk=6
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*((-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*yup(3))-dconjg(inpup(idb1,idq0))
     -     *yup(6))*yup(31) -dconjg(inpup(idq0,idc2))*
     -     (-((dconjg(yup(10)) + dconjg(yup(16)))*(dconjg(yup(18))
     -     *yup(9)*yup(14) +2*dotup(idc1,idk1)*yup(4)*yup(27))) + 
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*dconjg(yup(34))
     -     *yup(9)*(yup(20) + yup(39)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(15))*
     -    pmomup(5,idb1)*yup(28)*(yup(20) + yup(39)))/dconjg(yup(8))
c...helicity(22,3)=(-,-,-,+,-,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(7,3)=(+,+,+,-,-,+)=helicity(43,1)(q_c1,q_c2,q_b1,q_b2)
        kk=7
	if(iqqbar.eq.0) then
	bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*(dotup(idc1,idc1)
     -     *(-(dconjg(inpup(idb2,idq0))*yup(13)) - 
     -     dconjg(inpup(idb1,idq0))*yup(17))*yup(19) - 
     -     dconjg(inpup(idq0,idc2))*(2*dotup(idc1,idk1)*yup(14)*
     -     (yup(29)+yup(30))+(yup(12)- yup(18))*(yup(4)*(dconjg(yup(2))
     -     + yup(25)) -dconjg(yup(34))*(yup(20) + yup(39))))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(23,3)=(-,-,-,+,+,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(8,3)=(+,+,+,-,-,-)=helicity(44,1)(q_c1,q_c2,q_b1,q_b2)
        kk=8
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     ((dconjg(yup(10)) + dconjg(yup(16)))*yup(14) + 
     -     yup(4)*(dconjg(yup(2)) + yup(25)) -dconjg(yup(34)) 
     -     *(yup(20) + yup(39))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(24,3)=(-,-,-,+,+,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
        bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30))-dconjg(yup(34))*
     -     (dconjg(inpup(idc2,idk2))*(dconjg(yup(7))+yup(25))*yup(39)- 
     -     dconjg(inpup(idq0,idc2))*yup(20)*yup(53) + 
     -     dconjg(inpup(idq0,idc2))*yup(25)*yup(54))+ 
     -     dconjg(inpup(idq0,idk2))*(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk1))
     -     + yup(25)*yup(30)) +dconjg(yup(34))*
     -     (dconjg(inpup(idq0,idc2))*dotup(idb1,idb1)*inpup(idc1,idk1)
     -     + yup(39)*yup(55))))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(inpup(idb1,idq0))*dconjg(yup(2))*yup(17) + 
     -     dconjg(inpup(idq0,idk2))*(dconjg(yup(2)) + 
     -     dconjg(inpup(idq0,idk2))*inpup(idk2,idk1))*yup(29))) + 
     -     dconjg(yup(34))*(dconjg(inpup(idq0,idc2))*
     -     dconjg(inpup(idq0,idk2))*dotup(idb1,idb1)*inpup(idc1,idk1)- 
     -     dconjg(inpup(idq0,idc2))*(dconjg(yup(2))*yup(54) -
     -     yup(39)*yup(60)) +yup(20)*(-(dconjg(inpup(idc2,idk2))*
     -     (dconjg(yup(2))+dconjg(yup(7))))+dconjg(inpup(idq0,idk2))
     -     *yup(73))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(-(dotup(idc1,idc1)*
     -     (dconjg(yup(2)) + yup(25))*(dconjg(inpup(idb2,idq0))
     -     *yup(13) +dconjg(inpup(idb1,idq0))*yup(17) + 
     -     dconjg(inpup(idq0,idk2))*yup(31)))-dconjg(inpup(idc2,idk2)) 
     -     *dconjg(yup(34))*(inpup(idc1,idk1)*
     -     (-(dconjg(inpup(idb1,idq0))*yup(21))-dconjg(inpup(idb2,idq0))
     -     *yup(38)) +(dconjg(yup(2))+ yup(25))*(yup(20) + yup(39))))
c...helicity(24,3)=(-,-,-,+,+,+)
        do mm=1,4
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(9,3)=(+,-,-,+,+,+)=dconjg(helicity(12,1)(q_c1,q_c2,q_b1,q_b2))
        kk=9
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *dconjg(yup(10))*dconjg(yup(12))*yup(14) - dconjg(yup(9))
     -     *dconjg(yup(10))*dconjg(yup(14))*yup(14) +(dconjg(yup(4))* 
     -     dconjg(yup(12)) -dconjg(yup(9))*dconjg(yup(14)))*
     -     dconjg(yup(16))*yup(14) + dconjg(yup(4))*dconjg(yup(12))
     -     *yup(4)*(dconjg(yup(2)) + yup(25)) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(29) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(28)*yup(30) + dconjg(yup(9))*
     -     dconjg(yup(12))*yup(27)*(yup(29) + yup(30))) + 
     -     4*dconjg(yup(18))*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*yup(4)*(dconjg(yup(2)) + yup(25)) + 
     -     dconjg(yup(9))*(yup(27)+yup(28))*(yup(29)+yup(30))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
	  do mm=8,9
	    bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
	  end do
c...helicity(25,3)=(-,+,+,-,-,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(8))*dotup(idb1,idb1)*(yup(27) + yup(28)) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(22))*inpup(idb1,idk1)
     -     *yup(30) + dconjg(inpup(idq0,idk2))*dconjg(yup(32))*
     -     inpup(idb1,idk1)*yup(30)+dconjg(yup(7))*yup(11)*yup(30) + 
     -     dconjg(yup(7))*yup(18)*yup(30) - dconjg(inpup(idq0,idc2))*
     -     inpup(idk1,idb1)*yup(12)*yup(37)+dconjg(inpup(idc2,idk2))*
     -     inpup(idb1,idk1)*yup(30)*yup(37)+(dconjg(inpup(idc1,idq0)) 
     -     *inpup(idk1,idb1)*yup(12) + dconjg(inpup(idc1,idk2))*
     -     inpup(idb1,idk1)*yup(30))*yup(43)+yup(53)*(dconjg(yup(37)) 
     -     *yup(37) +yup(43)*yup(65)))
        bfun(2,ii,kk)=-4*pmomup(5,idb1)*pmomup(5,idc1)*((yup(27)+yup(28)
     -     )*(dconjg(yup(8))*dotup(idb1,idb1)-dconjg(yup(2))*yup(12) 
     -     +yup(30)*yup(60))+yup(43)*(-(dconjg(inpup(idc1,idk2))*
     -     inpup(idk1,idb2)*yup(35)) +yup(17)*(dconjg(yup(49)) - 
     -     yup(72))) +yup(37)*(-(dconjg(inpup(idc2,idk2))*
     -     inpup(idk1,idb2)*yup(35)) +yup(17)*(-yup(56) + yup(73))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(2))
     -     + yup(25))*((yup(11) + yup(18))*yup(31) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idq0,idb1)*yup(43) + 
     -     dconjg(inpup(idc1,idk2))*inpup(idq0,idb2)*yup(69) + 
     -     dconjg(inpup(idc2,idk2))* (inpup(idq0,idb1)*yup(37) + 
     -     inpup(idq0,idb2)*yup(71)))
	  do mm=1,4
	    bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
	  end do
c...helicity(25,3)=(-,+,+,-,-,-)
        do mm=1,4
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(10,3)=(+,-,-,+,+,-)=dconjg(helicity(11,1)(q_c1,q_c2,q_b1,q_b2))
        kk=10
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(yup(12) - yup(18))*(dconjg(yup(2)) + yup(25)) + 
     -     (dconjg(yup(9))*yup(18)*yup(19) + (2*dconjg(yup(8))*
     -     dotup(idc1,idk1) +dconjg(yup(9))*(yup(12) - yup(18)))*
     -     (yup(27) + yup(28)))*(yup(29) + yup(30))))/yup(8)
	  bfun(9,ii,kk)=0.0d0
	  do mm=8,9
	    bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
	  end do
c...helicity(26,3)=(-,+,+,-,-,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(11,3)=(+,-,-,+,-,+)=dconjg(helicity(10,1)(q_c1,q_c2,q_b1,q_b2))
        kk=11
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (dconjg(yup(14))*(dconjg(yup(10))+dconjg(yup(16)))*yup(14)- 
     -     (dconjg(yup(12)) - dconjg(yup(18)))*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/dconjg(yup(8))
	bfun(9,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(15))
     -     *yup(5)+yup(8)*yup(28))*(yup(29)+yup(30)))/dconjg(yup(8))
        do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
	  end do
c...helicity(27,3)=(-,+,+,-,+,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(12,3)=(+,-,-,+,-,-)=dconjg(helicity(9,1)(q_c1,q_c2,q_b1,q_b2))
        kk=12
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(12) - yup(18))*(yup(29) + yup(30))*(yup(19) - yup(27)
     -     - yup(28) + yup(31)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
        do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(28,3)=(-,+,+,-,+,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if
	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(dotup(idb1,idb1)
     -     *yup(8)*(yup(27) + yup(28)) + dconjg(inpup(idq0,idk1))*
     -     dconjg(yup(22))*inpup(idb1,idk2)*yup(30) + 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(32))*inpup(idb1,idk2)*
     -     yup(30) +dconjg(yup(12))*yup(1)*yup(30) + dconjg(yup(12))
     -     *yup(9)*yup(30) + dconjg(inpup(idq0,idc2))*inpup(idb1,idk2)
     -     *yup(7)*yup(37) + dconjg(inpup(idc2,idk1))*inpup(idb1,idk2)
     -     *yup(30)*yup(37) + inpup(idb1,idk2)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(7))+dconjg(inpup(idc1,idk1))
     -     *yup(30))*yup(43) +yup(46)*(dconjg(yup(37))*yup(37) + 
     -     yup(43)*yup(65)))
        bfun(2,ii,kk)=-4*pmomup(5,idb1)*pmomup(5,idc1)*((yup(27) + 
     -     yup(28))*(dotup(idb1,idb1)*yup(8) + yup(30)*yup(50) + 
     -     dconjg(yup(13))*yup(66))+yup(43)*(dconjg(inpup(idc1,idk1))*
     -     inpup(idb2,idk2)*yup(35) +yup(6)*(-dconjg(yup(54)) + 
     -     yup(67))) +yup(37)*(dconjg(inpup(idc2,idk1))*
     -     inpup(idb2,idk2)*yup(35) +yup(6)*(-yup(45) + yup(68))))
	bfun(4,ii,kk)=4*pmomup(5,idb1)*pmomup(5,idc1)*(-((yup(1)+yup(9))
     -     *(dconjg(yup(17))*yup(21) +dconjg(yup(13))*yup(38))) - 
     -     yup(8)*(yup(69)*yup(70) + yup(37)*yup(83) + yup(43)*yup(84)
     -     + yup(71)*yup(85)))
        do mm=1,4
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(28,3)=(-,+,+,-,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(13,3)=(+,-,-,-,+,+)=-dconjg(helicity(36,1)(q_c1,q_c2,q_b1,q_b2))
        kk=13
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(9))*
     -     (dconjg(yup(10))*dconjg(yup(24)) -dconjg(yup(14))* 
     -     dotup(idb1,idb1))*inpup(idq0,idk2)*yup(14) +inpup(idq0,idk1)*
     -     (dotup(idb1,idb1)*(dconjg(yup(4))*(dconjg(yup(12)) -
     -     dconjg(yup(18)))*yup(4) + dconjg(yup(12))*dconjg(yup(14))
     -     *yup(14)) - dconjg(yup(24))*(dconjg(yup(10))*dconjg(yup(12))
     -     *yup(14)+(dconjg(yup(12))-dconjg(yup(18)))*yup(4)*yup(25)))+ 
     -     dconjg(yup(35))*(-2*dotup(idc1,idk1)*inpup(idk2,idk1)*yup(27)
     -     * yup(35) +dconjg(yup(9))*(dconjg(yup(12))-dconjg(yup(18)))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36)))))/xup(2)
	  bfun(9,ii,kk)=(-4*dconjg(yup(35))*inpup(idk2,idk1)
     -     *pmomup(5,idc1)*yup(28)*yup(35))/xup(2)
	  do mm=8,9
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do 
c...helicity(29,3)=(-,+,+,+,-,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idc1)*(dconjg(yup(8))*dotup(idb1,idb1)
     -     *(dconjg(yup(39))*inpup(idc1,idq0) +dconjg(yup(24))* 
     -     inpup(idc2,idq0))+dconjg(yup(35))*(dconjg(yup(42))+yup(36))*
     -     (-(inpup(idk1,idb1)*yup(12)) +inpup(idq0,idb1)*yup(53)))
        bfun(2,ii,kk)= -4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(5))*inpup(idc1,idq0) + dconjg(yup(4))*
     -     inpup(idc2,idq0))*(yup(12) + yup(16)) +dotup(idb1,idb1) 
     -     *inpup(idq0,idk1)*(yup(16)*(yup(27) + yup(28)) - 
     -     dconjg(inpup(idc2,idk2))*inpup(idq0,idb1)*yup(37)) - 
     -     dconjg(inpup(idc1,idk2))*dotup(idb1,idb1)*inpup(idq0,idb1)
     -     *inpup(idq0,idk1)*yup(43)+dconjg(yup(35))*inpup(idq0,idb1)*
     -     ((yup(11) + yup(18))*yup(59) +(dconjg(yup(42)) + yup(36)) 
     -     *yup(60)))
	  bfun(4,ii,kk)=4*pmomup(5,idc1)*(yup(11) + yup(18))*
     -     (dotup(idb1,idb1)*(-(dconjg(yup(6))*inpup(idq0,idb1)) - 
     -     dconjg(yup(3))*inpup(idq0,idb2) + dconjg(inpup(idq0,idk1))
     -     *inpup(idq0,idk1)**2)+ dconjg(yup(35))*inpup(idb1,idk1)*
     -     (-dconjg(yup(38)) + yup(61)))
	  do mm=1,4
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do 
c...helicity(29,3)=(-,+,+,+,-,-)
        do mm=1,4
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(14,3)=(+,-,-,-,+,-)=-dconjg(helicity(35,1)(q_c1,q_c2,q_b1,q_b2))
        kk=14
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)
     -     *(yup(12) - yup(18))*(dconjg(yup(4))*dotup(idb1,idb1) - 
     -     dconjg(yup(24))*yup(25)) +2*dotup(idc1,idk1)*
     -     (-(inpup(idq0,idk1)*yup(16)*yup(27)*yup(29)) + 
     -     dconjg(yup(8))*dconjg(yup(35))*inpup(idq0,idb1)*yup(36)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*(dconjg(yup(41))*
     -     inpup(idq0,idb1)*yup(18)+inpup(idq0,idb1)*(yup(12)-yup(18))*
     -     (dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(9,ii,kk)=(-4*dconjg(yup(3))*dconjg(yup(35))*
     -    inpup(idq0,idb1)*pmomup(5,idc1)*yup(11))/yup(8)
	  do mm=8,9
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(30,3)=(-,+,+,+,-,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(15,3)=(+,-,-,-,-,+)=-dconjg(helicity(34,1)(q_c1,q_c2,q_b1,q_b2))
        kk=15
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*(inpup(idq0,idk2)*
     -     ((dconjg(yup(10))*dconjg(yup(24)) - dconjg(yup(14))
     -     *dotup(idb1,idb1))*yup(14)+2*dconjg(yup(35))*dotup(idc1,idk1)
     -     *yup(35))+(dconjg(yup(12))-dconjg(yup(18)))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) +inpup(idq0,idb1) 
     -     *(dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*inpup(idq0,idk2)*pmomup(5,idc1)*
     -     (dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idq0,idb1)
     -     *yup(5) +yup(2)*yup(28)*yup(29)))/dconjg(yup(8))
	  do mm=8,9
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(31,3)=(-,+,+,+,+,-)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(16,3)=(+,-,-,-,-,-)=-dconjg(helicity(33,1)(q_c1,q_c2,q_b1,q_b2))
        kk=16
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(-((dconjg(yup(38))+dconjg(yup(41)))*
     -     inpup(idq0,idb1)) + inpup(idq0,idb1)*(dconjg(yup(42)) +
     -     yup(36))))/xup(1)
	  bfun(9,ii,kk)=0.0d0
	  do mm=8,9
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(32,3)=(-,+,+,+,+,+)
        do mm=8,9
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(39))*inpup(idc1,idq0) +dconjg(yup(24)) 
     -     *inpup(idc2,idq0))*yup(8)+ dconjg(yup(35))*(dconjg(yup(42))
     -     +yup(36))*(inpup(idb1,idk2)*yup(7)+inpup(idq0,idb1)*yup(46)))
        bfun(2,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     (dconjg(yup(15))*inpup(idc1,idq0) + dconjg(yup(14))*
     -     inpup(idc2,idq0))*yup(2) + dconjg(inpup(idk2,idk1))*
     -     dotup(idb1,idb1)*inpup(idq0,idk2)**2*(yup(27) + yup(28)) + 
     -     dotup(idb1,idb1)*inpup(idq0,idk2)*(yup(2)*(yup(27)+yup(28))- 
     -     dconjg(inpup(idc2,idk1))*inpup(idq0,idb1)*yup(37)) - 
     -     dconjg(inpup(idc1,idk1))*dotup(idb1,idb1)*inpup(idq0,idb1)*
     -     inpup(idq0,idk2)*yup(43)+dconjg(yup(35))*inpup(idq0,idb1)*
     -     ((yup(1)+yup(9))*yup(48)+(dconjg(yup(42))+yup(36))*yup(50)))
	  bfun(4,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -     ((dconjg(yup(20)) + dconjg(yup(39)))*inpup(idc1,idq0) + 
     -     (dconjg(yup(23))+dconjg(yup(24)))*inpup(idc2,idq0))*yup(8)- 
     -     inpup(idq0,idk2)*((yup(1)+yup(9))*(dotup(idb1,idb1)*yup(31)- 
     -     dconjg(yup(35))*yup(51)) + dconjg(inpup(idb2,idk1))*
     -     dconjg(yup(35))*(inpup(idc2,idq0)*yup(83) + 
     -     inpup(idc1,idq0)*yup(84))))
	  do mm=1,4
	    bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(32,3)=(-,+,+,+,+,+)
        do mm=1,4
	    bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if (ibcstate.eq.1) go to 1223

c...helicity(33,3)=(+,+,-,+,+,+)=-dconjg(helicity(16,1)(q_c1,q_c2,q_b1,q_b2)
        kk=33
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -    (dconjg(yup(12))-dconjg(yup(18)))*pmomup(5,idb1)*(yup(23)+
     -     yup(24))*yup(34))/xup(2)
	  bfun(9,ii,kk)=0.0d0
	  do mm=8,9
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(49,3)=(-,-,+,-,-,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idq0,idk2))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30)) - 
     -     yup(34)*(-(dconjg(inpup(idc1,idq0))*dconjg(yup(45))*
     -     yup(25)) +dconjg(inpup(idc1,idk2))*yup(24)*(dconjg(yup(7)) 
     -     + yup(25)) +dconjg(inpup(idc1,idq0))*yup(23)*yup(53)) + 
     -     dconjg(inpup(idq0,idk2))*(dotup(idc1,idc1)*
     -     (-(dconjg(inpup(idb2,idq0))*dconjg(yup(35))*inpup(idb1,idk1))
     -     + yup(25)*yup(30)) +yup(34)*(-(dconjg(inpup(idc1,idq0))*
     -     dotup(idb1,idb1)*inpup(idc2,idk1)) +yup(24)*yup(81))))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(-(dconjg(inpup(idb1,idq0))
     -     *dconjg(yup(2))*dotup(idc1,idc1)*yup(17)) - 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(2))*dotup(idc1,idc1)
     -     *yup(29) - dconjg(inpup(idq0,idk2))**2*dotup(idc1,idc1)*
     -     inpup(idk2,idk1)*yup(29) + dconjg(inpup(idc1,idq0))*
     -     dconjg(yup(2))*dconjg(yup(45))*yup(34) - 
     -     dconjg(inpup(idc1,idq0))*dconjg(inpup(idq0,idk2))*
     -     dotup(idb1,idb1)*inpup(idc2,idk1)*yup(34) - 
     -     dconjg(inpup(idc1,idk2))*dconjg(yup(2))*yup(23)*yup(34) - 
     -     dconjg(inpup(idc1,idk2))*dconjg(yup(7))*yup(23)*yup(34) + 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(49))*yup(23)*
     -     yup(34) - dconjg(inpup(idc1,idq0))*yup(24)*yup(34)*yup(60))
	  bfun(4,ii,kk)=-4*pmomup(5,idb1)*(dconjg(inpup(idb2,idq0))
     -     *dotup(idc1,idc1)*yup(13)*(dconjg(yup(2)) + yup(25)) + 
     -     dconjg(inpup(idb1,idq0))*dotup(idc1,idc1)*yup(17)*
     -     (dconjg(yup(2)) + yup(25)) + dconjg(inpup(idq0,idk2))
     -     *dotup(idc1,idc1)*yup(25)*yup(31)-dconjg(inpup(idb1,idq0))*
     -     dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*yup(21)*yup(34)+ 
     -     dconjg(inpup(idc1,idk2))*yup(23)*yup(25)*yup(34) +
     -     dconjg(inpup(idc1,idk2))*yup(24)*yup(25)*yup(34) + 
     -     dconjg(yup(2))*(dconjg(inpup(idq0,idk2))*dotup(idc1,idc1)*
     -     yup(31) + dconjg(inpup(idc1,idk2))*(yup(23) + yup(24))
     -     *yup(34))-dconjg(inpup(idb2,idq0))*dconjg(inpup(idc1,idk2))*
     -     inpup(idc2,idk1)*yup(34)*yup(38))
	  do mm=1,4
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(49,3)=(-,-,+,-,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(34,3)=(+,+,-,+,+,-)=-dconjg(helicity(15,1)(q_c1,q_c2,q_b1,q_b2)
        kk=34
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk2))*(dconjg(yup(9))*dconjg(yup(14))
     -     *yup(18) +2*dconjg(yup(4))*dotup(idc1,idk1)*yup(27)) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(9))*(-yup(12)+yup(18))
     -     *yup(34)))/yup(8)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*pmomup(5,idb1)
     -    *yup(15)*(dconjg(yup(2))+yup(25))*yup(28))/yup(8)
	  do mm=8,9
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(50,3)=(-,-,+,-,-,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(35,3)=(+,+,-,+,-,+)=-dconjg(helicity(14,1)(q_c1,q_c2,q_b1,q_b2)
        kk=35
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     pmomup(5,idb1)*yup(9)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(4)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idq0))*dconjg(yup(11))*
     -     pmomup(5,idb1)*yup(5)*(yup(29) + yup(30)))/dconjg(yup(8))
	  do mm=8,9
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(51,3)=(-,-,+,-,+,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(36,3)=(+,+,-,+,-,-)=-dconjg(helicity(13,1)(q_c1,q_c2,q_b1,q_b2)
        kk=36
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*(yup(12) - yup(18))*
     -     (yup(23)+yup(24))*(dconjg(inpup(idq0,idk1))*dconjg(yup(4))+ 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(14)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*pmomup(5,idb1)*
     -     yup(27)*yup(28)*(yup(29) + yup(30)))/xup(1)
	  do mm=8,9
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(52,3)=(-,-,+,-,+,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=-4*pmomup(5,idb1)*(-(dconjg(inpup(idb2,idq0))*
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(35))*dotup(idc1,idc1)
     -     *inpup(idb1,idk2))+dconjg(inpup(idq0,idk1))*dconjg(yup(10))*
     -     dotup(idc1,idc1)*yup(30) +dconjg(inpup(idq0,idk1))**2 
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(30) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(10))*dconjg(yup(56))*
     -     yup(34) -dconjg(inpup(idc1,idq0))*dconjg(inpup(idq0,idk1))*
     -     dotup(idb1,idb1)*inpup(idc2,idk2)*yup(34) - 
     -     dconjg(inpup(idc1,idk1))*dconjg(yup(10))*yup(24)*yup(34) - 
     -     dconjg(inpup(idc1,idk1))*dconjg(yup(12))*yup(24)*yup(34) - 
     -     dconjg(inpup(idc1,idq0))*yup(23)*yup(34)*yup(46) + 
     -     dconjg(inpup(idq0,idk1))*yup(24)*yup(34)*yup(74))
        bfun(2,ii,kk)=4*pmomup(5,idb1)*(dconjg(inpup(idq0,idk1))**2
     -     *dotup(idc1,idc1)*inpup(idk2,idk1)*yup(29)-dconjg(yup(16))*
     -     (dotup(idc1,idc1)*(dconjg(inpup(idb1,idq0))*yup(6) + 
     -     dconjg(inpup(idq0,idk1))*yup(29))+dconjg(inpup(idc1,idk1))
     -     *yup(23)*yup(34)) - yup(34)*(dconjg(inpup(idc1,idk1))
     -     *dconjg(yup(12))*yup(23) + dconjg(inpup(idc1,idq0))*yup(24)*
     -     yup(50)- dconjg(inpup(idb2,idq0))*dconjg(inpup(idc1,idq0))
     -     *inpup(idc2,idk2)*yup(66) - dconjg(inpup(idq0,idk1))*
     -     (-(dconjg(inpup(idc1,idq0))*dotup(idb1,idb1)*
     -     inpup(idc2,idk2)) + yup(23)*yup(67))))
	  bfun(4,ii,kk)=4*pmomup(5,idb1)*(-(dconjg(inpup(idc1,idk1))
     -     *inpup(idc2,idk2)*yup(34)*(-(dconjg(inpup(idb1,idq0))*
     -     yup(21)) - dconjg(inpup(idb2,idq0))*yup(38))) - 
     -     dconjg(inpup(idb2,idq0))*dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(34)*yup(70) + dconjg(inpup(idq0,idk1))
     -     *dotup(idc1,idc1)*(dconjg(yup(13))*(dconjg(yup(61)) - 
     -     dconjg(inpup(idq0,idk1))*inpup(idk1,idb2))+dconjg(yup(17))*
     -     (dconjg(inpup(idq0,idk1))*inpup(idb1,idk1) + yup(75))) - 
     -     dconjg(inpup(idb1,idq0))*dconjg(inpup(idq0,idk1))*
     -     inpup(idc2,idk2)*yup(34)*yup(84))
	  do mm=1,4
	   bfun(mm,ii,kk)=-dconjg(bfun(mm,ii,kk))
        end do
c...helicity(52,3)=(-,-,+,-,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(37,3)=(+,+,-,-,+,+)=dconjg(helicity(40,1)(q_c1,q_c2,q_b1,q_b2)
        kk=37
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(30)*(dconjg(yup(9))*
     -     dconjg(yup(10))*yup(19) + (dconjg(yup(7))*dconjg(yup(10)) + 
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*yup(25))*yup(31)) + 
     -     4*dconjg(yup(9))*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     dconjg(yup(34))*dotup(idb1,idb1)*yup(34)+4*dconjg(yup(35))*
     -     (-2*dconjg(inpup(idb2,idq0))*dotup(idc1,idk1)*
     -     inpup(idk2,idk1)*yup(22)*yup(27) + dconjg(yup(9))*
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(2)
	  bfun(9,ii,kk)=0.0d0
        do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(53,3)=(-,-,+,+,-,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=-4*(dconjg(yup(8))*dotup(idb1,idb1)*
     -     dotup(idc1,idc1)*yup(30) - dconjg(inpup(idq0,idk2))*
     -     dconjg(yup(39))*dotup(idb1,idb1)*inpup(idc2,idk1)*yup(34)+ 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*dconjg(yup(45))*
     -     inpup(idk1,idb1)*yup(34) + dconjg(inpup(idc1,idk2))*
     -     dotup(idb1,idb1)*inpup(idb1,idk1)*yup(34)*yup(37) - 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)
     -     *yup(34)*yup(53) +dconjg(yup(35))*dotup(idc1,idc1)*yup(35)*
     -     yup(53) + dconjg(yup(3))*dconjg(yup(35))*dotup(idc1,idc1)*
     -     yup(57) -dotup(idb1,idb1)*yup(14)*yup(34)*yup(72) + 
     -     dotup(idb1,idb1)*yup(14)*yup(34)*yup(81))
        bfun(2,ii,kk)=4*dotup(idb1,idb1)*(dconjg(yup(8))*
     -     dotup(idc1,idc1)*yup(29) -(dconjg(yup(5))*
     -     (dconjg(yup(45))-dconjg(yup(68)))- dconjg(inpup(idc1,idq0))
     -     *inpup(idc2,idk1)*yup(16) +dconjg(inpup(idc1,idk2)) 
     -     *inpup(idq0,idk1)*yup(23))*yup(34)) +4*dconjg(yup(35))*
     -     ((dotup(idc1,idc1)*yup(17) - dconjg(inpup(idc1,idk2))
     -     *inpup(idc2,idb1)*yup(34))*yup(59) + 
     -     (-(dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34)) + dotup(idc1,idc1)*yup(35))*yup(60))
	  bfun(4,ii,kk)=-4*(-(dconjg(inpup(idq0,idk2))*dconjg(yup(35))*
     -    dconjg(yup(38))*dotup(idc1,idc1)*inpup(idb1,idk1)) + 
     -    dconjg(inpup(idq0,idk1))*dconjg(inpup(idq0,idk2))*
     -    dotup(idb1,idb1)*dotup(idc1,idc1)*inpup(idq0,idk1)**2 - 
     -    dconjg(yup(3))*dotup(idb1,idb1)*dotup(idc1,idc1)*
     -    yup(13) - dconjg(yup(6))*dotup(idb1,idb1)*dotup(idc1,idc1)
     -    *yup(17) +dconjg(inpup(idc1,idk2))*dconjg(yup(6))*
     -    dotup(idb1,idb1)*inpup(idc2,idb1)*yup(34) + 
     -    dconjg(inpup(idc1,idk2))*dconjg(yup(3))*
     -    dotup(idb1,idb1)*inpup(idc2,idb2)*yup(34) + 
     -    dconjg(inpup(idc1,idk2))*inpup(idc2,idk1)*yup(34)*
     -    (-(dotup(idb1,idb1)*yup(31)) +dconjg(yup(35))*yup(51)) + 
     -    dconjg(inpup(idq0,idk2))*dconjg(yup(35))*
     -    dotup(idc1,idc1)*inpup(idb1,idk1)*yup(61) - 
     -    dconjg(inpup(idc1,idk2))*dconjg(yup(35))*
     -    inpup(idb1,idk1)*yup(34)*yup(64))
	  do mm=1,4
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(53,3)=(-,-,+,+,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(38,3)=(+,+,-,-,+,-)=dconjg(helicity(39,1)(q_c1,q_c2,q_b1,q_b2)
        kk=38
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*(2*dconjg(yup(4))*dotup(idc1,idk1)*
     -     (dotup(idb1,idb1)*yup(14) - yup(16)*yup(23))*yup(27) +
     -     dconjg(yup(9))*(-(dconjg(yup(14))*yup(16)*yup(18)*yup(23)) + 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)*
     -     (-yup(12) + yup(18))*yup(34) + dotup(idb1,idb1)*
     -     (dconjg(yup(14))*yup(14)*yup(18) +dconjg(yup(34))*(yup(12) 
     -     - yup(18))*yup(34))) + dotup(idc1,idc1)*(2*dconjg(yup(8))
     -     *dconjg(yup(35))*dotup(idc1,idk1)*yup(35)+(yup(12)-yup(18))*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
	  bfun(9,ii,kk)=(4*(dconjg(yup(5))*dotup(idb1,idb1)*yup(15)*
     -     yup(28) + dconjg(yup(35))*(-(dconjg(inpup(idc1,idb2))*
     -     inpup(idk1,idb1)*yup(15)*yup(28)) + 
     -     dconjg(yup(8))*dotup(idc1,idc1)*yup(35))))/yup(8)
        do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(54,3)=(-,-,+,+,-,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(39,3)=(+,+,-,-,-,+)=dconjg(helicity(38,1)(q_c1,q_c2,q_b1,q_b2)
        kk=39
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*yup(9)*((dconjg(yup(12)) - dconjg(yup(18)))*
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(-((dconjg(inpup(idb2,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18))) - 
     -     2*dconjg(inpup(idb2,idq0))*dconjg(yup(14))*dotup(idc1,idk1))*
     -     inpup(idc2,idb1)) +dconjg(inpup(idc1,idb2))*(dconjg(yup(12))
     -     -dconjg(yup(18)))*inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*
     -     (dconjg(yup(10))*yup(19)*yup(30) + (-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dconjg(yup(35))*yup(35))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idc1,idb2))*dconjg(yup(11))*
     -     dconjg(yup(35))*inpup(idq0,idb1)*yup(5))/dconjg(yup(8))
	  do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(55,3)=(-,-,+,+,+,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(40,3)=(+,+,-,-,-,-)=dconjg(helicity(37,1)(q_c1,q_c2,q_b1,q_b2)
        kk=40
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*yup(9)*(yup(12) - yup(18))*
     -     (dconjg(yup(14))*yup(16)*yup(23) -dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) + dconjg(yup(14))*yup(14) - 
     -     dconjg(yup(34))*yup(34)) +dconjg(yup(35))*
     -     (-(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1))- 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) + dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(-4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -     dconjg(yup(42))*inpup(idq0,idb1)*yup(28))/xup(1)
	  do mm=8,9
	   bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(56,3)=(-,-,+,+,+,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*(dconjg(yup(35))*((dconjg(inpup(idc1,idb2))
     -     *inpup(idc2,idb1)*yup(34)-dotup(idc1,idc1)*yup(35))*yup(46)
     -     + (-(dconjg(yup(13))*dotup(idc1,idc1)) + 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idk2)*yup(34))*yup(47))- 
     -     dotup(idb1,idb1)*(dotup(idc1,idc1)*yup(8)*yup(30) + 
     -     yup(34)*(-(dconjg(inpup(idq0,idk1))*dconjg(yup(39))*
     -     inpup(idc2,idk2))+dconjg(inpup(idc1,idk1))*inpup(idb1,idk2)*
     -     yup(37) + yup(4)*(-dconjg(yup(54)) + yup(74)))))
        bfun(2,ii,kk)=4*(dconjg(yup(35))*((dotup(idc1,idc1)*yup(6) - 
     -     dconjg(inpup(idc1,idk1))*inpup(idc2,idb1)*yup(34))*yup(48) + 
     -     (-(dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*yup(34)) + 
     -     dotup(idc1,idc1)*yup(35))*yup(50)) +dotup(idb1,idb1)*
     -     (dotup(idc1,idc1)*yup(8)*yup(29)+yup(34)*(inpup(idc2,idk2)*
     -     (dconjg(inpup(idc1,idq0))*yup(2) +dconjg(inpup(idc1,idq0)) 
     -     *yup(7)) -dconjg(inpup(idc1,idk1))*inpup(idq0,idk2)*
     -     yup(23) + dconjg(yup(15))*yup(86))))
	bfun(4,ii,kk)=4*(-(dconjg(inpup(idq0,idk1))**2*dotup(idb1,idb1)*
     -     dotup(idc1,idc1)*inpup(idq0,idk1)*inpup(idq0,idk2)) + 
     -     dotup(idc1,idc1)* (dotup(idb1,idb1)*(dconjg(yup(13))*yup(3)+ 
     -     dconjg(yup(17))*yup(6)) +  dconjg(yup(35))*
     -     (dconjg(inpup(idb2,idk1))*inpup(idq0,idk2)*yup(21) + 
     -     dconjg(yup(13))*yup(82))) + inpup(idc2,idk2)*yup(34)*
     -     (-(dconjg(inpup(idq0,idk1))*(dconjg(yup(20)) +
     -     dconjg(yup(39)))*dotup(idb1,idb1))+dconjg(inpup(idc1,idk1))*
     -     (dotup(idb1,idb1)*yup(31) -dconjg(yup(35))*yup(51)) + 
     -     dconjg(inpup(idb2,idk1))*dconjg(yup(35))*yup(84)))
	  do mm=1,4
	     bfun(mm,ii,kk)=dconjg(bfun(mm,ii,kk))
        end do
c...helicity(56,3)=(-,-,+,+,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(41,3)=(+,-,+,+,+,+)=helicity(5,1)(q_c1,q_c2,q_b1,q_b2)
        kk=41
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(14))*yup(9)*(dotup(idb1,idb1)*yup(14) - yup(16)
     -     *yup(23))) -dconjg(inpup(idq0,idk1))*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*yup(4)*(yup(12)-yup(18)))-dconjg(yup(14)) 
     -     *yup(12)*(dotup(idb1,idb1)*yup(14) - yup(16)*yup(23)))-
     -     dconjg(yup(35))*((yup(12)-yup(18))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31)+dconjg(inpup(idb2,idq0))*yup(9)*(yup(22) +
     -     yup(32))) +2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     yup(27)*yup(35))))/xup(1)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idk1))*dconjg(yup(35))*
     -     pmomup(5,idc1)*yup(28)*yup(35))/xup(1)
c...helicity(57,3)=(-,+,-,-,-,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dotup(idb1,idb1)*((dconjg(yup(10))-dconjg(inpup(idq0,idk1))
     -     *inpup(idk2,idk1))*(yup(27)+yup(28))+(dconjg(yup(11)) +  
     -     dconjg(yup(18)))*yup(30) +inpup(idb1,idk2)*
     -     (dconjg(inpup(idq0,idc2))*yup(37)- dconjg(inpup(idc1,idq0))
     -     *yup(43)))) +dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     ((yup(22) + yup(32))*yup(46) + 
     -     (dconjg(yup(11)) + dconjg(yup(18)))*yup(47)))
        bfun(2,ii,kk)=4*pmomup(5,idc1)*(-(dotup(idb1,idb1)*yup(8)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23)))-dconjg(yup(35))*(yup(22)
     -     + yup(32))*(-(dconjg(inpup(idb2,idk1))*dconjg(yup(12))) + 
     -     dconjg(inpup(idb2,idq0))*yup(50)))
	  bfun(4,ii,kk)=-4*pmomup(5,idc1)*(dotup(idb1,idb1)*
     -    (-(dconjg(inpup(idq0,idk1))**2*(dconjg(yup(5))*
     -    inpup(idc1,idk2) +dconjg(yup(4))*inpup(idc2,idk2))) + 
     -    (dconjg(yup(11))+dconjg(yup(18)))*(dconjg(inpup(idb2,idq0))*
     -    yup(3)+dconjg(inpup(idb1,idq0))*yup(6)))+ (dconjg(yup(11))
     -    + dconjg(yup(18)))*dconjg(yup(35))*(-(dconjg(inpup(idb2,idk1)
     -    )*yup(21)) +dconjg(inpup(idb2,idq0))*yup(82)))
c...helicity(57,3)=(-,+,-,-,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(42,3)=(+,-,+,+,+,-)=helicity(6,1)(q_c1,q_c2,q_b1,q_b2)
        kk=42
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18)))*
     -     dotup(idb1,idb1)*yup(4)) - 2*dotup(idc1,idk1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(yup(35))*yup(8)*yup(22)+ 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(10))*yup(27)*yup(30)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*yup(22)*yup(31) + dconjg(inpup(idb2,idq0))
     -     *yup(9)*(dconjg(yup(12))*(yup(22)+yup(32))-dconjg(yup(18))*
     -     (yup(22) - yup(26) + yup(32))))))/dconjg(yup(8))
	  bfun(9,ii,kk)=(4*dconjg(inpup(idq0,idk1))*dconjg(yup(11))*
     -     dconjg(yup(35))*pmomup(5,idc1)*yup(35))/dconjg(yup(8))
c...helicity(58,3)=(-,+,-,-,-,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(43,3)=(+,-,+,+,-,+)=helicity(7,1)(q_c1,q_c2,q_b1,q_b2)
        kk=43
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(14))*(dotup(idb1,idb1)
     -     *yup(14) - yup(16)*yup(23)))+ dconjg(inpup(idb2,idq0))
     -     *dconjg(yup(35))*(-2*dotup(idc1,idk1)*yup(17)+(yup(12)-
     -     yup(18))*(yup(21)- yup(22) - yup(32)))))/yup(8)
	  bfun(9,ii,kk)=(4*dconjg(inpup(idk2,idq0))*pmomup(5,idc1)*
     -     (dconjg(yup(5))*yup(20) + yup(25)*yup(28))*yup(30))/yup(8)
c...helicity(59,3)=(-,+,-,-,+,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(44,3)=(+,-,+,+,-,-)=helicity(8,1)(q_c1,q_c2,q_b1,q_b2)
        kk=44
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*dconjg(yup(35))
     -     *pmomup(5,idc1)*(yup(21)-yup(22)+yup(26)-yup(32)))/xup(2)
	  bfun(9,ii,kk)=0.0d0
c...helicity(60,3)=(-,+,-,-,+,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

        if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))*
     -     dotup(idb1,idb1)*((dconjg(inpup(idq0,idk2))*inpup(idk2,idk1)
     -     +yup(25))*(yup(27) + yup(28)) + (dconjg(yup(1)) +
     -     dconjg(yup(9)))*yup(30) +inpup(idb1,idk1)*
     -     (dconjg(inpup(idq0,idc2))*yup(37)-dconjg(inpup(idc1,idq0))
     -     *yup(43)))) +dconjg(inpup(idb2,idq0))*dconjg(yup(35))*
     -     ((yup(22) + yup(32))*yup(53) + 
     -     (dconjg(yup(1)) + dconjg(yup(9)))*yup(57)))
        bfun(2,ii,kk)=4*pmomup(5,idc1)*(-(dconjg(yup(8))*
     -     dotup(idb1,idb1)*(-(dconjg(inpup(idc1,idq0))*yup(20)) + 
     -     dconjg(inpup(idq0,idc2))*yup(23))) -dconjg(yup(35))* (yup(22)
     -     +yup(32))*(-(dconjg(inpup(idb2,idk2))*dconjg(yup(7)))+ 
     -     dconjg(inpup(idb2,idq0))*yup(60)))
	  bfun(4,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idc1)*
     -    (dotup(idb1,idb1)*(-((dconjg(yup(1))+dconjg(yup(9)))*yup(31))+ 
     -    dconjg(yup(6))*(yup(22) + yup(32)) + dconjg(yup(3))*
     -    (dconjg(yup(36)) + yup(42))) + dconjg(yup(35))*
     -    ((dconjg(yup(1)) + dconjg(yup(9)))*yup(51) +inpup(idb1,idk1)*
     -    (-(dconjg(inpup(idc1,idq0))*yup(62))+dconjg(inpup(idq0,idc2))
     -    *yup(64))))
c...helicity(60,3)=(-,+,-,-,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=-dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(45,3)=(+,-,+,-,+,+)=helicity(45,1)(q_c1,q_c2,q_b1,q_b2)
        kk=45
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(14))*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9)) + 
     -     dconjg(inpup(idq0,idk1))*yup(12)) + 
     -     dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (yup(12) - yup(18)))*(yup(23) + yup(24)))/xup(1)
	  bfun(9,ii,kk)=0.0d0
c...helicity(61,3)=(-,+,-,+,-,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*dconjg(inpup(idq0,idk1))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(dconjg(inpup(idq0,idk1))*inpup(idk2,idk1)*
     -     (dconjg(inpup(idq0,idc2))*yup(24)-dconjg(inpup(idc1,idq0)) 
     -     *yup(39)) +dconjg(yup(10))*(dconjg(inpup(idq0,idc2))
     -     *yup(24) -dconjg(inpup(idc1,idq0))*yup(39) - 
     -     dconjg(inpup(idb2,idq0))*(dconjg(yup(36))+yup(42))))
        bfun(2,ii,kk)=4*dconjg(inpup(idq0,idk1))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))*inpup(idk2,idk1)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2)) 
     -     *yup(23))) +dconjg(yup(16))*(dconjg(inpup(idc1,idq0))*
     -     yup(20) -dconjg(inpup(idq0,idc2))*yup(23) + 
     -     dconjg(inpup(idb1,idq0))*(yup(22) + yup(32))))
	  bfun(4,ii,kk)=-4*dconjg(inpup(idq0,idk1))*
     -    (dconjg(yup(11)) + dconjg(yup(18)))*pmomup(5,idb1)*
     -    pmomup(5,idc1)*(-(dconjg(inpup(idb2,idq0))*dconjg(yup(61)))+ 
     -    dconjg(inpup(idq0,idk1))*(dconjg(yup(2)) + yup(25)) - 
     -    dconjg(inpup(idb1,idq0))*yup(75))
c...helicity(61,3)=(-,+,-,+,-,-)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(46,3)=(+,-,+,-,+,-)=helicity(46,1)(q_c1,q_c2,q_b1,q_b2)
        kk=46
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(23) + yup(24)))/dconjg(yup(8))
	  bfun(9,ii,kk)=0.0d0
c...helicity(62,3)=(-,+,-,+,-,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(47,3)=(+,-,+,-,-,+)=helicity(47)(q_c1,q_c2,q_b1,q_b2)
        kk=47
        if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))*
     -     dconjg(yup(14))*pmomup(5,idb1)*pmomup(5,idc1)*(yup(23) 
     -     + yup(24)))/yup(8)
	  bfun(9,ii,kk)=0.0d0
c...helicity(63,3)=(-,+,-,+,+,-)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

c...helicity(48,3)=(+,-,+,-,-,-)=helicity(48,1)(q_c1,q_c2,q_b1,q_b2)
        kk=48
	  if(iqqbar.eq.0) then
	  bfun(8,ii,kk)=0.0d0
	  bfun(9,ii,kk)=0.0d0
c...helicity(64,3)=(-,+,-,+,+,+)
        do mm=8,9
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if

	  if(iqqbar.eq.1) then
	  bfun(1,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))*inpup(idk2,idk1)*
     -     (dconjg(inpup(idq0,idc2))*yup(24)-dconjg(inpup(idc1,idq0)) 
     -     *yup(39))) +yup(25)*(dconjg(inpup(idq0,idc2))*yup(24) - 
     -     dconjg(inpup(idc1,idq0))*yup(39) -dconjg(inpup(idb2,idq0))*
     -     (dconjg(yup(36)) + yup(42))))
        bfun(2,ii,kk)=4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(dconjg(inpup(idq0,idk2))*inpup(idk2,idk1)*
     -     (-(dconjg(inpup(idc1,idq0))*yup(20))+dconjg(inpup(idq0,idc2))
     -     *yup(23)) +dconjg(yup(2))*(dconjg(inpup(idc1,idq0))*yup(20)- 
     -     dconjg(inpup(idq0,idc2))*yup(23)+dconjg(inpup(idb1,idq0))* 
     -     (yup(22)+yup(32))))
	  bfun(4,ii,kk)=-4*dconjg(inpup(idq0,idk2))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(-((dconjg(yup(1)) + dconjg(yup(9)))*
     -     (-(dconjg(inpup(idb1,idq0))*yup(21)) - 
     -     dconjg(inpup(idb2,idq0))*yup(38))) + 
     -     (dconjg(yup(2)) + yup(25))*(dconjg(inpup(idq0,idc2))*
     -     (yup(23) + yup(24)) - dconjg(inpup(idc1,idq0))*(yup(20)
     -      + yup(39))))
c...helicity(64,3)=(-,+,-,+,+,+)
        do mm=1,4
	   bfun(mm,ii,kk+16)=dconjg(bfun(mm,ii,kk))
        end do
	  end if
	
1223  continue	
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c           nine basic functions from first by k1 <-> k1             c
c                  and b quark <-> c quark.                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	subroutine bfourth
	implicit double precision (a-h,o-z)
	implicit integer(i-n)
	double complex bfun,colmat,yup,xup,inpup,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/tryup/bfun(9,4,64),yup(100),xup(2),idp,idq0,idk1,idk2,
     &	idb1,idb2,idc1,idc2
      common/counter/ibcstate,nev
c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

c...if (l1.ne.l3 .and. l2.ne.l4 .and. l5.ne.l6) then
c...hel(i,4)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,-l2,-l3,-l4,-l5,-l6)
c...(q_b1<->q_c1,q_b2->q_c2,k_1<->k_2), where l1,...,l6 are helicities 
c...of q_b1,q_b2, q_c1,q_c2,k_1,k_2; (l2=l3);
c...if (l1.ne.l3 .and. l2.ne.l4 .and. l5.eq.l6) then
c...hel(i,4)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,-l2,-l3,-l4,l5,l6)
c...(q_b1<->q_c1,q_b2->q_c2,k_1<->k_2)
c...if (l1.ne.l3 .and. l2.eq.l4 .and. l5.ne.l6) then
c...hel(i,4)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,l2,-l3,l4,-l5,-l6)
c...(q_b1<->q_c1,q_b2->q_c2,k_1<->k_2)
c...if (l1.ne.l3 .and. l2.eq.l4 .and. l5.eq.l6) then
c...hel(i,4)(l1,l2,l3,l4,l5,l6)=hel(i,1)(-l1,l2,-l3,l4,l5,l6)
c...(q_b1<->q_c1,q_b2->q_c2,k_1<->k_2)
c...other four conditions can be obtained from the same way.

	 ii=4
       do kk=1,64
	  bfun(1,ii,kk)=bfun(4,1,kk)
        bfun(2,ii,kk)=bfun(2,1,kk)
	  bfun(3,ii,kk)=bfun(3,1,kk)
	  bfun(4,ii,kk)=bfun(1,1,kk)
	  bfun(5,ii,kk)=bfun(5,1,kk)
	  bfun(9,ii,kk)=bfun(9,1,kk)+bfun(9,2,kk)-bfun(9,3,kk)
	 end do

c...helicity(1,4)=(+,+,+,+,+,+)=helicity(1,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=1
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(29)*(yup(9)*yup(16)*
     -     yup(19)+(yup(7)*yup(16) + yup(2)*(-yup(12) + yup(18)))*
     -     yup(31)) +4*dconjg(yup(34))*yup(9)*(yup(12) - yup(18))*
     -     (-(dconjg(yup(24))*yup(20)) +dotup(idb1,idb1)*yup(34)) + 
     -     4*dconjg(yup(35))*(dotup(idc1,idc1)*yup(9)*(yup(12)-yup(18))*
     -     yup(35) + 2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     inpup(idq0,idb1)*yup(27)*yup(36)))/xup(1)
c...helicity(17,4)=(-,-,-,-,-,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(2,4)=m(+,+,+,+,+,-)=helicity(3,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=2
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(24))*((yup(12)
     -     - yup(18))*(-(dconjg(yup(34))*yup(20))+yup(4)*yup(25)) + 
     -     2*dotup(idc1,idk1)*yup(14)*yup(29))+dotup(idb1,idb1)*(yup(12) 
     -     -yup(18))*(-(dconjg(yup(4))*yup(4))+dconjg(yup(34))*yup(34))+ 
     -     dotup(idc1,idc1)*(-(yup(16)*yup(19)*yup(29))+dconjg(yup(35))
     -     *(yup(12) - yup(18))*yup(35))))/yup(8)
c...helicity(18)=m(-,-,-,-,-,+)
	  bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(3,4)=(+,+,+,+,-,+)=helicity(2,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=3
	  bfun(8,ii,kk)=(4*((-(dconjg(yup(10))*dconjg(yup(24))) + 
     -     dconjg(yup(14))*dotup(idb1,idb1))*(dconjg(yup(18))*yup(9)
     -     *yup(14)+2*dotup(idc1,idk1)*yup(4)*yup(27))+(dconjg(yup(12))
     -     -dconjg(yup(18)))*dconjg(yup(34))*yup(9)*(-(dconjg(yup(24))
     -     *yup(20)) +dotup(idb1,idb1)*yup(34)) +dotup(idc1,idc1)*
     -     (2*dconjg(yup(35))*dotup(idc1,idk1)*yup(8)*yup(35) + 
     -     (dconjg(yup(12))-dconjg(yup(18)))*(yup(2)*yup(29)*yup(31)+ 
     -     dconjg(yup(35))*yup(9)*yup(35)))))/dconjg(yup(8))
c...helicity(19,4)=(-,-,-,-,+,-)
	  bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(4,4)=(+,+,+,+,-,-)=helicity(4,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=4
	  bfun(8,ii,kk)=(4*dconjg(yup(9))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*(dconjg(yup(24))*(dconjg(yup(10))*yup(14)- 
     -     dconjg(yup(34))*yup(20) + yup(4)*yup(25)) -dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4)+dconjg(yup(14))*yup(14)-
     -     dconjg(yup(34))*yup(34)) + dconjg(yup(35))*dotup(idc1,idc1)
     -     *yup(35)))/xup(2)
c...helicity(20,4)=(-,-,-,-,+,+)
	  bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(5,4)=(+,+,+,-,+,+)=helicity(33,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=5
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*(2*dconjg(inpup(idk2,idk1))
     -     *dotup(idc1,idk1)*yup(27)*yup(28)*(yup(29) + yup(30)) + 
     -     dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0))*(yup(9)*yup(13)
     -     *yup(19)+(yup(7)*yup(13)+yup(3)*(-yup(12)+yup(18)))*yup(31))
     -     ) - dconjg(inpup(idb1,idq0))*(yup(9)*yup(17)*yup(19) + 
     -     (yup(7)*yup(17) +yup(6)*(-yup(12) + yup(18)))*yup(31)))
     -     + dconjg(inpup(idq0,idc2))*dconjg(yup(34))*yup(9)*(yup(12)
     -     - yup(18))*(yup(20) + yup(39))))/xup(1)
c...helicity(21,4)=(-,-,-,+,-,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(6,4)=(+,+,+,-,+,-)=helicity(35,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=6
	bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idb1)*(dotup(idc1,idc1)
     -     *(-(dconjg(inpup(idb2,idq0))*yup(13))-dconjg(inpup(idb1,idq0) 
     -     )*yup(17))*yup(19) -dconjg(inpup(idq0,idc2))*(2*
     -     dotup(idc1,idk1)*yup(14)*(yup(29)+yup(30))+(yup(12)-yup(18))*
     -     (yup(4)*(dconjg(yup(2))+yup(25))-dconjg(yup(34))*(yup(20) 
     -     + yup(39))))))/yup(8)
c...helicity(22,4)=(-,-,-,+,-,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(7,4)=(+,+,+,-,-,+)=helicity(34,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=7
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*((-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dotup(idc1,idc1)*(-(dconjg(inpup(idb2,idq0)
     -     )*yup(3)) -dconjg(inpup(idb1,idq0))*yup(6))*yup(31) - 
     -     dconjg(inpup(idq0,idc2))*(-((dconjg(yup(10))+dconjg(yup(16)))
     -     *(dconjg(yup(18))*yup(9)*yup(14)+2*dotup(idc1,idk1)*yup(4) 
     -     *yup(27))) +(-dconjg(yup(12)) + dconjg(yup(18)))*
     -     dconjg(yup(34))*yup(9)*(yup(20)+yup(39)))))/dconjg(yup(8))
c...helicity(23,4)=(-,-,-,+,+,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(8,4)=(+,+,+,-,-,-)=helicity(36,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1)
        kk=8
	  bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idc2))*dconjg(yup(9))*
     -    (dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -    ((dconjg(yup(10)) + dconjg(yup(16)))*yup(14) + 
     -    yup(4)*(dconjg(yup(2)) + yup(25)) -dconjg(yup(34)) 
     -    *(yup(20) + yup(39))))/xup(2)
c...helicity(24,4)=(-,-,-,+,+,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(9,4)=(+,-,-,+,+,+)=dconjg(helicity(12,1)(q_c1,q_c2,q_b1,q_b2,
c...k_2,k_1))
        kk=9
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(yup(4))*dconjg(yup(10))*dconjg(yup(12))* yup(14) -
     -     dconjg(yup(9))*dconjg(yup(10))*dconjg(yup(14))*yup(14) + 
     -     (dconjg(yup(4))*dconjg(yup(12)) - dconjg(yup(9))*
     -     dconjg(yup(14)))*dconjg(yup(16))*yup(14) +dconjg(yup(4)) 
     -     *dconjg(yup(12))*yup(4)*(dconjg(yup(2)) + yup(25)) + 
     -     dconjg(yup(9))*dconjg(yup(12))*yup(28)*yup(29) +
     -     dconjg(yup(9))*dconjg(yup(12))*yup(28)*yup(30) + 
     -     dconjg(yup(9))*dconjg(yup(12))*yup(27)*(yup(29)+yup(30))) + 
     -     4*dconjg(yup(18))*pmomup(5,idb1)*pmomup(5,idc1)*(
     -     dconjg(yup(4))*yup(4)*(dconjg(yup(2))+yup(25))+dconjg(yup(9))
     -     *(yup(27) + yup(28))*(yup(29) + yup(30))))/xup(2)
	  bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(25,4)=(-,+,+,-,-,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(10,4)=(+,-,-,+,+,-)=dconjg(helicity(10,1)(q_c1,q_c2,q_b1,q_b2,
c...k_2,k_1))
        kk=10
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (dconjg(yup(14))*(dconjg(yup(10))+dconjg(yup(16)))*yup(14)- 
     -     (dconjg(yup(12)) - dconjg(yup(18)))*(yup(29) + yup(30))*
     -     (yup(27) + yup(28) - yup(31))))/dconjg(yup(8))
	  bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(26,4)=(-,+,+,-,-,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(11,4)=(+,-,-,+,-,+)=dconjg(helicity(11,1)(q_c1,q_c2,q_b1,q_b2,
c...k_2,k_1))
        kk=11
	  bfun(8,ii,kk)=(-4*pmomup(5,idb1)*pmomup(5,idc1)*(dconjg(yup(4))
     -     *yup(4)*(yup(12) - yup(18))*(dconjg(yup(2)) + yup(25)) + 
     -     (dconjg(yup(9))*yup(18)*yup(19) +(2*dconjg(yup(8))* 
     -     dotup(idc1,idk1) +dconjg(yup(9))*(yup(12) - yup(18)))*
     -     (yup(27) + yup(28)))*(yup(29) + yup(30))))/yup(8)
        bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(27,4)=(-,+,+,-,+,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(12,4)=(+,-,-,+,-,-)=dconjg(helicity(9,1)(q_c1,q_c2,q_b1,q_b2,
c...k_2,k_1))
        kk=12
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(yup(29)+ yup(30))*(yup(19) - yup(27) 
     -     -yup(28) + yup(31)))/xup(1)
        bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(28,4)=(-,+,+,-,+,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(13,4)=(+,-,-,-,+,+)=-dconjg(helicity(36,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1))
        kk=13
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(dconjg(yup(9))*
     -     (dconjg(yup(10))*dconjg(yup(24)) -dconjg(yup(14))* 
     -     dotup(idb1,idb1))*inpup(idq0,idk2)*yup(14) +inpup(idq0,idk1)*
     -     (dotup(idb1,idb1)*(dconjg(yup(4))*(dconjg(yup(12)) -
     -     dconjg(yup(18)))*yup(4) + dconjg(yup(12))*dconjg(yup(14))
     -     *yup(14)) - dconjg(yup(24))*(dconjg(yup(10))*dconjg(yup(12))
     -     *yup(14)+(dconjg(yup(12))-dconjg(yup(18)))*yup(4)*yup(25)))+ 
     -     dconjg(yup(35))*(-2*dotup(idc1,idk1)*inpup(idk2,idk1)*yup(27)
     -     * yup(35) +dconjg(yup(9))*(dconjg(yup(12))-dconjg(yup(18)))*
     -     inpup(idq0,idb1)*(dconjg(yup(42)) + yup(36)))))/xup(2)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(29,4)=(-,+,+,+,-,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(14,4)=(+,-,-,-,+,-)=-dconjg(helicity(34,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1))
        kk=14
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*yup(9)*(inpup(idq0,idk2)*
     -     ((dconjg(yup(10))*dconjg(yup(24)) - dconjg(yup(14))
     -     *dotup(idb1,idb1))*yup(14)+2*dconjg(yup(35))*dotup(idc1,idk1)
     -     *yup(35))+(dconjg(yup(12))-dconjg(yup(18)))*dconjg(yup(35))*
     -     (-(dconjg(yup(38))*inpup(idq0,idb1)) +inpup(idq0,idb1) 
     -     *(dconjg(yup(42)) + yup(36)))))/dconjg(yup(8))
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(30,4)=(-,+,+,+,-,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(15,4)=(+,-,-,-,-,+)=-dconjg(helicity(35,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1))
        kk=15
	  bfun(8,ii,kk)=(-4*pmomup(5,idc1)*(inpup(idq0,idk1)*yup(4)
     -     *(yup(12) - yup(18))*(dconjg(yup(4))*dotup(idb1,idb1) - 
     -     dconjg(yup(24))*yup(25)) +2*dotup(idc1,idk1)*
     -     (-(inpup(idq0,idk1)*yup(16)*yup(27)*yup(29)) + 
     -     dconjg(yup(8))*dconjg(yup(35))*inpup(idq0,idb1)*yup(36)) + 
     -     dconjg(yup(9))*dconjg(yup(35))*(dconjg(yup(41))*
     -     inpup(idq0,idb1)*yup(18)+inpup(idq0,idb1)*(yup(12)-yup(18))*
     -     (dconjg(yup(42)) + yup(36)))))/yup(8)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(31,4)=(-,+,+,+,+,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(16,4)=(+,-,-,-,-,-)=-dconjg(helicity(33,1)(q_c1,q_c2,q_b1,q_b2,k_2,k_1))
        kk=16
	  bfun(8,ii,kk)=(-4*dconjg(yup(35))*pmomup(5,idc1)*yup(9)*
     -     (yup(12)-yup(18))*(-((dconjg(yup(38))+dconjg(yup(41)))*
     -     inpup(idq0,idb1)) + inpup(idq0,idb1)*(dconjg(yup(42)) +
     -     yup(36))))/xup(1)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(32,4)=(-,+,+,+,+,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

        if(ibcstate.eq.1) go to 1224

c...helicity(33,4)=(+,+,-,+,+,+)=-dconjg(helicity(16,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=33
        bfun(8,ii,kk)=(-4*dconjg(inpup(idc1,idq0))*dconjg(yup(9))*
     -    (dconjg(yup(12))-dconjg(yup(18)))*pmomup(5,idb1)*(yup(23)+
     -     yup(24))*yup(34))/xup(2)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(49,4)=(-,-,+,-,-,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(34,4)=(+,+,-,+,+,-)=-dconjg(helicity(14,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=34
        bfun(8,ii,kk)=(-4*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     pmomup(5,idb1)*yup(9)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk1))*dconjg(yup(4)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/dconjg(yup(8))
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(50,4)=(-,-,+,-,-,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(35,4)=(+,+,-,+,-,+)=-dconjg(helicity(15,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=35
        bfun(8,ii,kk)=(4*pmomup(5,idb1)*(yup(23) + yup(24))*
     -     (dconjg(inpup(idq0,idk2))*(dconjg(yup(9))*dconjg(yup(14))
     -     *yup(18) +2*dconjg(yup(4))*dotup(idc1,idk1)*yup(27)) + 
     -     dconjg(inpup(idc1,idq0))*dconjg(yup(9))*(-yup(12)+yup(18))
     -     *yup(34)))/yup(8)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(51,4)=(-,-,+,-,+,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(36,4)=(+,+,-,+,-,-)=-dconjg(helicity(13,1)(q_c1,q_c2,q_b1,q_b2)
        kk=36
        bfun(8,ii,kk)=(-4*pmomup(5,idb1)*yup(9)*(yup(12) - yup(18))*
     -     (yup(23)+yup(24))*(dconjg(inpup(idq0,idk1))*dconjg(yup(4))+ 
     -     dconjg(inpup(idq0,idk2))*dconjg(yup(14)) + 
     -     dconjg(inpup(idc1,idq0))*yup(34)))/xup(1)
	  bfun(8,ii,kk)=-dconjg(bfun(8,ii,kk))
c...helicity(52,4)=(-,-,+,-,+,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(37,4)=(+,+,-,-,+,+)=dconjg(helicity(40,1)(q_c1,q_c2,q_b1,q_b2)
        kk=37
	  bfun(8,ii,kk)=(-4*dotup(idc1,idc1)*yup(30)*(dconjg(yup(9))*
     -     dconjg(yup(10))*yup(19) + (dconjg(yup(7))*dconjg(yup(10)) + 
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*yup(25))*yup(31)) + 
     -     4*dconjg(yup(9))*(dconjg(yup(12)) - dconjg(yup(18)))*
     -     dconjg(yup(34))*dotup(idb1,idb1)*yup(34)+4*dconjg(yup(35))*
     -     (-2*dconjg(inpup(idb2,idq0))*dotup(idc1,idk1)*
     -     inpup(idk2,idk1)*yup(22)*yup(27) + dconjg(yup(9))*
     -     (-dconjg(yup(12)) + dconjg(yup(18)))*
     -     (dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) - dotup(idc1,idc1)*yup(35))))/xup(2)
        bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(53,4)=(-,-,+,+,-,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(38,4)=(+,+,-,-,+,-)=dconjg(helicity(38,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=38
        bfun(8,ii,kk)=(-4*yup(9)*((dconjg(yup(12)) - dconjg(yup(18)))*
     -     dotup(idb1,idb1)*(dconjg(yup(4))*yup(4) - dconjg(yup(34))
     -     *yup(34)) +dconjg(yup(35))*(-((dconjg(inpup(idb2,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18))) - 
     -     2*dconjg(inpup(idb2,idq0))*dconjg(yup(14))*dotup(idc1,idk1))*
     -     inpup(idc2,idb1)) +dconjg(inpup(idc1,idb2))*(dconjg(yup(12))
     -     -dconjg(yup(18)))*inpup(idc2,idb1)*yup(34))+dotup(idc1,idc1)*
     -     (dconjg(yup(10))*yup(19)*yup(30) + (-dconjg(yup(12)) +
     -     dconjg(yup(18)))*dconjg(yup(35))*yup(35))))/dconjg(yup(8))
        bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(54,4)=(-,-,+,+,-,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(39,4)=(+,+,-,-,-,+)=dconjg(helicity(39,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=39
        bfun(8,ii,kk)=(4*(2*dconjg(yup(4))*dotup(idc1,idk1)*
     -     (dotup(idb1,idb1)*yup(14) - yup(16)*yup(23))*yup(27) +
     -     dconjg(yup(9))*(-(dconjg(yup(14))*yup(16)*yup(18)*yup(23)) + 
     -     dconjg(inpup(idc1,idb2))*dconjg(yup(35))*inpup(idc2,idb1)*
     -     (-yup(12) + yup(18))*yup(34) + dotup(idb1,idb1)*
     -     (dconjg(yup(14))*yup(14)*yup(18) +dconjg(yup(34))*(yup(12) 
     -     - yup(18))*yup(34))) + dotup(idc1,idc1)*(2*dconjg(yup(8))
     -     *dconjg(yup(35))*dotup(idc1,idk1)*yup(35)+(yup(12)-yup(18))*
     -     (yup(25)*yup(30)*yup(31) + dconjg(yup(9))*dconjg(yup(35))
     -     *yup(35)))))/yup(8)
	  bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(55,4)=(-,-,+,+,+,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(40,4)=(+,+,-,-,-,-)=dconjg(helicity(37,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=40
        bfun(8,ii,kk)=(4*yup(9)*(yup(12) - yup(18))*
     -     (dconjg(yup(14))*yup(16)*yup(23) -dotup(idb1,idb1)*
     -     (dconjg(yup(4))*yup(4) + dconjg(yup(14))*yup(14) - 
     -     dconjg(yup(34))*yup(34)) +dconjg(yup(35))*
     -     (-(dconjg(inpup(idb2,idk1))*dconjg(yup(4))*inpup(idc2,idb1))- 
     -     dconjg(inpup(idc1,idb2))*inpup(idc2,idb1)*
     -     yup(34) + dotup(idc1,idc1)*yup(35))))/xup(1)
	  bfun(8,ii,kk)=dconjg(bfun(8,ii,kk))
c...helicity(56,4)=(-,-,+,+,+,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(41,4)=(+,-,+,+,+,+)=helicity(5,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=41
        bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk2))
     -     *dconjg(yup(14))*yup(9)*(dotup(idb1,idb1)*yup(14) - yup(16)
     -     *yup(23))) -dconjg(inpup(idq0,idk1))*(-(dconjg(yup(4))
     -     *dotup(idb1,idb1)*yup(4)*(yup(12)-yup(18)))-dconjg(yup(14)) 
     -     *yup(12)*(dotup(idb1,idb1)*yup(14) - yup(16)*yup(23)))-
     -     dconjg(yup(35))*((yup(12)-yup(18))*(dconjg(inpup(idb2,idk1))
     -     *yup(22)*yup(31)+dconjg(inpup(idb2,idq0))*yup(9)*(yup(22) +
     -     yup(32))) +2*dconjg(inpup(idk2,idk1))*dotup(idc1,idk1)*
     -     yup(27)*yup(35))))/xup(1)
c...helicity(57,4)=(-,+,-,-,-,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(42,4)=(+,-,+,+,+,-)=helicity(7,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=42
        bfun(8,ii,kk)=(4*dconjg(yup(9))*pmomup(5,idc1)*
     -     (-(dconjg(inpup(idq0,idk2))*dconjg(yup(14))*(dotup(idb1,idb1)
     -     *yup(14) - yup(16)*yup(23)))+ dconjg(inpup(idb2,idq0))
     -     *dconjg(yup(35))*(-2*dotup(idc1,idk1)*yup(17)+(yup(12)-
     -     yup(18))*(yup(21)- yup(22) - yup(32)))))/yup(8)
c...helicity(58,4)=(-,+,-,-,-,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(43,4)=(+,-,+,+,-,+)=helicity(6,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=43
        bfun(8,ii,kk)=(4*pmomup(5,idc1)*(-(dconjg(inpup(idq0,idk1))
     -     *dconjg(yup(4))*(-dconjg(yup(12)) + dconjg(yup(18)))*
     -     dotup(idb1,idb1)*yup(4)) - 2*dotup(idc1,idk1)*
     -     (dconjg(inpup(idb2,idq0))*dconjg(yup(35))*yup(8)*yup(22)+ 
     -     dconjg(inpup(idq0,idk1))*dconjg(yup(10))*yup(27)*yup(30)) - 
     -     dconjg(yup(35))*(dconjg(inpup(idb2,idk1))*(dconjg(yup(12))-
     -     dconjg(yup(18)))*yup(22)*yup(31) + dconjg(inpup(idb2,idq0))
     -     *yup(9)*(dconjg(yup(12))*(yup(22)+yup(32))-dconjg(yup(18))*
     -     (yup(22) - yup(26) + yup(32))))))/dconjg(yup(8))
c...helicity(59,4)=(-,+,-,-,+,-)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(44,4)=(+,-,+,+,-,-)=helicity(8,1)(q_c1,q_c2,q_b1,q_b2)
        kk=44
        bfun(8,ii,kk)=(4*dconjg(inpup(idb2,idq0))*dconjg(yup(9))*
     -     (dconjg(yup(12)) - dconjg(yup(18)))*dconjg(yup(35))
     -     *pmomup(5,idc1)*(yup(21)-yup(22)+yup(26)-yup(32)))/xup(2)
c...helicity(60,4)=(-,+,-,-,+,+)
        bfun(8,ii,kk+16)=-dconjg(bfun(8,ii,kk))

c...helicity(45,4)=(+,-,+,-,+,+)=helicity(45,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=45
	  bfun(8,ii,kk)=(4*pmomup(5,idb1)*pmomup(5,idc1)*
     -     (dconjg(inpup(idq0,idk2))*dconjg(yup(14))*
     -     (-(dconjg(inpup(idq0,idk2))*yup(9)) + 
     -     dconjg(inpup(idq0,idk1))*yup(12)) + 
     -     dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))*
     -     (yup(12) - yup(18)))*(yup(23) + yup(24)))/xup(1)
c...helicity(61,4)=(-,+,-,+,-,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(46,4)=(+,-,+,-,+,-)=helicity(47,1)(q_c1,q_c2,q_b1,q_b2)
        kk=46
        bfun(8,ii,kk)=(-4*dconjg(inpup(idq0,idk2))**2*dconjg(yup(9))
     -     *dconjg(yup(14))*pmomup(5,idb1)*pmomup(5,idc1)*(yup(23) 
     -     + yup(24)))/yup(8)
c...helicity(62,4)=(-,+,-,+,-,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(47,4)=(+,-,+,-,-,+)=helicity(46)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=47
        bfun(8,ii,kk)=(4*dconjg(inpup(idq0,idk1))**2*dconjg(yup(4))
     -     *(dconjg(yup(12)) - dconjg(yup(18)))*pmomup(5,idb1)*
     -     pmomup(5,idc1)*(yup(23) + yup(24)))/dconjg(yup(8))
c...helicity(63,4)=(-,+,-,+,+,-)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

c...helicity(48,4)=(+,-,+,-,-,-)=helicity(48,1)(q_c1,q_c2,q_b1,q_b2,k2,k1)
        kk=48
	  bfun(8,ii,kk)=0.0d0
c...helicity(64,4)=(-,+,-,+,+,+)
        bfun(8,ii,kk+16)=dconjg(bfun(8,ii,kk))

1224  continue

c...relations from diagrams involving three gluon vertex.
       do kk=1,64
	    bfun(7,1,kk)=-bfun(4,1,kk)+bfun(2,1,kk)+bfun(8,1,kk)-
     &		2*dotup(7,1)*(2*bfun(5,1,kk)-2*bfun(5,2,kk)+bfun(9,1,kk))
	    bfun(7,2,kk)=-bfun(4,2,kk)+bfun(2,2,kk)+bfun(8,2,kk)-
     &		2*dotup(7,2)*(2*bfun(5,2,kk)-2*bfun(5,1,kk)+bfun(9,2,kk))
		bfun(7,3,kk)=-bfun(4,3,kk)+bfun(2,3,kk)+bfun(8,3,kk)-
     &		2*dotup(4,1)*(2*bfun(5,3,kk)-2*bfun(5,4,kk)+bfun(9,3,kk))
		bfun(7,4,kk)=-bfun(4,4,kk)+bfun(2,4,kk)+bfun(8,4,kk)-
     &		2*dotup(4,2)*(2*bfun(5,4,kk)-2*bfun(5,3,kk)+bfun(9,4,kk))
	    bfun(6,1,kk)=bfun(7,1,kk)+2*dotup(5,2)*bfun(9,1,kk)-
     &		bfun(3,2,kk)+bfun(1,2,kk)
	    bfun(6,2,kk)=bfun(7,2,kk)+2*dotup(5,1)*bfun(9,2,kk)-
     &		bfun(3,1,kk)+bfun(1,1,kk)
		bfun(6,3,kk)=bfun(7,3,kk)+2*dotup(6,2)*bfun(9,3,kk)-
     &		bfun(3,4,kk)+bfun(1,4,kk)
		bfun(6,4,kk)=bfun(7,4,kk)+2*dotup(6,1)*bfun(9,4,kk)-
     &		bfun(3,3,kk)+bfun(1,3,kk)
	  end do

	end
