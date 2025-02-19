c**********************************************************************

c...helicity amplitude of the free quark part: g+g->\bar{b}+c+b+\bar{c}.

	subroutine freehelicity
	implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex bfun,colmat,inpup,sppup,yup,xup,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/tryup/bfun(9,4,64),yup(100),xup(2),idp,idq0,idk1,idk2,
     &	idb1,idb2,idc1,idc2
c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode
c...whether getting the color-octet component contributions. here only
c...for gg->bc+b+~c (bc in color-octet s-wave states.).
      common/coloct/ioctet

      dimension propup(14,4)

c...the four subroutine are used to get bfun(). first->original from cb and
c...cc; second->inter-change k_1 and k_2; third->inter-change q_b1,q_b2 and 
c...q_c1,q_c2; four->inter-change both. here, bfun() stands for the basic
c...feynman diagrams which is free from color factors and proporgators.
c...the results are get from mathmatical programs: helicitybasic.nb
c...and texbasic.nb. here for easy programming, we use the number idk1,idk2,
c...idb1,idb2,idc1,idc2,idq0,idp to identify the particles correspondingly. 
    
      idq0=8
	idp=3

      do 700, ii=1,4
	 if (ii .eq. 1) then
	  idk1=1
	  idk2=2
	  idb1=4
	  idb2=6
	  idc1=7
	  idc2=5
	 end if

	 if (ii .eq. 2) then
	  idk1=2
	  idk2=1
	  idb1=4
	  idb2=6
	  idc1=7
	  idc2=5
       end if

       if (ii .eq. 3) then
	  idk1=1
	  idk2=2
	  idb1=7
	  idb2=5
	  idc1=4
	  idc2=6
       end if

       if (ii .eq. 4) then
	  idk1=2
	  idk2=1
	  idb1=7
	  idb2=5
	  idc1=4
	  idc2=6
       end if

c...define some basic spinnor products that are needed in helicity amplitude.
	 yup(1)=sppup(idk1,idc2,idq0)
	 yup(2)=sppup(idk1,idb2,idq0)
	 yup(3)=sppup(idk1,idq0,idb2)
	 yup(4)=sppup(idk1,idq0,idc2)
	 yup(5)=sppup(idk1,idq0,idc1)
	 yup(6)=sppup(idk1,idq0,idb1)
	 yup(7)=sppup(idk1,idk2,idq0)
	 yup(8)=sppup(idk1,idq0,idk2)
	 yup(9)=sppup(idk1,idc1,idq0)

	 yup(10)=sppup(idk2,idb1,idq0)
	 yup(11)=sppup(idk2,idc2,idq0)
	 yup(12)=sppup(idk2,idk1,idq0)
	 yup(13)=sppup(idk2,idq0,idb2)
	 yup(14)=sppup(idk2,idq0,idc2)
	 yup(15)=sppup(idk2,idq0,idc1)
	 yup(16)=sppup(idk2,idb2,idq0)
	 yup(17)=sppup(idk2,idq0,idb1)
	 yup(18)=sppup(idk2,idc1,idq0)

	 yup(19)=sppup(idq0,idk2,idq0)
	 yup(20)=sppup(idq0,idb1,idc1)
	 yup(21)=sppup(idq0,idk1,idb1)
	 yup(22)=sppup(idq0,idc2,idb1)
	 yup(23)=sppup(idq0,idb1,idc2)
	 yup(24)=sppup(idq0,idb2,idc2)
	 yup(25)=sppup(idq0,idb1,idk1)
	 yup(26)=sppup(idq0,idk2,idb1)
	 yup(27)=sppup(idq0,idc1,idq0)
	 yup(28)=sppup(idq0,idc2,idq0)
	 yup(29)=sppup(idq0,idb1,idq0)
	 yup(30)=sppup(idq0,idb2,idq0)
	 yup(31)=sppup(idq0,idk1,idq0)
	 yup(32)=sppup(idq0,idc1,idb1)
	 yup(33)=sppup(idq0,idk2,idc2)

	 yup(34)=sppup(idc2,idq0,idc1)
	 yup(35)=sppup(idb2,idq0,idb1)
	 yup(36)=sppup(idb2,idc2,idq0)
	 yup(37)=sppup(idb1,idq0,idc2)
	 yup(38)=sppup(idq0,idk1,idb2)
	 yup(39)=sppup(idq0,idb2,idc1)
	 yup(40)=sppup(idq0,idk2,idc1)
	 yup(41)=sppup(idq0,idk2,idb2)
	 yup(42)=sppup(idq0,idc1,idb2)
	 xup(1)=inpup(idq0,idk1)*inpup(idq0,idk2)
	 xup(2)=dconjg(xup(1))
       
	 if(iqqbar.eq.1) then
        yup(43)=sppup(idb1,idq0,idc1)
	  yup(44)=sppup(idc2,idb1,idk2)
        yup(45)=sppup(idc2,idk1,idk2)
        yup(46)=sppup(idk1,idb1,idk2)
	  yup(47)=sppup(idk1,idk2,idb1)
        yup(48)=sppup(idb2,idk1,idk2)
	  yup(49)=sppup(idk1,idb2,idc1)
	  yup(50)=sppup(idk1,idb2,idk2)

	  yup(51)=sppup(idb2,idk1,idb1)
        yup(52)=sppup(idc2,idk2,idb1)
	  yup(53)=sppup(idk2,idb1,idk1)
	  yup(54)=sppup(idk2,idk1,idc1)
	  yup(55)=sppup(idc2,idb1,idk1)
	  yup(56)=sppup(idc2,idk2,idk1)
	  yup(57)=sppup(idk2,idk1,idb1)
	  yup(58)=sppup(idk2,idb2,idc1)
	  yup(59)=sppup(idb2,idk2,idk1)
	  yup(60)=sppup(idk2,idb2,idk1)

	  yup(61)=sppup(idb2,idb1,idq0)
	  yup(62)=sppup(idb2,idb1,idc1)
	  yup(63)=sppup(idb2,idk2,idc1)
	  yup(64)=sppup(idb2,idb1,idc2)
	  yup(65)=sppup(idc1,idq0,idb1)
	  yup(66)=sppup(idk1,idk2,idb2)
	  yup(67)=sppup(idc1,idb2,idk2)
	  yup(68)=sppup(idc2,idb2,idk2)
	  yup(69)=sppup(idb2,idq0,idc1)
	  yup(70)=sppup(idc1,idb1,idb2)

	  yup(71)=sppup(idb2,idq0,idc2)
	  yup(72)=sppup(idc1,idk2,idk1)
	  yup(73)=sppup(idc2,idb2,idk1)
	  yup(74)=sppup(idc1,idb1,idk2)
	  yup(75)=sppup(idq0,idb2,idb1)
	  yup(76)=sppup(idc1,idk2,idb1)
	  yup(77)=sppup(idb2,idk2,idc2)
	  yup(78)=sppup(idc2,idk2,idb2)
	  yup(79)=sppup(idq0,idk1,idc1)
	  yup(80)=sppup(idq0,idk1,idc2)

	  yup(81)=sppup(idc1,idb1,idk1)
	  yup(82)=sppup(idk1,idb2,idb1)
	  yup(83)=sppup(idc2,idb2,idb1)
	  yup(84)=sppup(idc1,idb2,idb1)
	  yup(85)=sppup(idc2,idb1,idb2)
	  yup(86)=sppup(idk1,idb2,idc2)
	 end if

       if(iqqbar.eq.0) then
	  if (ii.eq.1) call bfirst
	  if (ii.eq.2) call bsecond
	  if (ii.eq.3) call bthird
	  if (ii.eq.4) call bfourth
	 end if
	 
	 if(iqqbar.eq.1) then
	  if (ii.eq.1) then
	    do kk=1,64  
		  bfun(1,ii,kk)=0.0d0
            bfun(2,ii,kk)=0.0d0
	      bfun(3,ii,kk)=0.0d0
	      bfun(4,ii,kk)=0.0d0
	    end do
		call bfirst
	  end if
	  if (ii.eq.3) then
	    do kk=1,64  
		  bfun(1,ii,kk)=0.0d0
            bfun(2,ii,kk)=0.0d0
	      bfun(3,ii,kk)=0.0d0
	      bfun(4,ii,kk)=0.0d0
	    end do
		call bthird
	  end if
	 end if

c...the needed basic propagators. propup(n,1) from cb and cc sets. other
c...propagators can be obtained from them by interchange gluons and quarks,
c...which can be achieved by interchanging the identity number of the 
c...particles (i.e. idk1,idk2,...). here the correspondence of the diagram 
c...labels, mcb1,mcb2,..., to the original feynman diagrams are shown in paper.
       if(iqqbar.eq.0) then
c...propup(1,1)<-mcb1,propup(2,1)<-mcb2,propup(3,1)<-mcb3,propup(4,1)<-mcb4
       if (ii.eq.1 .or. ii.eq.2) then
         propup(1,ii)=1/(-2*dotup(idb1,idk2))/(2*dotup(idc1,idc1)
     &	 +2*dotup(idc2,idc1)-2*dotup(idk1,idc2)-2*dotup(idk1,idc1))
     &	 /(-2*dotup(idk1,idc2))
	   propup(2,ii)=1/(-2*dotup(idb1,idk2))/(2*dotup(idc1,idc1)
     &	 +2*dotup(idc2,idc1)-2*dotup(idk1,idc2)-2*dotup(idk1,idc1))
     &	 /(-2*dotup(idc1,idk1))
	   propup(3,ii)=1/(-2*dotup(idb2,idk2))/(2*dotup(idc1,idc1)
     &	 +2*dotup(idc2,idc1)-2*dotup(idk1,idc2)-
     &     2*dotup(idk1,idc1))/(-2*dotup(idc2,idk1))
	   propup(4,ii)=1/(-2*dotup(idb2,idk2))/(2*dotup(idc1,idc1)
     &	 +2*dotup(idc2,idc1)-2*dotup(idk1,idc2)-
     &	 2*dotup(idk1,idc1))/(-2*dotup(idc1,idk1))
	 end if

c...propup(5,1)->mcc1,propup(6,1)->mcc3,propup(7,1)->mcc5
	 propup(5,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(-2*dotup(idc2,idk2))/(2*dotup(idk1,idk2)-
     &	 2*dotup(idk1,idc2)-2*dotup(idc2,idk2))
	 propup(6,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(-2*dotup(idc2,idk2))/(-2*dotup(idc1,idk1))
	 propup(7,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(-2*dotup(idc1,idk1))/(2*dotup(idk1,idk2)
     &     -2*dotup(idc1,idk1)-2*dotup(idc1,idk2))

c...propup(8,1)->mcc7, propup(9,1)->mcc8
	 if (ii.eq.1 .or. ii.eq.3) then
	   propup(8,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(2*dotup(idk1,idk2))/(2*dotup(idk1,idk2)
     &     -2*dotup(idk1,idc2)-2*dotup(idc2,idk2))
	   propup(9,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(2*dotup(idk1,idk2))/(2*dotup(idk1,idk2)
     &     -2*dotup(idc1,idk1)-2*dotup(idc1,idk2))
	 end if

c...propup(10,1)->mco1, propup(11,1)->mco2
	 propup(10,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(-2*dotup(idc2,idk1))/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2)
     &     -2*dotup(idb1,idk2)-2*dotup(idb2,idk2))
	 propup(11,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
     &	 /(-2*dotup(idc1,idk1))/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2)
     &	 -2*dotup(idb1,idk2)-2*dotup(idb2,idk2))

c...propup(12,1)->moo1
	 if (ii.eq.1 .or. ii.eq.2) then
	   propup(12,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))/
     &	 (2*dotup(idc1,idc1)+2*dotup(idc2,idc1))/(2*dotup(idb1,idb1)
     &     +2*dotup(idb1,idb2)-2*dotup(idb1,idk1)-2*dotup(idb2,idk1))
	 end if

c...propup(13,1)->moo3, propup(14,1)->moo4
	 if (ii.eq.1) then
	   propup(13,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))/
     &	 (2*dotup(idk1,idk2))/(2*dotup(idc1,idc1)+2*dotup(idc2,idc1))
	   propup(14,ii)=1/(2*dotup(idb1,idb1)+2*dotup(idb1,idb2))/
     &	 (2*dotup(idc1,idc1)+2*dotup(idc2,idc1))
	 end if
	 end if

       if(iqqbar.eq.1) then
	   if (ii.eq.1 .or. ii.eq.3) then
	     propup(1,ii)=1/(2*dotup(idk1,idk2))/(2*dotup(idk1,idk2)-
     &		 2*dotup(idk1,idb1)-2*dotup(idk2,idb1))/
     &         (2*dotup(idc1,idc1)+2*dotup(idc1,idc2))
		 propup(2,ii)=1/(2.*dotup(idk1,idk2))/(2*dotup(idk1,idk2)-
     &		 2*dotup(idk1,idb2)-2*dotup(idk2,idb2))/
     &         (2*dotup(idc1,idc1)+2*dotup(idc1,idc2))
		 propup(4,ii)=1/(-2*dotup(idk1,idb1)-2*dotup(idk1,idb2)+
     &		 2*dotup(idb1,idb1)+2*dotup(idb1,idb2))/
     &         (2*dotup(idc1,idc1)+2*dotup(idc1,idc2))/
     &         (2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
	     if(ii.eq.1) then
		   propup(3,ii)=1/(2*dotup(idk1,idk2))/
     &         (2*dotup(idc1,idc1)+2*dotup(idc1,idc2))/
     &         (2*dotup(idb1,idb1)+2*dotup(idb1,idb2))
	     end if
	   end if
	 end if
700   continue
      	
c...five collectted matrix elements according to the color factor c_1ij, 
c...c_2ij, c_3ij, c_4ij and c_5ij. note here the dsqrt part in the denominator
c...is needed, for all the quarks are massive. the following result is get
c...from the mathmatical program: matrixtotal.nb. here kk=1-64 stands for the 
c...helicity of the amplitude.

      if(iqqbar.eq.0) then
c0--------
      if(ioctet.eq.0) then
	 do 900, kk=1,64
	 colmat(1,kk)=(-((bfun(6,1,kk) + (bfun(9,1,kk) - bfun(9,2,kk))*
     -   (dotup(1,2) - dotup(5,1) - dotup(5,2)))*propup(8,1)) - 
     -   bfun(8,3,kk)*propup(9,3) - (4*bfun(5,3,kk) - 4*bfun(5,4,kk) +
     -   bfun(9,3,kk)-bfun(9,4,kk))*(dotup(1,2)-dotup(4,1)-dotup(4,2))*
     -   propup(9,3) + bfun(8,4,kk)*(propup(7,4) + propup(9,3)) + 
     -   2*bfun(5,1,kk)*dotup(5,1)*propup(10,1) +(-bfun(2,4,kk) 
     -   + bfun(4,4,kk))*propup(11,4) + 2*bfun(5,4,kk)*dotup(4,2)
     -   *propup(11,4) +(-bfun(2,1,kk) + bfun(4,1,kk))*propup(12,2) + 
     -   2*bfun(5,1,kk)*dotup(4,4)*propup(12,2)+2*bfun(5,1,kk)*
     -   dotup(4,6) *propup(12,2) + 2*bfun(5,1,kk)*dotup(5,1)*
     -   propup(12,2) +2*bfun(5,1,kk)*dotup(7,1)*propup(12,2) - 
     -   bfun(3,1,kk)*(propup(10,1) + propup(12,2)) +bfun(1,1,kk)* 
     -   (propup(1,1) + propup(10,1) + propup(12,2)) - (bfun(6,1,kk)
     -   - bfun(8,1,kk) + bfun(8,2,kk))*propup(13,1) +4*bfun(5,1,kk) 
     -   *dotup(1,2)*propup(13,1)-4*bfun(5,2,kk)*dotup(1,2)*propup(13,1) 
     -   +bfun(9,1,kk)*dotup(5,1)*propup(13,1) -bfun(9,2,kk)*dotup(5,1) 
     -   *propup(13,1) + bfun(9,1,kk)*dotup(5,2)*propup(13,1) - 
     -   bfun(9,2,kk)*dotup(5,2)*propup(13,1) - 4*bfun(5,1,kk)*
     -   dotup(7,1)*propup(13,1)+4*bfun(5,2,kk)*dotup(7,1)*propup(13,1)- 
     -   bfun(9,1,kk)*dotup(7,1)*propup(13,1) + bfun(9,2,kk)*
     -   dotup(7,1)*propup(13,1) - 4*bfun(5,1,kk)*dotup(7,2)*
     -   propup(13,1) + 4*bfun(5,2,kk)*dotup(7,2)*propup(13,1) - 
     -   bfun(9,1,kk)*dotup(7,2)*propup(13,1) + bfun(9,2,kk)
     -   *dotup(7,2)*propup(13,1) + bfun(6,2,kk)*(propup(5,2)
     -   + propup(8,1) + propup(13,1)) - ((2*bfun(5,1,kk) -
     -   4*bfun(5,2,kk)+bfun(9,1,kk)+bfun(9,2,kk))*propup(14,1))/2.)
     -   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))

	 colmat(2,kk)=((bfun(9,1,kk) - bfun(9,2,kk))*
     -   (dotup(1,2) - dotup(5,1) - dotup(5,2))*propup(8,1) - 
     -   bfun(8,4,kk)*propup(9,3) + 4*bfun(5,3,kk)*dotup(1,2)*
     -   propup(9,3) -4*bfun(5,4,kk)*dotup(1,2)*propup(9,3) + 
     -   bfun(9,3,kk)*dotup(1,2)*propup(9,3)-bfun(9,4,kk)*dotup(1,2) 
     -   *propup(9,3) -4*bfun(5,3,kk)*dotup(4,1)*propup(9,3) + 
     -   4*bfun(5,4,kk)*dotup(4,1)*propup(9,3) - bfun(9,3,kk)
     -   *dotup(4,1)*propup(9,3) + bfun(9,4,kk)*dotup(4,1)
     -   *propup(9,3) -4*bfun(5,3,kk)*dotup(4,2)*propup(9,3) + 
     -   4*bfun(5,4,kk)*dotup(4,2)*propup(9,3)-bfun(9,3,kk)*dotup(4,2) 
     -   *propup(9,3) +bfun(9,4,kk)*dotup(4,2)*propup(9,3) + 
     -   bfun(8,3,kk)*(propup(7,3) + propup(9,3)) +2*bfun(5,2,kk) 
     -   *dotup(5,2)*propup(10,2) +(-bfun(2,3,kk) + bfun(4,3,kk) 
     -   )*propup(11,3) +2*bfun(5,3,kk)*dotup(4,1)*propup(11,3) + 
     -   (-bfun(2,2,kk) + bfun(4,2,kk))*propup(12,1) + 
     -   2*bfun(5,2,kk)*dotup(4,4)*propup(12,1) +2*bfun(5,2,kk) 
     -   *dotup(4,6)*propup(12,1) + 2*bfun(5,2,kk)*dotup(5,2)
     -   *propup(12,1) +2*bfun(5,2,kk)*dotup(7,2)*propup(12,1) - 
     -   bfun(3,2,kk)*(propup(10,2) + propup(12,1)) + 
     -   bfun(1,2,kk)*(propup(1,2) + propup(10,2) + propup(12,1)) + 
     -   (-bfun(8,1,kk) + bfun(8,2,kk))*propup(13,1) - 
     -   4*bfun(5,1,kk)*dotup(1,2)*propup(13,1) +4*bfun(5,2,kk)* 
     -   dotup(1,2)*propup(13,1)-bfun(9,1,kk)*dotup(5,1)*propup(13,1) + 
     -   bfun(9,2,kk)*dotup(5,1)*propup(13,1) - bfun(9,1,kk)
     -   *dotup(5,2)*propup(13,1)+bfun(9,2,kk)*dotup(5,2)*propup(13,1)+ 
     -   4*bfun(5,1,kk)*dotup(7,1)*propup(13,1) - 4*bfun(5,2,kk)
     -   *dotup(7,1)*propup(13,1) + bfun(9,1,kk)*dotup(7,1)
     -   *propup(13,1) - bfun(9,2,kk)*dotup(7,1)*propup(13,1) + 
     -   4*bfun(5,1,kk)*dotup(7,2)*propup(13,1) - 4*bfun(5,2,kk)
     -   *dotup(7,2)*propup(13,1) + bfun(9,1,kk)*dotup(7,2)
     -   *propup(13,1) - bfun(9,2,kk)*dotup(7,2)*propup(13,1) - 
     -   bfun(6,2,kk)*(propup(8,1) + propup(13,1)) + 
     -   bfun(6,1,kk)*(propup(5,1) + propup(8,1) + propup(13,1)) - 
     -   ((-4*bfun(5,1,kk) + 2*bfun(5,2,kk) + bfun(9,1,kk) + 
     -   bfun(9,2,kk))*propup(14,1))/2.) 
     -   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	
	 colmat(3,kk)=(bfun(4,1,kk)*propup(4,1)+bfun(7,2,kk)*propup(6,2)+ 
     -   bfun(7,4,kk)*propup(6,4) - bfun(6,3,kk)*propup(8,3) - 
     -   (bfun(9,3,kk)-bfun(9,4,kk))*(dotup(1,2)-dotup(6,1)-dotup(6,2))
     -   *propup(8,3) +bfun(6,4,kk)*(propup(5,4) + propup(8,3)) - 
     -   (4*bfun(5,1,kk) - 4*bfun(5,2,kk) + bfun(9,1,kk) - 
     -   bfun(9,2,kk))*(dotup(1,2) - dotup(7,1) - dotup(7,2))*
     -   propup(9,1) - 2*bfun(5,1,kk)*dotup(5,1)*propup(10,1) + 
     -   (bfun(2,4,kk) - bfun(4,4,kk) - 2*bfun(5,4,kk)*dotup(4,2))*
     -   propup(11,4) - bfun(4,1,kk)*propup(12,2) - 
     -   2*bfun(5,1,kk)*dotup(4,4)*propup(12,2) - 
     -   2*bfun(5,1,kk)*dotup(4,6)*propup(12,2) - 
     -   2*bfun(5,1,kk)*dotup(5,1)*propup(12,2) - 
     -   2*bfun(5,1,kk)*dotup(7,1)*propup(12,2) + 
     -   bfun(2,1,kk)*(propup(2,1) + propup(12,2)) - 
     -   bfun(1,1,kk)*(propup(10,1) + propup(12,2)) + 
     -   bfun(3,1,kk)*(propup(3,1)+propup(10,1)+propup(12,2))+ 
     -   (bfun(6,1,kk) - bfun(6,2,kk))*propup(13,1) - 
     -   4*bfun(5,1,kk)*dotup(1,2)*propup(13,1) + 
     -   4*bfun(5,2,kk)*dotup(1,2)*propup(13,1) - 
     -   bfun(9,1,kk)*dotup(5,1)*propup(13,1) + 
     -   bfun(9,2,kk)*dotup(5,1)*propup(13,1) - 
     -   bfun(9,1,kk)*dotup(5,2)*propup(13,1) + 
     -   bfun(9,2,kk)*dotup(5,2)*propup(13,1) + 
     -   4*bfun(5,1,kk)*dotup(7,1)*propup(13,1) - 
     -   4*bfun(5,2,kk)*dotup(7,1)*propup(13,1) + 
     -   bfun(9,1,kk)*dotup(7,1)*propup(13,1) - 
     -   bfun(9,2,kk)*dotup(7,1)*propup(13,1) + 
     -   4*bfun(5,1,kk)*dotup(7,2)*propup(13,1) - 
     -   4*bfun(5,2,kk)*dotup(7,2)*propup(13,1) + 
     -   bfun(9,1,kk)*dotup(7,2)*propup(13,1) - 
     -   bfun(9,2,kk)*dotup(7,2)*propup(13,1) - 
     -   bfun(8,1,kk)*(propup(9,1) + propup(13,1)) + 
     -   bfun(8,2,kk)*(propup(7,2) + propup(9,1) + propup(13,1)) + 
     -   ((2*bfun(5,1,kk) - 4*bfun(5,2,kk) + bfun(9,1,kk) + 
     -   bfun(9,2,kk))*propup(14,1))/2.)
     -   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	
	 colmat(4,kk)=(bfun(4,2,kk)*propup(4,2)+bfun(7,1,kk)*propup(6,1)+ 
     -   bfun(7,3,kk)*propup(6,3) - (bfun(6,4,kk) - (bfun(9,3,kk) 
     -   -bfun(9,4,kk))*(dotup(1,2)-dotup(6,1)-dotup(6,2)))*propup(8,3)+ 
     -   bfun(6,3,kk)*(propup(5,3) + propup(8,3)) - (bfun(8,2,kk)
     -   -(4*bfun(5,1,kk)-4*bfun(5,2,kk)+bfun(9,1,kk) -bfun(9,2,kk))*
     -   (dotup(1,2) - dotup(7,1) - dotup(7,2)))*propup(9,1) - 
     -   2*bfun(5,2,kk)*dotup(5,2)*propup(10,2) +bfun(3,2,kk) 
     -   *(propup(3,2) + propup(10,2)) + (bfun(2,3,kk) - 
     -   bfun(4,3,kk) - 2*bfun(5,3,kk)*dotup(4,1))*propup(11,3) + 
     -   bfun(3,2,kk)*propup(12,1) -bfun(4,2,kk)*propup(12,1) - 
     -   2*bfun(5,2,kk)*dotup(4,4)*propup(12,1) - 
     -   2*bfun(5,2,kk)*dotup(4,6)*propup(12,1) - 
     -   2*bfun(5,2,kk)*dotup(5,2)*propup(12,1) - 
     -   2*bfun(5,2,kk)*dotup(7,2)*propup(12,1) + 
     -   bfun(2,2,kk)*(propup(2,2) + propup(12,1)) - 
     -   bfun(1,2,kk)*(propup(10,2) + propup(12,1)) + 
     -   (-bfun(6,1,kk) + bfun(6,2,kk))*propup(13,1) - 
     -   4*bfun(5,2,kk)*dotup(1,2)*propup(13,1) - 
     -   (bfun(8,2,kk)-4*bfun(5,1,kk)*dotup(1,2))*propup(13,1)+ 
     -   bfun(9,1,kk)*dotup(5,1)*propup(13,1) - 
     -   bfun(9,2,kk)*dotup(5,1)*propup(13,1) + 
     -   bfun(9,1,kk)*dotup(5,2)*propup(13,1) - 
     -   bfun(9,2,kk)*dotup(5,2)*propup(13,1) - 
     -   4*bfun(5,1,kk)*dotup(7,1)*propup(13,1) + 
     -   4*bfun(5,2,kk)*dotup(7,1)*propup(13,1) - 
     -   bfun(9,1,kk)*dotup(7,1)*propup(13,1) + 
     -   bfun(9,2,kk)*dotup(7,1)*propup(13,1) - 
     -   4*bfun(5,1,kk)*dotup(7,2)*propup(13,1) + 
     -   4*bfun(5,2,kk)*dotup(7,2)*propup(13,1) - 
     -   bfun(9,1,kk)*dotup(7,2)*propup(13,1) + 
     -   bfun(9,2,kk)*dotup(7,2)*propup(13,1) + 
     -   bfun(8,1,kk)*(propup(7,1) + propup(9,1) + propup(13,1)) + 
     -   ((-4*bfun(5,1,kk) + 2*bfun(5,2,kk) + bfun(9,1,kk) + 
     -   bfun(9,2,kk))*propup(14,1))/2.)
     &   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	
	 colmat(5,kk)=(-(bfun(6,3,kk)*propup(5,3))-bfun(6,4,kk)*
     -   propup(5,4)-bfun(8,1,kk)*propup(7,1)-bfun(8,2,kk)*propup(7,2)+ 
     -   (-bfun(1,3,kk) + bfun(3,3,kk) - 2*bfun(5,3,kk)*dotup(6,1))*
     -   propup(10,3) + (-bfun(1,4,kk) + bfun(3,4,kk) - 
     -   2*bfun(5,4,kk)*dotup(6,2))*propup(10,4) + (bfun(2,2,kk) -
     -   2*bfun(5,2,kk)*dotup(7,2))*propup(11,2) -bfun(1,2,kk)* 
     -   propup(12,1) +(bfun(2,2,kk) + bfun(3,2,kk) - 2*bfun(5,2,kk)
     -   *(dotup(4,4) + dotup(4,6) + dotup(5,2) +dotup(7,2))) 
     -   *propup(12,1) - bfun(4,2,kk)*(propup(4,2) +propup(11,2) +
     -   propup(12,1)) +(-bfun(1,1,kk) + bfun(3,1,kk))*propup(12,2)+ 
     -   bfun(2,1,kk)*(propup(11,1) + propup(12,2)) -bfun(4,1,kk)* 
     -   (propup(4,1) + propup(11,1) + propup(12,2)) +(-bfun(5,2,kk) 
     -   + bfun(9,1,kk) + bfun(9,2,kk))*propup(14,1) - bfun(5,1,kk)*
     -   (2*dotup(7,1)*propup(11,1) +2*(dotup(4,4) + dotup(4,6) + 
     -   dotup(5,1) + dotup(7,1))*propup(12,2) + propup(14,1)))
     &   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
900    continue
       end if
c0---------
c8---------
	 if(ioctet.eq.1) then
	  do kk=1,64
	    colmat(1,kk)=(-(bfun(7,1,kk)*propup(6,1)) + 
     -      (bfun(6,2,kk) - (bfun(9,1,kk) - bfun(9,2,kk))*
     -      (dotup(1,2) - dotup(5,1) - dotup(5,2)))*propup(8,1) - 
     -      bfun(6,1,kk)*(propup(5,1)+propup(8,1))+(bfun(8,2,kk)-
     -      (4*bfun(5,1,kk)-4*bfun(5,2,kk)+bfun(9,1,kk)-bfun(9,2,kk))*
     -      (dotup(1,2) - dotup(7,1) - dotup(7,2)))*propup(9,1) - 
     -      bfun(8,1,kk)*(propup(7,1) + propup(9,1)))
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
		colmat(2,kk)=(-(bfun(7,2,kk)*propup(6,2)) + 
     -      (bfun(6,1,kk) + (bfun(9,1,kk) - bfun(9,2,kk))*
     -      (dotup(1,2) - dotup(5,1) - dotup(5,2)))*propup(8,1) - 
     -      bfun(6,2,kk)*(propup(5,2)+propup(8,1))+(bfun(8,1,kk)+ 
     -      (4*bfun(5,1,kk)-4*bfun(5,2,kk)+bfun(9,1,kk)-bfun(9,2,kk))*
     -      (dotup(1,2) - dotup(7,1) - dotup(7,2)))*propup(9,1)- 
     -      bfun(8,2,kk)*(propup(7,2) + propup(9,1)))
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
	colmat(3,kk)=(bfun(7,1,kk)*propup(6,1)+bfun(7,4,kk)*propup(6,4)-
     -      (bfun(1,2,kk) + 2*bfun(5,2,kk)*dotup(5,2))*propup(10,2) + 
     -      (-bfun(1,3,kk) + bfun(3,3,kk) - 2*bfun(5,3,kk)*dotup(6,1))*
     -      propup(10,3) + (bfun(2,4,kk) - bfun(4,4,kk) - 
     -      2*bfun(5,4,kk)*dotup(4,2))*propup(11,4) - 
     -      (bfun(1,2,kk) - bfun(2,2,kk) + bfun(4,2,kk) + 
     -      2*bfun(5,2,kk)*(dotup(4,4) + dotup(4,6) + dotup(5,2) + 
     -      dotup(7,2)))*propup(12,1) + 
     -      bfun(3,2,kk)*(propup(3,2) + propup(10,2) + propup(12,1)) - 
     -      bfun(1,1,kk)*propup(12,2) + bfun(3,1,kk)*propup(12,2) - 
     -      bfun(4,1,kk)*(propup(11,1) + propup(12,2)) + 
     -      bfun(2,1,kk)*(propup(2,1) + propup(11,1) + propup(12,2)) + 
     -      2*bfun(5,1,kk)*(-(dotup(7,1)*propup(11,1)) - 
     -      (dotup(4,4) + dotup(4,6) + dotup(5,1) + dotup(7,1))*
     -      propup(12,2)) - (bfun(5,1,kk) + bfun(5,2,kk) - 
     -      bfun(9,1,kk) - bfun(9,2,kk))*propup(14,1))
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
	 colmat(4,kk)=(bfun(7,2,kk)*propup(6,2)+bfun(7,3,kk)*propup(6,3) 
     -      -(bfun(1,1,kk) + 2*bfun(5,1,kk)*dotup(5,1))*propup(10,1) + 
     -      (-bfun(1,4,kk) + bfun(3,4,kk) - 2*bfun(5,4,kk)*dotup(6,2))*
     -      propup(10,4) + (bfun(2,3,kk) - bfun(4,3,kk) - 
     -      2*bfun(5,3,kk)*dotup(4,1))*propup(11,3) - 
     -      bfun(1,2,kk)*propup(12,1) + bfun(3,2,kk)*propup(12,1) - 
     -      bfun(4,2,kk)*(propup(11,2) + propup(12,1)) + 
     -      bfun(2,2,kk)*(propup(2,2) + propup(11,2) + propup(12,1)) - 
     -      2*bfun(5,2,kk)*(dotup(7,2)*propup(11,2) + 
     -      (dotup(4,4) + dotup(4,6) + dotup(5,2) + dotup(7,2))*
     -      propup(12,1))- (bfun(1,1,kk) - bfun(2,1,kk) + bfun(4,1,kk))
     -      *propup(12,2)-2*bfun(5,1,kk)*(dotup(4,4)+dotup(4,6)+
     -      dotup(5,1)+dotup(7,1))*propup(12,2)+bfun(3,1,kk)*
     -      (propup(3,1)+propup(10,1)+propup(12,2)) - (bfun(5,1,kk) +
     -    bfun(5,2,kk) - bfun(9,1,kk) - bfun(9,2,kk))*propup(14,1))
     +    /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
     	  colmat(5,kk)=(-((bfun(6,4,kk) - (bfun(9,3,kk) - bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(6,1) - dotup(6,2)))*propup(8,3)) + 
     -      bfun(6,3,kk)*(propup(5,3) + propup(8,3)) - (bfun(8,2,kk) - 
     -      (4*bfun(5,1,kk)-4*bfun(5,2,kk)+bfun(9,1,kk)-bfun(9,2,kk))*
     -      (dotup(1,2) - dotup(7,1) - dotup(7,2)))*propup(9,1) + 
     -      (bfun(1,4,kk) - bfun(3,4,kk))*propup(10,4) + 
     -      2*bfun(5,4,kk)*dotup(6,2)*propup(10,4) + (bfun(1,1,kk) - 
     -      bfun(3,1,kk))*propup(12,2) - bfun(2,1,kk)*(propup(11,1) + 
     -      propup(12,2)) + bfun(4,1,kk)*(propup(4,1) + propup(11,1) + 
     -      propup(12,2)) +(-bfun(6,1,kk) + bfun(6,2,kk))*propup(13,1) - 
     -      bfun(8,2,kk)*propup(13,1) -(4*bfun(5,2,kk)*(dotup(1,2) - 
     -      dotup(7,1) - dotup(7,2)) - (bfun(9,1,kk) - bfun(9,2,kk))*
     -      (dotup(5,1) + dotup(5,2) - dotup(7,1) - dotup(7,2)))*
     -      propup(13,1) + bfun(8,1,kk)*(propup(7,1) + propup(9,1) +
     -      propup(13,1)) + bfun(5,1,kk)*(2*(dotup(4,4) + dotup(4,6) +
     -      dotup(5,1))*propup(12,2) + 2*dotup(7,1)*(propup(11,1) + 
     -      propup(12,2) - 2*propup(13,1)) + 4*(dotup(1,2) - 
     -      dotup(7,2))*propup(13,1) - propup(14,1)) +((4*bfun(5,2,kk) 
     -      - bfun(9,1,kk) - bfun(9,2,kk))*propup(14,1))/2.)
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
		colmat(6,kk)=(-((bfun(6,3,kk)+(bfun(9,3,kk)-bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(6,1) - dotup(6,2)))*propup(8,3)) + 
     -      bfun(6,4,kk)*(propup(5,4) + propup(8,3)) - 
     -      (4*bfun(5,1,kk) - 4*bfun(5,2,kk) + bfun(9,1,kk) - 
     -      bfun(9,2,kk))*(dotup(1,2) - dotup(7,1) - dotup(7,2))*
     -      propup(9,1) + (bfun(1,3,kk) - bfun(3,3,kk))*propup(10,3)+ 
     -      2*bfun(5,3,kk)*dotup(6,1)*propup(10,3) + 
     -      (bfun(1,2,kk) - bfun(3,2,kk))*propup(12,1) - 
     -      bfun(2,2,kk)*(propup(11,2) + propup(12,1)) + 
     -      bfun(4,2,kk)*(propup(4,2) + propup(11,2) + propup(12,1))+ 
     -      (bfun(6,1,kk) - bfun(6,2,kk))*propup(13,1) - 
     -      4*bfun(5,1,kk)*dotup(1,2)*propup(13,1) - 
     -      ((bfun(9,1,kk) - bfun(9,2,kk))*(dotup(5,1) +dotup(5,2) -  
     -      dotup(7,1) - dotup(7,2)) - 4*bfun(5,1,kk)*(dotup(7,1)
     -      + dotup(7,2)))*propup(13,1) - bfun(8,1,kk)*(propup(9,1) +
     -      propup(13,1)) + bfun(8,2,kk)*(propup(7,2) + propup(9,1) + 
     -      propup(13,1)) + bfun(5,2,kk)*(2*(dotup(4,4) + dotup(4,6)+
     -      dotup(5,2))*propup(12,1) + 2*dotup(7,2)*(propup(11,2) + 
     -      propup(12,1) - 2*propup(13,1)) + 4*(dotup(1,2) -
     -      dotup(7,1))*propup(13,1)-propup(14,1))-((-4*bfun(5,1,kk)+
     -      bfun(9,1,kk) + bfun(9,2,kk))*propup(14,1))/2.)
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
		colmat(7,kk)=(-(bfun(7,3,kk)*propup(6,3)) + 
     -      (bfun(6,4,kk) - (bfun(9,3,kk) - bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(6,1) - dotup(6,2)))*propup(8,3) - 
     -      bfun(6,3,kk)*(propup(5,3) + propup(8,3)) + 
     -      (bfun(8,4,kk) - (4*bfun(5,3,kk) - 4*bfun(5,4,kk) + 
     -      bfun(9,3,kk) - bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(4,1) - dotup(4,2)))*propup(9,3) - 
     -      bfun(8,3,kk)*(propup(7,3) + propup(9,3)))
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
	  	colmat(8,kk)=(-(bfun(7,4,kk)*propup(6,4)) + 
     -      (bfun(6,3,kk) + (bfun(9,3,kk) - bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(6,1) - dotup(6,2)))*propup(8,3) - 
     -      bfun(6,4,kk)*(propup(5,4) + propup(8,3)) + 
     -      (bfun(8,3,kk) + (4*bfun(5,3,kk) - 4*bfun(5,4,kk) + 
     -      bfun(9,3,kk) - bfun(9,4,kk))*
     -      (dotup(1,2) - dotup(4,1) - dotup(4,2)))*propup(9,3) - 
     -      bfun(8,4,kk)*(propup(7,4) + propup(9,3)))
     &      /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))	 
		colmat(9,kk)=(-(bfun(1,1,kk)*propup(1,1))-bfun(2,1,kk)*
     -      propup(2,1)-bfun(3,1,kk)*propup(3,1)-bfun(4,1,kk)*
     &      propup(4,1))/dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*
     &      dotup(8,5))
		colmat(10,kk)=(-(bfun(1,2,kk)*propup(1,2))-bfun(2,2,kk)*
     -	  propup(2,2) - bfun(3,2,kk)*propup(3,2) - bfun(4,2,kk)*
     -      propup(4,2))/dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*
     &      dotup(8,5))	 
	  end do
	 end if
c8---------
      end if

	if(iqqbar.eq.1) then
	 do kk=1,64
	colmat(1,kk)=(propup(2,1)*bfun(2,1,kk)+propup(1,3)*bfun(1,3,kk)
     &       -propup(3,1)*bfun(3,1,kk)-propup(4,3)*bfun(4,3,kk))
     &       /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	colmat(2,kk)=(propup(1,1)*bfun(1,1,kk)+propup(2,3)*bfun(2,3,kk)
     &       +propup(3,1)*bfun(3,1,kk)+propup(4,3)*bfun(4,3,kk))
     &       /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	colmat(3,kk)=(propup(4,1)*bfun(4,1,kk)+propup(4,3)*bfun(4,3,kk))
     &	   /dsqrt(dotup(8,6)*dotup(8,7)*dotup(8,4)*dotup(8,5))
	 end do
	end if

	end
