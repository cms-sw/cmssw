cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...subroutines for gg (or q\bar{q})->bc+b+\bar{c}:  bc in s-wave    c
c...xsection()------square of the amplitude in the c.m.s.            c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...       bc in color-singlet and color-octet states.               c
c copyright (c) z.x zhang, chafik driouich, paula eerola and x.g. wu c
c reference: comput.phys.commun. 159,192(2004); hep-ph/0309120;      c
c reference: comput.phys.commun. 174,241(2006); hep-ph/0502155;      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	subroutine xsection(sigscl,sigvct)
	implicit double precision (a-h,o-z)
	implicit integer (i-n)

	double precision colfac(10),cfscl(10),cfvct(10)
	double complex fincol(10,16),colmat,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/colflow/amp2cf(10),smatval

c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

c...transform the bound state information.
      common/counter/ibcstate,nev
	common/coloct/ioctet

c...initial value.
      smatval=0.0d0
	sigscl =0.0d0
	sigvct =0.0d0

	do ii=1,10
	  cfvct(ii) =0.0d0
	  cfscl(ii) =0.0d0
	  amp2cf(ii)=0.0d0
	  colfac(ii)=0.0d0
	end do

c...color factors for the hadronic production of bc meson.
c...it is used only for the subprocess: gg->bc+b+~c.
	if(iqqbar.eq.0) then
	  if(ioctet.eq.0) then
	    colfac(1) =4.0d0/27.0d0
	    colfac(2) =4.0d0/27.0d0
          colfac(3) =-1.0d0/27.0d0
          colfac(4) =3.0d0/2.0d0
	    colfac(5) =-1.0d0/3.0d0
	  end if
	  if(ioctet.eq.1) then
	    colfac(1) = 32.0d0/81.0d0
	    colfac(2) =-4.0d0/81.0d0
	    colfac(3) = 4.0d0/9.0d0
	    colfac(4) =-1.0d0/18.0d0
	    colfac(5) =-1.0d0/9.0d0
	    colfac(6) = 7.0d0/18.0d0
	    colfac(7) = 1.0d0/162.0d0
	    colfac(8) = 10.0d0/162.0d0
	    colfac(9) = 1.0d0/2.0d0
	    colfac(10)= 7.0d0/2.0d0
	  end if
	end if
c...change all the momenta into light-like.those light-like momenta
c...will be used in inner scalar product and spinner product.
      do 411, i=1,8
	 if (pmomup(5,i) .gt. 1.0d-16) then
	   do k=1,4
	    pmomzero(k,i)=pmomup(k,i)-pmomup(k,8)*pmomup(5,i)**2/
     &	    (2.0d0*dotup(8,i))
	   end do
	   pmomzero(5,i)=0.0d0
	 else
	   do j=1,5
	     pmomzero(j,i)=pmomup(j,i)
         end do
       end if
411	continue

c...get the helicity matrix elements for the free quark part.
c...note here for the condition ibcstate=1, the
c...obtained result for sigscl and sigvct is the same 
c...(this is only used to save time, because now we only need
c...the results for ibcstate=1).
	call freehelicity

	if(ibcstate.eq.1) then
	     ilow=1
	     iup =1
	end if
	if(ibcstate.eq.2) then
	     ilow=2
	     iup =2
	end if

c...begin of big circle.
	do ibc=ilow,iup
	 call bundhelicity(ibc)
c...get the five subsets matrix elements of the process gg->bc+b+\bar{c}:
c...i.e., summing the free quark part (gg->\bar{b}+c+b+\bar{c}) and
c...the bound state part (\bar{c}+c->bc) together.
	 do jj=1,8
	  fincol(1,jj)=colmat(1,jj)*bundamp(1)+colmat(1,jj+8)*bundamp(2)
     &	  +colmat(1,jj+32)*bundamp(3)+colmat(1,jj+40)*bundamp(4)
	  fincol(2,jj)=colmat(2,jj)*bundamp(1)+colmat(2,jj+8)*bundamp(2)
     &	  +colmat(2,jj+32)*bundamp(3)+colmat(2,jj+40)*bundamp(4)
	  fincol(3,jj)=colmat(3,jj)*bundamp(1)+colmat(3,jj+8)*bundamp(2)
     &	  +colmat(3,jj+32)*bundamp(3)+colmat(3,jj+40)*bundamp(4)
	  if(iqqbar.eq.0) then
	   fincol(4,jj)=colmat(4,jj)*bundamp(1)+colmat(4,jj+8)*bundamp(2)
     &	  +colmat(4,jj+32)*bundamp(3)+colmat(4,jj+40)*bundamp(4)
	   fincol(5,jj)=colmat(5,jj)*bundamp(1)+colmat(5,jj+8)*bundamp(2)
     &	  +colmat(5,jj+32)*bundamp(3)+colmat(5,jj+40)*bundamp(4)
	   if(ioctet.eq.1) then
	  fincol(6,jj)=colmat(6,jj)*bundamp(1)+colmat(6,jj+8)*bundamp(2)
     &	  +colmat(6,jj+32)*bundamp(3)+colmat(6,jj+40)*bundamp(4)
	  fincol(7,jj)=colmat(7,jj)*bundamp(1)+colmat(7,jj+8)*bundamp(2)
     &	  +colmat(7,jj+32)*bundamp(3)+colmat(7,jj+40)*bundamp(4)
          fincol(8,jj)=colmat(8,jj)*bundamp(1)+colmat(8,jj+8)*bundamp(2)
     &	  +colmat(8,jj+32)*bundamp(3)+colmat(8,jj+40)*bundamp(4)
	  fincol(9,jj)=colmat(9,jj)*bundamp(1)+colmat(9,jj+8)*bundamp(2)
     &	  +colmat(9,jj+32)*bundamp(3)+colmat(9,jj+40)*bundamp(4)
	    fincol(10,jj)=colmat(10,jj)*bundamp(1)+colmat(10,jj+8)
     &	  *bundamp(2)+colmat(10,jj+32)*bundamp(3)+colmat(10,jj+40)*
     &      bundamp(4)
	   end if
	  end if
       end do

	 do jj=17,24
	fincol(1,jj-8)=colmat(1,jj)*bundamp(2)+colmat(1,jj+8)*bundamp(1)
     &	  +colmat(1,jj+32)*bundamp(4)+colmat(1,jj+40)*bundamp(3)
	fincol(2,jj-8)=colmat(2,jj)*bundamp(2)+colmat(2,jj+8)*bundamp(1)
     &	  +colmat(2,jj+32)*bundamp(4)+colmat(2,jj+40)*bundamp(3)
	fincol(3,jj-8)=colmat(3,jj)*bundamp(2)+colmat(3,jj+8)*bundamp(1)
     &	  +colmat(3,jj+32)*bundamp(4)+colmat(3,jj+40)*bundamp(3)
c2---------
	  if(iqqbar.eq.0) then
	fincol(4,jj-8)=colmat(4,jj)*bundamp(2)+colmat(4,jj+8)*bundamp(1)
     &	  +colmat(4,jj+32)*bundamp(4)+colmat(4,jj+40)*bundamp(3)
	fincol(5,jj-8)=colmat(5,jj)*bundamp(2)+colmat(5,jj+8)*bundamp(1)
     &	  +colmat(5,jj+32)*bundamp(4)+colmat(5,jj+40)*bundamp(3)
c---------
	  if(ioctet.eq.1) then
	fincol(6,jj-8)=colmat(6,jj)*bundamp(2)+colmat(6,jj+8)*bundamp(1)
     &	  +colmat(6,jj+32)*bundamp(4)+colmat(6,jj+40)*bundamp(3)
	fincol(7,jj-8)=colmat(7,jj)*bundamp(2)+colmat(7,jj+8)*bundamp(1)
     &	  +colmat(7,jj+32)*bundamp(4)+colmat(7,jj+40)*bundamp(3)
	fincol(8,jj-8)=colmat(8,jj)*bundamp(2)+colmat(8,jj+8)*bundamp(1)
     &	  +colmat(8,jj+32)*bundamp(4)+colmat(8,jj+40)*bundamp(3)
	fincol(9,jj-8)=colmat(9,jj)*bundamp(2)+colmat(9,jj+8)*bundamp(1)
     &	  +colmat(9,jj+32)*bundamp(4)+colmat(9,jj+40)*bundamp(3)
	  fincol(10,jj-8)=colmat(10,jj)*bundamp(2)+colmat(10,jj+8)*
     &	  bundamp(1)+colmat(10,jj+32)*bundamp(4)+colmat(10,jj+40)*
     &      bundamp(3)
	  end if
c---------
	  end if
c2--------
       end do

c...to get the possibility of three color flows which are used in main 
c...program for the subprocess gg->bc+b+\bar{c}. these three color flow
c...directions are not orthogonal but are corresponding to particular
c...color flow, which can be equivalent derived from the color-flow
c...decomposition in prd67,014026(2003) by f.maltoni etal.
	 if (iqqbar.eq.0) then
c---------
       if(ioctet.eq.0) then
	  if (ibc.eq.1) then
         do jj=1,16
	   cfscl(1)=cfscl(1)+1.0d0/36.0d0*dble(abs(8*fincol(1,jj)
     &	-fincol(3,jj)))**2
	   cfscl(2)=cfscl(2)+1.0d0/36.0d0*dble(abs(8*fincol(2,jj)
     &	-fincol(4,jj)))**2
	   cfscl(3)=cfscl(3)+1.0d0/4.0d0*dble(abs(fincol(5,jj))**2)
	   end do
	  end if
	  if(ibc.eq.2) then
         do jj=1,16
	   cfvct(1)=cfvct(1)+1.0d0/36.0d0*dble(abs(8*fincol(1,jj)
     &	-fincol(3,jj)))**2
	   cfvct(2)=cfvct(2)+1.0d0/36.0d0*dble(abs(8*fincol(2,jj)
     &	-fincol(4,jj)))**2
	   cfvct(3)=cfvct(3)+1.0d0/4.0d0*dble(abs(fincol(5,jj))**2)
	   end do
	  end if
	 end if
c----------
       if(ioctet.eq.1) then
	  if (ibc.eq.1) then
         do jj=1,16
	    cfscl(1)=cfscl(1)+1.0d0/18.0d0*dble(abs(fincol(1,jj))**2)
	    cfscl(2)=cfscl(2)+1.0d0/18.0d0*dble(abs(fincol(2,jj))**2)
	    cfscl(3)=cfscl(3)+1.0d0/2.0d0*dble(abs(fincol(3,jj))**2)
	    cfscl(4)=cfscl(3)+1.0d0/2.0d0*dble(abs(fincol(4,jj))**2)
	    cfscl(5)=cfscl(3)+1.0d0/2.0d0*dble(abs(fincol(5,jj))**2)
	    cfscl(6)=cfscl(3)+1.0d0/2.0d0*dble(abs(fincol(6,jj))**2)
	    cfscl(7)=cfscl(3)+1.0d0/18.0d0*dble(abs(fincol(7,jj))**2)
	    cfscl(8)=cfscl(3)+1.0d0/18.0d0*dble(abs(fincol(8,jj))**2)
	    cfscl(9)=cfscl(3)+1.0d0/18.0d0*dble(abs(fincol(9,jj))**2)
	   cfscl(10)=cfscl(3)+1.0d0/18.0d0*dble(abs(fincol(10,jj))**2)
	   end do
	  end if
	  if(ibc.eq.2) then
         do jj=1,16
	    cfvct(1)=cfvct(1)+1.0d0/18.0d0*dble(abs(fincol(1,jj))**2)
	    cfvct(2)=cfvct(2)+1.0d0/18.0d0*dble(abs(fincol(2,jj))**2)
	    cfvct(3)=cfvct(3)+1.0d0/2.0d0*dble(abs(fincol(3,jj))**2)
	    cfvct(4)=cfvct(3)+1.0d0/2.0d0*dble(abs(fincol(4,jj))**2)
	    cfvct(5)=cfvct(3)+1.0d0/2.0d0*dble(abs(fincol(5,jj))**2)
	    cfvct(6)=cfvct(3)+1.0d0/2.0d0*dble(abs(fincol(6,jj))**2)
	    cfvct(7)=cfvct(3)+1.0d0/18.0d0*dble(abs(fincol(7,jj))**2)
	    cfvct(8)=cfvct(3)+1.0d0/18.0d0*dble(abs(fincol(8,jj))**2)
	    cfvct(9)=cfvct(3)+1.0d0/18.0d0*dble(abs(fincol(9,jj))**2)
	   cfvct(10)=cfvct(3)+1.0d0/18.0d0*dble(abs(fincol(10,jj))**2)
	   end do
	  end if
	 end if
c----------
       end if

c...to get the possibility of two color flow which is used in main 
c...program for the subprocess q\bar{q}->bc+b+\bar{c}. the two color
c...flow directions are orthogonal.
	 if(iqqbar.eq.1) then
	  if(ibc.eq.1) then
         do jj=1,16
	    cfscl(1)=cfscl(1)+9.0d0*dble(abs(-fincol(1,jj)/12.0d0+
     &     2.0d0/3.0d0*fincol(2,jj)-1.0d0/6.0d0*fincol(3,jj)))**2
     &     +3.0d0*dreal((-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)
     &     -1.0d0/6.0d0*fincol(3,jj))*dconjg(-fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))
	    cfscl(2)=cfscl(2)+9.0d0*dble(abs(fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))**2
     &     +3.0d0*dreal((-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)-
     &     1.0d0/6.0d0*fincol(3,jj))*dconjg(fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))
         end do
	  end if 
	  if(ibc.eq.2) then
	   do jj=1,16
	    cfvct(1)=cfvct(1)+9.0d0*dble(abs(-fincol(1,jj)/12.0d0+
     &     2.0d0/3.0d0*fincol(2,jj)-1.0d0/6.0d0*fincol(3,jj)))**2
     &     +3.0d0*dreal((-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)
     &     -1.0d0/6.0d0*fincol(3,jj))*dconjg(-fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))
	    cfvct(2)=cfvct(2)+9.0d0*dble(abs(fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))**2
     &     +3.0d0*dreal((-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)-
     &     1.0d0/6.0d0*fincol(3,jj))*dconjg(fincol(1,jj)/36.0d0-
     &     2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))
	   end do
	  end if	
	 end if

c... note here: since the cross-terms in the color-flow are neglected,
c... the total square of amplitude does not equal to the sum of the
c... square of the partial amplitude (pertain to a certain color-flow).
      
	if(iqqbar.eq.0) then
c6-----------
	 if(ibc.eq.1) then
c1-----------
	   if(ioctet.eq.0) then
	   do jj=1,16
	   sigscl=sigscl+colfac(1)*dble(abs(8*fincol(1,jj)-fincol(3,jj)))
     &	**2+colfac(2)*dble(abs(8*fincol(2,jj)-fincol(4,jj)))**2
     &    +colfac(3)*dreal((8*fincol(1,jj)-fincol(3,jj))*dconjg(8*
     &    fincol(2,jj)-fincol(4,jj)))+colfac(4)*dble(abs(fincol
     &    (5,jj)))**2+colfac(5)*dreal(dconjg(8*fincol(1,jj)+8*
     &    fincol(2,jj)-fincol(3,jj)-fincol(4,jj))*fincol(5,jj))
	   end do
	   end if
c1-----------
c2-----------
	  if(ioctet.eq.1) then
	   do jj=1,16
	   sigscl=sigscl+colfac(1)*abs(fincol(1,jj))**2+2*colfac(2)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(2,jj)))+2*colfac(3)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(3,jj)))+2*colfac(4)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(4,jj)))+2*colfac(6)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(5,jj)))+2*colfac(5)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(6,jj)))+2*colfac(7)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(7,jj)))+2*colfac(8)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(8,jj)))+2*colfac(7)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(9,jj)))+2*colfac(2)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(10,jj)))+colfac(1)*
     &       abs(fincol(2,jj))**2+2*colfac(4)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(3,jj)))+2*colfac(3)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(4,jj)))+2*colfac(5)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(5,jj)))+2*colfac(6)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(6,jj)))+2*colfac(8)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(7)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(2)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(7)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(10,jj)))+4.0d0*abs(fincol(3,jj))**2+
     &       2*colfac(9)*dreal(fincol(3,jj)*dconjg(fincol(4,jj)))+
     &       2*colfac(4)*dreal(fincol(3,jj)*dconjg(fincol(7,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(8,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(9,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(10,jj)))+
     &       4*abs(fincol(4,jj))**2+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(4)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(10,jj)))+colfac(10)*abs(fincol(5,jj))**2-
     &       2*dreal(fincol(5,jj)*dconjg(fincol(6,jj)))+2*colfac(6)*
     &       dreal(fincol(5,jj)*dconjg(fincol(7,jj)))+2*colfac(5)*
     &       dreal(fincol(5,jj)*dconjg(fincol(8,jj)))+2*colfac(6)*
     &       dreal(fincol(5,jj)*dconjg(fincol(9,jj)))+2*colfac(5)*
     &       dreal(fincol(5,jj)*dconjg(fincol(10,jj)))+colfac(10)*
     &       abs(fincol(6,jj))**2+2*colfac(5)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(6)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(5)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(6)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(10,jj)))+colfac(1)*abs(fincol(7,jj))**2+
     &	   2*colfac(2)*dreal(fincol(7,jj)*dconjg(fincol(8,jj)))+
     &	   2*colfac(7)*dreal(fincol(7,jj)*dconjg(fincol(9,jj)))+
     &	   2*colfac(2)*dreal(fincol(7,jj)*dconjg(fincol(10,jj)))
     &       +colfac(1)*abs(fincol(8,jj))**2+2*colfac(2)*
     &       dreal(fincol(8,jj)*dconjg(fincol(9,jj)))+2*colfac(7)*
     &       dreal(fincol(8,jj)*dconjg(fincol(10,jj)))+
     &       colfac(1)*abs(fincol(9,jj))**2+2*colfac(8)*
     &       dreal(fincol(9,jj)*dconjg(fincol(10,jj)))+
     &       colfac(1)*abs(fincol(10,jj))**2
	  end do
	 end if
c2----------
	 end if
c6-----------
c7-----------
	 if(ibc.eq.2) then
c3----------
	  if(ioctet.eq.0) then
	   do jj=1,16
	   sigvct=sigvct+colfac(1)*dble(abs(8*fincol(1,jj)-fincol(3,jj)))
     &	**2+colfac(2)*dble(abs(8*fincol(2,jj)-fincol(4,jj)))**2
     &    +colfac(3)*dreal((8*fincol(1,jj)-fincol(3,jj))*dconjg(8*
     &    fincol(2,jj)-fincol(4,jj)))+colfac(4)*dble(abs(fincol
     &    (5,jj)))**2+colfac(5)*dreal(dconjg(8*fincol(1,jj)+8*
     &    fincol(2,jj)-fincol(3,jj)-fincol(4,jj))*fincol(5,jj))
	   end do
	  end if
c3---------
c4---------
	  if(ioctet.eq.1) then
	   do jj=1,16
	   sigvct=sigvct+colfac(1)*abs(fincol(1,jj))**2+2*colfac(2)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(2,jj)))+2*colfac(3)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(3,jj)))+2*colfac(4)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(4,jj)))+2*colfac(6)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(5,jj)))+2*colfac(5)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(6,jj)))+2*colfac(7)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(7,jj)))+2*colfac(8)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(8,jj)))+2*colfac(7)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(9,jj)))+2*colfac(2)*
     &	   dreal(fincol(1,jj)*dconjg(fincol(10,jj)))+colfac(1)*
     &       abs(fincol(2,jj))**2+2*colfac(4)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(3,jj)))+2*colfac(3)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(4,jj)))+2*colfac(5)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(5,jj)))+2*colfac(6)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(6,jj)))+2*colfac(8)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(7)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(2)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(7)*dreal(fincol(2,jj)*
     &	   dconjg(fincol(10,jj)))+4.0d0*abs(fincol(3,jj))**2+
     &       2*colfac(9)*dreal(fincol(3,jj)*dconjg(fincol(4,jj)))+
     &       2*colfac(4)*dreal(fincol(3,jj)*dconjg(fincol(7,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(8,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(9,jj)))+
     &       2*colfac(3)*dreal(fincol(3,jj)*dconjg(fincol(10,jj)))+
     &       4*abs(fincol(4,jj))**2+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(4)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(3)*dreal(fincol(4,jj)*
     &	   dconjg(fincol(10,jj)))+colfac(10)*abs(fincol(5,jj))**2-
     &       2*dreal(fincol(5,jj)*dconjg(fincol(6,jj)))+2*colfac(6)*
     &       dreal(fincol(5,jj)*dconjg(fincol(7,jj)))+2*colfac(5)*
     &       dreal(fincol(5,jj)*dconjg(fincol(8,jj)))+2*colfac(6)*
     &       dreal(fincol(5,jj)*dconjg(fincol(9,jj)))+2*colfac(5)*
     &       dreal(fincol(5,jj)*dconjg(fincol(10,jj)))+colfac(10)*
     &       abs(fincol(6,jj))**2+2*colfac(5)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(7,jj)))+2*colfac(6)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(8,jj)))+2*colfac(5)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(9,jj)))+2*colfac(6)*dreal(fincol(6,jj)*
     &	   dconjg(fincol(10,jj)))+colfac(1)*abs(fincol(7,jj))**2+
     &	   2*colfac(2)*dreal(fincol(7,jj)*dconjg(fincol(8,jj)))+
     &	   2*colfac(7)*dreal(fincol(7,jj)*dconjg(fincol(9,jj)))+
     &	   2*colfac(2)*dreal(fincol(7,jj)*dconjg(fincol(10,jj)))
     &       +colfac(1)*abs(fincol(8,jj))**2+2*colfac(2)*
     &       dreal(fincol(8,jj)*dconjg(fincol(9,jj)))+2*colfac(7)*
     &       dreal(fincol(8,jj)*dconjg(fincol(10,jj)))+
     &       colfac(1)*abs(fincol(9,jj))**2+2*colfac(8)*
     &       dreal(fincol(9,jj)*dconjg(fincol(10,jj)))+
     &       colfac(1)*abs(fincol(10,jj))**2
	  end do
	  end if
c4----------
	 end if
c7----------
	end if

	if(iqqbar.eq.1) then
	  if(ibc.eq.1) then
	   do jj=1,16
	    sigscl=sigscl+9.0d0*dble(abs(fincol(1,jj)/36.0d0-2.0d0/9.0d0
     &    *fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))**2+6.0d0*dreal(
     &    (-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)-
     &    1.0d0/6.0d0*fincol(3,jj))*dconjg(fincol(1,jj)/36.0d0-
     &    2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))+
     &    9.0d0*dble(abs(-fincol(1,jj)/12.0d0+2.0d0/3.0d0
     &    *fincol(2,jj)-1.0d0/6.0d0*fincol(3,jj)))**2
	   end do	
	  end if
	  if(ibc.eq.2) then
	   do jj=1,16
	    sigvct=sigvct+9.0d0*dble(abs(fincol(1,jj)/36.0d0-2.0d0/9.0d0
     &    *fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))**2+6.0d0*dreal(
     &    (-fincol(1,jj)/12.0d0+2.0d0/3.0d0*fincol(2,jj)-
     &    1.0d0/6.0d0*fincol(3,jj))*dconjg(fincol(1,jj)/36.0d0-
     &    2.0d0/9.0d0*fincol(2,jj)+5.0d0/18.0d0*fincol(3,jj)))+
     &    9.0d0*dble(abs(-fincol(1,jj)/12.0d0+2.0d0/3.0d0
     &    *fincol(2,jj)-1.0d0/6.0d0*fincol(3,jj)))**2
	   end do	
	  end if
	end if

	end do
c...end of big circle.

c...collect the color-flow information:

	if(ibcstate.eq.1) then
	   do jj=1,10
	     amp2cf(jj)=cfscl(jj)
	     smatval=smatval+amp2cf(jj)
	   end do
	end if
	if(ibcstate.eq.2) then
	   do jj=1,10
	     amp2cf(jj)=cfvct(jj)
	     smatval=smatval+amp2cf(jj)
	   end do
	end if

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases.
c...double precision.
	do ii=1,3
	   if(amp2cf(ii).lt.1.0d-16) amp2cf(ii)=1.0d-16
	end do      
	if(smatval.lt.1.0d-16) smatval=1.0d-16
	if(sigscl.lt.1.0d-16)  sigscl =1.0d-16
	if(sigvct.lt.1.0d-16)  sigvct =1.0d-16
      
      return
	end
