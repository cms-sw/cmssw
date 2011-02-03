
c**********************************************************************

      double precision function totfun(zup,papawt)
	implicit double precision (a-h,o-z)
	implicit integer (i-n)

#include "inclcon.h"

c...pythia common block.
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
	common/pypars/mstp(200),parp(200),msti(200),pari(200)
	parameter (maxnup=500)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

c...user transformation.
	double complex colmat,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
c...transform the bound state information.
	common/counter/ibcstate,nev
	common/rconst/pi
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/confine/ptcut,etacut
	common/colflow/amp2cf(10),smatval

c...parameters transformtion used in totfun()
      common/funtrans/nq2,npdfu

c...to get the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly

c...generate---switch for full events.
	logical generate
	common/genefull/generate

c...to get the distribution of an extra factor z=(2(k1+k2).p_bc)/shat.
      common/extraz/zfactor,zmin,zmax
	common/outpdf/ioutpdf,ipdfnum
	common/intinip/iinip
	common/intinif/iinif

c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

      dimension xpp(-25:25),xppbar(-25:25),zup(7),pboo(4),pc(4),pl(4)
      data conv/3.8938573d+8/ !pb
      common /ppp/ pp(4,40),guv(4)

c------------------------------------------------

      totfun=0.0d0

c------------------------------------------------
      if(isubonly.eq.1) then
         x1=subfactor
	   x2=subfactor
	else
         taumin =((pmbc+pmb+pmc)/ecm)**2
	   taumax =1.0d0
	   tau=(taumax-taumin)*zup(6)+taumin
	   yymin= dlog(dsqrt(tau))
	   yymax=-dlog(dsqrt(tau))
	   yy   =(yymax-yymin)*zup(7)+yymin
	   x1 =dsqrt(tau)*exp(yy)
	   x2 =dsqrt(tau)*exp(-yy)
	end if

c-------------------------------------------------

c... gluon 1, in lab
      pup(1,1)= 0.0d0
      pup(2,1)= 0.0d0
      pup(3,1)= ecm*x1/2.0d0
      pup(4,1)= ecm*x1/2.0d0
      pup(5,1)= 0.0d0

c... gluon 2, in lab
      pup(1,2)= 0.0d0
      pup(2,2)= 0.0d0
      pup(3,2)=-ecm*x2/2.0d0
      pup(4,2)= ecm*x2/2.0d0
      pup(5,2)= 0.0d0
  
c...change momtuma of the final particals into lab coordinate system.
c...the original one getting from phase_gen is result for c.m. system.
	do i=1,4
          pboo(i)=pup(i,1)+pup(i,2)
      end do

      do 101, i=1,3
	    do j=1,4
	      pc(j)=pmomup(j,i+2)
          end do
		call lorentz(pboo,pc,pl)
	    do j=1,4
	      pmomup(j,i+2)=pl(j)
          end do
101   continue

c...set up kinematics of the out going particles: bc, b and c~.
      do i=3,5
	  do j=1,5
	    pup(j,i)=pmomup(j,i)
	  end do
	end do

c...for s-wave bound state, taking non-relativistic approximation.
	do i=1,5
	  pmomup(i,6)=pmb/(pmb+pmc)*pmomup(i,3)
	  pmomup(i,7)=pmc/(pmb+pmc)*pmomup(i,3)
      end do

c...incoming gluon momenta used in s_amp.for.
     	do i=1,2
	  do j=1,5
	    pmomup(j,i)=pup(j,i)
	  end do
	end do

c...this part is from the inner part of pythia subroutine pyp().
      ptbc  =dsqrt(pup(1,3)**2+pup(2,3)**2)
      pr    =max(1.0d-16,pup(5,3)**2+pup(1,3)**2+pup(2,3)**2)
	prs   =max(1.0d-16,pup(1,3)**2+pup(2,3)**2)
      eta   =sign(dlog(min((dsqrt(pr+pup(3,3)**2)+dabs(pup(3,3)))
     &          /dsqrt(pr),1.0d+20)),pup(3,3))
	pseta =sign(dlog(min((dsqrt(prs+pup(3,3)**2)+dabs(pup(3,3)))
     &          /dsqrt(prs),1.0d+20)),pup(3,3))

c...other confinement can also be added here.
	if(ptbc.lt.ptcut .or. abs(eta).gt.etacut) then
	  if (generate) then
	    do ii=1,10
	      amp2cf(ii)=0.0d0
	    end do
        end if
	  smatval=0.0d0
	  return
	end if

c...energy scale.
	if(nq2.eq.1) q2 =x1*x2*ecm**2/4.0d0
      if(nq2.eq.2) q2 =x1*x2*ecm**2
	if(nq2.eq.3.or.nq2.eq.8) q2 =ptbc**2.0d0+pup(5,3)**2
	if(nq2.eq.4) then
	  q=0.0d0
	  do i=3,5
	    q=q+dsqrt(pup(1,i)**2+pup(2,i)**2+pup(5,i)**2)
	  end do
	  q2=q**2
	end if
	if(nq2.eq.5) then
	  q=0.0d0
	  do i=3,5
	    q=q+dsqrt(pup(1,i)**2.0d0+pup(2,i)**2.0d0+pup(5,i)**2.0d0)
	  end do
	  q2=(q/3.0d0)**2.0d0
	end if
	if(nq2.eq.6.or.nq2.eq.7) then
	  q2=pmb**2+pup(1,4)**2+pup(2,4)**2
	end if
c...this is the energy scale used in gouz's program
	if(nq2.eq.9) then
	  q2=4.0d0*pmb**2
	end if
	
c...get the value of alphas. all are in leading order.
	if(ioutpdf.eq.1) then
	   if(ipdfnum.eq.100) alps =alpgrv(q2,1)*4*pi
	   if(ipdfnum.eq.200) alps =alpmsrt(dsqrt(q2),0.220d0,0)
	   if(ipdfnum.eq.300) alps =alpcteq(q2,1)*4*pi
	else   
	   alps =pyalps(q2)
      end if

c...two energy scale for alphas.
c...alphas^4=alphas^2(\mu_b**2)*alphas^2(\mu_c**2).
	if(nq2.eq.6.or.nq2.eq.8) then
	   alps1=alps
	   if(ioutpdf.eq.1) then
	      q2=4.0d0*pmc**2.0d0
		  if(ipdfnum.eq.100) alps2 =alpgrv(q2,1)*4*pi
	      if(ipdfnum.eq.200) alps2 =alpmsrt(dsqrt(q2),0.22d0,0)
	      if(ipdfnum.eq.300) alps2 =alpcteq(q2,1)*4*pi
	   else
	      alps2 =pyalps(4.0d0*pmc**2.0d0)
         end if
	   alps =dsqrt(alps1*alps2)
	end if

c...store scale choice and alphas.
      scalup=dsqrt(q2)
      aqcdup=alps

	if(isubonly.eq.0) then
	 if(ioutpdf.eq.0) then
c...tevatron
c...evaluate parton distribution for (g1<--p,g2<---p~)
	   if(npdfu.eq.1) then
	     call pypdfu(2212,x1,q2,xpp)
           call pypdfu(-2212,x2,q2,xppbar)
	   end if
c...lhc
c...evaluate parton distribution for (g1<--p,g2<---p)
 	   if(npdfu.eq.2) then
	     call pypdfu(2212,x1,q2,xpp)
           call pypdfu(2212,x2,q2,xppbar)
	   end if
	 else
	   if(ipdfnum.eq.100) then
	     call grv98pa(1, x1, q2, uv, dv, us, ds, ss, gl1)
           call grv98pa(1, x2, q2, uv, dv, us, ds, ss, gl2)
	     if(iqqbar.eq.0) then
		   xpp(21)=gl1
		   xppbar(21)=gl2
	     else
	       if(iqcode.eq.1) then
	         if(npdfu.eq.1) then     !tevatron
			    xpp(iqcode)=uv+us
	            xpp(-iqcode)=us
			    xppbar(iqcode)=uv+us
			    xppbar(-iqcode)=us
		     end if
	         if(npdfu.eq.2) then     !lhc
			    xpp(iqcode)=uv+us
	            xpp(-iqcode)=us
			    xppbar(iqcode)=us
			    xppbar(-iqcode)=uv+us
		     end if
		   end if
		   if(iqcode.eq.2) then
	         if(npdfu.eq.1) then     !tevatron (u-p,~u-~p and u-~p,~u-p)
			    xpp(iqcode)=dv+ds
	            xpp(-iqcode)=ds
			    xppbar(iqcode)=dv+ds
			    xppbar(-iqcode)=ds
		     end if
	         if(npdfu.eq.2) then     !lhc (u-p,~u-p and u-p,~u-p)
			    xpp(iqcode)=dv+ds
	            xpp(-iqcode)=ds
			    xppbar(iqcode)=ds
			    xppbar(-iqcode)=dv+ds
		     end if
		   end if
		   if(iqcode.eq.3) then      ! the same for tevatron or lhc
		     xpp(3)=2.0d0*ss
		     xppbar(3)=2.0d0*ss
		   end if
           end if
	   end if
	   if(ipdfnum.eq.200) then
	     qq=dsqrt(q2)
		 call mrstlo(x1,qq,1,upv,dnv,usea,dsea,str,chm,bot,glu1)
	     call mrstlo(x2,qq,1,upv,dnv,usea,dsea,str,chm,bot,glu2)
	     if(iqqbar.eq.0) then
		   xpp(21)=glu1
		   xppbar(21)=glu2
	     else
	       if(iqcode.eq.1) then
	         if(npdfu.eq.1) then     !tevatron (u-p,~u-~p and u-~p,~u-p)
			    xpp(iqcode)=upv+usea
	            xpp(-iqcode)=usea
			    xppbar(iqcode)=upv+usea
			    xppbar(-iqcode)=usea
		     end if
	         if(npdfu.eq.2) then     !lhc (u-p,~u-p and u-p,~u-p)
			    xpp(iqcode)=upv+usea
	            xpp(-iqcode)=usea
			    xppbar(iqcode)=usea
			    xppbar(-iqcode)=upv+usea
		     end if
		   end if
		   if(iqcode.eq.2) then
	         if(npdfu.eq.1) then     !tevatron (u-p,~u-~p and u-~p,~u-p)
			    xpp(iqcode)=dnv+dsea
	            xpp(-iqcode)=dsea
			    xppbar(iqcode)=dnv+dsea
			    xppbar(-iqcode)=dsea
		     end if
	         if(npdfu.eq.2) then     !lhc (u-p,~u-p and u-p,~u-p)
			    xpp(iqcode)=dnv+dsea
	            xpp(-iqcode)=dsea
			    xppbar(iqcode)=dsea
			    xppbar(-iqcode)=dnv+dsea
		     end if
		   end if
		   if(iqcode.eq.3) then
		     xpp(3)=str*2.0d0
		     xppbar(3)=str*2.0d0
             end if
		 end if
	   end if
	   if(ipdfnum.eq.300) then
		 qq=dsqrt(q2)
c...cteq6l.
	     if(iqqbar.eq.0) then
	       xpp(21)   =ctq6pdf(0,x1,qq)
		   xppbar(21)=ctq6pdf(0,x2,qq)
	     else
	       if(npdfu.eq.1) then         !tevatron
		     xpp(iqcode)=ctq6pdf(iqcode,x1,qq)
		     xpp(-iqcode)=ctq6pdf(-iqcode,x1,qq)
		     xppbar(iqcode)=ctq6pdf(iqcode,x2,qq)
		     xppbar(-iqcode)=ctq6pdf(-iqcode,x2,qq)
	       end if
	       if(npdfu.eq.2) then         !lhc
		     xpp(iqcode)=ctq6pdf(iqcode,x1,qq)
		     xpp(-iqcode)=ctq6pdf(-iqcode,x1,qq)
		     xppbar(iqcode)=ctq6pdf(-iqcode,x2,qq)
		     xppbar(-iqcode)=ctq6pdf(iqcode,x2,qq)
	       end if
		 end if
	   end if
	 end if
	end if	

c...this ensure the rightness of the extrapolation of the pdf.
c...(by using the pdfs, sometimes it will get negative value)
	if(xpp(21).lt.1.0d-16) xpp(21)=0.0d0
	if(xppbar(21).lt.1.0d-16) xppbar(21)=0.0d0

	if(iqqbar.eq.1) then
	 if(xpp(iqcode).lt.1.0d-16) xpp(iqcode)=0.0d0
	 if(xppbar(iqcode).lt.1.0d-16) xppbar(iqcode)=0.0d0
	 if(xpp(-iqcode).lt.1.0d-16) xpp(-iqcode)=0.0d0
	 if(xppbar(-iqcode).lt.1.0d-16) xppbar(-iqcode)=0.0d0
	end if  

c...for the sub-process, taking the constant alphas value.
	if(isubonly.eq.1) alps=0.20d0

c...if not generate s-wave, go to the part for p-wave.
	if(ibcstate.eq.3) goto 1005
	if(ibcstate.eq.4) goto 1005
	if(ibcstate.eq.5) goto 1005
	if(ibcstate.eq.6) goto 1005

c...common factor for s-wave states.
	phase =papawt*alps**4.0d0/(2.0d0**11*pi*3.0d0*dotup(1,2))

c...first to get the square of the amplitude.
c...note 1) the momenta inputed into this subroutine is pmomup(j,i):
c...bc+: i=3, b: i=4, ~c: i=5; 
c...j=1: p_x; j=2: p_y; j=3: p_z; j=4: e; j=5: mass;	2) all the momenta
c...now are in the lab system, you may directly get the momenta in c.m.s
c...of gluon-gluon subsystem before running lorentz transformation used
c...above. the cross-section will not change under the cordinate 
c...transformation. 3) sigscl and sigvct, after calling this subroutine,
c...is not the final cross-section, we need to add some coefficients.
c...there we don't need an extra q in xsection_bcy, because 4) the momenta
c...are now also stored in pup(j,i), which is transformed according to 
c...a pythia common block.
	call xsection(sigscl,sigvct)


c...get the right phase for the subprocess: q+~q->bc+b+~c from
c...the same precedure of subprocess g+g->bc+b+~c.
	if(iqqbar.eq.1) then
	  phase=phase*(2.0d0**6/3.0d0**2)
	end if

c...the correct cross-section for the subprocess to a particular particle
c...momenta. 
	sigscl=conv*phase*sigscl
	sigvct=conv*phase*sigvct

	if(ibcstate.eq.1) sigcross=sigscl
	if(ibcstate.eq.2) sigcross=sigvct

c...get the cross-section for s-wave.	
	if(isubonly.eq.0) then
	  if(iqqbar.eq.0) then
	   totfun =sigcross*xpp(21)*xppbar(21)/x1/x2
	   if(ioutpdf.eq.1 .and. ipdfnum.eq.300) then
	    totfun=sigcross*xpp(21)*xppbar(21)
	   end if
	  else
		totfun =sigcross*(xpp(iqcode)*xppbar(iqcode)+
     &		xpp(-iqcode)*xppbar(-iqcode))/x1/x2
          if(ioutpdf.eq.1 .and. ipdfnum.eq.300) then
	      totfun =sigcross*xpp(iqcode)*xppbar(iqcode)
	    end if
	  end if
	else
	  totfun =sigcross
	end if

c...getting an extra distribution about z=(2(k1+k2).p_bc)/shat.
      zfactor=2.0d0*(dotup(1,3)+dotup(2,3))/(x1*x2*ecm**2.0d0)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases.
	if(totfun.lt.1.0d-16) totfun=1.0d-16

	return

c************************************************
c...the following is for the p-wave states.
c************************************************

1005  continue

      if(ibcstate.ne.1.and.ibcstate.ne.2) then
c...the p-wave part is adopted from fdc program and here
c...is to make all the variable consistent to fdc.
c...in the lab system. 
c...pp(1,i)=e=pup(4,i); 
c...pp((2,3,4),i)=(p_x,p_y,p_z)=pup((1,2,3),i).
c...pp(4)=pup(5)--b quark; pp(5)=pup(4)--\bar{c} quark.
	 do i=1,3
	   pp(2,i)=pup(1,i)
	   pp(3,i)=pup(2,i)
	   pp(4,i)=pup(3,i)
	   pp(1,i)=pup(4,i)
	 end do

	 pp(2,4)=pup(1,5)
	 pp(3,4)=pup(2,5)
	 pp(4,4)=pup(3,5)
	 pp(1,4)=pup(4,5)

	 pp(2,5)=pup(1,4)
	 pp(3,5)=pup(2,4)
	 pp(4,5)=pup(3,4)
	 pp(1,5)=pup(4,4)

c...this is an overall color factor, where the average over
c...the initial gluons have been included, i.e. the factor
c...(1/2**6) has been included into cfact, i.e.
c...   cfact=(33.0d0/2.0d0)/(2.0d0**6)
	 cfact=0.2578125d0

c...this is to get constants for the gluon coupling constant g.
	 g3=dsqrt(4.0d0*pi*alps)

	 if(ibcstate.eq.3) then
	    ampofpw=amps2_1p1()
	    phase=papawt*cfact*g3**8/(2.0d0**9*pi**5*dotup(1,2))
	 end if
	 if(ibcstate.eq.4) then
	    ampofpw=amps2_3p0()
	    phase=papawt*cfact*g3**8/(2.0d0**9*pi**5*dotup(1,2))
	 end if
	 if(ibcstate.eq.5) then
	    ampofpw=amps2_3p1()
	    phase=papawt*cfact*g3**8/(2.0d0**9*pi**5*dotup(1,2))
	 end if
	 if(ibcstate.eq.6) then
	    ampofpw=amps2_3p2()
	    phase=papawt*cfact*g3**8/(2.0d0**9*pi**5*dotup(1,2))
	 end if
c...get the cross-sectin for p-wave.
	 if(isubonly.eq.1) then
	   totfun=conv*phase*ampofpw
	 else
	   totfun =conv*phase*ampofpw*xpp(21)*xppbar(21)/x1/x2
	   if(ioutpdf.eq.1 .and. ipdfnum.eq.300) then
	     totfun=conv*phase*ampofpw*xpp(21)*xppbar(21)
	   end if
	 end if
	end if

c...getting an extra distribution about z=(2(k1+k2).p_bc)/shat.
      zfactor=2.0d0*(dotup(1,3)+dotup(2,3))/(x1*x2*ecm**2.0d0)

c...the following is only to eliminate the numerical uncerntainty,
c...which in principle does not needed. however we added here 
c...to avoid some very particular cases.
	if(totfun.lt.1.0d-16) totfun=1.0d-16

	return
	end
