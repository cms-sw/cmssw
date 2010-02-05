
c*********************************************************************

c...generate allowed phase_space and right kinematical ranges(to ensure
c...the subprocess has enough energy).
      subroutine phpoint(zup,wt)
c...preamble: declarations.
      implicit double precision (a-h, o-z)
	implicit integer(i-n)

c...pythia common block.
	parameter (maxnup=500)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax

	double complex colmat,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)

c...to get the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly

      dimension zup(7),zzup(5)

c...initial value.
      wt=0.0d0

c--------------------------------------------
c****constraints from the kinematical region.

      taumin =((pmbc+pmb+pmc)/ecm)**2
	taumax =1.0d0
      
c------------------------------------------
      yymin = 0
      yymax = 0
      if(isubonly.eq.1) then
         x1=subfactor
	   x2=subfactor
	   if((x1*x2).lt.taumin) then
	     write(*,'(a)') 'energy not high enough!'
	     write(3,'(a)') 'energy not high enough!'
	     stop
	   end if
	else
	   tau  = (taumax-taumin)*zup(6)+taumin
	   yymin= dlog(dsqrt(tau))
	   yymax=-dlog(dsqrt(tau))
	   yy   = (yymax-yymin)*zup(7)+yymin
	   x1=dsqrt(tau)*exp(yy)
	   x2=dsqrt(tau)*exp(-yy)
	end if

c...in the c.m. system.
	pup(1,1) = 0.0d0
	pup(2,1) = 0.0d0
	pup(3,1) = dsqrt(x1*x2)*ecm/2.0d0
	pup(4,1) = dsqrt(x1*x2)*ecm/2.0d0
	pup(5,1) = 0.0d0

	pup(1,2) = 0.0d0
	pup(2,2) = 0.0d0
	pup(3,2) =-dsqrt(x1*x2)*ecm/2.0d0
	pup(4,2) = dsqrt(x1*x2)*ecm/2.0d0
	pup(5,2) = 0.0d0

c----------------------------------------------

      do i=1,5
	   zzup(i)=zup(i)
	end do
      
c...calculate the jacobian of the final particles in c.m. frame.
	et =pup(4,1)+pup(4,2)

      izero=0

c...increase the phase-space efficiency.
321   call phase_gen(zzup,et,wt)
	if((wt .lt. 1.0d-16).and.(izero.lt.100000))then
	  izero=izero+1
	  goto 321
	end if

c--------------------------------------------

	if(isubonly.eq.1) then
	  return
	else
	  wt=(taumax-taumin)*(yymax-yymin)*wt
	  return
	end if

c...after calling phase_gen all the particle's momenta are 
c...constructed effectively, which is in the c.m.s.

	end
	
