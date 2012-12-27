
c**********************************************************************
c**********************************************************************
c*** vegas program
c*** vegas()---- integration over (0,1) integratal variables
c*** generand--- generate random number according to grade function
c*** randa------ generate a group of random number with pyr(0)
c**********************************************************************
c**********************************************************************

c***********************************************************************
c...changes are made which make the program only to get the importance
c...function. 
      subroutine vegas(fxn,ndim,ncall,itmx,nprn,xint,xmax,imode)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c    arguments:                                                    c
c       fxn    = function to be integrated/mapped                  c
c       ndim   = # dimensions                                      c
c       ncall  = maximum total # of calls to the function          c
c       itmx   = maximum # of iterations allowed                   c
c       nprn   = printout level:                                   c
c              >=2  additionally inf. about accumulated values     c
c                     per iteration.                               c
c       xint   = integral value will be used when imode=1          c
c       xmax   = upper bound of of the function                    c
c       imode  = 0 get integral value will be used when imode=1    c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit double precision(a-h,o-z)
	implicit integer (i-n)

c...pythia common block.
      parameter (maxnup=500)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

#include "invegas.h"
#include "bcvegpy_set_par.inc"

c...user process event common block.
	common/grade/xi(NVEGBIN,10)
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/vegcross/vegsec,vegerr,iveggrade
	common/vegaspar/si,si2,swgt,schi,itime
c	character*8 begin_time,end_time

	dimension x(ndim),xin(NVEGBIN),r(NVEGBIN),ia(ndim),
     &    d(NVEGBIN,ndim),xmax(NVEGBIN,ndim)
c      data alph/1.5d0/

      alph=1.50d0
c...number of bins.
      nd=NVEGBIN
c===============================================================
c a))   initializing some variables
c===============================================================
      if(iveggrade.eq.0.and.(imode.eq.0)) then
      si    =0.0d0
      si2   =0.0d0
      swgt  =0.0d0
      schi  =0.0d0
	  itime =0
	  else
	  itime =itime+1
	  end if
	  
	calls =ncall
      xnd   =nd
      ndm   =nd-1
c...............................................................
c  defining the initial intervals distribution
c...............................................................
      if(iveggrade.eq.0.and.imode.eq.0) then
	  rc=1.0d0/xnd
        do 7 j=1,ndim
          xi(nd,j)=1.0d0
          dr=0.0d0
        do 7 i=1,ndm
          dr=dr+rc
          xi(i,j)=dr
7       continue
      end if
c...xint is the value of integral vegsec when imode=0      
      if(imode.eq.1.and.(iveggrade.eq.0)) then
         do idm=1,ndim
            do ibin=1,nd
               xmax(ibin,idm)=
     #              xint**(1d0/ndim)
            enddo
         enddo
      endif

c===============================================================
c b))  iterating loop
c===============================================================
      do 1000 it=1,itmx
c        if(it.eq.1) then
c	    call time(begin_time)
c	    write(*,'(a)') begin_time
c	    write(3,'(a)') begin_time
c	  end if
c...............................................................
c  initializing iteration variables
c...............................................................
        ti   =0.0d0
        sfun2=0.0d0
        do 10 j=1,ndim
        do 10 i=1,nd
         d(i,j)=0.0d0
10      continue
	  do 11 jj=1,ncall
          call generand(ndim,xnd,x,ia,wgt)
	    call phpoint(x,wt)
		if(wt.lt.1.0d-15) then
	      tempfxn=0.0d0
		else
		  tempfxn=fxn(x,wt)*wgt
	    end if
c...record the maximum differential cross-section.
		if(tempfxn.gt.crossmax) then
	       crossmax=tempfxn
		end if
          fun=tempfxn/calls
          fun2=fun*fun
          weight=wgt/calls
          ti=ti+fun
          sfun2=sfun2+fun2
          do j=1,ndim
            iaj=ia(j)
            d(iaj,j)=d(iaj,j)+fun2
          end do
c===============================================================
c----- update the upper bounding envelope
c===============================================================
		  if(imode.eq.1) then
            prod=1
            do idm=1,ndim
               prod=prod*xmax(ia(idm),idm)
            enddo
            prod=(tempfxn/prod)
            if(prod.gt.1) then
               if(it.eq.itmx)  call increasecnt
     #    ('Error: the xmax is ramain unstable change iteration value')
c This guarantees a 10% increase of the upper bound in this cell
               prod=1+0.1d0/ndim
               do idm=1,ndim
                  xmax(ia(idm),idm)=xmax(ia(idm),idm)
     #          * prod
               enddo
            endif
         endif   
11      continue
c...............................................................
c  computing the integral and error values
c...............................................................
        ti2=ti*ti
        tsi=dsqrt((sfun2*calls-ti2)/(calls-1.0d0))
        wgta=ti2/tsi**2
        si=si+ti*wgta
        si2=si2+ti2
        swgt=swgt+wgta
        schi=schi+ti2*wgta
        avgi=si/swgt
        chi2a=0.0d0
        sd=swgt*(it+itmx*itime)/si2
        if(it.gt.1)chi2a=sd*(schi/swgt-avgi*avgi)/(it-1)
        sd=1.0d0/dsqrt(sd)
        err=sd*100.0d0/avgi

c...record the cross-section and the corresponding err obtained.
c...may be used for pythia running(initialization).
	  vegsec=avgi  !pb
	  vegerr=err
c...............................................................
c  printing
c...............................................................        
c...cross unit pb.
	  if(nprn.ge.2) then
	   write(*,201) it+itmx*itime,avgi,sd,err
		write(3,201) it+itmx*itime,avgi,sd,err
c	    call time(end_time)
c		write(*,'(a)') end_time
c		write(3,'(a)') end_time
	  end if

778     format('integral value =',g10.4,'+/-',
     &   g10.4,3x,' chi**2=',g10.4, /' ')
201     format('iter.no',i3,' acc.result(pb)==>',g14.5,'+/-',g10.4,
     .   '... % err=',g10.2)
c===============================================================
c  c))   redefining the grid
c===============================================================
c...............................................................
c  smoothing the f**2 valued stored for each interval
c...............................................................
        do 23 j=1,ndim
          xo=d(1,j)
          xn=d(2,j)
          d(1,j)=(xo+xn)/2.0d0
          x(j)=d(1,j)
          do 22 i=2,ndm
            d(i,j)=xo+xn 
            xo=xn
            xn=d(i+1,j)
            d(i,j)=(d(i,j)+xn)/3.0d0
            x(j)=x(j)+d(i,j)
22        continue
            d(nd,j)=(xn+xo)/2.0d0
            x(j)=x(j)+d(nd,j)
23      continue
c...............................................................
c   computing the 'importance function' of each interval
c...............................................................
        do 28 j=1,ndim
         rc=0.0d0
         do i=1,nd
          r(i)=0.0d0
          if(d(i,j).le.0.) go to 224
          xo=x(j)/d(i,j)
          r(i)=((xo-1.0d0)/xo/dlog(xo))**alph
224       rc=rc+r(i)
         end do
c...............................................................
c  redefining the size of each interval
c...............................................................
        rc=rc/xnd
        kk=0
        xn=0.0d0
        dr=0.0d0
        i=0
25      kk=kk+1
        dr=dr+r(kk)
        xo=xn
        xn=xi(kk,j)
26      if(rc.gt.dr) go to 25
        i=i+1
        dr=dr-rc
        xin(i)=xn-(xn-xo)*dr/r(kk)
        if(i.lt.ndm) go to 26
        do i=1,ndm
          xi(i,j)=xin(i)
        end do
        xi(nd,j)=1.0d0
28      continue
1000  continue
	
	return
      end

c**************************************************************
      subroutine generand(ndim,xnd,x,ia,weight)
	implicit double precision(a-h,o-z)
      implicit integer (i-n)

#include "invegas.h"
#include "bcvegpy_set_par.inc"
      
	common/grade/xi(NVEGBIN,10)
	dimension ia(10),x(10),randx(10)

c...to get the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly

      weight=1.0d0

      call randa(ndim,randx)
c...............................................................
c  computing the point position
c...............................................................
	do 15 j=1,ndim
         xn=randx(j)*xnd+1.0d0
         ia(j)=xn
         xim1=0.0d0
         if(ia(j).gt.1) xim1=xi(ia(j)-1,j)
         xo=xi(ia(j),j)-xim1
         x(j)=xim1+(xn-ia(j))*xo
         weight=weight*xo*xnd
15    continue      

      if(isubonly.eq.1) then
         x(6)=subfactor
	   x(7)=subfactor
      end if

      return
	end

c*****************************************************************

      subroutine randa(n,randx)
      implicit double precision(a-h,o-z)
	implicit integer (i-n)
      dimension randx(n)

      do i=1,n
         randx(i)=pyr(0)
      end do

      return
      end

      subroutine gen(fxn,ndim,xmax,imode)
c imode=0 to initialize
c imode=1 to generate
c imode=3 store generation efficiency in eff
	implicit double precision(a-h,o-z)
	implicit integer (i-n)

c...user process event common block.
	common/subopen/subfactor,subenergy,isubonly
	common/grade/xi(NVEGBIN,8)
	dimension x(ndim),ia(ndim),xmax(NVEGBIN,ndim),ncell(ndim),xmmm(NVEGBIN,8)

      integer icalls,mcalls,idm,ibin
      save icalls,mcalls,xmmm

      nd=NVEGBIN
      xnd =nd*1.0d0
      if(imode.eq.0) then
         do idm=1,ndim           
            xmmm(1,idm)=xmax(1,idm)
            do ibin=2,nd
               xmmm(ibin,idm)=xmmm(ibin-1,idm)+
     #              xmax(ibin,idm)
            enddo
            do ibin=1,nd
               xmmm(ibin,idm)=xmmm(ibin,idm)/xmmm(nd,idm)
c               write(*,*), xmmm(ibin,idm)
            enddo
         enddo
         icalls=0
         mcalls=0
         return
      elseif(imode.eq.3) then
         if(icalls.gt.0) then
            eff=dble(mcalls)/icalls
         else
            eff=-1
         endif
         write(*,*), 'Efficiency in the generation is ',eff
         return
      endif
       mcalls=mcalls+1
 10   continue
      do idm=1,ndim
         r=pyr(0)
         do ibin=1,nd
            if(r.lt.xmmm(ibin,idm)) then
               ncell(idm)=ibin
               goto 1
            endif
         enddo
 1       continue
      enddo
      ubound=1
      do idm=1,ndim
         ubound=ubound*xmax(ncell(idm),idm)
      enddo
c       write(*,*),'The ubound is ',ubound 
      fun=0

      call generand(ndim,xnd,x,ia,wgt)
		call phpoint(x,wt)
		if(wt.lt.1.0d-15) then
	      fun=0.0d0
		else
		  fun=fun+fxn(x,wt)*wgt
	   end if
      if(fun.lt.0) then
         write(*,*) 'gen: non positive function'
         stop
      endif
      if(fun.gt.ubound) then
         call increasecnt 
     #    ('upper bound failure')
      endif
      ubound=ubound*pyr(0)
       icalls=icalls+1
      if(ubound.gt.fun) then
         call increasecnt
     #    ('vetoed calls in inclusive cross section')
         goto 10
      endif
c      print *,'The ubound*random is  ',ubound
c      print *,'the cross-section is ', fun
c....when ubound>fun>ubound*random() the event accept  
      end

      subroutine increasecnt(string)
      implicit none
      character *(*) string
      integer maxnum
      parameter (maxnum=100)
      character * 100 keywords(maxnum)
      real * 8 counters(maxnum)
      integer ncounters
      common/ccounters/keywords,counters,ncounters
      integer ini,j
      data ini/0/
      save ini
      if(ini.eq.0) then
         ncounters=0
         ini=1
      endif
      do j=1,ncounters
         if(string.eq.keywords(j)) then
            counters(j)=counters(j)+1
            return
         endif
      enddo
c not found
      if(ncounters.eq.maxnum) then
         write(*,*) ' too many counters requested'
         stop
      endif
      ncounters=ncounters+1
      keywords(ncounters)=string
      counters(ncounters)=1
      end


