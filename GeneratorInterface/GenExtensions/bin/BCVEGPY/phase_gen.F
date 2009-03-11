c********************************************************************
c********************************************************************
c*** for phase-space
c*** phase_gen() ---- generate the phase-space
c*** lorentz() ---- doing momentum transform.
c********************************************************************
c********************************************************************

c...from program rambos, which is a democratic multi-particle phase 
c...space generator.  authors:  s.d. ellis,  r. kleiss,  w.j. stirling
c...this is version 1.0 - written by r. kleiss modified slightly by 
c...ian hinchliffe. here we make little changes and let it only adaptable
c...to three final particals. the interesting reader can change back to
c...it's original form easily.

      subroutine phase_gen(y,et,wt)
      implicit double precision (a-h,o-z)
	implicit integer (i-n)

	double complex colmat,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/rconst/pi
      dimension xm(3),p(4,3),q(4,3),r(4),z(5),
     &   b(3),p2(3),xm2(3),e(3),v(3),y(5)
c      data acc/1.0d-14/,itmax/6/,in_it/0/,pi2log/1.837877d0/
      
      acc=1.0d-14
	itmax=6
	in_it=0
	pi2log=1.837877d0
      
	xm(1)=pmc
	xm(2)=pmb
	xm(3)=pmbc

	n=3

      if(in_it.eq.0) then
        in_it=1
        z(1)=0.0d0
        do k=2,5
          z(k)=z(k-1)+dlog(dfloat(k))
        end do
      end if

c count nonzero masses
      xmt=0.0d0
      nm=0
      do i=1,n
        if(xm(i) .gt. 1.0d-6) nm=nm+1
        xmt=xmt+dble(abs(xm(i)))
      end do

c generate n massless momenta in infinite phase space
      extra_weight=1.0d0
      do i=1,n-1
        if(i.eq.1)then
          c=2.0d0*y(i)-1.0d0
          s=dsqrt(1.0d0-c*c)
c...there is one redundant overall phi that can be chosing arbitrarily. 
c...by chosing the original f=0, we will always get
c...q(1,1)=0, which corresponds to a specific receive direction of the
c...final particles. this breaks the isotropy of the space, to avoid this, 
c...we may choose f=2*pi*pyr(0) .
c          f=0.0d0
          f=2*pi*pyr(0)
		fac=y(i+1)
        else
          c=2.0d0*y(i+1)-1.0d0
          s=dsqrt(1.0d0-c*c)
          f=2.0d0*pi*y(i+2)
          fac=y(i+3)
        end if
        q(4,i)=-dlog(fac)
        extra_weight=extra_weight*q(4,i)
        q(3,i)=q(4,i)*c
        q(2,i)=q(4,i)*s*cos(f)
        q(1,i)=q(4,i)*s*sin(f)
      end do

c calculate the parameters of the scale transformation
      do i=1,4
        r(i)=0.0d0
      end do
      do i=1,n-1
        do k=1,4
          r(k)=r(k)+q(k,i)
        end do
      end do

c calculate the n_th vector
      do k=1,3
        q(k,n)=-r(k)
      end do
      q(4,n)=dsqrt(q(1,n)**2+q(2,n)**2+q(3,n)**2)
      r(4)=r(4)+q(4,n)
      rmas=r(4)
      if(rmas .lt. 1.0d-6)then
        wt=0.d0
        return
      end if
      x=et/rmas

c scale the q's to the p's
      do i=1,n
        do k=1,4
          p(k,i)=x*q(k,i)
        end do
      end do

c return for weighted massless momenta
      wt=pi2log*(n-1)-dlog(et-p(4,n))*2.0d0*(1-n)-dlog(et)-
     &	dlog(2.0d0)-dlog(p(4,n))-z(2*n-3)
      if(nm.eq.0) then
        wt=extra_weight*dexp(wt)
        return
      end if

c massive particles: rescale the momenta by a factor x
      xmax=dsqrt(1.0d0-(xmt/et)**2)
      do i=1,n
        xm2(i)=xm(i)**2
        p2(i) =p(4,i)**2
      end do
      iter=0
      x=xmax
      accu=et*acc
      do while (iter.le.itmax)
        f0=-et
        g0=0.d0
        x2=x*x
        do i=1,n
          e(i)=dsqrt(xm2(i)+x2*p2(i))
          f0=f0+e(i)
          if(e(i) .gt. 0.0d0)then
            g0=g0+p2(i)/e(i)
          end if
        end do
        if(dble(abs(f0)).gt.accu.and.iter.le.itmax) then
          iter=iter+1
          x=x-f0/(x*g0)
        else if(dble(abs(f0)).le.accu) then
          iter=itmax+1
        end if
      end do

      do i=1,n
        v(i)=x*p(4,i)
        do k=1,3
          p(k,i)=x*p(k,i)
        end do
        p(4,i)=e(i)
      end do

c calculate the mass-effect weight factor
      wt2=1.0d0
      wt3=0.0d0
      do i=1,n
        if(e(i) .gt. 0.0d0)then
          wt2=wt2*v(i)/e(i)
          wt3=wt3+v(i)**2/e(i)
        end if
      end do
      wtm=(2.0d0*n-3.0d0)*dlog(x)+dlog(wt2/wt3*et)

c return for weighted massive momenta
      wt=wt+wtm
      wt=dexp(wt)*extra_weight
      if(xmt .gt. et)then
        wt=0.0d0
        return
      end if

c...b_c 
      pmomup(5,3)=xm(3)
      do j=1,4
	   pmomup(j,3)=p(j,3)
      end do

c...b quark
      pmomup(5,4)=xm(2)
      do j=1,4
	   pmomup(j,4)=p(j,2)
      end do
     
c...\bar{c} quark     
      pmomup(5,5)=xm(1)
      do j=1,4
	   pmomup(j,5)=p(j,1)
      end do

      return
      end

c**********************************************************

c...this performs a lorentz transformation
      subroutine lorentz(pb,pc,pl)
      implicit double precision (a-h,o-z)
      implicit integer(i-n)

      dimension pb(4),pc(4),pl(4)

      q    =sqrt(pb(4)**2-pb(1)**2-pb(2)**2-pb(3)**2)
	pl(4)=(pb(4)*pc(4)+pb(3)*pc(3)+pb(2)*pc(2)+pb(1)*pc(1))/q
      f    =(pl(4)+pc(4))/(q+pb(4))
      do  j=1,3
        pl(j)=pc(j)+f*pb(j)
      end do

      return
      end
      
