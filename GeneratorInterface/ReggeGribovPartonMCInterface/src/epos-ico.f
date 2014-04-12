c-----------------------------------------------------------------------
c  activate making inicon table by the follwing commands in optns file:
c-----------------------------------------------------------------------
c        set nevent 1
c        set bminim 3      (for example)
c        set bmaxim 5      (for example)
c        set ninicon 1000  (for example)
c        set icocore 1  !or 2
c        make icotable
c        switch  splitting on
c        switch  fusion on
c        switch clusterdecay off
c-----------------------------------------------------------------------
c  inicon table means storage of IcoE, IcoV, IcoF
c     IcoE ... energy density in the comoving frame
c     IcoV ... flow velocity in hyperbolic coordinates
c               v=(vx,vy,veta=vz/tau)
c                with the velocity in the frame moving with rapidity eta
c     IcoV ... net flavor density in the comoving frame
c  the indices of these fields ix,iy,iz also refer to hyperbolic coordinates
c     (x,y,eta) at given tau
c     the corresponding grid is defined in epos.incico
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
      subroutine IniCon(nin)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incico'
      double precision seedf
      character*80 fn
      logical table

      ii=index(fnhi(1:nfnhi),".histo")-1
      fn=fnhi(1:ii)//".ico"
      inquire(file=fn(1:ii+4),exist=table)

      if(icotabr.eq.1)then  !read from file

        if(.not.table)then
          write(ifmt,'(//10x,3a//)')'STOP: file '
     *             ,fn(1:ii+4),' not found !!!'
          stop
        endif
        write(ifmt,'(3a)')'read from ',fn(1:ii+4),' ...'
        open(97,file=fn,status='old')
        read(97,*) iversn
        read(97,*) laprojx,maprojx,latargx,matargx
        read(97,*) engyx
        read(97,*) bminimx,bmaximx
        read(97,*) tauicox
        read(97,*) iabs_ninicon_dum
        read(97,*) nxicox,nyicox,nzicox
        read(97,*)xminicox,xmaxicox,yminicox,ymaxicox,zminicox,zmaxicox
        if(laprojx.ne.laproj.or.maprojx.ne.maproj
     * .or.latargx.ne.latarg.or.matargx.ne.matarg)stop'(1)'
        if(engyx.ne.engy)    stop'variables dont match (2)    '
        if(bminimx.ne.bminim.or.bmaximx.ne.bmaxim)
     *                       stop'variables dont match (3)    '
        if(tauicox.ne.tauico)stop'variables dont match (4)    '
        if(nxicox.ne.nxico.or.nyicox.ne.nyico
     * .or.nzicox.ne.nzico)stop'(5)'
        if( xminicox.ne.xminico.or.xmaxicox.ne.xmaxico
     * .or.yminicox.ne.yminico.or.ymaxicox.ne.ymaxico
     * .or.zminicox.ne.zminico.or.zmaxicox.ne.zmaxico)stop'(6)'
        read(97,*) IcoE,IcoV,IcoF
        close(97)

      elseif(icotabr.eq.0)then  ! calculate

        !if(icotabm.eq.1.and.table)then
          !write(ifmt,'(//10x,3a//)')'STOP: file '
          !*             ,fn(1:ii+4),' already exists !!!'
          !stop
        !endif

        if(nin.eq.1)then
          write(ifmt,'(2a,i7,a)')' calculate inicon table ',
     &    'based on',iabs(ninicon),' ico-events ...'
          call IcoStr(1)
        endif

        call ranfgt(seedf)
        if(nin.le.iabs(ninicon).and.mod(nin,modsho).eq.0)
     &        write(ifmt,'(a,i7,a,f6.2,a,d27.16)')' ico-event ',nin
     &                   ,'   bimevt:',bimevt,'   seedf:',seedf
        if(nin.le.iabs(ninicon).and.jwseed.eq.1)then
          open(unit=1,file=fnch(1:nfnch-5)//'see',status='unknown')
          write(1,'(a,i7,a,f6.2,a,d27.16)')' ico-event ',nin
     &                   ,'   bimevt:',bimevt,'   seedf:',seedf
         close(1)
        endif
        !if(nin.le.iabs(ninicon).and.mod(nin,modsho).eq.0)
        !&         print*,'+++++ time: ',timefin-timeini
        seedc=seedf
        call IcoStr(2)

        if(nin.eq.iabs(ninicon))then
          write(ifmt,*)'normalize and diagonalize engy-mom tensor'
           call IcoStr(3)

           if(icotabm.eq.1)then  ! make table
            write(ifmt,'(2a)')' write inicon table into ',fn(1:ii+4)
            open(97,file=fn,status='unknown')
            write(97,*) iversn
            write(97,*) laproj,maproj,latarg,matarg
            write(97,*) engy
            write(97,*) bminim,bmaxim
            write(97,*) tauico
            write(97,*) iabs(ninicon)
            write(97,*) nxico,nyico,nzico
            write(97,*) xminico,xmaxico,yminico,ymaxico,zminico,zmaxico
            write(97,*) IcoE,IcoV,IcoF
            close(97)
          endif
         endif

      else

        stop'IniCon: wrong choice'

      endif

      end

c-----------------------------------------------------------------------
      subroutine IcoStr(iflag)
c-----------------------------------------------------------------------
c     energy momentum tensor and flavor current
c     and corresponding contractions from string method
c
c     iflag = 1 initialization
c           = 2 sum up density
c           = 3 normalization
c
c     output: common blocks /Ico3/  /Ico4/  /Ico5/
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incico'
      double precision IcoT(4,4,nxico,nyico,nzico)
     *                ,IcoC(3,4,nxico,nyico,nzico)
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common /Ico10/ntot,ish0,irs
      double precision p(4),v(4),u(4),eps,T(4,4),gamma,rap,rapx
      integer ic(2),jc(nflav,2)
      common/cranphi/ranphi
      logical ok
      goto (1,10,100),iflag
      return

c......................................................................
c                             initialization
c......................................................................

 1    do ix=1,nxico
        do iy=1,nyico
          do iz=1,nzico
            IcoE(ix,iy,iz)=0.
            do i=1,3
              IcoV(i,ix,iy,iz)=0.
            enddo
            do i=1,3
              IcoF(i,ix,iy,iz)=0.
            enddo
            do i=1,4
            do j=1,4
              IcoT(i,j,ix,iy,iz)=0.
            enddo
            enddo
            do i=1,3
            do j=1,4
              IcoC(i,j,ix,iy,iz)=0.
            enddo
            enddo
          enddo
        enddo
      enddo
      ntot=0
      return

c......................................................................
c                        fill arrays
c......................................................................

 10   ntot=ntot+1
      dxico=(xmaxico-xminico)/nxico
      dyico=(ymaxico-yminico)/nyico
      dzico=(zmaxico-zminico )/nzico
      phinll=phievt+ranphi

      do j=1,nptl
       ok=.true.
       if(icocore.eq.2
     . .and.(ityptl(j).lt.20.or.ityptl(j).ge.40))ok=.false.
       if(ok
     . .and.dezptl(j).lt.1e3.and.j.le.maxfra.and.istptl(j).eq.2)then
        aa=cos(phinll)
        bb=sin(phinll)
        x=xptl(j)*aa+yptl(j)*bb
        cc=-sin(phinll)
        dd=cos(phinll)
        y=xptl(j)*cc+yptl(j)*dd
        z=dezptl(j)
        ix=1+(x-xminico)/dxico
        iy=1+(y-yminico)/dyico
        iz=1+(z-zminico)/dzico
        if(ix.ge.1.and.ix.le.nxico.and.
     .  iy.ge.1.and.iy.le.nyico.and.
     .  iz.ge.1.and.iz.le.nzico)then
          !~~~~~~~~~~~~
          ! T^i^k = \int d3p p^i p^k /E f(\vec x,\vec p)
          ! N^k =   \int d3p p^k /E f_net(\vec x,\vec p)
          !~~~~~~~~~~~~
          rapx=zminico+(iz-0.5)*dzico
          amt2=pptl(5,j)**2+pptl(1,j)**2+pptl(2,j)**2
          pp=pptl(4,j)+abs(pptl(3,j))
          if(amt2.gt.0..and.pp.gt.0.)then
           amt=sqrt(amt2)
           rap=sign(1.,pptl(3,j))*alog(pp/amt)
           p(1)=pptl(1,j)
           p(2)=pptl(2,j)
           p(3)=amt*sinh(rap-rapx)
           p(4)=amt*cosh(rap-rapx)
           do i=1,4
            do k=1,4
              IcoT(i,k,ix,iy,iz)=IcoT(i,k,ix,iy,iz)+p(i)*p(k)/p(4)
            enddo
           enddo
           id=idptl(j)
           ida=iabs(id/10)
           ids=id/iabs(id)
           if(ida.ne.111.and.ida.ne.222.and.ida.ne.333)id=id/10*10
           if(ida.eq.111.or. ida.eq.222.or. ida.eq.333)id=id/10*10+ids
           if(ida.eq.213)id=1230*ids
           ic(1)=idtrai(1,id,1)
           ic(2)=idtrai(2,id,1)
           call iddeco(ic,jc)
           do i=1,3
            fi=jc(i,1)-jc(i,2)
            do k=1,4
             IcoC(i,k,ix,iy,iz)=IcoC(i,k,ix,iy,iz)+p(k)/p(4)*fi
            enddo
           enddo
          endif
          !~~~~~~~~~~~~
        endif
       endif
      enddo

      return

c............................................................................
c                 normalization and diagonalization
c............................................................................

 100  continue

      !~~~normalize

      vol= (xmaxico-xminico)/nxico
     .    *(ymaxico-yminico)/nyico
     .    *(zmaxico-zminico)/nzico  *tauico
      fac=1./ntot/vol

      do ix=1,nxico
        do iy=1,nyico
          do iz=1,nzico
            do i=1,4
             do k=1,4
              IcoT(i,k,ix,iy,iz)=IcoT(i,k,ix,iy,iz)*fac
             enddo
            enddo
            do i=1,3
             do k=1,4
              IcoC(i,k,ix,iy,iz)=IcoC(i,k,ix,iy,iz)*fac
             enddo
            enddo
          enddo
        enddo
      enddo

      !~~~diagonalize T

      do ix=1,nxico
        do iy=1,nyico
          do iz=1,nzico
            do i=1,4
            do k=1,4
             T(i,k)=IcoT(i,k,ix,iy,iz)
            enddo
            enddo
            call DiagTmunu(T,u,v,gamma,eps,nonz,ix,iy,iz)
            if(nonz.eq.1)then
              IcoE(ix,iy,iz)=eps
              IcoV(1,ix,iy,iz)=v(1)
              IcoV(2,ix,iy,iz)=v(2)
               !rapx=zminico+(iz-0.5)*dzico
               !rap=rapx+log(gamma*(1+v(3)))
               !IcoV(3,ix,iy,iz)=tanh(rap)
              IcoV(3,ix,iy,iz)=v(3) / tauico
              do i=1,3
               IcoF(i,ix,iy,iz)=IcoC(i,4,ix,iy,iz)*u(4)
               do  k=1,3
                IcoF(i,ix,iy,iz)=IcoF(i,ix,iy,iz)
     .                  -IcoC(i,k,ix,iy,iz)*u(k)
               enddo
              enddo
            endif
          enddo
        enddo
      enddo

      return
      end

c------------------------------------------------------------------------------------
      subroutine DiagTmunu(Tmunu,u,v,gamma,eps,nonz,ix,iy,iz)
c------------------------------------------------------------------------------------
      ! solve lambda * T * lambda = diag(eps,P,P,P), for symmetric T
      !
      !  lambda =  g      g*vx        g*vy        g*vz          =symmetric!
      !            g*vx   a*vx*vx+1   a*vy*vx     a*vz*vx
      !            g*vy   a*vx*vy     a*vy*vy+1   a*vz*vy
      !            g*vz   a*vx*vz     a*vy*vz     a*vz*vz+1
      ! with g=gamma, a=g*g/(g+1)
      !
      ! so T*lambda(v)=lambda(-v)*diag(eps,P,P,P)
      ! first column: four equations
      !                    eps=T00+T0k*vk
      !                -eps*vi=Ti0+Tik*vk
      ! solved iteratively to get eps, vi
      ! returns u,v,eps if nonz=1 (otherwise zero tensor)
      !  (by T.Kodama)
      !  (K. Werner: modifs)

      include 'epos.incico'
      double precision Tmunu(4,4), u(4),eps,gamma,g,a, Lor(4,4),w(4),sum
      double precision v(4),vx(3),tt(4,4),err,sg(4)

      sg(4)=1d0
      v(4)=1d0
      do i=1,3
       sg(i)=-1d0
      enddo
      sum=0d0
      do i=1,4
       do k=1,4
       sum=sum+dabs(Tmunu(i,k))
       end do
      end do
      nonz=0
      if(sum.eq.0.0d0)return
      nonz=1

      do k=1,3
       v(k)=0.
      end do
      eps=0

      do lrep=1,100
       epsx=eps
       do i=1,3
        vx(i)=v(i)
       end do
       eps=Tmunu(4,4)
       do k=1,3
        eps=eps-Tmunu(4,k)*vx(k)
       end do
       if(eps.le.0d0)then
         print*,'DiagTmunu: sum(abs(Tmunu))=',sum,'   eps=',eps
         print*,'Tmunu(4,nu):',(Tmunu(4,nu),nu=1,4)
         if(eps.gt.-1e-5)then
           nonz=0
           return
         else
           print*,'STOP in DiagTmunu: negative energy density.  '
           stop'(3003200808)'
         endif
       endif
       do i=1,3
        Tv=0d0
        do k=1,3
         Tv=Tv+Tmunu(i,k)*vx(k)
        end do
        v(i)=(Tmunu(i,4)-Tv)/eps
       end do
       if(lrep.gt.60)then
        do i=1,3
         v(i)=0.5d0*(vx(i)+v(i))
        enddo
       endif
       !print*,'Tmunu: ',lrep,abs(eps-epsx),(abs(v(i)-vx(i)),i=1,3)
       err=1d-6
       if(lrep.gt.50)err=1d-5
       if(lrep.gt.89)err=1d-4
       if(abs(eps-epsx).lt.err.and.abs(v(1)-vx(1)).lt.err
     . .and.abs(v(2)-vx(2)).lt.err.and.abs(v(3)-vx(3)).lt.err)goto1
        do i=1,4
          w(i)=0
          do k=1,4
          w(i)=w(i)+Tmunu(i,k)*v(k)*sg(k)
          enddo
          w(i)=w(i)-eps*v(i)
        enddo
        if(lrep.gt.95
     ..and.w(1)*w(1)+w(2)*w(2)+w(3)*w(3)+w(4)*w(4).lt.err)goto1
      end do

  1   v2=0.d0
      do i=1,3
       v2=v2+v(i)**2
      end do
      if(v2.ge.1.)stop'DiagTmunu: v2 ge 1.          '
      gamma=1./sqrt(abs(1.-v2))
      u(4)=gamma
      u(1)=v(1)*gamma
      u(2)=v(2)*gamma
      u(3)=v(3)*gamma

      !~~~check
      g=gamma
      a=g*g/(g+1d0)
      Lor(4,4)=g
      do k=1,3
       Lor(4,k)=-g*v(k)
       Lor(k,4)=Lor(4,k)
      enddo
      do i=1,3
      do k=1,3
       Lor(i,k)=a*v(i)*v(k)
      enddo
      enddo
      do k=1,3
       Lor(k,k)=Lor(k,k)+1
      enddo
      do i=1,4
      do k=1,4
        tt(i,k)=0d0
        do m=1,4
        do n=1,4
          tt(i,k)=tt(i,k)+Lor(i,m)*Tmunu(m,n)*Lor(n,k)
        enddo
        enddo
      enddo
      enddo
      err=err*1d2
      if(tt(1,4).gt.err.or.tt(2,4).gt.err.or.tt(3,4).gt.err)then
        print*,'************** DiagTmunu: nonzero T14 or T24 or T34'
     .  ,'  after ',lrep,' iterations'
        print*,'d_eps or d_v(i): ',abs(eps-epsx),(abs(v(i)-vx(i)),i=1,3)
        print*,'Tmunu ( ix=',ix-nxico/2-1,'  iy=',iy-nyico/2-1,' ) :'
     .  ,'  iz=',iz-nzico/2-1
        do i=1,4
        write(*,'(4f10.5,2x,4f10.5)')
     .  (sngl(tmunu(i,k)),k=1,4),(sngl(tt(i,k)),k=1,4)
        enddo
      endif

      end

