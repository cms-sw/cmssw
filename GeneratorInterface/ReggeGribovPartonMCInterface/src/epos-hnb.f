c----------------------------------------------------------------------
      subroutine hnbaaa(ip,iret)  !former hnbaaa156 from epos-yyy
c----------------------------------------------------------------------
c  microcanonical decay of cluster ip via loop over hnbmet
c----------------------------------------------------------------------
      include 'epos.inc'
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/citer/iter,itermx
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      common /cttaus/  tpro,zpro,ttar,ztar,ttaus,detap,detat
      integer jc(nflav,2)
      double precision p(5),c(5)
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      dimension be(4),pe(5),pa(5)
      common/yradx/yrad(maxp),phirad(maxp)
      common/xxxspecsy/ndrop(-4:4,-4:4,-4:4)
      common/cdelzet/delzet,delsgr /cvocell/vocell
      call utpri('hnbaaa',ish,ishini,4)


      ntry=0
 10   continue
      ntry=ntry+1

      if(ish.ge.3)then
      write(ifch,140)sngl(ttaus)
  140 format(/' ----------------------------------'/
     *'    droplet decay  at tau =',f6.2/
     *' ----------------------------------')
      write(ifch,*)'droplet:'
      call alist('&',ip,ip)
      endif

      iret=0
      do j=1,5
      c(j)=pptl(j,ip)
      enddo

      call idquac(ip,nqi,nsi,nai,jc)
      keu=jc(1,1)-jc(1,2)
      ked=jc(2,1)-jc(2,2)
      kes=jc(3,1)-jc(3,2)
      kec=jc(4,1)-jc(4,2)
      keb=jc(5,1)-jc(5,2)
      ket=jc(6,1)-jc(6,2)
c      write(ifch,*)'droplet uds=',keu,ked,kes,'   E=',pptl(5,ip)

    !~~~~~redefine energy in case of radial flow
      amin=utamnu(keu,ked,kes,kec,keb,ket,4)   !utamnu(...,4) and not utamnu(...,5)
      aumin=amuseg+yrmaxi        !for rad and long flow  
      ipo=ip                                   !could be too light after flow
      if(ityptl(ip).eq.60)ipo=iorptl(ip)
      tecmor=pptl(5,ipo)
      if(iappl.eq.4.or.iorsdf.ne.3
     &.or.ityptl(ip).eq.40.or.ityptl(ip).eq.50)then !not for droplets from remnants
        yrmax=0
      else
        yrmax=yrmaxi
                  !aumin=amin
                  !if(yrmax.gt.0.2)print*,'===',tecmor,aamin,yrmax
      endif
      fradflo=1.
      if(yrmax.gt.1e-2)fradflo=fradflii
      tecm=pptl(5,ip)
      if(tecm.lt.amin)then
        iret=1
        if(ish.ge.4)write(ifch,*)'Decay skipped (M too low) !'
     &                           ,tecm,amin
        goto 1000
      endif
      if(iappl.eq.4.or.iorsdf.ne.3
     &.or.ityptl(ip).eq.40.or.ityptl(ip).eq.50)then !not for droplets from remnants
        yco=0
      else
       if(ylongmx.lt.0.)then
        yco=delzet !* 1.75
       else
        yco=ylongmx
       endif
      endif
      corrco=1.
      if(yco.gt.0.)corrco=sinh(yco)/yco


      tecmxx=tecm
      if(iLHC.eq.1)then
        corr=fradflo/corrco
        if(tecm*corr.lt.amin.and.tecm.gt.0.)then
          fradflo=min(1.,1.1*amin/tecm*corrco) !if flow too large, do something anyway (saturation of flow)
          corr=fradflo/corrco
        endif
      else
        if(tecm*fradflo.lt.amin.and.tecm.gt.0.)fradflo=1.1*amin/tecm  !if flow too large, do something anyway (saturation of flow)
        corr=fradflo
      endif
      if(yrmax.gt.1e-2.and.tecmor.gt.aumin
     &  .and.tecm*corr.ge.amin) then
        ! redefine energy to account for collective flow
        ! \int_0^yrmax f(y) d2y = E_new (effective mass)
        ! \int_0^yrmax cosh(y) f(y) d2y = E_old
        tecm=tecm*fradflo
        if(tecm.lt.amin)stop'aaahnb: small mass. should not happen.   '
      else
        yrmax=0.
      endif
    !~~~~~redefine energy in case of long coll flow
! MANDATORY if RAD FLOW USED BECAUSE IT SMOOTH THE ETA DISTRIBUTION
! because of the grid structure for the cluster, fluctuations in eta
! appears if there is no smearing with long flow !
      tecmx=tecm
c      if(yco.gt.0..and.tecmor.gt.aumin) then
      if(yco.gt.0) then
        if(iLHC.eq.1.and.tecm.ge.aumin)then
          tecm=tecm/corrco
          do while(tecm.lt.amin)
            yco=yco*0.5
            corrco=sinh(yco)/yco
            tecm=tecmx/corrco
          enddo
        else
          tecm=tecm/corrco
          if(tecm.lt.aumin)then  !security to avoid minimum mass
            tecm=tecmx
            yco=0.
          endif
        endif
      else
        yco=0.
      endif
      if(ish.ge.4)write(ifch,*)'===== cluster energy: '
     &                               ,pptl(5,ip),tecmx,tecm,amin,aumin
     &                               ,delzet,yrmax,yco,ityptl(ip)

    !~~~~~~~~~define volume~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      volu=tecm/epscri(1)

    !~~~~~~~~~decay~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      call hnbini(iret)
      !if(iret.ne.0)write(ifch,*)'***** unsucessfull hnbini *****'
      if(iret.ne.0)goto 1000
      if(ioinct.ge.1)goto 1

      do iter=1,itermx
        naccit(iter)=0
        call hnbmet
      enddo

1     continue

      if(ioceau.eq.1.and.iappl.eq.1)call xhnbte(ip)

    !~~~~~~~~~~long coll flow -> particles~~~~~~~~~~~~~~~~
      tecmxxx=tecm
      if(yco.gt.0.) then
        errlim=0.0001
        tecm=tecmx
        niter=0
 611    energ=0.
        niter=niter+1
        do i=1,np
          yrad(i)=(2*rangen()-1)*yco
          be(3)=sinh(yrad(i))
          be(4)=cosh(yrad(i))
          energ=energ+be(4)*pcm(4,i)-be(3)*pcm(3,i)
        enddo
        if(abs(energ-tecm).gt.0.1.and.niter.lt.1000)then
          goto 611
        elseif(niter.ge.1000)then
          if(ish.ge.1)write(ifch,*)'Long Flow failed:'
     &                             ,energ,tecm
          yco=0
          tecm=tecmxxx
          goto 400
        endif
                !print*,'===== energy after flow boosts',energ,'   soll: ',tecm
        do j=1,4
          pe(j)=0.
        enddo
        do i=1,np
          be(1)= 0
          be(2)= 0
          be(3)= sinh(yrad(i))
          be(4)= cosh(yrad(i))
          call utlob3(1,be(1),be(2),be(3),be(4),1e0
     *         , pcm(1,i), pcm(2,i), pcm(3,i), pcm(4,i))
          do j=1,4
          pe(j)=pe(j)+pcm(j,i)
          enddo
        enddo
        pe(5)=sqrt(pe(4)**2-pe(3)**2-pe(2)**2-pe(1)**2)
       !write(6,'(a,5e11.3)')'flow boosts',pe
        do j=1,4
          pa(j)=0.
        enddo
        do i=1,np
          call utlob3(1,pe(1),pe(2),pe(3),pe(4),pe(5)
     *         , pcm(1,i), pcm(2,i), pcm(3,i), pcm(4,i))
          do j=1,4
            pa(j)=pa(j)+pcm(j,i)
          enddo
        enddo
        pa(5)=sqrt(pa(4)**2-pa(3)**2-pa(2)**2-pa(1)**2)
                !write(6,'(a,5e11.3)')' cms boost ',pa
        esoll=tecm
        scal=1.
        do ipass=1,200
          sum=0.
          do  j=1,np
            do k=1,3
              pcm(k,j)=scal*pcm(k,j)
            enddo
            pcm(4,j)=sqrt(pcm(1,j)**2+pcm(2,j)**2+pcm(3,j)**2
     *           +amass(j)**2)
            sum=sum+pcm(4,j)
          enddo
          scal=esoll/sum
          !write(6,*)'ipass,scal,e,esoll:'
          !    $         ,ipass,scal,sum,esoll
          if(abs(scal-1.).le.errlim) goto301
        enddo
 301    continue
        do j=1,4
          pa(j)=0.
        enddo
        do i=1,np
          do j=1,4
            pa(j)=pa(j)+pcm(j,i)
          enddo
        enddo
        pa(5)=sqrt(pa(4)**2-pa(3)**2-pa(2)**2-pa(1)**2)
        !write(6,'(a,5e11.3)')' rescaling ',pa
      endif

 400  continue
    !~~~~~~~~~~radial flow -> particles~~~~~~~~~~~~~~~~~~
      if(yrmax.gt.0.) then
        fecc=0
        aa=1
        bb=0
        cc=0
        dd=1
        if(ityptl(ip).eq.60)then
          ipo=iorptl(ip)
          xx=uptl(ipo)   ! <x**2>
          yy=optl(ipo)   ! <y**2>
          xy=desptl(ipo) ! <x*y>
          dta=0.5*abs(xx-yy)
          ev1=(xx+yy)/2+sqrt(dta**2+xy**2)
          ev2=(xx+yy)/2-sqrt(dta**2+xy**2)
          if(xy.lt.0..and.dta.ne.0.)then
            theta=0.5*atan(-xy/dta)
          elseif(xy.gt.0..and.dta.ne.0.)then
            theta=-0.5*atan(xy/dta)
          else
            theta=0
          endif
c          if(dta.ne.0.)then
c            theta=0.5*atan(abs(xy)/abs(dta))
c            if(    xy.gt.0..and.dta.gt.0.)then
c              theta=theta
c            elseif(xy.lt.0..and.dta.gt.0.)then
c              theta=-theta
c            elseif(xy.gt.0..and.dta.lt.0.)then
c              theta=pi-theta
c            elseif(xy.lt.0..and.dta.lt.0.)then
c              theta=theta-pi
c            endif
c          else
c            theta=2.*pi*rangen()
c          endif
          !eccx=(yy-xx)/(yy+xx)
          yy=ev1
          xx=ev2
          ecc=(yy-xx)/(yy+xx)
c          print*,'AA',ecc,theta
          if(iLHC.eq.1)then
            fecc=min(facecc,ecc) !be careful : fecc change <pt> since it is the elliptical deformation of the sub cluster(give strength of v2)
          else
            fecc=facecc*ecc
            fecc=min(0.3,fecc) !be careful : fecc change <pt> since it is the 
          endif
          phiclu=mod(phievt+theta,2.*pi) !do not change otherwise v2 is gone
          if(phiclu.lt.-pi)phiclu=phiclu+2*pi
          if(phiclu.gt.pi)phiclu=phiclu-2*pi
          aa=cos(phiclu)
          bb=sin(phiclu)
          cc=-sin(phiclu)
          dd=cos(phiclu)
        endif
        errlim=0.0001
        tecm=tecmxx
        niter=0
 610    energ=0.
        niter=niter+1
        do i=1,np
          yrad(i)=sqrt(rangen())
          phirad(i)=2.*pi*rangen()
          pt2=0.
          if(iLHC.eq.1)pt2=(pcm(1,i)**2+pcm(2,i)**2) !+amass(i)**2)
          bex=dsinh(dble(yrad(i)*yrmax))*cos(phirad(i))
     *       *(1+fecc/(1.+pt2))
          bey=dsinh(dble(yrad(i)*yrmax))*sin(phirad(i))
     *       *(1-fecc/(1.+pt2))
          be(1)=aa*bex+cc*bey
          be(2)=bb*bex+dd*bey
          be(3)=-0d0
          be(4)=sqrt(1+be(1)**2+be(2)**2)
          bp=0d0
          do k=1,3
            bp=bp+pcm(k,i)*be(k)
          enddo
          en=be(4)*pcm(4,i)+bp
          energ=energ+en
c          bp=sqrt(pcm(1,i)**2+pcm(2,i)**2)**yradpp
cc          bp=sqrt(amass(i)**2+pcm(1,i)**2+pcm(2,i)**2)**yradpp
c          be(1)=pcm(1,i)-bp*(aa*bex+cc*bey)
c          be(2)=pcm(2,i)-bp*(bb*bex+dd*bey)
c          be(4)=sqrt(be(1)**2+be(2)**2+pcm(3,i)**2
c     *           +amass(i)**2)
c          en=be(4)
c          energ=energ+en
        enddo
        if(abs(energ-tecm).gt.0.1.and.niter.lt.1000)then
          goto 610
        elseif(niter.ge.1000)then
          if(ish.ge.1)write(ifch,*)'Radial Flow failed:'
     &                             ,yrmax,energ,tecm,np
          iret=1
          if(ish.ge.1)write(ifch,*)'Decay skipped !'
          goto 1000
c          goto 300
        endif
        energ=0.
        do i=1,np
          pt2=0.
          if(iLHC.eq.1)pt2=(pcm(1,i)**2+pcm(2,i)**2)!+amass(i)**2)
          bex=dsinh(dble(yrad(i)*yrmax))*cos(phirad(i))
     *       *(1+fecc/(1.+pt2))
          bey=dsinh(dble(yrad(i)*yrmax))*sin(phirad(i))
     *       *(1-fecc/(1.+pt2))
          be(1)=aa*bex+cc*bey
          be(2)=bb*bex+dd*bey
          be(3)=0d0
          be(4)=sqrt(1+be(1)**2+be(2)**2)
          call utlob3(1,be(1),be(2),be(3),be(4),1e0
     *         , pcm(1,i), pcm(2,i), pcm(3,i), pcm(4,i))
c          bp=sqrt(pcm(1,i)**2+pcm(2,i)**2)**yradpp
cc          bp=sqrt(amass(i)**2+pcm(1,i)**2+pcm(2,i)**2)
c          pcm(1,i)=pcm(1,i)-bp*(aa*bex+cc*bey)
c          pcm(2,i)=pcm(2,i)-bp*(bb*bex+dd*bey)
c          pcm(4,i)=sqrt(pcm(1,i)**2+pcm(2,i)**2+pcm(3,i)**2
c     *           +amass(i)**2)
          energ=energ+pcm(4,i)
        enddo
        esoll=tecm
        scal=1.
        do ipass=1,200
          sum=0.
          do  j=1,np
            do k=1,3
              pcm(k,j)=scal*pcm(k,j)
            enddo
            pcm(4,j)=sqrt(pcm(1,j)**2+pcm(2,j)**2+pcm(3,j)**2
     *           +amass(j)**2)
            sum=sum+pcm(4,j)
          enddo
          scal=esoll/sum
c          write(6,*)'ipass,scal,e,esoll:'
c     $         ,ipass,scal,sum,esoll
          if(abs(scal-1.).le.errlim) goto300
        enddo
 300    continue
      else
        do n=1,np
          yrad(n)=0.
          phirad(n)=0.
        enddo
      endif
    !~~~~~~~~~~~~~~~

      nptlb=nptl
      do n=1,np
        nptl=nptl+1
        if(nptl.gt.mxptl)call utstop('hnbptl: mxptl too small&')
        idptl(nptl)=ident(n)
        do j=1,4
          p(j)=pcm(j,n)
        enddo
        p(5)=amass(n)
        call utlob2(-1,c(1),c(2),c(3),c(4),c(5),p(1),p(2),p(3),p(4),10)
        do j=1,5
          pptl(j,nptl)=p(j)
        enddo
        if(tecmor.gt.aumin)then
          ityptl(nptl)=60
        elseif(ityptl(ip).eq.40.or.ityptl(ip).eq.50)then
          ityptl(nptl)=ityptl(ip)+1
        else
          ityptl(nptl)=19
        endif
        ipo=iorptl(ip)
        iorptl(nptl)=ip
        jorptl(nptl)=ipo
c protection against very high momentum particle (it can happen very very boosted cluster (which do no really make sense anyway))
        if(iLHC.eq.1.and.p(4).ge.0.5*engy)then
          if(ish.ge.4)call alist('&',nptlb+1,nptl)
          nptl=nptlb
          iret=1
          if(ish.ge.4)write(ifch,*)'Decay skipped (p4 too high) !', ntry
          if(ntry.lt.10)goto 10
          goto 1000
        endif
        if(ityptl(ip).eq.60)then
          if(ityptl(nptl).eq.60)then
            xx=uptl(ipo)        ! <x**2>
            yy=optl(ipo)        ! <y**2>
            rini=sqrt(5./3.*(xx+yy)) !<r**2>=3/5*R**2 for sphere of radius R
            r=1.15*rini*yrad(n) !yrad=y/ymax
            tau=2.25/sqrt(yrad(n)**2+0.04)-0.75
            z=xorptl(3,ipo)
            t=xorptl(4,ipo)
           !zeta=0.5*log((t+z)/(t-z))-0.5*delzet+2*0.5*delzet*rangen()
            zeta=0.5*log((p(4)+p(3))/(p(4)-p(3)))
            z=tau*sinh(zeta)
            t=tau*cosh(zeta)
            xorptl(1,nptl)=xorptl(1,ipo)+r*cos(phirad(n))
            xorptl(2,nptl)=xorptl(2,ipo)+r*sin(phirad(n))
            xorptl(3,nptl)=z
            xorptl(4,nptl)=t
          else
            xorptl(1,nptl)=xorptl(1,ip)
            xorptl(2,nptl)=xorptl(2,ip)
            xorptl(3,nptl)=xorptl(3,ip)
            xorptl(4,nptl)=xorptl(4,ip)
          endif
        endif
      enddo

      if(ish.ge.4)then
        write(ifch,*)'decay products:'
        call alist('&',nptlb+1,nptl)
        if(ish.ge.5)then
          write(ifch,*)'momentum sum:'
          do kk=1,5
            pptl(kk,nptl+1)=0
            do ii=nptlb+1,nptl
              pptl(kk,nptl+1)=pptl(kk,nptl+1)+pptl(kk,ii)
            enddo
            pptl(kk,nptl+2)=c(kk)
          enddo
          call alist('&',nptl+1,nptl+2)
        endif
      endif

 1000 continue
      call utprix('hnbaaa',ish,ishini,4)
      return
      end


c####################################################################################
c####################################################################################
c####################################################################################
c####################################################################################
c####################################################################################
c#########                                                                  #########
c#########                  hnb and hgc routines                            #########
c#########                                                                  #########
c####################################################################################
c####################################################################################
c####################################################################################
c####################################################################################
c####################################################################################


c-----------------------------------------------------------------------
      subroutine hgcaaa
c-----------------------------------------------------------------------
c hadronic resonance gas in grand canonical treatment
c returns T, chemical potentials and hadronic yield
c (hadron chemical potentials as combinations of quark chemical potentials)
c
c input:
c   iostat: 1: Boltzmann approximation, 0: quantum statistics  /metr3/
c   tecm:                    droplet energy      /confg/
c   volu:                    droplet volume      /confg/
c   keu ked kes kec keb ket: net flavor number   /drop5/
c
c output:
c   tem    : temperature [GeV]                            /cgchg/
c   chem(1:nflav): quark chem. pot. [GeV]                 /cflav/
c   chemgc(1:nspecs): hadron chem. pot. [GeV]             /cgchg/
c   ptlngc(1:nspecs): hadron number                       /cgchg/
c   rmsngc(1:nspecs): standard deviation of hadron number /cgchg/
c
c exact treatment (iostat=0):
c for massive hadrons     : first in Boltzmann approximation with analytical
c                           expressions for particle and energy densities,
c                           then by using quantum statistics in integral form,
c                           extracting mu and T using numerical integration
c                           and an iterative procedure to solve for mu, T
c for massless hadrons    : using analytic expressions for massles particles
c                           and employing the  same algorithm as for massive
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cbol/rmsbol(mspecs),ptlbol(mspecs),chebol(mspecs),tembol
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/ciakt/gen,iafs,ians,genm
      common/cnrit/nrit
      gen=10.0**(-epsgc)
      genm=gen/10.

      isho=ish
      if(ishsub/100.eq.51)ish=mod(ishsub,100)

      iug=(1+iospec)/2*2-1


c     initialization
c     --------------
      kef(1)=keu
      kef(2)=ked
      kef(3)=kes
      kef(4)=kec
      kef(5)=keb
      kef(6)=ket

      if(iug.eq.1)nflavs=1
      if(iug.eq.3)nflavs=2
      if(iug.eq.5)nflavs=2
      if(iug.eq.7)nflavs=3
      if(iug.eq.9)nflavs=3
      if(iug.eq.11)nflavs=3
      tem=0.0
      do i=1,nflavs
      chem(i)=0.0
      enddo
      call hgchac(0)
      do i=1,nspecs
      ptlngc(i)=0.0
      rmsngc(i)=0.0
      enddo
      nrit=0

      if(ish.ge.5)then
        write(ifch,*)('-',l=1,10)
     *,' entry sr hgcaaa ',('-',l=1,30)
        write(ifch,'(1x,a,2x,3i3)')
     *'>>> grand canonical hadron gas for droplet with u d s content:'
     *,keu,ked,kes
        write(ifch,'(1x,a,2x,f7.3,2x,a,2x,f7.3)')
     *'mass [GeV]:',tecm,'volume [fm^3]:',volu
      endif

      if(iug.eq.1.and.keu.ne.0.and.ish.ge.5)then
      write(ifch,*)'inversion impossible !!!'
      write(ifch,*)'keu=0 required for this option'
      write(ifch,*)'T = mu(i) = 0 returned'
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      return
      endif
      if(iug.eq.3.and.(keu+ked).ne.0.and.ish.ge.5)then
      write(ifch,*)'inversion impossible !!!'
      write(ifch,*)'keu+ked=0 required for this option'
      write(ifch,*)'T = mu(i) = 0 returned'
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      return
      endif
      kf=keu+ked+kes+kec+keb+ket
      kf=abs(kf)
      if(kf.ne.0)then
      if(mod(kf,3).ne.0.and.ish.ge.5)then
      write(ifch,*)'inversion impossible !!!'
      write(ifch,*)'sum must be multiple of three'
      write(ifch,*)'T = mu(i) = 0 returned'
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      return
      endif
      endif


c     initial T (m=0, baryon free)
c     -------------------------------
      gfac=0.0

       if(iostat.eq.0.and.iospec.eq.iug)then
      do i=1,nspecs
      igsp=int(gspecs(i))
      if(mod(igsp,2).eq.0)then
      gfac=gfac+7.*gspecs(i)/8.
      else
      gfac=gfac+gspecs(i)
      endif
      enddo
      if(iabs(ispecs(nspecs)).lt.10)gfac=gfac+16.
      tem=(tecm/volu*hquer**3*30./pi**2/gfac)**.25
       else
      do i=1,nspecs
      gfac=gfac+gspecs(i)
      enddo
      if(iabs(ispecs(nspecs)).lt.10)gfac=gfac+16.
      tem=(tecm/volu*hquer**3*pi**2/gfac/3.)**.25
      tem=2.*tem
       endif

      if(ish.ge.5)write(ifch,1)'initial T :',tem
1     format(1x,a,3x,f9.6)

      if(ish.ge.5)write(ifch,*)'iospec: ',iospec

       if(ish.ge.5.and.iospec.ne.iug)then
      write(ifch,*)'inversion in Boltzmann approx. :'
       elseif(ish.ge.5.and.iospec.eq.iug)then
      write(ifch,*)'inversion for massless hadrons :'
       endif

       if(ish.ge.5)then
      if(nflavs.eq.1)write(ifch,'(3x,a,8x,a)')
     *'T:','chemu:'
      if(nflavs.eq.2)write(ifch,'(3x,a,8x,a,5x,a)')
     *'T:','chemu:','chemd:'
      if(nflavs.eq.3)write(ifch,'(3x,a,8x,a,5x,a,5x,a)')
     *'T:','chemu:','chemd:','chems:'
       endif

      k=1
10    continue
      if(ish.ge.9.and.mod(k,10).eq.0)
     *write(ifch,*)'hgc iteration:',k
      if(ish.ge.9)call hgccch(1)

c     search for temperature (chem=const)
c     -----------------------------------
      idt=0
      temo=tem

       if(iospec.eq.iug)then

c     massless particles
c     ------------------
      if(iostat.eq.0)then
      if(ish.ge.9)
     *write(ifch,*)'iteration (massless):',k
      call hgctm0
      elseif(iostat.eq.1)then
      if(ish.ge.9)
     *write(ifch,*)'iteration (Boltzmann, massless):',k
      call hgctbo(ibna)
      if(ibna.eq.1)then
      tem=temo
      goto20
      endif
      endif

       else

c     Boltzmann approxiamtion (massive particles)
c     -------------------------------------------
      if(ish.ge.9)
     *write(ifch,*)'iteration (Boltzmann, massive):',k
      call hgctbo(ibna)
      if(ibna.eq.1)then
      tem=temo
      goto20
      endif

       endif

      if(tem.le.1.e-6.and.ish.ge.5)then
      write(ifch,*)'inversion imposssible'
      write(ifch,*)'T:',tem
      if(ioinco.ge.1)call hnbmin(keu,ked,kes,kec)
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return
      endif

      dt=abs(temo-tem)
      if(dt.le.gen*temo.or.dt.le.genm)idt=1

c     search for chemical potentials (tem=const)
c     ------------------------------------------
      idch=0
      ibna=0

        do iafs=1,nflavs
      chemo=chem(iafs)

       if(iospec.eq.iug)then

c     massless particles
c     ------------------
      if(iostat.eq.0)then
      call hgccm0
      elseif(iostat.eq.1)then
      call hgccbo(ibna)
      endif

       else

c     Boltzmann approxiamtion (massive particles)
c     -------------------------------------------
      call hgccbo(ibna)

       endif

      dch=abs(chemo-chem(iafs))
      if(ish.ge.9)write(ifch,*)'dch:',dch
      if(dch.le.abs(gen*chemo).or.dch.le.genm)idch=idch+1
      if(ibna.eq.1)then
      chem(iafs)=chemo
      call hgchac(0)
      goto20
      endif

        enddo


c     new hadron chem. potentials
c     ---------------------------
      call hgchac(0)


      if(ish.ge.5.and.nflavs.eq.1)
     *write(ifch,'(1x,f8.6,2x,f9.6)')
     *tem,chem(1)
      if(ish.ge.5.and.nflavs.eq.2)
     *write(ifch,'(1x,f8.6,2x,f9.6,2x,f9.6)')
     *tem,chem(1),chem(2)
      if(ish.ge.5.and.nflavs.eq.3)
     *write(ifch,'(1x,f8.6,2x,f9.6,2x,f9.6,2x,f9.6)')
     *tem,chem(1),chem(2),chem(3)
      if(idch.eq.nflavs.and.idt.eq.1)goto20


      k=k+1

       if(k.gt.300)then
       if(ish.ge.5)
     *write(ifch,*)'failure in approximate solution'
      goto20
       endif

      goto10

20    continue
      if(ish.ge.9)call hgccch(0)
      if(ish.ge.5)write(ifch,'(1x,a,1x,f9.6)')'  T  :',tem
      do i=1,nflavs
      if(i.eq.1.and.ish.ge.5)
     *write(ifch,'(1x,a,1x,f9.6)')'chemu:',chem(1)
      if(i.eq.2.and.ish.ge.5)
     *write(ifch,'(1x,a,1x,f9.6)')'chemd:',chem(2)
      if(i.eq.3.and.ish.ge.5)
     *write(ifch,'(1x,a,1x,f9.6)')'chems:',chem(3)
      enddo


c     checking results
c     ----------------
      if(ish.ge.5)call hgcchb

c     particle yield
c     --------------
      call hgcpyi(1)

c     checking flavor conservation
c     ----------------------------
      if(ish.ge.5)call hgccfc

      if(iug.eq.iospec.and.iostat.eq.0)then
      if(ish.ge.5)write(ifch,*)
     *'approximation and exact treatment equal'
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return
      endif

c     continue or return approximate values
c     -------------------------------------
      do i=1,nspecs
      rmsbol(i)=rmsngc(i)
      ptlbol(i)=ptlngc(i)
      chebol(i)=chemgc(i)
      enddo
      tembol=tem
      if(iostat.eq.1)then
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return
      endif


c     quantum statistics
c     ------------------
      if(ish.ge.5)write(ifch,*)'quantum statistics:'
      if(ish.ge.5.and.nflavs.eq.1)write(ifch,'(3x,a,8x,a)')
     *'T:','chemu:'
      if(ish.ge.5.and.nflavs.eq.2)write(ifch,'(3x,a,8x,a,6x,a)')
     *'T:','chemu:','chemd:'
      if(ish.ge.5.and.nflavs.eq.3)write(ifch,'(3x,a,8x,a,6x,a,6x,a)')
     *'T:','chemu:','chemd:','chems:'
      k=1

30    continue
      if(ish.ge.9.and.mod(k,10).eq.0)
     *write(ifch,*)'hgc iteration:',k

c     new temperature
c     ---------------
      idt=0
      temo=tem
      call hgctex
      if(ish.ge.5.and.nflavs.eq.1)
     *write(ifch,'(1x,f10.8,2x,f10.7)')
     *tem,chem(1)
      if(ish.ge.5.and.nflavs.eq.2)
     *write(ifch,'(1x,f10.8,2x,f10.7,2x,f10.7)')
     *tem,chem(1),chem(2)
      if(ish.ge.5.and.nflavs.eq.3)
     *write(ifch,'(1x,f10.8,2x,f10.7,2x,f10.7,2x,f10.7)')
     *tem,chem(1),chem(2),chem(3)

      if(tem.le.1.e-6.and.ish.ge.5)then
      write(ifch,*)'inversion imposssible'
      write(ifch,*)'T:',tem
      call hnbmin(keu,ked,kes,kec)
      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return
      endif

      dt=abs(temo-tem)
      if(dt.le.gen*temo.or.dt.le.genm)idt=1
      if(ish.ge.9)write(ifch,*)'dtem:',dt

c     new quark chem. potentials
c     --------------------------
      idch=0
      do iafs=1,nflavs
      chemo=chem(iafs)
      call hgccex
      dch=abs(chemo-chem(iafs))
      if(ish.ge.9)write(ifch,*)'dche:',dch
      if(dch.le.abs(gen*chemo).or.dch.le.genm)idch=idch+1
      enddo

c     new hadron chem. potentials
c     ---------------------------
      call hgchac(0)

       if(idch.eq.nflavs.and.idt.eq.1)then

      if(ish.ge.5)write(ifch,*)'results:'
      if(ish.ge.5)write(ifch,51)'  T  :',tem
      if(nflavs.ge.1.and.ish.ge.5)write(ifch,51)'chemu:',chem(1)
      if(nflavs.ge.2.and.ish.ge.5)write(ifch,51)'chemd:',chem(2)
      if(nflavs.ge.3.and.ish.ge.5)write(ifch,51)'chems:',chem(3)
51    format(1x,a,3x,f9.6)

c     checking results
c     ----------------
      if(ish.ge.5)call hgcchh(i)

c     particle yield
c     --------------
      call hgcpyi(0)

c     checking flavor conservation
c     ----------------------------
      call hgccfc

      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return
       endif

       if(k.gt.300)then
       if(ish.ge.5)
     *write(ifch,*)'failure in exact solution'
      if(ish.ge.5)write(ifch,*)'results:'
      if(ish.ge.5)write(ifch,51)'  T  :',tem
      if(nflavs.ge.1.and.ish.ge.5)write(ifch,51)'chemu:',chem(1)
      if(nflavs.ge.2.and.ish.ge.5)write(ifch,51)'chemd:',chem(2)
      if(nflavs.ge.3.and.ish.ge.5)write(ifch,51)'chems:',chem(3)

c     particle yield
c     --------------
      call hgcpyi(0)

      if(ish.ge.5)write(ifch,*)('-',i=1,30)
     *,' exit sr hgcaaa ',('-',i=1,10)
      ish=isho
      return

       endif

      k=k+1
      goto30

      end


c---------------------------------------------------------------------
      function hgcbi0(x)
c---------------------------------------------------------------------
      DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9,y
      SAVE p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9
      DATA p1,p2,p3,p4,p5,p6,p7/1.0d0,3.5156229d0,3.0899424d0,
     *1.2067492d0,0.2659732d0,0.360768d-1,0.45813d-2/
      DATA q1,q2,q3,q4,q5,q6,q7,q8,q9/0.39894228d0,0.1328592d-1,
     *0.225319d-2,-0.157565d-2,0.916281d-2,-0.2057706d-1,0.2635537d-1,
     *-0.1647633d-1,0.392377d-2/
      if (abs(x).lt.3.75) then
        y=dble((x/3.75)**2)
        hgcbi0=sngl(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))))
      else
        ax=abs(x)
        y=dble(3.75/ax)
        hgcbi0=(exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *(q7+y*(q8+y*q9))))))))
      endif
      return
      end


c------------------------------------------------------------------------
      function hgcbi1(x)
c------------------------------------------------------------------------
      DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9,y
      SAVE p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9
      DATA p1,p2,p3,p4,p5,p6,p7/0.5d0,0.87890594d0,0.51498869d0,
     *0.15084934d0,0.2658733d-1,0.301532d-2,0.32411d-3/
      DATA q1,q2,q3,q4,q5,q6,q7,q8,q9/0.39894228d0,-0.3988024d-1,
     *-0.362018d-2,0.163801d-2,-0.1031555d-1,0.2282967d-1,-0.2895312d-1,
     *0.1787654d-1,-0.420059d-2/
      if (abs(x).lt.3.75) then
        y=dble((x/3.75)**2)
        hgcbi1=x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))))
      else
        ax=abs(x)
        y=dble(3.75/ax)
        hgcbi1=(exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *(q7+y*(q8+y*q9))))))))
        if(x.lt.0.)hgcbi1=-hgcbi1
      endif
      return
      END


c---------------------------------------------------------------------
      function hgcbk0(x)
c------------------------------------------------------------------------
      DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,y
      SAVE p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7
      DATA p1,p2,p3,p4,p5,p6,p7/-0.57721566d0,0.42278420d0,0.23069756d0,
     *0.3488590d-1,0.262698d-2,0.10750d-3,0.74d-5/
      DATA q1,q2,q3,q4,q5,q6,q7/1.25331414d0,-0.7832358d-1,0.2189568d-1,
     *-0.1062446d-1,0.587872d-2,-0.251540d-2,0.53208d-3/
      if (x.le.2.0) then
        y=dble(x*x/4.0)
        hgcbk0=(-log(x/2.0)*hgcbi0(x))+(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*
     *(p6+y*p7))))))
      else
        y=dble(2.0/x)
        hgcbk0=(exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *q7))))))
      endif
      return
      END


c---------------------------------------------------------------
      function hgcbk1(x)
c--------------------------------------------------------------------
      DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,y
      SAVE p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7
      DATA p1,p2,p3,p4,p5,p6,p7/1.0d0,0.15443144d0,-0.67278579d0,
     *-0.18156897d0,-0.1919402d-1,-0.110404d-2,-0.4686d-4/
      DATA q1,q2,q3,q4,q5,q6,q7/1.25331414d0,0.23498619d0,-0.3655620d-1,
     *0.1504268d-1,-0.780353d-2,0.325614d-2,-0.68245d-3/
      if (x.le.2.0) then
        y=dble(x*x/4.0)
        hgcbk1=(log(x/2.0)*hgcbi1(x))+(1.0/x)*(p1+y*(p2+y*(p3+y*(p4+y*
     *(p5+y*(p6+y*p7))))))
      else
        y=dble(2.0/x)
        hgcbk1=(exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *q7))))))
      endif
      return
      END


c-------------------------------------------------------------------
      function hgcbk(n,x)
c------------------------------------------------------------------
      tox=2.0/x
      bkm=hgcbk0(x)
      bk=hgcbk1(x)
      do 11 j=1,n-1
        bkp=bkm+j*tox*bk
        bkm=bk
        bk=bkp
11    continue
      hgcbk=bk
      return
      END


c----------------------------------------------------------------
      subroutine hgccbo(iba)
c----------------------------------------------------------------
c returns new chem(iafs) for boltzmann statistics
c  input:
c    tem
c    kef/volu
c  output:
c    chem(iafs)
c-----------------------------------------------------------------------
      common/cnsta/pi,pii,hquer,prom,piom,ainfin
      common/drop6/tecm,volu
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      parameter(nflav=6)
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      external hgcbk
      k=1
      iba=0
      c1=-0.5
      c2=0.5
      goto11

c     new chemical potential
c     ----------------------
10    chem(iafs)=c1+0.5*(c2-c1)
11    continue
      fd=0.0
      call hgchac(0)

        do i=1,nspecs

        if(ifok(iafs,i).ne.0)then
       if((chemgc(i)/tem).gt.70.)then
      hpd=1.e30
       else
      hpd=exp(chemgc(i)/tem)
       endif
       if(aspecs(i).ne.0.)then
      fk2=hgcbk(2,aspecs(i)/tem)
      hpd=hpd*gspecs(i)*aspecs(i)**2*tem*fk2
     */2./pi**2/hquer**3
       else
      hpd=hpd*gspecs(i)*tem**3/pi**2/hquer**3
       endif
      hfd=ifok(iafs,i)*hpd
      fd=fd+hfd
        endif

        enddo

      dfd=abs(fd-(kef(iafs)/volu))
      if(dfd.le.abs(gen*(kef(iafs)/volu)).or.dfd.le.genm)return
c     if(abs(fd).ge.100.)then
c     iba=1
c     return
c     endif


       if(fd.gt.(kef(iafs)/volu))then
      c2=chem(iafs)
      else
      c1=chem(iafs)
       endif

      k=k+1
      if(k.gt.300)return

      goto10

      end


c----------------------------------------------------------------------
      subroutine hgccch(iii)
c----------------------------------------------------------------------
c checks convergence of iterative algorithm
c plots iteration values for T and mu_i
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      parameter (nbin=500)
      common/cdatc/data(nbin),datb(nbin),datc(nbin),datd(nbin)
     *,date(nbin),datf(nbin),datg(nbin),dath(nbin),dati(nbin)
      common/cnrit/nrit
      character cen*4,cvol*4,cu*3,cd*3,cs*3

           if(iii.gt.0)then

      nrit=nrit+1
      data(nrit)=nrit
      datb(nrit)=tem
      datc(nrit)=chem(1)
      datd(nrit)=chem(2)
      date(nrit)=chem(3)

           elseif(iii.eq.0)then

      nrit=nrit+1
      data(nrit)=nrit
      datb(nrit)=tem
      datc(nrit)=chem(1)
      datd(nrit)=chem(2)
      date(nrit)=chem(3)
      do i=1,nrit
      datf(i)=datb(nrit)
      datg(i)=datc(nrit)
      dath(i)=datd(nrit)
      dati(i)=date(nrit)
      enddo

      x1=data(1)
      x2=data(nrit)
      write(cen,'(f4.1)')tecm
      write(cvol,'(f4.1)')volu
      write(cu,'(i3)')keu
      write(cd,'(i3)')ked
      write(cs,'(i3)')kes


      write(ifhi,'(a)')       'newpage zone 1 4 1 openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis Iteration"'
      write(ifhi,'(a)')       'text 0 0 "yaxis T (GeV)"'
      write(ifhi,'(a)')       'text 0.15 0.9 "E= '//cen//'"'
      write(ifhi,'(a)')       'text 0.4 0.9 "V= '//cvol//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),datb(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),datf(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis Iteration"'
      write(ifhi,'(a)')       'text 0 0 "yaxis [m]^1! (GeV)"'
      write(ifhi,'(a)')       'text 0.15 0.9 "Q^1!= '//cu//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),datc(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),datg(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis Iteration"'
      write(ifhi,'(a)')       'text 0 0 "yaxis [m]^2! (GeV)"'
      write(ifhi,'(a)')       'text 0.15 0.9 "Q^2!= '//cd//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),datd(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),dath(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis Iteration"'
      write(ifhi,'(a)')       'text 0 0 "yaxis [m]^3! (GeV)"'
      write(ifhi,'(a)')       'text 0.15 0.9 "Q^3!= '//cs//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),date(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(3a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrit
      write(ifhi,'(2e12.4)')data(j),dati(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

           endif

       return

       end

c-----------------------------------------------------------------------
      subroutine hgccex
c-----------------------------------------------------------------------
c returns new chem(iafs) for massive quantum statistics
c  input:
c    tem
c    kef/volu
c  output:
c    chem(iafs)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      external hgcfhn

      k=1

      c1=-0.5
      c2=0.5
      goto11

c     new chemical potential
c     ----------------------
10    chem(iafs)=c1+0.5*(c2-c1)
11    continue

      fd=0.0
        do ians=1,nspecs
       if(ifok(iafs,ians).ne.0)then

      call hgchac(0)
      call hgclim(a,b)
      if(b.eq.0.0)then
      hpd=0.0
      else
      call uttraq(hgcfhn,a,b,hpd)
      endif
      hpd=hpd*gspecs(ians)/2./pi**2/hquer**3
      fd=fd+hpd*ifok(iafs,ians)

       endif
        enddo

      dfd=abs(fd-(kef(iafs)/volu))
      if(dfd.le.abs(gen*(kef(iafs)/volu)).or.dfd.le.genm)return

       if(fd.gt.(kef(iafs)/volu))then
      c2=chem(iafs)
      else
      c1=chem(iafs)
       endif

      k=k+1
      if(k.gt.300)then
      if(ish.ge.5)
     *write(ifch,*)'failure at cex at iafs:',iafs
      return
      endif

      goto10

      end


c------------------------------------------------------------------
      subroutine hgccfc
c------------------------------------------------------------------
c checks flavor conservation in particle yield
c------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)

      if(ish.ge.5)write(ifch,*)'checking flavor conservation'
      do i=1,nflavs
      ckef=0.0
      do ii=1,nspecs
      ckef=ckef+ifok(i,ii)*ptlngc(ii)
      enddo
      dkef=abs(ckef-kef(i))
      if(dkef.le.1.e-2)then
      if(i.eq.1.and.ish.ge.5)write(ifch,*)'u conserved'
      if(i.eq.2.and.ish.ge.5)write(ifch,*)'d conserved'
      if(i.eq.3.and.ish.ge.5)write(ifch,*)'s conserved'
      else
      if(i.eq.1.and.ish.ge.5)write(ifch,*)'u not conserved'
      if(i.eq.2.and.ish.ge.5)write(ifch,*)'d not conserved'
      if(i.eq.3.and.ish.ge.5)write(ifch,*)'s not conserved'
      if(ish.ge.5)write(ifch,*)'df:',dkef
      endif
      enddo

      return
      end

c----------------------------------------------------------------
      subroutine hgcchb
c----------------------------------------------------------------
c checks results by numerical integration
c----------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      external hgcfbe
      external hgcfbn
      if(ish.ge.5)write(ifch,*)
     *'check by numer. calc. of expect. values:'
      iced=0
      ceden=0.0
       do ians=1,nspecs
      call hgclim(a,b)
      if(b.eq.0.0)then
      cedh=0.0
      else
      call uttraq(hgcfbe,a,b,cedh)
      endif
      if(ish.ge.9)write(ifch,*)'cedh:',cedh
      ced=cedh*gspecs(ians)/2./pi**2/hquer**3
      ceden=ceden+ced
       enddo

      if(iabs(ispecs(nspecs)).lt.10)
     *ceden=ceden+(8.*pi**2*tem**4/15.+bag4rt**4)/hquer**3

      if(ish.ge.5)write(ifch,*)'energy density   :',ceden
      ded=abs((tecm/volu)-ceden)
      if((tecm/volu)*gen.ge.ded.or.ded.le.gen)iced=1
      icfd=0

       do i=1,nflavs
      cfd=0.0
      do ians=1,nspecs
      call hgclim(a,b)
      if(b.eq.0.0)then
      hpd=0.0
      else
      call uttraq(hgcfbn,a,b,hpd)
      endif
      hfd=ifok(i,ians)*hpd*gspecs(ians)/2./pi**2/hquer**3
      if(ish.ge.9)write(ifch,*)'hfd:',hfd
      cfd=cfd+hfd
      enddo
      if(i.eq.1.and.ish.ge.5)write(ifch,5)'flavor density u :',cfd
      if(i.eq.2.and.ish.ge.5)write(ifch,5)'flavor density d :',cfd
      if(i.eq.3.and.ish.ge.5)write(ifch,5)'flavor density s :',cfd
5     format(1x,a,1x,f12.6)
      dfd=abs(cfd-(kef(i)/volu))
      if(abs(gen*(kef(i)/volu)).ge.dfd.or.dfd.le.gen)
     *icfd=icfd+1
       enddo

       if(iced.eq.1.and.icfd.eq.nflavs)then
      if(ish.ge.5)write(ifch,*)'results agree'
      else
      if(ish.ge.5)write(ifch,*)'results disagree'
       endif

      return
      end

c----------------------------------------------------------------
      subroutine hgcchh(icorr)
c----------------------------------------------------------------
c checks results by numerical integration
c----------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      external hgcfhe
      external hgcfhn
      icorr=0
      if(ish.ge.5)write(ifch,*)
     *'check by numer. calc. of expect. values:'

      iced=0
      ceden=0.0
       do ians=1,nspecs
      call hgclim(a,b)
      if(b.eq.0.0)then
      cedh=0.0
      else
      call uttraq(hgcfhe,a,b,cedh)
      endif
      if(ish.ge.9)write(ifch,*)'cedh:',cedh
      ced=cedh*gspecs(ians)/2./pi**2/hquer**3
      ceden=ceden+ced
       enddo

      if(iabs(ispecs(nspecs)).lt.10)
     *ceden=ceden+(8.*pi**2*tem**4/15.+bag4rt**4)/hquer**3

      if(ish.ge.5)write(ifch,*)'energy density   :',ceden
      ded=abs((tecm/volu)-ceden)
      if((tecm/volu)*gen.ge.ded.or.ded.le.gen)iced=1

      icfd=0

       do i=1,nflavs
      cfd=0.0
      do ians=1,nspecs
      call hgclim(a,b)
      if(b.eq.0.0)then
      hpd=0.0
      else
      call uttraq(hgcfhn,a,b,hpd)
      endif
      hfd=ifok(i,ians)*hpd*gspecs(ians)/2./pi**2/hquer**3
      if(ish.ge.9)write(ifch,*)'hfd:',hfd
      cfd=cfd+hfd
      enddo
      if(i.eq.1.and.ish.ge.5)write(ifch,5)'flavor density u :',cfd
      if(i.eq.2.and.ish.ge.5)write(ifch,5)'flavor density d :',cfd
      if(i.eq.3.and.ish.ge.5)write(ifch,5)'flavor density s :',cfd
5     format(1x,a,1x,f9.6)
      dfd=abs(cfd-(kef(i)/volu))
      if(abs(gen*(kef(i)/volu)).ge.dfd.or.dfd.le.gen)
     *icfd=icfd+1
       enddo

       if(iced.eq.1.and.icfd.eq.nflavs)then
      if(ish.ge.5)write(ifch,*)'results agree'
      icorr=1
      else
      if(ish.ge.5)write(ifch,*)'results disagree'
       endif

      return
      end


c--------------------------------------------------------------------
      subroutine hgccm0
c--------------------------------------------------------------------
c returns new quark chemical potentials for massless quantum statistics
c input:
c  tem
c  kef/volu
c output:
c  chem
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      external hgcfhn
      k=1
      z3=1.2020569

      c1=-0.5
      c2=0.5
      goto11

c     new chemical potential
c     ----------------------
10    chem(iafs)=c1+0.5*(c2-c1)
11    continue

      fd=0.0
      call hgchac(0)

               do i=1,nspecs
              if(ifok(iafs,i).ne.0)then

           igsp=int(gspecs(i))
          if(mod(igsp,2).eq.0)then

       if(ispecs(i).gt.0)then
      hpd=gspecs(i)*(chemgc(i)*tem**2+chemgc(i)**3/pi**2)/6./hquer**3
       else
      hpd=0.0
       endif

c            else
c      if(ispecs(i).gt.0)then
c     hpd=gspecs(i)*(chemgc(i)*tem**2/3.-chemgc(i)**3/pi**2/6.)/hquer**3
c      else
c     hpd=0.0
c      endif
c        endif

c     n=1
c0    xx=n*abs(chemgc(i))/tem
c     if(xx.le.60.)then
c     hpd=hpd+(-1.)**(n+1)/n**3/exp(xx)
c     n=n+1
c     goto20
c     endif
c     hpd=hpd*gspecs(i)*tem**3/pi**2/hquer**3
c     if(chemgc(i).eq.abs(chemgc(i)))then
c     hpd=gspecs(i)*(chemgc(i)*tem**2+chemgc(i)**3/pi**2)/6./hquer**3
c    *-hpd
c     endif

c      else
c     hpd=3.*gspecs(i)*tem**3*z3/4./pi**2/hquer**3
c      endif

         else

      hpd=gspecs(i)*tem**3*z3/pi**2/hquer**3

         endif

      hfd=hpd*ifok(iafs,i)
      fd=fd+hfd

       endif
        enddo

      dfd=abs(fd-(kef(iafs)/volu))
      if(dfd.le.abs(gen*(kef(iafs)/volu)).or.dfd.le.genm)return

       if(fd.gt.(kef(iafs)/volu))then
      c2=chem(iafs)
      else
      c1=chem(iafs)
       endif

      k=k+1
      if(k.gt.300)then
      if(ish.ge.5)
     *write(ifch,*)'failure at cm0 at iafs:',iafs
      return
      endif
      goto10
      end

c-----------------------------------------------------------------------
      function hgcfbe(x)
c-----------------------------------------------------------------------
c integrand of energy density
c------------------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=81.
      hgcfbe=0.0
      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.60.)return
      if(eex.lt.-60)then
      hgcfbe=1.e25
      return
      endif

      hgcfbe=sq*x**2*exp(-eex)

      return
      end

c-----------------------------------------------------------------
      function hgcfbf(x)
c-----------------------------------------------------------------
c integrand of mean square variance of  energy
c----------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=61
      hgcfbf=0.0

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.60.)return
      if(eex.lt.-60)then
      hgcfbf=1.e25
      return
      endif

      hgcfbf=(aspecs(ians)**2+x**2)*x**2*exp(-eex)

      return
      end

c-----------------------------------------------------------------
      function hgcfbn(x)
c-----------------------------------------------------------------
c integrand of hadron density
c-----------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=81.
      hgcfbn=0.0

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.80.)return
      if(eex.lt.-60)then
      hgcfbn=1.e25
      return
      endif

      hgcfbn=x**2*exp(-eex)

      return
      end

c-----------------------------------------------------------------------
      function hgcfhe(x)
c-----------------------------------------------------------------------
c integrand of energy density
c------------------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=81.
      hgcfhe=0.0
      igsp=int(gspecs(ians))

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.80.)return

       if(mod(igsp,2).ne.0)then
      d=-1.0
      if(eex.lt.1.e-10)return
       else
      d=1.0
       endif

      hgcfhe=sq*x**2/(exp(eex)+d)

      return
      end

c-----------------------------------------------------------------
      function hgcfhf(x)
c-----------------------------------------------------------------
c integrand of mean square variance of  energy
c----------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=61
      hgcfhf=0.0
      igsp=int(gspecs(ians))

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.60.)return
      if(eex.lt.(-60.))return

       if(mod(igsp,2).ne.0)then
      d=-1.0
      if(eex.lt.1.0e-10.and.eex.gt.(-1.0e-10))return
       else
      d=1.0
       endif

      hgcfhf=(aspecs(ians)**2+x**2)*x**2/(exp(eex)+2.0*d+exp(-eex))

      return
      end

c-----------------------------------------------------------------
      function hgcfhn(x)
c-----------------------------------------------------------------
c integrand of hadron density
c-----------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=81.
      hgcfhn=0.0
      igsp=int(gspecs(ians))

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.80.)return

       if(mod(igsp,2).ne.0)then
      d=-1.0
      if(eex.lt.1.e-10)return
       else
      d=1.0
       endif

      hgcfhn=x**2/(exp(eex)+d)

      return
      end

c-----------------------------------------------------------------
      function hgcfhw(x)
c-----------------------------------------------------------------
c integrand of mean square variance of hadron yield
c----------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      eex=61
      hgcfhw=0.0
      igsp=int(gspecs(ians))

      sq=sqrt(x**2+aspecs(ians)**2)
      if(tem.ne.0.0)eex=(sq-chemgc(ians))/tem
      if(eex.gt.60.)return
      if(eex.lt.(-60.))return

       if(mod(igsp,2).ne.0)then
      d=-1.0
      if(eex.lt.1.0e-10.and.eex.gt.(-1.0e-10))return
       else
      d=1.0
       endif

      hgcfhw=x**2/(exp(eex)+2.0*d+exp(-eex))

      return
      end


c-----------------------------------------------------------------
      subroutine hgchac(iboco)
c------------------------------------------------------------------
c returns hadronic chemical potentials as combinations of quark
c chemical potentials
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)

       do i=1,nspecs
      chemgc(i)=0.0
      do ii=1,nflavs
      chemgc(i)=chemgc(i)+ifok(ii,i)*chem(ii)
      if(ish.ge.9)write(ifch,*)'mu_i:',chem(ii),' k_i:',ifok(ii,i)
      enddo
      if(ish.ge.9)write(ifch,*)'mu_nu:',chemgc(i)
      igsp=int(gspecs(i))
      if(mod(igsp,2).ne.0.and.chemgc(i).gt.aspecs(i).and.iboco.eq.0)
     *chemgc(i)=aspecs(i)
       enddo

      return
      end


c-----------------------------------------------------------------------
      subroutine hgclim(a,b)
c----------------------------------------------------------------------
c returns integration limits for numerical evaluation of particle
c and energy densities using quantum statistics
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm

      igsp=int(gspecs(ians))

       if(mod(igsp,2).ne.0)then
      a=0.001
       else
      a=0.0
       endif

      b=0.0
      bb=(chemgc(ians)+tem*80.)**2-aspecs(ians)**2
      if(ish.ge.9)write(ifch,*)'bb:',bb
      if(bb.ge.0.0)b=sqrt(bb)
      if(bb.lt.0.0)then
      if(ish.ge.9)write(ifch,*)'failure at hgclim, bb=',bb
      if(ish.ge.9)write(ifch,'(1x,a,i5,a,2x,f12.6,1x,a,2x,f9.6)')
     *'mu(',ispecs(ians),'):',chemgc(ians),' T:',tem
      endif
      if(ish.ge.9)write(ifch,*)'ians:',ians,' a:',a,' b:',b
      return
      end

c------------------------------------------------------------------------
      subroutine hgcnbi(iret)
c-----------------------------------------------------------------------
c uses hgcaaa results to generate initial hadron set, nlattc, iozero
c input:
c    ptlngc(1:nspecs): particle number expectation values  /cgchg/
c output:
c     nump:           number of hadrons   /chnbin/
c     ihadro(1:nump): hadron ids          /chnbin/
c     nlattc:         lattice size        /clatt/
c     iozero:         zero weight         /metr1/
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/chnbin/nump,ihadro(maxp)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgctot/rmstot,ptltot
      common/camgc/amgc,samgc,amtot
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/clatt/nlattc,npmax
      common/cgcnb/nptlgc(mspecs)
      common/ctaue/taue
      common/cgck/k(nflav),kp(nflav),kps(nflav)
     *,idp(maxp),ida(mspecs),idb(mspecs)
      integer hgcndn

      iret=0
      isho=ish
      if(ishsub/100.eq.50)ish=mod(ishsub,100)

      if(ish.ge.7)write(ifch,*)('-',l=1,10)
     *,' entry sr hgcnbi ',('-',l=1,30)


      nh=nint(ptltot)
      iug=(1+iospec)/2*2-1
      if(iug.lt.9)call utstop('hgcnbi: iospec < 9&')

c     determine nlattc
c     ----------------
        if(ionlat.eq.1)then
      s1=ptltot+2.*rmstot
      s2=1.3*ptltot
      s=max(s1,s2,6.)
      nlattc=nint(s)
       elseif(ionlat.eq.2)then
      s1=ptltot+3.*rmstot
      s2=1.5*ptltot
      s=max(s1,s2,6.)
      nlattc=nint(s)
       elseif(ionlat.eq.3)then
      s1=ptltot+4.*rmstot
      s2=2.*ptltot
      s=max(s1,s2,6.)
      nlattc=nint(s)
        elseif(ionlat.eq.0)then
      nlattc=8*(tecm/10)*(1/(tecm/volu))**0.2*(nspecs/3.)**0.3
      if(aspecs(1).lt.0.010)nlattc=nlattc*3
      nlattc=max(nlattc,20)
        endif

      if(ish.ge.7)write(ifch,*)'nlattc:',nlattc

c     determine iozero
c     ----------------
      if(iozero.eq.-1)then
      iozero=nspecs
      elseif(iozero.eq.-2)then
      iozero=nspecs*int(sqrt(volu/tecm))
      endif

c     modify iozero for testing
c     -------------------------
      if(iozevt.gt.0)then
      iozero=(nrevt/iozevt+1)*iozinc   !nrevt=event number - 1 !!
      write(ifch,*)'nrevt+1:',nrevt+1,'   iozero:',iozero
      endif

c     initial hadron set
c     ------------------
      ammin=2.*aspecs(1)
      if(tecm.lt.ammin)then
      write(ifch,*)'impossible to generate hadron configuration'
      call utstop('hgcnbi: tecm less than two pi0 masses&')
      endif

      kk=1
100   continue

       if(kk.gt.20)then
       iret=1
      if(ish.ge.7)then
      write(ifch,*)'failed to generate hadron set for'
     *,' event:',nrevt+1
      write(ifch,*)'u d s :',keu,ked,kes,' E:',tecm
      write(ifch,*)('-',i=1,30)
     *,' exit sr hgcnbi ',('-',i=1,10)
      endif
      ish=isho
      return
        endif

      amtot=0.0
      do i=1,nspecs
      nptlgc(i)=0
      enddo
      do ii=1,nflavs
      k(ii)=kef(ii)
      enddo

      if(ish.ge.7)write(ifch,*)
     *'sample hadron multiplicities and total mass:'

      kbar=keu+ked+kes
      kpar=iabs(keu)+iabs(ked)+iabs(kes)
      nbar=kbar/3.
      if(ish.ge.7)write(ifch,*)'baryon number:',nbar,' parton number:'
     *,kpar

      nn=2
      if(ioinco.ne.2)then
      nn=hgcndn(0)
      else
      nn=nh
      endif
      nb=iabs(nbar)
      if(ish.ge.7)write(ifch,*)'<n>:',nh,' n_sample:',nn,' n_bar:',nb
      if(nn.gt.nb.and.nb.ne.0.and.nb.ge.nh)nn=nb
      if(nn.lt.nb.and.nb.ne.0)nn=nb
      km=kpar-iabs(kbar)
      nt=km/2+nb
      if(nt.gt.nn)nn=nt
      nn=max(nn,2)

      if(ioinco.eq.2)then
      nit=15*taue
      else
      itpn=100
      nit=nn*itpn
      endif
      nbb=0
      n=0

c     start with nb protons
      nptlgc(19)=nptlgc(19)+nb
      n=nb
      amtot=amtot+nb*aspecs(19)
      do ii=1,nflavs
      k(ii)=k(ii)-ifok(ii,19)*nb
      enddo
      nbb=nbb+nb


       do it=1,nit

      xsp=nspecs
      x0=0.5
      xib=x0+xsp*rangen()
      ib=nint(xib)
      if(ib.gt.nspecs)ib=nspecs
      if(ib.lt.1)ib=1
      kb=ifok(1,ib)+ifok(2,ib)+ifok(3,ib)
      if(rangen().lt.0.5.and.nptlgc(ib).ge.1)then
      ni=-1
      else
      ni=1
      endif
      as=1.0
      if(nptlgc(ib).eq.0)as=0.5
      if(nptlgc(ib).eq.1.and.ni.eq.(-1))as=2.0
      if(ish.ge.9)write(ifch,*)
     *'id:',ispecs(ib),' <i>:',ptlngc(ib),' ni:',ni

         if(ni.ne.0)then

       if(ptlngc(ib).gt.5.0)then

      pnla=hgcpnl(ib,0)
      pnlb=hgcpnl(ib,ni)
      pnlog=-pnla+pnlb
      if(ish.ge.9)write(ifch,*)'pnlog:',pnlog
      if(pnlog.lt.60)then
      pn=exp(pnlog)
      else
      pn=1.1
      endif

       else

      if(ni.eq.1)then
      pn=ptlngc(ib)/(nptlgc(ib)+1)
      elseif(ni.eq.(-1).and.ptlngc(ib).gt.1.e-20)then
      pn=nptlgc(ib)/ptlngc(ib)
      elseif(nptlgc(ib).gt.0)then
      pn=1.1
      else
      pn=0.0
      endif

       endif

       pm=1.0
       if(ioinfl.ge.0)then
      pmla=hgcpml(ib,0,ib,0)
      pmlb=hgcpml(ib,ni,ib,0)
      pmlog=-pmla+pmlb
      if(ish.ge.9)write(ifch,*)'pmlog:',pmlog
      if(pmlog.lt.60)then
      pm=exp(pmlog)
      else
      pm=1.1
      endif
       endif

      p=pn*pm*as
      r=rangen()
      if(r.le.p)then
      nptlgc(ib)=nptlgc(ib)+ni
      n=n+ni
      amtot=amtot+ni*aspecs(ib)
      do ii=1,nflavs
      k(ii)=k(ii)-ifok(ii,ib)*ni
      enddo
      if(kb.ne.0)nbb=nbb+ni
      if(ish.ge.7.and.ni.gt.0)write(ifch,*)'add:'
      if(ish.ge.7.and.ni.lt.0)write(ifch,*)'remove:'
      if(ish.ge.7)write(ifch,*)'id:',ispecs(ib),' <n_i>:',ptlngc(ib)
     *,' n_i:',nptlgc(ib)
      if(ish.ge.7)write(ifch,*)'<n>:',nn,' it:',it
      if(ish.ge.7)write(ifch,*)'<M>:',amgc,' M:',amtot
      if(ish.ge.7)write(ifch,*)'p:',p,' r:',r
      if(ish.ge.7)write(ifch,*)'flav defect: u:',k(1),' d:'
     *,k(2),' s:',k(3)
      if(n.ge.nn.and.ioinco.ne.2)goto102
      endif

       endif

       enddo


102   continue

       ndd=0
c      if(nbb.lt.nb)then
c      nba=nb-nbb
c     if(nbar.gt.0)then
c     if(ish.ge.7)write(ifch,*)'add protons: nba:',nba
c     nptlgc(19)=nptlgc(19)+nba
c     n=n+nba
c     amtot=amtot+aspecs(19)*nba
c     elseif(nbar.lt.0)then
c     if(ish.ge.7)write(ifch,*)'add aprotons: nba:',nba
c     nptlgc(20)=nptlgc(20)+nba
c     n=n+nba
c     amtot=amtot+aspecs(20)*nba
c     endif
c      endif
       if(n.lt.nn.and.ioinco.ne.2)then
      ndd=nn-n
      nd=mod(ndd,4)
      xn=n
      xnn=nn
      xl=(xnn-xn)/4.
      l=aint(xl)
      if(ish.ge.7)write(ifch,*)'add pions/etas: ndd:',ndd
     *,' l:',l,' nd:',nd
        if(l.ge.1)then
       do j=1,l
      nptlgc(1)=nptlgc(1)+1
      nptlgc(2)=nptlgc(2)+1
      nptlgc(3)=nptlgc(3)+1
      nptlgc(8)=nptlgc(8)+1
      amtot=amtot+aspecs(1)+aspecs(2)+aspecs(3)+aspecs(8)
       enddo
        endif
      if(nd.eq.1)then
      nptlgc(1)=nptlgc(1)+1
      amtot=amtot+aspecs(1)
      elseif(nd.eq.2)then
      nptlgc(2)=nptlgc(2)+1
      nptlgc(3)=nptlgc(3)+1
      amtot=amtot+aspecs(2)+aspecs(3)
      elseif(nd.eq.3)then
      nptlgc(2)=nptlgc(2)+1
      nptlgc(3)=nptlgc(3)+1
      nptlgc(1)=nptlgc(1)+1
      amtot=amtot+aspecs(2)+aspecs(3)+aspecs(1)
      endif
       endif

       if(n.eq.0.and.ioinco.eq.2)then
      nptlgc(2)=nptlgc(2)+1
      nptlgc(3)=nptlgc(3)+1
      amtot=amtot+aspecs(2)+aspecs(3)
       elseif(n.eq.1.and.ioinco.eq.2)then
      nptlgc(1)=nptlgc(1)+1
      amtot=amtot+aspecs(1)
       endif

      if(amtot.ge.tecm.and.ioinfl.ge.0)then
      if(ish.ge.7)write(ifch,*)
     *'total mass exceeded , redo configuration'
      kk=kk+1
      goto100
      endif


      iii=0
      if(ish.ge.7)then
        write(ifch,*)'u d s :',keu,ked,kes,' E:',tecm
        write(ifch,*)
     *'hadron set without flavor conservation:'
      endif
      do i=1,nspecs
      n=nptlgc(i)
      if(n.ge.1)then
      do j=1,n
      iii=iii+1
      if(iii.gt.maxp)stop'iii>maxp in hgcnbi'
      idp(iii)=ispecs(i)
      enddo
      endif
      enddo
      if(ish.ge.7)then
        write(ifch,'(1x,10i6)')(idp(i),i=1,iii)
        write(ifch,*)'flav defect: u:',k(1),' d:'
     *,k(2),' s:',k(3)
        write(ifch,*)'M:',amtot,' <M>:',amgc
      endif
      if(ioinfl.le.0)goto1000

      ll=1
      llmax=nn*25
      ior=1

120        if(k(1).ne.0.or.k(2).ne.0.or.k(3).ne.0)then

        if(kk.gt.6)ior=0

      if(ish.ge.7)write(ifch,*)
     *'remaining flavor defect before operation:',ll
      if(ish.ge.7)write(ifch,*)'flav defect: u:',k(1),' d:'
     *,k(2),' s:',k(3)

      nida=0
      do i=1,nspecs
      if(nptlgc(i).gt.0)then
      nida=nida+1
      ida(nida)=i
      endif
      enddo

      if(nida.eq.0)then
      if(ish.ge.7)write(ifch,*)'no proposals in a , redo'
      kk=kk+1
      goto100
      endif


      xna=0.5+nida*rangen()
      na=nint(xna)
      if(na.gt.nida)na=nida
      if(na.lt.1)na=1
      ia=ida(na)
      if(ish.ge.7)write(ifch,*)'nida:',nida,' ia:',ia

      nidb=0
      do ii=1,nflavs
      kp(ii)=k(ii)+ifok(ii,ia)
      kps(ii)=isign(1,kp(ii))
      enddo
      if(ish.ge.7)write(ifch,*)
     *'   assemble: u:',kp(1),' d:',kp(2),' s:',kp(3)
      do i=1,nspecs
      iacc=0
      naccsp=0
      naccmi=1
      do ii=1,nflavs
      naccsp=naccsp+iabs(ifok(ii,i))
      if(kp(ii).ne.0)then
      if(kps(ii)*ifok(ii,i).le.kps(ii)*kp(ii)
     *.and.kps(ii)*ifok(ii,i).gt.0)iacc=iacc+iabs(ifok(ii,i))
      endif
      enddo
      if(kp(1).eq.0.and.kp(2).eq.0.and.kp(3).eq.0)naccmi=0
      if(iacc.eq.naccsp.and.naccsp.ge.naccmi)then
      nidb=nidb+1
      idb(nidb)=i
      endif
      enddo

      if(nidb.eq.0)then
      if(ish.ge.7)write(ifch,*)'no proposals in b , redo'
      kk=kk+1
      goto100
      endif

      xnb=0.5+nidb*rangen()
      nb=nint(xnb)
      if(nb.gt.nidb)nb=nidb
      if(nb.lt.1)nb=1
      ib=idb(nb)
      if(ish.ge.7)write(ifch,*)'nidb:',nidb,' ib:',ib
      if(ish.ge.7)write(ifch,*)
     *'proposal:',ispecs(ia),' --> ',ispecs(ib)

      asym=1.0

c      if(asym.gt.0.0)then

       if(ptlngc(ia).gt.5.0)then
      pnali=hgcpnl(ia,0)
      pnalf=hgcpnl(ia,-1)
      pnalog=-pnali+pnalf
      if(ish.ge.7)write(ifch,*)'pnalog:',pnalog
      if(pnalog.lt.60)then
      pna=exp(pnalog)
      else
      pna=1.1
      endif
       else
      if(ptlngc(ia).gt.1.e-20)then
      pna=nptlgc(ia)/ptlngc(ia)
      elseif(nptlgc(ia).gt.0)then
      pna=1.1
      else
      pna=0.0
      endif
       endif

       if(ptlngc(ib).gt.5.0)then
      pnbli=hgcpnl(ib,0)
      pnblf=hgcpnl(ib,1)
      pnblog=-pnbli+pnblf
      if(ish.ge.7)write(ifch,*)'pnblog:',pnblog
      if(pnblog.lt.60)then
      pnb=exp(pnblog)
      else
      pnb=1.1
      endif
       else
      pnb=ptlngc(ib)/(nptlgc(ib)+1)
       endif


      pmli=hgcpml(ia,0,ib,0)
      pmlf=hgcpml(ia,-1,ib,1)
      pmlog=-pmli+pmlf
      if(ish.ge.7)write(ifch,*)'pmlog:',pmlog
      if(pmlog.lt.60)then
      pm=exp(pmlog)
      else
      pm=1.1
      endif

      p=pna*pnb*pm*asym
      if(ior.eq.0)then
      r=0.0
      else
      r=rangen()
      endif

c      else

c     r=1.0
c     p=0.0

c      endif

       if(r.lt.p)then
      if(ish.ge.7)write(ifch,*)'p:',p,' r:',r,' asymmetry:',asym
      if(ish.ge.7)write(ifch,*)'remove ',ispecs(ia),'  add ',ispecs(ib)
     *,'  proposal accepted'
      nptlgc(ia)=nptlgc(ia)-1
      nptlgc(ib)=nptlgc(ib)+1
      amtot=amtot-aspecs(ia)+aspecs(ib)
      do ii=1,nflavs
      k(ii)=k(ii)+ifok(ii,ia)-ifok(ii,ib)
      enddo
       endif


        if(k(1).ne.0.or.k(2).ne.0.or.k(3).ne.0)then
       ll=ll+1
      if(ll.le.llmax)then
      goto120
      else
      if(ish.ge.7)write(ifch,*)'failed to remove defect, redo'
      kk=kk+1
      goto100
       endif
        endif

         endif

1000  continue

      nump=0
      kcu=0
      kcd=0
      kcs=0
      do i=1,nspecs
      n=nptlgc(i)
      if(n.ge.1)then
      do j=1,n
      nump=nump+1
      ihadro(nump)=ispecs(i)
      kcu=kcu+ifok(1,i)
      kcd=kcd+ifok(2,i)
      kcs=kcs+ifok(3,i)
      enddo
      endif
      enddo

          if(ioinfl.gt.0)then
        if(kcu.ne.keu.or.kcd.ne.ked.or.kcs.ne.kes)then
      if(ish.ge.7)write(ifch,*)
     *'failed to remove flavor defect, redo configuration'
      kk=kk+1
      goto100
        endif
          endif

      if(ioinct.ge.1)then
        chitot=0.0
        nutot=nspecs
        do i=1,nspecs
        chi=0.0
        if(rmsngc(i).gt.1.e-10)chi=(ptlngc(i)-nptlgc(i))/rmsngc(i)
        chitot=chitot+chi**2
        enddo
        call xhgccc(chitot)

        u=0
        d=0
        s=0
        do i=1,nspecs
        u=u+ifok(1,i)*nptlgc(i)
        d=d+ifok(2,i)*nptlgc(i)
        s=s+ifok(3,i)*nptlgc(i)
        enddo
        call xhgcfl(u,d,s,0)
        call xhgcam(amtot,0)
      endif

      if(ish.ge.7)then
        write(ifch,*)
     *'initial hadron set for droplet decay:'
        write(ifch,'(1x,10i6)')(ihadro(i),i=1,nump)
      endif
       if(nump.ge.nlattc)then
         nlattc=nump+1
         if(ish.ge.7)then
           write(ifch,*)'initial set > nlattc !'
           write(ifch,*)'new nlattc:',nlattc
         endif
       endif
       if(ish.ge.7)then
         write(ifch,*)'keu:',kef(1),' kcu:',kcu,' ku:',k(1)
         write(ifch,*)'ked:',kef(2),' kcd:',kcd,' kd:',k(2)
         write(ifch,*)'kes:',kef(3),' kcs:',kcs,' ks:',k(3)
         write(ifch,*)' nh:',nh,' nump:',nump
         write(ifch,*)' nu:',nutot,'  chi^2:',chitot
         write(ifch,*)'iozero:',iozero,'  iomom:',iomom
         write(ifch,*)
     *'total mass:',amtot,' droplet mass:',tecm
         write(ifch,*)'trials needed:',kk
     *,' operations needed:',ll
         write(ifch,*)'iterations:',it,' pions added:',ndd
         write(ifch,*)('-',i=1,30)
     *,' exit sr hgcnbi ',('-',i=1,10)
       endif
      ish=isho
      return

      end

c--------------------------------------------------------------------
      integer function hgcndn(i)
c--------------------------------------------------------------------
c returns random multiplicity from gaussian distribution for species i
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgctot/rmstot,ptltot
      common/clatt/nlattc,npmax
      a=iowidn
      kk=0

       if(i.eq.0)then

1     continue
      kk=kk+1
      p=0.0
      nmin=2
      nh=nint(ptltot)
      nmax=nlattc
      xn=1.5+(nmax-nmin)*rangen()
      n=nint(xn)
      x=(n-ptltot)**2/2.0
      y=-70.
      if(rmstot.gt.1.e-15)y=-x/rmstot**2*a**2
      if(y.lt.70.)p=exp(y)
      if(rmstot.gt.1.e-15.and.iowidn.lt.0)p=p/sqrt(2.*pi)/rmstot
      if(p.ge.rangen())then
      hgcndn=n
      if(ish.ge.9)write(ifch,*)'hgcndn: k:',kk,' n:',hgcndn
      return
      else
      if(kk.le.25)goto1
      hgcndn=max(2,nh)
      if(ish.ge.9)write(ifch,*)'hgcndn: k:',kk,' n:',hgcndn
      return
      endif

       else

2     continue
      kk=kk+1
      p=0.0
      nmin=0
      nh=nint(ptlngc(i))
      nmax=2*nh
      nmax=max(2,nmax)
      xn=-0.5+(nmax-nmin)*rangen()
      n=nint(xn)
      x=(n-ptlngc(i))**2/2.0
      if(x.lt.1.e-30)then
      p=1.
      else
      y=-70.
      if(rmsngc(i).gt.1.e-15)y=-x/rmsngc(i)**2
      if(y.lt.70.)p=exp(y)
      if(rmsngc(i).gt.1.e-15.and.iowidn.lt.0)
     *p=p/sqrt(2.*pi)/rmsngc(i)
      endif
      if(p.ge.rangen())then
      hgcndn=n
      if(ish.ge.9)write(ifch,*)'hgcndn: k:',kk,' n:',hgcndn
      return
      else
      if(kk.le.25)goto2
      hgcndn=nh
      if(ish.ge.9)write(ifch,*)'hgcndn: k:',kk,' n:',hgcndn
      return
      endif

       endif

      end

c--------------------------------------------------------------------
      function hgcpml(i1,n1,i2,n2)
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/camgc/amgc,samgc,amtot
      common/cgcnb/nptlgc(mspecs)
      if(ish.ge.9)write(ifch,*)'i1:',i1,' i2:',i2
      if(ish.ge.9)write(ifch,*)'n1:',n1,' n2:',n2
      hgcpml=-1.e30
      ampr=n1*aspecs(i1)+n2*aspecs(i2)
      if((amtot+ampr).lt.tecm.and.(amtot+ampr).ge.0
     *.and.nptlgc(i1).ge.(-n1).and.nptlgc(i2).ge.(-n2))then
      hgcpml=0.0
      pl=(amtot-amgc+ampr)**2/2.0
      if(pl.lt.1.e-30)then
      hgcpml=0.0
      return
      endif
      if(samgc.gt.1.e-15)hgcpml=-pl/samgc**2
      endif
      if(ish.ge.9)write(ifch,*)'hgcpml:',hgcpml
      return
      end

c--------------------------------------------------------------------
      function hgcpnl(i,n)
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgcnb/nptlgc(mspecs)
      if(ish.ge.9)write(ifch,*)'i:',i,' n:',n
      hgcpnl=-1.e30
      if(nptlgc(i).ge.(-n))then
      pl=(nptlgc(i)-ptlngc(i)+n)**2/2.0
      if(pl.lt.1.e-30)then
      hgcpnl=0.0
      return
      endif
      if(rmsngc(i).gt.1.e-15)hgcpnl=-pl/rmsngc(i)**2
      endif
      if(ish.ge.9)write(ifch,*)'hgcpnl:',hgcpnl
      return
      end


c--------------------------------------------------------------------
      subroutine hgcpen
c--------------------------------------------------------------------
c returns array for twodimensional plot of energy- and flavor-
c density
c--------------------------------------------------------------------
c xpar1,xpar2 temperature range
c xpar3       # of bins for temperature
c xpar4,xpar5 chem.pot. range
c xpar6       # of bins for chem.pot.
c xpar7       max. density
c xpar8       strange chem.pot.
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/ciakt/gen,iafs,ians,genm
      parameter (nbin=100)
      real edensi(nbin,nbin),qdensi(nbin,nbin)
      external hgcfhe
      external hgcfhn
      external hgcfbe
      external hgcfbn

      iug=(1+iospec)/2*2-1

c     initialization
c     --------------

      if(iug.eq.1)nflavs=1
      if(iug.eq.3)nflavs=2
      if(iug.eq.5)nflavs=2
      if(iug.eq.7)nflavs=3
      if(iug.eq.9)nflavs=3
      if(iug.eq.11)nflavs=3
      tem=0.0
      do i=1,nflavs
      chem(i)=0.0
      enddo
      call hgchac(0)
      do i=1,nspecs
      ptlngc(i)=0.0
      rmsngc(i)=0.0
      enddo

      nbt=nint(xpar3)
      nbc=nint(xpar6)
      nbc=min(nbc,100)
      nbt=min(nbt,100)
      dt=(xpar2-xpar1)/nbt
      dc=(xpar5-xpar4)/nbc
      ymax=xpar7
      cs=xpar8


      t0=xpar1+dt/2.
      c0=xpar4+dc/2
      do i=1,nbc
      chem(1)=c0+(i-1)*dc
      chem(2)=chem(1)
      chem(3)=cs
      chem(4)=0.0
      chem(5)=0.0
      chem(6)=0.0
      call hgchac(0)
      do ii=1,nbt
      tem=t0+(ii-1)*dt
      if(ish.ge.5)write(ifch,*)' mu:',chem(1),' T:',tem

       qd=0.0
       ed=0.0

      do ians=1,nspecs

      call hgclim(a,b)

      if(b.eq.0.0)then
      hden=0.0
      elseif(iostat.eq.0)then
      call uttraq(hgcfhn,a,b,hden)
      elseif(iostat.eq.1)then
      call uttraq(hgcfbn,a,b,hden)
      endif
      hd=hden*gspecs(ians)/2./pi**2/hquer**3

      if(ish.ge.7)write(ifch,*)'i:',ians,' n_u:',ifok(1,ians),' hd:',hd

      qd=qd+ifok(1,ians)*hd+ifok(2,ians)*hd
      if(qd.gt.ymax)qd=ymax
c     if(qd.gt.ymax)qd=0.0
      if(qd.lt.-ymax)qd=-ymax
c     if(qd.lt.-ymax)qd=0.0


      if(b.eq.0.0)then
      edi=0.0
      elseif(iostat.eq.0)then
      call uttraq(hgcfhe,a,b,edi)
      elseif(iostat.eq.1)then
      call uttraq(hgcfbe,a,b,edi)
      endif
      edi=edi*gspecs(ians)/2./pi**2/hquer**3

      if(ish.ge.7)write(ifch,*)'i:',ians,' mu:',chemgc(ians)
     *                        ,' edi:',edi

      ed=ed+edi
      if(ed.gt.ymax)ed=ymax
c     if(ed.gt.ymax)ed=0.0
      enddo

      if(ish.ge.5)write(ifch,*)' ed:',ed,' qd:',qd
      edensi(i,ii)=ed
      qdensi(i,ii)=qd

      enddo
      enddo

      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar1,xpar2
      write(ifhi,'(a,2e11.3)')'yrange',xpar4,xpar5
      write(ifhi,'(a)')      'set ityp2d 5'
      write(ifhi,'(a,i4)')   'array2d',nbt
      do j=1,nbc
      do jj=1,nbt
      write(ifhi,'(e11.3)') edensi(j,jj)
      enddo
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot2d'

      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar1,xpar2
      write(ifhi,'(a,2e11.3)')'yrange',xpar4,xpar5
      write(ifhi,'(a)')      'set ityp2d 5'
      write(ifhi,'(a,i4)')   'array2d',nbt
      do j=1,nbc
      do jj=1,nbt
      write(ifhi,'(e11.3)') qdensi(j,jj)
      enddo
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot2d'

        return
        end

c--------------------------------------------------------------------
      subroutine hgcpfl
c--------------------------------------------------------------------
c returns array for twodimensional plot of energy- and flavor-
c density fluctuations
c--------------------------------------------------------------------
c xpar1,xpar2 temperature range
c xpar3       # of bins for temperature
c xpar4,xpar5 chem.pot. range
c xpar6       # of bins for chem.pot.
c xpar7       max. density
c xpar8       strange chem.pot.
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cflavs/nflavs,kef(nflav),chem(nflav)
      common/ciakt/gen,iafs,ians,genm
      parameter (nbin=100)
      real efl(nbin,nbin),qfl(nbin,nbin),v(nbin),wn(nbin),we(nbin)
      external hgcfhf
      external hgcfhe
      external hgcfhn
      external hgcfhw
      external hgcfbf
      external hgcfbe
      external hgcfbn

      iug=(1+iospec)/2*2-1

c     initialization
c     --------------

      if(iug.eq.1)nflavs=1
      if(iug.eq.3)nflavs=2
      if(iug.eq.5)nflavs=2
      if(iug.eq.7)nflavs=3
      if(iug.eq.9)nflavs=3
      if(iug.eq.11)nflavs=3
      tem=0.0
      do i=1,nflavs
      chem(i)=0.0
      enddo
      call hgchac(0)
      do i=1,nspecs
      ptlngc(i)=0.0
      rmsngc(i)=0.0
      enddo

      nbt=nint(xpar3)
      nbv=nint(xpar6)
      nbv=min(nbv,100)
      nbt=min(nbt,100)
      dt=(xpar2-xpar1)/nbt
      dv=(xpar5-xpar4)/nbv
      ymax=1.e20
      chem(1)=xpar7
      chem(2)=xpar7
      chem(3)=xpar8
      call hgchac(0)


      t0=xpar1+dt/2.
      v0=xpar4
      do i=1,nbv
      volu=v0+(i-1)*dv
      do ii=1,nbt
      tem=t0+(ii-1)*dt
      if(ish.ge.5)write(ifch,*)'volu:',volu,' tem:',tem

       ev=0.0
       ee=0.0
       qv=0.0
       qe=0.0

      do ians=1,nspecs

      call hgclim(a,b)

      if(b.eq.0.0)then
      hn=0.0
      hv=0.0
      elseif(iostat.eq.0)then
      call uttraq(hgcfhn,a,b,hn)
      call uttraq(hgcfhw,a,b,hv)
      elseif(iostat.eq.1)then
      call uttraq(hgcfbn,a,b,hn)
      hv=hn
      endif
      hn=hn*volu*gspecs(ians)/2./pi**2/hquer**3
      hv=hv*volu*gspecs(ians)/2./pi**2/hquer**3
      if(ish.ge.5)write(ifch,*)'hn:',hn,' hv:',hv

      hn=max(hn,1.e-15)
      qv=qv+hv
      qe=qe+hn


      if(qv.gt.ymax)qv=ymax
      if(qe.gt.ymax)qe=ymax


      if(b.eq.0.0)then
      eei=0.0
      evi=0.0
      elseif(iostat.eq.0)then
      call uttraq(hgcfhe,a,b,eei)
      call uttraq(hgcfhf,a,b,evi)
      elseif(iostat.eq.1)then
      call uttraq(hgcfbe,a,b,eei)
      call uttraq(hgcfbf,a,b,evi)
      endif
      eei=eei*volu*gspecs(ians)/2./pi**2/hquer**3
      evi=evi*volu*gspecs(ians)/2./pi**2/hquer**3
      if(ish.ge.5)write(ifch,*)'eei:',eei,' evi:',evi


      eei=max(eei,1.e-15)
      ev=ev+evi
      ee=ee+eei
      if(ev.gt.ymax)ev=ymax
      if(ee.gt.ymax)ee=ymax
      enddo
      if(ish.ge.5)write(ifch,*)'qv:',qv,' ev:',ev

      qfl(i,ii)=0.
      efl(i,ii)=0.
      if(ev.gt.0.0.and.ee.gt.1.e-15)efl(i,ii)=sqrt(ev)/ee
      if(qv.gt.0.0.and.ee.gt.1.e-15)qfl(i,ii)=sqrt(qv)/qe
      if(tem.eq.0.195)then
      we(i)=efl(i,ii)
      wn(i)=qfl(i,ii)
      v(i)=volu
      endif

      enddo
      enddo

      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar1,xpar2
      write(ifhi,'(a,2e11.3)')'yrange',xpar4,xpar5
      write(ifhi,'(a)')      'set ityp2d 5'
      write(ifhi,'(a,i4)')   'array2d',nbt
      do j=1,nbv
      do jj=1,nbt
      write(ifhi,'(e11.3)') efl(j,jj)
      enddo
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot2d'

      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar1,xpar2
      write(ifhi,'(a,2e11.3)')'yrange',xpar4,xpar5
      write(ifhi,'(a)')      'set ityp2d 5'
      write(ifhi,'(a,i4)')   'array2d',nbt
      do j=1,nbv
      do jj=1,nbt
      write(ifhi,'(e11.3)') qfl(j,jj)
      enddo
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot2d'

      write(ifhi,'(a)')      'newpage zone 1 2 1'
      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar4,xpar5
      write(ifhi,'(a)')      'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,i4)')   'array 2'
      do j=1,nbv
      write(ifhi,'(2e13.5)')v(j),we(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

      write(ifhi,'(a)')      'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',xpar4,xpar5
      write(ifhi,'(a)')      'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,i4)')   'array 2'
      do j=1,nbv
      write(ifhi,'(2e13.5)')v(j),wn(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'


        return
        end


c------------------------------------------------------------------
      subroutine hgcpyi(ist)
c------------------------------------------------------------------
c returns particle yield
c input:
c   tem   : temperature
c   chemgc: chemical potentials
c output:
c   ptlngc: expectation value of particle number for each species
c   rmsngc: standard deviation of ptlngc
c   ptltot: total particle number
c   rmstot: standard deviation of ptltot
c works for hadrons and partons
c  ist=1 boltzmann statistics
c  ist=0 quantum statistics
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgctot/rmstot,ptltot
      common/camgc/amgc,samgc,amtot
      common/ciakt/gen,iafs,ians,genm
      external hgcfhw
      external hgcfhn

       if(iabs(ispecs(nspecs)).lt.10)then

c     parton yield
c     ------------
      if(ish.ge.5)write(ifch,*)'parton yield:'
      gln=16.*1.20206*tem**3/pi**2*volu/hquer**3
      sdg=sqrt(gln)   !!???
      if(ish.ge.5)write(ifch,'(1x,a,f10.4,2x,a,f9.4,a)')
     *'<N(    0)> :',gln,' sigma :',sdg,' (qm-statistics!)'
      ptltot=gln
      rmstot=0.0
      vartot=gln

       else

      if(ish.ge.5)write(ifch,*)'hadronic yield:'
      ptltot=0.0
      rmstot=0.0
      vartot=0.0

       endif

      amgc=0.0
      samgc=0.0

       do ians=1,nspecs

c     hadronic yield
c     --------------
       if(ist.eq.0)then

      call hgclim(a,b)
      if(b.eq.0.0)then
      hden=0.0
      else
      call uttraq(hgcfhn,a,b,hden)
      endif
      ptlngc(ians)=hden*volu*gspecs(ians)/2./pi**2/hquer**3

       else

       if((chemgc(ians)/tem).gt.70.)then
      hpd=1.e30
       else
      hpd=exp(chemgc(ians)/tem)
       endif
       if(aspecs(ians).ne.0.)then
      fk2=hgcbk(2,aspecs(ians)/tem)
      hpd=hpd*gspecs(ians)*aspecs(ians)**2*tem*fk2
     */2./pi**2/hquer**3
       else
      hpd=hpd*gspecs(ians)*tem**3/pi**2/hquer**3
       endif
      ptlngc(ians)=hpd*volu

       endif

      ptltot=ptltot+ptlngc(ians)
      amgc=amgc+ptlngc(ians)*aspecs(ians)
      if(amgc.ge.tecm)amgc=tecm*0.9

c     standard deviation
c     ------------------
      rmsngc(ians)=0.0

       if(ist.eq.0)then

      call uttraq(hgcfhw,a,b,var)
      var=var*gspecs(ians)*volu/2./pi**2/hquer**3
      vartot=vartot+var
      if(var.ge.0.0)rmsngc(ians)=sqrt(var)
      samgc=samgc+var*aspecs(ians)

       else

      if(ptlngc(ians).ge.0.0)rmsngc(ians)=sqrt(ptlngc(ians))
      vartot=vartot+ptlngc(ians)
      samgc=samgc+ptlngc(ians)*aspecs(ians)

       endif


      if(ish.ge.7)write(ifch,'(2x,a,i5,a,2x,f8.4,5x,a,3x,f8.4)')
     *'m(',ispecs(ians),')  :',aspecs(ians),'mu :',chemgc(ians)
      if(ish.ge.5)write(ifch,'(1x,a,i5,a,2x,f8.4,2x,a,2x,f10.4)')
     *'<N(',ispecs(ians),')> :',ptlngc(ians),'sigma :',rmsngc(ians)

       enddo

      if(vartot.ge.0.0)rmstot=sqrt(vartot)
      if(samgc.ge.0.0)samgc=sqrt(samgc)
      if(amgc.ge.tecm)samgc=sqrt(amgc)
      if(ish.ge.5)write(ifch,'(1x,a,2x,f8.4,2x,a,2x,f10.4)')
     *'<N(  all)> :',ptltot,'sigma :',rmstot
      if(ish.ge.5)write(ifch,'(1x,a,2x,f8.4,2x,a,2x,f10.4)')
     *'<M_tot>    :',amgc,'sigma :',samgc

      return
      end

c------------------------------------------------------------------------
      subroutine hgctbo(iba)
c------------------------------------------------------------------------
c returns new tem using boltzmann statistics in analytic form
c  input:
c    chemgc
c    tecm/volu
c  output:
c    tem
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      external hgcbk
      external hgcbk1
      iba=0
      k=1
      t1=0.0
      t2=1.0

      goto15

10    tem=t1+.5*(t2-t1)
      if(tem.le.1.e-7)return
15    eden=0.0

        do i=1,nspecs

       if(aspecs(i).ne.0)then
      if(tem.ne.0.)arr=aspecs(i)/tem
      cba=(aspecs(i)/tem+12.*tem/aspecs(i)-3.*chemgc(i)/aspecs(i))
     **hgcbk(2,arr)+(3.-chemgc(i)/tem)*hgcbk1(arr)
       else
      cba=4.*tem-chemgc(i)
       endif

      if(cba.lt.0.0)then
      iba=1
      return
      endif

      x=0.
      if(tem.ne.0.)x=chemgc(i)/tem

       if(x.le.70.)then
      y=exp(x)
       else
      y=1.e30
       endif

       if(aspecs(i).ne.0.)then
      edi=y*(3./arr*hgcbk(2,arr)+hgcbk1(arr))
     **gspecs(i)*aspecs(i)**3*tem/2./pi**2/hquer**3
       else
      edi=y*3.*gspecs(i)*tem**4/pi**2/hquer**3
       endif

      eden=eden+edi

        enddo

      if(iabs(ispecs(nspecs)).lt.10)
     *eden=eden+(8.*pi**2*tem**4/15.+bag4rt**4)/hquer**3

      de=abs(eden-(tecm/volu))
      if(de.le.gen*(tecm/volu).or.de.le.genm)return
c     if(eden.ge.100.)return

       if(eden.gt.(tecm/volu))then
      t2=tem
      else
      t1=tem
       endif

       if(k.gt.300)return

      k=k+1
      goto10
      end

c----------------------------------------------------------------------
      subroutine hgctex
c----------------------------------------------------------------------
c returns new tem using massive quantum statistics in integral form
c  input:
c    chemgc
c    tecm/volu
c  output:
c    tem
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm
      external hgcfhe
      k=1
      t1=0.0
      t2=tem+0.1
      goto15

c     new temperature
c     ---------------
10    tem=t1+.5*(t2-t1)
15    continue
      if(tem.le.1.e-6)return
      eden=0.0

       do ians=1,nspecs
      call hgclim(a,b)
      if(b.eq.0.0)then
      edi=0.0
      else
      call uttraq(hgcfhe,a,b,edi)
      endif
      edi=edi*gspecs(ians)/2./pi**2/hquer**3
      eden=eden+edi
       enddo

      if(iabs(ispecs(nspecs)).lt.10)
     *eden=eden+(8.*pi**2*tem**4/15.+bag4rt**4)/hquer**3

      de=abs(eden-(tecm/volu))
      if(de.le.gen*(tecm/volu).or.de.le.genm)return

       if(eden.gt.(tecm/volu))then
      t2=tem
      else
      t1=tem
       endif

       if(k.gt.300)then
       if(ish.ge.5)
     *write(ifch,*)'failure in tex'
      return
       endif

      k=k+1
      goto10
      end

c-----------------------------------------------------------------
      subroutine hgctm0
c-----------------------------------------------------------------
c returns new tem using massless quantum statistics in analytic form
c  input:
c    chemgc
c    tecm/volu
c  output:
c    tem
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/ciakt/gen,iafs,ians,genm

      k=1

      t1=0.0
      t2=1.0
10    tem=t1+.5*(t2-t1)
      if(tem.le.1.e-6)return
      eden=0.0

        do i=1,nspecs

      igsp=int(gspecs(i))
      if(mod(igsp,2).eq.0)then
      edhm0=7./240.*pi**2*tem**4+chemgc(i)**2*tem**2/8.
     *+chemgc(i)**4/pi**2/16.
      else
      edhm0=pi**2*tem**4/30.+chemgc(i)**2*tem**2/4.
     *-chemgc(i)**4/pi**2/16.
      endif
      edi=edhm0*gspecs(i)/hquer**3


      eden=eden+edi
        enddo

      if(iabs(ispecs(nspecs)).lt.10)
     *eden=eden+(8.*pi**2*tem**4/15.+bag4rt**4)/hquer**3

      de=abs(eden-(tecm/volu))
      if(de.le.gen*(tecm/volu).or.de.le.genm)return

       if(eden.gt.(tecm/volu))then
      t2=tem
      else
      t1=tem
       endif

       if(k.gt.300)then
       if(ish.ge.5)
     *write(ifch,*)'failure in tm0'
      return
       endif

      k=k+1
      goto10
      end

c------------------------------------------------------------------------------
      subroutine hnbcor(mode)
c------------------------------------------------------------------------------
c determines(mode=1) and plots (mode=2) two particle  correlations
c for the configurations /confg/
c------------------------------------------------------------------------------
      include 'epos.inc'
      integer bns
      parameter (maxp=500,bns=100)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      dimension zwei(bns),zz(bns)!,phi(bns),yy(bns)
      common/cor/wert(bns),cwert(bns)
      character*6 cen,cvol

           if(mode.eq.1)then

      nctcor=nctcor+1

      if(nctcor.eq.1)then
      do nn=1,bns
      wert(nn)=0
      cwert(nn)=0
      enddo
      endif

      ll=0

      do ii=1,np-1
      do jj=ii+1,np

      ll=ll+1
      prod=0

      do kk=1,3
      prod=prod+pcm(kk,ii)*pcm(kk,jj)
      enddo

      cs=prod/pcm(5,ii)/pcm(5,jj)

      if(abs(cs).gt.1.)then
      cs=aint(cs)
      ang=acos(cs)
      else
      ang=acos(cs)
      endif

      if(cs.eq.1.)then
      nk=bns
      nw=1
      elseif(ang.eq.pi)then
      nk=1
      nw=bns
      else
      nw=1+aint(ang/pi*bns)
      nk=1+aint((cs+1.)/2.*bns)
      endif
      nw=min(nw,bns)
      nk=min(nk,bns)

      wert(nw)=wert(nw)+1
      cwert(nk)=cwert(nk)+1

      enddo
      enddo

           elseif(mode.eq.2)then

      do mm=1,bns
c      phi(mm)=.5*pi/bns+(mm-1)*pi/bns
      zwei(mm)=.5*2./bns+(mm-1)*2./bns-1.
c      yy(mm)=wert(mm)/nctcor
      zz(mm)=cwert(mm)/nctcor
      enddo

      write(cen,'(f6.1)')tecm
      write(cvol,'(f6.1)')volu

      write(ifhi,'(a)')    'newpage zone 1 1 1 openhisto'
      write(ifhi,'(a)')    'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')    'xrange -1 1'
      write(ifhi,'(a)')    'text 0 0 "xaxis cosine"'
      write(ifhi,'(a)')    'text 0 0 "yaxis counts"'
      write(ifhi,'(a)')    'text 0.4 0.91 "V='//cvol//'"'
      write(ifhi,'(a)')    'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a)')    'array 2'
         do mm=1,bns
      write(ifhi,'(2e13.5)')zwei(mm),zz(mm)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

           endif

      return
      end

c----------------------------------------------------------------------
      subroutine hnbfac(faclog)
c----------------------------------------------------------------------
c  returns log of factor for phase space weight
c  faclog= log{ prod[ m_i*(2*s_i+1)*volu/4/pi**3/hquer**3/(n_l+1-i) ] }
c      ~~~~~~~~~~~~~~
c  corresponds to eq. 67 of micro paper :
c         Cvol * Cdeg * Cident * Cmicro
c    the factors partly compensate each other !!
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
c      integer ii(maxp)
      common /clatt/nlattc,npmax

      faclog=0

c sum_i log m_i*g_i*volu/4/pi**3/hquer**3/(n_l+1-i) -> flog
      flog=0
      do i=1,np
      call hnbfaf(i,gg,am,ioma)
      flog=flog+alog(gg*am*volu/4/pi**3/hquer**3/(nlattc+1-i))
      enddo
      faclog=faclog+flog

      return
      end

c----------------------------------------------------------------------
      subroutine hnbfaf(i,gg,am,ioma)
c----------------------------------------------------------------------
c  returns degeneracy gg and mass am  for factor f5
c----------------------------------------------------------------------
      common/metr1/iospec,iocova,iopair,iozero,ioflac,iomom
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/drop6/tecm,volu

      ioma=5

      hquer=0.197327
      cc=0.216416
      dd=13.773935

      call hnbspi(ident(i),spideg)
      gg=spideg

      if(ioma.eq.1)am=amass(i)
      if(ioma.eq.2)am=tecm/np
      if(ioma.eq.3)am=1
      if(ioma.eq.4)
     *am=cc*dd*gg**(-0.25)*(tecm/volu)**(0.25)*hquer**(0.75)
      if(ioma.eq.5)am=0.5   ! 1GeV / 2  (dimension energy)
      if(iocova.eq.2)then
      am=0.5                ! 1 / 2     (no dimension)
      ioma=0
      endif
      return
      end

cc----------------------------------------------------------------------
c      subroutine hnbids(jc,ids,iwts,i)
cc----------------------------------------------------------------------
cc  returns i id-codes ids() corr to jc  and their weights iwts()
cc----------------------------------------------------------------------
c      parameter (mxids=200,mspecs=56,nflav=6)
c      common/metr1/iospec,iocova,iopair,iozero,ioflac,iomom
c      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
c      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
c      integer ids(mxids),jc(nflav,2),iwts(mxids),jc1mi2(nflav)
c
c      if(nspecs+1.gt.mxids)call utstop('hnbids: mxids too small&')
c
c      do n=1,nflav
c      jc1mi2(n)=jc(n,1)-jc(n,2)
c      enddo
c
c      i=0
c
c      do n=1,nflav
c      if(jc1mi2(n).ne.0)goto1
c      enddo
c      i=i+1
c      ids(i)=0
c      iwts(i)=iozero
c    1 continue
c
c           do j=1,nspecs
c      do n=1,nflav
c      if(jc1mi2(n).ne.ifok(n,j))goto2
c      enddo
c      i=i+1
c      ids(i)=ispecs(j)
c      iwts(i)=1
c    2 continue
c           enddo
c
c      return
c      end
c
c----------------------------------------------------------------------
      subroutine hnbiiw(x,f,df)
c----------------------------------------------------------------------
c returns fctn value and first derivative at x of the
c i-th integrated weight fctn minus random number
c for the asympotic phase space integral.
c input:
c   x:   x-value
c   iii: i-value (via common/ciiw/iii,rrr)
c   rrr: random number   ( " )
c output:
c   f:   fctn value
c   df:  first derivative
c----------------------------------------------------------------------
      common/ciiw/iii,rrr
      i=iii
      f=x**(2*i-2)*(i-(i-1)*x**2)-rrr
      df=2*i*(i-1)*(x**(2*i-3)-x**(2*i-1))
      return
      end

c----------------------------------------------------------------------
      subroutine hnbini(iret)
c----------------------------------------------------------------------
c  generates initial configuration
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/crnoz/rnoz(maxp-1)
      common/citer/iter,itermx
      common/cfact/faclog
      common/chnbin/nump,ihadro(maxp)
      common /clatt/nlattc,npmax
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      common/ctaue/taue
      if(ish.ge.7)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbini ',('-',i=1,30)

      iter=0

      nlattc=8*(tecm/10)*(1/(tecm/volu))**0.2*(nspecs/3.)**0.3
      if(aspecs(1).lt.0.010)nlattc=nlattc*3
      nlattc=max(nlattc,20)
      if(iternc.lt.0)iternc=1.500*nlattc

      itermx=iterma
      if(itermx.le.0)then
        e=tecm/volu
        b=1.1*(e+0.33)**0.66
        a=13.*(e+0.13)**(-0.65)
        tm=34.*(e+0.65)**(-0.61)
        t=a+b*volu
        taue=max(t,tm)
        itermx=(-itermx)*taue
      else
        taue=0
      endif
      if(ish.ge.5)write(ifch,*)'itermx:',itermx

      if(iternc.gt.itermx/2)iternc=itermx/2

      if(ioinco.eq.0)then
            call hnbmin(keu,ked,kes,kec)
            if(iograc.eq.1)call hgcaaa
      elseif(ioinco.ge.1)then
            nk=keu+ked+ked+kec
            if(tecm.lt.1.5.and.nk.eq.0)then
                      call hnbmin(keu,ked,kes,kec)
            elseif(tecm.lt.2.0.and.nk.ne.0)then
                      call hnbmin(keu,ked,kes,kec)
            else
                      call hgcaaa
                      call hgcnbi(iret)
                      if(iret.eq.1)then
                        call hnbmin(keu,ked,kes,kec)
                        if(ish.ge.5)then
                          write(ifch,*)'hadron set from hnbmin:'
                          write(ifch,'(10i6)')(ihadro(k),k=1,nump)
                        endif
                      endif
           endif
      endif

      np=nump+nadd
      if(np.gt.maxp)stop'np too large'

      nlattc=max(nlattc,1+int(np*1.2))
c      print *,np,nlattc
      if(nlattc-1.gt.maxp)stop'maxp too small'

      do i= 1, nlattc-1
      rnoz(i)=rangen()
      enddo

      if(nadd.gt.0)then
      do i=nump+1,np
      ihadro(i)=110
      enddo
      endif

           do i=1,np
      ident(i)=ihadro(i)
      amass(i)=-1
      do j=1,nspecs
      if(ident(i).eq.ispecs(j))then
      amass(i)=aspecs(j)
      goto1
      endif
      enddo
    1 continue
      if(amass(i).lt.0.)
     *call utstop('hnbini: invalid particle species&')
           enddo

      if(iocova.eq.1)call hnbody    !covariant
      if(iocova.eq.2)call hnbodz    !noncovariant
      call hnbfac(faclog)
      wtlog=wtxlog+faclog

      iret=0
      if(wtlog.le.-0.99999E+35)then
        if(ish.ge.1) then
          call utmsg('hnbini')
          write(ifch,*)'*****  wtlog for initl config < -1E+35'
          write(ifch,*)'***** wtlog:',wtlog
          write(ifch,*)'***** droplet mass:',tecm
          write(ifch,*)'***** flavour:'
          write(ifch,*)'*****',keu,ked,kes,kec,keb,ket
          write(ifch,'(1x,a,1x,10i6)')'*****',(ihadro(i),i=1,nump)
          call utmsgf
        endif
        iret=1
        goto1000
      endif

      if(ish.ge.7)then
        write(ifch,*)'initial configuration:'
        call hnbwri
      endif

      itermx=iterma
      if(itermx.le.0)then
      e=tecm/volu
      b=1.1*(e+0.33)**0.66
      a=13.*(e+0.13)**(-0.65)
      tm=34.*(e+0.65)**(-0.61)
      t=a+b*volu
      taue=max(t,tm)
      itermx=(-itermx)*taue
      else
      taue=0
      endif
      if(ish.ge.5)write(ifch,*)'itermx:',itermx

      if(iternc.gt.itermx/2)iternc=itermx/2

      nacc=0
      nrej=0

 1000 continue

      if(ish.ge.7)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbini ',('-',i=1,10)

      return
      end

cc----------------------------------------------------------------------
c      subroutine hnbint(tecmx,nevtxx,nsho)
cc----------------------------------------------------------------------
cc  calculates phase space integral of the minimal hadron configuration
cc  compatibel with keu, ked, kes, kec for a total mass of tecm
cc  by employing nevtxx simulations and printing results every nsho events
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      parameter(maxp=500)
c      common/chnbin/nump,ihadro(maxp)
c      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
c      tecm=tecmx
c      write(ifch,*)
c      write(ifch,'(1x,a,4i3,a,f10.4)')'droplet id:',keu,ked,kes,kec
c     *,'   droplet mass:',tecm
c      call hnbmin(keu,ked,kes,kec)
c      np=nump
c      if(np.gt.maxp)stop'np too large'
c      do i=1,np
c      id=ihadro(i)
c      if(id.eq.30)then
c          call idmass(2130,am)
c          amass(i)=2*am-0.100
c      else
c         call idmass(id,amass(i))
c      endif
c      enddo
c      wts=0
c      n=0
c           do ll=1,nevtxx
c      n=n+1
c      if(iocova.eq.1)call hnbody
c      if(iocova.eq.2)call hnbodz
c      wt=exp(wtxlog)
c      wts=wts+wt
c      if(mod(n,nsho).eq.0)
c     *write(ifch,'(a,i7,3x,a,e13.6,3x,a,e13.6,3x,a,e13.6)')
c     *'n:',n,'weight:',wt,'wts/n:',wts/n,'error:',wts/n/sqrt(1.*n)
c           enddo
c      return
c      end
cc----------------------------------------------------------------------
      subroutine hnbmet
c----------------------------------------------------------------------
c  change (or not) configuration via metropolis
c  configuration=np,tecm,amass(),ident(),pcm(),volu,wtlog
c    (common /confg/)
c  nlattc (in /clatt/) must be set before calling this routine
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/crnoz/rnoz(maxp-1)
      real rnozo(maxp-1)
      common/cfact/faclog
      dimension amasso(maxp),idento(maxp),pcmo(5,maxp)
      integer jc(nflav,2),jc1(nflav,2),jc2(nflav,2)
      common/citer/iter,itermx
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (literm=500)
      common/cmet/kspecs(mspecs),liter,lspecs(literm,mspecs)
     *,iterl(literm),iterc(literm)
c      parameter (mxpair=mspecs**2*4)
      common /clatt/nlattc,npmax
      parameter (nhise=100)
      common/chise/hise(mspecs,nhise)
      integer id1old(2),id2old(2),id1new(2),id2new(2)
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      if(ish.ge.7)then
        write(ifch,*)('-',i=1,10)
     *,' entry sr hnbmet ',('-',i=1,30)
        write(ifch,'(1x,a,i4)')'iteration:',iter
      endif
      if(mod(iter,iterpr).eq.0)write(ifmt,*)'iteration:',iter
      if(maxp.gt.np)then
      do n=np+1,maxp
      ident(n)=0
      enddo
      endif

c     for iter=1
c     ----------
           if(iter.eq.1)then
      liter=1
      do i=1,nspecs
      kspecs(i)=0
      nptot=0
      do li=1,literm
      lspecs(li,i)=0
      enddo
      enddo
      do li=1,literm
      iterc(li)=0
      enddo
      do j=1,mspecs
      do i=1,nhise
      hise(j,i)=0
      enddo
      enddo
      call hnbzmu(-1)
           endif

c     remember old configuration
c     --------------------------
      wtlo=wtlog
      wtlox=wtxlog
      faclo=faclog
      npo=np
      if(np-1.gt.0)then
      do i=1,np-1
      rnozo(i)=rnoz(i)
      enddo
      endif
      if(np.gt.0)then
      do i=1,np
      amasso(i)=amass(i)
      idento(i)=ident(i)
      do j=1,5
      pcmo(j,i)=pcm(j,i)
      enddo
      enddo
      endif

c     determine pair, construct new pair, update ident
c     ------------------------------------------------
      xab=1
      xba=1
           if(iopair.eq.1)then
c     (single pair method)
      call hnbpad(1,n1,n2,n3,n4,mm,jc)
      id1old(1)=ident(n1)
      id2old(1)=ident(n2)
      id1old(2)=0
      id2old(2)=0
      call hnbpaj(jc,iwpair,id1,id2)
      ident(n1)=id1
      ident(n2)=id2
      call hnbrmz
      id1new(1)=id1
      id2new(1)=id2
      id1new(2)=0
      id2new(2)=0
      xab=1
      xba=1
      nzold=0
      if(id1old(1).eq.0)nzold=nzold+1
      if(id2old(1).eq.0)nzold=nzold+1
      nznew=0
      if(id1new(1).eq.0)nznew=nznew+1
      if(id2new(1).eq.0)nznew=nznew+1

c     determine 2 pairs, construct 2 new pairs, update ident
c     ------------------------------------------------------
           elseif(iopair.eq.2)then
c     (double pair method)
      kkk=0
   25 call hnbpad(1,n1,n2,n3,n4,mm,jc)
      kkk=kkk+1
      id1old(1)=ident(n1)
      id2old(1)=ident(n2)
      call hnbpai(id1,id2,jc1)
      ident(n1)=id1
      ident(n2)=id2
      id1new(1)=id1
      id2new(1)=id2
      do i=1,nflav
      do j=1,2
      jc(i,j)=jc(i,j)-jc1(i,j)
      jc2(i,j)=jc(i,j)
      enddo
      enddo
    2 call hnbpad(2,n1,n2,n3,n4,mm,jc1)
      id1old(2)=ident(n3)
      id2old(2)=ident(n4)
      do i=1,nflav
      do j=1,2
      jc(i,j)=jc(i,j)+jc1(i,j)
      enddo
      enddo
      call hnbpaj(jc,iwpair,id1,id2)
      if(iwpair.eq.0)then
      do i=1,nflav
      do j=1,2
      jc(i,j)=jc2(i,j)
      enddo
      enddo
      if(ish.ge.7)write(ifch,*)'no pair possible'
      goto2
      endif
      ident(n3)=id1
      ident(n4)=id2
      id1new(2)=id1
      id2new(2)=id2
      call hnbrmz
      if(ish.ge.7)write(ifch,*)'wt-sum of 2. pairs (-->):',iwpair
     *,'   chosen pair:',id1,id2
      call hnbpaj(jc1,iwpais,idum1,idum2)
      if(ish.ge.7)write(ifch,*)'wt-sum of 2. pairs (<--):',iwpais
      nzold=0
      if(id1old(1).eq.0)nzold=nzold+1
      if(id2old(1).eq.0)nzold=nzold+1
      if(id1old(2).eq.0)nzold=nzold+1
      if(id2old(2).eq.0)nzold=nzold+1
      if(ish.ge.7)write(ifch,*)'number of zeros (old):',nzold
      nznew=0
      if(id1new(1).eq.0)nznew=nznew+1
      if(id2new(1).eq.0)nznew=nznew+1
      if(id1new(2).eq.0)nznew=nznew+1
      if(id2new(2).eq.0)nznew=nznew+1
      if(ish.ge.7)write(ifch,*)'number of zeros (new):',nznew
      if(iorejz.eq.1.and.nzold.eq.4.and.nznew.eq.4.and.kkk.le.50)goto25
      xab=1./iwpair*iozero**nznew
      xba=1./iwpais*iozero**nzold
      if(ish.ge.7)write(ifch,*)'asymmetry factor:',xba/xab
           else
      call utstop('hnbmet: invalid choice for iopair&')
           endif

c     determine masses/momenta/weight of trial configuration
c     ------------------------------------------------------
      if(np.ge.2)then
           do i=1,np
      amass(i)=-1
      do j=1,nspecs
      if(ident(i).eq.ispecs(j))then
      amass(i)=aspecs(j)
      goto1
      endif
      enddo
    1 continue
      if(amass(i).lt.0.)
     *call utstop('hnbmet: invalid particle species&')
           enddo
      keepr=0
c-c   call hnbolo(1000) !instead of "call hnbody" for testing
      keepr=1
      if(iocova.eq.1)call hnbody
      if(iocova.eq.2)call hnbodz
      else
      wtxlog=-1e35
      endif
      call hnbfac(faclog)
      wtlog=wtxlog+faclog
      if(ish.ge.7)then
        write(ifch,*)'trial configuration:'
        call hnbwri
      endif

c     accept or not trial configuration (metropolis)
c     ----------------------------------------------
      if(ish.ge.7)write(ifch,'(1x,a,4i5,a,4i5,a)')
     *'metropolis decision for '
     *,id1old(1),id2old(1),id1old(2),id2old(2),'   -->  '
     *,id1new(1),id2new(1),id1new(2),id2new(2),' :'
      iacc=0
           if(wtlog-wtlo.lt.30.)then
      q=exp(wtlog-wtlo)*xba/xab
      r=rangen()
      if(r.le.q)iacc=1
      if(ish.ge.7)write(ifch,*)'new weight / old weight:',q,'    '
     *,'random number:',r
           else
      iacc=1
      if(ish.ge.7)write(ifch,*)'log new weight / old weight:'
     *,wtlog-wtlo
           endif
           if(iacc.eq.1)then
      if(ish.ge.7)write(ifch,*)'new configuration accepted'
      nacc=nacc+1
      naccit(iter)=1
           else
      if(ish.ge.7)write(ifch,*)'old configuration kept'
      nrej=nrej+1
      wtlog=wtlo
      wtxlog=wtlox
      faclog=faclo
      np=npo
      if(np-1.gt.0)then
      do i=1,np-1
      rnoz(i)=rnozo(i)
      enddo
      endif
      if(np.gt.0)then
      do i=1,np
      amass(i)=amasso(i)
      ident(i)=idento(i)
      do j=1,5
      pcm(j,i)=pcmo(j,i)
      enddo
      enddo
      endif
           endif
           if(ioobsv.eq.0)then
      npit(iter)=np
      if(iter.gt.iternc)nptot=nptot+np
      else
      npob=0
      do i=1,np
      if(ioobsv.eq.ident(i))npob=npob+1
      enddo
      npit(iter)=npob
      if(iter.gt.iternc)nptot=nptot+npob
           endif
      if(ish.ge.7)then
        write(ifch,*)'actual configuration:'
        call hnbwri
        if(ish.eq.27)stop'change this?????????????' !call hnbcor(1)
      endif

c     printout/return
c     ---------------
      if(iosngl.ne.nrevt+1.and.iocite.ne.1)goto1000
      npmax=max(npmax,np)
           if(liter.le.literm)then
      iterc(liter)=iterc(liter)+1
      do i=1,np
      do j=1,nspecs
      if(ident(i).eq.ispecs(j))then
      lspecs(liter,j)=lspecs(liter,j)+1
      goto8
      endif
      enddo
    8 continue
      enddo
      if(mod(iter,iterpl).eq.0)then
      iterl(liter)=iter
      liter=liter+1
c     if(liter.le.literm)then
c     iterc(liter)=iterc(liter-1)
c     do j=1,nspecs
c     lspecs(liter,j)=lspecs(liter-1,j)
c     enddo
c     endif
      endif
           endif
      if(iter.le.iternc)return

           do i=1,np
      call hnbzen(i)  !fill energy histogram
      do j=1,nspecs
      if(ident(i).eq.ispecs(j))then
      kspecs(j)=kspecs(j)+1
      goto7
      endif
      enddo
    7 continue
           enddo
      call hnbzmu(1)  !fill multiplicity histogram

           if(iter.eq.itermx.and.npmax.ge.nlattc.and.ish.ge.1)then
      call utmsg('hnbmet')
      write(ifch,*)'*****  nlattc too small'
      write(ifch,*)'nlattc:',nlattc,'   npmax:',npmax
      call utmsgf
           endif

1000  continue
      if(ish.ge.7)then
        write(ifch,*)'accepted proposals:',nacc
     *,'  rejected proposals:',nrej
        write(ifch,*)('-',i=1,30)
     *,' exit sr hnbmet ',('-',i=1,10)
      endif
      return
      end

c----------------------------------------------------------------------
      subroutine hnbmin(keux,kedx,kesx,kecx)
c----------------------------------------------------------------------
c  returns min hadron set with given u,d,s,c content
c  input:
c     keux: net u quark number
c     kedx: net d quark number
c     kesx: net s quark number
c     kecx: net c quark number
c  output (written to /chnbin/):
c     nump: number of hadrons
c     ihadro(n): hadron id for n'th hadron
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/chnbin/nump,ihadro(maxp)
      logical wri
      character f1*11
      wri=.false.
      if(ish.ge.7)wri=.true.
      if(wri)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbmin ',('-',i=1,30)

      nump=0
      f1='(4i3,i7,i6)'
      ke=iabs(keux+kedx+kesx+kecx)

      if(keux+kedx+kesx+kecx.ge.0)then
      keu=keux
      ked=kedx
      kes=kesx
      kec=kecx
      isi=1
      else
      keu=-keux
      ked=-kedx
      kes=-kesx
      kec=-kecx
      isi=-1
      endif
      if(wri)write(ifch,'(4i3)')keux,kedx,kesx,kecx
      if(wri)write(ifch,'(4i3)')keu,ked,kes,kec

c get rid of anti-c and c (140, 240, -140, -240)
      if(kec.ne.0)then
   10 continue
      if(kec.lt.0)then
      kec=kec+1
      if(keu.gt.ked)then
      keu=keu-1
      nump=nump+1
      ihadro(nump)=140
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      else
      ked=ked-1
      nump=nump+1
      ihadro(nump)=240
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      goto10
      endif
   11 continue
      if(kec.gt.0)then
      kec=kec-1
      if(keu.lt.ked)then
      keu=keu+1
      nump=nump+1
      ihadro(nump)=-140
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      else
      ked=ked+1
      nump=nump+1
      ihadro(nump)=-240
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      goto11
      endif
      endif

c get rid of anti-s (130,230)
    5 continue
      if(kes.lt.0)then
      kes=kes+1
      if(keu.ge.ked)then
      keu=keu-1
      nump=nump+1
      ihadro(nump)=130
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      else
      ked=ked-1
      nump=nump+1
      ihadro(nump)=230
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      goto5
      endif

c get rid of anti-d (120, -230)
   6  continue
      if(ked.lt.0)then
      ked=ked+1
      if(keu.ge.kes)then
      keu=keu-1
      nump=nump+1
      ihadro(nump)=120
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      else
      kes=kes-1
      nump=nump+1
      ihadro(nump)=-230
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      goto6
      endif

c get rid of anti-u (-120, -130)
    7 continue
      if(keu.lt.0)then
      keu=keu+1
      if(ked.ge.kes)then
      ked=ked-1
      nump=nump+1
      ihadro(nump)=-120
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      else
      kes=kes-1
      nump=nump+1
      ihadro(nump)=-130
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      goto7
      endif

      if(keu+ked+kes+kec.ne.ke)call utstop('hnbmin: sum_kei /= ke&')

      keq=keu+ked

c get rid of s (3331, x330, xx30)
      i=4
    2 i=i-1
    3 continue
      if((4-i)*kes.gt.(i-1)*keq)then
      kes=kes-i
      keq=keq-3+i
      nump=nump+1
      if(i.eq.3)ihadro(nump)=3331
      if(i.eq.2)ihadro(nump)=0330
      if(i.eq.1)ihadro(nump)=0030
           if(i.lt.3)then
      do j=1,3-i
      l=1+2*rangen()
      if(keu.gt.ked)l=1
      if(keu.lt.ked)l=2
      if(l.eq.1)keu=keu-1
      if(l.eq.2)ked=ked-1
      ihadro(nump)=ihadro(nump)+l*10**(4-j)
      enddo
           endif
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      if(kes.lt.0)call utstop('hnbmin: negative kes&')
      if(keq.lt.0)call utstop('hnbmin: negative keq&')
      goto3
      endif
      if(i.gt.1)goto2

      if(keu+ked.ne.keq)call utstop('hnbmin: keu+ked /= keq&')

c get rid of d (2221, 1220, 1120)
      i=4
   12 i=i-1
   13 continue
      if((4-i)*ked.gt.(i-1)*keu)then
      ked=ked-i
      keu=keu-3+i
      if(i.eq.3)then
      nump=nump+2
      ihadro(nump)=1220
      ihadro(nump-1)=-120
      else
      nump=nump+1
      if(i.eq.2)ihadro(nump)=1220
      if(i.eq.1)ihadro(nump)=1120
      endif
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      if(ked.lt.0)call utstop('hnbmin: negative ked&')
      if(keu.lt.0)call utstop('hnbmin: negative keu&')
      goto13
      endif
      if(i.gt.1)goto12

      if(ked.ne.0)call utstop('hnbmin: ked .ne. 0&')

c get rid of u (1111)
    9 continue
      if(keu.gt.0)then
      keu=keu-3
      nump=nump+2
      ihadro(nump)=1120
      ihadro(nump-1)=120
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      if(keu.lt.0)call utstop('hnbmin: negative keu&')
      goto9
      endif

      if(keu.ne.0)call utstop('hnbmin: keu .ne. 0&')

      if(isi.eq.-1)then
      do i=1,nump
      ihadro(i)=isi*ihadro(i)
      enddo
      endif

      do lo=1,2
      if(nump.lt.2)then
      nump=nump+1
      ihadro(nump)=110
      if(wri)write(ifch,f1)keu,ked,kes,kec,nump,ihadro(nump)
      endif
      enddo

      if(wri)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbmin ',('-',i=1,10)
      return
      end

c-------------------------------------------------------------
      subroutine hnbody
c-------------------------------------------------------------
c   formerly subr genbod from genlib (cernlib).
c   modified by K. Werner, march 94.
c   subr to generate n-body event
c   according to fermi lorentz-invariant phase space.
c   the phase space integral is the sum over the weights wt divided
c   by the number of events (sum wt / n).
c   adapted from fowl (cern w505) sept. 1974 by f. james.
c   events are generated in their own center-of-mass,
c   but may be transformed to any frame using loren4.
c
c   input to and output from subr thru common block config.
c   input:
c             np=number of outgoing particles
c             tecm=total energy in center-of-mass
c             amass(i)=mass of ith outgoing particle
c   output:
c             pcm(1,i)=x-momentum if ith particle
c             pcm(2,i)=y-momentum if ith particle
c             pcm(3,i)=z-momentum if ith particle
c             pcm(4,i)=energy of ith particle
c             pcm(5,i)=momentum of ith particle
c             wtxlog=log of weight of event
c--------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      dimension emm(maxp)
      dimension rno(3*maxp-4)
c     !pcm1 is linear equiv. of pcm to avoid double indices
      dimension em(maxp),pd(maxp),ems(maxp),sm(maxp)
     *,pcm1(5*maxp)
      common/cffq/ffqlog(maxp)
      common/ciiw/iii,rrr
      equivalence (nt,np),(amass(1),em(1)),(pcm1(1),pcm(1,1))
      logical wri
      data twopi/6.2831853073/
      external hnbiiw
ctp060829      nas=5 !must be at least 3
      wri=.false.
      if(ish.ge.7)wri=.true.
      if(wri)then
        write(ifch,*)('-',i=1,10)
     *,' entry sr hnbody ',('-',i=1,30)
        write(ifch,1200)np,tecm
        write(ifch,*)'particle masses:'
        write(ifch,'(1x,10f6.3)')(amass(n),n=1,np)
      endif

c..... initialization

      ktnbod=ktnbod + 1
      if(ktnbod.le.1)then
        !... ffq(n) = pi * (twopi)**(n-2) / (n-2)!
        ffqlog(1)=-1e35
        ffqlog(2)=alog(pi)
        do n=3,maxp
        ffqlog(n)=ffqlog(n-1)+log(twopi/(n-2))
        enddo
      endif

      if(nt.lt.2) goto 1001
      if(nt.gt.maxp) goto 1002
      ntm1=nt-1
      ntm2=nt-2
      ntnm4=3*nt - 4
      emm(1)=em(1)
      tm=0.0
      do 2 i=1,nt
      ems(i)=em(i)**2
      tm=tm+em(i)
    2 sm(i)=tm
      tecmtm=tecm-tm
      if(tecmtm.le.0.0) goto 1000
      emm(nt)=tecm
      wtmlog=alog(tecmtm)*ntm2 + ffqlog(nt) - alog(tecm)

c...fill rno with 3*nt-4 random numbers, the first nt-2 being ordered

      do 3 i= 1, ntnm4
    3 rno(i)=rangen()
      if(ntm2) 9,5,4
    4 continue
      call flpsore(rno,ntm2)

c...calculate emm().......M_i

      do 6 j=2,ntm1
    6 emm(j)=rno(j-1)*tecmtm+sm(j)

c...calculate wtlog

    5 continue
      wtxlog=wtmlog
      ir=ntm2
      do 7 i=1,ntm1
        pd(i)=hnbpdk(emm(i+1),emm(i),em(i+1))
        if(pd(i).gt.0.)then
          pdlog=alog(pd(i))
        else
          pdlog=-1e35
        endif
        wtxlog=wtxlog+pdlog
    7 continue

c...complete specification of event (raubold-lynch method)

      pcm(1,1)=0.0
      pcm(2,1)=pd(1)
      pcm(3,1)=0.0
      do i=2,nt
        pcm(1,i)=0.0
        pcm(2,i)=-pd(i-1)
        pcm(3,i)=0.0
        ir=ir+1
        bang=twopi*rno(ir)
        cb=cos(bang)
        sb=sin(bang)
        ir=ir+1
        c=2.0*rno(ir)-1.0
        s=sqrt(1.0-c*c)
        if(i.ne.nt)then
          esys=sqrt(pd(i)**2+emm(i)**2)
          beta=pd(i)/esys
          gama=esys/emm(i)
          do j=1,i
            ndx=5*j - 5
            aa= pcm1(ndx+1)**2 + pcm1(ndx+2)**2 + pcm1(ndx+3)**2
            pcm1(ndx+5)=sqrt(aa)
            pcm1(ndx+4)=sqrt(aa+ems(j))
            call hnbrt2(c,s,cb,sb,pcm,j)
            psave=gama*(pcm(2,j)+beta*pcm(4,j))
            pcm(2,j)=psave
          enddo
        else !(i.eq.nt)
          do j=1,i
            aa=pcm(1,j)**2 + pcm(2,j)**2 + pcm(3,j)**2
            pcm(5,j)=sqrt(aa)
            pcm(4,j)=sqrt(aa+ems(j))
            call hnbrt2(c,s,cb,sb,pcm,j)
          enddo
        endif
      enddo

c...returns

  9   continue
      goto1111

 1000 continue
      if(wri)
     *write(ifch,*)'available energy zero or negative -> wtxlog=-1e35'
      wtxlog=-1e35
      goto1111

 1001 continue
      if(wri)
     *write(ifch,*)'less than 2 outgoing particles -> wtxlog=-1e35'
      wtxlog=-1e35
      goto1111

 1002 continue
      write(ifch,*)'too many outgoing particles'
      write(ifch,1150) ktnbod
 1150 format(47h0 above error detected in hnbody at call number,i7)
      write(ifch,1200) np,tecm
 1200 format(' np:',i6/' tecm:',f10.5)
      write(ifch,*)'particle masses:'
      write(ifch,'(1x,10f6.3)')(amass(jk),jk=1,np)
      stop

1111  continue
      if(wri)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbody ',('-',i=1,10)
      return
      end

c---------------------------------------------------------------------------------------------------------
      SUBROUTINE FLPSORE(A,N)
C---------------------------------------------------------------------------------------------------------
C CERN PROGLIB# M103    FLPSOR          .VERSION KERNFOR  3.15  820113
C ORIG. 29/04/78
C
C   SORT THE ONE-DIMENSIONAL FLOATING POINT ARRAY A(1),...,A(N) BY
C   INCREASING VALUES
C
C-    PROGRAM  M103  TAKEN FROM CERN PROGRAM LIBRARY,  29-APR-78
C----------------------------------------------------------------------------------------------------------
      DIMENSION A(N)
      COMMON /SLATE/ LT(20),RT(20)
      INTEGER R,RT
C
      LEVEL=1
      LT(1)=1
      RT(1)=N
   10 L=LT(LEVEL)
      R=RT(LEVEL)
      LEVEL=LEVEL-1
   20 IF(R.GT.L) GO TO 200
      IF(LEVEL) 50,50,10
C
C   SUBDIVIDE THE INTERVAL L,R
C     L : LOWER LIMIT OF THE INTERVAL (INPUT)
C     R : UPPER LIMIT OF THE INTERVAL (INPUT)
C     J : UPPER LIMIT OF LOWER SUB-INTERVAL (OUTPUT)
C     I : LOWER LIMIT OF UPPER SUB-INTERVAL (OUTPUT)
C
  200 I=L
      J=R
      M=(L+R)/2
      X=A(M)
  220 IF(A(I).GE.X) GO TO 230
      I=I+1
      GO TO 220
  230 IF(A(J).LE.X) GO TO 231
      J=J-1
      GO TO 230
C
  231 IF(I.GT.J) GO TO 232
      W=A(I)
      A(I)=A(J)
      A(J)=W
      I=I+1
      J=J-1
      IF(I.LE.J) GO TO 220
C
  232 LEVEL=LEVEL+1
      IF((R-I).GE.(J-L)) GO TO 30
      LT(LEVEL)=L
      RT(LEVEL)=J
      L=I
      GO TO 20
   30 LT(LEVEL)=I
      RT(LEVEL)=R
      R=J
      GO TO 20
   50 continue

      do i=1,n-1
        if(a(i).gt.a(i+1))stop'FLPSORE: ERROR.                    '
      enddo

      RETURN
      END





c-------------------------------------------------------------
      subroutine hnbodz
c-------------------------------------------------------------
c   subr to generate n-body event
c   according to non-invariant phase space.
c   the phase space integral is the sum over the weights exp(wtxlog)
c   divided by the number of events.
c   ref.: hagedorn, nuov. cim. suppl ix, x (1958) 646.
c   events are generated in their own center-of-mass.
c
c   input to and output from subr is thru common block config.
c   input:
c             np=number of outgoing particles
c             tecm=total energy in center-of-mass
c             amass(i)=mass of ith outgoing particle
c   output:
c             pcm(1,i)=x-momentum of ith particle
c             pcm(2,i)=y-momentum of ith particle
c             pcm(3,i)=z-momentum of ith particle
c             pcm(4,i)=energy of ith particle
c             pcm(5,i)=momentum of ith particle
c             wtxlog=log of weight of event
c--------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common /clatt/nlattc,npmax
      common/cffq/ffqlog(maxp)
      dimension ti(maxp),xi(maxp),si(maxp),zi(maxp)
      common/crnoz/rnoz(maxp-1)
      double precision ps(5)

      call utpri('hnbodz',ish,ishini,6)
      if(ish.ge.6)write(ifch,1200)np,tecm
      if(ish.ge.6)write(ifch,*)'particle masses:'
      if(ish.ge.6)write(ifch,'(1x,10f6.3)')(amass(n),n=1,np)

c initialization ktnbod=1
      ktnbod=ktnbod + 1
      if(ktnbod.gt.1) goto 1
c     !ffqlog(n) = log{ (4*pi)**n  / (n-1)! }
      ffqlog(1)=alog(4*pi)
      do n=2,maxp
      ffqlog(n)=ffqlog(n-1)+alog(4*pi/(n-1))
      enddo
    1 continue
c set wtxlog -infinity for np<2
      if(np.lt.2) goto 1001
c special treatment for np=2
      if(np.eq.2)then
      if(tecm.lt.amass(1)+amass(2)+0.00001)goto1000
      p0=utpcm(tecm,amass(1),amass(2))
      wtxlog=alog( 4*pi*p0
     */(1/sqrt(amass(1)**2+p0**2)+1/sqrt(amass(2)**2+p0**2)) )
      if(ish.ge.7)
     *write(ifch,*)'wtxlog:',wtxlog,'   (np=2 treatment)'
      bang=2*pi*rangen()
      cb=cos(bang)
      sb=sin(bang)
      c=2.0*rangen()-1.0
      s=sqrt(1.0-c*c)
      do 9 i=1,2
      is=2*i-3
      pcm(5,i)=p0
      pcm(1,i)=is*pcm(5,i)*s*cb
      pcm(2,i)=is*pcm(5,i)*s*sb
      pcm(3,i)=is*pcm(5,i)*c
      pcm(4,i)=sqrt(amass(i)**2+p0**2)
    9 continue
      goto1111
      endif
c stop if np too large
      if(np.gt.maxp) goto 1002
c initialization all ktnbod
      tm=0.0
      do 2 i=1,np
      tm=tm+amass(i)
    2 continue
      tt=tecm-tm
      if(tt.le.0.0) goto 1000
c prefactor
      wtxlog=alog(tt)*(np-1) + ffqlog(np)
      if(ish.ge.7)
     *write(ifch,*)'wtxlog:',wtxlog,'   (prefactor)'
c fill rnoz with np-1 random numbers
      if(keepr.eq.0)then
      do 3 i= 1, np-1
    3 rnoz(i)=rangen()
      else
      do lo=1,iomom
      j=1+rangen()*nlattc
      rnoz(j)=rangen()
      enddo
      endif
c calculate z_i distributed as i*z*(i-1)
      do i= 1, np-1
      zi(i)=rnoz(i)**(1./i)
      enddo
c calculate x_i
      xi(np)=1
      do i=np-1,1,-1
      xi(i)=zi(i)*xi(i+1)
      enddo
c calculate t_i, e_i, p_i
      if(ish.ge.9)write(ifch,*)'calculate t_i, e_i, p_i ...'
      do i=1,np-1
      si(i)=xi(i)*tt
      enddo
      ti(1)=si(1)
      if(ti(1).le.0.)ti(1)=1e-10
      ti(np)=tt-si(np-1)
      if(ti(np).le.0.)ti(np)=1e-10
      do i=np-1,2,-1
      ti(i)=si(i)-si(i-1)
      if(ti(i).le.0.)ti(i)=1e-10
      enddo
      do i=1,np
      pcm(1,i)=0
      pcm(2,i)=0
      pcm(3,i)=0
      pcm(4,i)=ti(i)+amass(i)
      p52=ti(i)*(ti(i)+2*amass(i))
      if(p52.gt.0)then
      pcm(5,i)=sqrt(p52)
      else
      pcm(5,i)=ti(i)*sqrt(1+2*amass(i)/ti(i))
      endif
      enddo
c calculate wtxlog
      call hnbraw(7,200,w)
      if(w.gt.0.)then
      wtxlog=wtxlog+alog(w)
      else
      wtxlog=wtxlog-1e+30
      endif
      do 7 i=1,np
      wtxlog=wtxlog+alog(pcm(5,i))+alog(ti(i)+amass(i))
    7 continue
      if(ish.ge.7)
     *write(ifch,*)'wtxlog:',wtxlog
c print
      if(ish.ge.7)then
      write(ifch,*)'momenta:'
      do j=1,4
      ps(j)=0
      enddo
      do i=1,np
      do j=1,4
      ps(j)=ps(j)+pcm(j,i)
      enddo
      write(ifch,'(1x,i3,5x,5f12.5)')i,(pcm(j,i),j=1,5)
      enddo
      ps(5)=dsqrt(ps(1)**2+ps(2)**2+ps(3)**2)
      write(ifch,'(1x,a4,8x,5f12.5)')'sum:',(sngl(ps(j)),j=1,5)
      endif
      if(w.le.0.)goto1111
c complete specification of event (random rotations and then deformations)
      call hnbrot
      if(ish.ge.7)write(ifch,*)'momenta after rotations:'
      call hnbrop(96,0)
      call hnbrod
      if(ish.ge.7)write(ifch,*)'momenta after deformations:'
      call hnbrop(96,1)
      goto1111

c error returns
 1000 continue
      if(ish.ge.6)
     *write(ifch,*)'available energy zero or negative -> wtxlog=-1e35'
      wtxlog=-1e35
      goto1111

 1001 continue
      if(ish.ge.6)
     *write(ifch,*)'less than 2 outgoing particles -> wtxlog=-1e35'
      wtxlog=-1e35
      goto1111

 1002 continue
      write(ifch,*)'too many outgoing particles'
      write(ifch,1150) ktnbod
 1150 format(47h0 above error detected in hnbody at call number,i7)
      write(ifch,1200) np,tecm
 1200 format(' np:',i6/' tecm:',f10.5)
      write(ifch,*)'particle masses:'
      write(ifch,'(1x,10f6.3)')(amass(jk),jk=1,np)
      stop

1111  continue
      call utprix('hnbodz',ish,ishini,6)
      return
      end

c-----------------------------------------------------------------------
      subroutine hnbolo(loops)
c-----------------------------------------------------------------------
c  loop over hnbody
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      a=0
      k=0
      do j=1,loops
c-c   if(mod(j,iterpr).eq.0)write(ifmt,*)'     iteration:',iter,j
      if(iocova.eq.1)call hnbody
      if(iocova.eq.2)call hnbodz
      if(ish.ge.8)write(ifch,*)'j:',j,'   wtxlog:',wtxlog
           if(wtxlog.gt.-1e30)then
      k=k+1
      if(k.eq.1)c=wtxlog
           if(a.gt.0.)then
      if(alog(a).lt.wtxlog-c-20)then
      a=0
      c=wtxlog
      endif
           endif
      a=a+exp(wtxlog-c)
           endif
      if(ish.ge.8)write(ifch,*)'k:',k,'   c:',c
      enddo
      a=a/loops
      wtxlog=alog(a)+c
      return
      end

c-----------------------------------------------------------------------
      function hnbpdk(a,b,c)
c-----------------------------------------------------------------------
c  formerly pdk from cernlib
c  returns momentum p for twobody decay  a --> b + c
c           a, b, c are the three masses
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c  this p is related to twobody phase space as R2 = pi * p /a
c-----------------------------------------------------------------------
      double precision aa,bb,cc,a2,b2,c2
      aa=a
      bb=b
      cc=c
      a2=aa*aa
      b2=bb*bb
      c2=cc*cc
      if(a2 + (b2-c2)**2/a2-2.0*(b2+c2).le.0.)then
      hnbpdk = 0
      else
      hnbpdk = 0.5*dsqrt(a2 + (b2-c2)**2/a2 - 2.0*(b2+c2))
      endif
      return
      end

c----------------------------------------------------------------------
      subroutine hnbpad(k,n1,n2,n3,n4,mm,jc)
c----------------------------------------------------------------------
c  k=1: determ pair indices k1,k2
c  k=2: determ pair indices k3,k4 (.ne. n1,n2)
c  k=1 and k=2: mm: type of pair, jc: flavour of pair
c----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),ic(2),jc1(nflav,2),ic1(2),jc2(nflav,2),ic2(2)
      common /clatt/nlattc,npmax
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog

      k1=n1
      k2=n2

c     determine n1,n2 and mm
c     ----------------------
    1 continue
      n1=1+rangen()*nlattc
      n1=min(n1,nlattc)
    2 continue
      n2=1+rangen()*nlattc
      n2=min(n2,nlattc)
      if(n2.eq.n1)goto2
      if(n2.lt.n1)then
      n1r=n1
      n1=n2
      n2=n1r
      endif
      if(k.eq.2)then
      if(n1.eq.k1.or.n1.eq.k2.or.n2.eq.k1.or.n2.eq.k2)goto1
      endif
      if(ident(n1).ne.0.and.ident(n2).ne.0)mm=1  ! hadron-hadron
      if(ident(n1).ne.0.and.ident(n2).eq.0)mm=2  ! hadron-empty
      if(ident(n1).eq.0.and.ident(n2).ne.0)mm=2  ! empty-hadron
      if(ident(n1).eq.0.and.ident(n2).eq.0)mm=3  ! empty-empty
      if(ish.ge.7)then
        write(ifch,'(a,i2)')' mm:',mm
        write(ifch,*)'to be replaced:',n1,ident(n1)
        write(ifch,*)'to be replaced:',n2,ident(n2)
      endif

c     flavour of n1+n2 --> jc
c     -----------------------
           if(mm.eq.1)then
      call idtr4(ident(n1),ic1)
      call iddeco(ic1,jc1)
      call idtr4(ident(n2),ic2)
      call iddeco(ic2,jc2)
      do i=1,nflav
      do j=1,2
      jc(i,j)=jc1(i,j)+jc2(i,j)
      enddo
      enddo
           elseif(mm.eq.2.and.ident(n1).ne.0)then
      call idtr4(ident(n1),ic)
      call iddeco(ic,jc)
           elseif(mm.eq.2.and.ident(n2).ne.0)then
      call idtr4(ident(n2),ic)
      call iddeco(ic,jc)
           else
      do i=1,nflav
      do j=1,2
      jc(i,j)=0
      enddo
      enddo
           endif

      if(k.eq.2)then
      n3=n1
      n4=n2
      endif

      return
      end

c----------------------------------------------------------------------
      subroutine hnbpai(id1,id2,jc)
c----------------------------------------------------------------------
c  returns arbitrary hadron pair id1,id2, flavour written to jc
c----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),jc1(nflav,2),ic1(2),jc2(nflav,2),ic2(2)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)

c     construct pair id1,id2
c     ----------------------
      i1=rangen()*(nspecs+iozero)-(iozero-1)
      i1=max(i1,0)
      i1=min(i1,nspecs)
      if(i1.eq.0)then
      id1=0
      do i=1,nflav
      do j=1,2
      jc1(i,j)=0
      enddo
      enddo
      else
      id1=ispecs(i1)
      call idtr4(id1,ic1)
      call iddeco(ic1,jc1)
      endif
      if(ish.ge.7)write(ifch,'(1x,a,i3,a,i5,a,6i2,3x,6i2)')
     *'i1:',i1,'   id1:',id1,'   jc1:',jc1
      i2=rangen()*(nspecs+iozero)-(iozero-1)
      i2=max(i2,0)
      i2=min(i2,nspecs)
      if(i2.eq.0)then
      id2=0
      do i=1,nflav
      do j=1,2
      jc2(i,j)=0
      enddo
      enddo
      else
      id2=ispecs(i2)
      call idtr4(id2,ic2)
      call iddeco(ic2,jc2)
      endif
      if(ish.ge.7)write(ifch,'(1x,a,i3,a,i5,a,6i2,3x,6i2)')
     *'i2:',i2,'   id2:',id2,'   jc2:',jc2
      if(ish.ge.7)write(ifch,'(a,i6,i6)')' pair:',id1,id2

c     determine jc
c     ------------
      do i=1,nflav
      do j=1,2
      jc(i,j)=jc1(i,j)+jc2(i,j)
      enddo
      enddo
      do i=1,nflav
      j12=jc(i,1)-jc(i,2)
      if(j12.ge.0)then
      jc(i,1)=j12
      jc(i,2)=0
      else
      jc(i,1)=0
      jc(i,2)=-j12
      endif
      enddo
      if(ish.ge.7)write(ifch,'(a,6i2,3x,6i2)')' jc:',jc

      return
      end

c----------------------------------------------------------------------
      subroutine hnbpaj(jc,iwpair,id1,id2)
c----------------------------------------------------------------------
c  returns sum of weights iwpair of possible pairs
c  and randomly chosen hadron pair id1,id2 for given flavour jc
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(mspecs=56,mxids=200)
      parameter(mxpair=mspecs**2*4)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cspec2/jspecs(2,nflav,mspecs)
      common/cspec3/lkfok(8,-3:3,-3:3,-3:3,-3:3)  !-charm
      common/cspec5/idpairst(2,mxpair,3**6),iwtpaist(0:mxpair,3**6)
     &              ,idxpair(0:2,0:2,0:2,-1:1,-1:1,-1:1),ipairst(3**6)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      dimension ids(mxids),iwts(mxids),jc(nflav,2)!,jc2(nflav,2)
      dimension idpair(2,mxpair),iwtpai(mxpair)
      dimension jc1mi2(nflav),jcmi(nflav)

c      nflv=nflav
c      if(nflv.gt.6)
c     *call utstop('hnbpaj: nflav.gt.6: modify this routine&')

c     construct possible pairs id1,id2
c     --------------------------------

      ipair=0
      iwpair=0
      idx=0
      if(jc(1,1).gt.2)then
        goto 1
      elseif(jc(1,1).lt.0)then
        goto 1
      elseif(jc(2,1).gt.2)then
        goto 1
      elseif(jc(2,1).lt.0)then
        goto 1
      elseif(jc(3,1).gt.2)then
        goto 1
      elseif(jc(3,1).lt.0)then
        goto 1
      elseif(jc(1,2).gt.1)then
        goto 1
      elseif(jc(1,2).lt.-1)then
        goto 1
      elseif(jc(2,2).gt.1)then
        goto 1
      elseif(jc(2,2).lt.-1)then
        goto 1
      elseif(jc(3,2).gt.1)then
        goto 1
      elseif(jc(3,2).lt.-1)then
        goto 1
      elseif((abs(jc(4,1))+abs(jc(5,1))+abs(jc(6,1))+abs(jc(4,2))
     &       +abs(jc(5,2))+abs(jc(6,2))).gt.0)then
        goto 1
      endif
      idx=idxpair(jc(1,1),jc(2,1),jc(3,1),jc(1,2),jc(2,2),jc(3,2))
      ipair=ipairst(idx)
      if(ipair.eq.0)return
      iwpair=iwtpaist(0,idx)
      do i=1,ipair
        idpair(1,i)=idpairst(1,i,idx)
        idpair(2,i)=idpairst(2,i,idx)
        iwtpai(i)=iwtpaist(i,idx)
      enddo
      goto 4           !pair fixed via table

c  id1=0:
 1    continue
      if(nspecs+1.gt.mxids)call utstop('hnbpaj: mxids too small&')

      jc1mi2(1)=jc(1,1)-jc(1,2)
      jc1mi2(2)=jc(2,1)-jc(2,2)
      jc1mi2(3)=jc(3,1)-jc(3,2)
      jc1mi2(4)=jc(4,1)-jc(4,2)
      jc1mi2(5)=jc(5,1)-jc(5,2)
      jc1mi2(6)=jc(6,1)-jc(6,2)

      nids=0

      if(jc1mi2(1).ne.0)goto11
      if(jc1mi2(2).ne.0)goto11
      if(jc1mi2(3).ne.0)goto11
      if(jc1mi2(4).ne.0)goto11
      if(jc1mi2(5).ne.0)goto11
      if(jc1mi2(6).ne.0)goto11
      nids=nids+1
      ids(nids)=0
      iwts(nids)=iozero
   11 continue

      do j=1,nspecs
      if(jc1mi2(1).ne.ifok(1,j))goto22
      if(jc1mi2(2).ne.ifok(2,j))goto22
      if(jc1mi2(3).ne.ifok(3,j))goto22
      if(jc1mi2(4).ne.ifok(4,j))goto22
      if(jc1mi2(5).ne.ifok(5,j))goto22
      if(jc1mi2(6).ne.ifok(6,j))goto22
      nids=nids+1
      ids(nids)=ispecs(j)
      iwts(nids)=1
   22 continue
      enddo

      if(nids.eq.0)goto2
      if(nids.gt.mxpair)call utstop('hnbpaj: mxpair too small&')
      do k=1,nids
      ipair=ipair+1
      idpair(1,ipair)=0
      idpair(2,ipair)=ids(k)
      iwtpai(ipair)=iozero*iwts(k)
      iwpair=iwpair+iwtpai(ipair)
c      if(ish.ge.6)write(ifch,'(a,i5,5x,a,i6,i6,5x,a,i6)')' pair nr:'
c     *,ipair,'ids:',0,ids(k),'weight:',iwtpai(ipair)
      enddo
    2 continue

c  id1>0:

        do i1=1,nspecs

c        if(ish.ge.6)then
c        do i=1,nflav
c      jc2(i,1)=jc(i,1)-jspecs(1,i,i1)
c      jc2(i,2)=jc(i,2)-jspecs(2,i,i1)
c        enddo
c      write(ifch,'(1x,a,i3,a,i6,a,6i2,3x,6i2)')
c     *'i1:',i1,'   id1:',ispecs(i1),'   jc1:'
c     *,(jspecs(1,i,i1),i=1,6),(jspecs(2,i,i1),i=1,6)
c      write(ifch,'(a,6i2,3x,6i2)')' jc2:',jc2
c        endif

      jcmi(1)=jc1mi2(1)-jspecs(1,1,i1)+jspecs(2,1,i1)
      jcmi(2)=jc1mi2(2)-jspecs(1,2,i1)+jspecs(2,2,i1)
      jcmi(3)=jc1mi2(3)-jspecs(1,3,i1)+jspecs(2,3,i1)
      jcmi(4)=jc1mi2(4)-jspecs(1,4,i1)+jspecs(2,4,i1)
      jcmi(5)=jc1mi2(5)-jspecs(1,5,i1)+jspecs(2,5,i1)
      jcmi(6)=jc1mi2(6)-jspecs(1,6,i1)+jspecs(2,6,i1)
c-charm      if(jcmi(4).ne.0)stop'HNBPAJ: c not treated'
      if(jcmi(5).ne.0)stop'HNBPAJ: b not treated'
      if(jcmi(6).ne.0)stop'HNBPAJ: t not treated'

      nids=0

      if(abs(jcmi(1)).gt.3)goto3
      if(abs(jcmi(2)).gt.3)goto3
      if(abs(jcmi(3)).gt.3)goto3
      if(abs(jcmi(4)).gt.3)goto3   !-charm

      if(jcmi(1).ne.0)goto111
      if(jcmi(2).ne.0)goto111
      if(jcmi(3).ne.0)goto111
      if(jcmi(4).ne.0)goto111  !-charm
      nids=nids+1
      ids(nids)=0
      iwts(nids)=iozero
  111 continue

      lkfok1=lkfok(1,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
      if(lkfok1.gt.0)then
       nids=nids+1
       ids(nids)=lkfok(2,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
       iwts(nids)=1
       if(lkfok1.gt.1)then
        nids=nids+1
        ids(nids)=lkfok(3,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
        iwts(nids)=1
        if(lkfok1.gt.2)then
         if(lkfok1.gt.7)       !-charm
     *   stop'HNBPAJ: dimension of lkfok too small'
         do ii=3,lkfok1
         nids=nids+1
         ids(nids)=lkfok(1+ii,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
         iwts(nids)=1
         enddo
        endif
       endif
      endif

c             do j=1,nspecs
c      if(jcmi(1).ne.ifok(1,j))goto222
c      if(jcmi(2).ne.ifok(2,j))goto222
c      if(jcmi(3).ne.ifok(3,j))goto222
c      if(jcmi(4).ne.ifok(4,j))goto222
c      if(jcmi(5).ne.ifok(5,j))goto222
c      if(jcmi(6).ne.ifok(6,j))goto222
c      nids=nids+1
c      ids(nids)=ispecs(j)
c      iwts(nids)=1
c  222 continue
c             enddo

      if(nids.eq.0)goto3
      if(ipair+nids.gt.mxpair)call utstop('hnbpaj: mxpair too small&')
      do k=1,nids
      ipair=ipair+1
      idpair(1,ipair)=ispecs(i1)
      idpair(2,ipair)=ids(k)
      iwtpai(ipair)=iwts(k)
      iwpair=iwpair+iwtpai(ipair)
      enddo
      if(ish.ge.7)then
      ipair0=ipair-nids
      do k=1,nids
      ipair0=ipair0+1
      write(ifch,'(a,i5,5x,a,i6,i6,5x,a,i6)')' pair nr:'
     *,ipair0,'ids:',ispecs(i1),ids(k),'weight:',iwtpai(ipair0)
      enddo
      endif
    3 continue

        enddo

c     no pair found
c     -------------
      if(ipair.eq.0)then
      if(iwpair.ne.0)call utstop('hnbpaj: iwpair.ne.0&')
      return
      endif


 4    continue
c     select pair
c     -----------
      r=rangen()
      ir=1+r*iwpair
      ir=min(ir,iwpair)
      is=0
      do ip=1,ipair
      is=is+iwtpai(ip)
      if(ir.le.is)then
      id1=idpair(1,ip)
      id2=idpair(2,ip)
c      if(ish.ge.6)write(ifch,*)'random number:',r
c     *,' --> chosen pair:',ip
      goto 1000
      endif
      enddo
      write(ifmt,*)'hnbpaj:',jc,idx,ipair,iwpair,r,ir
      call utstop('hnbpaj: no pair selected&')

1000  continue

      return
      end

c----------------------------------------------------------------------
      subroutine hnbpajini
c----------------------------------------------------------------------
c  initialize array to speed up hnbpaj calculation
c  store sum of weights iwpair of possible pairs in an array
c  for any combinations of quarks
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(mspecs=56,mxids=200)
      parameter(mxpair=mspecs**2*4)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cspec2/jspecs(2,nflav,mspecs)
      common/cspec3/lkfok(8,-3:3,-3:3,-3:3,-3:3)  !-charm
      common/cspec5/idpairst(2,mxpair,3**6),iwtpaist(0:mxpair,3**6)
     &              ,idxpair(0:2,0:2,0:2,-1:1,-1:1,-1:1),ipairst(3**6)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      dimension ids(mxids),iwts(mxids)
      dimension jc1mi2(3),jcmi(4)


c      write(ifmt,*)' Initialize droplet decay ...'

c     construct possible pairs id1,id2
c     --------------------------------
      idx=0
      do iaqs=-1,1
        do iaqd=-1,1
          do iaqu=-1,1
            do iqs=0,2
              do iqd=0,2
                do iqu=0,2

      idx=idx+1
      idxpair(iqu,iqd,iqs,iaqu,iaqd,iaqs)=idx

      ipair=0
      iwtpaist(0,idx)=0
      do i=1,mxids
        ids(i)=0
        iwts(i)=0
      enddo
      do i=1,mxpair
        idpairst(1,i,idx)=0
        idpairst(2,i,idx)=0
        iwtpaist(i,idx)=0
      enddo

c  id1=0:

      if(nspecs+1.gt.mxids)call utstop('hnbpajini: mxids too small&')

      jc1mi2(1)=iqu-iaqu
      jc1mi2(2)=iqd-iaqd
      jc1mi2(3)=iqs-iaqs

      nids=0

      if(jc1mi2(1).ne.0)goto11
      if(jc1mi2(2).ne.0)goto11
      if(jc1mi2(3).ne.0)goto11
      nids=nids+1
      ids(nids)=0
      iwts(nids)=iozero
   11 continue

      do j=1,nspecs
      if(jc1mi2(1).ne.ifok(1,j))goto22
      if(jc1mi2(2).ne.ifok(2,j))goto22
      if(jc1mi2(3).ne.ifok(3,j))goto22
      nids=nids+1
      ids(nids)=ispecs(j)
      iwts(nids)=1
   22 continue
      enddo

      if(nids.eq.0)goto2
      if(nids.gt.mxpair)call utstop('hnbpajini: mxpair too small&')
      do k=1,nids
      ipair=ipair+1
      idpairst(1,ipair,idx)=0
      idpairst(2,ipair,idx)=ids(k)
      iwtpaist(ipair,idx)=iozero*iwts(k)
      iwtpaist(0,idx)=iwtpaist(0,idx)+iwtpaist(ipair,idx)
c      if(ish.ge.6)write(ifch,'(a,i5,5x,a,i6,i6,5x,a,i6)')' pair nr:'
c     *,ipair,'ids:',0,ids(k),'weight:',iwtpai(ipair)
      enddo
    2 continue

c  id1>0:

        do i1=1,nspecs

c        if(ish.ge.6)then
c        do i=1,nflav
c      jc2(i,1)=jc(i,1)-jspecs(1,i,i1)
c      jc2(i,2)=jc(i,2)-jspecs(2,i,i1)
c        enddo
c      write(ifch,'(1x,a,i3,a,i6,a,6i2,3x,6i2)')
c     *'i1:',i1,'   id1:',ispecs(i1),'   jc1:'
c     *,(jspecs(1,i,i1),i=1,6),(jspecs(2,i,i1),i=1,6)
c      write(ifch,'(a,6i2,3x,6i2)')' jc2:',jc2
c        endif

      jcmi(1)=jc1mi2(1)-jspecs(1,1,i1)+jspecs(2,1,i1)
      jcmi(2)=jc1mi2(2)-jspecs(1,2,i1)+jspecs(2,2,i1)
      jcmi(3)=jc1mi2(3)-jspecs(1,3,i1)+jspecs(2,3,i1)
      jcmi(4)=0

      nids=0

      if(abs(jcmi(1)).gt.3)goto3
      if(abs(jcmi(2)).gt.3)goto3
      if(abs(jcmi(3)).gt.3)goto3

      if(jcmi(1).ne.0)goto111
      if(jcmi(2).ne.0)goto111
      if(jcmi(3).ne.0)goto111
      nids=nids+1
      ids(nids)=0
      iwts(nids)=iozero
  111 continue

      lkfok1=lkfok(1,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
      if(lkfok1.gt.0)then
       nids=nids+1
       ids(nids)=lkfok(2,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
       iwts(nids)=1
       if(lkfok1.gt.1)then
        nids=nids+1
        ids(nids)=lkfok(3,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
        iwts(nids)=1
        if(lkfok1.gt.2)then
         if(lkfok1.gt.7)       !-charm
     *   stop'HNBPAJINI: dimension of lkfok too small'
         do ii=3,lkfok1
         nids=nids+1
         ids(nids)=lkfok(1+ii,jcmi(1),jcmi(2),jcmi(3),jcmi(4))
         iwts(nids)=1
         enddo
        endif
       endif
      endif

c             do j=1,nspecs
c      if(jcmi(1).ne.ifok(1,j))goto222
c      if(jcmi(2).ne.ifok(2,j))goto222
c      if(jcmi(3).ne.ifok(3,j))goto222
c      if(jcmi(4).ne.ifok(4,j))goto222
c      if(jcmi(5).ne.ifok(5,j))goto222
c      if(jcmi(6).ne.ifok(6,j))goto222
c      nids=nids+1
c      ids(nids)=ispecs(j)
c      iwts(nids)=1
c  222 continue
c             enddo

      if(nids.eq.0)goto3
      if(ipair+nids.gt.mxpair)
     &     call utstop('hnbpajini: mxpair too small&')
      do k=1,nids
      ipair=ipair+1
      idpairst(1,ipair,idx)=ispecs(i1)
      idpairst(2,ipair,idx)=ids(k)
      iwtpaist(ipair,idx)=iwts(k)
      iwtpaist(0,idx)=iwtpaist(0,idx)+iwtpaist(ipair,idx)
      enddo
      ipairst(idx)=ipair
    3 continue

        enddo

c     no pair found
c     -------------
      if(ipair.eq.0)then
      if(iwtpaist(0,idx).ne.0)call utstop('hnbpajini: iwpair.ne.0&')
      endif


                enddo
              enddo
            enddo
          enddo
        enddo
      enddo

      return
      end

c--------------------------------------------------------------------
      subroutine hnbraw(npx,npy,w)
c--------------------------------------------------------------------
c returns random walk fctn w=w(0,p_1,p_2,...,p_n) for noncovariant
c phase space integral (see hagedorn, suppl nuov cim ix(x) (1958)646)
c input: dimension np and momenta p_i=pcm(5,i) via /confg/
c    1   < np <= npx : hagedorn method
c    npx < np <= npy : integral method
c    npy < np        : asymptotic method
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      integer ii(maxp),isi(maxp)
      double precision ppcm(maxp),ww,ppsum,ppmax
      external hnbrax
      common/cepsr/nepsr
      if(ish.ge.9)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbraw ',('-',i=1,30)

      if(np.lt.3)call utstop('hnbraw: np must be at least 3&')

      kper=5
      pi=3.1415927
      pmax=0
      do i=1,np
      pmax=pmax+pcm(5,i)
      enddo
      wio=0
      win=0
      whd=0

c     sum p_i - 2*p_max not positive
c     ------------------------------
      px=0
      ps=0
      do i=1,np
      px=max(px,pcm(5,i))
      ps=ps+pcm(5,i)
      enddo
      if(ps-2*px.le.0.)then
      w=0
      if(ish.ge.7)write(ifch,'(1x,a,e12.5,4x)')
     *'sum p_i - 2*p_max not positive -->  w:',w
      goto1000
      endif

c     asymptotic method
c     -----------------
      was=0
      do i=1,np
      was=was+pcm(5,i)**2
      enddo
      was=(was*2*pi/3)**(-1.5)
      if(ish.ge.7)write(ifch,'(1x,a,e12.5,4x)')
     *'asymptotic method: was:',was

      if(np.gt.npy)then
      w=was
      goto1000
      endif

      if(np.le.npx)goto9

c     integral method
c     ---------------
      if(ish.ge.9)write(ifch,*)'integral method...'
      itmax=8
      it=0
      b=pi*np*kper/pmax
      win=0
      nepsr=0
    3 continue
      it=it+1
      if(ish.ge.9)write(ifch,*)'it:',it
      b=b*5./3.
      wio=win
      call uttrap(hnbrax,0.,b,win)
      iok=0
      if(abs(win-wio).le.epsr*abs((win+wio)/2))iok=1
      if(it.eq.itmax)iok=1
      if(ish.ge.8.or.ish.ge.7.and.iok.eq.1)
     *write(ifch,'(1x,2(a,e12.5,2x),a,i2,2x,a,i4)')
     *'integral method:   win:',win
     *,'upper limit:',b,'it:',it,'nepsr:',nepsr
           if(it.eq.itmax
     *.and.abs(win-wio).gt.epsr*abs((win+wio)/2))then
      nepsr=nepsr+1
      if(ish.ge.9)then
      call utmsg('hnbraw')
      write(ifch,*)
     *'*****  requested accuracy could not be achieved'
      write(ifch,*)'achieved accuracy: '
     *,abs(win-wio)/abs((win+wio)/2)
      write(ifch,*)'requested accuracy:',epsr
      call utmsgf
      endif
           endif
      if(it.eq.1.or.iok.eq.0)goto3

      if(nepsr.eq.0)then
      w=win
      goto1000
      endif

      if(np.gt.20)then
        if(ish.ge.1)then
          call utmsg('hnbraw')
          write(ifch,*)
     *         '*****  requested accuracy could not be achieved'
          write(ifch,*)'achieved accuracy: '
     *         ,abs(win-wio)/abs((win+wio)/2)
          write(ifch,*)'requested accuracy:',epsr
          call utmsgf
        endif
        w=win
        goto1000
      endif

c     hagedorn method (double)
c     ------------------------
    9 continue
      ppmax=0
      do i=1,np
      ppcm(i)=pcm(5,i)
      ppmax=ppmax+ppcm(i)
      enddo
      ww=0
      do i=1,np
      ii(i)=0
      isi(i)=1
      enddo
      ppsum=ppmax
      i=0
      iprosi=1
      ww=iprosi*(ppsum/ppmax)**(np-3)
      if(ish.ge.8)
     *write(ifch,'(4x,i5,12x,f7.2,i5,f11.2)')np,sngl(ppsum)
     *,iprosi,sngl(ww)
    5 continue
      i=i+1
      if(i.gt.np)goto6
      if(ii(i).eq.1)goto5
      iprosi=-iprosi
      isi(i)=-isi(i)
      ppsum=ppsum+2*isi(i)*ppcm(i)
           if(ppsum.gt.0.or.ppsum.eq.0..and.isi(i).gt.0)then
      ww=ww+iprosi*(ppsum/ppmax)**(np-3)
      if(ish.ge.8)
     *write(ifch,'(4x,2i5,2f7.2,i5,f11.2)')
     *np,i,sngl(2*isi(i)*ppcm(i)),sngl(ppsum),iprosi,sngl(ww)
           else
      if(ish.ge.8)
     *write(ifch,'(4x,2i5,2f7.2,i5,4x,a)')
     *np,i,sngl(2*isi(i)*ppcm(i)),sngl(ppsum),iprosi,'not counted'
           endif
      ii(i)=1
      if(i.gt.1)then
      do j=1,i-1
      ii(j)=0
      enddo
      endif
      i=0
      goto5
    6 continue
      do i=1,np
      ww=ww*pmax/ppcm(i)/2./i
      enddo
      ww=-ww/pmax**3/pi/2.*np*(np-1)*(np-2)
      whd=ww
      if(ish.ge.7)write(ifch,'(1x,a,e12.5,4x,a)')
     *'hagedorn method:   whd:',whd,'double precision'

      w=whd

1000  continue
      if(ish.ge.9)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbraw ',('-',i=1,10)
      return
      end

c--------------------------------------------------------------------
      function hnbrax(x)
c--------------------------------------------------------------------
c returns integrand for random walk fctn w=w(0,p_1,p_2,...,p_n):
c 1./(2*pi**2) * x**2 * prod[sin(p_i*x)/(p_i*x)]
c input: dimension np and momenta p_i=pcm(5,i) via /confg/
c--------------------------------------------------------------------
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/cnsta/pi,pii,hquer,prom,piom,ainfin
      hnbrax= pii * x**2
      do i=1,np
      px=pcm(5,i)*x
      if(px.ne.0.)hnbrax=hnbrax*sin(px)/px
      enddo
      return
      end

c----------------------------------------------------------------------
      subroutine hnbrmz
c----------------------------------------------------------------------
c  removes intermediate zeros from ident
c  updates np
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
c      integer identx(maxp)
      common /clatt/nlattc,npmax
      if(ish.ge.9)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbrmz ',('-',i=1,30)
      if(np.eq.0)goto1000

c      do i=1,np
c      identx(i)=ident(i)
c      enddo
c      npx=np

      i=0
      np=nlattc+1

    1 i=i+1
      if(i.gt.nlattc)then
      np=nlattc
      goto1000
      endif
      if(ident(i).ne.0)goto1
    2 np=np-1
      if(np.eq.0)goto1000
      if(ident(np).eq.0)goto2

      if(ish.ge.9)then
      write(ifch,*)'ident:'
      write(ifch,'(1x,10i7)')(ident(j),j=1,nlattc)
      write(ifch,'(1x,a,i3,3x,a,i3)')'i:',i,'np:',np
      endif

      if(i.eq.np+1)goto1000

      ident(i)=ident(np)
      ident(np)=0
      goto1

1000  continue
      if(ish.ge.9)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbrmz ',('-',i=1,10)
      end

c----------------------------------------------------------------------
      subroutine hnbrod
c----------------------------------------------------------------------
c deformes polygon of a sequence of arbitrarily rotated momentum
c vectors such that the polygon gets closed
c    input: pcm(1-3,i) representing polygon
c    output: pcm(1-3,i) representing closed polygon
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      real x(3),y(3),z(3),w(3)
      if(ish.ge.8)write(ifch,*)'sr hnbrod: polygon deformation:'

      err=0.01

      kmax=1000
      fac=0.30
      x2max=(err*tecm)**2

      if(ish.ge.8)write(ifch,'(a,i4,a,f12.6)')
     *' kmax:',kmax,'   x2max:',x2max

      x(1)=0
      x(2)=0
      x(3)=0
      do i=1,np
      x(1)=x(1)+pcm(1,i)
      x(2)=x(2)+pcm(2,i)
      x(3)=x(3)+pcm(3,i)
      enddo ! i

      k=0
   1  continue

      x2=x(1)**2+x(2)**2+x(3)**2
      if(ish.ge.8)write(ifch,'(a,i3,a,3f9.3,a,f12.6)')
     *' it',k,':   x:',x,'      x2:',x2
      if(x2.le.x2max)goto1000
      if(k.gt.kmax)goto1001

      k=k+1
      ir=1+rangen()*np
      ir=min(ir,np)

      z(1)=-x(1)
      z(2)=-x(2)
      z(3)=-x(3)
      x(1)=x(1)-pcm(1,ir)
      x(2)=x(2)-pcm(2,ir)
      x(3)=x(3)-pcm(3,ir)
      y(1)=pcm(1,ir)
      y(2)=pcm(2,ir)
      y(3)=pcm(3,ir)
      if(ish.ge.9)write(ifch,'(a,i3,a,3f9.3,a,3f9.3,a,i4)')
     *' it',k,':   x:',x,'   y:',y,'  ir:',ir
      xxx=x(1)**2+x(2)**2+x(3)**2
      yyy=y(1)**2+y(2)**2+y(3)**2
      zzz=z(1)**2+z(2)**2+z(3)**2
         if(xxx.gt.0..and.yyy.gt.0..and.zzz.gt.0.)then
c      xx=sqrt(xxx)
      yy=sqrt(yyy)
      zz=sqrt(zzz)
      a=min(fac,fac*yy/zz)
      w(1)=y(1)+a*z(1)
      w(2)=y(2)+a*z(2)
      w(3)=y(3)+a*z(3)
      www=w(1)**2+w(2)**2+w(3)**2
         if(www.gt.0.)then
      ww=sqrt(www)
      y(1)=yy/ww*w(1)
      y(2)=yy/ww*w(2)
      y(3)=yy/ww*w(3)
      pcm(1,ir)=y(1)
      pcm(2,ir)=y(2)
      pcm(3,ir)=y(3)
         endif
         endif
      x(1)=x(1)+y(1)
      x(2)=x(2)+y(2)
      x(3)=x(3)+y(3)
      if(ish.ge.9)write(ifch,'(a,i3,a,3f9.3,a,3f9.3,a,i4)')
     *' it',k,':   x:',x,'   y:',y,'  ir:',ir

      goto1

 1001 continue
      call utmsg('hnbrod')
      write(ifch,*)'*****  total 3-momentum nonzero'
      write(ifch,'(3f12.5,5x,2f12.5)')(x(j),j=1,3),x2,x2max
      call utmsgf

 1000 continue
      return

      end

c----------------------------------------------------------------------
      subroutine hnbrop(ishx,ichk)
c----------------------------------------------------------------------
c  prints momenta of configuration (essentially to check rotation procedure)
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      double precision ps(5)
      err=0.01
      do j=1,4
      ps(j)=0
      enddo
      do i=1,np
      do j=1,4
      ps(j)=ps(j)+pcm(j,i)
      enddo
      if(ish.ge.ishx)write(ifch,'(1x,i3,5x,5f12.5)')i,(pcm(j,i),j=1,3)
     *,sqrt(pcm(1,i)**2+pcm(2,i)**2+pcm(3,i)**2),pcm(5,i)
      enddo
      ps(5)=dsqrt(ps(1)**2+ps(2)**2+ps(3)**2)
      if(ish.ge.ishx)write(ifch,'(1x,a4,8x,5f12.5)')
     *'sum:',(sngl(ps(j)),j=1,5)
           if(ichk.eq.1)then
           if(dabs(ps(1)).gt.err*tecm.or.dabs(ps(2)).gt.err*tecm
     *.or.dabs(ps(3)).gt.err*tecm)then
      call utmsg('hnbrop')
      write(ifch,*)'*****  total 3-momentum nonzero'
      write(ifch,'(9x,5f12.5)')(sngl(ps(j)),j=1,5)
      call utmsgf
           endif
           endif
      return
      end

c----------------------------------------------------------------------
      subroutine hnbrot
c----------------------------------------------------------------------
c rotates momenta of /confg/ randomly
c   input: pcm(5,i)
c   output: pcm(1-3,i)
c----------------------------------------------------------------------
      common/cnsta/pi,pii,hquer,prom,piom,ainfin
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      real u(3)

      do i=1,np
      u(3)=2.*rangen()-1.
      phi=2.*pi*rangen()
      u(1)=sqrt(1.-u(3)**2)*cos(phi)
      u(2)=sqrt(1.-u(3)**2)*sin(phi)
      pcm(1,i)=pcm(5,i)*u(1)
      pcm(2,i)=pcm(5,i)*u(2)
      pcm(3,i)=pcm(5,i)*u(3)
      enddo

      return
      end

cc-------------------------------------------------------------------
c      subroutine hnbrt2old(c,s,c2,s2,pr,i)
cc-------------------------------------------------------------------
cc  formerly subr rotes2 from cernlib
cc  this subr now does two rotations (xy and xz)
cc-------------------------------------------------------------------
c      parameter(maxp=500)
c      dimension pr(5*maxp)
c      k1 = 5*i - 4
c      k2 = k1 + 1
c      sa = pr(k1)
c      sb = pr(k2)
c      a      = sa*c - sb*s
c      pr(k2) = sa*s + sb*c
c      k2 = k2 + 1
c      b = pr(k2)
c      pr(k1) = a*c2 - b*s2
c      pr(k2) = a*s2 + b*c2
c      return
c      end
c
c-------------------------------------------------------------------
      subroutine hnbrt2(c,s,c2,s2,pr,i)
c-------------------------------------------------------------------
c  formerly subr rotes2 from cernlib
c  this subr now does two rotations (xy and xz)
c-------------------------------------------------------------------
      parameter(maxp=500)
      dimension pr(5,maxp)
      k1 = 5*i - 4
      k2 = k1 + 1
      sa = pr(1,i)
      sb = pr(2,i)
      a      = sa*c - sb*s
      pr(2,i) = sa*s + sb*c
      k2 = k2 + 1
      b = pr(3,i)
      pr(1,i) = a*c2 - b*s2
      pr(3,i) = a*s2 + b*c2
      return
      end

cc-----------------------------------------------------------------------
c      subroutine hnbsor(a,n)
cc-----------------------------------------------------------------------
cc cern proglib# m103    flpsor          .version kernfor  3.15  820113
cc orig. 29/04/78
cc-----------------------------------------------------------------------
cc   sort the one-dimensional floating point array a(1),...,a(n) by
cc   increasing values
cc-----------------------------------------------------------------------
c      dimension a(*)
c      common /slate/ lt(20),rt(20)
c      integer r,rt
cc
c      level=1
c      lt(1)=1
c      rt(1)=n
c   10 l=lt(level)
c      r=rt(level)
c      level=level-1
c   20 if(r.gt.l) go to 200
c      if(level) 50,50,10
cc
cc   subdivide the interval l,r
cc     l : lower limit of the interval (input)
cc     r : upper limit of the interval (input)
cc     j : upper limit of lower sub-interval (output)
cc     i : lower limit of upper sub-interval (output)
cc
c  200 i=l
c      j=r
c      m=(l+r)/2
c      x=a(m)
c  220 if(a(i).ge.x) go to 230
c      i=i+1
c      go to 220
c  230 if(a(j).le.x) go to 231
c      j=j-1
c      go to 230
cc
c  231 if(i.gt.j) go to 232
c      w=a(i)
c      a(i)=a(j)
c      a(j)=w
c      i=i+1
c      j=j-1
c      if(i.le.j) go to 220
cc
c  232 level=level+1
c      if(level.gt.20)stop'level too large'
c      if((r-i).ge.(j-l)) go to 30
c      lt(level)=l
c      rt(level)=j
c      l=i
c      go to 20
c   30 lt(level)=i
c      rt(level)=r
c      r=j
c      go to 20
c   50 return
c      end
c
c-----------------------------------------------------------------------
      subroutine hnbspd(iopt)
c-----------------------------------------------------------------------
c  defines particle species and masses and degeneracies.
c  input:
c    iopt=odd number: massless
c    iopt=even number: same as iopt-1, but massive
c    iopt= 1: pi0 (massless)
c    iopt= 2: pi0
c    iopt= 3: pi-,pi0,pi+ (massless)
c    iopt= 4: pi-,pi0,pi+
c    iopt= 5: pi-,pi0,pi+,prt,aprt,ntr,antr (massless)
c    iopt= 6: pi-,pi0,pi+,prt,aprt,ntr,antr
c    iopt= 7: 25 hadrons (massless)
c    iopt= 8: 25 hadrons
c    iopt= 9: 54 hadrons (massless)
c    iopt=10: 54 hadrons
c    iopt=11:  3 quarks  (massless)
c    iopt=12:  3 quarks
c    iopt=13:  54 hadrons + J/psi   (massless)
c    iopt=14:  54 hadrons + J/psi
c    iopt=15:  54 hadrons + J/psi + H  (massless)
c    iopt=16:  54 hadrons + J/psi + H
c  output:
c    nspecs: nr of species
c    ispecs: id's
c    aspecs: masses
c-----------------------------------------------------------------------
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nflav=6)
      integer jc(nflav,2),ic(2)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      common/cspec2/jspecs(2,nflav,mspecs)
      common/cspec3/lkfok(8,-3:3,-3:3,-3:3,-3:3)  !-charm
      common/cspec4/lkfoi(8,-3:3,-3:3,-3:3,-3:3)  !-charm
      parameter(nspe01=1,nspe03=3,nspe05=7,nspe07=25,nspe09=54)
      parameter(nspe11=6,nspe13=55,nspe15=56)
      real jspe01(nspe01),jspe03(nspe03),jspe05(nspe05),jspe07(nspe07)
     *,jspe09(nspe09),jspe11(nspe11),jspe13(nspe13),jspe15(nspe15)
      data jspe01/   110 /
      data jspe03/   110,  120, -120 /
      data jspe05/   110,  120, -120, 1120,-1120, 1220,-1220 /
      data jspe07/
     *   110,  120, -120,  130, -130,  230, -230,  220,  330
     *, 1120,-1120, 1220,-1220, 1130,-1130, 2130,-2130
     *, 1230,-1230, 2230,-2230, 1330,-1330, 2330,-2330 /
      data jspe09/
     *   110,  120, -120,  130, -130,  230, -230,  220,  330
     *,  111,  121, -121,  131, -131,  231, -231,  221,  331
     *, 1120,-1120, 1220,-1220, 1130,-1130, 2130,-2130
     *, 1230,-1230, 2230,-2230, 1330,-1330, 2330,-2330
     *, 1111,-1111, 1121,-1121, 1221,-1221, 2221,-2221, 1131,-1131
     *, 1231,-1231, 2231,-2231, 1331,-1331, 2331,-2331, 3331,-3331 /
      data jspe11/
     *     1,   -1,    2,   -2,    3,   -3   /
      data jspe13/
     *   110,  120, -120,  130, -130,  230, -230,  220,  330
     *,  111,  121, -121,  131, -131,  231, -231,  221,  331
     *, 1120,-1120, 1220,-1220, 1130,-1130, 2130,-2130
     *, 1230,-1230, 2230,-2230, 1330,-1330, 2330,-2330
     *, 1111,-1111, 1121,-1121, 1221,-1221, 2221,-2221, 1131,-1131
     *, 1231,-1231, 2231,-2231, 1331,-1331, 2331,-2331, 3331,-3331
     *, 441 /
      data jspe15/
     *   110,  120, -120,  130, -130,  230, -230,  220,  330
     *,  111,  121, -121,  131, -131,  231, -231,  221,  331
     *, 1120,-1120, 1220,-1220, 1130,-1130, 2130,-2130
     *, 1230,-1230, 2230,-2230, 1330,-1330, 2330,-2330
     *, 1111,-1111, 1121,-1121, 1221,-1221, 2221,-2221, 1131,-1131
     *, 1231,-1231, 2231,-2231, 1331,-1331, 2331,-2331, 3331,-3331
     *, 441 , 30 /

      if(iopt.gt.16)call utstop('hnbspd: invalid option&')
      ioptx=(1+iopt)/2*2-1

      if(ioptx.eq.1)nspecs=nspe01
      if(ioptx.eq.3)nspecs=nspe03
      if(ioptx.eq.5)nspecs=nspe05
      if(ioptx.eq.7)nspecs=nspe07
      if(ioptx.eq.9)nspecs=nspe09
      if(ioptx.eq.11)nspecs=nspe11
      if(ioptx.eq.13)nspecs=nspe13
      if(ioptx.eq.15)nspecs=nspe15
           do i=1,nspecs
      if(ioptx.eq.1)ispecs(i)=jspe01(i)
      if(ioptx.eq.3)ispecs(i)=jspe03(i)
      if(ioptx.eq.5)ispecs(i)=jspe05(i)
      if(ioptx.eq.7)ispecs(i)=jspe07(i)
      if(ioptx.eq.9)ispecs(i)=jspe09(i)
      if(ioptx.eq.11)ispecs(i)=jspe11(i)
      if(ioptx.eq.13)ispecs(i)=jspe13(i)
      if(ioptx.eq.15)ispecs(i)=jspe15(i)
      if(ioptx.eq.iopt)then
      aspecs(i)=0
      else
        id=ispecs(i)
        call idmass(id,am)
        aspecs(i)=am
      endif
      call hnbspi(ispecs(i),gg)
      gspecs(i)=gg
           enddo

      do nf=1,nflav
      ifoa(nf)=0
      enddo
      do iic=-3, 3                !-charm
      do iis=-3, 3
      do iid=-3, 3
      do iiu=-3, 3
      do ii=1,7
      lkfok(ii,iiu,iid,iis,iic)=0   !-charm
      lkfoi(ii,iiu,iid,iis,iic)=0   !-charm
      enddo
      enddo
      enddo
      enddo
      enddo
           do i=1,nspecs
      id=ispecs(i)
      call idtr4(id,ic)
      call iddeco(ic,jc)
      do nf=1,nflav
      ifok(nf,i)=jc(nf,1)-jc(nf,2)
      ifoa(nf)=ifoa(nf)+iabs(ifok(nf,i))
      jspecs(1,nf,i)=jc(nf,1)
      jspecs(2,nf,i)=jc(nf,2)
      enddo
      iiu=ifok(1,i)
      iid=ifok(2,i)
      iis=ifok(3,i)
      iic=ifok(4,i)  !-charm
      if(abs(iiu).gt.3)stop'HNBSPD: u-dimension of lkfok too small'
      if(abs(iid).gt.3)stop'HNBSPD: d-dimension of lkfok too small'
      if(abs(iis).gt.3)stop'HNBSPD: s-dimension of lkfok too small'
      if(abs(iic).gt.3)stop'HNBSPD: c-dimension of lkfok too small'   !-charm
c-charm      if(ifok(4,i).ne.0)stop'HNBSPD: lkfok needs index for c'
      if(ifok(5,i).ne.0)stop'HNBSPD: lkfok needs index for b'
      if(ifok(6,i).ne.0)stop'HNBSPD: lkfok needs index for t'
      lkfok(1,iiu,iid,iis,iic)=lkfok(1,iiu,iid,iis,iic)+1             !-charm
      lkfoi(1,iiu,iid,iis,iic)=lkfoi(1,iiu,iid,iis,iic)+1             !-charm
      ii=lkfok(1,iiu,iid,iis,iic)                                      !-charm
      if(ii.gt.7)stop'HNBSPD: ii-dimension of lkfok too small'
      lkfok(1+ii,iiu,iid,iis,iic)=id                                 !-charm
      lkfoi(1+ii,iiu,iid,iis,iic)=i                                 !-charm
c       write(6,'(i5,5x,3i5,5x,i5,5x,6i5)')
c     * id,iiu,iid,iis,(lkfok(iiu,iid,iis,kk),kk=1,7)
           enddo

      return
      end

c-------------------------------------------------------------
      subroutine hnbspf(ku,kd,ks,kc,kb,kt,j,n,spelog)
c-------------------------------------------------------------
c  returns spelog = log of factor for consid. different species
c  spelog is double precision
c  option ioflac determines the method:
c     ioflac=1: ignore flavour conservation
c     ioflac=2: flavour conservation implemented straightforward
c                 (only for nspecs=3,7)
c     ioflac=3: flavour conservation via generating fctn
c  further input:
c     ku,...,kt (integer) : flavour
c     j (integer) : excluded species
c     n (integer) : multiplicity
c-------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      integer m(7),l(7),ifot(nflav)
      common/csph/ifox(nflav),ifoy(nflav),jx,nx,ifom(nflav,mspecs)
      parameter(mxfacu=200)
      double precision faci(0:mxfacu)
      double precision utgam2,spelog,spe
c      parameter(numax=100,kqmax=100)
c      parameter(mxhh=200)
      if(ish.ge.9)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbspf ',('-',i=1,30)
      if(ish.ge.9)write(ifch,'(1x,a,9x,a,4x,a)')
     *' ku kd ks kc kb kt','j','n'
      if(ish.ge.9)write(ifch,'(1x,6i3,5x,2i5)')
     *ku,kd,ks,kc,kb,kt,j,n
      k=nspecs
      jx=j
      nx=n
      ifot(1)=ku
      ifot(2)=kd
      ifot(3)=ks
      ifot(4)=kc
      ifot(5)=kb
      ifot(6)=kt

           if(ioflac.eq.1)then

      if(ish.ge.9)write(ifch,'(1x,a,i1)')'ioflac=',ioflac
      g=0
      do i=1,nspecs
      if(i.ne.j)g=g+gspecs(i)
      enddo
      spelog=n*dlog(1.d0*g)

           elseif(ioflac.eq.2)then

      if(ish.ge.9)write(ifch,'(1x,a,i2)')'ioflac:',ioflac
           if(k.eq.3)then
      if(ish.ge.9)write(ifch,'(1x,a,i2)')'nspecs:',nspecs
      spe=0d0
           if(j.lt.1.or. j.gt.k)then
      do 1 n1=0,n
      do 2 n2=0,n-n1
      n3=n-n1-n2
      do 5 nf=1,nflav
      if(ifoa(nf).eq.0.and.ifot(nf).eq.0)goto5
      if(n1*ifok(nf,1)+n2*ifok(nf,2)+n3*ifok(nf,3).ne.ifot(nf))goto2
    5 continue
      spe=spe+utgam2(1.d0+n)
     &/utgam2(1.d0+n1)/utgam2(1.d0+n2)/utgam2(1.d0+n3)
     &*gspecs(1)**n1*gspecs(2)**n2*gspecs(3)**n3
    2 continue
    1 continue
           else
      do 3 i1=0,n
      i2=n-i1
      m(1)=0
      m(2)=i1
      m(3)=i2
      do i=1,3
      ii=1+mod(j-2+i,3)
      l(ii)=m(i)
      enddo
      n1=l(1)
      n2=l(2)
      n3=l(3)
      do 6 nf=1,nflav
      if(ifoa(nf).eq.0.and.ifot(nf).eq.0)goto6
      if(n1*ifok(nf,1)+n2*ifok(nf,2)+n3*ifok(nf,3).ne.ifot(nf))goto3
    6 continue
      spe=spe+utgam2(1.d0+n)
     &/utgam2(1.d0+n1)/utgam2(1.d0+n2)/utgam2(1.d0+n3)
     &*gspecs(1)**n1*gspecs(2)**n2*gspecs(3)**n3
    3 continue
           endif
      if(ish.ge.9)write(ifch,*)'spe:',spe
      spelog=-1000
      if(spe.gt.0.d0)spelog=dlog(spe)
      if(ish.ge.9)write(ifch,*)'spelog:',spelog
           elseif(k.eq.7)then
      if(ish.ge.9)write(ifch,'(1x,a,i2)')'nspecs:',nspecs
      if(n.gt.mxfacu)call utstop('hnbspf: mxfacu too small&')
      do lf=0,n
      faci(lf)=1.d0/utgam2(1d0+lf)
      enddo
      spe=0
           if(j.lt.1.or. j.gt.k)then
      do n1=0,n
      do n2=0,n-n1
      do n3=0,n-n1-n2
      do n4=0,n-n1-n2-n3
      do n5=0,n-n1-n2-n3-n4
      do 12 n6=0,n-n1-n2-n3-n4-n5
      n7=n-n1-n2-n3-n4-n5-n6
      do 15 nf=1,nflav
      if(ifoa(nf).eq.0.and.ifot(nf).eq.0)goto15
      if(n1*ifok(nf,1)+n2*ifok(nf,2)+n3*ifok(nf,3)+n4*ifok(nf,4)
     *+n5*ifok(nf,5)+n6*ifok(nf,6)+n7*ifok(nf,7).ne.ifot(nf))goto12
   15 continue
      spe=spe+1d0/faci(n)*faci(n1)*faci(n2)*faci(n3)*faci(n4)
     &*faci(n5)*faci(n6)*faci(n7)
     &*gspecs(1)**n1*gspecs(2)**n2*gspecs(3)**n3*gspecs(4)**n4
     &*gspecs(5)**n5*gspecs(6)**n6*gspecs(7)**n7
   12 continue
      enddo
      enddo
      enddo
      enddo
      enddo
           else
      do i1=0,n
      do i2=0,n-i1
      do i3=0,n-i1-i2
      do i4=0,n-i1-i2-i3
      do 13 i5=0,n-i1-i2-i3-i4
      i6=n-i1-i2-i3-i4-i5
      m(1)=0
      m(2)=i1
      m(3)=i2
      m(4)=i3
      m(5)=i4
      m(6)=i5
      m(7)=i6
      do i=1,7
      ii=1+mod(j-2+i,7)
      l(ii)=m(i)
      enddo
      n1=l(1)
      n2=l(2)
      n3=l(3)
      n4=l(4)
      n5=l(5)
      n6=l(6)
      n7=l(7)
      do 16 nf=1,nflav
      if(ifoa(nf).eq.0.and.ifot(nf).eq.0)goto16
      if(n1*ifok(nf,1)+n2*ifok(nf,2)+n3*ifok(nf,3)+n4*ifok(nf,4)
     *+n5*ifok(nf,5)+n6*ifok(nf,6)+n7*ifok(nf,7).ne.ifot(nf))goto13
   16 continue
      spe=spe+1d0/faci(n)*faci(n1)*faci(n2)*faci(n3)*faci(n4)
     &*faci(n5)*faci(n6)*faci(n7)
     &*gspecs(1)**n1*gspecs(2)**n2*gspecs(3)**n3*gspecs(4)**n4
     &*gspecs(5)**n5*gspecs(6)**n6*gspecs(7)**n7
   13 continue
      enddo
      enddo
      enddo
      enddo
           endif
      if(ish.ge.9)write(ifch,*)'spe:',spe
      spelog=-1000
      if(spe.gt.0.d0)spelog=dlog(spe)
      if(ish.ge.9)write(ifch,*)'spelog:',spelog
           else
      call utstop('hnbspf: ioflac=2 only for nspecs=3,7&')
           endif

           elseif(ioflac.eq.3)then

      call utstop('hnbspf: ioflac must be 1 or 2&')

           endif

      if(ish.ge.9)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbspf ',('-',i=1,10)
      return
      end

c-------------------------------------------------------------
      subroutine hnbspg(ku,kd,ks,kc,kb,kt,j,n,spelog)
c-------------------------------------------------------------
      include 'epos.inc'
      double precision spelog,spalog
      if(ioflac.ne.0)return
      ioflac=2
      call hnbspf(ku,kd,ks,kc,kb,kt,j,n,spalog)
      ioflac=3
      call hnbspf(ku,kd,ks,kc,kb,kt,j,n,spelog)
      ioflac=0
      write(ifch,*)'ioflac=2/3:',spalog,spelog
      return
      end

c----------------------------------------------------------------------
      subroutine hnbspi(id,spideg)
c----------------------------------------------------------------------
c  returns spin degeneracy spideg for particle id-code id
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nspec=62)
      dimension ispec(nspec),spid(nspec)
      data ispec/
     *     1,   -1,    2,   -2,    3,   -3
     *,  110,  120, -120,  220,  130, -130,  230, -230,  330
     *,  111,  121, -121,  221,  131, -131,  231, -231,  331
     *, 1120, 1220, 1130, 2130, 1230, 2230, 1330, 2330
     *, 1111, 1121, 1221, 2221, 1131, 1231, 2231, 1331, 2331, 3331
     *,-1120,-1220,-1130,-2130,-1230,-2230,-1330,-2330
     *,-1111,-1121,-1221,-2221,-1131,-1231,-2231,-1331,-2331,-3331
     *,441,30/
      data spid/
     *  6*6.
     *, 9*1.
     *, 9*3.
     *, 8*2.
     *,10*4.
     *, 8*2.
     *,10*4.
     *,1*3
     *,1*3/
      do i=1,nspec
      if(id.eq.ispec(i))then
      spideg=spid(i)
      fac=1
      !factb ... not used
      !factq ... not used
      call idflav(id,ifl1,ifl2,ifl3,jspin,index)
      ifls=0
      if(abs(ifl1).eq.3)ifls=ifls+1
      if(abs(ifl2).eq.3)ifls=ifls+1
      if(abs(ifl3).eq.3)ifls=ifls+1
      if(iLHC.eq.1)then
      if(ifls.ge.1)then
        if(abs(id).gt.1000)then
          fac=fac*(1+facts)
        elseif(abs(id).lt.1000)then
          fac=fac*(1-facts)
        endif
      elseif(abs(id).gt.1000)then
        fac=fac*(1+factb)
      endif
      else
      if(ifls.ge.1)then
        if(abs(id).gt.1000)then
          fac=fac*(1+facts)
        elseif(abs(id).lt.1000)then
          fac=fac*(1-facts)
        endif
      endif
      endif
      spideg=spideg*fac
      goto1
      endif
      enddo
      call utstop('hnbspi: id not found&')
    1 continue
      return
      end

c----------------------------------------------------------------------
      subroutine hnbtst(iof12)
c----------------------------------------------------------------------
c  calculates logs of prefactors and phase space integral
c  for ultrarelativistic limit (massless particles) and (2*s_i+1)=1
c  f12log and w15log=w35log+f12log not calculated calculated for iof12=0
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/ctst/psulog,wtulog
      integer ii(maxp)
      common /clatt/nlattc,npmax

      pi=3.1415927
      hquer=0.197327
      ish0=ish
      if(ishsub/100.eq.23)ish=mod(ishsub,100)
      do i=1,np
      ii(i)=1
      enddo

      if(ish.ge.7)write(ifch,*)('-',i=1,10)
     *,' entry sr hnbtst ',('-',i=1,30)
      if(ish.ge.7)write(ifch,*)'configuration:'
      if(ish.ge.7)write(ifch,*)(ident(i),i=1,np)
      if(ish.ge.7)write(ifch,*)'n_l:',nlattc,'   n_0:',nlattc-np

c log of prod m_i*volu/4/pi**3/hquer**3 -> f5log
      f5log=0
      do i=1,np
      call hnbfaf(i,gg,am,ioma)
      f5log=f5log+alog(gg*am*volu/4/pi**3/hquer**3)
      enddo
      if(ish.ge.7)write(ifch,*)'log(f5):',f5log

c log f4log=0
      f4log=0
      if(ish.ge.7)write(ifch,*)'log(f4):',f4log

c log of 1/prod n_alpha! -> f3log
      dbllog=0
      n1=1
      nx=1
    1 continue
      i=0
      x=0
      do n2=n1,np
      if(ident(n2).eq.ident(n1))then
      ii(n2)=0
      i=i+1
      x=x+alog(i*1.)
      endif
      if(ii(n2).ne.0.and.n2.gt.n1.and.nx.eq.n1
     *.and.ident(n2).ne.ident(n1))nx=n2
      enddo
      dbllog=dbllog+x
      if(nx.gt.n1)then
      n1=nx
      goto1
      endif
      f3log=-dbllog
      if(ish.ge.7)write(ifch,*)'log(f3):'
     *,f3log

c log of f3 * f4 * f5
      f35log=f5log+f4log+f3log
      if(ish.ge.7)write(ifch,*)'log(f3*f4*f5):',f35log

c log of phase space integral --> psilog
c ... initialization
      psilog=0.0
      if(iocova.eq.1)then
      psilog=alog(2.*np*np*(np-1)/tecm**4/pi)
      do i=1,np
      psilog=psilog+alog(tecm**2*pi/2./i/i)
      enddo
           elseif(iocova.eq.2)then
      psilog=-alog(2.*np-1)
      psilog=psilog+(np-1)*alog(pi/2.)
      do i=1,2*np-2
      psilog=psilog+alog((2.*np+i-2)/i)
      enddo
      do i=1,3*np-4
      psilog=psilog+alog(tecm/i)
      enddo
           endif
      if(ish.ge.7)write(ifch,*)'log(psi):',psilog

c log of phase space integral * f3 * f4 * f5
      w35log=f35log+psilog
      if(ish.ge.7)write(ifch,*)'log(f35*psi):',w35log

           if(iof12.ne.0)then

c log of macro/micro factor (f1*f2) --> f12log
      deglog=0
      do i=1,np
      deglog=deglog+alog(1.*i)
      enddo
      deglog=deglog+f3log
      do i=1,np
      deglog=deglog+alog(nlattc+1.-i)-alog(1.*i)
      enddo
      f12log=-deglog

      w15log=w35log+f12log
      if(ish.ge.7)then
        write(ifch,*)'log(f1*f2):',f12log
        write(ifch,*)'log(f15*psi):',w15log
        write(ifch,'(1x,4(a,3x))')
     *'log(fac):','log(psi):',' log(wt):','log(wta):'
        write(ifch,'(1x,4(f9.3,3x))')
     *f12log+f35log,psilog,w15log,w15log-f12log
      endif

           endif

      psulog=psilog
      wtulog=w35log

      if(ish.ge.7)write(ifch,*)('-',i=1,30)
     *,' exit sr hnbtst ',('-',i=1,10)
      ish=ish0
      return
      end

cc----------------------------------------------------------------------
c      subroutine hnbuex(x,e)
cc----------------------------------------------------------------------
cc  x --> x*10.**e with x.lt.10.**10.
cc----------------------------------------------------------------------
c           if(x.eq.0.)then
c      e=0.
c           else
c      e=int(alog10(abs(x)))/10*10
c      x=x/10.**e
c           endif
c      return
c      end
c
cc----------------------------------------------------------------------
c      subroutine hnbwin(n,w,q,i)
cc----------------------------------------------------------------------
cc  returns random index i according to weight w(i)
cc----------------------------------------------------------------------
c      real w(n),q(n)
c      q(1)=w(1)
c      do k=2,n
c      q(k)=q(k-1)+w(k)
c      enddo
c      y=rangen()*q(n)
c      do k=1,n
c      i=k
c      if(q(k).ge.y)goto1000
c      enddo
c      i=n
c1000  return
c      end
c
c----------------------------------------------------------------------
      subroutine hnbwri
c----------------------------------------------------------------------
c  writes (to ifch) an configuration
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/cfact/faclog
      write(ifch,'(1x,a,i5)')'np:',np
      write(ifch,'(1x,3(a,3x))')
     *'log(fac):','log(psi):',' log(wt):'
      if(wtlog.gt.-1e30.and.wtxlog.gt.-1e30)then
      write(ifch,'(1x,3(f9.3,3x))')faclog,wtxlog,wtlog
      else
      write(ifch,*)faclog,wtxlog,wtlog
      endif
      if(np.le.1)return
      call hnbtst(1)
      write(ifch,*)'particle id codes:'
      write(ifch,'(1x,10i6)')(ident(n),n=1,np)
      write(ifch,*)'particle masses:'
      write(ifch,'(1x,10f6.3)')(amass(n),n=1,np)
      end

c----------------------------------------------------------------------
      subroutine hnbzen(iii)
c----------------------------------------------------------------------
c analysis of events. energy spectra.
c for iii>0: filling histogram considering ptl iii
c----------------------------------------------------------------------
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nhise=100)
      common/chise/hise(mspecs,nhise)
      de=2./nhise/2.

      j=0

           if(iii.gt.0)then

      i=iii
      do l=1,nspecs
      if(ident(i).eq.ispecs(l))then
      j=l
      goto1
      endif
      enddo
    1 continue
      am=aspecs(j)
      e=pcm(4,i)
      ke=1+int((e-am)/(2*de))
      if(ke.ge.1.and.ke.le.nhise)hise(j,ke)=hise(j,ke)+1
      return

           else

      stop'STOP in hnbzen: iii=0'

           endif

      end

c----------------------------------------------------------------------
      subroutine hnbzmu(iii)
c----------------------------------------------------------------------
c analysis of events. multiplicity spectra.
c for iii<0: settting histograms to zero (should be first call)
c for iii>0: filling histogram considering ptl iii
c----------------------------------------------------------------------
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nhismu=500)
      common/chismu/hismu(mspecs,0:nhismu),hismus(nhismu)

           if(iii.lt.0)then

      do i=1,nhismu
      hismus(i)=0
      enddo
      do j=1,nspecs
      do i=0,nhismu
      hismu(j,i)=0
      enddo
      enddo
      goto1000

           elseif(iii.gt.0)then

      if(np.ge.1.and.np.le.nhismu)hismus(np)=hismus(np)+1
      do j=1,nspecs
      mu=0
      do i=1,np
      if(ident(i).eq.ispecs(j))mu=mu+1
      enddo
      if(mu.ge.0.and.mu.le.nhismu)hismu(j,mu)=hismu(j,mu)+1
      enddo
      goto1000

           else

      stop'STOP in sr hnbzmu: iii must not be 0'

           endif

1000  continue
      return
      end

c-----------------------------------------------------------------------
      subroutine xhgcam(amt,iii)
c-----------------------------------------------------------------------
c creates unnormalized histogram for total mass of grand
c canonically generated sample
c xpar1: nr. of bins
c xpar2: m_1 (lower boundary)
c xpar3: m_2 (upper boundary)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(nbmx=200)
      common/camdat/data(nbmx),datb(nbmx)
      parameter(mxclu=10000)
      real am(mxclu)
      character cen*6,cvol*6
c     the code needs am to be kept between calls
      save am
      data am /mxclu*0/

      if(iii.eq.0)then
      am(nrclu)=amt

      return

      elseif(iii.lt.0)then

      nbin=nint(xpar3)
      x1=xpar1
      x2=xpar2
      dam=(x2-x1)/nbin
      write(cen,'(f6.1)')tecm
      write(cvol,'(f6.1)')volu

      do i=1,nbin
      data(i)=x1+(i-1)*dam
      datb(i)=0.0
      enddo

      do i=1,nrclu
      xnb=(am(i)-x1)/dam+1.
      nb=nint(xnb)
      if(nb.le.nbin.and.nb.ge.1)datb(nb)=datb(nb)+1
      enddo

      write(ifhi,'(a)')       'newpage zone 1 2 1'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis total mass"'
      write(ifhi,'(a)')    'text 0 0 "yaxis N"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvol//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,nbin
      write(ifhi,'(2e13.5)')data(j),datb(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


      return

           endif

       end

c-----------------------------------------------------------------------
      subroutine xhgccc(chi)
c-----------------------------------------------------------------------
c creates unnormalized histogram for chi-squared test of initial
c configuration (grand-canonical results are used)
c for chi>0: chi-squared for each droplet configuration is written
c            to /cchi/
c for chi<0: creates histogram
c            xpar1 specifies lower limit
c            xpar2 specifies upper limit
c            xpar3 specifies bin width
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(nbin=200)
      common/chidat/data(nbin),datb(nbin)
      parameter(mxclu=10000)
      common/cchi/chi2(mxclu)
      character cnu*2,cinco*1,cen*6,cvol*6
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)

         if(chi.ge.0.0)then

      nrclu=nrclu+1
      chi2(nrclu)=chi

      return

         elseif(chi.lt.0.0)then

      x1=nint(xpar1)
      x2=nint(xpar2)
      da=xpar3
      write(cnu,'(i2)')nspecs
      write(cinco,'(i1)')ioinco
      write(cen,'(f6.1)')tecm
      write(cvol,'(f6.1)')volu

      if(x2.eq.0)x2=50.0
      da=max(0.1,da)
      a0=x1

      do i=1,nbin
      data(i)=a0+(i-1)*da
      datb(i)=0.0
      enddo

      do i=1,nrclu
      nb=(chi2(i)+da/2.-a0)/da
      if(nb.le.nbin.and.nb.ge.1)datb(nb)=datb(nb)+1
      enddo

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis [V]^2"'
      write(ifhi,'(a)')    'text 0 0 "yaxis f([V]^2,n?eff!)"'
      if(iappl.eq.4)write(ifhi,'(a,a)')'text 0.4 0.91 "V='//cvol//'"'
      if(iappl.eq.4)write(ifhi,'(a,a)')'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,nbin
      dat=datb(j)/nevent/da
      write(ifhi,'(2e13.5)')data(j),dat
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

      return

           endif

       end

c-----------------------------------------------------------------------
      subroutine xhgcen
c-----------------------------------------------------------------------
c  creates energy spectrum plot for decayed QM-droplet
c  using grand canonical results
c input:
c  xpar1 specifies particle species by paige id, 0 for all
c  xpar2 and xpar3 specify xrange of plot
c  xpar4 specifies line type : dashed (0), dotted (1), full (2) dado (3)
c  xpar5 specifies statistics to be used ,(0) same as iostat
c                                         (1) boltzmann
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/citer/iter,itermx
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cbol/rmsbol(mspecs),ptlbol(mspecs),chebol(mspecs),tembol
      character ctem*5,cit*5,cen*6,cvo*6,chem*5

      idpa=nint(xpar1)
      x1=xpar2
      x2=xpar3
      ltyp=nint(xpar4)
      ist=nint(xpar5)
      if(ist.eq.0.and.iostat.eq.1)ist=1

      id=0
      jx=100
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

      dx=(x2-x1)/2./jx
      x0=x1+dx

         do j=1,jx
         datx(j)=x0+(j-1)*dx*2.
         daty(j)=0.0

       if(id.eq.0)then

      do 10 i=1,nspecs
      dnde=0.0
        if(datx(j).ge.aspecs(i))then
      x=100.
      if(tem.ne.0.0.and.ist.eq.0)x=(datx(j)-chemgc(i))/tem
      if(tem.ne.0.0.and.ist.eq.1)x=(datx(j)-chebol(i))/tembol
      igsp=gspecs(i)
       if(x.ge.60)goto10
       if(mod(igsp,2).eq.0.and.ist.eq.0)then
      dnde=1./(exp(x)+1.)
       elseif(x.le.1.e-7.and.ist.eq.0)then
      dnde=1.e7
       elseif(ist.eq.0)then
      dnde=1./(exp(x)-1.)
       elseif(ist.eq.1)then
      dnde=exp(-x)
       endif
        endif
      daty(j)=daty(j)+dnde*gspecs(i)*volu/hquer**3/8./pi**3
10    continue

       else

      dnde=0.0
        if(datx(j).ge.aspecs(id))then
      x=100.
      if(tem.ne.0.0.and.ist.eq.0)x=(datx(j)-chemgc(id))/tem
      if(tem.ne.0.0.and.ist.eq.1)x=(datx(j)-chebol(id))/tembol
      igsp=gspecs(id)
       if(x.ge.60)goto11
       if(mod(igsp,2).eq.0.and.ist.eq.0)then
      dnde=1./(exp(x)+1.)
       elseif(x.le.1.e-7.and.ist.eq.0)then
      dnde=1.e7
       elseif(ist.eq.0)then
      dnde=1./(exp(x)-1.)
       elseif(ist.eq.1)then
      dnde=exp(-x)
       endif
        endif
11    daty(j)=dnde*gspecs(id)*volu/hquer**3/8./pi**3

       endif

         enddo

      ctem='     '
      chem='     '
      if(tem.gt.0.)write(ctem,'(f5.3)')tem
      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      if(id.gt.0)write(chem,'(f5.3)')chemgc(id)
      write(cit,'(i5)')itermx
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      elseif(ltyp.eq.3)then
      write(ifhi,'(a)')       'htyp ldd'
      endif
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis E?[n]! (GeV)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dN?[n]!/d^3!p"'
      write(ifhi,'(a,a)')     'text 0.3 0.10 "T='//ctem//'"'
      write(ifhi,'(a,a)')     'text 0.3 0.20 "[m]?[n]!='//chem//'"'
      write(ifhi,'(a,a)')     'text 0.3 0.20 "i?max!='//cit//'"'
      if(iocite.ne.1)then
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      endif
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

      return
      end

c-----------------------------------------------------------------------
      subroutine xhgcfl(u,d,s,iii)
c-----------------------------------------------------------------------
c creates unnormalized histogram for net flavor content of grand
c canonically generated sample
c xpar1: specifies width of plot, netflavor centered
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(nb=200)
      common/cfldat/data(nb),datb(nb),datc(nb),datu(nb)
     *,datd(nb),dats(nb)
      parameter(mxclu=10000)
      integer ku(mxclu),kd(mxclu),ks(mxclu)
      character cfl*3,cen*6,cvol*6
      save ku,kd,ks
      data ku/mxclu*0/,kd/mxclu*0/,ks/mxclu*0/

      if(iii.eq.0)then
      
      ku(nrclu)=u
      kd(nrclu)=d
      ks(nrclu)=s

      return

      elseif(iii.lt.0)then

      kwid=nint(xpar1)
      nbin=2*kwid+1
      x1u=keu-kwid
      x2u=keu+kwid
      x1d=ked-kwid
      x2d=ked+kwid
      x1s=kes-kwid
      x2s=kes+kwid
      write(cen,'(f6.1)')tecm
      write(cvol,'(f6.1)')volu

      do i=1,nbin
      data(i)=x1u+(i-1)
      datb(i)=x1d+(i-1)
      datc(i)=x1s+(i-1)
      datu(i)=0.0
      datd(i)=0.0
      dats(i)=0.0
      enddo

      do i=1,nrclu
      nbu=(ku(i)-x1u+1)
      nbd=(kd(i)-x1d+1)
      nbs=(ks(i)-x1s+1)
      if(nbu.le.nbin.and.nbu.ge.1)datu(nbu)=datu(nbu)+1
      if(nbd.le.nbin.and.nbd.ge.1)datd(nbd)=datd(nbd)+1
      if(nbs.le.nbin.and.nbs.ge.1)dats(nbs)=dats(nbs)+1
      enddo

      write(ifhi,'(a)')       'newpage zone 1 3 1'

      write(cfl,'(i3)')keu
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1u,x2u
      write(ifhi,'(a)')    'text 0 0 "xaxis net u content"'
      write(ifhi,'(a)')    'text 0 0 "yaxis N"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvol//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a,a)')     'text 0.65 0.91 "N?u!='//cfl//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,nbin
      write(ifhi,'(2e13.5)')data(j),datu(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(cfl,'(i3)')ked
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1d,x2d
      write(ifhi,'(a)')    'text 0 0 "xaxis net d content"'
      write(ifhi,'(a)')    'text 0 0 "yaxis N"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvol//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a,a)')     'text 0.65 0.91 "N?d!='//cfl//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,nbin
      write(ifhi,'(2e13.5)')datb(j),datd(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(cfl,'(i3)')kes
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1s,x2s
      write(ifhi,'(a)')    'text 0 0 "xaxis net s content"'
      write(ifhi,'(a)')    'text 0 0 "yaxis N"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvol//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a,a)')     'text 0.65 0.91 "N?s!='//cfl//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,nbin
      write(ifhi,'(2e13.5)')datc(j),dats(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      return

           endif

       end

c-----------------------------------------------------------------------
      subroutine xhgcmt
c-----------------------------------------------------------------------
c creates transverse mass spectrum for QM-droplet decay
c according to grand canonical results
c input:
c  xpar1 specifies particle species by paige id, 0 for all
c  xpar2 and xpar3 specify xrange of plot
c  xpar4 specifies line type : dashed (0), dotted (1), full (2)
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/citer/iter,itermx
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      character cen*6,cvo*6,cit*5,ctem*5

      idpa=nint(xpar1)
      x1=xpar2
      x2=xpar3
      ltyp=nint(xpar4)

      id=0
      jx=100
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

      dx=(x2-x1)/2./jx
      x0=x1+dx

         do j=1,jx
         datx(j)=x0+(j-1)*dx*2.
         daty(j)=0.0

       if(id.eq.0)then

      do 10 i=1,nspecs
      dndmt=0.0
      if(datx(j).ge.aspecs(i))then
      x=100.
      xx=100.
      if(tem.ne.0.)x=datx(j)/tem
      if(tem.ne.0.)xx=chemgc(i)/tem
      if(abs(xx).le.60)dndmt=gspecs(i)*volu/hquer**3*exp(xx)*datx(j)
     */4./pi**3*hgcbk1(x)
      endif
      daty(j)=daty(j)+dndmt
10    continue

       else

      dndmt=0.0
      if(datx(j).ge.aspecs(id))then
      x=100.
      xx=100.
      if(tem.ne.0.)x=datx(j)/tem
      if(tem.ne.0.)xx=chemgc(id)/tem
      if(abs(xx).le.60)dndmt=gspecs(id)*volu/hquer**3*exp(xx)*datx(j)
     */4./pi**3*hgcbk1(x)
      endif
      daty(j)=dndmt

       endif

         enddo

      write(cit,'(i5)')itermx
      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      write(ctem,'(f5.3)')tem
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      endif
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis m?t! (GeV)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dN?[n]!/d^2!m?t! "'
      write(ifhi,'(a,a)')     'text 0.3 0.10 "T='//ctem//'"'
      write(ifhi,'(a,a)')     'text 0.3 0.20 "i?max!='//cit//'"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

      return
      end

c-----------------------------------------------------------------------
      subroutine xhgcmu
c-----------------------------------------------------------------------
c creates multiplicity plot for decayed QM-droplet
c according to grand canonical results
c input:
c  xpar1 specifies species by paige id, 0 for total multiplicity
c  xpar2 specifies xrange to be set automatically (0) or by hand (1)
c  xpar3 and xpar4 xrange if xpar2 ne 0
c  xpar5 xrange = average+-sigma*xpar5
c  xpar6 specifies line type : dashed (0), dotted (1), full (2)
c  xpar7 specifies statistics : same as iostat (0)
c                               boltzmann (1)
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cbol/rmsbol(mspecs),ptlbol(mspecs),chebol(mspecs),tembol
      common/cgctot/rmstot,ptltot
      character cyield*8,cen*6,cvo*6,cinco*1


      idpa=nint(xpar1)
      ixra=nint(xpar2)
      iwid=nint(xpar5)
      ltyp=nint(xpar6)
      ist=nint(xpar7)
      if(ist.eq.0.and.iostat.eq.1)ist=1


      pn=0.0
      id=0
      jx=100
      ymin=1./nevent/10.
      if(nevent.le.10)ymin=ymin/10.
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

       if(ixra.eq.1)then
      x1=anint(xpar3)
      x2=anint(xpar4)
       else
      if(id.eq.0)then
      x1=anint(ptltot-iwid*rmstot)
      x2=anint(ptltot+iwid*rmstot)
      else
      x1=anint(ptlngc(id)-iwid*rmsngc(id))
      x2=anint(ptlngc(id)+iwid*rmsngc(id))
      endif
      x2=max(x2,3.0)
       endif

      x1=max(x1,0.0)
      dx=(x2-x1)/2./jx
      x0=x1+dx
      pn=0.0

      do j=1,jx
      datx(j)=x0+(j-1)*dx*2.
      if(id.eq.0)then

c     total multiplicity
c     ------------------
      x=100.
      if(rmstot.ge.1.e-10)x=(datx(j)-ptltot)**2/rmstot**2/2.

       if(x.ge.60)then
      pn=0.0
       else
      pn=exp(-x)/rmstot/sqrt(2.*pi)
       endif

      daty(j)=pn

         else

c     one species (specified by id)
c     ------------------------------
      x=100.
      if(rmsngc(id).ge.1.e-10.and.ist.eq.0)
     *x=(datx(j)-ptlngc(id))**2/rmsngc(id)**2/2.
      if(rmsbol(id).ge.1.e-10.and.ist.eq.1)
     *x=(datx(j)-ptlbol(id))**2/rmsbol(id)**2/2.

       if(x.ge.60)then
      pn=0.0
       else
      if(ist.eq.0)pn=exp(-x)/rmsngc(id)/sqrt(2*pi)
      if(ist.eq.1)pn=exp(-x)/rmsbol(id)/sqrt(2*pi)
       endif

      daty(j)=pn

         endif
         enddo

      if(id.eq.0)then
      write(cyield,'(f8.3)')ptltot
      else
      write(cyield,'(f8.3)')ptlngc(id)
      endif
      write(cinco,'(i1)')ioinco
      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      elseif(ltyp.eq.3)then
      write(ifhi,'(a)')       'htyp ldd'
      endif
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a,e11.3,a)')'yrange',ymin,'  auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis N?[n]!"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(N?[n]!)"'
      write(ifhi,'(a,a)')'text 0.3 0.10 "" "L#N?[n]!"G#='//cyield//'""'
      write(ifhi,'(a,a)')     'text 0.3 0.2 "conf?in!='//cinco//'"'
      if(iocite.ne.1)then
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      endif
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'


      return
      end


c-----------------------------------------------------------------------
      subroutine xhgcmx
c-----------------------------------------------------------------------
c creates multiplicity plot for decayed QM-droplet
c according to grand canonical results POISSON DISTRIB.!!!!
c input:
c  xpar1 specifies species by paige id, 0 for total multiplicity
c  xpar2 specifies xrange to be set automatically (0) or by hand (1)
c  xpar3 and xpar4 xrange if xpar2 ne 0
c  xpar5 xrange = average+-sigma*xpar5
c  xpar6 specifies line type : dashed (0), dotted (1), full (2) dado (3)
c  xpar7 specifies statistics : same as iostat (0)
c                               boltzmann (1)
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cbol/rmsbol(mspecs),ptlbol(mspecs),chebol(mspecs),tembol
      common/cgctot/rmstot,ptltot
      character cyield*8,cen*6,cvo*6,cinco*1


      idpa=nint(xpar1)
      ixra=nint(xpar2)
      iwid=nint(xpar5)
      ltyp=nint(xpar6)
      ist=nint(xpar7)
      if(ist.eq.0.and.iostat.eq.1)ist=1
      pn=0.


      id=0
      ymin=1./nevent/10.
      if(nevent.le.10)ymin=ymin/10.
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

       if(ixra.eq.1)then
      n1=nint(xpar3)
      n2=nint(xpar4)
       else
      if(id.eq.0)then
      n1=nint(ptltot-iwid*rmstot)
      n2=nint(ptltot+iwid*rmstot)
      else
      n1=nint(ptlngc(id)-iwid*rmsngc(id))
      n2=nint(ptlngc(id)+iwid*rmsngc(id))
      endif
      n2=max(n2,3)
       endif

      n1=max(n1,0)
      jx=n2+1

      do j=1,jx
      datx(j)=j-1
      jf=1
      if(j.gt.1)then
      do i=1,j-1
      jf=jf*i
      enddo
      endif
      if(id.eq.0)then

c     total multiplicity
c     ------------------

      daty(j)=1./jf*ptltot**(j-1)*exp(-ptltot)

         else

c     one species (specified by id)
c     ------------------------------

      if(ist.eq.0)pn=1./jf*ptlngc(id)**(j-1)*exp(-ptlngc(id))
      if(ist.eq.1)pn=1./jf*ptlbol(id)**(j-1)*exp(-ptlbol(id))

      daty(j)=pn

         endif
         enddo

      if(id.eq.0)then
      write(cyield,'(f8.3)')ptltot
      else
      write(cyield,'(f8.3)')ptlngc(id)
      endif
      write(cinco,'(i1)')ioinco
      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      elseif(ltyp.eq.3)then
      write(ifhi,'(a)')       'htyp ldd'
      endif
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2i3)')'xrange',n1,n2
      write(ifhi,'(a,e11.3,a)')'yrange',ymin,'  auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis N?[n]!"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(N?[n]!)"'
      write(ifhi,'(a,a)')'text 0.3 0.10 "" "L#N?[n]!"G#='//cyield//'""'
      write(ifhi,'(a,a)')     'text 0.3 0.2 "conf?in!='//cinco//'"'
      if(iocite.ne.1)then
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      endif
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'


      return
      end

c-----------------------------------------------------------------------
      subroutine xhgcpt
c-----------------------------------------------------------------------
c creates transverse momentum spectrum for decayed QM-droplet
c according to grand canonical results
c input:
c  xpar1 specifies particle species by paige id, 0 for all
c  xpar2 rapidity window
c  xpar3 and xpar4 specify xrange of plot
c  xpar5 specifies line type : dashed (0), dotted (1), full (2)
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/citer/iter,itermx
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      character crap*5,cen*6,cvo*6,cit*5

      idpa=nint(xpar1)
      y=xpar2
      x1=xpar3
      x2=xpar4
      ltyp=xpar5

      write(crap,'(f5.1)')y
      id=0
      jx=100
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

      dx=(x2-x1)/2./jx
      x0=x1+dx

         do j=1,jx
         datx(j)=x0+(j-1)*dx*2.
         daty(j)=0.0

       if(id.eq.0)then

      do 10 i=1,nspecs
      x=100.
      if(tem.ne.0.)
     *x=(sqrt(aspecs(i)**2+datx(j)**2)*cosh(y)-chemgc(i))/tem
       if(x.ge.60)then
      dndpt=0.0
       else
      dndpt=exp(-x)
       endif
      dndpt=dndpt*gspecs(i)*volu/hquer**3*cosh(y)
     **sqrt(aspecs(i)**2+datx(j)**2)/8./pi**3
      daty(j)=daty(j)+dndpt
10    continue

       else

      x=100.
      if(tem.ne.0.)
     *x=(sqrt(aspecs(id)**2+datx(j)**2)*cosh(y)-chemgc(id))/tem
       if(x.ge.60)then
      dndpt=0.0
       else
      dndpt=exp(-x)
       endif
      dndpt=dndpt*gspecs(id)*volu/hquer**3*cosh(y)
     **sqrt(aspecs(id)**2+datx(j)**2)/8./pi**3
      daty(j)=dndpt

       endif

         enddo

      write(cit,'(i5)')itermx
      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      endif
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis p?t! (GeV/c)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dN?[n]!/dyd^2!p?t!"'
      write(ifhi,'(a)')    'text 0.10 0.10 "y = '//crap//'"'
      write(ifhi,'(a)')    'text 0.10 0.30 "i?max! = '//cit//'"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

      return
      end

c-----------------------------------------------------------------------
      subroutine xhgcra
c-----------------------------------------------------------------------
c creates rapidity distribution for decayed QM-droplet
c according to grand canonical results
c input:
c  xpar1 specifies particle species by paige id, 0 for all
c  xpar2 and xpar3 specify xrange of plot
c  xpar4 specifies line type : dashed (0), dotted (1), full (2)
c output:
c  histo-file
c  newpage, zone and plot commands not included !!!
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nbin=200)
      real datx(nbin),daty(nbin)
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgctot/rmstot,ptltot
      character cen*6,cvo*6,cng*8

      idpa=nint(xpar1)
      x1=nint(xpar2)
      x2=nint(xpar3)
      ltyp=nint(xpar4)

      id=0
      jx=100
      ymin=1./nevent/10.
      if(nevent.le.10)ymin=ymin/10.
      do i=1,nspecs
      if(ispecs(i).eq.idpa)id=i
      enddo

      dx=(x2-x1)/2./jx
      x0=x1+dx

         do j=1,jx

         datx(j)=x0+(j-1)*dx*2.
         daty(j)=0.0
         y=datx(j)
         if(ish.ge.9)write(ifch,*)'cosh y:',cosh(y)

       if(id.eq.0)then

      do 10 i=1,nspecs
      dndy=0.0
      sum=aspecs(i)**2*tem+2.*aspecs(i)*tem**2/cosh(y)
     *+2.*tem**3/cosh(y)**2
      x=100.
      if(tem.ne.0.0)
     *x=(aspecs(i)*cosh(y)-chemgc(i))/tem

       if(x.ge.60.)then
      pro=0.0
       else
      pro=exp(-x)
      endif

      pro=pro*gspecs(i)*volu/hquer**3/4./pi**2

      if(pro.ge.(1.e-30).and.sum.ge.(1.e-30))then
      che=alog(pro)+alog(sum)
      else
      che=-61.0
      endif
      if(che.le.60.0.and.che.ge.(-60.0))dndy=pro*sum
c     if(che.le.60.0.and.che.ge.(-60.0))dndy=exp(che)

      daty(j)=daty(j)+dndy

10    continue

       else

      dndy=0.0
      sum=aspecs(id)**2*tem+2.*aspecs(id)*tem**2/cosh(y)
     *+2.*tem**3/cosh(y)**2
      x=100.
      if(tem.ne.0.0)
     *x=(aspecs(id)*cosh(y)-chemgc(id))/tem

       if(x.ge.60.)then
      pro=0.0
       else
      pro=exp(-x)
      endif

      pro=pro*gspecs(id)*volu/hquer**3/4./pi**2

      if(pro.ge.(1.e-30).and.sum.ge.(1.e-30))then
      che=alog(pro)+alog(sum)
      else
      che=-61.0
      endif
      if(che.le.60..and.che.ge.-60.)dndy=pro*sum

      daty(j)=dndy

       endif

         enddo

      write(cen,'(f6.1)')tecm
      write(cvo,'(f6.1)')volu
      if(id.eq.0)then
      write(cng,'(f8.3)')ptltot
      else
      write(cng,'(f8.3)')ptlngc(id)
      endif
      write(ifhi,'(a)')       'openhisto'
      if(ltyp.eq.0)then
      write(ifhi,'(a)')       'htyp lda'
      elseif(ltyp.eq.1)then
      write(ifhi,'(a)')       'htyp ldo'
      elseif(ltyp.eq.2)then
      write(ifhi,'(a)')       'htyp lfu'
      endif

      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a,e11.3,a)')'yrange',ymin,'  auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dN?[n]!/dy"'
      write(ifhi,'(a,a)')     'text 0.4 0.91 "V='//cvo//'"'
      write(ifhi,'(a,a)')     'text 0.15 0.91 "E='//cen//'"'
      write(ifhi,'(a,a)')     'text 0.3 0.10 "N?[n]!='//cng//'"'
      write(ifhi,'(a)')       'array 2'

         do j=1,jx
      write(ifhi,'(2e12.4)')datx(j),daty(j)
         enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

      return
      end

c-----------------------------------------------------------------------
      subroutine xhnben
c-----------------------------------------------------------------------
c produces histogram of energy spectrum (after metropolis run)
c complete histogram: openhisto ... closehisto
c iocite=1 required
c-----------------------------------------------------------------------
c xpar1: particle species (venus id-code)
c xpar2: 1: actual spectrum 2: fit
c xpar3: 1: de/d3p 2: ede/d3e
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nhise=100)
      common/chise/hise(mspecs,nhise)
      parameter (literm=500)
      common/cmet/kspecs(mspecs),liter,lspecs(literm,mspecs)
     *,iterl(literm),iterc(literm)
      real datx(nhise),daty(nhise),dats(nhise)
      common/citer/iter,itermx
      character ch*1,chid*5,cyield*9,ctem*5
      de=2./nhise/2.

      if(iocite.ne.1)stop'STOP: xhnben: iocite=1 required'

      idcode=nint(xpar1)
      mode=nint(xpar2)
      kind=nint(xpar3)

           do j=1,nspecs
           if(idcode.eq.ispecs(j))then

      id=idcode
      am=aspecs(j)
      yield=1.*kspecs(j)/(itermx-iternc)
      if(kind.eq.1)ch=' '
      if(kind.eq.2)ch='e'
      ll=kind-1
      e0=am+de
      nebins=0
        do i=1,nhise
      e=e0+(i-1)*2*de
      p1=sqrt((e-de)**2-am**2)
      p2=sqrt((e+de)**2-am**2)
      d3p=4*pi*(p2**3-p1**3)/3
      datx(i)=e
      y=(1-ll+ll*e)*hise(j,i)/(itermx-iternc)/d3p
      if(y.gt.0.)then
      nebins=nebins+1
      daty(i)=alog(y)
      d=y/sqrt(hise(j,i))
      dats(i)=1e10
      if(y-d.gt.0.)dats(i)=alog(y+d)-alog(y-d)
      else
      daty(i)=-100
      dats(i)=1e10
      endif
c-c   if(e.lt.0.2)dats(i)=1e10
        enddo
      a=0.
      b=0.
        if(nebins.ge.3)then
      call utfit(datx,daty,nhise,dats,1,a,b,siga,sigb,chi2,q)
      tem=-1./b
      if(tem.lt.0.050.or.tem.gt.10.)then
      tem=0.
      a=0.
      b=0.
      endif
        endif
      do i=1,nhise
      daty(i)=exp(daty(i))
      enddo
      write(chid,'(i5)')id
      write(cyield,'(f9.4)')yield
      ctem='     '
      if(tem.gt.0.)write(ctem,'(f5.3)')tem
      write(ifhi,'(a)')    'openhisto xrange 0 3'
      write(ifhi,'(a)')    'htyp lin xmod lin ymod log'
      write(ifhi,'(a,a)')  'text 0 0 "title id='//chid
     *                           ,'   N='//cyield//'   T='//ctem//'"'
      write(ifhi,'(a)')    'text 0 0 "xaxis energy (GeV)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis '//ch//' dn/d3p (GeV-3)"'
      write(ifhi,'(a)')    'array 2'
      do i=1,nhise
      if(mode.eq.1)write(ifhi,'(2e12.4)')datx(i),daty(i)
      if(mode.eq.2)write(ifhi,'(2e12.4)')datx(i),exp(a+b*datx(i))
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'

           endif
           enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine xhnbit
c-----------------------------------------------------------------------
c produces histogram of multiplicity versus iterations (after metropolis run)
c complete histogram: openhisto ... closehisto
c iocite=1 required
c-----------------------------------------------------------------------
c xpar1: particle species (0=all, else venus id-code)
c xpar2: 1:actual multiplicity 2:average multiplicity 3:grand canonical
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (literm=500)
      common/cmet/kspecs(mspecs),liter,lspecs(literm,mspecs)
     *,iterl(literm),iterc(literm)
      real datlx(literm),datly(literm)
      common/citer/iter,itermx
      character chid*5,ctecm*5,cvolu*6
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      common/cgctot/rmstot,ptltot

      if(iocite.ne.1)stop'STOP: xhnbit: iocite=1 required'

      idcode=nint(xpar1)
      mode=nint(xpar2)

           if(idcode.eq.0)then

      yield=0
      do j=1,nspecs
      yield=yield+1.*kspecs(j)/(itermx-iternc)
      enddo
      datlx(1)=(iterl(1)+1)/2.
      do li=2,liter-1
      datlx(li)=(iterl(li)+iterl(li-1)+1)/2.
      enddo
      x1=0
      x2=iterl(liter-1)
      do li=1,liter-1
      y=0
      do j=1,nspecs
      y=y+lspecs(li,j)
      enddo
      if(mode.eq.1)datly(li)=y/iterc(li)
      if(mode.eq.2)datly(li)=yield
      if(mode.eq.3)datly(li)=ptltot
      enddo
      write(ctecm,'(f5.1)')tecm
      write(cvolu,'(f6.1)')volu
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,a)')     'text 0 0 "title E = '//ctecm//'   V = '
     *                                 ,cvolu//'"'
      write(ifhi,'(a)')       'text 0 0 "xaxis iterations"'
      write(ifhi,'(a)')       'text 0 0 "yaxis multiplicity"'
      write(ifhi,'(a)')       'array 2'
      do i=1,liter-1
      write(ifhi,'(2e12.4)')   datlx(i),datly(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

           else

           do j=1,nspecs
           if(idcode.eq.ispecs(j))then

      yield=1.*kspecs(j)/(itermx-iternc)
      write(chid,'(i5)')idcode
      do li=1,liter-1
      datlx(li)=iterl(li)
      enddo
      x1=0
      x2=datlx(liter-1)
      do li=1,liter-1
      if(mode.eq.1)datly(li)=lspecs(li,j)*1./iterc(li)
      if(mode.eq.2)datly(li)=yield
      if(mode.eq.3)datly(li)=ptlngc(j)
      enddo
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "title id='//chid//'"'
      write(ifhi,'(a)')       'text 0 0 "xaxis iterations "'
      write(ifhi,'(a)')       'text 0 0 "yaxis multiplicity"'
      write(ifhi,'(a)')       'array 2'
      do i=1,liter-1
      write(ifhi,'(2e12.4)')   datlx(i),datly(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

           endif
           enddo

           endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xhnbmu
c-----------------------------------------------------------------------
c produces histogram of multiplicity distribution (after metropolis run)
c complete histogram: openhisto ... closehisto
c iocite=1 required
c-----------------------------------------------------------------------
c xpar1: particle species (0=all, else venus id-code)
c xpar2: xrange automatic (0) or given via xpar3,4 (else)
c xpar3,4: xrange
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nhismu=500)
      common/chismu/hismu(mspecs,0:nhismu),hismus(nhismu)
      parameter (literm=500)
      common/cmet/kspecs(mspecs),liter,lspecs(literm,mspecs)
     *,iterl(literm),iterc(literm)
      real datx(nhismu),daty(nhismu)
      common/citer/iter,itermx
      common /clatt/nlattc,npmax
      character chid*5,cyield*9,ctecm*5,cvolu*6

      if(iocite.ne.1)stop'STOP: xhnbmu: iocite=1 required'

      idcode=nint(xpar1)
      ixr=nint(xpar2)
      xx1=xpar3
      xx2=xpar4

      write(ctecm,'(f5.1)')tecm
      write(cvolu,'(f6.1)')volu

           if(idcode.eq.0)then

      yield=0
      do j=1,nspecs
      yield=yield+1.*kspecs(j)/(itermx-iternc)
      enddo
      write(cyield,'(f9.4)')yield
      i1=0
      i2=nlattc
      mus=0
      do i=1,nhismu
      if(i1.eq.0.and.nint(hismus(i)).gt.0)i1=i
      if(nint(hismus(i)).gt.0)i2=i
      mus=mus+hismus(i)
      enddo
      ij=0.5*(i1+i2)*0.20
      if(itermx.le.1000)ij=0.5*(i1+i2)*0.40
      if(itermx.le.100)ij=0.5*(i1+i2)*0.80
      i1=i1-ij
      i1=max(i1,2)
      i2=i2+ij
      ii=10
      if(i1.le.50)ii=5
      if(i1.le.20)ii=2
      i1=i1/ii*ii
      i2=i2/ii*ii+ii
           do i=i1,i2
      l=1+i-i1
      datx(l)=i
      daty(l)=hismus(i)/mus
           enddo
      jx=1+i2-i1
      if(ixr.eq.0)then
      x1=i1
      x2=i2
      else
      x1=xx1
      x2=xx2
      endif
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp lin xmod lin ymod log'
      write(ifhi,'(a,a)')     'text 0 0 "title E = '//ctecm//'   V = '
     *                              ,cvolu//'"'
      write(ifhi,'(a)')       'text 0 0 "xaxis multiplicity n  "'
      write(ifhi,'(a)')       'text 0 0 "yaxis dN/dn"'
      write(ifhi,'(a)')       'text 0.30 0.25 "N?MC!='//cyield//'"'
      write(ifhi,'(a)')       'array 2'
      do i=1,jx
      write(ifhi,'(2e12.4)')   datx(i),daty(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

           else

           do j=1,nspecs
           if(idcode.eq.ispecs(j))then

      yield=1.*kspecs(j)/(itermx-iternc)
      write(cyield,'(f9.4)')yield
      write(chid,'(i5)')idcode
      i1=0
      i2=nlattc
      mus=0
      do i=0,nhismu
      if(i1.eq.0.and.nint(hismu(j,i)).gt.0)i1=i
      if(nint(hismu(j,i)).gt.0)i2=i
      mus=mus+hismu(j,i)
      enddo
      ij=0.5*(i1+i2)*0.30
      if(itermx.le.1000)ij=0.5*(i1+i2)*0.60
      if(itermx.le.100)ij=0.5*(i1+i2)*1.20
      i1=i1-ij
      i1=max(i1,0)
      i2=i2+ij
      ii=10
      if(i1.le.50)ii=5
      if(i1.le.20)ii=2
      i1=i1/ii*ii
      i2=i2/ii*ii+ii
           do i=i1,i2
      l=1+i-i1
      datx(l)=i
      daty(l)=hismu(j,i)/mus
           enddo
      jx=1+i2-i1
      if(ixr.eq.0)then
      x1=i1
      x2=i2
      else
      x1=xx1
      x2=xx2
      endif
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp lin xmod lin ymod log'
      write(ifhi,'(a)')       'text 0 0 "title id='//chid//'"'
      write(ifhi,'(a)')       'text 0 0 "xaxis multiplicity n  "'
      write(ifhi,'(a)')       'text 0 0 "yaxis dN/dn"'
      write(ifhi,'(a)')       'text 0.30 0.25 "N?MC!='//cyield//'"'
      write(ifhi,'(a)')       'array 2'
      do i=1,jx
      write(ifhi,'(2e12.4)')   datx(i),daty(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

           endif
           enddo

           endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xhnbmz
c-----------------------------------------------------------------------
c produces histogram of multiplicity distribution from droplet decay
c or average multiplicity versus iterations
c for massless hadrons
c complete histogram: openhisto ... closehisto
c-----------------------------------------------------------------------
c xpar1: particle species (0=all, else venus id-code)
c xpar2: lower limit multiplicity
c xpar3: upper limit multiplicity
c xpar4: lower limit total multiplicity   (also necc for xpar1.ne.0)
c xpar5: upper limit  "      "            (also necc for xpar1.ne.0)
c xpar6: sets htyp: 1->lfu, 2->ldo, 3->lda, 4->ldd
c xpar7: 0: multiplicity distribution
c        >0: av multiplicity vs iterations (itermx=xpar7)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/ctst/psulog,wtulog
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      parameter (nhismu=500)
      common/cflac/ifok(nflav,mspecs),ifoa(nflav)
      real datx(nhismu),datyu(nhismu)
      character cyieur*9
      real pzlog(nhismu)
      double precision spelog,cc,bb,dsu
      common/cyield/yield
      character*3 htyp

      idcode=nint(xpar1)
      x1=xpar2
      x2=xpar3
      i1=nint(xpar2)
      i2=nint(xpar3)
      ii1=nint(xpar4)
      ii2=nint(xpar5)
      ih=nint(xpar6)
      htyp='lin'
      if(ih.eq.1)htyp='lfu'
      if(ih.eq.2)htyp='ldo'
      if(ih.eq.3)htyp='lda'
      if(ih.eq.4)htyp='ldd'
      itmax=nint(xpar7)

      wtrlog=-1e30
           do i=ii1,ii2
      if(i.ge.2)then
      np=i
      do k=1,np
      ident(k)=110
      enddo
      call hnbtst(0)
      wtzlog=wtulog
      if(ioflac.eq.0)call hnbspg(keu,ked,kes,kec,keb,ket,0,np,spelog)
      if(ioflac.ne.0)call hnbspf(keu,ked,kes,kec,keb,ket,0,np,spelog)
      wtulog=wtulog+spelog
      else
      wtzlog=-1000
      wtulog=-1000
      endif
      pzlog(1+i-ii1)=wtzlog
      datyu(1+i-ii1)=wtulog
      wtrlog=max(wtrlog,wtulog)
           enddo
      yield=0
      su=0
           do i=ii1,ii2
      l=1+i-ii1
      pzlog(l)=pzlog(l)-wtrlog
      datyu(l)=datyu(l)-wtrlog
      if(datyu(l).gt.-50.)then
      datyu(l)=exp(datyu(l))
      else
      datyu(l)=exp(-50.)
      endif
      yield=yield+i*datyu(l)
      su=su+datyu(l)
           enddo
      yield=yield/su
           do i=ii1,ii2
      l=1+i-ii1
      datx(l)=i
      datyu(l)=datyu(l)/su
           enddo
      jx=1+ii2-ii1
      write(cyieur,'(f9.4)')yield
c     ---
        if(idcode.eq.0.and.itmax.eq.0)then
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp '//htyp//' xmod lin ymod log'
      write(ifhi,'(a)')       'text 0.30 0.15 "N?ana!='//cyieur//'"'
      write(ifhi,'(a)')       'array 2'
      do i=1,jx
      write(ifhi,'(2e12.4)')   datx(i),datyu(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'
        elseif(idcode.eq.0)then
      write(ifhi,'(a,2e11.3)')'openhisto xrange',0.,itmax*1.
      write(ifhi,'(a)')       'htyp '//htyp//' xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      itm=20
      do i=1,itm
      write(ifhi,'(2e12.4)')   (i-1.)*itmax/(itm-1.),yield
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'
        endif
c     ---
      if(idcode.eq.0)return

           do j=1,nspecs
           if(idcode.eq.ispecs(j))then

      wtrlog=-1e30
           do i=i1,i2
      l=1+i-i1
      datx(l)=i
           enddo
      yield=0
      suj=0
      dsu=su
           do i=i1,i2
      l=1+i-i1
      bb=0
      nfi=0
      do ntot=max(i+1,ii1),min(i2*nspecs,ii2)
      nfi=nfi+1
      cc=1d0
      do kc=1,i
      cc=cc*(1.+ntot-kc)/kc*gspecs(j)
      enddo
      ku=keu-i*ifok(1,j)
      kd=ked-i*ifok(2,j)
      ks=kes-i*ifok(3,j)
      kc=kec-i*ifok(4,j)
      kb=keb-i*ifok(5,j)
      kt=ket-i*ifok(6,j)
      if(ioflac.eq.0)call hnbspg(ku,kd,ks,kc,kb,kt,j,ntot-i,spelog)
      if(ioflac.ne.0)call hnbspf(ku,kd,ks,kc,kb,kt,j,ntot-i,spelog)
      cc=cc*dexp(spelog)
      bb=bb+cc*dexp(1.d0*pzlog(1+ntot-ii1))/dsu
      enddo
      datyu(l)=bb
      yield=yield+i*datyu(l)
      suj=suj+datyu(l)
           enddo
      yield=yield/suj
      jx=1+i2-i1
      write(cyieur,'(f9.4)')yield
c     ---
        if(itmax.eq.0)then
      write(ifhi,'(a,2e11.3)')'openhisto xrange',x1,x2
      write(ifhi,'(a)')       'htyp '//htyp//' xmod lin ymod log'
      write(ifhi,'(a)')       'text 0.30 0.15 "N?ana!='//cyieur//'"'
      write(ifhi,'(a)')       'array 2'
      do i=1,jx
      write(ifhi,'(2e12.4)')   datx(i),datyu(i)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'
        else
      write(ifhi,'(a,2e11.3)')'openhisto xrange',0.,itmax*1.
      write(ifhi,'(a)')       'htyp '//htyp//' xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      itm=20
      do i=1,itm
      write(ifhi,'(2e12.4)')   (i-1.)*itmax/(itm-1.),yield
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'
        endif
c     ---
      return

           endif
           enddo

      end

c-----------------------------------------------------------------------
      subroutine xhnbte(iii)
c-----------------------------------------------------------------------
c fills histograms (iii>=0) or writes histogram to histo-file (iii<0)
c regarding exponential autocorrelation time and acceptance rate
c
c input:
c   requires complete run with application hadron (iappl=1)
c   or application metropolis (iappl=4)
c   ioceau=1 necessary
c
c  output:
c   for iii=0 (only valid for iappl=4):
c     data(nrevt): nrevt  (event number)               /cdat/
c     datb(nrevt): taui   (calculated corr time)       /cdat/
c     datc(nrevt): accrat (acceptance rate)            /cdat/
c     datd(nrevt): taue   (parametrized corr time)     /cdat/
c   for iii>0 (only valid for iappl=1):
c     nrclu=nrclu+1                                    /cnrclu/
c     data(nrclu): nrclu  (droplet number)             /cdat/
c     datb(nrclu): taui-taue (calc - param corr time)  /cdat/
c     datc(nrclu): accrat (acceptance rate)            /cdat/
c     datd(nrclu): avnp (average particle number)      /cdat/
c   for iii<0:
c     writes complete histogram (openhisto ... closehisto) to histofile
c       for iappl=4:                for iappl=1:
c         xpar1=1: (data,datb,datd) xpar1=1: (data,datb)
c         xpar1=2: (data,datc)      xpar1=2: (data,datd)
c                                   xpar1=3: (data,datc)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      common/citer/iter,itermx
      common /clatt/nlattc,npmax
      common/cgctot/rmstot,ptltot
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      parameter (nbin=500)
      common/cdat/ data(nbin),datb(nbin),datc(nbin),datd(nbin)
      real dev(maxit)
      character cobs*5,cnc*5,cdz*5,czer*5
     *,cmom*5,cnp*7,cen*7,cvol*7,clatt*5,cit*5
      common/ctaue/taue

      if(ioceau.ne.1)stop'STOP: ioceau=1 required'
      if(iii.eq.0.and.iappl.ne.4)stop'STOP: iappl=4 required'
      if(iii.gt.0.and.iappl.ne.1)stop'STOP: iappl=1 required'

      if(iii.lt.0)jjj=nint(xpar1)

      id=0
      ish0=ish
c     ish=98

c          ----------------
           if(iii.ge.0)then
c          ----------------

      if(iii.gt.0)nrclu=nrclu+1
      if(nrclu.gt.500)return

c     mean
c     ----
      xnptot=nptot
      avnp=xnptot/(itermx-iternc)
      if(ish.ge.9)write(ifch,*)'event:',nrevt,'   droplet:',nrclu
     *,'   avnp:',avnp

c     calculate corfct_0
c     ------------------
      corzer=0.0
      do i=iternc+1,itermx
      dev(i)=npit(i)-avnp
      corzer=corzer+dev(i)**2
      enddo
      corzer=corzer/(itermx-iternc)
      if(ish.ge.9)write(ifch,*)'c_0:',corzer

c     calculate corfct_1
c     ------------------
      corone=0.0
      do i=iternc+1,itermx-1
      corone=corone+dev(i)*dev(i+1)
      enddo
      corone=corone/(itermx-iternc-1)

c     calculate initial autocorrelation time
c     -----------------------------------------
      if(corone.gt.1.e-30.and.corzer.gt.1.e-30)then
      r=alog(corone)-alog(corzer)
      if(ish.ge.9)write(ifch,*)'log rho_1:',r
      taui=(-1.)/r
      else
      taui=0.
      endif
      if(ish.ge.9)write(ifch,*)'tau_init:',taui

c     calculate parametrized autocorrelation time (if necessary)
c     ----------------------------------------------------------
      if(taue.eq.0.0)then
      e=tecm/volu
      b=1.1*(e+0.33)**0.66
      a=13.*(e+0.13)**(-0.65)
      tm=34.*(e+0.65)**(-0.61)
      t=a+b*volu
      taue=max(t,tm)
      endif

c     calculate acceptance rate
c     -------------------------
      xa=nacc
      ya=itermx
      accrat=xa/ya

c     write to data/b/c/d
c     -------------------
       if(iii.eq.0)then
      if(iozevt.gt.0)then
      data(nrevt)=iozero
      else
      data(nrevt)=nrevt
      endif
      datb(nrevt)=taui
      datc(nrevt)=accrat
      datd(nrevt)=taue
       else
      data(nrclu)=nrclu
      datb(nrclu)=taui-taue
      datc(nrclu)=accrat
      datd(nrclu)=avnp
       endif

c          -----------------------------------
           elseif(iii.lt.0.and.iappl.eq.4)then
c          -----------------------------------

      write(cmom,'(i3)')iomom
      write(cen,'(f7.3)')tecm
       if(ioobsv.eq.0)then
      write(cnp,'(f7.3)')ptltot
       else
       do i=1,nspecs
       if(ioobsv.eq.ispecs(i))id=i
       enddo
      write(cnp,'(f7.3)')ptlngc(id)
       endif
      write(cvol,'(f7.3)')volu
      write(clatt,'(i3)')nlattc
      write(cit,'(i5)')itermx
      if(ioobsv.eq.0)then
      write(cobs,'(a)')'all'
      else
      write(cobs,'(i5)')ioobsv
      endif
      write(cnc,'(i5)')iternc
      if(iozevt.eq.0)write(czer,'(i5)')iozero
      if(iozevt.gt.0)write(cdz,'(i5)')iozinc

      x1=1
      x2=nevent

      if(jjj.eq.1)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      if(iozevt.gt.0)then
      write(ifhi,'(a)')       'text 0 0 "xaxis iozero"'
      else
      write(ifhi,'(a)')       'text 0 0 "xaxis event"'
      endif
      write(ifhi,'(a)')       'text 0 0 "yaxis [t]?exp!"'
      write(ifhi,'(a)')       'text 0.05 0.95 "E='//cen//'"'
      write(ifhi,'(a)')       'text 0.2  0.95 "V='//cvol//'"'
      write(ifhi,'(a)')       'text 0.35 0.95 "N?g!='//cnp//'"'
      write(ifhi,'(a)')       'text 0.55 0.95 "observable  '//cobs//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nevent
      write(ifhi,'(2e12.4)')data(j),datb(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nevent
      write(ifhi,'(2e12.4)')data(j),datd(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      elseif(jjj.eq.2)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      if(iozevt.gt.0)then
      write(ifhi,'(a)')       'text 0 0 "xaxis iozero"'
      else
      write(ifhi,'(a)')       'text 0 0 "xaxis event"'
      endif
      write(ifhi,'(a)')       'text 0 0 "yaxis acceptence rate"'
      write(ifhi,'(a)')       'text 0.05 0.95 "iomom= '//cmom//'"'
      write(ifhi,'(a)')       'text 0.2  0.95 "nlattc= '//clatt//'"'
      if(iozevt.eq.0)
     *write(ifhi,'(a)')       'text 0.35 0.95 "iozero= '//czer//'"'
      write(ifhi,'(a)')       'text 0.55 0.95 "itermx= '//cit//'"'
      write(ifhi,'(a)')       'text 0.75 0.95 "iternc= '//cnc//'"'
      if(iozevt.gt.0)
     *write(ifhi,'(a)')       'text 0.35  0.95 "dzero= '//cdz//'"'
      if(iorejz.eq.1)
     *write(ifhi,'(a)')    'text 0.25 0.05 "zeros rejected !"'
      if(ioinco.ge.1)then
      write(ifhi,'(a)')    'text 0.05 0.05 "hot start"'
      else
      write(ifhi,'(a)')    'text 0.05 0.05 "cold start"'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nevent
      write(ifhi,'(2e12.4)')data(j),datc(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      endif

c          -----------------------------------
           elseif(iii.lt.0.and.iappl.eq.1)then
c          -----------------------------------

      if(ioobsv.eq.0)then
      write(cobs,'(a)')'all'
      else
      write(cobs,'(i5)')ioobsv
      endif

      x1=1
      x2=nrclu

      if(jjj.eq.1)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis droplet"'
      write(ifhi,'(a)')       'text 0 0 "yaxis [D][t]?exp!"'
      write(ifhi,'(a)')       'text 0.05 0.91 "[D][t]?exp!=[t]?measured!
     *-[t]?parametrized"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a,a,a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,nrclu
      write(ifhi,'(2e12.4)')data(j),datb(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      elseif(jjj.eq.2)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis droplet"'
      write(ifhi,'(a)')       'text 0 0 "yaxis N?obs!"'
      write(ifhi,'(a)')       'text 0.05 0.95 "observable  '//cobs//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nrclu
      write(ifhi,'(2e12.4)')data(j),datd(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      elseif(jjj.eq.3)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis droplet"'
      write(ifhi,'(a)')       'text 0 0 "yaxis accep. rate"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nrclu
      write(ifhi,'(2e12.4)')data(j),datc(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      endif

c          -----
           endif
c          -----

      ish=ish0
      return
      end

c-------------------------------------------------------------------------
      subroutine xhnbti(iii)
c-------------------------------------------------------------------------
c fills histograms (iii=0) or writes histogram to histo-file (iii<0)
c regarding integrated autocorrelation time and corresponding multiplicity
c and variance
c
c input:
c   requires complete run with application metropolis (iappl=4)
c   iociau=1 necessary
c   iompar (parameter for windowing algorithm by  a.d.sokal) must
c   be set to 3 < c_M < 11
c
c  output:
c   for iii=0 (only valid for iappl=4):
c     data(nrevt): nrevt (event number)              /cdat/
c     datb(nrevt): tau   (calculated int corr time)  /cdat/
c     datc(nrevt): stau  (variance tau)              /cdat/
c     datd(nrevt): avnp  (multiplicity)              /cdat/
c     date(nrevt): sobs  (variance multiplicity)     /cdat/
c     datf(nrevt):       (gc multiplicity)           /cdat/
c   for iii=0 and iosngl>0:
c     writes complete set of histograms (newpage zone 1 3 1
c     openhisto ... closehisto plot0 ... openhisto ... closehisto plot 0)
c     concerning acceptance rate, rejection rate, correlation function
c     for specific event, specified by value of iosngl (=nrevt+1)
c   for iii<0:
c     writes complete histogram (openhisto ... closehisto) to histofile
c       xpar1=1: (data,datb,datc)
c       xpar1=2: (data,datd,date,datf)
c------------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      common/citer/iter,itermx
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common /clatt/nlattc,npmax
      common/cgctot/rmstot,ptltot
      parameter (mspecs=56)
      common/cspecs/nspecs,ispecs(mspecs),aspecs(mspecs),gspecs(mspecs)
      common/cgchg/rmsngc(mspecs),ptlngc(mspecs),chemgc(mspecs),tem
      parameter (nbin=500)
      common/cdat2/data(nbin),datb(nbin),datc(nbin),datd(nbin)
     *,date(nbin),datf(nbin),datg(nbin),dath(nbin)
      common/cdat3/datx(nbin),daty(nbin),datz(nbin),datr(nbin)
     *,dats(nbin)
      real corfct(maxit),dev(maxit)
      character cobs*5,cdz*5,ccuev*5,cmpar*3,ctau*7
      character cmom*5,cnp*7,cen*7,cvol*7,clatt*5,cit*5,cavnp*7
      character cnacc*10,cnrej*10,caver*10,cioz*5,ciom*3,cnlat*5

      if(iociau.ne.1)stop'STOP: iociau=1 required'
      if(iii.eq.0.and.iappl.ne.4)stop'STOP: iappl=4 required'
      if(iii.gt.0)stop'STOP: iii>0 not supported'

      jjj=nint(xpar1)
      id=0

c          ----------------
           if(iii.eq.0)then
c          ----------------

c     mean
c     ----
      xnptot=nptot
      avnp=xnptot/(itermx-iternc)
      if(ish.ge.9)write(ifch,*)'event:',nrevt,'   avnp:',avnp

c     normalization of corfct_i
c     -------------------------
      corzer=0.0
      do i=iternc+1,itermx
      dev(i)=npit(i)-avnp
      if(ish.ge.9)write(ifch,*)'i:',i,'  dev_i:',dev(i)
      corzer=corzer+dev(i)**2
      enddo
      corzer=corzer/(itermx-iternc)
      if(ish.ge.9)write(ifch,*)'c_0:',corzer

c     calculate corfct_i
c     ------------------
      nt=itermx-iternc-1
      do it=1,nt
      corfct(it)=0.0
      do i=iternc+1,itermx-it
      corfct(it)=corfct(it)+dev(i)*dev(i+it)
      enddo
      corfct(it)=corfct(it)/(itermx-iternc-it)
      if(it.le.10.and.ish.ge.9)
     *write(ifch,*)'t:',it,'  c_t:',corfct(it)
      enddo

c     calculate initial autocorrelation time
c     -----------------------------------------
      if(corfct(1).gt.1.e-30.and.corzer.gt.1.e-30)then
      r=alog(corfct(1))-alog(corzer)
      if(ish.ge.9)write(ifch,*)'log rho_1:',r
      taui=(-1.)/r
      else
      taui=0.
      endif
      if(ish.ge.9)write(ifch,*)'tau_init:',taui

c     calculate integrated autocorrelation time
c     -----------------------------------------
      k=1
      mpar=iompar
      tau=taui
      taux=taui
      taum=0.0
c...  initialize
      mcut=0
      if(ish.ge.9)write(ifch,*)'initial tau:',tau,'   c_M:',mpar

        if(corzer.gt.1.e-30)then

5     mcut=mpar*abs(taux)
      tauo=tau
      tau=.5
      do it=1,mcut
      tau=tau+corfct(it)/corzer
      enddo
      taum=taum+tau
      taux=taum/k
      if(ish.ge.9)write(ifch,*)'iteration:',k,'   M:',mcut,'  tau:',tau
      if(mcut.lt.(mpar*tau).or.mcut.gt.(10.*tau))then
      dt=abs(tau-tauo)
      if(k.lt.20.and.dt.gt.0.2)then
      k=k+1
      goto5
      endif
      endif
      mcut=mpar*abs(taux)
      if(ish.ge.9)write(ifch,*)'tau_mean:',taux,'   M:',mcut
      tau=0.5
      do it=1,mcut
      tau=tau+corfct(it)/corzer
      enddo

       endif

      vtau=(2.*mcut+1.)*2./(itermx-iternc)*tau**2
      stau=0.0
      if(vtau.ge.0.0)stau=sqrt(vtau)
      if(ish.ge.9)
     *write(ifch,*)'tau_int:',tau,'   var:',vtau,'   sig:',stau

c     calculate variance of observable
c     --------------------------------
      vobs=2.*tau*corzer/(itermx-iternc)
      sobs=0.0
      if(vobs.ge.0.0)sobs=sqrt(vobs)

c     write to data-f
c     ---------------
       if(ioobsv.eq.0)then
      datf(nrevt)=ptltot
       else
      do j=1,np
      if(ioobsv.eq.ispecs(j))id=j
      enddo
      datf(nrevt)=ptlngc(id)
       endif
      datb(nrevt)=tau
      datc(nrevt)=stau
      date(nrevt)=sobs
      datd(nrevt)=avnp
      if(iozevt.gt.0)then
      data(nrevt)=iozero
      else
      data(nrevt)=nrevt
      endif

c          -------------------------
           if(iosngl.eq.nrevt+1)then
c          -------------------------

      nb=itermx/iterpl
      if(nb.gt.nbin)nb=nbin

      datx(1)=iterpl/2
      daty(1)=naccit(1)
      datz(1)=1-naccit(1)
      if(iterpl.ge.2)then
      do j=1,iterpl-1
      daty(1)=daty(1)+naccit(1+j)
      datz(1)=datz(1)+1-naccit(1+j)
      enddo
      endif
      datr(1)=daty(1)/iterpl
      dats(1)=datz(1)/iterpl
      do i=2,nb
      datx(i)=datx(i-1)+iterpl
      daty(i)=daty(i-1)
      datz(i)=datz(i-1)
      do j=1,iterpl
      daty(i)=daty(i)+naccit((i-1)*iterpl+j)
      datz(i)=datz(i)+1-naccit((i-1)*iterpl+j)
      enddo
      datr(i)=daty(i)/i/iterpl
      dats(i)=datz(i)/i/iterpl
      enddo
      b=nacc
      c=itermx
      avrate=b/c
      write(cnacc,'(i6)')nacc
      write(cnrej,'(i6)')nrej
      write(caver,'(f5.3)')avrate
      write(cioz,'(i5)')iozero
      write(ciom,'(i3)')iomom
      write(cnlat,'(i5)')nlattc
      x1=datx(1)
      x2=datx(nb)

      write(ifhi,'(a)')       'newpage zone 1 3 1 openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis iterations"'
      write(ifhi,'(a)')       'text 0 0 "yaxis acceptence rate"'
      write(ifhi,'(a)')       'text 0.6 0.5 "accepted '//cnacc//'"'
      write(ifhi,'(a)')       'text 0.6 0.4 "rejected  '//cnrej//'"'
      write(ifhi,'(a)')       'text 0.6 0.3 "aver. rate  '//caver//'"'
      write(ifhi,'(a)')       'text 0.4 0.5 "nlattc='//cnlat//'"'
      write(ifhi,'(a)')       'text 0.4 0.4 "iozero='//cioz//'"'
      write(ifhi,'(a)')       'text 0.4 0.3 "iomom='//ciom//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nb
      write(ifhi,'(2e12.4)')datx(j),datr(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 2'
      do j=1,nb
      write(ifhi,'(2e12.4)')datx(j),dats(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

      m=min(mcut,500)
      do i=1,m
      datg(i)=i
      dath(i)=1000.
      if(corzer.gt.1.e-30)dath(i)=corfct(i)/corzer
      enddo
      write(ccuev,'(i5)')nrevt+1
      write(cmpar,'(i3)')mpar
      write(ctau,'(i7)')tau
      x1=1.
      x2=m

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis t"'
      write(ifhi,'(a)')       'text 0 0 "yaxis correl. func."'
      write(ifhi,'(a)')       'text 0.8 0.95 "event '//ccuev//'"'
      write(ifhi,'(a)')'text 0.05 0.95  "window parameter= '//cmpar//'"'
      write(ifhi,'(a)')       'text 0.35 0.95  "tau= '//ctau//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a,a,a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 2'
      do j=1,m
      write(ifhi,'(2e12.4)')datg(j),dath(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'

c          -----
           endif
c          -----

c          --------------------
           elseif(iii.lt.0)then
c          --------------------

      write(cmom,'(i3)')iomom
       if(ioobsv.eq.0)then
      write(cnp,'(f7.3)')ptltot
       else
      do j=1,np
      if(ioobsv.eq.ispecs(j))id=j
      enddo
      write(cnp,'(f7.3)')ptlngc(id)
       endif
      write(cen,'(f7.3)')tecm
      write(cvol,'(f7.3)')volu
      write(clatt,'(i3)')nlattc
      write(cit,'(i5)')itermx
      write(cavnp,'(f7.3)')avnp
      if(iozevt.gt.0)
     *write(cdz,'(i5)')iozinc
      write(cmpar,'(i3)')mpar
      if(ioobsv.eq.0)then
      write(cobs,'(a)')'all'
      else
      write(cobs,'(i5)')ioobsv
      endif

      x1=data(1)
      x2=data(nevent)

      if(jjj.eq.1)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp pnt xmod lin ymod lin'
      if(iozevt.gt.0)then
      write(ifhi,'(a)')       'text 0 0 "xaxis iozero"'
      else
      write(ifhi,'(a)')       'text 0 0 "xaxis event"'
      endif
      write(ifhi,'(a)')       'text 0 0 "yaxis [t]?int!"'
      write(ifhi,'(a)')'text 0.05 0.95  "window parameter '//cmpar//'"'
      if(iozevt.gt.0)
     *write(ifhi,'(a)')       'text 0.8  0.95 "dzero= '//cdz//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'array 3'
      do j=1,nevent
      write(ifhi,'(3e12.4)')data(j),datb(j),datc(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      elseif(jjj.eq.2)then

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp pnt xmod lin ymod lin'
      if(iozevt.gt.0)then
      write(ifhi,'(a)')       'text 0 0 "xaxis iozero"'
      else
      write(ifhi,'(a)')       'text 0 0 "xaxis event"'
      endif
      write(ifhi,'(a)')       'text 0 0 "yaxis multiplicity"'
      write(ifhi,'(a)')       'text 0.05 0.95 "E='//cen//'"'
      write(ifhi,'(a)')       'text 0.2 0.95 "V='//cvol//'"'
      write(ifhi,'(a)')       'text 0.35 0.95 "N?g!='//cnp//'"'
      write(ifhi,'(a)')       'text 0.55 0.95 "observable  '//cobs//'"'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a,a,a)')'yrange',' auto',' auto'
      write(ifhi,'(a)')       'array 3'
      do j=1,nevent
      write(ifhi,'(3e12.4)')data(j),datd(j),date(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto   plot 0-'


      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lda xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      do j=1,nevent
      write(ifhi,'(2e12.4)')data(j),datf(j)
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto'

      endif

c          -----
           endif
c          -----

      return
      end

