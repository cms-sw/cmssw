c----------------------------------------------------------------------
      subroutine ahydro
c----------------------------------------------------------------------
      include 'epos.inc'
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/cranphi/ranphi
      ntry=0
 1    ntry=ntry+1
      if(ntry.eq.100)stop'in ahydro: infine loop (080719)   '
      nevt=0
      nptl=0
      rmproj=1.19*maproj**(1./3.)-1.61*maproj**(-1./3.)+fctrmx*.54
      rmtarg=1.19*matarg**(1./3.)-1.61*matarg**(-1./3.)+fctrmx*.54
      b1=bminim
      b2=min(rmproj+rmtarg,bmaxim)
      bimevt=sqrt(b1**2+(b2**2-b1**2)*rangen())
      phievt=0
      ranphi=0
      call InitializeHyperbola !does not affect results but necessary
      call HydroFO(ier)
      if(ier.eq.1)goto1
      do n=1,nptl
        iorptl(n)=0
        jorptl(n)=0
        istptl(n)=0
        ifrptl(1,n)=0
        ifrptl(2,n)=0
        tivptl(1,n)=xorptl(4,n)
        call idtau(idptl(n),pptl(4,n),pptl(5,n),taugm)
        r=rangen()
        tivptl(2,n)=tivptl(1,n)+taugm*(-alog(r))
        radptl(n)=0.
        dezptl(n)=0.
        itsptl(n)=0
        rinptl(n)=0
      enddo
      end

c----------------------------------------------------------------------
      subroutine amicro(iret)
c----------------------------------------------------------------------
c  microcanonical decay of cluster specified via keu...ket, tecm, volu
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      double precision seedp
      data ncntmi/0/
      save ncntmi
      call utpri('amicro',ish,ishini,4)
      ncntmi=ncntmi+1

      if(ncntmi.eq.1)then
        call ranfgt(seedp)      !not to change the seed ...
        if(hydt.ne.'---')then
          call HydroTable2(0)
          call DefineParticles
        else
          call ManiParticles(93,0)
        endif
        call ranfst(seedp)      ! ... after this initialization
      endif

      iret=0
      nevt=0
      nptl=0
c      dez=0.5e-4
      call InitializeHyperbola !does not affect results but necessary

c  50  continue
      call GraCan
      energ=0
      do i=1,np
        energ=energ+pcm(4,i)
      enddo
      !print*,'+++++',energ,tecm,tecm/volu
ccc      if(abs(energ-tecm).gt.0.1) goto50   !uncomment for energy conservation

      do n=1,np
        nptl=nptl+1
        if(nptl.gt.mxptl)call utstop('StaHadShort: mxptl too small&',
     +sizeof('StaHadShort: mxptl too small&'))
        idptl(nptl)=ident(n)
        do j=1,4
          pptl(j,nptl)=pcm(j,n)
          xorptl(j,nptl)=0
        enddo
        pptl(5,nptl)=amass(n)
        ityptl(nptl)=19
      enddo

      do n=1,nptl
        iorptl(n)=0
        jorptl(n)=0
        istptl(n)=0
        ifrptl(1,n)=0
        ifrptl(2,n)=0
        tivptl(1,n)=0
        call idtau(idptl(n),pptl(4,n),pptl(5,n),taugm)
        r=rangen()
        tivptl(2,n)=tivptl(1,n)+taugm*(-alog(r))
        radptl(n)=0.
        itsptl(n)=0
        rinptl(n)=-9999
      enddo

      call utprix('amicro',ish,ishini,4)
      return
      end

c-----------------------------------------------------------------------
      subroutine GraCan
c-----------------------------------------------------------------------
      ! decays cluster specified via tfo and volu using
      !    hadron gas (ioclude=4) or recombination (ioclude=5)
      ! pcm(5,n) is the usual energy (ioclude=4)
      ! or the effective energy, based on quark mass sum (ioclude=5)
      !        so for the flow boost use this energy
      !----------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/copt/istat
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      parameter (mspez=54)
      common/cspez1/nspez,ispez(mspez),aspez(2,mspez),gspez(mspez)
      common/cspez2/kspez,jspez(mspez)
      common/cspez3/fuga(mspez)
      common/cspez4/ffstat(2,0:mspez+2) /ctfo/tfo
      real u(3)
      io3=ioclude-3
      if(io3.lt.1.or.io3.gt.2)stop'in GraCan: wrong ioclude (140808) '
      yie= volu * ffstat(io3,nspez) * ffstat(io3,mspez+2)
      np=yie
      if(rangen().le.yie-np)np=np+1
      do n=1,np
        r=rangen()*yie
        i=0
        do while(yie*ffstat(io3,i)/ffstat(io3,nspez).lt.r)
         i=i+1
        enddo
        ident(n)=ispez(i)
        amass(n)=aspez(1,i)
        fug=fuga(i)
        js=jspez(i)
        x=RanBoseFermi(aspez(io3,i),fug,js,tfo,istat)
        p=x*tfo
        e=sqrt(amass(n)**2+p**2)
        ex=sqrt(aspez(io3,i)**2+p**2)
        u(3)=2.*rangen()-1.
        phi=2.*pi*rangen()
        u(1)=sqrt(1.-u(3)**2)*cos(phi)
        u(2)=sqrt(1.-u(3)**2)*sin(phi)
        pcm(1,n)=p*u(1)
        pcm(2,n)=p*u(2)
        pcm(3,n)=p*u(3)
        pcm(4,n)=e
        pcm(5,n)=ex
      enddo
      end

c----------------------------------------------------------------------
      subroutine HydroTable2(ichk)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      parameter (mspez=54)
      common/cspez1/nspez,ispez(mspez),aspez(2,mspez),gspez(mspez)
      common/cspez2/kspez,jspez(mspez) /cstep/netastep /crapi/delrapi
      common/ctempcrit/tempcrit,epscrit
      character tabname*550
      
      if(hydt.eq.'---')stop'in HydroTable2. no table found (150808) '
      write(tabname,'(a)')fnnx(1:nfnnx)//'epos.ini'//hydt//' '
      call HydroTable(ichk,tabname)
      epscri(3)=epscrit
      kspez=1
      nspez=54
      end

c------------------------------------------------------------------------------
      subroutine HydroTable(ichk,tabname) ! 13 Aug 08 (same as gyx.f except inc)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      character txt*40, tabname*550
      common/ctempcrit/tempcrit,epscrit/ctfo/tfo
      nchar=index(tabname,' ')-1
      write(*,'(a,$)')
     * ' reading table '//tabname(1:nchar)//' ...'
      open(unit=3,file=tabname(1:nchar),status='old'
     * ,err=99)
      read(3,'(a)')txt
      read(3,*)maprojx,matargx,engyx,epscrit,tempcrit
      tfo=tempcrit
      if(ichk.eq.1)then
       if(maprojx.ne.maproj)stop'HydroTable: maprojx.ne.maproj.       '
       if(matargx.ne.matarg)stop'HydroTable: matargx.ne.matarg.       '
       if(engyx.ne.engy)stop'HydroTable: engyx.ne.engy.       '
      endif
      read(3,*)ncenthy,netahy,ntauhy,nphihy,nradhy
      if(ncenthx.lt.ncenthy)stop'HydroTable: ncenthx too small.   '
      if(netahx.lt.netahy)  stop'HydroTable: netahx too small.   '
      if(ntauhx.lt.ntauhy)  stop'HydroTable: ntauhx too small.   '
      if(nphihx.lt.nphihy)  stop'HydroTable: nphihx too small.   '
      if(nradhx.lt.nradhy)  stop'HydroTable: nradhx too small.   '
      read(3,*)(ntauhoc(ncent),ncent=1,ncenthy)
      read(3,*)(centhy(ncent),ncent=1,ncenthy)
     *         ,(etahy(neta),  neta=  1,netahy)
     *         ,(phihy(nphi),  nphi=  1,nphihy)
     *         ,(radhy(nrad),  nrad=  1,nradhy)
      read(3,*)((tauhoc(ncent,ntau),ncent=1,ncenthy), ntau=1,ntauhy)
      read(3,*)((((epsii(ncent,neta,nphi,nrad),ncent=1,ncenthy)
     *          ,neta=1,netahy),nphi=1,nphihy),nrad=1,nradhy)
      read(3,*)meos,meosmu
      read(3,*)((eos(i,m),i=1,3),m=1,meos)
     *          ,((eosmu(i,m),i=1,17),m=1,meosmu)
      read(3,*)mcenty
      read(3,*)(bcenty(m),m=1,mcenty),(pcenty(m),m=1,mcenty)
      read(3,*)nbimp
      read(3,*)(bimpar(1,m),m=1,nbimp),(bimpar(2,m),m=1,nbimp)
      read(3,*)((((rar(ncent,neta,ntau,nphi),ncent=1,ncenthy)
     *          ,neta=1,netahy),ntau=1,ntauhy),nphi=1,nphihy)
      read(3,*)(((((var(i,ncent,neta,ntau,nphi),i=1,3),ncent=1,ncenthy)
     *           ,neta=1,netahy),ntau=1,ntauhy),nphi=1,nphihy)
      close(3)
      do ntau=1,ntauhx
        tauhy(ntau)=tauhoc(1,ntau)
      enddo
      do ncent=2,ncenthy
        do ntau=1,ntauhoc(ncent)
          if(abs(tauhoc(ncent,ntau)-tauhy(ntau)).gt.1e-4)
     *     stop'in HydroTable: different tau grids.       '
        enddo
      enddo
      print*,' done'
      return
   99 print*,'HydroTable: error opening hydro table'
      print*,'      file=',tabname(1:nchar)
      stop'070817'
      end

c----------------------------------------------------------------------------------
      subroutine SymmTab(ncent1,ncent2)   ! 13 Aug 08 (same as gyx.f except inc)
c----------------------------------------------------------------------------------
      include 'epos.inchy'
      common/cderivs/
     .  ddrdtau(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
     . ,ddrdphi(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
     . ,ddrdeta(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
      common/cgmma/gaa(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)

                       !..............................................
      write(*,'(a,$)')' making symmetric tables ...'
                       !''''''''''''''''''''''''''''''''''''''''''''''

      do ncent=ncent1,ncent2
       do meta=1-netahy,netahy-1
        do ntau=1,ntauhx
         do nphi=1,nphihy
           waa(ncent,meta,ntau,nphi)=0
           raa(ncent,meta,ntau,nphi)=0
           vaa(1,ncent,meta,ntau,nphi)=0
           vaa(2,ncent,meta,ntau,nphi)=0
           vaa(3,ncent,meta,ntau,nphi)=0
           ddrdphi(ncent,meta,ntau,nphi)=0
           ddrdtau(ncent,meta,ntau,nphi)=0
           ddrdeta(ncent,meta,ntau,nphi)=0
           gaa(ncent,meta,ntau,nphi)=0
         enddo
        enddo
       enddo
      enddo

      do ncent=ncent1,ncent2
      ntauho=ntauhoc(ncent)

      do neta=1,netahy
       meta=neta-1
       zetahy(meta)=etahy(neta)
       zetahy(-meta)=-etahy(neta)
      enddo
      if(mod(nphihy,2).ne.1)stop'in Symm (2103200820)   '
      do neta=1,netahy
        meta=neta-1
        do ntau=1,ntauho
          do nphi=1,nphihy
           !-------------------------
           !  symmetry: eta -> -eta ;  x -> -x ;  y -> y
           !    or   eta -> -eta ;  r -> r ;  phi -> pi-phi
           !    vtg -> -vtg
           !    v3 -> -v3
           !    drdphi -> -drdphi
           !    drdeta -> -drdeta
           !-------------------------
           nphimed=nphihy/2+1
           if(nphi.le.nphimed)then
            jphi=nphimed+1-nphi
           else
            jphi=nphihy-(nphi-nphimed)
           endif
           if(meta.ne.0)then
           raa(  ncent, meta,ntau,nphi)= rar(  ncent,neta,ntau,nphi)
           vaa(1,ncent, meta,ntau,nphi)= var(1,ncent,neta,ntau,nphi)
           vaa(2,ncent, meta,ntau,nphi)= var(2,ncent,neta,ntau,nphi)
           vaa(3,ncent, meta,ntau,nphi)= var(3,ncent,neta,ntau,nphi)
           raa(  ncent,-meta,ntau,nphi)= rar(  ncent,neta,ntau,jphi)
           vaa(1,ncent,-meta,ntau,nphi)= var(1,ncent,neta,ntau,jphi)
           vaa(2,ncent,-meta,ntau,nphi)=-var(2,ncent,neta,ntau,jphi)
           vaa(3,ncent,-meta,ntau,nphi)=-var(3,ncent,neta,ntau,jphi)
           else
           raa(ncent, meta,ntau,nphi)=
     .     0.5*(rar(ncent,neta,ntau,nphi)+rar(ncent,neta,ntau,jphi))
           vaa(1,ncent,meta,ntau,nphi)=
     .     0.5*(var(1,ncent,neta,ntau,nphi)+var(1,ncent,neta,ntau,jphi))
           vaa(2,ncent, meta,ntau,nphi)=
     .     0.5*(var(2,ncent,neta,ntau,nphi)-var(2,ncent,neta,ntau,jphi))
           vaa(3,ncent, meta,ntau,nphi)=
     .     0.5*(var(3,ncent,neta,ntau,nphi)-var(3,ncent,neta,ntau,jphi))
           endif
          enddo
        enddo
      enddo

      do meta=1-netahy,netahy-1
        do nphi=1,nphihy
            n=ntauho
            do while(raa(ncent,meta,n,nphi).eq.0.0.and.n.gt.2)
              n=n-1
            enddo
            n=n+1
            n=min(n,ntauho)
            ntauhec(ncent,meta,nphi)=n
        enddo
      enddo

      dphi=phihy(2)-phihy(1)
      dtau=tauhy(2)-tauhy(1)
      deta=etahy(2)-etahy(1)
      do meta=1-netahy,netahy-1
        mem=meta-1
        mem=max(mem,1-netahy)
        mep=meta+1
        mep=min(mep,netahy-1)
        do nphi=1,nphihy
          npp=nphi+1
          npp=min(npp,nphihy)
          npm=nphi-1
          npm=max(npm,1)
          do ntau=1,ntauhec(ncent,meta,nphi)
            ntm=ntau-1
            ntm=max(ntm,1)
            ntp=ntau+1
            ntp=min(ntp,ntauhec(ncent,meta,nphi))
            ddrdphi(ncent,meta,ntau,nphi)=(raa(ncent,meta,ntau,npp )
     .              -raa(ncent,meta,ntau,npm )) / ((npp-npm)*dphi)
            ddrdtau(ncent,meta,ntau,nphi)=(raa(ncent,meta,ntp ,nphi)
     .              -raa(ncent,meta,ntm ,nphi)) / ((ntp-ntm)*dtau)
            ddrdeta(ncent,meta,ntau,nphi)=(raa(ncent,mep ,ntau,nphi)
     .              -raa(ncent,mem ,ntau,nphi)) / ((mep-mem)*deta)
          enddo
        enddo
      enddo

      do meta=1-netahy,netahy-1
        do nphi=1,nphihy
          do ntau=1,ntauhec(ncent,meta,nphi)
            vv=sqrt(vaa(1,ncent,meta,ntau,nphi)**2
     .             +vaa(2,ncent, meta,ntau,nphi)**2
     .             +vaa(3,ncent, meta,ntau,nphi)**2)
            gaa(ncent,meta,ntau,nphi)=1./sqrt((1-vv)*(1+vv))
          enddo
        enddo
      enddo

      do meta=1-netahy,netahy-1
       do nphi=1,nphihy
        phi=phihy(nphi)
        do ntau=1,ntauhec(ncent,meta,nphi)
          tau=tauhy(ntau)
          rad=raa(ncent,meta,ntau,nphi)
          vrd=vaa(1,ncent,meta,ntau,nphi)
          vtg=vaa(2,ncent,meta,ntau,nphi)
          v3=vaa(3,ncent,meta,ntau,nphi)
          vv=sqrt(vrd**2+vtg**2+v3**2)
          gm=1./sqrt((1-vv)*(1+vv))
          dVs=      - ddrdtau(ncent,meta,ntau,nphi) * tau * rad *gm
     .              +                    rad * tau       *gm*vrd
     .              + ddrdphi(ncent,meta,ntau,nphi) * tau       *gm*vtg
     .              - ddrdeta(ncent,meta,ntau,nphi) * rad       *gm*v3
          waa(ncent,meta,ntau,nphi)=abs(dVs)
        enddo !tau
       enddo !phi
      enddo !neta

      enddo !ncent

      print*,' done'

      end

c--------------------------------------------------------------------------------------
      subroutine DefineParticles   ! 14 Aug 08 (same as gyx.f except inc)
c--------------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/copt/istat
      parameter (mspez=54)
      common/cspez1/nspez,ispez(mspez),aspez(2,mspez),gspez(mspez)
      common/cspez2/kspez,jspez(mspez)
      common/cspez3/fuga(mspez)
      common/cspez4/ffstat(2,0:mspez+2) /ctfo/tfo
      parameter (nlag=15)
      real xlag(nlag),wlag(nlag)
      parameter (klax=5)
      common/cspez7/klass(mspez)
      data klass/
     *  0, 1, 1, 2, 2, 4*0
     *, 9*0
     *, 8*0
     *, 8*0
     *,10*0
     *,10*0/
      data ispez/
     *   110,  120, -120,  130, -130,  230, -230,  220,  330
     *,  111,  121, -121,  131, -131,  231, -231,  221,  331
     *, 1120,-1120, 1220,-1220, 1130,-1130, 2130,-2130
     *, 1230,-1230, 2230,-2230, 1330,-1330, 2330,-2330
     *, 1111,-1111, 1121,-1121, 1221,-1221, 2221,-2221, 1131,-1131
     *, 1231,-1231, 2231,-2231, 1331,-1331, 2331,-2331, 3331,-3331 /
      data jspez/
     *  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 , 36*1/
      data gspez/
     *  9*1.
     *, 9*3.
     *, 8*2.
     *, 8*2.
     *,10*4.
     *,10*4./
      data (aspez(1,m),m=1,mspez)/
     *    1.349766E-01   !pi               0
     *,2* 1.3957018E-01  !pi               +
     *,2* 4.93677E-01    !K                +
     *,2* 4.97648E-01    !K                0
     *,   5.4751E-01     !eta              0
     *,   9.5778E-01     !eta'(958)        0
     *,3* 7.755E-01      !rho(770)         0
     *,2* 8.9166E-01     !K*(892)          +
     *,2* 8.9600E-01     !K*(892)          0
     *,   7.8265E-01     !omega(782)       0
     *,   1.019460E+00   !phi(1020)        0
     *,2* 9.3827203E-01  !p                +
     *,2* 9.3956536E-01  !n                0
     *,2* 1.18937E+00    !Sigma            +
     *,2* 1.115683E+00   !Lambda           0
     *,2* 1.192642E+00   !Sigma            0
     *,2* 1.197449E+00   !Sigma            -
     *,2* 1.31483E+00    !Xi               0
     *,2* 1.32131E+00    !Xi               -
     *,8* 1.2320E+00     !Delta(1232)      -
     *,2* 1.3828E+00     !Sigma(1385)      +
     *,2* 1.3837E+00     !Sigma(1385)      0
     *,2* 1.3872E+00     !Sigma(1385)      -
     *,2* 1.53180E+00    !Xi(1530)         0
     *,2* 1.5350E+00     !Xi(1530)         -
     *,2* 1.67245E+00    !Omega            -
     */
c      data (aspez(1,m),m=1,mspez)/
c     *     0.13496, 2* 0.13957, 2* 0.49367, 2* 0.49767, 0.54880,0.95760
c     *, 3* 0.77000, 2* 0.88810, 2* 0.89220,    0.78260, 1.01960
c     *, 2* 0.93828, 2* 0.93957, 2* 1.18940, 2* 1.11560
c     *, 2* 1.19250, 2* 1.19740, 2* 1.31490, 2* 1.32130
c     *, 8* 1.23200, 2* 1.38230, 2* 1.38200, 2* 1.38750
c     *, 2* 1.53180, 2* 1.53500, 2* 1.67220   /
      parameter (nquark=3)
      real amaq(0:nquark)
      data amaq /0., 0.337, 0.337, 0.486/
      do m=1,mspez
        id=ispez(m)
        call idflav(id,i1,i2,i3,jspin,index)
        amx=amaq(abs(i1))+amaq(abs(i2))+amaq(abs(i3))
        aspez(2,m)=amx
      enddo

      write(*,'(a,$)')' DefineParticles  ...'

      istat=1  !0=Boltzmann, 1=Bose/Fermi
      pi=3.1415927
      hbar=0.197327

      call gaulag(xlag,wlag,nlag,0.)
      do n=1,nlag
      wlag(n)=wlag(n)*exp(xlag(n))
      enddo

      !write(*,'(4x,a,$)')'fuga:'
      do m=kspez,nspez
        id=ispez(m)
        chem=0
        if(meosmu.gt.0)then
         ihi=idxHiranoTable(id)
         if(ihi.gt.0)then
          k=1
          do while(eosmu(1,k).gt.tfo)
           k=k+1
          enddo
          f=(tfo-eosmu(1,k))/(eosmu(1,k-1)-eosmu(1,k))
          chem=eosmu(ihi,k)*(1-f) + eosmu(ihi,k-1)*f
         endif
        endif
        fuga(m)=exp(chem/tfo)
        !if(m.lt.6)write(*,'(i5,a,f6.2,$)')id,':',fuga(m)
      enddo

      !write(*,'(/25x,a,$)')'yie:'
      facphase=1./(2*pi*hbar)**3
      do m=0,kspez-1
        ffstat(1,m)=0
        ffstat(2,m)=0
      enddo
       eesum=0
       hhsum=0
       do m=kspez,nspez
        id=ispez(m)
        am=aspez(1,m)
        amx=aspez(2,m)
        esum=0
        fsum=0
        gsum=0
        hsum=0
        do n=1,nlag
          x=xlag(n)  ! p/tfo
          e=sqrt(am**2+x**2*tfo**2)
          w=exp(-sqrt(am**2/tfo**2+x**2))  * fuga(m)
          fsum=fsum+wlag(n)*x**2*w /(1+istat*jspez(m)*w)
          esum=esum+wlag(n)*x**2*w /(1+istat*jspez(m)*w) *e
          wx=exp(-sqrt(amx**2/tfo**2+x**2)) !???? * fuga(m)
          gsum=gsum+wlag(n)*x**2*wx /(1+istat*jspez(m)*wx)
          hsum=hsum+wlag(n)*x**2*wx /(1+istat*jspez(m)*wx) *e
        enddo
        esum=esum * facphase * gspez(m) * 4 * pi * tfo**2  * tfo
        fsum=fsum * facphase * gspez(m) * 4 * pi * tfo**2  * tfo
        gsum=gsum * facphase * gspez(m) * 4 * pi * tfo**2  * tfo
        hsum=hsum * facphase * gspez(m) * 4 * pi * tfo**2  * tfo
        ffstat(1,m)=ffstat(1,m-1)+fsum
        ffstat(2,m)=ffstat(2,m-1)+gsum
        eesum=eesum+esum
        hhsum=hhsum+hsum
        !if(m.lt.6)write(*,'(i5,a,f8.5,$)')id,':',fsum
      enddo
      ffstat(1,mspez+1)=eesum
      ffstat(1,mspez+2)=1
      ffstat(2,mspez+1)=hhsum
      ffstat(2,mspez+2)=eesum/hhsum

      print*,'  E/V=',eesum,'  Ereco/V=',hhsum,'    done'
      !do n=1,nlag
      !print*,n,xlag(n),wlag(n)
      !enddo
      end

c--------------------------------------------------------------------------------------
      subroutine RestFrameFO(nsimu,bimp
     . ,jcent,ier,iprint,ianl,isto) ! 14 Aug 08 (same as gyx.f except inc)
c--------------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/copt/istat
      parameter (mspez=54,klax=5)
      common/cspez1/nspez,ispez(mspez),aspez(2,mspez),gspez(mspez)
      common/cspez2/kspez,jspez(mspez)  /cho/netaho
      common/cpro/probdi(20,100),dprob,mxprob
      common/crap/nraphi,jfac  /cstep/netastep
      common/cspez3/fuga(mspez)
      common/cspez4/ffstat(2,0:mspez+2)  /ctfo/tfo
      real u(3),q(4),waii(2),wbii(2),wcii(2)
      real wrii(2),wwii(2),wxii(2),wyii(2),wzii(2)
      parameter(numiv=100,rapmax=5,rapmin=-rapmax,ptvmax=2)
      common/cana1/rapeta(klax,-netahx+1:netahx-1,numiv)
      common/cana1b/rapar(klax,numiv),v2rapar(klax,numiv)
      common/cana1c/phaar(klax,numiv,numiv)
      common/cana1d/sapar(klax,numiv),v2sapar(klax,numiv)
      common/cana1e/ptvar(klax,numiv),ptwar(klax,numiv)
      common/cspez7/klass(mspez)
      common/cderivs/
     .  ddrdtau(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
     . ,ddrdphi(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
     . ,ddrdeta(ncenthx,-netahx+1:netahx-1,ntauhx,nphihx)
      ier=1
      if(bimp.gt.centhy(ncenthy)+0.5)return
      ier=0
      pi=3.1415927
      hbar=0.197327
      delrax= 2*rapmax   / numiv
      delptv= ptvmax   / numiv
      do i=1,100
       probdi(jcent,i)=0
      enddo
      mxprob=jcent
      dprob=0.1
      ncent=2
      do while(ncent.lt.ncenthy.and.centhy(ncent).lt.bimp)
        ncent=ncent+1
      enddo
      n1=ncent-1
      n2=ncent
      frac=(bimp-centhy(n1))/(centhy(n2)-centhy(n1))
      g1=1-frac
      g2=frac
      ntauho=max(ntauhoc(n1),ntauhoc(n2))

      if(iprint.eq.1)then
       print*,'++++++++',ncent, bimp, centhy(n1),centhy(n2),g1,g2
       print*,'++++++++',ntauho,netaho,jfac*netastep,nsimu
       write(*,'(a,$)')' Rest Frame freeze out ...    '
      endif

      dphi=phihy(2)-phihy(1)
      dtau=tauhy(2)-tauhy(1)
      deta=etahy(2)-etahy(1)
      dleta=(etahy(2)-etahy(1))/jfac
      dall=dphi*dtau*dleta*0.125
      do nphi=1,nphihy
      if(iprint.eq.1.and.
     .mod(nphihy+1-nphi,10).eq.0)write(*,'(i3,$)')(nphihy+1-nphi)/10
      phi=phihy(nphi)
      fphi=2
      if(nphi.eq.1.or.nphi.eq.nphihy)fphi=1
      do ntau=1,ntauho
      tau=tauhy(ntau)
      ftau=2
      if(ntau.eq.1.or.ntau.eq.ntauho)ftau=1
      do meta=-netaho+netastep,netaho,netastep
        do ii=1,2
        if(ii.eq.1)then
          mt=meta-netastep
        else                 !if(ii.eq.2)
          mt=meta
        endif
        wwii(ii)=g1*waa(n1,mt,ntau,nphi)    +g2*waa(n2,mt,ntau,nphi)
        wwii(ii)=max(0.,wwii(ii))
        wxii(ii)=g1*vaa(1,n1,mt,ntau,nphi)  +g2*vaa(1,n2,mt,ntau,nphi)
        wyii(ii)=g1*vaa(2,n1,mt,ntau,nphi)  +g2*vaa(2,n2,mt,ntau,nphi)
        wzii(ii)=g1*vaa(3,n1,mt,ntau,nphi)  +g2*vaa(3,n2,mt,ntau,nphi)
        wrii(ii)=g1*raa(n1,mt,ntau,nphi)    +g2*raa(n2,mt,ntau,nphi)
        waii(ii)=g1*ddrdtau(n1,mt,ntau,nphi)+g2*ddrdtau(n2,mt,ntau,nphi)
        wbii(ii)=g1*ddrdphi(n1,mt,ntau,nphi)+g2*ddrdphi(n2,mt,ntau,nphi)
        wcii(ii)=g1*ddrdeta(n1,mt,ntau,nphi)+g2*ddrdeta(n2,mt,ntau,nphi)
        enddo
        jmax=jfac*netastep
        do j=0,jmax
          f=2
          if(j.eq.0.or.j.eq.jmax)f=1
          fall=fphi*ftau*f
          dVs=wwii(1)+j/float(jmax)*(wwii(2)-wwii(1))
          dVs=dVs*dall*fall
          vr=wxii(1)+j/float(jmax)*(wxii(2)-wxii(1))
          vt=wyii(1)+j/float(jmax)*(wyii(2)-wyii(1))
          v3=wzii(1)+j/float(jmax)*(wzii(2)-wzii(1))
          v1=vr*cos(phi)+vt*sin(phi)
          v2=vr*sin(phi)-vt*cos(phi)
          gmx=max(1e-8, 1.-vr*vr-vt*vt-v3*v3)
          gm=1./sqrt(gmx)
          rad=wrii(1)+j/float(jmax)*(wrii(2)-wrii(1))
          ieta=(meta-netastep)*jfac+j
          eta=ieta*dleta
          finc=10
          volu=abs(dVs)*finc
          io3=ioclude-3
          if(io3.lt.1.or.io3.gt.2)
     .     stop'in RestFrameFO: wrong ioclude (150808) '
          yie= volu * ffstat(io3,nspez) * ffstat(io3,mspez+2)
          do nsim=1,nsimu
            np=yie
            if(rangen().le.yie-np)np=np+1
            if(np.gt.0)then
              do n=1,np
                r=rangen()*yie
                i=0
                do while(yie*ffstat(io3,i)/ffstat(io3,nspez).lt.r)
                 i=i+1
                enddo
                kss=klass(i)
                id=ispez(i)
                am=aspez(1,i)
                fug=fuga(i)
                js=jspez(i)
                x=RanBoseFermi(aspez(io3,i),fug,js,tfo,istat)
                p=x*tfo
                e=sqrt(p**2+aspez(1,i)**2)
                ex=sqrt(p**2+aspez(io3,i)**2)
                !print*,id,e
                u(3)=2.*rangen()-1.
                angle=2.*pi*rangen()
                u(1)=sqrt(1.-u(3)**2)*cos(angle)
                u(2)=sqrt(1.-u(3)**2)*sin(angle)
                q(1)=p*u(1)
                q(2)=p*u(2)
                q(3)=p*u(3)
                q(4)=ex
                call utlob3(-1, v1*gm , v2*gm , v3*gm , gm ,1e0
     .                   , q(1), q(2), q(3), q(4))
                w1=FOFactor(wrii(1),waii(1),wbii(1),wcii(1),tau,phi,q)
                w2=FOFactor(wrii(2),waii(2),wbii(2),wcii(2),tau,phi,q)
                fof=w1+j/float(jmax)*(w2-w1)
                fof=fof*dall*fall
                probab=fof/volu/e
                ij=1+probab/dprob
                if(ij.ge.1.and.ij.le.100)
     .           probdi(jcent,ij)=probdi(jcent,ij)+1
                if(rangen().le.probab)then !accept
                  if(io3.eq.2)q(4)=e
                  rap=eta +  0.5*log((q(4)+q(3))/(q(4)-q(3)))
                  pt2=q(1)**2+q(2)**2
                  ptr=sqrt(pt2)
                  pha=sign(1.,q(2))*acos(q(1)/ptr)
                  if(pha.lt.0.)pha=pha+2*pi
                  if(ianl.eq.1)then !~~~~~~~~~~~~~~~~~~~~~~~~~~
                    pz=sqrt(am**2+ptr**2)*sinh(rap)
                    pz2p=pz/sqrt(pz**2+ptr**2)
                    nrap=1+(rap-rapmin)/delrax
                    npha=1+pha/(2*pi/numiv)
                    sap=0.501
                    if(pz2p.le.-1.)then
                     nsap=0
                    elseif(pz2p.ge.1.)then
                     nsap=numiv+1
                    else
                     sap=0.5*log((1+pz2p)/(1-pz2p))
                     nsap=1+(sap-rapmin)/delrax
                    endif
                    nptv=1+ptr/delptv
                    if(nrap.ge.1.and.nrap.le.numiv.and.kss.gt.0)then
                      rapeta(kss,meta,nrap)=rapeta(kss,meta,nrap)+1
                      rapar(kss,nrap)=rapar(kss,nrap)+1
                      if(pt2.gt.0.)
     .                v2rapar(kss,nrap)=v2rapar(kss,nrap)
     .                 +(q(1)**2-q(2)**2)/pt2
                      if(npha.ge.1.and.npha.le.numiv)then
                        phaar(kss,nrap,npha)=phaar(kss,nrap,npha)+1
                      endif
                    endif
                    if(nsap.ge.1.and.nsap.le.numiv.and.kss.gt.0)then
                      sapar(kss,nsap)=sapar(kss,nsap)+1
                      if(pt2.gt.0.)
     .                v2sapar(kss,nsap)=v2sapar(kss,nsap)
     .                 +(q(1)**2-q(2)**2)/pt2
                    endif
                    if(nptv.ge.1.and.nptv.le.numiv.and.kss.gt.0
     .              .and.abs(rap).le.0.5)then
                      ptvar(kss,nptv)=ptvar(kss,nptv)+1
                    endif
                    if(nptv.ge.1.and.nptv.le.numiv.and.kss.gt.0
     .              .and.abs(sap).le.0.5)then
                      ptwar(kss,nptv)=ptwar(kss,nptv)+1
                    endif
                  endif !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  if(isto.eq.1)
     .            call FOStore(id,ptr,pha,am,rap,rad,phi,tau,eta)
                endif
              enddo !np
            endif !np.gt.0
          enddo !nsim
        enddo !j

      enddo !meta
      enddo !ntau
      enddo !nphi

      if(iprint.eq.1)print*,' done'
      end

c--------------------------------------------------------------------------------------
      subroutine FOStore(id,ptr,pha,am,rap,rad,phi,tau,eta)
c--------------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cranphi/ranphi
      data ncntfoa/0/
      save ncntfoa
      ncntfoa=ncntfoa+1
      if(ncntfoa.eq.1)nptlb=nptl
      phinull=phievt+ranphi
      nptl=nptl+1
      if(nptl.gt.mxptl)
     . call utstop('FOStore: mxptl too small&',
     +sizeof('FOStore: mxptl too small&'))
      idptl(nptl)=id
      pptl(1,nptl)=ptr*cos(pha+phinull)
      pptl(2,nptl)=ptr*sin(pha+phinull)
      pptl(3,nptl)=sqrt(am**2+ptr**2)*sinh(rap)
      pptl(4,nptl)=sqrt(am**2+ptr**2)*cosh(rap)
      pptl(5,nptl)=am
      ityptl(nptl)=60
      xorptl(1,nptl)=rad*cos(phi+phinull)
      xorptl(2,nptl)=rad*sin(phi+phinull)
      xorptl(3,nptl)=tau*sinh(eta)
      xorptl(4,nptl)=tau*cosh(eta)
      end

c--------------------------------------------------------------------------------------
      function FOFactor(wr,wa,wb,wc,tau,phi,q) ! 13 Aug 08 (same as gyx.f)
c--------------------------------------------------------------------------------------
      real q(4)
      rad=wr
      qrd= cos(phi)*q(1)+sin(phi)*q(2)
      qtg= sin(phi)*q(1)-cos(phi)*q(2)
      fof=      - wa * tau * rad *q(4)
     .          +      rad * tau *qrd
     .          + wb * tau       *qtg
     .          - wc * rad       *q(3)
      FOFactor=fof
      end

c---------------------------------------------------------------------------------
      function RanBoseFermi(am,fug,js,tfo,istat) ! 13 Aug 08 (same as gyx.f)
c---------------------------------------------------------------------------------
c   generates randomly x=p/tfo according to Bose/Fermi
c------------------------------------------------------------------------------
  1   x=RanTherm(am/tfo)
      if(istat.eq.0)goto2  !Boltznann
      p=x*tfo
      e=sqrt(p**2+am**2)
      w=exp(-e/tfo)*fug
      if(js.eq.-1)then !bosons
        w0=exp(-am/tfo)*fug
        pacc=(1.-w0)/(1.-w)
      elseif(js.eq.1)then !fermions
        pacc=1./(1.+w)
      else
        stop'in RanBoseFermi: unknown statistics (080726) '
      endif
      !print*,'+++++',am,js,pacc
      if(rangen().gt.pacc)goto1
  2   RanBoseFermi=x
      end

c------------------------------------------------------------------------------
      function RanTherm(a) ! 13 Aug 08 (same as gyx.f)
c------------------------------------------------------------------------------
c   generates a random number according to f(x) ~ x**2*exp(-sqrt(x*2+a**2))
c   in the interval from zero to infinity
c------------------------------------------------------------------------------
      !ntry=0
  1   i=2
      if(rangen().le.a**3/(a**3+3*a**2+6*a+6))i=1
      !ntry=ntry+1
      if(i.eq.1)then
        x=a*rangen()**(1./3.)
        if(rangen().gt.exp(a-sqrt(x**2+a**2)))goto1
      elseif(i.eq.2)then
        !f(z)~a**2+2*a*z+z**2)*exp(-z)  from zero to infty
        r=rangen()
        if(r.lt.a**2/(a**2+2*a+2))then
           z=-log(rangen())
        elseif(r.lt.(a**2+2*a)/(a**2+2*a+2))then
           z=-log(rangen())-log(rangen())
        else
           z=-log(rangen())-log(rangen())-log(rangen())
        endif
        x=a+z
        if(rangen().gt.exp(x-sqrt(x**2+a**2)))goto1
      endif
      RanTherm=x
      !ia=a*100
      !print*,ia,'   ',ntry,'    ',x
      end

c---------------------------------------------------------------------------------
      subroutine DefineRapScale ! 13 Aug 08 (same as gyx.f except inc)
c---------------------------------------------------------------------------------
      include 'epos.inchy'
      common/crap/nraphi,jfac /cho/netaho /cstep/netastep /crapi/delrapi
      deta=etahy(2)-etahy(1)
      write(*,'(a,$)')' DefineRapScale  ...'
      jfac=1
      delrap=deta
      do while(delrap.gt.delrapi)
        jfac=jfac+1
        delrap=deta/jfac
      enddo
      write(*,'(f7.2,i3,f7.2,5x,$)')deta,jfac,delrap
      nraphi=5
      rapmn=-delrap*nraphi
      do while(rapmn.gt.-2.)
        nraphi=nraphi+1
        if(nraphi.gt.nraphx)stop'(1539070608)   '
        rapmn=-delrap*nraphi
      enddo
      netaho=netahy-1
      do while(mod(netaho,netastep).ne.0)
      netaho=netaho-1
      enddo
      !print*,'$$$$$$ nraphi,rapmn,netaho:',nraphi,rapmn,netaho
      print*,' done'
      end

c----------------------------------------------------------------------
      subroutine HydroFO(ier)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr
      parameter (klax=5)
      parameter(numiv=100,rapmax=5,rapmin=-rapmax)
      common/cana1b/rapar(klax,numiv),v2rapar(klax,numiv)
      common/cana1d/sapar(klax,numiv),v2sapar(klax,numiv)
      common/cstep/netastep /crapi/delrapi
      double precision seedp

      data icntcf /0/
      save icntcf
      icntcf=icntcf+1
      if(icntcf.eq.1)then
        call ranfgt(seedp)      !not to change the seed ...
        do n=1,klax
         do nu=1,numiv
          rapar(n,nu)=0
          v2rapar(n,nu)=0
          sapar(n,nu)=0
          v2sapar(n,nu)=0
         enddo
        enddo
        netastep=1
        delrapi=0.2
        call HydroTable2(0)
        call SymmTab(1,ncenthy)
        call DefineParticles
        call DefineRapScale
        call ranfst(seedp)      ! ... after this initialization
      endif
      if(iappl.eq.9)then
        call CheckBimp(ier)
        if(ier.eq.1)return
        call GetNpart
      endif
      call RestFrameFO(1,bimevt,1,ier,0,1,1)
      end

c----------------------------------------------------------------------
      subroutine xEnergy
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mspez=54)
      common/cspez4/ffstat(2,0:mspez+2)
      common/ccsum/eesum,hhsum
      eesum=ffstat(1,mspez+1)
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   energy     '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') ' openhisto htyp pgs xmod lin ymod lin '
      write(ifhi,'(a)') ' array 2'
      write(ifhi,'(2e11.3)')eesum*volu,0.005
      write(ifhi,'(a)') ' endarray closehisto plot 0'
      end

c----------------------------------------------------------------------
      subroutine GetNpart
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      mcent=2
      do while(mcent.lt.mcenty.and.bcenty(mcent).lt.bimevt)
        mcent=mcent+1
      enddo
      m1=mcent-1
      m2=mcent
      hrac=(bimevt-bcenty(m1))/(bcenty(m2)-bcenty(m1))
      h1=1-hrac
      h2=hrac
      npartic=h1*pcenty(m1)+h2*pcenty(m2)
      npjevt=npartic/2
      ntgevt=npartic-npjevt
      !print*,'++++++++'
      !.    ,mcent, bimevt, bcenty(m1),bcenty(m2),h1,h2,npartic
      end

c----------------------------------------------------------------------
      subroutine CheckBimp(ier)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      data icntcb /0/
      save icntcb, wref,bref
      icntcb=icntcb+1
      if(icntcb.eq.1)then
        nmax=0
        wmax=0
        do n=1,nbimp
         if(bimpar(2,n).gt.wmax)then
           wmax=bimpar(2,n)
           nmax=n
         endif
        enddo
        nref=nmax*0.75
        bref=bimpar(1,nref)
        wref=bimpar(2,nref)
      endif
      ier=0
      if(bimevt.lt.bref)then
        q=1
      else
        w=wref/bref*bimevt
        n=2
        do while(n.lt.nbimp.and.bimpar(1,n).lt.bimevt)
          n=n+1
        enddo
        n1=n-1
        n2=n
        frac=(bimevt-bimpar(1,n1))/(bimpar(1,n2)-bimpar(1,n1))
        g1=1-frac
        g2=frac
        wx=g1*bimpar(2,n1)+g2*bimpar(2,n2)
        wx=max(0.,wx)
        q=wx/w
      endif
      if(rangen().gt.q)ier=1
      !if(ier.eq.0)print*,'+++++',bimevt,q

      end

c----------------------------------------------------------------------
      subroutine xCoopFryPt(kss)
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(numiv=100,ptvmax=2,klax=5)
      common/cana1e/ptvar(klax,numiv),ptwar(klax,numiv)
      delptv= ptvmax   / numiv
      pi=3.1415927
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   pt    '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') 'openhisto htyp lin xmod lin ymod log '
      write(ifhi,'(a,f7.2)') 'xrange 0 2 '
      write(ifhi,'(a)') 'txt  "xaxis p?t!"'
      write(ifhi,'(a)') 'txt  "yaxis dn / 2[p] p?t! dp?t!"'
      write(ifhi,'(a)') 'array 2'
      do n=1,numiv
        x=(n-0.5)*delptv
        y=ptvar(kss,n)/float(nevent)/delptv/2./pi/x
        write(ifhi,'(2e14.6)')x,y
      enddo
      write(ifhi,'(a)') 'endarray closehisto plot 0'
c      write(ifhi,'(a)') 'openhisto htyp lin xmod lin ymod lin '
c      write(ifhi,'(a,f7.2)') 'xrange 0 2 '
c      write(ifhi,'(a)') 'array 2'
c      do n=1,numiv
c        x=(n-0.5)*delptv
c        y=0
c        if(ptwar(kss,n).gt.0.)y=ptwar(kss,n)/float(nevent)/delptv/2./pi/x
c        write(ifhi,'(2e14.6)')x,y
c      enddo
c      write(ifhi,'(a)') 'endarray closehisto plot 0'
      end

c----------------------------------------------------------------------
      subroutine xCoopFryRap(kss)
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(numiv=100,rapmax=5,rapmin=-rapmax,klax=5)
      common/cana1b/rapar(klax,numiv),v2rapar(klax,numiv)
      common/cana1d/sapar(klax,numiv),v2sapar(klax,numiv)
      delrax= 2*rapmax   / numiv
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   rapidity distribution     '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
      write(ifhi,'(a)') ' xrange -5 5'
      write(ifhi,'(a)') ' txt  "xaxis y "'
      write(ifhi,'(a)') ' txt  "yaxis dn/dy "'
      write(ifhi,'(a)') ' array 2'
      do n=1,numiv
        x=rapmin+(n-0.5)*delrax
        y=rapar(kss,n)/float(nevent)/delrax
        write(ifhi,'(2e11.3)')x,y
      enddo
      write(ifhi,'(a)') ' endarray closehisto plot 0'
c      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
c      write(ifhi,'(a)') ' xrange -5 5'
c      write(ifhi,'(a)') ' txt  "xaxis  [c] "'
c      write(ifhi,'(a)') ' txt  "yaxis dn/d[c]"'
c      write(ifhi,'(a)') ' array 2'
c      do n=1,numiv
c        x=rapmin+(n-0.5)*delrax
c        y=sapar(kss,n)/float(nevent)/delrax
c        write(ifhi,'(2e11.3)')x,y
c      enddo
c      write(ifhi,'(a)') ' endarray closehisto plot 0'
      end

c----------------------------------------------------------------------
      subroutine xCoopFryV2(kss)
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(numiv=100,rapmax=5,rapmin=-rapmax,klax=5)
      common/cana1b/rapar(klax,numiv),v2rapar(klax,numiv)
      common/cana1d/sapar(klax,numiv),v2sapar(klax,numiv)
      delrax= 2*rapmax   / numiv
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   v2     '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
      write(ifhi,'(a,f7.2)') ' xrange -5 5 yrange 0 auto'
      write(ifhi,'(a)') ' txt  "xaxis y "'
      write(ifhi,'(a)') ' txt  "yaxis v?2!"'
      write(ifhi,'(a)') ' array 2'
      do n=1,numiv
        x=rapmin+(n-0.5)*delrax
        y=0
        if(rapar(kss,n).gt.0.)y=v2rapar(kss,n)/rapar(kss,n)
        write(ifhi,'(2e11.3)')x,y
      enddo
      write(ifhi,'(a)') ' endarray closehisto plot 0'
c      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
c      write(ifhi,'(a,f7.2)') ' xrange -5 5 yrange 0 auto'
c      write(ifhi,'(a)') ' txt  "xaxis  [c]"'
c      write(ifhi,'(a)') ' txt  "yaxis v?2!"'
c      write(ifhi,'(a)') ' array 2'
c      do n=1,numiv
c        x=rapmin+(n-0.5)*delrax
c        y=0
c        if(sapar(kss,n).gt.0.)y=v2sapar(kss,n)/sapar(kss,n)
c        write(ifhi,'(2e11.3)')x,y
c      enddo
c      write(ifhi,'(a)') ' endarray closehisto plot 0'
      end

c----------------------------------------------------------------------
      subroutine DropletDecay(ip,iret)
c----------------------------------------------------------------------
      ! statistical hadronization with imposed flow
      !   (to be distiguished from real hydro flow as in StaHadDF)
      !---------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      parameter(maxp=500)
      common/confg/np,amass(maxp),ident(maxp),pcm(5,maxp),wtxlog,wtlog
      common/citer/iter,itermx
      integer jc(nflav,2)
      double precision p(5),c(5)!,yyrmax
      parameter(maxit=50000)
      common/count/nacc,nrej,naccit(maxit),nptot,npit(maxit)
      dimension uu(4),pe(5),pa(5)
      common/ylong/ylong(maxp)
      common/vradi/vradi(maxp),vtang(maxp),phifop(maxp),radfop(maxp)
     . ,taufop(maxp)
      common/xxxspecsy/ndrop(-4:4,-4:4,-4:4)
      common/cdelzet/delzet,delsgr /cvocell/vocell
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      common/cranphi/ranphi /ctfo/tfo
      double precision seedp
      !data vv2 /0./ nvv2 /0/  vv3 /0./
      !save vv2,nvv2,vv3
      data icntsta /0/
      save icntsta
      icntsta=icntsta+1
      if(ioclude.eq.1)stop'ioclude.eq.1 no longer supported.    '
      if(ioclude.eq.2)stop'ioclude.eq.2 no longer supported.    '

      call utpri('DropletDecay',ish,ishini,4)

      if(ish.ge.3)then
      write(ifch,140)
  140 format(/' ----------------------------------'/
     *        '    droplet decay'/
     *        ' ----------------------------------')
      write(ifch,*)'droplet:'
      call alist('&',
     +sizeof('&'),ip,ip)
      endif
      call ManiParticles(1,1) !store parameters, to be restored at the end

      if(icntsta.eq.1)then
        call ranfgt(seedp)      !not to change the seed ...
        call ManiParticles(92,0)
        !redefine parameters, for this subroutine; for the moment same parameters
        ! for remnants and droplets, this could easily be changed
        call ManiParticles(1,2) !store local parameters
        call ranfst(seedp)      ! ... after this initialization
      else
        call ManiParticles(-1,2) !restore local parameters
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
      !print*,'droplet uds=',keu,ked,kes,'   E=',pptl(5,ip)

    !~~~~~define masses~~~~~~~~~~~~~~~~
      amin=utamnu(keu,ked,kes,kec,keb,ket,5)
      aumin=amuseg+yrmaxi
      ipo=ip
      if(ityptl(ip).eq.60)ipo=iorptl(ip)
      tecmor=pptl(5,ipo)
      tecm=pptl(5,ip)
      tecmxx=tecm

      if(iappl.eq.4.or.iorsdf.ne.3
     &.or.ityptl(ip).eq.40.or.ityptl(ip).eq.50)then !not for droplets from remnants
        yrmax=0
      else
        yrmax=yrmaxi             !define in ainit
      endif


    !~~~~~redefine energy in case of imposed radial flow~~~~~~~~~~~~~~~~
      fradfli=1.
      if(yrmax.gt.1e-5)fradfli=fradflii                !define in ainit
      if(tecm*fradfli.lt.amin.and.tecm.gt.0.)fradfli=1.1*amin/tecm  !if flow too large, do something anyway (saturation of flow)
      if(yrmax.gt.1e-5.and.tecmor.gt.aumin
     &.and.tecm*fradfli.gt.amin) then
         tecm=tecm*fradfli
      else
        yrmax=0.
        fradfli=1.
      endif

    !~~~~~redefine energy in case of long coll flow
      if(iappl.eq.4.or.iorsdf.ne.3
     &.or.ityptl(ip).eq.40.or.ityptl(ip).eq.50)then !not for droplets from remnants
        yco=0
      else
       if(ylongmx.lt.0.)then
        yco=delzet * 1.75
       else
        yco=ylongmx
       endif
      endif
      tecmx=tecm
      if(yco.gt.0..and.tecmor.gt.aumin) then
        tecm=tecm/sinh(yco)*yco
      else
        yco=0.
      endif
      !print*,'========= cluster energy: ',pptl(5,ip),tecmx,tecm

    !~~~~~define volu~~~~~~~~~~~~~~~~
      volu=tecm/epscri(3)

    !~~~~~~~~~decay~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ibarini=(keu+ked+kes+kec+keb+ket)/3 !for baryon number conservation
        niter=0
    4   niter=niter+1
        call GraCan
        energ=0
        ibaryon=ibarini
        do i=1,np
          energ=energ+pcm(4,i)
          if(abs(ident(i)).gt.1000)ibaryon=ibaryon-sign(1,ident(i))
        enddo
        !print*,'++4+++',energ,tecm,tecm/volu,ibaryon
        if((abs(energ-tecm).gt.0.1.or.ibaryon.ne.0)
     &       .and.niter.le.maxit) goto 4 !comment for energy non-conservation
        if(niter.gt.maxit)then
         iret=1
         goto1000
        endif

    !~~~~~~~~~~long coll flow -> particles~~~~~~~~~~~~~~~~
      tecmxxx=tecm
      if(yco.gt.0.) then
        errlim=0.0001
        tecm=tecmx
        niter=0
 611    energ=0.
        niter=niter+1
        do i=1,np
          ylong(i)=(2*rangen()-1)*yco
          uu(3)=sinh(ylong(i))
          uu(4)=cosh(ylong(i))
          energ=energ+uu(4)*pcm(4,i)+uu(3)*pcm(3,i)
        enddo
        if(abs(energ-tecm).gt.0.1)then
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
          uu(1)= 0
          uu(2)= 0
          uu(3)= sinh(ylong(i))
          uu(4)= cosh(ylong(i))
          call utlob3(-1,uu(1),uu(2),uu(3),uu(4),1e0
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
      if(fradfli.lt.1.) then
        errlim=0.0001
        tecm=tecmxx
        phinull=phievt+ranphi
        do n=1,np
          vradi(n)=tanh(sqrt(rangen())*yrmax)
          vtang(n)=0.
          phifop(n)=rangen()*2*pi
          radfop(n)=0.
          taufop(n)=0.
        enddo
        energ=0.
        do n=1,np
          co=cos(phifop(n)+phinull)
          si=sin(phifop(n)+phinull)
          v1=vradi(n)*co+vtang(n)*si
          v2=vradi(n)*si-vtang(n)*co
          v3=0
          a=1.-v1*v1-v2*v2-v3*v3
          if(a.le.0.)stop'in DropletDecay (20032008)        '
          gm=1./sqrt(a)

          uu(1)=v1*gm
          uu(2)=v2*gm
          uu(3)=v3*gm
          uu(4)=gm
          call utlob3(-1,uu(1),uu(2),uu(3),uu(4),1e0
     *         , pcm(1,n), pcm(2,n), pcm(3,n), pcm(5,n))
          energ=energ+sqrt(amass(n)**2
     *                     +pcm(1,n)**2+pcm(2,n)**2+pcm(3,n)**2)
        enddo
        esoll=tecm
        !print*,tecm,energ
        iiscale=1
        if(iiscale.eq.1)then
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
            if(abs(scal-1.).le.errlim) goto300
          enddo
  300     continue
        endif
      else
        phinull=0.
        do n=1,np
          vradi(n)=0.
          vtang(n)=0.
          phifop(n)=0.
          radfop(n)=0.
          taufop(n)=0.
        enddo
      endif
    !~~~~~~~~~~~~~~~

      nptlb=nptl
      do n=1,np
        nptl=nptl+1
        if(nptl.gt.mxptl)
     .   call utstop('DropletDecay: mxptl too small&',
     +sizeof('DropletDecay: mxptl too small&'))
        idptl(nptl)=ident(n)
        do j=1,4
          p(j)=pcm(j,n)
        enddo
        p(5)=amass(n)
        call utlob2(-1,c(1),c(2),c(3),c(4),c(5),p(1),p(2),p(3),p(4),10)
        do j=1,5
          pptl(j,nptl)=p(j)
        enddo
        if(fradfli.lt.1.) then
          ityptl(nptl)=60
        elseif(ityptl(ip).eq.40.or.ityptl(ip).eq.50)then
          ityptl(nptl)=ityptl(ip)+1
        else
          ityptl(nptl)=19
        endif
        if(ityptl(ip).eq.60)then
         if(ityptl(nptl).eq.60)then
          ipo=iorptl(ip)
          phi=phifop(n)
          tau=taufop(n)
          r=radfop(n)
          !---add r-randomness
          !dr=5
          !do while(dr.lt.-2.or.dr.gt.2.)
          ! dr=sqrt(3.)*(rangen()+rangen()+rangen()+rangen()-2)
          !enddo
          !r=r+dr
          !---------
          zeta=0.5*log((p(4)+p(3))/(p(4)-p(3)))
          !deleta=etahy(2)-etahy(1)
          !zeta=zetaor-0.5*delzet+delzet*rangen()
          z=tau*sinh(zeta)
          t=tau*cosh(zeta)
          xorptl(1,nptl)=r*cos(phifop(n)+phinull)
          xorptl(2,nptl)=r*sin(phifop(n)+phinull)
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
      if(ish.ge.3)then
        write(ifch,*)'decay products:'
        call alist('&',
     +sizeof('&'),nptlb+1,nptl)
        if(ish.ge.5)then
          write(ifch,*)'momentum sum:'
          do kk=1,5
          pptl(kk,nptl+1)=0
          do ii=nptlb+1,nptl
          pptl(kk,nptl+1)=pptl(kk,nptl+1)+pptl(kk,ii)
          enddo
          pptl(kk,nptl+2)=c(kk)
          enddo
          call alist('&',
     +sizeof('&'),nptl+1,nptl+2)
        endif
      endif

 1000 continue
      call ManiParticles(-1,1)
      call utprix('DropletDecay',ish,ishini,4)
      end

c--------------------------------------------------------------------
      subroutine ManiParticles(isens,isto)
c--------------------------------------------------------------------
      ! isens=1 -> store,  isens=-1 -> restore,  isens=9... -> redefine
      !-----------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      parameter (mspez=54,msto=2)
      common/cspez1/nspez,ispez(mspez),aspez(2,mspez),gspez(mspez)
      common/cspez2/kspez,jspez(mspez)
      common/cspez3/fuga(mspez)
      common/cspez4/ffstat(2,0:mspez+2)
      common/ctfo/tfo
      common/sp1/ffstatSave(2,0:mspez+2,msto),fugaSave(mspez,msto)
      common/sp2/iocludeSave(msto),kspezSave(msto),nspezSave(msto)
      common/sp3/tfoSave(msto),meosmuSave(msto),epscri3Save(msto)
      if(isens.eq.1)then !~~~~~~~~~~~~~~store~~~~~~~~~~~~~~
        iocludeSave(isto)=ioclude
        kspezSave(isto)=kspez
        nspezSave(isto)=nspez
        if(kspez.gt.0)then
         do m=kspez,nspez
          ffstatSave(1,m,isto)=ffstat(1,m)
          ffstatSave(2,m,isto)=ffstat(2,m)
          fugaSave(m,isto)=fuga(m)
         enddo
        endif
        do i=1,2
         do j=1,2
          ffstatSave(i,mspez+j,isto)=ffstat(i,mspez+j)
         enddo
        enddo
        tfoSave(isto)=tfo
        meosmuSave(isto)=meosmu
        epscri3Save(isto)=epscri(3)
      elseif(isens.eq.-1)then !~~~~~~~~~~~~~~restore~~~~~~~~~~~~~~
        ioclude=iocludeSave(isto)
        kspez=kspezSave(isto)
        nspez=nspezSave(isto)
        if(kspez.gt.0)then
          do m=kspez,nspez
           ffstat(1,m)=ffstatSave(1,m,isto)
           ffstat(2,m)=ffstatSave(2,m,isto)
           fuga(m)=fugaSave(m,isto)
          enddo
        endif
        do i=1,2
         do j=1,2
          ffstat(i,mspez+j)=ffstatSave(i,mspez+j,isto)
         enddo
        enddo
        tfo=tfoSave(isto)
        meosmu=meosmuSave(isto)
        epscri(3)=epscri3Save(isto)
      elseif(isens.eq.92)then !~~~~~~~~~~~~~~redefine set2~~~~~~~~~~~~~~
        !used for remnant and droplet decay --- may be modified!!!
        ioclude=5             !choice of hadronization (4 or 5)
        kspez=1               !choice of
        nspez=54              !   particles
        tfo=0.130             !freeze out temperature used in this routine
        meosmu=0              !no chemical potentials used in this routine
        epscri(3)=0.0765      !should be consitent with tfo see EoS table below
        call DefineParticles
      elseif(isens.eq.93)then !~~~~~~~~~~~~~~redefine set3~~~~~~~~~~~~~
        !used for application micro
        kspez=1               !choice of
        nspez=54              !   particles
        tfo=0.130             !freeze out temperature used in this routine
        meosmu=0              !no chemical potentials used in this routine
        epscri(3)=0.0765      !should be consitent with tfo see EoS table below
        call DefineParticles
      endif
      !--------------------------!
      ! epscri(3)  !    tfo      !
      !--------------------------!
      !   0.045    !   0.11954   !
      !   0.050    !   0.12157   !
      !   0.055    !   0.12327   !
      !   0.060    !   0.12496   !
      !   0.065    !   0.12665   !
      !   0.070    !   0.12801   !
      !   0.075    !   0.1297    !
      !   0.080    !   0.13072   !
      !   0.085    !   0.13207   !
      !   0.090    !   0.13343   !
      !   0.095    !   0.13444   !
      !   0.100    !   0.13546   !
      !   0.105    !   0.13648   !
      !   0.110    !   0.13749   !
      !   0.115    !   0.13851   !
      !   0.120    !   0.13952   !
      !   0.125    !   0.1402    !
      !--------------------------!
      end

c------------------------------------------------------------------------------
      subroutine xSpaceTime
c------------------------------------------------------------------------------
      include 'epos.inc'
      if(iSpaceTime.eq.1.and.ioclude.gt.1)then
         call xCoreCorona(0,0)
         do m=0,4,2
          do meta=-m,m,max(1,2*m)
           call xFoMass(meta)
           call xFoRadius(meta)
           call xFoRadVelocity(meta)
           call xFoTanVelocity(meta)
          enddo
         enddo
         do m=0,4,2
          do meta=-m,m,max(1,2*m)
           call xFreezeOutTauX(meta)
          enddo
         enddo
         call xFeff
         !call xFreezeOutTauEta
         !call xFreezeOutTZ
         call xEos
      elseif(iSpaceTime.eq.1)then
         call xCoreCorona(0,0)
         !stop'bjinta: space-time plots require ioclude>1.          '
      endif
      end

c------------------------------------------------------------------------------
      subroutine xEos
c------------------------------------------------------------------------------
      include'epos.inc'
      include'epos.inchy'
      common/latt/temp(14),press(14),epsi(14)
      call Lattice

      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   eos 1     '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
      write(ifhi,'(a)') ' xrange 0.   3.  yrange 0 auto'
      write(ifhi,'(a)') ' txt  "xaxis [e] (GeV/fm^3!)"'
      write(ifhi,'(a)') ' txt  "yaxis p([e]) (GeV/fm^3!)"'
      write(ifhi,'(a)') ' array 2'
      do m=1,meos
       if(eos(1,m).le.3.)write(ifhi,'(2e11.3)')eos(1,m),eos(2,m)
      enddo
      write(ifhi,'(a)') ' endarray closehisto plot 0-'
      write(ifhi,'(a)') ' openhisto htyp poc  '
      write(ifhi,'(a)') ' array 2'
      do m=2,13
       if(epsi(m).le.3.)write(ifhi,'(2e11.3)')epsi(m),press(m)
      enddo
      write(ifhi,'(a)') 'endarray closehisto plot 0'
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   eos 2     '
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
      write(ifhi,'(a)') ' xrange 0.   3.  yrange 0 auto'
      write(ifhi,'(a)') ' txt  "xaxis [e] (GeV/fm^3!)"'
      write(ifhi,'(a)') ' txt  "yaxis T([e]) (GeV)"'
      write(ifhi,'(a)') ' array 2'
      do m=1,meos
       if(eos(1,m).le.3.)write(ifhi,'(2e11.3)')eos(1,m),eos(3,m)
      enddo
      write(ifhi,'(a)') ' endarray closehisto plot 0-'
      write(ifhi,'(a)') ' openhisto htyp poc  '
      write(ifhi,'(a)') ' array 2'
      do m=2,13
       if(epsi(m).le.3.)write(ifhi,'(2e11.3)')epsi(m),temp(m)
      enddo
      write(ifhi,'(a)') 'endarray closehisto plot 0'
      !write(ifhi,'(a)')       '!##################################'
      !write(ifhi,'(a,i3)')    '!   eosmu     '
      !write(ifhi,'(a)')       '!##################################'
      !write(ifhi,'(a)') ' openhisto htyp lin xmod lin ymod lin '
      !write(ifhi,'(a)') ' xrange 0.   0.2  yrange 0 auto'
      !write(ifhi,'(a)') ' txt  "xaxis T (GeV)"'
      !write(ifhi,'(a)') ' txt  "yaxis [m]?[p]!(T) (GeV)"'
      !write(ifhi,'(a)') ' array 2'
      !do m=1,meosmu
      ! if(eos(1,m).le.3.)write(ifhi,'(2e11.3)')eosmu(1,m),eosmu(2,m)
      !enddo
      !write(ifhi,'(a)') 'endarray closehisto plot 0'
      end

c------------------------------------------------------------------------------
      subroutine Lattice
c------------------------------------------------------------------------------
      parameter (hc=0.1973,Tc=0.170/hc)
      common/latt/temp(14),press(14),epsi(14)
      real t2tc(14),p2T4(14),e2T4(14)
      ! T/Tc  no units
      data t2tc /0.80,0.87,0.96,1.02,1.07,1.14,1.20,1.28
     *          ,1.35,1.52,1.70,1.90,2.24,2.55/
      ! p/T^4,  no units
      data p2T4 /0.05,0.15,0.39,0.60,0.86,1.12,1.40,1.66
     *          ,1.91,2.32,2.65,2.89,3.19,3.41/
      do i=1,14
       temp(i)=t2tc(i)*Tc*hc                ! in GeV
       press(i)=p2T4(i)*(t2tc(i)*Tc)**4*hc  ! in GeV/fm^3
      enddo
      do i=2,13
        f1=p2T4(i-1)*(t2tc(i-1)*Tc)**4
        f2=p2T4(i  )*(t2tc(i  )*Tc)**4
        f3=p2T4(i+1)*(t2tc(i+1)*Tc)**4
        a=(t2tc(i  )*Tc - t2tc(i-1)*Tc)
        b=(t2tc(i+1)*Tc - t2tc(i  )*Tc)
        s=(f2-f1)/a*b/(a+b) + (f3-f2)/b*a/(a+b)
        s2T3=s / (t2tc(i)*Tc)**3
        e2T4(i)=  s2T3 - p2T4(i)
        epsi(i)=e2T4(i)*(t2tc(i)*Tc)**4*hc  ! in GeV/fm^3
      enddo
      end

c------------------------------------------------------------------------------
      subroutine xFreezeOutTauX(meta)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      !..........................................................................
      nhis=1
      npl=0
      nplx=0
      deleta=etahy(2)-etahy(1)
      eta1=zetahy(meta)-deleta/2
      eta2=zetahy(meta)+deleta/2
      etaav=zetahy(meta)
      taumax=tauhoc(ncentr,ntauhu(ncentr,meta))+4
      do n=1,nptl
        if(ityptl(n).eq.60
     * .and.istptl(n).ne.12.and.istptl(n).ne.11)then
         if(istptl(iorptl(n)).eq.11)then
          tau=0
          tau2=xorptl(4,n)**2-xorptl(3,n)**2
          if(tau2.gt.0.)tau=sqrt(tau2)
          if(tau.lt.taumax)then
           rap=
     *     .5*alog((xorptl(4,n)+xorptl(3,n))/(xorptl(4,n)-xorptl(3,n)))
           if(rap.ge.eta1.and.rap.le.eta2)then
           if(abs(xorptl(2,n)).le.2)then
            npl=npl+1
            nplx=nplx+1
            if(npl.eq.1)then
              if(nplx.gt.1)
     *        write(ifhi,'(a)')       '  endarray closehisto plot 0-'
              write(ifhi,'(a)')       '!-------------------------------'
              write(ifhi,'(a)')       '!   tau-x      '
              write(ifhi,'(a)')       '!-------------------------------'
              write(ifhi,'(a)')       '!newpage'
              write(ifhi,'(a,i1)')    'openhisto name t-r-0-',nhis
              write(ifhi,'(a)')       'htyp prl xmod lin ymod lin'
              write(ifhi,'(a)')       'xrange -10 10'
              write(ifhi,'(a,f5.1)')  'yrange 0 ',taumax
              write(ifhi,'(a)')       'txt  "xaxis x (fm)"'
              write(ifhi,'(a)')       'txt  "yaxis [t] (fm/c)"'
              write(ifhi,'(a,f4.1,a)')'text 0.35 0.90 "[c]=',etaav,'"'
              write(ifhi,'(a)')    'text 0.02 0.9 """B#y"B#"L#2fm""'
          write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
              write(ifhi,'(a)')       'array 2'
            endif
            write(ifhi,'(2e11.3)')xorptl(1,n),tau
            if(npl.eq.1000)then
              nhis=nhis+1
              npl=0
            endif
           endif
           endif
          endif
         endif
        endif
      enddo
      if(nplx.gt.0)write(ifhi,'(a)')  '  endarray closehisto plot 0-'
    !..........................................................................
       nhis=1
      npl=0
      nplx=0
      npli=20
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra
     *  .and.(istptl(n).eq.0.or.istptl(n).eq.1))then
           rap=1e10
           xp=tptl(n)+zptl(n)
           xm=tptl(n)-zptl(n)
           if(xp.gt.0.0.and.xm.gt.0.0)rap=.5*alog(xp/xm)
           if(rap.ge.-0.5.and.rap.le.0.5)then
           if(abs(xorptl(2,n)).le.2)then
            npl=npl+npli
            nplx=nplx+1
            if(npl.eq.npli)then
              if(nplx.gt.1)
     *        write(ifhi,'(a)')       '  endarray closehisto plot 0-'
              write(ifhi,'(a)')       '!-------------------------------'
              write(ifhi,'(a)')       '!   tau-x    corona  '
              write(ifhi,'(a)')       '!-------------------------------'
              write(ifhi,'(a,i1)')    'openhisto name t-r-0-cor-',nhis
              write(ifhi,'(a)')       'htyp pgl '
              write(ifhi,'(a)')       'array 2'
            endif
            x=xorptl(1,n)
            y=xorptl(2,n)
            z=xorptl(3,n)
            t=xorptl(4,n)
            tau=0
            tau2=t**2-z**2
            if(tau2.gt.0.)tau=sqrt(tau2)
            if(abs(y).le.2)
     .      write(ifhi,'(2e11.3)')x,tau
            dt=0.1
            do k=1,npli-1
            x=x+pptl(1,n)/pptl(4,n)*dt
            y=y+pptl(2,n)/pptl(4,n)*dt
            z=z+pptl(3,n)/pptl(4,n)*dt
            t=t+dt
            tau=0
            tau2=t**2-z**2
            if(tau2.gt.0.)tau=sqrt(tau2)
            if(abs(y).le.2)
     .            write(ifhi,'(2e11.3)')x,tau
            enddo
            if(npl.eq.1000)then
              nhis=nhis+1
              npl=0
            endif
           endif
           endif
        endif
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0'
    !..........................................................................
       end

c------------------------------------------------------------------------------
      subroutine xFeff
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr
      write(ifhi,'(a)') '!-------------------------------'
      write(ifhi,'(a)') '!   feff      '
      write(ifhi,'(a)') '!-------------------------------'
      write(ifhi,'(a)') '!newpage'
      write(ifhi,'(a)') 'openhisto name feff'
      write(ifhi,'(a)') 'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)') 'xrange -6 6'
      write(ifhi,'(a)') 'yrange 0 1.2'
      write(ifhi,'(a)') 'txt  "xaxis [c] "'
      write(ifhi,'(a)') 'txt  "yaxis f?eff!"'
      write(ifhi,'(a,f4.1,a)') 'text 0.65 0.9 "',centhy(ncentr),'fm"'
      write(ifhi,'(a)') 'array 2'
      do meta=-netahy+1,netahy-1
        neta=abs(meta)+1
        eta=etahy(neta)
        if(meta.lt.0)eta=-eta
        write(ifhi,'(2e11.3)') eta,feff(ncentr,meta)
      enddo
      write(ifhi,'(a)')    '  endarray closehisto plot 0'
      end

c------------------------------------------------------------------------------
      subroutine xFreezeOutTauEta
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      taumax=tauhoc(ncentr,ntauhu(ncentr,0))+4
    !..........................................................................
      nhis=1
      npl=0
      do n=1,nptl
        if(ityptl(n).eq.60
     * .and.istptl(n).ne.12.and.istptl(n).ne.11)then
         if(istptl(iorptl(n)).eq.11)then
          tau=0
          tau2=xorptl(4,n)**2-xorptl(3,n)**2
          if(tau2.gt.0.)tau=sqrt(tau2)
          if(tau.lt.taumax)then
            npl=npl+1
            if(npl.eq.1)then
              write(ifhi,'(a)')      '!-------------------------------'
              write(ifhi,'(a)')      '!   tau-eta      '
              write(ifhi,'(a)')      '!-------------------------------'
              write(ifhi,'(a)')      '!newpage'
              write(ifhi,'(a,i1)')   'openhisto name t-eta-',nhis
              write(ifhi,'(a)')      'htyp prl xmod lin ymod lin'
              write(ifhi,'(a)')      'xrange -4 4'
              write(ifhi,'(a,f5.1)')  'yrange 0 ',taumax
              write(ifhi,'(a)')    'txt  "xaxis [c] "'
              write(ifhi,'(a)')    'txt  "yaxis [t] (fm/c)"'
      write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
              write(ifhi,'(a)')       'array 2'
            endif
            eta=
     *     .5*alog((xorptl(4,n)+xorptl(3,n))/(xorptl(4,n)-xorptl(3,n)))
            write(ifhi,'(2e11.3)') eta,tau
            if(npl.eq.1000)then
              write(ifhi,'(a)')    '  endarray closehisto plot 0-'
              nhis=nhis+1
              npl=0
            endif
          endif
         endif
        endif
      enddo
      if(npl.ne.0)write(ifhi,'(a)')  '  endarray closehisto plot 0'
      if(npl.eq.0)stop'xFreezeOutTZ: no particles!!!!!            '
      end

c------------------------------------------------------------------------------
      subroutine xFreezeOutTZ
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common/cen/ncentr
    !..........................................................................
      nhis=1
      npl=0
      do n=1,nptl
        if(ityptl(n).eq.60
     * .and.istptl(n).ne.12.and.istptl(n).ne.11)then
         if(istptl(iorptl(n)).eq.11)then
            npl=npl+1
            if(npl.eq.1)then
              write(ifhi,'(a)')      '!-------------------------------'
              write(ifhi,'(a)')      '!   t-z      '
              write(ifhi,'(a)')      '!-------------------------------'
              write(ifhi,'(a)')      '!newpage'
              write(ifhi,'(a,i1)')   'openhisto name t-z-',nhis
              write(ifhi,'(a)')      'htyp prl xmod lin ymod lin'
              write(ifhi,'(a)')      'xrange -25 25'
              write(ifhi,'(a)')      'yrange 0 25 '
              write(ifhi,'(a)')    'txt  "xaxis z (fm)"'
              write(ifhi,'(a)')    'txt  "yaxis t (fm/c)"'
      write(ifhi,'(a,f4.1,a)')'text 0.70 0.22 "',centhy(ncentr),'fm"'
              write(ifhi,'(a)')       'array 2'
            endif
            write(ifhi,'(2e11.3)') xorptl(3,n),xorptl(4,n)
            if(npl.eq.1000)then
              write(ifhi,'(a)')    '  endarray closehisto plot 0-'
              nhis=nhis+1
              npl=0
            endif
         endif
        endif
      enddo
      if(npl.ne.0)write(ifhi,'(a)')  '  endarray closehisto plot 0'
      if(npl.eq.0)stop'xFreezeOutTZ: no particles!!!!!            '
      end

c------------------------------------------------------------------------------
      subroutine xFoMass(meta)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      taumax=tauhoc(ncentr,ntauhu(ncentr,meta))+2
      netahyxx=netahy
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!   hydro freeze out rate     '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')       '!newpage'
      do ii=1,3
        if(ii.eq.1)nphi=2
        if(ii.eq.2)nphi=1+nphihy/4
        if(ii.eq.3)nphi=1+nphihy/2
        write(ifhi,'(a,3i1)')'openhisto htyp lin name w-',meta,nphi,ii
        if(ii.eq.1)then !----------------------
         write(ifhi,'(a,f4.1)')'xmod lin xrange 0. ',taumax
         write(ifhi,'(a)')    'txt  "xaxis [t] (fm/c)"'
         write(ifhi,'(a)') 'ymod lin yrange auto auto '
         write(ifhi,'(a,f4.1,a)')'text 0.1 0.9 "  [c]=',zetahy(meta),'"'
         write(ifhi,'(a)')'txt "yaxis w "'
         write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
        endif       !-------------------------------
        write(ifhi,'(a)')'array 2'
        deltau=tauhoc(ncentr,2)-tauhoc(ncentr,1)
        do ntau=2,ntauhu(ncentr,meta)
         dy=waa(ncentr,meta,ntau,nphi)
         write(ifhi,'(2e13.5)')tauhoc(ncentr,ntau)-deltau/2,dy
        enddo
        write(ifhi,'(a)') 'endarray closehisto '
        if(ii.ne.3)write(ifhi,'(a)') 'plot 0-'
        if(ii.eq.3)write(ifhi,'(a)') 'plot 0'
      enddo
      end

c------------------------------------------------------------------------------
      subroutine xFoRadius(meta)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      taumax=tauhoc(ncentr,ntauhu(ncentr,meta))+2
      netahyxx=netahy
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!        hydro freeze out radius     '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')    '!newpage'
      do ii=1,3
        if(ii.eq.1)nphi=2
        if(ii.eq.2)nphi=1+nphihy/4
        if(ii.eq.3)nphi=1+nphihy/2
        write(ifhi,'(a,3i1)')'openhisto htyp lin name r-',meta,nphi,ii
        if(ii.eq.1)then !----------------------
         write(ifhi,'(a,f4.1)')'xmod lin xrange 0. ',taumax
         write(ifhi,'(a)')'txt  "xaxis [t] (fm/c)"'
         write(ifhi,'(a)') 'ymod lin yrange auto auto '
         write(ifhi,'(a,f4.1,a)')'text 0.1 0.9 "  [c]=',zetahy(meta),'"'
         write(ifhi,'(a)')'txt "yaxis r  (fm) "'
         write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
        endif !-------------------------------
        write(ifhi,'(a)')'array 2'
        do ntau=2,ntauhu(ncentr,meta)
         write(ifhi,'(2e13.5)')tauhoc(ncentr,ntau)
     .     ,raa(ncentr,meta,ntau,nphi)
         enddo
        write(ifhi,'(a)') 'endarray closehisto '
        if(ii.ne.3)write(ifhi,'(a)') 'plot 0-'
        if(ii.eq.3)write(ifhi,'(a)') 'plot 0'
      enddo
      end

c------------------------------------------------------------------------------
      subroutine xFoRadVelocity(meta)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      taumax=tauhoc(ncentr,ntauhu(ncentr,meta))+2
      netahyxx=netahy
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!     hydro freeze out rad velocity      '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')    '!newpage'
      do ii=1,3
        if(ii.eq.1)nphi=2
        if(ii.eq.2)nphi=1+nphihy/4
        if(ii.eq.3)nphi=1+nphihy/2
        write(ifhi,'(a,3i1)')'openhisto htyp lin name y-',meta,nphi,ii
        if(ii.eq.1)then !----------------------
         write(ifhi,'(a,f4.1)')'xmod lin xrange 0. ',taumax
         write(ifhi,'(a)')'txt "xaxis [t] (fm/c)"'
         write(ifhi,'(a)') 'ymod lin yrange auto auto '
         write(ifhi,'(a,f4.1,a)')'text 0.1 0.9 "  [c]=',zetahy(meta),'"'
         write(ifhi,'(a)')'txt "yaxis v?rad!  "'
         write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
        endif !-------------------------------
        write(ifhi,'(a)')'array 2'
        do ntau=2,ntauhu(ncentr,meta)
         write(ifhi,'(2e13.5)')tauhoc(ncentr,ntau)
     .     ,vaa(1,ncentr,meta,ntau,nphi)
        enddo
        write(ifhi,'(a)') 'endarray closehisto '
        if(ii.ne.3)write(ifhi,'(a)') 'plot 0-'
        if(ii.eq.3)write(ifhi,'(a)') 'plot 0'
      enddo
      end
c------------------------------------------------------------------------------
      subroutine xFoTanVelocity(meta)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr /ctauhu/ntauhu(ncenthx,1-netahx:netahx-1)
      taumax=tauhoc(ncentr,ntauhu(ncentr,meta))+2
      netahyxx=netahy
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!    hydro freeze out tang velocity      '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')    '!newpage'
      do ii=1,4
        if(ii.eq.1)nphi=2
        if(ii.eq.2)nphi=1+nphihy/8
        if(ii.eq.3)nphi=1+nphihy/8*3
        if(ii.eq.4)nphi=1+nphihy/2
        write(ifhi,'(a,3i1)')'openhisto htyp lin name y-',meta,nphi,ii
        if(ii.eq.1)then !----------------------
         write(ifhi,'(a,f4.1)')'xmod lin xrange 0. ',taumax
         write(ifhi,'(a)')'txt "xaxis [t] (fm/c)"'
         write(ifhi,'(a)') 'ymod lin yrange auto auto '
         write(ifhi,'(a,f4.1,a)')'text 0.1 0.9 "  [c]=',zetahy(meta),'"'
         write(ifhi,'(a)')'txt "yaxis v?tan!  "'
          write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
        endif !-------------------------------
        write(ifhi,'(a)')'array 2'
        do ntau=2,ntauhu(ncentr,meta)
         write(ifhi,'(2e13.5)')tauhoc(ncentr,ntau)
     .     ,vaa(2,ncentr,meta,ntau,nphi)
        enddo
        write(ifhi,'(a)') 'endarray closehisto '
        if(ii.ne.4)write(ifhi,'(a)') 'plot 0-'
        if(ii.eq.4)write(ifhi,'(a)') 'plot 0'
      enddo
      end

c-----------------------------------------------------------------------
      subroutine xCoreCorona(iii,jjj)
c-----------------------------------------------------------------------
c     space-time evolution of core and corona
c
c     cluster ............   ist=11  ity=60
c     core particles .....   ist=0   ity=60
c     corona particles ...   ist=0   ity/=60
c
c    iii=1: plot also binary collisions
c    jjj>0: multiplicity trigger (useful for pp)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.inchy'
      common/cen/ncentr
      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common/cdelzet/delzet,delsgr
      parameter (myy=48,mrr=100)
      real yy(myy),rr(mrr)
      common/cranphi/ranphi

      phinll=phievt+ranphi
      phi=   phievt+ranphi
      !print*,'   EventPhi=',phievt,'   RandomPhi=',ranphi
      rapmax=6
      radmax=10
      r1=0.0
      if(maproj.gt.1)r1=radnuc(maproj)
      r2=0.0
      if(matarg.gt.1)r2=radnuc(matarg)
      if(maproj.eq.1.and.matarg.gt.1)r1=r2
      if(maproj.gt.1.and.matarg.eq.1)r2=r1
      a=7.8
      b=bimevt/2
      n1=koievt
      n2=0
      do k=1,koll
       if(itpr(k).gt.0)n2=n2+1
      enddo
      n3=nglevt

      if(jjj.gt.0)then
      multy1=0
       do i=maproj+matarg+1,nptl
        if(istptl(i).eq.0)then
          amt=pptl(5,i)**2+pptl(1,i)**2+pptl(2,i)**2
          rap=1000
          if(amt.gt.0..and.pptl(4,i).gt.0.)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,i))*alog((pptl(4,i)+abs(pptl(3,i)))/amt)
          endif
          ch=0
          if(abs(idptl(i)).ge.100.and.abs(idptl(i)).lt.10000)then
            call idchrg(idptl(i),ch)
            if(abs(ch).gt.0.1.and.abs(rap).le.1.)multy1=multy1+1
          endif
         endif
        enddo
        ih1=jjj/100
        ih2=mod(jjj,100)
        if(0.5*multy1.lt.ih1.or.0.5*multy1.gt.ih2)return
      endif
      xmax=1+int(a*1.5)

      write(ifhi,'(a)')       '!---------------------------------'
      write(ifhi,'(a)')       '!   core particles                '
      write(ifhi,'(a)')       '!---------------------------------'
      write(ifhi,'(a)')       '!newpage'
      write(ifhi,'(a)')      'openhisto name st1'
      write(ifhi,'(a)')       'htyp prv xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-xmax,xmax
      write(ifhi,'(a,2e11.3)')'yrange',-a,a
      write(ifhi,'(a)')    'text 0.85 -0.25 " x (fm)"'
      write(ifhi,'(a)')    'txt "yaxis y (fm)"'
      write(ifhi,'(a)')       'text 0.05 0.90 "core"'
      write(ifhi,'(a)')       'text 0.05 0.80 "corona"'
      write(ifhi,'(a,f4.1,a)')'text 0.82 0.07 "[c]=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
      write(ifhi,'(a)')       'array 2'
      ncore=0
      do i=1,nptl
       if(abs(sptl(i)).lt.0.5*delsgr.and.istptl(i).eq.2)then
         write(ifhi,'(2e11.3)')xptl(i)*cos(phi)+yptl(i)*sin(phi)
     .      ,                      -xptl(i)*sin(phi)+yptl(i)*cos(phi)
         ncore=ncore+1
       endif
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       '!----------------------------------'
      write(ifhi,'(a)')       '!   corona particles               '
      write(ifhi,'(a)')       '!----------------------------------'
      write(ifhi,'(a)')      'openhisto name st2'
      write(ifhi,'(a)')       'htyp pgv xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      ncorona=0
      do i=1,nptl
       if(abs(sptl(i)).lt.0.5*delsgr.and.istptl(i).eq.0
     .  .and.ityptl(i).ne.60.and.ityptl(i).ne.19)then
         write(ifhi,'(2e11.3)')xptl(i)*cos(phi)+yptl(i)*sin(phi)
     .      ,                      -xptl(i)*sin(phi)+yptl(i)*cos(phi)
        ncorona=ncorona+1
       endif
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'
      !print*,'b=',bimevt,'   ncorona:ncore =  ',ncorona,':',ncore
         if(iii.eq.1)then
      write(ifhi,'(a)')       '!----------------------------------'
      write(ifhi,'(a)')       '!   binary collisions             '
      write(ifhi,'(a)')       '!----------------------------------'
      write(ifhi,'(a)')    'plot 0-'
      write(ifhi,'(a)')      'openhisto name coo'
      write(ifhi,'(a)')       'htyp pbl xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-xmax,xmax
      write(ifhi,'(a,2e11.3)')'yrange',-a,a
      write(ifhi,'(a)')       'array 2'
      do k=1,koll
       if(itpr(k).gt.0)then
         write(ifhi,'(2e11.3)')coord(1,k)*cos(phi)+coord(2,k)*sin(phi)
     * ,                      -coord(1,k)*sin(phi)+coord(2,k)*cos(phi)
       endif
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto'
         endif
      if(r1.ne.0.0)then
        write(ifhi,'(a)')    'plot 0-'
        write(ifhi,'(a)')       '!----------------------------------'
        write(ifhi,'(a)')       '!   hard spheres             '
        write(ifhi,'(a)')       '!----------------------------------'
        write(ifhi,'(a)')   'openhisto name stc1 htyp lyu'
        write(ifhi,'(a)')   'array 2'
        do j=-50,50
         phi=j/50.*pi*0.55
         write(ifhi,'(2e11.3)')r1*cos(phi)-b,r1*sin(phi)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto'
      endif
      if(r2.ne.0.0)then
        write(ifhi,'(a)')    'plot 0-'
        write(ifhi,'(a)')   'openhisto name stc1 htyp lyu'
        write(ifhi,'(a)')   'array 2'
        do j=-50,50
         phi=j/50.*pi*0.55
         write(ifhi,'(2e11.3)')-r1*cos(phi)+b,r1*sin(phi)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto'
      endif

      write(ifhi,'(a)')    'plot 0'

   !........................................................................................
      if(ioclude.le.1)return
   !........................................................................................
      delrap=2*rapmax/float(myy)
      do m=1,myy
        yy(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n).eq.2)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        if(abs(rinp).le.1.and.abs(routp).le.1.)then
          rapx=dezptl(n)
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(rapx+rapmax)/delrap+1
          if(m.gt.myy)m=myy
          yy(m)=yy(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    core segment energy per d[c]dxdy         '
      write(ifhi,'(a)')'!           vs space-time rapidity rapx       '
      write(ifhi,'(a)')'!   (same as histogram rapx eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')       '!newpage'
      write(ifhi,'(a)')    'openhisto name rapx'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2f7.3)') 'xrange ',-rapmax,rapmax
      write(ifhi,'(a)')        'yrange 0 auto '
      write(ifhi,'(a)')  'txt "title initial energy          "'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.70 "x=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.60 "y=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
      write(ifhi,'(a)')  'txt  "xaxis space-time rapidity [c] "'
      write(ifhi,'(a)')  'txt  "yaxis dE/d[c]dxdy "'
      write(ifhi,'(a)')       'array 2'
      do m=1,myy
        write(ifhi,'(2e11.3)')-rapmax+(m-0.5)*delrap, yy(m)/4./delrap
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
   !........................................................................................
      delrap=2*rapmax/float(myy)
      do m=1,myy
        yy(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n)/2.eq.0)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        if(abs(rinp).le.1.and.abs(routp).le.1.)then
          rapx=dezptl(n)
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(rapx+rapmax)/delrap+1
          if(m.ge.1.and.m.le.myy)yy(m)=yy(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    corona segment energy per d[c]dxdy       '
      write(ifhi,'(a)')'!           vs space-time rapidity rapx       '
      write(ifhi,'(a)')'!   (same as histogram rapx eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')    'openhisto name rapx'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2f7.3)') 'xrange ',-rapmax,rapmax
      write(ifhi,'(a)')        'yrange 0 auto '
      write(ifhi,'(a)')  'txt "title initial energy          "'
      write(ifhi,'(a)')  'txt  "xaxis space-time rapidity [c] "'
      write(ifhi,'(a)')  'txt  "yaxis dE/d[c]dxdy "'
      write(ifhi,'(a)')       'array 2'
      do m=1,myy
        write(ifhi,'(2e11.3)')-rapmax+(m-0.5)*delrap, yy(m)/4./delrap
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
      call xEiniEta(1)
      write(ifhi,'(a)')  'plot 0'
   !........................................................................................
      delrad=2*radmax/float(mrr)
      do m=1,mrr
        rr(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n).eq.2)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        rapx=dezptl(n)
        if(abs(rapx).le.1.and.abs(routp).le.1.)then
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(rinp+radmax)/delrad+1
          if(m.gt.mrr)m=mrr
          rr(m)=rr(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    core segment energy per d[c]dxdy vs x    '
      write(ifhi,'(a)')'!   (same as histogram rinp eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')       '!newpage'
      write(ifhi,'(a)')    'openhisto name rapx'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2f7.3)') 'xrange ',-radmax,radmax
      write(ifhi,'(a)')        'yrange 0 auto '
      write(ifhi,'(a)')  'txt "title initial energy          "'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.70 "[c]=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.60 "y=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
      write(ifhi,'(a)')  'txt  "xaxis x (fm)"'
      write(ifhi,'(a)')  'txt  "yaxis dE/d[c]dxdy "'
      write(ifhi,'(a)')       'array 2'
      do m=1,mrr
        write(ifhi,'(2e11.3)')-radmax+(m-0.5)*delrad, rr(m)/4./delrad
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
   !........................................................................................
      delrad=2*radmax/float(mrr)
      do m=1,mrr
        rr(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n)/2.eq.0)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        rapx=dezptl(n)
        if(abs(rapx).le.1.and.abs(routp).le.1.)then
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(rinp+radmax)/delrad+1
          if(m.gt.mrr)m=mrr
          rr(m)=rr(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    corona segment energy per d[c]dxdy vs x  '
      write(ifhi,'(a)')'!   (same as histogram rinp eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')    'openhisto name rapx'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      do m=1,mrr
        write(ifhi,'(2e11.3)')-radmax+(m-0.5)*delrad, rr(m)/4./delrad
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
      call xEiniX(1)
      write(ifhi,'(a)')  'plot 0'
   !........................................................................................
      delrad=2*radmax/float(mrr)
      do m=1,mrr
        rr(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n).eq.2)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        rapx=dezptl(n)
        if(abs(rapx).le.1.and.abs(rinp).le.1.)then
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(routp+radmax)/delrad+1
          if(m.gt.mrr)m=mrr
          rr(m)=rr(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    core segment energy per d[c]dxdy vs y    '
      write(ifhi,'(a)')'!   (same as histogram routp eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')       '!newpage'
      write(ifhi,'(a)')    'openhisto name rout'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2f7.3)') 'xrange ',-radmax,radmax
      write(ifhi,'(a)')        'yrange 0 auto '
      write(ifhi,'(a)')  'txt "title initial energy          "'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.70 "[c]=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.05 0.60 "x=0"'
      write(ifhi,'(a,f4.1,a)')'text 0.65 0.9 "',centhy(ncentr),'fm"'
      write(ifhi,'(a)')  'txt  "xaxis y (fm)"'
      write(ifhi,'(a)')  'txt  "yaxis dE/d[c]dxdy "'
      write(ifhi,'(a)')       'array 2'
      do m=1,mrr
        write(ifhi,'(2e11.3)')-radmax+(m-0.5)*delrad, rr(m)/4./delrad
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
   !........................................................................................
      delrad=2*radmax/float(mrr)
      do m=1,mrr
        rr(m)=0
      enddo
      do n=1,nptl
        if(dezptl(n).lt.1e3.and.n.le.maxfra.and.istptl(n)/2.eq.0)then
        routp=-sin(phinll)*xptl(n)+cos(phinll)*yptl(n)
        rinp = cos(phinll)*xptl(n)+sin(phinll)*yptl(n)
        rapx=dezptl(n)
        if(abs(rapx).le.1.and.abs(rinp).le.1.)then
          eco=0
          amt=pptl(5,n)**2+pptl(1,n)**2+pptl(2,n)**2
          if(amt.gt.0..and.pptl(4,n)+abs(pptl(3,n)).gt.0.d0)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,n))*alog((pptl(4,n)+abs(pptl(3,n)))/amt)
            eco=amt*cosh(rap-rapx)
          endif
          m=(routp+radmax)/delrad+1
          if(m.gt.mrr)m=mrr
          rr(m)=rr(m)+eco
        endif
        endif
      enddo
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!    corona segment energy per d[c]dxdy vs y  '
      write(ifhi,'(a)')'!   (same as histogram routp eco... in optns)  '
      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')    'openhisto name rout'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      do m=1,mrr
        write(ifhi,'(2e11.3)')-radmax+(m-0.5)*delrad, rr(m)/4./delrad
      enddo
      write(ifhi,'(a)')  '  endarray closehisto plot 0-'
      call xEiniY(1)
      write(ifhi,'(a)')  'plot 0'

      end

c------------------------------------------------------------------------------
      subroutine xEini(ii)
c------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.inchy'
      common/cen/ncentr

      entry xEiniX(ii)
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!        hydro initial energy vs x  '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')'openhisto  array 2'
      do nr=nradhy,2,-1
       write(ifhi,'(2e13.5)')
     .     -radhy(nr),epsii(ncentr,ii,1,nr)*tauhoc(ncentr,1)
      enddo
      do nr=1,nradhy
       write(ifhi,'(2e13.5)')
     .     radhy(nr),epsii(ncentr,ii,1,nr)*tauhoc(ncentr,1)
      enddo
      write(ifhi,'(a)') 'endarray closehisto '
      return

      entry xEiniY(ii)
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!        hydro initial energy vs y     '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')'openhisto  array 2'
      do nr=nradhy,2,-1
       write(ifhi,'(2e13.5)')
     .    -radhy(nr),epsii(ncentr,ii,nphihy,nr)*tauhoc(ncentr,1)
      enddo
      do nr=1,nradhy
       write(ifhi,'(2e13.5)')
     .    radhy(nr),epsii(ncentr,ii,nphihy,nr)*tauhoc(ncentr,1)
      enddo
      write(ifhi,'(a)') 'endarray closehisto '
      return

      entry xEiniEta(ii)
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a,i3)') '!        hydro initial energy vs y     '
      write(ifhi,'(a)')    '!----------------------------------------'
      write(ifhi,'(a)')'openhisto  array 2'
      do neta=netahy,2,-1
       write(ifhi,'(2e13.5)')
     .   -etahy(neta),epsii(ncentr,neta,1,1)*tauhoc(ncentr,1)
      enddo
      do neta=1,netahy
       write(ifhi,'(2e13.5)')
     .   etahy(neta),epsii(ncentr,neta,1,1)*tauhoc(ncentr,1)
      enddo
      write(ifhi,'(a)') 'endarray closehisto '
      return

      end

c-----------------------------------------------------------------------------
      integer function idxHiranoTable(id)
c-----------------------------------------------------------------------------
      ida=abs(id)
      ihi=0
      if(ida.eq.120.or.id.eq.110)   ihi=2       !pion
      if(id.eq.220)                 ihi=3       !eta
      if(ida.eq.121.or.id.eq.111)   ihi=4       !rho
      if(id.eq.221)                 ihi=5       !omega
      !if(id.eq.)                        ihi=6         !sigma
      if(id.eq.330)                 ihi=7       !eta prime
      !if(id.eq.)                        ihi=8         !f_0
      !if(id.eq.)                        ihi=9         !a_0
      if(id.eq.331)                 ihi=10      !phi
      !if(id.eq.)                        ihi=11          !h_1
      if(ida.eq.130.or.ida.eq.230)  ihi=12      !K
      if(ida.eq.131.or.ida.eq.231)  ihi=13      !K star
      if(ida.eq.1120.or.ida.eq.1220)ihi=14      !N
      if(ida.eq.1111.or.ida.eq.1121)ihi=15      !Delta
      if(ida.eq.1221.or.ida.eq.2221)ihi=15      !Delta
      if(ida.eq.2130)               ihi=16      !Lambda
      if(ida.eq.1130.or.ida.eq.1230)ihi=17      !Sigma
      if(ida.eq.2230)               ihi=17      !Sigma
      idxHiranoTable=ihi
      end

c----------------------------------------------------------------------
      subroutine InitializeHyperbola
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      common /cttaus/  tpro,zpro,ttar,ztar,ttaus,detap,detat
      ttaus=1
      ypjtl=6
      yhaha=3
      etapro=(ypjtl-yhaha)*etafac
      etatar=-yhaha*etafac
      detap=dble(etapro)
      detat=dble(etatar)
      tpro=dcosh(detap)
      zpro=dsinh(detap)
      ttar=dcosh(detat)
      ztar=dsinh(detat)
      end

c----------------------------------------------------------------------
      integer function idGet()
c----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),ic(2)
      if(ish.ge.5)write(ifch,'(a/6i4)')
     *' keu ked kes kec keb ket:',keu,ked,kes,kec,keb,ket
      do i=1,6
      jc(i,1)=0
      jc(i,2)=0
      enddo
      if(keu.ge.0)jc(1,1)=keu
      if(ked.ge.0)jc(2,1)=ked
      if(kes.ge.0)jc(3,1)=kes
      if(kec.ge.0)jc(4,1)=kec
      if(keb.ge.0)jc(5,1)=keb
      if(ket.ge.0)jc(6,1)=ket
      if(keu.lt.0)jc(1,2)=-keu
      if(ked.lt.0)jc(2,2)=-ked
      if(kes.lt.0)jc(3,2)=-kes
      if(kec.lt.0)jc(4,2)=-kec
      if(keb.lt.0)jc(5,2)=-keb
      if(ket.lt.0)jc(6,2)=-ket
      if(ish.ge.5)write(ifch,'(a/6i4/6i4)')' jc:',jc
      idr=0
      do  nf=1,nflav
        do  ij=1,2
          if(jc(nf,ij).ge.10)idr=7*10**8
        enddo
      enddo
      if(idr/10**8.ne.7)then
        call idenco(jc,ic,ireten)
        if(ic(1).eq.0.and.ic(2).eq.0)then
          ic(1)=100000
          ic(2)=100000
        endif

        idr=8*10**8+ic(1)*100+ic(2)/100
        if(ish.ge.5)write(ifch,'(a,i9)')' id:',idr
      else
        idr=idr
     *       +mod(jc(1,1)+jc(2,1)+jc(3,1)+jc(4,1),10**4)*10**4
     *       +mod(jc(1,2)+jc(2,2)+jc(3,2)+jc(4,2),10**4)
        call idtrbi(jc,ibptl(1,1),ibptl(2,1),ibptl(3,1),ibptl(4,1))
      endif
      idGet=idr
      end

