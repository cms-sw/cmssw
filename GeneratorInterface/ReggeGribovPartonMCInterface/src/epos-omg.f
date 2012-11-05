c----------------------------------------------------------------------
c The two elementary fuctions of our approach, the profile fuction G
c and the Phi exponent G~, are here referred to as Gpro and Gtilde.
c Both functions can be written as
c
c               G = \sum_type  alp * xp**bet * xm**betp
c
c The parameters alp, bet, betp are calculated in GfunParK (k-mode,
c b is takento the one of pair k) or GfunPar (b-mode: arbitrary b) as
c
c  Gpro:   bet  = betD'  + epsGp + gamD*b**2 + epsG -alppar
c          bet' = betD'' + epsGt + gamD*b**2 + epsG -alppar
c
c          alp  = alpD*f * s**(betD+gamD*b**2+epsG) * exp(-b**2/delD)
c
c  Gtilde: bet~  = bet  + 1
c          bet~' = bet' + 1
c
c          alp~  = alp * gam(bet~)          * gam(bet~')
c                      * gam(1+alppro)      * gam(1+alptar)
c                      * gam(1+alppro+bet~) * gam(1+alptar+bet~')
c                      * (1+epsGt')         * (1+epsGt')
c
c The parameters depend implicitely on s.
c
c In the program we use om1 = Gpro
c  (they differ by a constant which is actually one)
c and om5 = om1 * 0.5
c All functions related to om1 are called om1... .
c
c The inclusive Pomeron distributions are
c
c      PomInc(xp,xm) = Gpro(xp,xm) * (1-xp)**alppro * (1-xm)**alptar
c
c----------------------------------------------------------------------


c----------------------------------------------------------------------
      subroutine GfunParK(irea)   !---MC---
c----------------------------------------------------------------------
c  calculates parameters alp,bet,betp of the G functions (k-mode)
c  and the screening exponents epsilongp(k,i), epsilongt(k,i), epsilongs(k,i)
c----------------------------------------------------------------------
c  Gpro parameters written to /comtilde/atilde(,)btildep(,),btildepp(,)
c Gtilde parameters written to /cgtilde/atildg(,)btildgp(,),btildgpp(,)
c  two subscripts: first=type, second=collision k
c Certain pieces of this routine are only done if irea is <= or = zero.
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'
      double precision atildg,btildgp,btildgpp
      common/cgtilde/atildg(idxD0:idxD1,kollmx)
     *,btildgp(idxD0:idxD1,kollmx),btildgpp(idxD0:idxD1,kollmx)
      double precision utgam2,coefgdp,coefgdt
      parameter(nbkbin=40)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      common /cgtilnu/ cfbetpnp,cfbetppnp,cfbetpnm,cfbetppnm,cfalpro
     &,cfaltar,cfbpap,cfbpam,cfbppap,cfbppam
      double precision cfbetpnp,cfbetppnp,cfbetpnm,cfbetppnm,cfalpro
     &,cfaltar,cfbpap,cfbpam,cfbppap,cfbppam,gamv,eps
      parameter (eps=1.d-25)
      parameter(nxeps=20,nyeps=32)
      common/cxeps1/w(0:nxeps,nyeps),y1(nyeps),y2(nyeps)
      common/cxeps2/db,b1,b2
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/nucl3/phi,bimp
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)

      b1=0.03
      b2=bkmx*1.2
      db=(b2-b1)/nyeps
      call utprj('GfunParK ',ish,ishini,10)

      cfbetpnp=0.d0
      cfbetppnp=0.d0
      cfbetpnm=0.d0
      cfbetppnm=0.d0
      cfalpro=dble(ucfpro)
      cfaltar=dble(ucftar)
      cfbpap=1.d0
      cfbppap=1.d0
      cfbpam=1.d0
      cfbppam=1.d0
      alpfom=0.
      sy=engy*engy

      do k=1,koll
        do i=ntymi,ntymx
          atilde(i,k)=0.d0
          btildep(i,k)=0.d0
          btildepp(i,k)=0.d0
        enddo
        do i=idxD0,idxD1
          atildg(i,k)=0.d0
          btildgp(i,k)=0.d0
          btildgpp(i,k)=0.d0
        enddo
      enddo


! calculate collision number according to Glauber -----------------------

      bglaub=sqrt(sigine/10./pi)
      nglevt=0
      do ko=1,koll
        r=bk(ko)
        if(r.le.bglaub)nglevt=nglevt+1
      enddo

! Z_parton_projectile (zparpro) and Z_parton_target (zpartar)-----------

      ztav=0.
      zpav=0.
      if(iscreen.eq.1.or.isplit.eq.1)then

        b2x=b2xscr
        alpfom=alpfomi*fegypp
        rho0p=conrho(1,0.)
        rho0t=conrho(2,0.)
        bcut=0.

        do k=1,koll
          ip=iproj(k)
          it=itarg(k)
        !....... targ partons seen by proj
          if(lproj(ip).gt.0)then
            absb=max(1.e-9,bk(k))
            b2=absb*absb
            zkp=fegypp*exp(-b2/b2x)
            zpartar(k)=min(zkp,epscrx)
            if(lproj3(ip).gt.1)then
              do lt=1,lproj3(ip)
                kp=kproj3(ip,lt)
                if(kp.ne.k)then
                  ikt=itarg(kp)
                  rtarg=sqrt(xtarg(ikt)**2+ytarg(ikt)**2+ztarg(ikt)**2)
                  rho=conrho(2,rtarg)/rho0t
                  fegyAA=epscrw*fscro(engy/egyscr,rho)
                  absb=max(1.e-9,abs(bk(kp))-bcut)
                  b2=absb*absb
                  zkp=fegyAA*exp(-b2/b2x)
                  zpartar(k)=zpartar(k)+min(zkp,epscrx)
                endif
c                alpfom=max(alpfom,dble(zpartar(k)))
              enddo
            endif
          else
            zpartar(k)=0.
          endif
          ztav=ztav+zpartar(k)
         !...........proj partons seen by targ
          if(ltarg(it).gt.0)then
            absb=max(1.e-9,bk(k))
            b2=absb*absb
            zkt=fegypp*exp(-b2/b2x)
            zparpro(k)=min(zkt,epscrx)
            if(ltarg3(it).gt.1)then
              do lp=1,ltarg3(it)
                kt=ktarg3(it,lp)
                if(kt.ne.k)then
                  ikp=iproj(kt)
                  rproj=sqrt(xproj(ikp)**2+yproj(ikp)**2+zproj(ikp)**2)
                  rho=conrho(1,rproj)/rho0p
                  fegyAA=epscrw*fscro(engy/egyscr,rho)
                  absb=max(1.e-9,abs(bk(kt))-bcut)
                  b2=absb*absb
                  zkt=fegyAA*exp(-b2/b2x)
                  zparpro(k)=zparpro(k)+min(zkt,epscrx)
                endif
c                alpfom=max(alpfom,dble(zparpro(k)))
              enddo
            endif
          else
            zparpro(k)=0.
          endif
          zpav=zpav+zparpro(k)
          xzcutpar(k)=dble(exp(-min(50.,xzcut*(zparpro(k)+zpartar(k)))))
        enddo

      else                      ! no screening

        do k=1,koll
          zparpro(k)=0.
          zpartar(k)=0.
          xzcutpar(k)=1d0
        enddo

      endif

c calculation of epsilongp epsilongt

      if(iscreen.eq.1)then

          !ip=0
       do k=1,koll
          !ipp=ip
        epsG=0.
        epsilongs(k,0)=0.
        epsilongs(k,1)=0.
        ip=iproj(k)             !...........projectile
       if(lproj(ip).gt.0)then
          x=zparpro(k)
          epsilongs(k,0)=sign(abs(epscrd)*x
     &                       ,epscrd)
          epsilongp(k,0)=max(-betDp(0,iclpro,icltar)-0.95+alppar,
     &                        epscrs*x)
c     &                       min(epscrx,epscrs*x))
          if(sy.ge.sfshlim)then
            epsilongp(k,1)=max(-betDp(1,iclpro,icltar)-0.95+alppar,
     &                          epscrh*x)
c     &                          min(epscrx,epscrh*x))
          else
            epsilongp(k,1)=epsilongp(k,0)
c            epsilongs(k,1)=epsilongs(k,0)
          endif
          gammaV(k)=1.d0
        else
         epsilongp(k,0)=0.
         epsilongp(k,1)=0.
         gammaV(k)=1.d0
        endif

        it=itarg(k)             !...........target
        if(ltarg(it).gt.0)then
          x=zpartar(k)
          epsilongs(k,1)=sign(abs(epscrd)*x
     &                       ,epscrd)
          epsilongt(k,0)=max(-betDpp(0,iclpro,icltar)-0.95+alppar,
     &                        epscrs*x)
c     &                        min(epscrx,epscrs*x))
          if(sy.ge.sfshlim)then
            epsilongt(k,1)=max(-betDpp(1,iclpro,icltar)-0.95+alppar,
     &                          epscrh*x)
c     &                          min(epscrx,epscrh*x))
          else
            epsilongt(k,1)=epsilongt(k,0)
c            epsilongs(k,1)=epsilongs(k,0)
          endif
cc          gammaV(k)=gammaV(k)
        else
         epsilongt(k,0)=0.
         epsilongt(k,1)=0.
         gammaV(k)=gammaV(k)
        endif

       enddo

      else                      ! no screening

       do k=1,koll
        epsilongs(k,0)=0.
        epsilongs(k,1)=0.
        epsilongp(k,0)=0.
        epsilongp(k,1)=0.
        epsilongt(k,0)=0.
        epsilongt(k,1)=0.
        gammaV(k)=1.d0
       enddo

      endif


!..............alpha beta betap for Gtilde (->PhiExpo).......................

      imax=idxD1
      if(iomega.eq.2)imax=1

      do k=1,koll

        b=bk(k)
        b2=bk(k)*bk(k)
        ip=iproj(k)
        it=itarg(k)

        if(b.lt.(nbkbin-1)*bkbin)then
          ibk=int(bk(k)/bkbin)+1
          if(isetcs.gt.1.and.iclegy.lt.iclegy2)then
            egy0=egylow*egyfac**float(iclegy-1)
            xkappa1=xkappafit(iclegy,iclpro,icltar,ibk)
     *         +(bk(k)-bkbin*float(ibk-1))/bkbin
     *         *(xkappafit(iclegy,iclpro,icltar,ibk+1)
     *         -xkappafit(iclegy,iclpro,icltar,ibk))
            xkappa2=xkappafit(iclegy+1,iclpro,icltar,ibk)
     *         +(bk(k)-bkbin*float(ibk-1))/bkbin
     *         *(xkappafit(iclegy+1,iclpro,icltar,ibk+1)
     *         -xkappafit(iclegy+1,iclpro,icltar,ibk))
            xkappa=xkappa1+(xkappa2-xkappa1)/log(egyfac)
     *         *(log(engy)-log(egy0))
            xkappa=facmc*xkappa
          else
            xkappa=xkappafit(iclegy,iclpro,icltar,ibk)
     *         +(bk(k)-bkbin*float(ibk-1))/bkbin
     *         *(xkappafit(iclegy,iclpro,icltar,ibk+1)
     *         -xkappafit(iclegy,iclpro,icltar,ibk))
            xkappa=facmc*xkappa
          endif
        else
          xkappa=1.
        endif
        gfactorp=1.!+(gfactor-1)*exp(-5*b/gwidth/bglaub)
        gfactort=1.!+(gfactor-1)*exp(-5*b/gwidth/bglaub)

        do i=idxDmin,imax
          gamV=gammaV(k)
c          if(i.lt.2)then
c          if(i.eq.0)then
c            epsG=epsilongs(k,i)
          if(i.eq.2)then
            if(epscrd.lt.0.)then
              epsG=epsilongs(k,0)+epsilongs(k,1)
              epsGp=0.
              epsGt=0.
            else
              epsG=0.
              epsGp=epsilongs(k,0)
              epsGt=epsilongs(k,1)
            endif
          else
            epsG=0.
            epsGp=0.
            epsGt=0.
          endif
          gamb=gamD(i,iclpro,icltar)*b2
          atildg(i,k)=dble(alpD(i,iclpro,icltar))
     *            *cfalpro*cfaltar
     *            *gamv
c          if(i.eq.0) atildg(i,k)=atildg(i,k)
          atildg(i,k)=atildg(i,k)
     *            *dble(xkappa*xkappa)
          if(i.lt.2)then
            atildg(i,k)=atildg(i,k)*dble(
     *            chad(iclpro)*chad(icltar)
     *            *exp(-b2/delD(i,iclpro,icltar))
     *            *sy**(betD(i,iclpro,icltar)+gamb+epsG)
     *            *gfactorp *gfactort)
            epsGp=epsilongp(k,i)
            epsGt=epsilongt(k,i)
            btildgp(i,k)=dble(betDp(i,iclpro,icltar)
     *                    +epsGp
     *                    +gamb-alppar)+1.d0
            btildgpp(i,k)=dble(betDpp(i,iclpro,icltar)
     *                    +epsGt
     *                    +gamb-alppar)+1.d0
          else
            absb=abs(bk(k))-bmxdif(iclpro,icltar)
            b2a=absb*absb
            atildg(i,k)=atildg(i,k)*dble(
     *                  sy**(betD(i,iclpro,icltar)+epsG)
     *                 *exp(-b2a/delD(i,iclpro,icltar)))
            btildgp(i,k)=dble(betDp(i,iclpro,icltar)-alppar+epsGp)+1.d0
          btildgpp(i,k)=dble(betDpp(i,iclpro,icltar)-alppar+epsGt)+1.d0
          endif
          coefgdp=utgam2(1.d0+dble(alplea(iclpro))+btildgp(i,k))
          coefgdt=utgam2(1.d0+dble(alplea(icltar))+btildgpp(i,k))
          atildg(i,k)=atildg(i,k)
     *            *utgam2(btildgp(i,k))*utgam2(btildgpp(i,k))
     *            /coefgdp/coefgdt
   !...........prepare plot in xepsilon
          if(irea.eq.0)then
           kk=max(1,int((bk(k)-b1)/db)+1)
           if(i.lt.2)then
             if(i.eq.0)w(0,kk)=w(0,kk)+1
             if(i.eq.0)w(1,kk)=w(1,kk)+epsGp
             if(i.eq.0)w(2,kk)=w(2,kk)+epsGt
c                     w(3+i,kk)=w(3+i,kk)+abs(epsG)
         !...5-8 soft ... 9-12 semi
             w(5+4*i,kk)=w(5+4*i,kk)
     *              +betDp(i,iclpro,icltar)   !prj eff
     *              +epsGp+gamb
             w(6+4*i,kk)=w(6+4*i,kk)
     *              +betDpp(i,iclpro,icltar)  !tgt eff
     *              +epsGt+gamb
             w(7+4*i,kk)=w(7+4*i,kk)
     *              +betDp(i,iclpro,icltar)  !prj unscr
     *              +gamb
             w(8+4*i,kk)=w(8+4*i,kk)
     *              +betDpp(i,iclpro,icltar) !tgt unscr
     *              +gamb
             if(i.eq.0)w(13,kk)=w(13,kk)+zparpro(k)
             if(i.eq.0)w(14,kk)=w(14,kk)+zpartar(k)
           else
             if(epscrd.lt.0.)then
               w(3,kk)=w(3,kk)+abs(epsG)
             else
               w(3,kk)=w(3,kk)+epsGp
               w(4,kk)=w(4,kk)+epsGt
             endif
           endif
          endif
   !................
        enddo
      enddo

!...........................................................................

           zppevt=zpav/koll
           zptevt=ztav/koll
           if(irea.eq.0)then
             ktot=int(bimp)+1
             if(ktot.le.nyeps)then
               w(15,ktot)=w(15,ktot)+zppevt
               w(16,ktot)=w(16,ktot)+zptevt
               w(17,ktot)=w(17,ktot)+1
             endif
             n=1+int(float(nglevt)/(0.1*maproj*matarg))*(nyeps-1)
             if(nglevt.ge.1.and.n.ge.1.and.n.le.nyeps)then
               w(18,n)=w(18,n)+zppevt
               w(19,n)=w(19,n)+zptevt
               w(20,n)=w(20,n)+1
             endif
           endif


!........alpha beta betap for Gpro...........................................

      if(irea.le.0)then
      do k=1,koll
        ip=iproj(k)
        it=itarg(k)

        b2=bk(k)*bk(k)
        imax=ntymx
        if(iomega.eq.2)imax=1
        do i=ntymin,imax

c          if(i.lt.2)then
c          if(i.eq.0)then
c            epsG=epsilongs(k,i)
          if(i.eq.2)then
            if(epscrd.lt.0.)then
              epsG=epsilongs(k,0)+epsilongs(k,1)
              epsGp=0.
              epsGt=0.
            else
              epsG=0.
              epsGp=epsilongs(k,0)
              epsGt=epsilongs(k,1)
            endif
          else
            epsG=0.
            epsGp=0.
            epsGt=0.
          endif
          gamb=gamD(i,iclpro,icltar)*b2

          atilde(i,k)=dble(alpD(i,iclpro,icltar))
          if(i.lt.2)then
          atilde(i,k)=atilde(i,k)*dble(
     *              exp(-b2/delD(i,iclpro,icltar))
     *              *sy**(betD(i,iclpro,icltar)
     *                    +gamb+epsG)
     *              *chad(iclpro)*chad(icltar))
            epsGp=epsilongp(k,i)
            epsGt=epsilongt(k,i)
            btildep(i,k)=dble(betDp(i,iclpro,icltar)
     *                    +epsGp
     *                    +gamb-alppar)
            btildepp(i,k)=dble(betDpp(i,iclpro,icltar)
     *                    +epsGt
     *                    +gamb-alppar)
          else
            absb=abs(bk(k))-bmxdif(iclpro,icltar)
            b2a=absb*absb
            atilde(i,k)=atilde(i,k)*dble(
     *                  sy**(betD(i,iclpro,icltar)+epsG)
     *                 *exp(-b2a/delD(i,iclpro,icltar)))

            btildep(i,k)=dble(betDp(i,iclpro,icltar)-alppar+epsGp)
            btildepp(i,k)=dble(betDpp(i,iclpro,icltar)-alppar+epsGt)
          endif

          if(btildep(i,k)+1d0.lt.-eps.or.btildepp(i,k)+1d0.lt.-eps)then
            write(ifmt,*)' k,b,ip,it,gamb,alppar',k,bk(k),ip,it,gamb
     *                                           ,alppar
            write(ifmt,*)' gammaV,epsGP1/2,epsGT1/2,epsGS1/2'
     *           ,gammaV(k),epsilongp(k,0),epsilongp(k,1)
     *     ,epsilongt(k,0),epsilongt(k,1),epsilongs(k,0),epsilongs(k,1)
            write(ifmt,*)'*******************************************'
            write(ifmt,*)" atilde,btildep,btildepp "
            do ii=ntymin,ntymx
              write(ifmt,*)ii,atilde(ii,k),btildep(ii,k),btildepp(ii,k)
            enddo
            call utstop('Error in epos-omg in GfunPark&',
     +sizeof('Error in epos-omg in GfunPark&'))
          endif
        enddo

      enddo
      endif

!...........................................................................

      if(ish.ge.10)then
      do k=1,koll
        ip=iproj(k)
        it=itarg(k)
        write(ifch,*)' k,b,ip,it,',k,bk(k),ip,it
        write(ifch,*)' zparpro,zpartar,xzcutpar'
     *      ,zparpro(k),zpartar(k),xzcutpar(k)
        write(ifch,*)' gammaV,epsilonGP1/2,epsilonGT1/2,epsilonGs1/2'
     *      ,gammaV(k),epsilongp(k,0),epsilongp(k,1)
     *      ,epsilongt(k,0),epsilongt(k,1),epsilongs(k,0),epsilongs(k,1)
        write(ifch,*)'*******************************************'
        write(ifch,*)" atilde,btildep,btildepp "
        do i=ntymin,ntymx
        write(ifch,*)i,atilde(i,k),btildep(i,k),btildepp(i,k)
        enddo
        write(ifch,*)" atildg,btildgp,btildgpp "
        do i=ntymin,ntymx
        write(ifch,*)i,atildg(i,k),btildgp(i,k),btildgpp(i,k)
        enddo
        call GfunPar(xpar7,xpar7,1,0,bk(k),sy,alp,bet,betp
     &                  ,epsp,epst,epss,gamvv)
        call GfunPar(xpar7,xpar7,1,1,bk(k),sy,alp,bet,betp
     &                  ,epsp,epst,epss,gamvv)
      enddo
      endif

      call utprjx('GfunParK ',ish,ishini,10)

      return
      end

c----------------------------------------------------------------------
      subroutine GfunPar(zzip,zzit,m,i,b,spp,alp,bet,betp,epsp,epst
     &                  ,epss,gamvv)
c----------------------------------------------------------------------
c  calculates parameters alp,bet,betp of the G functions for pp (b-mode)
c  and the screening exponents epsp,epst,epss,gamvv
c----------------------------------------------------------------------
c  zzip:additional z component (nuclear effect projectile side)
c  zzit:additional z component (nuclear effect target side)
c  m=1: profile function Gpro,  i=0: soft       i=2: diff
c  m=2: Gtilde,                 i=1: semi       (no screening for diff)
c  b: impact param, spp: pp energy squared
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      include 'epos.incems'
      parameter(nbkbin=40)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      common /kwrite/ xkapZ
      double precision utgam2,coefgdp,coefgdt,dalp,dbet,dbetp,eps
      parameter(eps=1.d-20)

      call utprj('GfunPar ',ish,ishini,10)

      ee=sqrt(spp)
c      bglaub2=FbGlaub2(ee)
      rs=r2had(iclpro)+r2had(icltar)+slopom*log(spp)
      bglaub2=4.*.0389*rs
      if(ish.ge.10)write(ifch,*)'Gf in:',m,i,b,bglaub2,spp,zzip,zzit
     &                                    ,iclpro,icltar
      b2=b*b
      cfalpro=ucfpro
      cfaltar=ucftar
      gamb=gamD(i,iclpro,icltar)*b2

      if(iscreen.ne.0)then
        absb=max(1.e-9,abs(b))
        b2a=absb*absb
        b2x=2.*epscrp*bglaub2
        zzp=epscrw*exp(-b2a/b2x)*fscra(ee/egyscr)
        zzp=min(zzp,epscrx)+zzip !saturation
        zzt=epscrw*exp(-b2a/b2x)*fscra(ee/egyscr)
        zzt=min(zzt,epscrx)+zzit !saturation

        x=zzp
        epsG=0.
        if(i.eq.1.and.spp.ge.sfshlim)then
          epsGp=max(-betDp(i,iclpro,icltar)-0.95+alppar,
     &               epscrh*x)
c     &               min(epscrx,epscrh*x))
        elseif(i.le.1)then
          epsGp=max(-betDp(i,iclpro,icltar)-0.95+alppar,
     &               epscrs*x)
c     &               min(epscrx,epscrs*x))
        else
          if(epscrd.lt.0.)then
            epsG=epsG+sign(abs(epscrd)*x,epscrd)
            epsGp=0.
          else
            epsGp=sign(abs(epscrd)*x,epscrd)
            epsG=0.
          endif
        endif
        gamV=1.
        x=zzt
        if(i.eq.1.and.spp.ge.sfshlim)then
          epsGt=max(-betDpp(i,iclpro,icltar)-0.95+alppar,
     &               epscrh*x)
c     &               min(epscrx,epscrh*x))
        elseif(i.le.1)then
          epsGt=max(-betDpp(i,iclpro,icltar)-0.95+alppar,
     &               epscrs*x)
c     &               min(epscrx,epscrs*x))
        else
          if(epscrd.lt.0.)then
            epsG=epsG+sign(abs(epscrd)*x,epscrd)
            epsGt=0.
          else
            epsGt=sign(abs(epscrd)*x,epscrd)
            epsG=0.
          endif
        endif
c        gamV=gamV
      else
        zzp=0.
        zzt=0.
        epsGp=0.
        epsGt=0.
        epsG=0.
        gamV=1.
      endif

      gfactorp=1.!+(gfactor-1)*exp(-5*b/gwidth/bglaub)
      gfactort=1.!+(gfactor-1)*exp(-5*b/gwidth/bglaub)

      rho=betD(i,iclpro,icltar)+gamb+epsG
      

      if(m.eq.1)then

        dalp=dble(alpD(i,iclpro,icltar))
        if(i.lt.2)then
          dalp=dalp
     *       *exp(min(50d0,dble(rho*log(spp)-b2/delD(i,iclpro,icltar))))
          dbet=dble(betDp(i,iclpro,icltar)
     *         +epsGp
     *         +gamb-alppar)
          dbetp=dble(betDpp(i,iclpro,icltar)
     *         +epsGt
     *         +gamb-alppar)
        else
          absb=abs(b)-bmxdif(iclpro,icltar)
          b2a=absb*absb
          dalp=dalp
     *       *exp(min(50d0,dble((betD(i,iclpro,icltar)+epsG)*log(spp)
     *            -b2a/delD(i,iclpro,icltar))))
          dbet=dble(betDp(i,iclpro,icltar)-alppar+epsGp)
          dbetp=dble(betDpp(i,iclpro,icltar)-alppar+epsGt)
        endif

        if((dbet+1.d0).lt.-eps.or.(dbetp+1.d0).lt.-eps)then
          write(*,*)'m,i,b,spp,alp,bet,betp',m,i,b,spp,dalp,dbet,dbetp
          call utstop('Error : beta < -1 in Gfunpar in epos-omg&',
     +sizeof('Error : beta < -1 in Gfunpar in epos-omg&'))
        endif

      elseif(m.eq.2)then
        xkappa=1.
c        if(i.eq.0.and.b.lt.(nbkbin-1)*bkbin)then
        if(b.lt.(nbkbin-1)*bkbin)then
          ibk=int(b/bkbin)+1
          if(isetcs.gt.1.and.iclegy.lt.iclegy2)then
            egy0=egylow*egyfac**float(iclegy-1)
            xkappa1=xkappafit(iclegy,iclpro,icltar,ibk)
     *         +(b-bkbin*float(ibk-1))/bkbin
     *           *(xkappafit(iclegy,iclpro,icltar,ibk+1)
     *            -xkappafit(iclegy,iclpro,icltar,ibk))
            xkappa2=xkappafit(iclegy+1,iclpro,icltar,ibk)
     *         +(b-bkbin*float(ibk-1))/bkbin
     *           *(xkappafit(iclegy+1,iclpro,icltar,ibk+1)
     *            -xkappafit(iclegy+1,iclpro,icltar,ibk))
            xkappa=xkappa1+(xkappa2-xkappa1)/log(egyfac)
     *         *(log(ee)-log(egy0))
            xkappa=facmc*xkappa
          else
            xkappa=xkappafit(iclegy,iclpro,icltar,ibk)
     *         +(b-bkbin*float(ibk-1))/bkbin
     *           *(xkappafit(iclegy,iclpro,icltar,ibk+1)
     *            -xkappafit(iclegy,iclpro,icltar,ibk))
            xkappa=facmc*xkappa
          endif
          xkapZ=xkappa
        endif

        dalp=dble(alpD(i,iclpro,icltar)
     *        *cfalpro*cfaltar
     *        *gamV)
c        if(i.eq.0)alp=alp
        dalp=dalp
     *        *dble(xkappa)*dble(xkappa)

        if(i.lt.2)then
          dalp=dalp
     *     *exp(min(50d0,dble(rho*log(spp)-b2/delD(i,iclpro,icltar))))
     *        *dble(chad(iclpro)*chad(icltar)
     *        *gfactorp *gfactort)
          dbet=dble(betDp(i,iclpro,icltar)
     *        +epsGp
     *        +gamb-alppar+1.)
          dbetp=dble(betDpp(i,iclpro,icltar)
     *        +epsGt
     *        +gamb-alppar+1.)
        else
          absb=abs(b)-bmxdif(iclpro,icltar)
          b2a=absb*absb
          dalp=dalp
     *     *exp(min(50d0,dble((betD(i,iclpro,icltar)+epsG)*log(spp)
     *          -b2a/delD(i,iclpro,icltar))))
          dbet=dble(betDp(i,iclpro,icltar)-alppar+1.+epsGp)
          dbetp=dble(betDpp(i,iclpro,icltar)-alppar+1.+epsGt)
        endif
        coefgdp=utgam2(1.d0+dble(alplea(iclpro))+dbet)
        coefgdt=utgam2(1.d0+dble(alplea(icltar))+dbetp)
        dalp=dalp*utgam2(dbet)*utgam2(dbetp)/coefgdp/coefgdt
      else

        stop'GproPar: wrong m value.              '

      endif


      alp=sngl(dalp)
      bet=sngl(dbet)
      betp=sngl(dbetp)
      epsp=epsGp
      epst=epsGt
      epss=epsG
      gamvv=gamV

      alpUni(i,m)=dalp
      betUni(i,m)=dbet
      betpUni(i,m)=dbetp


      if(ish.ge.10)write(ifch,*)'   GfunPar :',alp,bet,betp,epsp,epst
     &                                    ,epss,gamvv

      call utprjx('GfunPar ',ish,ishini,10)
      end

c----------------------------------------------------------------------
      subroutine GfomPar(b,spp)
c----------------------------------------------------------------------
c  calculates parameters of the fom functions for pp (b-mode)
c----------------------------------------------------------------------
c  b: impact param, spp: pp energy squared
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      include 'epos.incems'

      call utprj('GfomPar ',ish,ishini,6)

      ee=sqrt(spp)
      rs=r2had(iclpro)+r2had(icltar)+slopom*log(spp)
      bglaub2=4.*.0389*rs

      if(iscreen.ne.0)then
        absb=max(1.e-9,abs(b))
        b2a=absb*absb
        b2x=2.*epscrp*bglaub2
        zzp=(epscrw*exp(-b2a/b2x))*fscra(ee/egyscr)
        zzp=min(zzp,epscrx) !saturation
        zzt=(epscrw*exp(-b2a/b2x))*fscra(ee/egyscr)
        zzt=min(zzt,epscrx) !saturation
      else
        zzp=0.
        zzt=0.
      endif


      z0=alpfomi!*epscrw*fscra(ee/egyscr)
      if(z0.gt.0.)then
        z1=zzp
        zzpUni=dble(z1**gamfom/z0)*exp(-dble(b*b/delD(1,iclpro,icltar)))
c      zzpUni=dble(4.*z0*(z1/z0)**1.5)
        z1=zzt
        zztUni=dble(z1**gamfom/z0)*exp(-dble(b*b/delD(1,iclpro,icltar)))
c      zztUni=dble(4.*z0*(z1/z0)**1.5)
      else
        zzpUni=0d0
        zztUni=0d0
      endif

      if(ish.ge.6)write(ifch,*)'   GfomPar :',zzpUni,zztUni

      call utprjx('GfomPar ',ish,ishini,6)
      end

c----------------------------------------------------------------------
      function fscra(x)
c----------------------------------------------------------------------
      fscra=0
      x2=x*x
      if(x2.gt.1.)fscra=log(x2)!**2
      end

c----------------------------------------------------------------------
      function fscro(x,rho)
c----------------------------------------------------------------------
      include 'epos.incpar'
      fscro=znurho*rho
      x2=x*x
c      if(x2.gt.1.)fscro=sqrt(log(x2)**2+fscro**2)
      if(x2.gt.1.)fscro=log(x2)*(1.+fscro)
      end

c----------------------------------------------------------------------
      function FbGlaub2(x)
c----------------------------------------------------------------------
c  calculates (glauber radius)^2 from pp cross section (data fit)
c(estimated if not already calculated --> not any more to have smoother xs)
c----------------------------------------------------------------------
c  x: pp energy
c----------------------------------------------------------------------

      include 'epos.inc'

c      if(sigine.eq.0.)then
        if(iclpro+icltar.eq.3)then !pi+p
          siginex=20.+0.08*log(x)**3.-0.004*log(x)**4.
        elseif(iclpro+icltar.eq.5)then !K+p
          siginex=16.+0.08*log(x)**3.-0.004*log(x)**4.
        else
          siginex=30.+0.095*log(x)**3.-0.004*log(x)**4.
        endif
c      else
c       siginex=sigine
c      endif
      FbGlaub2=siginex/10./pi

      return
      end

c----------------------------------------------------------------------
      subroutine recalcZPtn !???????????? not updated !!!
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
       stop'recalcZPtn not valid any more!!!!!!!'
c      if(koll.eq.1.and.maproj.eq.1.and.matarg.eq.1)then
c       npom=nprt(1)
c       k=1
c       ip=iproj(1)
c       it=itarg(1)
c       zparpro(k)=max(0,npom-1)*0.2
c       zpartar(k)=0
c       zpartar(k)=max(0,npom-1)*0.2
c       ztav=zpartar(k)
c       zpav=zparpro(k)
c       zppevt=zpav
c       zptevt=ztav
c      endif
      end

c----------------------------------------------------------------------
      double precision function om1(xh,yp,b)   !---test---
c----------------------------------------------------------------------
c om1 = G * C * gamd    (C and gamd usually 1)
c xh - fraction of the energy squared s for the pomeron;
c b - impact parameter between the pomeron ends;
c yp - rapidity for the pomeron;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision Gf,xp,xm,xh,yp

      Gf=0.d0
      xp=sqrt(xh)*exp(yp)
      xm=xh/xp
      spp=engy**2
      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
        Gf=Gf+dble(alp)*xp**dble(bet)*xm**dble(betp)
      enddo
      om1=Gf
     *  * dble(chad(iclpro)*chad(icltar))
      end

c----------------------------------------------------------------------
      double precision function om1intb(b)   !---test---
c----------------------------------------------------------------------
c  om1 integrated over xp and xm for given b
c  Calculation by analytical integration
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision cint,cint2,eps
      parameter(eps=1.d-20)

      spp=engy*engy
      imax=idxD1
      if(iomega.eq.2)imax=1
      cint=0.d0
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
        cint2=dble(gamv*alp)
        if((bet+1.0).gt.eps)then
          cint2=cint2/dble(bet+1.0)
        else
          cint2=-cint2*log(xminDf)
        endif
        if((betp+1.0).gt.eps)then
          cint2=cint2/dble(betp+1.0)
        else
          cint2=-cint2*log(xminDf)
        endif
        cint=cint+cint2
      enddo

      if(cint.lt.-eps)then
        write(*,*) 'WARNING ! om1intb in epos-omg is <0 !!!!!'
        write(*,*) 'WARNING ! => om1intb set to 1e-3 !!!!!'
        write(*,*) 'WARNING ! => bmax=3.5 fm !!!!!'
        cint=1.d-3
      endif

      om1intb=cint
     *       *dble(chad(iclpro)*chad(icltar))

      return
      end

c----------------------------------------------------------------------
      double precision function om1intbk(k)   !---MC---
c----------------------------------------------------------------------
c  Diffractive part of om1 integrated over xp and xm for given pair k
c  Calculation by analytical integration
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'
      double precision cint,cint2,eps,bet,betp
      parameter(eps=1.d-20)

      imax=idxD1
      if(iomega.eq.2)imax=1
      om1intbk=0d0
      cint=0
      do i=idxDmin,imax
        bet=btildep(i,k)
        betp=btildepp(i,k)
        cint2=atilde(i,k)
        if((bet+1.d0).gt.eps)then
          cint2=cint2/(bet+1.d0)
        else
          cint2=-cint2*log(xminDf)
        endif
        if((betp+1.d0).gt.eps)then
          cint2=cint2/(betp+1.d0)
        else
          cint2=-cint2*log(xminDf)
        endif
        cint=cint+cint2
      enddo

      if(cint.lt.-eps)then
        write(*,*) 'WARNING ! om1intbk in epos-omg is <0 !!!!!'
        write(*,*) 'WARNING ! => om1intbk set to 1e-3 !!!!!'
        write(*,*) 'WARNING ! => bmax=3.5 fm !!!!!'
        cint=1.d-3
      endif

      om1intbk=cint
     *       *dble(chad(iclpro)*chad(icltar))

      return
      end

c----------------------------------------------------------------------
      double precision function om1intbi(b,iqq)   !---MC---
c----------------------------------------------------------------------
c  om1 integrated over xp and xm for given b
c  Calculation by analytical integration of contribution iqq
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision eps,cint
      parameter(eps=1.d-20)

      spp=engy*engy
      call Gfunpar(0.,0.,1,iqq,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
      cint=dble(gamv*alp)
      if(dble(bet+1.0).gt.eps)then
        cint=cint/dble(bet+1.0)
      else
        cint=-cint*log(xminDf)
      endif
      if(dble(betp+1.0).gt.eps)then
        cint=cint/dble(betp+1.0)
      else
        cint=-cint*log(xminDf)
      endif
      if(cint.lt.-eps)then
        write(*,*) 'WARNING ! om1intbi in epos-omg is <0 !!!!!'
        write(*,*) 'WARNING ! => om1intbi set to 1e-3 !!!!!'
        write(*,*) 'WARNING ! => bmax=3.5 fm !!!!!'
        cint=1.d-3
      endif

      om1intbi=cint
     *       *dble(chad(iclpro)*chad(icltar))

      return
      end

c----------------------------------------------------------------------
      double precision function om1intbc(b)   !---MC---
c----------------------------------------------------------------------
c  om1*F*F integrated over xp and xm for given b
c  Calculation by analytical integration
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      double precision cint,gamom,deltap,deltam
      double precision utgam2,Fp,Fm

      spp=engy**2
      om1intbc=0.d0
      Fp=dble(ucfpro)   !gamma(1+alplea)
      Fm=dble(ucftar)

      imax=idxD1
      if(iomega.eq.2)imax=1

      cint=0.d0

      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
       gamom=dble(alp*gamv*chad(iclpro)*chad(icltar))
        deltap=dble(bet)
        deltam=dble(betp)
        cint=cint+gamom*utgam2(deltap+1.d0)*utgam2(deltam+1.d0)
     &            /utgam2(2.d0+deltap+dble(alplea(iclpro)))
     &            /utgam2(2.d0+deltam+dble(alplea(icltar)))
      enddo

      om1intbc=cint*Fp*Fm

      if(om1intbc.lt.-1.d-10)then
        write(*,*) 'WARNING ! om1intbc in epos-omg is <0 !!!!!'
        write(*,*) 'WARNING ! => om1intbc set to 0. !!!!!'
        om1intbc=0.d0
      endif

      return
      end

cc----------------------------------------------------------------------
c      double precision function om1intbci(b,iqq)   !---MC---
cc----------------------------------------------------------------------
cc  om1*F*F integrated over xp and xm for given b and given Pomeron type iqq
cc  Calculation by analytical integration
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      include 'epos.incems'
c      include 'epos.incsem'
c      double precision cint,gamom,deltap,deltam
c      double precision utgam2,Fp,Fm,eps
c
c      spp=engy**2
c      om1intbci=0.d0
c      Fp=dble(ucfpro)   !gamma(1+alplea)
c      Fm=dble(ucftar)
c
c      i=iqq
c      call Gfunpar(0.,0.,1,i,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
c      gamom=dble(alp*gamv)*chad(iclpro)*chad(icltar)
c      deltap=dble(bet)
c      deltam=dble(betp)
c      cint=gamom*utgam2(deltap+1.d0)*utgam2(deltam+1.d0)
c     &            /utgam2(2.d0+deltap+dble(alplea(iclpro)))
c     &            /utgam2(2.d0+deltam+dble(alplea(icltar)))
c
c      om1intbci=cint*Fp*Fm
c
c      if(om1intbci.lt.-1.d-10)then
c        write(*,*) 'WARNING ! om1intbci in epos-omg is <0 !!!!!'
c        write(*,*) 'WARNING ! => om1intbci set to 0. !!!!!'
c        om1intbci=0.d0
c      endif
c
c      return
c      end
c
c----------------------------------------------------------------------
      double precision function om1intgck(k,xprem,xmrem)   !---MC---
c----------------------------------------------------------------------
c  om1*(xprem-xp)*(xmrem-xm) integrated over xp and xm for given k
c  Calculation by analytical integration
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      double precision cint,gamom,deltap,deltam,xprem,xmrem

      om1intgck=0.d0

      imax=idxD1
      if(iomega.eq.2)imax=1

      cint=0.d0
      do i=idxDmin,imax
        gamom=dble(atilde(i,k))
        deltap=dble(btildep(i,k))
        deltam=dble(btildepp(i,k))
        cint=cint+gamom/(deltap+1.d0)/(deltam+1.d0)
     &            /(2.d0+deltap) /(2.d0+deltam)
     &            *xprem**(deltap+2.d0)
     &            *xmrem**(deltam+2.d0)
      enddo
      om1intgck=cint

      return
      end

c----------------------------------------------------------------------
      double precision function om1intgc(b)   !---test---
c----------------------------------------------------------------------
c  om1*(1-xp)*(1-xm) integrated over xp and xm for given b
c  Calculation by analytical integration
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incsem'
      double precision cint,gamom,deltap,deltam,eps
      parameter(eps=1.d-20)

      spp=engy**2
      om1intgc=0.d0

      imax=idxD1
      if(iomega.eq.2)imax=1



      cint=0.d0

      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,spp,alp,bet,betp,epsp,epst,epss,gamv)
        gamom=dble(alp*gamv*chad(iclpro)*chad(icltar))
        deltap=dble(bet)
        deltam=dble(betp)
        if((deltap+1.d0).gt.eps)then
          gamom=gamom/(deltap+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        if((deltam+1.d0).gt.eps)then
          gamom=gamom/(deltam+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        cint=cint+gamom /(2.d0+deltap) /(2.d0+deltam)
      enddo
      om1intgc=cint

      if(om1intgc.lt.eps)then
        write(*,*) b,deltap,deltam,gamom
        write(*,*) 'WARNING ! om1intgc in epos-omg is <0 !!!!!'
        write(*,*) 'WARNING ! => om1intgc set to 0. !!!!!'
        om1intgc=0.d0
      endif

      return
      end


c----------------------------------------------------------------------
        subroutine integom1(irea)
c----------------------------------------------------------------------
c  om1 integrated over xp and xm for  all b=bk(k) if irea=0
c  result written to :
c    om1int(k)   = om1intb(bk(k))      = \int om1
c    om1intc(k)  = om1intgc(bk(k)...   = \int om1*(1-xp)*(1-xm)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      double precision om1intbk,om1intgck,PomIncExactk

      if(irea.le.0)then


        do k=1,koll

          om1int(k)=om1intbk(k) !only diffractive contribution
          om1intc(k)=om1intgck(k,1d0,1d0)

        enddo

        if(irea.eq.0.and.zrminc.gt.0..and.xzcut.gt.0.)then
          do k=1,koll
            PomInck(k)=PomIncExactk(k)
          enddo
        endif

      endif


      return
      end


c----------------------------------------------------------------------
      double precision function om1xpk(xp,xpr1i,k)   !---test---
c----------------------------------------------------------------------
c \int dxm om1*(1-xp)*(1-xm)   (normalized)
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      double precision xp,gamomx(ntymi:ntymx),cint,gamom
      double precision deltap(ntymi:ntymx),deltam(ntymi:ntymx),eps
     &                 ,xpr1,xmr1,xpr1i
      parameter(eps=1.d-20)

      om1xpk=0.d0
      if(xp.ge.xpr1i)return

      xpr1=1.d0
      xmr1=1.d0
      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
          deltap(i)=btildep(i,k)
          deltam(i)=btildepp(i,k)
          gamomx(i)=atilde(i,k)*xmr1**(deltam(i)+2.d0)/(2.d0+deltam(i))
          if((deltam(i)+1.d0).gt.eps)then
            gamomx(i)=gamomx(i)/(deltam(i)+1.d0)
          else
            gamomx(i)=-gamomx(i)*log(xminDf)
          endif
      enddo


      cint=0.d0
      do i=imin,imax
          gamom=gamomx(i)*xpr1**(deltap(i)+2.d0)/(2.d0+deltap(i))
          if((deltap(i)+1.d0).gt.eps)then
            gamom=gamom/(deltap(i)+1.d0)
          else
            gamom=-gamom*log(xminDf)
          endif
          cint=cint+gamom
      enddo


      do i=imin,imax
        om1xpk=om1xpk+gamomx(i)*xp**deltap(i)
     &                       *(xpr1-xp)

      enddo

      om1xpk=om1xpk/cint

      return
      end


c----------------------------------------------------------------------
      double precision function om1xmk(xp,xm,xpr1i,xmr1i,k)   !---test---
c----------------------------------------------------------------------
c om1(xp,xm)*(1-xp)*(1-xm)   (normalized for fixed xp)
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      double precision xp,xm,gamomx(ntymi:ntymx),cint,gamom
      double precision deltam(ntymi:ntymx),eps,xpr1,xmr1,xpr1i,xmr1i
      parameter(eps=1.d-20)

      om1xmk=0.d0
      if(xp.ge.xpr1i)return
      if(xm.ge.xmr1i)return
      xpr1=1.d0
      xmr1=1.d0

      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
          gamomx(i)=atilde(i,k)*xp**btildep(i,k)*(xpr1-xp)
          deltam(i)=btildepp(i,k)
      enddo

      cint=0.d0
      do i=imin,imax
          gamom=gamomx(i)*xmr1**(deltam(i)+2.d0)/(2.d0+deltam(i))
          if((deltam(i)+1.d0).gt.eps)then
            gamom=gamom/(deltam(i)+1.d0)
          else
            gamom=-gamom*log(xminDf)
          endif
          cint=cint+gamom
      enddo


      do i=imin,imax
        om1xmk=om1xmk+gamomx(i)*xm**deltam(i)*(xmr1-xm)
      enddo

      om1xmk=om1xmk/cint

      return
      end

c----------------------------------------------------------------------
      double precision function om1xpr(atil,btilp,btilpp
     &                                   ,xpremi,xmremi,ir)   !---MC---
c----------------------------------------------------------------------
c Random number generated from the function om1xpk. We solve the equation
c which give om1xprk by Newton-Raphson + secant method.
c k - pair indice;
c ir - 1 to get xp, -1 to get xm (be carrefull to inverse xpremi et xmremi
c when calling with ir=-1)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      double precision x,x0,x1,gamomx(ntymi:ntymx),eps,f0t,f1t,f00
      double precision xt,fx,fpx,r,f1,f0,cint,deltx,prec,drangen,xmrem
      double precision deltap(ntymi:ntymx),deltam(ntymi:ntymx),xprem
      parameter (eps=1.d-20)
      double precision xpremi,xmremi,xlmin,atil(ntymi:ntymx)
     &                ,btilp(ntymi:ntymx),btilpp(ntymi:ntymx)


      om1xpr=0.d0
      if(xpremi.gt.1d0.or.xmremi.gt.1d0)return
      x0=log(xminDf)
      x1=log(xpremi)
      xprem=xpremi
      xmrem=xmremi
      imin=ntymin
      imax=ntymx
      xlmin=1.d0
      xt=0d0
      if(iomega.eq.2)imax=1

      do i=imin,imax
        if(ir.gt.0)then
          deltap(i)=btilp(i)
          deltam(i)=btilpp(i)
        else
          deltap(i)=btilpp(i)
          deltam(i)=btilp(i)
        endif
        gamomx(i)=atil(i)*xmrem**(deltam(i)+2.d0)/(2.d0+deltam(i))
        if((deltam(i)+1.d0).gt.eps)then
          gamomx(i)=gamomx(i)/(deltam(i)+1.d0)
        else
          xlmin=log(xminDf)
          gamomx(i)=-gamomx(i)*xlmin
        endif
      enddo

      f0=0.d0
      f1=0.d0
      do i=imin,imax
        if((deltap(i)+1.d0).gt.eps)then
          f0=f0+gamomx(i)
     &          *(xprem*exp(x0)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x0)**(2.d0+deltap(i))/(2.d0+deltap(i)))
          f1=f1+gamomx(i)
     &          *(xprem*exp(x1)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x1)**(2.d0+deltap(i))/(2.d0+deltap(i)))
        else
          xlmin=log(xminDf)
          f0=f0+gamomx(i)*(xprem*(x0-xlmin)-exp(x0)+xminDf)
          f1=f1+gamomx(i)*(xprem*(x1-xlmin)-exp(x1)+xminDf)
        endif
      enddo
      f00=f0
      cint=f1-f00
      f0=-(f0-f00)/cint
      f1=-(f1-f00)/cint
      ntry=0
 11   ntry=ntry+1
      r=drangen(dble(ntry))
      f0t=f0+r
      f1t=f1+r
      if(f1t*f0t.ge.eps.and.ntry.lt.100)goto 11
      if(f1t*f0t.ge.eps)then
        do i=imin,imax
         write(ifmt,*)i,gamomx(i),deltap(i),deltam(i)
        enddo
        write(ifmt,*)x0,f0,f0t,x1,f1,f1t,r,cint,ntry
        call utstop('om1xpr (2)&',
     +sizeof('om1xpr (2)&'))
      endif
      f0=f0t
      f1=f1t
      if(abs(f0).le.eps) then
        om1xpr=exp(x0)
        return
      endif
      if(abs(f1).le.eps) then
        om1xpr=exp(x1)
        return
      endif
      x=0.5d0*(x1+x0)
      deltx=abs(x1-x0)


      ntry=0

 111  continue
      if(ntry.le.1000)then
      fx=0.d0
      fpx=0.d0
      do i=imin,imax
        if((deltap(i)+1.d0).gt.eps)then
          fx=fx+gamomx(i)
     &          *(xprem*exp(x)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x)**(2.d0+deltap(i))/(2.d0+deltap(i)))
          fpx=fpx+gamomx(i)*exp(x)**deltap(i)*(xprem-exp(x))
        else
          fx=fx+gamomx(i)*(xprem*(x-xlmin)-exp(x)+xminDf)
          fpx=fpx+gamomx(i)*(xprem/exp(x)-1.d0)
        endif
      enddo
      fx=-(fx-f00)/cint+r
      fpx=fpx/cint
      xt=x-fx/fpx

      if (f0*fx.lt.0.D0) then
        f1=fx
        x1=x
      else
        f0=fx
        x0=x
      endif
      if ((xt.lt.x0.or.xt.gt.x1).and.abs(f1-f0).gt.eps) then
        xt=x1-f1*(x1-x0)/(f1-f0)
      endif

       else

        write(ifmt,*)'Warning in om1xpr, to much try !'

      endif


      if(abs(x-xt).gt.deltx*0.5d0) then
        xt=(x1+x0)*0.5D0
      endif
      deltx=abs(x-xt)
      if(abs(x).gt.eps)then
        prec=abs(deltx/x)
      else
        prec=0d0
        call utstop('Problem in om1xpr&',
     +sizeof('Problem in om1xpr&'))
      endif

      if (prec.gt.1.d-3.and.abs(f1-f0).gt.eps.and.ntry.le.1000) then
         x=xt
         ntry=ntry+1
         goto 111
      endif

      om1xpr=exp(x)

      return
      end

c----------------------------------------------------------------------
      double precision function om1xprk(k,xpremi,xmremi,ir)   !---MC---
c----------------------------------------------------------------------
c Random number generated from the function om1xpk. We solve the equation
c which give om1xprk by Newton-Raphson + secant method.
c k - pair indice;
c ir - 1 to get xp, -1 to get xm (be carrefull to inverse xpremi et xmremi
c when calling with ir=-1)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      double precision x,x0,x1,gamomx(ntymi:ntymx),eps,f0t,f1t,f00
      double precision xt,fx,fpx,r,f1,f0,cint,deltx,prec,drangen,xmrem
      double precision deltap(ntymi:ntymx),deltam(ntymi:ntymx),xprem
      parameter (eps=1.d-20)
      double precision xpremi,xmremi,xlmin


      om1xprk=0.d0
      x0=log(xmremi)
      x1=log(xpremi)
      xprem=1.d0
      xmrem=1.d0
      imin=ntymin
      imax=ntymx
      xlmin=1.d0
      xt=0d0
      if(iomega.eq.2)imax=1

      do i=imin,imax
        if(ir.gt.0)then
          deltap(i)=btildep(i,k)
          deltam(i)=btildepp(i,k)
        else
          deltap(i)=btildepp(i,k)
          deltam(i)=btildep(i,k)
        endif
        gamomx(i)=atilde(i,k)*xmrem**(deltam(i)+2.d0)/(2.d0+deltam(i))
        if((deltam(i)+1.d0).gt.eps)then
          gamomx(i)=gamomx(i)/(deltam(i)+1.d0)
        else
          xlmin=log(xminDf)
          gamomx(i)=-gamomx(i)*xlmin
        endif
      enddo

      f0=0.d0
      f1=0.d0
      do i=imin,imax
        if((deltap(i)+1.d0).gt.eps)then
          f0=f0+gamomx(i)
     &          *(xprem*exp(x0)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x0)**(2.d0+deltap(i))/(2.d0+deltap(i)))
          f1=f1+gamomx(i)
     &          *(xprem*exp(x1)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x1)**(2.d0+deltap(i))/(2.d0+deltap(i)))
        else
          xlmin=log(xminDf)
          f0=f0+gamomx(i)*(xprem*(x0-xlmin)-exp(x0)+xminDf)
          f1=f1+gamomx(i)*(xprem*(x1-xlmin)-exp(x1)+xminDf)
        endif
      enddo
      f00=f0
      cint=f1-f00
      f0=-(f0-f00)/cint
      f1=-(f1-f00)/cint
      ntry=0
 11   ntry=ntry+1
      r=drangen(dble(ntry))
      f0t=f0+r
      f1t=f1+r
      if(f1t*f0t.ge.eps.and.ntry.lt.100)goto 11
      if(f1t*f0t.ge.eps)then
        do i=imin,imax
         write(ifmt,*)i,gamomx(i),deltap(i),deltam(i)
        enddo
        write(ifmt,*)x0,f0,f0t,x1,f1,f1t,r,cint,ntry,bk(k),k
        call utstop('om1xprk (2)&',
     +sizeof('om1xprk (2)&'))
      endif
      f0=f0t
      f1=f1t
      if(abs(f0).le.eps) then
        om1xprk=exp(x0)
        return
      endif
      if(abs(f1).le.eps) then
        om1xprk=exp(x1)
        return
      endif
      x=0.5d0*(x1+x0)
      deltx=abs(x1-x0)


      ntry=0

 111  continue
      if(ntry.le.1000)then
      fx=0.d0
      fpx=0.d0
      do i=imin,imax
        if((deltap(i)+1.d0).gt.eps)then
          fx=fx+gamomx(i)
     &          *(xprem*exp(x)**(1.d0+deltap(i))/(1.d0+deltap(i))
     &           -exp(x)**(2.d0+deltap(i))/(2.d0+deltap(i)))
          fpx=fpx+gamomx(i)*exp(x)**deltap(i)*(xprem-exp(x))
        else
          fx=fx+gamomx(i)*(xprem*(x-xlmin)-exp(x)+xminDf)
          fpx=fpx+gamomx(i)*(xprem/exp(x)-1.d0)
        endif
      enddo
      fx=-(fx-f00)/cint+r
      fpx=fpx/cint
      xt=x-fx/fpx

      if (f0*fx.lt.0.D0) then
        f1=fx
        x1=x
      else
        f0=fx
        x0=x
      endif
      if ((xt.lt.x0.or.xt.gt.x1).and.abs(f1-f0).gt.eps) then
        xt=x1-f1*(x1-x0)/(f1-f0)
      endif

       else

        write(ifmt,*)'Warning in om1xprk, to much try !'

      endif


      if(abs(x-xt).gt.deltx*0.5d0) then
        xt=(x1+x0)*0.5D0
      endif
      deltx=abs(x-xt)
      if(abs(x).gt.eps)then
        prec=abs(deltx/x)
      else
        prec=0d0
        call utstop('Problem in om1xprk&',
     +sizeof('Problem in om1xprk&'))
      endif

      if (prec.gt.1.d-3.and.abs(f1-f0).gt.eps.and.ntry.le.1000) then
         x=xt
         ntry=ntry+1
         goto 111
      endif

      om1xprk=exp(x)

      return
      end

c----------------------------------------------------------------------
      double precision function om1xmrk(k,xp,xpremi,xmremi,ir)   !---MC---
c----------------------------------------------------------------------
c Random number generated from the function om1xmk. We solve the equation
c which give om1xmrk by Newton-Raphson + secant method.
c k - pair indice;
c ir - 1 to get xm, -1 to get xp (be carrefull to inverse xpremi et xmremi
c when calling with ir=-1)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      double precision x,x0,x1,gamomx(ntymi:ntymx),eps,xp,f0t,f1t
      double precision xt,fx,fpx,r,f1,f0,cint,deltx,prec,f00
      double precision deltam(ntymi:ntymx),drangen,xprem,xmrem
      double precision xpremi,xmremi,xlmin
      parameter (eps=1.d-20)

      om1xmrk=0.d0
c      if(xp.ge.xpremi)return

      xprem=1.d0
      xmrem=1.d0
      x0=log(xpremi)
      x1=log(xmremi)
      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
        if(ir.gt.0)then
          gamomx(i)=atilde(i,k)*xp**btildep(i,k)*(xprem-xp)
          deltam(i)=btildepp(i,k)
        else
          gamomx(i)=atilde(i,k)*xp**btildepp(i,k)*(xprem-xp)
          deltam(i)=btildep(i,k)
        endif
      enddo



      f0=0.d0
      f1=0.d0
      xlmin=0.d0
      do i=imin,imax
        if(abs(deltam(i)+1.d0).gt.eps)then
          f0=f0+gamomx(i)
     &          *(xmrem*exp(x0)**(1.d0+deltam(i))/(1.d0+deltam(i))
     &           -exp(x0)**(2.d0+deltam(i))/(2.d0+deltam(i)))
          f1=f1+gamomx(i)
     &          *(xmrem*exp(x1)**(1.d0+deltam(i))/(1.d0+deltam(i))
     &           -exp(x1)**(2.d0+deltam(i))/(2.d0+deltam(i)))
        else
          xlmin=log(xminDf)
          f0=f0+gamomx(i)*(xmrem*(x0-xlmin)-exp(x0)+xminDf)
          f1=f1+gamomx(i)*(xmrem*(x1-xlmin)-exp(x1)+xminDf)
        endif
      enddo
      f00=f0
      cint=f1-f00
      f0=-(f0-f00)/cint
      f1=-(f1-f00)/cint
      ntry=0
 11   ntry=ntry+1
      r=drangen(dble(ntry))
      f0t=f0+r
      f1t=f1+r
      if(f1t*f0t.ge.eps.and.ntry.lt.100)goto 11
      if(f1t*f0t.ge.eps)then
        write(ifmt,*)x0,f0,f0t,x1,f1,f1t,r,cint,ntry
        call utstop('Error(2) in epos-omg in om1xmrk&',
     +sizeof('Error(2) in epos-omg in om1xmrk&'))
      endif
      f0=f0t
      f1=f1t
      if(abs(f0).lt.eps) then
        om1xmrk=exp(x0)
        return
      endif
      if(abs(f1).lt.eps) then
        om1xmrk=exp(x1)
        return
      endif
      x=0.5d0*(x1+x0)
      deltx=abs(x1-x0)


      ntry=0
      xt=0d0

 111  continue
      if(ntry.le.1000)then
      fx=0.d0
      fpx=0.d0
      do i=imin,imax
        if(abs(deltam(i)+1.d0).gt.eps)then
          fx=fx+gamomx(i)
     &          *(xmrem*exp(x)**(1.d0+deltam(i))/(1.d0+deltam(i))
     &           -exp(x)**(2.d0+deltam(i))/(2.d0+deltam(i)))
          fpx=fpx+gamomx(i)*exp(x)**deltam(i)*(xmrem-exp(x))
        else
          fx=fx+gamomx(i)*(xmrem*(x-xlmin)-exp(x)+xminDf)
          fpx=fpx+gamomx(i)*(xmrem/exp(x)-1.d0)
        endif
      enddo
      fx=-(fx-f00)/cint+r
      fpx=fpx/cint
      xt=x-fx/fpx

      if (f0*fx.lt.-eps) then
        f1=fx
        x1=x
      else
        f0=fx
        x0=x
      endif
      if ((xt.lt.x0-eps.or.xt.gt.x1+eps).and.abs(f1-f0).gt.eps) then
        xt=x1-f1*(x1-x0)/(f1-f0)
      endif

       else

        write(ifmt,*)'Warning in om1xmrk, to much try !'

      endif

      if(abs(x-xt).gt.deltx*0.5d0) then
        xt=(x1+x0)*0.5D0
      endif
      deltx=abs(x-xt)
      if(abs(x).gt.eps)then
        prec=abs(deltx/x)
      else
        prec=0d0
        call utstop('Problem in om1xmrk&',
     +sizeof('Problem in om1xmrk&'))
      endif

      if (prec.gt.1.d-3.and.abs(f1-f0).gt.eps.and.ntry.le.1000) then
         x=xt
         ntry=ntry+1
         goto 111
      endif

      om1xmrk=exp(x)

      return
      end

c----------------------------------------------------------------------
      double precision function om1xk(xh,k)   !---test---
c----------------------------------------------------------------------
c \int dxp om1   (normalised)
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'

      double precision xh,gamomx(ntymi:ntymx),cint,alpp(ntymi:ntymx)
     &,delta(ntymi:ntymx),deltap(ntymi:ntymx),deltam(ntymi:ntymx),eps
     &,gamom
      parameter(eps=1.d-20)


      om1xk=0.d0

      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
        gamomx(i)=atilde(i,k)
        deltap(i)=btildep(i,k)
        deltam(i)=btildepp(i,k)

        delta(i)=(deltap(i)+deltam(i))*0.5d0
        alpp(i)=deltap(i)-deltam(i)

      enddo

      cint=0.d0
      do i=imin,imax
        gamom=gamomx(i)
        if((deltap(i)+1.d0).gt.eps)then
          gamom=gamom/(deltap(i)+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        if((deltam(i)+1.d0).gt.eps)then
          gamom=gamom/(deltam(i)+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        cint=cint+gamom
      enddo


      do i=imin,imax
        if(abs(alpp(i)).gt.eps)then
        om1xk=om1xk+gamomx(i)/alpp(i)*xh**deltam(i)*(1.d0-xh**alpp(i))
        else
        om1xk=om1xk-gamomx(i)*xh**delta(i)*log(xh)
        endif
      enddo

      om1xk=om1xk/cint

      return
      end

c----------------------------------------------------------------------
      double precision function om1yk(xh,yp,k)   !---test---
c----------------------------------------------------------------------
c om1 normalized for fixed xp
c xh - fraction of the energy squared s for the pomeron;
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision xh,yp,gamomy(ntymi:ntymx),alpp(ntymi:ntymx),cint
      double precision deltap,deltam,eps
      parameter(eps=1.d-20)

      om1yk=0.d0

      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
          gamomy(i)=atilde(i,k)
          deltap=btildep(i,k)
          deltam=btildepp(i,k)

        alpp(i)=deltap-deltam
        gamomy(i)=gamomy(i)*xh**((deltap+deltam)*0.5d0)

      enddo

      cint=0.d0
      do i=imin,imax
        if(abs(alpp(i)).gt.eps)then
          cint=cint-gamomy(i)/alpp(i)*xh**(alpp(i)*0.5d0)
     &                               *(1.d0-xh**(-alpp(i)))
        else
          cint=cint-gamomy(i)*log(xh)
        endif
      enddo

      do i=imin,imax
        if(abs(alpp(i)).gt.eps)then
        om1yk=om1yk+gamomy(i)*exp(alpp(i)*yp)
        else
        om1yk=om1yk+gamomy(i)
        endif
      enddo

      om1yk=om1yk/cint


      return
      end

c----------------------------------------------------------------------
      double precision function om1xrk(k)   !---test---
c----------------------------------------------------------------------
c Random number generated from the function om1xk. We solve the equation
c which give om1xrk by Newton-Raphson + secant method.
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incpar'

      double precision x,x0,x1,gamomx(ntymi:ntymx),eps,prec,drangen
      double precision xt,fx,fpx,r,f1,f0,cint,deltx,alpp(ntymi:ntymx)
     &,delta(ntymi:ntymx),deltap(ntymi:ntymx),deltam(ntymi:ntymx)
     &,gamom,f0t,f1t
      parameter (eps=1.d-20)


      om1xrk=0.d0

      imin=ntymin
      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=imin,imax
        gamomx(i)=atilde(i,k)
        deltap(i)=btildep(i,k)
        deltam(i)=btildepp(i,k)

        delta(i)=(deltap(i)+deltam(i))*0.5d0
        alpp(i)=deltap(i)-deltam(i)

      enddo

      cint=0.d0
      do i=imin,imax
        gamom=gamomx(i)
        if((deltap(i)+1.d0).gt.eps)then
          gamom=gamom/(deltap(i)+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        if((deltam(i)+1.d0).gt.eps)then
          gamom=gamom/(deltam(i)+1.d0)
        else
          gamom=-gamom*log(xminDf)
        endif
        cint=cint+gamom
      enddo

      x0=eps
      x1=1.d0
      f0=0.d0
      f1=0.d0
      do i=imin,imax

        if(abs(alpp(i)).lt.eps)then
          if(delta(i)+1.d0.gt.eps)then
        f0=f0-gamomx(i)/(delta(i)+1.d0)*x0**(delta(i)+1.d0)
     &        *(log(x0)-1.d0/(delta(i)+1.d0))
        f1=f1-gamomx(i)/(delta(i)+1.d0)*x1**(delta(i)+1.d0)
     &        *(log(x1)-1.d0/(delta(i)+1.d0))
          else
        f0=f0-0.5d0*gamomx(i)*(log(x0)**2-log(xminDf)**2)
        f1=f1-0.5d0*gamomx(i)*(log(x1)**2-log(xminDf)**2)
          endif
        else
          if(abs(deltap(i)+1.d0).gt.eps
     &  .and.abs(deltam(i)+1.d0).gt.eps)then
        f0=f0+gamomx(i)/alpp(i)*(x0**(deltam(i)+1.d0)/(deltam(i)+1.d0)
     &                          -x0**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        f1=f1+gamomx(i)/alpp(i)*(x1**(deltam(i)+1.d0)/(deltam(i)+1.d0)
     &                          -x1**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        elseif(abs(deltap(i)+1.d0).gt.eps)then
        f0=f0+gamomx(i)/alpp(i)*(log(x0/xminDf)
     &                          -x0**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        f1=f1+gamomx(i)/alpp(i)*(log(x1/xminDf)
     &                          -x1**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        elseif(abs(deltam(i)+1.d0).gt.eps)then
        f0=f0-gamomx(i)/alpp(i)*(log(x0/xminDf)
     &                          -x0**(deltam(i)+1.d0)/(deltam(i)+1.d0))
        f1=f1-gamomx(i)/alpp(i)*(log(x1/xminDf)
     &                          -x1**(deltam(i)+1.d0)/(deltam(i)+1.d0))
          endif
        endif
      enddo
      f0=-f0/cint
      f1=-f1/cint
      ntry=0
 11   ntry=ntry+1
      r=drangen(dble(ntry))
      f0t=f0+r
      f1t=f1+r
      if(f1t*f0t.ge.eps.and.ntry.lt.100)goto 11
      if(f1t*f0t.ge.eps)then
        do i=imin,imax
         write(ifmt,*)i,gamomx(i),deltap(i),deltam(i),alpp(i),delta(i)
        enddo
        write(ifmt,*)x0,f0,f0t,x1,f1,f1t,r,cint,ntry,bk(k),k
        call utstop('om1xrk (1)&',
     +sizeof('om1xrk (1)&'))
      endif
      f0=f0t
      f1=f1t
c      if(f1*f0.gt.eps)then
c        call utmsg('om1xrk')
c        write(ifch,*)'Poblem with x0, no root ! --> om1xrk=xminDf'
c        write(ifmt,*)'Poblem with x0, no root ! --> om1xrk=xminDf'
c        write(ifmt,*)f0,f1,cint,r
c        call utmsgf
c        om1xrk=x0
c        return
c      endif
      if(abs(f0).lt.eps) then
        om1xrk=x0
        return
      endif
      if(abs(f1).lt.eps) then
        om1xrk=x1
        return
      endif
c      x=(x1+x0)*0.5D0
      x=sqrt(x1*x0)
      deltx=abs(x1-x0)

      ntry=0
      fx=0.d0
      fpx=0.d0
      xt=x
 111  continue

      if(ntry.le.1000)then
      fx=0.d0
      fpx=0.d0
      do i=imin,imax
        if(abs(alpp(i)).lt.eps)then
          if(delta(i)+1.d0.gt.eps)then
        fx=fx-gamomx(i)/(delta(i)+1.d0)*x**(delta(i)+1.d0)
     &        *(log(x)-1.d0/(delta(i)+1.d0))
        fpx=fpx-gamomx(i)*x**delta(i)*log(x)
          else
        fx=fx-0.5d0*gamomx(i)*(log(x)**2-log(xminDf)**2)
        fpx=fpx-gamomx(i)*log(x)/x
          endif
        else
          if(abs(deltap(i)+1.d0).gt.eps
     &  .and.abs(deltam(i)+1.d0).gt.eps)then
        fx=fx+gamomx(i)/alpp(i)*(x**(deltam(i)+1.d0)/(deltam(i)+1.d0)
     &                          -x**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        fpx=fpx+gamomx(i)/alpp(i)*x**deltam(i)*(1.d0-x**alpp(i))
        elseif(abs(deltap(i)+1.d0).gt.eps)then
        fx=fx+gamomx(i)/alpp(i)*(log(x/xminDf)
     &                          -x**(deltap(i)+1.d0)/(deltap(i)+1.d0))
        fpx=fpx+gamomx(i)/alpp(i)*x**deltam(i)*(1.d0-x**alpp(i))
        elseif(abs(deltam(i)+1.d0).gt.eps)then
        fx=fx-gamomx(i)/alpp(i)*(log(x/xminDf)
     &                          -x**(deltam(i)+1.d0)/(deltam(i)+1.d0))
        fpx=fpx+gamomx(i)/alpp(i)*x**deltam(i)*(1.d0-x**alpp(i))
          endif
        endif
      enddo
      fx=-fx/cint+r
      fpx=fpx/cint
      xt=x-fx/fpx

      if (f0*fx.lt.-eps) then
        f1=fx
        x1=x
      else
        f0=fx
        x0=x
      endif
      if ((xt.lt.x0-eps.or.xt.gt.x1+eps).and.abs(f1-f0).gt.eps) then
        xt=x1-f1*(x1-x0)/(f1-f0)
      endif

      else

        write(ifmt,*)'Warning in om1xrk, to much try !'

      endif

      if(abs(x-xt).gt.deltx*0.5d0) then
        xt=sqrt(x1*x0)
      endif
      deltx=abs(x-xt)
      if(abs(x).gt.eps)then
        prec=deltx/x
      else
        prec=0d0
        call utstop('Problem in om1xrk&',
     +sizeof('Problem in om1xrk&'))
      endif

      if (prec.gt.1.d-3.and.abs(f1-f0).gt.eps.and.ntry.le.1000)then
         x=xt
         ntry=ntry+1
         goto 111
      endif

      om1xrk=x

      return
      end

c----------------------------------------------------------------------
      double precision function om1yrk(xh)   !---test---
c----------------------------------------------------------------------
c Random number generated from the function om1yk(xh). We solve the
c equation which give om1yrk by Newton-Raphson + secant method.
c xh - fraction of the energy squared s for the pomeron;
c k - pair indice;
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'

      double precision xh,r!,y0,y1,y,gamomy(ntymi:ntymx),eps,ymin,prec,yt

      r=dble(rangen())

      om1yrk=(0.5d0-r)*log(xh)
      return

      end

c----------------------------------------------------------------------
      function ffom12aii(iq,je1,je2)   !---test---
c----------------------------------------------------------------------
      include 'epos.inc'
      ig=5
      xmin=0.01/engy
      xmax=1
      r2=0
      do i2=1,ig
      do m2=1,2
       xm=xmin+(xmax-xmin)*(.5+tgss(ig,i2)*(m2-1.5))
       r1=0
       do i1=1,ig
       do m1=1,2
        xp=xmin+(xmax-xmin)*(.5+tgss(ig,i1)*(m1-1.5))
        f=ffom12a(xp,xm,iq,iq,je1,je2)
        r1=r1+wgss(ig,i1)*f
       enddo
       enddo
       f=r1*0.5*(xmax-xmin)
       r2=r2+wgss(ig,i2)*f
      enddo
      enddo
      ffom12aii=r2*0.5*(xmax-xmin)
      end

c----------------------------------------------------------------------
      function ffom12ai(xp,iq1,iq2,je1,je2)   !---test---
c----------------------------------------------------------------------
      include 'epos.inc'
      ig=5
      xmin=0.01/engy
      xmax=1
      r2=0
      do i2=1,ig
      do m2=1,2
       xm=xmin+(xmax-xmin)*(.5+tgss(ig,i2)*(m2-1.5))
       f=ffom12a(xp,xm,iq1,iq2,je1,je2)
       r2=r2+wgss(ig,i2)*f
      enddo
      enddo
      ffom12ai=r2*0.5*(xmax-xmin)
      end

c----------------------------------------------------------------------
      function ffom12a(xp,xm,iq1,iq2,je1,je2)   !---test---
c----------------------------------------------------------------------
c
c      2*om52*F*F == PomInc
c
c  xp - xplus
c  xm - xminus
c                              iq=1 .... sea-sea
c  iq1 - min iq                iq=2 .... val-sea
c  iq2 - max iq                iq=3 .... sea-val
c                              iq=4 .... val-val
c  je = emission type (projectile and target side)
c          0 ... no emissions
c          1 ... emissions
c       else ... all
c
c  already b-averaged  (\int d2b /sigine*10)
c----------------------------------------------------------------------
      include 'epos.inc'

      sy=engy*engy
      xh=xm*xp
ctp060829      yp=0.5*log(xp/xm)
      ffom12a=0
      do i=iq1,iq2
       if(i.eq.1)then
         ffom12a=ffom12a+2*om52pi(sy*xh,1.,1.,0,je1,je2)
       elseif(i.eq.2)then
         ffom12a=ffom12a+2*om52pi(sy*xh,xp,1.,1,je1,je2)
       elseif(i.eq.3)then
         ffom12a=ffom12a+2*om52pi(sy*xh,xm,1.,2,je1,je2)
       elseif(i.eq.4)then
               ffom12a=ffom12a+2*om52pi(sy*xh,xp,xm,3,je1,je2)
       endif
      enddo
      ffom12a=ffom12a
     *           *alpff(iclpro)*xp**betff(1)*(1-xp)**alplea(iclpro)
     *           *alpff(icltar)*xm**betff(2)*(1-xm)**alplea(icltar)

      end

c----------------------------------------------------------------------
      function ffom11a(xp,xm,iq1,iq2)   !---test---
c----------------------------------------------------------------------
c
c      int(db) om1ff /sigine*10
c
c  xp - xplus                  iq=-1 ... fit
c  xm - xminus                 iq=0 .... soft
c                              iq=1 .... gg
c  iq1 - min iq                iq=2 .... qg
c  iq2 - max iq                iq=3 .... gq
c                              iq=4 .... qq
c                              iq=5 .... diff
c----------------------------------------------------------------------
      include 'epos.inc'
      common/geom/rmproj,rmtarg,bmax,bkmx
      ig=5
      bmid=bkmx/2.
      r=0.d0
      do i=1,ig
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*tgss(ig,i))
          f=ffom11(xp,xm,bb,iq1,iq2)
          r=r+bb*wgss(ig,i)*f
        enddo
      enddo
      ffom11a=r*2.*pi*bmid  /sigine*10
      return
      end

c----------------------------------------------------------------------
      function ffom11(xp,xm,b,iq1,iq2)   !---test---
c----------------------------------------------------------------------
c
c       2*om5*F*F == PomInc
c
c  xp - xplus                  iq=-1 ... fit
c  xm - xminus                 iq=0 .... soft
c  b - impact parameter        iq=1 .... gg
c  iq1 - min iq                iq=2 .... qg
c  iq2 - max iq                iq=3 .... gq
c                              iq=4 .... qq
c                              iq=5 .... diff
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision om51

      if(xm.ge.0.)then

       xh=xm*xp
       yp=0.5*log(xp/xm)
       ffom11=2.*sngl(om51(dble(xh),dble(yp),b,iq1,iq2))
     *     *(1-xm)**alplea(icltar)*(1-xp)**alplea(iclpro)

      else   !xm integration

       ig=5
       xmin=0.01/engy
       xmax=1
       r=0
       do i=1,ig
       do m=1,2
        xmm=xmin*(xmax/xmin)**(.5+tgss(ig,i)*(m-1.5))
        xh=xmm*xp
        yp=0.5*log(xp/xmm)
        f=2.*sngl(om51(dble(xh),dble(yp),b,iq1,iq2))
     *     *(1-xmm)**alplea(icltar)*(1-xp)**alplea(iclpro)
        r=r+wgss(ig,i)*f*xmm
       enddo
       enddo
       ffom11=r*0.5*log(xmax/xmin)

      endif

      end

c----------------------------------------------------------------------
      double precision function om51(xh,yp,b,iq1,iq2)   !---test---
c----------------------------------------------------------------------
c xh - xplus*xminus     iq=-1 ... fit        (om1 * 0.5)
c yp - rapidity         iq=0 .... soft
c b - impact param      iq=1 .... gg
c iq1 - min iq          iq=2 .... qg
c iq2 - max iq          iq=3 .... gq
c                       iq=4 .... qq
c                       iq=5 .... diff
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision xp,xm,xh,yp,om51p,om1
      om51=0.d0
      if(xh.le.0.d0.or.xh.gt.1.d0)return

      sy=engy*engy
      xp=sqrt(xh)*exp(yp)
      xm=xh/xp

      if(iq1.eq.-1.and.iq2.eq.-1)then
        om51=0.5d0*om1(xh,yp,b)
      elseif(iq1.ge.0)then
        om51=0.d0
        do i=iq1,iq2
          if(i.ne.5)then
            i1=min(i,1)
          call Gfunpar(0.,0.,1,i1,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
          om51=om51+om51p(sy*sngl(xh),xh,yp,b,i)
     *           *xp**dble(epsp)*xm**dble(epst)
     *           *dble(sy)**dble(epss)
          else
          call Gfunpar(0.,0.,1,2,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
          om51=om51+0.5d0*dble(alp)*xp**dble(bet)*xm**dble(betp)
          endif
       enddo
      else
        stop'om5: choice of iq1 and iq2 is nonsense.     '
      endif

      end

c----------------------------------------------------------------------
      double precision function om5s(sx,xh,yp,b,iq1,iq2)   !---test---
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision om51
      double precision xh,yp
      ss=sx/xh
      engyx=engy
      engy=sqrt(ss)
      om5s=om51(xh,yp,b,iq1,iq2)
      engy=engyx
      end

c----------------------------------------------------------------------
      double precision function om5Jk(k,xh,yp,iqq)   !---MC---
c----------------------------------------------------------------------
c partial om5
c xh - fraction of the energy squared s for the pomeron;
c b - impact parameter between the pomeron ends;
c yp - rapidity for the pomeron;
c iqq=0 - soft
c iqq=1 - gg
c iqq=2 - qg
c iqq=3 - gq
c iqq=4 - qq
c iqq=5 - diffractif
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      double precision xh,yp,om51p
      double precision plc,s
      common/cems5/plc,s

      sy=sngl(s*xh)
      b=bk(k)
      epsGp=0.
      epsGt=0.

      if(iqq.ne.5)then
        om5Jk=om51p(sy,xh,yp,b,iqq)
c Screening effect on Pomeron type set in WomTy
        if(iscreen.ne.0)then
          xp=sqrt(xh)*exp(yp)
          xm=xh/xp
          i1=min(iqq,1)
c use pp screening even for nuclei (avoid to many diffractive collisions)
c          call Gfunpar(0.,0.,1,i1,b,sngl(s),alp,bet,betp,epsp,epst,epss
c     &                 ,gamv)
c          epsGp=epsp
c          epsGt=epst
cc          epsG=epsilongs(k,i1)
          if(iqq.eq.0)then
            epsGp=epsilongp(k,i1)
            epsGt=epsilongt(k,i1)
          elseif(gfactor.gt.0.)then
            epsGp=epsilongp(k,i1)*exp(-min(50.,gfactor*zparpro(k)))
            epsGt=epsilongt(k,i1)*exp(-min(50.,gfactor*zpartar(k)))
          endif
          om5Jk=om5Jk*xp**dble(epsGp)*xm**dble(epsGt)!*s**dble(epsG))
        endif
      else
        xp=sqrt(xh)*exp(yp)
        xm=xh/xp
        om5Jk=0.5d0*atilde(2,k)*xp**btildep(2,k)*xm**btildepp(2,k)
      endif
      return
      end

c----------------------------------------------------------------------
      double precision function om5J(zzp,zzt,xh,yp,b,iq)   !---test---
c----------------------------------------------------------------------
c xh - xplus*xminus     iq=-1 ... fit        (om1 * 0.5)
c yp - rapidity         iq=0 .... soft
c b - impact param      iq=1 .... gg
c                       iq=2 .... qg
c                       iq=3 .... gq
c zzp - projectile Z    iq=4 .... qq
c zzt - target Z        iq=5 .... diff
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision xp,xm,xh,yp,om51p,om1
      om5J=0.d0
      if(xh.le.0.d0.or.xh.gt.1.d0)return

      sy=engy*engy
      xp=sqrt(xh)*exp(yp)
      xm=xh/xp
      epsGp=0.
      epsGt=0.

      if(iq.eq.-1)then
        om5J=0.5d0*om1(xh,yp,b)
      elseif(iq.ne.5)then
        i1=min(iq,1)
        call Gfunpar(zzp,zzt,1,i1,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
c        call Gfunpar(0.,0.,1,i1,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
c        epsG=epss
        if(iq.eq.0)then
          epsGp=epsp
          epsGt=epst
        elseif(gfactor.gt.0.)then
          epsGp=epsp*exp(-min(50.,gfactor*zzp))
          epsGt=epst*exp(-min(50.,gfactor*zzt))
        endif
        om5J=om51p(sy*sngl(xh),xh,yp,b,iq)
     &      *xp**dble(epsGp)*xm**dble(epsGt)!*sy**dble(epsG)
      else
        call Gfunpar(zzp,zzt,1,2,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
c        call Gfunpar(0.,0.,1,2,b,sy,alp,bet,betp,epsp,epst,epss,gamv)
      om5J=0.5d0*alp*xp**dble(bet)*xm**dble(betp)
      endif

      end

c----------------------------------------------------------------------
      double precision function omIgamint(b,iqq)   !---test---
c----------------------------------------------------------------------
c - integrated chi~(b)FF/2 for cut I diagram (simple Pomeron)
c b - impact parameter between the pomeron ends;
c yp - rapidity for the pomeron;
c iqq=0 effective one
c iqq=1 soft
c iqq=2 gg
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision Df

      Df=0.d0
      sy=engy*engy
      omIgamint=0.d0
      imax=idxD1
      if(iomega.eq.2)imax=1

      if(iqq.eq.0)then
        coefp=1.+alplea(iclpro)
        coeft=1.+alplea(icltar)

        do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,sy,alpx,betx,betpx,epsp,epst,epss,gamv)
          betp=1.+betx
          betpp=1.+betpx
          Df=alpx*dble(utgam1(betp)*utgam1(betpp)*ucfpro
     *         *ucftar/utgam1(betp+coefp)/utgam1(betpp+coeft))
          omIgamint=omIgamint+Df
        enddo
      else
        call utstop('Wrong iqq in omIgamint&',
     +sizeof('Wrong iqq in omIgamint&'))
      endif

      omIgamint=omIgamint
     *     *dble(chad(iclpro)*chad(icltar))

      omIgamint=omIgamint*0.5d0

      return
      end

c-----------------------------------------------------------------------
      subroutine WomTy(w,xh,yp,k)
c-----------------------------------------------------------------------
c - w(ity) for group iqq of cut enhanced diagram giving
c the probability of the type of the same final state.
c k - pair indice;
c xh - fraction of the energy squared s for the pomeron;
c yp - rapidity for the pomeron;
c xpr,xmr impulsion fraction of remnant
c    ity = 0   - soft
c    ity = 1   - gg
c    ity = 2   - qg
c    ity = 3   - gq
c    ity = 4   - qq
c Modified by gfactor to increase semihard int. probability
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      include 'epos.incsem'
      doubleprecision xh,yp,om5Jk,w(0:7),ww

      do i=0,7
        w(i)=0.d0
      enddo

      do i=0,5
        w(i)=om5Jk(k,xh,yp,i)
      enddo
      if(gfactor.lt.0..and.w(1).gt.xggfit*w(0))then      !??????????????
        corfac=exp(-dble(abs(gfactor)*(zparpro(k)+zpartar(k)))
     &                          *max(0d0,1d0-sqrt(xggfit*w(0)/w(1))))
        ww=w(0)+w(1)     !gg interaction probability
        w(0)=w(0)*corfac        !soft suppressed
        w(1)=ww-w(0)            !semi-hard increased


c        corfac=float(iotst1) !??????????????????????
c        ww=w(0)+w(1)+w(2)+w(3)+w(4)+w(5)
c        whard=0.
c        do i=1,4
c          w(i)=w(i)*corfac
c          whard=whard+w(i)
c        enddo
c        if(whard.gt.ww)then
c         do i=1,4
c          w(i)=w(i)/whard*ww
c         enddo
c         whard=ww
c        endif
c        w05=w(0)+w(5)
c        if(whard.lt.ww)then
c          w(0)=w(0)/w05*(ww-whard)
c          w(5)=w(5)/w05*(ww-whard)
c        else
c          w(0)=0
c          w(5)=0
c        endif
        !write(*,'(2f11.4)')(ww-w05)/ww,(ww-w(0)-w(5))/ww
      endif

      return
      end


c-----------------------------------------------------------------------
      double precision function Womegak(xp,xm,xprem,xmrem,k)   !---MC---
c-----------------------------------------------------------------------
c - sum(omGam(xp,xm))*(1-xp)*(1-xm) for group of cut enhanced
c diagram giving the same final state (without nuclear effect).
c xp,xm - fraction of the loght cone momenta of the pomeron;
c k - pair index
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision xp,xm,xprem,xmrem

      Womegak=0.d0

      imax=ntymx
      if(iomega.eq.2)imax=1

      do i=ntymin,imax
        Womegak=Womegak+atilde(i,k)*xp**btildep(i,k)*xm**btildepp(i,k)
      enddo


      Womegak=Womegak*(xprem-xp)*(xmrem-xm)

      return
      end


cc----------------------------------------------------------------------
c      double precision function omNpcut(xp,xm,xprem,xmrem,bh,iqq)   !---test---
cc----------------------------------------------------------------------
cc Sum of all cut diagrams
cc iqq=0 ideal G
cc iqq=1 exact G + diff
cc----------------------------------------------------------------------
c      include "epos.inc"
c      double precision om51,xh,yp,xprem,xmrem,xp,xm!,omYcutI
c
c      omNpcut=0.d0
c      xh=xp*xm
c      if(abs(xh).gt.1.d-10)then
c        yp=0.5d0*log(xp/xm)
c      else
c        yp=0.d0
c      endif
c
c      if(iqq.eq.0)omNpcut=om51(xh,yp,bh,-1,-1)
c      if(iqq.eq.1)omNpcut=om51(xh,yp,bh,0,5)
c
c      omNpcut=omNpcut*2.d0
c
c      return
c      end
c
c----------------------------------------------------------------------
      double precision function omGam(xp,xm,bh)   !---test---
c-----------------------------------------------------------------------
c Cut diagram part for calculation of probability distribution
c xp,xm impulsion fraction of remnant
c bh - impact parameter between the pomeron ends;
c-----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incems"
      double precision om51,xp,xm,xh,yp,eps!,omYgam
      parameter (eps=1.d-20)

      omGam=0.d0
      if(xp.lt.eps.or.xm.lt.eps)return
      xh=xp*xm
      if(abs(xh).gt.1.d-10)then
        yp=0.5d0*log(xp/xm)
      else
        yp=0.d0
      endif

      omGam=om51(xh,yp,bh,-1,-1)

      omGam=2.d0*omGam

      return
      end

c----------------------------------------------------------------------
      double precision function omGamk(k,xp,xm)   !---MC---
c-----------------------------------------------------------------------
c Cut diagram part for calculation of probability distribution (for omega)
c xp,xm impulsion fraction of remnant
c bh - impact parameter between the pomeron ends;
c-----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incems"
      double precision xp,xm
      omGamk=0.d0
      imax=ntymx
      if(iomega.eq.2)imax=1
      do i=ntymin,imax
        omGamk=omGamk+atilde(i,k)*xp**btildep(i,k)*xm**btildepp(i,k)
      enddo

      return
      end

c----------------------------------------------------------------------
      double precision function omGamint(bh)   !---test---
c-----------------------------------------------------------------------
c Integrated cut diagram part for calculation of probability distribution
c bh - impact parameter between the pomeron ends;
c-----------------------------------------------------------------------
      include "epos.inc"
      double precision omIgamint!,omYgamint

      omGamint=2.d0*omIgamint(bh,0)

      return
      end





c----------------------------------------------------------------------
      block data dgdata
c----------------------------------------------------------------------
c constants for numerical integration (gaussian weights)
c----------------------------------------------------------------------
      double precision dgx1,dga1
      common /dga20/ dgx1(10),dga1(10)


      data dgx1/
     &   .765265211334973D-01,
     &   .227785851141645D+00,
     &   .373706088715420D+00,
     &   .510867001950827D+00,
     &   .636053680726515D+00,
     &   .746331906460151D+00,
     &   .839116971822219D+00,
     &   .912234428251326D+00,
     &   .963971927277914D+00,
     &   .993128599185095D+00/
      data dga1/
     &   .152753387130726D+00,
     &   .149172986472604D+00,
     &   .142096109318382D+00,
     &   .131688638449177D+00,
     &   .118194531961518D+00,
     &   .101930119817233D+00,
     &   .832767415767047D-01,
     &   .626720483341090D-01,
     &   .406014298003871D-01,
     &   .176140071391506D-01/

      end



c----------------------------------------------------------------------
      double precision function Phiexact(zzip,zzit,fj,xp,xm,s,b) !---test---
c----------------------------------------------------------------------
c    Exact expression of the Phi function for pp collision
c    zzip : additionnal component for Z (nuclear effect projectile side)
c    zzit : additionnal component for Z (nuclear effect target side)
c    fj   : overall factor for cross section (elastic or inelastic)
c    xp,xm: momentum fraction
c    s    : energy square
c    b    : impact parameter
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision al(idxD0:idxD1),betp(idxD0:idxD1)
     *,z,xIrst!,ffacto
      double precision zp(idxD0:idxD1),Phitmp,betpp(idxD0:idxD1)
     *,yp,ym,xm,xp
      double precision eps
      parameter(eps=1.d-20)
      dimension ipr(idxD0:idxD1),imax(idxD0:idxD1)

      if(idxD0.ne.0.or.idxD1.ne.2) stop "Problem in PhiExact"
      Phitmp=0.d0

      if(xp.gt.eps.and.xm.gt.eps.and.xp.le.1.d0+eps
     &   .and.xm.le.1.d0+eps)then


       do i=idxD0,idxD1
        imax(i)=0
        ipr(i)=0
        zp(i)=1.d0
        al(i)=0.d0
        betp(i)=0.d0
        betpp(i)=0.d0
       enddo

       imax0=idxD1
       if(iomega.eq.2)imax0=1

       do i=idxDmin,imax0
        imax(i)=10+max(5,int(log10(s)))
        if(b.ge.1.)imax(i)=4+max(3,int(log10(sqrt(s))))
        imax(i)=min(30,imax(i))
       enddo
       Phitmp=0.d0
       do i=idxDmin,imax0
         call Gfunpar(zzip,zzit,1,i,b,s,alpx,betx,betpx,epsp,epst,epss
     *               ,gamv)
         betp(i)=dble(betx)+1.d0
         betpp(i)=dble(betpx)+1.d0
         al(i)=dble(alpx*gamv)
     *         *dble(chad(iclpro)*chad(icltar))
       enddo

       do ipr0=0,imax(0)
          ipr(0)=ipr0
          zp(0)=1.d0
        if (ipr(0).ne.0) zp(0)=(-dble(fj)*al(0))**ipr(0)*facto(ipr(0))
        do ipr1=0,imax(1)
           ipr(1)=ipr1
           zp(1)=1.d0
        if (ipr(1).ne.0) zp(1)=(-dble(fj)*al(1))**ipr(1)*facto(ipr(1))
        do ipr2=0,imax(2)
           ipr(2)=ipr2
           zp(2)=1.d0
        if (ipr(2).ne.0) zp(2)=(-dble(fj)*al(2))**ipr(2)*facto(ipr(2))
          yp=0.d0
          ym=0.d0
          z=1.d0
          isum=0
          do i=idxDmin,imax0
            yp=yp+dble(ipr(i))*betp(i)
            ym=ym+dble(ipr(i))*betpp(i)
            isum=isum+ipr(i)
            z=z*zp(i)
          enddo

          z=z*xIrst(1,xp,yp,betp,ipr)
          z=z*xIrst(2,xm,ym,betpp,ipr)

          Phitmp=Phitmp+z

         enddo
         enddo
        enddo



      endif

      PhiExact=Phitmp


      return
      end


c----------------------------------------------------------------------
      double precision function PhiExpoK(k,xp,xm)   !---MC---
c----------------------------------------------------------------------
c    Exponential expression of the Phi function for pp collision
c    for given k
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'

      double precision xp,xm,Phitmp,Gt1
      double precision atildg,btildgp,btildgpp
      common/cgtilde/atildg(idxD0:idxD1,kollmx)
     *,btildgp(idxD0:idxD1,kollmx),btildgpp(idxD0:idxD1,kollmx)


      Phitmp=0.d0

      imax=idxD1
      if(iomega.eq.2)imax=1

      Phitmp=0.d0
      Gt1=0.d0
      do i=idxDmin,imax
       Gt1=Gt1+atildg(i,k)*xp**btildgp(i,k)*xm**btildgpp(i,k)
      enddo

      Phitmp=exp(-Gt1)

      PhiExpoK=Phitmp

      return
      end

c----------------------------------------------------------------------
      double precision function PhiExpoK2(k,xp,xm)   !---xs---
c----------------------------------------------------------------------
c    Exponential expression of the Phi function for pp collision
c    for given k without diffractive part
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'

      double precision xp,xm,Phitmp,Gt1
      double precision atildg,btildgp,btildgpp
      common/cgtilde/atildg(idxD0:idxD1,kollmx)
     *,btildgp(idxD0:idxD1,kollmx),btildgpp(idxD0:idxD1,kollmx)


      Phitmp=0.d0

      imax=1

      Phitmp=0.d0
      Gt1=0.d0
      do i=idxDmin,imax
       Gt1=Gt1+atildg(i,k)*xp**btildgp(i,k)*xm**btildgpp(i,k)
      enddo

      Phitmp=exp(-Gt1)

      PhiExpoK2=Phitmp

      return
      end

c----------------------------------------------------------------------
      double precision function Phiexpo(zzip,zzit,fj,xp,xm,s,b)   !---MC---
c----------------------------------------------------------------------
c    Exponential expression of the Phi function for pp collision
c    for given b
c input :
c    zzip : additionnal component for Z (nuclear effect projectile side)
c    zzit : additionnal component for Z (nuclear effect target side)
c    fj   : overall factor for cross section (elastic or inelastic)
c    xp,xm: momentum fraction
c    s    : energy square
c    b    : impact parameter
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'

      parameter(nbkbin=40)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      double precision AlTi
      double precision BeTip,BeTipp
      double precision xp,xm,Phitmp,Gt1

      imax=idxD1
      if(iomega.eq.2)imax=1

      Gt1=0.d0
      do i=idxDmin,imax
        call Gfunpar(zzip,zzit,2,i,b,s,alpx,betx,betpx,epsp,epst,epss
     &              ,gamv)
        BeTip =dble(betx)
        BeTipp=dble(betpx)
        AlTi  =dble(alpx)
        Gt1=Gt1+AlTi*xp**BeTip*xm**BeTipp*dble(fj*xkappa**(fj-1.))

      enddo

      Phitmp=exp(-Gt1)

      PhiExpo=Phitmp
     &     *xp**dble(alplea(iclpro))
     &     *xm**dble(alplea(icltar))

      return
      end

c----------------------------------------------------------------------
      double precision function PhiUnit(xp,xm)   !---test---
c----------------------------------------------------------------------
c    Exponential expression of the Phi function for pp collision
c    for given b
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'

      double precision AlTi
      double precision BeTip,BeTipp
      double precision xp,xm,Phitmp,Gt1

      imax=idxD1
      if(iomega.eq.2)imax=1

      Gt1=0.d0
      do i=idxDmin,imax
        BeTip =betUni(i,2)
        BeTipp=betpUni(i,2)
        AlTi  =alpUni(i,2)
        Gt1=Gt1+AlTi*xp**BeTip*xm**BeTipp
c        write(ifch,*)'Phiunit',i,xp,xm,Gt1,AlTi,BeTip,BeTipp
      enddo

      Phitmp=exp(-Gt1)

      PhiUnit=Phitmp
     &     *xp**dble(alplea(iclpro))
     &     *xm**dble(alplea(icltar))


      return
      end


cc----------------------------------------------------------------------
c      double precision function PhiUnit(xp,xm,s,b)   !---inu---
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      double precision xp,xm,PhiExpo,Znorm
c
c      PhiUnit=Phiexpo(0.,0.,1.,xp,xm,s,b)
c     &          /Znorm(s,b)
c
c      return
c      end
c

c----------------------------------------------------------------------
      double precision function Hrst(s,b,xp,xm)   !test
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'
      parameter(idxD2=8)
      double precision GbetUni,GbetpUni,HbetUni,HbetpUni,HalpUni
      common/DGamUni/GbetUni(  idxD0:idxD2),HbetUni(  idxD0:idxD2),
     &               GbetpUni(idxD0:idxD2),HbetpUni(idxD0:idxD2),
     &               HalpUni(idxD0:idxD2)
      double precision al(idxD0:idxD2),betp(idxD0:idxD2)
     *,z,xJrst!,ffacto
      double precision zp(idxD0:idxD2),Htmp,betpp(idxD0:idxD2)
     *,yp,ym,xp,xm
      dimension ipr(idxD0:idxD2),imax(idxD0:idxD2)

      if(idxD0.ne.0.or.idxD1.ne.2) stop "Problem in Hrst"
      Htmp=0.d0
      do i=idxD0,idxD2
        imax(i)=0
        ipr(i)=0
        zp(i)=1.d0
        al(i)=0.d0
      enddo

      if(xp.ge.0.d0.and.xm.ge.0.d0.and.xp.lt.1.d0.and.xm.le.1.d0)then

      imax0=idxD1
      if(iomega.eq.2)imax0=1
      imax1=idxD2
      if(iomega.eq.2)imax1=imax1-1

      do i=idxDmin,imax1
        imax(i)=max(2,int(log10(100.*s)/3.))
c        if(i.ge.2)imax(i)=imax(i)*2
        if(b.ge.1.5)imax(i)=2   !max(2,imax(i)/2)
        imax(i)=min(30,imax(i))
        if(i.gt.imax0)then
          if((zzpUni*zztUni.lt.1.d-6)
     &   .or.(xp.lt.0.1d0.and.xm.lt.0.1d0))then
            imax(i)=0
          else
            imax(i)=1     !imax(i)/3
          endif
        endif
      enddo

      Htmp=0.d0
        do i=idxDmin,imax1
          betp(i)=HbetUni(i)
          betpp(i)=HbetpUni(i)
          al(i)=HalpUni(i)
        enddo

        do ipr0=0,imax(0)
c          write(ifmt,*)'Hrst ipr0,xp,xm :',ipr0,xp,xm
           ipr(0)=ipr0
           zp(0)=1.d0
           if (ipr(0).ne.0) zp(0)=al(0)**ipr(0)*facto(ipr(0))
         do ipr1=0,imax(1)
            ipr(1)=ipr1
            zp(1)=1.d0
            if (ipr(1).ne.0) zp(1)=al(1)**ipr(1)*facto(ipr(1))
         do ipr2=0,imax(2)
            ipr(2)=ipr2
            zp(2)=1.d0
            if (ipr(2).ne.0) zp(2)=al(2)**ipr(2)*facto(ipr(2))
         do ipr3=0,imax(3)
            ipr(3)=ipr3
            zp(3)=1.d0
            if (ipr(3).ne.0) zp(3)=al(3)**ipr(3)*facto(ipr(3))
         do ipr4=0,imax(4)
            ipr(4)=ipr4
            zp(4)=1.d0
            if (ipr(4).ne.0) zp(4)=al(4)**ipr(4)*facto(ipr(4))
         do ipr5=0,imax(5)
            ipr(5)=ipr5
            zp(5)=1.d0
            if (ipr(5).ne.0) zp(5)=al(5)**ipr(5)*facto(ipr(5))
         do ipr6=0,imax(6)
            ipr(6)=ipr6
            zp(6)=1.d0
            if (ipr(6).ne.0) zp(6)=al(6)**ipr(6)*facto(ipr(6))
         do ipr7=0,imax(7)
            ipr(7)=ipr7
            zp(7)=1.d0
            if (ipr(7).ne.0) zp(7)=al(7)**ipr(7)*facto(ipr(7))
         do ipr8=0,imax(8)
            ipr(8)=ipr8
            zp(8)=1.d0
            if (ipr(8).ne.0) zp(8)=al(8)**ipr(8)*facto(ipr(8))
           if (ipr(0)+ipr(1)+ipr(2)+ipr(3)+ipr(4)+ipr(5)
     &        +ipr(6)+ipr(7)+ipr(8).ne.0) then
             yp=0.d0
             ym=0.d0
             z=1.d0
             do i=idxDmin,imax1
               yp=yp+dble(ipr(i))*betp(i)
               ym=ym+dble(ipr(i))*betpp(i)
               z=z*zp(i)
             enddo
             z=z*xJrst(xp,yp,GbetUni,ipr)
             z=z*xJrst(xm,ym,GbetpUni,ipr)
             Htmp=Htmp+z
           endif
          enddo
         enddo
       enddo
          enddo
         enddo
       enddo
          enddo
         enddo
       enddo

      endif

      Hrst=Htmp

      return
      end

c----------------------------------------------------------------------
      double precision function HrstI(s,b,xp,xm)   !test
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'
      parameter(idxD2=8)
      double precision GbetUni,GbetpUni,HbetUni,HbetpUni,HalpUni
      common/DGamUni/GbetUni(  idxD0:idxD2),HbetUni(  idxD0:idxD2),
     &               GbetpUni(idxD0:idxD2),HbetpUni(idxD0:idxD2),
     &               HalpUni(idxD0:idxD2)
      double precision al(idxD0:idxD2),betp(idxD0:idxD2)
     *,z,xJrstI!,ffacto
      double precision zp(idxD0:idxD2),Htmp,betpp(idxD0:idxD2)
     *,yp,ym,xp,xm
      dimension ipr(idxD0:idxD2),imax(idxD0:idxD2)

      if(idxD0.ne.0.or.idxD1.ne.2) stop "Problem in HrstI"
      Htmp=0d0
      do i=idxD0,idxD2
        imax(i)=0
        ipr(i)=0
        zp(i)=1.d0
        al(i)=0.d0
      enddo


      if(xp.ge.0.d0.and.xm.ge.0.d0.and.xp.lt.1.d0.and.xm.le.1.d0)then


      imax0=idxD1
      if(iomega.eq.2)imax0=1
      imax1=idxD2
      if(iomega.eq.2)imax1=imax1-1

      do i=idxDmin,imax1
        imax(i)=max(3,int(log10(s)/2.))
c        if(i.ge.2)imax(i)=imax(i)*2
        if(b.ge.1.5)imax(i)=max(2,imax(i)/2)
        imax(i)=min(30,imax(i))
        if(i.gt.imax0)then
          if((zzpUni*zztUni.lt.1.d-6)
     &   .or.(xp.lt.0.1d0.and.xm.lt.0.1d0))then
            imax(i)=0
          else
            imax(i)=1   !imax(i)/3
          endif
        endif
      enddo

      Htmp=0.d0
        do i=idxDmin,imax1
          betp(i)=HbetUni(i)
          betpp(i)=HbetpUni(i)
          al(i)=HalpUni(i)
        enddo
        do ipr0=0,imax(0)
           ipr(0)=ipr0
           zp(0)=1.d0
           if (ipr(0).ne.0) zp(0)=al(0)**ipr(0)*facto(ipr(0))
         do ipr1=0,imax(1)
            ipr(1)=ipr1
            zp(1)=1.d0
            if (ipr(1).ne.0) zp(1)=al(1)**ipr(1)*facto(ipr(1))
         do ipr2=0,imax(2)
            ipr(2)=ipr2
            zp(2)=1.d0
            if (ipr(2).ne.0) zp(2)=al(2)**ipr(2)*facto(ipr(2))
         do ipr3=0,imax(3)
            ipr(3)=ipr3
            zp(3)=1.d0
            if (ipr(3).ne.0) zp(3)=al(3)**ipr(3)*facto(ipr(3))
         do ipr4=0,imax(4)
            ipr(4)=ipr4
            zp(4)=1.d0
            if (ipr(4).ne.0) zp(4)=al(4)**ipr(4)*facto(ipr(4))
         do ipr5=0,imax(5)
            ipr(5)=ipr5
            zp(5)=1.d0
            if (ipr(5).ne.0) zp(5)=al(5)**ipr(5)*facto(ipr(5))
         do ipr6=0,imax(6)
            ipr(6)=ipr6
            zp(6)=1.d0
            if (ipr(6).ne.0) zp(6)=al(6)**ipr(6)*facto(ipr(6))
         do ipr7=0,imax(7)
            ipr(7)=ipr7
            zp(7)=1.d0
            if (ipr(7).ne.0) zp(7)=al(7)**ipr(7)*facto(ipr(7))
         do ipr8=0,imax(8)
            ipr(8)=ipr8
            zp(8)=1.d0
            if (ipr(8).ne.0) zp(8)=al(8)**ipr(8)*facto(ipr(8))
           if (ipr(0)+ipr(1)+ipr(2)+ipr(3)+ipr(4)+ipr(5)
     &        +ipr(6)+ipr(7)+ipr(8).ne.0) then
             yp=0.d0
             ym=0.d0
             z=1.d0
             do i=idxDmin,imax1
               yp=yp+dble(ipr(i))*betp(i)
               ym=ym+dble(ipr(i))*betpp(i)
               z=z*zp(i)
             enddo
             z=z*xJrstI(xp,yp,GbetUni,ipr)
             z=z*xJrstI(xm,ym,GbetpUni,ipr)
             Htmp=Htmp+z
           endif
          enddo
         enddo
       enddo
          enddo
         enddo
       enddo
          enddo
         enddo
       enddo

      endif

      HrstI=Htmp

      return
      end



cc----------------------------------------------------------------------
c      double precision function HrstI(s,xp,xm)   !---inu---
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      include 'epos.incems'
c      include 'epos.incsem'
c      include 'epos.incpar'
c      double precision al(idxD0:idxD1),betp(idxD0:idxD1)
c     *,z,xJrstI!,ffacto
c      double precision zp(idxD0:idxD1),Htmp,betpp(idxD0:idxD1)
c     *,yp,ym,xp,xm
c      dimension ipr(idxD0:idxD1),imax(idxD0:idxD1)
c
c      if(idxD0.ne.0.or.idxD1.ne.2) stop "Problem in HrstI"
c      HrstI=0.d0
c      do i=idxD0,idxD1
c        imax(i)=0
c        ipr(i)=0
c        zp(i)=1.d0
c        al(i)=0.d0
c      enddo
c
c      if(xp.ge.0.d0.and.xm.ge.0.d0.and.xp.lt.1.d0.and.xm.lt.1.d0)then
c
c      HrstI=0.d0
c
c      imax0=idxD1
c      if(iomega.eq.2)imax0=1
c
c      do i=idxDmin,imax0
c        imax(i)=max(5,int(log10(s)))
cc        if(i.ge.2)imax(i)=imax(i)*2
c        imax(i)=min(30,imax(i))
c      enddo
c      Htmp=0.d0
c        do i=idxDmin,imax0
c          betp(i)=betUni(i,1)+1.d0
c          betpp(i)=betpUni(i,1)+1.d0
c          al(i)=alpUni(i,1)*dble(chad(iclpro)*chad(icltar))
c        enddo
c
c        do ipr0=0,imax(0)
c           ipr(0)=ipr0
c           zp(0)=1.d0
c          if (ipr(0).ne.0) zp(0)=al(0)**ipr(0)*facto(ipr(0))
c         do ipr1=0,imax(1)
c            ipr(1)=ipr1
c            zp(1)=1.d0
c          if (ipr(1).ne.0) zp(1)=al(1)**ipr(1)*facto(ipr(1))
c         do ipr2=0,imax(2)
c            ipr(2)=ipr2
c            zp(2)=1.d0
c          if (ipr(2).ne.0) zp(2)=al(2)**ipr(2)*facto(ipr(2))
c             if (ipr(0)+ipr(1)+ipr(2).ne.0) then
c             yp=0.d0
c             ym=0.d0
c             z=1.d0
c             do i=idxDmin,imax0
c               yp=yp+dble(ipr(i))*betp(i)
c               ym=ym+dble(ipr(i))*betpp(i)
c               z=z*zp(i)
c             enddo
c             z=z*xJrstI(xp,yp,betp,ipr)
c             z=z*xJrstI(xm,ym,betpp,ipr)
c             Htmp=Htmp+z
c           endif
c          enddo
c         enddo
c       enddo
c
c       HrstI=Htmp
c
c      endif
c
c      return
c      end
c

cc----------------------------------------------------------------------
c        double precision function ffacto(n)   !---test---
cc----------------------------------------------------------------------
c
c        ffacto=1.D0
c        do i=1,n
c          ffacto=ffacto*dble(i)
c        enddo
c        return
c        end
c

c----------------------------------------------------------------------
      double precision function xIrst(id,x,y,bet,ipr)   !---test---
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision y,gammag,utgam2,x,bet(idxD0:idxD1)
      dimension ipr(idxD0:idxD1)

      if(id.eq.1)iclrem=iclpro
      if(id.eq.2)iclrem=icltar
      imax=idxD1
      if(iomega.eq.2)imax=1
      if(y.le.160.)then
       xIrst=gammag(iclrem,y)*x**dble(alplea(iclrem))
      else
       xIrst=0
      endif
      if(xIrst.gt.0.d0)then
        do i=idxDmin,imax
          if(ipr(i).ne.0.and.bet(i).gt.1.d-10)
     &         xIrst=xIrst*utgam2(bet(i))**dble(ipr(i))
        enddo
        if (abs(y).gt.1.d-10) xIrst=xIrst*x**y
      endif
      return
      end


c----------------------------------------------------------------------
      double precision function xJrst(x,y,Gbeta,ipr)   !---inu---
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(idxD2=8)
      double precision y,utgam2,x,Gbeta(idxD0:idxD2),eps,gam
      dimension ipr(idxD0:idxD2)

      eps=1.d-10

      imax=idxD2
      if(iomega.eq.2)imax=imax-1

      gam=utgam2(y)

      if(gam.lt.1.d99)then

      if ((x-1.d0).gt.eps.or.(y-1.d0).gt.eps) then
                        xJrst=(1.d0-x)**(y-1.d0)/gam
                        do i=idxDmin,imax
      if (ipr(i).ne.0)   xJrst=xJrst*Gbeta(i)**dble(ipr(i))
                        enddo
          else
c            write (*,*) 'Warning in xJrst, infinite value !'
                        xJrst=(1.d0-x+eps)**(y-1.d0)/gam
                        do i=idxDmin,imax
      if (ipr(i).ne.0)   xJrst=xJrst*Gbeta(i)**dble(ipr(i))
                        enddo
      endif
      else
        xJrst=0.d0
      endif

      return
      end


c----------------------------------------------------------------------
      double precision function xJrstI(x,y,Gbeta,ipr)   !---inu---
c----------------------------------------------------------------------
c Function used for the integration of H*Phi. We do the changement of
c variable (1-x)=z**alpha. The power alpha can be change if necessary.
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(idxD2=8)
      double precision y,utgam2,x,Gbeta(idxD0:idxD2),alpha,w,gam
      dimension ipr(idxD0:idxD2)

      alpha=4.d0
      w=alpha*(y-1.d0)+alpha-1.d0
      imax=idxD2
      if(iomega.eq.2)imax=imax-1

      gam=utgam2(y)

      if(gam.lt.1.d99)then

      if (w.ge.0)then

                        xJrstI=alpha*x**w/gam
                        do i=idxDmin,imax
      if (ipr(i).ne.0)   xJrstI=xJrstI*Gbeta(i)**dble(ipr(i))
                        enddo

         else
           write(*,*) 'x,y,bet,ipr,w',x,y,Gbeta,ipr,w
          stop 'Error in xJrstI in epos-omg, integration not possible'
       endif

      else
        xJrstI=0.d0
      endif


      return
      end

c----------------------------------------------------------------------
      double precision function HPhiInt(s,b)   !---inu---
c----------------------------------------------------------------------
c  Set integrated over xp and xm (x and y) H(x,y)*Phi(x,y) for a
c  given b by gauss method
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter(idxD2=8)
      double precision GbetUni,GbetpUni,HbetUni,HbetpUni,HalpUni
      common/DGamUni/GbetUni(  idxD0:idxD2),HbetUni(  idxD0:idxD2),
     &               GbetpUni(idxD0:idxD2),HbetpUni(idxD0:idxD2),
     &               HalpUni(idxD0:idxD2)
      double precision xhm,x,y,yhm,w,Hrst,utgam2,PhiUnit!,PhiExact
c      double precision zp2,zm2,HrstI,eps
c      common /ar3/  x1(7),a1(7)
      common /ar9/    x9(3),a9(3)

      eps=0d0 !1.d-5

      imax0=idxD1
      imax1=idxD2
      if(iomega.eq.2)then
        imax0=1
        imax1=imax1-1
      endif
      do i=idxDmin,imax0
        HbetUni(i)=betUni(i,1)+1.d0
        HbetpUni(i)=betpUni(i,1)+1.d0
        GbetUni(i)=utgam2(HbetUni(i))
        GbetpUni(i)=utgam2(HbetpUni(i))
        HalpUni(i)=alpUni(i,1)*dble(chad(iclpro)*chad(icltar))
      enddo
      do i=0,1
        HbetUni(imax0+1+i)=betUni(i,1)+1.d0+betfom
        HbetUni(imax0+3+i)=betUni(i,1)+1.d0
        HbetUni(imax0+5+i)=betUni(i,1)+1.d0+betfom
        HbetpUni(imax0+1+i)=betpUni(i,1)+1.d0
        HbetpUni(imax0+3+i)=betpUni(i,1)+1.d0+betfom
        HbetpUni(imax0+5+i)=betpUni(i,1)+1.d0+betfom
        GbetUni(imax0+1+i)=utgam2(HbetUni(imax0+1+i))
        GbetUni(imax0+3+i)=utgam2(HbetUni(imax0+3+i))
        GbetUni(imax0+5+i)=utgam2(HbetUni(imax0+5+i))
        GbetpUni(imax0+1+i)=utgam2(HbetpUni(imax0+1+i))
        GbetpUni(imax0+3+i)=utgam2(HbetpUni(imax0+3+i))
        GbetpUni(imax0+5+i)=utgam2(HbetpUni(imax0+5+i))
        HalpUni(imax0+1+i)=zztUni*alpUni(i,1)
        HalpUni(imax0+3+i)=zzpUni*alpUni(i,1)
        HalpUni(imax0+5+i)=zzpUni*zztUni*alpUni(i,1)
      enddo

      w=0.d0
      xhm=.5d0*(1d0-eps)
      yhm=.5d0*(1d0-eps)
      do m=1,2
        do i=1,3
c        do i=1,7
          x=xhm+dble((2*m-3)*x9(i))*xhm
c          write(ifmt,*)'HPhiInt, xp int :',x
          do n=1,2
            do j=1,3
c            do j=1,7
              y=yhm+dble((2*n-3)*x9(j))*yhm
             w=w+dble(a9(i)*a9(j))*Hrst(s,b,x,y)
     &          *PhiUnit(x,y)
c     &          *Phiexact(0.,0.,1.,x,y,s,b)
            enddo
          enddo
        enddo
      enddo

      HPhiInt=w*xhm*yhm


c      w=0.d0
c      xhm=.5d0*eps
c      yhm=.5d0*eps
c      do m=1,2
c        do i=1,7
c          x=1d0-eps+xhm+dble((2*m-3)*x1(i))*xhm
c          do n=1,2
c            do j=1,7
c              y=1d0-epsyhm+dble((2*n-3)*x1(j))*yhm
c              zp2=1.d0-x**4
c              zm2=1.d0-y**4
c              w=w+dble(a1(i)*a1(j))*HrstI(s,x,y)
cc     &             *PhiUnit(zp2,zm2)
c     &             *Phiexact(0.,0.,1.,zp2,zm2,s,b)
c            enddo
c          enddo
c        enddo
c      enddo
c
c      HPhiInt=HPhiInt+w*xhm*yhm

      return
      end



c----------------------------------------------------------------------
      subroutine Kfit(iiclegy)
c----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incsem"
      double precision Znorm
      parameter(nbkbin=40)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      parameter (nmax=30)
      logical lnoch

      if(iiclegy.eq.-1.or.iiclegy.gt.iclegy2)then
        do iiiegy=1,nclegy
        do iiipro=1,nclha
        do iiitar=1,nclha
        do iiibk=1,nbkbin
          xkappafit(iiiegy,iiipro,iiitar,iiibk)=1.
        enddo
        enddo
        enddo
        enddo

      else

      if(isetcs.le.1)then
        s=engy*engy
        eps=0.05
      else
        s=(egylow*egyfac**(iiclegy-1))**2.
        eps=0.001
      endif

      write(ifmt,*)"Fit xkappa ..."
      if(ish.ge.2)then
        write(ifmt,*)"Kfit s,bkbin,iclegy,ipro,itar"
     *       ,s,bkbin,iiclegy,iclpro,icltar
      endif


      b=0.
      if(isetcs.le.1.or.iiclegy.eq.iclegy2)then
        xkf=1.
      else
        xkf=xkappafit(iiclegy+1,iclpro,icltar,1)
      endif
      xkfs=1.
      delta=0.
      deltas=0.


      do 5 ib=1,nbkbin-1
        b=float(ib-1)*bkbin
        xkappafit(iiclegy,iclpro,icltar,ib)=1.
        if(b.gt.3.+0.05*log(s).or.s.le.20.*q2min)then
          xkf=1.
          goto 5
        endif
        if(ib.gt.1.and.ish.ge.3)write(ifch,*)"    End",delta,xkf
        delta=1.-sngl(Znorm(s,b))
        if(delta.le.0d0)then
          if(xkf.ne.1.)then
            xkappafit(iiclegy,iclpro,icltar,ib)=xkf
            delta=1.-sngl(Znorm(s,b))
          endif
        else!if(xkf.ne.1.)then
          goto 5
        endif
        if(abs(delta).lt.eps)then
          if(delta.lt.0d0)then
            xkfs=xkf-delta
            deltas=delta
          endif
          xkf=1.
          goto 5
        elseif(ib.le.nbkbin-1)then

          if(delta.gt.0.d0)then
            xkf0=1.
            xkf1=xkf
            delta0=delta
            xkf2=xkf-delta0
            xkappafit(iiclegy,iclpro,icltar,ib)=xkf2
            delta1=1.-sngl(Znorm(s,b))
            if(delta1.lt.0.d0)then
              xkf0=xkf2
              xkf1=xkf
              delta=delta1
              xkf=xkf0
            else
              xkf1=max(delta0,xkf2)
              xkf0=0.
              xkf=xkf1
            endif
          else
            xkf0=xkf
            xkf1=1.-delta
            xkf2=xkf
            delta1=delta
          endif

          if(ib.eq.1)then
            deltas=delta
            xkfs=max(0.00001,1.-delta)
          endif

          if(delta.le.deltas)xkf=xkfs
          if(ish.ge.3)write(ifch,*)"    Start",ib,b,delta,xkf,xkf0,xkf1
          if(xkf.eq.xkf2)delta=delta1

          n=0
          delta0=delta
          lnoch=.true.
 10       continue
          n=n+1
          if(n.le.nmax.and.xkf1.ne.xkf0)then
            if(abs(xkf-xkf2).gt.1e-6.or.abs(delta).gt.abs(deltas))then
              xkappafit(iiclegy,iclpro,icltar,ib)=xkf
              delta=1.-sngl(Znorm(s,b))
            endif
            if(ish.ge.5)write(ifch,*)"    step",ib,n,delta,xkf,delta0
            if(delta*delta0.ge.0.)then
              if(lnoch.and.abs(delta).gt.abs(delta0))goto 5
            else
              lnoch=.false.
            endif
            if(abs(delta).gt.eps)then
              if(delta.gt.0.)then
                xkf1=xkf
                xkf=(xkf1+xkf0)*0.5
                delta0=delta
              else
                xkf0=xkf
                xkf=(xkf1+xkf0)*0.5
                delta0=delta
              endif
              goto 10
            endif
          else
            if(ish.ge.2)
     *      write(ifmt,*)"Warning in Kfit, nmax reached : xkappafit=1."
            xkappafit(iiclegy,iclpro,icltar,ib)=xkf
          endif
        endif

 5    continue

      if(ish.ge.3)write(ifch,*)"    End",delta,xkf
      if(xkf.gt.1.+eps)write(ifmt,*)
     *     "Warning in Kfit, xkappafit not yet 1"
      xkappafit(iiclegy,iclpro,icltar,nbkbin)=1.

      endif

      return
      end

c----------------------------------------------------------------------
      double precision function Znorm(s,b)   !---inu---
c----------------------------------------------------------------------
      include 'epos.inc'
      common /kwrite/ xkapZ
      double precision HPhiInt,PhiUnit!,PhiExact

c      write(ifmt,*)'Z calculation for (s,b) :',s,b
      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alpx,betx,betpx,epsp,epst,epss,gamv)
        call Gfunpar(0.,0.,2,i,b,s,alpx,betx,betpx,epsp,epst,epss,gamv)
      enddo
      call GfomPar(b,s)
      Znorm=HPhiInt(s,b)
c      write(ifch,*)'int',Znorm,' phi',Phiexact(0.,0.,1.,1.d0,1.d0,s,b)
      Znorm=Znorm
     &       +PhiUnit(1.d0,1.d0)
c     &       +Phiexact(0.,0.,1.,1.d0,1.d0,s,b)

      !write(ifmt,*)'Z=',Znorm,xkapZ,b
      return
      end


c------------------------------------------------------------
      double precision function gammag(iclrem,x)   !---test---
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      double precision x,utgam2

      gammag=utgam2(dble(alplea(iclrem))+1.D0)
     &       /utgam2(dble(alplea(iclrem))+1.D0+x)

      return
      end



cc----------------------------------------------------------------------
c        double precision function PomNbri(iqq)   !---xsigma---
cc----------------------------------------------------------------------
cc integral d2b om1intbci
cc iqq, Pomeron type
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      double precision om1intbci
c      common/geom/rmproj,rmtarg,bmax,bkmx
c      common /ar3/  x1(7),a1(7)
c
c      bmid=bkmx/2.
c      PomNbri=0.d0
c      do i=1,7
c        do m=1,2
c          bb=bmid*(1.+(2.*m-3)*x1(i))
c          PomNbri=PomNbri+dble(bb*a1(i))*om1intbci(bb,iqq)
c        enddo
c      enddo
c
c      PomNbri=PomNbri*dble(2.*pi*bmid)
c
c      return
c      end
c
c
c


c####################################################################################
c#############   former chk #########################################################
c####################################################################################








cc----------------------------------------------------------------------
c      double precision function PomIncII(b)   !---check---
cc----------------------------------------------------------------------
cc  integral_dx_dy om1*F_remn*F_remn for a given b   !---check---
cc----------------------------------------------------------------------
c      include 'epos.inc'
c      include 'epos.incems'
c      include 'epos.incsem'
c      include 'epos.incpar'
c       double precision cint,gamom(idxD0:idxD1),deltap(idxD0:idxD1)
c     &,deltapp(idxD0:idxD1),utgam2
c
cc Calculation by analytical integration (perfect but it changes
cc if om1 change):
c
c      s=engy**2
c      imax=1
c      if(iomega.eq.2)imax=1
c      do i=idxDmin,imax
c        call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
c        gamom(i)=dble(alp*gamv)*chad(iclpro)*chad(icltar)
c        deltap(i)=dble(bet)
c        deltapp(i)=dble(betp)
c
cc Integration possible only if delta(i)>-1
c
c       if(deltap(i).le.-1.d0.or.deltapp(i).le.-1.d0)
c     &       stop 'Error in epos-par-300 in PomIncII'
c      enddo
c
c      cint=0.d0
c      do i=idxDmin,imax
c       cint=cint+gamom(i)*utgam2(deltap(i)+1.d0)*utgam2(deltapp(i)+1.d0)
c     &            *dble(ucfpro*ucftar)
c     &            /utgam2(dble(alplea(iclpro))+deltap(i)+2.d0)
c     &            /utgam2(dble(alplea(icltar))+deltapp(i)+2.d0)
c      enddo
c
c      PomIncII=cint
c
c      return
c      end
c

c----------------------------------------------------------------------
        double precision function PomIncXIExact(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncXExact
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncXExact
      common /ar3/  x1(7),a1(7)
      common/geom/rmproj,rmtarg,bmax,bkmx

      bmid=bkmx/2.
      PomIncXIExact=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncXIExact=PomIncXIExact+dble(bb*a1(i))*PomIncXExact(x,bb)
        enddo
      enddo

      PomIncXIExact=PomIncXIExact*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncXIUnit(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncXUnit
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncXUnit
      common /ar3/  x1(7),a1(7)
      common/geom/rmproj,rmtarg,bmax,bkmx

      bmid=bkmx/2.
      PomIncXIUnit=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncXIUnit=PomIncXIUnit+dble(bb*a1(i))*PomIncXUnit(x,bb)
        enddo
      enddo

      PomIncXIUnit=PomIncXIUnit*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncPIExact(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncPExact
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncPExact
      common/geom/rmproj,rmtarg,bmax,bkmx
      common /ar3/  x1(7),a1(7)

      bmid=bkmx/2.
      PomIncPIExact=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncPIExact=PomIncPIExact+dble(bb*a1(i))*PomIncPExact(x,bb)
        enddo
      enddo

      PomIncPIExact=PomIncPIExact*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncPIUnit(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncPUnit
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncPUnit
      common/geom/rmproj,rmtarg,bmax,bkmx
      common /ar3/  x1(7),a1(7)

      bmid=bkmx/2.
      PomIncPIUnit=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncPIUnit=PomIncPIUnit+dble(bb*a1(i))*PomIncPUnit(x,bb)
        enddo
      enddo

      PomIncPIUnit=PomIncPIUnit*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncMIExact(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncMExact
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncMExact
      common/geom/rmproj,rmtarg,bmax,bkmx
      common /ar3/  x1(7),a1(7)

      bmid=bkmx/2.
      PomIncMIExact=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncMIExact=PomIncMIExact+dble(bb*a1(i))*PomIncMExact(x,bb)
        enddo
      enddo

      PomIncMIExact=PomIncMIExact*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncMIUnit(x)   !---check---
c----------------------------------------------------------------------
c integral d2b PomIncMUnit
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision x,PomIncMUnit
      common/geom/rmproj,rmtarg,bmax,bkmx
      common /ar3/  x1(7),a1(7)

      bmid=bkmx/2.
      PomIncMIUnit=0.d0
      do i=1,7
        do m=1,2
          bb=bmid*(1.+(2.*m-3)*x1(i))
          PomIncMIUnit=PomIncMIUnit+dble(bb*a1(i))*PomIncMUnit(x,bb)
        enddo
      enddo

      PomIncMIUnit=PomIncMIUnit*dble(2.*pi*bmid)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncMExact(xm,b)   !---check---
c----------------------------------------------------------------------
c incluse Pomeron distribution \int dx+ { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision AlTiP,BeTip,al,bep,bepp,xpInt,utgam2,xm

      s=engy**2
      PomIncMExact=0.d0
      imax=1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
        bep =dble(bet)
        bepp=dble(betp)
        al  =dble(alp*gamv)

        BeTip=bep+1.d0
        xpInt=utgam2(BeTip)*dble(ucfpro)
     *                    /utgam2(1.d0+dble(alplea(iclpro))+BeTip)
        AlTiP=al*xpInt
        PomIncMExact=PomIncMExact+AlTiP*xm**bepp
     *                            *(1.d0-xm)**dble(alplea(icltar))
      enddo

      return
      end

c----------------------------------------------------------------------
      double precision function PomIncMUnit(xm,b)   !---check---
c----------------------------------------------------------------------
c incluse  Unitarized Pomeron distribution  \int dx+
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision Df,xp,xm,G2,w,xpm
      double precision PoInU!,Znorm
      common /ar3/  x1(7),a1(7)

      s=engy**2

c Calculation by numeric integration :
      w=0.d0
      xpm=.5d0
      do m=1,2
        do j=1,7
          xp=xpm*(1.d0+dble((2.*m-3.)*x1(j)))
          Df=0.d0
          do i=idxDmin,idxD1
            call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
            Df=Df+dble(alp)*xp**dble(bet)*xm**dble(betp)
          enddo
          call GfomPar(b,s)
          G2=Df*(1.d0+zztUni*xp**betfom)*(1.d0+zzpUni*xm**betfom)
          w=w+dble(a1(j))*PoInU(xp,xm,s,b)*G2
        enddo
      enddo
      w=w*xpm


      PomIncMUnit=w!/Znorm(s,b)

      return
      end


c----------------------------------------------------------------------
      double precision function PomIncPExact(xp,b)   !---check---
c----------------------------------------------------------------------
c incluse Pomeron distribution  \int dx- { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision AlTiP,BeTipp,al,bep,bepp,xmInt,utgam2,xp

      s=engy**2
      PomIncPExact=0.d0
      imax=1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
        bep=dble(bet)
        bepp=dble(betp)
        al=dble(alp*gamv)
        BeTipp=bepp+1.d0
        xmInt=utgam2(BeTipp)*dble(ucftar)
     *                    /utgam2(1.d0+dble(alplea(icltar))+BeTipp)
        AlTiP=al*xmInt
        PomIncPExact=PomIncPExact+AlTiP*xp**bep
     *                            *(1.d0-xp)**dble(alplea(iclpro))
      enddo

      return
      end

c----------------------------------------------------------------------
      double precision function PomIncPUnit(xp,b)   !---check---
c----------------------------------------------------------------------
c incluse  Unitarized Pomeron distribution  \int dx-
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      double precision Df,xp,xm,G2,w,xmm
      double precision PoInU!,Znorm
      common /ar3/  x1(7),a1(7)

      s=engy**2
      imax=1
      if(iomega.eq.2)imax=1

c Calculation by numeric integration :
      w=0.d0
      xmm=.5d0
      do m=1,2
        do j=1,7
          xm=xmm*(1.d0+dble((2.*m-3.)*x1(j)))
          Df=0.d0
          do i=idxDmin,imax
            call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
            Df=Df+alp*xp**dble(bet)*xm**dble(betp)
          enddo
          call GfomPar(b,s)
          G2=Df*(1.d0+zztUni*xp**betfom)*(1.d0+zzpUni*xm**betfom)
          w=w+dble(a1(j))*PoInU(xp,xm,s,b)*G2
        enddo
      enddo
      w=w*xmm


      PomIncPUnit=w!/Znorm(s,b)

      return
      end


c----------------------------------------------------------------------
        double precision function PomIncJExact(b)   !---check---
c----------------------------------------------------------------------
c integral of Pomeron distribution  \int dy dx { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision allea,PomIncXExact,xh
      common /ar3/  x1(7),a1(7)

      allea=2.d0+dble(alplea(iclpro)+alplea(icltar))
      PomIncJExact=0.d0
      do i=1,7
        do m=1,2
          xh=1.d0-(.5d0+dble(x1(i)*(float(m)-1.5)))**(1.d0/allea)
          PomIncJExact=PomIncJExact+dble(a1(i))
     &       *PomIncXExact(xh,b)/(1.d0-xh)**(allea-1.d0)
        enddo
      enddo
      PomIncJExact=PomIncJExact/allea/2.d0

      return
      end


c----------------------------------------------------------------------
        double precision function PomIncExactk(k)   !---MC---
c----------------------------------------------------------------------
c integral of Pomeron distribution  \int dy dx { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision allea,PomIncXExactk,xh
      common /ar3/  x1(7),a1(7)

      allea=2.d0+dble(alplea(iclpro)+alplea(icltar))
      PomIncExactk=0.d0
      do i=1,7
        do m=1,2
          xh=1.d0-(.5d0+dble(x1(i)*(float(m)-1.5)))**(1.d0/allea)
          PomIncExactk=PomIncExactk+dble(a1(i))
     &       *PomIncXExactk(k,xh)/(1.d0-xh)**(allea-1.d0)
        enddo
      enddo
      PomIncExactk=PomIncExactk/allea/2.d0

      return
      end


c----------------------------------------------------------------------
        double precision function PomIncJUnit(b)   !---check---
c----------------------------------------------------------------------
c integral of Pomeron distribution  \int dy dx { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision PomIncXUnit,xh,xhm
      common /ar3/  x1(7),a1(7)

      PomIncJUnit=0.d0
      xhm=.5d0
      do i=1,7
        do m=1,2
          xh=xhm*(1.d0+dble(x1(i)*(2.*float(m)-3.)))
          PomIncJUnit=PomIncJUnit+dble(a1(i))
     &                                *PomIncXUnit(xh,b)
        enddo
      enddo
      PomIncJUnit=PomIncJUnit*xhm

      return
      end


c----------------------------------------------------------------------
      double precision function PomIncXExactk(k,xh)   !---MC---
c----------------------------------------------------------------------
c incluse Pomeron distribution  \int dy { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision xh,Df,xp,xm,w,ymax,bet,betp,alp
      common /ar3/  x1(7),a1(7)

      imax=1
      if(iomega.eq.2)imax=1
c Calculation by numeric integration :
      w=0.d0
      ymax=-.5d0*log(xh)
      do m=1,2
        do j=1,7
          xp=sqrt(xh)*exp(dble((2.*m-3.)*x1(j))*ymax)
          xm=xh/xp
          Df=0.d0
          do i=idxDmin,imax
            bet=btildep(i,k)+dble(alppar)
            betp=btildepp(i,k)+dble(alppar)
            alp=atilde(i,k)
            Df=Df+alp*xp**bet*xm**betp
     *      *(1.d0-xp)**dble(alplea(iclpro))
     *      *(1.d0-xm)**dble(alplea(icltar))
          enddo
          w=w+dble(a1(j))*Df
        enddo
      enddo
      w=w*ymax*xh**dble(-alppar)


      PomIncXExactk=w

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncXExact(xh,b)   !---check---
c----------------------------------------------------------------------
c incluse Pomeron distribution  \int dy { 2G F_remn F_remn }
c (optimized integration but with no y dependance)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      double precision AlTiP,bep,bepp,factor,factor1
      double precision xpmin,xh,xp,xm,ymax,y
      common /ar3/  x1(7),a1(7)

      imax=1
      if(iomega.eq.2)imax=1
      s=engy**2
      PomIncXExact=0.d0
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alpx,betx,betpx,epsp,epst,epss,gamv)
        bep  =betx
        bepp =betpx
        AlTiP=alpx*gamv
        PomIncXExact=PomIncXExact+AlTiP*xh**((bep+bepp)/2.d0)
      enddo

      factor=0.d0
      allea=min(alplea(iclpro),alplea(icltar))+1.
      xpmin=max(sqrt(xh),exp(-1.d0))
      do i=1,7
        do m=1,2
          xp=1.d0-(1.d0-xpmin)*(.5d0+dble(x1(i)*(float(m)-1.5)))
     *                                         **(1.d0/dble(allea))
          xm=xh/xp
          factor=factor+dble(a1(i))
     *        *((1.d0-xp)**dble(alplea(iclpro)-allea+1.)
     *        *(1.d0-xm)**dble(alplea(icltar))
     *        +(1.d0-xp)**dble(alplea(icltar)-allea+1.)
     *        *(1.d0-xm)**dble(alplea(iclpro)))/xp
        enddo
      enddo
      factor=factor*(1.d0-xpmin)**dble(allea)/dble(allea)


      if(xpmin.gt.1.00001d0*sqrt(xh))then
        ymax=-log(xh)-2.d0
        factor1=0.d0
        do i=1,7
          do m=1,2
            y=ymax*dble(x1(i)*(2*m-3))
            xp=sqrt(xh*exp(y))
            xm=xh/xp
            factor1=factor1+dble(a1(i))*(1.d0-xp)**dble(alplea(iclpro))
     *                                 *(1.d0-xm)**dble(alplea(icltar))
          enddo
        enddo
        factor=factor+factor1*ymax
      endif

      factor=factor/2.d0

      PomIncXExact=PomIncXExact*factor

      return
      end


c----------------------------------------------------------------------
      double precision function PomIncXUnit(xh,b)   !---check---
c----------------------------------------------------------------------
c incluse  Unitarized Pomeron distribution  \int dy
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      double precision xh,Df,xp,xm,w
      double precision PoInU,ymax!,Znorm
      common /ar3/  x1(7),a1(7)

      imax=1
      if(iomega.eq.2)imax=1
      s=engy**2
ctp060829      sy=s*sngl(xh)
c Calculation by numeric integration :
      w=0.d0
      ymax=-.5d0*log(xh)
      do m=1,2
        do j=1,7
          xp=sqrt(xh)*exp(dble((2.*m-3.)*x1(j))*ymax)
          xm=xh/xp
          Df=0.d0
          do i=idxDmin,imax
            call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
            Df=Df+alp*xp**dble(bet)*xm**dble(betp)
          enddo
          call GfomPar(b,s)
          Df=Df*(1.d0+zztUni*xp**betfom)*(1.d0+zzpUni*xm**betfom)
          w=w+dble(a1(j))*Df*PoInU(xp,xm,s,b)
        enddo
      enddo
      w=w*ymax


      PomIncXUnit=w!/Znorm(s,b)

      return
      end



c----------------------------------------------------------------------
      double precision function PoInU(xp,xm,s,b)   !---check---
c----------------------------------------------------------------------
c Function : PhiU(1-xp,1-xm) + /int(H(z+ + x+,z- + x-)PhiU(z+,z-)dz+dz-)
c----------------------------------------------------------------------
      include 'epos.inc'
      double precision xp,xm,zp,zm,zp2,zm2,zpmin,zmmin,deltzp,deltzm
      double precision zpm,zmm,w,HrstI,PhiUnit,Hrst,eps!,PhiExact
      common /ar3/  x1(7),a1(7)

      eps=1.d-5

      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alpx,betx,betpx,epsp,epst,epss,gamv)
        call Gfunpar(0.,0.,2,i,b,s,alpx,betx,betpx,epsp,epst,epss,gamv)
      enddo
      call GfomPar(b,s)

      if (1.d0-xp-eps.gt.0.d0.and.1.d0-xm-eps.gt.0.d0) then
        w=0.d0
        zpmin=1.d0-xp-eps
        zmmin=1.d0-xm-eps
        zpm=.5d0*zpmin
        zmm=.5d0*zmmin
        do m=1,2
          do i=1,7
            zp=zpm+dble((2*m-3)*x1(i))*zpm
            do n=1,2
              do j=1,7
                zm=zmm+dble((2*n-3)*x1(j))*zmm
            w=w+dble(a1(i)*a1(j))*Hrst(s,b,zp+xp,zm+xm)
     &                              *PhiUnit(zp,zm)
c     &                             *Phiexact(0.,0.,1.,zp,zm,s,b)
             enddo
           enddo
         enddo
       enddo
       PoInU=w*zpm*zmm
       deltzp=eps
       deltzm=eps
      else
        PoInU=0.d0
        zpmin=0.d0
        zmmin=0.d0
        deltzp=1.d0-xp
        deltzm=1.d0-xm
      endif

      w=0.d0
      zpm=0d0
      zmm=0d0
      if(abs(deltzp).gt.1.d-10.and.abs(deltzm).gt.1.d-10)then
      zpm=.5d0*deltzp
      zmm=.5d0*deltzm
      do m=1,2
        do i=1,7
          zp=zpmin+zpm+dble((2*m-3)*x1(i))*zpm
          do n=1,2
            do j=1,7
              zm=zmmin+zmm+dble((2*n-3)*x1(j))*zmm
              zp2=1.d0-xp-zp**2
              zm2=1.d0-xm-zm**2
              w=w+dble(a1(i)*a1(j))*HrstI(s,b,zp,zm)
     &             *PhiUnit(zp2,zm2)
c     &             *Phiexact(0.,0.,1.,zp2,zm2,s,b)
            enddo
          enddo
        enddo
      enddo
      endif

      PoInU=PoInU+w*zpm*zmm
     &           +PhiUnit(1.d0-xp,1.d0-xm)
c     &           +Phiexact(0.,0.,1.,1.d0-xp,1.d0-xm,s,b)

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncExact(xp,xm,b)   !---check---
c----------------------------------------------------------------------
c inclusive Pomeron distribution  { 2G F_remn F_remn }
c----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incsem"
      double precision Df,xp,xm

      Df=0.d0
      s=engy**2
      imax=1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
        Df=Df+alp*gamv*xp**dble(bet)*xm**dble(betp)
      enddo
      Df=dble(chad(iclpro)*chad(icltar))
     *  *Df

      PomIncExact=Df
     *            *(1.d0-xp)**dble(alplea(iclpro))
     *            *(1.d0-xm)**dble(alplea(icltar))

      return
      end

c----------------------------------------------------------------------
        double precision function PomIncUnit(xp,xm,b)   !---check---
c----------------------------------------------------------------------
c inclusive Pomeron distribution  { Sum{int{G*Phi} }
c----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incpar"
      include "epos.incsem"
      double precision PoInU,xp,xm,om1,xh,yp


      xh=xp*xm
      yp=0.d0
      if(xm.ne.0.d0)yp=0.5d0*log(xp/xm)
      PomIncUnit=om1(xh,yp,b)*PoInU(xp,xm,engy*engy,b)
     &          *(1.d0+zztUni*xp**betfom)*(1.d0+zzpUni*xm**betfom)

      return
      end


cc----------------------------------------------------------------------
c        double precision function PomIncUnitMC(xp,xm,b)   !---check---
cc----------------------------------------------------------------------
cc inclusive Pomeron distribution  { Sum{int{G*Phi} }
cc----------------------------------------------------------------------
c      include "epos.inc"
c      include "epos.incpar"
c      include "epos.incsem"
c      include "epos.incems"
c      parameter(mmax=20)
c      double precision Gtab,Phitab,xxpu(mmax),xxmu(mmax)
c      double precision Zm,xp,xm,pGtab,Z,omNpcut,xprem,xmrem,
c     *                 sxp,sxm,PhiExpo
c
c      PomIncUnitMC=0.d0
c      if(xp.lt.1.d0.and.xm.lt.1.d0)then
c      m=10
c
c      sy=engy*engy
c      nmcint=2000
c      nmax=nmcint
c      do i=1,mmax
c        xxpu(i)=0.d0
c        xxmu(i)=0.d0
c      enddo
c      xprem=1.d0
c      xmrem=1.d0
c      sxp=xprem-xp
c      sxm=xmrem-xm
c
c      Gtab=omNpcut(xp,xm,sxp,sxm,b,0)
c      Phitab=Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
c      Z=Gtab*Phitab
c      Zm=0.d0
c
c      do mtmp=2,m
c
c        write(*,*)"GPhi",mtmp-1,Zm,Z
c        Zm=0.d0
c        n=0
c
c 10     continue
c          n=n+1
c          if(mod(n,1000000).eq.0)write(*,*)
c     &              "Calculation of PomIncUnit(",mtmp,")->",n
c          xxpu(1)=xp
c          xxmu(1)=xm
c          sxp=xxpu(1)
c          sxm=xxmu(1)
c          pGtab=1.d0
c          do i=2,mtmp
c            rnau=rangen()*sngl(xprem-sxp)
c            xxpu(i)=dble(rnau)
c            sxp=sxp+xxpu(i)
c            rnau=rangen()*sngl(xmrem-sxm)
c            xxmu(i)=dble(rnau)
c            sxm=sxm+xxmu(i)
c          enddo
c          if(sxp.lt.xprem.and.sxm.lt.xmrem)then
c            do i=1,mtmp
c              Gtab=omNpcut(xxpu(i),xxmu(i),xprem-sxp,xmrem-sxm,b,0)
c              pGtab=pGtab*Gtab
c            enddo
c          Zm=Zm+pGtab*Phiexpo(0.,0.,1.,xprem-sxp,xmrem-sxm,sy,b)
c          endif
c        if(n.lt.nmax)goto 10
c        Zm=Zm/dble(nmax)*fctrl(m-mtmp)*facto(mtmp)
c        Z=Z+Zm
c      enddo
c
c      PomIncUnitMC=Z/dble(chad(iclpro)*chad(icltar))
c      endif
c
c      return
c      end
c
c
cc----------------------------------------------------------------------
c        double precision function PhiMCExact(xp,xm,b)   !---check---
cc----------------------------------------------------------------------
cc virtual emissions  { Sum{int{-GFF} }
cc----------------------------------------------------------------------
c      include "epos.inc"
c      include "epos.incpar"
c      include "epos.incsem"
c      include "epos.incems"
c      parameter(mmax=20)
c      double precision Gtab,xxpu(mmax),xxmu(mmax)
c      double precision Zm,xp,xm,pGtab,Z,om51,sxp,sxm,xh,yp
cc     *                 ,omNpuncut
c
c      PhiMCExact=0.d0
c      if(xp.le.1.d0.and.xm.le.1.d0)then
c      m=6
c
c      sy=engy*engy
c      nmcint=50000
c      nmax=nmcint
c      do i=1,mmax
c        xxpu(i)=0.d0
c        xxmu(i)=0.d0
c      enddo
c
c      Z=xp**dble(alplea(iclpro))
c     * *xm**dble(alplea(icltar))
c      Zm=0.d0
c
c      do mtmp=1,m
c
c        write(*,*)"GPhi",mtmp-1,Zm,Z/xp**dble(alplea(iclpro))
c     *                              /xm**dble(alplea(icltar))
c        Zm=0.d0
c        n=0
c
c 10     continue
c          n=n+1
c          if(mod(n,1000000).eq.0)write(*,*)
c     &              "Calculation of Phiexact(0.,0.,",mtmp,")->",n
c          sxp=0.d0
c          sxm=0.d0
c          pGtab=1.d0
c          do i=1,mtmp
c            rnau=rangen()!*sngl(xp-sxp)
c            xxpu(i)=dble(rnau)
c            sxp=sxp+xxpu(i)
c            rnau=rangen()!*sngl(xm-sxm)
c            xxmu(i)=dble(rnau)
c            sxm=sxm+xxmu(i)
c          enddo
c          if(sxp.lt.xp.and.sxm.lt.xm)then
c            do i=1,mtmp
c              xh=xxpu(i)*xxmu(i)
c              if(abs(xh).gt.1.d-30)then
c                yp=0.5d0*log(xxpu(i)/xxmu(i))
c              else
c                yp=0.d0
c              endif
c              Gtab=2*om51(xh,yp,b,0,4)
cc     *            +omNpuncut(sy*sngl(xh),xh,yp,b,1) !om1(xh,yp,b)
c              pGtab=pGtab*(-Gtab)
c            enddo
c            Zm=Zm+pGtab*(xp-sxp)**dble(alplea(iclpro))
c     *           *(xm-sxm)**dble(alplea(icltar))
c          endif
c        if(n.lt.nmax)goto 10
c        Zm=Zm/dble(nmax)*fctrl(m-mtmp)*facto(m)
c        Z=Z+Zm
c      enddo
c
c      PhiMCExact=Z
c      endif
c
c      return
c      end

c----------------------------------------------------------------------
        double precision function Gammapp(sy,b,mtmp)   !---check---
c----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incpar"
      include "epos.incsem"
      include "epos.incems"
      parameter(mmax=20)
      double precision Gtab,xxpu(mmax),xxmu(mmax),PhiExpo
      double precision Zm,xp,xm,pGtab,om1,sxp,sxm,xh,yp

      Gammapp=0.d0

      xp=1.d0
      xm=1.d0
      nmcint=20000
      nmax=nmcint
      do i=1,mmax
        xxpu(i)=0.d0
        xxmu(i)=0.d0
      enddo
      Zm=0.d0

        n=0

 10     continue
          n=n+1
          if(mod(n,10000).eq.0)write(*,*)
     &              "Calculation of Gammapp(",mtmp,")->",n
          sxp=0.d0
          sxm=0.d0
          pGtab=1.d0
          do i=1,mtmp
            rnau=rangen()!*sngl(xp-sxp)
            xxpu(i)=dble(rnau)
            sxp=sxp+xxpu(i)
            rnau=rangen()!*sngl(xm-sxm)
            xxmu(i)=dble(rnau)
            sxm=sxm+xxmu(i)
          enddo
          if(sxp.lt.xp.and.sxm.lt.xm)then
            do i=1,mtmp
              xh=xxpu(i)*xxmu(i)
              if(abs(xh).gt.1.d-30)then
                yp=0.5d0*log(xxpu(i)/xxmu(i))
              else
                yp=0.d0
              endif
              Gtab=om1(xh,yp,b)
              pGtab=pGtab*Gtab
            enddo
            Zm=Zm+pGtab*Phiexpo(0.,0.,1.,xp-sxp,xm-sxm,sy,b)
          endif
        if(n.lt.nmax)goto 10
        Zm=Zm/dble(nmax)!**2.d0*(xp*xm)**dble(mtmp)

      Gammapp=Zm

      return
      end

cc----------------------------------------------------------------------
c        double precision function GammaMCnew(sy,b,mtmp)   !---check---
cc----------------------------------------------------------------------
c      include "epos.inc"
c      include "epos.incpar"
c      include "epos.incsem"
c      include "epos.incems"
c      parameter(mmax=20)
c      common /psar7/ delx,alam3p,gam3p
c      double precision Gtab,xxpu(mmax),xxmu(mmax)
c      double precision Zm,xp,xm,pGtab,omGam,Zmtot,
c     *                 sxp,sxm,PhiExpo!,om1,yp,xh
c
c      GammaMCnew=0.d0
c      Zmtot=0.d0
c      xp=1.d0
c      xm=1.d0
c      nmcint=1000
c      nmax=nmcint
c
c
c      do i=1,mmax
c        xxpu(i)=0.d0
c        xxmu(i)=0.d0
c      enddo
c
c      Zm=0.d0
c
c        n=0
c
c 10     continue
c          n=n+1
c          if(mod(n,1000000).eq.0)write(*,*)
c     &              "Calculation of GammaMCnew(",mtmp,")->",n
c          sxp=0.d0
c          sxm=0.d0
c          pGtab=1.d0
c          do i=1,mtmp
c            rnau=rangen()
c            xxpu(i)=dble(rnau)
c            sxp=sxp+xxpu(i)
c            rnau=rangen()
c            xxmu(i)=dble(rnau)
c            sxm=sxm+xxmu(i)
c          enddo
c          if(sxp.lt.xp.and.sxm.lt.xm)then
c            i=0
c            do k=1,mtmp
c                i=i+1
c                Gtab=omGam(xxpu(i),xxmu(i),b) !om1(xh,yp,b)
c                pGtab=pGtab*Gtab
c              enddo
c            Zm=Zm+pGtab*Phiexpo(0.,0.,1.,xp-sxp,xm-sxm,sy,b)
c          endif
c        if(n.lt.nmax)goto 10
c
c          Zmtot=Zmtot+Zm/dble(nmax)
c
c      GammaMCnew=Zmtot
c
c      return
c      end
cc----------------------------------------------------------------------
c      double precision function GammaMC(sy,b,mtmp)   !---check---
cc----------------------------------------------------------------------
c      include "epos.inc"
c      include "epos.incpar"
c      include "epos.incsem"
c      include "epos.incems"
c      parameter(mmax=20)
c      double precision Gtab,xxpu(mmax),xxmu(mmax)
c      double precision Zm,xp,xm,pGtab,om1,
c     *                 sxp,sxm,xh,yp,PhiExpo!,omNpcut
c
c      GammaMC=0.d0
c
c      xp=1.d0
c      xm=1.d0
c      nmcint=50000
c      nmax=nmcint
c      do i=1,mmax
c        xxpu(i)=0.d0
c        xxmu(i)=0.d0
c      enddo
c
c      Zm=0.d0
c
c        n=0
c
c 10     continue
c          n=n+1
c          if(mod(n,1000000).eq.0)write(*,*)
c     &              "Calculation of GammaMC(",mtmp,")->",n
c          sxp=0.d0
c          sxm=0.d0
c          pGtab=1.d0
c          do i=1,mtmp
c            rnau=rangen()!*sngl(xp-sxp)
c            xxpu(i)=dble(rnau)
c            sxp=sxp+xxpu(i)
c            rnau=rangen()!*sngl(xm-sxm)
c            xxmu(i)=dble(rnau)
c            sxm=sxm+xxmu(i)
c          enddo
c          if(sxp.lt.xp.and.sxm.lt.xm)then
c            do i=1,mtmp
c              xh=xxpu(i)*xxmu(i)
c              if(abs(xh).gt.1.d-30)then
c                yp=0.5d0*log(xxpu(i)/xxmu(i))
c              else
c                yp=0.d0
c              endif
c              Gtab=om1(xh,yp,b)!omNpcut(xxpu(i),xxmu(i),xp-sxp,xm-sxm,b,0) !om1(xh,yp,b)
c              pGtab=pGtab*Gtab
c            enddo
c            Zm=Zm+pGtab*Phiexpo(0.,0.,1.,xp-sxp,xm-sxm,sy,b)
c          endif
c        if(n.lt.nmax)goto 10
c        Zm=Zm/dble(nmax)*fctrl(n-mtmp)*facto(mtmp)
c
c      GammaMC=Zm
c
c      return
c      end
c
c
c

c----------------------------------------------------------------------
        double precision function GammaGauss(sy,b,mtmp)   !---check---
c----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incpar"
      include "epos.incsem"
      include "epos.incems"
      parameter(mmax=3)
      common /psar7/ delx,alam3p,gam3p
      double precision xpmin,xmmin,Gst,zm(mmax),zp(mmax)
     *,xpmax,xmmax,zpmin(mmax),zmmin(mmax),zpmax(mmax)
      double precision xp,xm,pGtab,omGam,dzp(mmax),Gp1,Gm1,xmin,eps
     *,sxp,sxm,PhiExpo,zmmax(mmax),dzm(mmax),Gp2,Gm2,Gp3,Gm3,G0
c     *,PhiExact
      common /ar3/  x1(7),a1(7)
c      double precision dgx1,dga1
c      common /dga20/ dgx1(10),dga1(10)

      GammaGauss=0.d0
      xp=1.d0
      xm=1.d0
      xmin=1.d-13
      eps=1.d-15

      if(mtmp.eq.0)then
        nmax1=0
        jmax1=0
        nmax2=0
        jmax2=0
        nmax3=0
        jmax3=0
      elseif(mtmp.eq.1)then
        nmax1=2
        jmax1=7
        nmax2=0
        jmax2=0
        nmax3=0
        jmax3=0
      elseif(mtmp.eq.2)then
        nmax1=2
        jmax1=7
        nmax2=2
        jmax2=7
        nmax3=0
        jmax3=0
      elseif(mtmp.eq.3)then
        nmax1=2
        jmax1=7
        nmax2=2
        jmax2=7
        nmax3=2
        jmax3=7
      else
        write(*,*)"m not between 0 and 3, return ..."
        return
      endif

        xpmin=xmin
        xmmin=xmin
        xpmax=1.d0
        xmmax=1.d0
      do i=1,mmax
        zp(i)=0.d0
        zm(i)=0.d0
        dzp(i)=0.d0
        dzm(i)=0.d0
        zmmin(i)=0.d0
        zpmax(i)=0.d0
        zpmin(i)=0.d0
        zmmax(i)=0.d0
      enddo

        G0=1.d0

        if(mtmp.eq.0)then
          sxp=xp
          sxm=xm
          G0=Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
        endif


c        write(*,*)'x+/-',xmmin,xmmax,xpmin,xpmax


        dzm(1)=0.d0
        if(abs(xmmin-xmmax).ge.eps.and.mtmp.ge.1)then
        zmmax(1)=-log(xmmin)
        zmmin(1)=-log(xmmax)
        if(abs(xmmin-xmin).lt.eps)then
          zmmin(1)=-log(min(xmmax,1.d0-xmmin-xmmin))
          zmmax(1)=-log(max(xmmin,1.d0-xmmax-xmmax))
        endif
        dzm(1)=(zmmax(1)-zmmin(1))/2.d0
        endif

        dzp(1)=0.d0
        if(abs(xpmin-xpmax).ge.eps.and.mtmp.ge.1)then
        zpmax(1)=-log(xpmin)
        zpmin(1)=-log(xpmax)
        if(abs(xpmin-xmin).lt.eps)then
          zpmin(1)=-log(min(xpmax,1.d0-xpmin-xpmin))
          zpmax(1)=-log(max(xpmin,1.d0-xpmax-xpmax))
        endif
        dzp(1)=(zpmax(1)-zpmin(1))/2.d0
        endif

c        write(*,*)'bornes1=',exp(-zpmax(1)),exp(-zpmin(1))
c     &,exp(-zmmax(1)),exp(-zmmin(1))

        Gp1=0.d0
        do np1=1,nmax1
        do jp1=1,jmax1
          zp(1)=zpmin(1)+dzp(1)*(1.d0+dble(2.*np1-3.)*dble(x1(jp1)))
          Gm1=0.d0
          if(dzm(1).eq.0.d0)then
            nmax1=1
            jmax1=1
          endif
          do nm1=1,nmax1
            do jm1=1,jmax1
              if(dzm(1).ne.0.d0)then
              zm(1)=zmmin(1)+dzm(1)*(1.d0+dble(2.*nm1-3.)*dble(x1(jm1)))
              else
              zm(1)=zp(1)
              endif

              if(mtmp.eq.1)then
              sxp=xp
              sxm=xm
              do i=1,mtmp
                sxp=sxp-exp(-zp(i))
                sxm=sxm-exp(-zm(i))
              enddo
              pGtab=1.d0
              k=0
              do l=1,mtmp
                k=k+1
                if(dzp(k).ge.0.d0.and.dzm(k).ge.0.d0)then
                  Gst=omGam(exp(-zp(k)),exp(-zm(k)),b)
                  pGtab=pGtab*Gst
                  if(Gst.eq.0.d0)
     &                write(*,*)'j1=',k,exp(-zp(k)),exp(-zm(k))
     &     ,exp(-zpmin(k)),exp(-zpmax(k)),dzp(k),dzm(k),jp1
                else
                  pGtab=0.d0
                  write(*,*)'error1 ?',dzp(k),dzm(k)
                endif
              enddo
              if(sxp.gt.0.d0.and.sxm.gt.0.d0)then
                if(dzm(1).ne.0.d0)then
                  Gm1=Gm1+pGtab*
     &dble(a1(jm1))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zm(1))
c     &dble(a1(jm1))*Phiexact(0.,0.,1.,sxp,sxm,sy,b)
c     &*exp(-zm(1))
                  else
                  Gp1=Gp1+pGtab*
     &dble(a1(jp1))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zp(1))
c     &dble(a1(jp1))*Phiexact(0.,0.,1.,sxp,sxm,sy,b)
c     &*exp(-zp(1))
                endif
c          write(*,*)'m=1',mtmp,Gm1,Gp1,pGtab,sxp,sxm
              endif
              endif

        dzp(2)=0.d0
        if(abs(xpmin-xpmax).ge.eps.and.mtmp.ge.2)then
        zpmin(2)=-log(min(min(xpmax,1.d0-exp(-zp(1))),
     &                     1.d0-xpmin-exp(-zp(1))))
        zpmax(2)=-log(max(xpmin,1.d0-xpmax-exp(-zp(1))))
      if(abs(xpmax+xpmax+xpmax-3.d0*dble(1./delx)).lt.eps)then
          zpmin(2)=-log(xpmax)
          zpmax(2)=-log(xpmin)
        endif
          dzp(2)=(zpmax(2)-zpmin(2))/2.d0
        endif

        dzm(2)=0.d0
        if(abs(xmmin-xmmax).ge.eps.and.mtmp.ge.2)then
            zmmin(2)=-log(min(min(xmmax,1.d0-exp(-zm(1))),
     &                     1.d0-xmmin-exp(-zm(1))))
            zmmax(2)=-log(max(xmmin,1.d0-xmmax-exp(-zm(1))))
      if(abs(xmmax+xmmax+xmmax-3.d0*dble(1./delx)).lt.eps)then
            zmmin(2)=-log(xmmax)
            zmmax(2)=-log(xmmin)
          endif
          dzm(2)=(zmmax(2)-zmmin(2))/2.d0
        endif
c        write(*,*)'bornes2=',exp(-zpmax(2)),exp(-zpmin(2))
c     &,exp(-zmmax(2)),exp(-zmmin(2)),xpmax(2),1.d0-exp(-zp(1))
c     &,1.d0-xpmin(3)-exp(-zp(1)),xpmin(2),1.d0-xpmax(3)-exp(-zp(1))

        Gp2=0.d0
        do np2=1,nmax2
        do jp2=1,jmax2
          zp(2)=zpmin(2)+dzp(2)*(1.d0+dble(2.*np2-3.)*dble(x1(jp2)))
          Gm2=0.d0
          if(dzm(2).eq.0.d0)then
            nmax2=1
            jmax2=1
          endif
          do nm2=1,nmax2
            do jm2=1,jmax2
              if(dzm(2).ne.0.d0)then
              zm(2)=zmmin(2)+dzm(2)*(1.d0+dble(2.*nm2-3.)*dble(x1(jm2)))
              else
              zm(2)=zp(2)
              endif

              if(mtmp.eq.2)then
              sxp=xp
              sxm=xm
              do i=1,mtmp
                sxp=sxp-exp(-zp(i))
                sxm=sxm-exp(-zm(i))
              enddo
              pGtab=1.d0
              k=0
              do l=1,mtmp
                k=k+1
                if(dzp(k).ge.0.d0.and.dzm(k).ge.0.d0)then
               Gst=omGam(exp(-zp(k)),exp(-zm(k)),b)
                  pGtab=pGtab*Gst
                  if(Gst.eq.0.d0)
     &                write(*,*)'j2=',k,exp(-zp(k)),exp(-zm(k))
     &     ,exp(-zpmin(k)),exp(-zpmax(k)),dzp(k),dzm(k),jp1,jp2
                else
                  pGtab=0.d0
                  write(*,*)'error2 ?',dzp(k),dzm(k)
                endif
              enddo
              if(sxp.gt.0.d0.and.sxm.gt.0.d0)then
                if(dzm(2).ne.0.d0)then
                  Gm2=Gm2+pGtab*
     &dble(a1(jm2))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zm(2))
c     &dble(a1(jm2))*Phiexact(0.,0.,1.,sxp,sxm,sy,b,mk)
c     &*exp(-zm(2))
                  else
                  Gp2=Gp2+pGtab*
     &dble(a1(jp2))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zp(2))
c     &dble(a1(jp2))*Phiexact(0.,0.,1.,sxp,sxm,sy,b,mk)
c     &*exp(-zp(2))
                endif
c          write(*,*)'m=2',mtmp,Gm2,Gp2,pGtab,sxp,sxm
              endif
              endif

        dzp(3)=0.d0
        if(abs(xpmin-xpmax).ge.eps.and.mtmp.ge.3)then
        zpmin(3)=-log(min(xpmax,1.d0-exp(-zp(1))-exp(-zp(2))))
        zpmax(3)=-log(xpmin)
        dzp(3)=(zpmax(3)-zpmin(3))/2.d0
        endif

        dzm(3)=0.d0
        if(abs(xmmin-xmmax).ge.eps.and.mtmp.ge.3)then
        zmmin(3)=-log(min(xmmax,1.d0-exp(-zm(1))-exp(-zm(2))))
        zmmax(3)=-log(xmmin)
        dzm(3)=(zmmax(3)-zmmin(3))/2.d0
        endif

c        write(*,*)'bornes3=',exp(-zpmax(3)),exp(-zpmin(3))
c     &,exp(-zmmax(3)),exp(-zmmin(3))

        Gp3=0.d0
        do np3=1,nmax3
        do jp3=1,jmax3
          zp(3)=zpmin(3)+dzp(3)*(1.d0+dble(2.*np3-3.)*dble(x1(jp3)))
          Gm3=0.d0
          if(dzm(3).eq.0.d0)then
            nmax3=1
            jmax3=1
          endif
          do nm3=1,nmax3
            do jm3=1,jmax3
              if(dzm(3).ne.0.d0)then
              zm(3)=zmmin(3)+dzm(3)*(1.d0+dble(2.*nm3-3.)*dble(x1(jm3)))
              else
              zm(3)=zp(3)
              endif

              sxp=xp
              sxm=xm
              do i=1,mtmp
                sxp=sxp-exp(-zp(i))
                sxm=sxm-exp(-zm(i))
              enddo
              pGtab=1.d0
              k=0
              do l=1,mtmp
                k=k+1
                if(dzp(k).ge.0.d0.and.dzm(k).ge.0.d0)then
               Gst=omGam(exp(-zp(k)),exp(-zm(k)),b)
                  pGtab=pGtab*Gst
                  if(Gst.eq.0.d0)
     &                write(*,*)'j3=',k,exp(-zp(k)),exp(-zm(k))
     &   ,exp(-zpmin(k)),exp(-zpmax(k)),dzp(k),dzm(k),jp1,jp2,jp3
                else
                  pGtab=0.d0
                  write(*,*)'error3 ?',k,dzp(k),dzm(k)
                endif
              enddo
              if(sxp.gt.0.d0.and.sxm.gt.0.d0)then
                if(dzm(3).ne.0.d0)then
                  Gm3=Gm3+pGtab
     &*dble(a1(jm3))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zm(3))
                  else
                  Gp3=Gp3+pGtab
     &*dble(a1(jp3))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)
     &*exp(-zp(3))
                endif
              endif
            enddo
          enddo
         if(dzm(3).ne.0.d0)Gp3=Gp3+Gm3*dble(a1(jp3))*exp(-zp(3))*dzm(3)
         nmax3=2
         jmax3=7
        enddo
      enddo
              if(mtmp.gt.2.and.dzm(2).ne.0.d0)then
                Gm2=Gm2+Gp3*dble(a1(jm2))*exp(-zm(2))*dzp(3)
              elseif(mtmp.gt.2)then
                Gp2=Gp2+Gp3*dble(a1(jp2))*exp(-zp(2))*dzp(3)
              endif
            enddo
          enddo
         if(dzm(2).ne.0.d0)Gp2=Gp2+Gm2*dble(a1(jp2))*exp(-zp(2))*dzm(2)
         nmax2=2
         jmax2=7
        enddo
      enddo
              if(mtmp.gt.1.and.dzm(1).ne.0.d0)then
                Gm1=Gm1+Gp2*dble(a1(jm1))*exp(-zm(1))*dzp(2)
              elseif(mtmp.gt.1)then
                Gp1=Gp1+Gp2*dble(a1(jp1))*exp(-zp(1))*dzp(2)
              endif
            enddo
          enddo
         if(dzm(1).ne.0.d0)Gp1=Gp1+Gm1*dble(a1(jp1))*exp(-zp(1))*dzm(1)
         nmax1=2
         jmax1=7
        enddo
      enddo

      if(mtmp.gt.0)G0=Gp1*dzp(1)
      write(*,*)"int:",G0

      GammaGauss=GammaGauss+G0

      return

      end

c-----------------------------------------------------------------------
      double precision function omWi(sy,b)   !---check---
c-----------------------------------------------------------------------
c cut enhanced diagram integrated over xp, xm, xpr,xmr
c (with ideal G)
c b - impact parameter between the pomeron ends;
c sy- total energy
c-----------------------------------------------------------------------
      include "epos.inc"
      include "epos.incpar"
      include "epos.incsem"
      include "epos.incems"
      common /psar7/ delx,alam3p,gam3p
      double precision xpmin,xmmin,zp,zm,alpp,alpm,xjacp,xjacm
     *,xpmax,xmmax,zpmin,zmmin,zpmax,chg
      double precision xp,xm,pGtab,omGam,dzp,Gp1,Gm1,xmin,eps
     *,sxp,sxm,PhiExpo,zmmax,dzm!,gamp,gamm,gampp,gammp
c     *,PhiExact
      common /ar3/  x1(7),a1(7)
c      double precision dgx1,dga1
c      common /dga20/ dgx1(10),dga1(10)

      omWi=0.d0

        xmin=1.d-30
        eps=1.d-15
        chg=1.d0/dble(delx)
        b2=b*b
        gamb=gamD(1,iclpro,icltar)
ctp060829        gamp=dble(gamb*b2/2.-alppar)
ctp060829        gamm=dble(gamb*b2/2.-alppar)
ctp060829        gampp=1.d0+2.d0*gamp
ctp060829        gammp=1.d0+2.d0*gamm

        nmax=2
        jmax=7

        xpmin=xmin
        xmmin=xmin
        xpmax=1.d0
        xmmax=1.d0
        zpmin=0.d0
        zmmin=0.d0
        zpmax=0.d0
        zmmax=0.d0
        zp=0.d0
        zm=0.d0
        dzp=0.d0
        dzm=0.d0



        do intp=1,2
        do intm=1,2

          if(intp.eq.1)then

          xpmin=xmin
          xpmax=chg
          alpp=(1.d0+2.d0*dble(gamb*b2/2.))

          else

          xpmin=chg
          xpmax=1.d0
          alpp=1.d0!(1.d0+2.d0*dble(gamb*b2/2.))

          endif

          if(intm.eq.1)then

          xmmin=xmin
          xmmax=chg
          alpm=(1.d0+2.d0*dble(gamb*b2/2.))

          else

          xmmin=chg
          xmmax=1.d0
          alpm=1.d0!(1.d0+2.d0*dble(gamb*b2/2.))

          endif
c        write(*,*)'x+/-',intp,intm,xmmin,xmmax,xpmin,xpmax,alpp,alpm


        dzm=0.d0
        if(abs(xmmin-xmmax).ge.eps)then
          if(alpm.eq.0.d0)then
            zmmax=-log(xmmin)
            zmmin=-log(xmmax)
          else
            zmmin=xmmin**alpm
            zmmax=xmmax**alpm
          endif
          dzm=(zmmax-zmmin)/2.d0
        endif

        dzp=0.d0
        if(abs(xpmin-xpmax).ge.eps)then
          if(alpp.eq.0.d0)then
            zpmax=-log(xpmin)
            zpmin=-log(xpmax)
          else
            zpmin=xpmin**alpp
            zpmax=xpmax**alpp
          endif
          dzp=(zpmax-zpmin)/2.d0
        endif


        Gp1=0.d0

        if(abs(dzp).gt.eps.and.abs(dzm).gt.eps)then
c        write(*,*)'Ca passe ...'

        do np1=1,nmax
        do jp1=1,jmax
          zp=zpmin+dzp*(1.d0+dble(2.*np1-3.)*dble(x1(jp1)))
c          zp=zpmin+dzp*(1.d0+dble(2.*np1-3.)*dgx1(jp1))
          if(alpp.eq.0.d0)then
            xp=exp(-zp)
            xjacp=xp
          else
            xp=zp**(1.d0/alpp)
            xjacp=zp**(1.d0/alpp-1.d0)/alpp
          endif

          Gm1=0.d0
          do nm1=1,nmax
            do jm1=1,jmax
                zm=zmmin+dzm*(1.d0+dble(2.*nm1-3.)*dble(x1(jm1)))
c                zm=zmmin+dzm*(1.d0+dble(2.*nm1-3.)*dgx1(jm1))
                if(alpm.eq.0.d0)then
                  xm=exp(-zm)
                  xjacm=xm
                else
                  xm=zm**(1.d0/alpm)
                  xjacm=zm**(1.d0/alpm-1.d0)/alpm
                endif

              sxp=1.d0-xp
              sxm=1.d0-xm
              pGtab=1.d0
              if(dzp.ge.0.d0.and.dzm.ge.0.d0)then
                pGtab=omGam(xp,xm,b)
                if(pGtab.eq.0.d0)
     &          write(*,*)'j1=',xp,xm,xmmin,xmmax,dzp,dzm,jp1
              else
                pGtab=0.d0
                write(*,*)'error ?',dzp,dzm
              endif
              if(sxp.gt.0.d0.and.sxm.gt.0.d0)then
                if(dzm.ne.0.d0)then
                  Gm1=Gm1+pGtab*
     &dble(a1(jm1))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)*xjacm
c     &dga1(jm1)*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)*xjacm
c     &dble(a1(jm1))*Phiexact(0.,0.,1.,sxp,sxm,sy,b)*xjacm
                  else
                  Gp1=Gp1+pGtab*
     &dble(a1(jp1))*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)*xjacp
c     &dga1(jp1)*Phiexpo(0.,0.,1.,sxp,sxm,sy,b)*xjacp
c     &dble(a1(jp1))*Phiexact(0.,0.,1.,sxp,sxm,sy,b)*xjacp
                endif
c          write(*,*)'m=1',mtmp,Gm1,Gp1,pGtab,sxp,sxm
              endif

            enddo
          enddo
          if(dzm.ne.0.d0)Gp1=Gp1+Gm1*dble(a1(jp1))*dzm*xjacp
c          if(dzm.ne.0.d0)Gp1=Gp1+Gm1*dga1(jp1)*dzm*xjacp
        enddo
        enddo
        endif

        omWi=omWi+Gp1*dzp

        enddo
        enddo


      return
      end

c-----------------------------------------------------------------------
      double precision function Womint(sy,bh)   !---check---
c-----------------------------------------------------------------------
c - chi~(xp,xm)/2. for group of cut enhanced diagram giving
c the same final state integrated over xpr and xmr (with ideal G)
c bh - impact parameter between the pomeron ends;
c xh - fraction of the energy squared s for the pomeron;
c yp - rapidity for the pomeron;
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision omWi

      Womint=omWi(sy,bh)

      return
      end



c-----------------------------------------------------------------------
      double precision function WomGamint(bh)   !---check---
c-----------------------------------------------------------------------
c - chi~(xp,xm)/2. for group of integrated cut enhanced diagram giving
c the same final for proposal.
c bh - impact parameter between the pomeron ends;
c xh - fraction of the energy squared s for the pomeron;
c yp - rapidity for the pomeron;
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision omGamint

      WomGamint=omGamint(bh)

      return
      end

