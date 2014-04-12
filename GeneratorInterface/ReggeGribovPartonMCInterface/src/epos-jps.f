c-----------------------------------------------------------------------
      subroutine jpsifo(npjpsi)
c-----------------------------------------------------------------------
c   forms a jpsi
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/nucl3/phi,bimp
      parameter (ndep=129,ndet=129)
      common/cdep/xdep(ndep),qdep(ndep),wdep(ndep)
      common/cdet/xdet(ndet),qdet(ndet),wdet(ndet)
      parameter (nptj=129)
      common /cptj/xptj(nptj),qptj(nptj),wptj(nptj)
      parameter (mxbim=12)
      common/jpsi1/bimmax,kolran,delt,taumi,jpsinu,jpsidr,taudmx
      parameter (mxmass=20)

      parameter (nxmdk=20)
      parameter (ntjpsi=150)
      common/jpsi7/xydens(ntjpsi,mxbim,nxmdk,nxmdk),a4min,a4max
      common/jpsi8/xys(mxbim,nxmdk,nxmdk),a5min,a5max
      common/jpsi9/ami(ntjpsi,mxmass),a6min,a6max

      call utpri('jpsifo',ish,ishini,4)
      if(ish.ge.6)write(ifch,'(a)')' jpsi formation'

c     trigger
c     -------

      ymax=0.5
      ymin=-0.5

c     jpsi momenta
c     ------------
      id=441
      call idmass(id,amass)
      s=amass**2
 2    rqptj=rangen()*qptj(nptj)
      pt=utinvt(nptj,xptj,qptj,rqptj)
      phi=2.*pi*rangen()
      px=pt*cos(phi)
      py=pt*sin(phi)
      lo=0
    1 lo=lo+1
      if(lo.gt.10)call utstop('jpsifo: lo > 10 &')
      z=0.19*sqrt(-2*alog(rangen()))*cos(2*pi*rangen()) !1-dim gauss


      if(z.gt.1.)goto 1
      pz=z*engy/2*ransig()
      e=sqrt(s+px**2+py**2+pz**2)
      amt=sqrt(amass**2+pt**2)
      y=sign(1.,pz)*alog( (e+abs(pz))/amt )
      if(y.lt.ymin.or.y.gt.ymax)goto 2

c     fill /cptl/
c     -----------
      if(npjpsi.eq.0)then
        nptl=nptl+1
        npjpsi=nptl
      endif
      if(npjpsi.gt.mxptl)then
        print *,npjpsi,mxptl
        call utstop('jpsifo: npjpsi>mxptl&')
      endif
      istptl(npjpsi)=1
      idptl(npjpsi)=id
      pptl(1,npjpsi)=px
      pptl(2,npjpsi)=py
      pptl(3,npjpsi)=pz
      pptl(4,npjpsi)=e
      pptl(5,npjpsi)=amass
      kolran=1+rangen()*kolevt
      xorptl(1,npjpsi)=coord(1,kolran)
      xorptl(2,npjpsi)=coord(2,kolran)
      xorptl(3,npjpsi)=coord(3,kolran)
      xorptl(4,npjpsi)=coord(4,kolran)
      iorptl(npjpsi)=0
      jorptl(npjpsi)=0
      tivptl(1,npjpsi)=xorptl(4,npjpsi)
      tivptl(2,npjpsi)=ainfin
      ifrptl(1,npjpsi)=0
      ifrptl(2,npjpsi)=0
      if(ish.ge.6) then
        call alist("&",npjpsi,npjpsi)
        write (ifch,*) xorptl(1,npjpsi)
     $       ,xorptl(2,npjpsi),xorptl(3,npjpsi),xorptl(4,npjpsi)
     $       ,tivptl(1,npjpsi),tivptl(2,npjpsi)
        ii=iproj(kolran)
        jj=maproj+itarg(kolran)
        call alist("collision&",ii,ii)
        call alist("&",jj,jj)
      endif
      a4min=-15.
      a4max= 15.
      a5min=-15.
      a5max= 15.
      a6min= 2.
      a6max= 10.
      call utprix('jpsifo',ish,ishini,4)
      return
      end

c-----------------------------------------------------------------------
      function sptj(x)
c-----------------------------------------------------------------------
c     jpsi pt-distribution in 200 gev pp
c-----------------------------------------------------------------------
      a=0.95
      c=1/0.363
      z=x/a
      sptj=1/a*c**c/utgam1(c)*z**(c-1)*exp(-c*z)
      return
      end

c-----------------------------------------------------------------------
      subroutine jpsian(ifirst)
c-----------------------------------------------------------------------
c     jpsi analysis.
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      parameter (mxbim=12,ntjpsi=150,mxtauc=16)
      common/jpsi1/bimmax,kolran,delt,taumi,jpsinu,jpsidr,taudmx
      common/jpsi2/jjtot(mxbim),jjnuc(mxbim),jjjtau(mxbim,mxtauc)
      common/jpsi3/jjjtot(mxbim,ntjpsi),jjjdro(mxbim,ntjpsi)
      common/jpsi4/nnucl(mxbim,ntjpsi),nclose(mxbim,ntjpsi,3)
      common/jpsi5/ndrop(mxbim,ntjpsi),jjjnt(mxbim,mxtauc)
      parameter (mxmass=20,mxassy=20)
      common/jpsi6/ndrp2(mxbim,ntjpsi,mxmass,mxassy)
     &     ,ndrop3(mxbim,ntjpsi,mxmass,mxassy)
      parameter (nxmdk=20)
      common/jpsi7/xydens(ntjpsi,mxbim,nxmdk,nxmdk),a4min,a4max
      common/jpsi8/xys(mxbim,nxmdk,nxmdk),a5min,a5max
      common/jpsi9/ami(ntjpsi,mxmass),a6min,a6max
      common/jpsi10/ndrop0(mxbim,ntjpsi)

      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat /ctain/mtain
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/nucl3/phi,bimp
      parameter (ndep=129,ndet=129)
      common/cdep/xdep(ndep),qdep(ndep),wdep(ndep)
      common/cdet/xdet(ndet),qdet(ndet),wdet(ndet)

      common/c9ptl/tauptl(mxptl),ss0ptl(mxptl)

      call utpri('jpsian',ish,ishini,5)

      detap=(ypjtl-yhaha)*etafac
      detat=-yhaha*etafac
      tpro=dcosh(detap)
      zpro=dsinh(detap)
      ttar=dcosh(detat)
      ztar=dsinh(detat)

      jpsinu=1
      jpsidr=1

c      fac=1                     !    <-------- should be one finally
      taudmx=4
      rad=sqrt(0.62 / pi)
      taud=0
      nucia=0
      taumi=-2
      delt=0.1
      bimmax=amin1(rmproj+rmtarg,bmaxim)
      delbim=bimmax/mxbim
      ii=iproj(kolran)
      jj=maproj+itarg(kolran)


c.....first event: delete commom blocks...............................
      if(ifirst.eq.1)then
        ifirst=0
        do nbim=1,mxbim
          jjtot(nbim)=0
          jjnuc(nbim)=0
          do ix=1,nxmdk
            do iy=1,nxmdk
              xys(nbim,ix,iy)=0.
            enddo
          enddo
          do nt=1,ntjpsi
            jjjtot(nbim,nt)=0
            nnucl(nbim,nt)=0
            jjjdro(nbim,nt)=0
            ndrop(nbim,nt)=0
            ndrop0(nbim,nt)=0
            do kk=1,3
              nclose(nbim,nt,kk)=0
            enddo
            do ix=1,nxmdk
              do iy=1,nxmdk
                xydens(nt,nbim,ix,iy)=0.
              enddo
            enddo
          enddo
          do mm=1,mxtauc
            jjjtau(nbim,mm)=0
          enddo
        enddo
      endif

      nbim=1+int(bimevt/delbim)
      if(nbim.lt.0.or.nbim.gt.mxbim) goto 5
      jjtot(nbim)=jjtot(nbim)+1 !events pro bin

      do 1 i=1,nptl
        if(idptl(i).eq.441)j=i
 1    continue

c     if(jpsidr.eq.1)then
c     write(6,'(a,i5,a,f6.2,a,f6.2,a,f6.2)')'ip=',ip
c     *,'   rin= '
c     *,'   mass= ',pptl(5,ip)
c     enddo
c     endif
c     endif

      taumax=0.
      ttaus=taumi-delt
      do 2 nt=1,ntjpsi
        idrin=0
        ttaus=ttaus+delt        !increment of time
        if(ish.ge.6)write(ifch,*) 'ttaus:-->',ttaus,ii,jj
        jpsiex=1
        call jtain(j,xj,yj,zj,tj,n,1)
        if(n.eq.1.or.n.eq.2.or.n.eq.9)jpsiex=0 !goto 2
        if(jpsiex.eq.1)jjjtot(nbim,nt)=jjjtot(nbim,nt)+1

c       nucleons
c       --------

        if(jpsinu.eq.1.and.jpsiex.eq.1)then !test jpsi-nucleon collision
          do 6 i=1,maproj+matarg
            if(i.eq.ii.or.i.eq.jj)goto 6
            nnucl(nbim,nt)=nnucl(nbim,nt)+1
            t=sngl(ttaus)
            x=xorptl(1,i)+(t-xorptl(4,i))*pptl(1,i)/pptl(4,i)
            y=xorptl(2,i)+(t-xorptl(4,i))*pptl(2,i)/pptl(4,i)
            z=xorptl(3,i)+(t-xorptl(4,i))*pptl(3,i)/pptl(4,i)
            pde=(pptl(3,i)+pptl(3,j))/(pptl(4,i)+pptl(4,j))
            gam2i=1-pde**2
            if(gam2i.eq.0.)goto 6
            dist=sqrt((x-xj)**2+(y-yj)**2
     &           +1/gam2i*(z-zj-(t-tj)*pde)**2)
            if(dist.le.rad)then
              nclose(nbim,nt,1)=nclose(nbim,nt,1)+1
              nucia=1
              if(ish.ge.6)then
                write (ifch,*) "nucl dist:",dist,' dist(sig)='
     $               ,rad
                call alist("&",i,i)
                call alist("&",j,j)
              endif
            elseif(dist.le.rad+1)then
              nclose(nbim,nt,2)=nclose(nbim,nt,2)+1
            elseif(dist.le.rad+3)then
              nclose(nbim,nt,3)=nclose(nbim,nt,3)+1
            endif
 6        continue
        endif

c       particles
c       ---------
        do 8 i=1,nptl
c          if ( i.eq.ii.or.i.eq.jj ) goto 8
          call jtain(i,x,y,z,t,n,1)
          call jtaus(z,tz,sz)
          if(n.eq.1.or.n.eq.2.or.n.eq.9)goto 8
c         calculate s
c         -----------
          iad=iabs(idptl(i))
c          s=(pptl(4,i)+pptl(4,j))**2-(pptl(3,i)+pptl(3,j))**2
c     $         -(pptl(2,i)+pptl(2,j))**2-(pptl(1,i)+pptl(1,j))**2
          if ( iad.eq.120 .or. iad.eq.110 ) then !pion
            sig=1.              !
          elseif ( iad.eq.121 .or. iad.eq.111 ) then ! rho
            sig=1.              !
          elseif ( iad.eq.1120 .or. iad.eq.1220 ) then
            sig=3.0             ! ???? or 6 ????
          else
            goto 8
          endif
          call jtaus(zj,tzj,szj)         !????????????????? OK ?
          dist=sqrt((x-xj)**2+(y-yj)**2+(sz-szj)**2)
          if ( dist .lt. sqrt(0.1*sig/pi) ) then
            istptl(j)=2
            if(ish.ge.6)then
              write (ifch,*) "dist:",dist,' dist(sig)='
     $             ,sqrt(0.1*sig/pi),' sig=',sig
              call alist("&",i,i)
              call alist("&",j,j)
            endif
          endif
 8      continue

c     droplets
c     --------

        if(jpsidr.eq.1)then
          call jtaus(zj,tzj,szj)
          do 3 i=maproj+matarg+1,nptl
c...........x-y distribution of strings..............................
            if(istptl(i).eq.29.and.nt.eq.1)then
              call jtain(i,x,y,z,t,n,1)
              if(x.gt.a5min.and.x.lt.a5max.and.
     &             y.gt.a5min.and.y.lt.a5max)then
                ix=(x-a5min)/(a5max-a5min)*nxmdk + 1
                iy=(y-a5min)/(a5max-a5min)*nxmdk + 1
                xys(nbim,ix,iy)=xys(nbim,ix,iy)+pptl(5,i)
              endif
            endif
            if(istptl(i).gt.10)goto 3
            if(i.eq.j)goto 3

c...................................................................

            call jtain(i,x,y,z,t,n,1)
            if(n.eq.1.or.n.eq.2.or.n.eq.9)goto 3
            stop'jpsian: change!!!!        ' !call jintep(i,x,y,z,t,sz,eps,rho)
            if(eps.lt.aouni)goto 3 !min-dichte
            ndrop(nbim,nt)=ndrop(nbim,nt)+1 !droplets at time nt
            ndrop0(nbim,nt)=ndrop0(nbim,nt)+pptl(5,i) !mass
            des=0 !?????????????????????????????????
            o=sz+des
            u=sz-des
            r=0 !( xxxx(i) +sngl(ttaus) ) *fac
c..............assym-mass-distribution...............................
            assym=log(des/r)
            amass=pptl(5,i)
            a1min=-5.
            a1max=5.
            a2min=0.
            a2max=40.
            if(assym.ge.a1min.and.assym.lt.a1max
     &           .and.amass.ge.a2min.and.amass.lt.a2max
     &           ) then
              nassym=(assym-a1min)/(a1max-a1min)*mxassy+1
              namass=(amass-a2min)/(a2max-a2min)*mxmass+1
              ndrp2(nbim,nt,namass,nassym)=
     &             ndrp2(nbim,nt,namass,nassym)+1
            endif

c..............vol-mass-distribution...............................
            a3min=-2.
            a3max=3.
            v=log(pi*r**2.*2.*des)/log(10.)
            if(v.ge.a3min.and.v.lt.a3max
     &           .and.amass.ge.a2min.and.amass.lt.a2max
     &           ) then
              nv=(v-a3min)/(a3max-a3min)*mxassy+1
              namass=(amass-a2min)/(a2max-a2min)*mxmass+1
              ndrop3(nbim,nt,namass,nv)=
     &             ndrop3(nbim,nt,namass,nv)+1
            endif
c..............x-y distribution of droplet..............................
            ix=(x-a4min)/(a4max-a4min)*nxmdk + 1
            iy=(y-a4min)/(a4max-a4min)*nxmdk + 1
            xydens(nt,nbim,ix,iy)=xydens(nt,nbim,ix,iy)+eps

            if(jpsiex.eq.0)goto 3

c           if(mod(nt,10).eq.1)
c           write(6,'(f5.2,i6,5x,3f7.2,5x,4f6.2)')sngl(ttaus),i,
c           *              u,szj,o,sqrt((x-xj)**2+(y-yj)**2),r,v,eps

            if(szj.lt.u.or.szj.gt.o)goto 3
            if((x-xj)**2+(y-yj)**2.gt.r**2)goto 3

c           write(6,'(a,f5.2,a,i5,a)')'***** t=',sngl(ttaus)
c           *,' -- jpsi in droplet ',i,' *****'

            taud=taud+delt
            taumax=max(taud,taumax)

c           write (*,*) taud,taumax

            idrin=1
            jjjdro(nbim,nt)=jjjdro(nbim,nt)+1
 3        continue

c     if (idrin.ne.1)taud=max(taud-delt,0.)

          if (idrin.ne.1)taud=0.
        endif

 2    continue                  !end nt-loop

      ijmod=2
      if(ijmod.eq.2)taud=taumax
      if(nucia.eq.1)jjnuc(nbim)=jjnuc(nbim)+1
      if(taud.gt.0.)then
        do ntaud=1,mxtauc
          tauc=ntaud*taudmx/mxtauc
          if(taud.gt.tauc)jjjtau(nbim,ntaud)=jjjtau(nbim,ntaud)+1
          if(nucia.eq.1.or.taud.gt.tauc)
     &         jjjnt(nbim,ntaud)=jjjnt(nbim,ntaud)+1
        enddo
      else
        do ntaud=1,mxtauc
          if(nucia.eq.1)
     &         jjjnt(nbim,ntaud)=jjjnt(nbim,ntaud)+1
        enddo
      endif

    5 continue
      call utprix('jpsian',ish,ishini,5)
      return
      end

c-----------------------------------------------------------------------
      subroutine jtauan(is,im)
c-----------------------------------------------------------------------
c     display collision
c     im = ijk
c          k > 0   -->  post-script
c          j > 0   -->  povray
c            j = 1  -->  time and z in n-n cms
c            j = 2  -->  time and z on hyberbola
c          i > 0   -->  text  ( changes in alist per time step )
c     cut in zeta-x (or y) plane for tau
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat /ctain/mtain
      parameter (mxbim=12,ntjpsi=150,mxtauc=16)
      common/jpsi1/bimmax,kolran,delt,taumi,jpsinu,jpsidr,taudmx
      common/jpsi2/jjtot(mxbim),jjnuc(mxbim),jjjtau(mxbim,mxtauc)
      common/jpsi3/jjjtot(mxbim,ntjpsi),jjjdro(mxbim,ntjpsi)
      common/jpsi4/nnucl(mxbim,ntjpsi),nclose(mxbim,ntjpsi,3)
      common/jpsi5/ndrop(mxbim,ntjpsi),jjjnt(mxbim,mxtauc)
      parameter (mxmass=20,mxassy=20)
      common/jpsi6/ndrp2(mxbim,ntjpsi,mxmass,mxassy)
     &     ,ndrop3(mxbim,ntjpsi,mxmass,mxassy)
      parameter (nxmdk=20)
      common/jpsi7/xydens(ntjpsi,mxbim,nxmdk,nxmdk),a4min,a4max
      common/jpsi8/xys(mxbim,nxmdk,nxmdk),a5min,a5max
      common/jpsi9/ami(ntjpsi,mxmass),a6min,a6max
      common/jpsi10/ndrop0(mxbim,ntjpsi)
      character*20 name,nnrr
      character*28 filename
      character*12 color(20)
      character*12 colpo(20)
      logical lcalc!,zet
      dimension isch(mxptl)
c      zet=.true.
c      zet=.false.
      xmin=-10.
      xmax=10.
      zmin=-10.
      zmax=10.

c      zevent=float(nevent*jpsi)
      if(mod(im,10).ne.0)then
        name='tau-'
        n=1
 5      l=4
        ll=int(log(real(n))/log(10.))+1
        do ii=ll,1,-1
          l=l+1
          name(l:l)=char(48+mod(int(n/10**(ii-1)),10))
        enddo
        name(l+1:l+3)='.ps'
        l=l+3
        inquire(file=name(1:l),exist=lcalc)
        if (lcalc)then
          n=n+1
          goto 5
        endif
        write(*,*) 'jtauan name ',name
        ifps=92
        open(unit=ifps,file=name(1:l),status='unknown')
        WRITE(ifps,'(a)') '%!PS-Adobe-2.0'
        WRITE(ifps,'(a)') '%%Title: tt2.fig'
        WRITE(ifps,'(a)') '%%Orientation: Portrait'
        WRITE(ifps,'(a)') '%%BeginSetup'
        WRITE(ifps,'(a)') '%%IncludeFeature: *PageSize A4'
        WRITE(ifps,'(a)') '%%EndSetup'
        WRITE(ifps,'(a)') '%%EndComments'
        WRITE(ifps,*) '/l {lineto} bind def'
        WRITE(ifps,*) '/rl {rlineto} bind def'
        WRITE(ifps,*) '/m {moveto} bind def'
        WRITE(ifps,*) '/rm {rmoveto} bind def'
        WRITE(ifps,*) '/s {stroke} bind def'
        WRITE(ifps,*) '/gr {grestore} bind def'
        WRITE(ifps,*) '/gs {gsave} bind def'
        WRITE(ifps,*) '/cp {closepath} bind def'
        WRITE(ifps,*) '/tr {translate} bind def'
        WRITE(ifps,*) '/sc {scale} bind def'
        WRITE(ifps,*) '/sd {setdash} bind def'
        WRITE(ifps,*) '/sdo {[.01 .05] 0 sd} bind def'
        WRITE(ifps,*) '/sdf {[1 .0] 0 sd} bind def'
        WRITE(ifps,*) '/n {newpath} bind def'
        WRITE(ifps,*) '/slw {setlinewidth } bind def'
        write(ifps,*) '/srgb {setrgbcolor} bind def'
        write(ifps,*) '/lgrey      { 0 0.95 0.95 srgb} bind def'
        write(ifps,*) '/black      { 0 0 0 srgb} bind def'
        write(ifps,*) '/red        { 1 0 0 srgb} bind def  '
        write(ifps,*) '/green      { 0 1 0  srgb} bind def  '
        write(ifps,*) '/blue       { 0 0 1  srgb} bind def  '
        write(ifps,*) '/yellow     { 1 0.5 0  srgb} bind def  '
        write(ifps,*) '/turquoise  { 0 1 1  srgb} bind def  '
        write(ifps,*) '/purple     { 1 0 1  srgb} bind def  '
c.......write(ifps,*) '/  {   srgb} bind def  '
c.......write(ifps,*) '/  {   srgb} bind def  '
        write(ifps,*) '/ef {eofill} bind def'
        WRITE(ifps,'(a)') '%%EndProlog'
        WRITE(ifps,*) 'gsave'
        WRITE(ifps,*) '/Helvetica findfont 10 scalefont setfont'
      endif
      color(9)='lgrey     '
      color(1)='black     '
      color(2)='red       '
      color(3)='green     '
      color(4)='blue      '
      color(7)='yellow    '
      color(5)='turquoise '
      color(6)='purple    '
      colpo(1)='Red  '
      colpo(2)='Green  '
      colpo(3)='Blue  '
      colpo(4)='Yellow  '
      colpo(5)='Cyan '
      colpo(6)='Magenta  '
      colpo(7)='Black  '
      colpo(8)='Aquamarine '


      do i=1,mxptl
        isch(i)=0
      enddo

c      gray0=.95
c      gray1=0.
      iyb=0                     !????????????????????
      np=0
      nt=-10
      deltau=0.1
      taumin=-1
      ttaus=0
      do while (ttaus.lt.20.)
        nt=nt+1
       ! ttaus=dble(taumin+deltau*(factau**(1.*nt-1.)-1)/(factau-1.))
        ttaus=taumin+deltau*nt
        tau=ttaus
        np=np+1
        if(mod(im,10).ne.0)then
          write(ifps,'(a,i4)') '%%Page: number ',np
          write(ifps,'(a)') 'gsave'
          WRITE(ifps,*) '100 700 tr'
          scale=0.125
          WRITE(ifps,*) 1./scale,1./scale,' sc'
          WRITE(ifps,*) scale/2.,' slw'
          WRITE(ifps,*) '/Helvetica findfont ',15.*scale
     &         ,' scalefont setfont'
          write(ifps,*) color(1),' n ',zmin,xmin,' m ( tau:'
     $         ,tau,') show '
          WRITE(ifps,*) '/Helvetica findfont ',2.*scale
     &         ,' scalefont setfont'
        endif

*--------------------------------------------------------------------*
*------ povray ------------------------------------------------------*
*--------------------------------------------------------------------*

        if (mod(im/100,10).gt.0) then
          write (ifch,*)   "-----",np,", tau:",ttaus,"------"
        endif
        if (mod(im/10,10).gt.0) then
          write (nnrr,'(i5)') np
          li=6-log(1.*np+0.1)/log(10.)
          write (*,*)   "--->"//nnrr(li:5)//"<-----",li,ttaus
          ifpo=55
          filename="tau."//nnrr(li:5)//".pov"
          open(unit=ifpo,file=filename,status='unknown')
          write (ifpo,'(a)') '#include "colors.inc";'
c         write (ifpo,'(a)') '#include "shapes.inc" '
c         write (ifpo,'(a)') '#include "textures.inc" '
          write (ifpo,'(a)') 'background {color White} '
          write (ifpo,'(a)') 'camera {location <0,0,-120> '
          write (ifpo,'(a)') '     direction <0,0,2> look_at <0,0,0>} '
          write (ifpo,'(a)') 'light_source{<0,300,0> color White} '
          write (ifpo,'(a)') 'light_source{<0,5,-90> color White} '
          write (ifpo,'(a)') ' '
          write (ifpo,'(a)') ' '
        endif
        do i=1,nptl
          if (istptl(i).gt.1) goto 123
            if((tivptl(2,i)-tivptl(1,i)).lt.1e-3
     $           .and.idptl(i).gt.1000000.and.iyb.eq.0)
     $           then
              write (*,*) 'tiv1=tiv2 !!!!!!!!',i
              tivptl(2,i)=tivptl(1,i)+100.
            endif
c...........calculate coordinates....................................
            if(mod(im/10,10).eq.1) then !n-n cms frame
              if (istptl(i).gt.1
     $             .or.ttaus.lt.tivptl(1,i)
     $             .or.ttaus.gt.tivptl(2,i)) goto 123
              x=xorptl(1,i)+(ttaus-xorptl(4,i))*pptl(1,i)/pptl(4,i)
              y=xorptl(2,i)+(ttaus-xorptl(4,i))*pptl(2,i)/pptl(4,i)
              z=xorptl(3,i)+(ttaus-xorptl(4,i))*pptl(3,i)/pptl(4,i)
            else                !  hyperbola frame
              call jtain(i,x,y,z,t,n,0)
              call jtaus(z,tz,sz)
              z=sz
              if(n.ne.0) goto 123
            endif
c...........plot sphere or cylinder ................................
            if(idptl(i).gt.700000000)
     $           then
              if(mod(im/10,10).eq.1)then

              else
                des=0 !?????????????????????????????
                r=0   !(xxxx(i)+vrad*sngl(ttaus))
                o=sz+des
                u=sz-des
                print *,ttaus,o,u,r,x,y
                ic=4
                if (mod(im/10,10).gt.0) then
                  write (ifpo,111) o,x,y,u,x,y,r,colpo(ic)
                endif
              endif
c.............text output of changes in time step ...................
              if (mod(im/100,10).gt.0) then
                if(isch(i).eq.0)then
                  write (ifch,'("> ",$)')
                  call alist("&",i,i)
                  isch(i)=1
                endif
              endif
c$$$              o=sz+des
c$$$              u=sz-des
c$$$              r=(xxxx(i)+vrad*sngl(ttaus))
c$$$              rr2=r**2-(y-yb)**2
c$$$              if(rr2.gt.0.)then
c$$$                write(ifps,*)
c$$$     &               ,' n ',u,x-r,' m ',o,x-r,' l '
c$$$     &               ,o,x+r,' l ',u,x+r,' l cp s '
c$$$                write(ifps,*) ' n ',u,x-r,' m (',i,iorptl(i),') show '
c$$$              endif
            else
c.............cylinder................................................
              r=0.8
              ic=1
              if(abs(idptl(i)).lt.999) r=0.5
              if(iabs(idptl(i)).eq.1120) ic=2
              if(iabs(idptl(i)).eq.1220) ic=3
              if(iabs(idptl(i)).eq.441) ic=5
              if(mod(im/10,10).gt.0)then
                write (ifpo,110) z,x,y,r,colpo(ic) ! sphere
              endif
c.............text...................................................
              if(mod(im/100,10).gt.0)then
                if(isch(i).eq.0)then
                  write (ifch,'("> ",$)')
                  call alist("&",i,i)
                  isch(i)=1
                endif
              endif
            endif
            goto 124
 123      continue
c.........text................................
          if(mod(im/100,10).gt.0)then
            if(isch(i).eq.1)then
              write (ifch,'("< ",$)')
              call alist("&",i,i)
              isch(i)=0
            endif
          endif
 124      continue
        enddo
c.......
 110    format('sphere {<',G12.6,',',g12.6,',',g12.6,'>,',g12.6
     $       ,'pigment {color ',a,'}}')
 111    format('cylinder {<',
     $       G12.6,',',g12.6,',',g12.6,
     $       '>,<',
     $       G12.6,',',g12.6,',',g12.6,
     $       '>,',
     $       g12.6,
     $       'pigment {color ',a,'}}')
        if(mod(im/10,10).gt.0)then
          close(unit=ifpo)
        endif
*-------------------------------------------------------------------*
*-------   end   povray   ------------------------------------------*
*-------   begin post-script ---------------------------------------*
*-------------------------------------------------------------------*

        if(mod(im,10).eq.0) goto 159
        yb=-6.
        dy=12./12.
        yb=yb-dy/2
        do iyb=0,11
          yb=yb+dy
          WRITE(ifps,*) 'gsave'
          WRITE(ifps,*) (xmax-xmin)*1.1*float(int(iyb/4))
     &         ,-(xmax-xmin)*1.1*mod(iyb,4),' tr'
          write(ifps,*) ' n ',zmin,xmin,' m ',zmax,xmin,' l '
     &         ,zmax,xmax,' l ',zmin,xmax,' l cp s '
c          ttaus=dble(tau)
c.........particles in layer iyb.............
          do 10 i=1,nptl
            if (istptl(i).gt.1) goto 10
            if((tivptl(2,i)-tivptl(1,i)).lt.1e-3
     $           .and.idptl(i).gt.1000000.and.iyb.eq.0)
     $           then
              write (*,*) 'tiv1=tiv2 !!!!!!!!',i
              tivptl(2,i)=tivptl(1,i)+100.
            endif
            call jtain(i,x,y,z,t,n,0)
            call jtaus(z,tz,sz)
            if(n.ne.0) goto 10
            if(
     $           (is.eq.0.or.is.eq.i.or.is.eq.iorptl(i)))then
*.............
*.............   is  is the particle number to observe
*.............   if is=0 then all particles
*.............
c              .and.abs(y-yb).lt.dy/2)then
              des=0 !??????????????????????????????????
              if(iyb.eq.11.and
     $             .abs(tivptl(2,i)-tivptl(1,i)-100.).le.1e-4 ) then
                tivptl(2,i)=tivptl(1,i)+0.01
              endif
              o=sz+des
              u=sz-des
              r=0   !(xxxx(i)+vrad*sngl(ttaus))
              rr2=r**2-(y-yb)**2
              if(rr2.gt.0.)then
                r=sqrt(rr2)
c                write (*,*) i,des,o,u,r,y
                write(ifps,*)  color(mod(i,5)+2)
     &               ,' n ',u,x-r,' m ',o,x-r,' l '
     &               ,o,x+r,' l ',u,x+r,' l cp s '
                write(ifps,*) ' n ',u,x-r,' m (',i,iorptl(i),') show '
              endif
            elseif(abs(y-yb).lt.dy/2.and.zmin.lt.sz.and.sz.lt.zmax
     &             .and.xmin.lt.x.and.x.lt.xmax)then
              r=0.8
              ic=1
              if(abs(idptl(i)).lt.999)r=0.5
              if(abs(idptl(i)).lt.999)ic=2
              if(abs(idptl(i)).eq.1120)ic=3
              if(abs(idptl(i)).eq.1220)ic=4
              if(idptl(i).eq.441) ic=7

              io=iorptl(i)
              if(is.eq.0.or.io.eq.is)then
                write(ifps,*) ' n ',sz,x,r,0,360,' arc ',color(ic),' s '
                write(ifps,*) ' n ',sz-r,x,' m (',i,io,') show '
              endif
            endif
 10       continue
          write(ifps,*) color(1),' n ',zmin,xmin,' m (',yb,') show '
          WRITE(ifps,*) 'grestore'
        enddo                   !yb bin
        write(ifps,'(a)') 'grestore'
        write(ifps,*) 'showpage'
 159    continue
      enddo


c        write(ifps,*) ' n ',y0,x0,' m ',y1,x0,' l ',y1,x1,' l '
c     &    ,y0,x1,' l cp s '
c        write(ifps,*) ' n ',(y0+y1)/2-10.*scale,(x0+x1)/2-5.*scale
c     &    ,' m (',ii,jj,') show '
c        ii=nob-1


      if(mod(im,10).ne.0)then
        write(ifps,*) 'gr'
        write(ifps,'(a)') '%%Trailer'
        write(ifps,'(a,i4)') '%%Pages: ',np
        write(ifps,'(a)') '%%EOF'
        close(unit=ifps)
      endif
      return
      end


c-----------------------------------------------------------------------
      subroutine jpsihi
c-----------------------------------------------------------------------
c     histogram
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxbim=12,ntjpsi=150,mxtauc=16)
      common/jpsi1/bimmax,kolran,delt,taumi,jpsinu,jpsidr,taudmx
      common/jpsi2/jjtot(mxbim),jjnuc(mxbim),jjjtau(mxbim,mxtauc)
      common/jpsi3/jjjtot(mxbim,ntjpsi),jjjdro(mxbim,ntjpsi)
      common/jpsi4/nnucl(mxbim,ntjpsi),nclose(mxbim,ntjpsi,3)
      common/jpsi5/ndrop(mxbim,ntjpsi),jjjnt(mxbim,mxtauc)
      parameter (mxmass=20,mxassy=20)
      common/jpsi6/ndrp2(mxbim,ntjpsi,mxmass,mxassy)
     &     ,ndrop3(mxbim,ntjpsi,mxmass,mxassy)
      parameter (nxmdk=20)
      common/jpsi7/xydens(ntjpsi,mxbim,nxmdk,nxmdk),a4min,a4max
      common/jpsi8/xys(mxbim,nxmdk,nxmdk),a5min,a5max
      common/jpsi9/ami(ntjpsi,mxmass),a6min,a6max
      common/jpsi10/ndrop0(mxbim,ntjpsi)

      zevent=float(nevent*jpsi)

      write(ifhi,'(a)')    'cd /users/theoric/werner/histo/newdata'
      write(ifhi,'(a)')       'newpage'

c     suppression as a function of b
c     ------------------------------

      write(ifhi,'(a)')       'zone 1 2 1 openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')       'text 0 0 "yaxis J(b) et Jnuc(b) / J"'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto'
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      write(ifhi,'(4e12.4)')bimmax-bim,jjtot(j)/zevent,0.,zevent
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)') 'text 0 0 " "'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto'
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      write(ifhi,'(4e12.4)')bimmax-bim,jjnuc(j)/zevent,0.,zevent
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'


      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')       'text 0 0 "yaxis survival ratio"'
      write(ifhi,'(a,3e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0.2 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
        bim=(j-0.5)*bimmax/mxbim
        rat=0
        if(jjtot(j).gt.0.)rat=float(jjtot(j)-jjnuc(j))/jjtot(j)
        write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      if(maproj.eq.208.and.matarg.eq.208)then
        write(ifhi,'(a)')       'closehisto plot 0-'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'set fmsc 1.0'
        write(ifhi,'(a,f4.1,a)')'column c1 = ( ',bimmax,' - c1  )'
        write(ifhi,'(a)')       'column c2 = ( c2 * 0.02 )'
        write(ifhi,'(a)')       'input na50 ratio-b plot 0'
      else
        write(ifhi,'(a)')       'closehisto plot 0'
      endif

c     nr of jpsi vs t
c     ---------------

      write(ifhi,'(a)')       'zone 3 4 1'
      do nb=mxbim,1,-1
        bim=(nb-0.5)*bimmax/mxbim
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
        write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
        write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
        write(ifhi,'(a)')       'text 0 0 "yaxis J(b,t) / J"'
        write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
        write(ifhi,'(3a)')      'yrange',' 0 ',' auto'
        write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
        write(ifhi,'(a)')       'array 4'
        do j=1,ntjpsi
          tau=taumi+(j-0.5)*delt
          write(ifhi,'(4e12.4)')tau,float(jjjtot(nb,j))/nevent,0.,nevent
        enddo
        write(ifhi,'(a)')       '  endarray'
        write(ifhi,'(a)')       'closehisto plot 0'
      enddo

c     nr of nucleons vs t
c     -------------------

      if(jpsinu.eq.1)then
        write(ifhi,'(a)')       'zone 3 4 1'
        do nb=mxbim,1,-1
          bim=(nb-0.5)*bimmax/mxbim
          write(ifhi,'(a)')       'openhisto'
          write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
          write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
          write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
          write(ifhi,'(a)')       'text 0 0 "yaxis N(b,t) / J"'
          write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
          write(ifhi,'(3a)')'yrange',' 0 ',' auto'
          write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
          write(ifhi,'(a)')       'array 4'
          do j=1,ntjpsi
            tau=taumi+(j-0.5)*delt
            rat=0
            if(jjjtot(nb,j).gt.0)rat=nnucl(nb,j)/float(jjjtot(nb,j))
            write(ifhi,'(4e12.4)')tau,rat,0.,float(jjjtot(nb,j))
          enddo
          write(ifhi,'(a)')       '  endarray'
          write(ifhi,'(a)')       'closehisto plot 0'
        enddo
      endif

c     nr of close nucleons vs t
c     -------------------------

      if(jpsinu.eq.1)then
        write(ifhi,'(a)')       'zone 3 4 1'
        do nb=mxbim,1,-1
          bim=(nb-0.5)*bimmax/mxbim
          write(ifhi,'(a)')       'openhisto'
          write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
          write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
          write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
          write(ifhi,'(a)')       'text 0 0 "yaxis Nclose(b,t) / J"'
          write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
          write(ifhi,'(3a)')'yrange',' 0 ',' auto '
          write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
          write(ifhi,'(a)')       'array 4'
          do j=1,ntjpsi
            tau=taumi+(j-0.5)*delt
            rat=0
            if(jjjtot(nb,j).ne.0)rat=nclose(nb,j,1)/float(jjjtot(nb,j))
            write(ifhi,'(4e12.4)')tau,rat,0.,float(jjjtot(nb,j))
          enddo
          write(ifhi,'(a)')       '  endarray'
          write(ifhi,'(a)')       'closehisto plot 0'
        enddo
      endif

c     number of droplets
c     ------------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 3 4 1'
      do nb=mxbim,1,-1
      bim=(nb-0.5)*bimmax/mxbim
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
      write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
      write(ifhi,'(a)') 'text 0 0 "yaxis D(b,t) / J"'
      write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=1,ntjpsi
      tau=taumi+(j-0.5)*delt
      rat=0
      if(jjjtot(nb,j).ne.0)rat=ndrop(nb,j)/float(jjjtot(nb,j))
      write(ifhi,'(4e12.4)')tau,rat,0.,float(jjjtot(nb,j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'
      enddo
      endif

c     number of droplets
c     ------------------

      if(jpsidr.eq.1)then
        write(ifhi,'(a)')       'zone 3 4 1'
        do nb=mxbim,1,-1
          bim=(nb-0.5)*bimmax/mxbim
          write(ifhi,'(a)')       'openhisto'
          write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
          write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
          write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
          write(ifhi,'(a)') 'text 0 0 "yaxis mass*D(b,t) / J"'
          write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
          write(ifhi,'(3a)')'yrange',' 0 ',' auto '
          write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
          write(ifhi,'(a)')       'array 4'
          do j=1,ntjpsi
            tau=taumi+(j-0.5)*delt
            rat=0
            if(jjjtot(nb,j).ne.0)rat=ndrop0(nb,j)/float(jjjtot(nb,j))
            write(ifhi,'(4e12.4)')tau,rat,0.,float(jjjtot(nb,j))
          enddo
          write(ifhi,'(a)')       '  endarray'
          write(ifhi,'(a)')       'closehisto plot 0'
        enddo
      endif

c$$$c     assymetry and mass of droplets
c$$$c     ------------------
c$$$
c$$$      if(jpsidr.eq.1)then
c$$$         do nb=11,1,-5
c$$$            bim=(nb-0.5)*bimmax/mxbim
c$$$            write(ifhi,'(a)')       'zone 3 4 1'
c$$$            do nt=40,150,10
c$$$               tau=taumi+(nt-0.5)*delt
c$$$
c$$$      write(ifhi,'(a)')       'openhisto'
c$$$      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
c$$$      write(ifhi,*) 'text .1 .90 "b= ',bim,' fm ','t=',tau,'"'
c$$$      write(ifhi,'(a)')       'text 0 0 "xaxis mass "'
c$$$      write(ifhi,'(a)') 'text 0 0 "yaxis assym"'
c$$$      write(ifhi,'(a,2e11.3)')'xrange ',0.,40.
c$$$      write(ifhi,'(a,2e11.3)')'yrange ',-5.,5.
c$$$      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
c$$$      write(ifhi,'(a,i)')       'set ityp2d 5'
c$$$      write(ifhi,'(a,i)')       'array2d ',mxmass
c$$$
c$$$      do i=1,mxassy
c$$$         do j=1,mxmass
c$$$            rat=0.
c$$$            rat=float(ndrp2(nb,nt,j,i))
c$$$     &           /zevent        ! nevent
c$$$     &           /(40./20)      ! mass-bin
c$$$     &           /(10./20)      ! log(assy)-bin
c$$$     &           /1.            ! b-bin
c$$$            write (ifhi,*) rat
c$$$         enddo
c$$$      enddo
c$$$
c$$$      write(ifhi,'(a)')       '  endarray'
c$$$      write(ifhi,'(a)')       'closehisto plot2d'
c$$$      enddo
c$$$      enddo
c$$$      endif
c$$$
c$$$c     volume and mass of droplets
c$$$c     ------------------
c$$$
c$$$      if(jpsidr.eq.1)then
c$$$         do nb=11,1,-5
c$$$            bim=(nb-0.5)*bimmax/mxbim
c$$$            write(ifhi,'(a)')       'zone 3 4 1'
c$$$            do nt=40,150,10
c$$$               tau=taumi+(nt-0.5)*delt
c$$$
c$$$      write(ifhi,'(a)')       'openhisto'
c$$$      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
c$$$      write(ifhi,*) 'text .1 .90 "b= ',bim,' fm ','t=',tau,'"'
c$$$      write(ifhi,'(a)')       'text 0 0 "xaxis mass "'
c$$$      write(ifhi,'(a)') 'text 0 0 "yaxis volume"'
c$$$      write(ifhi,'(a,2e11.3)')'xrange ',0.,40.
c$$$      write(ifhi,'(a,2e11.3)')'yrange ',-2.,3.
c$$$      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
c$$$      write(ifhi,'(a,i)')       'array2d ',mxmass
c$$$
c$$$      do i=1,mxassy
c$$$         do j=1,mxmass
c$$$            rat=0.
c$$$            rat=float(ndrop3(nb,nt,j,i))
c$$$     &           /zevent        ! nevent
c$$$     &           /(40./20)      ! mass-bin
c$$$     &           /(5./20)       ! log(v)-bin
c$$$     &           /1.            ! b-bin
c$$$            write (ifhi,*) rat
c$$$         enddo
c$$$      enddo
c$$$
c$$$      write(ifhi,'(a)')       '  endarray'
c$$$      write(ifhi,'(a)')       'closehisto plot2d'
c$$$      enddo
c$$$      enddo
c$$$      endif
c$$$
c$$$c     xy of droplets
c$$$c     ------------------
c$$$
c$$$      if(jpsidr.eq.1)then
c$$$        write(ifhi,'(a)')       'zone 3 4 1'
c$$$        do nb=11,1,-5
c$$$          bim=(nb-0.5)*bimmax/mxbim
c$$$          do nt=40,150,10
c$$$            tau=taumi+(nt-0.5)*delt
c$$$            write(ifhi,'(a)')       'openhisto'
c$$$            write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
c$$$            write(ifhi,*) 'text .1 .90 "xy b= ',bim,' fm ','t=',tau,'"'
c$$$            write(ifhi,'(a)')       'text 0 0 "xaxis x "'
c$$$            write(ifhi,'(a)') 'text 0 0 "yaxis y "'
c$$$            write(ifhi,'(a,2e11.3)')'xrange ',a4min,a4max
c$$$            write(ifhi,'(a,2e11.3)')'yrange ',a4min,a4max
c$$$            write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
c$$$            write(ifhi,'(a,i)')       'array2d ',nxmdk
c$$$            do i=1,nxmdk
c$$$              do j=1,nxmdk
c$$$                rat=0.
c$$$                rat=xydens(nt,nb,i,j)
c$$$     &            /zevent       ! nevent
c$$$     &            /((a4max-a4min)/float(nxmdk)) ! x-bin
c$$$     &            /((a4max-a4min)/float(nxmdk)) ! y-bin
c$$$c...............&                 /1.      ! b-bin
c$$$                write (ifhi,*) rat
c$$$              enddo
c$$$            enddo
c$$$            write(ifhi,'(a)')       '  endarray'
c$$$            write(ifhi,'(a)')       'closehisto plot2d'
c$$$          enddo
c$$$        enddo
c$$$      endif
c$$$
c$$$
c$$$c$$$c.....michael-verteilung.........................................
c$$$c$$$      write(ifhi,'(a)')       'zone 3 4 1'
c$$$c$$$      do j=1,ntjpsi
c$$$c$$$        tau=taumi+(j-0.5)*delt
c$$$c$$$
c$$$c$$$        write(ifhi,'(a)')       'openhisto'
c$$$c$$$        write(ifhi,'(a)')       'htyp lin xmod lin ymod log'
c$$$c$$$        write(ifhi,'(a,f5.2,a)')'text .1 .90 "tau= ',tau,' fm"'
c$$$c$$$        write(ifhi,'(a)')       'text 0 0 "xaxis mass t (fm)"'
c$$$c$$$        write(ifhi,'(a)')       'text 0 0 "yaxis Nclose(b,t) / J"'
c$$$c$$$        write(ifhi,'(a,2e11.3)')'xrange',a6min,a6max
c$$$c$$$        write(ifhi,'(3a)')'yrange',' 0 ',' auto '
c$$$c$$$        write(ifhi,'(a)')       'array 2'
c$$$c$$$        do k=1,mxmass
c$$$c$$$          rat=0.
c$$$c$$$          amass=(k-0.5)/(mxmass)*(a6max-a6min)+a6min
c$$$c$$$          rat=ami(j,k)/zevent/(a6max-a6min)*mxmass
c$$$c$$$          write(ifhi,'(2e12.4)')amass,rat
c$$$c$$$        enddo
c$$$c$$$        write(ifhi,'(a)')       '  endarray'
c$$$c$$$        write(ifhi,'(a)')       'closehisto plot 0'
c$$$c$$$      enddo
c$$$
c$$$c     xy of strings
c$$$c     -------------
c$$$
c$$$      if(jpsidr.eq.1)then
c$$$        write(ifhi,'(a)')       'zone 3 4 1'
c$$$        do nb=11,1,-1
c$$$          bim=(nb-0.5)*bimmax/mxbim
c$$$          write(ifhi,'(a)')       'openhisto'
c$$$          write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
c$$$          write(ifhi,*) 'text .1 .90 "xy b= ',bim,' fm ','t=',tau,'"'
c$$$          write(ifhi,'(a)')       'text 0 0 "xaxis x "'
c$$$          write(ifhi,'(a)') 'text 0 0 "yaxis y "'
c$$$          write(ifhi,'(a,2e11.3)')'xrange ',a4min,a4max
c$$$          write(ifhi,'(a,2e11.3)')'yrange ',a4min,a4max
c$$$          write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
c$$$          write(ifhi,'(a,i)')       'array2d ',nxmdk
c$$$          do i=1,nxmdk
c$$$             do j=1,nxmdk
c$$$                rat=0.
c$$$                rat=xys(nb,i,j)
c$$$     &               /zevent    ! nevent
c$$$     &               /((a5max-a5min)/float(nxmdk)) ! x-bin
c$$$     &               /((a5max-a5min)/float(nxmdk)) ! y-bin
c$$$c...............&                 /1.      ! b-bin
c$$$                write (ifhi,*) rat
c$$$             enddo
c$$$          enddo
c$$$          write(ifhi,'(a)')       '  endarray'
c$$$          write(ifhi,'(a)')       'closehisto plot2d'
c$$$       enddo
c$$$      endif


c     fraction of jpsis in a droplet
c     ------------------------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 3 4 1'
      do nb=mxbim,1,-1
      bim=(nb-0.5)*bimmax/mxbim
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "b= ',bim,' fm"'
      write(ifhi,'(a)')       'text 0 0 "xaxis time t (fm)"'
      write(ifhi,'(a)') 'text 0 0 "yaxis Jdrop(b,t) / J"'
      write(ifhi,'(a,2e11.3)')'xrange',taumi,taumi+ntjpsi*delt
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=1,ntjpsi
      tau=taumi+(j-0.5)*delt
      rat=0
      if(jjjtot(nb,j).ne.0)rat=jjjdro(nb,j)/float(jjjtot(nb,j))
      write(ifhi,'(4e12.4)')tau,rat,0.,float(jjjtot(nb,j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'
      enddo
      endif

c     fraction of jpsis with taud gt tauc
c     -----------------------------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 2 4 1'
      do ntauc=1,mxtauc
      tauc=ntauc*(taudmx/mxtauc)
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "tauc= ',tauc,' fm/c"'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')
     *'text 0 0 "yaxis J(b, taud) / J(b)"'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      rat=0
      if(jjtot(j).gt.0.)rat=float(jjjtau(j,ntauc))/jjtot(j)
      write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'
      enddo
      endif

c     droplet survival ratio
c     --------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 2 4 1'
      do ntauc=1,mxtauc
      tauc=ntauc*(taudmx/mxtauc)
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "tauc= ',tauc,' fm/c"'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')      'text 0 0 "yaxis droplet survival ratio"'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      rat=0
      if(jjtot(j).gt.0.)rat=float(jjtot(j)-jjjtau(j,ntauc))/jjtot(j)
      write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0'
      enddo
      endif

c     total approx. survival ratio
c     --------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 2 4 1'
      do ntauc=1,mxtauc
      tauc=ntauc*(taudmx/mxtauc)
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "tauc= ',tauc,' fm/c"'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')      'text 0 0 "yaxis tot. ap. survival ratio"'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      rat=0
      if(jjtot(j).gt.0.)rat=float(jjtot(j)-jjjtau(j,ntauc))/jjtot(j)
      write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto '
      if(maproj.eq.208.and.matarg.eq.208)then
      write(ifhi,'(a,a)')   'openhisto htyp lfu input jpbpb',
     &        ' j-nucl mult plot 0- '
      write(ifhi,'(a)')'openhisto htyp ldo input jpbpb j-nucl plot 0- '
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'set fmsc 1.0'
      write(ifhi,'(a)')       'column c1 = ( 11.9 - c1  )'
      write(ifhi,'(a)')       'column c2 = ( c2 * 0.019 )'
      write(ifhi,'(a)')       'input na50 ratio-b plot 0'
      elseif(maproj.eq.32.and.matarg.eq.32)then
         write(ifhi,'(a,a)')   'openhisto htyp lfu input jss',
     &        ' j-nucl mult plot 0- '
      write(ifhi,'(a)')'openhisto htyp ldo input jss j-nucl plot 0 '
      endif
      enddo
      endif

c     total survival ratio
c     --------------

      if(jpsidr.eq.1)then
      write(ifhi,'(a)')       'zone 2 4 1'
      do ntauc=1,mxtauc
      tauc=ntauc*(taudmx/mxtauc)
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lfu xmod lin ymod lin'
      write(ifhi,'(a,f5.2,a)')'text .1 .90 "tauc= ',tauc,' fm/c"'
      write(ifhi,'(a)')       'text 0 0 "xaxis bmax-b (fm)"'
      write(ifhi,'(a)')      'text 0 0 "yaxis total survival ratio"'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bimmax
      write(ifhi,'(3a)')'yrange',' 0 ',' auto '
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      rat=0
      if(jjtot(j).gt.0.)rat=float(jjtot(j)-jjjnt(j,ntauc))/jjtot(j)
      write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      write(ifhi,'(a)')       'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp ldo xmod lin ymod lin'
      write(ifhi,'(a)')       'columnweight 4 column c4 = ( 0 ) '
      write(ifhi,'(a)')       'array 4'
      do j=mxbim,1,-1
      bim=(j-0.5)*bimmax/mxbim
      rat=0
      if(jjtot(j).gt.0.)rat=float(jjtot(j)-jjnuc(j))/jjtot(j)
      write(ifhi,'(4e12.4)')bimmax-bim,rat,0.,float(jjtot(j))
      enddo
      write(ifhi,'(a)')       '  endarray'
      if(maproj.eq.208.and.matarg.eq.208)then
      write(ifhi,'(a)')       'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'set fmsc 1.0'
      write(ifhi,'(a)')       'column c1 = ( 11.9 - c1  )'
      write(ifhi,'(a)')       'column c2 = ( c2 * 0.019 )'
      write(ifhi,'(a)')       'input na50 ratio-b plot 0'
      else
      write(ifhi,'(a)')       ' closehisto plot 0'
      endif
      enddo
      endif

      end

