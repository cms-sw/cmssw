c  reshuffled from sem, sto, sha

c    contains psahot and related stuff
c             ------



c-----------------------------------------------------------------------
      subroutine psahot(kcol,ncolp,iret)
c-----------------------------------------------------------------------
c psahot - showering (semihard parton-parton interaction)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'
      double precision ept(4),ept1(4),xx,wpt(2),eprt,pl,plprt,wplc
     *,wp0,wm0,s,si,smin,xmin,xp1,wpi,wmi,xp,xm,wp1,wm1,wp2,wm2
     *,wpq,wmq,wpi1,wmi1,pxh,pyh,pth,xmm,xp2,xg,zg,smax,xmax
     *,xmax1,xmin1,zp0,psutz,xpmax0,xpmin0,gb0,tmax0,tmin0,zpm,gb
     *,gbyj,tmax,tmin,t,gb7,x,s2,discr,qt,qt2,x1min,x1max,t1min,t1max
     *,t1,xq1,qq,qqmax,qqmin,pt2,xmin2,ptnew,xpmin,xpmax,xm0,psuds
c     *,xprh,xmrh
      dimension ep3(4),ey(3),bx(6)
     *,qmin(2),iqc(2),nqc(2),ncc(2,2),amqt(4)
      parameter (mjstr=20000)
      common /psar7/  delx,alam3p,gam3p
      common /psar29/eqj(4,mjstr),iqj(mjstr),ncj(2,mjstr),ioj(mjstr),nj
      common /psar30/iorj(mjstr),ityj(mjstr),bxj(6,mjstr),q2j(mjstr)
      common /testj/  ajeth(4),ajete(5),ajet0(7)
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      integer icp(2),ict(2),nemis(2)
      integer jcp(nflav,2),jct(nflav,2),jcpr(nflav,2),jctr(nflav,2)
      common/cprtx/nprtjx,pprtx(5,2)/ciptl/iptl

      call utpri('psahot',ish,ishini,3)

      iret=0
      alpq=-(alppar+1.)/2.
      qqcut=q2min  !????????????pt2cut


      nptl1=nptl
      iptl=nppr(ncolp,kcol)
      ip=iproj(kcol)
      it=itarg(kcol)
      do i=1,2
        icp(i)=icproj(i,ip)
        ict(i)=ictarg(i,it)
      enddo
      idpomr=idhpr(ncolp,kcol)
      bpomr=bhpr(ncolp,kcol)

      q2finsave=q2fin
      zzzz=zparpro(kcol)+zpartar(kcol)
      zz=1.+zoeinc*zzzz**2               !<-----
      q2fin=q2fin*zz

c      print *,kcol,zzzz,zz,q2fin

      ajeth(idpomr+1)=ajeth(idpomr+1)+1.
c      write(*,*)ajeth
      idfpomr=idfpr(ncolp,kcol)
      if(ish.ge.3)write(ifch,*)'Start psahot (icp,ict):',ip,icp,it,ict
     *,ncolp,kcol,iptl,idpomr,idfpomr,bpomr


      if(idfpomr.eq.0)stop'idfpomr??????'
      if(ish.ge.3)then
        write(ifch,20)iptl
     *  ,sqrt(pptl(1,iptl)**2+pptl(2,iptl)**2),pptl(3,iptl)
     *  ,pptl(4,iptl),pptl(5,iptl)
     *  ,istptl(iptl),ityptl(iptl)
20      format(1x,i4,3x,4(e11.5,1x),i2,1x,i3)
      endif
      istptl(iptl)=31
      
csp  initialise to 0
1        do i=1,nj
      ncj(1,i)=0
      ncj(2,i)=0
      enddo  
csp end initialisation

      nj=0
      nptl=nptl1
      if(iremn.ge.2)then
        call iddeco(icp,jcp)
        call iddeco(ict,jct)
      endif

      wp0=dsqrt(xpr(ncolp,kcol))*dexp(ypr(ncolp,kcol))*dble(engy)     !???? new
      wm0=dsqrt(xpr(ncolp,kcol))*dexp(-ypr(ncolp,kcol))*dble(engy)    !double


      amqt(1)=sqrt(sngl(xxp1pr(ncolp,kcol)**2+xyp1pr(ncolp,kcol)**2))
      amqt(2)=sqrt(sngl(xxp2pr(ncolp,kcol)**2+xyp2pr(ncolp,kcol)**2))
      amqt(3)=sqrt(sngl(xxm2pr(ncolp,kcol)**2+xym2pr(ncolp,kcol)**2))
      amqt(4)=sqrt(sngl(xxm1pr(ncolp,kcol)**2+xym1pr(ncolp,kcol)**2))
      amqpt=amqt(1)+amqt(2)+amqt(3)+amqt(4)

      s2min=4.*q2min
      if(sngl(wp0*wm0).le.(sqrt(s2min)+amqpt)**2)then
        if(ish.ge.1)then
          call utmsg('psahot: insufficient pomeron mass&')
          write (ifch,*)'mass:',dsqrt(wp0*wm0),amqpt+sqrt(s2min)
          call utmsgf
        endif
        iret=1
        goto 16
      endif

      ih=iproj(kcol)
      jh=itarg(kcol)
c      xprh=xpp(ih)
c      xmrh=xmt(jh)
      rp=r2had(iclpro)+r2had(icltar)+slopom*log(engy**2)
      z=exp(-bpomr**2/(4.*.0389*rp))

      if(z.eq.0)then
       write(ifch,*)'psahot : z,ih,jh ! -> ',z,ih,jh
       call gakli2(ih,ih)
       call gakli2(jh,jh)
       call gakli2(iptl,iptl)
       stop
      endif

      do l=1,4
        bx(l)=xorptl(l,iptl)
      enddo
      bx(5)=tivptl(1,iptl)
      bx(6)=tivptl(2,iptl)
      ity=ityptl(iptl)

      if(idpomr.eq.0)then        !gg-pomeron
        iqq=0   !type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
        pxh=0.d0  !p_x for sh pomeron
        pyh=0.d0  !p_y for sh pomeron
      elseif(idpomr.eq.1)then    !qg-pomeron
        iqq=1
        pxh=xxp1pr(ncolp,kcol)
        pyh=xyp1pr(ncolp,kcol)
        amqpt=amqpt-amqt(1)
      elseif(idpomr.eq.2)then    !gq-pomeron
        iqq=2
        pxh=xxm2pr(ncolp,kcol)
        pyh=xym2pr(ncolp,kcol)
        amqpt=amqpt-amqt(3)
      elseif(idpomr.eq.3)then    !qq-pomeron
        iqq=3
        pxh=xxp1pr(ncolp,kcol)+xxm2pr(ncolp,kcol)
        pyh=xyp1pr(ncolp,kcol)+xym2pr(ncolp,kcol)
        amqpt=amqpt-amqt(1)-amqt(3)
      else
        stop'unknown pomeron'
      endif
      pth=pxh**2+pyh**2

      nj0=nj
      if(ish.ge.6)then
        write(ifch,*)'iptl,nptl,wp0,wm0,z,iqq,bx:'
        write(ifch,*) iptl,nptl,wp0,wm0,z,iqq,bx
      endif

      s=wp0*wm0                !lc+*lc- for the semihard interaction
      smin=dble(s2min)+pth          !mass cutoff for the hard pomeron
      xmin=smin/s
      smax=(dsqrt(s)-dble(amqpt))**2  !max mass for the hard pomeron
      xmax=smax/s
      !wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
      if(iqq.ne.3)then     !not for val-val
c xmin is the minimum value of zpm which is the fraction of the Pomeron
c energy going into the hard part. If zpm=1 nothing left is the soft
c preevolution part (momentum of soft string ends)
c        print *,xmin,zopinc*fegypp*exp(-bpomr**2/b2xscr),xmax
c     *,smin,max(xmin,min(0.99*xmax,zzzz*zzsoft))*ss
      xmin=max(xmin,min((sqrt(s)-dble(amqpt+ammsdd))**2/s
     &  ,dble(max(2.*fegypp*exp(-bpomr**2/b2xscr),zzzz)*zopinc)))
c      xmin=max(xmin,1d0-exp(-dble(zzsoft*zzzz**2)))  !???????????????
      smin=xmin*s         
      endif
      !wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
       if(smax.le.smin)then
        write (ifmt,*)'smax,smin',smax,smin
        iret=1
        goto 16
      endif
      xpmax0=psutz(s,smin,dble(amqpt**2))   !max x+ for the hard pomeron
      xpmin0=xmin/xpmax0              !min x+ for the hard pomeron
      xp1=wp0/dble(engy)              !lc+ share for the semihard interaction
      xp2=wm0/dble(engy)              !lc- share for the semihard interaction


c-----------------------------------------------------------------
c  determine LC momenta wpi,wmi for hard Pomeron
c-----------------------------------------------------------------

      if(ish.ge.4)write(ifch,*)
     & 'determine LC momenta wpi,wmi for hard Pomeron'

      if(iremn.ge.2)then
        if(iclpro.eq.2)then
          if(iabs(idptl(ip)).eq.1120)then !proj=proton
            nquu1=jcp(1,1)+jcp(1,2)
            nqud1=jcp(2,1)+jcp(2,2)
          elseif(iabs(idptl(ip)).eq.1220)then !proj=neutron
            nquu1=jcp(2,1)+jcp(2,2)
            nqud1=jcp(1,1)+jcp(1,2)
          else    !to avoid flavor problem with exotic projectile (but should not happen (only gg)
            nquu1=0
            nqud1=0
          endif
        elseif(iclpro.eq.1)then
          if(iabs(idptl(ip)).eq.120)then
            nquu1=jcp(1,1)+jcp(1,2)
            nqud1=jcp(2,1)+jcp(2,2)
          else    !to avoid flavor problem with exotic projectile (but should not happen (only gg)
            nquu1=0
            nqud1=0
          endif
        elseif(iclpro.eq.3)then
          if(iabs(idptl(ip)).eq.130)then !proj=Kch
            nquu1=jcp(1,1)+jcp(1,2)
            nqud1=jcp(3,1)+jcp(3,2)
          else                           !proj=K0
            nquu1=jcp(2,1)+jcp(2,2)
            nqud1=jcp(3,1)+jcp(3,2)
          endif
        else                    !charm
          if(iabs(idptl(ip)).eq.140)then
            nquu1=jcp(1,1)+jcp(1,2)
            nqud1=jcp(4,1)+jcp(4,2)
          elseif(iabs(idptl(ip)).eq.240)then
            nquu1=jcp(2,1)+jcp(2,2)
            nqud1=jcp(4,1)+jcp(4,2)
          elseif(iabs(idptl(ip)).eq.340)then
            nquu1=jcp(3,1)+jcp(3,2)
            nqud1=jcp(4,1)+jcp(4,2)
          else
            nquu1=jcp(4,1)
            nqud1=jcp(4,2)
          endif
        endif
        if(iabs(idptl(maproj+it)).eq.1220)then !targ=neutron
          nquu2=jct(2,1)+jct(2,2)
          nqud2=jct(1,1)+jct(1,2)
        else
          nquu2=jct(1,1)+jct(1,2)
          nqud2=jct(2,1)+jct(2,2)
        endif
      endif

      iq1=0
      iq2=0
      wwgg=0.
      wwgq=0.
      wwqg=0.
      wwqq=0.
      wwdd=0.

      !------------------------------------------
      if(iqq.eq.3)then     !     val-val
      !------------------------------------------

       if(ish.ge.4)write(ifch,*)'val-val'
        xmin1=xmin**dble(delh+.4)
        xmax1=xmax**dble(delh+.4)
        zp0=dsqrt(xmin)
        if(zp0.ge.1.d0)call utstop('zp0 in sem&',
     +sizeof('zp0 in sem&'))
      !........ kinematical bounds
        tmax0=dlog((1.d0+dsqrt(1.d0-zp0))/(1.d0-dsqrt(1.d0-zp0)))
        tmin0=dlog((1.d0+dsqrt(1.d0-xpmax0))
     *                   /(1.d0-dsqrt(1.d0-xpmax0)))
        if(iclpro.ne.4)then
          call psjti0(sngl(smax-pth),sqq,sqqb,1,1)
          call psjti0(sngl(smax-pth),sqqp,sqqpb,1,2)
          call psjti0(sngl(smax-pth),sqaq,sqaqb,-1,1)
        else
          call psjti0(sngl(smax-pth),sqqp,sqqpb,4,2)
          sqq=0.
          sqaq=0.
        endif
        if(iremn.ge.2)then
          if(nquu1.gt.nqud1.or.iclpro.ne.2)then
            uv1=psdfh4(sngl(zp0*xp1),q2min,0.,iclpro,1)
            dv1=psdfh4(sngl(zp0*xp1),q2min,0.,iclpro,2)
          else         !if nquu<nqud => no u or no d
            uv1=psdfh4(sngl(zp0*xp1),q2min,0.,1,1)
            dv1=uv1
          endif
          if(nquu1.eq.0)uv1=0d0
          if(nqud1.eq.0)dv1=0d0
          if(nquu2.gt.nqud2)then
            uv2=psdfh4(sngl(zp0*xp2),q2min,0.,icltar,1)
            dv2=psdfh4(sngl(zp0*xp2),q2min,0.,icltar,2)
          else                  !if nquu<nqud => no u or no d
            uv2=psdfh4(sngl(zp0*xp2),q2min,0.,1,1)
            dv2=uv2
          endif
          if(nquu2.eq.0)uv2=0d0
          if(nqud2.eq.0)dv2=0d0
        else
          uv1=psdfh4(sngl(zp0*xp1),q2min,0.,iclpro,1)
          dv1=psdfh4(sngl(zp0*xp1),q2min,0.,iclpro,2)
          uv2=psdfh4(sngl(zp0*xp2),q2min,0.,icltar,1)
          dv2=psdfh4(sngl(zp0*xp2),q2min,0.,icltar,2)
        endif
        wwuu=uv1*uv2*sqq
        if(iclpro.eq.2)then
          wwdd=dv1*dv2*sqq
        elseif(iclpro.eq.1)then
          wwdd=dv1*dv2*sqaq
        elseif(iclpro.eq.3)then
          wwdd=dv1*dv2*sqqp
        elseif(iclpro.eq.4)then
          wwuu=uv1*uv2*sqqp
          wwdd=0.
        endif
        wwud=uv1*dv2*sqqp
        wwdu=dv1*uv2*sqqp
        wudt=wwuu+wwdd+wwud+wwdu
        gb0=dble(wudt)/xmax**dble(delh)/xmin**0.4*
     *    (1.d0-zp0*xp1)**dble(-1.-alpq-alplea(iclpro))*
     *    (1.d0-zp0*xp2)**dble(-1.-alpq-alplea(icltar))*(tmax0-tmin0)
     *    *(1.d0-zp0)**dble(.5+alpq)*(1.d0-zp0)**dble(alpq)  *5.d0

3       zpm=(xmin1+dble(rangen())*(xmax1-xmin1))**dble(1./(delh+.4))   !zpm
        if(iclpro.ne.4)then
          call psjti0(sngl(zpm*s-pth),sqq,sqqb,1,1)
          call psjti0(sngl(zpm*s-pth),sqqp,sqqpb,1,2)
          call psjti0(sngl(zpm*s-pth),sqaq,sqaqb,-1,1)
        else
          call psjti0(sngl(zpm*s-pth),sqqp,sqqpb,4,2)
          sqq=0.
          sqaq=0.
        endif
        xpmax=psutz(s,zpm*s,dble(amqpt**2))  !max x+ for sh pomeron
        tmax=dlog((1.d0+dsqrt(1.d0-dsqrt(zpm)))
     *             /(1.d0-dsqrt(1.d0-dsqrt(zpm))))
        tmin=dlog((1.d0+dsqrt(1.d0-xpmax))/(1.d0-dsqrt(1.d0-xpmax)))

        t=(tmin+dble(rangen())*(tmax-tmin))
        xp=1.d0-((1.d0-dexp(-t))/(1.d0+dexp(-t)))**2  !x+_v
        xm=zpm/xp                             !x-_v
        if(xm.gt.xp.and.ish.ge.1)write(ifmt,*)'xm,xp',xm,xp
        gb=(1.d0-xm)**alpq*(1.d0-xp)**(.5+alpq)*(tmax-tmin)
        if(rangen().lt..5)then
          xp=xm
          xm=zpm/xp
        endif

        if(iremn.ge.2)then
          if(nquu1.gt.nqud1.or.iclpro.ne.2)then
            uv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,1)
            dv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,2)
          else         !if nquu<nqud => no u or no d
            uv1=psdfh4(sngl(xp*xp1),q2min,0.,1,1)
            dv1=uv1
          endif
          if(nquu1.eq.0)uv1=0d0
          if(nqud1.eq.0)dv1=0d0
          if(nquu2.gt.nqud2)then
            uv2=psdfh4(sngl(xm*xp2),q2min,0.,icltar,1)
            dv2=psdfh4(sngl(xm*xp2),q2min,0.,icltar,2)
          else                  !if nquu<nqud => no u or no d
            uv2=psdfh4(sngl(xm*xp2),q2min,0.,1,1)
            dv2=uv2
          endif
          if(nquu2.eq.0)uv2=0d0
          if(nqud2.eq.0)dv2=0d0
        else
          uv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,1)
          dv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,2)
          uv2=psdfh4(sngl(xm*xp2),q2min,0.,icltar,1)
          dv2=psdfh4(sngl(xm*xp2),q2min,0.,icltar,2)
        endif

        wwuu=uv1*uv2*sqq
        if(iclpro.eq.2)then
          wwdd=dv1*dv2*sqq
        elseif(iclpro.eq.1)then
          wwdd=dv1*dv2*sqaq
        elseif(iclpro.eq.3)then
          wwdd=dv1*dv2*sqqp
        elseif(iclpro.eq.4)then
          wwuu=uv1*uv2*sqqp
          wwdd=0.
        endif
        wwud=uv1*dv2*sqqp
        wwdu=dv1*uv2*sqqp
        wudt=wwuu+wwdd+wwud+wwdu
        if(wudt.lt.1d-16)then
          if(ish.ge.1)write(ifmt,*)'No more valence quark for psahot !'
          write(ifch,*)'No more valence quark for psahot !'
     &                     ,ip,it,nquu1,nqud1,nquu2,nqud2
          iret=1
          goto 16
        endif

        gb=gb*dble(wudt)/zpm**dble(delh+0.4)
     *    *(1.d0-xp*xp1)**dble(-1.-alpq-alplea(iclpro))
     *    *(1.d0-xm*xp2)**dble(-1.-alpq-alplea(icltar))/gb0
c          if(ish.ge.4)then
          if(gb.gt.1.d0.and.ish.ge.1)write (ifch,*)
     *      'gb-qq,iclpro,zpm,xp,tmax,tmin,xpmax',
     *      gb,iclpro,zpm,xp,tmax,tmin,xpmax
c          endif
        if(dble(rangen()).gt.gb)goto 3

        aks=rangen()*wudt
        if(aks.le.wwuu)then
          if(iclpro.le.2)then
            iq1=1
          elseif(iclpro.eq.3)then
            if(iabs(idptl(ip)).eq.130)then !proj=Kch
              iq1=1
            else !proj=K0
              iq1=2
            endif
          else   !charm
            if(iabs(idptl(ip)).eq.140)then
              iq1=1
            elseif(iabs(idptl(ip)).eq.240)then
              iq1=2
            elseif(iabs(idptl(ip)).eq.340)then
              iq1=3
            else
              iq1=4
            endif
          endif
          iq2=1
        elseif(aks.le.wwuu+wwdd)then
          if(iclpro.eq.2)then
            iq1=2
          elseif(iclpro.eq.1)then
            iq1=-2
          elseif(iclpro.eq.3)then
            iq1=-3
          else
            iq1=-4
          endif
          iq2=2
        elseif(aks.le.wwuu+wwdd+wwud)then
          if(iclpro.le.2)then
            iq1=1
          elseif(iclpro.eq.3)then
            if(iabs(idptl(ip)).eq.130)then !proj=Kch
              iq1=1
            else !proj=K0
              iq1=2
            endif
          else   !charm
            if(iabs(idptl(ip)).eq.140)then
              iq1=1
            elseif(iabs(idptl(ip)).eq.240)then
              iq1=2
            elseif(iabs(idptl(ip)).eq.340)then
              iq1=3
            else
              iq1=4
            endif
          endif
          iq2=2
        else
          if(iclpro.eq.2)then
            iq1=2
          elseif(iclpro.eq.1)then
            iq1=-2
          elseif(iclpro.eq.3)then
            iq1=-3
          else
            iq1=-4
          endif
          iq2=1
        endif

        wpi=xp*wp0       !lc+ for the semihard interaction
        wmi=xm*wm0       !lc- for the semihard interaction
        wp1=(wp0-wpi)
        wm1=(wm0-wmi)
        wp1=wp1*psutz(wp1*wm1,dble(amqt(2)**2),dble(amqt(4)**2))
        wm1=wm1-amqt(2)**2/wp1

      !-------------------------------------
      else  ! sea-sea  val-sea  sea-val
      !-------------------------------------

       if(ish.ge.4)write(ifch,*)'sea-sea  val-sea  sea-val'
        xmin1=xmin**(delh-dels)
        xmax1=xmax**(delh-dels)

        if(iqq.eq.0)then    !rejection function normalization
          gb0=dlog(xpmax0/xpmin0)*(1.d0-dsqrt(xmin))**(2.*betpom) !y_soft =
        else
          tmax0=acos(dsqrt(xpmin0))  !kinematical limits for t=cos(x+-)**2
          tmin0=acos(dsqrt(xpmax0))
          if(iqq.eq.1)then
            uv1=psdfh4(sngl(xpmin0*xp1),q2min,0.,iclpro,1)
     *        *sngl(1.d0-xpmin0*xp1)**(-1.-alpq-alplea(iclpro))
            dv1=psdfh4(sngl(xpmin0*xp1),q2min,0.,iclpro,2)
     *        *sngl(1.d0-xpmin0*xp1)**(-1.-alpq-alplea(iclpro))
          else
            uv1=psdfh4(sngl(xpmin0*xp2),q2min,0.,icltar,1)
     *        *sngl(1.d0-xpmin0*xp2)**(-1.-alpq-alplea(icltar))
            dv1=psdfh4(sngl(xpmin0*xp2),q2min,0.,icltar,2)
     *        *sngl(1.d0-xpmin0*xp2)**(-1.-alpq-alplea(icltar))
          endif
          gb0=(1.d0-xmin/xpmax0)**betpom*dble(uv1+dv1)
     *      *xpmin0**(-0.5+dels)
     *      *(1.d0-xpmin0)**(0.5+alpq)*(tmax0-tmin0)
          if(ish.ge.6)write (ifch,*)
     *      'gb0,tmax0,tmin0,xpmax0,xpmin0,xmin,xp1,xp2',
     *      gb0,tmax0,tmin0,xpmax0,xpmin0,xmin,xp1,xp2
        endif
        if(iclpro.ne.4.or.iqq.ne.1)then
          call psjti0(sngl(smax-pth),sj,sjb,iqq,0) !inclusive (sj) and born
        else
          call psjti0(sngl(smax-pth),sj,sjb,4,0)
        endif
        gb0=gb0*dble(sj)/xmax**delh      *1.5d0    !rejection function norm.
        if(ish.ge.6)write (ifch,*)'gb0,sj,z',gb0,sj,z

        if(gb0.le.0.)then
          write (ifmt,*)'gb0<0, smax,pth',smax,pth
          iret=1
          goto 16
        endif

        ! sharing of light cone momenta between soft preevolution and
        ! hard interaction: ( first energy-momentum is shared according to
        ! f_hard(yj)~zpm**(delh-dels-1) and then rejected as
        ! w_rej ~sigma_hard_tot(yj) / exp(delh*yj)

 4        continue
        zpm=(xmin1+dble(rangen())*(xmax1-xmin1))**dble(1./(delh-dels)) !zpm
        if(iclpro.ne.4.or.iqq.ne.1)then
          call psjti0(sngl(zpm*s-pth),sgg,sggb,0,0)!inclusive (sgg) and born
          call psjti0(sngl(zpm*s-pth),sgq,sgqb,0,1)
          call psjti0(sngl(zpm*s-pth),sqq,sqqb,1,1)
          call psjti0(sngl(zpm*s-pth),sqaq,sqaqb,-1,1)
          call psjti0(sngl(zpm*s-pth),sqqp,sqqpb,1,2)
          sqq=(sqq+sqaq+2.*(naflav-1)*sqqp)/naflav/2.
        else
          call psjti0(sngl(zpm*s-pth),sgq,sgqb,4,0)
          call psjti0(sngl(zpm*s-pth),sqq,sqqb,4,1)
        endif
        xpmax=psutz(s,zpm*s,dble(amqpt**2))  !max x+ for sh pomeron
        xpmin=zpm/xpmax
        if(ish.ge.8)write (ifch,*)'zpm,xpmax,xpmin',zpm,xpmax,xpmin

        if(iqq.eq.0)then
          xp=xpmin*(xpmax/xpmin)**rangen()  !lc+ share for the hard interaction
          xm=zpm/xp           !lc- share for the hard interaction
        else
          tmax=acos(dsqrt(xpmin))  !kinematical limits for t=cos(x+-)**2
          tmin=acos(dsqrt(xpmax))
          t=tmin+dble(rangen())*(tmax-tmin)
          xp=cos(t)**2
          xm=zpm/xp
        endif

        if(ish.ge.8)write(ifch,*)'zpm,xp,xm,xpmax,xpmin:',
     *      zpm,xp,xm,xpmax,xpmin


        if(iqq.eq.0)then  ! --------- sea-sea -----------

          dwwgg1=0.
          dwwgq1=0.
          dwwqg1=0.
          dwwqq1=0.

          dwwgg2=0.
          dwwgq2=0.
          dwwqg2=0.
          dwwqq2=0.

          wwgg=sngl((1.d0-xp)*(1.d0-xm))**betpom
          wwgq=sngl(1.d0-xp)**betpom*EsoftQZero(sngl(xm))
          wwqg=sngl(1.d0-xm)**betpom*EsoftQZero(sngl(xp))
          wwqq=EsoftQZero(sngl(xp))*EsoftQZero(sngl(xm))

          if(idfpomr.eq.1)then
            rh=r2had(iclpro)+r2had(icltar)-slopom*sngl(dlog(zpm))
            wwgg=(wwgg*z**(rp/rh)/rh+dwwgg1+dwwgg2)
     *        *(r2had(iclpro)+r2had(icltar))/z
            wwgq=(wwgq*z**(rp/rh)/rh+dwwgq1+dwwgq2)
     *        *(r2had(iclpro)+r2had(icltar))/z
            wwqg=(wwqg*z**(rp/rh)/rh+dwwqg1+dwwqg2)
     *        *(r2had(iclpro)+r2had(icltar))/z
            wwqq=(wwqq*z**(rp/rh)/rh+dwwqq1+dwwqq2)
     *        *(r2had(iclpro)+r2had(icltar))/z
          elseif(idfpomr.eq.2)then
            rh=r2had(iclpro)+alam3p/2.-slopom*sngl(dlog(zpm))
            wwgg=wwgg/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
            wwqg=wwqg/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
            wwgq=wwgq/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
            wwqq=wwqq/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
          elseif(idfpomr.eq.3)then
            rh=r2had(icltar)+alam3p/2.-slopom*sngl(dlog(zpm))
            wwgg=wwgg/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
            wwqg=wwqg/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
            wwgq=wwgq/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
            wwqq=wwqq/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
          elseif(idfpomr.eq.4)then
            rh=alam3p-slopom*sngl(dlog(zpm))
            wwgg=wwgg/rh*alam3p*z**(rp/rh-1.)
            wwqg=wwqg/rh*alam3p*z**(rp/rh-1.)
            wwgq=wwgq/rh*alam3p*z**(rp/rh-1.)
            wwqq=wwqq/rh*alam3p*z**(rp/rh-1.)
          else
            stop'psahot-idfpomr????'
          endif

          wwgg=wwgg*sgg*(1.-glusea)**2
          wwgq=wwgq*sgq*(1.-glusea)*glusea
          wwqg=wwqg*sgq*(1.-glusea)*glusea
          wwqq=wwqq*sqq*glusea**2
          gbyj=dlog(xpmax/xpmin)*dble(wwgg+wwgq+wwqg+wwqq)
          wpi=wp0*xp               !lc+ for the hard interaction
          wmi=wm0*xm               !lc+-for the hard interaction
          gbyj=gbyj/zpm**dble(delh)/gb0     !rejection fu
          if(gbyj.ge.1.d0.and.ish.ge.2)write(ifmt,*)'gbyj',gbyj
          if(dble(rangen()).gt.gbyj)goto 4                     !rejection
          wp1=wp0-wpi
          wm1=wm0-wmi
          call pslcsh(wp1,wm1,wp2,wm2,amqt,dble(amqpt))

        else  ! --------- val-sea  sea-val -----------

          dwwg=0.
          dwwq=0.

          wwgq=sngl(1.d0-xm)**betpom
          wwqq=EsoftQZero(sngl(xm))
          if(idfpomr.eq.1)then
            rh=r2had(iclpro)+r2had(icltar)-slopom*sngl(dlog(xm))
            wwgq=(wwgq*z**(rp/rh)/rh+dwwg)
     *        *(r2had(iclpro)+r2had(icltar))/z
            wwqq=(wwqq*z**(rp/rh)/rh+dwwq)
     *        *(r2had(iclpro)+r2had(icltar))/z
          else              !tp071031 not used anymore
            if(iqq.eq.1)then
              rh=r2had(iclpro)+alam3p/2.-slopom*sngl(dlog(xm))
              wwgq=wwgq/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
              wwqq=wwqq/rh*(r2had(iclpro)+alam3p/2.)*z**(rp/rh-1.)
            else
              rh=r2had(icltar)+alam3p/2.-slopom*sngl(dlog(xm))
              wwgq=wwgq/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
              wwqq=wwqq/rh*(r2had(icltar)+alam3p/2.)*z**(rp/rh-1.)
            endif
          endif

          wwgq=wwgq*sgq*(1.-glusea)*sngl(xp)**(-0.5+dels)
     *      *sngl(1.d0-xp)**(0.5+alpq)
          wwqq=wwqq*sqq*glusea*sngl(xp)**(-0.5+dels)
     *      *sngl(1.d0-xp)**(0.5+alpq)

          if(iqq.eq.1)then         !valence quark-gluon hard interaction
            if(iremn.ge.2)then
              if(nquu1.gt.nqud1.or.iclpro.ne.2)then
                uv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,1)
                dv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,2)
              else       !if nquu<nqud => no u or no d
                uv1=psdfh4(sngl(xp*xp1),q2min,0.,1,1)
                dv1=uv1
              endif
              if(nquu1.eq.0)uv1=0d0
              if(nqud1.eq.0)dv1=0d0
            else
              uv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,1)
              dv1=psdfh4(sngl(xp*xp1),q2min,0.,iclpro,2)
            endif
            if(uv1+dv1.lt.1d-16)then
           if(ish.ge.1)write(ifmt,*)'No more valence quark for psahot !'
         write(ifch,*)'No more valence quark in projectile for psahot !'
     &                     ,ip,nquu1,nqud1
              iret=1
              goto 16
            endif
            wpi=wp0*xp             !lc+ for the hard interaction
            wmi=wm0*xm             !lc+-for the hard interaction
            aks=rangen()
            if(aks.le.uv1/(uv1+dv1))then
              if(iclpro.le.2)then
                iq1=1
              elseif(iclpro.eq.3)then
                if(iabs(idptl(ip)).eq.130)then !proj=Kch
                  iq1=1
                else !proj=K0
                  iq1=2
                endif
              else              !charm
                if(iabs(idptl(ip)).eq.140)then
                  iq1=1
                elseif(iabs(idptl(ip)).eq.240)then
                  iq1=2
                elseif(iabs(idptl(ip)).eq.340)then
                  iq1=3
                else
                  iq1=4
                endif
              endif
            else
              if(iclpro.eq.2)then
                iq1=2
              elseif(iclpro.eq.1)then
                iq1=-2
              elseif(iclpro.eq.3)then
                iq1=-3
              else
                iq1=-4
              endif
            endif
            gbyj=dble((wwgq+wwqq)*(uv1+dv1))*(1.d0-xp*xp1)**
     *        (-1.-alpq-alplea(iclpro))
          else                !gluon-valence quark hard interaction
            xm=xp
            xp=zpm/xm
            if(iremn.ge.2)then
              if(nquu2.gt.nqud2)then
                uv1=psdfh4(sngl(xm*xp2),q2min,0.,icltar,1)
                dv1=psdfh4(sngl(xm*xp2),q2min,0.,icltar,2)
              else              !if nquu<nqud => no u or no d
                uv1=psdfh4(sngl(xm*xp2),q2min,0.,1,1)
                dv1=uv1
              endif
              if(nquu2.eq.0)uv1=0d0
              if(nqud2.eq.0)dv1=0d0
            else
              uv1=psdfh4(sngl(xm*xp2),q2min,0.,icltar,1)
              dv1=psdfh4(sngl(xm*xp2),q2min,0.,icltar,2)
            endif
            if(uv1+dv1.lt.1d-16)then
           if(ish.ge.1)write(ifmt,*)'No more valence quark for psahot !'
             write(ifch,*)'No more valence quark in target for psahot !'
     &                     ,it,nquu2,nqud2
              iret=1
              goto 16
            endif
            wpi=wp0*xp             !lc+ for the hard interaction
            wmi=wm0*xm             !lc+-for the hard interaction
            aks=rangen()
            if(aks.le.uv1/(uv1+dv1))then
              iq2=1
            else
              iq2=2
            endif
            gbyj=dble(wwgq+wwqq)*dble(uv1+dv1)*
     *        (1.d0-xm*xp2)**(-1.-alpq-alplea(icltar))
          endif

          gbyj=gbyj*(tmax-tmin)/zpm**delh /gb0 /2.1d0 !rejection
          if(ish.ge.6)write (ifch,*)
     *      'gbyj,zpm,xp,tmax,tmin,xpmax,xpmin',
     *      gbyj,zpm,xp,tmax,tmin,xpmax,xpmin
          if(dble(rangen()).gt.gbyj)goto 4

          wp1=wp0-wpi
          wm1=wm0-wmi
          if(ish.ge.8)write (ifch,*)'q_sea mass check',wp1*wm1,amqpt
          if(iqq.eq.1)then         !valence quark-gluon hard interaction
            amq1=amqt(3)**2
            s24=(amqt(2)+amqt(4))**2
          else
            amq1=amqt(1)**2
            s24=(amqt(2)+amqt(4))**2
            xp=xm
            xm=zpm/xp
          endif
          x1max=psutz(wp1*wm1,dble(amq1),dble(s24))
          x1min=dble(amq1)/x1max/wp1/wm1
          t1min=(1.d0/x1max-1.d0)
          t1max=(1.d0/x1min-1.d0)
5           t1=t1min*(t1max/t1min)**dble(rangen())
          if(ish.ge.8)write (ifch,*)'t1,t1min,t1max',t1,t1min,t1max
          xq1=1.d0/(1.d0+t1)
          if(dble(rangen()).gt.(xq1*(1.d0-xq1))**(1.+(-alpqua)))goto 5
          if(iqq.eq.1)then         !valence quark-gluon hard interacti
            wm2=wm1*(1.d0-xq1)
            wm1=wm1*xq1
            wp1=wp1-dble(amq1)/wm1
            if(ish.ge.8)write (ifch,*)'q_sea+ mass check',
     *        wp1*wm2,s24
            wp1=wp1*psutz(wp1*wm2,dble(amqt(2)**2),dble(amqt(4)**2))
            wm2=wm2-dble(amqt(2))**2/wp1
          else
            wp2=wp1*(1.d0-xq1)
            wp1=wp1*xq1
            wm1=wm1-dble(amq1)/wp1
            if(ish.ge.8)write (ifch,*)'q_sea- mass check',
     *        wp2*wm1,s24
            wm1=wm1*psutz(wm1*wp2,dble(amqt(4)**2),dble(amqt(2)**2))
            wp2=wp2-amqt(4)**2/wm1
          endif

        endif  ! -------------------

      !-------------------------------
      endif
      !-------------------------------

c-------------------------------------------------------------------------
c  flavor and momenta of end partons of the hard Pomeron
c-------------------------------------------------------------------------

       if(ish.ge.4)write(ifch,*)
     &  'flavor and momenta of end partons of the hard Pomeron'

  6   continue
      wpi1=wpi
      wmi1=wmi
      nj=nj0          !initialization for the number of jets
      if(ish.ge.8)write (ifch,*)'5-ww,smin',wpi*wmi,smin

      rrr=rangen()
      jqq=0

      call iddeco(icp,jcp)
      call iddeco(ict,jct)
      if(iremn.ge.2)then
        do j=1,2
          do n=1,nrflav
            jcpr(n,j)=jcpref(n,j,ip)
            jctr(n,j)=jctref(n,j,it)
          enddo
          do n=nrflav+1,nflav
            jcpr(n,j)=0
            jctr(n,j)=0
          enddo
        enddo
      endif

      iret1=0
      iret2=0

      if(iqq.eq.1.or.iqq.eq.2)then
        if(rrr.lt.wwqq/(wwgq+wwqq))jqq=1
      elseif(iqq.eq.0)then
        if(rrr.lt.wwqg/(wwgg+wwqg+wwgq+wwqq))then
          jqq=1
        elseif(rrr.lt.(wwqg+wwgq)/(wwgg+wwqg+wwgq+wwqq))then
          jqq=2
        elseif(rrr.lt.(wwqg+wwgq+wwqq)/(wwgg+wwqg+wwgq+wwqq))then
          jqq=3
        endif
      endif

      if((iqq-1)*(iqq-3).eq.0)then

        iqc(1)=iq1                                !proj=particle
        if(iabs(idptl(ip)).eq.1220)iqc(1)=3-iq1      !proj=neutron
        if(idptl(ip).lt.0)iqc(1)=-iqc(1)               !proj=antiparticle
        nj=nj+1
        ifl1=iabs(iqc(1))

        if(iqc(1).gt.0)then
          if(iremn.ne.0)then
            if(iremn.eq.3)then
              call idsufl3(ifl1,1,jcp) !remove valence quark from remnant
              ifl=idrafl(iclpro,jcpr,2,'v',1,iret3) !take sea antiquark from jcpr
            elseif(iremn.eq.2)then
              call idsufl3(ifl1,1,jcp) !remove valence quark from remnant
              ifl=idrafl(iclpro,jcpr,2,'s',0,iret3) !take sea antiquark freely
              call idsufl3(ifl,2,jcpr) !remove sea antiquark from remnant
            else
              call idsufl(ifl1,1,jcp,iret1) !remove valence quark from remnant
              if(iret1.ne.1)then
                ifl=idrafl(iclpro,jcp,2,'s',1,iret2) !take sea antiquark
              endif
            endif
          else
            ifl=ifl1
          endif
        elseif(iqc(1).lt.0)then
          if(iremn.ne.0)then
            if(iremn.eq.3)then
              call idsufl3(ifl1,2,jcp)
              ifl=idrafl(iclpro,jcpr,1,'v',1,iret3) !take sea quark
            elseif(iremn.eq.2)then
              call idsufl3(ifl1,2,jcp)
              ifl=idrafl(iclpro,jcpr,1,'s',0,iret3) !take sea quark
              call idsufl3(ifl,1,jcpr) !remove sea quark from remnant
            else
              call idsufl(ifl1,2,jcp,iret1)
              if(iret1.ne.1)then
                ifl=idrafl(iclpro,jcp,1,'s',1,iret2)
              endif
            endif
          else
            ifl=ifl1
          endif
        else
          call utstop('No quark for hard Pomeron+ in psahot!&',
     +sizeof('No quark for hard Pomeron+ in psahot!&'))
        endif

        if(iret1.eq.1.or.iret2.eq.1)then
          ifl=ifl1
         if(ish.ge.3)write(ifmt,*)'Not enough space in rem (psahot 1)'
         if(ish.ge.5)write(ifch,*)'Not enough space in rem (psahot 1)'
          call iddeco(icp,jcp)
          iret1=0
          iret2=0
        endif

        if(ish.ge.5)write(ifch,*)'flavor vq+,sqb+',iqc(1)
     &                                            ,-isign(ifl,iqc(1))

        if(ifl.eq.3)then
          iqj(nj)=-isign(ifl,iqc(1))*4/3
        elseif(ifl.eq.4)then
          iqj(nj)=-isign(ifl,iqc(1))/4
        else
          iqj(nj)=-isign(ifl,iqc(1))
        endif
        eqj(1,nj)=.5*sngl(wp1+dble(amqt(2))**2/wp1)
        eqj(2,nj)=wp1-eqj(1,nj)
        eqj(3,nj)=xxp2pr(ncolp,kcol)
        eqj(4,nj)=xyp2pr(ncolp,kcol)
        if(ish.ge.8)write (ifch,*)'q_v+ mass check',(eqj(1,nj)-
     *    eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)
        ncc(1,1)=nj
        ncc(2,1)=0

      else

        nj=nj+1
        if(idfpomr.lt.3.and.iremn.ne.0)then
          if(iremn.eq.3)then
            iflq=idrafl(iclpro,jcpr,1,'v',1,iret3) !take sea quark
            iflqb=idrafl(iclpro,jcpr,2,'v',1,iret3) !take sea antiquark
          elseif(iremn.eq.2)then
            iflq=idrafl(iclpro,jcpr,1,'s',0,iret3) !take sea quark
            call idsufl3(iflq,1,jcpr) !remove sea quark from remnant
            iflqb=idrafl(iclpro,jcpr,2,'s',0,iret3) !take sea antiquark
            call idsufl3(iflqb,2,jcpr) !remove sea antiquark from remnant
          else
            iflq=idrafl(iclpro,jcp,1,'s',1,iret1) !Choose flavor of sea quark.
            if(iret1.ne.1)iflqb=idrafl(iclpro,jcp,2,'s',1,iret2) !Choose antiquark.
          endif
        else
          iflq=idrafl(iclpro,jcp,0,'s',0,iret1)   !Choose flavor of sea quark.
          iflqb=iflq                   !antiquark=quark (vertex end)
        endif

        if(iret1.eq.1.or.iret2.eq.1)then
          iflqb=iflq
         if(ish.ge.3)write(ifmt,*)'Not enough space in rem (psahot 2)'
         if(ish.ge.5)write(ifch,*)'Not enough space in rem (psahot 2)'
          call iddeco(icp,jcp)
          iret1=0
          iret2=0
        endif

        if(ish.ge.5)write(ifch,*)'flavor sq+,sqb+',iflq,-iflqb

        if(iflqb.eq.3)then
          iqj(nj)=-iflqb*4/3
        elseif(iflqb.eq.4)then
          iqj(nj)=-iflqb/4
        else
          iqj(nj)=-iflqb
        endif
        ioj(nj)=7
        if(iflq.eq.3)then
          iqj(nj+1)=iflq*4/3
        elseif(iflq.eq.4)then
          iqj(nj+1)=iflq/4
        else
          iqj(nj+1)=iflq
        endif
        ioj(nj+1)=7

        eqj(1,nj)=.5*sngl(wp1+dble(amqt(1))**2/wp1)
        eqj(2,nj)=wp1-eqj(1,nj)
        eqj(3,nj)=xxp1pr(ncolp,kcol)
        eqj(4,nj)=xyp1pr(ncolp,kcol)
        if(ish.ge.8)write (ifch,*)'q_s1+ mass check',(eqj(1,nj)-
     *    eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)
        eqj(1,nj+1)=.5*sngl(wp2+dble(amqt(2))**2/wp2)
        eqj(2,nj+1)=wp2-eqj(1,nj+1)
        eqj(3,nj+1)=xxp2pr(ncolp,kcol)
        eqj(4,nj+1)=xyp2pr(ncolp,kcol)
        nj=nj+1
        if(ish.ge.8)write (ifch,*)'q_s2+ mass check',(eqj(1,nj)-
     *    eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)

c gluon initiate space like cascade
        if(jqq.eq.0.or.iqq.eq.0.and.jqq.eq.2)then
          iqc(1)=0
          ncc(1,1)=nj-1
          ncc(2,1)=nj
c quark initiate space like cascade
        else
c choose the first parton for the space like cascade (projectile)
          iqc(1)=int(3*rangen()+1.)*(2.*int(.5+rangen())-1.)
          if(1.-1./(wp0*(wmi1-smin/wpi1)).le.xp)goto 1
7         zg=1d0-dble(rangen())*(1.d0-xp)
          if(ish.ge.8)write (ifch,*)'6-zg,xp',zg,xp
          if(dble(rangen()).gt.zg**dels*((1.d0-xp/zg)/(1.d0-xp))
     *                                              **betpom)goto 7
          xg=xp/zg
          wpq=wp0*(xg-xp)
          wmq=1.d0/wpq
          wmi1=wmi1-wmq
          if(wmi1*wpi1.le.smin)goto 1


c add the corresponding anti-parton in the list to compensate the emitted one
          nj=nj+1
          if(iabs(iqc(1)).eq.3)then
            iqj(nj)=-iqc(1)*4/3
          else
            iqj(nj)=-iqc(1)
          endif
          ioj(nj)=-7
          eqj(1,nj)=.5*wmq
          eqj(2,nj)=-eqj(1,nj)
          eqj(3,nj)=0.
          eqj(4,nj)=0.
          if(ish.ge.8)write (ifch,*)'q_s3+ mass check',eqj(1,nj)**2-
     *      eqj(2,nj)**2-eqj(3,nj)**2-eqj(4,nj)**2
          if(iqc(1).gt.0)then
            ncj(1,nj)=nj-1
            ncj(1,nj-1)=nj
            ncj(2,nj)=0
            ncj(2,nj-1)=0
            ncc(1,1)=nj-2
            ncc(2,1)=0
          else
            ncj(1,nj)=nj-2
            ncj(1,nj-2)=nj
            ncj(2,nj)=0
            ncj(2,nj-2)=0
            ncc(1,1)=nj-1
            ncc(2,1)=0
          endif
        endif
      endif

      if((iqq-2)*(iqq-3).eq.0)then
        iqc(2)=iq2             !tar=particle (can not be pion or kaon !)
        if(iabs(idptl(maproj+it)).eq.1220)iqc(2)=3-iq2     !targ=neutron
        if(idptl(maproj+it).lt.0)iqc(2)=-iqc(2)  !targ=antiparticle
        ifl2=iabs(iqc(2))

        nj=nj+1
        if(iqc(2).gt.0)then
          if(iremn.ne.0)then
            if(iremn.eq.3)then
              call idsufl3(ifl2,1,jct) !remove valence quark from remnant
              ifl=idrafl(icltar,jctr,2,'v',1,iret3) !take sea antiquark from jctr
            elseif(iremn.eq.2)then
              call idsufl3(ifl2,1,jct) !remove valence quark from remnant
              ifl=idrafl(icltar,jctr,2,'s',0,iret3) !take sea antiquark freely
              call idsufl3(ifl,2,jctr) !remove sea antiquark from remnant
            else
              call idsufl(ifl2,1,jct,iret1) !remove valence quark from remnant
              if(iret1.ne.1)then
                ifl=idrafl(icltar,jct,2,'s',1,iret2) !take sea antiquark
              endif
            endif
          else
            ifl=ifl2
          endif
        elseif(iqc(2).lt.0)then
          if(iremn.ne.0)then
            if(iremn.eq.3)then
              call idsufl3(ifl2,2,jct)
              ifl=idrafl(icltar,jctr,1,'v',1,iret3) !take sea quark
            elseif(iremn.eq.2)then
              call idsufl3(ifl2,2,jct)
              ifl=idrafl(icltar,jctr,1,'s',0,iret3) !take sea quark
              call idsufl3(ifl,1,jctr) !remove sea quark from remnant
            else
              call idsufl(ifl2,2,jct,iret1)
              if(iret1.ne.1)then
                ifl=idrafl(icltar,jct,1,'s',1,iret2)
              endif
            endif
          else
            ifl=ifl2
          endif
        else
          call utstop('No quark for hard Pomeron- in psahot!&',
     +sizeof('No quark for hard Pomeron- in psahot!&'))
        endif

        if(iret1.eq.1.or.iret2.eq.1)then
          ifl=ifl2
         if(ish.ge.3)write(ifmt,*)'Not enough space in rem (psahot 3)'
         if(ish.ge.5)write(ifch,*)'Not enough space in rem (psahot 3)'
          call iddeco(ict,jct)
          iret1=0
          iret2=0
        endif

        if(ish.ge.5)write(ifch,*)'flavor vq-,sqb-',iqc(2)
     &                                            ,-isign(ifl,iqc(2))

        if(ifl.eq.3)then
          iqj(nj)=-isign(ifl,iqc(2))*4/3
        elseif(ifl.eq.4)then
          iqj(nj)=-isign(ifl,iqc(2))/4
        else
          iqj(nj)=-isign(ifl,iqc(2))
        endif

        eqj(1,nj)=.5*sngl(wm1+dble(amqt(4))**2/wm1)
        eqj(2,nj)=eqj(1,nj)-sngl(wm1)
        eqj(3,nj)=xxm1pr(ncolp,kcol)
        eqj(4,nj)=xym1pr(ncolp,kcol)
        if(ish.ge.8)write (ifch,*)'q_v- mass check',(eqj(1,nj)
     *    +eqj(2,nj))*(eqj(1,nj)-eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)
        ncc(1,2)=nj
        ncc(2,2)=0
      else
        nj=nj+1
        if(mod(idfpomr,2).ne.0.and.iremn.ne.0)then
          if(iremn.eq.3)then
            iflq=idrafl(icltar,jctr,1,'v',1,iret3) !take sea quark
            iflqb=idrafl(icltar,jctr,2,'v',1,iret3) !take sea antiquark
          elseif(iremn.eq.2)then
            iflq=idrafl(icltar,jctr,1,'s',0,iret3) !take sea quark
            call idsufl3(iflq,1,jctr) !remove sea antiquark from remnant
            iflqb=idrafl(icltar,jctr,2,'s',0,iret3) !take sea antiquark
            call idsufl3(iflqb,2,jctr) !remove sea antiquark from remnant
          else
            iflq=idrafl(icltar,jct,1,'s',1,iret2) !Choose flavor of sea quark.
            if(iret1.ne.1)iflqb=idrafl(icltar,jct,2,'s',1,iret2) !Choose  antiquark.
          endif
        else
          iflq=idrafl(iclpro,jcp,0,'s',0,iret1)   !Choose flavor of sea quark.
          iflqb=iflq                   !antiquark=quark (vertex end)
        endif

        if(iret1.eq.1.or.iret2.eq.1)then
          iflqb=iflq
         if(ish.ge.3)write(ifmt,*)'Not enough space in rem (psahot 4)'
         if(ish.ge.5)write(ifch,*)'Not enough space in rem (psahot 4)'
          call iddeco(ict,jct)
          iret1=0
          iret2=0
        endif

        if(ish.ge.5)write(ifch,*)'flavor sq-,sqb-',iflq,-iflqb

        if(iflqb.eq.3)then
          iqj(nj)=-iflqb*4/3
        elseif(iflqb.eq.4)then
          iqj(nj)=-iflqb/4
        else
          iqj(nj)=-iflqb
        endif
        if(iflq.eq.3)then
          iqj(nj+1)=iflq*4/3
        elseif(iflq.eq.4)then
          iqj(nj+1)=iflq/4
        else
          iqj(nj+1)=iflq
        endif
        ioj(nj+1)=8

        eqj(1,nj)=.5*sngl(wm1+dble(amqt(3))**2/wm1)
        eqj(2,nj)=eqj(1,nj)-sngl(wm1)
        eqj(3,nj)=xxm2pr(ncolp,kcol)
        eqj(4,nj)=xym2pr(ncolp,kcol)
        if(ish.ge.8)write (ifch,*)'q_s1- mass check',(eqj(1,nj)-
     *    eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)
        eqj(1,nj+1)=.5*sngl(wm2+dble(amqt(4))**2/wm2)
        eqj(2,nj+1)=eqj(1,nj+1)-wm2
        eqj(3,nj+1)=xxm1pr(ncolp,kcol)
        eqj(4,nj+1)=xym1pr(ncolp,kcol)
        nj=nj+1
        if(ish.ge.8)write (ifch,*)'q_s2- mass check',(eqj(1,nj)-
     *    eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
        eqj(1,nj)=sqrt(eqj(2,nj)**2+eqj(3,nj)**2+eqj(4,nj)**2)

c gluon initiate space like cascade
        if(jqq.eq.0.or.iqq.eq.0.and.jqq.eq.1)then
          iqc(2)=0
          ncc(1,2)=nj-1
          ncc(2,2)=nj
c quark initiate space like cascade
        else
c choose the first parton for the space like cascade (target)
          iqc(2)=int(3*rangen()+1.)*(2.*int(.5+rangen())-1.)
          if(1.-1./(wm0*(wpi1-smin/wmi1)).le.xm)goto 1
8           zg=1.d0-dble(rangen())*(1.d0-xm)
          if(ish.ge.8)write (ifch,*)'7-zg,xm',zg,xm
          if(rangen().gt.zg**dels*((1.d0-xm/zg)/(1.d0-xm))**betpom)
     *      goto 8
          xg=xm/zg
          wmq=wm0*(xg-xm)
          wpq=1.d0/wmq
          wpi1=wpi1-wpq
          if(wmi1*wpi1.le.smin)goto 1

c add the corresponding anti-parton in the list to compensate the emitted one
          nj=nj+1
          if(iabs(iqc(2)).eq.3)then
            iqj(nj)=-iqc(2)*4/3
          else
            iqj(nj)=-iqc(2)
          endif
          ioj(nj)=-8

          eqj(1,nj)=.5*sngl(wpq)
          eqj(2,nj)=eqj(1,nj)
          eqj(3,nj)=0.
          eqj(4,nj)=0.
          if(ish.ge.8)write (ifch,*)'q_s3- mass check',(eqj(1,nj)-
     *      eqj(2,nj))*(eqj(1,nj)+eqj(2,nj))-eqj(3,nj)**2-eqj(4,nj)**2
          if(iqc(2).gt.0)then
            ncj(1,nj)=nj-1
            ncj(1,nj-1)=nj
            ncj(2,nj)=0
            ncj(2,nj-1)=0
            ncc(1,2)=nj-2
            ncc(2,2)=0
          else
            ncj(1,nj)=nj-2
            ncj(1,nj-2)=nj
            ncj(2,nj)=0
            ncj(2,nj-2)=0
            ncc(1,2)=nj-1
            ncc(2,2)=0
          endif
        endif
      endif
      if(jqq.ne.0)then
        if(iqq.ne.0.or.iqq.eq.0.and.jqq.eq.3)then
          if(iclpro.ne.4.or.iqq.ne.1)then
            call psjti0(sngl(wpi1*wmi1-pth),sqq1,sqqb1,1,1)
            call psjti0(sngl(wpi1*wmi1-pth),sqaq1,sqaqb1,-1,1)
            call psjti0(sngl(wpi1*wmi1-pth),sqqp1,sqqpb1,1,2)
            sqq1=(sqq1+sqaq1+2.*(naflav-1)*sqqp1)/naflav/2.
          else
            call psjti0(sngl(wpi1*wmi1-pth),sqq1,sqqb1,4,1)
          endif
          gbs=sqq1/sqq
        else
          call psjti0(sngl(wpi1*wmi1-pth),sgq1,sgqb1,0,1)
          gbs=sgq1/sgq
        endif
        if(ish.ge.8)write (ifch,*)'gbs',gbs
        if(rangen().gt.gbs)goto 6
      endif


c---------------------------------------------------------------
c        inner partons of the hard Pomeron
c---------------------------------------------------------------

       if(ish.ge.4)write(ifch,*)
     &  'inner partons of the hard Pomeron'
      nj00=nj
      wpi=wpi1
      wmi=wmi1
      si=wpi*wmi-pxh**2-pyh**2     !total energy squared for the hard
      if(ish.ge.7)write (ifch,*)'si,wpi,wmi',si,wpi,wmi

      ept(1)=.5d0*(wpi+wmi)
      ept(2)=.5d0*(wpi-wmi)
      ept(3)=pxh
      ept(4)=pyh
      qmin(1)=q2min          !effective momentum cutoff above current la
      qmin(2)=q2min          !effective momentum cutoff below current la
      qminn=max(qmin(1),qmin(2)) !effective momentum cutoff for the bo
c        si=psnorm(ept)       !4-momentum squared for the hard pomeron
      jfirst=1
      jj=int(1.5+rangen())
      nemis(1)=0
      nemis(2)=0

  9   continue ! <<<<----------- ladder rung ---------------------------

      if(ish.ge.4)write(ifch,*)'ladder rung'
      pt2=ept(3)**2+ept(4)**2
      pt=sqrt(sngl(pt2))
      if(iabs(iqc(1)).ne.4)then
        q2mass=0.
      else
        q2mass=qcmass**2
        si=si-dble(q2mass)
      endif
      s2min=4.*qminn+q2mass         !mass cutoff for born scattering
      wwmin=5.*qminn+q2mass-2.*pt*sqrt(q2ini)
      wwmin=(wwmin+sqrt(wwmin**2+4.*(q2mass+pt2)*(qminn-q2ini)))
     */(1.-q2ini/qminn)/2.
      if(ish.ge.5)write(ifch,*)'qminn,q2mass,pt,wwmin:'
      if(ish.ge.5)write(ifch,*)qminn,q2mass,pt,wwmin

      wpt(1)=ept(1)+ept(2)            !lc+ for the current jet emissi
      wpt(2)=ept(1)-ept(2)            !lc- for the current jet emissi
      
      sjord=0.
      if(iabs(iqc(1)).ne.4)then
        if(jfirst.eq.1)then
          sj=psjti(qmin(jj),qqcut,sngl(si),iqc(jj),iqc(3-jj),0)
          sj2=psjti1(qmin(3-jj),qmin(jj),qqcut
     *              ,sngl(si),iqc(3-jj),iqc(jj),0)
          if(ish.ge.5)write(ifch,*)'si,sj,sj2:',si,sj,sj2
          if(rangen().gt.sj2/sj.and.si.gt.1.1d0*dble(wwmin))goto 112
          jfirst=0
          jj=3-jj
          sj=sj2
          goto 111
        endif
       sj=psjti1(qmin(jj),qmin(3-jj),qqcut,sngl(si),iqc(jj),iqc(3-jj),0)
111     sjb=psbint(qmin(1),qmin(2),qqcut,sngl(si),iqc(1),iqc(2),0) !born parton-parton
      else
        sjord=psjci(qmin(2),sngl(si),iqc(2))
        sj=sjord
      sjb=psbint(qmin(1),qmin(2),qqcut,sngl(si)+q2mass,iqc(1),iqc(2),0)
        if(qmin(2).eq.q2min)then
          wwmins=2.5*q2min*(1.+sqrt(1.+
     *    4.*(pt2+q2mass)/q2min))
       if(si.gt.dble(wwmins))call psjti0(sngl(si)+q2mass,sj,sjb
     *                                                  ,iqc(1),iqc(2))
        endif
      endif
      if(ish.ge.5)write(ifch,*)'si,pt2,wwmin,s2min,sj,sjb,iqc:'
      if(ish.ge.5)write(ifch,*)si,pt2,wwmin,s2min,sj,sjb,iqc

      if(si.lt.1.1d0*dble(wwmin))goto 12 !------>>>>>>>
      if(rangen().lt.sjb/sj)goto 12 !------>>>>>>>

      if(iabs(iqc(1)).eq.4)jj=min(2,int(sjord/sj+rangen())+1)  !?????????

112   continue

      if(iabs(iqc(jj)).ne.4)then

        discr=dble((sngl(si)+2.*pt*sqrt(q2ini)-q2mass)**2
     *  -4.*q2ini*(5.*sngl(si)+q2mass+sngl(pt2)))
        if(discr.lt.0.d0.and.ish.ge.1)write(ifmt,*)'discr,si,pt,wwmin',
     *  discr,si,pt,wwmin
        discr=dsqrt(discr)
        qqmax=(si+2.d0*dble(pt*sqrt(q2ini))-dble(q2mass)+discr)/2.d0
     *  /(5.d0+(dble(q2mass)+pt2)/si)
        qqmin=qqmax-discr/(5.d0+(dble(q2mass)+pt2)/si)
        if(s2min.gt.4.d0*qqmin+dble(q2mass))then
          xmm=.5d0*(si-s2min+2.d0*dble(pt*sqrt(q2ini)))
          discr=xmm**2-si*dble(q2ini)*(1.d0+(dble(q2mass)+pt2)/si)
          if(discr.lt.0..and.ish.ge.1)write(ifmt,*)'discr1,si,pt,wwmin',
     *    discr,si,pt,wwmin
          qqmin=(xmm-dsqrt(discr))/(1.d0+(dble(q2mass)+pt2)/si)
        endif

        xmax=min(1.d0-dble(q2ini)/qqmax,.9999d0)
        if(qqmin.lt.dble(qmin(jj)))then
          qqmin=dble(qmin(jj))
          xmin=max(1.d0-((dble(pt)*dsqrt(qqmin)+dsqrt(pt2*qqmin+
     *    si*(si-s2min-qqmin*(1.d0+(dble(q2mass)+pt2)/si))))/si)**2,
     *    (s2min+qqmin*(1.d0+(dble(q2mass)+pt2)/si)-2.d0*dble(pt)
     *    *dsqrt(qqmin))/si)
          if(xmin.le.0.d0)xmin=s2min/si
        else
          xmin=1.d0-dble(q2ini)/qqmin
        endif

        qm0=qmin(jj)
        xm0=1.0-dble(q2ini/qm0)
        if(xm0.gt..999*xmax.or.xm0.lt.1.001*xmin)then
          xm0=.5d0*(xmax+xmin)
        endif
        s2max=sngl(xm0*si-qm0*(dble(q2mass)+pt2)/si)
     *       +2.*pt*sqrt(q2ini)
        xx=xm0

        if(iabs(iqc(1)).ne.4)then
          if(jfirst.eq.1)then
            sj0=psjti(qm0,qqcut,s2max,0,iqc(3-jj),0)*
     *      psfap(xx,iqc(jj),0)+
     *      psjti(qm0,qqcut,s2max,7,iqc(3-jj),0)
     *      *psfap(xx,iqc(jj),1)
          else
            sj0=psjti1(qm0,qmin(3-jj),qqcut,s2max,0,iqc(3-jj),0)*
     *      psfap(xx,iqc(jj),0)+
     *      psjti1(qm0,qmin(3-jj),qqcut,s2max,7,iqc(3-jj),0)
     *      *psfap(xx,iqc(jj),1)
          endif
        else
          sj0=psjci(qm0,s2max,0)*psfap(xx,iqc(jj),0)+
     *    psjci(qm0,s2max,1)*psfap(xx,iqc(jj),1)
        endif

        gb0=dble(sj0/log(q2ini/qcdlam)*qm0*5.)*psuds(qm0,iqc(jj))
        if(ish.ge.5)write(ifch,*)'gb0,qm0,xm0,s2max:',
     *  gb0,qm0,xm0,s2max
c        if(gb0.le.0.)stop'gb0<=0'
        if(gb0.le.0.d0)then
          write(ifmt,*)'gb0.le.0.  si,pt2:',si,pt2
          iret=1
          goto 16
        endif

        if(xm0.le..5d0)then
          gb0=gb0*xm0**(1.-delh)
        else
          gb0=gb0*(1.d0-xm0)*2.d0**delh
        endif

        xmin2=max(.5d0,xmin)
        xmin1=xmin**delh
        xmax1=min(xmax,.5d0)**delh
        if(xmin.ge..5d0)then
          djl=1.
        elseif(xmax.le..5d0)then
          djl=0.
        else
          djl=1./(1.+((2.*sngl(xmin))**delh-1.)/delh/
     *    log(2.*(1.-sngl(xmax))))
        endif

      else

        xmin=5.d0*dble(q2min)/si
        xmax=min(si/(si+5.0*(pt2+dble(q2mass))),.9999d0)
        qqmax=xmax*si/5.d0
        qqmin=dble(q2min)

        qm0=sngl(qqmin)
        xm0=2.d0/(1.d0+sqrt(1.d0+4.d0*(pt2+dble(q2mass))/qm0))
        if(xm0.gt..999d0*xmax.or.xm0.lt.1.001d0*xmin)then
          xm0=.5d0*(xmax+xmin)
        endif
        s2max=sngl(xm0*si)
        xx=xm0

        sj0=psjti(qm0,qmin(3-jj),s2max,0,iqc(3-jj),0)*
     *  psfap(xx,iqc(jj),0)
        gb0=dble(sj0/log(qm0/qcdlam)*qm0   *5.)
        gb0=gb0*xm0**(1.-delh)
        if(gb0.le.0.d0)then
          if(ish.ge.2)write(ifch,*)'gb0.le.0. (charm)  si,pt2:',si,pt2
          iret=1
          goto 16
        endif
        djl=0.
        xmin2=0d0
        xmin1=xmin**delh
        xmax1=xmax**delh

      endif

      if(ish.ge.5)write(ifch,*)'xmin,xmax,qqmin,qqmax:',
     *xmin,xmax,qqmin,qqmax

      ntry=0
10    continue
      ntry=ntry+1
      if(ntry.gt.5000000)then
        if(ish.ge.1)write(*,*)'Reject hard string (too many rejection)'
     &,kcol,ncolp,nptl,gb7
        iret=1
        goto 16
      endif
      if(rangen().gt.djl)then        !lc momentum share in the cur
        x=(xmin1+dble(rangen())*(xmax1-xmin1))**(1./delh)
      else
        x=1.d0-(1.d0-xmin2)*((1.d0-xmax)/(1.d0-xmin2))**rangen()
      endif
      qq=qqmin/(1.d0+dble(rangen())*(qqmin/qqmax-1.d0))
      if(ish.ge.7)write(ifch,*)'x,qq:',x,qq,ntry

      if(iabs(iqc(jj)).ne.4)then

        qt2=qq*(1.d0-x)
        if(qt2.lt.dble(q2ini))then
          if(ish.ge.7)write(ifch,*)'qt2:',qt2
          goto 10
        endif

        qmin2=max(qminn,sngl(qq))
        qt=dsqrt(qt2)
        call pscs(bcos,bsin)
        ep3(3)=sngl(qt)*bcos   !new parton pt-s
        ep3(4)=sngl(qt)*bsin
        ptnew=(ept(3)-dble(ep3(3)))**2+(ept(4)-dble(ep3(4)))**2
        s2min2=4.*qmin2+q2mass

        s2=x*(si-qq)-ptnew-qq*(dble(q2mass)+pt2)/si+pt2  !new ladder mass
        if(s2.lt.dble(s2min2))then
          if(ish.ge.7)write(ifch,*)'s2,s2min2:',s2,s2min2
          goto 10  !rejection in case of too low mass
        endif

        xx=x
        if(iabs(iqc(1)).ne.4)then
          if(jfirst.eq.1)then
            sj1=psjti(sngl(qq),qqcut,sngl(s2),0,iqc(3-jj),0)
            if(iqc(jj).ne.0)then                             !f1 - f2
              sj2=psjti(sngl(qq),qqcut,sngl(s2),iqc(jj),iqc(3-jj),0)
            elseif(iqc(3-jj).eq.0)then                       !f1 - g
              sj2=psjti(sngl(qq),qqcut,sngl(s2),1,0,0)
            else                                             !g  - f2
              sj2=psjti(sngl(qq),qqcut,sngl(s2),1,1,0)/naflav/2.      !q  - q
     *        +psjti(sngl(qq),qqcut,sngl(s2),-1,1,0)/naflav/2.        !q~ - q
     *        +psjti(sngl(qq),qqcut,sngl(s2),1,2,0)*(1.-1./naflav)    !q' - q
            endif
          else
            sj1=psjti1(sngl(qq),qmin(3-jj),qqcut,sngl(s2),0,iqc(3-jj),0)
            if(iqc(jj).ne.0)then                             !f1 - f2
              sj2=psjti1(sngl(qq),qmin(3-jj),qqcut
     *                   ,sngl(s2),iqc(jj),iqc(3-jj),0)
            elseif(iqc(3-jj).eq.0)then                       !f1 - g
              sj2=psjti1(sngl(qq),qmin(3-jj),qqcut,sngl(s2),1,0,0)
            else                                             !g  - f2
              sj2=psjti1(sngl(qq),qmin(3-jj),qqcut
     *                         ,sngl(s2),1,1,0)/naflav/2.         !q - q
     *           +psjti1(sngl(qq),qmin(3-jj),qqcut
     *                         ,sngl(s2),-1,1,0)/naflav/2.   !q~ - q
     *           +psjti1(sngl(qq),qmin(3-jj),qqcut
     *                      ,sngl(s2),1,2,0)*(1.-1./naflav)!q' - q
            endif
          endif
        else
          sj1=psjci(sngl(qq),sngl(s2),0)
          sj2=psjci(sngl(qq),sngl(s2),1)
        endif

        !...gb7 - rejection function for x and q**2 simulation
        gb7=dble((sj1*psfap(xx,iqc(jj),0)+sj2*psfap(xx,iqc(jj),1))/
     *  log(sngl(qt2)/qcdlam))*psuds(sngl(qq),iqc(jj))*qq/gb0

        if(x.le..5d0)then
          gb7=gb7*x**(1.-delh)
        else
          gb7=gb7*(1.d0-x)*2.d0**delh
        endif
        if(gb7.gt.1..and.ish.ge.2)write(ifmt,*)'gb7,qq,x,qm0,xm0',
     *  gb7,qq,x,qm0,xm0
        if(ish.ge.7)write(ifch,*)'gb7:',gb7
        if(dble(rangen()).gt.gb7)goto 10
      else
        qmin2=max(qminn,sngl(qq))
        s2min2=4.*qmin2
        s2=x*si-qq               !new ladder mass
        if(s2.lt.dble(s2min2))goto 10  !rejection in case of too low mass

        call pscs(bcos,bsin)
        xmm=x*(ept(3)*dble(bcos)+ept(4)*dble(bsin))
        discr=xmm**2+qq*(1.d0-x)-x**2*(pt2+dble(q2mass))
        if(discr.lt.0.d0)goto 10
        qt=xmm+dsqrt(discr)
        ep3(3)=sngl(ept(3)-qt*dble(bcos))   !new parton pt-s
        ep3(4)=sngl(ept(4)-qt*dble(bsin))
        qt2=dble(ep3(3)**2+ep3(4)**2)

        xx=x
        sj1=psjti(sngl(qq),qqcut,sngl(s2),0,iqc(3-jj),0)
        sj2=0.
        !.... gb7 - rejection function for x and q**2 simulation
        gb7=dble(sj1*psfap(xx,iqc(jj),0)/
     *  log(sngl(qq)/qcdlam))*qq/gb0
        gb7=gb7*x**(1.-delh)
        if(gb7.gt.1..and.ish.ge.2)write(ifmt,*)'gb7,qq,x,qm0,xm0',
     *  gb7,qq,x,qm0,xm0
        if(ish.ge.7)write(ifch,*)'gb7:',gb7
        if(dble(rangen()).gt.gb7)goto 10
      endif

! parton type selection iqc -> iqnew (space like) + iq1 (time like)
      nqc(2)=0
      iqnew=iqc(jj)
      if(rangen().lt.sj1/(sj1+sj2))then
        if(iqc(jj).eq.0)then    !g -> g + g  (jt=1)
          jt=1
          jq=int(1.5+rangen())
          nqc(1)=ncc(jq,jj)
        else                    !q -> g + q  (jt=2)
          jt=2
          if(iqc(jj).gt.0)then
            jq=1
          else
            jq=2
          endif
          nqc(1)=0
          iqnew=0
        endif
        iq1=iqc(jj)
        jo=ncc(jq,jj)           !parton origin
        if(jo.ne.0)then
          jo=ioj(jo)
        else
          jo=ioj(ncc(3-jq,jj))
        endif
      else
        if(iqc(jj).ne.0)then    !q -> q + g  (jt=3)
          iq1=0
          jt=3
          if(iqc(jj).gt.0)then
            jq=1
          else
            jq=2
          endif
          nqc(1)=ncc(1,jj)
        else                    !g -> q + q  (jt=4)
          jt=4
          jq=int(1.5+rangen())
          iq1=int(naflav*rangen()+1.)*(3-2*jq)
          iqnew=-iq1
          nqc(1)=ncc(jq,jj)
        endif
        jo=ncc(1,jj)             !parton origin
        if(jo.ne.0)then
          jo=ioj(jo)
        else
          jo=ioj(ncc(2,jj))
        endif
      endif
      if(jo.ge.70.and.jo.ne.99)jo=jo/10
      if(ish.ge.5)write(ifch,'(a,i3,a4,i3,a4,i3)')' Process :'
     *                       ,iqc(jj),' -> ',iqnew,'  + ',iq1

      if(iabs(iqc(jj)).ne.4)then
        q2part=0.
c        write(*,*)'sem:',wpt(jj),pt2,q2mass,wpt(3-jj),jj
        if(dabs(wpt(3-jj)).gt.1.d-20)then
          wplc=wpt(jj)-(pt2+dble(q2mass))/wpt(3-jj)
        else
          if(ish.gt.1)write(ifmt,*)'Problem with wpt in sem',wpt(3-jj)
          wplc=wpt(jj)-(pt2+dble(q2mass))
        endif
        qp2max=max(0.,sngl(qt2))
      else
        q2part=q2mass
        wplc=wpt(jj)
        qp2max=max(0.,sngl(qq))
      endif
      eprt=max(dsqrt(qt2+dble(q2part)),.5d0*((1.d0-x)*wplc+
     *(qt2+dble(q2part))/(1.d0-x)/wplc))
      pl=((1.d0-x)*wplc-eprt)*dble(3-2*jj)
      zeta=sqrt(qp2max/si)/sqrt(x*(1.-x))
      if(iq1.eq.0)then
        iq2ini=9
        jo=iq2ini
        if(zeta.gt.zetacut)jo=-jo
      else
        iq2ini=iq1
        jo=iq2ini
      endif
      if(ish.ge.5)write(ifch,*)'qq,eprt,iq2ini,jo,E2-Q2',qp2max,eprt
     *,iq2ini,jo,eprt**2-qt2
      ntest=0
11    ntest=ntest+1
      call timsh1(qp2max,sngl(eprt),iq2ini,jo)
      amprt=pprt(5,1)**2
      plprt=max(0.d0,eprt**2-qt2)-dble(amprt)
      if(plprt.lt.0.d0)then
        if(ntest.lt.10000)then
          goto 11
        else
          iret=1
          goto 16
        endif
      endif
      ep3(1)=sngl(eprt)
      ep3(2)=sngl(dsqrt(plprt))
      if(pl.lt.0.d0)ep3(2)=-ep3(2)
      ey(1)=1.
      ey(2)=1.
      ey(3)=1.
      do i=1,4
        ept1(i)=ept(i)-dble(ep3(i))
      enddo
      call psdefrot(ep3,s0xh,c0xh,s0h,c0h)
      s2new=psnorm(ept1)

      if(iabs(iqc(jj)).ne.4.and.s2new-q2mass.gt.s2min2.or.
     *iabs(iqc(jj)).eq.4.and.s2new.gt.s2min2)then
        if(iabs(iqc(1)).ne.4.or.jj.eq.1)then
          if(jfirst.eq.1)then
            gb=dble(psjti(sngl(qq),qqcut,s2new,iqnew,iqc(3-jj),0))
          else
            gb=dble(psjti1(sngl(qq),qmin(3-jj),qqcut,s2new,iqnew,
     *                                               iqc(3-jj),0))
          endif
        else
          gb=dble(psjci(sngl(qq),s2new-q2mass,iqnew))
        endif
        if(iqnew.eq.0)then
          gb=gb/dble(sj1)
        else
          gb=gb/dble(sj2)
        endif
        if(dble(rangen()).gt.gb)goto 10
      else
        goto 10
      endif

      if(ish.ge.6)write(ifch,*)'jt,jj,jq,nqc:',jt,jj,jq,nqc
      nprtjx=0
      call psreti(nqc,jq,1,ey,s0xh,c0xh,s0h,c0h)

      if(jt.eq.1)then
        ncc(jq,jj)=nqc(2)
      elseif(jt.eq.2)then
        ncc(jq,jj)=ncc(1,jj)
        ncc(3-jq,jj)=nqc(1)
      elseif(jt.eq.3)then
        ncc(1,jj)=nqc(2)
      elseif(jt.eq.4)then
        ncc(1,jj)=ncc(3-jq,jj)
      endif
      iqc(jj)=iqnew

      do i=1,4
        ept(i)=ept1(i)
      enddo
      ! c.m. energy squared, minimal  4-momentum transfer square and gluon 4-v
      ! for the next ladder run
      qmin(jj)=sngl(qq)
      qminn=qmin2
      si=dble(s2new)
      nemis(jj)=nemis(jj)+1

      goto 9  !  ---------------next ladder rung ------------>>>>>>>>>>>


 12   continue !------------------- Born process------------------------

      if(ish.ge.4)write(ifch,*)'Born process'
      if(ish.ge.6)write(ifch,*)'iqc,si:',iqc,si
      qq=dble(qminn)

   !kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
      xpprbor(ncolp,kcol)=(ept(1)+ept(2))/dble(engy)
      xmprbor(ncolp,kcol)=(ept(1)-ept(2))/dble(engy)
      nemispr(1,ncolp,kcol)=nemis(1)
      nemispr(2,ncolp,kcol)=nemis(2)
   !   write(*,'(a,2f8.3,i3,3x,2i3)')'------------'
   !  *          ,xpprbor(ncolp,kcol),xmprbor(ncolp,kcol),nj-nj00,nemis
   !kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk

      call psabor(sngl(si),sngl(qq),iqc,ncc,ept,0,iptl,bx)

   !kkkkkkkkkkkkk out Born partons without timelike cascade
      if(nprtjx.eq.2)then
       do ii=1,2
        ptprboo(ii,ncolp,kcol)=sqrt(pprtx(1,ii)**2+pprtx(2,ii)**2)
       enddo
      else
        stop'psahot: should not happen!!!!                '
      endif
   !kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
   !     ptxxx=max(  ptprboo(1,ncolp,kcol) , ptprboo(2,ncolp,kcol)  )
   !            print*,sqrt(wm0*wp0),sqrt(wmi*wpi),ptxxx  !++++++++++++++++++++++++++
   !      if(ptxxx.lt.10)goto1
   !            print*,'  ++++++++++++++++++++++++++++++++++++ '
   !kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
      if(nj.ne.0.)then
        do i=1,nj
          do l=1,6
            bxj(l,i)=bx(l)
          enddo
          ityj(i)=ity
          iorj(i)=iptl
        enddo
      endif

      call psjarr(jfl)       !kinky strings formation

      if(jfl.eq.0.and.ish.ge.4)write(ifch,*)
     *'jfl,nj,nptl',jfl,nj,nptl
      if(jfl.eq.0)goto 1

c --- update remnant flavour ---

      iret1=0
      call idenco(jcp,icp,iret1)
      if(iret1.eq.1)call utstop('Problem with proj rem in psahot !&',
     +sizeof('Problem with proj rem in psahot !&'))
      iret2=0
      call idenco(jct,ict,iret2)
      if(iret2.eq.1)call utstop('Problem with targ rem in psahot !&',
     +sizeof('Problem with targ rem in psahot !&'))

      do i=1,2
        icproj(i,ip)=icp(i)
        ictarg(i,it)=ict(i)
      enddo

      if(ish.ge.3)write(ifch,*)'End psahot (icp,ict):',icp,ict

      if(iremn.ge.2)then        !uses precalculated flavors

        do j=1,2
          do n=1,nrflav
            jcpref(n,j,ip)=jcpr(n,j)
            jctref(n,j,it)=jctr(n,j)
          enddo
        enddo
        if(ish.ge.3)then
          write(ifch,'(a,6i3,2x,6i3)')' proj:  ',jcpr
          write(ifch,'(a,6i3,2x,6i3)')' targ:  ',jctr
        endif

      endif


c ------------------------------

16    continue
      call utprix('psahot',ish,ishini,3)

      q2fin=q2finsave
      return
      end

c------------------------------------------------------------------------
      subroutine psjarr(jfl)
c-----------------------------------------------------------------------
c
c   final jets rearrangement according to their colour connection
c   and write to /cptl/
c
c jfl - flag for the rejection (in case of jfl=0)
c-----------------------------------------------------------------------
c Input:

      parameter (mjstr=20000)
      common /psar29/ eqj(4,mjstr),iqj(mjstr),ncj(2,mjstr),ioj(mjstr),nj

c eqj(1:4,k) - 4-momentum (qgs) for k-th parton;
c bxj(1:4,k) - coordinates for k-th parton formation point;
c iqj(k) - ID (qgs) for k-th parton;
c ncj(1:2,k) - colour connected partons indexes for k-th parton;
c nj - number of partons
c-----------------------------------------------------------------------
      dimension mark(mjstr)
      double precision  ept(4),eptot(2)
      include 'epos.inc'
      include 'epos.incsem'

      if(nj.eq.0)then
        jfl=1
        return
      endif
csp : TEST if string are full  
     
      do k=1,nj      
        if(iqj(k).eq.0)then   !gluon must have two neighbours        
          if(ncj(1,k).eq.0)then !first neigbour missing
csp          write(*,*)'correction'
            do kk=1,nj          !look to which parton he is connected
              if(ncj(1,kk).eq.k)then
                if(ncj(2,k).ne.kk)ncj(1,k)=kk !if not already connected : connection
              elseif(ncj(2,kk).eq.k)then
                if(ncj(1,k).ne.kk)ncj(1,k)=kk
                endif
            enddo  
           endif
           if(ncj(2,k).eq.0)then !second neigbour missing
csp           write(*,*)'correction'
            do kk=1,nj
              if(ncj(2,kk).eq.k)then
                if(ncj(1,k).ne.kk)ncj(2,k)=kk
                elseif(ncj(1,kk).eq.k)then
                if(ncj(2,k).ne.kk)ncj(2,k)=kk
                endif
            enddo  
           endif          
        endif
      enddo
      
      
csp END OF TEST
c      do k=1,nj  !???????????????
c       eqj(1,k)=dsqrt(0d0+eqj(2,k)**2+eqj(3,k)**2+eqj(4,k)**2)
c      enddo
      if(ish.ge.3)then
        write (ifch,*)'psjarr: nj',nj
        do k=1,nj
          if(iqj(k).ne.0)ncj(2,k)=0
          write(ifch,'(a,i4)')' parton',k
          write(ifch,'(i6,2x,4e10.3,2x,2i3)')iqj(k)
     *    ,(eqj(j,k),j=1,4),(ncj(j,k),j=1,2)
        enddo
      endif

      jfl=0
      do i=1,nj
        mark(i)=1
      enddo
      nptl0=nptl
c total energy of hte two half of the Pomeron
      eptot(1)=0d0
      eptot(2)=0d0

1     continue
      do ij=1,nj
        if(mark(ij).ne.0.and.iqj(ij).ne.0)goto 2
      enddo
2     continue
      jfirst=1
c to calculate the total energy of the 2 big strings produce by a pomeron
c we first fix to which string (1 or 2) the sub-string belong to
      ij0=ncj(1,ij)
      kkk=1
      if(ij0.gt.0)then
        if(ncj(1,ij0).eq.ij)kkk=1
        if(ncj(2,ij0).eq.ij)kkk=2
      endif
      if(iabs(iqj(ij)).le.2)then
        am1=amhadr(1)
      elseif(iabs(iqj(ij)).eq.4)then
        am1=amhadr(3)
      elseif(iabs(iqj(ij)).eq.40)then
        am1=qcmass
      else
        am1=amhadr(2)
      endif

      do i=1,4
        ept(i)=0.
      enddo

3     continue
      call pspawr(ij,kkk)

      mark(ij)=0

      do i=1,4
        ept(i)=ept(i)+eqj(i,ij)
      enddo
      eptot(kkk)=eptot(kkk)+eqj(1,ij)

      if(iqj(ij).ne.0)then
        if(jfirst.ne.1)then
          if(iabs(iqj(ij)).le.2)then
            am2=amhadr(1)
          elseif(iabs(iqj(ij)).eq.4)then
            am2=amhadr(3)
          elseif(iabs(iqj(ij)).eq.40)then
            am2=qcmass
          else
            am2=amhadr(2)
          endif
          amj=(am1+am2+stmass)**2
          sm=psnorm(ept)
          if(sm.lt.amj)then
            nptl=nptl0
            goto 999
          endif

          if(nptl-nptl0.lt.nj)then
            goto 1
          else
            if(iLHC.ne.1)then
c at the end of the process, save eptot(kkk) is qsqptl of each particle
              do k=nptl0+1,nptl
                qsqptl(k)=sngl(eptot(nint(qsqptl(k)))**2)
              enddo
            endif
            
            if(ish.ge.3)then
              write (ifch,*)'psjarr: nptl',nptl
              do k=nptl0+1,nptl
                write(ifch,'(a,i4)')' particle',k
                write(ifch,'(i5,2x,6e10.3)')idptl(k)
     *          ,(pptl(j,k),j=1,5),sqrt(qsqptl(k))
              enddo
            endif
            jfl=1
            goto 999
          endif
        else
          jfirst=0
          njpar=ij
          ij=ncj(1,ij)
          if(ij.eq.0)write(ifch,*)
     &'IN PSJARR ij=0 :parton,',njpar,'with no connection'
          goto 3
        endif

      else
        if(ncj(1,ij).eq.njpar)then
          njdau=ncj(2,ij)
        else
          njdau=ncj(1,ij)
        endif
        njpar=ij
        ij=njdau
        if(ij.eq.0)write(ifch,*)
     &'IN PSJARR ij=0 :parton,',njpar, 'with no connection'
        goto 3
      endif
      
      
  999 continue

      end

c------------------------------------------------------------------------
      subroutine pspawr(kj,kkk)
c-----------------------------------------------------------------------
c pspawr - writing final parton kj into particle list
c------------------------------------------------------------------------
c Input:
      parameter (mjstr=20000)
      common /psar29/ eqj(4,mjstr),iqj(mjstr),ncj(2,mjstr),ioj(mjstr),nj
      common /psar30/ iorj(mjstr),ityj(mjstr),bxj(6,mjstr),q2j(mjstr)

c eqj(1:4,k) - 4-momentum (qgs) for k-th parton;
c bxj(1:4,k) - coordinates for k-th parton formation point;
c iqj(k) - ID (qgs) for k-th parton;
c ncj(1:2,k) - colour connected partons indexes for k-th parton;
c nj - number of partons
c kkk is the indice of the original half of the pomeron
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      common/ciptl/iptl

      nptl=nptl+1
      if(ish.ge.9)write (ifch,*)'nptl,kj (sto)',nptl,kj
      if(nptl.ge.mxptl.or.kj.le.0)then
       write (ifmt,*)'nptl,kj',nptl,kj
       call alist('Error in pspawr: nptl or kj out of bounds &',
     +sizeof('Error in pspawr: nptl or kj out of bounds &'),1,nptl)
       call utstop('nptl or kj out of bounds&',
     +sizeof('nptl or kj out of bounds&'))
      endif

      if(ifrptl(1,iptl).eq.0)ifrptl(1,iptl)=nptl
      ifrptl(2,iptl)=nptl

      pptl(1,nptl)=eqj(3,kj)
      pptl(2,nptl)=eqj(4,kj)
      pptl(3,nptl)=eqj(2,kj)
      pptl(4,nptl)=eqj(1,kj)
      pptl(5,nptl)=0.
      idptl(nptl)=psidd(iqj(kj))
      iorptl(nptl)=iorj(kj)
      jorptl(nptl)=ioj(kj)
      istptl(nptl)=20
      do i=1,4
        xorptl(i,nptl)=bxj(i,kj)
      enddo
      tivptl(1,nptl)=bxj(5,kj)
      tivptl(2,nptl)=bxj(6,kj)
      ityptl(nptl)=ityj(kj)
c register to which big string the particle belongs to
      if(iLHC.eq.1)then
        qsqptl(nptl)=q2j(kj)
      else
        qsqptl(nptl)=float(kkk)
      endif
          !kkkkkkkkkkkkkkkkkkkkkkk
       !    write(*,'(a,2i4,i6,f8.3)')'.... ',kj,nptl,idptl(nptl)
       !*     ,sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2)
      return
      end

c------------------------------------------------------------------------
      subroutine psabor(si,qq,iqc,ncc,ept,jdis,iptl,coordo)
c------------------------------------------------------------------------
c psabor - highest virtuality subprocess in the ladder
c si - c.m. energy squared for the process,
c qq - p_t-cutoff for the process due to the evolution,
c iqc(i), i=1,2 - incoming parton types(0-g,(+-)1-u(u~),(+-)2-d(d~)etc.),
c ncc(i,j), i,j=1,2 - incoming partons color connections,
c ept(4) - total 4-momentum for the system of the 2 partons
c jdis=0 - hadronic process in pp; 1 - resolved photon process
c------------------------------------------------------------------------
      double precision ept(4),psutz,psuds
      dimension ep3(4),ey(3),wsub(5),iqc(2),ncc(2,2),nqc(2),coordo(6)
      parameter (mjstr=20000)
      common /psar29/eqj(4,mjstr),iqj(mjstr),ncj(2,mjstr),ioj(mjstr),nj
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      common/cprtx/nprtjx,pprtx(5,2)
      include 'epos.incsem'
      include 'epos.inc'
      call utpri('psabor',ish,ishini,5)

      if(iabs(iqc(1)).ne.4)then   !gluon or light quark
        q2mass=0.
      else                        !c-quark
        q2mass=qcmass**2
      endif
      p1=si/(1.+q2mass/si)
      if(p1.gt.4.*qq)then                 !|t|-cutoff (p^2_t>qq)
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
      else
        tmin=2.*qq
      endif
      tmax=.5*p1

      fborn=0.
      qt2=tmin*(1.d0-tmin/p1)
      if(qt2.lt..999d0*qq.and.ish.ge.2)write(ifmt,*)'qt20,qq',qt2,qq
      if(iqc(1).ne.0.or.iqc(2).ne.0)then
        do l=1,5       !sum over different subprocesses
          wsub(l)=psbori(si,tmin,iqc(1),iqc(2),l)  !matrix element
        if(l.le.3)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)**2
        elseif(l.le.4)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)*alfe/2/pi
        else
          wsub(l)=wsub(l)*(alfe/2/pi)**2
        endif
          fborn=fborn+wsub(l)
        enddo
        fborn=tmin**2*fborn
      else
        do l=1,5
          wsub(l)=psbori(si,.5*si,iqc(1),iqc(2),l)
        if(l.le.3)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)**2
        elseif(l.le.4)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)*alfe/2/pi
        else
          wsub(l)=wsub(l)*(alfe/2/pi)**2
        endif
          fborn=fborn+wsub(l)
        enddo
        fborn=.25*si**2*fborn
      endif
       if(jdis.eq.0)then
         scale1=qt2
       else
         scale1=4.*qt2
       endif
      gb0=dble(fborn)*psuds(scale1,iqc(1))*psuds(qt2,iqc(2))
      if(ish.ge.7)write(ifch,*)'tmin,gb0:',tmin,gb0

c------------------------------------------------
c 4-momentum transfer squared is simulated first as dq_t**2/q_t**4 from
c tmin to s/2
14    q2=tmin/(1.-rangen()*(1.-tmin/tmax))    !q2=min(|t|,|u|)
      qt2=q2*(1.-q2/p1)                       !qt2=p_t^2 for the process
      if(qt2.lt.qq.and.ish.ge.2)write(ifmt,*)'qt2,qq',qt2,qq
      if(rangen().lt..5)then  !|u|=q2, |t|=p1-q2
        jm=2                  !first parton to be considered
        tq=p1-q2
      else                    !|t|=q2, |u|=p1-q2
        jm=1                  !first parton to be considered
        tq=q2
      endif

      fborn=0.
      do l=1,5                       !sum over different subprocesses
        wsub(l)=psbori(si,tq,iqc(1),iqc(2),l)
        if(l.le.3)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)**2
        elseif(l.le.4)then
          wsub(l)=wsub(l)*pssalf(qt2/qcdlam)*alfe/2/pi
        else
          wsub(l)=wsub(l)*(alfe/2/pi)**2
        endif
        fborn=fborn+wsub(l)
      enddo
       if(jdis.eq.0)then
         scale1=qt2
       else
         scale1=4.*qt2
       endif
      gb=dble(q2**2*fborn)
     &*psuds(scale1,iqc(1))*psuds(qt2,iqc(2))/gb0 !rejection function
      if(ish.ge.7)write(ifch,*)'q2,qt2,gb:',q2,qt2,gb

      if(dble(rangen()).gt.gb)goto 14                   !rejection

c determination of the color configuration
      nqc(2)=0
      if(iqc(1).eq.0.and.iqc(2).eq.0)then      !g+g
        jq=int(1.5+rangen())    !jq=1(2) - transfer of color (anticolor)
        nqc(1)=ncc(jq,jm)

        if(rangen().lt.wsub(1)/fborn)then      !gg->gg
          if(rangen().lt..5)then
            jt=1                !anticolor-color annihilation
            nqc(2)=0
            njc1=ncc(3-jq,jm)
            njc2=ncc(jq,3-jm)
            if(iqj(njc1).ne.0)then
              ncj(1,njc1)=njc2
            else
              ncj(jq,njc1)=njc2
            endif
            if(iqj(njc2).ne.0)then
              ncj(1,njc2)=njc1
            else
              ncj(3-jq,njc2)=njc1
            endif
          else
            jt=2                    !produced gluons get color and
            nqc(2)=ncc(3-jq,3-jm)   !anticolor from the 2 parents
          endif
        else                                   !gg->qq~
          jt=9                  !anticolor-color annihilation
          iqc(jm)=int(naflav*rangen()+1)*(3-2*jq) !(anti)quark flavor
          iqc(3-jm)=-iqc(jm)
          njc1=ncc(3-jq,jm)
          njc2=ncc(jq,3-jm)
          if(iqj(njc1).ne.0)then
            ncj(1,njc1)=njc2
          else
            ncj(jq,njc1)=njc2
          endif
          if(iqj(njc2).ne.0)then
            ncj(1,njc2)=njc1
          else
            ncj(3-jq,njc2)=njc1
          endif
        endif

      elseif(iqc(1)*iqc(2).eq.0)then       !q(q~)+g
        if(iqc(1)+iqc(2).gt.0)then
          jq=1                             !q
        else
          jq=2                             !q~
        endif
       if(rangen().lt.wsub(1)/fborn)then  !q(q~)g->q(q~)g
        if(rangen().lt..5)then      !anticolor-color annihilation
          if(iqc(jm).eq.0)then
            jt=3                    !first parton=g
            nqc(1)=ncc(jq,jm)
            njc1=ncc(3-jq,jm)
            njc2=ncc(1,3-jm)
            if(iqj(njc1).ne.0)then
              ncj(1,njc1)=njc2
            else
              ncj(jq,njc1)=njc2
            endif
            if(iqj(njc2).ne.0)then
              ncj(1,njc2)=njc1
            else
              ncj(3-jq,njc2)=njc1
            endif

          else
            jt=4                    !first parton=q(q~)
            nqc(1)=0
            njc1=ncc(1,jm)
            njc2=ncc(3-jq,3-jm)
            if(iqj(njc1).ne.0)then
              ncj(1,njc1)=njc2
            else
              ncj(3-jq,njc1)=njc2
            endif
            if(iqj(njc2).ne.0)then
              ncj(1,njc2)=njc1
            else
              ncj(jq,njc2)=njc1
            endif
          endif

        else                        !color transfer
          if(iqc(jm).eq.0)then
            jt=5                    !first parton=g
            nqc(2)=ncc(3-jq,jm)
            nqc(1)=ncc(1,3-jm)
          else                      !first parton=q(q~)
            jt=6
            nqc(1)=ncc(jq,3-jm)
          endif
        endif

       else          !q(q~)g->q(q~)-gamma (+-color annihilation)
          if(iqc(jm).eq.0)then
            jt=11                    !first parton=g
            nqc(1)=ncc(jq,jm)
            njc1=ncc(3-jq,jm)
            njc2=ncc(1,3-jm)
            if(iqj(njc1).ne.0)then
              ncj(1,njc1)=njc2
            else
              ncj(jq,njc1)=njc2
            endif
            if(iqj(njc2).ne.0)then
              ncj(1,njc2)=njc1
            else
              ncj(3-jq,njc2)=njc1
            endif
            iqc(jm)=iqc(3-jm)
            iqc(3-jm)=10              !make the second output is gamma.

          else
            jt=12                    !first parton=q(q~)
            nqc(1)=ncc(jq,3-jm)                 !here nqc(1) is gluon.
            njc1=ncc(1,jm)
            njc2=ncc(3-jq,3-jm)
            if(iqj(njc1).ne.0)then
              ncj(1,njc1)=njc2
            else
              ncj(3-jq,njc1)=njc2
            endif
            if(iqj(njc2).ne.0)then
              ncj(1,njc2)=njc1
            else
              ncj(jq,njc2)=njc1
            endif
            iqc(3-jm)=10
          endif
       endif

       elseif(iqc(1)*iqc(2).gt.0)then
        jt=7                        !qq->qq (q~q~->q~q~)
        if(iqc(1).gt.0)then
          jq=1
        else
          jq=2
        endif
        nqc(1)=ncc(1,3-jm)

      else                          ! qq~ ->
        if(iqc(jm).gt.0)then
          jq=1
        else
          jq=2
        endif
        aks=rangen()
        if(aks.lt.(wsub(1)+wsub(2))/fborn)then
          jt=8                     ! qq~->qq~ (anticolor-color annihilation)
          if(aks.gt.wsub(1)/fborn)then
            iqa=iabs(iqc(jm))
            iq=int((naflav-1)*rangen())+1
            if(iq.eq.iqa)iq=naflav
            iqc(jm)=iq*iqc(jm)/iqa
            iqc(3-jm)=-iqc(jm)
          endif
          nqc(1)=0
          njc1=ncc(1,jm)
          njc2=ncc(1,3-jm)
          if(iqj(njc1).ne.0)then
            ncj(1,njc1)=njc2
          else
            ncj(3-jq,njc1)=njc2
          endif
          if(iqj(njc2).ne.0)then
            ncj(1,njc2)=njc1
          else
            ncj(jq,njc2)=njc1
          endif
        elseif(aks.lt.(wsub(1)+wsub(2)+wsub(3))/fborn)then
          jt=10                    !color transfer  qq~->gg
          iqc(1)=0
          iqc(2)=0
          nqc(1)=ncc(1,jm)
          nqc(2)=0
        elseif(aks.lt.(wsub(1)+wsub(2)+wsub(3)+wsub(4))/fborn)then
          jt=13                   ! qq~->g+gamma
          nqc(1)=ncc(1,jm)
          nqc(2)=ncc(1,3-jm)
          iqc(jm)=0
          iqc(3-jm)=10
        else
          jt=14                  ! qq~->gamma+gamma
          njc1=ncc(1,jm)
          njc2=ncc(1,3-jm)
          if(iqj(njc1).ne.0)then
            ncj(jq,njc1)=njc2
          else
            ncj(3-jq,njc1)=njc2
          endif
          if(iqj(njc2).ne.0)then
            ncj(3-jq,njc2)=njc1
          else
            ncj(jq,njc2)=njc1
          endif
          iqc(jm)=10
          iqc(3-jm)=10
        endif
      endif

      if(jt.ne.8.and.jt.ne.9)then
        jq2=jq
      else
        jq2=3-jq
      endif

      call psdeftr(si+q2mass,ept,ey)    !lorentz boost to c.m. frame

      qt=sqrt(qt2)                      !p_t
      call pscs(bcos,bsin)              !cos and sin of the polar angle
      if(iabs(iqc(1)).ne.4)then
clight cone momentum share for the first parton
        z=sngl(psutz(dble(si),dble(qt2),dble(qt2)))
        if((jt.eq.11.and.jm.eq.1).or.(jt.eq.12.and.jm.eq.2)
     $   .or.(jt.eq.13.and.jm.eq.2))z=1-z
        wp3=z*sqrt(si)
        wm3=qt2/wp3
      elseif(jm.eq.1)then
        z=sngl(psutz(dble(si),dble(qt2+q2mass),dble(qt2)))
        wp3=z*sqrt(si)
        wm3=(qt2+q2mass)/wp3
      else
        z=sngl(psutz(dble(si),dble(qt2),dble(qt2+dble(q2mass))))
        wp3=z*sqrt(si)
        wm3=qt2/wp3
      endif
      ep3(1)=.5*(wp3+wm3)               !parton 4-momentum
      ep3(2)=.5*(wp3-wm3)
      ep3(3)=qt*bcos
      ep3(4)=qt*bsin
      call psdefrot(ep3,s0xh,c0xh,s0h,c0h)   !spacial rotation to z-axis

      zeta=2.                        !2=back-to-back emission (angle=pi)
      if(iqc(jm).eq.0)then
        iq2ini1=9
        jo1=iq2ini1
        if(zeta.gt.zetacut)jo1=-jo1
c        q2fin=q2fin+zoeinc
      else
        iq2ini1=iqc(jm)
        jo1=iq2ini1
      endif
      if(iqc(3-jm).eq.0)then
        iq2ini2=9
        jo2=iq2ini2
        if(zeta.gt.zetacut)jo2=-jo2
c        q2fin=q2fin+zoeinc
      else
        iq2ini2=iqc(3-jm)
        jo2=iq2ini2
      endif
      if(jt.le.10)then
        qq1=qt2
        qq2=qt2
      elseif(jt.le.13)then
        qq1=qt2
        qq2=0
      else
        qq1=0
        qq2=0
      endif

      call timsh2(qq1,qq2,sqrt(si),iq2ini1,iq2ini2,jo1,jo2)  !final state cascade

      if(jt.le.10)then      !color connection for the 2nd parton
        if(ish.ge.6)write(ifch,*)'jt,jq,nqc:',jt,jq,nqc
        call psreti(nqc,jq,1,ey,s0xh,c0xh,s0h,c0h) !color conn. reconstruction
        if(jt.eq.1)then
          nqc(1)=nqc(2)
          nqc(2)=ncc(3-jq,3-jm)
        elseif(jt.eq.2)then
          nqc(2)=ncc(3-jq,jm)
          nqc(1)=ncc(jq,3-jm)
        elseif(jt.eq.3)then
          nqc(1)=nqc(2)
        elseif(jt.eq.4)then
          nqc(2)=nqc(1)
          nqc(1)=ncc(jq,3-jm)
        elseif(jt.eq.5)then
          nqc(1)=ncc(jq,jm)
        elseif(jt.eq.6)then
          nqc(2)=ncc(3-jq,3-jm)
          nqc(1)=ncc(1,jm)
        elseif(jt.eq.7)then
          nqc(1)=ncc(1,jm)
        elseif(jt.eq.9)then
          nqc(1)=ncc(3-jq,3-jm)
        elseif(jt.eq.10)then
          nqc(1)=nqc(2)
          nqc(2)=ncc(1,3-jm)
        endif
        if(ish.ge.6)write(ifch,*)'jt,jq2,nqc:',jt,jq2,nqc
        call psreti(nqc,jq2,2,ey,s0xh,c0xh,s0h,c0h) !color conn. reconstr.
      elseif(jt.le.13)then
        if(ish.ge.6)write(ifch,*)'jt,jq,nqc:',jt,jq,nqc
        call psreti(nqc,jq,1,ey,s0xh,c0xh,s0h,c0h) !color conn. reconstruction
        ep3(1)=pprt(4,2)
        ep3(2)=pprt(3,2)
        ep3(3)=pprt(1,2)
        ep3(4)=pprt(2,2)
        call psrotat(ep3,s0xh,c0xh,s0h,c0h)  !special rotation for photon.
        call pstrans(ep3,ey,1)
        nptl=nptl+1
        pptl(1,nptl)=ep3(3)
        pptl(2,nptl)=ep3(4)
        pptl(3,nptl)=ep3(2)
        pptl(4,nptl)=ep3(1)
        pptl(5,nptl)=0
        idptl(nptl)=10
        iorptl(nptl)=iptl
        istptl(nptl)=0
        jorptl(nptl)=0
        do i=1,4
          xorptl(i,nptl)=coordo(i)
        enddo
        tivptl(1,nptl)=coordo(5)
        tivptl(2,nptl)=coordo(6)
        ityptl(nptl)=71
        ifrptl(1,nptl)=0
        ifrptl(2,nptl)=0
      else
        if(ish.ge.6)write(ifch,*)'jt,iqc:',jt,iqc
        do j=1,2
          ep3(1)=pprt(4,j)
          ep3(2)=pprt(3,j)
          ep3(3)=pprt(1,j)
          ep3(4)=pprt(2,j)
          call psrotat(ep3,s0xh,c0xh,s0h,c0h)  !special rotation for photon.
          call pstrans(ep3,ey,1)
          nptl=nptl+1
          pptl(1,nptl)=ep3(3)
          pptl(2,nptl)=ep3(4)
          pptl(3,nptl)=ep3(2)
          pptl(4,nptl)=ep3(1)
          pptl(5,nptl)=0
          idptl(nptl)=10
          iorptl(nptl)=iptl
          istptl(nptl)=0
          jorptl(nptl)=0
          do i=1,4
            xorptl(i,nptl)=coordo(i)
          enddo
          tivptl(1,nptl)=coordo(5)
          tivptl(2,nptl)=coordo(6)
          ityptl(nptl)=72
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
        enddo
      endif
      call utprix('psabor',ish,ishini,5)
      return
      end

c------------------------------------------------------------------------
      subroutine psreti(nqc,jort,nfprt,ey,s0xh,c0xh,s0h,c0h)
c-----------------------------------------------------------------------
c jet reconstructuring procedure - 4-momenta for all final jets
c nqc(i) - colour connections for the jet
c jort - color orientation for gluons (=1 if +color goes first, =-1 otherwise)

      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
c nprtj - number of partons in the jet (including virtual ones)
c pprt - 5-momenta for the partons
c idprt - parton id
c iorprt - parent parton position in the list
c idaprt - daughter partons positions in the list

c output:
      parameter (mjstr=20000)
      common /psar29/ eqj(4,mjstr),iqj(mjstr),ncj(2,mjstr),ioj(mjstr),nj
c nj - number of final jets
c eqj(i,j) - 4-momentum for the final jet j
c iqj(j) - flavour for the final jet j
c ncj(m,j) - colour connections for the final jet j
      common /psar30/ iorj(mjstr),ityj(mjstr),bxj(6,mjstr),q2j(mjstr)
c-----------------------------------------------------------------------
      dimension ep3(4),nqc(2),ncc(2,ntim),ey(3)
      include 'epos.inc'
      include 'epos.incsem'
      common/cprtx/nprtjx,pprtx(5,2)

      if(ish.ge.6)then
        write (ifch,*)'nprtj',nprtj
        do i=1,nprtj
          write (ifch,*)'i,ic,np,ndd',i,idprt(i),iorprt(i),
     *    idaprt(1,i),idaprt(2,i)
        enddo
      endif

      ncc(1,nfprt)=nqc(1)
      if(idprt(nfprt).eq.9)ncc(2,nfprt)=nqc(2)
      iprt=nfprt

      if(nprtjx.eq.2)then !out Born before timelike cascade
       ep3(1)=pprtx(4,iprt)
       ep3(2)=pprtx(3,iprt)
       ep3(3)=pprtx(1,iprt)
       ep3(4)=pprtx(2,iprt)
       call psrotat(ep3,s0xh,c0xh,s0h,c0h)
       call pstrans(ep3,ey,1)
       pprtx(4,iprt)=ep3(1)
       pprtx(3,iprt)=ep3(2)
       pprtx(1,iprt)=ep3(3)
       pprtx(2,iprt)=ep3(4)
      endif

1     continue

      idau1=idaprt(1,iprt)
      idau2=idaprt(2,iprt)
      icp=idprt(iprt)

      if(ish.ge.6)then
        write (ifch,*)'1-iprt,icp,idau1,idau2',iprt,icp,idau1,idau2,
     *  ncc(1,iprt)
        if(icp.eq.9)write (ifch,*)'ncc2',ncc(2,iprt)
      endif

      if(idau1.ne.0.)then         !virtual parton
        icd1=idprt(idau1)

        if(icp.eq.9)then
          if(icd1.ne.9)then      !g -> qq~
            ncc(1,idau1)=ncc(jort,iprt)
            ncc(1,idau2)=ncc(3-jort,iprt)
          else                    !g -> gg
            ncc(1,idau1)=ncc(1,iprt)
            ncc(2,idau1)=0
            ncc(2,idau2)=ncc(2,iprt)
            ncc(1,idau2)=0
          endif
        else                      !q -> qg
          ncc(1,idau1)=0
          if(icp*(3-2*jort).gt.0)then
            ncc(1,idau2)=ncc(1,iprt)
            ncc(2,idau2)=0
          else
            ncc(1,idau2)=0
            ncc(2,idau2)=ncc(1,iprt)
          endif
        endif
        iprt=idau1
        goto 1
      else

        nj=nj+1
        ep3(1)=pprt(4,iprt)
        ep3(2)=pprt(3,iprt)
        ep3(3)=pprt(1,iprt)
        ep3(4)=pprt(2,iprt)
        call psrotat(ep3,s0xh,c0xh,s0h,c0h)
        call pstrans(ep3,ey,1)
        do i=1,4
          eqj(i,nj)=ep3(i)
        enddo

        if(icp.eq.9)then
          iqj(nj)=0
        elseif(iabs(icp).lt.3)then
          iqj(nj)=icp
        elseif(iabs(icp).eq.3)then
          iqj(nj)=icp*4/3
        else
          iqj(nj)=icp*10
        endif

        ioj(nj)=jorprt(iprt) !flavor of mother parton
        q2j(nj)=q2prt(iprt)

        if(iqj(nj).ne.0)then
          njc=ncc(1,iprt)
          if(njc.ne.0)then
            ncj(1,nj)=njc
            iqc=iqj(njc)
            if(iqc.ne.0)then
              ncj(1,njc)=nj
            else
              if(iqj(nj).gt.0)then
                ncj(2,njc)=nj
              else
                ncj(1,njc)=nj
              endif
            endif
          else
            ncc(1,iprt)=nj
          endif
        else

          do m=1,2
            if(jort.eq.1)then
              m1=m
            else
              m1=3-m
            endif
            njc=ncc(m1,iprt)
            if(njc.ne.0)then
              ncj(m,nj)=njc
              iqc=iqj(njc)
              if(iqc.ne.0)then
                ncj(1,njc)=nj
              else
                ncj(3-m,njc)=nj
              endif
            else
              ncc(m1,iprt)=nj
            endif
          enddo
        endif
        if(ish.ge.6)then
          write (ifch,*)'jet-nj,iprt,icp,iqj(nj),ioj(nj),ncj',
     *    nj,iprt,icp,iqj(nj),ioj(nj),ncj(1,nj)
          if(icp.eq.9)write (ifch,*)'ncj2',ncj(2,nj)
        endif
      endif

2     continue
      if(iprt.ne.nfprt)then
        icp=idprt(iprt)
        ipar=iorprt(iprt)
        idau1=idaprt(1,ipar)
        idau2=idaprt(2,ipar)
        if(ish.ge.6)then
          write (ifch,*)'2-iprt,icp,idau1,idau2,ipar',
     *    iprt,icp,idau1,idau2,ipar,ncc(1,iprt)
          if(icp.eq.9)write (ifch,*)ncc(2,iprt)
        endif

        if(idau1.eq.iprt)then
          if(icp.eq.9)then                   !g -> gg
            ncc(1,ipar)=ncc(1,iprt)
            ncc(1,idau2)=ncc(2,iprt)
          else
            icpar=idprt(ipar)
            if(icpar.eq.9)then               !g -> qq~
              ncc(jort,ipar)=ncc(1,iprt)
            else                             !q -> qg
              if(icp*(3-2*jort).gt.0)then
                ncc(2,idau2)=ncc(1,iprt)
              else
                ncc(1,idau2)=ncc(1,iprt)
              endif
            endif
          endif
          iprt=idau2
          goto 1

        else
          if(icp.eq.9)then
            icpar=idprt(ipar)
            if(icpar.eq.9)then                !g -> gg
              ncc(2,ipar)=ncc(2,iprt)
              ncc(2,idau1)=ncc(1,iprt)
            else                              !q -> qg
              if(icpar*(3-2*jort).gt.0)then
                ncc(1,ipar)=ncc(1,iprt)
                ncc(1,idau1)=ncc(2,iprt)
              else
                ncc(1,ipar)=ncc(2,iprt)
                ncc(1,idau1)=ncc(1,iprt)
              endif
            endif
          else
            ncc(3-jort,ipar)=ncc(1,iprt)
          endif
          iprt=ipar
          goto 2
        endif
      else
        if(ish.ge.6)write (ifch,*)'3-iprt,ncc',iprt,ncc(1,iprt)
        nqc(1)=ncc(1,nfprt)
        if(idprt(nfprt).eq.9)nqc(2)=ncc(2,nfprt)
      endif
      return
      end

