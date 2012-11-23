c------------------------------------------------------------------------
      function ffsigiut(xx1,xx2,jpp,je1,je2)
c------------------------------------------------------------------------
c
c   \int(dt) \int(du)  ffsig *s/sh**3 *2*pi*alpha**2 *delta(uh+th+sh)
c
c-----------------------------------------------------------------------
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'
      include 'epos.inc'
      double precision tmin,tmax,t,sh2,sqrtq2s

      ig=3
      s=engy**2
      sh=s*xx1*xx2
      ffsigiut=0.
      if(sh.le.4.*q2min)return
      sh2=dble(sh/2.)
c      tmin=sh/2-sqrt(sh*sh/4-q2min*sh)
      sqrtq2s=sqrt(dble(q2min*sh))
      tmin=sh2-sqrt((sh2-sqrtq2s)*(sh2+sqrtq2s))
      tmax=sh2
      do i=1,ig
      do m=1,2
        t=2d0*tmin/(1d0+tmin/tmax-dble(tgss(ig,i)*(2*m-3))
     &       *(1d0-tmin/tmax))
        qq=sngl(t*(1d0-t/dble(sh)))
        ft=ffsigj(sngl(t),qq,xx1,xx2,jpp,je1,je2)/sh**3
     *         * (2*pi*pssalf(qq/qcdlam))**2
        ffsigiut=ffsigiut+wgss(ig,i)*ft*sngl(t)**2
      enddo
      enddo
      ffsigiut=ffsigiut
     *    *0.5*sngl(1d0/tmin-1d0/tmax)
     *    *2*pi*s
     *   /2      !CS for parton pair
      return
      end

c-----------------------------------------------------------------------
      function ffsigj(t,qt,x1,x2,jpp,je1,je2)
c-----------------------------------------------------------------------
c
c      \sum  x1*f_i(x1,qt) * x2*f_k(x2,qt) * B_ik
c
c        B_ik = psbori = contribution to Born xsection:
c                         dsigmaBorn/d2pt/dy
c                          = s/pi * delta(s+t+u) * 2*pi*alpha**2 /s**2 * B_ik
c
c  qt = virtuality scale
c  x1, x2 = light cone momentum fractions
c
c  x*f_j(x,qt) = function fparton(x,qt,j)
c
c-----------------------------------------------------------------------
c jpp: type of Pomeron
c          1 ... sea-sea
c          2 ... val-sea
c          3 ... sea-val
c          4 ... val-val
c          5 ... all
c je = emission type
c          0 ... no emissions
c          1 ... emissions
c          2 ... all
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'

      s=engy**2*x1*x2

      if(jpp.ne.5)then
      ji1=mod(jpp+1,2)+1
      ji2=(jpp+1)/2
      sea1=pifpartone(x1,qt,-1,je1,ji1)
      g1=  pifpartone(x1,qt, 0,je1,ji1)
      uv1= pifpartone(x1,qt, 1,je1,ji1)
      dv1= pifpartone(x1,qt, 2,je1,ji1)
      sea2=pifpartone(x2,qt,-1,je2,ji2)
      g2=  pifpartone(x2,qt, 0,je2,ji2)
      uv2= pifpartone(x2,qt, 1,je2,ji2)
      dv2= pifpartone(x2,qt, 2,je2,ji2)
      else
      sea1=pifpartone(x1,qt,-1,je1,1)+pifpartone(x1,qt,-1,je1,2)
      g1=  pifpartone(x1,qt, 0,je1,1)+pifpartone(x1,qt, 0,je1,2)
      uv1= pifpartone(x1,qt, 1,je1,1)+pifpartone(x1,qt, 1,je1,2)
      dv1= pifpartone(x1,qt, 2,je1,1)+pifpartone(x1,qt, 2,je1,2)
      sea2=pifpartone(x2,qt,-1,je2,1)+pifpartone(x2,qt,-1,je2,2)
      g2=  pifpartone(x2,qt, 0,je2,1)+pifpartone(x2,qt, 0,je2,2)
      uv2= pifpartone(x2,qt, 1,je2,1)+pifpartone(x2,qt, 1,je2,2)
      dv2= pifpartone(x2,qt, 2,je2,1)+pifpartone(x2,qt, 2,je2,2)
      endif

      ffsigj= ffborn(s,t,  g1*g2                                  !gg

     *  ,(uv1+dv1+2.*naflav*sea1)*g2+g1*(uv2+dv2+2.*naflav*sea2)    !gq

     *  ,(uv1+sea1)*(uv2+sea2)                                      !qq
     *      +(dv1+sea1)*(dv2+sea2)+sea1*sea2*(naflav-1)*2.

     *  ,(uv1+sea1)*sea2+(uv2+sea2)*sea1                            !qa
     *    +(dv1+sea1)*sea2+(dv2+sea2)*sea1+sea1*sea2*(naflav-2)*2.

     *  ,dv1*uv2+dv2*uv1+(uv2+dv2)*sea1*(naflav-1)*2.                    !qqp
     *    +(uv1+dv1)*sea2*(naflav-1)*2.
     *    +sea1*sea2*naflav*(naflav-1)*4.

     *)
      end

c-----------------------------------------------------------------------
      function ffsig(t,qt,x1,x2)    !former psjy
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'

      s=engy**2*x1*x2

      g1=  pifpartone(x1,qt, 0,2,1)+pifpartone(x1,qt, 0,2,2)
      uv1= pifpartone(x1,qt, 1,2,1)+pifpartone(x1,qt, 1,2,2)
      dv1= pifpartone(x1,qt, 2,2,1)+pifpartone(x1,qt, 2,2,2)
      sea1=pifpartone(x1,qt,-1,2,1)+pifpartone(x1,qt,-1,2,2)
      g2=  pifpartone(x2,qt, 0,2,1)+pifpartone(x2,qt, 0,2,2)
      uv2= pifpartone(x2,qt, 1,2,1)+pifpartone(x2,qt, 1,2,2)
      dv2= pifpartone(x2,qt, 2,2,1)+pifpartone(x2,qt, 2,2,2)
      sea2=pifpartone(x2,qt,-1,2,1)+pifpartone(x2,qt,-1,2,2)

      ffsig= ffborn(s,t,  g1*g2                                  !gg

     *  ,(uv1+dv1+2.*naflav*sea1)*g2+g1*(uv2+dv2+2.*naflav*sea2)   !gq

     *  ,(uv1+sea1)*(uv2+sea2)                                     !qq
     *      +(dv1+sea1)*(dv2+sea2)+sea1*sea2*(naflav-1)*2.

     *  ,(uv1+sea1)*sea2+(uv2+sea2)*sea1                           !qa
     *    +(dv1+sea1)*sea2+(dv2+sea2)*sea1+sea1*sea2*(naflav-2)*2.

     *  ,dv1*uv2+dv2*uv1+(uv2+dv2)*sea1*(naflav-1)*2.             !qqp
     *    +(uv1+dv1)*sea2*(naflav-1)*2.
     *    +sea1*sea2*naflav*(naflav-1)*4.

     *)
      end

c------------------------------------------------------------------------
      function ffborn(s,t,gg,gq,qq,qa,qqp)
c------------------------------------------------------------------------

      ffborn=
     *( psbori(s,t,0,0,1)+psbori(s,s-t,0,0,1)
     * +psbori(s,t,0,0,2)+psbori(s,s-t,0,0,2)) /2.   *gg             !gg

     *+(psbori(s,t,0,1,1)+psbori(s,s-t,0,1,1))       *gq             !gq

     *+(psbori(s,t,1,1,1)+psbori(s,s-t,1,1,1))/2.    *qq             !qq

     *+(psbori(s,t,1,-1,1)+psbori(s,s-t,1,-1,1)+psbori(s,t,1,-1,2)+
     * psbori(s,s-t,1,-1,2)+psbori(s,t,1,-1,3)+psbori(s,s-t,1,-1,3)) !qa
     *                                               *qa

     *+(psbori(s,t,1,2,1)+psbori(s,s-t,1,2,1))       *qqp            !qq'

      end

c-----------------------------------------------------------------------
      function pifpartone(xx,qq,j,je,ji)  ! pol interpolation of partone
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'
      common/tabfptn/kxxmax,kqqmax,fptn(20,20,-1:2,0:2,2)
      real wi(3),wj(3)
      common /cpifpartone/npifpartone
      data npifpartone /0/
      npifpartone=npifpartone+1
      if(npifpartone.eq.1)call MakeFpartonTable

      qqmax=engy**2/4.
      xxmin=0.01/engy
      xxmax=1

      xxk=1.+log(xx/xxmin)/log(xxmax/xxmin)*(kxxmax-1)
      qqk=1.+log(qq/q2min)/log(qqmax/q2min)*(kqqmax-1)
      kxx=int(xxk)
      kqq=int(qqk)
      if(kxx.lt.1)kxx=1
      if(kqq.lt.1)kqq=1
      if(kxx.gt.(kxxmax-2))kxx=kxxmax-2
      if(kqq.gt.(kqqmax-2))kqq=kqqmax-2

      wi(2)=xxk-kxx
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wj(2)=qqk-kqq
      wj(3)=wj(2)*(wj(2)-1.)*.5
      wj(1)=1.-wj(2)+wj(3)
      wj(2)=wj(2)-2.*wj(3)
      pifpartone=0
      do kx=1,3
      do kq=1,3
        pifpartone=pifpartone+fptn(kxx+kx-1,kqq+kq-1,j,je,ji)
     *              *wi(kx)*wj(kq)
      enddo
      enddo
      end

c-----------------------------------------------------------------------
      subroutine MakeFpartonTable
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'
      common/tabfptn/kxxmax,kqqmax,fptn(20,20,-1:2,0:2,2)
      write (*,'(a,$)')'(Fparton table'
      kxxmax=10
      kqqmax=10
      qqmax=engy**2/4.
      xxmin=0.01/engy
      xxmax=1
      do ji=1,2
       do je=0,2
        write(*,'(a,$)')'.'
        do j=-1,2
         do kxx=1,kxxmax
          xx=xxmin*(xxmax/xxmin)**((kxx-1.)/(kxxmax-1.))
          do kqq=1,kqqmax
           qq=q2min*(qqmax/q2min)**((kqq-1.)/(kqqmax-1.))
           fptn(kxx,kqq,j,je,ji)= fpartone(xx,qq,j,je,ji)
          enddo
         enddo
        enddo
       enddo
      enddo
      write (*,'(a,$)')'done)'
      end

c------------------------------------------------------------------------
      function fpartone(xx,qq,j,je,ji)                 !former pspdf0 (sha)
c-----------------------------------------------------------------------
c
c  parton distribution function for proton  ( actually x*f(x) !!!!!!! )
c
c xx = light cone momentum fraction
c qq = virtuality scale
c j = parton type
c         -1 ... sea  (distribution function per flavor)
c          0 ... g
c          1 ... u
c          2 ... d
c je = emission type
c          0 ... no emissions
c          1 ... emissions
c          2 ... all
c ji = initial parton type
c          1 ... sea (q et g)
c          2 ... val
c-----------------------------------------------------------------------
      double precision z,xmin,xm,zx,psuds
      common/ar3/    x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      fpartone=0
      if(je.eq.1)goto888

c ...... f_0 * sudakov.........

      if(j.eq.0.and.ji.eq.1)then
        fpartone=fzeroGlu(xx,2,1)         !hadron class 2, projectile side
      elseif((j.eq.1.or.j.eq.2).and.ji.eq.2)then
        fpartone=psdfh4(xx,q2min,0.,2,j)
      elseif(j.eq.-1.and.ji.eq.1)then
        fpartone=fzeroSea(xx,2,1)
      endif
      fpartone=fpartone*sngl(psuds(qq,j)/psuds(q2min,j))
      if(je.eq.0)goto999

c......... integral f_0 E_qcd............

 888  continue
      xmin=dble(xx)/(1.d0-dble(q2ini/qq))
      if(xmin.lt.1.d0)then
        dpd1=0.
        dpd2=0.
        xm=max(xmin,0.3d0)

 !numerical integration xm -> 1

        do i=1,7
        do m=1,2
          zx=1.d0-(1.d0-xm)*(.5d0+(dble(m)-1.5d0)*dble(x1(i)))**.25d0
          z=xx/zx

          gl=fzeroGlu(sngl(zx),2,1)
          uv=psdfh4(sngl(zx),q2min,0.,2,1)
          dv=psdfh4(sngl(zx),q2min,0.,2,2)
          sea=fzeroSea(sngl(zx),2,1)

          fz=0
          if(j.eq.0)then
            if(ji.eq.1)
     *        fz=gl *psevi(q2min,qq,z,1,1)
     *          +sea*psevi(q2min,qq,z,2,1)  !ccccc
            if(ji.eq.2)
     *           fz=(uv+dv)*psevi(q2min,qq,z,2,1)
          elseif(j.eq.1.and.ji.eq.2)then
            fz=psevi(q2min,qq,z,3,2)*uv
          elseif(j.eq.2.and.ji.eq.2)then
            fz=psevi(q2min,qq,z,3,2)*dv
          elseif(j.eq.-1)then
            akns=psevi(q2min,qq,z,3,2)            !nonsinglet contribution
            aks=(psevi(q2min,qq,z,2,2)-akns)      !singlet contribution
            if(ji.eq.1)
     *        fz=psevi(q2min,qq,z,1,2)*gl
     *          +sea*aks+sea*akns !ccccc
            if(ji.eq.2)
     *        fz=(uv+dv)*aks
          endif
          dpd1=dpd1+a1(i)*fz/sngl(zx)**2/sngl(1.d0-zx)**3
        enddo
        enddo
        dpd1=dpd1*sngl(1.d0-xm)**4/8.*xx

 !numerical integration  xmin -> xm

        if(xm.gt.xmin)then
          do i=1,7
          do m=1,2
            zx=xx+(xm-xx)
     &         *((xmin-xx)/(xm-xx))**(.5d0-(dble(m)-1.5d0)*dble(x1(i)))
            z=xx/zx

            gl=fzeroGlu(sngl(zx),2,1)
            uv=psdfh4(sngl(zx),q2min,0.,2,1)
            dv=psdfh4(sngl(zx),q2min,0.,2,2)
            sea=fzeroSea(sngl(zx),2,1)

            fz=0
            if(j.eq.0)then
              if(ji.eq.1)
     *        fz=gl *psevi(q2min,qq,z,1,1)
     *          +sea*psevi(q2min,qq,z,2,1)     !ccccc
              if(ji.eq.2)
     *                fz=(uv+dv)*psevi(q2min,qq,z,2,1)
            elseif(j.eq.1.and.ji.eq.2)then
              fz=psevi(q2min,qq,z,3,2)*uv
            elseif(j.eq.2.and.ji.eq.2)then
              fz=psevi(q2min,qq,z,3,2)*dv
            elseif(j.eq.-1)then
              akns=psevi(q2min,qq,z,3,2)            !nonsinglet contribution
              aks=(psevi(q2min,qq,z,2,2)-akns)      !singlet contribution
              if(ji.eq.1)
     *          fz=psevi(q2min,qq,z,1,2)*gl
     *              +sea*aks+sea*akns     !ccccc
              if(ji.eq.2)
     *          fz=(uv+dv)*aks
            endif
            dpd2=dpd2+a1(i)*fz*sngl((1.d0-xx/zx)/zx)
          enddo
          enddo
          dpd2=dpd2*sngl(log((xm-xx)/(xmin-xx))*.5d0*xx)
        endif
        fpartone=fpartone+dpd2+dpd1
      endif

  999 continue
      if(j.lt.0)fpartone=fpartone/naflav/2.
      return
      end

c------------------------------------------------------------------------
      function fparton(xx,qq,j)                 !former pspdf0 (sha)
c-----------------------------------------------------------------------
c
c  parton distribution function for proton  ( actually x*f(x) !!!!!!! )
c
c xx = light cone momentum fraction
c qq = virtuality scale
c j = parton type
c         -1 ... sea  (dsistribution fuction per flavor)
c          0 ... g
c          1 ... u
c          2 ... d
c
c-----------------------------------------------------------------------
c (see pages 105 - 107 of our report)
c
c  fparton(xx) = xx * f(xx)   !!!!!
c
c     f_j(xx,qq) = \sum_k \int(xx<x<1) dx/x f0_k(x) Eqcd_k_j(xx/x,qq)
c
c      f0_k = fzeroGlu or fzeroSea
c
c      Eqcd=E~qcd+delta*sudakov,  E~qcd: at least one emission
c
c-----------------------------------------------------------------------
      double precision z,xmin,xm,zx,psuds
      common/ar3/    x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

c ...... f_0 * sudakov.........

      if(j.eq.0)then
        fparton=fzeroGlu(xx,2,1)
      elseif(j.eq.1.or.j.eq.2)then
        fparton=psdfh4(xx,q2min,0.,2,j)
      else
        fparton=fzeroSea(xx,2,1)
      endif
      fparton=fparton*sngl(psuds(qq,j)/psuds(q2min,j))

c......... integral f_0 E_qcd............

      xmin=xx/(1.d0-dble(q2ini/qq))
      if(xmin.lt.1.d0)then
        dpd1=0.
        dpd2=0.
        xm=max(xmin,.3d0)

 !numerical integration xm -> 1

        do i=1,7
        do m=1,2
          zx=1.d0-(1.d0-xm)*(.5d0+(dble(m)-1.5d0)*dble(x1(i)))**.25d0
          z=xx/zx

          gl=fzeroGlu(sngl(zx),2,1)
          uv=psdfh4(sngl(zx),q2min,0.,2,1)
          dv=psdfh4(sngl(zx),q2min,0.,2,2)
          sea=fzeroSea(sngl(zx),2,1)

          if(j.eq.0)then
            fz=psevi(q2min,qq,z,1,1)*gl
     *            +(uv+dv+sea)*psevi(q2min,qq,z,2,1)
          elseif(j.eq.1)then
            fz=psevi(q2min,qq,z,3,2)*uv
          elseif(j.eq.2)then
            fz=psevi(q2min,qq,z,3,2)*dv
          else
            akns=psevi(q2min,qq,z,3,2)            !nonsinglet contribution
            aks=(psevi(q2min,qq,z,2,2)-akns)      !singlet contribution
            fz=(psevi(q2min,qq,z,1,2)*gl+(uv+dv+sea)*aks+sea*akns)
          endif
          dpd1=dpd1+a1(i)*fz/sngl(zx)**2/sngl(1.d0-zx)**3
        enddo
        enddo
        dpd1=dpd1*sngl((1.d0-xm)**4/8.*xx)

 !numerical integration  xmin -> xm

        if(xm.gt.xmin)then
          do i=1,7
          do m=1,2
            zx=xx+(xm-xx)*((xmin-xx)/(xm-xx))
     *             **(.5d0-(dble(m)-1.5)*dble(x1(i)))
            z=xx/zx

            gl=fzeroGlu(sngl(zx),2,1)
            uv=psdfh4(sngl(zx),q2min,0.,2,1)
            dv=psdfh4(sngl(zx),q2min,0.,2,2)
            sea=fzeroSea(sngl(zx),2,1)

            if(j.eq.0)then
              fz=psevi(q2min,qq,z,1,1)*gl+(uv+dv+sea)*
     *        psevi(q2min,qq,z,2,1)
            elseif(j.eq.1)then
              fz=psevi(q2min,qq,z,3,2)*uv
            elseif(j.eq.2)then
              fz=psevi(q2min,qq,z,3,2)*dv
            else
              akns=psevi(q2min,qq,z,3,2)            !nonsinglet contribution
              aks=(psevi(q2min,qq,z,2,2)-akns)      !singlet contribution
              fz=(psevi(q2min,qq,z,1,2)*gl+(uv+dv+sea)*aks+sea*akns)
            endif
            dpd2=dpd2+a1(i)*fz*sngl((1.d0-xx/zx)/zx)
          enddo
          enddo
          dpd2=dpd2*sngl(log((xm-xx)/(xmin-xx))*.5d0*xx)
        endif
        fparton=fparton+dpd2+dpd1
      endif
      if(j.lt.0)fparton=fparton/naflav/2.
      return
      end

c------------------------------------------------------------------------
      function fzeroGlu(z,k,ipt)
c-----------------------------------------------------------------------
c
c        x*f(x)
c
c   f = F & EsoftGluon         &=convolution
c
c   F(x) = alpff(k)*x**betff(ipt)*(1-x)**alplea(k)
c
c   EsoftGluon(x) = x**(-1-dels) * EsoftGluonTil(x)
c
c z - light cone x
c k - hadron class
c ipt - 1=proj 2=targ
c-----------------------------------------------------------------------
      double precision xpmin,xp
      include 'epos.inc'
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'

      fzeroGlu=0.
      xpmin=z
      xpmin=xpmin**(1+betff(ipt)+dels)
      do i=1,7
      do m=1,2
        xp=(.5*(1.+xpmin+(2*m-3)*x1(i)*(1.-xpmin)))**(1./
     *            (1+betff(ipt)+dels))
        zz=z/xp
        fzeroGlu=fzeroGlu+a1(i)*(1.-xp)**alplea(k)*EsoftGluonTil(zz)
      enddo
      enddo
      fzeroGlu=fzeroGlu*.5*(1.-xpmin)/(1+betff(ipt)+dels)

      fzeroGlu=fzeroGlu *alpff(k) *z**(-dels)

      end

c------------------------------------------------------------------------
      function fzeroSea(z,k,ipt)
c-----------------------------------------------------------------------
c
c        x*f(x)
c
c   f = F & EsoftQuark         &=convolution
c
c   F(x) = alpff(k)*x**betff(ipt)*(1-x)**alplea(k)
c
c   EsoftQuark(x) = x**(-1-dels) * EsoftQuarkTil(x)
c
c z - light cone x of the quark,
c k - hadron class
c-----------------------------------------------------------------------
      double precision xpmin,xp
      common /ar3/   x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      fzeroSea=0.
      xpmin=z
      xpmin=xpmin**(1+betff(ipt)+dels)
      do i=1,7
      do m=1,2
        xp=(.5*(1.+xpmin+(2*m-3)*x1(i)*(1.-xpmin)))**(1./
     *            (1+betff(ipt)+dels))
        zz=z/xp
        fzeroSea=fzeroSea+a1(i)*(1.-xp)**alplea(k)*EsoftQuarkTil(zz)
      enddo
      enddo
      fzeroSea=fzeroSea*.5*(1.-xpmin)/(1+betff(ipt)+dels)

      fzeroSea=fzeroSea *alpff(k) *z**(-dels)

      end

c------------------------------------------------------------------------
      function EsoftGluonTil(zz)
c-----------------------------------------------------------------------
c   EsoftGluon = zz^(-1-dels) * EsoftGluonTil
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      EsoftGluonTil=gamsoft*(1-glusea)*(1.-zz)**betpom
      end

c------------------------------------------------------------------------
      function EsoftQuarkTil(zz)
c-----------------------------------------------------------------------
c   EsoftQuark = zz^(-1-dels) * EsoftQuarkTil
c-----------------------------------------------------------------------
      double precision zmin,z
      common /ar3/   x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      EsoftQuarkTil=0.
      zmin=zz
      zmin=zmin**(1.+dels)
      do i=1,7
      do m=1,2
        z=(.5d0*(1.+zmin+(2*m-3)*x1(i)*(1.d0-zmin)))
     *  **(1.d0/(1.d0+dels))
        EsoftQuarkTil=EsoftQuarkTil+a1(i)*max(1.d-5,(1.d0-zz/z))**betpom
     *  *(z**2+(1.-z)**2)
      enddo
      enddo
      EsoftQuarkTil=EsoftQuarkTil*1.5*(1.d0-zmin)/(1.+dels)
                                                !1.5=naflav/2 at Q0
      EsoftQuarkTil=gamsoft*glusea*EsoftQuarkTil

      end

c------------------------------------------------------------------------
      function EsoftQZero(zz)    ! former psftilf
c-----------------------------------------------------------------------
c
c   EsoftQuark = EsoftQZero * wsplit * z^(-1-dels) * gamsoft
c
c zz - ratio of the quark and pomeron light cone x (zz=x_G/x_P)
c integration over quark to gluon light cone momentum ratio (z=x/x_G):
c
c   EsoftQZero = int(dz) z^dels * (1-zz/z)^betpom * P_qG(z)
c
c-----------------------------------------------------------------------
      double precision zmin,z
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'

      EsoftQZero=0.
      zmin=zz
      zmin=zmin**(1.+dels)
      do i=1,7
      do m=1,2
        z=(.5d0*(1.+zmin+(2*m-3)*x1(i)*(1.d0-zmin)))
     *  **(1.d0/(1.d0+dels))
        EsoftQZero=EsoftQZero+a1(i)*max(1.d-5,(1.d0-zz/z))**betpom
     *  *(z**2+(1.-z)**2)
      enddo
      enddo
      EsoftQZero=EsoftQZero*1.5*(1.d0-zmin)/(1.+dels)   !1.5=naflav/2 at Q0
      return
      end

c------------------------------------------------------------------------
      function ffsigi(qq,y0)                   !former psjx1  (sto)
c------------------------------------------------------------------------
c
c    dsigma/dpt_jet =  \int dy \int dx1  ffsig(x1,x2(x1))
c
c x1=xplus, x2=xminus
c x2=x2(x1) due to u+t+s=0
c ( s=x1*x2*spp, t/spp=-x1*xt*exp(-y)/2, u/spp=-x2*xt*exp(y)/2 )
c
c qq = pt**2,  xt=2.*sqrt(qq/s)
c rapidity range: 0 to y0
c
c    ffsig = function ffsig(t,qq,x1,x2)
c
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'
      double precision xx1,xx2,xt,ymax,ymin,y,xmin,xmax
      ig=3
      ig1=3
      s=engy**2
      ffsigi=0.
      if(s.le.4.*qq)return
      if(qq.lt.q2min)return
      xt=2d0*sqrt(dble(qq)/dble(s))
      ymax=min(dble(y0),log(1d0/xt+sqrt((1d0/xt-1d0)*(1d0/xt+1d0))))
      ymin=-ymax                          !final result must be divided by 2
      do i=1,ig
      do m=1,2
        y=.5d0*(ymax+ymin+(ymin-ymax)*dble((2*m-3)*tgss(ig,i)))
       !for xx1-integration, use variable x=xx1-xt*exp(y)/2.,with xmin<x<xmax
        xmin=xt**2/2.d0/(2.d0-xt*exp(-y))                    !condition x2<1
        xmax=1.d0-xt*exp(y)/2.d0                             !condition x1<1
        fx=0.
        do i1=1,ig1
        do m1=1,2
          xx1=xt*exp(y)/2.d0+xmin*(xmax/xmin)**dble(.5
     &                                           +tgss(ig1,i1)*(m1-1.5))
          xx2=xt*exp(-y)*xx1/(2.d0*xx1-xt*exp(y))
          z=sngl(xx1*xx2)
          sh=z*s
          aa=1.-4.*qq/sh
          aa=max(1e-10,aa)
          t=sh/2.*(1.-sqrt(aa))               !formula in parton-parton cms
          ft=ffsig(t,qq,sngl(xx1),sngl(xx2))
          fx=fx+wgss(ig1,i1)*ft/sh**2
        enddo
        enddo
        fx=fx*0.5*sngl(log(xmax/xmin))       !dx/x=0.5*log(xmax/xmin)dt (gauss)
        ffsigi=ffsigi+wgss(ig,i)*fx
      enddo
      enddo
      ffsigi=ffsigi*0.5*sngl(ymax-ymin)    !dy=0.5*(ymax-ymin)dt (gauss)
     *  *2*pi*(2*pi*pssalf(qq/qcdlam))**2      !alpha = 2*pi*pssalf
     *   *2*sqrt(qq)                 !d2pt=2*pi*pt*dpt
     *   /2   ! y interval  2 * Delta_y
     *   /2   ! condition t < sqrt(s)/2,
              !     since t > sqrt(s)/2 is automatically included,
              !      see psbori
      return
      end

c------------------------------------------------------------------------
      function psbori(s,t,j,l,n)
c-----------------------------------------------------------------------
c contribution to the born cross-section:
c
c   dsigmaBorn/d2pt/dy = s/pi * delta(s+t+u) * 2*pi*alpha**2 /s**2 *psbori
c
c s - c.m. energy squared for the born scattering,
c t - invariant variable for the born scattering |(p1-p3)**2|,
c j - parton type at current end of the ladder (0 - g, 1,-1,2,... - q)
c l - parton type at opposite end of the ladder (0 - g, 1,-1,2,... - q)
c n - subprocess number
c-----------------------------------------------------------------------
      include 'epos.incsem'

      psbori=0.
      u=s-t
      if(u.le.0.d0)return

      if(iabs(j).ne.4)then           !light quarks and gluons
        if(n.eq.1)then
          if(j.eq.0.and.l.eq.0)then                   !gg->gg
            psbori=(3.-t*u/s**2+s*u/t**2+s*t/u**2)*4.5
          elseif(j*l.eq.0)then                        !gq->gq
            psbori=(s**2+u**2)/t**2+(s/u+u/s)/2.25
          elseif(j.eq.l)then                          !qq->qq
            psbori=((s**2+u**2)/t**2+(s**2+t**2)/u**2)/2.25
     *      -s**2/t/u/3.375
          elseif(j.eq.-l)then                         !qq~->qq~
            psbori=((s**2+u**2)/t**2+(u**2+t**2)/s**2)/2.25
     *      +u**2/t/s/3.375
          else                                        !qq'->qq'
            psbori=(s**2+u**2)/t**2/2.25
          endif
        elseif(n.eq.2)then
          if(j.eq.0.and.l.eq.0)then                   !gg->qq~
            psbori=.5*(t/u+u/t)-1.125*(t*t+u*u)/s**2
          elseif(j.eq.-l)then                         !qq~->q'q'~
            psbori=(t*t+u*u)/s**2/1.125
          else
            psbori=0.
          endif
        elseif(n.eq.3)then
          if(j.ne.0.and.j.eq.-l)then                  !qq~->gg
            psbori=32./27.*(t/u+u/t)-(t*t+u*u)/s**2/.375
          else
            psbori=0.
          endif

c............ n=4 for photon product processes, make e_q**2 =2/9.,
c                 the average value of charge squared for all types of quarks.
        elseif(n.eq.4) then
          if(j.ne.0.and.j.eq.-l)then                   !qq~->g+gamma
            psbori=16*factgam*(u/t+t/u)/81.
          elseif (j*l.eq.0.and.j+l.ne.0) then          !q(q~)g->q(q~)+gamma
            psbori=2*factgam*(u/s+s/u)/27.
          else
            psbori=0.
          endif
ctp090305 temporary to avoid hard gamma which produce fragmentation problem in psahot
          psbori=0.     !????????????
        elseif(n.eq.5) then
          if(j.ne.0.and.j.eq.-l)then                   !qq~->gamma+gamma
            psbori=4*factgam*(t/u+u/t)/81.
          else
            psbori=0.
          endif
ctp090305 temporary to avoid hard gamma which produce fragmentation problem in psahot
          psbori=0.     !????????????
        endif

      elseif(n.eq.1)then                                            !c-quark

        if(l.eq.0)then                                !cg->cg
          xm=qcmass**2/s/u
          psbori=(s**2+u**2)/t**2+(s/u+u/s)/2.25
     *    -4.*qcmass**2/t+xm*(xm*t**2-t)/.5625+4.*qcmass**2*xm
        else                                          !cq->cq
          psbori=(s**2+u**2)/t**2/2.25-qcmass**2/t/1.125
        endif

      else

        psbori=0.

      endif
      return
      end

c-----------------------------------------------------------------------
      double precision function om51p(sy,xh,yp,b,iqq)
c-----------------------------------------------------------------------
c om5p - chi~(x,y)
c xh - fraction of the energy squared s for the pomeron;
c yp - rapidity for the pomeron;
c b - impact parameter between the pomeron ends;
c iqq =-1  - 0+1+2+3+4,
c iqq = 0  - soft pomeron,
c iqq = 1  - gg,
c iqq = 2  - qg,
c iqq = 3  - gq,
c iqq = 4  - qq,
c iqq = 5  - soft(int)|b,
c iqq = 6  - gg(int)|b,
c iqq = 7  - soft(proj)|b,
c iqq = 8  - gg(proj)|b,
c iqq = 9  - qg(proj)|b,
c iqq = 10 - total fro-uncut integrated,
c iqq = 11 - total uncut integrated,
c iqq = 12 - soft(int),
c iqq = 13 - gg(int),
c iqq = 14 - <b^2*soft(int)>,
c iqq = 15 - <b^2*gg(int)>,
c iqq = 16 - soft(proj-int),
c iqq = 17 - gg(proj-int),
c iqq = 18 - qg(proj-int),
c iqq = 19 - <b^2*soft(proj)>,
c iqq = 20 - <b^2*gg(proj)>,
c iqq = 21 - <b^2*qg(proj)>
c-----------------------------------------------------------------------
      double precision xh,yp!,coefom1,coefom2
      common /psar7/  delx,alam3p,gam3p
      common /psar37/ coefom1,coefom2
      include 'epos.inc'
      include 'epos.incsem'

      xp=dsqrt(xh)*exp(yp)
      if(xh.ne.0.d0)then
        xm=xh/xp
      else
        xm=0.
      endif
      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))
      zb=exp(-b**2/(4.*.0389*rp))
      rh=r2had(iclpro)+r2had(icltar)

      if(iqq.eq.0)then          !soft
c      rp=r2hads(iclpro)+r2hads(icltar)+slopoms*log(max(1.,sy))
      zb=exp(-b**2/(4.*.0389*rp))
        om51p=chad(iclpro)*chad(icltar)*gamhads(iclpro)
     *  *gamhads(icltar)*sy**dels*(xp*xm)**(-alppar)*zb/rp
      elseif(iqq.le.4)then      !gg,qg,gq,qq
        om51p=psvin(sy,xp,xm,zb,iqq)
      elseif(iqq.eq.5)then      !soft(int)|b
c        rh=alam3p+slopoms*log(max(1.,sy))
        om51p=sy**dels*zb**(rp/rh)/rh
      elseif(iqq.eq.6)then      !gg(int)|b
        om51p=psvin(sy,xp,xm,zb,14)
      elseif(iqq.eq.7)then      !soft(proj)b
c        rh=r2hads(iclpro)+.5*alam3p+slopoms*log(max(1.,sy))
        om51p=chad(iclpro)*gamhads(iclpro)*sy**dels
     *  *xp**(-alppar)*zb**(rp/rh)/rh
       elseif(iqq.eq.8)then     !gg(proj)b
        om51p=psvin(sy,xp,xm,zb,16)
       elseif(iqq.eq.9)then     !qg(proj)b
        om51p=psvin(sy,xp,xm,zb,18)
       elseif(iqq.eq.10)then    !total fro-uncut integrated
         om51p=0.d0
         return
       elseif(iqq.eq.11)then    !total uncut integrated
        om51p=psvin(sy,xp,xm,zb,9)
c        om51p=om51p+dble(coefom1)/2.d0*om51p**2+dble(coefom2)/6.d0*om51p**3 !!!!!!!!!!
c        if(om51p.gt.100.d0)om51p=100.d0
      elseif(iqq.eq.12)then      !soft(int)
        om51p=sy**dels*4.*.0389
      elseif(iqq.eq.13)then      !gg(int)
        om51p=psvin(sy,xp,xm,zb,5)
      elseif(iqq.eq.14)then      !<b^2*soft(int)>
c        rh=alam3p+slopoms*log(max(1.,sy))
        om51p=sy**dels*rh*(4.*.0389)**2
      elseif(iqq.eq.15)then      !<b^2*gg(int)>
        om51p=psvin(sy,xp,xm,zb,15)
      elseif(iqq.eq.16)then      !soft(proj-int)
        om51p=chad(iclpro)*gamhads(iclpro)*sy**dels
     *  *xp**(-alppar)*4.*.0389
       elseif(iqq.eq.17)then     !gg(proj-int)
        om51p=psvin(sy,xp,xm,zb,6)
       elseif(iqq.eq.18)then     !qg(proj-int)
        om51p=psvin(sy,xp,xm,zb,7)
      elseif(iqq.eq.19)then      !<b^2*soft(proj)>
c        rh=r2hads(iclpro)+.5*alam3p+slopoms*log(max(1.,sy))
        om51p=chad(iclpro)*gamhads(iclpro)*sy**dels
     *  *xp**(-alppar)*rh*(4.*.0389)**2
       elseif(iqq.eq.20)then     !<b^2*gg(proj)>
        om51p=psvin(sy,xp,xm,zb,17)
       elseif(iqq.eq.21)then     !<b^2*qg(proj)>
        om51p=psvin(sy,xp,xm,zb,19)  
      else
        om51p=0.
        call utstop("Unknown iqq in om51p !&",
     +sizeof("Unknown iqq in om51p !&"))
      endif

      return
      end

cc-----------------------------------------------------------------------
c      double precision function om2p(xh,yp,xprem0,xmrem0,b,iqq)
cc-----------------------------------------------------------------------
cc om2p - chi~(x,y) for cut pomeron
cc xh - fraction of the energy squared s for the pomeron;
cc yp - rapidity for the pomeron;
cc xprem - x+ for the projectile remnant;
cc xmrem - x- for the target remnant;
cc b - impact parameter between the pomeron ends;
cc iqq = 0  - total,
cc iqq = 1  - 1-cut,
cc iqq = 2  - Y+,
cc iqq = -2 - Y-,
cc iqq = 3  - 1-cut(soft),
cc iqq = 4  - 1+(gg),
cc iqq = 5  - 1+(qg),
cc iqq = 6  - 1+(gq),
cc iqq = 7  - 1+(difr)
cc iqq = -7 - 1-(difr)
cc-----------------------------------------------------------------------
c      double precision xh,yp,xprem0,xmrem0
c      include 'epos.inc'
c      include 'epos.incsem'
c
c      om2p=0.d0
c      sy=xh*engy**2
c      xprem=sngl(xprem0)
c      xmrem=sngl(xmrem0)
c      xp=dsqrt(xh)*dexp(yp)
c      if(xh.ne.0.d0)then
c        xm=xh/xp
c      else
c        xm=0.
c      endif
c      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))
c      zb=exp(-b**2/(4.*.0389*rp))
c
c      if(iqq.eq.0)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,2)
c     *  +psvy(xp,xprem,xm,xmrem,b,-2)
c     *  +psvy(xp,xprem,xm,xmrem,b,3)
c     *  +psvy(xp,xprem,xm,xmrem,b,-3)
c     *  +psvy(xp,xprem,xm,xmrem,b,9)
c     *  +psvy(xp,xprem,xm,xmrem,b,-9)
c     *  +psvx(xp,xprem,xm,xmrem,b,1)
c     *  +psvx(xp,xprem,xm,xmrem,b,2)
c     *  +psvx(xp,xprem,xm,xmrem,b,-2)
c     *  +psvx(xp,xprem,xm,xmrem,b,6)
c     *  +psvx(xp,xprem,xm,xmrem,b,-6)
c        om2p=om2p+(chad(iclpro)*chad(icltar)*gamhad(iclpro)
c     *  *gamhad(icltar)*sy**dels*(xp*xm)**(-alppar)*zb/rp
c     *  +psvin(sy,xp,xm,zb,1)+psvin(sy,xp,xm,zb,2)
c     *  +psvin(sy,xp,xm,zb,3)+psvin(sy,xp,xm,zb,4))
c      elseif(iqq.eq.1)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,2)+psvy(xp,xprem,xm,xmrem,b,-2)
c     *  +psvx(xp,xprem,xm,xmrem,b,1)
c      elseif(iqq.eq.2)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,3)
c     *  +psvx(xp,xprem,xm,xmrem,b,2)
c      elseif(iqq.eq.-2)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,-3)
c     *  +psvx(xp,xprem,xm,xmrem,b,-2)
c      elseif(iqq.eq.3)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,4)+psvy(xp,xprem,xm,xmrem,b,-4)
c     *  +psvx(xp,xprem,xm,xmrem,b,3)
c      elseif(iqq.eq.4)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,5)+psvy(xp,xprem,xm,xmrem,b,7)
c     *  +psvy(xp,xprem,xm,xmrem,b,-5)+psvy(xp,xprem,xm,xmrem,b,-7)
c     *  +psvx(xp,xprem,xm,xmrem,b,4)+psvx(xp,xprem,xm,xmrem,b,-4)
c      elseif(iqq.eq.5)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,6)+psvy(xp,xprem,xm,xmrem,b,-8)
c     *  +psvx(xp,xprem,xm,xmrem,b,5)
c      elseif(iqq.eq.6)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,-6)+psvy(xp,xprem,xm,xmrem,b,8)
c     *  +psvx(xp,xprem,xm,xmrem,b,-5)
c      elseif(iqq.eq.7)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,9)
c     *  +psvx(xp,xprem,xm,xmrem,b,6)
c      elseif(iqq.eq.-7)then
c        om2p=psvy(xp,xprem,xm,xmrem,b,-9)
c     *  +psvx(xp,xprem,xm,xmrem,b,-6)
c      else
c        stop'om2p-wrong iqq!!!'
c      endif
c      return
c      end
c
cc-----------------------------------------------------------------------
c      double precision function om3p(xh,yp,xleg,xprem,xmrem,xlrem
c     *,b1,b2,b12,iqq)
cc-----------------------------------------------------------------------
cc om3p - chi~(x,y) for cut pomeron (nuclear effects)
cc xh     - fraction of the energy squared s for the pomeron;
cc yp     - rapidity for the pomeron;
cc xleg   - x for the pomeron leg;
cc xprem  - x+ for the projectile remnant;
cc xmrem  - x- for the target remnant;
cc xlrem  - x for the leg remnant;
cc b1     - impact parameter between the pomeron ends;
cc b2     - impact parameter for the second pomeron end;
cc iqq = 1  - uncut+,
cc iqq = 2  - cut+,
cc iqq = 3  - scr+,
cc iqq = 4  - diffr+,
cc iqq = 5  - uncut-,
cc iqq = 6  - cut-,
cc iqq = 7  - scr-,
cc iqq = 8  - diff-
cc iqq = 9  - uncut-h+,
cc iqq = 10 - uncut-h-,
cc iqq = 11 - uncut-YY+,
cc iqq = 12 - uncut-YY-,
cc-----------------------------------------------------------------------
c      double precision xh,yp,xleg,xprem,xmrem,xlrem
c
c      om3p=0.d0
c      return !!!!!!!!!!!!!!!
cc      if(iqq.ne.1.and.iqq.ne.5.and.iqq.ne.9.and.iqq.ne.10
cc     *.and.iqq.ne.11.and.iqq.ne.12)return
c
cc$$$      xp=dsqrt(xh)*exp(yp)
cc$$$      if(xh.ne.0.d0)then
cc$$$        xm=xh/xp
cc$$$      else
cc$$$        xm=0.d0
cc$$$      endif
cc$$$
cc$$$      return
c      end
c
cc-----------------------------------------------------------------------
c      double precision function om4p(xx1,xx2,xx3,xx4
c     *,b12,b13,b14,b23,b24,b34,iqq)
cc-----------------------------------------------------------------------
cc om4p - chi for 2-leg contributions
cc xx_i - x+- for pomeron ends;
cc b_ij - impact parameter diff. between pomeron ends;
cc iqq = 1   - uncut-H,
cc iqq = 2   - uncut-YY+,
cc iqq = 3   - uncut-YY-
cc-----------------------------------------------------------------------
c      double precision xx1,xx2xx3,xx4
c      om4p=0.d0
c      return
c      end
c
cc------------------------------------------------------------------------
c      function omi5pp(sy,xpp,xpm,z,iqq)   !former psfsh1
cc-----------------------------------------------------------------------
cc omi5pp - integrated semihard interaction eikonal
cc sy - energy squared for the hard interaction,
cc z - impact parameter factor, z=exp(-b**2/rp),
cc iqq - type of the hard interaction:
cc 0  - soft, 1 - gg, 2 - qg, 3 - gq
cc-----------------------------------------------------------------------
c      common /ar3/    x1(7),a1(7)
c      common /ar9/    x9(3),a9(3)
c      include 'epos.inc'
c      include 'epos.incsem'
c      fsy(zsy)=zsy**dels   !*(1.-1./zsy)**betpom
c
c      omi5pp=0.
c      if(iclpro.eq.4.and.iqq.eq.2.or.icltar.eq.4.and.iqq.eq.3)then
c        spmin=4.*q2min+2.*qcmass**2
c      elseif(iqq.ne.0)then
c        spmin=4.*q2min
c      else
c        spmin=0.
c      endif
c      if(sy.le.spmin)return
c
c      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))
c      alpq=(alppar+1.)/2.
c      if(iqq.eq.3)then
c        iclt=iclpro
c        iclp=icltar
c      else
c        iclp=iclpro
c        iclt=icltar
c      endif
c
c      if(iqq.eq.0)then
c        xpmax=(1.-spmin/sy)**(1.+alplea(iclp))
c        do i=1,3
c        do m=1,2
c          xp=1.-(xpmax*(.5+x9(i)*(m-1.5)))**(1./(1.+alplea(iclp)))
c          xmmax=(1.-spmin/sy/xp)**(1.+alplea(iclt))
c          do i1=1,3
c          do m1=1,2
c            xm=1.-(xmmax*(.5+x9(i1)*(m1-1.5)))**(1./(1.+alplea(iclt)))
c
c            sy1=sy*xp*xm
c            rh=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy1))
c            omi5pp=omi5pp+a9(i)*a9(i1)*fsy(sy1)*xmmax*z**(rp/rh)/rh
c     *      *(xp*xm)**(-alppar)
c          enddo
c          enddo
c        enddo
c        enddo
c        omi5pp=omi5pp*xpmax/(1.+alplea(iclp))/(1.+alplea(iclt))
c     *  *chad(iclpro)*chad(icltar)*gamhad(iclpro)*gamhad(icltar)
c     *  *(xpp*xpm)**(1.-alppar)/4.
c        return
c      else
c
c        xmin=(spmin/sy)**(delh-dels)
c        do i=1,3
c        do m=1,2
c          zh=(.5*(1.+xmin-(2*m-3)*x9(i)*(1.-xmin)))**(1./(delh-dels))
c          if(iclpro.eq.4.and.iqq.eq.2.or.icltar.eq.4.and.iqq.eq.3)then
c            call psjti0(zh*sy,sgq,sgqb,4,0)
c            call psjti0(zh*sy,sqq,sqqb,4,1)
c          else
c            call psjti0(zh*sy,sgg,sggb,0,0)
c            call psjti0(zh*sy,sgq,sgqb,0,1)
c            call psjti0(zh*sy,sqq,sqqb,1,1)
c            call psjti0(zh*sy,sqaq,sqaqb,-1,1)
c            call psjti0(zh*sy,sqqp,sqqpb,1,2)
c            sqq=(sqq+sqaq+2.*(naflav-1)*sqqp)/naflav/2.
c          endif
c
c          if(iqq.eq.1)then
c            stg=0.
c            do i1=1,3
c            do m1=1,2
c              xx=.5+x9(i1)*(m1-1.5)
c              xp=zh**xx
c              xm=zh/xp
c
c              xp1max=(1.-xp)**(1.+alplea(iclp))
c              xm1max=(1.-xm)**(1.+alplea(iclt))
c              do i2=1,3
c              do m2=1,2
c                xp1=1.-(xp1max*(.5+x9(i2)*(m2-1.5)))
c     *          **(1./(1.+alplea(iclp)))
c                do i3=1,3
c                do m3=1,2
c                  xm1=1.-(xm1max*(.5+x9(i3)*(m3-1.5)))
c     *            **(1./(1.+alplea(iclt)))
c                  if(xp1.lt.xp.or.xm1.lt.xm)write (*,*)'xp1,xm1,xp,xm'
c     *            ,xp1,xm1,xp,xm
c
c                  rh=r2had(iclpro)+r2had(icltar)+slopom
c     *            *log(xp1*xm1/xp/xm)
c                  glu1=(1.-xp/xp1)**betpom*(1.-glusea)
c                  sea1=EsoftQZero(xp/xp1)*glusea
c                  glu2=(1.-xm/xm1)**betpom*(1.-glusea)
c                  sea2=EsoftQZero(xm/xm1)*glusea
c                  stg=stg+a9(i1)*a9(i2)*a9(i3)*(glu1*glu2*sgg
c     *            +(glu1*sea2+sea1*glu2)*sgq+sea1*sea2*sqq)
c     *            *xp1max*xm1max*(xp1*xm1)**(dels-alppar)
c     *            *z**(rp/rh)/rh
c                enddo
c                enddo
c              enddo
c              enddo
c            enddo
c            enddo
c            omi5pp=omi5pp-a9(i)*log(zh)*stg/zh**delh
c
c          else
c            stq=0.
c            xpmin=zh**(dels+.5)
c            do i1=1,3
c            do m1=1,2
c              xp=(.5*(1.+xpmin-(2*m1-3)*x9(i1)*(1.-xpmin)))
c     *        **(1./(dels+.5))
c              xm=zh/xp
c              if(xp*xpp.lt..99999)then
c                uv1=psdfh4(xp*xpp,q2min,0.,iclp,1)
c                dv1=psdfh4(xp*xpp,q2min,0.,iclp,2)
c                xm1max=(1.-xm)**(1.+alplea(iclt))
c                do i2=1,3
c                do m2=1,2
c                  xm1=1.-(xm1max*(.5+x9(i2)*(m2-1.5)))
c     *            **(1./(1.+alplea(iclt)))
c
c                  rh=r2had(iclpro)+r2had(icltar)+slopom*log(xm1/xm)
c                  glu2=(1.-xm/xm1)**betpom*(1.-glusea)
c                  sea2=EsoftQZero(xm/xm1)*glusea
c                  stq=stq+a9(i1)*a9(i2)*(glu2*sgq+sea2*sqq)*(uv1+dv1)
c     *            *z**(rp/rh)/rh*xm1max*xm1**(dels-alppar)/sqrt(xp)
c     *            *((1.-xp)/(1.-xp*xpp))**(1.-alpq+alplea(iclp))
c                enddo
c                enddo
c              endif
c            enddo
c            enddo
c            stq=stq*(1.-xpmin)
c            omi5pp=omi5pp+a9(i)*stq/zh**delh
c          endif
c        enddo
c        enddo
c      endif
c
c      omi5pp=omi5pp*(1.-xmin)/(delh-dels)
c      if(iqq.eq.1)then
c        omi5pp=omi5pp*chad(iclp)*chad(iclt)*gamhad(iclp)
c     *  *gamhad(iclt)*ffrr**2*(xpp*xpm)**(1.-alppar)
c     *  /(1.+alplea(iclp))/(1.+alplea(iclt))*pi/8.*factk
c      else
c        omi5pp=omi5pp*chad(iclp)*chad(iclt)*ffrr*gamhad(iclt)
c     *  *xpp**(1.-alpq)*xpm**(1.-alppar)/(.5+dels)
c     *  /(1.+alplea(iclt))/16.*factk
c      endif
c      return
c      end
c
c------------------------------------------------------------------------
      function om52pi(sy,xpp,xpm,iqq,je1,je2)   !modified om51pp
c-----------------------------------------------------------------------
c      sy  - energy squared for the hard interaction
c
c      iqq = 0  - sea-sea,
c      iqq = 1  - val-sea,
c      iqq = 2  - sea-val,
c      iqq = 3  - val-val,
c
c      je = emission type
c               0 ... no emissions
c               1 ... emissions
c            else ... all
c
c       already b-averaged  (\int d2b /sigine*10)
c-----------------------------------------------------------------------
      common /ar3/    x1(7),a1(7)
      common /psar7/  delx,alam3p,gam3p
      include 'epos.inc'
      include 'epos.incsem'
      if(iqq.lt.0.or.iqq.gt.3)stop'om52pi: unvalid  iqq'

      om52pi=0.

      ef1=0
      ef2=0
      ef3=0
      ef4=0
      if( je1.ge.1             .and. je2.ge.1)             ef1=1
      if( je1.ge.1             .and.(je2.eq.0.or.je2.eq.2))ef2=1
      if((je1.eq.0.or.je1.eq.2).and. je2.ge.1)             ef3=1
      if((je1.eq.0.or.je1.eq.2).and.(je2.eq.0.or.je2.eq.2))ef4=1

      spmin=4.*q2min
      if(sy.le.spmin)goto999

      if(iqq.eq.1)then
        iclv=iclpro
ctp060829        icls=icltar
      elseif(iqq.eq.2)then
ctp060829        icls=iclpro
        iclv=icltar
      endif

      delss=dels
      if(iqq.eq.3)delss=-0.5
      xmin=spmin/sy
      xmin=xmin**(delh-delss)
      alpq=(alppar+1.)/2.

c numerical integration over zh
      do i=1,7
      do m=1,2
        zh=(.5*(1.+xmin-(2*m-3)*x1(i)*(1.-xmin)))**(1./(delh-delss))
         sgg=  ef1  *pijet(2,q2min,q2min,zh*sy,0,0)
     *   + (ef2+ef3)*pijet(1,q2min,q2min,zh*sy,0,0)
     *   +     ef4  *pijet(0,q2min,q2min,zh*sy,0,0)
         sgq=  ef1  *pijet(2,q2min,q2min,zh*sy,0,1)
     *   + (ef2+ef3)*pijet(1,q2min,q2min,zh*sy,0,1)
     *   +     ef4  *pijet(0,q2min,q2min,zh*sy,0,1)
         sqq=  ef1  *pijet(2,q2min,q2min,zh*sy,1,1)
     *   + (ef2+ef3)*pijet(1,q2min,q2min,zh*sy,1,1)
     *   +     ef4  *pijet(0,q2min,q2min,zh*sy,1,1)
        sqaq=  ef1  *pijet(2,q2min,q2min,zh*sy,-1,1)
     *   + (ef2+ef3)*pijet(1,q2min,q2min,zh*sy,-1,1)
     *   +     ef4  *pijet(0,q2min,q2min,zh*sy,-1,1)
        sqqp=  ef1  *pijet(2,q2min,q2min,zh*sy,1,2)
     *   + (ef2+ef3)*pijet(1,q2min,q2min,zh*sy,1,2)
     *   +     ef4  *pijet(0,q2min,q2min,zh*sy,1,2)
        sqqi=sqq
        sqq=(sqq+sqaq+2.*(naflav-1)*sqqp)/naflav/2.
        if(iqq.eq.0)then
          stg=0.
          do i1=1,7
          do m1=1,2
            xx=.5+x1(i1)*(m1-1.5)
            xp=zh**xx
            xm=zh/xp
            glu1=EsoftGluonTil(xp)
            sea1=EsoftQuarkTil(xp)
            glu2=EsoftGluonTil(xm)
            sea2=EsoftQuarkTil(xm)
            dstg= glu1*glu2*sgg
     *            +(glu1*sea2+sea1*glu2)*sgq   !ccccc
     *              +sea1*sea2*sqq   !ccccc
            stg=stg+a1(i1)*dstg
          enddo
          enddo
          om52pi=om52pi-a1(i)*log(zh)*stg/zh**delh
        elseif(iqq.eq.3)then
          stq=0.  !int^1_(sqrt(z)) dx_p / x_p / sqrt(1-x_p) =int^(tmax)_(0) dt
          tmax=sqrt(1.-sqrt(zh))        !t=ln((1+sqrt(1-x_p))/(1-sqrt(1-x_p)))
          tmax=log((1.+tmax)/(1.-tmax))
          if(tmax.gt.1.e-20)then
          do i1=1,7
          do m1=1,2
            t=tmax*(.5+x1(i1)*(m1-1.5))
            z01=((1.d0-exp(-1.d0*t))/(1.d0+exp(-1.d0*t)))**2
            xp=1.-z01
            xm=zh/xp
            if(xp*xpp.le..9999.and.xm*xpm.le..9999
     *      .or.xm*xpp.le..9999.and.xp*xpm.le..9999)then
              stq=stq+a1(i1)
     *               *(psharg(xp*xpp,xm*xpm,sqqi,sqqp,sqaq)
     *                 +psharg(xm*xpp,xp*xpm,sqqi,sqqp,sqaq))
     *            *max(1e-20,1.-xp)**(.5-alpq)
     *            *max(1e-20,1.-xm)**(-alpq)
     *               *xp**delss*xm**delss
     *        *xpp**alppar/gamhad(iclpro)             ! Eval
     *        *xpm**alppar/gamhad(icltar)             ! Eval
            endif
          enddo
          enddo
          stq=stq*tmax
          endif
          om52pi=om52pi+a1(i)*stq/zh**delh
        elseif(iqq.eq.1.or.iqq.eq.2)then
          stq=0.
          tmax=acos(sqrt(zh))
          do i1=1,7
          do m1=1,2
            t=tmax*(.5+x1(i1)*(m1-1.5))
            xp=cos(t)**2
            xm=zh/xp
            if(xp*xpp.lt..99999)then
              uv1=psdfh4(xp*xpp,q2min,0.,iclv,1)      ! Eval
              dv1=psdfh4(xp*xpp,q2min,0.,iclv,2)      ! Eval
              glu2=EsoftGluonTil(xm)
              sea2=EsoftQuarkTil(xm)
              dstq=0
              if(xp.ne.1.)
     *        dstq=(glu2*sgq+sea2*sqq)*(uv1+dv1)
     *        *(1.-xp*xpp)**(-1.+alpq-alplea(iclv)) ! Eval
     *        *xp**(delss-.5)*(1.-xp)**(-alpq+.5)    ! Eval *sqrt(1-x)/sqrt(x)
     *        *xpp**alppar/gamhad(iclv)             ! Eval
              stq=stq+a1(i1)*dstq
            endif
          enddo
          enddo
          stq=stq*tmax
          om52pi=om52pi+a1(i)*stq/zh**delh
        else
          stop'om52pi: unvalid  iqq (2).            '
        endif
      enddo
      enddo

      om52pi=om52pi*(1.-xmin)/(delh-delss)

      if(iqq.eq.0)then
        om52pi=om52pi/4
      elseif(iqq.eq.3)then
        om52pi=om52pi/4
     *  * utgam1(2.+alplea(iclpro)-alpq)                           ! Eval
     *     /utgam1(1.+alplea(iclpro))/utgam1(1.-alpq)           ! Eval
     *  * utgam1(2.+alplea(icltar)-alpq)                           ! Eval
     *     /utgam1(1.+alplea(icltar))/utgam1(1.-alpq)           ! Eval
     *  /xpp**alpq/xpm**alpq                                       ! Eval
      elseif(iqq.le.2)then
        om52pi=om52pi/2
     *  *utgam1(2.+alplea(iclv)-alpq)/utgam1(1.+alplea(iclv)) ! Eval
     *  /utgam1(1.-alpq)                                      ! Eval
     *  /xpp**alpq                                            ! Eval
      endif

 999  continue
      om52pi=om52pi*factk * .0390   /sigine*10  /2.
       end

c------------------------------------------------------------------------
      function psharg(zh1,zh2,sqq,sqqp,sqaq)
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'

      alpq=(alppar+1.)/2.
      if(zh1.le..9999.and.zh2.le..9999)then
        uv1=psdfh4(zh1,q2min,0.,iclpro,1)
        dv1=psdfh4(zh1,q2min,0.,iclpro,2)
        uv2=psdfh4(zh2,q2min,0.,icltar,1)
        dv2=psdfh4(zh2,q2min,0.,icltar,2)
        if(iclpro.eq.2.and.icltar.eq.2)then       !proton
          fff=sqq*(uv1*uv2+dv1*dv2)+sqqp*(uv1*dv2+dv1*uv2)
        elseif(iclpro.eq.1.or.icltar.eq.1)then   !pion
          fff=sqq*uv1*uv2+sqaq*dv1*dv2+sqqp*(uv1*dv2+dv1*uv2)
        elseif(iclpro.eq.3.or.icltar.eq.3)then   !kaon
          fff=sqq*uv1*uv2+sqqp*(uv1*dv2+dv1*uv2+dv1*dv2)
        elseif(iclpro.eq.4.or.icltar.eq.4)then   !J/psi
          fff=sqq*uv1*(uv2+dv2)
        else
          fff=0.
          call utstop("Projectile not know in psharg !&",
     +sizeof("Projectile not know in psharg !&"))
        endif
        psharg=fff
     *               *(1.-zh1)**(-1.+alpq-alplea(iclpro))
     *               *(1.-zh2)**(-1.+alpq-alplea(icltar))
      else
        psharg=0.
      endif
      return
      end

c------------------------------------------------------------------------
      function om51pp(sy,xpp,z,iqq)   !former psfsh
c-----------------------------------------------------------------------
c om51pp - semihard interaction eikonal
c sy  - energy squared for the hard interaction,
c z   - impact parameter factor, z=exp(-b**2/rp),
c iqq - type of the hard interaction:
c   0 - gg, 1 - qg, 2 - gq, 3 - gg(int), 4 - gg(proj), 5 - qg(proj),
c   6 - gg(int)|b=0, 7 - <b^2*gg(int)>, 8 - gg(proj)|b=0,
c   9 - <b^2*gg(proj)>, 10 - qg(proj)|b=0, 11 - <b^2*qg(proj)>
c-----------------------------------------------------------------------
      common /ar3/    x1(7),a1(7)
      common /psar7/  delx,alam3p,gam3p
      include 'epos.inc'
      include 'epos.incsem'

      om51pp=0.
      if(iqq.eq.0.or.iqq.eq.3.or.iqq.eq.4
     *.or.iqq.eq.6.or.iqq.eq.7.or.iqq.eq.8.or.iqq.eq.9
     *.or.iclpro.ne.4.and.(iqq.eq.1.or.iqq.eq.5
     *.or.iqq.eq.10.or.iqq.eq.11)
     *.or.icltar.ne.4.and.iqq.eq.2)then
        spmin=4.*q2min
      else
        spmin=4.*q2min+2.*qcmass**2
      endif
      if(sy.le.spmin)goto999

      icls=iclpro
      if(iqq.eq.1.or.iqq.eq.5.or.iqq.eq.10.or.iqq.eq.11)then
        iclv=iclpro
        icls=icltar
      elseif(iqq.eq.2)then
        icls=iclpro
        iclv=icltar
      endif

      xmin=spmin/sy
      xmin=xmin**(delh-dels)
      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))
      alpq=(alppar+1.)/2.

c numerical integration over zh
      do i=1,7
      do m=1,2
        zh=(.5*(1.+xmin-(2*m-3)*x1(i)*(1.-xmin)))**(1./
     *  (delh-dels))
        if(iqq.eq.0.or.iqq.eq.3.or.iqq.eq.4
     *  .or.iqq.eq.6.or.iqq.eq.7.or.iqq.eq.8.or.iqq.eq.9
     *  .or.iclpro.ne.4.and.(iqq.eq.1.or.iqq.eq.5
     *  .or.iqq.eq.10.or.iqq.eq.11)
     *  .or.icltar.ne.4.and.iqq.eq.2)then
          call psjti0(zh*sy,sgg,sggb,0,0)  !inclusive (sj) and born (sjb)
          call psjti0(zh*sy,sgq,sgqb,0,1)
          call psjti0(zh*sy,sqq,sqqb,1,1)
          call psjti0(zh*sy,sqaq,sqaqb,-1,1)
          call psjti0(zh*sy,sqqp,sqqpb,1,2)
          sqq=(sqq+sqaq+2.*(naflav-1)*sqqp)/naflav/2.
c...........test.......
c      tgg=   psjet(q2min,q2min,q2min,zh*sy,0,0,0)
c     *   +2*psjet1(q2min,q2min,q2min,zh*sy,0,0,0)
c     *   +  psborn(q2min,q2min,q2min,zh*sy,0,0,0,1)
c      tgq=   psjet(q2min,q2min,q2min,zh*sy,0,1,0)
c     *   +2*psjet1(q2min,q2min,q2min,zh*sy,0,1,0)
c     *   +  psborn(q2min,q2min,q2min,zh*sy,0,1,0,1)
c      tqq=   psjet(q2min,q2min,q2min,zh*sy,1,1,0)
c     *   +2*psjet1(q2min,q2min,q2min,zh*sy,1,1,0)
c     *   +  psborn(q2min,q2min,q2min,zh*sy,1,1,0,1)
c      tqa=   psjet(q2min,q2min,q2min,zh*sy,-1,1,0)
c     *   +2*psjet1(q2min,q2min,q2min,zh*sy,-1,1,0)
c     *   +  psborn(q2min,q2min,q2min,zh*sy,-1,1,0,1)
c      tqqp=  psjet(q2min,q2min,q2min,zh*sy,1,2,0)
c     *   +2*psjet1(q2min,q2min,q2min,zh*sy,1,2,0)
c     *   +  psborn(q2min,q2min,q2min,zh*sy,1,2,0,1)
c      write(6,'(f12.2,3x,2f7.3,2(3x,2f7.3))')
c     * zh*sy,tgg,sgg, tgq,sgq, tqqp,sqqp
c.......................
        else
          call psjti0(zh*sy,sgq,sgqb,4,0)
          call psjti0(zh*sy,sqq,sqqb,4,1)
        endif

        if(iqq.eq.0.or.iqq.eq.3.or.iqq.eq.4
     *  .or.iqq.eq.6.or.iqq.eq.7.or.iqq.eq.8.or.iqq.eq.9)then
          stg=0.
          do i1=1,7
          do m1=1,2
            xx=.5+x1(i1)*(m1-1.5)
            xp=zh**xx
            xm=zh/xp
            glu1=(1.-xp)**betpom*(1.-glusea)
            sea1=EsoftQZero(xp)*glusea
            glu2=(1.-xm)**betpom*(1.-glusea)
            sea2=EsoftQZero(xm)*glusea
            rh=0.
            if(iqq.eq.0)then
              rh=r2had(iclpro)+r2had(icltar)-slopom*log(zh)
            elseif(iqq.eq.3.or.iqq.eq.4)then
              rh=1.
            elseif(iqq.eq.6.or.iqq.eq.7)then
              rh=alam3p-slopom*log(zh)
            elseif(iqq.eq.8.or.iqq.eq.9)then
              rh=r2had(iclpro)+.5*alam3p-slopom*log(zh)
            endif
            dstg=(glu1*glu2*sgg+
     *      (glu1*sea2+sea1*glu2)*sgq+sea1*sea2*sqq)
     *      *z**(rp/rh)/rh
            if(iqq.eq.7.or.iqq.eq.9)dstg=dstg*rh**2
            stg=stg+a1(i1)*dstg
          enddo
          enddo
          om51pp=om51pp-a1(i)*log(zh)*stg/zh**delh
        else
          stq=0.
          tmax=acos(sqrt(zh))
          do i1=1,7
          do m1=1,2
            t=tmax*(.5+x1(i1)*(m1-1.5))
            xp=cos(t)**2
            xm=zh/xp
            if(xp*xpp.lt..99999)then
              uv1=psdfh4(xp*xpp,q2min,0.,iclv,1)
              dv1=psdfh4(xp*xpp,q2min,0.,iclv,2)
              glu2=(1.-xm)**betpom*(1.-glusea)
              sea2=EsoftQZero(xm)*glusea
              rh=0.
              if(iqq.le.2)then
                rh=r2had(iclpro)+r2had(icltar)-slopom*log(xm)
              elseif(iqq.eq.5)then
                rh=1.
              elseif(iqq.le.10.or.iqq.le.11)then
                rh=r2had(iclpro)+.5*alam3p-slopom*log(xm)
              endif
              dstq=0
              if(xp.ne.1.)
     *        dstq=(glu2*sgq+sea2*sqq)*(uv1+dv1)
     *        *z**(rp/rh)/rh
     *        *(1.-xp*xpp)**(-1.+alpq-alplea(iclv))
     *        *xp**(dels-.5)*(1.-xp)**(-alpq+.5)
              if(iqq.eq.11)dstq=dstq*rh**2
              stq=stq+a1(i1)*dstq
            endif
          enddo
          enddo
          stq=stq*tmax
          om51pp=om51pp+a1(i)*stq/zh**delh
        endif
      enddo
      enddo

      om51pp=om51pp*(1.-xmin)/(delh-dels)/sy**delh/2.
      if(iqq.eq.0)then
        om51pp=om51pp*chad(iclpro)*chad(icltar)*gamhad(iclpro)
     *  *gamhad(icltar)*ffrr**2*pi
      elseif(iqq.eq.3)then
        om51pp=om51pp*ffrr**2*pi*4.*.0389
      elseif(iqq.eq.6)then
        om51pp=om51pp*ffrr**2*pi
      elseif(iqq.eq.7)then
        om51pp=om51pp*ffrr**2*pi*(4.*.0389)**2
      elseif(iqq.eq.4.or.iqq.eq.8.or.iqq.eq.9)then
        om51pp=om51pp*ffrr**2*pi*chad(iclpro)*gamhad(iclpro)
        if(iqq.eq.4)om51pp=om51pp*4.*.0389
        if(iqq.eq.9)om51pp=om51pp*(4.*.0389)**2
      elseif(iqq.le.2)then
        om51pp=om51pp*chad(iclpro)*chad(icltar)*ffrr*gamhad(icls)
     *  *utgam1(2.+alplea(iclv)-alpq)/utgam1(1.+alplea(iclv))
     *  /utgam1(1.-alpq)/2./xpp**alpq
      elseif(iqq.eq.5.or.iqq.eq.10.or.iqq.eq.11)then
        om51pp=om51pp*chad(iclv)*ffrr
     *  *utgam1(2.+alplea(iclv)-alpq)/utgam1(1.+alplea(iclv))
     *  /utgam1(1.-alpq)/2./xpp**alpq
        if(iqq.eq.5)om51pp=om51pp*4.*.0389
        if(iqq.eq.11)om51pp=om51pp*(4.*.0389)**2
      endif
 999  continue
      end

c-------------------------------------------------------------------------------
      subroutine epocrossc(niter,gtot,gprod,gabs,gcoh,gqel,gdd)
c-------------------------------------------------------------------------------
c epocrossc - nucleus-nucleus (nucleus-hydrogen) interaction cross sections
c by calculation will real nuclear profiles and eikonal (simplified simulations)
c gtot  - total cross section
c gprod - production cross section (all diffraction included)
c gabs  - cut Pomerons cross section (no diffraction at all)
c gdd   - proj (ionudi=2) or proj or targ (ionudi=0/3) excited diffraction
c         cross section
c gcoh  - coherent (elastic with respect to the projectile) cross section
c      (non excited diff proj if ionudi=2, non excited proj+targ if ionudi=0/3)
c
c Be careful : this function is not symmetric for gdd and gqel (only projectile
c diffraction) in case of ionudi=2.
c (target diffraction is not treated explicitely and contributes to
c gprod, gdd, gcoh and gtot).
c
c WARNING : results are sure only in case of ionudi=1 (no substraction from
c           diffractive part) in particular for AA with A > 10 (nuclear diff
c           not well described). For pA seems to be OK with ionudi 2 and 3.
c
c code from QGSJET programs by S.Ostapchenko
c-------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)
      common/geom/rmproj,rmtarg,bmax,bkmx
      dimension wabs(28),wdd(28),wcoh(28),wprod(28),wqel(28)
     &         ,b0(28),ai(28)
      common /ar3/ x1(7),a1(7)
      double precision xgabs,xgdd,xgprod,xgcoh,xgqel

      call utpri('epocrs',ish,ishini,2)
      if(ish.ge.2)write(ifch,201)niter,bmax
      kollini=koll        !koll modified
      do i=1,7
       b0(15-i)=bmax*sqrt((1.+x1(i))/2.)
       b0(i)=bmax*sqrt((1.-x1(i))/2.)
       ai(i)=a1(i)*bmax**2*pi*5.05        !factor change cs
       ai(15-i)=ai(i)
      enddo
      if(maproj.gt.1.or.matarg.gt.1)then
        difn=max(difnuc(maproj),difnuc(matarg))
      else
        difn=1.
      endif
      do i=1,7
        tp=(1.+x1(i))/2.
        tm=(1.-x1(i))/2.
        b0(14+i)=bmax-log(tp)*difn
        b0(29-i)=bmax-log(tm)*difn
        ai(14+i)=a1(i)*b0(14+i)/tp*10.*difn*pi
        ai(29-i)=a1(i)*b0(29-i)/tm*10.*difn*pi
      enddo
      do i=1,28
       wabs(i)=0.
       wdd(i)=0.
       wprod(i)=0.
       wcoh(i)=0.
       wqel(i)=0.
      enddo
      do nc=1,niter
        if(maproj.eq.1)then
          xproj(1)=0.
          yproj(1)=0.
          zproj(1)=0.
        else
          call conxyz('p',mamx,xproj,yproj,zproj,ypjtl-yhaha)
        endif
        if(matarg.eq.1)then
          xtarg(1)=0.
          ytarg(1)=0.
          ztarg(1)=0.
        else
          call conxyz('t',mamx,xtarg,ytarg,ztarg,yhaha)
        endif

        do i=1,28
          call epogcr(b0(i),xgabs,xgdd,xgprod,xgcoh,xgqel)
          wabs(i)=wabs(i)+sngl(xgabs)
          wdd(i)=wdd(i)+sngl(xgdd)
          wprod(i)=wprod(i)+sngl(xgprod)
          wcoh(i)=wcoh(i)+sngl(xgcoh)
          wqel(i)=wqel(i)+sngl(xgqel)
        enddo
      enddo

      gabs=0.
      gdd=0.
      gcoh=0.
      gprod=0.
      gqel=0.
      do i=1,28
       wabs(i)=wabs(i)/niter
       wdd(i)=wdd(i)/niter
       wcoh(i)=wcoh(i)/niter
       wprod(i)=wprod(i)/niter
       wqel(i)=wqel(i)/niter
       gabs=gabs+ai(i)*wabs(i)
       gdd=gdd+ai(i)*wdd(i)
       gcoh=gcoh+ai(i)*wcoh(i)
       gqel=gqel+ai(i)*wqel(i)
       gprod=gprod+ai(i)*wprod(i)
      enddo


      gtot=gprod+gcoh            !total=all cut (with diff) + all uncut
      if(ish.ge.2)write (ifch,202)gtot,gprod,gabs,gdd,gcoh,gqel

201   format(2x,'epocrossc - A-B interaction cross sections,'
     *,' N of iter.:',i5,' bmax:',f5.2)
202   format(2x,'epocrossc: gtot=',e10.3,2x,'gprod=',e10.3,2x
     *,'gabs=',e10.3/4x,'gdd=',e10.3,2x,'gcoh=',e10.3,'gqel=',e10.3)


      koll=kollini
      call utprix('epocrs',ish,ishini,2)

      return
      end

c-------------------------------------------------------------------------------
      subroutine epogcr(b,gabs,gdd,gprod,gcoh,gqel)
c-------------------------------------------------------------------------------
c epogcr - integrands (b-profiles) for nucleus-nucleus cross sections
c b - impact parameter
c code from QGSJET programs by S.Ostapchenko
c-------------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incpar'
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/scrangle/ phik3(kollmx),thetak3(kollmx)
      double precision vin,gabs,gdd,gprod,gcoh,fdd,gqel,fdt,vdt,vcu

      if(ish.ge.9)write (ifch,201)b
      gprod=1d0
      gabs=1d0
      gdd=1d0
      fdd=1d0
      fdt=1d0
      bx=0
      by=0

      if(maproj.eq.1.and.matarg.eq.1)then
        if(b.gt.bkmx)then
          koll=0
        else
          koll=1
          bk(1)=b
          iproj(1)=1
          itarg(1)=1
          lproj(1)=1
          ltarg(1)=1
          lproj3(1)=1
          ltarg3(1)=1
          kproj3(1,1)=1
          ktarg3(1,1)=1
          kproj(1,1)=1
          ktarg(1,1)=1
        endif
      else
        bx=b
        by=0.
        koll=0
        do i=1,maproj
          lproj(i)=0
          lproj3(i)=0
        enddo
        do j=1,matarg
          ltarg(j)=0
          ltarg3(j)=0
        enddo
        do 12 i=1,maproj
        do 11 j=1,matarg
          bij=sqrt((xproj(i)+bx-xtarg(j))**2+(yproj(i)+by-ytarg(j))**2)
          if(bij.gt.bkmx)goto 11

          koll=koll+1
          if(koll.gt.kollmx)call utstop('epogcr: kollmx too small&',
     +sizeof('epogcr: kollmx too small&'))
          bk(koll)=bij
          bkx(koll)=xproj(i)+bx-xtarg(j)
          bky(koll)=yproj(i)+by-ytarg(j)
          iproj(koll)=i
          itarg(koll)=j
          lproj(i)=lproj(i)+1
          ltarg(j)=ltarg(j)+1
          kproj(i,lproj(i))=koll
          ktarg(j,ltarg(j))=koll
          if(iscreen.ne.0.and.bij.le.bkmxndif)then
            if(zbrmax.gt.0..and.bij.gt.zbcut+zbrmax*rangen())goto 11
            lproj3(i)=lproj3(i)+1
            ltarg3(j)=ltarg3(j)+1
            kproj3(i,lproj3(i))=koll
            ktarg3(j,ltarg3(j))=koll
c define angle for anti-shadowing
            if(abs(bky(koll)).gt.1.e-6)then
              if(abs(bkx(koll)).gt.1.e-6)then
                phik3(koll)=atan(bky(koll)/bkx(koll))
              else
                phik3(koll)=sign(0.5*pi,bky(koll))
              endif
            elseif(bkx(koll).lt.0.)then
              phik3(koll)=pi
            endif
            if(bk(koll).gt.0.)then
              thetak3(koll)=atan(bglaubx/bk(koll))
            else
              thetak3(koll)=0.5*pi
            endif
          endif

 11     continue
 12     continue
      endif
      if(koll.eq.0)then
        gabs=0d0
        gdd=0d0
        gprod=0d0
        gcoh=0d0
        gqel=0d0
        goto 1000
      endif
      if(iscreen.ne.0)call CalcScrPair(b)

      irea=-1
      call GfunParK(irea)
      if(ionudi.eq.0
     &  .and.(maproj.ne.1.or.matarg.ne.1).and.nglevt.eq.0)then
        gabs=0d0
        gdd=0d0
        gprod=0d0
        gcoh=0d0
        gqel=0d0
        goto 1000
      endif
      call integom1(irea)

      do n=1,maproj
       call epov(n,vin,vcu,vdt)
       gprod=gprod*vin
       gabs=gabs*vcu
       fdd=fdd*(1.-rexdif(iclpro))
     &        **(1.+rexres(iclpro)*float(lproj(n)-1))
       fdt=fdt*vdt
      enddo
      gprod=min(gprod,1.d0)
      gcoh=1d0-2d0*sqrt(gprod)+gprod
      gprod=1d0-gprod
      gabs=max(0d0,1d0-gabs)          !cut (no diffraction)
      gdd=max(0d0,gprod-gabs)       !diffractive part
      gqel=0d0
      if(ionudi.eq.2.and.maproj+matarg.gt.2)then
        gqel=fdd*gdd      !quasielastic = diffractive without excited proj.
        if(iLHC.eq.1)gqel=gqel-fdd*fdt*gdd  !DPE counted as inelastic
        gdd=gdd-gqel             !only excited projectile diffraction
      elseif(iLHC.ne.1)then
        gqel=fdd*fdt*gdd !quasielastic = diffractive without excited proj. or targ
        gdd=gdd-gqel     !inelastic part due to excited diffraction
      endif
 1000 continue
      if(ish.ge.9)write (ifch,202)gabs,gdd,gprod,gcoh,gqel,fdd,fdt

201   format(2x,'epogcr-integrands for nucleus-nucleus cross sections,'
     *,' b=',e10.3)
202   format(2x,'epogcr: gabs=',e10.3,2x,'gdd=',e10.3,2x,'gprod=',e10.3
     *,2x,'gcoh=',e10.3,2x,'gqel=',e10.3,2x,'fdd=',e10.3,' fdt=',e10.3)
      return
      end

c=============================================================================
      subroutine epov(n,vin,vcu,vdt)
c epov - eikonal factors for nucleus-nucleus interaction
c (used for cross-section calculation)
c n - projectile nucleon indice
c vin - all uncut pomerons
c vcu - all uncut non diff pomerons
c vdt - non diffractive excitation factor for target
c code from QGSJET programs by S.Ostapchenko
c----------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)
      double precision vvv2,vvv1,dv,vin,vcu,vdt,PhiExpoK,PhiExpoK2

      if(ish.ge.9)write (ifch,201)xproj(n),yproj(n)

      vin=0.d0
      vcu=0.d0
      vvv1=1.d0
      vvv2=1.d0
      dv=1.d0
      do m=1,lproj(n)
        k=kproj(n,m)
        vvv2=vvv2*max(0.d0,PhiExpoK2(k,1.d0,1.d0))
        vvv1=vvv1*max(0.d0,PhiExpoK(k,1.d0,1.d0))
        dv=dv*(1.-rexdif(icltar))
     &        **(1.+rexres(icltar)*float(ltarg(m)-1))
      enddo
      vcu=vvv2
      vin=vvv1                    !exp(-2 * chi)
      vdt=dv

      if(ish.ge.9)write (ifch,202)vin,vcu,vdt
      if(ish.ge.9)write (ifch,203)

201   format(2x,'epov - eikonal factor: nucleon coordinates x=',
     *e10.3,2x,'y=',e10.3)
202   format(2x,'vin=',e10.3,2x,'vcu=',e10.3,2x,'vdt=',e10.3)
203   format(2x,'epov - end')
      return
      end

c------------------------------------------------------------------------
      subroutine psfz(iqq,gz2,b)
c-----------------------------------------------------------------------
c hadron-nucleus cross sections calculation
c b - impact parameter squared
C iqq - 1 = elastic cross section
C       2 = inelastic cross section
c-----------------------------------------------------------------------
      double precision PhiExpo
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incpar'
      common /ar3/ x1(7),a1(7)
      external pttcs,pprcs

      gz2=0.
      e1=exp(-1.)
      if(iomega.eq.2)then      !no dif
        rs=r2had(iclpro)+r2had(icltar)+slopom*log(engy**2)
      else
        rs=r2had(iclpro)+r2had(icltar)+max(slopom,slopoms)*log(engy**2)
     &     +gwidth*(r2had(iclpro)+r2had(icltar))
     &     +bmxdif(iclpro,icltar)/4./0.0389
      endif
      rpom=4.*.0389*rs
      kollini=koll        !koll modified in zzfz
      koll=1
      if(iscreen.ne.0.and.(maproj.gt.1.or.matarg.gt.1))then
        call zzfz(zzp,zzt,kollth,b)
        koll=kollth
      else
        zzp=0.
        zzt=0.
      endif

      do i1=1,7
      do m=1,2
        z=.5+x1(i1)*(m-1.5)
        zv1=exp(-z)
        zv2=(e1*z)
        b1=sqrt(-rpom*log(zv1))
        b2=sqrt(-rpom*log(zv2))

        if(maproj.eq.1.and.matarg.eq.1)then
          cg1=1.
          cg2=1.
        elseif(matarg.eq.1)then
          cg1=ptrot(pprcs,b,b1)
          cg2=ptrot(pprcs,b,b2)
        else
          cg1=ptrot(pttcs,b,b1)
          cg2=ptrot(pttcs,b,b2)
        endif

        vv21=sngl(Phiexpo(zzp,zzt,1.,1.d0,1.d0,engy**2,b1))
        vv22=sngl(Phiexpo(zzp,zzt,1.,1.d0,1.d0,engy**2,b2))
        if(iqq.ne.1)then
          gz2=gz2+a1(i1)*(cg1*(1.-vv21)+cg2*(1.-vv22)/z)
        else
          vv11=sngl(Phiexpo(zzp,zzt,0.5,1.d0,1.d0,engy**2,b1))
          vv12=sngl(Phiexpo(zzp,zzt,0.5,1.d0,1.d0,engy**2,b2))
          gz2=gz2+a1(i1)*(cg1*(vv21-2.*vv11+1.)
     &                   +cg2*(vv22-2.*vv12+1.)/z)
        endif
      enddo
      enddo
      gz2=gz2*rpom*0.5

      koll=kollini

      return
      end

c------------------------------------------------------------------------
      subroutine zzfz(zzp,zzt,kollth,b)
c-----------------------------------------------------------------------
c hadron-nucleus cross sections calculation
c b - impact parameter squared
C xsfct - 0.5 = total cross section
C         1.0 = inelastic cross section
c-----------------------------------------------------------------------
      common /psar50/ zznuc,b2xnuc
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incpar'
      common /ar3/ x1(7),a1(7)
      external  pttcs,pprcs,pttzz,pprzz

      zzp=0.
      zzt=0.
      kollth=1
      if(iscreen.eq.0.or.(maproj.eq.1.and.matarg.eq.1))return

      rs=r2had(iclpro)+r2had(icltar)+slopom*log(engy**2)
      rpom=4.*.0389*rs
      bgl2=2.*rpom*epscrp
      zzpp=epscrw*fscra(engy/egyscr)
c caculate the radius where Z is saturated at epscrx to define the bases
c of nuclear shadowing
      satrad=0.
      if(zzpp.gt.0.)satrad=-bgl2*log(epscrx/zzpp)
      bglx=zbrads*sqrt(max(0.1,satrad))
      fzbrmax=1.
      if(zbrmax.gt.0)fzbrmax=zbrmax
      fzbcut=1.
      if(zbcut.gt.0)fzbcut=zbcut*bglx
      fzbrads=1.
      if(bglx.gt.0)fzbrads=bglx
      fnuc=1.2*fzbcut/fzbrads
      b2xnuc=bgl2+4.*fzbrmax*sqrt(float(maproj*matarg))*fnuc


      e1=exp(-1.)

      colp=0.
      colt=0.
      do i1=1,7
      do m=1,2
        z=.5+x1(i1)*(m-1.5)
        zv1=exp(-z)
        zv2=(e1*z)
        b1=sqrt(-rpom*log(zv1))
        b2=sqrt(-rpom*log(zv2))


        if(maproj.gt.1)then
          cg1=ptrot(pprcs,b,b1)
          cg2=ptrot(pprcs,b,b2)
          colnuc=a1(i1)*(cg1+cg2/z)
          colp=colp+colnuc
          rho=0.05
          zznuc=epscrw*fscro(engy/egyscr,rho)
          zp1=ptrot(pprzz,b,b1)
          zp2=ptrot(pprzz,b,b2)
          zzp=zzp+a1(i1)*(zp1+zp2/z)
        endif
        if(matarg.gt.1)then
          cg1=ptrot(pttcs,b,b1)
          cg2=ptrot(pttcs,b,b2)
          colnuc=a1(i1)*(cg1+cg2/z)
          colt=colt+colnuc
          rho=0.05
          zznuc=epscrw*fscro(engy/egyscr,rho)
          zt1=ptrot(pttzz,b,b1)
          zt2=ptrot(pttzz,b,b2)
          zzt=zzt+a1(i1)*(zt1+zt2/z)
        endif

      enddo
      enddo
      colp=sqrt(colp)
      colt=sqrt(colt)
      if(colp.gt.1.)then
        kollth=nint(max(1.,colp))
        colp=fnuc*log(colp)
        zzp=sqrt(zzp)
        zzp=0.01*zzp*colp*bgl2
c saturation
        zzp=min(zzp,colp*epscrx)
      else
        zzp=0.
      endif
      if(colt.gt.1.)then
        kollth=nint(max(1.,kollth+colt))
        colt=fnuc*log(colt)
        zzt=sqrt(zzt)
        zzt=0.01*zzt*colt*bgl2
c saturation
        zzt=min(zzt,colt*epscrx)
      else
        zzt=0.
      endif
c      zzp=zzp*2.   !correction to have formula=MC
c      zzt=zzt*2.

c      print *,'ici',b,zzp,zzt,kollth,b2xnuc

      return
      end


c------------------------------------------------------------------------
      function ptgau(func,bm,ipt,iqq)
c-----------------------------------------------------------------------
c impact parameter integration for impact parameters <bm -
c for nucleus-nucleus and hadron-nucleus cross-sections calculation
c ipt=1 : projectile, ipt=2 : target
c iqq=1 : elastic xsection, iqq=2 : inelastic cross section
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /ar3/ x1(7),a1(7)
      external func

      ptgau=0.
      do i=1,7
      do m=1,2
        b=bm*sqrt(.5+x1(i)*(m-1.5))
        ptgau=ptgau+func(b,ipt,iqq)*a1(i)
      enddo
      enddo
      ptgau=ptgau*bm**2*pi*.5
      return
      end

c------------------------------------------------------------------------
      function ptgau1(bm,ipt,iqq)
c-----------------------------------------------------------------------
c impact parameter integration for impact parameters >bm -
c for hadron-nucleus cross-sections calculation
c ipt=1 : projectile, ipt=2 : target
c iqq=1 : elastic xsection, iqq=2 : inelastic cross section
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /ar5/    x5(2),a5(2)

      ptgau1=0.
      if(ipt.eq.1)then
        difn=difnuc(maproj)
      else
        difn=difnuc(matarg)
      endif
      do i=1,2
        b=bm+x5(i)*difn
        ptgau1=ptgau1+ptfau(b,ipt,iqq)*a5(i)*exp(x5(i))*b*2.*pi*difn
      enddo
      return
      end
c------------------------------------------------------------------------
      function ptgau2(bm,iqq)
c-----------------------------------------------------------------------
c impact parameter integration for impact parameters >bm -
c for nucleus-nucleus cross-sections calculation
c iqq=1 : elastic xsection, iqq=2 : inelastic cross section
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /ar5/    x5(2),a5(2)

      ptgau2=0.
      difn=difnuc(maproj)+difnuc(matarg)
      do i=1,2
        b=bm+x5(i)*difn
        ptgau2=ptgau2+ptfauAA(b,iqq)*a5(i)*exp(x5(i))*b*2.*pi*difn
      enddo
      return
      end


c------------------------------------------------------------------------
      function ptfau(b,ipt,iqq)
c-----------------------------------------------------------------------
c ptfau - integrands for hadron-nucleus cross-sections calculation
c ipt=1 : projectile, ipt=2 : target
c iqq=1 : elastic xsection, iqq=2 : inelastic cross section
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /psar35/ anorm,anormp

      call psfz(iqq,gz2,b)

      if(ipt.eq.1)then
        ptfau=1.-max(0.,(1.-anormp*gz2))**maproj
      else
        ptfau=1.-max(0.,(1.-anorm*gz2))**matarg
      endif

      return
      end

c------------------------------------------------------------------------
      function ptfauAA(b,iqq)
c-----------------------------------------------------------------------
c ptfau - integrands for hadron-nucleus cross-sections calculation
c iqq=1 : elastic xsection, iqq=2 : inelastic cross section
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /ar3/    x1(7),a1(7)
      common /psar35/ anorm,anormp
      external pprcs

      ptfauAA=0.
      e1=exp(-1.)
      rs=r2had(iclpro)+r2had(icltar)+max(slopom,slopoms)*log(engy**2)
     &     +gwidth*(r2had(iclpro)+r2had(icltar))
     &     +bmxdif(iclpro,icltar)/4./0.0389
      rpom=4.*.0389*rs
      do i1=1,7
      do m=1,2
        z=.5+x1(i1)*(m-1.5)
        zv1=exp(-z)
        zv2=(e1*z)
        b1=sqrt(-rpom*log(zv1))
        b2=sqrt(-rpom*log(zv2))
        call psfz(iqq,gz21,b1)
        call psfz(iqq,gz22,b2)
        ptfau1=max(0.,(1.-anorm*gz21))**matarg
        ptfau2=max(0.,(1.-anorm*gz22))**matarg
        cg1=ptrot(pprcs,b,b1)
        cg2=ptrot(pprcs,b,b2)
        ptfauAA=ptfauAA+a1(i1)*(cg1*(1.-ptfau1)+cg2*(1.-ptfau2)/z)
      enddo
      enddo
      ptfauAA=ptfauAA*rpom/2.
      ptfauAA=1.-max(0.,(1.-anormp*ptfauAA))**maproj

      return
      end

c------------------------------------------------------------------------
      function ptrot(func,s,b)
c-----------------------------------------------------------------------
c convolution of nuclear profile functions (axial angle integration)
c-----------------------------------------------------------------------
      common /ar8/ x2(4),a2
      external func

      ptrot=0.
      do i=1,4
        sb1=b**2+s**2-2.*b*s*(2.*x2(i)-1.)
        sb2=b**2+s**2-2.*b*s*(1.-2.*x2(i))
       ptrot=ptrot+(func(sb1)+func(sb2))
      enddo
      ptrot=ptrot*a2
      return
      end

c------------------------------------------------------------------------
      function pttcs(b0)
c-----------------------------------------------------------------------
c ptt - nuclear profile function value at imp param squared b*difnuc**2
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /psar34/ rrr,rrrm
      common /ar5/    x5(2),a5(2)
      common /ar9/    x9(3),a9(3)

      b=b0/difnuc(matarg)**2
      pttcs=0.
      zm=rrrm**2-b
      if(zm.gt.4.*b)then
        zm=sqrt(zm)
      else
        zm=2.*sqrt(b)
      endif

      do i=1,3
        z1=zm*(1.+x9(i))*0.5
        z2=zm*(1.-x9(i))*0.5
        quq=sqrt(b+z1**2)-rrr
        if (quq.lt.85.)pttcs=pttcs+a9(i)/(1.+exp(quq))
        quq=sqrt(b+z2**2)-rrr
        if (quq.lt.85.)pttcs=pttcs+a9(i)/(1.+exp(quq))
      enddo
      pttcs=pttcs*zm*0.5

      dt=0.
      do i=1,2
        z1=x5(i)+zm
        quq=sqrt(b+z1**2)-rrr-x5(i)
        if (quq.lt.85.)dt=dt+a5(i)/(exp(-x5(i))+exp(quq))
      enddo

      pttcs=pttcs+dt
      return
      end


c------------------------------------------------------------------------
      function pttzz(b0)
c-----------------------------------------------------------------------
c ptt - nuclear Z function value at imp param squared b*difnuc**2
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      common /psar34/ rrr,rrrm
      common /psar50/ zznuc,b2xnuc
      common /ar5/    x5(2),a5(2)
      common /ar9/    x9(3),a9(3)

      pttzz=0.
      b=b0/difnuc(matarg)**2
c      absb=max(1.e-9,sqrt(b0)-zbcut)
      absb=max(1.e-9,sqrt(b0))
      bsq=absb*absb
      zm=rrrm**2-b
      if(zm.gt.4.*b)then
        zm=sqrt(zm)
      else
        zm=2.*sqrt(b)
      endif

      do i=1,3
        z1=zm*(1.+x9(i))*0.5
        z2=zm*(1.-x9(i))*0.5
        quq=sqrt(b+z1**2)-rrr
        if (quq.lt.85.)pttzz=pttzz+a9(i)/(1.+exp(quq))
        quq=sqrt(b+z2**2)-rrr
        if (quq.lt.85.)pttzz=pttzz+a9(i)/(1.+exp(quq))
      enddo
      pttzz=pttzz*zm*0.5

      dt=0.
      do i=1,2
        z1=x5(i)+zm
        quq=sqrt(b+z1**2)-rrr-x5(i)
        if (quq.lt.85.)dt=dt+a5(i)/(exp(-x5(i))+exp(quq))
      enddo

      pttzz=max(0.,(pttzz+dt)-1.)*zznuc*exp(-bsq/2./b2xnuc)

      return
      end

c------------------------------------------------------------------------
      function pprcs(b0)
c-----------------------------------------------------------------------
c ppr - nuclear profile function value at imp param squared b*difnuc**2
c-----------------------------------------------------------------------
      include 'epos.inc'
      common /psar41/ rrrp,rrrmp
      common /ar5/    x5(2),a5(2)
      common /ar9/    x9(3),a9(3)

      b=b0/difnuc(maproj)**2
      pprcs=0.
      zm=rrrmp**2-b
      if(zm.gt.4.*b)then
        zm=sqrt(zm)
      else
        zm=2.*sqrt(b)
      endif

      do i=1,3
        z1=zm*(1.+x9(i))*0.5
        z2=zm*(1.-x9(i))*0.5
        quq=sqrt(b+z1**2)-rrrp
        if (quq.lt.85.)pprcs=pprcs+a9(i)/(1.+exp(quq))
        quq=sqrt(b+z2**2)-rrrp
        if (quq.lt.85.)pprcs=pprcs+a9(i)/(1.+exp(quq))
      enddo
      pprcs=pprcs*zm*0.5

      dt=0.
      do i=1,2
        z1=x5(i)+zm
        quq=sqrt(b+z1**2)-rrrp-x5(i)
        if (quq.lt.85.)dt=dt+a5(i)/(exp(-x5(i))+exp(quq))
      enddo

      pprcs=pprcs+dt
      return
      end

c------------------------------------------------------------------------
      function pprzz(b0)
c-----------------------------------------------------------------------
c ppr - Z nuclear function value at imp param squared b*difnuc**2
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      common /psar41/ rrrp,rrrmp
      common /psar50/ zznuc,b2xnuc
      common /ar5/    x5(2),a5(2)
      common /ar9/    x9(3),a9(3)

      pprzz=0.
      b=b0/difnuc(maproj)**2
c      absb=max(1.e-9,sqrt(b0)-zbcut)
      absb=max(1.e-9,sqrt(b0))
      bsq=absb*absb
      zm=rrrmp**2-b
      if(zm.gt.4.*b)then
        zm=sqrt(zm)
      else
        zm=2.*sqrt(b)
      endif

      do i=1,3
        z1=zm*(1.+x9(i))*0.5
        z2=zm*(1.-x9(i))*0.5
        quq=sqrt(b+z1**2)-rrrp
        if (quq.lt.85.)pprzz=pprzz+a9(i)/(1.+exp(quq))
        quq=sqrt(b+z2**2)-rrrp
        if (quq.lt.85.)pprzz=pprzz+a9(i)/(1.+exp(quq))
      enddo
      pprzz=pprzz*zm*0.5

      dt=0.
      do i=1,2
        z1=x5(i)+zm
        quq=sqrt(b+z1**2)-rrrp-x5(i)
        if (quq.lt.85.)dt=dt+a5(i)/(exp(-x5(i))+exp(quq))
      enddo

      pprzz=max(0.,(pprzz+dt)-1.)*zznuc*exp(-bsq/2./b2xnuc)

      return
      end

c------------------------------------------------------------------------------
      function pscrse(ek,mapr,matg,iqq)
c------------------------------------------------------------------------------
c hadron-nucleus (hadron-proton) and nucl-nucl particle production cross section
c ek     - lab kinetic energy for the interaction
c maproj - projec mass number
c matarg - target mass number
c iqq=1    - ela cross section
c     >2   - ine cross section (2 used for cut (changing iomega), 3 uses table,
c                               4 used for ine without table)
c------------------------------------------------------------------------------
      dimension wk(3),wa(3),wb(3)
      include 'epos.inc'
      common /psar33/ asect(7,4,7),asectn(7,7,7)
      common /psar34/ rrr,rrrm
      common /psar35/ anorm,anormp
      common /psar41/ rrrp,rrrmp
      external ptfau,ptfauAA

      pscrse=0.
      call idmass(1120,amt1)
      call idmass(1220,amt2)
      amtar=0.5*(amt1+amt2)
      if(matg.eq.1)amtar=amt1
      if(mapr.eq.1)then
        call idmass(idproj,ampro)
      else
        ampro=amtar
      endif
      egy=ek+ampro
c      p=sqrt(max(0.,egy**2-ampro**2))
      egy=sqrt( 2*egy*amtar+amtar**2+ampro**2 )

      if(isetcs.le.1.or.iqq.ne.3)then
        maprojsave=maproj
        matargsave=matarg
        engysave=engy
        maproj=mapr
        matarg=matg
        engy=egy
        if(matg.eq.1.and.mapr.eq.1)then
          if(iqq.eq.1)then !sig ela
            call psfz(1,gz2,0.)
          else             !sig ine
            call psfz(2,gz2,0.)
          endif
          gin=gz2*pi*10.
        elseif(mapr.eq.1)then
          rad=radnuc(matg)
          bm=rad+2.
          rrr=rad/difnuc(matg)
          rrrm=rrr+log(9.)
          anorm=1.5/pi/rrr**3/(1.+(pi/rrr)**2)/difnuc(matg)**2
          if(iqq.ne.1)then
            gin=(ptgau(ptfau,bm,2,2)+ptgau1(bm,2,2))*10. !sig ine
          else
            gin=(ptgau(ptfau,bm,2,1)+ptgau1(bm,2,1))*10. !sig ela
          endif
        elseif(matg.eq.1)then
          rad=radnuc(mapr)
          bm=rad+2.
          rrrp=rad/difnuc(mapr)
          rrrmp=rrrp+log(9.)
          anormp=1.5/pi/rrrp**3/(1.+(pi/rrrp)**2)/difnuc(mapr)**2
          if(iqq.ne.1)then
            gin=(ptgau(ptfau,bm,1,2)+ptgau1(bm,1,2))*10. !sig ine
          else
            gin=(ptgau(ptfau,bm,1,1)+ptgau1(bm,1,1))*10. !sig ela
          endif
         else
          rad=radnuc(matg)+1.
          radp=radnuc(mapr)+1.
          bm=rad+radp+2.
          rrr=rad/difnuc(matg)
          rrrm=rrr+log(9.)
          rrrp=radp/difnuc(mapr)
          rrrmp=rrrp+log(9.)
          anorm=1.5/pi/rrr**3/(1.+(pi/rrr)**2)/difnuc(matg)**2
          anormp=1.5/pi/rrrp**3/(1.+(pi/rrrp)**2)/difnuc(mapr)**2
          if(iqq.ne.1)then
            gin=(ptgau(ptfauAA,bm,2,2)+ptgau2(bm,2))*10. !sig ine
          else
            gin=(ptgau(ptfauAA,bm,2,1)+ptgau2(bm,1))*10. !sig ela
          endif
        endif
        pscrse=gin
        maproj=maprojsave
        matarg=matargsave
        engy=engysave
      else
        ye=log10(max(1.,egy/1.5))+1.
        je=min(5,int(ye))

        wk(2)=ye-je
        wk(3)=wk(2)*(wk(2)-1.)*.5
        wk(1)=1.-wk(2)+wk(3)
        wk(2)=wk(2)-2.*wk(3)

        ya=matg
        ya=log(ya)/.69315+1.
        ja=min(int(ya),4)
        wa(2)=ya-ja
        wa(3)=wa(2)*(wa(2)-1.)*.5
        wa(1)=1.-wa(2)+wa(3)
        wa(2)=wa(2)-2.*wa(3)

        if(mapr.eq.1)then

          do i=1,3
            do m=1,3
              pscrse=pscrse+asect(je+i-1,iclpro,ja+m-1)*wk(i)*wa(m)
            enddo
          enddo

        else

          yb=mapr
          yb=log(yb)/.69315+1.
          jb=min(int(yb),4)
          wb(2)=yb-jb
          wb(3)=wb(2)*(wb(2)-1.)*.5
          wb(1)=1.-wb(2)+wb(3)
          wb(2)=wb(2)-2.*wb(3)

          do i=1,3
            do m=1,3
              do n=1,3
            pscrse=pscrse+asectn(je+i-1,jb+n-1,ja+m-1)*wk(i)*wa(m)*wb(n)
              enddo
            enddo
          enddo

        endif

        pscrse=exp(pscrse)
      endif
      return
      end

c------------------------------------------------------------------------------
      function eposcrse(ek,mapro,matar,id)
c------------------------------------------------------------------------------
c inelastic cross section of epos
c (id=0 corresponds to air)
c ek     - kinetic energy for the interaction
c maproj - projec mass number     (1<maproj<64)
c matarg - target mass number     (1<matarg<64)
c------------------------------------------------------------------------------
      include 'epos.inc'

      eposcrse=0.
      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          eposcrse=eposcrse+airwnxs(k)*pscrse(ek,mapro,mt,3)
        enddo
      else
        eposcrse=pscrse(ek,mapro,matar,3)
      endif

      return
      end

c------------------------------------------------------------------------------
      function eposinecrse(ek,mapro,matar,id)
c------------------------------------------------------------------------------
c inelastic cross section of epos not using tabulated xs
c (id=0 corresponds to air)
c ek     - kinetic energy for the interaction
c maproj - projec mass number     (1<maproj<64)
c matarg - target mass number     (1<matarg<64)
c------------------------------------------------------------------------------
      include 'epos.inc'

      eposinecrse=0.
      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          eposinecrse=eposinecrse+airwnxs(k)*pscrse(ek,mapro,mt,4)
        enddo
      else
        eposinecrse=pscrse(ek,mapro,matar,4)
      endif

      return
      end

c------------------------------------------------------------------------------
      function eposelacrse(ek,mapro,matar,id)
c------------------------------------------------------------------------------
c elastic cross section of epos
c (id=0 corresponds to air)
c ek     - kinetic energy for the interaction
c maproj - projec mass number     (1<maproj<64)
c matarg - target mass number     (1<matarg<64)
c------------------------------------------------------------------------------
      include 'epos.inc'

      eposelacrse=0.
      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          eposelacrse=eposelacrse+airwnxs(k)*pscrse(ek,mapro,mt,1)
        enddo
      else
        eposelacrse=pscrse(ek,mapro,matar,1)
      endif

      return
      end


c------------------------------------------------------------------------------
      function eposcutcrse(ek,mapro,matar,id)
c------------------------------------------------------------------------------
c total cross section of epos
c (id=0 corresponds to air)
c ek     - kinetic energy for the interaction
c maproj - projec mass number     (1<maproj<64)
c matarg - target mass number     (1<matarg<64)
c------------------------------------------------------------------------------
      include 'epos.inc'

      eposcutcrse=0.
      iomegasave=iomega
      iomega=2
      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          eposcutcrse=eposcutcrse+airwnxs(k)*pscrse(ek,mapro,mt,2)
        enddo
      else
        eposcutcrse=pscrse(ek,mapro,matar,2)
      endif
      iomega=iomegasave

      return
      end

c------------------------------------------------------------------------------
      subroutine crseaaEpos(sigt,sigi,sigc,sige)
c------------------------------------------------------------------------------
c nucleus-nucleus (hadron) cross section of epos from simplified (realistic)
c simulations
c (id=0 corresponds to air)
c  sigt = sig tot
c  sigi = sig inelastic (cut + projectile diffraction)
c  sigc = sig cut
c  sige = sig elastic (includes target diffraction)
c------------------------------------------------------------------------------
      include 'epos.inc'
      niter=20000
      if(idtarg.eq.0)then
        sigt=0.
        sigc=0.
        sigi=0.
        sige=0.
        sigd=0.
        sigql=0.
        do k=1,3
          matarg=int(airanxs(k))
          call epocrossc(niter,xsigt,xsigi,xsigc,xsige,xsigql,xsigd)
          sigt=sigt+airwnxs(k)*xsigt
          sigi=sigi+airwnxs(k)*xsigi
          sigc=sigc+airwnxs(k)*xsigc
          sige=sige+airwnxs(k)*xsige
          sigd=sigd+airwnxs(k)*xsigd
          sigql=sigql+airwnxs(k)*xsigql
        enddo
      else
        call epocrossc(niter,sigt,sigi,sigc,sige,sigql,sigd)
      endif
      if(ionudi.ne.1)then
        sige=sige+sigql      !add non-excited diffractive projectile to elastic
        sigi=sigi-sigql      !do not count non-excited diffractive projectile in inelastic
        if(maproj+matarg.gt.2)then
          sigc=sigc+sigd*0.95   !for absorbtion cross section remove 5% of the
                                !excited projectile diffractive cross section
                                !which "looks like" non excited (approximation)
        endif
      endif
      end


cc------------------------------------------------------------------------
c      function pshard1(sy,xpp,xpm,z)
cc-----------------------------------------------------------------------
cc pshard - qq-pomeron eikonal
cc sy - energy squared for the pomeron,
cc xpp - lc+ for the pomeron,
cc xpm - lc- for the pomeron
cc-----------------------------------------------------------------------
c      common /ar3/   x1(7),a1(7)
c      common /ar9/   x9(3),a9(3)
c      include 'epos.inc'
c      include 'epos.incsem'
c
c      pshard1=0.
c      if(iclpro.ne.4.and.icltar.ne.4)then
c        spmin=4.*q2min
c      else
c        spmin=4.*q2min+2.*qcmass**2
c      endif
c      if(sy.le.spmin)return
c
c      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))
c      alpq=(alppar+1.)/2.
c      xmin=spmin/sy             !min hard pomeron mass share
c      xminl=xmin**(delh+.5)
c
c      do i=1,3
c      do m=1,2
c        zh=(.5*(1.+xminl-(2*m-3)*x9(i)*(1.-xminl)))**(1./(delh+.5))
c        if(iclpro.ne.4.and.icltar.ne.4)then
c          call psjti0(zh*sy,sqq,sqqb,1,1)
c          call psjti0(zh*sy,sqqp,sqqpb,1,2)
c          call psjti0(zh*sy,sqaq,sqaqb,-1,1)
c        else
c          call psjti0(zh*sy,sqq,sqqb,4,1)
c          sqq=0.
c          sqaq=0.
c        endif
c
c        stq=0.
c        do i1=1,3
c        do m1=1,2
c          xx=.5+x9(i1)*(m1-1.5)
c          xp=zh**xx
c          xm=zh/xp
c          if(xp*xpp.le..9999.and.xm*xpm.le..9999.or.
c     *    xm*xpp.le..9999.and.xp*xpm.le..9999)then
c          stq=stq+a9(i1)*psharf(xp*xpp,xm*xpm,sqq,sqqp,sqaq)
c     *    *(1.-xp)**(1.+alplea(iclpro)-alpq)
c     *    *(1.-xm)**(1.+alplea(icltar)-alpq)
c          endif
c        enddo
c        enddo
c        pshard1=pshard1-a9(i)*stq/zh**(delh+0.5)*log(zh)
c      enddo
c      enddo
c      pshard1=pshard1*(1.-xminl)/(delh+.5)/4.*factk
c     **chad(iclpro)*chad(icltar)*(xpp*xpm)**(1.-alpq)
c     **z**(rp/(r2had(iclpro)+r2had(icltar)))
c     */(8.*pi*(r2had(iclpro)+r2had(icltar)))
c      return
c      end
c
c------------------------------------------------------------------------
      function pshard(sy,xpp,xpm)
c-----------------------------------------------------------------------
c pshard - qq-pomeron eikonal
c sy - energy squared for the pomeron,
c xpp - lc+ for the pomeron,
c xpm - lc- for the pomeron
c-----------------------------------------------------------------------
      double precision z01
      common /ar3/   x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      pshard=0.
      if(iclpro.ne.4.and.icltar.ne.4)then
        spmin=4.*q2min
      else
        spmin=4.*q2min+2.*qcmass**2
      endif
      if(sy.le.spmin)return

      alpq=(alppar+1.)/2.
      xmin=spmin/sy             !min hard pomeron mass share
      xminl=xmin**(delh+.5)

      do i=1,7
      do m=1,2
        zh=(.5*(1.+xminl-(2*m-3)*x1(i)*(1.-xminl)))**(1./(delh+.5))
        if(iclpro.ne.4.and.icltar.ne.4)then
          call psjti0(zh*sy,sqq,sqqb,1,1)
          call psjti0(zh*sy,sqqp,sqqpb,1,2)
          call psjti0(zh*sy,sqaq,sqaqb,-1,1)
        else
          call psjti0(zh*sy,sqq,sqqb,4,1)
          sqqp=0.
          sqaq=0.
        endif

        stq=0.  !int^1_(sqrt(z)) dx_p / x_p / sqrt(1-x_p) =int^(tmax)_(0) dt
        tmax=sqrt(1.-sqrt(zh))        !t=ln((1+sqrt(1-x_p))/(1-sqrt(1-x_p)))
        tmax=log((1.+tmax)/(1.-tmax))
        if(tmax.gt.1.e-20)then
        do i1=1,7
        do m1=1,2
          t=tmax*(.5+x1(i1)*(m1-1.5))
          z01=((1.d0-exp(-1.d0*t))/(1.d0+exp(-1.d0*t)))**2
          xp=1.-z01
          xm=zh/xp
          if(xp*xpp.le..9999.and.xm*xpm.le..9999.or.
     *    xm*xpp.le..9999.and.xp*xpm.le..9999)then
          stq=stq+a1(i1)*(psharf(xp*xpp,xm*xpm,sqq,sqqp,sqaq)+
     *    psharf(xm*xpp,xp*xpm,sqq,sqqp,sqaq))
     *    *z01**(.5-alpq)/(1.-xm)**alpq
          endif
        enddo
        enddo
        stq=stq*tmax
        endif
        pshard=pshard+a1(i)*stq/zh**(delh+0.5)
      enddo
      enddo
      pshard=pshard*(1.-xminl)/(delh+.5)/4.*
     *utgam1(2.+alplea(iclpro)-alpq)/utgam1(1.+alplea(iclpro))/
     *utgam1(1.-alpq)*
     *utgam1(2.+alplea(icltar)-alpq)/utgam1(1.+alplea(icltar))/
     *utgam1(1.-alpq)*
     *chad(iclpro)*chad(icltar)/(8.*pi*(r2had(iclpro)+r2had(icltar)))*
     *(xpp*xpm)**(-alpq)/sy**delh
      return
      end

c------------------------------------------------------------------------
      function psharf(zh1,zh2,sqq,sqqp,sqaq)
c-----------------------------------------------------------------------
      include 'epos.incsem'
      include 'epos.inc'

      alpq=(alppar+1.)/2.
      if(zh1.le..9999.and.zh2.le..9999)then
        uv1=psdfh4(zh1,q2min,0.,iclpro,1)
        dv1=psdfh4(zh1,q2min,0.,iclpro,2)
        uv2=psdfh4(zh2,q2min,0.,icltar,1)
        dv2=psdfh4(zh2,q2min,0.,icltar,2)
        if(iclpro.eq.2.and.icltar.eq.2)then       !proton
          fff=sqq*(uv1*uv2+dv1*dv2)+sqqp*(uv1*dv2+dv1*uv2)
        elseif(iclpro.eq.1.or.icltar.eq.1)then   !pion
          fff=sqq*uv1*uv2+sqaq*dv1*dv2+sqqp*(uv1*dv2+dv1*uv2)
        elseif(iclpro.eq.3.or.icltar.eq.3)then   !kaon
          fff=sqq*uv1*uv2+sqqp*(uv1*dv2+dv1*uv2+dv1*dv2)
        elseif(iclpro.eq.4.or.icltar.eq.4)then   !J/psi
          fff=sqq*uv1*(uv2+dv2)
        else
          fff=0.
          call utstop("Projectile not know in psharg !&",
     +sizeof("Projectile not know in psharg !&"))
        endif
        psharf=fff*(1.-zh1)**(-1.+alpq-alplea(iclpro))*
     *  (1.-zh2)**(-1.+alpq-alplea(icltar))
      else
        psharf=0.
      endif
      return
      end

c------------------------------------------------------------------------
      function psvin(sy,xpp,xpm,z,iqq)
c-----------------------------------------------------------------------
c psvin - contributions to the interaction eikonal
c sy  - energy squared for the hard interaction,
c xpp - lc+ for the sh pomeron,
c xpm - lc- for the sh pomeron,
c z   - impact parameter factor, z=exp(-b**2/4*rp),
c iqq = 1  - gg,
c iqq = 2  - qg,
c iqq = 3  - gq,
c iqq = 4  - qq,
c iqq = 5  - gg(int),
c iqq = 6  - gg(proj),
c iqq = 7  - qg(proj),
c iqq = 9  - total uncut-integrated,
c iqq = 10 - total cut,
c iqq = 14  - gg(int)|b=0,
c iqq = 15  - <b^2*gg(int)>,
c iqq = 16  - gg(proj)|b=0,
c iqq = 17  - <b^2*gg(proj)>,
c iqq = 18  - qg(proj)|b=0,
c iqq = 19  - <b^2*qg(proj)>
c-----------------------------------------------------------------------
      dimension wk(3),wi(3),wj(3),wz(3),fa(3)
      common /psar2/  edmax,epmax
      common /psar4/  fhgg(11,10,8),fhqg(11,10,80)
     *,fhgq(11,10,80),fhqq(11,10,80),fhgg0(11,10),fhgg1(11,10,4)
     *,fhqg1(11,10,40),fhgg01(11),fhgg02(11),fhgg11(11,4)
     *,fhgg12(11,4),fhqg11(11,10,4),fhqg12(11,10,4)
     *,ftoint(11,14,2,2,3)
      common /psar7/  delx,alam3p,gam3p
      include 'epos.inc'
      include 'epos.incsem'

      if(iqq.eq.3)then
        xp=xpm
        xm=xpp
        iclp=icltar
        iclt=iclpro
      else
        xp=xpp
        xm=xpm
        iclp=iclpro
        iclt=icltar
      endif
      rp=r2had(iclpro)+r2had(icltar)+slopom*log(max(1.,sy))

      psvin=0.
      if(iqq.eq.1.or.iqq.eq.5.or.iqq.eq.6.or.iqq.eq.14
     *.or.iqq.eq.15.or.iqq.eq.16.or.iqq.eq.17
     *.or.iclpro.ne.4.and.(iqq.eq.2.or.iqq.eq.7
     *.or.iqq.eq.18.or.iqq.eq.19)
     *.or.icltar.ne.4.and.iqq.eq.3
     *.or.iclpro.ne.4.and.icltar.ne.4)then
        spmin=4.*q2min
      else
        spmin=4.*q2min+2.*qcmass**2
      endif
      if(sy.le.spmin.and.(iqq.le.7.or.iqq.gt.13))return

      if(iqq.le.7.or.iqq.gt.13)then
        yl=log(sy/spmin)/log(epmax/2./spmin)*10.+1
        k=int(yl)
        if(k.gt.9)k=9
        wk(2)=yl-k
        wk(3)=wk(2)*(wk(2)-1.)*.5
        wk(1)=1.-wk(2)+wk(3)
        wk(2)=wk(2)-2.*wk(3)

        if(iqq.ne.4)then  !---------------- not 4 ------------------

          if(iqq.eq.5)then
            if(k.eq.1)then
              psvin=max(0.,exp(fhgg01(k+1))*wk(2)
     *        +exp(fhgg01(k+2))*wk(3))
            else
              psvin=exp(fhgg01(k)*wk(1)+fhgg01(k+1)*wk(2)
     *        +fhgg01(k+2)*wk(3))
            endif
            psvin=psvin*factk*sy**delh
            return

          elseif(iqq.eq.15)then
            if(k.eq.1)then
              psvin=max(0.,exp(fhgg02(k+1))*wk(2)
     *        +exp(fhgg02(k+2))*wk(3))
            else
              psvin=exp(fhgg02(k)*wk(1)+fhgg02(k+1)*wk(2)
     *        +fhgg02(k+2)*wk(3))
            endif
            psvin=psvin*factk*sy**delh
            return

          elseif(iqq.eq.6)then
            if(k.eq.1)then
              psvin=max(0.,exp(fhgg11(k+1,iclpro))*wk(2)
     *        +exp(fhgg11(k+2,iclpro))*wk(3))
            else
              psvin=exp(fhgg11(k,iclpro)*wk(1)+fhgg11(k+1,iclpro)*wk(2)
     *        +fhgg11(k+2,iclpro)*wk(3))
            endif
            psvin=psvin*factk*sy**delh*xp**(-alppar)
            return

          elseif(iqq.eq.17)then
            if(k.eq.1)then
              psvin=max(0.,exp(fhgg12(k+1,iclpro))*wk(2)
     *        +exp(fhgg12(k+2,iclpro))*wk(3))
            else
              psvin=exp(fhgg12(k,iclpro)*wk(1)+fhgg12(k+1,iclpro)*wk(2)
     *        +fhgg12(k+2,iclpro)*wk(3))
            endif
            psvin=psvin*factk*sy**delh*xp**(-alppar)
            return

          elseif(iqq.eq.7.or.iqq.eq.19)then
            if(xp.lt..2)then
              xl=log(10.*xp)/log(2.)+5.
            else
              xl=5.*xp+5.
            endif
            i=int(xl)
            if(i.lt.1)i=1
            if(i.eq.5)i=4
            if(i.gt.8)i=8
            wi(2)=xl-i
            wi(3)=wi(2)*(wi(2)-1.)*.5
            wi(1)=1.-wi(2)+wi(3)
            wi(2)=wi(2)-2.*wi(3)
            do k1=1,3
              fa(k1)=0.
            do i1=1,3
              k2=k+k1-1
              fhhh=0.
              if(iqq.eq.7)then
                fhhh=fhqg11(k2,i+i1-1,iclpro)
              elseif(iqq.eq.19)then
                fhhh=fhqg12(k2,i+i1-1,iclpro)
              endif
              fa(k1)=fa(k1)+fhhh*wi(i1)
            enddo
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*factk*sy**delh
            return
          endif

          jz=int(10.*z)
          if(jz.gt.8)jz=8
          if(jz.lt.1)jz=1
          wz(2)=10.*z-jz
          wz(3)=wz(2)*(wz(2)-1.)*.5
          wz(1)=1.-wz(2)+wz(3)
          wz(2)=wz(2)-2.*wz(3)

          if(iqq.eq.14)then
            do k1=1,3
              k2=k+k1-1
              fa(k1)=fhgg0(k2,jz)*wz(1)+fhgg0(k2,jz+1)
     *        *wz(2)+fhgg0(k2,jz+2)*wz(3)
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*z*factk*sy**delh

          elseif(iqq.eq.16)then
            do k1=1,3
              k2=k+k1-1
              fa(k1)=fhgg1(k2,jz,iclpro)*wz(1)+fhgg1(k2,jz+1,iclpro)
     *        *wz(2)+fhgg1(k2,jz+2,iclpro)*wz(3)
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*z*factk*sy**delh*xp**(-alppar)

          elseif(iqq.eq.18)then
            if(xp.lt..2)then
              xl=log(10.*xp)/log(2.)+5.
            else
              xl=5.*xp+5.
            endif
            i=int(xl)
            if(i.lt.1)i=1
            if(i.eq.5)i=4
            if(i.gt.8)i=8
            wi(2)=xl-i
            wi(3)=wi(2)*(wi(2)-1.)*.5
            wi(1)=1.-wi(2)+wi(3)
            wi(2)=wi(2)-2.*wi(3)
            do k1=1,3
              fa(k1)=0.
            do i1=1,3
            do l1=1,3
              k2=k+k1-1
              l2=jz+l1-1+10*(iclpro-1)
              fhhh=fhqg1(k2,i+i1-1,l2)
              fa(k1)=fa(k1)+fhhh*wi(i1)*wz(l1)
            enddo
            enddo
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*z*factk*sy**delh

          elseif(iqq.eq.1)then   !1111111111111111111111111111111111

            do k1=1,3
              k2=k+k1-1
              iclpt=iclpro+4*(icltar-1)
              fa(k1)=fhgg(k2,jz,iclpt)*wz(1)+fhgg(k2,jz+1,iclpt)
     *        *wz(2)+fhgg(k2,jz+2,iclpt)*wz(3)
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*z*factk*sy**delh*(xp*xm)**(-alppar)

          else  ! 2222222222222222222222 3333333333333333333333 ....

            if(xp.lt..2)then
              xl=log(10.*xp)/log(2.)+5.
            else
              xl=5.*xp+5.
            endif
            i=int(xl)
            if(i.lt.1)i=1
            if(i.eq.5)i=4
            if(i.gt.8)i=8
            wi(2)=xl-i
            wi(3)=wi(2)*(wi(2)-1.)*.5
            wi(1)=1.-wi(2)+wi(3)
            wi(2)=wi(2)-2.*wi(3)
            do k1=1,3
              fa(k1)=0.
            do i1=1,3
            do l1=1,3
              k2=k+k1-1
              fhhh=0.
              if(iqq.eq.2)then
                l2=jz+l1-1+10*(iclpro+4*(icltar-1)-1)
                fhhh=fhqg(k2,i+i1-1,l2)
              elseif(iqq.eq.3)then
                l2=jz+l1-1+10*(iclpro+4*(icltar-1)-1)
                fhhh=fhgq(k2,i+i1-1,l2)
              endif
              fa(k1)=fa(k1)+fhhh*wi(i1)*wz(l1)
            enddo
            enddo
            enddo
            if(k.eq.1)then
              psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
            else
              psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
            endif
            psvin=psvin*xm**(-alppar)*z*factk*sy**delh
          endif

        else ! ------------- 4444444444444444444 -----------------------

          if(xp.lt..2)then
            xl1=log(10.*xp)/log(2.)+5.
          else
            xl1=5.*xp+5.
          endif
          i=max(1,int(xl1))
          if(i.eq.5)i=4
          i=min(8,i)
          wi(2)=xl1-i
          wi(3)=wi(2)*(wi(2)-1.)*.5
          wi(1)=1.-wi(2)+wi(3)
          wi(2)=wi(2)-2.*wi(3)

          if(xm.lt..2)then
            xl2=log(10.*xm)/log(2.)+5.
          else
            xl2=5.*xm+5.
          endif
          j=max(1,int(xl2))
          if(j.eq.5)j=4
          j=min(8,j)
          wj(2)=xl2-j
          wj(3)=wj(2)*(wj(2)-1.)*.5
          wj(1)=1.-wj(2)+wj(3)
          wj(2)=wj(2)-2.*wj(3)

          do k1=1,3
            fa(k1)=0.
          do i1=1,3
          do j1=1,3
            k2=k+k1-1
            j2=j+j1-1+10*(iclp+4*(iclt-1)-1)
            fa(k1)=fa(k1)+fhqq(k2,i+i1-1,j2)*wi(i1)*wj(j1)
          enddo
          enddo
          enddo
          if(k.eq.1)then
            psvin=max(0.,exp(fa(2))*wk(2)+exp(fa(3))*wk(3))
          else
            psvin=exp(fa(1)*wk(1)+fa(2)*wk(2)+fa(3)*wk(3))
          endif
          psvin=psvin*z**(rp/(r2had(iclpro)+r2had(icltar)))*
     *    factk*sy**delh

        endif !--------------------------------------------

        return
      endif

      yl=log(sy)/log(1.e8)*10.+1
      k=max(1,int(yl))
      k=min(k,9)     !?????????????9
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      if(z.gt..1)then
        zz=10.*z+4
      else
        zz=50.*z
      endif
      jz=min(12,int(zz))
      if(jz.eq.0)jz=1
      if(jz.eq.4)jz=3
      wz(2)=zz-jz
      wz(3)=wz(2)*(wz(2)-1.)*.5
      wz(1)=1.-wz(2)+wz(3)
      wz(2)=wz(2)-2.*wz(3)

      if(iqq.eq.9)then
        do k1=1,3
        do l1=1,3
          k2=k+k1-1
          l2=jz+l1-1
          psvin=psvin+ftoint(k2,l2,icdp,icdt,iclp)*wk(k1)*wz(l1)
        enddo
        enddo
        psvin=exp(psvin)*z

      endif
      return
      end

c------------------------------------------------------------------------
      function psbint(q1,q2,qqcut,ss,m1,l1,jdis)
c-----------------------------------------------------------------------
c psbint - born cross-section interpolation
c q1 - virtuality cutoff at current end of the ladder;
c q2 - virtuality cutoff at opposite end of the ladder;
c qqcut - p_t cutoff for the born process;
c s  - total c.m. energy squared for the scattering,
c m1 - parton type at current end of the ladder (0 - g, 1,-1,2,... - q)
c l1 - parton type at opposite end of the ladder (0 - g, 1,-1,2,... - q)
c-----------------------------------------------------------------------
      dimension wi(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar21/ csbor(20,160,2)
      include 'epos.incsem'
      double precision psuds

      psbint=0.
      if(jdis.eq.0)then
        qq=max(q1,q2)
      else
        qq=max(q1/4.,q2)
      endif
      qq=max(qq,qqcut)
      if(iabs(m1).ne.4)then
        q2mass=0.
        if(m1.ne.0.and.m1.eq.l1)then
          m=2
          l=2
        elseif(m1.ne.0.and.m1.eq.-l1)then
          m=3
          l=1
        elseif(m1.ne.0.and.l1.ne.0.and.m1.ne.l1)then
          m=3
          l=2
        else
          m=min(1,iabs(m1))+1
          l=min(1,iabs(l1))+1
        endif
      else
        q2mass=qcmass**2
        m=4
        l=min(1,iabs(l1))+1
      endif
      s=ss-q2mass
      spmin=4.*q2min+q2mass
      s2min=4.*qq+q2mass
      if(s.le.s2min)return

      p1=s/(1.+q2mass/s)
      if(p1.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
      else
        tmin=2.*qq
      endif
      qmax=p1/4.
      tmax=p1/2.

      ml=20*(m-1)+80*(l-1)
      qli=log(qq/q2min)/log(qmax/q2min)*19.+1.
      sl=log(s/spmin)/log(epmax/2./spmin)*19.+1.
      k=int(sl)
      i=int(qli)
      if(k.lt.1)k=1
      if(i.lt.1)i=1
      if(k.gt.18)k=18
      if(i.gt.18)i=18

      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      do i1=1,3
      do k1=1,3
        psbint=psbint+csbor(i+i1-1,k+k1+ml-1,jdis+1)
     *  *wi(i1)*wk(k1)
      enddo
      enddo
      psbint=exp(psbint)*(1./tmin-1./tmax)
      if(jdis.eq.0.and.qq.gt.q1)then
        psbint=psbint*sngl(psuds(qq,m1)/psuds(q1,m1))
      elseif(jdis.eq.1.and.4.*qq.gt.q1)then
        psbint=psbint*sngl(psuds(4.*qq,m1)/psuds(q1,m1))
      endif
      if(qq.gt.q2)psbint=psbint*sngl(psuds(qq,l1)/psuds(q2,l1))
      return
      end

c-----------------------------------------------------------------------
      function psborn(q1,q2,qqcut,s,j,l,jdis,md)
c-----------------------------------------------------------------------
c
c    hard 2->2 parton scattering born cross-section
c       including sudakov on both sides
c
c q1 - virtuality cutoff at current end of the ladder;
c q2 - virtuality cutoff at opposite end of the ladder;
c qqcut - p_t cutoff for the born process;
c s - c.m. energy squared for the scattering;
c j - parton type at current end of the ladder (0 - g, 1,2 etc. - q);
c l - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q).
c-----------------------------------------------------------------------
      common /ar3/   x1(7),a1(7)
      double precision sud0,psbornd,psuds
      include 'epos.inc'
      include 'epos.incsem'

      psborn=0

      if(jdis.eq.0)then
        qq=max(q1,q2)
      else
        qq=max(q1/4.,q2)
      endif
      qq=max(qq,qqcut)
c      if(j.ne.3)then  !kkkkkkkkkk  charm is 3 ???
      if(j.ne.4)then
        j1=j
        q2mass=0.
      else
        j1=4
        q2mass=qcmass**2
      endif
      p1=s/(1.+q2mass/s)
      if(p1.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
      else
        tmin=2.*qq
!        return !tmin=2.*qq   !kkkkkkk !?????????????  tp  why not ?
      endif
      tmax=p1/2.
      sud0=psuds(q1,j1)*psuds(q2,l)

      psbornd=0.d0
      do i=1,7
      do m=1,2
        t=2.*tmin/(1.+tmin/tmax-x1(i)*(2*m-3)
     &  *(1.-tmin/tmax))
        qt=t*(1.-t/p1)
        if(qt.lt..999*qq.and.ish.ge.1)write(ifch,*)'psborn:qt,qq,q1,q2'
     &                                             ,qq,qt,q1,q2

        if(jdis.eq.0)then
          scale=qt
        else
          scale=qt*4.
        endif
        if(j1.eq.0.and.l.eq.0)then
          fb=ffborn(s,t, 1. , 0. , 0. , 0. , 0. )    !gg
        elseif(j1*l.eq.0)then
          fb=ffborn(s,t, 0. , 1. , 0. , 0. , 0.)     !qg
        elseif(j1.eq.l)then
          fb=ffborn(s,t, 0. , 0. , 1. , 0. , 0.)     !qq
        elseif(j1.eq.-l)then
          fb=ffborn(s,t, 0. , 0. , 0. , 1. , 0.)     !qq
        else
          fb=ffborn(s,t, 0. , 0. , 0. , 0. , 1.)     !qq
        endif
        fb=fb*pssalf(qt/qcdlam)**2
        psbornd=psbornd+dble(a1(i)*fb)*dble(t)**2
     &  *psuds(scale,j1)*psuds(qt,l)
      enddo
      enddo
      psbornd=psbornd*dble(2.*pi**3)/dble(s)**2/sud0*2
     *    /2   !CS for parton pair
      if(md.eq.1)psbornd=psbornd*(1./tmin-1./tmax)
      psborn=sngl(psbornd)
      return
      end

c------------------------------------------------------------------------
      function psdgh(s,qq,long)
c-----------------------------------------------------------------------
c psdgh
c s - energy squared for the interaction (hadron-hadron),
c-----------------------------------------------------------------------
      common/ar3/    x1(7),a1(7)
      common /cnsta/ pi,pii,hquer,prom,piom,ainfin
      include 'epos.incsem'
      double precision psuds

      xd=qq/s
      if(long.eq.0)then
        psdgh=(psdfh4(xd,q2min,0.,2,1)/2.25+psdfh4(xd,q2min,0.,2,2)/9.
     *  +psdfh4(xd,q2min,0.,2,3)/9.+
     *  2.*(psdfh4(xd,q2min,0.,2,-1)+psdfh4(xd,q2min,0.,2,-2)+
     *  psdfh4(xd,q2min,0.,2,-3))/4.5)
     *  *sngl(psuds(qq,1)/psuds(q2min,1))*4.*pi**2*alfe/qq
      else
        psdgh=0.
      endif

      dgh=0.
      if(long.eq.0)then
        s2min=qq/(1.-q2ini/qq)
      else
        s2min=4.*max(q2min,qcmass**2)+qq
        s2min=s2min/(1.-4.*q2ini/(s2min-qq))
      endif
      xmin=s2min/s

      if(xmin.lt.1.)then
        do i=1,7          !numerical integration over z1
        do m=1,2
          if(long.eq.0)then
            z1=qq/s+(xmin-qq/s)*((1.-qq/s)/(xmin-qq/s))
     *      **(.5+(m-1.5)*x1(i))
          else
            z1=.5*(1.+xmin+(2*m-3)*x1(i)*(1.-xmin))
          endif
          call psdint(z1*s,qq,sds,sdn,sdb,sdt,sdr,1,long)
          call psdint(z1*s,qq,sdsg,sdng,sdbg,sdtg,sdrg,0,long)
          tu=psdfh4(z1,q2min,0.,2,1)
          td=psdfh4(z1,q2min,0.,2,2)
          ts=psdfh4(z1,q2min,0.,2,3)
          tg=psdfh4(z1,q2min,0.,2,0)
          tsea=2.*(psdfh4(z1,q2min,0.,2,-1)+psdfh4(z1,q2min,0.,2,-2)
     *    +psdfh4(z1,q2min,0.,2,-3))
          gy=sdn*(tu/2.25+td/9.+ts/9.+tsea/4.5)+sdtg*tg/4.5
     *    +sdt*(tu+td+ts+tsea)/4.5
          dgh=dgh+a1(i)*gy*(1.-qq/s/z1)
        enddo
        enddo
        dgh=dgh*log((1.-qq/s)/(xmin-qq/s))*.5
      endif
      psdgh=psdgh+dgh
      return
      end

c------------------------------------------------------------------------
      function psdh(s,qq,iclpro0,long)
c-----------------------------------------------------------------------
c pshard - hard quark-quark interaction cross-section
c s - energy squared for the interaction (hadron-hadron),
c iclpro0 - type of the primary hadron (nucleon)
c-----------------------------------------------------------------------
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'
      include 'epos.inc'
      double precision psuds

      xd=qq/s
      qqs=q2min
      if(long.eq.0.and.(idisco.eq.0.or.idisco.eq.1))then
        psdh=(psdfh4(xd,qqs,0.,iclpro0,1)/2.25+
     *  psdfh4(xd,qqs,0.,iclpro0,2)/9.)
     *  *sngl(psuds(qq,1)/psuds(qqs,1))
     *  *4.*pi**2*alfe/qq
      else
        psdh=0.
      endif

      dh=0.
      if(long.eq.0)then
        s2min=qq/(1.-q2ini/qq)
      else
        s2min=4.*max(q2min,qcmass**2)+qq
        s2min=s2min/(1.-4.*q2ini/(s2min-qq))
      endif
      xmin=s2min/s
      if(xmin.lt.1.)then
        do i=1,7          !numerical integration over z1
        do m=1,2
          if(long.eq.0)then
            z1=qq/s+(xmin-qq/s)*((1.-qq/s)/(xmin-qq/s))
     *      **(.5+(m-1.5)*x1(i))
          else
            z1=.5*(1.+xmin+(2*m-3)*x1(i)*(1.-xmin))
          endif
          call psdint(z1*s,qq,sds,sdn,sdb,sdt,sdr,1,long)
          tu=psdfh4(z1,qqs,0.,iclpro0,1)
          td=psdfh4(z1,qqs,0.,iclpro0,2)
          gy=sdt*(tu+td)/4.5+sdn*(tu/2.25+td/9.)
          if(long.eq.0)then
            gy=gy*(1.-qq/s/z1)
          else
            gy=gy/z1
          endif
          dh=dh+a1(i)*gy
        enddo
        enddo
        if(long.eq.0)then
          dh=dh*log((1.-qq/s)/(xmin-qq/s))*.5
        else
          dh=dh*(1.-xmin)*.5
        endif
      endif
      psdh=psdh+dh
      return
      end

c------------------------------------------------------------------------
      function psdsh(s,qq,iclpro0,dqsh,long)
c-----------------------------------------------------------------------
c psdsh - semihard interaction eikonal
c s - energy squared for the interaction (hadron-hadron),
c iclpro0 - hadron class,
c z - impact parameter factor, z=exp(-b**2/rp),
c iqq - type of the hard interaction (0 - gg, 1 - qg, 2 - gq)
c-----------------------------------------------------------------------
      common /ar3/    x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'
      double precision psuds

      xd=qq/s
      if(long.eq.0.and.(idisco.eq.0.or.idisco.eq.1))then
        dqsh=fzeroSeaZZ(xd,iclpro0)/xd**dels
     *  *ffrr*4.*pi*gamhad(iclpro0)/
     *  4.5*sngl(psuds(qq,1)/psuds(q2min,1))
     *  *4.*pi**2*alfe/qq
      else
        dqsh=0.
      endif

      if(long.eq.0)then
        s2min=qq/(1.-q2ini/qq)
      else
        s2min=qq+4.*max(q2min,qcmass**2)
      endif
      xmin=s2min/s
      xmin=xmin**(delh-dels)
      dsh=0.
      if(xmin.lt.1.)then
c numerical integration over z1
        do i=1,7
        do m=1,2
          z1=(.5*(1.+xmin-(2*m-3)*x1(i)*(1.-xmin)))**(1./
     *    (delh-dels))
          call psdint(z1*s,qq,sdsg,sdng,sdbg,sdtg,sdrg,0,long)
          call psdint(z1*s,qq,sdsq,sdnq,sdbq,sdtq,sdrq,1,long)
          dsh=dsh+a1(i)/z1**delh*(sdtg*fzeroGluZZ(z1,iclpro0)
     *    +(sdtq+sdnq)*fzeroSeaZZ(z1,iclpro0))
        enddo
        enddo
        dsh=dsh*(1.-xmin)/(delh-dels)/2.
      endif
      psdsh=dqsh+dsh*ffrr*4.*pi*gamhad(iclpro0)/4.5  !*ccorr(1,1,iclpro0)
      return
      end

cc------------------------------------------------------------------------
c      function psdsh1(s,qq,iclpro0,dqsh,long)
cc-----------------------------------------------------------------------
cc psdsh - semihard interaction eikonal
cc s - energy squared for the interaction (hadron-hadron),
cc iclpro0 - hadron class,
cc z - impact parameter factor, z=exp(-b**2/rp),
cc iqq - type of the hard interaction (0 - gg, 1 - qg, 2 - gq)
cc-----------------------------------------------------------------------
c      common /ar3/    x1(7),a1(7)
c      include 'epos.inc'
c      include 'epos.incsem'
cc      double precision psuds
c
c      psdsh1=0.       !only for plotting in psaevp : not use any more
c
cc$$$      xd=qq/s
cc$$$      write(ifch,*)'Psdsh1 for xd,qq',xd,qq
cc$$$      if(long.eq.0.and.(idisco.eq.0.or.idisco.eq.1))then
cc$$$        dqsh=psftist(xd)/4.5*sngl(psuds(qq,1)/psuds(q2min,1))
cc$$$     *  *4.*pi**2*alfe/qq
cc$$$      else
cc$$$        dqsh=0.
cc$$$      endif
cc$$$
cc$$$      if(long.eq.0)then
cc$$$        s2min=qq/(1.-q2ini/qq)
cc$$$      else
cc$$$        s2min=qq+4.*max(q2min,qcmass**2)
cc$$$      endif
cc$$$      xmin=s2min/s
cc$$$      xmin=xmin**(delh-dels)
cc$$$      dsh=0.
cc$$$      if(xmin.lt.1.)then
cc$$$c numerical integration over z1
cc$$$        do i=1,7
cc$$$        do m=1,2
cc$$$          z1=(.5*(1.+xmin-(2*m-3)*x1(i)*(1.-xmin)))**(1./
cc$$$     *    (delh-dels))
cc$$$          call psdint(z1*s,qq,sdsg,sdng,sdbg,sdtg,sdrg,0,long)
cc$$$          call psdint(z1*s,qq,sdsq,sdnq,sdbq,sdtq,sdrq,1,long)
cc$$$          dsh=dsh+a1(i)/z1**delh*(sdtg*psftigt(z1)
cc$$$     *    +(sdtq+sdnq)*psftist(z1))*z1**dels
cc$$$        enddo
cc$$$        enddo
cc$$$        dsh=dsh*(1.-xmin)/(delh-dels)/2.
cc$$$      endif
cc$$$      psdsh1=dqsh+dsh/4.5
c      return
c      end
c

c------------------------------------------------------------------------
      function psev0(q1,qq,xx,j)
c-----------------------------------------------------------------------
      double precision xx,psuds,psev00
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'

      psev0=0.
      psev00=0.d0
      do i=1,7
      do m=1,2
        if(j.eq.1)then           !g->q
          qi=2.*q1/(1.+q1/qq+(1.-q1/qq)*(2.*m-3.)*x1(i))
          psev00=psev00+a1(i)*qi*psuds(qi,0)/psuds(qi,1)
     *    /log(qi*(1.d0-xx)/qcdlam)
        else                     !q->g
          qi=(.5*(q1+qq+(q1-qq)*(2.*m-3.)*x1(i)))
          psev00=psev00+a1(i)/qi/psuds(qi,0)*psuds(qi,1)
     *    /log(qi*(1.d0-xx)/qcdlam)
        endif
      enddo
      enddo

      if(j.eq.1)then
        psev00=psev00*(1.d0/q1-1.d0/qq)*psuds(qq,1)/psuds(qq,0)/2.d0
      else
        psev00=psev00*(qq-q1)*psuds(qq,0)/psuds(qq,1)/2.d0
      endif
      psev00=psev00/log(log(qq*(1.d0-xx)/qcdlam)
     &             /log(q1*(1.d0-xx)/qcdlam))
      psev0=sngl(psev00)
      return
      end

c------------------------------------------------------------------------
      function psev(q1,qq,xx,j,l,n)
c------------------------------------------------------------------------
      double precision xx,zmax,zmax1,zmin,zmin1,z,psuds,fk,fq
     &,fz1,fz2
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'

      zmax=1.d0-q2ini/qq
      zmin=xx/zmax
      qmax=qq
      fz1=0.d0
      fz2=0.d0

      if(zmin.lt.zmax)then
      if(zmin.lt..1d0)then
        zmax1=min(.1d0,zmax)
        do i=1,7
        do m=1,2
          if(n.eq.2)then
            z=xx+(zmin-xx)*((zmax1-xx)/(zmin-xx))**(.5+(m-1.5)*x1(i))
          elseif(j.eq.1)then
            z=zmin*(zmax1/zmin)**(.5+(m-1.5)*x1(i))
          else
            z=(.5d0*(zmax1+zmin+(zmax1-zmin)*(2*m-3)*x1(i)))
          endif
          qmin=max(q2ini/(1.d0-xx/z),q2ini/(1.d0-z))
          qmin=max(qmin,q1)

          do k=1,2
            fq=0.d0
            do i1=1,7
            do m1=1,2
              if(n.eq.2)then
                qi=qmin*(qmax/qmin)**(.5+x1(i1)*(m1-1.5))
              else
                qi=(.5*(qmax+qmin+(qmax-qmin)*(2.*m1-3.)*x1(i1)))
              endif

              if(j.eq.3.and.k.eq.1)then
                fk=0.d0
              else
                if(n.eq.2)then
                  fk=dble(psevi0(q1,qi,xx/z,min(2,j),k))
                else
                  fk=dble(psevi(q1,qi,xx/z,j,k)/qi)
                endif
              endif
              qt=qi*(1.d0-z)
              fq=fq+a1(i1)*fk/psuds(qi,l-1)*pssalf(qt/qcdlam)
            enddo
            enddo
            if(n.eq.2)then
              fq=fq*log(qmax/qmin)*(1.d0-xx/z)
            elseif(j.eq.1)then
              fq=fq*(qmax-qmin)
            else
              fq=fq*(qmax-qmin)/z
            endif
            fz1=fz1+a1(i)*fq*psfap(z,k-1,l-1)
          enddo
        enddo
        enddo
        if(n.eq.2)then
          fz1=fz1*log((zmax1-xx)/(zmin-xx))/4.
        elseif(j.eq.1)then
          fz1=fz1*log(zmax1/zmin)/4.
        else
          fz1=fz1*(zmax1-zmin)/4.
        endif
      endif

      if(zmax.gt..1d0)then
        zmin1=max(.1d0,zmin)
        do i=1,7
        do m=1,2
          z=1.d0-(1.d0-zmax)*((1.d0-zmin1)/(1.d0-zmax))**
     *    (.5+x1(i)*(m-1.5))
          qmin=max(q2ini/(1.d0-z),q2ini/(1.d0-xx/z))
          qmin=max(qmin,q1)

          do k=1,2
            fq=0.
            do i1=1,7
            do m1=1,2
              if(n.eq.2)then
                qi=qmin*(qmax/qmin)**(.5+x1(i1)*(m1-1.5))
              else
                qi=(.5*(qmax+qmin+(qmax-qmin)*(2.*m1-3.)*x1(i1)))
              endif

              if(j.eq.3.and.k.eq.1)then
                fk=0.d0
              else
                if(n.eq.2)then
                  fk=dble(psevi0(q1,qi,xx/z,min(2,j),k))
                else
                  fk=dble(psevi(q1,qi,xx/z,j,k)/qi)
                endif
              endif
              qt=qi*(1.d0-z)
              fq=fq+a1(i1)*fk/psuds(qi,l-1)*pssalf(qt/qcdlam)
            enddo
            enddo
            if(n.eq.2)then
              fq=fq*log(qmax/qmin)
            else
              fq=fq*(qmax-qmin)
            endif
            fz2=fz2+a1(i)*fq*psfap(z,k-1,l-1)*(1.d0/z-1.d0)
          enddo
        enddo
        enddo
        fz2=fz2*log((1.d0-zmin1)/(1.d0-zmax))/4.
      endif
      endif
      psev=sngl((fz1+fz2)*psuds(qq,l-1))
      return
      end

c------------------------------------------------------------------------
      function psevi0(q1,qq,xx,m,l)
c------------------------------------------------------------------------
      double precision xx,xmax,psuds
      dimension wi(3),wj(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar31/ evk0(21,21,54)
      include 'epos.inc'
      include 'epos.incsem'

      xmax=1.d0-2.d0*q2ini/epmax
      qmin=max(1.d0*q2min,q2ini/(1.d0-xx))
      qm1=max(q1,qmin)
      if(qq.gt..5001*epmax.and.ish.ge.1)then
        write(ifch,*)'0-extrap.:q1,qq,epmax,xx,m,l:',q1,qq,epmax,xx,m,l
c        stop
      endif
      if(xx.ge.xmax.or.qq.le.1.000*qm1)then
        psevi0=0.
c        write (*,*)'xx,xmax,qq,qm1,qmin,q1',xx,xmax,qq,qm1,qmin,q1
        return
      endif

      if(m.eq.l)then
        psevi0=1.
      else
        if(xx.lt..1d0)then
          yx=log(10.d0*xx)+13.
          k=int(yx)
          if(k.gt.11)k=11
          if(k.lt.1)k=1
        elseif(xx.lt..9d0)then
          yx=10.*xx+12.
          k=int(yx)
          if(k.gt.19)k=19
        else
          yx=log(10.d0*(1.d0-xx))/log(10.d0*(1.d0-xmax))*6.+21
          k=int(yx)
          if(k.gt.25)k=25
        endif
        wk(2)=yx-k
        wk(3)=wk(2)*(wk(2)-1.)*.5
        wk(1)=1.-wk(2)+wk(3)
        wk(2)=wk(2)-2.*wk(3)

        qli=log(qq/qmin)/log(.5*epmax/qmin)*20.+1.
        qlj=log(qm1/qmin)/log(qq/qmin)*20.+1.
        i=int(qli)
        if(i.gt.19)i=19
        if(i.lt.1)i=1
        wi(2)=qli-i
        wi(3)=wi(2)*(wi(2)-1.)*.5
        wi(1)=1.-wi(2)+wi(3)
        wi(2)=wi(2)-2.*wi(3)

        j=int(qlj)
        if(j.lt.1)j=1
        if(j.gt.19)j=19
        wj(2)=qlj-j
        wj(3)=wj(2)*(wj(2)-1.)*.5
        wj(1)=1.-wj(2)+wj(3)
        wj(2)=wj(2)-2.*wj(3)

        psevi0=0.
        do i1=1,3
        do j1=1,3
        do k1=1,3
          psevi0=psevi0+evk0(i+i1-1,j+j1-1,k+k1-1+27*(m-1))
     *    *wi(i1)*wj(j1)*wk(k1)
        enddo
        enddo
        enddo
        psevi0=exp(psevi0)
      endif
      psevi0=psevi0*psfap(xx,m-1,l-1)*log(log(qq*(1.d0-xx)/qcdlam)
     */log(qm1*(1.d0-xx)/qcdlam))*sngl(psuds(qq,m-1)/psuds(q1,m-1))/4.5
      return
      end

c------------------------------------------------------------------------
      function psevi(q1,qq,xx,m,l)
c------------------------------------------------------------------------
c       m l: 1 1 ... gluon -> gluon
c            2 1 ... quark -> gluon
c            1 2 ... gluon -> quark
c            3 2 ... quark -> quark non singlet
c            2 2 ... quark -> quark all
c                             singlet = all - non singlet
c-----------------------------------------------------------------------
      double precision xx,xmax,psuds
      dimension wi(3),wj(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar32/ evk(21,21,135)
      include 'epos.inc'
      include 'epos.incsem'

      psevi=0.
      xmax=1.d0-2.d0*q2ini/epmax
      if(qq.gt..5001*epmax.and.ish.ge.1)then
        write(ifch,*)'1-extrap.:q1,qq,epmax,xx,m,l:',q1,qq,epmax,xx,m,l
c        stop
      endif
      qmin=max(1.d0*q2min,q2ini/(1.d0-xx))
      qm1=max(q1,qmin)
      if(xx.ge.xmax.or.qq.le.1.0001*qm1)then
        return
      endif
      qmin1=max(1.d0*qmin,q2ini/(1.d0-dsqrt(xx)))
      if(qq.le.1.0001*qmin1)then
        psevi=psevi0(q1,qq,xx,min(m,2),l)
        return
      endif

      if(xx.lt..1d0)then
        yx=log(10.d0*xx)+13.
        k=int(yx)
        if(k.gt.11)k=11
        if(k.lt.1)k=1
      elseif(xx.lt..9d0)then
        yx=10.*xx+12.
        k=int(yx)
        if(k.gt.19)k=19
      else
        yx=log(10.d0*(1.d0-xx))/log(10.d0*(1.d0-xmax))*6.+21
        k=int(yx)
        if(k.gt.25)k=25
      endif
      wk(2)=yx-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      qli=log(qq/qmin)/log(.5*epmax/qmin)*20.+1.
      qlj=log(qm1/qmin)/log(qq/qmin)*20.+1.
      i=int(qli)
      if(i.lt.1)i=1
      if(i.gt.19)i=19
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      j=int(qlj)
      if(j.lt.1)j=1
      if(j.gt.19)j=19
      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.)*.5
      wj(1)=1.-wj(2)+wj(3)
      wj(2)=wj(2)-2.*wj(3)

      do i1=1,3
      do j1=1,3
      do k1=1,3
        if(m.eq.3)then
          k2=k+k1-1+108
        else
          k2=k+k1-1+27*(m-1)+54*(l-1)
        endif
        psevi=psevi+evk(i+i1-1,j+j1-1,k2)
     *  *wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
      psevi=exp(psevi)*psfap(xx,m-1,l-1)*log(log(qq*(1.d0-xx)/qcdlam)
     */log(qm1*(1.d0-xx)/qcdlam))/4.5
      if(q1.lt.qm1)psevi=psevi*sngl(psuds(qm1,m-1)/psuds(q1,m-1))
      return
      end

c------------------------------------------------------------------------
      function psjci(q1,s,l1)
c-----------------------------------------------------------------------
c psjci - inclusive ordered ladder cross-section interpolation for c-quark
c q1 - virtuality cutoff at current end of the ladder
c s - total c.m. energy squared for the ladder,
c l1 - parton type at current end of the ladder (0-g, 1,2,etc.-q)
c-----------------------------------------------------------------------
      dimension wi(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar23/ cschar(20,20,2)
      include 'epos.incsem'

      psjci=0.
      q2mass=qcmass**2
      spmin=4.*q2min+q2mass
      qq=q1
      s2min=4.*qq+q2mass
      if(s.le.s2min)return

      smins=s2min/(1.-q2ini/q1)
c      if(s.le.smins)goto 1
      if(s.le.smins.or.qq.le.q2min)goto 1        !??????? ctp070618

      p1=s/(1.+q2mass/s)
      if(p1.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
      else
        tmin=2.*qq
      endif
      tmax=p1/2.
      qmax=p1/4.

      l=min(1,iabs(l1))+1
      qli=log(qq/q2min)/log(qmax/q2min)*19.+1.
      sl=log(s/spmin)/log(epmax/2./spmin)*19.+1.
      k=int(sl)
      i=int(qli)
      if(i.lt.1)i=1
      if(k.gt.18)k=18
      if(i.gt.18)i=18

      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      do i1=1,3
      do k1=1,3
        psjci=psjci+cschar(i+i1-1,k+k1-1,l)*wi(i1)*wk(k1)
      enddo
      enddo
      psjci=exp(psjci)*(1./tmin-1./tmax)
      return
1     psjci=psbint(q2min,q1,0.,s,4,l1,0)
      return
      end

c-----------------------------------------------------------------------
      function psjct(s,l)
c-----------------------------------------------------------------------
c psjct - unordered ladder cross-section for c-quark
c s - c.m. energy squared for the scattering;
c l - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q).
c-----------------------------------------------------------------------
      double precision xx,zmax,qmax,qmin,qi,zmin,fsj,z,s2,sj
      common /ar3/   x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      psjct=0.
      q2mass=qcmass**2
      zmax=dble(s)/(dble(s)+dble(5.*q2mass))
      qmax=zmax**2*dble(q2mass)/(1.d0-zmax)
      qmin=dble(q2min)

      if(qmax.lt.qmin.and.ish.ge.1)write(ifch,*)'psjct:qmin,qmax'
     *                                          ,qmin,qmax
      do i=1,7
      do m=1,2
        qi=2.d0*qmin/(1.d0+qmin/qmax+dble((2*m-3)*x1(i))
     *              *(1.d0-qmin/qmax))
        zmax=(2.d0/(1.d0+dsqrt(1.d0+4.d0*dble(q2mass)/qi)))**delh
        zmin=(5.d0*qi/dble(s))**delh

        fsj=0.d0
        if(zmax.lt.zmin.and.ish.ge.1)write(ifch,*)'psjct:zmin,zmax'
     *                                            ,zmin,zmax
        do i1=1,7
        do m1=1,2
          z=(.5d0*(zmax+zmin+dble((2*m1-3)*x1(i1))
     *      *(zmax-zmin)))**(1./delh)
          s2=z*dble(s)-qi
          xx=z
          sj=dble(psjti(sngl(qi),q2min,sngl(s2),0,l,0)*psfap(xx,1,0))*z
          fsj=fsj+dble(a1(i1))*sj*dble(pssalf(sngl(qi)/qcdlam))/z**delh
        enddo
        enddo
        fsj=fsj*(zmax-zmin)
        psjct=psjct+a1(i)*sngl(fsj*qi)
      enddo
      enddo
      psjct=psjct*sngl(1./qmin-1./qmax)/delh/4.
      return
      end

c------------------------------------------------------------------------
      function psjet1(q1,q2,qqcut,s,j,l,jdis)
c-----------------------------------------------------------------------
c psjet1 - ordered parton ladder cross-section
c q1 - virtuality cutoff at current end of the ladder;
c q2 - virtuality cutoff at opposite end of the ladder;
c qqcut - p_t cutoff for the born process;
c s - c.m. energy squared for the scattering;
c j - parton type at current end of the ladder (0 - g, 1,2 etc. - q);
c l - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q).
c-----------------------------------------------------------------------
      double precision xx,z,qq,xmax,xmin,s2min,smin,p1,q2ms,q2inis,xmin1
     *,sh,qtmin,t,xmax1,fx1,fx2,psuds
      common /ar3/   x1(7),a1(7)
      common /ar9/ x9(3),a9(3)
      include 'epos.inc'
      include 'epos.incsem'

      psjet1=0.
      if(jdis.eq.0)then
        qq=dble(max(q1,q2))
      elseif(jdis.eq.1)then
        qq=dble(max(q1/4.,q2))
      else
        qq=dble(max(q1,q2/4.))
      endif
      qq=max(qq,dble(qqcut))
      if(l.ne.3)then
        q2mass=0.
      else
        q2mass=qcmass**2
      endif
      s2min=dble(q2mass)+4.d0*qq
      if(jdis.eq.0.or.jdis.eq.2)then
        smin=s2min/(1.d0-dble(q2ini)/qq)
      else
        smin=s2min/(1.d0-dble(q2ini)/qq/4.d0)
      endif
      if(dble(s).le.smin)return

      q2ms=dble(q2mass)/dble(s)
      q2inis=dble(q2ini)/dble(s)
      p1=dble(s)/(1.d0+q2ms)


      if(jdis.eq.0.or.jdis.eq.2)then
        xmax=.5d0*(1.d0+q2ms)+dsqrt(.25d0*(1.d0-q2ms)**2-4.d0*q2inis)
      else
        xmax=.5d0*(1.+q2ms)+dsqrt(.25d0*(1.-q2ms)**2-q2inis)
      endif
      xmin=max(1.d0+q2ms-xmax,s2min/dble(s))
      if(xmin.ge.xmax.and.ish.ge.1)then
        write(ifch,*)'jti1,xmin,xmax',xmin,xmax
c        return
      endif

      fx1=0.d0
      fx2=0.d0
      if(xmax.gt..8d0)then
        xmin1=max(xmin,.8d0)
        do i=1,3
        do m=1,2
          z=1.d0-(1.d0-xmax)*((1.d0-xmin1)/(1.d0-xmax))**
     *    (.5d0+dble(x9(i)*(m-1.5)))
          sh=z*dble(s)
          xx=z
          p1=sh/(1.d0+dble(q2mass)/sh)

          if(jdis.eq.0.or.jdis.eq.2)then
            qtmin=max(qq,dble(q2ini)/(1.d0-z))
          else
            qtmin=max(qq,dble(q2ini)/(1.d0-z)/4.d0)
          endif
          tmin=2.d0*dble(qtmin)/(1.d0+dsqrt(1.d0-4.d0*dble(qtmin)/p1))
          tmax=p1/2.d0

          ft=0.
          if(tmin.ge.tmax.and.ish.ge.1)write(ifch,*)'psjet1:tmin,tmax'
     *                                              ,tmin,tmax
          do i1=1,3
          do m1=1,2
            t=2.d0*tmin/(1.d0+tmin/tmax-dble(x9(i1)*(2*m1-3))
     &      *(1.d0-tmin/tmax))
            qt=sngl(t*(1.d0-t/p1))
c            if(qt.lt.qtmin)write (*,*)'psjet1:qt,qq',qt,qq

            if(jdis.eq.0)then
              scale1=qt
              scale2=qt
            elseif(jdis.eq.1)then
              scale1=qt*4.
              scale2=qt
            elseif(jdis.eq.2)then
              scale1=qt
              scale2=qt*4.
            endif
            fb=0.
            do n=1,3
              fb=fb+psjetj(q1,scale1,sngl(t),xx,sngl(sh),j,l,n)
            enddo
            ft=ft+a9(i1)*fb*pssalf(qt/qcdlam)**2*sngl(t**2
     *      *psuds(scale2,l))
          enddo
          enddo
          fx1=fx1+dble(a9(i)*ft)*(1.d0/tmin-1.d0/tmax)/sh**2*(1.d0-z)
        enddo
        enddo
        fx1=fx1*dlog((1.d0-xmin1)/(1.d0-xmax))
      endif

      if(xmin.lt..8d0)then
        xmax1=min(xmax,.8d0)**(-delh)
        xmin1=xmin**(-delh)
        do i=1,3
        do m=1,2
          z=(.5d0*(xmax1+xmin1+(xmin1-xmax1)*dble((2*m-3)*x9(i))))
     *    **(-1./delh)
          sh=z*dble(s)
          xx=z
          p1=sh/(1.d0+dble(q2mass)/sh)

          if(jdis.eq.0.or.jdis.eq.2)then
            qtmin=max(qq,dble(q2ini)/(1.d0-z))
          else
            qtmin=max(qq,dble(q2ini)/(1.d0-z)/4.d0)
          endif
          tmin=2.d0*dble(qtmin)/(1.d0+dsqrt(1.d0-4.d0*dble(qtmin)/p1))
          tmax=p1/2.d0

          ft=0.
          if(tmin.ge.tmax.and.ish.ge.1)write(ifch,*)'psjet1:tmin,tmax'
     &                                              ,tmin,tmax
          do i1=1,3
          do m1=1,2
            t=2.d0*tmin/(1.d0+tmin/tmax-dble(x9(i1)*(2*m1-3))
     &      *(1.d0-tmin/tmax))
            qt=sngl(t*(1.d0-t/p1))
          if(qt.lt.sngl(qtmin).and.ish.ge.1)write(ifch,*)'psjet1:qt,qq'
     &                                               ,qt,qq

            if(jdis.eq.0)then
              scale1=qt
              scale2=qt
            elseif(jdis.eq.1)then
              scale1=qt*4.
              scale2=qt
            elseif(jdis.eq.2)then
              scale1=qt
              scale2=qt*4.
            endif
            fb=0.
            do n=1,3
              fb=fb+psjetj(q1,scale1,sngl(t),xx,sngl(sh),j,l,n)
            enddo
            ft=ft+a9(i1)*fb*pssalf(qt/qcdlam)**2*sngl(t**2
     *      *psuds(scale2,l))
          enddo
          enddo
        fx2=fx2+dble(a9(i)*ft)*(1.d0/tmin-1.d0/tmax)/sh**2*z**(1.+delh)
        enddo
        enddo
        fx2=fx2*(xmin1-xmax1)/dble(delh)
      endif
      psjet1=sngl((fx1+fx2)/psuds(q2,l))*pi**3*2
     *    /2    !CS for parton pair
      return
      end

c-----------------------------------------------------------------------
      function psjet(q1,q2,qqcut,s,j,l,jdis)
c-----------------------------------------------------------------------
c     parton ladder cross-section
c     with at least one emission on each side
c
c q1 - virtuality cutoff at current end of the ladder;
c q2 - virtuality cutoff at opposite end of the ladder;
c qqcut - p_t cutoff for the born process;
c s - c.m. energy squared for the scattering;
c j - parton type at current end of the ladder (0 - g, 1,2 etc. - q);
c l - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q).
c-----------------------------------------------------------------------
      double precision xx1,xx2,qq,s2min,xmin,xmax,xmin1,xmax1,t,tmin
     *,tmax,sh,z,qtmin,ft,fx1,fx2
      common /ar3/   x1(7),a1(7)
      common /ar9/ x9(3),a9(3)
      include 'epos.inc'
      include 'epos.incsem'
      common/ccctest/iiitest
      iiitest=0

      psjet=0.
      if(jdis.eq.0)then
        qq=dble(max(q1,q2))
      else
        qq=dble(max(q1/4.,q2))
      endif
      qq=max(qq,dble(qqcut))
      s2min=4.d0*qq
      if(dble(s).le.s2min/(1.d0-dble(q2ini)/qq)**2)return   !kkkkkkk

      phi=acos(1.-54.*q2ini/s)/3.
      zmax=(1.+2.*cos(phi))**2/9.                 !kkkkkkk
      zmin=(1.-cos(phi)+sqrt(3.d0)*sin(phi))/3.   !kkkkkkk
      zmin=max(zmin**2,sngl(s2min/dble(s)))
      if(zmin.gt.zmax.and.ish.ge.1)write(ifch,*)'psjet:zmin,zmax'
     *                                           ,zmin,zmax
      zmin=zmin**(-delh)
      zmax=zmax**(-delh)
      do i=1,3
      do m=1,2
        z=dble(.5*(zmax+zmin+(zmin-zmax)*(2*m-3)*x9(i)))**(-1./delh)
        xmin=dsqrt(z)
        sh=z*dble(s)

        qtmin=max(qq,dble(q2ini)/(1.d0-dsqrt(z)))
        tmin=max(0.d0,1.d0-4.d0*qtmin/sh)
        tmin=2.d0*qtmin/(1.d0+dsqrt(tmin))         !kkkkkkk
        tmax=sh/2.d0

        ft=0.d0
c        if(tmin.gt.tmax)write (*,*)'psjet:tmin,tmax',tmin,tmax
        do i1=1,3
        do m1=1,2
          t=2.d0*tmin/(1.d0+tmin/tmax-dble(x9(i1)*(2*m1-3))
     &    *(1.d0-tmin/tmax))
          qt=t*(1.d0-t/sh)
c          if(qt.lt.qtmin)write (*,*)'psjet:qt,qq',qt,qq
          xmax=1.d0-q2ini/qt
          xmin=max(dsqrt(z),z/xmax)   !xm>xp !!!
          if(xmin.gt.xmax.and.ish.ge.1)write(ifch,*)'psjet:xmin,xmax'
     *                                              ,xmin,xmax
          fx1=0.d0
          fx2=0.d0
          if(xmax.gt..8d0)then
            xmin1=max(xmin,.8d0)
            do i2=1,3
            do m2=1,2
              xx1=1.d0-(1.d0-xmax)*((1.d0-xmin1)/(1.d0-xmax))**
     *        dble(.5+x9(i2)*(m2-1.5))
              xx2=z/xx1

              fb=0.
                fb=fb+psjeti(q1,q2,qt,sngl(t),xx1,xx2,sngl(sh)
     *                       ,j,l,jdis)
     *          +psjeti(q1,q2,qt,sngl(t),xx2,xx1,sngl(sh)
     *                       ,j,l,jdis)
              fx1=fx1+dble(a9(i2)*fb)*(1.d0/xx1-1.d0)
     *                               *pssalf(qt/qcdlam)**2
            enddo
            enddo
            fx1=fx1*dlog((1.d0-xmin1)/(1.d0-xmax))
          endif
          if(xmin.lt..8d0)then
            xmax1=min(xmax,.8d0)
            do i2=1,3
            do m2=1,2
              xx1=xmin*(xmax1/xmin)**dble(.5+x9(i2)*(m2-1.5))
              xx2=z/xx1

              fb=0.
                fb=fb+psjeti(q1,q2,qt,sngl(t),xx1,xx2,sngl(sh)
     *                       ,j,l,jdis)
     *          +psjeti(q1,q2,qt,sngl(t),xx2,xx1,sngl(sh)
     *                       ,j,l,jdis)
              fx2=fx2+dble(a9(i2))*fb*pssalf(qt/qcdlam)**2
            enddo
            enddo
            fx2=fx2*dlog(xmax1/xmin)
          endif
          ft=ft+dble(a9(i1))*(fx1+fx2)*t**2
        enddo
        enddo
        ft=ft*(1.d0/tmin-1.d0/tmax)
        psjet=psjet+a9(i)*sngl(ft*z**(1.+delh)/sh**2)
      enddo
      enddo
      psjet=psjet*(zmin-zmax)/delh*pi**3
     *         /2.    !CS for parton pair
      return
      end

c-----------------------------------------------------------------------
      function pijet(ii,qi,qq,sk,m1,l1) !polynomial interpol of jet CS
c-----------------------------------------------------------------------
c  ii ..... type of CS (2 = bothside, 1 = oneside, 0 = no emission, Born)
c  qi ..... virtuality cutoff at current end of the ladder
c  qq ..... virtuality cutoff of Born
c  sk ..... energy squared for the scattering
c  m1,l1 .. parton types
c-----------------------------------------------------------------------
      include 'epos.incsem'
      common/psar2/edmax,epmax
      common/tabcsjet/ksmax,iqmax,jqmax,csjet(0:2,2,20,20,20,3,2)
      real wi(3),wj(3),wk(3)
      common/cpijet/npijet
      data npijet/0/
      npijet=npijet+1
      if(npijet.eq.1)call MakeCSTable

      if(m1.ne.0.and.m1.eq.l1)then
        m=2
        l=2
      elseif(m1.ne.0.and.m1.eq.-l1)then
        m=3
        l=1
      elseif(m1.ne.0.and.l1.ne.0.and.m1.ne.l1)then
        m=3
        l=2
      else
        m=min(1,iabs(m1))+1
        l=min(1,iabs(l1))+1
      endif

      qqmin=min(qi,qq)
      qmax=sk/4.
      spmin=4.*q2min
      spmed=spmin*(epmax/2./spmin)**(1./(ksmax-1.))
      if(sk.le.spmed)then
        kk=2
        spmax=spmed
      else
        kk=1
        spmax=epmax/2.
      endif

      qli=1.+log(qi/q2min)/log(qmax/q2min)*(iqmax-1)
      qlj=1.+log(qq/qqmin)/log(qmax/qqmin)*(jqmax-1)
      sl= 1.+log(sk/spmin)/log(spmax/spmin)*(ksmax-1)
      k=int(sl)
      i=int(qli)
      j=int(qlj)
      if(k.lt.1)k=1
      if(j.lt.1)j=1
      if(i.lt.1)i=1
      if(k.gt.(ksmax-2))k=ksmax-2
      if(i.gt.(iqmax-2))i=iqmax-2
      if(j.gt.(jqmax-2))j=jqmax-2

      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.)*.5
      wj(1)=1.-wj(2)+wj(3)
      wj(2)=wj(2)-2.*wj(3)

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      pijet=0
      do i1=1,3
      do j1=1,3
      do k1=1,3
        pijet=pijet+csjet(ii,kk,k+k1-1,i+i1-1,j+j1-1,m,l)
     *  *wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
          ! if(ii.eq.2)print*,' '
          ! write(*,'(i2,f6.0,i2,3x,3(2f5.2,2x),f5.2)')
          !*  ii,sk,k,(wk(kk1),csjet(ii,kk,k+kk1-1,1,1,m,l),kk1=1,3) ,pijet
      end

c-----------------------------------------------------------------------
      subroutine MakeCSTable     !tabulates psjet
c-----------------------------------------------------------------------
c   last two indices of table: parton types
c        1 1 ... gg
c        1 2 ... gq
c        2 1 ... qg
c        2 2 ... qq
c        3 1 ... qa
c        3 2 ... qq'
c-----------------------------------------------------------------------
      include 'epos.incsem'
      common/psar2/edmax,epmax
      common/tabcsjet/ksmax,iqmax,jqmax,csjet(0:2,2,20,20,20,3,2)
      write (*,'(a,$)')'(CS table'
      ksmax=10
      iqmax=3
      jqmax=3
      spmin=4.*q2min
      do kk=1,2
       if(kk.eq.1)then
         spmax=epmax/2.
       else               !if(kk.eq.2)
         spmax=spmin*(epmax/2./spmin)**(1./(ksmax-1.))
       endif
       do m=1,3                 !parton type at upper end of the ladder
        write (*,'(a,$)')'.'
         do l=1,2              !parton type at lower end of the ladder
         m1=m-1
         l1=l-1
         if(m.eq.3.and.l.eq.1)l1=-m1
        do k=1,ksmax
          sk=spmin*(spmax/spmin)**((k-1.)/(ksmax-1.))
          qmax=sk/4.
          do i=1,iqmax
           qi=q2min*(qmax/q2min)**((i-1.)/(iqmax-1.))
           do j=1,jqmax
            qq=qi*(qmax/qi)**((j-1.)/(jqmax-1.))
                !write(*,'(i3,4f8.3,2i4,$)')j, qi,q2min,qq,sk,m1,l1
            csjet(2,kk,k,i,j,m,l)= psjet(qi,q2min,qq,sk,m1,l1,0)
            csjet(1,kk,k,i,j,m,l)=psjet1(qi,q2min,qq,sk,m1,l1,0)
            csjet(0,kk,k,i,j,m,l)=psborn(qi,q2min,qq,sk,m1,l1,0,1)
       !   if(i.eq.1.and.j.eq.1.and.m.eq.1.and.l.eq.1)
       ! *write(*,'(2f8.2,f13.2,2i3,3x,i3,3f8.3)')
       ! * qi,qq,sk,m1,l1,k,csjet(2,kk,k,i,j,m,l)
       ! *             ,csjet(1,kk,k,i,j,m,l),csjet(0,kk,k,i,j,m,l)
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
      write (*,'(a,$)')'done)'
      end

c-----------------------------------------------------------------------
      function psjeti(q1,q2,qt,t,xx1,xx2,s,j,l,jdis)
c-----------------------------------------------------------------------
c
c      E~qcd_ji * E~qcd_lk * B_ik
c
c        B_ik = psbori = contribution to Born xsection:
c                         dsigmaBorn/d2pt/dy
c                         = s/pi * delta(s+t+u) * 2*pi*alpha**2 /s**2 * B_ik
c
c        E~qcd: at least one emission
c
c q1  - virtuality cutoff at current end of the ladder
c q2  - virtuality cutoff at opposite end of the ladder
c xx1 - feinman x for the first parton for the born process
c xx2 - feinman x for the second parton for the born process
c s   - c.m. energy squared for the born scattering
c t   - invariant variable for the scattering |(p1-p3)**2|,
c j   - parton type at current end of the ladder (0 - g, 1,-1,2,... - q)
c l   - parton type at opposite end of the ladder (0 - g, 1,-1,2,... - q)
c-----------------------------------------------------------------------
c reminder
c     psevi: 1 1 ... gluon -> gluon
c            2 1 ... quark -> gluon
c            1 2 ... gluon -> quark
c            3 2 ... quark -> quark non singlet
c            2 2 ... quark -> quark all
c                          singlet = all - non singlet
c-----------------------------------------------------------------------
      double precision xx1,xx2
      include 'epos.incsem'
      common/ccctest/iiitest

      if(jdis.eq.0)then
        scale=qt
      else
        scale=qt*4.
      endif
      if(j.eq.0.and.l.eq.0)then  ! gluon-gluon --->
        akg1=psevi(q1,scale,xx1,1,1)                  !gluon contribution
        akg2=psevi(q2,qt,xx2,1,1)                  !gluon contribution
        aks1=psevi(q1,scale,xx1,1,2)/naflav/2.  !singlet contribution per quark
        aks2=psevi(q2,qt,xx2,1,2)/naflav/2.  !singlet contribution per quark
        psjeti=ffborn(s,t,akg1*akg2
     *              ,(akg1*aks2+aks1*akg2)*naflav*2.    !ccccc
     *               ,aks1*aks2*naflav*2.
     *               ,aks1*aks2*naflav*2.
     *               ,aks1*aks2*naflav*2.*(naflav-1)*2.
     *)
      elseif(j.eq.0)then     !  gluon-quark --->
        akg1=psevi(q1,scale,xx1,1,1)                  !gluon contribution
        akg2=psevi(q2,qt,xx2,2,1)                  !gluon contribution
        aks1=psevi(q1,scale,xx1,1,2)/naflav/2.         !singlet contribution
        akns2=psevi(q2,qt,xx2,3,2)                 !nonsinglet contribution
        aks2=(psevi(q2,qt,xx2,2,2)-akns2)/naflav/2. !singlet contribution
        psjeti=ffborn(s,t,akg1*akg2
     *              ,(akg1*(akns2+aks2*naflav*2.)+aks1*akg2*naflav*2.)
     *              ,aks1*(akns2+aks2*naflav*2.)
     *              ,aks1*(akns2+aks2*naflav*2.)
     *              ,aks1*(akns2+aks2*naflav*2.)*(naflav-1)*2.)
      elseif(l.eq.0)then   ! quark-gluon --->
        akg1=psevi(q1,scale,xx1,2,1)                  !gluon contribution
        akg2=psevi(q2,qt,xx2,1,1)                  !gluon contribution
        akns1=psevi(q1,scale,xx1,3,2)                 !nonsinglet contribution
        aks1=(psevi(q1,scale,xx1,2,2)-akns1)/naflav/2. !singlet contribution
        aks2=psevi(q2,qt,xx2,1,2)/naflav/2.         !singlet contribution
        psjeti=ffborn(s,t,akg1*akg2
     *             ,(akg2*(akns1+aks1*naflav*2.)+aks2*akg1*naflav*2.)
     *             ,aks2*(akns1+aks1*naflav*2.)
     *             ,aks2*(akns1+aks1*naflav*2.)
     *             ,aks2*(akns1+aks1*naflav*2.)*(naflav-1)*2.)
      else     !  quark-quark --->
        akg1=psevi(q1,scale,xx1,2,1)                  !gluon contribution
        akg2=psevi(q2,qt,xx2,2,1)                  !gluon contribution
        akns1=psevi(q1,scale,xx1,3,2)                 !nonsinglet contribution
        aks1=(psevi(q1,scale,xx1,2,2)-akns1)/naflav/2.!singlet contribution
        akns2=psevi(q2,qt,xx2,3,2)                 !nonsinglet contribution
        aks2=(psevi(q2,qt,xx2,2,2)-akns2)/naflav/2.!singlet contribution

        if(j.eq.l)then
         psjeti=ffborn(s,t,akg1*akg2
     *     ,(akg2*(akns1+aks1*naflav*2.)+akg1*(akns2+aks2*naflav*2.))
     *     ,((akns1+aks1)*(akns2+aks2)+aks1*aks2*(2.*naflav-1.))
     *     ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)
     *     ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)*(naflav-1)*2.)
        elseif(j.eq.-l)then
         psjeti=ffborn(s,t,akg1*akg2
     *     ,(akg2*(akns1+aks1*naflav*2.)+akg1*(akns2+aks2*naflav*2.))
     *     ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)
     *     ,((akns1+aks1)*(akns2+aks2)+aks1*aks2*(2.*naflav-1.))
     *     ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)*(naflav-1)*2.)
        else                           !j.ne.l,-l
         psjeti=ffborn(s,t,akg1*akg2
     *    ,(akg2*(akns1+aks1*naflav*2.)+akg1*(akns2+aks2*naflav*2.))
     *    ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)
     *    ,(akns1*aks2+akns2*aks1+aks1*aks2*naflav*2.)
     *    ,(akns1*akns2+akns1*aks2*(naflav-1)*2.
     *    +akns2*aks1*(naflav-1)*2.+aks1*aks2*naflav*2.*(naflav-1)*2.))
        endif
      endif
      return
      end

c-----------------------------------------------------------------------
      function psjetj(q1,scale,t,xx,s,j,l,n)
c-----------------------------------------------------------------------
c psjetj - integrand for the ordered ladder cross-section
c q1 - virtuality cutoff at current end of the ladder,
c scale - born process scale,
c t  - invariant variable for the scattering |(p1-p3)**2|,
c xx - feinman x for the first parton for the born process
c s  - c.m. energy squared for the born scattering,
c j  - parton type at current end of the ladder (0 - g, 1,-1,2,... - q)
c l  - parton type at opposite end of the ladder (0 - g, 1,-1,2,... - q)
c n  - subprocess number
c-----------------------------------------------------------------------
      double precision xx
      include 'epos.incsem'

      m=min(1,iabs(j))+1
      if(l.ne.3)then
        if(l.eq.0)then
          psjetj=psevi(q1,scale,xx,m,1)*(psbori(s,t,0,0,n)+               !gg
     *    psbori(s,s-t,0,0,n))/2.
     *    +psevi(q1,scale,xx,m,2)*(psbori(s,t,1,0,n)+                     !qg
     *    psbori(s,s-t,1,0,n))
        elseif(j.eq.0)then
          aks=psevi(q1,scale,xx,1,2)/naflav/2.  !singlet contribution per quark
          psjetj=psevi(q1,scale,xx,1,1)*(psbori(s,t,0,1,n)+               !gq
     *    psbori(s,s-t,0,1,n))
     *    +aks*(psbori(s,t,1,1,n)+psbori(s,s-t,1,1,n))/2.             !qq
     *    +aks*(psbori(s,t,-1,1,n)+psbori(s,s-t,-1,1,n))              !qq~
     *    +aks*(psbori(s,t,1,2,n)+psbori(s,s-t,1,2,n))*(naflav-1)*2.   !qq'
        else
          akg=psevi(q1,scale,xx,2,1)                  !gluon contribution
          akns=psevi(q1,scale,xx,3,2)                 !nonsinglet contribution
          aks=(psevi(q1,scale,xx,2,2)-akns)/naflav/2.  !singlet contribution
          if(j.eq.l)then
            psjetj=akg*(psbori(s,t,0,1,n)+psbori(s,s-t,0,1,n))        !gq
     *      +(akns+aks)*(psbori(s,t,1,1,n)+psbori(s,s-t,1,1,n))/2.    !qq
     *      +aks*(psbori(s,t,-1,1,n)+psbori(s,s-t,-1,1,n))            !qq~
     *      +aks*(psbori(s,t,1,2,n)+psbori(s,s-t,1,2,n))*(naflav-1)*2. !qq'
          elseif(j.eq.-l)then
            psjetj=akg*(psbori(s,t,0,1,n)+psbori(s,s-t,0,1,n))        !gq
     *      +aks*(psbori(s,t,1,1,n)+psbori(s,s-t,1,1,n))/2.           !qq
     *      +(akns+aks)*(psbori(s,t,-1,1,n)+psbori(s,s-t,-1,1,n))     !qq~
     *      +aks*(psbori(s,t,1,2,n)+psbori(s,s-t,1,2,n))*(naflav-1)*2.!qq'
          else
            psjetj=akg*(psbori(s,t,0,1,n)+psbori(s,s-t,0,1,n))        !gq
     *      +aks*(psbori(s,t,1,1,n)+psbori(s,s-t,1,1,n))/2.           !qq
     *      +aks*(psbori(s,t,-1,1,n)+psbori(s,s-t,-1,1,n))            !qq~
     *      +(akns+aks*(naflav-1)*2.)*
     *      (psbori(s,t,1,2,n)+psbori(s,s-t,1,2,n))                   !qq'
          endif
        endif
      elseif(n.eq.1)then
        p1=s/(1.+qcmass**2/s)
        psjetj=psevi(q1,scale,xx,m,1)*(psbori(s,t,4,0,n)+                 !cg
     *  psbori(s,p1-t,4,0,n))
     *  +psevi(q1,scale,xx,m,2)*(psbori(s,t,4,1,n)+                       !cq
     *  psbori(s,p1-t,4,1,n))
      else
        psjetj=0.
      endif
      return
      end

c------------------------------------------------------------------------
      function psjti(q1,qqcut,s,m1,l1,jdis)
c-----------------------------------------------------------------------
c psjti - inclusive hard cross-section interpolation - for any ordering
c in the ladder
c q1 - virtuality cutoff at current end of the ladder
c qqcut - p_t cutoff for the born process;
c s  - total c.m. energy squared for the ladder
c m1 - parton type at current end of the ladder (0-g, 1,2,etc.-q)
c l1 - parton type at opposite end of the ladder (0-g, 1,2,etc.-q)
c-----------------------------------------------------------------------
      dimension wi(3),wj(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar19/ cstot(20,20,240)
      include 'epos.incsem'

      psjti=0.
c      jdis1=jdis
      if(jdis.eq.0)then
        qqmin=q1
        qmax=s/4.
      else
        qqmin=max(q2min,q1/4.)
        qmax=s
      endif
      qq=max(qqmin,qqcut)
      spmin=4.*q2min
      s2min=4.*qq
      if(s.le.s2min)return

      if(jdis.eq.0)then
        smins=s2min/(1.-q2ini/qq)
      else
        smins=s2min/(1.-q2ini/qq/4.)
      endif
      if(s.le.smins)goto 1

      if(s.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/s))
      else
        tmin=2.*qq
      endif
      tmax=s/2.

      if(m1.ne.0.and.m1.eq.l1)then
        m=2
        l=2
      elseif(m1.ne.0.and.m1.eq.-l1)then
        m=3
        l=1
      elseif(m1.ne.0.and.l1.ne.0.and.m1.ne.l1)then
        m=3
        l=2
      else
        m=min(1,iabs(m1))+1
        l=min(1,iabs(l1))+1
      endif

      ml=20*(m-1)+60*(l-1)+120*jdis
      qli=log(q1/q2min)/log(qmax/q2min)*19.+1.
      qlj=log(qq/qqmin)/log(s/4./qqmin)*19.+1.
      sl=log(s/spmin)/log(epmax/2./spmin)*19.+1.
      k=int(sl)
      i=int(qli)
      j=int(qlj)
      if(j.lt.1)j=1
      if(i.lt.1)i=1
      if(k.gt.18)k=18
      if(i.gt.18)i=18
      if(j.gt.18)j=18

      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.)*.5
      wj(1)=1.-wj(2)+wj(3)
      wj(2)=wj(2)-2.*wj(3)

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      do i1=1,3
      do j1=1,3
      do k1=1,3
        psjti=psjti+cstot(i+i1-1,j+j1-1,k+k1+ml-1)
     *  *wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
      psjti=exp(psjti)*(1./tmin-1./tmax)
      return
1     continue
      psjti=psbint(q1,q2min,qqcut,s,m1,l1,jdis)
      return
      end

c------------------------------------------------------------------------
      subroutine psjti0(ss,sj,sjb,m1,l1)
c-----------------------------------------------------------------------
c psjti0 - inclusive hard cross-section interpolation -
c for minimal virtuality cutoff in the ladder
c s - total c.m. energy squared for the ladder,
c sj - inclusive jet cross-section,
c sjb - born cross-section,
c m1 - parton type at current end of the ladder (0-g, 1,2,etc.-q)
c l1 - parton type at opposite end of the ladder (0-g, 1,2,etc.-q)
c-----------------------------------------------------------------------
      dimension wk(3)
      common /psar2/  edmax,epmax
      common /psar22/ cstotzero(20,4,2),csborzer(20,4,2)
      include 'epos.incsem'

      sj=0.
      sjb=0.
      if(iabs(m1).ne.4)then
        q2mass=0.
        if(m1.ne.0.and.m1.eq.l1)then
          m=2
          l=2
        elseif(m1.ne.0.and.m1.eq.-l1)then
          m=3
          l=1
        elseif(m1.ne.0.and.l1.ne.0.and.m1.ne.l1)then
          m=3
          l=2
        else
          m=min(1,iabs(m1))+1
          l=min(1,iabs(l1))+1
        endif
      else
        q2mass=qcmass**2
        m=4
        l=min(1,iabs(l1))+1
      endif
      s=ss-q2mass
      qq=q2min
      spmin=4.*qq+q2mass
      if(s.le.spmin)return

      p1=s/(1.+q2mass/s)
      if(p1.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
      else
        tmin=2.*qq
      endif
      tmax=.5*p1

      sl=log(s/spmin)/log(epmax/2./spmin)*19.+1.
      k=int(sl)
      if(k.gt.18)k=18
      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      do k1=1,3
        sj=sj+cstotzero(k+k1-1,m,l)*wk(k1)
        sjb=sjb+csborzer(k+k1-1,m,l)*wk(k1)
      enddo

      sjb=exp(sjb)*(1./tmin-1./tmax)
      sj=max(sjb,exp(sj)*(1./tmin-1./tmax))
      return
      end

c------------------------------------------------------------------------
      function psjti1(q1,q2,qqcut,s,m1,l1,jdis)
c-----------------------------------------------------------------------
c psjti1 - inclusive hard cross-section interpolation - for strict order
c in the ladder
c q1 - virtuality cutoff at current end of the ladder
c q2 - virtuality cutoff at opposite end of the ladder
c qqcut - p_t cutoff for the born process;
c s - total c.m. energy squared for the ladder,
c m1 - parton type at current end of the ladder (0-g, 1,2,etc.-q)
c l1 - parton type at opposite end of the ladder (0-g, 1,2,etc.-q)
c-----------------------------------------------------------------------
      dimension wi(3),wj(3),wk(3)
      common /psar2/  edmax,epmax
      common /psar20/ csord(20,20,240)
      include 'epos.incsem'
      double precision psuds

      psjti1=0.
      if(jdis.eq.0)then
        qqmin=max(q1,q2)
      else
        qqmin=max(q1,q2/4.)
      endif
      qq=max(qqmin,qqcut)
      spmin=4.*q2min
      s2min=4.*qq
      if(s.le.s2min)return

      smins=s2min/(1.-q2ini/qq)
      if(s.le.smins)goto 1

      if(s.gt.4.*qq)then
        tmin=2.*qq/(1.+sqrt(1.-4.*qq/s))
      else
        tmin=2.*qq
      endif
      tmax=s/2.

      if(m1.ne.0.and.m1.eq.l1)then
        m=2
        l=2
      elseif(m1.ne.0.and.m1.eq.-l1)then
        m=3
        l=1
      elseif(m1.ne.0.and.l1.ne.0.and.m1.ne.l1)then
        m=3
        l=2
      else
        m=min(1,iabs(m1))+1
        l=min(1,iabs(l1))+1
      endif

      ml=20*(m-1)+60*(l-1)+120*jdis
      qli=log(q1/q2min)/log(s/4./q2min)*19.+1.
      qlj=log(qq/qqmin)/log(s/4./qqmin)*19.+1.
      sl=log(s/spmin)/log(epmax/2./spmin)*19.+1.
      k=int(sl)
      i=int(qli)
      j=int(qlj)
      if(j.lt.1)j=1
      if(i.lt.1)i=1
      if(k.gt.18)k=18
      if(i.gt.18)i=18
      if(j.gt.18)j=18

      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.)*.5
      wi(1)=1.-wi(2)+wi(3)
      wi(2)=wi(2)-2.*wi(3)

      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.)*.5
      wj(1)=1.-wj(2)+wj(3)
      wj(2)=wj(2)-2.*wj(3)

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      do i1=1,3
      do j1=1,3
      do k1=1,3
        k2=k+k1+ml-1
        psjti1=psjti1+csord(i+i1-1,j+j1-1,k2)
     *  *wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
      psjti1=exp(psjti1)*(1./tmin-1./tmax)

      if(jdis.eq.0.and.qq.gt.q2)then
        psjti1=psjti1*sngl(psuds(qq,l1)/psuds(q2,l1))
      elseif(jdis.eq.1.and.4.*qq.gt.q2)then
        psjti1=psjti1*sngl(psuds(4.*qq,l1)/psuds(q2,l1))
      endif
      return
1     continue
      if(jdis.eq.0)then
        psjti1=psbint(q1,q2,qqcut,s,m1,l1,0)
      else
        psjti1=psbint(q2,q1,qqcut,s,l1,m1,1)
      endif
      return
      end

c------------------------------------------------------------------------
      function pspdfg(xx,qqs,qq,iclpro0,j)
c-----------------------------------------------------------------------
c pspdf - parton distribution function
c qq  - virtuality scale
c qqs - initial virtuality for the input distributions
c iclpro0 - hadron class
c j   - parton type
c-----------------------------------------------------------------------
      double precision z
      common/ar3/    x1(7),a1(7)
      include 'epos.incsem'
      double precision psuds

      pspdfg=psdfh4(xx,qqs,0.,iclpro0,j)
      if(j.gt.0)pspdfg=pspdfg+psdfh4(xx,qqs,0.,iclpro0,-j)  !+sea contr.
      pspdfg=pspdfg*sngl(psuds(qq,j)/psuds(qqs,j))

      xmin=xx/(1.-q2ini/qq)
      if(xmin.ge.1.)return

      dpd1=0.
      dpd2=0.
      xm=max(xmin,.3)
      do i=1,7         !numerical integration over zx
      do m=1,2
        zx=1.-(1.-xm)*(.5+(m-1.5)*x1(i))**.25
        z=xx/zx

        if(j.eq.0)then
          aks=psevi(qqs,qq,z,2,1)                  !quark contribution
          akg=psevi(qqs,qq,z,1,1)                  !gluon contribution
          akns=0.
        else
          akg=psevi(qqs,qq,z,1,2)/naflav/2.         !gluon contribution
          akns=psevi(qqs,qq,z,3,2)            !nonsinglet contribution
          aks=(psevi(qqs,qq,z,2,2)-akns)/naflav/2.  !quark contribution
        endif

        fz=akg*psdfh4(zx,qqs,0.,iclpro0,0)
     *  +akns*psdfh4(zx,qqs,0.,iclpro0,j)
     *  +aks*(psdfh4(zx,qqs,0.,iclpro0,1)+
     *  2.*psdfh4(zx,qqs,0.,iclpro0,-1)
     *  +psdfh4(zx,qqs,0.,iclpro0,2)+2.*psdfh4(zx,qqs,0.,iclpro0,-2)
     *  +2.*psdfh4(zx,qqs,0.,iclpro0,-3))
        if(j.gt.0)fz=fz+akns*psdfh4(zx,qqs,0.,iclpro0,-j)

        dpd1=dpd1+a1(i)*fz/zx**2/(1.-zx)**3
      enddo
      enddo
      dpd1=dpd1*(1.-xm)**4/8.*xx

      if(xm.gt.xmin)then
        do i=1,7         !numerical integration
        do m=1,2
          zx=xx+(xm-xx)*((xmin-xx)/(xm-xx))**(.5-(m-1.5)*x1(i))
          z=xx/zx

          if(j.eq.0)then
            aks=psevi(qqs,qq,z,2,1)                  !quark contribution
            akg=psevi(qqs,qq,z,1,1)                  !gluon contribution
            akns=0.
          else
            akg=psevi(qqs,qq,z,1,2)/naflav/2.         !gluon contribution
            akns=psevi(qqs,qq,z,3,2)            !nonsinglet contribution
            aks=(psevi(qqs,qq,z,2,2)-akns)/naflav/2.  !quark contribution
          endif

          fz=akg*psdfh4(zx,qqs,0.,iclpro0,0)
     *    +akns*psdfh4(zx,qqs,0.,iclpro0,j)
     *    +aks*(psdfh4(zx,qqs,0.,iclpro0,1)
     *    +2.*psdfh4(zx,qqs,0.,iclpro0,-1)
     *    +psdfh4(zx,qqs,0.,iclpro0,2)+2.*psdfh4(zx,qqs,0.,iclpro0,-2)
     *    +2.*psdfh4(zx,qqs,0.,iclpro0,-3))
          if(j.gt.0)fz=fz+akns*psdfh4(zx,qqs,0.,iclpro0,-j)

          dpd2=dpd2+a1(i)*fz*(1.-xx/zx)/zx
        enddo
        enddo
        dpd2=dpd2*log((xm-xx)/(xmin-xx))*.5*xx
      endif
      pspdfg=pspdfg+dpd2+dpd1
      return
      end

c-----------------------------------------------------------------------
      subroutine psaevp
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      qq=xpar1
      jmod=nint(xpar2)
      iologb=1

      if(jmod.eq.0)then            !??????????????ttttttt
      write(*,*)"no more triple Pomeron, xpar2=0 in psaevp not accepted"
      write(*,*)"use xpar2=1 instead"
      jmod=1
      endif

      do i=1,nrbins
        if(iologb.eq.0)then
          xx=xminim+(xmaxim-xminim)*(i-.5)/nrbins
        else
          xx=xminim*(xmaxim/xminim)**((i-.5)/nrbins)
        endif
        ar(i,1)=xx
        ar(i,2)=0.
        if(jmod.eq.0)then            !evolution+matrix element +3P (ours)
          ww=qq/xx
          ar(i,3)=(psdh(ww,qq,2,0)+psdh(ww,qq,2,1)
c     *    +psdsh1(ww,qq,2,dqsh,0)+psdsh1(ww,qq,2,dqsh,1)
     *    )/(4.*pi**2*alfe)*qq
        elseif(jmod.eq.1)then        !evolution+matrix element (ours)
          ww=qq/xx
          ar(i,3)=(psdh(ww,qq,2,0)+psdh(ww,qq,2,1)+
     *    psdsh(ww,qq,2,dqsh,0)+psdsh(ww,qq,2,dqsh,1)
     *    )/(4.*pi**2*alfe)*qq
        elseif(jmod.eq.2)then    !just evolution (grv)
          ar(i,3)=(pspdfg(xx,q2min,qq,2,1)/2.25+
     *    pspdfg(xx,q2min,qq,2,2)/9.+
     *    pspdfg(xx,q2min,qq,2,-1)*2./3.6+
     *    pspdfg(xx,q2min,qq,2,-3)*2./9.)
          if(naflav.eq.4)ar(i,3)=ar(i,3)+pspdfg(xx,q2min,qq,2,-4)
     *    *2./2.25
        elseif(jmod.eq.3)then    !grv
          ar(i,3)=(psdfh4(xx,qq,0.,2,1)+2.*psdfh4(xx,qq,0.,2,-1))/2.25
     *    +(psdfh4(xx,qq,0.,2,2)+2.*psdfh4(xx,qq,0.,2,-2))/9.
     *    +2.*psdfh4(xx,qq,0.,2,-3)/9.  !
        elseif(jmod.eq.4)then         !just evolution (ours)
          ar(i,3)=(fparton(xx,qq,1)/2.25+fparton(xx,qq,2)/9.+
     *    fparton(xx,qq,-1)*6./4.5)                     !uv+dv+6*sea
          if(naflav.eq.4)ar(i,3)=ar(i,3)+fparton(xx,qq,-4)*2./2.25
        elseif(jmod.eq.5)then         !grv+res
          ww=qq/xx
          ar(i,3)=(psdgh(ww,qq,0)+psdgh(ww,qq,1)
     *    )/(4.*pi**2*alfe)*qq
        endif
        ar(i,4)=0.
      enddo
      return
      end

c------------------------------------------------------------------------
      subroutine pscs(c,s)
c-----------------------------------------------------------------------
c pscs - cos (c) and sin (s) generation for uniformly distributed angle
c-----------------------------------------------------------------------
1     s1=2.*rangen()-1.
      s2=2.*rangen()-1.
      s3=s1*s1+s2*s2
      if(s3.gt.1.)goto 1
      s3=sqrt(s3)
      c=s1/s3
      s=s2/s3
      return
      end

c------------------------------------------------------------------------
      subroutine psdefrot(ep,s0x,c0x,s0,c0)
c-----------------------------------------------------------------------
c psdefrot - determination of the parameters the spacial rotation to the
c system for 4-vector ep
c s0, c0 - sin and cos for the zx-rotation;
c s0x, c0x - sin and cos for the xy-rotation
c-----------------------------------------------------------------------
      dimension ep(4)

c transverse momentum square for the current parton (ep)
      pt2=ep(3)**2+ep(4)**2
      if(pt2.ne.0.)then
        pt=sqrt(pt2)
c system rotation to get pt=0 - euler angles are determined (c0x = cos t
c s0x = sin theta, c0 = cos phi, s0 = sin phi)
        c0x=ep(3)/pt
        s0x=ep(4)/pt
c total momentum for the gluon
        pl=sqrt(pt2+ep(2)**2)
        s0=pt/pl
        c0=ep(2)/pl
      else
        c0x=1.
        s0x=0.
        pl=abs(ep(2))
        s0=0.
        c0=ep(2)/pl
      endif

      ep(2)=pl
      ep(3)=0.
      ep(4)=0.
      return
      end

c------------------------------------------------------------------------
      subroutine psdeftr(s,ep,ey)
c-----------------------------------------------------------------------
c psdeftr - determination of the parameters for the lorentz transform to
c rest frame system for 4-vector ep of mass squared s
c-----------------------------------------------------------------------
      dimension ey(3)
      double precision ep(4)

      do i=1,3
        if(ep(i+1).eq.0.d0)then
          ey(i)=1.
        else
          wp=ep(1)+ep(i+1)
          wm=ep(1)-ep(i+1)
          if(wp.gt.1.e-8.and.wm/wp.lt.1.e-8)then
            ww=s
            do l=1,3
              if(l.ne.i)ww=ww+ep(l+1)**2
            enddo
            wm=ww/wp
          elseif(wm.gt.1.e-8.and.wp/wm.lt.1.e-8)then
            ww=s
            do l=1,3
              if(l.ne.i)ww=ww+ep(l+1)**2
            enddo
            wp=ww/wm
          endif
          ey(i)=sqrt(wm/wp)
          ep(1)=wp*ey(i)
          ep(i+1)=0.
        endif
      enddo
      ep(1)=dsqrt(dble(s))
      return
      end

c------------------------------------------------------------------------
      function psdfh4(xxx,qqs,qq,icq,iq)
c------------------------------------------------------------------------
c psdfh4 - GRV structure functions
c------------------------------------------------------------------------
      common /psar8/  stmass ,amhadr(8),qcmass
      common /psar36/ alvc

      psdfh4=0.
!      if(x.gt..99999)return
      x=min(xxx,0.99999)              !warning ! but necessary for idraflx

      if(icq.eq.2)then
        if(qqs.le.0.232**2)return
        sq=log(log(qqs/.232**2)/log(.23/.232**2))
        if(sq.le.0.)return
        if(iq.eq.0)then                                 !gluon
          alg=.524
          betg=1.088
          aag=1.742-.93*sq
          bbg=-.399*sq**2
          ag=7.486-2.185*sq
          bg=16.69-22.74*sq+5.779*sq*sq
          cg=-25.59+29.71*sq-7.296*sq*sq
          dg=2.792+2.215*sq+.422*sq*sq-.104*sq*sq*sq
          eg=.807+2.005*sq
          eeg=3.841+.361*sq
          psdfh4=(1.-x)**dg*(x**aag*(ag+bg*x+cg*x**2)*log(1./x)**bbg
     *    +sq**alg*exp(-eg+sqrt(eeg*sq**betg*log(1./x))))
        elseif(iq.eq.1.or.iq.eq.2)then                  !u_v or d_v
          aau=.59-.024*sq
          bbu=.131+.063*sq
          auu=2.284+.802*sq+.055*sq*sq
          au=-.449-.138*sq-.076*sq*sq
          bu=.213+2.669*sq-.728*sq*sq
          cu=8.854-9.135*sq+1.979*sq*sq
          du=2.997+.753*sq-.076*sq*sq
          uv=auu*x**aau*(1.-x)**du*
     *    (1.+au*x**bbu+bu*x+cu*x**1.5)

          aad=.376
          bbd=.486+.062*sq
          add=.371+.083*sq+.039*sq*sq
          ad=-.509+3.31*sq-1.248*sq*sq
          bd=12.41-10.52*sq+2.267*sq*sq
          ccd=6.373-6.208*sq+1.418*sq*sq
          dd=3.691+.799*sq-.071*sq*sq
          dv=add*x**aad*(1.-x)**dd*
     *    (1.+ad*x**bbd+bd*x+ccd*x**1.5)

          if(iq.eq.1)then                              !u_v
            psdfh4=uv
          elseif(iq.eq.2)then                          !d_v
            psdfh4=dv
          endif
        elseif(iq.eq.-3)then                           !s_sea
          als=.914
          bets=.577
          aas=1.798-.596*sq
          as=-5.548+3.669*sqrt(sq)-.616*sq
          bs=18.92-16.73*sqrt(sq)+5.168*sq
          ds=6.379-.35*sq+.142*sq*sq
          es=3.981+1.638*sq
          ees=6.402
          psdfh4=(1.-x)**ds*sq**als/log(1./x)**aas*(1.+as*sqrt(x)
     *    +bs*x)*exp(-es+sqrt(ees*sq**bets*log(1./x)))
        elseif(iabs(iq).lt.3)then                      !u_sea or d_sea
          aadel=.409-.005*sq
          bbdel=.799+.071*sq
          addel=.082+.014*sq+.008*sq*sq
          adel=-38.07+36.13*sq-.656*sq*sq
          bdel=90.31-74.15*sq+7.645*sq*sq
          ccdel=0.
          ddel=7.486+1.217*sq-.159*sq*sq
          delv=addel*x**aadel*(1.-x)**ddel*
     *    (1.+adel*x**bbdel+bdel*x+ccdel*x**1.5)

          alud=1.451
          betud=.271
          aaud=.41-.232*sq
          bbud=.534-.457*sq
          aud=.89-.14*sq
          bud=-.981
          cud=.32+.683*sq
          dud=4.752+1.164*sq+.286*sq*sq
          eud=4.119+1.713*sq
          eeud=.682+2.978*sq
          udsea=(1.-x)**dud*(x**aaud*(aud+bud*x+cud*x**2)
     *    *log(1./x)**bbud+sq**alud*exp(-eud+sqrt(eeud*sq**betud*
     *    log(1./x))))

          if(iq.eq.-1)then                           !u_sea
            psdfh4=(udsea-delv)/2.
          elseif(iq.eq.-2)then                       !d_sea
            psdfh4=(udsea+delv)/2.
          endif
        else
          psdfh4=0.
        endif
      elseif(icq.eq.1.or.icq.eq.3)then
        if(qqs.le.0.204**2)return
        sq=log(log(qqs/.204**2)/log(.26/.204**2))
        if(sq.le.0.)return
        if(iq.eq.1.or.iq.eq.2)then
          aapi=.517-.02*sq
          api=-.037-.578*sq
          bpi=.241+.251*sq
          dpi=.383+.624*sq
          anorm=1.212+.498*sq+.009*sq**2
          psdfh4=.5*anorm*x**aapi*(1.-x)**dpi*
     *    (1.+api*sqrt(x)+bpi*x)
        elseif(iq.eq.0)then
          alfpi=.504
          betpi=.226
          aapi=2.251-1.339*sqrt(sq)
          api=2.668-1.265*sq+.156*sq**2
          bbpi=0.
          bpi=-1.839+.386*sq
          cpi=-1.014+.92*sq-.101*sq**2
          dpi=-.077+1.466*sq
          epi=1.245+1.833*sq
          eppi=.51+3.844*sq
          psdfh4=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *    log(1./x)**bbpi+sq**alfpi*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
        elseif(iq.eq.-3)then
          alfpi=.823
          betpi=.65
          aapi=1.036-.709*sq
          api=-1.245+.713*sq
          bpi=5.58-1.281*sq
          dpi=2.746-.191*sq
          epi=5.101+1.294*sq
          eppi=4.854-.437*sq
          psdfh4=sq**alfpi/log(1./x)**aapi*(1.-x)**dpi*
     *    (1.+api*sqrt(x)+bpi*x)*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x)))
        elseif(iabs(iq).lt.3)then
          alfpi=1.147
          betpi=1.241
          aapi=.309-.134*sqrt(sq)
          api=.219-.054*sq
          bbpi=.893-.264*sqrt(sq)
          bpi=-.593+.24*sq
          cpi=1.1-.452*sq
          dpi=3.526+.491*sq
          epi=4.521+1.583*sq
          eppi=3.102
          psdfh4=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *    log(1./x)**bbpi+sq**alfpi*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
        else
          psdfh4=0.
        endif
      elseif(icq.eq.0)then
        if(qqs.le.0.204**2)return
        sq=log(log(qqs/.204**2)/log(.26/.204**2))
        if(sq.le.0.)return
        if(iq.eq.0)then
          alfpi=.504
          betpi=.226
          aapi=2.251-1.339*sqrt(sq)
          api=2.668-1.265*sq+.156*sq**2
          bbpi=0.
          bpi=-1.839+.386*sq
          cpi=-1.014+.92*sq-.101*sq**2
          dpi=-.077+1.466*sq
          epi=1.245+1.833*sq
          eppi=.51+3.844*sq
          psdfh4=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *    log(1./x)**bbpi+sq**alfpi*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
     *    *.543
        else
          alfpi=.823
          betpi=.65
          aapi=1.036-.709*sq
          api=-1.245+.713*sq
          bpi=5.58-1.281*sq
          dpi=2.746-.191*sq
          epi=5.101+1.294*sq
          eppi=4.854-.437*sq
          str=sq**alfpi/log(1./x)**aapi*(1.-x)**dpi*
     *    (1.+api*sqrt(x)+bpi*x)*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x)))
          if(iq.eq.3)then
            psdfh4=str*.543*2.
          else
            aapi=.517-.02*sq
            api=-.037-.578*sq
            bpi=.241+.251*sq
            dpi=.383+.624*sq
            anorm=1.212+.498*sq+.009*sq**2
            val=.5*anorm*x**aapi*(1.-x)**dpi*
     *      (1.+api*sqrt(x)+bpi*x)

            alfpi=1.147
            betpi=1.241
            aapi=.309-.134*sqrt(sq)
            api=.219-.054*sq
            bbpi=.893-.264*sqrt(sq)
            bpi=-.593+.24*sq
            cpi=1.1-.452*sq
            dpi=3.526+.491*sq
            epi=4.521+1.583*sq
            eppi=3.102
            sea=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *      log(1./x)**bbpi+sq**alfpi*
     *      exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
            if(iq.eq.1)then
              psdfh4=(.836*(val+2.*sea)-.587*str)
            elseif(iq.eq.2)then
              psdfh4=(.25*(val+2.*sea)+.587*str)
            else
              psdfh4=0.
            endif
          endif
        endif
        psdfh4=psdfh4/(1.+qq/.59)**2

      elseif(icq.eq.4)then
        if(qqs.le.qcmass**2)return
        sq=log(log(qqs/qcmass**2)/log(.23/qcmass**2))
        if(sq.le.0.)return
        if(iq.eq.2)then
          psdfh4=x**3*(1.-x)**alvc*(alvc+3.)*(alvc+2.)*(alvc+1.)
        else
          aapi=.517-.02*sq
          api=-.037-.578*sq
          bpi=.241+.251*sq
          dpi=.383+.624*sq
          anorm=1.212+.498*sq+.009*sq**2
          psdfh4=.5*anorm*x**aapi*(1.-x)**dpi*
     *    (1.+api*sqrt(x)+bpi*x)
        endif
      else
        psdfh4=0.
      endif
      return
      end


c------------------------------------------------------------------------
      function psfap(x,j,l)
c-----------------------------------------------------------------------
c psfap - altarelli-parisi function (multiplied by x)
c x - light cone momentum share value,
c j - type of the parent parton (0-g;1,2,etc.-q)
c l - type of the daughter parton (0-g;1,2,etc.-q)
c-----------------------------------------------------------------------
      double precision x
      include 'epos.incsem'

      if(j.eq.0)then
        if(l.eq.0)then
          psfap=((1.d0-x)/x+x/(1.d0-x)+x*(1.d0-x))*6.d0
        else
          psfap=(x**2+(1.d0-x)**2)*naflav
        endif
      else
        if(l.eq.0)then
          psfap=(1.d0+(1.d0-x)**2)/x/.75d0
        else
          psfap=(x**2+1.d0)/(1.d0-x)/.75d0
        endif
      endif
      return
      end

cc------------------------------------------------------------------------
c      function psgen(a1,a2)
cc-----------------------------------------------------------------------
cc psgen - x-values generation according to distribution
cc x1^(-a1) x2^(-0.5)
cc-----------------------------------------------------------------------
c      common/lept1/engy,elepti,elepto,angmue,icinpu
c
c      aa=max(a1,a2)
c1     continue
c      if(aa.lt.1.)then
c        x1=.5*rangen()**(1./(1.-aa))
c      elseif(aa.eq.1.)then
c        x1=.5/engy**rangen()
c      else
c        x1=.5*(1.+rangen()*(engy**(aa-1.)-1.))**(1./(1.-aa))
c      endif
c      if(x1.lt.1.e-7.or.x1.gt..999999)then
c        goto 1
c      endif
c      if(rangen().lt..5)then
c        gb=x1**(aa-a1)*.5**aa/(1.-x1)**a2
c      else
c        x1=1.-x1
c        gb=(1.-x1)**(aa-a2)*.5**aa/x1**a1
c      endif
c      if(rangen().gt.gb)goto 1
c      psgen=x1
c      return
c      end
c
c------------------------------------------------------------------------
      function psidd(icc)
c-----------------------------------------------------------------------
c psidd - kink type decoder
c-----------------------------------------------------------------------
      if(icc.eq.0)then                    !g
        psidd=9
      elseif(iabs(icc).le.2)then          !u,u~,d,d~
        psidd=icc
      elseif(iabs(icc).eq.4)then          !s,s~
        psidd=icc/4*3
      elseif(iabs(icc).gt.10)then         !c,c~ etc.
        psidd=icc/10
      elseif(icc.eq.3)then                !ud
        psidd=1200
      elseif(icc.eq.-3)then               !u~d~
        psidd=-1200
      elseif(icc.eq.6)then                !uu
        psidd=1100
      elseif(icc.eq.-6)then               !u~u~
        psidd=-1100
      elseif(icc.eq.7)then                !dd
        psidd=2200
      elseif(icc.eq.-7)then               !d~d~
        psidd=-2200
      else
        psidd=0.
        write (*,*)'psidd?????????',icc
      endif
      return
      end

cc------------------------------------------------------------------------
c       function pslam(s,a,b)
cc-----------------------------------------------------------------------
cc kinematical function for two particle decay - maximal pt-value
cc a - first particle mass squared,
cc b - second particle mass squared,
cc s - two particle invariant mass squared
cc-----------------------------------------------------------------------
c       pslam=.25/s*(s+a-b)**2-a
c       return
c       end
c
c------------------------------------------------------------------------
      function psjvrg1(qt,s,y0)
c-----------------------------------------------------------------------
      common /ar3/   x1(7),a1(7)
      common /cnsta/ pi,pii,hquer,prom,piom,ainfin
      include 'epos.incsem'
      double precision xt,ymin,ymax,y,xmin,xmax,xx1,xx2

      psjvrg1=0.
      if(s.le.4.*qt)return

      xt=2.d0*sqrt(dble(qt)/dble(s))
      ymax=min(dble(y0),log(1d0/xt+sqrt((1d0/xt-1d0)*(1d0/xt+1d0))))
      ymin=-ymax

      do i=1,7
      do m=1,2
        y=.5d0*(ymax+ymin+(ymin-ymax)*dble((2*m-3)*x1(i)))
        xmin=xt**2/2.d0/(2.d0-xt*exp(-y))
        xmax=1.d0-xt*exp(y)/2.d0

        fx=0.
        do i1=1,7
        do m1=1,2
          xx1=xt*exp(y)/2d0+xmin*(xmax/xmin)**dble(.5+x1(i1)*(m1-1.5))
          xx2=xt*exp(-y)*xx1/(2.d0*xx1-xt*exp(y))
          z=sngl(xx1*xx2)
          sh=z*s
          t=sngl(dble(sh)/2d0*(1d0
     &                      -sqrt(max(0d0,1d0-4d0*dble(qt)/dble(sh)))))
          ft=psjvrx(t,qt,sngl(xx1),sngl(xx2),sh)
          fx=fx+a1(i1)*ft/sh**2
        enddo
        enddo
        fx=fx*sngl(log(xmax/xmin))
        psjvrg1=psjvrg1+a1(i)*fx
      enddo
      enddo
      psjvrg1=psjvrg1*sngl(ymax-ymin)*pi**3
     **pssalf(qt/qcdlam)**2*sqrt(qt)
      return
      end

c-----------------------------------------------------------------------
      function psjvrx(t,qt,xx1,xx2,s)
c-----------------------------------------------------------------------
      include 'epos.incsem'

      g1=psdfh4(xx1,qt,0.,2,0)
      ub1=psdfh4(xx1,qt,0.,2,-1)
      u1=psdfh4(xx1,qt,0.,2,1)+ub1
      db1=psdfh4(xx1,qt,0.,2,-2)
      d1=psdfh4(xx1,qt,0.,2,2)+db1
      sb1=psdfh4(xx1,qt,0.,2,-3)
      s1=sb1
      g2=psdfh4(xx2,qt,0.,2,0)
      ub2=psdfh4(xx2,qt,0.,2,-1)
      u2=psdfh4(xx2,qt,0.,2,1)+ub2
      db2=psdfh4(xx2,qt,0.,2,-2)
      d2=psdfh4(xx2,qt,0.,2,2)+db2
      sb2=psdfh4(xx2,qt,0.,2,-3)
      s2=sb2

      psjvrx=g1*g2*(psbori(s,t,0,0,1)+psbori(s,s-t,0,0,1)
     *+psbori(s,t,0,0,2)+psbori(s,s-t,0,0,2))/2.
     *+(psbori(s,t,0,1,1)+psbori(s,s-t,0,1,1))*
     *(g2*(u1+ub1+d1+db1+s1+sb1)+g1*(u2+ub2+d2+db2+s2+sb2))
     *+(psbori(s,t,1,1,1)+psbori(s,s-t,1,1,1))/2.*
     *(u1*u2+ub1*ub2+d1*d2+db1*db2+s1*s2+sb1*sb2)
     *+(psbori(s,t,1,-1,1)+psbori(s,s-t,1,-1,1)+psbori(s,t,1,-1,2)+
     *psbori(s,s-t,1,-1,2)+psbori(s,t,1,-1,3)+psbori(s,s-t,1,-1,3))*
     *(u1*ub2+ub1*u2+d1*db2+db1*d2+s1*sb2+sb1*s2)
     *+(psbori(s,t,1,2,1)+psbori(s,s-t,1,2,1))*
     *((u1+ub1)*(d2+db2+s2+sb2)+(u2+ub2)*(d1+db1+s1+sb1)+
     *(d1+db1)*(u2+ub2+s2+sb2)+(d2+db2)*(u1+ub1+s1+sb1)+
     *(s1+sb1)*(u2+ub2+d2+db2)+(s2+sb2)*(u1+ub1+d1+db1))
      return
      end

c------------------------------------------------------------------------
      function psjwo1(qt,s,y0)
c-----------------------------------------------------------------------
      common /ar3/   x1(7),a1(7)
      common /cnsta/ pi,pii,hquer,prom,piom,ainfin
      double precision xt,ymax,ymin,y,xmin,xmax,xx1,xx2
      include 'epos.incsem'

      psjwo1=0.
      if(s.le.4.*qt)return

      xt=2.d0*sqrt(dble(qt)/dble(s))
      ymax=min(dble(y0),log(1d0/xt+sqrt((1d0/xt-1d0)*(1d0/xt+1d0))))
      ymin=-ymax

      do i=1,7
      do m=1,2
        y=.5d0*(ymax+ymin+(ymin-ymax)*dble(2*m-3)*dble(x1(i)))
        xmin=xt**2/2.d0/(2.d0-xt*exp(-y))
        xmax=1.d0-xt*exp(y)/2.d0

        fx=0.
        do i1=1,7
        do m1=1,2
          xx1=xt*exp(y)/2.d0+xmin*(xmax/xmin)**dble(.5+x1(i1)*(m1-1.5))
          xx2=xt*exp(-y)/(2.d0-xt*exp(y)/xx1)
          z=sngl(xx1*xx2)
          sh=z*s
          t=sngl(dble(sh)/2d0*(1d0-sqrt(1d0-4d0*dble(qt)/dble(sh))))
          ft=psjwox(t,qt,sngl(xx1),sngl(xx2),sh)
          fx=fx+a1(i1)*ft/sh**2
        enddo
        enddo
        fx=fx*log(xmax/xmin)
        psjwo1=psjwo1+a1(i)*fx
      enddo
      enddo
      psjwo1=psjwo1*sngl(ymax-ymin)*pi**3
     **pssalf(qt/qcdlam)**2*sqrt(qt)
      return
      end

c-----------------------------------------------------------------------
      function psjwox(t,qt,xx1,xx2,s)
c-----------------------------------------------------------------------
      double precision x,scale,upv1,dnv1,sea1,str1,chm1,gl1,
     *upv2,dnv2,sea2,str2,chm2,gl2
      scale=sqrt(qt)
      x=xx1
      call strdo1(x,scale,upv1,dnv1,sea1,str1,chm1,gl1)
      x=xx2
      call strdo1(x,scale,upv2,dnv2,sea2,str2,chm2,gl2)

      psjwox=gl1*gl2*(psbori(s,t,0,0,1)+psbori(s,s-t,0,0,1)
     *+psbori(s,t,0,0,2)+psbori(s,s-t,0,0,2)+psbori(s,t,0,0,3)
     *+psbori(s,s-t,0,0,3))/2.
     *+(psbori(s,t,0,1,1)+psbori(s,s-t,0,1,1)
     *+psbori(s,t,0,1,2)+psbori(s,s-t,0,1,2)+psbori(s,t,0,1,3)
     *+psbori(s,s-t,0,1,3))*(gl2*(upv1+dnv1+4.*sea1+2.*str1+2.*chm1)+
     *gl1*(upv2+dnv2+4.*sea2+2.*str2+2.*chm2))
     *+(psbori(s,t,1,1,1)+psbori(s,s-t,1,1,1)
     *+psbori(s,t,1,1,2)+psbori(s,s-t,1,1,2)+psbori(s,t,1,1,3)+
     *psbori(s,s-t,1,1,3))/2.*
     *((upv1+sea1)*(upv2+sea2)+(dnv1+sea1)*(dnv2+sea2)+2.*sea1*sea2
     *+2.*str1*str2+2.*chm1*chm2)
     *+(psbori(s,t,1,-1,1)+psbori(s,s-t,1,-1,1)+psbori(s,t,1,-1,2)+
     *psbori(s,s-t,1,-1,2)+psbori(s,t,1,-1,3)+psbori(s,s-t,1,-1,3))*
     *((upv1+sea1)*sea2+sea1*(upv2+sea2)+(dnv1+sea1)*sea2+
     *sea1*(dnv2+sea2)+2.*str1*str2+2.*chm1*chm2)
     *+(psbori(s,t,1,2,1)
     *+psbori(s,s-t,1,2,1)+psbori(s,t,1,2,2)+psbori(s,s-t,1,2,2)
     *+psbori(s,t,1,2,3)+psbori(s,s-t,1,2,3))*
     *(upv1*dnv2+upv2*dnv1+(upv1+dnv1)*(2.*sea2+2.*str2+2.*chm2)+
     *(upv2+dnv2)*(2.*sea1+2.*str1+2.*chm1)+
     *4.*sea1*(2.*sea2+2.*str2+2.*chm2)+2.*str1*(4.*sea2+2.*chm2)+
     *2.*chm1*(4.*sea2+2.*str2))
      return
      end

c------------------------------------------------------------------------
      subroutine pslcsh(wp1,wm1,wp2,wm2,samqt,amqpt)
c-----------------------------------------------------------------------
c pslcsh - sh pomeron lc momentum sharing between two strings
c------------------------------------------------------------------------
      double precision amqt(4),yqm(4),yqm1(4),xlp(4),xlm(4),am23,sx,y2
     *,wp1,wp2,wm1,wm2,s,sq,psutz,yqmax,y,amjp,amjm,y1,s12,s34,x34,amqpt
      dimension samqt(4)
      include 'epos.inc'

      s=wp1*wm1
      sq=dsqrt(s)
      do i=1,4
        amqt(i)=dble(samqt(i))
        yqm(i)=dlog(sq/amqt(i)*psutz(s,amqt(i)**2,(amqpt-amqt(i))**2))
      enddo
      yqmax=max(yqm(1),yqm(2))

1     y=yqmax*dble(rangen())
      j=int(1.5+rangen())
      if(y.gt.yqm(j))goto 1

      amjp=amqt(j)*dexp(y)
      amjm=amqt(j)*dexp(-y)
      do i=3,4
        am23=amqt(3-j)+amqt(7-i)
        sx=(am23+amjp)*(am23+amjm)
        yqm1(i)=dlog(sq/amqt(i)*psutz(s,amqt(i)**2,sx))
      enddo
      yqmax1=max(yqm1(3),yqm1(4))
      if(dble(rangen()).gt.yqmax1/max(yqm(3),yqm(4)))goto 1

      y1=yqmax1*dble(rangen())
      j1=int(3.5+rangen())
      if(y1.gt.yqm1(j1))goto 1

      amjp1=amqt(j1)*exp(y1)
      amjm1=amqt(j1)*exp(-y1)
      s12=(amqt(3-j)+amjp)*(amqt(3-j)+amjm)
      s34=(amqt(7-j1)+amjp1)*(amqt(7-j1)+amjm1)
      y2=dlog(sq/(amqt(3-j)+amjp)*psutz(s,s12,s34))

      xlp(j)=amqt(j)/sq*dexp(y+y2)
      xlm(j)=amqt(j)/sq*dexp(-y-y2)
      xlp(3-j)=amqt(3-j)/sq*dexp(y2)
      xlm(3-j)=amqt(3-j)/sq*dexp(-y2)
      x34=1.-xlm(1)-xlm(2)
      xlm(7-j1)=x34/(1.+amjp1/amqt(7-j1))
      xlm(j1)=x34-xlm(7-j1)
c      write (*,*)'xlc',xlp(1),xlp(2),xlm(3),xlm(4)
      if(dble(rangen()).gt.(xlp(1)*xlp(2)*xlm(3)*xlm(4))**(-alpqua)*
     *(xlp(j)*(1.d0-xlp(j))*xlm(j1)*(1.d0-xlm(j1))))goto 1

      wp2=xlp(2)*wp1
      wp1=xlp(1)*wp1
      wm2=xlm(4)*wm1
      wm1=xlm(3)*wm1
c      write (*,*)'wp1,wm1,wp2,wm2',wp1,wm1,wp2,wm2
      return
      end

c------------------------------------------------------------------------
      function psnorm(ep)
c-----------------------------------------------------------------------
c 4-vector squared calculation
c-----------------------------------------------------------------------
      double precision sm2,ep(4)
      sm2=ep(1)**2
      do i=1,3
        sm2=sm2-ep(i+1)**2
      enddo
      psnorm=sm2
      return
      end

c------------------------------------------------------------------------
      subroutine psrotat(ep,s0x,c0x,s0,c0)
c-----------------------------------------------------------------------
c psrotat - spacial rotation to the lab. system for 4-vector ep
c s0, c0 - sin and cos for the zx-rotation;
c s0x, c0x - sin and cos for the xy-rotation
c-----------------------------------------------------------------------
      dimension ep(4),ep1(3)

      ep1(3)=ep(4)
      ep1(2)=ep(2)*s0+ep(3)*c0
      ep1(1)=ep(2)*c0-ep(3)*s0

      ep(2)=ep1(1)
      ep(4)=ep1(2)*s0x+ep1(3)*c0x
      ep(3)=ep1(2)*c0x-ep1(3)*s0x
      return
      end

cc------------------------------------------------------------------------
c      subroutine psrotat1(ep,s0x,c0x,s0,c0)
cc-----------------------------------------------------------------------
cc psrotat - spacial rotation to the lab. system for 4-vector ep
cc s0, c0 - sin and cos for the zx-rotation;
cc s0x, c0x - sin and cos for the xy-rotation
cc-----------------------------------------------------------------------
c      dimension ep(4),ep1(3)
c
c      ep1(1)=ep(2)
c      ep1(3)=-ep(3)*s0x+ep(4)*c0x
c      ep1(2)=ep(3)*c0x+ep(4)*s0x
c
c      ep(4)=ep1(3)
c      ep(3)=-ep1(1)*s0+ep1(2)*c0
c      ep(2)=ep1(1)*c0+ep1(2)*s0
c      return
c      end
c
c-----------------------------------------------------------------------
      function pssalf(qq)
c-----------------------------------------------------------------------
c pssalf - effective qcd coupling (alpha_s/2/pi)
c-----------------------------------------------------------------------
      include "epos.incsem"
      pssalf=2./(11.-naflav/1.5)/log(qq)
      return
      end

c------------------------------------------------------------------------
      subroutine pstrans(ep,ey,jj)
c-----------------------------------------------------------------------
c pstrans - lorentz boosts according to the parameters ey ( determining
c shift along the z,x,y-axis respectively (ey(1),ey(2),ey(3)))
c jj=1 - inverse transformation to the lab. system;
c jj=-1 - direct transformation
c-----------------------------------------------------------------------
      dimension ey(3),ep(4)

      if(jj.eq.1)then
c lorentz transform to lab. system according to 1/ey(i) parameters
        do i=1,3
          if(ey(4-i).ne.1.)then
            wp=(ep(1)+ep(5-i))/ey(4-i)
            wm=(ep(1)-ep(5-i))*ey(4-i)
            ep(1)=.5*(wp+wm)
            ep(5-i)=.5*(wp-wm)
          endif
        enddo
      else
c lorentz transform to lab. system according to ey(i) parameters
        do i=1,3
          if(ey(i).ne.1.)then
            wp=(ep(1)+ep(i+1))*ey(i)
            wm=(ep(1)-ep(i+1))/ey(i)
            ep(1)=.5*(wp+wm)
            ep(i+1)=.5*(wp-wm)
          endif
        enddo
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function psuds(q,m)
c-----------------------------------------------------------------------
c psuds - spacelike sudakov formfactor
c q - maximal value of the effective momentum,
c m - type of parton (0 - g, 1,2, etc. - q)
c-----------------------------------------------------------------------
      dimension wi(3)
      common /psar15/ sudx(40,2)
      include 'epos.incsem'
      double precision dps,qlm,ffacs,qlm0,qlmi

      j=min(iabs(m),1)+1

      if(q.gt.q2ini)then
        qli=log(q/q2min)*2.+1.
        i=int(qli)
        if(i.lt.1)i=1
        if(i.gt.38)i=38
        wi(2)=qli-i
        wi(3)=wi(2)*(wi(2)-1.)*.5
        wi(1)=1.-wi(2)+wi(3)
        wi(2)=wi(2)-2.*wi(3)
        dps=0.d0
        do i1=1,3
          dps=dps+dble(sudx(i+i1-1,j)*wi(i1))
        enddo

        qlm0=dble(log(q2ini/qcdlam))
        qlm=dble(log(q/qcdlam))
        qlmi=qlm-qlm0         !=log(q/q2ini)
        psuds=(qlm*log(qlm/qlm0)-qlmi)

        ffacs=(11.d0-dble(naflav)/1.5d0)/12.d0
        if(j.eq.1)then
          psuds=(psuds-ffacs*log(qlm/qlm0)
     *    +dps*(1.d0-dble(q2ini/q)))/ffacs
        else
          psuds=(psuds-log(qlm/qlm0)*.75d0
     *    +dps*(1.d0-dble(q2ini/q)))*4.d0/9.d0/ffacs
        endif
        psuds=exp(-psuds)
      else
        psuds=1.d0
      endif
      return
      end

c------------------------------------------------------------------------
      function psudx(q,j)
c-----------------------------------------------------------------------
c psudx - part of the bspacelike sudakov formfactor
c q - maximal value of the effective momentum,
c j - type of parton (1 - g, 2 - q)
c-----------------------------------------------------------------------
      common /ar3/    x1(7),a1(7)
      include 'epos.incsem'

      psudx=0.

      do i=1,7
      do m=1,2
        qt=.5*(q2ini+q-x1(i)*(2.*m-3.)*(q2ini-q))
        if(j.eq.1)then
          zm=1.-qt/q
          dps=((11.-naflav/1.5)/12.-zm**2*(1.-naflav/12.)+
     *    (zm**3/3.-zm**4/4.)*(1.-naflav/3.))*q/qt
        else
          dps=(1.-qt/q/4.)
        endif
        psudx=psudx+a1(i)*dps/log(qt/qcdlam)
      enddo
      enddo
      psudx=psudx*.5
      return
      end

c------------------------------------------------------------------------
      double precision function psutz(s,a,b)
c-----------------------------------------------------------------------
c psutz - kinematical function for two particle decay - light cone momen
c share for the particle of mass squared a,
c b - partner's mass squared,
c s - two particle invariant mass
c-----------------------------------------------------------------------
      double precision a1,b1,s1,x,dx,s,a,b
      a1=dsqrt(a)
      b1=dsqrt(b)
      s1=dsqrt(s)
      x=(1.d0+(a1-b1)*(a1+b1)/s)/2.d0
      dx=(x-a1/s1)*(x+a1/s1)
c      x=.5*(1.+(a-b)/s)
c      dx=(x*x-a/s)
      if(dx.gt.0.d0)then
        x=x+dsqrt(dx)
      else
        x=a1/s1
      endif
      psutz=min(0.999999999d0,x)
      return
      end

c------------------------------------------------------------------------
      block data ptdata
c-----------------------------------------------------------------------
c constants for numerical integration (gaussian weights)
c-----------------------------------------------------------------------
      common /ar3/ x1(7),a1(7)
      common /ar4/ x4(2),a4(2)
      common /ar5/ x5(2),a5(2)
      common /ar8/ x2(4),a2
      common /ar9/ x9(3),a9(3)

      data x1/.9862838,.9284349,.8272013,.6872929,.5152486,
     *.3191124,.1080549/
      data a1/.03511946,.08015809,.1215186,.1572032,
     *.1855384,.2051985,.2152639/
      data x2/.00960736,.0842652,.222215,.402455/
      data a2/.392699/
      data x4/ 0.339981,0.861136/
      data a4/ 0.652145,0.347855/
      data x5/.585786,3.41421/
      data a5/.853553,.146447/
      data x9/.93247,.661209,.238619/
      data a9/.171324,.360762,.467914/
      end

c------------------------------------------------------------------------
      subroutine strdo1(x,scale,upv,dnv,sea,str,chm,gl)
c------------------------------------------------------------------------
c :::::::::::: duke owens set 1 ::::::::::::::::::::::::::::
c------------------------------------------------------------------------
      implicit double precision(a-h,o-z)
      double precision
     +       f(5),a(6,5),b1(3,6,5)
      data q0,ql1/2.d0,.2d0/
      data b1/3.d0,0.d0,0.d0,.419d0,.004383d0,-.007412d0,
     &3.46d0,.72432d0,-.065998d0,4.4d0,-4.8644d0,1.3274d0,
     &6*0.d0,1.d0,
     &0.d0,0.d0,.763d0,-.23696d0,.025836d0,4.d0,.62664d0,-.019163d0,
     &0.d0,-.42068d0,.032809d0,6*0.d0,1.265d0,-1.1323d0,.29268d0,
     &0.d0,-.37162d0,-.028977d0,8.05d0,1.5877d0,-.15291d0,
     &0.d0,6.3059d0,-.27342d0,0.d0,-10.543d0,-3.1674d0,
     &0.d0,14.698d0,9.798d0,0.d0,.13479d0,-.074693d0,
     &-.0355d0,-.22237d0,-.057685d0,6.3494d0,3.2649d0,-.90945d0,
     &0.d0,-3.0331d0,1.5042d0,0.d0,17.431d0,-11.255d0,
     &0.d0,-17.861d0,15.571d0,1.564d0,-1.7112d0,.63751d0,
     &0.d0,-.94892d0,.32505d0,6.d0,1.4345d0,-1.0485d0,
     &9.d0,-7.1858d0,.25494d0,0.d0,-16.457d0,10.947d0,
     &0.d0,15.261d0,-10.085d0/
      wn=1.d0
      s= log( log( max(q0,scale)/ql1)/ log(q0/ql1))
      do 10 i=1,5
      do 10 j=1,6
   10 a(j,i)=b1(1,j,i)+s*(b1(2,j,i)+s*b1(3,j,i))
      do 40 i=1,5
   40 f(i)=a(1,i)*x**a(2,i)*(wn-x)**a(3,i)*(wn+x*
     &    (a(4,i)+x*(a(5,i)+x*a(6,i))))
      do 50 i=1,2
      aa=wn+a(2,i)+a(3,i)
   50 f(i)=f(i)*utgam2(aa)/((wn+a(2,i)*a(4,i)/aa)
     &*utgam2(a(2,i))*utgam2(wn+a(3,i)))
      upv=f(1)-f(2)
      dnv=f(2)
      sea=f(3)/6.d0
      str=sea
      chm=f(4)
      gl =f(5)
      return
      end



c------------------------------------------------------------------------
      function fzeroGluZZ(z,k)   ! former psftild
c-----------------------------------------------------------------------
c
c    fzeroGluZZComplete = fzeroGluZZ * z^(-1-dels) * gamsoft * gamhad
c
c  A = 8*pi*s0*gampar*gamtilde
c integration over semihard pomeron light cone momentum share xp==u
c
c fzeroGluZZ = (1-glusea) * engy^epszero
c  * int(du) u^(epszero-alppar+dels) (1-u)^alplea * (1-z/u)**betpom
c
c z - light cone x of the gluon,
c k - hadron class
c-----------------------------------------------------------------------
      double precision xpmin,xp
      include 'epos.inc'
      common /ar3/   x1(7),a1(7)
      include 'epos.incsem'

      fzeroGluZZ=0.
      xpmin=z
      xpmin=xpmin**(1.-alppar+dels+epszero)
      do i=1,7
      do m=1,2
        xp=(.5*(1.+xpmin+(2*m-3)*x1(i)*(1.-xpmin)))**(1./
     *  (1.-alppar+dels+epszero))
        fzeroGluZZ=fzeroGluZZ+a1(i)*(1.-xp)**alplea(k)*(1.-z/xp)**betpom
      enddo
      enddo
      fzeroGluZZ=
     *  fzeroGluZZ*.5*(1.-xpmin)/(1.-alppar+dels+epszero)
     *     *(1.-glusea)  *engy**epszero
      return
      end

c------------------------------------------------------------------------
      function fzeroSeaZZ(z,k)     ! former psftile
c-----------------------------------------------------------------------
c
c    fzeroSeaZZComplete = fzeroSeaZZ * z^(-1-dels) * gamsoft * gamhad
c
c  gamsoft = 8*pi*s0*gampar*gamtilde
c integration over semihard pomeron light cone momentum share xp==u
c
c fzeroSeaZZ = glusea * engy^epszero
c   * int(du) u^(epszero-alppar+dels) (1-u)^alplea * EsoftQZero(z/u)
c
c z - light cone x of the quark,
c k - hadron class
c-----------------------------------------------------------------------
      double precision xpmin,xp
      common /ar3/   x1(7),a1(7)
      include 'epos.inc'
      include 'epos.incsem'

      fzeroSeaZZ=0.
      xpmin=z
      xpmin=xpmin**(1.-alppar+dels+epszero)
      do i=1,7
      do m=1,2
        xp=(.5*(1.+xpmin+(2*m-3)*x1(i)*(1.-xpmin)))**(1./
     *  (1.-alppar+dels+epszero))
        zz=z/xp
        fzeroSeaZZ=fzeroSeaZZ+a1(i)*(1.-xp)**alplea(k)*EsoftQZero(zz)
      enddo
      enddo
      fzeroSeaZZ=fzeroSeaZZ*.5*(1.-xpmin)/(1.-alppar+dels+epszero)
     *     *glusea  *engy**epszero
      return
      end


c########################################################################
c########################################################################
      subroutine psaini
c########################################################################
c########################################################################

c-----------------------------------------------------------------------
c common initialization procedure
c if isetcs = 0, alpD, betD, etc ... in inirj are not used and xkappa=1
c if isetcs = 1, alpD, betD, etc ... in inirj are not used but xkappa.ne.1
c if isetcs = 2, alpD, betD, xkappa, etc ... in inirj are used and
c                cross section from calculation in inics are read.
c    if epos.inics doesn't exist, it produces only the calculated part of it.
c if isetcs = 3, alpD, betD, xkappa, etc ... in inirj are used and
c                cross section from simulation in inics are read.
c    if epos.inics doesn't exist, it produces the calculated AND the
c    simulated part of it both for ionudi=1 and 3. Only the values for
c    ionudi=1 (elastic for diffraction counted in xs) are always correct.
c    AA xs with ionudi=3 do not always correspond to MC simulations.
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incsem'
      include 'epos.incems'
      logical lcalc!,lcalc2
c      double precision om5p,xh,yh,v3pom(4),om2p
      dimension gamhad0(nclha),r2had0(nclha),chad0(nclha)
     *,alplea0(nclha),asect11(7,4,7),asect13(7,4,7),asect21(7,4,7)
     *,asect23(7,4,7),asect31(7,7,7),asect33(7,7,7)
     *,asect41(7,7,7),asect43(7,7,7)!,cgam(idxD)
      common /psar2/  edmax,epmax
      common /psar4/  fhgg(11,10,8),fhqg(11,10,80)
     *,fhgq(11,10,80),fhqq(11,10,80),fhgg0(11,10),fhgg1(11,10,4)
     *,fhqg1(11,10,40),fhgg01(11),fhgg02(11),fhgg11(11,4)
     *,fhgg12(11,4),fhqg11(11,10,4),fhqg12(11,10,4)
     *,ftoint(11,14,2,2,3)
      common /psar7/  delx,alam3p,gam3p
      common /psar9/  alpr
      common /psar15/ sudx(40,2)
      common /psar19/ cstot(20,20,240)
      common /psar20/ csord(20,20,240)
      common /psar21/ csbor(20,160,2)
      common /psar22/ cstotzero(20,4,2),csborzer(20,4,2)
      common /psar23/ cschar(20,20,2)
      common /psar25/ csdsi(21,21,104)
      common /psar27/ csds(21,26,4),csdt(21,26,2),csdr(21,26,2)
      common /psar33/ asect(7,4,7),asectn(7,7,7)
      common /psar34/ rrr,rrrm
      common /psar35/ anorm,anormp
      common /psar41/ rrrp,rrrmp
      common /psar36/ alvc
      common /psar37/ coefom1,coefom2
      common /psar38/ vfro(11,14,3,2)
      common /psar39/ vnorm(11,14,3,2,2)
c$$$      common /psar40/ coefxu1(idxD,nclha,10)
c$$$     *,coefxu2(idxD,idxD,nclha,10),coefxc2(idxD,idxD,nclha,10)
      common/producetab/ producetables              !used to link with CRMC
      logical producetables
      common /ar3/    x1(7),a1(7)
      common /testj/  ajeth(4),ajete(5),ajet0(7)
      parameter(nbkbin=40)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      common/geom/rmproj,rmtarg,bmax,bkmx
      character textini*38
      external ptfau,ptfauAA


      call utpri('psaini',ish,ishini,4)

c    for fragmentation
c    -----------------
c number of flavors in fragmentation not less than active flavor in hard string 
      nrflav=min(max(nrflav,naflav),nflavems)
      pmqu2=pmqu**2
      difud=pmqd**2-pmqu2
      difus=pmqs**2-pmqu2
      difuuu=(pmqq+pmqu+pmqu)**2-pmqu2
      difuud=(pudd*pmqq+pmqd+pmqu)**2-pmqu2
      difuus=(puds*pmqq+pmqs+pmqu)**2-pmqu2
      difudd=(pudd*pudd*pmqq+pmqd+pmqd)**2-pmqu2
      difuds=(pudd*puds*pmqq+pmqs+pmqd)**2-pmqu2
      difuss=(puds*puds*pmqq+pmqs+pmqs)**2-pmqu2
      if(nrflav.gt.3)then
        difuc=pmqc**2-pmqu2
        difuuc=(pudc*pmqq+pmqc+pmqu)**2-pmqu2
        difudc=(pudd*pudc*pmqq+pmqc+pmqd)**2-pmqu2
        difusc=(puds*pudc*pmqq+pmqc+pmqs)**2-pmqu2
        difucc=(pudc*pudc*pmqq+pmqc+pmqs)**2-pmqu2
      else
        difuc=0.
        difuuc=0.
        difudc=0.
        difusc=0.
        difucc=0.
        rstrac(1)=0.
        rstrac(2)=0.
        rstrac(3)=0.
        rstrac(4)=0.
      endif

      if(iappl.ne.6)then

      do i=1,4
      ajeth(i)=0.
      enddo
      do i=1,5
      ajete(i)=0.
      ajet0(i)=0.
      enddo
      ajet0(6)=0.
      ajet0(7)=0.


      if(isetcs.le.1)then              !for Kfit
        bkbin=0.3
      else
        bkbin=0.1
      endif
      xkappa=1.

      edmax=edmaxi  !1.e12     defined in epos-bas
      epmax=epmaxi  !1.e12     defined in epos-bas

c fix enhanced diagrams at minimum energy = 2.5
      delx=1.5 !sqrt(egymin*egymin/exp(1.))
c arbitrary value for alam3p (not good if too small (infinite loop in rsh))
      alam3p=0.5*(r2had(1)+r2had(2)+r2had(3)) !0.6
      gam3p=.1



c   interface to 'bas'
c    ----------------

      dels=alppom-1.
      alpqua=(alppar+1.)/2.
      if(abs(alpqua).lt.1.e-6)call utstop('alpar should not be -1 !&',
     +sizeof('alpar should not be -1 !&'))
      alpr=-2.+alpqua      !x-exponent for remnant mass


c   omega coeffs
c    ----------------
      coefom0=utgam1(1.+dels-alppar)*utgam1(1.+alplea(iclpro))
     */utgam1(2.+alplea(iclpro)+dels-alppar)
     **utgam1(1.+dels-alppar)*utgam1(1.+alplea(icltar))
     */utgam1(2.+alplea(icltar)+dels-alppar)
      coefom1=1.-utgam1(1.+dels-alppar)**2*utgam1(1.+alplea(iclpro))
     */utgam1(1.+alplea(iclpro)+2.*(1.+dels-alppar))
     **utgam1(1.+dels-alppar)**2*utgam1(1.+alplea(icltar))
     */utgam1(1.+alplea(icltar)+2.*(1.+dels-alppar))/coefom0**2
      coefom2=3.*coefom1-1.
     *+utgam1(1.+dels-alppar)**3*utgam1(1.+alplea(iclpro))
     */utgam1(1.+alplea(iclpro)+3.*(1.+dels-alppar))
     **utgam1(1.+dels-alppar)**3*utgam1(1.+alplea(icltar))
     */utgam1(1.+alplea(icltar)+3.*(1.+dels-alppar))/coefom0**3
      if(ish.ge.4)write(ifch,*)'coefom',coefom0,coefom1,coefom2,delx

c soft pomeron: abbreviations
c---------------------------------------
      if(iappl.eq.1.or.iappl.eq.8.or.iappl.eq.9)then


c---------------------------------------
c auxiliary constants:
c---------------------------------------
        stmass=.05               !string mass cutoff

c---------------------------------------
c parton density normalization
        sq=log(log(q2min/.232**2)/log(.23/.232**2))
        du=2.997+.753*sq-.076*sq*sq
        qnorm=0.
        do i=1,7
        do m=1,2
          xx=.5+x1(i)*(m-1.5)
          xxq=1.-xx**(1./(1.+du))
          qnorm=qnorm+a1(i)*(psdfh4(xxq,q2min,0.,2,1)+
     *    psdfh4(xxq,q2min,0.,2,2))/(1.-xxq)**du
        enddo
        enddo
        qnorm=qnorm*.5/(1.+du)
        qnormp=qnorm
ckkkkk-----------------------------
c        ffrr=(1.-qnorm)/4./pi/gamhad(2)
c     *  *utgam1(2.+betpom-dels)/utgam1(1.-dels)
c     *  /utgam1(1.+betpom)/utgam1(1.+alplea(2))/
c     *  utgam1(2.-alppar)*utgam1(3.+alplea(2)-alppar)
c      ffrr=(1.-qnorm)/4./pi/gamhad(2)
c     *  *utgam1(2.+betpom-dels)/utgam1(1.-dels)
c     *  /utgam1(1.+betpom)
c      write(6,*)'===========',ffrr
        ffrr=gamtil
     *  /utgam1(1.+alplea(2))/
     *  utgam1(2.-alppar)*utgam1(3.+alplea(2)-alppar)
      gamsoft=ffrr*4.*pi
ckkkkkkk-------------------------------
        if(ish.ge.4)write (ifch,*)'rr,qnorm',ffrr,qnorm


        sq=log(log(q2min/.232**2)/log(.25/.232**2))
        dpi=.367+.563*sq
        qnorm=0.
        do i=1,7
        do m=1,2
          xx=.5+x1(i)*(m-1.5)
          xxq=1.-xx**(1./(1.+dpi))
          qnorm=qnorm+a1(i)*(psdfh4(xxq,q2min,0.,1,1)+
     *    psdfh4(xxq,q2min,0.,1,2))/(1.-xxq)**dpi
        enddo
        enddo
        qnorm=qnorm*.5/(1.+dpi)
        cftmp=1./(1.-qnormp)*(1.-qnorm)
     *  *utgam1(alplea(2)+1.)/utgam1(alplea(2)+3.-alppar)
     *  /utgam1(alplea(1)+1.)*utgam1(alplea(1)+3.-alppar)
        gamhad(1)=gamhad(2)*cftmp
        if(gamhadsi(1).lt.0.)then
          gamhads(1)=gamhad(1)
        else
          gamhads(1)=gamhad(1)*gamhadsi(1)
        endif
        gamhad(1)=gamhads(1)
        if(ish.ge.4)
     *  write (ifch,*)'gamhad(1),gamhads(1)',gamhad(1),gamhads(1)

        if(gamhadsi(2).lt.0.)then
          gamhads(2)=gamhad(2)
        else
          gamhads(2)=gamhad(2)*gamhadsi(2)
        endif
        gamhad(2)=gamhads(2)
        if(ish.ge.4)
     *  write (ifch,*)'gamhad(2),gamhads(2)',gamhad(2),gamhads(2)

        qnorm=0.
        do i=1,7
        do m=1,2
          xx=.5+x1(i)*(m-1.5)
          xxq=1.-xx**(1./(1.+dpi))
          qnorm=qnorm+a1(i)*(psdfh4(xxq,q2min,0.,1,1)+
     *    psdfh4(xxq,q2min,0.,1,2))/(1.-xxq)**dpi
        enddo
        enddo
        qnorm=qnorm*.5/(1.+dpi)
        cftmp=1./(1.-qnormp)*(1.-qnorm)
     *  *utgam1(alplea(2)+1.)/utgam1(alplea(2)+3.-alppar)
     *  /utgam1(alplea(3)+1.)*utgam1(alplea(3)+3.-alppar)
        gamhad(3)=gamhad(2)*cftmp
        if(gamhadsi(3).lt.0.)then
          gamhads(3)=gamhad(3)
        else
          gamhads(3)=gamhad(3)*gamhadsi(3)
        endif
        gamhad(3)=gamhads(3)
        if(ish.ge.4)
     *  write (ifch,*)'gamhad(3),gamhads(3)',gamhad(3),gamhads(3)

        quamas=.35
        gamhad(4)=gamhad(1)*(quamas/qcmass)**2
        if(gamhadsi(4).lt.0.)then
          gamhads(4)=gamhad(4)
        else
          gamhads(4)=gamhad(4)*gamhadsi(4)
        endif
        gamhad(4)=gamhads(4)
        if(ish.ge.4)
     *  write (ifch,*)'gamhad(4),gamhads(4)',gamhad(4),gamhads(4)
        gnorm=0.
        do i=1,7
        do m=1,2
          xx=.5+x1(i)*(m-1.5)
          xxg=xx**(1./(1.-dels))
          gnorm=gnorm+a1(i)*(fzeroGluZZ(xxg,4)+fzeroSeaZZ(xxg,4))
        enddo
        enddo
        gnorm=gnorm/(1.-dels)*2.*pi*gamhad(4)*ffrr
        alvc=6./(1.-gnorm)-4.
        if(ish.ge.4) write (ifch,*)'rr,qnorm,gnorm,alvc',
     *  ffrr,qnorm,gnorm,alvc

c        write (*,*)'rr-c,qnorm,gnorm,alvc',ffrr,qnorm,gnorm,alvc
      endif

c-----------------------------------------------
c tabulation of inclusive jet cross sections
c--------------------------------------------------

      do i=1,40
        qi=q2min*exp(.5*(i-1))
        sudx(i,1)=psudx(qi,1)
        sudx(i,2)=psudx(qi,2)
      enddo
      if(ish.ge.4)write(ifch,*)'bare cross sections ...'

      call psaevc

ccc      call MakeCSTable

      inquire(file=fnii(1:nfnii),exist=lcalc)
      if(lcalc)then
       if(inicnt.eq.1)then
        write(ifmt,'(3a)')'read from ',fnii(1:nfnii),' ...'
        open(1,file=fnii(1:nfnii),status='old')
        read (1,*)qcdlam0,q2min0,q2ini0,naflav0,epmax0,pt2cut0
        if(qcdlam0.ne.qcdlam)write(ifmt,'(a)')'initl: wrong qcdlam'
        if(q2min0 .ne.q2min )write(ifmt,'(a)')'initl: wrong q2min'
        if(q2ini0 .ne.q2ini )write(ifmt,'(a)')'initl: wrong q2ini'
        if(naflav0.ne.naflav)write(ifmt,'(a)')'initl: wrong naflav'
        if(epmax0 .ne.epmax )write(ifmt,'(a)')'initl: wrong epmax'
        if(pt2cut0 .ne.pt2cut )write(ifmt,'(a)')'initl: wrong pt2cut'
        if(qcdlam0.ne.qcdlam.or.q2min0 .ne.q2min .or.q2ini0 .ne.q2ini
     *  .or.naflav0.ne.naflav.or.epmax0 .ne.epmax.or. pt2cut.ne.pt2cut0)
     *  then
          write(ifmt,'(//a//)')'   initl has to be reinitialized!!!'
          stop
        endif
        read (1,*)csbor,csord,cstot,cstotzero,csborzer
        close(1)
       endif

       goto 1

      elseif(.not.producetables)then
        write(ifmt,*) "Missing epos.initl file !"        
        write(ifmt,*) "Please correct the defined path ",
     &"or force production ..."
        stop

      endif

      write(ifmt,'(a)')'initl does not exist -> calculate tables  ...'

      write (*,*)'Born xsection csbor'
      spmin=4.*q2min
      spminc=4.*q2min+qcmass**2
      do m=1,4   !parton type at upper end of the ladder (1...4 - g,u,d,c)
      do k=1,20
        if(m.ne.4)then
          sk=spmin*(epmax/2./spmin)**((k-1)/19.)
          p1=sk
        else
          sk=spminc*(epmax/2./spminc)**((k-1)/19.)
          p1=sk/(1.+qcmass**2/sk)
        endif
        qmax=p1/4.
      do i=1,20
        qq=q2min*(qmax/q2min)**((i-1)/19.)
      do l=1,2    !parton type at lower end of the ladder
        k1=k+20*(m-1)+80*(l-1)
        m1=m-1
        if(m.eq.3.and.l.eq.1)then  !dd~
          l1=-m1
        else                       !du
          l1=l-1
        endif                                       !born cr.-sect.
        csbor(i,k1,1)=log(max(1.e-30,psborn(qq,qq,qq,sk,m1,l1,0,0)))
        if(m.ne.4)then
        csbor(i,k1,2)=log(max(1.e-30,psborn(4.*qq,qq,qq,sk,m1,l1,1,0)))
        endif
      enddo
      enddo
      enddo
      enddo

      write (*,*)'ordered jet xsection csord'
      do m=1,4            !parton type at upper end of the ladder
      do k=1,20
        write (*,*)'   m=',m,'/4  k=',k,'/20'
        if(m.ne.4)then
          sk=spmin*(epmax/2./spmin)**((k-1)/19.)  !c.m. energy squared for the hard
          p1=sk
        else
          sk=spminc*(epmax/2./spminc)**((k-1)/19.)
          p1=sk/(1.+qcmass**2/sk)
        endif
        qmax=p1/4.
        tmax=p1/2.
      do i=1,20             !cross-sections initialization
        qi=q2min*(qmax/q2min)**((i-1)/19.)
      do j=1,20
        qq=qi*(qmax/qi)**((j-1)/19.)
        if(p1.gt.4.*qq)then
          tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
        else
          tmin=2.*qq
        endif
      do l=1,2              !parton type at lower end of the ladder
        m1=m-1
        if(m.eq.3.and.l.eq.1)then
          l1=-m1
        else
          l1=l-1
        endif
        if(m.ne.4)then
          k1=k+20*(m-1)+60*(l-1)
          if(k.eq.1.or.i.eq.20.or.j.eq.20)then
            csord(i,j,k1)=log(max(1.e-30,psborn(qi,qq,qq,sk,m1,l1,0,0)))
            csord(i,j,k1+120)=
     *                 log(max(1.e-30,psborn(4.*qq,qi,qq,sk,l1,m1,1,0)))
          else
            csord(i,j,k1)=log(psjet1(qi,qq,qq,sk,m1,l1,0)
     *      /(1./tmin-1./tmax)+psborn(qi,qq,qq,sk,m1,l1,0,0))
            csord(i,j,k1+120)=log(psjet1(qi,4.*qq,qq,sk,m1,l1,2)
     *      /(1./tmin-1./tmax)+psborn(4.*qq,qi,qq,sk,l1,m1,1,0))

          endif
        elseif(j.eq.1)then
          if(k.eq.1.or.i.eq.20)then
         cschar(i,k,l)=log(max(1.e-30,psborn(q2min,qi,qq,sk,m1,l1,0,0)))
          else
            cschar(i,k,l)=log(psjet1(qi,q2min,qq,sk,l1,m1,0)
     *      /(1./tmin-1./tmax)+psborn(q2min,qi,qq,sk,m1,l1,0,0))
          endif
        endif
      enddo
      enddo
      enddo
      enddo
      enddo

      write (ifmt,*)'tests:'
      write (ifmt,'(a,a)')' n-1      sk       qi       qj       qq  '
     * ,'      born   born-i      ord    ord-i  '
      do k=1,7
          sk=spmin*(epmax/2./spmin)**((k-1)/19.)
          if(k.ge.5)sk=spmin*1.5**(k-4)
      do n=1,2
        if(n.eq.1)then
          qmax1=sk/4.
          qmax2=sk/4.
        else             !if(n.eq.2)then
          qmax1=sk/4.
          qmax2=sk
        endif
      do i=1,3
        qi=q2min*(qmax1/q2min)**((i-1)/3.)
      do j=1,3
        qj=q2min*(qmax2/q2min)**((j-1)/3.)
        qqmax=sk/4.
        if(n.eq.1)then
          qqmin=max(qi,qj)
        else
          qqmin=max(qi,qj/4.)
        endif
      do lq=1,3
        qq=qqmin*(qqmax/qqmin)**((lq-1)/3.)
        if(sk.gt.4.*qq)then
          tmin=2.*qq/(1.+sqrt(1.-4.*qq/sk))
        else
          tmin=2.*qq
        endif
        tmax=sk/2.
      do m=1,1             !parton type at upper end of the ladder (1
      do l=1,1              !parton type at lower end of the ladder (1
        m1=m-1
        if(m.eq.3.and.l.eq.1)then
          l1=-m1
        else
          l1=l-1
        endif
       a=psborn(qj,qi,qq,sk,l1,m1,n-1,0)*(1./tmin-1./tmax)
       b=psbint(qj,qi,qq,sk,l1,m1,n-1)
       c=psjet1(qi,qj,qq,sk,m1,l1,2*(n-1))
     *    +psborn(qj,qi,qq,sk,l1,m1,n-1,0)*(1./tmin-1./tmax)
       d=psjti1(qi,qj,qq,sk,m1,l1,n-1)
       write (ifmt,'(i3,4f9.1,3x,4f9.4)')n-1,sk,qi,qj,qq,a,b,c,d
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo

      write (*,*)'jet xsection cstot'
      do k=1,20
        write (*,*)'k=',k,'/20'
        sk=spmin*(epmax/2./spmin)**((k-1)/19.)  !c.m. energy squared for the hard
        qmax=sk/4.
        tmax=sk/2.
      do i=1,20             !cross-sections initialization
      do n=1,2
        if(n.eq.1)then
          qi=q2min*(qmax/q2min)**((i-1)/19.)
        else
          qi=q2min*(4.*qmax/q2min)**((i-1)/19.)
        endif
      do j=1,20
        if(n.eq.1)then
          qq=qi*(qmax/qi)**((j-1)/19.)
        else
          qq=max(q2min,qi/4.)*(qmax/max(q2min,qi/4.))**
     *    ((j-1)/19.)
        endif
        if(sk.gt.4.*qq)then
          tmin=2.*qq/(1.+sqrt(1.-4.*qq/sk))
        else
          tmin=2.*qq
        endif
      do m=1,3              !parton type at upper end of the ladder (1
      do l=1,2              !parton type at lower end of the ladder (1
        m1=m-1
        if(m.eq.3.and.l.eq.1)then
          l1=-m1
        else
          l1=l-1
        endif
        k1=k+20*(m-1)+60*(l-1)+120*(n-1)
        if(k.eq.1.or.i.eq.20.or.j.eq.20)then
       cstot(i,j,k1)=log(max(1.e-30,psborn(qi,q2min,qq,sk,m1,l1,n-1,0)))
        else
          if(n.eq.1)then
            cstot(i,j,k1)=log((psjet(qi,q2min,qq,sk,m1,l1,0)+
     *      psjti1(qi,q2min,qq,sk,m1,l1,0)+
     *      psjti1(q2min,qi,qq,sk,l1,m1,0)
     *      -psbint(qi,q2min,qq,sk,m1,l1,0))/(1./tmin-1./tmax))
          else
            cstot(i,j,k1)=log((psjet(qi,q2min,qq,sk,m1,l1,1)+
     *      psjet1(qi,q2min,qq,sk,m1,l1,1)+
     *      psjti1(q2min,qi,qq,sk,l1,m1,1))/(1./tmin-1./tmax))
          endif
        endif
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo

c total and born hard cross-sections logarithms for minimal cutoff
c (q2min), interpolated in the psjti0 procedure
      spmin=4.*q2min
      spminc=4.*q2min+qcmass**2
      do m=1,4
      do l=1,2
        m1=m-1
        if(m.eq.3.and.l.eq.1)then
          l1=-m1
        else
          l1=l-1
        endif
      do k=1,20
        if(m.ne.4)then
          sk=spmin*(epmax/2./spmin)**((k-1)/19.)  !c.m. energy squared for the hard
          p1=sk
          qq=q2min
        else
          sk=spminc*(epmax/2./spminc)**((k-1)/19.)
          p1=sk/(1.+qcmass**2/sk)
          qq=q2min
        endif
        if(p1.gt.4.*qq)then
          tmin=2.*qq/(1.+sqrt(1.-4.*qq/p1))
        else
          tmin=2.*qq
        endif
        tmax=p1/2.

        k1=k+20*(m-1)+80*(l-1)
        csborzer(k,m,l)
     *      =log(max(1.e-30,psborn(q2min,q2min,qq,sk,m1,l1,0,0)))
        if(k.eq.1)then
          cstotzero(k,m,l)=csborzer(k,m,l)
        elseif(m.ne.4)then
          cstotzero(k,m,l)=log(psjti(q2min,qq,sk,m1,l1,0)/
     *    (1./tmin-1./tmax))
        else
          smins=2.5*q2min*(1.+sqrt(1.+4.*qcmass**2/q2min))
          if(sk.le.smins)then
            cstotzero(k,m,l)=log(psjci(q2min,sk,l1)/(1./tmin-1./tmax))
          else
            cstotzero(k,m,l)=log((psjci(q2min,sk,l1)+psjct(sk,l1))
     *      /(1./tmin-1./tmax))
          endif
        endif
      enddo
      enddo
      enddo

      write(ifmt,'(a)')'write to initl ...'
      open(1,file=fnii(1:nfnii),status='unknown')
      write (1,*)qcdlam,q2min,q2ini,naflav,epmax,pt2cut
      write (1,*)csbor,csord,cstot,cstotzero,csborzer,cschar
      close(1)

1     continue

      if(iappl.ne.8)goto 3
      if(ish.ge.3)write(ifch,*)'dis cross sections ...'
      inquire(file=fnid(1:nfnid),exist=lcalc)
      if(lcalc)then
       if(inicnt.eq.1)then
        write(ifmt,'(3a)')'read from ',fnid(1:nfnid),' ...'
        open(1,file=fnid(1:nfnid),status='old')
        read (1,*)qcdlam0,q2min0,q2ini0,naflav0,epmax0,edmax0
        if(qcdlam0.ne.qcdlam)write(ifmt,'(a)')'inidi: wrong qcdlam'
        if(q2min0 .ne.q2min )write(ifmt,'(a)')'inidi: wrong q2min'
        if(q2ini0 .ne.q2ini )write(ifmt,'(a)')'inidi: wrong q2ini'
        if(naflav0.ne.naflav)write(ifmt,'(a)')'inidi: wrong naflav'
        if(epmax0 .ne.epmax )write(ifmt,'(a)')'inidi: wrong epmax'
        if(edmax0 .ne.edmax )write(ifmt,'(a)')'inidi: wrong edmax'
        if(qcdlam0.ne.qcdlam.or.q2min0 .ne.q2min.or.q2ini0 .ne.q2ini
     *  .or.naflav0.ne.naflav.or.epmax0 .ne.epmax
     *  .or.edmax0 .ne.edmax)then
           write(ifmt,'(//a//)')'   inidi has to be reinitialized!!!'
           stop
        endif
        read (1,*)csdsi,csds,csdt,csdr
        close(1)
       endif
       goto 3

      elseif(.not.producetables)then
        write(ifmt,*) "Missing epos.inidi file !"        
        write(ifmt,*) "Please correct the defined path ",
     &"or force production ..."
        stop

      endif

      write(ifmt,'(a)')'inidi does not exist -> calculate tables  ...'
      do j=1,21
        qq=q2min*exp(.5*(j-1))                !photon virtuality

        do m=1,2               !parton type at the end of the ladder
          q2mass=qcmass**2
          s2min=4.*max(q2mass,q2min)+qq
          if(m.eq.2)s2min=s2min/(1.-4.*q2ini/(s2min-qq))
        do k=1,26
          write (*,*)'sin,j,m,k',j,m,k
          sk=s2min*(edmax/s2min)**(.04*(k-1))      !c.m. energy squared
          if(k.eq.26)sk=1.01*sk
          qmin=q2min
          if(m.eq.1)then
            qmax=(sk-qq)/4.
          else
            qmax=(sk-qq+sqrt((sk-qq)**2-16.*sk*q2ini))/8.
          endif

          do i=1,21               !cross-sections calculation
            qi=qmin*(qmax/qmin)**((i-1)/20.)
            tmax=.5*sk
            qtq=4.*max(q2mass,qi)/(sk-qq)
            if(qtq.lt.1.)then
              tmin=.5*sk*qtq/(1.+sqrt(1.-qtq))
            else
              tmin=.5*sk
            endif

            do ilong=1,2
              k1=k+26*(m-1)+52*(ilong-1)
              if(m.eq.1)then
                if(tmax.gt.1.01*tmin)then
                  sij=psds(qi,qq,sk,m-1,ilong-1)
                  if(sij.lt.0.)write (*,*)'qi,qq,sk,m,long,sij',
     *            qi,qq,sk,m,ilong,sij
                  csdsi(i,j,k1)=log(max(0.,sij)/(1./tmin-1./tmax)
     *            +psdbor(qi,qq,sk,ilong-1))
                else
                  csdsi(i,j,k1)=
     *            log(max(1.e-25,psdbor(qi,qq,sk,ilong-1)))
                endif
              else
                csdsi(i,j,k1)=psds(qi,qq,sk,m-1,ilong-1)
              endif
            enddo
          enddo
        enddo
        enddo
      enddo

      do j=1,21
        qq=q2min*exp(.5*(j-1))                       !photon virtuality
        s2min=max(4.*qq,16.*q2min)    !pt2dis=qq
      do m=1,2
      do k=1,26
        do ilong=1,2
          k1=k+26*(m-1)+52*(ilong-1)
          csds(j,k,m+2*(ilong-1))=csdsi(1,j,k1)
        enddo

        sk=(s2min+qq)*(edmax/(s2min+qq))**(.04*(k-1))
        csdt(j,k,m)=psdres(qq,sk,s2min,m-1)
        csdr(j,k,m)=psdrga(qq,sk-qq,s2min,m-1)
      enddo
      enddo
      enddo

      write(ifmt,'(a)')'write to inidi ...'

      write(ifmt,'(a)')'write to inidi ...'
      open(1,file=fnid(1:nfnid),status='unknown')
      write (1,*)qcdlam,q2min,q2ini,naflav,epmax,edmax
      write (1,*)csdsi,csds,csdt,csdr
      close(1)
3     continue

c---------------------------------------
c tabulation of semihard eikonals
c---------------------------------------

!!!!!!!!!      if(iappl.eq.1)then

      if(ish.ge.4)write(ifch,*)'semihard eikonals ...'
5     continue
      inquire(file=fnrj,exist=lcalc)
      if(lcalc)then
       if(inicnt.eq.1)then
        write(ifmt,'(3a)')'read from ',fnrj(1:nfnrj),' ...'
        open(1,file=fnrj(1:nfnrj),status='old')
        read (1,*)alpqua0,alplea0,alppom0,slopom0,
     *  gamhad0,r2had0,chad0,
     *  qcdlam0,q2min0,q2ini0,betpom0,glusea0,naflav0,
     *  factk0,pt2cut0,gamtil0
        if(alpqua0.ne.alpqua)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong alpqua',alpqua0,alpqua
        if(alppom0.ne.alppom)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong alppom',alppom0,alppom
        if(slopom0.ne.slopom)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong slopom',slopom0,slopom
        iii=2
        if(gamhad0(iii).ne.gamhad(iii))write(ifmt,'(a,i1,a,2f8.4)')
     *  'inirj: wrong gamhad(',iii,')',gamhad0(iii),gamhad(iii)
        do iii=1,3
        if(r2had0(iii) .ne.r2had(iii) )write(ifmt,'(a,i1,a,2f8.4)')
     *  'inirj: wrong r2had(',iii,')',r2had0(iii),r2had(iii)
        if(chad0(iii)  .ne.chad(iii)  )write(ifmt,'(a,i1,a,2f8.4)')
     *  'inirj: wrong chad(',iii,')',chad0(iii),chad(iii)
        if(alplea0(iii).ne.alplea0(iii))write(ifmt,'(a,i1,a,2f8.4)')
     *  'inirj: wrong alplea(',iii,')',alplea0(iii),alplea(iii)
        enddo
        if(qcdlam0.ne.qcdlam)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong qcdlam',qcdlam0,qcdlam
        if(q2min0 .ne.q2min )write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong q2min',q2min0,q2min
        if(q2ini0 .ne.q2ini )write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong q2ini',q2ini0,q2ini
        if(betpom0.ne.betpom)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong betpom',betpom0,betpom
        if(glusea0.ne.glusea)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong glusea',glusea0,glusea
        if(naflav0.ne.naflav)write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong naflav',naflav0,naflav
        if(factk0 .ne.factk )write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong factk', factk0,factk
        if(pt2cut0 .ne.pt2cut )write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong pt2cut', pt2cut0,pt2cut
        if(gamtil0 .ne.gamtil )write(ifmt,'(a,2f8.4)')
     *  'inirj: wrong gamtil', gamtil0,gamtil
        if(alpqua0.ne.alpqua.or.alppom0.ne.alppom
     *  .or.slopom0.ne.slopom.or.gamhad0(2).ne.gamhad(2)
     *  .or.r2had0(1).ne.r2had(1).or.r2had0(2).ne.r2had(2)
     *  .or.r2had0(3).ne.r2had(3)
     *  .or.chad0(1).ne.chad(1).or.chad0(2).ne.chad(2)
     *  .or.chad0(3).ne.chad(3)
     *  .or.alplea0(1).ne.alplea(1).or.alplea0(2).ne.alplea(2)
     *  .or.alplea0(3).ne.alplea(3)
     *  .or.qcdlam0.ne.qcdlam.or.q2min0 .ne.q2min
     *  .or.q2ini0 .ne.q2ini.or.gamtil0.ne.gamtil
     *  .or.betpom0.ne.betpom.or.glusea0.ne.glusea.or.naflav0.ne.naflav
     *  .or.factk0 .ne.factk .or.pt2cut0.ne.pt2cut)then
           write(ifmt,'(//a//)')'   inirj has to be reinitialized!!!!'
           stop
        endif

        read(1,*)fhgg,fhqg,fhgq,fhqq,fhgg0,fhgg1,fhqg1
     *  ,fhgg01,fhgg02,fhgg11,fhgg12,fhqg11,fhqg12
     *  ,ftoint,vfro,vnorm,coefxu1,coefxu2,coefxc2
        read(1,*)bkbin0,iclpro10,iclpro20,icltar10,icltar20,iclegy10
     *   ,iclegy20,egylow0,egymax0,iomega0,egyscr0,epscrw0,epscrp0
        if(isetcs.gt.1)then
        textini='                                      '
        if(iclpro10.ne.iclpro1)write(textini,'(a,2i8)')
     *  'inirj: wrong iclpro1  ',iclpro10,iclpro1
        if(iclpro20.ne.iclpro2)write(textini,'(a,2i8)')
     *  'inirj: wrong iclpro2  ',iclpro20,iclpro2
        if(icltar10.ne.icltar1)write(textini,'(a,2i8)')
     *  'inirj: wrong icltar1  ',icltar10,icltar1
        if(icltar20.ne.icltar2)write(textini,'(a,2i8)')
     *  'inirj: wrong icltar2  ',icltar20,icltar2
        if(iclegy10.ne.iclegy1)write(textini,'(a,2i8)')
     *  'inirj: wrong iclegy1  ',iclegy10,iclegy1
        if(iclegy20.ne.iclegy2)write(textini,'(a,2i8)')
     *  'inirj: wrong iclegy2  ',iclegy20,iclegy2
        if(iomega0.ne.iomega)write(textini,'(a,2i8)')
     *  'inirj: wrong iomega   ',iomega0,iomega
        if(egylow0.ne.egylow)write(textini,'(a,2f8.4)')
     *  'inirj: wrong egylow   ',egylow0,egylow
        if(egymax0.ne.egymax)write(textini,'(a,2f8.4)')
     *  'inirj: wrong egymax   ',egymax0,egymax
        if(epscrw0.ne.epscrw)write(textini,'(a,2f8.4)')
     *  'inirj: wrong epscrw    ',epscrw0,epscrw
        if(epscrp0.ne.epscrp)write(textini,'(a,2f8.4)')
     *  'inirj: wrong epscrp   ',epscrp0,epscrp
        if(bkbin0.ne.bkbin)write(textini,'(a,2f8.4)')
     *  'inirj: wrong bkbin',bkbin0,bkbin
        if(textini.ne.'                                      ')then
           write(ifmt,'(//10x,a//10x,a//)')textini,
     *     'inirj has to be reinitialized!!!!'
           stop
        endif
        do iiipro=iclpro1,iclpro2
        do iiitar=icltar1,icltar2
        do iiiegy=iclegy1,iclegy2
        do iiib=1,nbkbin
          read(1,*)xkappafit(iiiegy,iiipro,iiitar,iiib)
        enddo
        xkappafit(iiiegy,iiipro,iiitar,nbkbin)=1.
        do iiib=2,nbkbin-1
          if(xkappafit(iiiegy,iiipro,iiitar,iiib).lt.1.)then
            xkappafit(iiiegy,iiipro,iiitar,iiib)=max(1.,0.5*
     *        (xkappafit(iiiegy,iiipro,iiitar,iiib-1)
     *        +xkappafit(iiiegy,iiipro,iiitar,iiib+1)))
          endif
        enddo
        do iiidf=idxD0,idxD
         read(1,*)alpDs(iiidf,iiiegy,iiipro,iiitar),
     *   alpDps(iiidf,iiiegy,iiipro,iiitar),
     *   alpDpps(iiidf,iiiegy,iiipro,iiitar),
     *   betDs(iiidf,iiiegy,iiipro,iiitar),
     *   betDps(iiidf,iiiegy,iiipro,iiitar),
     *   betDpps(iiidf,iiiegy,iiipro,iiitar),
     *   gamDs(iiidf,iiiegy,iiipro,iiitar),
     *   delDs(iiidf,iiiegy,iiipro,iiitar)
        enddo
        enddo
        enddo
        enddo
      endif

        close(1)

      endif


        goto 4

      elseif(.not.producetables)then
        write(ifmt,*) "Missing epos.inirj file !"        
        write(ifmt,*) "Please correct the defined path ",
     &"or force production ..."
        stop

      endif

      write(ifmt,'(a)')'inirj does not exist -> calculate tables  ...'

      engysave=engy
      maprojsave=maproj
      matargsave=matarg
      iclpros=iclpro
      icltars=icltar
      spmin=4.*q2min
      spminc=4.*q2min+2.*qcmass**2
      icltar=2

      write(ifmt,'(a)')'  tabulate om5 ...'

      do iy=1,11
        sy=spmin*(epmax/2./spmin)**((iy-1)/10.)
        syc=spminc*(epmax/2./spminc)**((iy-1)/10.)
        iclpro=2
        icltar=2
        if(iy.eq.1)then
          fhgg01(iy)=-80.
          fhgg02(iy)=-80.
        else
          fhgg01(iy)=log(om51pp(sy,1.,1.,3))
          fhgg02(iy)=log(om51pp(sy,1.,1.,7))
        endif

        do iclpro=iclpro1,iclpro2
          if(iy.eq.1)then
            fhgg11(iy,iclpro)=-80.
            fhgg12(iy,iclpro)=-80.
          else
            fhgg11(iy,iclpro)=log(om51pp(sy,1.,1.,4))
            fhgg12(iy,iclpro)=log(om51pp(sy,1.,1.,9))
          endif
          do ix=1,10
            if(ix.le.5)then
              xp=.1*2.**(ix-5)
            else
              xp=.2*(ix-5)
            endif
            if(iy.eq.1)then
              fhqg11(iy,ix,iclpro)=-80.
              fhqg12(iy,ix,iclpro)=-80.
            elseif(iclpro.eq.4)then
              fhqg11(iy,ix,iclpro)=log(om51pp(syc,1.,1.,5))
              fhqg12(iy,ix,iclpro)=log(om51pp(syc,1.,1.,11))
            else
              fhqg11(iy,ix,iclpro)=log(om51pp(sy,xp,1.,5))
              fhqg12(iy,ix,iclpro)=log(om51pp(sy,xp,1.,11))
            endif
          enddo
        enddo

      do iz=1,10
        z=.1*iz

        iclpro=2
        icltar=2
        if(iy.eq.1)then
          fhgg0(iy,iz)=-80.
        else
          fhgg0(iy,iz)=log(om51pp(sy,1.,z,6)/z)
        endif

        do iclpro=iclpro1,iclpro2
          if(iy.eq.1)then
            fhgg1(iy,iz,iclpro)=-80.
          else
            fhgg1(iy,iz,iclpro)=log(om51pp(sy,1.,z,8)/z)
          endif

          do ix=1,10
            if(ix.le.5)then
              xp=.1*2.**(ix-5)
            else
              xp=.2*(ix-5)
            endif
            if(iy.eq.1)then
              fhqg1(iy,ix,iz+10*(iclpro-1))=-80.
            elseif(iclpro.eq.4)then
              fhqg1(iy,ix,iz+10*(iclpro-1))=log(om51pp(syc,xp,z,10)/z)
            else
              fhqg1(iy,ix,iz+10*(iclpro-1))=log(om51pp(sy,xp,z,10)/z)
            endif
          enddo
        enddo
      enddo
      enddo

      do iclpro=iclpro1,iclpro2 !hadron type (1 - pion, 2 - nucleon, 3 - kaon, 4 - charm)
      do icltar=icltar1,icltar2 !hadron type (2 - nucleon)
        do iy=1,11
          sy=spmin*(epmax/2./spmin)**((iy-1)/10.)
          syc=spminc*(epmax/2./spminc)**((iy-1)/10.)
          do iz=1,10
            z=.1*iz
            if(iy.eq.1)then
              fhgg(iy,iz,iclpro+4*(icltar-1))=-80.
            else
              fhgg(iy,iz,iclpro+4*(icltar-1))=log(om51pp(sy,1.,z,0)/z)
            endif

          do ix=1,10
            if(ix.le.5)then
              xp=.1*2.**(ix-5)
            else
              xp=.2*(ix-5)
            endif
            if(iy.eq.1)then
              fhqg(iy,ix,iz+10*(iclpro+4*(icltar-1)-1))=-80.
              fhgq(iy,ix,iz+10*(iclpro+4*(icltar-1)-1))=-80.
            else
              if(iclpro.ne.4)then
                syx=sy
              else
                syx=syc
              endif
              fhqg(iy,ix,iz+10*(iclpro+4*(icltar-1)-1))=
     *        log(om51pp(syx,xp,z,1)/z)
              if(icltar.ne.4)then
                syx=sy
              else
                syx=syc
              endif
              fhgq(iy,ix,iz+10*(iclpro+4*(icltar-1)-1))=
     *        log(om51pp(syx,xp,z,2)/z)
            endif
          enddo
          enddo

          do ix1=1,10
            if(ix1.le.5)then
              xpph=.1*2.**(ix1-5)
            else
              xpph=.2*(ix1-5)
            endif
          do ix2=1,10
            if(ix2.le.5)then
              xmm=.1*2.**(ix2-5)
            else
              xmm=.2*(ix2-5)
            endif

            if(iy.eq.1)then
              fhqq(iy,ix1,ix2+10*(iclpro+4*(icltar-1)-1))=-80.
            else
              if(iclpro.ne.4.and.icltar.ne.4)then
                syx=sy
              else
                syx=syc
              endif
              fhqq(iy,ix1,ix2+10*(iclpro+4*(icltar-1)-1))=
     *        log(pshard(syx,xpph,xmm))
            endif
          enddo
          enddo
        enddo
      enddo

      enddo

      if(isetcs.gt.1)then


        write(ifmt,'(a)')'  tabulate fit parameters ...'

      engysave=engy
      do iclpro=iclpro1,iclpro2 !hadron type (1 - pion, 2 - nucleon, 3 - kaon, 4 - charm)
      do icltar=icltar1,icltar2 !hadron type (2 - nucleon)
      do iclegy=iclegy2,iclegy1,-1
        call param
      enddo
      do iiclegy=iclegy2,iclegy1,-1
        engy=egyfac**(iiclegy-1)*egylow
        call paramini(0)
        call Kfit(iiclegy)
      enddo
      enddo
      enddo
      engy=engysave

      endif

      write(ifmt,'(a)')'  write to inirj ...'
      open(1,file=fnrj,status='unknown')
      write (1,*)alpqua,alplea,alppom,slopom,gamhad,r2had,chad,
     *qcdlam,q2min,q2ini,betpom,glusea,naflav,factk,pt2cut,gamtil
      write (1,*)fhgg,fhqg,fhgq,fhqq,fhgg0,fhgg1,fhqg1
     *,fhgg01,fhgg02,fhgg11,fhgg12,fhqg11,fhqg12
     *,ftoint,vfro,vnorm,coefxu1,coefxu2,coefxc2
      write(1,*)bkbin,iclpro1,iclpro2,icltar1,icltar2,iclegy1,iclegy2
     *,egylow,egymax,iomega,egyscr,epscrw,epscrp
      do iiipro=iclpro1,iclpro2
       do iiitar=icltar1,icltar2
        do iiiegy=iclegy1,iclegy2
        do iiib=1,nbkbin
          write(1,*)xkappafit(iiiegy,iiipro,iiitar,iiib)
        enddo
        do iiidf=idxD0,idxD
         write(1,*)alpDs(iiidf,iiiegy,iiipro,iiitar),
     *   alpDps(iiidf,iiiegy,iiipro,iiitar),
     *   alpDpps(iiidf,iiiegy,iiipro,iiitar),
     *   betDs(iiidf,iiiegy,iiipro,iiitar),
     *   betDps(iiidf,iiiegy,iiipro,iiitar),
     *   betDpps(iiidf,iiiegy,iiipro,iiitar),
     *   gamDs(iiidf,iiiegy,iiipro,iiitar),
     *   delDs(iiidf,iiiegy,iiipro,iiitar)
        enddo
        enddo
       enddo
      enddo

      close(1)

      engy=engysave
      maproj=maprojsave
      matarg=matargsave
      iclpro=iclpros
      icltar=icltars
      inicnt=1
      goto 5

4     continue

c--------------------------------------
c inelastic cross sections
c---------------------------------------

      if(isetcs.ge.2)then !--------------------

      if(ish.ge.4)write(ifch,*)'cross sections ...'
 6    continue
      inquire(file=fncs,exist=lcalc)
      if(lcalc)then
       if(inicnt.eq.1)then
        write(ifmt,'(3a)')'read from ',fncs(1:nfncs),' ...'
        open(1,file=fncs(1:nfncs),status='old')
        read (1,*)alpqua0,alplea0,alppom0,slopom0,
     *  gamhad0,r2had0,chad0,
     *  qcdlam0,q2min0,q2ini0,betpom0,glusea0,naflav0,
     *  factk0,pt2cut0
        if(alpqua0.ne.alpqua)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong alpqua',alpqua0,alpqua
        if(alppom0.ne.alppom)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong alppom',alppom0,alppom
        if(slopom0.ne.slopom)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong slopom',slopom0,slopom
        iii=2
        if(gamhad0(iii).ne.gamhad(iii))write(ifmt,'(a,i1,a,2f8.4)')
     *  'inics: wrong gamhad(',iii,')',gamhad0(iii),gamhad(iii)
        do iii=1,3
        if(r2had0(iii) .ne.r2had(iii) )write(ifmt,'(a,i1,a,2f8.4)')
     *  'inics: wrong r2had(',iii,')',r2had0(iii),r2had(iii)
        if(chad0(iii)  .ne.chad(iii)  )write(ifmt,'(a,i1,a,2f8.4)')
     *  'inics: wrong chad(',iii,')',chad0(iii),chad(iii)
        if(alplea0(iii).ne.alplea0(iii))write(ifmt,'(a,i1,a,2f8.4)')
     *  'inics: wrong alplea(',iii,')',alplea0(iii),alplea(iii)
        enddo
        if(qcdlam0.ne.qcdlam)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong qcdlam',qcdlam0,qcdlam
        if(q2min0 .ne.q2min )write(ifmt,'(a,2f8.4)')
     *  'inics: wrong q2min',q2min0,q2min
        if(q2ini0 .ne.q2ini )write(ifmt,'(a,2f8.4)')
     *  'inics: wrong q2ini',q2ini0,q2ini
        if(betpom0.ne.betpom)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong betpom',betpom0,betpom
        if(glusea0.ne.glusea)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong glusea',glusea0,glusea
        if(naflav0.ne.naflav)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong naflav',naflav0,naflav
        if(factk0 .ne.factk )write(ifmt,'(a,2f8.4)')
     *  'inics: wrong factk', factk0,factk
        if(pt2cut0 .ne.pt2cut )write(ifmt,'(a,2f8.4)')
     *  'inics: wrong pt2cut', pt2cut0,pt2cut
        if(alpqua0.ne.alpqua.or.alppom0.ne.alppom
     *  .or.slopom0.ne.slopom.or.gamhad0(2).ne.gamhad(2)
     *  .or.r2had0(1).ne.r2had(1).or.r2had0(2).ne.r2had(2)
     *  .or.r2had0(3).ne.r2had(3)
     *  .or.chad0(1).ne.chad(1).or.chad0(2).ne.chad(2)
     *  .or.chad0(3).ne.chad(3)
     *  .or.alplea0(1).ne.alplea(1).or.alplea0(2).ne.alplea(2)
     *  .or.alplea0(3).ne.alplea(3)
     *  .or.qcdlam0.ne.qcdlam.or.q2min0 .ne.q2min
     *  .or.q2ini0 .ne.q2ini
     *  .or.betpom0.ne.betpom.or.glusea0.ne.glusea.or.naflav0.ne.naflav
     *  .or.factk0 .ne.factk .or.pt2cut0.ne.pt2cut)then
           write(ifmt,'(//a//)')'   inics has to be reinitialized!!!!'
           stop
        endif

        read(1,*)isetcs0,iclpro10,iclpro20,icltar10,icltar20,iclegy10
     *   ,iclegy20,egylow0,egymax0,iomega0,egyscr0,epscrw0,epscrp0

        if(iclpro10.ne.iclpro1)write(ifmt,'(a,2i2)')
     *  'inics: wrong iclpro1',iclpro10,iclpro1
        if(iclpro20.ne.iclpro2)write(ifmt,'(a,2i2)')
     *  'inics: wrong iclpro2',iclpro20,iclpro2
        if(icltar10.ne.icltar1)write(ifmt,'(a,2i2)')
     *  'inics: wrong icltar1',icltar10,icltar1
        if(icltar20.ne.icltar2)write(ifmt,'(a,2i2)')
     *  'inics: wrong icltar2',icltar20,icltar2
        if(iclegy10.ne.iclegy1)write(ifmt,'(a,2i4)')
     *  'inics: wrong iclegy1',iclegy10,iclegy1
        if(iclegy20.ne.iclegy2)write(ifmt,'(a,2i4)')
     *  'inics: wrong iclegy2',iclegy20,iclegy2
        if(iomega0.ne.iomega)write(textini,'(a,2i8)')
     *  'inics: wrong iomega ',iomega0,iomega
        if(egylow0.ne.egylow)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong egylow',egylow0,egylow
        if(egymax0.ne.egymax)write(ifmt,'(a,2f12.4)')
     *  'inics: wrong egymax',egymax0,egymax
        if(egyscr0.ne.egyscr)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong egyscr ',egyscr0,egyscr
        if(epscrw0.ne.epscrw)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong epscrw',epscrw0,epscrw
        if(epscrp0.ne.epscrp)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong epscrp',epscrp0,epscrp
        if(isetcs0.lt.isetcs)write(ifmt,'(a,2f8.4)')
     *  'inics: wrong isetcs',isetcs0,isetcs
        if(iclpro10.ne.iclpro1.or.iclpro20.ne.iclpro2
     *   .or.icltar10.ne.icltar1.or.icltar20.ne.icltar2
     *   .or.iclegy10.ne.iclegy1.or.iclegy20.ne.iclegy2
     *   .or.egylow0.ne.egylow.or.egymax0.ne.egymax
     *   .or.egyscr0.ne.egyscr.or.epscrw0.ne.epscrw.or.isetcs0.lt.isetcs
     *   .or.epscrp0.ne.epscrp)then
           write(ifmt,'(//a//)')'   inics has to be reinitialized!!!!'
           stop
        endif
        if(isetcs.eq.2)then
          if(ionudi.eq.1)then
            read (1,*)asect,asect13,asect21,asect23,asectn
     *               ,asect33,asect41,asect43
          else  !ionudi=3
            read (1,*)asect11,asect,asect21,asect23,asect31
     *               ,asectn,asect41,asect43
          endif
        elseif(isetcs.eq.3)then
          if(ionudi.eq.1)then
            read (1,*)asect11,asect13,asect,asect23,asect31
     *               ,asect33,asectn,asect43
          else  !ionudi=3
            read (1,*)asect11,asect13,asect21,asect,asect31
     *               ,asect33,asect41,asectn
          endif
        else
           write(ifmt,'(//a//)')' Wrong isetcs in psaini !!!!'
        endif

        close(1)

      endif


        goto 7


      elseif(.not.producetables)then
        write(ifmt,*) "Missing epos.inics file !"        
        write(ifmt,*) "Please correct the defined path ",
     &"or force production ..."
        stop

      endif

      ifradesave=ifrade
      iremnsave=iremn
      idprojsave=idproj
      idprojinsave=idprojin
      idtargsave=idtarg
      idtarginsave=idtargin
      laprojsave=laproj
      latargsave=latarg
      maprojsave=maproj
      matargsave=matarg
      icltarsave=icltar
      iclprosave=iclpro
      engysave=engy
      pnllsave=pnll
      elabsave=elab
      ecmssave=ecms
      iclegysave=iclegy
      nrevtsave=nrevt
      neventsave=nevent
      ntevtsave=ntevt
      isetcssave=isetcs
      noebinsave=noebin
      isigmasave=isigma
      bminimsave=bminim
      bmaximsave=bmaxim
      bimevtsave=bimevt
      bkmxndifsave=bkmxndif
c      fctrmxsave=fctrmx
      ionudisave=ionudi


      isetcs=2
      isigma=1
      noebin=1
      idtarg=1120
      idtargin=1120
      bminim=0.
      bmaxim=10000.
      ifrade=0            !to save time, no fragmentation
      iremn=0             !to save time, simple remnants
      ionudi=3            !to have both ionudi=1 and 3 in tables

      write(ifmt,'(a)')'inics does not exist -> calculate tables  ...'

c initialize random numbers
      if(seedj.ne.0d0)then
        call ranfini(seedj,iseqsim,2)
      else
        stop 'seedi = 0 ... Please define it !'
      endif
      call aseed(2)

      laproj=-1
      maproj=1
      icltar=2
      do iclpro=1,4
       if(iclpro.lt.iclpro1.or.iclpro.gt.iclpro2)then
         do ie=1,7
           do iia=1,7
             asect11(ie,iclpro,iia)=0.
             asect21(ie,iclpro,iia)=0.
             asect13(ie,iclpro,iia)=0.
             asect23(ie,iclpro,iia)=0.
           enddo
         enddo
       else
         do ie=1,7
           engy=1.5*10.**(ie-1)
           call paramini(0)
           bkmxndif=conbmxndif()
           if(ish.ge.1)
     &     write(ifch,*)'  calcul.   ',ie,'  (',iclpro,')',engy
           write(ifmt,*)'  calcul.   ',ie,'  (',iclpro,')',engy

           sigine=0.
           do iia=1,7
            matarg=2**(iia-1)
            if(matarg.eq.1)then !hadron-proton interaction
c ine=cut+diff
              call psfz(2,gz2,0.)
              gin=gz2*pi*10.
c cut
              iomegasave=iomega
              iomega=2
              call psfz(2,gz2,0.)
              iomega=iomegasave
              gcut=gz2*pi*10.
c diff
              difpart=gin-gcut
c  non excited projectile and target
              gqela=(1.-rexdif(iclpro))*(1.-rexdif(icltar))*difpart
              gin3=max(1.,gin-gqela)              
            else
              call conini
              rad=radnuc(matarg)
              bm=rad+2.
              rrr=rad/difnuc(matarg)
              rrrm=rrr+log(9.)
              anorm=1.5/pi/rrr**3/(1.+(pi/rrr)**2)/difnuc(matarg)**2
c             gela=(ptgau(ptfau,bm,2,1)+ptgau1(bm,2,1))*10. !sig_ela
c in=cut+diff
              gcut=(ptgau(ptfau,bm,2,2)+ptgau1(bm,2,2))*10. !sig_in
              gin=gcut
c cut
              iomegasave=iomega
              iomega=2
              gcut=(ptgau(ptfau,bm,2,2)+ptgau1(bm,2,2))*10. !sig_cut
              iomega=iomegasave
c diff
              difpart=gin-gcut
c  non excited projectile
              gqela=(1.-rexdif(iclpro))
     &             **(1.+rexres(iclpro)*float(matarg-1)**0.3)
c  non excited target
              gqela=gqela*(1.-rexdif(icltar))
              gqela=gqela*difpart
              gin3=max(1.,gin-gqela)
            endif
            if(ish.ge.1)write (ifch,226)matarg,gin,gin3
226         format(2x,'psaini: hadron-nucleus (',i3,') cross sections:'/
     *       4x,'gin,gin3=',2e10.3)
            write(ifmt,*)'  matarg,gin,gin3:',matarg,gin,gin3
            asect11(ie,iclpro,iia)=log(gin)
            asect13(ie,iclpro,iia)=log(gin3)
           enddo
         enddo

         if(isetcssave.ge.3)then

         if(iclpro.eq.1)then
          idprojin=120
         elseif(iclpro.eq.2)then
          idprojin=1120
         elseif(iclpro.eq.3)then
          idprojin=130
         endif
         do ie=1,7
          engy=1.5*10.**(ie-1)
           if(engy.le.egymin)engy=egymin
           if(engy.ge.egymax)engy=egymax
           write(ifmt,*)'  simul.   ',ie,'  (',iclpro,')',engy
           if(ish.ge.1)
     &     write(ifch,*)'  simul.   ',ie,'  (',iclpro,')',engy
           do iia=1,7
            matarg=2**(iia-1)
            latarg=min(1,matarg/2)
c            fctrmx=max(ftcrmxsave,float(matarg))          !to get stable pA and AA cross section, this number has to be large for large A
            ntevt=0
            nrevt=0
            pnll=-1.
            elab=-1.
            ecms=-1.
            ekin=-1.
            call conini
            call ainit
            nevent=50000
            if(matarg.eq.1)nevent=1
            call epocrossc(nevent,sigt,sigi,sigc,sige,sigql,sigd)
c do not count non-excited diffractive projectile in inelastic
            sigi3=sigi-sigql
            if(ish.ge.1)write (ifch,228)matarg,sigi,sigi3
 228        format(2x,'simul.: hadron-nucleus (',i3,') cross sections:'/
     *       4x,'gin,gin3=',2e10.3)
            write(ifmt,*)'  matarg,sigi,sigi3 :',matarg,sigi,sigi3
            asect21(ie,iclpro,iia)=log(sigi)
            asect23(ie,iclpro,iia)=log(sigi3)
c            do  n=1,nevent
c              ntry=0
c 222          ntevt=ntevt+1
c              iret=0
c              ntry=ntry+1
c              bimevt=-1.
c              if(ntry.lt.10000)then
cc if random sign for projectile, set it here
c                idproj=idprojin*(1-2*int(rangen()+0.5d0))
c                call emsaaa(iret)
c                if(iret.gt.0)goto 222
c              else
c                ntevt=ntry
c              endif
c            enddo
c            a=pi*bmax**2
c            if(a.gt.0..and.ntevt.gt.0.)then
c             xs=anintine/float(ntevt)*a*10.
c             write(ifmt,*)'  matarg,nevent,ntevt,bmax,xs :'
c     .       ,matarg,anintine,ntevt,bmax,xs
c             write(ifch,*)'  matarg,nevent,ntevt,bmax,xs :'
c     .       ,matarg,anintine,ntevt,bmax,xs
c             asect2(ie,iclpro,iia)=log(xs)
c            else
c             write(ifmt,*)' Problem ? ',iclpro,matarg,bmax,ntevt
c             asect2(ie,iclpro,iia)=0.
c            endif
          enddo
        enddo
        else
          do ie=1,7
            do iia=1,7
              asect21(ie,iclpro,iia)=0.
              asect23(ie,iclpro,iia)=0.
            enddo
          enddo
        endif
       endif
      enddo

      idprojin=1120
      iclpro=2
      icltar=2
      do ie=1,7
        engy=1.5*10.**(ie-1)
        call paramini(0)
        bkmxndif=conbmxndif()
        if(ish.ge.1)
     &  write(ifch,*)'  calcul. AB  ',ie,engy
        write(ifmt,*)'  calcul. AB  ',ie,engy

        do iia=1,7
          maproj=2**(iia-1)
          laproj=max(1,maproj/2)
        do iib=1,7
          matarg=2**(iib-1)
          latarg=max(1,matarg/2)
          sigine=0.
          if(matarg.eq.1.and.maproj.eq.1)then !proton-proton interaction
c ine=cut+diff
            call psfz(2,gz2,0.)
            gin=gz2*pi*10.
c cut
            iomegasave=iomega
            iomega=2
            call psfz(2,gz2,0.)
            iomega=iomegasave
            gcut=gz2*pi*10.
c diff
            difpart=gin-gcut
c  non excited projectile and target
            gqela=(1.-rexdif(iclpro))*(1.-rexdif(icltar))*difpart
            gin3=max(1.,gin-gqela)              
          else
            call conini
            if(maproj.eq.1)then
              rad=radnuc(matarg)
              bm=rad+2.
              rrr=rad/difnuc(matarg)
              rrrm=rrr+log(9.)
              anorm=1.5/pi/rrr**3/(1.+(pi/rrr)**2)/difnuc(matarg)**2
c              gela=(ptgau(ptfau,bm,2,1)+ptgau1(bm,2,1))*10. !sig_ela
c in=cut+diff
              gcut=(ptgau(ptfau,bm,2,2)+ptgau1(bm,2,2))*10. !sig_in
              gin=gcut
c cut
              iomegasave=iomega
              iomega=2
              gcut=(ptgau(ptfau,bm,2,2)+ptgau1(bm,2,2))*10. !sig_cut
              iomega=iomegasave
c diff
              difpart=gin-gcut
c  non excited projectile
              gqela=(1.-rexdif(iclpro))
     &             **(1.+rexres(iclpro)*float(matarg-1)**0.3)
c  non excited target
              gqela=gqela*(1.-rexdif(icltar))**(1.+float(matarg)**0.3)
              gqela=gqela*difpart
              gin3=max(1.,gin-gqela)
            elseif(matarg.eq.1)then
              radp=radnuc(maproj)
              bm=radp+2.
              rrrp=radp/difnuc(maproj)
              rrrmp=rrrp+log(9.)
              anormp=1.5/pi/rrrp**3/(1.+(pi/rrrp)**2)/difnuc(maproj)**2
c              gtot=(ptgau(ptfau,bm,1,1)+ptgau1(bm,1,1))*10. !sig_in
c in=cut+diff
              gcut=(ptgau(ptfau,bm,1,2)+ptgau1(bm,1,2))*10. !sig_in
              gin=gcut     !in=cut+diff
c cut
              iomegasave=iomega
              iomega=2
              gcut=(ptgau(ptfau,bm,1,2)+ptgau1(bm,1,2))*10. !sig_cut
              iomega=iomegasave
c diff
              difpart=gin-gcut
c  non excited projectile
              gqela=(1.-rexdif(iclpro))**(1.+float(maproj)**0.3)
c  non excited target
              gqela=gqela*(1.-rexdif(icltar))
     &             **(1.+rexres(icltar)*float(maproj-1)**0.3)
              gqela=gqela*difpart
              gin3=max(1.,gin-gqela)
            else
              rad=radnuc(matarg)+1.
              radp=radnuc(maproj)+1.
              bm=rad+radp+2.
              rrr=rad/difnuc(matarg)
              rrrm=rrr+log(9.)
              rrrp=radp/difnuc(maproj)
              rrrmp=rrrp+log(9.)
              anorm=1.5/pi/rrr**3/(1.+(pi/rrr)**2)/difnuc(matarg)**2
              anormp=1.5/pi/rrrp**3/(1.+(pi/rrrp)**2)/difnuc(maproj)**2
c ine=cut+diff
c              gtot=(ptgau(ptfauAA,bm,2,1)+ptgau2(bm,1))*10.
              gcut=(ptgau(ptfauAA,bm,2,2)+ptgau2(bm,2))*10.
c              gin=gtot
              gin=gcut
c cut
              iomegasave=iomega
              iomega=2
              gcut=(ptgau(ptfauAA,bm,2,2)+ptgau2(bm,2))*10. !sig_cut
              iomega=iomegasave
c diff
              difpart=gin-gcut
c  non excited projectile
              gqelap=(1.-rexdif(iclpro))
     &             **(1.+rexres(iclpro)*float(matarg-1)**0.3)
              gqelap=gqelap**(1.+float(maproj)**0.3)
c  non excited target
              gqelat=(1.-rexdif(icltar))
     &             **(1.+rexres(icltar)*float(maproj-1)**0.3)
              gqelat=gqelat**(1.+float(maproj)**0.3)
              gqela=gqelap*gqelat*difpart
              gin3=gin-gqela
            endif
          endif
          if(ish.ge.1)write (ifch,227)maproj,matarg,gin,gin3
 227      format(2x,'psaini: nucleus-nucleus (',i3,'-',i3
     *       ,') cross sections:',/,4x,'gin,gin3=',2e10.3)
            write(ifmt,*)'  maproj,matarg,gin,gin3 :'
     *       ,maproj,matarg,gin,gin3
            asect31(ie,iia,iib)=log(gin)
            asect33(ie,iia,iib)=log(gin3)

          enddo
        enddo
      enddo

      if(isetcssave.ge.3)then

      do ie=1,7
        engy=1.5*10.**(ie-1)
        if(engy.le.egymin)engy=egymin
        if(engy.ge.egymax)engy=egymax
        write(ifmt,*)'  AB xs   ',ie,engy
        if(ish.ge.1)
     &  write(ifch,*)'  AB xs   ',ie,engy
        do iia=1,7
          maproj=2**(iia-1)
          laproj=max(1,maproj/2)
        do iib=1,7
          matarg=2**(iib-1)
          latarg=max(1,matarg/2)
c          fctrmx=max(ftcrmxsave,float(max(maproj,matarg))) !to get stable pA and AA cross section, this number has to be large for large A
          ntevt=0
          nrevt=0
          pnll=-1.
          elab=-1.
          ecms=-1.
          ekin=-1.
          call conini
          call ainit
          nevent=10000
          if(maproj+matarg.eq.2)nevent=1
          call epocrossc(nevent,sigt,sigi,sigc,sige,sigql,sigd)
c do not count non-excited diffractive projectile in inelastic
          sigi3=sigi-sigql
          if(ish.ge.1)write (ifch,229)maproj,matarg,sigi,sigi3
 229      format(2x,'simul.: nucleus-nucleus (',i3,'-',i3
     *       ,') cross sections:',/,4x,'gin,gin3=',2e10.3)
         write(ifmt,*)'  maproj,matarg,sigi,sigi3 :',maproj,matarg
     &                                               ,sigi,sigi3
          asect41(ie,iia,iib)=log(sigi)
          asect43(ie,iia,iib)=log(sigi3)

c          do  n=1,nevent
c            ntry=0
c 223        ntevt=ntevt+1
c            iret=0
c            ntry=ntry+1
c            bimevt=-1.
c            if(ntry.lt.10000)then
c              call emsaaa(iret)
c              if(iret.gt.0)goto 223
c            else
c              ntevt=ntry
c            endif
c          enddo
c          a=pi*bmax**2
c          if(a.gt.0..and.ntevt.gt.0.)then
c            xs=anintine/float(ntevt)*a*10.
c          write(ifmt,*)'  maproj,matarg,nevent,ntevt,bmax,xs :'
c     &                         ,maproj,matarg,anintine,ntevt,bmax,xs
c          write(ifch,*)'  maproj,matarg,nevent,ntevt,bmax,xs :'
c     &                         ,maproj,matarg,anintine,ntevt,bmax,xs
c            asect4(ie,iia,iib)=log(xs)
c          else
c            write(ifmt,*)' Problem ? ',maproj,matarg,bmax,ntevt
c            asect4(ie,iia,iib)=0.
c          endif
        enddo
      enddo
      enddo
      else
        do ie=1,7
          do iia=1,7
            do iib=1,7
              asect41(ie,iia,iib)=0.
              asect43(ie,iia,iib)=0.
            enddo
          enddo
        enddo
      endif

      ifrade=ifradesave
      iremn=iremnsave
      idproj=idprojsave
      idprojin=idprojinsave
      idtarg=idtargsave
      idtargin=idtarginsave
      laproj=laprojsave
      latarg=latargsave
      maproj=maprojsave
      matarg=matargsave
      icltar=icltarsave
      iclpro=iclprosave
      engy=engysave
      pnll=pnllsave
      elab=elabsave
      ecms=ecmssave
      iclegy=iclegysave
      nrevt=nrevtsave
      nevent=neventsave
      ntevt=ntevtsave
      isetcs=isetcssave
      noebin=noebinsave
      isigma=isigmasave
      bminim=bminimsave
      bmaxim=bmaximsave
      bimevt=bimevtsave
      bkmxndif=bkmxndifsave
      ionudi=ionudisave
c      fctrmx=fctrmxsave
      inicnt=1

      write(ifmt,'(a)')'write to inics ...'
      open(1,file=fncs,status='unknown')
      write (1,*)alpqua,alplea,alppom,slopom,gamhad,r2had,chad,
     *qcdlam,q2min,q2ini,betpom,glusea,naflav,factk,pt2cut
      write(1,*)isetcs,iclpro1,iclpro2,icltar1,icltar2,iclegy1,iclegy2
     *,egylow,egymax,iomega,egyscr,epscrw,epscrp
      write (1,*)asect11,asect13,asect21,asect23
     *          ,asect31,asect33,asect41,asect43

      close(1)


      goto 6

 7    continue

      endif !----------isetcs.ge.2-----------

      endif

      call utprix('psaini',ish,ishini,4)

      return
      end

cc-----------------------------------------------------------------------
c      function fjetxx(jpp,je1,je2)
cc-----------------------------------------------------------------------
cc   almost exactly psjet, just with Eqcd replaced by fparton
cc    for testing
cc   gives indeed the same result as jetx
cc   so the integration seems correct
cc-----------------------------------------------------------------------
c      double precision xx1,xx2,s2min,xmin,xmax,xmin1,xmax1,t,tmin
c     *,tmax,sh,z,qtmin,ft,fx1,fx2
c      common /ar3/   x1(7),a1(7)
c      common /ar9/ x9(3),a9(3)
c      include 'epos.inc'
c      include 'epos.incsem'
c
c      fjetxx=0.
c      s=engy*engy
c      s2min=4.d0*q2min
c
c      zmin=s2min/dble(s)
c      zmax=1
c
c      zmin=zmin**(-delh)
c      zmax=zmax**(-delh)
c      do i=1,3
c      do m=1,2
c        z=dble(.5*(zmax+zmin+(zmin-zmax)*(2*m-3)*x9(i)))**(-1./delh)
c        xmin=dsqrt(z)
c        sh=z*dble(s)
c        qtmin=max(dble(q2min),dble(q2ini)/(1.d0-dsqrt(z)))
c        tmin=max(0.d0,1.d0-4.d0*qtmin/sh)
c        tmin=2.d0*qtmin/(1.d0+dsqrt(tmin))
c        tmax=sh/2.d0
c        ft=0.d0
c        do i1=1,3
c        do m1=1,2
c          t=2.d0*tmin/(1.d0+tmin/tmax-dble(x9(i1)*(2*m1-3))
c     &    *(1.d0-tmin/tmax))
c          qt=t*(1.d0-t/sh)
c          xmax=1.d0-q2ini/qt
c          xmin=max(dsqrt(z),z/xmax)   !xm<xp !!!
c          if(xmin.gt.xmax.and.ish.ge.1)write(ifmt,*)'fjetxx:xmin,xmax'
c     *                                              ,xmin,xmax
c          fx1=0.d0
c          fx2=0.d0
c          if(xmax.gt..8d0)then
c            xmin1=max(xmin,.8d0)
c            do i2=1,3
c            do m2=1,2
c              xx1=1.d0-(1.d0-xmax)*((1.d0-xmin1)/(1.d0-xmax))**
c     *        dble(.5+x9(i2)*(m2-1.5))
c              xx2=z/xx1
c                fb=ffsigj(sngl(t),qt,sngl(xx1),sngl(xx2),jpp,je1,je2)
c     *       +ffsigj(sngl(t),qt,sngl(xx2),sngl(xx1),jpp,je1,je2)
c              fx1=fx1+dble(a9(i2)*fb)*(1.d0/xx1-1.d0)
c     *                               *pssalf(qt/qcdlam)**2
c            enddo
c            enddo
c            fx1=fx1*dlog((1.d0-xmin1)/(1.d0-xmax))
c          endif
c          if(xmin.lt..8d0)then
c            xmax1=min(xmax,.8d0)
c            do i2=1,3
c            do m2=1,2
c              xx1=xmin*(xmax1/xmin)**dble(.5+x9(i2)*(m2-1.5))
c              xx2=z/xx1
c
c              fb=0.
c              fb=fb
c     *             +ffsigj(sngl(t),qt,sngl(xx1),sngl(xx2),jpp,je1,je2)
c     *       +ffsigj(sngl(t),qt,sngl(xx2),sngl(xx1),jpp,je1,je2)
c              fx2=fx2+dble(a9(i2))*fb*pssalf(qt/qcdlam)**2
c            enddo
c            enddo
c            fx2=fx2*dlog(xmax1/xmin)
c          endif
c          ft=ft+dble(a9(i1))*(fx1+fx2)*t**2
c        enddo
c        enddo
c        ft=ft*(1.d0/tmin-1.d0/tmax)
c        fjetxx=fjetxx+a9(i)*sngl(ft*z**(1.+delh)/sh**2)
c     *          /z  ! ffsig = xp f xm f sigma
c      enddo
c      enddo
c      fjetxx=fjetxx*(zmin-zmax)/delh*pi**3
c  !   *         /2.   !???????????????  kkkkkkkkk
c      return
c      end
c
c
