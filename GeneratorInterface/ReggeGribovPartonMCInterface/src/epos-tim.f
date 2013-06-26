c---------------------------------------------------------------------
      subroutine timann
c---------------------------------------------------------------------
c      electron-positron
c---------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      common/nfla/nfla
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      integer jcp(6,2),jcm(6,2)
      dimension p1(5),jorprt2(ntim),ng(ntim)

      call utpri('timann',ish,ishini,5)
      egyevt=engy
      qsqevt=engy**2

 123  nprtj=1
      en=engy
      q20=0.
      nptl=0
      nptl=nptl+1               !electron
      pptl(1,nptl)=0.
      pptl(2,nptl)=0.
      pptl(3,nptl)=sqrt(en**2/4-2.61121e-7)
      pptl(4,nptl)=en/2.
      pptl(5,nptl)=0.000511
      idptl(nptl)=12
      istptl(nptl)=1

      nptl=nptl+1               !positron
      pptl(1,nptl)=0.
      pptl(2,nptl)=0.
      pptl(3,nptl)=-sqrt(en**2/4-2.61121e-7)
      pptl(4,nptl)=en/2.
      pptl(5,nptl)=0.000511
      idptl(nptl)=-12
      istptl(nptl)=1

      nptl=nptl+1               !virtual gamma
      pptl(1,nptl)=0.
      pptl(2,nptl)=0.
      pptl(3,nptl)=0.
      pptl(4,nptl)=en
      pptl(5,nptl)=en
      idptl(nptl)=10
      iorptl(nptl)=nptl-2
      jorptl(nptl)=nptl-1
      istptl(nptl)=1

      pprt(1,nprtj)=0.
      pprt(2,nprtj)=0.
      pprt(3,nprtj)=0.
      pprt(4,nprtj)=en
      pprt(5,nprtj)=en
      idaprt(1,nprtj)=2
      idaprt(2,nprtj)=3
      if(q20.gt.0.)then
        pprt(5,nprtj)=sqrt(q20)
      else
        pprt(5,nprtj)=en
      endif
      nfla=0
      do i=1,naflav
        call idmass(i,am)
        if (2.*am.lt.en) nfla=i
      enddo
      if(itflav.eq.0)then
        s=engy**2
        dlz=2.4
        amz=91.1885
        al=1./real(137.035989d0)
        gf=1.16639e-5
        ak=sqrt(2.)*gf*amz**2/(16*pi*al)
        chi1=ak*s*(s-amz**2)/((s-amz**2)**2+dlz**2*amz**2)
        chi2=ak**2*s**2/((s-amz**2)**2+dlz**2*amz**2)
        qe=-1.
        ve=0.25-2.*qe*0.232
        ae=-0.5
        qf=2./3.
        vf=sign(.5,qf)-2.*qf*0.232
        af=sign(.5,qf)
        dsmax1=
     $       2.*(qf**2-2.*qf*ve*vf*chi1
     $       +(ae**2+ve**2)*(af**2+vf**2)*chi2)
     $       + abs(-4*qf*ae*af*chi1+8*ae*ve*af*vf*chi2)

        qf=-1./3.
        vf=sign(.5,qf)-2.*qf*0.232
        af=sign(.5,qf)
        dsmax2=
     $       2.*(qf**2-2.*qf*ve*vf*chi1
     $       +(ae**2+ve**2)*(af**2+vf**2)*chi2)
     $       + abs(-4*qf*ae*af*chi1+8*ae*ve*af*vf*chi2)

 100    iq1=1+INT(nfla*rangen())
        call idchrg(iq1,qf)
        ct=-1.+2.*rangen()
        vf=sign(.5,qf)-2.*qf*0.232
        af=sign(.5,qf)
        dsigma=
     $       (1.+ct**2)*(qf**2-2.*qf*ve*vf*chi1
     $       +(ae**2+ve**2)*(af**2+vf**2)*chi2)
     $       + ct*(-4*qf*ae*af*chi1+8*ae*ve*af*vf*chi2)
        if(rangen().gt.dsigma/max(dsmax1,dsmax2)) goto 100
      else
        iq1=itflav
      endif
      if(rangen().lt.0.5)iq1=-iq1


      nprtj=nprtj+1
      idprt(nprtj)=iq1
      pprt(4,nprtj)=en/2.
      pprt(5,nprtj)=en
      iorprt(nprtj)=1
      jorprt(nprtj)=iq1
      q2prt(nprtj)=en**2

      nprtj=nprtj+1
      idprt(nprtj)=-iq1
      pprt(4,nprtj)=en/2.
      pprt(5,nprtj)=en
      iorprt(nprtj)=1
      jorprt(nprtj)=-iq1
      q2prt(nprtj)=en**2

      call timsho(2,3)

      jorprt2(1)=0               !!color-connection, no origin!!
      jt=1
      if(idprt(idaprt(1,1)).lt.0)jt=2
      do i=1,nprtj
        ng(i)=0
        if(idaprt(1,i).ne.0) then
          js=jt
          if(idprt(i).lt.0.and.
     &      ((idprt(idaprt(2,i)).eq.9.and.jt.eq.1).or.
     &      (idprt(idaprt(1,i)).eq.9.and.jt.eq.2)))then
            js=3-jt
          elseif(idprt(i).gt.0.and.idprt(i).ne.9.and.
     &        ((idprt(idaprt(2,i)).eq.9.and.jt.eq.2).or.
     &        (idprt(idaprt(1,i)).eq.9.and.jt.eq.1)))then
            js=3-jt
          elseif(idprt(i).eq.9.and.idprt(idaprt(1,i)).ne.9.and.
     &        ((idprt(idaprt(1,i)).lt.0.and.jt.eq.2).or.
     &        (idprt(idaprt(1,i)).gt.0.and.jt.eq.1)))then
            js=3-jt
          endif
          jorprt2(idaprt(3-js,i))=jorprt2(i)
          jorprt2(i)=idaprt(js,i)
          jorprt2(idaprt(js,i))=idaprt(3-js,i)
        else
          j=i
          do while(iorprt(j).ne.0)
            ng(i)=ng(i)+1
            j=iorprt(j)
          enddo
        endif
      enddo

      if(ish.ge.5)then
        i=1
        do while(i.ne.0)
          if(idaprt(1,i) .eq. 0 ) then
            write(ifch,*) idprt(i)
            write(ifch,*) '|'
          endif
          i=jorprt2(i)
        enddo
      endif

      iptl=nptl
      i=1
      do while(i.gt.0)
        if(idaprt(1,i) .eq. 0) then
          nptl=nptl+1
          do j=1,5
            pptl(j,nptl)=pprt(j,i)
          enddo
          idptl(nptl)=idprt(i)
          istptl(nptl)=20
          ityptl(nptl)=30
          iorptl(nptl)=iptl
          jorptl(nptl)=jorprt(i)   !type of mother parton
          qsqptl(nptl)=q2prt(i)
        endif
        i=jorprt2(i)
      enddo
      ifrptl(1,iptl)=iptl+1
      ifrptl(2,iptl)=nptl

      nk1=iptl+1
 441  nk2=nk1+1
      do while (idptl(nk2).eq.9)
        nk2=nk2+1
      enddo
      do i=1,4
        p1(i)=0.
        do j=nk1,nk2
          p1(i)=p1(i)+pptl(i,j)
        enddo
      enddo
      p1(5)=sqrt(max(0.,p1(4)**2-p1(3)**2-p1(2)**2-p1(1)**2))
      do i=1,2
        do j=1,6
          jcm(j,i)=0
          jcp(j,i)=0
        enddo
      enddo
      ii=1
      if(idptl(nk1).lt.0)ii=2
      jcp(abs(idptl(nk1)),ii)=1
      jcm(abs(idptl(nk2)),3-ii)=1
      amm= utamnx(jcp,jcm)
      if(amm.gt.p1(5))goto 123
      nk1=nk2+1
      if(nk1.lt.nptl)goto 441


      if(ish.ge.5)then
        write(ifch,*)
        do i=1,nprtj
          write(ifch,98) i,(pprt(j,i),j=1,5),idprt(i)
     &    ,iorprt(i),idaprt(1,i),idaprt(2,i),jorprt2(i),ng(i)
        enddo
        write(ifch,*)
        do i=1,nprtj
          if(pprt(5,i).eq.0.)
     &    write(ifch,99) i,(pprt(j,i),j=1,5),idprt(i),ng(i)
        enddo
        write(ifch,*)
        write(ifch,*)
      endif

 99   format(i4,5g10.3,2i4)
 98   format(i4,5g10.3,6i4)

      call utprix('timann',ish,ishini,5)
      return
      end

c---------------------------------------------------------------------
      subroutine timsh1(q20,en,idfla,jo)
c---------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      common/nfla/nfla
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      common/cprtx/nprtjx,pprtx(5,2)
      nfla=naflav
      nprtj=1
      pprt(1,nprtj)=0.
      pprt(2,nprtj)=0.
      pprt(3,nprtj)=0.
      pprt(4,nprtj)=en
      pprt(5,nprtj)=sqrt(q20)
      q2prt(nprtj)=q20
      idprt(nprtj)=idfla
      iorprt(nprtj)=nprtj
      jorprt(nprtj)=jo
      call timsho(1,0)
      nprtjx=0
      return
      end


c---------------------------------------------------------------------
      subroutine timsh2(q20,q21,en,idfla1,idfla2,jo1,jo2)
c---------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      common/nfla/nfla
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      common/cprtx/nprtjx,pprtx(5,2)
      nfla=naflav

      nprtj=1
      pprt(1,nprtj)=0.
      pprt(2,nprtj)=0.
      pprt(3,nprtj)=0.
      pprt(4,nprtj)=en/2.
      pprt(5,nprtj)=sqrt(q20)
      q2prt(nprtj)=q20
      idprt(nprtj)=idfla1
      iorprt(nprtj)=nprtj
      jorprt(nprtj)=jo1

      nprtj=2
      pprt(1,nprtj)=0.
      pprt(2,nprtj)=0.
      pprt(3,nprtj)=0.
      pprt(4,nprtj)=en/2.
      pprt(5,nprtj)=sqrt(q21)
      q2prt(nprtj)=q21
      idprt(nprtj)=idfla2
      iorprt(nprtj)=nprtj
      jorprt(nprtj)=jo2

      call timsho(1,2)

      nprtjx=1
      pprtx(1,nprtjx)=0.
      pprtx(2,nprtjx)=0.
      pprtx(3,nprtjx)=en/2.
      pprtx(4,nprtjx)=en/2.
      pprtx(5,nprtjx)=0
      nprtjx=2
      pprtx(1,nprtjx)=0.
      pprtx(2,nprtjx)=0.
      pprtx(3,nprtjx)=-en/2.
      pprtx(4,nprtjx)=en/2.
      pprtx(5,nprtjx)=0

      return
      end

c---------------------------------------------------------------------
      subroutine timsho(j1,j2)
c---------------------------------------------------------------------


      include 'epos.inc'
      include 'epos.incsem'
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj
      dimension pz(ntim),id(2,ntim)
      dimension ij(2),jo(2,2),ee(2),amm2(-6:10),q2s(2)
      logical first
      common/nfla/nfla
      call utpri('timsho',ish,ishini,5)
      if(iappl.eq.6)then
        do i=1,6
          call idmass(i,am)
          amm2(i)=am**2
          amm2(-i)=am**2
        enddo
      else
        do i=1,6
          amm2(i)=0.
          amm2(-i)=0.
        enddo
      endif
      amm2(9)=0.
      amm2(0)=0.
      amm2(10)=0.
      nn=nprtj
      ij(1)=j1
      ij(2)=j2
      ii2=2
      if(j2.eq.0)ii2=1

      ii=1
      first=.true.
      n10=0
 10   n10=n10+1
      if(float(n10).gt.1e7)then
        goto9999
      endif
      io=iorprt(ij(ii))
      idfl=idprt(ij(ii))
      q2start=pprt(5,ij(ii))**2
      E=pprt(4,ij(ii))
      if(ij(2).eq.j2.and.ii2.eq.2)E=pprt(4,ij(1))+pprt(4,ij(2))
      zetamx=0.
      if(ij(1).ne.j1)then
        zetamx=pprt(5,io)/pprt(4,io)/sqrt(pz(io)*(1.-pz(io)))
      endif
      

c      call timdev(idfl,q2start,E,zetamx,idfla,idflb,qa2,z)
c......................................................................
      q2=q2start
      z=0.5
      PT2MIN=max(qcdlam*1.1,q2fin)
      ALFM=LOG(PT2MIN/qcdlam)
      if(ish.ge.9)then
        write (ifch,*) '---------------------',ii
     $       ,pprt(5,ij(1))     !,pprt(5,ij(2))
        write(ifch,*) ' idfl,q2start,zetamx:',idfl,q2start,zetamx
      endif
      if (q2.lt.4.*q2fin+2.*amm2(idfl) )then
        if(ish.ge.9)then
          write(ifch,'(a,f3.0)') 'null:',0.
        endif
        q2=amm2(idfl)
        idfla=0
        idflb=0
        goto 999
      endif
      

 390  zc=.5*(1.-sqrt(max(0.000001,1.-4.*q2fin/q2)))
      if(ish.ge.9)then
        write(ifch,*) 'zc=',zc
      endif
      IF(idfl.EQ.9) THEN
        FBRqqb=nfla*(0.5-ZC)
        FBR=6.*LOG((1.-ZC)/ZC)+FBRqqb
      ELSE
        FBRqqb=0.
        FBR=(8./3.)*LOG((1.-ZC)/ZC)
      endif

      B0=(33.-2.*nfla)/6.


c.....select new q2
      r=rangen()
      q2=q2*exp(log(r)*B0*ALFM/FBR)
      if(ish.ge.9)then
        write(ifch,*) 'q^2=',q2
      endif
      if (q2.lt.4.*q2fin+2.*amm2(idfl))then
        q2=amm2(idfl)
        if(ish.ge.9)then
          write(ifch,'(a,f3.0)') 'null:',0.
        endif
        idfla=0
        idflb=0
        goto 999
      endif
      

c.....select flavor and z-value .....................................
      IF(idfl.EQ.9) THEN
        if(rangen()*FBR.lt.FBRqqb)then
                                ! .................g -> qqbar
          Z=ZC+(1.-2.*ZC)*rangen()
          IF(Z**2+(1.-Z)**2.LT.rangen()) GOTO 390
          idfla=int(1.+rangen()*real(nfla))
          idflb=-idfla
        else                    !..................g -> gg
          Z=(1.-ZC)*(ZC/(1.-ZC))**rangen()
          IF(rangen().GT.0.5) Z=1.-Z
          IF((1.-Z*(1.-Z))**2.lt.rangen()) GOTO 390
          idfla=9
          idflb=9
        endif
      ELSE
        Z=1.-(1.-ZC)*(ZC/(1.-ZC))**rangen() !!........q -> qg
        IF(1.+Z**2.LT.2.*rangen()) GOTO 390
          idfla=idfl
          idflb=9
      endif

      if(q2*z*(1.-z).le.qcdlam) goto 390
      if(alfm.lt.rangen()*log(q2*z*(1.-z)/qcdlam)) goto 390
      

      if(ij(1).ne.j1.or.ij(2).eq.0)then
        if(E.le.sqrt(q2))goto 390
        pzz=sqrt((E-sqrt(q2))*(E+sqrt(q2)))
        pt2=(E**2*(z*(1.-z)*q2-z*amm2(idflb)-(1.-z)*amm2(idfla))
     $       -.25*(amm2(idflb)-amm2(idfla)-q2)**2+q2*amm2(idfla))/pzz**2
        if (pt2.le.0.) then
          if(ish.ge.9)then
            write(ifch,*) 'z not good for pt2:',z,pt2
          endif
          goto 390
        endif
      endif

      zeta = sqrt(q2)/E/sqrt(z*(1.-z))
      if(zetamx.gt.0)then
        if (zeta.gt.zetamx)then
          if(ish.ge.9)then
            write(ifch,*) zeta,' > ',zetamx,'zeta-Ablehnung'
          endif
          goto 390
        endif
      endif
 999  continue
 
c......................................................................
      if(zeta.gt.zetacut)then
        q2s(ii)=q2
        if(idfla.eq.9)then                !anything -> gluon jet
          jo(1,ii)=-9
        elseif(jorprt(ij(ii)).eq.-9)then  !gluon -> quark jet
          jo(1,ii)=idfla
        else                              !no change   
          jo(1,ii)=jorprt(ij(ii))
        endif
        if(idflb.eq.9)then                !anything -> gluon jet
          jo(2,ii)=-9
        elseif(jorprt(ij(ii)).eq.-9)then  !gluon -> quark jet
          jo(2,ii)=idflb
        else
          jo(2,ii)=jorprt(ij(ii))
        endif
      else                             !no change   
        jo(1,ii)=jorprt(ij(ii))
        jo(2,ii)=jorprt(ij(ii))
        q2s(ii)=q2prt(ij(ii))
      endif

      pprt(5,ij(ii))=sqrt(q2)
      id(1,ii)=idfla
      id(2,ii)=idflb
      pz(ij(ii))=z
      if(first)then
        ii=2
        first=.false.
        if(ii2.eq.2)goto 10
      endif

      if(ij(2).eq.0)then
        z1=1.
        z2=0.
        pt=0.
        alpha=0.
        pprt(3,ij(1))=sqrt(E**2-pprt(5,ij(1))**2)
      elseif(ij(1).eq.j1.and.ij(2).eq.j2)then
        E=pprt(4,ij(1))+pprt(4,ij(2))
        ee(1)=E*.5+(pprt(5,ij(1))**2-pprt(5,ij(2))**2)/2./E
        ee(2)=E-ee(1)
        ii=1
        do while (ii.le.2)
          if(ee(ii)-pprt(5,ij(ii)).lt.0.) then
            if(ish.ge.5)then
              write(ifch,*) 'goto 11'
            endif
            ii=1
            if ( pprt(5,ij(1))**2-amm2(idprt(ij(1))).lt.
     $           pprt(5,ij(2))**2-amm2(idprt(ij(2))) ) ii=2
            goto 10
          endif
c          zc=.5*(1.-sqrt(1.-pprt(5,ij(ii))**2/ee(ii)**2))
c          if(pz(ij(ii)).lt.zc.or.pz(ij(ii)).gt.1.-zc)then
c            if(ish.ge.7)then
c              write(ifch,*) 'first branching rejected'
c            endif
c            goto 10
c          endif
          z=pz(ij(ii))
          q2=pprt(5,ij(ii))**2
          pzz=sqrt((ee(ii)-pprt(5,ij(ii)))*(ee(ii)+pprt(5,ij(ii))))
          if(pzz.gt.0.)then
           pt2=(ee(ii)**2*(z*(1.-z)*q2-z*amm2(id(2,ii))
     $          -(1.-z)*amm2(id(1,ii)))
     $          -.25*(amm2(id(2,ii))-amm2(id(1,ii))-q2)**2
     $          +q2*amm2(id(1,ii)))/pzz**2
          else
           pt2=0.
          endif
          if(id(1,ii).ne.0.and.pt2.le.0.)then
            if(ish.ge.7)then
              write(ifch,*) 'first branching rejected for pt2',ii
     $             ,z1,q2,ee(ii),pprt(5,ij(ii)),id(1,ii),id(2,ii)
            endif
            goto 10
          endif
          ii=ii+1
        enddo
        z1=ee(1)/E
        z2=ee(2)/E
        if(ish.ge.7)then
          write(ifch,*) 'z of first branching',z1
        endif
        pprt(3,ij(1))= sqrt(max(0.,ee(1)**2-pprt(5,ij(1))**2))
        pprt(3,ij(2))=-sqrt(max(0.,ee(2)**2-pprt(5,ij(2))**2))
        pt=0.
        alpha=0.
      else
        E=pprt(4,io)
        z1=pz(io)
        z2=1.-z1
        am0=pprt(5,io)
        am1=pprt(5,ij(1))
        am2=pprt(5,ij(2))
        aM=am2**2-am0**2-am1**2
        pzz=sqrt((E-am0)*(E+am0))
        pprt(3,ij(1))=.5*(aM+2.*z1*E**2)/pzz
        pprt(3,ij(2))=pzz-pprt(3,ij(1))
        pt2=(E**2*(z1*z2*am0**2-z1*am2**2-z2*am1**2)
     $       -.25*aM**2+am0**2*am1**2)/pzz**2
        if(ish.ge.8)then
          write(ifch,*) 'pt2,pzz=',pt2,pzz,z1**2*E*pprt(5,io),z1,E
     $         ,pprt(5,io)
        endif
        if(pt2.lt.0.) then
          ii=1
          if ( pprt(5,ij(1))**2-amm2(idprt(ij(1))).lt.
     $         pprt(5,ij(2))**2-amm2(idprt(ij(2))) ) ii=2
          goto 10
        endif
        pt=sqrt(pt2)
        alpha=2.*pi*rangen()
      endif
      pprt(4,ij(1))=z1*E
      pprt(1,ij(1))=cos(alpha)*pt
      pprt(2,ij(1))=sin(alpha)*pt
      if(ii2.eq.2)then
        pprt(4,ij(2))=z2*E
        pprt(1,ij(2))=-cos(alpha)*pt
        pprt(2,ij(2))=-sin(alpha)*pt
      endif
      if(ij(1).ne.j1)then
        if(pprt(1,io).ne.0..or.pprt(2,io).ne.0.)then
          do ii=1,2
            call utrota(-1,pprt(1,io),pprt(2,io),pprt(3,io)
     &           ,pprt(1,ij(ii)),pprt(2,ij(ii)),pprt(3,ij(ii)))
          enddo
        endif
      endif
      if(ij(1).ne.j1)then
        if(pprt(3,io).lt.0.)then
          do k=1,3
            do ii=1,2
              pprt(k,ij(ii)) = -pprt(k,ij(ii))
            enddo
          enddo
        endif
      endif
      do ii=1,ii2
        if(id(1,ii).ne.0)then
          idprt(nprtj+1)=id(1,ii)
          idprt(nprtj+2)=id(2,ii)
          pprt(4,nprtj+1)=pz(ij(ii))*pprt(4,ij(ii))
          pprt(5,nprtj+1)=pz(ij(ii))*pprt(5,ij(ii))
          pprt(4,nprtj+2)=(1.-pz(ij(ii)))*pprt(4,ij(ii))
          pprt(5,nprtj+2)=(1.-pz(ij(ii)))*pprt(5,ij(ii))
          q2prt(nprtj+1)=q2s(ii)
          q2prt(nprtj+2)=q2s(ii)
          iorprt(nprtj+1)=ij(ii)
          iorprt(nprtj+2)=ij(ii)
          jorprt(nprtj+1)=jo(1,ii)
          jorprt(nprtj+2)=jo(2,ii)
          idaprt(1,ij(ii))=nprtj+1
          idaprt(2,ij(ii))=nprtj+2
          idaprt(1,nprtj+1)=0
          idaprt(2,nprtj+1)=0
          idaprt(1,nprtj+2)=0
          idaprt(2,nprtj+2)=0
          nprtj=nprtj+2
        else
          idaprt(1,ij(ii))=0
          idaprt(2,ij(ii))=0
        endif
      enddo
      if(ish.ge.5)then
        if(ij(1).ne.j1)then
          write(ifch,98) io,(pprt(j,io),j=1,5),pz(io),jorprt(io)
     &    ,idprt(io)
          write(ifch,*) '->'
        endif
        write(ifch,99) ij(1),(pprt(j,ij(1)),j=1,5),pz(ij(1))
     &  ,jo(1,1),jo(2,1),idprt(ij(1)),'->',id(1,1),id(2,1)
        if(ij(2).ne.0)write(ifch,99) ij(2),
     &  (pprt(j,ij(2)),j=1,5),pz(ij(2))
     &  ,jo(1,2),jo(2,2),idprt(ij(2)),'->',id(1,2),id(2,2)
        write(ifch,*)
      endif

 98   format(i4,6g10.3,3i3)
 99   format(i4,6g10.3,3i3,a,2i4)


      if(ij(1).le.nn)ij(1)=nn-1
      ij(1)=ij(1)+2
      ij(2)=ij(1)+1
      ii=1
      ii2=2
      first=.true.
      if(ij(1).le.nprtj)goto 10

      if(ish.ge.5)then
        write(ifch,*)
        do i=1,nprtj
          write(ifch,'(i4,1x,5g10.4,2i4,a,2i4)')
     &    i,(pprt(j,i),j=1,5),idprt(i)
     &    ,iorprt(i),'  -->',idaprt(1,i),idaprt(2,i)
        enddo
        write(ifch,*)
      endif
 9999 call utprix('timsho',ish,ishini,5)
      return
      end

