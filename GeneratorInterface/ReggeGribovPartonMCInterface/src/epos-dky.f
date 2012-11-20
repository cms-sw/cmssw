c-----------------------------------------------------------------------
      subroutine decayall(n)
c-----------------------------------------------------------------------
c  decay of objects n to nptl, including their children
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      ttaus=1
      np1=n
 1    np2=nptl
      do ip=np1,np2
         if(istptl(ip).eq.0)then  !consider last generation particles
            call hdecas(ip,iret)
            if(iret.eq.1)stop'error in hdecas detected in decay'
         endif
      enddo
      np1=np2+1
      if(np1.le.nptl)goto1
      end


c-----------------------------------------------------------------------
      subroutine hdecas(i,iret)
c-----------------------------------------------------------------------
c  decay of object i  (main decay routine)
c-----------------------------------------------------------------------

      include 'epos.inc'
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat,zor,tor
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat
      double precision ttaux,ttauz
      integer jcdu(nflav,2)

      iret=0
      nptlb=nptl

c no last generation -> no decay

      if(istptl(i).ne.0)return

      if(nptl.gt.mxptl-10)then
        call alist('end&',1,nptl)
        call utstop('hdecas: mxptl too small&')
      endif
c entry

      call utpri('hdecas',ish,ishini,5)
      ttauz=ttaus


c skip nuclei

      if(idptl(i).gt.1e9)return

c small droplet decay

      if(iabs(idptl(i)).gt.1e8)then
        stop'hdecas: no longer supported (2).       '
      endif

c  ordinary decay

      call idmass(111,amrho0)
      call idmass(221,amomeg)
      ioi=iorptl(i)
      if(ioi.gt.0.and.(idptl(i).eq.111.or.idptl(i).eq.221))then
        if(.not.(iabs(idptl(ioi)).lt.10000
     *       .and.jorptl(i).eq.0))then

          if(iLHC.eq.1.and.((ityptl(i).ge.20.and.ityptl(i).le.39)
     *        .or.ityptl(i).eq.42.or.ityptl(i).eq.52))then
c mix rho and omegas only from string production and if not decay product
            if(idptl(i).eq.111)idptl(i)=221
            if(idptl(i).eq.221.and.ityptl(i).ge.30.and.ityptl(i).le.39
     *         .and.rangen().gt.0.5)idptl(i)=111
          elseif(iLHC.eq.0.and..not.(ityptl(i).eq.60))then
            if(idptl(i).eq.111)idptl(i)=221
            if(idptl(i).eq.221.and.rangen().gt.0.5)idptl(i)=111
          endif

        endif
      endif

      if(ctaumin.gt.0.)then
        call idtau(idptl(i),1.,1.,ctau)       !ctau in fm
        if(ctau*1.e-13.gt.ctaumin)goto 1000   !ctaumin in cm
      endif

      ida=iabs(idptl(i))

      if(.not.(iappl.eq.7.and.i.eq.1))then
      if(mod(ndecay        ,10).eq.1
     *.and.ida.ne.0.and.ida.lt.10000)goto1000
      if(mod(ndecay/10     ,10).eq.1.and.ida.eq.  20)goto1000
      if(mod(ndecay/100    ,10).eq.1.and.ida.eq.2130)goto1000
      if(mod(ndecay/1000   ,10).eq.1.and.ida.eq.1130)goto1000
      if(mod(ndecay/1000   ,10).eq.1.and.ida.eq.2230)goto1000
      if(mod(ndecay/10000  ,10).eq.1.and.ida.eq.2330)goto1000
      if(mod(ndecay/10000  ,10).eq.1.and.ida.eq.1330)goto1000
      if(mod(ndecay/100000 ,10).eq.1.and.ida.eq.3331)goto1000
      if(mod(ndecay/1000000,10).eq.1.and.ida.eq. 110)goto1000

      if(nrnody.gt.0)then
      do nod=1,nrnody
      if(idptl(i).eq.nody(nod))goto 1000
      enddo
      endif


      endif

      call hdecay(i,iret)
      if(iret.eq.1)goto1000
      if(nptl.le.nptlb)then
        iret=-1
        goto 1000
      endif

c ---successful decay---

      istptl(i)=1
      ifrptl(1,i)=nptlb+1
      ifrptl(2,i)=nptl

      t=tivptl(2,i)
      x=xorptl(1,i)+(t-xorptl(4,i))*pptl(1,i)/pptl(4,i)
      y=xorptl(2,i)+(t-xorptl(4,i))*pptl(2,i)/pptl(4,i)
      z=xorptl(3,i)+(t-xorptl(4,i))*pptl(3,i)/pptl(4,i)
      call jtaux(t,z,ttaux)
      ttaus=ttaux
      if( ttaus.gt.0d0 ) then
        call jtauin
        call jtaus(z,ttest,sz)
        if (abs(t-ttest).gt.1e-5.and.ish.ge.1) then
          call utmsg('hdecas')
          write(ifch,*)'*****  t /= ttest'
          write(ifch,*)t,ttest,i,z,t,xorptl(3,i),xorptl(4,i)
     $         ,pptl(3,i),pptl(4,i)
          call utmsgf
        endif
      endif

c loop over decay products

      do 20 n=nptlb+1,nptl
      iorptl(n)=i
      jorptl(n)=0
      istptl(n)=0
      ifrptl(1,n)=0
      ifrptl(2,n)=0
      rad=0
      phi=0
      ti=t
      zi=z
      xorptl(1,n)=x + rad*cos(phi)
      xorptl(2,n)=y + rad*sin(phi)
      xorptl(3,n)=zi
      xorptl(4,n)=ti
      io=n
1     io=iorptl(io)
      if(ish.ge.4)write(ifch,*)'io = ',io,'  origin: ',iorptl(io)
      if(io.eq.iorptl(io))call utmsg("Strange iorptl in hdecas&")
c security to avoid infinite loop
      if(iorptl(io).gt.0.and.io.ne.iorptl(io))goto 1  
      if(ish.ge.4)write(ifch,*)'origin: ',io,idptl(io)
      zor=xorptl(3,io)
      tor=xorptl(4,io)
      call idquac(io,nq,ndummy1,ndummy2,jcdu)
      r=rangen()
      tauran=-taurea*alog(r)
      call jtaix(n,tauran,zor,tor,zis,tis)
      tivptl(1,n)=amax1(ti,tis)
      call idtau(idptl(n),pptl(4,n),pptl(5,n),taugm)
      r=rangen()
      tivptl(2,n)=t+taugm*(-alog(r))
      ityptl(n)=ityptl(i)
      radptl(n)=0.
      dezptl(n)=0.
      itsptl(n)=itsptl(i)
      rinptl(n)=rinptl(i)
20    continue

      if(iabs(idptl(nptlb+1)).le.6) then
        call gakli2(0,0)
        write (*,*) 'nptlb+1,nptl:',nptlb+1,nptl
        istptl(nptlb+1)=1
        do n=nptlb+2,nptl
          istptl(n)=20
        enddo
        call gakfra(0,iret)
        if(iret.eq.1)goto1000
        call gakli2(0,0)
      endif

1000  continue
      ttaus=ttauz
      call jtauin
      call utprix('hdecas',ish,ishini,5)
      return
      end

c-----------------------------------------------------------------------
      subroutine hdecay(ip,iret)
c-----------------------------------------------------------------------
c  decays particle ip from /cptl/
c  for ip being no resonance: call StaHad
c  for ip being resonance: standard resonance decay  procedure
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat
      common/wco/wmass2,wgam2
      parameter (mxlook=10000,mxdky=2000)
      common/dkytab/look(mxlook),cbr(mxdky),mode(5,mxdky)
      dimension pgen(5,10),rnd(10),u(3),beta(3)
     1     ,reduce(10)
      dimension prest(4,10),kno(10)
      data reduce/1.,1.,2.,5.,15.,60.,250.,1500.,1.2E4,1.2E5/
      data twome/1.022006e-3/

c          fctn definitions
      dot(i1,i2)=prest(4,i1)*prest(4,i2)-prest(1,i1)*prest(1,i2)
     *-prest(2,i1)*prest(2,i2)-prest(3,i1)*prest(3,i2)
c          charged w propagator.
      wprop(z)=(z-wmass2**2)**2+(wmass2*wgam2)**2

      call utpri('hdecay',ish,ishini,5)

      ipp=ip
      iret=0
      nptlb=nptl

      if(ish.ge.5)write(ifch,*)'ip,id,mass: ',ip,idptl(ip),pptl(5,ip)


      if(model.eq.4.and.iappl.eq.7)then
        if(abs(idptl(ipp)).gt.13.and.abs(idptl(ipp)).ne.1120
     &.and.abs(idptl(ipp)).ne.15)call decaymod(ipp,iret)
        if(iret.gt.0)goto 1000
        naddptl=0
        goto 900 
      endif

c     no k_long decay
c     ---------------
c     if(idptl(ip).eq.-20)goto1000

c     select decay mode
c     -----------------
      ntry=0
2     ntry=ntry+1
           if(ntry.gt.100)then
      if(ish.ge.1)then
      call utmsg('hdecay')
      write(ifch,*)'*****  decay not possible. iret = 1.'
      call utmsgf
      endif
      iret=1
      goto1000
           endif
      idlv1=idptl(ip)
      amss=pptl(5,ip)

c Decay of deuteron

      if(abs(idlv1).eq.17)then
        amss=1.01*amss
        naddptl=2
        call idmass(1120,amnew)
        pptl(5,nptl+1)=amnew
        idptl(nptl+1)=sign(1120,idlv1)
        sum=amnew
        call idmass(1220,amnew)
        pptl(5,nptl+2)=amnew
        idptl(nptl+2)=sign(1220,idlv1)
        sum=sum+amnew
        goto 111
      endif

c Decay of triton

      if(abs(idlv1).eq.18)then
        amss=1.01*amss
        naddptl=3
        call idmass(1120,amnew)
        pptl(5,nptl+1)=amnew
        idptl(nptl+1)=sign(1120,idlv1)
        sum=amnew
        call idmass(1220,amnew)
        pptl(5,nptl+2)=amnew
        idptl(nptl+2)=sign(1220,idlv1)
        sum=sum+amnew
        call idmass(1220,amnew)
        pptl(5,nptl+3)=amnew
        idptl(nptl+3)=sign(1220,idlv1)
        sum=sum+amnew
         goto 111
      endif

c Decay of alpha

      if(abs(idlv1).eq.19)then
        amss=1.01*amss
        naddptl=4
        call idmass(1120,amnew)
        pptl(5,nptl+1)=amnew
        idptl(nptl+1)=sign(1120,idlv1)
        sum=amnew
        call idmass(1220,amnew)
        pptl(5,nptl+2)=amnew
        idptl(nptl+2)=sign(1220,idlv1)
        sum=sum+amnew
        call idmass(1120,amnew)
        pptl(5,nptl+3)=amnew
        idptl(nptl+3)=sign(1120,idlv1)
        sum=sum+amnew
        call idmass(1220,amnew)
        pptl(5,nptl+4)=amnew
        idptl(nptl+4)=sign(1220,idlv1)
        sum=sum+amnew
        goto 111
      endif

c  select one of the decay channel
      ipoint=look(iabs(idlv1))-1
      if(idlv1.eq.-20)ipoint=look(320)-1
      if(ipoint.lt.0) goto1000
      try=rangen()
100   ipoint=ipoint+1
      if(ish.ge.4)write(ifch,*)'ipoint,cbr,try',ipoint,cbr(ipoint),try
      if(try.gt.cbr(ipoint)) goto100
      naddptl=0
      sum=0.
c      nstart=nptl+1                  !?????????????????unused
      new=0
      do 110 i=1,5         !store id and mass of products
        if(mode(i,ipoint).eq.0) goto 110
        if(nptl+naddptl+1.gt.mxptl) goto 9999
        if(iabs( mode(1,ipoint)) .le. 6.and.i.eq.2)then   !decay into quark ???
          call vedi(mode(1,ipoint),mode(2,ipoint),k3,idlv1)
          idptl(new)=idlv1
          call idmass(idlv1,amnew)
          pptl(5,new)=amnew
          sum=pptl(5,new)
        else                                 !decay into particles
          naddptl=naddptl+1
          new=nptl+naddptl
          idptl(new)=mode(i,ipoint)
          idlv1=idptl(new)
          call idmass(idlv1,pptl(5,new))
          sum=sum+pptl(5,new)
        endif
 110  continue
 111  continue
      if(naddptl.ne.1.and.sum.ge.amss)goto 2
 112  naddptl1=naddptl-1
      do 120 j=1,5
      pgen(j,1)=pptl(j,ip)
120   continue
      pgen(5,1)=amss !needed because of deuteron, triton and alpha decay and OK

      pgen(5,naddptl)=pptl(5,nptl+naddptl)
      if(naddptl.eq.1) goto 700            !one body decay
      if(naddptl.eq.2) goto 400            !two body decay

      if(ish.ge.4)write(ifch,*)'>= 3 body decay'

c     use kroll-wada distribution for pi0 and eta dalitz decays.
c     ----------------------------------------------
      if(.not.((idptl(ip).eq.110.or.idptl(ip).eq.220).and.
     1iabs(idptl(nptl+2)).eq.12)) goto 130
      ntry=0             !decay of pi0 or eta into electron
125   ntry=ntry+1
           if(ntry.gt.10)then
      if(ish.ge. 0)then
      call utmsg('hdecay')
      write(ifch,*)'*****  ntry > 10. iret = 1.'
      write(ifch,*)'***** amee,ree,wtee',amee,ree,wtee
      call utmsgf
      endif
      iret=1
      goto1000
           endif
      amee=twome*(pptl(5,ip)/twome)**rangen()
      ree=(twome/amee)**2
      wtee=(1.-(amee/pptl(5,ip))**2)**3*sqrt(1.-ree)*(1.+.5*ree)
      if(wtee.lt.rangen()) goto125
      pgen(5,2)=amee
      goto400
130   continue

c     calculate maximum phase-space weight
c     ------------------------------------
      wtmax=1./reduce(naddptl)
      sum1=pgen(5,1)
      sum2=sum-pptl(5,nptl+1)
      do 200 i=1,naddptl1
      wtmax=wtmax*utpcm(sum1,sum2,pptl(5,nptl+i))
      sum1=sum1-pptl(5,nptl+i)
      sum2=sum2-pptl(5,nptl+i+1)
200   continue

c     generate uniform naddptl-body phase space
c     --------------------------------------
      ntry=0
300   ntry=ntry+1
           if(ntry.gt.10000)then
      if(ish.ge. 0)then
      call utmsg('hdecay')
      write(ifch,*)'*****  infinite loop (2). iret = 1.'
      write(ifch,*)'***** ip,idptl(ip),pptl(5,ip):'
     *,ip,idptl(ip),pptl(5,ip)
      write(ifch,*)'***** wt,wtmax:',wt,wtmax
      write(ifch,*)'***** i,pgen(5,i),pptl(5,nptl+i),idptl(nptl+i):'
      do i=1,naddptl
      write(ifch,*)i,pgen(5,i),pptl(5,nptl+i),idptl(nptl+i)
      enddo
      call utmsgf
      endif
      iret=1
      goto1000
           endif
      rnd(1)=1.
      jsave=1
      do 310 i=2,naddptl1
      rnew=rangen()
      i1=i-1
      do 320 jj1=1,i1
      j=i-jj1
      jsave=j+1
      if(rnew.le.rnd(j)) goto310
      rnd(jsave)=rnd(j)
320   continue
310   rnd(jsave)=rnew
      rnd(naddptl)=0.
      wt=1.
      sum1=sum
      do 330 i=2,naddptl
      sum1=sum1-pptl(5,nptl+i-1)
      pgen(5,i)=sum1+rnd(i)*(pgen(5,1)-sum)
      a=pgen(5,i-1)
      b=pgen(5,i)
      c=pptl(5,nptl+i-1)
      wt=wt*utpcm(a,b,c)
330   continue
      if(wt.lt.rangen()*wtmax) goto300

c     carry out two-body decays in pgen frames
c     ----------------------------------------
400   continue
      if(ish.ge.4)write(ifch,*)'2 body decay'
      do 410 i=1,naddptl1
      qcm=utpcm(pgen(5,i),pgen(5,i+1),pptl(5,nptl+i))
      u(3)=2.*rangen()-1.
      phi=2.*pi*rangen()
      u(1)=sqrt(1.-u(3)**2)*cos(phi)
      u(2)=sqrt(1.-u(3)**2)*sin(phi)
      do 420 j=1,3
      pptl(j,nptl+i)=qcm*u(j)
      pgen(j,i+1)=-pptl(j,nptl+i)
420   continue
      pptl(4,nptl+i)=sqrt(qcm**2+pptl(5,nptl+i)**2)
      pgen(4,i+1)=sqrt(qcm**2+pgen(5,i+1)**2)
410   continue
      do 430 j=1,4
      pptl(j,nptl+naddptl)=pgen(j,naddptl)
430   continue

c     boost pgen frames to lab frame
c          also save momenta in rest frame (last frame)
c     -------------------------------------------------
      do 500 ii=1,naddptl1
      i=naddptl-ii
      do 510 j=1,3
      beta(j)=pgen(j,i)/pgen(4,i)
510   continue
      gamma=pgen(4,i)/pgen(5,i)
      do 520 k=i,naddptl
      k1=nptl+k
      bp=beta(1)*pptl(1,k1)+beta(2)*pptl(2,k1)+beta(3)*pptl(3,k1)
      do 530 j=1,3
      prest(j,k)=pptl(j,k1)
      pptl(j,k1)=pptl(j,k1)+gamma*beta(j)*(pptl(4,k1)
     1+bp*gamma/(gamma+1.))
530   continue
      prest(4,k)=pptl(4,k1)
      pptl(4,k1)=gamma*(pptl(4,k1)+bp)
      if(pptl(4,k1).lt.1.d-5)then
        pptl(4,k1)=sqrt(pptl(1,k1)*pptl(1,k1)+pptl(2,k1)*pptl(2,k1)
     &                 +pptl(3,k1)*pptl(3,k1))
      endif
520   continue
500   continue

c     matrix elements
c     ---------------
        if(iabs(idptl(ip)).eq.14)then                  !muon decay
          goto 650
        elseif(naddptl.eq.3)then
          if(idptl(ip).eq.221.or.idptl(ip).eq.331)then  !omeg and phi decay
            goto 610
          elseif(iabs(idptl(ip)).eq.130.or.       !Kl and K decay
     1       idptl(ip).eq.-20)then
            if(iabs(idptl(nptl+2)).lt.20)then   !semi-leptonic
              goto 630
            else                                !hadronic
              goto 640
            endif
          elseif(iabs(idptl(nptl+1)).lt.20.and. !other semi-leptonic decay
     1       idptl(nptl+1).ne.10)then
            goto 620
          elseif(iabs(idptl(nptl+2)).le.6)then
            goto 605            !decay into quark
          else
            goto 800
          endif
        else
         goto 800
        endif

 605    wt=pptl(5,ip)*pptl(5,nptl+1)*dot(2,3)
        IF(wt.LT.rangen()*pptl(5,ip)**4/16.) goto 300
        ams=sqrt(dot(2,2)+dot(3,3)+2.*dot(2,3))
        kno(1)=idptl(nptl+2)
        kno(2)=idptl(nptl+3)
        if(ammin(kno(1),kno(2)).gt.ams)then
          call vedi(kno(1),kno(2),iddum,idlv2)
          idptl(nptl+2)=idlv2
          call idmass(idlv2,amnew2)
          pptl(5,nptl+2)=amnew2
          naddptl=2
          goto 112
        endif
c......multiplicity
        PS =sqrt(dot(2,2))
        psq=sqrt(dot(3,3))
c        PSP=PS                  !!???????????????unused
        np=0                    !!!!?????
        nq=2
        CNDE=4.5*LOG(MAX((ams-PS-PSQ)/0.7,1.1))
c        IF(MMAT.EQ.12) CNDE=CNDE+PARJ(63)
 769    NTRY=NTRY+1
        IF(NTRY.GT.1000) THEN
          write(*,*)'hdecay caught in infinite loop'
          write(ifch,*)'hdecay caught in infinite loop'
          iret=1
          goto 1000
        ENDIF
        GAUSS=SQRT(-2.*CNDE*LOG(MAX(1E-10,rangen())))*
     &       SIN(2.*pi*rangen())
        ND=0.5+0.5*NP+0.25*NQ+CNDE+GAUSS
        IF(ND.LT.NP+NQ/2.OR.ND.LT.2.OR.ND.GT.10) GOTO 769


c......choose hadrons


        kno(3)=kno(1)
        kno(4)=kno(2)

        CONTINUE
        IF(ND.EQ.NP+NQ/2) GOTO 773
        DO I=nptl+2,nptl+2+nd-nq/2-1
          JT=2+1+INT((NQ-1) * rangen() )
          CALL vedi(kno(JT),0,KFL2,idlv3)
          idptl(i)=idlv3
c          IF(K(I,2).EQ.0) GOTO 769
          kno(JT)=-KFL2
        enddo
 773    CONTINUE
        CALL vedi(kno(3),kno(4),KFLDMP,idlv4)
        idptl(nptl+2+nd-nq/2)=idlv4
        sum=0.
        do i=nptl+2,nptl+2+nd-nq/2
          call idmass(idptl(i),am)
          pptl(5,i)=am
          sum=sum+am
        enddo
        if(sum.gt.ams) goto 769
c......goto phase space dis....
        ip=nptl+2+nd-nq/2+1
        do j=1,4
          pptl(j,ip)=pptl(j,ipp)-pptl(j,nptl+1)
        enddo
        pptl(5,ip)=ams
        idptl(ip)=sign(80,idptl(ipp))
        nptl=nptl+1
        naddptl=nd
        goto 112


c     omeg and phi decay
c          use vectors in rest frame
c     ------------------------------
610   wt=(pptl(5,nptl+1)*pptl(5,nptl+2)*pptl(5,nptl+3))**2
     1-(pptl(5,nptl+1)*dot(2,3))**2
     2-(pptl(5,nptl+2)*dot(1,3))**2
     3-(pptl(5,nptl+3)*dot(1,2))**2
     4+2.*dot(1,2)*dot(2,3)*dot(1,3)
      if(wt.lt.rangen()*pptl(5,ip)**6/108.) goto300
      goto800

c     semileptonic and quark decays
c          use vectors in rest frame, where ip has (m,0,0,0)
c          include w propagator
c     ------------------------------------------------------
620   wt=(pptl(5,ip)*prest(4,2))*dot(1,3)
      s12=pptl(5,nptl+1)**2+pptl(5,nptl+2)**2+2.*dot(1,2)
      s12max=pptl(5,ip)**2
      wt=wt*wprop(s12max)/wprop(s12)
      if(wt.lt.rangen()*pptl(5,ip)**4/16.) goto 300
      goto 800

c     semileptonic kaon decays
c          use vectors in rest frame, where ip has (m,0,0,0)
c          include form factor FML
c     ------------------------------------------------------
630   if(iabs(idptl(ip)).eq.130)then
        if(iabs(idptl(nptl+2)).eq.12)then
          ncha=1          !K   -> Pi0 + e + Nu
        else
          ncha=2          !K   -> Pi0 + Mu + Nu
        endif
      else
        if(iabs(idptl(nptl+2)).eq.12)then
          ncha=3          !K0  -> Pi + e + Nu
        else
          ncha=4          !K0  -> Pi + Mu + Nu
        endif
      endif

      wt=FML(ncha,pptl(5,ip),pptl(5,nptl+1),pptl(5,nptl+2)
     &       ,prest(4,1),prest(4,2),prest(4,3))
      if(wt.lt.rangen()) goto 300
      goto 800

c     hadronic kaon decays
c          use vectors in rest frame, where ip has (m,0,0,0)
c          include form factor FM
c     ------------------------------------------------------
640   if(iabs(idptl(ip)).eq.130)then
        if(iabs(idptl(nptl+3)).eq.120)then
          ncha=1          !K   -> 3 Pi
        else
          ncha=2          !K   ->  Pi + 2 Pi0
        endif
      else
        if(iabs(idptl(nptl+1)).eq.110)then
          ncha=3          !K0  -> 3 Pi0
        else
          ncha=4          !K0  -> 2 Pi + Pi0
        endif
      endif
      S0=(pptl(5,ip)**2+pptl(5,nptl+1)**2+pptl(5,nptl+2)**2
     &   +pptl(5,nptl+3)**2)/3.d0
      S1=pptl(5,ip)**2+pptl(5,nptl+1)**2-2.*prest(4,1)*pptl(5,ip)
      S2=pptl(5,ip)**2+pptl(5,nptl+2)**2-2.*prest(4,2)*pptl(5,ip)
      S3=pptl(5,ip)**2+pptl(5,nptl+3)**2-2.*prest(4,3)*pptl(5,ip)
      wt=FM(ncha,S0,S1,S2,S3)
      if(wt.lt.rangen()) goto 300
      goto 800

c     muon decays
c          use vectors in rest frame, where ip has (m,0,0,0)
c          include form factor FMU
c     ------------------------------------------------------
650   xxx=2.*prest(4,1)/pptl(5,ip)            !reduced energy of electron
      if(xxx.gt.1.) goto 300
      wt=FMU(xxx)
      rrr=rangen()
      if(wt.lt.rrr) goto 300
      goto 800

c     one-particle decays
c     -------------------
700   continue
      do 710 j=1,5
      pptl(j,nptl+1)=pptl(j,ip)
710   continue

c     swap particles and antiparticles if idptl(ip)<0
c     -----------------------------------------------
 800    continue
        if(iabs(idptl(ip)).eq.80)then
          nptl=nptl-1
          naddptl=naddptl+1
        endif
        if(idptl(ipp).ge.0.or.iabs(idptl(ipp)).eq.20) goto 900
        do 810 i=1,naddptl
          idabs=iabs(idptl(nptl+i))
          ifl1=idabs/1000
          ifl2=mod(idabs/100,10)
          ifl3=mod(idabs/10,10)
          if(ifl1.eq.0.and.ifl2.ne.0.and.ifl2.eq.ifl3) goto 810
          if(idabs.eq.9.or.idabs.eq.10.or.idabs.eq.20) goto 810
          if(idabs.eq.29.or.idabs.eq.30.or.idabs.eq.40) goto 810
          idptl(nptl+i)=-idptl(nptl+i)
 810    continue

 900    continue
        nptl=nptl+naddptl
        if(nptl.gt.mxptl)call utstop('hdecay: nptl>mxptl&')
c        nqk=0           !???????????????????unused
        if(iabs(idptl(nptl)).lt.10.or.mod(idptl(nptl),100).eq.0)then
c          call utstop('hdecay: decay ptcl is parton&')
        endif

c     print
c     -----

      if(ish.ge.3)then
      write(ifch,140)sngl(ttaus)
  140 format(/' ----------------------------'/
     *'    decay  at tau =',f6.2/
     *' ----------------------------')
      write(ifch,*)'decaying object:'
      call alist('&',ip,ip)
      write(ifch,*)'decay products:'
      call alist('&',nptlb+1,nptl)
      endif
      if(ish.ge.5)then
      write(ifch,*)'momentum sum:'
      do kk=1,5
      pptl(kk,nptl+1)=0
      do ii=nptlb+1,nptl
      pptl(kk,nptl+1)=pptl(kk,nptl+1)+pptl(kk,ii)
      enddo
      enddo
      call alist('&',nptl+1,nptl+1)
      endif

c     exit
c     ----

 1000 continue
      ip=ipp
      if(iret.ne.0.and.ish.ge.1)then
        write(ifmt,'(a)')'hdecay: redo event'
        write(ifch,'(a)')'hdecay: redo event'
      endif
      call utprix('hdecay',ish,ishini,5)
      return

 9999   call utstop('hdecay: mxptl too small&')
        end

c---------------------------------------------------------------------
      subroutine vedi(k1,k2,k3,id)
c---------------------------------------------------------------------
      include 'epos.inc'
      if(k2.eq.0)then
        if(rangen().lt.pdiqua.and.iabs(k1).lt.6)then
          ifl1=int(rangen()/pud)+1
          ifl2=int(rangen()/pud)+1
          k3=-min(ifl1,ifl2)*1000-max(ifl1,ifl2)*100
        else
          k3=int(rangen()/pud)+1
        endif
        if(k1.gt.0.and.k1.le.6)k3=-k3
        if(k1.lt.-1000)k3=-k3
      else
        k3=k2
      endif
      id=idsp(k1,k3)
      if(iabs(id).le.999) then
        ids=max(mod(iabs(id)/100,10),mod(iabs(id)/10,10))
        if(ids.le.2)then
          idr=sign(iabs(id)+int(rangen()+0.5),id)
        elseif(ids.eq.3)then
          idr=sign(iabs(id)+int(rangen()+0.6),id)
        else
          idr=sign(iabs(id)+int(rangen()+0.75),id)
        endif
      else
        idr=sign(iabs(id)+int(0.5+rangen()),id)
      endif
      id=idr
      if(ish.ge.5)write(ifch,*) 'Flavor:',k1,k2,k3,id
      end

c-----------------------------------------------------------------------
      subroutine hdecin(lprint)
c-----------------------------------------------------------------------
c     sets up /dkytab/
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/wco/wmass2,wgam2
      dimension imode(6)
      character*8 idlabl,lmode(6),lres
      character*8 iblank
      logical lprint
      parameter (mxlook=10000,mxdky=2000)
      common/dkytab/look(mxlook),cbr(mxdky),mode(5,mxdky)
      common/nodcay/nodcay,noeta,nopi0,nonunu,noevol,nohadr
      logical nodcay,noeta,nopi0,nonunu,noevol,nohadr
      parameter (ndectb=1193)
      real dectab(7,ndectb)

      data ((dectab(i,j),i=1,7),j=  1, 18)/
     *  110., .98850,  10.,  10.,   0.,   0.,   0.
     *, 110.,1.00000,  10.,  12., -12.,   0.,   0.
     *, 220., .38000,  10.,  10.,   0.,   0.,   0.
     *, 220., .71000, 110., 110., 110.,   0.,   0.
     *, 220., .94600, 120.,-120., 110.,   0.,   0.
     *, 220., .99500, 120.,-120.,  10.,   0.,   0.
     *, 220.,1.00000,  10.,  12., -12.,   0.,   0.
     *, 330., .44100, 220., 120.,-120.,   0.,   0.
     *, 330., .66100, 220., 110., 110.,   0.,   0.
     *, 330., .95900, 111.,  10.,   0.,   0.,   0.
     *, 330., .98000, 221.,  10.,   0.,   0.,   0.
     *, 330.,1.00000,  10.,  10.,   0.,   0.,   0.
     *, 121.,1.00000, 120., 110.,   0.,   0.,   0.
     *, 111., .99989, 120.,-120.,   0.,   0.,   0.
     *, 111., .99993,  12., -12.,   0.,   0.,   0.
     *, 111.,1.00000,  14., -14.,   0.,   0.,   0.
     *, 221., .89900, 120.,-120., 110.,   0.,   0.
     *, 221., .91200, 120.,-120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j= 19, 36)/
     *  221., .99992, 110.,  10.,   0.,   0.,   0.
     *, 221.,1.00000,  12., -12.,   0.,   0.,   0.
     *, 331., .48600, 130.,-130.,   0.,   0.,   0.
     *, 331., .83700, 230.,-230.,   0.,   0.,   0.
     *, 331., .98400, 120.,-120., 110.,   0.,   0.
     *, 331., .99944, 220.,  10.,   0.,   0.,   0.
     *, 331., .99975,  12., -12.,   0.,   0.,   0.
     *, 331.,1.00000,  14., -14.,   0.,   0.,   0.
     *, 230., .50000,  20.,   0.,   0.,   0.,   0.
     *, 230.,1.00000, -20.,   0.,   0.,   0.,   0.
     *, 131., .66670, 230., 120.,   0.,   0.,   0.
     *, 131.,1.00000, 130., 110.,   0.,   0.,   0.
     *, 231., .66670, 130.,-120.,   0.,   0.,   0.
     *, 231.,1.00000, 230., 110.,   0.,   0.,   0.
     *, 240., .11000,  12., -11., 230.,   0.,   0.
     *, 240., .17000,  12., -11., 231.,   0.,   0.
     *, 240., .28000,  14., -13., 230.,   0.,   0.
     *, 240., .34000,  14., -13., 231.,   0.,   0./
      data ((dectab(i,j),i=1,7),j= 37, 54)/
     *  240., .37800, 230.,-120.,   0.,   0.,   0.
     *, 240., .56300, 230.,-121.,   0.,   0.,   0.
     *, 240., .60800, 231.,-120.,   0.,   0.,   0.
     *, 240., .62100, 230.,-120., 110.,   0.,   0.
     *, 240., .71000, 130.,-120.,-120.,   0.,   0.
     *, 240., .80100, 230.,-120.,-120., 120.,   0.
     *, 240., .87900, 130.,-120.,-120., 110.,   0.
     *, 240., .95400, 230.,-120., 110., 110.,   0.
     *, 240., .96600, 230.,-130.,   0.,   0.,   0.
     *, 240., .97600, 331.,-120.,   0.,   0.,   0.
     *, 240., .98800,-130., 231.,   0.,   0.,   0.
     *, 240.,1.00000,-131., 230.,   0.,   0.,   0.
     *, 140., .04500, -12.,  11., 130.,   0.,   0.
     *, 140., .07500, -12.,  11., 131.,   0.,   0.
     *, 140., .12000, -14.,  13., 130.,   0.,   0.
     *, 140., .15000, -14.,  13., 131.,   0.,   0.
     *, 140., .20300, 130.,-120.,   0.,   0.,   0.
     *, 140., .22700, 230., 110.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j= 55, 72)/
     *  140., .24700, 230., 220.,   0.,   0.,   0.
     *, 140., .28900, 230., 221.,   0.,   0.,   0.
     *, 140., .45100, 130.,-121.,   0.,   0.,   0.
     *, 140., .53600, 131.,-120.,   0.,   0.,   0.
     *, 140., .56200, 231., 110.,   0.,   0.,   0.
     *, 140., .57600, 230., 111.,   0.,   0.,   0.
     *, 140., .58700, 130.,-120., 110.,   0.,   0.
     *, 140., .60300, 230.,-120., 120.,   0.,   0.
     *, 140., .72700, 130.,-120.,-120., 120.,   0.
     *, 140., .87600, 230.,-120., 120., 110.,   0.
     *, 140., .96900, 130.,-120., 110., 110.,   0.
     *, 140.,1.00000, 230., 110., 110., 110.,   0.
     *, 340., .03250,  12., -11., 220.,   0.,   0.
     *, 340., .06500,  12., -11., 331.,   0.,   0.
     *, 340., .09750,  14., -13., 220.,   0.,   0.
     *, 340., .13000,  14., -13., 331.,   0.,   0.
     *, 340., .17900,-130., 230.,   0.,   0.,   0.
     *, 340., .22800,-120., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j= 73, 90)/
     *  340., .33800,-131., 230.,   0.,   0.,   0.
     *, 340., .44800,-130., 231.,   0.,   0.,   0.
     *, 340., .55800,-120., 331.,   0.,   0.,   0.
     *, 340., .57500,-130., 230., 110.,   0.,   0.
     *, 340., .59200,-230., 230.,-120.,   0.,   0.
     *, 340., .69400,-130., 230.,-120., 120.,   0.
     *, 340., .79600,-130., 230., 110., 110.,   0.
     *, 340., .89800,-130., 130.,-120., 110.,   0.
     *, 340.,1.00000,-230., 230.,-120., 110.,   0.
     *, 241., .64000, 140.,-120.,   0.,   0.,   0.
     *, 241., .92000, 240., 110.,   0.,   0.,   0.
     *, 241.,1.00000, 240.,  10.,   0.,   0.,   0.
     *, 141., .55000, 140., 110.,   0.,   0.,   0.
     *, 141.,1.00000, 140.,  10.,   0.,   0.,   0.
     *, 341.,1.00000, 340.,  10.,   0.,   0.,   0.
     *, 441., .07400,  12., -12.,   0.,   0.,   0.
     *, 441., .14800,  14., -14.,   0.,   0.,   0.
     *, 441., .15210,-121., 120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j= 91,108)/
     *  441., .15620, 111., 110.,   0.,   0.,   0.
     *, 441., .16020, 121.,-120.,   0.,   0.,   0.
     *, 441., .16300,-121., 111., 120.,   0.,   0.
     *, 441., .16580, 121.,-121., 110.,   0.,   0.
     *, 441., .16860, 121., 111.,-120.,   0.,   0.
     *, 441., .28740, 120.,-120., 130.,-130.,   0.
     *, 441., .40620, 110., 110., 130.,-130.,   0.
     *, 441., .52500, 120.,-120., 120.,-120.,   0.
     *, 441., .64380, 120.,-120., 110., 110.,   0.
     *, 441., .76260, 110., 110., 110., 110.,   0.
     *, 441., .88130, 120.,-120., 230.,-230.,   0.
     *, 441.,1.00000, 110., 110., 230., 230.,   0.
     *, 150., .06000, -12.,  11., 140.,   0.,   0.
     *, 150., .12000, -12.,  11., 141.,   0.,   0.
     *, 150., .18000, -14.,  13., 140.,   0.,   0.
     *, 150., .24000, -14.,  13., 141.,   0.,   0.
     *, 150., .25500, -16.,  15., 140.,   0.,   0.
     *, 150., .27000, -16.,  15., 141.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=109,122)/
     *  150., .28050, 140., 120.,   0.,   0.,   0.
     *, 150., .29100, 140., 121.,   0.,   0.,   0.
     *, 150., .30150, 141., 120.,   0.,   0.,   0.
     *, 150., .31200, 141., 121.,   0.,   0.,   0.
     *, 150., .32650, 140.,-340.,   0.,   0.,   0.
     *, 150., .34100, 140.,-341.,   0.,   0.,   0.
     *, 150., .35550, 141.,-340.,   0.,   0.,   0.
     *, 150., .37000, 141.,-341.,   0.,   0.,   0.
     *, 150., 0.820  ,   1.,  -4.,   1.,  -2.,   0.
     *, 150., 0.920  ,   1.,  -2.,   1.,  -4.,   0.
     *, 150., 0.975  ,   1.,  -4.,   4.,  -3.,   0.
     *, 150., 0.985  ,   1.,  -3.,   4.,  -4.,   0.
     *, 150., 0.995  ,   1.,  -1.,   1.,  -2.,   0.
     *, 150., 1.     ,   1.,  -1.,   4.,  -3.,   0./
      data ((dectab(i,j),i=1,7),j=123,142)
     */ 250., .06000, -12.,  11., 240.,   0.,   0.
     *, 250., .12000, -12.,  11., 241.,   0.,   0.
     *, 250., .18000, -14.,  13., 240.,   0.,   0.
     *, 250., .24000, -14.,  13., 241.,   0.,   0.
     *, 250., .25500, -16.,  15., 240.,   0.,   0.
     *, 250., .27000, -16.,  15., 241.,   0.,   0.
     *, 250., .28050, 240., 120.,   0.,   0.,   0.
     *, 250., .29100, 240., 121.,   0.,   0.,   0.
     *, 250., .30150, 241., 120.,   0.,   0.,   0.
     *, 250., .31200, 241., 121.,   0.,   0.,   0.
     *, 250., .32650, 240.,-340.,   0.,   0.,   0.
     *, 250., .34100, 240.,-341.,   0.,   0.,   0.
     *, 250., .35550, 241.,-340.,   0.,   0.,   0.
     *, 250., .37000, 241.,-341.,   0.,   0.,   0.
     *, 250., 0.820  ,   2.,  -4.,   1.,  -2.,   0.
     *, 250., 0.920  ,   2.,  -2.,   1.,  -4.,   0.
     *, 250., 0.975  ,   2.,  -4.,   4.,  -3.,   0.
     *, 250., 0.985  ,   2.,  -3.,   4.,  -4.,   0.
     *, 250., 0.995  ,   2.,  -1.,   1.,  -2.,   0.
     *, 250., 1.     ,   2.,  -1.,   4.,  -3.,   0./
      data ((dectab(i,j),i=1,7),j=143,176)/
     *     238*1. /
      data ((dectab(i,j),i=1,7),j=177,190)
     * /350., .06000,  12., -11., 340.,   0.,   0.
     *, 350., .12000,  12., -11., 341.,   0.,   0.
     *, 350., .18000,  14., -13., 340.,   0.,   0.
     *, 350., .24000,  14., -13., 341.,   0.,   0.
     *, 350., .25500,  16., -15., 340.,   0.,   0.
     *, 350., .27000,  16., -15., 341.,   0.,   0.
     *, 350., .28050, 340., 120.,   0.,   0.,   0.
     *, 350., .29100, 340., 121.,   0.,   0.,   0.
     *, 350., .30150, 341., 120.,   0.,   0.,   0.
     *, 350., .31200, 341., 121.,   0.,   0.,   0.
     *, 350., .32650, 340.,-340.,   0.,   0.,   0.
     *, 350., .34100, 340.,-341.,   0.,   0.,   0.
     *, 350., .35550, 341.,-340.,   0.,   0.,   0.
     *, 350., .37000, 341.,-341.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=191,196)/
     *  350., 0.820  ,   3.,  -4.,   1.,  -2.,   0.
     *, 350., 0.920  ,   3.,  -2.,   1.,  -4.,   0.
     *, 350., 0.975  ,   3.,  -4.,   4.,  -3.,   0.
     *, 350., 0.985  ,   3.,  -3.,   4.,  -4.,   0.
     *, 350., 0.995  ,   3.,  -1.,   1.,  -2.,   0.
     *, 350., 1.     ,   3.,  -1.,   4.,  -3.,   0./
      data ((dectab(i,j),i=1,7),j=197,244)/
     *     336*1. /
      data ((dectab(i,j),i=1,7),j=245,262)/
     *  160., .33330,  -1.,   2.,  -5.,   0.,   0.
     *, 160., .66660,  -4.,   3.,  -5.,   0.,   0.
     *, 160., .77770,  11., -12.,  -5.,   0.,   0.
     *, 160., .88880,  13., -14.,  -5.,   0.,   0.
     *, 160.,1.00000, -15.,  16.,  -5.,   0.,   0.
     *, 260., .33330,  -1.,   2.,  -5.,   0.,   0.
     *, 260., .66660,  -4.,   3.,  -5.,   0.,   0.
     *, 260., .77770, -11.,  12.,  -5.,   0.,   0.
     *, 260., .88880, -13.,  14.,  -5.,   0.,   0.
     *, 260.,1.00000, -15.,  16.,  -5.,   0.,   0.
     *, 360., .33330,  -1.,   2.,  -5.,   0.,   0.
     *, 360., .66660,  -4.,   3.,  -5.,   0.,   0.
     *, 360., .77770, -11.,  12.,  -5.,   0.,   0.
     *, 360., .88880, -13.,  14.,  -5.,   0.,   0.
     *, 360.,1.00000, -15.,  16.,  -5.,   0.,   0.
     *, 151.,1.00000, 150.,  10.,   0.,   0.,   0.
     *, 251.,1.00000, 250.,  10.,   0.,   0.,   0.
     *, 351.,1.00000, 350.,  10.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=263,280)/
     *  161.,1.00000, 160.,  10.,   0.,   0.,   0.
     *, 261.,1.00000, 260.,  10.,   0.,   0.,   0.
     *, 361.,1.00000, 360.,  10.,   0.,   0.,   0.
     *,1230.,1.00000,2130.,  10.,   0.,   0.,   0.
     *,1111.,1.00000,1120., 120.,   0.,   0.,   0.
     *,1121., .66670,1120., 110.,   0.,   0.,   0.
     *,1121.,1.00000,1220., 120.,   0.,   0.,   0.
     *,1221., .66670,1220., 110.,   0.,   0.,   0.
     *,1221.,1.00000,1120.,-120.,   0.,   0.,   0.
     *,2221.,1.00000,1220.,-120.,   0.,   0.,   0.
     *,1131., .88000,2130., 120.,   0.,   0.,   0.
     *,1131., .94000,1130., 110.,   0.,   0.,   0.
     *,1131.,1.00000,1230., 120.,   0.,   0.,   0.
     *,1231., .88000,2130., 110.,   0.,   0.,   0.
     *,1231., .94000,1130.,-120.,   0.,   0.,   0.
     *,1231.,1.00000,2230., 120.,   0.,   0.,   0.
     *,2231., .88000,2130.,-120.,   0.,   0.,   0.
     *,2231., .94000,1230.,-120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=281,298)/
     * 2231.,1.00000,2230., 110.,   0.,   0.,   0.
     *,1331., .66670,2330., 120.,   0.,   0.,   0.
     *,1331.,1.00000,1330., 110.,   0.,   0.,   0.
     *,2331., .66670,1330.,-120.,   0.,   0.,   0.
     *,2331.,1.00000,2330., 110.,   0.,   0.,   0.
     *,  16., .18000,  12., -11.,  15.,   0.,   0.
     *,  16., .36000,  14., -13.,  15.,   0.,   0.
     *,  16., .45100,-120.,  15.,   0.,   0.,   0.
     *,  16., .66000,-121.,  15.,   0.,   0.,   0.
     *,  16., .78000, 110., 110.,-120.,  15.,   0.
     *,  16., .83600, 120.,-120.,-120.,  15.,   0.
     *,  16.,1.00000, 120., 110.,-120.,-120.,  15.
     *,2140., .03750, -12.,  11.,2130.,   0.,   0.
     *,2140., .07500, -12.,  11.,1231.,   0.,   0.
     *,2140., .11250, -14.,  13.,2130.,   0.,   0.
     *,2140., .15000, -14.,  13.,1231.,   0.,   0.
     *,2140., .18200,2130., 120.,   0.,   0.,   0.
     *,2140., .21300,1230., 120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=299,316)/
     * 2140., .24400,1120.,-230.,   0.,   0.,   0.
     *,2140., .29500,1131., 110.,   0.,   0.,   0.
     *,2140., .34600,1231., 120.,   0.,   0.,   0.
     *,2140., .39700,1121.,-230.,   0.,   0.,   0.
     *,2140., .44800,1111.,-130.,   0.,   0.,   0.
     *,2140., .49900,1130., 111.,   0.,   0.,   0.
     *,2140., .55000,1230., 121.,   0.,   0.,   0.
     *,2140., .60100,1120.,-231.,   0.,   0.,   0.
     *,2140., .65800,1120.,-230., 120.,-120.,   0.
     *,2140., .71500,1120.,-230., 110., 110.,   0.
     *,2140., .77200,1120.,-130., 120., 110.,   0.
     *,2140., .82900,1220.,-230., 120., 110.,   0.
     *,2140., .88600,1220.,-130., 120., 120.,   0.
     *,2140., .94300,2130., 120., 120.,-120.,   0.
     *,2140.,1.00000,2130., 120., 110., 110.,   0.
     *,1140.,1.00000,2140., 120.,   0.,   0.,   0.
     *,1240.,1.00000,2140., 110.,   0.,   0.,   0.
     *,2240.,1.00000,2140.,-120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=317,334)/
     * 1340., .03750, -12.,  11.,1330.,   0.,   0.
     *,1340., .07500, -12.,  11.,1331.,   0.,   0.
     *,1340., .11250, -14.,  13.,1330.,   0.,   0.
     *,1340., .15000, -14.,  13.,1331.,   0.,   0.
     *,1340., .19900,1330., 120.,   0.,   0.,   0.
     *,1340., .24800,1231., 130.,   0.,   0.,   0.
     *,1340., .28800,1330., 120.,   0.,   0.,   0.
     *,1340., .32800,1131.,-230.,   0.,   0.,   0.
     *,1340., .36800,1330., 121.,   0.,   0.,   0.
     *,1340., .40800,1130.,-230.,   0.,   0.,   0.
     *,1340., .44800,1330., 120., 110.,   0.,   0.
     *,1340., .48800,2330., 120., 120.,   0.,   0.
     *,1340., .52800,1130.,-130., 120.,   0.,   0.
     *,1340., .56800,1130.,-230., 110.,   0.,   0.
     *,1340., .60800,1230.,-230., 120.,   0.,   0.
     *,1340., .66400,2130.,-230., 120., 110.,   0.
     *,1340., .72000,2130.,-130., 120., 120.,   0.
     *,1340., .77600,1130.,-230., 120.,-120.,   0./
      data ((dectab(i,j),i=1,7),j=335,352)/
     * 1340., .83200,1130.,-230., 110., 110.,   0.
     *,1340., .88800,1330., 120., 120.,-120.,   0.
     *,1340., .94400,1330., 120., 110., 110.,   0.
     *,1340.,1.00000,2330., 120., 120., 110.,   0.
     *,3140., .03750, -12.,  11.,1330.,   0.,   0.
     *,3140., .07500, -12.,  11.,1331.,   0.,   0.
     *,3140., .11250, -14.,  13.,1330.,   0.,   0.
     *,3140., .15000, -14.,  13.,1331.,   0.,   0.
     *,3140., .19900,1330., 120.,   0.,   0.,   0.
     *,3140., .24800,1231., 130.,   0.,   0.,   0.
     *,3140., .28800,1330., 120.,   0.,   0.,   0.
     *,3140., .32800,1131.,-230.,   0.,   0.,   0.
     *,3140., .36800,1330., 121.,   0.,   0.,   0.
     *,3140., .40800,1130.,-230.,   0.,   0.,   0.
     *,3140., .44800,1330., 120., 110.,   0.,   0.
     *,3140., .48800,2330., 120., 120.,   0.,   0.
     *,3140., .52800,1130.,-130., 120.,   0.,   0.
     *,3140., .56800,1130.,-230., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=353,370)/
     * 3140., .60800,1230.,-230., 120.,   0.,   0.
     *,3140., .66400,2130.,-230., 120., 110.,   0.
     *,3140., .72000,2130.,-130., 120., 120.,   0.
     *,3140., .77600,1130.,-230., 120.,-120.,   0.
     *,3140., .83200,1130.,-230., 110., 110.,   0.
     *,3140., .88800,1330., 120., 120.,-120.,   0.
     *,3140., .94400,1330., 120., 110., 110.,   0.
     *,3140.,1.00000,2330., 120., 120., 110.,   0.
     *,2340., .03750, -12.,  11.,2330.,   0.,   0.
     *,2340., .07500, -12.,  11.,2331.,   0.,   0.
     *,2340., .11250, -14.,  13.,2330.,   0.,   0.
     *,2340., .15000, -14.,  13.,2331.,   0.,   0.
     *,2340., .17500,2330., 120.,   0.,   0.,   0.
     *,2340., .20000,1330., 110.,   0.,   0.,   0.
     *,2340., .22500,1130.,-130.,   0.,   0.,   0.
     *,2340., .25000,1230.,-230.,   0.,   0.,   0.
     *,2340., .29500,2331., 120.,   0.,   0.,   0.
     *,2340., .34000,1331., 110.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=371,388)/
     * 2340., .38500,1131.,-130.,   0.,   0.,   0.
     *,2340., .43000,1231.,-230.,   0.,   0.,   0.
     *,2340., .47500,2330., 121.,   0.,   0.,   0.
     *,2340., .52000,1330., 111.,   0.,   0.,   0.
     *,2340., .56500,1130.,-131.,   0.,   0.,   0.
     *,2340., .61000,1230.,-231.,   0.,   0.,   0.
     *,2340., .64900,2130.,-230., 120.,-120.,   0.
     *,2340., .68800,2130.,-230., 110., 110.,   0.
     *,2340., .72700,2130.,-130., 120., 110.,   0.
     *,2340., .76600,1130.,-230.,-120., 110.,   0.
     *,2340., .80500,1130.,-130., 120.,-120.,   0.
     *,2340., .84400,1130.,-130., 110., 110.,   0.
     *,2340., .88300,1330., 120.,-120., 110.,   0.
     *,2340., .92200,1330., 110., 110., 110.,   0.
     *,2340., .96100,2330., 120., 120.,-120.,   0.
     *,2340.,1.00000,2330., 120., 110., 110.,   0.
     *,3240., .03750, -12.,  11.,2330.,   0.,   0.
     *,3240., .07500, -12.,  11.,2331.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=389,406)/
     * 3240., .11250, -14.,  13.,2330.,   0.,   0.
     *,3240., .15000, -14.,  13.,2331.,   0.,   0.
     *,3240., .17500,2330., 120.,   0.,   0.,   0.
     *,3240., .20000,1330., 110.,   0.,   0.,   0.
     *,3240., .22500,1130.,-130.,   0.,   0.,   0.
     *,3240., .25000,1230.,-230.,   0.,   0.,   0.
     *,3240., .29500,2331., 120.,   0.,   0.,   0.
     *,3240., .34000,1331., 110.,   0.,   0.,   0.
     *,3240., .38500,1131.,-130.,   0.,   0.,   0.
     *,3240., .43000,1231.,-230.,   0.,   0.,   0.
     *,3240., .47500,2330., 121.,   0.,   0.,   0.
     *,3240., .52000,1330., 111.,   0.,   0.,   0.
     *,3240., .56500,1130.,-131.,   0.,   0.,   0.
     *,3240., .61000,1230.,-231.,   0.,   0.,   0.
     *,3240., .64900,2130.,-230., 120.,-120.,   0.
     *,3240., .68800,2130.,-230., 110., 110.,   0.
     *,3240., .72700,2130.,-130., 120., 110.,   0.
     *,3240., .76600,1130.,-230.,-120., 110.,   0./
      data ((dectab(i,j),i=1,7),j=407,424)/
     * 3240., .80500,1130.,-130., 120.,-120.,   0.
     *,3240., .84400,1130.,-130., 110., 110.,   0.
     *,3240., .88300,1330., 120.,-120., 110.,   0.
     *,3240., .92200,1330., 110., 110., 110.,   0.
     *,3240., .96100,2330., 120., 120.,-120.,   0.
     *,3240.,1.00000,2330., 120., 110., 110.,   0.
     *,3340., .07500, -12.,  11.,3331.,   0.,   0.
     *,3340., .15000, -12.,  11.,3331.,   0.,   0.
     *,3340., .25000,1330.,-230.,   0.,   0.,   0.
     *,3340., .31000,3331., 120.,   0.,   0.,   0.
     *,3340., .37000,1331.,-230.,   0.,   0.,   0.
     *,3340., .43000,1330.,-231.,   0.,   0.,   0.
     *,3340., .49000,2330.,-230., 120.,   0.,   0.
     *,3340., .55000,1330.,-230., 110.,   0.,   0.
     *,3340., .61000,1330.,-130., 120.,   0.,   0.
     *,3340., .67500,3331., 120., 120.,-120.,   0.
     *,3340., .74000,3331., 120., 110., 110.,   0.
     *,3340., .80500,1330.,-230., 120.,-120.,   0./
      data ((dectab(i,j),i=1,7),j=425,442)/
     * 3340., .87000,1330.,-230., 110., 110.,   0.
     *,3340., .93500,2330.,-230., 120., 110.,   0.
     *,3340.,1.00000,2330.,-130., 120., 120.,   0.
     *,1141.,1.00000,2140., 120.,   0.,   0.,   0.
     *,1241.,1.00000,2140., 110.,   0.,   0.,   0.
     *,2241.,1.00000,2140.,-120.,   0.,   0.,   0.
     *,1341., .66670,2340., 120.,   0.,   0.,   0.
     *,1341.,1.00000,1340., 110.,   0.,   0.,   0.
     *,2341., .66670,1340.,-120.,   0.,   0.,   0.
     *,2341.,1.00000,2340., 110.,   0.,   0.,   0.
     *,3341.,1.00000,3340., 110.,   0.,   0.,   0.
     *,1150., .06000,  12., -11.,1140.,   0.,   0.
     *,1150., .12000,  12., -11.,1141.,   0.,   0.
     *,1150., .18000,  14., -13.,1140.,   0.,   0.
     *,1150., .24000,  14., -13.,1141.,   0.,   0.
     *,1150., .25500,  16., -15.,1140.,   0.,   0.
     *,1150., .27000,  16., -15.,1141.,   0.,   0.
     *,1150., .28925,1140.,-120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=443,460)/
     * 1150., .30850,1140.,-121.,   0.,   0.,   0.
     *,1150., .32775,1141.,-120.,   0.,   0.,   0.
     *,1150., .34700,1141.,-121.,   0.,   0.,   0.
     *,1150., .35775,1140., 340.,   0.,   0.,   0.
     *,1150., .36850,1140., 341.,   0.,   0.,   0.
     *,1150., .37925,1141., 340.,   0.,   0.,   0.
     *,1150., .39000,1141., 341.,   0.,   0.,   0.
     *,1150., .42050,1140.,-120., 110.,   0.,   0.
     *,1150., .45100,1140.,-120., 220.,   0.,   0.
     *,1150., .48150,1140.,-120., 111.,   0.,   0.
     *,1150., .51200,1140.,-120., 221.,   0.,   0.
     *,1150., .54250,1140.,-121., 110.,   0.,   0.
     *,1150., .57300,1140.,-121., 220.,   0.,   0.
     *,1150., .60350,1140.,-121., 111.,   0.,   0.
     *,1150., .63400,1140.,-121., 221.,   0.,   0.
     *,1150., .66450,1141.,-120., 110.,   0.,   0.
     *,1150., .69500,1141.,-120., 220.,   0.,   0.
     *,1150., .72550,1141.,-120., 111.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=461,478)/
     * 1150., .75600,1141.,-120., 221.,   0.,   0.
     *,1150., .78650,1141.,-121., 110.,   0.,   0.
     *,1150., .81700,1141.,-121., 220.,   0.,   0.
     *,1150., .84750,1141.,-121., 111.,   0.,   0.
     *,1150., .87800,1141.,-121., 221.,   0.,   0.
     *,1150., .89325,1140.,-130., 230.,   0.,   0.
     *,1150., .90850,1140.,-130., 231.,   0.,   0.
     *,1150., .92375,1140.,-131., 230.,   0.,   0.
     *,1150., .93900,1140.,-131., 231.,   0.,   0.
     *,1150., .95425,1141.,-130., 230.,   0.,   0.
     *,1150., .96950,1141.,-130., 231.,   0.,   0.
     *,1150., .98475,1141.,-131., 230.,   0.,   0.
     *,1150.,1.00000,1141.,-131., 231.,   0.,   0.
     *,1250., .06000,  12., -11.,1240.,   0.,   0.
     *,1250., .12000,  12., -11.,1241.,   0.,   0.
     *,1250., .18000,  14., -13.,1240.,   0.,   0.
     *,1250., .24000,  14., -13.,1241.,   0.,   0.
     *,1250., .25500,  16., -15.,1240.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=479,496)/
     * 1250., .27000,  16., -15.,1241.,   0.,   0.
     *,1250., .28925,1240.,-120.,   0.,   0.,   0.
     *,1250., .30850,1240.,-121.,   0.,   0.,   0.
     *,1250., .32775,1241.,-120.,   0.,   0.,   0.
     *,1250., .34700,1241.,-121.,   0.,   0.,   0.
     *,1250., .35775,1240., 340.,   0.,   0.,   0.
     *,1250., .36850,1240., 341.,   0.,   0.,   0.
     *,1250., .37925,1241., 340.,   0.,   0.,   0.
     *,1250., .39000,1241., 341.,   0.,   0.,   0.
     *,1250., .42050,1240.,-120., 110.,   0.,   0.
     *,1250., .45100,1240.,-120., 220.,   0.,   0.
     *,1250., .48150,1240.,-120., 111.,   0.,   0.
     *,1250., .51200,1240.,-120., 221.,   0.,   0.
     *,1250., .54250,1240.,-121., 110.,   0.,   0.
     *,1250., .57300,1240.,-121., 220.,   0.,   0.
     *,1250., .60350,1240.,-121., 111.,   0.,   0.
     *,1250., .63400,1240.,-121., 221.,   0.,   0.
     *,1250., .66450,1241.,-120., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=497,514)/
     * 1250., .69500,1241.,-120., 220.,   0.,   0.
     *,1250., .72550,1241.,-120., 111.,   0.,   0.
     *,1250., .75600,1241.,-120., 221.,   0.,   0.
     *,1250., .78650,1241.,-121., 110.,   0.,   0.
     *,1250., .81700,1241.,-121., 220.,   0.,   0.
     *,1250., .84750,1241.,-121., 111.,   0.,   0.
     *,1250., .87800,1241.,-121., 221.,   0.,   0.
     *,1250., .89325,1240.,-130., 230.,   0.,   0.
     *,1250., .90850,1240.,-130., 231.,   0.,   0.
     *,1250., .92375,1240.,-131., 230.,   0.,   0.
     *,1250., .93900,1240.,-131., 231.,   0.,   0.
     *,1250., .95425,1241.,-130., 230.,   0.,   0.
     *,1250., .96950,1241.,-130., 231.,   0.,   0.
     *,1250., .98475,1241.,-131., 230.,   0.,   0.
     *,1250.,1.00000,1241.,-131., 231.,   0.,   0.
     *,1350., .06000,  12., -11.,1340.,   0.,   0.
     *,1350., .12000,  12., -11.,1341.,   0.,   0.
     *,1350., .18000,  14., -13.,1340.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=515,532)/
     * 1350., .24000,  14., -13.,1341.,   0.,   0.
     *,1350., .25500,  16., -15.,1340.,   0.,   0.
     *,1350., .27000,  16., -15.,1341.,   0.,   0.
     *,1350., .28925,1340.,-120.,   0.,   0.,   0.
     *,1350., .30850,1340.,-121.,   0.,   0.,   0.
     *,1350., .32775,1341.,-120.,   0.,   0.,   0.
     *,1350., .34700,1341.,-121.,   0.,   0.,   0.
     *,1350., .35775,1340., 340.,   0.,   0.,   0.
     *,1350., .36850,1340., 341.,   0.,   0.,   0.
     *,1350., .37925,1341., 340.,   0.,   0.,   0.
     *,1350., .39000,1341., 341.,   0.,   0.,   0.
     *,1350., .42050,1340.,-120., 110.,   0.,   0.
     *,1350., .45100,1340.,-120., 220.,   0.,   0.
     *,1350., .48150,1340.,-120., 111.,   0.,   0.
     *,1350., .51200,1340.,-120., 221.,   0.,   0.
     *,1350., .54250,1340.,-121., 110.,   0.,   0.
     *,1350., .57300,1340.,-121., 220.,   0.,   0.
     *,1350., .60350,1340.,-121., 111.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=533,550)/
     * 1350., .63400,1340.,-121., 221.,   0.,   0.
     *,1350., .66450,1341.,-120., 110.,   0.,   0.
     *,1350., .69500,1341.,-120., 220.,   0.,   0.
     *,1350., .72550,1341.,-120., 111.,   0.,   0.
     *,1350., .75600,1341.,-120., 221.,   0.,   0.
     *,1350., .78650,1341.,-121., 110.,   0.,   0.
     *,1350., .81700,1341.,-121., 220.,   0.,   0.
     *,1350., .84750,1341.,-121., 111.,   0.,   0.
     *,1350., .87800,1341.,-121., 221.,   0.,   0.
     *,1350., .89325,1340.,-130., 230.,   0.,   0.
     *,1350., .90850,1340.,-130., 231.,   0.,   0.
     *,1350., .92375,1340.,-131., 230.,   0.,   0.
     *,1350., .93900,1340.,-131., 231.,   0.,   0.
     *,1350., .95425,1341.,-130., 230.,   0.,   0.
     *,1350., .96950,1341.,-130., 231.,   0.,   0.
     *,1350., .98475,1341.,-131., 230.,   0.,   0.
     *,1350.,1.00000,1341.,-131., 231.,   0.,   0.
     *,2150., .06000,  12., -11.,2140.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=551,568)/
     * 2150., .12000,  12., -11.,1241.,   0.,   0.
     *,2150., .18000,  14., -13.,2140.,   0.,   0.
     *,2150., .24000,  14., -13.,1241.,   0.,   0.
     *,2150., .25500,  16., -15.,2140.,   0.,   0.
     *,2150., .27000,  16., -15.,1241.,   0.,   0.
     *,2150., .28925,2140.,-120.,   0.,   0.,   0.
     *,2150., .30850,2140.,-121.,   0.,   0.,   0.
     *,2150., .32775,1241.,-120.,   0.,   0.,   0.
     *,2150., .34700,1241.,-121.,   0.,   0.,   0.
     *,2150., .35775,2140., 340.,   0.,   0.,   0.
     *,2150., .36850,2140., 341.,   0.,   0.,   0.
     *,2150., .37925,1241., 340.,   0.,   0.,   0.
     *,2150., .39000,1241., 341.,   0.,   0.,   0.
     *,2150., .42050,2140.,-120., 110.,   0.,   0.
     *,2150., .45100,2140.,-120., 220.,   0.,   0.
     *,2150., .48150,2140.,-120., 111.,   0.,   0.
     *,2150., .51200,2140.,-120., 221.,   0.,   0.
     *,2150., .54250,2140.,-121., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=569,586)/
     * 2150., .57300,2140.,-121., 220.,   0.,   0.
     *,2150., .60350,2140.,-121., 111.,   0.,   0.
     *,2150., .63400,2140.,-121., 221.,   0.,   0.
     *,2150., .66450,1241.,-120., 110.,   0.,   0.
     *,2150., .69500,1241.,-120., 220.,   0.,   0.
     *,2150., .72550,1241.,-120., 111.,   0.,   0.
     *,2150., .75600,1241.,-120., 221.,   0.,   0.
     *,2150., .78650,1241.,-121., 110.,   0.,   0.
     *,2150., .81700,1241.,-121., 220.,   0.,   0.
     *,2150., .84750,1241.,-121., 111.,   0.,   0.
     *,2150., .87800,1241.,-121., 221.,   0.,   0.
     *,2150., .89325,2140.,-130., 230.,   0.,   0.
     *,2150., .90850,2140.,-130., 231.,   0.,   0.
     *,2150., .92375,2140.,-131., 230.,   0.,   0.
     *,2150., .93900,2140.,-131., 231.,   0.,   0.
     *,2150., .95425,1241.,-130., 230.,   0.,   0.
     *,2150., .96950,1241.,-130., 231.,   0.,   0.
     *,2150., .98475,1241.,-131., 230.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=587,604)/
     * 2150.,1.00000,1241.,-131., 231.,   0.,   0.
     *,2250., .06000,  12., -11.,2240.,   0.,   0.
     *,2250., .12000,  12., -11.,2241.,   0.,   0.
     *,2250., .18000,  14., -13.,2240.,   0.,   0.
     *,2250., .24000,  14., -13.,2241.,   0.,   0.
     *,2250., .25500,  16., -15.,2240.,   0.,   0.
     *,2250., .27000,  16., -15.,2241.,   0.,   0.
     *,2250., .28925,2240.,-120.,   0.,   0.,   0.
     *,2250., .30850,2240.,-121.,   0.,   0.,   0.
     *,2250., .32775,2241.,-120.,   0.,   0.,   0.
     *,2250., .34700,2241.,-121.,   0.,   0.,   0.
     *,2250., .35775,2240., 340.,   0.,   0.,   0.
     *,2250., .36850,2240., 341.,   0.,   0.,   0.
     *,2250., .37925,2241., 340.,   0.,   0.,   0.
     *,2250., .39000,2241., 341.,   0.,   0.,   0.
     *,2250., .42050,2240.,-120., 110.,   0.,   0.
     *,2250., .45100,2240.,-120., 220.,   0.,   0.
     *,2250., .48150,2240.,-120., 111.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=605,622)/
     * 2250., .51200,2240.,-120., 221.,   0.,   0.
     *,2250., .54250,2240.,-121., 110.,   0.,   0.
     *,2250., .57300,2240.,-121., 220.,   0.,   0.
     *,2250., .60350,2240.,-121., 111.,   0.,   0.
     *,2250., .63400,2240.,-121., 221.,   0.,   0.
     *,2250., .66450,2241.,-120., 110.,   0.,   0.
     *,2250., .69500,2241.,-120., 220.,   0.,   0.
     *,2250., .72550,2241.,-120., 111.,   0.,   0.
     *,2250., .75600,2241.,-120., 221.,   0.,   0.
     *,2250., .78650,2241.,-121., 110.,   0.,   0.
     *,2250., .81700,2241.,-121., 220.,   0.,   0.
     *,2250., .84750,2241.,-121., 111.,   0.,   0.
     *,2250., .87800,2241.,-121., 221.,   0.,   0.
     *,2250., .89325,2240.,-130., 230.,   0.,   0.
     *,2250., .90850,2240.,-130., 231.,   0.,   0.
     *,2250., .92375,2240.,-131., 230.,   0.,   0.
     *,2250., .93900,2240.,-131., 231.,   0.,   0.
     *,2250., .95425,2241.,-130., 230.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=623,640)/
     * 2250., .96950,2241.,-130., 231.,   0.,   0.
     *,2250., .98475,2241.,-131., 230.,   0.,   0.
     *,2250.,1.00000,2241.,-131., 231.,   0.,   0.
     *,2350., .06000,  12., -11.,2340.,   0.,   0.
     *,2350., .12000,  12., -11.,2341.,   0.,   0.
     *,2350., .18000,  14., -13.,2340.,   0.,   0.
     *,2350., .24000,  14., -13.,2341.,   0.,   0.
     *,2350., .25500,  16., -15.,2340.,   0.,   0.
     *,2350., .27000,  16., -15.,2341.,   0.,   0.
     *,2350., .28925,2340.,-120.,   0.,   0.,   0.
     *,2350., .30850,2340.,-121.,   0.,   0.,   0.
     *,2350., .32775,2341.,-120.,   0.,   0.,   0.
     *,2350., .34700,2341.,-121.,   0.,   0.,   0.
     *,2350., .35775,2340., 340.,   0.,   0.,   0.
     *,2350., .36850,2340., 341.,   0.,   0.,   0.
     *,2350., .37925,2341., 340.,   0.,   0.,   0.
     *,2350., .39000,2341., 341.,   0.,   0.,   0.
     *,2350., .42050,2340.,-120., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=641,658)/
     * 2350., .45100,2340.,-120., 220.,   0.,   0.
     *,2350., .48150,2340.,-120., 111.,   0.,   0.
     *,2350., .51200,2340.,-120., 221.,   0.,   0.
     *,2350., .54250,2340.,-121., 110.,   0.,   0.
     *,2350., .57300,2340.,-121., 220.,   0.,   0.
     *,2350., .60350,2340.,-121., 111.,   0.,   0.
     *,2350., .63400,2340.,-121., 221.,   0.,   0.
     *,2350., .66450,2341.,-120., 110.,   0.,   0.
     *,2350., .69500,2341.,-120., 220.,   0.,   0.
     *,2350., .72550,2341.,-120., 111.,   0.,   0.
     *,2350., .75600,2341.,-120., 221.,   0.,   0.
     *,2350., .78650,2341.,-121., 110.,   0.,   0.
     *,2350., .81700,2341.,-121., 220.,   0.,   0.
     *,2350., .84750,2341.,-121., 111.,   0.,   0.
     *,2350., .87800,2341.,-121., 221.,   0.,   0.
     *,2350., .89325,2340.,-130., 230.,   0.,   0.
     *,2350., .90850,2340.,-130., 231.,   0.,   0.
     *,2350., .92375,2340.,-131., 230.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=659,720)/
     * 434*1./
      data ((dectab(i,j),i=1,7),j=721,738)/
     * 2350., .93900,2340.,-131., 231.,   0.,   0.
     *,2350., .95425,2341.,-130., 230.,   0.,   0.
     *,2350., .96950,2341.,-130., 231.,   0.,   0.
     *,2350., .98475,2341.,-131., 230.,   0.,   0.
     *,2350.,1.00000,2341.,-131., 231.,   0.,   0.
     *,3150., .06000,  12., -11.,3140.,   0.,   0.
     *,3150., .12000,  12., -11.,1341.,   0.,   0.
     *,3150., .18000,  14., -13.,3140.,   0.,   0.
     *,3150., .24000,  14., -13.,1341.,   0.,   0.
     *,3150., .25500,  16., -15.,3140.,   0.,   0.
     *,3150., .27000,  16., -15.,1341.,   0.,   0.
     *,3150., .28925,3140.,-120.,   0.,   0.,   0.
     *,3150., .30850,3140.,-121.,   0.,   0.,   0.
     *,3150., .32775,1341.,-120.,   0.,   0.,   0.
     *,3150., .34700,1341.,-121.,   0.,   0.,   0.
     *,3150., .35775,3140., 340.,   0.,   0.,   0.
     *,3150., .36850,3140., 341.,   0.,   0.,   0.
     *,3150., .37925,1341., 340.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=739,756)/
     * 3150., .39000,1341., 341.,   0.,   0.,   0.
     *,3150., .42050,3140.,-120., 110.,   0.,   0.
     *,3150., .45100,3140.,-120., 220.,   0.,   0.
     *,3150., .48150,3140.,-120., 111.,   0.,   0.
     *,3150., .51200,3140.,-120., 221.,   0.,   0.
     *,3150., .54250,3140.,-121., 110.,   0.,   0.
     *,3150., .57300,3140.,-121., 220.,   0.,   0.
     *,3150., .60350,3140.,-121., 111.,   0.,   0.
     *,3150., .63400,3140.,-121., 221.,   0.,   0.
     *,3150., .66450,1341.,-120., 110.,   0.,   0.
     *,3150., .69500,1341.,-120., 220.,   0.,   0.
     *,3150., .72550,1341.,-120., 111.,   0.,   0.
     *,3150., .75600,1341.,-120., 221.,   0.,   0.
     *,3150., .78650,1341.,-121., 110.,   0.,   0.
     *,3150., .81700,1341.,-121., 220.,   0.,   0.
     *,3150., .84750,1341.,-121., 111.,   0.,   0.
     *,3150., .87800,1341.,-121., 221.,   0.,   0.
     *,3150., .89325,3140.,-130., 230.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=757,774)/
     * 3150., .90850,3140.,-130., 231.,   0.,   0.
     *,3150., .92375,3140.,-131., 230.,   0.,   0.
     *,3150., .93900,3140.,-131., 231.,   0.,   0.
     *,3150., .95425,1341.,-130., 230.,   0.,   0.
     *,3150., .96950,1341.,-130., 231.,   0.,   0.
     *,3150., .98475,1341.,-131., 230.,   0.,   0.
     *,3150.,1.00000,1341.,-131., 231.,   0.,   0.
     *,3250., .06000,  12., -11.,3240.,   0.,   0.
     *,3250., .12000,  12., -11.,2341.,   0.,   0.
     *,3250., .18000,  14., -13.,3240.,   0.,   0.
     *,3250., .24000,  14., -13.,2341.,   0.,   0.
     *,3250., .25500,  16., -15.,3240.,   0.,   0.
     *,3250., .27000,  16., -15.,2341.,   0.,   0.
     *,3250., .28925,3240.,-120.,   0.,   0.,   0.
     *,3250., .30850,3240.,-121.,   0.,   0.,   0.
     *,3250., .32775,2341.,-120.,   0.,   0.,   0.
     *,3250., .34700,2341.,-121.,   0.,   0.,   0.
     *,3250., .35775,3240., 340.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=775,792)/
     * 3250., .36850,3240., 341.,   0.,   0.,   0.
     *,3250., .37925,2341., 340.,   0.,   0.,   0.
     *,3250., .39000,2341., 341.,   0.,   0.,   0.
     *,3250., .42050,3240.,-120., 110.,   0.,   0.
     *,3250., .45100,3240.,-120., 220.,   0.,   0.
     *,3250., .48150,3240.,-120., 111.,   0.,   0.
     *,3250., .51200,3240.,-120., 221.,   0.,   0.
     *,3250., .54250,3240.,-121., 110.,   0.,   0.
     *,3250., .57300,3240.,-121., 220.,   0.,   0.
     *,3250., .60350,3240.,-121., 111.,   0.,   0.
     *,3250., .63400,3240.,-121., 221.,   0.,   0.
     *,3250., .66450,2341.,-120., 110.,   0.,   0.
     *,3250., .69500,2341.,-120., 220.,   0.,   0.
     *,3250., .72550,2341.,-120., 111.,   0.,   0.
     *,3250., .75600,2341.,-120., 221.,   0.,   0.
     *,3250., .78650,2341.,-121., 110.,   0.,   0.
     *,3250., .81700,2341.,-121., 220.,   0.,   0.
     *,3250., .84750,2341.,-121., 111.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=793,810)/
     * 3250., .87800,2341.,-121., 221.,   0.,   0.
     *,3250., .89325,3240.,-130., 230.,   0.,   0.
     *,3250., .90850,3240.,-130., 231.,   0.,   0.
     *,3250., .92375,3240.,-131., 230.,   0.,   0.
     *,3250., .93900,3240.,-131., 231.,   0.,   0.
     *,3250., .95425,2341.,-130., 230.,   0.,   0.
     *,3250., .96950,2341.,-130., 231.,   0.,   0.
     *,3250., .98475,2341.,-131., 230.,   0.,   0.
     *,3250.,1.00000,2341.,-131., 231.,   0.,   0.
     *,3350., .06000,  12., -11.,3340.,   0.,   0.
     *,3350., .12000,  12., -11.,3341.,   0.,   0.
     *,3350., .18000,  14., -13.,3340.,   0.,   0.
     *,3350., .24000,  14., -13.,3341.,   0.,   0.
     *,3350., .25500,  16., -15.,3340.,   0.,   0.
     *,3350., .27000,  16., -15.,3341.,   0.,   0.
     *,3350., .28925,3340.,-120.,   0.,   0.,   0.
     *,3350., .30850,3340.,-121.,   0.,   0.,   0.
     *,3350., .32775,3341.,-120.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=811,828)/
     * 3350., .34700,3341.,-121.,   0.,   0.,   0.
     *,3350., .35775,3340., 340.,   0.,   0.,   0.
     *,3350., .36850,3340., 341.,   0.,   0.,   0.
     *,3350., .37925,3341., 340.,   0.,   0.,   0.
     *,3350., .39000,3341., 341.,   0.,   0.,   0.
     *,3350., .42050,3340.,-120., 110.,   0.,   0.
     *,3350., .45100,3340.,-120., 220.,   0.,   0.
     *,3350., .48150,3340.,-120., 111.,   0.,   0.
     *,3350., .51200,3340.,-120., 221.,   0.,   0.
     *,3350., .54250,3340.,-121., 110.,   0.,   0.
     *,3350., .57300,3340.,-121., 220.,   0.,   0.
     *,3350., .60350,3340.,-121., 111.,   0.,   0.
     *,3350., .63400,3340.,-121., 221.,   0.,   0.
     *,3350., .66450,3341.,-120., 110.,   0.,   0.
     *,3350., .69500,3341.,-120., 220.,   0.,   0.
     *,3350., .72550,3341.,-120., 111.,   0.,   0.
     *,3350., .75600,3341.,-120., 221.,   0.,   0.
     *,3350., .78650,3341.,-121., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=829,846)/
     * 3350., .81700,3341.,-121., 220.,   0.,   0.
     *,3350., .84750,3341.,-121., 111.,   0.,   0.
     *,3350., .87800,3341.,-121., 221.,   0.,   0.
     *,3350., .89325,3340.,-130., 230.,   0.,   0.
     *,3350., .90850,3340.,-130., 231.,   0.,   0.
     *,3350., .92375,3340.,-131., 230.,   0.,   0.
     *,3350., .93900,3340.,-131., 231.,   0.,   0.
     *,3350., .95425,3341.,-130., 230.,   0.,   0.
     *,3350., .96950,3341.,-130., 231.,   0.,   0.
     *,3350., .98475,3341.,-131., 230.,   0.,   0.
     *,3350.,1.00000,3341.,-131., 231.,   0.,   0.
     *,1160., .33300,   1.,  -2.,1500.,   0.,   0.
     *,1160., .66700,   4.,  -3.,1500.,   0.,   0.
     *,1160., .77800, -12.,  11.,1500.,   0.,   0.
     *,1160., .88900, -14.,  13.,1500.,   0.,   0.
     *,1160.,1.00000, -16.,  15.,1500.,   0.,   0.
     *,1260., .33300,   1.,  -2.,2500.,   0.,   0.
     *,1260., .66700,   4.,  -3.,2500.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=847,864)/
     * 1260., .77800, -12.,  11.,2500.,   0.,   0.
     *,1260., .88900, -14.,  13.,2500.,   0.,   0.
     *,1260.,1.00000, -16.,  15.,2500.,   0.,   0.
     *,2260., .33300,   1.,  -2.,2500.,   0.,   0.
     *,2260., .66700,   4.,  -3.,2500.,   0.,   0.
     *,2260., .77800, -12.,  11.,2500.,   0.,   0.
     *,2260., .88900, -14.,  13.,2500.,   0.,   0.
     *,2260.,1.00000, -16.,  15.,2500.,   0.,   0.
     *,2160., .33300,   1.,  -2.,1500.,   0.,   0.
     *,2160., .66700,   4.,  -3.,1500.,   0.,   0.
     *,2160., .77800, -12.,  11.,1500.,   0.,   0.
     *,2160., .88900, -14.,  13.,1500.,   0.,   0.
     *,2160.,1.00000, -16.,  15.,1500.,   0.,   0.
     *,1360., .33300,   1.,  -2.,3500.,   0.,   0.
     *,1360., .66700,   4.,  -3.,3500.,   0.,   0.
     *,1360., .77800, -12.,  11.,3500.,   0.,   0.
     *,1360., .88900, -14.,  13.,3500.,   0.,   0.
     *,1360.,1.00000, -16.,  15.,3500.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=865,882)/
     * 2360., .33300,   1.,  -2.,3500.,   0.,   0.
     *,2360., .66700,   4.,  -3.,3500.,   0.,   0.
     *,2360., .77800, -12.,  11.,3500.,   0.,   0.
     *,2360., .88900, -14.,  13.,3500.,   0.,   0.
     *,2360.,1.00000, -16.,  15.,3500.,   0.,   0.
     *,3360., .33300,   1.,  -2.,3500.,   0.,   0.
     *,3360., .66700,   4.,  -3.,3500.,   0.,   0.
     *,3360., .77800, -12.,  11.,3500.,   0.,   0.
     *,3360., .88900, -14.,  13.,3500.,   0.,   0.
     *,3360.,1.00000, -16.,  15.,3500.,   0.,   0.
     *,1151.,1.00000,1150.,  10.,   0.,   0.,   0.
     *,1251.,1.00000,1250.,  10.,   0.,   0.,   0.
     *,2251.,1.00000,2250.,  10.,   0.,   0.,   0.
     *,1351.,1.00000,1350.,  10.,   0.,   0.,   0.
     *,2351.,1.00000,2350.,  10.,   0.,   0.,   0.
     *,3351.,1.00000,3350.,  10.,   0.,   0.,   0.
     *,1161.,1.00000,1160.,  10.,   0.,   0.,   0.
     *,1261.,1.00000,1260.,  10.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=883,886)/
     * 2261.,1.00000,2260.,  10.,   0.,   0.,   0.
     *,1361.,1.00000,1360.,  10.,   0.,   0.,   0.
     *,2361.,1.00000,2360.,  10.,   0.,   0.,   0.
     *,3361.,1.00000,3360.,  10.,   0.,   0.,   0./
c    *---------------------------------------------
c    *    delta++ resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=887,900)/
c    *--dl++(1620)---------------------------------
     * 1112., .30000,1120., 120.,   0.,   0.,   0.
     *,1112., .66000,1111., 110.,   0.,   0.,   0.
     *,1112., .90000,1121., 120.,   0.,   0.,   0.
     *,1112.,1.00000,1120., 120., 110.,   0.,   0.
c    *--dl++(1700)---------------------------------
     *,1113., .15000,1120., 120.,   0.,   0.,   0.
     *,1113., .51000,1111., 110.,   0.,   0.,   0.
     *,1113., .75000,1121., 120.,   0.,   0.,   0.
     *,1113.,1.00000,1120., 120., 110.,   0.,   0.
c    *--dl++(1925)---------------------------------
     *,1114., .28000,1120., 120.,   0.,   0.,   0.
     *,1114., .40600,1111., 110.,   0.,   0.,   0.
     *,1114., .49000,1121., 120.,   0.,   0.,   0.
     *,1114., .69000,1120., 121.,   0.,   0.,   0.
     *,1114., .70000,1130., 130.,   0.,   0.,   0.
     *,1114.,1.00000,1122., 120.,   0.,   0.,   0./
c    *---------------------------------------------
c    *    delta- resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=901,914)/
c    *--dl-(1620)----------------------------------
     * 2222., .30000,1220.,-120.,   0.,   0.,   0.
     *,2222., .66000,2221., 110.,   0.,   0.,   0.
     *,2222., .90000,1221.,-120.,   0.,   0.,   0.
     *,2222.,1.00000,1220., 110.,-120.,   0.,   0.
c    *--dl-(1700)----------------------------------
     *,2223., .15000,1220.,-120.,   0.,   0.,   0.
     *,2223., .51000,2221., 110.,   0.,   0.,   0.
     *,2223., .75000,1221.,-120.,   0.,   0.,   0.
     *,2223.,1.00000,1220., 110.,-120.,   0.,   0.
c    *--dl-(1925)----------------------------------
     *,2224., .28000,1220.,-120.,   0.,   0.,   0.
     *,2224., .40600,2221., 110.,   0.,   0.,   0.
     *,2224., .49000,1221.,-120.,   0.,   0.,   0.
     *,2224., .69000,1220.,-121.,   0.,   0.,   0.
     *,2224., .70000,2230., 230.,   0.,   0.,   0.
     *,2224.,1.00000,1222.,-120.,   0.,   0.,   0./
c    *---------------------------------------------
c    *    n*+ resonances + delta+ resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=915,931)/
c    *--n*+(1440)----------------------------------
     * 1122., .20000,1120., 110.,   0.,   0.,   0.
     *,1122., .60000,1220., 120.,   0.,   0.,   0.
     *,1122., .68000,1111.,-120.,   0.,   0.,   0.
     *,1122., .73000,1121., 110.,   0.,   0.,   0.
     *,1122., .76000,1221., 120.,   0.,   0.,   0.
     *,1122., .84000,1120., 120.,-120.,   0.,   0.
     *,1122., .87000,1120., 110., 110.,   0.,   0.
     *,1122.,1.00000,1220., 120., 110.,   0.,   0.
c    *--n*+(1530)----------------------------------
     *,1123., .17000,1120., 110.,   0.,   0.,   0.
     *,1123., .51000,1220., 120.,   0.,   0.,   0.
     *,1123., .57000,1111.,-120.,   0.,   0.,   0.
     *,1123., .61000,1121., 110.,   0.,   0.,   0.
     *,1123., .63000,1221., 120.,   0.,   0.,   0.
     *,1123., .67000,1120., 120.,-120.,   0.,   0.
     *,1123., .68000,1120., 110., 110.,   0.,   0.
     *,1123., .75000,1220., 120., 110.,   0.,   0.
     *,1123.,1.00000,1120., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=932,948)/
c    *--dl+(1620)----------------------------------
     * 1124., .20000,1120., 110.,   0.,   0.,   0.
     *,1124., .30000,1220., 120.,   0.,   0.,   0.
     *,1124., .54000,1111.,-120.,   0.,   0.,   0.
     *,1124., .58000,1121., 110.,   0.,   0.,   0.
     *,1124., .90000,1221., 120.,   0.,   0.,   0.
     *,1124., .96000,1120., 120.,-120.,   0.,   0.
     *,1124.,1.00000,1220., 120., 110.,   0.,   0.
c    *--n*+(1665)----------------------------------
     *,1125., .16700,1120., 110.,   0.,   0.,   0.
     *,1125., .49970,1220., 120.,   0.,   0.,   0.
     *,1125., .62470,1111.,-120.,   0.,   0.,   0.
     *,1125., .70800,1121., 110.,   0.,   0.,   0.
     *,1125., .74970,1221., 120.,   0.,   0.,   0.
     *,1125., .82080,1120., 120.,-120.,   0.,   0.
     *,1125., .85190,1120., 110., 110.,   0.,   0.
     *,1125., .96300,1220., 120., 110.,   0.,   0.
     *,1125., .97300,1120., 220.,   0.,   0.,   0.
     *,1125.,1.00000,2130., 130.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=949,955)/
c    *--dl+(1700)----------------------------------
     * 1126., .10000,1120., 110.,   0.,   0.,   0.
     *,1126., .15000,1220., 120.,   0.,   0.,   0.
     *,1126., .39000,1111.,-120.,   0.,   0.,   0.
     *,1126., .43000,1121., 110.,   0.,   0.,   0.
     *,1126., .75000,1221., 120.,   0.,   0.,   0.
     *,1126., .91500,1120., 120.,-120.,   0.,   0.
     *,1126.,1.00000,1220., 120., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=956,969)/
c    *--n*+(1710)----------------------------------
     * 1127., .04430,1120., 110.,   0.,   0.,   0.
     *,1127., .13290,1220., 120.,   0.,   0.,   0.
     *,1127., .23790,1111.,-120.,   0.,   0.,   0.
     *,1127., .30790,1121., 110.,   0.,   0.,   0.
     *,1127., .34290,1221., 120.,   0.,   0.,   0.
     *,1127., .41190,1120., 120.,-120.,   0.,   0.
     *,1127., .48090,1120., 110., 110.,   0.,   0.
     *,1127., .54990,1220., 120., 110.,   0.,   0.
     *,1127., .66070,1120., 220.,   0.,   0.,   0.
     *,1127., .72800,2130., 130.,   0.,   0.,   0.
     *,1127., .74930,1230., 130.,   0.,   0.,   0.
     *,1127., .76000,1130., 230.,   0.,   0.,   0.
     *,1127., .84000,1120., 111.,   0.,   0.,   0.
     *,1127.,1.00000,1220., 121.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=970,980)/
c    *--dl+(1925)----------------------------------
     * 1128., .18700,1120., 110.,   0.,   0.,   0.
     *,1128., .28000,1220., 120.,   0.,   0.,   0.
     *,1128., .36400,1111.,-120.,   0.,   0.,   0.
     *,1128., .37800,1121., 110.,   0.,   0.,   0.
     *,1128., .49000,1221., 120.,   0.,   0.,   0.
     *,1128., .62300,1120., 111.,   0.,   0.,   0.
     *,1128., .69000,1220., 121.,   0.,   0.,   0.
     *,1128., .69350,1130., 230.,   0.,   0.,   0.
     *,1128., .69900,1230., 130.,   0.,   0.,   0.
     *,1128., .89900,1122., 110.,   0.,   0.,   0.
     *,1128.,1.00000,1222., 120.,   0.,   0.,   0./
c    *---------------------------------------------
c    *    n*0  resonances + delta0 resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=981,997)/
c    *----------n*0(1440)--------------------------
     * 1222., .20000,1220., 110.,   0.,   0.,   0.
     *,1222., .60000,1120.,-120.,   0.,   0.,   0.
     *,1222., .68000,2221., 120.,   0.,   0.,   0.
     *,1222., .73000,1221., 110.,   0.,   0.,   0.
     *,1222., .76000,1121.,-120.,   0.,   0.,   0.
     *,1222., .84000,1220., 120.,-120.,   0.,   0.
     *,1222., .87000,1220., 110., 110.,   0.,   0.
     *,1222.,1.00000,1120.,-120., 110.,   0.,   0.
c    *----------n*0(1530)--------------------------
     *,1223., .17000,1220., 110.,   0.,   0.,   0.
     *,1223., .51000,1120.,-120.,   0.,   0.,   0.
     *,1223., .57000,2221., 120.,   0.,   0.,   0.
     *,1223., .61000,1221., 110.,   0.,   0.,   0.
     *,1223., .63000,1121.,-120.,   0.,   0.,   0.
     *,1223., .67000,1220., 120.,-120.,   0.,   0.
     *,1223., .68000,1220., 110., 110.,   0.,   0.
     *,1223., .75000,1120.,-120., 110.,   0.,   0.
     *,1223.,1.00000,1220., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=998,1014)/
c    *----------dl0(1620)--------------------------
     * 1224., .20000,1220., 110.,   0.,   0.,   0.
     *,1224., .30000,1120.,-120.,   0.,   0.,   0.
     *,1224., .54000,2221., 120.,   0.,   0.,   0.
     *,1224., .58000,1221., 110.,   0.,   0.,   0.
     *,1224., .90000,1121.,-120.,   0.,   0.,   0.
     *,1224., .96500,1220., 120.,-120.,   0.,   0.
     *,1224.,1.00000,1120.,-120., 110.,   0.,   0.
c    *----------n*0(1665)--------------------------
     *,1225., .16700,1220., 110.,   0.,   0.,   0.
     *,1225., .49970,1120.,-120.,   0.,   0.,   0.
     *,1225., .62470,2221., 120.,   0.,   0.,   0.
     *,1225., .70800,1221., 110.,   0.,   0.,   0.
     *,1225., .74970,1121.,-120.,   0.,   0.,   0.
     *,1225., .82080,1220., 120.,-120.,   0.,   0.
     *,1225., .85190,1220., 110., 110.,   0.,   0.
     *,1225., .96300,1120.,-120., 110.,   0.,   0.
     *,1225., .97300,1220., 220.,   0.,   0.,   0.
     *,1225.,1.00000,2130., 230.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1015,1021)/
c    *----------dl0(1700)--------------------------
     * 1226., .10000,1220., 110.,   0.,   0.,   0.
     *,1226., .15000,1120.,-120.,   0.,   0.,   0.
     *,1226., .39000,2221., 120.,   0.,   0.,   0.
     *,1226., .43000,1221., 110.,   0.,   0.,   0.
     *,1226., .75000,1121.,-120.,   0.,   0.,   0.
     *,1226., .91500,1220., 120.,-120.,   0.,   0.
     *,1226.,1.00000,1120.,-120., 110.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1022,1035)/
c    *----------n*0(1710)--------------------------
     * 1227., .04430,1220., 110.,   0.,   0.,   0.
     *,1227., .13290,1120.,-120.,   0.,   0.,   0.
     *,1227., .23790,2221., 120.,   0.,   0.,   0.
     *,1227., .30790,1221., 110.,   0.,   0.,   0.
     *,1227., .34290,1121.,-120.,   0.,   0.,   0.
     *,1227., .41190,1220., 120.,-120.,   0.,   0.
     *,1227., .48090,1220., 110., 110.,   0.,   0.
     *,1227., .54990,1120.,-120., 110.,   0.,   0.
     *,1227., .66070,1220., 220.,   0.,   0.,   0.
     *,1227., .72800,2130., 230.,   0.,   0.,   0.
     *,1227., .73870,1230., 230.,   0.,   0.,   0.
     *,1227., .76000,2230., 130.,   0.,   0.,   0.
     *,1227., .92000,1120.,-121.,   0.,   0.,   0.
     *,1227.,1.00000,1220., 111.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1036,1046)/
c    *----------dl0(1925)--------------------------
     * 1228., .18700,1220., 110.,   0.,   0.,   0.
     *,1228., .28000,1120.,-120.,   0.,   0.,   0.
     *,1228., .36400,2221., 120.,   0.,   0.,   0.
     *,1228., .37800,1221., 110.,   0.,   0.,   0.
     *,1228., .49000,1121.,-120.,   0.,   0.,   0.
     *,1228., .55700,1220., 111.,   0.,   0.,   0.
     *,1228., .69000,1120.,-121.,   0.,   0.,   0.
     *,1228., .69350,2230., 130.,   0.,   0.,   0.
     *,1228., .70000,1230., 230.,   0.,   0.,   0.
     *,1228., .80000,1122.,-120.,   0.,   0.,   0.
     *,1228.,1.00000,1222., 110.,   0.,   0.,   0./
c    *---------------------------------------------
c    *   lambda resonances + sigma0 resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1047,1059)/
c    *----------lambda(1405)-----------------------
     * 1233., .33000,1230., 110.,   0.,   0.,   0.
     *,1233., .66000,2230., 120.,   0.,   0.,   0.
     *,1233.,1.00000,1130.,-120.,   0.,   0.,   0.
c    *----------lambda(1520)-----------------------
     *,1234., .22500,1120.,-130.,   0.,   0.,   0.
     *,1234., .48000,1220.,-230.,   0.,   0.,   0.
     *,1234., .62000,1230., 110.,   0.,   0.,   0.
     *,1234., .76000,2230., 120.,   0.,   0.,   0.
     *,1234., .90000,1130.,-120.,   0.,   0.,   0.
     *,1234., .96000,2130., 120.,-120.,   0.,   0.
     *,1234., .99000,2130., 110., 110.,   0.,   0.
     *,1234., .99330,1130.,-120., 110.,   0.,   0.
     *,1234., .99660,2230., 120., 110.,   0.,   0.
     *,1234.,1.00000,1230., 120.,-120.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1060,1075)/
c    *----------lambda(1645)-----------------------
     * 1235., .10000,1120.,-130.,   0.,   0.,   0.
     *,1235., .20000,1220.,-230.,   0.,   0.,   0.
     *,1235., .35000,1230., 110.,   0.,   0.,   0.
     *,1235., .50000,2230., 120.,   0.,   0.,   0.
     *,1235., .65000,1130.,-120.,   0.,   0.,   0.
     *,1235., .75000,2130., 120.,-120.,   0.,   0.
     *,1235., .80000,2130., 110., 110.,   0.,   0.
     *,1235., .84500,1130.,-120., 110.,   0.,   0.
     *,1235., .89000,2230., 120., 110.,   0.,   0.
     *,1235., .93500,1230., 120.,-120.,   0.,   0.
     *,1235.,1.00000,2130., 220.,   0.,   0.,   0.
c    *----------sigma0(1665)-----------------------
     *,1236., .10000,1120.,-130.,   0.,   0.,   0.
     *,1236., .20000,1220.,-230.,   0.,   0.,   0.
     *,1236., .40000,2230., 120.,   0.,   0.,   0.
     *,1236., .60000,1130.,-120.,   0.,   0.,   0.
     *,1236.,1.00000,2130., 110.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1076,1084)/
c    *----------sigma0(1776)-----------------------
     * 1237., .17500,1120.,-130.,   0.,   0.,   0.
     *,1237., .35000,1220.,-230.,   0.,   0.,   0.
     *,1237., .38750,2230., 120.,   0.,   0.,   0.
     *,1237., .42500,1130.,-120.,   0.,   0.,   0.
     *,1237., .57500,2130., 110.,   0.,   0.,   0.
     *,1237., .60000,2231., 120.,   0.,   0.,   0.
     *,1237., .62500,1131.,-120.,   0.,   0.,   0.
     *,1237., .75000,1234., 110.,   0.,   0.,   0.
     *,1237.,1.00000,1230., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1085,1094)/
c    *----------lambda(1845)-----------------------
     * 1238., .17000,1120.,-130.,   0.,   0.,   0.
     *,1238., .34000,1220.,-230.,   0.,   0.,   0.
     *,1238., .44000,1230., 110.,   0.,   0.,   0.
     *,1238., .54000,2230., 120.,   0.,   0.,   0.
     *,1238., .64000,1130.,-120.,   0.,   0.,   0.
     *,1238., .70000,1231., 110.,   0.,   0.,   0.
     *,1238., .76000,2231., 120.,   0.,   0.,   0.
     *,1238., .82000,1131.,-120.,   0.,   0.,   0.
     *,1238., .91000,1120.,-131.,   0.,   0.,   0.
     *,1238.,1.00000,1220.,-231.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1095,1106)/
c    *----------sigma0(1930)-----------------------
     * 1239., .07500,1120.,-130.,   0.,   0.,   0.
     *,1239., .15000,1220.,-230.,   0.,   0.,   0.
     *,1239., .20000,1121.,-130.,   0.,   0.,   0.
     *,1239., .25000,1221.,-230.,   0.,   0.,   0.
     *,1239., .32500,1120.,-131.,   0.,   0.,   0.
     *,1239., .40000,1220.,-231.,   0.,   0.,   0.
     *,1239., .47500,2230., 120.,   0.,   0.,   0.
     *,1239., .55000,1130.,-120.,   0.,   0.,   0.
     *,1239., .70000,2130., 110.,   0.,   0.,   0.
     *,1239., .77500,2231., 120.,   0.,   0.,   0.
     *,1239., .85000,1131.,-120.,   0.,   0.,   0.
     *,1239.,1.00000,1234., 110.,   0.,   0.,   0./
c    *---------------------------------------------
c    *            sigma+ resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1107,1118)/
c    *----------sigma+(1665)-----------------------
     * 1132., .20000,1120.,-230.,   0.,   0.,   0.
     *,1132., .40000,1130., 110.,   0.,   0.,   0.
     *,1132., .60000,1230., 120.,   0.,   0.,   0.
     *,1132.,1.00000,2130., 120.,   0.,   0.,   0.
c    *----------sigma+(1776)-----------------------
     *,1133., .35000,1120.,-230.,   0.,   0.,   0.
     *,1133., .38750,1130., 110.,   0.,   0.,   0.
     *,1133., .42500,1230., 120.,   0.,   0.,   0.
     *,1133., .57500,2130., 120.,   0.,   0.,   0.
     *,1133., .60000,1131., 110.,   0.,   0.,   0.
     *,1133., .62500,1231., 120.,   0.,   0.,   0.
     *,1133., .75000,1234., 120.,   0.,   0.,   0.
     *,1133.,1.00000,1130., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1119,1128)/
c    *----------sigma+(1930)-----------------------
     * 1134., .15000,1120.,-230.,   0.,   0.,   0.
     *,1134., .22500,1111.,-130.,   0.,   0.,   0.
     *,1134., .25000,1121.,-230.,   0.,   0.,   0.
     *,1134., .40000,1120.,-231.,   0.,   0.,   0.
     *,1134., .47500,1130., 110.,   0.,   0.,   0.
     *,1134., .55000,1230., 120.,   0.,   0.,   0.
     *,1134., .70000,2130., 120.,   0.,   0.,   0.
     *,1134., .77500,1131., 110.,   0.,   0.,   0.
     *,1134., .85000,1231., 120.,   0.,   0.,   0.
     *,1134.,1.00000,1234., 120.,   0.,   0.,   0./
c    *---------------------------------------------
c    *            sigma- resonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1129,1140)/
c    *----------sigma-(1665)-----------------------
     * 2232., .20000,1220.,-130.,   0.,   0.,   0.
     *,2232., .40000,2230., 110.,   0.,   0.,   0.
     *,2232., .60000,1230.,-120.,   0.,   0.,   0.
     *,2232.,1.00000,2130.,-120.,   0.,   0.,   0.
c    *----------sigma-(1776)-----------------------
     *,2233., .35000,1220.,-130.,   0.,   0.,   0.
     *,2233., .38750,2230., 110.,   0.,   0.,   0.
     *,2233., .42500,1230.,-120.,   0.,   0.,   0.
     *,2233., .57500,2130.,-120.,   0.,   0.,   0.
     *,2233., .60000,2231., 110.,   0.,   0.,   0.
     *,2233., .62500,1231.,-120.,   0.,   0.,   0.
     *,2233., .75000,1234.,-120.,   0.,   0.,   0.
     *,2233.,1.00000,2230., 220.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1141,1150)/
c    *----------sigma-(1930)-----------------------
     * 2234., .15000,1220.,-130.,   0.,   0.,   0.
     *,2234., .17500,1221.,-130.,   0.,   0.,   0.
     *,2234., .25000,2221.,-230.,   0.,   0.,   0.
     *,2234., .40000,1220.,-131.,   0.,   0.,   0.
     *,2234., .47500,2230., 110.,   0.,   0.,   0.
     *,2234., .55000,1230.,-120.,   0.,   0.,   0.
     *,2234., .70000,2130.,-120.,   0.,   0.,   0.
     *,2234., .77500,2231., 110.,   0.,   0.,   0.
     *,2234., .85000,1231.,-120.,   0.,   0.,   0.
     *,2234.,1.00000,1234.,-120.,   0.,   0.,   0./
c    *---------------------------------------------
c    *      additional mesonresonances
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1151,1159)/
c    *-----------f0(975)---------------------------
     *  332., .50000, 120.,-120.,   0.,   0.,   0.
     *, 332., .75000, 110., 110.,   0.,   0.,   0.
     *, 332., .87500, 130.,-130.,   0.,   0.,   0.
     *, 332.,1.00000, 230.,-230.,   0.,   0.,   0.
c    *-----------a0(980)---------------------------
     *, 112., .56000, 110., 220.,   0.,   0.,   0.
     *, 112., .78000, 130.,-130.,   0.,   0.,   0.
     *, 112.,1.00000, 230.,-230.,   0.,   0.,   0.
c    *-----------a+(980)---------------------------
     *, 122., .60000, 120., 220.,   0.,   0.,   0.
     *, 122.,1.00000, 130.,-230.,   0.,   0.,   0./
c    *---------------------------------------------
c    *      weak baryon decays
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1160,1169)/
c    *-----------lambda(1116)----------------------
     * 2130.,0.64200,1120.,-120.,   0.,   0.,   0.
     *,2130.,1.00000,1220., 110.,   0.,   0.,   0.
c    *-----------sigma+(1180)----------------------
     *,1130.,0.51580,1120., 110.,   0.,   0.,   0.
     *,1130.,1.00000,1220., 120.,   0.,   0.,   0.
c    *-----------sigma-(1180)----------------------
     *,2230.,1.00000,1220.,-120.,   0.,   0.,   0.
c    *---------kaskade-(1360)----------------------
     *,2330.,1.00000,2130.,-120.,   0.,   0.,   0.
c    *---------kaskade0(1360)----------------------
     *,1330.,1.00000,2130., 110.,   0.,   0.,   0.
c    *---------omega-(1680)------------------------
     *,3331.,0.68000,2130.,-130.,   0.,   0.,   0.
     *,3331.,0.82000,1330.,-120.,   0.,   0.,   0.
     *,3331.,1.00000,2330., 110.,   0.,   0.,   0./
c    *---------------------------------------------
c    *      weak meson decays
c    *---------------------------------------------
      data ((dectab(i,j),i=1,7),j=1170,1171)/
c    *-----------k0s()--------------------------
     *   20., .68610, 120.,-120.,   0.,   0.,   0.
     *,  20.,1.00000, 110., 110.,   0.,   0.,   0./
      data ((dectab(i,j),i=1,7),j=1172,ndectb)/
c    *-----------k0l-------------------------------
     *  320., .2113, 110., 110., 110.,   0.,   0.
     *, 320., .2113, 110., 110., 110.,   0.,   0.
     *, 320., .2120, 110., 110., 110.,   0.,   0.
     *, 320., .3380, 120.,-120., 110.,   0.,   0.
     *, 320., .4744, 120.,  14., -13.,   0.,   0.
     *, 320., .6108,-120., -14.,  13.,   0.,   0.
     *, 320., .8054, 120.,  12., -11.,   0.,   0.
     *, 320.,1.0000,-120., -12.,  11.,   0.,   0.
c    *-----------k+-------------------------------
     *, 130., .6352 , -14.,  13.,   0.,   0.,   0.
     *, 130., .8468 , 120., 110.,   0.,   0.,   0.
     *, 130., .9027 , 120., 120.,-120.,   0.,   0.
     *, 130., .92   , 120., 110., 110.,   0.,   0.
     *, 130., .9518 , 110., -14.,  13.,   0.,   0.
     *, 130.,1.     , 110., -12.,  11.,   0.,   0.
c    *-----------pi+------------------------------
     *, 120., 1.    , -14.,  13.,   0.,   0.,   0.
c    *-----------mu-------------------------------
     *,  14., 1.    ,  12., -11.,  13.,   0.,   0.
c    *-----------etac-------------------------------
     *, 440.,  .32  , 230.,-230., 110.,   0.,   0.
     *, 440.,  .64  , 220., 110., 110.,   0.,   0.
     *, 440.,  .76  , 120.,-120., 130.,-130.,   0.
     *, 440.,  .88  , 120.,-120., 120.,-120.,   0.
     *, 440., 1.    , 130.,-130., 130.,-130.,   0.
c    *-----------etac-------------------------------
     *,1220., 1.    ,1120.,  12., -11.,   0.,   0./

      call idresi

c     determine wmass2,wgam2
c     ----------------------
      alfa=1./137.036
      gf=1.16570e-5
      sin2w=.215
      sinw=sqrt(sin2w)
c      cosw=sqrt(1.-sin2w)           !?????????????????unused
      amw=sqrt(pi*alfa/(.9304*sqrt(2.)*gf))/sinw
      wmass2=amw
      call idmass(5,amlep5)
      call idmass(6,amlep6)
      ngam=12
      if(amlep5+amlep6.gt.amw) ngam=9
      wgam2=gf*amw**3/(6.*pi*sqrt(2.))*ngam

      data iblank/' '/
      ird=0
      do 1 i=1,mxlook
1     look(i)=0
      do 2 i=1,mxdky
      do 3 j=1,5
3     mode(j,i)=0
2     cbr(i)=0.
      nodcay=.false.
      noeta=.false.
      nopi0=.false.
      nonunu=.false.
      noevol=.false.
      nohadr=.false.
      if(lprint) write(ifch,10)
10    format('1',30('*')/' *',28x,'*'/
     1' *',5x,'isajet decay table',5x,'*'/
     2' *',28x,'*'/' ',30('*')//
     36x,'part',18x,'decay mode',19x,'cum br',15x,'ident',17x,
     4'decay ident')
      loop=0
      iold=0
      if(nodcay) return

200   loop=loop+1
      if(loop.gt.mxdky) goto9999
220   do 210 i=1,5
      imode(i)=0
      lmode(i)=iblank
210   continue
      ird=ird+1
      if(ird.gt.ndectb)return
c      if(ird.gt.1171)return   ! ??????????????????????????
      ires=nint(dectab(1,ird))
      br=dectab(2,ird)
      do 215 i=1,5
215   imode(i)=nint(dectab(2+i,ird))
      if(nopi0.and.ires.eq.110) goto220
      if(noeta.and.ires.eq.220) goto220
      if(ires.eq.iold) goto230
      if(ires.lt.0.or.ires.gt.mxlook)
     *call utstop('hdecin: ires out of range&')
      look(ires)=loop
230   iold=ires
      cbr(loop)=br
      do 240 i=1,5
      mode(i,loop)=imode(i)
      if(imode(i).ne.0) lmode(i)=idlabl(imode(i))
240   continue
      lres=idlabl(ires)
      if(lprint) write(ifch,20) lres,(lmode(k),k=1,5),
     1br,ires,(imode(k),k=1,5)
20    format(6x,a5,6x,5(a5,2x),3x,f8.5,15x,i5,4x,5(i5,2x))
      goto200

9999  write(ifch,*)'loop=', loop
      call utstop('hdecin: loop > mxdky&')

      end

C -----------------------------------------------
      FUNCTION FM(NQ,S0,S1,S2,S3)
C -----------------------------------------------
C Normalized TRANSITION MATRIX FOR THE DALIZT PLOT DISTRI.
C OF K -> 3 PIONS. PARAMETRIZATION OF WEINBERG
C AS DESCRIBE IN PARTICLE DATA BOOK.
C G IS THE LINEAR COEFFICIENT (SLOPE g)
C H IS THE QUADRATIC COEFFICIENT h
C D IS THE QUADRATIC COEFFICIENT k
C Amax is the maximum of this amplitude (taken from Corsika by D. Heck)
C NQ is the decay channel :
C   1 - K -> 3 Pi
C   2 - K -> Pi + 2 Pi0
C   3 - K0 -> 3 Pi0
C   4 - K0 -> 2 Pi + Pi0
C -----------------------------------------------
      DIMENSION G(4),H(4),D(4),Amax(4)
      PARAMETER (PIM=139.57E-3)
      DATA G/-0.2154,0.594,0.,0.67/
      DATA H/0.01,0.035,0.,0.079/
      DATA D/-0.01,0.,0.,0.0098/
      DATA Amax/1.27,1.84,1.,2.22/

      FM=1.+G(NQ)*(S3-S0)/(PIM*PIM)+H(NQ)*((S3-S0)/(PIM*PIM))**2
     *+D(NQ)*((S2-S1)/(PIM*PIM))**2
      FM=FM/Amax(NQ)

      RETURN
      END
C -----------------------------------------------
      FUNCTION FML(N,AM,RM1,RM2,E1S,E2S,E3S)
C -----------------------------------------------
C Normalized DALITZ PLOT DENSITY (RHO)
C OF K -> 1 PION + 2 LEPTONS
C AS DESCRIBE IN PARTICLE DATA BOOK.
C CLP IS THE LAMBDA + FORM FACTOR COEFFICIENT
C CLN IS THE LAMBDA 0 FORM FACTOR COEFFICIENT
C EEP IS E'pion
C GP IS THE F+(t) FORM FACTOR (t=AM*AM+SM1-2.D0*AM*E1S)
C H IS EPS(t)=F-(t)/F+(t) WHERE F- IS CALCULATED FROM F0
C Amax is the maximum of this density (taken from Corsika by D. Heck)
C N is the decay channel :
C   1 - K -> Pi0 + e + Nu
C   2 - K -> Pi0 + Mu + Nu
C   3 - K0 -> Pi + e + Nu
C   4 - K0 -> Pi + Mu + Nu
C -----------------------------------------------
      DIMENSION CLP(4),CLN(4),Amax(4)
      DATA CLP/0.0276,0.031,0.0288,0.034/
      DATA CLN/0.0,0.006,0.,0.025/
      DATA Amax/1.28e-2,1.194e-2,1.31e-2,1.241e-2/

      SM1=RM1*RM1
      SM2=RM2*RM2
      EEP=0.5D0*(AM*AM+SM1-SM2)/AM-E1S
      GP=1.+CLP(N)*(AM*AM+SM1-2.*AM*E1S)/SM1
      H=(AM*AM-SM1)/SM1*(CLN(N)-CLP(N))/GP
      FML=GP*GP*(AM*(2.*E2S*E3S-AM*EEP)+
     *SM2*(0.25*EEP-E3S)+H*SM2*(E3S-0.5*EEP)+
     *0.25*H*H*SM2*EEP)
      FML=FML/Amax(N)
      RETURN
      END
C -----------------------------------------------
      FUNCTION FMU(X)
C -----------------------------------------------
C PROBABILITY DISTRI. FOR ELECTRON ENERGY FROM MUON DECAY :
C MU -> 2NU + E. DESCRIBE IN PARTICLE DATA BOOK.
C (SIMPLIFY DIFFERENTIAL DECAY RATE INTEGRATED)
C X REDUCED ENERGY OF PARTICLE
C -----------------------------------------------

      FMU=2.*(3.-2.*X)*X*X

      RETURN
      END
