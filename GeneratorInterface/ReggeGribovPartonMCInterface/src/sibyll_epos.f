c 15.02.2004 Link routines between SIBYLL2.1 and EPOS.
c author T. Pierog


c-----------------------------------------------------------------------
      subroutine IniSibyll
c-----------------------------------------------------------------------
c Primary initialization for Sibyll
c-----------------------------------------------------------------------
      include 'epos.inc'
      COMMON /S_DEBUG/ Ncall, Ndebug
      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)

      call utpri('inisib',ish,ishini,6)
      write(ifmt,'(a,i6)')'initialize Sibyll ...'

      Ndebug=0
      if(ish.ge.3)Ndebug=ish-2

C... SIBYLL initialization
      CALL SIBYLL_INI

C...Cross sections for nucleus-nucleus and hadron nucleus
      CALL NUC_NUC_INI

C...define all particles as unstable
      do i=1,49
        IDB(i) = abs(IDB(i))   ! >0 means unstable
      enddo


      egymin=10.1
      egymax=1e7
      irescl=0
      

      call utprix('inisib',ish,ishini,6)
      end

c-----------------------------------------------------------------------
      subroutine IniEvtSib
c-----------------------------------------------------------------------
c Initialization for each type of event (for given proj, targ and egy)
c-----------------------------------------------------------------------
      include 'epos.inc'
      COMMON /S_CSYDEC/ CBR(102), KDEC(612), LBARP(49), IDB(49)
      common/geom/rmproj,rmtarg,bmax,bkmx


      if(matarg.gt.18.or.maproj.gt.64)
     &  call utstop('Mass too big for Sibyll (Mtrg<18, Mprj<64) !&',
     +sizeof('Mass too big for Sibyll (Mtrg<18, Mprj<64) !&'))
      id=idtrafo('nxs','sib',idproj)
      ida=abs(id)
      if(ida.lt.6.or.ida.gt.14)
     &  call utstop('projectile no allowed in Sibyll !&',
     +sizeof('projectile no allowed in Sibyll !&'))
      if(idtarg.ne.0.and.idtarg.ne.1120)
     &  call utstop('target no allowed in Sibyll !&',
     +sizeof('target no allowed in Sibyll !&'))
      if(bminim.gt.0.or.bmaxim.lt.1000)
     &  write(ifmt,*)'Only min bias event in Sibyll ... no b !'
      call iclass(idproj,iclpro)
      bminim=0.
      bmaxim=10000.
      bmax=10.+maproj+matarg
      sibincs=fsibcrse(engy,maproj,matarg)
      if(engy.lt.egymin)sibincs=0.          !below egymin, no interaction
      call xsigma


c Epos syntax to allow (or not) particle decay in Sibyll
c (part taken from epos-dky: hdecas)

      if(idecay.eq.1.and.ctaumin.le.0.)then

      if(ndecay.eq.1.or.mod(ndecay/10,10).eq.1)then          !Kshort, Klong
        IDB(11) = -abs(IDB(11))   ! <0 means stable
        IDB(12) = -abs(IDB(12))
      endif
      if(ndecay.eq.1.or.mod(ndecay/100,10).eq.1)then         !Lambda
        IDB(39) = -abs(IDB(39))
      endif
      if(ndecay.eq.1.or.mod(ndecay/1000,10).eq.1)then        !sigma+
        IDB(34) = -abs(IDB(34))
      endif
      if(ndecay.eq.1.or.mod(ndecay/1000,10).eq.1)then        !sigma-
        IDB(36) = -abs(IDB(36))
      endif
      if(ndecay.eq.1.or.mod(ndecay/10000,10).eq.1)then       !Xi+/-
        IDB(38) = -abs(IDB(38))
      endif
      if(ndecay.eq.1.or.mod(ndecay/10000,10).eq.1)then       !Xi0
        IDB(37) = -abs(IDB(37))
      endif
      if(ndecay.eq.1.or.mod(ndecay/100000 ,10).eq.1)then     !omega
        IDB(49) = -abs(IDB(49))
      endif
      if(ndecay.eq.1.or.mod(ndecay/1000000,10).eq.1)then     !pi0
        IDB(6) = -abs(IDB(6))
      endif

      if(nrnody.gt.0)then                      !all other particle
        do nod=1,nrnody
          idd=abs(idtrafo('nxs','sib',nody(nod)))
          if(idd.lt.50)IDB(idd) = -abs(IDB(idd))
        enddo
      endif

      else

C...define all particles as stable
      do i=1,49
        IDB(i) = -abs(IDB(i))   ! <0 means stable
      enddo

      endif

      if(ish.ge.2)write(ifch,*)
     &  'Sibyll used with (E,proj,maproj,matarg)',engy,id,maproj
     &  ,matarg

      return
      end

c-----------------------------------------------------------------------
      subroutine emssib(iret)
c-----------------------------------------------------------------------
c  call Sibyll to simulate interaction
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/geom/rmproj,rmtarg,bmax,bkmx
C  SIBYLL
      COMMON /S_PLIST/ P(8000,5), LLIST(8000), NP
      COMMON /S_PLNUC/ PA(5,40000), LLA(40000), NPA
      PARAMETER (NW_max = 20)
      PARAMETER (NS_max = 20, NH_max = 50)
      PARAMETER (NJ_max = (NS_max+NH_max)*NW_max)
      COMMON /S_CHIST/ X1J(NJ_max),X2J(NJ_max),
     &    X1JSUM(NW_max),X2JSUM(NW_max),PTJET(NJ_max),PHIJET(NJ_max),
     &    NNPJET(NJ_max),NNPSTR(2*NW_max),NNSOF(NW_max),NNJET(NW_max),
     &    JDIF(NW_max),NW,NJET,NSOF

      iret=0
      ncol=0
      b1=bminim
      b2=min(bmax,bmaxim)
      a=pi*(b2**2-b1**2)

      if(a.gt.0..and.rangen().gt.sibincs/10./a)goto 1001   !no interaction
      if(ish.ge.3)call alist('Determine Sibyll Production&',
     +sizeof('Determine Sibyll Production&'),0,0)

      nptl=0
      NP=0
      NPA=0
      
      ncol=1
      nevt=1
      kolevt=ncol
      npjevt=maproj
      ntgevt=matarg
      pmxevt=pnll
      egyevt=engy
      bimevt=0.
      bimp=0.
      phievt=0.
      phi=0.
      anintine=anintine+1.

      call conre
      call conwr

      itrg=matarg
      if(idtargin.eq.0)itrg=0
      if(maproj.eq.1)then             !hadronic projectile
        L0=idtrafo('nxs','sib',idproj)
        CALL SIBYLL (L0, itrg, engy)
        CALL DECSIB
        if(ish.ge.5)write(ifch,'(a,i5)')
     $         ' number of particles from Sibyll :',NP
c save interaction type
        if(JDIF(1).eq.0)then
          typevt=1                !ND
        elseif(JDIF(1).eq.3)then
          typevt=2                !DD
        else
          typevt=3                !SD
        endif
        do k=1,NP

c LLIST is the code of final particle, P - its 4-momentum and mass.
          ic=LLIST(k)
            
          if(ish.ge.7)write(ifch,'(a,i5,a,i5,2a,4(e10.4,1x))')
     $       ' Sibyll particle ',k,' id :',ic,' before conversion'
     $     , ' momentum :',(P(k,i),i=1,5)

          nptl=nptl+1
          if(nptl.gt.mxptl)call utstop('Sibyll: mxptl too small&',
     +sizeof('Sibyll: mxptl too small&'))

          if(abs(ic).ge.10000)then
            ic=ic-sign(10000,ic)
            istptl(nptl)=1
          else
            istptl(nptl)=0
          endif

          id=idtrafo('sib','nxs',ic)
          if(ish.ge.7)write(ifch,'(a,i5,a,i5,a)')
     $         ' epos particle ',nptl,' id :',id,' after conversion'
            

          pptl(1,nptl)=P(k,1)   !P_x
          pptl(2,nptl)=P(k,2)   !P_y
          pptl(3,nptl)=P(k,3)   !P_z
          pptl(4,nptl)=abs(P(k,4))   !E
          pptl(5,nptl)=P(k,5)   !mass
          ityptl(nptl)=0
          iorptl(nptl)=1
          jorptl(nptl)=maproj+matarg
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=0.
          xorptl(2,nptl)=0.
          xorptl(3,nptl)=0.
          xorptl(4,nptl)=0.
          tivptl(1,nptl)=0.
          tivptl(2,nptl)=0.
          idptl(nptl)=id
          
            
          if(ish.ge.5)write(ifch,'(a,i5,a,i5,a,4(e10.4,1x),f6.3)')
     $         ' particle from Sibyll ',nptl,' id :',idptl(nptl)
     $         , ' momentum :',(pptl(i,nptl),i=1,5)


        enddo
        ns=0
        if(NW.lt.matarg)then
          ns=matarg-NW
          do is=maproj+1,maproj+ns !make the ns first target nucleon actives
            istptl(is)=0
          enddo
        endif
      else                          !for nucleus projectile
        ns=0           !number of projectile spectators
        nbar=0
        IAP = maproj
        CALL SIBNUC (IAP, itrg, engy)
        if(ish.ge.5)write(ifch,'(a,i5)')
     $         ' number of particles from Sibyll :',NPA
        do 100 k=1,NPA

c LLIST is the code of final particle, P - its 4-momentum and mass.
          ic=LLA(k)
            
          if(ish.ge.7)write(ifch,'(a,i5,a,i5,2a,4(e10.4,1x))')
     $       ' Sibyll particle ',k,' id :',ic,' before conversion'
     $     , ' momentum :',(PA(i,k),i=1,5)


          nNuc=0
          if(ic.ge.1001) then                !count spectators
            nNuc=ic-1000
            if(infragm.le.1
     &         .or.dble(PA(4,k))/dble(nNuc).lt.egymin)then   !nuclear interaction only above min energy, otherwise : fragmentation
              ns=ns+nNuc
              goto 100
            elseif(ic.eq.1001)then
              if(drangen(dummy).lt.0.45d0) then
                ic = 13
              else
                ic = 14
              endif
              nNuc=0
            else
              ptm=sqrt(PA(1,k)*PA(1,k)+PA(2,k)*PA(2,k)+PA(5,k)*PA(5,k))
              PA(4,k)=PA(4,k)*float(nNuc)            !energy by nucleon
              PA(3,k)=sign(sqrt((PA(4,k)+ptm)*(PA(4,k)-ptm)),PA(3,k))
            endif
          endif
          nptl=nptl+1
          if(nptl.gt.mxptl)call utstop('Sibyll: mxptl too small&',
     +sizeof('Sibyll: mxptl too small&'))
          id=idtrafo('sib','nxs',ic)
          if(ish.ge.7)write(ifch,'(a,i5,a,i10,a)')
     $         ' epos particle ',nptl,' id :',id,' after conversion'
            

          nbar=nbar+nNuc
          if(abs(id).gt.1000.and.nNuc.eq.0)nbar=nbar+sign(1,id)
          pptl(1,nptl)=PA(1,k)  !P_x
          pptl(2,nptl)=PA(2,k)  !P_y
          pptl(3,nptl)=PA(3,k)  !P_z
          pptl(4,nptl)=PA(4,k)  !E
          pptl(5,nptl)=PA(5,k)  !mass
          istptl(nptl)=0
          ityptl(nptl)=0
          iorptl(nptl)=1
          jorptl(nptl)=maproj+matarg
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=0.
          xorptl(2,nptl)=0.
          xorptl(3,nptl)=0.
          xorptl(4,nptl)=0.
          tivptl(1,nptl)=0.
          tivptl(2,nptl)=0.
          idptl(nptl)=id
          
            
          if(ish.ge.5)write(ifch,'(a,i5,a,i10,a,4(e10.4,1x),f6.3)')
     $         ' particle from Sibyll ',nptl,' id :',idptl(nptl)
     $         , ' momentum :',(pptl(i,nptl),i=1,5)

 100    continue

        ntw=nbar-(maproj-ns)
        nsf=0
        if(ntw.lt.matarg)then
          nts=matarg-ntw
          do is=maproj+1,maproj+nts !make the nts first target nucleon actives (not wounded)
            istptl(is)=0
          enddo
        else
          nsf=maproj+matarg-nbar
        endif
          if(ish.ge.5)write(ifch,'(a,i3,a,i3,a,i2,a)')
     $         ' target spectators :',matarg-ntw
     $        ,' projectile spectators (ns) :',nsf,' (',ns,')'
        if((infragm.le.1.or.nsf.gt.ns).and.nsf.le.maproj)then
          if(infragm.eq.2)ns=nsf-ns
          if(infragm.eq.1.and.ns.gt.0)then
c  remaining nucleus is one fragment
            nptl=nptl+1
            istptl(nptl)=0
            pptl(1,nptl)=0.d0
            pptl(2,nptl)=0.d0
            pptl(4,nptl)=0.d0
            inucl=0
            do is=1,ns
              inucl=inucl+1
              pptl(4,nptl)=pptl(4,nptl)+pptl(4,is)
            enddo
            iprot= int(dble(inucl) / 2.15d0 + 0.7d0)
            idnucl=1000000000+iprot*10000+inucl*10 !code for nuclei
            call idmass(idnucl,am)
            pptl(5,nptl)=am  !mass
            ptot=(pptl(4,nptl)+am)*(pptl(4,nptl)-am)
            pptl(3,nptl)=sqrt(ptot)
            ityptl(nptl)=0
            istptl(nptl)=0
            iorptl(nptl)=1
            jorptl(nptl)=maproj
            ifrptl(1,nptl)=0
            ifrptl(2,nptl)=0
            xorptl(1,nptl)=xorptl(1,1)
            xorptl(2,nptl)=xorptl(2,1)
            xorptl(3,nptl)=xorptl(3,1)
            xorptl(4,nptl)=xorptl(4,1)
            tivptl(1,nptl)=tivptl(1,1)
            tivptl(2,nptl)=tivptl(2,1)
            idptl(nptl)=idnucl
          else
            do is=1,ns         !make the nsf first projectile nucleon actives (not wounded)
              istptl(is)=0
            enddo
          endif
        endif
      endif


1000  return

1001  iret=-1 
      goto 1000 

      end


c------------------------------------------------------------------------------
      function fsibcrse(egy,mapro,matar)
c------------------------------------------------------------------------------
c hadron-proton particle production cross section with Sibyll.
c egy - center of mass energy
c------------------------------------------------------------------------------
      include 'epos.inc'
      dimension SDIF(3)

      if(iclpro.eq.1)then
        L=2
      elseif(iclpro.eq.2)then
        L=1
      else
        L=3
      endif
      call SIB_SIGMA_HP(L,egy,ST,SEL,SINEL,SDIF,SL,RHO)
      if(matar.gt.1)then
C  calculate hadron-A(matar) cross section
        CALL GLAUBER(matar,ST,SL,RHO,SIGTA,SIGELAdum,SIGQEA)
        fsibcrse=SIGTA-SIGQEA
      else
        fsibcrse=SINEL
      endif

      if(mapro.gt.1)fsibcrse=ainfin !???????? temporary

      return
      end

c------------------------------------------------------------------------------
      double precision function sibcrse(ek,mapro,matar,id)
c------------------------------------------------------------------------------
c inelastic cross section of Sibyll 
c if id=0, target = air
c ek - kinetic energy in GeV
c maproj - projec mass number     (1<maproj<64)
c matarg - projec mass number
c id - proj id (sibyll code)
c------------------------------------------------------------------------------
      include 'epos.inc'
      double precision egy
      COMMON /CLENNN/ SSIGNUC(60), ALNUC(60)

      sibcrse=0.d0
      call idmass(1120,amt1)
      call idmass(1220,amt2)
      amtar=0.5*(amt1+amt2)
      if(mapro.eq.1)call idmass(idproj,ampro)
      if(matar.eq.1)call idmass(idtarg,amtar)
      egy=dble(ek+ampro)
      if(id.eq.0)then
        if(maproj.eq.1)then
          sqs=sqrt( 2*real(egy)*amtar+amtar**2+ampro**2 )
          if(iclpro.eq.1)then
            L=2
          elseif(iclpro.eq.2)then
            L=1
          else
            L=3
          endif
          call SIB_SIGMA_HAIR (L,sqs,sibcr)
          sibcrse=dble(sibcr)
        else
          EO=real(egy)*1.e-3         !e0 in TeV
          CALL  SIGNUC_INI(mapro,E0) !  fills SSIGNUC and ALNUC
          sibcrse  = dble(SSIGNUC(mapro))
        endif
      else
        sqs=sqrt( 2*real(egy)*amtar+amtar**2+ampro**2 )
        sibcrse=dble(fsibcrse(sqs,mapro,matar))
      endif

      return
      end

c--------------------------------------------------------------------
      function S_RNDM(idum)
c--------------------------------------------------------------------
c random number generator
c--------------------------------------------------------------------
      include 'epos.inc'

      S_RNDM=rangen()
      if(irandm.eq.1)write(ifch,*)'S_RNDM()= ',S_RNDM,idum

      return
      end
 


