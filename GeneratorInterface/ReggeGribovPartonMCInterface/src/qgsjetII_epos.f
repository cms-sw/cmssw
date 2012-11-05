c 07.06.2006 Link routines between QGSJet-II-03 and EPOS.
c author T. Pierog


c-----------------------------------------------------------------------
      subroutine IniQGSJETII
c-----------------------------------------------------------------------
c Primary initialization for QGSJET-II
c-----------------------------------------------------------------------
      parameter(iapmax=208)
      include 'epos.inc'
      integer debug
      common /qgdebug/ debug
      common /qgarr43/ moniou
      double precision bqgs2,bmaxqgs,bmaxnex,bminnex,xan,xbn
      common /qgsIInex1/xan(iapmax,3),xbn(iapmax,3)
     *,bqgs2,bmaxqgs,bmaxnex,bminnex
      CHARACTER DATDIR*(132)

      call utpri('iniqgsII',ish,ishini,6)
      write(ifmt,'(a,i6)')'initialize QGSJET-II ...'

      debug=0
      if(ish.ge.3)debug=ish-2
      moniou=ifch

c model parameter setting
      call qgset
c common initialization procedure
      DATDIR="qgsjetII"
      call qgaini(DATDIR)
      BMAXNEX=dble(bmaxim)
      BMINNEX=dble(bminim)
      egymin=0.1
      egymax=egymax
      irescl=0
      

      call utprix('iniqgsII',ish,ishini,6)
      end

c-----------------------------------------------------------------------
      subroutine IniEvtQGSII
c-----------------------------------------------------------------------
c Initialization for each type of event (for given proj, targ and egy)
c-----------------------------------------------------------------------
      parameter(iapmax=208)
      include 'epos.inc'
      common/geom/rmproj,rmtarg,bmax,bkmx
c QGSJET-II Common
      double precision bqgs,bmaxqgs,bmaxnex,bminnex,xa,xb,e0
      common /qgsIInex1/xa(iapmax,3),xb(iapmax,3)
     *,bqgs,bmaxqgs,bmaxnex,bminnex


      if(matarg.gt.iapmax.or.maproj.gt.iapmax)
     &  call utstop('Nucleus too big for QGSJET-II (Mmax=208) !&',
     +sizeof('Nucleus too big for QGSJET-II (Mmax=208) !&'))
      call iclass(idproj,iclpro)
      call iclass(idtarg,icltar)
      icp=idtrafo('nxs','qgs',idproj)
      if(icp.eq.0)icp=1-2*int(rangen()+0.5)       !pi0=pi+ or p-
      if(abs(icp).gt.5)
     &  call utstop('Projectile not allowed in QGSJET-II !&',
     +sizeof('Projectile not allowed in QGSJET-II !&'))
      e0=dble(elab)
      call qgini(e0,icp,maproj,matarg)
      call qgini(e0,icp,maproj,matarg)        !again to set bm properly
      bmax=BMAXQGS
      qgsIIincs=qgsIIcrse(ekin,maproj,matarg,idtarg)
      if(engy.lt.egymin)qgsIIincs=0.          !below egymin, no interaction
      call xsigma                             !change bm in qgfz
      bkmx=sqrt(sigine/10./pi)        !10= fm^2 -> mb
      if(ish.ge.2)write(ifch,*)
     &  'QGSJET-II used with (E,proj,maproj,matarg,bmax)',e0,icp
     &  ,maproj,matarg,bmax

      return
      end

c-----------------------------------------------------------------------
      subroutine emsqgsII(iret)
c-----------------------------------------------------------------------
c  call qgsjet-II to simulate interaction
c-----------------------------------------------------------------------
      parameter(iapmax=208,nptmax=95000)
      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      double precision bqgs,bmaxqgs,bmaxnex,bminnex,xa,xb,esp
      common /qgsIInex1/xa(iapmax,3),xb(iapmax,3)
     *,bqgs,bmaxqgs,bmaxnex,bminnex
      common/nucl3/phi,bimp
      common/col3/ncol,kolpt
      common /qgarr12/ nsp
      common /qgarr14/ esp(4,nptmax),ich(nptmax)
c nsf - number of secondary fragments;
c iaf(i) - mass of the i-th fragment
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr55/ nwt,nwp

      ncol=0
      iret=0
      b1=bminim
      b2=min(bmax,bmaxim)
      a=pi*(b2**2-b1**2)

      if(a.gt.0..and.rangen().gt.qgsIIincs/10./a)goto 1001   !no interaction
      if(ish.ge.3)call alist('Determine QGSJET-II Production&',
     +sizeof('Determine QGSJET-II Production&'),0,0)

      nptl=0
      nsp=0
      call qgconf
      
      ncol=1
      nevt=1
      kolevt=ncol
      koievt=0
      npjevt=maproj
      ntgevt=matarg
      pmxevt=pnll
      egyevt=engy
      if(BQGS.ge.0.d0)then
        bimevt=real(BQGS)
        bimp=real(BQGS)
        phi=2.*pi*rangen()
        phievt=phi
      else
        bimevt=0.
        bimp=0.
        phievt=0.
        phi=0.
      endif
      anintine=anintine+1.

      call conre
      call conwr
      call conqgsII



c keep the projectile spectators as fragments
      if(infragm.eq.2)then

        if(NSF.gt.0)then
          do is=1,NSF           !count the number of spectators
          if(ish.ge.7)write(ifch,'(a,i5,a,i5)')
     $       ' Projecticle Fragment ',is,' Mass :',IAF(is)
            nptl=nptl+1
            istptl(nptl)=0
            if(IAF(is).eq.1)then
              id=idptl(is)
              pptl(3,nptl)=pptl(3,is)
              pptl(4,nptl)=pptl(4,is)
              pptl(5,nptl)=pptl(5,is)
            else
              if(IAF(is).eq.2)then
                id=17
              elseif(IAF(is).eq.3)then
                id=18
              elseif(IAF(is).eq.4)then
                id=19
              else
                inucl=IAF(is)
                iprot= int(dble(inucl) / 2.15d0 + 0.7d0)
                id=1000000000+iprot*10000+inucl*10 !code for nuclei
              endif
              call idmass(id,am)
              pptl(4,nptl)=dble(IAF(is))*pptl(4,is)      !Etot
              pptl(5,nptl)=am                             !mass
              pz2tmp=(pptl(4,nptl)+am)*(pptl(4,nptl)-am)
              if(pz2tmp.gt.0.d0)then
                pptl(3,nptl)=sqrt(pz2tmp)                 !Pz
              else
                write(*,*)'Warning in emsqgsII !'
                write(*,*)'energy of fragment too small :',IAF(is),am
     &                     ,pptl(4,nptl)
                pptl(3,nptl)=pptl(4,nptl)
              endif
            endif
            pptl(1,nptl)=0.d0 !P_x
            pptl(2,nptl)=0.d0 !P_y
            ityptl(nptl)=0
            iorptl(nptl)=1
            jorptl(nptl)=maproj+matarg
            ifrptl(1,nptl)=0
            ifrptl(2,nptl)=0
            xorptl(1,nptl)=0.d0
            xorptl(2,nptl)=0.d0
            xorptl(3,nptl)=0.d0
            xorptl(4,nptl)=0.d0
            tivptl(1,nptl)=0.d0
            tivptl(2,nptl)=0.d0
            idptl(nptl)=id
            if(ish.ge.5)write(ifch,'(a,i5,a,i5,a,4(e10.4,1x),f6.3)')
     $       ' Fragment from qgsjetII ',nptl,' id :',idptl(nptl)
     $  , ' momentum :',(pptl(k,nptl),k=1,5)

          enddo
        endif

c make the projectile spectators as free nucleons

      else
        ns=0
        if(NSF.gt.0)then
          do is=1,NSF           !count the number of spectators
            ns=ns+IAF(is)
          enddo
          if(infragm.eq.1)then
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
            do is=1,ns          !make the ns first projectile nucleon actives
              istptl(is)=0
            enddo
          endif
        endif
      endif

c restore target spectators
      ns=0
      if(NWT.lt.matarg)then
        ns=matarg-NWT
        do is=maproj+1,maproj+ns  !make the ns first target nucleon actives
          istptl(is)=0
        enddo
      endif


      do is=1,nsp

c ich is the type of secondary hadron, esp - its transverse momentum,
c and its energy
c the following notations for the particles types are used: 0 - pi0, 1 -
c pi+,
c -1 - pi-, 2 - p, -2 - p, 3 - n, -3 - n, 4 - k+, -4 - k-, 5 - k0s, -5 -
c k0l
          ic=ich(is)
          if(ish.ge.7)write(ifch,'(a,i5,a,i5,2a,4(e10.4,1x))')
     $       ' qgsjet particle ',is,' id :',ic,' before conversion'
     $     , ' momentum :',(esp(k,is),k=1,4)

            nptl=nptl+1
            if(nptl.gt.mxptl)call utstop('qgsjet: mxptl too small&',
     +sizeof('qgsjet: mxptl too small&'))
            id=idtrafo('qgs','nxs',ic)
            if(ish.ge.7)write(ifch,'(a,i5,a,i5,a)')
     $       ' epos particle ',nptl,' id :',id,' after conversion'
            call idmass(id,am)
            

            pptl(1,nptl)=real(esp(3,is)) !P_x
            pptl(2,nptl)=real(esp(4,is)) !P_y
            pptl(3,nptl)=real(esp(2,is)) !P_z
            pptl(4,nptl)=real(esp(1,is)) !E
            pptl(5,nptl)=am              !mass
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

c boost in CMS frame
            call utlob5(yhaha, pptl(1,nptl), pptl(2,nptl)
     .        , pptl(3,nptl), pptl(4,nptl), pptl(5,nptl))
            
            if(ish.ge.5)write(ifch,'(a,i5,a,i5,a,4(e10.4,1x),f6.3)')
     $       ' particle from qgsjet ',nptl,' id :',idptl(nptl)
     $  , ' momentum :',(pptl(k,nptl),k=1,5)


      enddo

1000  return

1001  iret=-1 
      goto 1000 

      end

c-----------------------------------------------------------------------
      subroutine conqgsII
c-----------------------------------------------------------------------
c  determines interaction configuration
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      common/nucl3/phi,bimp
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)
      common/cncl2/diffbpr(mamx,mamx),diffbtg(mamx,mamx)
      common/cncl3/iactpr(mamx),iacttg(mamx)
      common/cfacmss/facmss

      call utpri('cqgsII ',ish,ishini,4)  

      iret=0

c     initialisations
c     ---------------

      vel=tanh(ypjtl-yhaha)+tanh(yhaha)
 
c     determine phi, bimp, coll, iproj, itarg, x/y/zproj, x/y/ztarg
c     ---------------------------------------------------------------


      if(maproj.eq.1.and.matarg.eq.1)then
           
      koll=1
      do n=1,4
        coord(n,1)=0.
      enddo
      bk(1)=bimp
      iproj(1)=1
      itarg(1)=1
      phi=phievt
      xproj(1)=bimp*cos(phi)
      yproj(1)=bimp*sin(phi)
      zproj(1)=0
      xtarg(1)=0
      ytarg(1)=0
      ztarg(1)=0
      lproj(1)=1
      ltarg(1)=1
      lproj3(1)=1
      ltarg3(1)=1
      kproj(1,1)=1
      ktarg(1,1)=1

           else
             
      bx=0
      by=0
      if(maproj.gt.0)then
        phi=phievt
        bimp=bimevt
        bx=cos(phi)*bimp
        by=sin(phi)*bimp
      endif
      if(maproj.eq.0)goto 1000
      koll=0
      do i=1,maproj
      lproj(i)=0
      enddo
      do j=1,matarg
      ltarg(j)=0
      enddo
      do 12 i=1,maproj
      do 11 j=1,matarg
      bij=sqrt((xproj(i)+bx-xtarg(j))**2+(yproj(i)+by-ytarg(j))**2)
      if(ish.ge.7)write(ifch,*)'i_p:',i,' i_t:',j,' b_ij:',bij
      if(bij.gt.bkmx)goto 11                

      if(koll.ge.kollmx)goto 1000
      koll=koll+1

      bk(koll)=bij
      bkx(koll)=xproj(i)+bx-xtarg(j)
      bky(koll)=yproj(i)+by-ytarg(j)
      iproj(koll)=i
      itarg(koll)=j
      lproj(i)=lproj(i)+1
      ltarg(j)=ltarg(j)+1
      kproj(i,lproj(i))=koll
      ktarg(j,ltarg(j))=koll


11    continue
12    continue


      do k=1,koll
        do n=1,4
          coord(n,k)=0.
        enddo
      enddo

          endif

      if(ish.ge.3)write(ifch,*)'koll=',koll
      if(koll.eq.0)goto 1000


c     determine coord
c     --------------- 
      do kl=1,koll
      i=iproj(kl)
      j=itarg(kl)
      dist=ztarg(j)-zproj(i)
      coord(1,kl)=(xproj(i)+xtarg(j))*0.5
      coord(2,kl)=(yproj(i)+ytarg(j))*0.5
      coord(3,kl)=(zproj(i)+ztarg(j))*0.5
      coord(4,kl)=dist/vel
      enddo
      
c     determine number of collisions according glauber model
c     ------------
      nglevt=koll
       
c     exit
c     ----
1000  continue
      if(ish.ge.5)then
      write(ifch,*)'ia,x/y/zproj:'
      do mm=1,maproj
      write(ifch,*)mm,iactpr(mm),xproj(mm),yproj(mm),zproj(mm)
      enddo
      write(ifch,*)'ia,x/y/ztarg:'
      do mm=1,matarg
      write(ifch,*)mm,iacttg(mm),xtarg(mm),ytarg(mm),ztarg(mm)
      enddo
      write(ifch,*)'iret',iret
      endif
      call utprix('cqgsII ',ish,ishini,4)
      return

      end


c------------------------------------------------------------------------------
      function qgsIIcrse(egy,mapro,matar,id)
c------------------------------------------------------------------------------
c inelastic cross section of qgsjet-II 
c (id=0 corresponds to air)
c egy - kinetic energy
c maproj - projec mass number     (1<maproj<64)
c matarg - target mass number     (1<matarg<64)
c------------------------------------------------------------------------------
      include 'epos.inc'
      double precision egyl,qgsect

      qgsIIcrse=0.
      call iclass(idproj,icpro)
      call idmass(1120,amt1)
      call idmass(1220,amt2)
      amtar=0.5*(amt1+amt2)
      if(matar.eq.1)amtar=amt1
      if(mapro.eq.1)then
        call idmass(idproj,ampro)
      else
        ampro=mapro*amtar
      endif
      egyl=dble(egy+ampro)

      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          qgsIIcrse=qgsIIcrse+airwnxs(k)*qgsect(egyl,icpro,mapro,mt)
        enddo
      else
        qgsIIcrse=qgsect(egyl,icpro,mapro,matar)
      endif

      return
      end

c--------------------------------------------------------------------
      double precision function qgran(b10)
c--------------------------------------------------------------------
c Random number generator
c--------------------------------------------------------------------
      include 'epos.inc'
      double precision b10,drangen
      qgran=drangen(b10)
      if(irandm.eq.1)write(ifch,*)'qgran()= ',qgran

      return
      end

