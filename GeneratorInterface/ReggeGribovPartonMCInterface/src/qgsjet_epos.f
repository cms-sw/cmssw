c 25.04.2003 Link routines between QGSJet and epos.
c author T. Pierog


c-----------------------------------------------------------------------
      subroutine IniQGSJet
c-----------------------------------------------------------------------
c Primary initialization for QGSJet
c-----------------------------------------------------------------------
      include 'epos.inc'
      COMMON /Q_DEBUG/  DEBUG
      COMMON /Q_AREA43/ MONIOU
      integer debug
      double precision BQGS,BMAXQGS,BMAXNEX,BMINNEX,XA(210,3),XB(210,3)
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX  !ctp

      call utpri('iniqgs',ish,ishini,6)
      write(ifmt,'(a,i6)')'initialize QGSJet ...'

      debug=0
      if(ish.ge.3)debug=ish-2
      moniou=ifch

c common model parameters setting
      call psaset
c particular model parameters setting
      call xxaset
c common initialization procedure
      call qgspsaini
      BMAXNEX=dble(bmaxim)
      BMINNEX=dble(bminim)
      egymin=0.1
      egymax=egymax
      irescl=0
      

      call utprix('iniqgs',ish,ishini,6)
      end

c-----------------------------------------------------------------------
      subroutine IniEvtQGS
c-----------------------------------------------------------------------
c Initialization for each type of event (for given proj, targ and egy)
c-----------------------------------------------------------------------
      include 'epos.inc'
      common/geom/rmproj,rmtarg,bmax,bkmx
c QGSJet Common
      double precision XA(210,3),XB(210,3),BQGS,BMAXQGS,BMAXNEX,BMINNEX
     &  ,e0
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX

      if(matarg.gt.210.or.maproj.gt.210)
     &  call utstop('Nucleus too big for QGSJet (Mmax=210) !&')
      call iclass(idproj,iclpro)
      call iclass(idtarg,icltar)
      e0=dble(elab)
      icp=idtrafo('nxs','qgs',idproj)
      if(icp.eq.0)icp=1-2*int(rangen()+0.5)       !pi0=pi+ or p-
      call xxaini(e0,icp,maproj,matarg)
      bmax=BMAXQGS
      qgsincs=fqgscrse(ekin,maproj,matarg)
      if(engy.lt.egymin)qgsincs=0.          !below egymin, no interaction
      call xsigma
      bkmx=sqrt(sigine/10./pi)        !10= fm^2 -> mb
      if(ish.ge.2)write(ifch,*)
     &  'QGSJet used with (E,proj,maproj,matarg,bmax)',e0,icp,maproj
     &  ,matarg,bmax

      return
      end

c-----------------------------------------------------------------------
      subroutine emsqgs(iret)
c-----------------------------------------------------------------------
c  call qgsjet to simulate interaction
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      double precision XA(210,3),XB(210,3),BQGS,BMAXQGS,BMAXNEX,BMINNEX
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX
      common/nucl3/phi,bimp
      common /q_area12/ nsp
      common /q_area14/ esp(4,95000),ich(95000)
      double precision esp
c NSF - number of secondary fragments;
c IAF(i) - mass of the i-th fragment
      COMMON /Q_AREA13/ NSF,IAF(210)
      COMMON /Q_AREA99/ NWT

      iret=0
      b1=bminim
      b2=min(bmax,bmaxim)
      a=pi*(b2**2-b1**2)

      if(a.gt.0..and.rangen().gt.qgsincs/10./a)goto 1001   !no interaction
      if(ish.ge.3)call alist('Determine QGSJet Production&',0,0)

      nptl=0
      nsp=0
      call psconf

      nevt=1
      kolevt=-1
      koievt=-1
      kohevt=-1
      npjevt=maproj
      ntgevt=matarg
      pmxevt=pnll
      egyevt=engy
      if(BQGS.ge.0.d0)then
        bimevt=real(BQGS)
        bimp=real(BQGS)
        phievt=2.*pi*rangen()
      else
        bimevt=0.
        bimp=0.
        phievt=0.
      endif
      anintine=anintine+1.

      call conre
      call conwr
      
c keep the projectile spectators as fragments
      npns=0
      npps=0
      ns=0
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
              if(id.eq.1120)npps=npps+1
              if(id.eq.1220)npns=npns+1
            else
              if(IAF(is).eq.2)then
                id=17
                npps=npps+1
                npns=npns+1
              elseif(IAF(is).eq.3)then
                id=18
                npps=npps+1
                npns=npns+2
              elseif(IAF(is).eq.4)then
                id=19
                npps=npps+2
                npns=npns+2
              else
                inucl=IAF(is)
                iprot= int(dble(inucl) / 2.15d0 + 0.7d0)
                id=1000000000+iprot*10000+inucl*10 !code for nuclei
                npps=npps+iprot
                npns=npns+inucl-iprot
              endif
              call idmass(id,am)
              pptl(4,nptl)=dble(IAF(is))*pptl(4,is)      !Etot
              pptl(5,nptl)=am                             !mass
              pz2tmp=(pptl(4,nptl)+am)*(pptl(4,nptl)-am)
              if(pz2tmp.gt.0.d0)then
                pptl(3,nptl)=sqrt(pz2tmp)                 !Pz
              else
                write(*,*)'Warning in emsqgs !'
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
     $       ' Fragment from qgsjet ',nptl,' id :',idptl(nptl)
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
            npps=npps+iprot
            npns=npns+inucl-iprot
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
            do is=maproj-ns+1,maproj          !make the ns last projectile nucleon final
              if(idptl(is).eq.1120)npps=npps+1
              if(idptl(is).eq.1220)npns=npns+1
            enddo
          endif
        endif
      endif

c number of participants
      if(laproj.gt.1)then
        npjevt=maproj-npps-npns
        npppar=max(0,laproj-npps)
        npnpar=npjevt-npppar
c set participant projectile as non spectators
        do i=1,maproj
          if(idptl(i).eq.1120)then
            if(npppar.gt.0)then
              npppar=npppar-1
            else                !restore spectators
              iorptl(i)=0
              if(infragm.eq.0)istptl(i)=0
            endif
          endif
          if(idptl(i).eq.1220)then
            if(npnpar.gt.0)then
              npnpar=npnpar-1
            else                !restore spectators
              iorptl(i)=0
              if(infragm.eq.0)istptl(i)=0
            endif
          endif
        enddo
      endif

c restore target spectators
      ns=0
      if(NWT.lt.matarg)then
c number of participants
        ntgevt=NWT
        do is=ntgevt+1,matarg         !make the last target nucleon final
          iorptl(maproj+is)=0
          istptl(maproj+is)=0
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
            if(nptl.gt.mxptl)call utstop('qgsjet: mxptl too small&')
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

            if(noebin.lt.0)pptl(3,nptl)=-pptl(3,nptl) !exchange projectile <-> target side in case of fake DIS
            
            if(ish.ge.5)write(ifch,'(a,i5,a,i5,a,4(e10.4,1x),f6.3)')
     $       ' particle from qgsjet ',nptl,' id :',idptl(nptl)
     $  , ' momentum :',(pptl(k,nptl),k=1,5)


      enddo

1000  return

1001  iret=-1 
      goto 1000 

      end


c------------------------------------------------------------------------------
      function fqgscrse(ek,mapr,matg)
c------------------------------------------------------------------------------
c hadron-nucleus (hadron-proton) and nucl-nucl particle production cross section
c with qgsjet.
c ek - kinetic lab energy
c maproj - projec mass number     (1<maproj<210)
c matarg - target mass number     (1<matarg<210)
c------------------------------------------------------------------------------
      dimension wk(3),wa(3),wb(3)
      include 'epos.inc'
      double precision gsect,qgsasect
      COMMON /Q_XSECT/  GSECT(10,5,4)
      COMMON /Q_AREA48/ QGSASECT(10,6,4)

      fqgscrse=0.
      call idmass(1120,amt1)
      call idmass(1220,amt2)
      amtar=0.5*(amt1+amt2)
      if(matg.eq.1)amtar=amt1
      if(mapr.eq.1)then
        call idmass(idproj,ampro)
      else
        ampro=mapr*amtar
      endif
      egy=ek+ampro
      ye=max(1.,log10(egy))
      je=min(8,int(ye))

      wk(2)=ye-je
      wk(3)=wk(2)*(wk(2)-1.)*.5
      wk(1)=1.-wk(2)+wk(3)
      wk(2)=wk(2)-2.*wk(3)

      ya=real(matg)
      ya=log(ya)/1.38629+1.
      ja=min(int(ya),2)
      wa(2)=ya-ja
      wa(3)=wa(2)*(wa(2)-1.)*.5
      wa(1)=1.-wa(2)+wa(3)
      wa(2)=wa(2)-2.*wa(3)

      if(mapr.eq.1)then

        do i=1,3
          do m=1,3
         fqgscrse=fqgscrse+real(gsect(je+i-1,iclpro,ja+m-1))*wk(i)*wa(m)
          enddo
        enddo

      else

        yb=mapr
        yb=log(yb/2.)/.69315+1.
        jb=min(int(yb),4)
        wb(2)=yb-jb
        wb(3)=wb(2)*(wb(2)-1.)*.5
        wb(1)=1.-wb(2)+wb(3)
        wb(2)=wb(2)-2.*wb(3)

        do i=1,3
          do m=1,3
            do n=1,3
              fqgscrse=fqgscrse+real(qgsasect(je+i-1,jb+n-1,ja+m-1)
     &                     *wk(i)*wa(m)*wb(n))
            enddo
          enddo
        enddo

      endif

      fqgscrse=exp(fqgscrse)
      return
      end

c------------------------------------------------------------------------------
      function qgscrse(egy,mapro,matar,id)
c------------------------------------------------------------------------------
c inelastic cross section of qgsjet 
c (id=0 corresponds to air)
c egy - kintetic energy
c maproj - projec mass number     (1<maproj<210)
c matarg - target mass number     (1<matarg<210)
c------------------------------------------------------------------------------
      include 'epos.inc'

      qgscrse=0.
      if(id.eq.0)then
        do k=1,3
          mt=int(airanxs(k))
          qgscrse=qgscrse+airwnxs(k)*fqgscrse(egy,mapro,mt)
        enddo
      else
        qgscrse=fqgscrse(egy,mapro,matar)
      endif

      return
      end

c--------------------------------------------------------------------
      double precision function psran(b10)


c--------------------------------------------------------------------
c  Random number generator
c--------------------------------------------------------------------
      double precision b10,drangen
      include 'epos.inc'
      psran=drangen(b10)
      if(irandm.eq.1)write(ifch,*)'psran()= ',psran

      return
      end
 


