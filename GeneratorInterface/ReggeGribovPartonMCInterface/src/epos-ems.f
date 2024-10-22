c-----------------------------------------------------------------------
      subroutine emsaa(iret)
c-----------------------------------------------------------------------
c  energy-momentum sharing
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      common/cwzero/wzero,wzerox
      double precision omega,omlog,oma,omb,wab,wba,wmatrix,wzero,nbar
     *,wzerox,rrr,eps,xprem,xmrem,om1intgck
      parameter(eps=1.d-30)
      common/col3/ncol,kolpt
c      logical modu
      common/cems5/plc,s
      double precision s,px,py,pomass,plc!,PhiExpo
      common/ems6/ivp0,iap0,idp0,isp0,ivt0,iat0,idt0,ist0
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      common/nucl3/phi,bimp
      common/epoquasi/iquasi
      logical vpom,difint
      dimension ishuff(2*mamx,2),icp(2),ict(2),jcp(nflav,2),jct(nflav,2)
     &          ,nishuff(2)
      call utpri('emsaa ',ish,ishini,4)

      irea=iret

      do j=1,2
        do i=1,nflav
          jcp(i,j)=0
          jct(i,j)=0
        enddo
      enddo

      iret=0
      iret2=0

c     initialize
c     ----------

      call emsipt   !initialize projectile and target
      call emsigr   !initialize grid



c Metropolis

      if(iokoll.ne.1)then

        nSprmx=0
        do k=1,koll
          nSprmx=nSprmx+nprmx(k)
        enddo

        omlog=0
        nemsi=nemsi+1
        if(nemsi.le.4.and.iemsi1.eq.1)call xEmsI1(1,0,omlog)
        if(ish.ge.6)write (ifch,*)'after xEmsI1'
        if(nemsi.le.4.and.iemsi2.eq.1)call xEmsI2(1,0)
        if(ish.ge.6)write (ifch,*)'after xEmsI2'
        if(ish.ge.6)call XPrint('Before Markov:&')


c     Markov
c     ------

      if(ish.ge.4)write(ifch,*)'Markov Process'
      kint=int(max(15.,2.*engy**0.2))
      if(koll.gt.50)kint=3*kint/int(log(float(koll)))
      kmcmx=nSprmx*kint        !50*kint  !100*kint


      do kmc=1,kmcmx               !-----> start Metropolis

       knprmx=0
       rrr=dble(rangen())
       do ik=1,koll
         knprmx=knprmx+nprmx(ik)
         if(rrr.le.dble(knprmx)/dble(nSprmx))then ! k-th pair
           k=ik
           goto 10
         endif
       enddo
 10    continue

       ip=iproj(k)
       it=itarg(k)
       n=1+int(rangen()*float(nprmx(k)))  ! n-th spot for k-th pair
       nbar=dble(npr(0,k))
       if(idpr(n,k).eq.0)nbar=nbar-1d0

       xprem=1.d0!xpp(ip)+xppr(n,k)        !consistently, it should be 1.
       xmrem=1.d0!xmt(it)+xmpr(n,k)
       wzerox=(nbar+1d0)
       wzero=wzerox    / ( wzerox
     &                    +om1intgck(k,xprem,xmrem)*gammaV(k) )

       if(ish.ge.8)write(ifch,*)'wzero',k,n,wzero,wzerox,gammaV(k)
     &                          ,om1intgck(k,xprem,xmrem)
       if(ish.ge.1.and.100000*(kmc/100000).eq.kmc)
     & write(ifmt,*)'kmc',kmc,kmcmx

       call StoCon(1,k,n)
       call RemPom(k,n)
       call ProPo(k,n)
       call ProXY(k,n)

       call StoCon(2,k,n)

       if(idpr(n,k).eq.0.and.idx0.eq.0)then
         accept=accept+1.
       else

         omb=omega(n,k)
         if(omb.le.0.d0)then
           reject=reject+1.
           call RemPom(k,n)
           call StoCon(-1,k,n)
         else

           wab=wmatrix(k,n)
           if(ish.ge.8)write(ifch,*)'omb',omb,wab,k,n
           if(wab.le.0.d0)then
             write (ifmt,*)'wab,kmc',wab,omb,kmc,k,n,xpr(n,k),ypr(n,k)
     &  ,xppr(n,k),xmpr(n,k),xpp(ip),xmt(it),ip,it,idpr(n,k)
             write(ifmt,'(a,i12,d25.15)')'ems,seedf',nrevt+1,seedc
             iret=1
             goto 1000
           endif
           call RemPom(k,n)
           call StoCon(-1,k,n)
           oma=omega(n,k)
           wba=wmatrix(k,n)
           if(oma.ge.0.d0.and.oma.le.eps*omb*wba/wab)then
             accept=accept+1.
             call RemPom(k,n)
             call StoCon(-2,k,n)
             omlog=omlog+dlog(omb)
             goto 500
           elseif(oma.le.1.d-300.or.oma.ne.oma.or.omb.ne.omb)then
             write (ifmt,*)'oma,kmc',oma,omb,kmc,k,n,xpr(n,k),ypr(n,k)
     &  ,xppr(n,k),xmpr(n,k),idpr(n,k),npr(1,k),xpp(ip),xmt(it),ip,it
             write(ifmt,'(a,i12,d25.15)')'ems,seedf',nrevt+1,seedc
             iret=1
             goto 1000
           endif

           z=sngl(omb/oma*wba/wab)
           if(ish.ge.8)write(ifch,*)'z,oma',z,oma,wba,k,n
           if(rangen().gt.z)then
             reject=reject+1.
           else
             accept=accept+1.
             call RemPom(k,n)
             call StoCon(-2,k,n)
             omlog=omlog-dlog(oma)+dlog(omb)
           endif

 500       continue

         endif

         endif

       if(nemsi.le.4)then
         kplot=int(float(kmc)/float(kmcmx)*100.)
         if(iemsi1.eq.1)call xEmsI1(1,kplot,omlog)
         if(iemsi2.eq.1)call xEmsI2(1,kplot)
       endif

      enddo                     !-----> end Metropolis


      else

        n=1
        
        do k=1,koll
          
          call ProPo(k,n)
          call ProXY(k,n)
          
        enddo

      endif

c --- Plot Pomeron b-distributions ---

      if(ish.ge.6)call XPrint('After Markov :&')

      if(iemsb.eq.1)then ! plot
       do k=1,koll
        call xEmsB(1,1,k)
        if(nprt(k).gt.0)call xEmsB(1,2,k)
       enddo
      endif

      if(iemsbg.eq.1)then ! plot
        call xEmsBg(3,0,0)
        do k=1,koll
          call xEmsBg(1,0,k)
          if(nprt(k).gt.0)then
            call xEmsBg(1,-1,k)
            do n=1,nprmx(k)
              if(idpr(n,k).ne.0)call xEmsBg(1,idpr(n,k),k)
            enddo
          endif
        enddo
      endif

c --- Plot distr of pomeron number ---


      if(iemspm.eq.1)then
       do k=1,koll
           call xEmsPm(1,k,nprt(k),nprmx(k))
       enddo
      endif


c --- Count all interactions ---

      ncol=0
      ncolh=0
      do k=1,koll
        if(nprt(k).gt.0)then
          ncol=ncol+1
          if(isplit.eq.1)then
            do n=1,nprmx(k)
              if(xpr(n,k).gt.xzcutpar(k))itpr(k)=1  !for nuclear splitting
            enddo
          endif
          ip=iproj(k)
          it=itarg(k)
          kolp(ip)=kolp(ip)+nprt(k) !number of cut Pomerons
          kolt(it)=kolt(it)+nprt(k) !on remnants
        endif
      enddo

c --- Calculate Z (written to zzremn)


      do ip=1,maproj
       call CalcZZ(1,ip)
      enddo
      do it=1,matarg
       call CalcZZ(-1,it)
      enddo

c -- Split Enhanced Pomerons and fix their nature ---

      if(isplit.eq.1.and.ncol.gt.0)then

        if (iLHC.eq.1)then  !make random selection to avoid assymetry
          
          nishuff(1)=0
          nishuff(2)=0
          do ip=1,maproj
            nishuff(1)=nishuff(1)+1
            ishuff(nishuff(1),1)=ip 
          enddo
          do it=1,matarg
            nishuff(2)=nishuff(2)+1
            ishuff(nishuff(2),2)=it 
          enddo

          do while(nishuff(1)+nishuff(2).gt.0)

c random selection
            if(nishuff(1).gt.0.and.nishuff(2).gt.0)then
              ir=1+int(rangen()+0.5)
            elseif(nishuff(1).gt.0)then
              ir=1
            else
              ir=2
            endif

            indx=1+int(rangen()*float(nishuff(ir)))
            if(ir.eq.1)then
              ip=ishuff(indx,ir)
              if(lproj3(ip).ne.0.and.kolp(ip).eq.0)call ProNucSpl( 1,ip)
            else
              it=ishuff(indx,ir)
              if(ltarg3(it).ne.0.and.kolt(it).eq.0)call ProNucSpl(-1,it)
            endif
            ishuff(indx,ir)=ishuff(nishuff(ir),ir)
            nishuff(ir)=nishuff(ir)-1

          enddo

        else

          do ip=1,maproj
            if(lproj3(ip).ne.0.and.kolp(ip).eq.0)call ProNucSpl( 1,ip)
          enddo
          do it=1,matarg
            if(ltarg3(it).ne.0.and.kolt(it).eq.0)call ProNucSpl(-1,it)
          enddo

        endif

        if(ish.ge.6)call XPrint('After ProNucSpl:&')

      endif

c -- Fix Pomeron type ---

      do k=1,koll
        itpr(k)=0
        do n=1,nprmx(k)
          if(idfpr(n,k).eq.0)call ProPoTy(k,n)
        enddo
      enddo



c --- Fix Remnant Excitation

      do ip=1,maproj
       if(lproj(ip).ne.0)then
         call ProReEx( 1,ip)
         if(iremn.ge.2)call UpdateFlav(ip,jcp,0) !reset jcpref to 0
       endif
      enddo
      do it=1,matarg
       if(ltarg(it).ne.0)then
         call ProReEx(-1,it)
         if(iremn.ge.2)call UpdateFlav(it,jct,0) !reset jctref to 0
       endif
      enddo


c --- LHC tune : remove unnecessary diffractive Pomerons
      if(iLHC.eq.1)then
        do k=1,koll
          ip=iproj(k)
          it=itarg(k)
c remove temporary diffractive Pomeron if at least on remnant excited
c          ymean=0.
          do n=1,nprmx(k)
            if(idpr(n,k).eq.-1)then
              idpr(n,k)=1
              if((iep(ip).gt.0.or.iet(it).gt.0)
     &             .and.xpr(n,k).le.xzcutpar(k))call VirPom(k,n,0)
c              if(iep(ip).gt.0.or.iet(it).gt.0)then
c                if(xpr(n,k).ge.xzcutpar(k))then
c                  call VirPom(k,n,0)
c                else
c                  ymean=ymean+ypr(n,k)
c                endif
c              endif
           endif
          enddo
cc put excitation on the side of the pomeron
c          if((ymean.gt.0..and.iet(it).gt.0).or.
c     &       (ymean.lt.0..and.iep(ip).gt.0))then
c            ietmp=iep(ip)
c            iep(ip)=iet(it)
c            iet(it)=ietmp
c          endif
        enddo
      endif

c --- Count real interactions ---

      ncol=0
      do k=1,koll
        if(nprt(k).gt.0)then        !inelastic
          ncol=ncol+1
          if(itpr(k).lt.0)then
            itpr(k)=-1
          else
            itpr(k)=1              !diffractive with Pomeron
          endif
        elseif(itpr(k).gt.0)then    !diffractive
          ncol=ncol+1
          call ProDiSc(k)
          itpr(k)=2
        endif
      enddo
      if(ish.ge.5)write(ifch,*)'ncol:',ncol



c --- fix all variables


      if(ish.ge.4)write(ifch,*)'fix all variables'


c ---  recalculate Zptn


c      if(irzptn.eq.1)call recalcZPtn


      typevt=0                !ela
      if(maproj+matarg.eq.2)then     !pp
        if(itpr(1).ne.0)then
          anintine=anintine+1.
          if(itpr(1).gt.0)then
            if(ionudi.eq.1
     &        .or.iep(1).ne.0.or.iet(1).ne.0.or.itpr(1).eq.1)then
              anintdiff=anintdiff+1.
              if((iep(1).eq.0.and.iet(1).eq.2).or.
     &           (iet(1).eq.0.and.iep(1).eq.2))anintsdif=anintsdif+1.
              if(iep(1).eq.0.and.iet(1).eq.2)typevt=-4    !SD tar
              if(iet(1).eq.0.and.iep(1).eq.2)typevt=4     !SD pro
              if(iep(1).eq.2.and.iet(1).eq.2)typevt=2     !DD
              if(iep(1).eq.0.and.iet(1).eq.0)typevt=3     !CD
            else
              anintine=anintine-1. !diffractive without excitation = elastic
            endif
          else
            typevt=1                                      !ND
          endif
        endif
      else
        aidif=0.
        aidifp=0.
        aidift=0.
        aiine=0.
        do k=1,koll
          ip=iproj(k)
          it=itarg(k)
          if(aidif.ge.0..and.itpr(k).gt.0)then
            aidifp=aidifp+iep(ip)+(2-itpr(k))*0.00001
            aidift=aidift+iet(it)+(2-itpr(k))*0.00001
            if(ionudi.eq.1)then !count all diff as inelastic (to compare to tabulated cs)
              aidif=aidif+1.
            endif
          elseif(itpr(k).eq.-1)then
            aiine=aiine+1.
            aidif=-ainfin
          endif
        enddo
        if(ionudi.eq.2)then
          aidif=aidif+aidifp
        else
          aidif=aidif+aidifp+aidift
        endif
        if(aidif.gt.0.)then
          anintdiff=anintdiff+1.
          anintine=anintine+1.
          if(aidifp.gt.0.5.and.aidift.le.0.5)then
            anintsdif=anintsdif+1.
            typevt=4                        !SD pro
          endif
          if(aidifp.gt.0.5.and.aidift.gt.0.5)then
            typevt=2                        !DD
          endif
          if(ionudi.ne.2)then
            if(aidifp.le.0.5.and.aidift.gt.0.5)then
              anintsdif=anintsdif+1.
              typevt=-4                      !SD tar
            elseif(typevt.le.0.5.and.aidifp.gt.0..and.aidift.gt.0.)then
              typevt=3                      !CD
            endif
          endif
        elseif(aiine.gt.0.)then
          anintine=anintine+1.
          typevt=1                          !ND
        endif
      endif

      if(ish.ge.6)call XPrint('After fixing:&')


c --- Plot MC pomeron number ---

      if(nemsi.le.4.and.irea.ge.0)then
       if(iemsi1.eq.1)call xEmsI1(1,100,omlog)
       if(iemsi2.eq.1)call xEmsI2(1,100)
       if(iemsi1.eq.1.and.ncol.gt.0)call xEmsI1(2,0,omlog)
       if(iemsi2.eq.1.and.ncol.gt.0)call xEmsI2(2,0)
       if((iemsi1.eq.1.or.iemsi2.eq.1).and.ncol.eq.0)nemsi=nemsi-1
      endif

      if(iemsb.eq.1)then        ! plot
        do k=1,koll
          if(itpr(k).eq.0)call xEmsB(1,3,k) !nothing
          if(itpr(k).eq.-1)call xEmsB(1,4,k) !cut
          if(itpr(k).gt.0)call xEmsB(1,5,k) !diffr
          if(abs(itpr(k)).eq.1)call xEmsB(1,6,k) !cut+diffr cut
        enddo
      endif


c check for diffractive interaction without excitation
      difint=.true.
      ieptot=0
      if(maproj+matarg.eq.2)ieptot=1     !not used for pp
      do k=1,koll
        if(itpr(k).eq.2)then
          ip=iproj(k)
          it=itarg(k)
          ieptot=ieptot+iep(ip)
          if(ionudi.ne.2)ieptot=ieptot+iet(it)
c for CR, ionudi=1, count diffraction without excitation as inelastic (part of the xs)
          if(ionudi.ne.1.and.iep(ip).eq.0.and.iet(it).eq.0)then
            ncol=ncol-1
            itpr(k)=0
            kolp(ip)=kolp(ip)-1
            kolt(it)=kolt(it)-1
          endif
        else
          if(iLHC.eq.1.and.abs(itpr(k)).eq.1)then
            difint=.false.
          elseif(iLHC.eq.0)then   !bug in CR version for ionudi=2 (difint=F always !)
            difint=.false.
          endif
        endif
      enddo
      if(difint.and.ionudi.eq.2.and.ieptot.eq.0)then
        ncol=0                  !for ionudi=2
        iret=0
        goto 1000               !no projectile excitation = elastic
      endif

      iquasi=0
      if(ncol.eq.0)goto 998
      if(difint.and.ieptot-1.le.0)then
        iquasi=1
        if(ish.ge.2)write(ifch,*)'EPOS Quasi-elastic event'
        goto 998
      endif

c --- Treat Pomerons ---------------------------------------


c --- Check minimum mass ---

      do k=1,koll
        ip=iproj(k)
        it=itarg(k)
      do n=1,nprmx(k)
        if(xpr(n,k).lt.(cumpom/engy)**2)then
          nnb=nbkpr(n,k)
          nnv=nvpr(n,k)
          if(nnv.ne.0)then
            nbkpr(nnv,k)=0                  !if bckp Pomeron
          endif
          if(nnb.ne.0)then
            ivi=1
            call VirPom(k,nnb,ivi)            !if hard backup exist
            nbkpr(n,k)=0                    !remove it
          endif
          ivi=2
          call VirPom(k,n,ivi)
        elseif(itpr(k).eq.1.and.abs(idfpr(n,k)).eq.1)then
c diffractive cut Pomeron should not change remnant excitation
          idfs=sign(1,idfpr(n,k))
          if(iep(ip).eq.0.and.iet(it).eq.0)then
            idfpr(n,k)=idfs*4       !not linked to both proj and targ
          elseif(iep(ip).eq.0)then
            idfpr(n,k)=idfs*3       !linked to targ
            iet(it)=1               !target excitation is inelastic type
          elseif(iet(it).eq.0)then
            idfpr(n,k)=idfs*2       !linked to proj
            iep(ip)=1               !projectile excitation is inelastic type
          endif
        endif
      enddo
      enddo

c --- Set String End Type and Pt

      do k=1,koll
        ip=iproj(k)
        it=itarg(k)
        do n=1,nprmx(k)

          if(idpr(n,k).gt.0)then

          ntry=0
          vpom=.false.
          ivpi=ivp(ip)
          ivti=ivt(it)
          idpi=idp(ip)
          idti=idt(it)
          do i=1,2
            icp(i)=icproj(i,ip)
            ict(i)=ictarg(i,it)
          enddo
          if(iremn.ge.2)then    !save jcpref and jctref into jcp and jct
            call UpdateFlav(ip,jcp,1)
            call UpdateFlav(it,jct,2)
          endif

 100      ntry=ntry+1
          iret=0
          if(ntry.ge.200)vpom=.true.
          if(ntry.gt.1)then
       if(ish.ge.4)write(ifch,*)'Try again setting string ends for k,n'
     &                               ,k,n,ntry
            ivp(ip)=ivpi
            ivt(it)=ivti
            idp(ip)=idpi
            idt(it)=idti
            do i=1,2
              icproj(i,ip)=icp(i)
              ictarg(i,it)=ict(i)
            enddo
            if(iremn.ge.2)then       !restore jcpref and jctref from jcp and jct
              call UpdateFlav(ip,jcp,-1)
              call UpdateFlav(it,jct,-2)
            endif
            call RmPt(k,n)
          endif

          if(nvpr(n,k).eq.0)call ProSeTy(k,n)      !Not for backup Pomeron
          call ProSePt(k,n,iret)
          if(iret.eq.1)then
            if(vpom)then
              ivi=13
              call VirPom(k,n,ivi)
            else
              goto 100
            endif
          endif

c      enddo
c      enddo

c --- Check Pomeron mass

c      do k=1,koll
c      do n=1,nprmx(k)
       if(idpr(n,k).ne.0.and.ivpr(n,k).ne.0)then
        px=xxp1pr(n,k)+xxp2pr(n,k)+xxm1pr(n,k)+xxm2pr(n,k)
        py=xyp1pr(n,k)+xyp2pr(n,k)+xym1pr(n,k)+xym2pr(n,k)
        pomass=xpr(n,k)*s-px*px-py*py
        if(pomass.lt.amprmn(idhpr(n,k)))then
          nnv=nvpr(n,k)
          nnb=nbkpr(n,k)
          idfpom=iabs(idfpr(n,k))
          if(vpom)then
            ivi=3
            call VirPom(k,n,ivi)  !call RmPt(k,n)
            if(nnv.ne.0)then    !bckp Pomeron
              nbkpr(nnv,k)=0
            endif
            if(nnb.ne.0)then    !big Pomeron with bckp one
              ivpr(nnb,k)=1
              nvpr(nnb,k)=0
              idfpr(nnb,k)=idfpom
              npr(1,k)=npr(1,k)+1
              npr(3,k)=npr(3,k)-1
            endif
          else
            goto 100
          endif
        endif
       endif
c      enddo
c      enddo

c --- Define String ends for "backup" Pomerons ---

c      do k=1,koll
c      do n=1,nprmx(k)
c        if(nvpr(n,k).ne.0)call ProSeX(k,n,iret)
c        if(iret.eq.1)then
c          if(vpom)then
c            nn=nvpr(n,k)
c            ivi=7
c            call VirPom(k,n,ivi)
c            nbkpr(nn,k)=0
c          else
c            goto 100
c          endif
c        endif
        iret=0
        iret2=0
c      enddo
c      enddo

c --- Define String ends for "normal" Pomerons ---

c      do k=1,koll
c      do n=1,nprmx(k)
        if(nvpr(n,k).eq.0)call ProSeX(k,n,iret)   !Not for backup Pomeron
        if(iret.eq.1)then
          if(vpom)then
            ivi=12
            call VirPom(k,n,ivi)
          else
            goto 100
          endif
        endif
        iret=0
        iret2=0

      endif

      enddo
      enddo


c --- Write ---

 998  call emszz
      if(ncol.eq.0)then
        iret=0
        goto 1000
      endif


      do k=1,koll
       if(abs(itpr(k)).eq.1)call emswrpom(k,iproj(k),maproj+itarg(k))
      enddo


c --- Treat hard Pomeron

      ncolh=0
      do k=1,koll
        ncolhp=0
        do n=1,nprmx(k)
          if(idpr(n,k).eq.3)then
            if(ishpom.eq.1)then
              call psahot(k,n,iret)
              if(iret.eq.0)ncolhp=ncolhp+1
              if(iret.eq.1)then
                if(nbkpr(n,k).ne.0)then
                  nn=nbkpr(n,k)
                  call ProSeTy(k,nn)
                  call ProSeX(k,nn,iret2)
                  if(iret2.eq.1)then
                    ivi=15
                    call VirPom(k,nn,ivi)
                    if(ivi.lt.0)then
                      jerr(7)=jerr(7)+1
                      iret=1
                      goto 1000
                    endif
                    istptl(nppr(nn,k))=32
                    nbkpr(n,k)=0
                  else
                    ivpr(nn,k)=1
                    nvpr(nn,k)=0
                    idfpr(nn,k)=idfpr(n,k)
                    npr(1,k)=npr(1,k)+1
                    npr(3,k)=npr(3,k)-1
                    ansff=ansff+1 !counters
                    anshf=anshf-1
                  endif
                endif
                ivi=16
                call VirPom(k,n,ivi)
                if(ivi.lt.0)then
                  jerr(7)=jerr(7)+1
                  iret=1
                  goto 1000
                endif
                istptl(nppr(n,k))=32
              elseif(nbkpr(n,k).ne.0)then
                nn=nbkpr(n,k)
                ivi=17
                call VirPom(k,nn,ivi)
                if(ivi.lt.0)then
                  jerr(7)=jerr(7)+1
                  iret=1
                  goto 1000
                endif
                istptl(nppr(nn,k))=32
                nbkpr(n,k)=0
              endif
              iret=0
            else
              istptl(nppr(n,k))=32
              if(nbkpr(n,k).ne.0)then
                nn=nbkpr(n,k)
                istptl(nppr(nn,k))=32
              endif
            endif
          endif
        enddo
        if(ncolhp.gt.0)ncolh=ncolh+1     !count hard binary collisions
      enddo
      kohevt=ncolh     !update number of hard collisions

      if(iLHC.eq.0.and.iremn.ge.2)then
c --- Add valence quark to jcpref and jctref for soft string ends ---
        do ip=1,maproj
          if(iep(ip).ne.-1)then
            call UpdateFlav(ip,jcp,10)
            do nnn=1,nrflav
              jcpval(nnn,1,ip)=jcp(nnn,1)
            enddo
            do nnn=1,nrflav
              jcpval(nnn,2,ip)=jcp(nnn,2)
            enddo
          else
            icp(1)=icproj(1,ip)
            icp(2)=icproj(2,ip)
            call iddeco(icp,jcp)
            do nnn=1,nrflav
              jcpval(nnn,1,ip)=jcp(nnn,1)
            enddo
            do nnn=1,nrflav
              jcpval(nnn,2,ip)=jcp(nnn,2)
            enddo
          endif
        enddo
        do it=1,matarg
          if(iet(it).ne.-1)then
            call UpdateFlav(it,jct,20)
            do nnn=1,nrflav
              jctval(nnn,1,it)=jct(nnn,1)
            enddo
            do nnn=1,nrflav
              jctval(nnn,2,it)=jct(nnn,2)
            enddo
          else
            ict(1)=ictarg(1,it)
            ict(2)=ictarg(2,it)
            call iddeco(ict,jct)
            do nnn=1,nrflav
              jctval(nnn,1,it)=jct(nnn,1)
            enddo
            do nnn=1,nrflav
              jctval(nnn,2,it)=jct(nnn,2)
            enddo
          endif
        enddo
      endif

c --- Treat "normal" soft Pomerons ---

      do k=1,koll
        do n=1,nprmx(k)
          if(nvpr(n,k).eq.0)then
            if(isopom.eq.1)then
              call ProSeF(k,n,iret)
              if(iret.eq.1)then
                ivi=18
                call VirPom(k,n,ivi)
                if(ivi.lt.0)then
                  jerr(7)=jerr(7)+1
                  iret=1
                  goto 1000
                endif
                istptl(nppr(n,k))=32
              endif
              iret=0
            else
              istptl(nppr(n,k))=32
            endif
          endif
        enddo
      enddo



c --- Treat Remnants -----------------------------------------

c --- Fix Pion Exchange in diffractive remnants

      do ip=1,maproj
       if(iep(ip).eq.2)call ProReEx( 2,ip)
      enddo
      do it=1,matarg
       if(iet(it).eq.2)call ProReEx( -2,it)
      enddo

c --- Diffractive Pt and check Pomeron status

      iret=1
      do k=1,koll
        call ProDiPt(k,1,iret)
      enddo
      if(iret.ne.0)then
        jerr(8)=jerr(8)+1
        ivi=99
        if(ish.ge.2)then
          write(ifch,*)'All Pomeron lost, redo event !'
          write(ifmt,*)'All Pomeron lost, redo event !'
        endif
        iret=1
        goto 1000
      endif

      if(iLHC.eq.1.and.iremn.ge.2)then
c --- Add valence quark to jcpref and jctref for soft string ends ---
        do ip=1,maproj
          if(iep(ip).ne.-1)then
            call UpdateFlav(ip,jcp,10)
            do nnn=1,nrflav
              jcpval(nnn,1,ip)=jcp(nnn,1)
            enddo
            do nnn=1,nrflav
              jcpval(nnn,2,ip)=jcp(nnn,2)
            enddo
          else
            icp(1)=icproj(1,ip)
            icp(2)=icproj(2,ip)
            call iddeco(icp,jcp)
            do nnn=1,nrflav
              jcpval(nnn,1,ip)=jcp(nnn,1)
            enddo
            do nnn=1,nrflav
              jcpval(nnn,2,ip)=jcp(nnn,2)
            enddo
          endif
        enddo
        do it=1,matarg
          if(iet(it).ne.-1)then
            call UpdateFlav(it,jct,20)
            do nnn=1,nrflav
              jctval(nnn,1,it)=jct(nnn,1)
            enddo
            do nnn=1,nrflav
              jctval(nnn,2,it)=jct(nnn,2)
            enddo
          else
            ict(1)=ictarg(1,it)
            ict(2)=ictarg(2,it)
            call iddeco(ict,jct)
            do nnn=1,nrflav
              jctval(nnn,1,it)=jct(nnn,1)
            enddo
            do nnn=1,nrflav
              jctval(nnn,2,it)=jct(nnn,2)
            enddo
          endif
        enddo
      endif

      do ip=1,maproj
c Here and later "kolp(ip).ne.0" replaced by "iep(ip).ne.-1" to count
c projectile and target nucleons which are counted in paires but are not used
c in collision (no diffractive or inelastic interaction) as slow particles
c at the end. Then we can use them in ProRem to give mass to all other nucleons
c and avoid energy conservation violation that utrescl can not treat
c (and it gives a reasonnable number of grey particles even if distributions
c are not really reproduced).
c       if(kolp(ip).ne.0)call ProCop(ip,ip)
       if(iep(ip).ne.-1)call ProCop(ip,ip)
      enddo
      do it=1,matarg
       if(iet(it).ne.-1)call ProCot(it,maproj+it)
c       if(kolt(it).ne.0)call ProCot(it,maproj+it)
      enddo


c ---- Remnant Masses (ProReM)


      if(ish.ge.6)call XPrint('Before  ProReM:&')
      ntry=0
      iret=0
      call StoRe(1)             !Store Remnant configuration
 123  ntry=ntry+1
      nishuff(1)=0
      nishuff(2)=0
      do ip=1,maproj
        if(iep(ip).eq.0)then
          nishuff(1)=nishuff(1)+1
          ishuff(nishuff(1),1)=ip      !positive for non excited projectile
        elseif(iep(ip).gt.0)then
          nishuff(2)=nishuff(2)+1
          ishuff(nishuff(2),2)=ip      !positive for excited projectile
        endif
      enddo
      do it=1,matarg
        if(iet(it).eq.0)then
          nishuff(1)=nishuff(1)+1
          ishuff(nishuff(1),1)=-it !negative for non excited  target
        elseif(iet(it).gt.0)then
          nishuff(2)=nishuff(2)+1
          ishuff(nishuff(2),2)=-it !negative for excited  target
        endif
      enddo

c      do ir=1,2         !first set mass of non excited remnant
cc      do ir=2,1,-1         !first set mass of excited remnant

c      do while(nishuff(ir).gt.0)
      do while(nishuff(1)+nishuff(2).gt.0)

c random selection
        if(nishuff(1).gt.0.and.nishuff(2).gt.0)then
          ir=1+int(rangen()+0.5)
        elseif(nishuff(1).gt.0)then
          ir=1
        else
          ir=2
        endif

        indx=1+int(rangen()*float(nishuff(ir)))
        if(ishuff(indx,ir).gt.0)then
          ip=ishuff(indx,ir)
          call ProReM( 1,ip,iret)
        else
          it=-ishuff(indx,ir)
          call ProReM(-1,it,iret)
        endif
        if(ish.ge.10)call XPrint('In  ProReM:&')

        if(iret.eq.1)then
          !----------------------------------------
          !If there is a problem, try again shuffle (30 times),
          !if it doesn't work, for pp, try 10 times with the same type
          !of event and if doesn't work redo event;
          !for pA redo event ; and for AB (with A or B >10)
          !continue with some ghosts ...
          !----------------------------------------
          if(ntry.lt.30)then
            if(ish.ge.3)write(ifch,*)'shuffle, try again',ntry
            call StoRe(-1)         !Restore Remnant configuration
            iret=0
            goto 123
          elseif(ntry.gt.30.or.maproj.le.20.or.matarg.le.20)then
            if(ish.ge.2)write(ifch,*)'ProRem, redo event ! ntry=',ntry
            if(ish.ge.1)write(ifmt,*)'ProRem, redo event ! ntry=',ntry
            iret=1
            goto 1000
          else
            if(ish.ge.3)write(ifch,*)'shuffle, try again forcing ...'
            call StoRe(-1)         !Restore Remnant configuration
            iret=10
            goto 123
          endif
        endif

        ishuff(indx,ir)=ishuff(nishuff(ir),ir)
        nishuff(ir)=nishuff(ir)-1

       enddo
c      enddo

c --- Correction for Diffractive Pt (from Ralph but seems to be less good for NA49)

c      do k=1,koll
c        call ProDiPt(k,2,idum)
c      enddo


      iret=0
      if(ish.ge.6)call XPrint('After ProReM:&')


c --- Write Z into zpaptl for connected strings


      do ip=1,maproj
        if(kolp(ip).ne.0)call WriteZZ(1,ip)
      enddo
      do it=1,matarg
        if(kolt(it).ne.0)call WriteZZ(-1,it)
      enddo


c --- Write Remnants


      do ip=1,maproj
c       if(kolp(ip).ne.0)call emswrp(ip,ip)
       if(iep(ip).ne.-1)call emswrp(ip,ip)
      enddo

      do it=1,matarg
c       if(kolt(it).ne.0)call emswrt(it,maproj+it)
       if(iet(it).ne.-1)call emswrt(it,maproj+it)
      enddo


c --- Remnant Flavors (ProReF)


      do ip=1,maproj
        call ProReF(1,ip,iret)
        if(iret.ne.0)goto 1000
      enddo
      do it=1,matarg
        call ProReF(-1,it,iret)
        if(iret.ne.0)goto 1000
      enddo


c     plot
c     ----

       if(iemspx.eq.1)then
       do ko=1,koll
        if(nprt(ko).gt.0)then
         do np=1,nprmx(ko)
          if(idpr(np,ko).gt.0)then
           call xEmsPx(1,sngl(xpr(np,ko)),sngl(ypr(np,ko)),nprt(ko))
          endif
         enddo
        endif
       enddo
      endif

      if(iemspbx.eq.1)then
       do k=1,koll
        if(nprt(k).gt.0)then
         do n=1,nprmx(k)
          if(idpr(n,k).eq.3)then
            je1=min(1,nemispr(1,n,k))
            je2=min(1,nemispr(2,n,k))
            jex=1+je1+2*je2
            if(itpr(k).eq.-1)then
              call xEmsP2(1,1+idhpr(n,k),jex
     *            ,sngl(xppr(n,k))
     *            ,sngl(xmpr(n,k))
     *            ,sngl(xpprbor(n,k)),sngl(xmprbor(n,k))
     *            ,ptprboo(1,n,k),ptprboo(2,n,k)  )
            else !diffractive hard pomeron
              call xEmsP2(1,0,jex
     *            ,sngl(xppr(n,k))
     *            ,sngl(xmpr(n,k))
     *            ,sngl(xpprbor(n,k)),sngl(xmprbor(n,k))
     *            ,ptprboo(1,n,k),ptprboo(2,n,k)  )
            endif
          endif
         enddo
        endif
       enddo
      endif


      if(iemsse.eq.1)then
       do ko=1,koll
        if(nprt(ko).gt.0)then
         do np=1,nprmx(ko)
          if(idpr(np,ko).gt.0)then
           ptp1=sngl(xxp1pr(np,ko)**2+xyp1pr(np,ko)**2)
           ptp2=sngl(xxp2pr(np,ko)**2+xyp2pr(np,ko)**2)
           ptm1=sngl(xxm1pr(np,ko)**2+xym1pr(np,ko)**2)
           ptm2=sngl(xxm2pr(np,ko)**2+xym2pr(np,ko)**2)
           call xEmsSe(1,sngl(xp1pr(np,ko)),ptp1,1,1)
           call xEmsSe(1,sngl(xp2pr(np,ko)),ptp2,1,1)
           call xEmsSe(1,sngl(xm1pr(np,ko)),ptm1,-1,1)
           call xEmsSe(1,sngl(xm2pr(np,ko)),ptm2,-1,1)
           call xEmsSe(1,sngl(xp1pr(np,ko)),sngl(xm1pr(np,ko)),1,2)
           call xEmsSe(1,sngl(xm2pr(np,ko)),sngl(xp2pr(np,ko)),1,2)
          endif
         enddo
        endif
       enddo
      endif

      if(iemsdr.eq.1)then
       do i=maproj+matarg+1,nptl
        if(istptl(iorptl(i)).eq.41)then
          xpdr=(pptl(4,i)+pptl(3,i))/sngl(plc)
          xmdr=(pptl(4,i)-pptl(3,i))/sngl(plc)
          if(ityptl(i).eq.41)call xEmsDr(1,xpdr,xmdr,1)
          if(ityptl(i).eq.51)call xEmsDr(1,xpdr,xmdr,2)
          if(ityptl(i).eq.42)call xEmsDr(1,xpdr,xmdr,3)
          if(ityptl(i).eq.52)call xEmsDr(1,xpdr,xmdr,4)
        endif
       enddo
      endif

      if(iemsrx.eq.1)then
       do i=1,maproj
        if(kolp(i).gt.0)call xEmsRx(1,1,sngl(xpp(i)),sngl(xmp(i)))
       enddo
       do j=1,matarg
        if(kolt(j).gt.0)call xEmsRx(1,2,sngl(xmt(j)),sngl(xpt(j)))
       enddo
      endif

      if(ixbDens.eq.1)call xbDens(1)

c     exit
c     ----

 1000 continue
c      write(*,*)'emsaa-iret',iret
      if(ish.ge.2.and.iret.ne.0)write(ifch,*)'iret not 0 (ems)=> redo'
     &                                       ,iret,ivi
      call utprix('emsaa ',ish,ishini,4)
      return
      end


c----------------------------------------------------------------------
      subroutine StoCon(mode,k,n)
c----------------------------------------------------------------------
c store or restore configuration
c   mode = 1 (store) or -1 (restore)
c   k = collision index
c   n = pomeron index
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      ip=iproj(k)
      it=itarg(k)

      if(mode.eq.1)then

       do i=0,3
        nprx0(i)=npr(i,k)
       enddo
       nprtx0=nprt(k)
       idx0=idpr(n,k)
       xxpr0=xpr(n,k)
       yx0=ypr(n,k)
       xxppr0=xppr(n,k)
       xxmpr0=xmpr(n,k)
       nppx0=npp(ip)
       nptx0=npt(it)
       xppx0=xpp(ip)
       xppstx0=xppmx(ip)
       xmpstx0=xppmn(ip)
       xmtx0=xmt(it)
       xptstx0=xmtmx(it)
       xmtstx0=xmtmn(it)

      elseif(mode.eq.2)then

       do i=0,3
        nprx(i)=npr(i,k)
       enddo
       nprtx=nprt(k)
       idx=idpr(n,k)
       xxpr=xpr(n,k)
       yx=ypr(n,k)
       xxppr=xppr(n,k)
       xxmpr=xmpr(n,k)
       nppx=npp(ip)
       nptx=npt(it)
       xppx=xpp(ip)
       xppstx=xppmx(ip)
       xmpstx=xppmn(ip)
       xmtx=xmt(it)
       xptstx=xmtmx(it)
       xmtstx=xmtmn(it)

      elseif(mode.eq.-1)then

       do i=0,3
        npr(i,k)=nprx0(i)
       enddo
       nprt(k)=nprtx0
       idpr(n,k)=idx0
       xpr(n,k)=xxpr0
       ypr(n,k)=yx0
       xppr(n,k)=xxppr0
       xmpr(n,k)=xxmpr0
       npp(ip)=nppx0
       npt(it)=nptx0
       xpp(ip)=xppx0
       xppmx(ip)=xppstx0
       xppmn(ip)=xmpstx0
       xmt(it)=xmtx0
       xmtmx(it)=xptstx0
       xmtmn(it)=xmtstx0

      elseif(mode.eq.-2)then

       do i=0,3
        npr(i,k)=nprx(i)
       enddo
       nprt(k)=nprtx
       idpr(n,k)=idx
       xpr(n,k)=xxpr
       ypr(n,k)=yx
       xppr(n,k)=xxppr
       xmpr(n,k)=xxmpr
       npp(ip)=nppx
       npt(it)=nptx
       xpp(ip)=xppx
       xppmx(ip)=xppstx
       xppmn(ip)=xmpstx
       xmt(it)=xmtx
       xmtmx(it)=xptstx
       xmtmn(it)=xmtstx

      else
      call utstop('mode should integer from -2 to 2 (without 0)&')
      endif
      return
      end

c-------------------------------------------------------------------------
      subroutine RemPom(k,n)
c-------------------------------------------------------------------------
c remove pomeron
c-------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'

      ip=iproj(k)
      it=itarg(k)
      npr(idpr(n,k),k)=npr(idpr(n,k),k)-1  !nr of pomerons
      nprt(k)=npr(1,k)+npr(3,k)
      if(idpr(n,k).gt.0)then
       npp(ip)=npp(ip)-1                     !nr of pomerons per proj
       npt(it)=npt(it)-1                     !nr of pomerons per targ
       idpr(n,k)=0
       xpp(ip)=xpp(ip)+xppr(n,k)
       xmt(it)=xmt(it)+xmpr(n,k)
       xpr(n,k)=0.d0
       ypr(n,k)=0.d0
       xppr(n,k)=0.d0
       xmpr(n,k)=0.d0



      endif

      end

c-------------------------------------------------------------------------
      subroutine ProPo(k,n)
c-------------------------------------------------------------------------
c propose pomeron type = idpr(n,k
c-------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision wzero,wzerox
      common/cwzero/wzero,wzerox

      ip=iproj(k)
      it=itarg(k)

      idpr(n,k)=0

      if(dble(rangen()).gt.wzero)then
        idpr(n,k)=1


c nbr of pomerons per proj
       npp(ip)=npp(ip)+1
c nbr of pomerons per targ
       npt(it)=npt(it)+1

      endif

      npr(idpr(n,k),k)=npr(idpr(n,k),k)+1 !nr of pomerons
      nprt(k)=npr(1,k)+npr(3,k)


      end


c-------------------------------------------------------------------------
      subroutine ProXY(k,n)
c-------------------------------------------------------------------------
c propose pomeron x,y
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incems'
      include 'epos.incsem'
      double precision xp,xm,om1xprk,om1xmrk,anip,anit,eps
     &,xprem,xmrem,xprm,xmrm
      parameter (eps=1.d-30)


      ip=iproj(k)
      it=itarg(k)


      xpr(n,k)=0.d0
      ypr(n,k)=0.d0

      if(idpr(n,k).ne.0)then
          xprem=xpp(ip)
          xmrem=xmt(it)
c because of fom, it's not symetric any more if we choose always xp first
c and then xm ... so choose it randomly.
          if(rangen().lt.0.5)then
            xp=om1xprk(k,xprem,xminDf,1)
            xmrm=xmrem
            xprm=xminDf
            xm=om1xmrk(k,xp,xprm,xmrm,1)
          else
            xm=om1xprk(k,xmrem,xminDf,-1)
            xmrm=xminDf
            xprm=xprem
            xp=om1xmrk(k,xm,xmrm,xprm,-1)
          endif
          xpr(n,k)=xp*xm
          ypr(n,k)=0.d0
          if(xm.gt.eps.and.xp.gt.eps)then
            ypr(n,k)=0.5D0*dlog(xp/xm)
            xppr(n,k)=xp
            xmpr(n,k)=xm
          else
            if(ish.ge.1)write(ifmt,*)'Warning in ProXY ',xp,xm
            npr(idpr(n,k),k)=npr(idpr(n,k),k)-1
            idpr(n,k)=0
            npr(idpr(n,k),k)=npr(idpr(n,k),k)+1
            xpr(n,k)=0.d0
            ypr(n,k)=0.d0
            xppr(n,k)=0.d0
            xmpr(n,k)=0.d0
            nprt(k)=npr(1,k)+npr(3,k)
            npp(ip)=npp(ip)-1   !nr of pomerons per proj
            npt(it)=npt(it)-1   !nr of pomerons per targ
          endif

c Update xp and xm of remnant, and change the limit to have big enought mass.

        anip=dble(npp(ip))
        anit=dble(npt(it))
        xpp(ip)=xpp(ip)-xppr(n,k)
        xppmn(ip)=min(1.d0,anip*xpmn(ip)/xmpmx(ip))
        xmt(it)=xmt(it)-xmpr(n,k)
        xmtmn(it)=min(1.d0,anit*xtmn(it)/xptmx(it))

      endif

      end

c-------------------------------------------------------------------------
      double precision function wmatrix(k,n)
c-------------------------------------------------------------------------
c proposal matrix w(a->b), considering pomeron type, x, y
c-------------------------------------------------------------------------

      include 'epos.incems'
      double precision wzero,wzerox,Womegak,xprem,xmrem,om1intgck
      common/cwzero/wzero,wzerox


c      ip=iproj(k)
c      it=itarg(k)

      if(idpr(n,k).eq.0)then
        wmatrix=wzero
      else
          xprem=1.d0!xpp(ip)+xppr(n,k)
          xmrem=1.d0!xmt(it)+xmpr(n,k)
          wmatrix=(1d0-wzero)/om1intgck(k,xprem,xmrem)
     *           *Womegak(xppr(n,k),xmpr(n,k),xprem,xmrem,k)
      endif


      end

c-------------------------------------------------------------------------
      double precision function omega(n,k)
c-------------------------------------------------------------------------
c calculates partial omega for spot (k,n)
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      common/cwzero/wzero,wzerox
      double precision wzero,wzerox,eps
      parameter(eps=1.d-15)
      double precision PhiExpoK,omGamk,xp,xm,fom
      double precision plc,s
      common/cems5/plc,s
      common/nucl3/phi,bimp

      omega=0.d0

      ip=iproj(k)
      it=itarg(k)

      if(xpp(ip).lt.xppmn(ip)+eps.or.xpp(ip).gt.1.d0+eps)goto 1001
      if(xmt(it).lt.xmtmn(it)+eps.or.xmt(it).gt.1.d0+eps)goto 1001

      omega=xpp(ip)**dble(alplea(iclpro))
     &     *xmt(it)**dble(alplea(icltar))

c      ztg=0
c      zpj=0
c      nctg=0
c      ncpj=0
c      zsame=nprt(k)
c      if(idpr(n,k).gt.0)then
c        if(nprt(k).le.0)stop'omega: nprt(k) should be positive !!!!    '
c        zsame=zsame-1
c      endif
c      nlpop=nint(zsame)
c      nlpot=nint(zsame)
c      bglaub2=sigine/10./pi        !10= fm^2 -> mb
c      bglaub=sqrt(bglaub2)
c      b2x=epscrp*epscrp*bglaub2
c      b2=bk(k)**2
c      ztgx=epscrw*exp(-b2/2./b2x)*fscra(engy/egyscr)
c      zpjx=epscrw*exp(-b2/2./b2x)*fscra(engy/egyscr)
c
c      if(koll.gt.1)then
c        do li=1,lproj(ip)
c          kk=kproj(ip,li)
c          if(kk.ne.k)then
c            b2=bk(kk)**2
c            if(b2.le.bglaub2)nctg=nctg+1
c            ztg=ztg+epscrw*exp(-b2/2./b2x)*fscro(engy/egyscr)
c            nlpop=nlpop+nprt(kk)
c          endif
c        enddo
c        do li=1,ltarg(it)
c          kk=ktarg(it,li)
c          if(kk.ne.k)then
c            b2=bk(kk)**2
c            if(b2.le.bglaub2)ncpj=ncpj+1
c            zpj=zpj+epscrw*exp(-b2/2./b2x)*fscro(engy/egyscr)
c            nlpot=nlpot+nprt(kk)
c          endif
c        enddo
c      endif
      !  zpjx+zpj is equal to zparpro(k)
      !  ztgx+ztg is equal to zpartar(k)
      zprj=zparpro(k)  !zsame+zpj
      ztgt=zpartar(k)  !zsame+ztg
c      if(npp(ip).gt.nfctrl)stop'nfctrl too small (1)         '
c      if(npt(it).gt.nfctrl)stop'nfctrl too small (2)         '
      if(idpr(n,k).eq.0)then
        omega=omega*wzerox
      else
        xp=xppr(n,k)
        xm=xmpr(n,k)
c        !-------------------------------------------------------------------------
c        ! fom : part of Phi regularization; Phi -> Phi^(n) (n = number of Poms)
c        ! Phi^(0) relevant for Xsect unchanged, apart of (maybe) normalization (Z)
c        !-------------------------------------------------------------------------
        omega=omega*omGamk(k,xp,xm)*gammaV(k)*fom(zprj,xm,bk(k))
     &                                       *fom(ztgt,xp,bk(k))
      endif

      omega=omega*PhiExpoK(k,xpp(ip),xmt(it))


      if(omega.le.0.d0)goto 1001

      if(koll.gt.1)then
        do li=1,lproj(ip)
          kk=kproj(ip,li)
          if(itarg(kk).ne.it)then
            ipl=iproj(kk)
            itl=itarg(kk)
            omega=omega*PhiExpoK(kk,xpp(ipl),xmt(itl))
            if(omega.le.0.d0)goto 1001
          endif
        enddo
        do li=1,ltarg(it)
          kk=ktarg(it,li)
          if(iproj(kk).ne.ip)then
            ipl=iproj(kk)
            itl=itarg(kk)
            omega=omega*PhiExpoK(kk,xpp(ipl),xmt(itl))
            if(omega.le.0.d0)goto 1001
          endif
        enddo
      endif

      if(omega.lt.1.d-100)then
        if(ish.ge.6)write(*,*)'omega-exit',omega
        omega=0.d0
      elseif(omega.gt.1.d100)then
        if(ish.ge.6)write(*,*)'omega-exit',omega
        omega=0.d0
      endif

      return

 1001 continue

      omega=0.d0
      return

      end

c-------------------------------------------------------------------------
      double precision function fom(z,x,b)
c-------------------------------------------------------------------------
      include 'epos.inc'
      double precision x,u,w,z0
      !----------------------------------------------------------------
      ! part of Phi regularization; Phi -> Phi^(n) (n = number of Poms)
      ! Phi^(0) relevant for Xsect unchanged
      !----------------------------------------------------------------
      fom=1d0
      if(z.gt.0..and.alpfomi.gt.0.)then
       z0=dble(alpfomi)
       u=dble(z**gamfom)
c       u=z0*dble(z/z0)**2.
       w=u/z0*exp(-dble(b*b/delD(1,iclpro,icltar)))
c       w=10.d0*u
       !---------------------------------------------------
       !e=exp(-0.05*u)  !analytic function with e(0)=1
       !fom=((1-u)+(u+w)*sqrt(x**2+((u-1+e)/(u+w))**2))
       !     fom(z=0)=1  fom(x=0)=e  fom(x=1)~w
       !---------------------------------------------------
       fom=1.d0+w*x**betfom
       !---------------------------------------------------
      endif
      end

c-------------------------------------------------------------------------
      subroutine ProNucSpl(ir,ii)
c-------------------------------------------------------------------------
c propose nuclear splitting
c for proj (iep) if ir=1 or target (iet) if ir=-1
c If remnant full of parton, force excitation to mimic fan diagram connections
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      double precision alp,eps,xrr,zfrac(kollmx),zsum,xk,proba,xp,xm,xr0
     &,drangen,omGamk,PomInt!,PomIncPExact,PomIncMExact
      integer knopp(kollmx)
      parameter(eps=1.d-10)


      if(ir.eq.1)then                   !proj

        ip=ii
        zzz=zzremn(ip,1)!excite more if many nucleon connected or if in nucleus
        if(ish.ge.4)write(ifch,*)'ProNucSpl proj:',ip,zzz
        r=rangen()
        if(r.gt.exp(-min(50.,zrminc*zzz)))then
          iep(ip)=5
          if(kolp(ip).eq.0)then
          if(1.d0-xpp(ip).gt.eps)stop'ProNucSpl: should not happen (2)'
            alp=1.d0/(1.d0+dble(alplea(iclpro)))
            ncon=0
            zsum=0d0
            do l=1,lproj3(ip)
              kp=kproj3(ip,l)
              it=itarg(kp)
              if(kolt(it).gt.0)then
                do m=1,ltarg3(it)
                  kt=ktarg3(it,m)
                  if(itpr(kt).gt.0)then
                    do n=1,nprmx(kt)
                      if(xpr(n,kt).gt.xzcutpar(kt))then
                        ncon=ncon+1
                        knopp(ncon)=kt
                        zfrac(ncon)=dble(zparpro(kt))
                        zsum=zsum+zfrac(ncon)
                      endif
                    enddo
                  endif
                enddo
              endif
            enddo
            if(ish.ge.4)write(ifch,*)'ProNucSpl zsum:',zsum,ncon
            if(zsum.gt.0d0)then
              xr0=xpp(ip)-drangen(xpp(ip))**alp
              xrr=xr0
              if(ish.ge.6)write(ifch,*)'xrr:',xrr
              do nc=1,ncon
                k=knopp(nc)
                xk=zfrac(nc)/zsum*xr0
                if(ish.ge.6)write(ifch,*)'xk:',nc,k,xk
                ipp=iproj(k)
                itt=itarg(k)
                do n=1,nprmx(k)
                  if(xpr(n,k).gt.xzcutpar(k))then
                    xp=xppr(n,k)+xk
                    if(xp.lt.1d0)then
c accept xp with probability GFF/PomIncExact
                     PomInt=PomInck(k)
                     if(PomInt.gt.0d0)then
                      proba=omGamk(k,xp,xmpr(n,k))
     &                   *xrr**dble(alplea(iclpro))
     &                   *xpp(ipp)**dble(alplea(iclpro))
     &                   *xmt(itt)**dble(alplea(icltar))
     &                   /PomInt
                      if(drangen(proba).lt.proba)then !accept xp for pair k
                        xppr(n,k)=xp
                        xpr(n,k)=xppr(n,k)*xmpr(n,k)
                        ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
                        xpp(ip)=xpp(ip)-xk
                        knucnt(1,k)=knucnt(1,k)+1 !store info of momentum transfer
                        irnuc(knucnt(1,k),1,k)=ip !in case of virpom later
                        npnuc(knucnt(1,k),1,k)=n
                        xxnuc(knucnt(1,k),1,k)=xk
                        if(ish.ge.6)write(ifch,*)'Transfer:'
     &                                          ,knucnt(1,k),k,n,xk,ip
                        goto 10
                      endif
                     endif
                    endif
                  endif
                enddo
                xrr=xrr-xk
 10             continue
              enddo
              if(xrr.lt.eps)then
                iep(ip)=0       !excitation not possible
                zzremn(ip,1)=0.
              endif
              if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iep(ip),xrr
            else
              iep(ip)=0
              zzremn(ip,1)=0.
              if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iep(ip)
            endif
          else
            if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iep(ip)
          endif
        else
          iep(ip)=0
        endif


      elseif(ir.eq.-1)then      !targ

        it=ii
        zzz=zzremn(it,2)!excite more if many nucleon connected or if in nucleus
        if(ish.ge.4)write(ifch,*)'ProNucSpl targ:',it,zzz
        r=rangen()
        if(r.gt.exp(-min(50.,zrminc*zzz)))then
          iet(it)=5
          if(kolt(it).eq.0)then
          if(1.d0-xmt(it).gt.eps)stop'ProNucSpl: should not happen (4)'
            alp=1.d0/(1.d0+dble(alplea(icltar)))
            ncon=0
            zsum=0d0
            do l=1,ltarg3(it)
              kt=ktarg3(it,l)
              ip=iproj(kt)
              if(kolp(ip).gt.0)then
                do m=1,lproj3(ip)
                  kp=kproj(ip,m)
                  if(itpr(kp).gt.0)then
                    do n=1,nprmx(kp)
                      if(xpr(n,kp).gt.xzcutpar(kp))then
                        ncon=ncon+1
                        knopp(ncon)=kp
                        zfrac(ncon)=dble(zpartar(kp))
                        zsum=zsum+zfrac(ncon)
                      endif
                    enddo
                  endif
                enddo
              endif
            enddo
            if(ish.ge.4)write(ifch,*)'ProNucSpl zsum:',zsum,ncon
            if(zsum.gt.0d0)then
              xr0=xmt(it)-drangen(xmt(it))**alp
              xrr=xr0
              if(ish.ge.6)write(ifch,*)'xrr:',xrr
              do nc=1,ncon
                k=knopp(nc)
                xk=zfrac(nc)/zsum*xr0
                if(ish.ge.6)write(ifch,*)'xk:',nc,k,xk
                ipp=iproj(k)
                itt=itarg(k)
                do n=1,nprmx(k)
                  if(xpr(n,k).gt.xzcutpar(k))then
                    xm=xmpr(n,k)+xk
                    if(xm.lt.1d0)then
c accept xp with probability GFF/PomIncExact
                     PomInt=PomInck(k)
                     if(PomInt.gt.0d0)then
                      proba=omGamk(k,xppr(n,k),xm)
     &                     *xpp(ipp)**dble(alplea(iclpro))
     &                     *xmt(itt)**dble(alplea(icltar))
     &                     *xrr**dble(alplea(icltar))
     &                     / PomInt
                      if(drangen(proba).lt.proba)then !accept xp for pair k
                        xmpr(n,k)=xm
                        xpr(n,k)=xppr(n,k)*xmpr(n,k)
                        ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
                        xmt(it)=xmt(it)-xk
                        knucnt(2,k)=knucnt(2,k)+1 !store info of momentum transfer
                        irnuc(knucnt(2,k),2,k)=it !in case of virpom later
                        npnuc(knucnt(2,k),2,k)=n
                        xxnuc(knucnt(2,k),2,k)=xk
                        if(ish.ge.6)write(ifch,*)'Transfer:'
     &                                          ,knucnt(2,k),k,n,xk,it
                        goto 20
                      endif
                     endif
                    endif
                  endif
                enddo
                xrr=xrr-xk
 20             continue
              enddo
              if(xrr.lt.eps)then
                iet(it)=0       !excitation not possible
                zzremn(it,2)=0.
              endif
              if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iet(it),xrr
            else
              iet(it)=0
              zzremn(it,2)=0.
              if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iet(it)
            endif
          else
            if(ish.ge.4)write(ifch,*)'ProNucSpl out:',iet(it)
          endif
        else
          iet(it)=0
        endif
      endif

      end

c-------------------------------------------------------------------------
      subroutine ProPoTy(k,n)
c-------------------------------------------------------------------------
c propose pomeron type
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      common/cems5/plc,s
      double precision s,plc
      double precision ww,w0,w1,w2,w3,w4,w5,w(0:7),aks,eps,zdiff
     *,xh,yp!,xp,xm
      parameter(eps=1.d-10)
      logical cont
      dimension nnn(3),kkk(3)

      if(idpr(n,k).eq.0)return
      ip=iproj(k)
      it=itarg(k)
      if(ish.ge.4)write(ifch,*)'ProPoTy:k,n,idpr,x',k,n,ip,it,nprt(k)
     *                                              ,idpr(n,k),xpr(n,k)
      if(idpr(n,k).ne.1)call utstop('ProPoTy: should not happen&')

      cont=.true.
      do i=1,3
        nnn(i)=0
        kkk(i)=0
      enddo

      idfpr(n,k)=1
      xh=xpr(n,k)
      yp=ypr(n,k)
c      xp=xppr(n,k)
c      xm=xmpr(n,k)
      nnn(3)=n
      kkk(3)=k

      if(iep(ip).ne.5)iep(ip)=-1
      if(iet(it).ne.5)iet(it)=-1


      idpr(n,k)=1

        w0=0.d0
        w1=0.d0
        w2=0.d0
        w3=0.d0
        w4=0.d0
        w5=0.d0

        call WomTy(w,xh,yp,k)


        if(w(0).gt.0.d0)w0=w(0)
        if(w(1).gt.0.d0)w1=w(1)
        if(iremn.ge.2)then
          if(w(2).gt.0.d0)then                 !q-g
            if(ivp(ip).gt.0)then
              w2=w(2)
            else
              w1=w1+w(2)
            endif
          endif
          if(w(3).gt.0.d0)then                 !g-q
            if(ivt(it).gt.0)then
              w3=w(3)
            else
              w1=w1+w(3)
            endif
          endif
          if(w(4).gt.0.d0)then                 !q-q
            if(ivp(ip)*ivt(it).gt.0)then
              w4=w(4)
            else
              w1=w1+w(4)
            endif
          endif
        else
          if(w(2).gt.0.d0)w2=w(2)
          if(w(3).gt.0.d0)w3=w(3)
          if(w(4).gt.0.d0)w4=w(4)
        endif
        if(w(5).gt.0.d0)w5=w(5)

        ww=w0+w1+w2+w3+w4+w5
        if(ish.ge.4)write(ifch,*)'ProPoTy:ww,ww_i'
     *       ,ww,w0/ww*100.d0,w1/ww*100.d0,w2/ww*100.d0
     *       ,w3/ww*100.d0,w4/ww*100.d0,w5/ww*100.d0


        aks=dble(rangen())*ww

        if(ww.lt.eps.or.aks.le.w0)then            !soft pomeron

          itpr(k)=-2*npommx        !Pair is not diffractive
          if(ish.ge.5)write(ifch,*)'ProPoTy:idpr',idpr(n,k)

        elseif(aks.ge.ww-w5)then !diffractive interaction

          itpr(k)=itpr(k)+2
c the probability to have a real diffractive Pomeron increase with Z and hard P
c          zdiff=exp(-min(50.d0,dble(zdfinc*(zparpro(k)+zpartar(k)))*w1))
          zdiff=1d0/sqrt(1d0+dble(zdfinc*(zparpro(k)+zpartar(k)))*w1)
c          print *,xpr(n,k),zdiff,w1!,zdiff2
          if(ish.ge.5)write(ifch,*)'ProPoTy:itpr',itpr(k),zdiff
          if(xpr(n,k).gt.xzcutpar(k).and.rangen().gt.zdiff)then
c          if(rangen().gt.zdiff)then
c High mass diffraction : Keep pomeron and choose between soft and semi-hard
            aks=dble(rangen())*(w0+w1)
            if(aks.gt.w0)then
              idpr(n,k)=3
              npr(3,k)=npr(3,k)+1
              npr(1,k)=npr(1,k)-1
              bhpr(n,k)=bk(k)
              idhpr(n,k)=0
c            elseif(iLHC.eq.1)then
c              idpr(n,k)=-1
            endif
          else
           if(iLHC.eq.1)then     !LHC tune 
c keep soft Pomeron for later 
              idpr(n,k)=-1
           else                     !original CR
c Low mass diffraction : no pomeron
c restore x from nuclear splitting
            if(knucnt(1,k).gt.0)then
              do nuc=1,knucnt(1,k)
                if(npnuc(nuc,1,k).eq.n)then
                  ipp=irnuc(nuc,1,k)
                  xpp(ipp)=xpp(ipp)+xxnuc(nuc,1,k)
                  if(xpp(ipp)-1d0.ge.-1d-10)iep(ipp)=0
                  xppr(n,k)=xppr(n,k)-xxnuc(nuc,1,k)
                  xpr(n,k)=xppr(n,k)*xmpr(n,k)
                  ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
                  npnuc(nuc,1,k)=0 !to be sure not to use it again
                endif
              enddo
            endif
            if(knucnt(2,k).gt.0)then
              do nuc=1,knucnt(2,k)
                if(npnuc(nuc,2,k).eq.n)then
                  itt=irnuc(nuc,2,k)
                  xmt(itt)=xmt(itt)+xxnuc(nuc,2,k)
                  if(xmt(itt)-1d0.ge.-1d-10)iet(itt)=0
                  xmpr(n,k)=xmpr(n,k)-xxnuc(nuc,2,k)
                  xpr(n,k)=xppr(n,k)*xmpr(n,k)
                  ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
                  npnuc(nuc,2,k)=0 !to be sure not to use it again
                endif
              enddo
            endif
            call RemPom(k,n)
            idfpr(n,k)=0
            npr(0,k)=npr(0,k)+1 !nr of empty cells
            kolp(ip)=kolp(ip)-1 !suppress diffractive collision from the remnant
            kolt(it)=kolt(it)-1 !it will be restored if the pair is diffractive
            if(ish.ge.6)write(ifch,*)'ProPoTy:idpr',idpr(n,k)
          endif

         endif

        else

          itpr(k)=-2*npommx        !Pair is not diffractive
          idpr(n,k)=3
          if(ish.ge.5)write(ifch,*)'ProPoTy:idpr',idpr(n,k)
          npr(3,k)=npr(3,k)+1
          npr(1,k)=npr(1,k)-1
          bhpr(n,k)=bk(k)

          aks=aks-w0
          if(aks.le.w1)then                             !gg-pomeron
            idhpr(n,k)=0
          elseif(aks.le.w1+w2)then                      !qg-pomeron
            idhpr(n,k)=1
            ivp(ip)=ivp(ip)-1
          elseif(aks.le.w1+w2+w3)then                   !gq-pomeron
            idhpr(n,k)=2
            ivt(it)=ivt(it)-1
          elseif(aks.le.w1+w2+w3+w4)then                !qq-pomeron
            idhpr(n,k)=3
            ivp(ip)=ivp(ip)-1
            ivt(it)=ivt(it)-1
          else
            call utstop('ems-unknown pomeron&')
          endif
          if(ish.ge.6)write(ifch,*)'ProPoTy:idhpr',idhpr(n,k)
     &         ,' |',ip,ivp(ip),' |',it,ivt(it)

        endif
        
        if(idfpr(n,k).eq.1)then
          antot=antot+1
          antotf=antotf+1
          if(abs(idpr(n,k)).eq.1)then
            ansf=ansf+1
            ansff=ansff+1
          endif
          if(idpr(n,k).eq.3)then
            ansh=ansh+1
            anshf=anshf+1
          endif
        endif

      do i=3,1,-1

        if(nnn(i).ne.0.and.kkk(i).ne.0.and.cont)then

          if(idpr(nnn(i),kkk(i)).eq.3)then

                       !Backup soft Pomeron if sh not possible later

            kb=kkk(i)
            nb=nnn(i)
            ip=iproj(kb)
            it=itarg(kb)
            do nn=1,nprmx(kb)
              if(idpr(nn,kb).eq.0)then !empty spot
                nbkpr(nb,kb)=nn
                nvpr(nn,kb)=nb
                idpr(nn,kb)=1
                ivpr(nn,kb)=2
                xpr(nn,kb)=xpr(nb,kb)
                ypr(nn,kb)=ypr(nb,kb)
                xppr(nn,kb)=xppr(nb,kb)
                xmpr(nn,kb)=xmpr(nb,kb)
                idfpr(nn,kb)=-idfpr(nb,kb)
                bhpr(nn,kb)=bhpr(nb,kb)
                idp1pr(nn,kb)=0
                idp2pr(nn,kb)=0
                idm1pr(nn,kb)=0
                idm2pr(nn,kb)=0
                xm1pr(nn,kb)=0.d0
                xp1pr(nn,kb)=0.d0
                xm2pr(nn,kb)=0.d0
                xp2pr(nn,kb)=0.d0
                xxm1pr(nn,kb)=0.d0
                xym1pr(nn,kb)=0.d0
                xxp1pr(nn,kb)=0.d0
                xyp1pr(nn,kb)=0.d0
                xxm2pr(nn,kb)=0.d0
                xym2pr(nn,kb)=0.d0
                xxp2pr(nn,kb)=0.d0
                xyp2pr(nn,kb)=0.d0
                goto 10
              endif
            enddo
      if(ish.ge.2)write(ifmt,*)'no empty lattice site, backup lost'

 10         continue
          endif
        endif
      enddo

      return
      end

c-------------------------------------------------------------------------
      subroutine ProDiSc(k)
c-------------------------------------------------------------------------
c propose diffractive scattering
c-------------------------------------------------------------------------

      include 'epos.incems'

      ip=iproj(k)
      it=itarg(k)
      kolp(ip)=kolp(ip)+itpr(k)/2 !number of diffractive Pomerons
      kolt(it)=kolt(it)+itpr(k)/2 !on remnants


      end

c-------------------------------------------------------------------------
      subroutine ProReEx(ir,ii)
c-------------------------------------------------------------------------
c propose remnant excitation
c for proj (iep) if ir=1 or target (iet) if ir=-1:
c 0 = no,  1 = inel excitation,  2 = diffr excitation
c fixed before : 5 = excit due to split without connection
c fixed after : 3 = large excitation due to # quark > 3
c               6 = active spectator (get small pt and used for mass)
c propose "pion exchange" process for diffractive remnant excitation
c for proj (iep) if ir=2 or target (iet) if ir=-2:
c 4 = diffr excitation but with fixed minimum excited mass
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'


      if(ir.eq.1)then                   !proj

        ip=ii
        mine=0
        mdif=0
        do l=1,lproj(ip)
          kp=kproj(ip,l)
          if(itpr(kp).lt.0)mine=1
          if(itpr(kp).gt.0)mdif=1
        enddo
        r=rangen()
        if(mine.eq.1)then   !inelastic
          if(iremn.eq.1)then
c increase excitation probability with number of close enough nucleons
            if(r.lt.1.-(1.-rexndi(iclpro))**(kolp(ip)
     &          *(1.+rexres(iclpro)*log(max(1.,float(lproj(ip)))))))then
              iep(ip)=1
            else
              iep(ip)=0
            endif
          elseif(iremn.ne.0)then  
c increase of inelastic remnant excitation in pA needed for Barton
            if(r.lt.1.-(1.-rexndi(iclpro))
     &         **(1.+rexres(iclpro)*float(lproj(ip)-1)))then
              iep(ip)=1
            else
              iep(ip)=0
            endif
          else!if(iremn.ne.2)then
            if(r.lt.rexndi(iclpro))then
              iep(ip)=1
            else
              iep(ip)=0
            endif
          endif
        elseif(mdif.eq.1)then        !diffr
          if(iremn.eq.1)then
c increase excitation probability with number of close enough nucleons
            if(r.lt.1.-(1.-rexdif(iclpro))**(kolp(ip)
     &          *(1.+rexres(iclpro)*log(max(1.,float(lproj(ip)))))))then
              iep(ip)=2
            else
              iep(ip)=0
            endif
          elseif(iremn.ne.0)then
c increase of diffractive remnant excitation in pA needed for Barton
            if(r.lt.1.-(1.-rexdif(iclpro))
     &        **(1.+rexres(iclpro)*float(lproj(ip)-1)))then
              iep(ip)=2
            else
              iep(ip)=0
            endif
          else
            if(r.lt.1.-(1.-rexdif(iclpro)))then
              iep(ip)=2
            else
              iep(ip)=0
            endif
          endif
        elseif(iep(ip).ne.5)then
c for non-excited spectators
          iep(ip)=0
        endif

      elseif(ir.eq.-1)then                !targ

        it=ii
        mine=0
        mdif=0
        do l=1,ltarg(it)
          kt=ktarg(it,l)
          if(itpr(kt).lt.0)mine=1
          if(itpr(kt).gt.0)mdif=1
        enddo
        r=rangen()
        if(mine.eq.1)then   !inelastic
          if(iremn.eq.1)then
            if(r.lt.1.-(1.-rexndi(icltar))**(kolt(it)
     &          *(1.+rexres(icltar)*log(max(1.,float(ltarg(it)))))))then
              iet(it)=1
            else
              iet(it)=0
            endif
          elseif(iremn.ne.0)then
c increase of inelastic remnant excitation in pA needed for Barton
            if(r.lt.1.-(1.-rexndi(icltar))
     &         **(1.+rexres(icltar)*float(ltarg(it)-1)))then
              iet(it)=1
            else
              iet(it)=0
            endif
          else
            if(r.lt.rexndi(icltar))then
              iet(it)=1
             else
              iet(it)=0
            endif
         endif
        elseif(mdif.eq.1)then        !diffr
          if(iremn.eq.1)then
              if(r.lt.1.-(1.-rexdif(icltar))**(kolt(it)
     &      *(1.+rexres(icltar)*log(max(1.,float(ltarg(it)))))))then
                iet(it)=2
              else
                iet(it)=0
              endif
            elseif(iremn.ne.0)then
c increase of diffractive remnant excitation in pA needed for Barton anb xsection
              if(r.lt.1.-(1.-rexdif(icltar))
     &      **(1.+rexres(icltar)*float(ltarg(it)-1)))then
                iet(it)=2
              else
                iet(it)=0
              endif
            else
              if(r.lt.1.-(1.-rexdif(icltar)))then
                iet(it)=2
              else
                iet(it)=0
              endif
            endif
        elseif(iet(it).ne.5)then
          iet(it)=0
        endif

      elseif(ir.eq.2)then                !proj diff excitation

c minimum mass excitation
        ip=ii
        r=rangen()
c        if(r.lt.rexpdif(iclpro))iep(ip)=4  
        if(r.lt.rexpdif(iclpro)
     & **(1.+rexres(iclpro)*float(lproj(ip)-1)))iep(ip)=4  

      elseif(ir.eq.-2)then                !targ diff excitation

cminimum mass excitation
        it=ii
        r=rangen()
c        if(r.lt.rexpdif(icltar))iet(it)=4  
        if(r.lt.rexpdif(icltar)
     & **(1.+rexres(icltar)*float(ltarg(it)-1)))iet(it)=4  

      endif

      end


c-------------------------------------------------------------------------
      subroutine ProDiPt(k,iqq,iret)
c-------------------------------------------------------------------------
c propose transverse momentum for diffractive interaction
c iqq=1  : fix pt for non diffractive pair (and check if all pairs are still valid)
c iqq=2  : diffractive pt with mass dependence
c-------------------------------------------------------------------------

      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.inc'
      double precision xxe(kollmx),xye(kollmx),pt2,am0,am1,am2!,p5sqpr,p5sqtg
      double precision plc,s,xxpnew,xypnew,xxtnew,xytnew,rannorm
      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      save xxe,xye

      ip=iproj(k)
      it=itarg(k)
      pt=0.
      phi=0.


c generate p_t for diffractive
      if(iqq.eq.1)then

       if(ptdiff.gt.0.)then
         if(itpr(k).eq.2)then
           pt=ranpt()*ptdiff/(1.+0.02*max(0.,sngl(log(s))))
         elseif(itpr(k).eq.0)then   !pt for non-wounded nucleon (usefull in ProRem to avoid problem in utrescl)
           if(iLHC.eq.1)then
             pt = sngl(RANNORM(0.088D0,0.044D0))  !limited by some data like sal.optns
           else
             ptnw=0.005
             pt=ranptd()*ptnw
           endif
           if(kolp(ip).eq.0.and.iep(ip).le.0)iep(ip)=6   !active spectators
           if(kolt(it).eq.0.and.iet(it).le.0)iet(it)=6
         else
           xxe(k)=0d0
           xye(k)=0d0
           goto 10
         endif
         phi=2.*pi*rangen()
         xxe(k)=dble(pt*cos(phi))
         xye(k)=dble(pt*sin(phi))
       else
         xxe(k)=0d0
         xye(k)=0d0
       endif

c update remnant p_t

 10    xxp(ip)=xxp(ip)-xxe(k)
       xyp(ip)=xyp(ip)-xye(k)
       xxt(it)=xxt(it)+xxe(k)
       xyt(it)=xyt(it)+xye(k)

       if(ish.ge.8)write(ifch,'(a,i5,3i4,4g13.5)')
     &                    'ProDiPt',k,ip,it,itpr(k),pt,phi,xxe(k),xye(k)

       if(itpr(k).ne.0.and.itpr(k).ne.3)iret=0
!to simulate the fact that originally we had a Pomeron
c         if(koll.le.2)then
c           call StoCon(-k,k,1)  !to fixe mass of corresponding remnants
c           xpp(ip)=xpp(ip)-xppr(1,k)
c           xpt(it)=xpt(it)+xppr(1,k)
c           xmt(it)=xmt(it)-xmpr(1,k)
c           xmp(ip)=xmp(ip)+xmpr(1,k)
c           idpr(1,k)=0
c           xpr(1,k)=0.d0
c           ypr(1,k)=0.d0
c           xppr(1,k)=0.d0
c           xmpr(1,k)=0.d0
c         endif
c         p5sqpr=xpp(ip)*xmp(ip)*s-dble(amproj*amproj)
c         p5sqtg=xpt(it)*xmt(it)*s-dble(amtarg*amtarg)
c         phi=2.*pi*rangen()
c         ntry=0
c 20      ntry=ntry+1
c         pt=ranptcut(ptsems)*ptsend**2
c         if(ntry.lt.100.and.(p5sqpr-dble(pt*pt).lt.0.d0
c     &                   .or.p5sqtg-dble(pt*pt).lt.0.d0))then
c             goto 20
c         else
c           pt=ranptcut(ptsems)*ptsendi
c         endif
c         xxe(k)=dble(pt*cos(phi))
c         xye(k)=dble(pt*sin(phi))
c         xxp(ip)=xxp(ip)-xxe(k)
c         xyp(ip)=xyp(ip)-xye(k)
c         xxt(it)=xxt(it)+xxe(k)
c         xyt(it)=xyt(it)+xye(k)
c       endif

      elseif(itpr(k).eq.2.and.ptdiff.ne.0.)then

        pt2=xxe(k)*xxe(k)+xye(k)*xye(k)
        if(pt2.gt.0d0)then
          am0=dble(amproj**2*amtarg**2)
          am1=max(dble(amproj**2),xpp(ip)*xmp(ip)*s
     &              -xxp(ip)*xxp(ip)-xyp(ip)*xyp(ip))
          am2=max(dble(amtarg**2),xpt(it)*xmt(it)*s
     &              -xxp(it)*xxp(it)-xyp(it)*xyp(it))
          ptd=ptdiff/(1.+0.02*max(0.,sngl(log(s*am0/am1/am2)))) !0.02 comes from data (Z. Phys. C 67, 227-237, 1995)
c           ad=pi/4./ptd**2
c           r=rangen()
          pt=ranpt()*ptd        !sqrt(-alog(r)/ad)
        else
          return
        endif
        if(ish.ge.8)write(ifch,'(a,i5,2i4,5g13.5)')
     &                    'ProDiPt',k,ip,it,pt,sqrt(pt2),ptd,am1,am2
c suppress the pt given with iqq=1 and give a new one taking into account the mass (iqq=2) with the same angle phi
        pt=pt/sqrt(pt2)
        xxe(k)=xxe(k)*pt
        xye(k)=xye(k)*pt

c update remnant p_t if enough energy available
        xxpnew=xxp(ip)-xxe(k)
        xypnew=xyp(ip)-xye(k)
        xxtnew=xxt(it)+xxe(k)
        xytnew=xyt(it)+xye(k)
        if((iep(ip).eq.0.or.
     &      xpp(ip)*xmp(ip)*s-xxpnew*xxpnew-xypnew*xypnew
     &      .gt.1.3d0*dble(pptl(5,npproj(ip)))**2)
     &.and.(iet(it).eq.0.or.
     &      xpt(it)*xmt(it)*s-xxtnew*xxtnew-xytnew*xytnew
     &      .gt.1.3d0*dble(pptl(5,nptarg(it)))**2))then
          xxp(ip)=xxp(ip)-xxe(k)
          xyp(ip)=xyp(ip)-xye(k)
          xxt(it)=xxt(it)+xxe(k)
          xyt(it)=xyt(it)+xye(k)
        endif

       endif

       end

c-------------------------------------------------------------------------
      subroutine ProSePt(k,n,iret)
c-------------------------------------------------------------------------
c propose transverse momentum for string ends
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      common/cems5/plc,s
      double precision s,plc
      double precision x1p,x2p,x1t,x2t

      if(ivpr(n,k).eq.2)return            !Backup Pomeron

      ip=iproj(k)
      it=itarg(k)
      amk0=1. ! included in ptsend !(qmass(1)+qmass(2)+qmass(3))/3.     !mass for mt distribution

      ptsecut=ptsecu        !cut for gaussian distribution (center around 0.4)

c generate p_t for string ends  (proj)
      iret=0
      ntry=0
 10   ntry=ntry+1
      xxp1pr(n,k)=0d0
      xyp1pr(n,k)=0d0
      xxp2pr(n,k)=0d0
      xyp2pr(n,k)=0d0
      xxm1pr(n,k)=0d0
      xym1pr(n,k)=0d0
      xxm2pr(n,k)=0d0
      xym2pr(n,k)=0d0
      x1p=0d0
      x2p=0d0
      x1t=0d0
      x2t=0d0
      pt=0.
      phi=0.
      if(ntry.gt.100)then
        iret=1
        goto 1000               !no pt
      endif
c
c      !---proj-----
        ptsef=ptsend
        if(iep(ip).eq.0)ptsef=ptsendi
        ptsendx = ptsems
        ptsendy = ptsendx
        if(iLHC.eq.0)ptsendy = ptsendx*2

        ipt=1

c 2 step pt : first give pt between remnant and Pomeron and then between 
c string ends on the same side.
        if(iLHC.eq.-1)ipt=2

        do ii=1,ipt

      if(idp1pr(n,k).gt.0)then
        if(ii.eq.1)then
         if(idp1pr(n,k).eq.4.or.idp1pr(n,k).eq.5)then   !diquarks
           amk1=amk0*ptsendy+qmass(0) !mass for mt distribution with bounding energy for diquark
         else
           amk1=amk0*ptsendx
         endif
c         if(iep(ip).eq.0)amk1=0.
         if(iep(ip).eq.0)then
           pt=ranptd()*ptsef
         else
           pt=ranptcut(ptsecut)*ptsef
c           pt=ranptd()*ptsef
           pt=pt+amk1
         endif
c         pt=ranptcut(ptsecut)*ptsef
c         pt=pt+amk1
c         pt=ranptd()*ptsef
c         pt=sqrt(pt*pt+amk1*amk1)
       else
         pt=ranpt()*ptfraqq
       endif
         phi=2.*pi*rangen()
         xxp1pr(n,k)=xxp1pr(n,k)+dble(pt*cos(phi))
         xyp1pr(n,k)=xyp1pr(n,k)+dble(pt*sin(phi))
      else
         xxp1pr(n,k)=0d0
         xyp1pr(n,k)=0d0
      endif
      if(idp2pr(n,k).gt.0)then
        if(ii.eq.1)then
         if(idp2pr(n,k).eq.4.or.idp2pr(n,k).eq.5)then
           amk1=amk0*ptsendy+qmass(0) !mass for mt distribution with bounding energy for diquark
         else
           amk1=amk0*ptsendx
         endif
c         if(iep(ip).eq.0)amk1=0.
         if(iep(ip).eq.0)then
           pt=ranptd()*ptsef
         else
           pt=ranptcut(ptsecut)*ptsef
c           pt=ranptd()*ptsef
           pt=pt+amk1
         endif
c         pt=ranptcut(ptsecut)*ptsef
c         pt=pt+amk1
c         pt=ranptd()*ptsef
c         pt=sqrt(pt*pt+amk1*amk1)
         phi=2.*pi*rangen()
       else    !use pt and phi from other string ends
         pt=-pt
       endif 
         xxp2pr(n,k)=xxp2pr(n,k)+dble(pt*cos(phi))
         xyp2pr(n,k)=xyp2pr(n,k)+dble(pt*sin(phi))
      else
         xxp2pr(n,k)=0d0
         xyp2pr(n,k)=0d0
      endif
c generate p_t for string ends  (targ)


c      !---targ-----
        ptsef=ptsend
        if(iet(it).eq.0)ptsef=ptsendi
        ptsendx = ptsems
        ptsendy = ptsendx
        if(iLHC.eq.0)ptsendy = ptsendx*2.

      if(idm1pr(n,k).gt.0)then
        if(ii.eq.1)then
         if(idm1pr(n,k).eq.4.or.idm1pr(n,k).eq.5)then
           amk1=amk0*ptsendy+qmass(0) !mass for mt distribution with bounding energy for diquark
         else
           amk1=amk0*ptsendx
         endif
c         if(iet(it).eq.0)amk1=0.
         if(iet(it).eq.0)then
           pt=ranptd()*ptsef
         else
           pt=ranptcut(ptsecut)*ptsef
c           pt=ranptd()*ptsef
           pt=pt+amk1
         endif
c         pt=ranptcut(ptsecut)*ptsef
c         pt=pt+amk1
c         pt=ranptd()*ptsef
c         pt=sqrt(pt*pt+amk1*amk1)
       else
         pt=ranpt()*ptfraqq
       endif
         phi=2.*pi*rangen()
         xxm1pr(n,k)=xxm1pr(n,k)+dble(pt*cos(phi))
         xym1pr(n,k)=xym1pr(n,k)+dble(pt*sin(phi))
      else
         xxm1pr(n,k)=0d0
         xym1pr(n,k)=0d0
      endif
      if(idm2pr(n,k).gt.0)then
        if(ii.eq.1)then
         if(idm2pr(n,k).eq.4.or.idm2pr(n,k).eq.5)then
           amk1=amk0*ptsendy+qmass(0) !mass for mt distribution with bounding energy for diquark
         else
           amk1=amk0*ptsendx
         endif
c         if(iet(it).eq.0)amk1=0.
         if(iet(it).eq.0)then
           pt=ranptd()*ptsef
         else
           pt=ranptcut(ptsecut)*ptsef
c           pt=ranptd()*ptsef
           pt=pt+amk1
         endif
c         pt=ranptcut(ptsecut)*ptsef
c         pt=pt+amk1
c         pt=ranptd()*ptsef
c         pt=sqrt(pt*pt+amk1*amk1)
         phi=2.*pi*rangen()
       else    !use pt and phi from other string ends
         pt=-pt
       endif 
         xxm2pr(n,k)=xxm2pr(n,k)+dble(pt*cos(phi))
         xym2pr(n,k)=xym2pr(n,k)+dble(pt*sin(phi))
      else
         xxm2pr(n,k)=0d0
         xym2pr(n,k)=0d0
      endif

      if(ii.eq.1)then    !balance pt bwteen string ends and remnant

        x1p=xxp(ip)-xxp1pr(n,k)-xxp2pr(n,k)
        x2p=xyp(ip)-xyp1pr(n,k)-xyp2pr(n,k)
        x1t=xxt(it)-xxm1pr(n,k)-xxm2pr(n,k)
        x2t=xyt(it)-xym1pr(n,k)-xym2pr(n,k)

        if(iLHC.eq.1)then       !check energy
          if(x1p**2+x2p**2+2.*amproj**2.ge.xpp(ip)*s)goto 10
          if(x1t**2+x2t**2+2.*amtarg**2.ge.xmt(it)*s)goto 10
        endif

      endif

      if(ish.ge.8)write(ifch,*) 'ProSePt',ii,n,k
     *   ,sqrt(xxp1pr(n,k)**2+xyp1pr(n,k)**2)
     *   ,sqrt(xxp2pr(n,k)**2+xyp2pr(n,k)**2)
     *   ,sqrt(xxm1pr(n,k)**2+xym1pr(n,k)**2)
     *   ,sqrt(xxm2pr(n,k)**2+xym2pr(n,k)**2)

      enddo


c update remnant p_t (pomeron)
        xxp(ip)=x1p
        xyp(ip)=x2p
        xxt(it)=x1t
        xyt(it)=x2t

c update backup soft pomeron p_t if exist

 1000   if(nbkpr(n,k).ne.0)then
          nn=nbkpr(n,k)
          xxp1pr(nn,k)=xxp1pr(n,k)
          xyp1pr(nn,k)=xyp1pr(n,k)
          xxp2pr(nn,k)=xxp2pr(n,k)
          xyp2pr(nn,k)=xyp2pr(n,k)
          xxm1pr(nn,k)=xxm1pr(n,k)
          xym1pr(nn,k)=xym1pr(n,k)
          xxm2pr(nn,k)=xxm2pr(n,k)
          xym2pr(nn,k)=xym2pr(n,k)
        endif

        if(ish.ge.6)then
        write(ifch,*) 'ProSePt'
        write(ifch,'(4i14/4d14.3/4d14.3/)')
     * idp1pr(n,k),idp2pr(n,k),idm1pr(n,k),idm2pr(n,k)
     *,xxp1pr(n,k),xxp2pr(n,k),xxm1pr(n,k),xxm2pr(n,k)
     *,xyp1pr(n,k),xyp2pr(n,k),xym1pr(n,k),xym2pr(n,k)
        endif

        end

c-----------------------------------------------------------------------
      subroutine ProSeX(k,n,iret)
c-----------------------------------------------------------------------
c calculates x of string ends
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      common/cems5/plc,s
      double precision s,plc
      common/cems10/a(0:ntypmx),b(0:ntypmx),d(0:ntypmx)
      double precision a,b,d
     *,xp,xm,ap1,ap2,am1,am2,aamin1,aamin2,u
     *,xmn1,xmn2

      iret=0

      if(abs(itpr(k)).ne.1)return
      if(idpr(n,k).ne.1.or.ivpr(n,k).eq.0)return

      if(idp1pr(n,k).eq.0.and.idp2pr(n,k).eq.0
     * .and.idm1pr(n,k).eq.0.and.idm2pr(n,k).eq.0)
     *call utstop('no Pomeron in ProSex&')

      xp=xppr(n,k)
      xm=xmpr(n,k)
      ap1=a(idp1pr(n,k))
      ap2=a(idp2pr(n,k))
      am1=a(idm1pr(n,k))
      am2=a(idm2pr(n,k))
      aamin1=ammn(idp1pr(n,k)+idm2pr(n,k))
      aamin2=ammn(idp2pr(n,k)+idm1pr(n,k))
      xmn1=(aamin1**2+(xxp1pr(n,k)+xxm2pr(n,k))**2
     &               +(xyp1pr(n,k)+xym2pr(n,k))**2)/s
      xmn2=(aamin2**2+(xxp2pr(n,k)+xxm1pr(n,k))**2
     &               +(xyp2pr(n,k)+xym1pr(n,k))**2)/s

      ntry=0
 999  ntry=ntry+1
      if(ntry.gt.100)then
        iret=1
        if(ish.ge.5)write(ifch,*)'Problem in ProSex(k,n)',k,n
        return
      endif

    1 u=dble(rangen())**(1d0/(1d0+ap1))
      if(dble(rangen()).gt.(1d0-u)**ap2)goto1
      xp1pr(n,k)=u*xp
      xp2pr(n,k)=(1-u)*xp
    2 u=dble(rangen())**(1d0/(1d0+am1))
      if(dble(rangen()).gt.(1d0-u)**am2)goto2
      xm1pr(n,k)=u*xm
      xm2pr(n,k)=(1-u)*xm

      if(xp1pr(n,k)*xm2pr(n,k).lt.xmn1)then
      goto 999
c       fc=xp1pr(n,k)*xm2pr(n,k)/xmn1   !avoid virpom
c       if(fc.eq.0.)goto 999
c       xp1pr(n,k)=xp1pr(n,k)/sqrt(fc)
c       xm2pr(n,k)=xm2pr(n,k)/sqrt(fc)
      endif
      if(xp2pr(n,k)*xm1pr(n,k).lt.xmn2)then
      goto 999
c       fc=xp2pr(n,k)*xm1pr(n,k)/xmn2   !avoid virpom
c       if(fc.eq.0.)goto 999
c       xp2pr(n,k)=xp2pr(n,k)/sqrt(fc)
c       xm1pr(n,k)=xm1pr(n,k)/sqrt(fc)
      endif

      if(ish.ge.6)then
       write(ifch,*) 'ProSeX'
       write(ifch,'(2d28.3,i8)') xp,xm,ntry
       write(ifch,'(4d14.3)')xp1pr(n,k),xp2pr(n,k),xm1pr(n,k),xm2pr(n,k)
       write(ifch,'(4d14.3/)')xp1pr(n,k)*xm2pr(n,k)
     *                   ,xp2pr(n,k)*xm1pr(n,k),  xmn1, xmn2
      endif

      end
c-------------------------------------------------------------------------
      subroutine RmPt(k,n)
c-------------------------------------------------------------------------
c remove pt from pomeron
c-------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      ip=iproj(k)
      it=itarg(k)
      xxp(ip)=xxp(ip)+xxp1pr(n,k)+xxp2pr(n,k)
      xyp(ip)=xyp(ip)+xyp1pr(n,k)+xyp2pr(n,k)
      xxt(it)=xxt(it)+xxm1pr(n,k)+xxm2pr(n,k)
      xyt(it)=xyt(it)+xym1pr(n,k)+xym2pr(n,k)
      xp1pr(n,k)=0d0
      xp2pr(n,k)=0d0
      xm1pr(n,k)=0d0
      xm2pr(n,k)=0d0
      xxm1pr(n,k)=0d0
      xym1pr(n,k)=0d0
      xxp1pr(n,k)=0d0
      xyp1pr(n,k)=0d0
      xxm2pr(n,k)=0d0
      xym2pr(n,k)=0d0
      xxp2pr(n,k)=0d0
      xyp2pr(n,k)=0d0
      idp1pr(n,k)=0
      idm2pr(n,k)=0
      idp2pr(n,k)=0
      idm1pr(n,k)=0
      end

c-------------------------------------------------------------------------
      subroutine VirPom(k,n,id)
c-------------------------------------------------------------------------
c create virtual pomeron
c virtual pomeron: ivpr(n,k)=0, otherwise ivpr(n,k)=1
c-------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      common/col3/ncol,kolpt
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      double precision plc,s
      common/cems5/plc,s
      integer jcp(nflav,2),jct(nflav,2)
c      data nvir/0/
c      save nvir

      call utpri('VirPom',ish,ishini,3)

      if(idpr(n,k).eq.0)return

      ip=iproj(k)
      it=itarg(k)

      nnv=nvpr(n,k)
      nnb=nbkpr(n,k)

c                        nvir=nvir+1
c                   print *,'  ',id,'   ',nvir

      if(ish.ge.3)then
      write(ifch,*)"virpom ",id," (n,k)",n,k,nnb,nnv,nppr(n,k),itpr(k)
     &                            ,nprt(k),idpr(n,k),npr(1,k),npr(3,k)
      if(ish.ge.5)write(ifch,*)"remnant in",xpp(ip),xmt(it)
      endif

      if(nnv.ne.0)then
        nn=nnv
        kk=k
        if(idpr(nn,kk).eq.0)then
          nvpr(n,k)=0
        endif
      endif

      if(nnb.ne.0)then
        nn=nnb
        kk=k
        if(idpr(nn,kk).eq.0)then
          nbkpr(n,k)=0
        endif
      endif


      if(nbkpr(n,k).eq.0.and.nvpr(n,k).eq.0)then     !normal Pomeron

      npr(0,k)=npr(0,k)+1
      npp(ip)=npp(ip)-1
      npt(it)=npt(it)-1
      npr(idpr(n,k),k)=npr(idpr(n,k),k)-1
      nprt(k)=npr(1,k)+npr(3,k)
      antotf=antotf-1
      if(idpr(n,k).eq.1)ansff=ansff-1
      if(idpr(n,k).eq.3)anshf=anshf-1
      kolp(ip)=kolp(ip)-1
      kolt(it)=kolt(it)-1
      xxp(ip)=xxp(ip)+xxp1pr(n,k)+xxp2pr(n,k)
      xyp(ip)=xyp(ip)+xyp1pr(n,k)+xyp2pr(n,k)
      xxt(it)=xxt(it)+xxm1pr(n,k)+xxm2pr(n,k)
      xyt(it)=xyt(it)+xym1pr(n,k)+xym2pr(n,k)

c restore x from nuclear splitting
      if(knucnt(1,k).gt.0)then
        do nuc=1,knucnt(1,k)
          if(npnuc(nuc,1,k).eq.n)then
            ipp=irnuc(nuc,1,k)
            xpp(ipp)=xpp(ipp)+xxnuc(nuc,1,k)
            if(xpp(ipp).ge.1d0)iep(ipp)=0
            xppr(n,k)=xppr(n,k)-xxnuc(nuc,1,k)
            xpr(n,k)=xppr(n,k)*xmpr(n,k)
            ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
            npnuc(nuc,1,k)=0    !to be sure not to use it again
          endif
        enddo
      endif
      if(knucnt(2,k).gt.0)then
        do nuc=1,knucnt(2,k)
          if(npnuc(nuc,2,k).eq.n)then
            itt=irnuc(nuc,2,k)
            xmt(itt)=xmt(itt)+xxnuc(nuc,2,k)
            if(xmt(itt).ge.1d0)iet(itt)=0
            xmpr(n,k)=xmpr(n,k)-xxnuc(nuc,2,k)
            xpr(n,k)=xppr(n,k)*xmpr(n,k)
            ypr(n,k)=0.5D0*log(xppr(n,k)/xmpr(n,k))
            npnuc(nuc,2,k)=0    !to be sure not to use it again
          endif
        enddo
      endif

      xpp(ip)=xpp(ip)+xppr(n,k)
      xmt(it)=xmt(it)+xmpr(n,k)


      if(abs(itpr(k)).eq.1.and.nprt(k).eq.0)then !no more Pomeron on this pair
        if(itpr(k).gt.0)then
          itpr(k)=2             !this pair is diffractive
          if(id.gt.0.and.iep(ip).eq.0.and.iet(it).eq.0)itpr(k)=3  !this pair is empty now
        else
          itpr(k)=3             !this pair is empty now
        endif
      endif

      endif

      istring=idp1pr(n,k)+idp2pr(n,k)+idm1pr(n,k)+idm2pr(n,k)
      if(istring.ne.0.and.iremn.ge.2)then
        if(ish.ge.7)write(ifch,*)"restore flavor:",istring

        if(idp1pr(n,k).eq.2)ivp(ip)=ivp(ip)+1 !update number of valence quark
        if(idm1pr(n,k).eq.2)ivt(it)=ivt(it)+1
        if(idp2pr(n,k).eq.2)ivp(ip)=ivp(ip)+1
        if(idm2pr(n,k).eq.2)ivt(it)=ivt(it)+1
        if(idp1pr(n,k).eq.5)idp(ip)=idp(ip)+1 !update number of valence diquark
        if(idm1pr(n,k).eq.5)idt(it)=idt(it)+1
        if(idp2pr(n,k).eq.5)idp(ip)=idp(ip)+1
        if(idm2pr(n,k).eq.5)idt(it)=idt(it)+1
        if(iLHC.eq.1)then
        if(idp1pr(n,k).eq.4)idp(ip)=idp(ip)-1 !update number of diquark
        if(idm1pr(n,k).eq.4)idt(it)=idt(it)-1
        if(idp2pr(n,k).eq.4)idp(ip)=idp(ip)-1
        if(idm2pr(n,k).eq.4)idt(it)=idt(it)-1
        endif

        if(iremn.eq.3)then      !virtual Pomeron (remove unnecessary flavors for string ends)
          do j=1,2
            do i=1,nrflav
              jcp(i,j)=jcpref(i,j,ip)
              jct(i,j)=jctref(i,j,it)
            enddo
            do i=nrflav+1,nflav
              jcp(i,j)=0
              jct(i,j)=0
            enddo
          enddo
          if(ish.ge.7)write(ifch,*)"in:",jcp,' |',jct
          iret=0

c Projectile diquark-antidiquark pair
          iaq=nint(1.5+sign(0.5,float(idproj)))
          iq=3-iaq
          if(idp1pr(n,k).eq.4)then  !diquark
c    first quark
            idum=idrafl(iclpro,jcp,iaq,'v',0,iret)      !pick anti-quark
            ntry=0
            do while (jcp(idum,iq).eq.0.and.ntry.lt.100)!look for the corresponding quark
              ntry=ntry+1
              idum=idrafl(iclpro,jcp,iaq,'v',0,iret)
            enddo
            if(ntry.lt.100)then          !if OK, then remove the pair and pick a second quark
              call idsufl3(idum,1,jcp)
              call idsufl3(idum,2,jcp)
              if(jcp(idum,1)-jcpval(idum,1,ip).lt.0) !check valence quark number
     &             jcpval(idum,1,ip)=jcpval(idum,1,ip)-1
              if(jcp(idum,2)-jcpval(idum,2,ip).lt.0)
     &             jcpval(idum,2,ip)=jcpval(idum,2,ip)-1

c   second quark
              idum=idrafl(iclpro,jcp,iaq,'v',0,iret)
              ntry2=0
              do while (jcp(idum,iq).eq.0.and.ntry2.lt.100)!look for the corresponding antiquark
                ntry2=ntry2+1
                idum=idrafl(iclpro,jcp,iaq,'v',0,iret)
              enddo
              if(ntry2.lt.100)then          !if OK, then remove the pair
                call idsufl3(idum,1,jcp)
                call idsufl3(idum,2,jcp)
                if(jcp(idum,1)-jcpval(idum,1,ip).lt.0)
     &               jcpval(idum,1,ip)=jcpval(idum,1,ip)-1
                if(jcp(idum,2)-jcpval(idum,2,ip).lt.0)
     &               jcpval(idum,2,ip)=jcpval(idum,2,ip)-1
              else          !if not (because quarks already used by other valid string), then redo event to avoid problem in flavor conservation
                if(id.ge.15)then
                  id=-1
                  return
                else
                  call utstop("Virpom:should not happen (2) !&")
                endif
              endif
            else      !if no pair has be found (because quarks already used by other valid string), then redo event to avoid problem in flavor conservation
              if(id.ge.15)then
                id=-1
                return
              else
                call utstop("Virpom:should not happen  (3) !&")
              endif
            endif

c Projectile quark-antiquark pair
          else
            idum=idrafl(iclpro,jcp,iaq,'v',0,iret)      !pick anti-quark
            ntry=0
            do while (jcp(idum,iq).eq.0.and.ntry.lt.100)  !look for the corresponding quark
              ntry=ntry+1
              idum=idrafl(iclpro,jcp,iaq,'v',0,iret)
            enddo
            if(ntry.lt.100)then          !if OK, then remove the pair
              call idsufl3(idum,1,jcp)
              call idsufl3(idum,2,jcp)
              if(jcp(idum,1)-jcpval(idum,1,ip).lt.0)
     &             jcpval(idum,1,ip)=jcpval(idum,1,ip)-1
              if(jcp(idum,2)-jcpval(idum,2,ip).lt.0)
     &             jcpval(idum,2,ip)=jcpval(idum,2,ip)-1
            else                         !if not (because quarks already used by other valid string),then redo event to avoid problem in flavor conservation
              if(id.ge.15)then
                id=-1
                return
              else
                call utstop("Virpom:should not happen (4) !&")
              endif
            endif
          endif

c Target diquark-antidiquark pair
          iaq=nint(1.5+sign(0.5,float(idtarg)))
          iq=3-iaq
          if(idm1pr(n,k).eq.4)then  !diquark
c    first quark
            idum=idrafl(icltar,jct,iaq,'v',0,iret)
            ntry=0
            do while (jct(idum,iq).eq.0.and.ntry.lt.100)
              ntry=ntry+1
              idum=idrafl(icltar,jct,iaq,'v',0,iret)
            enddo
            if(ntry.lt.100)then
              call idsufl3(idum,1,jct)
              call idsufl3(idum,2,jct)
              if(jct(idum,1)-jctval(idum,1,it).lt.0)
     &             jctval(idum,1,it)=jctval(idum,1,it)-1
              if(jct(idum,2)-jctval(idum,2,it).lt.0)
     &             jctval(idum,2,it)=jctval(idum,2,it)-1
c    second quark
              idum=idrafl(icltar,jct,1,'v',0,iret)
              ntry2=0
              do while (jct(idum,2).eq.0.and.ntry2.lt.100)
                ntry2=ntry2+1
                idum=idrafl(icltar,jct,1,'v',0,iret)
              enddo
              if(ntry2.lt.100)then
                call idsufl3(idum,1,jct)
                call idsufl3(idum,2,jct)
                if(jct(idum,1)-jctval(idum,1,it).lt.0)
     &               jctval(idum,1,it)=jctval(idum,1,it)-1
                if(jct(idum,2)-jctval(idum,2,it).lt.0)
     &               jctval(idum,2,it)=jctval(idum,2,it)-1
              else
                if(id.ge.15)then
                  id=-1
                  return
                else
                  call utstop("Virpom:should not happen (5) !&")
                endif
              endif
            else
              if(id.ge.15)then
                id=-1
                return
              else
                call utstop("Virpom:should not happen (6) !&")
              endif
            endif

c Target quark-antiquark pair
          else
            idum=idrafl(icltar,jct,1,'v',0,iret)
            ntry=0
            do while (jct(idum,2).eq.0.and.ntry.lt.100)
              ntry=ntry+1
              idum=idrafl(icltar,jct,1,'v',0,iret)
            enddo
            if(ntry.lt.100)then
              call idsufl3(idum,1,jct)
              call idsufl3(idum,2,jct)
              if(jct(idum,1)-jctval(idum,1,it).lt.0)
     &             jctval(idum,1,it)=jctval(idum,1,it)-1
              if(jct(idum,2)-jctval(idum,2,it).lt.0)
     &             jctval(idum,2,it)=jctval(idum,2,it)-1
            else
              if(id.ge.15)then
                id=-1
                return
              else
                call utstop("Virpom:should not happen (7) !&")
              endif
            endif
          endif

          if(ish.ge.7)write(ifch,*)"out:",jcp,' |',jct
          do j=1,2
            do i=1,nrflav
              jcpref(i,j,ip)=jcp(i,j)
              jctref(i,j,it)=jct(i,j)
            enddo
          enddo

        endif
      endif


      ivpr(n,k)=0
      nbkpr(n,k)=0
      nvpr(n,k)=0
      idpr(n,k)=0
      idfpr(n,k)=0
      xpr(n,k)=0d0
      ypr(n,k)=0d0
      xppr(n,k)=0d0
      xmpr(n,k)=0d0
      idp1pr(n,k)=0
      idp2pr(n,k)=0
      idm1pr(n,k)=0
      idm2pr(n,k)=0
      xm1pr(n,k)=0d0
      xp1pr(n,k)=0d0
      xm2pr(n,k)=0d0
      xp2pr(n,k)=0d0
      xxm1pr(n,k)=0d0
      xym1pr(n,k)=0d0
      xxp1pr(n,k)=0d0
      xyp1pr(n,k)=0d0
      xxm2pr(n,k)=0d0
      xym2pr(n,k)=0d0
      xxp2pr(n,k)=0d0
      xyp2pr(n,k)=0d0

       if(ish.ge.5)write(ifch,*)"remnant out",xpp(ip),xmt(it),itpr(k)

      call utprix('VirPom',ish,ishini,3)

      end

c-----------------------------------------------------------------------
      subroutine StoRe(imod)
c-----------------------------------------------------------------------
c Store Remnant configuration (imod=1) before shuffle  to restore the
c initial configuration (imod=-1) in case of problem.
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      if(imod.eq.1)then

c       initialize projectile

        do i=1,maproj
          iepst(i)=iep(i)
          xppst(i)=xpp(i)
          xmpst(i)=xmp(i)
          xposst(i)=xpos(i)
        enddo

c       initialize target

        do j=1,matarg
          ietst(j)=iet(j)
          xmtst(j)=xmt(j)
          xptst(j)=xpt(j)
          xtosst(j)=xtos(j)
        enddo

      elseif(imod.eq.-1)then

c       restore projectile

        do i=1,maproj
          iep(i)=iepst(i)
          xpp(i)=xppst(i)
          xmp(i)=xmpst(i)
          xpos(i)=xposst(i)
        enddo

c       restore target

        do j=1,matarg
          iet(j)=ietst(j)
          xmt(j)=xmtst(j)
          xpt(j)=xptst(j)
          xtos(j)=xtosst(j)
        enddo

      else

        call utstop('Do not know what to do in StoRe.&')

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine UpdateFlav(ir,jc,mod)
c-----------------------------------------------------------------------
C Add valence quark to sea quarks in projectile jcpref (mod=10) or target
c jctref (mod=20) for soft string ends (mod=0 reset jcrpref and
c jctref to 0).
c For mod=1 or 2, save jcref into jc.
c For mod=-1 or -2, put jc into jcref.
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      dimension ic(2),jc(nflav,2),jc2(nflav,2)

      if(mod.eq.0)then
        do j=1,2
          do i=1,nrflav
            jcpref(i,j,ir)=0
            jctref(i,j,ir)=0
          enddo
        enddo
      elseif(mod.eq.-1)then
        do j=1,2
          do i=1,nrflav
            jcpref(i,j,ir)=jc(i,j)
          enddo
        enddo
      elseif(mod.eq.-2)then
        do j=1,2
          do i=1,nrflav
            jctref(i,j,ir)=jc(i,j)
          enddo
        enddo
      elseif(mod.eq.1)then
        do j=1,2
          do i=1,nrflav
            jc(i,j)=jcpref(i,j,ir)
          enddo
        enddo
      elseif(mod.eq.2)then
        do j=1,2
          do i=1,nrflav
            jc(i,j)=jctref(i,j,ir)
          enddo
        enddo
      elseif(mod.eq.10)then
        ic(1)=icproj(1,ir)
        ic(2)=icproj(2,ir)
        call iddeco(ic,jc)
        itest=0
        do j=1,2
          do i=1,nrflav
            jcpref(i,j,ir)=jcpref(i,j,ir)+jc(i,j)
          enddo
        enddo

c cancel quark and antiquarks to avoid to much remnant excitation
        do i=1,nrflav

          if(iLHC.eq.1)then

          if(jcpref(i,1,ir).ge.jcpref(i,2,ir))then
            jcpref(i,1,ir)=jcpref(i,1,ir)-jcpref(i,2,ir)
            jcpref(i,2,ir)=0
c update valence quarks (cancel first sea quarks)
            if(jcpref(i,1,ir)-jc(i,1).lt.0)jc(i,1)=jcpref(i,1,ir)
            jc(i,2)=0
          else
            jcpref(i,2,ir)=jcpref(i,2,ir)-jcpref(i,1,ir)
            jcpref(i,1,ir)=0
c update valence quarks (cancel first sea quarks)
            if(jcpref(i,2,ir)-jc(i,2).lt.0)jc(i,2)=jcpref(i,2,ir)
            jc(i,1)=0
          endif

          endif

          do j=1,2
            itest=itest+jcpref(i,j,ir)
            jc2(i,j)=jcpref(i,j,ir)
          enddo
        enddo
        if(itest.eq.0)then !do not leave empty remnant
          idum=idrafl(iclpro,jc2,1,'r',3,iretso)     !create q-qb
          do j=1,2
            do i=1,nrflav
              jcpref(i,j,ir)=jc2(i,j)
            enddo
          enddo
        endif
      if(ish.ge.6)write(ifch,'(a,i3,a,1x,4i3,3x,4i3)')
     & 'jcpref(',ir,') ini:',((jcpref(i,j,ir),i=1,nflavems),j=1,2)
      elseif(mod.eq.20)then
        ic(1)=ictarg(1,ir)
        ic(2)=ictarg(2,ir)
        call iddeco(ic,jc)
        itest=0
        do j=1,2
          do i=1,nrflav
            jctref(i,j,ir)=jctref(i,j,ir)+jc(i,j)
          enddo
        enddo

        do i=1,nrflav

          if(iLHC.eq.1)then

c cancel quark and antiquarks to avoid to much remnant excitation
          if(jctref(i,1,ir).ge.jctref(i,2,ir))then
            jctref(i,1,ir)=jctref(i,1,ir)-jctref(i,2,ir)
            jctref(i,2,ir)=0
c update valence quarks (cancel first sea quarks)
            if(jctref(i,1,ir)-jc(i,1).lt.0)jc(i,1)=jctref(i,1,ir)
            jc(i,2)=0
          else
            jctref(i,2,ir)=jctref(i,2,ir)-jctref(i,1,ir)
            jctref(i,1,ir)=0
c update valence quarks (cancel first sea quarks)
            if(jctref(i,2,ir)-jc(i,2).lt.0)jc(i,2)=jctref(i,2,ir)
            jc(i,1)=0
          endif
          
          endif

          do j=1,2
            itest=itest+jctref(i,j,ir)
            jc2(i,j)=jctref(i,j,ir)
          enddo
        enddo
        if(itest.eq.0)then !do not leave empty remnant
          idum=idrafl(icltar,jc2,1,'r',3,iretso)     !create q-qb
          do j=1,2
            do i=1,nrflav
              jctref(i,j,ir)=jc2(i,j)
            enddo
          enddo
        endif
      if(ish.ge.6)write(ifch,'(a,i3,a,1x,4i3,3x,4i3)')
     & 'jctref(',ir,') ini:',((jctref(i,j,ir),i=1,nflavems),j=1,2)
      else
        stop'mod not recognized in UpdateFlav'
      endif
      end

c-----------------------------------------------------------------------
      subroutine CalcZZ(ir,m)
c-----------------------------------------------------------------------
C Calculates zz for remnant m for proj (ir=1) or target (ir=-1)
c   writes it to zzremn(m, 1 or 2)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incpar'
      if(isplit.eq.1)then
        if(ir.eq.1)then
          zz=0.
          if(lproj3(m).ge.1)then
            do l=1,lproj3(m)
              kpair=kproj3(m,l)
              zpar=zparpro(kpair)
              zz=zz+min(zpar,epscrx)
             enddo
          endif
          zzremn(m,1)=zz
       elseif(ir.eq.-1)then
          zz=0
          if(ltarg3(m).ge.1)then
            do l=1,ltarg3(m)
              kpair=ktarg3(m,l)
              zpar=zpartar(kpair)
              zz=zz+min(zpar,epscrx)
            enddo
          endif
          zzremn(m,2)=zz
        else
          stop'CalcZZ: invalid option.          '
        endif
      else
        if(ir.eq.1) zzremn(m,1)=0
        if(ir.eq.-1)zzremn(m,2)=0
      endif
      end

c-----------------------------------------------------------------------
      subroutine WriteZZ(ir,irem)
c-----------------------------------------------------------------------
c Write Z into zpaptl(K) for connected strings
c                 K is the index for the string end
c                 on the corresponding remnant side
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      common/cems5/plc,s
      double precision s,plc

      if(ir.eq.1)then
        jrem=1
      elseif(ir.eq.-1)then
        jrem=2
      else
        jrem=0
        call utstop("Wrong ir in WriteZZ !&")
      endif

      do li=1,lremn(irem,jrem)
        kkk=kremn(irem,li,jrem)
c        ip=iproj(kkk)
c        it=itarg(kkk)
         amtot=0.
         do n=1,nprmx(kkk)
           if(idpr(n,kkk).ne.0)amtot=amtot+sngl(xpr(n,kkk)*s)
         enddo
         amtot=sqrt(amtot)
         do n=1,nprmx(kkk)
          if(idpr(n,kkk).ne.0)then
           npom=nppr(n,kkk)
c              write(ifch,*)'remn',irem,' (',jrem,' )     pom',npom
c     &            ,'    ',zzremn(irem,jrem)
           ie=0
           is1=0
           if(ifrptl(1,npom).gt.0)then
            do is=ifrptl(1,npom),ifrptl(2,npom)
              if(ie.eq.0)is1=is
              if(idptl(is).ne.9)ie=ie+1
              if(ie.eq.2)then
               is2=is
               ie=0
               if(ir.eq. 1)then
c  Z for remnant ip (low if alone and high in nucleus)
                 zpaptl(1,is1)=zzremn(irem,jrem)
c  sum of Z of remnant itt linked to ip (high if connected to many other remn)
                 zpaptl(2,is1)=amtot !float(nprt(kkk)) !float(lproj(ip))
c                 zpaptl(2,is1)=0.
c                 if(lproj(ip).ge.1)then
c                   do l=1,lproj(ip)
c                     kpair=kproj(ip,l)
c                     itt=itarg(kpair)
c                     zpaptl(2,is1)=zpaptl(2,is1)+zzremn(itt,2)
c                   enddo
c                 endif
               endif
               if(ir.eq.-1)then
c  Z for remnant it (low if alone and high in nucleus)
                 zpaptl(1,is2)=zzremn(irem,jrem)
c  sum of Z of remnant ipp linked to it (high if connected to many other remn)
                 zpaptl(2,is2)=float(nprt(kkk)) !float(ltarg(it))
c                 zpaptl(2,is2)=0.
c                 if(ltarg(it).ge.1)then
c                   do l=1,ltarg(it)
c                     kpair=ktarg(it,l)
c                     ipp=iproj(kpair)
c                     zpaptl(2,is2)=zpaptl(2,is2)+zzremn(ipp,1)
c                   enddo
c                 endif
               endif
c               do isi=is1,is2
c                write(ifch,*)' ',isi,idptl(isi),zpaptl(1,isi),zpaptl(2,isi)
c               enddo
              endif
            enddo
           endif
          endif
        enddo
      enddo

      end

c-----------------------------------------------------------------------
      subroutine ProReM(ir,irem,iret)
c-----------------------------------------------------------------------
c propose remnant mass of remnant irem in case of proj (ir=1)
c or target (ir=-1)
c   (-> xmp, xpt)
c iret : input : if iret=10 force to give mass even if no more energy,
c        when input not 10 : output = error if 1
c Energy is taken only from the other side nucleon which are close enough
c to form a pair even if that pair was not used for a collision.
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      double precision rr,xxx,xmin,xmax,msmin,xmmin,xpt2rem,xtest0,xtmp
      double precision at,alp,xi,xii,eps,sx,xmin0,xtest(mamx),fxtest
      parameter(eps=1.d-20)
      common/cemsr5/at(0:1,0:5)
      double precision plc,s,p5sq,aremn,aremnex,xxmax,drangen!,xmdrmax
      common/cems5/plc,s
      integer icrmn(2),jc(nflav,2)
      logical cont,force,drop,excited
      character cremn*4
      dimension k2j(mamx)

      call utpri('ProReM',ish,ishini,5)

      if(iret.eq.10)then
        force=.true.
      else
        iret=0
        force=.false.
      endif
      ntrymx=50
      do j=1,2
        do i=1,nflav
          jc(i,j)=0
        enddo
      enddo

c uncomment the following two lines to force the excitation

ccc      force=.true.   !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ccc      ntrymx=1       !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

c initial definitions

      ntry=0
      iremo1=0
      jremo=0
c... initialize
      jrem=0.0
      amremn=0.0
      if(ir.eq.1)then
        cremn='targ'
        jrem=1
        jremo=2
        masso=lproj(irem)     !number of target nucleon linked to irem
        do k=1,masso
          k2j(k)=itarg(kproj(irem,k))
          xme(k2j(k))=0.d0
        enddo
        icrmn(1)=icremn(1,irem,jrem)
        if(icrmn(1).eq.999999)then    !more than 9 quark : use jcpref
          do j=1,2
            do i=1,nrflav
              jc(i,j)=jcpref(i,j,irem)
            enddo
          enddo
        else
          icrmn(2)=icremn(2,irem,jrem)
          call iddeco(icrmn,jc)
        endif
        amremn=amproj
           !idx=isign(iabs(idproj)/10*10+1,idproj)
           !call idmass(idx,amremn)
        iremo1=itarg(1)
        msmin=dble(amremn*amremn)
        zz=1.
        if(iez(irem,jrem).eq.3.or.iez(irem,jrem).eq.5)
     &  zz=zz+zzremn(irem,1)*zmsinc
      elseif(ir.eq.-1)then
        cremn='proj'
        jrem=2
        jremo=1
        masso=ltarg(irem)  !number of projectile nucleon linked to irem
        do k=1,masso
          k2j(k)=iproj(ktarg(irem,k))
          xme(k2j(k))=0.d0
        enddo
        icrmn(1)=icremn(1,irem,jrem)
        if(icrmn(1).eq.999999)then    !more than 9 quark : use jctref
          do j=1,2
            do i=1,nrflav
              jc(i,j)=jctref(i,j,irem)
            enddo
          enddo
        else
          icrmn(2)=icremn(2,irem,jrem)
          call iddeco(icrmn,jc)
        endif
        amremn=amtarg
           !idx=isign(iabs(idtarg)/10*10+1,idtarg)
           !call idmass(idx,amremn)
        iremo1=iproj(1)
        msmin=dble(amremn*amremn)
        zz=1.
        if(iez(irem,jrem).eq.3.or.iez(irem,jrem).eq.5)
     &  zz=zz+zzremn(irem,2)*zmsinc
      endif
      drop=.false.
      if(iremn.ge.2.and.(iez(irem,jrem).eq.3.or.iez(irem,jrem).eq.5))
     &   drop=.true.
      excited=.false.
      if(iez(irem,jrem).gt.0.and.iez(irem,jrem).ne.6)
     &   excited=.true.

c for spectators only low mass and few partners, so do not care about energy
      if(iez(irem,jrem).eq.6)force=.true.

c defs

      sx=s*xpz(irem,jrem)
      xpt2rem=xxz(irem,jrem)**2d0+xyz(irem,jrem)**2d0

c  fremnux (+) and not fremnux2 (-) which gives a mass too low in case of gethadron where q and aq do not cancel


      if(excited)then
        aremn=dble(max(amremn,fremnux(jc)))
c       if(iremn.eq.2.and.iez(irem,jrem).eq.3)      !droplet
c     &     aremn=dble(max(amremn,fremnux(jc)))
        if(iremn.ge.2)then
          aremnex=aremn+amemn(idz(irem,jrem),iez(irem,jrem))
c         if(drop)aremnex=aremnex*zz
        else
          aremnex=aremn+amemn(idz(irem,jrem),iez(irem,jrem))
        endif
      elseif(iLHC.eq.1)then !minimum mass for spectators should be as low as possible
        aremn=amremn
        aremnex=dble(max(amremn,fremnux2(jc)))
      else    !minimum mass for spectators should be as low as possible
        aremn=dble(max(amremn,fremnux2(jc)))
        aremnex=aremn
      endif


      if(ish.ge.8)write(ifch,10)ir,irem,masso,icrmn,iez(irem,jrem),force
     &                         ,amremn,fremnux(jc),aremn,aremnex
     &                         ,xpz(irem,jrem),xpt2rem,sx
 10   format('prorem :  ',i3,2i4,2i7,i2,L2,/
     &      ,'    mass :',4g13.5,/
     &      ,' x,pt,sx :',3g13.5)
c ntry

    1 ntry=ntry+1
      if(ntry.gt.ntrymx)then
        if(ish.ge.5)then
          call utmsg('ProReM')
          write(ifch,*)'Remnant mass assignment not possible (ntry)'
     &                 ,ir,irem
          if(force)write(ifch,*)'Ignore p4 conservation'
          call utmsgf
        endif
        if(.not.force)then
          iret=1
          goto 1000
        else
c not enough energy availabe : force last mass and check
          goto 900
        endif
      endif

c check

      if(xpz(irem,jrem).le.0.d0)then
        write(ifch,*)'ProRem ipp',xpz(irem,jrem)
     &                           ,jrem,irem,lremn(irem,jrem)
        do li=1,lremn(irem,jrem)
          kkk=kremn(irem,li,jrem)
          write(ifch,*)'kkk',kkk
        enddo
        call XPrint('ProRem :&')
        call utstop('Big problem in ProRem !&')
      endif

c xtest = xminus-max,  corresponding mostly to a remnant mass 0.2

      xtest0=0.d0
      fxtest=0.4d0*(1d0+drangen(xxx)) !1.d0 !0.3d0
      do k=1,masso
        j=k2j(k)
        cont=.false.
ctp        if(xmz(j,jremo).gt.eps.and.iez(j,jrem).gt.0)then !xmz(,jremo)=xplus
ctp060824        if(xmz(j,jremo).gt.eps.and.iez(j,jrem).ge.0)then !xmz(,jremo)=xplus
c        if(iez(j,jremo).gt.0.or.koll.eq.1)then !xmz(,jremo)=xplus
          if(xmz(j,jremo).gt.eps)then !xmz(,jremo)=xplus
            cont=.true.
            xmmin=xzos(j,jremo)/xmz(j,jremo)
          else
            xmmin=xzos(j,jremo)
          endif
          xtest(j)=xpz(j,jremo)-xmmin !maximal momentum available
!this term is very important for non excited remnants in pp, it changes the xf
! distribution of proton and the multiplicity at low energy. Fxtest should not
! be to close to 0. otherwise it makes a step in xf distribution of p at
! 1-fxtest but if fxtest=1, multiplicity at low energy is too high ...
! but better (and smoother) with exponential decrease).
          if(.not.cont)then
            if(xtest(j).gt.0d0)then
              xtest(j)=min(xtest(j),fxtest/xpz(irem,jrem))
            else
              xtest(j)=min(1.d0,fxtest/xpz(irem,jrem))
            endif
          endif
c        else
c          xtest(j)=0.01d0 !maximal momentum available for non exited state
c        endif
         xtest0=max(xtest0,xtest(j))
c        print *,iep(1),iet(1),iez(irem,jrem),xtest(j),xpz(j,jremo),xmmin
c     & ,xzos(j,jremo),xmz(j,jremo)
      enddo
ctp060824      if(.not.cont)xtest=min(1.d0,0.2d0/xpz(irem,jrem))



c determine xminus

c      xmin0=1.05*(aremn**2d0+xxz(irem,jrem)**2d0+xyz(irem,jrem)**2d0)/sx
c      xmin=1.1*(aremnex**2d0+xxz(irem,jrem)**2d0+xyz(irem,jrem)**2d0)/sx
      xmin0=1.1d0*(aremn**2d0+xpt2rem)/sx
      if(iLHC.eq.1.and.xmin0.ge.1.d0)xmin0=min(xmin0,0.9d0)
      if(ish.ge.1.and.xmin0.ge.1d0)
     &   write(ifch,*)"Warning in ProReM with xmin0 !"
      
      if(iez(irem,jrem).eq.4)then !pion exchange, minim should not change
        xmin=dble(xmindiff)*(aremnex**2d0+xpt2rem)/sx
      else
        xmin=dble(xminremn)*(aremnex**2d0+xpt2rem)/sx
      endif
c      xmax=min(1.d6/s,xtest0)             !to avoid ultra high mass remnants
      xmax=xtest0
c for diffractive remnant, mass should never exceed 5% of the proj or targ energy
c      if(iez(irem,jrem).eq.1)then
c        xmax=min(xmax,max(dble(xminremn),xmin))
c      elseif(iez(irem,jrem).eq.2)then
c        xmax=min(xmax,max(dble(xmindiff),xmin))
c      endif
c      if(iez(irem,jrem).eq.1.or.iez(irem,jrem).eq.3)then
c       xtmp=max(dble(min(1.,xminremn*float(maproj+matarg-1))),xmin)
c     &               *drangen(xmin)
      xtmp=1.d0
      if(excited)then
      if(iez(irem,jrem).eq.2)then
c        xtmp=max(min(1d0,dble(xmindiff)),xmin)!*drangen(xmin)
        xtmp=min(1d0,dble(xmxrem)*dble(masso)
     &                      *drangen(xmin)**0.05)
c        xtmp=dble(xmindiff)
      elseif(iez(irem,jrem).eq.1)then
        xtmp=min(1d0,dble(xmxrem)*dble(masso)
     &                      *drangen(xmin)**0.05)
c        xtmp=dble(xminremn)
      elseif(drop)then     !3 or 5
c       xtmp=max(dble(min(1.,xminremn*float(maproj+matarg-1))),xmin)
c     &               *drangen(xmin)
        xtmp=min(1d0,dble(xmxrem)*zz*dble(masso)
     &                         *drangen(xmin)**0.05)
c        xtmp=dble(xminremn)
      endif
      endif
      xmax=min(xmax,max(xtmp,xmin))
      if(ish.ge.8)write(ifch,*)'ntry',ntry,xmin,xmax,xtmp
     *                               ,xmax*dble(masso),xmin0,excited
      if(koll.eq.1)xmax=min(xmax,xpz(iremo1,jremo))
      xxmax=xmax*dble(masso)-eps
      if(iLHC.eq.1)xxmax=min(1d0-eps,xxmax)      !check energy limit
      if(xmin.ge.xxmax)then
        xmax=xxmax
        xmin=xmin0
        if(xmin0.ge.xmax-eps)then
          if(.not.force)then
            iret=1
          elseif(excited)then
            xmz(irem,jrem)=min(1.-xpz(irem,jrem),
     &                       xmin0+0.5d0*(1d0-xmin0)*drangen(xmin)) !random not to form a peak
          else
            xxx=(aremn**2d0+xpt2rem)/sx
            xmz(irem,jrem)=xxx
c            xmin0=max(0.5d0,(1d0-((amzmn(idz(irem,jremo),jremo)
c     &                   +6d0*drangen(xxx))**2+xpt2rem)/sx))*xxx
            if(iLHC.eq.1)then     !LHC tune (more reasonnable xsi distribution)
c          xmin0=(1d0-xmin**0.3)*xxx
              xmin0=max(0.35d0*(1d0+drangen(xxx))
     &          ,1d0-((amzmn(idz(irem,jremo),jremo)
     &             +engy**drangen(xxx))**2+xpt2rem)/sx)*xxx
            else   !original CR version
              xmin0=max(0.35d0*(1d0+drangen(xxx))
     &          ,1d0-((amzmn(idz(irem,jremo),jremo)
     &             +sqrt(engy)*drangen(xxx)**0.5)**2+xpt2rem)/sx)*xxx
            endif
          endif
          goto 1000
        endif
      elseif(xmin.ge.xmax)then
        xmax=1d0
      endif
      rr=dble(rangen())
      alp=0.
      xxx=0.d0
      if(excited)then
c        xmin=xmin-xpt2rem/sx                     !no pt
c        xmax=xmax-xpt2rem/sx                     !no pt
        alp=at(idz(irem,jrem),iez(irem,jrem))/dble(zz)

        if(dabs(alp-1.d0).lt.eps)then
          xxx=xmax**rr*xmin**(1d0-rr)
        else
          xxx=(rr*xmax**(1d0-alp)+(1d0-rr)*xmin**(1d0-alp))
     &                                             **(1d0/(1d0-alp))
        endif
c        xxx=xxx+xpt2rem/sx                       !no pt
!smooth distribution
        if(iez(irem,jrem).eq.4)xmin=xmin0
        xmin0=xmin+(1d0-exp(-2d0*drangen(xxx)**2))*(xxx-xmin)
      else
        if(masso.eq.1)ntry=ntrymx   !xxx is fixed so 1 try is enough
c        xmin=dble(amremn)**2d0/sx                !no pt
c        xxx=xmin+xpt2rem/sx                      !no pt
        xmin=(dble(aremn)**2d0+xpt2rem)/sx
        xxx=xmin
        if(xmin.gt.xmax+eps)then
          if(ish.ge.6)write(ifch,*)'xmin>xmax for proj not possible (2)'
     &                 ,ir,irem
          if(.not.force)then
            iret=1
          else
            xmz(irem,jrem)=xxx
          endif
          goto 1000
        endif
c to have a nice diffractive peak, do not allow too much fluctuation
c this function is more or less a fit of the diffractive peak
c (pp100, ep-forward (ZEUS), NA49, pipp100, taking into account the 
c contribution of inelastic remnants)
          if(iLHC.eq.1)then     !LHC tune (more reasonnable xsi distribution)
c          xmin0=(1d0-xmin**0.3)*xxx
            xmin0=max(0.35d0*(1d0+drangen(xxx))
     &          ,1d0-((amzmn(idz(irem,jremo),jremo)
     &             +engy**drangen(xxx))**2+xpt2rem)/sx)*xxx
          else   !original CR version
            xmin0=max(0.35d0*(1d0+drangen(xxx))
     &          ,1d0-((amzmn(idz(irem,jremo),jremo)
     &             +sqrt(engy)*drangen(xxx)**0.5)**2+xpt2rem)/sx)*xxx
          endif

c       write(*,*)'->',xmin0/xxx,sx,log10(1d0-xmin0/xxx)
c     &,1d0-((amzmn(idz(irem,jremo),jremo)
c     &+5d0*exp(-0.5d0*drangen(xxx)**2))**2+xpt2rem)/sx
c        xmin0=dble(0.9+0.09*rangen())*xxx
      endif
      if(ish.ge.8)write(ifch,*)'alp',alp,xmin,xxx,xmax,zz
      msmin=xmin*sx
c      msmin=xmin*sx+xpt2rem                      !no pt

c partition xminus between nucleons of the other side

      xii=1d0
      ii=masso
      kk=int(rangen()*float(ii))+1   ! choose ramdomly a nucleon to start

      do while(ii.gt.0)

        iro=k2j(kk)
        cont=iez(iro,jremo).lt.0.or.xme(iro).lt.-0.99d0
        do while(cont)
          kk=kk+1
          if(kk.gt.masso)kk=kk-masso
          iro=k2j(kk)
          ii=ii-1
          if(ii.lt.1)then
            ntry=ntrymx
            goto 1
          endif
          cont=iez(iro,jremo).lt.0.or.xme(iro).lt.-0.99d0
        enddo

        if(ii-1.gt.0)then
         xi=xii*dble(rangen())**(1.d0/dble(ii-1))
        else
         xi=0d0
        endif
        xme(iro)=xxx*(xii-xi)

        xmmin=xzos(iro,jremo)
        if(xmz(iro,jremo).gt.eps)then
          xmmin=xmmin/xmz(iro,jremo)
        elseif(koll.eq.1.and.xtest(iro).gt.eps)then
          xmmin=xmmin/min(xpz(irem,jrem),xtest(iro))
        elseif(xtest(iro).gt.eps)then
          xmmin=xmmin/xtest(iro)
        endif
        if((xpz(iro,jremo)-xme(iro)).lt.xmmin)then
          if(ish.ge.8)write(ifch,*)'     skip ',cremn,' ',ii,masso,ntry
     &                      ,iro,xme(iro),xpz(iro,jremo)-xme(iro),xmmin
          xme(iro)=-1.d0
          if(ii.le.1)goto 1
        else
          xii=xi
          if(ish.ge.8)write(ifch,*)'       ok ',cremn,' ',ii,masso,ntry
     &                      ,iro,xme(iro),xme(iro)/xxx
        endif
        kk=kk+1
        if(kk.gt.masso)kk=kk-masso
        ii=ii-1

      enddo

c check xmz(irem,jrem)

 900  xmz(irem,jrem)=xxx

      p5sq=xpz(irem,jrem)*xmz(irem,jrem)*s
      if(ish.ge.8)write(ifch,*)'final mass',irem,p5sq,msmin
     &,xpz(irem,jrem),xmz(irem,jrem),force
      if(p5sq-msmin.lt.-1d-10)then
        if(ish.ge.5)then
          call utmsg('ProReM')
          write(ifch,*)'Remnant mass assignment not possible (M<Mmin)!'
     &                 ,ir,irem
          if(force)write(ifch,*)'Ignore p4 conservation'
          call utmsgf
        endif
        if(.not.force)then
          iret=1
        elseif(xpz(irem,jrem).gt.0.d0)then
          xmz(irem,jrem)=min(1.-xpz(irem,jrem),
     &                       xmin+0.5d0*(1d0-xmin)*drangen(xmin))   !random not to form a peak
        endif
        goto 1000
      endif

c subtract xme

      do k=1,masso
        iro=k2j(k)
        if(xme(iro).gt.0.d0)then
          xpz(iro,jremo)=xpz(iro,jremo)-xme(iro)  !xpz(,jremo)=xminus
        endif
      enddo

 1000 continue
      if(iret.ne.1)xzos(irem,jrem)=xmin0*xpz(irem,jrem)

      call utprix('ProReM',ish,ishini,5)

      end

c-----------------------------------------------------------------------
      subroutine ProSeTy(k,n)
c-----------------------------------------------------------------------
c creates proposal for string ends, idp., idm.
c updates quark counters
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      common/ems6/ivp0,iap0,idp0,isp0,ivt0,iat0,idt0,ist0
      double precision pes,xfqp,xfqt   !so01
      parameter(eps=1.e-6)
      common/ems9/xfqp(0:9),xfqt(0:9)
      common/emsx3/pes(0:3,0:6)
      integer jcp(nflav,2),jct(nflav,2)
     &       ,jcpi(nflavems,2),jcti(nflavems,2)
      logical go

      if(idpr(n,k).eq.2)stop'no Reggeons any more'

      iret=0
      ip=iproj(k)
      it=itarg(k)
      if(iremn.ge.3)then
        do j=1,2
          do i=1,nrflav
            jcp(i,j)=jcpref(i,j,ip)
            jct(i,j)=jctref(i,j,it)
          enddo
          do i=nrflav+1,nflav
            jcp(i,j)=0
            jct(i,j)=0
          enddo
        enddo
      endif
      
      idp1pr(n,k)=0
      idm1pr(n,k)=0
      idp2pr(n,k)=0
      idm2pr(n,k)=0
      idsppr(n,k)=0
      idstpr(n,k)=0
      pssp=0.
      pvsp=0.
      pvap=0.
      pddp=0.
      psvvp=0.
      paasp=0.
      psst=0.
      pvst=0.
      pvat=0.
      pddt=0.
      psvvt=0.
      paast=0.

      if(iLHC.eq.1)then

c for hard Pomeron, define which string ends are connected to valence quark
c treat gluon has soft string ends (including diquarks but can not be
c a "soft" valence like in soft Pomerons) later
      if(idpr(n,k).eq.3)then
        go=.false.
        if(ivp0.eq.iap0.and.rangen().lt.0.5)go=.true.    !meson
        idsppr(n,k)=5
        if(idhpr(n,k).eq.3.or.idhpr(n,k).eq.1)then
          if(iremn.ge.2)ivp(ip)=ivp(ip)-1
          if(iap0.eq.0.or.go)then !baryon
            idp1pr(n,k)=2
          else                    !antibaryon
            idp2pr(n,k)=2
          endif
        endif
        idstpr(n,k)=5
        if(idhpr(n,k).eq.3.or.idhpr(n,k).eq.2)then
          if(iremn.ge.2)ivt(it)=ivp(it)-1
          if(iat0.eq.0)then     !baryon
            idm1pr(n,k)=2
          else                  !antibaryon
            idm2pr(n,k)=2
          endif
        endif
      endif

      if(idpr(n,k).ne.0)then

c    projectile

       if(idfpr(n,k).eq.1.or.idfpr(n,k).eq.2)then

       ntry=0
       ivpi=ivp(ip)
       idpi=idp(ip)
       idspi=idsppr(n,k)
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcpi(i,j)=jcp(i,j)
           enddo
         enddo
       endif
  1    ntry=ntry+1
      if(ntry.gt.10)call utstop('something goes wrong in sr ProSeTy&')
       ivp(ip)=ivpi
       idp(ip)=idpi
       idsppr(n,k)=idspi
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcp(i,j)=jcpi(i,j)
           enddo
         enddo
       endif
       pss=wgtval+wgtsea
       if(pss.gt.0.)then
         pss=wgtsea/pss
       else
         pss=0.
       endif
       if(iremn.ge.2)then
         if(iap0.eq.0)then
           pvs=0.
           if(ivp(ip).ne.0.and.idpr(n,k).ne.3)pvs=1.-pss
           pva=0.
           psvv=0.
           if(idp(ip).ne.0.and.idp2pr(n,k).ne.2)psvv=wgtqqq(iclpro)
           paas=0.
         elseif(ivp0.eq.0)then
           pva=0.
           if(ivp(ip).ne.0.and.idpr(n,k).ne.3)pva=1.-pss
           pvs=0.
           psvv=0.
           paas=0.
           if(idp(ip).ne.0.and.idp1pr(n,k).ne.2)paas=wgtqqq(iclpro)
         else                   !for meson, no soft string with valence quark (we do not know whether the quark or the antiquark will be used by hard string)
           pvs=0.
           pva=0.
c diquark or antidiquark can be created once in meson remnant
           psvv=0.
           paas=0.
           if(1+idp(ip).ne.0)then
             if(idp2pr(n,k).ne.2)psvv=wgtqqq(iclpro)
             if(idp1pr(n,k).ne.2)paas=wgtqqq(iclpro)
           endif
         endif
         pdd=wgtdiq/(1.+float(abs(idp(ip))))
c         if(idpr(n,k).eq.3)then
c           pdd=0.
c           psvv=0.
c           paas=0.
c         endif
       elseif(iremn.ne.0)then
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         if(idp2pr(n,k).ne.2)psvv=wgtqqq(iclpro)
         if(idp1pr(n,k).ne.2)paas=wgtqqq(iclpro)
         pdd=wgtdiq/(1.+float(abs(idp(ip))))
       else
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         pdd=wgtdiq/(1.+float(abs(idp(ip))))
       endif
       if(idp1pr(n,k).eq.2)then  !with valence quark only 1 SE available
         psd=pdd
         pds=0.
         pdd=0.
       elseif(idp2pr(n,k).eq.2)then  !with valence antiquark only 1 SE available
         pds=pdd
         psd=0.
         pdd=0.
       else
         psd=pdd
         pds=pdd
         pdd=pdd**2
       endif
       su=1.-min(1.,pdd+psd+pds)            !diquark probability
       pss=(1.-min(1.,pvs+pva))*su        !no more valence quark: take from sea
       pvs=pvs*su
       pva=pva*su
       su=1.-min(1.,psvv+paas)      !stopping probability
       pss=pss*su
       pvs=pvs*su
       pva=pva*su
       psd=psd*su
       pds=pds*su
       pdd=pdd*su
       su=pss+pvs+pva+pdd+psd+pds+psvv+paas
       pssp = pss /su
       pvsp = pvs /su
       pvap = pva /su
       psdp = psd /su
       pdsp = pds /su
       pddp = pdd /su
       psvvp= psvv/su
       paasp= paas/su
       r=rangen()
       if(r.gt.(pssp+pvsp+pvap+psdp+pdsp+psvvp+paasp)
     &                               .and.pddp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=4
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=4
        idsppr(n,k)=idsppr(n,k)+4
        if(iremn.ge.2)idp(ip)=idp(ip)+2
        if(iremn.eq.3)then   !add diquark flavor to jcpref for ProSeF later (sea quark)
          idum=idrafl(iclpro,jcp,1,'s',3,iret)
          idum=idrafl(iclpro,jcp,1,'d',3,iret)
          idum=idrafl(iclpro,jcp,1,'s',3,iret)
          idum=idrafl(iclpro,jcp,1,'d',3,iret)
        endif
      elseif(r.gt.(pssp+pvsp+pvap+psdp+psvvp+paasp).and.pdsp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=4
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=1
        idsppr(n,k)=idsppr(n,k)+4
        if(iremn.ge.2)idp(ip)=idp(ip)+1
        if(iremn.eq.3)then   !add diquark flavor to jcpref for ProSeF later (sea quark)
          idum=idrafl(iclpro,jcp,1,'s',3,iret)
          idum=idrafl(iclpro,jcp,1,'d',3,iret)
        endif
       elseif(r.gt.(pssp+pvsp+pvap+psvvp+paasp).and.psdp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=1
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=4
        idsppr(n,k)=idsppr(n,k)+4
        if(iremn.ge.2)idp(ip)=idp(ip)+1
        if(iremn.eq.3)then   !add diquark flavor to jcpref for ProSeF later (sea quark)
          idum=idrafl(iclpro,jcp,1,'s',3,iret)
          idum=idrafl(iclpro,jcp,1,'d',3,iret)
        endif
       elseif(r.gt.(pssp+pvsp+pvap+psvvp).and.paasp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=5
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=1
        idsppr(n,k)=idsppr(n,k)+5
        if(iremn.ge.2)idp(ip)=idp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(pssp+pvsp+pvap+pddp).and.psvvp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=1
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=5
        idsppr(n,k)=idsppr(n,k)+5
        if(iremn.ge.2)idp(ip)=idp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(pssp+pvsp).and.pvap.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=1
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=2
        idsppr(n,k)=idsppr(n,k)+2
        if(iremn.ge.2)ivp(ip)=ivp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       elseif(r.gt.pssp.and.pvsp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=2
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=1
        idsppr(n,k)=idsppr(n,k)+2
        if(iremn.ge.2)ivp(ip)=ivp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,2,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       elseif(pssp.gt.eps)then
        if(idp1pr(n,k).ne.2)idp1pr(n,k)=1
        if(idp2pr(n,k).ne.2)idp2pr(n,k)=1
        idsppr(n,k)=idsppr(n,k)+1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       else
        goto 1
       endif

       else
        idp1pr(n,k)=1
        idp2pr(n,k)=1
        idsppr(n,k)=0
       endif


c    target

       if(idfpr(n,k).eq.1.or.idfpr(n,k).eq.3)then


       ntry=0
       ivti=ivt(it)
       idti=idt(it)
       idsti=idstpr(n,k)
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcti(i,j)=jct(i,j)
           enddo
         enddo
       endif
  2    ntry=ntry+1
       if(ntry.gt.10)call utstop('something goes wrong in sr ProSeTy&')
       ivt(it)=ivti
       idt(it)=idti
       idstpr(n,k)=idsti
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jct(i,j)=jcti(i,j)
           enddo
         enddo
       endif
       pss=wgtval+wgtsea
       if(pss.gt.0.)then
         pss=wgtsea/pss
       else
         pss=0.
       endif
       if(iremn.ge.2)then
         if(iat0.eq.0)then
           pvs=0.
           if(ivt(it).ne.0.and.idpr(n,k).ne.3)pvs=1.-pss
           pva=0.
           psvv=0.
           if(idt(it).ne.0.and.idm2pr(n,k).ne.2)psvv=wgtqqq(icltar)
           paas=0.
         elseif(ivt0.eq.0)then
           pva=0.
           if(ivt(it).ne.0.and.idpr(n,k).ne.3)pva=1.-pss
           pvs=0.
           psvv=0.
           paas=0.
           if(idt(it).ne.0.and.idm1pr(n,k).ne.2)paas=wgtqqq(icltar)
         else                   !for meson, no soft string with valence quark (we do not know whether the quark or the antiquark will be used by hard string)
           pvs=0.
           pva=0.
c diquark or antidiquark can be created once in meson remnant
           psvv=0.
           paas=0.
           if(1+idt(it).ne.0)then
             if(idm2pr(n,k).ne.2)psvv=wgtqqq(icltar)
             if(idm1pr(n,k).ne.2)paas=wgtqqq(icltar)
           endif
         endif
         pdd=wgtdiq/(1.+float(abs(idt(it))))
c         if(idpr(n,k).eq.3)then
c           pdd=0.
c           psvv=0.
c           paas=0.
c         endif
       elseif(iremn.ne.0)then
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         if(idm2pr(n,k).ne.2)psvv=wgtqqq(icltar)
         if(idm1pr(n,k).ne.2)paas=wgtqqq(icltar)
         pdd=wgtdiq/(1.+float(abs(idt(it))))
       else
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         pdd=wgtdiq/(1.+float(abs(idt(it))))
       endif
       if(idm1pr(n,k).eq.2)then  !with valence quark only 1 SE available
         psd=pdd
         pds=0.
         pdd=0.
       elseif(idm2pr(n,k).eq.2)then  !with valence antiquark only 1 SE available
         pds=pdd
         psd=0.
         pdd=0.
       else
         psd=pdd
         pds=pdd
         pdd=pdd**2
       endif
       su=1.-min(1.,pdd+pds+psd)            !diquark probability
       pss=(1.-min(1.,pvs+pva))*su        !no more valence quark: take from sea
       pvs=pvs*su
       pva=pva*su
       su=1.-min(1.,psvv+paas)      !stopping probability
       pss=pss*su
       pvs=pvs*su
       pva=pva*su
       pds=pds*su
       psd=psd*su
       pdd=pdd*su
       su=pss+pvs+pva+pdd+psd+pds+psvv+paas
       psst = pss /su
       pvst = pvs /su
       pvat = pva /su
       psdt = psd /su
       pdst = pds /su
       pddt = pdd /su
       psvvt= psvv/su
       paast= paas/su
       r=rangen()
       if(r.gt.(psst+pvst+pvat+psdt+pdst+psvvt+paast)
     &                               .and.pddt.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=4
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=4
        idstpr(n,k)=idstpr(n,k)+4
        if(iremn.ge.2)idt(it)=idt(it)+2
        if(iremn.eq.3)then   !add diquark flavor to jctref for ProSeF later (sea quark)
          idum=idrafl(icltar,jct,1,'s',3,iret)
          idum=idrafl(icltar,jct,1,'d',3,iret)
          idum=idrafl(icltar,jct,1,'s',3,iret)
          idum=idrafl(icltar,jct,1,'d',3,iret)
        endif
      elseif(r.gt.(psst+pvst+pvat+psdt+psvvt+paast).and.pdst.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=4
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=1
        idstpr(n,k)=idstpr(n,k)+4
        if(iremn.ge.2)idt(it)=idt(it)+1
        if(iremn.eq.3)then   !add diquark flavor to jctref for ProSeF later (sea quark)
          idum=idrafl(icltar,jct,1,'s',3,iret)
          idum=idrafl(icltar,jct,1,'d',3,iret)
        endif
       elseif(r.gt.(psst+pvst+pvat+psvvt+paast).and.psdt.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=1
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=4
        idstpr(n,k)=idstpr(n,k)+4
        if(iremn.ge.2)idt(it)=idt(it)+1
        if(iremn.eq.3)then   !add diquark flavor to jctref for ProSeF later (sea quark)
          idum=idrafl(icltar,jct,1,'s',3,iret)
          idum=idrafl(icltar,jct,1,'d',3,iret)
        endif
       elseif(r.gt.(psst+pvst+pvat+psvvt).and.paast.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=5
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=1
        idstpr(n,k)=idstpr(n,k)+5
        if(iremn.ge.2)idt(it)=idt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(psst+pvst+pvat+pddt).and.psvvt.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=1
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=5
        idstpr(n,k)=idstpr(n,k)+5
        if(iremn.ge.2)idt(it)=idt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(psst+pvst).and.pvat.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=1
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=2
        idstpr(n,k)=idstpr(n,k)+2
        if(iremn.ge.2)ivt(it)=ivt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       elseif(r.gt.psst.and.pvst.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=2
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=1
        idstpr(n,k)=idstpr(n,k)+2
        if(iremn.ge.2)ivt(it)=ivt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,2,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       elseif(psst.gt.eps)then
        if(idm1pr(n,k).ne.2)idm1pr(n,k)=1
        if(idm2pr(n,k).ne.2)idm2pr(n,k)=1
        idstpr(n,k)=idstpr(n,k)+1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       else
        goto 2
       endif

       else
        idm1pr(n,k)=1
        idm2pr(n,k)=1
        idstpr(n,k)=0
       endif

      else

        idp1pr(n,k)=0
        idm2pr(n,k)=0
        idp2pr(n,k)=0
        idm1pr(n,k)=0

      endif

      else       !iLHC

      if(idpr(n,k).eq.3)then
       pssp=0.
       pvsp=0.
       pvap=0.
       pddp=0.
       psvvp=0.
       paasp=0.
       psst=0.
       pvst=0.
       pvat=0.
       pddt=0.
       psvvt=0.
       paast=0.
       if(idhpr(n,k).eq.3)then  !so01
        idp1pr(n,k)=2
        idp2pr(n,k)=8
        idm1pr(n,k)=2
        idm2pr(n,k)=8
       elseif(idhpr(n,k).eq.2)then
        idp1pr(n,k)=1
        idp2pr(n,k)=1
        idm1pr(n,k)=2
        idm2pr(n,k)=8
       elseif(idhpr(n,k).eq.1)then
        idp1pr(n,k)=2
        idp2pr(n,k)=8
        idm1pr(n,k)=1
        idm2pr(n,k)=1
       elseif(idhpr(n,k).eq.0)then
        idp1pr(n,k)=1
        idp2pr(n,k)=1
        idm1pr(n,k)=1
        idm2pr(n,k)=1
       else
        call utstop('ProSeTy-idhpr????&')
       endif
       if(iremn.eq.3)then       !add flavor to jcpref and jctref for psahot and ProSeF later (sea quark)
         idum=idrafl(iclpro,jcp,1,'s',3,iret)
         idum=idrafl(icltar,jct,1,'s',3,iret)
       endif


      elseif(idpr(n,k).eq.1)then

c    projectile

       if(idfpr(n,k).eq.1.or.idfpr(n,k).eq.2)then

       ntry=0
       ivpi=ivp(ip)
       idpi=idp(ip)
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcpi(i,j)=jcp(i,j)
           enddo
         enddo
       endif
 3     ntry=ntry+1
       if(ntry.gt.10)call utstop('something goes wrong in sr ProSeTy&')
       ivp(ip)=ivpi
       idp(ip)=idpi
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcp(i,j)=jcpi(i,j)
           enddo
         enddo
       endif
       pss=wgtval+wgtsea
       if(pss.gt.0.)then
         pss=wgtsea/pss
       else
         pss=0.
       endif
       if(iremn.ge.2)then
         if(iap0.eq.0)then
           pvs=0.
           if(ivp(ip).ne.0)pvs=1.-pss
           pva=0.
           psvv=0.
           if(idp(ip).ne.0)psvv=wgtqqq(iclpro)
           paas=0.
         elseif(ivp0.eq.0)then
           pva=0.
           if(ivp(ip).ne.0)pva=1.-pss
           pvs=0.
           psvv=0.
           paas=0.
           if(idp(ip).ne.0)paas=wgtqqq(iclpro)
         else                   !for meson, no soft string with valence quark (we do not know whether the quark or the antiquark will be used by hard string)
           pvs=0.
           pva=0.
c diquark or antidiquark can be created once in meson remnant
           psvv=0.
           paas=0.
           if(1+idp(ip).ne.0)then
             psvv=wgtqqq(iclpro)
             paas=wgtqqq(iclpro)
           endif
         endif
         pdd=wgtdiq
       elseif(iremn.ne.0)then
         pvs=0.
         pva=0.
         psvv=wgtqqq(iclpro)
         paas=wgtqqq(iclpro)
         pdd=wgtdiq
       else
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         pdd=wgtdiq
       endif
       su=1.-min(1.,pdd)            !diquark probability
       pss=(1.-min(1.,pvs+pva))*su        !no more valence quark: take from sea
       pvs=pvs*su
       pva=pva*su
       su=1.-min(1.,psvv+paas)      !stopping probability
       pdd=pdd*su
       pss=pss*su
       pvs=pvs*su
       pva=pva*su
       su=pss+pvs+pva+pdd+psvv+paas
       pssp = pss /su
       pvsp = pvs /su
       pvap = pva /su
       pddp = pdd /su
       psvvp= psvv/su
       paasp= paas/su
       r=rangen()
       if(r.gt.(pssp+pvsp+pvap+pddp+psvvp).and.paasp.gt.eps)then
        idp1pr(n,k)=5
        idp2pr(n,k)=1
        idsppr(n,k)=6
        if(iremn.ge.2)idp(ip)=idp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(pssp+pvsp+pvap+pddp).and.psvvp.gt.eps)then
        idp1pr(n,k)=1
        idp2pr(n,k)=5
        idsppr(n,k)=5
        if(iremn.ge.2)idp(ip)=idp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(pssp+pvsp+pvap).and.pddp.gt.eps)then
        idp1pr(n,k)=4
        idp2pr(n,k)=4
        idsppr(n,k)=4
        if(iremn.eq.3)then   !add diquark flavor to jcpref for ProSeF later (sea quark)
          idum=idrafl(iclpro,jcp,1,'s',3,iret)
          idum=idrafl(iclpro,jcp,1,'d',3,iret)
        endif
       elseif(r.gt.(pssp+pvsp).and.pvap.gt.eps)then
        idp1pr(n,k)=1
        idp2pr(n,k)=2
        idsppr(n,k)=3
        if(iremn.ge.2)ivp(ip)=ivp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       elseif(r.gt.pssp.and.pvsp.gt.eps)then
        idp1pr(n,k)=2
        idp2pr(n,k)=1
        idsppr(n,k)=2
        if(iremn.ge.2)ivp(ip)=ivp(ip)-1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,2,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       elseif(pssp.gt.eps)then
        idp1pr(n,k)=1
        idp2pr(n,k)=1
        idsppr(n,k)=1
        if(iremn.eq.3)idum=idrafl(iclpro,jcp,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark)
       else
        goto 3
       endif

       else
        idp1pr(n,k)=1
        idp2pr(n,k)=1
        idsppr(n,k)=0
       endif


c    target

       if(idfpr(n,k).eq.1.or.idfpr(n,k).eq.3)then


       ntry=0
       ivti=ivt(it)
       idti=idt(it)
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jcti(i,j)=jct(i,j)
           enddo
         enddo
       endif
 4     ntry=ntry+1
       if(ntry.gt.10)call utstop('something goes wrong in sr ProSeTy&')
       ivt(it)=ivti
       idt(it)=idti
       if(iremn.eq.3)then
         do j=1,2
           do i=1,nrflav
             jct(i,j)=jcti(i,j)
           enddo
         enddo
       endif
       pss=wgtval+wgtsea
       if(pss.gt.0.)then
         pss=wgtsea/pss
       else
         pss=0.
       endif
       if(iremn.ge.2)then
         if(iat0.eq.0)then
           pvs=0.
           if(ivt(it).ne.0)pvs=1.-pss
           pva=0.
           psvv=0.
           if(idt(it).ne.0)psvv=wgtqqq(icltar)
           paas=0.
         elseif(ivt0.eq.0)then
           pva=0.
           if(ivt(it).ne.0)pva=1.-pss
           pvs=0.
           psvv=0.
           paas=0.
           if(idt(it).ne.0)paas=wgtqqq(icltar)
         else                   !for meson, no soft string with valence quark (we do not know whether the quark or the antiquark will be used by hard string)
           pvs=0.
           pva=0.
           psvv=0.
           paas=0.
c diquark or antidiquark can be created once in meson remnant
           if(1+idt(it).ne.0)then
             psvv=wgtqqq(icltar)
             paas=wgtqqq(icltar)
           endif
         endif
         pdd=wgtdiq
       elseif(iremn.ne.0)then
         pvs=0.
         pva=0.
         psvv=wgtqqq(icltar)
         paas=wgtqqq(icltar)
         pdd=wgtdiq
       else
         pvs=0.
         pva=0.
         psvv=0.
         paas=0.
         pdd=wgtdiq
       endif
c no more valence quark: take from sea
       su=1.-min(1.,pdd)            !diquark probability
       pss=(1.-min(1.,pvs+pva))*su        !no more valence quark: take from sea
       pvs=pvs*su
       pva=pva*su
       su=1.-min(1.,psvv+paas)      !stopping probability
       pdd=pdd*su
       pss=pss*su
       pvs=pvs*su
       pva=pva*su
       su=pss+pvs+pva+pdd+psvv+paas
       psst = pss /su
       pvst = pvs /su
       pvat = pva /su
       pddt = pdd /su
       psvvt= psvv/su
       paast= paas/su
       r=rangen()
       if(r.gt.(psst+pvst+pvat+pddt+psvvt).and.paast.gt.eps)then
        idm1pr(n,k)=5
        idm2pr(n,k)=1
        idstpr(n,k)=6
        if(iremn.ge.2)idt(it)=idt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(psst+pvst+pvat+pddt).and.psvvt.gt.eps)then
        idm1pr(n,k)=1
        idm2pr(n,k)=5
        idstpr(n,k)=5
        if(iremn.ge.2)idt(it)=idt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jcpref for ProSeF later (sea quark) (only a q-aq pair because we replace diquark by q-aq (baryon "decay" or "stopping")
       elseif(r.gt.(psst+pvst+pvat).and.pddt.gt.eps)then
        idm1pr(n,k)=4
        idm2pr(n,k)=4
        idstpr(n,k)=4
        if(iremn.eq.3)then   !add diquark flavor to jctref for ProSeF later (sea quark)
          idum=idrafl(icltar,jct,1,'s',3,iret)
          idum=idrafl(icltar,jct,1,'d',3,iret)
        endif
       elseif(r.gt.(psst+pvst).and.pvat.gt.eps)then
        idm1pr(n,k)=1
        idm2pr(n,k)=2
        idstpr(n,k)=3
        if(iremn.ge.2)ivt(it)=ivt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       elseif(r.gt.psst.and.pvst.gt.eps)then
        idm1pr(n,k)=2
        idm2pr(n,k)=1
        idstpr(n,k)=2
        if(iremn.ge.2)ivt(it)=ivt(it)-1
        if(iremn.eq.3)idum=idrafl(icltar,jct,2,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       elseif(psst.gt.eps)then
        idm1pr(n,k)=1
        idm2pr(n,k)=1
        idstpr(n,k)=1
        if(iremn.eq.3)idum=idrafl(icltar,jct,1,'s',3,iret) !add flavor to jctref for ProSeF later (sea quark)
       else
        goto 4
       endif

       else
        idm1pr(n,k)=1
        idm2pr(n,k)=1
        idstpr(n,k)=0
       endif

      elseif(idpr(n,k).eq.0)then

        idp1pr(n,k)=0
        idm2pr(n,k)=0
        idp2pr(n,k)=0
        idm1pr(n,k)=0

      endif

      endif

        if(ish.ge.6)then
      write(ifch,'(a,2(6(f4.2,1x),2x),$)')'ProSeTy ',
     * pssp,pvsp,pvap,pddp,psvvp,paasp, psst,pvst,pvat,pddt,psvvt,paast
      write(ifch,'(2x,3i3,2x,2(i2,1x,2i2,1x,i2,i3,2x))')idpr(n,k),n,k
     * ,idsppr(n,k),idp1pr(n,k),idp2pr(n,k),ivp(ip),idp(ip)
     * ,idstpr(n,k),idm1pr(n,k),idm2pr(n,k),ivt(it),idt(it)
        endif

      if(iremn.eq.3)then
        do j=1,2
          do i=1,nrflav
            jcpref(i,j,ip)=jcp(i,j)
            jctref(i,j,it)=jct(i,j)
          enddo
        enddo
        if(ish.ge.6)then
          write(ifch,'(a,i3,a,1x,4i3,3x,4i3)')'jcpref(',ip,'):',jcp
          write(ifch,'(a,i3,a,1x,4i3,3x,4i3)')'jctref(',it,'):',jct
        endif
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine ProSeF(k,n,iret)
c-----------------------------------------------------------------------
c starting from string properties as already determined in EMS,
c one determines string end flavors
c by checking compatibility with remnant masses.
c strings are written to /cems/ and then to /cptl/
c remnant ic is updated (icproj,ictarg)
c------------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      double precision plc,s,pstg,pend
      common/cems5/plc,s
      common/cems/pstg(5,2),pend(4,4),idend(4)
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      integer icp(2),ict(2),ic(2),icp1(2),icp2(2),icm1(2),icm2(2)
      integer jcp(nflav,2),jct(nflav,2),jcpv(nflav,2),jctv(nflav,2)
      integer jcp1(nflav,2),jcp2(nflav,2),jcm1(nflav,2),jcm2(nflav,2)
      common/col3/ncol,kolpt /cfacmss/facmss /cts/its


c     entry
c     -----

      iret=0

      if(ncol.eq.0)return
      if(abs(itpr(k)).ne.1)return

      ip=iproj(k)
      it=itarg(k)

      if(idpr(n,k).eq.0.or.ivpr(n,k).eq.0)return
      if(idpr(n,k).eq.2)stop'Reggeon'
      if(idpr(n,k).eq.3)return
      call utpri('ProSeF',ish,ishini,5)
      if(ish.ge.5)then
          write(ifch,*)'soft Pomeron'
          write(ifch,*)'k:',k,'  n:',n,'  ip:',ip,'  it:',it
      endif
      np=nppr(n,k)

c         string ends

          pend(1,1)=xxp1pr(n,k)
          pend(2,1)=xyp1pr(n,k)
          pend(3,1)=xp1pr(n,k)*plc/2d0
          pend(4,1)=dsqrt(pend(1,1)**2+pend(2,1)**2+pend(3,1)**2)
          pend(1,2)=xxp2pr(n,k)
          pend(2,2)=xyp2pr(n,k)
          pend(3,2)=xp2pr(n,k)*plc/2d0
          pend(4,2)=dsqrt(pend(1,2)**2+pend(2,2)**2+pend(3,2)**2)
          pend(1,4)=xxm1pr(n,k)
          pend(2,4)=xym1pr(n,k)
          pend(3,4)=-xm1pr(n,k)*plc/2d0
          pend(4,4)=dsqrt(pend(1,4)**2+pend(2,4)**2+pend(3,4)**2)
          pend(1,3)=xxm2pr(n,k)
          pend(2,3)=xym2pr(n,k)
          pend(3,3)=-xm2pr(n,k)*plc/2d0
          pend(4,3)=dsqrt(pend(1,3)**2+pend(2,3)**2+pend(3,3)**2)

c         strings

          pstg(1,1)=xxp1pr(n,k)+xxm2pr(n,k)
          pstg(2,1)=xyp1pr(n,k)+xym2pr(n,k)
          pstg(3,1)=(xp1pr(n,k)-xm2pr(n,k))*plc/2d0
          pstg(4,1)=(xp1pr(n,k)+xm2pr(n,k))*plc/2d0
          pstg(5,1)=dsqrt((pstg(4,1)-pstg(3,1))*(pstg(4,1)+pstg(3,1))
     &                   -pstg(1,1)**2-pstg(2,1)**2)
          pstg(1,2)=xxp2pr(n,k)+xxm1pr(n,k)
          pstg(2,2)=xyp2pr(n,k)+xym1pr(n,k)
          pstg(3,2)=(xp2pr(n,k)-xm1pr(n,k))*plc/2d0
          pstg(4,2)=(xp2pr(n,k)+xm1pr(n,k))*plc/2d0
          pstg(5,2)=dsqrt((pstg(4,2)-pstg(3,2))*(pstg(4,2)+pstg(3,2))
     &                   -pstg(2,2)**2-pstg(1,2)**2)

c         initialize

          ntry=0
  777     ntry=ntry+1
          if(ntry.gt.100)goto1001

          if(iremn.ge.2)then    !uses precalculated flavors
            do i=1,2
              icp(i)=icproj(i,ip)
              ict(i)=ictarg(i,it)
            enddo
            if(iLHC.eq.1)then
              call iddeco(icp,jcpv)
              call iddeco(ict,jctv)
            endif
            do j=1,2
              do i=1,nrflav
                jcp(i,j)=jcpref(i,j,ip)
                jct(i,j)=jctref(i,j,it)
                if(iLHC.eq.0)then
                jcpv(i,j)=jcpval(i,j,ip)
                jctv(i,j)=jctval(i,j,it)
                endif
              enddo
              do i=nrflav+1,nflav
                jcp(i,j)=0
                jct(i,j)=0
                jcpv(i,j)=0
                jctv(i,j)=0
               enddo
            enddo
          else
            do i=1,2
              icp(i)=icproj(i,ip)
              ict(i)=ictarg(i,it)
            enddo
            call iddeco(icp,jcp)
            call iddeco(ict,jct)
            do j=1,2
              do i=1,nflav
                jcpv(i,j)=0
                jctv(i,j)=0
              enddo
            enddo
          endif
          do i=1,2
           icp1(i)=0
           icp2(i)=0
           icm1(i)=0
           icm2(i)=0
           do j=1,nflav
            jcp1(j,i)=0
            jcp2(j,i)=0
            jcm1(j,i)=0
            jcm2(j,i)=0
           enddo
          enddo
          idpj0=idtr2(icp)
          idtg0=idtr2(ict)
          do j=1,4
           idend(j)=0
          enddo

          if(ish.ge.7)then
            write(ifch,'(a,3x,6i3,3x,6i3,i9)')' proj: '
     *     ,jcp,idpj0
            write(ifch,'(a,6i3,3x,6i3)')' proj val:  ',jcpv
          endif
          if(ish.ge.7)then
            write(ifch,'(a,3x,6i3,3x,6i3,i9)')' targ: '
     *    ,jct,idtg0
            write(ifch,'(a,6i3,3x,6i3)')' targ val:  ',jctv
          endif

c         determine string flavors

          call fstrfl(jcp,jct,jcpv,jctv,icp1,icp2,icm1,icm2
     *                ,idp1pr(n,k),idp2pr(n,k),idm1pr(n,k),idm2pr(n,k)
     *                                   ,idsppr(n,k),idstpr(n,k),iret)
          if(iret.ne.0)goto 1002

c         check mass string 1

          ic(1)=icp1(1)+icm2(1)
          ic(2)=icp1(2)+icm2(2)
          if(ic(1).gt.0.or.ic(2).gt.0)then
           am=sngl(pstg(5,1))
           call iddeco(icp1,jcp1)
           call iddeco(icm2,jcm2)
           ammns=utamnx(jcp1,jcm2)
           if(ish.ge.7)write(ifch,'(a,2i7,2e12.3)')
     *           ' string 1 - ic,mass,min.mass:',ic,am,ammns
           if(am.lt.ammns*facmss)then
             goto 777   !avoid virpom
           endif
           if(iLHC.eq.1)then
           idend(1)=idtra(icp1,0,0,0)
           idend(3)=idtra(icm2,0,0,0)
           else
           idend(1)=idtra(icp1,0,0,3)
           idend(3)=idtra(icm2,0,0,3)
           endif
           if(ish.ge.7)write(ifch,'(a,2i6)') ' string 1 - SE-ids:'
     *      ,idend(1),idend(3)
          endif

c         check mass string 2

          ic(1)=icp2(1)+icm1(1)
          ic(2)=icp2(2)+icm1(2)
          if(ic(1).gt.0.or.ic(2).gt.0)then
           am=sngl(pstg(5,2))
           call iddeco(icp2,jcp2)
           call iddeco(icm1,jcm1)
           ammns=utamnx(jcp2,jcm1)
           if(ish.ge.7)write(ifch,'(a,2i7,2e12.3)')
     *           ' string 2 - ic,mass,min.mass:',ic,am,ammns
           if(am.lt.ammns*facmss)then
             goto 777  !avoid virpom
           endif
           if(iLHC.eq.1)then
           idend(2)=idtra(icp2,0,0,0)
           idend(4)=idtra(icm1,0,0,0)
           else
           idend(2)=idtra(icp2,0,0,3)
           idend(4)=idtra(icm1,0,0,3)
           endif
           if(ish.ge.7)write(ifch,'(a,2i6)') ' string 2 - SE-ids:'
     *      ,idend(2),idend(4)
          endif

          if(ish.ge.5)then
          write(ifch,'(a,i10)')' pom:   '
     *    ,idptl(np)
          write(ifch,'(a,2i5)')' str 1: '
     *    ,idend(1),idend(3)
          write(ifch,'(a,2i5)')' str 2: '
     *    ,idend(2),idend(4)
          endif

c         update remnant ic

c determine icp,ict
c Similar process for hard pomeron in epos-rsh !!!!

          if(iremn.ge.2)then    !uses precalculated flavors

            do j=1,2
              do i=1,nrflav
                jcpref(i,j,ip)=jcp(i,j)
                jctref(i,j,it)=jct(i,j)
                if(iLHC.eq.0)then
                jcpval(i,j,ip)=jcpv(i,j)
                jctval(i,j,it)=jctv(i,j)
                endif
              enddo
            enddo
            if(iLHC.eq.1)then
            call idenco(jcpv,icp,iret)
            if(iret.ne.0)goto 1002
            call idenco(jctv,ict,iret)
            if(iret.ne.0)goto 1002
            do i=1,2
              icproj(i,ip)=icp(i)
              ictarg(i,it)=ict(i)
            enddo
            endif
            if(ish.ge.5)then
              write(ifch,'(a,6i3,3x,6i3)')' proj:  ',jcp
              write(ifch,'(a,6i3,3x,6i3)')' proj val:  ',jcpv
              write(ifch,'(a,6i3,3x,6i3)')' targ:  ',jct
              write(ifch,'(a,6i3,3x,6i3)')' targ val:  ',jctv
            endif

          else

            call idenco(jcp,icp,iret)
            if(iret.ne.0)goto 1002
            call idenco(jct,ict,iret)
            if(iret.ne.0)goto 1002
            do i=1,2
              icproj(i,ip)=icp(i)
              ictarg(i,it)=ict(i)
            enddo
            if(ish.ge.5)then
              write(ifch,'(a,2i7,1x,a)')' proj:  '
     *             ,(icp(l),l=1,2)
              write(ifch,'(a,2i7,1x,a)')' targ:  '
     *             ,(ict(l),l=1,2)
            endif

          endif

c         write strings to /cptl/

          its=idp1pr(n,k)+idm2pr(n,k)
          call fstrwr(1,1,3,k,n)
          its=idp2pr(n,k)+idm1pr(n,k)
          call fstrwr(2,2,4,k,n)

c     exit
c     ----

1000  continue
      call utprix('ProSeF',ish,ishini,5)
      return

 1002 jerr(1)=jerr(1)+1         ! > 9 quarks per flavor attempted.
 1001 iret=1
      if(ish.ge.5)write(ifch,'(a)')'Problem in ProSeF ... '
      goto 1000

      end

c-----------------------------------------------------------------------
      subroutine fstrfl(jcp,jct,jcpv,jctv,icp1,icp2,icm1,icm2
     *                         ,idp1,idp2,idm1,idm2,idsp,idst,iret)
c-----------------------------------------------------------------------
c knowing the string end types (idp1,idp2,idm1,idm2)
c               and remnant flavors (icp,ict)
c               and remnant link of the string (idsp and idst)
c   for LHC     (idsp/t=100 with one idp/idm=2 means that the valence quark 
c                to use is define in the corresponding icp/icm
c                (using just 1 to 6 for flavor identification (no diquark)))
c one determines quark flavors of string ends (icp1,icp2,icm1,icm2)
c               and updates remnant flavors (icp,ict)
c iret=0   ok
c iret=1   problem, more than 9 quarks per flavor attempted
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      integer icp1(2),icp2(2),icm1(2),icm2(2)
      integer jcp(nflav,2),jct(nflav,2)
     &       ,jcpi(nflavems,2),jcti(nflavems,2)
      integer iq(2,4),jcpv(nflav,2),jctv(nflav,2)
      character m
c      data neuz/0/proz/0/dtaz/0/
c      save neuz,proz,dtaz

      call utpri('fstrfl',ish,ishini,7)

c     entry
c     -----

      idum=0
      iret=0
      iret1=0
      iret2=0
      iret3=0
      iret4=0

      if(idp1.eq.8)stop'fstrfl: fragm quarks not used any more'
      if(idp2.eq.8)stop'fstrfl: fragm quarks not used any more'
      if(idm1.eq.8)stop'fstrfl: fragm quarks not used any more'
      if(idm2.eq.8)stop'fstrfl: fragm quarks not used any more'

c determine flavors of string ends (u,d,s)

      if(ish.ge.7)then
       write(ifch,'(a,3x,2i3)')' string 1, SE types:',idp1,idm2
       write(ifch,'(a,3x,2i3)')' string 2, SE types:',idp2,idm1
      endif

c empty

      if(idp1.eq.0)then
       iq(1,1)=0
       iq(2,1)=0
      endif
      if(idp2.eq.0)then
       iq(1,2)=0
       iq(2,2)=0
      endif
      if(idm1.eq.0)then
       iq(1,4)=0
       iq(2,4)=0
      endif
      if(idm2.eq.0)then
       iq(1,3)=0
       iq(2,3)=0
      endif
      do j=1,2
        do n=1,nrflav
          jcpi(n,j)=jcp(n,j)
          jcti(n,j)=jct(n,j)
        enddo
      enddo

c Projectile

      if(idsp.eq.0.or.iremn.eq.0)then
c give the same flavor to quark and antiquark not to change remnant flavor

        if(idp1.eq.4)then
c diquarks, code 4
          iq(1,1)=idrafl(iclpro,jcp,1,'d',0,iret)
          iq(2,1)=idrafl(iclpro,jcp,1,'d',0,iret)
          iq(1,2)=iq(1,1)
          iq(2,2)=iq(2,1)
        else
c sea quarks, code 1
          iq(1,1)=idrafl(iclpro,jcp,1,'s',0,iret)
          iq(2,1)=0
          iq(1,2)=iq(1,1)
          iq(2,2)=0
        endif

      elseif(iremn.ge.2)then
c count valence quarks properly

c valence quarks

        if(idp1.eq.2)then

          if(iLHC.eq.1)then
            if(idsp.eq.100)then
              iq(1,1)=icp1(1)   !flavor of hard quark already defined
            else
              iq(1,1)=idrafl(iclpro,jcpv,1,'v',0,idum)
            endif
            if(iq(1,1).gt.0)then !if still exist, update jcp and jcpv
              call idsufl3(iq(1,1),1,jcpv)
            else                ! if not, use jcp directly and sea
              iq(1,1)=idrafl(iclpro,jcp,1,'s',1,idum)
            endif
          else

          iq(1,1)=idrafl(iclpro,jcpv,1,'v',0,idum)
          if(iq(1,1).gt.0)then          !if still exist, update jcp and jcpv
            call idsufl3(iq(1,1),1,jcpv)
            call idsufl3(iq(1,1),1,jcp)
          else                          ! if not, use jcp directly
            iq(1,1)=idrafl(iclpro,jcp,1,'v',1,idum)
          endif

          endif

          iq(2,1)=0
        endif

        if(idp2.eq.2)then

          if(iLHC.eq.1)then
            if(idsp.eq.100)then
              iq(1,2)=icp2(2)   !flavor of hard antiquark already defined
            else
              iq(1,2)=idrafl(iclpro,jcpv,2,'v',0,idum)
            endif
            if(iq(1,2).gt.0)then !if still exist, update jcp and jcpv
              call idsufl3(iq(1,2),2,jcpv)
            else                ! if not, use jcp directly and sea
              iq(1,2)=idrafl(iclpro,jcp,2,'s',1,idum)
            endif
          else

          iq(1,2)=idrafl(iclpro,jcpv,2,'v',0,idum)
          if(iq(1,2).gt.0)then          !if still exist, update jcp and jcpv
            call idsufl3(iq(1,2),2,jcpv)
            call idsufl3(iq(1,2),2,jcp)
          else                          ! if not, use jcp directly
            iq(1,2)=idrafl(iclpro,jcp,2,'v',1,idum)
          endif
          endif
          iq(2,2)=0
        endif

c sea quarks
        m='v'           !iremn=3

        if(idp1.eq.1)then
          if(iremn.eq.2)m='s'
          j=1                    !quark
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(1,1)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
          iq(2,1)=0
        elseif(idp1.ge.4)then
          if(iremn.eq.2)m='d'
          j=2                    !anti-diquark
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(1,1)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(2,1)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
        endif
        if(idp2.eq.1)then
          if(iremn.eq.2)m='s'
          j=2                    !antiquark
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(1,2)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
          iq(2,2)=0
        elseif(idp2.ge.4)then
          if(iremn.eq.2)m='d'
          j=1                    !diquark
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(1,2)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
          i=idrafl(iclpro,jcp,j,m,1,idum)
          iq(2,2)=i
          if(iLHC.eq.0.and.jcp(i,j)-jcpv(i,j).lt.0)jcpv(i,j)=jcpv(i,j)-1
        endif

      elseif(iremn.ne.0)then
c free remant content

c valence quarks

        if(idp1.eq.2)then
          if(iLHC.eq.1.and.idsp.eq.100)then
            iq(1,1)=icp1(1)         !flavor of hard quark already defined
          else
            iq(1,1)=idrafl(iclpro,jcp,1,'v',1,iret)
          endif
          iq(2,1)=0
        endif
        if(idp2.eq.2)then
          if(iLHC.eq.1.and.idsp.eq.100)then
            iq(1,2)=icp2(1)     !flavor of hard antiquark already defined
          else
            iq(1,2)=idrafl(iclpro,jcp,2,'v',1,iret)
          endif
          iq(2,2)=0
        endif

c sea quarks

        if(idp1.eq.1)then
          iq(1,1)=idrafl(iclpro,jcp,1,'s',1,iret1)
          iq(2,1)=0
        endif
        if(idp2.eq.1)then
          iq(1,2)=idrafl(iclpro,jcp,2,'s',1,iret2)
          iq(2,2)=0
        endif

c diquarks, code 4

        if(idp1.eq.4.or.idp2.eq.4)then
          iq(1,1)=idrafl(iclpro,jcp,2,'d',1,iret1)
          iq(2,1)=idrafl(iclpro,jcp,2,'d',1,iret1)
          iq(1,2)=idrafl(iclpro,jcp,1,'d',1,iret2)
          iq(2,2)=idrafl(iclpro,jcp,1,'d',1,iret2)
        endif

c diquarks, code 5 (former valence, but actually sea)

        if(idp1.eq.5)then
          iq(1,1)=idrafl(iclpro,jcp,2,'d',1,iret1)
          iq(2,1)=idrafl(iclpro,jcp,2,'d',1,iret1)
        endif
        if(idp2.eq.5)then
          iq(1,2)=idrafl(iclpro,jcp,1,'d',1,iret2)
          iq(2,2)=idrafl(iclpro,jcp,1,'d',1,iret2)
        endif


        if(iret.ne.0)goto 1000



c in case of saturated remnants, use the same flavor for quark and anti-quark
c at string-end
        if(iret1.ne.0.or.iret2.ne.0)then
          do j=1,2
            do n=1,nrflav
              jcp(n,j)=jcpi(n,j)
            enddo
          enddo
          if(idp1.gt.idp2.or.(idp1.eq.idp2.and.rangen().gt.0.5))then
            iq(1,2)=iq(1,1)
            iq(2,2)=iq(2,1)
          else
            iq(1,1)=iq(1,2)
            iq(2,1)=iq(2,2)
          endif
        endif

      endif

c Target

      if(idst.eq.0.or.iremn.eq.0)then
c give the same flavor to quark and antiquark not to change remnant flavor


        if(idm1.eq.4)then
c diquarks, code 4
          iq(1,4)=idrafl(icltar,jct,1,'d',0,iret)
          iq(2,4)=idrafl(icltar,jct,1,'d',0,iret)
          iq(1,3)=iq(1,4)
          iq(2,3)=iq(2,4)
        else
c sea quarks,code 1
          iq(1,4)=idrafl(icltar,jct,1,'s',0,iret)
          iq(2,4)=0
          iq(1,3)=iq(1,4)
          iq(2,3)=0
        endif

      elseif(iremn.ge.2)then
c count valence quarks properly

c valence quarks

        if(idm1.eq.2)then

          if(iLHC.eq.1)then
            if(idst.eq.100)then
              iq(1,4)=icm1(1)   !flavor of hard quark already defined
            else
              iq(1,4)=idrafl(icltar,jctv,1,'v',0,idum)
            endif
            if(iq(1,4).gt.0)then !if still exist, update jct and jctv
              call idsufl3(iq(1,4),1,jctv)
            else                ! if not, use jct directly
              iq(1,4)=idrafl(icltar,jct,1,'s',1,idum)
            endif
          else

          iq(1,4)=idrafl(icltar,jctv,1,'v',0,idum)
          if(iq(1,4).gt.0)then          !if still exist, update jct and jctv
            call idsufl3(iq(1,4),1,jctv)
            call idsufl3(iq(1,4),1,jct)
          else                          ! if not, use jct directly
            iq(1,4)=idrafl(icltar,jct,1,'v',1,idum)
          endif

          endif

          iq(2,4)=0
        endif
        if(idm2.eq.2)then

          if(iLHC.eq.1)then
            if(idst.eq.100)then
              iq(1,3)=icm2(2)   !flavor of hard antiquark already defined
            else
              iq(1,3)=idrafl(icltar,jctv,2,'v',0,idum)
            endif
            if(iq(1,3).gt.0)then !if still exist, update jct and jctv
              call idsufl3(iq(1,3),2,jctv)
            else                ! if not, use jct directly
              iq(1,3)=idrafl(icltar,jct,2,'s',1,idum)
            endif
          else

          iq(1,3)=idrafl(icltar,jctv,2,'v',0,idum)
          if(iq(1,3).gt.0)then          !if still exist, update jct and jctv
            call idsufl3(iq(1,3),2,jctv)
            call idsufl3(iq(1,3),2,jct)
          else                          ! if not, use jct directly
            iq(1,3)=idrafl(icltar,jct,2,'v',1,idum)
          endif
          endif
          iq(2,3)=0
        endif

c sea quarks
        m='v'           !iremn=3

        if(idm1.eq.1)then
          if(iremn.eq.2)m='s'
          j=1                    !quark
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(1,4)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
          iq(2,4)=0
        elseif(idm1.ge.4)then
          if(iremn.eq.2)m='d'
          j=2                   !anti-diquark
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(1,4)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(2,4)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
        endif
        if(idm2.eq.1)then
          if(iremn.eq.2)m='s'
          j=2                    !antiquark
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(1,3)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
          iq(2,3)=0
        elseif(idm2.ge.4)then
          if(iremn.eq.2)m='d'
          j=1                    !diquark
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(1,3)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
          i=idrafl(icltar,jct,j,m,1,idum)
          iq(2,3)=i
          if(iLHC.eq.0.and.jct(i,j)-jctv(i,j).lt.0)jctv(i,j)=jctv(i,j)-1
        endif

      elseif(iremn.ne.0)then

c valence quarks

        if(idm1.eq.2)then
          if(iLHC.eq.1.and.idst.eq.100)then
            iq(1,4)=icm1(1)         !flavor of hard quark already defined
          else
            iq(1,4)=idrafl(icltar,jct,1,'v',1,iret)
          endif
          iq(2,4)=0
        endif
        if(idm2.eq.2)then
          if(iLHC.eq.1.and.idst.eq.100)then
            iq(1,3)=icm2(1)         !flavor of hard antiquark already defined
          else
            iq(1,3)=idrafl(icltar,jct,2,'v',1,iret)
          endif
          iq(2,3)=0
        endif

c sea quarks

        if(idm1.eq.1)then
          iq(1,4)=idrafl(icltar,jct,1,'s',1,iret4)
          iq(2,4)=0
        endif
        if(idm2.eq.1)then
          iq(1,3)=idrafl(icltar,jct,2,'s',1,iret3)
          iq(2,3)=0
        endif

c diquarks, code 4

        if(idm1.eq.4.or.idm2.eq.4)then
          iq(1,4)=idrafl(icltar,jct,2,'d',1,iret3)
          iq(2,4)=idrafl(icltar,jct,2,'d',1,iret3)
          iq(1,3)=idrafl(icltar,jct,1,'d',1,iret4)
          iq(2,3)=idrafl(icltar,jct,1,'d',1,iret4)
        endif

c diquarks, code 5 (former valence, but actually sea)

        if(idm1.eq.5)then
          iq(1,4)=idrafl(icltar,jct,2,'d',1,iret4)
          iq(2,4)=idrafl(icltar,jct,2,'d',1,iret4)
        endif
        if(idm2.eq.5)then
          iq(1,3)=idrafl(icltar,jct,1,'d',1,iret3)
          iq(2,3)=idrafl(icltar,jct,1,'d',1,iret3)
        endif


        if(iret.ne.0)goto 1000



c in case of saturated remnants, use the same flavor for quark and anti-quark
c at string-end

        if(iret3.ne.0.or.iret4.ne.0)then
          do j=1,2
            do n=1,nrflav
              jct(n,j)=jcti(n,j)
            enddo
          enddo
          if(idm1.gt.idm2.or.(idm1.eq.idm2.and.rangen().gt.0.5))then
            iq(1,4)=iq(1,3)
            iq(2,4)=iq(2,3)
          else
            iq(1,3)=iq(1,4)
            iq(2,3)=iq(2,4)
          endif
        endif

      endif

      ifla=iq(1,1)
      iflb=iq(2,1)
      iflc=iq(1,3)
      ifld=iq(2,3)
      if(ish.ge.7)write(ifch,'(a,2i5,4x,2i5)')
     *' string 1, string ends:',ifla,iflb,iflc,ifld

      if(ifla.gt.0)then
       if(iflb.eq.0)then
        icp1(1)=10**(6-ifla)
        icp1(2)=0
       else
        icp1(1)=0
        icp1(2)=10**(6-ifla)
        icp1(2)=icp1(2)+10**(6-iflb)
       endif
      endif

      if(iflc.gt.0)then
       if(ifld.eq.0)then
        icm2(1)=0
        icm2(2)=10**(6-iflc)
       else
        icm2(1)=10**(6-iflc)
        icm2(1)=icm2(1)+10**(6-ifld)
        icm2(2)=0
       endif
      endif

      ifla=iq(1,4)
      iflb=iq(2,4)
      iflc=iq(1,2)
      ifld=iq(2,2)
      if(ish.ge.7)write(ifch,'(a,2i5,4x,2i5)')
     *' string 2, string ends:',ifla,iflb,iflc,ifld

      if(ifla.gt.0)then
       if(iflb.eq.0)then
        icm1(1)=10**(6-ifla)
        icm1(2)=0
       else
        icm1(1)=0
        icm1(2)=10**(6-ifla)
        icm1(2)=icm1(2)+10**(6-iflb)
       endif
      endif

      if(iflc.gt.0)then
       if(ifld.eq.0)then
        icp2(1)=0
        icp2(2)=10**(6-iflc)
       else
        icp2(1)=10**(6-iflc)
        icp2(1)=icp2(1)+10**(6-ifld)
        icp2(2)=0
       endif
      endif

      if(ish.ge.7)then
        write(ifch,'(a,2i7,4x,2i7)')
     *  ' SE-forw:',icp1(1),icp1(2),icp2(1),icp2(2)
        write(ifch,'(a,2i7,4x,2i7)')
     *  ' SE-back:',icm1(1),icm1(2),icm2(1),icm2(2)
        write(ifch,'(a,3x,6i3,3x,6i3)')' proj:',jcp
        write(ifch,'(a,3x,6i3,3x,6i3)')' proj val:',jcpv
        write(ifch,'(a,3x,6i3,3x,6i3)')' targ:',jct
        write(ifch,'(a,3x,6i3,3x,6i3)')' targ val:',jctv
      endif

c     exit
c     ----

1000  continue
      call utprix('fstrfl',ish,ishini,7)
      return
      end


cc-----------------------------------------------------------------------
c      subroutine fremfl(icp,ict,iret)
cc-----------------------------------------------------------------------
cc checks projectile and target flavor (icp,ict)
cc in case of reggeon exchange they do not correspond to hadrons.
cc one transfers therefore flavor from one side to the other in order
cc to have hadron flavor.
cc icp and ict are modified correspondingly
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      integer icp(2),ict(2),jcp(6,2),jct(6,2),kp(4),kt(4)
c
c      call utpri('fremfl',ish,ishini,7)
c
cc     entry
cc     -----
c
c      iret=0
c
c      call iddeco(icp,jcp)
c      call iddeco(ict,jct)
c
c      iakp=0
c      iakt=0
c      ikp=0
c      ikt=0
c      do l=1,4
c       kp(l)=jcp(l,1)-jcp(l,2)
c       kt(l)=jct(l,1)-jct(l,2)
c       iakp=iakp+iabs(kp(l))
c       iakt=iakt+iabs(kt(l))
c       ikp=ikp+kp(l)
c       ikt=ikt+kt(l)
c      enddo
c      if(ish.ge.7)write(ifch,*)'iak_p:',iakp,' ik_p:',ikp
c      if(ish.ge.7)write(ifch,*)'iak_t:',iakt,' ik_t:',ikt
c
c      if(iakp.eq.4)then
c       if(ikp.eq.4.or.ikp.eq.-2)then
c        ifl=idrafl(jcp,1,'v',iret)
c        iqp=2      ! subtract quark
c        iqt=1      ! add quark
c       elseif(ikp.eq.-4.or.ikp.eq.2)then
c        ifl=idrafl(jcp,2,'v',iret)
c        iqp=1      ! subtract antiquark
c        iqt=2      ! add antiquark
c       else
c        call utstop('fremfl&')
c       endif
c      elseif(iakt.eq.4)then
c       if(ikt.eq.4.or.ikt.eq.-2)then
c        ifl=idrafl(jct,1,'v',iret)
c        iqp=1      ! add quark
c        iqt=2      ! subtract quark
c       elseif(ikt.eq.-4.or.ikt.eq.2)then
c        ifl=idrafl(jct,2,'v',iret)
c        iqp=2      ! add antiquark
c        iqt=1      ! subtract antiquark
c       else
c        call utstop('fremfl&')
c       endif
c      elseif(iakp.eq.3)then
c       if(ikp.gt.0)then
c        ifl=idrafl(jcp,1,'v',iret)
c        iqp=2      ! subtract quark
c        iqt=1      ! add quark
c       else
c        ifl=idrafl(jcp,2,'v',iret)
c        iqp=1      ! subtract antiquark
c        iqt=2      ! add antiquark
c       endif
c      elseif(iakt.eq.3)then
c       if(ikt.gt.0)then
c        ifl=idrafl(jct,1,'v',iret)
c        iqp=1      ! add quark
c        iqt=2      ! subtract quark
c       else
c        ifl=idrafl(jct,2,'v',iret)
c        iqp=2      ! add antiquark
c        iqt=1      ! subtract antiquark
c       endif
c      elseif(iakp.eq.2)then
c       if(ikp.gt.0)then
c        ifl=idrafl(jct,1,'v',iret)
c        iqp=1      ! add quark
c        iqt=2      ! subtract quark
c       else
c        ifl=idrafl(jct,2,'v',iret)
c        iqp=2      ! add antiquark
c        iqt=1      ! subtract antiquark
c       endif
c      elseif(iakt.eq.2)then
c       if(ikt.gt.0)then
c        ifl=idrafl(jct,1,'v',iret)
c        iqp=2      ! subtract quark
c        iqt=1      ! add quark
c       else
c        ifl=idrafl(jct,2,'v',iret)
c        iqp=1      ! subtract antiquark
c        iqt=2      ! add antiquark
c       endif
c      elseif(iakp.eq.1)then
c       if(ikp.gt.0)then
c        ifl=idrafl(jcp,2,'v',iret)
c        iqp=2      ! add antiquark
c        iqt=1      ! subtract antiquark
c       else
c        ifl=idrafl(jcp,1,'v',iret)
c        iqp=1      ! add quark
c        iqt=2      ! subtract quark
c       endif
c      elseif(iakt.eq.1)then
c       if(ikt.gt.0)then
c        ifl=idrafl(jct,2,'v',iret)
c        iqp=1      ! subtract antiquark
c        iqt=2      ! add antiquark
c       else
c        ifl=idrafl(jct,1,'v',iret)
c        iqp=2      ! subtract quark
c        iqt=1      ! add quark
c       endif
c      else
c       call utstop('fremfl: error&')
c      endif
c
c      if(ish.ge.7)write(ifch,*)'iq_p:',iqp,' iq_t:',iqt,' if:',ifl
c      call uticpl(icp,ifl,iqp,iret)
c      if(iret.ne.0)goto1000
c      call uticpl(ict,ifl,iqt,iret)
c      if(iret.ne.0)goto1000
c
cc     exit
cc     ----
c
c1000  continue
c      call utprix('fremfl',ish,ishini,7)
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine fstrwr(j,ii,jj,k,n)
c-----------------------------------------------------------------------
c take pstg(5,j),pend(4,ii),idend(ii),pend(4,jj),idend(jj)  (/cems/)
c and write it to /cptl/
c-----------------------------------------------------------------------
c  j:     string 1 or 2
c  ii,jj: string end (1,2: proj; 3,4: targ)
c  k:     current collision
c  n:     current pomeron
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      double precision pstg,pend,ptt3!,utpcmd
      common/cems/pstg(5,2),pend(4,4),idend(4)
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      double precision  pp(4)
      common/cts/its

      call utpri('fstrwr',ish,ishini,7)

      if(idend(ii).ne.0.and.idend(jj).ne.0)then

c string
c        id1=abs(idend(ii))
c        id2=abs(idend(jj))
c        call idmass(id1,am1)
c        call idmass(id2,am2)
c        if(id1.gt.100)then
c          am1=am1+qmass(0)
c        endif
c        if(id2.gt.100)then
c          am2=am2+qmass(0)
c        endif
c        ptt3=utpcmd(pstg(5,j),dble(am1),dble(am2),iret)
c        if(iret.ne.0.or.pstg(5,j)-dble(am1)-dble(am2).le.0d0)then
c          ptt3=0.5d0*pstg(5,j)
c          am1=0.
c          am2=0.
c        endif
        am1=0.
        am2=0.
        ptt3=0.5d0*pstg(5,j)

       call utlob2(1,pstg(1,j),pstg(2,j),pstg(3,j),pstg(4,j),pstg(5,j)
     * ,pend(1,ii),pend(2,ii),pend(3,ii),pend(4,ii),20)
       pp(1)=0d0
       pp(2)=0d0
       pp(3)=ptt3!.5d0*pstg(5,j)
       pp(4)=sqrt(ptt3*ptt3+dble(am1*am1))!.5d0*pstg(5,j)
       call utrot2
     * (-1,pend(1,ii),pend(2,ii),pend(3,ii),pp(1),pp(2),pp(3))
       call utlob2(-1,pstg(1,j),pstg(2,j),pstg(3,j),pstg(4,j),pstg(5,j)
     * ,pp(1),pp(2),pp(3),pp(4),21)

       npom=nppr(n,k)
       if(ifrptl(1,npom).eq.0)ifrptl(1,npom)=nptl+1
       ifrptl(2,npom)=nptl+2
       istptl(npom)=31

       nptl=nptl+1
       pptl(1,nptl)=sngl(pp(1))
       pptl(2,nptl)=sngl(pp(2))
       pptl(3,nptl)=sngl(pp(3))
       pptl(4,nptl)=sngl(pp(4))
       pptl(5,nptl)=am1 !0.
       istptl(nptl)=20
       iorptl(nptl)=npom
       jorptl(nptl)=0
       ifrptl(1,nptl)=0
       ifrptl(2,nptl)=0
       xorptl(1,nptl)=coord(1,k)
       xorptl(2,nptl)=coord(2,k)
       xorptl(3,nptl)=coord(3,k)
       xorptl(4,nptl)=coord(4,k)
       tivptl(1,nptl)=xorptl(4,nptl)
       tivptl(2,nptl)=xorptl(4,nptl)
       idptl(nptl)=idend(ii)
       ityptl(nptl)=ityptl(npom)+j
       itsptl(nptl)=its
       rinptl(nptl)=-9999
       qsqptl(nptl)=pstg(4,j)**2
       zpaptl(1,nptl)=0.
       zpaptl(2,nptl)=0.

       nptl=nptl+1
       do i=1,4
        pptl(i,nptl)=sngl(pstg(i,j))-pptl(i,nptl-1)
       enddo
       pptl(5,nptl)=am2!0.

       istptl(nptl)=20
       iorptl(nptl)=nppr(n,k)
       jorptl(nptl)=0
       ifrptl(1,nptl)=0
       ifrptl(2,nptl)=0
       xorptl(1,nptl)=coord(1,k)
       xorptl(2,nptl)=coord(2,k)
       xorptl(3,nptl)=coord(3,k)
       xorptl(4,nptl)=coord(4,k)
       tivptl(1,nptl)=xorptl(4,nptl)
       tivptl(2,nptl)=xorptl(4,nptl)
       idptl(nptl)=idend(jj)
       ityptl(nptl)=ityptl(npom)+j
       itsptl(nptl)=its
       rinptl(nptl)=-9999
       qsqptl(nptl)=pstg(4,j)**2
       zpaptl(1,nptl)=0.
       zpaptl(2,nptl)=0.

       if(ish.ge.7)then
        write(ifch,100)' kink:',(pptl(l,nptl-1),l=1,4),idptl(nptl-1)
        write(ifch,100)' kink:',(pptl(l,nptl),l=1,4),idptl(nptl)
       endif

      elseif(idend(ii).ne.0.and.idend(jj).eq.0)then

c resonance

       npom=nppr(n,k)
       if(ifrptl(1,npom).eq.0)ifrptl(1,npom)=nptl+1
       ifrptl(2,npom)=nptl+1
       istptl(npom)=31

       nptl=nptl+1
       idptl(nptl)=idend(ii)
       pptl(1,nptl)=sngl(pstg(1,j))
       pptl(2,nptl)=sngl(pstg(2,j))
       pptl(3,nptl)=sngl(pstg(3,j))
       pptl(4,nptl)=sngl(pstg(4,j))
       pptl(5,nptl)=sngl(pstg(5,j))
       istptl(nptl)=0
       iorptl(nptl)=npom
       jorptl(nptl)=0
       ifrptl(1,nptl)=0
       ifrptl(2,nptl)=0
       xorptl(1,nptl)=coord(1,k)
       xorptl(2,nptl)=coord(2,k)
       xorptl(3,nptl)=coord(3,k)
       xorptl(4,nptl)=coord(4,k)
       tivptl(1,nptl)=coord(4,k)
       call idtau(idptl(nptl),pptl(4,nptl),pptl(5,nptl),taugm)
       tivptl(2,nptl)=tivptl(1,nptl)+taugm*(-alog(rangen()))
       ityptl(nptl)=ityptl(npom)+2+j
       itsptl(nptl)=its
       rinptl(nptl)=-9999
       qsqptl(nptl)=0.
       zpaptl(1,nptl)=0.
       zpaptl(2,nptl)=0.

       if(ish.ge.7)then
        write(ifch,100)'  res:',(pptl(l,nptl),l=1,4),idptl(nptl)
       endif
      elseif(idend(ii).eq.0.and.idend(jj).eq.0)then
       goto1000
      else
       call utstop('error in fstrwr&')
      endif

  100 format(a,4e9.3,i5)

1000  continue
      call utprix('fstrwr',ish,ishini,7)
      return
      end

c-----------------------------------------------------------------------
      subroutine ProReF(ir,m,iretxx)
c-----------------------------------------------------------------------
c  proposes flavor for remnant m for proj (ir=1) or target (ir=-1)
c  and writes remnant into /cptl/ as string or hadron
c   ityptl definitions:
c      51  41  ...  rmn drop
c      52  42  ...  rmn str inel
c      53  43  ...  rmn str diff
c      54  44  ...  rmn str inel with split (or after droplet or hadron split)
c      55  45  ...  rmn res
c      56  46  ...  rmn from split without connexion
c      57  47  ...  rmn res active spectators
c      58  48  ...  rmn res from diff
c      59  49  ...  hadron split
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      double precision plc,s   ,ptt1,ptt2,ptt3
      common/cems5/plc,s
      double precision tpro,zpro,ttar,ztar,ttaus,detap,detat,zor,tor
      common/cttaus/tpro,zpro,ttar,ztar,ttaus,detap,detat
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)
      double precision amasmin,amasini,xmdrmax,xmdrmin!,utpcmd
      integer icf(2),icb(2)
      integer jcf(nflav,2),jcval(nflav,2)!,jcdummy(nflav,2)
      logical gdrop, ghadr,gproj
      double precision ept(5),ep(4),aa(5),am2t,piq1,piq2,piq3
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      common /ems12/iodiba,bidiba  ! defaut iodiba=0. if iodiba=1, study H-Dibaryon
      character c*1,c1*1,c2*1

      call utpri('ProReF',ish,ishini,3)

      iretxx=0

      if(ir.ne.1.and.ir.ne.-1)stop'ProReF: wrong ir'

      irmdropx=irmdrop
 55   idrop=0
      gdrop=.false.
      ghadr=.false.
      iret=0
      dens=0.0765
      do j=1,2
        do i=1,nflav
          jcf(i,j)=0
        enddo
      enddo

      flow=1.
      if(ir.eq.1)then
c        if(kolp(m).le.0)goto1000
        if(iep(m).le.-1)goto1000
        gproj=.true.
        mm=npproj(m)
        iept=iep(m)
        zz=zzremn(m,1)
        iclpt=iclpro
        isopt=isoproj
        if(iremn.ge.2)then         !number of valence quarks still in proj
          if((iept.eq.3.or.iept.eq.5).and.yrmaxi.gt.1.e-5)
     &      flow=1./fradflii**2
          do nnn=1,nrflav
            jcval(nnn,1)=jcpval(nnn,1,m)
            jcval(nnn,2)=jcpval(nnn,2,m)
          enddo
          do nnn=nrflav+1,nflav
            jcval(nnn,1)=0
            jcval(nnn,2)=0
          enddo
        else
          do nnn=1,nflav
            jcval(nnn,1)=0
          enddo
          do nnn=1,nflav
            jcval(nnn,2)=0
          enddo
        endif
      elseif(ir.eq.-1)then
c        if(kolt(m).le.0)goto1000
        if(iet(m).le.-1)goto1000
        gproj=.false.
        mm=nptarg(m)
        iept=iet(m)
        zz=zzremn(m,2)
        iclpt=icltar
        isopt=isotarg
        if(iremn.ge.2)then         !number of valence quarks still in proj
          if((iept.eq.3.or.iept.eq.5).and.yrmaxi.gt.1.e-5)
     &      flow=1./fradflii**2
          do nnn=1,nrflav
            jcval(nnn,1)=jctval(nnn,1,m)
            jcval(nnn,2)=jctval(nnn,2,m)
          enddo
          do nnn=nrflav+1,nflav
            jcval(nnn,1)=0
            jcval(nnn,2)=0
          enddo
        else
          do nnn=1,nflav
            jcval(nnn,1)=0
          enddo
          do nnn=1,nflav
            jcval(nnn,2)=0
          enddo
        endif
      else
        call utstop('ProReF: ir ???&')
      endif
      if(ish.ge.3)
     &write(ifch,*)'remnant particle index:',mm,m,iclpt,isopt

      if(ish.ge.8)call alist('ProRef&',1,nptl)
      antotre=antotre+1.

      mmini=mm
      nptlini=nptl
      minfra=min(minfra,nptlini)   !for trigger condition

      do l=1,5
       ept(l)=dble(pptl(l,mm))
      enddo

      ifrptl(1,mm)=0
      ifrptl(2,mm)=0

c  initialize forward and backward ic (to transform remnant into string)

      if(gproj)then
        icf(1)=icproj(1,m)
        icf(2)=icproj(2,m)
        if(icf(1).eq.999999)then    !more than 9 quark : use jcpref
          do j=1,2
            do i=1,nrflav
              jcf(i,j)=jcpref(i,j,m)
            enddo
          enddo
        else
          call iddeco(icf,jcf)
        endif
      else                     !gtarg
        icf(1)=ictarg(1,m)
        icf(2)=ictarg(2,m)
        if(icf(1).eq.999999)then    !more than 9 quark : use jctref
          do j=1,2
            do i=1,nrflav
              jcf(i,j)=jctref(i,j,m)
            enddo
          enddo
        else
          call iddeco(icf,jcf)
        endif
      endif
      icb(1)=0
      icb(2)=0

      call idquacjc(jcf,nqu,naq)
c use RemoveHadron if too many c quarks
      if(nrflav.gt.3)then
        nqc=jcf(4,1)+jcf(4,2)
        if(nqu.lt.3.and.jcf(4,1).gt.1.or.
     &     naq.lt.3.and.jcf(4,2).gt.1.or.
     &             jcf(4,1)*jcf(4,2).gt.1 )nqc=4
      else
        nqc=0
      endif
      if(iremn.ge.2)then
        ier=0
        ires=0
        id=idtra(icf,ier,ires,0)
        if(ier.eq.0)then
          call idspin(id,ispin,jspin,istra)
        else
          ispin=0
          jspin=0
          istra=0
        endif
      endif

c define masses

      amasmin=dble(fremnux(jcf))**2.d0
      if(ept(5).le.0.d0)then
        ept(5)=dble(fremnux(jcf)*(1.+rangen()))
        if(ish.ge.2)then
          call utmsg('ProReF')
          write(ifch,*)'zero remnant mass -> amasmin'
          call utmsgf
        endif
      endif
      am2t=sqrt(ept(1)**2+ept(2)**2+ept(5)**2)
      if(iLHC.eq.1.and.ept(4).gt.am2t.and.(iept.eq.0.or.iept.eq.6))then
        ept(3)=sign(sqrt((ept(4)+am2t)*(ept(4)-am2t)),ept(3))
      else
        ept(4)=sqrt(ept(3)*ept(3)+ept(2)*ept(2)+ept(1)*ept(1)
     &           +ept(5)*ept(5))
      endif
      am2t=(ept(4)+ept(3))*(ept(4)-ept(3))-(ept(1)**2+ept(2)**2)
      if(ish.ge.2
     &   .and.(am2t.lt.-1d0.or.abs(am2t-ept(5)*ept(5)).gt.ept(5)))then
          write(ifch,*)'Precision problem in ProRef, p:',
     &             (ept(k),k=1,4),ept(5)*ept(5),am2t
      endif

      if(ish.ge.3)then
        if(gproj)then
            write(ifch,'(a,5e11.3,2i7)')' proj:'
     &      ,(sngl(ept(k)) ,k=1,5),(icproj(k,m) ,k=1,2)
        else    !gtarg
           write(ifch,'(a,5e11.3,2i7)')' targ:'
     &      ,(sngl(ept(k)) ,k=1,5),(ictarg(k,m),k=1,2)
         endif
      endif

      amasini=ept(5)*ept(5)

      xmdrmin=dble(fremnux(jcf)+amdrmin)**2
      xmdrmax=dble(fremnux(jcf)+amdrmax)**2


      if(ish.ge.4)write(ifch,*)'remnant masses:',am2t,amasini,amasmin
     &                ,xmdrmin,zz,iept

c.............................exotic ...................................

c      if(amasini.gt.amasmin.and.irmdropx.eq.1)then

c      if(.not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
      if((iept.eq.3.or.iept.eq.5.or.
c     &   (iept.eq.1.and.iremn.eq.3!.and.amasini.le.xmdrmin
c     &    .and.(jcf(4,1)+jcf(4,2).eq.0)).or.
     &   .not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
     &           .or.(nqu.eq.1.and.naq.eq.1))).and.nqc.le.3
     &    .and.amasini.gt.amasmin.and.irmdropx.eq.1)then

c      if((
c     &   .not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
c     &           .or.(nqu.eq.1.and.naq.eq.1)).or.
c     &   (iept.ne.0.and.iept.le.2.and.reminv/ept(5).gt.rangen()))
c     &    .and.amasini.gt.amasmin.and.irmdropx.eq.1)then

         !print*,'-------------------------------------------' !!!
         !print*,jcf
         !print*,icf,sqrt(amasini),sqrt(amasmin),sqrt(xmdrmin)  !!!
         !print*,nqu,naq                                      !!!
c charm not possible in droplet
        if(iremn.ge.2.or.
     &     (amasini.gt.xmdrmin.or.nqc.ne.0))then
          if(iremn.eq.2)then
         call getdropx(ir,iept,m,icf,jcf,jcval,zz,ept,aa
     &                                          ,gdrop,xmdrmax)
          else
         call getdroplet(ir,iept,icf,jcf,zz,ept,aa,gdrop,xmdrmax)
          endif
          !--------------------------------
          !emit a droplet, update the remnant string flavor and 5-momentum
          ! input
          !     ir ......... 1  projectile, -1  target remnant
          !     ept ........ remnant  5-momentum
          !     jcf ........ remnant jc
          ! output
          !     gdrop ...  .true. = successful droplet emission
          !                          jcf, ept ....... droplet  ic and 5-momentum
          !                          icf, a ......... remnant string jc and 5-momentum
          !               .false. = unsuccessful
          !                          jcf, ept .... unchanged,
          !                          emits hadrons instead of droplet
c         !                          considered as droplet jc and 5-momentum
          !-------------------------------------
        endif

c redefine energy and charm quarks in droplet
        amasini=ept(5)*ept(5)
        nqc=jcf(4,1)+jcf(4,2)
c use remove hadrons if droplet too heavy (should not happen) or charm
        if(amasini.gt.1e4.or.nqc.ne.0)goto 500

        !...........droplet
        !also in case of unsuccessful drop emission, then remnant = droplet !
        idrop=1
        nptl=nptl+1
        t=xorptl(4,mm)
        istptl(mm)=41
        ifrptl(1,mm)=nptl
        ifrptl(2,mm)=nptl
        tivptl(2,mm)=t
c            Remnant radius to have eps=dens GeV/fm3
        radptl(nptl)=(3.*sngl(ept(5))/4./pi/dens)**0.3333
        dezptl(nptl)=0.
        do l=1,5
          pptl(l,nptl)=sngl(ept(l))
        enddo
        if(gdrop)then
          idx=0
        else
          if(iLHC.eq.1)then
            idx=idtra(icf,0,0,0)
          else
            idx=idtra(icf,0,0,3)
          endif
        endif
        if(abs(idx).gt.100)then
         amx=sngl(ept(5))
         call idres(idx,amx,idrx,iadjx)
         idx=idrx
        else
         idx=0
        endif
        if(idx.eq.0)then
          istptl(nptl)=10
          call idenct(jcf,idptl(nptl)
     *    ,ibptl(1,nptl),ibptl(2,nptl),ibptl(3,nptl),ibptl(4,nptl))
          if(gproj)then
            ityptl(nptl)=40
          else  !gtarg
            ityptl(nptl)=50
          endif
        else
          istptl(nptl)=0
          idptl(nptl)=idx
          pptl(5,nptl)=amx
          pptl(4,nptl)=sqrt(amx*amx+pptl(1,nptl)*pptl(1,nptl)
     &       +pptl(2,nptl)*pptl(2,nptl)+pptl(3,nptl)*pptl(3,nptl))
          if(gproj)then
            ityptl(nptl)=45
            if(iept.eq.6)ityptl(nptl)=47
          else  !gtarg
            ityptl(nptl)=55
            if(iept.eq.6)ityptl(nptl)=57
          endif
        endif
        iorptl(nptl)=mm
        jorptl(nptl)=0
        ifrptl(1,nptl)=0
        ifrptl(2,nptl)=0
        xorptl(1,nptl)=xorptl(1,mm)
        xorptl(2,nptl)=xorptl(2,mm)
        xorptl(3,nptl)=xorptl(3,mm)
        xorptl(4,nptl)=t
        tivptl(1,nptl)=t
        call idtau(idptl(nptl),pptl(4,nptl),pptl(5,nptl),taugm)
        tivptl(2,nptl)=tivptl(1,nptl)+taugm*(-alog(rangen()))
        do l=1,4
          ibptl(l,nptl)=0
        enddo
        andropl=andropl+1
        if(ish.ge.3)write(ifch,*)'Proref,ept(5),id',ept(5),idptl(nptl)
        !print*,nptl,idptl(nptl),sngl(ept(5)),pptl(5,nptl)  !!!

        !..........remnant update
        if(gdrop)then  !drop emission: new remnant -> ept, icf
          idrop=0
          do l=1,5
            ept(l)=aa(l)
          enddo
          call iddeco(icf,jcf)
          call idquacjc(jcf,nqu,naq)
          if(iret.eq.1)call utstop('Pb in ProRef in strg+drop process&')
          !!!  print*,'new remnant:',icf,ept(5)    !!!
          nptl=nptl+1
          t=xorptl(4,mm)
          ifrptl(2,mm)=nptl
          do l=1,5
            pptl(l,nptl)=sngl(ept(l))
          enddo
          idptl(nptl)=idptl(mm)
          istptl(nptl)=40
          iorptl(nptl)=mm
          jorptl(nptl)=0
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=xorptl(1,mm)
          xorptl(2,nptl)=xorptl(2,mm)
          xorptl(3,nptl)=xorptl(3,mm)
          xorptl(4,nptl)=t
          tivptl(1,nptl)=t
          tivptl(2,nptl)=ainfin
          if(gproj)then
            ityptl(nptl)=40
          else   !gtarg
            ityptl(nptl)=50
          endif
          do l=1,4
            ibptl(l,nptl)=0
          enddo
        endif

        !........decay mini-droplet......
        mm=nptlini+1
        nptlb=nptl
        if(iabs(idptl(mm)).gt.10**8)then

          iret=0
          if(iorsdf.ne.3.or.pptl(5,mm).gt.100.
     &       .or.amasini.le.amasmin*flow)then      !decay here only if no fusion or large mass or mass too low for flow

          if(ish.ge.3)write(ifch,*)'Decay remnant droplet...'
          if(nptlb.gt.mxptl-10)call utstop('ProRef: mxptl too small&')

          if(ifrade.gt.0.and.ispherio.eq.0)then
            if(ioclude.eq.3.or.dble(pptl(5,mm)).lt.xmdrmin)then
              call hnbaaa(mm,iret)
            else
              call DropletDecay(mm,iret)!Decay remn
              iret=0
            endif
          endif
          if(iret.ne.1.and.nptl.ne.nptlb)then ! ---successful decay---
            istptl(mm)=istptl(mm)+1
            ifrptl(1,mm)=nptlb+1
            ifrptl(2,mm)=nptl
            t=tivptl(2,mm)
            x=xorptl(1,mm)+(t-xorptl(4,mm))*pptl(1,mm)/pptl(4,mm)
            y=xorptl(2,mm)+(t-xorptl(4,mm))*pptl(2,mm)/pptl(4,mm)
            z=xorptl(3,mm)+(t-xorptl(4,mm))*pptl(3,mm)/pptl(4,mm)
            do 21 n=nptlb+1,nptl
              iorptl(n)=mm
              jorptl(n)=0
              istptl(n)=0
              ifrptl(1,n)=0
              ifrptl(2,n)=0
              radius=0.8*sqrt(rangen())
              phi=2*pi*rangen()
              ti=t
              zi=z
              xorptl(1,n)=x + radius*cos(phi)
              xorptl(2,n)=y + radius*sin(phi)
              xorptl(3,n)=zi
              xorptl(4,n)=ti
              iioo=mm
              zor=dble(xorptl(3,iioo))
              tor=dble(xorptl(4,iioo))
c              call idquac(iioo,nq,ndummy1,ndummy2,jcdummy)
              r=rangen()
              tauran=-taurea*alog(r)
              call jtaix(n,tauran,zor,tor,zis,tis)
              tivptl(1,n)=amax1(ti,tis)
              call idtau(idptl(n),pptl(4,n),pptl(5,n),taugm)
              r=rangen()
              tivptl(2,n)=t+taugm*(-alog(r))
              ityptl(n)=ityptl(n)+1
              if(iept.eq.6)ityptl(n)=ityptl(n)+6
              radptl(n)=0.
              dezptl(n)=0.
              itsptl(n)=0
              rinptl(nptl)=-9999
   21       continue
            if(iabs(idptl(nptlb+1)).le.6) then
              call gakli2(0,0)
              if(ish.ge.1)write (ifmt,*)'string from drop:nptlb+1,nptl:'
     *                                 ,nptlb+1,nptl
              istptl(nptlb+1)=1
              do n=nptlb+2,nptl
                istptl(n)=20
                zpaptl(1,n)=0.
                zpaptl(2,n)=0.
              enddo
              call gakfra(0,iret)
              call gakli2(0,0)
            endif
            jerr(4)=jerr(4)+1
          elseif(ifrade.gt.0.and.ispherio.eq.0)then ! Unsuccessful decay
            jerr(5)=jerr(5)+1
            if(ish.ge.4)write(ifch,*)
     *         '***** Unsuccessful remnant cluster decay'
     *             ,' --> do RemoveHadrons instead.'
            mm=mmini
            nptl=nptlini
            irmdropx=0
            goto 55
          endif

          endif
        endif

        if(idrop.eq.1)goto 1000
        !successful drop decay, no additional string, nothing to do

      endif

c...............................................................

 500  mm=mmini
      if(gdrop)mm=nptlini+2
      istptl(mm)=41
      ifrptl(1,mm)=nptl+1

c........................remove hadrons.........................

      if(.not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
     &          .or.(nqu.eq.1.and.naq.eq.1)))then
        if(irmdropx.eq.irmdrop)then
          jerr(6)=jerr(6)+1
             !call utmsg('ProReF')
             !write(ifch,*)'***** condition for droplet treatment: '
             !write(ifch,*)'*****  amasini.gt.amasmin.and.irmdropx.eq.1 = '
             !*           ,amasini.gt.amasmin.and.irmdropx.eq.1
             !write(ifch,*)'***** amasini,amasmin,irmdropx:'
             !*                 ,amasini,amasmin,irmdropx
             !write(ifch,*)'***** nqu,naq:',nqu,naq
             !write(ifch,*)'***** call RemoveHadrons'
             !call utmsgf
        endif
       call RemoveHadrons(gproj,ghadr,m,mm,jcf,jcval,icf,ept,iret)
       if(iret.ne.0)then
         iretxx=1
         goto 1000
       endif
      endif

c........................ determine idr (0=string, else=resonance).......

      if(icf(1).eq.0.and.icf(2).eq.0)then
        id=110
      else
        if(iLHC.eq.1)then
          id=idtra(icf,0,0,0)
        else
          id=idtra(icf,0,0,3)
        endif
      endif
      idr=0
      am=sngl(ept(5))
      call idres(id,am,idr,iadj)
c      if(iabs(mod(idr,10)).le.2.and.idr.ne.0)then
c       id=idr
c      else
c       idr=0
c      endif                                !ckeck on-shell mass (see uti)
      if(iadj.ne.0.and.iept.gt.0.and.ept(5).gt.0.d0
     &     .and.(dabs((ept(4)+ept(3))*(ept(4)-ept(3))
     $           -ept(2)**2-ept(1)**2-dble(am)**2).gt.0.3d0))idr=0

      if(ish.ge.3)then
        write(ifch,'(a,5e11.3)')' updt:',(sngl(ept(k)) ,k=1,5)
        write(ifch,*)'            icf: ',icf,' idr: ',idr,' iept: ',iept
      endif

c      if(iept.eq.3)stop'ProReF: iept=3 ???'

c...........................................string...................
      if(iept.gt.0.and.iept.ne.6.and.idr.eq.0)then

        !... nqu of remainder string

        anstrg0=anstrg0+1
        if(gdrop)anstrg1=anstrg1+1

        call iddeco(icf,jcf)
        nqu=0
        nqv=0
        nav=0
        do l=1,nrflav
          nqu=nqu+jcf(l,1)-jcf(l,2)
          nqv=nqv+jcval(l,1)+jcval(l,2)
          nav=nav+jcval(l,2)
        enddo

c        if(zrminc.lt.0.)stop'ProReF: not supported any more.         '

        !......determine forward momentum ep


        am1=0.
        am2=0.
        ptt1=0d0
        ptt2=0d0
        if(iLHC.eq.1)then
          pt=ranptcut(1.)*ptfraqq
          if(pt.lt.0.5d0*ept(5))then
            phi=2.*pi*rangen()
            ptt1=dble(pt*cos(phi))
            ptt2=dble(pt*sin(phi))
          endif
          ptt3=dble(ir)*sqrt((0.5d0*ept(5))**2-ptt1*ptt1-ptt2*ptt2)
        else
          ptt3=dble(ir)*0.5d0*ept(5)
        endif

        ep(1)=ptt1
        ep(2)=ptt2
        ep(3)=ptt3
cc        ep(4)=0.5d0*ept(5)
        ep(4)=sqrt(ptt3*ptt3+ptt2*ptt2+ptt1*ptt1+dble(am1*am1))

c        if(abs(ept(3)).le.ptsend)then
c          phi=2.*pi*rangen()
c          theta=2.*pi*rangen()
c          ca=cos(theta)
c          sa=sin(theta)
c          call utroa2(dble(phi),dble(ca),dble(sa),0d0,ep(1),ep(2),ep(3))
c        endif
        call utlob2(-1,ept(1),ept(2),ept(3),ept(4),ept(5)
     *     ,ep(1),ep(2),ep(3),ep(4),25)


        xxx=min(1.,sngl(abs(ep(3)/ep(4))))
        qqs=sngl(ept(5)**2)

        !....determine forward and backward flavor icf, icb

        if(iremn.ge.2)then
          xm3val=9.
          xm2val=3.
          xm1val=1.
          ntryx=0
 33       xx1=0.
          xx2=0.
          xx3=0.
          del=1./(1.-alppar)
          if(nqv.eq.3)then
            xx1=min(1.,ranptcut(xm3val))
            xx2=min(1.,ranptcut(xm3val))
            xx3=min(1.,ranptcut(xm3val))
          elseif(nqv.eq.2)then
            xx1=min(1.,ranptcut(xm2val))
            xx2=min(1.,ranptcut(xm2val))
            xx3=rangen()**del
          elseif(nqv.eq.1)then
            xx1=min(1.,ranptcut(xm1val))
            xx2=rangen()**del
            xx3=rangen()**del
          else
            xx1=rangen()**del
            xx2=rangen()**del
            xx3=rangen()**del
          endif
          if(ntryx.lt.1000)then
            if(xx1+xx2+xx3.gt.1)goto 33
          else
            xx1=rangen()
            xx2=rangen()*(1.-xx1)
            xx3=rangen()*(1.-xx1-xx2)
          endif
          xx1=xxx*xx1
          xx2=xxx*xx2
          xx3=xxx*xx3
          piq1=0d0
          piq2=0d0
          piq3=0d0
          if(iept.eq.4)then
            ireminv=0       !no inversion for very low mass diffraction
          else
c inversion needed for inelatic remnant because of cascade (NA49)
            ireminv=1  
          endif
       if(nqu.eq.3)then      !---baryon---
          c="s"
          if(nqv.ge.1)c="v"
          iq1=idraflx(piq1,xx1,qqs,iclpt,jcf,jcval,1,isopt,c)
          c="s"
          if(nqv.ge.2)c="v"
          iq2=idraflx(piq2,xx2,qqs,iclpt,jcf,jcval,1,isopt,c)
          c="s"
          if(nqv.ge.3)c="v"
          iq3=idraflx(piq3,xx3,qqs,iclpt,jcf,jcval,1,isopt,c)
c rescale x to have heavier quark backward (neutron in ZEUS or lambda in NA49 not forward)
c          if(iept.eq.2)then
c            if(isopt.gt.0)then
c              xx1=xx1/float(iq1)
c              xx2=xx2/float(iq2)
c              xx3=xx3/float(iq3)
c            elseif(isopt.lt.0)then
c              if(iq1.lt.3)then
c                xx1=xx1/float(3-iq1)
c              else
c                xx1=xx1/float(iq1)
c              endif
c              if(iq2.lt.3)then
c                xx2=xx2/float(3-iq2)
c              else
c                xx2=xx2/float(iq2)
c              endif
c              if(iq3.lt.3)then
c                xx3=xx3/float(3-iq3)
c              else
c                xx3=xx3/float(iq3)
c              endif
c            endif
c          endif
          call neworderx(xx3,xx2,xx1,iq3,iq2,iq1)
          if(xx2-xx3.gt.reminv*(xx1-xx2))ireminv=0
c put always strange quarks in diquark (for lambda and cascade (NA49))
          if(iq3.ge.3.and.ireminv.eq.0)ireminv=1 !here inversion only in diffraction except for strange particles (lambda and cascade very central)
c if inversion for diffractive and inelastic
c          if(iq1+iq2.lt.6.and.iq3.ge.3.and.ireminv.eq.0)then
c            iqtmp=iq3
c            if(iq2.eq.3.or.(iq1.ne.3.and.rangen().gt.0.5))then
c              iq3=iq1
c              iq1=iqtmp
c            else
c              iq3=iq2
c              iq2=iqtmp
c            endif
c          endif
          if(ireminv.eq.0)then
            call uticpl(icf,iq3,2,iret) ! antiquark
            call uticpl(icb,iq3,1,iret) ! quark
          else
            call uticpl(icf,iq3,2,iret) ! antiquark
            call uticpl(icb,iq3,1,iret) ! quark
            call uticpl(icf,iq2,2,iret) ! antiquark
            call uticpl(icb,iq2,1,iret) ! quark
          endif
        elseif(nqu.eq.-3)then !---antibaryon---
          c="s"
          if(nqv.ge.1)c="v"
          iq1=idraflx(piq1,xx1,qqs,iclpt,jcf,jcval,2,isopt,c)
          c="s"
          if(nqv.ge.2)c="v"
          iq2=idraflx(piq2,xx2,qqs,iclpt,jcf,jcval,2,isopt,c)
          c="s"
          if(nqv.ge.3)c="v"
          iq3=idraflx(piq3,xx3,qqs,iclpt,jcf,jcval,2,isopt,c)
c rescale x to have heavier quark backward (neutron in ZEUS or lambda in NA49 not forward)
c          if(iept.eq.2)then
c            if(isopt.gt.0)then
c              xx1=xx1/float(iq1)
c              xx2=xx2/float(iq2)
c              xx3=xx3/float(iq3)
c            elseif(isopt.lt.0)then
c              if(iq1.lt.3)then
c                xx1=xx1/float(3-iq1)
c              else
c                xx1=xx1/float(iq1)
c              endif
c              if(iq2.lt.3)then
c                xx2=xx2/float(3-iq2)
c              else
c                xx2=xx2/float(iq2)
c              endif
c              if(iq3.lt.3)then
c                xx3=xx3/float(3-iq3)
c              else
c                xx3=xx3/float(iq3)
c              endif
c            endif
c          endif
          call neworderx(xx3,xx2,xx1,iq3,iq2,iq1)
          if(xx2-xx3.gt.reminv*(xx1-xx2))ireminv=0
c put always strange quarks in diquark
          if(iq3.ge.3.and.ireminv.eq.0)ireminv=1
c          if(iq1+iq2.lt.6.and.iq3.ge.3.and.ireminv.eq.0)then
c            iqtmp=iq3
c            if(iq2.eq.3.or.(iq1.ne.3.and.rangen().gt.0.5))then
c              iq3=iq1
c              iq1=iqtmp
c            else
c              iq3=iq2
c              iq2=iqtmp
c            endif
c          endif
          if(ireminv.eq.0)then
            call uticpl(icf,iq3,1,iret) ! quark
            call uticpl(icb,iq3,2,iret) ! antiquark
          else
            call uticpl(icf,iq1,1,iret) ! quark
            call uticpl(icb,iq1,2,iret) ! antiquark
            call uticpl(icf,iq2,1,iret) ! quark
            call uticpl(icb,iq2,2,iret) ! antiquark
          endif
        elseif(nqu.eq.0)then !---meson---
          xx3=0.    !no third quark
          iq3=0
          if(nqv.eq.2)then
            c1="v"
            c2="v"
            j=min(2,1+int(0.5+rangen()))
          elseif(nav.ne.0)then    !valence antiquark
            c1="v"
            c2="s"
            j=2
          elseif(nqv.ne.0)then    !valence quark
            c1="v"
            c2="s"
            j=1
          else                    !only sea quarks
            c1="s"
            c2="s"
            j=min(2,1+int(0.5+rangen()))
          endif
          iq1=idraflx(piq1,xx1,qqs,iclpt,jcf,jcval,j,isopt,c1)
          iq2=idraflx(piq2,xx2,qqs,iclpt,jcf,jcval,3-j,isopt,c2)
          if(xx1.gt.xx2)ireminv=0
         if(ireminv.eq.1)then
            call uticpl(icf,iq1,3-j,iret) ! subtract quark 1 forward
            call uticpl(icb,iq1,j,iret) ! add quark 1 backward
          else
            call uticpl(icf,iq2,j,iret) ! subtract antiquark 2 forward
            call uticpl(icb,iq2,3-j,iret) ! add antiquark 2 backward
          endif
        else
          call utmsg('ProReF')
          write(ifch,*)'***** neither baryon nor antibaryon nor meson.'
          write(ifch,*)'*****  number of net quarks:',nqu
          write(ifmt,*)'ProReF: no hadron; ',nqu,' quarks  --> redo'
          iretxx=1
          goto 1000
        endif
        if(ish.ge.3)write(ifch,'(a,2i3,3(i2,e13.6))')' inversion:',isopt
     &         ,ireminv,iq1,xx1,iq2,xx2,iq3,xx3
        else
        ireminv=0
        if(iept.ne.0)then
          if(rangen().lt.reminv)ireminv=1
        endif
        if(nqu.eq.3)then      !---baryon---
          iq=idrafl(iclpt,jcf,1,'v',1,iret)
          call uticpl(icf,iq,2,iret)       ! antiquark
          call uticpl(icb,iq,1,iret)       ! quark
          if(ireminv.eq.1)then
           iq=idrafl(iclpt,jcf,1,'v',1,iret)
           call uticpl(icf,iq,2,iret)       ! antiquark
           call uticpl(icb,iq,1,iret)       ! quark
          endif
        elseif(nqu.eq.-3)then !---antibaryon---
          iq=idrafl(iclpt,jcf,2,'v',1,iret)
          call uticpl(icf,iq,1,iret)       ! quark
          call uticpl(icb,iq,2,iret)       ! antiquark
          if(ireminv.eq.1)then
           iq=idrafl(iclpt,jcf,2,'v',1,iret)
           call uticpl(icf,iq,1,iret)       ! quark
           call uticpl(icb,iq,2,iret)       ! antiquark
          endif
        elseif(nqu.eq.0)then !---meson---
           iq1=idrafl(iclpt,jcf,1,'v',1,iret)
           iq2=idrafl(iclpt,jcf,2,'v',1,iret)
           if(rangen().gt.0.5)then
             call uticpl(icf,iq1,2,iret) ! subtract quark
             call uticpl(icb,iq1,1,iret) ! add quark
           else
             call uticpl(icf,iq2,1,iret) ! subtract antiquark
             call uticpl(icb,iq2,2,iret) ! add antiquark
           endif
c        elseif(nqu.eq.0)then !---meson---
c          if(iept.ne.1.and.iept.ne.6.and.rangen().lt.0.5)then
c           iq=idrafl(iclpt,jcf,1,'v',1,iret)
c           call uticpl(icf,iq,2,iret)       ! subtract quark
c           call uticpl(icb,iq,1,iret)       ! add quark
c          else
cc put quark in forward direction always for inelastic
c           iq=idrafl(iclpt,jcf,2,'v',1,iret)
c           call uticpl(icf,iq,1,iret)       ! subtract antiquark
c           call uticpl(icb,iq,2,iret)       ! add antiquark
c          endif
        else
          if(ish.ge.1)then
          call utmsg('ProReF')
          write(ifch,*)'***** neither baryon nor antibaryon nor meson.'
          write(ifch,*)'*****  number of net quarks:',nqu
          endif
          write(ifmt,*)'ProReF: no hadron; ',nqu,' quarks  --> redo'
          iretxx=1
          goto1000
        endif
      endif


        !..... forward string end

        nptl=nptl+1
        if(nptl.gt.mxptl)call utstop('ProRef: mxptl too small&')
        pptl(1,nptl)=sngl(ep(1))
        pptl(2,nptl)=sngl(ep(2))
        pptl(3,nptl)=sngl(ep(3))
        pptl(4,nptl)=sngl(ep(4))
        pptl(5,nptl)=am1 !0.
        istptl(nptl)=20
        iorptl(nptl)=mm
        if(.not.gdrop)istptl(mm)=41
        jorptl(nptl)=0
        if(.not.ghadr.and..not.gdrop)ifrptl(1,mm)=nptl
        ifrptl(2,mm)=nptl
        xorptl(1,nptl)=xorptl(1,mm)
        xorptl(2,nptl)=xorptl(2,mm)
        xorptl(3,nptl)=xorptl(3,mm)
        xorptl(4,nptl)=xorptl(4,mm)
        tivptl(1,nptl)=xorptl(4,nptl)
        tivptl(2,nptl)=xorptl(4,nptl)
        if(iLHC.eq.1)then
          idptl(nptl)=idtra(icf,0,0,0)
        else
          idptl(nptl)=idtra(icf,0,0,3)
        endif
        if(gproj)then
          if(iep(m).lt.1)stop'ProReF: iep(m)<1     '
          ityptl(nptl)=41+iep(m)  ! =42 =43 =44 =46 =47
          if(iep(m).eq.4)ityptl(nptl)=42
          if(gdrop.and.iep(m).ne.6)ityptl(nptl)=44
          if(ghadr)ityptl(nptl)=44
        else  !gtarg
          if(iet(m).lt.1)stop'ProReF: iet(m)<1     '
          ityptl(nptl)=51+iet(m)  !=52 =53 =54 =56 =57
          if(iet(m).eq.4)ityptl(nptl)=52
          if(gdrop.and.iet(m).ne.6)ityptl(nptl)=54
          if(ghadr)ityptl(nptl)=54
        endif
        itsptl(nptl)=1
        qsqptl(nptl)=qqs
        rinptl(nptl)=-9999
        !write(6,'(a,i9,$)')'     ',idptl(nptl) !======================
        zpaptl(1,nptl)=zz
        if(gproj)then
          zpaptl(2,nptl)=float(lproj(m))
c          zpaptl(2,nptl)=0.
c          if(lproj(m).ge.1)then
c            do l=1,lproj(m)
c              kpair=kproj(m,l)
c              itt=itarg(kpair)
c              zpaptl(2,nptl)=zpaptl(2,nptl)+zzremn(itt,2)
c            enddo
c          endif
        else  !gtarg
          zpaptl(2,nptl)=float(ltarg(m))
c          zpaptl(2,nptl)=0.
c          if(ltarg(m).ge.1)then
c            do l=1,ltarg(m)
c              kpair=ktarg(m,l)
c              ipp=iproj(kpair)
c              zpaptl(2,nptl)=zpaptl(2,nptl)+zzremn(ipp,1)
c            enddo
c          endif
        endif
        if(ish.ge.3)then
          write(ifch,'(a,5e11.3,$)')' kink:',(pptl(k,nptl),k=1,5)
          write(ifch,*)' id: ',idptl(nptl)
        endif
        !....... backward string end

        nptl=nptl+1
        if(nptl.gt.mxptl)call utstop('ProRef: mxptl too small&')
        pptl2=0.
        do i=1,3
         pptl(i,nptl)=sngl(ept(i)-ep(i))
         pptl2=pptl2+pptl(i,nptl)*pptl(i,nptl)
        enddo
        pptl(5,nptl)=am2 !0.
        pptl2=pptl2+pptl(5,nptl)*pptl(5,nptl)
        pptl(4,nptl)=sqrt(pptl2)
        pptl2=sngl(ept(4)-ep(4))
        if(ish.ge.1.and.abs(pptl2-pptl(4,nptl)).gt.max(0.1,
     &                                         0.1*abs(pptl2)))then
          write(ifmt,*)
     &    'Warning in ProRef: inconsistent backward string end energy !'
     &    ,pptl(4,nptl),pptl2,abs(pptl2-pptl(4,nptl)),am1,am2,ptt3,ep(4)
          if(ish.ge.2)write(ifch,*)
     &    'Warning in ProRef: inconsistent backward string end energy !'
     &    ,(pptl(kkk,nptl),kkk=1,4),pptl2,abs(pptl2-pptl(4,nptl))
        endif
        istptl(nptl)=20
        iorptl(nptl)=mm
        jorptl(nptl)=0
        ifrptl(2,mm)=nptl
        ifrptl(1,nptl)=0
        ifrptl(2,nptl)=0
        xorptl(1,nptl)=xorptl(1,mm)
        xorptl(2,nptl)=xorptl(2,mm)
        xorptl(3,nptl)=xorptl(3,mm)
        xorptl(4,nptl)=xorptl(4,mm)
        tivptl(1,nptl)=xorptl(4,nptl)
        tivptl(2,nptl)=xorptl(4,nptl)
        if(iLHC.eq.1)then
          idptl(nptl)=idtra(icb,0,0,0)
        else
          idptl(nptl)=idtra(icb,0,0,3)
        endif
        if(gproj)then
          ityptl(nptl)=41+iep(m)  ! =42 =43 =47
          if(iep(m).eq.4)ityptl(nptl)=42
          if(gdrop.and.iep(m).ne.6)ityptl(nptl)=44
          if(ghadr)ityptl(nptl)=44
        else  !gtarg
          ityptl(nptl)=51+iet(m)  !=52 =53 =57
          if(iet(m).eq.4)ityptl(nptl)=52
          if(gdrop.and.iet(m).ne.6)ityptl(nptl)=54
          if(ghadr)ityptl(nptl)=54
        endif
        itsptl(nptl)=1
        qsqptl(nptl)=qqs
        rinptl(nptl)=-9999
        !write(6,'(a,i9)')'     ',idptl(nptl)
        zpaptl(1,nptl)=0.
        zpaptl(2,nptl)=1.
        if(ish.ge.3)then
          write(ifch,'(a,5e11.3,$)')' kink:',(pptl(k,nptl),k=1,5)
          write(ifch,*)' id: ',idptl(nptl)
        endif

c............................no string = resonance...................
      else

        anreso0=anreso0+1
        if(gdrop)anreso1=anreso1+1

        nptl=nptl+1
        if(idr.ne.0)id=idr
        if(nptl.gt.mxptl)call utstop('ProRef: mxptl too small&')
        if(iept.eq.0.or.iept.eq.6)call idmass(id,am)
        idptl(nptl)=id
        pptl(1,nptl)=sngl(ept(1))
        pptl(2,nptl)=sngl(ept(2))
        am2t=sqrt(ept(2)*ept(2)+ept(1)*ept(1)+dble(am*am))
        if(iLHC.eq.1.and.ept(4).gt.am2t)then   !conserve value of E on not pz
          pptl(4,nptl)=sngl(ept(4))
          pptl(3,nptl)=sngl(sign(sqrt((ept(4)+am2t)*(ept(4)-am2t))
     &                          ,ept(3)))
        else
          pptl(3,nptl)=sngl(ept(3))
          pptl(4,nptl)=sngl(sqrt(ept(3)*ept(3)+am2t))
        endif
        pptl(5,nptl)=am
        istptl(nptl)=0
        iorptl(nptl)=mm
        if(.not.gdrop)istptl(mm)=41
        jorptl(nptl)=0
        if(.not.ghadr.and..not.gdrop)ifrptl(1,mm)=nptl
        ifrptl(2,mm)=nptl
        ifrptl(1,nptl)=0
        ifrptl(2,nptl)=0
        xorptl(1,nptl)=xorptl(1,mm)
        xorptl(2,nptl)=xorptl(2,mm)
        xorptl(3,nptl)=xorptl(3,mm)
        xorptl(4,nptl)=xorptl(4,mm)
        tivptl(1,nptl)=xorptl(4,nptl)
        call idtau(idptl(nptl),pptl(4,nptl),pptl(5,nptl),taugm)
        tivptl(2,nptl)=tivptl(1,nptl)+taugm*(-alog(rangen()))
        if(gproj)then
          ityptl(nptl)=45
          if(gdrop)then
            ityptl(nptl)=46
          elseif(iept.eq.6)then
            ityptl(nptl)=47
          elseif(iept.eq.2.or.iept.eq.4)then
c          elseif(iept.eq.2)then
            ityptl(nptl)=48
          elseif(ghadr)then
            ityptl(nptl)=49
          else
            mine=0
            mdif=0
            do l=1,lproj(m)
              kp=kproj(m,l)
              if(abs(itpr(kp)).eq.1)mine=1
              if(itpr(kp).eq.2)mdif=1
            enddo
            if(mine.eq.0.and.mdif.eq.1)ityptl(nptl)=48
          endif
        else   !gtarg
          ityptl(nptl)=55
          if(gdrop)then
            ityptl(nptl)=56
          elseif(iept.eq.6)then
            ityptl(nptl)=57
          elseif(iept.eq.2.or.iept.eq.4)then
c          elseif(iept.eq.2)then
            ityptl(nptl)=58
          elseif(ghadr)then
            ityptl(nptl)=59
          else
            mine=0
            mdif=0
            do l=1,lproj(m)
              kp=kproj(m,l)
              if(abs(itpr(kp)).eq.1)mine=1
              if(itpr(kp).eq.2)mdif=1
            enddo
            if(mine.eq.0.and.mdif.eq.1)ityptl(nptl)=58
          endif
        endif
        itsptl(nptl)=0
        qsqptl(nptl)=0.
        rinptl(nptl)=-9999

        if(ish.ge.3)write(ifch,'(a,5e10.3,i7)')' nucl:'
     *         ,(pptl(i,nptl),i=1,5),idptl(nptl)

      endif
c.......................................................................
c      print *,iep(1),iet(1),ityptl(nptl)
 1000 call utprix('ProReF',ish,ishini,3)
ctp060829        if(ityptl(nptl).gt.60)print*,ityptl(nptl)
      return

      end

c-----------------------------------------------------------------------
      subroutine RemoveHadrons(gproj,ghadr,m,mm,jcf,jcv
     &                        ,icf,ept,iret)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      integer jcf(nflav,2),jcv(nflav,2),icf(2)
      double precision aa(5),ept(5)
      logical ghadr,gproj
      common/ems6/ivp0,iap0,idp0,isp0,ivt0,iat0,idt0,ist0
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)

      iret=0

      if(iremn.ge.2)then
        if(gproj)then
          idrf=idp(m)
        else
          idrf=idt(m)
        endif
      else
        if(gproj)then
          idrf=idp0
        else
          idrf=idt0
        endif
      endif
      call idquacjc(jcf,nqu,naq)
      if(nqu.eq.naq.and.(nqu.le.2.or.idrf.eq.0))then
        nmes=nqu
        nmes=nmes-1             !string is aq-q
        nbar=0
      elseif(nqu.gt.naq)then
        nmes=naq
        nbar=(nqu-nmes)/3     !nbar baryons
        if(nmes.eq.0.or.idrf.eq.1)then
          nbar=nbar-1        !string is qq-q
        else
          nmes=nmes-1        !string is aq-q
        endif
      elseif(nqu.lt.naq)then
        nmes=nqu
        nbar=(naq-nmes)/3    !nbar antibaryons
        if(nmes.eq.0.or.idrf.eq.1)then
          nbar=nbar-1        !string is aqaq-aq
        else
          nmes=nmes-1        !string is aq-q
        endif
      else
        nbar=nqu/3
        nmes=nqu-3*nbar
        nbar=nbar+naq/3
        nbar=nbar-1             !string is qq-q or aqaq-aq
      endif
      if(ish.ge.5)
     &       write(ifch,*)'RemoveHadron part (nq,na,nb,nm,dq):'
     &                     ,nqu,naq,nbar,nmes,idrf
      if(nmes+nbar.gt.0)ghadr=.true.
c  remove mesons
       if(nmes.gt.0)then
          do mes=1,nmes
            !write(ifch,*)'remove meson',mes,' / ',nmes
            call gethadron(1,idd,aa,jcf,jcv,ept,gproj,iret)
            if(iret.ne.0)goto 1000
              nptl=nptl+1
              if(nptl.gt.mxptl)
     &             call utstop('RemoveHadrons: mxptl too small&')
              idptl(nptl)=idd
              do i=1,5
                pptl(i,nptl)=sngl(aa(i))
              enddo
              iorptl(nptl)=mm
              jorptl(nptl)=0
              if(mes.eq.1)then
                ifrptl(1,mm)=nptl
                ifrptl(2,mm)=nptl
              else
                ifrptl(2,mm)=nptl
              endif
              ifrptl(1,nptl)=0
              ifrptl(2,nptl)=0
              istptl(nptl)=0
              if(gproj)then
                ityptl(nptl)=49
                xorptl(1,nptl)=xproj(m)
                xorptl(2,nptl)=yproj(m)
                xorptl(3,nptl)=zproj(m)
              else   !gtarg
                ityptl(nptl)=59
                xorptl(1,nptl)=xtarg(m)
                xorptl(2,nptl)=ytarg(m)
                xorptl(3,nptl)=ztarg(m)
              endif
              xorptl(4,nptl)=xorptl(4,mm)
              tivptl(1,nptl)=xorptl(4,nptl)
              call idtau(idptl(nptl),pptl(4,nptl),pptl(5,nptl),taugm)
              tivptl(2,nptl)=tivptl(1,nptl)+taugm*(-alog(rangen()))
              qsqptl(nptl)=0.
c           deleted: after abstracting a meson,
c           check if the NEW remnant is a H-Dibaryon
          enddo
        endif
c remove (anti)baryons
        call idquacjc(jcf,nqu,naq)
        if(nbar.gt.0)then
          do nb=1,nbar
            !write(ifch,*)'remove baryon',nb,' / ',nbar
            prq=float(nqu/3)
            pra=float(naq/3)
            psum=prq+pra
            if(psum.gt.0.)then
              if(rangen()*psum.le.prq)then      !baryon
                call gethadron(2,idd,aa,jcf,jcv,ept,gproj,iret)
                nqu=nqu-3
              else                              !antibaryon
                call gethadron(3,idd,aa,jcf,jcv,ept,gproj,iret)
                naq=naq-3
             endif
            else
              iret=1
            endif
            if(iret.ne.0)goto 1000
              nptl=nptl+1
              if(nptl.gt.mxptl)
     &             call utstop('RemoveHadron: mxptl too small&')
              idptl(nptl)=idd
              do i=1,5
                pptl(i,nptl)=sngl(aa(i))
              enddo
              iorptl(nptl)=mm
              jorptl(nptl)=0
              if(nmes.eq.0.and.nb.eq.1)then
                ifrptl(1,mm)=nptl
                ifrptl(2,mm)=nptl
              else
                ifrptl(2,mm)=nptl
              endif
              ifrptl(1,nptl)=0
              ifrptl(2,nptl)=0
              istptl(nptl)=0
              if(gproj)then
                ityptl(nptl)=49
                xorptl(1,nptl)=xproj(m)
                xorptl(2,nptl)=yproj(m)
                xorptl(3,nptl)=zproj(m)
              else    !gtarg
                ityptl(nptl)=59
                xorptl(1,nptl)=xtarg(m)
                xorptl(2,nptl)=ytarg(m)
                xorptl(3,nptl)=ztarg(m)
              endif
              xorptl(4,nptl)=xorptl(4,mm)
              tivptl(1,nptl)=xorptl(4,nptl)
              call idtau(idptl(nptl),pptl(4,nptl),pptl(5,nptl),taugm)
              tivptl(2,nptl)=tivptl(1,nptl)+taugm*(-alog(rangen()))
              qsqptl(nptl)=0.
c             deleted: after abstracting a (anti)baryon,
c                                  check if the NEW remnant is a H-Dibaryon
          enddo
        endif
        call idenco(jcf,icf,iret)

 1000 return
      end

c------------------------------------------------------------------
         subroutine gethadron(imb,idf,a,jc,jcv,ep,gproj,iret)
c------------------------------------------------------------------
c       goal:  emit a hadron (imb= 1 meson, 2 baryon, 3 antibaryon)
c              update the remnant flavor and 5-momentum
c
c       idf ,a : hadron id and 5-momentum
c       gproj  : T  projectile, F  target remnant
c       jc, ep : remnant flavor and 5-momentum
c       iret   : in case of error, keep correct momentum in remnant
c                and lose the quarks of the (not) emitted hadron
c-----------------------------------------------------------------

        include 'epos.inc'
        include 'epos.incems'
        common/cems5/plc,s
        double precision s,plc
        double precision ep(5),a(5),re(5),p1(5)
        integer jc(nflav,2),jcv(nflav,2),jcini(nflav,2),jcvini(nflav,2)
     &    ,ifh(3),ic(2)
        common /ems12/iodiba,bidiba  ! defaut iodiba=0. if iodiba=1, study H-Dibaryon
        double precision ptm,qcm,u(3),utpcmd,ptt,phi,sxini,sxini0,strmas
     &                  ,ampt2dro,ampt2str,p5sq,amasex,drangen,xmaxrm
        logical gproj

        call utpri('gethad',ish,ishini,5)

        iret=0
        do i=1,5
          a(i)=0.d0
          re(i)=ep(i)
        enddo
        ic(1)=0
        ic(2)=0
        do j=1,2
          do i=1,nflav
            jcini(i,j)=jc(i,j)
            jcvini(i,j)=jcv(i,j)
          enddo
        enddo

        if(iremn.ge.2)then
          if(ish.ge.5)then
            write(ifch,*)'remnant flavor and 5-momentum:',jc
            write(ifch,*)'                              ',jcv
            write(ifch,*)'momentum :',ep,gproj,imb
          endif
          call idquacjc(jcvini,nqv,nav)
        else
          if(ish.ge.5)
     &       write(ifch,*)'remnant flavor and 5-momentum:',jc,ep,gproj
     &                                                          ,imb
          nqv=0
          nav=0
        endif
       !write(*,'(/a,5f8.3)')'p before: ',ep

        if(gproj)then
          iclpt=iclpro
        else
          iclpt=icltar
        endif

c boost remnant in rest frame
         if(ish.ge.6) write (ifch,*) 'on-shell check'
         do k=1,5
           p1(k)=ep(k)
         enddo
         p1(5)=(p1(4)-p1(3))*(p1(4)+p1(3))-p1(2)**2-p1(1)**2
         if(p1(5).gt.0d0.and.abs(p1(5)-ep(5)*ep(5)).lt.ep(5))then
           p1(5)=sqrt(p1(5))
         else
           if(ish.ge.1)write(ifch,*)'Precision problem in gethad, p:',
     &          (p1(k),k=1,5),ep(5)*ep(5)
           p1(5)=0d0
        endif

c       initial limits
        mamos=4
        ptm=p1(5)
        sxini0=ptm*ptm
        idf=0
c redo

        nredo=0
 777    continue
        nredo=nredo+1
        if(nredo.gt.1)then       !restore initial flavors
          ic(1)=0
          ic(2)=0
          do j=1,2
            do i=1,nflav
              jc(i,j)=jcini(i,j)
              jcv(i,j)=jcvini(i,j)
            enddo
          enddo
          if(iremn.ge.2)then
            call idquacjc(jcvini,nqv,nav)
          endif
          if(ish.ge.7)write(ifch,*)'Restore flavor',idf,jc
          idf=0
          if(ptm.eq.0.or.nredo.gt.20)then
            if(ish.ge.4)write(ifch,*)
     &         'Pb with hadron momentum in Gethad !'
            iret=1
          endif
        endif

c  get the id and mass of hadron, the remnant jc is updated
        iret2=0
          if(imb.eq.1)then              ! a meson
            j=1
            if(nqv.gt.0)then
              i=idraflz(jcv,j)
              jc(i,j)=jc(i,j)-1
              nqv=nqv-1
            else
              i=idrafl(iclpt,jc,j,'v',1,iret2)
              if(iLHC.eq.1.and.iret2.ne.0)goto 77
            endif
            ifq=i
            j=2
            if(nav.gt.0)then
              i=idraflz(jcv,j)
              jc(i,j)=jc(i,j)-1
              nav=nav-1
            else
              i=idrafl(iclpt,jc,j,'v',1,iret2)
              if(iLHC.eq.1.and.iret2.ne.0)goto 77
            endif
            ifa=i
c            write(ifch,*)'ici',ifq,ifa,jc,'| ',jcv
            ic(1)=10**(6-ifq)
            ic(2)=10**(6-ifa)
            ier=0
            idf=idtra(ic,ier,idum,0)
            if(ier.ne.0)then
              if(ifq.le.ifa)then
                idf=ifq*100+ifa*10
              else
                idf=-(ifq*10+ifa*100)
              endif
            endif
            call idmass(idf,amss)

          elseif(imb.eq.2)then            ! a baryon
            j=1
            do ik=1,3
              if(nqv.gt.0)then
                i=idraflz(jcv,j)
                jc(i,j)=jc(i,j)-1
                nqv=nqv-1
              else
                i=idrafl(iclpt,jc,j,'v',1,iret2)
              if(iLHC.eq.1.and.iret2.ne.0)goto 77
              endif
              ifh(ik)=i
              ic(j)=ic(j)+10**(6-i)
            enddo
            ier=0
            idf=idtra(ic,ier,idum,0)
            if(ier.ne.0)then
              call neworder(ifh(1),ifh(2),ifh(3))
              idf=ifh(1)*1000+ifh(2)*100+ifh(3)*10
              if(ifh(1).ne.ifh(2).and.ifh(2).ne.ifh(3)
     $             .and.ifh(1).ne.ifh(3))  idf=2130
              if(ifh(1).eq.ifh(2).and.ifh(2).eq.ifh(3))idf=idf+1
            endif
            call idmass(idf,amss)

          elseif(imb.eq.3)then           ! an antibaryon
            j=2
            do ik=1,3
              if(nav.gt.0)then
                i=idraflz(jcv,j)
                jc(i,j)=jc(i,j)-1
                nav=nav-1
              else
                i=idrafl(iclpt,jc,j,'v',1,iret2)
              if(iLHC.eq.1.and.iret2.ne.0)goto 77
              endif
              ifh(ik)=i
              ic(j)=ic(j)+10**(6-i)
            enddo
            ier=0
            idf=idtra(ic,ier,idum,0)
            if(ier.ne.0)then
              call neworder(ifh(1),ifh(2),ifh(3))
              idf=ifh(1)*1000+ifh(2)*100+ifh(3)*10
              if(ifh(1).ne.ifh(2).and.ifh(2).ne.ifh(3)
     $             .and.ifh(1).ne.ifh(3))  idf=2130
              if(ifh(1).eq.ifh(2).and.ifh(2).eq.ifh(3))idf=idf+1
              idf=-idf
            endif
            call idmass(idf,amss)
           else
            call utstop('This imb does not exist in gethad !&')
           endif

   77     if(iret2.ne.0)then
          write(ifmt,*)'warning in gethadron: imb=',imb,'  iclpt:',iclpt
          write(ifmt,*)'   jc: ',jc,'  j: ',j,'   (1=q,2=aq)  --> redo'
          call utmsg('gethad')
          write(ifch,*)'Not enough quark ??? ... redo event !'
          call utmsgf
          iret=1
          goto 1000
          endif

c fix pt
          amasex=dble(amss)
          strmas=dble(utamnz(jc,mamos))

          ptt=dble(ranpt()*alpdro(2))**2         !pt+pl
          if(iret.ne.0)ptt=min(ptt,sxini0)
          if(ptt.gt.sxini0)goto 777
          sxini=sqrt(sxini0-ptt)



          a(5)=amasex
          re(5)=sxini-a(5)
          if(re(5).lt.strmas)then
            call idquacjc(jc,nq,na)
            if(nq+na.le.3)then
              idtmp=idtra(ic,1,idum,0)
              amtmp=0.
              call idmass(idtmp,amtmp)
              if(re(5).lt.amtmp)then
                if(ish.ge.6)write(ifch,*)
     &           'Pb with initial mass in Gethad, retry',idf
     &       ,amasex,re(5),strmas,sxini,ptm,ptt,amtmp,idtmp,ic,iret
                if(iret.eq.0)then
                  goto 777
                else
                  if(ish.ge.6)write(ifch,*)
     &           'Continue with minimal mass for remnant',re(5)
     &                                                   ,amtmp
                  re(5)=amtmp
                endif
              else
                strmas=amtmp
              endif
            endif
          endif

          ampt2dro=amasex**2d0
          ampt2str=strmas**2d0

c two body decay
          iret2=0
          if(iret.eq.1)then
c If energy to small, then produce a new particle adding the needed missing energy (limited energy violation to avoid stop and corrected in utrescl)
            xmaxrm=a(5)*a(5)+re(5)*re(5)
            if(ptm*ptm-xmaxrm.lt.0d0)then
             ptm=1.1d0*sqrt(2.d0*abs(a(5))*abs(re(5))+xmaxrm)
             p1(5)=ptm
             p1(4)=sqrt(p1(3)*p1(3)+p1(2)*p1(2)+p1(1)*p1(1)+p1(5)*p1(5))
            endif
          endif
          if(ish.ge.6)write(ifch,*)'2 body decay',ptm,a(5),re(5),iret
          qcm=utpcmd(ptm,a(5),re(5),iret2)
          if(iret2.ne.0)then
            if(iret.eq.0)then
             goto 777
            else
c              call utstop('Problem with qcm in gethadron !&')
              if(ish.ge.1)then
                call utmsg('gethad')
                write(ifch,*)'Problem with qcm  ... redo event !'
                call utmsgf
              endif
              iret=1
              return
            endif
          endif
          u(3)=2.d0*drangen(qcm)-1.d0
          phi=2.d0*dble(pi)*drangen(u(3))
          u(1)=sqrt(1.d0-u(3)**2)*cos(phi)
          u(2)=sqrt(1.d0-u(3)**2)*sin(phi)
          if(u(3).ge.0d0)then          !send always hadron backward
            do j=1,3
              re(j)=qcm*u(j)
              a(j)=-re(j)
            enddo
          else
            do j=1,3
              a(j)=qcm*u(j)
              re(j)=-a(j)
            enddo
          endif

          re(4)=sqrt(qcm**2+re(5)**2)
          a(4)=sqrt(qcm**2+a(5)**2)

          if(ish.ge.6)write(ifch,*)'boost : ',qcm
     &      ,' and momentum in rest frame : ',re,a


c Fix re of remnant

c boost string in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,re(1),re(2),re(3),re(4),81)

         p5sq=(re(4)+re(3))*(re(4)-re(3))-(re(1)*re(1)+re(2)*re(2))
         if(p5sq.ge.ampt2str)then
           re(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with remnant mass -> retry'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2str
             write(ifch,*)'   momentum four vector:',(re(ii),ii=1,4)
           endif
           if(iret.eq.0)then
             goto 777
           else
             if(ish.ge.6)write(ifch,*)
     &            'Finish with minimal mass for remnant',re(5)
           endif
         endif

c Fix a of hadron

c boost hadron in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,a(1),a(2),a(3),a(4),82)

         p5sq=(a(4)+a(3))*(a(4)-a(3))-(a(1)**2.d0+a(2)**2.d0)
         if(abs(p5sq-ampt2dro).le.0.1)then
           a(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with hadron mass'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2dro
             write(ifch,*)'   momentum four vector:',(a(ii),ii=1,4)
           endif
           a(4)=sqrt(a(5)*a(5)+a(3)*a(3)+a(2)*a(2)+a(1)*a(1))
           if(ish.ge.6)write(ifch,*)'Fix E with M and P:',(a(ii),ii=1,5)
         endif


c        if(iret.eq.1)then      !If problem with momenta do not update remnant
c
c          if(ish.ge.4)
c     *    write(ifch,*)'no hadron emission in gethad'
c
c        else     !update the 3-momentum and energy of remnant: ep

          if(ish.ge.1.and.abs(ep(4)-re(4)-a(4)).gt.1.d-2*ep(4))then
            write(ifmt,*)'Pb with energy conservation in gethad'
            if(ish.ge.6)then
              write(ifch,*)'Pb with energy conservation :'
              write(ifch,*)'   p1_ini:',ep(1),'  p1:',re(1)+a(1)
              write(ifch,*)'   p2_ini:',ep(2),'  p2:',re(2)+a(2)
              write(ifch,*)'   p3_ini:',ep(3),'  p3:',re(3)+a(3)
              write(ifch,*)'   p4_ini:',ep(4),'  p4:',re(4)+a(4)
            endif
          endif

          do i=1,5
            ep(i)=re(i)
          enddo
          if(ish.ge.5)then
            write(ifch,*)'get hadron with id and 5-momentum:',idf, a
          endif

c        endif

        !do i=1,5
        !  sm(i)=ep(i)+a(i)
        !enddo
        !write(*,'(a,5f8.3,i5)')'p after:  ',sm,iret

c      ghost condition
c         if(abs((a(4)+a(3))*(a(4)-a(3))
c     $           -a(2)**2-a(1)**2-a(5)**2).gt.0.3
c     $      .and.  abs(1.-abs(a(3))/a(4)).gt.0.01)print*,iret,dd

c$$$        if(iodiba.eq.1)then  ! for H-dibaryon study ??????????
c$$$          call idenco(jc,ic,iret)
c$$$          if(ic(1).eq.222000.and.ic(2).eq.0)ep(5)=ep(5)-bidiba
c$$$        endif

        if(ish.ge.5)then
          write(ifch,*)'new remnant flavor and 5-momentum:',jc, ep,iret
        endif
        iret=0
c          write(ifmt,*)'get hadron with id and 5-momentum:',idf, a
c          write(ifmt,*)'new remnant flavor and 5-momentum:',jc, ep

 1000 call utprix('gethad',ish,ishini,5)

      return
      end



c------------------------------------------------------------------
         subroutine getdroplet(ir,iept,ic,jc,z,ep,a,pass,xmdrmax)
c------------------------------------------------------------------
c  emit a droplet, update the remnant string flavor and 5-momentum
c
c input
c       ir ........ 1  projectile, -1  target remnant
c       iept ...... particle excitation
c       ep ........ remnant  5-momentum
c       jc ........ remnant jc
c       z  ........ Z factor from splitting
c output
c       pass ...  .true. = successful droplet emission
c                            jc, ep ....... droplet  ic and 5-momentum
c                            ic, a ........ remnant string jc and 5-momentum
c                 .false. = unsuccessful
c                            jc, ep .... unchanged,
c                            considered as droplet jc and 5-momentum
c-----------------------------------------------------------------

        include 'epos.inc'
        include 'epos.incems'
        double precision ep(5),a(5),p1(5),re(5),eps,amasex,xmdrmax
        double precision xxx,rr,alp,p5sq,xmin,xmax,ampt2str
     &  ,sxini,strmas,xxxmax,xxxmin,ampt2dro,xmdrmaxi
        parameter(eps=1.d-20)
        integer jc(nflav,2),ic(2),icx(2)
        integer jcini(nflav,2),jcfin(nflav,2)
        logical pass
        common/cems5/plc,s
        double precision s,plc,ptm,qcm,u(3),utpcmd,ptt,drangen,phi

        call utpri('getdro',ish,ishini,4)

        iret=0
        iret2=0
        xmdrmaxi=min(50.d0,xmdrmax)
        pass=.true.
        idps=0
        idms=0
        do i=1,nflav
          jcini(i,1)=jc(i,1)
          jcini(i,2)=jc(i,2)
          jcfin(i,1)=0
          jcfin(i,2)=0
        enddo


        call idquacjc(jcini,nqu,naq)

        do i=1,5
          a(i)=0.d0
          re(i)=0.d0
        enddo
        npart=nqu+naq
        nqc=jcini(4,1)+jcini(4,2)

        if(ir.eq.1)then
           iclpt=iclpro
         else
           iclpt=icltar
         endif

         if(ish.ge.5)then
           write(ifch,10)'remnant flavor and 5-momentum:'
     &                    ,jc,ep,nqu,naq,nqc,iept
 10        format(a,/,'jc:',6i3,' |',6i3,/,'ep:',5(e10.3,1x),/,4i4)
         endif

c  get id of string ends, the remnant string jc is updated
         if(iremn.eq.3)then  !  remnant content=string content (droplet empty)

           do i=1,nflav
             jcfin(i,1)=jcini(i,1)
             jcfin(i,2)=jcini(i,2)
             jcini(i,1)=0
             jcini(i,2)=0
           enddo

         else

         if(npart.lt.3.and.ep(5).lt.xmdrmax.and.nqc.eq.0)then !light droplet with few quarks
            pass=.false.
            goto 1000
         elseif(npart.lt.3)then    !few quarks but heavy, add some quarks to extract a q-qbar string (should not exit directly because of the large mass)
           ifq=idrafl(iclpt,jcini,2,'r',3,iret2)
           if(nqu.eq.1.and.naq.eq.1)then
             idps=1
             idms=1
             nqu=2
             naq=2
           else
             call utstop('This should not happen (getdrop) !&')
           endif
         elseif(nqu.eq.2.and.naq.eq.2)then
           idps=1
           idms=1
         elseif(naq.eq.0)then
           idps=5
           idms=1
         elseif(nqu.eq.0)then
           idps=1
           idms=5
         else                 !There is enough q or aq to do qq-q string


           if(jcini(4,1)-jcini(4,2).eq.0)then !if c-cbar

             idps=1
             idms=1

           else

c One chooses the first q or aq

           rrr=rangen()
           npart=nqu+naq
           if(jcini(4,1)+jcini(4,2).ne.0)then !if some charm take it out
             if(jcini(4,1).ne.0)then
               idps=1
               nqu=nqu-1
             else
               idms=1
               naq=naq-1
             endif
           elseif(rrr.gt.float(naq)/float(npart))then
             idps=1
             nqu=nqu-1
           else
             idms=1
             naq=naq-1
           endif

c One chooses the second one

           rrr=rangen()
           npart=nqu+naq
           if(idps.eq.1.and.jcini(4,1).ne.0)then !if some charm take it out
             idps=5
           elseif(idms.eq.1.and.jcini(4,2).ne.0)then !if some charm take it out
             idms=5
           elseif(rrr.gt.float(naq)/float(npart))then
             if(idps.eq.1.and.nqu.ge.2)then
               idps=5
             else
               idps=1
             endif
           else
             if(idms.eq.1.and.naq.ge.2)then
               idms=5
             else
               idms=1
             endif
           endif

c If there is already 2 q or 2 aq as string end, we know that we need
c a third one to complete the string

           if(idps.eq.5)idms=1
           if(idms.eq.5)idps=1
           if(idps.eq.1.and.idms.ne.5)idms=1
           if(idms.eq.1.and.idps.ne.5)idps=1

         endif

         endif

         if(ish.ge.5)then
           write(ifch,*)'remnant string ends :',idps,idms
         endif

          if(idps.ne.5.and.idms.ne.5)then              ! q-aq string
            if(jcini(4,1).eq.1)then
              ifq=idrafl(iclpt,jcini,1,'c',1,iret)
            else
              ifq=idrafl(iclpt,jcini,1,'v',1,iret)
            endif
            if(jcini(4,1).eq.1)then
              ifa=idrafl(iclpt,jcini,2,'c',1,iret)
            else
              ifa=idrafl(iclpt,jcini,2,'v',1,iret)
            endif
            jcfin(ifq,1)=1
            jcfin(ifa,2)=1

          elseif(idps.eq.5)then                       ! qq-q string
            do ik=1,3
              if(jcini(4,1).ne.0)then
                i=idrafl(iclpt,jcini,1,'c',1,iret)
              else
                i=idrafl(iclpt,jcini,1,'v',1,iret)
              endif
              jcfin(i,1)=jcfin(i,1)+1
            enddo

          elseif(idms.eq.5)then                        !aqaq-aq string
            do ik=1,3
              if(jcini(4,2).ne.0)then
                i=idrafl(iclpt,jcini,2,'c',1,iret)
              else
                i=idrafl(iclpt,jcini,2,'v',1,iret)
              endif
              jcfin(i,2)=jcfin(i,2)+1
            enddo
          endif

          endif      !iremn=3

          if(iret.ne.0)call utstop('Not enough quark in getdro ???&')
          if(jcini(4,1)+jcini(4,2).ne.0)
     &         call utstop('There is sitll charm quark in getdro???&')

c string id

         call idenco(jcfin,icx,iret)
         if(iret.eq.1)then
           call utstop('Exotic flavor in getdroplet !&')
         endif


c boost remnant in rest frame
      if(ish.ge.6) write (ifch,*) 'on-shell check'
        do k=1,5
          p1(k)=ep(k)
        enddo
        p1(5)=(p1(4)-p1(3))*(p1(4)+p1(3))-p1(2)**2-p1(1)**2
        if(p1(5).gt.0d0.and.abs(p1(5)-ep(5)*ep(5)).lt.ep(5))then
          p1(5)=sqrt(p1(5))
        else
          if(ish.ge.2)write(ifch,*)'Precision problem in getdro, p:',
     &             (p1(k),k=1,5),ep(5)*ep(5)
          p1(5)=ep(5)
          p1(4)=sqrt(p1(3)*p1(3)+p1(2)*p1(2)+p1(1)*p1(1)+p1(5)*p1(5))
        endif
      if(ish.ge.6) write (ifch,*) 'boost vector:',p1

c limits for momenta

      mamod=4
      mamos=4
      fad=alpdro(1)
      if(iremn.eq.3)fad=fad*(1.+z*zdrinc)
      fad=max(1.5,fad)
      ptm=p1(5)
      amasex=dble(fad*utamnz(jcini,mamod))
      fas=2.
      if(iremn.eq.3)then
        id=idtra(icx,ier,ires,0)
        if(ier.eq.0)then
          call idmass(id,amass)           !minimum is particle mass
          strmas=dble(amass)
        else
          strmas=dble(fas*utamnz(jcfin,mamos))
        endif
      else
        strmas=dble(fas*utamnz(jcfin,mamos))
      endif


c redo

       nredo=0
 777   continue
       nredo=nredo+1
       if(nredo.eq.10)then
          amasex=1.5d0*dble(utamnz(jcini,mamod))
          if(iremn.ne.3)strmas=1.5d0*dble(utamnz(jcfin,mamos))
       elseif(nredo.gt.20)then
          !write(ifch,*)'nredo.gt.20 -> only drop'
         if(ish.ge.4)write(ifch,*)
     &     'Pb with string mass in Getdrop, continue with gethad'
          pass=.false.
         goto 1000
       endif

c fix pt

          sxini=ptm*ptm
          ptt=dble(ranpt()*alpdro(2))**2         !pt
          if(ptt.ge.sxini)goto 777
          sxini=sqrt(sxini-ptt)


          ampt2dro=amasex**2d0
          ampt2str=strmas**2d0
          if(ampt2dro.gt.xmdrmaxi)then
            xmdrmaxi=2d0*ampt2dro
c            write(ifmt,*)'Warning Mmin>Mmax in Getdroplet'
          endif

          xxxmax=min(xmdrmaxi,(sxini-strmas)**2)    !strmas/(strmas+ampt2)
          xxxmin=ampt2dro

          if(xxxmin.gt.xxxmax)then
            !write(ifch,*)'Warning Mmin>sxini -> only drop'
           if(ish.ge.4)write(ifch,*)
     &     'Pb with ampt2 in Getdrop, retry',nredo,ir
     &             ,ampt2dro,ampt2str,xxxmin,xxxmax,sxini,ptt,xmdrmaxi
            goto 777
          endif



c fix mass

            rr=drangen(xxxmax)
            xmax=xxxmax
            xmin=xxxmin
            alp=dble(alpdro(3))
            if(dabs(alp-1.d0).lt.eps)then
              xxx=xmax**rr*xmin**(1d0-rr)
            else
              xxx=(rr*xmax**(1d0-alp)+(1d0-rr)*xmin**(1d0-alp))
     &                                                **(1d0/(1d0-alp))
            endif


c        write(ifch,*)'ini',xmin,xxx,xmax,rr,ampt2dro
c    &                   ,(sxini-sqrt(xxx)),ampt2str,p1(5)



          re(5)=sqrt(xxx)
          a(5)=sxini-re(5)
          if(a(5).lt.strmas)then
            if(ish.ge.6)write(ifch,*)
     &           'Pb with initial mass in Getdrop, retry',ir
     &       ,xmin,xxx,xmax,rr,ampt2dro,ampt2str,a(5)
            goto 777
          endif


c two body decay
          if(ish.ge.6)write(ifch,*)'2 body decay',ptm,re(5),a(5)
          qcm=utpcmd(ptm,re(5),a(5),iret)
          u(3)=0.d0 !2.d0*drangen(qcm)-1.d0
          phi=2.d0*dble(pi)*drangen(u(3))
          u(1)=sqrt(1.d0-u(3)**2)*cos(phi)
          u(2)=sqrt(1.d0-u(3)**2)*sin(phi)
          if(u(3).lt.0d0)then          !send always droplet backward
c          if(u(3).gt.0d0)then          !send always droplet forward     ?????
            do j=1,3
              re(j)=qcm*u(j)
              a(j)=-re(j)
            enddo
          else
            do j=1,3
              a(j)=qcm*u(j)
              re(j)=-a(j)
            enddo
          endif

          re(4)=sqrt(qcm**2+re(5)**2)
          a(4)=sqrt(qcm**2+a(5)**2)

          if(ish.ge.6)write(ifch,*)'momentum in rest frame : ',re,a



c Fix a of string

c boost string in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,a(1),a(2),a(3),a(4),71)

         p5sq=(a(4)+a(3))*(a(4)-a(3))-(a(1)**2.d0+a(2)**2.d0)
         if(p5sq.gt.ampt2str)then
           a(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with string mass -> retry'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2str
             write(ifch,*)'   momentum four vector:',(a(ii),ii=1,4)
           endif
           goto 777
         endif

c Fix ep of droplet

c boost droplet in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,re(1),re(2),re(3),re(4),72)

         p5sq=(re(4)+re(3))*(re(4)-re(3))-(re(1)*re(1)+re(2)*re(2))
         if(p5sq.gt.ampt2dro)then
           re(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with droplet mass -> retry'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2dro
             write(ifch,*)'   momentum four vector:',(re(ii),ii=1,4)
           endif
           goto 777
         endif


       if(ish.ge.1.and.abs(ep(4)-re(4)-a(4)).gt.1.d-2*ep(4))then
         write(ifmt,*)'Pb with energy conservation in getdro'
         if(ish.ge.6)then
           write(ifch,*)'Pb with energy conservation :'
           write(ifch,*)'   p1_ini:',ep(1),'  p1:',re(1)+a(1)
           write(ifch,*)'   p2_ini:',ep(2),'  p2:',re(2)+a(2)
           write(ifch,*)'   p3_ini:',ep(3),'  p3:',re(3)+a(3)
         endif
       endif

c If OK, save flavors of droplet and string
         do i=1,5
           ep(i)=re(i)
         enddo
         ic(1)=icx(1)
         ic(2)=icx(2)
         do i=1,nflav
           jc(i,1)=jcini(i,1)
           jc(i,2)=jcini(i,2)
         enddo

         if(ish.ge.6)then
           write(ifch,20)'droplet:',jc,ep
           write(ifch,30)'string remnant:',ic,a
         endif
 20      format(a,/,'jc:',6i3,' |',6i3,/,'ep:',5(e10.3,1x))
 30      format(a,/,'ic:',i7,' |',i7,/,'a:',5(e10.3,1x))

 1000    continue
         call utprix('getdro',ish,ishini,4)
         end

c------------------------------------------------------------------
         subroutine getdropx(ir,iept,m,ic,jc,jcv,z,ep,a,pass,xmdrmax)
c------------------------------------------------------------------
c  emit a droplet taken into account momentum fraction without screening,
c  update the remnant string flavor and 5-momentum (to be used with iremn=2)
c
c input
c       ir ........ 1  projectile, -1  target remnant
c       iept ...... particle excitation
c       m  ........ remnant index
c       ep ........ remnant  5-momentum
c       jc ........ remnant jc
c       jcv ....... remnant jc valence quark
c       z  ........ Z factor from splitting
c output
c       pass ...  .true. = successful droplet emission
c                            jc, ep ....... droplet  ic and 5-momentum
c                            ic, a ........ remnant string jc and 5-momentum
c                 .false. = unsuccessful
c                            jc, ep .... unchanged,
c                            considered as droplet jc and 5-momentum
c-----------------------------------------------------------------

        include 'epos.inc'
        include 'epos.incems'
        double precision ep(5),a(5),p1(5),re(5),eps,amasex,xmdrmax
        double precision xxx,rr,alpm,p5sq,xmin,xmax,ampt2str,xmsmax
     &  ,sxini,strmas,xxxmax,xxxmin,ampt2dro,xmdrmaxi,xprmi,xmrmi
     &  ,xprmd,xmrmd,xprms,xmrms,xpti,ypti,xptd,yptd,xpts,ypts,xptt,yptt
        double precision om1xpr,xremd,xrems,freduc
     &       ,atil(ntymi:ntymx),btilp(ntymi:ntymx),btilpp(ntymi:ntymx)
        parameter(eps=1.d-20)
        integer jc(nflav,2),jcv(nflav,2),ic(2),icx(2)
        integer jcvini(nflav,2),jcini(nflav,2)
     &         ,jcfin(nflav,2),jcvfin(nflav,2)
        logical pass
        common/cems5/plc,s
        double precision s,plc,ptm,qcm,u(3),utpcmd,ptt,drangen,phi
        logical strcomp,valqu

        call utpri('getdrx',ish,ishini,4)

        iret=0
        xmdrmaxi=min(50.d0,xmdrmax)
        pass=.true.
        idps=0
        idms=0
        do i=1,nflav
          jcini(i,1)=jc(i,1)
          jcini(i,2)=jc(i,2)
          jcvini(i,1)=jcv(i,1)
          jcvini(i,2)=jcv(i,2)
          jcfin(i,1)=0
          jcfin(i,2)=0
          jcvfin(i,1)=0
          jcvfin(i,2)=0
        enddo


        call idquacjc(jcini,nqu,naq)
        call idquacjc(jcvini,nqv,nav)

        do i=1,5
          a(i)=0.d0
          re(i)=0.d0
        enddo
        nqc=jcini(4,1)+jcini(4,2)

        idrf=0
        if(nqu-naq.ne.0)idrf=1
        if(ir.eq.1)then
           iclpt=iclpro
           if(idrf.eq.0)idrf=idp(m)  !change it only if not 1
           xprmi=xpp(m)
           xmrmi=xmp(m)
           xpti=xxp(m)
           ypti=xyp(m)
           if(lproj3(m).gt.0)then
             nlnk=max(1,nint(z*float(lproj3(m))))
           else
             nlnk=0
           endif
        else
           iclpt=icltar
           if(idrf.eq.0)idrf=idt(m)  !change it only if not 1
           xprmi=xmt(m)
           xmrmi=xpt(m)
           xpti=xxt(m)
           ypti=xyt(m)
           if(ltarg3(m).gt.0)then
             nlnk=max(1,nint(z*float(ltarg3(m))))
           else
             nlnk=0
           endif
         endif

         if(ish.ge.5)then
           write(ifch,10)'remnant flavor and 5-momentum:'
     &                    ,jc,jcv,ep,nqu,naq,nqv,nav,nqc,idrf,iept,nlnk
 10        format(a,/,'jc:',6i3,' |',6i3,/,'jcv:',6i3,' |',6i3,/
     &            ,'ep:',5(e10.3,1x),/,8i4)
         endif

c check formation conditions

         strcomp=.false.
         valqu=.false.    !if true, valence quark will always be in strings : reduce lambda production
         if((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
     &          .or.(nqu.eq.1.and.naq.eq.1).and.nlnk.ne.0)then 
c not enough quark for the droplet, check mass
           if(iept.ne.5.and.ep(5)*ep(5).lt.xmdrmax.and.nqc.eq.0)then
             pass=.false.       !continue without droplet
             if(ish.ge.4)write(ifch,*)
     &      'Normal remnant in Getdropx, continue only with droplet ...'
             goto 1000
           endif
c create q-aq from sea (but no charm)
           do n=1,nlnk
             idum=idrafl(iclpt,jcini,1,'r',3,iret)
             nqu=nqu+1
             naq=naq+1
           enddo
           strcomp=.true.
           valqu=.false.
         elseif(mod(nqu-naq,3).ne.0)then
           call utstop('This should not happen (getdropx) !&')
         endif

c  get id of string ends, the remnant string jc is updated

c First remove all charm

         if(nqc.ne.0.and.jcini(4,1)-jcini(4,2).eq.0)then !if c-cbar

           if(jcini(4,1).eq.1)then
             idps=1
             idms=1
           else
             call utstop('getdropx can not manage more than c-cb !&')
           endif

         elseif(nqc.ne.0.and.jcini(4,1)*jcini(4,2).ne.0)then

           call utstop('getdropx can not manage c quarks this way !&')

         else


           if(nqc.ne.0)then !if some charm take it out
             if(jcini(4,1).ne.0)then
               if(nqu.lt.3)then
                 idrf=0         !can not use c in antibaryon
               elseif(jcini(4,1).gt.1)then
                 idrf=1         !more than 1 c quark only in baryon
               endif
             elseif(jcini(4,2).ne.0)then
               if(naq.lt.3)then
                 idrf=0         !can not use cb in baryon
               elseif(jcini(4,2).gt.1)then
                 idrf=1         !more than 1 c antiquark only in antibaryon
               endif
             endif
             if(idrf.ne.0.and.jcini(4,1).gt.0.and.jcini(4,1).le.3)then
               idps=5
               idms=1
         elseif(idrf.ne.0.and.jcini(4,2).gt.0.and.jcini(4,2).le.3)then
               idps=1
               idms=5
             elseif(jcini(4,1).gt.1.or.jcini(4,2).gt.1)then
               call utstop('getdropx can not use more than 3 c/cb !&')
             endif
           endif

c take into account number of diquark in final remnant string

           if(idps.eq.0)then

             if(idrf.ne.0)then  !use a diquark
               if(nqu.gt.naq)then        !qq-q
                 idps=5
                 idms=1
               elseif(nqu.lt.naq)then    !qbqb-qb
                 idps=1
                 idms=5
               endif
             else               !q-qb
               idps=1
               idms=1
             endif

           endif

         endif                  !string end type

         if(ish.ge.5)then
           write(ifch,*)'remnant string ends :',idps,idms
         endif

c choose flavor with priority to valence quark (after charm)

          if(idps.ne.5.and.idms.ne.5)then              ! q-aq string
            j=1
            if(jcini(4,j).gt.0)then
              i=4
              jcini(i,j)=jcini(i,j)-1
              if(jcvini(i,j).gt.0)then
                jcvfin(i,j)=jcvfin(i,j)+1
                jcvini(i,j)=jcvini(i,j)-1
                nqv=nqv-1
              endif
            elseif(valqu.and.nqv.gt.0)then
              i=idraflz(jcvini,j)
              jcvfin(i,j)=jcvfin(i,j)+1
              jcini(i,j)=jcini(i,j)-1
              nqv=nqv-1
            else
              i=idrafl(iclpt,jcini,j,'v',1,iret)
              if(jcini(i,j)-jcvini(i,j).lt.0)then
                jcvini(i,j)=jcvini(i,j)-1
                jcvfin(i,j)=jcvfin(i,j)+1
              endif
            endif
            ifq=i
            j=2
            if(jcini(4,j).gt.0)then
              i=4
              jcini(i,j)=jcini(i,j)-1
              if(jcvini(i,j).gt.0)then
                jcvfin(i,j)=jcvfin(i,j)+1
                jcvini(i,j)=jcvini(i,j)-1
                nav=nav-1
              endif
            elseif(valqu.and.nav.gt.0)then
              i=idraflz(jcvini,j)
              jcvfin(i,j)=jcvfin(i,j)+1
              jcini(i,j)=jcini(i,j)-1
              nav=nav-1
            else
              i=idrafl(iclpt,jcini,j,'v',1,iret)
              if(jcini(i,j)-jcvini(i,j).lt.0)then
                jcvini(i,j)=jcvini(i,j)-1
                jcvfin(i,j)=jcvfin(i,j)+1
              endif
            endif
            ifa=i
            jcfin(ifq,1)=1
            jcfin(ifa,2)=1

          elseif(idps.eq.5)then                       ! qq-q string
            j=1
            do ik=1,3
              if(jcini(4,j).ne.0)then
                i=4
                jcini(i,j)=jcini(i,j)-1
                if(jcvini(i,j).gt.0)then
                  jcvfin(i,j)=jcvfin(i,j)+1
                  jcvini(i,j)=jcvini(i,j)-1
                  nqv=nqv-1
                endif
              elseif(valqu.and.nqv.gt.0)then
                i=idraflz(jcvini,j)
                jcvfin(i,j)=jcvfin(i,j)+1
                jcini(i,j)=jcini(i,j)-1
                nqv=nqv-1
              else
                i=idrafl(iclpt,jcini,j,'v',1,iret)
                if(jcini(i,j)-jcvini(i,j).lt.0)then
                  jcvini(i,j)=jcvini(i,j)-1
                  jcvfin(i,j)=jcvfin(i,j)+1
                endif
              endif
              jcfin(i,j)=jcfin(i,j)+1
            enddo

          elseif(idms.eq.5)then                        !aqaq-aq string
            j=2
            do ik=1,3
              if(jcini(4,j).gt.0)then
                i=4
                jcini(i,j)=jcini(i,j)-1
                if(jcvini(i,j).gt.0)then
                  jcvfin(i,j)=jcvfin(i,j)+1
                  jcvini(i,j)=jcvini(i,j)-1
                  nav=nav-1
                endif
              elseif(valqu.and.nav.gt.0)then
                i=idraflz(jcvini,j)
                jcvfin(i,j)=jcvfin(i,j)+1
                jcini(i,j)=jcini(i,j)-1
                nav=nav-1
              else
                i=idrafl(iclpt,jcini,j,'v',1,iret)
                if(jcini(i,j)-jcvini(i,j).lt.0)then
                  jcvini(i,j)=jcvini(i,j)-1
                  jcvfin(i,j)=jcvfin(i,j)+1
                endif
              endif
              jcfin(i,j)=jcfin(i,j)+1
            enddo

          endif

          if(iret.ne.0)call utstop('Not enough quark in getdropx ???&')
          if(jcini(4,1)+jcini(4,2).ne.0)
     &         call utstop('There is sitll charm quark in getdropx???&')

c string id

         call idenco(jcfin,icx,iret)
         if(iret.eq.1)then
           call utstop('Exotic flavor in getdropx !&')
         endif


c boost remnant in rest frame
      if(ish.ge.6) write (ifch,*) 'on-shell check'
        do k=1,5
          p1(k)=ep(k)
        enddo
        p1(5)=(p1(4)-p1(3))*(p1(4)+p1(3))-p1(2)**2-p1(1)**2
        if(p1(5).gt.0d0.and.abs(p1(5)-ep(5)*ep(5)).lt.ep(5))then
          p1(5)=sqrt(p1(5))
        else
          if(ish.ge.2)write(ifch,*)'Precision problem in getdropx, p:',
     &             (p1(k),k=1,5),ep(5)*ep(5)
          p1(5)=ep(5)
          p1(4)=sqrt(p1(3)*p1(3)+p1(2)*p1(2)+p1(1)*p1(1)+p1(5)*p1(5))
        endif

c limits for momenta

      mamod=4
      mamos=4
      fad=alpdro(1)
      fad=max(1.5,fad*(1.+z*zdrinc))
      ptm=p1(5)
      amasex=dble(fad*utamnz(jcini,mamod))
      fas=2.
      strmas=dble(fas*utamnz(jcfin,mamos))
      ampt2dro=amasex**2d0
      ampt2str=strmas**2d0
      if(ampt2dro.gt.xmdrmaxi)then
        xmdrmaxi=2d0*ampt2dro
c       write(ifmt,*)'Warning Mmin>Mmax in Getdropx'
      endif


c redo

       xxxmin=1d0-xprmi
       if(xxxmin.gt.ampt2dro/(s*xmrmi))then
         xmrmd=ampt2dro/(s*xxxmin)
       else
         nlnk=0
       endif
       nredo=-1
       freduc=1d0
 777   continue
       nredo=nredo+1
       if(strcomp.and.nredo.eq.20)then  !after 19 try and remnant compatible with a string
         pass=.false.         !continue without droplet
         if(ish.ge.4)write(ifch,'(a,2i3,4e12.5)')
     &     'Pb with splitting in Getdropx, continue without split ...'
     &     ,nlnk,nvirt,xxxmax,xxxmin,ep(5)**2,xmdrmax
         goto 1000
       elseif(nredo.eq.10.or.nredo.eq.26)then        !reduce minimum mass
          amasex=1.5d0*dble(utamnz(jcini,mamod))
          strmas=1.5d0*dble(utamnz(jcfin,mamos))
          ampt2dro=amasex**2d0
          ampt2str=strmas**2d0
          xmrmd=ampt2dro/(s*xxxmin)
       elseif(nredo.eq.20)then    !after 19 try, use 2 body decay
         xmrmd=1d0+xmrmi
         if(ish.ge.4)write(ifch,*)
     &     'nredo>20, use 2 body decay ...',nvirt,xxxmax,xxxmin
         amasex=dble(fad*utamnz(jcini,mamod))
         strmas=dble(fas*utamnz(jcfin,mamos))
         ampt2dro=amasex**2d0
         ampt2str=strmas**2d0
         if(ish.ge.6) write (ifch,*) 'boost vector:',p1
       elseif(nredo.ge.30)then
          !write(ifch,*)'nredo.gt.20 -> only drop'
         if(ish.ge.4)write(ifch,*)
     &    'Pb with string mass in Getdropx, continue without split ...'
          pass=.false.
         goto 1000
       endif

       if(xmrmd.lt.xmrmi.and.nlnk.gt.0)then        !kinetic compatibility

         xmrms=xmrmi-xmrmd

c fix the virtual number of collision (no screening)
         iscreensave=iscreen
         iscreen=0
         imin=ntymin
         imax=ntymx
         if(iomega.eq.2)imax=1
         spp=sngl(s)
         nvirt=0
         xxxmax=0d0
         xpts=0d0
         ypts=0d0
         if(ir.eq.1)then
           do l=1,lproj3(m)    !use all pairs attached to remnant
             kp=kproj3(m,l)
             nvirt=nvirt+1
             do i=imin,imax
               call Gfunpar(0.,0.,1,i,bk(kp),spp,alp,bet,betp,epsp,epst
     &                                                       ,epss,gamv)
               atil(i)=dble(alp)
               btilp(i)=dble(bet)
               btilpp(i)=dble(betp)
             enddo
             xprms=1d0-xxxmax
c take x from an "unscreened" Pomeron (reduction factor if too high)
             xxx=om1xpr(atil,btilp,btilpp,xprms,xmrmi,ir)*freduc
             ptt=dble(ranptcut(1.)*alpdro(2))
c             ptt=dble(ranptcut(ptsems)*alpdro(2))
             phi=2d0*dble(pi)*drangen(ptt)
             xprms=1d0-xxxmax-xxx
             xptt=xpts+ptt*cos(phi)
             yptt=ypts+ptt*sin(phi)
             xrems=xprms*xmrms*s-(xpti+xptt)**2-(ypti+yptt)**2
             if(xrems.gt.ampt2str)then
               xxxmax=xxxmax+xxx
               xpts=xptt
               ypts=yptt
             endif
           enddo
         else
           do l=1,ltarg3(m)    !use all pairs attached to remnant
             kt=ktarg3(m,l)
             nvirt=nvirt+1
             do i=imin,imax
               call Gfunpar(0.,0.,1,i,bk(kt),spp,alp,bet,betp,epsp,epst
     &                                                       ,epss,gamv)
               atil(i)=dble(alp)
               btilp(i)=dble(bet)
               btilpp(i)=dble(betp)
             enddo
             xprms=1d0-xxxmax
c take x from an "unscreened" Pomeron (reduction factor if too high)
            xxx=om1xpr(atil,btilp,btilpp,xprms,xmrmi,ir)*freduc
             ptt=dble(ranptcut(1.)*alpdro(2))
c             ptt=dble(ranptcut(ptsems)*alpdro(2))
             phi=2d0*dble(pi)*drangen(ptt)
             xprms=1d0-xxxmax-xxx
             xptt=xpts+ptt*cos(phi)
             yptt=ypts+ptt*sin(phi)
             xrems=xprms*xmrms*s-(xpti+xptt)**2-(ypti+yptt)**2
             if(xrems.gt.ampt2str)then
               xxxmax=xxxmax+xxx
               xpts=xptt
               ypts=yptt
             endif
           enddo
         endif
         iscreen=iscreensave

         if(xxxmax.le.xxxmin)goto 777


c check string mass and energy

         xprms=1d0-xxxmax
         xpts=xpti+xpts
         ypts=ypti+ypts
         xrems=xprms*xmrms*s-xpts*xpts-ypts*ypts
         if(xrems.lt.ampt2str)then
           if(ish.ge.4)write(ifch,*)
     &          'Pb with string mass in Getdropx, retry',nredo,ir
     &          ,ampt2str,xrems,xprms,xmrms,xpts,ypts
           goto 777
         endif

c check droplet mass and energy

c Droplet mass should not exceed to much mdrmaxi. Use random to smooth distrib.
         xmsmax=xmdrmaxi*(1.+drangen(xmdrmaxi))
         xprmd=xprmi-xprms
         xptd=xpti-xpts
         yptd=ypti-ypts
         xremd=xprmd*xmrmd*s-xptd*xptd-yptd*yptd
         if(xremd.lt.ampt2dro)then
c Droplet should not have a mass too low.
           if(ish.ge.4)write(ifch,*)
     &          'Pb with drop mass (low) in Getdropx, retry',nredo,ir
     &          ,ampt2dro,xremd,xprmd,xmrmd,xptd,yptd
           goto 777
         elseif(xremd.ge.xmsmax)then
c Droplet should not have a mass too high.
           if(ish.ge.4)write(ifch,*)
     &          'Pb with drop mass (high) in Getdropx, retry',nredo,ir
     &          ,xremd,xmsmax,xprmd,xmrmd,xptd,yptd
           freduc=freduc*0.5d0
           goto 777
         endif


         re(1)=xptd
         re(2)=yptd
         if(ir.eq.1)then
           re(3)=(xprmd-xmrmd)*plc*0.5d0
         else
           re(3)=(xmrmd-xprmd)*plc*0.5d0
         endif
         re(4)=(xprmd+xmrmd)*plc*0.5d0
         re(5)=sqrt(xremd)

         a(1)=xpts
         a(2)=ypts
         if(ir.eq.1)then
           a(3)=(xprms-xmrms)*plc*0.5d0
         else
           a(3)=(xmrms-xprms)*plc*0.5d0
         endif
         a(4)=(xprms+xmrms)*plc*0.5d0
         a(5)=sqrt(xrems)



       else   !if xm to small, use two body decay (should be rare)

         if(ish.ge.6)write (ifch,*)'kinematic limit -> boost vector:',p1

c fix pt

          sxini=ptm*ptm
          ptt=dble(ranpt()*alpdro(2))**2         !pt
          if(ptt.ge.sxini)goto 777
          sxini=sqrt(sxini-ptt)



          xmsmax=xmdrmaxi*(1.+drangen(xmdrmaxi))
          xxxmax=min(xmsmax,(sxini-strmas)**2)    !strmas/(strmas+ampt2)
          xxxmin=ampt2dro

          if(xxxmin.gt.xxxmax)then
            !write(ifch,*)'Warning Mmin>sxini -> only drop'
           if(ish.ge.4)write(ifch,*)
     &     'Pb with ampt2 in Getdropx, retry',nredo,ir
     &             ,ampt2dro,ampt2str,xxxmin,xxxmax,sxini,ptt,xmsmax
            goto 777
          endif



c fix mass

            rr=drangen(xxxmax)
            xmax=xxxmax
            xmin=xxxmin
            alpm=dble(alpdro(3))
            if(dabs(alpm-1.d0).lt.eps)then
              xxx=xmax**rr*xmin**(1d0-rr)
            else
              xxx=(rr*xmax**(1d0-alpm)+(1d0-rr)*xmin**(1d0-alpm))
     &                                                **(1d0/(1d0-alpm))
            endif


c        write(ifch,*)'ini',xmin,xxx,xmax,rr,ampt2dro
c     &                   ,(sxini-sqrt(xxx)),ampt2str,p1(5)



          re(5)=sqrt(xxx)
          a(5)=sxini-re(5)
          if(a(5).lt.strmas)then
            if(ish.ge.6)write(ifch,*)
     &           'Pb with initial mass in Getdropx, retry',ir
     &       ,xmin,xxx,xmax,rr,ampt2dro,ampt2str,a(5)
            goto 777
          endif


c two body decay
          if(ish.ge.6)write(ifch,*)'2 body decay',ptm,re(5),a(5)
          qcm=utpcmd(ptm,re(5),a(5),iret)
          u(3)=2.d0*drangen(qcm)-1.d0
          phi=2.d0*dble(pi)*drangen(u(3))
          u(1)=sqrt(1.d0-u(3)**2)*cos(phi)
          u(2)=sqrt(1.d0-u(3)**2)*sin(phi)
          do j=1,3
            re(j)=qcm*u(j)
            a(j)=-re(j)
          enddo

          re(4)=sqrt(qcm**2+re(5)**2)
          a(4)=sqrt(qcm**2+a(5)**2)

          if(ish.ge.6)write(ifch,*)'momentum in rest frame : ',re,a



c Fix a of string

c boost string in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,a(1),a(2),a(3),a(4),73)

         p5sq=(a(4)+a(3))*(a(4)-a(3))-(a(1)**2.d0+a(2)**2.d0)
         if(p5sq.gt.ampt2str)then
           a(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with string mass -> retry'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2str
             write(ifch,*)'   momentum four vector:',(a(ii),ii=1,4)
           endif
           goto 777
         endif

c Fix ep of droplet

c boost droplet in collision frame
        call utlob2(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $       ,re(1),re(2),re(3),re(4),74)

         p5sq=(re(4)+re(3))*(re(4)-re(3))-(re(1)*re(1)+re(2)*re(2))
         if(p5sq.gt.ampt2dro)then
           re(5)=sqrt(p5sq)
         else
           if(ish.ge.6)then
             write(ifch,*)'Pb with droplet mass -> retry'
             write(ifch,*)'   m^2:',p5sq,'  m_min^2:',ampt2dro
             write(ifch,*)'   momentum four vector:',(re(ii),ii=1,4)
           endif
           goto 777
         endif

       endif     !test of xm


       if(ish.ge.1.and.abs(ep(4)-re(4)-a(4)).gt.1.e-2*ep(4))then
         write(ifmt,*)'Pb with energy conservation in getdropx'
         if(ish.ge.6)then
           write(ifch,*)'Pb with energy conservation :'
           write(ifch,*)'   p1_ini:',ep(1),'  p1:',re(1)+a(1)
           write(ifch,*)'   p2_ini:',ep(2),'  p2:',re(2)+a(2)
           write(ifch,*)'   p3_ini:',ep(3),'  p3:',re(3)+a(3)
         endif
       endif

c If OK, save flavors of droplet and string
         do i=1,5
           ep(i)=re(i)
         enddo
         ic(1)=icx(1)
         ic(2)=icx(2)
         do i=1,nflav
           jc(i,1)=jcini(i,1)
           jc(i,2)=jcini(i,2)
           jcv(i,1)=jcvfin(i,1)
           jcv(i,2)=jcvfin(i,2)
         enddo

         if(ish.ge.6)then
           write(ifch,20)'droplet:',jc,ep
           write(ifch,30)'string remnant:',ic,a
           write(ifch,'(a)')'valence:'
           write(ifch,'(6i3)')jcv
         endif
 20      format(a,/,'jc:',6i3,' |',6i3,/,'ep:',5(e10.3,1x))
 30      format(a,/,'ic:',i7,' |',i7,/,'a:',5(e10.3,1x))

 1000    continue
         call utprix('getdrx',ish,ishini,4)
         end

c-----------------------------------------------------
       subroutine neworder(n1, n2, n3)
c-----------------------------------------------------
c make 3 integers ordered like 1 2 3
c------------------------------------------------------
            if(n2.lt.n1)then
              ifb=n2
              n2=n1
              n1=ifb
            endif
            if(n3.lt.n1)then
              ifb=n3
              n3=n2
              n2=n1
              n1=ifb
            elseif(n3.lt.n2)then
              ifb=n3
              n3=n2
              n2=ifb
            endif
         end

c-----------------------------------------------------
       subroutine neworderx(x1,x2,x3,i1,i2,i3)
c-----------------------------------------------------
c make 3 reals ordered like 1 2 3
c------------------------------------------------------
            if(x2.lt.x1)then
              xfb=x2
              x2=x1
              x1=xfb
              ifb=i2
              i2=i1
              i1=ifb
            endif
            if(x3.lt.x1)then
              xfb=x3
              x3=x2
              x2=x1
              x1=xfb
              ifb=i3
              i3=i2
              i2=i1
              i1=ifb
            elseif(x3.lt.x2)then
              xfb=x3
              x3=x2
              x2=xfb
              ifb=i3
              i3=i2
              i2=ifb
            endif
         end

c-----------------------------------------------------------------------
      function idtr2(ic)
c-----------------------------------------------------------------------
c transforms ic to id such that only hadrons have nonzero id
c-----------------------------------------------------------------------
      parameter (nidt=30)
      integer idt(3,nidt),ic(2)
      data idt/
     * 100000,100000, 110   ,100000,010000, 120   ,010000,010000, 220
     *,100000,001000, 130   ,010000,001000, 230   ,001000,001000, 330
     *,100000,000100, 140   ,010000,000100, 240   ,001000,000100, 340
     *,000100,000100, 440
     *,300000,000000,1111   ,210000,000000,1120   ,120000,000000,1220
     *,030000,000000,2221   ,201000,000000,1130   ,111000,000000,1230
     *,021000,000000,2230   ,102000,000000,1330   ,012000,000000,2330
     *,003000,000000,3331   ,200100,000000,1140   ,110100,000000,1240
     *,020100,000000,2240   ,101100,000000,1340   ,011100,000000,2340
     *,002100,000000,3340   ,100200,000000,1440   ,010200,000000,2440
     *,001200,000000,3440   ,000300,000000,4441/

      idtr2=0
      if(ic(1).eq.0.and.ic(2).eq.0)then
       if(rangen().ge.0.5)then
        idtr2=110
        ic(1)=100000
        ic(2)=100000
       else
        idtr2=220
        ic(1)=10000
        ic(2)=10000
       endif
       return
      endif
      do 1 i=1,nidt
       if(ic(2).eq.idt(1,i).and.ic(1).eq.idt(2,i))idtr2=-idt(3,i)
       if(ic(1).eq.idt(1,i).and.ic(2).eq.idt(2,i))idtr2=idt(3,i)
1     continue
      return
      end

c----------------------------------------------------------------------
      subroutine emsini(e,idpji,idtgi)
c----------------------------------------------------------------------
c  energy-momentum sharing initializations
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      common/cemsr5/at(0:1,0:5)
      common/cems5/plc,s
      common/cems10/a(0:ntypmx),b(0:ntypmx),d(0:ntypmx)
      common/ems6/ivp0,iap0,idp0,isp0,ivt0,iat0,idt0,ist0
      double precision d,a,b,plc,s,amd,dcel,xvpr,xdm,at,xdm2
      common/ems3/dcel,ad
      common/cems13/xvpr(0:3)


c parameter test

      if(nflavems.lt.nrflav)
     &   call utstop("nflavems<nrflav : change it in epos-ems !&")


c abreviations

      plc=dble(e)
      s=plc**2
      amd=0.5d0   !dble(delrex) !(large enough in case of strangeness in string end


c alpha (0=0, 1=s, 2=v, 4=d, 8=f)

      a(0)=0d0
      a(1)=dble(alpsea)
      a(2)=dble(alpval)
      a(3)= 0.0d0
      a(4)=dble(alpdiq)
      a(5)=dble(a(4))
      a(6)= 0.0d0
      a(7)= 0.0d0
      a(8)=dble(a(2))
      a(9)= 0.0d0

c beta (0=0, 1=s, 2=v, 4=d, 8=f)

      b(0)=0.0d0
      b(1)=dble(-alpqua)
      b(2)=dble(-alpqua)
      b(3)=0.0d0
      b(4)=0.0d0
      b(5)=0.0d0
      b(6)=0.0d0
      b(7)=0.0d0
      b(8)=dble(-alpqua)
      b(9)=0.0d0


c alpha_trailing and beta_trailing (0=meson, 1=baryon;
c                                   0=no excit, 1=nondiffr, 2=diffr,
c                                   3=nondiffr split, 5=diffr split)

      at(0,0)=0.0d0
      at(0,1)=dble(alpndi)
      at(0,2)=dble(alpdi)
      at(0,3)=dble(alpdro(3))
      at(0,4)=10d0
      at(0,5)=dble(alpdro(3))
      at(1,0)=0.0d0
      at(1,1)=dble(alpndi)
      at(1,2)=dble(alpdi)
      at(1,3)=dble(alpdro(3))
      at(1,4)=10d0
      at(1,5)=dble(alpdro(3))

c minimal string masses ( i+j, each one: 0=0, 1=s, 2=v, 4=d, 5=d, 8=f)

      ammn(0)=0d0
      ammn(1)=0d0
      ammn(2)=dble(ammsqq)+amd
      ammn(3)=dble(ammsqq)
      ammn(4)=dble(ammsqq)
      ammn(5)=dble(ammsqd)+amd
      ammn(6)=dble(ammsqd)+amd
      ammn(7)=0d0
      ammn(8)=dble(ammsdd)+amd
      ammn(9)=dble(ammsqd)+amd
      ammn(10)=dble(ammsqd)+amd
      ammn(12)=dble(ammsqd)+amd
      ammn(16)=0.14d0

c minimal pomeron masses (0=soft or gg, 1=qg, 2=gq, 3=qq)

      amprmn(0)=ammsqq
      amprmn(1)=dsqrt(4d0*dble(q2min))
      amprmn(2)=amprmn(1)
      amprmn(3)=amprmn(1)

c cutoff for virtual pomeron (0=0, 1=soft Pom, 2=regge, 3=hard)

c      xvpr(0)=0d0
c      xvpr(1)=dble(cumpom**2)/s
c      xvpr(2)=dble(cumpom**2)/s
c      xvpr(3)=0.0d0**2/s

c minimal remnant masses (0=meson, 1=baryon)

      idpj=idpji
      xdm=0.35d0                  !<pt>
      call idmass(idpj,ampj)
      if(iabs(idpj).gt.1000)then
       ampmn(0)=0.14d0+xdm
       ampmn(1)=dble(ampj)+xdm
      else
       ampmn(0)=dble(ampj)+xdm
       ampmn(1)=0.94d0+xdm
      endif
      idtg=idtgi
      if(idtg.eq.0)idtg=1120
      call idmass(idtg,amtg)
      if(iabs(idtg).gt.1000)then
       amtmn(0)=0.14d0+xdm
       amtmn(1)=dble(amtg)+xdm
      else
       amtmn(0)=dble(amtg)+xdm
       amtmn(1)=0.94d0+xdm
      endif

c minimal excitation masses (0=meson, 1=baryon
c                            0=no excit, 1=nondiffr, 2=diffr,
c                                   6=nondiffr but no pomeron)

      xdm2=0.35d0
      amemn(0,0)=0.d0
      amemn(1,0)=0.d0
      amemn(0,4)=0.d0
      amemn(1,4)=0.d0
      amemn(0,6)=0.d0
      amemn(1,6)=0.d0

      amemn(0,1)=xdm2!+dble(delrex)
      amemn(0,2)=xdm2!+dble(delrex)
      amemn(0,3)=xdm2!+dble(delrex)
      amemn(0,5)=xdm2+dble(delrex) !remnant excited without connexion (split)

      amemn(1,1)=xdm2!+dble(delrex)
      amemn(1,2)=xdm2!+dble(delrex)
      amemn(1,3)=xdm2!+dble(delrex)
      amemn(1,5)=xdm2+dble(delrex) !remnant excited without connexion (split)

c maximal excitation masses (0=no excit, 1=nondiffr, 2=diffr)

      amemx(0)=2d0*xdm
      amemx(1)=plc
      amemx(2)=plc

      if(idpj.gt.1000)then     ! baryon

c initial quark configuration
       ivp0=3
       iap0=0
       idp0=1
       isp0=1

c no val quark for exotic projectile
       if(iremn.ge.2.and.(idpj.ne.1120.and.idpj.ne.1220))ivp0=0

      elseif(idpj.lt.-1000)then     ! antibaryon

c initial quark configuration
       ivp0=0
       iap0=3
       idp0=1
       isp0=1

c no val quark for exotic projectile
       if(iremn.ge.2.and.(idpj.ne.-1120.and.idpj.ne.-1220))iap0=0

      else      ! meson

c initial quark configuration
       ivp0=1
       iap0=1
       idp0=0
       if(iclpro.eq.1)then
         isp0=0
       else
         isp0=1
       endif

c no val quark for exotic projectile
       if(iremn.ge.2.and.(mod(abs(idpj/100),10).gt.4
     &                 .or.mod(abs(idpj/10),10).gt.4
     &    .or.mod(abs(idpj/100),10)/mod(abs(idpj/10),10).eq.1))then
         ivp0=0
         iap0=0
       endif
      endif

      if(idtg.gt.1000)then    ! baryon

c initial quark configuration
       ivt0=3
       iat0=0
       idt0=1
       ist0=0

c no val quark for exotic target
       if(iremn.ge.2.and.(idtg.ne.1120.and.idtg.ne.1220))ivt0=0

      elseif(idtg.lt.-1000)then   ! antibaryon

c initial quark configuration
       ivt0=0
       iat0=3
       idt0=1
       ist0=0

c no val quark for exotic target
       if(iremn.ge.2.and.(idtg.ne.-1120.and.idtg.ne.-1220))iat0=0

      else       ! meson

c initial quark configuration
       ivt0=1
       iat0=1
       if(icltar.eq.1)then
         idt0=0
       else
         idt0=1
       endif
       ist0=0

c no val quark for exotic target
       if(iremn.ge.2.and.(mod(abs(idtg/100),10).gt.4
     &                 .or.mod(abs(idtg/10),10).gt.4
     &    .or.mod(abs(idtg/100),10)/mod(abs(idtg/10),10).eq.1))then
         ivt0=0
         iat0=0
       endif

      endif


c eikonal parameters

       dcel=dble(chad(iclpro)*chad(icltar))

c counters

       antot=0.
       ansh=0.
       ansf=0.
       antotf=0.
       anshf=0.
       ansff=0.
       pp4max=0.
       pp4ini=0.
       andropl=0.
       anstrg0=0.
       anstrg1=0.
       anreso0=0.
       anreso1=0.
       anghadr=0.
       antotre=0.
       anintdiff=0.
       anintsdif=0.
       anintine=0.

      return
      end

c-----------------------------------------------------------------------
      subroutine emsigr
c-----------------------------------------------------------------------
c initialize grid
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)

      call utpri('emsigr',ish,ishini,5)

      do k=1,koll  !----k-loop---->

c determine length of k-th line of grid

       o=max(1.e-5,min(sngl(om1intc(k)),float(npommx)))!if GFF used for propo
        if(ish.ge.7)write(ifch,*)'emsigr:k,o',k,o
       n=0
       if(o.le.50)then
         p=1./(exp(o)-1)
       else
         p=0.
       endif
10     n=n+1
       p=p*o/n
        if(ish.ge.7)write(ifch,*)'emsigr:n,p',n,p
       if((p.gt.1e-4.or.n.lt.int(o)).and.n.lt.npommx
     *.and.n.lt.nprmax)goto 10

       if(ish.ge.5)write(ifch,*)'emsigr:nmax,b',n,bk(k)

       npr(0,k)=n
       nprmx(k)=n
       nprt(k)=0
       do i=1,3
        npr(i,k)=0
       enddo


c initial value for interaction type

       itpr(k)=0

c initial value for nuclear splitting

       do ir=1,2
         knucnt(ir,k)=0
         do ncon=1,mamx
           npnuc(ncon,ir,k)=0
           irnuc(ncon,ir,k)=0
           xxnuc(ncon,ir,k)=0d0
         enddo
       enddo

c initialize grid


       do n=1,nprmx(k)
        idpr(n,k)=0
        idfpr(n,k)=0
        ivpr(n,k)=1
        nppr(n,k)=0
        nbkpr(n,k)=0
        nvpr(n,k)=0
        idsppr(n,k)=0
        idstpr(n,k)=0
        idrpr(n,k)=0
        idhpr(n,k)=0
        bhpr(n,k)=0.
        xpr(n,k)=0d0
        ypr(n,k)=0d0
        xppr(n,k)=0d0
        xmpr(n,k)=0d0
        xp1pr(n,k)=0d0
        xp2pr(n,k)=0d0
        xm1pr(n,k)=0d0
        xm2pr(n,k)=0d0
        xp1pr(n,k)=0d0
        xp2pr(n,k)=0d0
        xm1pr(n,k)=0d0
        xm2pr(n,k)=0d0
        idp1pr(n,k)=0
        idp2pr(n,k)=0
        idm1pr(n,k)=0
        idm2pr(n,k)=0
        xxp1pr(n,k)=0d0
        xyp1pr(n,k)=0d0
        xxp2pr(n,k)=0d0
        xyp2pr(n,k)=0d0
        xxm1pr(n,k)=0d0
        xym1pr(n,k)=0d0
        xxm2pr(n,k)=0d0
        xym2pr(n,k)=0d0
       enddo

      enddo !  <----k-loop-----

      call utprix('emsigr',ish,ishini,5)
      return
      end

c-----------------------------------------------------------------------
      subroutine emsipt
c-----------------------------------------------------------------------
c initialize projectile and target
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      common/cems5/plc,s
      common/ems3/dcel,ad
      common/ems6/ivp0,iap0,idp0,isp0,ivt0,iat0,idt0,ist0
      common /cncl/xproj(mamx),yproj(mamx),zproj(mamx)
     *            ,xtarg(mamx),ytarg(mamx),ztarg(mamx)

      double precision dcel,s,plc

c initialize projectile

      do i=1,maproj
       idp(i)=idp0
       ivp(i)=ivp0+iap0
       iap(i)=iap0
       isp(i)=isp0
       iep(i)=-1
       ifp(i)=0
       kolp(i)=0
       npp(i)=0
       npproj(i)=0
       xxp(i)=0d0
       xyp(i)=0d0
       xpmn(i)=(amemn(idp(i),0)+ampmn(isp(i)))**2/s
       xpmx(i)=dmin1(1d0,(amemx(0)+ampmn(isp(i)))**2/s)
       xpos(i)=0.9d0*(amemx(0)+ampmn(isp(i)))**2/s
       xppmx(i)=0.5d0/(1d0+1d0/dble(maproj)**0.3d0)!1d0-dsqrt(xpmn(i))/maproj
       xmpmx(i)=0.5d0/(1d0+1d0/dble(matarg)**0.3d0)!1d0-dsqrt(xpmn(i))/matarg
       xmpmn(i)=xpmn(i)/xppmx(i)
       xppmn(i)=xpmn(i)/xmpmx(i)
       xpp(i)=1d0
       xmp(i)=0d0
       xppst(i)=0.d0
       xmpst(i)=0.d0
       xposst(i)=0.d0
      enddo

c initialize target

      do j=1,matarg
       idt(j)=idt0
       ivt(j)=ivt0+iat0
       iat(j)=iat0
       ist(j)=ist0
       iet(j)=-1
       ift(j)=0
       kolt(j)=0
       npt(j)=0
       nptarg(j)=0
       xxt(j)=0d0
       xyt(j)=0d0
       xtmn(j)=(amemn(idt(j),0)+amtmn(ist(j)))**2/s
       xtmx(j)=dmin1(1d0,(amemx(0)+amtmn(ist(j)))**2/s)
       xtos(j)=0.9d0*(amemx(0)+amtmn(ist(j)))**2/s
       xmtmx(j)=0.5d0/(1d0+1d0/dble(matarg)**0.3d0)!1d0-dsqrt(xtmn(j))/matarg
       xptmx(j)=0.5d0/(1d0+1d0/dble(maproj)**0.3d0)!1d0-dsqrt(xtmn(j))/maproj
       xptmn(j)=xtmn(j)/xmtmx(j)
       xmtmn(j)=xtmn(j)/xptmx(j)
       xmt(j)=1d0
       xpt(j)=0d0
       xmtst(j)=0.d0
       xptst(j)=0.d0
       xtosst(j)=0.d0
      enddo

      return
      end


c-----------------------------------------------------------------------
      subroutine emszz
c-----------------------------------------------------------------------
c     completes /cptl/ for nucleons, checks for no interaction
c     writes   /cevt/
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common/nucl3/phi,bimp
      common/col3/ncol,kolpt
      integer kolpz(mamx),koltz(mamx)

      call utpri('emszz ',ish,ishini,6)

c     write /cptl/
c     ------------

      if(iokoll.eq.1)then   ! precisely matarg collisions

c nothing to do
        ntg=0
        npj=0
        ncoli=0

      else

c determine ncol

       ncolx=ncol
       ncol=0
       ncoli=0
       do 8 k=1,koll
       if(ish.ge.7)write(ifch,*)'k,itpr,ncol,ncolx',k,itpr(k),ncol,ncolx
        if(itpr(k).eq.0)goto 8
        if(abs(itpr(k)).eq.1)ncoli=ncoli+1
          ncol=ncol+1
          if(itpr(k).ne.3)then          !empty pair, remnant not modified
            i=iproj(k)
            j=itarg(k)
            istptl(i)=1
            iorptl(i)=-1
            tivptl(2,i)=coord(4,k)
            istptl(maproj+j)=1
            iorptl(maproj+j)=-1
            tivptl(2,maproj+j)=coord(4,k)
          endif
8      continue
       if(ncolx.ne.ncol)write(6,*)'ncolx,ncol:', ncolx,ncol
       if(ncolx.ne.ncol)call utstop('********ncolx.ne.ncol********&')
       if(ncol.eq.0)goto1001

c determine npj, ntg

       do ip=1,maproj
        kolpz(ip)=0
       enddo
       do it=1,matarg
        koltz(it)=0
       enddo
      do k=1,koll
       if(itpr(k).ne.0.and.itpr(k).ne.3)then
        ip=iproj(k)
        it=itarg(k)
        kolpz(ip)=kolpz(ip)+1
        koltz(it)=koltz(it)+1
       endif
      enddo
      npj=0
      do ip=1,maproj
       if(kolpz(ip).gt.0.or.iep(ip).ge.3)npj=npj+1
      enddo
      ntg=0
      do it=1,matarg
       if(koltz(it).gt.0.or.iet(it).ge.3)ntg=ntg+1
      enddo
c     write(6,*)'npj,ntg,npj+ntg:',npj,ntg,npj+ntg

       endif

c     write /cevt/
c     ------------

      nevt=1
      bimevt=bimp
      phievt=phi
      kolevt=ncol
      koievt=ncoli
      kohevt=0      !not yet defined
      npjevt=npj
      ntgevt=ntg
      pmxevt=pnll
      egyevt=engy
      !print*,' ===== ',kolevt,koievt' ====='

c     exit
c     ----

      if(ish.ge.7)then
      do n=1,nptl
      write(ifch,115)iorptl(n),jorptl(n),n,istptl(n)
     *,tivptl(1,n),tivptl(2,n)
      enddo
  115 format(1x,'/cptl/',2i6,2i10,2(e10.3,1x))
      endif

1000  continue
      call utprix('emszz ',ish,ishini,6)
      return

1001  continue
      if(ish.ge.3)then
      write(ifch,*)
      write(ifch,*)'   ***** no interaction!!!'
      write(ifch,*)'   ***** ncol=0 detected in emszz'
      write(ifch,*)
      endif
      goto 1000

      end

c-----------------------------------------------------------------------
      subroutine ProCop(i,ii)
c-----------------------------------------------------------------------
c Propose Coordinates of remnants from active projectile nucleons
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      double precision xmptmp,aproj
      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      integer icrmn(2),jc(nflav,2),icini(2)
      double precision s,plc

      nptl=nptl+1
      npproj(i)=nptl
      idptl(nptl)=idptl(ii)*100+99  !100*10**idp(i)+iep(i)
      istptl(nptl)=40
      ityptl(nptl)=40
      iorptl(nptl)=ii
      jorptl(nptl)=0
      ifrptl(1,nptl)=0
      ifrptl(2,nptl)=0
      do j=1,2
        do k=1,nflav
          jc(k,j)=0
        enddo
      enddo

      istptl(ii)=1

c     determine kolz

      if(lproj(i).gt.1)then
        zmax=-ainfin
        kolz=0
        do l=1,lproj(i)
          k=kproj(i,l)
          z=coord(3,k)
          if(itpr(k).ne.0.and.z.gt.zmax)then
            zmax=z
            kolz=k
          endif
        enddo
      else
        kolz=1
      endif
c      if(kolz.eq.0)call utstop(' kolz=0 (proj)&')
      if(kolz.eq.0)then
        t=0.
      else
        t=coord(4,kolz)
      endif

      xorptl(1,nptl)=xorptl(1,ii)
      xorptl(2,nptl)=xorptl(2,ii)
      xorptl(3,nptl)=xorptl(3,ii)
      xorptl(4,nptl)=t
      tivptl(1,nptl)=t
      tivptl(2,nptl)=t
      naq=0
      nqu=0

      if(iremn.ge.2)then   !update icproj
        idp(i)=min(1,abs(idp(i)))
        k=1
        nqu=0
        do n=1,nrflav
          jc(n,k)=jcpref(n,k,i)
          nqu=nqu+jc(n,k)
        enddo
        k=2
        naq=0
        do n=1,nrflav
          jc(n,k)=jcpref(n,k,i)
          naq=naq+jc(n,k)
        enddo
        isum=nqu+naq
        call idenco(jc,icrmn,iret)
        if(iret.eq.0.and.(isum.le.3.or.iremn.ne.3))then
          icproj(1,i)=icrmn(1)
          icproj(2,i)=icrmn(2)
        elseif(iremn.eq.3)then
      write(ifch,*)'Problem in projectile flavor :',i,' ->',jc,' :',isum
          call utstop('Procop: Problem in projectile flavor !&')
        else     !for iremn=2 and large number of quark define icproj=999999
          icproj(1,i)=999999
          icproj(2,i)=999999
        endif
      endif

      icrmn(1)=icproj(1,i)
      icrmn(2)=icproj(2,i)

      if(iremn.ge.1)then      !excited remnant ?
        call idtr4(idptl(ii),icini)
        if(ish.ge.5)write(ifch,*)'Procop icini proj',i,icini,' ->',icrmn
        if((icrmn(1)-icini(1))+(icrmn(2)-icini(2)).ne.0)then
          if(iep(i).eq.6)then
            write(ifch,'(a,d25.15)')
     &'Flavor problem in proj for pseudo-inelastic collision !',seedc
          elseif(iep(i).eq.0)then
            iep(i)=1
          endif
        endif

        if(iremn.eq.2)then
          if(.not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
     &       .or.(nqu.eq.1.and.naq.eq.1)))iep(i)=3

        endif
      endif

      if(ish.ge.5)write(ifch,'(a,i3,a,i3,a,i2)')
     &            'Procop part ',ii,', iep(',i,'): ',iep(i)

      if(iremn.le.1)call iddeco(icrmn,jc)
      if(iep(i).ge.1.and.iep(i).ne.6)then
        aproj=dble(max(amproj,fremnux(jc)))
      else
        aproj=dble(max(amproj,fremnux2(jc)))
      endif
c      aprojex=max(ampmn(isp(i))+amemn(idp(i),iep(i))
c     &           ,dble(fremnux(jc)))
      xmptmp=(aproj**2+xxp(i)*xxp(i)+xyp(i)*xyp(i))
     &       /(xpp(i)*s)
      xpos(i)=xpp(i)*xmptmp
      if(ish.ge.5)write(ifch,*)'Procop mass : ',aproj,xpos(i)*s
      if(xmptmp.gt.1.d0)then
        xmptmp=0.d0
      if(ish.ge.1)write(ifmt,*)'Warning in ProCop, Remnant mass too low'
      endif

      pptl(1,nptl)=sngl(xxp(i))
      pptl(2,nptl)=sngl(xyp(i))
      pptl(3,nptl)=sngl((xpp(i)-xmptmp)*plc/2d0)
      pptl(4,nptl)=sngl((xpp(i)+xmptmp)*plc/2d0)
      pptl(5,nptl)=aproj

c      write(ifmt,*)'ProCop',i,nptl

      return

      end

c-----------------------------------------------------------------------
      subroutine ProCot(j,jj)
c-----------------------------------------------------------------------
c Propose Coordinates of remnants from active targets nucleons
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      double precision xpttmp,atarg
      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      integer icrmn(2),jc(nflav,2),icini(2)
      double precision s,plc

      nptl=nptl+1
      nptarg(j)=nptl

      idptl(nptl)=idptl(jj)*100+99    !100*10**idt(j)+iet(j)
      istptl(nptl)=40
      ityptl(nptl)=50
      iorptl(nptl)=jj
      jorptl(nptl)=0
      ifrptl(1,nptl)=0
      ifrptl(2,nptl)=0
      do k=1,2
        do i=1,nflav
          jc(i,k)=0
        enddo
      enddo

      istptl(jj)=1

c     determine kolz

      if(ltarg(j).gt.1)then
        zmin=ainfin
        kolz=0
        do l=1,ltarg(j)
          k=ktarg(j,l)
          z=coord(3,k)
          if(itpr(k).ne.0.and.z.lt.zmin)then
            zmin=z
            kolz=k
          endif
        enddo
      else
        kolz=1
      endif
c      if(kolz.eq.0)call utstop(' kolz=0 (targ)&')
      if(kolz.eq.0)then
        t=0.
      else
        t=coord(4,kolz)
      endif

      xorptl(1,nptl)=xorptl(1,jj)
      xorptl(2,nptl)=xorptl(2,jj)
      xorptl(3,nptl)=xorptl(3,jj)
      xorptl(4,nptl)=t
      tivptl(1,nptl)=t
      tivptl(2,nptl)=t
      naq=0
      nqu=0

      if(iremn.ge.2)then   !update ictarg
        idt(j)=min(1,abs(idt(j)))
        k=1
        nqu=0
        do n=1,nrflav
          jc(n,k)=jctref(n,k,j)
          nqu=nqu+jc(n,k)
        enddo
        k=2
        naq=0
        do n=1,nrflav
          jc(n,k)=jctref(n,k,j)
          naq=naq+jc(n,k)
        enddo
        isum=nqu+naq
        call idenco(jc,icrmn,iret)
        if(iret.eq.0.and.(isum.le.3.or.iremn.ne.3))then
          ictarg(1,j)=icrmn(1)
          ictarg(2,j)=icrmn(2)
        elseif(iremn.eq.3)then
      write(ifch,*)'Problem in projectile flavor :',j,' ->',jc,' :',isum
          call utstop('Procot: Problem in target flavor !&')
        else     !for iremn=2 and large number of quark define ictarg=999999
          ictarg(1,j)=999999
          ictarg(2,j)=999999
        endif
      endif

      icrmn(1)=ictarg(1,j)
      icrmn(2)=ictarg(2,j)

      if(iremn.ge.1)then      !excited remnant ?
        call idtr4(idptl(jj),icini)
        if(ish.ge.5)write(ifch,*)'Procot icini targ',j,icini,' ->',icrmn
        if((icrmn(1)-icini(1))+(icrmn(2)-icini(2)).ne.0)then
          if(iet(j).eq.6)then
            write(ifch,'(a,d25.15)')
     &'Flavor problem in targ for pseudo-inelastic collision !',seedc
          elseif(iet(j).eq.0)then
            iet(j)=1
          endif
        endif

        if(iremn.eq.2)then
          if(.not.((nqu.eq.3.and.naq.eq.0).or.(nqu.eq.0.and.naq.eq.3)
     &       .or.(nqu.eq.1.and.naq.eq.1)))iet(j)=3

        endif
      endif
      if(ish.ge.5)write(ifch,'(a,i3,a,i3,a,i2)')
     &            'Procot part ',jj,', iet(',j,'): ',iet(j)



      if(iremn.le.1)call iddeco(icrmn,jc)
      if(iet(j).ge.1.and.iet(j).ne.6)then
        atarg=dble(max(amtarg,fremnux(jc)))
      else
        atarg=dble(max(amtarg,fremnux2(jc)))
      endif
c      atargex=max(amtmn(ist(j))+amemn(idt(j),iet(j))
c     &           ,dble(fremnux(jc)))
      xpttmp=(atarg**2+xxt(j)*xxt(j)+xyt(j)*xyt(j))
     &       /(xmt(j)*s)
      xtos(j)=xpttmp*xmt(j)
      if(ish.ge.5)write(ifch,*)'Procot mass : ',atarg,xtos(j)*s
      if(xpttmp.gt.1.d0)then
        xpttmp=0.d0
      if(ish.ge.1)write(ifch,*)'Warning in ProCot, Remnant mass too low'
      endif

      pptl(1,nptl)=sngl(xxt(j))
      pptl(2,nptl)=sngl(xyt(j))
      pptl(3,nptl)=sngl((xpttmp-xmt(j))*plc/2d0)
      pptl(4,nptl)=sngl((xpttmp+xmt(j))*plc/2d0)
      pptl(5,nptl)=atarg

c      write(ifmt,*)'ProCot',j,nptl

      return
      end

c-----------------------------------------------------------------------
      subroutine emswrp(i,ii)
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      double precision p5sq
      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      double precision s,plc
      parameter(eps=1.e-5)

      if(npproj(i).eq.0)then
        write(*,*)'emswrp i ii',i,ii
        call utstop('emswrp with npproj=0 should never happen !&')

c        t=xorptl(4,kolp(i))
c        istptl(ii)=1
c        iorptl(ii)=-1
c        tivptl(2,ii)=t
c        nptl=nptl+1
c        npproj(i)=nptl
c        idptl(nptl)=idptl(ii)*100+99 !100*10**idp(i)+iep(i)
c        istptl(nptl)=40
c        ityptl(nptl)=40
c        iorptl(nptl)=ii
c        jorptl(nptl)=kolp(i)
c        ifrptl(1,nptl)=0
c        ifrptl(2,nptl)=0
c        xorptl(1,nptl)=xorptl(1,ii)
c        xorptl(2,nptl)=xorptl(2,ii)
c        xorptl(3,nptl)=xorptl(3,ii)
c        xorptl(4,nptl)=t
c        tivptl(1,nptl)=t
c        tivptl(2,nptl)=t
        mm=nptl
c        kolp(i)=1
      else
        mm=npproj(i)
      endif
      if(iLHC.eq.1.and.(iep(i).eq.0.or.iep(i).eq.6))
     &xmp(i)=min(1d0-xpp(i),xmp(i))
      pptl(1,mm)=sngl(xxp(i))
      pptl(2,mm)=sngl(xyp(i))
      pptl(3,mm)=sngl((xpp(i)-xmp(i))*plc/2d0)
      pptl(4,mm)=sngl((xpp(i)+xmp(i))*plc/2d0)
      if(pptl(4,mm).lt.-eps)call utstop('E pro<0 !&')
      p5sq=xpp(i)*xmp(i)*s-xxp(i)*xxp(i)-xyp(i)*xyp(i)
      if(p5sq.gt.1.d-10)then
        pptl(5,mm)=sngl(sqrt(p5sq))
      elseif(iep(i).eq.0)then
        pptl(5,mm)=pptl(5,ii)
      else
        if(ish.ge.2)then
          write(ifch,*)'problem with mass for projectile, '
     &         ,'continue with zero mass'
          write(ifch,*)i,mm,xxp(i),xyp(i),xpp(i),xmp(i),p5sq
        endif
        pptl(5,mm)=0.
      endif

      do l=1,4
       ibptl(l,mm)=0
      enddo

      return

      end

c-----------------------------------------------------------------------
      subroutine emswrt(j,jj)
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      double precision p5sq
      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      double precision s,plc
      parameter(eps=1.e-5)

      if(nptarg(j).eq.0)then

        write(*,*)'emswrt j jj',j,jj
        call utstop('emswrt with nptarg=0 should never happen !&')

c        t=xorptl(4,kolt(j))
c        istptl(jj)=1
c        iorptl(jj)=-1
c        tivptl(2,jj)=t
c        nptl=nptl+1
c        nptarg(j)=nptl
c        idptl(nptl)=idptl(jj)*100+99 !100*10**idp(i)+iep(i)
c        istptl(nptl)=40
c        ityptl(nptl)=50
c        iorptl(nptl)=jj
c        jorptl(nptl)=kolt(j)
c        ifrptl(1,nptl)=0
c        ifrptl(2,nptl)=0
c        xorptl(1,nptl)=xorptl(1,jj)
c        xorptl(2,nptl)=xorptl(2,jj)
c        xorptl(3,nptl)=xorptl(3,jj)
c        xorptl(4,nptl)=t
c        tivptl(1,nptl)=t
c        tivptl(2,nptl)=t
c... initialize
        mm=nptl
c        kolt(j)=1
      else
        mm=nptarg(j)
      endif
      if(iLHC.eq.1.and.(iet(j).eq.0.or.iet(j).eq.6))
     &xpt(j)=min(1d0-xmt(j),xpt(j))
      pptl(1,mm)=sngl(xxt(j))
      pptl(2,mm)=sngl(xyt(j))
      pptl(3,mm)=sngl((xpt(j)-xmt(j))*plc/2d0)
      pptl(4,mm)=sngl((xpt(j)+xmt(j))*plc/2d0)
      if(pptl(4,mm).lt.-eps)call utstop('E targ<0 !&')
      p5sq=xpt(j)*xmt(j)*s-xxt(j)*xxt(j)-xyt(j)*xyt(j)
      if(p5sq.gt.1.d-10)then
        pptl(5,mm)=sngl(sqrt(p5sq))
      elseif(iet(j).eq.0)then
        pptl(5,mm)=pptl(5,jj)
      else
        if(ish.ge.2)then
          write(ifch,*)'problem with mass for target, '
     &            ,'continue with zero mass'
          write(ifch,*)j,mm,xxt(j),xyt(j),xpt(j),xmt(j),p5sq
        endif
        pptl(5,mm)=0.
      endif

      do l=1,4
       ibptl(l,mm)=0
      enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine emswrpom(k,i,j)
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      common/cems5/plc,s
      common/emsptl/nppr(npommx,kollmx),npproj(mamx),nptarg(mamx)
      double precision s,px,py,plc

      do 30 n=1,nprmx(k)
       if(idpr(n,k).eq.0.or.ivpr(n,k).eq.0)goto30
       nptl=nptl+1
       nppr(n,k)=nptl
       px=xxp1pr(n,k)+xxp2pr(n,k)+xxm1pr(n,k)+xxm2pr(n,k)
       py=xyp1pr(n,k)+xyp2pr(n,k)+xym1pr(n,k)+xym2pr(n,k)
       pptl(1,nptl)=sngl(px)
       pptl(2,nptl)=sngl(py)
       pptl(3,nptl)=sngl(dsqrt(xpr(n,k))*dsinh(ypr(n,k))*plc)
       pptl(4,nptl)=sngl(dsqrt(xpr(n,k))*dcosh(ypr(n,k))*plc)
       pptl(5,nptl)=sngl(dsqrt(xpr(n,k)*s-px*px-py*py))
   !    print*,pptl(5,nptl)/plc
       idptl(nptl)=idpr(n,k)*10000
     &     +idp1pr(n,k)*1000
     &     +idp2pr(n,k)*100
     &     +idm1pr(n,k)*10
     &     +idm2pr(n,k)
       idptl(nptl)=idptl(nptl)*100+99
       istptl(nptl)=30
       iorptl(nptl)=i
       jorptl(nptl)=j
       ifrptl(1,nptl)=0
       ifrptl(2,nptl)=0
       xorptl(1,nptl)=coord(1,k)
       xorptl(2,nptl)=coord(2,k)
       xorptl(3,nptl)=coord(3,k)
       xorptl(4,nptl)=coord(4,k)
       tivptl(1,nptl)=coord(4,k)
       tivptl(2,nptl)=coord(4,k)
       if(idpr(n,k).eq.1)then
        ityptl(nptl)=20
        if(itpr(k).gt.0)ityptl(nptl)=25
       elseif(idpr(n,k).eq.3)then
        ityptl(nptl)=30
        if(itpr(k).gt.0)ityptl(nptl)=35
       else
        call utstop('emswrpom: unknown id&')
       endif
       do l = 1,4
        ibptl(l,nptl)=0
       enddo
30    continue

      return
      end

c-----------------------------------------------------------------------
      subroutine emsfrag(iret)
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      double precision pfrx(mamxx),pfry(mamxx),pfrz(mamxx),xmean,ymean
     &                ,zmean,spec
      integer          ityp(mamxx)

      iret=0

c Projectile fragment(s)

      irest = maproj*100+abs(laproj)
      inew=0
      idrest=0
      mapro=maproj
      xmean=0d0
      ymean=0d0
      zmean=0d0
      spec=0d0
      amrest=0.
      imin=maproj
      imax=1

      do is=1,maproj
        
        if(istptl(is).eq.0)then
          if ( iorptl(is) .eq. 0 ) then
            if(infragm.eq.0)then   !keep free nucleons
c  copy spectators at the end of the list (necessary for hepmc interface)
              nptl=nptl+1
              if(nptl.gt.mxptl)then
                iret=1
                goto 1000
              endif
              call utrepl(nptl,is)
              istptl(is)=1
              ifrptl(1,is)=nptl
              ifrptl(2,is)=nptl
              istptl(nptl)=0
              iorptl(nptl)=is
            else
c  compose projectile spectators to remaining nucleus
              spec=spec+1d0
              tivptl(2,is)=0d0
              xmean=xmean+xorptl(1,is)
              ymean=ymean+xorptl(2,is)
              zmean=0d0
              amrest=amrest+pptl(5,is)
              imin=min(imin,is)
              imax=max(imax,is)
              istptl(is)=1
              ifrptl(1,is)=nptl+1
              ifrptl(2,is)=nptl+1
              idrest = is
              id=idptl(is)
              if     ( id .eq. 1120 ) then
                inew  = inew + 101
                irest = irest - 101
              elseif ( id .eq. 1220 ) then
                inew  = inew + 100
                irest = irest - 100
              endif
            endif
          endif
        elseif( iorptl(is) .le. 0  .and.  istptl(is) .eq. 1 ) then
          if( iorptl(is) .eq. 0 )jorptl(is)=1
          mapro=mapro-1
        endif
        
      enddo

      if(inew.eq.0)goto 100

      xmean=xmean/spec
      ymean=ymean/spec
      zmean=zmean/spec
      nptla=nptl
c prepare intermediate particle to produce nuclear fragment
      nptl=nptl+1
      if(nptl.gt.mxptl)then
        iret=1
        goto 1000
      endif

      if( inew .eq. 100 .or. inew .eq. 101 ) then
c  remaining nucleus is single neutron or proton
        call utrepl(nptl,idrest)
        ifrptl(1,idrest)=nptl
        ifrptl(2,idrest)=nptl
        istptl(nptl)=0
        iorptl(nptl)=idrest
        goto 100

      else

c intermediate particles for father/mother relationship
        idptl(nptl)=800000000+inew
        ea = float(inew/100)*pptl(4,idrest)
c  momenta squared
        ptm = sqrt(max(0.,(ea-amrest)*(ea+amrest)))
        istptl(nptl)=51
        pptl(1,nptl)=0.
        pptl(2,nptl)=0.
        pptl(3,nptl)=ptm
        pptl(4,nptl)=sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2
     *                     +pptl(3,nptl)**2+amrest**2)
        pptl(5,nptl)=amrest         !mass
        ityptl(nptl)=40
        iorptl(nptl)=imax
        jorptl(nptl)=imax
        ifrptl(1,nptl)=nptl+1
        ifrptl(2,nptl)=0
        xorptl(1,nptl)=0d0
        xorptl(2,nptl)=0d0
        xorptl(3,nptl)=0d0
        xorptl(4,nptl)=0d0
        tivptl(1,nptl)=0d0
        tivptl(2,nptl)=0d0

        if ( infragm .ge. 2 ) then
c  remaining nucleus is evaporating nucleons and alpha particles
         jfin  = 0
         call epovapor( mapro,inew,jfin,ityp,pfrx,pfry,pfrz )
         if ( jfin .eq. 0 )then   !something failed
           iret=1
           goto 1000
         endif
c loop to treat the remnants of the desintegrated fragment
         do  135  j = 1, jfin
          if(ityp(j).lt.0.)then
            idnucl=-ityp(j)
            inucl= idnucl/100
            if(idnucl.eq.402)then   !helium (alpha)
              idnucl=19
            elseif(idnucl.eq.301)then   !tritium
              idnucl=18
            elseif(idnucl.eq.201)then   !deuterium
              idnucl=17
            else
              iprot= mod(idnucl,100)
              idnucl=1000000000+iprot*10000+inucl*10 !PDG code for nucleus
            endif
          else
            inucl=1
            idnucl=ityp(j)
          endif
          ea = float(inucl)*pptl(4,idrest)
c  momenta squared
          call idmass(idnucl,am)
          ptm = ( ea - am ) * ( ea + am )
          pt2 = sngl( pfrx(j)**2 + pfry(j)**2 )
          if(ish.ge.6)write(ifch,*) 'pro fragment: j,id,ea,ptm,pt2=',
     *                                       j,idnucl,ea,ptm,pt2
          if ( pt2 + pfrz(j)**2 .ge. ptm ) then
            if (ish.ge.2) write(ifch,*) 'emsfrag: pt reject particle',j
            nnn=0
            is=0
            do while (is.lt.maproj.and.nnn.lt.inucl)
              is=is+1
              if(istptl(is).eq.1
     &             .and.jorptl(is).eq.0.and.iorptl(is).eq.0)then
                nnn=nnn+1
c  copy spectators at the end of the list (necessary for hepmc interface)
                nptl=nptl+1
                if(nptl.gt.mxptl)then
                  iret=1
                  goto 1000
                endif
                call utrepl(nptl,is)
                jorptl(is)=1
                ifrptl(1,is)=nptl
                ifrptl(2,is)=nptl
                istptl(nptl)=0
                iorptl(nptl)=is
              endif
            enddo
            goto 135
          else
            plong = sqrt(ptm-pt2)
          endif
          nptl=nptl+1
          if(nptl.gt.mxptl)then
            iret=1
            goto 1000
          endif
          istptl(nptl)=0
          pptl(1,nptl)=sngl(pfrx(j))
          pptl(2,nptl)=sngl(pfry(j))
          pptl(3,nptl)=plong+sngl(pfrz(j))   !OK if plong >> pfrz
          pptl(4,nptl)=sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2
     *                     +pptl(3,nptl)**2+am**2)
          pptl(5,nptl)=am    !mass
          ityptl(nptl)=0
          iorptl(nptl)=nptla+1
          jorptl(nptl)=0
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=xmean
          xorptl(2,nptl)=ymean
          xorptl(3,nptl)=zmean
          xorptl(4,nptl)=zmean
          tivptl(1,nptl)=zmean
          tivptl(2,nptl)=tivptl(2,idrest)
          idptl(nptl)=idnucl
 135    continue

        elseif ( infragm .eq. 1 ) then
c  remaining nucleus is one fragment
          nptl=nptl+1
          if(nptl.gt.mxptl)then
            iret=1
            goto 1000
          endif
          istptl(nptl)=0
          pptl(1,nptl)=0.d0
          pptl(2,nptl)=0.d0
          pptl(4,nptl)=0.d0
          inucl=0
          do is=1,maproj
            if(iorptl(is).eq.0.and.jorptl(is).eq.0)then
              inucl=inucl+1
              pptl(4,nptl)=pptl(4,nptl)+dble(pptl(4,is))
            endif
          enddo
          if(inucl.ne.inew/100)call utstop('Pb in emsfrag !&')
          idnucl=1000000000+mod(inew,100)*10000+(inew/100)*10
          call idmass(idnucl,am)
          pptl(5,nptl)=am    !mass
          ptot=(pptl(4,nptl)+am)*(pptl(4,nptl)-am)
          pptl(3,nptl)=sqrt(ptot)
          ityptl(nptl)=0
          istptl(nptl)=0
          iorptl(nptl)=nptla+1
          jorptl(nptl)=0
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=xmean
          xorptl(2,nptl)=ymean
          xorptl(3,nptl)=zmean
          xorptl(4,nptl)=zmean
          tivptl(1,nptl)=zmean
          tivptl(2,nptl)=tivptl(2,idrest)
          idptl(nptl)=idnucl
        endif
        ifrptl(2,nptla+1)=nptl
        if(ifrptl(1,nptla+1).gt.ifrptl(2,nptla+1))then
          ifrptl(1,nptla+1)=0
          ifrptl(2,nptla+1)=0
        endif
      endif

      do is=nptla+1,nptl
          if(ish.ge.5)write(ifch,'(a,i5,a,i10,a,4(e10.4,1x),f6.3)')
     $       ' Projectile fragments ',is,' id :',idptl(is)
     $  , ' momentum :',(pptl(k,is),k=1,5)
      enddo

 100  continue

c Target fragment(s)

      irest = matarg*100+abs(latarg)
      inew=0
      matar=matarg
      xmean=0d0
      ymean=0d0
      zmean=0d0
      spec=0d0
      amrest=0.
      imin=maproj+matarg
      imax=maproj+1

      do is=maproj+1,maproj+matarg
        
        if(istptl(is).eq.0)then
          if ( iorptl(is) .eq. 0 ) then
            if(infragm.eq.0)then   !keep free nucleons
c  copy spectators at the end of the list (necessary for hepmc interface)
              nptl=nptl+1
              if(nptl.gt.mxptl)then
                iret=1
                goto 1000
              endif
              call utrepl(nptl,is)
              istptl(is)=1
              ifrptl(1,is)=nptl
              ifrptl(2,is)=nptl
              istptl(nptl)=0
              iorptl(nptl)=is
            else
c  compose projectile spectators to remaining nucleus
              spec=spec+1d0
              tivptl(2,is)=0d0
              xmean=xmean+xorptl(1,is)
              ymean=ymean+xorptl(2,is)
              zmean=0d0
              amrest=amrest+pptl(5,is)
              imin=min(imin,is)
              imax=max(imax,is)
              istptl(is)=1
              ifrptl(1,is)=nptl+1
              ifrptl(2,is)=nptl+1
              idrest = is
              id=idptl(is)
              if     ( id .eq. 1120 ) then
                inew  = inew + 101
                irest = irest - 101
              elseif ( id .eq. 1220 ) then
                inew  = inew + 100
                irest = irest - 100
              endif
            endif
          endif
          
        elseif( iorptl(is) .le. 0  .and.  istptl(is) .eq. 1 ) then
          if( iorptl(is) .eq. 0 ) jorptl(is)=1
          matar=matar-1
        endif
        
      enddo

      if(inew.eq.0)goto 1000

      xmean=xmean/spec
      ymean=ymean/spec
      zmean=zmean/spec
      nptla=nptl
c prepare intermediate particle to produce nuclear fragment
      nptl=nptl+1
      if(nptl.gt.mxptl)then
        iret=1
        goto 1000
      endif

      if( inew .eq. 100 .or. inew .eq. 101 ) then
c  remaining nucleus is single neutron or proton
        call utrepl(nptl,idrest)
        ifrptl(1,idrest)=nptl
        ifrptl(2,idrest)=nptl
        istptl(nptl)=0
        iorptl(nptl)=idrest
        goto 1000

      else

c intermediate particles for father/mother relationship
        idptl(nptl)=800000000+inew
        ea = float(inew/100)*pptl(4,idrest)
c  momenta squared
        ptm = sqrt(max(0.,(ea-amrest)*(ea+amrest)))
        istptl(nptl)=51
        pptl(1,nptl)=0.
        pptl(2,nptl)=0.
        pptl(3,nptl)=-ptm
        pptl(4,nptl)=sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2
     *                     +pptl(3,nptl)**2+amrest**2)
        pptl(5,nptl)=amrest         !mass
        ityptl(nptl)=50
        iorptl(nptl)=imax
        jorptl(nptl)=imax
        ifrptl(1,nptl)=nptl+1
        ifrptl(2,nptl)=0
        xorptl(1,nptl)=0d0
        xorptl(2,nptl)=0d0
        xorptl(3,nptl)=0d0
        xorptl(4,nptl)=0d0
        tivptl(1,nptl)=0d0
        tivptl(2,nptl)=0d0

        if ( infragm .ge. 2 ) then
c  remaining nucleus is evaporating nucleons and alpha particles
         jfin  = 0
         call epovapor( matar,inew,jfin,ityp,pfrx,pfry,pfrz )
         if ( jfin .eq. 0 )then   !something failed
           iret=1
           goto 1000
         endif
c loop to treat the remnants of the desintegrated fragment
         do  235  j = 1, jfin
          if(ityp(j).lt.0.)then
            idnucl=-ityp(j)
            inucl= idnucl/100
            if(idnucl.eq.402)then   !helium (alpha)
              idnucl=19
            elseif(idnucl.eq.301)then   !tritium
              idnucl=18
            elseif(idnucl.eq.201)then   !deuterium
              idnucl=17
            else
              iprot= mod(idnucl,100)
              idnucl=1000000000+iprot*10000+inucl*10 !PDG code for nucleus
            endif
          else
            inucl=1
            idnucl=ityp(j)
          endif
          ea = float(inucl)*pptl(4,idrest)
c  momenta squared
          call idmass(idnucl,am)
          ptm = ( ea - dble(am) ) * ( ea + dble(am) )
          pt2 = sngl( pfrx(j)**2 + pfry(j)**2 )
          if(ish.ge.6)write(ifch,*) 'tar fragment: j,id,ea,ptm,pt2=',
     *                                       j,idnucl,ea,ptm,pt2
          if ( pt2 + pfrz(j)**2 .ge. ptm ) then
            if (ish.ge.2) write(ifch,*) 'emsfrag: pt reject particle',j
            nnn=0
            is=maproj
            do while (is.lt.maproj+matarg.and.nnn.lt.inucl)
              is=is+1
              if(istptl(is).eq.1
     &             .and.jorptl(is).eq.0.and.iorptl(is).eq.0)then
                nnn=nnn+1
c  copy spectators at the end of the list (necessary for hepmc interface)
                nptl=nptl+1
                if(nptl.gt.mxptl)then
                  iret=1
                  goto 1000
                endif
                call utrepl(nptl,is)
                jorptl(is)=1
                ifrptl(1,is)=nptl
                ifrptl(2,is)=nptl
                istptl(nptl)=0
                iorptl(nptl)=is
              endif
            enddo
            goto 235
          else
            plong=-sqrt(ptm-pt2)
          endif
          nptl=nptl+1
          if(nptl.gt.mxptl)then
            iret=1
            goto 1000
          endif
          istptl(nptl)=0
          pptl(1,nptl)=sngl(pfrx(j))
          pptl(2,nptl)=sngl(pfry(j))
          pptl(3,nptl)=plong+sngl(pfrz(j))   !OK if plong >> pfrz
          pptl(4,nptl)=sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2
     *                     +pptl(3,nptl)**2+am**2)
          pptl(5,nptl)=am    !mass
          ityptl(nptl)=0
          iorptl(nptl)=nptla+1
          jorptl(nptl)=0
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=xmean
          xorptl(2,nptl)=ymean
          xorptl(3,nptl)=zmean
          xorptl(4,nptl)=zmean
          tivptl(1,nptl)=zmean
          tivptl(2,nptl)=tivptl(2,idrest)
          idptl(nptl)=idnucl
 235    continue

        elseif ( infragm .eq. 1 ) then
c  remaining nucleus is one fragment
          nptl=nptl+1
          if(nptl.gt.mxptl)then
            iret=1
            goto 1000
          endif
          istptl(nptl)=0
          pptl(1,nptl)=0.d0
          pptl(2,nptl)=0.d0
          pptl(4,nptl)=0.d0
          inucl=0
          do is=maproj+1,maproj+matarg
            if(iorptl(is).eq.0.and.jorptl(is).eq.0)then
              inucl=inucl+1
              pptl(4,nptl)=pptl(4,nptl)+dble(pptl(4,is))
            endif
          enddo
          if(inucl.ne.inew/100)call utstop('Pb in emsfrag !&')
          idnucl=1000000000+mod(inew,100)*10000+(inew/100)*10
          call idmass(idnucl,am)
          pptl(5,nptl)=am    !mass
          ptot=(pptl(4,nptl)+am)*(pptl(4,nptl)-am)
          pptl(3,nptl)=sqrt(ptot)
          ityptl(nptl)=0
          istptl(nptl)=0
          iorptl(nptl)=nptla+1
          jorptl(nptl)=0
          ifrptl(1,nptl)=0
          ifrptl(2,nptl)=0
          xorptl(1,nptl)=xmean
          xorptl(2,nptl)=ymean
          xorptl(3,nptl)=zmean
          xorptl(4,nptl)=zmean
          tivptl(1,nptl)=zmean
          tivptl(2,nptl)=tivptl(2,idrest)
          idptl(nptl)=idnucl
        endif
        ifrptl(2,nptla+1)=nptl
        if(ifrptl(1,nptla+1).gt.ifrptl(2,nptla+1))then
          ifrptl(1,nptla+1)=0
          ifrptl(2,nptla+1)=0
        endif
      endif

      do is=nptla+1,nptl
          if(ish.ge.5)write(ifch,'(a,i5,a,i10,a,4(e10.4,1x),f6.3)')
     $       ' Target fragments ',is,' id :',idptl(is)
     $  , ' momentum :',(pptl(k,is),k=1,5)
      enddo


 1000 continue


      end

cc--------------------------------------------------------------------------
c      subroutine reaction(idpj,idtg,ireac)
cc--------------------------------------------------------------------------
cc returns reaction code ireac
cc--------------------------------------------------------------------------
c      iap=iabs(idpj/10)
c      iat=iabs(idtg/10)
c      isp=idpj/10/iap
c      ist=idtg/10/iat
c      call idchrg(idpj,cp)
c      call idchrg(idtg,ct)
c      ac=abs(cp+ct)
c      if(iap.gt.100)then
c       if(iat.gt.100)then
c        if(isp.eq.1)then
c         if(ist.eq.1)then
c          ireac=1
c         else
c          ireac=6
c         endif
c        else
c         if(ist.eq.1)then
c          ireac=6
c         else
c          ireac=1
c         endif
c        endif
c       elseif(iat.eq.11.or.iat.eq.12.or.iat.eq.22)then
c        if(ac.ge.2.)then
c         ireac=2
c        else
c         ireac=3
c        endif
c       else
c        if(ac.ge.2.)then
c         ireac=4
c        else
c         ireac=5
c        endif
c       endif
c      elseif(iap.eq.11.or.iap.eq.12.or.iap.eq.22)then
c       if(iat.gt.100)then
c        if(ac.ge.2.)then
c         ireac=2
c        else
c         ireac=3
c        endif
c       elseif(iat.eq.11.or.iat.eq.12.or.iat.eq.22)then
c        ireac=7
c       else
c        ireac=8
c       endif
c      else
c       if(iat.gt.100)then
c        if(ac.ge.2.)then
c         ireac=4
c        else
c         ireac=5
c        endif
c       elseif(iat.eq.11.or.iat.eq.12.or.iat.eq.22)then
c        ireac=8
c       else
c        ireac=9
c       endif
c      endif
c
c      end
c
c-----------------------------------------------------------------------
      subroutine xEmsI1(iii,kc,omlog)
c-----------------------------------------------------------------------
c plot omlog vs iter
c plot  nr of pomerons vs iter
c plot number of collisions vs iter
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'

      parameter(nbin=100)
      common/cmc/ot(0:nbin),zz(0:nbin),i(0:nbin)
     *,yt1,yt2,kx(0:nbin)
      parameter(nbim=100)
      common/cmc1/xp(0:nbim),xt(0:nbim),x(0:nbim),o(0:nbim)
     *,y1,y2,car
      character car*5
      double precision xp,xt,x,omlog,om1intbc
      character ce*8
      double precision plc,s,seedp
      common/cems5/plc,s

c      if(iemsi2.eq.0)call utstop('ERROR in XemsI1: iemsi2 = 0&')

       if(iii.eq.1)then

      o(kc)=sngl(omlog)
      nptk=0
      kollx=0
      do ko=1,koll
      nptk=nptk+nprt(ko)
c      if(itpr(ko).gt.0)then
      if(nprt(ko).gt.0)then
       kollx=kollx+1
      endif
      enddo
      zz(kc)=nptk
      kx(kc)=kollx

        elseif(iii.eq.2)then

      call ranfgt(seedp)
      sum=0
      kollx=0
      sumg=0
      kollg=0
      kollini=koll
      koll=1
      do ko=1,kollini
ctp060829       ip=iproj(ko)
ctp060829       it=itarg(ko)
       om1i=sngl(om1intbc(bk(ko)))
ctp060829         wk=1.
ctp060829         wp=0.
ctp060829         wt=0.
       om1g=sngl(om1intbc(bk(ko)))
       sum=sum+om1i
       sumg=sumg+om1g
       if(rangen().lt.1.-exp(-om1i))then
        kollx=kollx+1
       endif
       if(rangen().lt.1.-exp(-om1g))then
        kollg=kollg+1
       endif
      enddo
      koll=kollini
      call ranfst(seedp)

      x1=0
      x2=nbin
      write(ce,'(f8.2)')sngl(plc)

      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   log omega       for event ',nrevt+1
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name omega-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')       'yrange auto auto '
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis ln[W]"'
      write(ifhi,'(a,a)')  'text 0.5 0.90 "E ='//ce//'"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbim
      write(ifhi,'(2e11.3)')float(k),o(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i3)')'! nr of coll`s  for event ',nrevt+1
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name coll-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis nr of collisions"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),float(kx(k))
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),float(kollx)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),float(kollg)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i3)')'! nr of pom`s  for event ',nrevt+1
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name pom-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis nr of Pomerons"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),zz(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      if(sum.lt.4*zz(nbin))then
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),sum
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbin
      write(ifhi,'(2e11.3)')float(k),sumg
         enddo
      write(ifhi,'(a)')    '  endarray'
      endif
      write(ifhi,'(a)')    'closehisto plot 0'

        endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsI2(iii,kc)
c-----------------------------------------------------------------------
c plot quanities vs iter
c   plot 1: <x> for Pomeron vs iter
c   plot 2: <x> for projectile vs iter
c   plot 3: <x> for target vs iter
c arguments:
c   iii:   modus (1,2)
c   kc:    iteration step
c   omega: config probability
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'

      parameter(nbim=100)
      common/cmc1/xp(0:nbim),xt(0:nbim),x(0:nbim),o(0:nbim)
     *,y1,y2,car
      character car*5
      double precision xp,xt,x,xpo,xpj,xtg
      common/cemsi2/xpo,xpj,xtg

        if(iii.eq.1)then

      npom=0
      xpo=0
      do k=1,koll
c       ip=iproj(k)
c       it=itarg(k)
       if(nprmx(k).gt.0)then
        do n=1,nprmx(k)
         if(idpr(n,k).gt.0.and.ivpr(n,k).gt.0)then
          xpo=xpo+xpr(n,k)
          npom=npom+1
         endif
        enddo
       endif
      enddo
      if(npom.gt.0)xpo=xpo/npom

      npk=0
      xpj=0d0
      do i=1,maproj
       if(xpp(i).lt.0.999)then
        xpj=xpj+xpp(i)!*xmp(i)
        npk=npk+1
       endif
      enddo
      if(npk.gt.0)xpj=xpj/dble(npk)

      ntk=0
      xtg=0d0
      do j=1,matarg
       if(xmt(j).lt.0.999)then
        xtg=xtg+xmt(j)!*xpt(j)
        ntk=ntk+1
       endif
      enddo
      if(ntk.gt.0)xtg=xtg/dble(ntk)

      x(kc)=xpo
      xp(kc)=xpj
      xt(kc)=xtg

        elseif(iii.eq.2)then

      x1=0
      x2=nbim

      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   average x  Pom   for event ',nrevt+1
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name avxPom-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis average x Pomeron"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbim
      write(ifhi,'(2e11.3)')float(k),x(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   average x proj   for event ',nrevt+1
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name avxProj-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis average x proj"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbim
      write(ifhi,'(2e11.3)')float(k),xp(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i3)')    '!   average x targ   for event ',nrevt+1
      write(ifhi,'(a)')       '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name avxTarg-',nrevt+1
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',x1,x2
      write(ifhi,'(a)')    'text 0 0 "xaxis iteration"'
      write(ifhi,'(a)')    'text 0 0 "yaxis average x targ"'
      write(ifhi,'(a)')       'array 2'
         do k=0,nbim
      write(ifhi,'(2e11.3)')float(k),xt(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'
        endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsRx(iii,id,xp,xm)
c-----------------------------------------------------------------------
c plot  x+, x-, x, y distribution of remnants
c-----------------------------------------------------------------------

      include 'epos.inc'

      parameter(nbix=50,nbiy=50,nid=2)
      common/cxp/nxp(nid),nxm(nid),nx(nid),ny(nid)
     *,wxp(nbix,nid),wxm(nbix,nid),wx(nbix,nid),wy(nbiy,nid)
     *,xpu,xpo,xmu,xmo,xu,xo,yu,yo,dy

      if(iemsrx.eq.0)call utstop('ERROR in XemsRx: iemsrx = 0&')

        if(iii.eq.0)then

      xpu=10/engy**2
      xpo=1
      xmu=10/engy**2
      xmo=1
      xu=10/engy**2
      xo=1
      yu=-alog(engy**2)
      yo=alog(engy**2)
      dy=(yo-yu)/nbiy
      do j=1,nid
       nxp(j)=0
       nxm(j)=0
       nx(j)=0
       do i=1,nbix
        wxp(i,j)=0
        wxm(i,j)=0
        wx(i,j)=0
       enddo
       ny(j)=0
       do i=1,nbiy
        wy(i,j)=0
       enddo
      enddo

        elseif(iii.eq.1)then

      i=0
      if(xp.lt.xpu)goto1
      i=1+int(alog(xp/xpu)/alog(xpo/xpu)*nbix)
      if(i.gt.nbix)goto1
      if(i.lt.1)goto1
      wxp(i,id)=wxp(i,id)+1
      nxp(id)=nxp(id)+1
1     continue

      if(xm.lt.xmu)goto2
      i=1+int(alog(xm/xmu)/alog(xmo/xmu)*nbix)
      if(i.gt.nbix)goto2
      if(i.lt.1)goto2
      wxm(i,id)=wxm(i,id)+1
      nxm(id)=nxm(id)+1
2     continue

      x=xp*xm
      if(x.lt.xu)goto3
      i=1+int(alog(x/xu)/alog(xo/xu)*nbix)
      if(i.gt.nbix)goto3
      if(i.lt.1)goto3
      wx(i,id)=wx(i,id)+1
      nx(id)=nx(id)+1
3     continue

      if(xm.le.0.)goto4
      if(xp.le.0.)goto4
      y=0.5*alog(xp/xm)
      if(y.lt.yu)goto4
      i=int((y-yu)/dy)+1
      if(i.gt.nbiy)goto4
      if(i.lt.1)goto4
      wy(i,id)=wy(i,id)+1
      ny(id)=ny(id)+1
4     continue

        elseif(iii.eq.2)then

      do j=1,nid
      if(j.eq.1)then
        iclrem=iclpro
      elseif(j.eq.2)then
        iclrem=icltar
      else
        iclrem=0
      endif
      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a)')      '!   remnant xp distribution      '
      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a,i1)')    'openhisto name xpRemnant-',j
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xpu,xpo
      write(ifhi,'(a)')    'text 0 0 "xaxis remnant x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(x+)"'
      write(ifhi,'(a)')       'array 2'
         do i=1,nbix
      x=xpu*(xpo/xpu)**((i-0.5)/nbix)
      dx=xpu*(xpo/xpu)**(1.*i/nbix)*(1.-(xpo/xpu)**(-1./nbix))
      if(nxp(j).ne.0)write(ifhi,'(2e11.3)')x,wxp(i,j)/dx/nxp(j)
      if(nxp(j).eq.0)write(ifhi,'(2e11.3)')x,0.
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'array 2'
         do i=1,nbix
      x=xu*(xo/xu)**((i-0.5)/nbix)
      write(ifhi,'(2e11.3)')x,x**alplea(iclrem)*(1+alplea(iclrem))
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a)')      '!   remnant xm distribution      '
      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a,i1)')    'openhisto name xmRemnant-',j
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xmu,xmo
      write(ifhi,'(a)')    'text 0 0 "xaxis remnant x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(x-)"'
      write(ifhi,'(a)')       'array 2'
         do i=1,nbix
      x=xmu*(xmo/xmu)**((i-0.5)/nbix)
      dx=xmu*(xmo/xmu)**(1.*i/nbix)*(1.-(xmo/xmu)**(-1./nbix))
      if(nxm(j).ne.0)write(ifhi,'(2e11.3)')x,wxm(i,j)/dx/nxm(j)
      if(nxm(j).eq.0)write(ifhi,'(2e11.3)')x,0.
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a)')      '!   remnant x distribution      '
      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a,i1)')    'openhisto name xRemnant-',j
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xu,xo
      write(ifhi,'(a)')    'text 0 0 "xaxis remnant x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(x)"'
      write(ifhi,'(a)')       'array 2'
         do i=1,nbix
      x=xu*(xo/xu)**((i-0.5)/nbix)
      dx=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
      if(nx(j).ne.0)write(ifhi,'(2e11.3)')x,wx(i,j)/dx/nx(j)
      if(nx(j).eq.0)write(ifhi,'(2e11.3)')x,0.
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a)')      '!   remnant y distribution      '
      write(ifhi,'(a)')      '!----------------------------------'
      write(ifhi,'(a,i1)')    'openhisto name yRemnant-',j
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',yu,yo
      write(ifhi,'(a)')    'text 0 0 "xaxis remnant y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(y)"'
      write(ifhi,'(a)')       'array 2'
         do i=1,nbix
      y=yu+dy/2.+(i-1)*dy
      if(ny(j).ne.0)write(ifhi,'(2e11.3)')y,wy(i,j)/dy/ny(j)
      if(ny(j).eq.0)write(ifhi,'(2e11.3)')y,0.
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      enddo

        endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsPm(iii,ko,nmci,nmcmx)
c-----------------------------------------------------------------------
c m (pomeron number) distribution for different b-bins.
c arguments:
c   iii:  modus (0,1,2)
c   ko:   pair number (1 - AB)
c   nmc:  number of pomerons
c   nmcmx: number max of pomerons
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbin=200)
      parameter(nbib=32)
      common/cn/wn(0:nbin,nbib),wnmc(0:nbin,nbib),npmx(nbib),nn(nbib)
     &         ,nn2(nbib),dn(nbib)
      common/cb1/db,b1,b2,bb(nbib),nbibx
      double precision plc,s,om1intbc
      character ce*8,cb*4
      common/cems5/plc,s
      common/cemspm/sumb(nbib)

      if(iemspm.eq.0)call utstop('ERROR in XemsPm: iemspm = 0&')

        if(iii.eq.0)then

      do k=1,nbib
       nn(k)=0
       nn2(k)=0
       sumb(k)=0
       do i=0,nbin
        wnmc(i,k)=0
       enddo
      enddo
      nbibx=6
      b1=0
      b2=2
      db=(b2-b1)/nbibx


        elseif(iii.eq.1)then

      k=int((bk(ko)-b1)/db)+1
c      nmc=nmci
      if(k.gt.nbibx)k=nbibx
      if(k.lt.1)k=1
      dn(k)=max(1.,float(nmcmx)/float(nbin))
      nmc=nint(float(nmci)/dn(k)+0.499999)
      if(nmc.gt.nbin)nmc=nbin
      if(nmc.lt.0)return
      nn(k)=nn(k)+1
      wnmc(nmc,k)=wnmc(nmc,k)+1./dn(k)
      sumb(k)=sumb(k)+bk(ko)


        elseif(iii.eq.2)then

      kollini=koll
      koll=1         !to have screening for pp

      do 1 k=1,nbibx

       bb(k)=b1+(k-0.5)*db
       if(maproj.eq.1.and.matarg.eq.1.and.bmaxim.eq.0.)bb(k)=b1
       om1i=sngl(om1intbc(bb(k)))
       wntmp=0.
       do 10 i=0,nbin
         wn(i,k)=0.
         if(wntmp.gt.1e5)goto 10
         do j=i,i+int(dn(k))-1
           if(j.eq.0)then
             wntmp=exp(-om1i)
           else
             wntmp=wntmp*om1i/j
           endif
           wn(i,k)=wn(i,k)+wntmp/dn(k)
         enddo
         if(wn(i,k).gt.0.000001*(1.-exp(-om1i)))npmx(k)=i
 10    continue

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! distr of Pomeron number vs b'
      write(ifhi,'(a)')   '!##################################'
      write(ce,'(f8.2)')sngl(plc)
      write(cb,'(f4.2)')bb(k)
      if(nn(k).gt.0)then
      write(ifhi,'(a,i1)')    'openhisto name mPom-',k
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,float(npmx(k))*dn(k)
      write(ifhi,'(a)')    'text 0 0 "xaxis number m of Pomerons"'
      write(ifhi,'(a)')    'text 0 0 "yaxis prob(m)"'
      if(k.eq.1)
     *write(ifhi,'(a,a)')     'text 0.5 0.90 "E ='//ce//'"'
      write(ifhi,'(a,a)')     'text 0.5 0.80 "b ='//cb//'"'
      write(ifhi,'(a)')       'array 2'
         do i=0,nbin
      write(ifhi,'(2e11.3)')float(i)*dn(k),wnmc(i,k)/max(1,nn(k))
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      endif

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! distr of Pomeron number vs b'
      write(ifhi,'(a)')   '!   traditional approach'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1)')    'openhisto name mPomTradi-',k
      write(ifhi,'(a)')       'htyp lba'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,float(npmx(k))*dn(k)
      write(ifhi,'(a)')       'array 2'
         do i=0,nbin
      write(ifhi,'(2e11.3)')float(i)*dn(k),wn(i,k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

 1    continue

      koll=kollini

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsB(iii,jjj,ko)
c-----------------------------------------------------------------------
c b distribution at different stages
c arguments:
c   iii:  modus (0,1,2)
c   jjj:  stage or type of interaction
c     just after Metropolis:
c           1 ... all
c           2 ... interaction
c     after defining diffraction:
c           3 ... nothing
c           4 ... cut
c           5 ... diffr
c           6 ... cut + diffr cut
c   ko:   pair number (1 - AB)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      parameter(njjj=6)
      parameter(nbib=32)
      common/cxemsb1/w(0:njjj,nbib),nn(njjj)
      common/cxemsb2/db,b1,b2
      common/cxemsb3/njjj1
      double precision PhiExact,om1intbi,PhiExpo!,PhiUnit
      common/geom/rmproj,rmtarg,bmax,bkmx
      dimension uua2(nbib),uuo2(nbib),uu3(nbib)

      if(iemsb.eq.0)call utstop('ERROR in XemsB: iemsB = 0&')

        if(iii.eq.0)then

      do k=1,nbib
       do j=0,njjj
        w(j,k)=0
       enddo
      enddo
      do j=1,njjj
       nn(j)=0
      enddo
      njjj1=0

        elseif(iii.eq.1)then

      b1=0
      b2=bkmx*1.2
      db=(b2-b1)/nbib
      k=int((bk(ko)-b1)/db)+1
      if(k.gt.nbib)return
      if(k.lt.1)return
      w(jjj,k)=w(jjj,k)+1
      nn(jjj)=nn(jjj)+1
      if(jjj.eq.1)njjj1=1

        elseif(iii.eq.2)then

      if(njjj1.ne.1)call utstop
     &('xEmsB must be called also with jjj=1&')
      ymax=0
      kollini=koll
      koll=1
      do k=1,nbib
       x=b1+(k-0.5)*db
       y=w(1,k)/nn(1)/(pi*((x+0.5*db)**2-(x-0.5*db)**2))
       ymax=max(ymax,y)
      enddo
      fk=bkmx**2*pi
      ymax=1.4

      do 1 j=1,njjj
       if(nn(j).eq.0)goto1

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distr exact theory '
      write(ifhi,'(a)')   '!##################################'
         if(j.ge.2.and.j.le.6)then
      write(ifhi,'(a,i1,a)')  'openhisto name b',j,'Exact'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1+(k-0.5)*db
      if(j.eq.2)then
        uuo2(k)=sngl(PhiExpo(0.,0.,1.,1.d0,1.d0,engy**2,b))
        uua2(k)=min(uuo2(k),max(0.,
     &          sngl(Phiexact(0.,0.,1.,1.d0,1.d0,engy**2,b))))
        uu3(k)=sngl(min(50d0,exp(om1intbi(b,2)/dble(r2hads(iclpro)
     &                                             +r2hads(icltar)))))
      endif
      if(j.eq.2)y=(1.-uua2(k))
      if(j.eq.3)y=uua2(k)
      if(j.eq.4.or.j.eq.6)y=(1.-uua2(k)*uu3(k))
      if(j.eq.5)y=uua2(k)*(uu3(k)-1.)
      write(ifhi,'(2e11.3)')b,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
         endif
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distr unitarized theory '
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1,a)')  'openhisto name b',j,'Unit'
      write(ifhi,'(a)')       'htyp lbf xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1+(k-0.5)*db
      if(j.eq.1)y=1
      if(j.eq.2)y=(1.-uuo2(k))
      if(j.eq.3)y=uuo2(k)
      if(j.eq.4.or.j.eq.6)y=(1.-uuo2(k)*uu3(k))
      if(j.eq.5)y=uuo2(k)*(uu3(k)-1.)
      write(ifhi,'(2e11.3)')b,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distr for cross section '
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1,a)')  'openhisto name b',j,'Unit'
      write(ifhi,'(a)')       'htyp lge xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1+(k-0.5)*db
      if(j.eq.1)y=1
      if(j.eq.2)y=(1.-(uuo2(k)+uua2(k))*0.5)
      if(j.eq.3)y=(uuo2(k)+uua2(k))*0.5
      if(j.eq.4.or.j.eq.6)y=(1.-(uuo2(k)+uua2(k))*0.5*uu3(k))
      if(j.eq.5)y=(uuo2(k)+uua2(k))*0.5*(uu3(k)-1.)
      write(ifhi,'(2e11.3)')b,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distribution simulation'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1,a)')  'openhisto name b',j,'Simu'
      write(ifhi,'(a)')       'htyp lrf xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.0,b2
      write(ifhi,'(a,2e11.3)')'yrange',0.,ymax
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      if(j.eq.1)write(ifhi,'(a)')'text 0.1 0.35 "after Metropolis"'
      if(j.eq.1)write(ifhi,'(a)')'text 0.2 0.20 "all "'
      if(j.eq.2)write(ifhi,'(a)')'text 0.3 0.85 "after Metropolis"'
      if(j.eq.2)write(ifhi,'(a)')'text 0.5 0.70 "interaction "'
      if(j.eq.3)write(ifhi,'(a)')'text 0.3 0.85 "nothing"'
      if(j.eq.4)write(ifhi,'(a)')'text 0.3 0.85 "cut"'
      if(j.eq.5)write(ifhi,'(a)')'text 0.3 0.85 "diffr"'
      if(j.eq.6)write(ifhi,'(a)')'text 0.3 0.85 "cut + diffr cut"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      x=b1+(k-0.5)*db
      if(j.eq.1)y=fk*w(j,k)/nn(1)/(pi*((x+0.5*db)**2-(x-0.5*db)**2))
      if(j.ne.1)y=0.
      if(j.ne.1.and.w(1,k).ne.0.)y=w(j,k)/w(1,k)
      if(nn(j).gt.0)write(ifhi,'(2e11.3)')x,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

   1  continue

      koll=kollini

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsBg(iii,jjj,ko)
c-----------------------------------------------------------------------
c b distribution at different stages for different group
c arguments:
c   iii:  modus (0,1,2,3)
c   jjj:  group of interaction (1,2 ... ,7)
c   ko:   pair number (1 - AB)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      parameter(njjj=7)
      parameter(nbib=16)
      common/cxemsb4/wg(-1:njjj,nbib),nng(nbib),uug(nbib),kollx
      common/cxemsb5/dbg,b1g,b2g
      common/cxemsb6/njjj0
      double precision seedp,PhiExpo!,PhiExact
      common/geom/rmproj,rmtarg,bmax,bkmx

      if(iemsbg.eq.0)call utstop('ERROR in XemsBg: iemsbg = 0&')

        if(iii.eq.0)then

      do k=1,nbib
       nng(k)=0
       do j=-1,njjj
        wg(j,k)=0
       enddo
      enddo
      njjj0=0
      kollx=0

        elseif(iii.eq.1)then

      b1g=0
      b2g=bkmx*1.2
      dbg=(b2g-b1g)/nbib
      k=int((bk(ko)-b1g)/dbg)+1
      if(k.gt.nbib)return
      if(k.lt.1)return
      if(jjj.eq.-1.or.jjj.eq.0)then
        wg(jjj,k)=wg(jjj,k)+1
      else
        wg(jjj,k)=wg(jjj,k)+1
        nng(k)=nng(k)+1
      endif
      if(jjj.eq.0)njjj0=1

        elseif(iii.eq.3)then

          call ranfgt(seedp)
          do k=1,koll
            om1i=sngl(om1intc(k))
            if(rangen().lt.1.-exp(-om1i))then
c            om1i=sngl(PhiExpo(0.,0.,1.,1.d0,1.d0,engy*engy,bk(k)))
c            if(rangen().lt.1.-om1i)then
              kollx=kollx+1
            endif
          enddo
          call ranfst(seedp)

        elseif(iii.eq.2)then

      if(njjj0.ne.1)call utstop
     &('xEmsBg must be called also with jjj=0&')
      ymax=1.4
      kollini=koll
      koll=1

      wtot=1.
      if(matarg+maproj.gt.2)then
      wtot=0.
      do k=1,nbib
       wtot=wtot+wg(-1,k)
      enddo
      if(kollx.gt.0)wtot=wtot/float(kollx)
      endif

      do 1 j=1,njjj

      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distribution simulation'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1,a)')  'openhisto name bg',j,'Simu'
      write(ifhi,'(a)')       'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,b2g
      write(ifhi,'(a,2e11.3)')'yrange',0.,ymax
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      if(wtot.gt.0.d0)
     &write(ifhi,'(a,f7.4,a)')    'text 0.5 0.8 "alpha=',1./wtot,'"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1g+(k-0.5)*dbg
      y=0.
      if(nng(k).ne.0.and.wg(0,k).ne.0)
     &              y=wg(j,k)/float(nng(k))*wg(-1,k)/wg(0,k)!/wtot
c      if(wg(0,k).ne.0..and.nng(k).ne.0)y=wg(j,k)/nng(k)*wg(-1,k)/wg(0,k)
c!???????????? better normalization ? probability to have an interaction
c in epos compared to eikonal probability, instead of normalized by the
c probability of a collision for a pair (the number collision/number
c active pair).
      uug(k)=uug(k)+y
      write(ifhi,'(2e11.3)')b,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
   1  continue
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distr tot simul theory '
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')  'openhisto name btotSimu'
      write(ifhi,'(a)')       'htyp pfc xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1g+(k-0.5)*dbg
      write(ifhi,'(2e11.3)')b,uug(k)
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a)')   '! b distr unitarized theory '
      write(ifhi,'(a)')   '!##################################'
      write(ifhi,'(a,i1,a)')  'openhisto name bg',j,'Unit'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis impact parameter b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P(b)"'
      write(ifhi,'(a)')       'array 2'
         do k=1,nbib
      b=b1g+(k-0.5)*dbg
c      a1=Phiexact(0.,0.,1.,1.d0,1.d0,engy**2,b)
       a1=sngl(PhiExpo(0.,0.,1.,1.d0,1.d0,engy**2,b))
      y=(1.-a1)
      write(ifhi,'(2e11.3)')b,y
         enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      koll=kollini

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsPx(iii,xmc,ymc,npos)
c-----------------------------------------------------------------------
c plot  x-distribution and y-distribution of Pomerons
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx

      parameter(nbix=30,nbib=51)
      common/cx/x(2,nbix),dx(2,nbix),wxmc(2,nbix),wxmcI(2,nbix)
     * ,xl(2,nbix),dxl(2,nbix),wxp(2,nbix),wxm(2,nbix),wxpI(2,nbix)
     *,wxmI(2,nbix),wxpY(2,nbix),wxmY(2,nbix),wxmcY(2,nbix)
      parameter(nbiy=50)
      common/cy/y(nbiy),wymc(nbiy),wymcY(nbiy),wymcI(nbiy),nyp,nym
      double precision PomIncXExact,PomIncPExact,PomIncMExact,dcel
      double precision PomIncXIExact,PomIncPIExact,PomIncMIExact
      common/ems3/dcel,ad
      common/cemspx/xu,xo,yu,yo,dy,xlu,xlo,bb,nn,db,mm,nm,nt
      character mod*5, imod*5, txtxm*6

      nposi=5

      if(iemspx.eq.0)call utstop('ERROR in XemsPx: iemspx = 0&')

      if(iii.eq.0)then

       xu=0.1/engy**2
       xo=1.
       xlu=0.01/engy
       xlo=1.
       yu=-alog(engy**2)
       yo=alog(engy**2)
       dy=(yo-yu)/nbiy
        do i=1,nbix
        x(1,i)=xu*(xo/xu)**((i-0.5)/nbix)
        x(2,i)=xu+(xo-xu)*((i-0.5)/nbix)
        dx(1,i)=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
        dx(2,i)=(xo-xu)/nbix
        wxmc(1,i)=0.
        wxmc(2,i)=0.
        wxmcI(1,i)=0.
        wxmcI(2,i)=0.
        wxmcY(1,i)=0.
        wxmcY(2,i)=0.
       enddo
       do i=1,nbix
        xl(1,i)=xlu*(xlo/xlu)**((i-0.5)/nbix)
        xl(2,i)=xlu+(xlo-xlu)*((i-0.5)/nbix)
        dxl(1,i)=xlu*(xlo/xlu)**(1.*i/nbix)*(1.-(xlo/xlu)**(-1./nbix))
        dxl(2,i)=(xlo-xlu)/nbix
        wxp(1,i)=0.
        wxp(2,i)=0.
        wxm(1,i)=0.
        wxm(2,i)=0.
        wxpI(1,i)=0.
        wxpI(2,i)=0.
        wxmI(1,i)=0.
        wxmI(2,i)=0.
        wxpY(1,i)=0.
        wxpY(2,i)=0.
        wxmY(1,i)=0.
        wxmY(2,i)=0.
       enddo
       do i=1,nbiy
        y(i)=yu+dy/2.+float(i-1)*dy
        wymc(i)=0.
        wymcI(i)=0.
        wymcY(i)=0.
       enddo
       mm=0
       nt=0
       nyp=0
       nym=0
       db=bkmx*2./float(nbib-1)

      elseif(iii.eq.1)then

       xp=sqrt(xmc)*exp(ymc)
       xm=sqrt(xmc)*exp(-ymc)
       mm=mm+1

       if(xmc.lt.xu)goto11
       i=1+int(alog(xmc/xu)/alog(xo/xu)*nbix)
       if(i.gt.nbix)goto1
       if(i.lt.1)goto1
       wxmc(1,i)=wxmc(1,i)+1.
       if(npos.eq.1)    wxmcI(1,i)=wxmcI(1,i)+1.
       if(npos.eq.nposi)wxmcY(1,i)=wxmcY(1,i)+1.
1      continue
       i=1+int((xmc-xu)/(xo-xu)*nbix)
       if(i.gt.nbix)goto11
       if(i.lt.1)goto11
       wxmc(2,i)=wxmc(2,i)+1.
       if(npos.eq.1)    wxmcI(2,i)=wxmcI(2,i)+1.
       if(npos.eq.nposi)wxmcY(2,i)=wxmcY(2,i)+1.
11     continue

       if(xp.lt.xlu)goto12
       i=1+int(alog(xp/xlu)/alog(xlo/xlu)*nbix)
       if(i.gt.nbix)goto2
       if(i.lt.1)goto2
       wxp(1,i)=wxp(1,i)+1.
       if(npos.eq.1)    wxpI(1,i)=wxpI(1,i)+1.
       if(npos.eq.nposi)wxpY(1,i)=wxpY(1,i)+1.
2      continue
       i=1+int((xp-xlu)/(xlo-xlu)*nbix)
       if(i.gt.nbix)goto12
       if(i.lt.1)goto12
       wxp(2,i)=wxp(2,i)+1.
       if(npos.eq.1)    wxpI(2,i)=wxpI(2,i)+1.
       if(npos.eq.nposi)wxpY(2,i)=wxpY(2,i)+1.
12     continue

       if(xm.lt.xlu)goto13
       i=1+int(alog(xm/xlu)/alog(xlo/xlu)*nbix)
       if(i.gt.nbix)goto3
       if(i.lt.1)goto3
       wxm(1,i)=wxm(1,i)+1.
       if(npos.eq.1)    wxmI(1,i)=wxmI(1,i)+1.
       if(npos.eq.nposi)wxmY(1,i)=wxmY(1,i)+1.
3      continue
       i=1+int((xm-xlu)/(xlo-xlu)*nbix)
       if(i.gt.nbix)goto13
       if(i.lt.1)goto13
       wxm(2,i)=wxm(2,i)+1.
       if(npos.eq.1)    wxmI(2,i)=wxmI(2,i)+1.
       if(npos.eq.nposi)wxmY(2,i)=wxmY(2,i)+1.
13     continue

       if(ymc.lt.yu)return
       i=int((ymc-yu)/dy)+1
       if(i.gt.nbiy)return
       if(i.lt.1)return
       wymc(i)=wymc(i)+1
       if(npos.eq.1)    wymcI(i)=wymcI(i)+1
       if(npos.eq.nposi)wymcY(i)=wymcY(i)+1
       if(ymc.gt.0)nyp=nyp+1
       if(ymc.lt.0)nym=nym+1

      elseif(iii.eq.2)then

       if(maproj.eq.1.and.matarg.eq.1.and.bminim.eq.bmaxim)then
        mmmm=1
        bb=bmaxim
        ff=float(nrevt)/float(ntevt)
        imod='   dn'
       elseif(maproj.eq.1.and.matarg.eq.1)then
        mmmm=3
        ff=1.
        imod='   dn'
       elseif(bminim.lt.0.001.and.bmaxim.gt.20)then
        mmmm=2
        area=pi*(rmproj+rmtarg)**2
        ff=area*float(nrevt)/float(ntevt)/(maproj*matarg)/sigine*10
        imod='   dn'
       else
        write(ifmt,*)'xEmsPx ignored'
        return
       endif
       kollini=koll
       koll=1

       kk1=nint(xpar1)
       kk2=nint(xpar2)

       do kk=kk1,kk2

       if(kk.eq.1)mod=' log '
       if(kk.eq.2)mod=' lin '

       write(ifhi,'(a)')       '!----------------------------------'
       write(ifhi,'(a)')       '!   Pomeron x distribution    '//mod
       write(ifhi,'(a)')       '!----------------------------------'

       write(ifhi,'(a)')  'openhisto name xPomSimuL'//mod(3:4)
       write(ifhi,'(a)')  'htyp lru xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xu,xo
       write(ifhi,'(a)')    'text 0 0 "xaxis x?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx?PE!"'
       if(kk.eq.1)write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "f=',ff,'"'
       if(kk.eq.2)write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "f=',ff,'"'
       write(ifhi,'(a)')       'array 2'
       s1=0
       do i=1,nbix
       u=x(kk,i)
       z=ff*wxmc(kk,i)/dx(kk,i)/nrevt
       s1=s1+z*dx(kk,i)
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a)')    'closehisto plot 0-'

       write(ifhi,'(a)')       'openhisto name xPomUnitL'//mod(3:4)
       write(ifhi,'(a)')  'htyp lba xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xu,xo
       write(ifhi,'(a)')    'text 0 0 "xaxis x?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx?PE!"'
       write(ifhi,'(a)')       'array 2'
       s2=0
       do i=1,nbix
        u=x(kk,i)
        if(mmmm.eq.1)z=sngl(PomIncXExact(dble(u),bb))
        if(mmmm.eq.2)z=sngl(PomIncXIExact(dble(u)))/sigine*10
        if(mmmm.eq.3)z=sngl(PomIncXIExact(dble(u)))/sigine*10
        s2=s2+dx(kk,i)*z
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a,f5.3,a,f5.3,a)')
     *                       'text .1 .85 "I= ',s1,' (',s2,')"'
       write(ifhi,'(a)')    'closehisto plot 0'

       write(ifhi,'(a)')           '!--------------------------------'
       write(ifhi,'(a)')           '!   Pomeron y distribution   '//mod
       write(ifhi,'(a)')           '!--------------------------------'

       write(ifhi,'(a)')       'openhisto name yPomSimuL'//mod(3:4)
       write(ifhi,'(a)')       'htyp lru xmod lin ymod'//mod
       write(ifhi,'(a,2e11.3)')'xrange',yu,yo
       write(ifhi,'(a)')    'text 0 0 "xaxis y?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom!/dy?PE!"'
       write(ifhi,'(a,f5.2,a)')'text 0.1 0.7 "f=',ff,'"'
       write(ifhi,'(a)')       'array 2'
       s1=0
       do i=1,nbiy
       u=y(i)
       z=ff*wymc(i)/dy/nrevt
       s1=s1+z*dy
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a)')    'closehisto plot 0'

       write(ifhi,'(a)')       '!----------------------------------'
       write(ifhi,'(a)')       '!   Pomeron x+ distribution    '//mod
       write(ifhi,'(a)')       '!----------------------------------'

       write(ifhi,'(a)')   'openhisto name xpPomSimuL'//mod(3:4)
       write(ifhi,'(a)')   'htyp lru xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xlu,xlo
       write(ifhi,'(a)')    'text 0 0 "xaxis x+?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx+?PE!"'
       if(kk.eq.1)write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "f=',ff,'"'
       if(kk.eq.2)write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "f=',ff,'"'
       write(ifhi,'(a)')       'array 2'
       s1=0
       do i=1,nbix
       u=xl(kk,i)
       z=ff*wxp(kk,i)/dxl(kk,i)/nrevt
       s1=s1+z*dxl(kk,i)
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a)')    'closehisto plot 0-'

       write(ifhi,'(a)')       'openhisto name xpPomUnitL'//mod(3:4)
       write(ifhi,'(a)')   'htyp lba xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xlu,xlo
       write(ifhi,'(a)')    'text 0 0 "xaxis x+?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx+?PE!"'
       write(ifhi,'(a)')       'array 2'
       s2=0
       do i=1,nbix
        u=xl(kk,i)
        if(mmmm.eq.1)z=sngl(PomIncPExact(dble(u),bb))
        if(mmmm.eq.2)z=sngl(PomIncPIExact(dble(u)))/sigine*10
        if(mmmm.eq.3)z=sngl(PomIncPIExact(dble(u)))/sigine*10
        s2=s2+dxl(kk,i)*z
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a,f5.3,a,f5.3,a)')
     *                       'text .1 .85 "I= ',s1,' (',s2,')"'
       write(ifhi,'(a)')    'closehisto plot 0'

       write(ifhi,'(a)')       '!----------------------------------'
       write(ifhi,'(a)')       '!   x-?PE! distribution    '//mod
       write(ifhi,'(a)')       '!----------------------------------'

       write(ifhi,'(a)')   'openhisto name xmPomSimuL'//mod(3:4)
       write(ifhi,'(a)')   'htyp lru xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xlu,xlo
       write(ifhi,'(a)')    'text 0 0 "xaxis x-?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx-?PE!"'
       if(kk.eq.1)write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "f=',ff,'"'
       if(kk.eq.2)write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "f=',ff,'"'
       write(ifhi,'(a)')       'array 2'
       s1=0
       do i=1,nbix
       u=xl(kk,i)
       z=ff*wxm(kk,i)/dxl(kk,i)/nrevt
       s1=s1+z*dxl(kk,i)
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a)')    'closehisto plot 0-'

       write(ifhi,'(a)')       'openhisto name xmPomUnitL'//mod(3:4)
       write(ifhi,'(a)')   'htyp lba xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange',xlu,xlo
       write(ifhi,'(a)')    'text 0 0 "xaxis x-?PE!"'
       write(ifhi,'(a)') 'text 0 0 "yaxis'//imod//'?Pom! / dx-"'
       write(ifhi,'(a)')       'array 2'
       s2=0
       do i=1,nbix
        u=xl(kk,i)
        if(mmmm.eq.1)z=sngl(PomIncMExact(dble(u),bb))
        if(mmmm.eq.2)z=sngl(PomIncMIExact(dble(u))/sigine*10)
        if(mmmm.eq.3)z=sngl(PomIncMIExact(dble(u))/sigine*10)
        s2=s2+dxl(kk,i)*z
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       write(ifhi,'(a,f5.3,a,f5.3,a)')
     *                       'text .1 .85 "I= ',s1,' (',s2,')"'
       write(ifhi,'(a)')    'closehisto plot 0'

  !................................................................

       xm=-1. !xm integration
       txtxm='xm int'
       do jjb=0,3
       b=jjb*0.5
       do jj=0,2

       write(ifhi,'(a)')       '!----------------------------------'
       write(ifhi,'(a,3i1)')   '!   ffom11    '//mod,jjb,jj
       write(ifhi,'(a)')       '!----------------------------------'

       write(ifhi,'(a,2i1)')'openhisto name ffom11L'//mod(3:4),jjb,jj+8
       write(ifhi,'(a)')    'htyp lin xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange ',xlu,xlo
       write(ifhi,'(a)')'txt "xaxis  x+?PE!"'
       write(ifhi,'(a)')'txt "yaxis dn?Pom! / dx+?PE! "'
       write(ifhi,'(a)')'text 0.05 0.1  "fit and exact, all contrib."'
       if(jjb.lt.3)write(ifhi,'(a,f4.1,3a)')
     *             'txt "title ffom11   b =',b,'   ',txtxm,'"'
       if(jjb.ge.3)write(ifhi,'(3a)')
     *             'txt "title ffom11   b aver   ',txtxm,'"'
       write(ifhi,'(a)')       'array 2'
       do i=1,nbix
       u=xl(kk,i)
       if(jjb.lt.3.and.jj.eq.0)z= ffom11(u,xm,b,-1,-1)
       if(jjb.lt.3.and.jj.eq.1)z= ffom11(u,xm,b,0,5)
       if(jjb.lt.3.and.jj.eq.2)z= ffom11(u,xm,b,0,4)
       if(jjb.eq.3.and.jj.eq.0)z=ffom11a(u,xm,-1,-1)
       if(jjb.eq.3.and.jj.eq.1)z=ffom11a(u,xm,0,5)
       if(jjb.eq.3.and.jj.eq.2)z=ffom11a(u,xm,0,4)
        write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       if(jj.le.1)write(ifhi,'(a)')    'closehisto plot 0-'
       if(jj.eq.2)write(ifhi,'(a)')    'closehisto plot 0'

       enddo
       enddo

       do jjb=0,3
       b=jjb*0.5
       do jjj=1,6
       jj=jjj
       if(jjj.eq.6)jj=0

       write(ifhi,'(a)')       '!----------------------------------'
       write(ifhi,'(a,3i1)')   '!   ffom11    '//mod,jjb,jj
       write(ifhi,'(a)')       '!----------------------------------'

       write(ifhi,'(a,3i1)')'openhisto name om1ffL'//mod(3:4),jjb,jj
       if(jj.ne.0)write(ifhi,'(a)')    'htyp lin xmod'//mod//'ymod log'
       if(jj.eq.0)write(ifhi,'(a)')    'htyp lro xmod'//mod//'ymod log'
       write(ifhi,'(a,2e11.3)')'xrange ',xlu,xlo
       if(jj.eq.1)then
       write(ifhi,'(a)') 'txt "xaxis  x+?PE!"'
       write(ifhi,'(a)') 'txt "yaxis  dn?Pom! / dx+?PE!  "'
       if(kk.eq.2)then
        write(ifhi,'(a)') 'text 0.1 0.2  "soft sea-sea"'
        write(ifhi,'(a)') 'text 0.1 0.1  "val-sea sea-val val-val"'
       else
        write(ifhi,'(a)') 'text 0.05 0.8  "soft"'
        write(ifhi,'(a)') 'text 0.05 0.7  "diff"'
        write(ifhi,'(a)') 'text 0.05 0.6  "sea-sea"'
        write(ifhi,'(a)') 'text 0.05 0.5  "val-sea"'
        write(ifhi,'(a)') 'text 0.05 0.4  "sea-val"'
        write(ifhi,'(a)') 'text 0.05 0.3  "val-val"'
      endif
       if(jjb.lt.3)write(ifhi,'(a,f4.1,3a)')
     *             'txt "title ffom11   b =',b,'  ',txtxm,'"'
       if(jjb.ge.3)write(ifhi,'(3a)')
     *             'txt "title ffom11   b aver  ',txtxm,'"'
       endif
       write(ifhi,'(a)')       'array 2'
       do i=1,nbix
       u=xl(kk,i)
       if(jjb.lt.3)z= ffom11(u,xm,b,jj,jj)
       if(jjb.eq.3)z=ffom11a(u,xm,jj,jj)
       write(ifhi,'(2e11.3)')u,z
       enddo
       write(ifhi,'(a)')    '  endarray'
       if(jjj.ne.6)write(ifhi,'(a)')    'closehisto plot 0-'
       if(jjj.eq.6)write(ifhi,'(a)')    'closehisto plot 0'

       enddo
       enddo

      enddo

      koll=kollini
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsP2(iii,jaa,jex,xpd,xmd,xpb,xmb,pt1,pt2)
c-----------------------------------------------------------------------
c plot  x+ distributions of Pomeron ends (PE) (xpd)
c          and Pomeron's in Born (IB) partons (xpb),
c     and pt dist of Pomeron's out Born (OB) partons
c       integrated over x- bins (xmd,xmb)
c  iii=0: initialize
c  ii=1: fill arrays
c  iii>=2: make histogram
c           (2 - Pomeron end PE, 3 - in Born IB, 4 - out Born OB)
c  jaa: type of semihard Pomeron
c         0= sea-sea diff, 
c         1= sea-sea, 2= val=sea, 3= sea-val, 4= val-val
c         5= all  for iii=2
c  jex: emission type
c         1= no emission, 2= proj emis, 3= targ emis, 4= both sides
c         5= all  for iii=2
c-----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbixp=25,nbixm=5,nbipt=20)
      common/cxb/xlp(2,nbixp),dxlp(2,nbixp)
     *          ,xlm(2,nbixm),dxlm(2,nbixm)
     *          ,wxb(2,0:4,4,nbixp,nbixm)
     *          ,wxe(2,0:4,4,nbixp,nbixm)
      common/cptb/ptu,pto,ptob(nbipt),wptob(0:4,4,nbipt)
      common/cemspbx/xlub1,xlub2,xlob
ctp060829      character imod*5

      if(iemspbx.eq.0)call utstop('ERROR in xEmsP2: iemspbx = 0&')

      if(iii.eq.0)then

       xlub1=0.01/engy
       xlub2=0.
       xlob=1.
       do i=1,nbixp
        xlp(1,i)=xlub1*(xlob/xlub1)**((i-0.5)/nbixp)
        xlp(2,i)=xlub2+(xlob-xlub2)*((i-0.5)/nbixp)
        dxlp(1,i)=xlub1*(xlob/xlub1)**(1.*i/nbixp)
     *             *(1.-(xlob/xlub1)**(-1./nbixp))
        dxlp(2,i)=(xlob-xlub2)/nbixp
       enddo
       do i=1,nbixm
        xlm(1,i)=xlub1*(xlob/xlub1)**((i-0.5)/nbixm)
        xlm(2,i)=xlub2+(xlob-xlub2)*((i-0.5)/nbixm)
        dxlm(1,i)=xlub1*(xlob/xlub1)**(1.*i/nbixm)
     *             *(1.-(xlob/xlub1)**(-1./nbixm))
        dxlm(2,i)=(xlob-xlub2)/nbixm
       enddo
       do i=1,nbixp
       do j=1,nbixm
       do jaai=0,4
       do jexi=1,4
        wxb(1,jaai,jexi,i,j)=0.
        wxb(2,jaai,jexi,i,j)=0.
        wxe(1,jaai,jexi,i,j)=0.
        wxe(2,jaai,jexi,i,j)=0.
       enddo
       enddo
       enddo
       enddo
       ptu=2
       pto=20
       do i=1,nbipt
       ptob(i)=ptu+(pto-ptu)*(i-0.5)/nbipt
       do jaai=0,4
       do jexi=1,4
       wptob(jaai,jexi,i)=0
       enddo
       enddo
       enddo

      elseif(iii.eq.1)then

       xp=xpb
       xm=xmb
       if(xp.lt.xlub1)goto2
       if(xm.lt.xlub1)goto2
       i=1+int(alog(xp/xlub1)/alog(xlob/xlub1)*nbixp)
       if(i.gt.nbixp)goto2
       if(i.lt.1)goto2
       j=1+int(alog(xm/xlub1)/alog(xlob/xlub1)*nbixm)
       if(j.gt.nbixm)goto2
       if(j.lt.1)goto2
       wxb(1,jaa,jex,i,j)=wxb(1,jaa,jex,i,j)+1.
2      continue

       if(xp.lt.xlub2)goto12
       if(xm.lt.xlub2)goto12
       i=1+int((xp-xlub2)/(xlob-xlub2)*nbixp)
       if(i.gt.nbixp)goto12
       if(i.lt.1)goto12
       j=1+int((xm-xlub2)/(xlob-xlub2)*nbixm)
       if(j.gt.nbixm)goto12
       if(j.lt.1)goto12
       wxb(2,jaa,jex,i,j)=wxb(2,jaa,jex,i,j)+1.
12     continue

       xp=xpd
       xm=xmd
       if(xp.lt.xlub1)goto22
       if(xm.lt.xlub1)goto22
       i=1+int(alog(xp/xlub1)/alog(xlob/xlub1)*nbixp)
       if(i.gt.nbixp)goto22
       if(i.lt.1)goto22
       j=1+int(alog(xm/xlub1)/alog(xlob/xlub1)*nbixm)
       if(j.gt.nbixm)goto22
       if(j.lt.1)goto22
       wxe(1,jaa,jex,i,j)=wxe(1,jaa,jex,i,j)+1.
  22   continue

       if(xp.lt.xlub2)goto32
       if(xm.lt.xlub2)goto32
       i=1+int((xp-xlub2)/(xlob-xlub2)*nbixp)
       if(i.gt.nbixp)goto32
       if(i.lt.1)goto32
       j=1+int((xm-xlub2)/(xlob-xlub2)*nbixm)
       if(j.gt.nbixm)goto32
       if(j.lt.1)goto32
       wxe(2,jaa,jex,i,j)=wxe(2,jaa,jex,i,j)+1.
  32   continue

       do m=1,2
       if(m.eq.1)pt=pt1
       if(m.eq.2)pt=pt2
       i=1+int((pt-ptu)/(pto-ptu)*nbipt)
       if(i.lt.1)goto42
       if(i.gt.nbipt)goto42
       wptob(jaa,jex,i)=wptob(jaa,jex,i)+1
   42  continue
       enddo

      elseif(iii.ge.2)then

       if(maproj.eq.1.and.matarg.eq.1.and.bminim.eq.bmaxim)then
ctp060829        mmmm=1
ctp060829        bb=bmaxim
        ff=float(nrevt)/float(ntevt)
ctp060829        imod='   dn'
       elseif(maproj.eq.1.and.matarg.eq.1)then
ctp060829        mmmm=3
        ff=1.
ctp060829        imod='   dn'
       elseif(bminim.lt.0.001.and.bmaxim.gt.20)then
ctp060829        mmmm=2
        area=pi*(rmproj+rmtarg)**2
        ff=area*float(nrevt)/float(ntevt)/(maproj*matarg)/sigine*10
ctp060829        imod='   dn'
       else
        write(ifmt,*)'xEmsP2 ignored'
        return
       endif

       j1=1  !nint(xpar1)   !first xminus bin
       j2=5  !nint(xpar2)   !last xminus bin
       if(iii.eq.4)j2=1
       kkk=2 !nint(xpar3)   !1 (log binning) 2 (lin binning)
       if(kkk.eq.1)then
ctp060829         xmi1=xlub1*(xlob/xlub1)**((j1-1.)/nbixm)
ctp060829         xmi2=xlub1*(xlob/xlub1)**((j2-0.)/nbixm)
         xlub=xlub1
       elseif(kkk.eq.2)then
ctp060829         xmi1=xlub2+(xlob-xlub2)*((j1-1.)/nbixm)
ctp060829         xmi2=xlub2+(xlob-xlub2)*((j2-0.)/nbixm)
         xlub=xlub2
       endif

       jaa1=jaa
       jaa2=jaa
       jex1=jex
       jex2=jex
       if(jaa.eq.5)then
       jaa1=0
       jaa2=4
       endif
       if(jex.eq.5)then
       jex1=1
       jex2=4
       endif

       if(jex.eq.1)then
        je1=0
        je2=0
       elseif(jex.eq.2)then
        je1=1
        je2=0
       elseif(jex.eq.3)then
        je1=0
        je2=1
       elseif(jex.eq.4)then
        je1=1
        je2=1
       elseif(jex.eq.5)then
        je1=2
        je2=2
       endif

       if(iii.eq.2)then

        write(ifhi,'(a)')       '!----------------------------------'
        write(ifhi,'(a,3i1)')   '!   PE    ',jaa,jex
        write(ifhi,'(a)')       '!----------------------------------'

        sum=ffom12aii(max(1,jaa),je1,je2)
        write(ifhi,'(a,2i1)')'openhisto name ffom12a',jaa,jex
        write(ifhi,'(a)')'htyp lin xmod lin ymod log'
        write(ifhi,'(a,2e11.3)')'xrange ',xlub,xlob
        write(ifhi,'(a)')    'txt "xaxis  x+?PE!"'
        write(ifhi,'(a)')    'txt "yaxis dn?semi! / dx+?PE!    "'
       write(ifhi,'(a,2i1,a)')'txt "title ffom12a + MC   (',jaa,jex,')"'
        write(ifhi,'(a)')    'array 2'
        do i=1,nbixp
         u=xlp(kkk,i)
         z=ffom12ai(u,max(1,jaa1),jaa2,je1,je2)
         write(ifhi,'(2e11.3)')u,z
        enddo
        write(ifhi,'(a)')    '  endarray'
        if(jex.eq.5)then
          write(ifhi,'(a)')    'closehisto plot 0-'
          write(ifhi,'(a,2i1)')'openhisto name ffom11',jaa,jex
          write(ifhi,'(a)')'htyp lba'
          write(ifhi,'(a)')'text 0.05 0.5 "+ ffom11a "'
          write(ifhi,'(a)')'array 2'
          do i=1,nbixp
           u=xlp(kkk,i)
           z=ffom11a(u,-1.,max(1,jaa1),jaa2)
           write(ifhi,'(2e11.3)')u,z
          enddo
          write(ifhi,'(a)')    '  endarray'
        endif

       elseif(iii.eq.3)then

        write(ifhi,'(a)')       '!----------------------------------'
        write(ifhi,'(a,3i1)')   '!   IB    ',jaa,jex
        write(ifhi,'(a)')       '!----------------------------------'

    !.......total integral
        s2min=4*q2min
        zmin=s2min/engy**2
        zmax=1
        xpmin0 = 0.01/engy
        xpmax=1
        ig1=3
        ig2=3
        r1=0
        do i1=1,ig1
        do m1=1,2
          z=zmin*(zmax/zmin)**(.5+tgss(ig1,i1)*(m1-1.5))
          xpmin=max(z,xpmin0)
          r2=0
          if(xpmin.lt.xpmax)then
          do i2=1,ig2
          do m2=1,2
            xp=xpmin*(xpmax/xpmin)**(.5+tgss(ig2,i2)*(m2-1.5))
            xm=z/xp
            r2=r2+wgss(ig2,i2)*ffsigiut(xp,xm,max(1,jaa),je1,je2)
          enddo
          enddo
          endif
          r2=r2*0.5*log(xpmax/xpmin)
          r1=r1+wgss(ig1,i1)*r2*z
        enddo
        enddo
        r1=r1*0.5*log(zmax/zmin)
        res=  r1 * factk * .0390  /sigine*10
        sum=res
   !.......plot
        xx2min = 0.01/engy     !max(xpar1,0.01/engy)
        xx2max = 1             !xpar2
        xx1min = 0.01/engy     !max(xpar3,0.01/engy)
        xx1max = 1             !xpar4
        nbins  = 10            !nint(xpar5)

        write(ifhi,'(a,2i1)') 'openhisto xrange 0 1 name ffsig',jaa,jex
        write(ifhi,'(a)') 'yrange auto auto htyp lin xmod lin ymod log'
        write(ifhi,'(a)') 'txt "xaxis x+?IB!         "              '
        write(ifhi,'(a)') 'txt "yaxis dn?semi! / dx+?IB!  "'
        write(ifhi,'(a,2i1,a)')'txt "title ffsig + MC   (',jaa,jex,')"'
        write(ifhi,'(a)') 'array 2'
        del=(xx1max-xx1min)/nbins
        do ii=1,nbins
          xx1=xx1min+(ii-0.5)*del
          ig2=3
          r2=0
          do i2=1,ig2
          do m2=1,2
            xx2=xx2min*(xx2max/xx2min)**(.5+tgss(ig2,i2)*(m2-1.5))
            r2=r2+wgss(ig2,i2)*ffsigiut(xx1,xx2,max(1,jaa),je1,je2)*xx2
          enddo
          enddo
          sig=r2*0.5*log(xx2max/xx2min)
          sig   = sig * factk * .0390   /sigine*10
          write(ifhi,'(2e12.4)')xx1,sig
        enddo
        write(ifhi,'(a)')  '  endarray'

       elseif(iii.eq.4)then

        write(ifhi,'(a)')       '!----------------------------------'
        write(ifhi,'(a,3i1)')   '!   OB    ',jaa,jex
        write(ifhi,'(a)')       '!----------------------------------'

      !...... integral
        y2     = 10
        ptmin  = 2
        ptmax  = 6
        sum=0
        ig=2
        do i=1,ig
        do m=1,2
              pt=ptmin*(ptmax/ptmin)**(.5+tgss(ig,i)*(m-1.5))
          sig=ffsigi(pt**2,y2)
          sig   =sig    * factk * .0390 /sigine*10  * 2   ! 2 partons!
              sum=sum+wgss(ig,i)*sig*pt
        enddo
        enddo
        sum=sum*0.5*log(ptmax/ptmin)
      !...... pt distr
        y2     = 10
        ptmin  = 2
        ptmax  = 20
        nbins  = 18
        sx=engy**2
        do jj=3,1,-1
        write(ifhi,'(a,i1)')'openhisto name jet',jj
        write(ifhi,'(a)')'xrange 0 20 xmod lin ymod log '
        write(ifhi,'(a)') 'txt "xaxis pt?OB!         "           '
        write(ifhi,'(a)') 'txt "yaxis dn?ptn! / dpt?OB!  "'
        if(jj.eq.1)write(ifhi,'(a)')'htyp lro'
        if(jj.eq.2)write(ifhi,'(a)')'htyp lgo'
        if(jj.eq.3)write(ifhi,'(a)')'htyp lyo'
        write(ifhi,'(a,f7.2,a)')  'text 0.05 0.1 "1/f=',1./ff,'"'
        write(ifhi,'(a)')'array 2'
        delpt=(ptmax-ptmin)/nbins
        do i=1,nbins
          pt=ptmin+(i-0.5)*delpt
          sig=1
          if(jj.eq.1)then
            sig=ffsigi(pt**2,y2)      ! our stuff
          elseif(jj.eq.2)then
            if(engy.ge.10.)sig=psjvrg1(pt**2,sx,y2) ! grv
          elseif(jj.eq.3)then
            if(engy.ge.10.)sig=psjwo1(pt**2,sx,y2)   !duke-owens
          endif
          sig   =sig    * factk * .0390 /sigine*10 * 2
          write(ifhi,'(2e12.4)')pt,sig
        enddo
        write(ifhi,'(a)')       '  endarray'
        if(jj.ne.1)write(ifhi,'(a)')       'closehisto'
        if(jj.ne.1)write(ifhi,'(a)')  'plot 0-'
        enddo

       endif

       x=0.1+(min(3,iii)-2)*0.30
       y=0.2+(min(3,iii)-2)*0.55
       if(engy.gt.100.)then
       write(ifhi,'(a,2f5.2,a,f6.3,a)')'text',x,y,' "   form ',sum,'"'
       else
       write(ifhi,'(a,2f5.2,a,f6.5,a)')'text',x,y,' "   form ',sum,'"'
       endif
       write(ifhi,'(a)')  'closehisto plot 0-'

       write(ifhi,'(a)') "!-----------------------------"
       write(ifhi,'(a)') "! MC   "
       write(ifhi,'(a)') "!-----------------------------"

       if(iii.eq.2)
     *  write(ifhi,'(a,i1,i1)')'openhisto name dndxPE',jaa,jex
       if(iii.eq.3)
     *  write(ifhi,'(a,i1,i1)')'openhisto name dndxIB',jaa,jex
       if(iii.eq.4)
     *  write(ifhi,'(a,i1,i1)')'openhisto name dndptOB',jaa,jex
       write(ifhi,'(a)')     'htyp prs'
       write(ifhi,'(a)')     'array 2'
       sum=0
       imax=nbixp
       if(iii.eq.4)imax=nbipt
       do i=1,imax
        u=xlp(kkk,i)
        if(iii.eq.4)u=ptob(i)
        z=0
        do j=j1,j2
        do jaai=jaa1,jaa2
        do jexi=jex1,jex2
         if(iii.eq.2)z=z+wxe(kkk,jaai,jexi,i,j)
         if(iii.eq.3)z=z+wxb(kkk,jaai,jexi,i,j)
         if(iii.eq.4)z=z+wptob(jaai,jexi,i)
        enddo
        enddo
        enddo
        del=dxlp(kkk,i)
        if(iii.eq.4)del=(pto-ptu)/nbipt
        z=z/del*ff/nrevt
        write(ifhi,'(2e11.3)')u,z
        sum=sum+z*del
       enddo
       write(ifhi,'(a)')    '  endarray'
       x=0.1+(min(3,iii)-2)*0.30
       y=0.1+(min(3,iii)-2)*0.55
       if(engy.gt.100)then
       write(ifhi,'(a,2f5.2,a,f6.3,a)')'text',x,y,' "   simu ',sum,'"'
       else
       write(ifhi,'(a,2f5.2,a,f6.5,a)')'text',x,y,' "   simu ',sum,'"'
       endif
       write(ifhi,'(a)')    'closehisto'

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsSe(iii,xmc,ptmc,ih,iqq)
c-----------------------------------------------------------------------
c     iqq = 1 : String End mass and rapidity
c     iqq = 2 : String mass and rapidity
c-----------------------------------------------------------------------

      include 'epos.inc'

      parameter(nbix=50)
      common/cxpar/nx(2),x(nbix),wxmc(nbix,2),xmn,xmx,xu,xo
      parameter(nbiy=40)
      common/cypar/ny(2),y(nbiy),wymc(nbiy,2),ymin,ymax,dy,yu,yo

      s=engy**2

      if(iii.eq.0)then

       nx(iqq)=0
       xu=0.1/engy**2
       xo=1.
       do i=1,nbix
         x(i)=xu*(xo/xu)**((i-0.5)/nbix)
         wxmc(i,iqq)=0
       enddo
       yo=alog(s)
       yu=-yo
       dy=(yo-yu)/nbiy
       ny(iqq)=0
       do i=1,nbiy
         y(i)=yu+dy/2.+(i-1)*dy
         wymc(i,iqq)=0
       enddo

      elseif(iii.eq.1)then

       if(xmc.lt.xu)return
       if(ptmc.eq.0.)return
       ymc=0.
       if(iqq.eq.1)ymc=0.5*alog(xmc*s/ptmc)*ih
       if(iqq.eq.2)ymc=0.5*alog(xmc/ptmc)
       i=1+int(alog(xmc/xu)/alog(xo/xu)*nbix)
       if(i.gt.nbix)goto1
       if(i.lt.1)goto1
       wxmc(i,iqq)=wxmc(i,iqq)+1
       nx(iqq)=nx(iqq)+1
1      continue
       if(ymc.lt.yu)return
       i=int((ymc-yu)/dy)+1
       if(i.gt.nbiy)return
       if(i.lt.1)return
       wymc(i,iqq)=wymc(i,iqq)+1
       ny(iqq)=ny(iqq)+1

      elseif(iii.eq.2)then

       write(ifhi,'(a)')        '!--------------------------------'
       write(ifhi,'(a)')        '!   string end x distr       '
       write(ifhi,'(a)')        '!--------------------------------'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lin'
        write(ifhi,'(a)')       'xmod log ymod log'
        write(ifhi,'(a,2e11.3)')'xrange',xu,xo
        if(iqq.eq.1)write(ifhi,'(a)')    'text 0 0 "xaxis string end x"'
        if(iqq.eq.2)write(ifhi,'(a)')    'text 0 0 "xaxis string x"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(x)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbix
         dx=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
         if(nx(iqq).gt.0)
     *   write(ifhi,'(2e11.3)')x(i),wxmc(i,iqq)/dx/nx(iqq)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto plot 0'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lin'
        write(ifhi,'(a)')       'xmod lin ymod lin'
        write(ifhi,'(a,2e11.3)')'xrange',yu,yo
        if(iqq.eq.1)write(ifhi,'(a)')    'text 0 0 "xaxis string end y"'
        if(iqq.eq.2)write(ifhi,'(a)')    'text 0 0 "xaxis string y"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(y)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbiy
         if(ny(iqq).gt.0)
     *   write(ifhi,'(2e11.3)')y(i),wymc(i,iqq)/dy/ny(iqq)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto plot 0'
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine xEmsDr(iii,xpmc,xmmc,ie)
c-----------------------------------------------------------------------

      include 'epos.inc'

      parameter(nbix=50,nie=4)
      common/cxpardr/nxp(nie),nxm(nie),x(nbix),wxpmc(nbix,nie)
     &      ,wxmmc(nbix,nie),xmn,xmx,xu,xo,wxmc(nbix,nie),nx(nie)
      parameter(nbiy=40)
      common/cypardr/ny(nie),y(nbiy),wymc(nbiy,nie),ymin,ymax,dy,yu,yo

      s=engy**2

      if(iii.eq.0)then

       do ni=1,nie
         nxp(ni)=0
         nxm(ni)=0
         nx(ni)=0
       enddo
       xu=0.1/engy**2
       xo=1.
       do i=1,nbix
         x(i)=xu*(xo/xu)**((i-0.5)/nbix)
         do ni=1,nie
           wxpmc(i,ni)=0
           wxmmc(i,ni)=0
           wxmc(i,ni)=0
         enddo
       enddo
       yo=alog(s)
       yu=-yo
       dy=(yo-yu)/nbiy
       do ni=1,nie
         ny(ni)=0
       enddo
       do i=1,nbiy
         y(i)=yu+dy/2.+(i-1)*dy
         do ni=1,nie
           wymc(i,ni)=0
         enddo
       enddo

      elseif(iii.eq.1)then

       if(ie.lt.1.or.ie.gt.nie)return

       if(xpmc.lt.xu)return
       i=1+int(alog(xpmc/xu)/alog(xo/xu)*nbix)
       if(i.gt.nbix)goto1
       if(i.lt.1)goto1
       wxpmc(i,ie)=wxpmc(i,ie)+1
       nxp(ie)=nxp(ie)+1
       if(xmmc.lt.xu)return
       i=1+int(alog(xmmc/xu)/alog(xo/xu)*nbix)
       if(i.gt.nbix)goto1
       if(i.lt.1)goto1
       wxmmc(i,ie)=wxmmc(i,ie)+1
       nxm(ie)=nxm(ie)+1
1      continue
       if(xmmc.ge.xu)then
         ymc=0.5*alog(xpmc/xmmc)
       else
         return
       endif
       if(ymc.lt.yu)return
       i=int((ymc-yu)/dy)+1
       if(i.gt.nbiy)return
       if(i.lt.1)return
       wymc(i,ie)=wymc(i,ie)+1
       ny(ie)=ny(ie)+1

       xmc=xpmc*xmmc
       if(xmc.lt.xu)return
       i=1+int(alog(xmc/xu)/alog(xo/xu)*nbix)
       if(i.gt.nbix)return
       if(i.lt.1)return
       wxmc(i,ie)=wxmc(i,ie)+1
       nx(ie)=nx(ie)+1

      elseif(iii.eq.2)then

        do ii=1,nie

       if(ii.eq.1)write(ifhi,'(a)')'!-----  projectile droplet  ----'
       if(ii.eq.2)write(ifhi,'(a)')'!-----    target droplet    ----'
       if(ii.eq.3)write(ifhi,'(a)')'!-----  projectile string end  ----'
       if(ii.eq.4)write(ifhi,'(a)')'!-----    target string end    ----'
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       '!   droplet/string x+ distr       '
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lru'
        write(ifhi,'(a)')       'xmod log ymod log'
        write(ifhi,'(a,2e11.3)')'xrange',xu,xo
        if(ii.eq.1.or.ii.eq.2)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis droplet x+"'
        if(ii.eq.3.or.ii.eq.4)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis string end x+"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(x)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbix
         dx=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
         if(nxp(ii).gt.0)
     *   write(ifhi,'(2e11.3)')x(i),wxpmc(i,ii)/dx/nxp(ii)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto plot 0-'
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       '!   droplet/string x- distr       '
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lba'
        write(ifhi,'(a)')       'xmod log ymod log'
        write(ifhi,'(a,2e11.3)')'xrange',xu,xo
        if(ii.eq.1.or.ii.eq.2)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis droplet x-"'
        if(ii.eq.3.or.ii.eq.4)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis string end x-"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(x)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbix
         dx=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
         if(nxm(ii).gt.0)
     *   write(ifhi,'(2e11.3)')x(i),wxmmc(i,ii)/dx/nxm(ii)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto plot 0'
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       '!   droplet/string y distr       '
        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lin'
        write(ifhi,'(a)')       'xmod lin ymod lin'
        write(ifhi,'(a,2e11.3)')'xrange',yu,yo
        if(ii.eq.1.or.ii.eq.2)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis droplet y"'
        if(ii.eq.3.or.ii.eq.4)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis string end y"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(y)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbiy
         if(ny(ii).gt.0)
     *   write(ifhi,'(2e11.3)')y(i),wymc(i,ii)/dy/ny(ii)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    'closehisto plot 0'

      enddo

        write(ifhi,'(a)')       '!--------------------------------'
        write(ifhi,'(a)')       '!   droplet/string mass distr       '
        write(ifhi,'(a)')       '!--------------------------------'
      do ii=1,nie


        if(ii.eq.2.or.ii.eq.4)write(ifhi,'(a)')    'closehisto plot 0-'
        if(ii.eq.3)write(ifhi,'(a)')    'closehisto plot 0'
        write(ifhi,'(a)')       'openhisto'
        if(ii.eq.1.or.ii.eq.3)write(ifhi,'(a)')       'htyp lru'
        if(ii.eq.2.or.ii.eq.4)write(ifhi,'(a)')       'htyp lba'
        write(ifhi,'(a)')       'xmod log ymod log'
        write(ifhi,'(a,2e11.3)')'xrange',sqrt(xu*s),sqrt(s*xo)
        if(ii.eq.1.or.ii.eq.2)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis droplet mass (GeV)"'
        if(ii.eq.4.or.ii.eq.3)
     *  write(ifhi,'(a)')    'text 0 0 "xaxis string end mass (GeV)"'
        write(ifhi,'(a)')    'text 0 0 "yaxis P(x)"'
        write(ifhi,'(a)')       'array 2'
        do i=1,nbix
         dx=xu*(xo/xu)**(1.*i/nbix)*(1.-(xo/xu)**(-1./nbix))
         if(nx(ii).gt.0)
     *   write(ifhi,'(2e11.3)')sqrt(x(i)*s),wxmc(i,ii)/dx/nx(ii)
        enddo
        write(ifhi,'(a)')    '  endarray'
      enddo
       write(ifhi,'(a)')    'closehisto plot 0'

      endif

      return
      end

cc--------------------------------------------------------------------------
c      subroutine xtype(k,n,i1,i2,text)
cc--------------------------------------------------------------------------
c
c      include 'epos.inc'
c      include 'epos.incems'
c      parameter(itext=40)
c      character  text*40
c
c      imax=itext+1
c      do i=itext,1,-1
c      if(text(i:i).eq.'&')imax=i
c      enddo
c
c      ip=iproj(k)
c      it=itarg(k)
c
c      if(i1.eq.1)then
c         write(ifch,*)
c         write(ifch,*)('-',ll=1,27)
c         write(ifch,*)'  '//text(1:imax-1)
c         write(ifch,*)('-',ll=1,27)
c      endif
c
c      if(i2.eq.1)then
c         write(ifch,*)
c         write(ifch,*)'k:',k,'   n:',n,'   ip:',ip,'   it:',it
c         write(ifch,*)'bk:',bk(k)
c         if(n.ne.0)write(ifch,*)'idpr:',idpr(n,k)
c         write(ifch,*)'iep:',iep(ip),'   iet:',iet(it)
c         write(ifch,*)'idp:',idp(ip),'   idt:',idt(it)
c      endif
c
c      end
c
c------------------------------------------------------------------------
      subroutine XPrint(text)
c------------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision xpptot,xmptot,xpttot,xmttot
c      parameter(itext=15)
      character  text*(*)
      imax=index(text,'&')
      if(imax.gt.1)write(ifch,'(1x,a)')text(1:imax-1)

      write(ifch,'(a)')
     *' k:     itpr:   npr0: npr1: nprmx:   Pomeron id lattice:'
      do k=1,koll
       write(ifch,'(1x,i6,1x,i4,4x,i4,2x,i4,3x,i4,a3,$)')
     *              k,itpr(k),npr(0,k),npr(1,k),nprmx(k),'   '
       do n=1,nprmx(k)
        write(ifch,'(i2,$)')idpr(n,k)
       enddo
       write(ifch,*)' '
      enddo

      xpptot=0d0
      xmptot=0d0
      xpttot=0d0
      xmttot=0d0
      write(ifch,'(a)')' Pomeron xy lattice:'
      do k=1,koll
       do n=1,nprmx(k)
       xpptot=xpptot+xppr(n,k)
       xmttot=xmttot+xmpr(n,k)
        write(ifch,'(i6,1x,i2,1x,d10.3,1x,d10.3,3x,$)')
     *                  k,n,xpr(n,k),ypr(n,k)
       enddo
       write(ifch,*)' '
      enddo

      write(ifch,'(a)')' projectile remnants x+,x-,px,py,x,iep:'
      do ip=1,maproj
       xpptot=xpptot+xpp(ip)
       xmptot=xmptot+xmp(ip)
       write(ifch,'(i3,2x,5d12.3,i3)')ip,xpp(ip),xmp(ip),xxp(ip),xyp(ip)
     *                             ,xpos(ip),iep(ip)
      enddo

      write(ifch,'(a)')' target remnants x-,x+,px,py,x,iet:'
      do it=1,matarg
       xpttot=xpttot+xpt(it)
       xmttot=xmttot+xmt(it)
       write(ifch,'(i3,2x,5d12.3,i3)')it,xmt(it),xpt(it),xxt(it),xyt(it)
     *                             ,xtos(it),iet(it)
      enddo

      write(ifch,*)' remnant balance x+,x-:'
     &,(xpptot+xpttot)/dble(maproj)
     &,(xmptot+xmttot)/dble(matarg)
      end


c-------------------------------------------------------------------------
      subroutine xfom
c-------------------------------------------------------------------------
      include 'epos.inc'
      double precision fom,x
      write(ifhi,'(a)')     '!##################################'
      write(ifhi,'(a,i3)')  '!   fom     '
      write(ifhi,'(a)')     '!##################################'
      b=0.
      do i=1,6
        z=0.2*exp(0.8*i)
        xi=0.01+0.16*float(i-1)
        write(ifhi,'(a,i1)') 'openhisto name fom',i
        write(ifhi,'(a)')    'htyp lin xmod lin ymod log'
        write(ifhi,'(a)')    'xrange 0 1'
        write(ifhi,'(a)')    'yrange 0.1 1000 '
        write(ifhi,'(a)')    'text 0 0 "xaxis x "'
        write(ifhi,'(a)')    'text 0 0 "yaxis fom"'
        if(z.lt.10.)
     &   write(ifhi,'(a,f4.2,a,f4.1,a)')'text ',xi,' 0.9 "',z,'"'
        if(z.ge.10.)
     &   write(ifhi,'(a,f4.2,a,f4.0,a)')'text ',xi,' 0.9 "',z,'"'
        write(ifhi,'(a)')    'array 2'
        do n=1,99
          x=dble(n)*0.01d0
          write(ifhi,'(2e11.3)')x,fom(z,x,b)
        enddo
        write(ifhi,'(a)')    '  endarray'
        write(ifhi,'(a)')    '  closehisto '
        if(i.lt.6)write(ifhi,'(a)')    'plot 0-'
        if(i.eq.6)write(ifhi,'(a)')    'plot 0'
      enddo
      end


c-----------------------------------------------------------------------
      subroutine xbDens(jjj)
c-----------------------------------------------------------------------
c plots b distribution for all pairs
c----------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      common/geom/rmproj,rmtarg,bmax,bkmx

      if(jjj.eq.1)then
c prepare plot for xbDens
      if(ixbDens.eq.1)then
        iii=1     !proj
        Nnucla=0
        do ip=1,maproj
          if(lproj(ip).ne.0)then
            Nnucla=Nnucla+1
            do l=1,lproj(ip)
C...fix compilation warning             
C              k=kproj(ip,l)
C              b=bk(k)
C              i=1+int(b/bkmx*float(mxnucl))
              i=1+int(bk(kproj(ip,l))/bkmx*float(mxnucl)) 
              if(i.le.mxnucl)bnucl(i,iii)=bnucl(i,iii)+1.
            enddo
          endif
          if(lproj3(ip).ne.0)then
            do l=1,lproj3(ip)
C...fix compilation warning
C              k=kproj3(ip,l)
C              b=bk(k)
C              i=1+int(b/bkmx*float(mxnucl))
              i=1+int(bk(kproj3(ip,l))/bkmx*float(mxnucl))
              if(i.le.mxnucl)bnucl(i,iii+2)=bnucl(i,iii+2)+1.
            enddo
          endif
        enddo
        xbtot(iii)=xbtot(iii)+float(Nnucla)
        iii=2     !targ
        Nnucla=0
        do it=1,matarg
          if(ltarg(it).ne.0)then
            Nnucla=Nnucla+1
            do l=1,ltarg(it)
              k=ktarg(it,l)
              b=bk(k)
              i=1+int(b/bkmx*float(mxnucl))
              if(i.le.mxnucl)bnucl(i,iii)=bnucl(i,iii)+1.
            enddo
          endif
          if(ltarg3(it).ne.0)then
            do l=1,ltarg3(it)
              k=ktarg3(it,l)
              b=bk(k)
              i=1+int(b/bkmx*float(mxnucl))
              if(i.le.mxnucl)bnucl(i,iii+2)=bnucl(i,iii+2)+1.
            enddo
          endif
        enddo
        xbtot(iii)=xbtot(iii)+float(Nnucla)
      endif

      else

      if(xbtot(1).gt.0.)then
        xbtot(3)=xbtot(1)
        xbtot(4)=xbtot(2)
        write(ifhi,'(a)')       'openhisto'
        write(ifhi,'(a)')       'htyp lin name bdens'
        write(ifhi,'(a)')       '- txt "xaxis b (fm)" '
        write(ifhi,'(a)')       '+ txt "yaxis P(b) proj " '
        write(ifhi,'(a)')       '+ txt "yaxis P(b) targ " '
        write(ifhi,'(a)')       '+ txt "yaxis P(b) scr proj " '
        write(ifhi,'(a)')       '+ txt "yaxis P(b) scr targ " '
        write(ifhi,'(a)')       'array 5'
        db=bkmx/float(mxnucl)
        do j=1,mxnucl
          b=(j-0.5)*db
          d=pi*((b+db)**2-b**2)
          write(ifhi,'(2e12.4)') b,(bnucl(j,iii)/xbtot(iii)/d,iii=1,4)
        enddo
        write(ifhi,'(a)')       '  endarray'
        write(ifhi,'(a)')       'closehisto'
        write(ifhi,'(a)')       'plot bdens+1- plot bdens+2-'
        write(ifhi,'(a)')       'plot bdens+3- plot bdens+4 '
      endif

      endif

      end
