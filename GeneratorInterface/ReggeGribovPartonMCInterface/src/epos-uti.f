c-----------------------------------------------------------------------
      subroutine utresc(iret)
c-----------------------------------------------------------------------
c  if irescl=1 rescaling is done, otherwise the purpose of going through
c  this routine is to not change the seed in case of no rescaling
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision p1,esoll,ppp,seedp,psoll,pt1soll,pt2soll
      dimension p1(5),p2(4),p0(5,mamx+mamx),pini(mxptl)
      logical force,nolead(mxptl),lspec(mxptl),lim
      data scalmean/0./scalevt/0./
      save scalmean,scalevt
      call utpri('utresc',ish,ishini,4)

      errlim=0.005 !max(0.001,1./engy)
      if(iLHC.eq.1)errlim=max(0.00005,0.5/engy)

      iret=0
      nptlpt=iabs(maproj)+iabs(matarg)
      call ranfgt(seedp)        !not to change the seed ...
      if(nptl.le.nptlpt) goto 9999

      if(ish.ge.8)then
        call alistf('list before boost&')
      endif
      esoll=0.d0
      psoll=0.d0
      p1(1)=0.d0
      p1(2)=0.d0
      p1(3)=0.d0
      p1(4)=0.d0
      p2(3)=0.d0
      p2(4)=0.d0
      ipmax=4
      imin=nptlpt+1
      if(iappl.eq.1)then
        imin=1
        ipmax=2
        if(iLHC.eq.1)ipmax=0
      endif
c store projectile and target in p0 and sum pz an E in p1(3) and p1(4)
      do i=1,nptlpt
        nolead(i)=.false.
        do j=1,5
          p0(j,i)=dble(pptl(j,i))
        enddo
c calculate total energy of primaries
c       if(mod(istptl(i),10).eq.1)then
          do j=ipmax+1,4
            p1(j)=p1(j)+dble(pptl(j,i))
          enddo
c       endif
c calculate total energy of secondaries
        if(mod(istptl(i),10).eq.0)then
          do j=ipmax+1,4
            p2(j)=p2(j)+dble(pptl(j,i))
          enddo
        endif
      enddo
c fix secondaries counted in the system
      do i=nptlpt+1,nptl
       if(mod(istptl(i),10).eq.0)then
c check maximum energy
         if(iLHC.eq.1.and.pptl(4,i).gt.engy*0.51)then
           if(ish.ge.1)write(ifch,*)'Energy of particle too high !'
     &                                     ,i,ityptl(i),pptl(4,i)
c           call utstop('utresc&')
           if(ityptl(i).eq.48.or.ityptl(i).eq.58  !diffractive resonance
     &    .or.ityptl(i).eq.47.or.ityptl(i).eq.57)then   !active spectators
             pptl(4,i)=0.5*engy
             amt=sqrt(pptl(1,i)**2+pptl(2,i)**2+pptl(5,i)**2)
             pptl(3,i)=(pptl(4,i)+amt)*(pptl(4,i)-amt)
             if(pptl(3,i).gt.0.)then
               pptl(3,i)=sqrt(pptl(3,i))
             else
               iret=1
             endif
           else
             iret=1
           endif
           if(iret.eq.1)then
             if(ish.ge.1)write(ifch,*)'Warning in utresc: redo event...'
c             call utstop('Energy of particle too high !&')
             goto 9999
           endif
         endif
c fix pt (p1(1) and p2(1)) from secondaries
         do j=1,ipmax
           p1(j)=p1(j)+dble(pptl(j,i))
         enddo
c calculate total energy of secondaries
         do j=ipmax+1,4
           p2(j)=p2(j)+dble(pptl(j,i))
         enddo
         lspec(i)=.false.
         if(((ityptl(i).eq.45.or.ityptl(i).eq.48).and.maproj.ge.100)
     &  .or.((ityptl(i).eq.55.or.ityptl(i).eq.58).and.matarg.ge.100))
     &   lspec(i)=.true.
         if((abs(pptl(3,i)/pnullx).le.0.9
     & .and.abs(pptl(3,i)).gt.pptl(5,i)).or.lspec(i))then
           nolead(i)=.true.
c           write(ifch,*)'nolead',i
         else
           nolead(i)=.false.
c           write(ifch,*)'lead',i
         endif
       endif
      enddo
      psoll=max(dble(pnullx),abs(p1(3)))
c check if energy is conserved before boost
      if(iappl.eq.1)then
        diff=abs(p2(3)-p1(3))
        scal=p2(4)/p1(4)
        if(abs(scal-1.).le.errlim.and.abs(diff/psoll).lt.errlim
     &     .and.(iLHC.eq.0.or.
     &    (abs(p2(1)).lt.errlim.and.abs(p2(2)).lt.errlim)))then
          if(ish.ge.4) 
     & write (ifch,'(2x,a,2g14.6)') 'Energy OK: ',scal,abs(diff/psoll)
          goto 9999
        else
         diff=0.
         scal=1.
        endif
      endif
c calculate boost vector to have sum of pt=0 and sum of pz=0
      ppp=(p1(4)+p1(3))*(p1(4)-p1(3))-p1(2)*p1(2)-p1(1)*p1(1)
      if(ppp.gt.0.d0)then
        p1(5)=sqrt(ppp)
      else
        iret=1
        write(ifch,*)'p1',p1(1),p1(2),p1(3),p1(4),ppp
        if(ish.ge.2)write (ifch,*) 'Problem in utresc (1): redo event'
        write (ifmt,*) 'Problem in utresc (1): redo event'
c       call utstop('utresc&')
        goto 9999
      endif
      esoll=p1(5)
      if(ish.ge.4) write (ifch,'(a,5g14.6)') 'boost-vector: ',p1

c     trafo
c     -----
      pmax=0.d0
      npart=0
      pt1soll=0.d0
      pt2soll=0.d0
      do i=imin,nptl
        if(mod(istptl(i),10).le.1)then
          call utlob4(1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $         ,pptl(1,i),pptl(2,i),pptl(3,i),pptl(4,i))
          if(mod(istptl(i),10).eq.0.and.i.gt.nptlpt)then
            npart=npart+1
            pt1soll=pt1soll+dble(pptl(1,i))
            pt2soll=pt2soll+dble(pptl(2,i))
          endif
        endif
        if(i.gt.nptlpt.and.nolead(i))pmax=max(pmax,abs(pptl(3,i)))
      enddo

      if(ish.ge.6)then
        call alistf('list after boost&')
      endif

      if(ish.ge.5)write(ifch,'(a)')'--------rescale momenta----------'

      if(iLHC.eq.1)then

        if(ish.ge.6)write(ifch,*)'ptsoll:',pt1soll,pt2soll,npart
        pt1soll=pt1soll/dble(npart)
        pt2soll=pt2soll/dble(npart)
        do i=nptlpt+1,nptl
          if(mod(istptl(i),10).eq.0)then
            pptl(1,i)=pptl(1,i)-sngl(pt1soll)
            pptl(2,i)=pptl(2,i)-sngl(pt2soll)
            pptl(4,i)=sqrt(pptl(1,i)**2+pptl(2,i)**2
     &                    +pptl(3,i)**2+pptl(5,i)**2)
          endif
        enddo

      endif

      if(ish.ge.6)write(ifch,*)'esoll,psoll,pmax:',esoll,psoll,pmax

c     rescale momenta in rest frame
c     -----------------------------
      scal=1.
      diff0=0.
c      ndif0=1
      ferr=0.05
      force=.false.
      npart=nptl-imin+1
      do ipass=1,300
        sum=0.
        sum3=0.
        difft=diff0
        ndif=0
        nfirst=int(rangen()*float(npart))    !start list randomly
        do  i=0,npart-1
          j=imin+i+nfirst
          if(j.gt.nptl)j=imin+i+nfirst-npart
          if(mod(istptl(j),10).eq.0)then
c modified particles
            if(nolead(j))then
c            if(j.gt.nptlpt)then
c            if(abs(pptl(3,j))/pnullx.lt.0.9)then  !not spectator or diffraction
              if(scal.eq.1..and.abs(diff0).lt.1.e-6)then
                ndif=ndif+1
                pini(j)=pptl(3,j)
              else
                pptl3new=0.
                if( force .or.(
     &            ityptl(j)/10.eq.4.or.ityptl(j)/10.eq.5
     &                      ))then !try just remnant first
                  ndif=ndif+1
                  diff=sign(min(0.3*abs(pini(j)),
     &                      rangen()*abs(difft)),difft)
                  pptl3new=scal*(pptl(3,j)-diff)
c                  write(ifch,*)'par',j,pptl3new,pptl(3,j),diff,difft
c     &                 ,ndif,pmax,scal
                  if(abs(pptl3new).lt.pmax)then
c particle should not be too fast or too modified
                    if(abs(pptl3new-pini(j)).lt.ferr*abs(pini(j))
     &         .or.(lspec(j).and.abs(pptl3new).lt.abs(0.8*pini(j))))then
c                  write(ifch,*)'used'
                      difft=difft-diff
                      pptl(3,j)=scal*(pptl(3,j)-diff)
                      pptl(4,j)=sqrt(pptl(1,j)**2+pptl(2,j)**2
     *                     +pptl(3,j)**2+pptl(5,j)**2)
                    endif
                  endif
                endif
              endif
            endif
c sum over all particles
            sum=sum+pptl(4,j)
            sum3=sum3+pptl(3,j)
          endif
        enddo

        diff=sum3
        scal=sngl(esoll)/(sum-diff)
        if(ish.ge.6)write(ifch,*)
     $       'ipass,scal,diff/psoll,e,pz,ndif,f:'
     $   ,ipass,scal,diff/psoll,sum,sum3,ndif,force
        if(abs(scal-1.).le.errlim.and.abs(diff/psoll).lt.errlim)
     $  goto 300
        if(ndif.gt.0.and.(force.or.ipass.lt.150))then
c          ndif0=ndif
          diff0=diff
        elseif(abs(scal-1.).le.1e-2.and.abs(diff/psoll).lt.5e-2
     &                                           .and.iLHC.eq.0)then
          goto 300
        elseif(force)then
          if(ish.ge.2)
     $    write(ifmt,*)'Warning in utresc: no more allowed particle'
          goto 302
        else
          force=.true.
          ferr=0.1
          diff=0.
        endif
      enddo
 302  if(iLHC.eq.1)then
        lim=.not.(abs(scal-1.).le.errlim.and.abs(diff/psoll).lt.errlim)
      else
        lim=abs(scal)+abs(diff/psoll).gt.2.5
      endif
 
      if(ish.ge.1)then
        call utmsg('utrescl')
        write(ifch,*)'*****  scal=',scal,diff/psoll,lim
        call utmsgf
      endif

      if(lim)then
        if(ish.ge.1)then
          write(ifmt,*)'Warning in utresc !'
          write(ifch,'(a,i10,d25.15)')'redo EPOS event ...'
     &                                ,nint(seedj),seedc
        endif
        iret=1
        goto 9999
      endif

c     trafo
c     -----
 300  continue

      do i=1,nptl
        if(i.le.nptlpt)then
          do j=1,5
            pptl(j,i)=p0(j,i)
          enddo
        else
          if(mod(istptl(i),10).le.1)then
            call utlob4(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $           ,pptl(1,i),pptl(2,i),pptl(3,i),pptl(4,i))
          endif
        endif
      enddo

      if(ish.ge.4)call alist('list after rescaling&',1,nptl)

 9999 continue
      if(ish.ge.2)then
        scalevt=scalevt+1.
        scalmean=scalmean+scal
        write(ifch,*)' average rescaling factor: ',scalmean
     &                                            /scalevt
      endif
      call ranfst(seedp)        ! ... after this subroutine
      call utprix('utresc',ish,ishini,4)

      end

c-----------------------------------------------------------------------
      subroutine utghost(iret)
c-----------------------------------------------------------------------
c  if irescl=1 make particle on-shell if not
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision seedp
      call utpri('ughost',ish,ishini,4)

      iret=0
      nptlpt=iabs(maproj)+iabs(matarg)
      if(iappl.eq.6.or.iappl.eq.8)nptlpt=3   ! ee or DIS
      call ranfgt(seedp)        !not to change the seed ...
      if(nptl.le.nptlpt) goto 9999

      if(ish.ge.5)write(ifch,'(a)')'---------mark ghosts---------'

c     mark ghosts
c     -----------
      do  j=nptlpt+1,nptl
        if(istptl(j).le.1.and.pptl(4,j).gt.0.d0)then
          if(iLHC.eq.0.or.mod(abs(idptl(j)),10).le.1)then !for LHC tune don't fix mass of resonnances (to keep width)
            amass=pptl(5,j)
            call idmass(idptl(j),amass)
            if(abs(idptl(j)).gt.100.and.
     &       abs(pptl(5,j)-amass).gt.0.01*amass)then
              if(ish.ge.5)write(ifch,*)'wrong particle mass',j,idptl(j)
     &                                           ,pptl(5,j),amass
              amass=pptl(5,j)
              call idres(idptl(j),amass,idr,iadj)
              if(idr.ne.0)then
                pptl(5,j)=amass
                idptl(j)=idr
              else
                call idmass(idptl(j),amass)
                pptl(5,j)=amass
              endif
              call idtau(idptl(j),pptl(4,j),pptl(5,j),taugm)
              tivptl(2,j)=tivptl(1,j)+taugm*(-alog(rangen()))
            else
              pptl(5,j)=amass
            endif
          endif
          if(abs((pptl(4,j)+pptl(3,j))*(pptl(4,j)-pptl(3,j))
     $         -pptl(2,j)**2-pptl(1,j)**2-pptl(5,j)**2).gt.0.3
     $       .and.abs(1.-abs(pptl(3,j))/pptl(4,j)).gt.0.01)then
        !print*,'ghost',ityptl(j),idptl(j)
           if(ish.ge.1)write(ifmt,*)'ghost:',j,idptl(j),ityptl(j)
           if(ish.ge.5)then
              write(ifch,'(a,$)')'ghost:'
              call alistc("&",j,j)
            endif
            ityptl(j)=100+ityptl(j)/10
          elseif(irescl.ge.1)then
c ensure that all particles are really on-shell
            pptl(4,j)=sqrt(pptl(1,j)**2+pptl(2,j)**2
     *                    +pptl(3,j)**2+pptl(5,j)**2)
          endif
        elseif(mod(istptl(j),10).eq.0)then
c if not droplet with fusion
          if(istptl(j).ne.10.or.iorsdf.ne.3)then
            if(ish.ge.1)then
              write(ifmt,*)'Lost particle (E=0)'
              write(ifch,*)'Lost particle (E=0) :'
              call alistc("utghost&",j,j)
            endif
            istptl(j)=istptl(j)+2
          endif
        endif
      enddo

      if(ish.ge.5)write(ifch,'(a)')'---------treat ghosts---------'

c     treat ghosts
c     ------------
      ifirst=1
      scal=1.
      pfif=0.
      efif=0.
      ntry=0
 132  nfif=0
      psum=0
      esum=0.
      ntry=ntry+1
      do  j=nptlpt+1,nptl
        if(mod(istptl(j),10).eq.0)then
          if(ityptl(j).gt.100)then
            nfif=nfif+1
            if(ifirst.eq.1)then
              pfif=pfif+pptl(3,j)
              if(pptl(4,j).gt.0.)efif=efif+pptl(4,j)
            endif
            if(irescl.ge.1) then
              if(ifirst.gt.1)then
                if(pptl(4,j).gt.0.)then
                  Einv=1./pptl(4,j)
                  amt=1.-(pptl(5,j)*Einv)**2+(pptl(1,j)*Einv)**2
     $                +(pptl(2,j)*Einv)**2
                else
                  amt=-1.
                endif
                if(amt.gt.0.)then
                  pptl(3,j)=sign(pptl(4,j),pptl(3,j))*sqrt(amt)
                else
                  y=(rangen()+rangen()+rangen()+rangen()-2.)/2.*yhaha
                  y=sign(abs(y),pptl(3,j))
                  pptl(3,j)
     $                 =sqrt(pptl(5,j)**2+pptl(1,j)**2
     $                 +pptl(2,j)**2)*sinh(y)
                  pptl(4,j)
     $                 =sqrt(pptl(5,j)**2+pptl(1,j)**2
     $                 +pptl(2,j)**2)*cosh(y)
                  efif=efif+pptl(4,j)
                endif
                ifirst=0
              else
c                do k=1,3
                do k=3,3
                  pptl(k,j)=pptl(k,j)*scal
                enddo
                pptl(4,j)=sqrt(pptl(1,j)**2+pptl(2,j)**2+pptl(3,j)**2
     *                 +pptl(5,j)**2)
              endif
            endif
            psum=psum+pptl(3,j)
            esum=esum+pptl(4,j)
            if(ish.ge.5)
     $           write (ifch,*) 'nrevt,psum,esum,pfif,efif,nfif,scal'
     $           ,nrevt,psum,esum,pfif,efif,nfif,scal
          endif
        endif
      enddo
      if ( ish.ge.5 )  write (ifch,*) 'tot',nfif,efif,pfif,esum,psum


      if(nfif.gt.5.or.(esum.gt.0.05*engy.and.nfif.ne.1))then
        if(ifirst.eq.0)then
          do  j=nptlpt+1,nptl
            if ( ityptl(j).ge.101 .and. ityptl(j).le.105 )then
              if((psum-pfif)*(1.-scal).ge.0)
     &             pptl(3,j)=pptl(3,j)-(psum-pfif)/nfif
            endif
          enddo
        else
          ifirst=2
          goto 132
        endif
        scal=efif/esum
        if ( ish.ge.5 )  write (ifch,*) 'scal',scal
        if ( abs(scal-1.) .gt. 0.05 ) then
          if(ntry.le.1000)then
            goto 132
          else
            iret=1
            if(ish.ge.2)write (ifch,*) 'Problem in utghost : redo event'
            if(ish.ge.1)write (ifmt,*) 'Problem in utghost : redo event'
            goto 9999
         endif
        endif
      else
        do  j=nptlpt+1,nptl
          if ( ityptl(j).ge.101 .and. ityptl(j).le.105 )then
            pptl(4,j)=sqrt(pptl(1,j)**2+pptl(2,j)**2+pptl(3,j)**2
     *                 +pptl(5,j)**2)
          endif
        enddo
      endif

      if(ish.ge.5)write(ifch,'(a)')'---------Check Ghost list---------'

c Check Ghost list

      if(ish.ge.5)then
        do  j=nptlpt+1,nptl
          if(mod(istptl(j),10).eq.0)then
            if(ityptl(j).le.105.and.ityptl(j).ge.101)then
              write(ifch,'(a,$)')'ghost:'
              call alistc("&",j,j)
            endif
          endif
        enddo
      endif


 9999 continue
      call ranfst(seedp)        ! ... after this subroutine
      call utprix('ughost',ish,ishini,4)

      end

c-----------------------------------------------------------------------
      subroutine utrsph(iret)
c-----------------------------------------------------------------------
c  if irescl=1 and ispherio=1 rescaling is done for particle used by
c  spherio as initial condition.
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      double precision p1,esoll
      dimension p1(5),p0(5,mamx+mamx)
      call utpri('utrsph',ish,ishini,4)

      errlim=0.0001

      iret=0
      nptlpt=iabs(maproj)+iabs(matarg)
      if(nptl.le.nptlpt) goto 9999

      esoll=0.d0
      p1(1)=0.d0
      p1(2)=0.d0
      p1(3)=0.d0
      p1(4)=0.d0
      do i=nptlpt+1,nptl
        if((istptl(i).le.11
     $   .and.(iorptl(i).ge.1.and.istptl(iorptl(i)).eq.41))
     $   .or.istptl(i).eq.20.or.istptl(i).eq.21)then
         do j=1,2
           p1(j)=p1(j)+dble(pptl(j,i))
         enddo
       endif
      enddo
      do i=1,nptlpt
         do j=1,5
           p0(j,i)=pptl(j,i)
         enddo
         do j=3,4
           p1(j)=p1(j)+dble(pptl(j,i))
         enddo
      enddo
      p1(5)=dsqrt((p1(4)+p1(3))*(p1(4)-p1(3))-p1(2)**2.d0-p1(1)**2.d0)
      esoll=p1(5)
      if(ish.ge.4) write (ifch,'(a,5g13.6)') 'boost-vector',p1

c     trafo
c     -----
      do i=1,nptl
        if((istptl(i).le.11
     $   .and.(iorptl(i).ge.1.and.istptl(iorptl(i)).eq.41))
     $   .or.istptl(i).eq.20.or.istptl(i).eq.21
     $   .or.(istptl(i).eq.0.and.i.le.nptlpt))then
          call utlob4(1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $         ,pptl(1,i),pptl(2,i),pptl(3,i),pptl(4,i))
        endif
      enddo


      if(ish.ge.5)write(ifch,'(a)')'------------------'

c     rescale momenta in rest frame
c     -----------------------------

      scal=1.
      diff=0.
      do ipass=1,1000
        sum=0.
        sum3=0.
        ndif=0
        do  j=1,nptl
        if((istptl(j).le.11
     $   .and.(iorptl(j).ge.1.and.istptl(iorptl(j)).eq.41))
     $   .or.istptl(j).eq.20.or.istptl(j).eq.21
     $   .or.(istptl(j).eq.0.and.j.le.nptlpt))then
            if(j.gt.nptlpt)then
              ndif=ndif+1
              pptl(3,j)=scal*(pptl(3,j)-diff)
              pptl(4,j)=sqrt(pptl(1,j)**2+pptl(2,j)**2+pptl(3,j)**2
     *           +pptl(5,j)**2)
            endif
            sum=sum+pptl(4,j)
            sum3=sum3+pptl(3,j)
          endif
        enddo

        diff=sum3/real(ndif)
        scal=real(esoll)/sum
        if(ish.ge.6)write(ifch,*)'ipass,scal,diff,e,esoll,pz,ndif:'
     $       ,ipass,scal,diff,sum,esoll,sum3,ndif
        if(abs(scal-1.).le.errlim.and.abs(diff).lt.10.*errlim) goto300
      enddo
      if(ish.ge.1)then
        call utmsg('hresph')
        write(ifch,*)'*****  scal=',scal,diff
        call utmsgf
      endif


c     trafo
c     -----
 300  continue
c      do i=nptlpt+1,nptl
      do i=1,nptl
        if((istptl(i).le.11
     $   .and.(iorptl(i).ge.1.and.istptl(iorptl(i)).eq.41))
     $   .or.istptl(i).eq.20.or.istptl(i).eq.21
     $   .or.(istptl(i).eq.0.and.i.le.nptlpt))then
          call utlob4(-1,p1(1),p1(2),p1(3),p1(4),p1(5)
     $         ,pptl(1,i),pptl(2,i),pptl(3,i),pptl(4,i))
        endif
        if(i.le.nptlpt)then
          do j=1,5
            pptl(j,i)=p0(j,i)
          enddo
        endif
      enddo

 9999 call utprix('utrsph',ish,ishini,4)

      end

cc-----------------------------------------------------------------------
c      double precision function dddlog(xxx)
cc-----------------------------------------------------------------------
c      double precision xxx
c      dddlog=-1d50
c      if(xxx.gt.0d0)dddlog=dlog(xxx)
c      end
c
ccc-----------------------------------------------------------------------
c      subroutine randfl(jc,iqa0,iflav,ic,isame)
cc-----------------------------------------------------------------------
cc     returns random flavour ic(2) (iqa0=1:quark,2:antiquark,11:diquark)
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      real probab(nflav),probsu(nflav+1)
c      integer jc(nflav,2),jc0(nflav,2),ic(2)
c      if(ish.ge.6)then
c      write(ifch,*)('-',i=1,10)
c     *,' entry sr randfl ',('-',i=1,30)
c      write(ifch,*)'iqa0:',iqa0
c      write(ifch,*)'jc:'
c      write(ifch,*)jc
c      endif
c      iflav=0
c      ic(1)=0
c      ic(2)=0
c      do 10 n=1,nflav
c      do 10 i=1,2
c10    jc0(n,i)=0
c      iqa1=iqa0*10
c9999  iqa1=iqa1/10
c      if(iqa1.eq.0)goto9998
c      iqa=mod(iqa1,10)
c      su=0
c      do 20 i=1,nflav
c      probab(i)=jc(i,iqa)-jc0(i,iqa)
c      if(isame.eq.1)probab(i)=probab(i)*(jc(i,3-iqa)-jc0(i,3-iqa))
c20    su=su+probab(i)
c      if(su.lt..5)then
c      iflav=0
c      ic(1)=0
c      ic(2)=0
c      goto9998
c      endif
c      probsu(1)=0.
c      do 30 i=1,nflav
c      probsu(i+1)=probsu(i)+probab(i)/su
c      if(probsu(i+1)-probsu(i).lt.1e-5)probsu(i+1)=probsu(i)
c30    continue
c      r=rangen()*probsu(nflav+1)
c      do 50 i=1,nflav
c      if(probsu(i).le.r.and.r.lt.probsu(i+1))iflav=i
c50    continue
c      jc0(iflav,iqa)=jc0(iflav,iqa)+1
c      if(isame.eq.1)jc0(iflav,3-iqa)=jc0(iflav,3-iqa)+1
c      call idenco(jc0,ic,ireten)
c      if(ireten.eq.1)call utstop('randfl: idenco ret code = 1&')
c      if(ish.ge.6)then
c      write(ifch,*)'probab:'
c      write(ifch,*)probab
c      write(ifch,*)'probsu:'
c      write(ifch,*)probsu
c      write(ifch,*)'ran#:',r,'   flav:',iflav
c      endif
c      goto9999
c9998  continue
c      if(ish.ge.6)write(ifch,*)('-',i=1,30)
c     *,' exit sr randfl ',('-',i=1,10)
c      return
c      end
c
c
cc-----------------------------------------------------------------------
c      subroutine ranhvy(x,eps)
cc-----------------------------------------------------------------------
cc     generates x for heavy particle fragmentation according to
cc     the peterson form
cc          d(x)=1/(x*(1-1/x-eps/(1-x))**2)
cc              =d0(x)*d1(x)*d2(x)
cc          d0(x)=(1-x)**2/((1-x)**2+eps)**2
cc          d1(x)=x
cc          d2(x)=(((1-x)**2+eps)/((1-x)**2+eps*x))**2
cc     using x=1-y**pow
cc     generates flat in x if eps>1.
cc-----------------------------------------------------------------------
c      data aln4/1.3863/
c      if(eps.lt.1.) then
c        pow=alog((3.+eps)/eps)/aln4
c        ymx=(eps*(3.*pow-1.)/(pow+1.))**(.5/pow)
c        zmx=1-ymx**pow
c        d0mx=(1-zmx)**2/((1.-zmx)**2+eps)**2*pow*ymx**(pow-1.)
c        d2mx=2./(2.-sqrt(eps))
c      else
c        pow=1.
c        zmx=0.
c        d0mx=(1.-zmx)**2/((1.-zmx)**2+eps)**2
c        d2mx=1.+eps
c      endif
cc
cc          generate z according to (1-z)**2/((1-z)**2+eps*z)**2
c1     continue
c      y=rangen()
c      z=1.-y**pow
cc
c      d0z=(1.-z)**2/((1.-z)**2+eps)**2*pow*y**(pow-1.)
c      if(d0z.lt.rangen()*d0mx) goto1
cc
cc          check remaining factors
c      d1=z
c      d2=(((1.-z)**2+eps)/((1.-z)**2+eps*z))**2
c      if(d1*d2.lt.rangen()*d2mx) goto1
cc
cc          good x
c      x=z
c      return
c      end
c
*-- Author :    D. HECK IK FZK KARLSRUHE       27/04/1994
C=======================================================================

      SUBROUTINE EPOVAPOR( MAPRO,INEW,JFIN,ITYP,PFRX,PFRY,PFRZ )

C-----------------------------------------------------------------------
C  (E)VAPOR(ATION OF NUCLEONS AND ALPHA PARTICLES FROM FRAGMENT)
C
C  TREATES THE REMAINING UNFRAGMENTED NUCLEUS
C  EVAPORATION FOLLOWING CAMPI APPROXIMATION.
C  SEE: X. CAMPI AND J. HUEFNER, PHYS.REV. C24 (1981) 2199
C  AND  J.J. GAIMARD, THESE UNIVERSITE PARIS 7, (1990)
C  THIS SUBROUTINE IS CALLED FROM SDPM, DPMJST, NSTORE, AND VSTORE.
C  ARGUMENTS INPUT:
C   MAPRO        = NUMBER OF NUCLEONS OF PROJECTILE
C   INEW         = PARTICLE TYPE OF SPECTATOR FRAGMENT
C  ARGUMENTS OUTPUT:
C   JFIN         = NUMBER OF FRAGMENTS
C   ITYP(1:JFIN) = NATURE (PARTICLE CODE) OF FRAGMENTS
C   PFRX(1:JFIN) = TRANSVERSE MOMENTUM OF FRAGMENTS IN X-DIRECTION
C   PFRY(1:JFIN) = TRANSVERSE MOMENTUM OF FRAGMENTS IN Y-DIRECTION
C   PFRZ(1:JFIN) = LONGITUDINAL MOMENTUM OF FRAGMENTS IN Z-DIRECTION
C
C  FROM CORSIKA AND ADAPTED BY T. PIEROG TO INCLUDE LONG MOMENTUM AND
C  MORE REALISTIC FRAGMENTS
C-----------------------------------------------------------------------

      IMPLICIT NONE
      include 'epos.inc'
      common/eporansto2/irndmseq
      integer irndmseq

      DOUBLE PRECISION PFR(mamxx),PFRX(mamxx),PFRY(mamxx),PFRZ(mamxx)
     *                ,RD(2*mamxx),SPFRY,SPFRZ,drangen
      DOUBLE PRECISION AFIN,AGLH,APRF,BGLH,EEX,PHIFR,RANNORM,SPFRX
      INTEGER          ITYP(mamxx),IARM,INEW,ITYPRM,INRM,IS,IZRM,JC,lseq
     *                ,JFIN,K,L,LS,MAPRO,MF,NFIN,NINTA,NNUC,NPRF,NNSTEP
      SAVE
      EXTERNAL         RANNORM
C-----------------------------------------------------------------------

      IF(ish.ge.7)WRITE(ifch,*)'EPOVAPOR : MAPRO,INEW=',MAPRO,INEW

      lseq   = irndmseq
      ITYPRM = INEW
      NPRF   = INEW/100
      NINTA  = MAPRO - NPRF
      IF ( NINTA .EQ. 0 ) THEN
C  NO NUCLEON HAS INTERACTED
        JFIN    = 1
        PFRX(1)  = 0.D0
        PFRY(1)  = 0.D0
        PFRZ(1)  = 0.D0
        ITYP(1) = -ITYPRM
        IF(ish.ge.7)WRITE(ifch,*) 'EPOVAPOR : JFIN,NINTA=',JFIN,NINTA
        RETURN
      ENDIF

C  EXCITATION ENERGY EEX OF PREFRAGMENT
C  SEE: J.J. GAIMARD, THESE UNIVERSITE PARIS 7, (1990), CHPT. 4.2
      EEX = 0.D0
      CALL RMMARD( RD,2*NINTA,lseq )
      DO  L = 1, NINTA
        IF ( RD(NINTA+L) .LT. RD(L) ) RD(L) = 1.D0 - RD(L)
        EEX = EEX + RD(L)
      ENDDO
C  DEPTH OF WOODS-SAXON POTENTIAL TO FERMI SURFACE IS 0.040 GEV
      IF(ish.ge.7)WRITE(ifch,*)'EPOVAPOR : EEX=',SNGL(EEX*0.04D0),
     &                                            ' GEV'
C  EVAPORATION: EACH EVAPORATION STEP NEEDS ABOUT 0.020 GEV, THEREFORE
C  NNSTEP IS EEX * 0.04/0.02 = EEX * 2.
      NNSTEP = INT( EEX*2.D0 )

      IF ( NNSTEP .LE. 0 ) THEN
C  EXCITATION ENERGY TOO SMALL, NO EVAPORATION
        JFIN    = 1
        PFRX(1)  = 0.D0
        PFRY(1)  = 0.D0
        PFRZ(1)  = 0.D0
        ITYP(1) = -ITYPRM
        IF(ish.ge.7)WRITE(ifch,*) 'EPOVAPOR : JFIN,EEX=',JFIN,SNGL(EEX)
        RETURN
      ENDIF

C  AFIN IS ATOMIC NUMBER OF FINAL NUCLEUS
      APRF = DBLE(NPRF)
      AFIN = APRF - 1.6D0 * DBLE(NNSTEP)
      NFIN = MAX( 0, INT( AFIN+0.5D0 ) )
C  CORRESPONDS TO DEFINITION; FRAGMENTATION-EVAPORATION
C  CONVOLUTION EMU07 /MODEL ABRASION EVAPORATION (JNC FZK APRIL 94)
C  NNUC IS NUMBER OF EVAPORATING NUCLEONS
      NNUC = NPRF - NFIN
      IF(ish.ge.7)WRITE(ifch,*) 'EPOVAPOR : NFIN,NNUC=',NFIN,NNUC
      JC   = 0

      IF     ( NNUC .LE. 0 ) THEN
C  NO EVAPORATION
        JFIN    = 1
        PFRX(1)  = 0.D0
        PFRY(1)  = 0.D0
        PFRZ(1)  = 0.D0
        ITYP(1) = -ITYPRM
        RETURN

      ELSEIF ( NNUC .GE. 4 ) THEN
C  EVAPORATION WITH FORMATION OF ALPHA PARTICLES POSSIBLE
C  IARM, IZRM, INRM ARE NUMBER OF NUCLEONS, PROTONS, NEUTRONS OF
C  REMAINDER
        DO  LS = 1, NNSTEP
          IARM = ITYPRM/100
          IF ( IARM .LE. 0 ) GOTO 100
          IZRM = MOD(ITYPRM,100)
          INRM = IARM - IZRM
          JC   = JC + 1
          CALL RMMARD( RD,2,lseq )
          IF ( RD(1) .LT. 0.2D0  .AND.  IZRM .GE. 2
     *                           .AND.  INRM .GE. 2 ) THEN
            ITYP(JC) = -402          !alpha
            NNUC     = NNUC - 4
            ITYPRM   = ITYPRM - 402
          ELSE
            IF ( IZRM .EQ. 1 .AND. INRM .GT. IZRM ) THEN
              ITYP(JC) = 1220
              ITYPRM   = ITYPRM - 100              
            ELSEIF ( INRM .EQ. 1 .AND. IZRM .GT. INRM ) THEN
              ITYP(JC) = 1120
              ITYPRM   = ITYPRM - 101              
            ELSEIF(RD(2)*(IZRM+INRM).LT.IZRM.AND.IZRM.GE.INRM)THEN
              ITYP(JC) = 1120
              ITYPRM   = ITYPRM - 101
            ELSE
              ITYP(JC) = 1220
              ITYPRM   = ITYPRM - 100
            ENDIF
            NNUC = NNUC - 1
          ENDIF
          IF ( NNUC .LE. 0 ) GOTO 50
        ENDDO
      ENDIF

      IF ( NNUC .LT. 4 ) THEN
C  EVAPORATION WITHOUT FORMATION OF ALPHA PARTICLES
        CALL RMMARD( RD,NNUC,lseq )
        DO  IS = 1, NNUC
          IARM = ITYPRM/100
          IF ( IARM .LE. 0 ) GOTO 100
          IZRM = MOD(ITYPRM,100)
          INRM = IARM - IZRM
          JC   = JC + 1
          IF ( IZRM .EQ. 1 .AND. INRM .GT. IZRM ) THEN
            ITYP(JC) = 1220
            ITYPRM   = ITYPRM - 100              
          ELSEIF ( INRM .EQ. 1 .AND. IZRM .GT. IZRM ) THEN
            ITYP(JC) = 1120
            ITYPRM   = ITYPRM - 101              
          ELSEIF ( RD(IS)*IARM .LT. IZRM .AND. IZRM .GE. INRM ) THEN
            ITYP(JC) = 1120
            ITYPRM   = ITYPRM - 101
          ELSE
            ITYP(JC) = 1220
            ITYPRM   = ITYPRM - 100
          ENDIF
        ENDDO
      ENDIF

 50   CONTINUE
      IARM = ITYPRM/100
      IF ( IARM .LE. 0 ) GOTO 100
      IZRM = MOD(ITYPRM,100)
      INRM = IARM - IZRM
      JC = JC + 1
      IF     ( IARM .EQ. 5 ) THEN     !EXCLUDED
        IF ( IZRM .GE. INRM ) THEN
          ITYP(JC) = 1120
          ITYPRM   = ITYPRM - 101              
        ELSE
          ITYP(JC) = 1220
          ITYPRM   = ITYPRM - 100
        ENDIF
        JC = JC + 1
        ITYP(JC) = -ITYPRM
      ELSEIF     ( ITYPRM .GT. 200 ) THEN
        ITYP(JC) = -ITYPRM
      ELSEIF ( ITYPRM .EQ. 200 ) THEN
        ITYP(JC) = 1220
        JC = JC + 1
        ITYP(JC) = 1220
      ELSEIF ( ITYPRM .EQ. 101 ) THEN
        ITYP(JC) = 1120
      ELSEIF ( ITYPRM .EQ. 100 ) THEN
        ITYP(JC) = 1220
      ELSE
        JC = JC - 1
        IF ( ITYPRM .NE. 0 ) WRITE(*,*)
     *                  'EPOVAPOR : ILLEGAL PARTICLE ITYPRM =',ITYPRM
      ENDIF

  100 CONTINUE
      JFIN = JC
      IF(ish.ge.7)WRITE(ifch,*) 
     *   'EPOVAPOR :  NO        ITYP     PFR       PFL'
      IF     ( infragm .EQ. 2 ) THEN
C  EVAPORATION WITH PT AFTER PARAMETRIZED JACEE DATA
        DO  MF = 1, JFIN
          IF(ITYP(MF).LT.0)THEN
            IARM=-ITYP(MF)/100
          ELSE
            IARM=1
          ENDIF
          PFR(MF) = RANNORM(0.088D0,0.044D0)
          PFRZ(MF)= (2*int(drangen(PFR(MF))+0.5d0)-1)
     &   *RANNORM(0.300D0/DBLE(IARM),0.100D0/SQRT(DBLE(IARM)))    !Fermi motion about 300 MeV
          IF(ish.ge.7)WRITE(ifch,*) MF,ITYP(MF),SNGL(PFR(MF))
     &                                         ,SNGL(PFRZ(MF))
        ENDDO
      ELSEIF ( infragm .EQ. 3 ) THEN
C  EVAPORATION WITH PT AFTER GOLDHABER''S MODEL (PHYS.LETT.53B(1974)306)
        DO  MF = 1, JFIN
          K    = MAX( 1, -ITYP(MF)/100 )
          BGLH = K * (MAPRO - K) / DBLE(MAPRO-1)
C  THE VALUE 0.103 [GEV] IS SIGMA(0)=P(FERMI)/SQRT(5.)
*         AGLH = 0.103D0 * SQRT( BGLH )
C  THE VALUE 0.090 [GEV] IS EXPERIMENTALLY DETERMINED SIGMA(0)
          AGLH = 0.090D0 * SQRT( BGLH )
          PFR(MF) = RANNORM(0.D0,AGLH)
          PFRZ(MF)= RANNORM(0.000D0,0.500D0)    !from pAg at 100 GeV
          IF(ish.ge.7)WRITE(ifch,*) MF,ITYP(MF),SNGL(PFR(MF))
     &                                         ,SNGL(PFRZ(MF))
        ENDDO
      ELSE
C  EVAPORATION WITHOUT TRANSVERSE MOMENTUM
        DO  MF = 1, JFIN
          PFR(MF) = 0.D0
          PFRZ(MF)= 0.D0
          IF(ish.ge.7)WRITE(ifch,*) MF,ITYP(MF),SNGL(PFR(MF))
     &                                         ,SNGL(PFRZ(MF))
        ENDDO
      ENDIF
C  CALCULATE RESIDUAL TRANSVERSE MOMENTUM
      SPFRX = 0.D0
      SPFRY = 0.D0
      SPFRZ = 0.D0
      CALL RMMARD( RD,JFIN,lseq )
      DO  MF = 1, JFIN
        PHIFR = PI * RD(MF)
        PFRX(MF) = PFR(MF) * COS( PHIFR )
        PFRY(MF) = PFR(MF) * SIN( PHIFR )
        SPFRY = SPFRY + PFRY(MF)
        SPFRX = SPFRX + PFRX(MF)
        SPFRZ = SPFRZ + PFRZ(MF)
      ENDDO
C  CORRECT ALL TRANSVERSE MOMENTA FOR MOMENTUM CONSERVATION
      SPFRX = SPFRX / JFIN
      SPFRY = SPFRY / JFIN
      SPFRZ = SPFRZ / JFIN
      DO  MF = 1, JFIN
        PFRX(MF) = PFRX(MF) - SPFRX
        PFRY(MF) = PFRY(MF) - SPFRY
        PFRZ(MF) = PFRZ(MF) - SPFRZ
      ENDDO

      IF(ish.ge.7)WRITE(ifch,*) 'EPOVAPOR : NINTA,JFIN=',NINTA,JFIN

      RETURN
      END

*-- Author :    The CORSIKA development group   21/04/1994
C=======================================================================

      DOUBLE PRECISION FUNCTION RANNORM( A,B )

C-----------------------------------------------------------------------
C  RAN(DOM NUMBER) NOR(MALLY DISTRIBUTED)
C
C  GENERATES NORMAL DISTRIBUTED RANDOM NUMBER
C  DELIVERS 2 UNCORRELATED RANDOM NUMBERS,
C  THEREFORE RANDOM CALLS ARE ONLY NECESSARY EVERY SECOND TIME.
C  REFERENCE : NUMERICAL RECIPES, W.H. PRESS ET AL.,
C              CAMBRIDGE UNIVERSITY PRESS, 1992  ISBN 0 521 43064 X
C  ARGUMENTS:
C   A      = MEAN VALUE
C   B      = STANDARD DEVIATION
C
C  FROM CORSIKA
C-----------------------------------------------------------------------

      IMPLICIT NONE
      double precision facrdm,u1rdm,u2rdm,drangen
      logical knordm
      data knordm/.true./

      DOUBLE PRECISION A,B,RR
      SAVE facrdm,u1rdm,u2rdm,knordm
C-----------------------------------------------------------------------

      IF ( KNORdm ) THEN
  1     CONTINUE
        U1rdm = 2.D0*drangen(a) - 1.D0
        U2rdm = 2.D0*drangen(b) - 1.D0
        RR = U1rdm**2 + U2rdm**2
        IF ( RR .GE. 1.D0  .OR.  RR .EQ. 0.D0 ) GOTO 1
        FACrdm = SQRT( (-2.D0) * LOG(RR) / RR )

        RANNORM = FACrdm * U1rdm * B + A
        KNORdm   = .FALSE.
      ELSE
        RANNORM = FACrdm * U2rdm * B + A
        KNORdm   = .TRUE.
      ENDIF

      RETURN
      END

c-----------------------------------------------------------------------
      function ransig()
c-----------------------------------------------------------------------
c     returns randomly +1 or -1
c-----------------------------------------------------------------------
      ransig=1
      if(rangen().gt.0.5)ransig=-1
      return
      end

cc-----------------------------------------------------------------------
c      function ranxq(n,x,q,xmin)
cc-----------------------------------------------------------------------
cc     returns random number according to x(i) q(i) with x>=xmin
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      real x(n),q(n)
c      imin=1
c      if(xmin.eq.0.)goto3
c      i1=1
c      i2=n
c1     i=i1+(i2-i1)/2
c      if(x(i).lt.xmin)then
c      i1=i
c      elseif(x(i).gt.xmin)then
c      i2=i
c      else
c      imin=i
c      goto3
c      endif
c      if(i2-i1.gt.1)goto1
c      imin=i2
c3     continue
c      if(q(imin).gt.q(n)*.9999)then
c      ranxq=xmin
c      goto4
c      endif
c      qran=q(imin)+rangen()*(q(n)-q(imin))
c      ranxq=utinvt(n,x,q,qran)
c4     continue
c
c      if(ranxq.lt.xmin)then
c      call utmsg('ranxq ')
c      write(ifch,*)'*****  ranxq=',ranxq,' <       xmin=',xmin
c      write(ifch,*)'q(imin) q q(n):',q(imin),qran,q(n)
c      write(ifch,*)'x(imin) x x(n):',x(imin),ranxq,x(n)
c      call utmsgf
c      ranxq=xmin
c      endif
c
c      return
c      end
c
cc  ***** end r-routines
cc  ***** beg s-routines
c
cc-----------------------------------------------------------------------
c      function sbet(z,w)
cc-----------------------------------------------------------------------
c      sbet=utgam1(z)*utgam1(w)/utgam1(z+w)
c      return
c      end
c
cc-----------------------------------------------------------------------
c      function smass(a,y,z)
cc-----------------------------------------------------------------------
cc     returns droplet mass (in gev) (per droplet, not (!) per nucleon)
cc     according to berger/jaffe mass formula, prc35(1987)213 eq.2.31,
cc     see also c. dover, BNL-46322, intersections-meeting, tucson, 91.
cc     a: massnr, y: hypercharge, z: charge,
cc-----------------------------------------------------------------------
c      common/cmass/thet,epsi,as,ac,dy,dz,ym,cz,zm,sigma,rzero
c      ymin=ym*a
c      zmin=cz/(dz/a+zm/a**(1./3.))
c      smass=epsi*a+as*a**(2./3.)+(ac/a**(1./3.)+dz/a/2.)*(z-zmin)**2
c     *+dy/a/2.*(y-ymin)**2
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine smassi(theta)
cc-----------------------------------------------------------------------
cc     initialization for smass.
cc     calculates parameters for berger/jaffe mass formula
cc     (prc35(1987)213 eq.2.31, see also c. dover, BNL-46322).
cc     theta: parameter that determines all parameters in mass formula.
cc-----------------------------------------------------------------------
c      common/cmass/thet,epsi,as,ac,dy,dz,ym,cz,zm,sigma,rzero
c      thet=theta
c
c      astr=.150
c      pi=3.14159
c      alp=1./137.
c
c      co=cos(theta)
c      si=sin(theta)
c      bet=(1+co**3)/2.
c      rzero=si/astr/(  2./3./pi*(1+co**3)  )**(1./3.)
cctp060829      cs=astr/si
c      cz=-astr/si*(  (  .5*(1+co**3)  )**(1./3.)-1  )
c      sigma=6./8./pi*(astr/si)**3*(co**2/6.-si**2*(1-si)/3.-
c     *1./3./pi*(pi/2.-theta-sin(2*theta)+si**3*alog((1+co)/si)))
c
c      epsi=astr*((.5*(1+co**3))**(1./3.)+2)/si
c      as=4*pi*sigma*rzero**2
c      ac=3./5.*alp/rzero
c      dz=astr/si*bet**(1./3.)*co**2*
c     *(co**4*(1+bet**(2./3.))+(1+bet)**2)/
c     *(  (2*co**2+bet**(1./3.))*(co**4*(1+bet**(2./3.))+(1+bet)**2)-
c     *(co**4+bet**(1./3.)*(1+bet))*((2*bet**(2./3.)-1)*co**2+1+bet)  )
c      dy=astr/6.*(1+co**3)**3/si*
c     *(  1+(1+co)/(4*(1+co**3))**(2./3.)  )/
c     *(co**6+co+co*(.5*(1+co**3))**(4./3.))
c      zm=6*alp/(5*rzero)
c      ym=(1-co**3)/(1+co**3)
c
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine smassp
cc-----------------------------------------------------------------------
cc     prints smass.
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      common/cmass/thet,epsi,as,ac,dy,dz,ym,cz,zm,sigma,rzero
c      real eng(14),ymi(14),zmi(14)
c      pi=3.14159
c      write(ifch,*)'parameters of mass formula:'
c      write(ifch,*)'theta=',thet,'   epsi=',epsi
c      write(ifch,*)'as=',as,'   ac=',ac
c      write(ifch,*)'dy=',dy,'   dz=',dz
c      write(ifch,*)'ym=',ym
c      write(ifch,*)'cz dz zm=',cz,dz,zm
c      write(ifch,*)'sigma**1/3=',sigma**(1./3.),'   rzero=',rzero
c      write(ifch,*)'mass:'
c      write(ifch,5000)(j,j=1,14)
c5000  format(5x,'a:',14i5)
c      do 4 j=1,14
c      a=j
c      ymi(j)=ym*a
c4     zmi(j)=cz/(dz/a+zm/a**(1./3.))
c      write(ifch,5002)(ymi(j),j=1,14)
c5002  format(1x,'ymin: ',14f5.2)
c      write(ifch,5003)(zmi(j),j=1,14)
c5003  format(1x,'zmin: ',14f5.2)
c      do 2 i=1,15
c      ns=11-i
c      do 3 j=1,14
c      a=j
c      y=a-ns
c      z=0.
c3     eng(j)=smass(a,y,z)/a
c      write(ifch,5001)ns,(eng(j),j=1,14)
c5001  format(1x,'s=',i2,2x,14f5.2)
c2     continue
c      write(ifch,*)'mass-mass(free):'
c      write(ifch,5000)(j,j=1,14)
c      do 5 i=1,15
c      ns=11-i
c      do 6 j=1,14
c      a=j
c      y=a-ns
c      z=0.
c      call smassu(a,y,z,ku,kd,ks,kc)
c6     eng(j)=(smass(a,y,z)-utamnu(ku,kd,ks,kc,0,0,3))/a
c      write(ifch,5001)ns,(eng(j),j=1,14)
c5     continue
c
c      stop
c      end
c
cc-----------------------------------------------------------------------
c      subroutine smasst(kux,kdx,ksx,kcx,a,y,z)
cc-----------------------------------------------------------------------
cc     input: kux,kdx,ksx,kcx = net quark numbers (for u,d,s,c quarks).
cc     output: massnr a, hypercharge y and charge z.
cc-----------------------------------------------------------------------
c      sg=1
c      if(kux+kdx+ksx+kcx.lt.0.)sg=-1
c      ku=sg*kux
c      kd=sg*kdx
c      ks=sg*ksx
c      kc=sg*kcx
c      k=ku+kd+ks+kc
c      if(mod(k,3).ne.0)stop'noninteger baryon number'
c      a=k/3
c      y=a-ks
c      nz=2*ku-kd-ks+2*kc
c      if(mod(nz,3).ne.0)stop'noninteger charge'
c      z=nz/3
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine smassu(ax,yx,zx,ku,kd,ks,kc)
cc-----------------------------------------------------------------------
cc     input: massnr ax, hypercharge yx and charge zx.
cc     output: ku,kd,ks,kc = net quark numbers (for u,d,s,c quarks).
cc-----------------------------------------------------------------------
c      sg=1
c      if(ax.lt.0.)sg=-1
c      a=sg*ax
c      y=sg*yx
c      z=sg*zx
c      ku=nint(a+z)
c      kd=nint(a-z+y)
c      ks=nint(a-y)
c      kc=0
c      return
c      end
c
cc-----------------------------------------------------------------------
c      function spoc(a,b,c,d,x)
cc-----------------------------------------------------------------------
cc     power fctn with cutoff
cc-----------------------------------------------------------------------
c      spoc=0
c      if(a.eq.0..and.b.eq.0.)return
c      spoc =a+b*x**c
c      spoc0=a+b*d**c
c      spoc=amin1(spoc,spoc0)
c      spoc=amax1(0.,spoc)
c      return
c      end
c
c-----------------------------------------------------------------------
      function utacos(x)
c-----------------------------------------------------------------------
c     returns acos(x) for -1 <= x <= 1 , acos(+-1) else
c-----------------------------------------------------------------------
      include 'epos.inc'
      argum=x
      if(x.lt.-1.)then
      if(ish.ge.1)then
      call utmsg('utacos')
      write(ifch,*)'*****  argum = ',argum,' set -1'
      call utmsgf
      endif
      argum=-1.
      elseif(x.gt.1.)then
      if(ish.ge.1)then
      call utmsg('utacos')
      write(ifch,*)'*****  argum = ',argum,' set 1'
      call utmsgf
      endif
      argum=1.
      endif
      utacos=acos(argum)
      return
      end

c----------------------------------------------------------------------
      function utamnu(keux,kedx,kesx,kecx,kebx,ketx,modus)
c----------------------------------------------------------------------
c     returns min mass of droplet with given u,d,s,c content
c     keux: net u quark number
c     kedx: net d quark number
c     kesx: net s quark number
c     kecx: net c quark number
c     kebx: net b quark number
c     ketx: net t quark number
c     modus: 4=two lowest multiplets; 5=lowest multiplet
c----------------------------------------------------------------------
      common/files/ifop,ifmt,ifch,ifcx,ifhi,ifdt,ifcp,ifdr
      common/csjcga/amnull,asuha(7)
      common/drop4/asuhax(7),asuhay(7)

      if(modus.lt.4.or.modus.gt.5)stop'UTAMNU: not supported'
c 1    format(' flavours:',6i5 )
c 100  format(' flavours+mass:',6i5,f8.2 )
c      write(ifch,1)keux,kedx,kesx,kecx,kebx,ketx

      amnull=0.

      do i=1,7
      if(modus.eq.4)asuha(i)=asuhax(i)    !two lowest multiplets
      if(modus.eq.5)asuha(i)=asuhay(i)    !lowest multiplet
      enddo

      ke=iabs(keux+kedx+kesx+kecx+kebx+ketx)

      if(keux+kedx+kesx+kecx+kebx+ketx.ge.0)then
      keu=keux
      ked=kedx
      kes=kesx
      kec=kecx
      keb=kebx
      ket=ketx
      else
      keu=-keux
      ked=-kedx
      kes=-kesx
      kec=-kecx
      keb=-kebx
      ket=-ketx
      endif

c      write(ifch,*)keu,ked,kes,kec,keb,ket

c   removing top mesons  to remove t quarks or antiquarks
      if(ket.ne.0)then
12    continue
      ii=sign(1,ket)
      ket=ket-ii
      if(ii*keu.le.ii*ked)then
      keu=keu+ii
      else
      ked=ked+ii
      endif
      amnull=amnull+200.    ! ???????
      if(ket.ne.0)goto12
      endif

c   removing bottom mesons  to remove b quarks or antiquarks
      if(keb.ne.0)then
11    continue
      ii=sign(1,keb)
      keb=keb-ii
      if(ii*keu.le.ii*ked)then
      keu=keu+ii
      else
      ked=ked+ii
      endif
      amnull=amnull+6. !5.28   ! (more than B-meson)
      if(keb.ne.0)goto11
      endif

c   removing charm mesons  to remove c quarks or antiquarks
      if(kec.ne.0)then
10    continue
      ii=sign(1,kec)
      kec=kec-ii
      if(keu*ii.le.ked*ii)then
      keu=keu+ii
      else
      ked=ked+ii
      endif
      amnull=amnull+2.2 !1.87  ! (more than D-meson)
      if(kec.ne.0)goto10
      endif

c      write(ifch,100)keu,ked,kes,kec,keb,ket,amnull

c   removing mesons to remove s antiquarks
5     continue
      if(kes.lt.0)then
      amnull=amnull+asuha(6)
      if(keu.ge.ked)then
      keu=keu-1
      else
      ked=ked-1
      endif
      kes=kes+1
      goto5
      endif

c   removing mesons to remove d antiquarks
6     continue
      if(ked.lt.0)then
      if(keu.ge.kes)then
      amnull=amnull+asuha(5)
      keu=keu-1
      else
      amnull=amnull+asuha(6)
      kes=kes-1
      endif
      ked=ked+1
      goto6
      endif

c   removing mesons to remove u antiquarks
7     continue
      if(keu.lt.0)then
      if(ked.ge.kes)then
      amnull=amnull+asuha(5)
      ked=ked-1
      else
      amnull=amnull+asuha(6)
      kes=kes-1
      endif
      keu=keu+1
      goto7
      endif

c      write(ifch,100)keu,ked,kes,kec,keb,ket,amnull
c      print*,keu,ked,kes,kec,keb,ket,amnull

      if(keu+ked+kes+kec+keb+ket.ne.ke)
     *call utstop('utamnu: sum_kei /= ke&')
      keq=keu+ked
      keqx=keq
      amnux=0

c   removing strange baryons
      i=4
2     i=i-1
3     continue
      if((4-i)*kes.gt.(i-1)*keq)then
      amnux=amnux+asuha(1+i)
      kes=kes-i
      keq=keq-3+i
      if(kes.lt.0)call utstop('utamnu: negative kes&')
      if(keq.lt.0)call utstop('utamnu: negative keq&')
      goto3
      endif
      if(i.gt.1)goto2
      if(keqx.gt.keq)then
      do 8 k=1,keqx-keq
      if(keu.ge.ked)then
      keu=keu-1
      else
      ked=ked-1
      endif
8     continue
      endif

      if(keu+ked.ne.keq)call utstop('utamnu: keu+ked /= keq&')
c      write(ifch,100)keu,ked,kes,kec,keb,ket,amnull+amnux
c      print*,keu,ked,kes,kec,keb,ket,amnull+amnux

c   removing nonstrange baryons
9     continue
      if(keu.gt.2*ked)then
      amnux=amnux+asuha(7)
      keu=keu-3
      if(keu.lt.0)call utstop('utamnu: negative keu&')
      goto9
      endif
      if(ked.gt.2*keu)then
      amnux=amnux+asuha(7)
      ked=ked-3
      if(ked.lt.0)call utstop('utamnu: negative ked&')
      goto9
      endif
      keq=keu+ked

c      write(ifch,100)keu,ked,kes,kec,keb,ket,amnull+amnux
c      print*,keu,ked,kes,kec,keb,ket,amnull+amnux

      if(mod(keq,3).ne.0)call utstop('utamnu: mod(keq,3) /= 0&')
      amnux=amnux+asuha(1)*keq/3

c      write(ifch,100)keu,ked,kes,kec,keb,ket,amnull+amnux
c      print*,keu,ked,kes,kec,keb,ket,amnull+amnux

      amnull=amnull+amnux

      if(amnull.eq.0)amnull=asuha(5)

      utamnu=amnull
      return
      end

c-----------------------------------------------------------------------
      function utamnx(jcp,jcm)
c-----------------------------------------------------------------------
c returns minimum mass for the decay of jcp---jcm (by calling utamnu).
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jcp(nflav,2),jcm(nflav,2)

      do i=1,nflav
      do j=1,2
      if(jcp(i,j).ne.0)goto1
      enddo
      enddo
      keu=jcm(1,1)-jcm(1,2)
      ked=jcm(2,1)-jcm(2,2)
      kes=jcm(3,1)-jcm(3,2)
      kec=jcm(4,1)-jcm(4,2)
      keb=jcm(5,1)-jcm(5,2)
      ket=jcm(6,1)-jcm(6,2)
      utamnx=utamnu(keu,ked,kes,kec,keb,ket,5)
      return
1     continue

      do i=1,nflav
      do j=1,2
      if(jcm(i,j).ne.0)goto2
      enddo
      enddo
      keu=jcp(1,1)-jcp(1,2)
      ked=jcp(2,1)-jcp(2,2)
      kes=jcp(3,1)-jcp(3,2)
      kec=jcp(4,1)-jcp(4,2)
      keb=jcp(5,1)-jcp(5,2)
      ket=jcp(6,1)-jcp(6,2)
      utamnx=utamnu(keu,ked,kes,kec,keb,ket,5)
      return
2     continue

      keu=jcp(1,1)-jcp(1,2)
      ked=jcp(2,1)-jcp(2,2)
      kes=jcp(3,1)-jcp(3,2)
      kec=jcp(4,1)-jcp(4,2)
      keb=jcp(5,1)-jcp(5,2)
      ket=jcp(6,1)-jcp(6,2)
      ke=keu+ked+kes+kec+keb+ket
      if(mod(ke+1,3).eq.0)then
        keu=keu+1
        amms1=utamnu(keu,ked,kes,kec,keb,ket,5)
        keu=keu-1
        ked=ked+1
        amms2=utamnu(keu,ked,kes,kec,keb,ket,5)
      elseif(mod(ke-1,3).eq.0)then
        keu=keu-1
        amms1=utamnu(keu,ked,kes,kec,keb,ket,5)
        keu=keu+1
        ked=ked-1
        amms2=utamnu(keu,ked,kes,kec,keb,ket,5)
      else
        amms1=0
        amms2=0
        amms3=0
        amms4=0
        call utstop('utamnx: no singlet possible (1)&')
      endif
      keu=jcm(1,1)-jcm(1,2)
      ked=jcm(2,1)-jcm(2,2)
      kes=jcm(3,1)-jcm(3,2)
      kec=jcm(4,1)-jcm(4,2)
      keb=jcm(5,1)-jcm(5,2)
      ket=jcm(6,1)-jcm(6,2)
      ke=keu+ked+kes+kec+keb+ket
      if(mod(ke+1,3).eq.0)then
        keu=keu+1
        amms3=utamnu(keu,ked,kes,kec,keb,ket,5)
        keu=keu-1
        ked=ked+1
        amms4=utamnu(keu,ked,kes,kec,keb,ket,5)
      elseif(mod(ke-1,3).eq.0)then
        keu=keu-1
        amms3=utamnu(keu,ked,kes,kec,keb,ket,5)
        keu=keu+1
        ked=ked-1
        amms4=utamnu(keu,ked,kes,kec,keb,ket,5)
      else
        call utstop('utamnx: no singlet possible (2)&')
      endif
      utamnx=min(amms1+amms3,amms2+amms4)
c       print *,amms1,amms3,amms2,amms4,jcp,jcm
      return
      end



cc-----------------------------------------------------------------------
c      function utamny(jcp,jcm)
cc-----------------------------------------------------------------------
cc returns minimum mass of jcp+jcm (by calling utamnu).
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      integer jcp(nflav,2),jcm(nflav,2),jc(nflav,2)
c      do 7 nf=1,nflav
c      jc(nf,1)=jcp(nf,1)+jcm(nf,1)
c7     jc(nf,2)=jcp(nf,2)+jcm(nf,2)
c      keu=jc(1,1)-jc(1,2)
c      ked=jc(2,1)-jc(2,2)
c      kes=jc(3,1)-jc(3,2)
c      kec=jc(4,1)-jc(4,2)
c      keb=jc(5,1)-jc(5,2)
c      ket=jc(6,1)-jc(6,2)
c      utamny=utamnu(keu,ked,kes,kec,keb,ket,5)
c      return
c      end
c
c-----------------------------------------------------------------------
      function utamnz(jc,modus)
c-----------------------------------------------------------------------
c returns minimum mass of jc (by calling utamnu).
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2)
      keu=jc(1,1)-jc(1,2)
      ked=jc(2,1)-jc(2,2)
      kes=jc(3,1)-jc(3,2)
      kec=jc(4,1)-jc(4,2)
      keb=jc(5,1)-jc(5,2)
      ket=jc(6,1)-jc(6,2)
      utamnz=utamnu(keu,ked,kes,kec,keb,ket,modus)
      return
      end

c-----------------------------------------------------------------------
      subroutine utar(i1,i2,i3,x0,x1,x2,x3,xx)
c-----------------------------------------------------------------------
c     returns the array xx with xx(1)=x0 <= xx(i) <= xx(i3)=x3
c-----------------------------------------------------------------------
      real xx(i3)
      do 1 i=1,i1-1
  1   xx(i)=x0+(i-1.)/(i1-1.)*(x1-x0)
      do 2 i=i1,i2-1
  2   xx(i)=x1+(i-i1*1.)/(i2-i1*1.)*(x2-x1)
      do 3 i=i2,i3
  3   xx(i)=x2+(i-i2*1.)/(i3-i2*1.)*(x3-x2)
      return
      end

cc---------------------------------------------------------------------
c      subroutine utaxis(i,j,a1,a2,a3)
cc-----------------------------------------------------------------------
cc     calculates the axis defined by the ptls i,j in the i,j cm system
cc---------------------------------------------------------------------
c      include 'epos.inc'
c      double precision pi1,pi2,pi3,pi4,pj1,pj2,pj3,pj4,p1,p2,p3,p4,p5
c     *,err,a
c      a1=0
c      a2=0
c      a3=1
c      pi1=dble(pptl(1,i))
c      pi2=dble(pptl(2,i))
c      pi3=dble(pptl(3,i))
c      pi4=dble(pptl(4,i))
c      pj1=dble(pptl(1,j))
c      pj2=dble(pptl(2,j))
c      pj3=dble(pptl(3,j))
c      pj4=dble(pptl(4,j))
c      p1=pi1+pj1
c      p2=pi2+pj2
c      p3=pi3+pj3
c      p4=pi4+pj4
c      p5=dsqrt(p4**2-p3**2-p2**2-p1**2)
c      call utlob2(1,p1,p2,p3,p4,p5,pi1,pi2,pi3,pi4,50)
c      call utlob2(1,p1,p2,p3,p4,p5,pj1,pj2,pj3,pj4,51)
c           err=(pi1+pj1)**2+(pi2+pj2)**2+(pi3+pj3)**2
c           if(err.gt.1d-3)then
c      call utmsg('utaxis')
c      write(ifch,*)'*****  err=',err
c      write(ifch,*)'pi:',pi1,pi2,pi3,pi4
c      write(ifch,*)'pj:',pj1,pj2,pj3,pj4
c      call utmsgf
c           endif
c      a=dsqrt( (pj1-pi1)**2 + (pj2-pi2)**2 + (pj3-pi3)**2 )
c      if(a.eq.0.d0)return
c      a1=sngl((pi1-pj1)/a)
c      a2=sngl((pi2-pj2)/a)
c      a3=sngl((pi3-pj3)/a)
c      return
c      end
c
cc---------------------------------------------------------------------
c      subroutine uthalf(i,j,zz1,zz2,iret)
cc-----------------------------------------------------------------------
cc     give equal energy (E_i+E_j)/2 to particle i+j in their cm system
cc---------------------------------------------------------------------
c      include 'epos.inc'
c      double precision pi1,pi2,pi3,pi4,pi5,pj1,pj2,pj3,pj4,pj5
c     *,p1,p2,p3,p4,p5,err,pt,pti,sinp,cosp,pmax,phi,drangen!,rrr
c      iret=0
c      pi1=dble(pptl(1,i))
c      pi2=dble(pptl(2,i))
c      pi3=dble(pptl(3,i))
c      pi5=dble(pptl(5,i))
c      pi4=sqrt(pi1**2+pi2**2+pi3**2+pi5**2)
c      pj1=dble(pptl(1,j))
c      pj2=dble(pptl(2,j))
c      pj3=dble(pptl(3,j))
c      pj5=dble(pptl(5,j))
c      pj4=sqrt(pj1**2+pj2**2+pj3**2+pj5**2)
c      if(ish.ge.6)then
c        write(ifch,*)'uthalf for ',i,' and ',j
c        write(ifch,*)'in ',idptl(i),pi1,pi2,pi3,pi4,pi5
c        write(ifch,*)'<> ',idptl(j),pj1,pj2,pj3,pj4,pj5
c      endif
c      p1=pi1+pj1
c      p2=pi2+pj2
c      p3=pi3+pj3
c      p4=pi4+pj4
c      p5=(p4-p3)*(p4+p3)-p2**2-p1**2
c      if(p5.lt.0d0.or.(pi3.lt.0.99d0*pj3.and.mod(ityptl(j)/10,10).eq.4)
c     &       .or.(pi3.gt.0.99d0*pj3.and.mod(ityptl(j)/10,10).eq.5))then
cc      if(p5.lt.0d0)then
c        if(ish.ge.7)write(ifch,*)'Inversion not possible (1)',p5,pi3,pj3
c        iret=1
c        return
c      else
c        p5=sqrt(p5)
c      endif
c      call utlob2(1,p1,p2,p3,p4,p5,pi1,pi2,pi3,pi4,50)
c      call utlob2(1,p1,p2,p3,p4,p5,pj1,pj2,pj3,pj4,51)
c           err=(pi1+pj1)**2+(pi2+pj2)**2+(pi3+pj3)**2
c           if(err.gt.1d-3)then
c      call utmsg('uthalf')
c      write(ifch,*)'*****  err=',err
c      write(ifch,*)'pi:',pi1,pi2,pi3,pi4
c      write(ifch,*)'pj:',pj1,pj2,pj3,pj4
c      call utmsgf
c           endif
c      if(ish.ge.8)then
c        write(ifch,*)'pi 1:',pi1,pi2,pi3,pi4,pi5
c     &       ,sqrt(pi1**2+pi2**2+pi3**2+pi5**2)
c        write(ifch,*)'pj 1:',pj1,pj2,pj3,pj4,pj5
c     &       ,sqrt(pj1**2+pj2**2+pj3**2+pj5**2)
c      endif
c
c      phi=drangen(p5)*2d0*dble(pi)
c      pti=sqrt(pi1*pi1+pi2*pi2)
c
cc      sinp=abs(asin(pi2/pti))
c
c      cosp=cos(phi)
c      sinp=sin(phi)
cc      cosp=pi1/pti
cc      sinp=pi2/pti
c      pini=abs(pi3)
c      pmax=pini
c
c      ntry=0
c 10   ntry=ntry+1
c      pi1=-pj1
c      pi2=-pj2
c      pi3=-pj3
c
c
cc      rrr=dble(ranptcut(4.))*max(0.1d0,dble(zz))+0.01d0
cc      rrr=dble(ranptcut(zz*max(1.,float(ntry/10)**3)))        !flip if pt too large
cc      rrr=dble(min(1.,ranptcut(zz/float(max(1,ntry/10)))))        !return if pt too large
c
c      rrr=rangen()
c      phi=dble(pi*zz1*(1.-rrr*zz2/float(max(1,ntry/10))))
c      call utroa2(phi,cosp,sinp,0d0,pi1,pi2,pi3)    !rotation around an axis perpendicular to p3
c      pt=pi1*pi1+pi2*pi2
c
ccc      pi3=pini*(1d0-(pmax/pini+1d0)*min(1d0,rrr))
ccc     &   *sign(1.d0,pi3)
cc      pi3=pini*(1d0-(pmax/pini+1d0)*(1d0-min(1d0,rrr)))
cc     &   *sign(1.d0,pi3)
cccc      pi3=(1d0-exp(-drangen(pti)))*pi3
cccc      pi3=min(1d0,(drangen()+exp(-min(100d0,dble(zz)))))*pi3
cccc      pi3=(dble(reminv)+exp(-min(100d0,dble(zz))))*pi3
cc      pt=((pi4+pi3)*(pi4-pi3)-pi5*pi5)
c      if(ish.ge.8)write(ifch,*)'ut',ntry,zz,rrr,-pi3/pj3,pt,pti*pti
c     &,idptl(i),idptl(j)
c      if((pt.lt.0d0.or.pt.gt.max(2.*pti*pti,1.d0))
cc      if(pt.lt.0d0
c     &                               .and.ntry.lt.1000)then
c        goto 10
c      elseif(ntry.ge.1000)then
c        if(ish.ge.7)write(ifch,*)'Inversion not possible (2)',pt,pti*pti
c        iret=1
c        return         !pion distribution fall at xf=1
cc        pi3=pj3       !if flip all particle with large pt
cc        pt=pti*pti    !then very very hard pi+ spectra (flat at xf=1 like in pp400 !)
c                       !but to hard for K, rho, eta, etc ..
c      endif
cc      print *,'ut',ntry,phi/(pi-sinp),zz,rrr,-pi3/pj3,pt/pti/pti
cc      pt=sqrt(pt)
cc      pi1=cosp*pt
cc      pi2=sinp*pt
c
c      pj1=-pi1
c      pj2=-pi2
c      pj3=-pi3
cc      pi4=sqrt(pi3*pi3+pt*pt+pi5*pi5)
cc      pj4=sqrt(pj3*pj3+pt*pt+pj5*pj5)
c      pi4=sqrt(pi3*pi3+pt+pi5*pi5)
c      pj4=sqrt(pj3*pj3+pt+pj5*pj5)
c      if(ish.ge.8)then
c        write(ifch,*)'pi 2:',pi1,pi2,pi3,pi4,pi5
c     &       ,sqrt(pi1**2+pi2**2+pi3**2+pi5**2)
c        write(ifch,*)'pj 2:',pj1,pj2,pj3,pj4,pj5
c     &       ,sqrt(pj1**2+pj2**2+pj3**2+pj5**2)
c      endif
c      call utlob2(-1,p1,p2,p3,p4,p5,pi1,pi2,pi3,pi4,-50)
c      call utlob2(-1,p1,p2,p3,p4,p5,pj1,pj2,pj3,pj4,-51)
c      if(pi3/dble(pptl(3,i)).gt.1.00001d0
c     &.or.(pi3.gt.1.001d0*pj3.and.mod(ityptl(j)/10,10).eq.4)
c     &.or.(pi3.lt.1.001d0*pj3.and.mod(ityptl(j)/10,10).eq.5))then
c        if(ish.ge.7)write(ifch,*)'Inversion not possible (3)',pi3,pj3
c        iret=1
c        return
c      endif
c      id=idptl(i)
c      pptl(1,i)=sngl(pi1)
c      pptl(2,i)=sngl(pi2)
c      pptl(3,i)=sngl(pi3)
c      pptl(4,i)=sngl(pi4)
c      pptl(5,i)=sngl(pi5)
cc      pptl(5,i)=sngl(pj5)
cc      idptl(i)=idptl(j)
c      pptl(1,j)=sngl(pj1)
c      pptl(2,j)=sngl(pj2)
c      pptl(3,j)=sngl(pj3)
c      pptl(4,j)=sngl(pj4)
c      pptl(5,j)=sngl(pj5)
cc      pptl(5,j)=sngl(pi5)
cc      idptl(j)=id
c      if(ish.ge.6)then
c        write(ifch,*)'out ',idptl(i),pi1,pi2,pi3,pi4
c        write(ifch,*)' <> ',idptl(j),pj1,pj2,pj3,pj4
c      endif
c
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine utchm(arp,arm,ii)
cc-----------------------------------------------------------------------
cc     checks whether arp**2=0 and arm**2=0.
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      double precision arp(4),arm(4),difp,difm
c      difp=arp(4)**2-arp(1)**2-arp(2)**2-arp(3)**2
c      difm=arm(4)**2-arm(1)**2-arm(2)**2-arm(3)**2
c      if(dabs(difp).gt.1e-3*arp(4)**2
c     *.or.dabs(difm).gt.1e-3*arm(4)**2)then
c      call utmsg('utchm ')
c      write(ifch,*)'*****  mass non zero  -  ',ii
c      write(ifch,*)'jet-mass**2`s:    ',difp,difm
c      write(ifch,*)'energy**2`s:      ',arp(4)**2,arm(4)**2
c      write(ifch,*)(sngl(arp(i)),i=1,4)
c      write(ifch,*)(sngl(arm(i)),i=1,4)
c      call utmsgf
c      endif
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine utclea(nptlii,nptl0)
c-----------------------------------------------------------------------
c     starting from nptlii
c     overwrites istptl=99 particles in /cptl/, reduces so nptl
c     and update minfra and maxfra
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer newptl(mxptl)!,oldptl(mxptl),ii(mxptl)

      ish0=ish
      if(ishsub/100.eq.18)ish=mod(ishsub,100)

      call utpri('utclea',ish,ishini,2)

      nptli=max(maproj+matarg+1,nptlii)
      minfra0=minfra
      maxfra0=maxfra
      minfra1=maxfra
      maxfra1=minfra
      if(ish.ge.2)write(ifch,*)'entering subr utclea:',nptl
     &                                                ,minfra,maxfra
      if(ish.ge.7)then
      write(ifch,*)('-',l=1,68)
      write(ifch,*)'sr utclea. initial.'
      write(ifch,*)('-',l=1,68)
      do 34 n=nptli,nptl
      write(ifch,116)iorptl(n),jorptl(n),n,ifrptl(1,n),ifrptl(2,n)
     *,idptl(n),sqrt(pptl(1,n)**2+pptl(2,n)**2),pptl(3,n),pptl(5,n)
     *,istptl(n),ityptl(n)
34    continue
116   format(1x,i6,i6,4x,i6,4x,i6,i6,i12,3x,3(e8.2,1x),i3,i3)
      endif

c      ish=ish0
c      ish0=ish
c      if(ishsub/100.eq.18)ish=mod(ishsub,100)

      i=nptli-1
1     i=i+1
      if(i.gt.nptl)goto 1000
      if(istptl(i).eq.99)goto 2
      newptl(i)=i
c      oldptl(i)=i
      goto 1

2     i=i-1
      j=i
3     i=i+1
4     j=j+1
      if(j.gt.nptl)goto 5
      newptl(j)=0
      if(istptl(j).eq.99)goto 4
      newptl(j)=i
c      oldptl(i)=j
c      write(ifch,*)'move',j,' to ',i
c       write(ifch,*)idptl(i),ityptl(i),idptl(j),ityptl(j),minfra,maxfra
      call utrepl(i,j)
      if(j.ge.minfra0.and.j.le.maxfra0)then
        minfra1=min(minfra1,i)
        maxfra1=max(maxfra1,i)
      endif
      goto 3

5     nptl=i-1
      if(nptl.eq.0)then
        nptl0=0
        goto 1000
      endif

20    n0=newptl(nptl0)
      if(n0.gt.0)then
      nptl0=n0
      else
      nptl0=nptl0-1
      if(nptl0.gt.0)goto 20
      endif


c      do 11 k=1,nptl
c      io=iorptl(k)
c      if(io.le.0)ii(k)=io
c      if(io.gt.0)ii(k)=newptl(io)
c11    continue
c      do 12 k=1,nptl
c12    iorptl(k)=ii(k)
c
c      do 13 k=1,nptl
c      jo=jorptl(k)
c      if(jo.le.0)ii(k)=jo
c      if(jo.gt.0)ii(k)=newptl(jo)
c13    continue
c      do 14 k=1,nptl
c14    jorptl(k)=ii(k)
c
c      do 15 k=1,nptl
c      if1=ifrptl(1,k)
c      if(if1.le.0)ii(k)=if1
c      if(if1.gt.0)ii(k)=newptl(if1)
c15    continue
c      do 16 k=1,nptl
c16    ifrptl(1,k)=ii(k)
c
c      do 17 k=1,nptl
c      if2=ifrptl(2,k)
c      if(if2.le.0)ii(k)=if2
c      if(if2.gt.0)ii(k)=newptl(if2)
c17    continue
c      do 18 k=1,nptl
c18    ifrptl(2,k)=ii(k)
c
c      do 19 k=1,nptl
c      if(ifrptl(1,k).eq.0.and.ifrptl(2,k).gt.0)ifrptl(1,k)=ifrptl(2,k)
c      if(ifrptl(2,k).eq.0.and.ifrptl(1,k).gt.0)ifrptl(2,k)=ifrptl(1,k)
c19    continue

1000  continue

      if(minfra1.lt.minfra0)minfra=minfra1
      if(maxfra1.ge.minfra1)maxfra=maxfra1

      if(ish.ge.2)then
      write(ifch,*)'exiting subr utclea:'
      do 35 n=1,nptl
      write(ifch,116)iorptl(n),jorptl(n),n,ifrptl(1,n),ifrptl(2,n)
     *,idptl(n),sqrt(pptl(1,n)**2+pptl(2,n)**2),pptl(3,n),pptl(5,n)
     *,istptl(n),ityptl(n)
35    continue
      endif

      if(ish.ge.2)write(ifch,*)'exiting subr utclea:',nptl
     &                                                ,minfra,maxfra

      call utprix('utclea',ish,ishini,2)
      ish=ish0
      return
      end

c---------------------------------------------------------------------
      subroutine utfit(x,y,ndata,sig,mwt,a,b,siga,sigb,chi2,q)
c---------------------------------------------------------------------
c linear fit to data
c input:
c    ndata: nr of data points
c    x(),y(),sig(): data
c    mwt: unweighted (0) or weighted (else) data points
c output:
c    a,b: parameters of linear fit a+b*x
c---------------------------------------------------------------------
      INTEGER mwt,ndata
      REAL a,b,chi2,q,siga,sigb,sig(ndata),x(ndata),y(ndata)
CU    USES utgmq
      INTEGER i
      REAL sigdat,ss,st2,sx,sxoss,sy,t,wt,utgmq
      sx=0.
      sy=0.
      st2=0.
      b=0.
      if(mwt.ne.0) then
        ss=0.
        do 11 i=1,ndata
          wt=1./(sig(i)**2)
          ss=ss+wt
          sx=sx+x(i)*wt
          sy=sy+y(i)*wt
11      continue
      else
        do 12 i=1,ndata
          sx=sx+x(i)
          sy=sy+y(i)
12      continue
        ss=float(ndata)
      endif
      sxoss=sx/ss
      if(mwt.ne.0) then
        do 13 i=1,ndata
          t=(x(i)-sxoss)/sig(i)
          st2=st2+t*t
          b=b+t*y(i)/sig(i)
13      continue
      else
        do 14 i=1,ndata
          t=x(i)-sxoss
          st2=st2+t*t
          b=b+t*y(i)
14      continue
      endif
      b=b/st2
      a=(sy-sx*b)/ss
      siga=sqrt((1.+sx*sx/(ss*st2))/ss)
      sigb=sqrt(1./st2)
      chi2=0.
      if(mwt.eq.0) then
        do 15 i=1,ndata
          chi2=chi2+(y(i)-a-b*x(i))**2
15      continue
        q=1.
        sigdat=sqrt(chi2/(ndata-2))
        siga=siga*sigdat
        sigb=sigb*sigdat
      else
        do 16 i=1,ndata
          chi2=chi2+((y(i)-a-b*x(i))/sig(i))**2
16      continue
        q=utgmq(0.5*(ndata-2),0.5*chi2)
      endif
      return
      END

c-----------------------------------------------------------------------
      function utgam1(x)
c-----------------------------------------------------------------------
c  gamma fctn tabulated
c  single precision
c-----------------------------------------------------------------------
      double precision utgamtab,utgam,al,dl
      common/gamtab/utgamtab(10000)

      if(x.gt.0.01.and.x.lt.99.99)then
        al=100.d0*dble(x)
        k1=int(al)
        k2=k1+1
        dl =al-dble(k1)
        utgam1=real(utgamtab(k2)*dl+utgamtab(k1)*(1.d0-dl))
      elseif(x.eq.0.)then
        utgam1=0.
      else
        utgam1=real(utgam(dble(x)))
      endif

      end

c-----------------------------------------------------------------------
      double precision function utgam2(x)
c-----------------------------------------------------------------------
c  gamma fctn tabulated
c  double precision
c-----------------------------------------------------------------------
      double precision utgamtab,x,al,dl,utgam
      common/gamtab/utgamtab(10000)

      if(x.gt.0.01d0.and.x.le.99.99d0)then
        al=100.d0*x
        k1=int(al)
        k2=k1+1
        dl =al-dble(k1)
        utgam2=utgamtab(k2)*dl+utgamtab(k1)*(1.d0-dl)
      elseif(x.eq.0.d0)then
        utgam2=0.d0
      else
        utgam2=utgam(x)
      endif

      end

c-----------------------------------------------------------------------
      double precision function utgam(x)
c-----------------------------------------------------------------------
c  gamma fctn
c  double precision
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision c(13),x,z,f
      data c
     1/ 0.00053 96989 58808, 0.00261 93072 82746, 0.02044 96308 23590,
     2  0.07309 48364 14370, 0.27964 36915 78538, 0.55338 76923 85769,
     3  0.99999 99999 99998,-0.00083 27247 08684, 0.00469 86580 79622,
     4  0.02252 38347 47260,-0.17044 79328 74746,-0.05681 03350 86194,
     5  1.13060 33572 86556/
      utgam=0d0
      z=x
      if(x .gt. 170.d0) goto6
      if(x .gt. 0.0d0) goto1
      if(x .eq. int(x)) goto5
      z=1.0d0-z
    1 f=1.0d0/z
      if(z .le. 1.0d0) goto4
      f=1.0d0
    2 continue
      if(z .lt. 2.0d0) goto3
      z=z-1.0d0
      f=f*z
      goto2
    3 z=z-1.0d0
    4 utgam=
     1 f*((((((c(1)*z+c(2))*z+c(3))*z+c(4))*z+c(5))*z+c(6))*z+c(7))/
     2 ((((((c(8)*z+c(9))*z+c(10))*z+c(11))*z+c(12))*z+c(13))*z+1.0d0)
      if(x .gt. 0.0d0) return
      utgam=3.141592653589793d0/(sin(3.141592653589793d0*x)*utgam)
      return
    5 write(ifch,10)sngl(x)
   10 format(1x,'argument of gamma fctn = ',e20.5)
      call utstop('utgam : negative integer argument&')
    6 write(ifch,11)sngl(x)
   11 format(1x,'argument of gamma fctn = ',e20.5)
      call utstop('utgam : argument too large&')
      end

c---------------------------------------------------------------------
      subroutine utgcf(gammcf,a,x,gln)
c---------------------------------------------------------------------
      INTEGER ITMAX
      REAL a,gammcf,gln,x,EPS,FPMIN
      PARAMETER (ITMAX=100,EPS=3.e-7,FPMIN=1.e-30)
CU    USES utgmln
      INTEGER i
      REAL an,b,c,d,del,h,utgmln
      gln=utgmln(a)
      b=x+1.-a
      c=1./FPMIN
      d=1./b
      h=d
      do 11 i=1,ITMAX
        an=-i*(i-a)
        b=b+2.
        d=an*d+b
        if(abs(d).lt.FPMIN)d=FPMIN
        c=b+an/c
        if(abs(c).lt.FPMIN)c=FPMIN
        d=1./d
        del=d*c
        h=h*del
        if(abs(del-1.).lt.EPS)goto 1
11    continue
      call utstop("a too large, ITMAX too small in utgcf&")
1     gammcf=exp(-x+a*log(x)-gln)*h
      return
      END

c---------------------------------------------------------------------
      function utgmln(xx)
c---------------------------------------------------------------------
      REAL utgmln,xx
      INTEGER j
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      utgmln=tmp+log(stp*ser/x)
      return
      END

c---------------------------------------------------------------------
      function utgmq(a,x)
c---------------------------------------------------------------------
      REAL a,utgmq,x
CU    USES utgcf,utgser
      REAL gammcf,gamser,gln
      if(x.lt.0..or.a.le.0.) call utstop("bad arguments in utgmq&")
      if(x.lt.a+1.)then
        call utgser(gamser,a,x,gln)
        utgmq=1.-gamser
      else
        call utgcf(gammcf,a,x,gln)
        utgmq=gammcf
      endif
      return
      END

c---------------------------------------------------------------------
      subroutine utgser(gamser,a,x,gln)
c---------------------------------------------------------------------
      INTEGER ITMAX
      REAL a,gamser,gln,x,EPS
      PARAMETER (ITMAX=100,EPS=3.e-7)
CU    USES utgmln
      INTEGER n
      REAL ap,del,sum,utgmln
      gln=utgmln(a)
      if(x.le.0.)then
        if(x.lt.0.)call utstop("x < 0 in utgser&")
        gamser=0.
        return
      endif
      ap=a
      sum=1./a
      del=sum
      do 11 n=1,ITMAX
        ap=ap+1.
        del=del*x/ap
        sum=sum+del
        if(abs(del).lt.abs(sum)*EPS)goto 1
11    continue
      call utstop("a too large, ITMAX too small in utgser&")
1     gamser=sum*exp(-x+a*log(x)-gln)
      return
      END

c-------------------------------------------------------------------------
      subroutine uticpl(ic,ifla,iqaq,iret)
c-------------------------------------------------------------------------
c  adds a quark (iqaq=1) or antiquark (iqaq=2) of flavour ifla
c  to 2-id ic
c-------------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),ic(2)
      iret=0
      if(ifla.eq.0)return
      call iddeco(ic,jc)
      if(ish.ge.8)write(ifch,'(2i8,12i3)')ic,jc
      jqaq=3-iqaq
      if(jc(ifla,jqaq).gt.0)then
      jc(ifla,jqaq)=jc(ifla,jqaq)-1
      else
      jc(ifla,iqaq)=jc(ifla,iqaq)+1
      endif
      call idcomj(jc)
      call idenco(jc,ic,ireten)
      if(ish.ge.8)write(ifch,'(2i8,12i3)')ic,jc
      if(ireten.eq.1)iret=1
      if(ic(1).eq.0.and.ic(2).eq.0.and.ireten.eq.0)then
      ic(1)=100000
      ic(2)=100000
      endif
      return
      end

cc-----------------------------------------------------------------------
c      subroutine utindx(n,xar,x,i)
cc-----------------------------------------------------------------------
cc  input:  dimension n
cc          array xar(n) with xar(i) > xar(i-1)
cc          some number x between xar(1) and xar(n)
cc  output: the index i such that x is between xar(i)  and xar(i+1)
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      real xar(n)
c           if(x.lt.xar(1))then
c      if(ish.ge.5)then
c      call utmsg('utindx')
c      write(ifch,*)'*****  x=',x,' < xar(1)=',xar(1)
c      call utmsgf
c      endif
c      i=1
c      return
c           elseif(x.gt.xar(n))then
c      if(ish.ge.5)then
c      call utmsg('utindx')
c      write(ifch,*)'*****  x=',x,' > xar(n)=',xar(n)
c      call utmsgf
c      endif
c      i=n
c      return
c           endif
c      lu=1
c      lo=n
c1     lz=(lo+lu)/2
c      if((xar(lu).le.x).and.(x.le.xar(lz)))then
c      lo=lz
c      elseif((xar(lz).lt.x).and.(x.le.xar(lo)))then
c      lu=lz
c      else
c      call utstop('utindx: no interval found&')
c      endif
c      if((lo-lu).ge.2) goto1
c      if(lo.le.lu)call utstop('utinvt: lo.le.lu&')
c      i=lu
c      return
c      end
c
c-----------------------------------------------------------------------
      function utinvt(n,x,q,y)
c-----------------------------------------------------------------------
c     returns x with y=q(x)
c-----------------------------------------------------------------------
      include 'epos.inc'
      real x(n),q(n)
      if(q(n).eq.0.)call utstop('utinvt: q(n)=0&')
           if(y.lt.0.)then
      if(ish.ge.1)then
      call utmsg('utinvt')
      write(ifch,*)'*****  y=',y,' < 0'
      call utmsgf
      endif
      y=0.
           elseif(y.gt.q(n))then
      if(ish.ge.1)then
      call utmsg('utinvt')
      write(ifch,*)'*****  y=',y,' > ',q(n)
      call utmsgf
      endif
      y=q(n)
           endif
      lu=1
      lo=n
1     lz=(lo+lu)/2
      if((q(lu).le.y).and.(y.le.q(lz)))then
      lo=lz
      elseif((q(lz).lt.y).and.(y.le.q(lo)))then
      lu=lz
      else
      write(ifch,*)'q(1),y,q(n):',q(1),y,q(n)
      write(ifch,*)'lu,lz,lo:',lu,lz,lo
      write(ifch,*)'q(lu),q(lz),q(lo):',q(lu),q(lz),q(lo)
      call utstop('utinvt: no interval found&')
      endif
      if((lo-lu).ge.2) goto1
      if(lo.le.lu)call utstop('utinvt: lo.le.lu&')
      utinvt=x(lu)+(y-q(lu))*(x(lo)-x(lu))/(q(lo)-q(lu))
      return
      end

c-----------------------------------------------------------------------
      subroutine utlob2(isig,p1,p2,p3,p4,p5,x1,x2,x3,x4,idi)
c-----------------------------------------------------------------------
c  performs a lorentz boost, double prec.
c  isig=+1 is to boost the four vector x1,x2,x3,x4 such as to obtain it
c  in the frame specified by the 5-vector p1...p5 (5-vector=4-vector+mass).
c  isig=-1: the other way round, that means,
c  if the 4-vector x1...x4 is given in some frame characterized by
c  p1...p5 with respect to to some lab-frame, utlob2 returns the 4-vector
c  x1...x4  in the lab frame.
c  idi is a call identifyer (integer) to identify the call in case of problem
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision beta(4),z(4),p1,p2,p3,p4,p5,pp,bp,x1,x2,x3,x4
     *,xx0,x10,x20,x30,x40,x4x,x0123
           if(ish.ge.2)then
      if(ish.ge.9)then
      write(ifch,101)x1,x2,x3,x4,(x4-x3)*(x4+x3)-x2*x2-x1*x1
      write(ifch,301)p1,p2,p3,p4,p5,(p4-p3)*(p4+p3)-p2*p2-p1*p1
101   format(' utlob2: x =  ',5e13.5)
301   format('         p =  ',6e13.5)
      endif
      pp=(p4-p3)*(p4+p3)-p2*p2-p1*p1
      if(dabs(pp-p5*p5).gt.1e-3*p4*p4.and.dabs(pp-p5*p5).gt.1e-3)then
      call utmsg('utlob2')
      write(ifch,*)'*****  p**2 .ne. p5**2'
      write(ifch,*)'call identifyer:',idi
      write(ifch,*)'p**2,p5**2: ',pp,p5*p5
      write(ifch,*)'p: ',p1,p2,p3,p4,p5
      call utmsgf
      endif
      x10=x1
      x20=x2
      x30=x3
      x40=x4
           endif
      xx0=(x4-x3)*(x4+x3)-x2*x2-x1*x1
      if(p5.le.0.)then
      call utmsg('utlob2')
      write(ifch,*)'*****  p5 negative.'
      write(ifch,*)'call identifyer:',idi
      write(ifch,*)'p(5): ',p1,p2,p3,p4,p5
      write(ifmt,*)'call identifyer:',idi
      write(ifmt,*)'p(5): ',p1,p2,p3,p4,p5
      call utmsgf
      call utstop('utlob2: p5 negative.&')
      endif
      z(1)=x1
      z(2)=x2
      z(3)=x3
      z(4)=x4
      beta(1)=-p1/p5
      beta(2)=-p2/p5
      beta(3)=-p3/p5
      beta(4)= p4/p5
      bp=0.
      do 220 k=1,3
220   bp=bp+z(k)*isig*beta(k)
      do 230 k=1,3
230   z(k)=z(k)+isig*beta(k)*z(4)
     *+isig*beta(k)*bp/(beta(4)+1.)
      z(4)=beta(4)*z(4)+bp
      x1=z(1)
      x2=z(2)
      x3=z(3)
      x4=z(4)
      if(ish.ge.9)
     *write(ifch,101)x1,x2,x3,x4,(x4-x3)*(x4+x3)-x2*x2-x1*x1
      x4x=x4
      x0123=xx0+x1*x1+x2*x2+x3*x3
      if(x0123.gt.0.)then
      x4=sign( dsqrt(x0123) , x4x )
      else
      x4=0
      endif
      if(ish.ge.9)then
      write(ifch,101)x1,x2,x3,x4,(x4-x3)*(x4+x3)-x2*x2-x1*x1
      endif
           if(ish.ge.2)then
      if(ish.ge.9)write(ifch,*)'check x**2_ini -- x**2_fin'
      if(dabs(x4-x4x).gt.1d-2*dabs(x4).and.dabs(x4-x4x).gt.1d-2)then
      call utmsg('utlob2')
      write(ifch,*)'*****  x**2_ini .ne. x**2_fin.'
      write(ifch,*)'call identifyer:',idi
      write(ifch,*)'x1 x2 x3 x4 x**2 (initial/final/corrected):'
102   format(5e13.5)
      write(ifch,102)x10,x20,x30,x40,(x40-x30)*(x40+x30)-x20*x20-x10*x10
      write(ifch,102)x1,x2,x3,x4x,(x4x-x3)*(x4x+x3)-x2*x2-x1*x1
      write(ifch,102)x1,x2,x3,x4,(x4-x3)*(x4+x3)-x2*x2-x1*x1
      call utmsgf
      endif
           endif
      if(ish.ge.9)write(ifch,*)'return from utlob2'
      return
      end

c-----------------------------------------------------------------------
      subroutine utlob3(isig,p1,p2,p3,p4,p5,x1,x2,x3,x4)
c-----------------------------------------------------------------------
c  performs a lorentz boost, double prec.
c  but arguments are single precision
c-----------------------------------------------------------------------
      double precision xx1,xx2,xx3,xx4
      xx1=dble(x1)
      xx2=dble(x2)
      xx3=dble(x3)
      xx4=dble(x4)
      call utlob2(isig
     *,dble(p1),dble(p2),dble(p3),dble(p4),dble(p5)
     *,xx1,xx2,xx3,xx4,52)
      x1=sngl(xx1)
      x2=sngl(xx2)
      x3=sngl(xx3)
      x4=sngl(xx4)
      return
      end

c-----------------------------------------------------------------------
      subroutine utlob5(yboost,x1,x2,x3,x4,x5)
c-----------------------------------------------------------------------
      amt=sqrt(x5**2+x1**2+x2**2)
      y=sign(1.,x3)*alog((x4+abs(x3))/amt)
      y=y-yboost
      x4=amt*cosh(y)
      x3=amt*sinh(y)
      return
      end

c-----------------------------------------------------------------------
      subroutine utlob4(isig,pp1,pp2,pp3,pp4,pp5,x1,x2,x3,x4)
c-----------------------------------------------------------------------
c  performs a lorentz boost, double prec.
c  but arguments are partly single precision
c-----------------------------------------------------------------------
      double precision xx1,xx2,xx3,xx4,pp1,pp2,pp3,pp4,pp5
      xx1=dble(x1)
      xx2=dble(x2)
      xx3=dble(x3)
      xx4=dble(x4)
      call utlob2(isig,pp1,pp2,pp3,pp4,pp5,xx1,xx2,xx3,xx4,53)
      x1=sngl(xx1)
      x2=sngl(xx2)
      x3=sngl(xx3)
      x4=sngl(xx4)
      return
      end


c-----------------------------------------------------------------------
      subroutine utlobo(isig,p1,p2,p3,p4,p5,x1,x2,x3,x4)
c-----------------------------------------------------------------------
c     performs a lorentz boost
c-----------------------------------------------------------------------
      include 'epos.inc'
      real beta(4),z(4)
      if(p5.le.0.)then
      call utmsg('utlobo')
      write(ifch,*)'*****  mass <= 0.'
      write(ifch,*)'p(5): ',p1,p2,p3,p4,p5
      call utmsgf
      call utstop('utlobo: mass <= 0.&')
      endif
      z(1)=x1
      z(2)=x2
      z(3)=x3
      z(4)=x4
      beta(1)=-p1/p5
      beta(2)=-p2/p5
      beta(3)=-p3/p5
      beta(4)= p4/p5
      bp=0.
      do 220 k=1,3
220   bp=bp+z(k)*isig*beta(k)
      do 230 k=1,3
230   z(k)=z(k)+isig*beta(k)*z(4)
     *+isig*beta(k)*bp/(beta(4)+1.)
      z(4)=beta(4)*z(4)+bp
      x1=z(1)
      x2=z(2)
      x3=z(3)
      x4=z(4)
      return
      end

c-----------------------------------------------------------------------
      subroutine utloc(ar,n,a,l)
c-----------------------------------------------------------------------
      real ar(n)
      do 1 i=1,n
      l=i-1
      if(a.lt.ar(i))return
1     continue
      l=n
      return
      end

cc-----------------------------------------------------------------------
c      subroutine utlow(cone)
cc-----------------------------------------------------------------------
c      character*1 cone
c      if(cone.eq.'A')cone='a'
c      if(cone.eq.'B')cone='b'
c      if(cone.eq.'C')cone='c'
c      if(cone.eq.'D')cone='d'
c      if(cone.eq.'E')cone='e'
c      if(cone.eq.'F')cone='f'
c      if(cone.eq.'G')cone='g'
c      if(cone.eq.'H')cone='h'
c      if(cone.eq.'I')cone='i'
c      if(cone.eq.'J')cone='j'
c      if(cone.eq.'K')cone='k'
c      if(cone.eq.'L')cone='l'
c      if(cone.eq.'M')cone='m'
c      if(cone.eq.'N')cone='n'
c      if(cone.eq.'O')cone='o'
c      if(cone.eq.'P')cone='p'
c      if(cone.eq.'Q')cone='q'
c      if(cone.eq.'R')cone='r'
c      if(cone.eq.'S')cone='s'
c      if(cone.eq.'T')cone='t'
c      if(cone.eq.'U')cone='u'
c      if(cone.eq.'V')cone='v'
c      if(cone.eq.'W')cone='w'
c      if(cone.eq.'X')cone='x'
c      if(cone.eq.'Y')cone='y'
c      if(cone.eq.'Z')cone='z'
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine utlow3(cthree)
cc-----------------------------------------------------------------------
c      character cthree*3
c      do 1 i=1,3
c1     call utlow(cthree(i:i))
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine utlow6(csix)
cc-----------------------------------------------------------------------
c      character csix*6
c      do 1 i=1,6
c1     call utlow(csix(i:i))
c      return
c      end
c
cc-----------------------------------------------------------------------
c      function utmom(k,n,x,q)
cc-----------------------------------------------------------------------
cc     calculates kth moment for f(x) with q(i)=int[0,x(i)]f(z)dz
cc-----------------------------------------------------------------------
c      real x(n),q(n)
c      if(n.lt.2)call utstop('utmom : dimension too small&')
c      utmom=0
c      do 1 i=2,n
c1     utmom=utmom+((x(i)+x(i-1))/2)**k*(q(i)-q(i-1))
c      utmom=utmom/q(n)
c      return
c      end
c
c-----------------------------------------------------------------------
      function utpcm(a,b,c)
c-----------------------------------------------------------------------
c     calculates cm momentum for a-->b+c
c-----------------------------------------------------------------------
      val=(a*a-b*b-c*c)*(a*a-b*b-c*c)-(2.*b*c)*(2.*b*c)
      if(val.lt.0..and.val.gt.-1e-4)then
      utpcm=0
      return
      endif
      utpcm=sqrt(val)/(2.*a)
      return
      end

c-----------------------------------------------------------------------
      double precision function utpcmd(a,b,c,iret)
c-----------------------------------------------------------------------
c     calculates cm momentum for a-->b+c
c-----------------------------------------------------------------------
      double precision a,b,c,val
      iret=0
      val=(a*a-b*b-c*c)*(a*a-b*b-c*c)-(2.*b*c)*(2.*b*c)
      utpcmd=0d0
      if(val.lt.0d0.and.val.gt.-1d-4)then
        return
      elseif(val.lt.0d0)then
        iret=1
        return
      endif
      utpcmd=sqrt(val)/(2.d0*a)
      return
      end

c-----------------------------------------------------------------------
      subroutine utpri(text,ishi,ishini,ishx)
c-----------------------------------------------------------------------
      include 'epos.inc'
      character*6 text
c      double precision seedx                               !!!
      ishini=ishi
      if(ishevt.ne.0.and.nrevt+1.ne.ishevt)return
      if(nrpri.gt.0)then
      do nr=1,nrpri
      if(subpri(nr)(1:6).eq.text)then
      ishi=ishpri(nr)
      endif
      enddo
      endif
      if(ish.ge.ishx)then
        write(ifch,'(1x,43a)')
     *  ('-',i=1,10),' entry ',text,' ',('-',i=1,30)
c       call ranfgt(seedx)                                   !!!
c       if(ish.ge.ishx)write(ifch,*)'seed:',seedx            !!!
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine utprix(text,ishi,ishini,ishx)
c-----------------------------------------------------------------------
      include 'epos.inc'
      character*6 text
      if(ishevt.ne.0.and.nrevt+1.ne.ishevt)return
      if(ish.ge.ishx)write(ifch,'(1x,44a)')
     *('-',i=1,30),' exit ',text,' ',('-',i=1,11)
      ishi=ishini
      return
      end

c-----------------------------------------------------------------------
      subroutine utprj(text,ishi,ishini,ishx)
c-----------------------------------------------------------------------
      include 'epos.inc'
      character*20 text
c      double precision seedx                               !!!
      idx=index(text,' ')-1
      ishini=ishi
      if(ishevt.ne.0.and.nrevt+1.ne.ishevt)return
      if(nrpri.gt.0)then
      do nr=1,nrpri
      if(subpri(nr)(1:idx).eq.text(1:idx))then
      ishi=ishpri(nr)
      endif
      enddo
      endif
      if(ish.ge.ishx)then
        write(ifch,'(1x,43a)')
     *  ('-',i=1,10),' entry ',text(1:idx),' ',('-',i=1,30)
c       call ranfgt(seedx)                                   !!!
c       if(ish.ge.ishx)write(ifch,*)'seed:',seedx            !!!
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine utprjx(text,ishi,ishini,ishx)
c-----------------------------------------------------------------------
      include 'epos.inc'
      character*20 text
      idx=index(text,' ')-1
      if(ishevt.ne.0.and.nrevt+1.ne.ishevt)return
      if(ish.ge.ishx)write(ifch,'(1x,44a)')
     *('-',i=1,30),' exit ',text(1:idx),' ',('-',i=1,11)
      ishi=ishini
      return
      end

c-----------------------------------------------------------------------
      function utquad(m,x,f,k)
c-----------------------------------------------------------------------
c     performs an integration according to simpson
c-----------------------------------------------------------------------
      real x(m),f(m)
      utquad=0
      do 1 i=1,k-1
  1   utquad=utquad+(f(i)+f(i+1))/2*(x(i+1)-x(i))
      return
      end

c-----------------------------------------------------------------------
      subroutine utquaf(fu,n,x,q,x0,x1,x2,x3)
c-----------------------------------------------------------------------
c     returns q(i) = integral [x(1)->x(i)] fu(x) dx
c-----------------------------------------------------------------------
      include 'epos.inc'
      real x(n),q(n)
      parameter (m=10)
      real xa(m),fa(m)
      external fu
      if(x1.lt.x0.or.x2.lt.x1.or.x3.lt.x2)then
      if(ish.ge.1)then
      call utmsg('utquaf')
      write(ifch,*)'   xi=',x0,x1,x2,x3
      call utmsgf
      endif
      endif
      call utar(n/3,n*2/3,n,x0,x1,x2,x3,x)
      q(1)=0
      do 2 i=2,n
      do 3 k=1,m
      z=x(i-1)+(k-1.)/(m-1.)*(x(i)-x(i-1))
      xa(k)=z
3     fa(k)=fu(z)
      q(i)=q(i-1)+utquad(m,xa,fa,m)
2     continue
      return
      end

c-----------------------------------------------------------------------
      subroutine utrepl(i,j)
c-----------------------------------------------------------------------
c     i is replaced by j in /cptl/
c-----------------------------------------------------------------------
      include 'epos.inc'
      do 1 k=1,5
1     pptl(k,i)  =pptl(k,j)
      iorptl(i)  = 0 !iorptl(j)
      idptl(i)   =idptl(j)
      istptl(i)  =istptl(j)
      do 2 k=1,2
2     tivptl(k,i)=tivptl(k,j)
      do 3 k=1,2
3     ifrptl(k,i)= 0 !ifrptl(k,j)
      jorptl(i)  = 0 !jorptl(j)
      do 4 k=1,4
4     xorptl(k,i)=xorptl(k,j)
      do 5 k=1,4
5     ibptl(k,i) =ibptl(k,j)
      ityptl(i)  =ityptl(j)
      iaaptl(i)  =iaaptl(j)
      radptl(i)  =radptl(j)
      desptl(i)  =desptl(j)
      dezptl(i)  =dezptl(j)
      qsqptl(i)  =qsqptl(j)
      zpaptl(1,i)=zpaptl(1,j)
      zpaptl(2,i)=zpaptl(2,j)
      itsptl(i)  =itsptl(j)
      rinptl(i)  =rinptl(j)
      return
      end

c-----------------------------------------------------------------------
      subroutine utrepla(i,j)
c-----------------------------------------------------------------------
c     i is replaced by j in /cptl/
c-----------------------------------------------------------------------
      include 'epos.inc'
      do 1 k=1,5
1     pptl(k,i)  =pptl(k,j)
      iorptl(i)  = iorptl(j)
      idptl(i)   =idptl(j)
      istptl(i)  =istptl(j)
      do 2 k=1,2
2     tivptl(k,i)=tivptl(k,j)
      do 3 k=1,2
3     ifrptl(k,i)= ifrptl(k,j)
      jorptl(i)  = jorptl(j)
      do 4 k=1,4
4     xorptl(k,i)=xorptl(k,j)
      do 5 k=1,4
5     ibptl(k,i) =ibptl(k,j)
      ityptl(i)  =ityptl(j)
      iaaptl(i)  =iaaptl(j)
      radptl(i)  =radptl(j)
      desptl(i)  =desptl(j)
      dezptl(i)  =dezptl(j)
      qsqptl(i)  =qsqptl(j)
      zpaptl(1,i)=zpaptl(1,j)
      zpaptl(2,i)=zpaptl(2,j)
      itsptl(i)  =itsptl(j)
      rinptl(i)  =rinptl(j)
      return
      end

cc-----------------------------------------------------------------------
c      subroutine utresm(icp1,icp2,icm1,icm2,amp,idpr,iadj,ireten)
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      integer icm(2),icp(2),jcm(nflav,2),jcp(nflav,2)
c      icm(1)=icm1
c      icm(2)=icm2
c      icp(1)=icp1
c      icp(2)=icp2
c      CALL IDDECO(ICM,JCM)
c      CALL IDDECO(ICP,JCP)
c      do 37 nf=1,nflav
c      do 37 k=1,2
c37    jcP(nf,k)=jcp(nf,k)+jcm(nf,k)
c      CALL IDENCO(JCP,ICP,IRETEN)
c      IDP=IDTRA(ICP,0,0,3)
c      call idres(idp,amp,idpr,iadj)
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine utroa1(phi,a1,a2,a3,x1,x2,x3)
cc-----------------------------------------------------------------------
cc  rotates x by angle phi around axis a.
cc  normalization of a is irrelevant.
cc-----------------------------------------------------------------------
c      double precision aaa,aa(3),xxx,xx(3),e1(3),e2(3),e3(3),xp,xt,dphi
c      dphi=phi
c      xx(1)=x1
c      xx(2)=x2
c      xx(3)=x3
c      aa(1)=a1
c      aa(2)=a2
c      aa(3)=a3
c      aaa=0
c      xxx=0
c      do i=1,3
c      aaa=aaa+aa(i)**2
c      xxx=xxx+xx(i)**2
c      enddo
c      if(xxx.eq.0d0)return
c      if(aaa.eq.0d0)call utstop('utroa1: zero rotation axis&')
c      aaa=dsqrt(aaa)
c      xxx=dsqrt(xxx)
cc e3 = a / !a!
c      do i=1,3
c      e3(i)=aa(i)/aaa
c      enddo
cc x_parallel
c      xp=0
c      do i=1,3
c      xp=xp+xx(i)*e3(i)
c      enddo
cc x_transverse
c      if(xxx**2-xp**2.le.0.)return
c      xt=dsqrt(xxx**2-xp**2)
cc e1 = vector x_transverse / absolute value x_transverse
c      do i=1,3
c      e1(i)=(xx(i)-e3(i)*xp)/xt
c      enddo
cc e2 orthogonal e3,e1
c      call utvec2(e3,e1,e2)
cc rotate x
c      do i=1,3
c      xx(i)=xp*e3(i)+xt*dcos(dphi)*e1(i)+xt*dsin(dphi)*e2(i)
c      enddo
cc back to single precision
c      x1=xx(1)
c      x2=xx(2)
c      x3=xx(3)
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine utroa1(phi,a1,a2,a3,x1,x2,x3)
c-----------------------------------------------------------------------
c  rotates x by angle phi around axis a (argument single precision)
c  normalization of a is irrelevant.
c-----------------------------------------------------------------------
      double precision aa(3),xx(3),dphi
      dphi=phi
      xx(1)=x1
      xx(2)=x2
      xx(3)=x3
      aa(1)=a1
      aa(2)=a2
      aa(3)=a3
      call utroa2(dphi,aa(1),aa(2),aa(3),xx(1),xx(2),xx(3))
c back to single precision
      x1=sngl(xx(1))
      x2=sngl(xx(2))
      x3=sngl(xx(3))
      return
      end

c-----------------------------------------------------------------------
      subroutine utroa2(phi,a1,a2,a3,x1,x2,x3)
c-----------------------------------------------------------------------
c  rotates x by angle phi around axis a.
c  normalization of a is irrelevant.
c  double precision phi,a1,a2,a3,x1,x2,x3
c-----------------------------------------------------------------------
      double precision phi,a1,a2,a3,x1,x2,x3
      double precision aaa,aa(3),xxx,xx(3),e1(3),e2(3),e3(3),xp,xt,dphi
      dphi=phi
      xx(1)=x1
      xx(2)=x2
      xx(3)=x3
      aa(1)=a1
      aa(2)=a2
      aa(3)=a3
      aaa=0d0
      xxx=0d0
      do i=1,3
      aaa=aaa+aa(i)**2
      xxx=xxx+xx(i)**2
      enddo
      if(xxx.eq.0d0)return
      if(aaa.eq.0d0)call utstop('utroa1: zero rotation axis&')
      aaa=1.0/dsqrt(aaa)
c e3 = a / !a!
      do i=1,3
      e3(i)=aa(i)*aaa
      enddo
c x_parallel
      xp=0
      do i=1,3
      xp=xp+xx(i)*e3(i)
      enddo
c x_transverse
      if(xxx-xp**2.le.0.)return
      xt=dsqrt(xxx-xp**2)
c e1 = vector x_transverse / absolute value x_transverse
      do i=1,3
      e1(i)=(xx(i)-e3(i)*xp)/xt
      enddo
c e2 orthogonal e3,e1
      call utvec2(e3,e1,e2)
c rotate x
      do i=1,3
      xx(i)=xp*e3(i)+xt*cos(dphi)*e1(i)+xt*sin(dphi)*e2(i)
      enddo
      xxx=0d0
      do i=1,3
      xxx=xxx+xx(i)**2
      enddo
c back to single precision
      x1=xx(1)
      x2=xx(2)
      x3=xx(3)
      return
      end

cc--------------------------------------------------------------------
c      function utroot(funcd,x1,x2,xacc)
cc--------------------------------------------------------------------
cc combination of newton-raphson and bisection method for root finding
cc input:
cc   funcd: subr returning fctn value and first derivative
cc   x1,x2: x-interval
cc   xacc:  accuracy
cc output:
cc   utroot: root
cc--------------------------------------------------------------------
c      include 'epos.inc'
c      INTEGER MAXIT
c      REAL utroot,x1,x2,xacc
c      EXTERNAL funcd
c      PARAMETER (MAXIT=100)
c      INTEGER j
c      REAL df,dx,dxold,f,fh,fl,temp,xh,xl
c      call funcd(x1,fl,df)
c      call funcd(x2,fh,df)
c      if((fl.gt.0..and.fh.gt.0.).or.(fl.lt.0..and.fh.lt.0.))
c     *call utstop('utroot: root must be bracketed&')
c      if(fl.eq.0.)then
c        utroot=x1
c        return
c      else if(fh.eq.0.)then
c        utroot=x2
c        return
c      else if(fl.lt.0.)then
c        xl=x1
c        xh=x2
c      else
c        xh=x1
c        xl=x2
c      endif
c      utroot=.5*(x1+x2)
c      dxold=abs(x2-x1)
c      dx=dxold
c      call funcd(utroot,f,df)
c      do 11 j=1,MAXIT
c        if(((utroot-xh)*df-f)*((utroot-xl)*df-f).ge.0..or. abs(2.*
c     *f).gt.abs(dxold*df) ) then
c          dxold=dx
c          dx=0.5*(xh-xl)
c          utroot=xl+dx
c          if(xl.eq.utroot)return
c        else
c          dxold=dx
c          dx=f/df
c          temp=utroot
c          utroot=utroot-dx
c          if(temp.eq.utroot)return
c        endif
c        if(abs(dx).lt.xacc) return
c        call funcd(utroot,f,df)
c        if(f.lt.0.) then
c          xl=utroot
c        else
c          xh=utroot
c        endif
c11    continue
c      call utmsg('utroot')
c      write(ifch,*)'*****  exceeding maximum iterations'
c      write(ifch,*)'dx:',dx
c      call utmsgf
c      return
c      END
c
c-----------------------------------------------------------------------
      subroutine utrot2(isig,ax,ay,az,x,y,z)
c-----------------------------------------------------------------------
c     performs a rotation, double prec.
c-----------------------------------------------------------------------
      include 'epos.inc'
      double precision ax,ay,az,x,y,z,rx,ry,rz
     *,alp,bet,cosa,sina,cosb,sinb,xs,ys,zs
         if(ax**2.eq.0.and.ay**2.eq.0.and.az**2.eq.0.)then
      write(ifch,*)'ax**2,ay**2,az**2:',ax**2,ay**2,az**2
      write(ifch,*)'ax,ay,az:',ax,ay,az
      call utstop('utrot2: zero vector.&')
         endif
         if(az.ge.0.)then
      rx=ax
      ry=ay
      rz=az
         else
      rx=-ax
      ry=-ay
      rz=-az
         endif
      if(rz**2+ry**2.ne.0.)then
      alp=dabs(dacos(rz/dsqrt(rz**2+ry**2)))*sign(1.,sngl(ry))
      bet=
     *dabs(dacos(dsqrt(rz**2+ry**2)/dsqrt(rz**2+ry**2+rx**2)))*
     *sign(1.,sngl(rx))
      else
      alp=3.1415927d0/2d0
      bet=3.1415927d0/2d0
      endif
      cosa=dcos(alp)
      sina=dsin(alp)
      cosb=dcos(bet)
      sinb=dsin(bet)
           if(isig.ge.0)then
      xs=x*cosb-y*sina*sinb-z*cosa*sinb
      ys=       y*cosa     -z*sina
      zs=x*sinb+y*sina*cosb+z*cosa*cosb
           else     !if(isig.lt.0)then
      xs= x*cosb            +z*sinb
      ys=-x*sinb*sina+y*cosa+z*cosb*sina
      zs=-x*sinb*cosa-y*sina+z*cosb*cosa
           endif
      x=xs
      y=ys
      z=zs
      return
      end


c-----------------------------------------------------------------------
      subroutine utrot4(isig,ax,ay,az,x,y,z)
c-----------------------------------------------------------------------
c     performs a rotation, double prec.
c     arguments partly single
c-----------------------------------------------------------------------
      double precision ax,ay,az,xx,yy,zz
      xx=dble(x)
      yy=dble(y)
      zz=dble(z)
      call utrot2(isig,ax,ay,az,xx,yy,zz)
      x=sngl(xx)
      y=sngl(yy)
      z=sngl(zz)
      return
      end

c-----------------------------------------------------------------------
      subroutine utrota(isig,ax,ay,az,x,y,z)
c-----------------------------------------------------------------------
c     performs a rotation
c-----------------------------------------------------------------------
         if(az.ge.0.)then
      rx=ax
      ry=ay
      rz=az
         else
      rx=-ax
      ry=-ay
      rz=-az
         endif
      if(rz.eq.0..and.ry.eq.0.)then
        alp=0.
        stop
      else
        alp=abs(utacos(rz/sqrt(rz**2+ry**2)))*sign(1.,ry)
      endif
      bet=
     *abs(utacos(sqrt(rz**2+ry**2)/sqrt(rz**2+ry**2+rx**2)))*sign(1.,rx)
      cosa=cos(alp)
      sina=sin(alp)
      cosb=cos(bet)
      sinb=sin(bet)
           if(isig.ge.0)then
      xs=x*cosb-y*sina*sinb-z*cosa*sinb
      ys=       y*cosa     -z*sina
      zs=x*sinb+y*sina*cosb+z*cosa*cosb
           else        !if(isig.lt.0)then
      xs= x*cosb            +z*sinb
      ys=-x*sinb*sina+y*cosa+z*cosb*sina
      zs=-x*sinb*cosa-y*sina+z*cosb*cosa
           endif
      x=xs
      y=ys
      z=zs
      return
      end

c-----------------------------------------------------------------------
      subroutine utstop(text)
c-----------------------------------------------------------------------
c  returns error message and stops execution.
c  text is an optonal text to appear in the error message.
c  text is a character string of length 40;
c     for shorter text, it has to be terminated by &;
c        example: call utstop('error in subr xyz&')
c-----------------------------------------------------------------------
      include 'epos.inc'
c      parameter(itext=40)
      character  text*(*)  ,txt*6
      imax=index(text,'&')
      do 1 j=1,2
      if(j.eq.1)then
        ifi=ifch
      else        !if(j.eq.2)
        ifi=ifmt
      endif
      if(imax.gt.1)then
      write(ifi,101)('*',k=1,72),text(1:imax-1)
     *,nrevt+1,nint(seedj),seedc,('*',k=1,72)
      else
      write(ifi,101)('*',k=1,72),' '
     *,nrevt+1,nint(seedj),seedc,('*',k=1,72)
      endif
101   format(
     *1x,72a1
     */1x,'***** stop in ',a
     */1x,'***** current event number: ',i12
     */1x,'***** initial seed for current run:',i10
     */1x,'***** initial seed for current event:',d25.15
     */1x,72a1)
1     continue
c      c=0.
c      b=a/c
      stop
      entry utmsg(txt)
      imsg=imsg+1
      write(ifch,'(1x,74a1)')('*',j=1,72)
      write(ifch,100)txt,nrevt+1,nint(seedj),seedc
100   format(1x,'***** msg from ',a6,'.   es:',i7,2x,i9,2x,d23.17)
      return
      entry utmsgf
      if(ish.eq.1)return
      write(ifch,'(1x,74a1)')('*',j=1,72)
      end

c-----------------------------------------------------------------
      subroutine uttrap(func,a,b,s)
c-----------------------------------------------------------------
c trapezoidal method for integration.
c input: fctn func and limits a,b
c output: value s of the integral
c-----------------------------------------------------------------
      include 'epos.inc'

      INTEGER JMAX
      REAL a,b,func,s
      EXTERNAL func
      PARAMETER (JMAX=10)
CU    USES uttras
      INTEGER j
      REAL olds
      olds=-1.e30
      do 11 j=1,JMAX
        if(ish.ge.9)write(ifch,*)'sr uttrap:   j:',j
        call uttras(func,a,b,s,j)
        ds=abs(s-olds)
        if (ds.lt.epsr*abs(olds)) return
        olds=s
11    continue

c-c   nepsr=nepsr+1
      if(ish.ge.9)then
      call utmsg('uttrap')
      write(ifch,*)
     *'*****  requested accuracy could not be achieved'
      write(ifch,*)'achieved accuracy: ',ds/abs(olds)
      write(ifch,*)'requested accuracy:',epsr
      call utmsgf
      endif

      END

c-----------------------------------------------------------------
      subroutine uttraq(func,a,b,s)
c-----------------------------------------------------------------
c trapezoidal method for integration.
c input: function func and limits a,b
c output: value s of the integral
c-----------------------------------------------------------------

      REAL a,b,func,s
      EXTERNAL func
      PARAMETER (eps=1.e-6)
CU    USES uttras
      INTEGER j
      REAL olds
      olds=-1.e30
      j=1
10      call uttras(func,a,b,s,j)
        ds=abs(s-olds)
        if (ds.le.eps*abs(olds)) return
        olds=s
        if(j.ge.15)return
        j=j+1
      goto10
      END

c-----------------------------------------------------------------
      subroutine uttras(func,a,b,s,n)
c-----------------------------------------------------------------
c performs one iteration of the trapezoidal method for integration
c-----------------------------------------------------------------
      INTEGER n
      REAL a,b,s,func
      EXTERNAL func
      INTEGER it,j
      REAL del,sum,tnm,x
      if (n.eq.1) then
        s=0.5*(b-a)*(func(a)+func(b))
      else
        it=2**(n-2)
        tnm=it
        del=(b-a)/tnm
        x=a+0.5*del
        sum=0.
        do 11 j=1,it
          sum=sum+func(x)
          x=x+del
11      continue
        s=0.5*(s+(b-a)*sum/tnm)
      endif
      return
      END

cc-----------------------------------------------------------------------
c      subroutine utvec1(a,b,c)
cc-----------------------------------------------------------------------
cc  returns vector product c = a x b .
cc-----------------------------------------------------------------------
c      real a(3),b(3),c(3)
c      c(1)=a(2)*b(3)-a(3)*b(2)
c      c(2)=a(3)*b(1)-a(1)*b(3)
c      c(3)=a(1)*b(2)-a(2)*b(1)
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine utvec2(a,b,c)
c-----------------------------------------------------------------------
c  returns vector product c = a x b .
c  a,b,c double precision.
c-----------------------------------------------------------------------
      double precision a(3),b(3),c(3)
      c(1)=a(2)*b(3)-a(3)*b(2)
      c(2)=a(3)*b(1)-a(1)*b(3)
      c(3)=a(1)*b(2)-a(2)*b(1)
      return
      end

c-------------------------------------------------------------------
      subroutine utword(line,i,j,iqu)
c-------------------------------------------------------------------
c  finds the first word of the character string line(j+1:1000).
c  the word is line(i:j) (with new i and j).
c  if j<0 or if no word found --> new line read.
c  a text between quotes "..." is considered one word;
c  stronger: a text between double quotes ""..."" is consid one word
c  stronger: a text between "{ and }" is considered one word
c-------------------------------------------------------------------
c  input:
c    line: character string (*1000)
c    i: integer between 1 and 1000
c    iqu: for iqu=1 a "?" is written to output before reading a line,
c         otherwise (iqu/=1) nothing is typed
c  output:
c    i,j: left and right end of word (word=line(i:j))
c-------------------------------------------------------------------
      include 'epos.inc'
      parameter(mempty=2)
      character*1 empty(mempty),mk
      character line*1000
      character*2 mrk
      data empty/' ',','/
      parameter(mxdefine=40)
      character w1define*100,w2define*100
      common/cdefine/ndefine,l1define(mxdefine),l2define(mxdefine)
     &               ,w1define(mxdefine),w2define(mxdefine)

      j0=0
      if(j.ge.0)then
      i=j
      goto 1
      endif

    5 continue
      if(iqu.eq.1.and.iprmpt.gt.0)write(ifmt,'(a)')'?'
      if(nopen.eq.0)then
        ifopx=ifop
      elseif(nopen.gt.0)then
        ifopx=20+nopen
      else !if(nopen.lt.0)
        ifopx=ifcp
      endif
      read(ifopx,'(a1000)',end=9999)line
      if(iecho.eq.1.or.(nopen.ge.0.and.kcpopen.eq.1))then
       kmax=2
       do k=3,1000
       if(line(k:k).ne.' ')kmax=k
       enddo
      else
        kmax=0
      endif
      if(nopen.ge.0.and.kcpopen.eq.1)
     &  write(ifcp,'(a)')line(1:kmax)
      if(iecho.eq.1)
     &  write(ifmt,'(a)')line(1:kmax)
      i=0

    1 i=i+1
      if(i.gt.1000)goto 5
      if(line(i:i).eq.'!')goto 5
      do ne=1,mempty
      if(line(i:i).eq.empty(ne))goto 1
      enddo

      nbla=1
      mrk='  '
      mk=' '
      if(line(i:i).eq.'~')mk='~'
      if(line(i:i+1).eq.'"{')mrk='}"'
      if(line(i:i+1).eq.'""')mrk='""'
      if(mrk.ne.'  ')goto 10
      if(line(i:i).eq.'"')mk='"'
      if(mk.ne.' ')goto 8
      j=i-1
    6 j=j+1
      if(j.gt.1000)goto 7
      if(line(j:j).eq.'!')goto 7
      do ne=1,mempty
      if(line(j:j).eq.empty(ne))goto 7
      enddo
      goto 6

    8 continue
      if(i.ge.1000-1)stop'utword: make line shorter!!!         '
      i=i+1
      j=i
      if(line(j:j).eq.mk)stop'utword: empty string!!!           '
    9 j=j+1
      if(j.gt.1000)then                 !reach the end of the line
        j=j-nbla+2
        goto 7
      endif
      if(line(j:j).eq.' ')then
        nbla=nbla+1
      else
        nbla=2
      endif
      if(line(j:j).eq.mk)then
      line(i-1:i-1)=' '
      line(j:j)=' '
      goto 7
      endif
      goto 9

   10 continue
      if(i.ge.1000-3)stop'utword: make line shorter!!!!          '
      i=i+2
      j=i
      if(line(j:j+1).eq.mrk)stop'utword: empty string!!!!        '
   11 j=j+1
      if(j.gt.1000-1)then                 !reach the end of the line
        j=j-nbla+2
        goto 7
      endif
      if(line(j:j+1).eq.mrk)then
      line(i-2:i-1)='  '
      line(j:j+1)='  '
      goto 7
      endif
      if(line(j:j).eq.' ')then
        nbla=nbla+1
      else
        nbla=2
      endif
      goto 11

    7 j=j-1
      !--------#define---------------
      if(ndefine.gt.0)then
        do ndf=1,ndefine
          l1=l1define(ndf)
          l2=l2define(ndf)
          do i0=i,j+1-l1
            if(line(i0:i0-1+l1).eq.w1define(ndf)(1:l1))then
              if(l2.eq.l1)then
                line(i0:i0-1+l1)=w2define(ndf)(1:l2)
              elseif(l2.lt.l1)then
                line(i0:i0+l2-1)=w2define(ndf)(1:l2)
                do k=i0+l2,i0-1+l1
                  line(k:k)=' '
                enddo
              elseif(l2.gt.l1)then
                do k=i0+l1,i0+l2-1
                  if(line(k:k).ne.' ')
     &        stop'utword: no space for `define` replacement.   '
                enddo
                line(i0:i0+l2-1)=w2define(ndf)(1:l2)
                j=i0+l2-1
              endif
            endif
          enddo
        enddo
       do k=i,j
         if(line(k:k).ne.' ')j0=j
       enddo
       j=j0
      endif
      !--------
      return

9999  close(ifopx)
      nopen=nopen-1
      if(nopen.eq.0.and.iprmpt.eq.-1)iprmpt=1
      goto 5
      end

c--------------------------------------------------------------------
      subroutine utworn(line,j,ne)
c--------------------------------------------------------------------
c  returns number ne of nonempty characters of line(j+1:1000)
c--------------------------------------------------------------------
      character line*1000
      ne=0
      do l=j+1,1000
      if(line(l:l).ne.' ')ne=ne+1
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine getairmol(iz,ia)
c-----------------------------------------------------------------------
      include 'epos.inc'
      i=0
      r=rangen()
      do while(r.gt.0.)  ! choose air-molecule
        i=i+1
        r=r-airwnxs(i)
      enddo
      iz = nint(airznxs(i))
      ia = nint(airanxs(i))
      end

c----------------------------------------------------------------------

      subroutine factoriel

c----------------------------------------------------------------------
c tabulation of fctrl(n)=n!, facto(n)=1/n! and utgamtab(x) for x=0 to 50
c----------------------------------------------------------------------
      include 'epos.incems'
      double precision utgamtab,utgam,x
      common/gamtab/utgamtab(10000)

      nfctrl=100
      fctrl(0)=1.D0
      facto(0)=1.D0
      do i=1,min(npommx,nfctrl)
        fctrl(i)=fctrl(i-1)*dble(i)
        facto(i)=1.d0/fctrl(i)
      enddo

      do k=1,10000
        x=dble(k)/100.d0
        utgamtab(k)=utgam(x)
      enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine fremnu(amin,ca,cb,ca0,cb0,ic1,ic2,ic3,ic4)
c-----------------------------------------------------------------------
      common/hadr2/iomodl,idproj,idtarg,wexcit
      real pnll,ptq
      common/hadr1/pnll,ptq,exmass,cutmss,wproj,wtarg
      parameter (nflav=6)
      integer ic(2),jc(nflav,2)
      ic(1)=ca
      ic(2)=cb
      call iddeco(ic,jc)
      keu=jc(1,1)-jc(1,2)
      ked=jc(2,1)-jc(2,2)
      kes=jc(3,1)-jc(3,2)
      kec=jc(4,1)-jc(4,2)
      keb=jc(5,1)-jc(5,2)
      ket=jc(6,1)-jc(6,2)
      amin=utamnu(keu,ked,kes,kec,keb,ket,5)  !???4=2mults, 5=1mult
        if(ca-ca0.eq.0..and.cb-cb0.eq.0..and.rangen().gt.wexcit)then
      ic3=0
      ic4=0
      ic1=ca
      ic2=cb
       else
      amin=amin+exmass
      n=0
      do i=1,4
      do j=1,2
      n=n+jc(i,j)
      enddo
      enddo
      k=1+rangen()*n
      do i=1,4
      do j=1,2
      k=k-jc(i,j)
      if(k.le.0)goto 1
      enddo
      enddo
1     if(j.eq.1)then
      ic3=10**(6-i)
      ic4=0
      else
      ic3=0
      ic4=10**(6-i)
      endif
      ic1=int(ca)-ic3
      ic2=int(cb)-ic4
        endif
      return
      end


c-----------------------------------------------------------------------
      function fremnux(jc)
c-----------------------------------------------------------------------
      real pnll,ptq
      common/hadr1/pnll,ptq,exmass,cutmss,wproj,wtarg
      parameter (nflav=6)
      integer jc(nflav,2)!,ic(2)
c      ic(1)=210000
c      ic(2)=0
c      call iddeco(ic,jc)
      keu=jc(1,1)-jc(1,2)
      ked=jc(2,1)-jc(2,2)
      kes=jc(3,1)-jc(3,2)
      kec=jc(4,1)-jc(4,2)
      keb=jc(5,1)-jc(5,2)
      ket=jc(6,1)-jc(6,2)
      fremnux=utamnu(keu,ked,kes,kec,keb,ket,4) !+exmass  !???4=2mults, 5=1mult
      return
      end

c-----------------------------------------------------------------------
      function fremnux2(jc)
c-----------------------------------------------------------------------
      real pnll,ptq
      common/hadr1/pnll,ptq,exmass,cutmss,wproj,wtarg
      parameter (nflav=6)
      integer jc(nflav,2)!,ic(2)
c      ic(1)=210000
c      ic(2)=0
c      call iddeco(ic,jc)
      keu=jc(1,1)-jc(1,2)
      ked=jc(2,1)-jc(2,2)
      kes=jc(3,1)-jc(3,2)
      kec=jc(4,1)-jc(4,2)
      keb=jc(5,1)-jc(5,2)
      ket=jc(6,1)-jc(6,2)
      fremnux2=utamnu(keu,ked,kes,kec,keb,ket,5) !+exmass  !???4=2mults, 5=1mult
      return
      end

c-----------------------------------------------------------------------
      function fremnux3(jci)
c-----------------------------------------------------------------------
c minimum mass from ic counting all quarks
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),jci(nflav,2)!,ic(2)
c      ic(1)=210000
c      ic(2)=0
c      print *,'start',ic
      fremnux3=0.
      do j=1,2
        do i=1,nflav
          jc(i,j)=jci(i,j)
        enddo
      enddo
c      call iddeco(ic,jc)
      call idquacjc(jc,nqua,naqu)
        do ii=1,2
      if(ii.eq.1)then
        nqu=nqua
      else
        nqu=naqu
      endif
      if(nqu.ge.3)then
        do while(jc(3,ii).ne.0.and.nqu.ge.3)  !count baryons with s quark
          jc(3,ii)=jc(3,ii)-1
          if(jc(3,ii).gt.0)then
            jc(3,ii)=jc(3,ii)-1
            if(jc(3,ii).gt.0)then
              jc(3,ii)=jc(3,ii)-1
              fremnux3=fremnux3+asuhax(4)
            elseif(jc(2,ii).gt.0)then
              jc(2,ii)=jc(2,ii)-1
              fremnux3=fremnux3+asuhax(4)
            elseif(jc(1,ii).gt.0)then
              jc(1,ii)=jc(1,ii)-1
              fremnux3=fremnux3+asuhax(4)
            endif
          elseif(jc(2,ii).gt.0)then
            jc(2,ii)=jc(2,ii)-1
            if(jc(1,ii).gt.0)then
              jc(1,ii)=jc(1,ii)-1
              fremnux3=fremnux3+asuhax(3)
            elseif(jc(2,ii).gt.0)then
              jc(2,ii)=jc(2,ii)-1
              fremnux3=fremnux3+asuhax(3)
            endif
          elseif(jc(1,ii).gt.0)then
            jc(1,ii)=jc(1,ii)-2
            fremnux3=fremnux3+asuhay(3)
          endif
          nqu=nqu-3
        enddo
        do while(jc(2,ii).ne.0.and.nqu.ge.3)  !count baryons with d quark
          jc(2,ii)=jc(2,ii)-1
          if(jc(1,ii).gt.0)then
            jc(1,ii)=jc(1,ii)-1
            if(jc(2,ii).gt.0)then
              jc(2,ii)=jc(2,ii)-1
              fremnux3=fremnux3+asuhay(2)
            elseif(jc(1,ii).gt.0)then
              jc(1,ii)=jc(1,ii)-1
              fremnux3=fremnux3+asuhay(2)
            endif
          elseif(jc(2,ii).gt.0)then
            jc(2,ii)=jc(2,ii)-2
            fremnux3=fremnux3+asuhay(3)
          endif
          nqu=nqu-3
        enddo
        do while(jc(1,ii).ne.0.and.nqu.ge.3)  !count baryons with s quark
          jc(1,ii)=jc(1,ii)-3
          fremnux3=fremnux3+asuhay(3)
          nqu=nqu-3
        enddo
        if(ii.eq.1)then
          nqua=nqu
        else
          naqu=nqu
        endif
      endif
c      print *,ii,nqua,naqu,jc,fremnux3
      enddo
      if(nqua+naqu.ne.0)then
      do while(jc(3,1).ne.0)    !count mesons with s quark
        jc(3,1)=jc(3,1)-1
        if(jc(3,2).gt.0)then
          jc(3,2)=jc(3,2)-1
          fremnux3=fremnux3+asuhax(6)
        elseif(jc(2,2).gt.0)then
          jc(2,2)=jc(2,2)-1
          fremnux3=fremnux3+asuhay(6)
        elseif(jc(1,2).gt.0)then
          jc(1,2)=jc(1,2)-1
          fremnux3=fremnux3+asuhay(6)
        endif
      enddo
      do while(jc(2,1).ne.0)    !count mesons with d quark
        jc(2,1)=jc(2,1)-1
        if(jc(2,2).gt.0)then
          jc(2,2)=jc(2,2)-1
          fremnux3=fremnux3+asuhay(5)
        elseif(jc(1,2).gt.0)then
          jc(1,2)=jc(1,2)-1
          fremnux3=fremnux3+asuhay(5)
        endif
      enddo
      do while(jc(1,1).ne.0)    !count mesons with s quark
        jc(1,1)=jc(1,1)-1
        if(jc(1,2).gt.0)then
          jc(1,2)=jc(1,2)-1
          fremnux3=fremnux3+asuhay(5)
        endif
      enddo
      endif
c      fremnux3=fremnux3+0.5
c      print *,'stop',nqua,naqu,fremnux3

      return
      end

c-----------------------------------------------------------------------
      subroutine fremnx(ammax,amin,sm,ic3,ic4,iret)
c-----------------------------------------------------------------------
      common/psar9/ alpr
      include 'epos.inc'
      iret=0
      if(ic3.eq.0.and.ic4.eq.0)then
        if(ammax.lt.amin**2)then
          iret=1
          return
        endif
        sm=amin**2
      else
c       ammax1=min(ammax,(engy/4.)**2)
        ammax1=ammax
        if(ammax1.lt.amin**2)then
          iret=1
          return
        endif
        if(alpr.eq.-1.)then
          sm=amin**2*(ammax1/amin**2)**rangen()
        else
          sm=amin**2*(1.+((ammax1/amin**2)**(1.+alpr)-1.)
     *    *rangen())**(1./(1.+alpr))
        endif
      endif
      return
      end

      SUBROUTINE gaulag(x,w,n,alf)
      INTEGER n,MAXIT
      REAL alf,w(n),x(n)
      DOUBLE PRECISION EPS
      PARAMETER (EPS=3.D-14,MAXIT=10)
CU    USES gammln
      INTEGER i,its,j
      REAL ai,gammln
      DOUBLE PRECISION p1,p2,p3,pp,z,z1
      z=0.
      do 13 i=1,n
        if(i.eq.1)then
          z=(1.+alf)*(3.+.92*alf)/(1.+2.4*n+1.8*alf)
        else if(i.eq.2)then
          z=z+(15.+6.25*alf)/(1.+.9*alf+2.5*n)
        else
          ai=i-2
          z=z+((1.+2.55*ai)/(1.9*ai)+1.26*ai*alf/(1.+3.5*ai))*
     *(z-x(i-2))/(1.+.3*alf)
        endif
        do 12 its=1,MAXIT
          p1=1.d0
          p2=0.d0
          do 11 j=1,n
            p3=p2
            p2=p1
            p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j
11        continue
          pp=(n*p1-(n+alf)*p2)/z
          z1=z
          z=z1-p1/pp
          if(abs(z-z1).le.EPS)goto 1
12      continue
        call utstop("too many iterations in gaulag")
1       x(i)=z
        w(i)=-exp(gammln(alf+n)-gammln(float(n)))/(pp*n*p2)
13    continue
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 4+1$!].

      FUNCTION gammln(xx)
      REAL gammln,xx
      INTEGER j
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+log(stp*ser/x)
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 4+1$!].

      function polar(x,y)

      pi=3.1415927
      if(abs(x).gt.1.e-6)then
        phi=atan(y/x)
        if(x.lt.0.)phi=pi+phi
        if(phi.lt.0)phi=2*pi+phi
      else
        phi=0.5*pi
        if(y.lt.0)phi=phi+pi
      endif
      polar=phi

      end

      subroutine getJKNcentr
      end
