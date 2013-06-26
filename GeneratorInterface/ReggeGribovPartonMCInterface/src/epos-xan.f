c---------------------------------------------------------------------
      subroutine xiniall
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall

      parameter (mxfra=5)
      common/pfra/nfra,ifra(mxfra),ivfra(2,mxhis),itfra(mxtri,mxhis)
     $     ,imofra(3,mxfra),iffra(mxfra),r1fra(3,mxfra),r2fra(3,mxfra)
     $     ,emax(mxfra)
      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)

      nhis=0
      nfra=0
      imulty1=0
      ipairs1=0
      do n=0,mxxhis
        icorrtrig(n)=0
        ihardevent(n)=0
        ijetfind1(n)=0
        ijetfind2(n)=0
        ifastjet(n)=0
        ijetevent(n)=0
        imux(n)=0
        icaltrig(n)=0
      enddo
      do n=1,mxhis
        do m=1,mxpara
          xpara(m,n)=0
        enddo
      enddo
      ncontrall=0
      noerrall=0

      end

c---------------------------------------------------------------------
      subroutine xini
c---------------------------------------------------------------------
c  called after beginhisto
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall
      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)

      parameter (mxfra=5)
      common/pfra/nfra,ifra(mxfra),ivfra(2,mxhis),itfra(mxtri,mxhis)
     $     ,imofra(3,mxfra),iffra(mxfra),r1fra(3,mxfra),r2fra(3,mxfra)
     $     ,emax(mxfra)
      character line*1000,cvar*6
      logical go
      common/nl/noplin  /cnnnhis/nnnhis
      character*1000 cline
      common/cjjj/jjj,cline

      call utpri('xini  ',ish,ishini,5)

      i=1
                                !      iapl=0
                                !      nhis=0
      j=jjj     !-1
      line=cline
                                !      nfra=1
                                !      ifra(1)=iframe
      iapl=0
      if(nfra.eq.0)then
        nfra=1
        ifra(1)=iframe
      endif
      nhis=nhis+1
      nnnhis=nhis
      if(nhis.gt.mxhis)stop'xini: mxhis too small.       '
      noweak(nhis)=0
      ionoerr=0
c      newfra=0
      indfra=1
c      nepfra=0
      inpfra=1
 1    call utword(line,i,j,0)
      if(line(i:j).eq.'application')then !-----------
        call utword(line,i,j,1)
        if(line(i:j).eq.'analysis')then
                                !iapl=0
                                !nhis=nhis+1
                                !newfra=0
                                !indfra=1
                                !nepfra=0
                                !inpfra=1
        else
          iapl=1
        endif
      elseif(line(i:j).eq.'input')then !-----------
        call utword(line,i,j,0)
        if(nopen.ge.0)then
         nopen=nopen+1
         if(nopen.gt.9)stop'too many nested input commands'
         open(unit=20+nopen,file=line(i:j),status='old')
         if(iprmpt.eq.1)iprmpt=-1
        endif
      elseif(line(i:j).eq.'runprogram')then !-----------
        if(iapl.eq.0)then
        else
          goto 9999
        endif
      elseif(line(i:j).eq.'frame'.or.line(i:j).eq.'frame+')then !------
        ifp=1
        if(line(i:j).eq.'frame+')ifp=2
        call utword(line,i,j,1)
        if(line(i:j).eq.'total')then
          nfp=iframe
        elseif(line(i:j).eq.'nucleon-nucleon')then
          nfp=11
        elseif(line(i:j).eq.'target')then
          nfp=12
        elseif(line(i:j).eq.'gamma-nucleon')then
          nfp=21
        elseif(line(i:j).eq.'lab')then
          nfp=22
        elseif(line(i:j).eq.'breit')then
          nfp=23
        elseif(line(i:j).eq.'thrust')then
          nfp=33
        elseif(line(i:j).eq.'sphericity')then
          nfp=32
        else
          nfp=0
          call utstop("Wrong frame in xini !&")
        endif
        go=.true.
        inl=0
        do l=1,nfra
          if(ifra(l).eq.nfp)then
            inl=l
            go=.false.
          endif
        enddo
        if (go) then
          nfra=nfra+1
          inl=nfra
          ifra(nfra)=nfp
        endif
        if(ifp.eq.1)then
          indfra=inl
c          newfra=nfp
          ivfra(1,nhis)=indfra
          ivfra(2,nhis)=indfra
        else
          inpfra=inl
c          nepfra=nfp
        endif
      elseif(line(i:j).eq.'binning')then !-----------
        call utword(line,i,j,1)
        if(line(i:j).eq.'lin')then
          iologb=0
          iocnxb=0
        elseif(line(i:j).eq.'log')then
          iologb=1
          iocnxb=0
        elseif(line(i:j).eq.'clin')then
          iologb=0
          iocnxb=1
        elseif(line(i:j).eq.'clog')then
          iologb=1
          iocnxb=1
        else
          print *, 'what the heck is ',line(i:j),' binning?'
          print *, 'I will use the linear (lin) one'
        endif
      elseif(line(i:j).eq.'setm')then !-----------
        if(iapl.eq.0) then
          print *,"You should use histogram instead of setm, please"
          stop
        endif
      elseif(line(i:j).eq.'set')then !-----------
        call utword(line,i,j,1)
        if(line(i:j).eq.'iologb')then
          call utword(line,i,j,1)
          read(line(i:j),*) iologb
        elseif(line(i:j).eq.'iocnxb')then
          call utword(line,i,j,1)
          read(line(i:j),*) iocnxb
        elseif(line(i:j).eq.'etacut')then
          call utword(line,i,j,1)
          read(line(i:j),*) etacut
        elseif(line(i:j).eq.'nemsi')then
          call utword(line,i,j,1)
          read(line(i:j),*)nemsi
        endif
      elseif(line(i:j).eq.'xpara')then !-----------
        call utword(line,i,j,1)
        read(line(i:j),*)ipara
        if(ipara.gt.mxpara)stop'mxpara too small.         '
        call utword(line,i,j,1)
        read(line(i:j),*)val
        xpara(ipara,nhis)=val
      elseif(line(i:j).eq.'xparas')then !-----------
        call utword(line,i,j,1)
        read(line(i:j),*)ipara
        if(ipara.gt.mxpara)stop'mxpara too small.'
        do ii=1,ipara
        call utword(line,i,j,1)
        read(line(i:j),*)val
        xpara(ii,nhis)=val
        enddo
      elseif(line(i:j).eq.'echo')then !-----------
        call utword(line,i,j,1)
        if(line(i:j).eq.'on')iecho=1
        if(line(i:j).eq.'off')iecho=0
        if(line(i:j).ne.'on'.and.line(i:j).ne.'off')
     *  stop'invalid option'
      elseif(line(i:j).eq.'noweak')then !-----------
        noweak(nhis)=1
      elseif(line(i:j).eq.'histogram'
     *       .or.line(i:j).eq.'hi')then !-----------
        nac(nhis)=1
        call utword(line,i,j,1) !xvaria
        cvar='      '
        cvar=line(i:j)
        call xtrans(cvar,inom,ifrnew,nhis)
        if(inom.eq.-1)then
          if(line(i:i).ge.'0'.and.line(i:i).le.'9')then
            inom=298
            read(line(i:j),*) sval(1,nhis)
          endif
        endif
        ivar(1,nhis)=inom
        if(ifrnew.ne.0)then     !check frame for e+e- event
          go=.true.             !shape variables
          do l=1,nfra
            if(ifra(l).eq.ifrnew)then
              indfra=l
              go=.false.        !have it already
            endif
          enddo
          if (go) then
            nfra=nfra+1
            ifra(nfra)=ifrnew
            indfra=nfra
          endif
        endif
        call utword(line,i,j,1) !yvaria
        cvar='      '
        cvar=line(i:j)
        call xtrans(cvar,inom,ifrnew,nhis)
        ivar(2,nhis)=inom
        if(inom.eq.-1)then
          if(line(i:i).ge.'0'.and.line(i:i).le.'9')then
            inom=299
            read(line(i:j),*) sval(2,nhis)
          endif
        endif
        if(inom.eq.-1)ivar(1,nhis)=inom

        ivfra(1,nhis)=indfra
        ivfra(2,nhis)=indfra

        call utword(line,i,j,1) !normation
        read(line(i:j),*) inorm(nhis)

        call utword(line,i,j,1) !xmin
        if(line(i:j).eq.'egy')then
         if(engy.gt.0)then
          egy=engy
         elseif(ecms.gt.0.)then
          egy=ecms
         elseif(elab.gt.0)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*elab*atg+atg**2+apj**2 )
         elseif(ekin.gt.0.)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*(ekin+apj)*atg+atg**2+apj**2 )
         elseif(pnll.gt.0.)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*sqrt(pnll**2+apj**2)*atg+atg**2+apj**2 )
         else
          stop'pb in xini (1).   '
         endif
         xmin(nhis)=egy-0.5
        else
         read(line(i:j),*) xmin(nhis)
        endif

        call utword(line,i,j,1) !xmax
        if(line(i:j).eq.'egy')then
         if(engy.gt.0)then
          egy=engy
         elseif(ecms.gt.0.)then
          egy=ecms
         elseif(elab.gt.0)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*elab*atg+atg**2+apj**2 )
         elseif(ekin.gt.0.)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*(ekin+apj)*atg+atg**2+apj**2 )
         elseif(pnll.gt.0.)then
          call idmass(idproj,apj)
          call idmass(idtarg,atg)
          egy=sqrt( 2*sqrt(pnll**2+apj**2)*atg+atg**2+apj**2 )
         else
          stop'pb in xini (2).   '
         endif
         xmax(nhis)=egy+0.5
        else
         read(line(i:j),*) xmax(nhis)
        endif

        call utword(line,i,j,1) !nbin
        read(line(i:j),*) nbin(nhis)
        do l=1,nbin(nhis)
          bin(l,nac(nhis),nhis)=0.
          zcbin(l,nac(nhis),nhis)=0
        enddo
        lookcontr(nhis)=0
        lookcontrx(nhis)=0
        inoerr(nhis)=0
      elseif(line(i:j).eq.'idcode')then !-----------
        call utword(line,i,j,1) !idcode
        if(line(i:i+2).eq.'995')stop'xini: idcode 995 not supported'
        if(line(i:i+2).eq.'994')stop'xini: idcode 994 not supported'
        nidcod(nhis)=nidcod(nhis)+1
        read(line(i:j),*) idcod(nidcod(nhis),nhis)
        idmod(nidcod(nhis),nhis)=.false.
      elseif(line(i:j).eq.'idcode+')then !-----------
        stop'xini: idcode+ not supported'
        call utword(line,i,j,1) !idcode
        if(line(i:i+2).eq.'995')stop'xini: idcode 995 not supported'
        if(line(i:i+2).eq.'994')stop'xini: idcode 994 not supported'
        nidcod(nhis)=nidcod(nhis)+1
        read(line(i:j),*) idcod(nidcod(nhis),nhis)
        idmod(nidcod(nhis),nhis)=.true.
      elseif(line(i:j).eq.'trigger')then !-----------
        call utword(line,i,j,1)
        ntc=1
        imo=1
        ncontr=0
        icontrtyp(nhis)=0
        if(line(i:j).eq.'or'.or.line(i:j).eq.'contr')then
          imo=2
          if(line(i:j).eq.'contr')imo=3
          call utword(line,i,j,1)
          read(line(i:j),*)ztc
          ntc=nint(ztc)
          call utword(line,i,j,1)
          if(imo.eq.3)then
            ncontr=ntc
            ncontrall=ncontrall+ncontr
            if(ncontrall.gt.mxcontr)stop'xini: mxcontr too small.     '
            if(ncontr.gt.mxcnt)stop'xini: mxcnt too small.     '
            lookcontr(nhis)=ncontrall-ncontr+1
            lookcontrx(nhis)=ncontrall
            do nb=1,nbin(nhis)
              do nn=1,ncontr
                bbin(nb,nac(nhis),lookcontr(nhis)-1+nn)=0.d0
                zbbin(nb,nac(nhis),lookcontr(nhis)-1+nn)=0.d0
              enddo
            enddo
            do nn=1,ncontr
                    nccevt(lookcontr(nhis)-1+nn)=0
            enddo
          endif
        endif
        do n=1,ntc
          if(n.ne.1)call utword(line,i,j,1) !trigger-name
          cvar='      '
          ifp=1
          if(line(j:j).eq.'+')then
            cvar=line(i:j-1)
            ifp=2
          else
            cvar=line(i:j)
            ifp=1
          endif
          call xtrans(cvar,inom,ifrnew,nhis)
          if(inom.gt.0)then
            ntri(nhis)=ntri(nhis)+1
            if(ntc.eq.1)then
              ntrc(ntri(nhis),nhis)=1
            elseif(n.eq.1)then
              ntrc(ntri(nhis),nhis)=2
            elseif(n.eq.ntc)then
              ntrc(ntri(nhis),nhis)=3
            else
              ntrc(ntri(nhis),nhis)=0
            endif
            if(imo.eq.3)then
              ntrc(ntri(nhis),nhis)=-1
              if(n.eq.1)then
                icontrtyp(nhis)=1+inom/100
              else
                if(1+inom/100.ne.icontrtyp(nhis))
     *               stop'xini: type mismatch'
              endif
            endif
            itri(ntri(nhis),nhis)=inom
            if(ifp.eq.1)then
              itfra(ntri(nhis),nhis)=indfra
            else
              itfra(ntri(nhis),nhis)=inpfra
            endif
            xmitrp(ntri(nhis),nhis)=100.
            xmatrp(ntri(nhis),nhis)=100.
            call utword(line,i,j,1) !-----------xmin----------
            if(line(i:j).eq.'inf')then
              xmitri(ntri(nhis),nhis)=1e30
            elseif(line(i:j).eq.'-inf')then
              xmitri(ntri(nhis),nhis)=-1e30
            elseif(line(i:j).eq.'A')then
              xmitri(ntri(nhis),nhis)=maproj
            elseif(line(i:j).eq.'A+1')then
              xmitri(ntri(nhis),nhis)=maproj+1
            elseif(line(i:j).eq.'A+B')then
              xmitri(ntri(nhis),nhis)=maproj+matarg
            elseif(line(i:j).eq.'A+B+1')then
              xmitri(ntri(nhis),nhis)=maproj+matarg+1
            elseif(line(i:j).eq.'lead')then    !leading particle (neads Standard Variable)
              xmitri(ntri(nhis),nhis)=-123456
              imulty1=1
            elseif(line(i:j).eq.'jet')then    !jet from fastjet
              xmitri(ntri(nhis),nhis)=nhis*100
              iok=0
              do i=1,ifastjet(0)
                if(ifastjet(i).eq.nhis)iok=1
              enddo
              if(iok.eq.0)then
                ifastjet(0)=ifastjet(0)+1
                if(ifastjet(0).gt.mxxhis)stop'mxxhis too small'
                ifastjet(ifastjet(0))=nhis
              endif
            else
              kk=0
              do k=i+1,j-1
                if(line(k:k).eq.'%')kk=k
              enddo
              if(kk.eq.0)then
                read(line(i:j),*)xmitri(ntri(nhis),nhis)
              else
                read(line(i:kk-1),*)xmitrp(ntri(nhis),nhis)
                read(line(kk+1:j),*)xmitri(ntri(nhis),nhis)
              endif
            endif
            call utword(line,i,j,1) !-----------xmax------------
            if(line(i:j).eq.'inf')then
              xmatri(ntri(nhis),nhis)=1e30
            elseif(line(i:j).eq.'-inf')then
              xmatri(ntri(nhis),nhis)=-1e30
            elseif(line(i:j).eq.'A')then
              xmatri(ntri(nhis),nhis)=maproj
            elseif(line(i:j).eq.'A+1')then
              xmatri(ntri(nhis),nhis)=maproj+1
            elseif(line(i:j).eq.'A+B')then
              xmatri(ntri(nhis),nhis)=maproj+matarg
            elseif(line(i:j).eq.'A+B+1')then
              xmatri(ntri(nhis),nhis)=maproj+matarg+1
            elseif(line(i:j).eq.'lead')then    !leading particle (neads Standard Variable)
              xmatri(ntri(nhis),nhis)=-123456
              imulty1=1
            elseif(line(i:j).eq.'jet')then    !jet form fastjet
              xmatri(ntri(nhis),nhis)=nhis*100
            else
              kk=0
              do k=i+1,j-1
                if(line(k:k).eq.'%')kk=k
              enddo
              if(kk.eq.0)then
                read(line(i:j),*)xmatri(ntri(nhis),nhis)
                xmatrp(ntri(nhis),nhis)=100.
              else
                read(line(i:kk-1),*)xmatrp(ntri(nhis),nhis)
                read(line(kk+1:j),*)xmatri(ntri(nhis),nhis)
              endif
            endif
            !---exchange min-max------------------
            if(xmitri(ntri(nhis),nhis).gt.xmatri(ntri(nhis),nhis))then
              xmatri_save=xmatri(ntri(nhis),nhis)
              xmatrp_save=xmatrp(ntri(nhis),nhis)
              xmatri(ntri(nhis),nhis)=xmitri(ntri(nhis),nhis)
              xmatrp(ntri(nhis),nhis)=xmitrp(ntri(nhis),nhis)
              xmitri(ntri(nhis),nhis)=xmatri_save
              xmitrp(ntri(nhis),nhis)=xmatrp_save
            endif
            !-------------------------------------
          else
            ivar(1,nhis)=-1
            call utword(line,i,j,1) !xmin
            call utword(line,i,j,1) !xmax
          endif
        enddo
      elseif(line(i:j).eq.'noerrorbut')then !-----------
        ionoerr=ionoerr+1
        if(ionoerr.gt.2)stop'xini: not more than 2 noerrorbut !   '
        noerrall=noerrall+1
        if(noerrall.gt.mxhis/2)stop'xini: to many noerrorbut     '

        call utword(line,i,j,1) !variable-name
        cvar=line(i:j)
        call xtrans(cvar,inom,ifrnew,nhis)
        if(inom.gt.0)then
          if(inom.gt.100)then
            write(*,*)'xini: noerrorbut can not be used with :',cvar
            stop'xini: error with noerrorbut!'
          endif
          noerrhis(nhis)=noerrall-ionoerr+1
          noerr(noerrhis(nhis),ionoerr)=inom
          do nb=1,nbin(nhis)
             ebin(nb,nac(nhis),ionoerr-1+noerrhis(nhis))=0.d0
                  zebin(nb,nac(nhis),ionoerr-1+noerrhis(nhis))=0.d0
          enddo
        else
          ionoerr=ionoerr-1
          noerrall=noerrall-1
        endif
        inoerr(nhis)=ionoerr
      elseif(line(i:j).eq.'write')then !-----------
        call utword(line,i,j,1)
      elseif(line(i:j).eq.'writearray')then !-----------
        call utword(line,i,j,1)
        iologb=0
        iocnxb=0
      elseif(line(i:j).eq.'writehisto')then !-----------
        call utword(line,i,j,1)
        iologb=0
        iocnxb=0
      elseif(line(i:j).eq.'endhisto'
     .      .or.line(i:j).eq.'eh')then   !-----------
        ilog(nhis)=.false.
        icnx(nhis)=.false.
        if(iologb.eq.1)ilog(nhis)=.true.
        if(iocnxb.eq.1)icnx(nhis)=.true.
        if(ilog(nhis))then
          xinc(nhis)=1./log(xmax(nhis)/xmin(nhis))*nbin(nhis)
        else
          xinc(nhis)=float(nbin(nhis))/(xmax(nhis)-xmin(nhis))
        endif
        iologb=0
        iocnxb=0
        jjj=j
        cline=line
        goto 9999
      endif
      goto 1

 9999 continue
      if(ish.ge.5)then
        do n=1,nhis
          write (ifch,*) n,': ',ivar(1,n),ivar(2,n)
     $         ,'(',ivfra(1,n),ivfra(2,n)
     $         ,')',inorm(n)
     $         ,xmin(n),xmax(n),ilog(n),icnx(n)
     $         ,nbin(n),(idcod(j,n),j=1,nidcod(n))
     $         ,' tri:',ntri(n),(itri(j,n),j=1,ntri(n)),'('
     $         ,(itfra(j,n),j=1,ntri(n)),')'
     $         ,(xmitri(j,n),j=1,ntri(n)) ,(xmatri(j,n),j=1,ntri(n))
        enddo
        write (ifch,*) (ifra(j),j=1,nfra)
      endif
      call utprix('xini  ',ish,ishini,5)
      return
      end


c---------------------------------------------------------------------
      subroutine xana
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall

      parameter (mxfra=5)
      common/pfra/nfra,ifra(mxfra),ivfra(2,mxhis),itfra(mxtri,mxhis)
     $     ,imofra(3,mxfra),iffra(mxfra),r1fra(3,mxfra),r2fra(3,mxfra)
     $     ,emax(mxfra)
      double precision bofra
      common/dfra/bofra(5,mxfra)
      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj

      double precision pgampr,rgampr
      common/cgampr/pgampr(5),rgampr(4)

      common/photrans/phoele(4),ebeam

      dimension ten(4,3)
      logical go,goo(mxcnt),cont

      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)

      call utpri('xana  ',ish,ishini,4)

      if(ish.ge.2)then
          call alist('fill histograms&',0,0)
      endif

      do n=1,nhis
      do i=1,mypara
        ypara(i,n)=0
      enddo
      enddo


      if(ish.ge.5)write(ifch,*)'frames ...'

      if(iappl.eq.6)then
        if(mod(iolept/10,10).eq.1) call gakjet(1)
        if(mod(iolept/100,10).eq.1) call gakjet(2)
      endif

      do l=1,nfra
        emax(l)=egyevt/2
        if(ifra(l).eq.12)emax(l)=sqrt(pnll**2+prom**2)
        if(ifra(l).eq.iframe)then
          if(iappl.eq.1.and.iframe.eq.22)emax(l)=ebeam
          imofra(1,l)=0
          imofra(2,l)=0
          imofra(3,l)=0
        elseif(ifra(l).eq.11.or.ifra(l).eq.12)then
          imofra(1,l)=0
          imofra(2,l)=0
          bofra(1,l)=0d0
          bofra(2,l)=0d0
          bofra(3,l)=dsinh(dble(yhaha))
          bofra(4,l)=dcosh(dble(yhaha))
          bofra(5,l)=1d0
          if(ifra(l).eq.11.and.iframe.eq.12)then
            imofra(3,l)=1       ! target -> NN
          elseif(ifra(l).eq.12.and.iframe.eq.11)then
            imofra(3,l)=-1      ! NN -> target
          else
            imofra(3,l)=0       ! not known
          endif
        elseif(ifra(l).eq.21)then
          if(iframe.ne.21)then
            print *, 'invalid frame request'
            print *, 'choose frame gamma-nucleon for event run'
            stop'bye bye'
          endif
        elseif(ifra(l).eq.22)then
          if(iappl.eq.1)emax(l)=ebeam
          if(iframe.eq.21)then
            imofra(1,l)=-1      !'  trafo gN -> lab'
            imofra(1,l)=0
            r1fra(1,l)=rgampr(1)
            r1fra(2,l)=rgampr(2)
            r1fra(3,l)=rgampr(3)
            imofra(2,l)=0
            if(iappl.eq.1)then
              imofra(3,l)=-2
              bofra(1,l)=dsinh(dble(yhaha)) !used for first boost in targ frame
              bofra(2,l)=dcosh(dble(yhaha)) !here : pgampr(1)=pgampr(2)=0.
              bofra(3,l)=pgampr(3)
              bofra(4,l)=pgampr(4)
              bofra(5,l)=pgampr(5)
            else
              imofra(3,l)=-1
              bofra(1,l)=pgampr(1)
              bofra(2,l)=pgampr(2)
              bofra(3,l)=pgampr(3)
              bofra(4,l)=pgampr(4)
              bofra(5,l)=pgampr(5)
            endif
          elseif(iframe.eq.22)then
                                ! nothing to do already gN-frame
          else
            print *, 'invalid frame request'
            print *, 'choose frame gamma-nucleon or lab for event run'
            stop'bye bye'
          endif
        elseif(ifra(l).eq.23)then
          if(iframe.eq.21)then
            imofra(1,l)=0       ! gN -> breit-frame
            r1fra(1,l)=rgampr(1)
            r1fra(2,l)=rgampr(2)
            r1fra(3,l)=rgampr(3)
            imofra(2,l)=0
            imofra(3,l)=1
            bofra(1,l)=0d0
            bofra(2,l)=0d0
            bofra(3,l)=rgampr(4)
            bofra(4,l)=sqrt(rgampr(1)**2+rgampr(2)**2+rgampr(3)**2)
            bofra(5,l)=sqrt( bofra(4,l)**2-rgampr(4)**2)
          elseif(iframe.eq.23)then
                                ! nothing to do already breit-frame
          else
            print *, 'invalid frame request'
            print *, 'choose frame gamma-nucleon or lab for event run'
            stop'bye bye'
          endif
        elseif(ifra(l).eq.33.or.ifra(l).eq.36)then
          if(ifra(l).eq.33)then
            call gakthru(ten,2)
          else
            call gakthru(ten,3)
          endif
          if(ten(4,1).lt.0.)then
            imofra(1,l)=0
            imofra(2,l)=0
            imofra(3,l)=0
          else
            arox=ten(1,1)
            aroy=ten(2,1)
            aroz=ten(3,1)
            brox=ten(1,2)
            broy=ten(2,2)
            broz=ten(3,2)
            call utrota(1,arox,aroy,aroz,brox,broy,broz)
            imofra(1,l)=1
            r1fra(1,l)=arox
            r1fra(2,l)=aroy
            r1fra(3,l)=aroz
            imofra(2,l)=1
            r2fra(1,l)=brox
            r2fra(2,l)=broy
            r2fra(3,l)=broz
            imofra(3,l)=0       !no boost
          endif
          bofra(1,l)=dble(ten(4,1)) !usually this is for boosting
          bofra(2,l)=dble(ten(4,2)) !I abuse it to store the eigenvalues
          bofra(3,l)=dble(ten(4,3)) !
        elseif(ifra(l).eq.32.or.ifra(l).eq.34.or.ifra(l).eq.35)then
          if(ifra(l).eq.32)then
            call gaksphe(ten,2.,2)
          elseif(ifra(l).eq.34)then
            call gaksphe(ten,1.,2)
          else
            call gaksphe(ten,2.,3)
          endif
          if(ten(4,1).lt.0.)then
            imofra(1,l)=0
            imofra(2,l)=0
            imofra(3,l)=0
          else
            arox=ten(1,1)
            aroy=ten(2,1)
            aroz=ten(3,1)
            brox=ten(1,2)
            broy=ten(2,2)
            broz=ten(3,2)
            call utrota(1,arox,aroy,aroz,brox,broy,broz)
            imofra(1,l)=1
            r1fra(1,l)=arox
            r1fra(2,l)=aroy
            r1fra(3,l)=aroz
            imofra(2,l)=1
            r2fra(1,l)=brox
            r2fra(2,l)=broy
            r2fra(3,l)=broz
            imofra(3,l)=0
          endif
          bofra(1,l)=dble(ten(4,1))
          bofra(2,l)=dble(ten(4,2))
          bofra(3,l)=dble(ten(4,3))
        endif
      enddo

      do n=1,nhis
        itrevt(n)=.false.
        if(ivar(1,n).ge.100.and.ivar(1,n).le.199) sval(1,n)=0.
        if(ivar(2,n).ge.100.and.ivar(2,n).le.199) sval(2,n)=0.
        if(ivar(1,n).gt.300.and.ivar(1,n).lt.400)then
          call xval(n,ivar(1,n),ivfra(1,n),0,x) !initializing of  variables
        endif
        if(ivar(2,n).gt.300.and.ivar(2,n).lt.400)then
          call xval(n,ivar(2,n),ivfra(2,n),0,y) !
        endif
        do j=1,ntri(n)
          valtri(j,n)=0.
        enddo
        do j=1,nbin(n)  !copy bins
          bin(j,3-nac(n),n)=bin(j,nac(n),n)
          zcbin(j,3-nac(n),n)=zcbin(j,nac(n),n)
        enddo
        if(lookcontr(n).gt.0)then
          do j=1,nbin(n)
            do loo=lookcontr(n),lookcontrx(n)
                    bbin(j,3-nac(n),loo)=bbin(j,nac(n),loo)
                    zbbin(j,3-nac(n),loo)=zbbin(j,nac(n),loo)
            enddo
          enddo
        endif
        if(inoerr(n).gt.0)then
          do j=1,nbin(n)
            do nn=1,inoerr(n)
              ebin(j,3-nac(n),nn-1+noerrhis(n))=ebin(j,nac(n),
     &                                          nn-1+noerrhis(n))
              zebin(j,3-nac(n),nn-1+noerrhis(n))=zebin(j,nac(n),
     &                                          nn-1+noerrhis(n))
            enddo
          enddo
        endif
      enddo

      if(imulty1.eq.1)then
        if(ish.ge.5)write(ifch,*)'Calculate standard variables ...'
        call StandardVariables
      endif
      if(ipairs1.eq.1)then
        if(ish.ge.5)write(ifch,*)'Calculate pair variables ...'
        call PairVariables
      endif
      if(ish.ge.5)write(ifch,*)'Call corrtrig ...'
      do n=1,icorrtrig(0)
        call corrtrig(icorrtrig(n))
      enddo
      if(ish.ge.5)write(ifch,*)'Call hardevent ...'
      do n=1,ihardevent(0)
        call hardevent(ihardevent(n))
      enddo
      if(ish.ge.5)write(ifch,*)'Call mux ...'
      do n=1,imux(0)
        call mux(imux(n))
      enddo
      if(ish.ge.5)write(ifch,*)'Call caltrig ...'
      do n=1,icaltrig(0)
        call caltrig(icaltrig(n))
      enddo
      if(ish.ge.5)write(ifch,*)'Call jetfind ...'
      do n=1,ijetfind1(0)
        call jetfind(1,ijetfind1(n))
      enddo
      do n=1,ijetfind2(0)
        call jetfind(2,ijetfind2(n))
      enddo
      if(ish.ge.5)write(ifch,*)'Call fastjet ...'
      do n=1,ifastjet(0)
        call fastjet(ifastjet(n))
      enddo
c don't change order here : jetevent should be called always after fastjet !
      if(ish.ge.5)write(ifch,*)'Call jetevent ...'
      do n=1,ijetevent(0)
        call jetevent(ijetevent(n))
      enddo


c...........................loop nptl...................................
      ncontr=0
      if(ish.ge.5)write(ifch,*)'Loop nptl ...'
      do j=1,nptl
        if(iorptl(j).lt.0.or.(istptl(j).lt.100.and.istptl(j).gt.istmax))
     &                                                            goto 8
        if(ish.ge.5)write(ifch,*)'ptl :',j
        call idchrg(idptl(j),ch)
        do i=1,nfra
          iffra(i)=0            !flag if frame calculated or not
        enddo
        do n=1,nhis
          if(ivar(1,n).eq.-1.or.ivar(2,n).eq.-1)goto 9
          if(ivar(1,n).ge.200.and.ivar(2,n).ge.200)goto 9 !skip particle loop if event variables

c...........check ids
          go=nidcod(n).eq.0
          do i=1,nidcod(n)
            if(istptl(j).eq.0.and.idcod(i,n).eq.10000)then    !all final particle
              go=.true.     
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9995)then    !all particles but nuclei
              if(abs(idptl(j)).lt.10000) go=.true.     
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9990)then  !all hadrons
              if((abs(idptl(j)).ge.100.or.abs(idptl(j)).eq.20)
     $         .and.abs(idptl(j)).lt.10000) go=.true.      
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9985)then !neutral particles
              if(abs(ch).lt.0.1.and.abs(idptl(j)).lt.10000) go=.true. 
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9980)then !charged particles
              if(abs(ch).gt.0.1.and.abs(idptl(j)).lt.10000) go=.true. 
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9975)then  !neutral hadrons
              if(abs(ch).lt.0.1.and.(abs(idptl(j)).ge.100
     $ .or.abs(idptl(j)).eq.20).and.abs(idptl(j)).lt.10000) go=.true.
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9970)then  !charged hadrons
              if(abs(ch).gt.0.1.and.abs(idptl(j)).ge.100
     $         .and.abs(idptl(j)).lt.10000) go=.true.
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.-9960)then !negative hadrons
              if(ch.lt.-0.1.and.abs(idptl(j)).ge.100
     $             .and.abs(idptl(j)).lt.10000)go=.true.
            elseif(istptl(j).eq.0.and.idcod(i,n).eq.9960)then  !positive hadrons
              if(ch.gt.0.1.and.abs(idptl(j)).ge.100
     $         .and.abs(idptl(j)).lt.10000)go=.true.
            elseif((istptl(j).le.1.or.istptl(j).ge.10)
     $            .and.ityptl(n).ne.61
     $            .and.idcod(i,n).eq.idptl(j))then
              go=.true.
            elseif(istptl(j).gt.100.and.idcod(i,n).eq.9999)then   !jets from fastjet
              go=.true.
            endif
          enddo
          if(ish.ge.10)write(ifch,*)j,' id,ist',idptl(j),istptl(j),go

c...........check weak decay
          if(go)then
            if(noweak(n).eq.1)then  !do not consider weak decay products
             if(iorptl(j).ne.0)then
              idora=abs( idptl(iorptl(j)) )
              if(  idora.eq.20   .or.idora.eq.2130
     &               .or.idora.eq.2230 .or.idora.eq.1130
     &         .or.idora.eq.2330 .or.idora.eq.1330
     &         .or.idora.eq.3331 )go=.false.
             ! print *, j,n, '   ', idptl(j),idora,go
             endif
            endif
          endif

c...........check triggers
          if(go)then
            if(ish.ge.7)write(ifch,*)'  check triggers in histogram ',n
            ncontr=0
            do i=1,ntri(n)
              if(ish.ge.7)write(ifch,*)'  trigger variable: ',itri(i,n)
              if(itri(i,n).lt.100)then
                call xval(n,itri(i,n),itfra(i,n),j,x)
                if(ntrc(i,n).ne.-1)then
                  call triggercondition(i,n,x,go)
                else
                  ncontr=ncontr+1
                  goo(ncontr)=.true.
                  call triggercondition(i,n,x,goo(ncontr))
                  if((ivar(1,n).gt.100.and.ivar(1,n).lt.200)
     .                 .or.(ivar(2,n).gt.100.and.ivar(2,n).lt.200))then
                    print*,'!-----------------------------------------'
                    print*,'!  100-199 event variables can not be used'
                    print*,'! in connection with "trigger contr ..."  '
                    print*,'!-----------------------------------------'
                    stop'in xana (1).                      '
                  endif
                endif
              elseif(itri(i,n).lt.200)then
                if(ntrc(i,n).eq.-1)then
                    print*,'!-----------------------------------------'
                    print*,'!  100-199 event variables can not be used'
                    print*,'! in connection with "trigger contr ..."  '
                    print*,'!-----------------------------------------'
                    stop'in xana (2).                      '
                endif
                call xval(n,itri(i,n),itfra(i,n),j,x)
                valtri(i,n)=valtri(i,n)+x
              endif
            enddo
          endif

c............fill histogram
          if(go)then
            if(ish.ge.6)write(ifch,*)'  fill histogram '
     &            ,n,ivar(1,n),ivar(2,n),ivfra(2,n)
            cont=.false.
            if(ivar(1,n).lt.100.or.ivar(2,n).lt.100)then
              if(ivar(2,n).lt.100)then
                call xval(n,ivar(2,n),ivfra(2,n),j,y)
                sval(2,n)=y
                cont=.true.
              endif
              if(ivar(1,n).lt.100)then
                call xval(n,ivar(1,n),ivfra(1,n),j,x)
                if(x.ge.xmin(n).and.x.le.xmax(n))then
                  norm3=mod(inorm(n)/100,10)
                  if(norm3.eq.1)then
                    y=y*x
                  elseif(norm3.eq.2.and.ivar(1,n).eq.63.and.x.ne.0.)then
                    y=y/(x+pptl(5,j))/2/pi
                  elseif(norm3.eq.2.and.ivar(1,n).ne.63.and.x.ne.0.)then
                    y=y/x/2/pi
                  elseif(norm3.eq.4.and.x.ne.0.)then
                    y=y/x**1.5
                  elseif(norm3.eq.5.and.x.ne.0.)then
                    y=y/x
                  elseif(norm3.eq.7.and.x.ne.0.)then
                    y=y/x/sqrt(x-pptl(5,j))
                  endif
                  if(icnx(n))then
                    call fillhistoconex(n,x,y,ivfra(2,n),j)   !for conex
                  else
                    if(ilog(n))then
                      nb=1+int(log(x/xmin(n))*xinc(n))
                    else
                      nb=1+int((x-xmin(n))*xinc(n))
                    endif
                    bin(nb,nac(n),n)=bin(nb,nac(n),n)+y
                   if(ncontr.gt.0)then  !ptl trigger contr
                      do nn=1,ncontr
                        if(goo(nn))then
                             bbin(nb,nac(n),lookcontr(n)-1+nn)=
     &                  bbin(nb,nac(n),lookcontr(n)-1+nn)+y
                             zbbin(nb,nac(n),lookcontr(n)-1+nn)=
     &                  zbbin(nb,nac(n),lookcontr(n)-1+nn)+1
                        endif
                      enddo
                    endif
                    if(inoerr(n).gt.0)then
                      do nn=1,inoerr(n)
                       call xval(n,noerr(noerrhis(n),nn),ivfra(2,n),j,y)
                        ebin(nb,nac(n),nn-1+noerrhis(n))=
     &                       ebin(nb,nac(n),nn-1+noerrhis(n))+y
                        zebin(nb,nac(n),nn-1+noerrhis(n))=
     &                       zebin(nb,nac(n),nn-1+noerrhis(n))+1
                      enddo
                    endif
                    zcbin(nb,nac(n),n)=zcbin(nb,nac(n),n)+1
                  endif
                  itrevt(n)=.true.
                endif
              endif
            endif
            if(ivar(1,n).gt.100.and.ivar(1,n).lt.200)then
              call xval(n,ivar(1,n),ivfra(1,n),j,x)
              sval(1,n)=sval(1,n)+x
            endif
            if(ivar(2,n).gt.100.and.ivar(2,n).lt.200)then
              call xval(n,ivar(2,n),ivfra(2,n),j,y)
              sval(2,n)=sval(2,n)+y
              cont=.true.
            endif
            if(ivar(1,n).gt.300.and.ivar(1,n).lt.400)then
              call xval(n,ivar(1,n),ivfra(1,n),j,x)
            endif
            if(ivar(2,n).gt.300.and.ivar(2,n).lt.400)then
              call xval(n,ivar(2,n),ivfra(2,n),j,y)
              cont=.true.
            endif
            if(ish.ge.6.and.cont)write (ifch,*)
     *                           '   ---> histo n,x,y:',n,x,y
          endif
   9      continue
        enddo
  8     continue
      enddo
c...........................end loop nptl...........................

      do n=1,nhis
      if(ivar(1,n).eq.-1.or.ivar(2,n).eq.-1)goto 99

c........check event triggers

      if(ish.ge.7)write(ifch,*)'  check event triggers in histogram ',n
       go=.true.
        ncontr=0
        do i=1,ntri(n)
          if(itri(i,n).gt.100)then
            if(itri(i,n).lt.200)then
              x=valtri(i,n)
            else
              call xval(n,itri(i,n),itfra(i,n),0,x)
            endif
            if(ntrc(i,n).ne.-1)then
              call triggercondition(i,n,x,go)
            else
              ncontr=ncontr+1
              goo(ncontr)=.true.
              call triggercondition(i,n,x,goo(ncontr))
            endif
          endif
        enddo

c........event variables > 200

        if(go)then
          if(ivar(1,n).gt.100)then
            if(ivar(1,n).gt.200.and.ivar(1,n).lt.300)then
              call xval(n,ivar(1,n),ivfra(1,n),0,x)
            elseif(ivar(1,n).gt.300.and.ivar(1,n).lt.400)then
              call xval(n,ivar(1,n),ivfra(1,n),nptl+1,x)
            elseif(ivar(1,n).gt.100.and.ivar(1,n).lt.200)then
              x=sval(1,n)
            else
              call xval(n,ivar(1,n),ivfra(1,n),0,x)
            endif
            if(ivar(2,n).gt.200.and.ivar(2,n).lt.300)then
              call xval(n,ivar(2,n),ivfra(2,n),0,y)
            elseif(ivar(2,n).gt.300.and.ivar(2,n).lt.400)then
              call xval(n,ivar(2,n),ivfra(2,n),nptl+1,y)
            elseif(ivar(2,n).gt.0.and.ivar(2,n).lt.200)then
              y=sval(2,n)
            else             !inom>500
              call xval(n,ivar(2,n),ivfra(2,n),0,y)
            endif
c The following doesn't work for ivar(2,n)<100, since particle number is not defined !
c            if(ivar(2,n).gt.200.and.ivar(2,n).lt.300)then
c              call xval(n,ivar(2,n),ivfra(2,n),0,y)
c            elseif(ivar(2,n).gt.300.and.ivar(2,n).lt.400)then
c              call xval(n,ivar(2,n),ivfra(2,n),nptl+1,y)
c            elseif(ivar(2,n).gt.100.and.ivar(2,n).lt.200)then
c              y=sval(2,n)
c            else
c              call xval(n,ivar(2,n),ivfra(2,n),0,y)
c            endif
            if(mod(inorm(n)/100,10).eq.1)y=y*x
            if(mod(inorm(n)/100,10).eq.2.and.x.ne.0.)y=y/x/2/pi
            if(mod(inorm(n)/100,10).eq.4.and.x.ne.0.)y=y/x**1.5
            if(mod(inorm(n)/100,10).eq.5.and.x.ne.0.)y=y/x
            sval(1,n)=x
            sval(2,n)=y
            if(ish.ge.6)write (ifch,*) 'histo n,x,y:',n,x,y
            if(x.ge.xmin(n).and.x.le.xmax(n))then
              if(ilog(n))then
                nb=1+int(log(x/xmin(n))*xinc(n))
              else
                nb=1+int((x-xmin(n))*xinc(n))
              endif
              bin(nb,nac(n),n)=bin(nb,nac(n),n)+y
              if(ncontr.gt.0)then
                do nn=1,ncontr
                  if(goo(nn))
     &                    bbin(nb,nac(n),lookcontr(n)-1+nn)=
     &             bbin(nb,nac(n),lookcontr(n)-1+nn)+y
                enddo
              endif
              zcbin(nb,nac(n),n)=zcbin(nb,nac(n),n)+1
              itrevt(n)=.true.
            endif
          endif
        endif

c........particle variables

        if(go)then
          if(ivar(1,n).le.100)then
            if(ncontr.gt.0)then  !event trigger contr
              do nb=1,nbin(n)
                do nn=1,ncontr
                  if(goo(nn))
     &                     bbin(nb,nac(n),lookcontr(n)-1+nn)=
     &              bbin(nb,nac(n),lookcontr(n)-1+nn)
     &              +bin(nb,nac(n),n)-bin(nb,3-nac(n),n)
                enddo
              enddo
            endif
          endif
        endif

c............event ok (increase ncevt) or not (take copy)

        if(go)then
          ncevt(n)=ncevt(n)+1
          if(ncontr.gt.0)then
            do nn=1,ncontr
              loo=lookcontr(n)-1+nn
              if(goo(nn))
     &        nccevt(loo)=nccevt(loo)+1
            enddo
          endif
        else
          if(ish.ge.6)write (ifch,*) 'event rejected for histo',n
          nac(n)=3-nac(n)
          itrevt(n)=.false.
        endif

 99   continue
      enddo

      call utprix('xana  ',ish,ishini,4)
      end

c--------------------------------------------------------------------
      subroutine triggercondition(i,n,x,go)
c--------------------------------------------------------------------
c ntrc is used to distinguish the different usage of trigger:
c
c    trigger var xmin xmax
c             ntrc=1
c    trigger or n var1 xmin1 xmax1 var2 xmin2 xmax2 ... varn xminn xmaxn
c          1  ntrc=2
c          2  ntrc=0
c              ...
c         n-1 ntrc=0
c          n  ntrc=3
c    trigger contr n var1 xmin1 xmax1 var2 xmin2 xmax2 ... varn xminn xmaxn
c             ntrc=-1
c--------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall
      logical go,gox,ok,goz
                xmn=xmitri(i,n)
                xmx=xmatri(i,n)
                if(xmn.eq.-123456.and.xmx.eq.-123456)then  !for leading part
                  xmn=float(idlead)
                  xmx=float(idlead)
                endif
                pmn=xmitrp(i,n)
                pmx=xmatrp(i,n)
                if(abs(ntrc(i,n)).eq.1)then
                  goz=.true.
                  if(pmn.gt.99.999.and.pmx.gt.99.999)then
                    if(x.lt.xmn.or.x.gt.xmx)goz=.false.
                  else
                    if(x.lt.xmn-0.5.or.x.gt.xmx+0.5)goz=.false.
                    ok=rangen().le.xmitrp(i,n)/100.
                    if(.not.ok.and.x.lt.xmn+0.5)goz=.false.
                    ok=rangen().le.xmatrp(i,n)/100.
                    if(.not.ok.and.x.gt.xmx-0.5)goz=.false.
                  endif
                  if(.not.goz)go=.false.
                else
                  if(ntrc(i,n).eq.2)gox=.false.
                  goz=.true.
                  if(pmn.gt.99.999.and.pmx.gt.99.999)then
                    if(x.lt.xmn.or.x.gt.xmx)goz=.false.
                  else
                    if(x.lt.xmn-0.5.or.x.gt.xmx+0.5)goz=.false.
                    ok=rangen().le.xmitrp(i,n)/100.
                    if(.not.ok.and.x.lt.xmn+0.5)goz=.false.
                    ok=rangen().le.xmatrp(i,n)/100.
                    if(.not.ok.and.x.gt.xmx-0.5)goz=.false.
                  endif
                  if(goz)gox=.true.
                  if(ntrc(i,n).eq.3.and..not.gox)go=.false.
                endif
                if(ish.ge.9)write(ifch,*)'trigger conditions '
     &                                   ,i,n,xmn,x,xmx,go
                end

c-----------------------------------------------------------------------
      subroutine fillhistoconex(n,x,y,lf,j)   !for conex
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall

           if(.not.(mod(inorm(n),10).ne.4
     &      .and.mod(inorm(n),10).ne.6
     &      .and.mod(inorm(n)/100,10).ne.3))return
                    if(ilog(n))then
                      c=(xmax(n)/xmin(n))**(1./real(nbin(n)))
                      nde=nint(1./log10(c))
                      nb=max(1,1+int(log10(x/xmin(n))*nde))
                      xmb=xmin(n)*c**(nb-0.5)
                      if(x.gt.xmb.and.nb.lt.nbin(n))then
                        if(x.gt.xmax(n))
     &                    write(ifmt,*)'xana max ?',x,xmax(n),nb
                        nbx=1
                        xmx=c*xmb
                      elseif(x.lt.xmb.and.nb.gt.1)then
                        if(x.lt.xmin(n))write(ifmt,*)'xana min ?',x,nb
                        nbx=-1
                        xmx=xmb/c
                      else
                        nbx=0
                        xmx=0.
                      endif
                    else
                      c=(xmax(n)-xmin(n))/real(nbin(n))
                      nb=max(1,1+int((x-xmin(n))/c))
                      xmb=xmin(n)+c*(nb-0.5)
                      if(x.gt.xmb)then
                        nbx=1
                        xmx=c+xmb
                      elseif(x.lt.xmb)then
                        nbx=-1
                        xmx=xmb-c
                      else
                        nbx=0
                        xmx=0.
                      endif
                    endif
                    xc=(x-xmx)/(xmb-xmx)
                    xc=max(0.,min(1.,xc))
                    bin(nb,nac(n),n)=bin(nb,nac(n),n)+xc*y
                    if(nbx.ne.0)bin(nb+nbx,nac(n),n)
     &                  =bin(nb+nbx,nac(n),n)+(1.-xc)*y
                    zcbin(nb,nac(n),n)=zcbin(nb,nac(n),n)+1
                    if(inoerr(n).gt.0)then
                      do nn=1,inoerr(n)
                        call xval(n,noerr(noerrhis(n),nn),lf,j,y2)
                        ebin(nb,nac(n),nn-1+noerrhis(n))=
     &                       ebin(nb,nac(n),nn-1+noerrhis(n))+y2
                        zebin(nb,nac(n),nn-1+noerrhis(n))=
     &                       zebin(nb,nac(n),nn-1+noerrhis(n))+1
                      enddo
                    endif
      end

c---------------------------------------------------------------------
      subroutine xhis(n)
c---------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall
      dimension xx(mxbin)

      double precision histoweight
      common/chiswei/histoweight
      common/cyield/yield
      common/csigma/sigma
      double precision dcel
      common/ems3/dcel,ad
      common/geom/rmproj,rmtarg,bmax,bkmx
      save cnormx

      if(ivar(1,n).eq.-1)then
        nrbins=0
        goto 9999
      endif

c.......here normalization.......................................
c           see also   "..........fill histogram"
c.................................................................
c     the norm ( inorm(n) ) is a number hijk which normalizes to:
c
c  k  0:  * 1
c     1:  / number of events
c     2:  / number of triggered events
c     4:  / bin-counts
c     5:  / bin sum
c     6:  / number of summed bin-counts (yield=1.)
c     7:  uses same normalization as one histo before
c
c  j  0:  * 1
c     1:  / bin-width
c     2:  * sigma_total / bin-width
c     3:  * sigma_diff / bin-width
c
c  i  0:  * 1
c     1:  y => y*x
c     2:  y => y/x/2/pi (modified for mt0)
c     3:  kno-scaling
c     4:  y => y/x**1.5
c     5:  y => y/x
c     6:  y => y*xi (for conex, xi=x of the bin)
c     7:  y => y/x/(x-m)
c
c  h  0: normal
c     1: accumulated
c
c.................................................................

      norm1=mod(inorm(n),10)
      norm2=mod(inorm(n)/10,10)
      norm3=mod(inorm(n)/100,10)
      norm4=mod(inorm(n)/1000,10)
      nctbin=0
      sumbin=0
      do l=1,nbin(n)
        nctbin=nctbin+zcbin(l,nac(n),n)
        sumbin=sumbin+bin(l,nac(n),n)
        if(norm1.eq.4.and.zcbin(l,nac(n),n).ne.0d0)then
          bin(l,nac(n),n)=bin(l,nac(n),n)/zcbin(l,nac(n),n)
          if(lookcontr(n).gt.0)then
            do loo=lookcontr(n),lookcontrx(n)
              if(zbbin(l,nac(n),loo).ne.0.)
     &           bbin(l,nac(n),loo)=bbin(l,nac(n),loo)
     &            /zbbin(l,nac(n),loo)
            enddo
          endif
        endif
        if(ilog(n))then
          xx(l)=xmin(n)*(xmax(n)/xmin(n))**((float(l)-.5)/nbin(n))
        else
          xx(l)=(float(l)-0.5)*(xmax(n)-xmin(n))/nbin(n)+xmin(n)
        endif
      enddo
      cnorm=1.
      if(norm1.eq.1)cnorm=1./float(nevent)
      if(norm1.eq.2)then
        if(ncevt(n).ne.0)then
          cnorm=1./float(ncevt(n))
        else
          cnorm=0.
        endif
      endif
      if(norm1.eq.5.and.sumbin.ne.0.)cnorm=1./sumbin
      if(norm1.eq.6.and.nctbin.ne.0)cnorm=1./float(nctbin)
      if(norm1.eq.7)cnorm=cnormx
      cnormx=cnorm
      if(ntevt.ne.0)
     &   sigma=10.*pi*bmax**2.*nevent/ntevt !total (untriggered) sigma
      if(norm2.eq.3)then      !differential (triggered) sigma
        if(ntevt.ne.0)
     &     sigma=10.*pi*bmax**2.*ncevt(n)/ntevt
      endif
      if(norm3.eq.3)then      !kno
        first=0.
        secnd=0.
        do l=1,nbin(n)
          if(nctbin.ne.0)first=first+xx(l)*zcbin(l,nac(n),n)/nctbin
          if(nctbin.ne.0)secnd=secnd
     $           +xx(l)**2*zcbin(l,nac(n),n)/nctbin
        enddo
      else
        first=1.
      endif
      if(ilog(n))then
        if(norm2.eq.2.or.norm2.eq.3) cnorm=cnorm*sigma
      else
        if(norm2.ge.1.and.norm2.le.3) cnorm=cnorm*xinc(n)
        if(norm2.eq.2.or.norm2.eq.3) cnorm=cnorm*sigma
      endif
      do l=1,nbin(n)
        bnorm=0.
        if(ilog(n).and.norm2.ge.1.and.norm2.le.3)then
          bnorm=1./(xmin(n)*exp(float(l)/xinc(n))*(1.-exp(-1./xinc(n))))
          bin(l,nac(n),n) =  bin(l,nac(n),n) * bnorm
        endif
        bin(l,nac(n),n) =  bin(l,nac(n),n) * cnorm
        if(lookcontr(n).gt.0)then
          if(ilog(n).and.norm2.ge.1.and.norm2.le.3)then
            do loo=lookcontr(n),lookcontrx(n)
              bbin(l,nac(n),loo)=bbin(l,nac(n),loo) * bnorm
            enddo
          endif
        endif
      enddo
      f=first
      nrbins=nbin(n)
      nctbin=0
      yield=0.
      shft=0
       if(nint(xpara(1,n)).eq.999963)shft=xpara(2,n)
      do ii=1,nbin(n)
        g=1
        if(norm3.eq.1.and.xx(ii).ne.0.)g=1./xx(ii)
        if(norm3.eq.2)g=2*pi*(xx(ii)+shft)
        if(norm3.eq.4)g=xx(ii)**1.5
        if(norm3.eq.5)g=xx(ii)
        if(norm3.eq.7)g=0
        yield=yield+bin(ii,nac(n),n)/xinc(n)*hisfac*f*g
      enddo
      do l=1,nbin(n)
        x=(xx(l)+xshift)      !*xhfact
        ar(l,1)=x/f
        sigbin=0
        if(zcbin(l,nac(n),n).ne.0d0)
     *   sigbin=bin(l,nac(n),n)*hisfac*f/sqrt(zcbin(l,nac(n),n))
        if(norm4.eq.0.or.l.eq.1)then
          ar(l,3)=bin(l,nac(n),n)*hisfac*f
          if(lookcontr(n).gt.0)then
           do loo=lookcontr(n),lookcontrx(n)
             r=1
             if(norm1.eq.2.and.nccevt(loo).ne.0.)
     *          r=float(ncevt(n))/nccevt(loo)
             lo=loo-lookcontr(n)+1
             ary(l,lo)=bbin(l,nac(n),loo)*hisfac*f*cnorm*r
             if(zbbin(l,nac(n),loo).gt.0.)then
               ardy(l,lo)=ary(l,lo)/sqrt(zbbin(l,nac(n),loo))
             else
               ardy(l,lo)=0
             endif
             if(norm1.eq.4)ardy(l,lo)=zbbin(l,nac(n),loo)
            enddo
          endif
          if(norm3.eq.6)then   !conex
           ar(l,3)=ar(l,3)*xx(l)
          endif
          ar(l,4)=sigbin
        else
          ar(l,3)=ar(l-1,3)+bin(l,nac(n),n)*hisfac*f
          ar(l,4)=sqrt(ar(l-1,4)**2+sigbin**2)
        endif
        if(inoerr(n).ge.1)then
          if(zebin(l,nac(n),noerrhis(n)).gt.0.d0)then
         ar(l,4)=ebin(l,nac(n),noerrhis(n))/zebin(l,nac(n),noerrhis(n))
          else
            ar(l,4)=0.
          endif
        endif
        if(inoerr(n).eq.2)then
          if(zebin(l,nac(n),noerrhis(n)+1).gt.0.d0)then
      ar(l,5)=ebin(l,nac(n),noerrhis(n)+1)/zebin(l,nac(n),noerrhis(n)+1)
          else
            ar(l,5)=0.
          endif
        endif
        if(norm1.eq.4)ar(l,4)=zcbin(l,nac(n),n)
      enddo
      ionoerr=inoerr(n)
      histoweight=dble(ncevt(n))
      if(norm1.eq.1)histoweight=dble(nevent)
      if(norm1.eq.4)histoweight=0d0

 9999 hisfac=1.
      xshift=0
      end

c-----------------------------------------------------------------------
      integer function nsdiff(insdif,now)
c-----------------------------------------------------------------------
c  returns  1 if trigger condition for NSD fulfilled and 0 otherwise
c  for  UA1 (insdif=0) or UA5 (insdif=1) or CDF (insdif=2) or STAR (insdif=3,4)
C  or BRAHMS (insdif=5) or NA61 (insdif=6) or CMS (insdif=7)
c  or ATLAS (insdif=8) or ALICE (insdif=9, 10 and 11) 
c  or CMS hadron level (insdif=12) or CMS hadron level double sided (insdif=13)
c  now ... noweak(histogram number) (obsolete)
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer ncevt,nsdi(0:20)
      logical cont
      data ncevt/1/
      save nsdi,ncevt
c initialization for each event
      if(ncevt.eq.nrevt)then
        ncevt=ncevt+1
        do i=0,20
          nsdi(i)=-1
        enddo
      endif
      nsdiff=0
           if(insdif.ge.0)then
         if(nsdi(insdif).lt.0)then
      iii1=0
      iii2=0
      iii3=0
      ipos=0
      ineg=0
         do npts=1,nptl
        if(istptl(npts).ne.0)goto 60
          cont=   idptl(npts).ne.120 .and.idptl(npts).ne.-120
     *   .and.idptl(npts).ne.130 .and.idptl(npts).ne.-130
     *   .and.idptl(npts).ne.1120.and.idptl(npts).ne.-1120
     *   .and.idptl(npts).ne.1130.and.idptl(npts).ne.-1130
     *   .and.idptl(npts).ne.2230.and.idptl(npts).ne.-2230
     *   .and.idptl(npts).ne.2330.and.idptl(npts).ne.-2330
     *   .and.idptl(npts).ne.3331.and.idptl(npts).ne.-3331
        if(insdif.ne.7.and.insdif.ne.14)then
          if(cont)goto 60
        endif
c just to avoid warning with gfortran when "now" is not used
        nowdum=now
c        if(now.eq.1)then  !do not consider weak decay products
c         if(iorptl(npts).ne.0)then
c          idora=abs( idptl(iorptl(npts)) )
c          if(  idora.eq.20   .or.idora.eq.2130
c     &     .or.idora.eq.2230 .or.idora.eq.1130
c     &     .or.idora.eq.2330 .or.idora.eq.1330
c     &     .or.idora.eq.3331 )goto 60
c          endif
c         endif
        pz=pptl(3,npts)
        pt=sqrt(pptl(2,npts)**2+pptl(1,npts)**2)
        ppp=sqrt(pz**2+pt**2)
        Etot=pptl(4,npts)
        if(ppp.gt.abs(pz))then
          yyy=.5*log((ppp+pz)/(ppp-pz))
        else
          yyy=sign(100.,pz)
        endif
        if(insdif.eq.0)then
          if(yyy.gt.1.5  .and. yyy.lt.5.5)iii1=1
          if(yyy.gt.-5.5 .and. yyy.lt.-1.5)iii2=1
        elseif(insdif.eq.1)then
          if(yyy.gt.2.   .and. yyy.lt.5.6)iii1=1
          if(yyy.gt.-5.6 .and. yyy.lt.-2.)iii2=1
        elseif(insdif.eq.2)then
          if(yyy.gt.3.2  .and. yyy.lt.5.9)iii1=1
          if(yyy.gt.-5.9 .and. yyy.lt.-3.2)iii2=1
          if(yyy.gt.0.   .and. yyy.lt.3.0)ipos=ipos+1
          if(yyy.gt.-3.0 .and. yyy.lt.0. )ineg=ineg+1
        elseif(insdif.eq.3)then
          if(yyy.gt.-5.0 .and. yyy.lt.-3.3 )iii1=1
          if(yyy.gt. 3.3 .and. yyy.lt. 5.0 )iii2=1
        elseif(insdif.eq.4)then
          if(yyy.gt.-5.0 .and. yyy.lt.-3.1 )iii1=1
          if(yyy.gt. 3.1 .and. yyy.lt. 5.0 )iii2=1
        elseif(insdif.eq.5)then
          if(yyy.gt.-5.25 .and. yyy.lt.-3.26 )iii1=1
          if(yyy.gt. 3.26 .and. yyy.lt. 5.25 )iii2=1
        elseif(insdif.eq.6)then  !NA61 trigger if NO charged particle with theta<5.26 mrad
          if(pptl(3,npts).gt.0..and.yyy.lt.100.)then
           theta=sqrt(pptl(1,npts)**2+pptl(2,npts)**2)/pptl(3,npts)
           if(theta.lt.5.26e-3)iii1=1
          endif
        elseif(insdif.eq.7)then   !CMS NSD corrected using PYTHIA (2010)
          if(yyy.gt.-5.2 .and. yyy.lt.-2.9 .and. Etot .gt.3.)iii1=1
          if(yyy.gt. 2.9 .and. yyy.lt. 5.2 .and. Etot .gt.3.)iii2=1
          if(yyy.gt. -2.5 .and. yyy.lt. 2.5 .and. pt .gt.0.2 .and. cont)
     &                                                       iii3=1
        elseif(insdif.eq.8)then   !ATLAS
          if(yyy.gt.-2.5 .and. yyy.lt.2.5.and.pt.gt.0.5)iii1=1
          iii2=1
        elseif(insdif.eq.9)then   !ALICE 900 GeV
          if(yyy.gt.-3.7 .and. yyy.lt.-1.7 )iii1=1
          if(yyy.gt. 2.8 .and. yyy.lt. 5.1 )iii2=1
        elseif(insdif.eq.10)then   !ALICE 2.36 TeV
          if(yyy.gt.-2 .and. yyy.lt.2 )iii1=1
          iii2=1
        elseif(insdif.eq.11)then   !ALICE Inel>0
          if(yyy.gt.-1 .and. yyy.lt.1 )iii1=1
          iii2=1
        elseif(insdif.eq.12)then   !CMS hadron level NSD trigger (2011)
          if(yyy.gt.-4.4 .and. yyy.lt.-3.9 )iii1=1
          if(yyy.gt. 3.9 .and. yyy.lt. 4.4 )iii2=1
        elseif(insdif.eq.13)then   !CMS hadron level doubl sided trigger (2012)
          if(yyy.gt.-5. .and. yyy.lt.-3. .and. Etot .gt. 3. )iii1=1
          if(yyy.gt. 3. .and. yyy.lt. 5. .and. Etot .gt. 3. )iii2=1
        elseif(insdif.eq.14)then   !CMS hadron level single sided trigger (HF 2012)
          if(yyy.gt.-4.9 .and. yyy.lt.-2.9 .and. Etot .gt. 5. )iii1=1
          if(yyy.gt. 2.9 .and. yyy.lt. 4.9 .and. Etot .gt. 5. )iii2=1
        endif
60      continue
         enddo
        if(insdif.le.1)then
          if(iii1.eq.1 .and. iii2.eq.1) nsdiff=1
        elseif(insdif.eq.2)then
          if((iii1.eq.1 .and. iii2.eq.1) .and.
     *    ((ipos.ne.0 .and. ineg.ne.0) .and. ipos+ineg.ge.4)) nsdiff=1
        elseif(insdif.eq.14)then
          if(iii1.eq.1 .or. iii2.eq.1) nsdiff=1
        elseif(insdif.eq.3.or.insdif.eq.4
     *         .or.insdif.eq.5.or.insdif.eq.8.or.insdif.ge.9)then
          if(iii1.eq.1 .and. iii2.eq.1) nsdiff=1
        elseif(insdif.eq.6)then
          if(iii1.eq.0 .and. iii2.eq.0)then
            nsdiff=1
          endif
        elseif(insdif.eq.7)then
          if(iii1.eq.1 .and. iii2.eq.1 .and.iii3.eq.1)then
            nsdiff=1
          endif
        endif
        nsdi(insdif)=nsdiff
        else
          nsdiff=nsdi(insdif)
        endif
           else
      stop'in nsdiff. argument of nsdiff not authorized.        '
           endif
      end

c-----------------------------------------------------------------------
      integer function isdiff(isdif)
c-----------------------------------------------------------------------
c  returns  1 if trigger condition for single diff fulfilled and 0 otherwise
c  for  UA4 Mult distri (isdif=1) or UA4 xsection (isdif=2) 
c  or CDF SD (isdif=3) or CDF DPE (isdif=4) or CDF min bias (for DD) 
c  (isdif=5)
c-----------------------------------------------------------------------
      include 'epos.inc'
      isdiff=0
           if(isdif.ge.1)then
      iii0=0
      iii1=0
      iii2=0
      iii3=0
      iii4=0
      Et1=0.
      Et2=0.
      do npts=1,nptl
        if(istptl(npts).ne.0)goto 60
        if(   abs(idptl(npts)).ne.120
     *   .and.abs(idptl(npts)).ne.130
     *   .and.abs(idptl(npts)).ne.1120
     *   .and.abs(idptl(npts)).ne.1130
     *   .and.abs(idptl(npts)).ne.2230
     *   .and.abs(idptl(npts)).ne.2330
     *   .and.abs(idptl(npts)).ne.3331)goto 60
        ppt=pptl(1,npts)**2+pptl(2,npts)**2
        ppp=sqrt(ppt+pptl(3,npts)**2)
        ppt=sqrt(ppt)
        yyy=0.
        if(pptl(3,npts).ne.0..and.ppt.ne.0.)yyy=sign(1.,pptl(3,npts))*
     *   log((ppp+abs(pptl(3,npts)))/ppt)
c        if(ppp.gt.abs(pptl(3,npts)))then
c          yyy=.5*log((ppp+pptl(3,npts))/(ppp-pptl(3,npts)))
c        else
c          yyy=sign(100.,pptl(3,npts))
c        endif
        if(isdif.le.2)yyy=-sign(1.,float(ilprtg))*yyy   !trigger on antiproton (target : ilprt=-1)
c        if(idptl(npts).eq.-1120)then
          if(abs(pptl(3,npts)).gt.0.)then
            theta=sign(1.,float(ilprtg))*ppt/pptl(3,npts)
            if((isdif.le.2.and.theta.gt.2.5e-3.and.theta.lt.4.5e-3).or.
     *         (isdif.gt.2.and.theta.gt.0.2e-3.and.theta.lt.1.2e-3))then
              iii0=iii0+1
c              write(ifch,*)'la',
c              print *,'la',ilprtg,yyy,iii1,iii2
c     &      ,npts,idptl(npts),ppt,pptl(3,npts),theta,ityptl(npts)
            endif
c          endif
        endif
        if(isdif.eq.1)then
          if(yyy.gt.2.5   .and. yyy.lt.5.6)iii1=1
        elseif(isdif.eq.2)then
          if(yyy.gt.3.  .and. yyy.lt.5.6)iii1=1
          if(yyy.gt.-5.6 .and. yyy.lt.-4.4 )iii2=1
        elseif(isdif.eq.3)then
          if(yyy.gt.2.4  .and. yyy.lt.5.9)iii1=1
          if(yyy.gt.-4.2 .and. yyy.lt.1.1)iii3=1
          if(yyy.gt.-5.9 .and. yyy.lt.-2.4)iii2=1
          if(yyy.gt.-1.1 .and. yyy.lt.4.2)iii4=1
        elseif(isdif.eq.4)then
          if(ilprtg.eq.-1)then  !antiproton = target
            if(yyy.gt.2.4  .and. yyy.lt.5.9)iii1=1
            if(yyy.gt.-5.9 .and. yyy.lt.-3.2)iii2=iii2+1
          else                  !antiproton = projectile
            if(yyy.gt.-5.9 .and. yyy.lt.-2.4)iii1=1
            if(yyy.gt. 3.2 .and. yyy.lt.5.9)iii2=iii2+1
          endif
        elseif(isdif.eq.5)then
          if(yyy.gt.3.2  .and. yyy.lt.5.9)iii1=1
          if(yyy.gt.-5.9 .and. yyy.lt.-3.2)iii2=1
          if(abs(yyy).lt.2.4)Et1=Et1+ppt
          if(Et1.gt.0.2)iii3=1
          if(abs(yyy).gt.2.2 .and. abs(yyy).lt.4.2 )Et2=Et2+ppt
          if(Et2.gt.1.)iii4=1
        endif
60      continue
      enddo
      if(isdif.eq.1)then
        if(iii0.eq.1 .and. iii1.eq.1) isdiff=1
      elseif(isdif.eq.2)then
        if(iii0.eq.1 .and. iii1.eq.1 .and. iii2.ne.1) isdiff=1
      elseif(isdif.eq.3)then
        if( (iii1.ne.1 .and. iii3.eq.1) .or.
     &      (iii2.ne.1 .and. iii4.eq.1)       ) isdiff=1
      elseif(isdif.eq.4)then
        if(iii0.eq.1 .and. iii1.ne.1 .and. iii2.le.6) isdiff=1
c        if(isdiff.eq.1)then
c      do npts=1,nptl
c        if(istptl(npts).ne.0)goto 80
c        if(   abs(idptl(npts)).ne.120
c     *   .and.abs(idptl(npts)).ne.130
c     *   .and.abs(idptl(npts)).ne.1120
c     *   .and.abs(idptl(npts)).ne.1130
c     *   .and.abs(idptl(npts)).ne.2230
c     *   .and.abs(idptl(npts)).ne.2330
c     *   .and.abs(idptl(npts)).ne.3331)goto 80
c        ppt=pptl(1,npts)**2+pptl(2,npts)**2
c        ppp=sqrt(ppt+pptl(3,npts)**2)
c        ppt=sqrt(ppt)
c        yyy=0.
c        if(pptl(3,npts).ne.0..and.ppt.ne.0.)yyy=sign(1.,pptl(3,npts))*
c     *   log((ppp+abs(pptl(3,npts)))/ppt)
c        print *,nrevt,yyy,idptl(npts),ityptl(npts)
c80      continue
c        enddo
c          print *,'dpe',iii0
c        endif
      elseif(isdif.eq.5)then
        if(iii1+iii2+iii3+iii4.eq.4) isdiff=1
      else
        stop'in sdiff. argument of sdiff not authorized.        '
      endif
             endif
      end

c----------------------------------------------------------------------
      subroutine xtrans(cvar,inom,ifr,n)
c----------------------------------------------------------------------
      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)

      character*6 cvar
      ifr=0
      if(cvar.eq.'numptl')then
        inom=1
      elseif(cvar.eq.'npaptl')then
        inom=2
      elseif(cvar.eq.'npmptl')then
        inom=3
      elseif(cvar.eq.'ispptl')then
        inom=4
      elseif(cvar.eq.'rapx')then
        inom=5
      elseif(cvar.eq.'iptlfr')then
        inom=6
      elseif(cvar.eq.'rinp')then
        inom=7
      elseif(cvar.eq.'eco')then
        inom=8
      elseif(cvar.eq.'tau')then
        inom=9
      elseif(cvar.eq.'ctr')then
        inom=10
      elseif(cvar.eq.'v2np')then
        inom=11
        imulty1=1             !to switch on the calculation of "Standard variable"
      elseif(cvar.eq.'absrap')then
        inom=12
      elseif(cvar.eq.'rap')then
        inom=13
      elseif(cvar.eq.'xp')then
        inom=14
      elseif(cvar.eq.'xe')then
        inom=15
      elseif(cvar.eq.'pt')then
        inom=16
      elseif(cvar.eq.'p1a')then
        inom=17
      elseif(cvar.eq.'p2a')then
        inom=18
      elseif(cvar.eq.'xi')then
        inom=19
      elseif(cvar.eq.'xf')then
        inom=20
      elseif(cvar.eq.'t')then
        inom=21
      elseif(cvar.eq.'rapmi')then
        inom=22
      elseif(cvar.eq.'eta')then
        inom=23
      elseif(cvar.eq.'theta')then
        inom=24
      elseif(cvar.eq.'pt2')then
        inom=25
      elseif(cvar.eq.'et')then
        inom=26
      elseif(cvar.eq.'idptl')then
        inom=27
      elseif(cvar.eq.'istptl')then
        inom=28
      elseif(cvar.eq.'mass')then
        inom=29
      elseif(cvar.eq.'idaptl')then
        inom=30
      elseif(cvar.eq.'egy')then
        inom=31
      elseif(cvar.eq.'rapwro')then
        inom=32
      elseif(cvar.eq.'mt')then
        inom=33
      elseif(cvar.eq.'pplus')then
        inom=34
      elseif(cvar.eq.'pminus')then
        inom=35
      elseif(cvar.eq.'p5')then
        inom=36
      elseif(cvar.eq.'pa')then
        inom=37
      elseif(cvar.eq.'sob')then
        inom=38
      elseif(cvar.eq.'idpom')then
        inom=39
      elseif(cvar.eq.'p3a')then
        inom=40
      elseif(cvar.eq.'cmass')then
        inom=41
      elseif(cvar.eq.'arappi')then
        inom=42
      elseif(cvar.eq.'itsptl')then
        inom=50
      elseif(cvar.eq.'ityptl')then
        inom=51
      elseif(cvar.eq.'idoptl')then
        inom=52
      elseif(cvar.eq.'iptl')then
        inom=53
      elseif(cvar.eq.'index')then
        inom=54
      elseif(cvar.eq.'p1')then
        inom=55
      elseif(cvar.eq.'p2')then
        inom=56
      elseif(cvar.eq.'p3')then
        inom=57
      elseif(cvar.eq.'p4')then
        inom=58
      elseif(cvar.eq.'xg')then
        inom=59
      elseif(cvar.eq.'ek')then
        inom=60
      elseif(cvar.eq.'beta')then
        inom=61
      elseif(cvar.eq.'mt0')then
        inom=63
      elseif(cvar.eq.'qsqptl')then
        inom=64
      elseif(cvar.eq.'xelab')then
        inom=65
      elseif(cvar.eq.'hgtc05')then
        inom=66
        imulty1=1             !to switch on the calculation of "Standard variable"
      elseif(cvar.eq.'hadtyp')then
        inom=67
        imulty1=1
      elseif(cvar.eq.'hgtc1')then
        inom=68
        imulty1=1
      elseif(cvar.eq.'x4')then
        inom=69
      elseif(cvar.eq.'npn')then
        inom=70
      elseif(cvar.eq.'routp')then
        inom=71
      elseif(cvar.eq.'hgtc3')then
        inom=72
        imulty1=1
      elseif(cvar.eq.'mu14')then
        inom=73
        imulty1=1
      elseif(cvar.eq.'delphi')then
        inom=74
        iok=0
        !------------------------------------------------------------
        !icorrtrig stores the histogram numbers of those histograms which
        !use the delphi variable (and therfore require a call corrtrig
        !------------------------------------------------------------
        do i=1,icorrtrig(0)
         if(icorrtrig(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          icorrtrig(0)=icorrtrig(0)+1
          if(icorrtrig(0).gt.mxxhis)stop'mxxhis too small'
        icorrtrig(icorrtrig(0))=n
        endif
      elseif(cvar.eq.'v2')then
        inom=75
      elseif(cvar.eq.'pt4')then
        inom=76
      elseif(cvar.eq.'rin')then
        inom=77
      elseif(cvar.eq.'theh1p')then
        inom=78
      elseif(cvar.eq.'theh1t')then
        inom=79
      elseif(cvar.eq.'phi')then
        inom=80
      elseif(cvar.eq.'isoft')then
        inom=81
      elseif(cvar.eq.'mux')then
        inom=82
        imux(0)=imux(0)+1
        imux(imux(0))=n
      elseif(cvar.eq.'v4')then
        inom=83
      elseif(cvar.eq.'x3')then
        inom=84
      elseif(cvar.eq.'jorptl')then
        inom=85
      elseif(cvar.eq.'ptlead')then
        inom=86
        iok=0
        !------------------------------------------------------------
        !icorrtrig stores the histogram numbers of those histograms which
        !use the ptlead variable (and therfore require a call corrtrig
        !------------------------------------------------------------
        do i=1,icorrtrig(0)
         if(icorrtrig(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          icorrtrig(0)=icorrtrig(0)+1
          if(icorrtrig(0).gt.mxxhis)stop'mxxhis too small'
        icorrtrig(icorrtrig(0))=n
        endif
      elseif(cvar.eq.'mu25')then
        inom=87
        imulty1=1
      elseif(cvar.eq.'pai')then
        inom=88
        ipairs1=1
      elseif(cvar.eq.'co2')then
        inom=89
        ipairs1=1
      elseif(cvar.eq.'co3')then
        inom=90
        ipairs1=1
      elseif(cvar.eq.'rad')then
        inom=91
      elseif(cvar.eq.'abseta')then
        inom=92
      elseif(cvar.eq.'phiexp')then
        inom=93
      elseif(cvar.eq.'mu24')then
        inom=94
        imulty1=1
      elseif(cvar.eq.'mulevt')then
        inom=101
      elseif(cvar.eq.'etevt')then
        inom=102
      elseif(cvar.eq.'enevt')then
        inom=103
      elseif(cvar.eq.'ev6evt')then
        inom=104
      elseif(cvar.eq.'xenevt')then
        inom=105
      elseif(cvar.eq.'netevt')then
        inom=106
      elseif(cvar.eq.'ptevt')then
        inom=107
      elseif(cvar.eq.'pmxevt')then
        inom=108
      elseif(cvar.eq.'numevt')then
        inom=201
      elseif(cvar.eq.'egyevt')then
        inom=202
      elseif(cvar.eq.'bimevt')then
        inom=203
      elseif(cvar.eq.'xbjevt')then
        inom=204
      elseif(cvar.eq.'qsqevt')then
        inom=205
      elseif(cvar.eq.'yevt')then
        inom=206
      elseif(cvar.eq.'eloevt')then
        inom=207
      elseif(cvar.eq.'nd1evt')then
        inom=208
      elseif(cvar.eq.'nd2evt')then
        inom=209
      elseif(cvar.eq.'theevt')then
        inom=210
      elseif(cvar.eq.'nspevt')then
        inom=211
      elseif(cvar.eq.'nhpevt')then
        inom=212
      elseif(cvar.eq.'sigtot')then
        inom=213
      elseif(cvar.eq.'sigela')then
        inom=214
      elseif(cvar.eq.'sloela')then
        inom=215
      elseif(cvar.eq.'nrgevt')then
        inom=216
      elseif(cvar.eq.'qevt')then
        inom=217
      elseif(cvar.eq.'qtlevt')then
        inom=218
      elseif(cvar.eq.'nd0evt')then
        inom=219
      elseif(cvar.eq.'threvt')then
        inom=220
        ifr=33                  !set thrust-frame
      elseif(cvar.eq.'omtevt')then
        inom=221
        ifr=33                  !set thrust-frame
      elseif(cvar.eq.'tmaevt')then
        inom=222
        ifr=33                  !set thrust-frame
      elseif(cvar.eq.'tmievt')then
        inom=223
        ifr=33                  !set thrust-frame
      elseif(cvar.eq.'oblevt')then
        inom=224
        ifr=33                  !set thrust-frame
      elseif(cvar.eq.'sphevt')then
        inom=230
        ifr=32                  !set sph-frame
      elseif(cvar.eq.'aplevt')then
        inom=231
        ifr=32                  !set sph-frame
      elseif(cvar.eq.'cpaevt')then
        inom=232
        ifr=34                  !set sph2-frame
      elseif(cvar.eq.'dpaevt')then
        inom=233
        ifr=34                  !set sph2-frame
      elseif(cvar.eq.'npoevt')then
        inom=234
      elseif(cvar.eq.'npnevt')then
        inom=235
      elseif(cvar.eq.'ikoevt')then
        inom=236
      elseif(cvar.eq.'iktevt')then
        inom=237
      elseif(cvar.eq.'npxevt')then
        inom=238
      elseif(cvar.eq.'nd6evt')then
        inom=239
      elseif(cvar.eq.'mu1evt')then
        inom=240
        imulty1=1
      elseif(cvar.eq.'muievt')then
        inom=241
        imulty1=1
      elseif(cvar.eq.'hgtevt')then
        inom=242
        imulty1=1
      elseif(cvar.eq.'difevt')then
        inom=243
      elseif(cvar.eq.'dixevt')then
        inom=244
      elseif(cvar.eq.'nd7evt')then
        inom=245
      elseif(cvar.eq.'nd8evt')then
        inom=246
      elseif(cvar.eq.'nd9evt')then
        inom=247
      elseif(cvar.eq.'ndaevt')then
        inom=248
      elseif(cvar.eq.'ndbevt')then
        inom=249
      elseif(cvar.eq.'qinevt')then
        inom=250
      elseif(cvar.eq.'qfievt')then
        inom=251
      elseif(cvar.eq.'einevt')then
        inom=252
      elseif(cvar.eq.'efievt')then
        inom=253
      elseif(cvar.eq.'pinevt')then
        inom=254
      elseif(cvar.eq.'pfievt')then
        inom=255
      elseif(cvar.eq.'pxfevt')then    ! leading proton xf in cms
        inom=256
      elseif(cvar.eq.'pi+xf')then     ! pi+xf: pi+ yield at cms xf>0.01
        inom=257
      elseif(cvar.eq.'pi-xf')then     ! pi-xf: pi- yield at cms xf>0.01
        inom=258
      elseif(cvar.eq.'sigcut')then
        inom=260
      elseif(cvar.eq.'keu')then
        inom=261
      elseif(cvar.eq.'ked')then
        inom=262
      elseif(cvar.eq.'kes')then
        inom=263
      elseif(cvar.eq.'kolevt')then
        inom=265
      elseif(cvar.eq.'sigsd')then
        inom=266
      elseif(cvar.eq.'nglevt')then
        inom=267
      elseif(cvar.eq.'kppevt')then   ! collision numbers per participant
        inom=268
      elseif(cvar.eq.'npievt')then   ! pion + multiplicity per event
        inom=269
      elseif(cvar.eq.'np2evt')then   ! pion + multiplicity per participant
        inom=270
      elseif(cvar.eq.'sigdif'.or.cvar.eq.'sigdifr')then
        inom=271
      elseif(cvar.eq.'koievt')then
        inom=272
      elseif(cvar.eq.'ineevt')then
        inom=273
      elseif(cvar.eq.'elaevt')then
        inom=274
      elseif(cvar.eq.'itgevt')then
        inom=275
        iok=0
        do i=1,icorrtrig(0)
          if(icorrtrig(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          icorrtrig(0)=icorrtrig(0)+1
          if(icorrtrig(0).gt.mxxhis)stop'mxxhis too small'
          icorrtrig(icorrtrig(0))=n
        endif
      elseif(cvar.eq.'hrdevt')then
        inom=276
        iok=0
        do i=1,ihardevent(0)
          if(ihardevent(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          ihardevent(0)=ihardevent(0)+1
          if(ihardevent(0).gt.mxxhis)stop'mxxhis too small'
          ihardevent(ihardevent(0))=n
        endif
      elseif(cvar(2:6).eq.'j1evt'.or.cvar(2:6).eq.'j2evt')then
        iok=0
        do i=1,ijetfind1(0)
          if(ijetfind1(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          ijetfind1(0)=ijetfind1(0)+1
          if(ijetfind1(0).gt.mxxhis)stop'mxxhis too small'
          ijetfind1(ijetfind1(0))=n
        endif
        if(cvar.eq.'ej1evt')inom=277
        if(cvar.eq.'pj1evt')inom=278
        if(cvar(2:6).eq.'j2evt')then
          iok=0
          do i=1,ijetfind2(0)
            if(ijetfind2(i).eq.n)iok=1
          enddo
          if(iok.eq.0)then
            ijetfind2(0)=ijetfind2(0)+1
            if(ijetfind2(0).gt.mxxhis)stop'mxxhis too small'
            ijetfind2(ijetfind2(0))=n
          endif
          if(cvar.eq.'ej2evt')inom=279
          if(cvar.eq.'pj2evt')inom=280
        endif
      elseif(cvar.eq.'zppevt')then
        inom=281
      elseif(cvar.eq.'zptevt')then
        inom=282
      elseif(cvar.eq.'***not used***')then
        inom=283
      elseif(cvar.eq.'nd3evt')then
        inom=284
      elseif(cvar.eq.'nd4evt')then
        inom=285
      elseif(cvar.eq.'mubevt')then
        inom=286
        imulty1=1
      elseif(cvar.eq.'nd5evt')then
        inom=287
      elseif(cvar.eq.'ekievt')then
        inom=288
      elseif(cvar.eq.'sd1evt')then
        inom=289
      elseif(cvar.eq.'sd2evt')then
        inom=290
      elseif(cvar.eq.'mdevt')then
        inom=291
        imulty1=1     !to switch on the calculation of "Standard variable"
      elseif(cvar.eq.'m2devt')then
        inom=292
        imulty1=1
      elseif(cvar.eq.'tdevt')then
        inom=293
        imulty1=1
      elseif(cvar.eq.'ndpevt')then
        inom=294
      elseif(cvar.eq.'rapgap')then
        inom=295
        imulty1=1
      elseif(cvar.eq.'ng1evt')then
        inom=296
      elseif(cvar.eq.'r21evt')then
        inom=297
      elseif(cvar.eq.'aimevt')then
        inom=301
      elseif(cvar.eq.'wjbevt')then
        inom=302
      elseif(cvar.eq.'njbevt')then
        inom=303
      elseif(cvar.eq.'djbevt')then
        inom=304
      elseif(cvar.eq.'tjbevt')then
        inom=305
      elseif(cvar.eq.'hjmevt')then
        inom=306
      elseif(cvar.eq.'ljmevt')then
        inom=307
      elseif(cvar.eq.'djmevt')then
        inom=308
      elseif(cvar.eq.'ybal')then
        inom=310
      elseif(cvar.eq.'yabal')then
        inom=310
      elseif(cvar.eq.'sigine')then
        inom=312
      elseif(cvar.eq.'sigiaa')then
        inom=313
      elseif(cvar.eq.'alpdsf')then
        inom=314
      elseif(cvar.eq.'alpdsh')then
        inom=315
      elseif(cvar.eq.'betdsf')then
        inom=316
      elseif(cvar.eq.'betdsh')then
        inom=317
      elseif(cvar.eq.'rexdip')then
        inom=318
      elseif(cvar.eq.'rexdit')then
        inom=319
      elseif(cvar.eq.'m14evt')then
        inom=320
        imulty1=1
      elseif(cvar.eq.'ht3evt')then
        inom=321
      elseif(cvar.eq.'sigiex')then
        inom=322
      elseif(cvar.eq.'sigdex')then
        inom=323
      elseif(cvar.eq.'sigsex')then
        inom=324
      elseif(cvar.eq.'ekievt')then
        inom=325
      elseif(cvar.eq.'sigcaa')then
        inom=326
      elseif(cvar.eq.'sigtaa')then
        inom=327
      elseif(cvar.eq.'xkappa')then
        inom=328
      elseif(cvar.eq.'gamdsf')then
        inom=329
      elseif(cvar.eq.'gamdsh')then
        inom=330
      elseif(cvar.eq.'deldsf')then
        inom=331
      elseif(cvar.eq.'deldsh')then
        inom=332
      elseif(cvar.eq.'nd6evt')then
        inom=333
      elseif(cvar.eq.'muxevt')then
        inom=334
        imux(0)=imux(0)+1
        imux(imux(0))=n
      elseif(cvar.eq.'typevt')then
        inom=335
      elseif(cvar.eq.'m25evt')then
        inom=339
        imulty1=1
      elseif(cvar.eq.'segevt')then
        inom=340
      elseif(cvar.eq.'ielevt')then
        inom=341
      elseif(cvar.eq.'mc1evt')then
        inom=342
        imulty1=1
      elseif(cvar.eq.'sdcdf')then
        inom=343
      elseif(cvar.eq.'dpecdf')then
        inom=344
      elseif(cvar.eq.'ddcdf')then
        inom=345
      elseif(cvar.eq.'phievt')then
        inom=346
      elseif(cvar.eq.'ndcevt')then
        inom=347
      elseif(cvar.eq.'jetevt')then
        inom=348
c activate jetevent trigger
        iok=0
        do i=1,ijetevent(0)
          if(ijetevent(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          ijetevent(0)=ijetevent(0)+1
          if(ijetevent(0).gt.mxxhis)stop'mxxhis too small'
          ijetevent(ijetevent(0))=n
        endif
c activate fastjet
        iok=0
        do i=1,ifastjet(0)
          if(ifastjet(i).eq.n)iok=1
        enddo
        if(iok.eq.0)then
          ifastjet(0)=ifastjet(0)+1
          if(ifastjet(0).gt.mxxhis)stop'mxxhis too small'
          ifastjet(ifastjet(0))=n
        endif
      elseif(cvar.eq.'epszer')then
        inom=349
      elseif(cvar.eq.'xsievt')then
        inom=350
        imulty1=1
      elseif(cvar.eq.'xsicms')then
        inom=351
      elseif(cvar.eq.'calevt')then
        inom=352
        icaltrig(0)=icaltrig(0)+1
        icaltrig(icaltrig(0))=n
      elseif(cvar.eq.'fgpevt')then
        inom=353
        icaltrig(0)=icaltrig(0)+1
        icaltrig(icaltrig(0))=n
      elseif(cvar.eq.'bgpevt')then
        inom=354
        icaltrig(0)=icaltrig(0)+1
        icaltrig(icaltrig(0))=n
      elseif(cvar.eq.'gapevt')then
        inom=355
        icaltrig(0)=icaltrig(0)+1
        icaltrig(icaltrig(0))=n
      elseif(cvar.eq.'sigdd')then
        inom=356
      elseif(cvar.eq.'styevt')then
        inom=357
      elseif(cvar.eq.'ndsevt')then
        inom=358
      elseif(cvar.eq.'m24evt')then
        inom=359
        imulty1=1
      elseif(cvar.eq.'ndhevt')then
        inom=360
      elseif(cvar.eq.'ox1evt')then
        inom=501
      elseif(cvar.eq.'ox2evt')then
        inom=502
      elseif(cvar.eq.'ox3evt')then
        inom=503
      elseif(cvar.eq.'ox4evt')then
        inom=504
      elseif(cvar.eq.'eglevt')then  ! eccentricity
        inom=505
      elseif(cvar.eq.'fglevt')then  ! eccentricity_part
        inom=506
      elseif(cvar.eq.'rglevt')then  ! ratio ng2 / ng1
        inom=507
      elseif(cvar.eq.'sglevt')then  ! area S
        inom=508
      elseif(cvar.eq.'ptrevt')then
        inom=509
      elseif(cvar.eq.'rr2evt')then
        inom=510
        imulty1=1             !to switch on the calculation of "Standard variable"
      elseif(cvar.eq.'perevt')then
        inom=511
      elseif(cvar.eq.'paievt')then
        inom=512
        ipairs1=1
      elseif(cvar.eq.'co2evt')then
        inom=513
        ipairs1=1
      elseif(cvar.eq.'co3evt')then
        inom=514
        ipairs1=1
      else
        print *,' '
        print *,'              xtrans: unknown variable ',cvar
        print *,' '
c       inom=-1
        stop
      endif
      end

c----------------------------------------------------------------------
      subroutine xval(n,inom,lf,j,x)
c----------------------------------------------------------------------
c   n ...... histogram index
c   inom ... variable index
c              1-100 particle variables
c              101-200 accumulative event variables
c              > 200 other event variables
c   lf ..... frame index
c   particle index (used for particle variables)
c----------------------------------------------------------------------
      include 'epos.inc'
      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)
      common/zeus2/qtl

      parameter (ntim=1000)
      common/cprt/pprt(5,ntim),q2prt(ntim),idaprt(2,ntim),idprt(ntim)
     &,iorprt(ntim),jorprt(ntim),nprtj

      common/cxyzt/xptl(mxptl),yptl(mxptl),zptl(mxptl),tptl(mxptl)
     *,optl(mxptl),uptl(mxptl),sptl(mxptl),rptl(mxptl,3)
      common/cpairs/paievt,co2evt,co3evt

      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      double precision ebin,zebin
      common/errbins/ebin(mxbin,2,mxhis/2),zebin(mxbin,2,mxhis/2),
     $inoerr(mxhis),noerr(mxhis/2,2),noerrhis(mxhis/2),noerrall
      parameter (mxfra=5)
      common/pfra/nfra,ifra(mxfra),ivfra(2,mxhis),itfra(mxtri,mxhis)
     $     ,imofra(3,mxfra),iffra(mxfra),r1fra(3,mxfra),r2fra(3,mxfra)
     $     ,emax(mxfra)
      common/cphi2/phi2pos,phi2neg
      common/cen/ncentr  
      parameter(nbkbin=100)
      common /kfitd/ xkappafit(nclegy,nclha,nclha,nbkbin),xkappa,bkbin
      common/cpairs2/paipi(40),co2pi(40),co3pi(40)
     .              ,paipr(40),co2pr(40),co3pr(40)
     .            ,ipaipi(40),ico2pi(40),ico3pi(40)
     .            ,ipaipr(40),ico2pr(40),ico3pr(40)
     .            ,maxpt,delpt

      double precision bofra,bofra1,bofra2,bofra3,bofra4,bofra5,xd
      common/dfra/bofra(5,mxfra)
      dimension p(5,mxfra),aimuni(10,mxhis),xor(5,mxfra)
      common/cranphi/ranphi
      save p,aimuni,xor
      phinll=phievt+ranphi
      if(phinll.lt.-pi)phinll=phinll+2*pi
      if(phinll.gt.pi)phinll=phinll-2*pi
      if(iffra(lf).eq.0.and.j.ne.0)then
        do l=1,5
          p(l,lf)=pptl(l,j)
        enddo
        do l=1,4
          xor(l,lf)=xorptl(l,j)
        enddo
        if(imofra(1,lf).ne.0)then
          call utrota(imofra(1,lf),r1fra(1,lf),r1fra(2,lf),r1fra(3,lf)
     $         ,p(1,lf),p(2,lf),p(3,lf))
          call utrota(imofra(1,lf),r1fra(1,lf),r1fra(2,lf),r1fra(3,lf)
     $         ,xor(1,lf),xor(2,lf),xor(3,lf))
        endif
        if(imofra(2,lf).ne.0)then !the x-z exchanged is ok !!
          call utrota(imofra(2,lf),r2fra(3,lf),r2fra(2,lf),r2fra(1,lf)
     $         ,p(3,lf),p(2,lf),p(1,lf))
          call utrota(imofra(2,lf),r2fra(3,lf),r2fra(2,lf),r2fra(1,lf)
     $         ,xor(3,lf),xor(2,lf),xor(1,lf))
        endif
        if(imofra(3,lf).ne.0)then
          imof3=sign(1,imofra(3,lf))
          if(abs(imofra(3,lf)).gt.1)then
            bofra1=0d0
            bofra2=0d0
            bofra5=1d0
            call utlob5(imof3*yhaha
     $                 ,p(1,lf),p(2,lf),p(3,lf),p(4,lf),p(5,lf))
            bofra3=bofra(1,lf)
            bofra4=bofra(2,lf)
            call utlob4(imof3,bofra1,bofra2,bofra3,bofra4,bofra5
     $         ,xor(1,lf),xor(2,lf),xor(3,lf),xor(4,lf))
            bofra3=bofra(3,lf)
            bofra4=bofra(4,lf)
            bofra5=bofra(5,lf)
          else
            bofra1=bofra(1,lf)
            bofra2=bofra(2,lf)
            bofra3=bofra(3,lf)
            bofra4=bofra(4,lf)
            bofra5=bofra(5,lf)
          endif
          call utlob4(imof3,bofra1,bofra2,bofra3,bofra4,bofra5
     $         ,p(1,lf),p(2,lf),p(3,lf),p(4,lf))
          call utlob4(imof3,bofra1,bofra2,bofra3,bofra4,bofra5
     $         ,xor(1,lf),xor(2,lf),xor(3,lf),xor(4,lf))
        endif
        iffra(lf)=1
      endif

c--------------------------------- 1 - 100 ----------------------------
      if(inom.eq.1)then
        x=1.
      elseif(inom.eq.2)then
        x=isign(1,idptl(j))
      elseif(inom.eq.3)then
        chrg=0
        if(iabs(idptl(j)).le.9999
     $       .and.mod(iabs(idptl(j)),10).le.1)
     $       call idchrg(idptl(j),chrg)
        if(chrg.eq.0.)then
          x=0
        else
          x=int(sign(1.,chrg))
        endif
      elseif(inom.eq.4)then
        iad=abs(idptl(j))
        jspin=mod(iad,10)
        x=0.
        if (iad.ge.100.and.iad.lt.1000) x=1./(1.+2.*jspin)
        if (iad.ge.1000.and.iad.lt.9999) x=1./(2.+2*jspin)
      elseif(inom.eq.5)then    !'rapx'  !st-rap for string segments only !!!!!!!!!!
        x=dezptl(j)
      elseif(inom.eq.6)then                                      !'iptlfr'
        x=0
        if(j.ge.minfra.and.j.le.maxfra)x=1
      elseif(inom.eq.7)then                                        !'rinp'
        aa=cos(phinll)
        bb=sin(phinll)
        x=xptl(j)*aa+yptl(j)*bb
      elseif(inom.eq.8)then                        !'eco' !engy in comoving frame
        x=0
        amt=p(5,lf)**2+p(1,lf)**2+p(2,lf)**2
        if(amt.gt.0..and.p(4,lf)+abs(p(3,lf)).gt.0.d0)then
          amt=sqrt(amt)
          rap=sign(1.,p(3,lf))*alog((p(4,lf)+abs(p(3,lf)))/amt)
          rapx=dezptl(j)
          x=amt*cosh(rap-rapx)
        endif
      elseif(inom.eq.9)then                                       !'tau'
        x=-999999
        !if(iorptl(j).ne.0)then
        ! jo=iorptl(j)
         dt=xorptl(4,j)  !-xorptl(4,jo)
         dz=xorptl(3,j)  !-xorptl(3,jo)
         x2=dt**2-dz**2
         if(x2.gt.0.)x=sqrt(x2)
        !endif
      elseif(inom.eq.10)then                                       !'ctr'
c        x=ctrevt
      elseif(inom.eq.11)then                                       !'v2np'
        phi=polar( p(1,lf) , p(2,lf) )
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        eta=0
        if(p(3,lf).ne.0..and.pt.ne.0.)eta=sign(1.,p(3,lf))*
     *       alog((sqrt(p(3,lf)**2+pt**2)+abs(p(3,lf)))/pt)
        if(eta.gt.0)then
        phi2=phi2neg
        else
        phi2=phi2pos
        endif
        x=cos(2*(phi-phi2))
      elseif(inom.eq.12)then                                      !'absrap'
        amt=p(5,lf)**2+p(1,lf)**2+p(2,lf)**2
        if(amt.gt.0..and.p(4,lf)+abs(p(3,lf)).gt.0.d0)then
          amt=sqrt(amt)
          x=alog((p(4,lf)+abs(p(3,lf)))/amt)
        else
          x=0.                  !
        endif
      elseif(inom.eq.13)then    !'rap'
        amt=p(5,lf)**2+p(1,lf)**2+p(2,lf)**2
        if(amt.gt.0..and.p(4,lf)+abs(p(3,lf)).gt.0.d0)then  !not correct if particles off-shell
c        if(amt.gt.0..and.p(4,lf)+p(3,lf).gt.0.d0)then    !not correct and assymetric if particles off-shell
          amt=sqrt(amt)
          x=sign(1.,p(3,lf))*log((p(4,lf)+abs(p(3,lf)))/amt)  !not correct if particles off-shell
c          x=alog((p(4,lf)+p(3,lf))/amt)    !not correct and assymetric if particles off-shell
c          x=0.5*alog((p(4,lf)+p(3,lf))/(p(4,lf)-p(3,lf)))  !always correct but numerically unstable
c          if(abs(x).lt.0.05.and.idptl(j).eq.120)
c     &    write(ifch,*)'pion epo',p(4,lf),p(3,lf),amt,x
        else
          x=0.                  !
        endif
      elseif(inom.eq.14)then                                         !'xp'
        x=sqrt(p(3,lf)**2+p(2,lf)**2+p(1,lf)**2)/emax(lf)
      elseif(inom.eq.15)then                                         !'xe'
        x=min(1.,p(4,lf)/emax(lf))
      elseif(inom.eq.16)then                                         !'pt'
        x=sqrt(p(2,lf)**2+p(1,lf)**2)
      elseif(inom.eq.17)then
        x=abs(p(1,lf))
      elseif(inom.eq.18)then
        x=abs(p(2,lf))
      elseif(inom.eq.19)then
        x=-log(sqrt(p(3,lf)**2+p(2,lf)**2+p(1,lf)**2)/emax(lf))
      elseif(inom.eq.20)then                                     !'xf'
        m=mod(ifra(lf)/10,10)
        if(m.eq.1.or.noebin.lt.0)then
c          pmax=sqrt((engy/2)**2-prom*2)
          pmax=pnullx               !???????????????????
          if(mod(ifra(lf),10).eq.2)pmax=pnll
          x=p(3,lf)/pmax
        else
          x=p(3,lf)/emax(lf)
        endif
c        if(x.gt.0.95.and.idptl(j).eq.1220)then
c          write(ifch,'(a,d25.15)')'ici !!!!!!!!!',seedc
c          stop
c        endif
      elseif(inom.eq.21)then
c        pmax=pmxevt
c        pmax=sqrt((engy/2)**2-prom*2)
        pmax=pnullx             !???????????????????
        if(mod(ifra(lf),10).eq.2)pmax=pnll
        x=-(amproj**2-2.*sqrt(amproj**2+pmax**2)*p(4,lf)
     *      +2.*abs(pmax*p(3,lf))+p(5,lf)**2)
      elseif(inom.eq.22)then
        amt=sqrt(p(5,lf)**2+p(1,lf)**2+p(2,lf)**2)
        if(amt.ne.0.)then
          x=-sign(1.,p(3,lf))*alog((p(4,lf)+abs(p(3,lf)))/amt)
        else
          x=0.                  !
        endif
      elseif(inom.eq.23)then                                     !'eta'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        if(p(3,lf).eq.0.)then
          x=0.
        elseif(pt.ne.0.)then
          x=sign(1.,p(3,lf))*
     *       alog((sqrt(p(3,lf)**2+pt**2)+abs(p(3,lf)))/pt)
        else
          x=sign(1000.,p(3,lf))
        endif
      elseif(inom.eq.24)then                                     !'theta (deg)'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        x=90
        if(p(3,lf).ne.0.)x=atan(pt/p(3,lf))/pi*180.
        if(x.lt.0.)x=180.+x
      elseif(inom.eq.25)then                                     !'pt2'
        x=p(2,lf)**2+p(1,lf)**2
      elseif(inom.eq.26)then                                     !'et'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        x=0
        eef=p(4,lf)
c        if(idptl(j).ge.1000)eef=eef-prom
c        if(idptl(j).le.-1000)eef=eef+prom
        p2=p(3,lf)**2+p(2,lf)**2+p(1,lf)**2
        if(p2.ne.0.)x=eef*pt/sqrt(p2)
      elseif(inom.eq.27)then                                     !'idptl'
        x=idptl(j)
      elseif(inom.eq.28)then    !istptl
        x=istptl(j)
      elseif(inom.eq.29)then    !mass
        x=p(5,lf)
        if(istptl(j).le.1)call idmass(idptl(j),x)
      elseif(inom.eq.30)then    !idaptl
        x=abs(idptl(j))
      elseif(inom.eq.31)then    !egy
        x=egyevt
      elseif(inom.eq.32)then    !arapwro
        x=0
        pt2=p(2,lf)**2+p(1,lf)**2
        if(p(3,lf).ne.0.)x=sign(1.,p(3,lf))*
     *       alog((sqrt(p(3,lf)**2+pt2+.13957**2)+abs(p(3,lf)))
     *       /sqrt(pt2+.13957**2))
      elseif(inom.eq.33)then                                  !'mt'
        x=sqrt(p(2,lf)**2+p(1,lf)**2+p(5,lf)**2)
      elseif(inom.eq.34)then                                  !'pplus'
        x=sign(1.,p(3,lf)) * (p(4,lf)+abs(p(3,lf)))
      elseif(inom.eq.35)then                                  !'pminus'
        x=sign(1.,p(3,lf)) * (p(4,lf)-abs(p(3,lf)))
      elseif(inom.eq.36)then                                  !'p5' (mass)
        x=p(5,lf)
      elseif(inom.eq.37)then                                  !pa
        x=sqrt(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2)
      elseif(inom.eq.38)then                                  !'pa'
        if(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2.ne.0)
     *       x=egyevt**2/sqrt(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2)*p(4,lf)
      elseif(inom.eq.39)then                                  !idpom
        x=idptl(j)/1000000
      elseif(inom.eq.40)then                                  !p3a
        x=abs(p(3,lf))
      elseif(inom.eq.41)then
        cm2=p(4,lf)**2-p(3,lf)**2-p(2,lf)**2-p(1,lf)**2         !cmass
        x=sign(sqrt(abs(cm2)),cm2)
      elseif(inom.eq.42)then    !arappi
        x=0
        pt2=p(2,lf)**2+p(1,lf)**2
        if(p(3,lf).ne.0.)
     *       x=alog((sqrt(p(3,lf)**2+pt2+.13957**2)+abs(p(3,lf)))
     *       /sqrt(pt2+.13957**2))
      elseif(inom.eq.50)then
        x=itsptl(j)
      elseif(inom.eq.51)then
        x=ityptl(j)
      elseif(inom.eq.52)then                      !'idoptl'
        x=0.
        if(iorptl(j).ne.0) x=idptl(iorptl(j))
      elseif(inom.eq.53)then
        x=j
      elseif(inom.eq.54)then                       !'sloela'
        call idflav(idptl(j),ifl1,ifl2,ifl3,jspin,indx)
        x=indx
      elseif(inom.eq.55)then                       !'p1'
        x=p(1,lf)
      elseif(inom.eq.56)then                       !'p2'
        x=p(2,lf)
      elseif(inom.eq.57)then                       !'p3'
        x=p(3,lf)
      elseif(inom.eq.58)then                       !'p4'
        x=p(4,lf)
      elseif(inom.eq.59)then                       !E/p_max
c        pmax=sqrt((engy/2)**2-prom*2)
          pmax=pnullx               !???????????????????
        if(mod(ifra(lf),10).eq.2)pmax=pnll
        x=p(4,lf)/pmax
      elseif(inom.eq.60)then                       !'ek'
        x=p(4,lf)-p(5,lf)
      elseif(inom.eq.61)then                       !'beta'
        x=p(3,lf)/p(4,lf)
      elseif(inom.eq.63)then                       !'mt0'
        x=sqrt(p(2,lf)**2+p(1,lf)**2+p(5,lf)**2)-p(5,lf)
      elseif(inom.eq.64)then                       !qsqptl
        x=qsqptl(j)
      elseif(inom.eq.65)then                       !xelab=Elab/Eolab
        x=p(4,lf)/(ecms**2/2/prom-prom)
        if(x.gt.0.9999999) x=.9999999
      elseif(inom.eq.66)then    !'hgtc05' ... charged ptl mult |[c]|<0.5
        x=multc05
      elseif(inom.eq.67)then    !'hadtyp' ... primary (1) or secondary (2) hadron
        if(j.le.nbdky)then
          x=1
        else
          x=2
        endif
      elseif(inom.eq.68)then    !'hgtc1'
        x=multc1
      elseif(inom.eq.69)then                       !'x4'
        x=xor(4,lf)
      elseif(inom.eq.70)then                       !'npn'
        x=npjevt+ntgevt
      elseif(inom.eq.71)then                       !'routp'
        cc=-sin(phinll)
        dd=cos(phinll)
        x=xptl(j)*cc+yptl(j)*dd
      elseif(inom.eq.72)then    !'hgtc3' ... charged ptl mult |eta|<3.15  /6.3
        x=multc3/6.3
      elseif(inom.eq.73)then    !'mu14' ... charged ptl mult |eta|<1  pt>.4
        x=multc14
      elseif(inom.eq.74)then    !'delphi' ... azimuthhal correlation
        x=10000.
        pt=sqrt(p(1,lf)**2+p(2,lf)**2)

        if(nint(ypara(1,n)).ne.0.and.j.ne.nint(ypara(1,n)).and.
     $           pt.gt.0)then
           phi=sign(1.,p(2,lf))*acos(p(1,lf)/pt)
               x=phi-ypara(2,n)
           phiz= ypara(3,n)
           if   (x.lt.(-2+phiz)*pi)then
            x=x+4*pi
           elseif(x.lt.(0+phiz)*pi)then
            x=x+2*pi
           elseif(x.gt.(4+phiz)*pi)then
            x=x-4*pi
           elseif(x.gt.(2+phiz)*pi)then
            x=x-2*pi
           endif
        endif
      elseif(inom.eq.75)then    !'v2'
c        if(iranphi.ne.1)stop'\n\n ERROR 29062010b \n\n'
        aa=cos(phinll)
        bb=sin(phinll)
        cc=-sin(phinll)
        dd=cos(phinll)
        px=p(1,lf)*aa+p(2,lf)*bb
        py=p(1,lf)*cc+p(2,lf)*dd
        pt2=p(2,lf)**2+p(1,lf)**2
        x=0
        if(pt2.gt.0.)x=(px**2-py**2)/pt2
      elseif(inom.eq.76)then                                     !'pt4'
        x=(p(2,lf)**2+p(1,lf)**2)**2
      elseif(inom.eq.77)then                                     !'rin'
        x=rinptl(j)
      elseif(inom.eq.78)then              !'theta for H1 (rad) for (proj side)'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        p1=p(1,lf)
        p2=p(2,lf)
        p3=p(3,lf)
        p4=p(4,lf)
        p5=p(5,lf)
        if(abs(p3).gt.1e-5)then
c put the particle in the projectile frame
          call utlob5(yhaha,p1,p2,p3,p4,p5)
c put the particle in a frame where the projectile (proton) has 820 GeV (HERA)
          call utlob4(-1,0d0,0d0,819.99946d0,820.d0,0.938d0,p1,p2,p3,p4)
          x=atan(pt/p3)
          if(x.lt.0.)x=pi+x
        else
          x=0.5*pi
        endif
      elseif(inom.eq.79)then             !'theta for H1 (rad) (for target side)'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        p1=p(1,lf)
        p2=p(2,lf)
        p3=p(3,lf)
        p4=p(4,lf)
        p5=p(5,lf)
        if(abs(p3).gt.1e-5)then
c put the particle in the projectile frame
          call utlob5(-yhaha,p1,p2,p3,p4,p5)
c put the particle in a frame where the projectile (proton) has 820 GeV (HERA)
          call utlob4(-1,0d0,0d0,-819.99946d0,820d0,0.938d0,p1,p2,p3,p4)
          x=atan(pt/p3)
          if(x.gt.0.)x=pi-x
        else
          x=0.5*pi
        endif
      elseif(inom.eq.80)then    !'phi'
        x=1000
        pt=sqrt(p(1,lf)**2+p(2,lf)**2)
        if(pt.gt.0.)then
           phi=sign(1.,p(2,lf))*acos(p(1,lf)/pt)
           x=phi-phinll
        endif
        if(x.lt.-pi)x=x+2*pi
        if(x.gt.pi)x=x-2*pi
      elseif(inom.eq.81)then  !'isoft'
        x=0
        it=ityptl(j)
        if(it.ge.20.and.it.le.29)x=1
        if(it.ge.40.and.it.le.60)x=1
      elseif(inom.eq.82)then  !'mux' ... charged ptl mult 
        x= ypara(1,n)        
      elseif(inom.eq.83)then    !'v4'
        aa=cos(phinll)
        bb=sin(phinll)
        cc=-sin(phinll)
        dd=cos(phinll)
        px=p(1,lf)*aa+p(2,lf)*bb
        py=p(1,lf)*cc+p(2,lf)*dd
        pt2=p(2,lf)**2+p(1,lf)**2
        x=0
        if(pt2.gt.0.)x=px**2/pt2   !cos**2
        x=8*x**2-8*x+1
      elseif(inom.eq.84)then                       !'x3'
        x=xor(3,lf)
      elseif(inom.eq.85)then   !jorptl
        x=jorptl(j)
      elseif(inom.eq.86)then    !'ptlead' ... pt of particle with higher pt
        x=0.
        if(nint(ypara(1,n)).ne.0)x=ypara(4,n)
      elseif(inom.eq.87)then    !'mu25' ... charged ptl mult |eta|<2.5  pt>.5
        x=multc25
      elseif(inom.eq.88)then    !'pai' 
        x=0
        if(bimevt.gt.xpara(1,n).and.bimevt.lt.xpara(2,n))then
          ida=idptl(j)
          pt=sqrt(p(2,lf)**2+p(1,lf)**2)
          ipt=pt/delpt+1
          if(ipt.ge.1.and.ipt.le.maxpt)then
            if(ida.eq.120)then
              if(ipaipi(ipt).eq.0)then 
                x=paipi(ipt)
                ipaipi(ipt)=1
              endif
            elseif(ida.eq.1120)then
              if(ipaipr(ipt).eq.0)then 
                x=paipr(ipt)
                !print*,'++++++++++',ipt,ipaipr(ipt),x
                ipaipr(ipt)=1
              endif
            endif
          endif 
        endif 
      elseif(inom.eq.89)then    !'co2' 
        x=0
        if(bimevt.gt.xpara(1,n).and.bimevt.lt.xpara(2,n))then
          ida=idptl(j)
          pt=sqrt(p(2,lf)**2+p(1,lf)**2)
          ipt=pt/delpt+1
          if(ipt.ge.1.and.ipt.le.maxpt)then
            if(ida.eq.120)then
              if(ico2pi(ipt).eq.0)then 
                x=co2pi(ipt)
                ico2pi(ipt)=1
              endif
            elseif(ida.eq.1120)then
              if(ico2pr(ipt).eq.0)then 
                x=co2pr(ipt)
                ico2pr(ipt)=1
              endif
            endif
          endif  
        endif 
      elseif(inom.eq.90)then    !'co3' 
        x=0
        if(bimevt.gt.xpara(1,n).and.bimevt.lt.xpara(2,n))then
          ida=idptl(j)
          pt=sqrt(p(2,lf)**2+p(1,lf)**2)
          ipt=pt/delpt+1
          if(ipt.ge.1.and.ipt.le.maxpt)then
            if(ida.eq.120)then
              if(ico3pi(ipt).eq.0)then 
                x=co3pi(ipt)
                ico3pi(ipt)=1
              endif
            elseif(ida.eq.1120)then
              if(ico3pr(ipt).eq.0)then 
                x=co3pr(ipt)
                ico3pr(ipt)=1
              endif
            endif
          endif  
        endif  
      elseif(inom.eq.91)then                       !'rad'
        x=0.001       !unit is fm !
        x1=xor(1,lf)
        x2=xor(2,lf)
        xd=dble(x1)**2+dble(x2)**2
        if(xd.gt.0.d0.and.xd.eq.xd)x=sqrt(xd)
      elseif(inom.eq.92)then                                     !'abseta'
        pt=sqrt(p(2,lf)**2+p(1,lf)**2)
        if(p(3,lf).eq.0.)then
          x=0.
        elseif(pt.ne.0.)then
          x=sign(1.,p(3,lf))*
     *       alog((sqrt(p(3,lf)**2+pt**2)+abs(p(3,lf)))/pt)
        else
          x=sign(1000.,p(3,lf))
        endif
c        pp=sqrt(p(2,lf)**2+p(1,lf)**2+p(3,lf)**2)
c        x=0
c        if(pp-p(3,lf).gt.0..and.pp+p(3,lf).gt.0.)x=
c     *       0.5*log((pp+p(3,lf))/(pp-p(3,lf)))
         x=abs(x)
      elseif(inom.eq.93)then                                     !'phiexp'
        x=1000
        pt=sqrt(p(1,lf)**2+p(2,lf)**2)
        if(pt.gt.0.)x=sign(1.,p(2,lf))*acos(p(1,lf)/pt)
        if(x.lt.-pi)x=x+2*pi
        if(x.gt.pi)x=x-2*pi
      elseif(inom.eq.94)then    !'mu24' ... charged ptl mult |eta|<2.4 (CMS)
        x=multc24
c--------------------------------- 101 - 200 ----------------------------

      elseif(inom.eq.101)then           !mulevt
        x=1.
      elseif(inom.eq.102)then                      !'etevt'
        x=0
        if(istptl(j).eq.0)then
         eef=p(4,lf)
         if(maproj.gt.1.or.matarg.gt.1)then
           if(idptl(j).ge.1000)eef=eef-prom
           if(idptl(j).le.-1000)eef=eef+prom
         endif
         pp=sqrt(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2)
         if(pp.ne.0.)x=eef*sqrt(p(1,lf)**2+p(2,lf)**2)/pp
         if(x.ne.x)then
           write(ifch,*)x,eef,p(1,lf),p(2,lf),p(3,lf),pp,prom,idptl(j),j
           call alist('xan&',1,nptl)
           stop 'probleme dans xan'
         endif
         endif
      elseif(inom.eq.103)then
        x=p(4,lf)/1000.
      elseif(inom.eq.104)then                       !'ev6evt'
        x=0
        if(istptl(j).eq.0)then
         pt=sqrt(p(2,lf)**2+p(1,lf)**2)
         eta=0
         if(p(3,lf).ne.0..and.pt.ne.0.)eta=sign(1.,p(3,lf))*
     *   alog((sqrt(p(3,lf)**2+pt**2)+abs(p(3,lf)))/pt)
         if(pt.eq.0.)eta=sign(1e5,p(3,lf))
         if(eta.gt.6.0)then
          eef=p(4,lf)
          if(idptl(j).ge.1000)eef=eef-prom
          if(idptl(j).le.-1000)eef=eef+prom
          pp=sqrt(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2)
          if(pp.ne.0.)x=0.001*eef
         endif
        endif
      elseif(inom.eq.105)then
        !etot=maproj*emax(lf)+matarg*0.94  !nur richtig fur target frame!!!!!
        etot=maproj*emax(lf)+matarg*emax(lf)
        x=p(4,lf)/etot
      elseif(inom.eq.106)then
        x=isign(1,idptl(j))
      elseif(inom.eq.107)then                       !'ptevt'
        x=sqrt(p(2,lf)**2+p(1,lf)**2)
      elseif(inom.eq.108)then                       !'pmxevt'
        x=pmxevt

c--------------------------------- > 200 ----------------------------

      elseif(inom.eq.201)then
        x=1.
      elseif(inom.eq.202)then
        x=egyevt
      elseif(inom.eq.203)then
        x=bimevt
      elseif(inom.eq.204)then                       !'xbjevt'
        x=xbjevt
      elseif(inom.eq.205)then                       !'qsqevt'
        x=qsqevt
      elseif(inom.eq.206)then                       !'yevt'
        x=qsqevt/xbjevt/ecms**2
      elseif(inom.eq.207)then                       !'eloevt'
c        x=qsqevt/4./elepti+elepti*(1.-qsqevt/xbjevt/ecms**2)
        x=elepto
      elseif(inom.eq.208)then                       !nd1evt
        x=nsdiff(1,noweak(n))
      elseif(inom.eq.209)then                       !'nd2evt'
        x=nsdiff(2,noweak(n))
      elseif(inom.eq.210)then                       !'theevt'
c        eloevt=qsqevt/4./elepti+elepti*(1.-qsqevt/xbjevt/ecms**2)
c        x=acos(1-qsqevt/2./elepti/eloevt)/pi*180.
        x=acos(1-qsqevt/2./elepti/elepto)/pi*180.
      elseif(inom.eq.211)then                       !'nspevt'
        x=0
        do i=1,nptl
         if((istptl(i).eq.30.or.istptl(i).eq.31)
     &      .and.int(idptl(i)/1000000).eq.1)x=x+1
        enddo
      elseif(inom.eq.212)then                       !'nhpevt'
        x=0
        do i=1,nptl
         if((istptl(i).eq.30.or.istptl(i).eq.31)
     &      .and.int(idptl(i)/1000000).eq.3)x=x+1
        enddo
      elseif(inom.eq.213)then                       !'sigtot'
        x=sigtot
      elseif(inom.eq.214)then                       !'sigela'
        x=sigela
      elseif(inom.eq.215)then                       !'sloela'
        x=sloela
      elseif(inom.eq.216)then                       !'nrgevt'
        x=0
        do i=1,nptl
          if(istptl(i).eq.31.and.int(idptl(i)/10000).eq.2)x=x+1
        enddo
      elseif(inom.eq.217)then                       !qevt
        x=sqrt(qsqevt)
      elseif(inom.eq.218)then   !qevt
        if(iappl.eq.8)then
          x=qtl
        else
          x=pprt(1,5)
        endif
      elseif(inom.eq.219)then                       !'nd0evt'  UA1
        x=nsdiff(0,noweak(n))
      elseif(inom.eq.220)then!------------------------------------------
        x=sngl(bofra(1,lf))     !thrust
      elseif(inom.eq.221)then
        x=1.-sngl(bofra(1,lf))  !1-thrust
      elseif(inom.eq.222)then
        x=sngl(bofra(2,lf))     !major
      elseif(inom.eq.223)then
        x=sngl(bofra(3,lf))     !minor
      elseif(inom.eq.224)then
        x=sngl(bofra(2,lf)-bofra(3,lf)) !oblateness
      elseif(inom.eq.230)then!------------------------------------------
        x=1.5*(1.-sngl(bofra(1,lf))) !spherecity
      elseif(inom.eq.231)then
        x=1.5*sngl(bofra(3,lf)) !aplanarity
      elseif(inom.eq.232)then
        x=3.*sngl(bofra(1,lf)*bofra(2,lf)+bofra(1,lf)*bofra(3,lf)
     &       +bofra(2,lf)*bofra(3,lf)) !c-parameter
      elseif(inom.eq.233)then
        x=27.*sngl(bofra(1,lf)*bofra(2,lf)*bofra(3,lf))   !d-parameter
      elseif(inom.eq.234)then                       !npoevt
        x=0
        do i=1,nptl
         if(istptl(i).eq.30.or.istptl(i).eq.31)x=x+1
        enddo
      elseif(inom.eq.235)then                       !npnevt
        x=npjevt+ntgevt   !npnevt
      elseif(inom.eq.236)then                       !ikoevt
        x=ikoevt
      elseif(inom.eq.237)then                       !iktevt
c        x=zkotest
      elseif(inom.eq.238)then  !npxevt ... nr of pomerons, including absorbed
        x=0
        do i=1,nptl
         if(istptl(i).eq.30.or.istptl(i).eq.31)x=x+1
         if(mod(abs(idptl(i)),100).eq.94)x=x+0.5
        enddo
      elseif(inom.eq.239)then                       !'nd6evt'
        x=nsdiff(6,noweak(n))
      elseif(inom.eq.240)then    !mu1evt ... charged ptl multipl for central rap
        x=multy1
      elseif(inom.eq.241)then    !muievt ... charged ptl multipl
        x=multyi
      elseif(inom.eq.242)then    !hgtevt ... charged ptl multipl for central eta
        x=multc05
      elseif(inom.eq.243)then                       !difevt
        npom=0
        do i=1,nptl
         if(istptl(i).eq.30.or.istptl(i).eq.31)npom=npom+1
        enddo
        x=0
        if(npom.eq.0)x=1
      elseif(inom.eq.244)then                       !dixevt
        zpom=0
        do i=1,nptl
         if(istptl(i).eq.30.or.istptl(i).eq.31)zpom=zpom+1
         if(mod(abs(idptl(i)),100).eq.94)zpom=zpom+0.5
        enddo
        x=0
        if(abs(zpom).lt.0.001)x=1
      elseif(inom.eq.245)then                       !'nd7evt' CMS NSD
        x=nsdiff(7,noweak(n))
      elseif(inom.eq.246)then                       !'nd8evt'  ATLAS
        x=nsdiff(8,noweak(n))
      elseif(inom.eq.247)then                       !'nd9evt'  ALICE 900 GeV
        x=nsdiff(9,noweak(n))
      elseif(inom.eq.248)then                       !'ndaevt'  ALICE 2.36 TeV
        x=nsdiff(10,noweak(n))
      elseif(inom.eq.249)then                       !'ndbevt'  ALICE Inel > 0
        x=nsdiff(11,noweak(n))
      elseif(inom.eq.250)then
        if(iappl.eq.8)then      !mass in
          x=-pptl(5,6)
        else
          x=pprt(5,3)
        endif
      elseif(inom.eq.251)then
        if(iappl.eq.8)then      !mass out
          x=pptl(5,7)
        else
          x=pprt(5,2)
        endif
      elseif(inom.eq.252)then
        if(iappl.eq.8)then
          x=-pptl(4,6)
        else
          x=pprt(4,2)
        endif
      elseif(inom.eq.253)then
        if(iappl.eq.8)then
          x=pptl(4,7)
        else
          x=pprt(4,3)
        endif
      elseif(inom.eq.254)then
        if(iappl.eq.8)then
          x=abs(pptl(3,6))
        else
          x=abs(pprt(3,2))
        endif
      elseif(inom.eq.255)then
        if(iappl.eq.8)then
          x=abs(pptl(3,7))
          do l=1,5
            p(l,lf)=pptl(l,7)
          enddo
          if(imofra(1,lf).ne.0)then
            call utrota(imofra(1,lf),r1fra(1,lf),r1fra(2,lf),r1fra(3,lf)
     $           ,p(1,lf),p(2,lf),p(3,lf))
          endif
          if(imofra(2,lf).ne.0)then !the x-z exchanged is ok !!
            call utrota(imofra(2,lf),r2fra(3,lf),r2fra(2,lf),r2fra(1,lf)
     $           ,p(3,lf),p(2,lf),p(1,lf))
          endif
          if(imofra(3,lf).ne.0)then
            call utlob4(imofra(3,lf),bofra(1,lf),bofra(2,lf)
     $           ,bofra(3,lf) ,bofra(4,lf),bofra(5,lf)
     $           ,p(1,lf),p(2,lf),p(3,lf),p(4,lf))
          endif
          x=abs(p(3,lf))
        else
          x=abs(pprt(3,3))
        endif
      elseif(inom.eq.256)then  !pxfevt: leading proton xf in cms
        x=-2
c       pmax=sqrt((ecms/2.)**2-prom**2)
          pmax=pnullx               !???????????????????
        do i=1,nptl
          if(idptl(i).eq.1120.and.istptl(i).eq.0)then
            if(iframe.eq.11)then
              pz=pptl(3,i)
            else
              amt=sqrt(prom**2+pptl(1,i)**2+pptl(2,i)**2)
              rap=alog((pptl(4,i)+pptl(3,i))/amt)
     &           -alog((sqrt(pnll**2+ecms**2)+pnll)/ecms)
              pz=amt*sinh(rap)
            endif
            x=max(x,pz/pmax)
          endif
        enddo
      elseif(inom.eq.257)then  !  pi+xf: pi+ yield at cms xf>0.01
        x=0.
c        pmax=sqrt((ecms/2)**2-prom*2)
          pmax=pnullx               !???????????????????
        do i=1,nptl
          if(idptl(i).eq.120.and.istptl(i).eq.0)then
            if(iframe.eq.11)then
              pz=pptl(3,i)
            else
              amt=sqrt(pptl(5,i)**2+pptl(1,i)**2+pptl(2,i)**2)
              rap=alog((pptl(4,i)+pptl(3,i))/amt)
     &           -alog((sqrt(pnll**2+ecms**2)+pnll)/ecms)
              pz=amt*sinh(rap)
            endif
            if(pz/pmax.gt.0.01)x=x+1.
          endif
        enddo
      elseif(inom.eq.258)then  !  pi-xf: pi- yield at cms xf>0.01
        x=0.
c        pmax=sqrt((ecms/2)**2-prom*2)
          pmax=pnullx               !???????????????????
        do i=1,nptl
          if(idptl(i).eq.-120.and.istptl(i).eq.0)then
            if(iframe.eq.11)then
              pz=pptl(3,i)
            else
              amt=sqrt(pptl(5,i)**2+pptl(1,i)**2+pptl(2,i)**2)
              rap=alog((pptl(4,i)+pptl(3,i))/amt)
     &           -alog((sqrt(pnll**2+ecms**2)+pnll)/ecms)
              pz=amt*sinh(rap)
            endif
            if(pz/pmax.gt.0.01)x=x+1.
          endif
        enddo
      elseif(inom.eq.260)then!------------------------------
        x=sigcut
      elseif(inom.eq.261)then
        x=keu
      elseif(inom.eq.262)then
        x=ked
      elseif(inom.eq.263)then
        x=kes
      elseif(inom.eq.265)then
        x=kolevt
      elseif(inom.eq.266)then
        x=sigsd
      elseif(inom.eq.267)then
        x=nglevt
      elseif(inom.eq.268)then  ! kppevt : collision number per participant
        x=kolevt/float(npjevt+ntgevt)
      elseif(inom.eq.269)then  ! npievt : pion + multi per event
        x=0
        do i=1,nptl
         if(idptl(i).eq.120)x=x+1
        enddo
      elseif(inom.eq.270)then  ! np2evt : pion + multi per event
        x=0
        do i=1,nptl
         if(idptl(i).eq.120)x=x+1
        enddo
        x=x/float(npjevt+ntgevt)
      elseif(inom.eq.271)then
        x=sigdif
      elseif(inom.eq.272)then  !number of inelastic collisions per event
        x=koievt
      elseif(inom.eq.273)then  ! inelasticity (energy loss of leading particle)
        x=0.
        do i=maproj+matarg+1,nptl
          if(istptl(i).eq.0)then
            if((((abs(idptl(i)).gt.1000.and.abs(idptl(i)).lt.10000)
     *           .and.idproj.gt.1000).or.(iabs(idptl(i)).gt.100
     *           .and.idproj.lt.1000)).and.pptl(4,i)
     *           .gt.x.and.pptl(3,i).gt.0.)x=pptl(4,i)
          endif
        enddo
        Eini=pptl(4,1)
        if(Eini.gt.0.)x=(Eini-x)/Eini
      elseif(inom.eq.274)then  ! elasticity (energy of leading particle)
        x=0.
        do i=maproj+matarg+1,nptl
          if(istptl(i).eq.0)then
            if((((abs(idptl(i)).gt.1000.and.abs(idptl(i)).lt.10000)
     *           .and.idproj.gt.1000).or.(iabs(idptl(i)).gt.100
     *           .and.idproj.lt.1000)).and.pptl(4,i)
     *           .gt.x.and.pptl(3,i).gt.0.)x=pptl(4,i)
          endif
        enddo
        Eini=pptl(4,1)
        if(Eini.gt.0.)x=x/Eini
      elseif(inom.eq.275)then         !'itgevt'
        x=0
        if(nint(ypara(1,n)).ne.0)x=1
      elseif(inom.eq.276)then         !'hrdevt' ......  1 = hard event
        x=0
        if(nint(ypara(1,n)).ne.0)x=1
      elseif(inom.eq.277)then         !'ej1evt' .... et of jet 1
        x=0
        if(nint(ypara(1,n)).ne.0)
     &  x=ypara(2,n)
      elseif(inom.eq.278)then         !'pj1evt' .... phi of jet 1
        x=1000
        if(nint(ypara(1,n)).ne.0)
     &  x=ypara(4,n)
      elseif(inom.eq.279)then         !'ej2evt' .... et of jet 2
        x=0
        if(nint(ypara(6,n)).ne.0)
     &  x=ypara(7,n)
      elseif(inom.eq.280)then         !'pj2evt' .... delta_phi of jet 2 1
        x=1000
        if(nint(ypara(6,n)).ne.0)then
          x=ypara(9,n)-ypara(4,n)
           if(x.lt.-2.5*pi)then
            x=x+4*pi
           elseif(x.lt.-0.5*pi)then
            x=x+2*pi
          elseif(x.gt.3.5*pi)then
            x=x-4*pi
          elseif(x.gt.1.5*pi)then
            x=x-2*pi
          endif
        endif
      elseif(inom.eq.281)then         !'zppevt'
        x=zppevt
      elseif(inom.eq.282)then         !'zptevt'
        x=zptevt
      elseif(inom.eq.283)then
        stop '**********not used*********'
      elseif(inom.eq.284)then                       !'nd3evt'
        x=nsdiff(3,noweak(n))
      elseif(inom.eq.285)then                       !'nd4evt'
        x=nsdiff(4,noweak(n))
      elseif(inom.eq.286)then         !'mubevt'
        x=multeb
      elseif(inom.eq.287)then                       !'nd5evt'
        x=nsdiff(5,noweak(n))
      elseif(inom.eq.288)then
        x=ekievt
      elseif(inom.eq.289)then                       !'diffmevt'
        x=isdiff(1)
      elseif(inom.eq.290)then                       !'diffxevt'
        x=isdiff(2)
      elseif(inom.eq.291.or.inom.eq.292)then  ! mass of produced system (inelasticity of leading particle )
        x=0.
        i=idlead
        if(i.gt.0)then
          pmax=pnullx
          if(mod(ifra(lf),10).eq.2)pmax=pnll
          x=abs(pptl(3,i)/pmax)
          x=(1.-x)*engy*engy
          if(inom.eq.291.and.x.gt.0.)x=sqrt(x)
c         write(ifch,*)'ici',i,idptl(i),x,abs(pptl(3,i)/pmax)
        endif
      elseif(inom.eq.293)then  ! tdevt : -t of leading particle
        x=0.
        i=idlead
        if(i.gt.0)then
          pmax=pnullx
          if(mod(ifra(lf),10).eq.2)pmax=pnll
c        xxx=(amproj**2-2.*sqrt(amproj**2+pmax**2)*pptl(4,i)
c     *      +2.*abs(pmax*pptl(4,i))+pptl(5,i)**2)
          ppt=sqrt(pptl(1,i)**2+pptl(2,i)**2)
          if(abs(pptl(3,i)).gt.0.)then
            theta=atan(ppt/pptl(3,i))
          else
            theta=pi/2.
          endif
          x=abs(pptl(3,i)/pmax)
c -t definition of UA4 (Phys Lett B136,217)
          x=pptl(5,i)**2*(1.-x)**2/x+2*x*pmax*pmax*(1.-cos(theta))
c         write(*,*)'ici',i,idptl(i),theta,x,xxx
        endif
      elseif(inom.eq.294)then          !'ndpevt' pomeron from diffraction
        x=0
        do i=1,nptl
         if((istptl(i).eq.30.or.istptl(i).eq.31)
     &      .and.mod(ityptl(i),10).eq.5)x=x+1
        enddo
      elseif(inom.eq.295)then          !'rapgap' rapidity gap
        x=rapgap
      elseif(inom.eq.296)then        !'ng1evt'
        x=ng1evt
      elseif(inom.eq.297)then        !'r21evt'
        x=0
        if(ng1evt.ne.0)x=ng2evt/float(ng1evt)
      elseif(inom.eq.298)then
        x=sval(1,n)
      elseif(inom.eq.299)then
        x=sval(2,n)
      elseif(inom.eq.301)then   !---------------------------------------------
        if(j.eq.0)then          !initialize
          do l=1,4
            aimuni(l,n)=0.
          enddo
        elseif(j.gt.nptl)then   !final calculation
          am2=aimuni(4,n)**2-aimuni(3,n)**2
     $         -aimuni(2,n)**2-aimuni(1,n)**2
          x=sign(sqrt(abs(am2)),am2)
c          print *, x
        else                    !routine work
          do l=1,4
            aimuni(l,n)=aimuni(l,n)+p(l,lf)
          enddo
c          print *, j,(p(l,lf),l=1,5)
        endif
      elseif(inom.ge.302.and.inom.le.305)then   !-----------------------
        if(j.eq.0)then          !initialize
          do l=1,4
            aimuni(l,n)=0.
          enddo
        elseif(j.gt.nptl)then   !final calculation
          if(inom.eq.302) x=max(aimuni(1,n)/2/(aimuni(2,n)+aimuni(4,n))
     $         ,aimuni(3,n)/2/(aimuni(2,n)+aimuni(4,n)))
          if(inom.eq.303) x=min(aimuni(1,n)/2/(aimuni(2,n)+aimuni(4,n))
     $         ,aimuni(3,n)/2/(aimuni(2,n)+aimuni(4,n)))
          if(inom.eq.304) x=abs(aimuni(1,n)/2/(aimuni(2,n)+aimuni(4,n))
     $         -aimuni(3,n)/2/(aimuni(2,n)+aimuni(4,n)))
          if(inom.eq.305) x=aimuni(1,n)/2/(aimuni(2,n)+aimuni(4,n))
     $         +aimuni(3,n)/2/(aimuni(2,n)+aimuni(4,n))
        else                    !routine work
          l=0
          if(p(3,lf).lt.0.)l=2
          aimuni(1+l,n)=aimuni(1+l,n)+sqrt(p(1,lf)**2+p(2,lf)**2)
          aimuni(2+l,n)=aimuni(2+l,n)
     $         +sqrt(p(1,lf)**2+p(2,lf)**2+p(3,lf)**2)

        endif
      elseif(inom.eq.306.or.inom.eq.307.or.inom.eq.308)then !---------
        if(j.eq.0)then          !initialize
          do ll=1,8
            aimuni(ll,n)=0.
          enddo
        elseif(j.gt.nptl)then   !final calculation
          am2a=aimuni(4,n)**2-aimuni(3,n)**2
     $         -aimuni(2,n)**2-aimuni(1,n)**2
          am2b=aimuni(8,n)**2-aimuni(7,n)**2
     $         -aimuni(6,n)**2-aimuni(5,n)**2
          if(inom.eq.306)x=(max(0.,am2a,am2b))/engy**2
          if(inom.eq.307)x=(max(0.,min(am2a,am2b)))/engy**2
          if(inom.eq.308)x=(abs(am2a-am2b))/engy**2
        else                    !routine work
          ll=0
          if(p(3,lf).lt.0.)ll=4
          do l=1,4
            aimuni(l+ll,n)=aimuni(l+ll,n)+p(l,lf)
          enddo
        endif
      elseif (inom.eq.310.or.inom.eq.311) then !---------
        if(j.eq.0)then          !initialize
          aimuni(1,n)=0
          aimuni(2,n)=0
          do i=1,nptl
c            charge=0.
             if(istptl(i).eq.0) then
               if (idptl(i).eq.idcod(1,n)) aimuni(1,n)=aimuni(1,n)+1.
               if (idptl(i).eq.idcod(2,n)) aimuni(2,n)=aimuni(2,n)+1.
             endif
           enddo
        elseif(j.gt.nptl)then   !final calculation
          if(aimuni(1,n).eq.0.or.aimuni(2,n).eq.0) then
            ncevt(n)=ncevt(n)-1
          endif
          x=xmin(n)-100.
          do i=1,nbin(n)
            zcbin(i,nac(n),n)=abs(zcbin(i,nac(n),n))
         enddo
       else                    !routine work
          if( istptl(j).eq.0
     $         .and. aimuni(1,n).ne.0. .and. aimuni(2,n).ne.0. ) then
            id1=idptl(j)
            if(id1.eq.idcod(1,n) .or. id1.eq.idcod(2,n)) then
              y1=sign(1.,pptl(3,j))*alog((pptl(4,j)+abs(pptl(3,j)))
     *             /sqrt(pptl(5,j)**2+pptl(1,j)**2+pptl(2,j)**2))
              do i=1,nptl
                if(i.eq.j .or. istptl(i).ne.0) goto 124
                id2=idptl(i)
                if(id2.eq.idcod(1,n) .or. id2.eq.idcod(2,n)) then
                  y2=sign(1.,pptl(3,i))*alog((pptl(4,i)+abs(pptl(3,i)))
     *                 /sqrt(pptl(5,i)**2+pptl(1,i)**2+pptl(2,i)**2))
                  dy=(y2-y1)
                  if(inom.eq.311) dy=abs(dy)
                  ib=1+int((dy-xmin(n))*xinc(n))
                  if(dy.ge.xmin(n).and.dy.le.xmax(n)) then
                    if( id1.eq.idcod(1,n) ) then
                      if( id2.eq.idcod(2,n) ) then
                        bin(ib,nac(n),n)=bin(ib,nac(n),n)+.5/aimuni(2,n)
                        zcbin(ib,nac(n),n)=zcbin(ib,nac(n),n)+1
                      else
                        bin(ib,nac(n),n)=bin(ib,nac(n),n)-.5/aimuni(1,n)
                        zcbin(ib,nac(n),n)=zcbin(ib,nac(n),n)-1
                      endif
                    else        !id1 is idcod(2,n)
                      if(id2.eq.idcod(1,n)) then
                        bin(ib,nac(n),n)=bin(ib,nac(n),n)+.5/aimuni(1,n)
                        zcbin(ib,nac(n),n)=zcbin(ib,nac(n),n)+1
                      else
                        bin(ib,nac(n),n)=bin(ib,nac(n),n)-.5/aimuni(2,n)
                        zcbin(ib,nac(n),n)=zcbin(ib,nac(n),n)-1
                      endif
                    endif
                  endif
                endif
 124            continue 
              enddo
            endif
          endif
        endif
      elseif (inom.eq.312) then !---------
        x=sigine
      elseif (inom.eq.313) then !---------
        x=sigineaa
      elseif (inom.eq.314) then !---------
        x=alpD(idxD0,iclpro,icltar)
      elseif (inom.eq.315) then !---------
        x=alpD(1,iclpro,icltar)
      elseif (inom.eq.316) then !---------
        x=betD(idxD0,iclpro,icltar)
        if(x.lt.0.)x=-10.*x
      elseif (inom.eq.317) then !---------
        x=betD(1,iclpro,icltar)
      elseif (inom.eq.318) then !---------
        x=rexdif(iclpro)
      elseif (inom.eq.319) then !---------
        x=rexdif(icltar)
      elseif(inom.eq.320)then    !m14evt ... multipl |eta|<1, pt>0.4
        x=multc14
      elseif(inom.eq.321)then    !ht3evt ... height |eta|<3.15
        x=multc3/6.3
      elseif (inom.eq.322) then !---------
        x=sigineex
      elseif (inom.eq.323) then !---------
        x=sigdifex
      elseif (inom.eq.324) then !---------
        x=sigsdex
      elseif (inom.eq.325) then !---------
        x=ekin
      elseif (inom.eq.326) then !---------
        x=sigcutaa
      elseif (inom.eq.327) then !---------
        x=sigtotaa
      elseif (inom.eq.328) then !---------
        x=xkappafit(iclegy,iclpro,icltar,1)
      elseif (inom.eq.329) then !---------
        x=gamD(idxD0,iclpro,icltar)
      elseif (inom.eq.330) then !---------
        x=gamD(1,iclpro,icltar)
      elseif (inom.eq.331) then !---------
        x=delD(idxD0,iclpro,icltar)
      elseif (inom.eq.332) then !---------
        x=delD(1,iclpro,icltar)
      elseif(inom.eq.333)then                       !'nd6evt'
        x=nsdiff(6,noweak(n))
      elseif(inom.eq.334)then !'muxevt' ... multipl 
        x= ypara(1,n)         
      elseif(inom.eq.335)then                       
        x=abs(nint(typevt))                        !ND(1), DD(2), or SD(3)
      elseif(inom.eq.339)then    !m25evt ... multipl |eta|<2.5, pt>0.5
        x=multc25
      elseif(inom.eq.340)then    !'segevt' ... segment multiplicity
c        x=segevt
      elseif(inom.eq.341)then    !'ielevt'
        x=nsdiff(11,noweak(n))
      elseif(inom.eq.342)then                       !'mc1evt' charged particle
        x=multc1                                    ! mult for |eta|<1
      elseif(inom.eq.343)then   !CDF SD trigger 'sdcdf'
        x=isdiff(3)
      elseif(inom.eq.344)then   !CDF DPE trigger 'dpecdf'
        x=isdiff(4)
      elseif(inom.eq.345)then   !CDF DD trigger 'ddcdf'
        x=isdiff(5)
      elseif(inom.eq.346)then   !'phievt'
        x=phievt
      elseif(inom.eq.347)then   !'ndcevt'  CMS hadron level NSD (2011)
        x=nsdiff(12,noweak(n))
      elseif(inom.eq.348)then   !'jetevt' ......  1 = jet event
        x=0
        if(nint(ypara(1,n)).ne.0)x=1
      elseif(inom.eq.349)then   !'epszero (Z for pp)'
        x=epszero
      elseif(inom.eq.350)then  !xsievt:  xsi = (M_X^2/s) 
c (where M_X = sqrt{ (sum E)^2 - (sum vec-p)^2 } with sum E and sum vec-p 
c being resp. the sum of the energies and the sum of the 3-momenta of the 
c generated particles in the event, excluding the proton with the largest 
c laboratory momentum)
        x=xsi
cc xsievt:  xsi = 1-xF_leading
c        x=-2
cc       pmax=sqrt((ecms/2.)**2-prom**2)
c          pmax=pnullx               !???????????????????
c        do i=1,nptl
c          if(abs(idptl(i)).gt.100.and.istptl(i).eq.0)then
c            if(iframe.eq.11)then
c              pz=pptl(3,i)
c            else
c              amt=sqrt(prom**2+pptl(1,i)**2+pptl(2,i)**2)
c              rap=alog((pptl(4,i)+pptl(3,i))/amt)
c     &           -alog((sqrt(pnll**2+ecms**2)+pnll)/ecms)
c              pz=amt*sinh(rap)
c            endif
c            x=max(x,abs(pz/pmax))
c          endif
c        enddo
c        x=max(0.,1.-x)
      elseif(inom.eq.351)then  !xsicms: CMS determination of xsi=M2_X/s
                               !using xsi=Sum [(E+pz)_i/sqrt(s)]      where i runs on every reconstructed particles (= |eta|<4.9 charged + neutral)
        x=-2
c       pmax=sqrt((ecms/2.)**2-prom**2)
        pmax=pnullx               !???????????????????
        Ef=0.
        Eb=0.
        Pf=0.
        Pb=0.
        Esum=0.
        Psum=0.
        do i=1,nptl
          if(istptl(i).eq.0)then
            pt=sqrt(pptl(2,i)**2+pptl(1,i)**2)
            pz=pptl(3,i)
            eta=0.
            if(abs(pz).gt.0..and.pt.gt.0.)eta=sign(1.,pz)*
     *   log((sqrt(pz**2+pt**2)+abs(pz))/pt)
            if(pt.eq.0.)eta=sign(1e5,pz)
            if(eta.ge.4.9)then
              Ef=Ef+pptl(4,i)
              Pf=Pf+pptl(3,i)
            elseif(eta.le.-4.9)then
              Eb=Eb+pptl(4,i)
              Pb=Pb+pptl(3,i)
            else
              Esum=Esum+pptl(4,i)
              Psum=Psum+pptl(3,i)
            endif
c        write(ifch,*)'ici',i,eta,Ef,Eb,Esum,Psum,Ef-Pf,Eb+Pb
          endif
        enddo
        if(Ef.ge.Eb)then
          x=Esum+Psum+Eb+Pb
        else
          x=Esum-Psum+Ef-Pf
        endif
        x=max(0.,x/ecms)
c        write(ifmt,*)'ici',x,min(Esum-Psum,Esum+Psum)/ecms
      elseif(inom.eq.352)then   !'calevt' ......  energy in eta range
        x= ypara(1,n)      
      elseif(inom.eq.353)then   !'fgpevt' ......  max forward rapidity gap in eta range
        x= ypara(2,n)      
      elseif(inom.eq.354)then   !'bgpevt' ......  max backward rapidity gap in eta range
        x= ypara(3,n)      
      elseif(inom.eq.355)then   !'gapevt' ......  max backward rapidity gap in eta range
        x= ypara(4,n)      
      elseif(inom.eq.356)then
        x=sigdd
      elseif(inom.eq.357)then                       
        x=nint(typevt)            !ND(1), DD(2), or SD(3) with fusion < 0
      elseif(inom.eq.358)then   !'ndsevt'  CMS hadron level double sided trigger (2012)
        x=nsdiff(13,noweak(n))
      elseif(inom.eq.359)then    !m24evt ... multipl |eta|<2.4 (CMS)
        x=multc24
      elseif(inom.eq.360)then   !'ndhevt'  CMS hadron level single sided trigger (HF 2012)
        x=nsdiff(14,noweak(n))
      elseif (inom.eq.501) then !---------
        x=sval(1,1)
      elseif (inom.eq.502) then !---------
        x=sval(1,2)
      elseif (inom.eq.503) then !---------
        x=sval(1,3)
      elseif (inom.eq.504) then !---------
        x=sval(1,4)
      elseif(inom.eq.505)then    !'eglevt'  eccentricity
        x=eglevt
      elseif(inom.eq.506)then    !'fglevt'  eccentricity_part
        x=fglevt
      elseif(inom.eq.507)then    !'rglevt'  ratio ng2 / ng1
        x=0
        if(ng1evt.ne.0)
     .  x=ng2evt/float(ng1evt)
      elseif(inom.eq.508)then    !'sglevt'  area S
        x=sglevt
      elseif(inom.eq.509)then                       !'ptrevt'
        x=0
        do i=maproj+matarg+1,minfra
        if(istptl(i).eq.25)then
          pt=sqrt(pptl(1,i)**2+pptl(2,i)**2)
          x=max(x,pt)
        endif
        enddo
      elseif(inom.eq.510)then                       !'rr2evt'
        x=cos(2*(phi2neg-phi2pos))
        !write(ifmt,*)'+++++ rr2evt +++++ ',x,phi2neg,phi2pos
      elseif(inom.eq.511)then  !'perevt'
        call getJKNcentr
        x=(ncentr-0.5)*5
      elseif(inom.eq.512)then  !'paievt'
        x=paievt  
      elseif(inom.eq.513)then  !'co2evt'
        x=co2evt      
      elseif(inom.eq.514)then  !'co3evt'
        x=co3evt
      endif                     !---------------------------------------
      end

c----------------------------------------------------------------------
      subroutine mux(n)
c----------------------------------------------------------------------
      ! input
      !   n = histogram number
      !   xpara(1,n) ... etamin
      !   xpara(2,n) ... etamax
      !   xpara(3,n) ... ptmin
      !   xpara(4,n) ... ptmax
      !   xpara(5,n) ... factor
      !   xpara(6,n) ... divisor
      !   xpara(7,n) ... absolute value of eta (1)
      !
      ! output
      !   ypara(1,n) ... multiplicity
      !--------------------------------------------------------------
      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      mul=0
      do i=maproj+matarg+1,nptl
        if(istptl(i).eq.0)then
          pt=pptl(1,i)**2+pptl(2,i)**2
          pp=sqrt(pptl(1,i)**2+pptl(2,i)**2+pptl(3,i)**2)
          if(pt.gt.0.)then
            pt=sqrt(pt)
            eta=sign(1.,pptl(3,i))*alog((pp+abs(pptl(3,i)))/pt)
          else
            eta=1000.
          endif
          if(xpara(7,n).ge.1.)eta=abs(eta)
          if(abs(idptl(i)).ge.100
     $   .and.abs(idptl(i)).lt.10000)then
            call idchrg(idptl(i),ch)
            if(abs(ch).gt.0.1)then
              if(     eta.ge.xpara(1,n)
     *           .and.eta.le.xpara(2,n)
     *           .and.pt .gt.xpara(3,n)
     *           .and.pt .lt.xpara(4,n)   )mul=mul+1
            endif
          endif
        endif
      enddo
      ypara(1,n)=mul*xpara(5,n)/xpara(6,n) 
      !print*,'+++++++',n,mul
      !. ,xpara(1,n),xpara(2,n),xpara(3,n)
      !. ,xpara(4,n),xpara(5,n),xpara(6,n)
      end

c----------------------------------------------------------------------
      subroutine PairVariables
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (maxpv=20000)
      real phixx(maxpv),etaxx(maxpv),ptxx(maxpv)
      integer idxx(maxpv)
      common/cpairs/paievt,co2evt,co3evt
      common/cpairs2/paipi(40),co2pi(40),co3pi(40)
     .              ,paipr(40),co2pr(40),co3pr(40)
     .            ,ipaipi(40),ico2pi(40),ico3pi(40)
     .            ,ipaipr(40),ico2pr(40),ico3pr(40)
     .            ,maxpt,delpt
      maxpt=40
      delpt=8./maxpt
      do j=1,maxpt
        ipaipi(j)=0
        ico2pi(j)=0
        ico3pi(j)=0
        ipaipr(j)=0
        ico2pr(j)=0
        ico3pr(j)=0
        paipi(j)=0.
        co2pi(j)=0.
        co3pi(j)=0.
        paipr(j)=0.
        co2pr(j)=0.
        co3pr(j)=0.
      enddo
      paievt=0
      co2evt=0
      co3evt=0
      ii=0
      do i=maproj+matarg+1,nptl
       if(istptl(i).eq.0)then
          px=pptl(1,i)
          py=pptl(2,i)
          pt=pptl(1,i)**2+pptl(2,i)**2
          if(pt.gt.0.)then
            pt=sqrt(pt)
            theta=atan(pt/pptl(3,i))
            if(theta.lt.0.)theta=theta+pi
            eta=-log(tan(theta*0.5))
          else
            eta=1000.
          endif
          if(abs(eta).lt.0.8)then
            call idchrg(idptl(i),ch)
            if(abs(ch).gt.0.1)then
              ii=ii+1
              if(ii.gt.maxpv)then
                write(ifmt,*)
     .          '***** ERROR: PairVariables: arrays too small'
                 stop'\n\n PairVariables: arrays too small \n\n'
              endif
              phixx(ii)=polar(px,py)
              etaxx(ii)=eta
              ptxx(ii)=pt
              idxx(ii)=idptl(i)
              !print*,'+++++',ii,pt,idxx(ii)
            endif    
          endif  
        endif
      enddo      
      do m=1,ii
      do n=1,ii 
        if(    (etaxx(m).lt.-0.5 .and. etaxx(n).gt.0.5)
     .    .or. (etaxx(n).lt.-0.5 .and. etaxx(m).gt.0.5) )then
          if(abs(etaxx(m)-etaxx(n)).lt.0.999)
     .    stop'\n\n ERROR 04112011\n\n' 
          !m.ne.n automatic in this case
          paievt=paievt+1
          co2mn=cos(2*(phixx(m)-phixx(n)))
          co3mn=cos(3*(phixx(m)-phixx(n)))
          co2evt=co2evt+co2mn
          co3evt=co3evt+co3mn
          !iam=abs(idxx(m))
          !im=ptxx(m)/delpt+1
          ian=abs(idxx(n))
          in=ptxx(n)/delpt+1
          if(in.ge.1.and.in.le.maxpt)then
            if(ian.eq.120)then
              paipi(in)=paipi(in)+1
              co2pi(in)=co2pi(in)+co2mn
              co3pi(in)=co3pi(in)+co3mn
              !print*,'+++pion++',in, idxx(n),ptxx(n)
            elseif(ian.eq.1120)then
              paipr(in)=paipr(in)+1
              co2pr(in)=co2pr(in)+co2mn
              co3pr(in)=co3pr(in)+co3mn
              !print*,'+++p++',in, idxx(n),ptxx(n)
            endif
          endif
        endif
      enddo
      enddo
      !print*,'+++++ pairs +++++',ii,paievt,co2evt,co3evt
      sum1=0
      sum2=0
      sum3=0
      do j=1,min(20,maxpt)
         !write(*,'(a,i5,2(f7.0,2f7.3,3x))') '+++++++',j,
         !.   paipi(j),co2pi(j),co3pi(j),paipr(j),co2pr(j),co3pr(j)
         sum1=sum1+paipi(j)+paipr(j)
         sum2=sum2+co2pi(j)+co2pr(j)
         sum3=sum3+co3pi(j)+co3pr(j)
      enddo
      !print*,'+++++ sum +++++     ',sum1,sum2,sum3
      end

c----------------------------------------------------------------------
      subroutine StandardVariables
c----------------------------------------------------------------------
      include 'epos.inc'
      common/stavar/multc05,multy1,multc14,multyi,multc3,imulty1,multeb
     &     ,multc1,multc83,multc24,multc25,rapgap,ipairs1,xsi
      parameter(mxxhis=70)
      common/varhis/icorrtrig(0:mxxhis),ihardevent(0:mxxhis)
     &,ijetfind1(0:mxxhis),ijetfind2(0:mxxhis),imux(0:mxxhis)
     &,ifastjet(0:mxxhis),ijetevent(0:mxxhis),icaltrig(0:mxxhis)
      logical CDF
      common/cphi2/phi2pos,phi2neg
      double precision sumE,sumP(4)

      Emax=0.
      Pmax=0.
      multy1=0
      multc05=0
      multc14=0
      multc24=0
      multc25=0
      multc1=0
      multyi=0
      multc3=0
      multeb=0
      multc83=0
      rapgap=0.
      etamn=-1000.
      etamx=1000.
      avcos2p=0
      avsin2p=0
      avcos2n=0
      avsin2n=0
      sumE=0d0
      sumP(1)=0d0
      sumP(2)=0d0
      sumP(3)=0d0
      sumP(4)=0d0
      imax=0

      do i=maproj+matarg+1,nptl
        if(istptl(i).eq.0)then
          amt=pptl(5,i)**2+pptl(1,i)**2+pptl(2,i)**2
          px=pptl(1,i)
          py=pptl(2,i)
          pt=pptl(1,i)**2+pptl(2,i)**2
          pp=sqrt(pptl(1,i)**2+pptl(2,i)**2+pptl(3,i)**2)
          et=sqrt(pptl(1,i)**2+pptl(2,i)**2+pptl(5,i)**2)
          if(amt.gt.0..and.pptl(4,i).gt.0.)then
            amt=sqrt(amt)
            rap=sign(1.,pptl(3,i))*alog((pptl(4,i)+abs(pptl(3,i)))/amt)
          else
            rap=1000.
          endif
          if(pt.gt.0.)then
            pt=sqrt(pt)
            theta=atan(pt/pptl(3,i))
            if(theta.lt.0.)theta=theta+pi
            et=sin(theta)*pptl(4,i)
            eta=-log(tan(theta*0.5))
          else
            eta=1000.
          endif
          if(eta.gt.0.)then
            a=polar(px,py)
            avcos2p=avcos2p+cos(2*a)
            avsin2p=avsin2p+sin(2*a)
          else
            a=polar(px,py)
            avcos2n=avcos2n+cos(2*a)
            avsin2n=avsin2n+sin(2*a)
          endif
          sumE=sumE+dble(pptl(4,i))
          sumP(1)=sumP(1)+dble(pptl(1,i))
          sumP(2)=sumP(2)+dble(pptl(2,i))
          sumP(3)=sumP(3)+dble(pptl(3,i))
          if(idptl(i).eq.idproj.and.pp.gt.Pmax)then
            imax=i
            Pmax=pp
          endif
          if(abs(idptl(i)).ge.100
     $         .and.abs(idptl(i)).lt.10000)then
            call idchrg(idptl(i),ch)
            CDF=.false.
            if(abs(ch).gt.0.1)then
c---multyi---charged ptl multipl
              multyi=multyi+1
c---multy1---charged ptl multipl for central rap
              if(abs(rap).le.1.)multy1=multy1+1
              if(abs(eta).le.0.5)multc05=multc05+1
              if(abs(eta).le.1.and.pt.gt.0.4)multc14=multc14+1
              if(abs(eta).le.0.8.and.pt.gt.0.3
     *        .and.pt.lt.4.)multc83=multc83+1
              if(abs(eta).lt.2.4)multc24=multc24+1
              if(abs(eta).le.2.5.and.pt.gt.0.5)multc25=multc25+1
              if(abs(eta).le.1)multc1=multc1+1
              if(abs(rap).le.3.15)multc3=multc3+1
c---multeb---charged ptl multipl for back rap
              if(eta.gt.-3.8.and.eta.lt.-2.8)multeb=multeb+1
              if(abs(eta).lt.1.2.and.pt.gt.0.3)then
                CDF=.true.      !CDF CTC acceptance
              elseif(abs(eta).gt.3.2.and.abs(eta).lt.5.9)then
                CDF=.true.      !CDF BBC acceptance
              endif
            endif
            if(abs(eta).lt.2.4.and.et.gt.0.2)then
              CDF=.true.     !CDF central and plug calorimeters acceptance
            elseif(abs(eta).gt.2.2.and.abs(eta).lt.4.2.and.et.gt.1.)then
              CDF=.true.     !CDF forward calorimeters acceptance
            endif
            if(CDF)then
              if(eta.le.0)etamn=max(etamn,eta)
              if(eta.ge.0)etamx=min(etamx,eta)
            endif
          endif
          if(ilprtg.eq.1)then
            if((((abs(idptl(i)).gt.1000.and.abs(idptl(i)).lt.10000)
     *         .and.abs(idproj).gt.1000).or.(iabs(idptl(i)).gt.100
     *         .and.abs(idproj).lt.1000)).and.pptl(4,i)
     *         .gt.Emax.and.pptl(3,i).gt.0.)then
              Emax=pptl(4,i)
              idlead=i
            endif
          else
            if(abs(idptl(i)).gt.1000.and.abs(idptl(i)).lt.10000
     *        .and.pptl(4,i).gt.Emax.and.pptl(3,i).lt.0.)then
              Emax=pptl(4,i)
              idlead=i
            endif
          endif
        endif
      enddo
      if(imax.gt.0)then
        sumE=sumE-dble(pptl(4,imax))
        sumP(1)=sumP(1)-dble(pptl(1,imax))
        sumP(2)=sumP(2)-dble(pptl(2,imax))
        sumP(3)=sumP(3)-dble(pptl(3,imax))
      endif
      sumP(4)=sqrt(sumP(1)**2+sumP(2)**2+sumP(3)**2)
      xsi=sngl((sumE+sumP(4))*(sumE-sumP(4))/dble(engy)**2)
      rapgap=etamx-etamn
      if(rapgap.gt.100)rapgap=-1.    !not defined
      phi2pos=polar(avcos2p,avsin2p)/2.
      phi2neg=polar(avcos2n,avsin2n)/2.
      !write(ifmt,*)'+++++ phi2pos phi2neg +++++ '
      !. ,phi2pos, phi2neg
      end

c----------------------------------------------------------------------
      subroutine jetfind(m,n)
c----------------------------------------------------------------------
c   m = 1 ou 2 (two different definitions)
c   n = histogram
c input(jet definition):
c   xpara(1,n) ... output (m=1) (0=et, 1=pt)
c   xpara(2,n) ... etamin (m=1)
c   xpara(3,n) ... etamax (m=1)
c   xpara(4,n) ... rmax   (m=1) (rmax defining the cone)
c   xpara(5,n) ... ichd   (m=1) (1=charged, 0=all)
c   xpara(6,n) ... output (m=2) (0=et, 1=pt)
c   xpara(7,n) ... etamin (m=2)
c   xpara(8,n) ... etamax (m=2)
c   xpara(9,n) ... rmax   (m=2)
c   xpara(10,n) .. ichd   (m=2)
c output (jet properties):
c   ypara(1,n) ... 1 (found) or 0 if not  (m=1)
c   ypara(2,n) ... et or pt               (m=1)
c   ypara(3,n) ... eta of center          (m=1)
c   ypara(4,n) ... phi of center          (m=1)
c   ypara(5,n)
c   ypara(6,n) ... 1 (found) or 0 if not  (m=2)
c   ypara(7,n) ... et or pt               (m=2)
c   ypara(8,n) ... eta of center          (m=2)
c   ypara(9,n) ... phi of center          (m=2)
c   ypara(10,n)
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      parameter (mxval=5)
      real ptx(mxval),lst(mxval),etax(mxval),phix(mxval)
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)

      if(m.ne.1.and.m.ne.2)stop'jetfind: value of m not valid.      '

      ipt   = nint(xpara(1+5*(m-1),n))
      etamin=           xpara(2+5*(m-1),n)
      etamax=           xpara(3+5*(m-1),n)
      rmax  =           xpara(4+5*(m-1),n)
      ichd  = nint(xpara(5+5*(m-1),n))

      ifound=0
      do l=1,mxval
        ptx(l)=0
        lst(l)=0
        etax(l)=0
        phix(l)=0
      enddo

ctp060829      pp1=0
ctp060829      pp2=0
ctp060829      pp3=0

      do i=maproj+matarg+1,nptl
        iok=0
        if(istptl(i).eq.0.and.abs(idptl(i)).lt.10000)iok=1
        if(iok.eq.1)call idchrg(idptl(i),ch)
        if(ichd.eq.1.and.nint(ch).eq.0)iok=0
        if(iok.eq.1)then
          p1=pptl(1,i)
          p2=pptl(2,i)
          p3=pptl(3,i)
          pt=sqrt(p1**2+p2**2)
                if(pt.gt.0)then
            eta=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
            phi=sign(1.,p2)*acos(p1/pt)
          else
            eta=10000
            phi=0
          endif
          do k=1,mxval
            iok=1
            if(m.eq.2)then
              dphi=phi-ypara(4,n)
              if(dphi.lt.-pi)dphi=dphi+2*pi
              if(dphi.gt. pi)dphi=dphi-2*pi
              if(abs(dphi).lt.pi/2)iok=0
            endif
            if(iok.eq.1.and.pt.gt.ptx(k)
     &        .and.eta.le.etamax.and.eta.ge.etamin)then
              do l=mxval,k+1,-1
               ptx(l)=ptx(l-1)
               lst(l)=lst(l-1)
               etax(l)=etax(l-1)
               phix(l)=phix(l-1)
              enddo
               ptx(k)=pt
               lst(k)=i
               etax(k)=eta
               phix(k)=phi
              goto2
            endif
          enddo
  2       continue
        endif
      enddo

      kk=0
      etx=0

      do k=1,mxval
       if(lst(k).ne.0)then

        ifound=1
        et=0
        etaxx=etax(k)
        phixx=phix(k)
        do j=maproj+matarg+1,nptl
          iok=0
          if(istptl(j).eq.0.and.abs(idptl(j)).lt.10000)iok=1
          if(iok.eq.1)call idchrg(idptl(j),ch)
          if(ichd.eq.1.and.nint(ch).eq.0)iok=0
          if(iok.eq.1)then
            p1=pptl(1,j)
            p2=pptl(2,j)
            p3=pptl(3,j)
            pt=sqrt(p1**2+p2**2)
            am=pptl(5,j)
                  if(pt.gt.0)then
              eta=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
              phi=sign(1.,p2)*acos(p1/pt)
            else
              eta=-10000
              phi=0
            endif
            if(eta.le.etamax.and.eta.ge.etamin)then
              deta=eta-etaxx
              dphi=phi-phixx
              if(dphi.lt.-pi)dphi=dphi+2*pi
              if(dphi.gt. pi)dphi=dphi-2*pi
              if(deta**2+dphi**2.lt.rmax**2)then
                if(ipt.eq.0)then           !output is et
                  et=et+sqrt(pt**2+am**2)
                else                       !output is pt
                  et=et+pt
                endif
              endif
            endif
          endif
        enddo
        if(et.gt.etx)then
          etx=et
          kk=k
        endif

       endif
      enddo

      ypara(1+5*(m-1),n)=ifound
      ypara(2+5*(m-1),n)=etx
      if(kk.gt.0)then
       ypara(3+5*(m-1),n)=etax(kk)
       ypara(4+5*(m-1),n)=phix(kk)
      endif
      return
      end


c----------------------------------------------------------------------
      subroutine fastjet(n)
c----------------------------------------------------------------------
c   n = histogram (to define istptl=100*n of id=9999)
c input(jet definition):
c   xpara(1,n) ... algorithm (1.0=kt, 0.0=Cam/Aachen,  -1.0 = anti-kt)
c   xpara(2,n) ... rapmin (for particles used to define jet)
c   xpara(3,n) ... rapmax (for particles used to define jet)
c   xpara(4,n) ... rmax   (rmax defining the cone)
c   xpara(5,n) ... ichd   (1=charged, 0=all)
c output : new particles (jets four momentum) in particle list triggered
c          by : trigger istptl jet jet
c          can be used as usual particle to plot pt, phi, etc ...
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      parameter (mxval=10000)
      double precision p(4,mxval), rmax, algo,jets(4,mxval)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)

      algo  = dble(xpara(1,n))
      rapmin=      xpara(2,n)
      rapmax=      xpara(3,n)
      rmax  = dble(xpara(4,n))
      ichd  = nint(xpara(5,n))

      npart=0
      do i=maproj+matarg+1,nptl
        if(istptl(i).eq.0.and.abs(idptl(i)).lt.10000)then
          call idchrg(idptl(i),ch)
          if(nint(ch).ne.0.or.ichd.eq.0)then
            p1=pptl(1,i)
            p2=pptl(2,i)
            p3=pptl(3,i)
            p4=pptl(4,i)
            p5=pptl(5,i)
            amt=p5**2+p1**2+p2**2
            if(amt.gt.0..and.p4+abs(p3).gt.0.)then 
              amt=sqrt(amt)
              rap=sign(1.,p3)*log((p4+abs(p3))/amt) !not correct if particles off-shell
c             rap=0.5*log((p4+p3)/(p4-p3))  !always correct but numerically unstable
            else
              rap=0.         
            endif
            if(rap.ge.rapmin.and.rap.le.rapmax)then  !particle used for jet
              npart=npart+1
              if(npart.gt.mxval)then
        write(ifmt,*)'Too many particles (mxval) for Fastjet ! skip ...'
                return
              endif
              p(1,npart)=dble(p1)
              p(2,npart)=dble(p2)
              p(3,npart)=dble(p3)
              p(4,npart)=dble(p4)
            endif
          endif
        endif
      enddo

      if(npart.gt.0)then
c.....run the clustering with a pp generalised-kt sequential recombination alg
        call fastjetppgenkt(p,npart,rmax,algo,jets,njets) ! ... now you have the jets
        if(njets.gt.0)then
          nptl0=nptl+1
          do i=1,njets
            nptl=nptl+1
            istptl(nptl)=100*n     !used for trigger
            idptl(nptl)=9999       !jet ID
            pptl(1,nptl)=sngl(jets(1,i))  !jet px momentum
            pptl(2,nptl)=sngl(jets(2,i))  !jet py momentum
            pptl(3,nptl)=sngl(jets(3,i))  !jet pz momentum
            pptl(4,nptl)=sngl(jets(4,i))  !jet E  momentum
            p5x=sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2)  !jet Et
            p5=0
            p5= pptl(4,nptl)**2
     .      -pptl(1,nptl)**2-pptl(2,nptl)**2-pptl(3,nptl)**2
            if(p5.gt.0)p5=sqrt(p5)
            pptl(5,nptl)=p5
            !print*,'+++++',p5,p5x
c            write(ifch,*)'fastjet',i,njets,nptl,idptl(nptl),istptl(nptl)
c     &         ,sqrt(pptl(1,nptl)**2+pptl(2,nptl)**2)
          enddo
          if(ish.ge.5)call alist('list after fastjet&',nptl0,nptl)
        endif
      endif


      return
      end


c----------------------------------------------------------------------
      subroutine jetevent(n)
c----------------------------------------------------------------------
c   n = histogram
c input(jet event conditions) using particles (jets) found with fastjet:
c common parameters with fastjet
c   xpara(1,n) ... algorithm (1.0=kt, 0.0=Cam/Aachen,  -1.0 = anti-kt)
c   xpara(2,n) ... rapmin (for particles used to define jet)
c   xpara(3,n) ... rapmax (for particles used to define jet)
c   xpara(4,n) ... rmax   (rmax defining the cone)
c   xpara(5,n) ... ichd   (1=charged, 0=all)
c   xpara(6,n) ... Et_min for event selection
c   xpara(7,n) ... number of jets needed for event selection
c   xpara(8,n) ... delta phi needed for dijet event selection (if xpara(7,n)=2)
c                  if 0, not used, >0 uses delta_phi-pi, <0 uses delta_phi
c output (jet event found or not):
c   ypara(1,n) ... 1 (found) or 0 if not
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      dimension inumj(10000)

      ypara(1,n)=0
      njet=nint(xpara(7,n))
      numj=0
c count the number of jets with Et>Et_min in the event
      do i=maproj+matarg+1,nptl
       if(istptl(i).eq.100*n.and.pptl(5,i).ge.xpara(6,n))then
         numj=numj+1
         if(numj.le.10000)then
c save position of jet in particle list
           inumj(numj)=i
         else
           write(ifmt,*)"Too many jets in jetevent, last are skipped!"
         endif
       endif
      enddo
c      write(ifch,*)"jetevent",numj,abs(xpara(8,n)),inumj(1),inumj(2)
c if enough jets, analyse them
      if(numj.ge.njet)then
c dijet selection
        if(abs(xpara(8,n)).gt.0)then
c test delta phi for 2 higher pt jets (fastjet provide jet list ordered by pt)
          j=inumj(1)
          p1=pptl(1,j)
          p2=pptl(2,j)
          pt=pptl(5,j)
          if(pt.gt.0)then
            phi1=sign(1.,p2)*acos(p1/pt)
          else
            phi1=0.
          endif
          j=inumj(2)
          p1=pptl(1,j)
          p2=pptl(2,j)
          pt=pptl(5,j)
          if(pt.gt.0)then
            phi2=sign(1.,p2)*acos(p1/pt)
          else
            phi2=0.
          endif
c      write(ifch,*)"jetevent phi",pptl(5,inumj(1)),pptl(5,inumj(2))
c     &                       ,phi1,phi2
c test delta phi 
          phi0=pi
          if(xpara(8,n).lt.0.)phi0=0.
          if(abs(abs(phi1-phi2)-phi0).le.abs(xpara(8,n)))then
            ypara(1,n)=1        !event selected (dijet)
          endif
        else                    !event selected (enough jet with Et>Etmin)
          ypara(1,n)=1
        endif
      endif
      return
      end

c----------------------------------------------------------------------
      subroutine hardevent(n)
c----------------------------------------------------------------------
c   n = histogram
c input(jet event conditions):
c   xpara(2,n) ... pt1
c   xpara(3,n) ... pt2
c   xpara(4,n) ... absetamax
c   xpara(5,n) ... rmax        (r=sqrt(deltaeta**2+deltaphi**2))
c   xpara(6,n) ... Et_min
c output (jet event found or not):
c   ypara(1,n) ... 1 (found) or 0 if not
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)

      ypara(1,n)=0
      do i=maproj+matarg+1,nptl
       if(abs(idptl(i)).ge.100.and.abs(idptl(i)).lt.10000.
     $  and.istptl(i).eq.0)then
        call idchrg(idptl(i),ch)
        if(abs(ch).gt.0.1)then
          p1=pptl(1,i)
          p2=pptl(2,i)
          p3=pptl(3,i)
          pt=sqrt(p1**2+p2**2)
          if(pt.gt.0)then
            eta=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
            phi=sign(1.,p2)*acos(p1/pt)
          else
            eta=10000
            phi=0
          endif
          if(pt.ge.xpara(2,n).and.abs(eta).lt.xpara(4,n))then
            et1=pptl(4,i)*pt/sqrt(p3**2+pt**2)
            do j=maproj+matarg+1,nptl
              if(j.ne.i
     $        .and.abs(idptl(j)).ge.100.and.abs(idptl(j)).lt.10000.
     $        .and.istptl(j).eq.0)then
                call idchrg(idptl(j),ch)
                if(abs(ch).gt.0.1.and.abs(idptl(j)).ge.100
     $          .and.abs(idptl(j)).lt.10000.and.istptl(j).eq.0)then
                  p1=pptl(1,j)
                  p2=pptl(2,j)
                  p3=pptl(3,j)
                  pt=sqrt(p1**2+p2**2)
                        if(pt.gt.0)then
                   etax=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
                   phix=sign(1.,p2)*acos(p1/pt)
                  else
                    etax=-10000
                    phix=0
                  endif
                  if(pt.ge.xpara(3,n).and.abs(etax).lt.xpara(4,n))then
                    deta=eta-etax
                    dphi=phi-phix
                    if(dphi.lt.-pi)dphi=dphi+2*pi
                    if(dphi.gt. pi)dphi=dphi-2*pi
                    if(deta**2+dphi**2.lt.xpara(5,n)**2)then
                    et2=pptl(4,j)*pt/sqrt(p3**2+pt**2)
                     if(et1+et2.gt.xpara(6,n))then
                      ypara(1,n)=1
                      goto1
                     endif
                    endif
                  endif
                endif
              endif
            enddo
          endif
        endif
       endif
      enddo

   1  continue
      return
      end

c----------------------------------------------------------------------
      subroutine corrtrig(n)
c----------------------------------------------------------------------
c   n = histogram
c input(trigger conditions):
c   xpara(1,n) ... mode (0,1)
c   xpara(2,n) ... ptmin
c   xpara(3,n) ... ptmax
c   xpara(4,n) ... etamin
c   xpara(5,n) ... etamax
c   xpara(6,n) ...
c   xpara(7,n) ...
c output (triggered particle (the one with highest pt if there are several)):
c   ypara(1,n) ... iptl or 0 if no particle found
c   ypara(2,n) ... phi of particle
c   ypara(3,n) ... phi_null
c   ypara(4,n) ... pt lead
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)

      pt0=xpara(2,n)

      do i=maproj+matarg+1,nptl
       if(abs(idptl(i)).ge.100.and.abs(idptl(i)).lt.10000.
     $  and.istptl(i).eq.0)then
        call idchrg(idptl(i),ch)
        if(abs(ch).gt.0.1)then
          p1=pptl(1,i)
          p2=pptl(2,i)
          p3=pptl(3,i)
          pt=sqrt(p1**2+p2**2)
          pt=max(pt,1e-20)
          eta=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
          phi=sign(1.,p2)*acos(p1/pt)
          if(pt.ge.pt0.and.pt.le.xpara(3,n).and.eta.gt.xpara(4,n)
     $      .and.eta.lt.xpara(5,n))then
            pt0=pt
            ypara(1,n)=i
            ypara(2,n)=phi
            ypara(4,n)=pt
          endif
        endif
       endif
      enddo
      ypara(3,n)=-0.5
      if(nint(xpara(1,n)).eq.1)ypara(3,n)=0.0

      return
      end

c----------------------------------------------------------------------
      subroutine caltrig(n)
c----------------------------------------------------------------------
c   n = histogram
c input(trigger conditions):
c   xpara(1,n) ... mode (0,1,2) (one eta side, 2 eta side independently or, 
c                                2 eta side simultaneously)
c   xpara(2,n) ... etamin
c   xpara(3,n) ... etamax
c   xpara(4,n) ... 0 all or 1 charged or 2 charged + photons
c   xpara(5,n) ... ptmin
c   xpara(6,n) ... ptmax
c output (triggered energy):
c   ypara(1,n) ... max E for a particle in eta range (per side)
c----------------------------------------------------------------------

      include 'epos.inc'
      parameter (mxhis=500,mxcontr=500,mxidcd=60,mxtri=50,mxbin=405)
      parameter (mypara=10,mxpara=10)
      logical ilog,icnx,itrevt,idmod
      double precision bin,bbin,zcbin,zbbin
      common/bins/bin(mxbin,2,mxhis),zcbin(mxbin,2,mxhis)
     $     ,bbin(mxbin,2,mxcontr),itrevt(mxhis),zbbin(mxbin,2,mxcontr)
     $     ,nac(mxhis),ilog(mxhis),icnx(mxhis),xinc(mxhis),ncevt(mxhis)
     $     ,sval(2,mxhis),valtri(mxtri,mxhis),ntrc(mxtri,mxhis)
     $     ,xmin(mxhis),xmax(mxhis),nhis,noweak(mxhis)
     $     ,ivar(2,mxhis),inorm(mxhis),nbin(mxhis),nidcod(mxhis)
     $     ,idcod(mxidcd,mxhis),idmod(mxidcd,mxhis),ntri(mxhis)
     $     ,itri(mxtri,mxhis),xmitri(mxtri,mxhis),xmatri(mxtri,mxhis)
     $     ,xmitrp(mxtri,mxhis),xmatrp(mxtri,mxhis),xpara(mxpara,mxhis)
     $     ,ypara(mypara,mxhis),lookcontr(mxhis)
     $     ,lookcontrx(mxhis),ncontrall,icontrtyp(mxhis),nccevt(mxcontr)
      logical cont
      double precision Eforw,Eback

      mode=nint(xpara(1,n))
      etamin=xpara(2,n)
      etamax=xpara(3,n)
      ptmin=xpara(5,n)
      ptmax=xpara(6,n)
      ichrd=nint(xpara(4,n))
      Eforw=0.d0
      Eback=0.d0
      etaf=1000.
      etab=1000.

      do i=maproj+matarg+1,nptl
       if(istptl(i).eq.0)then
        call idchrg(idptl(i),ch)
        if(ichrd.ge.1)then
          cont=abs(ch).gt.0.1.or.(idptl(i).eq.10.and.ichrd.eq.2)
        else
          cont=.true.
        endif
        if(cont)then
          p1=pptl(1,i)
          p2=pptl(2,i)
          p3=pptl(3,i)
          pt=sqrt(p1**2+p2**2)
          pt=max(pt,1e-20)
          if(pt.gt.ptmin.and.pt.lt.ptmax)then
            eta=sign(1.,p3)*alog((sqrt(p3**2+pt**2)+abs(p3))/pt)
c          phi=sign(1.,p2)*acos(p1/pt)
            if(eta.gt.etamin.and.eta.lt.etamax)
     *           Eforw=max(Eforw,dble(pptl(4,i)))
            if(eta.lt.-etamin.and.eta.gt.-etamax)
     *           Eback=max(Eback,dble(pptl(4,i)))
            if(eta.le.etamax)etaf=min(etaf,etamax-eta)
            if(eta.ge.etamin)etab=min(etab,eta-etamin)
          endif
        endif
       endif
      enddo
      if(mode.eq.0)then
        ypara(1,n)=sngl(Eforw)
      elseif(mode.eq.1)then
        ypara(1,n)=sngl(max(Eforw,Eback))
      elseif(mode.eq.2)then
        ypara(1,n)=sngl(min(Eforw,Eback))
      endif
      ypara(2,n)=etaf
      ypara(3,n)=etab
      ypara(4,n)=max(etab,etaf)
c      if(typevt.eq.3)print *,mode, ypara(1,n),Eforw,Eback,etaf,etab
c      print *,typevt,mode, ypara(1,n),Eforw,Eback,etaf,etab


      return
      end

c----------------------------------------------------------------------
