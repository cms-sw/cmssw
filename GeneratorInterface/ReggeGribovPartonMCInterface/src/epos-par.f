c 15/02/2005 epos 1.03


c----------------------------------------------------------------------
      subroutine paramini(imod)
c----------------------------------------------------------------------
c  Set  parameters of the parametrisation of the eikonals.
c
c xDfit=Sum(i=0,1)(alpD(i)*xp**betDp(i)*xm**betDpp(i)*s**betD(i)
c                            *xs**(gamD(i)*b2)*exp(-b2/delD(i))
c
c Parameters stored in /Dparam/ (epos.inc)
c if imod=0, do settings only for iclpro, if imod=1, do settings
c for iclpro=2 and iclpro
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision PhiExact,y!,Dsoftshval
      call utpri('parini',ish,ishini,3)

c Initialisation of the variables

      call Class('paramini  ')

c Variables used for xparg (only graphical application)

      spmin=4.*q2min         !Definition of spmin in psvin (epos-sha)
      sfshlim=3.*spmin          !transition energy for soft->hard
      emaxDf=engy            !energy for fit subroutines
      smaxDf=emaxDf**2       !energy squared for fit subroutines
c      sfshlim=100.
      nptf=10                !number of point for the fit
      xmaxDf=1.d0            !maximum limit for the fit
      xminDf=1.d0/dble(smaxDf)   !minimum limit
      xggfit=xminDf*sfshlim
      xshmin=3.d-2           !minimum limit for sh variance fit
      if(smaxDf.lt.sfshlim) xshmin=1.d0
      xfitmin=0.1d0   !sh fitted between f(1) and f(1)*xfitmin
      if(smaxDf.lt.sfshlim) xfitmin=1.d0

      bmaxDf=2.              !maximum b for variance fit

      idxDmin=idxD0          !minimum indice for parameters
      ntymin=ntymi           !minimum indice for diagram type

      ucfpro=utgam1(1.+alplea(iclpro))
      ucftar=utgam1(1.+alplea(icltar))



      iiiclegy=iclegy

c for pi or K - p crosse section calculation, we need alpD, ... for
c iclpro=1 or 3 and iclpro=2
        iiiclpro1=iclpro
        iiidirec=1
        if(imod.eq.0)then
          iiiclpro2=iclpro
        else
          iiiclpro2=2
          if(iiiclpro1.lt.iiiclpro2)iiidirec=-1
        endif
        iclprosave=iclpro

        do iiipro=iiiclpro2,iiiclpro1,iiidirec

          iclpro=iiipro

      if(ish.ge.4)write(ifch,*)'gamini'
     &          ,iscreen,iclpro,icltar,iclegy,smaxDf,sfshlim,spmin

      if(isetcs.le.1)then                      !if set mode, do fit

c First try for fit parameters

c linear fit of the 2 components of G as a function of x
        call pompar(alpsf,betsf,0) !soft (taking into account hard)
        call pompar(alpsh,betsh,1) !sh
c        betsh=0.31
c        alpsh0=sngl(Dsoftshval(smaxDf,1d0,0.d0,0.,2))*smaxDf**(-betsh)
c        alpsh1=sngl(Dsoftshval(smaxDf,1d0,0.d0,0.,0))*smaxDf**(-betsh)
c        if(alpsh0.lt.alpsf)alpsh1=alpsh0
c        if(alpsh0*smaxDf**betsh.lt.alpsf*smaxDf**betsf)alpsh1=alpsh0
c        if(smaxDf.gt.100.*sfshlim)then
c          xfmin=1.e-2
c          alpsh2=sngl(Dsoftshval(xfmin*smaxDf,dble(xfmin),0.d0,0.,0))
c     &             *(xfmin*smaxDf)**(-betsh)
c        else
c          alpsh2=-1e10
c        endif
c        alpsh=max(alpsh1,alpsh2)
c        alpsh=max(alpsh0,alpsh2)
c Gaussian fit of the 2 components of G as a function of x and b
        call variance(delsf,gamsf,0)
        call variance(delsh,gamsh,1)
        gamsf=max(0.,gamsf)
        gamsh=max(0.,gamsh)


c Fit GFF


c       fit parameters
        numminDf=3             !minimum indice
        numparDf=4             !maximum indice
        betac=100.               !temperature for chi2
        betae=1.               !temperature for error
        fparDf=0.8             !variation amplitude for range

        nmcxDf=20000          !number of try for x fit

c starting values

        parDf(1,3)=alpsf
        parDf(2,3)=betsf
        parDf(3,3)=alpsh
        parDf(4,3)=betsh

c        if(smaxDf.ge.sfshlim)then
c          call paramx           !alpD and betD

c          call pompar(alpsf,betsf,-1) !soft (taking into account hard)
c          parDf(1,3)=alpsf
c          parDf(2,3)=max(-0.95+alppar,betsf)
c          parDf(2,3)=max(0.,betsf)

c        endif

        alpsf=parDf(1,3)
        betsf=parDf(2,3)
        alpsh=parDf(3,3)
        betsh=parDf(4,3)


      else                                     !else parameters from table (inirj)
        nbpsf=iDxD0
        if(iclegy2.gt.1)then
          al=1.+(log(engy)-log(egylow))/log(egyfac) !energy class
          i2=min(iiiclegy+1,iclegy2)
          i1=i2-1
        else
          i1=iclegy
          i2=iclegy
          al=float(iclegy)
        endif
        dl=al-i1
        dl1=max(0.,1.-dl)
                                !linear interpolation
        alpsf=alpDs(nbpsf,i2,iclpro,icltar)*dl
     &       +alpDs(nbpsf,i1,iclpro,icltar)*dl1
        alpsh=alpDs(1,i2,iclpro,icltar)*dl
     &       +alpDs(1,i1,iclpro,icltar)*dl1
        betsf=betDs(nbpsf,i2,iclpro,icltar)*dl
     &       +betDs(nbpsf,i1,iclpro,icltar)*dl1
        betsh=betDs(1,i2,iclpro,icltar)*dl
     &       +betDs(1,i1,iclpro,icltar)*dl1
        gamsf=gamDs(nbpsf,i2,iclpro,icltar)*dl
     &       +gamDs(nbpsf,i1,iclpro,icltar)*dl1
        gamsh=gamDs(1,i2,iclpro,icltar)*dl
     &       +gamDs(1,i1,iclpro,icltar)*dl1
        delsf=delDs(nbpsf,i2,iclpro,icltar)*dl
     &       +delDs(nbpsf,i1,iclpro,icltar)*dl1
        delsh=delDs(1,i2,iclpro,icltar)*dl
     &       +delDs(1,i1,iclpro,icltar)*dl1


c For the Plots
        parDf(1,3)=alpsf
        parDf(2,3)=betsf
        parDf(3,3)=alpsh
        parDf(4,3)=betsh

      endif

c     if energy too small to have semi-hard interaction -> only soft diagram

      if(smaxDf.lt.sfshlim.and.idxD0.eq.0)then !no hard: soft+hard=soft
        alpsf=alpsf/2.
        alpsh=alpsf
        betsh=betsf
        gamsh=gamsf
        delsh=delsf
      endif



c Print results

      if(ish.ge.4)then
        write(ifch,*)"parameters for iclpro:",iclpro
        write(ifch,*)"alp,bet,gam,del sf:",alpsf,betsf,gamsf,delsf
        write(ifch,*)"alp,bet,gam,del sh:",alpsh,betsh,gamsh,delsh
      endif


c Record parameters


      alpD(idxD0,iclpro,icltar)=alpsf
      alpDp(idxD0,iclpro,icltar)=0.
      alpDpp(idxD0,iclpro,icltar)=0.
      betD(idxD0,iclpro,icltar)=betsf
      betDp(idxD0,iclpro,icltar)=betsf
      betDpp(idxD0,iclpro,icltar)=betsf
      gamD(idxD0,iclpro,icltar)=gamsf
      delD(idxD0,iclpro,icltar)=delsf

      alpD(1,iclpro,icltar)=alpsh
      alpDp(1,iclpro,icltar)=0.
      alpDpp(1,iclpro,icltar)=0.
      betD(1,iclpro,icltar)=betsh
      betDp(1,iclpro,icltar)=betsh
      betDpp(1,iclpro,icltar)=betsh
      gamD(1,iclpro,icltar)=gamsh
      delD(1,iclpro,icltar)=delsh

      if(iomega.lt.2.and.alpdif.ne.1.)then
        alpDp(2,iclpro,icltar)=0.
        alpDpp(2,iclpro,icltar)=0.
        betD(2,iclpro,icltar)=0. !max(0.,betsf)
        alpdifs=alpdif
c        alpdif=0.99
        betDp(2,iclpro,icltar)=-alpdifs+alppar
        betDpp(2,iclpro,icltar)=-alpdifs+alppar
c        alpD(2,iclpro,icltar)=(alpsf+alpsh)*wdiff(iclpro)*wdiff(icltar)
        alpD(2,iclpro,icltar)=wdiff(iclpro)*wdiff(icltar)
     &            /utgam1(1.-alpdifs)**2
     &            *utgam1(2.-alpdifs+alplea(iclpro))
     &            *utgam1(2.-alpdifs+alplea(icltar))
     &            /chad(iclpro)/chad(icltar)
c        alpdif=alpdifs
        gamD(2,iclpro,icltar)=0.
        delD(2,iclpro,icltar)=4.*.0389*(gwidth*(r2had(iclpro)
     &                    +r2had(icltar))+slopoms*log(smaxDf))
      else
        alpD(2,iclpro,icltar)=0.
        alpDp(2,iclpro,icltar)=0.
        alpDpp(2,iclpro,icltar)=0.
        betD(2,iclpro,icltar)=0.
        betDp(2,iclpro,icltar)=0.
        betDpp(2,iclpro,icltar)=0.
        gamD(2,iclpro,icltar)=0.
        delD(2,iclpro,icltar)=1.
      endif
      if(ish.ge.4)write(ifch,*)"alp,bet,betp,del dif:"
     &   ,alpD(2,iclpro,icltar),betD(2,iclpro,icltar)
     &   ,betDp(2,iclpro,icltar),delD(2,iclpro,icltar)


      bmxdif(iclpro,icltar)=conbmxdif()     !important to do it before kfit,  because it's used in.
c          call Kfit(-1)    !xkappafit not used : if arg=-1, set xkappafit to 1
      if(isetcs.le.1)then
        if(isetcs.eq.0)then
          call Kfit(-1)         !xkappafit not used : if arg=-1, set xkappafit to 1)
        else
c          call Kfit(-1)    !xkappafit not used : if arg=-1, set xkappafit to 1)
          call Kfit(iclegy)
        endif
c for plots record alpDs, betDs, etc ...
        alpDs(idxD0,iclegy,iclpro,icltar)=alpsf
        alpDps(idxD0,iclegy,iclpro,icltar)=0.
        alpDpps(idxD0,iclegy,iclpro,icltar)=0.
        betDs(idxD0,iclegy,iclpro,icltar)=betsf
        betDps(idxD0,iclegy,iclpro,icltar)=betsf
        betDpps(idxD0,iclegy,iclpro,icltar)=betsf
        gamDs(idxD0,iclegy,iclpro,icltar)=gamsf
        delDs(idxD0,iclegy,iclpro,icltar)=delsf

        alpDs(1,iclegy,iclpro,icltar)=alpsh
        alpDps(1,iclegy,iclpro,icltar)=0.
        alpDpps(1,iclegy,iclpro,icltar)=0.
        betDs(1,iclegy,iclpro,icltar)=betsh
        betDps(1,iclegy,iclpro,icltar)=betsh
        betDpps(1,iclegy,iclpro,icltar)=betsh
        gamDs(1,iclegy,iclpro,icltar)=gamsh
        delDs(1,iclegy,iclpro,icltar)=delsh
      endif

      enddo

      if(iclpro.ne.iclprosave)stop'problem in parini with iclpro'

c initialize some variable for screening
      if(iscreen.ne.0)then
        fegypp=epscrw*fscra(engy/egyscr)
        b2xscr=2.*epscrp*4.*.0389
     &       *(r2had(iclpro)+r2had(icltar)+slopom*log(engy**2))
c caculate the radius where Z is saturated at epscrx to define the bases
c of nuclear shadowing
        satrad=0.
        if(fegypp.gt.0.)satrad=-b2xscr*log(epscrx/fegypp)
        bglaubx=zbrads*sqrt(max(0.1,satrad))
        zbcutx=zbcut*bglaubx
c        print *,'---->',bglaubx,zbcutx
      endif

      if(ish.ge.4)then             !check PhiExact value for x=1
        y=Phiexact(0.,0.,1.,1.d0,1.d0,smaxDf,0.)
        write(ifch,*)'PhiExact=',y
      endif

      call utprix('parini',ish,ishini,3)

      return
      end

c----------------------------------------------------------------------
      subroutine Class(text)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      parameter (eps=1.e-5)    !to correct for precision problem)
      character*10 text
      if(iclegy1.eq.iclegy2)then
        iclegy=iclegy1
      else
      iclegy=1+int( (log(engy)-log(egylow))/log(egyfac) + eps ) !energy class
      if(iclegy.gt.iclegy2)then
         write(ifch,*)'***********************************************'
         write(ifch,*)'Warning in ',text
         write(ifch,*)'Energy above the range used for the fit of D:'
         write(ifch,*)egylow*egyfac**(iclegy1-1),egylow*egyfac**iclegy2
         write(ifch,*)'***********************************************'
         iclegy=iclegy2
      endif
      if(iclegy.lt.iclegy1)then
         write(ifch,*)'***********************************************'
         write(ifch,*)'Warning in ',text
         write(ifch,*)'Energy below the range used for the fit of D:'
         write(ifch,*)egylow*egyfac**(iclegy1-1),egylow*egyfac**iclegy2
         write(ifch,*)'***********************************************'
         iclegy=iclegy1
      endif
      endif
      end

c----------------------------------------------------------------------
      subroutine param
c----------------------------------------------------------------------
c  Set the parameter of the parametrisation of the eikonale.
c  We group the parameters into 4 array with a dimension of idxD1(=1)
c  to define xDfit (see below).
c
c xDfit=Sum(i,0,1)(alpD(i)*xp**betDp(i)*xm**betDpp(i)*s**betD(i)
c                            *xs**(gamD(i)*b2)*exp(-b2/delD(i))
c
c subroutine used for tabulation.
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

c Initialisation of the variables

      emaxDf=egyfac**(iclegy-1)*egylow
      smaxDf=emaxDf**2

      spmin=4.*q2min         !Definition of spmin in psvin (epos-sha)
      sfshlim=3.*spmin
      nptf=10
      xmaxDf=1.d0
      xminDf=1d0/dble(smaxDf)
      xshmin=3.d-2           !minimum limit for sh variance fit
      if(smaxDf.lt.sfshlim) xshmin=1.d0
      xfitmin=0.1d0   !sh fitted between f(1) and f(1)*xfitmin
      if(smaxDf.lt.sfshlim) xfitmin=1.d0
      bmaxDf=2.


      if(idxD0.ne.0.and.idxD1.ne.1) stop "* idxD0/1 are not good! *"

      engytmp=engy
      engy=emaxDf

c Initialisation of the parameters

      do i=1,nbpf
        do j=1,4
          parDf(i,j)=1.
        enddo
      enddo

c.......Calculations of the parameters

c First try for fit parameters

c linear fit of the 2 components of G as a function of x
      call pompar(alpsf,betsf,0) !soft
      call pompar(alpsh,betsh,1) !sh
c Gaussian fit of the 2 components of G as a function of x and b
      call variance(delsf,gamsf,0)
      call variance(delsh,gamsh,1)
      gamsf=max(0.,gamsf)
      gamsh=max(0.,gamsh)

c Fit GFF

c      fit parameters
      numminDf=3                !minimum indice
      numparDf=4                !maximum indice
      betac=100.                 !temperature for chi2
      betae=1.                 !temperature for error
      fparDf=0.8                 !variation amplitude for range

      nmcxDf=20000              !number of try for x fit

c starting values

      parDf(1,3)=alpsf
      parDf(2,3)=betsf
      parDf(3,3)=alpsh
      parDf(4,3)=betsh

c      if(smaxDf.ge.3.*sfshlim)then
c        call paramx             !alpD and betD
c
c        call pompar(alpsf,betsf,-1) !soft (taking into account hard)
c        parDf(1,3)=alpsf
c        parDf(2,3)=max(-0.95+alppar,betsf)
c
c      endif

      alpsf=parDf(1,3)
      betsf=parDf(2,3)
      alpsh=parDf(3,3)
      betsh=parDf(4,3)

      if(ish.ge.4)then
        write(ifch,*)"param: fit parameters :",iscreen,iclpro,icltar
     *                                        ,iclegy,engy
        write(ifch,*)"alp,bet,gam,del sf:",alpsf,betsf,gamsf,delsf
        write(ifch,*)"alp,bet,gam,del sh:",alpsh,betsh,gamsh,delsh
      endif

      alpDs(idxD0,iclegy,iclpro,icltar)=alpsf
      alpDps(idxD0,iclegy,iclpro,icltar)=betsf
      alpDpps(idxD0,iclegy,iclpro,icltar)=0.
      betDs(idxD0,iclegy,iclpro,icltar)=betsf
      betDps(idxD0,iclegy,iclpro,icltar)=betsf
      betDpps(idxD0,iclegy,iclpro,icltar)=betsf
      gamDs(idxD0,iclegy,iclpro,icltar)=gamsf
      delDs(idxD0,iclegy,iclpro,icltar)=delsf

      alpDs(1,iclegy,iclpro,icltar)=alpsh
      alpDps(1,iclegy,iclpro,icltar)=betsh
      alpDpps(1,iclegy,iclpro,icltar)=0.
      betDs(1,iclegy,iclpro,icltar)=betsh
      betDps(1,iclegy,iclpro,icltar)=betsh
      betDpps(1,iclegy,iclpro,icltar)=betsh
      gamDs(1,iclegy,iclpro,icltar)=gamsh
      delDs(1,iclegy,iclpro,icltar)=delsh

      engy=engytmp

      return
      end


c----------------------------------------------------------------------

        subroutine pompar(alpha,beta,iqq)

c----------------------------------------------------------------------
c  Return the power beta and the factor alpha of the fit of the eikonal
c of a pomeron of type iqq : D(X)=alpha*(X)**beta
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision X,D1,D0,X0,D,droot
      double precision Dsoftshval,xmax
      dimension xlnXs(maxdataDf),xlnD(maxdataDf),sigma(maxdataDf)

      do i=1,nptf
        sigma(i)=1.e-2
      enddo

      if(iqq.le.0) then

        iscr=iqq
        xmax=min(0.1d0,10.d0*xminDf)
        X0=xminDf
        if(ish.ge.4)write (ifch,*) 'pompar (0) x0,xmax=',X0,xmax

        do i=0,nptf-1
          X=X0
          if (i.ne.0) X=X*(xmax/X0)**(dble(i)/dble(nptf-1))
          D=max(1.d-10,Dsoftshval(sngl(X)*smaxDf,X,0.d0,0.,iscr))
          if(D.eq.1.d-10)then
            write(ifch,*)
     &    "Warning in pompar ! Dsoftshval(0) could be negative"
            sigma(i+1)=1.e5
          endif
          xlnXs(i+1)=sngl(dlog(X*dble(smaxDf)))
          xlnD(i+1)=sngl(dlog(D))
        enddo


c Fit of D(X) between X0 and xmax

        call fit(xlnXs,xlnD,nptf,sigma,0,a,beta)
        if(beta.gt.10.)beta=10.

        alpha=exp(a)
c        alpha=sngl(Dsoftshval(sngl(X0)*smaxDf,X0,0.d0,0.,iscr))
c     &       *(sngl(X0)*smaxDf)**(-beta)


      elseif(iqq.eq.1.and.xfitmin.ne.1.d0) then

        iscr=2
c        xmax=max(0.01d0,min(1d0,dble(sfshlim*100./smaxDf)))
        xmax=1d0!min(1d0,dble(sfshlim*1000./smaxDf))

c Definition of D0=D(X0)

        D1=Dsoftshval(sngl(xmax)*smaxDf,xmax,0.d0,0.,iscr)

        D0=xfitmin*D1

c Calculation of X0 and D(X)

        X0=droot(D0,D1,xmax,iscr)
        if(ish.ge.4)write (ifch,*) 'pompar (1) x0,xmax=',X0,xmax

        do i=0,nptf-1
          X=X0
          if (i.ne.0) X=X*(xmax/X0)**(dble(i)/dble(nptf-1))
          D=max(1.d-10,Dsoftshval(sngl(X)*smaxDf,X,0.d0,0.,iscr))
          if(D.eq.1.d-10)then
            write(ifch,*)
     &    "Warning in pompar ! Dsoftshval(1) could be negative"
            sigma(i+1)=1.e5
          endif
          xlnXs(i+1)=sngl(dlog(X*dble(smaxDf)))
          xlnD(i+1)=sngl(dlog(D))
        enddo


c Fit of D(X) between X0 and xmax

        call fit(xlnXs,xlnD,nptf,sigma,0,a,beta)
        if(beta.gt.10.)beta=10.

        alpha=exp(a)
c        alpha=sngl(Dsoftshval(sngl(xmax)*smaxDf,xmax,0.d0,0.,iscr))
c     &       *(sngl(xmax)*smaxDf)**(-beta)


      elseif(iqq.eq.10.and.xfitmin.ne.1.d0) then    ! iqq=10

        iscr=0
        xmax=1.d0 !2.d0/max(2.d0,dlog(dble(smaxDf)/1.d3))

c Definition of D0=D(X0)

        D1=Dsoftshval(sngl(xmax)*smaxDf,xmax,0.d0,0.,iscr)

        D0=xfitmin*D1

c Calculation of X0 and D(X)

        X0=droot(D0,D1,xmax,iscr)
        if(ish.ge.4)write (ifch,*) 'pompar (1) x0,xmax=',X0,xmax

        do i=0,nptf-1
          X=X0
          if (i.ne.0) X=X*(xmax/X0)**(dble(i)/dble(nptf-1))
          D=max(1.d-10,Dsoftshval(sngl(X)*smaxDf,X,0.d0,0.,iscr))
          if(D.eq.1.d-10)then
            write(ifch,*)
     &    "Warning in pompar ! Dsoftshval(10) could be negative"
            sigma(i+1)=1.e5
          endif
          xlnXs(i+1)=sngl(dlog(X*dble(smaxDf)))
          xlnD(i+1)=sngl(dlog(D))
        enddo


c Fit of D(X) between X0 and xmax

        call fit(xlnXs,xlnD,nptf,sigma,0,a,beta)
        if(beta.gt.10.)beta=10.

        alpha=sngl(Dsoftshval(sngl(xmax)*smaxDf,xmax,0.d0,0.,iscr))
     &       *(sngl(xmax)*smaxDf)**(-beta)



      else                      !iqq=-1 or iqq=1 and xfitmin=1

c Calculation of X0 and D(X)
        iscr=0

        X0=1.d0/dble(smaxDf)
        xmax=max(2.d0/dble(smaxDf),
     &       min(max(0.03d0,dble(smaxDf)/2.d5),0.1d0))

        if(ish.ge.4)write (ifch,*) 'pompar (-1) x0,xmax=',X0,xmax

        do i=0,nptf-1
          X=X0
          if (i.ne.0) X=X*(xmax/X0)**(dble(i)/dble(nptf-1))
          D=max(1.d-10,Dsoftshval(sngl(X)*smaxDf,X,0.d0,0.,iscr))
          if(D.eq.1.d-10)then
            write(ifch,*)
     &    "Warning in pompar ! Dsoftshval(-1) could be negative"
            sigma(i+1)=1.e5
          endif
          xlnXs(i+1)=sngl(dlog(X*dble(smaxDf)))
          xlnD(i+1)=sngl(dlog(D))
        enddo

c Fit of D(X) between X0 and xmax

        call fit(xlnXs,xlnD,nptf,sigma,0,a,beta)
        if(beta.gt.10.)beta=10.


        alpha=exp(a)
c        alpha=sngl(Dsoftshval(sngl(xmax)*smaxDf,xmax,0.d0,0.,iscr))
c     &           *(sngl(xmax)*smaxDf)**(-beta)
      endif

        if(ish.ge.4)write(ifch,*) '%%%%%%%%%%%%% pompar %%%%%%%%%%%%%'
        if(ish.ge.4)write(ifch,*) 'alpD ini =',alpha,' betD ini=',beta

      return
      end

c----------------------------------------------------------------------

      double precision function droot(d0,d1,xmax,iscr)

c----------------------------------------------------------------------
c Find x0 which gives f(x0)=D(x0*S)-d0=0
c iqq=0 soft pomeron
c iqq=1 semi-hard pomeron
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision Dsoftshval,d0,d1,d2,x0,x1,x2,f0,f1,f2,xmax
      parameter (kmax=1000)


      k=0
      x0=max(1.d0/dble(sfshlim),1d-5)
      x1=xmax
 5    d2=dabs(Dsoftshval(sngl(x0)*smaxDf,x0,0.d0,0.,iscr))
      if(d2.lt.1.d-10.and.x0.lt.x1)then
        x0=dsqrt(x0*x1)
c        write(ifch,*)"droot",x0,x1,d0,d1,d2
        goto 5
        elseif(d2.gt.d0)then
        droot=x0
c        write(ifch,*)"droot",x0,x1,d0,d1,d2
        return
      endif
      f0=d2-d0
      f1=d1-d0
      if(f0*f1.lt.0.d0)then


 10   x2=dsqrt(x0*x1)
      d2=dabs(Dsoftshval(sngl(x2)*smaxDf,x2,0.d0,0.,iscr))
      f2=d2-d0
      k=k+1
c        write (ifch,*) '******************* droot **************'
c        write (ifch,*) x0,x1,x2,f0,f1,f2

      if (f0*f2.lt.0.D0) then
        x1=x2
        f1=f2
      else
        x0=x2
        f0=f2
      endif

      if (dabs((x1-x0)/x1).gt.(1.D-5).and.k.le.kmax.and.x1.ne.x0) then
        goto 10
      else
        if (k.gt.kmax) then
          write(ifch,*)'??? Warning in Droot: Delta=',dabs((x1-x0)/x1)
c.........stop 'Error in Droot, too many steps'
        endif
        droot=dsqrt(x1*x0)
      endif

      else
        droot=dsqrt(x1*x0)
      endif

      return
      end

c----------------------------------------------------------------------

      double precision function drootom(d0,dmax,bmax,eps)

c----------------------------------------------------------------------
c Find b0 which gives f(b0)=(1-exp(-om(b0,iqq)))/dmax-d0=0
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision om1intbc,d0,d1,d2,f0,f1,f2,dmax
      parameter (kmax=1000)


      k=0
      b0=0.
      b1=bmax
      d2=(1.d0-exp(-om1intbc(b1)))/dmax
      if(d2.gt.d0)then
        drootom=b1
c        write(*,*)"drootom exit (1)",b0,b1,d0,d1,d2
        return
      endif
      d1=(1.d0-exp(-om1intbc(b0)))/dmax
      f0=d1-d0
      f1=d2-d0
      if(f0*f1.lt.0.d0)then


 10   b2=0.5*(b0+b1)
      d2=(1.d0-dexp(-om1intbc(b2)))/dmax
      f2=d2-d0
      k=k+1
c      write (*,*) '******************* drootom **************'
c      write (*,*) b0,b1,b2,f0,f1,f2

      if (f1*f2.lt.0.D0) then
        b0=b2
        f0=f2
      else
        b1=b2
        f1=f2
      endif

      if (abs(f2).gt.eps.and.k.le.kmax.and.b1.ne.b0) then
        goto 10
      else
        if (k.gt.kmax) then
          write(ifch,*)'??? Warning in Drootom: Delta=',abs((b1-b0)/b1)
c.........stop 'Error in Droot, too many steps'
        endif
        drootom=0.5*(b1+b0)
      endif

      else
c        write(*,*)"drootom exit (2)",b0,b1,d0,d1,d2
        drootom=0.5*(b1+b0)
      endif

      return
      end

c----------------------------------------------------------------------
      subroutine variance(r2,alp,iqq)
c----------------------------------------------------------------------
c fit sigma2 into : 1/sigma2(x)=1/r2-alp*log(x*s)
c  iqq=0 -> soft pomeron
c  iqq=1 -> semi-hard pomeron
c  iqq=2 -> sum
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      dimension Xs(maxdataDf),vari(maxdataDf),sigma(maxdataDf)
      double precision X,X0,xmax

      do i=1,nptf
        sigma(i)=1.e-2
      enddo

      if(iqq.eq.0.or.xshmin.gt.0.95d0)then
        X0=xminDf
        xmax=xmaxDf
      elseif(iqq.eq.2)then
        X0=xshmin
        xmax=xmaxDf
      else
        X0=1d0/dlog(max(exp(2.d0),dble(smaxDf)/1.d3))
c        if(smaxDf.lt.100.*q2min)X0=.95d0
        xmax=xmaxDf
      endif
      if(iqq.ne.3.and.iqq.ne.4)then

        do i=0,nptf-1
          X=X0
          if (i.ne.0) X=X*(xmax/X0)**(dble(i)/dble(nptf-1))
          Xs(i+1)=log(sngl(X)*smaxDf)
          sig2=sigma2(X,iqq)
          if(sig2.le.0.) call utstop
     &   ('In variance, initial(1) sigma2 not def!&')
          vari(i+1)=1./sig2
        enddo

c Fit of the variance of D(X,b) between X0 and xmaxDf

        call fit(Xs,vari,nptf,sigma,0,tr2,talp)
        r2=1./tr2
        alp=-talp
c in principle, the formula to convert 1/(del+eps*log(sy)) into
c  1/del*(1-eps/del*log(sy)) is valid only if eps/del*log(sy)=alp*r2*log(sy)
c is small. In practice, since the fit of G(x) being an approximation, each
c component of the fit should not be taken separatly but we should consider
c G as one function. Then it works even with large alp (gamD).
c        ttt=alp*r2*log(smaxDf)
c        if(ttt.gt.0.5)
c     &    write(ifmt,*)'Warning, G(b) parametrization not optimal : ',
c     &          'gamD too large compared to delD !',ttt
      else
        if(iqq.eq.3)r2=sigma2(xmaxDf,3)
        if(iqq.eq.4)r2=sigma2(xshmin,3)
        if(r2.le.0.) call utstop
     &('In variance, initial(2) sigma2 not def!&')
        alp=0.
      endif

      if(ish.ge.4)then
        write(ifch,*) '%%%%%%%%%% variance ini %%%%%%%%%%%%'
        write(ifch,*) 'X0=',X0
        write(ifch,*) 'delD ini=',r2
        write(ifch,*) 'gamD ini=',alp
      endif

      return
      end



c----------------------------------------------------------------------

        function sigma2(x,iqq)

c----------------------------------------------------------------------
c Return the variance for a given x of :
c For G :
c iqq=0 the soft pomeron
c iqq=1 the semi-hard and valence quark pomeron
c iqq=2 the sum
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      double precision x,Dsoftshval,sfsh,om51p,eps,range,sig2!,omNpuncut
      external varifit
      double precision varifit,Db(maxdataDf),bf(maxdataDf)

      bmax=bmaxDf
      sig2=bmax*0.5
      bmin=-bmax
      eps=1.d-10
      ierro=0

      if(iqq.eq.0)then
        range=sig2
        sfsh=om51p(sngl(x)*smaxDf,x,0.d0,0.,0)
        if (dabs(sfsh).gt.eps) then
        do i=0,nptf-1
          bf(i+1)=dble(bmin+float(i)*(bmax-bmin)/float(nptf-1))
          Db(i+1)=om51p(sngl(x)*smaxDf,x,0.d0,sngl(bf(i+1)),0)/sfsh
        enddo
        else
          ierro=1
        endif
      elseif(iqq.eq.1.and.xshmin.lt..95d0)then
        range=sig2
        sfsh=0.d0
        do j=1,4
          sfsh=sfsh+om51p(sngl(x)*smaxDf,x,0.d0,0.,j)
        enddo
        if (dabs(sfsh).gt.eps) then
        do i=0,nptf-1
          bf(i+1)=dble(bmin+float(i)*(bmax-bmin)/float(nptf-1))
          Db(i+1)=0.d0
          do j=1,4
           Db(i+1)=Db(i+1)+om51p(sngl(x)*smaxDf,x,0.d0,sngl(bf(i+1)),j)
          enddo
          Db(i+1)=Db(i+1)/sfsh
        enddo
        else
          ierro=1
        endif
      else
        sig2=2.d0*sig2
        range=sig2
        iscr=0
        sfsh=Dsoftshval(sngl(x)*smaxDf,x,0.d0,0.,iscr)
        if (dabs(sfsh).gt.eps) then
        do i=0,nptf-1
          bf(i+1)=dble(bmin+float(i)*(bmax-bmin)/float(nptf-1))
          Db(i+1)=Dsoftshval(sngl(x)*smaxDf,x,0.d0,sngl(bf(i+1)),iscr)
     &              /sfsh
        enddo
        else
          ierro=1
        endif
      endif

c Fit of D(X,b) between -bmaxDf and bmaxDf

      if(ierro.ne.1)then
        nptft=nptf
        call minfit(varifit,bf,Db,nptft,sig2,range)
        sigma2=sngl(sig2)
      else
        sigma2=0.
      endif

      return
      end

c----------------------------------------------------------------------

        subroutine paramx

c----------------------------------------------------------------------
c updates the 4 parameters alpsf,betsf,alpsh,betsh by fitting GFF
c  parDf(1,3) parDf(2,3) ... alp, bet soft
c  parDf(3,3) parDf(4,3) ... alp, bet semihard
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision Dsoftshpar

      external Dsoftshpar

      dimension range(nbpf)

      call givedatax

      !determine parameter range
      do i=numminDf,numparDf
        range(i)=fparDf*parDf(i,3)
        parDf(i,1)=parDf(i,3)-range(i)
        parDf(i,2)=parDf(i,3)+range(i)
      enddo


   !   write(ifch,*) '%%%%%%%%%%%%%%%%%%% fitx %%%%%%%%%%%%%%%%%%%%%%%'

      call fitx(Dsoftshpar,nmcxDf,chi2,err)

   !   write(ifch,*) 'chi2=',chi2
   !   write(ifch,*) 'err=',err
   !   write(ifch,*) 'alpD(1)=',parDf(1,3),' betD(1)=',parDf(2,3)
   !   write(ifch,*) 'alpD(2)=',parDf(3,3),' betD(2)=',parDf(4,3)

      return
      end


c----------------------------------------------------------------------
      subroutine givedatax
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision X,X0,X1,Dsoftshval,Xseuil

      numdataDf=nptf

      X0=xminDf
      X1=xmaxDf
      Xseuil=1d0 !min(1.d0,dble(sfshlim*1e4)*XminDf)
c      print *,'--------------->',Xseuil

c Fit of G(X) between X0 and X1
      do i=0,nptf-1
        X=X0
        if (i.ne.0) X=X*(X1/X0)**(dble(i)/dble(nptf-1))
        datafitD(i+1,2)=max(1.e-10,
     &                  sngl(Dsoftshval(sngl(X)*smaxDf,X,0.d0,0.,1)))
        datafitD(i+1,1)=sngl(X)
        datafitD(i+1,3)=1.
        if (X.gt.Xseuil)
     &  datafitD(i+1,3)=exp(-min((sngl(Xseuil/X)-1.),50.))
      enddo

      return
      end





c----------------------------------------------------------------------

      function sigma1i(x)

c----------------------------------------------------------------------
c Return the variance of the sum of the soft pomeron and the semi-hard
c pomeron for a given x.
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      double precision x,Dsoftshval,Dint


      iscr=0
      Dint=Dsoftshval(sngl(x)*smaxDf,x,0.d0,0.,iscr)

      sigma1i=0.
      if(Dint.ne.0.)
     &sigma1i=sngl(-1.d0/dlog(Dsoftshval(sngl(x)*smaxDf,x,0.d0,1.,iscr)
     &   /Dint))


      return
      end


c----------------------------------------------------------------------

      SUBROUTINE minfit(func,x,y,ndata,a,range)

c----------------------------------------------------------------------
c Given a set of data points x(1:ndata),y(1:ndata), and the range of
c the parameter a, fit it on function func by minimizing chi2.
c In input a define the expected value of a, and on output they
c correspond to  the fited value.
c ---------------------------------------------------------------------
      include 'epos.inc'
      double precision x(ndata),y(ndata),func,a,range,Smin,Som,a1,a2,eps
     *,amin,rr,yp
      parameter (eps=1.d-5)
      external func


      Smin=1.d20
      amin=a



      a1=a-range
      a2=a+range

      do j=1,2000
        rr=dble(rangen())
        a=a1+(a2-a1)*rr
        k=0

 10     if(a.lt.0.d0.and.k.lt.100) then
          rr=dble(rangen())
          a=a1+(a2-a1)*rr
          k=k+1
          goto 10
        endif
        if(k.ge.100) call utstop
     &('Always negative variance in minfit ...&')

        Som=0.d0
        do k=1,ndata
             yp=min(1.d10,func(x(k),a))  !proposal function
              Som=Som+(yp-y(k))**2.d0
        enddo
        if(Som.lt.Smin)then

          if(Smin.lt.1.)then
            if(a.gt.amin)then
              a1=amin
            else
              a2=amin
            endif
          endif
          amin=a
          Smin=Som
        endif
        if(Smin.lt.eps)goto 20
      enddo

 20   continue
      a=amin

      return
      end



c----------------------------------------------------------------------
      subroutine fitx(func,nmc,chi2,err)
c----------------------------------------------------------------------
c  Determines parameters of the funcion func
c  representing the best fit of the data.
c  At the end of the run, the "best" parameters are stored in parDf(n,3).
c  The function func has to be defined via "function" using the parameters
c  parDf(n,3), n=1,numparDf .
c  Parameters as well as data are stored on /fitpar/:
c    numparDf: number of parameters  (input)
c    parDf: array containing parameters:
c         parDf(n,1): lower limit    (input)
c         parDf(n,2): upper limit    (input)
c         parDf(n,3): current parameter (internal and output = final result)
c         parDf(n,4): previous parameter (internal)
c    numdataDf: number of data points  (input)
c    datafitD: array containing data:
c         datafitD(i,1): x value       (input)
c         datafitD(i,2): y value       (input)
c         datafitD(i,3): error         (input)
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      double precision func,x
      external func

 !     write (ifch,*) 'numparDf,numminDf',numparDf,numminDf


c initial configuration (better if one start directly with initial one)

c      do n=numminDf,numparDf
c              parDf(n,3)=parDf(n,1)+rangen()*(parDf(n,2)-parDf(n,1))
c      enddo

      chi2=0.
      err=0.
      do i=1,numdataDf
        x=dble(datafitD(i,1))
        fx=sngl(func(x))
        chi2=chi2+(log(fx)-log(datafitD(i,2)))**2/datafitD(i,3)**2
        err=err+(log(fx)-log(datafitD(i,2)))/datafitD(i,3)**2
      enddo
      err=abs(err)/float(numdataDf)

c metropolis iteration

      do i=1,nmc
c        if(mod(i,int(float(nmc)/1000.)).eq.0)then
          betac=betac*(1.+1./float(nmc))!1.05
          betae=betae*(1.+1./float(nmc))!1.05
c        endif
c        if(mod(i,int(float(nmc)/20.)).eq.0)write(ifch,*)i,chi2,err

        do n=numminDf,numparDf
          parDf(n,4)=parDf(n,3)
        enddo
        chi2x=chi2
        errx=err

        n=numminDf+int(rangen()*(numparDf-numminDf+1))
        n=max(n,numminDf)
        n=min(n,numparDf)
c              if(mod(i,int(float(nmc)/20.)).eq.0)write(ifch,*)n

        parDf(n,3)=parDf(n,1)+rangen()*(parDf(n,2)-parDf(n,1))
        chi2=0
        err=0
        do j=1,numdataDf
          x=dble(datafitD(j,1))
          fx=sngl(func(x))
          chi2=chi2+(log(fx)-log(datafitD(j,2)))**2/datafitD(j,3)**2
          err=err+(log(fx)-log(datafitD(j,2)))/datafitD(j,3)**2
        enddo
        err=abs(err)/float(numdataDf)

        if(chi2.gt.chi2x.and.rangen()
     $             .gt.exp(-min(50.,max(-50.,betac*(chi2-chi2x))))
     &             .or.err.gt.errx.and.rangen()
     $             .gt.exp(-min(50.,max(-50.,betae*(err-errx))))
     &                                                     ) then
          do n=numminDf,numparDf
            parDf(n,3)=parDf(n,4)
          enddo
          chi2=chi2x
          err=errx
        endif

      enddo

      return
      end


c----------------------------------------------------------------------

        SUBROUTINE fit(x,y,ndata,sig,mwt,a,b)

c----------------------------------------------------------------------
c Given a set of data points x(1:ndata),y(1:ndata) with individual standard
c deviations sig(1:ndata), fit them to a straight line y = a + bx by
c minimizing chi2 .
c Returned are a,b and their respective probable uncertainties siga and sigb,
c the chi­square chi2, and the goodness-of-fit probability q (that the fit
c would have chi2 this large or larger). If mwt=0 on input, then the standard
c deviations are assumed to be unavailable: q is returned as 1.0 and the
c normalization of chi2 is to unit standard deviation on all points.
c ---------------------------------------------------------------------

        implicit none
        INTEGER mwt,ndata
        REAL sig(ndata),x(ndata),y(ndata)
        REAL a,b,siga,sigb,chi2 !,q
        INTEGER i
        REAL sigdat,ss,st2,sx,sxoss,sy,t,wt


        sx=0.                 !Initialize sums to zero.
        sy=0.
        st2=0.
        b=0.
        if(mwt.ne.0) then ! Accumulate sums ...
          ss=0.
          do i=1,ndata          !...with weights
            wt=1./(sig(i)**2)
            ss=ss+wt
            sx=sx+x(i)*wt
            sy=sy+y(i)*wt
          enddo
        else
          do i=1,ndata          !...or without weights.
            sx=sx+x(i)
            sy=sy+y(i)
          enddo
          ss=float(ndata)
        endif
        sxoss=sx/ss
        if(mwt.ne.0) then
          do i=1,ndata
            t=(x(i)-sxoss)/sig(i)
            st2=st2+t*t
            b=b+t*y(i)/sig(i)
          enddo
        else
          do i=1,ndata
            t=x(i)-sxoss
            st2=st2+t*t
            b=b+t*y(i)
          enddo
        endif
        b=b/st2                 !Solve for a, b, oe a , and oe b .
        a=(sy-sx*b)/ss
        siga=sqrt((1.+sx*sx/(ss*st2))/ss)
        sigb=sqrt(1./st2)
        chi2=0.                 !Calculate chi2 .
c        q=1.
        if(mwt.eq.0) then
          do i=1,ndata
            chi2=chi2+(y(i)-a-b*x(i))**2
          enddo

c For unweighted data evaluate typical sig using chi2, and adjust
c the standard deviations.

          sigdat=sqrt(chi2/(ndata-2))
          siga=siga*sigdat
          sigb=sigb*sigdat
        else
          do i=1,ndata
            chi2=chi2+((y(i)-a-b*x(i))/sig(i))**2
          enddo
        endif

        if(chi2.ge.0.2)then
          b=(y(ndata)-y(1))/(x(ndata)-x(1))
          a=y(ndata)-b*x(ndata)
        endif


c        write(*,*) '$$$$$$$$ fit : a,b,siga,sigb,chi2,q $$$$$$$$$'
c        write(*,*) a,b,siga,sigb,chi2!???????????????


        return
        END



c----------------------------------------------------------------------

      double precision function varifit(x,var)

c----------------------------------------------------------------------
      double precision x,var

      varifit=dexp(-min(50.d0,x**2.d0/var))

      return
      end



c----------------------------------------------------------------------

      double precision function Dsoftshval(sy,x,y,b,iscr)

c----------------------------------------------------------------------
c iscr=-1 sum of om5p (i), i from 0 to 4 - fit of hard
c iscr=0 sum of om5p (i), i from 0 to 4
c iscr=1 sum of om5p (i), i from 0 to 4 * F * F
c iscr=2 sum of om5p (i), i from 1 to 4 (semihard + valence quark)
c----------------------------------------------------------------------
      double precision x,om51p,y,xp,xm
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      Dsoftshval=0.d0

      if(iscr.le.0)then
        do i=0,4
          Dsoftshval=Dsoftshval+om51p(sy,x,y,b,i)
        enddo


      elseif(iscr.eq.1)then
        xp=dsqrt(x)*dexp(y)
        if(dabs(xp).ge.1.d-15)then
          xm=x/xp
        else
          xm=1.d0
          write(ifch,*)'Warning in Dsoftshval in epos-par'
        endif
        do i=0,4
          Dsoftshval=Dsoftshval+om51p(sy,x,y,b,i)
        enddo
        Dsoftshval=Dsoftshval*(1.d0-xm)**dble(alplea(icltar))
     &                       *(1.d0-xp)**dble(alplea(iclpro))
      elseif(iscr.eq.2)then
        do i=1,4
          Dsoftshval=Dsoftshval+om51p(sy,x,y,b,i)
        enddo
      endif


      Dsoftshval=2.d0*Dsoftshval
     &     /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))

      if(iscr.eq.-1.and.parDf(3,3).lt.parDf(1,3))Dsoftshval=Dsoftshval
     &     -dble(parDf(3,3)*sy**parDf(4,3))

      return
      end

c----------------------------------------------------------------------

      double precision function Dsoftshpar(x)

c----------------------------------------------------------------------
      double precision x,xp,xm
      include 'epos.inc'
      include 'epos.incpar'

      Dsoftshpar=dble(
     &        parDf(1,3)*(sngl(x)*smaxDf)**parDf(2,3)
     &       +parDf(3,3)*(sngl(x)*smaxDf)**parDf(4,3) )
      xp=dsqrt(x)
      xm=xp
      Dsoftshpar=Dsoftshpar*(1.d0-xm)**dble(alplea(icltar))
     &                       *(1.d0-xp)**dble(alplea(iclpro))
      Dsoftshpar=min(max(1.d-15,Dsoftshpar),1.d15)

      return
      end
