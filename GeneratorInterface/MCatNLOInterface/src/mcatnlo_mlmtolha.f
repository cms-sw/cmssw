c Interface to LHAPDF, v4.1 onwards. The inclusion of LHAGLUE allows the 
c use of PDFLIB format. This file has been derived from mcatnlo_mlmtopdf.f.
c Calls to newmode and pdftomlm are trivial here, but are kept for easier
c future changes
c
      subroutine mlmpdf(mode,ih,q2,x,fx,nf)
      implicit real * 8 (a-h,o-z)
      real * 4 q2,x,fx(-nf:nf)
      common/trans/nptype,ngroup,nset
c x*pdf(x) in PDG format
      dimension fxp(-6:6)
c used by pdfset
      dimension val(20)
c used by pdfset
      character * 20 parm(20)
c used by structp
      parameter (zero=0.d0)
      parameter (izero=0)
      logical ini
      data ini/.true./,imode/0/
c

      if(ih.ne.5)then
c incoming particle is not an electron: use PDFLIB
        if(ini.or.imode.ne.mode) then
          ini = .false.
          imode = mode
c pass from our conventions to PDFLIB conventions for parameter settings. 
c See subroutine newmode for comments on the conventions adopted. Proper
c settings of /LHAPDFC/ and /LHACONTROL/ must have been adopted before
c runtime (and before the call to PDFSET)

          call newmode(ih,mode)
          parm(1) = 'DEFAULT'
          val (1) =  nset
          call pdfset(parm,val)
        endif
        xd = dble(x)
        qd = dble(sqrt(q2))
c first convert structm format into pftopdg format (fxp), in order to use
c what already done in mcatnlo_mlmtopdf.f where pftopdg is available. For
c photons, use structp and consider only on-shell photons
        if(abs(ih).le.4)then
          if(abs(ih).le.3)then
            call structm(xd,qd,upv,dnv,usea,dsea,str,chm,bot,top,glu)
          elseif(ih.eq.4)then
            q2d = dble(q2)
            call structp(xd,q2d,zero,izero,
     #                   upv,dnv,usea,dsea,str,chm,bot,top,glu)
          else
            write(*,*)'Unknow particles in LHAPDF'
            stop
          endif
c If a nucleon, define fxp assuming LHAPDF returns proton densities, i.e.
c up=upv+usea, upbar=usea, dn=dnv+dsea, dnbar=dsea. If a pion or photon,
c use the SAME conventions (which are wrong) according to PDFLIB manual.
c Correct below in order to obtain the physical densities from which
c to calculate the luminosities
          fxp( 0)=glu
          fxp( 1)=dnv+dsea
          fxp( 2)=upv+usea
          fxp( 3)=str
          fxp( 4)=chm
          fxp( 5)=bot
          fxp( 6)=top
          fxp(-1)=dsea
          fxp(-2)=usea
          fxp(-3)=str
          fxp(-4)=chm
          fxp(-5)=bot
          fxp(-6)=top
        else
          write(*,*)'Unknow particles in LHAPDF'
          stop
        endif
c in our conventions 1=up, 2=down; PDFLIB 1=down, 2=up. With
c f(1)<-->f(2) we mean also f(-1)<-->f(-2)
c in the following lines, deals with particles only (no antiparticles)
c proton(ih=1) ==> f(1)<-->f(2)
c neutron(ih=2) ==> no action (f(1)<-->f(2) for PDFLIB convention and
c    f(1)<-->f(2) for isospin symmetry (u_proton=d_neutron....)
c pion+(ih=3) ==> f(2)<-->f(-2), since PDFLIB has d=u=q_v+q_sea, 
c    ubar=dbar=q_sea
c photon(ih=4) ==> f(-1)<-->f(-2) and f(i)=f(-i)/2, i=1,2 since PDFLIB 
c    has f(i)=2*f(-i), and f(1)<-->f(2)
c Notice that in the jet package pions and neutrons are not used. If
c selected, they are rejected by the routine pdfpar. This routine
c is therefore a completely general interface with PDFLIB
        if(abs(ih).eq.1) then
          tmp     = fxp(1)
          fxp(1)  = fxp(2)
          fxp(2)  = tmp
          tmp     = fxp(-1)
          fxp(-1) = fxp(-2)
          fxp(-2) = tmp
        elseif(abs(ih).eq.2) then
          continue
        elseif(abs(ih).eq.3) then
          tmp = fxp(-2)
          fxp(-2) = fxp(2)
          fxp(2)  = tmp
        elseif(abs(ih).eq.4) then
          tmp     = fxp(-1)
          fxp(-1) = fxp(-2)
          fxp(-2) = tmp
          fxp(1)=fxp(-1)
          fxp(2)=fxp(-2)
c this is (p+n)/2
        elseif(ih.eq.0) then
          va  = (fxp(1)+fxp(2))/2
          sea = (fxp(-1)+fxp(-2))/2
          fxp(1)  = va
          fxp(2)  = va
          fxp(-1) = sea
          fxp(-2) = sea
        else
          write(*,*) ' ih was', ih,' instead of 0, +-1, +-2, +-3, or 4'
          stop
        endif
c for particles, ich=1, for antiparticles, ich=-1
        if(ih.lt.0) then
          ich = -1
        else
          ich = 1
        endif
c divide by x and exchange q with qbar in the case of antiparticles
        do j=-nf,nf
          fx(j) = sngl(fxp(j*ich)/xd)
        enddo
      else
c incoming "hadron" is an electron
        if(ini.or.imode.ne.mode) then
          ini = .false.
          imode = mode
          iset=mode-50
        endif
        if(iset.eq.1) then
          call elpdf_lac1(q2,x,fx,nf)
        elseif(iset.eq.2) then
          call elpdf_grv(q2,x,fx,nf)
        elseif(iset.eq.3) then
          call elpdf_user(q2,x,fx,nf)
        else
          write(*,*)'Electron set non implemented'
          stop
        endif
      endif
      return
      end


      subroutine pdfpar(mode,ih,xlam,scheme,iret)
      implicit real * 8 (a-h,o-z)
      common/trans/nptype,ngroup,nset
      common/w50512/qcdl4,qcdl5
      dimension val(20)
      character * 20 parm(20)
      character * 2 scheme
      logical ini
      data ini/.true./
c iret#0 when problem occur
      iret = 0
      if(ini)then
        write(*,*)'This is an interface to LHAPDF'
        ini = .false.
      endif
c modify the following if non-nucleon PDFs will be included in LHAPDF
      if(abs(ih).gt.5)then
        write(*,*) ' Hadron tpye ',ih,' not implemented'
        iret=1
        return
      endif
c fake values. If kept, the main program crashes
      scheme='XX'
      xlam=0.0
c incoming particle is not an electron: use LHAPDF
      if(ih.ne.5)then
c the scheme of the LHAPDF set is not given in any common block.
c Set it by hand in the main program
        scheme = '**'
        call newmode(ih,mode)
        parm(1) = 'DEFAULT'
        val (1) =  nset
c set the parameters
        call pdfset(parm,val)
c Lambda_QCD_5, as given by LHAPDF; this may be wrong, so be careful
        xlam = qcdl5
      else
c incoming particle is an electron
        if(mode.eq.51)then
          scheme='MS'
          xlam=.130
        elseif(mode.eq.52)then
          scheme='DG'
          xlam=.130
        elseif(mode.eq.53)then
          scheme='**'
          xlam=.001
        else
          write(*,*)'Electron set not implemented'
          iret=1
        endif
      endif
      return
      end


      subroutine prntsf
c     prints details of the structure function sets
c
      write(*,100)                             
     #  '  Refer to LHAPDF on-line manual to see'
     # ,'  PDFLIB-compatible labeling conventions'
 100  format(1x,a,100(/,1x,a))
      return
      end


      subroutine newmode(ih,mode)
c Trivial except for ih in the case of LHAPDF, since PDFLIB-type set
c numbers are specified without using a group number; this format is
c therefore compatible with mlm's one. For particle types (LHAPDF v4.1
c doesn't seem to have a particle type classification -- retain here
c that of PDFLIB, to allow more flexibility for the future).
c
c                        MC@NLO               PDFLIB
c 
c  nucleons           -2,-1,0,1,2                1
c  pions                  -3,3                   2
c  photons                  4                    3
c 
      implicit real * 8 (a-h,o-z)
      common/trans/nptype,ngroup,nset
c
      if(abs(ih).le.2)then
        nptype=1
      elseif(abs(ih).eq.3)then
        nptype=2
      elseif(ih.eq.4)then
        nptype=3
      else
        write(6,*)'Hadron type not implemented in PDFLIB/LHAPDF'
        stop
      endif
      ngroup=-1
      nset=mode
      return
      end


      subroutine pdftomlm(ipdfih,ipdfgroup,ipdfndns,ihmlm,ndnsmlm)
c Performs the inverse operation of newmode. From LHAPDFv4.0, proton,
c pions, and photons are included; the common block w50511 is not 
c filled by LHAGLUE, and thus the information on particle type is not
c available in versions up to v4.1; keep the following for more flexibility
      implicit real * 8 (a-h,o-z)
c
      if(ipdfih.eq.1)then
        ihmlm=1
      elseif(ipdfih.eq.2)then
        ihmlm=3
      elseif(ipdfih.eq.3)then
        ihmlm=4
      else
        write(*,*)'Wrong hadron type in LHAPDF:',ipdfih
        stop
      endif
      ndnsmlm=ipdfndns
      return
      end


      subroutine setlhacblk(strin)
c Must be called BEFORE SETPAR. It sets the variables that control LHAPDF,
c stored in the common block /LHACONTROL/, taken from LHAGLUE. /LHAPDFC/ 
c can also be set here if needed (in this version, use the default and
c define logical link at runtime instead)
      implicit none
      character*70 strin,strout
      character*20 lhaparm(20)
      double precision lhavalue(20)
      common/lhacontrol/lhaparm,lhavalue
c
c change the following to collect statistics for PDFs outside range
      lhaparm(16)='NOSTAT'
c 'EXTRAPOLATE' will extrapolate PDFs outside their range
      call fk88low_to_upp(strin,strout)
      lhaparm(18)=strout
c set to 'SILENT' to suppress output completely
      lhaparm(19)='LOWKEY'
c
      return
      end


C------- ALPHA QCD -------------------------------------
c Program to calculate alfa strong with nf flavours,
c as a function of lambda with 5 flavors.
c The value of alfa is matched at the thresholds q = mq.
c When invoked with nf < 0 it chooses nf as the number of
c flavors with mass less then q.
c
      function alfas(q2,xlam,inf)
      implicit real * 8 (a-h,o-z)
      data olam/0.d0/,pi/3.14159d0/
      data xmb/5.d0/,xmc/1.5d0/

CCC --------------------------------------------------------------
CCC added by fabian stoeckli (fabian.stoeckli@cern.ch)
CCC 12.2.2007
CCC in order to keep in memory values of b5 and bp5
      common/fstpar/b5,bp5
CCC --------------------------------------------------------------

      if(xlam.ne.olam) then
        olam = xlam
        b5  = (33-2*5)/pi/12
        bp5 = (153 - 19*5) / pi / 2 / (33 - 2*5)
        b4  = (33-2*4)/pi/12
        bp4 = (153 - 19*4) / pi / 2 / (33 - 2*4)
        b3  = (33-2*3)/pi/12
        bp3 = (153 - 19*3) / pi / 2 / (33 - 2*3)
        xlc = 2 * log(xmc/xlam)
        xlb = 2 * log(xmb/xlam)
        xllc = log(xlc)
        xllb = log(xlb)
        c45  =  1/( 1/(b5 * xlb) - xllb*bp5/(b5 * xlb)**2 )
     #        - 1/( 1/(b4 * xlb) - xllb*bp4/(b4 * xlb)**2 )
        c35  =  1/( 1/(b4 * xlc) - xllc*bp4/(b4 * xlc)**2 )
     #        - 1/( 1/(b3 * xlc) - xllc*bp3/(b3 * xlc)**2 ) + c45
      endif
      q   = sqrt(q2)
      xlq = 2 * log( q/xlam )
      xllq = log( xlq )
      nf = inf
      if( nf .lt. 0) then
        if( q .gt. xmb ) then
          nf = 5
        elseif( q .gt. xmc ) then
          nf = 4
        else
          nf = 3
        endif
      endif
      if    ( nf .eq. 5 ) then
        alfas = 1/(b5 * xlq) -  bp5/(b5 * xlq)**2 * xllq
      elseif( nf .eq. 4 ) then
        alfas = 1/( 1/(1/(b4 * xlq) - bp4/(b4 * xlq)**2 * xllq) + c45 )
      elseif( nf .eq. 3 ) then
        alfas = 1/( 1/(1/(b3 * xlq) - bp3/(b3 * xlq)**2 * xllq) + c35 )
      else
        print *,'error in alfa: unimplemented # of light flavours',nf
        stop
      endif
      return
      end
