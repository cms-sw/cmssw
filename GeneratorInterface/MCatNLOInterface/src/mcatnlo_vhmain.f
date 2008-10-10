      SUBROUTINE VHMAIN
      implicit none
      include 'vhgcblks.h'
      real * 8 value(20),xmass(-5:21),xmomshifts(4)
      real * 8 pi,xicut,delta,xicutss,deltas,deltac,tvsamp,thsamp,
     #  etacut,wdtwon,wdtwmsb,wdtzon,wdtzmsb,tmas,xpdflam4,xpdflam5,
     #  tmp,xm0v,vgammax,xmvlow,xmvupp,xm0h,hgammax,xmhlow,xmhupp,
     #  ecm,xfh,xren,ac1,ac2,bwvmdpl,bwvmdmn,bwhmdpl,bwhmdmn,
     #  rohlim,xtotal,ytotal,dtot,av3a,d3a,av3nega,d3nega,ctime,
     #  avtot,xares,yares,xbres,ybres,al_gfun,be_gfun,ccc_gfun,
     #  wgtaev,wgtbev,xrenmc,xfhmc,av3b,d3b,av3negb,d3negb,
     #  evfrac,evprcfrac,dummy,xmone,bwvdelfmax,bwhdelfmax
      integer ih1,ih2,ndns1,ndns2,iinput,iprespl,ifxdaem,isubttype,
     #  nsamp,iprdct,idec,loproc,maproc,ifk88istrl,ifk88ih,
     #  ifk88ndns,ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,iverbose,
     #  ichkmom,ibswrite,itmpih,itmpndns,idpdfset,iprdct0hw,iwidth,
     #  iseed0,iproc,it1,it2,iseld,ncl3,ifuntype,ndim,nwild,itd1,itd2,
     #  ibscall,mx_of_evta,mx_of_evtb,iwgtnorm,inloscale,imcscale,
     #  iwrong,iwrong1,ionshell,ilepmass,ifk88seed,i,maxevt,iseed,
     #  maxtrials,ntotal,ndiff,nevts,ntrls,iunita,iunitb,ioutput,
     #  itot,ii,iunit,ia1ora2,iv0hw,il0hw,iprdct0hwtmp,izero,ione
      character * 2 scheme
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical evgen
      external sig5azw_und,sig5afw_und,sig5afw_dec
      external sig5bzw_und,sig5bfw_und,sig5bfw_dec
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
c
c common /strfvh0/ is only in strfvh:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfvh0/ih1,ih2,ndns1,ndns2
c quark and gluon masses, used by Herwig. PDF labeling convention
      common/parmass/xmass
c al_gfun, be_gfun, ccc_gfun are the parameters entering gfun
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c iwgtnorm=0 for weight=+1/-1, iwgtnorm=1 otherwise
      common/ciwgtnorm/iwgtnorm
c wgtaev and wgtbev are the norms of weights for H and S events respectively
      common/cwgtev/wgtaev,wgtbev
c iprespl=0 ==> preserves rapidity
c iprespl=1 ==> preserves longitudinal momentum
      common/ciprespl/iprespl
c ifxdaem=0 ==> uses running alpha_EM(M^2)
c ifxdaem=1 ==> uses alpha_EM=1/137.0359895
      common/cifxdaem/ifxdaem
c ichkmom=0 --> enables checks on kinematics
      common/cichkmom/ichkmom
c----------------------------------------------------------
c Variables that control the integrations
c
      common/cisubttype/isubttype
      common/parsub/xicut,delta
      common/xisave/xicutss
      common/pmerge/deltas,deltac
      common/ctsamp/tvsamp,thsamp
      common/samp/nsamp
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
c Partial W and Z decay widths in the on-shell and MSbar schemes
      common/partdec/wdtwon,wdtwmsb,wdtzon,wdtzmsb
c----------------------------------------------------------
c inloscale controls the reference scale in the NLO computation
      common/cinloscale/inloscale
c imcscale controls the reference scale in the MC subtraction terms
      common/cimcscale/imcscale
c----------------------------------------------------------
c Process: iprdct=240     W^+ H  with decay
c                 260     W^- H  with decay
c                 270     Z H    with decay
c                 248     W^+ H  without decay
c                 268     W^- H  without decay
c                 278     Z H    without decay
c These are NLO internal codes; for HERWIG codes, see getnloiproc
      common/ciprdct/iprdct
c Decay of the vector boson: idec=0    -->   V decays
c                            idec=1    -->   V doesn't decay
      common/cidec/idec
c----------------------------------------------------------
c The following refer to the computation of MC subtraction terms
c ia1ora2=1 -> full invariants, ia1ora2=2 -> simplified invariants
      common/cia1ora2/ia1ora2
c Subprocesses: 'qq', 'qg', corresponding to jproc=jproc0=2,3
c In the integration routines, loproc<=jproc<=maproc
      common/cwchproc/loproc,maproc
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c Average shifts in momenta, due to quark and lepton masses
      common/cshifts/xmomshifts
c----------------------------------------------------------
c ifuntype=1 for sig5a, ifuntype=2 for sig5b
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c Flag to put leptons on shell, according to PDF masses
      common/cilepmass/ilepmass
c----------------------------------------------------------
c Common blocks for general MC@NLO routines
c common block for internal rnd number generation, independent of bases
      common/cifk88seed/ifk88seed
c common block fk88ipdfs is filled by our interface to MLMPDF
      common/fk88ipdfs/ifk88ih,ifk88ndns
c common block w50511 and w50512 are filled by PDFLIB 
      common/w50511/ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,tmas
      common/w50512/xpdflam4,xpdflam5

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      character*100 stfilename
      common/fstbases/ basesoutput,stfilename
CCCC  ================================================================


CCCC ====================================================================
CCCC  added by fabian stoeckli (fabian.stoeckli@cern.ch)
CCCC  12.2.2007
CCCC  common block and variabes added for CMSSW interface
      character * 20 mmgname
      character * 2 mmscheme
      character * 4 mmpart1,mmpart2
      real * 8 mmecm,mmxren,mmxfh,mmxrenmc,mmxfhmc,mmxmh0,mmgah,mmxmt,
     % mmgammax,mmxmhl,mmxmhu,mmxmass1,mmxmass2,mmxmass3,mmxmass4,
     % mmxmass5,mmxmass21,mmxlam,mmxm0,mmgav,mmxm0v,mmtwidth
      real * 8 mmxwm,mmxzm,mmxww,mmxzw,mmv1gammax,mmv1massinf,
     % mmv1masssup
      real * 8 mmv2gammax,mmv2massinf,mmv2masssup,mmvud,mmvus,mmvub,
     % mmvcd,mmvcs,mmvcb
      real * 8 mmvtd,mmvts,mmvtb,mmxlamherw
      integer mmmaxevt,mmidpdfset,mmiwgtnorm,mmiseed,mmibornex,
     % mmit1,mmit2
      integer mmivcode,mmil1code,mmil2code,mmaemrun,mmiproc

      integer mmidecseed, mmifk88seed

cccc common block for the input parameters
      common/params/mmecm,mmxren,mmxfh,mmxrenmc,mmxfhmc,mmxmh0,mmgah,
     % mmxmt,mmgammax,mmxmhl,mmxmhu,mmxmass1,mmxmass2,mmxmass3,mmxmass4,
     % mmxmass5,mmxmass21,mmxlam,mmxm0,mmgav,mmxm0v,mmtwidth,
     % mmxwm,mmxzm,mmxww,mmxzw,
     % mmv1gammax,mmv1massinf,mmv1masssup,mmv2gammax,mmv2massinf,
     % mmv2masssup,mmvud,mmvus,mmvub,
     % mmvcd,mmvcs,mmvcb,mmvtd,mmvts,mmvtb,mmxlamherw,
     % mmmaxevt,mmidpdfset,mmiwgtnorm,
     % mmiseed,mmibornex,mmit1,mmit2,mmivcode,mmil1code,mmil2code,
     % mmaemrun,mmiproc,
     % mmidecseed, mmifk88seed,
     % mmgname,mmscheme,
     % mmpart1,mmpart2

      integer mmdecide
      common/decpar/mmdecide
CCC =================================================================

C
C------------------------------------------------------------------------
C                             START                                     -
C------------------------------------------------------------------------
c iinput=1 ==> all inputs are given by the user
      iinput=2
      mmdecide = 3
c iverbose=1 ==> writes more on standard output
      iverbose=0
c ichkmom=0 ==> enables checks on kinematics
      ichkmom=1
c initialization of internal randon number generation
C     changed by F. STOECKLI, 10.10.2008
C      ifk88seed=1
      ifk88seed=mmifk88seed
c if linked to PDFLIB, these quantities stay negative
      ifk88ih=-100
      ifk88ndns=-100
c forces the code to get Lambda value if not obtained from PDFLIB/MLMPDF
      xpdflam5=-1.d0
C Set system dependent parameters
      call sysdep
c----- vegas prints nothing
c      call nopr(0)
c Bases writes data file
      ibswrite=1
c-----
c Open the file collecting all the input parameter. This file is meant 
c to be converted in a command file in a subsequent run
      open(unit=11,file='vhglog',status=newver)
c

CCC =================================================================
CCC   added by fabian stoeckli
CCC   read string in from file
      open(unit=33,file=stfilename,status='old')
CCC =================================================================

      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter prefix for name of BASES files'
         read (*,*) pref
      else
         read(33,*) pref
      endif
      write(11,*) ''''//pref(1:ifk88istrl(pref))//'''',
     # '  ! prefix for BASES files'
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter prefix for name of event files'
         read (*,*) prefev
      else
         read(33,*) prefev
      endif
      close(33)
      write(11,*) ''''//prefev(1:ifk88istrl(prefev))//'''',
     # '  ! prefix for event files'
c----------------------------------------------------------
c Parameters of the run
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter pair ECM(GeV),fren[NLO],ffact[NLO],fren[MC],ffact[MC]'
      write(*,*)' fren=mu_ren/mu0'
      write(*,*)' ffact=mu_fac/mu0'
      write(*,*)' mu_ren=renormalization scale'
      write(*,*)' mu_fac=factorization scale'
      write(*,*)' mu0=reference scale'
         read(*,*) ecm,xren,xfh,xrenmc,xfhmc
      else
         ecm=mmecm
         xren=mmxren
         xfh=mmxfh
         xrenmc=mmxrenmc
         xfhmc=mmxfhmc
      endif
      write(11,'(5(1x,d10.4),1x,a)') ecm,xren,xfh,xrenmc,xfhmc
     #     ,'! Ecm, fren, ffact, frenmc, ffactmc'
      sh = ecm**2
c Will allow more flexibility in future versions
      xfh = xren
      xrenmc = xren
      xfhmc = xfh
      xren2 = xren**2
      xf2h1 = xfh**2
      xf2h2 = xfh**2
      xren2mc = xrenmc**2
      xf2h1mc = xfhmc**2
      xf2h2mc = xfhmc**2
c----------------------------------------------------------
c Select process; enter HERWIG code, and converts to NLO codes
      if(iinput.lt.2) then         
      write(*,*)' '
      write(*,*)'Enter -(1)2600-ID for WH production'
      write(*,*)'      -(1)2700-ID for ZH production'
         read(*,*) iprdct0hw
      else
         iprdct0hw=mmiproc
      endif
      write(11,'(1x,i6,27x,a)') iprdct0hw,'! -2600/2700=WH/ZH'
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter -1 for W^-'
      write(*,*)'       0 for Z'
      write(*,*)'       1 for W^+'
         read(*,*) iv0hw
      else
         iv0hw=mmivcode
      endif
      write(11,'(1x,i2,31x,a)') iv0hw,'! -1/0/1=W-/Z/W+'
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter IL=1..6, following HERWIG conventions'
      write(*,*)'      IL=7 for undecayed vector bosons'
         read(*,*) il0hw
      else
         il0hw=mmil1code
      endif
      write(11,'(1x,i2,31x,a)') il0hw,'! 1..6 -> V dec, 7 -> V undec'
      iprdct0hwtmp=mod(-iprdct0hw,10000)
      call getnloiproc(iprdct0hwtmp,iv0hw,il0hw)
      if(iprdct.eq.248.or.iprdct.eq.268.or.iprdct.eq.278)then
        idec=1
      elseif(iprdct.eq.240.or.iprdct.eq.260.or.iprdct.eq.270)then
        idec=0
      else
        write(*,*)'Inconsistent entries for process identification'
        stop
      endif
c Set constants
      call setpvh()
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter vector boson mass and width (GeV)'
         read(*,*)xm0v,gav
      else
         xm0v=mmxm0v
         gav=mmgav
      endif
      write(11,'(2(1x,d10.4),12x,a)') xm0v,gav,'! M_V, Ga_V'
      xm0v2 = xm0v**2
      xm0i2 = xm0v2
      gai = gav
      if(iinput.lt.2) then       
      write(*,*)' '
      write(*,*)'Enter GammaX, M_V(min), M_V(max)'
      write(*,*)'  If GammaX>0, the boson mass is chosen in the range'
      write(*,*)'      M0-GammaX*width < M_V < M0+GammaX*width'
      write(*,*)'  and M_V(min), M_V(max) are ignored'
      write(*,*)'  If GammaX<0, the boson mass is chosen in the range'
      write(*,*)'            M_V(min) < M_V < M_V(max)'
         read(*,*)vgammax,xmvlow,xmvupp
      else
         vgammax=mmv1gammax
         xmvlow=mmv1massinf
         xmvupp=mmv1masssup
      endif
      write(11,'(3(1x,d10.4),1x,a)') vgammax,xmvlow,xmvupp,
     #  '! GammaX, M_V(min), M_V(max)'
      if(vgammax.eq.0)then
        write(*,*)'Enter a non-null value for GammaX'
        stop
      endif
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter Higgs mass and width (GeV)'
         read(*,*)xm0h,gah
      else
         xm0h=mmxmh0
         gah=mmgah
      endif
      write(11,'(2(1x,d10.4),12x,a)') xm0h,gah,'! M_H, Ga_H'
      xm0h2 = xm0h**2
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter GammaX, M_H(min), M_H(max)'
      write(*,*)'  If GammaX>0, the Higgs mass is chosen in the range'
      write(*,*)'      M0-GammaX*width < M_H < M0+GammaX*width'
      write(*,*)'  and M_H(min), M_H(max) are ignored'
      write(*,*)'  If GammaX<0, the Higgs mass is chosen in the range'
      write(*,*)'            M_H(min) < M_H < M_H(max)'
         read(*,*)hgammax,xmhlow,xmhupp
      else
         hgammax=mmgammax
         xmhlow=mmxmhl
         xmhupp=mmxmhu
      endif
      write(11,'(3(1x,d10.4),1x,a)') hgammax,xmhlow,xmhupp,
     #  '! GammaX, M_H(min), M_H(max)'
      if(hgammax.eq.0)then
        write(*,*)'Enter a non-null value for GammaX'
        stop
      endif
c Zero width are only allowed in the case of undecayed vector boson;
c if the vector boson width is zero, so must be the Higgs width
      if(gav.ge.0.001d0)then
        iwidth=1
      else
        iwidth=0
        gav=0.d0
      endif
      if(iwidth.eq.1)then
        if(gah.eq.0.d0)then
          write(*,*)'Inconsistent entries for widths'
          stop
        endif
      else
        if(gah.ne.0.d0)then
          write(*,*)'Inconsistent entries for widths'
          stop
        elseif(idec.eq.0)then
          write(*,*)'Spin correlations require non-zero width'
          stop
        endif
      endif
c Sampling in t, the inverse of the BW function (see xbwmass2(t))
      tvsamp=1.d0
      thsamp=1.d0
c Change t(v,h)samp if efficiency problems arise
c      if(xmlow.lt.60.d0)tsamp=1.5d0-0.5*(max(xmlow,30.d0)-30.d0)/30.d0
c Quark and gluon masses (must be consistent with Herwig)
      do i=-5,21
        xmass(i)=0.d0
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter d, u, s, c, b, glu (Herwig) masses'
         read(*,*)xmass(1),xmass(2),xmass(3),xmass(4),xmass(5),xmass(21)
      else
         xmass(1)=mmxmass1
         xmass(2)=mmxmass2
         xmass(3)=mmxmass3
         xmass(4)=mmxmass4
         xmass(5)=mmxmass5
         xmass(21)=mmxmass21
      endif
      write(11,'(6(1x,d10.4),1x,a)') xmass(1),xmass(2),xmass(3),
     #  xmass(4),xmass(5),xmass(21),'! quark and gluon masses'
      do i=-5,-1
        xmass(i)=xmass(-i)
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     #  'Enter beam type for beam1 and beam2 (p, pbar, n, nbar):'
         read(*,*) part1,part2
      else
         part1=mmpart1
         part2=mmpart2
      endif
      write(11,'(1x,a,2x,a,19x,a)') ''''//part1//'''',
     #  ''''//part2//'''','! hadron types'
      strin=part1
      call fk88low_to_upp(strin,strout)
      part1=strout
      strin=part2
      call fk88low_to_upp(strin,strout)
      part2=strout
      if( (part1.ne.'P   ' .and. part1.ne.'PBAR' .and. 
     #     part1.ne.'N   ' .and. part1.ne.'NBAR') .or.
     #    (part2.ne.'P   ' .and. part2.ne.'PBAR' .and. 
     #     part2.ne.'N   ' .and. part2.ne.'NBAR') )then
        write(*,*)'This code only works for hadronic collisions'
        stop
      endif
      call getihpart(part1,itmpih)
      ih1=itmpih
      call getihpart(part2,itmpih)
      ih2=itmpih
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter group name and id number for PDF set'
         read(*,*)gname,idpdfset
      else
         gname=mmgname
         idpdfset=mmidpdfset
      endif
      write(11,'(1x,a,1x,i6,21x,a)') 
     # ''''//gname(1:ifk88istrl(gname))//'''',
     # idpdfset,'! PDF group and id number'
      strin=gname
      call fk88low_to_upp(strin,strout)
      if(strout.eq.'LHAPDF'.or.strout.eq.'LHAEXT')then
        lhapdf='FREEZE'
        if(strout.eq.'LHAEXT')lhapdf='EXTRAPOLATE'
        call setlhacblk(lhapdf)
        parm(1)='DEFAULT'
      else
        lhapdf='NOTLKD'
        parm(1)=gname
      endif
      value(1)=idpdfset
      call pdfset(parm,value)
      if(ipdfih.ne.1)then
        write(*,*)'PDFLIB could not understand the input'
        write(*,*)'Hadron type:',ipdfih
        stop
      endif
      if(ifk88ih.eq.-100.and.ifk88ndns.eq.-100)then
        if(lhapdf.eq.'NOTLKD')then
c the code is linked to PDFLIB; get the MLM pdf id number from
c ipdfih, ipdfgroup, and ipdfndns returned by PDFLIB in /w50511/
          call pdftomlm(ipdfih,ipdfgroup,ipdfndns,itmpih,itmpndns)
          ndns1=itmpndns
          ndns2=itmpndns
        elseif(lhapdf.eq.'FREEZE'.or.lhapdf.eq.'EXTRAPOLATE')then
c the code is linked to LHAPDF, which doesn't fill /w50511/
          call pdftomlm(ione,izero,idpdfset,itmpih,itmpndns)
          ndns1=itmpndns
          ndns2=itmpndns
        else
          write(*,*) 'Unknown lhapdf value: ',lhapdf
          stop
        endif
      else
c the code is linked to the interface to MLMPDF
        ndns1=ifk88ndns
        ndns2=ifk88ndns
      endif
c
      scheme='**'
      xlam=xpdflam5
c
      if(xlam.gt.0) then
         if(iinput.lt.2) then            
         write(*,*)' '
         write(*,*) 'Enter Lambda_QCD_5, < 0 for default'
            read(*,*) tmp
         else
            tmp=mmxlam
         endif
         write(11,'(1x,d10.4,23x,a)') tmp,'! Lambda_5, < 0 for default'
         if(tmp.gt.0) xlam=tmp
      else
         dowhile(xlam.le.0)
            if(iinput.lt.2) then
            write(*,*)' '
            write(*,*)'Enter Lambda_5_2loop'
               read(*,*) xlam
            else
               xlam=mmxlam
            endif
c            if (xlam.le.0) call prntsf
         enddo
         write(11,'(1x,d10.4,23x,a)') xlam,'! lambda'
      endif
C      write(*,*) 'Lambda_5=',xlam,' GeV'
c
 22   if(scheme.ne.'DI'.and.scheme.ne.'MS') then
         if(iinput.lt.2) then
         write(*,*)' '
         write(*,'(1x,a)') 'Enter scheme: ''DI'' or ''MS'''
            read(*,*) scheme
         else
            scheme=mmscheme
         endif
         if(scheme.ne.'DI'.and.scheme.ne.'MS') then
c            call prntsf
            goto 22
         endif
         write(11,'(1x,a,29x,a)') ''''//scheme//'''','! scheme'
      endif
C      write(*,*) 'Scheme=',scheme
      schhad1=scheme
      schhad2=scheme
c-----------------------------------------------------------------
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha and beta for the function G'
        write(*,*)' Defaults are: alpha=1, beta=0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_gfun,be_gfun
        ccc_gfun=0.d0
        write(11,'(3(1x,d10.4),1x,a)') al_gfun,be_gfun,ccc_gfun,
     #    '! alpha, beta, c'
      else
        al_gfun=1.d0
        be_gfun=0.1d0
        ccc_gfun=0.d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     # 'Enter accuracies for grid setup and for integral evaluation'
        read(*,*)ac1,ac2
        write(11,'(2(2x,d10.4),10x,a)') ac1,ac2,'! ac1,ac2'
      else
        ac1=0.2d0
        ac2=0.05d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     #    'For the computation of the MEs in the MC subtraction terms'
        write(*,*)'Enter 1 to use full 2->3 invariants'
        write(*,*)'      2 to use simplified invariants'
        write(*,*)' The default is 1'
        read(*,*) ia1ora2
        write(11,'(1x,i2,31x,a)') ia1ora2,
     #    '! 1 for full, 2 for simplified invariants'
      else
        ia1ora2=1
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'For the computation of alpha_S in NLO terms'
        write(*,*)'Enter 1 to set mu_0^2=M_V^2+pt_V^2'
        write(*,*)'      2 to set mu_0=M_V'
        write(*,*)'      3 to set mu_0=max(M_V,pt_V)'
        write(*,*)'      4 to set mu_0=max(M_V-pt_V,pt_V)'
        write(*,*)' The default is 1'
        read(*,*) inloscale
        write(11,'(1(1x,i8),25x,a)') inloscale,
     #    '! 1->mu_0=mh+pt, 2->mu_0=mh'
      else
        inloscale=1
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'For the computation of alpha_S in MC terms'
        write(*,*)'Enter 1 to set mu_0^2=M_V^2+pt_V^2'
        write(*,*)'      2 to set mu_0=M_V'
        write(*,*)'      3 to set mu_0=max(M_V,pt_V)'
        write(*,*)'      4 to set mu_0=max(M_V-pt_V,pt_V)'
        write(*,*)'      5 to set mu_0=M_V-pt_V/2'
        write(*,*)' The default is 5'
        read(*,*) imcscale
        write(11,'(1(1x,i8),25x,a)') imcscale,
     #    '! 1->mu_0=mh+pt, 2->mu_0=mh'
      else
        imcscale=5
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter the maximum number of events to generate;'
      write(*,*)'enter 0 to skip the event generation step'
         read(*,*)maxevt
      else
         maxevt=mmmaxevt
      endif
      write(11,'(1(1x,i8),25x,a)') maxevt,'! number of events'
      evgen=.true.
      if(maxevt.eq.0)then
        evgen=.false.
        maxevt=100000
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter 0 to have +1/-1 event weights'
      write(*,*)'      1 to normalize the weights, in such a way that'
      write(*,*)'        their sum is equal to the total rate'
         read(*,*)iwgtnorm
      else
         iwgtnorm=mmiwgtnorm
      endif
      write(11,'(1(1x,i8),25x,a)') iwgtnorm,
     #  '! 0 => wgt=+1/-1, 1 otherwise'
c iseed0 is the seed for the integration step, iseed is the seed
c for the event generation step
      iseed0=12345
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter the seed for random numbers; it will be used'
      write(*,*)'to generate events. Enter 0 for default'
         read(*,*)iseed
      else
         iseed=mmiseed
      endif
      write(11,'(1(1x,i8),25x,a)') iseed,'! seed for rnd numbers'
      if(iseed.lt.0)then
        stop
      elseif(iseed.eq.0)then
        iseed=iseed0
      endif
c Here, assume that the unweighting efficiency is larger than 10%
      maxtrials=10*maxevt
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to use standard subtraction'
        write(*,*)'      1 to use zeta subtraction'
        read(*,*)isubttype
        write(11,'(1(1x,i8),25x,a)') isubttype,
     #    '! 0=subt, 1=zeta subt'
      else
        isubttype=0
      endif
      if(iinput.eq.1)then
        if(isubttype.eq.0)then
          write(*,*)' '
          write(*,*)'enter xicut and delta (defaults: 1, 2)'
          read(*,*)xicut,delta
          write(11,'(2(2x,d10.4),10x,a)') xicut,delta,'! xicut,delta'
        else
          write(*,*)' '
          write(*,*)'Enter zi ( [ 2*kt(gluon)/sqrt(shat) ]^2 < zi )'
          read(*,*) etacut
          write(11,'(1x,d10.4,23x,a)') etacut,'! zi'
          xicut = 1.d0
          delta = 1.d0
        endif
      else
        if(isubttype.eq.0)then
          xicut = 1.d0
          delta = 2.d0
        else
          etacut = 1.d0
          xicut = 1.d0
          delta = 1.d0
        endif
      endif
      xicutss = xicut
      deltas = 0
      deltac = 0
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to preserve rapidity'
        write(*,*)'      1 to preserve longitudinal momentum'
        read(*,*)iprespl
        write(11,'(1(1x,i8),25x,a)') iprespl,'! 0=y, 1=k_3 preserved'
      else
        iprespl=0
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter 0 to use running alpha_EM'
      write(*,*)'      1 to use alpha_EM=1/137.0359895'
         read(*,*)ifxdaem
      else
         ifxdaem=mmaemrun
      endif
      write(11,'(1(1x,i8),25x,a)') ifxdaem,
     #  '! 0=running, 1=fixed alpha_EM'
      if(ifxdaem.ne.0.and.ifxdaem.ne.1)then
        write(*,*)'No such option for alpha_em'
        stop
      endif
      if(ifxdaem.eq.0)ze2=0.d0
c---------------------------------------------------------------
c Initialize parameters, such as labelling for parton processes
      call parsetpvh()
c Select partonic subprocess
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 2 for qq, 3 for qg, 0 for all'
        read(*,*) iproc
        write(11,'(1x,i2,31x,a)') iproc,'! 2=qq, 3=qg, 0=all'
      else
        iproc=0
      endif
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to leave the partons massless'
        write(*,*)'      2 to put partons on mass shell'
        read(*,*) ionshell
        write(11,'(1x,i1,32x,a)') 
     #      ionshell,'! 0=massless, 2=massive partons'
      else
        ionshell=2
      endif
      if(ionshell.ne.0.and.ionshell.ne.2) then
        write(*,*) 'Error: enter 0 or 2.'
        stop
      endif
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to leave the leptons massless'
        write(*,*)'      2 to put leptons on mass shell'
        read(*,*) ilepmass
        write(11,'(1x,i1,32x,a)') 
     #      ilepmass,'! 0=massless, 2=massive leptons'
      else
        ilepmass=2
      endif
      if(ilepmass.ne.0.and.ilepmass.ne.2) then
        write(*,*) 'Error: enter 0 or 2.'
        stop
      endif
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter number of iterations'
      write(*,*)'for grid setup and for integral evaluation;'
      write(*,*)'set either one to 0 to skip the integration step'
         read(*,*) it1,it2
      else
         it1=mmit1
         it2=mmit2
      endif
      write(11,'(2(1x,i4),24x,a)') it1,it2,'! itmx1,itmx2'
      iseld=1
      if(it1.eq.0.or.it2.eq.0)iseld=0
c---------------------------------------------------------------
c Integration parameters
c
      if(iinput.eq.1)then
        if(iseld.eq.1)then
          write(*,*)' '
          write(*,*)
     #     'Enter number of calls for bases (<0 for default)'
          read(*,*)ncl3
          write(11,'(1x,i9,24x,a)')ncl3,'! # of calls for bases'
        endif
      else
        ncl3=-1
      endif
      if(ncl3.lt.0)then
        if(iwidth.eq.1.and.idec.eq.0)then
          ncl3=250000
        else
          ncl3=120000
        endif
      endif
c---- close logfile
      close(11)
c----------------------------------------------------------------
c  *********************  START INTEGRATION *********************
c----------------------------------------------------------------
      ifuntype=0
      loproc = 2
      maproc = 3
      if(iproc.ne.0) then
        loproc=iproc
        maproc=iproc
      endif
      prefn = pref
      prefnev = prefev
c tau generated according to a flat distribution in (1/tau)**nsamp
      nsamp = 1
c
      if(iwidth.eq.1)then
        if(idec.eq.0)then
          ndim=10
          nwild=10
        else
          ndim=8
          nwild=8
        endif
        xmv2=-1.d10
        xmh2=-1.d10
        if(vgammax.ge.0)then
          xmvlow2=(max(0.d0,xm0v-vgammax*gav))**2
          xmvupp2=(xm0v+vgammax*gav)**2
        else
          xmvlow2=xmvlow**2
          xmvupp2=xmvupp**2
        endif
        xmvlow2=max(100.d0,xmvlow2)
        xmvupp2=min(sh*0.95,xmvupp2)
        if(xmvlow2.gt.xmvupp2)then
          write(*,*)'Error in pair mass range'
          stop
        endif
        bwvmdpl=xmvupp2-xm0v2
        bwvmdmn=xm0v2-xmvlow2
        bwvfmpl=atan(bwvmdpl/(xm0v*gav))
        bwvfmmn=atan(bwvmdmn/(xm0v*gav))
        bwvdelf=(bwvfmpl+bwvfmmn)/pi
        if(hgammax.ge.0)then
          xmhlow2=(max(0.d0,xm0h-hgammax*gah))**2
          xmhupp2=(xm0h+hgammax*gah)**2
        else
          xmhlow2=xmhlow**2
          xmhupp2=xmhupp**2
        endif
        xmhlow2=max(100.d0,xmhlow2)
        xmhupp2=min(sh*0.95,xmhupp2)
        if(xmhlow2.gt.xmhupp2)then
          write(*,*)'Error in pair mass range'
          stop
        endif
        bwhmdpl=xmhupp2-xm0h2
        bwhmdmn=xm0h2-xmhlow2
        bwhfmpl=atan(bwhmdpl/(xm0h*gah))
        bwhfmmn=atan(bwhmdmn/(xm0h*gah))
        bwhdelf=(bwhfmpl+bwhfmmn)/pi
      else
        ndim=6
        nwild=6
        xmv2=xm0v2
        xmh2=xm0h2
        rohlim=(xm0v+xm0h)**2/sh
        if(xicut.gt.(1-rohlim))xicut=1-rohlim
        bwvfmpl=0.d0
        bwvfmmn=0.d0
        bwvdelf=0.d0
        bwhfmpl=0.d0
        bwhfmmn=0.d0
        bwhdelf=0.d0
      endif
c Perform the integration step
      if(iseld.eq.1)then
        xtotal=0.d0
        ytotal=0.d0
        dtot=0.d0
        xares=0.d0
        yares=0.d0
        xbres=0.d0
        ybres=0.d0
        mx_of_evta=0
        mx_of_evtb=0
c
        ifuntype=1
        call fk88strcat(prefn,'_a',fnamea)
        if(iwidth.eq.1.and.idec.eq.0)then
          call run_bases(sig5afw_dec,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        elseif(iwidth.eq.1.and.idec.eq.1)then
          call run_bases(sig5afw_und,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        elseif(iwidth.eq.0.and.idec.eq.1)then
          call run_bases(sig5azw_und,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        else
          write(*,*)'Unknown options',iwidth,idec
          stop
        endif
        if(basesoutput.gt.0) then
        write(*,*)'   '
        write(*,*)'|integral[a]|:',av3a,' +- ',d3a
        write(*,*)' integral[a] :',av3nega,' +- ',d3nega
        endif
        xares=av3a
        yares=av3nega
        xtotal=xtotal+xares
        ytotal=ytotal+yares
        dtot=dtot+d3nega**2
c
        ifuntype=2
        call fk88strcat(prefn,'_b',fnameb)
        if(iwidth.eq.1.and.idec.eq.0)then
          call run_bases(sig5bfw_dec,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        elseif(iwidth.eq.1.and.idec.eq.1)then
          call run_bases(sig5bfw_und,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        elseif(iwidth.eq.0.and.idec.eq.1)then
          call run_bases(sig5bzw_und,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        endif
        if(basesoutput.gt.0) then
        write(*,*)'   '
        write(*,*)'|integral[b]|:',av3b,' +- ',d3b
        write(*,*)' integral[b] :',av3negb,' +- ',d3negb
        endif
        xbres=av3b
        ybres=av3negb
        xtotal=xtotal+xbres
        ytotal=ytotal+ybres
        dtot=dtot+d3negb**2
c
        avtot=ytotal
        dtot=sqrt(dtot)
        call fk88strcat(prefn,'.integrals',fname)
        open(unit=21,file=fname,
     #       form='formatted',status='unknown')
        write(21,240)xares
        write(21,240)xbres
        write(21,240)yares
        write(21,240)ybres
        close(21)
 240    format(1x,d14.8)
      endif
c Sanity check
      if(isubttype.eq.1.and.delta.ne.1.d0)then
        write(*,*)'Fatal error: xicut, delta=',xicut,delta
        stop
      endif
      if(iseld.eq.0)then
c Read integrals from disk only if the integration step has been skipped
        call fk88strcat(prefn,'.integrals',fname)
        open(unit=21,file=fname,
     #       form='formatted',status='old')
        read(21,240)xares
        read(21,240)xbres
        read(21,240)yares
        read(21,240)ybres
        close(21)
      endif
c
c Generates events when evgen=.true.; if evgen=.false., maxevt=100000 in
c order to estimate the number of negative weights
      if(maxevt.ne.0)then
        ntotal=0
        xtotal=0.d0
        ytotal=0.d0
        xtotal=xtotal+xares+xbres
        ytotal=ytotal+yares+ybres
        avtot=ytotal
        if(iseld.eq.0)dtot=0.d0
c For future upgrades, define the weights of H and S events; this is 
c necessary when the relative number of H and S events is not generated
c according to total rates
        if(iwgtnorm.eq.0)then
          wgtaev=1.d0
          wgtbev=1.d0
        else
          wgtaev=xtotal/dfloat(maxevt)
          wgtbev=xtotal/dfloat(maxevt)
        endif
        mx_of_evta=int(maxevt*xares/xtotal)
        mx_of_evtb=int(maxevt*xbres/xtotal)
        ntotal=ntotal+mx_of_evta+mx_of_evtb
        ndiff=maxevt-ntotal
        if(ndiff.gt.0)mx_of_evta=mx_of_evta+ndiff
        if(ndiff.lt.0)then
          write(6,*)'Fatal error:',maxevt,ntotal
          stop
        endif
        if(basesoutput.gt.0) then
        if(evgen)then
          write(*,*)'  '
          write(*,*)
     #  'The following number of events will be generated'
          write(*,*)'# events[a]:',mx_of_evta
          write(*,*)'# events[b]:',mx_of_evtb
        endif
        write(*,*)'  '
        write(*,*)
     #  'Estimated fractions of events with negative weights'
        endif
        evfrac=0.d0
        evprcfrac=(xares-yares)/
     #            (xares+yares)
        evprcfrac=evprcfrac/(1+evprcfrac)
        evfrac=evfrac+evprcfrac*mx_of_evta
        if(basesoutput.gt.0) then
        write(*,*)'Events[a]: w<0/all:',evprcfrac
        endif
        evprcfrac=(xbres-ybres)/
     #            (xbres+ybres)
        evprcfrac=evprcfrac/(1+evprcfrac)
        if(basesoutput.gt.0) then
        write(*,*)'Events[b]: w<0/all:',evprcfrac
        endif
        evfrac=evfrac+evprcfrac*mx_of_evtb
        evfrac=evfrac/dfloat(maxevt)
        if(basesoutput.gt.0) then
        write(*,*)'Events[all]: w<0/all:',evfrac
        endif
c
        if(.not.evgen)goto 111
        fname=prefnev
        call fk88strcat(fname,'_a.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evta
        close(22)
        call fk88strcat(fname,'_b.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtb
        close(22)
c
        fname=prefn
        fnamev=prefnev
c
        iwrong=0
        iwrong1=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=1
        call fk88strcat(fname,'_a',fnamea)
        call fk88strcat(fnamev,'_a.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        if(iwidth.eq.1.and.idec.eq.0)then
          call run_spring(sig5afw_dec,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        elseif(iwidth.eq.1.and.idec.eq.1)then
          call run_spring(sig5afw_und,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        elseif(iwidth.eq.0.and.idec.eq.1)then
          call run_spring(sig5azw_und,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        endif
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[a]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(7*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(5*nevts)
            endif
          enddo
        endif
c
        iwrong=0
        iwrong1=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=2
        call fk88strcat(fname,'_b',fnameb)
        call fk88strcat(fnamev,'_b.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        if(iwidth.eq.1.and.idec.eq.0)then
          call run_spring(sig5bfw_dec,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        elseif(iwidth.eq.1.and.idec.eq.1)then
          call run_spring(sig5bfw_und,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        elseif(iwidth.eq.0.and.idec.eq.1)then
          call run_spring(sig5bzw_und,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        endif
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[b]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(6*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(4*nevts)
            endif
          enddo
        endif
c write a single event file
        iunita=21
        call fk88strcat(prefnev,'_a.events',fname1)
        open(unit=iunita,file=fname1,form='formatted',status='old')
        read(iunita,250)mx_of_evta
        iunitb=22
        call fk88strcat(prefnev,'_b.events',fname1)
        open(unit=iunitb,file=fname1,form='formatted',status='old')
        read(iunitb,250)mx_of_evtb
c
        call fk88strcat(prefnev,'.events',fname1)
        ioutput=30
        open(unit=ioutput,file=fname1,form='formatted',
     #       status='unknown')
c Write all the quantities which identify the run
        write(ioutput,801)
     #    ecm,xren,xfh,xrenmc,xfhmc,
     #    '--> CM energy, muR/mu0[NLO], muF/mu0[NLO], '//
     #    'muR/mu0[MC], muF/mu0[MC]'
        write(ioutput,802)abs(iprdct0hw),'--> 2600/2700=WH/ZH'
        write(ioutput,810)iv0hw,il0hw,'--> IV (-1,0,1), IL (1,..,7)'
        write(ioutput,813)xm0v,gav,vgammax,'--> M_V, Ga_V, GammaX(V)'
        write(ioutput,813)xm0h,gah,hgammax,'--> M_H, Ga_H, GammaX(H)'
        write(ioutput,804)xmass(1),xmass(2),
     #                    xmass(3),xmass(4),
     #                    xmass(5),xmass(21),
     #                    '--> quark and gluon masses'
        write(ioutput,805)part1,part2,'--> colliding particles'
        write(ioutput,806)gname(1:8),idpdfset,
     #    '--> PDF group and id number'
        write(ioutput,807)xlam,scheme,'--> Lambda_5, scheme'
        write(ioutput,811)'P,M','--> Format of v3.1 and higher'
        write(ioutput,250)maxevt
        itot=maxevt
        do ii=1,maxevt
          call whichone(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
          call retrieve_evenvh(iunit,ii,dummy)
          call store_evenvh(ioutput,xmone)
        enddo
        call crosscheck(itot,mx_of_evta,mx_of_evtb)
        close(iunita)
        close(iunitb)
        close(ioutput)
 111    continue
      endif
      if(basesoutput.gt.0) then
      if(iprdct.ge.270)then
        write(*,*) '   '
        write(*,*)'Decay width in MSbar and on-shell schemes'
        write(*,*)'MSbar is used in the present computation'
        write(*,*)'MSbar:    Gamma =',wdtzmsb
        write(*,*)'On-shell: Gamma =',wdtzon
        if(iwidth.eq.1)then
          write(*,*)'MSbar:    Gamma/Gamma_tot =',wdtzmsb/gav
          write(*,*)'On-shell: Gamma/Gamma_tot =',wdtzon/gav
        endif
      else
        write(*,*) '   '
        write(*,*)'Decay width in MSbar and on-shell schemes'
        write(*,*)'MSbar is used in the present computation'
        write(*,*)'MSbar:    Gamma =',wdtwmsb
        write(*,*)'On-shell: Gamma =',wdtwon
        if(iwidth.eq.1)then
          write(*,*)'MSbar:    Gamma/Gamma_tot =',wdtwmsb/gav
          write(*,*)'On-shell: Gamma/Gamma_tot =',wdtwon/gav
        endif
      endif
      endif
      if(iwidth.eq.1.and.iverbose.eq.1)then
        bwvdelfmax=( atan((sh-xm0v2)/(xm0v*gav))+
     #               atan(xm0v2/(xm0v*gav)) )/pi
        bwhdelfmax=( atan((sh-xm0h2)/(xm0h*gah))+
     #               atan(xm0h2/(xm0h*gah)) )/pi
        write(*,*) '   '
        write(*,*)
     # 'Cross section left out from the invariant mass integrations'
        write(*,*)'For -Inf<M<Inf:'
        write(*,*)'  M_V     --> ',1-bwvdelf,' %'
        write(*,*)'  M_H     --> ',1-bwhdelf,' %'
        write(*,*)'  M_V+M_H --> ',1-bwvdelf*bwhdelf,' %'
        write(*,*)'For 0<M<Sqrt(S):'
        write(*,*)'  M_V     --> ',1-bwvdelf/bwvdelfmax,' %'
        write(*,*)'  M_H     --> ',1-bwhdelf/bwhdelfmax,' %'
        write(*,*)'  M_V+M_H --> ',1-bwvdelf*bwhdelf/
     #                               (bwvdelfmax*bwhdelfmax),' %'
      endif
      if(basesoutput.gt.0) then
      write(*,*) '   '
      write(*,*) 'Total for fully inclusive'
      write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
      write(*,202) 
      write(*,270)xm0v,gav,vgammax,xm0h,gah,hgammax
      write(*,201) 'tot'
      write(*,300)ecm,xfh,xren,avtot,dtot
      endif
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #  '  nl=',i2,'  lambda5=',d10.4)
 201  format(' ecm or ebeam  xf   xr   ',a,
     # '        err    ')
 202  format(' M_V       Ga_V      GammaX(V)    ',
     #       ' M_H       Ga_H      GammaX(H) ')
 270  format(3(1x,1pd9.3),4x,3(1x,1pd9.3))
 300  format((1x,1pd9.3),4x,2(1x,0pf4.2),2(1x,0pd10.4))
 250  format(1x,i8)
 801  format(5(1x,d10.4),1x,a)
 802  format(1x,i6,1x,a)
 803  format(2(1x,d10.4),1x,a)
 804  format(6(1x,d10.4),1x,a)
 805  format(2(1x,a4),1x,a)
 806  format(1x,a8,1x,i6,1x,a)
 807  format(1x,d10.4,1x,a2,1x,a)
 810  format(2(1x,i2),1x,a)
 811  format(1x,a3,1x,a)
 813  format(3(1x,d10.4),1x,a)
      end


C      subroutine getset(str,ndns,ih)
C      implicit real * 8 (a-h,o-z)
C      character * (*) str
C 2    write(*,*) str
C      write(*,*)
C     # '   (< 0 for a display of the features of the various sets'
C      read(*,*) ndns
C      if(ndns.lt.0) then
C        call prntsf
C        go to 2
C      endif
C      end


C      subroutine toend(iunit)
C      ios = 0    
C      dowhile(ios.eq.0)
C         read(unit=iunit,fmt='(1x)',iostat=ios)
C      enddo                        
C      end


C      subroutine getihpart(part,ih)
Cc Converts particle naming conventions, for Herwig to MLM
C      implicit real * 8 (a-h,o-z)
C      character * 4 part
Cc
C      ih=-100
C      if(part.eq.'P   ')then
C        ih=1
C      elseif(part.eq.'PBAR')then
C        ih=-1
C      elseif(part.eq.'N   ')then
C        ih=2
C      elseif(part.eq.'NBAR')then
C        ih=-2
C      elseif(part.eq.'GAMA')then
C        ih=4
C      elseif(part.eq.'E-  ')then
C        ih=5
C      else
C        write(*,*)'Error in getihpart'
C        write(*,*)'No such particle in MLM:',part
C        stop
C      endif
C      return
C      end


      subroutine strfvh(x1,x2,sf)
c Return parton densities through the matrix
c  sf(idr,jproc,itype), with the following conventions:
c  idr=1 -> qqbar, qg
c  idr=2 -> 0    , qbarg
c  idr=3 -> qbarq, gq
c  idr=4 -> 0    , gqbar
c  jproc=2,3 -> qqbar, qg processes respectively
c  itype -> identifies the individual contribution to a given jproc
      implicit none
      real*4 fh1x1(-5:5),fh2x2(-5:5),smuf2h1,smuf2h2
      real * 8 pi,x1,x2,sf(4,2:3,6)
      integer ih1,ih2,ndns1,ndns2,ii,jproc,itype,iprdct
      parameter(pi=3.14159265358979312D0)
      include 'vhgcblks.h'
      common/strfvh0/ih1,ih2,ndns1,ndns2
      common/ciprdct/iprdct
      real*8 zel(1:6),zel2(1:6)
      real * 8 ckm2(1:6,1:6)
      common/charges/zel,zel2
      common/cckm2/ckm2
      integer ipdfscale
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c
      do ii=1,4
        do jproc=2,3
          do itype=1,6
            sf(ii,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      if(ipdfscale.eq.1)then
        smuf2h1=sngl(xmuf2h1)
        smuf2h2=sngl(xmuf2h2)
      elseif(ipdfscale.eq.2)then
        smuf2h1=sngl(xmumcf2h1)
        smuf2h2=sngl(xmumcf2h2)
      else
        write(*,*)'Fatal error in strfvh: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
c
c jproc=2
      if(iprdct.eq.270.or.iprdct.eq.278)then
        do ii=1,nl
          sf(1,2,ii)=dble(fh1x1( ii) * fh2x2(-ii))
          sf(3,2,ii)=dble(fh1x1(-ii) * fh2x2( ii))
        enddo
      elseif(iprdct.eq.240.or.iprdct.eq.248)then
        sf(1,2,1)=ckm2(1,2)*dble(fh1x1( 1) * fh2x2(-2)) 
        sf(1,2,2)=ckm2(1,3)*dble(fh1x1( 1) * fh2x2(-3)) 
        sf(1,2,3)=ckm2(1,5)*dble(fh1x1( 1) * fh2x2(-5)) 
        sf(1,2,4)=ckm2(4,2)*dble(fh1x1( 4) * fh2x2(-2)) 
        sf(1,2,5)=ckm2(4,3)*dble(fh1x1( 4) * fh2x2(-3)) 
        sf(1,2,6)=ckm2(4,5)*dble(fh1x1( 4) * fh2x2(-5)) 
        sf(3,2,1)=ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 1)) 
        sf(3,2,2)=ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 1)) 
        sf(3,2,3)=ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 1)) 
        sf(3,2,4)=ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 4)) 
        sf(3,2,5)=ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 4)) 
        sf(3,2,6)=ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 4)) 
      elseif(iprdct.eq.260.or.iprdct.eq.268)then
        sf(1,2,1)=ckm2(1,2)*dble(fh1x1( 2) * fh2x2(-1)) 
        sf(1,2,2)=ckm2(4,2)*dble(fh1x1( 2) * fh2x2(-4)) 
        sf(1,2,3)=ckm2(1,3)*dble(fh1x1( 3) * fh2x2(-1)) 
        sf(1,2,4)=ckm2(4,3)*dble(fh1x1( 3) * fh2x2(-4)) 
        sf(1,2,5)=ckm2(1,5)*dble(fh1x1( 5) * fh2x2(-1)) 
        sf(1,2,6)=ckm2(4,5)*dble(fh1x1( 5) * fh2x2(-4)) 
        sf(3,2,1)=ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 2)) 
        sf(3,2,2)=ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 2)) 
        sf(3,2,3)=ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 3)) 
        sf(3,2,4)=ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 3)) 
        sf(3,2,5)=ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 5)) 
        sf(3,2,6)=ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 5)) 
      else
        write(*,*)'Error in strfvh: unknown IPROC',iprdct
        stop
      endif
c jproc=3
      if(iprdct.eq.270.or.iprdct.eq.278)then
        do ii=1,nl
          sf(1,3,ii)=dble(fh1x1( ii) * fh2x2(  0))
          sf(2,3,ii)=dble(fh1x1(-ii) * fh2x2(  0))
          sf(3,3,ii)=dble(fh1x1(  0) * fh2x2( ii))
          sf(4,3,ii)=dble(fh1x1(  0) * fh2x2(-ii))
        enddo
      elseif(iprdct.eq.240.or.iprdct.eq.248)then
        sf(1,3,1)=ckm2(1,2)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,2)=ckm2(1,3)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,3)=ckm2(1,5)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,4)=ckm2(4,2)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,5)=ckm2(4,3)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,6)=ckm2(4,5)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(2,3,1)=ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,2)=ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,3)=ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,4)=ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,5)=ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(2,3,6)=ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(3,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,2)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,3)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,4)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,5)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(4,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,2)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,3)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,4)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,5)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-5)) 
        sf(4,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-5)) 
      elseif(iprdct.eq.260.or.iprdct.eq.268)then
        sf(1,3,1)=ckm2(1,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,2)=ckm2(4,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,3)=ckm2(1,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,4)=ckm2(4,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,5)=ckm2(1,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(1,3,6)=ckm2(4,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(2,3,1)=ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,2)=ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,3)=ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,4)=ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,5)=ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,6)=ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(3,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,2)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,3)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,4)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,5)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(3,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(4,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,2)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,3)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,4)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,5)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-4)) 
      else
        write(*,*)'Error in strfvh: unknown IPROC',iprdct
        stop
      endif
c
      return
      end
c
c
c NLO cross section
c
c
      function sig5azw_und(xx)
      implicit none
      real * 8 sig5azw_und,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5a_und
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1.d0
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(4)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5azw_und = tot5a_und(s,x,y,cth1,cth2,xjac)
      return
      end


      function sig5afw_und(xx)
      implicit none
      real * 8 sig5afw_und,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(8)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5a_und,tt,xbwmass3
c
c xx(1) --> xmv2, xx(2) --> xmh2, xx(3) --> tau, xx(4)-->ycm, xx(5) --> x, 
c xx(6) --> y, xx(7) --> cth1, xx(8) --> cth2
c
      xjac = 1.d0
c
      tt=xx(1)
      if(tvsamp.ne.1)then
        write(*,*)'Error #1 in sig5afw_und'
        stop
      endif
      xmv2=xbwmass3(tt,xm0v2,gav,bwvdelf,bwvfmmn)
      xjac=xjac*bwvdelf
      if(xmv2.lt.xmvlow2.or.xmv2.gt.xmvupp2)then
        write(*,*)'Error #2 in sig5afw_und',xmv2
        stop
      endif
c
      tt=xx(2)
      if(thsamp.ne.1)then
        write(*,*)'Error #3 in sig5afw_und'
        stop
      endif
      xmh2=xbwmass3(tt,xm0h2,gah,bwhdelf,bwhfmmn)
      xjac=xjac*bwhdelf
      if(xmh2.lt.xmhlow2.or.xmh2.gt.xmhupp2)then
        write(*,*)'Error #4 in sig5afw_und',xmh2
        stop
      endif
c
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(5)**2
      xjac = xjac * xx(5) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(6)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(3)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(4)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(7),cth1,xjac,rox)
c
      th2 = xx(8) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5afw_und = tot5a_und(s,x,y,cth1,cth2,xjac)
      return
      end


      function sig5afw_dec(xx)
      implicit none
      real * 8 sig5afw_dec,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(10)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5a_dec,tt,xbwmass3,bwfunc,
     #  cthl,phil
c
c xx(1) --> xmv2, xx(2) --> xmh2, xx(3) --> tau, xx(4)-->ycm, xx(5) --> x, 
c xx(6) --> y, xx(7) --> cth1, xx(8) --> cth2, xx(9) --> cthl, xx(10) --> phil
c
      xjac = 1.d0
c
      tt=xx(1)
      if(tvsamp.ne.1)then
        write(*,*)'Error #1 in sig5afw_dec'
        stop
      endif
      xmv2=xbwmass3(tt,xm0v2,gav,bwvdelf,bwvfmmn)
      xjac=xjac*bwvdelf/bwfunc(xmv2,xm0v2,gav)
      if(xmv2.lt.xmvlow2.or.xmv2.gt.xmvupp2)then
        write(*,*)'Error #2 in sig5afw_dec',xmv2
        stop
      endif
c
      tt=xx(2)
      if(thsamp.ne.1)then
        write(*,*)'Error #3 in sig5afw_dec'
        stop
      endif
      xmh2=xbwmass3(tt,xm0h2,gah,bwhdelf,bwhfmmn)
      xjac=xjac*bwhdelf
      if(xmh2.lt.xmhlow2.or.xmh2.gt.xmhupp2)then
        write(*,*)'Error #4 in sig5afw_dec',xmh2
        stop
      endif
c
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(5)**2
      xjac = xjac * xx(5) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(6)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(3)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(4)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)then
        ze2=4*pi*xalfaem(xmi2)
        ze2v=4*pi*xalfaem(xmv2)
      endif
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(7),cth1,xjac,rox)
c
      th2 = xx(8) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      cthl = 2*xx(9)-1
      xjac = 2*xjac
c
      phil = 2*pi*xx(10)
      xjac = 2*pi*xjac
c
      sig5afw_dec = tot5a_dec(s,x,y,cth1,cth2,cthl,phil,xjac)
      return
      end


      function tot5a_und(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5a_und,tot5as_und,tot5az_und,s,x,yi,cth1,cth2,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5as_und(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5az_und(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5a_und:',isubttype
        stop
      endif
      tot5a_und=tmp
      return
      end


      function tot5a_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      implicit none
      real * 8 tot5a_dec,tot5as_dec,tot5az_dec,s,x,yi,cth1,cth2,
     #  cthl,phil,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5as_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5az_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      else
        write(*,*)'Fatal error in tot5a_dec:',isubttype
        stop
      endif
      tot5a_dec=tmp
      return
      end


      function tot5as_und(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5as_und,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      integer idec
      common/cidec/idec
      real * 8 sf(4,2:3,6)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,ro,beta,rox,betax,xphsp_ns,
     #  xphsp_s,xphspb,x1,x2,tk,uk,q1q,q2q,dummy,xnorm,ffunval5,
     #  ffunction5,zg2_mc,zgmu2_vh,ytmp,zhwfct,zherw_spl,x1t,x2t,
     #  x1soft,x2soft,x1x2j,x1x2jac,zherw_smn,xint,xtmp,zg2_nlo,
     #  zgmu2_nvh,xsum
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,i2b,
     #  itoosoftkin,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
      ro = 2*(xmv2+xmh2)/s-(xmv2-xmh2)**2/s**2
      beta = sqrt(1-ro)
      rox = 2*(xmv2+xmh2)/sx-(xmv2-xmh2)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4)
      xphspb = xjac * betax/(32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,zero,zero,str,
     #             tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
        zg2_nlo = zgmu2_nvh()
        ipdfscale=1
        call strfvh(x1,x2,sf)
        xnorm = ze2**2 * zg2_nlo * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          call freal_und(s,x,yi,tk,uk,q1q,q2q,jproc,vecre,axlre)
          do j=1,4
            do itype=1,6
              do i=3,3
                vv(j,jproc,itype)=vv(j,jproc,itype)+
     #            sf(j,jproc,itype)*xnorm*
     #            ( vecre(j,i)*xveccoef(itype,i) +
     #              axlre(j,i)*xaxlcoef(itype,i) )
              enddo
            enddo
          enddo
        enddo
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
        ffunval5 = ffunction5(x,yi)
        if(ffunval5.ne.0.d0)then
          zg2_mc = zgmu2_vh()
          ipdfscale=2
          xnorm = ze2**2 * zg2_mc * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          ytmp=1.d0
          zhwfct=zherw_spl(x,yi)
          x1t=x1soft(x1,x2,x,yi)/zhwfct
          x2t=x2soft(x1,x2,x,yi)
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvh(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                     zero,zero,x1,x2,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=3,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
          ytmp=-1.d0
          zhwfct=zherw_smn(x,yi)
          x1t=x1soft(x1,x2,x,yi)
          x2t=x2soft(x1,x2,x,yi)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvh(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                     zero,zero,x1,x2,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=3,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checkvh(xsum,dummy,iret)
      if(iret.eq.1)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,zero,zero,str,
     #             tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,zero,zero,str,
     #               tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5as_und=abs(xint)
c
      return
      end


      function tot5az_und(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5az_und,xs,xx,xyi,xcth1,xcth2,xjac
c
      tot5az_und=0
      write(*,*)'Zeta subtraction non implemented'
      stop
      return
      end


      function tot5as_dec(xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac)
      implicit none
      real * 8 tot5as_dec,xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      integer idec
      common/cidec/idec
      real * 8 sf(4,2:3,6)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,ro,beta,rox,betax,xphsp_ns,
     #  xphsp_s,xphspb,x1,x2,tk,uk,q1q,q2q,dummy,xnorm,ffunval5,
     #  ffunction5,zg2_mc,zgmu2_vh,ytmp,zhwfct,zherw_spl,x1t,x2t,
     #  x1soft,x2soft,x1x2j,x1x2jac,zherw_smn,xint,xtmp,zg2_nlo,
     #  zgmu2_nvh,xsum,cthl,phil,v1a,v1b,v1c
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,i2b,
     #  itoosoftkin,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      phil = xphil
      cthl = xcthl
      sx = x*s
      xii = 1-x
      ro = 2*(xmv2+xmh2)/s-(xmv2-xmh2)**2/s**2
      beta = sqrt(1-ro)
      rox = 2*(xmv2+xmh2)/sx-(xmv2-xmh2)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4) * 1/(2*pi*32*pi2)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4) * 1/(2*pi*32*pi2)
      xphspb = xjac * betax/(32*pi2) * 1/(2*pi*32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        zg2_nlo = zgmu2_nvh()
        ipdfscale=1
        call strfvh(x1,x2,sf)
        xnorm = ze2**2 * ze2v * zg2_nlo * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          call freal_dec(s,x,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                   jproc,vecre,axlre)
          do j=1,4
            do itype=1,6
              do i=3,3
                vv(j,jproc,itype)=vv(j,jproc,itype)+
     #            sf(j,jproc,itype)*xnorm*
     #            ( vecre(j,i)*xveccoef(itype,i) +
     #              axlre(j,i)*xaxlcoef(itype,i) )
              enddo
            enddo
          enddo
        enddo
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
        ffunval5 = ffunction5(x,yi)
        if(ffunval5.ne.0.d0)then
          zg2_mc = zgmu2_vh()
          ipdfscale=2
          xnorm = ze2**2 * ze2v * zg2_mc * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          ytmp=1.d0
          zhwfct=zherw_spl(x,yi)
          x1t=x1soft(x1,x2,x,yi)/zhwfct
          x2t=x2soft(x1,x2,x,yi)
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvh(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                     phil,cthl,x1,x2,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=3,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
          ytmp=-1.d0
          zhwfct=zherw_smn(x,yi)
          x1t=x1soft(x1,x2,x,yi)
          x2t=x2soft(x1,x2,x,yi)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvh(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                     phil,cthl,x1,x2,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=3,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checkvh(xsum,dummy,iret)
      if(iret.eq.1)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,phil,cthl,str,
     #               tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5as_dec=abs(xint)
c
      return
      end


      function tot5az_dec(xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac)
      implicit none
      real * 8 tot5az_dec,xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac
c
      tot5az_dec=0
      write(*,*)'Zeta subtraction non implemented'
      stop
      return
      end


      function sig5bzw_und(xx)
      implicit none
      real * 8 sig5bzw_und,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5b_und
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1.d0
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(4)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5bzw_und = tot5b_und(s,x,y,cth1,cth2,xjac)
      return
      end


      function sig5bfw_und(xx)
      implicit none
      real * 8 sig5bfw_und,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(8)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5b_und,tt,xbwmass3
c
c xx(1) --> xmv2, xx(2) --> xmh2, xx(3) --> tau, xx(4)-->ycm, xx(5) --> x, 
c xx(6) --> y, xx(7) --> cth1, xx(8) --> cth2
c
      xjac = 1.d0
c
      tt=xx(1)
      if(tvsamp.ne.1)then
        write(*,*)'Error #1 in sig5bfw_und'
        stop
      endif
      xmv2=xbwmass3(tt,xm0v2,gav,bwvdelf,bwvfmmn)
      xjac=xjac*bwvdelf
      if(xmv2.lt.xmvlow2.or.xmv2.gt.xmvupp2)then
        write(*,*)'Error #2 in sig5bfw_und',xmv2
        stop
      endif
c
      tt=xx(2)
      if(thsamp.ne.1)then
        write(*,*)'Error #3 in sig5bfw_und'
        stop
      endif
      xmh2=xbwmass3(tt,xm0h2,gah,bwhdelf,bwhfmmn)
      xjac=xjac*bwhdelf
      if(xmh2.lt.xmhlow2.or.xmh2.gt.xmhupp2)then
        write(*,*)'Error #4 in sig5bfw_und',xmh2
        stop
      endif
c
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(5)**2
      xjac = xjac * xx(5) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(6)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(3)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(4)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(7),cth1,xjac,rox)
c
      th2 = xx(8) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5bfw_und = tot5b_und(s,x,y,cth1,cth2,xjac)
      return
      end


      function sig5bfw_dec(xx)
      implicit none
      real * 8 sig5bfw_dec,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(10)
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tvsamp,thsamp
      common/ctsamp/tvsamp,thsamp
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,rohlim,zzz,x,ttt,th,y,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xalfaem,rox,cth1,th2,cth2,tot5b_dec,tt,xbwmass3,bwfunc,
     #  cthl,phil
c
c xx(1) --> xmv2, xx(2) --> xmh2, xx(3) --> tau, xx(4)-->ycm, xx(5) --> x, 
c xx(6) --> y, xx(7) --> cth1, xx(8) --> cth2, xx(9) --> cthl, xx(10) --> phil
c
      xjac = 1.d0
c
      tt=xx(1)
      if(tvsamp.ne.1)then
        write(*,*)'Error #1 in sig5bfw_dec'
        stop
      endif
      xmv2=xbwmass3(tt,xm0v2,gav,bwvdelf,bwvfmmn)
      xjac=xjac*bwvdelf/bwfunc(xmv2,xm0v2,gav)
      if(xmv2.lt.xmvlow2.or.xmv2.gt.xmvupp2)then
        write(*,*)'Error #2 in sig5bfw_dec',xmv2
        stop
      endif
c
      tt=xx(2)
      if(thsamp.ne.1)then
        write(*,*)'Error #3 in sig5bfw_dec'
        stop
      endif
      xmh2=xbwmass3(tt,xm0h2,gah,bwhdelf,bwhfmmn)
      xjac=xjac*bwhdelf
      if(xmh2.lt.xmhlow2.or.xmh2.gt.xmhupp2)then
        write(*,*)'Error #4 in sig5bfw_dec',xmh2
        stop
      endif
c
      rohlim=(sqrt(xmv2)+sqrt(xmh2))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(5)**2
      xjac = xjac * xx(5) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(6)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
      y    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohlimx = rohlim/x
      taumax = 1/x
      ximax0 = rohlimx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(3)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(4)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
      xmi2=s*x
      if(ifxdaem.eq.0)then
        ze2=4*pi*xalfaem(xmi2)
        ze2v=4*pi*xalfaem(xmv2)
      endif
      xicut=xicutss
      if(xicut.gt.(1-rohlim))xicut=1-rohlim
c
      rox = 2*(xmv2+xmh2)/(s*x)-(xmv2-xmh2)**2/(s*x)**2
      call zzchvar(xx(7),cth1,xjac,rox)
c
      th2 = xx(8) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      cthl = 2*xx(9)-1
      xjac = 2*xjac
c
      phil = 2*pi*xx(10)
      xjac = 2*pi*xjac
c
      sig5bfw_dec = tot5b_dec(s,x,y,cth1,cth2,cthl,phil,xjac)
      return
      end


      function tot5b_und(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5b_und,tot5bs_und,tot5bz_und,s,x,yi,cth1,cth2,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5bs_und(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5bz_und(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5b_und:',isubttype
        stop
      endif
      tot5b_und=tmp
      return
      end


      function tot5b_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      implicit none
      real * 8 tot5b_dec,tot5bs_dec,tot5bz_dec,s,x,yi,cth1,cth2,
     #  cthl,phil,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5bs_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5bz_dec(s,x,yi,cth1,cth2,cthl,phil,xjac)
      else
        write(*,*)'Fatal error in tot5b_dec:',isubttype
        stop
      endif
      tot5b_dec=tmp
      return
      end


      function tot5bs_und(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bs_und,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      integer idec
      common/cidec/idec
      real * 8 sf(4,2:3,6)
      real * 8 xintsvc(1:4,2:3,3),xbornvc(1:4,2:3,3)
      real * 8 xcsvc(1:4,2:3,3),xsvvc(1:4,2:3,3)
      real * 8 xintsax(1:4,2:3,3),xbornax(1:4,2:3,3)
      real * 8 xcsax(1:4,2:3,3),xsvax(1:4,2:3,3)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 veccl(1:4,1:3),axlcl(1:4,1:3)
      real * 8 vecbrn(1:4,1:3),axlbrn(1:4,1:3)
      real * 8 vec2sv(1:4,1:3),axl2sv(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,ro,beta,rox,betax,xphsp_ns,
     #  xphsp_s,xphspb,x1,x2,tk,uk,q1q,q2q,dummy,xnorm,ffunval5,
     #  ffunction5,zg2_mc,zgmu2_vh,ytmp,zhwfct,zherw_spl,x1t,x2t,
     #  x1soft,x2soft,x1x2j,x1x2jac,zherw_smn,xint,xtmp,zg2_nlo,
     #  zgmu2_nvh,xsum,xlmude,xnormc,xnormb,xnormsv
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
      ro = 2*(xmv2+xmh2)/s-(xmv2-xmh2)**2/s**2
      beta = sqrt(1-ro)
      rox = 2*(xmv2+xmh2)/sx-(xmv2-xmh2)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4)
      xphspb = xjac * betax/(32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
      ffunval5 = ffunction5(x,yi)
      if(x1.lt.1.and.x2.lt.1.and.ffunval5.ne.0.d0)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,zero,zero,str,
     #             tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
        zg2_mc = zgmu2_vh()
        ipdfscale=2
        xnorm = ze2**2 * zg2_mc * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        ytmp=1.d0
        zhwfct=zherw_spl(x,yi)
        x1t=x1soft(x1,x2,x,yi)/zhwfct
        x2t=x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfvh(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                   zero,zero,x1,x2,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
        ytmp=-1.d0
        zhwfct=zherw_smn(x,yi)
        x1t=x1soft(x1,x2,x,yi)
        x2t=x2soft(x1,x2,x,yi)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfvh(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                   zero,zero,x1,x2,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yi.gt.1-delta) then
        ytmp = 1.d0
        x1t = x1soft(x1,x2,x,yi)/x
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invvh(xmv2,xmh2,s,x,ytmp,cth1,cth2,zero,zero,str,
     #               tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1-yi) ) 
          xlmude = log(s/xmuf2h1)+log(delta/2)
          xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /(8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal_und(s,x,ytmp,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call f2pr_und(s,q2q,x,x,ytmp,xlmude,jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      if(yi.lt.-1+delta) then
        ytmp = -1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)/x
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invvh(xmv2,xmh2,s,x,ytmp,cth1,cth2,zero,zero,str,
     #               tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1+yi) ) 
          xlmude = log(s/xmuf2h2)+log(delta/2)
          xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /(8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal_und(s,x,ytmp,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call f2pr_und(s,q1q,x,x,ytmp,xlmude,jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
c     Soft part of the counter-event:
c
      if(xii.lt.xicut) then
        xtmp = 1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)
          call invvh(xmv2,xmh2,sx,xtmp,yi,cth1,cth2,zero,zero,str,
     #               tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
          xnorm = - xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          xnormb = x1x2j * ze2**2 * xphspb /(2*xicut)
          xnormsv = x1x2j * ze2**2 * zg2_nlo * xphspb / 
     #             (8*pi2 * 2*xicut)
          do jproc=loproc,maproc
            call freal_und(sx,xtmp,yi,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call fborn_und(sx,q1q,jproc,vecbrn,axlbrn)
            call f2sv_und(sx,q1q,jproc,vec2sv,axl2sv)
            do j=1,4
              do i=3,3
                xintsvc(j,jproc,i)=xnorm*vecre(j,i)
                xintsax(j,jproc,i)=xnorm*axlre(j,i)
                xbornvc(j,jproc,i)=xnormb*vecbrn(j,i)
                xbornax(j,jproc,i)=xnormb*axlbrn(j,i)
                xsvvc(j,jproc,i)=xnormsv*vec2sv(j,i)
                xsvax(j,jproc,i)=xnormsv*axl2sv(j,i)
                xcsvc(j,jproc,i)=0.d0
                xcsax(j,jproc,i)=0.d0
              enddo
            enddo
          enddo
c
          if(yi.gt.1-delta) then
             ytmp = 1.d0
             call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,zero,zero,
     #                  str,tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
             xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1-yi) ) 
             xlmude = log(sx/xmuf2h1)+log(delta/2)
             xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal_und(sx,xtmp,ytmp,tk,uk,q1q,q2q,jproc,
     #                        vecre,axlre)
               call f2pr_und(sx,q2q,x,xtmp,ytmp,xlmude,jproc,
     #                       veccl,axlcl)
               do j=1,4
                 do i=3,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          if(yi.lt.-1+delta) then
             ytmp = -1.d0
             call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,zero,zero,
     #                  str,tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
             xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1+yi) ) 
             xlmude = log(sx/xmuf2h2)+log(delta/2)
             xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal_und(sx,xtmp,ytmp,tk,uk,q1q,q2q,jproc,
     #                        vecre,axlre)
               call f2pr_und(sx,q1q,x,xtmp,ytmp,xlmude,jproc,
     #                       veccl,axlcl)
               do j=1,4
                 do i=3,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xcsvc(j,jproc,i)+
     #                              xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xcsax(j,jproc,i)+
     #                              xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          do jproc=loproc,maproc
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)
     #              +sf(j,jproc,itype)*(
     #  xveccoef(itype,i)*( xintsvc(j,jproc,i)+xbornvc(j,jproc,i)+
     #                      xsvvc(j,jproc,i)+xcsvc(j,jproc,i) )+
     #  xaxlcoef(itype,i)*( xintsax(j,jproc,i)+xbornax(j,jproc,i)+
     #                      xsvax(j,jproc,i)+xcsax(j,jproc,i) ) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      call checkvh(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,zero,zero,str,
     #             tk,uk,q1q,q2q,dummy,dummy,dummy,idec)
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        ycm = 0.5d0*log(x1t/x2t)
        tau=x*tau
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5bs_und=abs(xint)
c
      return
      end


      function tot5bz_und(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bz_und,xs,xx,xyi,xcth1,xcth2,xjac
c
      tot5bz_und=0
      write(*,*)'Zeta subtraction non implemented'
      stop
      return
      end


      function tot5bs_dec(xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac)
      implicit none
      real * 8 tot5bs_dec,xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'vhgcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      integer idec
      common/cidec/idec
      real * 8 sf(4,2:3,6)
      real * 8 xintsvc(1:4,2:3,3),xbornvc(1:4,2:3,3)
      real * 8 xcsvc(1:4,2:3,3),xsvvc(1:4,2:3,3)
      real * 8 xintsax(1:4,2:3,3),xbornax(1:4,2:3,3)
      real * 8 xcsax(1:4,2:3,3),xsvax(1:4,2:3,3)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 veccl(1:4,1:3),axlcl(1:4,1:3)
      real * 8 vecbrn(1:4,1:3),axlbrn(1:4,1:3)
      real * 8 vec2sv(1:4,1:3),axl2sv(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,ro,beta,rox,betax,xphsp_ns,
     #  xphsp_s,xphspb,x1,x2,tk,uk,q1q,q2q,dummy,xnorm,ffunval5,
     #  ffunction5,zg2_mc,zgmu2_vh,ytmp,zhwfct,zherw_spl,x1t,x2t,
     #  x1soft,x2soft,x1x2j,x1x2jac,zherw_smn,xint,xtmp,zg2_nlo,
     #  zgmu2_nvh,xsum,xlmude,xnormc,xnormb,xnormsv,cthl,phil,
     #  v1a,v1b,v1c
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      phil = xphil
      cthl = xcthl
      sx = x*s
      xii = 1-x
      ro = 2*(xmv2+xmh2)/s-(xmv2-xmh2)**2/s**2
      beta = sqrt(1-ro)
      rox = 2*(xmv2+xmh2)/sx-(xmv2-xmh2)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4) * 1/(2*pi*32*pi2)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4) * 1/(2*pi*32*pi2)
      xphspb = xjac * betax/(32*pi2) * 1/(2*pi*32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
      ffunval5 = ffunction5(x,yi)
      if(x1.lt.1.and.x2.lt.1.and.ffunval5.ne.0.d0)then
        call invvh(xmv2,xmh2,s,x,yi,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        zg2_mc = zgmu2_vh()
        ipdfscale=2
        xnorm = ze2**2 * ze2v * zg2_mc * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        ytmp=1.d0
        zhwfct=zherw_spl(x,yi)
        x1t=x1soft(x1,x2,x,yi)/zhwfct
        x2t=x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfvh(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                   phil,cthl,x1,x2,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
        ytmp=-1.d0
        zhwfct=zherw_smn(x,yi)
        x1t=x1soft(x1,x2,x,yi)
        x2t=x2soft(x1,x2,x,yi)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfvh(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsuvh(jproc,idec,xmv2,xmh2,s,x,yi,cth1,cth2,
     #                   phil,cthl,x1,x2,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yi.gt.1-delta) then
        ytmp = 1.d0
        x1t = x1soft(x1,x2,x,yi)/x
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invvh(xmv2,xmh2,s,x,ytmp,cth1,cth2,phil,cthl,str,
     #               tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * ze2v * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1-yi) ) 
          xlmude = log(s/xmuf2h1)+log(delta/2)
          xnormc = x1x2j * ze2**2 * ze2v * zg2_nlo * xphspb /
     #             (8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal_dec(s,x,ytmp,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                     jproc,vecre,axlre)
            call f2pr_dec(s,q2q,v1a,v1b,v1c,x,x,ytmp,xlmude,
     #                    jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      if(yi.lt.-1+delta) then
        ytmp = -1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)/x
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invvh(xmv2,xmh2,s,x,ytmp,cth1,cth2,phil,cthl,str,
     #               tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * ze2v * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1+yi) ) 
          xlmude = log(s/xmuf2h2)+log(delta/2)
          xnormc = x1x2j * ze2**2 * ze2v * zg2_nlo * xphspb /
     #             (8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal_dec(s,x,ytmp,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                     jproc,vecre,axlre)
            call f2pr_dec(s,q1q,v1a,v1b,v1c,x,x,ytmp,xlmude,
     #                    jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
c     Soft part of the counter-event:
c
      if(xii.lt.xicut) then
        xtmp = 1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)
          call invvh(xmv2,xmh2,sx,xtmp,yi,cth1,cth2,phil,cthl,str,
     #               tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
          zg2_nlo = zgmu2_nvh()
          call strfvh(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * ze2v * zg2_nlo * xphsp_s
          xnorm = - xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          xnormb = x1x2j * ze2**2 * ze2v * xphspb /(2*xicut)
          xnormsv = x1x2j * ze2**2 * ze2v * zg2_nlo * xphspb / 
     #             (8*pi2 * 2*xicut)
          do jproc=loproc,maproc
            call freal_dec(sx,xtmp,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                     jproc,vecre,axlre)
            call fborn_dec(sx,q1q,v1a,v1b,jproc,vecbrn,axlbrn)
            call f2sv_dec(sx,q1q,v1a,v1b,v1c,jproc,vec2sv,axl2sv)
            do j=1,4
              do i=3,3
                xintsvc(j,jproc,i)=xnorm*vecre(j,i)
                xintsax(j,jproc,i)=xnorm*axlre(j,i)
                xbornvc(j,jproc,i)=xnormb*vecbrn(j,i)
                xbornax(j,jproc,i)=xnormb*axlbrn(j,i)
                xsvvc(j,jproc,i)=xnormsv*vec2sv(j,i)
                xsvax(j,jproc,i)=xnormsv*axl2sv(j,i)
                xcsvc(j,jproc,i)=0.d0
                xcsax(j,jproc,i)=0.d0
              enddo
            enddo
          enddo
c
          if(yi.gt.1-delta) then
             ytmp = 1.d0
             call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,phil,cthl,
     #                  str,tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
             xnorm = x1x2j * ze2**2 * ze2v * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1-yi) ) 
             xlmude = log(sx/xmuf2h1)+log(delta/2)
             xnormc = x1x2j * ze2**2 * ze2v * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal_dec(sx,xtmp,ytmp,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                        jproc,vecre,axlre)
               call f2pr_dec(sx,q2q,v1a,v1b,v1c,x,xtmp,ytmp,xlmude,
     #                       jproc,veccl,axlcl)
               do j=1,4
                 do i=3,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          if(yi.lt.-1+delta) then
             ytmp = -1.d0
             call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,phil,cthl,
     #                  str,tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
             xnorm = x1x2j * ze2**2 * ze2v * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1+yi) ) 
             xlmude = log(sx/xmuf2h2)+log(delta/2)
             xnormc = x1x2j * ze2**2 * ze2v * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal_dec(sx,xtmp,ytmp,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                        jproc,vecre,axlre)
               call f2pr_dec(sx,q1q,v1a,v1b,v1c,x,xtmp,ytmp,xlmude,
     #                       jproc,veccl,axlcl)
               do j=1,4
                 do i=3,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xcsvc(j,jproc,i)+
     #                              xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xcsax(j,jproc,i)+
     #                              xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          do jproc=loproc,maproc
            do j=1,4
              do itype=1,6
                do i=3,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)
     #              +sf(j,jproc,itype)*(
     #  xveccoef(itype,i)*( xintsvc(j,jproc,i)+xbornvc(j,jproc,i)+
     #                      xsvvc(j,jproc,i)+xcsvc(j,jproc,i) )+
     #  xaxlcoef(itype,i)*( xintsax(j,jproc,i)+xbornax(j,jproc,i)+
     #                      xsvax(j,jproc,i)+xcsax(j,jproc,i) ) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      call checkvh(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invvh(xmv2,xmh2,sx,xtmp,ytmp,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        ycm = 0.5d0*log(x1t/x2t)
        tau=x*tau
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5bs_dec=abs(xint)
c
      return
      end


      function tot5bz_dec(xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac)
      implicit none
      real * 8 tot5bz_dec,xs,xx,xyi,xcth1,xcth2,xcthl,xphil,xjac
c
      tot5bz_dec=0
      write(*,*)'Zeta subtraction non implemented'
      stop
      return
      end
c
c
c End of cross section routines
c
c
c
c
c Begin of utility functions for zeta subtraction
c
c
C      function svn(ro)
C      implicit none
C      real*8 svn,ro,tmp,be4,ybar,etacut
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(ro.lt.1.d0-sqrt(etacut))then
C        be4=(1-ro)**2
C        ybar=sqrt(1-etacut/be4)
C        tmp=-(1-ro)*ybar+sqrt(etacut)*asin(ybar)
C      endif
C      svn=tmp
C      return
C      end


C      function f1fun(ro)
C      implicit real * 8 (a-z)
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(ro.lt.1.d0-sqrt(etacut))then
C        be4=(1-ro)**2
C        ybar=sqrt(1-etacut/be4)
C        tmp=log((1+ybar)/(1-ybar))*( log(etacut/be4)
C     #        -log(1-ybar**2)/2.d0-log(2.d0) )
C     #     +ddilog((1+ybar)/2.d0)-ddilog((1-ybar)/2.d0) 
C        tmp=tmp/4.d0
C      endif
C      f1fun=tmp
C      return
C      end


C      function bdelta(x)
C      implicit none
C      real*8 bdelta,x,tmp,etacut
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(x.lt.1.d0-dsqrt(etacut))tmp=sqrt(1-etacut/(1-x)**2)
C      bdelta=tmp
C      return
C      end
c
c
c End of utility functions for zeta subtraction
c
c
c
c
c Begin of event-generation routines
c
c
      subroutine sprfvh()
c This routine is called by run_spring; the entry is dummy, all the 
c parameters must be passed through common blocks
      implicit none
      integer iunit
      parameter (iunit=22)
      real*8 xone
      parameter (xone=1.d0)
      real*8 ycm,tau
      common/x1x2/ycm,tau
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer iret
      real*8 ycm0
c
      call xovh(iret)
      if(iret.eq.1)then
        if(i0.lt.1.or.i0.gt.4)then
          write(*,*)'Fatal error in sprfvh'
          stop
        endif
        ycm0=ycm
        call getx1x2(tau,ycm0)
        call getmvh(tau,ycm0)
        call store_evenvh(iunit,xone)
      endif
      return
      end


C      subroutine getx1x2(tau,ycm)
C      implicit none
C      real*8 tau,ycm,x1,x2,stau,ey
C      common/cx1x2/x1,x2
Cc
C      stau=sqrt(tau)
C      ey=exp(ycm)
C      x1=stau*ey
C      x2=stau/ey
C      return
C      end


      subroutine getmvh(xtau,xycm)
      implicit none
      real*8 xtau,xycm
      include 'vhgcblks.h'
      real*8 pi
      parameter (pi=3.14159265358979312D0)
      integer i,j,imax,itype
      real*8 tau,ycm,theta,cth,sth,fk88random,sqsh,ycmnew
      real*8 x1,x2
      common/cx1x2/x1,x2
      real*8 xmom_cm(7,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_lb(7,4)
      common/cxmomlb/xmom_lb
      real*8 xmom_prime(7,4)
      common/cxmomprime/xmom_prime
      integer ionshell
      common/cionshell/ionshell
      integer ichkmom
      common/cichkmom/ichkmom
      integer idec
      common/cidec/idec
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      itype=idec+1
      if(itype.eq.1)then
        imax=7
      elseif(itype.eq.2)then
        imax=5
      else
        write(6,*)'Wrong option in getmvh'
        stop
      endif
      tau=xtau
      ycm=xycm
      call getx1x2(tau,ycm)
c perform a random rotation in the transverse plane
      theta=2*pi*fk88random(ifk88seed)
      cth=cos(theta)
      sth=sin(theta)
      do i=3,imax
        call transrot(cth,sth,xmom_cm(i,1),xmom_cm(i,2))
      enddo
      if(ichkmom.eq.0)call checkmvh(xmom_cm,sh,0.d0,3,itype)
c put partons on Herwig mass shell
      if(ionshell.eq.0)then
c keep the parton massless
        sqsh=sqrt(sh)
        xmom_lb(1,1)=0.d0
        xmom_lb(1,2)=0.d0
        xmom_lb(1,3)=x1*sqsh/2.d0
        xmom_lb(1,4)=x1*sqsh/2.d0
        xmom_lb(2,1)=0.d0
        xmom_lb(2,2)=0.d0
        xmom_lb(2,3)=-x2*sqsh/2.d0
        xmom_lb(2,4)=x2*sqsh/2.d0
        do i=3,imax
          call boost(-ycm,
     #         xmom_cm(i,1),xmom_cm(i,2),
     #         xmom_cm(i,3),xmom_cm(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      else
c put the partons on Herwig mass shell
        call put_on_shevh(ycm,ycmnew)
        do i=1,imax
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(idec.eq.1)then
        do i=6,7
          do j=1,4
            xmom_lb(i,j)=0.d0
          enddo
        enddo
      endif
      if(ichkmom.eq.0)call checkmvh(xmom_lb,sh,-ycm,2,itype)
      call momnewformvh()
      return
      end


      subroutine momnewformvh()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(7,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:7)
      common/procmass/xmss
      integer i
c
      do i=1,7
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


C      subroutine boost(y,a1,a2,a3,a4,b1,b2,b3,b4)
C      implicit none
C      real*8 y,a1,a2,a3,a4,b1,b2,b3,b4
Cc
C      b1=a1
C      b2=a2
C      b3=a3*cosh(y)-a4*sinh(y)
C      b4=a4*cosh(y)-a3*sinh(y)
C      return
C      end


C      subroutine transrot(cth,sth,xpt1,xpt2)
C      implicit none
C      real*8 cth,sth,xpt1,xpt2,pt1,pt2
Cc
C      pt1=xpt1
C      pt2=xpt2
C      xpt1=pt1*cth+pt2*sth
C      xpt2=-pt1*sth+pt2*cth
C      return
C      end


      subroutine put_on_shevh(ycm,ycmnew)
      implicit none
      include 'vhgcblks.h'
      integer i2b,i,j
      real*8 xmss(1:7),xtmp(1:4),xk1tmp(1:4),ytmp1(1:4),ytmp2(1:4),
     #  xavg3(1:3)
      real*8 ycm,ycmnew,pi,one,delta_thrs,shat,xkp2prime_norm2,
     #  xkp2prime_norm,xkprime_0,xsign,xnorm_3,delta,gamma,xmprime,
     #  xk1prime_norm,fakemass,xk1tmp_norm,xkprime_norm,zero,
     #  xlepnorm,tmplmass,xavgnorm
      parameter (pi=3.14159265358979312D0)
      parameter (zero=0.d0)
      parameter (one=1.d0)
      parameter (delta_thrs=0.5d-3)
      common/procmass/xmss
      real*8 xmass(-5:21)
      common/parmass/xmass
c Lepton masses
      real * 8 xlep1mass,xlep2mass
      common/clepmass/xlep1mass,xlep2mass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is V, 5 is H, 6 and 7 are the outgoing
c fermion and antifermion respectively. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+5+6+7)=0 
      real*8 xmom_cm(7,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(7,4)
      common/cxmomprime/xmom_prime
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
c here, ionshell=1 or ionshell=2
      integer ionshell
      common/cionshell/ionshell
      integer ilepmass
      common/cilepmass/ilepmass
      integer ichkmom
      common/cichkmom/ichkmom
      integer idec
      common/cidec/idec
c 
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      xmss(4) = sqrt(xmv2)
      xmss(5) = sqrt(xmh2)
      if(idec.eq.0)then
        if(ilepmass.eq.0)then
          xmss(6) = 0.d0
          xmss(7) = 0.d0
        elseif(ilepmass.eq.2)then
          xmss(6) = xlep1mass
          xmss(7) = xlep2mass
        else
          write(*,*)'Error in put_on_shevh: unknown ilepmass',ilepmass
          stop
        endif
      elseif(idec.eq.1)then
        xmss(6) = -1.d10
        xmss(7) = -1.d10
      else
        write(6,*) 'Error in put_on_shevh: idec=',idec
        stop
      endif
c i2b=0 --> 3-body kinematics; i2b=1 --> 2-body kinematics
      i2b = 0
      if(xmom_cm(3,4).lt.1.d-14)i2b=1
      if(ionshell.eq.1)then
c don't change the 3-momenta of partons 1,2 and 3, if possible
        do i=1,3
          do j=1,3
            xmom_prime(i,j)=xmom_cm(i,j)
          enddo
        enddo
        shat=(xmom_cm(1,4)+xmom_cm(2,4))**2
      elseif(ionshell.eq.2)then
c don't change the 3-momentum of parton 3, and shat, if possible
        do j=1,3
          xmom_prime(3,j)=xmom_cm(3,j)
        enddo
        do i=1,2
          do j=1,2
            xmom_prime(i,j)=xmom_cm(i,j)
          enddo
        enddo
        shat=(xmom_cm(1,4)+xmom_cm(2,4))**2
        call getxmvh(shat,ycm,
     #               xmom_cm(1,3),xmss(1),
     #               xmom_cm(2,3),xmss(2),
     #               xmom_prime(1,3),xmom_prime(2,3))
      else
        write(*,*)'Error in put_on_shevh: unknown ionshell'
        stop
      endif
      xkprime_0=0.d0
      do i=1,3
        xsign=1.d0
        if(i.eq.3)xsign=-1.d0
        if(i.eq.3.and.i2b.eq.1)then
          xmom_prime(i,4)=0.d0
        else
          call getenergy(xmom_prime(i,1),xmom_prime(i,2),
     #                   xmom_prime(i,3),xmss(i),xmom_prime(i,4))
        endif
        xkprime_0=xkprime_0+xsign*xmom_prime(i,4)
      enddo
c compute the modulus of the 3-momentum of the pair, which is equal
c to that of parton 3 in the CM frame. The energy doesn't play any role
      call fillvec(xmom_cm(3,1),xmom_cm(3,2),
     #             xmom_cm(3,3),xmom_cm(3,4),xtmp)
      xkprime_norm=xnorm_3(xtmp)
c delta is the would-be invariant mass of the pair, minus the sum
c of the masses of the V and H
      delta=sqrt(xkprime_0**2-xkprime_norm**2)-xmss(4)-xmss(5)
      if(delta.lt.delta_thrs)then
c parton 3-momenta cannot be kept fixed: the total available energy
c is not sufficient; modify 3-momenta of the incoming partons
        gamma=sqrt( (xmss(4)+xmss(5)+delta_thrs)**2+xkprime_norm**2 )+
     #        xmom_prime(3,4)
        if(gamma.lt.(xmss(1)+xmss(2)))then
          write(6,*)'Error #0 in put_on_shevh'
          write(6,*)gamma,xmom_prime(3,4)
          stop
        endif
        xkp2prime_norm2=( gamma**2-2*(xmss(1)**2+xmss(2)**2)+
     #                    (xmss(1)**2-xmss(2)**2)**2/gamma**2 )/4.d0
        xkp2prime_norm=sqrt(xkp2prime_norm2)
        xmom_prime(1,3)=sign(1.d0,xmom_cm(1,3))*xkp2prime_norm
        xmom_prime(1,4)=gamma/2.d0*(1+(xmss(1)**2-xmss(2)**2)/gamma**2)
        xmom_prime(2,3)=sign(1.d0,xmom_cm(2,3))*xkp2prime_norm
        xmom_prime(2,4)=gamma/2.d0*(1-(xmss(1)**2-xmss(2)**2)/gamma**2)
        xkprime_0=xmom_prime(1,4)+xmom_prime(2,4)-xmom_prime(3,4)
        shat=(xmom_prime(1,4)+xmom_prime(2,4))**2 -
     #       (xmom_prime(1,3)+xmom_prime(2,3))**2
      endif
c now the parton 3-momenta have been defined in such a way
c that the momenta of the V and H can be transformed.
c xtmp is the 4-momentum of the pair. CM frame stays the same,
c so does the boost
      ycmnew=ycm
      xtmp(1)=-xtmp(1)
      xtmp(2)=-xtmp(2)
      xtmp(3)=-xtmp(3)
      xtmp(4)=xkprime_0
      xmprime=sqrt(xkprime_0**2-xkprime_norm**2)
      xk1prime_norm=xmprime**2-2*(xmss(4)**2+xmss(5)**2)+
     #              (xmss(4)**2-xmss(5)**2)**2/xmprime**2
      xk1prime_norm=sqrt(xk1prime_norm)/2.d0
      do j=1,3
        xavg3(j)=0.d0
      enddo
      do i=4,5
        xsign=1.d0
        if(i.eq.5)xsign=-1.d0
        call fillvec(xmom_cm(i,1),xmom_cm(i,2),
     #               xmom_cm(i,3),xmom_cm(i,4),ytmp1)
        call xhwulof(xtmp,xmprime,
     #               ytmp1,xmss(i),
     #               xk1tmp,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Error #1 in put_on_shevh'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        xk1tmp_norm=xnorm_3(xk1tmp)
c xavg is the direction along which the V and H momenta are placed
c in the new VH rest frame. It is arbitrarily defined by averaging 
c (hence the 1/2 in the definition) the directions of the original 
c V and H momenta. It may not have modulus 1, so normalize it
        do j=1,3
          xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
        enddo
      enddo
      xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
      do j=1,3
        xavg3(j)=xavg3(j)/xavgnorm
      enddo
      do i=4,5
        xsign=1.d0
        if(i.eq.5)xsign=-1.d0
        do j=1,3
          xk1tmp(j)=xsign*xk1prime_norm*xavg3(j)
        enddo
        xk1tmp(4)=xmprime/2.d0*
     #            (1+xsign*(xmss(4)**2-xmss(5)**2)/xmprime**2)
        call xhwulob(xtmp,xmprime,
     #               xk1tmp,xmss(i),
     #               ytmp2,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Error #2 in put_on_shevh'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                    xmom_prime(i,3),xmom_prime(i,4))
      enddo
      if(idec.eq.0)then
c repeat what done for the VH, boosting the lepton momenta in
c the new V rest frame
        call fillvec(xmom_prime(4,1),xmom_prime(4,2),
     #               xmom_prime(4,3),xmom_prime(4,4),xtmp)
        xlepnorm=xmss(4)**2-2*(xlep1mass**2+xlep2mass**2)+
     #           (xlep1mass**2-xlep2mass**2)**2/xmss(4)**2
        xlepnorm=sqrt(xlepnorm)/2.d0
        do j=1,3
          xavg3(j)=0.d0
        enddo
        do i=6,7
          if(i.eq.6)then
            xsign=1.d0
            tmplmass=xlep1mass
          else
            xsign=-1.d0
            tmplmass=xlep2mass
          endif
          call fillvec(xmom_cm(i,1),xmom_cm(i,2),
     #                 xmom_cm(i,3),xmom_cm(i,4),ytmp1)
          call xhwulof(xtmp,xmss(4),
     #                 ytmp1,tmplmass,
     #                 xk1tmp,fakemass)
          xk1tmp_norm=xnorm_3(xk1tmp)
          do j=1,3
            xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
          enddo
        enddo
        xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
        do j=1,3
          xavg3(j)=xavg3(j)/xavgnorm
        enddo
        do i=6,7
          if(i.eq.6)then
            xsign=1.d0
            tmplmass=xlep1mass
          else
            xsign=-1.d0
            tmplmass=xlep2mass
          endif
          do j=1,3
            xk1tmp(j)=xsign*xlepnorm*xavg3(j)
          enddo
          xk1tmp(4)=xmss(4)/2.d0*
     #             (1+xsign*(xlep1mass**2-xlep2mass**2)/xmss(4)**2)
          call xhwulob(xtmp,xmss(4),
     #                 xk1tmp,tmplmass,
     #                 ytmp2,fakemass)
          call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                      xmom_prime(i,3),xmom_prime(i,4))
        enddo
      else
        do i=6,7
          do j=1,4
            xmom_prime(i,j)=0.d0
          enddo
        enddo
      endif
      if(ichkmom.eq.0)then
        call checkmvh(xmom_prime,shat,0.d0,4,2)
        if(idec.eq.0)then
          call checkdec(xmom_prime,4,6,7)
          call checkmvh(xmom_prime,shat,0.d0,4,1)
        endif
        if(xmass(1).eq.0.and.xmass(2).eq.0.and.xmass(3).eq.0.and.
     #     xmass(4).eq.0.and.xmass(5).eq.0.and.xmass(21).eq.0.and.
     #     xlep1mass.eq.0.and.xlep2mass.eq.0)then
          call checkonsh(1)
        else
          call checkonsh(2)
        endif
      endif
      return
      end


      subroutine getxmvh(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
c After putting the momenta on shell, the two incoming partons may
c travel in the same direction. This routine prevents this to happen,
c redefining Herwig masses if necessary
      implicit none
      real*8 shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23
      real*8 tiny,fact,sqs,xm1s,xm2s,xkp2prime_norm2,xkp2prime_norm,
     #  ytmp,e1,e2,p13p,p23p,s1p,s2p,xif,sol
      integer iflag,idone,ileg
      parameter (fact=0.98d0)
      parameter (tiny=1.d-6)
c
      sqs=sqrt(shat)
      xm1s=xm1
      xm2s=xm2
      ytmp=-ycm
      idone=0
 100  continue
      xkp2prime_norm2=( shat-2*(xm1**2+xm2**2)+
     #                  (xm1**2-xm2**2)**2/shat )/4.d0
      xkp2prime_norm=sqrt(xkp2prime_norm2)
      if(sign(1.d0,p13cm).ne.1.d0.or.sign(1.d0,p23cm).ne.-1.d0)then
        write(*,*)'Error # 0 in getxmvh'
        stop
      endif
      p13=xkp2prime_norm
      p23=-xkp2prime_norm
      e1=sqrt(shat)/2.d0*(1+(xm1**2-xm2**2)/shat)
      e2=sqrt(shat)/2.d0*(1-(xm1**2-xm2**2)/shat)
      p13p=p13*cosh(ytmp)-e1*sinh(ytmp)
      p23p=p23*cosh(ytmp)-e2*sinh(ytmp)
      s1p=sign(1.d0,p13p)
      s2p=sign(1.d0,p23p)
      iflag=0
      if(s1p.eq.1.d0 .and. s2p.eq.-1.d0)then
        iflag=1
      elseif(s1p.eq.-1.d0 .and. s2p.eq.-1.d0)then
        if(ytmp.lt.0.d0)then
          write(*,*)'Wrong y sign, # 1'
          stop
        endif
        ileg=1
        xif=xm2**2/shat
      elseif(s1p.eq.1.d0 .and. s2p.eq.1.d0)then
        if(ytmp.gt.0.d0)then
          write(*,*)'Wrong y sign, # 2'
          stop
        endif
        ileg=2
        xif=xm1**2/shat
      else
        write(*,*)'Error # 1 in getxmvh'
        stop
      endif
      if(iflag.eq.0)then
        sol=xif+cosh(2*ytmp)-
     #      sqrt(2.d0)*cosh(ytmp)*sqrt(cosh(2*ytmp)-1+2*xif)
        if(sol.le.0.d0.or.idone.eq.1)then
c The procedure failed; pass the massless event to Herwig, and let Herwig
c deal with it
          xm1=0.d0
          xm2=0.d0
          p13=sqs/2.d0
          p23=-sqs/2.d0
          return
        endif
        if(ileg.eq.1)then
          xm1=fact*sqrt(sol*shat)
          if(xm1.gt.xm1s)then
            write(*,*)'Mass # 1 too large in getxmvh'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmvh'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmvh'
          stop
        endif
        idone=1
        goto 100
      endif
      return
      end


C      subroutine fillvec(p1,p2,p3,p4,ytmp)
C      implicit none
C      real*8 p1,p2,p3,p4,ytmp(1:4)
Cc
C      ytmp(1)=p1
C      ytmp(2)=p2
C      ytmp(3)=p3
C      ytmp(4)=p4
C      return
C      end


C      subroutine getvec(ytmp,p1,p2,p3,p4)
C      implicit none
C      real*8 ytmp(1:4),p1,p2,p3,p4
Cc
C      p1=ytmp(1)
C      p2=ytmp(2)
C      p3=ytmp(3)
C      p4=ytmp(4)
C      return
C      end

Cc-----------------------------------------------------------------------
C      subroutine xhwulob(ps,ps5,pi,pi5,pf,pf5)
Cc     transforms pi (given in rest frame of ps) into pf (in lab)
Cc     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
Cc-----------------------------------------------------------------------
C      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
C      if (ps(4).eq.ps5) then
C        pf(1)= pi(1)
C        pf(2)= pi(2)
C        pf(3)= pi(3)
C        pf(4)= pi(4)
C      else
C        pf4  = (pi(1)*ps(1)+pi(2)*ps(2)
C     &         +pi(3)*ps(3)+pi(4)*ps(4))/ps5
C        fn   = (pf4+pi(4)) / (ps(4)+ps5)
C        pf(1)= pi(1) + fn*ps(1)
C        pf(2)= pi(2) + fn*ps(2)
C        pf(3)= pi(3) + fn*ps(3)
C        pf(4)= pf4
C      end if
C      pf5= pi5
C      end

Cc-----------------------------------------------------------------------
C      subroutine xhwulof(ps,ps5,pi,pi5,pf,pf5)
Cc     transforms pi (given in lab) into pf (in rest frame of ps)
Cc     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
Cc-----------------------------------------------------------------------
C      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
C      if (ps(4).eq.ps5) then
C        pf(1)= pi(1)
C        pf(2)= pi(2)
C        pf(3)= pi(3)
C        pf(4)= pi(4)
C      else
C        pf4  = (pi(4)*ps(4)-pi(3)*ps(3)
C     &         -pi(2)*ps(2)-pi(1)*ps(1))/ps5
C        fn   = (pf4+pi(4)) / (ps(4)+ps5)
C        pf(1)= pi(1) - fn*ps(1)
C        pf(2)= pi(2) - fn*ps(2)
C        pf(3)= pi(3) - fn*ps(3)
C        pf(4)= pf4
C      end if
C      pf5= pi5
C      end


C      subroutine getenergy(p1,p2,p3,xm,en)
C      implicit none
C      real*8 p1,p2,p3,xm,en
Cc
C      en=sqrt(p1**2+p2**2+p3**2+xm**2)
C      return
C      end


C      function dotprod(p1,p2,p3,p4,q1,q2,q3,q4)
C      implicit none
C      real*8 dotprod,p1,p2,p3,p4,q1,q2,q3,q4
Cc
C      dotprod=p4*q4-p1*q1-p2*q2-p3*q3
C      return
C      end


C      function xnorm_3(p)
Cc Evaluates the norm of the spatial component of a four-momentum
Cc The result is positive by definition, regardless of the 4-metric
C      implicit none
C      real*8 xnorm_3,p(1:4),tmp
Cc
C      tmp=p(1)*p(1)+p(2)*p(2)+p(3)*p(3)
C      xnorm_3=sqrt(tmp)
C      return
C      end


C      subroutine vecsum(p,pfact,q,qfact,r)
Cc Weighted sum of the four-vectors p and q. The result is r
C      implicit none
C      integer i
C      real*8 p(1:4),q(1:4),r(1:4),pfact,qfact
Cc
C      do i=1,4
C        r(i)=pfact*p(i)+qfact*q(i)
C      enddo
C      return
C      end


      subroutine checkonsh(itype)
c Checks that put_on_shevh is harmless if masses are zero (itype=1),
c or computes (itype=2) the average of the shifts due to the masses
      real*8 tiny
      parameter (tiny=1.d-4)
      integer itype
      real*8 xmom_cm(7,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_prime(7,4)
      common/cxmomprime/xmom_prime
      real*8 xmomshifts(4)
      common/cshifts/xmomshifts
      integer i,j,imax,iflag
      integer idec
      common/cidec/idec
c
      if(itype.ne.1.and.itype.ne.2)then
        write(*,*)'Unknown option in checkonsh'
        stop
      endif
      iflag=1
      imax=5
      if(idec.eq.0)imax=7
      do i=1,imax
        do j=1,4
          if(abs(xmom_cm(i,j)).lt.1.d0)then
            xtmp=xmom_cm(i,j)-xmom_prime(i,j)
          else
            xtmp=(xmom_cm(i,j)-xmom_prime(i,j))/xmom_cm(i,j)
          endif
          if(abs(xtmp).gt.tiny)iflag=0
          xmomshifts(j)=xmomshifts(j)+abs(xtmp)
        enddo
      enddo
      if(iflag.eq.0.and.itype.eq.1)then
        write(*,*)'Error in checkonsh'
        write(*,*)'  '
        write(*,*)'xmom_cm:'
        do i=1,imax
          write(*,'(4(d14.8,1x))') (xmom_cm(i,j),j=1,4)
        enddo
        write(*,*)'  '
        write(*,*)'xmom_prime:'
        do i=1,imax
          write(*,'(4(d14.8,1x))') (xmom_prime(i,j),j=1,4)
        enddo
        stop
      endif
      return
      end


      subroutine xovh(iret)
c This routine is called by sprfvh; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkvh), and no operation is performed
      implicit none
      integer iret,iretvv,iretvvs,iproc,iproclo,iprocma,i,itype,
     #  iwh,iflag,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o
      real*8 wwx(4,2:3,6),xsum,xsumabs,xsumvvs,xsumabsvvs,xstsign,
     #  xg,wh,rmax,fk88random
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      integer ifuntype
      common/cifuntype/ifuntype
      real*8 vv(4,2:3,6)
      common/cvv/vv
      real*8 vvs(4,2:3,6)
      common/cvvs/vvs
      integer iwrong,iwrong1
      common/ciwrong/iwrong,iwrong1
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer ivbhpro(4,2:3,6)
      common/civbhpro/ivbhpro
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cidpart/idp1,idp2,idp3
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer nl
      common/nl/nl
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      i0=0
      jproc0=0
      itype0=0
      iret=0
      call checkvh(xsum,xsumabs,iretvv)
      call checkvhs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xovh:',iretvv,iretvvs
        stop
      endif
      if(iretvv.eq.1)then
        iret=iretvv
        if(ifuntype.eq.1)then
          iproclo=loproc
          iprocma=maproc
        elseif(ifuntype.eq.2)then
          iproclo=loproc
          iprocma=maproc
        else
          write(*,*)'Fatal error in xovh: ifuntype=',ifuntype
          stop
        endif
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,6
                wwx(i,iproc,itype)=vvs(i,iproc,itype)
              enddo
            enddo
          enddo
        else
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,6
                wwx(i,iproc,itype)=vv(i,iproc,itype)
              enddo
            enddo
          enddo
        endif
        xstsign=sign(1.d0,xsum)
        xg=fk88random(ifk88seed)
        wh=0.d0
        iwh=0
        iflag=0
        rmax=0.d0
        do iproc=iproclo,iprocma
          do i=1,4
            do itype=1,6
              if(iwh.eq.0)then
                wh=wh+abs(wwx(i,iproc,itype))/xsumabs
                if(wh.gt.xg)then
                  i0=i
                  jproc0=iproc
                  itype0=itype
                  iwh=1
                endif
              endif
              if(wwx(i,iproc,itype).ne.0.d0)then
                if(xstsign.ne.sign(1.d0,wwx(i,iproc,itype)))then
                  if(iflag.eq.0)then
                    iwrong=iwrong+1
                    iflag=1
                  endif
                  rmax=max(rmax,abs(wwx(i,iproc,itype)))
                endif
              endif
            enddo
          enddo
        enddo
        if(iflag.eq.1)then
          if(xsum.ne.0.d0)rmax=rmax/xsum
          if(rmax.gt.0.05d0.or.xsum.eq.0.d0)iwrong1=iwrong1+1
        endif
        if(i0.eq.0.or.jproc0.eq.0.or.itype0.eq.0)then
          write(*,*)'Fatal error in xovh',i0,jproc0,itype0
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0)
        i1=idp1(i0,jproc0,itype0)
        i2=idp2(i0,jproc0,itype0)
        i3=idp3(i0,jproc0,itype0)
        call parcrossivh(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossivh(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit none
      integer jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o,
     # iallzero,ihprotrans(402:406)
      parameter (iallzero=1)
      integer ifuntype
      common/cifuntype/ifuntype
      data ihprotrans/401,0,403,403,401/
c
      if( (ifuntype.eq.1) .or. (ifuntype.eq.2.and.jproc0.eq.2) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif(ifuntype.eq.2.and.jproc0.eq.3)then
        if(ihpro.eq.401.or.ihpro.eq.403.or.ihpro.eq.407)then
          write(*,*)'Error #1 in parcrossivh:',ihpro,i1,i2,i3
          stop
        endif
        i1hproo=ihprotrans(ihpro)
        if(i1.eq.21)then
          ip1o=-i3
          ip2o=i2
          ip3o=i1
        elseif(i2.eq.21)then
          ip1o=i1
          ip2o=-i3
          ip3o=i2
        endif
      else
        write(*,*)'parcrossivh: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfvh(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checkvh(xsum,xsumabs,iret)
c iret=0 -> all vv entries are equal to zero
c iret=1 -> there is at least one entry which is not zero
c xsum is the sum of all the entries of vv
c xsumabs is the sum of the absolute value of all the entries of vv
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      real * 8 vv(4,2:3,6)
      common/cvv/vv
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=loproc,maproc
        do i=1,4
          do itype=1,6
            if(vv(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vv(i,jproc,itype)
            xsumabs=xsumabs+abs(vv(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine checkvhs(xsum,xsumabs,iret)
c identical to checkvh, except for the fact that works on vvs instead of vv,
c and jproc is not fixed
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      real * 8 vvs(4,2:3,6)
      common/cvvs/vvs
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=2,2
        do i=1,4
          do itype=1,6
            if(vvs(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vvs(i,jproc,itype)
            xsumabs=xsumabs+abs(vvs(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


C      function itoosoftkin()
Cc Returns 1 when a three-body kinematics can be safely approximated
Cc with a two-body kinematics. It is useful when three-body NLO configurations
Cc are obtained, which cannot be produced through showering
C      implicit none
C      integer itoosoftkin,itmp
Cc
C      itmp=0
C      itoosoftkin=itmp
C      return
C      end
c
c
c End of event-generation routines
c
c
c
c
c
c Begin of phase-space routines
c
c
      subroutine invvh(xm12,xm22,s,x,y,cth1,cth2,phil,cthl,str,
     #                 tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
c This routine has been obtained by modifying the analogous routine
c in the VV code. The notations have been changed: xm12 and xm22 where
c formerly called xmw2 and xmz2 respectively; pw, pz, pp, yw, yz, yp
c have been renamed ptv1, ptv2, ptvg, y1, y2, yg respectively.
c The names of the invariants are taken from Nucl.Phys.B383:3-44,1992 [FNR] 
c (q1q is q_1 of the paper, q2q is q_2, q1c is \hat{q}_1, q2c is \hat{q}_2.
c The hard process is
c   a(p1)+b(p2) --> V(k1)+H(k2)+c(k)
c                   |
c                   +--> l(l1)+lbar(l2)
c where a, b, and c are light partons, V is a vector boson with k1^2=xm12,
c H is a SM Higgs with k2^2=xm22, and l, lbar are the two leptons emerging
c from the V decay: at this stage, the difference between the lepton and
c the antilepton is purely conventional, and this convention can be freely
c modified without any modification to this routine. The process can be
c described by the same invariants as in FNR [eqs.(2.6) and (2.7)], 
c augmented by 
c
c   v1a = (p1-l1)^2
c   v1b = (p2-l1)^2
c   v1c = (k2+l1)^2 - k2^2
c
c and the llbar invariant mass, which coincides with xm12. In terms of the
c invariants, the dot products are (see newinv.m)
c
c    p1.p2 = s/2
c    p1.k  = -tk/2
c    p2.k  = -uk/2
c    p1.k1 = -(q1q-xm12)/2
c    p2.k2 = -(q2q-xm22)/2
c    k1.k2 = (s2-xm12-xm22)/2
c    p2.k1 = -(q2c-xm12)/2
c    p1.k2 = -(q1c-xm22)/2
c    k.k1  = (w1-xm12)/2
c    k.k2  = (w2-xm22)/2
c    p1.l1 = -v1a/2
c    p2.l1 = -v1b/2
c    k2.l1 = v1c/2
c    k.l1  = (-xm12-v1a-v1b-v1c)/2
c    k.l2  = (xm12-q1q+q2q-tk+v1a+v1b+v1c)/2
c    k2.l2 = (-xm12-xm22+s+tk+uk-v1c)/2
c    p1.l2 = (xm12-q1q+v1a)/2
c    p2.l2 = (-xm22+q2q+s+uk+v1b)/2
c
c The four momenta are given in the VH rest frame as follows
c     p1 = p10*(1,0,spsi2,cpsi2)
c     p2 = p20*(1,0,spsi ,cpsi )
c     k  = p20*(1,0,spsi1,cpsi1).
c     k1 = (k10, bx*sth2*sth1, bx*cth2*sth1, bx*cth1)
c     k2 = (k20,-bx*sth2*sth1,-bx*cth2*sth1,-bx*cth1).
c If str='p1', we define p1 = p10 (1,0,0,1) (psi2 =0), with psi and psi1
c determined using momentum conservation; according to the work done for 
c Drell Yan, the other options for str have been disabled.
c The momenta of the leptons are defined in the V rest frame
c     l1 = sqrt(xm12)/2*(1, spl*sthl, cpl*sthl, cthl)
c     l2 = sqrt(xm12)/2*(1,-spl*sthl,-cpl*sthl,-cthl)
c and boosted numerically to the VH rest frame in order to compute 
c the invariants. Here, spl=sin(phil), cpl=cos(phil), sthl^2=1-cthl^2,
c where phil and cthl are the two phase-space (angular) variables which
c fully describe the V decay in its rest frame.
c Notice that  bx = sqrt(s2)/2 * beta_x[paper]
c
c Call with:   idec=0    -->   V decays
c              idec=1    -->   V doesn't decay (v1a, v1b, v1c left blank)
c
      implicit none
      real * 8 xm12,xm22,s,x,y,cth1,cth2,phil,cthl,
     # tk,uk,q1q,q2q,v1a,v1b,v1c
      integer idec
      character * 2 str
      real * 8 ptv1,ptv2,ptvg,ptvl1,ptvl2,y1,y2,yg,yl1,yl2
      common/perpen/ptv1(2),ptv2(2),ptvg(2),ptvl1(2),ptvl2(2)
      common/ycmvar/y1,y2,yg,yl1,yl2
      real * 8 s2,drs2,p10,p20,k0,k10,k20,bx,sth1,cpsi,
     # spsi,cpsi2,spsi2,cpsi1,spsi1,xktsq,xkt1sq,xkt2sq,
     # xkt,xkt1,xkt2,tmp,sqs,tiny,zero,sth2,q1c,q2c,w1,w2,xm1,
     # k11,k12,k13,cpl,spl,sthl,l1bar0,l1bar1,l1bar2,l1bar3,
     # l10,l11,l12,l13,tp1l2,tp2l2,xktl1sq,xktl2sq,xktl1,xktl2,
     # e1lab,pl1lab,e2lab,pl2lab,el1lab,pll1lab,el2lab,pll2lab,
     # cosrho,sinrho
      integer i,j
      parameter (tiny=1.d-14)
      parameter (zero=0.d0)
      real*8 xmom_cm(7,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c
      tk=-s*(1-x)*(1-y)/2.d0
      uk=-s*(1-x)*(1+y)/2.d0
      s2 = tk+uk+s
      drs2 = 2*sqrt(s2)
      p10 = (s+tk)/drs2
      p20 = (s+uk)/drs2
      k0  = -(tk+uk)/drs2
      k10 = (s2+xm12-xm22)/drs2
      k20 = (s2+xm22-xm12)/drs2
      bx=sqrt(s2**2+xm22**2+xm12**2-2*(s2*xm22+s2*xm12+xm22*xm12))/drs2
      sth1 = sqrt(1-cth1**2)
      sth2 = sqrt(1-cth2**2)
      if(str.eq.'p1') then
         cpsi2 = 1
         spsi2 = 0
         cpsi = 1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
         spsi = 4*(1-x)*sqrt(x*(1-y**2))/
     #          ((1+y+x*(1-y))*(1-y+x*(1+y)))
         cpsi1 = (1+y-x*(1-y))/(1+y+x*(1-y))
         spsi1 = sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
      else
         write(6,*) 'Error in invvh: str=',str
         stop
      endif
      q1q = xm12 - 2*p10*(k10-bx*(cth2*sth1*spsi2+cth1*cpsi2))
      q2q = xm22 - 2*p20*(k20+bx*(cth2*sth1*spsi +cth1*cpsi ))
      q1c = xm12 + xm22 - s - tk - q1q
      q2c = xm12 + xm22 - s - uk - q2q
      w1  = xm12 - q1q + q2q - tk
      w2  = xm22 - q2q + q1q - uk
c
      if(abs(q1q-xm12).lt.tiny) then
        y1  = 1.d8
      elseif(abs(q2c-xm12).lt.tiny) then
        y1  = -1.d8
      else
        y1 = .5d0*log( (xm12-q2c)/(xm12-q1q) )
      endif
      if(abs(q1c-xm22).lt.tiny) then
        y2  = 1.d8
      elseif(abs(q2q-xm22).lt.tiny) then
        y2  = -1.d8
      else
        y2 = .5d0*log( (xm22-q2q)/(xm22-q1c) )
      endif
      if(abs(tk).lt.tiny) then
        yg  = 1.d8
      elseif(abs(uk).lt.tiny) then
        yg  = -1.d8
      else
        yg  = .5d0*log( uk/tk )
      endif
c-----------------------------------------------------------------
c xktsq, xkt1sq e xkt2sq are the square of transverse momenta of g, V, 
c and H respectively. The axis orientation is such that V is always
c along the x direction. The component of p_T(H) along the y direction
c is always positive or zero
c
      xktsq = uk*tk/s
      if(xktsq.eq.0) then
         ptv1(1) = bx*sth1
         ptv1(2) = 0.d0
         ptv2(1) = -ptv1(1)
         ptv2(2) = 0.d0
         ptvg(1) = 0.d0
         ptvg(2) = 0.d0
         xkt1 = ptv1(1)
         xkt2 = xkt1
      else
         xkt1sq = (xm12-q2c)*(xm12-q1q)/s - xm12
         xkt2sq = (xm22-q2q)*(xm22-q1c)/s - xm22
         xkt = sqrt(xktsq)
         xkt1 = sqrt(xkt1sq)
         xkt2 = sqrt(xkt2sq)
         ptv1(1) = xkt1
         ptv1(2) = 0.d0
         ptv2(1) = (xktsq-xkt1sq-xkt2sq)/(2.d0*xkt1)
         tmp = xkt2sq-ptv2(1)**2
         if(tmp.gt.0.d0)then
            ptv2(2) = sqrt(tmp)
         else
            ptv2(2) = 0.d0
         endif
         ptvg(1) = (xkt2sq-xkt1sq-xktsq)/(2.d0*xkt1)
         tmp = xktsq-ptvg(1)**2
         if(tmp.gt.0.d0)then
            ptvg(2) = -sqrt(tmp)
         else
            ptvg(2) = 0.d0
         endif
      endif
      if(ichkmom.eq.0)call checkptcon(ptv1,ptv2,ptvg)
c
      sqs=sqrt(s)
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqs/2.d0
      xmom_cm(1,4)=sqs/2.d0
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqs/2.d0
      xmom_cm(2,4)=sqs/2.d0
      if(tk.eq.0.d0.and.uk.eq.0.d0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=0.d0
        xmom_cm(3,4)=0.d0
      elseif(tk.eq.0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=-uk/(2*sqs)
        xmom_cm(3,4)=xmom_cm(3,3)
      elseif(uk.eq.0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=tk/(2*sqs)
        xmom_cm(3,4)=-xmom_cm(3,3)
      else
        xmom_cm(3,1)=ptvg(1)
        xmom_cm(3,2)=ptvg(2)
        xmom_cm(3,3)=sqs/2.d0*(1-x)*y
        xmom_cm(3,4)=sqs/2.d0*(1-x)
      endif
      e1lab=(2*xm12-q1q-q2c)/(2*sqs)
      pl1lab=(q1q-q2c)/(2*sqs)
      xmom_cm(4,1)=ptv1(1)
      xmom_cm(4,2)=ptv1(2)
      xmom_cm(4,3)=pl1lab
      xmom_cm(4,4)=e1lab
      e2lab=(2*xm22-q1c-q2q)/(2*sqs)
      pl2lab=(q1c-q2q)/(2*sqs)
      xmom_cm(5,1)=ptv2(1)
      xmom_cm(5,2)=ptv2(2)
      xmom_cm(5,3)=pl2lab
      xmom_cm(5,4)=e2lab
c
      if(idec.eq.0)then
        cpl=cos(phil)
        spl=sin(phil)
        sthl=sqrt(1-cthl**2)
        xm1=sqrt(xm12)
c Momentum of k1 in the rest frame of k1+k2
        k11=bx*sth2*sth1
        k12=bx*cth2*sth1
        k13=bx*cth1
c Momentum of l1 in the rest frame of k1
        l1bar0=xm1/2.d0
        l1bar1=l1bar0*spl*sthl
        l1bar2=l1bar0*cpl*sthl
        l1bar3=l1bar0*cthl
c (l10,..,l13) is the momentum of l1 in the rest frame of k1+k2
        call hvhlb4(k11,k12,k13,k10,xm1,
     #              l1bar1,l1bar2,l1bar3,l1bar0,
     #              l11,l12,l13,l10)
c
        v1a=-2*p10*(l10-l12*spsi2-l13*cpsi2)
        v1b=-2*p20*(l10-l12*spsi -l13*cpsi )
        v1c=2*(k20*l10+bx*(sth2*sth1*l11+cth2*sth1*l12+cth1*l13 ))
        tp1l2=xm12-q1q+v1a
        tp2l2=-xm22+q2q+s+uk+v1b
        xktl1sq=v1a*v1b/s
        xktl1=sqrt(xktl1sq)
        xktl2sq=tp1l2*tp2l2/s
        xktl2=sqrt(xktl2sq)
        yl1=0.5d0*log(v1b/v1a)
        yl2=0.5d0*log(tp2l2/tp1l2)
        el1lab=-(v1a+v1b)/(2.d0*sqs)
        pll1lab=(v1a-v1b)/(2.d0*sqs)
c Exploit the fact that V is always along x in the transverse plane
        if(xkt1.ne.0.d0)then
          ptvl1(1)=(e1lab*el1lab-pl1lab*pll1lab-xm12/2.d0)/xkt1
          tmp=xktl1sq-ptvl1(1)**2
          if(tmp.gt.0.d0)then
            ptvl1(2)=sqrt(tmp)
            if(x.lt.1.d0.and.abs(y).lt.1.d0)then
c tmp here is pt(H)_y * pt(l1)_y, and we have set pt(H)_y > 0
              tmp=e2lab*el1lab-pl2lab*pll1lab-ptv2(1)*ptvl1(1)-v1c/2.d0
              if(tmp.lt.0.d0)ptvl1(2)=-ptvl1(2)
            else
c The rest frame of k1+k2 coincides with the partonic cm frame in the
c soft and (up to a longitudinal boost) collinear limits. We can thus
c get ptv1(1) by rotating k1 as given before, and apply the same rotation
c to the lepton. Notice that in our parametrization the components 1 and 2
c are proportional to sin and cos (usually, it is the opposite)
              cosrho=k12/ptv1(1)
              sinrho=k11/ptv1(1)
              if( sign(1.d0,ptvl1(2)) .ne.
     #          sign(1.d0,-l12*sinrho+l11*cosrho) )ptvl1(2)=-ptvl1(2)
            endif
          else
            ptvl1(2)=0.d0
          endif
        else
          ptvl1(1)=0.d0
          ptvl1(2)=0.d0
        endif
        xmom_cm(6,1)=ptvl1(1)
        xmom_cm(6,2)=ptvl1(2)
        xmom_cm(6,3)=pll1lab
        xmom_cm(6,4)=el1lab
        el2lab=(tp1l2+tp2l2)/(2.d0*sqs)
        pll2lab=(tp2l2-tp1l2)/(2.d0*sqs)
        ptvl2(1)=ptv1(1)-ptvl1(1)
        ptvl2(2)=ptv1(2)-ptvl1(2)
        xmom_cm(7,1)=ptvl2(1)
        xmom_cm(7,2)=ptvl2(2)
        xmom_cm(7,3)=pll2lab
        xmom_cm(7,4)=el2lab
      elseif(idec.eq.1)then
        v1a=0.d0
        v1b=0.d0
        v1c=0.d0
        do i=6,7
          do j=1,4
            xmom_cm(i,j)=0.d0
          enddo
        enddo
      else
        write(6,*) 'Error in invvh: idec=',idec
        stop
      endif
      if(ichkmom.eq.0)then
        call checkmvh(xmom_cm,s,0.d0,1,2)
        if(idec.eq.0)then
          call checkmvh(xmom_cm,s,0.d0,1,1)
          call checkdec(xmom_cm,4,6,7)
        endif
      endif
      return
      end


      subroutine checkmvh(xmom,smax,ybst,iflag,itype)
      implicit none
      real * 8 xmom(7,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-4)
      integer iflag,itype,i,j,jj,jflag,jeflag,jmax
c
      if(itype.eq.1)then
        jmax=7
      elseif(itype.eq.2)then
        jmax=5
      else
        write(6,*)'Wrong option in checkmvh'
        stop
      endif
      jflag=0
      jeflag=0
      xpmax=sqrt(smax)/2.d0*(1+vtiny)
      do i=1,4
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=1,jmax
          if((itype.eq.1.and.j.ne.4).or.itype.eq.2)then
            if(i.ne.4.and.xmom(j,i).gt.xpmax)jeflag=1
            xsign=1.d0
            if(j.eq.1.or.j.eq.2)xsign=-1.d0
            xsum(i)=xsum(i)+xmom(j,i)*xsign
            xsuma(i)=xsuma(i)+abs(xmom(j,i))
          endif
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'iflag,itype,i=',iflag,itype,i
          write(*,*)'smax,y=',smax,ybst
          write(*,*)'x1,x2=',x1,x2
          do j=1,7
            write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=1,4)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=1,4)
        stop
      endif
      if(jeflag.eq.1)then
        write(*,*)'Momentum component larger than sqrt(s)/2'
        write(*,*)'iflag=',iflag
        write(*,*)'s,pmax,y=',smax,xpmax,ybst
        write(*,*)'x1,x2=',x1,x2
        do j=1,7
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end


C      subroutine checkptcon(ptvl1,ptvl2,ptvg)
C      implicit none
C      real*8 ptvl1(2),ptvl2(2),ptvg(2),tiny,pt1,pt2,ptmax
C      parameter (tiny=1.d-5)
C      integer jj
Cc
C      ptmax=max(abs(ptvl1(1)),abs(ptvl2(1)),abs(ptvg(1)),
C     #          abs(ptvl1(2)),abs(ptvl2(2)),abs(ptvg(2)))
C      pt1=ptvl1(1)+ptvl2(1)+ptvg(1)
C      pt2=ptvl1(2)+ptvl2(2)+ptvg(2)
C      if(pt1.gt.ptmax*tiny.or.pt2.gt.ptmax*tiny)then
C        write(*,*)'Transverse momentum is not conserved'
C        write(*,'(4(d14.8,1x))') (ptvl1(jj),jj=1,2)
C        write(*,'(4(d14.8,1x))') (ptvl2(jj),jj=1,2)
C        write(*,'(4(d14.8,1x))') (ptvg(jj),jj=1,2)
C        stop
C      endif
C      return
C      end


      subroutine checkdec(xmom,idec,iprod1,iprod2)
      implicit none
      real * 8 xmom(7,4)
      real * 8 tiny,xsum(4),xsuma(4),xrat(4)
      parameter (tiny=5.d-3)
      integer idec,iprod1,iprod2,jflag,i,jj
c
      jflag=0
      do i=1,4
        xsum(i)=xmom(idec,i)-xmom(iprod1,i)-xmom(iprod2,i)
        xsuma(i)=abs(xmom(idec,i))+
     #           abs(xmom(iprod1,i))+abs(xmom(iprod2,i))
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Checkdec: momentum is not conserved'
          write(*,'(4(d14.8,1x))') (xmom(idec,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod1,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod2,jj),jj=1,4)
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=1,4)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=1,4)
        stop
      endif
      return
      end


c This is obtained by modifying the routine HVHLB4 of HERWIG
C-----------------------------------------------------------------------
      subroutine hvhlb4(ps1,ps2,ps3,ps4,ps5,
     #                  pi1,pi2,pi3,pi4,
     #                  pf1,pf2,pf3,pf4)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN REST FRAME OF PS) INTO PF (IN LAB)
C     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
C-----------------------------------------------------------------------
      real*8 ps1,ps2,ps3,ps4,ps5,pi1,pi2,pi3,pi4,pf1,pf2,pf3,pf4,
     # tmp,fn
      if (ps4.eq.ps5) then
        pf1= pi1
        pf2= pi2
        pf3= pi3
        pf4= pi4
      else
        tmp  = (pi1*ps1+pi2*ps2
     &         +pi3*ps3+pi4*ps4)/ps5
        fn   = (tmp+pi4) / (ps4+ps5)
        pf1= pi1 + fn*ps1
        pf2= pi2 + fn*ps2
        pf3= pi3 + fn*ps3
        pf4= tmp
      end if
      end


C      function bwfunc(s,xm02,gah)
Cc Returns the Breit Wigner function, normalized in such a way that
Cc its integral in the range (-inf,inf) is one
C      implicit none
C      real*8 bwfunc,s,xm02,gah
C      real*8 pi,xm0
C      parameter (pi=3.1415926535897932d0)
Cc
C      xm0=sqrt(xm02)
C      bwfunc=xm0*gah/(pi*((s-xm02)**2+xm02*gah**2))
C      return
C      end


      function xbwmass3(t,xm02,ga,bwdelf,bwfmmn)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmass3,t,xm02,ga,bwdelf,bwfmmn
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      xbwmass3=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
      return
      end


C      subroutine zzchvar(parth1,cth1,xjac,ro)
Cc
Cc Given 0<parth1<1 returns -1<cth1<1
Cc and multiplies xjac times the d cth1 / d parth1 jacobian
Cc
C      implicit none
C      real * 8 parth1,cth1,xjac,ro,bb,xlgbb,yy,expyy
C      bb = 1-ro**2/16
C      xlgbb = log((1+bb)/(1-bb))
C      yy = ( parth1 * 2 - 1 ) * xlgbb
C      xjac = xjac * 2 * xlgbb
C      expyy = exp(-yy)
C      cth1 = (1-expyy)/(1+expyy)/bb
C      xjac = xjac * 2 * expyy/(1+expyy)**2 / bb
C      return
C      end
c
c
c End of phase-space routines
c
c
C      FUNCTION FK88RANDOM(SEED)
C*     -----------------
C* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
C* Use seed = 1 as first value.
C*
C      IMPLICIT INTEGER(A-Z)
C      DOUBLE PRECISION MINV,FK88RANDOM
C      SAVE
C      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
C      PARAMETER(MINV=0.46566128752458d-09)
C      HI = SEED/Q
C      LO = MOD(SEED,Q)
C      SEED = A*LO - R*HI
C      IF(SEED.LE.0) SEED = SEED + M
C      FK88RANDOM = SEED*MINV
C      END
c
c
c Initialization
c
c
      subroutine setpvh()
c Since the structure of VH production is identical to that of DrellYan, the 
c matrix elements can be written as in Aurenche and Lindfors NPB185(81)274.
c Some reference is made to that paper. Notice that the conventions for the
c a and b coefficients entering the Vqq or Vll vertices are not the same
c here as in AL (i.e., they differ wrt those in mcatnlo_llmain.f); we use
c here the conventions of ESW. Refer to the routines xmewrap_und and
c xmewrap_dec for the normalization
      implicit none
      include 'vhgcblks.h'
      real * 8 pi,zup,zdo,aem,xmz,xmw,xmz2,xmw2,xme,xmmu,xmtau,
     #  cthw2,sthw2,deccf,xalfaem
      parameter (pi=3.14159265358979312D0)
      parameter (zup=2.d0/3.d0,zdo=-1.d0/3.d0)
c Values from PDG 2003
      parameter (xmz=91.1876d0)
      parameter (xmw=80.425d0)
      parameter (xme=0.510998902d-3)
      parameter (xmmu=105.6583568d-3)
      parameter (xmtau=1776.99d-3)
      real * 8 zel(1:6),zel2(1:6),ac(1:6),vc(1:6)
      real * 8 ckm(1:6,1:6),ckm2(1:6,1:6),xkm(1:6,1:6)
      real * 8 aupp(1:6),bupp(1:6)
      real * 8 vcel,acel,aelupp,belupp
      real * 8 vcnu,acnu,anuupp,bnuupp
      real * 8 weinsc(1:3),xlepacf(1:3),xlepbcf(1:3)
      real * 8 xqrkacf(1:6,1:3),xqrkbcf(1:6,1:3)
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      real * 8 xlep1mass,xlep2mass
      real * 8 wdtwon,wdtwmsb,wdtzon,wdtzmsb
      common/charges/zel,zel2
      common/cckm2/ckm2
      common/cxvacoef/xveccoef,xaxlcoef
      common/clepmass/xlep1mass,xlep2mass
      common/partdec/wdtwon,wdtwmsb,wdtzon,wdtzmsb
      integer iprdct
      common/ciprdct/iprdct
      integer ilepnlo
      common/cilepnlo/ilepnlo
      integer ip4,ip5,ip6
      common/ci2part/ip4,ip5,ip6
c Charged lepton from W decay
      integer ichlw(1:3)
      data ichlw/11,13,15/
c Neutrino from W decay
      integer ineuw(1:3)
      data ineuw/12,14,16/
c Charged lepton from Z decay
      integer ichlz(1:6)
      data ichlz/11,13,15,0,0,0/
c Neutrino from Z decay
      integer ineuz(1:6)
      data ineuz/0,0,0,12,14,16/
      integer i,j
c
c Number of light flavours
      nl=5
c Fermi constant, from PDG2002
      gf=1.16639d-5
c alpha_em
      aem=1/137.0359895d0
c electron charge squared
      ze2=4*pi*aem
      ze2v=ze2
c sin and cos squared of theta_W; MSbar scheme, from PDG2003
      sthw2=0.23113d0
      cthw2=1-sthw2
c The matrix ZEL(I) returns the electric charge of flavour I. The
c numbering convention for the flavours should match the one of
c the structure function package. In our case, 
c 1=UP, 2=DOWN, 3=STRANGE, 4=CHARM, 5=BOTTOM, 6=TOP
c ZEL2(I) returns the charge squared.
      zel(1)=zup
      zel(2)=zdo
      zel(3)=zdo
      zel(4)=zup
      zel(5)=zdo
      zel(6)=zup
      do i=1,6
        zel2(i)=zel(i)**2
      enddo
c axial and vector couplings to the Z (Ellis Stirling Webber)
      ac(1)=0.5d0
      ac(2)=-0.5d0
      ac(3)=-0.5d0
      ac(4)=0.5d0
      ac(5)=-0.5d0
      ac(6)=0.5d0
      vc(1)=0.5d0-4*sthw2/3.d0
      vc(2)=-0.5d0+2*sthw2/3.d0
      vc(3)=-0.5d0+2*sthw2/3.d0
      vc(4)=0.5d0-4*sthw2/3.d0
      vc(5)=-0.5d0+2*sthw2/3.d0
      vc(6)=0.5d0-4*sthw2/3.d0
c A and B coefficients, for qqZ couplings
      do i=1,6
        aupp(i)=vc(i)**2+ac(i)**2
        bupp(i)=-2*vc(i)*ac(i)
      enddo
c A and B coefficients, for charged leptons-Z couplings
      vcel=-0.5d0+2*sthw2
      acel=-0.5d0
      aelupp=vcel**2+acel**2
      belupp=-2*vcel*acel
c A and B coefficients, for neutrinos-Z couplings
      vcnu=0.5d0
      acnu=0.5d0
      anuupp=vcnu**2+acnu**2
      bnuupp=-2*vcnu*acnu
c ckm(i,j)=|CKM matrix elements|, with  i=1,4,6 --> up,charm,top
c                                       j=2,3,5 --> down,strange,bottom
c Values from PDG 2003
      do i=1,6
        do j=1,6
          ckm(i,j)=0.d0
        enddo
      enddo
c Values from PDG 2003; individual results from eqs.(11.6)-(11.13); not
c used since unitary is not taken into account
      xkm(1,2)=0.9734d0
      xkm(1,3)=0.2196d0
      xkm(1,5)=3.6d-3
      xkm(4,2)=0.224d0
      xkm(4,3)=0.996d0
      xkm(4,5)=41.2d-3
      xkm(6,2)=0.0035d0
      xkm(6,3)=0.04d0
      xkm(6,5)=0.999d0
c Centers of the ranges given in eq.(11.2), supposedly taking unitarity
c into accout; with the following entries, it holds better than 0.1%
      ckm(1,2)=0.9748d0
      ckm(1,3)=0.2225d0
      ckm(1,5)=0.0036d0
      ckm(4,2)=0.2225d0
      ckm(4,3)=0.9740d0
      ckm(4,5)=0.041d0
      ckm(6,2)=0.009d0
      ckm(6,3)=0.0405d0
      ckm(6,5)=0.9992d0
      do i=1,6
        do j=1,6
          ckm2(i,j)=ckm(i,j)**2
        enddo
      enddo
c Theta_W dependence of the couplings; these are the factors
c   g_VH^2 * g_Vqq^2 / e^4                for undecayed vector boson
c   g_VH^2 * g_Vqq^2 * g_Vll^2 / e^6      for decayed vector boson
c Entries 1 and 2 are unused here, since they correspond to gamma exchange
c and gamma/Z intereference terms respectively, as in NPB185(81)274.
c A factor M_V^2 due to the VVH vertex has been inserted in the routines
c that return the matrix elements
      if(iprdct.eq.270)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(16*sthw2**3*cthw2**3)
      elseif(iprdct.eq.278)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(4*sthw2**2*cthw2**2)
      elseif(iprdct.eq.248.or.iprdct.eq.268)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(8*sthw2**2)
      elseif(iprdct.eq.240.or.iprdct.eq.260)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(64*sthw2**3)
      else
        write(*,*)'Error in setpvh: unknown IPROC',iprdct
        stop
      endif
c Coefficients of the couplings which depend on lepton identities; set to
c one in the case of undecayed vector boson
      if(iprdct.eq.270)then
        if(ilepnlo.le.3)then
          xlepacf(1)=0.d0
          xlepbcf(1)=0.d0
          xlepacf(2)=0.d0
          xlepbcf(2)=0.d0
          xlepacf(3)=aelupp
          xlepbcf(3)=belupp
        else
          xlepacf(1)=0.d0
          xlepbcf(1)=0.d0
          xlepacf(2)=0.d0
          xlepbcf(2)=0.d0
          xlepacf(3)=anuupp
          xlepbcf(3)=bnuupp
        endif
      elseif(iprdct.eq.240.or.iprdct.eq.260)then
        xlepacf(1)=0.d0
        xlepbcf(1)=0.d0
        xlepacf(2)=0.d0
        xlepbcf(2)=0.d0
        xlepacf(3)=2.d0
        xlepbcf(3)=-2.d0
      elseif(iprdct.eq.248.or.iprdct.eq.268.or.iprdct.eq.278)then
        xlepacf(1)=0.d0
        xlepbcf(1)=0.d0
        xlepacf(2)=0.d0
        xlepbcf(2)=0.d0
        xlepacf(3)=1.d0
        xlepbcf(3)=0.d0
      endif
c Coefficients of the couplings which depend on quark identities.
c The first index runs over types of incoming parton pairs, 
c according to the conventions given in the subroutine strfvh; 
c the second index runs in principle over the values 1,2,3, but only
c 3 is non trivial (see the comment preceding the definition of weinsc)
      do i=1,6
        do j=1,3
          xqrkacf(i,j)=0.d0
          xqrkbcf(i,j)=0.d0
        enddo
      enddo
      if(iprdct.eq.270)then
        do i=1,6
          xqrkacf(i,3)=aupp(i)
          xqrkbcf(i,3)=bupp(i)
        enddo
      elseif(iprdct.eq.278)then
        do i=1,6
          xqrkacf(i,3)=aupp(i)
        enddo
      elseif(iprdct.eq.240.or.iprdct.eq.260)then
        do i=1,6
          xqrkacf(i,3)=2.d0
          xqrkbcf(i,3)=-2.d0
        enddo
      elseif(iprdct.eq.248.or.iprdct.eq.268)then
        do i=1,6
          xqrkacf(i,3)=2.d0
        enddo
      endif
c Coefficients multiplying the vector (xveccoef()) and axial (xaxlcoef())
c part of the cross section, as in eqs.(7) and (8) of NPB185(81)274. The
c axial part is unused for undecayed vector boson. When multiplied by the
c corresponding output of the subroutine xmewrap, the complete partonic
c cross section is obtained, up g_S and e^4 (for undecayed vector boson)
c or e^6 (for decayed vector boson). The index conventions follow those 
c of strfvh 
      do i=1,6
        do j=1,3
          xveccoef(i,j)=xqrkacf(i,j)*xlepacf(j)*weinsc(j)
          xaxlcoef(i,j)=xqrkbcf(i,j)*xlepbcf(j)*weinsc(j)
        enddo
      enddo
c Lepton masses
      if(iprdct.eq.240.or.iprdct.eq.260.or.iprdct.eq.270)then
        if(ilepnlo.lt.1.or.ilepnlo.gt.6)then
          write(*,*)'Error in setpvh: unknown lepton type',ilepnlo
          stop
        endif
        if(iprdct.eq.270)then
          if(ilepnlo.eq.1)then
            xlep1mass=xme
            xlep2mass=xme
          elseif(ilepnlo.eq.2)then
            xlep1mass=xmmu
            xlep2mass=xmmu
          elseif(ilepnlo.eq.3)then
            xlep1mass=xmtau
            xlep2mass=xmtau
          else
            xlep1mass=0.d0
            xlep2mass=0.d0
          endif
        elseif(iprdct.eq.240)then
          if(ilepnlo.eq.1)then
            xlep1mass=0.d0
            xlep2mass=xme
          elseif(ilepnlo.eq.2)then
            xlep1mass=0.d0
            xlep2mass=xmmu
          elseif(ilepnlo.eq.3)then
            xlep1mass=0.d0
            xlep2mass=xmtau
          else
            write(*,*)'Error in setpvh: inconsistent entries'
            stop
          endif
        elseif(iprdct.eq.260)then
          if(ilepnlo.eq.1)then
            xlep1mass=xme
            xlep2mass=0.d0
          elseif(ilepnlo.eq.2)then
            xlep1mass=xmmu
            xlep2mass=0.d0
          elseif(ilepnlo.eq.3)then
            xlep1mass=xmtau
            xlep2mass=0.d0
          else
            write(*,*)'Error in setpvh: inconsistent entries'
            stop
          endif
        endif
      endif
c Partial W and Z decay widths in the on-shell and MSbar schemes
      if(iprdct.ge.270)then
        if(ilepnlo.le.3)then
          deccf=aelupp
        else
          deccf=anuupp
        endif
        xmz2=xmz**2
        wdtzon=deccf*gf*xmz2*xmz/(6*sqrt(2.d0)*pi)
        wdtzmsb=deccf*xmz*xalfaem(xmz2)/(12*sthw2*cthw2)
      else
        xmw2=xmw**2
        wdtwon=gf*xmw2*xmw/(6*sqrt(2.d0)*pi)
        wdtwmsb=xmw*xalfaem(xmw2)/(12*sthw2)
      endif
c Final-state particle identities (except light parton)
      if(iprdct.eq.240)then
c W^+ H with decay
        ip4=25
        ip5=ineuw(ilepnlo)
        ip6=-ichlw(ilepnlo)
      elseif(iprdct.eq.248)then
c W^+ H without decay
        ip4=24
        ip5=25
        ip6=0
      elseif(iprdct.eq.260)then
c W^- H with decay
        ip4=25
        ip5=ichlw(ilepnlo)
        ip6=-ineuw(ilepnlo)
      elseif(iprdct.eq.268)then
c W^- H without decay
        ip4=-24
        ip5=25
        ip6=0
      elseif(iprdct.eq.270)then
c Z H with decay
        ip4=25
        if(ilepnlo.le.3)then
          ip5=ichlz(ilepnlo)
          ip6=-ichlz(ilepnlo)
        else
          ip5=ineuz(ilepnlo)
          ip6=-ineuz(ilepnlo)
        endif
      elseif(iprdct.eq.278)then
c Z H without decay
        ip4=23
        ip5=25
        ip6=0
      else
        write(*,*)'Error in setpvh: unknown process',iprdct
        stop
      endif
      return
      end


      subroutine parsetpvh()
      implicit none
      integer jproc,i,itype,iprdct
      integer imapp(0:5)
      integer ivbhpro(4,2:3,6)
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cimapp/imapp
      common/civbhpro/ivbhpro
      common/cidpart/idp1,idp2,idp3
      common/ciprdct/iprdct
      integer nl
      common/nl/nl
c
c imapp(i) returns the PDG id number (1=d, 2=u, 3=s, 4=c, 5=b, 21=g)
c given our id number (1=u, 2=d, 3=s, 4=c, 5=b, 0=g)
      imapp(0)=21
      imapp(1)=2
      imapp(2)=1
      imapp(3)=3
      imapp(4)=4
      imapp(5)=5
c
c ivbhpro returns the process number associated to the entries; this is
c identical to i1hpro (see the routine store_evenvh)
      do i=1,4
        do jproc=2,3
          do itype=1,6
            ivbhpro(i,jproc,itype)=0
          enddo
        enddo
      enddo
c 
      if(iprdct.eq.270.or.iprdct.eq.278)then
c ZH production
        do itype=1,nl
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      elseif(iprdct.eq.240.or.iprdct.eq.248)then
c W+H production
        do itype=1,6
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
        enddo
        do itype=1,6
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      elseif(iprdct.eq.260.or.iprdct.eq.268)then
c W-H production
        do itype=1,6
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
        enddo
        do itype=1,6
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      else
        write(*,*)'Unknown process in parsetpvh:',iprdct
        stop
      endif
c
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=outgoing) in the process associated to the
c entries. The labelling scheme of PDG has been used
      do i=1,4
        do jproc=2,3
          do itype=1,6
            idp1(i,jproc,itype)=0
            idp2(i,jproc,itype)=0
            idp3(i,jproc,itype)=0
          enddo
        enddo
      enddo
c
      if(iprdct.eq.270.or.iprdct.eq.278)then
c ZH production
        do itype=1,nl
          idp1(1,2,itype)=imapp(itype)
          idp1(3,2,itype)=-imapp(itype)
c
          idp2(1,2,itype)=-imapp(itype)
          idp2(3,2,itype)=imapp(itype)
c
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
c
          idp1(1,3,itype)=imapp(itype)
          idp1(2,3,itype)=-imapp(itype)
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
c
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp2(3,3,itype)=imapp(itype)
          idp2(4,3,itype)=-imapp(itype)
c
          idp3(1,3,itype)=imapp(itype)
          idp3(2,3,itype)=-imapp(itype)
          idp3(3,3,itype)=imapp(itype)
          idp3(4,3,itype)=-imapp(itype)
c
        enddo
      elseif(iprdct.eq.240.or.iprdct.eq.248)then
c W+H production
        idp1(1,2,1)=imapp(1)
        idp1(1,2,2)=imapp(1)
        idp1(1,2,3)=imapp(1)
        idp1(1,2,4)=imapp(4)
        idp1(1,2,5)=imapp(4)
        idp1(1,2,6)=imapp(4)
c
        idp1(3,2,1)=-imapp(2)
        idp1(3,2,2)=-imapp(3)
        idp1(3,2,3)=-imapp(5)
        idp1(3,2,4)=-imapp(2)
        idp1(3,2,5)=-imapp(3)
        idp1(3,2,6)=-imapp(5)
c
        idp2(1,2,1)=-imapp(2)
        idp2(1,2,2)=-imapp(3)
        idp2(1,2,3)=-imapp(5)
        idp2(1,2,4)=-imapp(2)
        idp2(1,2,5)=-imapp(3)
        idp2(1,2,6)=-imapp(5)
c
        idp2(3,2,1)=imapp(1)
        idp2(3,2,2)=imapp(1)
        idp2(3,2,3)=imapp(1)
        idp2(3,2,4)=imapp(4)
        idp2(3,2,5)=imapp(4)
        idp2(3,2,6)=imapp(4)
c
        do itype=1,6
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
        enddo
c
        idp1(1,3,1)=imapp(1)
        idp1(1,3,2)=imapp(1)
        idp1(1,3,3)=imapp(1)
        idp1(1,3,4)=imapp(4)
        idp1(1,3,5)=imapp(4)
        idp1(1,3,6)=imapp(4)
c
        idp1(2,3,1)=-imapp(2)
        idp1(2,3,2)=-imapp(2)
        idp1(2,3,3)=-imapp(3)
        idp1(2,3,4)=-imapp(3)
        idp1(2,3,5)=-imapp(5)
        idp1(2,3,6)=-imapp(5)
c
        idp2(3,3,1)=imapp(1)
        idp2(3,3,2)=imapp(1)
        idp2(3,3,3)=imapp(1)
        idp2(3,3,4)=imapp(4)
        idp2(3,3,5)=imapp(4)
        idp2(3,3,6)=imapp(4)
c
        idp2(4,3,1)=-imapp(2)
        idp2(4,3,2)=-imapp(2)
        idp2(4,3,3)=-imapp(3)
        idp2(4,3,4)=-imapp(3)
        idp2(4,3,5)=-imapp(5)
        idp2(4,3,6)=-imapp(5)
c
        idp3(1,3,1)=imapp(2)
        idp3(1,3,2)=imapp(3)
        idp3(1,3,3)=imapp(5)
        idp3(1,3,4)=imapp(2)
        idp3(1,3,5)=imapp(3)
        idp3(1,3,6)=imapp(5)
c
        idp3(2,3,1)=-imapp(1)
        idp3(2,3,2)=-imapp(4)
        idp3(2,3,3)=-imapp(1)
        idp3(2,3,4)=-imapp(4)
        idp3(2,3,5)=-imapp(1)
        idp3(2,3,6)=-imapp(4)
c
        idp3(3,3,1)=imapp(2)
        idp3(3,3,2)=imapp(3)
        idp3(3,3,3)=imapp(5)
        idp3(3,3,4)=imapp(2)
        idp3(3,3,5)=imapp(3)
        idp3(3,3,6)=imapp(5)
c
        idp3(4,3,1)=-imapp(1)
        idp3(4,3,2)=-imapp(4)
        idp3(4,3,3)=-imapp(1)
        idp3(4,3,4)=-imapp(4)
        idp3(4,3,5)=-imapp(1)
        idp3(4,3,6)=-imapp(4)
c
        do itype=1,6
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
        enddo
      elseif(iprdct.eq.260.or.iprdct.eq.268)then
c W- production
        idp1(1,2,1)=imapp(2)
        idp1(1,2,2)=imapp(2)
        idp1(1,2,3)=imapp(3)
        idp1(1,2,4)=imapp(3)
        idp1(1,2,5)=imapp(5)
        idp1(1,2,6)=imapp(5)
c
        idp1(3,2,1)=-imapp(1)
        idp1(3,2,2)=-imapp(4)
        idp1(3,2,3)=-imapp(1)
        idp1(3,2,4)=-imapp(4)
        idp1(3,2,5)=-imapp(1)
        idp1(3,2,6)=-imapp(4)
c
        idp2(1,2,1)=-imapp(1)
        idp2(1,2,2)=-imapp(4)
        idp2(1,2,3)=-imapp(1)
        idp2(1,2,4)=-imapp(4)
        idp2(1,2,5)=-imapp(1)
        idp2(1,2,6)=-imapp(4)
c
        idp2(3,2,1)=imapp(2)
        idp2(3,2,2)=imapp(2)
        idp2(3,2,3)=imapp(3)
        idp2(3,2,4)=imapp(3)
        idp2(3,2,5)=imapp(5)
        idp2(3,2,6)=imapp(5)
c
        do itype=1,6
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
        enddo
c
        idp1(1,3,1)=imapp(2)
        idp1(1,3,2)=imapp(2)
        idp1(1,3,3)=imapp(3)
        idp1(1,3,4)=imapp(3)
        idp1(1,3,5)=imapp(5)
        idp1(1,3,6)=imapp(5)
c
        idp1(2,3,1)=-imapp(1)
        idp1(2,3,2)=-imapp(1)
        idp1(2,3,3)=-imapp(1)
        idp1(2,3,4)=-imapp(4)
        idp1(2,3,5)=-imapp(4)
        idp1(2,3,6)=-imapp(4)
c
        idp2(3,3,1)=imapp(2)
        idp2(3,3,2)=imapp(2)
        idp2(3,3,3)=imapp(3)
        idp2(3,3,4)=imapp(3)
        idp2(3,3,5)=imapp(5)
        idp2(3,3,6)=imapp(5)
c
        idp2(4,3,1)=-imapp(1)
        idp2(4,3,2)=-imapp(1)
        idp2(4,3,3)=-imapp(1)
        idp2(4,3,4)=-imapp(4)
        idp2(4,3,5)=-imapp(4)
        idp2(4,3,6)=-imapp(4)
c
        idp3(1,3,1)=imapp(1)
        idp3(1,3,2)=imapp(4)
        idp3(1,3,3)=imapp(1)
        idp3(1,3,4)=imapp(4)
        idp3(1,3,5)=imapp(1)
        idp3(1,3,6)=imapp(4)
c
        idp3(2,3,1)=-imapp(2)
        idp3(2,3,2)=-imapp(3)
        idp3(2,3,3)=-imapp(5)
        idp3(2,3,4)=-imapp(2)
        idp3(2,3,5)=-imapp(3)
        idp3(2,3,6)=-imapp(5)
c
        idp3(3,3,1)=imapp(1)
        idp3(3,3,2)=imapp(4)
        idp3(3,3,3)=imapp(1)
        idp3(3,3,4)=imapp(4)
        idp3(3,3,5)=imapp(1)
        idp3(3,3,6)=imapp(4)
c
        idp3(4,3,1)=-imapp(2)
        idp3(4,3,2)=-imapp(3)
        idp3(4,3,3)=-imapp(5)
        idp3(4,3,4)=-imapp(2)
        idp3(4,3,5)=-imapp(3)
        idp3(4,3,6)=-imapp(5)
c
        do itype=1,6
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
        enddo
      else
        write(*,*)'Unknown process in parsetpvh:',iprdct
        stop
      endif
c
      call parcheckpvh()
      return
      end


      subroutine parcheckpvh()
      implicit none
      integer iallzero,i,jproc,itype,ihpro,i1,i2,i3
      parameter (iallzero=0)
      integer ivbhpro(4,2:3,6)
      common/civbhpro/ivbhpro
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cidpart/idp1,idp2,idp3
c
      call parcheckinvh()
      do jproc=2,3
        do i=1,4
          do itype=1,6
            ihpro=ivbhpro(i,jproc,itype)
            i1=idp1(i,jproc,itype)
            i2=idp2(i,jproc,itype)
            i3=idp3(i,jproc,itype)
            call parcheckfvh(ihpro,i1,i2,i3,iallzero)
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfvh(ihpro,i1,i2,i3,iallzero)
      implicit none
      integer ihpro,i1,i2,i3,iallzero,isum
      real*8 tiny,chin,chout,chall
      parameter (tiny=1.d-8)
      logical ferror
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
      integer iprdct
      common/ciprdct/iprdct
c
      ferror=.false.
      isum=abs(i1)+abs(i2)+abs(i3)
      chin=chrg(i1)+chrg(i2)
      chout=chrg(i3)
      chall=chin-chout-chprdct
      if(iallzero.eq.0)then
c i1,i2,i3 must be either all nonzero, or all zero
        if( ( (i1.ne.0) .and. 
     #        (i2.eq.0.or.i3.eq.0) ) .or.
     #      ( (i2.ne.0) .and. 
     #        (i1.eq.0.or.i3.eq.0) ) .or.
     #      ( (i3.ne.0) .and. 
     #        (i1.eq.0.or.i2.eq.0) ) )ferror=.true.
      elseif(iallzero.eq.1)then
c all process parameters must be different from zero
        if(i1.eq.0.or.i2.eq.0.or.
     #     i3.eq.0.or.ihpro.eq.0)ferror=.true.
      else
        write(*,*)'parcheckfvh called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or.
     #       i3.ne.21 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        (i1+i2).ne.0 ) ) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.le.0 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        i1.ne.i3 ) ) )ferror=.true.
c 403 is qbarq
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. 
     #       i3.ne.21 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        (i1+i2).ne.0 ) ) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.ge.0 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        i1.ne.i3 ) ) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.le.0 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        i2.ne.i3 ) ) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.ge.0 .or. 
     #      ( (iprdct.eq.270.or.iprdct.eq.278) .and.
     #        i2.ne.i3 ) ) )ferror=.true.
c 407 is gg
        if( ihpro.eq.407 )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfvh'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinvh()
      implicit none
      integer i
      real*8 chup,chdn
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
      integer iprdct
      common/ciprdct/iprdct
c
      do i=-5,21
        chrg(i)=1000.d0
      enddo
      chrg(1)=chdn
      chrg(2)=chup
      chrg(3)=chdn
      chrg(4)=chup
      chrg(5)=chdn
      chrg(21)=0.d0
      do i=1,5
        chrg(-i)=-chrg(i)
      enddo
      if(iprdct.eq.270.or.iprdct.eq.278)then
        chprdct=0.d0
      elseif(iprdct.eq.240.or.iprdct.eq.248)then
        chprdct=1.d0
      elseif(iprdct.eq.260.or.iprdct.eq.268)then
        chprdct=-1.d0
      else
        write(*,*)'Unknown process in parcheckinvh:',iprdct
        stop
      endif
      return
      end


      subroutine getnloiproc(iprdct0hw,iv0hw,il0hw)
c Converts the HERWIG process codes for VH production into the codes
c used in the NLO computation. HERWIG conventions are
c     W^+(-->ll) H       -2600-ID      IL=1,2,3   IV=1    ID=0,..,12,99
c     W^+ H              -2600-ID      IL=7       IV=1    ID=0,..,12,99
c     W^-(-->ll) H       -2600-ID      IL=1,2,3   IV=-1   ID=0,..,12,99
c     W^- H              -2600-ID      IL=7       IV=-1   ID=0,..,12,99
c     Z(-->ll)H          -2700-ID      IL=1,..,6  IV=0    ID=0,..,12,99
c     Z H                -2700-ID      IL=7       IV=0    ID=0,..,12,99
c This routine in meant to be called after setting 
c iprdct0hw=mod(-iprdct0hw,10000). Furthermore
c     IL=1,..,6   for Z   ==>  e,nu_e,mu,nu_mu,tau,nu_tau
c     IL=1,2,3    for W   ==>  e,mu,tau
c The NLO conventions are
c     W^+(-->ll) H       240
c     W^+ H              248
c     W^-(-->ll) H       260
c     W^- H              268
c     Z(-->ll)H          270
c     Z H                278
c with
c     IL=1,..,6   ==>  e,mu,tau,nu_e,nu_mu,nu_tau    in all cases
c 
      implicit none
      integer iprdct0hw,iv0hw,il0hw
      integer iprdct,ilepnlo,imap(1:7)
      data imap/1,4,2,5,3,6,7/
      common/ciprdct/iprdct
      common/cilepnlo/ilepnlo
c
      if( iprdct0hw.lt.2600 .or. iprdct0hw.gt.2799 .or.
     #    ( iprdct0hw.ge.2600.and.iprdct0hw.le.2699 .and.
     #      (abs(iv0hw).ne.1.or.il0hw.le.0.or.il0hw.ge.8.or.
     #       (il0hw.ge.4.and.il0hw.le.6) ) .or.
     #    ( iprdct0hw.ge.2700.and.iprdct0hw.le.2799 .and.
     #      iv0hw.ne.0.or.il0hw.le.0.or.il0hw.ge.8 ) ) )then
        write(*,*)'Inconsistent entries in getnloiproc'
        write(*,*)iprdct0hw,iv0hw,il0hw
        stop
      endif
      iprdct=10*(iprdct0hw/100)-10*(1+iv0hw)*iv0hw+
     #       8*(il0hw/7)
      ilepnlo=il0hw
      if(iprdct.eq.270.or.iprdct.eq.278)ilepnlo=imap(il0hw)
      return
      end
c
c
c End initialization
c
c
c
c
c Begin of event file utilities
c
c
C      subroutine whichone(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
c Determines the type of event at random
C      implicit none
C      integer iseed,itot,mx_of_evta,mx_of_evtb,iunit,i0
C      real*8 xpa,xpb,tiny,one,xsum,rnd,fk88random,prob
C      parameter (tiny=1.d-4)
C      logical flag
Cc
C      if(itot.le.0)then
C        write(6,*)'Fatal error #1 in whichone'
C        stop
C      endif
C      xpa=dfloat(mx_of_evta)/dfloat(itot)
C      xpb=dfloat(mx_of_evtb)/dfloat(itot)
C      one=xpa+xpb
C      if(abs(one-1.d0).gt.tiny)then
C        write(6,*)'Error #1 in whichone: probability not normalized'
C        stop
C      endif
C      i0=0
C      flag=.true.
C      xsum=0.d0
C      rnd=fk88random(iseed)
C      do while(flag)
C        if(i0.gt.2)then
C          write(6,*)'Fatal error #2 in whichone'
C          stop
C        endif
C        i0=i0+1
C        prob=xpa
C        if(i0.gt.1)prob=xpb
C        xsum=xsum+prob
C        if(rnd.lt.xsum)then
C          flag=.false.
C          itot=itot-1
C          if(i0.le.1)then
C            mx_of_evta=mx_of_evta-1
C          else
C            mx_of_evtb=mx_of_evtb-1
C          endif
C          iunit=20+i0
C        endif
C      enddo
C      return
C      end


C      subroutine crosscheck(itot,mx_of_evta,mx_of_evtb)
Cc Checks whether whichone did it right
C     implicit none
C      integer itot,mx_of_evta,mx_of_evtb
Cc
C      if(itot.ne.0)then
C        write(6,*)'Error: itot=',itot
C        stop
C      endif
C      if(mx_of_evta.ne.0)then
C        write(6,*)'Error: mx_of_evta=',mx_of_evta
C        stop
C      endif
C      if(mx_of_evtb.ne.0)then
C        write(6,*)'Error: mx_of_evtb=',mx_of_evtb
C        stop
C      endif
C      return
C      end


      subroutine retrieve_evenvh(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evenvh
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6
      common/ci2part/ip4,ip5,ip6
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(7,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      if(idec.eq.0)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5,ip6
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,3),
     #    ((xmom_lb(i,j),j=1,4),i=5,7)
      elseif(idec.eq.1)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,5)
      else
        write(6,*) 'Error in retrieve_evenvh: idec=',idec
        stop
      endif
      goto 999
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(1x,d14.8)
 904  format(28(1x,d14.8))
 997  write(*,*)'unexpected end of file, iunit=',iunit
      stop
 998  write(*,*)'format error'
      write(77,*)'event #:',ii
      write(77,901)i1hpro,iccode,np
      write(77,902)ip1,ip2,ip3,ip4,ip5,ip6
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,7)
      stop
 999  continue
      return
      end


      subroutine store_evenvh(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip6 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_evenvh), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar -> g X
c    402        q g    -> q X
c    403        qbar q -> g X
c    404        qbar g -> qbar X
c    405        g q    -> q X
c    406        g qbar -> qbar X
c    407        g g    -> g X
c X being either the VH or the llbarH system here (thus, 407 is unused).
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6
      common/ci2part/ip4,ip5,ip6
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
c xmom_lb(i,j) is the j component of the four vector of the particle # i,
c given in the laboratory frame. j=4 is the energy for MC@NLO versions
c up to 2.31, the mass for version 3.1 onwards. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is V, 5 is H, 6 and 7 are the outgoing
c fermion and antifermion respectively (leptons from the V decay).
c Momentum conservation is (1+2)-(3+4+5)=0 or (1+2)-(3+5+6+7)=0 
      real*8 xmom_lb(7,4)
      common/cxmomlb/xmom_lb
      integer iwgtnorm
      common/ciwgtnorm/iwgtnorm
      real*8 wgtaev,wgtbev
      common/cwgtev/wgtaev,wgtbev
c
      if(xpmone.eq.-1)then
c Events are already stored in temporary files, and are passed to this
c routines through common blocks filled by retrieve_evenvh
        xevwgt=xevsign
        xfact=1.d0
      elseif(xpmone.eq.1)then
c Events are obtained from SPRING, and are written to temporary files
c for the first time
        if(idec.eq.0)then
          np=6
        elseif(idec.eq.1)then
          np=5
        else
          write(6,*) 'Error in store_evenvh: idec=',idec
          stop
        endif
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev
        else
          xevwgt=xevsign*wgtaev
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_evenvh: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      if(idec.eq.0)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5,ip6
        write(iunit,903)xevwgt
        write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,3),
     #                  ((xmom_lb(i,j),j=1,4),i=5,7)
      elseif(idec.eq.1)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5
        write(iunit,903)xevwgt
        write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,5)
      endif
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(1x,d14.8)
 904  format(28(1x,d14.8))
      return
      end
c
c
c End of event file utilities
c
c
c
c
c Begin of MC subtraction terms
c
c
      subroutine xmcsuvh(jproc,idec,xm12,xm22,s,x,y,cth1,cth2,
     #                   phil,cthl,x1,x2,ycnt,vecout,axlout)
      implicit none
      real*8 xm12,xm22,s,x,y,cth1,cth2,phil,cthl,x1,x2,ycnt,
     # vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc,idec
      real*8 xii,tiny,vcf,vtf,xmin,xlim,xlim1,xlim2,al_gfun,be_gfun,
     # ccc_gfun,yd,ydead_mod,xfact,z,zherw_spl,xi,xiherw_spl,
     # ap,ap_kern,xjac_xizspl,zherw_smn,xiherw_smn,xjac_xizsmn,
     # tt,gfact,gfun,sbar,tbar,v1abar,v1bbar,v1cbar,xsoft,
     # vecin(1:4,1:3),axlin(1:4,1:3)
      integer icode,itwo,i,j,ia1ora2
      parameter (tiny=1.d-6)
      parameter (vcf=4/3.d0)
      parameter (vtf=1/2.d0)
      parameter (xmin=0.69519410160110384d0)
      parameter (itwo=2)
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
      common/cia1ora2/ia1ora2
c
      if(abs(ycnt).ne.1.d0) then
        write(6,*)'xmcsuvh called improperly: ycnt=',ycnt
        stop
      endif
      xii=1-x
      xlim=0.d0
      xlim1=0.d0
      xlim2=0.d0
      call getmcinvll(xm12,xm22,s,x,y,cth1,cth2,phil,cthl,x1,x2,
     #                sbar,tbar,v1abar,v1bbar,v1cbar,ia1ora2)
      if(idec.eq.0)then
        call fborn_dec(sbar,tbar,v1abar,v1bbar,itwo,vecin,axlin)
      elseif(idec.eq.1)then
        call fborn_und(sbar,tbar,itwo,vecin,axlin)
      else
        write(6,*) 'Error in xmcsuvh: idec=',idec
        stop
      endif
      if(jproc.eq.2)then
        icode=4
        xsoft=1.d0-be_gfun+xmin*be_gfun
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(x.gt.xsoft)then
            xlim2=ap_kern(x,icode)*(1+y)
          endif
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1+y)/(3+y)-
     #              2*(1-x)*(1+y)*(7+6*y+3*y**2)/(3+y)**2
              xlim1=vcf*xlim1
            elseif(1-y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1-y)
              xlim1=vcf*xlim1
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim1=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
        elseif(ycnt.eq.-1.d0)then
          if(x.gt.xsoft)then
            xlim2=ap_kern(x,icode)*(1-y)
          endif
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1-y)/(3-y)-
     #              2*(1-x)*(1-y)*(7-6*y+3*y**2)/(3-y)**2
              xlim1=vcf*xlim1
            elseif(1+y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1+y)
              xlim1=vcf*xlim1
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim1=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
        endif
        tt=(1.d0-x)/(1.d0-xsoft)
        gfact=gfun(tt)
        xlim=4*(xlim1*gfact+xlim2*(1.d0-gfact))/s
        do i=1,4
          do j=1,3
            vecout(i,j)=xlim*vecin(i,j)
            axlout(i,j)=xlim*axlin(i,j)
          enddo
        enddo
      elseif(jproc.eq.3)then
        do i=1,4
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
        enddo
        icode=2
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1+y)
              xlim=vtf*xlim
            elseif(1-y.lt.tiny)then
              xlim=2-6*x+8*x**2-4*x**3-
     #             (4*x**5-19*x**4+31*x**3-23*x**2+8*x-1)*(1-y)/x
              xlim=vtf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do i=3,4
            do j=1,3
              vecout(i,j)=xlim*vecin(i,j)
              axlout(i,j)=xlim*axlin(i,j)
            enddo
          enddo
        elseif(ycnt.eq.-1.d0)then
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1-y)
              xlim=vtf*xlim
            elseif(1+y.lt.tiny)then
              xlim=2-6*x+8*x**2-4*x**3-
     #             (4*x**5-19*x**4+31*x**3-23*x**2+8*x-1)*(1+y)/x
              xlim=vtf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do i=1,2
            do j=1,3
              vecout(i,j)=xlim*vecin(i,j)
              axlout(i,j)=xlim*axlin(i,j)
            enddo
          enddo
        endif
      else
        write(*,*)'xmcsuvh: unknown process',jproc
        stop
      endif
      return
      end


      subroutine getmcinvll(xm12,xm22,s,x,y,cth1,cth2,phil,cthl,x1,x2,
     #                      sbar,tbar,v1abar,v1bbar,v1cbar,iinv)
c Returns the invariants of the reduced hard process, consistently with
c the boosts performed by Herwig; x1 and x2 are the Bjorken x's relevant 
c to the 2-->4 hard process. Refer to the comment at the beginning of
c invar for the definition of the kinematics. Call with
c    iinv=1  -->  exact (2->4) to (2->3) relations
c    iinv=2  -->  collinear approximation
c WARNING: the invariants returned by invar are defined with the
c conventions of FNR, NPB383(92)3. Thus, for consistency, we define
c           tbar = (p1-k1)^2 == -2p1bar.k1bar+k1bar^2
c whereas the analytical results are relevant to the dot product only
      implicit none
      real*8 xm12,xm22,s,x,y,cth1,cth2,phil,cthl,x1,x2,sbar,tbar,
     # v1abar,v1bbar,v1cbar
      integer iinv
      real*8 tk,uk,q1q,q2q,q1c,q2c,v1a,v1b,v1c,xmn,xpl,galonred,
     # betalon,dm12,one
      parameter (one=1.d0)
      integer idec
      parameter (idec=0)
      character * 2 str
      parameter (str='p1')
c
      if(iinv.eq.1)then
        call invvh(xm12,xm22,s,x,y,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        q1c=xm12+xm22-s-tk-q1q
        q2c=xm12+xm22-s-uk-q2q
        sbar=s*x
        xmn=((s+uk)*x1/s-(s+tk)*x2/s)/2.d0
        xpl=((s+uk)*x1/s+(s+tk)*x2/s)/2.d0
        galonred=sqrt(xpl**2-x1*x2*tk*uk/s**2)
        betalon=-xmn/galonred
        dm12=xm12-xm22
        tbar=-sbar/2.d0*( 1-(x2*(q1q-q1c-dm12)+x1*(q2q-q2c+dm12))/
     #                      (2*s*galonred) )
     #       -dm12/2.d0*(1-betalon) + xm12
        v1abar=-1/2.d0*(1-betalon)*(v1c+xm12)+
     #         sbar*(x2*v1a-x1*v1b)/(2*s*galonred)
        v1bbar=-v1abar-v1c-xm12
        v1cbar=v1c
      elseif(iinv.eq.2)then
c Use here the y=1 kinematics to compute the reduced invariants. The y=-1
c kinematics could also be used, with tbar=q1q, v1abar=v1a, v1bbar=v1b*x
        call invvh(xm12,xm22,s,x,one,cth1,cth2,phil,cthl,str,
     #             tk,uk,q1q,q2q,v1a,v1b,v1c,idec)
        sbar=s*x
        tbar=q2q
        v1abar=v1a*x
        v1bbar=v1b
        v1cbar=v1c
      else
        write(*,*)'Error in getmcinvll: iinv=',iinv
        stop
      endif 
      return
      end


C      function ffunction5(xx,yy)
C      implicit real * 8 (a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      tmp=0.d0
C      yd=ydead(x)
C      if(abs(y).ge.yd)tmp=1.d0
C      ffunction5=tmp
C      return
C      end


C      function ydead(x)
C      implicit real*8(a-h,o-z)
C      parameter (xmin=0.69519410160110384d0)
Cc
C      tmp=0.d0
C      if(x.lt.xmin)tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
C      ydead=tmp
C      return
C      end


C      function ydead_mod(x)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      if(1-x.lt.tiny)then
C        tmp=-1/3.d0-28.d0*(X-1)/27.d0
C      else
C        tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
C      endif
C      ydead_mod=tmp
C      return
C      end


C      function gfun(xx)
C      implicit real*8(a-h,o-z)
C      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
Cc
C      x=xx
C      tmp=1.d0
C      if(x.lt.0.d0)then
C        write(6,*)'Fatal error in gfun'
C        stop
C      endif
C      if(x.le.1.d0.and.al_gfun.gt.0.d0)
C     #  tmp=x**(2*al_gfun)/(x**(2*al_gfun)+(1-x)**(2*al_gfun))
C      gfun=tmp
C      return
C      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invariants. The functions
c for xi, z, and the jacobian have been checked numerically to coincide with
c those of the hvq package (except when closer than tiny to IR limits, since
c here more terms are kept -- which are numerically irrelevant). The present
c for is simpler and faster to computed, thanks to the unique choice of E0
c
c
C      function zherw_spl(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=1.d0+(Y+3)*(X-1)/4.d0
C      elseif(1-y.lt.tiny)then
C        tmp=X-(X**2-3*X+2)*(Y-1)/4.d0
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-t/xmv2
C        xb=u*t/xmv2
C        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
C      endif
C      zherw_spl=tmp
C      return
C      end


C      function zherw_smn(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=1-(Y-3)*(X-1)/4.d0
C      elseif(1+y.lt.tiny)then
C        tmp=X+(X**2-3*X+2)*(Y+1)/4.d0
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-u/xmv2
C        xb=u*t/xmv2
C        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
C      endif
C      zherw_smn=tmp
C      return
C      end


C      function xiherw_spl(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=-(2*Y-2)/(Y+3)-(2*Y**3+2*Y**2-2*Y-2)*(X-1)/(Y**2+6*Y+9)
C      elseif(1-y.lt.tiny)then
C        tmp=-X*(Y-1)/2
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-t/xmv2
C        z1=zherw_spl(x,y)
C        tmp=xa*z1**2/(1-z1)
C      endif
C      xiherw_spl=tmp
C      return
C      end


C      function xiherw_smn(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=-(2*Y+2)/(Y-3)+(2*Y**3-2*Y**2-2*Y+2)*(X-1)/(Y**2-6*Y+9)
C      elseif(1+y.lt.tiny)then
C        tmp=X*(Y+1)/2
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-u/xmv2
C        z1=zherw_smn(x,y)
C        tmp=xa*z1**2/(1-z1)
C      endif
C      xiherw_smn=tmp
C      return
C      end


C      function xjac_xizspl(xx,yy)
C      implicit none
C      real*8 xjac_xizspl,x,y,xx,yy
C      real*8 z,xi,zherw_spl,xiherw_spl,tmp,tiny
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=2/(Y+3)+(2*Y**2+4*Y+2)*(X-1)/(Y**2+6*Y+9)
C      elseif(1-y.lt.tiny)then
C        tmp=X/2-(3*X**2-8*X+6)*(Y-1)/8
C      else
C        z=zherw_spl(x,y)
C        xi=xiherw_spl(x,y)
C        tmp=(X-1)**2*(Y-1)*(X*Y-Y+X+1)*Z**5/
C     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
C      endif
C      xjac_xizspl=tmp
C      return
C      end


C      function xjac_xizsmn(xx,yy)
C      implicit none
C      real*8 xjac_xizsmn,x,y,xx,yy
C      real*8 z,xi,zherw_smn,xiherw_smn,tmp,tiny
C      parameter (tiny=1.d-4)
c
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=2/(Y-3)-(2*Y**2-4*Y+2)*(X-1)/(Y**2-6*Y+9)
C      elseif(1+y.lt.tiny)then
C        tmp=-X/2-(3*X**2-8*X+6)*(Y+1)/8
C      else
C        z=zherw_smn(x,y)
C        xi=xiherw_smn(x,y)
C        tmp=-(X-1)**2*(Y+1)*(X*Y-Y-X-1)*Z**5/
C     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
C      endif
C      xjac_xizsmn=-tmp
C      return
C      end
c
c
c End of utility routines for xi, z, and 2-->2 invariants
c
c
c
c
c Begin of utility routines for Bjorken x's
c
c
C      function x1soft(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x1soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
C        rx=sqrt(x)
C        tmp=x1*csi*rx
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x1*(1-(1-x)*(1+y)/2.d0)
C        elseif(1-y.lt.tiny)then
C          tmp=x*x1*(1+(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
C        elseif(1+y.lt.tiny)then
C          tmp=x1*(1-(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=0.5d0*(sqrt(xb**2+4*xa)-xb)
C        endif
C      else
C        write(*,*)'Error in x1soft',iprespl
C        stop
C      endif
C      x1soft=tmp
C      return
C      end


C     function x2soft(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x2soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
C        rx=sqrt(x)
C        tmp=x2*rx/csi
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x2*(1-(1-x)*(1-y)/2.d0)
C        elseif(1-y.lt.tiny)then
C          tmp=x2*(1-(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
C        elseif(1+y.lt.tiny)then
C          tmp=x*x2*(1+(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=0.5d0*(sqrt(xb**2+4*xa)+xb)
C        endif
C      else
C        write(*,*)'Error in x2soft',iprespl
C        stop
C      endif
C      x2soft=tmp
C      return
C      end


C      function x1x2jac(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x1x2jac,xx1,xx2,xx,yy,tiny,x1,x2,x,y,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        tmp=x
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x+(1-x)**2*(1-y**2)*x1*x2/(2.d0*(x1+x2)**2)
C        elseif(1-y.lt.tiny)then
C          tmp=x+(1-x)**2*x*(1-y)*x1*x2/(x*x1+x2)**2
C        elseif(1+y.lt.tiny)then
C          tmp=x+(1-x)**2*x*(1+y)*x1*x2/(x1+x*x2)**2
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=x*((1-y+x*(1+y))*x1+(1+y+x*(1-y))*x2)/
C     #        (2.d0*sqrt(xb**2+4*xa))
C        endif
C      else
C        write(*,*)'Error in x1x2jac',iprespl
C        stop
C      endif
C      x1x2jac=abs(tmp)
C      return
C      end
c
c
c Running couplings
c
c
      function zgmu2_nvh()
c Sets the scales for NLO subtraction terms
      implicit none
      real * 8 zgmu2_nvh
      real * 8 pi,ptv1,ptv2,ptvg,ptvl1,ptvl2,ptvh2,ptvh
      common/perpen/ptv1(2),ptv2(2),ptvg(2),ptvl1(2),ptvl2(2)
      parameter (pi=3.14159265358979312D0)
      include 'vhgcblks.h'
      real * 8 xmu2,as,alfas
      integer inloscale
      common/cinloscale/inloscale
c
      ptvh2 = ptvg(1)**2 + ptvg(2)**2
      if(inloscale.eq.1)then
        xmu2 = ptvh2 + xmi2
      elseif(inloscale.eq.2)then
        xmu2 = xmi2
      elseif(inloscale.eq.3)then
        xmu2 = max(xmi2,ptvh2)
      elseif(inloscale.eq.4)then
        ptvh = sqrt(ptvh2)
        xmu2 = ( max(sqrt(xmi2)-ptvh,ptvh) )**2
      else
        write(*,*)'Unknown option in zgmu2_nvh',inloscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nl)
      zgmu2_nvh = 4.d0*pi*as
      zg = sqrt(zgmu2_nvh)
      end


      function zgmu2_vh()
c Sets the scales for MC subtraction terms
      implicit none
      real * 8 zgmu2_vh
      real * 8 pi,ptv1,ptv2,ptvg,ptvl1,ptvl2,ptvh2,ptvh
      common/perpen/ptv1(2),ptv2(2),ptvg(2),ptvl1(2),ptvl2(2)
      parameter (pi=3.14159265358979312D0)
      include 'vhgcblks.h'
      real * 8 xmu2,as,alfas
      integer imcscale
      common/cimcscale/imcscale
c
      ptvh2 = ptvg(1)**2 + ptvg(2)**2
      if(imcscale.eq.1)then
        xmu2 = ptvh2 + xmi2
      elseif(imcscale.eq.2)then
        xmu2 = xmi2
      elseif(imcscale.eq.3)then
        xmu2 = max(xmi2,ptvh2)
      elseif(imcscale.eq.4)then
        ptvh = sqrt(ptvh2)
        xmu2 = ( max(sqrt(xmi2)-ptvh,ptvh) )**2
      elseif(imcscale.eq.5)then
        ptvh = sqrt(ptvh2)
        xmu2 = ( sqrt(xmi2)-ptvh/2.d0 )**2
      else
        write(*,*)'Unknown option in zgmu2_vh',imcscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nl)
      zgmu2_vh = 4.d0*pi*as
      zg = sqrt(zgmu2_vh)
      end


Cc-------------------------------------------------------------------------
C      function xalfaem(q2)
Cc Alpha_em(MSbar) at the scale q2 = q^2. 
Cc Uses alpha_Thomson below the electron mass, an interpolation between
Cc m_e and m_tau, and the evolution equation above m_b. This function is
Cc taken from the gamma*gamma* --> hadrons package
Cc-------------------------------------------------------------------------
C      implicit real*8 (a-z)
C      integer npoints,ideg
C      parameter (npoints=3,ideg=3)
C      real*4 ooa(npoints),xlogmu(npoints),divdif
Cc 1/alpha_em at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
C      data ooa     / 137.036, 135.95, 133.513 /
Cc logs of sqrt(q2) at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
C      data xlogmu  / -7.57914, -2.2481, 0.574927 /
C      data zm/91.2d0/,ooaz/127.934d0/,pi/3.1415927d0/,nc/3/
Cc
C      if(q2.lt.exp(2.*xlogmu(1))) then
C         xalfaem = 1.d0/ooa(1)	 
C      elseif(q2.lt.exp(2.*xlogmu(3))) then
C         xlogq = log(q2)/2.d0
C         xalfaem = 1.d0/divdif(ooa,xlogmu,npoints,sngl(xlogq),ideg)
C      elseif(q2.lt.5.**2) then
C         b = 3 + 2*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
C         xlq = log(q2) - 2.*xlogmu(3)
C         xalfaem = 1d0/ooa(3)/(1.d0 - 1.d0/3.d0/pi/ooa(3)*b*xlq)
C      else
C         b = 3 + 3*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
C         xlq = log(q2/zm**2)
C         xalfaem = 1d0/ooaz/(1.d0 - 1.d0/3.d0/pi/ooaz*b*xlq)
C      endif
C      return
C      end
