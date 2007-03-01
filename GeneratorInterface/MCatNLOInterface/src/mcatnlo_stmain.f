      SUBROUTINE STMAIN
      implicit none
      include 'stpcblks.h'
      real * 8 value(20),xmass(-5:21),xmomshifts(4),vickm(1:6,1:6)
      real * 8 pi,xicut,deltai,deltao,xicutss,deltas,deltac,
     #  etacut,tmas,xpdflam4,xpdflam5,tmp,ecm,xfh,xren,ac1,ac2,
     #  rohlim,xtotal,ytotal,dtot,av3ai,d3ai,av3negai,d3negai,
     #  av3ao,d3ao,av3negao,d3negao,ctime,avtot,dummy,powmeas,
     #  xares,yares,xbres,ybres,xmone,wgtaev,wgtbev,xrenmc,xfhmc,
     #  zero,alsfi,besfi,alcli,becli,alsfo,besfo,alclo,beclo,av3bi,
     #  av3bo,d3bi,d3bo,av3negbi,av3negbo,d3negbi,d3negbo,evfrac,
     #  evprcfrac,zmw,al_ispcfun,be_ispcfun,al_ospcfun,be_ospcfun,
     #  xmw,gaw,xmt,twidth,xm012,ga1,bw1delf,bw1fmmn,xm1low2,xm1upp2,
     #  brrtop1,xlep1mass,xlep2mass,gammax1,xm1low,xm1upp,bw1mdpl,
     #  bw1mdmn,bw1fmpl
      integer ih1,ih2,ndns1,ndns2,iinput,iprespl,ifxdaem,isubttype,
     #  nsamp,iinout,loproc,maproc,ifk88istrl,ifk88ih,ifk88ndns,
     #  ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,iverbose,ichkmom,
     #  ichkpid,ibswrite,itmpih,itmpndns,idpdfset,iprdct0hw,
     #  iseed0,iproc,it1,it2,iseld,ncl3,ifuntype,ndim,nwild,itd1,itd2,
     #  ibscall,loproco,maproco,iwgtnorm,inloscale,imcscale,ia1ora2,
     #  iwrong,iwrong1,ionshell,ifk88seed,i,j,maxevt,iseed,maxtrials,
     #  mx_of_evtai,mx_of_evtao,mx_of_evtbi,mx_of_evtbo,ntotal,
     #  ndiff,nevts,ntrls,iunitai,iunitbi,iunitao,iunitbo,ioutput,
     #  itot,ii,iunit,izero,ione,idec,iwidth,il1hw,neventsuw,
     #  nqeventsuw,ifailuw,ncntuws,nqcntuws,nmaxuw,nqmaxuw,ideconsh
      character * 2 scheme
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnameai,fnameao,fnamebi,fnamebo
      character * 80 fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical evgen
      external sig5azw_in,sig5bzw_in
      external sig5azw_out,sig5bzw_out
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (zero=0.d0)
      parameter (izero=0)
      parameter (ione=1)
c
c common /strfst0/ is only in strfst:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfst0/ih1,ih2,ndns1,ndns2
c W mass entered by the user
      common/czmw/zmw
c quark and gluon masses, used by Herwig. PDF labeling convention
      common/parmass/xmass
c CKM matrix elements entered by the user
      common/cvickm/vickm
c alsfi and besfi are the parameters entering gfunsost (initial state)
      common/cgfunsfi/alsfi,besfi
c alsfo and besfo are the parameters entering gfunsost (final state)
      common/cgfunsfo/alsfo,besfo
c alcli and becli are the parameters entering gfuncost (initial state)
      common/cgfuncli/alcli,becli
c alclo and beclo are the parameters entering gfuncost (final state)
      common/cgfunclo/alclo,beclo
c al_ispcfun, be_ispcfun are the parameters entering spcdamp_in
      common/cspcipar/al_ispcfun,be_ispcfun
c al_ospcfun, be_ospcfun are the parameters entering spcdamp_out
      common/cspcopar/al_ospcfun,be_ospcfun
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
c ichkpid=0 --> enables checks on parton identities
      common/cichkpid/ichkpid
c----------------------------------------------------------
c Variables that control the integrations
c
      common/cisubttype/isubttype
      common/parsub/xicut,deltai,deltao
      common/xisave/xicutss
      common/pmerge/deltas,deltac
      common/samp/nsamp
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
c powmeas controls damping speed of the measurement functions
      common/cpowm/powmeas
c----------------------------------------------------------
c Top decay variables
c Decay of the top: idec=0    -->   top decay
c                   idec=1    -->   top doesn't decay
      common/cidec/idec
c Mass ranges: iwidth=0    -->   W on shell
c              iwidth=1    -->   W off shell
      common/ciwidth/iwidth
c Type of W decay; il1hw is entered following HERWIG conventions:
c     IL=1,2,3    for W   ==>  e,mu,tau
      common/cilhw/il1hw
c W mass and width (W mass squared is in cmass)
      common/cwparam/xmw,gaw
c top mass and width; top mass and its square are also stored in cmass 
      common/ctparam/xmt,twidth
c W mass ranges
      common/cbw1/xm012,ga1,bw1delf,bw1fmmn
      common/bounds/xm1low2,xm1upp2
c top branching ratio
      common/brratios/brrtop1
c mass of leptons from W decay
      common/clepmass/xlep1mass,xlep2mass
c----------------------------------------------------------
c inloscale controls the reference scale in the NLO computation
      common/cinloscale/inloscale
c imcscale controls the reference scale in the MC subtraction terms
      common/cimcscale/imcscale
c----------------------------------------------------------
c The following refer to the computation of MC subtraction terms
c ia1ora2=1 -> full invariants, ia1ora2=2 -> simplified invariants
      common/cia1ora2/ia1ora2
c----------------------------------------------------------
c Subprocesses: 'qq', 'qg', corresponding to jproc=jproc0=2,3
c In the integration routines, loproc<=jproc<=maproc. In the case
c of OUT contribution, we use loproco and maproco, in such a way
c as to exclude qg contributions
      common/cwchproc/loproc,maproc
      common/cwotproc/loproco,maproco
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c Common blocks for statistics relevant to secondary unweighting
      common/c1iunwgt/neventsuw,nqeventsuw,ifailuw
      common/c2iunwgt/ncntuws,nqcntuws,nmaxuw,nqmaxuw
c Average shifts in momenta, due to quark and lepton masses
      common/cshifts/xmomshifts
c----------------------------------------------------------
c ifuntype=1 for sig5a[in], ifuntype=2 for sig5b[in]
c ifuntype=3 for sig5a[out], ifuntype=4 for sig5b[out]
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c Flag to put top decay products on shell
      common/cideconsh/ideconsh
c----------------------------------------------------------
c Common blocks for general MC@NLO routines
c common block for internal rnd number generation, independent of bases
      common/cifk88seed/ifk88seed
c common block fk88ipdfs is filled by our interface to MLMPDF
      common/fk88ipdfs/ifk88ih,ifk88ndns
c common block w50511 and w50512 are filled by PDFLIB 
      common/w50511/ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,tmas
      common/w50512/xpdflam4,xpdflam5
C

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      character*100 stfilename
      common/fstbases/ basesoutput, stfilename
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
     % mmaemrun,mmiproc,mmgname,mmscheme,
     % mmpart1,mmpart2

      integer mmdecide
      common/decpar/mmdecide
CCC =================================================================


C------------------------------------------------------------------------
C                             START                                     -
C------------------------------------------------------------------------
c iinput=1 ==> all inputs are given by the user
      iinput=2
      mmdecide=7
c iverbose=1 ==> writes more on standard output
      iverbose=0
c ichkmom=0 ==> enables checks on kinematics
      ichkmom=1
c ichkpid=0 ==> enables checks on parton identities
      ichkpid=1
c initialization of internal randon number generation
      ifk88seed=1
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
      open(unit=11,file='stplog',status=newver)
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
     # 'Enter Ecm(GeV),fren[NLO],ffact[NLO],fren[MC],ffact[MC]'
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
      write(*,*)'Enter -(1)2000-IT for all channels'
      write(*,*)'      -(1)2010-IT for s-channel'
      write(*,*)'      -(1)2020-IT for t-channel'
      write(*,*)'  with IT=0 for t+tbar production'
      write(*,*)'       IT=1 for tbar production only'
      write(*,*)'       IT=4 for t production only'
         read(*,*) iprdct0hw
      else
         iprdct0hw=mmiproc
      endif
      write(11,'(1x,i6,27x,a)') iprdct0hw,
     #  '! -2000/10/20=all, s-ch, t-ch'
      iprdct0hw=mod(-iprdct0hw,10000)
      call getnloiprst(iprdct0hw)
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter top mass and width (GeV)'
         read(*,*)xm1,twidth
      else
         xm1=mmxmt
         twidth=mmtwidth
      endif
      write(11,'(2(1x,d10.4),12x,a)') xm1,twidth,'! M_top, Gamma_top'
      xm12 = xm1**2
      xmt = xm1
c Mass of the light parton recoiling against the top at LO
      xm2  = 0.d0
      xm22 = 0.d0
c Mass of the W, and its width (enters propagators, and top decay)
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter W mass and width (GeV)'
         read(*,*)zmw,gaw
      else
         zmw=mmxwm
         gaw=mmxww
      endif
      write(11,'(2(1x,d10.4),12x,a)') zmw,gaw,'! M_W, Gamma_W'
c Top decay parameters
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter IL=1,2,3 following HERWIG conventions'
      write(*,*)'               for t->W(->l_IL nu_IL) b'
      write(*,*)'      IL=7 for undecayed top'
         read(*,*) il1hw
      else
         il1hw=mmil1code
      endif
      write(11,'(1x,i2,31x,a)') il1hw,
     #  '! 1,2,3 -> t dec, 7 -> t undec'
      if(il1hw.eq.7)then
        idec=1
      elseif(il1hw.ge.1.and.il1hw.le.3)then
        idec=0
      else
        write(*,*) 'Unknown decay option:',il1hw
        stop
      endif
      if(idec.eq.0)then
        if(iinput.lt.2) then           
        write(*,*)' '
        write(*,*)'Enter GammaX, M_V1(min), M_V1(max) for W'
        write(*,*)
     #   '  If GammaX>0, the boson mass is chosen in the range'
        write(*,*)'      M0-GammaX*width < M_W < M0+GammaX*width'
        write(*,*)'  and M_V1(min), M_V1(max) are ignored'
        write(*,*)
     #   '  If GammaX<0, the boson mass is chosen in the range'
        write(*,*)'            M_V1(min) < M_W < M_V1(max)'
        write(*,*)
     #'  If GammaX=0, the boson mass is set equal to the pole mass'
           read(*,*)gammax1,xm1low,xm1upp
        else
           gammax1=mmv1gammax
           xm1low=mmv1massinf
           xm1upp=mmv1masssup
        endif
        write(11,'(3(1x,d10.4),1x,a)') gammax1,xm1low,xm1upp,
     #   '! GammaX, M_V1(min), M_V1(max)'
        if(gammax1.lt.0.and.xm1low.ge.xm1upp)then
          write(*,*)'Enter a non-zero range'
          stop
        endif
      else
        twidth=0.d0
        gaw=0.d0
      endif
c CKM matrix elements
      do i=1,6
        do j=1,6
          vickm(i,j)=0.d0
        enddo
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter |V_ud|, |V_us|, |V_ub|'
      write(*,*)' all equal to zero to use PDG values'
         read(*,*)vickm(1,2),vickm(1,3),vickm(1,5)
      else
         vickm(1,2)=mmvud
         vickm(1,3)=mmvus
         vickm(1,5)=mmvub
      endif
      write(11,'(3(1x,d10.4),1x,a)')vickm(1,2),vickm(1,3),vickm(1,5),
     #      '! |V_ud|,|V_us|,|V_ub|'
      if(iinput.lt.2) then
      write(*,*)'Enter |V_cd|, |V_cs|, |V_cb|'
         read(*,*)vickm(4,2),vickm(4,3),vickm(4,5)
      else
         vickm(4,2)=mmvcd
         vickm(4,3)=mmvcs
         vickm(4,5)=mmvcb
      endif
      write(11,'(3(1x,d10.4),1x,a)')vickm(4,2),vickm(4,3),vickm(4,5),
     #      '! |V_cd|,|V_cs|,|V_cb|'
      if(iinput.lt.2) then
      write(*,*)'Enter |V_td|, |V_ts|, |V_tb|'
         read(*,*)vickm(6,2),vickm(6,3),vickm(6,5)
      else
         vickm(6,2)=mmvtd
         vickm(6,3)=mmvts
         vickm(6,5)=mmvtb
      endif
      write(11,'(3(1x,d10.4),1x,a)')vickm(6,2),vickm(6,3),vickm(6,5),
     #      '! |V_td|,|V_ts|,|V_tb|'
c Set constants
      call setpst()
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
c PDFs
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
         write(*,*) gname
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
        write(*,*)
     #   'Enter alpha and beta for G_soft (initial state)'
        write(*,*)' Defaults are: alpha=1, beta=-0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<|beta|<=1'
        read(*,*) alsfi,besfi
        write(11,'(2(1x,d10.4),12x,a)') alsfi,besfi,
     #    '! alpha, beta [soft,in]'
      else
        alsfi=1.d0
        besfi=-0.5d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     #   'Enter alpha and beta for G_coll (initial state)'
        write(*,*)' Defaults are: alpha=1, beta=-0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<|beta|<=1'
        read(*,*) alcli,becli
        write(11,'(2(1x,d10.4),12x,a)') alcli,becli,
     #    '! alpha, beta [coll,in]'
      else
        alcli=1.d0
        becli=-0.5d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     #   'Enter alpha and beta for G_soft (final state)'
        write(*,*)' Defaults are: alpha=1, beta=-0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<|beta|<=1'
        read(*,*) alsfo,besfo
        write(11,'(2(1x,d10.4),12x,a)') alsfo,besfo,
     #    '! alpha, beta [soft,out]'
      else
        alsfo=1.d0
        besfo=-0.1d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     #   'Enter alpha and beta for G_coll (final state)'
        write(*,*)' Defaults are: alpha=1, beta=-0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<|beta|<=1'
        read(*,*) alclo,beclo
        write(11,'(2(1x,d10.4),12x,a)') alclo,beclo,
     #    '! alpha, beta [coll,out]'
      else
        alclo=1.d0
        beclo=-0.1d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha and beta for the function SPC_damp[in]'
        write(*,*)' Defaults are: alpha=1, beta=0.5'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_ispcfun,be_ispcfun
        write(11,'(2(1x,d10.4),12x,a)') al_ispcfun,be_ispcfun,
     #    '! alpha, beta (spin corr, in)'
      else
        al_ispcfun=1.d0
        be_ispcfun=0.5d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha and beta for the function SPC_damp[out]'
        write(*,*)' Defaults are: alpha=1, beta=0.5'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_ospcfun,be_ospcfun
        write(11,'(2(1x,d10.4),12x,a)') al_ospcfun,be_ospcfun,
     #    '! alpha, beta (spin corr, out)'
      else
        al_ospcfun=1.d0
        be_ospcfun=0.5d0
      endif
c-----------------------------------------------------------------
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
        write(*,*)'Enter exponent of measurement functions (default 4)'
        read(*,*)powmeas
        write(11,'(1x,d10.4,23x,a)') powmeas,'! powmeas'
      else
        powmeas=2.d0
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
        write(*,*)'Enter 1 to set mu_0^2=M_t^2+(Sum_i pt_i)/2'
        write(*,*)'      2 to set mu_0=M_t'
        write(*,*)' The default is 1'
        read(*,*) inloscale
        write(11,'(1(1x,i8),25x,a)') inloscale,
     #    '! 1->mu_0=mt+<pt>, 2->mu_0=mt'
      else
        inloscale=1
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'For the computation of alpha_S in MC terms'
        write(*,*)'Enter 1 to set mu_0^2=M_t^2+(Sum_i pt_i)/2'
        write(*,*)'      2 to set mu_0=M_t'
        write(*,*)' The default is 1'
        read(*,*) imcscale
        write(11,'(1(1x,i8),25x,a)') imcscale,
     #    '! 1->mu_0=mt+<pt>, 2->mu_0=mt'
      else
        imcscale=1
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
c iseed0 is the seed for the integration step
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
          write(*,*)'enter xicut, deltaI, deltaO '
          read(*,*)xicut,deltai,deltao
          write(11,'(3(1x,d10.4),1x,a)') xicut,deltai,deltao,
     #      '! xicut,deltaI,deltaO'
        else
          write(*,*)' '
          write(*,*)'Enter zi ( [ 2*kt(gluon)/sqrt(shat) ]^2 < zi )'
          read(*,*) etacut
          write(11,'(1x,d10.4,23x,a)') etacut,'! zi'
          xicut = 1.d0
c Set deltaI=1 as prescribed by the zeta subtraction (see first MC@NLO paper)
          deltai = 1.d0
          deltao = 1.d0
        endif
        xicutss = xicut
        deltas = 0
        deltac = 0
      else
        if(isubttype.eq.0)then
          xicut = 0.4d0
          deltai = 1.7d0
          deltao = 0.2d0
          xicutss = xicut
          deltas = 0
          deltac = 0
        else
          etacut = 1.d0
          deltai = 1.d0
          deltao = 1.d0
        endif
      endif
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
      call parsetpst()
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
      if(iproc.ne.0.and.iproc.ne.2.and.iproc.ne.3) then
        write(*,*)'Option not implemented: iproc=',iproc
        stop
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
        write(*,*)'Enter 0 to leave the top decay products massless'
        write(*,*)'      2 to put them on their mass shell'
        read(*,*) ideconsh
        write(11,'(1x,i1,32x,a)') 
     #      ideconsh,'! 0=massless, 2=massive decay products'
      else
        ideconsh=2
      endif
      if(ideconsh.ne.0.and.ideconsh.ne.2) then
        write(*,*) 'Error: enter 0 or 2.'
        stop
      endif
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 1 for in+out'
        write(*,*) '      2 for in only'
        write(*,*) '      3 for out only'
        read(*,*) iinout
        write(11,'(1x,i2,31x,a)') iinout,'! 1=in+out,2=in,3=out'
      else
        iinout=1
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
      if(ncl3.lt.0)ncl3=120000
c---- close logfile
      close(11)
c----------------------------------------------------------------
c  *********************  START INTEGRATION *********************
c----------------------------------------------------------------
      ifuntype=0
      loproc = 2
      maproc = 3
      loproco = 2
      maproco = 2
      if(iproc.ne.0) then
        loproc=iproc
        maproc=iproc
      endif
      if(iproc.eq.3) then
        loproco=0
        maproco=-1
      endif
c When top decay, compute the relevant parameters
      if(idec.eq.0)then
        if(gammax1.ne.0.d0.and.gaw.eq.0.d0)then
          write(*,*)'Non-zero mass range require non-zero width'
          stop
        endif
        xm012=xmw2
        if(gammax1.eq.0)then
          iwidth=0
          xm1low2=-1.d0
          xm1upp2=-1.d0
          bw1delf=0.d0
        elseif(gammax1.ne.0)then
          iwidth=1
          ga1=gaw
          if(gammax1.ge.0)then
            xm1low2=(max( 1.d-1,xlep1mass+xlep2mass,
     #                    xmw-gammax1*ga1 ))**2
            xm1upp2=(min(xmt-1.d-1,xmw+gammax1*ga1))**2
          else
            xm1low2=(max(1.d-1,xlep1mass+xlep2mass,xm1low))**2
            xm1upp2=(min(xmt-1.d-1,xm1upp))**2
          endif
          if(xm1low2.gt.xm1upp2)then
            write(*,*)'Error in pair mass range #1'
            write(*,*)xm1low2,xm1upp2
            stop
          endif
c Parameters for the Breit Wigner
          bw1mdpl=xm1upp2-xmw2
          bw1mdmn=xmw2-xm1low2
          bw1fmpl=atan(bw1mdpl/(xmw*ga1))
          bw1fmmn=atan(bw1mdmn/(xmw*ga1))
          bw1delf=(bw1fmpl+bw1fmmn)/pi
        endif
c Initialize parameters relevant to decay
        call setpstdec()
      endif
c
      prefn = pref
      prefnev = prefev
c tau generated according to a flat distribution in (1/tau)**nsamp
      nsamp = 1
c
      ndim=6
      nwild=6
      rohlim=(xm1+xm2)**2/sh
      xicut=xicutss*(1-rohlim)
c Perform the integration step
      if(iseld.eq.1)then
        xtotal=0.d0
        ytotal=0.d0
        dtot=0.d0
        xares=0.d0
        yares=0.d0
        xbres=0.d0
        ybres=0.d0
        av3ai=0.d0
        av3ao=0.d0
        d3ai=0.d0
        d3ao=0.d0
        av3negai=0.d0
        av3negao=0.d0
        d3negai=0.d0
        d3negao=0.d0
        av3bi=0.d0
        av3bo=0.d0
        d3bi=0.d0
        d3bo=0.d0
        av3negbi=0.d0
        av3negbo=0.d0
        d3negbi=0.d0
        d3negbo=0.d0
        mx_of_evtai=0
        mx_of_evtao=0
        mx_of_evtbi=0
        mx_of_evtbo=0
c
        call fk88strcat(prefn,'_ain',fnameai)
        call fk88strcat(prefn,'_aout',fnameao)
        if(iinout.eq.1.or.iinout.eq.2)then
          ifuntype=1
          call run_bases(sig5azw_in,fnameai,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3ai,d3ai,av3negai,d3negai,ctime,itd1,itd2,
     #      iseed0,ibswrite,ibscall)
        endif
        if( (iinout.eq.1.or.iinout.eq.3).and.loproco.eq.2 )then
          ifuntype=3
          call run_bases(sig5azw_out,fnameao,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3ao,d3ao,av3negao,d3negao,ctime,itd1,itd2,
     #      iseed0,ibswrite,ibscall)
        endif
        if(basesoutput.gt.0) then
        if(iinout.eq.1.or.iinout.eq.2)then
          write(*,*)'   '
          write(*,*)'|integral[a,in]|:',av3ai,' +- ',d3ai
          write(*,*)' integral[a,in] :',av3negai,' +- ',d3negai
        endif
        if(iinout.eq.1.or.iinout.eq.3)then
          write(*,*)'   '
          write(*,*)'|integral[a,out]|:',av3ao,' +- ',d3ao
          write(*,*)' integral[a,out] :',av3negao,' +- ',d3negao
        endif
        endif
        xares=av3ai+av3ao
        yares=av3negai+av3negao
        xtotal=xtotal+xares
        ytotal=ytotal+yares
        dtot=dtot+d3negai**2+d3negao**2
c
        call fk88strcat(prefn,'_bin',fnamebi)
        call fk88strcat(prefn,'_bout',fnamebo)
        if(iinout.eq.1.or.iinout.eq.2)then
          ifuntype=2
          call run_bases(sig5bzw_in,fnamebi,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3bi,d3bi,av3negbi,d3negbi,ctime,itd1,itd2,
     #      iseed0,ibswrite,ibscall)
        endif
        if( (iinout.eq.1.or.iinout.eq.3).and.loproco.eq.2 )then
          ifuntype=4
          call run_bases(sig5bzw_out,fnamebo,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3bo,d3bo,av3negbo,d3negbo,ctime,itd1,itd2,
     #      iseed0,ibswrite,ibscall)
        endif
        if(basesoutput.gt.0) then
        if(iinout.eq.1.or.iinout.eq.2)then
          write(*,*)'   '
          write(*,*)'|integral[b,in]|:',av3bi,' +- ',d3bi
          write(*,*)' integral[b,in] :',av3negbi,' +- ',d3negbi
        endif
        if(iinout.eq.1.or.iinout.eq.3)then
          write(*,*)'   '
          write(*,*)'|integral[b,out]|:',av3bo,' +- ',d3bo
          write(*,*)' integral[b,out] :',av3negbo,' +- ',d3negbo
        endif
        endif
        xbres=av3bi+av3bo
        ybres=av3negbi+av3negbo
        xtotal=xtotal+xbres
        ytotal=ytotal+ybres
        dtot=dtot+d3negbi**2+d3negbo**2
c
        avtot=ytotal
        dtot=sqrt(dtot)
        call fk88strcat(prefn,'.integrals',fname)
        open(unit=21,file=fname,
     #       form='formatted',status='unknown')
        write(21,240)av3ai,av3ao
        write(21,240)av3bi,av3bo
        write(21,240)av3negai,av3negao
        write(21,240)av3negbi,av3negbo
        close(21)
 240    format(2(1x,d14.8))
      endif
c Sanity check
      if(isubttype.eq.1.and.deltai.ne.1.d0)then
        write(*,*)'Fatal error: xicut, deltaI=',xicut,deltai
        stop
      endif
      if(iseld.eq.0)then
c Read integrals from disk only if the integration step has been skipped
        call fk88strcat(prefn,'.integrals',fname)
        open(unit=21,file=fname,
     #       form='formatted',status='old')
        read(21,240)av3ai,av3ao
        read(21,240)av3bi,av3bo
        read(21,240)av3negai,av3negao
        read(21,240)av3negbi,av3negbo
        close(21)
      endif
c
c Generates events when evgen=.true.; if evgen=.false., maxevt=100000 in
c order to estimate the number of negative weights
      if(maxevt.ne.0)then
        ntotal=0
        xtotal=0.d0
        ytotal=0.d0
        xtotal=xtotal+av3ai+av3ao+av3bi+av3bo
        ytotal=ytotal+av3negai+av3negao+av3negbi+av3negbo
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
        mx_of_evtai=int(maxevt*av3ai/xtotal)
        mx_of_evtao=int(maxevt*av3ao/xtotal)
        mx_of_evtbi=int(maxevt*av3bi/xtotal)
        mx_of_evtbo=int(maxevt*av3bo/xtotal)
        ntotal=ntotal+mx_of_evtai+mx_of_evtao+
     #                mx_of_evtbi+mx_of_evtbo
        ndiff=maxevt-ntotal
        if(ndiff.gt.0)mx_of_evtai=mx_of_evtai+ndiff
        if(ndiff.lt.0)then
          write(6,*)'Fatal error:',maxevt,ntotal
          stop
        endif
        if(basesoutput.gt.0) then
        if(evgen)then
          write(*,*)'  '
          write(*,*)
     #  'The following number of events will be generated'
          write(*,*)'# events[a,in]: ',mx_of_evtai
          write(*,*)'# events[a,out]:',mx_of_evtao
          write(*,*)'# events[b,in]: ',mx_of_evtbi
          write(*,*)'# events[b,out]:',mx_of_evtbo
        endif
        write(*,*)'  '
        write(*,*)
     #  'Estimated fractions of events with negative weights'
        endif
        evfrac=0.d0
        if(av3ai.ne.0.d0)then
          evprcfrac=(av3ai-av3negai)/
     #              (av3ai+av3negai)
          evprcfrac=evprcfrac/(1+evprcfrac)
          evfrac=evfrac+evprcfrac*mx_of_evtai
          if(basesoutput.gt.0) then
          write(*,*)'Events[a,in]: w<0/all:',evprcfrac
          endif
        endif
        if(av3ao.ne.0.d0)then
          evprcfrac=(av3ao-av3negao)/
     #              (av3ao+av3negao)
          evprcfrac=evprcfrac/(1+evprcfrac)
          evfrac=evfrac+evprcfrac*mx_of_evtao
          if(basesoutput.gt.0) then
          write(*,*)'Events[a,out]: w<0/all:',evprcfrac
          endif
        endif
        if(av3bi.ne.0.d0)then
          evprcfrac=(av3bi-av3negbi)/
     #              (av3bi+av3negbi)
          evprcfrac=evprcfrac/(1+evprcfrac)
          evfrac=evfrac+evprcfrac*mx_of_evtbi
          if(basesoutput.gt.0) then
          write(*,*)'Events[b,in]: w<0/all:',evprcfrac
          endif
        endif
        if(av3bo.ne.0.d0)then
          evprcfrac=(av3bo-av3negbo)/
     #              (av3bo+av3negbo)
          evprcfrac=evprcfrac/(1+evprcfrac)
          evfrac=evfrac+evprcfrac*mx_of_evtbo
          if(basesoutput.gt.0) then
          write(*,*)'Events[b,out]: w<0/all:',evprcfrac
          endif
        endif
        evfrac=evfrac/dfloat(maxevt)
        if(basesoutput.gt.0) then
        write(*,*)'Events[all]: w<0/all:',evfrac
        endif
c
        if(.not.evgen)goto 111
        fname=prefnev
        call fk88strcat(fname,'_ain.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtai
        close(22)
        call fk88strcat(fname,'_aout.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtao
        close(22)
        call fk88strcat(fname,'_bin.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtbi
        close(22)
        call fk88strcat(fname,'_bout.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtbo
        close(22)
c
        fname=prefn
        fnamev=prefnev
c
        iwrong=0
        iwrong1=0
        neventsuw=0
        nqeventsuw=0
        ifailuw=0
        ncntuws=0
        nqcntuws=0
        nmaxuw=0
        nqmaxuw=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=1
        call fk88strcat(fname,'_ain',fnameai)
        call fk88strcat(fnamev,'_ain.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        call run_spring(sig5azw_in,fnameai,mx_of_evtai,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[a,in]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          if(idec.eq.0)then
            if(neventsuw.ne.mx_of_evtai)then
              write(*,*)'Error in spin correlations [a,in]'
              stop
            endif
            write(*,*)'   '
            write(*,*)'Secondary unweighting for spin correlations'
            write(*,*)'Failures',ifailuw
            write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
            write(*,*)'Maximum trials',nmaxuw
            write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
            if(iwidth.eq.1)then
              write(6,*)'Maximum trials [Q]',nqmaxuw
              write(6,*)'Efficiency [Q]',
     #                  nqeventsuw/dfloat(nqcntuws)
            endif
          endif
          write(*,*)'   '
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(8*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(5*nevts)
            endif
          enddo
        endif
c
        iwrong=0
        iwrong1=0
        neventsuw=0
        nqeventsuw=0
        ifailuw=0
        ncntuws=0
        nqcntuws=0
        nmaxuw=0
        nqmaxuw=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=3
        call fk88strcat(fname,'_aout',fnameao)
        call fk88strcat(fnamev,'_aout.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        call run_spring(sig5azw_out,fnameao,mx_of_evtao,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[a,out]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          if(idec.eq.0)then
            if(neventsuw.ne.mx_of_evtao)then
              write(*,*)'Error in spin correlations [a,out]'
              stop
            endif
            write(*,*)'   '
            write(*,*)'Secondary unweighting for spin correlations'
            write(*,*)'Failures',ifailuw
            write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
            write(*,*)'Maximum trials',nmaxuw
            write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
            if(iwidth.eq.1)then
              write(6,*)'Maximum trials [Q]',nqmaxuw
              write(6,*)'Efficiency [Q]',
     #                  nqeventsuw/dfloat(nqcntuws)
            endif
          endif
          write(*,*)'   '
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(7*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(4*nevts)
            endif
          enddo
        endif
c
        iwrong=0
        iwrong1=0
        neventsuw=0
        nqeventsuw=0
        ifailuw=0
        ncntuws=0
        nqcntuws=0
        nmaxuw=0
        nqmaxuw=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=2
        call fk88strcat(fname,'_bin',fnamebi)
        call fk88strcat(fnamev,'_bin.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        call run_spring(sig5bzw_in,fnamebi,mx_of_evtbi,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[b,in]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          if(idec.eq.0)then
            if(neventsuw.ne.mx_of_evtbi)then
              write(*,*)'Error in spin correlations [b,in]'
              stop
            endif
            write(*,*)'   '
            write(*,*)'Secondary unweighting for spin correlations'
            write(*,*)'Failures',ifailuw
            write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
            write(*,*)'Maximum trials',nmaxuw
            write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
            if(iwidth.eq.1)then
              write(6,*)'Maximum trials [Q]',nqmaxuw
              write(6,*)'Efficiency [Q]',
     #                  nqeventsuw/dfloat(nqcntuws)
            endif
          endif
          write(*,*)'   '
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(8*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(5*nevts)
            endif
          enddo
        endif
c
        iwrong=0
        iwrong1=0
        neventsuw=0
        nqeventsuw=0
        ifailuw=0
        ncntuws=0
        nqcntuws=0
        nmaxuw=0
        nqmaxuw=0
        do i=1,4
          xmomshifts(i)=0.d0
        enddo
        ifuntype=4
        call fk88strcat(fname,'_bout',fnamebo)
        call fk88strcat(fnamev,'_bout.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        call run_spring(sig5bzw_out,fnamebo,mx_of_evtbo,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[b,out]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
          if(idec.eq.0)then
            if(neventsuw.ne.mx_of_evtbo)then
              write(*,*)'Error in spin correlations [b,out]'
              stop
            endif
            write(*,*)'   '
            write(*,*)'Secondary unweighting for spin correlations'
            write(*,*)'Failures',ifailuw
            write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
            write(*,*)'Maximum trials',nmaxuw
            write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
            if(iwidth.eq.1)then
              write(6,*)'Maximum trials [Q]',nqmaxuw
              write(6,*)'Efficiency [Q]',
     #                  nqeventsuw/dfloat(nqcntuws)
            endif
          endif
          write(*,*)'   '
          write(*,*)'Average momentum shifts due to masses'
          do i=1,4
            if(idec.eq.0)then
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(7*nevts)
            else
              write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(4*nevts)
            endif
          enddo
        endif
c write a single event file
        iunitai=21
        call fk88strcat(prefnev,'_ain.events',fname1)
        open(unit=iunitai,file=fname1,form='formatted',status='old')
        read(iunitai,250)mx_of_evtai
        iunitbi=22
        call fk88strcat(prefnev,'_bin.events',fname1)
        open(unit=iunitbi,file=fname1,form='formatted',status='old')
        read(iunitbi,250)mx_of_evtbi
        iunitao=23
        call fk88strcat(prefnev,'_aout.events',fname1)
        open(unit=iunitao,file=fname1,form='formatted',status='old')
        read(iunitao,250)mx_of_evtao
        iunitbo=24
        call fk88strcat(prefnev,'_bout.events',fname1)
        open(unit=iunitbo,file=fname1,form='formatted',status='old')
        read(iunitbo,250)mx_of_evtbo
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
        write(ioutput,802)abs(iprdct0hw),'--> 2000/10/20=all/s-ch/t-ch'
        write(ioutput,803)xm1,twidth,'--> M_top, Gamma_top'
        write(ioutput,803)sqrt(xmw2),gaw,'--> M_W, Gamma_W'
        write(ioutput,815)il1hw,'--> IL1 (1,2,3,7)'
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
          call whichost(iseed,itot,mx_of_evtai,mx_of_evtbi,
     #                             mx_of_evtao,mx_of_evtbo,iunit)
          call retrieve_evenst(iunit,ii,dummy)
          call store_evenst(ioutput,xmone)
        enddo
        call crosschest(itot,mx_of_evtai,mx_of_evtbi,
     #                       mx_of_evtao,mx_of_evtbo)
        close(iunitai)
        close(iunitao)
        close(iunitbi)
        close(iunitbo)
        close(ioutput)
 111    continue
      endif
      if(basesoutput.gt.0) then
      if(idec.eq.0)then
        write(*,*) '   '
        write(*,*)'Branching ratio used in the computation:'
        write(*,*)' BR(t -> b e nu)=          ',brrtop1
      endif 
      write(*,*) '   '
      write(*,*) 'Total for fully inclusive'
      write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
      write(*,202) 
      write(*,270)xm1
      write(*,201) 'tot'
      write(*,300)ecm,xfh,xren,avtot,dtot
      endif
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #  '  nl=',i2,'  lambda5=',d10.4)
 201  format(' ecm or ebeam  xf   xr   ',a,
     # '        err    ')
 202  format(' M_top')
 270  format(1x,1pd9.3)
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
 814  format(1x,d10.4,1x,a)
 815  format(1x,i2,1x,a)
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


      subroutine strfst(x1,x2,sf)
c Return parton densities through the matrix
c  sf(idr,jproc,itype,ich,ittbar), with the following conventions:
c   idr -> identifies the partonic process given jproc and ich
c   jproc=2,3 -> q(bar)q(bar), q(bar)g processes respectively
c   itype -> identifies the individual contribution to a given jproc
c   ich=1,2 -> s-channel, t-channel
c   ittbar=1,2 -> t production (IT=4), tbar production (IT=1), IT being
c                 HERWIG v6.50? labeling convention
c ckm2(i,j)=|CKM matrix elements|^2, with  i=1,4,6 --> up,charm,top
c                                          j=2,3,5 --> down,strange,bottom
c and the following combination must be also defined (here in setpst)
c ruckm   = |V_ud|^2+|V_us|^2+|V_ub|^2
c rcckm   = |V_cd|^2+|V_cs|^2+|V_cb|^2
c rtckm   = |V_td|^2+|V_ts|^2+|V_tb|^2
c rducckm = |V_ud|^2+|V_cd|^2
c rsucckm = |V_us|^2+|V_cs|^2
c rbucckm = |V_ub|^2+|V_cb|^2
      implicit none
      real*4 fh1x1(-5:5),fh2x2(-5:5),smuf2h1,smuf2h2
      real * 8 pi,x1,x2,sf(8,2:3,9,2,2)
      integer ih1,ih2,ndns1,ndns2,ii,jproc,itype,ich,ittbar
      parameter(pi=3.14159265358979312D0)
      include 'stpcblks.h'
      common/strfst0/ih1,ih2,ndns1,ndns2
      real*8 zel(1:6),zel2(1:6)
      real*8 ckm2(1:6,1:6)
      real*8 ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      common/charges/zel,zel2
      common/cckm2/ckm2
      common/cckmfct/ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      integer ipdfscale
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c
      do jproc=2,3
        do ich=ichmin,ichmax
          do ii=1,idrmax(jproc,ich)
            do itype=1,itypemax(jproc)
              do ittbar=ittmin,ittmax
                sf(ii,jproc,itype,ich,ittbar)=0.d0
              enddo
            enddo
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
        write(*,*)'Fatal error in strfst: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
c
c jproc=2
      if(ittmin.eq.1)then
c t production
        if(ichmin.eq.1)then
c s-channel
          sf(1,2,1,1,1)=ckm2(1,2)*dble(fh1x1( 1) * fh2x2(-2))*rtckm
          sf(1,2,2,1,1)=ckm2(1,3)*dble(fh1x1( 1) * fh2x2(-3))*rtckm
          sf(1,2,3,1,1)=ckm2(1,5)*dble(fh1x1( 1) * fh2x2(-5))*rtckm
          sf(1,2,4,1,1)=ckm2(4,2)*dble(fh1x1( 4) * fh2x2(-2))*rtckm
          sf(1,2,5,1,1)=ckm2(4,3)*dble(fh1x1( 4) * fh2x2(-3))*rtckm
          sf(1,2,6,1,1)=ckm2(4,5)*dble(fh1x1( 4) * fh2x2(-5))*rtckm
c
          sf(3,2,1,1,1)=ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 1))*rtckm
          sf(3,2,2,1,1)=ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 1))*rtckm
          sf(3,2,3,1,1)=ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 1))*rtckm
          sf(3,2,4,1,1)=ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 4))*rtckm
          sf(3,2,5,1,1)=ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 4))*rtckm
          sf(3,2,6,1,1)=ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 4))*rtckm
        endif
        if(ichmax.eq.2)then
c t-channel
          sf(1,2,1,2,1)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2( 1))*ruckm
          sf(1,2,2,2,1)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2( 1))*ruckm
          sf(1,2,3,2,1)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2( 1))*ruckm
          sf(1,2,4,2,1)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2( 4))*rcckm
          sf(1,2,5,2,1)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2( 4))*rcckm
          sf(1,2,6,2,1)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2( 4))*rcckm
c
          sf(2,2,1,2,1)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2(-2))*rducckm
          sf(2,2,2,2,1)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2(-3))*rsucckm
          sf(2,2,3,2,1)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2(-5))*rbucckm
          sf(2,2,4,2,1)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2(-2))*rducckm
          sf(2,2,5,2,1)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2(-3))*rsucckm
          sf(2,2,6,2,1)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2(-5))*rbucckm
          sf(2,2,7,2,1)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2(-2))*rducckm
          sf(2,2,8,2,1)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2(-3))*rsucckm
          sf(2,2,9,2,1)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2(-5))*rbucckm
c
          sf(3,2,1,2,1)=ckm2(6,2)*dble(fh1x1( 1) * fh2x2( 2))*ruckm
          sf(3,2,2,2,1)=ckm2(6,3)*dble(fh1x1( 1) * fh2x2( 3))*ruckm
          sf(3,2,3,2,1)=ckm2(6,5)*dble(fh1x1( 1) * fh2x2( 5))*ruckm
          sf(3,2,4,2,1)=ckm2(6,2)*dble(fh1x1( 4) * fh2x2( 2))*rcckm
          sf(3,2,5,2,1)=ckm2(6,3)*dble(fh1x1( 4) * fh2x2( 3))*rcckm
          sf(3,2,6,2,1)=ckm2(6,5)*dble(fh1x1( 4) * fh2x2( 5))*rcckm
c
          sf(4,2,1,2,1)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2( 2))*rducckm
          sf(4,2,2,2,1)=ckm2(6,2)*dble(fh1x1(-3) * fh2x2( 2))*rsucckm
          sf(4,2,3,2,1)=ckm2(6,2)*dble(fh1x1(-5) * fh2x2( 2))*rbucckm
          sf(4,2,4,2,1)=ckm2(6,3)*dble(fh1x1(-2) * fh2x2( 3))*rducckm
          sf(4,2,5,2,1)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2( 3))*rsucckm
          sf(4,2,6,2,1)=ckm2(6,3)*dble(fh1x1(-5) * fh2x2( 3))*rbucckm
          sf(4,2,7,2,1)=ckm2(6,5)*dble(fh1x1(-2) * fh2x2( 5))*rducckm
          sf(4,2,8,2,1)=ckm2(6,5)*dble(fh1x1(-3) * fh2x2( 5))*rsucckm
          sf(4,2,9,2,1)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2( 5))*rbucckm
        endif
      endif
      if(ittmax.eq.2)then
c tbar production
        if(ichmin.eq.1)then
c s-channel
          sf(1,2,1,1,2)=ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 2))*rtckm
          sf(1,2,2,1,2)=ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 3))*rtckm
          sf(1,2,3,1,2)=ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 5))*rtckm
          sf(1,2,4,1,2)=ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 2))*rtckm
          sf(1,2,5,1,2)=ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 3))*rtckm
          sf(1,2,6,1,2)=ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 5))*rtckm

          sf(3,2,1,1,2)=ckm2(1,2)*dble(fh1x1( 2) * fh2x2(-1))*rtckm
          sf(3,2,2,1,2)=ckm2(1,3)*dble(fh1x1( 3) * fh2x2(-1))*rtckm
          sf(3,2,3,1,2)=ckm2(1,5)*dble(fh1x1( 5) * fh2x2(-1))*rtckm
          sf(3,2,4,1,2)=ckm2(4,2)*dble(fh1x1( 2) * fh2x2(-4))*rtckm
          sf(3,2,5,1,2)=ckm2(4,3)*dble(fh1x1( 3) * fh2x2(-4))*rtckm
          sf(3,2,6,1,2)=ckm2(4,5)*dble(fh1x1( 5) * fh2x2(-4))*rtckm
        endif
        if(ichmax.eq.2)then
c t-channel
          sf(1,2,1,2,2)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2(-1))*ruckm
          sf(1,2,2,2,2)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2(-1))*ruckm
          sf(1,2,3,2,2)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2(-1))*ruckm
          sf(1,2,4,2,2)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2(-4))*rcckm
          sf(1,2,5,2,2)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2(-4))*rcckm
          sf(1,2,6,2,2)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2(-4))*rcckm
c
          sf(2,2,1,2,2)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2( 2))*rducckm
          sf(2,2,2,2,2)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2( 3))*rsucckm
          sf(2,2,3,2,2)=ckm2(6,2)*dble(fh1x1(-2) * fh2x2( 5))*rbucckm
          sf(2,2,4,2,2)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2( 2))*rducckm
          sf(2,2,5,2,2)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2( 3))*rsucckm
          sf(2,2,6,2,2)=ckm2(6,3)*dble(fh1x1(-3) * fh2x2( 5))*rbucckm
          sf(2,2,7,2,2)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2( 2))*rducckm
          sf(2,2,8,2,2)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2( 3))*rsucckm
          sf(2,2,9,2,2)=ckm2(6,5)*dble(fh1x1(-5) * fh2x2( 5))*rbucckm
c
          sf(3,2,1,2,2)=ckm2(6,2)*dble(fh1x1(-1) * fh2x2(-2))*ruckm
          sf(3,2,2,2,2)=ckm2(6,3)*dble(fh1x1(-1) * fh2x2(-3))*ruckm
          sf(3,2,3,2,2)=ckm2(6,5)*dble(fh1x1(-1) * fh2x2(-5))*ruckm
          sf(3,2,4,2,2)=ckm2(6,2)*dble(fh1x1(-4) * fh2x2(-2))*rcckm
          sf(3,2,5,2,2)=ckm2(6,3)*dble(fh1x1(-4) * fh2x2(-3))*rcckm
          sf(3,2,6,2,2)=ckm2(6,5)*dble(fh1x1(-4) * fh2x2(-5))*rcckm
c
          sf(4,2,1,2,2)=ckm2(6,2)*dble(fh1x1( 2) * fh2x2(-2))*rducckm
          sf(4,2,2,2,2)=ckm2(6,2)*dble(fh1x1( 3) * fh2x2(-2))*rsucckm
          sf(4,2,3,2,2)=ckm2(6,2)*dble(fh1x1( 5) * fh2x2(-2))*rbucckm
          sf(4,2,4,2,2)=ckm2(6,3)*dble(fh1x1( 2) * fh2x2(-3))*rducckm
          sf(4,2,5,2,2)=ckm2(6,3)*dble(fh1x1( 3) * fh2x2(-3))*rsucckm
          sf(4,2,6,2,2)=ckm2(6,3)*dble(fh1x1( 5) * fh2x2(-3))*rbucckm
          sf(4,2,7,2,2)=ckm2(6,5)*dble(fh1x1( 2) * fh2x2(-5))*rducckm
          sf(4,2,8,2,2)=ckm2(6,5)*dble(fh1x1( 3) * fh2x2(-5))*rsucckm
          sf(4,2,9,2,2)=ckm2(6,5)*dble(fh1x1( 5) * fh2x2(-5))*rbucckm
        endif
      endif
c jproc=3
      if(ittmin.eq.1)then
c t production
        if(ichmin.eq.1)then
c s-channel
          sf(1,3,1,1,1)=dble(fh1x1( 1) * fh2x2( 0))*ruckm*rtckm
          sf(1,3,2,1,1)=dble(fh1x1( 4) * fh2x2( 0))*rcckm*rtckm
c
          sf(2,3,1,1,1)=dble(fh1x1(-2) * fh2x2( 0))*rducckm*rtckm
          sf(2,3,2,1,1)=dble(fh1x1(-3) * fh2x2( 0))*rsucckm*rtckm
          sf(2,3,3,1,1)=dble(fh1x1(-5) * fh2x2( 0))*rbucckm*rtckm
c
          sf(3,3,1,1,1)=dble(fh1x1( 0) * fh2x2( 1))*ruckm*rtckm
          sf(3,3,2,1,1)=dble(fh1x1( 0) * fh2x2( 4))*rcckm*rtckm
c
          sf(4,3,1,1,1)=dble(fh1x1( 0) * fh2x2(-2))*rducckm*rtckm
          sf(4,3,2,1,1)=dble(fh1x1( 0) * fh2x2(-3))*rsucckm*rtckm
          sf(4,3,3,1,1)=dble(fh1x1( 0) * fh2x2(-5))*rbucckm*rtckm
        endif
        if(ichmax.eq.2)then
c t-channel
          sf(1,3,1,2,1)=dble(fh1x1( 2) * fh2x2( 0))*
     #                  ckm2(6,2)*(ruckm+rcckm)
          sf(1,3,2,2,1)=dble(fh1x1( 3) * fh2x2( 0))*
     #                  ckm2(6,3)*(ruckm+rcckm)
          sf(1,3,3,2,1)=dble(fh1x1( 5) * fh2x2( 0))*
     #                  ckm2(6,5)*(ruckm+rcckm)
c
          sf(2,3,1,2,1)=dble(fh1x1( 0) * fh2x2( 1))*ruckm*rtckm
          sf(2,3,2,2,1)=dble(fh1x1( 0) * fh2x2( 4))*rcckm*rtckm
c
          sf(3,3,1,2,1)=dble(fh1x1( 2) * fh2x2( 0))*
     #                  ckm2(6,2)*(rducckm+rsucckm+rbucckm)
          sf(3,3,2,2,1)=dble(fh1x1( 3) * fh2x2( 0))*
     #                  ckm2(6,3)*(rducckm+rsucckm+rbucckm)
          sf(3,3,3,2,1)=dble(fh1x1( 5) * fh2x2( 0))*
     #                  ckm2(6,5)*(rducckm+rsucckm+rbucckm)
c
          sf(4,3,1,2,1)=dble(fh1x1( 0) * fh2x2(-2))*rducckm*rtckm
          sf(4,3,2,2,1)=dble(fh1x1( 0) * fh2x2(-3))*rsucckm*rtckm
          sf(4,3,3,2,1)=dble(fh1x1( 0) * fh2x2(-5))*rbucckm*rtckm
c
          sf(5,3,1,2,1)=dble(fh1x1( 1) * fh2x2( 0))*ruckm*rtckm
          sf(5,3,2,2,1)=dble(fh1x1( 4) * fh2x2( 0))*rcckm*rtckm
c
          sf(6,3,1,2,1)=dble(fh1x1( 0) * fh2x2( 2))*
     #                  ckm2(6,2)*(ruckm+rcckm)
          sf(6,3,2,2,1)=dble(fh1x1( 0) * fh2x2( 3))*
     #                  ckm2(6,3)*(ruckm+rcckm)
          sf(6,3,3,2,1)=dble(fh1x1( 0) * fh2x2( 5))*
     #                  ckm2(6,5)*(ruckm+rcckm)
c
          sf(7,3,1,2,1)=dble(fh1x1(-2) * fh2x2( 0))*rducckm*rtckm
          sf(7,3,2,2,1)=dble(fh1x1(-3) * fh2x2( 0))*rsucckm*rtckm
          sf(7,3,3,2,1)=dble(fh1x1(-5) * fh2x2( 0))*rbucckm*rtckm
c
          sf(8,3,1,2,1)=dble(fh1x1( 0) * fh2x2( 2))*
     #                  ckm2(6,2)*(rducckm+rsucckm+rbucckm)
          sf(8,3,2,2,1)=dble(fh1x1( 0) * fh2x2( 3))*
     #                  ckm2(6,3)*(rducckm+rsucckm+rbucckm)
          sf(8,3,3,2,1)=dble(fh1x1( 0) * fh2x2( 5))*
     #                  ckm2(6,5)*(rducckm+rsucckm+rbucckm)
        endif
      endif
      if(ittmax.eq.2)then
c tbar production
        if(ichmin.eq.1)then
c s-channel
          sf(1,3,1,1,2)=dble(fh1x1(-1) * fh2x2( 0))*ruckm*rtckm
          sf(1,3,2,1,2)=dble(fh1x1(-4) * fh2x2( 0))*rcckm*rtckm
c
          sf(2,3,1,1,2)=dble(fh1x1( 2) * fh2x2( 0))*rducckm*rtckm
          sf(2,3,2,1,2)=dble(fh1x1( 3) * fh2x2( 0))*rsucckm*rtckm
          sf(2,3,3,1,2)=dble(fh1x1( 5) * fh2x2( 0))*rbucckm*rtckm
c
          sf(3,3,1,1,2)=dble(fh1x1( 0) * fh2x2(-1))*ruckm*rtckm
          sf(3,3,2,1,2)=dble(fh1x1( 0) * fh2x2(-4))*rcckm*rtckm
c
          sf(4,3,1,1,2)=dble(fh1x1( 0) * fh2x2( 2))*rducckm*rtckm
          sf(4,3,2,1,2)=dble(fh1x1( 0) * fh2x2( 3))*rsucckm*rtckm
          sf(4,3,3,1,2)=dble(fh1x1( 0) * fh2x2( 5))*rbucckm*rtckm
        endif
        if(ichmax.eq.2)then
c t-channel
          sf(1,3,1,2,2)=dble(fh1x1(-2) * fh2x2( 0))*
     #                  ckm2(6,2)*(ruckm+rcckm)
          sf(1,3,2,2,2)=dble(fh1x1(-3) * fh2x2( 0))*
     #                  ckm2(6,3)*(ruckm+rcckm)
          sf(1,3,3,2,2)=dble(fh1x1(-5) * fh2x2( 0))*
     #                  ckm2(6,5)*(ruckm+rcckm)
c
          sf(2,3,1,2,2)=dble(fh1x1( 0) * fh2x2(-1))*ruckm*rtckm
          sf(2,3,2,2,2)=dble(fh1x1( 0) * fh2x2(-4))*rcckm*rtckm
c
          sf(3,3,1,2,2)=dble(fh1x1(-2) * fh2x2( 0))*
     #                  ckm2(6,2)*(rducckm+rsucckm+rbucckm)
          sf(3,3,2,2,2)=dble(fh1x1(-3) * fh2x2( 0))*
     #                  ckm2(6,3)*(rducckm+rsucckm+rbucckm)
          sf(3,3,3,2,2)=dble(fh1x1(-5) * fh2x2( 0))*
     #                  ckm2(6,5)*(rducckm+rsucckm+rbucckm)
c
          sf(4,3,1,2,2)=dble(fh1x1( 0) * fh2x2( 2))*rducckm*rtckm
          sf(4,3,2,2,2)=dble(fh1x1( 0) * fh2x2( 3))*rsucckm*rtckm
          sf(4,3,3,2,2)=dble(fh1x1( 0) * fh2x2( 5))*rbucckm*rtckm
c
          sf(5,3,1,2,2)=dble(fh1x1(-1) * fh2x2( 0))*ruckm*rtckm
          sf(5,3,2,2,2)=dble(fh1x1(-4) * fh2x2( 0))*rcckm*rtckm
c
          sf(6,3,1,2,2)=dble(fh1x1( 0) * fh2x2(-2))*
     #                  ckm2(6,2)*(ruckm+rcckm)
          sf(6,3,2,2,2)=dble(fh1x1( 0) * fh2x2(-3))*
     #                  ckm2(6,3)*(ruckm+rcckm)
          sf(6,3,3,2,2)=dble(fh1x1( 0) * fh2x2(-5))*
     #                  ckm2(6,5)*(ruckm+rcckm)
c
          sf(7,3,1,2,2)=dble(fh1x1( 2) * fh2x2( 0))*rducckm*rtckm
          sf(7,3,2,2,2)=dble(fh1x1( 3) * fh2x2( 0))*rsucckm*rtckm
          sf(7,3,3,2,2)=dble(fh1x1( 5) * fh2x2( 0))*rbucckm*rtckm
c
          sf(8,3,1,2,2)=dble(fh1x1( 0) * fh2x2( -2))*
     #                  ckm2(6,2)*(rducckm+rsucckm+rbucckm)
          sf(8,3,2,2,2)=dble(fh1x1( 0) * fh2x2( -3))*
     #                  ckm2(6,3)*(rducckm+rsucckm+rbucckm)
          sf(8,3,3,2,2)=dble(fh1x1( 0) * fh2x2( -5))*
     #                  ckm2(6,5)*(rducckm+rsucckm+rbucckm)
        endif
      endif
c
      return
      end
c
c
c NLO cross section
c
c
      function sig5azw_in(xx)
      implicit none
      real * 8 sig5azw_in,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,rohlim,zzz,x,ttt,th,yi,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xmi2,xalfaem,rox,cth1,th2,cth2,tot5a_in
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1.d0
      rohlim=(sqrt(xm12)+sqrt(xm22))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the initial state collinear regions
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
      yi    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+yi)/2.d0)/(1-(1-x)*(1-yi)/2.d0))
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
        xxa1 = (1+x-yi*(1-x))/2.d0
        xxa2 = (1+x+yi*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
c
c Hard coded choice for scale of running e.m. coupling: mtop
      xmi2=xm12
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
c
      rox = 2*(xm12+xm22)/(s*x)-(xm12-xm22)**2/(s*x)**2
c zzchvar: a change of variables xx(5) --> cth1
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5azw_in = tot5a_in(s,x,yi,cth1,cth2,xjac)
      return
      end


      function tot5a_in(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5a_in,tot5as_in,tot5az_in,s,x,yi,cth1,cth2,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5as_in(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5az_in(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5a_in:',isubttype
        stop
      endif
      tot5a_in=tmp
      return
      end


      function tot5as_in(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5as_in,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,two,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
      parameter (two=2.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      common/cpsave/ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      real * 8 vv(8,2:3,9,2,2),vvs(8,2:3,9,2,2)
      common/cvv/vv
      common/cvvs/vvs
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer idrmap(8,2:3,2)
      common/cidrmap/idrmap
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      real * 8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      logical lzone(8,2)
      real * 8 xinv(5)
      real * 8 sf(8,2:3,9,2,2)
      real * 8 vecre(8,2),vecmc(8,2),xmcz(8,2)
      real * 8 vecmccl(8,2),vecmcsf(8,2),vecmcsc(8,2)
      real * 8 gfsf(2:3),gfcl(2:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,rox,betax,xphsp_ns,zgw4,
     # x1,x2,tk,uk,q1q,q2q,zg2_nlo,zgmu2_nst,xnorm,s_in,zg2_mc,
     # zgmu2_st,zhwfct,x1t,x2t,x1soft,x2soft,x1x2j,x1x2jac,xsum,
     # dummy,xint,xtmp,ytmp,gfactsf,gfactcl,xfact,xphsp_s
      integer loproc,maproc,jproc,j,itype,i,itt,jp,ipdfscale,ileg,
     # iret,i2b,itoosoftkin,idec
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
      common/cidec/idec
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
      rox = 2*(xm12+xm22)/sx-(xm12-xm22)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4)
c
      zgw4 = (ze2/sthw2)**2
c
      do jproc=2,3
        do i=1,2
          do j=1,8
            do itype=1,9
              do itt=1,2
                vv(j,jproc,itype,i,itt)=0.d0
                vvs(j,jproc,itype,i,itt)=0.d0
              enddo
            enddo
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invar_in(xm12,xm22,s,x,yi,cth1,cth2,str,
     #                tk,uk,q1q,q2q,xinv)
        zg2_nlo = zgmu2_nst() 
        ipdfscale=1
        call strfst(x1,x2,sf)
C This corresponds to Eq. (4.37) in FKS
        xnorm = zgw4 * zg2_nlo * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          call frest(s,x,yi,two,two,tk,uk,q1q,q2q,xinv,jproc,vecre)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              do itype=1,itypemax(jproc)
                do itt=ittmin,ittmax
                  jp=idrmap(j,jproc,itt)
                  vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #              sf(j,jproc,itype,i,itt)*xnorm*
     #              vecre(jp,i)*s_in(xinv,jproc)
                enddo
              enddo
            enddo
          enddo
        enddo
c
c MC subtraction terms: pure MC
c
        zg2_mc = zgmu2_st()
        ipdfscale=2
        xnorm = zgw4 * zg2_mc * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          ileg=1
          call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,cth1,cth2,
     #      two,zero,zero,two,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #      lzone,xmcz,vecmc)
          gfsf(jproc)=gfactsf
          gfcl(jproc)=gfactcl
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              if(lzone(j,i))then
                zhwfct=xmcz(j,i)
                x1t=x1soft(x1,x2,x,yi)/zhwfct
                x2t=x2soft(x1,x2,x,yi)
                if(x1t.lt.1.and.x2t.lt.1)then
                  call strfst(x1t,x2t,sf)
                  x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                 sf(j,jproc,itype,i,itt)*xnorm*x1x2j*vecmc(jp,i)
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
c
          ileg=2
          call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,cth1,cth2,
     #      two,zero,zero,two,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #      lzone,xmcz,vecmc)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              if(lzone(j,i))then
                zhwfct=xmcz(j,i)
                x1t=x1soft(x1,x2,x,yi)
                x2t=x2soft(x1,x2,x,yi)/zhwfct
                if(x1t.lt.1.and.x2t.lt.1)then
                  call strfst(x1t,x2t,sf)
                  x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                 sf(j,jproc,itype,i,itt)*xnorm*x1x2j*vecmc(jp,i)
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
        enddo
c
c MC subtraction term: collinear ME
c
        do jproc=loproc,maproc
          if(gfcl(jproc).lt.1.d0)then
            if(yi.gt.0.d0)then
              ytmp=1.d0
              x1t=x1soft(x1,x2,x,yi)/x
              x2t=x2soft(x1,x2,x,yi)
              xfact=1.d0/( xii*(1-yi) )
            else
              ytmp=-1.d0
              x1t=x1soft(x1,x2,x,yi)
              x2t=x2soft(x1,x2,x,yi)/x
              xfact=1.d0/( xii*(1+yi) )
            endif
            if(x1t.lt.1.and.x2t.lt.1)then
              x1x2j = x1x2jac(x1,x2,x,yi)/x
              call invar_in(xm12,xm22,s,x,ytmp,cth1,cth2,str,
     #                      tk,uk,q1q,q2q,xinv)
              zg2_nlo = zgmu2_nst() 
              ipdfscale=1
              call strfst(x1t,x2t,sf)
              xnorm = xfact * x1x2j * zgw4 * zg2_nlo * xphsp_ns
              xnorm = xnorm * (1-gfcl(jproc))
              call frest(s,x,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecmccl)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmccl(jp,i)
                    enddo
                  enddo
                enddo
              enddo
            endif
          endif
c
c MC subtraction term: soft ME
c
          if(gfsf(jproc).lt.1.d0)then
            xtmp=1.d0
            x1t=x1soft(x1,x2,x,yi)
            x2t=x2soft(x1,x2,x,yi)
            if(x1t.lt.1.and.x2t.lt.1)then
              x1x2j = x1x2jac(x1,x2,x,yi)
              call invar_in(xm12,xm22,sx,xtmp,yi,cth1,cth2,str,
     #                      tk,uk,q1q,q2q,xinv)
              zg2_nlo = zgmu2_nst() 
              ipdfscale=1
              call strfst(x1t,x2t,sf)
              xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_s 
              xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
              xnorm = xnorm * (1-gfsf(jproc))
              call frest(sx,xtmp,yi,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecmcsf)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                  sf(j,jproc,itype,i,itt)*xnorm*
     #                  vecmcsf(jp,i)*s_in(xinv,jproc)
                    enddo
                  enddo
                enddo
              enddo
c
c MC subtraction term: soft-collinear ME
c
              if(gfcl(jproc).lt.1.d0)then
                if(yi.gt.0.d0)then
                  ytmp=1.d0
                  xfact=1.d0/( xii*(1-yi) )
                else
                  ytmp=-1.d0
                  xfact=1.d0/( xii*(1+yi) )
                endif
                call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                        tk,uk,q1q,q2q,xinv)
                xnorm = xfact * x1x2j * zgw4 * zg2_nlo * xphsp_s
                xnorm = - xnorm * (1-gfsf(jproc)) * (1-gfcl(jproc))
                call frest(sx,xtmp,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                     jproc,vecmcsc)
                do i=ichmin,ichmax
                  do j=1,idrmax(jproc,i)
                    do itype=1,itypemax(jproc)
                      do itt=ittmin,ittmax
                        jp=idrmap(j,jproc,itt)
                        vv(j,jproc,itype,i,itt)=
     #                    vv(j,jproc,itype,i,itt)-
     #                    sf(j,jproc,itype,i,itt)*xnorm*vecmcsc(jp,i)
                      enddo
                    enddo
                  enddo
                enddo
              endif
            endif
          endif
        enddo
      endif
c
      call checkst(xsum,dummy,iret)
      if(iret.eq.1)then
        call invar_in(xm12,xm22,s,x,yi,cth1,cth2,str,
     #                tk,uk,q1q,q2q,xinv)
        if(idec.eq.0)then
          ps=s
          px=x
          pyi=yi
          pcth1=cth1
          pcth2=cth2
          pphii=-1.d8
          pyj=-1.d8
          pphij=-1.d8
        endif
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,xinv)
          if(idec.eq.0)then
            ps=sx
            px=xtmp
            pyi=ytmp
            pcth1=cth1
            pcth2=cth2
            pphii=-1.d8
            pyj=-1.d8
            pphij=-1.d8
          endif
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5as_in=abs(xint)
c
      return
      end


C Empty now, but for zeta-subtraction, to be done,
C if the number of negative weight events for single top
C happens to be too large
      function tot5az_in(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5az_in,xs,xx,xyi,xcth1,xcth2,xjac
      tot5az_in=0.d0
      return
      end


      function sig5bzw_in(xx)
      implicit none
      real * 8 sig5bzw_in,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      integer iprespl
      common/ciprespl/iprespl
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,rohlim,zzz,x,ttt,th,yi,csi,rx,rohlimx,taumax,
     #  ximax0,ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,
     #  s,xmi2,xalfaem,rox,cth1,th2,cth2,tot5b_in
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1.d0
      rohlim=(sqrt(xm12)+sqrt(xm22))**2/sh
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-rohlim)
      xjac = xjac * (1-rohlim)
c
c To improve convergence in the initial state collinear regions
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
      yi    = cos(th)
      xjac = xjac * sin(th)
c
      csi = sqrt((1-(1-x)*(1+yi)/2.d0)/(1-(1-x)*(1-yi)/2.d0))
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
        xxa1 = (1+x-yi*(1-x))/2.d0
        xxa2 = (1+x+yi*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
      s=tau*sh
c
c Hard coded choice for scale of running e.m. coupling: mtop
      xmi2=xm12
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
c
      rox = 2*(xm12+xm22)/(s*x)-(xm12-xm22)**2/(s*x)**2
c zzchvar: a change of variables xx(5) --> cth1
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5bzw_in = tot5b_in(s,x,yi,cth1,cth2,xjac)
      return
      end


      function tot5b_in(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5b_in,tot5bs_in,tot5bz_in,s,x,yi,cth1,cth2,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5bs_in(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5bz_in(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5b_in:',isubttype
        stop
      endif
      tot5b_in=tmp
      return
      end


      function tot5bs_in(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bs_in,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,two,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
      parameter (two=2.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      common/cpsave/ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      real * 8 vv(8,2:3,9,2,2),vvs(8,2:3,9,2,2)
      common/cvv/vv
      common/cvvs/vvs
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer idrmap(8,2:3,2)
      common/cidrmap/idrmap
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      real * 8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      logical lzone(8,2)
      real * 8 xinv(5)
      real * 8 sf(8,2:3,9,2,2)
      real * 8 xintsvc(1:8,2:3,2),xbornvc(1:8,2:3,2)
      real * 8 xcsvc(1:8,2:3,2),xsvvc(1:8,2:3,2)
      real * 8 vecre(8,2),veccl(8,2)
      real * 8 vecbrn(8,2),vec2sv(8,2)
      real * 8 vecmc(8,2),xmcz(8,2)
      real * 8 vecmccl(8,2),vecmcsf(8,2),vecmcsc(8,2)
      real * 8 gfsf(2:3),gfcl(2:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,xphspb,xphsp_ns,xphsp_s,xint,
     #  x1,x2,x1t,x2t,tk,uk,q1q,q2q,zg2_nlo,ro,beta,rox,betax,
     #  zgmu2_nst,xnorm,ytmp,x1soft,x2soft,x1x2j,x1x2jac,xlmude,
     #  xnormc,xtmp,xnormb,xnormsv,zgw4,s_in,zg2_mc,zgmu2_st,
     #  zhwfct,xsum,dummy,gfactsf,gfactcl,xfact
      integer loproc,maproc,jproc,j,itype,i,itt,jp,ipdfscale,ileg,
     #  iret,idec
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
      common/cidec/idec
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
      ro = 2*(xm12+xm22)/s-(xm12-xm22)**2/s**2
      beta = sqrt(1-ro)
      rox = 2*(xm12+xm22)/sx-(xm12-xm22)**2/sx**2
      betax = sqrt(1-rox)
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac * betax * s/(2*1024*pi**4)
      xphsp_s = xjac * betax * sx/(2*1024*pi**4)
      xphspb = xjac * betax/(32*pi2)
c
      zgw4 = (ze2/sthw2)**2
c
      do jproc=2,3
        do i=1,2
          do j=1,8
            do itype=1,9
              do itt=1,2
                vv(j,jproc,itype,i,itt)=0.d0
                vvs(j,jproc,itype,i,itt)=0.d0
              enddo
            enddo
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invar_in(xm12,xm22,s,x,yi,cth1,cth2,str,
     #                tk,uk,q1q,q2q,xinv)
c
c MC subtraction terms: pure MC
c
        zg2_mc = zgmu2_st()
        ipdfscale=2
        xnorm = zgw4 * zg2_mc * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          ileg=1
          call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,cth1,cth2,
     #      two,zero,zero,two,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #      lzone,xmcz,vecmc)
          gfsf(jproc)=gfactsf
          gfcl(jproc)=gfactcl
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              if(lzone(j,i))then
                zhwfct=xmcz(j,i)
                x1t=x1soft(x1,x2,x,yi)/zhwfct
                x2t=x2soft(x1,x2,x,yi)
                if(x1t.lt.1.and.x2t.lt.1)then
                  call strfst(x1t,x2t,sf)
                  x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                 sf(j,jproc,itype,i,itt)*xnorm*x1x2j*vecmc(jp,i)
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
c
          ileg=2
          call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,cth1,cth2,
     #      two,zero,zero,two,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #      lzone,xmcz,vecmc)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              if(lzone(j,i))then
                zhwfct=xmcz(j,i)
                x1t=x1soft(x1,x2,x,yi)
                x2t=x2soft(x1,x2,x,yi)/zhwfct
                if(x1t.lt.1.and.x2t.lt.1)then
                  call strfst(x1t,x2t,sf)
                  x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                 sf(j,jproc,itype,i,itt)*xnorm*x1x2j*vecmc(jp,i)
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
        enddo
c
c MC subtraction term: collinear ME
c
        do jproc=loproc,maproc
          if(gfcl(jproc).lt.1.d0)then
            if(yi.gt.0.d0)then
              ytmp=1.d0
              x1t=x1soft(x1,x2,x,yi)/x
              x2t=x2soft(x1,x2,x,yi)
              xfact=1.d0/( xii*(1-yi) )
            else
              ytmp=-1.d0
              x1t=x1soft(x1,x2,x,yi)
              x2t=x2soft(x1,x2,x,yi)/x
              xfact=1.d0/( xii*(1+yi) )
            endif
            if(x1t.lt.1.and.x2t.lt.1)then
              x1x2j = x1x2jac(x1,x2,x,yi)/x
              call invar_in(xm12,xm22,s,x,ytmp,cth1,cth2,str,
     #                      tk,uk,q1q,q2q,xinv)
              zg2_nlo = zgmu2_nst() 
              ipdfscale=1
              call strfst(x1t,x2t,sf)
              xnorm = xfact * x1x2j * zgw4 * zg2_nlo * xphsp_ns
              xnorm = xnorm * (1-gfcl(jproc))
              call frest(s,x,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecmccl)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmccl(jp,i)
                    enddo
                  enddo
                enddo
              enddo
            endif
          endif
c
c MC subtraction term: soft ME
c
          if(gfsf(jproc).lt.1.d0)then
            xtmp=1.d0
            x1t=x1soft(x1,x2,x,yi)
            x2t=x2soft(x1,x2,x,yi)
            if(x1t.lt.1.and.x2t.lt.1)then
              x1x2j = x1x2jac(x1,x2,x,yi)
              call invar_in(xm12,xm22,sx,xtmp,yi,cth1,cth2,str,
     #                      tk,uk,q1q,q2q,xinv)
              zg2_nlo = zgmu2_nst() 
              ipdfscale=1
              call strfst(x1t,x2t,sf)
              xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_s 
              xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
              xnorm = xnorm * (1-gfsf(jproc))
              call frest(sx,xtmp,yi,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecmcsf)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      jp=idrmap(j,jproc,itt)
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                  sf(j,jproc,itype,i,itt)*xnorm*
     #                  vecmcsf(jp,i)*s_in(xinv,jproc)
                    enddo
                  enddo
                enddo
              enddo
c
c MC subtraction term: soft-collinear ME
c
              if(gfcl(jproc).lt.1.d0)then
                if(yi.gt.0.d0)then
                  ytmp=1.d0
                  xfact=1.d0/( xii*(1-yi) )
                else
                  ytmp=-1.d0
                  xfact=1.d0/( xii*(1+yi) )
                endif
                call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                        tk,uk,q1q,q2q,xinv)
                xnorm = xfact * x1x2j * zgw4 * zg2_nlo * xphsp_s
                xnorm = - xnorm * (1-gfsf(jproc)) * (1-gfcl(jproc))
                call frest(sx,xtmp,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                     jproc,vecmcsc)
                do i=ichmin,ichmax
                  do j=1,idrmax(jproc,i)
                    do itype=1,itypemax(jproc)
                      do itt=ittmin,ittmax
                        jp=idrmap(j,jproc,itt)
                        vv(j,jproc,itype,i,itt)=
     #                    vv(j,jproc,itype,i,itt)+
     #                    sf(j,jproc,itype,i,itt)*xnorm*vecmcsc(jp,i)
                      enddo
                    enddo
                  enddo
                enddo
              endif
            endif
          endif
        enddo
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yi.gt.1-deltai) then
        ytmp = 1.d0
c The arguments of the pdf, see (A.43) in FW; x1, x2 are called z1, z2 there 
        x1t = x1soft(x1,x2,x,yi)/x
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invar_in(xm12,xm22,s,x,ytmp,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,xinv)
          zg2_nlo = zgmu2_nst()
          call strfst(x1t,x2t,sf)
          xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1-yi) ) 
c The following term comes from (5.7) in FKS; the 1/xi_c term has 
c the same kinematics as the collinear counter-term in the real event, 
c and is therefore conveniently included here, via f2pt
          xlmude = log(s/xmuf2h1)+log(deltai/2)
          xnormc = x1x2j * zgw4 * zg2_nlo * xphspb /(8*pi2 * deltai)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call frest(s,x,ytmp,two,two,tk,uk,q1q,q2q,xinv,jproc,vecre)
            call f2pt(s,q2q,x,x,ytmp,xlmude,jproc,veccl)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    jp=idrmap(j,jproc,itt)
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                sf(j,jproc,itype,i,itt)*
     #                ( xnorm*vecre(jp,i) + xnormc*veccl(jp,i) )
                  enddo
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      if(yi.lt.-1+deltai) then
        ytmp = -1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)/x
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/x
          call invar_in(xm12,xm22,s,x,ytmp,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,xinv)
          zg2_nlo = zgmu2_nst()
          call strfst(x1t,x2t,sf)
          xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1+yi) ) 
          xlmude = log(s/xmuf2h2)+log(deltai/2)
          xnormc = x1x2j * zgw4 * zg2_nlo * xphspb /(8*pi2 * deltai)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call frest(s,x,ytmp,two,two,tk,uk,q1q,q2q,xinv,jproc,vecre)
            call f2pt(s,q1q,x,x,ytmp,xlmude,jproc,veccl)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    jp=idrmap(j,jproc,itt)
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                sf(j,jproc,itype,i,itt)*
     #                ( xnorm*vecre(jp,i) + xnormc*veccl(jp,i) )
                  enddo
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
          call invar_in(xm12,xm22,sx,xtmp,yi,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,xinv)
          zg2_nlo = zgmu2_nst()
          call strfst(x1t,x2t,sf)
          xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_s 
          xnorm = - xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          xnormb = x1x2j * zgw4 * xphspb /(2*xicut)
          xnormsv = x1x2j * zgw4 * zg2_nlo * xphspb / 
     #             (8*pi2 * 2*xicut)
          do jproc=loproc,maproc
            call frest(sx,xtmp,yi,two,two,tk,uk,q1q,q2q,xinv,
     #                 jproc,vecre)
            call fstrn(sx,q1q,jproc,vecbrn)
            call f2st(sx,q1q,jproc,vec2sv)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                xintsvc(j,jproc,i)=xnorm*vecre(j,i)*s_in(xinv,jproc)
                xbornvc(j,jproc,i)=xnormb*vecbrn(j,i)*s_in(xinv,jproc)
                xsvvc(j,jproc,i)=xnormsv*vec2sv(j,i)*s_in(xinv,jproc)
                xcsvc(j,jproc,i)=0.d0
              enddo
            enddo
          enddo
c
          if(yi.gt.1-deltai) then
            ytmp = 1.d0
            call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                    tk,uk,q1q,q2q,xinv)
            xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_s
            xnorm = - xnorm * 1.d0/xii*( - 1/(1-yi) ) 
            xlmude = log(sx/xmuf2h1)+log(deltai/2)
            xnormc = x1x2j * zgw4 * zg2_nlo * xphspb /
     #               (8*pi2 * deltai)
            xnormc = -xnormc/xii
            do jproc=loproc,maproc
              call frest(sx,xtmp,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecre)
              call f2pt(sx,q2q,x,xtmp,ytmp,xlmude,jproc,veccl)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                               xnorm*vecre(j,i)
                  xcsvc(j,jproc,i)=xnormc*veccl(j,i)
                enddo
              enddo
            enddo
          endif
c
          if(yi.lt.-1+deltai) then
            ytmp = -1.d0
            call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                    tk,uk,q1q,q2q,xinv)
            xnorm = x1x2j * zgw4 * zg2_nlo * xphsp_s
            xnorm = - xnorm * 1.d0/xii*( - 1/(1+yi) ) 
            xlmude = log(sx/xmuf2h2)+log(deltai/2)
            xnormc = x1x2j * zgw4 * zg2_nlo * xphspb /
     #               (8*pi2 * deltai)
            xnormc = -xnormc/xii
            do jproc=loproc,maproc
              call frest(sx,xtmp,ytmp,two,two,tk,uk,q1q,q2q,xinv,
     #                   jproc,vecre)
              call f2pt(sx,q1q,x,xtmp,ytmp,xlmude,jproc,veccl)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                               xnorm*vecre(j,i)
                  xcsvc(j,jproc,i)=xcsvc(j,jproc,i)+
     #                             xnormc*veccl(j,i)
                enddo
              enddo
            enddo
          endif
          do jproc=loproc,maproc
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    jp=idrmap(j,jproc,itt)
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                sf(j,jproc,itype,i,itt)*
     #                ( xintsvc(jp,jproc,i)+xbornvc(jp,jproc,i)+
     #                  xsvvc(jp,jproc,i)+xcsvc(jp,jproc,i) )
                  enddo
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      call checkst(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invar_in(xm12,xm22,sx,xtmp,ytmp,cth1,cth2,str,
     #                tk,uk,q1q,q2q,xinv)
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        ycm = 0.5d0*log(x1t/x2t)
        tau=x*tau
        if(idec.eq.0)then
          ps=sx
          px=xtmp
          pyi=ytmp
          pcth1=cth1
          pcth2=cth2
          pphii=-1.d8
          pyj=-1.d8
          pphij=-1.d8
        endif
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
      tot5bs_in=abs(xint)
c
      return
      end


C Empty now, but for zeta-subtraction, to be done,
C if the number of negative weight events for single top
C happens to be too large
      function tot5bz_in(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bz_in,xs,xx,xyi,xcth1,xcth2,xjac
      tot5bz_in=0.d0
      return
      end


      function sig5azw_out(xx)
      implicit none
      real * 8 sig5azw_out,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,rohlim,ximax0,ximin0,tmp,s,rolim,ymax,ymin,zzz,
     #  x,yj,ro,yi,phij,xmi2,xalfaem,tot5a_out
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> yj, xx(5) --> yi,
c xx(6) --> phij
c
      xjac = 1.d0
      rohlim=(sqrt(xm12)+sqrt(xm22))**2/sh
c
      ximax0 = rohlim**(-nsamp)
      ximin0 = 1
      tmp = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac = xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      s = tau*sh
      rolim = rohlim/tau
c
      ymax= -log(tau)/2
      ymin=  log(tau)/2
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*max(1-rolim,xicut)
      xjac = xjac * max(1-rolim,xicut)
c
c To improve convergence in the final state collinear region
c
      zzz=pi*(tiny+(1.d0-tiny)*xx(4)**2)
      xjac=xjac*xx(4)*2*pi
      yj=cos(zzz)
      xjac=xjac*sin(zzz)
c
      ro = 2*(xm12+xm22)/s-(xm12-xm22)**2/s**2
      call zzchvar(xx(5),yi,xjac,ro)
c
      phij = xx(6) * 2 * pi
      xjac = xjac * 2* pi
c
c Hard coded choice for scale of running e.m. coupling: mtop
      xmi2=xm12
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
c
      sig5azw_out = tot5a_out(s,x,yj,phij,yi,xjac)
      return
      end


      function tot5a_out(s,x,yj,phij,yi,xjac)
      implicit none
      real * 8 tot5a_out,tot5as_out,tot5az_out,s,x,yj,phij,yi,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5as_out(s,x,yj,phij,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5az_out(s,x,yj,phij,yi,xjac)
      else
        write(*,*)'Fatal error in tot5a_out:',isubttype
        stop
      endif
      tot5a_out=tmp
      return
      end


      function tot5as_out(xs,xx,xyj,xphij,xyi,xjac)
      implicit none
      real * 8 tot5as_out,xs,xx,xyj,xphij,xyi,xjac
      real * 8 pi,pi2,zero,phii,two,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
      parameter (phii=0.d0)
      parameter (two=2.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      common/cpsave/ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      real * 8 vv(8,2:3,9,2,2),vvs(8,2:3,9,2,2)
      common/cvv/vv
      common/cvvs/vvs
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      real * 8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      logical lzone(8,2)
      real * 8 xinv(5)
      real * 8 sf(8,2:3,9,2,2)
      real * 8 vecre(8,2),vecmc(8,2),xmcz(8,2)
      real * 8 vecmccl(8,2),vecmcsf(8,2),vecmcsc(8,2)
      real * 8 gfsf(2:3),gfcl(2:3)
      real * 8 s,x,yj,phij,yi,xii,rolim,xphsp_ns,zgw4,x1,x2,tk,uk,
     # q1q,q2q,xij,zg2_nlo,zgmu2_nst,xnorm,s_out,zg2_mc,zgmu2_st,
     # x1t,x2t,xsum,dummy,xint,xtmp,ytmp,gfactsf,gfactcl,xphsp_sc
      integer loproco,maproco,jproc,j,itype,i,itt,ipdfscale,ileg,
     # iret,i2b,itoosoftkin,idec
      common/cwotproc/loproco,maproco
      common/cipdfscale/ipdfscale
      common/cidec/idec
c
      s = xs
      x = xx
      yj = xyj
      phij = xphij
      yi = xyi
      xii = 1-x
      rolim=(sqrt(xm12)+sqrt(xm22))**2/s
c Three-body phase-space factors; xij is inserted below
      xphsp_ns = xjac * s/(512*pi**4) * 1/(2-xii*(1-yj))
      xphsp_sc = xjac * s/(512*pi**4) * 1/2.d0
c
      zgw4 = (ze2/sthw2)**2
c
      do jproc=2,3
        do i=1,2
          do j=1,8
            do itype=1,9
              do itt=1,2
                vv(j,jproc,itype,i,itt)=0.d0
                vvs(j,jproc,itype,i,itt)=0.d0
              enddo
            enddo
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(xii.lt.1-rolim) then
        call invar_out(xm12,s,x,yi,phii,yj,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        zg2_nlo = zgmu2_nst() 
        ipdfscale=1
        call strfst(x1,x2,sf)
c This corresponds to Eq. (4.65) in FKS
        xnorm = zgw4 * zg2_nlo * xphsp_ns
        xnorm = xnorm * xij * 1.d0/xii * 1.d0/(1-yj) 
        do jproc=loproco,maproco
          call frest(s,x,two,xij,yj,tk,uk,q1q,q2q,xinv,jproc,vecre)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              do itype=1,itypemax(jproc)
                do itt=ittmin,ittmax
                  vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #              sf(j,jproc,itype,i,itt)*xnorm*
     #              vecre(j,i)*s_out(xinv,jproc)
                enddo
              enddo
            enddo
          enddo
        enddo
c
c MC subtraction terms: pure MC
c
        zg2_mc = zgmu2_st()
        ipdfscale=2
        x1t=x1
        x2t=x2
        call strfst(x1t,x2t,sf)
        xnorm = zgw4 * zg2_mc * xphsp_ns
        xnorm = xnorm * xij * 1.d0/xii * 1.d0/(1-yj) 
        do jproc=loproco,maproco
          do ileg=3,4
            call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,zero,zero,
     #        yj,phii,phij,xij,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #        lzone,xmcz,vecmc)
            gfsf(jproc)=gfactsf
            gfcl(jproc)=gfactcl
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                if(lzone(j,i))then
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmc(j,i)
                    enddo
                  enddo
                endif
              enddo
            enddo
          enddo
        enddo
c
c MC subtraction term: collinear ME
c
        do jproc=loproco,maproco
          if(gfcl(jproc).lt.1.d0)then
            ytmp=1.d0
            call invar_out(xm12,s,x,yi,phii,ytmp,phij,
     #                     tk,uk,q1q,q2q,xij,xinv)
            zg2_nlo = zgmu2_nst() 
            ipdfscale=1
            call strfst(x1t,x2t,sf)
            xnorm = zgw4 * zg2_nlo * xphsp_sc
            xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
            xnorm = xnorm * (1-gfcl(jproc))
            call frest(s,x,two,xij,ytmp,tk,uk,q1q,q2q,xinv,
     #                 jproc,vecmccl)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                sf(j,jproc,itype,i,itt)*xnorm*vecmccl(j,i)
                  enddo
                enddo
              enddo
            enddo
          endif
c
c MC subtraction term: soft ME
c
          if(gfsf(jproc).lt.1.d0)then
            xtmp=1.d0
            call invar_out(xm12,s,xtmp,yi,phii,yj,phij,
     #                     tk,uk,q1q,q2q,xij,xinv)
            zg2_nlo = zgmu2_nst() 
            ipdfscale=1
            call strfst(x1t,x2t,sf)
            xnorm = zgw4 * zg2_nlo * xphsp_sc
            xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
            xnorm = xnorm * (1-gfsf(jproc))
            call frest(s,xtmp,two,xij,yj,tk,uk,q1q,q2q,
     #                 xinv,jproc,vecmcsf)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                sf(j,jproc,itype,i,itt)*xnorm*
     #                vecmcsf(j,i)*s_out(xinv,jproc)
                  enddo
                enddo
              enddo
            enddo
c
c MC subtraction term: soft-collinear ME
c
            if(gfcl(jproc).lt.1.d0)then
              ytmp=1.d0
              call invar_out(xm12,s,xtmp,yi,phii,ytmp,phij,
     #                       tk,uk,q1q,q2q,xij,xinv)
              xnorm = zgw4 * zg2_nlo * xphsp_sc
              xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
              xnorm = - xnorm * (1-gfsf(jproc)) * (1-gfcl(jproc))
              call frest(s,xtmp,two,xij,ytmp,tk,uk,q1q,q2q,
     #                   xinv,jproc,vecmcsc)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)-
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmcsc(j,i)
                    enddo
                  enddo
                enddo
              enddo
            endif
          endif
        enddo
      endif
c
      call checkst(xsum,dummy,iret)
      if(iret.eq.1)then
        call invar_out(xm12,s,x,yi,phii,yj,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        if(idec.eq.0)then
          ps=s
          px=x
          pyi=yi
          pphii=phii
          pyj=yj
          pphij=phij
          pcth1=-1.d8
          pcth2=-1.d8
        endif
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invar_out(xm12,s,xtmp,yi,phii,ytmp,phij,
     #                   tk,uk,q1q,q2q,xij,xinv)
          if(idec.eq.0)then
            ps=s
            px=xtmp
            pyi=yi
            pphii=phii
            pyj=ytmp
            pphij=phij
            pcth1=-1.d8
            pcth2=-1.d8
          endif
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5as_out=abs(xint)
c
      return
      end


C Empty now, but for zeta-subtraction, to be done,
C if the number of negative weight events for single top
C happens to be too large
      function tot5az_out(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5az_out,xs,xx,xyi,xcth1,xcth2,xjac
      tot5az_out=0.d0
      return
      end


      function sig5bzw_out(xx)
      implicit none
      real * 8 sig5bzw_out,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      integer nsamp
      common/samp/nsamp
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,rohlim,ximax0,ximin0,tmp,s,rolim,ymax,ymin,zzz,
     #  x,yj,ro,yi,phij,xmi2,xalfaem,tot5b_out
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> yj, xx(5) --> yi,
c xx(6) --> phij
c
      xjac = 1.d0
      rohlim=(sqrt(xm12)+sqrt(xm22))**2/sh
c
      ximax0 = rohlim**(-nsamp)
      ximin0 = 1
      tmp = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac = xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      s = tau*sh
      rolim = rohlim/tau
c
      ymax= -log(tau)/2
      ymin=  log(tau)/2
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
c To improve convergence in the soft regions
c
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*max(1-rolim,xicut)
      xjac = xjac * max(1-rolim,xicut)
c
c To improve convergence in the final state collinear region
c
      zzz=pi*(tiny+(1.d0-tiny)*xx(4)**2)
      xjac=xjac*xx(4)*2*pi
      yj=cos(zzz)
      xjac=xjac*sin(zzz)
c
      ro = 2*(xm12+xm22)/s-(xm12-xm22)**2/s**2
      call zzchvar(xx(5),yi,xjac,ro)
c
      phij = xx(6) * 2 * pi
      xjac = xjac * 2* pi
c
c Hard coded choice for scale of running e.m. coupling: mtop
      xmi2=xm12
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmi2)
c
      sig5bzw_out = tot5b_out(s,x,yj,phij,yi,xjac)
      return
      end


      function tot5b_out(s,x,yj,phij,yi,xjac)
      implicit none
      real * 8 tot5b_out,tot5bs_out,tot5bz_out,s,x,yj,phij,yi,
     #  xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5bs_out(s,x,yj,phij,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5bz_out(s,x,yj,phij,yi,xjac)
      else
        write(*,*)'Fatal error in tot5b_out:',isubttype
        stop
      endif
      tot5b_out=tmp
      return
      end


      function tot5bs_out(xs,xx,xyj,xphij,xyi,xjac)
      implicit none
      real * 8 tot5bs_out,xs,xx,xyj,xphij,xyi,xjac
      real * 8 pi,pi2,zero,phii,two,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
      parameter (phii=0.d0)
      parameter (two=2.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      include 'stpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      common/cpsave/ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      real * 8 vv(8,2:3,9,2,2),vvs(8,2:3,9,2,2)
      common/cvv/vv
      common/cvvs/vvs
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      real * 8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      logical lzone(8,2)
      real * 8 xinv(5)
      real * 8 sf(8,2:3,9,2,2)
      real * 8 xintsvc(1:8,2:3,2),xbornvc(1:8,2:3,2)
      real * 8 xsvvc(1:8,2:3,2)
      real * 8 vecre(8,2),vecbrn(8,2),vec2sv(8,2)
      real * 8 vecmc(8,2),xmcz(8,2)
      real * 8 vecmccl(8,2),vecmcsf(8,2),vecmcsc(8,2)
      real * 8 gfsf(2:3),gfcl(2:3)
      real * 8 s,x,yj,phij,yi,xii,ro,beta,xphsp_ns,xphsp_sc,xphspb,
     # zgw4,xint,x1,x2,tk,uk,q1q,q2q,xij,zg2_nlo,zgmu2_nst,xnorm,
     # s_out,ytmp,xtmp,xnormb,xnormsv,rolim,zg2_mc,zgmu2_st,x1t,x2t,
     # xsum,dummy,gfactsf,gfactcl
      integer loproco,maproco,jproc,j,itype,i,itt,ileg,iret,
     # ipdfscale,idec
      common/cwotproc/loproco,maproco
      common/cipdfscale/ipdfscale
      common/cidec/idec
c
      s = xs
      x = xx
      yj = xyj
      phij = xphij
      yi = xyi
      xii = 1-x
      ro = 2*(xm12+xm22)/s-(xm12-xm22)**2/s**2
      beta = sqrt(1-ro)
      rolim=(sqrt(xm12)+sqrt(xm22))**2/s
c Three-body phase-space factors; xij is inserted below
      xphsp_ns = xjac * s/(512*pi**4) * 1/(2-xii*(1-yj))
      xphsp_sc = xjac * s/(512*pi**4) * 1/2.d0
c Born phase-space normalization
      xphspb = xjac * beta/(32*pi2)
c
      zgw4 = (ze2/sthw2)**2
c
      do jproc=2,3
        do i=1,2
          do j=1,8
            do itype=1,9
              do itt=1,2
                vv(j,jproc,itype,i,itt)=0.d0
                vvs(j,jproc,itype,i,itt)=0.d0
              enddo
            enddo
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(xii.lt.1-rolim) then
        call invar_out(xm12,s,x,yi,phii,yj,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
c
c MC subtraction terms: pure MC
c
        zg2_mc = zgmu2_st()
        ipdfscale=2
        x1t=x1
        x2t=x2
        call strfst(x1t,x2t,sf)
        xnorm = zgw4 * zg2_mc * xphsp_ns
        xnorm = xnorm * xij * 1.d0/xii * 1.d0/(1-yj) 
        do jproc=loproco,maproco
          do ileg=3,4
            call xmcsust(jproc,ileg,x1,x2,xm12,s,x,yi,zero,zero,
     #        yj,phii,phij,xij,tk,uk,q1q,q2q,gfactsf,gfactcl,
     #        lzone,xmcz,vecmc)
            gfsf(jproc)=gfactsf
            gfcl(jproc)=gfactcl
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                if(lzone(j,i))then
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmc(j,i)
                    enddo
                  enddo
                endif
              enddo
            enddo
          enddo
        enddo
c
c MC subtraction term: collinear ME
c
        do jproc=loproco,maproco
          if(gfcl(jproc).lt.1.d0)then
            ytmp=1.d0
            call invar_out(xm12,s,x,yi,phii,ytmp,phij,
     #                     tk,uk,q1q,q2q,xij,xinv)
            zg2_nlo = zgmu2_nst() 
            ipdfscale=1
            call strfst(x1t,x2t,sf)
            xnorm = zgw4 * zg2_nlo * xphsp_sc
            xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
            xnorm = xnorm * (1-gfcl(jproc))
            call frest(s,x,two,xij,ytmp,tk,uk,q1q,q2q,xinv,
     #                 jproc,vecmccl)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                sf(j,jproc,itype,i,itt)*xnorm*vecmccl(j,i)
                  enddo
                enddo
              enddo
            enddo
          endif
c
c MC subtraction term: soft ME
c
          if(gfsf(jproc).lt.1.d0)then
            xtmp=1.d0
            call invar_out(xm12,s,xtmp,yi,phii,yj,phij,
     #                     tk,uk,q1q,q2q,xij,xinv)
            zg2_nlo = zgmu2_nst() 
            ipdfscale=1
            call strfst(x1t,x2t,sf)
            xnorm = zgw4 * zg2_nlo * xphsp_sc
            xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
            xnorm = xnorm * (1-gfsf(jproc))
            call frest(s,xtmp,two,xij,yj,tk,uk,q1q,q2q,
     #                 xinv,jproc,vecmcsf)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                do itype=1,itypemax(jproc)
                  do itt=ittmin,ittmax
                    vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                sf(j,jproc,itype,i,itt)*xnorm*
     #                vecmcsf(j,i)*s_out(xinv,jproc)
                  enddo
                enddo
              enddo
            enddo
c
c MC subtraction term: soft-collinear ME
c
            if(gfcl(jproc).lt.1.d0)then
              ytmp=1.d0
              call invar_out(xm12,s,xtmp,yi,phii,ytmp,phij,
     #                       tk,uk,q1q,q2q,xij,xinv)
              xnorm = zgw4 * zg2_nlo * xphsp_sc
              xnorm = xnorm * xij * 1.d0/( xii*(1-yj) )
              xnorm = - xnorm * (1-gfsf(jproc)) * (1-gfcl(jproc))
              call frest(s,xtmp,two,xij,ytmp,tk,uk,q1q,q2q,
     #                   xinv,jproc,vecmcsc)
              do i=ichmin,ichmax
                do j=1,idrmax(jproc,i)
                  do itype=1,itypemax(jproc)
                    do itt=ittmin,ittmax
                      vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #                  sf(j,jproc,itype,i,itt)*xnorm*vecmcsc(j,i)
                    enddo
                  enddo
                enddo
              enddo
            endif
          endif
        enddo
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yj.gt.1-deltao.and.xii.lt.1-rolim) then
        ytmp = 1.d0
        call invar_out(xm12,s,x,yi,phii,ytmp,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        zg2_nlo = zgmu2_nst()
        call strfst(x1,x2,sf)
        xnorm = zgw4 * zg2_nlo * xphsp_sc
        xnorm = xnorm * xij * 1.d0/xii * ( - 1/(1-yj) ) 
        do jproc=loproco,maproco
          call frest(s,x,two,xij,ytmp,tk,uk,q1q,q2q,xinv,jproc,vecre)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              do itype=1,itypemax(jproc)
                do itt=ittmin,ittmax
                  vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #              sf(j,jproc,itype,i,itt)*xnorm*vecre(j,i)
                enddo
              enddo
            enddo
          enddo
        enddo
      endif
c
c     Soft part of the counter-event:
c
      if(xii.lt.xicut) then
        xtmp = 1.d0
        call invar_out(xm12,s,xtmp,yi,phii,yj,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        zg2_nlo = zgmu2_nst()
        call strfst(x1,x2,sf)
        xnorm = zgw4 * zg2_nlo * xphsp_sc
        xnorm = - xnorm * xij * 1.d0/xii * 1/(1-yj)
        xnormb = zgw4 * xphspb /(2*xicut)
        xnormsv = zgw4 * zg2_nlo * xphspb / 
     #           (8*pi2 * 2*xicut)
        do jproc=loproco,maproco
          call frest(s,xtmp,two,xij,yj,tk,uk,q1q,q2q,
     #               xinv,jproc,vecre)
          call fstrn(s,q1q,jproc,vecbrn)
          call f2st(s,q1q,jproc,vec2sv)
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              xintsvc(j,jproc,i)=xnorm*vecre(j,i)*s_out(xinv,jproc)
              xbornvc(j,jproc,i)=xnormb*vecbrn(j,i)*s_out(xinv,jproc)
              xsvvc(j,jproc,i)=xnormsv*vec2sv(j,i)*s_out(xinv,jproc)
            enddo
          enddo
        enddo
c
        if(yj.gt.1-deltao) then
          ytmp = 1.d0
          call invar_out(xm12,s,xtmp,yi,phii,ytmp,phij,
     #                   tk,uk,q1q,q2q,xij,xinv)
          xnorm = zgw4 * zg2_nlo * xphsp_sc
          xnorm = - xnorm * xij * 1.d0/xii *( - 1/(1-yj) ) 
          do jproc=loproco,maproco
            call frest(s,xtmp,two,xij,ytmp,tk,uk,q1q,q2q,
     #                 xinv,jproc,vecre)
            do i=ichmin,ichmax
              do j=1,idrmax(jproc,i)
                xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                             xnorm*vecre(j,i)
              enddo
            enddo
          enddo
        endif
c Now all counter events are being collected
        do jproc=loproco,maproco
          do i=ichmin,ichmax
            do j=1,idrmax(jproc,i)
              do itype=1,itypemax(jproc)
                do itt=ittmin,ittmax
                  vv(j,jproc,itype,i,itt)=vv(j,jproc,itype,i,itt)+
     #              sf(j,jproc,itype,i,itt)*
     #              ( xintsvc(j,jproc,i)+xbornvc(j,jproc,i)+
     #                xsvvc(j,jproc,i) ) 
                enddo
              enddo
            enddo
          enddo
        enddo
      endif
c
      call checkst(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invar_out(xm12,s,xtmp,yi,phii,ytmp,phij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        if(idec.eq.0)then
          ps=s
          px=xtmp
          pyi=yi
          pphii=phii
          pyj=ytmp
          pphij=phij
          pcth1=-1.d8
          pcth2=-1.d8
        endif
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
      tot5bs_out=abs(xint)
c
      return
      end


C Empty now, but for zeta-subtraction, to be done,
C if the number of negative weight events for single top
C happens to be too large
      function tot5bz_out(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bz_out,xs,xx,xyi,xcth1,xcth2,xjac
      tot5bz_out=0.d0
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
      subroutine sprfst()
c This routine is called by run_spring; the entry is dummy, all the 
c parameters must be passed through common blocks
      implicit none
      integer iunit
      parameter (iunit=22)
      real*8 xone
      parameter (xone=1.d0)
      real*8 ycm,tau
      common/x1x2/ycm,tau
      integer i0,jproc0,itype0,ich0,itt0
      common/cidproc/i0,jproc0,itype0,ich0,itt0
      integer idec
      common/cidec/idec
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      integer iret
      real*8 ycm0
c
      call xost(iret)
      if(iret.eq.1)then
        if(i0.lt.1.or.i0.gt.idrmax(jproc0,ich0))then
          write(*,*)'Fatal error in sprfst'
          stop
        endif
        if(idec.eq.0)call getspincost()
        ycm0=ycm
        call getx1x2(tau,ycm0)
        call getmst(tau,ycm0)
        call store_evenst(iunit,xone)
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


      subroutine getmst(xtau,xycm)
      implicit none
      real*8 xtau,xycm
      include 'stpcblks.h'
      real*8 pi
      parameter (pi=3.14159265358979312D0)
      integer i,imax,itype
      real*8 tau,ycm,theta,cth,sth,fk88random,sqsh,ycmnew
      real*8 x1,x2
      common/cx1x2/x1,x2
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_lb(8,4)
      common/cxmomlb/xmom_lb
      real*8 xmom_prime(8,4)
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
      imax=5
      if(idec.eq.0)imax=8
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
      if(ichkmom.eq.0)call checkmst(xmom_cm,sh,0.d0,3,2)
c determine colour connections
      call getcolcost()
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
        call put_on_shest(ycm,ycmnew)
        do i=1,imax
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(ichkmom.eq.0)then
        itype=idec+1
        call checkmst(xmom_lb,sh,-ycmnew,2,itype)
      endif
      call momnewformst()
      return
      end


      subroutine momnewformst()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(8,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:8)
      common/procmass/xmss
      integer i
c
      do i=1,8
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


      subroutine getcolcost()
c Determines colour connections. In the case of single-top production,
c this is almost trivial, and it amounts to determine whether, for H
c events, the gluon is radiated from the heavy- or light-quark weak 
c current. The colour code iccode is a two-digit number nm, where 
c n and m identify the quarks that are colour connected (i.e., those
c that are not connected to the gluon for H events); n and m can take
c the values 1..5, assigned according to NLO labeling conventions.
c In the case of S events there are two pairs of colour-connected quarks,
c and in order to define iccode we conventionally take the pair to which 
c the top belongs 
      implicit none
      include 'stpcblks.h'
      real*8 crnd,fk88random,s,dotprod,tk,uk,q1q,q2q,s2,q1c,q2c,
     # w1,w2,t12,t13,t14,t15,t23,t24,t25,t34,t35,t45,Prop1_1,
     # Prop2_1,rMApMB2_s1,MApMB2,rMCpMD2_s1,MCpMD2,xab,xcd,sum,
     # Prop1_3,Prop2_3,rMApMB2_s3,rMCpMD2_s3,rMApMB2_t1,rMCpMD2_t1,
     # Prop1_2,Prop2_2,rMApMB2_t2,rMCpMD2_t2,rMApMB2_t3,rMCpMD2_t3,
     # Prop1_4,Prop2_4,rMApMB2_t4,rMCpMD2_t4
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer iccrelabel
      integer i0,jproc0,itype0,ich0,itt0
      common/cidproc/i0,jproc0,itype0,ich0,itt0
      integer i1hpro
      common/ci1hpro/i1hpro
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      if(xmom_cm(3,4).eq.0.d0)then
c 2-body kinematics
        if(ich0.eq.1)then
          iccode=45
        elseif( (jproc0.eq.2.and.ich0.eq.2) .and.
     #          (i0.eq.1.or.i0.eq.2) )then
          iccode=14
        elseif( (jproc0.eq.2.and.ich0.eq.2) .and.
     #          (i0.eq.3.or.i0.eq.4) )then
          iccode=24
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.ge.1.and.i0.le.4) )then
          iccode=14
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.ge.5.and.i0.le.8) )then
          iccode=24
        else
          write(*,*)'Error #1 in getcolcost',
     #                 i0,jproc0,itype0,ich0,itt0
          stop
        endif
      else
c 3-body kinematics
        if( (jproc0.eq.3.and.ich0.eq.1) .and.
     #      (i0.eq.1.or.i0.eq.2) )then
          iccode=45
        elseif( (jproc0.eq.3.and.ich0.eq.1) .and.
     #      (i0.eq.3.or.i0.eq.4) )then
          iccode=45
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.eq.2.or.i0.eq.4) )then
          iccode=25
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.eq.5.or.i0.eq.7) )then
          iccode=15
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.eq.1.or.i0.eq.3) )then
          iccode=14
        elseif( (jproc0.eq.3.and.ich0.eq.2) .and.
     #          (i0.eq.6.or.i0.eq.8) )then
          iccode=24
        else
          if( jproc0.ne.2 .or.
     #        (ich0.eq.1.and.(i0.ne.1.and.i0.ne.3)) .or.
     #        (ich0.eq.2.and.(i0.le.0.or.i0.gt.4)) )then
            write(*,*)'Error #2 in getcolcost',
     #                   i0,jproc0,itype0,ich0,itt0
            stop
          endif
          crnd=fk88random(ifk88seed)
          s=   2*dotprod(xmom_cm(1,1),xmom_cm(1,2),
     #                   xmom_cm(1,3),xmom_cm(1,4),
     #                   xmom_cm(2,1),xmom_cm(2,2),
     #                   xmom_cm(2,3),xmom_cm(2,4))
          tk= -2*dotprod(xmom_cm(1,1),xmom_cm(1,2),
     #                   xmom_cm(1,3),xmom_cm(1,4),
     #                   xmom_cm(3,1),xmom_cm(3,2),
     #                   xmom_cm(3,3),xmom_cm(3,4))
          uk= -2*dotprod(xmom_cm(2,1),xmom_cm(2,2),
     #                   xmom_cm(2,3),xmom_cm(2,4),
     #                   xmom_cm(3,1),xmom_cm(3,2),
     #                   xmom_cm(3,3),xmom_cm(3,4))
          q1q=-2*dotprod(xmom_cm(1,1),xmom_cm(1,2),
     #                   xmom_cm(1,3),xmom_cm(1,4),
     #                   xmom_cm(4,1),xmom_cm(4,2),
     #                   xmom_cm(4,3),xmom_cm(4,4))+xm12
          q2q=-2*dotprod(xmom_cm(2,1),xmom_cm(2,2),
     #                   xmom_cm(2,3),xmom_cm(2,4),
     #                   xmom_cm(5,1),xmom_cm(5,2),
     #                   xmom_cm(5,3),xmom_cm(5,4))+xm22
          s2 = s+tk+uk
          q1c = xm12 + xm22 - s - tk - q1q
          q2c = xm12 + xm22 - s - uk - q2q
          w1  = xm12 - q1q + q2q - tk
          w2  = xm22 - q2q + q1q - uk
c
          t12 = s
          t13 = q1q-xm12
          t14 = q1c-xm22
          t15 = tk
          t23 = q2c-xm12
          t24 = q2q-xm22
          t25 = uk
          t34 = s2-xm12-xm22
          t35 = w1-xm12
          t45 = w2-xm22
c
          if(ich0.eq.1.and.i0.eq.1)then
            Prop1_1 = 1d0/(t12+t15+t25-xmW2)      
            Prop2_1 = 1d0/(t12-xmW2)      
            rMApMB2_s1 = MApMB2(xm12,t12,t13,t14,t15,
     #                          t23,t24,t25,t34,t35,t45)
            rMCpMD2_s1 = MCpMD2(xm12,t12,t13,t14,t15,
     #                          t23,t24,t25,t34,t35,t45)
            xab = Prop1_1**2*rMApMB2_s1
            xcd = Prop2_1**2*rMCpMD2_s1
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=45
              else
                iccode=12
              endif
            else
              iccode=45
            endif
          elseif(ich0.eq.1.and.i0.eq.3)then
            Prop1_3 = 1d0/(t12+t15+t25-xmW2)      
            Prop2_3 = 1d0/(t12-xmW2)      
            rMApMB2_s3 = MApMB2(xm12,t12,t23,t24,t25,
     #                          t13,t14,t15,t34,t35,t45)
            rMCpMD2_s3 = MCpMD2(xm12,t12,t23,t24,t25,
     #                          t13,t14,t15,t34,t35,t45)
            xab = Prop1_3**2*rMApMB2_s3
            xcd = Prop2_3**2*rMCpMD2_s3
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=45
              else
                iccode=12
              endif
            else
              iccode=45
            endif
          elseif(ich0.eq.2.and.i0.eq.1)then
            Prop1_1 = 1d0/(t24+t25+t45-xmW2)      
            Prop2_1 = 1d0/(t24-xmW2)
            rMApMB2_t1 = MApMB2(xm12,t24,t23,t12,t25,
     #                          t34,t14,t45,t13,t35,t15)
            rMCpMD2_t1 = MCpMD2(xm12,t24,t23,t12,t25,
     #                          t34,t14,t45,t13,t35,t15)
            xab = Prop1_1**2*rMApMB2_t1
            xcd = Prop2_1**2*rMCpMD2_t1
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=14
              else
                iccode=25
              endif
            else
              iccode=14
            endif
          elseif(ich0.eq.2.and.i0.eq.2)then
            Prop1_2 = 1d0/(t24+t45+t25-xmW2)      
            Prop2_2 = 1d0/(t24-xmW2)      
            rMApMB2_t2 = MApMB2(xm12,t24,t34,t14,t45,
     #                          t23,t12,t25,t13,t35,t15)
            rMCpMD2_t2 = MCpMD2(xm12,t24,t34,t14,t45,
     #                          t23,t12,t25,t13,t35,t15)
            xab = Prop1_2**2*rMApMB2_t2
            xcd = Prop2_2**2*rMCpMD2_t2
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=14
              else
                iccode=25
              endif
            else
              iccode=14
            endif
          elseif(ich0.eq.2.and.i0.eq.3)then
            Prop1_3 = 1d0/(t14+t15+t45-xmW2)      
            Prop2_3 = 1d0/(t14-xmW2)      
            rMApMB2_t3 = MApMB2(xm12,t14,t13,t12,t15,
     #                          t34,t24,t45,t23,t35,t25)
            rMCpMD2_t3 = MCpMD2(xm12,t14,t13,t12,t15,
     #                          t34,t24,t45,t23,t35,t25)
            xab = Prop1_3**2*rMApMB2_t3
            xcd = Prop2_3**2*rMCpMD2_t3
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=24
              else
                iccode=15
              endif
            else
              iccode=24
            endif
          elseif(ich0.eq.2.and.i0.eq.4)then
            Prop1_4 = 1d0/(t14+t45+t15-xmW2)      
            Prop2_4 = 1d0/(t14-xmW2)      
            rMApMB2_t4 = MApMB2(xm12,t14,t34,t24,t45,
     #                          t13,t12,t15,t23,t35,t25)
            rMCpMD2_t4 = MCpMD2(xm12,t14,t34,t24,t45,
     #                          t13,t12,t15,t23,t35,t25)
            xab = Prop1_4**2*rMApMB2_t4
            xcd = Prop2_4**2*rMCpMD2_t4
            sum = xab+xcd
            if(sum.ne.0.d0)then
              if((xab/sum).gt.crnd)then
                iccode=24
              else
                iccode=15
              endif
            else
              iccode=24
            endif
          else
            write(*,*)'Error #3 in getcolcost',
     #                   i0,jproc0,itype0,ich0,itt0
            stop
          endif
        endif
      endif
      if(idec.eq.0)iccode=iccrelabel(iccode)
      return
      end


      function iccrelabel(iccode)
c When the top decays, parton #5 appears in the event record as parton #4,
c and the top decay products are #5,#6,#7. Therefore, the colour connections
c must be relabeled replacing 4 with 5 and 5 with 4
      implicit none
      integer iccrelabel,iccode,ia,ib,iap,ibp,itmp,ifourfive
c
      ia=iccode/10
      ib=iccode-10*ia
      iap=ifourfive(ia)
      ibp=ifourfive(ib)
      if(iap.gt.ibp)then
        itmp=iap
        iap=ibp
        ibp=itmp
      endif
      if(iap.eq.ibp)then
        write(*,*)'Error in iccrelabel',iap,ibp
        stop
      endif
      iccrelabel=10*iap+ibp
      return
      end


      function ifourfive(i)
c Utility for iccrelabel
      implicit none
      integer ifourfive,i,itmp
c
      if(i.ge.1.and.i.le.3)then
        itmp=i
      elseif(i.eq.4)then
        itmp=5
      elseif(i.eq.5)then
        itmp=4
      else
        write(*,*)'Error in ifourfive',i
        stop
      endif
      ifourfive=itmp
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


      subroutine put_on_shest(ycm,ycmnew)
      implicit none
      include 'stpcblks.h'
      integer i2b,i,j,il,in,ib,ii
      real*8 xmss(1:8),xtmp(1:4),xk1tmp(1:4),ytmp1(1:4),ytmp2(1:4),
     #  xavg3(1:3),wvec(1:4),wvec2(1:4)
      real*8 ycm,ycmnew,pi,one,delta_thrs,shat,xkp2prime_norm2,
     #  xkp2prime_norm,xkprime_0,xsign,xnorm_3,delta,gamma,xmprime,
     #  xk1prime_norm,fakemass,xk1tmp_norm,xkprime_norm,xavgnorm,
     #  qw2,qw,xnormsq,xbwnorm,xlepnorm,tmplmass
      parameter (pi=3.14159265358979312D0)
      parameter (one=1.d0)
      parameter (delta_thrs=0.5d-3)
      common/procmass/xmss
      real*8 xmass(-5:21)
      common/parmass/xmass
c top mass and width; top mass squared is stored in fixvar; xmt must
c be used only in those parts of the code relevant to top decay
      real*8 xmt,twidth
      common/ctparam/xmt,twidth
c W mass and width
      real*8 xmw,gaw
      common/cwparam/xmw,gaw
c Lepton masses
      real*8 xlep1mass,xlep2mass
      common/clepmass/xlep1mass,xlep2mass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing FKS parton, 4 is the top or antitop, 5 is the
c other final-state light parton. When the top decays, 6=l, 7=nu, 8=b are 
c the decay products of the top. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+5+6+7+8)=0. In the following, 4+5 will 
c be referred to as the pair
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(8,4)
      common/cxmomprime/xmom_prime
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g. Only observables particles are
c listed here. Therefore, (ip3,ip5)=final-state light partons, ip4=top/tbar
c when the top doesn't decay, and (ip3,ip4)=final-state light partons,
c (ip5,ip6,ip7)=top decay products when the top decays
      integer ip1,ip2,ip3,ip4,ip5,ip6,ip7
      common/ci1part/ip1,ip2,ip3,ip4,ip5,ip6,ip7
c here, ionshell=1 or ionshell=2
      integer ionshell
      common/cionshell/ionshell
      integer ideconsh
      common/cideconsh/ideconsh
      integer ichkmom
      common/cichkmom/ichkmom
      integer idec
      common/cidec/idec
      integer iwidth
      common/ciwidth/iwidth
c 
c If the top doesn't (does) decay, ip5 (ip4) is the non-FKS outgoing parton.
c In terms of momenta, the non-FKS outgoing parton has always label 5
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      xmss(4) = xm1
      if(idec.eq.0)then
        xmss(5) = xmass(ip4)
        if(ideconsh.eq.0)then
          do i=6,8
            xmss(i) = 0.d0
          enddo
        elseif(ideconsh.eq.2)then
          xmss(6) = xlep1mass
          xmss(7) = xlep2mass
          xmss(8) = xmass(5)
        else
          write(*,*)'Error in put_on_shest: unknown ideconsh',ideconsh
          stop
        endif
      elseif(idec.eq.1)then
        xmss(5) = xmass(ip5)
        do i=6,8
          xmss(i) = -1.d10
        enddo
      else
        write(6,*) 'Error in put_on_shest: idec=',idec
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
        call getxmst(shat,ycm,
     #               xmom_cm(1,3),xmss(1),
     #               xmom_cm(2,3),xmss(2),
     #               xmom_prime(1,3),xmom_prime(2,3))
      else
        write(*,*)'Error in put_on_shest: unknown ionshell'
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
c of the masses of the top and the non-FKS parton
      delta=sqrt(xkprime_0**2-xkprime_norm**2)-xmss(4)-xmss(5)
      if(delta.lt.delta_thrs)then
c parton 3-momenta cannot be kept fixed: the total available energy
c is not sufficient; modify 3-momenta of the incoming partons
        gamma=sqrt( (xmss(4)+xmss(5)+delta_thrs)**2+xkprime_norm**2 )+
     #        xmom_prime(3,4)
        if(gamma.lt.(xmss(1)+xmss(2)))then
          write(6,*)'Error #0 in put_on_shest'
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
c that the momenta of the top and the non-FKS parton can be transformed.
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
          write(6,*)'Error #1 in put_on_shest'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        xk1tmp_norm=xnorm_3(xk1tmp)
c xavg is the direction along which the top and non-FKS parton momenta are 
c placed in the new pair rest frame. It is arbitrarily defined by averaging 
c (hence the 1/2 in the definition) the directions of the original top and
c non-FKS parton momenta. It may not have modulus 1, so normalize it
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
          write(6,*)'Error #2 in put_on_shest'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                    xmom_prime(i,3),xmom_prime(i,4))
      enddo
c 
      if(idec.eq.0)then
        il=6
        in=7
        ib=8
        call fillvec(xmom_prime(4,1),xmom_prime(4,2),
     #               xmom_prime(4,3),xmom_prime(4,4),xtmp)
c First deal with the Wb pair; define W momentum, and compute W mass
c (when iwidth=1, W is off shell)
        call vecsst(xmom_cm(il,1),xmom_cm(il,2),
     #              xmom_cm(il,3),xmom_cm(il,4),one,
     #              xmom_cm(in,1),xmom_cm(in,2),
     #              xmom_cm(in,3),xmom_cm(in,4),one,wvec)
        qw2=xnormsq(wvec)
        qw=sqrt(qw2)
        if( ichkmom.eq.0 .and. iwidth.eq.0 .and.
     #      abs(qw/xmw-1.d0).gt.1.d-4 )then
          write(6,*)'Error #3 in put_on_shest'
          write(6,*)qw,il,in
          stop
        endif
        if( ichkmom.eq.0 .and. iwidth.eq.1 .and.
     #      qw.gt.xmt )then
          write(6,*)'Error #4 in put_on_shest'
          write(6,*)qw,il,in
          stop
        endif
        xbwnorm=xm12-2*(xmss(ib)**2+qw2)+(xmss(ib)**2-qw2)**2/xm12
        xbwnorm=sqrt(xbwnorm)/2.d0
        do j=1,3
          xavg3(j)=0.d0
        enddo
        xsign=1.d0
        call xhwulof(xtmp,xmt,wvec,qw,xk1tmp,fakemass)
        xk1tmp_norm=xnorm_3(xk1tmp)
        do j=1,3
          xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
        enddo
        xsign=-1.d0
        call fillvec(xmom_cm(ib,1),xmom_cm(ib,2),
     #               xmom_cm(ib,3),xmom_cm(ib,4),ytmp1)
        call xhwulof(xtmp,xmt,ytmp1,xmss(ib),xk1tmp,fakemass)
        xk1tmp_norm=xnorm_3(xk1tmp)
        do j=1,3
          xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
        enddo
        xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
        do j=1,3
          xavg3(j)=xavg3(j)/xavgnorm
        enddo
        xsign=1.d0
        do j=1,3
          xk1tmp(j)=xsign*xbwnorm*xavg3(j)
        enddo
        xk1tmp(4)=xmt/2.d0*(1+xsign*(qw2-xmss(ib)**2)/xm12)
        call xhwulob(xtmp,xmt,xk1tmp,qw,wvec2,fakemass)
        xsign=-1.d0
        do j=1,3
          xk1tmp(j)=xsign*xbwnorm*xavg3(j)
        enddo
        xk1tmp(4)=xmt/2.d0*(1+xsign*(qw2-xmss(ib)**2)/xm12)
        call xhwulob(xtmp,xmt,xk1tmp,xmss(ib),ytmp2,fakemass)
        call getvec(ytmp2,xmom_prime(ib,1),xmom_prime(ib,2),
     #                    xmom_prime(ib,3),xmom_prime(ib,4))
c Next deal with the lepton pair; W has momentum wvec2
        xlepnorm=qw2-2*(xmss(il)**2+xmss(in)**2)+
     #           (xmss(il)**2-xmss(in)**2)**2/qw2
        xlepnorm=sqrt(xlepnorm)/2.d0
        do j=1,3
          xavg3(j)=0.d0
        enddo
        do i=1,2
          if(i.eq.1)then
            xsign=1.d0
            ii=il
          else
            xsign=-1.d0
            ii=in
          endif
          tmplmass=xmss(ii)
          call fillvec(xmom_cm(ii,1),xmom_cm(ii,2),
     #                 xmom_cm(ii,3),xmom_cm(ii,4),ytmp1)
          call xhwulof(wvec2,qw,ytmp1,tmplmass,xk1tmp,fakemass)
          xk1tmp_norm=xnorm_3(xk1tmp)
          do j=1,3
            xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
          enddo
        enddo
        xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
        do j=1,3
          xavg3(j)=xavg3(j)/xavgnorm
        enddo
        do i=1,2
          if(i.eq.1)then
            xsign=1.d0
            ii=il
          else
            xsign=-1.d0
            ii=in
          endif
          tmplmass=xmss(ii)
          do j=1,3
            xk1tmp(j)=xsign*xlepnorm*xavg3(j)
          enddo
          xk1tmp(4)=qw/2.d0*
     #      (1+xsign*(xmss(il)**2-xmss(in)**2)/qw2)
          call xhwulob(wvec2,qw,xk1tmp,tmplmass,ytmp2,fakemass)
          call getvec(ytmp2,xmom_prime(ii,1),xmom_prime(ii,2),
     #                      xmom_prime(ii,3),xmom_prime(ii,4))
        enddo
      else
        do i=6,8
          do j=1,4
            xmom_prime(i,j)=0.d0
          enddo
        enddo
      endif
c
      if(ichkmom.eq.0)then
        if(idec.eq.0)then
          call checkstec2(xmom_prime,4,6,7,8)
          call checkmst(xmom_prime,shat,0.d0,4,1)
        else
          call checkmst(xmom_prime,shat,0.d0,4,2)
        endif
        if(xmass(1).eq.0.and.xmass(2).eq.0.and.xmass(3).eq.0.and.
     #     xmass(4).eq.0.and.xmass(5).eq.0.and.xmass(21).eq.0.and.
     #     xlep1mass.eq.0.and.xlep2mass.eq.0)then
          call checkonst(1)
        else
          call checkonst(2)
        endif
      endif
      return
      end


      subroutine getxmst(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
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
        write(*,*)'Error # 0 in getxmst'
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
        write(*,*)'Error # 1 in getxmst'
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
            write(*,*)'Mass # 1 too large in getxmst'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmst'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmst'
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


C-----------------------------------------------------------------------
      SUBROUTINE HWSTB4(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN REST FRAME OF PS) INTO PF (IN LAB)
C     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
C-----------------------------------------------------------------------
      DOUBLE PRECISION PF4,FN,PS(5),PI(4),PF(4)
      IF (PS(4).EQ.PS(5)) THEN
        PF(1)= PI(1)
        PF(2)= PI(2)
        PF(3)= PI(3)
        PF(4)= PI(4)
      ELSE
        PF4  = (PI(1)*PS(1)+PI(2)*PS(2)
     &         +PI(3)*PS(3)+PI(4)*PS(4))/PS(5)
        FN   = (PF4+PI(4)) / (PS(4)+PS(5))
        PF(1)= PI(1) + FN*PS(1)
        PF(2)= PI(2) + FN*PS(2)
        PF(3)= PI(3) + FN*PS(3)
        PF(4)= PF4
      END IF
      END


CC-----------------------------------------------------------------------
C      SUBROUTINE HWULF4(PS,PI,PF)
CC-----------------------------------------------------------------------
CC     TRANSFORMS PI (GIVEN IN LAB) INTO PF (IN REST FRAME OF PS)
CC     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
CC-----------------------------------------------------------------------
C      DOUBLE PRECISION PF4,FN,PS(5),PI(4),PF(4)
C      IF (PS(4).EQ.PS(5)) THEN
C        PF(1)= PI(1)
C        PF(2)= PI(2)
C        PF(3)= PI(3)
C        PF(4)= PI(4)
C      ELSE
C        PF4  = (PI(4)*PS(4)-PI(3)*PS(3)
C     &         -PI(2)*PS(2)-PI(1)*PS(1))/PS(5)
C        FN   = (PF4+PI(4)) / (PS(4)+PS(5))
C        PF(1)= PI(1) - FN*PS(1)
C        PF(2)= PI(2) - FN*PS(2)
C        PF(3)= PI(3) - FN*PS(3)
C        PF(4)= PF4
C      END IF
C      END


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


C      function xnormsq(p)
Cc Computes p.p, assuming the energy is the fourth component
C      implicit none
C      real*8 xnormsq,p(1:4),dotprod
Cc
C      xnormsq=dotprod(p(1),p(2),p(3),p(4),p(1),p(2),p(3),p(4))
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


      subroutine vecsst(p1,p2,p3,p4,pfact,q1,q2,q3,q4,qfact,r)
c Weighted sum of the four-vectors p and q. The result is r
      implicit none
      real*8 p1,p2,p3,p4,pfact,q1,q2,q3,q4,qfact,r(1:4)
c
      r(1)=pfact*p1+qfact*q1
      r(2)=pfact*p2+qfact*q2
      r(3)=pfact*p3+qfact*q3
      r(4)=pfact*p4+qfact*q4
      return
      end


      subroutine checkonst(itype)
c Checks that put_on_shest is harmless if masses are zero (itype=1),
c or computes (itype=2) the average of the shifts due to the masses
      real*8 tiny
      parameter (tiny=1.d-4)
      integer itype
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_prime(8,4)
      common/cxmomprime/xmom_prime
      real*8 xmomshifts(4)
      common/cshifts/xmomshifts
      integer i,j,imax,iflag
      integer idec
      common/cidec/idec
c
      if(itype.ne.1.and.itype.ne.2)then
        write(*,*)'Unknown option in checkonst'
        stop
      endif
      iflag=1
      imax=5
      if(idec.eq.0)imax=8
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
        write(*,*)'Error in checkonst'
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


      subroutine xost(iret)
c This routine is called by sprfst; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkst), and no operation is performed
      implicit none
      integer iret,iretvv,iretvvs,iproc,iproclo,iprocma,i,itype,iz,io,
     #  ich,itt,iwh,iflag,ihpro,i1,i2,i3,i4,i5,i1hproo,ip1o,ip2o,ip3o
      real*8 wwx(8,2:3,9,2,2),xsum,xsumabs,xsumvvs,xsumabsvvs,xstsign,
     #  xg,wh,rmax,fk88random
      parameter (iz=0)
      parameter (io=1)
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      integer loproco,maproco
      common/cwotproc/loproco,maproco
      integer ifuntype
      common/cifuntype/ifuntype
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      real*8 vv(8,2:3,9,2,2)
      common/cvv/vv
      real*8 vvs(8,2:3,9,2,2)
      common/cvvs/vvs
      integer iwrong,iwrong1
      common/ciwrong/iwrong,iwrong1
      integer i0,jproc0,itype0,ich0,itt0
      common/cidproc/i0,jproc0,itype0,ich0,itt0
      integer ivbhpro(8,2:3,9,2,2)
      common/civbhpro/ivbhpro
      integer idp1(8,2:3,9,2,2),idp2(8,2:3,9,2,2)
      integer idp3(8,2:3,9,2,2),idp5(8,2:3,9,2,2)
      common/cidpart/idp1,idp2,idp3,idp5
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3,ip4,ip5,ip6,ip7
      common/ci1part/ip1,ip2,ip3,ip4,ip5,ip6,ip7
      integer ip1s,ip2s,ip3s,ip4s,ip5s,ip6s,ip7s
      common/ci1parts/ip1s,ip2s,ip3s,ip4s,ip5s,ip6s,ip7s
      integer ifk88seed
      common/cifk88seed/ifk88seed
      integer ichkpid
      common/cichkpid/ichkpid
      integer idec
      common/cidec/idec
      integer iallzero
      parameter (iallzero=1)
c
      i0=0
      jproc0=0
      itype0=0
      ich0=0
      itt0=0
      iret=0
      call checkst(xsum,xsumabs,iretvv)
      call checksts(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xost:',iretvv,iretvvs
        stop
      endif
      if(iretvv.eq.1)then
        iret=iretvv
        if(ifuntype.le.2)then
          iproclo=loproc
          iprocma=maproc
        elseif(ifuntype.le.4)then
          iproclo=loproco
          iprocma=maproco
        else
          write(*,*)'Fatal error in xost: ifuntype=',ifuntype
          stop
        endif
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do iproc=iproclo,iprocma
            do ich=ichmin,ichmax
              do i=1,idrmax(iproc,ich)
                do itype=1,itypemax(iproc)
                  do itt=ittmin,ittmax
                    wwx(i,iproc,itype,ich,itt)=
     #                vvs(i,iproc,itype,ich,itt)
                  enddo
                enddo
              enddo
            enddo
          enddo
        else
          do iproc=iproclo,iprocma
            do ich=ichmin,ichmax
              do i=1,idrmax(iproc,ich)
                do itype=1,itypemax(iproc)
                  do itt=ittmin,ittmax
                    wwx(i,iproc,itype,ich,itt)=
     #                vv(i,iproc,itype,ich,itt)
                  enddo
                enddo
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
          do ich=ichmin,ichmax
            do i=1,idrmax(iproc,ich)
              do itype=1,itypemax(iproc)
                do itt=ittmin,ittmax
                  if(iwh.eq.0)then
                    wh=wh+abs(wwx(i,iproc,itype,ich,itt))/xsumabs
                    if(wh.gt.xg)then
                      i0=i
                      jproc0=iproc
                      itype0=itype
                      ich0=ich
                      itt0=itt
                      iwh=1
                    endif
                  endif
                  if(wwx(i,iproc,itype,ich,itt).ne.0.d0)then
                    if(xstsign.ne.
     #                 sign(1.d0,wwx(i,iproc,itype,ich,itt)))then
                      if(iflag.eq.0)then
                        iwrong=iwrong+1
                        iflag=1
                      endif
                      rmax=max(rmax,abs(wwx(i,iproc,itype,ich,itt)))
                    endif
                  endif
                enddo
              enddo
            enddo
          enddo
        enddo
        if(iflag.eq.1)then
          if(xsum.ne.0.d0)rmax=rmax/xsum
          if(rmax.gt.0.05d0.or.xsum.eq.0.d0)iwrong1=iwrong1+1
        endif
        if(i0.eq.0.or.jproc0.eq.0.or.itype0.eq.0.or.
     #     ich0.eq.0.or.itt0.eq.0)then
          write(*,*)'Fatal error in xost',i0,jproc0,itype0,ich0,itt0
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0,ich0,itt0)
        i1=idp1(i0,jproc0,itype0,ich0,itt0)
        i2=idp2(i0,jproc0,itype0,ich0,itt0)
        i3=idp3(i0,jproc0,itype0,ich0,itt0)
        i5=idp5(i0,jproc0,itype0,ich0,itt0)
        if(abs(i3).gt.100.or.abs(i5).gt.100)call ckmunwgt(i3,i5)
        if(itt0.eq.1)then
          i4=6
        elseif(itt0.eq.2)then
          i4=-6
        else
          write(*,*)'Fatal error: wrong top identity',itt0
          stop
        endif
        if(ichkpid.eq.0)call parcheckfst(ihpro,i1,i2,i3,i5,iallzero,iz,
     #                                   i0,jproc0,itype0,ich0,itt0)
        call parcrossist(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
        ip4=i4
        ip5=i5
        if(ichkpid.eq.0)
     #    call parcheckfst(i1hpro,ip1,ip2,ip3,ip5,iallzero,io,
     #                     i0,jproc0,itype0,ich0,itt0)
c The top decays. In such case, ip3 and ip4 are final-state light partons
c identities, and ip5,ip6,ip7 are top decay products identities
        if(idec.eq.0)then
          ip4=ip5
          if(itt0.eq.1)then
            ip5=ip5s
            ip6=ip6s
            ip7=ip7s
          elseif(itt0.eq.2)then
            ip5=-ip5s
            ip6=-ip6s
            ip7=-ip7s
          endif
        endif
      endif
      return
      end


      subroutine parcrossist(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit none
      integer jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o,
     # ihprotq(402:406),ihprotqb(402:406)
      integer ifuntype
      common/cifuntype/ifuntype
      data ihprotq/401,0,409,403,409/
      data ihprotqb/408,0,403,408,401/
c
      if( (ifuntype.eq.1.or.ifuntype.eq.3) .or. 
     #    ((ifuntype.eq.2.or.ifuntype.eq.4).and.jproc0.eq.2) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif((ifuntype.eq.2.or.ifuntype.eq.4).and.jproc0.eq.3)then
        if(ihpro.eq.401.or.ihpro.eq.403.or.ihpro.eq.407.or.
     #     ihpro.eq.408.or.ihpro.eq.409)then
          write(*,*)'Error #1 in parcrossist:',ihpro,i1,i2,i3
          stop
        endif
        if(i3.gt.0.and.i3.le.5)then
          i1hproo=ihprotq(ihpro)
        elseif(i3.lt.0.and.i3.ge.-5)then
          i1hproo=ihprotqb(ihpro)
        else
          write(*,*)'Error #2 in parcrossist:',ihpro,i1,i2,i3
          stop
        endif
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
        write(*,*)'parcrossist: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      return
      end


      subroutine ckmunwgt(i3,i5)
c Determines parton identities of final state partons for the partonic
c processes whose weights involve sums over CKM matrix elements. The routines
c ckmcorr and ckmuncorr return parton identities following our conventions: 
c use imapp to convert them to PDG conventions
      implicit none
      integer i3,i5
      integer imapp(0:5)
      common/cimapp/imapp
c
      if(abs(i3).gt.10000.and.abs(i5).ne.999)then
        write(*,*)'Error #1 in ckmunwgt:',i3,i5
        stop
      endif
      if(abs(i5).eq.999)then
        call ckmcorr(i3,i5)
        i3=sign(1,i3)*imapp(abs(i3))
        i5=sign(1,i5)*imapp(abs(i5))
        return
      endif
      if(abs(i3).gt.100)then
        call ckmuncorr(i3)
        i3=sign(1,i3)*imapp(abs(i3))
      endif
      if(abs(i5).gt.100)then
        call ckmuncorr(i5)
        i5=sign(1,i5)*imapp(abs(i5))
      endif
      return
      end


      subroutine ckmcorr(i3,i5)
c Determines parton identities of two final state partons in the case in
c which they are connected to the same weak current
      implicit none
      integer i3,i5,i3a,iwh,iresu,iresd,ii,jj
      real * 8 fk88random,xden,xg,wh
      real * 8 ckm2(1:6,1:6)
      common/cckm2/ckm2
      real * 8 ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      common/cckmfct/ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      integer ifk88seed
      common/cifk88seed/ifk88seed
      integer imapd(3)
      data imapd/2,3,5/
      integer imapu(3)
      data imapu/1,4,6/
c
      i3a=abs(i3)
      if(i3a.eq.14235)then
        xden=ruckm+rcckm
      elseif(i3a.eq.23514)then
        xden=rducckm+rsucckm+rbucckm
      else
        write(*,*)'Error #1 in ckmcorr:',i3,i5
        stop
      endif
c
      xg=fk88random(ifk88seed)
      wh=0.d0
      iwh=0
      iresu=0
      iresd=0
      do ii=1,2
        do jj=1,3
          if(iwh.eq.0)then
            wh=wh+ckm2(imapu(ii),imapd(jj))/xden
            if(wh.gt.xg)then
              iresu=imapu(ii)
              iresd=imapd(jj)
              iwh=1
            endif
          endif
        enddo
      enddo
c
      if(iresu.eq.0.or.iresd.eq.0)then
        write(*,*)'Error #2 in ckmcorr:',i3,i5
        stop
      else
        if(i3a.eq.14235)then
          i3=sign(1,i3)*iresu
          i5=sign(1,i5)*iresd
        else
          i3=sign(1,i3)*iresd
          i5=sign(1,i5)*iresu
        endif
      endif
c
      return
      end


      subroutine ckmuncorr(ip)
c Determines parton identity on the basis of CKM weights
      implicit none
      integer ip,ipa,nel,iup,idown,iwh,ires,ii
      real * 8 fk88random,xden,xg,wh
      real * 8 ckm2(1:6,1:6)
      common/cckm2/ckm2
      real * 8 ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      common/cckmfct/ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      integer ifk88seed
      common/cifk88seed/ifk88seed
      integer imapd(3)
      data imapd/2,3,5/
      integer imapu(3)
      data imapu/1,4,6/
c
      ipa=abs(ip)
      if(ipa.gt.10000.or.ipa.eq.999)then
        write(*,*)'Error #1 in ckmuncorr:',ip
        stop
      endif
      if(ipa.eq.1235)then
        xden=ruckm
        nel=3
        iup=1
        idown=-1
      elseif(ipa.eq.4235)then
        xden=rcckm
        nel=3
        iup=4
        idown=-1
      elseif(ipa.eq.6235)then
        xden=rtckm
        nel=3
        iup=6
        idown=-1
      elseif(ipa.eq.142)then
        xden=rducckm
        nel=2
        iup=-1
        idown=2
      elseif(ipa.eq.143)then
        xden=rsucckm
        nel=2
        iup=-1
        idown=3
      elseif(ipa.eq.145)then
        xden=rbucckm
        nel=2
        iup=-1
        idown=5
      else
        write(*,*)'Error #2 in ckmuncorr:',ip
        stop
      endif
c
      xg=fk88random(ifk88seed)
      wh=0.d0
      iwh=0
      ires=0
      do ii=1,nel
        if(iwh.eq.0)then
          if(idown.eq.-1)then
            wh=wh+ckm2(iup,imapd(ii))/xden
            if(wh.gt.xg)then
              ires=imapd(ii)
              iwh=1
            endif
          elseif(iup.eq.-1)then
            wh=wh+ckm2(imapu(ii),idown)/xden
            if(wh.gt.xg)then
              ires=imapu(ii)
              iwh=1
            endif
          endif
        endif
      enddo
c
      if(ires.eq.0)then
        write(*,*)'Error #3 in ckmuncorr:',ip,iup,idown
        stop
      else
        ip=sign(1,ip)*ires
      endif
c
      return
      end


      subroutine checkst(xsum,xsumabs,iret)
c iret=0 -> all vv entries are equal to zero
c iret=1 -> there is at least one entry which is not zero
c xsum is the sum of all the entries of vv
c xsumabs is the sum of the absolute value of all the entries of vv
      implicit none
      integer jproc,iret,i,itype,ich,itt
      real * 8 vv(8,2:3,9,2,2)
      common/cvv/vv
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=2,3
        do ich=1,2
          do i=1,8
            do itype=1,9
              do itt=1,2
                if(vv(i,jproc,itype,ich,itt).ne.0.d0)iret=1
                xsum=xsum+vv(i,jproc,itype,ich,itt)
                xsumabs=xsumabs+abs(vv(i,jproc,itype,ich,itt))
              enddo
            enddo
          enddo
        enddo
      enddo
      return
      end


      subroutine checksts(xsum,xsumabs,iret)
c identical to checkst, except for the fact that works on vvs instead of vv,
c and jproc is not fixed
      implicit none
      integer jproc,iret,i,itype,ich,itt
      real * 8 vvs(8,2:3,9,2,2)
      common/cvvs/vvs
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=2,3
        do ich=1,2
          do i=1,8
            do itype=1,9
              do itt=1,2
                if(vvs(i,jproc,itype,ich,itt).ne.0.d0)iret=1
                xsum=xsum+vvs(i,jproc,itype,ich,itt)
                xsumabs=xsumabs+abs(vvs(i,jproc,itype,ich,itt))
              enddo
            enddo
          enddo
        enddo
      enddo
      return
      end


      subroutine getspincost()
c Determines the lepton momenta, by performing an unweighting using
c the exact real and Born lepton matrix elements
      implicit none
      real*8 pi,tolerance,two
      parameter (pi=3.14159265358979312D0)
      parameter (tolerance=1.d-2)
      parameter (two=2.d0)
      integer ione
      parameter (ione=1)
      character*2 str
      parameter (str='p1')
      include 'stpcblks.h'
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      real*8 ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      common/cpsave/ps,px,pyi,pcth1,pcth2,pphii,pyj,pphij
      real*8 xm012,ga1,bw1delf,bw1fmmn
      common/cbw1/xm012,ga1,bw1delf,bw1fmmn
      real*8 xm1low2,xm1upp2
      common/bounds/xm1low2,xm1upp2
      real*8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      real*8 xmt,twidth
      common/ctparam/xmt,twidth
      integer i0,jproc0,itype0,ich0,itt0
      common/cidproc/i0,jproc0,itype0,ich0,itt0
      integer idrlimcp(2,2:3,8),idrlimcm(2,2:3,8)
      common/cidrlims/idrlimcp,idrlimcm
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3,ip4,ip5,ip6,ip7
      common/ci1part/ip1,ip2,ip3,ip4,ip5,ip6,ip7
      integer ifuntype
      common/cifuntype/ifuntype
      integer iwidth
      common/ciwidth/iwidth
      integer ichkmom
      common/cichkmom/ichkmom
      integer neventsuw,nqeventsuw,ifailuw
      common/c1iunwgt/neventsuw,nqeventsuw,ifailuw
      integer ncntuws,nqcntuws,nmaxuw,nqmaxuw
      common/c2iunwgt/ncntuws,nqcntuws,nmaxuw,nqmaxuw
      integer idec
      common/cidec/idec
      integer ifk88seed
      common/cifk88seed/ifk88seed
      character*2 xproc(3)
      common/cxproc/xproc
      real*8 xtmp,prob,spcdamp_in,rrnd,fk88random,spcdamp_out,e1,f1,
     # g1,h1,phitq1,cthtq1,phitq2,cthtq2,o,q12,xbwmass3,rat1,qphsp,
     # q1,tk,uk,q1q,q2q,xij,xdec,xmadevst,unxdec,xsnglt,dmfactb1,
     # dmfact1,phspfact1,xboundb,rat,zyi,zxij,zyj,yitmp,xinv(5),
     # xtq(4),xbq(4),xel(4),xnu(4)
      integer iborn,iproj,icross,jjprc,jist,jidr,icntuw,iqcntuw
c
      if(ichkmom.eq.0)call spcchest(1)
      if(ifuntype.eq.2.or.ifuntype.eq.4)then
        if(px.ne.1.d0.or.xmom_cm(3,4).ne.0.d0)then
          write(*,*)'Error #1 in getspincost'
          stop
        else
          iborn=0
          iproj=0
          if(jproc0.eq.2)then
            icross=0
          else
            if(ifuntype.eq.4)then
              write(*,*)'Error #2 in getspincost'
              stop
            endif
            icross=1
          endif
          xtmp=px
        endif
      endif
      if(ifuntype.eq.1)then
        prob=spcdamp_in(px,pyi)
        rrnd=fk88random(ifk88seed)
        if(rrnd.ge.prob)then
c Close to the soft/collinear limits: use Born kinematics for the unweighting
          iborn=0
          iproj=1
          icross=1
          xtmp=1.d0
        else
c Away from the soft/collinear limit: use real kinematics for the unweighting
          iborn=1
          iproj=0
          icross=0
          xtmp=px
        endif
      endif
      if(ifuntype.eq.3)then
        if(jproc0.ne.2)then
          write(*,*)'Error #3 in getspincost'
          stop
        endif
        prob=spcdamp_out(px,pyj)
        rrnd=fk88random(ifk88seed)
        icross=0
        if(rrnd.ge.prob)then
c Close to the soft/collinear limits: use Born kinematics for the unweighting
          iborn=0
          iproj=1
          xtmp=1.d0
        else
c Away from the soft/collinear limit: use real kinematics for the unweighting
          iborn=1
          iproj=0
          xtmp=px
        endif
      endif
      yitmp=pyi
c When iproj=0, the Born and real kinematics are used to perform unweighting
c for S and H events respectively. When iproj=1, the real kinematics is close 
c to the soft/collinear limits, and the Born is used to unweight. In the case 
c of the qg process and initial-state collinear limit, the Born is chosen
c according to whether the real configuration is closer to the y->1 
c (pyi>0 ==> idrlimcp is used) or to the y->-1 (pyi<0 ==> idrlimcm is used) 
c limit. This strategy, which serves to set here the local value of idr
c (jidr), must also be adopted in the case of S[in] events due to the qg 
c contribution, and is associated with icross=1; icross=0 implies on the 
c other hand that the local and global values of idr are the same. Any 
c manipulations on parton identities must also be carried out here. It
c may happen that when icross=1 the corresponding collinear limit is zero;
c in such a case, use the other limit (for S[in] events), or freeze x or yi
c to a given value (for H[in] events); in such a case, we also set iproj=2
      if(icross.eq.0)then
        jjprc=jproc0
        jist=ich0
        jidr=i0
      elseif(icross.eq.1)then
        jjprc=2
        jist=ich0
        if(pyi.ge.0.d0)then
          jidr=idrlimcp(ich0,jproc0,i0)
        else
          jidr=idrlimcm(ich0,jproc0,i0)
        endif
        if(jidr.eq.0.and.ifuntype.eq.1)then
          xtmp=px
          yitmp=pyi
          iborn=1
          jjprc=jproc0
          jidr=i0
          if(px.gt.0.995d0)then
            xtmp=0.995d0
            iproj=2
          endif
          if(abs(pyi).gt.0.995d0)then
            yitmp=0.995d0*sign(1.d0,pyi)
            iproj=2
          endif
        elseif(jidr.eq.0.and.ifuntype.eq.2)then
          if(pyi.ge.0.d0)then
            jidr=idrlimcm(ich0,jproc0,i0)
          else
            jidr=idrlimcp(ich0,jproc0,i0)
          endif
        endif
      else
        write(*,*)'Error #4 in getspincost'
        stop
      endif
c
      neventsuw=neventsuw+1
      icntuw=0
 100  icntuw=icntuw+1
      e1=fk88random(ifk88seed)
      f1=fk88random(ifk88seed)
      g1=fk88random(ifk88seed)
      h1=fk88random(ifk88seed)
      phitq1=2*pi*e1
      cthtq1=-1.d0+2*f1
      phitq2=2*pi*g1
      cthtq2=-1.d0+2*h1
 300  continue
      if(iwidth.eq.1)then
        iqcntuw=0
 200    iqcntuw=iqcntuw+1
        o=fk88random(ifk88seed)
c First distribute q according to the matrix element upper bound,
c which can be done exactly the upper bound being a Breit Wigner
        q12=xbwmass3(o,xm012,ga1,bw1delf,bw1fmmn)
c Then reject some of the values generated according to the phase-space
c q-dependent factor. A 1->1+(1->2) phase-space decomposition has been used.
c Much better here than after computing matrix elements. The following
c form works since qphsp is a function decreasing with q2
        rat1=qphsp(q12,xm12)/qphsp(xm1low2,xm12)
        rrnd=fk88random(ifk88seed)
        if(rat1.lt.rrnd)goto 200
        nqcntuws=nqcntuws+iqcntuw
        if(iqcntuw.gt.nqmaxuw)nqmaxuw=iqcntuw
        nqeventsuw=nqeventsuw+1
        q1=sqrt(q12)
      else
        q12=xm012
        q1=sqrt(q12)
      endif
c No complications here due to off-shell top; can use the same kinematics 
c for decayed and undecayed matrix elements
      if(ifuntype.le.2)then
        call invar_in(xm12,xm22,ps,xtmp,yitmp,pcth1,pcth2,str,
     #                tk,uk,q1q,q2q,xinv)
        zyi=yitmp
        zxij=two
        zyj=two
      elseif(ifuntype.le.4)then
        call invar_out(xm12,ps,xtmp,yitmp,pphii,pyj,pphij,
     #                 tk,uk,q1q,q2q,xij,xinv)
        zyi=two
        zxij=xij
        zyj=pyj
      else
        write(*,*)'Error #5 in getspincost'
        stop
      endif
      call gentopdmst(xmt,q1,cthtq1,phitq1,cthtq2,phitq2,
     #                xtq,xbq,xel,xnu)
      if(ichkmom.eq.0)call checkstec1(xmt,xtq,xbq,xel,xnu,1)
      if(ichkmom.eq.0)call checkmst(xmom_cm,ps,0.d0,10,1)
      xdec=xmadevst(iborn,jist,jjprc,jidr,ps,xtmp,zyi,zyj,xmom_cm)
      unxdec=xsnglt(iborn,jist,jjprc,jidr,xm12,ps,xtmp,zyi,zxij,zyj,
     #              tk,uk,q1q,q2q,xinv)
      dmfactb1=256*xm12**2/16.d0
      dmfact1=1/(64.d0*sthw2**2)*
     #        1.d0/((q12-xm012)**2+xm012*ga1**2)
c e^4 -> gw^4; single-top cross sections in mcatnlo_stxsec factorize gw^4
      dmfact1=dmfact1*sthw2**2
      phspfact1=1.d0/(xm12*twidth**2)
      xboundb=dmfactb1*dmfact1*phspfact1
      rat=xdec/((1+tolerance)*unxdec*xboundb)
      if(rat.gt.1.d0)then
        ifailuw=ifailuw+1
        goto 300
      endif
      rrnd=fk88random(ifk88seed)
      if(rat.lt.rrnd)goto 100
      ncntuws=ncntuws+icntuw
      if(icntuw.gt.nmaxuw)nmaxuw=icntuw
c The event is accepted; regenerate kinematics if Born was used for 
c unweighting H events (to get xmom_cm corresponding to a real emission
c configuration), or if x or y values had been freezed
      if(iproj.eq.0)then
        if(px.ne.xtmp)then
          write(*,*)'Error #6 in getspincost',px,xtmp
          stop
        endif
      elseif(iproj.eq.1.or.iproj.eq.2)then
        if(ifuntype.le.2)then
          call invar_in(xm12,xm22,ps,px,pyi,pcth1,pcth2,str,
     #                  tk,uk,q1q,q2q,xinv)
        elseif(ifuntype.le.4)then
          call invar_out(xm12,ps,px,pyi,pphii,pyj,pphij,
     #                   tk,uk,q1q,q2q,xij,xinv)
        endif
        call gentopdmst(xmt,q1,cthtq1,phitq1,cthtq2,phitq2,
     #                  xtq,xbq,xel,xnu)
        if(ichkmom.eq.0)call checkmst(xmom_cm,ps,0.d0,20,1)
      else
        write(*,*)'Error #7 in getspincost'
        stop
      endif
      if(ichkmom.eq.0)call spcchest(2)
      return
      end


      function spcdamp_in(x,yi)
c This function is defined in such a way that
c    spcdamp_in=0  if  tt=0
c  0<spcdamp_in<1  if  0<tt<1
c    spcdamp_in=1  if  tt>1
c and tt is a measure in the (x,yi) plane, such that tt=0 in the soft
c and collinear limits (x=1, or yi=1, or yi=-1), growing monotonically
c away from these limits. In terms of invariants, tt=4*tk*uk/((1-xlim)*s)**2,
c which can easily be generalized for any kind of emissions. 
c Since when spcdamp_in=1 the real matrix elements are used in the 
c unweighting, xlim has been defined in such a way that, if be_ispcfun=1,
c spcdamp_in is equal to 1 in a region similar to the dead zone for E0=s. 
c This is by no means necessary, and the dependence upon xlim in tt can be
c eliminated altogether. Call this function with al_ispcfun>=1, 0<be_ispcfun<=1
      implicit none
      real * 8 spcdamp_in,x,yi,xmin,tt,xlim
      parameter (xmin=0.69519410160110384d0)
      real * 8 al_ispcfun,be_ispcfun
      common/cspcipar/al_ispcfun,be_ispcfun
c
      xlim=1.d0-be_ispcfun+xmin*be_ispcfun
      tt=(1-x)**2*(1-yi**2)/(1-xlim)**2
      if(tt.lt.0.d0)then
        write(*,*)'Error in spcdamp_in',tt
        stop
      endif
      if(tt.gt.1.d0)tt=1.d0
      spcdamp_in=tt**(2*al_ispcfun)/
     #          (tt**(2*al_ispcfun)+(1-tt)**(2*al_ispcfun))
      return
      end


      function spcdamp_out(x,yj)
c This function is defined in such a way that
c    spcdamp_out=0  if  tt=0
c  0<spcdamp_out<1  if  0<tt<1
c    spcdamp_out=1  if  tt>1
c and tt is a measure in the (x,yj) plane, such that tt=0 in the soft
c and collinear limits (x=1, or yj=1), growing monotonically away from 
c these limits. 
c Since when spcdamp_out=1 the real matrix elements are used in the 
c unweighting, xlim has been defined in such a way that, if be_ospcfun=1,
c spcdamp_in is equal to 1 in a region similar to the dead zone for E0=s. 
c This is by no means necessary, and the dependence upon xlim in tt can be
c eliminated altogether. Call this function with al_ospcfun>=1, 0<be_ospcfun<=1
      implicit none
      real * 8 spcdamp_out,x,yj,xmin,tt,xlim
      parameter (xmin=0.69519410160110384d0)
      real * 8 al_ospcfun,be_ospcfun
      common/cspcipar/al_ospcfun,be_ospcfun
c
      xlim=1.d0-be_ospcfun+xmin*be_ospcfun
      tt=(1-x)**2*(1-yj)/(1-xlim)**2
      if(tt.lt.0.d0)then
        write(*,*)'Error in spcdamp_out',tt
        stop
      endif
      if(tt.gt.1.d0)tt=1.d0
      spcdamp_out=tt**(2*al_ospcfun)/
     #           (tt**(2*al_ospcfun)+(1-tt)**(2*al_ospcfun))
      return
      end


      subroutine spcchest(iflag)
c Stores hard-process four-momenta (in xmom_save) at the beginning of 
c getspincost(), and checks as the last step of getspincost() that
c the manipulations carried out there did not change them, by comparing
c xmom_save and xmom_cm
      implicit none
      real*8 tiny,xmom_save(5,4)
      parameter (tiny=1.d-4)
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer iflag,i,j,itmp
      save xmom_save
c
      if(iflag.eq.1)then
        itmp=0
        do i=1,5
          do j=1,4
            xmom_save(i,j)=xmom_cm(i,j)
          enddo
        enddo
      elseif(iflag.eq.2)then
        itmp=0
        do i=1,5
          do j=1,4
            if(abs(xmom_save(i,j)-xmom_cm(i,j)).gt.tiny)itmp=1
          enddo
        enddo
      else
        write(*,*)'Wrong call to spcchest'
        stop
      endif
      if(itmp.eq.1)then
        write(*,*)'The check in spcchest failed'
        write(*,*)'Original momenta:'
        do i=1,5
          write(*,900)(xmom_save(i,j),j=1,4)
        enddo
        write(*,*)'  '
        write(*,*)'New momenta:'
        do i=1,5
          write(*,900)(xmom_cm(i,j),j=1,4)
        enddo
        stop
      endif
 900  format(4(1x,d14.8))
      return
      end


      function xsnglt(iborn,ist,jproc,idr,xmt2,s,x,yi,xij,yj,
     #                tk,uk,q1q,q2q,xinv)
c Wrapper for the undecayed matrix elements of the original code.
c For Born matrix elements, q1q is t (consistently with the
c routines invar_in and invar_out)
      implicit none
      real*8 xsnglt,xmt2,s,x,yi,xij,yj,tk,uk,q1q,q2q,xinv(5)
      integer iborn,ist,jproc,idr
      real*8 xmatin(8,2)
c
      if(iborn.eq.0)then
        call fstrn(s,q1q,jproc,xmatin)
      else
        call frest(s,x,yi,xij,yj,tk,uk,q1q,q2q,xinv,jproc,xmatin)
      endif
      xsnglt=xmatin(idr,ist)
      return
      end


      subroutine gentopdmst(xmt,xmw,cth1,phi1,cth2,phi2,
     #                      xtq,xbq,xel,xnu)
c Generates the four-momenta of the decay products of the top. These
c four-momenta are returned in the top rest frame (xbq, xel, xnu; the
c trivial top momentum is returned as well, xtq). The four-momenta
c are also boosted to the frame in which the top has momentum xmom_cm(4,*),
c and the common block cxmomcm is filled according to the identifications
c   l+ --> xmom_cm(6,*), nu --> xmom_cm(7,*), b --> xmom_cm(8,*), 
c consistently with the labelling conventions used in MC@NLO:
c   x(1)y(2) -> z(3)t(4)[->l+(6)nu(7)b(8)]w(5)
c The inputs of the routine are cth1,phi1,cth2,phi2, which are cosines of
c polar angles and azimuthal angles, with
c   (cth1,phi1) --> direction of W in the top rest frame
c   (cth2,phi2) --> direction of l+ in the W rest frame
      implicit none
      real*8 xmt,xmw,cth1,phi1,cth2,phi2,xtq(4),xbq(4),xel(4),xnu(4)
      real*8 xmt2,xmw2,sth1,sth2,ew,eb,pwx,pwy,pwz,pbx,pby,pbz,eel,
     # enu,pex,pey,pez,pnx,pny,pnz,tmp(5),tmp1(4),tmp2(4)
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer itop,iel,inu,ib
      parameter (itop=4)
      parameter (iel=6)
      parameter (inu=7)
      parameter (ib=8)
c
      xmt2=xmt**2
      xmw2=xmw**2
      sth1=sqrt(1-cth1**2)
      sth2=sqrt(1-cth2**2)
c
      xtq(1)=0.d0
      xtq(2)=0.d0
      xtq(3)=0.d0
      xtq(4)=xmt
c W and b momenta, top rest frame
      ew=(xmt2+xmw2)/(2*xmt)
      eb=(xmt2-xmw2)/(2*xmt)
      pwx=eb*sth1*cos(phi1)
      pwy=eb*sth1*sin(phi1)
      pwz=eb*cth1
      pbx=-pwx
      pby=-pwy
      pbz=-pwz
      xbq(1)=pbx
      xbq(2)=pby
      xbq(3)=pbz
      xbq(4)=eb
c l+ and nu momenta, W rest frame
      eel=xmw/2.d0
      enu=eel
      pex=eel*sth2*cos(phi2)
      pey=eel*sth2*sin(phi2)
      pez=eel*cth2
      pnx=-pex
      pny=-pey
      pnz=-pez
c Boost lepton momenta to top rest frame
      tmp(1)=pwx
      tmp(2)=pwy
      tmp(3)=pwz
      tmp(4)=ew
      tmp(5)=xmw
c Boost l+
      tmp1(1)=pex
      tmp1(2)=pey
      tmp1(3)=pez
      tmp1(4)=eel
      call hwstb4(tmp,tmp1,tmp2)
      xel(1)=tmp2(1)
      xel(2)=tmp2(2)
      xel(3)=tmp2(3)
      xel(4)=tmp2(4)
c Boost nu
      tmp1(1)=pnx
      tmp1(2)=pny
      tmp1(3)=pnz
      tmp1(4)=enu
      call hwstb4(tmp,tmp1,tmp2)
      xnu(1)=tmp2(1)
      xnu(2)=tmp2(2)
      xnu(3)=tmp2(3)
      xnu(4)=tmp2(4)
c Boost all momenta to cm frame
      tmp(1)=xmom_cm(itop,1)
      tmp(2)=xmom_cm(itop,2)
      tmp(3)=xmom_cm(itop,3)
      tmp(4)=xmom_cm(itop,4)
      tmp(5)=xmt
c
      call filltopdst(tmp,xel,iel)
      call filltopdst(tmp,xnu,inu)
      call filltopdst(tmp,xbq,ib)
c
      return
      end


      subroutine filltopdst(tmp,tmp1,ipart)
c Utility routine for gentopdmst; performs the boost and fills xmom_cm 
c for top decay products
      implicit none
      real*8 tmp(5),tmp1(4),tmp2(4)
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer ipart

      call hwstb4(tmp,tmp1,tmp2)
      xmom_cm(ipart,1)=tmp2(1)
      xmom_cm(ipart,2)=tmp2(2)
      xmom_cm(ipart,3)=tmp2(3)
      xmom_cm(ipart,4)=tmp2(4)
      return
      end

C      function qphsp(q12,xmt2)
Cc Non-trivial factor of the t->bW phase space, in the t rest frame; q12 is
Cc the W mass squared
C      implicit none
C      real*8 qphsp,q12,xmt2,tmp
Cc
C      tmp=0.d0
C      if(q12.gt.0.d0.and.q12.lt.xmt2)tmp=(xmt2-q12)/(2.d0*xmt2)
C      qphsp=tmp
C      return
C      end


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
      subroutine invar_in(xm12,xm22,s,x,y,cth1,cth2,str,
     #                    tk,uk,q1q,q2q,xinv)
c This routine has been obtained by modifying the analogous routine
c in the VH code. 
c The names of the invariants are taken from Nucl.Phys.B383:3-44,1992 [FNR] 
c (q1q is q_1 of the paper, q2q is q_2, q1c is \hat{q}_1, q2c is \hat{q}_2).
c
c The hard process is
c   a(p1)+b(p2) --> t(k1)+c(k2)+d(k)
c where a, b, c and d are light partons, t is top quark with k1^2=xm12.
c The quarks t and c are attached to the W-vertex, k2^2=xm22=0. The process 
c can be described by the same invariants as in FNR [eqs.(2.6) and (2.7)].
c
c In terms of the
c invariants, the dot products are 
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
c
c The four momenta are given in the t-c rest frame as follows
c     p1 = p10*(1,0,spsi2,cpsi2)
c     p2 = p20*(1,0,spsi ,cpsi )
c     k  = k0*(1,0,spsi1,cpsi1).
c     k1 = (k10, bx*sth2*sth1, bx*cth2*sth1, bx*cth1)
c     k2 = (k20,-bx*sth2*sth1,-bx*cth2*sth1,-bx*cth1).
c The argument str should be set to 'p1': then p1 = p10 (1,0,0,1) (psi2 =0), 
c with psi and psi1 determined using momentum conservation; according to the 
c work done for Drell Yan, the other options for str have been disabled.
c
c The four momenta of the partons in the c.m. frame of the incoming
c partons are stored in xmom_cm(ipart,icomp), with the conventions:
c   icomp=1 -> px, icomp=2 -> py, icomp=3 -> pz, icomp=4 -> E;
c   ipart=1 -> p1, ipart=2 -> p2, ipart=3 -> k, ipart=4 -> k1, ipart=5 -> k2.
c
c Notice that  bx = sqrt(s2)/2 * beta_x[FNR paper]
c
c
      implicit none
      real * 8 xm12,xm22,s,x,y,cth1,cth2,tk,uk,q1q,q2q,xinv(5)
      character * 2 str
c
      real * 8 ptv1,ptv2,ptvg,y1,y2,yg
      common/perpen/ptv1(2),ptv2(2),ptvg(2)
      common/ycmvar/y1,y2,yg
c
      real * 8 s2,drs2,p10,p20,k0,k10,k20,bx,sth1,cpsi,
     # spsi,cpsi2,spsi2,cpsi1,spsi1,xktsq,xkt1sq,xkt2sq,
     # xkt,xkt1,xkt2,tmp,sqs,tiny,zero,sth2,q1c,q2c,w1,w2,
     # e1lab,pl1lab,e2lab,pl2lab,beta,xcpsi1,xspsi1
      parameter (tiny=1.d-14)
      parameter (zero=0.d0)
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c     
      sqs=sqrt(s)
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
         write(6,*) 'Error in invar: str=',str
         stop
      endif
      q1q = xm12 - 2*p10*(k10-bx*(cth2*sth1*spsi2+cth1*cpsi2))
      q2q = xm22 - 2*p20*(k20+bx*(cth2*sth1*spsi +cth1*cpsi ))
      q1c = xm12 + xm22 - s - tk - q1q
      q2c = xm12 + xm22 - s - uk - q2q
      w1  = xm12 - q1q + q2q - tk
      w2  = xm22 - q2q + q1q - uk
c Here define xinv, according to
c   p_i.k = sqrt{s}*(1-x)*xinv(i)    i=1,2
c   k_i.k = sqrt{s}*(1-x)*xinv(i+3)  i=1,2
c and for consistency xinv(3)=0; xinv thus factor out analytically the 
c xi-dependence. These quantities are used to compute the soft limits
c of matrix elements and S_in 
      if(x.ne.1.d0)then
        beta=4*bx/drs2
        xcpsi1 = cpsi1
        xspsi1 = spsi1
      else
        beta=sqrt(1-2*(xm12+xm22)/s+(xm12-xm22)**2/s**2)
        xcpsi1 = y
        xspsi1 = sqrt(1-y**2)
      endif
      xinv(1) = sqs*(1-y)/4.d0
      xinv(2) = sqs*(1+y)/4.d0
      xinv(3) = 0d0
      xinv(4) = sqs/4.d0*(1-(xm22-xm12)/(x*s)-
     #    beta*(cth2*sth1*xspsi1+cth1*xcpsi1))
      xinv(5) = sqs/4.d0*(1+(xm22-xm12)/(x*s)+
     #    beta*(cth2*sth1*xspsi1+cth1*xcpsi1))
c
c Recall: y1,y2, yg are rapidities in the partonic cm frame
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
c xktsq, xkt1sq e xkt2sq are the square of transverse momenta of d, t,
c and c respectively. The axis orientation is such that t is always
c along the x direction. The component of p_T(t) along the y direction
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
c xmom_cm(1,mu) = p1(mu)
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqs/2.d0
      xmom_cm(1,4)=sqs/2.d0
c xmom_cm(2,mu) = p2(mu)
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqs/2.d0
      xmom_cm(2,4)=sqs/2.d0
c xmom_cm(3,mu) = k(mu)
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
c xmom_cm(4,mu) = k1(mu)
      e1lab=(2*xm12-q1q-q2c)/(2*sqs)
      pl1lab=(q1q-q2c)/(2*sqs)
      xmom_cm(4,1)=ptv1(1)
      xmom_cm(4,2)=ptv1(2)
      xmom_cm(4,3)=pl1lab
      xmom_cm(4,4)=e1lab
c xmom_cm(5,mu) = k2(mu)
      e2lab=(2*xm22-q1c-q2q)/(2*sqs)
      pl2lab=(q1c-q2q)/(2*sqs)
      xmom_cm(5,1)=ptv2(1)
      xmom_cm(5,2)=ptv2(2)
      xmom_cm(5,3)=pl2lab
      xmom_cm(5,4)=e2lab
c
      if(ichkmom.eq.0) call checkmst(xmom_cm,s,0.d0,1,2)
      return
      end


      subroutine invar_out(xm12,s,x,yi,phii,yj,phij,
     #                     tk,uk,q1q,q2q,xij,xinv)
c This routine has been obtained by modifying the analogous routine
c of the jet codes (hdyjetdiff.for and phyjetdiff.for). The names of the
c invariants are the same as in invar_in, and have been taken from 
c Nucl.Phys.B383:3-44,1992 [FNR] (q1q is q_1 of the paper, q2q is q_2, 
c q1c is \hat{q}_1, q2c is \hat{q}_2).
c
c As in invar_in, the hard process is 
c   a(p1)+b(p2) --> t(k1)+c(k2)+d(k)
c where a, b, c and d are light partons, t is top quark with k1^2=xm12.
c The quarks t and c are attached to the W-vertex, and k2^2=xm22=0. 
c Owing to the fact that xm22=0, and at variance with invar_in,
c we don't insert here xm22 in the entries of the routine
c
c The four momenta are given in the c.m. frame of the incoming partons
c according to the following parametrization
c    p1 = sqrt(s)/2*(1,0,0,1)
c    p2 = sqrt(s)/2*(1,0,0,-1)
c    k  = sqrt(s)/2*xii*(1,vk.R)
c    k2 = sqrt(s)/2*xij*(1,vk2.R)
c    k1 = p1+p2-k-k2
c where the three-vectors vk and vk2 are
c    vk  = (cos(phij)*sqrt(1-yj**2),sin(phij)*sqrt(1-yj**2),yj)
c    vk2 = (0,0,1)
c and R is the rotation matrix obtained by transposing the following matrix
c
c  (    cos(phii)*yi,     -sin(phii),     cos(phii)*sqrt(1-yi**2)  )
c  (    sin(phii)*yi,      cos(phii),     sin(phii)*sqrt(1-yi**2)  )
c  (  -sqrt(1-yi**2),              0,                          yi  )
c
c Note that vk2 corresponds to \hat{p}_j of eq.(4.57) of FKS, and
c vk corresponds to \hat{p} of eq.(4.56). Their parametrizations are
c therefore interchanged wrt those of the paper. The formalism is
c however unchanged, since in the treatment of the final-state singularities
c yi and phii don't play any role, and the only thing that matters is
c eq.(4.58), which still holds.
c
c Upon choosing the parametrization above, the phase space of 
c eq.(4.61) can be computed explicitly, and the delta function over
c energy is equivalent to a delta over xij, which therefore can
c be expressed as a function of xii and yj as done below.
c
c The four momenta of the partons in the c.m. frame of the incoming
c partons are stored in xmom_cm(ipart,icomp), with the conventions:
c   icomp=1 -> px, icomp=2 -> py, icomp=3 -> pz, icomp=4 -> E;
c   ipart=1 -> p1, ipart=2 -> p2, ipart=3 -> k, ipart=4 -> k1, ipart=5 -> k2.
c
      implicit none
      real * 8 xm12,s,x,yi,phii,yj,phij,tk,uk,q1q,q2q,xij,xinv(5)
      real * 8 xii,sqs,si,cphi,sphi,e2lab,xkt2,xx,yy,zz,aa1,bb1,cc1,
     #  eglab,dd2,aa2,bb2,cc2,q1c,w2,q2c,w1,s2
      real * 8 ptv1,ptv2,ptvg,y1,y2,yg
      common/perpen/ptv1(2),ptv2(2),ptvg(2)
      common/ycmvar/y1,y2,yg
      real*8 xmom_cm(8,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c
      sqs=sqrt(s)
      xii=1-x
      xij=2*(1-xm12/s-xii)/(2-xii*(1-yj))
      si=sqrt(1-yi**2)
      cphi=cos(phii)
      sphi=sin(phii)
c xmom_cm(1,mu) = p1(mu)
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqs/2.d0
      xmom_cm(1,4)=sqs/2.d0
c xmom_cm(2,mu) = p2(mu)
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqs/2.d0
      xmom_cm(2,4)=sqs/2.d0
c Parton c -- was nj in the jet code
      e2lab=sqs/2.d0*xij
      xkt2=e2lab*si
      ptv2(1)=xkt2*cphi
      ptv2(2)=xkt2*sphi
      xmom_cm(5,1)=ptv2(1)
      xmom_cm(5,2)=ptv2(2)
      xmom_cm(5,3)=e2lab*yi
      xmom_cm(5,4)=e2lab
      if(yi.eq.1.d0)then
        y2=1.d8
      elseif(yi.eq.-1.d0)then
        y2=-1.d8
      else
        y2=0.5d0*log((1.d0+yi)/(1.d0-yi))
      endif
c Parton d -- was ni in the jet code; in the frame where \vec{k2}=(0,0,1),
c \vec{k}=(xx,yy,zz); a rotation follows to the partonic c.m. frame
      xx=cos(phij)*sqrt(1-yj**2)
      yy=sin(phij)*sqrt(1-yj**2)
      zz=yj
      aa1=xx*yi*cphi-yy*sphi+zz*si*cphi
      bb1=xx*yi*sphi+yy*cphi+zz*si*sphi
      cc1=-xx*si+zz*yi
      eglab=sqs/2.d0*xii
      ptvg(1)=eglab*aa1
      ptvg(2)=eglab*bb1
      xmom_cm(3,1)=ptvg(1)
      xmom_cm(3,2)=ptvg(2)
      xmom_cm(3,3)=eglab*cc1
      xmom_cm(3,4)=eglab
      if(cc1.eq.1.d0)then
        yg=1.d8
      elseif(cc1.eq.-1.d0)then
        yg=-1.d8
      else
        yg=0.5d0*log((1.d0+cc1)/(1.d0-cc1))
      endif
c Top quark -- was labelled n3 in the jet code
      dd2=2.d0-xii-xij
      aa2=-(xij*cphi*si+xii*aa1)
      bb2=-(xij*sphi*si+xii*bb1)
      cc2=-(xij*yi+xii*cc1)
      ptv1(1)=sqs/2.d0*aa2
      ptv1(2)=sqs/2.d0*bb2
      xmom_cm(4,1)=ptv1(1)
      xmom_cm(4,2)=ptv1(2)
      xmom_cm(4,3)=sqs/2.d0*cc2
      xmom_cm(4,4)=sqs/2.d0*dd2
      if(dd2.eq.cc2)then
        y1=1.d8
      elseif(dd2.eq.-cc2)then
        y1=-1.d8
      else
        y1=0.5d0*log((dd2+cc2)/(dd2-cc2))
      endif
c Compute the invariants using the parametrization of the momenta given
c above, and the relations among them as given in FNR eq.(2.7)
      q1c=-s*xij*(1-yi)/2.d0
      tk=-s*xii*(1-cc1)/2.d0
      q2q=-s*xij*(1+yi)/2.d0
      uk=-s*xii*(1+cc1)/2.d0
      w2=s*xii*xij*(1-yj)/2.d0
      q1q=uk+q2q+w2
      q2c=tk+q1c+w2
      w1=s+q1c+q2q
      s2=s*(1-xii)
c Here define xinv, according to
c   p_i.k = sqrt{s}*(1-x)*xinv(i)    i=1,2
c   k_i.k = sqrt{s}*(1-x)*xinv(i+3)  i=1,2
c and for consistency xinv(3)=0; xinv thus factor out analytically the 
c xi-dependence. These quantities are used to compute the soft limits
c of matrix elements and S_out
      xinv(1)=sqs*(1.d0-cc1)/4.d0
      xinv(2)=sqs*(1.d0+cc1)/4.d0
      xinv(3)=0.d0
      xinv(4)=sqs*( dd2-aa1*aa2-bb1*bb2-cc1*cc2 )/4.d0
      xinv(5)=sqs*xij*(1-yj)/4.d0
c
      if(ichkmom.eq.0) call checkmst(xmom_cm,s,0.d0,1,2)
      return
      end


c Returns S_in=(k1.k)^a(k2.k)^a/[ (k1.k)^a(k2.k)^a+(p1.k)^a(p2.k)^a ]
c where a==powmeas should be chosen equal or larger than one. For qg
c contributions, returns one (since only initial-state singularities
c are present in such a case)
      function s_in(xinv,jproc)
      implicit none
      integer jproc,i
      real*8 s_in,xinv(5),powmeas,yinv(5),tmp
      common/cpowm/powmeas
c
      if(jproc.eq.2)then
        do i=1,5
          if(i.ne.3)yinv(i)=xinv(i)**powmeas
        enddo
        tmp=yinv(4)*yinv(5)/( yinv(4)*yinv(5)+yinv(1)*yinv(2) )
      elseif(jproc.eq.3)then
        tmp=1.d0
      else
        write(*,*)'Error in S_in: unknown jproc',jproc
        stop
      endif
      s_in=tmp
      return
      end


c Returns S_out=(p1.k)^a(p2.k)^a/[ (k1.k)^a(k2.k)^a+(p1.k)^a(p2.k)^a ]
c where a==powmeas should be chosen equal or larger than one. For qg
c contributions, returns zero (since only initial-state singularities
c are present in such a case)
      function s_out(xinv,jproc)
      implicit none
      integer jproc,i
      real*8 s_out,xinv(5),powmeas,yinv(5),tmp
      common/cpowm/powmeas
c
      if(jproc.eq.2)then
        do i=1,5
          if(i.ne.3)yinv(i)=xinv(i)**powmeas
        enddo
        tmp=yinv(1)*yinv(2)/( yinv(4)*yinv(5)+yinv(1)*yinv(2) )
      elseif(jproc.eq.3)then
        tmp=0.d0
      else
        write(*,*)'Error in S_out: unknown jproc',jproc
        stop
      endif
      s_out=tmp
      return
      end


      subroutine checkmst(xmom,smax,ybst,iflag,itype)
      implicit none
      real * 8 xmom(8,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-5)
      integer iflag,itype,i,j,jj,jflag,jeflag,jmax
c
      if(itype.eq.1)then
        jmax=8
      elseif(itype.eq.2)then
        jmax=5
      else
        write(6,*)'Wrong option in checkmst'
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
          write(*,*)'iflag,i=',iflag,i
          write(*,*)'smax,y=',smax,ybst
          write(*,*)'x1,x2=',x1,x2
          do j=1,8
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
        do j=1,8
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end


      subroutine checkstec1(xmt,xtq,xbq,xel,xnu,itop)
c Checks momentum conservation in top decay
      implicit none
      real*8 xmt,tiny,diff,xtq(4),xbq(4),xel(4),xnu(4)
      parameter (tiny=1.d-8)
      integer itop,i
c
      do i=1,4
        diff=xtq(i)-xbq(i)-xel(i)-xnu(i)
        if(abs(diff).gt.tiny*xmt)then
          write(6,*)'Subroutine checkstec1'
          write(6,*)'Momentum is not conserved in decay',i,itop
          stop
        endif
      enddo
      return
      end


      subroutine checkstec2(xmom,idec,iprod1,iprod2,iprod3)
c Checks momentum conservation in top decay, after manipulations 
c in put_on_shest()
      implicit none
      real * 8 xmom(8,4)
      real * 8 tiny,xsum(4),xsuma(4),xrat(4)
      parameter (tiny=5.d-3)
      integer idec,iprod1,iprod2,iprod3,jflag,i,jj
c
      jflag=0
      do i=1,4
        xsum(i)=xmom(idec,i)-xmom(iprod1,i)-
     #          xmom(iprod2,i)-xmom(iprod3,i)
        xsuma(i)=abs(xmom(idec,i))+abs(xmom(iprod1,i))+
     #           abs(xmom(iprod2,i))+abs(xmom(iprod3,i))
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Subroutine checkstec2'
          write(*,*)'Momentum is not conserved'
          write(*,*)idec,iprod1,iprod2,iprod3
          write(*,'(4(d14.8,1x))') (xmom(idec,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod1,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod2,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod3,jj),jj=1,4)
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


C      function xbwmass3(t,xm02,ga,bwdelf,bwfmmn)
Cc Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
Cc and the mass range (implicit in bwdelf and bwfmmn). This function
Cc is the inverse of F(M^2), where
Cc   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
Cc   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
Cc and therefore eats up the Breit-Wigner when changing integration 
Cc variable M^2 --> t
C      implicit none
C      real*8 xbwmass3,t,xm02,ga,bwdelf,bwfmmn
C      real*8 pi,xm0
C      parameter (pi=3.1415926535897932d0)
Cc
C      xm0=sqrt(xm02)
C      xbwmass3=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
C      return
C      end


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
      subroutine setpst()
      implicit none
      include 'stpcblks.h'
      real * 8 pi,aem,xmzpdg,xmwpdg,cthw2,sthw2,xmw,zmw,
     #  ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm,gaw
      parameter (pi=3.14159265358979312D0)
c Values from PDG 2003
      parameter (xmzpdg=91.1876d0)
      parameter (xmwpdg=80.425d0)
      real * 8 ckm(1:6,1:6),ckm2(1:6,1:6),vickm(1:6,1:6)
      common/czmw/zmw
      common/cwparam/xmw,gaw
      common/cckm2/ckm2
      common/cvickm/vickm
      common/cckmfct/ruckm,rcckm,rtckm,rducckm,rsucckm,rbucckm
      common/cweinan/sthw2,cthw2
      integer i,j,k,l,idrmax(2:3,2)
      common/cidrmax/idrmax
      integer idrlimcp(2,2:3,8),idrlimcm(2,2:3,8)
      common/cidrlims/idrlimcp,idrlimcm
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer ie0(8,2,2:3,4)
      common/cie0/ie0
c
c Number of light flavours
      nl=5
c Fermi constant, from PDG2002
      gf=1.16639d-5
c alpha_em
      aem=1/137.0359895d0
c electron charge squared
      ze2=4*pi*aem
c sin and cos squared of theta_W; MSbar scheme, from PDG2003
      sthw2=0.23113d0
      cthw2=1-sthw2
c W mass squared: used in the propagators
      xmw=xmwpdg
      if(zmw.ne.0.d0)xmw=zmw
      xmW2=xmw**2
c ckm(i,j)=|CKM matrix elements|, with  i=1,4,6 --> up,charm,top
c                                       j=2,3,5 --> down,strange,bottom
      if(vickm(1,2).eq.0.d0.and.vickm(1,3).eq.0.d0.and.
     #   vickm(1,5).eq.0.d0)then
        do i=1,6
          do j=1,6
            ckm(i,j)=0.d0
          enddo
        enddo
c Values from PDG 2003.
c Centers of the ranges given in eq.(11.2), supposedly taking unitarity
c into account; with the following entries, it holds better than 0.1%
        ckm(1,2)=0.9748d0
        ckm(1,3)=0.2225d0
        ckm(1,5)=0.0036d0
        ckm(4,2)=0.2225d0
        ckm(4,3)=0.9740d0
        ckm(4,5)=0.041d0
        ckm(6,2)=0.009d0
        ckm(6,3)=0.0405d0
        ckm(6,5)=0.9992d0
      else
        do i=1,6
          do j=1,6
            ckm(i,j)=vickm(i,j)
          enddo
        enddo
      endif
      do i=1,6
        do j=1,6
          ckm2(i,j)=ckm(i,j)**2
        enddo
      enddo
c Combinations used in strfst; need them also for unweighting
      ruckm=ckm2(1,2)+ckm2(1,3)+ckm2(1,5)
      rcckm=ckm2(4,2)+ckm2(4,3)+ckm2(4,5)
      rtckm=ckm2(6,2)+ckm2(6,3)+ckm2(6,5)
      rducckm=ckm2(1,2)+ckm2(4,2)
      rsucckm=ckm2(1,3)+ckm2(4,3)
      rbucckm=ckm2(1,5)+ckm2(4,5)
c Fills the array idrmap used in the main code; this uses the information
c on t production to get tbar production matrix elements
      call idrfill()
c Fills the array idrmax(jproc,ich) of maximum values for idr; depends 
c on process type and channel
      do j=2,3
        do i=1,2
          idrmax(j,i)=4
        enddo
      enddo
      idrmax(3,2)=8
c idrlimcp(ich,jproc,idr) returns the idr code relevant to the Born matrix 
c element that factorizes when the y->1 collinear limit is taken in the real 
c matrix element identified by (ich,jproc,idr); idrlimcm has a similar
c meaning for the y->-1 limit. If the limit is not singular, idrlimcp=0.
      do k=1,2
        do i=2,3
          do j=1,8
            idrlimcp(k,i,j)=0
            idrlimcm(k,i,j)=0
          enddo
        enddo
      enddo
      do k=1,2
        do j=1,idrmax(2,k)
          idrlimcp(k,2,j)=j
          idrlimcm(k,2,j)=j
        enddo
      enddo
      idrlimcm(1,3,1)=1
      idrlimcm(1,3,2)=3
      idrlimcp(1,3,3)=3
      idrlimcp(1,3,4)=1
      idrlimcm(2,3,1)=1
      idrlimcm(2,3,3)=2
      idrlimcm(2,3,5)=3
      idrlimcm(2,3,7)=4
      idrlimcp(2,3,2)=1
      idrlimcp(2,3,4)=2
      idrlimcp(2,3,6)=3
      idrlimcp(2,3,8)=4
c itypemax(jproc) is the maximum value the index itype can get -- see
c the routine strfst
      itypemax(2)=9
      itypemax(3)=3
c ie0(idr,ich,jproc,ileg) returns an integer that select the shower
c scale for emissions from leg ileg in the process identified by
c (idr,ich,jproc). See the comment at the beginning of gete0sq
c for the conventions relevant to shower scales. If the branching 
c is not allowed, ie0=0
      do i=1,8
        do k=1,2
          do j=2,3
            do l=1,4
              ie0(i,k,j,l)=0
            enddo
          enddo
        enddo
      enddo
c s-channel, qq processes
      ie0(1,1,2,1)=1
      ie0(1,1,2,2)=1
      ie0(1,1,2,3)=6
      ie0(1,1,2,4)=6
      ie0(3,1,2,1)=1
      ie0(3,1,2,2)=1
      ie0(3,1,2,3)=6
      ie0(3,1,2,4)=6
c s-channel, qg processes
      ie0(1,1,3,2)=1
      ie0(2,1,3,2)=1
      ie0(3,1,3,1)=1
      ie0(4,1,3,1)=1
c t-channel, qq processes
      ie0(1,2,2,1)=2
      ie0(1,2,2,2)=5
      ie0(1,2,2,3)=2
      ie0(1,2,2,4)=5
      ie0(2,2,2,1)=2
      ie0(2,2,2,2)=5
      ie0(2,2,2,3)=2
      ie0(2,2,2,4)=5
      ie0(3,2,2,1)=3
      ie0(3,2,2,2)=4
      ie0(3,2,2,3)=4
      ie0(3,2,2,4)=3
      ie0(4,2,2,1)=3
      ie0(4,2,2,2)=4
      ie0(4,2,2,3)=4
      ie0(4,2,2,4)=3
c t-channel, qg processes
      ie0(1,2,3,2)=5
      ie0(2,2,3,1)=2
      ie0(3,2,3,2)=5
      ie0(4,2,3,1)=2
      ie0(5,2,3,2)=4
      ie0(6,2,3,1)=3
      ie0(7,2,3,2)=4
      ie0(8,2,3,1)=3
c
      return
      end


      subroutine idrfill()
c Fills the array idrmap
      implicit none
      integer idr,jproc,itt
      integer idrmap(8,2:3,2)
      common/cidrmap/idrmap
c
c t production: trivial
      itt=1
      do idr=1,8
        do jproc=2,3
          idrmap(idr,jproc,itt)=idr
        enddo
      enddo
c tbar production: charge conjugation
      itt=2
      do idr=1,8
        do jproc=2,3
          idrmap(idr,jproc,itt)=idr
        enddo
      enddo
c
      return
      end


      subroutine parsetpst()
      implicit none
      integer jproc,i,itype,ich,itt,ichconj
      integer imapp(0:5)
      integer ivbhpro(8,2:3,9,2,2)
      integer idp1(8,2:3,9,2,2),idp2(8,2:3,9,2,2)
      integer idp3(8,2:3,9,2,2),idp5(8,2:3,9,2,2)
      common/cimapp/imapp
      common/civbhpro/ivbhpro
      common/cidpart/idp1,idp2,idp3,idp5
      integer itypemax(2:3)
      common/citypemax/itypemax
      integer ichkpid
      common/cichkpid/ichkpid
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
c identical to i1hpro (see the routine store_evenst)
      do i=1,8
        do jproc=2,3
          do itype=1,9
            do ich=1,2
              do itt=1,2
                ivbhpro(i,jproc,itype,ich,itt)=0
              enddo
            enddo
          enddo
        enddo
      enddo
c
c t production
c
c s-channel
      do itype=1,itypemax(2)
        ivbhpro(1,2,itype,1,1)=401
        ivbhpro(3,2,itype,1,1)=403
      enddo
      do itype=1,itypemax(3)
        ivbhpro(1,3,itype,1,1)=402
        ivbhpro(2,3,itype,1,1)=404
        ivbhpro(3,3,itype,1,1)=405
        ivbhpro(4,3,itype,1,1)=406
      enddo
c t-channel
      do itype=1,itypemax(2)
        ivbhpro(1,2,itype,2,1)=408
        ivbhpro(2,2,itype,2,1)=401
        ivbhpro(3,2,itype,2,1)=408
        ivbhpro(4,2,itype,2,1)=403
      enddo
      do itype=1,itypemax(3)
        ivbhpro(1,3,itype,2,1)=402
        ivbhpro(2,3,itype,2,1)=405
        ivbhpro(3,3,itype,2,1)=402
        ivbhpro(4,3,itype,2,1)=406
        ivbhpro(5,3,itype,2,1)=402
        ivbhpro(6,3,itype,2,1)=405
        ivbhpro(7,3,itype,2,1)=404
        ivbhpro(8,3,itype,2,1)=405
      enddo
c
c tbar production; charge conjugation of those relevant to t production,
c using the map 
c 401->403; 402->404; 403->401; 404->402; 
c 405->406; 406->405; 408->409; 409->408
c
c s-channel
      do itype=1,itypemax(2)
        ivbhpro(1,2,itype,1,2)=403
        ivbhpro(3,2,itype,1,2)=401
      enddo
      do itype=1,itypemax(3)
        ivbhpro(1,3,itype,1,2)=404
        ivbhpro(2,3,itype,1,2)=402
        ivbhpro(3,3,itype,1,2)=406
        ivbhpro(4,3,itype,1,2)=405
      enddo
c t-channel
      do itype=1,itypemax(2)
        ivbhpro(1,2,itype,2,2)=409
        ivbhpro(2,2,itype,2,2)=403
        ivbhpro(3,2,itype,2,2)=409
        ivbhpro(4,2,itype,2,2)=401
      enddo
      do itype=1,itypemax(3)
        ivbhpro(1,3,itype,2,2)=404
        ivbhpro(2,3,itype,2,2)=406
        ivbhpro(3,3,itype,2,2)=404
        ivbhpro(4,3,itype,2,2)=405
        ivbhpro(5,3,itype,2,2)=404
        ivbhpro(6,3,itype,2,2)=406
        ivbhpro(7,3,itype,2,2)=402
        ivbhpro(8,3,itype,2,2)=406
      enddo
c
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=FKS parton, 5=non-FKS parton) in the process 
c associated to the entries. The labelling scheme of PDG has been used.
c For some partonic subprocesses in single-top production, the identity
c of one of the final-state light partons is determined statistically
c on an event-by-event basis (in ckmunwgt called by xost). In such cases,
c special non-PDG labels are used here:
c  1235 -> a down-type parton; corresponding weight: ruckm
c  4235 -> a down-type parton; corresponding weight: rcckm
c  6235 -> a down-type parton; corresponding weight: rtckm
c  142  -> an up or charm; corresponding weight: rducckm
c  143  -> an up or charm; corresponding weight: rsucckm
c  145  -> an up or charm; corresponding weight: rbucckm
c The are also processes for which the identities of both final-state
c light partons are determined at runtime, since the two are attached
c to the same weak current. In such cases, we use |code|=999 for the
c identity of the non-FKS parton, and
c  14235 -> FKS parton is an up type
c  23514 -> FKS parton is a down type
c with weight ruckm+rcckm=rducckm+rsucckm+rbucckm for both codes
      do i=1,8
        do jproc=2,3
          do itype=1,9
            do ich=1,2
              do itt=1,2
                idp1(i,jproc,itype,ich,itt)=0
                idp2(i,jproc,itype,ich,itt)=0
                idp3(i,jproc,itype,ich,itt)=0
                idp5(i,jproc,itype,ich,itt)=0
              enddo
            enddo
          enddo
        enddo
      enddo
c
c t production
c
c s-channel
c
      idp1(1,2,1,1,1)=imapp(1)
      idp1(1,2,2,1,1)=imapp(1)
      idp1(1,2,3,1,1)=imapp(1)
      idp1(1,2,4,1,1)=imapp(4)
      idp1(1,2,5,1,1)=imapp(4)
      idp1(1,2,6,1,1)=imapp(4)
c
      idp2(1,2,1,1,1)=-imapp(2)
      idp2(1,2,2,1,1)=-imapp(3)
      idp2(1,2,3,1,1)=-imapp(5)
      idp2(1,2,4,1,1)=-imapp(2)
      idp2(1,2,5,1,1)=-imapp(3)
      idp2(1,2,6,1,1)=-imapp(5)
c
      idp1(3,2,1,1,1)=-imapp(2)
      idp1(3,2,2,1,1)=-imapp(3)
      idp1(3,2,3,1,1)=-imapp(5)
      idp1(3,2,4,1,1)=-imapp(2)
      idp1(3,2,5,1,1)=-imapp(3)
      idp1(3,2,6,1,1)=-imapp(5)
c
      idp2(3,2,1,1,1)=imapp(1)
      idp2(3,2,2,1,1)=imapp(1)
      idp2(3,2,3,1,1)=imapp(1)
      idp2(3,2,4,1,1)=imapp(4)
      idp2(3,2,5,1,1)=imapp(4)
      idp2(3,2,6,1,1)=imapp(4)
c
      do itype=1,6
        idp3(1,2,itype,1,1)=imapp(0)
        idp3(3,2,itype,1,1)=imapp(0)
        idp5(1,2,itype,1,1)=-6235
        idp5(3,2,itype,1,1)=-6235
      enddo
c
      idp1(1,3,1,1,1)=imapp(1)
      idp1(1,3,2,1,1)=imapp(4)
c
      idp2(1,3,1,1,1)=imapp(0)
      idp2(1,3,2,1,1)=imapp(0)
c
      idp3(1,3,1,1,1)=1235
      idp3(1,3,2,1,1)=4235
c
      idp5(1,3,1,1,1)=-6235
      idp5(1,3,2,1,1)=-6235
c
      idp1(2,3,1,1,1)=-imapp(2)
      idp1(2,3,2,1,1)=-imapp(3)
      idp1(2,3,3,1,1)=-imapp(5)
c
      idp2(2,3,1,1,1)=imapp(0)
      idp2(2,3,2,1,1)=imapp(0)
      idp2(2,3,3,1,1)=imapp(0)
c
      idp3(2,3,1,1,1)=-142
      idp3(2,3,2,1,1)=-143
      idp3(2,3,3,1,1)=-145
c
      idp5(2,3,1,1,1)=-6235
      idp5(2,3,2,1,1)=-6235
      idp5(2,3,3,1,1)=-6235
c
      idp1(3,3,1,1,1)=imapp(0)
      idp1(3,3,2,1,1)=imapp(0)
c           
      idp2(3,3,1,1,1)=imapp(1)
      idp2(3,3,2,1,1)=imapp(4)
c
      idp3(3,3,1,1,1)=1235
      idp3(3,3,2,1,1)=4235
c
      idp5(3,3,1,1,1)=-6235
      idp5(3,3,2,1,1)=-6235
c
      idp1(4,3,1,1,1)=imapp(0)
      idp1(4,3,2,1,1)=imapp(0)
      idp1(4,3,3,1,1)=imapp(0)
c
      idp2(4,3,1,1,1)=-imapp(2)
      idp2(4,3,2,1,1)=-imapp(3)
      idp2(4,3,3,1,1)=-imapp(5)
c
      idp3(4,3,1,1,1)=-142
      idp3(4,3,2,1,1)=-143
      idp3(4,3,3,1,1)=-145
c
      idp5(4,3,1,1,1)=-6235
      idp5(4,3,2,1,1)=-6235
      idp5(4,3,3,1,1)=-6235
c
c t-channel
c
      idp1(1,2,1,2,1)=imapp(2)
      idp1(1,2,2,2,1)=imapp(3)
      idp1(1,2,3,2,1)=imapp(5)
      idp1(1,2,4,2,1)=imapp(2)
      idp1(1,2,5,2,1)=imapp(3)
      idp1(1,2,6,2,1)=imapp(5)
c
      idp2(1,2,1,2,1)=imapp(1)
      idp2(1,2,2,2,1)=imapp(1)
      idp2(1,2,3,2,1)=imapp(1)
      idp2(1,2,4,2,1)=imapp(4)
      idp2(1,2,5,2,1)=imapp(4)
      idp2(1,2,6,2,1)=imapp(4)
c
      idp1(2,2,1,2,1)=imapp(2)
      idp1(2,2,2,2,1)=imapp(2)
      idp1(2,2,3,2,1)=imapp(2)
      idp1(2,2,4,2,1)=imapp(3)
      idp1(2,2,5,2,1)=imapp(3)
      idp1(2,2,6,2,1)=imapp(3)
      idp1(2,2,7,2,1)=imapp(5)
      idp1(2,2,8,2,1)=imapp(5)
      idp1(2,2,9,2,1)=imapp(5)
c
      idp2(2,2,1,2,1)=-imapp(2)
      idp2(2,2,2,2,1)=-imapp(3)
      idp2(2,2,3,2,1)=-imapp(5)
      idp2(2,2,4,2,1)=-imapp(2)
      idp2(2,2,5,2,1)=-imapp(3)
      idp2(2,2,6,2,1)=-imapp(5)
      idp2(2,2,7,2,1)=-imapp(2)
      idp2(2,2,8,2,1)=-imapp(3)
      idp2(2,2,9,2,1)=-imapp(5)
c
      idp1(3,2,1,2,1)=imapp(1)
      idp1(3,2,2,2,1)=imapp(1)
      idp1(3,2,3,2,1)=imapp(1)
      idp1(3,2,4,2,1)=imapp(4)
      idp1(3,2,5,2,1)=imapp(4)
      idp1(3,2,6,2,1)=imapp(4)
c
      idp2(3,2,1,2,1)=imapp(2)
      idp2(3,2,2,2,1)=imapp(3)
      idp2(3,2,3,2,1)=imapp(5)
      idp2(3,2,4,2,1)=imapp(2)
      idp2(3,2,5,2,1)=imapp(3)
      idp2(3,2,6,2,1)=imapp(5)
c
      idp1(4,2,1,2,1)=-imapp(2)
      idp1(4,2,2,2,1)=-imapp(3)
      idp1(4,2,3,2,1)=-imapp(5)
      idp1(4,2,4,2,1)=-imapp(2)
      idp1(4,2,5,2,1)=-imapp(3)
      idp1(4,2,6,2,1)=-imapp(5)
      idp1(4,2,7,2,1)=-imapp(2)
      idp1(4,2,8,2,1)=-imapp(3)
      idp1(4,2,9,2,1)=-imapp(5)
c
      idp2(4,2,1,2,1)=imapp(2)
      idp2(4,2,2,2,1)=imapp(2)
      idp2(4,2,3,2,1)=imapp(2)
      idp2(4,2,4,2,1)=imapp(3)
      idp2(4,2,5,2,1)=imapp(3)
      idp2(4,2,6,2,1)=imapp(3)
      idp2(4,2,7,2,1)=imapp(5)
      idp2(4,2,8,2,1)=imapp(5)
      idp2(4,2,9,2,1)=imapp(5)
c
      do itype=1,6
        idp3(1,2,itype,2,1)=imapp(0)
        idp3(2,2,itype,2,1)=imapp(0)
        idp3(3,2,itype,2,1)=imapp(0)
        idp3(4,2,itype,2,1)=imapp(0)
      enddo
      do itype=7,9
        idp3(2,2,itype,2,1)=imapp(0)
        idp3(4,2,itype,2,1)=imapp(0)
      enddo
c
      idp5(1,2,1,2,1)=1235
      idp5(1,2,2,2,1)=1235
      idp5(1,2,3,2,1)=1235
      idp5(1,2,4,2,1)=4235
      idp5(1,2,5,2,1)=4235
      idp5(1,2,6,2,1)=4235
c
      idp5(2,2,1,2,1)=-142
      idp5(2,2,2,2,1)=-143
      idp5(2,2,3,2,1)=-145
      idp5(2,2,4,2,1)=-142
      idp5(2,2,5,2,1)=-143
      idp5(2,2,6,2,1)=-145
      idp5(2,2,7,2,1)=-142
      idp5(2,2,8,2,1)=-143
      idp5(2,2,9,2,1)=-145
c
      idp5(3,2,1,2,1)=1235
      idp5(3,2,2,2,1)=1235
      idp5(3,2,3,2,1)=1235
      idp5(3,2,4,2,1)=4235
      idp5(3,2,5,2,1)=4235
      idp5(3,2,6,2,1)=4235
c
      idp5(4,2,1,2,1)=-142
      idp5(4,2,2,2,1)=-143
      idp5(4,2,3,2,1)=-145
      idp5(4,2,4,2,1)=-142
      idp5(4,2,5,2,1)=-143
      idp5(4,2,6,2,1)=-145
      idp5(4,2,7,2,1)=-142
      idp5(4,2,8,2,1)=-143
      idp5(4,2,9,2,1)=-145
c
      idp1(1,3,1,2,1)=imapp(2)
      idp1(1,3,2,2,1)=imapp(3)
      idp1(1,3,3,2,1)=imapp(5)
c
      idp2(1,3,1,2,1)=imapp(0)
      idp2(1,3,2,2,1)=imapp(0)
      idp2(1,3,3,2,1)=imapp(0)
c
      idp3(1,3,1,2,1)=-14235
      idp3(1,3,2,2,1)=-14235
      idp3(1,3,3,2,1)=-14235
c
      idp5(1,3,1,2,1)=999
      idp5(1,3,2,2,1)=999
      idp5(1,3,3,2,1)=999
c
      idp1(2,3,1,2,1)=imapp(0)
      idp1(2,3,2,2,1)=imapp(0)
c
      idp2(2,3,1,2,1)=imapp(1)
      idp2(2,3,2,2,1)=imapp(4)
c
      idp3(2,3,1,2,1)=-6235
      idp3(2,3,2,2,1)=-6235
c
      idp5(2,3,1,2,1)=1235
      idp5(2,3,2,2,1)=4235
c
      idp1(3,3,1,2,1)=imapp(2)
      idp1(3,3,2,2,1)=imapp(3)
      idp1(3,3,3,2,1)=imapp(5)
c
      idp2(3,3,1,2,1)=imapp(0)
      idp2(3,3,2,2,1)=imapp(0)
      idp2(3,3,3,2,1)=imapp(0)
c
      idp3(3,3,1,2,1)=23514
      idp3(3,3,2,2,1)=23514
      idp3(3,3,3,2,1)=23514
c
      idp5(3,3,1,2,1)=-999
      idp5(3,3,2,2,1)=-999
      idp5(3,3,3,2,1)=-999
c
      idp1(4,3,1,2,1)=imapp(0)
      idp1(4,3,2,2,1)=imapp(0)
      idp1(4,3,3,2,1)=imapp(0)
c
      idp2(4,3,1,2,1)=-imapp(2)
      idp2(4,3,2,2,1)=-imapp(3)
      idp2(4,3,3,2,1)=-imapp(5)
c
      idp3(4,3,1,2,1)=-6235
      idp3(4,3,2,2,1)=-6235
      idp3(4,3,3,2,1)=-6235
c
      idp5(4,3,1,2,1)=-142
      idp5(4,3,2,2,1)=-143
      idp5(4,3,3,2,1)=-145
c
      idp1(5,3,1,2,1)=imapp(1)
      idp1(5,3,2,2,1)=imapp(4)
c
      idp2(5,3,1,2,1)=imapp(0)
      idp2(5,3,2,2,1)=imapp(0)
c
      idp3(5,3,1,2,1)=-6235
      idp3(5,3,2,2,1)=-6235
c
      idp5(5,3,1,2,1)=1235
      idp5(5,3,2,2,1)=4235
c
      idp1(6,3,1,2,1)=imapp(0)
      idp1(6,3,2,2,1)=imapp(0)
      idp1(6,3,3,2,1)=imapp(0)
c
      idp2(6,3,1,2,1)=imapp(2)
      idp2(6,3,2,2,1)=imapp(3)
      idp2(6,3,3,2,1)=imapp(5)
c
      idp3(6,3,1,2,1)=-14235
      idp3(6,3,2,2,1)=-14235
      idp3(6,3,3,2,1)=-14235
c
      idp5(6,3,1,2,1)=999
      idp5(6,3,2,2,1)=999
      idp5(6,3,3,2,1)=999
c
      idp1(7,3,1,2,1)=-imapp(2)
      idp1(7,3,2,2,1)=-imapp(3)
      idp1(7,3,3,2,1)=-imapp(5)
c
      idp2(7,3,1,2,1)=imapp(0)
      idp2(7,3,2,2,1)=imapp(0)
      idp2(7,3,3,2,1)=imapp(0)
c
      idp3(7,3,1,2,1)=-6235
      idp3(7,3,2,2,1)=-6235
      idp3(7,3,3,2,1)=-6235
c
      idp5(7,3,1,2,1)=-142
      idp5(7,3,2,2,1)=-143
      idp5(7,3,3,2,1)=-145
c
      idp1(8,3,1,2,1)=imapp(0)
      idp1(8,3,2,2,1)=imapp(0)
      idp1(8,3,3,2,1)=imapp(0)
c
      idp2(8,3,1,2,1)=imapp(2)
      idp2(8,3,2,2,1)=imapp(3)
      idp2(8,3,3,2,1)=imapp(5)
c
      idp3(8,3,1,2,1)=23514
      idp3(8,3,2,2,1)=23514
      idp3(8,3,3,2,1)=23514
c
      idp5(8,3,1,2,1)=-999
      idp5(8,3,2,2,1)=-999
      idp5(8,3,3,2,1)=-999
c
c tbar production; charge conjugation of those relevant to t production
c
      do i=1,8
        do jproc=2,3
          do itype=1,9
            do ich=1,2
              idp1(i,jproc,itype,ich,2)=
     #          ichconj(idp1(i,jproc,itype,ich,1))
              idp2(i,jproc,itype,ich,2)=
     #          ichconj(idp2(i,jproc,itype,ich,1))
              idp3(i,jproc,itype,ich,2)=
     #          ichconj(idp3(i,jproc,itype,ich,1))
              idp5(i,jproc,itype,ich,2)=
     #          ichconj(idp5(i,jproc,itype,ich,1))
            enddo
          enddo
        enddo
      enddo
c
      if(ichkpid.eq.0)call parcheckpst()
      return
      end


      subroutine parcheckpst()
      implicit none
      integer iallzero,iz,i,jproc,itype,ich,itt,ihpro,i1,i2,i3,i5
      parameter (iallzero=0)
      parameter (iz=0)
      integer ivbhpro(8,2:3,9,2,2)
      common/civbhpro/ivbhpro
      integer idp1(8,2:3,9,2,2),idp2(8,2:3,9,2,2)
      integer idp3(8,2:3,9,2,2),idp5(8,2:3,9,2,2)
      common/cidpart/idp1,idp2,idp3,idp5
c
      do i=1,8
        do jproc=2,3
          do itype=1,9
            do ich=1,2
              do itt=1,2
                ihpro=ivbhpro(i,jproc,itype,ich,itt)
                i1=idp1(i,jproc,itype,ich,itt)
                i2=idp2(i,jproc,itype,ich,itt)
                i3=idp3(i,jproc,itype,ich,itt)
                i5=idp5(i,jproc,itype,ich,itt)
                call parcheckfst(ihpro,i1,i2,i3,i5,iallzero,iz,
     #                           i,jproc,itype,ich,itt)
              enddo
            enddo
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfst(ihpro,i1,i2,i3,i5,iallzero,ic,
     #                       idr,jproc,itype,ich,itt)
      implicit none
      integer ihpro,i1,i2,i3,i5,iallzero,ic,idr,jproc,itype,
     # ich,itt,isum
      real*8 tiny,chrg,chin,chout,chall,chprdct
      parameter (tiny=1.d-8)
      logical ferror
c
      ferror=.false.
      if(itt.eq.1)then
        chprdct=chrg(6)
      elseif(itt.eq.2)then
        chprdct=chrg(-6)
      else
        write(*,*)'Wrong itt in parcheckfst',itt
        stop
      endif
      isum=abs(i1)+abs(i2)+abs(i3)+abs(i5)
      if(isum.ne.0)chin=chrg(i1)+chrg(i2)
      if(iallzero.eq.0)then
c i1,i2,i3,i5 must be either all nonzero, or all zero
        if( ( (i1.ne.0) .and. 
     #        (i2.eq.0.or.i3.eq.0.or.i5.eq.0) ) .or.
     #      ( (i2.ne.0) .and. 
     #        (i1.eq.0.or.i3.eq.0.or.i5.eq.0) ) .or.
     #      ( (i3.ne.0) .and. 
     #        (i1.eq.0.or.i2.eq.0.or.i5.eq.0) ) .or.
     #      ( (i5.ne.0) .and. 
     #        (i1.eq.0.or.i2.eq.0.or.i3.eq.0) ) )ferror=.true.
        if(abs(i5).ne.999)then
          if(isum.ne.0)chout=chrg(i3)+chrg(i5)
        else
          if(isum.ne.0)chout=sign(1.d0,chrg(i3))
        endif
      elseif(iallzero.eq.1)then
c all process parameters must be different from zero
        if(i1.eq.0.or.i2.eq.0.or.
     #     i3.eq.0.or.ihpro.eq.0)ferror=.true.
        if(isum.ne.0)chout=chrg(i3)+chrg(i5)
      else
        write(*,*)'parcheckfst called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        chall=chin-chout-chprdct
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or. i3.ne.21) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.eq.21 .or. i5.eq.21 ) )ferror=.true.
c 403 is qbarq
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. i3.ne.21) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.eq.21 .or. i5.eq.21 ) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.eq.21 .or. i5.eq.21 ) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.eq.21 .or. i5.eq.21 ) )ferror=.true.
c 407 is gg
        if( ihpro.eq.407 )ferror=.true.
c 408 is qq
        if( ihpro.eq.408 .and.
     #      (i1.le.0 .or. i2.le.0 .or. i3.ne.21) )ferror=.true.
c 409 is qbarqbar
        if( ihpro.eq.409 .and.
     #      (i1.ge.0 .or. i2.ge.0 .or. i3.ne.21) )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfst'
        write(*,*)'ihpro,i1,i2,i3,i5:',ihpro,i1,i2,i3,i5
        write(*,*)'idr,jproc,itype,ich,itt:',idr,jproc,itype,ich,itt
        write(*,*)'chin,chout,chprdct,chall:',chin,chout,chprdct,chall
        write(*,*)'crossing:',ic
        stop
      endif
      return
      end


      function chrg(id)
      implicit none
      real*8 chrg,tmp
      integer id,ia
      real*8 chup,chdn
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
c
      ia=abs(id)
      if(ia.eq.1.or.ia.eq.3.or.ia.eq.5)then
        tmp=chdn
      elseif(ia.eq.2.or.ia.eq.4.or.ia.eq.6)then
        tmp=chup
      elseif(ia.eq.21)then
        tmp=0.d0
      elseif(ia.eq.1235.or.ia.eq.4235.or.ia.eq.6235)then
        tmp=chdn
      elseif(ia.eq.142.or.ia.eq.143.or.ia.eq.145)then
        tmp=chup
      elseif(ia.eq.14235)then
        tmp=chup
      elseif(ia.eq.23514)then
        tmp=chdn
      else
        write(*,*)'Error in chrg: id=',id
        stop
      endif
      chrg=sign(1.d0,dfloat(id))*tmp
      return
      end


      function ichconj(ip)
c Charge conjugation
      implicit none
      integer ichconj,ip,itmp
c
      if(abs(ip).ne.21)then
        itmp=-ip
      else
        itmp=ip
      endif
      ichconj=itmp
      return
      end


      subroutine getnloiprst(iprdct0hw)
c Converts the MC@NLO process codes for single top production into the codes
c used in the NLO computation. MC@NLO conventions are
c Process: iprdct=2000+IT   s+t+W channels [W not implemented yet]
c                 2010+IT   s channel      
c                 2020+IT   t channel      
c                 2030+IT   W channel      
c with
c                 IT=0  t+tbar production  
c                 IT=1  tbar production    
c                 IT=4  t production       
c This routine in meant to be called after setting 
c iprdct0hw=mod(-iprdct0hw,10000). Furthermore
c The NLO conventions are
c     ich=1    -> s-channel
c     ich=2    -> t-channel
c     ich=3    -> W-channel [not implemented yet]
c     ittbar=1 -> t production
c     ittbar=2 -> tbar production
c and this routines sets the ranges for ich and ittbar
c 
      implicit none
      integer iprdct0hw,itmp
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer ittmin,ittmax
      common/cittrange/ittmin,ittmax
c
      if(iprdct0hw.le.2009)then
c set ichmax=3 when W-channel implemented
        ichmin=1
        ichmax=2
        itmp=iprdct0hw-2000
      elseif(iprdct0hw.le.2019)then
        ichmin=1
        ichmax=1
        itmp=iprdct0hw-2010
      elseif(iprdct0hw.le.2029)then
        ichmin=2
        ichmax=2
        itmp=iprdct0hw-2020
      elseif(iprdct0hw.le.2039)then
        write(*,*)'getnloiprst: W-channel not yet implemented'
        stop
      else
        write(*,*)'getnloiprst: wrong process number',iprdct0hw
        stop
      endif
      if(itmp.eq.0)then
        ittmin=1
        ittmax=2
      elseif(itmp.eq.1)then
        ittmin=2
        ittmax=2
      elseif(itmp.eq.4)then
        ittmin=1
        ittmax=1
      else
        write(*,*)'getnloiprst: wrong process number',iprdct0hw
        stop
      endif
      return
      end


      subroutine setpstdec()
      implicit none
      include 'stpcblks.h'
      real * 8 pi,one,zero,xme,xmmu,xmtau,ze2_tdec,xalfaem,xmwme,
     # gawme,topdecw,brtop,tmpmss(3)
      parameter (pi=3.14159265358979312D0)
      parameter (one=1.d0)
      parameter (zero=0.d0)
c Values from PDG 2003
      parameter (xme=0.510998902d-3)
      parameter (xmmu=105.6583568d-3)
      parameter (xmtau=1776.99d-3)
      real*8 xmw,gaw
      common/cwparam/xmw,gaw
      real*8 xmt,twidth
      common/ctparam/xmt,twidth
      real*8 xlep1mass,xlep2mass
      common/clepmass/xlep1mass,xlep2mass
      real*8 xm1low2,xm1upp2
      common/bounds/xm1low2,xm1upp2
      real*8 brrtop1
      common/brratios/brrtop1
      real*8 cthw2,sthw2
      common/cweinan/sthw2,cthw2
      real*8 ckm2(1:6,1:6)
      common/cckm2/ckm2
      integer idec
      common/cidec/idec
      integer iwidth
      common/ciwidth/iwidth
      integer il1hw
      common/cilhw/il1hw
c Parton identities: if top doesn't decay, 4 is top, 3 and 5 outgoing
c light partons. If top decays, (3,5)-->(3,4), and top decay products
c are 5(l), 6(nu), and 7(b)
      integer ip1s,ip2s,ip3s,ip4s,ip5s,ip6s,ip7s
      common/ci1parts/ip1s,ip2s,ip3s,ip4s,ip5s,ip6s,ip7s
c PDG codes for charged leptons and neutrinos for a given IL (NLO) code;
c the particle code (not the antiparticle) is entered here
c Charged lepton from W decay
      integer ichlw(1:3)
      data ichlw/11,13,15/
c Neutrino from W decay
      integer ineuw(1:3)
      data ineuw/12,14,16/
c
      if(idec.eq.1)then
        write(*,*)'Routine setpstdec should not be called'
        stop
      endif
c Electron charge squared (computed at the W mass, used for top decay)
      ze2_tdec = 4*pi*xalfaem(xmw)
c Lepton masses and identities: xlep#mass is the mass of lepton # in 
c the decay of top. For these, assume here that only top is produced,
c and change signs on event-by-event basis
      tmpmss(1)=xme
      tmpmss(2)=xmmu
      tmpmss(3)=xmtau
      ip5s=-ichlw(il1hw)
      ip6s=ineuw(il1hw)
      ip7s=5
      if(il1hw.le.3)then
        xlep1mass=tmpmss(il1hw)
        xlep2mass=0.d0
      else
        write(*,*)'Error in setpstdec: inconsistent entries'
        stop
      endif
c Fills MadEvent common blocks. Set positron charge and QCD coupling g 
c equal to one, and use the actual values in the main code
      xmwme=xmw
      gawme=gaw
      call setmepst(xmwme,gawme,zero,zero,
     #              xmt,twidth,zero,sthw2,one,one)
c Compute branching ratio
      if(iwidth.eq.1)then
        brrtop1=topdecw(xmt,xmw,gaw,xm1low2,xm1upp2,sthw2)/twidth
        brrtop1=ckm2(6,5)*ze2_tdec**2*brrtop1
      else
        brtop=(xm12-xmw2)**2*(xm12+2*xmw2)/(6144*pi**3*xmt**3)
        brtop=brtop * 2/(sthw2**2 * xmw2*gaw**2)
        brtop=brtop/twidth
        brrtop1=ckm2(6,5)*ze2_tdec**2*brtop
      endif
      return
      end


C      function topdecw(xmt,xmw,wwidth,xmw2low,xmw2upp,sthw2)
Cc Returns top decay width integrated over W virtuality, as computed
Cc in topwidth.m. Insert a factor e^4*|Vtb|^2 for the correct normalization
C      implicit none
C      real*8 topdecw,xmt,xmw,wwidth,xmw2low,xmw2upp,sthw2
C      real*8 pi,norm,tmp
C      parameter (pi=3.1415926535897932d0)
Cc
C      norm=1/(3072*pi**3*wwidth*xmw*sthw2**2*xmt**3)
C      tmp=( xmt**6-6*wwidth**2*xmw**4+2*xmw**6+
C     #      3*xmt**2*xmw**2*(wwidth**2-xmw**2) )*
C     #    ( atan((xmw2upp-xmw**2)/(wwidth*xmw)) -
C     #      atan((xmw2low-xmw**2)/(wwidth*xmw)) ) +
C     #    wwidth*xmw* (
C     #      (xmw2upp-xmw2low)*(4*xmw**2+xmw2low+xmw2upp-3*xmt**2)+
C     #      xmw**2*(3*xmt**2+wwidth**2-3*xmw**2)*(
C     #        log(wwidth**2*xmw**2 + (xmw2low - xmw**2)**2) - 
C     #        log(wwidth**2*xmw**2 + (xmw2upp - xmw**2)**2) ) )
C      topdecw=norm*tmp
C      return
C      end
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
      subroutine whichost(iseed,itot,mx_of_evtai,mx_of_evtbi,
     #                    mx_of_evtao,mx_of_evtbo,iunit)
c Determines the type of event at random
      implicit none
      integer iseed,itot,mx_of_evtai,mx_of_evtbi,mx_of_evtao,
     # mx_of_evtbo,iunit,i0
      real*8 xp(4),tiny,one,xsum,rnd,fk88random,prob
      parameter (tiny=1.d-4)
      logical flag
c
      if(itot.le.0)then
        write(6,*)'Fatal error #1 in whichost'
        stop
      endif
      xp(1)=dfloat(mx_of_evtai)/dfloat(itot)
      xp(2)=dfloat(mx_of_evtbi)/dfloat(itot)
      xp(3)=dfloat(mx_of_evtao)/dfloat(itot)
      xp(4)=dfloat(mx_of_evtbo)/dfloat(itot)
      one=xp(1)+xp(2)+xp(3)+xp(4)
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'Error #1 in whichost: probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.4)then
          write(6,*)'Fatal error #2 in whichost'
          stop
        endif
        i0=i0+1
        prob=xp(i0)
        xsum=xsum+prob
        if(rnd.lt.xsum)then
          flag=.false.
          itot=itot-1
          if(i0.eq.1)then
            mx_of_evtai=mx_of_evtai-1
          elseif(i0.eq.2)then
            mx_of_evtbi=mx_of_evtbi-1
          elseif(i0.eq.3)then
            mx_of_evtao=mx_of_evtao-1
          elseif(i0.eq.4)then
            mx_of_evtbo=mx_of_evtbo-1
          endif
          iunit=20+i0
        endif
      enddo
      return
      end


      subroutine crosschest(itot,mx_of_evtai,mx_of_evtbi,
     #                           mx_of_evtao,mx_of_evtbo)
c Checks whether whichost did it right
      implicit none
      integer itot,mx_of_evtai,mx_of_evtbi,mx_of_evtao,mx_of_evtbo
c
      if(itot.ne.0)then
        write(6,*)'Error: itot=',itot
        stop
      endif
      if(mx_of_evtai.ne.0)then
        write(6,*)'Error: mx_of_evtai=',mx_of_evtai
        stop
      endif
      if(mx_of_evtbi.ne.0)then
        write(6,*)'Error: mx_of_evtbi=',mx_of_evtbi
        stop
      endif
      if(mx_of_evtao.ne.0)then
        write(6,*)'Error: mx_of_evtao=',mx_of_evtao
        stop
      endif
      if(mx_of_evtbo.ne.0)then
        write(6,*)'Error: mx_of_evtbo=',mx_of_evtbo
        stop
      endif
      return
      end


      subroutine retrieve_evenst(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evenst
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3,ip4,ip5,ip6,ip7
      common/ci1part/ip1,ip2,ip3,ip4,ip5,ip6,ip7
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(8,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      if(idec.eq.0)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,
     #                                 ip4,ip5,ip6,ip7
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,3),
     #                                 ((xmom_lb(i,j),j=1,4),i=5,8)
      elseif(idec.eq.1)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,5)
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
      write(77,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,5)
      stop
 999  continue
      return
      end


      subroutine store_evenst(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip5 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_evenst), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar    -> g X
c    402        q g       -> q X
c    403        qbar q    -> g X
c    404        qbar g    -> qbar X
c    405        g q       -> q X
c    406        g qbar    -> qbar X
c    407        g g       -> g X
c    408        q q       -> g X
c    409        qbar qbar -> g X
c X being the top and non-FKS parton pair here (thus, 407 is unused).
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact,brfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3,ip4,ip5,ip6,ip7
      common/ci1part/ip1,ip2,ip3,ip4,ip5,ip6,ip7
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
c partons, 3 is the outgoing FKS parton, 4 is the top or antitop, 5 is the
c other final-state light parton. When the top decays, 6=l, 7=nu, 8=b are 
c the decay products of the top. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+5+6+7+8)=0
      real*8 xmom_lb(8,4)
      common/cxmomlb/xmom_lb
      integer iwgtnorm
      common/ciwgtnorm/iwgtnorm
      real*8 wgtaev,wgtbev
      common/cwgtev/wgtaev,wgtbev
c Branching ratio, to be inserted in the case of decayed tops
      real*8 brrtop1
      common/brratios/brrtop1
c
      if(xpmone.eq.-1)then
c Events are already stored in temporary files, and are passed to this
c routines through common blocks filled by retrieve_evenst
        xevwgt=xevsign
        xfact=1.d0
      elseif(xpmone.eq.1)then
c Events are obtained from SPRING, and are written to temporary files
c for the first time
        if(idec.eq.0)then
          np=7
          brfact=brrtop1
        elseif(idec.eq.1)then
          np=5
          brfact=1.d0
        else
          write(6,*) 'Error in store_evenst: idec=',idec
          stop
        endif
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev*brfact
        else
          xevwgt=xevsign*wgtaev*brfact
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_evenst: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      if(idec.eq.0)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7
        write(iunit,903)xevwgt
        write(iunit,904)((xfact*xmom_lb(i,j),j=1,4),i=1,3),
     #                  ((xfact*xmom_lb(i,j),j=1,4),i=5,8)
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
      subroutine xmcsust(jproc,ileg,z1,z2,xm12,s,x,yi,cth1,cth2,yj,
     #  phii,phij,xij,tk,uk,q1q,q2q,gfactsf,gfactcl,lzone,z,xmatmc)
c Computes the MC counterterms for single-top production. The
c outputs of this routine are
c  gfactsf     -> the value of G_soft
c  gfactcl     -> the value of G_coll
c  lzone(8,2)  -> .true. if the corresponding contribution to the MC
c                 counterterms is non zero, .false. otherwise
c  z(8,2)      -> Herwig shower z
c  xmatmc(8,2) -> the MC counterterm proper, multiplied by G, defined as
c                 \Sigma in the notes except for g_S^2
c For all the array outputs, (8,2) correspond to (idr,ich)
      implicit none
      real * 8 z1,z2,xm12,s,x,yi,cth1,cth2,yj,phii,phij,xij,tk,uk,
     # q1q,q2q,gfactsf,gfactcl,z(8,2),xmatmc(8,2)
      real * 8 xm22,one,two,vcf,vtf,tiny,xlim1,sbar,tbar,zhest,
     # xihest,gfunsost,gfuncost,xfact,xjac,xjac_xiztost,ap,ap_kern,
     # betae0,beta,xmcs
      real * 8 e0sq(6),de0sqdx(6),de0sqdc(6),
     # xborn(8,2),xi(8,2)
      integer jproc,ileg,itwo,ich,idr,ie0sq,icode,iborn
      logical lzone(8,2)
      character * 2 str
      parameter (str='p1')
      parameter (xm22=0.d0)
      parameter (one=1.d0)
      parameter (two=2.d0)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (tiny=1.d-4)
      parameter (itwo=2)
      real * 8 xicut,deltai,deltao
      common/parsub/xicut,deltai,deltao
      real * 8 alsfi,besfi
      common/cgfunsfi/alsfi,besfi
      real * 8 alcli,becli
      common/cgfuncli/alcli,becli
      real * 8 alsfo,besfo
      common/cgfunsfo/alsfo,besfo
      real * 8 alclo,beclo
      common/cgfunclo/alclo,beclo
      integer ichmin,ichmax
      common/cichrange/ichmin,ichmax
      integer idrmax(2:3,2)
      common/cidrmax/idrmax
      integer idrlimcp(2,2:3,8),idrlimcm(2,2:3,8)
      common/cidrlims/idrlimcp,idrlimcm
      integer ie0(8,2,2:3,4)
      common/cie0/ie0
c
      do ich=1,2
        do idr=1,8
          z(idr,ich)=0.d0
          xmatmc(idr,ich)=0.d0
        enddo
      enddo
c Compute the shower scale and the Born cross section; the entries of
c the latter are are s=(p1+p2) and t=(p1-k1)
      call gete0sq(ileg,z1,z2,xm12,s,x,yi,cth1,cth2,
     #             yj,phii,phij,tk,uk,q1q,q2q,
     #             e0sq,de0sqdx,de0sqdc)
      sbar=2*e0sq(1)
      tbar=-2*e0sq(2)+xm12
      call fstrn(sbar,tbar,itwo,xborn)
      do ich=ichmin,ichmax
        do idr=1,idrmax(jproc,ich)
          lzone(idr,ich)=.false.
          ie0sq=ie0(idr,ich,jproc,ileg)
          if(ie0sq.ne.0)then
            z(idr,ich)=zhest(ileg,e0sq(ie0sq),xm12,s,x,yi,yj,
     #                       tk,uk,q1q,q2q)
            xi(idr,ich)=xihest(ileg,e0sq(ie0sq),xm12,s,x,yi,yj,
     #                       tk,uk,q1q,q2q)
            if(ileg.le.2)then
              if( z(idr,ich).ge.0.d0.and.xi(idr,ich).ge.0.d0 .and. 
     #            z(idr,ich)**2.ge.xi(idr,ich) )lzone(idr,ich)=.true.
            elseif(ileg.eq.3)then
              if( z(idr,ich).ge.0.d0.and.xi(idr,ich).ge.0.d0 .and. 
     #            (e0sq(ie0sq)*xi(idr,ich)*z(idr,ich)**2).ge.xm12 
     #            .and. xi(idr,ich).le.1.d0 )lzone(idr,ich)=.true.
            elseif(ileg.eq.4)then
              if( z(idr,ich).ge.0.d0.and.xi(idr,ich).ge.0.d0 .and. 
     #            xi(idr,ich).le.1.d0 )lzone(idr,ich)=.true.
            else
              write(*,*)'Error in xmcsust: unknown leg',ileg
              stop
            endif
          endif
        enddo
      enddo
c Evaluate the function G, whose value depends (apart from kinematics)
c on the parameters alpha and beta, which are different in the case of
c initial- and final-state emissions (however, legs 1 and 2 will return
c the same value, and so will legs 3 and 4). The function G computed
c here will be used for the pure soft part of the MC counterterms
      if(jproc.eq.2)then
        if(ileg.le.2)then
          gfactsf=gfunsost(x,s,xm12,alsfi,besfi)
          gfactcl=gfuncost(yi,alcli,becli,deltai)
        elseif(ileg.le.4)then
          gfactsf=gfunsost(x,s,xm12,alsfo,besfo)
          gfactcl=gfuncost(yj,alclo,beclo,deltao)
        else
          write(*,*)'Error in xmcsust: unknown leg',ileg
          stop
        endif
      elseif(jproc.eq.3)then
        gfactsf=1.d0
        gfactcl=1.d0
      else
        write(*,*)'xmcsust: unknown process',jproc
        stop
      endif
c Compute MC subtraction terms 
      if(jproc.eq.2)then
        icode=4
        do ich=ichmin,ichmax
          do idr=1,idrmax(jproc,ich)
            ie0sq=ie0(idr,ich,jproc,ileg)
            if(lzone(idr,ich))then
              if(ileg.eq.1)then
                iborn=idrlimcp(ich,jproc,idr)
                if(1-x.lt.tiny)then
                  xlim1=64*vcf*(1+yi)*e0sq(ie0sq)/
     #                  (s*(s*(1-yi)+4*e0sq(ie0sq)*(1+yi)))
                elseif(1-yi.lt.tiny)then
                  xlim1=8*vcf*(1+x**2)/s
                else
                  xfact=(1-x)*(1-yi**2)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=4/s*xfact*xjac*ap/xi(idr,ich)
                endif
              elseif(ileg.eq.2)then
                iborn=idrlimcm(ich,jproc,idr)
                if(1-x.lt.tiny)then
                  xlim1=64*vcf*(1-yi)*e0sq(ie0sq)/
     #                  (s*(s*(1+yi)+4*e0sq(ie0sq)*(1-yi)))
                elseif(1+yi.lt.tiny)then
                  xlim1=8*vcf*(1+x**2)/s
                else
                  xfact=(1-x)*(1-yi**2)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=4/s*xfact*xjac*ap/xi(idr,ich)
                endif
              elseif(ileg.eq.3)then
                iborn=idr
                if(1-x.lt.tiny)then
                  betae0=sqrt(1-xm12/e0sq(ie0sq))
                  xlim1=32*vcf*e0sq(ie0sq)*betae0*(1+betae0)*(1-yj)*
     #              ((s-xm12)*(1+yj)+2*e0sq(ie0sq)*(1+betae0)*
     #              (1-betae0*yj))/( (s*(1+yj)+xm12*(1-yj))*
     #                (2*e0sq(ie0sq)*(1+betae0)*(1-yj)+
     #                s*(1+yj)-xm12*(1-yj))**2 )
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=2/s*xfact*xjac*ap/xi(idr,ich)
                endif
              elseif(ileg.eq.4)then
                iborn=idr
                if(1-x.lt.tiny)then
                  xlim1=64*vcf*e0sq(ie0sq)/
     #                  ( s*(s*(1-yj)+4*e0sq(ie0sq)*(1+yj))-
     #                    xm12*(2*s-xm12)*(1-yj) )
                elseif(1-yj.lt.tiny)then
                  xlim1=4*vcf*(s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12))/
     #                  (s*(s-xm12)*(s*x-xm12))
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=2/s*xfact*xjac*ap/xi(idr,ich)
                endif
              else
                write(*,*)'Error in xmcsust: unknown leg',ileg
                stop
              endif
              xmcs=xlim1*xborn(iborn,ich)
            else
              xmcs=0.d0
            endif
            xmatmc(idr,ich)=xmcs*gfactsf*gfactcl
          enddo
        enddo
      elseif(jproc.eq.3)then
        icode=2
        do ich=ichmin,ichmax
          do idr=1,idrmax(jproc,ich)
            ie0sq=ie0(idr,ich,jproc,ileg)
            if(lzone(idr,ich))then
              if(ileg.eq.1)then
                iborn=idrlimcp(ich,jproc,idr)
                if(1-x.lt.tiny)then
                  xlim1=0.d0
                elseif(1-yi.lt.tiny)then
                  xlim1=8*vtf*(1-x)*(x**2+(1-x)**2)/s
                else
                  xfact=(1-x)*(1-yi**2)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=4/s*xfact*xjac*ap/xi(idr,ich)
                endif
              elseif(ileg.eq.2)then
                iborn=idrlimcm(ich,jproc,idr)
                if(1-x.lt.tiny)then
                  xlim1=0.d0
                elseif(1+yi.lt.tiny)then
                  xlim1=8*vtf*(1-x)*(x**2+(1-x)**2)/s
                else
                  xfact=(1-x)*(1-yi**2)
                  xjac=xjac_xiztost(ileg,e0sq(ie0sq),de0sqdx(ie0sq),
     #                 de0sqdc(ie0sq),xm12,s,x,yi,yj,tk,uk,q1q,q2q)
                  ap=ap_kern(z(idr,ich),abs(icode))/(1-z(idr,ich))
                  xlim1=4/s*xfact*xjac*ap/xi(idr,ich)
                endif
              elseif(ileg.le.4)then
                write(*,*)'Error in xmcsust: no final-state emission'
                write(*,*)'with such jproc',jproc,ileg
                stop
              else
                write(*,*)'Error in xmcsust: unknown leg',ileg
                stop
              endif
              xmcs=xlim1*xborn(iborn,ich)
            else
              xmcs=0.d0
            endif
            xmatmc(idr,ich)=xmcs*gfactsf*gfactcl
          enddo
        enddo
      endif
      return
      end


      function gfunsost(xx,xs,xxm12,alsf,besf)
c Gets smoothly to 0 in the soft limit. The functional form is given
c in eq.(A.86) of FW, with alpha==alsf. tilde{x}_{DZ} is replaced here
c by xgsoft, and x_{DZ} by xminsf. The function is different from 1
c in the region xgsoft<x<1. Call with
c  besf<0  ==> xminsf=4*m2/S_{hadr}
c  besf>0  ==> xminsf=tilde{rho} for standard subtraction
c              xminsf=1-sqrt{zeta} for zeta-subtraction
c  |besf|-->0 ==> xgsoft-->1
c  |besf|-->1 ==> xgsoft-->xminsf
c This function has been derived from the analogous function in the
c QQ code; alsf and besf are now given in input rather than in common;
c xm12 replaced xmq2; the functional form of rho has been modified to
c render it consistent with that relevant to single top production;
c the definition of xminsf for besf>0 doesn't depend on the (soft)
c subtraction parameter any longer.
c If alsf<0, gfunsost equals 1 everywhere. This option should be used
c for testing purposes only
      implicit none
      real * 8 gfunsost,xx,xs,xxm12,alsf,besf,x,s,xm12,xminsf,xgsoft,
     # tt,tmp
      real * 8 sh
      common/shadr/sh
      real * 8 etacut
      common/cetacut/etacut
      integer isubttype
      common/cisubttype/isubttype
c
      x=xx
      s=xs
      xm12=xxm12
      if(besf.lt.0.d0)then
        xminsf=xm12/sh
      else
        if(isubttype.eq.0)then
          xminsf=xm12/s
        elseif(isubttype.eq.1)then
          xminsf=1-sqrt(etacut)
        else
          write(*,*)'Fatal error #1 in gfunsost',isubttype
          stop
        endif
      endif
      xgsoft=1.d0-(1-xminsf)*abs(besf)
      if(xgsoft.gt.0.99d0)xgsoft=0.99d0
      tt=(x-xgsoft)/(1.d0-xgsoft)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfunsost',x
        stop
      endif
      tmp=1.d0
      if(alsf.gt.0.d0)then
        if(tt.gt.0.d0.and.x.lt.0.99d0)
     #    tmp=(1-tt)**(2*alsf)/(tt**(2*alsf)+(1-tt)**(2*alsf))
        if(x.ge.0.99d0)tmp=0.d0
      endif
      gfunsost=tmp
      return
      end
 

      function gfuncost(yy,alcl,becl,delta)
c Gets smoothly to 0 in the collinear limits; the function gfunsost
c must be called before this function. The functional form is given
c in eq.(A.86) of FW, with alpha==alcl. tilde{x}_{DZ} is replaced here
c by ygcoll, and x_{DZ} by ymincl. The function is different from 1
c in the region ygcoll<|y|<1. Call with
c  becl<0  ==> ymincl=0
c  becl>0  ==> ymincl=Max(0,1-delta) for standard subtraction
c              ymincl=0 for zeta-subtraction
c  |becl|-->0 ==> ygcoll-->1
c  |becl|-->1 ==> ygcoll-->ymincl
c This function has been derived from the analogous function in the
c QQ code; alcl, becl and delta are now given in input rather than in common;
c the dependence on gacl has been eliminated (was only useful for testing
c purposes), and as a consequence the entry xx has been removed
      implicit none
      real * 8 gfuncost,yy,alcl,becl,delta,y,ymincl,ygcoll,tt,tmp
      real * 8 etacut
      common/cetacut/etacut
      integer isubttype
      common/cisubttype/isubttype
c
      y=yy
      if(becl.lt.0.d0)then
        ymincl=0.d0
      else
        if(isubttype.eq.0)then
          ymincl=max(0.d0,1.d0-delta)
        elseif(isubttype.eq.1)then
          ymincl=0.d0
        else
          write(*,*)'Fatal error #1 in gfuncost',isubttype
          stop
        endif
      endif
      ygcoll=1.d0-(1-ymincl)*abs(becl)
      if(ygcoll.gt.0.99d0)ygcoll=0.99d0
      tt=(abs(y)-ygcoll)/(1.d0-ygcoll)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfuncost',tt
        stop
      endif
      tmp=1.d0
      if(alcl.gt.0.d0)then
        if(tt.gt.0.d0.and.abs(y).lt.0.99d0)
     #    tmp=(1-tt)**(2*alcl)/(tt**(2*alcl)+(1-tt)**(2*alcl))
        if(abs(y).ge.0.99d0)tmp=0.d0
      endif
      gfuncost=tmp
      return
      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invariants
c
c
c The following routines relevant to HERWIG shower variables, as
c computed for single-top production. Legs 1,2 and 4 are massless,
c leg 3 is massive and corresponds to the top quark. The invariants
c given in input to the routines are those computed by invar_in and
c invar_out, and use FNR conventions (i.e., are defined as (p-k)^2). 
c Those used within these routines follow the notes, and therefore use
c MNR conventions (i.e., are defined as -2p.k). Using eq.(2.7) of FNR
c and the table of the draft we obtain
c
c  MNR   FNR
c  q1c = m12-s-tk-q1q
c  q2c = m22-s-uk-q2q
c  w1  = -q1q+q2q-tk
c  w2  = q1q-q2q-uk
c
      function zhest(ileg,e0sq,xm12,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns Herwig shower variable z; xm12 is the top mass squared. The
c variables x, yi, and yj are the same as in invar_in and invar_out
      implicit none
      real * 8 zhest,e0sq,xm12,s,x,yi,yj,xtk,xuk,xq1q,xq2q
      integer ileg
      real * 8 tiny,v1,v2,xa,xb,ss,w1,w2,beta2,tbeta1,eps2,zeta1,
     # beta1,tbeta2,eps1,zeta2,beta,betae0
      parameter (tiny=1.d-4)
c
c incoming parton #1 (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          zhest=1-(1-x)*(s*(1-yi)+4*e0sq*(1+yi))/(8*e0sq)
        elseif(1-yi.lt.tiny)then
          zhest=x-(1-yi)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
        else
          v1=xtk
          v2=xuk
          xa=e0sq/v1
          xb=v2/s
          ss=1-(1+xb)/xa
          if(ss.ge.0.d0)then
            zhest=2*xa*(1-sqrt(ss))
          else
            zhest=-1.d0
          endif
        endif
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          zhest=1-(1-x)*(s*(1+yi)+4*e0sq*(1-yi))/(8*e0sq)
        elseif(1+yi.lt.tiny)then
          zhest=x-(1+yi)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
        else
          v1=xtk
          v2=xuk
          xa=e0sq/v2
          xb=v1/s
          ss=1-(1+xb)/xa
          if(ss.ge.0.d0)then
            zhest=2*xa*(1-sqrt(ss))
          else
            zhest=-1.d0
          endif
        endif
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.(w1+xm12))then
          zhest=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          zhest=1-(1-x)*( (s*(1+yj)+xm12*(1-yj))/(4*e0sq*(1+betae0))+
     #      betae0*s*(beta*s-(s-xm12)*yj)/(beta*s*(s*(1+beta)+xm12)) )
        else
          beta2=1-xm12/(s-w1)
          tbeta1=sqrt(1-(w1+xm12)/e0sq)
          eps2=1-xm12/(s-w1)
          zeta1=( (2*s-(s-w1)*eps2)*w2+
     #            (s-w1)*((w1+w2)*beta2-eps2*w1) )/
     #          ( (s-w1)*beta2*(2*s-(s-w1)*eps2+(s-w1)*beta2) )
          zhest=1-tbeta1*zeta1-w1/(2*(1+tbeta1)*e0sq)
        endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          zhest=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          zhest=1-(1-x)*( (s-xm12)*(1-yj)/(8*e0sq)+
     #                    s*(1+yj)/(2*(s-xm12)) )
        elseif(1-yj.lt.tiny)then
          zhest=(s*x-xm12)/(s-xm12)+(1-yj)*(1-x)*(s*x-xm12)*
     #          ( (s-xm12)**2*(s*(1-2*x)+xm12)+
     #            4*e0sq*s*(s*x-xm12*(2-x)) )/
     #          ( 8*e0sq*(s-xm12)**3 )
        else
          beta1=sqrt((1+xm12/(s-w2))**2-4*xm12*s/(s-w2)**2)
          tbeta2=sqrt(1-w2/e0sq)
          eps1=1+xm12/(s-w2)
          zeta2=( (2*s-(s-w2)*eps1)*w1+
     #            (s-w2)*((w1+w2)*beta1-eps1*w2) )/
     #          ( (s-w2)*beta1*(2*s-(s-w2)*eps1+(s-w2)*beta1) )
          zhest=1-tbeta2*zeta2-w2/(2*(1+tbeta2)*e0sq)
        endif
      else
        write(6,*)'zhest: unknown parton number'
        stop
      endif
      return
      end


      function xihest(ileg,e0sq,xm12,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns Herwig shower variable xi; xm12 is the top mass squared. The
c variables x, yi, and yj are the same as in invar_in and invar_out
      implicit none
      real * 8 xihest,e0sq,xm12,s,x,yi,yj,xtk,xuk,xq1q,xq2q
      integer ileg
      real * 8 tiny,z,zhest,v2,v1,w1,w2,beta,betae0
      parameter (tiny=1.d-3)
c
c incoming parton #1 (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          xihest=2*s*(1-yi)/(s*(1-yi)+4*e0sq*(1+yi))
        elseif(1-yi.lt.tiny)then
          xihest=(1-yi)*s*x**2/(4*e0sq)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            v2=xuk
            xihest=2*(1+v2/(s*(1-z)))
          else
            xihest=-1.d0
          endif
        endif
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          xihest=2*s*(1+yi)/(s*(1+yi)+4*e0sq*(1-yi))
        elseif(1+yi.lt.tiny)then
          xihest=(1+yi)*s*x**2/(4*e0sq)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            v1=xtk
            xihest=2*(1+v1/(s*(1-z)))
          else
            xihest=-1.d0
          endif
        endif
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        if(e0sq.le.(w1+xm12))then
          xihest=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          xihest=beta*(1+betae0)*(s*(1+beta)+xm12)*
     #           (s*(1+yj)+xm12*(1-yj))/
     #           ( beta*(s**2*(1+yj)-2*xm12*s+xm12**2*(1-yj))+
     #            (s-xm12)**2*(s*(1+yj)+xm12*(1-yj))/s+
     #            4*e0sq*(1+betae0)*(beta*s-betae0*(s-xm12)*yj) )
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xihest=w1/(2*z*(1-z)*e0sq)
          else
            xihest=-1.d0
          endif
        endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          xihest=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          xihest=2*(s-xm12)**2*(1-yj)/
     #           ( (s-xm12)**2*(1-yj)+4*e0sq*s*(1+yj) )
        elseif(1-yj.lt.tiny)then
          xihest=(s-xm12)**2*(1-yj)/(4*e0sq*s)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xihest=w2/(2*z*(1-z)*e0sq)
          else
            xihest=-1.d0
          endif
        endif
      else
        write(6,*)'xihest: unknown parton number'
        stop
      endif
      return
      end


      function xjac_xiztost(ileg,e0sq,de0sqdx,de0sqdc,xm12,s,x,yi,yj,
     #                      xtk,xuk,xq1q,xq2q)
c Returns the jacobian d(z,xi)/d(x,c), where z and xi are Herwig shower 
c variables, and x and c are FKS variables. In the case of initial-state
c emissions, we have x=xii and c=yi; in the case of final-state emissions,
c we have x=xii and c=yj. e0sq is the shower scale squared, and
c de0sqdx=d(e0sq)/dx, de0sqdc=F*d(e0sq)/dc, with F=(1-yi^2) for legs 1 and 2,
c F=1 for leg 3, and F=(1-yj) for leg 4
      implicit none 
      real * 8 xjac_xiztost,e0sq,de0sqdx,de0sqdc,xm12,s,x,yi,yj,
     # xtk,xuk,xq1q,xq2q
      integer ileg
      real * 8 tiny,tmp,z,zhest,xi,xihest,w1,w2,beta2,tbeta1,eps2,
     # zeta1,dw1dy,dw2dx,dw1dx,dw2dy,beta1,tbeta2,eps1,dq1cdx,dq2qdx,
     # dq1cdy,dq2qdy,beta,betae0,zmo,dw2dxred,w2red
      parameter (tiny=1.d-4)
c
      tmp=0.d0
c incoming parton #1 (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          tmp=-2*s/(s*(1-yi)+4*(1+yi)*e0sq)
        elseif(1-yi.lt.tiny)then
          tmp=-s*x**2/(4*e0sq)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.lt.0.d0)then
            xjac_xiztost=0.d0
            return
          endif
          xi=xihest(ileg,e0sq,xm12,s,x,yi,yj,
     #              xtk,xuk,xq1q,xq2q)
          tmp=-s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1-yi)/(2*e0sq)+
     #             de0sqdc/(2*e0sq) )
        endif
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          tmp=2*s/(s*(1+yi)+4*(1-yi)*e0sq)
        elseif(1+yi.lt.tiny)then
          tmp=s*x**2/(4*e0sq)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.lt.0.d0)then
            xjac_xiztost=0.d0
            return
          endif
          xi=xihest(ileg,e0sq,xm12,s,x,yi,yj,
     #              xtk,xuk,xq1q,xq2q)
          tmp=s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1+yi)/(2*e0sq)-
     #             de0sqdc/(2*e0sq) )
        endif
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.(w1+xm12))then
          xjac_xiztost=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          zmo=(s*(1+yj)+xm12*(1-yj))/(4*e0sq*(1+betae0))+
     #      betae0*s*(beta*s-(s-xm12)*yj)/(beta*s*(s*(1+beta)+xm12))
          tmp=betae0*s/(4*e0sq*zmo)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xi=xihest(ileg,e0sq,xm12,s,x,yi,yj,
     #                xtk,xuk,xq1q,xq2q)
            beta2=1-xm12/(s-w1)
            tbeta1=sqrt(1-(w1+xm12)/e0sq)
            eps2=1-xm12/(s-w1)
            zeta1=( (2*s-(s-w1)*eps2)*w2+
     #              (s-w1)*((w1+w2)*beta2-eps2*w1) )/
     #            ( (s-w1)*beta2*(2*s-(s-w1)*eps2+(s-w1)*beta2) )
            dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dw2dx=(1-yj)*(s*(1+yj-x*(2*(1+yj)+x*(1-yj)))+2*xm12)/
     #            (1+yj+x*(1-yj))**2
            dw1dx=dq1cdx+dq2qdx
            dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw2dy=-2*(1-x)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw1dy=dq1cdy+dq2qdy
            tmp=-( 
     #       (dw1dy*dw2dx-dw1dx*dw2dy)*
     #          (1+tbeta1)/(s-w1-xm12)+
     #       (de0sqdx*dw2dy-de0sqdc*dw2dx)*
     #          w1*(1+tbeta1)/(e0sq*(s-w1-xm12))+
     #       (de0sqdx*dw1dy-de0sqdc*dw1dx)*(
     #          (1-(1+xi)*z+xi*z**2+tbeta1**2*zeta1)*
     #          xm12/(2*e0sq**2*tbeta1**3)+
     #          w1*w2*(1+tbeta1)/(e0sq*(s-w1-xm12)**2) ) )*
     #       tbeta1/( 2*e0sq*(1+tbeta1)*z*(1-z) )
          endif
        endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          xjac_xiztost=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          zmo=(s-xm12)*(1-yj)/(8*e0sq)+s*(1+yj)/(2*(s-xm12))
          tmp=-s/(4*e0sq*zmo)
        elseif(1-yj.lt.tiny)then
          tmp=-(s-xm12)/(4*e0sq)
        else
          z=zhest(ileg,e0sq,xm12,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xi=xihest(ileg,e0sq,xm12,s,x,yi,yj,
     #                xtk,xuk,xq1q,xq2q)
            beta1=sqrt((1+xm12/(s-w2))**2-4*xm12*s/(s-w2)**2)
            tbeta2=sqrt(1-w2/e0sq)
            eps1=1+xm12/(s-w2)
            dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dw2dxred=(s*(1+yj-x*(2*(1+yj)+x*(1-yj)))+2*xm12)/
     #               (1+yj+x*(1-yj))**2
            dw2dx=(1-yj)*dw2dxred
            dw1dx=dq1cdx+dq2qdx
            dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw2dy=-2*(1-x)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw1dy=dq1cdy+dq2qdy
            w2red=s*(1-x)*(x-xm12/s)/(2-(1-x)*(1-yj))
            tmp=-( 
     #       -(dw1dy*dw2dx-dw1dx*dw2dy)*
     #           (1+tbeta2)/(beta1*(s-w2))+
     #        (de0sqdx*dw1dy*w2-de0sqdc*dw1dx*w2red)*
     #           (1+tbeta2)/(e0sq*beta1*(s-w2))+
     #        (de0sqdx*dw2dy-de0sqdc*dw2dxred)*(1+tbeta2)*w2*(
     #           beta1**4*(s-w2)**4*(s+w1)-
     #           2*beta1**3*(s-w2)**3*(xm12*(s+w1)-w1*(s+w2))-
     #           beta1**2*(s-w2)**2*( (s*(s+xm12)+w2**2)*(s-w1-xm12)-
     #                                w2*(2*s-xm12)*(s+w1+xm12) )-
     #          (2*beta1*(s-w2)+s+w2-xm12)*xm12*(3*s+w2-xm12)*
     #          (s*(w2-w1)-(w2-xm12)*(w1+w2)) )/
     #        ( e0sq*beta1**3*(s-w2)**4*
     #          (beta1*(s-w2)+s+w2-xm12)**2 ) )*
     #        tbeta2/( 2*e0sq*(1+tbeta2)*z*(1-z) )
          endif
        endif
      else
        write(6,*)'xjac_xiztost: unknown parton number'
        stop
      endif
      xjac_xiztost=abs(tmp)
      return 
      end 


      subroutine gete0sq(ileg,z1,z2,xm12,s,x,yi,cth1,cth2,
     #                   yj,phii,phij,xtk,xuk,xq1q,xq2q,
     #                   e0sq,de0sqdx,de0sqdc)
c Returns Herwig shower scale squared, and its derivatives, defined as
c follows: de0sqdx=d(e0sq)/dx, de0sqdc=F*d(e0sq)/dc, with F=(1-yi^2) for 
c legs 1 and 2, F=1 for leg 3, and F=(1-yj) for leg 4. In the case of 
c initial-state emissions, we have x=xii and c=yi; in the case of 
c final-state emissions, we have x=xii and c=yj. The scales and their 
c derivatives are stored in arrays, with the following conventions
c   e0sq(1) --> E0^2=|p1.p2|
c   e0sq(2) --> E0^2=|p1.k1|
c   e0sq(3) --> E0^2=|p1.k2|
c   e0sq(4) --> E0^2=|p2.k1|
c   e0sq(5) --> E0^2=|p2.k2|
c   e0sq(6) --> E0^2=|k1.k2|
c and analogously for de0sqdx and for de0sqdc
      implicit none
      real * 8 z1,z2,xm12,s,x,yi,cth1,cth2,yj,phii,phij,xtk,xuk,
     # xq1q,xq2q,e0sq(6),de0sqdx(6),de0sqdc(6)
      integer ileg
      real * 8 xm22,dm12,sm12,sth1,betax,q1q,q2q,q1c,q2c,w1,w2,uk,tk,
     # y,sbar,dsbardx,dsbardc,xmn,xpl,galonred,betalon,cpsi,spsi,dtkdx,
     # dukdx,dq1qdx,dq2qdx,dxpldx,dxmndx,dtkdy,dukdy,dq1qdy,dq2qdy,
     # dxpldy,dxmndy,tbar,dtbardx,dtbardc,beta,beta1,beta2,si,sj,cphij,
     # sphij,dq1cdx,dw2dx,dq1cdy,dw2dy,ubar,dubardx,dubardc,tiny,
     # p2k1,p2k2,k1k2
      integer ia1ora2
      common/cia1ora2/ia1ora2
      parameter (tiny=1.d-4)
      parameter (xm22=0.d0)
c
      if(ia1ora2.ne.1)then
        write(*,*)'gete0sq: unknown option',ia1ora2
        stop
      endif
      dm12=xm12-xm22
      sm12=xm12+xm22
      sth1=sqrt(1-cth1**2)
      beta=sqrt(1-2*(sm12)/s+dm12**2/s**2)
      betax=sqrt(1-2*(sm12)/(x*s)+dm12**2/(x*s)**2)
      q1q = xq1q-xm12
      q2q = xq2q-xm22
      q1c = xm12-s-xtk-xq1q
      q2c = xm22-s-xuk-xq2q
      w1  = -xq1q+xq2q-xtk
      w2  = xq1q-xq2q-xuk
      uk  = xuk
      tk  = xtk
      if(ileg.eq.1.or.ileg.eq.2)then
        y=yi
        sbar=x*s
        dsbardx=s
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1+dm12/s-beta*cth1)
          dtbardx=-s/2.d0*( 1-(s-sm12)/(s*beta)*cth1-
     #            z1/(z1+z2)*beta*sqrt(1-y**2)*cth2*sth1 )
          dtbardc=0.d0
        elseif(1-y.lt.tiny)then
          tbar=-s*x/2.d0*(1+dm12/(s*x)-betax*cth1)
          dtbardx=-s/2.d0*( 1-(s*x-sm12)/(s*x*betax)*cth1 )
          dtbardc=0.d0
        elseif(1+y.lt.tiny)then
          tbar=-s*x/2.d0*(1+dm12/(s*x)-betax*cth1)
          dtbardx=-s/2.d0*( 1-(s*x-sm12)/(s*x*betax)*cth1 )
          dtbardc=0.d0
        else
          xmn=((s+uk)*z1/s-(s+tk)*z2/s)/2.d0
          xpl=((s+uk)*z1/s+(s+tk)*z2/s)/2.d0
          galonred=sqrt(xpl**2-z1*z2*tk*uk/s**2)
          betalon=-xmn/galonred
          cpsi=1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
          spsi=4*(1-x)*sqrt(x*(1-y**2))/
     #         ((1+y+x*(1-y))*(1-y+x*(1+y)))
          dtkdx=s*(1-y)/2.d0
          dukdx=s*(1+y)/2.d0
          dq1qdx=1/(4*betax*x**2)*(1+y+x*(1-y))*
     #      ( dm12*betax+(sm12-dm12**2/(s*x))*cth1 ) -
     #      s/4.d0*(1-y)*(1+dm12/(x*s)-betax*cth1)
          dq2qdx=-s/4.d0*(1+y)*( 1-dm12/(x*s)+
     #      betax*(cpsi*cth1+spsi*cth2*sth1) ) -
     #      (1-y+x*(1+y))*(sm12-dm12**2/(s*x))/
     #        (4*x**2*betax)*(cth2*sth1*spsi+cth1*cpsi) -
     #      (1-y+x*(1+y))*dm12/(4*x**2)+
     #      betax*s*( 2*(1-x**2)*(1-y**2)/
     #      ((1+y+x*(1-y))**2*(1-y+x*(1+y)))*cth1 -
     #      (1+x)*(1-y**2)*((1-y**2)*(1+x**2)-2*x*(3-y**2))/
     #      (2*(1+y+x*(1-y))**2*(1-y+x*(1+y))*sqrt(x*(1-y**2)))*
     #      cth2*sth1 )
          dxpldx=(z1*(1+y)+z2*(1-y))/4.d0
          dxmndx=(z1*(1+y)-z2*(1-y))/4.d0
          dtkdy=s*(1-x)/2.d0
          dukdy=-s*(1-x)/2.d0
          dq1qdy=-s/4.d0*(1-x)*(1+dm12/(x*s)-betax*cth1)
          dq2qdy=s/4.d0*(1-x)*( 1-dm12/(x*s)+
     #      betax*(cpsi*cth1+spsi*cth2*sth1) ) -
     #      betax*s*( -4*(1-x)**2*x*y/
     #                ((1+y+x*(1-y))**2*(1-y+x*(1+y)))*cth1 +
     #      (1-x)*x*y*((1-y**2)*(1+x**2)-2*x*(3-y**2))/
     #      ( (1+y+x*(1-y))**2*(1-y+x*(1+y))*sqrt(x*(1-y**2)) )*
     #      cth2*sth1 )
          dxpldy=-(1-x)*(z1-z2)/4.d0
          dxmndy=-(1-x)*(z1+z2)/4.d0
          tbar=-sbar/2.d0*( 1-(z2*(q1q-q1c)+z1*(q2q-q2c))/
     #                        (2*s*galonred) )
     #         -dm12/2.d0*(1-betalon) 
          dtbardx=-(s+tk+uk)/2.d0*(
     #      ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )*
     #      ( 2*dxpldx*xpl-dukdx*tk*z1*z2/s**2-dtkdx*uk*z1*z2/s**2 )/
     #      ( 4*s*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) -
     #      ( (2*dq2qdx+dukdx)*z1+(2*dq1qdx+dtkdx)*z2 )/
     #      ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) ) -
     #      (dtkdx+dukdx)/2.d0*(1 -
     #          ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )/
     #          ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) )-
     #      dm12/2.d0*( dxmndx/sqrt(xpl**2-z1*z2*tk*uk/s**2)-xmn*
     #      ( 2*dxpldx*xpl-dukdx*tk*z1*z2/s**2-dtkdx*uk*z1*z2/s**2 )/
     #      ( 2*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) )
          dtbardc=-(s+tk+uk)/2.d0*(
     #      ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )*
     #      ( 2*dxpldy*xpl-dukdy*tk*z1*z2/s**2-dtkdy*uk*z1*z2/s**2 )/
     #      ( 4*s*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) -
     #      ( (2*dq2qdy+dukdy)*z1+(2*dq1qdy+dtkdy)*z2 )/
     #      ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) ) -
     #      (dtkdy+dukdy)/2.d0*(1 -
     #          ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )/
     #          ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) )-
     #      dm12/2.d0*( dxmndy/sqrt(xpl**2-z1*z2*tk*uk/s**2)-xmn*
     #      ( 2*dxpldy*xpl-dukdy*tk*z1*z2/s**2-dtkdy*uk*z1*z2/s**2 )/
     #      ( 2*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) )
          dtbardc=(1-y**2)*dtbardc
        endif
      elseif(ileg.eq.3)then
        sbar=s
        dsbardx=0.d0
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1+xm12/s+beta*yi)
        else
          beta2=sqrt((1-dm12/(s-w1))**2-4*s*xm22/(s-w1)**2)
          tbar=-s/2.d0*(1-(q2q-q1c)/(s-w1)*beta/beta2)-dm12/2.d0
        endif
        dtbardx=0.d0
        dtbardc=0.d0
      elseif(ileg.eq.4)then
        sbar=s
        dsbardx=0.d0
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1+xm12/s+beta*yi)
          dtbardx=-s/2.d0*beta*sqrt(1-yi**2)*sqrt(1-yj**2)*
     #            cos(phij)/(1-xm12/s)
          dtbardc=0.d0
        elseif(1-yj.lt.tiny)then
          tbar=-s/2.d0*(1+xm12/s+beta*yi)
          dtbardx=0.d0
          dtbardc=0.d0
        elseif(1+yj.lt.tiny)then
c Insert this special case only to prevent a division by zero below (1/sj).
c The dtbardc is proportional to 1/sqrt(1+yj), but it is damped in the
c jacobian. Therefore, set it to zero here
          if(s*x**2-xm12.gt.0.d0)then
            tbar=-s/2.d0*(1+xm12/s+beta*yi)
          else
            tbar=-s/2.d0*(1+xm12/s-beta*yi)
          endif          
          dtbardx=0.d0
          dtbardc=0.d0
        else
          beta1=sqrt((1+dm12/(s-w2))**2-4*s*xm12/(s-w2)**2)
          si=sqrt(1-yi**2)
          sj=sqrt(1-yj**2)
          cphij=cos(phij)
          sphij=sin(phij)
          dtkdx=s/2.d0*(1-yi*yj+si*sj*cphij)
          dukdx=s/2.d0*(1+yi*yj-si*sj*cphij)
          dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
          dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
          dw2dx=(1-yj)*(s*(1+yj-x*(2*(1+yj)+x*(1-yj)))+2*xm12)/
     #          (1+yj+x*(1-yj))**2
          dtkdy=s/2.d0*(1-x)*(yi+yj*si*cphij/sj)
          dukdy=-s/2.d0*(1-x)*(yi+yj*si*cphij/sj)
          dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
          dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
          dw2dy=-2*(1-x)*(s*x-xm12)/(1+yj+x*(1-yj))**2
          tbar=-s/2.d0*(1-(q1q-q2c)/(s-w2)*beta/beta1)-dm12/2.d0
          dtbardx=s*beta/(2*beta1*(s-w2))*( dukdx-dtkdx+dq2qdx-dq1cdx +
     #            dw2dx*(q1q-q2c)*(s-w2+xm12)/
     #            ((s-w2)**2-2*xm12*(s+w2)+xm12**2) )
          dtbardc=s*beta/(2*beta1*(s-w2))*( dukdy-dtkdy+dq2qdy-dq1cdy +
     #            dw2dy*(q1q-q2c)*(s-w2+xm12)/
     #            ((s-w2)**2-2*xm12*(s+w2)+xm12**2) )
          dtbardc=(1-yj)*dtbardc
        endif
      else
        write(*,*)'gete0sq: unknown parton number',ileg
        stop
      endif
c
      ubar=-sbar-tbar
      dubardx=-dsbardx-dtbardx
      dubardc=-dsbardc-dtbardc
c E0^2=|p1.p2|
      e0sq(1)=abs(sbar/2.d0)
      de0sqdx(1)=sign(1.d0,sbar)*dsbardx/2.d0
      de0sqdc(1)=sign(1.d0,sbar)*dsbardc/2.d0
c E0^2=|p1.k1|
      e0sq(2)=abs(tbar/2.d0)
      de0sqdx(2)=sign(1.d0,tbar)*dtbardx/2.d0
      de0sqdc(2)=sign(1.d0,tbar)*dtbardc/2.d0
c E0^2=|p1.k2|
      e0sq(3)=abs(ubar/2.d0)
      de0sqdx(3)=sign(1.d0,ubar)*dubardx/2.d0
      de0sqdc(3)=sign(1.d0,ubar)*dubardc/2.d0
c E0^2=|p2.k1|
      p2k1=-(ubar-xm12+xm22)/2.d0
      e0sq(4)=abs(p2k1)
      de0sqdx(4)=sign(1.d0,-p2k1)*dubardx/2.d0
      de0sqdc(4)=sign(1.d0,-p2k1)*dubardc/2.d0
c E0^2=|p2.k2|
      p2k2=-(tbar+xm12-xm22)/2.d0
      e0sq(5)=abs(p2k2)
      de0sqdx(5)=sign(1.d0,-p2k2)*dtbardx/2.d0
      de0sqdc(5)=sign(1.d0,-p2k2)*dtbardc/2.d0
c E0^2=|k1.k2|
      k1k2=(sbar-xm12-xm22)/2.d0
      e0sq(6)=abs(k1k2)
      de0sqdx(6)=sign(1.d0,k1k2)*dsbardx/2.d0
      de0sqdc(6)=sign(1.d0,k1k2)*dsbardc/2.d0
c
      return
      end
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


C      function x2soft(xx1,xx2,xx,yy)
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
      function zgmu2_nst()
c Sets the scales for NLO subtraction terms
      implicit none
      real * 8 zgmu2_nst
      real * 8 pi,ptv1,ptv2,ptvg
      common/perpen/ptv1(2),ptv2(2),ptvg(2)
      parameter (pi=3.14159265358979312D0)
      include 'stpcblks.h'
      real * 8 ptsum,xmu2,as,alfas
      integer inloscale
      common/cinloscale/inloscale
c
      if(inloscale.eq.1)then
        ptsum = ptv1(1)**2+ptv1(2)**2+
     #          ptv2(1)**2+ptv2(2)**2+
     #          ptvg(1)**2+ptvg(2)**2
        xmu2 = ptsum/4.d0+xm12
      elseif(inloscale.eq.2)then
        xmu2 = xm12
      else
        write(*,*)'Unknown option in zgmu2_nst',inloscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nl)
      zgmu2_nst = 4.d0*pi*as
      zg = sqrt(zgmu2_nst)
      end


      function zgmu2_st()
c Sets the scales for MC subtraction terms
      implicit none
      real * 8 zgmu2_st
      real * 8 pi,ptv1,ptv2,ptvg
      common/perpen/ptv1(2),ptv2(2),ptvg(2)
      parameter (pi=3.14159265358979312D0)
      include 'stpcblks.h'
      real * 8 ptsum,xmu2,as,alfas
      integer imcscale
      common/cimcscale/imcscale
c
      if(imcscale.eq.1)then
        ptsum = ptv1(1)**2+ptv1(2)**2+
     #          ptv2(1)**2+ptv2(2)**2+
     #          ptvg(1)**2+ptvg(2)**2
        xmu2 = ptsum/4.d0+xm12
      elseif(imcscale.eq.2)then
        xmu2 = xm12
      else
        write(*,*)'Unknown option in zgmu2_st',imcscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nl)
      zgmu2_st = 4.d0*pi*as
      zg = sqrt(zgmu2_st)
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
