      SUBROUTINE QQMAIN
c Integrates heavy quark pair cross sections, and produces the event
c file which serves as input to Herwig. Quantities relevant to H and S
c events are indentified with letters a and b respectively
      implicit none
      real * 8 
     #  value(20),ecmlst(100),
     #  xmlst(100),fh1lst(100),fh2lst(100),renlst(100),
     #  fh1mclst(100),fh2mclst(100),renmclst(100),
     #  xlep1mass(2),xlep2mass(2),xmomshifts(4)
      real * 8 
     #  xm,xpdflam4,xpdflam5,tmp,xfh,xfh1,xfh2,ecm,xren,betfac,
     #  delta,deltas,deltac,dtot,avtot,ac1,ac2,xtotal,ytotal,av3a,
     #  av3nega,d3a,d3nega,av3b,av3negb,d3b,d3negb,ctime,pi,tmas,
     #  etacut,alsf,besf,alcl,becl,gacl,wgtaev,wgtbev,dummy,xmone,
     #  xfhmc,xrenmc,evfrac,evprcfrac,xares,yares,xbres,ybres,
     #  xmw,gaw,xmt,twidth,xm012,ga1,bw1delf,bw1fmmn,xm022,ga2,
     #  bw2delf,bw2fmmn,xm1low2,xm1upp2,xm2low2,xm2upp2,brrtop1,
     #  brrtop2,xmw2,gammax1,xm1low,xm1upp,gammax2,xm2low,xm2upp,
     #  bw1mdpl,bw1mdmn,bw1fmpl,bw2mdpl,bw2mdmn,bw2fmpl,al_spcfun,
     #  be_spcfun
      integer 
     #  ih1,ih2,ndns1,ndns2,jloop,iseld,nlf,ncl3,jecm,
     #  loproc,maproc,iproc,iinput,iverbose,ichkmom,
     #  ibswrite,itmpih,itmpndns,idpdfset,ipdfih,ipdfgroup,ipdfndns,
     #  ifk88istrl,ifk88ih,ifk88ndns,maxevt,it1,it2,ifuntype,
     #  ndim,nwild,itd1,itd2,ibscall,iwgtnorm,iseed0,
     #  iseed,maxtrials,mode,lo,isubttype,iprespl,iwrong,iwrong1,
     #  ntotal,ndiff,nevts,ntrls,itot,ionshell,iunita,iunitb,
     #  ioutput,ii,iunit,i,itmpqq,itmpvv,mx_of_evta,mx_of_evtb,
     #  nlfp1sch,nlas,ia1ora2,iasmc,iassoft,ifk88seed,ip4,ip5,
     #  ip6,ip7,ip8,ip9,izero,ione,idec,iwidth,il1hw,il2hw,
     #  neventsuw,ifailuw,ncntuws,nmaxuw,nqmaxuw,nqeventsuw,
     #  nqcntuws,ideconsh
      character * 2 scheme,xproc(3)
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical iphflag,evgen
      external sqq5a,sqq5b
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
      include 'hvqcblks.h'
c
c common /strfqq0/ is only in strfqq:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfqq0/ih1,ih2,ndns1,ndns2
c quark and gluon masses, used by Herwig. PDF labeling convention
      real*8 xmass(-5:21)
      common/parmass/xmass
c alsf and besf are the parameters that control gfunsoft
      common/cgfunsfp/alsf,besf
c alcl, becl and gacl are the parameters that control gfuncoll
      common/cgfunclp/alcl,becl,gacl
c al_spcfun, be_spcfun are the parameters entering spcdamp
      common/cspcpar/al_spcfun,be_spcfun
c iwgtnorm=0 for weight=+1/-1, iwgtnorm=1 otherwise
      common/ciwgtnorm/iwgtnorm
c wgtaev and wgtbev are the norms of weights for H and S events respectively
      common/cwgtev/wgtaev,wgtbev
c iprespl=0 ==> preserves rapidity
c iprespl=1 ==> preserves longitudinal momentum
      common/ciprespl/iprespl
c ichkmom=0 --> enables checks on kinematics
      common/cichkmom/ichkmom
c----------------------------------------------------------
c Variables that control the integrations
c
      common/cisubttype/isubttype
      common/betfac/betfac,delta
      common/pmerge/deltas,deltac
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
      integer nsamp
      common/samp/nsamp
c----------------------------------------------------------
c Top decay variables
c Decay of the tops: idec=0    -->   tops decay
c                    idec=1    -->   tops don't decay, or b production
      common/cidec/idec
c Mass ranges: iwidth=0    -->   W on shell
c              iwidth=1    -->   W off shell
      common/ciwidth/iwidth
c Type of W decays; il1hw and il2hw are entered following HERWIG conventions:
c     IL=1,2,3    for W   ==>  e,mu,tau
      common/cilhw/il1hw,il2hw
c W mass and width
      common/cwparam/xmw,gaw
c top mass and width; top mass squared is stored in fixvar; xmt must
c be used only in those parts of the code relevant to top decay
      common/ctparam/xmt,twidth
c W mass ranges
      common/cbw1/xm012,ga1,bw1delf,bw1fmmn
      common/cbw2/xm022,ga2,bw2delf,bw2fmmn
      common/bounmc/xm1low2,xm1upp2,xm2low2,xm2upp2
c top branching ratios
      common/brratios/brrtop1,brrtop2
c mass of leptons from W decays
      common/clepmass/xlep1mass,xlep2mass
c----------------------------------------------------------
c nlfp1sch=0 --> use nl light flavours, nlfp1sch=1 --> nl+1 scheme
      common/cnlfp1sch/nlfp1sch
c nlas is the number of light flavours used in the computation of alpha_S
      common/cnlas/nlas
c Identities of final-state particles, except for the light parton (included
c as ip3 in ci1part), according to MC particle numbering scheme; used 
c when writing the event file
      common/ci2part/ip4,ip5,ip6,ip7,ip8,ip9
c----------------------------------------------------------
c The following refer to the computation of MC subtraction terms
c ia1ora2=1 -> full invqqiants, ia1ora2=2 -> simplified invqqiants
      common/cia1ora2/ia1ora2
c iasmc=1 -> as_nlo**3, iasmc=2 -> as_mc**3, iasmc=3 -> as_mc*as_nlo**2
      common/ciasmc/iasmc
c iassoft=0 -> as_nlo(hard), iassoft=1 -> as_nlo(soft)
      common/ciassoft/iassoft
c----------------------------------------------------------
c Subprocesses: prc = 'gg', 'qq', 'qg', corresponding to jproc=1,2,3
c and equal to xproc(jproc)
      common/cxproc/xproc
c In the integration routines, loproc<=jproc<=maproc
      common/cwchproc/loproc,maproc
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c Common blocks for statistics relevant to secondary unweighting
      common/c1iunwgt/neventsuw,nqeventsuw,ifailuw
      common/c2iunwgt/ncntuws,nqcntuws,nmaxuw,nqmaxuw
c Average shifts in momenta, due to quark and lepton masses
      common/cshifts/xmomshifts
c----------------------------------------------------------
c ifuntype=1 for sqq5a, ifuntype=2 for sqq5b
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c Flag to put top decay products on shell
      common/cideconsh/ideconsh
c Common blocks for general MC@NLO routines
c common block for internal rnd number generation, independent of bases
      common/cifk88seed/ifk88seed
c common block fk88ipdfs is filled by our interface to MLMPDF
      common/fk88ipdfs/ifk88ih,ifk88ndns
c common block w50511 and w50512 are filled by PDFLIB 
      common/w50511/ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,tmas
      common/w50512/xpdflam4,xpdflam5
c----------------------------------------------------------
c- list of subprocesses
      data xproc/'gg','qq','qg'/

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

C
C------------------------------------------------------------------------
C                             START                                     -
C------------------------------------------------------------------------
c iinput=1 ==> all inputs are given by the user
      iinput=2
      mmdecide=5
c iverbose=1 ==> writes more on standard output
      iverbose=0
c ichkmom=0 ==> enables checks on kinematics
      ichkmom=1
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
      open(unit=11,file='hvqlog',status=newver)
c
CCC =================================================================
CCC   added by fabian stoeckli
CCC   read string in from file
      open(unit=33,file=stfilename,status='old')
CCC =================================================================

      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)
     #        'Enter prefix for name of BASES files'
         read (*,*) pref
      else
         read(33,*) pref
      endif
      write(11,*) ''''//pref(1:ifk88istrl(pref))//'''',
     #     '  ! prefix for BASES files'
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)
     #        'Enter prefix for name of event files'         
         read (*,*) prefev
      else
         read(33,*) prefev
      endif
      write(11,*) ''''//prefev(1:ifk88istrl(prefev))//'''',
     #     '  ! prefix for event files'
c     
c-----------------------------------------------------------------
c Kept for backward compatibility; when set to false there's 
c no photon around
      iphflag=.false.
c----------------------------------------------------------
c Parameters of the run
c
      if(iphflag) then
      write(*,*)'Only hadron collisions are implemented'
         stop
      else
         if(iinput.lt.2) then
            write(*,*)' '
            write(*,*)
     # 'Enter pair ECM(GeV),fren[NLO],ffact[NLO],fren[MC],ffact[MC]'
            write(*,*)' fren=mu_ren/mu0'
            write(*,*)' ffact=mu_fac/mu0'
            write(*,*)' mu_ren=renormalization scale'
            write(*,*)' mu_fac=factorization scale'
            write(*,*)' mu0=reference scale'
         endif
         jecm = 0
         ecm = 1.d-8
c     Allow only one set of entries in this version; disallow negative
c     entries, energies of the two beams will be entered separately eventually
         dowhile(jecm.lt.1.and.ecm.gt.0)
           if(iinput.lt.2) then
              read(*,*) ecm,xren,xfh,xrenmc,xfhmc
           else
              ecm=mmecm
              xren=mmxren
              xfh=mmxfh
              xrenmc=mmxrenmc
              xfhmc=mmxfhmc
           endif
           write(11,'(5(1x,d10.4),1x,a)') ecm,xren,xfh,xrenmc,xfhmc,
     #          '! energy, fren, ffact, frenmc, ffactmc'
           jecm=jecm+1
           ecmlst(jecm)=ecm
           fh1lst(jecm)=xfh
           fh2lst(jecm)=xfh
           renlst(jecm)=xren
c     Will use xfhmc and xrenmc in future versions
           fh1mclst(jecm)=xfh
           fh2mclst(jecm)=xfh
           renmclst(jecm)=xren
         enddo 
         if(jecm.eq.100.and.xm.gt.0) then
C            write(*,*) 'no more than 100 values'
            stop
         endif
      endif
c
c Process number (redundant with mass -- keep it for consistency with Herwig)
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)'Enter -1705 for b-bbar production'
         write(*,*)'      -1706 for t-tbar'
         
         read(*,*) itmpvv
      else
         itmpvv=mmiproc
      endif
      if(itmpvv.ne.-1705 .and. itmpvv.ne.-1706 .and.
     #   itmpvv.ne.-11705 .and. itmpvv.ne.-11706 )then
         write(*,*) 'Error: wrong IPROC'
         stop
      endif
      write(11,'(1x,i6,27x,a)') itmpvv,'! -1705/1706=b/t'
c
c Heavy quark mass
C      write(*,*)' '
      if(iinput.lt.2) then
         if(itmpvv.eq.-1705.or.itmpvv.eq.-11705)then
            write(*,*)'Enter the bottom mass (GeV)'
         elseif(itmpvv.eq.-1706.or.itmpvv.eq.-11706)then
            write(*,*)'Enter the top mass (GeV)'
         endif
         read(*,*)xm
      else
         xm=mmxmt
      endif
      write(11,'(1x,d10.4,23x,a)') xm,'! M_Q'
      if(jecm.ne.1)then
        write(*,*)'Fatal error: multiple inputs',jecm
        stop
      endif
      xmlst(jecm)=xm                             
c
c Top decay parameters
      if(itmpvv.eq.-1706.or.itmpvv.eq.-11706)then
       if(iinput.lt.2) then
          write(*,*)' '
          write(*,*)'Enter IL=1,2,3 following HERWIG conventions'
          write(*,*)'               for t->W(->l_IL nu_IL) b'
          write(*,*)'      IL=7 for undecayed tops'
          write(*,*)'for W+ and W- from top and tbar'
          read(*,*) il1hw,il2hw
       else
          il1hw=mmil1code
          il2hw=mmil2code
       endif
       write(11,'(1x,i2,1x,i2,28x,a)') il1hw,il2hw,
     #      '! 1,2,3 -> t dec, 7 -> t undec'
       if( (il1hw.eq.7.and.il2hw.ne.7) .or.
     #         (il1hw.ne.7.and.il2hw.eq.7) )then
          write(*,*) 'Ws must both decay or being stable'
          stop
       elseif(il1hw.eq.7.and.il2hw.eq.7)then
          idec=1
       elseif( (il1hw.ge.1.and.il1hw.le.3) .and.
     #         (il2hw.ge.1.and.il2hw.le.3) )then
          idec=0
       else
          write(*,*) 'Unknown options:',il1hw,il2hw
          stop
       endif
       if(idec.eq.0)then
          if(iinput.lt.2) then
             xmt=xm
             write(*,*)' '
             write(*,*)'Enter top width'
             read(*,*)twidth
          else
             twidth=mmtwidth
          endif
          write(11,'(1x,d10.4,23x,a)') twidth,'! top width'
c
          if(iinput.lt.2) then
             write(*,*)' '
             write(*,*)'Enter W mass and width (GeV)'
             read(*,*)xmw,gaw
          else
             xmw=mmxwm
             gaw=mmxww
          endif
          write(11,'(2(1x,d10.4),12x,a)') xmw,gaw,'! M_W, Gamma_W'
          xmw2=xmw**2
c
          if(iinput.lt.2) then
             write(*,*)' '
             write(*,*)'Enter GammaX, M_V1(min), M_V1(max) for W+'
             write(*,*)
     #            '  If GammaX>0, the boson mass is chosen in the range'
             write(*,*)'      M0-GammaX*width < M_W+ < M0+GammaX*width'
             write(*,*)'  and M_V1(min), M_V1(max) are ignored'
             write(*,*)
     #            '  If GammaX<0, the boson mass is chosen in the range'
             write(*,*)'            M_V1(min) < M_W+ < M_V1(max)'
             write(*,*)
     #  '  If GammaX=0, the boson mass is set equal to the pole mass'
             read(*,*)gammax1,xm1low,xm1upp
          else
             gammax1=mmv1gammax
             xm1low=mmv1massinf
             xm1upp=mmv1masssup
          endif
          write(11,'(3(1x,d10.4),1x,a)') gammax1,xm1low,xm1upp,
     #     '! GammaX, M_V1(min), M_V1(max)'
          if(gammax1.lt.0.and.xm1low.ge.xm1upp)then
            write(*,*)'Enter a non-zero range'
            stop
          endif

          if(iinput.lt.2) then
             write(*,*)' '
             write(*,*)'Enter GammaX, M_V2(min), M_V2(max) for W-'
             write(*,*)
     #            '  If GammaX>0, the boson mass is chosen in the range'
             write(*,*)'      M0-GammaX*width < M_W- < M0+GammaX*width'
             write(*,*)'  and M_V2(min), M_V2(max) are ignored'
             write(*,*)
     #            '  If GammaX<0, the boson mass is chosen in the range'
             write(*,*)'            M_V2(min) < M_W- < M_V2(max)'
             write(*,*)
     #     '  If GammaX=0, the boson mass is set equal to the pole mass'
             read(*,*)gammax2,xm2low,xm2upp
          else
             gammax2=mmv2gammax
             xm2low=mmv2massinf
             xm2upp=mmv2masssup
          endif
          write(11,'(3(1x,d10.4),1x,a)') gammax2,xm2low,xm2upp,
     #    '! GammaX, M_V2(min), M_V2(max)'
          if(gammax2.lt.0.and.xm2low.ge.xm2upp)then
            write(*,*)'Enter a non-zero range'
            stop
          endif
        else
          xmt=0.d0
          twidth=0.d0
          xmw=0.d0
          gaw=0.d0
        endif
      else
        idec=1
      endif
c
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
c Initialize parameters, such as labelling for parton processes
      call parsetpqq()
c
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)
     #   'Enter beam type for beam1 and beam2 (p, pbar, n, nbar):'
         read(*,*) part1,part2
      else
         part1=mmpart1
         part2=mmpart2
      endif
      write(11,'(1x,a,2x,a,19x,a)') ''''//part1//'''',
     #  ''''//part2//'''','! colliding hadron types'
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
            write(*,*) 'Enter Lambda_QCD_5 (GeV), < 0 for default'
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
      write(*,*) 'Lambda_5=',xlam,' GeV'
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
      schhad1=scheme
      schhad2=scheme
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
        write(*,*)'Enter the number n of light flavours (4 or 5);'
        write(*,*)
     #    ' a negative entry will force the code to use the default:'
        write(*,*)
     #    ' n=3 for m<3, n=4 for 3<m<7, n=5 for m>7'
        read(*,*) nlf
        write(11,'(1x,i2,31x,a)') nlf,'! # of light flavours'
      else
        nlf=-1
      endif
c nl+1 or nl scheme; the former is only used for bottom production
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 for the nl scheme'
        write(*,*)'      1 for the nl+1 scheme'
        read(*,*) nlfp1sch
        write(11,'(1x,i2,31x,a)') nlfp1sch,
     #    '! 0 for nl, 1 for nl+1 scheme'
      else
        nlfp1sch=-1
      endif
c-----------------------------------------------------------------
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha, beta for the function G_soft'
        write(*,*)' Defaults are: alpha=1, beta=-0.1'
        read(*,*) alsf,besf
        write(11,'(2(2x,d10.4),10x,a)') alsf,besf,
     #    '! alpha, beta (soft)'
      else
        alsf=1.d0
        besf=-0.1d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha, beta, gamma for the function G_coll'
        write(*,*)' Defaults are: alpha=1, beta=-0.1, gamma=0'
        read(*,*) alcl,becl
        write(11,'(3(1x,d10.4),1x,a)') alcl,becl,gacl,
     #    '! alpha, beta, gamma (coll)'
      else
        alcl=1.d0
        becl=-0.1d0
        gacl=0.d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha and beta for the function SPC_damp'
        write(*,*)' Defaults are: alpha=1, beta=0.5'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_spcfun,be_spcfun
        write(11,'(2(1x,d10.4),12x,a)') al_spcfun,be_spcfun,
     #    '! alpha, beta (spin corr)'
      else
        al_spcfun=1.d0
        be_spcfun=0.5d0
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
        write(*,*)'Enter 1 to use full 2->3 invqqiants'
        write(*,*)'      2 to use simplified invqqiants'
        write(*,*)' The default is 1'
        read(*,*) ia1ora2
        write(11,'(1x,i2,31x,a)') ia1ora2,
     #    '! 1 for full, 2 for simplified invqqiants'
        write(*,*)' '
        write(*,*)
     #    'For the computation of alpha_S in the MC subtraction terms'
        write(*,*)'Enter 1 to use as_nlo**3'
        write(*,*)'      2 to use as_mc**3'
        write(*,*)'      3 to use as_mc*as_nlo**2'
        write(*,*)' The default is 2'
        read(*,*) iasmc
        write(11,'(1x,i2,31x,a)') iasmc,
     #    '! 1->as_nlo^3, 2->as_mc^3, 3->as_mc*as_nlo^2'
        write(*,*)' '
        write(*,*)'When using alpha_nlo in the MC subtraction terms'
        write(*,*)'Enter 0 to use as_nlo(2->3 configuration)'
        write(*,*)'      1 to use as_nlo(soft 2->2 configuration)'
        write(*,*)' The default is 0'
        read(*,*) iassoft
        write(11,'(1x,i2,31x,a)') iassoft,
     #    '! 0->as_nlo(hard), 1->as_nlo(soft)'
      else
        ia1ora2=1
        iasmc=2
        iassoft=0
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
     #  '! 0 => wgt=+1/-1, 1 => wgt=+w/-w'
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
c-----------------------------------------------------------------
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to use standard subtraction'
        write(*,*)'      1 to use zeta subtraction'
        read(*,*)isubttype
        write(11,'(1(1x,i8),25x,a)') isubttype,
     #                               '! 0=std subt, 1=zeta subt'
      else
        isubttype=1
      endif
      if(isubttype.eq.0)then
        write(*,*)' '
        write(*,*)'Enter betfact and delta (defaults: 0.3, 0.2)'
        read(*,*)betfac,delta
        write(11,'(2(2x,d10.4),10x,a)') betfac,delta,'! betfac,delta'
      else
        write(*,*)' '
        write(*,*)'Enter zi ( [ 2*kt(gluon)/sqrt(shat) ]^2 < zi )'
        write(*,*)' Default is: zi=0.3'
        if(iinput.lt.2) then
           read(*,*) etacut
        else
           etacut=0.3
        endif
        write(11,'(1x,d10.4,23x,a)') etacut,'! zi'
        betfac = 1.d0
        delta = 1.d0
      endif
      deltas = 0
      deltac = 0
c 
c We should actually choose iprespl=0 for Herwig 6.5 and iprespl=1 for
c Herwig 6.4 or lower, but in any case versions different from 6.5 have
c Thomas precession not taken into account (a NNLO effect anyhow).
c We always recommend the most recent version
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to preserve rapidity'
        write(*,*)'      1 to preserve longitudinal momentum'
        read(*,*)iprespl
        write(11,'(1(1x,i8),25x,a)') iprespl,'! 0=y, 1=k_3 preserved'
      else
        iprespl=0
      endif
c---------------------------------------------------------------
c Select subprocess
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 1 for gg, 2 for qq, 3 for qg, 0 for all'
        write(*,*) 'to select the subprocess'
        read(*,*) iproc
        write(11,'(1x,i2,31x,a)') iproc,'! 1=gg, 2=qq, 3=qg, 0=all'
      else
        iproc=0
      endif
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to leave the partons massless'
        write(*,*)'      1 to put partons on mass shell'
        write(*,*)
     #   '      2 to put partons on mass shell, with shat constant'
        read(*,*) ionshell
        write(11,'(1x,i1,32x,a)') 
     #      ionshell,'! 0=massless, 1=massive, 2=massive, shat const'
      else
        ionshell=2
      endif
      if(ionshell.ne.0.and.ionshell.ne.1.and.ionshell.ne.2) then
        write(*,*) 'Error: enter 0 or 1 or 2.'
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
          if(ncl3.lt.0)ncl3=120000
          write(11,'(1x,i9,24x,a)')ncl3,'! # of calls for bases'
        endif
      else
        ncl3=120000
      endif
c---- close logfile
      close(11)
c----------------------------------------------------------------
c  *********************  START INTEGRATION *********************
c----------------------------------------------------------------
      ifuntype=0
      loproc = 1
      maproc = 3
      if(iproc.ne.0) then
        loproc=iproc
        maproc=iproc
      endif
c When tops decay, compute the relevant parameters
      if(idec.eq.0)then
        if( (gammax1.ne.0.d0.and.gaw.eq.0.d0) .or.
     #      (gammax2.ne.0.d0.and.gaw.eq.0.d0) )then
          write(*,*)'Non-zero mass range require non-zero widths'
          write(*,*) gammax1, gammax2,gaw
          stop
        endif
        xm012=xmw2
        xm022=xmw2
        if(gammax1.eq.0.and.gammax2.eq.0)then
          iwidth=0
          xm1low2=-1.d0
          xm1upp2=-1.d0
          xm2low2=-1.d0
          xm2upp2=-1.d0
          bw1delf=0.d0
          bw2delf=0.d0
        elseif(gammax1.ne.0.and.gammax2.ne.0)then
          iwidth=1
          ga1=gaw
          ga2=gaw
          if(gammax1.ge.0)then
            xm1low2=(max( 1.d-1,xlep1mass(1)+xlep2mass(1),
     #                    xmw-gammax1*ga1 ))**2
            xm1upp2=(min(xmt-1.d-1,xmw+gammax1*ga1))**2
          else
            xm1low2=(max(1.d-1,xlep1mass(1)+xlep2mass(1),xm1low))**2
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
c
          if(gammax2.ge.0)then
            xm2low2=(max( 1.d-1,xlep1mass(2)+xlep2mass(2),
     #                    xmw-gammax2*ga2 ))**2
            xm2upp2=(min(xmt-1.d-1,xmw+gammax2*ga2))**2
          else
            xm2low2=(max(1.d-1,xlep1mass(2)+xlep2mass(2),xm2low))**2
            xm2upp2=(min(xmt-1.d-1,xm2upp))**2
          endif
          if(xm2low2.gt.xm2upp2)then
            write(*,*)'Error in pair mass range #2'
            write(*,*)xm2low2,xm2upp2
            stop
          endif
c Parameters for the Breit Wigner function
          bw2mdpl=xm2upp2-xmw2
          bw2mdmn=xmw2-xm2low2
          bw2fmpl=atan(bw2mdpl/(xmw*ga2))
          bw2fmmn=atan(bw2mdmn/(xmw*ga2))
          bw2delf=(bw2fmpl+bw2fmmn)/pi
        else
          write(*,*)'Both mass ranges must be non-zero'
          stop
        endif
c Initialize other parameters
        call setpqq()
      endif
c
      do jloop=1,jecm
c main loop (over energies and scale factors); jecm>1, append
c loop number at prefix
         prefn = pref
         if(jecm.gt.1) call fk88strnum(prefn,jloop)
         prefnev = prefev
         if(jecm.gt.1) call fk88strnum(prefnev,jloop)
         ecm = ecmlst(jloop)
         sh = ecm**2
         xm = xmlst(jloop)
         xm2 = xm**2
         if(nlf.lt.0) then
c number of light flavours on the basis of the mass
           if(xm.lt.3) then
             nl = 3
           elseif(xm.lt.7) then
             nl = 4
           else
             nl = 5
           endif
         else
           nl=nlf
         endif
c nl used in alpha_S computation -- allow nl+1 only for charm or bottom
         nlas=nl
         if(nlfp1sch.gt.0)then
           if(xm.lt.7.and.nlfp1sch.eq.1)nlas=nl+1
           if(xm.gt.7.and.nlfp1sch.eq.1)then
             write(*,*)
     #    'nl+1 scheme should not be used for this mass value'
             stop
           endif
         else
           if(xm.lt.7)then
             nlas=nl+1
             nlfp1sch=1
           else
             nlfp1sch=0
           endif
         endif
c check inputs
         if(nl.le.4)then
           if(xmass(nl+1).ne.xm)then
             write(*,*)'Two different values have been assigned to M_Q'
             stop
           endif
         endif
c Herwig code, and consistency check
         itmpqq=-1701-nl
         if(itmpqq.ne.itmpvv.and.(-10000+itmpqq).ne.itmpvv)then
           write(*,*)'Error in process codes:',itmpqq,itmpvv,nl
           stop
         endif
c Heavy quarks identities according to MC particle numbering scheme;
c if tops decay, ip4 and ip5 are identities of decay products, set
c in setpqq()
         if(idec.eq.1)then
           ip4 = nl+1
           ip5 = -ip4
         endif
c
         xfh1 = fh1lst(jloop)
         xfh2 = fh2lst(jloop)
         xren = renlst(jloop)
c- common block values for scale factor
         xren2 = xren**2
         xf2h1 = xfh1**2
         xf2h2 = xfh2**2
c- common block values for scale factor (MC terms)
         xren2mc = renmclst(jloop)**2
         xf2h1mc = fh1mclst(jloop)**2
         xf2h2mc = fh2mclst(jloop)**2
c tau generated according to a flat distribution in (1/tau)**nsamp
         nsamp = 1
c
         ndim=6
         nwild=5
c double differential
         if(iseld.eq.1)then
           xtotal=0.d0
           ytotal=0.d0
           xares=0.d0
           yares=0.d0
           xbres=0.d0
           ybres=0.d0
           mx_of_evta=0
           mx_of_evtb=0
           fname=prefn
c
           ifuntype=1
           call fk88strcat(fname,'_a',fnamea)
           call run_bases(sqq5a,fnamea,ndim,nwild,ncl3,it1,it2,
     #       ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #       ibswrite,ibscall)
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
           call fk88strcat(fname,'_b',fnameb)
           call run_bases(sqq5b,fnameb,ndim,nwild,ncl3,it1,it2,
     #       ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #       ibswrite,ibscall)
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
     #          form='formatted',status='unknown')
           write(21,240)xares
           write(21,240)xbres
           write(21,240)yares
           write(21,240)ybres
           close(21)
 240       format(1x,d14.8)
         endif
c Sanity check
         if(isubttype.eq.1.and.(betfac.ne.1.d0.or.delta.ne.1.d0))then
           write(*,*)'Fatal error: betfac, delta=',betfac,delta
           stop
         endif
         if(iseld.eq.0)then
c Read integrals from disk only if the integration step has been skipped
           call fk88strcat(prefn,'.integrals',fname)
           open(unit=21,file=fname,
     #          form='formatted',status='old')
           read(21,240)xares
           read(21,240)xbres
           read(21,240)yares
           read(21,240)ybres
           close(21)
         endif
c Generates events when evgen=.true.; if evgen=.false., maxevt=100000 in
c order to estimate the number of negative weights
         if(maxevt.ne.0)then
           ntotal=0
           xtotal=0.d0
           ytotal=0.d0
           xtotal=xtotal+xares+xbres
           ytotal=ytotal+yares+ybres
           avtot=ytotal
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
     #               (xares+yares)
           evprcfrac=evprcfrac/(1+evprcfrac)
           evfrac=evfrac+evprcfrac*mx_of_evta
           if(basesoutput.gt.0) then
           write(*,*)'Events[a]: w<0/all:',evprcfrac
           endif
           evprcfrac=(xbres-ybres)/
     #               (xbres+ybres)
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
     #          form='formatted',status='unknown')
           write(22,250)mx_of_evta
           close(22)
           call fk88strcat(fname,'_b.events',fname1)
           open(unit=22,file=fname1,
     #          form='formatted',status='unknown')
           write(22,250)mx_of_evtb
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
           call fk88strcat(fname,'_a',fnamea)
           call fk88strcat(fnamev,'_a.events',fname1)
           open(unit=22,file=fname1,
     #          form='formatted',status='old')
           call toend(22)
           call run_spring(sqq5a,fnamea,mx_of_evta,maxtrials,
     #                     nevts,ntrls,ndim,nwild,iseed)
           close(22)
           if(iverbose.eq.1)then
             write(*,*)'   '
             write(*,*)'Events[a]'
             write(*,*)'Trials:',ntrls
             write(*,*)'Events generated:',nevts
             write(*,*)'Unlike sign events(1):',iwrong
             write(*,*)'Unlike sign events(2):',iwrong1
             write(*,*)'Unlike sign(1)/all events:',
     #                 iwrong/dfloat(nevts)
             write(*,*)'Unlike sign(2)/all events:',
     #                 iwrong1/dfloat(nevts)
             if(idec.eq.0)then
               if(neventsuw.ne.mx_of_evta)then
                 write(*,*)'Error in spin correlations [a]'
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
     #                     nqeventsuw/dfloat(nqcntuws)
               endif
             endif
             write(*,*)'   '
             write(*,*)'Average momentum shifts due to masses'
             do i=1,4
               if(idec.eq.0)then
                 write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(11*nevts)
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
           ifuntype=2
           call fk88strcat(fname,'_b',fnameb)
           call fk88strcat(fnamev,'_b.events',fname1)
           open(unit=22,file=fname1,
     #          form='formatted',status='old')
           call toend(22)
           call run_spring(sqq5b,fnameb,mx_of_evtb,maxtrials,
     #                     nevts,ntrls,ndim,nwild,iseed)
           close(22)
           if(iverbose.eq.1)then
             write(*,*)'   '
             write(*,*)'Events[b]'
             write(*,*)'Trials:',ntrls
             write(*,*)'Events generated:',nevts
             write(*,*)'Unlike sign events(1):',iwrong
             write(*,*)'Unlike sign events(2):',iwrong1
             write(*,*)'Unlike sign(1)/all events:',
     #                 iwrong/dfloat(nevts)
             write(*,*)'Unlike sign(2)/all events:',
     #                 iwrong1/dfloat(nevts)
             if(idec.eq.0)then
               if(neventsuw.ne.mx_of_evtb)then
                 write(*,*)'Error in spin correlations [b]'
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
     #                     nqeventsuw/dfloat(nqcntuws)
               endif
             endif
             write(*,*)'   '
             write(*,*)'Average momentum shifts due to masses'
             do i=1,4
               if(idec.eq.0)then
                 write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(10*nevts)
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
     #          status='unknown')
c Write all the quantities which identify the run
           write(ioutput,801)
     #       ecmlst(jloop),renlst(jloop),fh1lst(jloop),
     #       renmclst(jloop),fh1mclst(jloop),
     #       '--> CM energy, muR/mu0[NLO], muF/mu0[NLO], '//
     #       'muR/mu0[MC], muF/mu0[MC]'
           write(ioutput,802)abs(itmpvv),'--> 1705/1706=b/t'
           if(itmpvv.eq.-1705.or.itmpvv.eq.-11705)then
             write(ioutput,814)xmlst(jloop),'--> M_b'
           else
             write(ioutput,803)xmlst(jloop),twidth,
     #                         '--> M_top, Gamma_top'
             write(ioutput,803)xmw,gaw,'--> M_W, Gamma_W'
             write(ioutput,810)il1hw,il2hw,'--> IL1, IL2 (1,2,3,7)'
           endif
           write(ioutput,804)xmass(1),xmass(2),
     #                       xmass(3),xmass(4),
     #                       xmass(5),xmass(21),
     #                       '--> quark and gluon masses'
           write(ioutput,805)part1,part2,'--> colliding particles'
           write(ioutput,806)gname(1:8),idpdfset,
     #       '--> PDF group and id number'
           write(ioutput,807)xlam,scheme,'--> Lambda_5, scheme'
           write(ioutput,811)'P,M','--> Format of v3.1 and higher'
           write(ioutput,250)maxevt
           itot=maxevt
           do ii=1,maxevt
             call whichoqq(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
             call retrieve_evenqq(iunit,ii,dummy)
             call store_evenqq(ioutput,xmone)
           enddo
           call crosscheqq(itot,mx_of_evta,mx_of_evtb)
           close(iunita)
           close(iunitb)
           close(ioutput)
 111       continue
         endif
         if(basesoutput.gt.0) then
         if(idec.eq.0)then
           write(*,*) '   '
           write(*,*)'Branching ratios used in the computation:'
           write(*,*)' BR(t -> b e+ nu)=          ',brrtop1
           write(*,*)' BR(tbar -> bbar e- nubar)= ',brrtop2
         endif 
         write(*,*) '   '
         write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
         write(*,'(a)')
     #  ' ecm           mass      f1   f2   r    tot        err'
         write(*,300)
     #      abs(ecmlst(jloop)),xmlst(jloop),
     #      fh1lst(jloop),fh2lst(jloop),renlst(jloop),avtot,dtot
         endif
c end of the main loop
      enddo
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #  '  nl=',i2,'  lambda5=',d10.4)
 300  format((1x,1pd9.3),4x,(1x,1pd9.3),3(1x,0pf4.2),
     # 2(1x,0pd10.4,1x,1pd6.0))
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


CC      subroutine toend(iunit)
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


      subroutine strfqq(x1,x2,sf)
c Return parton densities through the matrix
c  sf(idr,jproc,itype), with the following conventions:
c  idr=1 -> direct events
c  idr=2 -> charge-conjugated events
c  idr=3 -> reflected events
c  idr=4 -> charge-conjugated and reflected events
c  jproc=1,2,3 -> gg, qqbar, qg processes respectively
c  itype -> identifies the individual contribution to a given jproc
c
      implicit none
      real*4 fh1x1(-5:5),fh2x2(-5:5),fh1x2(-5:5),fh2x1(-5:5),
     #  smuf2h1,smuf2h2
      real*8 pi,x1,x2,sf(4,3,5)
      integer ih1,ih2,ndns1,ndns2,i,jproc,itype
      parameter(pi=3.14159265358979312D0)
      include 'hvqcblks.h'
      common/strfqq0/ih1,ih2,ndns1,ndns2
      integer ipdfscale
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c
      do i=1,4
        do jproc=1,3
          do itype=1,5
            sf(i,jproc,itype)=0.d0
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
        write(*,*)'Fatal error in strfqq: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x2),fh1x2,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x1),fh2x1,5)
c
      sf(1,1,1) = dble(fh1x1(0)*fh2x2(0))/4.d0
      sf(2,1,1) = sf(1,1,1)
      sf(3,1,1) = dble(fh2x1(0)*fh1x2(0))/4.d0
      sf(4,1,1) = sf(3,1,1)
c
      do itype=1,nl
        sf(1,2,itype) = dble(fh1x1( itype) * fh2x2(-itype))/2
        sf(2,2,itype) = dble(fh1x1(-itype) * fh2x2( itype))/2
        sf(3,2,itype) = dble(fh2x1( itype) * fh1x2(-itype))/2
        sf(4,2,itype) = dble(fh2x1(-itype) * fh1x2( itype))/2
      enddo
c
      do itype=1,nl
        sf(1,3,itype) = dble(fh1x1( itype) * fh2x2( 0))
        sf(2,3,itype) = dble(fh1x1(-itype) * fh2x2( 0))
        sf(3,3,itype) = dble(fh2x1( itype) * fh1x2( 0))
        sf(4,3,itype) = dble(fh2x1(-itype) * fh1x2( 0))
      enddo
c
      return
      end


      function sqq5a(xx)
c Integrand function for H events
      implicit none
      real * 8 sqq5a,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'hvqcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,th2,x,y,csi,cth1,rx,
     # ximax0,ximin0,ymax,ymin,s,ro,rox,rohx,tmp,tqq5a,
     # taumax,xxa1,xxa2,xxc,xxymax,xxymin
c
c xx(1) --> tau, xx(2) --> ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> th2
c
      xjac = 1
      roh   = 4*xm2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-roh)
      xjac = xjac * (1-roh)
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
c
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohx = roh/x
      taumax = 1/x
      ximax0 = rohx**(-nsamp)
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
c
      s = sh * tau
      ro = roh/tau
c
c Change variables from xx(5) to cth1, xjac--> xjac * d cth1/d xx(5)
c
      rox  = ro/x
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * pi
      xjac = xjac * pi
c
      sqq5a = tqq5a(s,x,y,cth1,th2,xjac)
      return
      end


      function tqq5a(s,xx,xy,xcth1,xth2,xjac)
c Implements standard subtraction
      implicit none
      character*2 str
      parameter (str='p1')
      real*8 tqq5a,s,xx,xy,xcth1,xth2,xjac
      real*8 pi,pi2,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      integer ione
      parameter (ione=1)
      character*2 xproc(3)
      common/cxproc/xproc
      real*8 betfac,delta,deltas,deltac
      common/betfac/betfac,delta
      common/pmerge/deltas,deltac
      include 'hvqcblks.h'
      real*8 ycm,tau
      common/x1x2/ycm,tau
      real*8 sf(4,3,5)
      integer ipdfscale
      common/cipdfscale/ipdfscale
      integer idec
      common/cidec/idec
      real*8 bsfsgn
      common/cbssgn/bsfsgn
      real*8 bsewgt
      common/cbswgt/bsewgt
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real*8 vv(4,3,5),vvs(4,3,5)
      common/cvv/vv
      real*8 x,y,cth1,th2,cth2,sx,rox,bx,ro,beta,
     # x1,x2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,zg2,
     # xnorm,f,xint,www,x1t,x2t,xtmp,ytmp,
     # zgmu2_nqq,zgmu6_mc,zg6,xphsp,xphspcm,xphspcp,xfact,
     # x1soft,x2soft,x1x2j,x1x2jqq,zhwfct,xsum,dummy,fpp,
     # gfactsf,gfactcl,xnormmc,xphspmc,betamc
      real*8 xmcxsec(1:4,1:3),xmce0sq(1:4,1:3),xmcz(1:4,1:3)
      real*8 xqrksc(1:3,1:4,1:3),xqbrsc(1:3,1:4,1:3)
      integer i,itype,jproc,loproc,maproc,ileg,ie0sq,i2b,
     # iret,itoosoftkin
      common/cjproc/jproc
      common/cwchproc/loproc,maproc
      logical flxsec(1:4,1:3),flagmc,fx1x2
c
      x=xx
      y=xy
      cth1=xcth1
      th2=xth2
      cth2=cos(th2)
c
      sx=s*x
      ro=4*xm2/s
      beta=sqrt(1-ro)
      rox=4*xm2/sx
      bx=sqrt(1-rox)
c
      x1=sqrt(tau)*exp(ycm)
      x2=tau/x1
c
      do jproc=1,3
        do i=1,4
          do itype=1,nl
            vv(i,jproc,itype)=0.d0
            vvs(i,jproc,itype)=0.d0
          enddo
        enddo
      enddo
      xnorm=xjac / (s * 64*pi2 * 16*pi2)
      xphsp=bx/(1-x)*( 1/(1-y) + 1/(1+y) ) 
      xphspcp=bx/((1-x)*(1-y))
      xphspcm=bx/((1-x)*(1+y))
      xnormmc=xjac / (64*pi2 * 16*pi2)
      xphspmc=1/(1-x)*( 1/(1-y) + 1/(1+y) ) 
c
c Event
c
      if(x1.lt.1.and.x2.lt.1)then
        ipdfscale=1
        call invqq(xm2,s,x,y,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        zg2=zgmu2_nqq()
        do jproc=loproc,maproc
          prc=xproc(jproc)
          f=fpp(s,x,y,xm2,q1q,q2q,w1h,w2h,cth2)
          www=zg2**3*xnorm*xphsp*f
          call strfqq(x1,x2,sf)
          do i=1,4
            do itype=1,nl
              vv(i,jproc,itype)=sf(i,jproc,itype)*www
            enddo
          enddo
        enddo
      endif
c
c MC subt term: pure MC
c
      ipdfscale=2
      do jproc=loproc,maproc
        call xmcsuqq(x1,x2,xm2,s,x,y,cth1,cth2,
     #    xmcxsec,xmce0sq,xmcz,xqrksc,xqbrsc,flxsec,flagmc,
     #    gfactsf,gfactcl)
        if(flagmc)then
          do ileg=1,4
            do ie0sq=1,3
              if(flxsec(ileg,ie0sq))then
                if(ileg.eq.1)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)/zhwfct
                  x2t=x2soft(x1,x2,x,y)
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1
                elseif(ileg.eq.2)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)
                  x2t=x2soft(x1,x2,x,y)/zhwfct
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1
                else
                  zhwfct=1.d0
                  x1t=x1
                  x2t=x2
                  betamc=beta
                  fx1x2=x1t.lt.1.and.x2t.lt.1
                endif
                if(fx1x2)then
                  zg6=zgmu6_mc(ileg,xm2,s,x,y,cth1,cth2,
     #                  xqrksc(1,ileg,ie0sq),xqrksc(2,ileg,ie0sq),
     #                  xqrksc(3,ileg,ie0sq),xqbrsc(1,ileg,ie0sq),
     #                  xqbrsc(2,ileg,ie0sq),xqbrsc(3,ileg,ie0sq))
                  x1x2j=x1x2jqq(x1,x2,x,y,ileg)/zhwfct
                  www=-zg6*xnormmc*xphspmc*x1x2j*betamc*
     #                 xmcxsec(ileg,ie0sq)
                  call strfqq(x1t,x2t,sf)
                  do i=1,4
                    do itype=1,nl
                      vv(i,jproc,itype)=vv(i,jproc,itype)+
     #                  sf(i,jproc,itype)*www
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
        endif
      enddo
c
c MC subt term: collinear ME
c
      if(gfactcl.lt.1.d0)then
        if(y.gt.0.d0)then
          ytmp=1.d0
          x1t=x1soft(x1,x2,x,y)/x
          x2t=x2soft(x1,x2,x,y)
          xfact=xnorm*xphspcp
        else
          ytmp=-1.d0
          x1t=x1soft(x1,x2,x,y)
          x2t=x2soft(x1,x2,x,y)/x
          xfact=xnorm*xphspcm
        endif
        if(x1t.lt.1.and.x2t.lt.1.and.x1.lt.1.and.x2.lt.1)then
          ipdfscale=1
          x1x2j=x1x2jqq(x1,x2,x,y,ione)/x
          call invqq(xm2,s,x,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2=zgmu2_nqq()
          do jproc=loproc,maproc
            prc=xproc(jproc)
            f=fpp(s,x,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
            www=-zg2**3*xfact*x1x2j*f*(1-gfactcl)
            call strfqq(x1t,x2t,sf)
            do i=1,4
              do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*www
              enddo
            enddo
          enddo
        endif
      endif
c
c MC subt term: soft ME
c
      if(gfactsf.lt.1.d0)then
        xtmp=1.d0
        x1t=x1soft(x1,x2,x,y)
        x2t=x2soft(x1,x2,x,y)
        if(x1t.lt.1.and.x2t.lt.1.and.x1.lt.1.and.x2.lt.1)then
          ipdfscale=1
          x1x2j=x1x2jqq(x1,x2,x,y,ione)
          call invqq(xm2,sx,xtmp,y,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2=zgmu2_nqq()
          do jproc=loproc,maproc
            prc=xproc(jproc)
            f=fpp(sx,xtmp,y,xm2,q1q,q2q,w1h,w2h,cth2)
            www=-zg2**3*(xnorm/x)*xphsp*x1x2j*f*(1-gfactsf)
            call strfqq(x1t,x2t,sf)
            do i=1,4
              do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*www
              enddo
            enddo
          enddo
        endif
      endif
c
c MC subt term: soft-collinear ME
c
      if(gfactsf.lt.1.d0.and.gfactcl.lt.1.d0)then
        if(y.gt.0.d0)then
          ytmp=1.d0
          xfact=(xnorm/x)*xphspcp
        else
          ytmp=-1.d0
          xfact=(xnorm/x)*xphspcm
        endif
        xtmp=1.d0
        x1t=x1soft(x1,x2,x,y)
        x2t=x2soft(x1,x2,x,y)
        if(x1t.lt.1.and.x2t.lt.1.and.x1.lt.1.and.x2.lt.1)then
          ipdfscale=1
          x1x2j=x1x2jqq(x1,x2,x,y,ione)
          call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2=zgmu2_nqq()
          do jproc=loproc,maproc
            prc=xproc(jproc)
            f=fpp(sx,xtmp,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
            www=zg2**3*xfact*x1x2j*f*(1-gfactsf)*(1-gfactcl)
            call strfqq(x1t,x2t,sf)
            do i=1,4
              do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*www
              enddo
            enddo
          enddo
        endif
      endif
c
      call checkqq(xsum,dummy,iret)
      if(iret.eq.1)then
        do jproc=loproc,maproc
          do i=1,4
            do itype=1,nl
              vvs(i,jproc,itype)=vv(i,jproc,itype)
            enddo
          enddo
        enddo
        call invqq(xm2,s,x,y,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        if(idec.eq.0)then
          ps=s
          px=x
          py=y
          pcth1=cth1
          pcth2=cth2
        endif
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(idec.eq.0)then
            ps=sx
            px=xtmp
            py=ytmp
            pcth1=cth1
            pcth2=cth2
          endif
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tqq5a=abs(xint)
c
      return
      end


      function sqq5b(xx)
c Integrand function for S events
      implicit none
      real * 8 sqq5b,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'hvqcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,th2,x,y,csi,cth1,rx,
     # ximax0,ximin0,ymax,ymin,s,ro,rox,rohx,tmp,tqq5b,
     # taumax,xxa1,xxa2,xxc,xxymax,xxymin
c
c xx(1) --> tau, xx(2) --> ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> th2
c
      xjac = 1
      roh   = 4*xm2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-roh)
      xjac = xjac * (1-roh)
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
c
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohx = roh/x
      taumax = 1/x**2
      ximax0 = rohx**(-nsamp)
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
c
      s = sh * tau
      ro = roh/tau
c
c Change variables from xx(5) to cth1, xjac--> xjac * d cth1/d xx(5)
c
      rox  = ro/x
      call zzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * pi
      xjac = xjac * pi
c
      sqq5b = tqq5b(s,x,y,cth1,th2,xjac)
      return
      end


      function tqq5b(s,xx,xy,xcth1,xth2,xjac)
      implicit none
      real * 8 tqq5b,tqq5bs,tqq5bz,s,xx,xy,xcth1,xth2,
     #  x,y,cth1,th2,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      x      = xx
      y      = xy
      cth1   = xcth1
      th2    = xth2
      if(isubttype.eq.0)then
        tmp=tqq5bs(s,x,y,cth1,th2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tqq5bz(s,x,y,cth1,th2,xjac)
      else
        write(*,*)'Fatal error in tqq5b:',isubttype
        stop
      endif
      tqq5b=tmp
      return
      end


      function tqq5bs(s,xx,xy,xcth1,xth2,xjac)
c Implements standard subtraction
      implicit none
      character*2 str
      parameter (str='p1')
      real*8 tqq5bs,s,xx,xy,xcth1,xth2,xjac
      real*8 pi,pi2,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      integer ione
      parameter (ione=1)
      character*2 xproc(3)
      common/cxproc/xproc
      real*8 betfac,delta,deltas,deltac
      common/betfac/betfac,delta
      common/pmerge/deltas,deltac
      include 'hvqcblks.h'
      real*8 ycm,tau
      common/x1x2/ycm,tau
      real*8 sf(4,3,5)
      integer ipdfscale
      common/cipdfscale/ipdfscale
      integer idec
      common/cidec/idec
      real*8 bsfsgn
      common/cbssgn/bsfsgn
      real*8 bsewgt
      common/cbswgt/bsewgt
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real*8 vv(4,3,5),vvs(4,3,5)
      common/cvv/vv
      real*8 x,y,cth1,th2,cth2,sx,rox,bx,btildex,rotildx,
     # x1,x2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,zg2,
     # xnorm,f,xint,www,xlgomx,x1t,x2t,xtmp,ytmp,
     # xlmude,xnormc,xcplus,xcminus,xintcm,xintffs,
     # xnormsv,xintcps,xintcms,xintcp,xnormb,bbb,x1soft,
     # x2soft,x1x2j,x1x2jqq,hvqborn,ppsv,fpp,ppcolp,ppcoll,
     # xphsp,xphspcp,xphspcm,zgmu6_mc,zg6,gfactsf,gfactcl,zhwfct,
     # dfact,gcpfact,zgmu2_nqq,gcmfact,gsffact,dfact1,
     # xsum,dummy,xnormmc,xphspmc,ro,beta,betamc,xnlfscheme
      real*8 xcs(3),xsv(3),xints(3),xborn(3)
      real*8 xmcxsec(1:4,1:3),xmce0sq(1:4,1:3),xmcz(1:4,1:3)
      real*8 xqrksc(1:3,1:4,1:3),xqbrsc(1:3,1:4,1:3)
      integer i,itype,ileg,ie0sq,iret,iproc,iproclo,iprocma
      integer jproc
      common/cjproc/jproc
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      logical flxsec(1:4,1:3),flagmc,fx1x2
c
      x=xx
      y=xy
      cth1=xcth1
      th2=xth2
      cth2=cos(th2)
c
      sx=s*x
      ro=4*xm2/s
      beta=sqrt(1-ro)
      rox=4*xm2/sx
      bx=sqrt(1-rox)
      btildex=bx*betfac
      rotildx=1-btildex**2
c
      x1=sqrt(tau)*exp(ycm)
      x2=tau/x1
c
      xlgomx=log(1-x)
      do jproc=1,3
        do i=1,4
          do itype=1,nl
            vv(i,jproc,itype)=0.d0
            vvs(i,jproc,itype)=0.d0
          enddo
        enddo
      enddo
      xnorm=xjac / (s * 64*pi2 * 16*pi2)
      xphsp=bx/(1-x)*( 1/(1-y) + 1/(1+y) ) 
      xphspcp=bx/((1-x)*(1-y))
      xphspcm=bx/((1-x)*(1+y))
      xnormmc=xjac / (64*pi2 * 16*pi2)
      xphspmc=1/(1-x)*( 1/(1-y) + 1/(1+y) ) 
c
c MC subt term: pure MC
c
      ipdfscale=2
      do jproc=loproc,maproc
        call xmcsuqq(x1,x2,xm2,s,x,y,cth1,cth2,
     #    xmcxsec,xmce0sq,xmcz,xqrksc,xqbrsc,flxsec,flagmc,
     #    gfactsf,gfactcl)
        if(flagmc)then
          do ileg=1,4
            do ie0sq=1,3
              if(flxsec(ileg,ie0sq))then
                if(ileg.eq.1)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)/zhwfct
                  x2t=x2soft(x1,x2,x,y)
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1.and.
     #                  (x*tau).lt.1
                elseif(ileg.eq.2)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)
                  x2t=x2soft(x1,x2,x,y)/zhwfct
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1.and.
     #                  (x*tau).lt.1
                else
                  zhwfct=1.d0
                  x1t=x1soft(x1,x2,x,y)
                  x2t=x2soft(x1,x2,x,y)
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  (x**2*tau).lt.1
                endif
                if(fx1x2)then
                  zg6=zgmu6_mc(ileg,xm2,s,x,y,cth1,cth2,
     #                  xqrksc(1,ileg,ie0sq),xqrksc(2,ileg,ie0sq),
     #                  xqrksc(3,ileg,ie0sq),xqbrsc(1,ileg,ie0sq),
     #                  xqbrsc(2,ileg,ie0sq),xqbrsc(3,ileg,ie0sq))
                  x1x2j=x1x2jqq(x1,x2,x,y,ione)/zhwfct
                  www=zg6*xnormmc*xphspmc*x1x2j*betamc*
     #                xmcxsec(ileg,ie0sq)
                  call strfqq(x1t,x2t,sf)
                  do i=1,4
                    do itype=1,nl
                      vv(i,jproc,itype)=vv(i,jproc,itype)+
     #                  sf(i,jproc,itype)*www
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
        endif
      enddo
c
c Counter-event (x,y)=(x,1) and MC subt term, collinear ME part
c
      if(y.gt.1-delta.or.gfactcl.lt.1.d0) then
         dfact=0.d0
         if(y.gt.1-delta)dfact=1.d0
         gcpfact=0.d0
         if(y.gt.0.d0.and.x1.lt.1.and.x2.lt.1)gcpfact=1-gfactcl
         ytmp=1.d0
         x1t=x1soft(x1,x2,x,y)/x
         x2t=x2soft(x1,x2,x,y)
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)/x
            call invqq(xm2,s,x,ytmp,cth1,cth2,'p1',
     #                 tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xintcp=0.d0
              xcplus=0.d0
              f=fpp(s,x,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
              xintcp=zg2**3*xnorm*xphspcp*x1x2j*f*(gcpfact-dfact)
              if(dfact.eq.1.d0)then
c Adding the collinear contribution
                xlmude=log(s/xmuf2h1)+log(delta/2)
                xnormc=zg2**3*x1x2j*xjac/(delta*16*pi2)
                xcplus=xnormc*bx/(1-x)*(
     #                        ppcolp(ytmp,s,q2q,x,xm2,xlmude)
     #                +xlgomx*ppcoll(ytmp,s,q2q,x,xm2) )
              endif
              www=xintcp + xcplus
              call strfqq(x1t,x2t,sf)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Counter-event (x,y)=(x,-1) and MC subt term, collinear ME part
c
      if(y.lt.-1+delta.or.gfactcl.lt.1.d0) then
         dfact=0.d0
         if(y.lt.-1+delta)dfact=1.d0
         gcmfact=0.d0
         if(y.lt.0.d0.and.x1.lt.1.and.x2.lt.1)gcmfact=1-gfactcl
         ytmp=-1.d0
         x1t=x1soft(x1,x2,x,y)
         x2t=x2soft(x1,x2,x,y)/x
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)/x
            call invqq (xm2,s,x,ytmp,cth1,cth2,'p1',
     #            tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xintcm=0.d0
              xcminus=0.d0
              f=fpp(s,x,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
              xintcm=zg2**3*xnorm*xphspcm*x1x2j*f*(gcmfact-dfact)
              if(dfact.eq.1.d0)then
c Adding the collinear contribution
                xlmude=log(s/xmuf2h2)+log(delta/2)
                xnormc=zg2**3*x1x2j*xjac/(delta*16*pi2)
                xcminus=xnormc*bx/(1-x)*(
     #                        ppcolp(ytmp,s,q1q,x,xm2,xlmude)
     #                +xlgomx*ppcoll(ytmp,s,q1q,x,xm2) )
              endif
              www=xintcm + xcminus
              call strfqq(x1t,x2t,sf)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Soft counter-events, and MC subt term, soft and soft-collinear ME parts
c
      if(x.gt.rotildx.or.gfactsf.lt.1.d0) then
         dfact=0.d0
         if(x.gt.rotildx)dfact=1.d0
         gsffact=0.d0
         if(x1.lt.1.and.x2.lt.1)gsffact=1-gfactsf
         xtmp=1.d0
         x1t=x1soft(x1,x2,x,y)
         x2t=x2soft(x1,x2,x,y)
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)
            call invqq(xm2,sx,xtmp,y,cth1,cth2,'p1',
     #           tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xints(jproc)=0.d0
              xborn(jproc)=0.d0
              xcs(jproc)=0.d0
              xsv(jproc)=0.d0
              f=fpp(sx,xtmp,y,xm2,q1q,q2q,w1h,w2h,cth2)
              xintffs=zg2**3*(xnorm/x)*xphsp*x1x2j*f*(gsffact-dfact)
              xints(jproc)=xints(jproc)+xintffs
              if(dfact.eq.1.d0)then
c Adding the soft-virtual contribution
                xnormsv=zg2**3*x1x2j*xjac/ 
     #                  (32*pi2*16*pi2*(1-rotildx))
                xsv(jproc)=xnormsv*bx*
     #                     ppsv(sx,q1q,xm2,xmur2,xmuf2h1,xmuf2h2)
c Adding the Born term
                xnormb=zg2**2*x1x2j*xjac/(16*pi*2*pi*(1-rotildx))
                bbb=hvqborn(sx,q1q,xm2,jproc)
                xborn(jproc)=xborn(jproc)+xnormb*bx*bbb*( 1.d0+
     #            xnlfscheme(xm2,xmur2,xmuf2h1,xmuf2h2,zg2,jproc) )
              endif
            enddo
            if(y.gt.1-delta.or.gfactcl.lt.1.d0) then
              ipdfscale=1
              dfact1=0.d0
              if(y.gt.1-delta.and.dfact.eq.1.d0)dfact1=1.d0
              gcpfact=0.d0
              if(y.gt.0.d0.and.x1.lt.1.and.x2.lt.1)gcpfact=1-gfactcl
              ytmp=1.d0
              call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,'p1',
     #          tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
              zg2=zgmu2_nqq()
              do jproc=loproc,maproc
                prc=xproc(jproc)
                f=fpp(sx,xtmp,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
                xintcps=-zg2**3*(xnorm/x)*xphspcp*x1x2j*f*
     #                  (gsffact*gcpfact-dfact1)
                xints(jproc)=xints(jproc)+xintcps
                if(dfact1.eq.1.d0)then
c Adding the collinear contribution
                  xlmude=log(sx/xmuf2h1)+log(delta/2)
                  xnormc=zg2**3*x1x2j*xjac/(delta*16*pi2)
                  xcs(jproc)=xcs(jproc) - xnormc*bx/(1-x)*(
     #                       ppcolp(ytmp,sx,q2q,xtmp,xm2,xlmude)
     #               +xlgomx*ppcoll(ytmp,sx,q2q,xtmp,xm2) )
                endif
              enddo
            endif
            if(y.lt.-1+delta.or.gfactcl.lt.1.d0) then
              ipdfscale=1
              dfact1=0.d0
              if(y.lt.-1+delta.and.dfact.eq.1.d0)dfact1=1.d0
              gcmfact=0.d0
              if(y.lt.0.d0.and.x1.lt.1.and.x2.lt.1)gcmfact=1-gfactcl
              ytmp=-1.d0
              call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,'p1',
     #          tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
              zg2=zgmu2_nqq()
              do jproc=loproc,maproc
                prc=xproc(jproc)
                f=fpp(sx,xtmp,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
                xintcms=-zg2**3*(xnorm/x)*xphspcm*x1x2j*f*
     #                  (gsffact*gcmfact-dfact1)
                xints(jproc)=xints(jproc) + xintcms
                if(dfact1.eq.1.d0)then
c Adding the collinear contribution
                  xlmude=log(sx/xmuf2h2)+log(delta/2)
                  xnormc=zg2**3*x1x2j*xjac/(delta*16*pi2)
                  xcs(jproc)=xcs(jproc) - xnormc*bx/(1-x)*(
     #                       ppcolp(ytmp,sx,q1q,xtmp,xm2,xlmude)
     #               +xlgomx*ppcoll(ytmp,sx,q1q,xtmp,xm2) )
                endif
              enddo
            endif
c
            call strfqq(x1t,x2t,sf)
            do jproc=loproc,maproc
              www=xints(jproc)+xsv(jproc)+xborn(jproc)+xcs(jproc)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Compute Born ME times luminosity to get flavour assignment; the
c normalization is irrelevant
c
      xtmp=1.d0
      x1t=x1soft(x1,x2,x,y)
      x2t=x2soft(x1,x2,x,y)
      if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
        ipdfscale=1
        call invqq(xm2,sx,xtmp,y,cth1,cth2,'p1',
     #      tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        zg2=zgmu2_nqq()
        call strfqq(x1t,x2t,sf)
        if(loproc.eq.3.and.maproc.eq.3)then
          iproclo=1
          iprocma=2
        else
          iproclo=loproc
          iprocma=maproc
        endif
        do iproc=iproclo,iprocma
          prc=xproc(iproc)
          www=hvqborn(sx,q1q,xm2,iproc)
          do i=1,4
            do itype=1,nl
              vvs(i,iproc,itype)=sf(i,iproc,itype)*www
            enddo
          enddo
        enddo
      endif
c
      call checkqq(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp=1.d0
        ytmp=1.d0
        call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        x1t=x1soft(x1,x2,x,y)
        x2t=x2soft(x1,x2,x,y)
        ycm=0.5d0*log(x1t/x2t)
        tau=x*tau
        if(idec.eq.0)then
          ps=sx
          px=xtmp
          py=ytmp
          pcth1=cth1
          pcth2=cth2
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
      tqq5bs=abs(xint)
c
      return
      end


      function tqq5bz(s,xx,xy,xcth1,xth2,xjac)
c Implements zeta subtraction
      implicit none
      character*2 str
      parameter (str='p1')
      real*8 tqq5bz,s,xx,xy,xcth1,xth2,xjac
      real*8 pi,pi2,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      integer ione
      parameter (ione=1)
      character*2 xproc(3)
      common/cxproc/xproc
      real*8 betfac,delta,deltas,deltac,etacut
      common/betfac/betfac,delta
      common/pmerge/deltas,deltac
      common/cetacut/etacut
      include 'hvqcblks.h'
      real*8 ycm,tau
      common/x1x2/ycm,tau
      real*8 sf(4,3,5)
      integer ipdfscale
      common/cipdfscale/ipdfscale
      integer idec
      common/cidec/idec
      real*8 bsfsgn
      common/cbssgn/bsfsgn
      real*8 bsewgt
      common/cbswgt/bsewgt
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real*8 vv(4,3,5),vvs(4,3,5)
      common/cvv/vv
      real*8 x,y,cth1,th2,cth2,sx,rox,bx,btildex,rotildx,
     # x1,x2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,zg2,
     # xnorm,f,xint,www,xlgomx,x1t,x2t,xtmp,ytmp,
     # xlmude,xnormc,xcplus,xcminus,xintcm,xintffs,
     # xnormsv,xintcps,xintcms,xintcp,xnormb,bbb,x1soft,
     # x2soft,x1x2j,x1x2jqq,hvqborn,ppsv,fpp,ppcolp,ppcoll,
     # xphsp,xphspcp,xphspcm,zgmu6_mc,zg6,gfactsf,gfactcl,zhwfct,
     # dfact,gcpfact,zgmu2_nqq,gcmfact,gsffact,dfact1,
     # xsum,dummy,xktrel,bdelta,svn,delppsv,bsub,xnormmc,
     # xphspmc,ro,beta,betamc,xnlfscheme
      real*8 xcs(3),xsv(3),xints(3),xborn(3)
      real*8 xmcxsec(1:4,1:3),xmce0sq(1:4,1:3),xmcz(1:4,1:3)
      real*8 xqrksc(1:3,1:4,1:3),xqbrsc(1:3,1:4,1:3)
      integer i,itype,ileg,ie0sq,iret,iproc,iproclo,iprocma
      integer jproc
      common/cjproc/jproc
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      logical flxsec(1:4,1:3),flagmc,fx1x2
c
      x=xx
      y=xy
      cth1=xcth1
      th2=xth2
      cth2=cos(th2)
c
      sx=s*x
      ro=4*xm2/s
      beta=sqrt(1-ro)
      rox=4*xm2/sx
      bx=sqrt(1-rox)
      btildex=bx*betfac
      rotildx=1-btildex**2
      xktrel = (1-x)**2*(1-y**2)
c
      x1=sqrt(tau)*exp(ycm)
      x2=tau/x1
c
      xlgomx=log(1-x)
      do jproc=1,3
        do i=1,4
          do itype=1,nl
            vv(i,jproc,itype)=0.d0
            vvs(i,jproc,itype)=0.d0
          enddo
        enddo
      enddo
      xnorm=xjac / (s * 64*pi2 * 16*pi2)
      xphsp=bx/(1-x)*( 1/(1-y) + 1/(1+y) ) 
      xphspcp=bx/((1-x)*(1-y))
      xphspcm=bx/((1-x)*(1+y))
      xnormmc=xjac / (64*pi2 * 16*pi2)
      xphspmc=1/(1-x)*( 1/(1-y) + 1/(1+y) ) 
c
c MC subt term: pure MC
c
      ipdfscale=2
      do jproc=loproc,maproc
        call xmcsuqq(x1,x2,xm2,s,x,y,cth1,cth2,
     #    xmcxsec,xmce0sq,xmcz,xqrksc,xqbrsc,flxsec,flagmc,
     #    gfactsf,gfactcl)
        if(flagmc)then
          do ileg=1,4
            do ie0sq=1,3
              if(flxsec(ileg,ie0sq))then
                if(ileg.eq.1)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)/zhwfct
                  x2t=x2soft(x1,x2,x,y)
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1.and.
     #                  (x*tau).lt.1
                elseif(ileg.eq.2)then
                  zhwfct=xmcz(ileg,ie0sq)
                  x1t=x1soft(x1,x2,x,y)
                  x2t=x2soft(x1,x2,x,y)/zhwfct
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  x1.lt.1.and.x2.lt.1.and.
     #                  (x*tau).lt.1
                else
                  zhwfct=1.d0
                  x1t=x1soft(x1,x2,x,y)
                  x2t=x2soft(x1,x2,x,y)
                  betamc=bx
                  fx1x2=x1t.lt.1.and.x2t.lt.1.and.
     #                  (x**2*tau).lt.1
                endif
                if(fx1x2)then
                  zg6=zgmu6_mc(ileg,xm2,s,x,y,cth1,cth2,
     #                  xqrksc(1,ileg,ie0sq),xqrksc(2,ileg,ie0sq),
     #                  xqrksc(3,ileg,ie0sq),xqbrsc(1,ileg,ie0sq),
     #                  xqbrsc(2,ileg,ie0sq),xqbrsc(3,ileg,ie0sq))
                  x1x2j=x1x2jqq(x1,x2,x,y,ione)/zhwfct
                  www=zg6*xnormmc*xphspmc*x1x2j*betamc*
     #                xmcxsec(ileg,ie0sq)
                  call strfqq(x1t,x2t,sf)
                  do i=1,4
                    do itype=1,nl
                      vv(i,jproc,itype)=vv(i,jproc,itype)+
     #                  sf(i,jproc,itype)*www
                    enddo
                  enddo
                endif
              endif
            enddo
          enddo
        endif
      enddo
c
c Counter-event (x,y)=(x,1) and MC subt term, collinear ME part
c
      if( (xktrel.lt.etacut.and.y.gt.1-delta) .or.
     #    gfactcl.lt.1.d0) then
         dfact=0.d0
         if(xktrel.lt.etacut.and.y.gt.1-delta)dfact=1.d0
         gcpfact=0.d0
         if(y.gt.0.d0.and.x1.lt.1.and.x2.lt.1)gcpfact=1-gfactcl
         ytmp=1.d0
         x1t=x1soft(x1,x2,x,y)/x
         x2t=x2soft(x1,x2,x,y)
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)/x
            call invqq(xm2,s,x,ytmp,cth1,cth2,'p1',
     #                 tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xintcp=0.d0
              xcplus=0.d0
              f=fpp(s,x,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
              xintcp=zg2**3*xnorm*xphspcp*x1x2j*f*(gcpfact-dfact)
              if(dfact.eq.1.d0)then
c Adding the collinear contribution
                xlmude=log(s/xmuf2h1)+log(delta/2)+
     #                 log( (1-bdelta(x))/delta )
                xnormc=zg2**3*x1x2j*xjac/(16*pi2*(1-bdelta(x)))
                xcplus=xnormc*bx/(1-x)*(
     #                        ppcolp(ytmp,s,q2q,x,xm2,xlmude)
     #                +xlgomx*ppcoll(ytmp,s,q2q,x,xm2) )
              endif
              www=xintcp + xcplus
              call strfqq(x1t,x2t,sf)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Counter-event (x,y)=(x,-1) and MC subt term, collinear ME part
c
      if( (xktrel.lt.etacut.and.y.lt.-1+delta) .or.
     #    gfactcl.lt.1.d0) then
         dfact=0.d0
         if(xktrel.lt.etacut.and.y.lt.-1+delta)dfact=1.d0
         gcmfact=0.d0
         if(y.lt.0.d0.and.x1.lt.1.and.x2.lt.1)gcmfact=1-gfactcl
         ytmp=-1.d0
         x1t=x1soft(x1,x2,x,y)
         x2t=x2soft(x1,x2,x,y)/x
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)/x
            call invqq (xm2,s,x,ytmp,cth1,cth2,'p1',
     #            tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xintcm=0.d0
              xcminus=0.d0
              f=fpp(s,x,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
              xintcm=zg2**3*xnorm*xphspcm*x1x2j*f*(gcmfact-dfact)
              if(dfact.eq.1.d0)then
c Adding the collinear contribution
                xlmude=log(s/xmuf2h2)+log(delta/2)+
     #                 log( (1-bdelta(x))/delta )
                xnormc=zg2**3*x1x2j*xjac/(16*pi2*(1-bdelta(x)))
                xcminus=xnormc*bx/(1-x)*(
     #                        ppcolp(ytmp,s,q1q,x,xm2,xlmude)
     #                +xlgomx*ppcoll(ytmp,s,q1q,x,xm2) )
              endif
              www=xintcm + xcminus
              call strfqq(x1t,x2t,sf)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Soft counter-events, and MC subt term, soft and soft-collinear ME parts
c
      if( (xktrel.lt.etacut.and.x.gt.rotildx) .or.
     #    gfactsf.lt.1.d0) then
         dfact=0.d0
         if(xktrel.lt.etacut.and.x.gt.rotildx)dfact=1.d0
         gsffact=0.d0
         if(x1.lt.1.and.x2.lt.1)gsffact=1-gfactsf
         xtmp=1.d0
         x1t=x1soft(x1,x2,x,y)
         x2t=x2soft(x1,x2,x,y)
         if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
            ipdfscale=1
            x1x2j=x1x2jqq(x1,x2,x,y,ione)
            call invqq(xm2,sx,xtmp,y,cth1,cth2,'p1',
     #           tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            zg2=zgmu2_nqq()
            do jproc=loproc,maproc
              prc=xproc(jproc)
              xints(jproc)=0.d0
              xborn(jproc)=0.d0
              xcs(jproc)=0.d0
              xsv(jproc)=0.d0
              f=fpp(sx,xtmp,y,xm2,q1q,q2q,w1h,w2h,cth2)
              xintffs=zg2**3*xnorm*xphsp*(x1x2j/x)*f*(gsffact-dfact)
              xints(jproc)=xints(jproc)+xintffs
              if(dfact.eq.1.d0)then
c Adding the soft-virtual contribution
                xnormsv=zg2**3*x1x2j*xjac/ 
     #                  (32*pi2*16*pi2*(1-rotildx+svn(rotildx)))
                delppsv=-bsub(sx,xm2,cth1,etacut,jproc)/(4.d0*sx)
                xsv(jproc)=xnormsv * bx * ( delppsv +
     #                ppsv(sx,q1q,xm2,xmur2,xmuf2h1,xmuf2h2) )
c Adding the Born term
                xnormb=zg2**2*x1x2j*xjac/
     #                 (16*pi*2*pi*(1-rotildx+svn(rotildx)))
                bbb=hvqborn(sx,q1q,xm2,jproc)
                xborn(jproc)=xborn(jproc)+xnormb*bx*bbb*( 1.d0+
     #            xnlfscheme(xm2,xmur2,xmuf2h1,xmuf2h2,zg2,jproc) )
              endif
            enddo
            if( (xktrel.lt.etacut.and.y.gt.1-delta) .or.
     #          gfactcl.lt.1.d0) then
              ipdfscale=1
              dfact1=0.d0
              if(xktrel.lt.etacut.and.y.gt.1-delta.and.
     #           dfact.eq.1.d0)dfact1=1.d0
              gcpfact=0.d0
              if(y.gt.0.d0.and.x1.lt.1.and.x2.lt.1)gcpfact=1-gfactcl
              ytmp=1.d0
              call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,'p1',
     #          tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
              zg2=zgmu2_nqq()
              do jproc=loproc,maproc
                prc=xproc(jproc)
                f=fpp(sx,xtmp,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
                xintcps=-zg2**3*xnorm*xphspcp*(x1x2j/x)*f*
     #                  (gsffact*gcpfact-dfact1)
                xints(jproc)=xints(jproc)+xintcps
                if(dfact1.eq.1.d0)then
c Adding the collinear contribution
                  xlmude=log(sx/xmuf2h1)+log(delta/2)+
     #                   log( (1-bdelta(x))/delta )
                  xnormc=zg2**3*x1x2j*xjac/(16*pi2*(1-bdelta(x)))
                  xcs(jproc)=xcs(jproc) - xnormc*bx/(1-x)*(
     #                       ppcolp(ytmp,sx,q2q,xtmp,xm2,xlmude)
     #               +xlgomx*ppcoll(ytmp,sx,q2q,xtmp,xm2) )
                endif
              enddo
            endif
            if( (xktrel.lt.etacut.and.y.lt.-1+delta) .or.
     #          gfactcl.lt.1.d0) then
              ipdfscale=1
              dfact1=0.d0
              if(xktrel.lt.etacut.and.y.lt.-1+delta.and.
     #           dfact.eq.1.d0)dfact1=1.d0
              gcmfact=0.d0
              if(y.lt.0.d0.and.x1.lt.1.and.x2.lt.1)gcmfact=1-gfactcl
              ytmp=-1.d0
              call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,'p1',
     #          tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
              zg2=zgmu2_nqq()
              do jproc=loproc,maproc
                prc=xproc(jproc)
                f=fpp(sx,xtmp,ytmp,xm2,q1q,q2q,w1h,w2h,cth2)
                xintcms=-zg2**3*xnorm*xphspcm*(x1x2j/x)*f*
     #                  (gsffact*gcmfact-dfact1)
                xints(jproc)=xints(jproc) + xintcms
                if(dfact1.eq.1.d0)then
c Adding the collinear contribution
                  xlmude=log(sx/xmuf2h2)+log(delta/2)+
     #                   log( (1-bdelta(x))/delta )
                  xnormc=zg2**3*x1x2j*xjac/(16*pi2*(1-bdelta(x)))
                  xcs(jproc)=xcs(jproc) - xnormc*bx/(1-x)*(
     #                       ppcolp(ytmp,sx,q1q,xtmp,xm2,xlmude)
     #               +xlgomx*ppcoll(ytmp,sx,q1q,xtmp,xm2) )
                endif
              enddo
            endif
c
            call strfqq(x1t,x2t,sf)
            do jproc=loproc,maproc
              www=xints(jproc)+xsv(jproc)+xborn(jproc)+xcs(jproc)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*www
                enddo
              enddo
            enddo
         endif
      endif
c
c Compute Born ME times luminosity to get flavour assignment; the
c normalization is irrelevant
c
      xtmp=1.d0
      x1t=x1soft(x1,x2,x,y)
      x2t=x2soft(x1,x2,x,y)
      if(x1t.lt.1.and.x2t.lt.1.and.(x*tau).lt.1)then
        ipdfscale=1
        call invqq(xm2,sx,xtmp,y,cth1,cth2,'p1',
     #      tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        zg2=zgmu2_nqq()
        call strfqq(x1t,x2t,sf)
        if(loproc.eq.3.and.maproc.eq.3)then
          iproclo=1
          iprocma=2
        else
          iproclo=loproc
          iprocma=maproc
        endif
        do iproc=iproclo,iprocma
          prc=xproc(iproc)
          www=hvqborn(sx,q1q,xm2,iproc)
          do i=1,4
            do itype=1,nl
              vvs(i,iproc,itype)=sf(i,iproc,itype)*www
            enddo
          enddo
        enddo
      endif
c
      call checkqq(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp=1.d0
        ytmp=1.d0
        call invqq(xm2,sx,xtmp,ytmp,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        x1t=x1soft(x1,x2,x,y)
        x2t=x2soft(x1,x2,x,y)
        ycm=0.5d0*log(x1t/x2t)
        tau=x*tau
        if(idec.eq.0)then
          ps=sx
          px=xtmp
          py=ytmp
          pcth1=cth1
          pcth2=cth2
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
      tqq5bz=abs(xint)
c
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
      subroutine sprfqq()
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
      integer idec
      common/cidec/idec
      integer iret
      real*8 ycm0
c
      call xoqq(iret)
      if(iret.eq.1)then
        if(idec.eq.0)call getspincoqq(jproc0)
        if(i0.eq.1)then
          call labmqq(ycm)
          ycm0=ycm
        elseif(i0.eq.2)then
          call conjug(ycm)
          ycm0=ycm
        elseif(i0.eq.3)then
          call reflqq(ycm)
          ycm0=-ycm
        elseif(i0.eq.4)then
          call refcon(ycm)
          ycm0=-ycm
        else
          write(*,*)'Fatal error in sprfqq'
          stop
        endif
        call getx1x2(tau,ycm0)
        call getmqq(tau,ycm0,i0)
        call store_evenqq(iunit,xone)
      endif
      return
      end


      subroutine labmqq(y)
c boost CM momenta to the lab system
c
      implicit none
      real * 8 y
      real * 8 yq10,yq20,yp0,pq10,pq20,pp0
      common/ycmvar/yq10,yq20,yp0
      common/perpen/pq10(2),pq20(2),pp0(2)
      include 'hvqcblks.h'
      integer j
      yq1 = yq10 + y
      yq2 = yq20 + y
      yp  = yp0  + y
      do j=1,2
         pq1(j) = pq10(j)
         pq2(j) = pq20(j)
         pp(j)  = pp0(j)
      enddo
      return
      entry conjug(y)
      yq1 = yq20 + y
      yq2 = yq10 + y
      yp  = yp0  + y
      do j=1,2
         pq1(j) = pq20(j)
         pq2(j) = pq10(j)
         pp(j)  = pp0(j)
      enddo
      return
      entry reflqq(y)
      yq1 = - yq10 - y
      yq2 = - yq20 - y
      yp  = - yp0  - y
      do j=1,2
         pq1(j) = - pq10(j)
         pq2(j) = - pq20(j)
         pp(j)  = - pp0(j)
      enddo
      return
      entry refcon(y)
      yq1 = - yq20 - y
      yq2 = - yq10 - y
      yp  = - yp0  - y
      do j=1,2
         pq1(j) = - pq20(j)
         pq2(j) = - pq10(j)
         pp(j)  = - pp0(j)
      enddo
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


      subroutine getmqq(tau,ycm,i0)
      implicit none
      integer i0
      real*8 tau,ycm
      include 'hvqcblks.h'
      real*8 pi
      parameter (pi=3.14159265358979312D0)
      integer i,j,k,imax,itype
      real*8 xsign,xtmp,theta,cth,sth,fk88random,sqsh,ycmnew
      real*8 x1,x2
      common/cx1x2/x1,x2
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_lb(11,4)
      common/cxmomlb/xmom_lb
      real*8 xmom_prime(11,4)
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
      if(i0.le.2)then
        xsign=1.d0
      elseif(i0.le.4)then
        xsign=-1.d0
      else
        write(*,*)'Fatal error in getmqq'
        stop
      endif
      imax=5
      if(idec.eq.0)imax=11
      do j=1,3
        xmom_cm(3,j)=xsign*xmom_cm(3,j)
        if(i0.eq.1.or.i0.eq.3)then
          do i=4,imax
            xmom_cm(i,j)=xsign*xmom_cm(i,j)
          enddo
        else
          xtmp=xsign*xmom_cm(5,j)
          xmom_cm(5,j)=xsign*xmom_cm(4,j)
          xmom_cm(4,j)=xtmp
          if(idec.eq.0)then
            do k=1,3
              xtmp=xsign*xmom_cm(k+8,j)
              xmom_cm(k+8,j)=xsign*xmom_cm(k+5,j)
              xmom_cm(k+5,j)=xtmp
            enddo
          endif
        endif
      enddo
      if(i0.eq.2.or.i0.eq.4)then
        xtmp=xmom_cm(5,4)
        xmom_cm(5,4)=xmom_cm(4,4)
        xmom_cm(4,4)=xtmp
        if(idec.eq.0)then
          do k=1,3
            xtmp=xmom_cm(k+8,4)
            xmom_cm(k+8,4)=xmom_cm(k+5,4)
            xmom_cm(k+5,4)=xtmp
          enddo
        endif
      endif
c perform a random rotation in the transverse plane
      theta=2*pi*fk88random(ifk88seed)
      cth=cos(theta)
      sth=sin(theta)
      call transrot(cth,sth,pq1(1),pq1(2))
      call transrot(cth,sth,pq2(1),pq2(2))
      call transrot(cth,sth,pp(1),pp(2))
      do i=3,imax
        call transrot(cth,sth,xmom_cm(i,1),xmom_cm(i,2))
      enddo
      if(ichkmom.eq.0)call checkmqq(xmom_cm,sh,0.d0,3,2)
c determine colour connections
      call getcolconn()
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
        call put_on_sheqq(ycm,ycmnew)
        do i=1,imax
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(ichkmom.eq.0)then
        itype=idec+1
        call checkmqq(xmom_lb,sh,-ycmnew,2,itype)
      endif
      call momnewformqq()
      return
      end


      subroutine momnewformqq()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(11,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:11)
      common/procmass/xmss
      integer i
c
      do i=1,11
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


      subroutine getcolconn()
c Determines colour connections. Derived from Bryan's subroutine UPFLOW
      implicit none
      include 'hvqcblks.h'
      real*8 xm,t1r,t2r,trn,crnd,fk88random,dotprod,s,tk,uk,q1q,q2q,
     #  s2,q1c,q2c,w1,w2,t(6)
      integer i
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_cross(5,4)
      common/cxmomcross/xmom_cross
      integer i1hpro
      common/ci1hpro/i1hpro
      integer iccode
      common/ciccode/iccode
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      if(xmom_cm(3,4).eq.0.d0)then
c 2-body kinematics
        if(i1hpro.eq.401)then
          iccode=1
        elseif(i1hpro.eq.403)then
          iccode=2
        elseif(i1hpro.eq.407)then
          crnd=fk88random(ifk88seed)
          t1r=
     # dotprod(xmom_cm(1,1),xmom_cm(1,2),xmom_cm(1,3),xmom_cm(1,4),
     #         xmom_cm(5,1),xmom_cm(5,2),xmom_cm(5,3),xmom_cm(5,4))
          t2r=
     # dotprod(xmom_cm(1,1),xmom_cm(1,2),xmom_cm(1,3),xmom_cm(1,4),
     #         xmom_cm(4,1),xmom_cm(4,2),xmom_cm(4,3),xmom_cm(4,4))
          iccode=3
          if( (t1r**2).lt.(crnd*(t1r**2+t2r**2)) )iccode=4
        else
          write(*,*)'Fatal error #1 in getcolconn: i1hpro=',i1hpro
          stop
        endif
      else
c 3-body kinematics
        if(i1hpro.lt.401.or.i1hpro.gt.407)then
          write(*,*)'Fatal error #2 in getcolconn: i1hpro=',i1hpro
          stop
        endif
        crnd=fk88random(ifk88seed)
        call xcrossing(i1hpro)
        s=   2*dotprod(xmom_cross(1,1),xmom_cross(1,2),
     #                 xmom_cross(1,3),xmom_cross(1,4),
     #                 xmom_cross(2,1),xmom_cross(2,2),
     #                 xmom_cross(2,3),xmom_cross(2,4))
        tk= -2*dotprod(xmom_cross(1,1),xmom_cross(1,2),
     #                 xmom_cross(1,3),xmom_cross(1,4),
     #                 xmom_cross(3,1),xmom_cross(3,2),
     #                 xmom_cross(3,3),xmom_cross(3,4))
        uk= -2*dotprod(xmom_cross(2,1),xmom_cross(2,2),
     #                 xmom_cross(2,3),xmom_cross(2,4),
     #                 xmom_cross(3,1),xmom_cross(3,2),
     #                 xmom_cross(3,3),xmom_cross(3,4))
        q1q=-2*dotprod(xmom_cross(1,1),xmom_cross(1,2),
     #                 xmom_cross(1,3),xmom_cross(1,4),
     #                 xmom_cross(4,1),xmom_cross(4,2),
     #                 xmom_cross(4,3),xmom_cross(4,4))
        q2q=-2*dotprod(xmom_cross(2,1),xmom_cross(2,2),
     #                 xmom_cross(2,3),xmom_cross(2,4),
     #                 xmom_cross(5,1),xmom_cross(5,2),
     #                 xmom_cross(5,3),xmom_cross(5,4))
        s2=s+tk+uk 
        q1c=-s-tk-q1q 
        q2c=-s-uk-q2q
        w1=-q1q+q2q-tk 
        w2=q1q-q2q-uk 
        xm=sqrt(xm2)
        if(i1hpro.lt.407)then
          call qqbplanar(s,tk,uk,q1q,q2q,s2,q1c,q2c,w1,w2,xm,t1r,t2r)
          if (t1r.gt.(t1r+t2r)*crnd) then
            iccode=2*(i1hpro-400)-1
          else
            iccode=2*(i1hpro-400)
          endif
        else
          call ggplanar(s,tk,uk,q1q,q2q,s2,q1c,q2c,w1,w2,xm,
     #                  t(1),t(2),t(3),t(4),t(5),t(6))
          do i=2,6
            t(i)=t(i)+t(i-1)
          enddo
          trn=t(6)*crnd
          do i=1,5
            if (trn.lt.t(i)) goto 10
          enddo
          i=6
 10       iccode=i+12
        endif
      endif
      return
      end


      subroutine xcrossing(i1hpro)
c Crosses parton 4-momenta, in order to determine colour connections for
c 2->3 processes. Derived from Bryan's subroutine UPFLOW
      integer i1hpro
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_cross(5,4)
      common/cxmomcross/xmom_cross
      real*8 xsign
      integer ihpro,i,j,ip,icros(1:3,1:7)
      data icros/
     #  1, 2, 3, 
     #  1,-3,-2, 
     #  2, 1, 3,
     # -3, 1,-2, 
     #  2,-3,-1,
     # -3, 2,-1,
     #  1, 2, 3/
c
      ihpro=i1hpro-400
c heavy quark 4-momenta are not affected
      do i=4,5
        do j=1,4
          xmom_cross(i,j)=xmom_cm(i,j)
        enddo
      enddo
c cross parton 4-momenta
      do i=1,3
        ip=icros(i,ihpro)
        xsign=1.d0
        if(ip.lt.0)xsign=-1.d0
        do j=1,4
          xmom_cross(i,j)=xsign*xmom_cm(iabs(ip),j)
        enddo
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


      subroutine put_on_sheqq(ycm,ycmnew)
      implicit none
      include 'hvqcblks.h'
      integer i2b,i,j,it,il,in,ib,ii
      real*8 xmss(1:11),xtmp(1:4),xk1tmp(1:4),ytmp1(1:4),ytmp2(1:4),
     #  xavg3(1:3),wvec(1:4),wvec2(1:4)
      real*8 ycm,ycmnew,pi,one,delta_thrs,shat,xkp2prime_norm2,
     #  xkp2prime_norm,xkprime_0,xsign,xnorm_3,delta,gamma,xmprime,
     #  xk1prime_norm,fakemass,xk1tmp_norm,xkprime_norm,xavgnorm,
     #  xnormsq,xbwnorm,xlepnorm,tmplmass,qw2,qw
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
      real*8 xlep1mass(2),xlep2mass(2)
      common/clepmass/xlep1mass,xlep2mass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is Q, 5 is Qbar. When the tops
c decay, 6=l+, 7=nu, 8=b are the decay products of the top, 9=l-, 10=nubar,
c 11=bbar are the decay products of the tbar. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+6+7+8+9+10+11)=0
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(11,4)
      common/cxmomprime/xmom_prime
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
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
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      xmss(4) = sqrt(xm2)
      xmss(5) = sqrt(xm2)
      if(idec.eq.0)then
        if(ideconsh.eq.0)then
          do i=6,11
            xmss(i) = 0.d0
          enddo
        elseif(ideconsh.eq.2)then
          xmss(6) = xlep1mass(1)
          xmss(7) = xlep2mass(1)
          xmss(8) = xmass(5)
          xmss(9) = xlep1mass(2)
          xmss(10) = xlep2mass(2)
          xmss(11) = xmass(5)
        else
          write(*,*)'Error in put_on_sheqq: unknown ideconsh',ideconsh
          stop
        endif
      elseif(idec.eq.1)then
        do i=6,11
          xmss(i) = -1.d10
        enddo
      else
        write(6,*) 'Error in put_on_sheqq: idec=',idec
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
        call getxmqq(shat,ycm,
     #               xmom_cm(1,3),xmss(1),
     #               xmom_cm(2,3),xmss(2),
     #               xmom_prime(1,3),xmom_prime(2,3))
      else
        write(*,*)'Fatal error in put_on_sheqq: unknown ionshell'
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
c delta is the would-be invqqiant mass of the pair, minus the sum
c of the masses of the heavy quarks
      delta=sqrt(xkprime_0**2-xkprime_norm**2)-xmss(4)-xmss(5)
      if(delta.lt.delta_thrs)then
c parton 3-momenta cannot be kept fixed: the total available energy
c is not sufficient; modify 3-momenta of the incoming partons
        gamma=sqrt( (xmss(4)+xmss(5)+delta_thrs)**2+xkprime_norm**2 )+
     #        xmom_prime(3,4)
        if(gamma.lt.(xmss(1)+xmss(2)))then
          write(6,*)'Fatal error #0 in put_on_sheqq'
          write(6,*)gamma,xmom_prime(3,4)
          stop
        endif
        xkp2prime_norm2=( gamma**2-2*(xmss(1)**2+xmss(2)**2)+
     #                    (xmss(1)**2-xmss(2)**2)**2/gamma**2 )/4.d0
        xkp2prime_norm=sqrt(xkp2prime_norm2)
        xmom_prime(1,3)=sign(1.d0,xmom_cm(1,3))*xkp2prime_norm
        xmom_prime(1,4)=sqrt(xkp2prime_norm2+xmss(1)**2)
        xmom_prime(2,3)=sign(1.d0,xmom_cm(2,3))*xkp2prime_norm
        xmom_prime(2,4)=sqrt(xkp2prime_norm2+xmss(2)**2)
        xkprime_0=xmom_prime(1,4)+xmom_prime(2,4)-xmom_prime(3,4)
        shat=(xmom_prime(1,4)+xmom_prime(2,4))**2 -
     #       (xmom_prime(1,3)+xmom_prime(2,3))**2
      endif
c now the parton 3-momenta have been defined in such a way
c that the momenta of the heavy quarks can be transformed.
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
          write(6,*)'Fatal error #1 in put_on_sheqq'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        xk1tmp_norm=xnorm_3(xk1tmp)
c xavg is the direction along which the Q1 and Q2 momenta are placed
c in the new QQ rest frame. It is arbitrarily defined by averaging 
c (hence the 1/2 in the definition) the directions of the original 
c Q1 and Q2 momenta. It may not have modulus 1, so normalize it
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
        xk1tmp(4)=sqrt(xk1prime_norm**2+xmss(i)**2)
        call xhwulob(xtmp,xmprime,
     #               xk1tmp,xmss(i),
     #               ytmp2,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Fatal error #2 in put_on_sheqq'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                    xmom_prime(i,3),xmom_prime(i,4))
      enddo
      if(idec.eq.0)then
        do it=4,5
          il=it+2+2*(it-4)
          in=it+3+2*(it-4)
          ib=it+4+2*(it-4)
          call fillvec(xmom_prime(it,1),xmom_prime(it,2),
     #                 xmom_prime(it,3),xmom_prime(it,4),xtmp)
c First deal with the Wb pair; define W momentum, and compute W mass
c (when iwidth=1, W is off shell)
          call vecsqq(xmom_cm(il,1),xmom_cm(il,2),
     #                xmom_cm(il,3),xmom_cm(il,4),one,
     #                xmom_cm(in,1),xmom_cm(in,2),
     #                xmom_cm(in,3),xmom_cm(in,4),one,wvec)
          qw2=xnormsq(wvec)
          qw=sqrt(qw2)
          if( ichkmom.eq.0 .and. iwidth.eq.0 .and.
     #        abs(qw/xmw-1.d0).gt.1.d-4 )then
            write(6,*)'Error #3 in put_on_sheqq'
            write(6,*)qw,it,il,in
            stop
          endif
          if( ichkmom.eq.0 .and. iwidth.eq.1 .and.
     #        qw.gt.xmt )then
            write(6,*)'Error #4 in put_on_sheqq'
            write(6,*)qw,it,il,in
            stop
          endif
          xbwnorm=xm2-2*(xmss(ib)**2+qw2)+(xmss(ib)**2-qw2)**2/xm2
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
     #                 xmom_cm(ib,3),xmom_cm(ib,4),ytmp1)
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
          xk1tmp(4)=xmt/2.d0*(1+xsign*(qw2-xmss(ib)**2)/xm2)
          call xhwulob(xtmp,xmt,xk1tmp,qw,wvec2,fakemass)
          xsign=-1.d0
          do j=1,3
            xk1tmp(j)=xsign*xbwnorm*xavg3(j)
          enddo
          xk1tmp(4)=xmt/2.d0*(1+xsign*(qw2-xmss(ib)**2)/xm2)
          call xhwulob(xtmp,xmt,xk1tmp,xmss(ib),ytmp2,fakemass)
          call getvec(ytmp2,xmom_prime(ib,1),xmom_prime(ib,2),
     #                      xmom_prime(ib,3),xmom_prime(ib,4))
c Next deal with the lepton pair; W has momentum wvec2
          xlepnorm=qw2-2*(xmss(il)**2+xmss(in)**2)+
     #             (xmss(il)**2-xmss(in)**2)**2/qw2
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
     #                   xmom_cm(ii,3),xmom_cm(ii,4),ytmp1)
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
     #        (1+xsign*(xmss(il)**2-xmss(in)**2)/qw2)
            call xhwulob(wvec2,qw,xk1tmp,tmplmass,ytmp2,fakemass)
            call getvec(ytmp2,xmom_prime(ii,1),xmom_prime(ii,2),
     #                        xmom_prime(ii,3),xmom_prime(ii,4))
          enddo
        enddo
      else
        do i=6,11
          do j=1,4
            xmom_prime(i,j)=0.d0
          enddo
        enddo
      endif
      if(ichkmom.eq.0)then
        if(idec.eq.0)then
          call checktdec2(xmom_prime,4,6,7,8)
          call checktdec2(xmom_prime,5,9,10,11)
          call checkmqq(xmom_prime,shat,0.d0,4,1)
        else
          call checkmqq(xmom_prime,shat,0.d0,4,2)
        endif
        if(xmass(1).eq.0.and.xmass(2).eq.0.and.xmass(3).eq.0.and.
     #     xmass(4).eq.0.and.xmass(5).eq.0.and.xmass(21).eq.0.and.
     #     xlep1mass(1).eq.0.and.xlep2mass(1).eq.0.and.
     #     xlep1mass(2).eq.0.and.xlep2mass(2).eq.0)then
          call checkonqq(1)
        else
          call checkonqq(2)
        endif
      endif
      return
      end


      subroutine getxmqq(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
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
        write(*,*)'Error # 0 in getxmqq'
        stop
      endif
      p13=xkp2prime_norm
      p23=-xkp2prime_norm
      e1=sqrt(p13**2+xm1**2)
      e2=sqrt(p23**2+xm2**2)
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
        write(*,*)'Error # 1 in getxmqq'
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
            write(*,*)'Mass # 1 too large in getxmqq'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmqq'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmqq'
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
      SUBROUTINE HWQQB4(PS,PI,PF)
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


C-----------------------------------------------------------------------
      SUBROUTINE HWULF4(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN LAB) INTO PF (IN REST FRAME OF PS)
C     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
C-----------------------------------------------------------------------
      DOUBLE PRECISION PF4,FN,PS(5),PI(4),PF(4)
      IF (PS(4).EQ.PS(5)) THEN
        PF(1)= PI(1)
        PF(2)= PI(2)
        PF(3)= PI(3)
        PF(4)= PI(4)
      ELSE
        PF4  = (PI(4)*PS(4)-PI(3)*PS(3)
     &         -PI(2)*PS(2)-PI(1)*PS(1))/PS(5)
        FN   = (PF4+PI(4)) / (PS(4)+PS(5))
        PF(1)= PI(1) - FN*PS(1)
        PF(2)= PI(2) - FN*PS(2)
        PF(3)= PI(3) - FN*PS(3)
        PF(4)= PF4
      END IF
      END


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


      function xnormsq(p)
c Computes p.p, assuming the energy is the fourth component
      implicit none
      real*8 xnormsq,p(1:4),dotprod
c
      xnormsq=dotprod(p(1),p(2),p(3),p(4),p(1),p(2),p(3),p(4))
      return
      end


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


      subroutine vecsqq(p1,p2,p3,p4,pfact,q1,q2,q3,q4,qfact,r)
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


      subroutine checkonqq(itype)
c Checks that put_on_sheqq is harmless if masses are zero (itype=1),
c or computes (itype=2) the average of the shifts due to the masses
      real*8 tiny
      parameter (tiny=1.d-4)
      integer itype
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_prime(11,4)
      common/cxmomprime/xmom_prime
      real*8 xmomshifts(4)
      common/cshifts/xmomshifts
      integer i,j,imax,iflag
      integer idec
      common/cidec/idec
c
      if(itype.ne.1.and.itype.ne.2)then
        write(*,*)'Unknown option in checkonqq'
        stop
      endif
      iflag=1
      imax=5
      if(idec.eq.0)imax=11
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
        write(*,*)'Error in checkonqq'
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


      subroutine xoqq(iret)
c This routine is called by sprfqq; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkqq), and no operation is performed
      implicit none
      integer iret,iretvv,iretvvs,iproc,iproclo,iprocma,i,itype,
     #  iwh,iflag,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o
      real*8 wwx(4,3,5),xsum,xsumabs,xsumvvs,xsumabsvvs,xstsign,
     #  xg,wh,rmax,fk88random
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      integer ifuntype
      common/cifuntype/ifuntype
      real*8 vv(4,3,5)
      common/cvv/vv
      real*8 vvs(4,3,5)
      common/cvvs/vvs
      integer iwrong,iwrong1
      common/ciwrong/iwrong,iwrong1
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer ivbhpro(4,3,5)
      common/civbhpro/ivbhpro
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
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
      call checkqq(xsum,xsumabs,iretvv)
      call checkqqs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xoqq:',iretvv,iretvvs
        stop
      endif
      if(iretvv.eq.1)then
        iret=iretvv
        if(ifuntype.eq.1)then
          iproclo=loproc
          iprocma=maproc
        elseif(ifuntype.eq.2)then
          if(loproc.eq.3.and.maproc.eq.3)then
            iproclo=1
            iprocma=2
          else
            iproclo=loproc
            iprocma=maproc
          endif
        else
          write(*,*)'Fatal error in xoqq: ifuntype=',ifuntype
          stop
        endif
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,nl
                wwx(i,iproc,itype)=vvs(i,iproc,itype)
              enddo
            enddo
          enddo
        else
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,nl
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
            do itype=1,nl
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
          write(*,*)'Fatal error in xoqq'
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0)
        i1=idp1(i0,jproc0,itype0)
        i2=idp2(i0,jproc0,itype0)
        i3=idp3(i0,jproc0,itype0)
        call parcrossiqq(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossiqq(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit none
      integer jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o,
     # iallzero,iahprotrans(402:406),ibhprotrans(402:406)
      parameter (iallzero=1)
      real*8 xg,fk88random
      integer ifuntype
      common/cifuntype/ifuntype
      integer ifk88seed
      common/cifk88seed/ifk88seed
      data iahprotrans/401,0,403,407,407/
      data ibhprotrans/407,0,407,403,401/
c
      if( (ifuntype.eq.1) .or. (ifuntype.eq.2.and.jproc0.ne.3) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif(ifuntype.eq.2.and.jproc0.eq.3)then
        if(ihpro.eq.401.or.ihpro.eq.403.or.ihpro.eq.407)then
          write(*,*)'Error #1 in parcrossiqq:',ihpro,i1,i2,i3
          stop
        endif
        xg=fk88random(ifk88seed)
        if(xg.lt.0.5d0)then
          i1hproo=iahprotrans(ihpro)
        else
          i1hproo=ibhprotrans(ihpro)
        endif
        if(i1hproo.ne.407)then
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
          ip1o=21
          ip2o=21
          ip3o=21
        endif
        if(i1.ne.21.and.i2.ne.21)then
          write(*,*)'Error #2 in parcrossiqq:',ihpro,i1,i2,i3
          stop
        endif
      else
        write(*,*)'parcrossiqq: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfqq(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checkqq(xsum,xsumabs,iret)
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
      real * 8 vv(4,3,5)
      common/cvv/vv
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=loproc,maproc
        do i=1,4
          do itype=1,nl
            if(vv(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vv(i,jproc,itype)
            xsumabs=xsumabs+abs(vv(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine checkqqs(xsum,xsumabs,iret)
c identical to checkqq, except for the fact that works on vvs instead of vv,
c and jproc is not fixed
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      real * 8 vvs(4,3,5)
      common/cvvs/vvs
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=1,2
        do i=1,4
          do itype=1,nl
            if(vvs(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vvs(i,jproc,itype)
            xsumabs=xsumabs+abs(vvs(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine getspincoqq(jproc0)
c Determines the lepton momenta, by performing an unweighting using
c the exact real and Born lepton matrix elements. This is done assuming
c idr=1; therefore, this routine must be called before labmqq...refcon
c and getmqq()
      implicit none
      integer jproc0
      real*8 pi,tolerance,bdredfact
      parameter (pi=3.14159265358979312D0)
      parameter (tolerance=1.d-2)
c Divide the bound by bdredfact to compensate for peculiar behaviour of
c ttbar cross section. May become a process-dependent correction if need be
      parameter (bdredfact=2.d0)
      integer ione
      parameter (ione=1)
      character*2 str,prcsave
      parameter (str='p1')
      include 'hvqcblks.h'
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      real*8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real*8 xm012,ga1,bw1delf,bw1fmmn
      common/cbw1/xm012,ga1,bw1delf,bw1fmmn
      real*8 xm022,ga2,bw2delf,bw2fmmn
      common/cbw2/xm022,ga2,bw2delf,bw2fmmn
      real*8 xm1low2,xm1upp2,xm2low2,xm2upp2
      common/bounmc/xm1low2,xm1upp2,xm2low2,xm2upp2
      real*8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      real*8 xmt,twidth
      common/ctparam/xmt,twidth
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
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
      real*8 xtmp,prob,spcdamp,rrnd,fk88random,e1,f1,g1,h1,e2,f2,
     # g2,h2,phitq1,cthtq1,phitq2,cthtq2,phiat1,cthat1,phiat2,cthat2,
     # o,p,q12,q22,xbwmass3,rat1,qphsp,q1,q2,tk,uk,q1q,q2q,q1c,q2c,w1,
     # w2,w1h,w2h,xdec,xmadevttb,unxdec,xttb,dmfactb1,dmfact1,
     # dmfactb2,dmfact2,phspfact1,phspfact2,xboundb,rat,xtq(4),xbq(4),
     # xel(4),xnu(4)
      integer iborn,iproj,icross,jjprc,icntuw,iqcntuw
c
      if(ichkmom.eq.0)call spccheqq(1)
      if(ifuntype.eq.2)then
        if(px.ne.1.d0.or.xmom_cm(3,4).ne.0.d0)then
          write(*,*)'Error #1 in getspincoqq'
          stop
        else
          iborn=0
          iproj=0
          if(jproc0.ne.3)then
            icross=0
          else
            icross=1
          endif
          xtmp=px
        endif
      endif
      if(ifuntype.eq.1)then
        prob=spcdamp(px,py)
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
c When iproj=0, the Born and real kinematics are used to perform unweighting
c for S and H events respectively. When iproj=1, the real kinematics is close 
c to the soft/collinear limits, and the Born is used to unweight. In the case 
c of the qg process, such Born is either gg (q||q <==> y->1) or qq 
c (g||q <==> y->-1). We choose gg (qq) if py>0 (py<0). This strategy,
c which serves to set here the local value of jproc (jjprc), must also 
c be adopted in the case of S events due to the qg contribution, and is
c associated with icross=1; icross=0 implies on the other hand that the 
c local and global values of jproc are the same. Any manipulations on 
c parton identities must also be carried out here
      if(icross.eq.0)then
        jjprc=jproc0
      elseif(icross.eq.1)then
        if(jproc0.ne.3)then
          jjprc=jproc0
        else
          if(py.ge.0.d0)then
            jjprc=1
          else
            jjprc=2
          endif
        endif
      else
        write(*,*)'Error #2 in getspincoqq'
        stop
      endif
      prcsave=prc
      prc=xproc(jjprc)
c
      neventsuw=neventsuw+1
      icntuw=0
 100  icntuw=icntuw+1
      e1=fk88random(ifk88seed)
      f1=fk88random(ifk88seed)
      g1=fk88random(ifk88seed)
      h1=fk88random(ifk88seed)
      e2=fk88random(ifk88seed)
      f2=fk88random(ifk88seed)
      g2=fk88random(ifk88seed)
      h2=fk88random(ifk88seed)
      phitq1=2*pi*e1
      cthtq1=-1.d0+2*f1
      phitq2=2*pi*g1
      cthtq2=-1.d0+2*h1
      phiat1=2*pi*e2
      cthat1=-1.d0+2*f2
      phiat2=2*pi*g2
      cthat2=-1.d0+2*h2
 300  continue
      if(iwidth.eq.1)then
        iqcntuw=0
 200    iqcntuw=iqcntuw+1
        o=fk88random(ifk88seed)
        p=fk88random(ifk88seed)
c First distribute q's according to the matrix element upper bound,
c which can be done exactly the upper bound being a Breit Wigner
        q12=xbwmass3(o,xm012,ga1,bw1delf,bw1fmmn)
        q22=xbwmass3(p,xm022,ga2,bw2delf,bw2fmmn)
c Then reject some of the values generated according to the phase-space
c q-dependent factor. A 1->1+(1->2) phase-space decomposition has been used.
c Much better here than after computing matrix elements. The following
c form works since qphsp is a function decreasing with q2
        rat1=( qphsp(q12,xm2)/qphsp(xm1low2,xm2) )*
     #       ( qphsp(q22,xm2)/qphsp(xm2low2,xm2) )
        rrnd=fk88random(ifk88seed)
        if(rat1.lt.rrnd)goto 200
        nqcntuws=nqcntuws+iqcntuw
        if(iqcntuw.gt.nqmaxuw)nqmaxuw=iqcntuw
        nqeventsuw=nqeventsuw+1
        q1=sqrt(q12)
        q2=sqrt(q22)
      else
        q12=xm012
        q1=sqrt(q12)
        q22=xm022
        q2=sqrt(q22)
      endif
c No complications here due to off-shell tops; can use the same kinematics 
c for decayed and undecayed matrix elements
      call invqq(xm2,ps,xtmp,py,pcth1,pcth2,str,
     #           tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      call gentopdmom(xmt,q1,cthtq1,phitq1,cthtq2,phitq2,
     #                xtq,xbq,xel,xnu,1)
      if(ichkmom.eq.0)call checktdec1(xmt,xtq,xbq,xel,xnu,1)
      call gentopdmom(xmt,q2,cthat1,phiat1,cthat2,phiat2,
     #                xtq,xbq,xel,xnu,2)
      if(ichkmom.eq.0)call checktdec1(xmt,xtq,xbq,xel,xnu,2)
      if(ichkmom.eq.0)call checkmqq(xmom_cm,ps,0.d0,10,1)
      xdec=xmadevttb(iborn,jjprc,ione,ps,tk,uk,xmom_cm)
      unxdec=xttb(iborn,jjprc,ione,xm2,ps,xtmp,py,
     #            tk,uk,q1q,q2q,w1h,w2h,pcth2)
      dmfactb1=256*xm2**2/16.d0
      dmfact1=1/(64.d0*sthw2**2)*
     #        1.d0/((q12-xm012)**2+xm012*ga1**2)
      dmfactb2=256*xm2**2/16.d0
      dmfact2=1/(64.d0*sthw2**2)*
     #        1.d0/((q22-xm022)**2+xm022*ga2**2)
      phspfact1=1.d0/(xm2*twidth**2)
      phspfact2=1.d0/(xm2*twidth**2)
      xboundb=dmfactb1*dmfact1*phspfact1*
     #        dmfactb2*dmfact2*phspfact2
      rat=xdec/((1+tolerance)*unxdec*xboundb)
      rat=rat*bdredfact
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
c configuration)
      if(iproj.eq.0)then
        if(px.ne.xtmp)then
          write(*,*)'Error #3 in getspincoqq'
          stop
        endif
      elseif(iproj.eq.1)then
        call invqq(xm2,ps,px,py,pcth1,pcth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        call gentopdmom(xmt,q1,cthtq1,phitq1,cthtq2,phitq2,
     #                  xtq,xbq,xel,xnu,1)
        call gentopdmom(xmt,q2,cthat1,phiat1,cthat2,phiat2,
     #                  xtq,xbq,xel,xnu,2)
        if(ichkmom.eq.0)call checkmqq(xmom_cm,ps,0.d0,20,1)
      else
        write(*,*)'Error #4 in getspincoqq'
        stop
      endif
      if(ichkmom.eq.0)call spccheqq(2)
      prc=prcsave
      return
      end


C      function spcdamp(x,y)
Cc This function is defined in such a way that
Cc    spcdamp=0  if  tt=0
Cc  0<spcdamp<1  if  0<tt<1
Cc    spcdamp=1  if  tt>1
Cc and tt is a measure in the (x,y) plane, such that tt=0 in the soft
Cc and collinear limits (x=1, or y=1, or y=-1), growing monotonically
Cc away from these limits. In terms of invqqiants, tt=4*tk*uk/((1-xlim)*s)**2,
Cc which can easily be generalized for any kind of emissions. 
Cc Since when spcdamp=1 the real matrix elements are used in the 
Cc unweighting, xlim has been defined in such a way that, if be_spcfun=1,
Cc spcdamp is equal to 1 in a region similar to the dead zone. This is
Cc by no means necessary, and the dependence upon xlim in tt can be
Cc eliminated altogether. Call this function with al_spcfun>=1, 0<be_spcfun<=1
C      implicit none
C      real * 8 spcdamp,x,y,xmin,tt,xlim
C      parameter (xmin=0.69519410160110384d0)
C      real * 8 al_spcfun,be_spcfun
C      common/cspcpar/al_spcfun,be_spcfun
Cc
C      xlim=1.d0-be_spcfun+xmin*be_spcfun
C      tt=(1-x)**2*(1-y**2)/(1-xlim)**2
C      if(tt.lt.0.d0)then
C        write(*,*)'Error in spcdamp',tt
C        stop
C      endif
C      if(tt.gt.1.d0)tt=1.d0
C      spcdamp=tt**(2*al_spcfun)/
C     #       (tt**(2*al_spcfun)+(1-tt)**(2*al_spcfun))
C      return
C      end


      subroutine spccheqq(iflag)
c Stores hard-process four-momenta (in xmom_save) at the beginning of 
c getspincoqq(), and checks as the last step of getspincoqq() that
c the manipulations carried out there did not change them, by comparing
c xmom_save and xmom_cm
      implicit none
      real*8 tiny,xmom_save(5,4)
      parameter (tiny=1.d-4)
      real*8 xmom_cm(11,4)
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
        write(*,*)'Wrong call to spccheqq'
        stop
      endif
      if(itmp.eq.1)then
        write(*,*)'The check in spccheqq failed'
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


      function xttb(iborn,jproc,idr,xmt2,s,x,y,
     #              tk,uk,q1q,q2q,w1h,w2h,cth2)
c Wrapper for the undecayed matrix elements of the original code.
c For Born matrix elements, q1q is t (consistently with the
c routine invqq). This function is called by getspincoqq, where only
c idr=1 is considered. Stop if other options are given in input
      implicit none
      real*8 xttb,xmt2,s,x,y,tk,uk,q1q,q2q,w1h,w2h,cth2
      integer iborn,jproc,idr
      real*8 s0,x0,y0,tk0,uk0,q1q0,q2q0,w1h0,w2h0,cth20,t0,
     # ggborn,qqborn,fpp
c
      if(idr.eq.1)then
        s0=s
        if(iborn.eq.0)then
          t0=q1q
        else
          x0=x
          y0=y
          tk0=tk
          uk0=uk
          q1q0=q1q
          q2q0=q2q
          w1h0=w1h
          w2h0=w2h
          cth20=cth2
        endif
      else
        write(*,*)'Error in xttb: use only direct events',idr
        stop
      endif
      if(iborn.eq.0.and.jproc.eq.1)then
        xttb=ggborn(s0,t0,xmt2)
      elseif(iborn.eq.0.and.jproc.eq.2)then
        xttb=qqborn(s0,t0,xmt2)
      else
        xttb=fpp(s0,x0,y0,xmt2,q1q0,q2q0,w1h0,w2h0,cth20)
      endif
      return
      end


      subroutine gentopdmom(xmt,xmw,cth1,phi1,cth2,phi2,
     #                      xtq,xbq,xel,xnu,iqrk)
c Generates the four-momenta of the decay products of the top (if iqrk=1)
c or of the tbar (if iqrk=2). These four-momenta are returned in the top/tbar 
c rest frame (xbq, xel, xnu; the trivial top/tbar momentum is returned as 
c well, xtq). The four-momenta are also boosted to the frame in which the 
c top/tbar has momentum xmom_cm(4,*)/xmom_cm(5,*), and the common block 
c xmomcm is filled according to the identifications
c   l+ --> xmom_cm(6,*), nu --> xmom_cm(7,*), b --> xmom_cm(8,*), 
c   l- --> xmom_cm(9,*), nub --> xmom_cm(10,*), bb --> xmom_cm(11,*), 
c consistently with the labelling conventions used in MC@NLO:
c   x(1)y(2) -> z(3)t(4)[->l+(6)nu(7)b(8)]tb(5)[->l-(9)nub(10)bb(11)]
c The inputs of the routine are cth1,phi1,cth2,phi2, which are cosines of
c polar angles and azimuthal angles, with
c   (cth1,phi1) --> direction of W in the top/tbar rest frame
c   (cth2,phi2) --> direction of l in the W rest frame
      implicit none
      real*8 xmt,xmw,cth1,phi1,cth2,phi2,xtq(4),xbq(4),xel(4),xnu(4)
      integer iqrk
      real*8 xmt2,xmw2,sth1,sth2,ew,eb,pwx,pwy,pwz,pbx,pby,pbz,eel,
     # enu,pex,pey,pez,pnx,pny,pnz,tmp(5),tmp1(4),tmp2(4)
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      integer itop,iel,inu,ib
c
      if(iqrk.eq.1)then
        itop=4
        iel=6
        inu=7
        ib=8
      elseif(iqrk.eq.2)then
        itop=5
        iel=9
        inu=10
        ib=11
      else
        write(6,*)'gentopdmom called improperly',iqrk
      endif
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
      call hwqqb4(tmp,tmp1,tmp2)
      xel(1)=tmp2(1)
      xel(2)=tmp2(2)
      xel(3)=tmp2(3)
      xel(4)=tmp2(4)
c Boost nu
      tmp1(1)=pnx
      tmp1(2)=pny
      tmp1(3)=pnz
      tmp1(4)=enu
      call hwqqb4(tmp,tmp1,tmp2)
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
      call filltopdec(tmp,xel,iel)
      call filltopdec(tmp,xnu,inu)
      call filltopdec(tmp,xbq,ib)
c
      return
      end


      subroutine filltopdec(tmp,tmp1,ipart)
c Utility routine for gentopdmom; performs the boost and fills xmom_cm 
c for top decay products
      implicit none
      real*8 tmp(5),tmp1(4),tmp2(4)
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      integer ipart
c
      call hwqqb4(tmp,tmp1,tmp2)
      xmom_cm(ipart,1)=tmp2(1)
      xmom_cm(ipart,2)=tmp2(2)
      xmom_cm(ipart,3)=tmp2(3)
      xmom_cm(ipart,4)=tmp2(4)
      return
      end


      function qphsp(q12,xmt2)
c Non-trivial factor of the t->bW phase space, in the t rest frame; q12 is
c the W mass squared
      implicit none
      real*8 qphsp,q12,xmt2,tmp
c
      tmp=0.d0
      if(q12.gt.0.d0.and.q12.lt.xmt2)tmp=(xmt2-q12)/(2.d0*xmt2)
      qphsp=tmp
      return
      end
c
c
c End of event-generation routines
c
c
c
c
c Begin of phase-space routines
c
c
      subroutine invqq(xm2,s,x,y,cth1,cth2,str,
     #     tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
c This routine upgrades the one used up to version 3.2, which was derived
c from MNR. Only str='p1' is a legal choice, which coincides with the 
c formulae of the MC@NLO paper. The dimensions of xmom_cm have been changed
c   xmom_cm(11,4) --> xmom_cm(5,4)
c to include the four-momenta of the leptons and b quarks emerging from
c top decays. The calls to the routine fillmom have been eliminated, to
c avoid potential (although very unlikely) numerical problems, in favour
c of the formulae adopted for VH production.
c This routine is meant to be called to generate the kinematics of 
c direct events, i.e. those for which
c   a(p1)+b(p2) --> Q(k1)+Qbar(k2)+c(k)
c The outputs are given in the common blocks perpen (pq1(2)==Q, 
c pq2(2)==Qbar, pp(2)==c transverse momenta), ycmvar (yq1==Q,
c yq2==Qbar ,yp==c rapidities in the parton c.m. frame), and stored
c as four-momenta in xmom_cm(ipart,icomp), with the conventions:
c   icomp=1 -> px, icomp=2 -> py, icomp=3 -> pz, icomp=4 -> E;
c   ipart=1 -> p1, ipart=2 -> p2, ipart=3 -> k, ipart=4 -> k1, ipart=5 -> k2
c The four-momenta of the decay products are computed elsewhere 
c (see the routine gentopdmom())
      implicit none
      real * 8 xm2,s,x,y,cth1,cth2,tk,uk,q1q,q2q,q1c,q2c,
     # w1,w2,w1h,w2h
      character * 2 str
      real * 8 pq1,pq2,pp,yq1,yq2,yp
      common/perpen/pq1(2),pq2(2),pp(2)
      common/ycmvar/yq1,yq2,yp
      real * 8 tiny,s2,drs2,p10,p20,k0,k10,k20,bx,sth1,cpsi,
     # spsi,cpsi2,spsi2,cpsi1,spsi1,xktsq,xkt1sq,xkt2sq,
     # xkt,xkt1,xkt2,tmp,sqs,e1lab,pl1lab,e2lab,pl2lab
      parameter (tiny=1.d-14)
      real*8 xmom_cm(11,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c
      tk  = -s/2*(1-x)*(1-y)
      uk  = -s/2*(1-x)*(1+y)
      s2  = tk+uk+s
      drs2 = 2*sqrt(s2)
      p10 = (s+tk)/drs2
      p20 = (s+uk)/drs2
      k0  = -(tk+uk)/drs2
      k10 = drs2/4
      k20 = drs2/4
      bx = sqrt(1-4*xm2/s2)
      sth1 = sqrt(1-cth1**2)
      if(str.eq.'p1') then
         cpsi2 = 1
         spsi2 = 0
         cpsi = 1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
         spsi = 4*(1-x)*sqrt(x*(1-y**2))/
     #          ((1+y+x*(1-y))*(1-y+x*(1+y)))
         cpsi1 = (1+y-x*(1-y))/(1+y+x*(1-y))
         spsi1 = sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
      else
         write(6,*) 'error in invqq: str=',str
         stop
      endif
      q1q = - 2*p10*k10*(1-bx*(cth2*sth1*spsi2+cth1*cpsi2))
      q2q = - 2*p20*k20*(1+bx*(cth2*sth1*spsi +cth1*cpsi ))
      q1c = - s - tk - q1q
      q2c = - s - uk - q2q
      w1  = - q1q + q2q - tk
      w2  = - q2q + q1q - uk
      w1h = 1-bx*(cth2*sth1*spsi1+cth1*cpsi1)
      w2h = 1+bx*(cth2*sth1*spsi1+cth1*cpsi1)
c
      if(abs(q1q).lt.tiny) then
        yq1  = 1.d8
      elseif(abs(q2c).lt.tiny) then
        yq1  = -1.d8
      else
        yq1 = .5d0*log( q2c/q1q )
      endif
      if(abs(q1c).lt.tiny) then
        yq2  = 1.d8
      elseif(abs(q2q).lt.tiny) then
        yq2  = -1.d8
      else
        yq2 = .5d0*log( q2q/q1c )
      endif
      if(tk.eq.0) then
         yp  = 1.d8
      elseif(uk.eq.0) then
         yp  = -1.d8
      else
         yp  = .5d0*log( uk/tk )
      endif
c-----------------------------------------------------------------
c xktsq, xkt1sq e xkt2sq are the square of transverse momenta of g, Q, 
c and Qb respectively. The axis orientation is such that Q is always
c along the x direction
c
      xktsq  = uk*tk/s
      if(xktsq.eq.0) then
         pq1(1) = sqrt(x*s)/2.d0*bx*sth1
         pq1(2) = 0.d0
         pq2(1) = -pq1(1)
         pq2(2) = 0.d0
         pp(1) = 0.d0
         pp(2) = 0.d0
      else
         xkt1sq = q2c*q1q/s - xm2
         xkt2sq = q2q*q1c/s - xm2
         xkt = sqrt(xktsq)
         xkt1 = sqrt(xkt1sq)
         xkt2 = sqrt(xkt2sq)
         pq1(1) = xkt1
         pq1(2) = 0.d0
         pq2(1) = (xktsq-xkt1sq-xkt2sq)/(2.d0*xkt1)
         tmp = xkt2sq-pq2(1)**2
         if(tmp.gt.0.d0)then
            pq2(2) = sqrt(tmp)
         else
            pq2(2) = 0.d0
         endif
         pp(1) = (xkt2sq-xkt1sq-xktsq)/(2.d0*xkt1)
         tmp = xktsq-pp(1)**2
         if(tmp.gt.0.d0)then
            pp(2) = -sqrt(tmp)
         else
            pp(2) = 0.d0
         endif
      endif
c Incoming parton
      sqs=sqrt(s)
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqs/2.d0
      xmom_cm(1,4)=sqs/2.d0
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqs/2.d0
      xmom_cm(2,4)=sqs/2.d0
c Outgoing light parton
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
        xmom_cm(3,1)=pp(1)
        xmom_cm(3,2)=pp(2)
        xmom_cm(3,3)=sqs/2.d0*(1-x)*y
        xmom_cm(3,4)=sqs/2.d0*(1-x)
      endif
c Heavy quark
      e1lab=-(q1q+q2c)/(2*sqs)
      pl1lab=(q1q-q2c)/(2*sqs)
      xmom_cm(4,1)=pq1(1)
      xmom_cm(4,2)=pq1(2)
      xmom_cm(4,3)=pl1lab
      xmom_cm(4,4)=e1lab
c Heavy antiquark
      e2lab=-(q1c+q2q)/(2*sqs)
      pl2lab=(q1c-q2q)/(2*sqs)
      xmom_cm(5,1)=pq2(1)
      xmom_cm(5,2)=pq2(2)
      xmom_cm(5,3)=pl2lab
      xmom_cm(5,4)=e2lab
c
      if(ichkmom.eq.0)call checkmqq(xmom_cm,s,0.d0,1,2)
      return
      end


      subroutine checkmqq(xmom,smax,ybst,iflag,itype)
      implicit none
      real * 8 xmom(11,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-5)
      integer iflag,itype,i,j,jj,jflag,jeflag,jmax
c
      if(itype.eq.1)then
        jmax=11
      elseif(itype.eq.2)then
        jmax=5
      else
        write(6,*)'Wrong option in checkmqq'
        stop
      endif
      jflag=0
      jeflag=0
      xpmax=sqrt(smax)/2.d0*(1+vtiny)
      do i=1,4
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=1,jmax
          if((itype.eq.1.and.j.ne.4.and.j.ne.5).or.itype.eq.2)then
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
          do j=1,11
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
        do j=1,11
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end


      subroutine checktdec1(xmt,xtq,xbq,xel,xnu,itop)
c Checks momentum conservation in top decay
      implicit none
      real*8 xmt,tiny,diff,xtq(4),xbq(4),xel(4),xnu(4)
      parameter (tiny=1.d-8)
      integer itop,i
c
      do i=1,4
        diff=xtq(i)-xbq(i)-xel(i)-xnu(i)
        if(abs(diff).gt.tiny*xmt)then
          write(6,*)'Subroutine checktdec1'
          write(6,*)'Momentum is not conserved in decay',i,itop
          stop
        endif
      enddo
      return
      end


      subroutine checktdec2(xmom,idec,iprod1,iprod2,iprod3)
c Checks momentum conservation in top decay, after manipulations 
c in put_on_sheqq()
      implicit none
      real * 8 xmom(11,4)
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
          write(*,*)'Subroutine checktdec2'
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


      subroutine chvqq(z,x,xjac,del)
c
      implicit none
      real * 8 z,x,xjac,del,odel
      real * 8 xld,x0,y0,yh,yl,ydif,y,t
      data odel/0.d0/
      if(odel.ne.del) then
         odel = del
         xld = log(del)
         x0  = 1 + del + 1/xld
         if(x0.le.0.or.x0.ge.1) then
            call hvqwarn('CHVAR')
            write(*,*) 'inappropriate delta'
            stop
         endif
         y0  = log(1-x0+del)
         yh = y0 - x0*xld
         yl = xld
         ydif = yh-yl
      endif
      y = yl + z*ydif
      xjac = xjac * ydif
      if(y.gt.y0) then
          x = - (yh - y)/xld
          xjac = xjac / (-xld)
      else
          t = exp(y)
          xjac = xjac * t
          x = 1 - t + del
      endif
      return
      end

C      subroutine zzchvar(parth1,cth1,xjac,ro)
c
c Given 0<parth1<1 returns -1<cth1<1
c and multiplies xjac times the d cth1 / d parth1 jacobian
c
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
Cc
c
c Initialization
c
c
      subroutine setpqq()
c Electroweak parameters for top decay; this routine is a modification
c of that of the VB code
      implicit none
      include 'hvqcblks.h'
      real * 8 pi,zero,one,xme,xmmu,xmtau
      parameter (pi=3.14159265358979312D0)
      parameter (one=1.d0)
      parameter (zero=0.d0)
c Values from PDG 2003
      parameter (xme=0.510998902d-3)
      parameter (xmmu=105.6583568d-3)
      parameter (xmtau=1776.99d-3)
      real * 8 xmw,gaw
      common/cwparam/xmw,gaw
      real * 8 xlep1mass(2),xlep2mass(2)
      common/clepmass/xlep1mass,xlep2mass
      real * 8 xm1low2,xm1upp2,xm2low2,xm2upp2
      common/bounmc/xm1low2,xm1upp2,xm2low2,xm2upp2
      real * 8 brrtop1,brrtop2
      common/brratios/brrtop1,brrtop2
      real * 8 sthw2,cthw2
      common/cweinan/sthw2,cthw2
      real * 8 xmt,twidth
      common/ctparam/xmt,twidth
      integer idec
      common/cidec/idec
      integer iwidth
      common/ciwidth/iwidth
c Type of V decays, with HERWIG conventions; see the beginning of this file
      integer il1hw,il2hw
      common/cilhw/il1hw,il2hw
c Identities of the vector bosons or leptons in the final state 
c (PDG conventions)
      integer ip4,ip5,ip6,ip7,ip8,ip9
      common/ci2part/ip4,ip5,ip6,ip7,ip8,ip9
c PDG codes for charged leptons and neutrinos for a given IL (NLO) code;
c the particle code (not the antiparticle) is entered here
c Charged lepton from W decay
      integer ichlw(1:3)
      data ichlw/11,13,15/
c Neutrino from W decay
      integer ineuw(1:3)
      data ineuw/12,14,16/
      real * 8 alfaem,xalfaem,xmwme,gawme,topdecw,brtop,xmw2,tmpmss(3)
c
c Electron charge squared (computed at the W mass)
      alfaem = xalfaem(xmw)
      ze2 = 4*pi*alfaem
c sin and cos squared of theta_W; MSbar scheme, from PDG2003
      sthw2=0.23113d0
      cthw2=1-sthw2
c Lepton masses and identities: xlep#mass(i) is the mass of lepton # in 
c the decay of top (i=1) or tbar (i=2)
      tmpmss(1)=xme
      tmpmss(2)=xmmu
      tmpmss(3)=xmtau
      ip4=-ichlw(il1hw)
      ip5=ineuw(il1hw)
      ip6=5
      ip7=ichlw(il2hw)
      ip8=-ineuw(il2hw)
      ip9=-5
      if(il1hw.le.3)then
        xlep1mass(1)=tmpmss(il1hw)
        xlep2mass(1)=0.d0
      else
        write(*,*)'Error in setpqq: inconsistent entries'
        stop
      endif
      if(il2hw.le.3)then
        xlep1mass(2)=tmpmss(il2hw)
        xlep2mass(2)=0.d0
      else
        write(*,*)'Error in setpqq: inconsistent entries'
        stop
      endif
c Fills MadEvent common blocks. Set positron charge and QCD coupling g 
c equal to one, and use the actual values in the main code
      xmwme=xmw
      gawme=gaw
      call setmepqq(xmwme,gawme,zero,zero,
     #              xmt,twidth,zero,sthw2,one,one)
c Compute branching ratios; we set here |Vtb|=1
      if(iwidth.eq.1)then
        brrtop1=topdecw(xmt,xmw,gaw,xm1low2,xm1upp2,sthw2)/twidth
        brrtop2=topdecw(xmt,xmw,gaw,xm2low2,xm2upp2,sthw2)/twidth
        brrtop1=ze2**2*brrtop1
        brrtop2=ze2**2*brrtop2
      else
        xmw2=xmw**2
        brtop=(xm2-xmw2)**2*(xm2+2*xmw2)/(6144*pi**3*xmt**3)
        brtop=brtop * 2/(sthw2**2 * xmw2*gaw**2)
        brtop=brtop/twidth
        brrtop1=ze2**2*brtop
        brrtop2=ze2**2*brtop
      endif
c
      return
      end


      function topdecw(xmt,xmw,wwidth,xmw2low,xmw2upp,sthw2)
c Returns top decay width integrated over W virtuality, as computed
c in topwidth.m. Insert a factor e^4*|Vtb|^2 for the correct normalization
      implicit none
      real*8 topdecw,xmt,xmw,wwidth,xmw2low,xmw2upp,sthw2
      real*8 pi,norm,tmp
      parameter (pi=3.1415926535897932d0)
c
      norm=1/(3072*pi**3*wwidth*xmw*sthw2**2*xmt**3)
      tmp=( xmt**6-6*wwidth**2*xmw**4+2*xmw**6+
     #      3*xmt**2*xmw**2*(wwidth**2-xmw**2) )*
     #    ( atan((xmw2upp-xmw**2)/(wwidth*xmw)) -
     #      atan((xmw2low-xmw**2)/(wwidth*xmw)) ) +
     #    wwidth*xmw* (
     #      (xmw2upp-xmw2low)*(4*xmw**2+xmw2low+xmw2upp-3*xmt**2)+
     #      xmw**2*(3*xmt**2+wwidth**2-3*xmw**2)*(
     #        log(wwidth**2*xmw**2 + (xmw2low - xmw**2)**2) - 
     #        log(wwidth**2*xmw**2 + (xmw2upp - xmw**2)**2) ) )
      topdecw=norm*tmp
      return
      end


      subroutine parsetpqq()
      implicit none
      integer jproc,ileg,ie0sq,i,itype
      integer imapp(0:5)
      integer ialwsplit(1:3,1:4,1:3)
      integer icllborn(1:3,1:4,1:3),icllkern(1:3,1:4,1:3)
      integer icolconn(1:3,1:4,1:3)
      integer ivbhpro(4,3,5)
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
      character * 2 xproc(3)
      common/cimapp/imapp
      common/cialwsplit/ialwsplit
      common/cicllsplit/icllborn,icllkern
      common/cicolconn/icolconn
      common/civbhpro/ivbhpro
      common/cidpart/idp1,idp2,idp3
      common/cxproc/xproc
c
      xproc(1)='gg'
      xproc(2)='qq'
      xproc(3)='qg'
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
c ialwsplit(jproc,ileg,ie0sq) returns 0 if there's no splitting from leg ileg
c in the 2-->3 process whose initial state is identified by jproc (jproc=1,2,3
c ==> prc=gg,qq,qg respectively), and whose colour connection corresponds to
c the scale choice ie0sq (ie0sq=1,2,3 ==> e0sq=s/2,-t/2,-u/2)
      do jproc=1,3
        do ileg=1,4
          do ie0sq=1,3
            ialwsplit(jproc,ileg,ie0sq)=0
          enddo
        enddo
      enddo
c jproc=1 (gg)
      do ileg=1,4
        do ie0sq=1,3
          if(ileg.le.2.or.(ileg.gt.2.and.ie0sq.ne.1))
     #      ialwsplit(1,ileg,ie0sq)=1
        enddo
      enddo
c jproc=2 (qq)
      do ileg=1,4
        ialwsplit(2,ileg,2)=1
      enddo
c jproc=3 (qg)
      do ie0sq=1,3
        ialwsplit(3,1,ie0sq)=1
      enddo
      ialwsplit(3,2,2)=1
c
c icllborn(jproc,ileg,ie0sq) returns the code (1==gg, 2==qq) for the initial
c state of the Born that factorizes in the MC subtraction terms
      do jproc=1,3
        do ileg=1,4
          do ie0sq=1,3
            icllborn(jproc,ileg,ie0sq)=0
          enddo
        enddo
      enddo
c Final-state emissions
      do jproc=1,3
        do ileg=3,4
          do ie0sq=1,3
            if(ialwsplit(jproc,ileg,ie0sq).eq.1)
     #        icllborn(jproc,ileg,ie0sq)=jproc
          enddo
        enddo
      enddo
c jproc=1 (gg)
      do ileg=1,2
        do ie0sq=1,3
          if(ialwsplit(1,ileg,ie0sq).eq.1)
     #     icllborn(1,ileg,ie0sq)=1
        enddo
      enddo
c jproc=2 (qq)
      do ileg=1,2
        do ie0sq=1,3
          if(ialwsplit(2,ileg,ie0sq).eq.1)
     #     icllborn(2,ileg,ie0sq)=2
        enddo
      enddo
c jproc=3 (qg)
      do ie0sq=1,3
        icllborn(3,1,ie0sq)=1
      enddo
      icllborn(3,2,2)=2
c
c icllkern(jproc,ileg,ie0sq) returns the code (1,..,4, see the function
c ap_kern) of the Altarelli-Parisi splitting function that factorizes in 
c the MC subtraction terms
      do jproc=1,3
        do ileg=1,4
          do ie0sq=1,3
            icllkern(jproc,ileg,ie0sq)=0
          enddo
        enddo
      enddo
c Final-state emissions
      do jproc=1,3
        do ileg=3,4
          do ie0sq=1,3
            if(ialwsplit(jproc,ileg,ie0sq).eq.1)
     #        icllkern(jproc,ileg,ie0sq)=4
          enddo
        enddo
      enddo
c jproc=1 (gg)
      do ileg=1,2
        do ie0sq=1,3
          if(ialwsplit(1,ileg,ie0sq).eq.1)
     #     icllkern(1,ileg,ie0sq)=1
        enddo
      enddo
c jproc=2 (qq)
      do ileg=1,2
        do ie0sq=1,3
          if(ialwsplit(2,ileg,ie0sq).eq.1)
     #     icllkern(2,ileg,ie0sq)=4
        enddo
      enddo
c jproc=3 (qg)
      do ie0sq=1,3
        icllkern(3,1,ie0sq)=3
      enddo
      icllkern(3,2,2)=2
c
c If icolconn(jproc,ileg,ie0sq)<0, the corresponding 2-->2 reduced matrix
c element squared (d\bar{sigma}) is multiplied by 1/|icolconn|; if it's 
c positive, it is also multiplied by u**2/(u**2+t**2) or t**2/(u**2+t**2), 
c depending on the colour structure chosen. See hvqborncol
      do jproc=1,3
        do ileg=1,4
          do ie0sq=1,3
            icolconn(jproc,ileg,ie0sq)=0
          enddo
        enddo
      enddo
c jproc=1 (gg)
      icolconn(1,1,1)=-2
      icolconn(1,1,2)=2
      icolconn(1,1,3)=2
      icolconn(1,2,1)=-2
      icolconn(1,2,2)=2
      icolconn(1,2,3)=2
      do ileg=3,4
        do ie0sq=1,3
          if(ialwsplit(1,ileg,ie0sq).eq.1)
     #     icolconn(1,ileg,ie0sq)=1
        enddo
      enddo
c jproc=2 (qq)
      do ileg=1,4
        do ie0sq=1,3
          if(ialwsplit(2,ileg,ie0sq).eq.1)
     #     icolconn(2,ileg,ie0sq)=-1
        enddo
      enddo
c jproc=3 (qg)
      icolconn(3,1,1)=-2
      icolconn(3,1,2)=2
      icolconn(3,1,3)=2
      icolconn(3,2,2)=-1
c
c ivbhpro returns the process number associated to the entries; this is
c identical to i1hpro (see the routine store_evenqq)
      do i=1,4
        do jproc=1,3
          do itype=1,5
            ivbhpro(i,jproc,itype)=0
          enddo
        enddo
      enddo
c 
      do i=1,4
        ivbhpro(i,1,1)=407
      enddo
c 
      do itype=1,5
        ivbhpro(1,2,itype)=401
        ivbhpro(2,2,itype)=403
        ivbhpro(3,2,itype)=403
        ivbhpro(4,2,itype)=401
        ivbhpro(1,3,itype)=402
        ivbhpro(2,3,itype)=404
        ivbhpro(3,3,itype)=405
        ivbhpro(4,3,itype)=406
      enddo
c
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=outgoing) in the process associated to the
c entries. The labelling scheme of PDG has been used
      do i=1,4
        do jproc=1,3
          do itype=1,5
            idp1(i,jproc,itype)=0
            idp2(i,jproc,itype)=0
            idp3(i,jproc,itype)=0
          enddo
        enddo
      enddo
c
      do i=1,4
        idp1(i,1,1)=21
        idp2(i,1,1)=21
        idp3(i,1,1)=21
      enddo
c
      do itype=1,5
        idp1(1,2,itype)=imapp(itype)
        idp1(2,2,itype)=-imapp(itype)
        idp1(3,2,itype)=-imapp(itype)
        idp1(4,2,itype)=imapp(itype)
c
        idp1(1,3,itype)=imapp(itype)
        idp1(2,3,itype)=-imapp(itype)
        idp1(3,3,itype)=21
        idp1(4,3,itype)=21
c
        idp2(1,2,itype)=-imapp(itype)
        idp2(2,2,itype)=imapp(itype)
        idp2(3,2,itype)=imapp(itype)
        idp2(4,2,itype)=-imapp(itype)
c
        idp2(1,3,itype)=21
        idp2(2,3,itype)=21
        idp2(3,3,itype)=imapp(itype)
        idp2(4,3,itype)=-imapp(itype)
c
        idp3(1,2,itype)=21
        idp3(2,2,itype)=21
        idp3(3,2,itype)=21
        idp3(4,2,itype)=21
c
        idp3(1,3,itype)=imapp(itype)
        idp3(2,3,itype)=-imapp(itype)
        idp3(3,3,itype)=imapp(itype)
        idp3(4,3,itype)=-imapp(itype)
      enddo
c
      call parchechpqq()
      return
      end


      subroutine parchechpqq()
      implicit none
      integer iallzero,i,jproc,itype,ihpro,i1,i2,i3
      parameter (iallzero=0)
      integer ivbhpro(4,3,5)
      common/civbhpro/ivbhpro
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
      common/cidpart/idp1,idp2,idp3
c
      call parcheckinqq()
      do i=1,4
        do jproc=1,3
          do itype=1,5
            ihpro=ivbhpro(i,jproc,itype)
            i1=idp1(i,jproc,itype)
            i2=idp2(i,jproc,itype)
            i3=idp3(i,jproc,itype)
            call parcheckfqq(ihpro,i1,i2,i3,iallzero)
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfqq(ihpro,i1,i2,i3,iallzero)
      implicit none
      integer ihpro,i1,i2,i3,iallzero,isum,chin,chout,chall
      real*8 tiny
      parameter (tiny=1.d-8)
      logical ferror
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
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
        write(*,*)'parcheckfqq called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or.
     #       i3.ne.21 .or. (i1+i2).ne.0) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.le.0 .or. i1.ne.i3) )ferror=.true.
c 403 is qqbar
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. 
     #       i3.ne.21 .or. (i1+i2).ne.0) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.ge.0 .or. i1.ne.i3) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.le.0 .or. i2.ne.i3) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.ge.0 .or. i2.ne.i3) )ferror=.true.
c 407 is gg
        if( ihpro.eq.407 .and.
     #      (i1.ne.21 .or. i2.ne.21 .or. 
     #       i3.ne.21) )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfqq'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinqq()
      implicit none
      integer i
      real*8 chup,chdn
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
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
      chprdct=0.d0
      return
      end
c
c
c Begin of event file utilities
c
c
      subroutine whichoqq(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
c Determines the type of event at random
      implicit none
      integer iseed,itot,mx_of_evta,mx_of_evtb,iunit,i0
      real*8 xpa,xpb,tiny,one,xsum,rnd,fk88random,prob
      parameter (tiny=1.d-4)
      logical flag
c
      if(itot.le.0)then
        write(6,*)'Fatal error #1 in whichoqq'
        stop
      endif
      xpa=dfloat(mx_of_evta)/dfloat(itot)
      xpb=dfloat(mx_of_evtb)/dfloat(itot)
      one=xpa+xpb
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'Error #1 in whichoqq: probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.2)then
          write(6,*)'Fatal error #2 in whichoqq'
          stop
        endif
        i0=i0+1
        prob=xpa
        if(i0.gt.1)prob=xpb
        xsum=xsum+prob
        if(rnd.lt.xsum)then
          flag=.false.
          itot=itot-1
          if(i0.le.1)then
            mx_of_evta=mx_of_evta-1
          else
            mx_of_evtb=mx_of_evtb-1
          endif
          iunit=20+i0
        endif
      enddo
      return
      end


      subroutine crosscheqq(itot,mx_of_evta,mx_of_evtb)
c Checks whether whichoqq did it right
      implicit none
      integer itot,mx_of_evta,mx_of_evtb
c
      if(itot.ne.0)then
        write(6,*)'Error: itot=',itot
        stop
      endif
      if(mx_of_evta.ne.0)then
        write(6,*)'Error: mx_of_evta=',mx_of_evta
        stop
      endif
      if(mx_of_evtb.ne.0)then
        write(6,*)'Error: mx_of_evtb=',mx_of_evtb
        stop
      endif
      return
      end


      subroutine retrieve_evenqq(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evenqq
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6,ip7,ip8,ip9
      common/ci2part/ip4,ip5,ip6,ip7,ip8,ip9
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(11,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      if(idec.eq.0)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,
     #                                 ip4,ip5,ip6,ip7,ip8,ip9
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,3),
     #                                 ((xmom_lb(i,j),j=1,4),i=6,11)
      elseif(idec.eq.1)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,5)
      endif
      goto 999
 901  format(1x,i3,2(1x,i2))
 902  format(9(1x,i3))
 903  format(1x,d14.8)
 904  format(36(1x,d14.8))
 997  write(*,*)'unexpected end of file, iunit=',iunit
      stop
 998  write(*,*)'format error'
      write(77,*)'event #:',ii
      write(77,901)i1hpro,iccode,np
      write(77,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7,ip8,ip9
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,11)
      stop
 999  continue
      return
      end


      subroutine store_evenqq(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, NON trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip9 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_evenqq), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar -> g Q Qb
c    402        q g    -> q Q Qb
c    403        qbar q -> g Q Qb
c    404        qbar g -> qbar Q Qb
c    405        g q    -> q Q Qb
c    406        g qbar -> qbar Q Qb
c    407        g g    -> g Q Qb
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g. Note that, at variance with
c what happens for xmom_lb, in the case if ipX only the identities of
c final-state particles are kept. Thus, (ip4,ip5)=(Q,Qbar) identities
c when tops don't decay. But (ip4,ip5,ip6)=identities of top decay
c products, and (ip7,ip8,ip9)=identities of tbar decay products when
c tops decay.
c iccode is the (internal) code which identifies the colour connection
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact,brfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6,ip7,ip8,ip9
      common/ci2part/ip4,ip5,ip6,ip7,ip8,ip9
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
c partons, 3 is the outgoing parton, 4 is Q, 5 is Qbar. When the tops
c decay, 6=l+, 7=nu, 8=b are the decay products of the top, 9=l-, 10=nubar,
c 11=bbar are the decay products of the tbar. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+6+7+8+9+10+11)=0
      real*8 xmom_lb(11,4)
      common/cxmomlb/xmom_lb
      integer iwgtnorm
      common/ciwgtnorm/iwgtnorm
      real*8 wgtaev,wgtbev
      common/cwgtev/wgtaev,wgtbev
c Branching ratios, to be inserted in the case of decayed tops
      real*8 brrtop1,brrtop2
      common/brratios/brrtop1,brrtop2
c
      if(xpmone.eq.-1)then
c Events are already stored in temporary files, and are passed to this
c routines through common blocks filled by retrieve_evenqq
        xevwgt=xevsign
        xfact=1.d0
      elseif(xpmone.eq.1)then
c Events are obtained from SPRING, and are written to temporary files
c for the first time
        if(idec.eq.0)then
          np=9
          brfact=brrtop1*brrtop2
        elseif(idec.eq.1)then
          np=5
          brfact=1.d0
        else
          write(6,*) 'Error in store_evenqq: idec=',idec
          stop
        endif
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev*brfact
        else
          xevwgt=xevsign*wgtaev*brfact
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_evenqq: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      if(idec.eq.0)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7,ip8,ip9
        write(iunit,903)xevwgt
        write(iunit,904)((xfact*xmom_lb(i,j),j=1,4),i=1,3),
     #                  ((xfact*xmom_lb(i,j),j=1,4),i=6,11)
      elseif(idec.eq.1)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5
        write(iunit,903)xevwgt
        write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,5)
      endif
 901  format(1x,i3,2(1x,i2))
 902  format(9(1x,i3))
 903  format(1x,d14.8)
 904  format(36(1x,d14.8))
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
      subroutine xmcsuqq(xz1,xz2,xxmq2,xs,xx,xy,xcth1,xcth2,
     #  xmcxsec,xmce0sq,xmcz,xqrksc,xqbrsc,flxsec,flagmc,
     #  gfactsf,gfactcl)
c xmcxsec is the main output of this routine, and is the analogue of the 
c G(x)-part in eqs.(A.83) and (A.84) of FW. It is defined as follows:
c  g^4 xmcxsec = P(z)/((1-x)xi) (4 tk uk/s**2) 4*M_born Jac(xi,z,phi;x,y,th2)
c In the arrays xmcxsec, xmce0sq, xmcz, and flxsec the two arguments 
c (ileg,ie0sq) are such that
c   ileg=1,2,3,4 --> emitting leg
c   ie0sq=1,2,3 --> scale choice, corresponding to s/2, -t/2, -u/2.
c xmce0sq the value of the Herwig scale E_0.
c xmcz the value of the Herwig variable z.
c xqrksc(iinv,ileg,ie0sq) have the formal meaning of 2p1.p2, -2p1.k1,
c and -2p2.k1 for iinv=1,2,3, which are the invqqiants needed to compute 
c the transverse momentum of the quark; xqbrsc is the analogous quantity 
c for the antiquark; pi and ki need not coincide with the actual one, they
c are only used for setting the scales.
c flxsec=.false. when xmcxsec=0, and flagmc=.false. if all flxsec=.false. 
c gfactsf and gfactcl are the values of G_soft(x,s,m2) and G_coll(x,y)
      implicit none
      character*2 str
      parameter (str='p1')
      real*8 tiny,vcf,vca
      parameter (tiny=1.d-5)
      parameter (vcf=4.d0/3.d0)
      parameter (vca=3.d0)
      real*8 xz1,xz2,xxmq2,xs,xx,xy,xcth1,xcth2,gfactsf,gfactcl
      real*8 xmcxsec(1:4,1:3),xmce0sq(1:4,1:3),xmcz(1:4,1:3)
      real*8 xqrksc(1:3,1:4,1:3),xqbrsc(1:3,1:4,1:3)
      logical flxsec(1:4,1:3),flagmc,flagxs(1:4)
      real*8 z1,z2,xmq2,x,y,cth1,cth2,
     #  si,tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi,
     #  so,tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho,
     #  s,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #  e0sq,de0sqdx,de0sqdc,xz,zherw,xxi,xiherw,
     #  gfunsoft,gfuncoll,gfactazi,gfunazi,
     #  tmp,tmp1,cfact,sth1,cthg,beta,betae0,ap,ap_kern,
     #  qin_kern,xfact,qk,xborn,hvqborncol,xazicorr,
     #  hvqcorr,xjac,xjac_xiztoxy,sx,dcosdcth1
      real*8 xmcxi(1:4,1:3)
      real*8 x2to2(1:4,1:3),dx2to2dx(1:4,1:3),dx2to2dc(1:4,1:3)
      integer jproc0,ifuntype,ileg,ie0sq,index,iborn
      integer ialwsplit(1:3,1:4,1:3)
      integer icllborn(1:3,1:4,1:3),icllkern(1:3,1:4,1:3)
      integer icolconn(1:3,1:4,1:3)
      common/cialwsplit/ialwsplit
      common/cicllsplit/icllborn,icllkern
      common/cicolconn/icolconn
      common/cjproc/jproc0
      common/cifuntype/ifuntype
c
      z1=xz1
      z2=xz2
      xmq2=xxmq2
      s=xs
      x=xx
      y=xy
      cth1=xcth1
      cth2=xcth2
      sx=s*x
      flagxs(1)=.true.
      flagxs(2)=.true.
      flagxs(3)=.true.
      flagxs(4)=.true.
c Compute the invqqiants
      call invqq(xmq2,s,x,y,cth1,cth2,str,
     #           tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi)
      si=s
      if(ifuntype.eq.1)then
        so=si
        tko=tki
        uko=uki
        q1qo=q1qi
        q2qo=q2qi
        q1co=q1ci
        q2co=q2ci
        w1o=w1i
        w2o=w2i
        w1ho=w1hi
        w2ho=w2hi
      elseif(ifuntype.eq.2.and.jproc0.ne.3)then
        if((sx*x).gt.(4*xmq2))then
          call invqq(xmq2,sx,x,y,cth1,cth2,str,
     #               tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho)
          so=sx
        else
          flagxs(3)=.false.
          flagxs(4)=.false.
        endif
      endif
c Generate the 2-->2 invqqiants; those for ileg=2 are identical to the
c corresponding ones for ileg=1, so we skip the generation in such a case
      do ileg=1,4
        if( (ileg.eq.1 .or. (ileg.ge.3.and.jproc0.ne.3)) .and.
     #      flagxs(ileg) )then
          call xinvtoinv(ileg,
     #      si,tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi,
     #      so,tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho,
     #      s,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          call get2to2wr(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #      tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #      x2to2(ileg,1),x2to2(ileg,2),x2to2(ileg,3),
     #      dx2to2dx(ileg,1),dx2to2dx(ileg,2),dx2to2dx(ileg,3),
     #      dx2to2dc(ileg,1),dx2to2dc(ileg,2),dx2to2dc(ileg,3))
        endif
      enddo
      do ie0sq=1,3
        x2to2(2,ie0sq)=x2to2(1,ie0sq)
        dx2to2dx(2,ie0sq)=dx2to2dx(1,ie0sq)
        dx2to2dc(2,ie0sq)=dx2to2dc(1,ie0sq)
      enddo
c Define the invqqiants for the settings of the scales
      do ileg=1,2
        do ie0sq=1,3
          xqrksc(1,ileg,ie0sq)=si
          xqrksc(2,ileg,ie0sq)=q1qi
          xqrksc(3,ileg,ie0sq)=q2ci
          xqbrsc(1,ileg,ie0sq)=si
          xqbrsc(2,ileg,ie0sq)=q1ci
          xqbrsc(3,ileg,ie0sq)=q2qi
        enddo
      enddo
      do ileg=3,4
        do ie0sq=1,3
          xqrksc(1,ileg,ie0sq)=so
          xqrksc(2,ileg,ie0sq)=q1qo
          xqrksc(3,ileg,ie0sq)=q2co
          xqbrsc(1,ileg,ie0sq)=so
          xqbrsc(2,ileg,ie0sq)=q1co
          xqbrsc(3,ileg,ie0sq)=q2qo
        enddo
      enddo
c Now get z and xi for the allowed splittings
      flagmc=.false.
      do ileg=1,4
        call xinvtoinv(ileg,
     #    si,tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi,
     #    so,tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho,
     #    s,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        do ie0sq=1,3
          if(ialwsplit(jproc0,ileg,ie0sq).eq.1.and.flagxs(ileg))then
            e0sq=abs(x2to2(ileg,ie0sq))/2.d0
            de0sqdx=abs(dx2to2dx(ileg,ie0sq))/2.d0*
     #              sign(1.d0,x2to2(ileg,ie0sq))
            de0sqdc=abs(dx2to2dc(ileg,ie0sq))/2.d0*
     #              sign(1.d0,x2to2(ileg,ie0sq))
            xz=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #           cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            xxi=xiherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #           cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
            if(ileg.le.2)then
              if( xz.ge.0.d0 .and. xxi.ge.0.d0. and.
     #            xz**2.ge.xxi )then
                flxsec(ileg,ie0sq)=.true.
                if(.not.flagmc)flagmc=.true.
              else
                flxsec(ileg,ie0sq)=.false.
              endif
            else
              if( xz.ge.0.d0 .and. xxi.ge.0.d0. and.
     #            (e0sq*xxi*xz**2).ge.xmq2.and.xxi.le.1.d0 )then
                flxsec(ileg,ie0sq)=.true.
                if(.not.flagmc)flagmc=.true.
              else
                flxsec(ileg,ie0sq)=.false.
              endif
            endif
            if(flxsec(ileg,ie0sq))then
              xmcz(ileg,ie0sq)=xz
              xmcxi(ileg,ie0sq)=xxi
              xmce0sq(ileg,ie0sq)=e0sq
            endif
          else
            flxsec(ileg,ie0sq)=.false.
          endif
        enddo
      enddo
c Even if flagmc=.false., evaluate the G functions, since they are used
c to compute the ME part of the MC subtraction term
      gfactsf=gfunsoft(x,s,xmq2)
      gfactcl=gfuncoll(x,y)
      gfactazi=gfunazi(x,y)
c Compute the cross sections if in the live zones. Also add the azimuthal
c correlation term (whose kernel is tmp1) when gfactazi#0; since this is
c of ME origin, the sum over ie0sq should not be performed. We (arbitrarily)
c associate it to ie0sq=1 (this is feasible since all the terms with azimuthal
c correlation have a ie0sq=1 contribution)
      if(flagmc)then
        do ileg=1,4
          call xinvtoinv(ileg,
     #      si,tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi,
     #      so,tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho,
     #      s,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          do ie0sq=1,3
            if(flxsec(ileg,ie0sq))then
              e0sq=xmce0sq(ileg,ie0sq)
              tmp1=0.d0
              if(1-x.lt.tiny.and.ileg.le.2)then
                if(jproc0.eq.1)then
                  cfact=vca
                elseif(jproc0.eq.2)then
                  cfact=vcf
                elseif(jproc0.eq.3)then
                  cfact=0.d0
                else
                  write(*,*)'Fatal error #1 in xmcsuqq:',jproc0
                  stop
                endif
                if(cfact.ne.0.d0)then
                  if(ileg.eq.1)then
                    tmp=16*e0sq*(1+y)/(s*(1-y)+4*e0sq*(1+y)) 
                  elseif(ileg.eq.2)then
                    tmp=16*e0sq*(1-y)/(s*(1+y)+4*e0sq*(1-y)) 
                  endif
                endif
                tmp=cfact*abs(tmp)
              elseif(1-x.lt.tiny.and.ileg.ge.3)then
                if(jproc0.ne.3)then
                  sth1=sqrt(1-cth1**2)
                  cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
                  beta=sqrt(1-4*xmq2/s)
                  betae0=sqrt(1-xmq2/e0sq)
                  if(ileg.eq.3)then
                    tmp=2*beta*betae0*(1-y**2)/( (1-beta*cthg)*
     #               (betae0-(betae0-beta)*(1-beta*cthg)/(1-beta**2)) )
                  elseif(ileg.eq.4)then
                    tmp=2*beta*betae0*(1-y**2)/( (1+beta*cthg)*
     #               (betae0-(betae0-beta)*(1+beta*cthg)/(1-beta**2)) )
                  endif
                  tmp=vcf*abs(tmp)
                else
                  tmp=0.d0
                endif
              elseif(1-y.lt.tiny.and.ileg.eq.1)then
                index=icllkern(jproc0,ileg,ie0sq)
                tmp=(1+y)*ap_kern(x,index)
                if(gfactazi.ne.0.d0.and.ie0sq.eq.1)
     #            tmp1=(1+y)*qin_kern(x,index)
              elseif(1+y.lt.tiny.and.ileg.eq.2)then
                index=icllkern(jproc0,ileg,ie0sq)
                tmp=(1-y)*ap_kern(x,index)
                if(gfactazi.ne.0.d0.and.ie0sq.eq.1)
     #            tmp1=(1-y)*qin_kern(x,index)
              else
                xfact=(1-x)*(1-y**2)
                xz=xmcz(ileg,ie0sq)
                xxi=xmcxi(ileg,ie0sq)
                index=icllkern(jproc0,ileg,ie0sq)
                ap=ap_kern(xz,index)/(1-xz)
                de0sqdx=abs(dx2to2dx(ileg,ie0sq))/2.d0*
     #                  sign(1.d0,x2to2(ileg,ie0sq))
                de0sqdc=abs(dx2to2dc(ileg,ie0sq))/2.d0*
     #                  sign(1.d0,x2to2(ileg,ie0sq))
                xjac=xjac_xiztoxy(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,
     #               y,cth1,cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
                tmp=xfact*xjac*ap/xxi
                if(ileg.le.2.and.gfactazi.ne.0.d0.and.ie0sq.eq.1)then
                  qk=qin_kern(xz,index)/(1-xz)
                  tmp1=xfact*xjac*qk/xxi
                endif
              endif
              iborn=icllborn(jproc0,ileg,ie0sq)
              xborn=hvqborncol(x2to2(ileg,1),x2to2(ileg,2),xmq2,
     #                         iborn,jproc0,ileg,ie0sq)
              if(ileg.le.2)xazicorr=hvqcorr(x2to2(ileg,1),
     #          x2to2(ileg,2),cth2,xmq2,iborn)
              xmcxsec(ileg,ie0sq)=4*tmp*xborn*gfactsf*gfactcl+
     #                            4*tmp1*xazicorr*gfactazi
              xmcxsec(ileg,ie0sq)=xmcxsec(ileg,ie0sq)*
     #                            dcosdcth1(ileg,x,cth1)
            else
              xmcxsec(ileg,ie0sq)=0.d0
            endif
          enddo
        enddo
      endif
      return
      end


      function dcosdcth1(ileg,x,cth1)
c Returns dcos(theta_in)/dcos(th1) or dcos(theta_out)/dcos(th1)
      implicit none
      real*8 dcosdcth1,x,cth1,tmp
      integer ileg
c
      tmp=1.d0
      if(ileg.eq.3.or.ileg.eq.4)then
        tmp=4*x/(1+x-(1-x)*cth1)**2
      elseif(ileg.ne.1.and.ileg.ne.2)then
        write(*,*)'Error in dcosdcth1: unknown ileg',ileg
        stop
      endif
      dcosdcth1=tmp
      return
      end
    

      subroutine xinvtoinv(ileg,
     #  si,tki,uki,q1qi,q2qi,q1ci,q2ci,w1i,w2i,w1hi,w2hi,
     #  so,tko,uko,q1qo,q2qo,q1co,q2co,w1o,w2o,w1ho,w2ho,
     #  s,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      implicit real*8(a-h,o-z)
      integer ileg
c
      if(ileg.le.2)then
        s=si
        tk=tki
        uk=uki
        q1q=q1qi
        q2q=q2qi
        q1c=q1ci
        q2c=q2ci
        w1=w1i
        w2=w2i
        w1h=w1hi
        w2h=w2hi
      elseif(ileg.le.4)then
        s=so
        tk=tko
        uk=uko
        q1q=q1qo
        q2q=q2qo
        q1c=q1co
        q2c=q2co
        w1=w1o
        w2=w2o
        w1h=w1ho
        w2h=w2ho
      else
        write(*,*)'Unknown leg in xinvtoinv:',ileg
        stop
      endif
      return
      end


      function gfunsoft(xx,xs,xxmq2)
c Gets smoothly to 0 in the soft limit. The functional form is given
c in eq.(A.86) of FW, with alpha==alsf. tilde{x}_{DZ} is replaced here
c by xgsoft, and x_{DZ} by xminsf. The function is different from 1
c in the region xgsoft<x<1. Call with
c  besf<0  ==> xminsf=4*m2/S_{hadr}
c  besf>0  ==> xminsf=tilde{rho} for standard subtraction
c              xminsf=1-sqrt{zeta} for zeta-subtraction
c When besf>0, besf-->0 ==> xgsoft-->1
c              besf-->1 ==> xgsoft-->xminsf
c If alsf<0, gfunsoft equals 1 everywhere. This option should be used
c for testing purposes only
      implicit real*8(a-h,o-z)
      common/shadr/sh
      common/cgfunsfp/alsf,besf
      common/cxgsoft/xgsoft
      common/betfac/betfac,delta
      common/cetacut/etacut
      common/cisubttype/isubttype
c
      x=xx
      s=xs
      xmq2=xxmq2
      if(besf.lt.0.d0)then
        xminsf=4*xmq2/sh
      else
        if(isubttype.eq.0)then
c This is tilde{ro}; don't use tilde{rox}, G_soft gets too fast to zero
          xminsf=1.d0-(1-4*xmq2/s)*betfac**2
        elseif(isubttype.eq.1)then
          xminsf=1-sqrt(etacut)
        else
          write(*,*)'Fatal error #1 in gfunsoft',isubttype
          stop
        endif
      endif
      xgsoft=1.d0-(1-xminsf)*abs(besf)
      if(xgsoft.gt.0.99d0)xgsoft=0.99d0
      tt=(x-xgsoft)/(1.d0-xgsoft)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfunsoft',x
        stop
      endif
      tmp=1.d0
      if(tt.gt.0.d0.and.alsf.gt.0.d0)
     #  tmp=(1-tt)**(2*alsf)/(tt**(2*alsf)+(1-tt)**(2*alsf))
      gfunsoft=tmp
      return
      end


      function gfuncoll(xx,yy)
c Gets smoothly to 0 in the collinear limits; the function gfunsoft
c must be called before this function. The functional form is given
c in eq.(A.86) of FW, with alpha==alcl. tilde{x}_{DZ} is replaced here
c by ygcoll, and x_{DZ} by ymincl. The function is different from 1
c in the region ygcoll<|y|<1. Call with
c  becl<0  ==> ymincl=0
c  becl>0  ==> ymincl=Max(0,1-delta) for standard subtraction
c              ymincl=0 for zeta-subtraction
c When becl>0, becl-->0 ==> ygcoll-->1
c              becl-->1 ==> ygcoll-->ymincl
c If the function is called with 
c  gacl<0 
c the function behaves as described above only for x>xgsoft (where xgsoft 
c is set by the function gfunsoft), while for x<xgsoft it returns one.
c If alcl<0, gfunsoft equals 1 everywhere. This option should be used
c for testing purposes only
      implicit real*8(a-h,o-z)
      common/cgfunclp/alcl,becl,gacl
      common/cxgsoft/xgsoft
      common/betfac/betfac,delta
      common/cetacut/etacut
      common/cisubttype/isubttype
c
      x=xx
      y=yy
      if(gacl.lt.0.d0.and.x.lt.xgsoft)then
        tmp=1.d0
      else
        if(becl.lt.0.d0)then
          ymincl=0.d0
        else
          if(isubttype.eq.0)then
            ymincl=max(0.d0,1.d0-delta)
          elseif(isubttype.eq.1)then
            ymincl=0.d0
          else
            write(*,*)'Fatal error #1 in gfuncoll',isubttype
            stop
          endif
        endif
      endif
      ygcoll=1.d0-(1-ymincl)*abs(becl)
      if(ygcoll.gt.0.99d0)ygcoll=0.99d0
      tt=(abs(y)-ygcoll)/(1.d0-ygcoll)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfuncoll',tt
        stop
      endif
      tmp=1.d0
      if(tt.gt.0.d0.and.alcl.gt.0.d0)
     #  tmp=(1-tt)**(2*alcl)/(tt**(2*alcl)+(1-tt)**(2*alcl))
      gfuncoll=tmp
      return
      end


      function gfunazi(xx,yy)
c This function multiplies the azimuthal correlation term in the MC 
c subtraction kernel; it is therefore unnecessary when the kernel
c smoothly vanishes (because of gfuncoll) in the collinear limit in 
c the whole x range, and it is set to zero in such a case (which 
c corresponds to gacl>0). Otherwise, gfunazi gets smoothly to 1 in 
c the collinear limit, in the region 0<x<xgsoft (where xgsoft is set 
c by the function gfunsoft). 
c When alcl<0, gfunazi is set equal to one. This option should be used
c for testing purposes only
      implicit real*8(a-h,o-z)
      common/cgfunclp/alcl,becl,gacl
      common/cxgsoft/xgsoft
c
      tmp=0.d0
      if(gacl.lt.0.d0.and.x.lt.xgsoft)then
        tga=gacl
        gacl=1.d0
        tmp=1.d0-gfuncoll(x,y)
        gacl=tga
      elseif(alcl.lt.0.d0)then
        tmp=1.d0
      endif
      gfunazi=tmp
      return
      end


Cc Taken from hdyjetdiff.for; the splitting is b(p)-->a(x*p)+c((1-x)*p)
C      function ap_kern(x,index)
Cc This function returns the quantity (1-x)*P_{ab}(x), where
Cc P_{ab} are the Altarelli-Parisi kernels, and the splitting partons
Cc {ab} are defined with the following conventions
Cc
Cc         index          ab
Cc
Cc           1            gg
Cc           2            qg
Cc           3            gq
Cc           4            qq
Cc
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
Cc
C      if(index.eq.1)then
C        ap_kern=2*vca*(x+(1-x)**2/x+x*(1-x)**2)
C      elseif(index.eq.2)then
C        ap_kern=vtf*(1-x)*(x**2+(1-x)**2)
C      elseif(index.eq.3)then
C        ap_kern=vcf*(1-x)*(1+(1-x)**2)/x
C      elseif(index.eq.4)then
C        ap_kern=vcf*(1+x**2)
C      else
C        write(6,*)'Error in ap_kern: wrong index value',index
C        stop
C      endif
C      return
C      end


      function qin_kern(x,index)
c This function returns the quantity (1-x)*Q_{a*b}(x), where
c Q_{a*b} are the kernels relevant to the azimuthal correlations
c in the splittings of incoming partons; their explicit form can be
c found in eqs.(B.42)--(B.45) of FKS (NPB467(96)399). The splitting 
c partons {a*b} are defined with the following conventions
c
c         index          ab
c
c           1            gg
c           2            qg
c           3            gq
c           4            qq
c
      implicit real * 8 (a-h,o-z)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
c
      if(index.eq.1)then
        qin_kern=-4*vca*(1-x)**2/x
      elseif(index.eq.2)then
        qin_kern=0.d0
      elseif(index.eq.3)then
        qin_kern=-4*vcf*(1-x)**2/x
      elseif(index.eq.4)then
        qin_kern=0.d0
      else
        write(6,*)'Error in qin_kern: wrong index value',index
        stop
      endif
      return
      end


      function hvqborn(xs,xt,xxmq2,ijproc)
c Returns the Born cross section for the partonic process whose initial 
c state is identified by ijproc. The normalization is such that
c dsigma_born(s,t) = g^4 hvqborn(s,t) dphi2(s)
c (see the function ggborn)
      implicit none
      real*8 hvqborn,xs,xt,xxmq2,s,t,xmq2,tmp,ggborn,qqborn
      integer ijproc,jproc
c
      s=xs
      t=xt
      xmq2=xxmq2
      jproc=ijproc
      if(jproc.eq.1)then
        tmp=ggborn(s,t,xmq2)
      elseif(jproc.eq.2)then
        tmp=qqborn(s,t,xmq2)
      elseif(jproc.eq.3)then
        tmp=0.d0
      else
        write(*,*)'Unknown process in hvqborn',jproc
        stop
      endif
      hvqborn=tmp
      return
      end


      function hvqborncol(xs,xt,xxmq2,iborn,jproc,ileg,ie0sq)
c Returns the 2-->2 reduced matrix element squared (d\bar{sigma}) which
c factorizes in the MC subtraction term; the definition is
c   hvqborncol = factor * hvqborn,   where
c   factor = 1/|icolconn|*[1,u**2/(u**2+t**2),t**2/(u**2+t**2)], 
c the term is square brackets depending on the colour structure; icolconn 
c is defined in parsetpqq, iborn is the 2-->2 initial state, whereas jproc 
c identifies the 2-->3 one.
      implicit none
      real*8 hvqborncol,xs,xt,xxmq2,s,t,xmq2,u,xfact,xborn,hvqborn
      integer iborn,jproc,ileg,ie0sq
      integer icolconn(1:3,1:4,1:3)
      common/cicolconn/icolconn
c
      s=xs
      t=xt
      xmq2=xxmq2
      u=-s-t
      if(icolconn(jproc,ileg,ie0sq).eq.0)then
        xfact=0.d0
      else
        xborn=hvqborn(s,t,xmq2,iborn)
        xfact=1.d0/dfloat(abs(icolconn(jproc,ileg,ie0sq)))
        if(icolconn(jproc,ileg,ie0sq).gt.0)then
          if(ie0sq.eq.1)then
            write(*,*)
     #        'hvqborncol: no such configuration in this process'
            stop
          elseif(ie0sq.eq.2)then
            xfact=xfact*u**2/(t**2+u**2)
          elseif(ie0sq.eq.3)then
            xfact=xfact*t**2/(t**2+u**2)
          else
            write(*,*)'Fatal error in hvqborncol',ie0sq
            stop
          endif
        endif
      endif
      hvqborncol=xfact*xborn
      return
      end


      function hvqcorr(xs,xt,xcth2,xxmq2,iborn)
c Returns the azimuthal correlation term; it is only used in the case
c in which G_coll does not vanish in the collinear limit for any value
c of x
      implicit real*8(a-h,o-z)
      parameter (xnc=3.d0)
c
      s=xs
      t=xt
      cth2=xcth2
      xmq2=xxmq2
      u=-s-t
      if(iborn.eq.1)then
        tmp=-xnc/(xnc**2-1)*(u/t+t/u-s**2/(xnc**2*t*u))*
     #       (xmq2/s-xmq2**2/(t*u))
        tmp=(2*cth2**2-1)*tmp
      else
        tmp=0.d0
      endif
      hvqcorr=tmp/(2*s)
      return
      end


      function xnlfscheme(xm2,xmur2,xmuf2h1,xmuf2h2,zg2,jproc)
c Returns the factor which multiplies sigma_Born with nl (nlfp1sch=0)
c or nl+1 (nlfp1sch=1) schemes; process code is 1,2,3 for gg,qq,qg
      implicit none
      real*8 xnlfscheme,xm2,xmur2,xmuf2h1,xmuf2h2,zg2,alfas,tmp
      real * 8 tf
      parameter (tf=0.5d0)
      real * 8 pi
      parameter (pi=3.14159265358979312D0)
      integer jproc,nlfp1sch
      common/cnlfp1sch/nlfp1sch
c
      if(nlfp1sch.eq.0)then
        tmp=0.d0
      elseif(nlfp1sch.eq.1)then
        alfas=zg2/(4.d0*pi)
        if(jproc.eq.1)then
          tmp=-alfas*tf/(3.d0*pi)*( log(xmur2/xmuf2h1)+
     #                              log(xmur2/xmuf2h2) )
        elseif(jproc.eq.2)then
          tmp=-alfas*2*tf/(3.d0*pi)*log(xmur2/xm2)
        elseif(jproc.eq.3)then
          tmp=0.d0
        else
          write(*,*)'Unknown process in xnlfscheme:',jproc
        endif
      else
        write(*,*)'Unknown scheme in xnlfscheme:',nlfp1sch
      endif
      xnlfscheme=tmp
      return
      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invqqiants
c
c
      function zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #               cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-5)
c
      sth1=sqrt(1-cth1**2)
c incoming parton (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          zherw=1-(1-x)*(s*(1-y)+4*e0sq*(1+y))/(8*e0sq)
        elseif(1-y.lt.tiny)then
          zherw=x-(1-y)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
        else
          xa=-tk/(2*e0sq)
          xb=-2*uk/s
          ss=1+xa*(2-xb)
          if(ss.ge.0.d0)then
            zherw=(sqrt(ss)-1)/xa
          else
            zherw=-1.d0
          endif
        endif
c incoming parton (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          zherw=1-(1-x)*(s*(1+y)+4*e0sq*(1-y))/(8*e0sq)
        elseif(1+y.lt.tiny)then
          zherw=x-(1+y)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
        else
          xa=-uk/(2*e0sq)
          xb=-2*tk/s
          ss=1+xa*(2-xb)
          if(ss.ge.0.d0)then
            zherw=(sqrt(ss)-1)/xa
          else
            zherw=-1.d0
          endif
        endif
c outgoing heavy quark
      elseif(ileg.eq.3)then
        if(e0sq.le.(w1+xmq2))then
          zherw=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          zherw=1+(x-1)*(betae0/beta-(betae0-beta)*
     #                  (1-beta*cthg)/(beta*(1-beta**2)))
        else
          beta10=sqrt(1-(w1+xmq2)/e0sq)
          beta2=sqrt(1-4*xmq2*s/(s-w1)**2)
          xkpp=( (s+w1)*w2+(s-w1)*((w1+w2)*beta2-w1) )/
     #         ( (s-w1)*beta2*(s+w1+(s-w1)*beta2) )
          xa=w1/(2*e0sq)
          xb=xkpp*(1+beta10)
          zherw=1-(xa+beta10*xb)/(1+beta10)
        endif
c outgoing heavy antiquark
      elseif(ileg.eq.4)then
        if(e0sq.le.(w2+xmq2))then
          zherw=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          zherw=1+(x-1)*(betae0/beta-(betae0-beta)*
     #                  (1+beta*cthg)/(beta*(1-beta**2)))
        else
          beta20=sqrt(1-(w2+xmq2)/e0sq)
          beta1=sqrt(1-4*xmq2*s/(s-w2)**2)
          xkpp=( (s+w2)*w1+(s-w2)*((w2+w1)*beta1-w2) )/
     #         ( (s-w2)*beta1*(s+w2+(s-w2)*beta1) )
          xa=w2/(2*e0sq)
          xb=xkpp*(1+beta20)
          zherw=1-(xa+beta20*xb)/(1+beta20)
        endif
      else
        write(6,*)'zherw: unknown parton number'
        stop
      endif
      return
      end


      function xiherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #                cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-5)
c
      sth1=sqrt(1-cth1**2)
c incoming parton (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          xiherw=2*s*(1-y)/(s*(1-y)+4*e0sq*(1+y))
        elseif(1-y.lt.tiny)then
          xiherw=(1-y)*s*x**2/(4*e0sq)
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.gt.0.d0)then
            xb=-2*uk/s
            xiherw=2-xb/(1-z)
          else
            xiherw=-1.d0
          endif
        endif
c incoming parton (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          xiherw=2*s*(1+y)/(s*(1+y)+4*e0sq*(1-y))
        elseif(1+y.lt.tiny)then
          xiherw=(1+y)*s*x**2/(4*e0sq)
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.gt.0.d0)then
            xb=-2*tk/s
            xiherw=2-xb/(1-z)
          else
            xiherw=-1.d0
          endif
        endif
c outgoing heavy quark
      elseif(ileg.eq.3)then
        if(e0sq.le.(w1+xmq2))then
          xiherw=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          xiherw=s*(1-beta*cthg)/(4*e0sq*(betae0/beta-
     #           (betae0-beta)*(1-beta*cthg)/(beta*(1-beta**2))))
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          xa=w1/(2*e0sq)
          if(z.gt.0.d0)then
            xiherw=xa/(z*(1-z))
          else
            xiherw=-1.d0
          endif
        endif
c outgoing heavy antquark
      elseif(ileg.eq.4)then
        if(e0sq.le.(w2+xmq2))then
          xiherw=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          xiherw=s*(1+beta*cthg)/(4*e0sq*(betae0/beta-
     #           (betae0-beta)*(1+beta*cthg)/(beta*(1-beta**2))))
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          xa=w2/(2*e0sq)
          if(z.gt.0.d0)then
            xiherw=xa/(z*(1-z))
          else
            xiherw=-1.d0
          endif
        endif
      else
        write(6,*)'xiherw: unknown parton number'
        stop
      endif
      return
      end


      function xjac_xiztoxy(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #                      cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-5)
c
      tmp=0.d0
      sth1=sqrt(1-cth1**2)
c incoming parton (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          tmp=-2*s/(s*(1-y)+4*(1+y)*e0sq)
        elseif(1-y.lt.tiny)then
          tmp=-s*x**2/(4*e0sq)
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.lt.0.d0)then
            xjac_xiztoxy=0.d0
            return
          endif
          xi=xiherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #              cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          tmp=-s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1-y)/(2*e0sq)+
     #             de0sqdc/(2*e0sq) )
        endif
c incoming parton (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          tmp=2*s/(s*(1+y)+4*(1-y)*e0sq)
        elseif(1+y.lt.tiny)then
          tmp=s*x**2/(4*e0sq)
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.lt.0.d0)then
            xjac_xiztoxy=0.d0
            return
          endif
          xi=xiherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #              cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          tmp=s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1+y)/(2*e0sq)-
     #             de0sqdc/(2*e0sq) )
        endif
c outgoing heavy quark
      elseif(ileg.eq.3)then
        if(e0sq.le.(w1+xmq2))then
          xjac_xiztoxy=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          tmp=-s*beta*betae0/(4*e0sq*(betae0-
     #        (1-beta*cthg)*(betae0-beta)/(1-beta**2)))
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.gt.0.d0)then
            betax=sqrt(1-4*xmq2/(s*x))
            beta10=sqrt(1-(w1+xmq2)/e0sq)
            beta2=sqrt(1-4*xmq2*s/(s-w1)**2)
            cpsip=(1+y-x*(1-y))/(1+y+x*(1-y))
            spsip=sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
            cthg=cpsip*cth1+spsip*cth2*sth1
            xkpp=( (s+w1)*w2+(s-w1)*((w1+w2)*beta2-w1) )/
     #           ( (s-w1)*beta2*(s+w1+(s-w1)*beta2) )
            dzde0sq=( w1*(2*beta10*(1+beta10)*e0sq+xmq2+w1) -
     #              2*(1+beta10)**2*e0sq*(xmq2+w1)*xkpp )/
     #              ( 4*beta10*(1+beta10)**2*e0sq**3 )
            dxkppdw1=-4*xmq2*s/(beta2*(s-w1)**3)*
     #    (  (s*(w1-w2)-w1*(w1+w2))*( s+w1+2*beta2*(s-w1) ) -
     #       beta2**2*(s-w1)**2*(w1+w2) )/
     #    (  beta2**2*(s-w1)*(s+w1+beta2*(s-w1))**2  ) +
     #     2*(((1+beta2)*s**2+(1-beta2)*w1**2)*w2-2*xmq2*s*(s+w2))/
     #     ( beta2*(s-w1)**2*(s+w1+beta2*(s-w1))**2 )
            dzdw1=-beta10*dxkppdw1+
     #            ( 2*xmq2+w1-2*e0sq*(1+beta10)+
     #              2*(1+beta10)**2*e0sq*xkpp )/
     #            ( 4*beta10*(1+beta10)**2*e0sq**2 )
            dzdw2=-beta10/(beta2*(s-w1))
            dw1dx=-s/2.d0*(1-betax*cthg)
     #            -cthg*xmq2*(1-x)/(betax*x**2)
            dw2dx=-s/2.d0*(1+betax*cthg)
     #            +cthg*xmq2*(1-x)/(betax*x**2)
            dw1dc=-s/2.d0*betax*(1-x)
            dw2dc=s/2.d0*betax*(1-x)
            dzdx=dzdw1*dw1dx+dzdw2*dw2dx
            dzdc=dzdw1*dw1dc+dzdw2*dw2dc
            xjacden=(4*betax*e0sq**2*x**2*(1-z)*z)
            xjacnum1=-dzdx*e0sq*(1-x)*x**2*s*betax**2+ 
     #        dzdc*e0sq*( betax*s*x**2+cthg*(-s*x**2+2*xmq2*(1+x)) )
            xjacnum2=betax*( -dzde0sq*e0sq*(1-x)*x**2*s*betax +
     #                      2*dzdc*x**2*w1 )
            xjacnum3=-2*dzdx*x**2*betax*w1+dzde0sq*e0sq*
     #               ( betax*s*x**2 + cthg*(-s*x**2+2*xmq2*(1+x)) )
            tmp=(xjacnum1+de0sqdx*xjacnum2+de0sqdc*xjacnum3)/xjacden
            tmp=tmp*4*x/(1+y+x*(1-y))**2
          endif
        endif
c outgoing heavy antiquark
      elseif(ileg.eq.4)then
        if(e0sq.le.(w2+xmq2))then
          xjac_xiztoxy=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=sqrt(1-4*xmq2/s)
          betae0=sqrt(1-xmq2/e0sq)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          tmp=s*beta*betae0/(4*e0sq*(betae0-
     #        (1+beta*cthg)*(betae0-beta)/(1-beta**2)))
        else
          z=zherw(ileg,e0sq,de0sqdx,de0sqdc,xmq2,s,x,y,cth1,
     #            cth2,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          if(z.gt.0.d0)then
            betax=sqrt(1-4*xmq2/(s*x))
            beta20=sqrt(1-(w2+xmq2)/e0sq)
            beta1=sqrt(1-4*xmq2*s/(s-w2)**2)
            cpsip=(1+y-x*(1-y))/(1+y+x*(1-y))
            spsip=sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
            cthg=cpsip*cth1+spsip*cth2*sth1
            xkpp=( (s+w2)*w1+(s-w2)*((w2+w1)*beta1-w2) )/
     #           ( (s-w2)*beta1*(s+w2+(s-w2)*beta1) )
            dzde0sq=( w2*(2*beta20*(1+beta20)*e0sq+xmq2+w2) -
     #              2*(1+beta20)**2*e0sq*(xmq2+w2)*xkpp )/
     #              ( 4*beta20*(1+beta20)**2*e0sq**3 )
            dxkppdw2=-4*xmq2*s/(beta1*(s-w2)**3)*
     #    (  (s*(w2-w1)-w2*(w2+w1))*( s+w2+2*beta1*(s-w2) ) -
     #       beta1**2*(s-w2)**2*(w2+w1) )/
     #    (  beta1**2*(s-w2)*(s+w2+beta1*(s-w2))**2  ) +
     #     2*(((1+beta1)*s**2+(1-beta1)*w2**2)*w1-2*xmq2*s*(s+w1))/
     #     ( beta1*(s-w2)**2*(s+w2+beta1*(s-w2))**2 )
            dzdw2=-beta20*dxkppdw2+
     #            ( 2*xmq2+w2-2*e0sq*(1+beta20)+
     #              2*(1+beta20)**2*e0sq*xkpp )/
     #            ( 4*beta20*(1+beta20)**2*e0sq**2 )
            dzdw1=-beta20/(beta1*(s-w2))
            dw1dx=-s/2.d0*(1-betax*cthg)
     #            -cthg*xmq2*(1-x)/(betax*x**2)
            dw2dx=-s/2.d0*(1+betax*cthg)
     #            +cthg*xmq2*(1-x)/(betax*x**2)
            dw1dc=-s/2.d0*betax*(1-x)
            dw2dc=s/2.d0*betax*(1-x)
            dzdx=dzdw1*dw1dx+dzdw2*dw2dx
            dzdc=dzdw1*dw1dc+dzdw2*dw2dc
            xjacden=-(4*betax*e0sq**2*x**2*(1-z)*z)
            xjacnum1=-dzdx*e0sq*(1-x)*x**2*s*betax**2+ 
     #        dzdc*e0sq*( -betax*s*x**2+cthg*(-s*x**2+2*xmq2*(1+x)) )
            xjacnum2=betax*( -dzde0sq*e0sq*(1-x)*x**2*s*betax -
     #                      2*dzdc*x**2*w2 )
            xjacnum3=2*dzdx*x**2*betax*w2-dzde0sq*e0sq*
     #               ( betax*s*x**2 - cthg*(-s*x**2+2*xmq2*(1+x)) )
            tmp=(xjacnum1+de0sqdx*xjacnum2+de0sqdc*xjacnum3)/xjacden
            tmp=tmp*4*x/(1+y+x*(1-y))**2
          endif
        endif
      else
        write(6,*)'xjac_xiztoxy: unknown parton number'
        stop
      endif
      xjac_xiztoxy=abs(tmp)
      return 
      end 


      subroutine get2to2wr(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #                     tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #                     sbar,tbar,ubar,dsbardx,dtbardx,dubardx,
     #                     dsbardc,dtbardc,dubardc)
c Wrapper for get2to2a1 and get2to2a2
      implicit real*8(a-h,o-z)
      common/cia1ora2/ia1ora2
c
      if(ia1ora2.eq.1)then
        call get2to2a1(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #                 tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #                 sbar,tbar,ubar,dsbardx,dtbardx,dubardx,
     #                 dsbardc,dtbardc,dubardc)
      elseif(ia1ora2.eq.2)then
        call get2to2a2(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #                 tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #                 sbar,tbar,ubar,dsbardx,dtbardx,dubardx,
     #                 dsbardc,dtbardc,dubardc)
      else
        write(6,*)'Fatal error in get2to2wr: unknown option'
        write(6,*)ia1ora2
        stop
      endif
      return
      end


      subroutine get2to2a1(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #                     tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #                     sbar,tbar,ubar,dsbardx,dtbardx,dubardx,
     #                     dsbardc,dtbardc,dubardc)
c Given the 2->3 kinematics, computes the 2->2 invqqiants sbar, tbar,
c and ubar. In the computation of the jacobian, the derivatives of
c these quantities are used as well. d{s,t,u}bardx is the derivative
c with respect to x, at y constant for legs 1 and 2, and at cthg and phig
c constant for legs 3 and 4. d{s,t,u}bardc is the derivative with respect
c to cthg for legs 3 and 4, whereas d{s,t,u}bardc=(1-y**2)*diff({s,t,b}bar,y)
c for legs 1 and 2
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-5)
c
      sth1=sqrt(1-cth1**2)
      beta=sqrt(1-4*xmq2/s)
      betax=sqrt(1-4*xmq2/(s*x))
      if(ileg.eq.1.or.ileg.eq.2)then
        sbar=x*s
        dsbardx=s
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1-beta*cth1)
          dtbardx=-1/2.d0*( s - (s-2*xmq2)*cth1/beta -
     #          s*beta*sqrt(1-y**2)*cth2*sth1*z1/(z1+z2) )
          dtbardc=0.d0
        elseif(1-y.lt.tiny)then
          tbar=-s*x/2.d0*(1-betax*cth1)
          dtbardx=-1/2.d0*( s - (s*x-2*xmq2)*cth1/(x*betax) )
          dtbardc=0.d0
        elseif(1+y.lt.tiny)then
          tbar=-s*x/2.d0*(1-betax*cth1)
          dtbardx=-1/2.d0*( s - (s*x-2*xmq2)*cth1/(x*betax) )
          dtbardc=0.d0
        else
          cpsi=1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
          spsi=4*(1-x)*sqrt(x*(1-y**2))/
     #         ((1+y+x*(1-y))*(1-y+x*(1+y)))
          xpl=((s+uk)*z1/s+(s+tk)*z2/s)/2.d0
          dtkdx=s*(1-y)/2.d0
          dukdx=s*(1+y)/2.d0
          dq1qdx=xmq2*(1+y+x*(1-y))*cth1/(2*betax*x**2) - 
     #            s/4.d0*(1-y)*(1-betax*cth1)
          dq2qdx=-s/4.d0*(1+y)*(1+betax*(cpsi*cth1+
     #                          spsi*cth2*sth1)) -
     #            xmq2*(1-y+x*(1+y))*(cpsi*cth1+
     #            spsi*cth2*sth1)/(2*betax*x**2) +
     #            2*betax*s*(1-x**2)*(1-y**2)*cth1/
     #            ((1+y+x*(1-y))**2*(1-y+x*(1+y))) -
     #            betax*s*cth2*sth1*
     #            (1+x)*(1-y**2)*((1-y**2)*(1+x**2)-2*x*(3-y**2))/
     #            (2*(1+y+x*(1-y))**2*(1-y+x*(1+y))*sqrt(x*(1-y**2)))
          dxpldx=(z1*(1+y)+z2*(1-y))/4.d0
          dtkdy=s*(1-x)/2.d0
          dukdy=-s*(1-x)/2.d0
          dq1qdy=-s/4.d0*(1-x)*(1-betax*cth1)
          dq2qdy=s/4.d0*(1-x)*(1+betax*(cpsi*cth1+
     #                         spsi*cth2*sth1)) +
     #           4*betax*s*(1-x)**2*x*y*cth1/
     #           ((1+y+x*(1-y))**2*(1-y+x*(1+y))) -
     #           betax*s*(1-x)*x*y*((1-y**2)*(1+x**2)-
     #           2*x*(3-y**2))*cth2*sth1/
     #           ( (1+y+x*(1-y))**2*(1-y+x*(1+y))*sqrt(x*(1-y**2)) )
          dxpldy=-(1-x)*(z1-z2)/4.d0
          tbar=-(s+tk+uk)/2.d0*( 1-(z2*(q1q-q1c)+z1*(q2q-q2c))/
     #                    (2*s*sqrt(xpl**2-z1*z2*tk*uk/s**2)) )
          dtbardx=-(s+tk+uk)/2.d0*(
     #      ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )*
     #      ( 2*dxpldx*xpl-dukdx*tk*z1*z2/s**2-dtkdx*uk*z1*z2/s**2 )/
     #      ( 4*s*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) -
     #      ( (2*dq2qdx+dukdx)*z1+(2*dq1qdx+dtkdx)*z2 )/
     #      ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) ) -
     #      (dtkdx+dukdx)/2.d0*(1 -
     #          ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )/
     #          ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) )
          dtbardc=-(s+tk+uk)/2.d0*(
     #      ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )*
     #      ( 2*dxpldy*xpl-dukdy*tk*z1*z2/s**2-dtkdy*uk*z1*z2/s**2 )/
     #      ( 4*s*(xpl**2-tk*uk*z1*z2/s**2)**(1.5d0) ) -
     #      ( (2*dq2qdy+dukdy)*z1+(2*dq1qdy+dtkdy)*z2 )/
     #      ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) ) -
     #      (dtkdy+dukdy)/2.d0*(1 -
     #          ( (2*q2q+s+uk)*z1+(2*q1q+s+tk)*z2 )/
     #          ( 2*s*sqrt(xpl**2-tk*uk*z1*z2/s**2) ) )
          dtbardc=(1-y**2)*dtbardc
        endif
      elseif(ileg.eq.3)then
        sbar=s
        dsbardx=0.d0
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1-beta*cth1)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          dtbardx=s/4.d0*(cthg*cth1-y+beta*(cthg-y*cth1-sth1**2))
          dtbardc=0.d0
        else
          beta2=sqrt(1-4*xmq2*s/(s-w1)**2)
          tbar=-s/2.d0*(1-(q2q-q1c)/(s-w1)*beta/beta2)
          cpsip=(1+y-x*(1-y))/(1+y+x*(1-y))
          spsip=sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
          cthg=cpsip*cth1+spsip*cth2*sth1
          ctho=-(1-x-(1+x)*cth1)/(1+x-(1-x)*cth1)
          dydcpsip=4*x/(1+x-cpsip*(1-x))**2
          dydx=2*(1-cpsip**2)/(1+x-cpsip*(1-x))**2
          dcpsipdx=-sth1*( cthg*sth1 + 
     #                     cth1*(cth1*cth2*spsip-cpsip*sth1) )/
     #             (2.d0*x)
          dcpsipdcthg=cth1+sth1*(cth2*cth1*spsip-cpsip*sth1)*
     #                cthg/(1-cthg**2)
          dcth1dx=-2*(1-ctho**2)/(1+x+(1-x)*ctho)**2
          dtkdx=s*(1-y)/2.d0+
     #          s*(1-x)/2.d0*(dydx+dydcpsip*dcpsipdx)
          dukdx=s*(1+y)/2.d0-
     #          s*(1-x)/2.d0*(dydx+dydcpsip*dcpsipdx)
          dw1dx=-s/2.d0*(1-betax*cthg)-xmq2*(1-x)*cthg/(betax*x**2)
          dq1qdx=-dtkdx/2.d0*(1-betax*cth1)+
     #            s/4.d0*(1+y+x*(1-y))*
     #            (2*xmq2/(s*betax*x**2)*cth1+betax*dcth1dx)
          dtkdc=s/2.d0*(1-x)*dydcpsip*dcpsipdcthg
          dukdc=-s/2.d0*(1-x)*dydcpsip*dcpsipdcthg
          dw1dc=-s/2.d0*(1-x)*betax
          dq1qdc=-dtkdc/2.d0*(1-betax*cth1)
          dtbardx=s/2.d0*( 
     #                  beta*(2*dq1qdx+2*dtkdx+dw1dx)/(beta2*(s-w1)) 
     #                  +4*beta*dw1dx*xmq2*s*(2*q1q+s+2*tk+w1)/
     #                   (beta2**3*(s-w1)**4) 
     #              +beta*dw1dx*(2*q1q+s+2*tk+w1)/(beta2*(s-w1)**2) )
          dtbardc=s/2.d0*( 
     #                  beta*(2*dq1qdc+2*dtkdc+dw1dc)/(beta2*(s-w1)) 
     #                  +4*beta*dw1dc*xmq2*s*(2*q1q+s+2*tk+w1)/
     #                   (beta2**3*(s-w1)**4) 
     #              +beta*dw1dc*(2*q1q+s+2*tk+w1)/(beta2*(s-w1)**2) )
        endif
      elseif(ileg.eq.4)then
        sbar=s
        dsbardx=0.d0
        dsbardc=0.d0
        if(1-x.lt.tiny)then
          tbar=-s/2.d0*(1-beta*cth1)
          cthg=y*cth1+sqrt(1-y**2)*cth2*sth1
          dtbardx=s/4.d0*(-cthg*cth1+y+beta*(cthg-y*cth1-sth1**2))
          dtbardc=0.d0
        else
          beta1=sqrt(1-4*xmq2*s/(s-w2)**2)
          tbar=-s/2.d0*(1-(q1q-q2c)/(s-w2)*beta/beta1)
          cpsip=(1+y-x*(1-y))/(1+y+x*(1-y))
          spsip=sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
          cthg=cpsip*cth1+spsip*cth2*sth1
          ctho=-(1-x-(1+x)*cth1)/(1+x-(1-x)*cth1)
          dydcpsip=4*x/(1+x-cpsip*(1-x))**2
          dydx=2*(1-cpsip**2)/(1+x-cpsip*(1-x))**2
          dcpsipdx=-sth1*( cthg*sth1 + 
     #                     cth1*(cth1*cth2*spsip-cpsip*sth1) )/
     #             (2.d0*x)
          dcpsipdcthg=cth1+sth1*(cth2*cth1*spsip-cpsip*sth1)*
     #                cthg/(1-cthg**2)
          dcth1dx=-2*(1-ctho**2)/(1+x+(1-x)*ctho)**2
          dtkdx=s*(1-y)/2.d0+
     #          s*(1-x)/2.d0*(dydx+dydcpsip*dcpsipdx)
          dukdx=s*(1+y)/2.d0-
     #          s*(1-x)/2.d0*(dydx+dydcpsip*dcpsipdx)
          dw2dx=-s/2.d0*(1+betax*cthg)+xmq2*(1-x)*cthg/(betax*x**2)
          dq1qdx=-dtkdx/2.d0*(1-betax*cth1)+
     #            s/4.d0*(1+y+x*(1-y))*
     #            (2*xmq2/(s*betax*x**2)*cth1+betax*dcth1dx)
          dtkdc=s/2.d0*(1-x)*dydcpsip*dcpsipdcthg
          dukdc=-s/2.d0*(1-x)*dydcpsip*dcpsipdcthg
          dw2dc=s/2.d0*(1-x)*betax
          dq1qdc=-dtkdc/2.d0*(1-betax*cth1)
          dtbardx=s/2.d0*( 
     #                  beta*(2*dq1qdx-dw2dx)/(beta1*(s-w2))
     #                  +4*beta*dw2dx*xmq2*s*(2*q1q+s-w2)/
     #                  (beta1**3*(s-w2)**4) +
     #                  beta*dw2dx*(2*q1q+s-w2)/(beta1*(s-w2)**2) )
          dtbardc=s/2.d0*( 
     #                  beta*(2*dq1qdc-dw2dc)/(beta1*(s-w2))
     #                  +4*beta*dw2dc*xmq2*s*(2*q1q+s-w2)/
     #                  (beta1**3*(s-w2)**4) +
     #                  beta*dw2dc*(2*q1q+s-w2)/(beta1*(s-w2)**2) )
        endif
      else
        write(6,*)'Fatal error in get2to2a1: unknown leg'
        write(6,*)ileg
        stop
      endif
      ubar=-sbar-tbar
      dubardx=-dsbardx-dtbardx
      dubardc=-dsbardc-dtbardc
      return
      end


      subroutine get2to2a2(ileg,z1,z2,xmq2,s,x,y,cth1,cth2,
     #                     tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,
     #                     sbar,tbar,ubar,dsbardx,dtbardx,dubardx,
     #                     dsbardc,dtbardc,dubardc)
c Given the 2->3 kinematics, computes the 2->2 invqqiants sbar, tbar,
c and ubar. In the computation of the jacobian, the derivatives of
c these quantities are used as well. d{s,t,u}bardx is the derivative
c with respect to x, at y constant for legs 1 and 2, and at cthg and phig
c constant for legs 3 and 4. d{s,t,u}bardc is the derivative with respect
c to cthg for legs 3 and 4, whereas d{s,t,u}bardc=(1-y**2)*diff({s,t,b}bar,y)
c for legs 1 and 2
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-5)
c
      sth1=sqrt(1-cth1**2)
      beta=sqrt(1-4*xmq2/s)
      betax=sqrt(1-4*xmq2/(s*x))
      if(ileg.eq.1.or.ileg.eq.2)then
        sbar=x*s
        dsbardx=s
        dsbardc=0.d0
        tbar=-s*x/2.d0*(1-betax*cth1)
        dtbardx=-s/2.d0*(1-betax*cth1)+xmq2*cth1/(betax*x)
        dtbardc=0.d0
        dtbardc=(1-y**2)*dtbardc
      elseif(ileg.eq.3.or.ileg.eq.4)then
        ctho=-(1-x-(1+x)*cth1)/(1+x-(1-x)*cth1)
        sbar=s
        dsbardx=0.d0
        dsbardc=0.d0
        tbar=-s/2.d0*(1-beta*ctho)
        dtbardx=0.d0
        dtbardc=0.d0
      else
        write(6,*)'Fatal error in get2to2a2: unknown leg'
        write(6,*)ileg
        stop
      endif
      ubar=-sbar-tbar
      dubardx=-dsbardx-dtbardx
      dubardc=-dsbardc-dtbardc
      return
      end
c
c
c End of utility routines for xi, z, and 2-->2 invqqiants
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


      function x1x2jqq(xx1,xx2,xx,yy,iileg)
      implicit none
      real*8 x1x2jqq,xx1,xx2,xx,yy,tiny,x1,x2,x,y,tmp,xa,xb
      parameter (tiny=1.d-5)
      integer iileg,ileg,iprespl
      common/ciprespl/iprespl
c
      x1=xx1
      x2=xx2
      x=xx
      y=yy
      ileg=iileg
      if(ileg.eq.1.or.ileg.eq.2)then
        if(iprespl.eq.0)then
          tmp=x
        elseif(iprespl.eq.1)then
          if(1-x.lt.tiny)then
            tmp=x+(1-x)**2*(1-y**2)*x1*x2/(2.d0*(x1+x2)**2)
          elseif(1-y.lt.tiny)then
            tmp=x+(1-x)**2*x*(1-y)*x1*x2/(x*x1+x2)**2
          elseif(1+y.lt.tiny)then
            tmp=x+(1-x)**2*x*(1+y)*x1*x2/(x1+x*x2)**2
          else
            xa=x*x1*x2
            xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
            tmp=x*((1-y+x*(1+y))*x1+(1+y+x*(1-y))*x2)/
     #          (2.d0*sqrt(xb**2+4*xa))
          endif
        else
          write(*,*)'Error # 1 in x1x2jqq',iprespl
          stop
        endif
      elseif(ileg.eq.3.or.ileg.eq.4)then
        tmp=1.d0
      else
        write(*,*)'Error # 2 in x1x2jqq',ileg
        stop
      endif
      x1x2jqq=abs(tmp)
      return
      end
c
c
c End of utility routines for Bjorken x's
c
c
c
c
c Begin of zeta-subtraction routines
c
c
      function bsubint(y)
      implicit none
      real * 8 bsubint,y
      real * 8 s,xm2,cth1,zeta
      integer iproc
c =1 for gg, 2 for qq
      common/kinparms/s,xm2,cth1,zeta,iproc
      real * 8 rp1p2,rk1k2,rp1k1,rp1k2,rp2k1,rp2k2,rk1k1,rk2k2
      real * 8 sth1, rho,b,t,u,m,m1,m2,m12,mmm
      real * 8 tf,ca,cf,da,df,bf
      parameter (tf=0.5d0,ca=3d0,cf=4d0/3d0,da=8d0,df=3d0,bf=5d0/12d0)
c 
      sth1=sqrt(1-cth1**2)
      rho=4*xm2/s
      b=sqrt(1-rho)
c 2.3 in MNR
      t=-s/2*(1-b*cth1)
      u=-s/2*(1+b*cth1)
c Eikonal factor integrated in d th2/(2 pi)
      rp1p2 = -8/(s*(y-1)*(y+1))
      rk1k2 =(4/(s*(b*cth1*y+1)*SQRT(1-b**2*sth1**2*(1-y**2)/(b*cth1*y+1
     1   )**2))-4/(s*(b*cth1*y-1)*SQRT(1-b**2*sth1**2*(1-y**2)/(b*cth1*y
     2   -1)**2)))*(1-2*xm2/s)
      rp1k1 = -4*(b*cth1-1)/(s*(y-1)*(b*cth1*y-1)*SQRT(1-b**2*sth1**2*(1
     1   -y**2)/(b*cth1*y-1)**2))
      rp1k2 = -4*(b*cth1+1)/(s*(y-1)*(b*cth1*y+1)*SQRT(1-b**2*sth1**2*(1
     1   -y**2)/(b*cth1*y+1)**2))
      rp2k1 = -4*(b*cth1+1)/(s*(y+1)*(b*cth1*y-1)*SQRT(1-b**2*sth1**2*(1
     1   -y**2)/(b*cth1*y-1)**2))
      rp2k2 = -4*(b*cth1-1)/(s*(y+1)*(b*cth1*y+1)*SQRT(1-b**2*sth1**2*(1
     1   -y**2)/(b*cth1*y+1)**2))
      rk1k1 = 16*xm2*(1-b**2*sth1**2*(1-y**2)/(b*cth1*y-1)**2)**((-3d0)/
     1   2d0)/(s**2*(b*cth1*y-1)**2)
      rk2k2 = 16*xm2*(1-b**2*sth1**2*(1-y**2)/(b*cth1*y+1)**2)**((-3d0)/
     1   2d0)/(s**2*(b*cth1*y+1)**2)
c
      if(iproc.eq.1) then
c gg case;
c A.13 and A.14 in MNR
      m1=1/(8*s)*(8*t*(t**2+u**2)/(s**2*u)+8*rho*t/u-2*rho**2*s**2/u**2)
      m2=1/(8*s)*(8*u*(t**2+u**2)/(s**2*t)+8*rho*u/t-2*rho**2*s**2/t**2)
      m12=1/(8*s)*(16*(u**2+t**2)/s**2+16*rho-4*s**2*rho**2/(t*u))
c A.12 in MNR
      mmm=m1*tf/da*( (rp1k2+rp2k1)*ca*cf + (rp1k1+rp2k2)*ca*(cf-ca/2)
     # +rp1p2*ca**2/2-(rk2k2+rk1k1)*cf**2+2*rk1k2*(cf-ca/2)**2   )
     # +m2*tf/da*( (rp2k2+rp1k1)*ca*cf + (rp2k1+rp1k2)*ca*(cf-ca/2)
     # + rp1p2*ca**2/2-(rk2k2+rk1k1)*cf**2+2*rk1k2*(cf-ca/2)**2)
     # +m12*tf/da*(cf-ca/2)*( (rp1k2+rp2k2+rp1k1+rp2k1)*ca
     # -(rk2k2+rk1k1)*cf+2*rk1k2*(cf-ca))
      elseif(iproc.eq.2) then
c qq case
c A.22
      m=1/(2*s)*cf*tf/df*(-4*t*u/s**2+rho+2)
c A.21
      mmm=m*(-2*(rp1k2+rp2k1)*(-ca/4+bf)+2*(rp1k1+rp2k2)*(ca/4+bf)
     # -(rk1k1+rk2k2)*cf+2*(rp1p2+rk1k2)*(cf-ca/2))
      else
         write(*,*) ' bsubint: wrong iproc=',iproc
         write(*,*) ' must be 1 (gg) or 2 (qq)'
         stop
      endif
c I should multiply by 4 tk uk /(1-y^2) = s^2
      bsubint=mmm*(log(b**4/zeta)+log(1-y**2))*s**2
c the above expression, integrated in y between -ybar and ybar,
c corresponds to                
c     _
c   / y   /        /      4                  2  \
c  |      |       |  log b / zeta  + log (1-y )  |                 |
c  |  dy  | d th  |  --------------------------- | f  (x,y,th ,th )|
c  | _    |     2 |                2             |  gg/qq    1   2 |x=1
c / -y   /         \          1 - y             /                      
c      
c   up to a factor of: 2 pi g^6
c (2 pi missing from theta 2 integration)
c
      end


      function bsub(as,axm2,acth1,azeta,iiproc)
      implicit none
      real * 8 bsub,as,axm2,acth1,azeta
      integer iiproc
      real * 8 s,xm2,cth1,zeta
      integer iproc
      common/kinparms/s,xm2,cth1,zeta,iproc
      real * 8 yb,b
      real * 8 dgaussfwn,bsubint
      external bsubint
c
      if(iiproc.eq.3)then
        bsub=0.d0
      else
        s=as
        xm2=axm2
        cth1=acth1
        zeta=azeta
        iproc=iiproc
        b=sqrt(1-4*xm2/s)
        if(1-zeta/b**4.lt.0) then
          bsub=0
        else
          yb=sqrt(1-zeta/b**4)
          bsub=dgaussfwn(bsubint,-yb,yb,1.d-6)
        endif
      endif
      end


c This is dgauss; the name has been changed to avoid potential conflicts
      function dgaussfwn(f,a,b,eps)
c.----------------------------------------------------------------------
c.
c.    gauss integral of the function f in interval a,b
c.    last update: 10/04/88
c.
c.----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension w(12),x(12)
      external f
      data const/1.e-12/
      data w
     &/0.101228536290376, 0.222381034453374, 0.313706645877887,
     & 0.362683783378362, 0.027152459411754, 0.062253523938648,
     & 0.095158511682493, 0.124628971255534, 0.149595988816577,
     & 0.169156519395003, 0.182603415044924, 0.189450610455069/
      data x
     &/0.960289856497536, 0.796666477413627, 0.525532409916329,
     & 0.183434642495650, 0.989400934991650, 0.944575023073233,
     & 0.865631202387832, 0.755404408355003, 0.617876244402644,
     & 0.458016777657227, 0.281603550779259, 0.095012509837637/
c--
c--   initialise
      delta=const*abs(a-b)
      dgaussfwn=0.
      aa=a
c--
c--   iteration loop
   10 y=b-aa
c--
c--   epsilon reached ??
      if (abs(y).le.delta) return
   20 bb=aa+y
      c1=0.5*(aa+bb)
      c2=c1-aa
      s8=0.
      s16=0.
      do 30 i=1,4
        u=x(i)*c2
   30 s8=s8+w(i)*(f(c1+u)+f(c1-u))
      do 40 i=5,12
        u=x(i)*c2
   40 s16=s16+w(i)*(f(c1+u)+f(c1-u))
      s8=s8*c2
      s16=s16*c2
      if (abs(s16-s8).gt.eps*(1.0+abs(s16))) goto 50
      dgaussfwn=dgaussfwn+s16
      aa=bb
      goto 10
   50 y=0.5*y
      if (abs(y).le.delta) write(6,9040)
      goto 20
9040  format(1H ,'**** DGAUSS: Too high Accuracy required !!     ****')
      end
c
c
c End of zeta-subtraction routines
c
c
c
c
c Begin of planar amplitudes squared
c
c
C----------------------------------------------------------------------
      SUBROUTINE qqbplanar(s,tk,uk,q1,q2,s2,q1c,q2c,w1,w2,m,t1r,t2r)
C----------------------------------------------------------------------
      IMPLICIT none
      REAL * 8 s,tk,uk,q1,q2,s2,q1c,q2c,w1,w2,m,t1r,t2r
c Planar invqqiant squared amplitudes, summed over spins and color 
c for the planar q qbar -> Q Qbar g process; misses a factor Ncolor^4 
c k1: quark momentum 
c k2: anti-quark momentum 
c p1: incoming light quark momentum 
c p2: incoming light antiquark momentum 
c k: radiated gluon momentum 
c 
c return values: 
c t1r: planar configuration p1,k,k1 
c t2r: planar configuration p2,k,k2 
c
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
c
c    d(p1,p1)=0 
c    d(p2,p2)=0 
c    d(k,k)=0 
c    d(k1,k1)=m2 
c    d(k2,k2)=m2 
c    d(p1,p2)=s/2 
c    d(p1,k)=-tk/2 
c    d(p2,k)=-uk/2 
c    d(p1,k1)=-q1/2 
c    d(p2,k2)=-q2/2 
c    d(k1,k2)=(s2-2*m2)/2 
c    d(p1,k2)=-q1c/2 
c    d(p2,k1)=-q2c/2 
c    d(k,k1)=w1/2 
c    d(k,k2)=w2/2 
c 
c  Relations among invqqiants 
c 
c          s2=s+tk+uk 
c         q1c=-s-tk-q1 
c         q2c=-s-uk-q2 
c         w1=-q1+q2-tk 
c         w2=q1-q2-uk 
      T1R = -(4*S*S2*W1**4+(8*S*S2*TK+(-4*S**2-12*Q2*S)*S2)*W1**3+(8*M**
     1   2*TK**3+((8*S-16*M**2)*S2+16*M**2*S)*TK**2+((8*M**2-4*S)*S2**2+
     2   ((-20*Q2-12*M**2)*S-4*S**2)*S2+12*M**2*S**2)*TK+2*S*S2**3+(4*Q2
     3   +4*M**2)*S*S2**2+(2*S**3+(8*Q2+4*M**2)*S**2+16*Q2**2*S)*S2+4*M*
     4   *2*S**3)*W1**2+(4*S*S2*TK**3+(((8*M**2-12*Q2)*S+16*M**2*Q2)*S2-
     5   4*S*S2**2)*TK**2+(2*S*S2**3+(8*Q2*S-16*M**2*Q2)*S2**2+(2*S**3+(
     6   4*Q2+12*M**2)*S**2+(16*Q2**2+16*M**2*Q2)*S)*S2)*TK-2*Q2*S*S2**3
     7   +(-4*Q2**2-4*M**2*Q2)*S*S2**2+(-2*Q2*S**3+(-4*Q2**2-4*M**2*Q2)*
     8   S**2-8*Q2**3*S)*S2)*W1+(4*M**2*S**2+(8*M**2*Q2+8*M**4)*S+8*M**2
     9   *Q2**2)*S2**2*TK)
     :   *tk*(uk*q1*q2*q1c*q2c*w2)**2
c     :   /(S**2*S2**2*TK*W1**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      T2R = -(4*S*S2*W2**4+(8*S*S2*UK+(-4*S**2-12*Q1*S)*S2)*W2**3+(8*M**
     1   2*UK**3+((8*S-16*M**2)*S2+16*M**2*S)*UK**2+((8*M**2-4*S)*S2**2+
     2   ((-20*Q1-12*M**2)*S-4*S**2)*S2+12*M**2*S**2)*UK+2*S*S2**3+(4*Q1
     3   +4*M**2)*S*S2**2+(2*S**3+(8*Q1+4*M**2)*S**2+16*Q1**2*S)*S2+4*M*
     4   *2*S**3)*W2**2+(4*S*S2*UK**3+(((8*M**2-12*Q1)*S+16*M**2*Q1)*S2-
     5   4*S*S2**2)*UK**2+(2*S*S2**3+(8*Q1*S-16*M**2*Q1)*S2**2+(2*S**3+(
     6   4*Q1+12*M**2)*S**2+(16*Q1**2+16*M**2*Q1)*S)*S2)*UK-2*Q1*S*S2**3
     7   +(-4*Q1**2-4*M**2*Q1)*S*S2**2+(-2*Q1*S**3+(-4*Q1**2-4*M**2*Q1)*
     8   S**2-8*Q1**3*S)*S2)*W2+(4*M**2*S**2+(8*M**2*Q1+8*M**4)*S+8*M**2
     9   *Q1**2)*S2**2*UK)
     :   *uk*(tk*q1*q2*q1c*q2c*w1)**2
c     :   /(S**2*S2**2*UK*W2**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      END
C----------------------------------------------------------------------
      SUBROUTINE ggplanar(s,tk,uk,q1,q2,s2,q1c,q2c,w1,w2,m,tr12,t1r2,t12
     1   r,tr21,t2r1,t21r)
      IMPLICIT none
      REAL * 8 s,tk,uk,q1,q2,s2,q1c,q2c,w1,w2,m,tr12,t1r2,t12r,tr21,t2r1
     1   ,t21r
c Planar invqqiant squared amplitudes, summed over spins and color   
c for the planar g g -> Q Qbar g process; misses a factor Ncolor^4   
c k1: quark moment   
c k2: anti-quark moment   
c p1: incoming gluon moment   
c p2: incoming gluon moment   
c k: radiated gluon moment   
c   
c Planar configurations are obtained with the gluon all on the same
c side of the fermion line; a given colour flow is specified by
c the ordering of the attachment of the gluons on the fermion
c line. For example: k1,k,p1,p2,k2 is the color structure
c                             
c    k1 -<----||--<--  ---<---  ---<--- k2
c             ||     ||       ||       
c             ^V     ^V       ^V       
c             ||     ||       ||       
c             ||     ||       ||       
c             k      p1       p2
c
c
c return values:   
c tr12: planar configuration k1,k,p1,p2,k2   
c t1r2:                      k1,p1,k,p2,k2   
c t12r:                      k1,p1,p2,k,k2   
c tr21:                      k1,k,p2,p1,k2   
c t2r1:                      k1,p2,k,p1,k2   
c t21r:                      k1,p2,p1,k,k2   
c txyz: xyz stand for incoming gluon 1, 2, and radiated gluon (r)   
c
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
c   
c    d(p1,p1)=0   
c    d(p2,p2)=0   
c    d(k,k)=0   
c    d(k1,k1)=m2   
c    d(k2,k2)=m2   
c    d(p1,p2)=s/2   
c    d(p1,k)=-tk/2   
c    d(p2,k)=-uk/2   
c    d(p1,k1)=-q1/2   
c    d(p2,k2)=-q2/2   
c    d(k1,k2)=(s2-2*m2)/2   
c    d(p1,k2)=-q1c/2   
c    d(p2,k1)=-q2c/2   
c    d(k,k1)=w1/2   
c    d(k,k2)=w2/2   
c   
c  Relations among invqqiants   
c   
c          s2=s+tk+uk   

c         q1c=-s-tk-q1   
c         q2c=-s-uk-q2   
c         w1=-q1+q2-tk   
c         w2=q1-q2-uk   
      TR12 = -(8*Q2*S*S2*TK*W1**5+(8*Q2*S*S2*TK**2+(8*Q2*S*S2**2+(-16*Q2
     1   *S**2-16*Q2**2*S)*S2)*TK-8*M**2*S**2*S2**2)*W1**4+(6*Q2*S*S2*TK
     2   **3+((8*M**2*Q2-24*Q2**2)*S-12*Q2*S**2)*S2*TK**2+(6*Q2*S*S2**3+
     3   ((-12*Q2-16*M**2)*S**2+8*M**2*Q2*S)*S2**2+(12*Q2*S**3+(24*Q2**2
     4   -16*M**2*Q2)*S**2+24*Q2**3*S)*S2)*TK+16*M**2*Q2*S**2*S2**2-16*M
     5   **2*Q2*S**3*S2)*W1**3+((2*Q2*S*S2-8*M**2*Q2**2)*TK**4+((-6*Q2*S
     6   **2+(8*M**2*Q2-12*Q2**2)*S+16*M**2*Q2**2)*S2-16*M**2*Q2**2*S)*T
     7   K**3+((-12*M**2*S**2-8*M**2*Q2**2)*S2**2+(6*Q2*S**3+(24*Q2**2-1
     8   6*M**2*Q2)*S**2+(24*Q2**3-8*M**2*Q2**2)*S)*S2-24*M**2*Q2**2*S**
     9   2)*TK**2+(2*Q2*S*S2**4+(8*M**2*Q2*S-6*Q2*S**2)*S2**3+(6*Q2*S**3
     :   +(-8*M**2*Q2-16*M**4)*S**2)*S2**2+(-4*Q2*S**4+(-12*Q2**2-16*M**
     ;   2*Q2)*S**3+(-24*Q2**3-8*M**2*Q2**2)*S**2-16*Q2**4*S)*S2-16*M**2
     <   *Q2**2*S**3)*TK-8*M**2*Q2**2*S**2*S2**2+16*M**2*Q2**2*S**3*S2-8
     =   *M**2*Q2**2*S**4)*W1**2+((-2*Q2*S**2-4*Q2**2*S)*S2*TK**4+((6*Q2
     >   **2*S-4*M**2*S**2)*S2**2+((6*Q2**2-4*M**2*Q2)*S**2+(12*Q2**3-16
     ?   *M**2*Q2**2+16*M**4*Q2)*S-16*M**2*Q2**3)*S2)*TK**3+(-6*Q2**2*S*
     @   S2**3+((-12*M**2*Q2-16*M**4)*S**2+(-12*Q2**3-8*M**2*Q2**2-16*M*
     1   *4*Q2)*S+16*M**2*Q2**3)*S2**2+(-2*Q2*S**4+(-6*Q2**2-4*M**2*Q2)*
     2   S**3+(-12*Q2**3-16*M**2*Q2**2+16*M**4*Q2)*S**2+(-16*Q2**4-16*M*
     3   *2*Q2**3)*S)*S2)*TK**2+(2*Q2**2*S*S2**4+(6*Q2**3+8*M**2*Q2**2+1
     4   6*M**4*Q2)*S*S2**3+((8*Q2**4+8*M**2*Q2**3)*S-16*M**4*Q2*S**2)*S
     5   2**2+(2*Q2**2*S**4+(6*Q2**3+8*M**2*Q2**2+16*M**4*Q2)*S**3+(8*Q2
     6   **4+8*M**2*Q2**3)*S**2+8*Q2**5*S)*S2)*TK)*W1+(-4*M**2*Q2*S**3+(
     7   -12*M**2*Q2**2-16*M**4*Q2-16*M**6)*S**2+(-16*M**2*Q2**3-16*M**4
     8   *Q2**2)*S-8*M**2*Q2**4)*S2**2*TK**2)
     9   *(uk*q1*q1c*q2c*w2)**2
c     9   /(Q2**2*S**2*S2**2*TK**2*W1**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      T1R2 = ((2*Q1*Q2*S2*TK**2+(4*Q1**2*Q2-2*Q1*Q2**2)*S2*TK+8*M**2*Q1*
     1   *2*Q2**2)*UK**4+((4*M**2*Q2*S2**2+(6*Q1*Q2**2+(4*M**2*Q1-6*Q1**
     2   2)*Q2)*S2)*TK**2+(-6*Q1**2*Q2*S2**2+(-6*Q1*Q2**3+(12*Q1**2-8*M*
     3   *2*Q1)*Q2**2+(-12*Q1**3+16*M**2*Q1**2-16*M**4*Q1)*Q2)*S2+16*M**
     4   2*Q1**2*Q2**2)*TK+(16*M**2*Q1**3*Q2-16*M**2*Q1**2*Q2**2)*S2)*UK
     5   **3+(2*Q1*Q2*S2*TK**4+(4*M**2*Q1*S2**2+((6*Q1**2+4*M**2*Q1)*Q2-
     6   6*Q1*Q2**2)*S2)*TK**3+((12*M**2*Q2**2+(12*M**2*Q1+16*M**4)*Q2+1
     7   2*M**2*Q1**2+16*M**4*Q1+16*M**6)*S2**2+(12*Q1*Q2**3+(16*M**2*Q1
     8   -24*Q1**2)*Q2**2+(12*Q1**3+16*M**2*Q1**2-16*M**4*Q1)*Q2)*S2+24*
     9   M**2*Q1**2*Q2**2)*TK**2+(6*Q1**2*Q2*S2**3+((12*Q1**3+8*M**2*Q1*
     :   *2+16*M**4*Q1)*Q2+16*M**2*Q1**3+16*M**4*Q1**2)*S2**2+(-8*Q1*Q2*
     ;   *4+(24*Q1**2-8*M**2*Q1)*Q2**3+(8*M**2*Q1**2-24*Q1**3)*Q2**2+(16
     <   *Q1**4+16*M**2*Q1**3)*Q2)*S2)*TK+(8*M**2*Q1**2*Q2**2-16*M**2*Q1
     =   **3*Q2+8*M**2*Q1**4)*S2**2)*UK**2+((4*Q1*Q2**2-2*Q1**2*Q2)*S2*T
     >   K**4+(-6*Q1*Q2**2*S2**2+(-12*Q1*Q2**3+(12*Q1**2+16*M**2*Q1)*Q2*
     ?   *2+(-6*Q1**3-8*M**2*Q1**2-16*M**4*Q1)*Q2)*S2+16*M**2*Q1**2*Q2**
     @   2)*TK**3+(6*Q1*Q2**2*S2**3+((12*Q1+16*M**2)*Q2**3+(8*M**2*Q1+16
     1   *M**4)*Q2**2+16*M**4*Q1*Q2)*S2**2+(16*Q1*Q2**4+(16*M**2*Q1-24*Q
     2   1**2)*Q2**3+(24*Q1**3+8*M**2*Q1**2)*Q2**2+(-8*Q1**4-8*M**2*Q1**
     3   3)*Q2)*S2)*TK**2+((-2*Q1*Q2**2-2*Q1**2*Q2)*S2**4+(-6*Q1*Q2**3-8
     4   *M**2*Q1*Q2**2+(-6*Q1**3-8*M**2*Q1**2-16*M**4*Q1)*Q2)*S2**3+(-8
     5   *Q1*Q2**4-8*M**2*Q1*Q2**3+(-8*Q1**4-8*M**2*Q1**3)*Q2)*S2**2+(-8
     6   *Q1*Q2**5+16*Q1**2*Q2**4-24*Q1**3*Q2**3+16*Q1**4*Q2**2-8*Q1**5*
     7   Q2)*S2)*TK)*UK+8*M**2*Q1**2*Q2**2*TK**4+(16*M**2*Q1*Q2**3-16*M*
     8   *2*Q1**2*Q2**2)*S2*TK**3+(8*M**2*Q2**4-16*M**2*Q1*Q2**3+8*M**2*
     9   Q1**2*Q2**2)*S2**2*TK**2)
     :   *(s*q1c*q2c*w1*w2)**2
c     :   /(Q1**2*Q2**2*S2**2*TK**2*UK**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      T12R = -(8*Q1*S*S2*UK*W2**5+(8*Q1*S*S2*UK**2+(8*Q1*S*S2**2+(-16*Q1
     1   *S**2-16*Q1**2*S)*S2)*UK-8*M**2*S**2*S2**2)*W2**4+(6*Q1*S*S2*UK
     2   **3+((8*M**2*Q1-24*Q1**2)*S-12*Q1*S**2)*S2*UK**2+(6*Q1*S*S2**3+
     3   ((-12*Q1-16*M**2)*S**2+8*M**2*Q1*S)*S2**2+(12*Q1*S**3+(24*Q1**2
     4   -16*M**2*Q1)*S**2+24*Q1**3*S)*S2)*UK+16*M**2*Q1*S**2*S2**2-16*M
     5   **2*Q1*S**3*S2)*W2**3+((2*Q1*S*S2-8*M**2*Q1**2)*UK**4+((-6*Q1*S
     6   **2+(8*M**2*Q1-12*Q1**2)*S+16*M**2*Q1**2)*S2-16*M**2*Q1**2*S)*U
     7   K**3+((-12*M**2*S**2-8*M**2*Q1**2)*S2**2+(6*Q1*S**3+(24*Q1**2-1
     8   6*M**2*Q1)*S**2+(24*Q1**3-8*M**2*Q1**2)*S)*S2-24*M**2*Q1**2*S**
     9   2)*UK**2+(2*Q1*S*S2**4+(8*M**2*Q1*S-6*Q1*S**2)*S2**3+(6*Q1*S**3
     :   +(-8*M**2*Q1-16*M**4)*S**2)*S2**2+(-4*Q1*S**4+(-12*Q1**2-16*M**
     ;   2*Q1)*S**3+(-24*Q1**3-8*M**2*Q1**2)*S**2-16*Q1**4*S)*S2-16*M**2
     <   *Q1**2*S**3)*UK-8*M**2*Q1**2*S**2*S2**2+16*M**2*Q1**2*S**3*S2-8
     =   *M**2*Q1**2*S**4)*W2**2+((-2*Q1*S**2-4*Q1**2*S)*S2*UK**4+((6*Q1
     >   **2*S-4*M**2*S**2)*S2**2+((6*Q1**2-4*M**2*Q1)*S**2+(12*Q1**3-16
     ?   *M**2*Q1**2+16*M**4*Q1)*S-16*M**2*Q1**3)*S2)*UK**3+(-6*Q1**2*S*
     @   S2**3+((-12*M**2*Q1-16*M**4)*S**2+(-12*Q1**3-8*M**2*Q1**2-16*M*
     1   *4*Q1)*S+16*M**2*Q1**3)*S2**2+(-2*Q1*S**4+(-6*Q1**2-4*M**2*Q1)*
     2   S**3+(-12*Q1**3-16*M**2*Q1**2+16*M**4*Q1)*S**2+(-16*Q1**4-16*M*
     3   *2*Q1**3)*S)*S2)*UK**2+(2*Q1**2*S*S2**4+(6*Q1**3+8*M**2*Q1**2+1
     4   6*M**4*Q1)*S*S2**3+((8*Q1**4+8*M**2*Q1**3)*S-16*M**4*Q1*S**2)*S
     5   2**2+(2*Q1**2*S**4+(6*Q1**3+8*M**2*Q1**2+16*M**4*Q1)*S**3+(8*Q1
     6   **4+8*M**2*Q1**3)*S**2+8*Q1**5*S)*S2)*UK)*W2+(-4*M**2*Q1*S**3+(
     7   -12*M**2*Q1**2-16*M**4*Q1-16*M**6)*S**2+(-16*M**2*Q1**3-16*M**4
     8   *Q1**2)*S-8*M**2*Q1**4)*S2**2*UK**2)
     9   *(tk*q2*q1c*q2c*w1)**2
c     9   /(Q1**2*S**2*S2**2*UK**2*W2**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      TR21 = -(8*Q1C*S*S2*UK*W1**5+(8*Q1C*S*S2*UK**2+(8*Q1C*S*S2**2+(-16
     1   *Q1C*S**2-16*Q1C**2*S)*S2)*UK-8*M**2*S**2*S2**2)*W1**4+(6*Q1C*S
     2   *S2*UK**3+((8*M**2*Q1C-24*Q1C**2)*S-12*Q1C*S**2)*S2*UK**2+(6*Q1
     3   C*S*S2**3+((-12*Q1C-16*M**2)*S**2+8*M**2*Q1C*S)*S2**2+(12*Q1C*S
     4   **3+(24*Q1C**2-16*M**2*Q1C)*S**2+24*Q1C**3*S)*S2)*UK+16*M**2*Q1
     5   C*S**2*S2**2-16*M**2*Q1C*S**3*S2)*W1**3+((2*Q1C*S*S2-8*M**2*Q1C
     6   **2)*UK**4+((-6*Q1C*S**2+(8*M**2*Q1C-12*Q1C**2)*S+16*M**2*Q1C**
     7   2)*S2-16*M**2*Q1C**2*S)*UK**3+((-12*M**2*S**2-8*M**2*Q1C**2)*S2
     8   **2+(6*Q1C*S**3+(24*Q1C**2-16*M**2*Q1C)*S**2+(24*Q1C**3-8*M**2*
     9   Q1C**2)*S)*S2-24*M**2*Q1C**2*S**2)*UK**2+(2*Q1C*S*S2**4+(8*M**2
     :   *Q1C*S-6*Q1C*S**2)*S2**3+(6*Q1C*S**3+(-8*M**2*Q1C-16*M**4)*S**2
     ;   )*S2**2+(-4*Q1C*S**4+(-12*Q1C**2-16*M**2*Q1C)*S**3+(-24*Q1C**3-
     <   8*M**2*Q1C**2)*S**2-16*Q1C**4*S)*S2-16*M**2*Q1C**2*S**3)*UK-8*M
     =   **2*Q1C**2*S**2*S2**2+16*M**2*Q1C**2*S**3*S2-8*M**2*Q1C**2*S**4
     >   )*W1**2+((-2*Q1C*S**2-4*Q1C**2*S)*S2*UK**4+((6*Q1C**2*S-4*M**2*
     ?   S**2)*S2**2+((6*Q1C**2-4*M**2*Q1C)*S**2+(12*Q1C**3-16*M**2*Q1C*
     @   *2+16*M**4*Q1C)*S-16*M**2*Q1C**3)*S2)*UK**3+(-6*Q1C**2*S*S2**3+
     1   ((-12*M**2*Q1C-16*M**4)*S**2+(-12*Q1C**3-8*M**2*Q1C**2-16*M**4*
     2   Q1C)*S+16*M**2*Q1C**3)*S2**2+(-2*Q1C*S**4+(-6*Q1C**2-4*M**2*Q1C
     3   )*S**3+(-12*Q1C**3-16*M**2*Q1C**2+16*M**4*Q1C)*S**2+(-16*Q1C**4
     4   -16*M**2*Q1C**3)*S)*S2)*UK**2+(2*Q1C**2*S*S2**4+(6*Q1C**3+8*M**
     5   2*Q1C**2+16*M**4*Q1C)*S*S2**3+((8*Q1C**4+8*M**2*Q1C**3)*S-16*M*
     6   *4*Q1C*S**2)*S2**2+(2*Q1C**2*S**4+(6*Q1C**3+8*M**2*Q1C**2+16*M*
     7   *4*Q1C)*S**3+(8*Q1C**4+8*M**2*Q1C**3)*S**2+8*Q1C**5*S)*S2)*UK)*
     8   W1+(-4*M**2*Q1C*S**3+(-12*M**2*Q1C**2-16*M**4*Q1C-16*M**6)*S**2
     9   +(-16*M**2*Q1C**3-16*M**4*Q1C**2)*S-8*M**2*Q1C**4)*S2**2*UK**2)
     :   *(tk*q1*q2*q2c*w2)**2
c     :   /(Q1C**2*S**2*S2**2*UK**2*W1**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      T2R1 = ((2*Q1C*Q2C*S2*TK**2+(4*Q1C**2*Q2C-2*Q1C*Q2C**2)*S2*TK+8*M*
     1   *2*Q1C**2*Q2C**2)*UK**4+((4*M**2*Q2C*S2**2+(6*Q1C*Q2C**2+(4*M**
     2   2*Q1C-6*Q1C**2)*Q2C)*S2)*TK**2+(-6*Q1C**2*Q2C*S2**2+(-6*Q1C*Q2C
     3   **3+(12*Q1C**2-8*M**2*Q1C)*Q2C**2+(-12*Q1C**3+16*M**2*Q1C**2-16
     4   *M**4*Q1C)*Q2C)*S2+16*M**2*Q1C**2*Q2C**2)*TK+(16*M**2*Q1C**3*Q2
     5   C-16*M**2*Q1C**2*Q2C**2)*S2)*UK**3+(2*Q1C*Q2C*S2*TK**4+(4*M**2*
     6   Q1C*S2**2+((6*Q1C**2+4*M**2*Q1C)*Q2C-6*Q1C*Q2C**2)*S2)*TK**3+((
     7   12*M**2*Q2C**2+(12*M**2*Q1C+16*M**4)*Q2C+12*M**2*Q1C**2+16*M**4
     8   *Q1C+16*M**6)*S2**2+(12*Q1C*Q2C**3+(16*M**2*Q1C-24*Q1C**2)*Q2C*
     9   *2+(12*Q1C**3+16*M**2*Q1C**2-16*M**4*Q1C)*Q2C)*S2+24*M**2*Q1C**
     :   2*Q2C**2)*TK**2+(6*Q1C**2*Q2C*S2**3+((12*Q1C**3+8*M**2*Q1C**2+1
     ;   6*M**4*Q1C)*Q2C+16*M**2*Q1C**3+16*M**4*Q1C**2)*S2**2+(-8*Q1C*Q2
     <   C**4+(24*Q1C**2-8*M**2*Q1C)*Q2C**3+(8*M**2*Q1C**2-24*Q1C**3)*Q2
     =   C**2+(16*Q1C**4+16*M**2*Q1C**3)*Q2C)*S2)*TK+(8*M**2*Q1C**2*Q2C*
     >   *2-16*M**2*Q1C**3*Q2C+8*M**2*Q1C**4)*S2**2)*UK**2+((4*Q1C*Q2C**
     ?   2-2*Q1C**2*Q2C)*S2*TK**4+(-6*Q1C*Q2C**2*S2**2+(-12*Q1C*Q2C**3+(
     @   12*Q1C**2+16*M**2*Q1C)*Q2C**2+(-6*Q1C**3-8*M**2*Q1C**2-16*M**4*
     1   Q1C)*Q2C)*S2+16*M**2*Q1C**2*Q2C**2)*TK**3+(6*Q1C*Q2C**2*S2**3+(
     2   (12*Q1C+16*M**2)*Q2C**3+(8*M**2*Q1C+16*M**4)*Q2C**2+16*M**4*Q1C
     3   *Q2C)*S2**2+(16*Q1C*Q2C**4+(16*M**2*Q1C-24*Q1C**2)*Q2C**3+(24*Q
     4   1C**3+8*M**2*Q1C**2)*Q2C**2+(-8*Q1C**4-8*M**2*Q1C**3)*Q2C)*S2)*
     5   TK**2+((-2*Q1C*Q2C**2-2*Q1C**2*Q2C)*S2**4+(-6*Q1C*Q2C**3-8*M**2
     6   *Q1C*Q2C**2+(-6*Q1C**3-8*M**2*Q1C**2-16*M**4*Q1C)*Q2C)*S2**3+(-
     7   8*Q1C*Q2C**4-8*M**2*Q1C*Q2C**3+(-8*Q1C**4-8*M**2*Q1C**3)*Q2C)*S
     8   2**2+(-8*Q1C*Q2C**5+16*Q1C**2*Q2C**4-24*Q1C**3*Q2C**3+16*Q1C**4
     9   *Q2C**2-8*Q1C**5*Q2C)*S2)*TK)*UK+8*M**2*Q1C**2*Q2C**2*TK**4+(16
     :   *M**2*Q1C*Q2C**3-16*M**2*Q1C**2*Q2C**2)*S2*TK**3+(8*M**2*Q2C**4
     ;   -16*M**2*Q1C*Q2C**3+8*M**2*Q1C**2*Q2C**2)*S2**2*TK**2)
     <    *(s*q1*q2*w1*w2)**2
c     <   /(Q1C**2*Q2C**2*S2**2*TK**2*UK**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      T21R = -(8*Q2C*S*S2*TK*W2**5+(8*Q2C*S*S2*TK**2+(8*Q2C*S*S2**2+(-16
     1   *Q2C*S**2-16*Q2C**2*S)*S2)*TK-8*M**2*S**2*S2**2)*W2**4+(6*Q2C*S
     2   *S2*TK**3+((8*M**2*Q2C-24*Q2C**2)*S-12*Q2C*S**2)*S2*TK**2+(6*Q2
     3   C*S*S2**3+((-12*Q2C-16*M**2)*S**2+8*M**2*Q2C*S)*S2**2+(12*Q2C*S
     4   **3+(24*Q2C**2-16*M**2*Q2C)*S**2+24*Q2C**3*S)*S2)*TK+16*M**2*Q2
     5   C*S**2*S2**2-16*M**2*Q2C*S**3*S2)*W2**3+((2*Q2C*S*S2-8*M**2*Q2C
     6   **2)*TK**4+((-6*Q2C*S**2+(8*M**2*Q2C-12*Q2C**2)*S+16*M**2*Q2C**
     7   2)*S2-16*M**2*Q2C**2*S)*TK**3+((-12*M**2*S**2-8*M**2*Q2C**2)*S2
     8   **2+(6*Q2C*S**3+(24*Q2C**2-16*M**2*Q2C)*S**2+(24*Q2C**3-8*M**2*
     9   Q2C**2)*S)*S2-24*M**2*Q2C**2*S**2)*TK**2+(2*Q2C*S*S2**4+(8*M**2
     :   *Q2C*S-6*Q2C*S**2)*S2**3+(6*Q2C*S**3+(-8*M**2*Q2C-16*M**4)*S**2
     ;   )*S2**2+(-4*Q2C*S**4+(-12*Q2C**2-16*M**2*Q2C)*S**3+(-24*Q2C**3-
     <   8*M**2*Q2C**2)*S**2-16*Q2C**4*S)*S2-16*M**2*Q2C**2*S**3)*TK-8*M
     =   **2*Q2C**2*S**2*S2**2+16*M**2*Q2C**2*S**3*S2-8*M**2*Q2C**2*S**4
     >   )*W2**2+((-2*Q2C*S**2-4*Q2C**2*S)*S2*TK**4+((6*Q2C**2*S-4*M**2*
     ?   S**2)*S2**2+((6*Q2C**2-4*M**2*Q2C)*S**2+(12*Q2C**3-16*M**2*Q2C*
     @   *2+16*M**4*Q2C)*S-16*M**2*Q2C**3)*S2)*TK**3+(-6*Q2C**2*S*S2**3+
     1   ((-12*M**2*Q2C-16*M**4)*S**2+(-12*Q2C**3-8*M**2*Q2C**2-16*M**4*
     2   Q2C)*S+16*M**2*Q2C**3)*S2**2+(-2*Q2C*S**4+(-6*Q2C**2-4*M**2*Q2C
     3   )*S**3+(-12*Q2C**3-16*M**2*Q2C**2+16*M**4*Q2C)*S**2+(-16*Q2C**4
     4   -16*M**2*Q2C**3)*S)*S2)*TK**2+(2*Q2C**2*S*S2**4+(6*Q2C**3+8*M**
     5   2*Q2C**2+16*M**4*Q2C)*S*S2**3+((8*Q2C**4+8*M**2*Q2C**3)*S-16*M*
     6   *4*Q2C*S**2)*S2**2+(2*Q2C**2*S**4+(6*Q2C**3+8*M**2*Q2C**2+16*M*
     7   *4*Q2C)*S**3+(8*Q2C**4+8*M**2*Q2C**3)*S**2+8*Q2C**5*S)*S2)*TK)*
     8   W2+(-4*M**2*Q2C*S**3+(-12*M**2*Q2C**2-16*M**4*Q2C-16*M**6)*S**2
     9   +(-16*M**2*Q2C**3-16*M**4*Q2C**2)*S-8*M**2*Q2C**4)*S2**2*TK**2)
     :   *(uk*q1*q2*q1c*w1)**2
c     :   /(Q2C**2*S**2*S2**2*TK**2*W2**2)
C--N.B. new definitions Ti-->Ti*(s*tk*uk*q1*q2*s2*q1c*q2c*w1*w2)**2
      END
c
c
c End of planar amplitudes squared
c
c

      function zgmu2_nqq()
c Sets the desired factorization scale and returns the strong coupling squared
c To be called is association to pure NLO terms
      implicit none
      real * 8 zgmu2_nqq
      real * 8 pi
      parameter (pi=3.14159265358979312D0)
      real * 8 pq10,pq20,pp0
      common/perpen/pq10(2),pq20(2),pp0(2)
      integer nlas
      common/cnlas/nlas
      include 'hvqcblks.h'
      real * 8 pt12,pt22,pt2,xmu2,as
      real * 8 alfas
c
      pt12= pq10(1)**2 + pq10(2)**2
      pt22= pq20(1)**2 + pq20(2)**2
      pt2 = (pt12+pt22)/2.d0
      xmu2 = pt2 + xm2
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nlas)
      zgmu2_nqq = 4.d0*pi*as
      zg = sqrt(zgmu2_nqq)
      end


      function zgmu2_qq(inlomc,sq,tq,uq,sqb,tqb,uqb)
c Sets the desired factorization scale and returns the strong coupling squared
c To be called is association to MC subtraction terms
      implicit none
      integer inlomc
      real * 8 zgmu2_qq,sq,tq,uq,sqb,tqb,uqb,
     # pt12,pt22,pt2,xmu2,as,alfas,pi
      real*8 pq10,pq20,pp0
      common/perpen/pq10(2),pq20(2),pp0(2)
      parameter (pi=3.14159265358979312D0)
      integer nlas
      common/cnlas/nlas
      include 'hvqcblks.h'
c
      if(inlomc.eq.1)then
        pt12= pq10(1)**2 + pq10(2)**2
        pt22= pq20(1)**2 + pq20(2)**2
      elseif(inlomc.eq.2)then
        pt12=tq*uq/sq-xm2
        pt22=tqb*uqb/sqb-xm2
      else
        write(6,*)'Fatal error in zgmu2_qq: unknown option'
        write(6,*)inlomc
        stop
      endif
      pt2 = (pt12+pt22)/2.d0
      xmu2 = pt2 + xm2
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nlas)
      zgmu2_qq = 4.d0*pi*as
      zg = sqrt(zgmu2_qq)
      end


      function zgmu6_mc(ileg,xm2,s,x,y,cth1,cth2,sq,tq,uq,sqb,tqb,uqb)
c Computes the coupling constant for MC subtraction terms
      implicit none
      character*2 str
      parameter (str='p1')
      real * 8 zgmu6_mc,xm2,s,x,y,cth1,cth2,sq,tq,uq,sqb,tqb,uqb,
     # tmp,zgmu2_qq,zg2nlo,zg2mc,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,
     # w2h,xtmp,shat
      integer ileg,ifuntype,iasmc,iassoft,ione,itwo
      parameter (ione=1)
      parameter (itwo=2)
      common/cifuntype/ifuntype
      common/ciasmc/iasmc
      common/ciassoft/iassoft
c
      if(iasmc.eq.1.or.iasmc.eq.3)then
        if(ifuntype.eq.1.or.(ifuntype.eq.2.and.ileg.le.2))then
          if(iassoft.eq.0)then
            shat=s
            xtmp=x
          elseif(iassoft.eq.1)then
            shat=s*x
            xtmp=1.d0
          else
            write(*,*)'Fatal error #1 in zgmu6_mc: unknown iassoft'
            write(*,*)iassoft
            stop
          endif
        elseif(ifuntype.eq.2.and.ileg.gt.2)then
          if(iassoft.eq.0)then
            shat=s*x
            xtmp=x
          elseif(iassoft.eq.1)then
            shat=s*x**2
            xtmp=1.d0
          else
            write(*,*)'Fatal error #2 in zgmu6_mc: unknown iassoft'
            write(*,*)iassoft
            stop
          endif
        else
          write(*,*)'Fatal error #3 in zgmu6_mc: unknown case'
          write(*,*)ifuntype,ileg
          stop
        endif
        call invqq(xm2,shat,xtmp,y,cth1,cth2,str,
     #    tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
      endif
      if(iasmc.eq.1)then
        zg2nlo=zgmu2_qq(ione,sq,tq,uq,sqb,tqb,uqb)
        tmp=zg2nlo**3
      elseif(iasmc.eq.2)then
        zg2mc=zgmu2_qq(itwo,sq,tq,uq,sqb,tqb,uqb)
        tmp=zg2mc**3
      elseif(iasmc.eq.3)then
        zg2nlo=zgmu2_qq(ione,sq,tq,uq,sqb,tqb,uqb)
        zg2mc=zgmu2_qq(itwo,sq,tq,uq,sqb,tqb,uqb)
        tmp=zg2mc*zg2nlo**2
      else
        write(*,*)'Fatal error #4 in zgmu6_mc: unknown iasmc',iasmc
        stop
      endif
      zgmu6_mc=tmp
      return
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


      subroutine HVQWARN(str)
      character *(*) str
      write(*,*) '********** WARNING **********'
      write(*,*) '*********  ',str,'  *********'
      end


