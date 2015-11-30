c.................... linana.f
c=======================================================================
c     10/26/01 update freezeout positions in case of interactions:
clin-3/2009 Note: freezeout spacetime values cannot be trusted for K0S & K0L 
c     as K0S/K0L are converted from K+/K- by hand at the end of hadron cascade.
      subroutine hbtout(nnew,nt,ntmax)
c
      PARAMETER  (MAXSTR=150001,MAXR=1)
clin-5/2008 give tolerance to regular particles (perturbative probability 1):
      PARAMETER  (oneminus=0.99999,oneplus=1.00001)
      dimension lastkp(MAXSTR), newkp(MAXSTR),xnew(3)
      common /para7/ ioscar,nsmbbbar,nsmmeson
cc      SAVE /para7/
      COMMON/hbt/lblast(MAXSTR),xlast(4,MAXSTR),plast(4,MAXSTR),nlast
cc      SAVE /hbt/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common /lastt/itimeh,bimp
cc      SAVE /lastt/
      COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      common/snn/efrm,npart1,npart2,epsiPz,epsiPt,PZPROJ,PZTARG
cc      SAVE /snn/
      COMMON/HJGLBR/NELT,NINTHJ,NELP,NINP
cc      SAVE /HJGLBR/
      COMMON/FTMAX/ftsv(MAXSTR),ftsvt(MAXSTR, MAXR)
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
clin-12/14/03:
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
      EXTERNAL IARFLV, INVFLV
      common /para8/ idpert,npertd,idxsec
clin-2/2012:
      common /phiHJ/iphirp,phiRP
      SAVE   
c
      do 1001 i=1,max0(nlast,nnew)
         lastkp(i)=0
 1001 continue
      do 1002 i=1,nnew
         newkp(i)=0
 1002 continue
c     for each of the particles, search the freezeout record (common /hbt/) 
c     to find & keep those which do not have interactions during this timestep:
      do 100 ip=1,nnew
         do 1004 iplast=1,nlast
            if(p(1,ip).eq.plast(1,iplast).and.
     1           p(2,ip).eq.plast(2,iplast).and.
     2           p(3,ip).eq.plast(3,iplast).and.
     3           e(ip).eq.plast(4,iplast).and.
     4           lb(ip).eq.lblast(iplast).and.
     5      dpertp(ip).eq.dplast(iplast).and.lastkp(iplast).eq.0) then
clin-5/2008 modified below to the above in case we have perturbative particles:
c     5           lastkp(iplast).eq.0) then
               deltat=nt*dt-xlast(4,iplast)
               ene=sqrt(plast(1,iplast)**2+plast(2,iplast)**2
     1              +plast(3,iplast)**2+plast(4,iplast)**2)
c     xnew gives the coordinate if a particle free-streams to current time:
               do 1003 ii=1,3
                  xnew(ii)=xlast(ii,iplast)+plast(ii,iplast)/ene*deltat
 1003          continue
                  dr=sqrt((r(1,ip)-xnew(1))**2+(r(2,ip)-xnew(2))**2
     1              +(r(3,ip)-xnew(3))**2)
c     find particles with dp=0 and dr<0.01, considered to be those 
c     without any interactions during this timestep, 
c     thus keep their last positions and time:
               if(dr.le.0.01) then
                  lastkp(iplast)=1
                  newkp(ip)=1
c                  if(lb(ip).eq.41) then
c                write(95,*) 'nt,ip,px,x=',nt,ip,p(1,ip),r(1,ip),ftsv(ip)
c                write(95,*) 'xnew=',xnew(1),xnew(2),xnew(3),xlast(4,ip)
c                  endif
clin-5/2009 Take care of formation time of particles read in at nt=ntmax-1:
                  if(nt.eq.ntmax.and.ftsv(ip).gt.((ntmax-1)*dt)) 
     1                 xlast(4,iplast)=ftsv(ip)
                  goto 100
               endif
            endif
 1004    continue
 100  continue
c     for current particles with interactions, fill their current info in 
c     the freezeout record (if that record entry needs not to be kept):
      do 150 ip=1,nnew
         if(newkp(ip).eq.0) then
            do 1005 iplast=1,nnew
               if(lastkp(iplast).eq.0) then
ctest off: write collision info
c                  if(lb(ip).eq.41) then
c                     write(95,*) 'nt,lb(ip)=',nt,lb(ip)
c                  write(95,*) '  last p=',plast(1,iplast),
c     1 plast(2,iplast),plast(3,iplast),plast(4,iplast)
c                  write(95,*) '  after p=',p(1,ip),p(2,ip),p(3,ip),e(ip)
c                  write(95,*) 'after x=',r(1,ip),r(2,ip),r(3,ip),ftsv(ip)
c                  endif
c
                  xlast(1,iplast)=r(1,ip)
                  xlast(2,iplast)=r(2,ip)
                  xlast(3,iplast)=r(3,ip)
                  xlast(4,iplast)=nt*dt
c
                  if(nt.eq.ntmax) then
c     freezeout time for decay daughters at the last timestep 
c     needs to include the decay time of the parent:
                     if(tfdcy(ip).gt.(ntmax*dt+0.001)) then
                        xlast(4,iplast)=tfdcy(ip)
c     freezeout time for particles unformed at the next-to-last timestep 
c     needs to be their formation time instead of (ntmax*dt):
                     elseif(ftsv(ip).gt.((ntmax-1)*dt)) then
                        xlast(4,iplast)=ftsv(ip)
                     endif
                  endif
                  plast(1,iplast)=p(1,ip)
                  plast(2,iplast)=p(2,ip)
                  plast(3,iplast)=p(3,ip)
                  plast(4,iplast)=e(ip)
                  lblast(iplast)=lb(ip)
                  lastkp(iplast)=1
clin-5/2008:
                  dplast(iplast)=dpertp(ip)
                  goto 150
               endif
 1005       continue
         endif
 150  continue
c     if the current particle list is shorter than the freezeout record,
c     condense the last-collision record by filling new record from 1 to nnew, 
c     and label these entries as keep:
      if(nnew.lt.nlast) then
         do 170 iplast=1,nlast
            if(lastkp(iplast).eq.0) then
               do 1006 ip2=iplast+1,nlast
                  if(lastkp(ip2).eq.1) then
                     xlast(1,iplast)=xlast(1,ip2)
                     xlast(2,iplast)=xlast(2,ip2)
                     xlast(3,iplast)=xlast(3,ip2)
                     xlast(4,iplast)=xlast(4,ip2)
                     plast(1,iplast)=plast(1,ip2)
                     plast(2,iplast)=plast(2,ip2)
                     plast(3,iplast)=plast(3,ip2)
                     plast(4,iplast)=plast(4,ip2)
                     lblast(iplast)=lblast(ip2)
                     lastkp(iplast)=1
clin-5/2008:
                     dplast(iplast)=dplast(ip2)
                     goto 170
                  endif
 1006          continue
            endif
 170     continue
      endif
      nlast=nnew
ctest off look inside each NT timestep (for debugging purpose):
c      do ip=1,nlast
c         write(99,*) ' p ',nt,ip,lblast(ip),plast(1,ip),
c     1        plast(2,ip),plast(3,ip),plast(4,ip),dplast(ip)
c         write(99,*) '  x ',nt,ip,lblast(ip),xlast(1,ip),
c     1        xlast(2,ip),xlast(3,ip),xlast(4,ip),dplast(ip)
c      enddo
c
      if(nt.eq.ntmax) then
clin-5/2008 find final number of perturbative particles (deuterons only):
         ndpert=0
         do ip=1,nlast
            if(dplast(ip).gt.oneminus.and.dplast(ip).lt.oneplus) then
            else
               ndpert=ndpert+1
            endif
         enddo
c
c         write(16,190) IAEVT,IARUN,nlast,bimp,npart1,npart2,
c     1 NELP,NINP,NELT,NINTHJ
clin-2/2012:
c         write(16,190) IAEVT,IARUN,nlast-ndpert,bimp,npart1,npart2,
c     1 NELP,NINP,NELT,NINTHJ
         write(16,191) IAEVT,IARUN,nlast-ndpert,bimp,npart1,npart2,
     1 NELP,NINP,NELT,NINTHJ,phiRP
clin-5/2008 write out perturbatively-produced particles (deuterons only):
         if(idpert.eq.1.or.idpert.eq.2)
     1        write(90,190) IAEVT,IARUN,ndpert,bimp,npart1,npart2,
     2        NELP,NINP,NELT,NINTHJ
         do 1007 ip=1,nlast
clin-12/14/03   No formation time for spectator projectile or target nucleons,
c     see ARINI1 in 'amptsub.f':
clin-3/2009 To be consistent with new particles produced in hadron cascade
c     that are limited by the time-resolution (DT) of the hadron cascade, 
c     freezeout time of spectator projectile or target nucleons is written as 
c     DT as they are read at the 1st timestep and then propagated to time DT: 
c
clin-9/2011 determine spectator nucleons consistently
c            if(plast(1,ip).eq.0.and.plast(2,ip).eq.0
c     1           .and.(sqrt(plast(3,ip)**2+plast(4,ip)**2)*2/HINT1(1))
c     2           .gt.0.99.and.(lblast(ip).eq.1.or.lblast(ip).eq.2)) then
            if(abs(plast(1,ip)).le.epsiPt.and.abs(plast(2,ip)).le.epsiPt
     1           .and.(plast(3,ip).gt.amax1(0.,PZPROJ-epsiPz)
     2                .or.plast(3,ip).lt.(-PZTARG+epsiPz))
     3           .and.(lblast(ip).eq.1.or.lblast(ip).eq.2)) then
clin-5/2008 perturbatively-produced particles (currently only deuterons) 
c     are written to ana/ampt_pert.dat (without the column for the mass); 
c     ana/ampt.dat has regularly-produced particles (including deuterons);
c     these two sets of deuteron data are close to each other(but not the same 
c     because of the bias from triggering the perturbative production); 
c     ONLY use one data set for analysis to avoid double-counting:
               if(dplast(ip).gt.oneminus.and.dplast(ip).lt.oneplus) then
                  write(16,200) INVFLV(lblast(ip)), plast(1,ip),
     1                 plast(2,ip),plast(3,ip),plast(4,ip),
     2                 xlast(1,ip),xlast(2,ip),xlast(3,ip),
     3                 xlast(4,ip)
clin-12/14/03-end
               else
                  if(idpert.eq.1.or.idpert.eq.2) then
                     write(90,250) INVFLV(lblast(ip)), plast(1,ip),
     1                 plast(2,ip),plast(3,ip),
     2                 xlast(1,ip),xlast(2,ip),xlast(3,ip),
     3                 xlast(4,ip)
                  else
                     write(99,*) 'Unexpected perturbative particles'
                  endif
               endif
            elseif(amax1(abs(xlast(1,ip)),abs(xlast(2,ip)),
     1              abs(xlast(3,ip)),abs(xlast(4,ip))).lt.9999) then
               if(dplast(ip).gt.oneminus.and.dplast(ip).lt.oneplus) then
            write(16,200) INVFLV(lblast(ip)), plast(1,ip),
     1           plast(2,ip),plast(3,ip),plast(4,ip),
     2           xlast(1,ip),xlast(2,ip),xlast(3,ip),xlast(4,ip)
               else
                  if(idpert.eq.1.or.idpert.eq.2) then
            write(90,250) INVFLV(lblast(ip)),plast(1,ip),
     1           plast(2,ip),plast(3,ip),
     2           xlast(1,ip),xlast(2,ip),xlast(3,ip),xlast(4,ip),
     3           dplast(ip)
                  else
                     write(99,*) 'Unexpected perturbative particles'
                  endif
               endif
            else
c     change format for large numbers:
               if(dplast(ip).gt.oneminus.and.dplast(ip).lt.oneplus) then
            write(16,201) INVFLV(lblast(ip)), plast(1,ip),
     1           plast(2,ip),plast(3,ip),plast(4,ip),
     2           xlast(1,ip),xlast(2,ip),xlast(3,ip),xlast(4,ip)
               else
                  if(idpert.eq.1.or.idpert.eq.2) then
                     write(90,251) INVFLV(lblast(ip)), plast(1,ip),
     1           plast(2,ip),plast(3,ip),
     2           xlast(1,ip),xlast(2,ip),xlast(3,ip),xlast(4,ip),
     3           dplast(ip)
                  else
                     write(99,*) 'Unexpected perturbative particles'
                  endif
               endif
            endif
 1007    continue
         if(ioscar.eq.1) call hoscar
      endif
 190  format(3(i7),f10.4,5x,6(i4))
 191  format(3(i7),f10.4,5x,6(i4),5x,f7.4)
clin-3/2009 improve the output accuracy of Pz
 200  format(I6,2(1x,f8.3),1x,f11.4,1x,f6.3,4(1x,f8.2))
 201  format(I6,2(1x,f8.3),1x,f11.4,1x,f6.3,4(1x,e8.2))
 250  format(I5,2(1x,f8.3),1x,f10.3,2(1x,f7.1),1x,f8.2,1x,f7.2,1x,e10.4)
 251  format(I5,2(1x,f8.3),1x,f10.3,4(1x,e8.2),1x,e10.4)
c     
        return
        end

c=======================================================================
        SUBROUTINE decomp(px0,py0,pz0,xm0,i,itq1)
c
        IMPLICIT DOUBLE PRECISION(D)  
        DOUBLE PRECISION  enenew, pxnew, pynew, pznew
        DOUBLE PRECISION  de0, beta2, gam
        common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
        COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
        common /decom/ptwo(2,5)
cc      SAVE /decom/
        COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
        COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
        common/embed/iembed,nsembd,pxqembd,pyqembd,xembd,yembd,
     1       psembd,tmaxembd,phidecomp
        SAVE   
c
        dcth=dble(RANART(NSEED))*2.d0-1.d0
        dPHI=dble(RANART(NSEED)*HIPR1(40))*2.d0
clin-6/2009 Added if embedding a high-Pt quark pair after string melting:
        if(iembed.ge.1.and.iembed.le.4) then
c     Decompose the parent high-Pt pion to q and qbar with an internal momentum
c     parallel to the pion direction so that one parton has ~the same hight Pt
c     and the other parton has a very soft Pt:
c     Note: htop() decomposes a meson to q as it(1) followed by qbar as it(2):
           if(i.eq.(natt-2*nsembd).or.i.eq.(natt-2*nsembd-1)) then
              dcth=0.d0
              dphi=dble(phidecomp)
           endif
        endif
c
        ds=dble(xm0)**2
        dpcm=dsqrt((ds-dble(ptwo(1,5)+ptwo(2,5))**2)
     1 *(ds-dble(ptwo(1,5)-ptwo(2,5))**2)/ds/4d0)
        dpz=dpcm*dcth
        dpx=dpcm*dsqrt(1.d0-dcth**2)*dcos(dphi)
        dpy=dpcm*dsqrt(1.d0-dcth**2)*dsin(dphi)
        de1=dsqrt(dble(ptwo(1,5))**2+dpcm**2)
        de2=dsqrt(dble(ptwo(2,5))**2+dpcm**2)
c
      de0=dsqrt(dble(px0)**2+dble(py0)**2+dble(pz0)**2+dble(xm0)**2)
        dbex=dble(px0)/de0
        dbey=dble(py0)/de0
        dbez=dble(pz0)/de0
c     boost the reference frame up by beta (pznew=gam(pz+beta e)):
      beta2 = dbex ** 2 + dbey ** 2 + dbez ** 2
      gam = 1.d0 / dsqrt(1.d0 - beta2)
      if(beta2.ge.0.9999999999999d0) then
         write(6,*) '1',dbex,dbey,dbez,beta2,gam
      endif
c
      call lorenz(de1,dpx,dpy,dpz,-dbex,-dbey,-dbez)
        ptwo(1,1)=sngl(pxnew)
        ptwo(1,2)=sngl(pynew)
        ptwo(1,3)=sngl(pznew)
        ptwo(1,4)=sngl(enenew)
      call lorenz(de2,-dpx,-dpy,-dpz,-dbex,-dbey,-dbez)
        ptwo(2,1)=sngl(pxnew)
        ptwo(2,2)=sngl(pynew)
        ptwo(2,3)=sngl(pznew)
        ptwo(2,4)=sngl(enenew)
c
      RETURN
      END

c=======================================================================
      SUBROUTINE HTOP
c
      PARAMETER (MAXSTR=150001)
      PARAMETER (MAXPTN=400001)
      PARAMETER (MAXIDL=4001)
      DOUBLE PRECISION  GX0, GY0, GZ0, FT0, PX0, PY0, PZ0, E0, XMASS0
      DOUBLE PRECISION  PXSGS,PYSGS,PZSGS,PESGS,PMSGS,
     1     GXSGS,GYSGS,GZSGS,FTSGS
      dimension it(4)
      COMMON/HMAIN2/KATT(MAXSTR,4),PATT(MAXSTR,4)
cc      SAVE /HMAIN2/
      COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
cc      SAVE /HMAIN1/
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &     PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &     XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
      COMMON /ilist7/ LSTRG0(MAXPTN), LPART0(MAXPTN)
cc      SAVE /ilist7/
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
cc      SAVE /ARPRC/
      common /decom/ptwo(2,5)
cc      SAVE /decom/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /NOPREC/ NNOZPC, ITYPN(MAXIDL),
     &     GXN(MAXIDL), GYN(MAXIDL), GZN(MAXIDL), FTN(MAXIDL),
     &     PXN(MAXIDL), PYN(MAXIDL), PZN(MAXIDL), EEN(MAXIDL),
     &     XMN(MAXIDL)
cc      SAVE /NOPREC/
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
c     7/20/01: use double precision
c     otherwise sometimes beta>1 and gamma diverge in lorenz():
      COMMON/SOFT/PXSGS(MAXSTR,3),PYSGS(MAXSTR,3),PZSGS(MAXSTR,3),
     &     PESGS(MAXSTR,3),PMSGS(MAXSTR,3),GXSGS(MAXSTR,3),
     &     GYSGS(MAXSTR,3),GZSGS(MAXSTR,3),FTSGS(MAXSTR,3),
     &     K1SGS(MAXSTR,3),K2SGS(MAXSTR,3),NJSGS(MAXSTR)
cc      SAVE /SOFT/
      common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
      DOUBLE PRECISION  vxp0,vyp0,vzp0
      common /precpa/ vxp0(MAXPTN), vyp0(MAXPTN), vzp0(MAXPTN)
cc      SAVE /precpa/
      common /para7/ ioscar,nsmbbbar,nsmmeson
      COMMON /AREVT/ IAEVT, IARUN, MISS
      common/snn/efrm,npart1,npart2,epsiPz,epsiPt,PZPROJ,PZTARG
      SAVE   
c
        npar=0
        nnozpc=0
clin-5b/2008 calculate the number of hadrons to be converted to q/qbar:
        if((isoft.eq.4.or.isoft.eq.5).and.(ioscar.eq.2.or.ioscar.eq.3)) 
     1       then
           nsmbbbar=0
           nsmmeson=0
           do i=1,natt
              id=ITYPAR(i)
              idabs=iabs(id)
              i2=MOD(idabs/10,10)
clin-9/2011 determine spectator nucleons consistently
c              if(PXAR(i).eq.0.and.PYAR(i).eq.0.and.PEAR(i)
c     1             .ge.(HINT1(1)/2*0.99).and.
c     2             .and.(id.eq.2112.or.id.eq.2212)) then
              if(abs(PXAR(i)).le.epsiPt.and.abs(PYAR(i)).le.epsiPt
     1             .and.(PZAR(i).gt.amax1(0.,PZPROJ-epsiPz)
     2                .or.PZAR(i).lt.(-PZTARG+epsiPz))
     3             .and.(id.eq.2112.or.id.eq.2212)) then
c     spectator proj or targ nucleons without interactions, do not enter ZPC:
              elseif(idabs.gt.1000.and.i2.ne.0) then
c     baryons to be converted to q/qbar:
                 nsmbbbar=nsmbbbar+1
              elseif((idabs.gt.100.and.idabs.lt.1000)
     1                .or.idabs.gt.10000) then
c     mesons to be converted to q/qbar:
                 nsmmeson=nsmmeson+1
              endif
           enddo

clin-6/2009:
           if(ioscar.eq.2.or.ioscar.eq.3) then
              write(92,*) iaevt,miss,3*nsmbbbar+2*nsmmeson,
     1             nsmbbbar,nsmmeson,natt,natt-nsmbbbar-nsmmeson
           endif
c           write(92,*) iaevt, 3*nsmbbbar+2*nsmmeson
c           write(92,*) ' event#, total # of initial partons after string 
c     1 melting'
c           write(92,*) 'String melting converts ',nsmbbbar, ' baryons &'
c     1, nsmmeson, ' mesons'
c           write(92,*) 'Total # of initial particles= ',natt
c           write(92,*) 'Total # of initial particles (gamma,e,muon,...) 
c     1 not entering ZPC= ',natt-nsmbbbar-nsmmeson
        endif
clin-5b/2008-over
        do 100 i=1,natt
           id=ITYPAR(i)
           idabs=iabs(id)
           i4=MOD(idabs/1000,10)
           i3=MOD(idabs/100,10)
           i2=MOD(idabs/10,10)
           i1=MOD(idabs,10)
           rnum=RANART(NSEED)
           ftime=0.197*PEAR(i)/(PXAR(i)**2+PYAR(i)**2+XMAR(i)**2)
           inozpc=0
           it(1)=0
           it(2)=0
           it(3)=0
           it(4)=0
c
clin-9/2011 determine spectator nucleons consistently
c           if(PXAR(i).eq.0.and.PYAR(i).eq.0.and.PEAR(i)
c     1 .ge.(HINT1(1)/2*0.99).and.((id.eq.2112).or.(id.eq.2212))) then
              if(abs(PXAR(i)).le.epsiPt.and.abs(PYAR(i)).le.epsiPt
     1             .and.(PZAR(i).gt.amax1(0.,PZPROJ-epsiPz)
     2                .or.PZAR(i).lt.(-PZTARG+epsiPz))
     3             .and.(id.eq.2112.or.id.eq.2212)) then
c     spectator proj or targ nucleons without interactions, do not enter ZPC:
              inozpc=1
           elseif(idabs.gt.1000.and.i2.ne.0) then
c     baryons:
              if(((i4.eq.1.or.i4.eq.2).and.i4.eq.i3)
     1 .or.(i4.eq.3.and.i3.eq.3)) then
                 if(i1.eq.2) then
                    if(rnum.le.(1./2.)) then
                       it(1)=i4
                       it(2)=i3*1000+i2*100+1
                    elseif(rnum.le.(2./3.)) then
                       it(1)=i4
                       it(2)=i3*1000+i2*100+3
                    else
                       it(1)=i2
                       it(2)=i4*1000+i3*100+3
                    endif
                 elseif(i1.eq.4) then
                    if(rnum.le.(2./3.)) then
                       it(1)=i4
                       it(2)=i3*1000+i2*100+3
                    else
                       it(1)=i2
                       it(2)=i4*1000+i3*100+3
                    endif
                 endif
              elseif(i4.eq.1.or.i4.eq.2) then
                 if(i1.eq.2) then
                    if(rnum.le.(1./2.)) then
                       it(1)=i2
                       it(2)=i4*1000+i3*100+1
                    elseif(rnum.le.(2./3.)) then
                       it(1)=i2
                       it(2)=i4*1000+i3*100+3
                    else
                       it(1)=i4
                       it(2)=i3*1000+i2*100+3
                    endif
                 elseif(i1.eq.4) then
                    if(rnum.le.(2./3.)) then
                       it(1)=i2
                       it(2)=i4*1000+i3*100+3
                    else
                       it(1)=i4
                       it(2)=i3*1000+i2*100+3
                    endif
                 endif
              elseif(i4.ge.3) then
                 it(1)=i4
                 if(i3.lt.i2) then
                    it(2)=i2*1000+i3*100+1
                 else
                    it(2)=i3*1000+i2*100+3
                 endif
              endif
c       antibaryons:
              if(id.lt.0) then
                 it(1)=-it(1)
                 it(2)=-it(2)
              endif
c     isoft=4or5 decompose diquark flavor it(2) to two quarks it(3)&(4):
              if(isoft.eq.4.or.isoft.eq.5) then
                 it(3)=MOD(it(2)/1000,10)
                 it(4)=MOD(it(2)/100,10)
              endif

           elseif((idabs.gt.100.and.idabs.lt.1000)
     1 .or.idabs.gt.10000) then
c     mesons:
              if(i3.eq.i2) then
                 if(i3.eq.1.or.i3.eq.2) then
                    if(rnum.le.0.5) then
                       it(1)=1
                       it(2)=-1
                    else
                       it(1)=2
                       it(2)=-2
                    endif
                 else
                    it(1)=i3
                    it(2)=-i3
                 endif
              else
                 if((isign(1,id)*(-1)**i3).eq.1) then
                    it(1)=i3
                    it(2)=-i2
                 else
                    it(1)=i2
                    it(2)=-i3
                 endif
              endif
           else
c     save other particles (leptons and photons) outside of ZPC:
              inozpc=1
           endif
c
           if(inozpc.eq.1) then
              NJSGS(i)=0
              nnozpc=nnozpc+1
              itypn(nnozpc)=ITYPAR(i)
              pxn(nnozpc)=PXAR(i)
              pyn(nnozpc)=PYAR(i)
              pzn(nnozpc)=PZAR(i)
              een(nnozpc)=PEAR(i)
              xmn(nnozpc)=XMAR(i)
              gxn(nnozpc)=GXAR(i)
              gyn(nnozpc)=GYAR(i)
              gzn(nnozpc)=GZAR(i)
              ftn(nnozpc)=FTAR(i)
           else
              NJSGS(i)=2
              ptwo(1,5)=ulmass(it(1))
              ptwo(2,5)=ulmass(it(2))
              call decomp(patt(i,1),patt(i,2),patt(i,3),XMAR(i),i,it(1))
              ipamax=2
              if((isoft.eq.4.or.isoft.eq.5)
     1 .and.iabs(it(2)).gt.1000) ipamax=1
              do 1001 ipar=1,ipamax
                 npar=npar+1
                 ityp0(npar)=it(ipar)
                 px0(npar)=dble(ptwo(ipar,1))
                 py0(npar)=dble(ptwo(ipar,2))
                 pz0(npar)=dble(ptwo(ipar,3))
                 e0(npar)=dble(ptwo(ipar,4))
                 xmass0(npar)=dble(ptwo(ipar,5))
                 gx0(npar)=dble(GXAR(i))
                 gy0(npar)=dble(GYAR(i))
                 gz0(npar)=dble(GZAR(i))
                 ft0(npar)=dble(ftime)
                 lstrg0(npar)=i
                 lpart0(npar)=ipar
                 vxp0(npar)=dble(patt(i,1)/patt(i,4))
                 vyp0(npar)=dble(patt(i,2)/patt(i,4))
                 vzp0(npar)=dble(patt(i,3)/patt(i,4))
 1001     continue
 200      format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,f8.2))
 201      format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,4(1x,e8.2))
c
              if((isoft.eq.4.or.isoft.eq.5)
     1 .and.iabs(it(2)).gt.1000) then
                 NJSGS(i)=3
                 xmdq=ptwo(2,5)
                 ptwo(1,5)=ulmass(it(3))
                 ptwo(2,5)=ulmass(it(4))
c     8/19/02 avoid actual argument in common blocks of DECOMP:
c                 call decomp(ptwo(2,1),ptwo(2,2),ptwo(2,3),xmdq)
             ptwox=ptwo(2,1)
             ptwoy=ptwo(2,2)
             ptwoz=ptwo(2,3)
             call decomp(ptwox,ptwoy,ptwoz,xmdq,i,it(1))
c
                 do 1002 ipar=1,2
                    npar=npar+1
                    ityp0(npar)=it(ipar+2)
                    px0(npar)=dble(ptwo(ipar,1))
                    py0(npar)=dble(ptwo(ipar,2))
                    pz0(npar)=dble(ptwo(ipar,3))
                    e0(npar)=dble(ptwo(ipar,4))
                    xmass0(npar)=dble(ptwo(ipar,5))
                    gx0(npar)=dble(GXAR(i))
                    gy0(npar)=dble(GYAR(i))
                    gz0(npar)=dble(GZAR(i))
                    ft0(npar)=dble(ftime)
                    lstrg0(npar)=i
                    lpart0(npar)=ipar+1
                    vxp0(npar)=dble(patt(i,1)/patt(i,4))
                    vyp0(npar)=dble(patt(i,2)/patt(i,4))
                    vzp0(npar)=dble(patt(i,3)/patt(i,4))
 1002        continue
              endif
c
           endif
 100        continue
      MUL=NPAR
c      
clin-5b/2008:
      if((isoft.eq.4.or.isoft.eq.5).and.(ioscar.eq.2.or.ioscar.eq.3)) 
     1     then
         if((natt-nsmbbbar-nsmmeson).ne.nnozpc) 
     1        write(92,*) 'Problem with the total # of initial particles
     2 (gamma,e,muon,...) not entering ZPC'
         if((3*nsmbbbar+2*nsmmeson).ne.npar) 
     1        write(92,*) 'Problem with the total # of initial partons
     2 after string melting'
      endif
c
      RETURN
      END

c=======================================================================
      SUBROUTINE PTOH
c
      PARAMETER (MAXSTR=150001)
      DOUBLE PRECISION  gxp,gyp,gzp,ftp,pxp,pyp,pzp,pep,pmp
      DOUBLE PRECISION  gxp0,gyp0,gzp0,ft0fom,drlocl
      DOUBLE PRECISION  enenew, pxnew, pynew, pznew, beta2, gam
      DOUBLE PRECISION  ftavg0,gxavg0,gyavg0,gzavg0,bex,bey,bez
      DOUBLE PRECISION  PXSGS,PYSGS,PZSGS,PESGS,PMSGS,
     1     GXSGS,GYSGS,GZSGS,FTSGS
      DOUBLE PRECISION  xmdiag,px1,py1,pz1,e1,px2,py2,pz2,e2,
     1     px3,py3,pz3,e3,xmpair,etot
clin-9/2012: improve precision for argument in sqrt():
      DOUBLE PRECISION  p1,p2,p3
      common /loclco/gxp(3),gyp(3),gzp(3),ftp(3),
     1     pxp(3),pyp(3),pzp(3),pep(3),pmp(3)
cc      SAVE /loclco/
      COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
cc      SAVE /HMAIN1/
      COMMON/HMAIN2/KATT(MAXSTR,4),PATT(MAXSTR,4)
cc      SAVE /HMAIN2/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &     K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &     PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
cc      SAVE /ARPRC/
      COMMON/SOFT/PXSGS(MAXSTR,3),PYSGS(MAXSTR,3),PZSGS(MAXSTR,3),
     &     PESGS(MAXSTR,3),PMSGS(MAXSTR,3),GXSGS(MAXSTR,3),
     &     GYSGS(MAXSTR,3),GZSGS(MAXSTR,3),FTSGS(MAXSTR,3),
     &     K1SGS(MAXSTR,3),K2SGS(MAXSTR,3),NJSGS(MAXSTR)
cc      SAVE /SOFT/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
      common /prtn23/ gxp0(3),gyp0(3),gzp0(3),ft0fom
cc      SAVE /prtn23/
      common /nzpc/nattzp
cc      SAVE /nzpc/
      common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
cc      SAVE /LUDAT1/ 
clin 4/19/2006
      common /lastt/itimeh,bimp
      COMMON/HJGLBR/NELT,NINTHJ,NELP,NINP
      COMMON /AREVT/ IAEVT, IARUN, MISS
      common /para7/ ioscar,nsmbbbar,nsmmeson
clin-5/2011
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
c
      dimension xmdiag(MAXSTR),indx(MAXSTR),ndiag(MAXSTR)
      SAVE   
c
      call coales
c     obtain particle mass here without broadening by Breit-Wigner width:
      mstj24=MSTJ(24)
      MSTJ(24)=0
        nuudd=0
        npich=0
        nrhoch=0
      ppi0=1.
      prho0=0.
c     determine hadron flavor except (pi0,rho0,eta,omega):
      DO 1001 ISG = 1, NSG
           if(NJSGS(ISG).ne.0) then
              NATT=NATT+1
              K1=K2SGS(ISG,1)
              k1abs=iabs(k1)
              PX1=PXSGS(ISG,1)
              PY1=PYSGS(ISG,1)
              PZ1=PZSGS(ISG,1)
              K2=K2SGS(ISG,2)
              k2abs=iabs(k2)
              PX2=PXSGS(ISG,2)
              PY2=PYSGS(ISG,2)
              PZ2=PZSGS(ISG,2)
c     5/02/01 try lowest spin states as first choices, 
c     i.e. octet baryons and pseudoscalar mesons (ibs=2*baryonspin+1):
              e1=PESGS(ISG,1)
              e2=PESGS(ISG,2)
              xmpair=dsqrt((e1+e2)**2-(px1+px2)**2-(py1+py2)**2
     1 -(pz1+pz2)**2)
              ibs=2
              imspin=0
              if(k1.eq.-k2.and.iabs(k1).le.2.
     1           and.NJSGS(ISG).eq.2) then
               nuudd=nuudd+1
               xmdiag(nuudd)=xmpair
               ndiag(nuudd)=natt
            endif
              K3=0
              if((isoft.eq.4.or.isoft.eq.5).and.NJSGS(ISG).eq.3) then
               K3=K2SGS(ISG,3)
               k3abs=iabs(k3)
               PX3=PXSGS(ISG,3)
               PY3=PYSGS(ISG,3)
               PZ3=PZSGS(ISG,3)
               e3=PESGS(ISG,3)
               xmpair=dsqrt((e1+e2+e3)**2-(px1+px2+px3)**2
     1              -(py1+py2+py3)**2-(pz1+pz2+pz3)**2)
              endif
c*****     isoft=3 baryon decomposition is different:
              if(isoft.eq.3.and.
     1           (k1abs.gt.1000.or.k2abs.gt.1000)) then
               if(k1abs.gt.1000) then
                  kdq=k1abs
                  kk=k2abs
               else
                  kdq=k2abs
                  kk=k1abs
               endif
               ki=MOD(kdq/1000,10)
               kj=MOD(kdq/100,10)
               if(MOD(kdq,10).eq.1) then
                  idqspn=0
               else
                  idqspn=1
               endif
c
               if(kk.gt.ki) then
                  ktemp=kk
                  kk=kj
                  kj=ki
                  ki=ktemp
               elseif(kk.gt.kj) then
                  ktemp=kk
                  kk=kj
                  kj=ktemp
               endif
c     
               if(ki.ne.kj.and.ki.ne.kk.and.kj.ne.kk) then
                  if(idqspn.eq.0) then
                     kf=1000*ki+100*kk+10*kj+ibs
                  else
                     kf=1000*ki+100*kj+10*kk+ibs
                  endif
               elseif(ki.eq.kj.and.ki.eq.kk) then
c     can only be decuplet baryons:
                  kf=1000*ki+100*kj+10*kk+4
               else
                  kf=1000*ki+100*kj+10*kk+ibs
               endif
c     form a decuplet baryon if the q+diquark mass is closer to its mass 
c     (and if the diquark has spin 1):
cc     for now only include Delta, which is present in ART:
cc                 if(idqspn.eq.1.and.MOD(kf,10).eq.2) then
               if(kf.eq.2112.or.kf.eq.2212) then
                  if(abs(sngl(xmpair)-ULMASS(kf)).gt.
     1                 abs(sngl(xmpair)-ULMASS(kf+2))) kf=kf+2
               endif
               if(k1.lt.0) kf=-kf
clin-6/22/01 isoft=4or5 baryons:
              elseif((isoft.eq.4.or.isoft.eq.5).and.NJSGS(ISG).eq.3) 
     1              then
               if(k1abs.gt.k2abs) then
                  ki=k1abs
                  kk=k2abs
               else
                  ki=k2abs
                  kk=k1abs
               endif
               if(k3abs.gt.ki) then
                  kj=ki
                  ki=k3abs
               elseif(k3abs.lt.kk) then
                  kj=kk
                  kk=k3abs
               else
                  kj=k3abs
               endif
c     
               if(ki.eq.kj.and.ki.eq.kk) then
c     can only be decuplet baryons (Delta-,++, Omega):
                  ibs=4
                  kf=1000*ki+100*kj+10*kk+ibs
               elseif(ki.ne.kj.and.ki.ne.kk.and.kj.ne.kk) then
c     form Lambda or Sigma according to 3-quark mass, 
c     for now neglect decuplet (Sigma*0 etc) which is absent in ART:
                  ibs=2
                  kf1=1000*ki+100*kj+10*kk+ibs
                  kf2=1000*ki+100*kk+10*kj+ibs
                  kf=kf1
                  if(abs(sngl(xmpair)-ULMASS(kf1)).gt.
     1                 abs(sngl(xmpair)-ULMASS(kf2))) kf=kf2
               else
                  ibs=2
                  kf=1000*ki+100*kj+10*kk+ibs
cc     for now only include Delta0,+ as decuplets, which are present in ART:
                  if(kf.eq.2112.or.kf.eq.2212) then
                     if(abs(sngl(xmpair)-ULMASS(kf)).gt.
     1                    abs(sngl(xmpair)-ULMASS(kf+2))) kf=kf+2
                  endif
               endif
               if(k1.lt.0) kf=-kf
c*****     mesons:
              else
               if(k1abs.eq.k2abs) then
                  if(k1abs.le.2) then
c     treat diagonal mesons later in the subroutine:
                     kf=0
                  elseif(k1abs.le.3) then
c     do not form eta', only form phi from s-sbar, since no eta' in ART:
                     kf=333
                  else
                     kf=100*k1abs+10*k1abs+2*imspin+1
                  endif
               else
                  if(k1abs.gt.k2abs) then
                     kmax=k1abs
                     kmin=k2abs
                  elseif(k1abs.lt.k2abs) then
                     kmax=k2abs
                     kmin=k1abs
                  endif
                  kf=(100*kmax+10*kmin+2*imspin+1)
     1                 *isign(1,k1+k2)*(-1)**kmax
c     form a vector meson if the q+qbar mass is closer to its mass:
                  if(MOD(iabs(kf),10).eq.1) then
                     if(abs(sngl(xmpair)-ULMASS(iabs(kf))).gt.
     1                    abs(sngl(xmpair)-ULMASS(iabs(kf)+2))) 
     2                    kf=(iabs(kf)+2)*isign(1,kf)
                  endif
               endif
              endif
              ITYPAR(NATT)=kf
              KATT(NATT,1)=kf
            if(iabs(kf).eq.211) then
               npich=npich+1
            elseif(iabs(kf).eq.213) then
               nrhoch=nrhoch+1
            endif
           endif
clin-7/2011-check charm hadron flavors:
c           if(k1abs.eq.4.or.k2abs.eq.4) then
c              if(k3.eq.0) then
c                 write(99,*) iaevt,k1,k2,kf,xmpair,
c     1                ULMASS(iabs(kf)),ULMASS(iabs(kf)+2),isg
c              else
c                 write(99,*) iaevt,k1,k2,k3,kf,xmpair,
c     1                ULMASS(iabs(kf)),ULMASS(iabs(kf)+2),isg
c              endif
c           endif
clin-7/2011-end
 1001   CONTINUE
c     assume Npi0=(Npi+ + Npi-)/2, Nrho0=(Nrho+ + Nrho-)/2 on the average:
        if(nuudd.ne.0) then
         ppi0=float(npich/2)/float(nuudd)
         prho0=float(nrhoch/2)/float(nuudd)
      endif      
c     determine diagonal mesons (pi0,rho0,eta and omega) from uubar/ddbar:
      npi0=0
      DO 1002 ISG = 1, NSG
         if(K2SGS(ISG,1).eq.-K2SGS(ISG,2)
     1        .and.iabs(K2SGS(ISG,1)).le.2.and.NJSGS(ISG).eq.2) then
            if(RANART(NSEED).le.ppi0) npi0=npi0+1
         endif
 1002 CONTINUE
c
      if(nuudd.gt.1) then
         call index1(MAXSTR,nuudd,xmdiag,indx)
      else
         indx(1)=1
      end if
c
      DO 1003 ix=1,nuudd
         iuudd=indx(ix)
         inatt=ndiag(iuudd)            
         if(ix.le.npi0) then
            kf=111
         elseif(RANART(NSEED).le.(prho0/(1-ppi0+0.00001))) then
            kf=113
         else
c     at T=150MeV, thermal weights for eta and omega(spin1) are about the same:
            if(RANART(NSEED).le.0.5) then
               kf=221
            else
               kf=223
            endif
         endif
         ITYPAR(inatt)=kf
         KATT(inatt,1)=kf
 1003 CONTINUE
c  determine hadron formation time, position and momentum:
      inatt=0
clin-6/2009 write out parton info after coalescence:
      if(ioscar.eq.3) then
         WRITE (85, 395) IAEVT, 3*nsmbbbar+2*nsmmeson,nsmbbbar,nsmmeson, 
     1     bimp, NELP,NINP,NELT,NINTHJ,MISS
      endif
 395  format(4I8,f10.4,5I5)
c
      DO 1006 ISG = 1, NSG
           if(NJSGS(ISG).ne.0) then
            inatt=inatt+1
              K1=K2SGS(ISG,1)
              k1abs=iabs(k1)
              PX1=PXSGS(ISG,1)
              PY1=PYSGS(ISG,1)
              PZ1=PZSGS(ISG,1)
              K2=K2SGS(ISG,2)
              k2abs=iabs(k2)
              PX2=PXSGS(ISG,2)
              PY2=PYSGS(ISG,2)
              PZ2=PZSGS(ISG,2)
              e1=PESGS(ISG,1)
              e2=PESGS(ISG,2)
c
              if(NJSGS(ISG).eq.2) then
               PXAR(inatt)=sngl(px1+px2)
               PYAR(inatt)=sngl(py1+py2)
               PZAR(inatt)=sngl(pz1+pz2)
               PATT(inatt,1)=PXAR(inatt)
               PATT(inatt,2)=PYAR(inatt)
               PATT(inatt,3)=PZAR(inatt)
               etot=e1+e2
clin-9/2012: improve precision for argument in sqrt():
               p1=px1+px2
               p2=py1+py2
               p3=pz1+pz2
c
              elseif((isoft.eq.4.or.isoft.eq.5).and.NJSGS(ISG).eq.3) 
     1              then
               PX3=PXSGS(ISG,3)
               PY3=PYSGS(ISG,3)
               PZ3=PZSGS(ISG,3)
               e3=PESGS(ISG,3)
               PXAR(inatt)=sngl(px1+px2+px3)
               PYAR(inatt)=sngl(py1+py2+py3)
               PZAR(inatt)=sngl(pz1+pz2+pz3)
               PATT(inatt,1)=PXAR(inatt)
               PATT(inatt,2)=PYAR(inatt)
               PATT(inatt,3)=PZAR(inatt)
               etot=e1+e2+e3
clin-9/2012: improve precision for argument in sqrt():
               p1=px1+px2+px3
               p2=py1+py2+py3
               p3=pz1+pz2+pz3
c
              endif
              XMAR(inatt)=ULMASS(ITYPAR(inatt))
clin-5/2011-add finite width to resonances (rho,omega,eta,K*,phi,Delta) after formation:
              kf=KATT(inatt,1)
              if(kf.eq.113.or.abs(kf).eq.213.or.kf.eq.221.or.kf.eq.223
     1             .or.abs(kf).eq.313.or.abs(kf).eq.323.or.kf.eq.333
     2             .or.abs(kf).eq.1114.or.abs(kf).eq.2114
     3             .or.abs(kf).eq.2214.or.abs(kf).eq.2224) then
                 XMAR(inatt)=resmass(kf)
              endif
c
              PEAR(inatt)=sqrt(PXAR(inatt)**2+PYAR(inatt)**2
     1           +PZAR(inatt)**2+XMAR(inatt)**2)
              PATT(inatt,4)=PEAR(inatt)
              EATT=EATT+PEAR(inatt)
            ipartn=NJSGS(ISG)
            DO 1004 i=1,ipartn
               ftp(i)=ftsgs(isg,i)
               gxp(i)=gxsgs(isg,i)
               gyp(i)=gysgs(isg,i)
               gzp(i)=gzsgs(isg,i)
               pxp(i)=pxsgs(isg,i)
               pyp(i)=pysgs(isg,i)
               pzp(i)=pzsgs(isg,i)
               pmp(i)=pmsgs(isg,i)
               pep(i)=pesgs(isg,i)
 1004       CONTINUE
            call locldr(ipartn,drlocl)
c
            tau0=ARPAR1(1)
            ftavg0=ft0fom+dble(tau0)
            gxavg0=0d0
            gyavg0=0d0
            gzavg0=0d0
            DO 1005 i=1,ipartn
               gxavg0=gxavg0+gxp0(i)/ipartn
               gyavg0=gyavg0+gyp0(i)/ipartn
               gzavg0=gzavg0+gzp0(i)/ipartn
 1005       CONTINUE
clin-9/2012: improve precision for argument in sqrt():
c            bex=dble(PXAR(inatt))/etot
c            bey=dble(PYAR(inatt))/etot
c            bez=dble(PZAR(inatt))/etot
            bex=p1/etot
            bey=p2/etot
            bez=p3/etot
c
            beta2 = bex ** 2 + bey ** 2 + bez ** 2
            gam = 1.d0 / dsqrt(1.d0 - beta2)
            if(beta2.ge.0.9999999999999d0) then
               write(6,*) '2',bex,bey,bez,beta2,gam
            endif
c
            call lorenz(ftavg0,gxavg0,gyavg0,gzavg0,-bex,-bey,-bez)
              GXAR(inatt)=sngl(pxnew)
              GYAR(inatt)=sngl(pynew)
              GZAR(inatt)=sngl(pznew)
              FTAR(inatt)=sngl(enenew)
clin 4/19/2006 write out parton info after coalescence:
              if(ioscar.eq.3) then
                 WRITE (85, 313) K2SGS(ISG,1),px1,py1,pz1,PMSGS(ISG,1),
     1                inatt,katt(inatt,1),xmar(inatt)
                 WRITE (85, 312) K2SGS(ISG,2),px2,py2,pz2,PMSGS(ISG,2),
     1                inatt,katt(inatt,1)
                 if(NJSGS(ISG).eq.3) WRITE (85, 312) K2SGS(ISG,3),
     1                px3,py3,pz3,PMSGS(ISG,3),inatt,katt(inatt,1)
              endif
 312       FORMAT(I6,4(1X,F10.3),1X,I6,1X,I6)
clin-5/02/2011
 313          FORMAT(I6,4(1X,F10.3),1X,I6,1X,I6,1X,F10.3)
c
           endif
 1006   CONTINUE
c     number of hadrons formed from partons inside ZPC:
      nattzp=natt
      MSTJ(24)=mstj24
c      
      RETURN
      END

c=======================================================================
clin-5/2011-add finite width to resonances (rho,omega,eta,K*,phi,Delta) after formation:
      FUNCTION resmass(kf)

      PARAMETER  (arho=0.775,aomega=0.783,aeta=0.548,aks=0.894,
     1     aphi=1.019,adelta=1.232)
      PARAMETER  (wrho=0.149,womega=0.00849,weta=1.30E-6,wks=0.0498,
     1     wphi=0.00426,wdelta=0.118)
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
      COMMON/RNDF77/NSEED
      SAVE   

      if(kf.eq.113.or.abs(kf).eq.213) then
         amass=arho
         wid=wrho
      elseif(kf.eq.221) then
         amass=aeta
         wid=weta
      elseif(kf.eq.223) then
         amass=aomega
         wid=womega
      elseif(abs(kf).eq.313.or.abs(kf).eq.323) then
         amass=aks
         wid=wks
      elseif(kf.eq.333) then
         amass=aphi
         wid=wphi
      elseif(abs(kf).eq.1114.or.abs(kf).eq.2114
     1        .or.abs(kf).eq.2214.or.abs(kf).eq.2224) then
         amass=adelta
         wid=wdelta
      endif
      dmin=amass-2*wid
      dmax=amass+2*wid
c     Delta mass needs to be big enough to decay to N+pi:
      if(amass.eq.adelta) dmin=1.078
c      
      FM=1.
      NTRY1=0
 10   DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
      NTRY1=NTRY1+1
      fmass=(amass*wid)**2/((DM**2-amass**2)**2+(amass*wid)**2)
check      write (99,*) ntry1,kf,amass,wid,fmass,DM
      IF((RANART(NSEED) .GT. FMASS/FM).AND. (NTRY1.LE.10)) GOTO 10
c     
      resmass=DM
      
      RETURN
      END

c=======================================================================
      SUBROUTINE coales

      PARAMETER (MAXSTR=150001)
      IMPLICIT DOUBLE PRECISION(D)
      DOUBLE PRECISION  gxp,gyp,gzp,ftp,pxp,pyp,pzp,pep,pmp
      DIMENSION IOVER(MAXSTR),dp1(2:3),dr1(2:3)
      DOUBLE PRECISION  PXSGS,PYSGS,PZSGS,PESGS,PMSGS,
     1     GXSGS,GYSGS,GZSGS,FTSGS
      double precision  dpcoal,drcoal,ecritl
      COMMON/SOFT/PXSGS(MAXSTR,3),PYSGS(MAXSTR,3),PZSGS(MAXSTR,3),
     &     PESGS(MAXSTR,3),PMSGS(MAXSTR,3),GXSGS(MAXSTR,3),
     &     GYSGS(MAXSTR,3),GZSGS(MAXSTR,3),FTSGS(MAXSTR,3),
     &     K1SGS(MAXSTR,3),K2SGS(MAXSTR,3),NJSGS(MAXSTR)
cc      SAVE /SOFT/
      common /coal/dpcoal,drcoal,ecritl
cc      SAVE /coal/
      common /loclco/gxp(3),gyp(3),gzp(3),ftp(3),
     1     pxp(3),pyp(3),pzp(3),pep(3),pmp(3)
cc      SAVE /loclco/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &     K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &     PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      SAVE   
c      
      do 1001 ISG=1, NSG
         IOVER(ISG)=0
 1001 continue
C1     meson q coalesce with all available qbar:
      do 150 ISG=1,NSG
         if(NJSGS(ISG).ne.2.or.IOVER(ISG).eq.1) goto 150
C     DETERMINE CURRENT RELATIVE DISTANCE AND MOMENTUM:
         if(K2SGS(ISG,1).lt.0) then
            write(6,*) 'Antiquark appears in quark loop; stop'
            stop
         endif
c         
         do 1002 j=1,2
            ftp(j)=ftsgs(isg,j)
            gxp(j)=gxsgs(isg,j)
            gyp(j)=gysgs(isg,j)
            gzp(j)=gzsgs(isg,j)
            pxp(j)=pxsgs(isg,j)
            pyp(j)=pysgs(isg,j)
            pzp(j)=pzsgs(isg,j)
            pmp(j)=pmsgs(isg,j)
            pep(j)=pesgs(isg,j)
 1002    continue
         call locldr(2,drlocl)
         dr0=drlocl
c     dp0^2 defined as (p1+p2)^2-(m1+m2)^2:
         dp0=dsqrt(2*(pep(1)*pep(2)-pxp(1)*pxp(2)
     &        -pyp(1)*pyp(2)-pzp(1)*pzp(2)-pmp(1)*pmp(2)))
c
         do 120 JSG=1,NSG
c     skip default or unavailable antiquarks:
            if(JSG.eq.ISG.or.IOVER(JSG).eq.1) goto 120
            if(NJSGS(JSG).eq.2) then
               ipmin=2
               ipmax=2
            elseif(NJSGS(JSG).eq.3.and.K2SGS(JSG,1).lt.0) then
               ipmin=1
               ipmax=3
            else
               goto 120
            endif
            do 100 ip=ipmin,ipmax
               dplocl=dsqrt(2*(pep(1)*pesgs(jsg,ip)
     1              -pxp(1)*pxsgs(jsg,ip)
     2              -pyp(1)*pysgs(jsg,ip)
     3              -pzp(1)*pzsgs(jsg,ip)
     4              -pmp(1)*pmsgs(jsg,ip)))
c     skip if outside of momentum radius:
               if(dplocl.gt.dpcoal) goto 120
               ftp(2)=ftsgs(jsg,ip)
               gxp(2)=gxsgs(jsg,ip)
               gyp(2)=gysgs(jsg,ip)
               gzp(2)=gzsgs(jsg,ip)
               pxp(2)=pxsgs(jsg,ip)
               pyp(2)=pysgs(jsg,ip)
               pzp(2)=pzsgs(jsg,ip)
               pmp(2)=pmsgs(jsg,ip)
               pep(2)=pesgs(jsg,ip)
               call locldr(2,drlocl)
c     skip if outside of spatial radius:
               if(drlocl.gt.drcoal) goto 120
c     q_isg coalesces with qbar_jsg:
               if((dp0.gt.dpcoal.or.dr0.gt.drcoal)
     1              .or.(drlocl.lt.dr0)) then
                  dp0=dplocl
                  dr0=drlocl
                  call exchge(isg,2,jsg,ip)
               endif
 100        continue
 120     continue
         if(dp0.le.dpcoal.and.dr0.le.drcoal) IOVER(ISG)=1
 150  continue
c
C2     meson qbar coalesce with all available q:
      do 250 ISG=1,NSG
         if(NJSGS(ISG).ne.2.or.IOVER(ISG).eq.1) goto 250
C     DETERMINE CURRENT RELATIVE DISTANCE AND MOMENTUM:
         do 1003 j=1,2
            ftp(j)=ftsgs(isg,j)
            gxp(j)=gxsgs(isg,j)
            gyp(j)=gysgs(isg,j)
            gzp(j)=gzsgs(isg,j)
            pxp(j)=pxsgs(isg,j)
            pyp(j)=pysgs(isg,j)
            pzp(j)=pzsgs(isg,j)
            pmp(j)=pmsgs(isg,j)
            pep(j)=pesgs(isg,j)
 1003    continue
         call locldr(2,drlocl)
         dr0=drlocl
         dp0=dsqrt(2*(pep(1)*pep(2)-pxp(1)*pxp(2)
     &        -pyp(1)*pyp(2)-pzp(1)*pzp(2)-pmp(1)*pmp(2)))
c
         do 220 JSG=1,NSG
            if(JSG.eq.ISG.or.IOVER(JSG).eq.1) goto 220
            if(NJSGS(JSG).eq.2) then
               ipmin=1
               ipmax=1
            elseif(NJSGS(JSG).eq.3.and.K2SGS(JSG,1).gt.0) then
               ipmin=1
               ipmax=3
            else
               goto 220
            endif
            do 200 ip=ipmin,ipmax
               dplocl=dsqrt(2*(pep(2)*pesgs(jsg,ip)
     1              -pxp(2)*pxsgs(jsg,ip)
     2              -pyp(2)*pysgs(jsg,ip)
     3              -pzp(2)*pzsgs(jsg,ip)
     4              -pmp(2)*pmsgs(jsg,ip)))
c     skip if outside of momentum radius:
               if(dplocl.gt.dpcoal) goto 220
               ftp(1)=ftsgs(jsg,ip)
               gxp(1)=gxsgs(jsg,ip)
               gyp(1)=gysgs(jsg,ip)
               gzp(1)=gzsgs(jsg,ip)
               pxp(1)=pxsgs(jsg,ip)
               pyp(1)=pysgs(jsg,ip)
               pzp(1)=pzsgs(jsg,ip)
               pmp(1)=pmsgs(jsg,ip)
               pep(1)=pesgs(jsg,ip)
               call locldr(2,drlocl)
c     skip if outside of spatial radius:
               if(drlocl.gt.drcoal) goto 220
c     qbar_isg coalesces with q_jsg:
               if((dp0.gt.dpcoal.or.dr0.gt.drcoal)
     1              .or.(drlocl.lt.dr0)) then
                  dp0=dplocl
                  dr0=drlocl
                  call exchge(isg,1,jsg,ip)
               endif
 200        continue
 220     continue
         if(dp0.le.dpcoal.and.dr0.le.drcoal) IOVER(ISG)=1
 250  continue
c
C3     baryon q (antibaryon qbar) coalesce with all available q (qbar):
      do 350 ISG=1,NSG
         if(NJSGS(ISG).ne.3.or.IOVER(ISG).eq.1) goto 350
         ibaryn=K2SGS(ISG,1)
C     DETERMINE CURRENT RELATIVE DISTANCE AND MOMENTUM:
         do 1004 j=1,2
            ftp(j)=ftsgs(isg,j)
            gxp(j)=gxsgs(isg,j)
            gyp(j)=gysgs(isg,j)
            gzp(j)=gzsgs(isg,j)
            pxp(j)=pxsgs(isg,j)
            pyp(j)=pysgs(isg,j)
            pzp(j)=pzsgs(isg,j)
            pmp(j)=pmsgs(isg,j)
            pep(j)=pesgs(isg,j)
 1004    continue
         call locldr(2,drlocl)
         dr1(2)=drlocl
         dp1(2)=dsqrt(2*(pep(1)*pep(2)-pxp(1)*pxp(2)
     &        -pyp(1)*pyp(2)-pzp(1)*pzp(2)-pmp(1)*pmp(2)))
c
         ftp(2)=ftsgs(isg,3)
         gxp(2)=gxsgs(isg,3)
         gyp(2)=gysgs(isg,3)
         gzp(2)=gzsgs(isg,3)
         pxp(2)=pxsgs(isg,3)
         pyp(2)=pysgs(isg,3)
         pzp(2)=pzsgs(isg,3)
         pmp(2)=pmsgs(isg,3)
         pep(2)=pesgs(isg,3)
         call locldr(2,drlocl)
         dr1(3)=drlocl
         dp1(3)=dsqrt(2*(pep(1)*pep(2)-pxp(1)*pxp(2)
     &        -pyp(1)*pyp(2)-pzp(1)*pzp(2)-pmp(1)*pmp(2)))
c
         do 320 JSG=1,NSG
            if(JSG.eq.ISG.or.IOVER(JSG).eq.1) goto 320
            if(NJSGS(JSG).eq.2) then
               if(ibaryn.gt.0) then
                  ipmin=1
               else
                  ipmin=2
               endif
               ipmax=ipmin
            elseif(NJSGS(JSG).eq.3.and.
     1              (ibaryn*K2SGS(JSG,1)).gt.0) then
               ipmin=1
               ipmax=3
            else
               goto 320
            endif
            do 300 ip=ipmin,ipmax
               dplocl=dsqrt(2*(pep(1)*pesgs(jsg,ip)
     1              -pxp(1)*pxsgs(jsg,ip)
     2              -pyp(1)*pysgs(jsg,ip)
     3              -pzp(1)*pzsgs(jsg,ip)
     4              -pmp(1)*pmsgs(jsg,ip)))
c     skip if outside of momentum radius:
               if(dplocl.gt.dpcoal) goto 320
               ftp(2)=ftsgs(jsg,ip)
               gxp(2)=gxsgs(jsg,ip)
               gyp(2)=gysgs(jsg,ip)
               gzp(2)=gzsgs(jsg,ip)
               pxp(2)=pxsgs(jsg,ip)
               pyp(2)=pysgs(jsg,ip)
               pzp(2)=pzsgs(jsg,ip)
               pmp(2)=pmsgs(jsg,ip)
               pep(2)=pesgs(jsg,ip)
               call locldr(2,drlocl)
c     skip if outside of spatial radius:
               if(drlocl.gt.drcoal) goto 320
c     q_isg may coalesce with q_jsg for a baryon:
               ipi=0
               if(dp1(2).gt.dpcoal.or.dr1(2).gt.drcoal) then
                  ipi=2
                  if((dp1(3).gt.dpcoal.or.dr1(3).gt.drcoal)
     1                 .and.dr1(3).gt.dr1(2)) ipi=3
               elseif(dp1(3).gt.dpcoal.or.dr1(3).gt.drcoal) then
                  ipi=3
               elseif(dr1(2).lt.dr1(3)) then
                  if(drlocl.lt.dr1(3)) ipi=3
               elseif(dr1(3).le.dr1(2)) then
                  if(drlocl.lt.dr1(2)) ipi=2
               endif
               if(ipi.ne.0) then
                  dp1(ipi)=dplocl
                  dr1(ipi)=drlocl
                  call exchge(isg,ipi,jsg,ip)
               endif
 300        continue
 320     continue
         if(dp1(2).le.dpcoal.and.dr1(2).le.drcoal
     1        .and.dp1(3).le.dpcoal.and.dr1(3).le.drcoal)
     2        IOVER(ISG)=1
 350  continue
c      
      RETURN
      END

c=======================================================================
      SUBROUTINE exchge(isg,ipi,jsg,ipj)
c
      implicit double precision  (a-h, o-z)
      PARAMETER (MAXSTR=150001)
      COMMON/SOFT/PXSGS(MAXSTR,3),PYSGS(MAXSTR,3),PZSGS(MAXSTR,3),
     &     PESGS(MAXSTR,3),PMSGS(MAXSTR,3),GXSGS(MAXSTR,3),
     &     GYSGS(MAXSTR,3),GZSGS(MAXSTR,3),FTSGS(MAXSTR,3),
     &     K1SGS(MAXSTR,3),K2SGS(MAXSTR,3),NJSGS(MAXSTR)
cc      SAVE /SOFT/
      SAVE   
c
      k1=K1SGS(isg,ipi)
      k2=K2SGS(isg,ipi)
      px=PXSGS(isg,ipi)
      py=PYSGS(isg,ipi)
      pz=PZSGS(isg,ipi)
      pe=PESGS(isg,ipi)
      pm=PMSGS(isg,ipi)
      gx=GXSGS(isg,ipi)
      gy=GYSGS(isg,ipi)
      gz=GZSGS(isg,ipi)
      ft=FTSGS(isg,ipi)
      K1SGS(isg,ipi)=K1SGS(jsg,ipj)
      K2SGS(isg,ipi)=K2SGS(jsg,ipj)
      PXSGS(isg,ipi)=PXSGS(jsg,ipj)
      PYSGS(isg,ipi)=PYSGS(jsg,ipj)
      PZSGS(isg,ipi)=PZSGS(jsg,ipj)
      PESGS(isg,ipi)=PESGS(jsg,ipj)
      PMSGS(isg,ipi)=PMSGS(jsg,ipj)
      GXSGS(isg,ipi)=GXSGS(jsg,ipj)
      GYSGS(isg,ipi)=GYSGS(jsg,ipj)
      GZSGS(isg,ipi)=GZSGS(jsg,ipj)
      FTSGS(isg,ipi)=FTSGS(jsg,ipj)
      K1SGS(jsg,ipj)=k1
      K2SGS(jsg,ipj)=k2
      PXSGS(jsg,ipj)=px
      PYSGS(jsg,ipj)=py
      PZSGS(jsg,ipj)=pz
      PESGS(jsg,ipj)=pe
      PMSGS(jsg,ipj)=pm
      GXSGS(jsg,ipj)=gx
      GYSGS(jsg,ipj)=gy
      GZSGS(jsg,ipj)=gz
      FTSGS(jsg,ipj)=ft
c
      RETURN
      END

c=======================================================================
      SUBROUTINE locldr(icall,drlocl)
c
      implicit double precision (a-h, o-z)
      dimension ftp0(3),pxp0(3),pyp0(3),pzp0(3),pep0(3)
      common /loclco/gxp(3),gyp(3),gzp(3),ftp(3),
     1     pxp(3),pyp(3),pzp(3),pep(3),pmp(3)
cc      SAVE /loclco/
      common /prtn23/ gxp0(3),gyp0(3),gzp0(3),ft0fom
cc      SAVE /prtn23/
      common /lor/ enenew, pxnew, pynew, pznew
cc      SAVE /lor/
      SAVE   
c     for 2-body kinematics:
      if(icall.eq.2) then
         etot=pep(1)+pep(2)
         bex=(pxp(1)+pxp(2))/etot
         bey=(pyp(1)+pyp(2))/etot
         bez=(pzp(1)+pzp(2))/etot
c     boost the reference frame down by beta to get to the pair rest frame:
         do 1001 j=1,2
            beta2 = bex ** 2 + bey ** 2 + bez ** 2
            gam = 1.d0 / dsqrt(1.d0 - beta2)
            if(beta2.ge.0.9999999999999d0) then
               write(6,*) '4',pxp(1),pxp(2),pyp(1),pyp(2),
     1              pzp(1),pzp(2),pep(1),pep(2),pmp(1),pmp(2),
     2          dsqrt(pxp(1)**2+pyp(1)**2+pzp(1)**2+pmp(1)**2)/pep(1),
     3          dsqrt(pxp(1)**2+pyp(1)**2+pzp(1)**2)/pep(1)
               write(6,*) '4a',pxp(1)+pxp(2),pyp(1)+pyp(2),
     1              pzp(1)+pzp(2),etot
               write(6,*) '4b',bex,bey,bez,beta2,gam
            endif
c
            call lorenz(ftp(j),gxp(j),gyp(j),gzp(j),bex,bey,bez)
            gxp0(j)=pxnew
            gyp0(j)=pynew
            gzp0(j)=pznew
            ftp0(j)=enenew
            call lorenz(pep(j),pxp(j),pyp(j),pzp(j),bex,bey,bez)
            pxp0(j)=pxnew
            pyp0(j)=pynew
            pzp0(j)=pznew
            pep0(j)=enenew
 1001    continue
c     
         if(ftp0(1).ge.ftp0(2)) then
            ilate=1
            iearly=2
         else
            ilate=2
            iearly=1
         endif
         ft0fom=ftp0(ilate)
c     
         dt0=ftp0(ilate)-ftp0(iearly)
         gxp0(iearly)=gxp0(iearly)+pxp0(iearly)/pep0(iearly)*dt0
         gyp0(iearly)=gyp0(iearly)+pyp0(iearly)/pep0(iearly)*dt0
         gzp0(iearly)=gzp0(iearly)+pzp0(iearly)/pep0(iearly)*dt0
         drlocl=dsqrt((gxp0(ilate)-gxp0(iearly))**2
     1        +(gyp0(ilate)-gyp0(iearly))**2
     2        +(gzp0(ilate)-gzp0(iearly))**2)
c     for 3-body kinematics, used for baryons formation:
      elseif(icall.eq.3) then
         etot=pep(1)+pep(2)+pep(3)
         bex=(pxp(1)+pxp(2)+pxp(3))/etot
         bey=(pyp(1)+pyp(2)+pyp(3))/etot
         bez=(pzp(1)+pzp(2)+pzp(3))/etot
         beta2 = bex ** 2 + bey ** 2 + bez ** 2
         gam = 1.d0 / dsqrt(1.d0 - beta2)
         if(beta2.ge.0.9999999999999d0) then
            write(6,*) '5',bex,bey,bez,beta2,gam
         endif
c     boost the reference frame down by beta to get to the 3-parton rest frame:
         do 1002 j=1,3
            call lorenz(ftp(j),gxp(j),gyp(j),gzp(j),bex,bey,bez)
            gxp0(j)=pxnew
            gyp0(j)=pynew
            gzp0(j)=pznew
            ftp0(j)=enenew
            call lorenz(pep(j),pxp(j),pyp(j),pzp(j),bex,bey,bez)
            pxp0(j)=pxnew
            pyp0(j)=pynew
            pzp0(j)=pznew
            pep0(j)=enenew
 1002    continue
c     
         if(ftp0(1).gt.ftp0(2)) then
            ilate=1
            if(ftp0(3).gt.ftp0(1)) ilate=3
         else
            ilate=2
            if(ftp0(3).ge.ftp0(2)) ilate=3
         endif
         ft0fom=ftp0(ilate)
c     
         if(ilate.eq.1) then
            imin=2
            imax=3
            istep=1
         elseif(ilate.eq.2) then
            imin=1
            imax=3
            istep=2
         elseif(ilate.eq.3) then
            imin=1
            imax=2
            istep=1
         endif
c     
         do 1003 iearly=imin,imax,istep
            dt0=ftp0(ilate)-ftp0(iearly)
            gxp0(iearly)=gxp0(iearly)+pxp0(iearly)/pep0(iearly)*dt0
            gyp0(iearly)=gyp0(iearly)+pyp0(iearly)/pep0(iearly)*dt0
            gzp0(iearly)=gzp0(iearly)+pzp0(iearly)/pep0(iearly)*dt0
 1003    continue
      endif
c
      RETURN
      END

c=======================================================================
        subroutine hoscar
c
        parameter (MAXSTR=150001,AMN=0.939457,AMP=0.93828)
        character*8 code, reffra, FRAME
        character*25 amptvn
        common/snn/efrm,npart1,npart2,epsiPz,epsiPt,PZPROJ,PZTARG
cc      SAVE /snn/
        common /lastt/itimeh,bimp 
cc      SAVE /lastt/
        COMMON/hbt/lblast(MAXSTR),xlast(4,MAXSTR),plast(4,MAXSTR),nlast
cc      SAVE /hbt/
        common/oscar1/iap,izp,iat,izt
cc      SAVE /oscar1/
        common/oscar2/FRAME,amptvn
cc      SAVE /oscar2/
        SAVE   
        data nff/0/
c
c       file header
        if(nff.eq.0) then
           write (19, 101) 'OSCAR1997A'
           write (19, 111) 'final_id_p_x'
           code = 'AMPT'
           if(FRAME.eq.'CMS') then
              reffra = 'nncm'
              xmp=(amp*izp+amn*(iap-izp))/iap
              xmt=(amp*izt+amn*(iat-izt))/iat
              ebeam=(efrm**2-xmp**2-xmt**2)/2./xmt
           elseif(FRAME.eq.'LAB') then
              reffra = 'lab'
              ebeam=efrm
           else
              reffra = 'unknown'
              ebeam=0.
           endif
           ntestp = 1
           write (19, 102) code, amptvn, iap, izp, iat, izt,
     &        reffra, ebeam, ntestp
           nff = 1
           ievent = 1
           phi = 0.
           if(FRAME.eq.'CMS') write(19,112) efrm
        endif
c       comment
c       event header
        write (19, 103) ievent, nlast, bimp, phi
c       particles
        do 99 i = 1, nlast
           ene=sqrt(plast(1,i)**2+plast(2,i)**2+plast(3,i)**2
     1          +plast(4,i)**2)
           write (19, 104) i, INVFLV(lblast(i)), plast(1,i),
     1          plast(2,i),plast(3,i),ene,plast(4,i),
     2          xlast(1,i),xlast(2,i),xlast(3,i),xlast(4,i)
 99     continue
        ievent = ievent + 1
 101        format (a10)
 111        format (a12)
 102        format (a4,1x,a20,1x,'(', i3, ',', i3, ')+(', i3, ',', 
     &           i3, ')', 2x, a4, 2x, e10.4, 2x, i8)
 103        format (i10, 2x, i10, 2x, f8.3, 2x, f8.3)
 104        format (i10, 2x, i10, 2x, 9(e12.6, 2x))
 112        format ('# Center-of-mass energy/nucleon-pair is',
     & f12.3,'GeV')
c
        return
        end

c=======================================================================
        subroutine getnp

        PARAMETER (MAXSTR=150001)
        COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
cc      SAVE /HMAIN1/
        COMMON/HMAIN2/KATT(MAXSTR,4),PATT(MAXSTR,4)
cc      SAVE /HMAIN2/
        COMMON /HPARNT/HIPR1(100), IHPR2(50), HINT1(100), IHNT2(50)
cc      SAVE /HPARNT/
        common/snn/efrm,npart1,npart2,epsiPz,epsiPt,PZPROJ,PZTARG
cc      SAVE /snn/
        SAVE   

        if(NATT.eq.0) then
           npart1=0
           npart2=0
           return
        endif
c
        PZPROJ=SQRT(HINT1(6)**2-HINT1(8)**2)
        PZTARG=SQRT(HINT1(7)**2-HINT1(9)**2)
        epsiPz=0.01
clin-9/2011-add Pt tolerance in determining spectator nucleons
c     (affect string melting runs when LAB frame is used):
        epsiPt=1e-6
c
        nspec1=0
        nspec2=0
        DO 1000 I = 1, NATT
clin-9/2011 determine spectator nucleons consistently
c           if((KATT(I,1).eq.2112.or.KATT(I,1).eq.2212)
c     1          .and.PATT(I, 1).eq.0.and.PATT(I, 2).eq.0) then
           if((KATT(I,1).eq.2112.or.KATT(I,1).eq.2212)
     1          .and.abs(PATT(I, 1)).le.epsiPt
     2          .and.abs(PATT(I, 2)).le.epsiPt) then
              if(PATT(I, 3).gt.amax1(0.,PZPROJ-epsiPz)) then
                 nspec1=nspec1+1
              elseif(PATT(I, 3).lt.(-PZTARG+epsiPz)) then
                 nspec2=nspec2+1
              endif
           endif
 1000   CONTINUE
        npart1=IHNT2(1)-nspec1
        npart2=IHNT2(3)-nspec2

        return
        end

c=======================================================================
c     2/18/03 use PYTHIA to decay eta,rho,omega,k*,phi and Delta
c     4/2012 added pi0 decay flag: 
c       ipion=0: resonance or pi0 in lb(i1); >0: pi0 in lpion(ipion).
        subroutine resdec(i1,nt,nnn,wid,idecay,ipion)

        PARAMETER (hbarc=0.19733)
        PARAMETER (AK0=0.498,APICH=0.140,API0=0.135,AN=0.940,ADDM=0.02)
        PARAMETER (MAXSTR=150001, MAXR=1)
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &       IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
        COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
cc      SAVE /LUJETS/
        COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
cc      SAVE /LUDAT1/
        COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)
cc      SAVE /LUDAT2/
        COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)
cc      SAVE /LUDAT3/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
        COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
        COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
        COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
        common/resdcy/NSAV,iksdcy
cc      SAVE /resdcy/
        common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1       px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
        EXTERNAL IARFLV, INVFLV
        COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
        COMMON/RNDF77/NSEED
        COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1       dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2       dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
cc      SAVE /RNDF77/
        common/phidcy/iphidcy,pttrig,ntrig,maxmiss,ipi0dcy
        SAVE   
        irun=idecay
clin-4/2012 for option of pi0 decay:
        if(nt.eq.ntmax.and.ipi0dcy.eq.1
     &       .and.((lb1.eq.4.and.ipion.eq.0).or.ipion.ge.1)) then
           kf=111
c        if(lb1.eq.0.or.lb1.eq.25.or.lb1.eq.26.or.lb1.eq.27
        elseif(lb1.eq.0.or.lb1.eq.25.or.lb1.eq.26.or.lb1.eq.27
     &       .or.lb1.eq.28.or.lb1.eq.29.or.iabs(lb1).eq.30
     &       .or.lb1.eq.24.or.(iabs(lb1).ge.6.and.iabs(lb1).le.9) 
     &       .or.iabs(lb1).eq.16) then
           kf=INVFLV(lb1)
        else
           return
        endif
c
        IP=1
c     label as undecayed and the only particle in the record:
        N=1
        K(IP,1)=1
        K(IP,3)=0
        K(IP,4)=0
        K(IP,5)=0
c
        K(IP,2)=kf
clin-4/2012 for option of pi0 decay:
        if(ipion.eq.0) then
c
        P(IP,1)=px1
        P(IP,2)=py1
        P(IP,3)=pz1
c        em1a=em1
c     eta or omega in ART may be below or too close to (pi+pi-pi0) mass, 
c     causing LUDECY error,thus increase their mass ADDM above this thresh,
c     noting that rho (m=0.281) too close to 2pi thrshold fails to decay:
        if((lb1.eq.0.or.lb1.eq.28).and.em1.lt.(2*APICH+API0+ADDM)) then
           em1=2*APICH+API0+ADDM
c     rho
        elseif(lb1.ge.25.and.lb1.le.27.and.em1.lt.(2*APICH+ADDM)) then
           em1=2*APICH+ADDM
c     K*
        elseif(iabs(lb1).eq.30.and.em1.lt.(APICH+AK0+ADDM)) then
           em1=APICH+AK0+ADDM
c     Delta created in ART may be below (n+pich) mass, causing LUDECY error:
        elseif(iabs(lb1).ge.6.and.iabs(lb1).le.9
     1          .and.em1.lt.(APICH+AN+ADDM)) then
           em1=APICH+AN+ADDM
        endif
c        if(em1.ge.(em1a+0.01)) write (6,*) 
c     1       'Mass increase in resdec():',nt,em1-em1a,lb1
        e1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
        P(IP,4)=e1
        P(IP,5)=em1
clin-5/2008:
        dpdecp=dpertp(i1)
clin-4/2012 for option of pi0 decay:
        elseif(nt.eq.ntmax.and.ipi0dcy.eq.1.and.ipion.ge.1) then        
           P(IP,1)=PPION(1,ipion,IRUN)
           P(IP,2)=PPION(2,ipion,IRUN)
           P(IP,3)=PPION(3,ipion,IRUN)
           P(IP,5)=EPION(ipion,IRUN)
           P(IP,4)=SQRT(P(IP,5)**2+P(IP,1)**2+P(IP,2)**2+P(IP,3)**2)
           dpdecp=dppion(ipion,IRUN)
ctest off
c           write(99,*) P(IP,4), P(IP,5), dpdecp, ipion, wid
        else
           print *, 'stopped in resdec() a'
           stop
        endif
c
        call ludecy(IP)
c     add decay time to daughter's formation time at the last timestep:
        if(nt.eq.ntmax) then
           tau0=hbarc/wid
           taudcy=tau0*(-1.)*alog(1.-RANART(NSEED))
           ndaut=n-nsav
           if(ndaut.le.1) then
              write(10,*) 'note: ndaut(<1)=',ndaut
              call lulist(2)
              stop
            endif
c     lorentz boost:
clin-4/2012 for option of pi0 decay:
            if(ipion.eq.0) then
               taudcy=taudcy*e1/em1
               tfnl=tfnl+taudcy
               xfnl=xfnl+px1/e1*taudcy
               yfnl=yfnl+py1/e1*taudcy
               zfnl=zfnl+pz1/e1*taudcy
            elseif(ipion.ge.1) then
               taudcy=taudcy*P(IP,4)/P(IP,5)
               tfnl=tfdpi(ipion,IRUN)+taudcy
               xfnl=RPION(1,ipion,IRUN)+P(IP,1)/P(IP,4)*taudcy
               yfnl=RPION(2,ipion,IRUN)+P(IP,2)/P(IP,4)*taudcy
               zfnl=RPION(3,ipion,IRUN)+P(IP,3)/P(IP,4)*taudcy
            else
               print *, 'stopped in resdec() b',ipion,wid,P(ip,4)
               stop
            endif
c     at the last timestep, assign rho, K0S or eta (decay daughter)
c     to lb(i1) only (not to lpion) in order to decay them again:
clin-4/2012 for option of pi0 decay:
c           if(n.ge.(nsav+2)) then
           if(n.ge.(nsav+2).and.ipion.eq.0) then
              do 1001 idau=nsav+2,n
                 kdaut=K(idau,2)
                 if(kdaut.eq.221.or.kdaut.eq.113
     1                .or.kdaut.eq.213.or.kdaut.eq.-213
     2                .or.kdaut.eq.310) then
c     switch idau and i1(nsav+1):
                    ksave=kdaut
                    pxsave=p(idau,1)
                    pysave=p(idau,2)
                    pzsave=p(idau,3)
                    esave=p(idau,4)
                    xmsave=p(idau,5)
                    K(idau,2)=K(nsav+1,2)
                    p(idau,1)=p(nsav+1,1)
                    p(idau,2)=p(nsav+1,2)
                    p(idau,3)=p(nsav+1,3)
                    p(idau,4)=p(nsav+1,4)
                    p(idau,5)=p(nsav+1,5)
                    K(nsav+1,2)=ksave
                    p(nsav+1,1)=pxsave
                    p(nsav+1,2)=pysave
                    p(nsav+1,3)=pzsave
                    p(nsav+1,4)=esave
                    p(nsav+1,5)=xmsave
c     note: phi decay may produce rho, K0s or eta, N*(1535) decay may produce 
c     eta, but only one daughter may be rho, K0s or eta:
                    goto 111
                 endif
 1001         continue
           endif
 111       continue
c     
           enet=0.
           do 1002 idau=nsav+1,n
              enet=enet+p(idau,4)
 1002      continue
c           if(abs(enet-e1).gt.0.02) 
c     1          write(93,*) 'resdec(): nt=',nt,enet-e1,lb1
        endif

        do 1003 idau=nsav+1,n
           kdaut=K(idau,2)
           lbdaut=IARFLV(kdaut)
c     K0S and K0L are named K+/K- during hadron cascade, and only 
c     at the last timestep they keep their real LB # before output;
c     K0/K0bar (from K* decay) converted to K0S and K0L at the last timestep:
           if(nt.eq.ntmax.and.(kdaut.eq.130.or.kdaut.eq.310
     1          .or.iabs(kdaut).eq.311)) then
              if(kdaut.eq.130) then
                 lbdaut=22
              elseif(kdaut.eq.310) then
                 lbdaut=24
              elseif(iabs(kdaut).eq.311) then
                 if(RANART(NSEED).lt.0.5) then
                    lbdaut=22
                 else
                    lbdaut=24
                 endif
              endif
           endif
c
           if(idau.eq.(nsav+1)) then
clin-4/2012 for option of pi0 decay:
              if(ipion.eq.0) then
                 LB(i1)=lbdaut
                 E(i1)=p(idau,5)
                 px1n=p(idau,1)
                 py1n=p(idau,2)
                 pz1n=p(idau,3)
clin-5/2008:
                 dp1n=dpdecp
              elseif(ipion.ge.1) then
                 LPION(ipion,IRUN)=lbdaut
                 EPION(ipion,IRUN)=p(idau,5)
                 PPION(1,ipion,IRUN)=p(idau,1)
                 PPION(2,ipion,IRUN)=p(idau,2)
                 PPION(3,ipion,IRUN)=p(idau,3)
                 RPION(1,ipion,IRUN)=xfnl
                 RPION(2,ipion,IRUN)=yfnl
                 RPION(3,ipion,IRUN)=zfnl
                 tfdpi(ipion,IRUN)=tfnl
                 dppion(ipion,IRUN)=dpdecp
              endif
c
           else
              nnn=nnn+1
              LPION(NNN,IRUN)=lbdaut
              EPION(NNN,IRUN)=p(idau,5)
              PPION(1,NNN,IRUN)=p(idau,1)
              PPION(2,NNN,IRUN)=p(idau,2)
              PPION(3,NNN,IRUN)=p(idau,3)
              RPION(1,NNN,IRUN)=xfnl
              RPION(2,NNN,IRUN)=yfnl
              RPION(3,NNN,IRUN)=zfnl
              tfdpi(NNN,IRUN)=tfnl
clin-5/2008:
              dppion(NNN,IRUN)=dpdecp
           endif
 1003   continue
        return
        end

c=======================================================================
        subroutine inidcy

        COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
cc      SAVE /LUJETS/
        common/resdcy/NSAV,iksdcy
cc      SAVE /resdcy/
        SAVE   
        N=1
        NSAV=N
        return
        end

c=======================================================================
clin-6/06/02 local parton freezeout motivated from critical density:
        subroutine local(t)
c
        implicit double precision  (a-h, o-z)
        PARAMETER (MAXPTN=400001)
        PARAMETER (r0=1d0)
        COMMON /para1/ mul
cc      SAVE /para1/
        COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &       PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &       XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
        common /frzprc/ 
     &       gxfrz(MAXPTN), gyfrz(MAXPTN), gzfrz(MAXPTN), ftfrz(MAXPTN),
     &       pxfrz(MAXPTN), pyfrz(MAXPTN), pzfrz(MAXPTN), efrz(MAXPTN),
     &       xmfrz(MAXPTN), 
     &       tfrz(302), ifrz(MAXPTN), idfrz(MAXPTN), itlast
cc      SAVE /frzprc/
        common /prec4/ vx(MAXPTN), vy(MAXPTN), vz(MAXPTN)
cc      SAVE /prec4/
        common /prec5/ eta(MAXPTN), rap(MAXPTN), tau(MAXPTN)
cc      SAVE /prec5/
        common /coal/dpcoal,drcoal,ecritl
cc      SAVE /coal/
        SAVE   
c
      do 1001 it=1,301
         if(t.ge.tfrz(it).and.t.lt.tfrz(it+1)) then
            if(it.eq.itlast) then
               return
            else
               itlast=it
               goto 50
            endif
         endif
 1001 continue
      write(1,*) 'local time out of range in LOCAL, stop',t,it
      stop
 50   continue
c
      do 200 ip=1,mul
c     skip partons which have frozen out:
         if(ifrz(ip).eq.1) goto 200
         if(it.eq.301) then
c     freezeout all the left partons beyond the time of 3000 fm:
            etcrit=1d6
            goto 150
         else
c     freezeout when transverse energy density < etcrit:
            etcrit=(ecritl*2d0/3d0)
         endif
c     skip partons which have not yet formed:
         if(t.lt.FT5(ip)) goto 200
         rap0=rap(ip)
         eta0=eta(ip)
         x0=GX5(ip)+vx(ip)*(t-FT5(ip))
         y0=GY5(ip)+vy(ip)*(t-FT5(ip))
         detdy=0d0
         do 100 itest=1,mul
c     skip self and partons which have not yet formed:
            if(itest.eq.ip.or.t.lt.FT5(itest)) goto 100
            ettest=eta(itest)
            xtest=GX5(itest)+vx(itest)*(t-FT5(itest))
            ytest=GY5(itest)+vy(itest)*(t-FT5(itest))
            drt=sqrt((xtest-x0)**2+(ytest-y0)**2)
c     count partons within drt<1 and -1<(eta-eta0)<1:
            if(dabs(ettest-eta0).le.1d0.and.drt.le.r0) 
     1           detdy=detdy+dsqrt(PX5(itest)**2+PY5(itest)**2
     2           +XMASS5(itest)**2)*0.5d0
 100     continue
         detdy=detdy*(dcosh(eta0)**2)/(t*3.1416d0*r0**2*dcosh(rap0))
c     when density is below critical density for phase transition, freeze out:
 150     if(detdy.le.etcrit) then
            ifrz(ip)=1
            idfrz(ip)=ITYP5(ip)
            pxfrz(ip)=PX5(ip)
            pyfrz(ip)=PY5(ip)
            pzfrz(ip)=PZ5(ip)
            efrz(ip)=E5(ip)
            xmfrz(ip)=XMASS5(ip)
            if(t.gt.FT5(ip)) then
               gxfrz(ip)=x0
               gyfrz(ip)=y0
               gzfrz(ip)=GZ5(ip)+vz(ip)*(t-FT5(ip))
               ftfrz(ip)=t
            else
c     if this freezeout time < formation time, use formation time & positions.
c     This ensures the recovery of default hadron when e_crit=infty:
               gxfrz(ip)=GX5(ip)
               gyfrz(ip)=GY5(ip)
               gzfrz(ip)=GZ5(ip)
               ftfrz(ip)=FT5(ip)
            endif
         endif
 200  continue
c
        return
        end

c=======================================================================
clin-6/06/02 initialization for local parton freezeout
        subroutine inifrz
c
        implicit double precision  (a-h, o-z)
        PARAMETER (MAXPTN=400001)
        common /ilist5/ ct(MAXPTN), ot(MAXPTN), tlarge
cc      SAVE /ilist5/
        common /frzprc/ 
     &       gxfrz(MAXPTN), gyfrz(MAXPTN), gzfrz(MAXPTN), ftfrz(MAXPTN),
     &       pxfrz(MAXPTN), pyfrz(MAXPTN), pzfrz(MAXPTN), efrz(MAXPTN),
     &       xmfrz(MAXPTN), 
     &       tfrz(302), ifrz(MAXPTN), idfrz(MAXPTN), itlast
cc      SAVE /frzprc/
        SAVE   
c
c     for freezeout time 0-10fm, use interval of 0.1fm; 
c     for 10-100fm, use interval of 1fm; 
c     for 100-1000fm, use interval of 10fm; 
c     for 1000-3000fm, use interval of 100fm: 
        step1=0.1d0
        step2=1d0
        step3=10d0
        step4=100d0
c     
        do 1001 it=1,101
           tfrz(it)=0d0+dble(it-1)*step1
 1001 continue
        do 1002 it=102,191
           tfrz(it)=10d0+dble(it-101)*step2
 1002   continue
        do 1003 it=192,281
           tfrz(it)=100d0+dble(it-191)*step3
 1003   continue
        do 1004 it=282,301
           tfrz(it)=1000d0+dble(it-281)*step4
 1004   continue
        tfrz(302)=tlarge
c
        return
        end

clin-5/2009 v2 analysis
c=======================================================================
c     idd=0,1,2,3 specifies different subroutines for partonic flow analysis.
      subroutine flowp(idd)
c
        implicit double precision  (a-h, o-z)
        real dt
        parameter (MAXPTN=400001)
csp
        parameter (bmt=0.05d0)
        dimension nlfile(3),nsfile(3),nmfile(3)
c
        dimension v2pp(3),xnpp(3),v2psum(3),v2p2sm(3),nfile(3)
        dimension tsp(31),v2pevt(3),v2pavg(3),varv2p(3)
        common /ilist1/
     &     iscat, jscat, next(MAXPTN), last(MAXPTN),
     &     ictype, icsta(MAXPTN),
     &     nic(MAXPTN), icels(MAXPTN)
cc      SAVE /ilist1/
        COMMON /para1/ mul
cc      SAVE /para1/
        COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &       PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &       XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
        COMMON /pflow/ v2p(30,3),xnpart(30,3),etp(30,3),
     1       s2p(30,3),v2p2(30,3),nevt(30)
cc      SAVE /pflow/
        COMMON /pflowf/ v2pf(30,3),xnpf(30,3),etpf(30,3),
     1                 xncoll(30),s2pf(30,3),v2pf2(30,3)
cc      SAVE /pflowf/
        COMMON /pfrz/ v2pfrz(30,3),xnpfrz(30,3),etpfrz(30,3),
     1       s2pfrz(30,3),v2p2fz(30,3),tscatt(31),
     2       nevtfz(30),iscatt(30)
cc      SAVE /pfrz/
        COMMON /hflow/ v2h(30,3),xnhadr(30,3),eth(30,3),
     1 v2h2(30,3),s2h(30,3)
cc      SAVE /hflow/
        COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
cc      SAVE itimep,iaevtp,v2pp,xnpp,v2psum,v2p2sm
cc      SAVE nfile,itanim,nlfile,nsfile,nmfile
        SAVE   
csp
        dimension etpl(30,3),etps(30,3),etplf(30,3),etpsf(30,3),
     &       etlfrz(30,3),etsfrz(30,3),
     &       xnpl(30,3),xnps(30,3),xnplf(30,3),xnpsf(30,3),
     &       xnlfrz(30,3),xnsfrz(30,3),
     &       v2pl(30,3),v2ps(30,3),v2plf(30,3),v2psf(30,3),
     &       s2pl(30,3),s2ps(30,3),s2plf(30,3),s2psf(30,3),
     &       DMYil(50,3),DMYfl(50,3),
     &       DMYis(50,3),DMYfs(50,3)
        data tsp/0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
     &       1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
     &       2  , 3,   4,   5,   6,   7,   8,   9,   10,  20,  30/
c     idd=0: initialization for flow analysis, called by artdri.f:
        if(idd.eq.0) then        
           nfile(1)=60
           nfile(2)=64
           nfile(3)=20
           OPEN (nfile(1),FILE='ana1/v2p.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(1)+1, 
     1 FILE = 'ana1/v2p-formed.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(1)+2, 
     1 FILE = 'ana1/v2p-active.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(1)+3, 
     1 FILE = 'ana1/v2ph.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(2),FILE='ana1/v2p-y2.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(2)+1, 
     1 FILE = 'ana1/v2p-formed2.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(2)+2, 
     1 FILE = 'ana1/v2p-active2.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(2)+3, 
     1 FILE = 'ana1/v2ph-y2.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(3),FILE='ana1/v2p-y1.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(3)+1, 
     1 FILE = 'ana1/v2p-formed1.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(3)+2, 
     1 FILE = 'ana1/v2p-active1.dat', STATUS = 'UNKNOWN')
           OPEN (nfile(3)+3, 
     1 FILE = 'ana1/v2ph-y1.dat', STATUS = 'UNKNOWN')
           OPEN (49, FILE = 'ana1/v2p-ebe.dat', STATUS = 'UNKNOWN')
           write(49, *) '    ievt,  v2p,  v2p_y2,   v2p_y1'
c
           OPEN (59, FILE = 'ana1/v2h.dat', STATUS = 'UNKNOWN')
           OPEN (68, FILE = 'ana1/v2h-y2.dat', STATUS = 'UNKNOWN')
           OPEN (69, FILE = 'ana1/v2h-y1.dat', STATUS = 'UNKNOWN')
           OPEN (88, FILE = 'ana1/v2h-ebe.dat', STATUS = 'UNKNOWN')
           write(88, *) '    ievt,  v2h,  v2h_y2,   v2h_y1'
csp07/05
           nlfile(1)=70
           nlfile(2)=72
           nlfile(3)=74
           OPEN (nlfile(1),FILE='ana1/mtl.dat', STATUS = 'UNKNOWN')
           OPEN (nlfile(1)+1, 
     1 FILE = 'ana1/mtl-formed.dat', STATUS = 'UNKNOWN')
           OPEN (nlfile(2),FILE='ana1/mtl-y2.dat', STATUS = 'UNKNOWN')
           OPEN (nlfile(2)+1, 
     1 FILE = 'ana1/mtl-formed2.dat', STATUS = 'UNKNOWN')
           OPEN (nlfile(3),FILE='ana1/mtl-y1.dat', STATUS = 'UNKNOWN')
           OPEN (nlfile(3)+1, 
     1 FILE = 'ana1/mtl-formed1.dat', STATUS = 'UNKNOWN')
           nsfile(1)=76
           nsfile(2)=78
           nsfile(3)=80
           OPEN (nsfile(1),FILE='ana1/mts.dat', STATUS = 'UNKNOWN')
           OPEN (nsfile(1)+1, 
     1 FILE = 'ana1/mts-formed.dat', STATUS = 'UNKNOWN')
           OPEN (nsfile(2),FILE='ana1/mts-y2.dat', STATUS = 'UNKNOWN')
           OPEN (nsfile(2)+1, 
     1 FILE = 'ana1/mts-formed2.dat', STATUS = 'UNKNOWN')
           OPEN (nsfile(3),FILE='ana1/mts-y1.dat', STATUS = 'UNKNOWN')
           OPEN (nsfile(3)+1, 
     1 FILE = 'ana1/mts-formed1.dat', STATUS = 'UNKNOWN')
           nmfile(1)=82
           nmfile(2)=83
           nmfile(3)=84
           OPEN (nmfile(1),FILE='ana1/Nmt.dat', STATUS = 'UNKNOWN')
           OPEN (nmfile(2),FILE='ana1/Nmt-y2.dat', STATUS = 'UNKNOWN')
           OPEN (nmfile(3),FILE='ana1/Nmt-y1.dat', STATUS = 'UNKNOWN')
clin-11/27/00 for animation:
           if(nevent.eq.1) then
c           OPEN (91, FILE = 'ana1/h-animate.dat', STATUS = 'UNKNOWN')
c           write(91,*) ntmax, dt
c           OPEN (92, FILE = 'ana1/p-animate.dat', STATUS = 'UNKNOWN')
           OPEN (93, FILE = 'ana1/p-finalft.dat', STATUS = 'UNKNOWN')
           endif
c
           itimep=0
           itanim=0
           iaevtp=0
csp
           do 1002 ii=1,50
              do 1001 iy=1,3
                 DMYil(ii,iy) = 0d0
                 DMYfl(ii,iy) = 0d0
                 DMYis(ii,iy) = 0d0
                 DMYfs(ii,iy) = 0d0
 1001         continue
 1002      continue
c
           do 1003 ii=1,31
              tscatt(ii)=0d0
 1003      continue
           do 1005 ii=1,30
              nevt(ii)=0
              xncoll(ii)=0d0
              nevtfz(ii)=0
              iscatt(ii)=0
              do 1004 iy=1,3
                 v2p(ii,iy)=0d0
                 v2p2(ii,iy)=0d0
                 s2p(ii,iy)=0d0
                 etp(ii,iy)=0d0
                 xnpart(ii,iy)=0d0
                 v2pf(ii,iy)=0d0
                 v2pf2(ii,iy)=0d0
                 s2pf(ii,iy)=0d0
                 etpf(ii,iy)=0d0
                 xnpf(ii,iy)=0d0
                 v2pfrz(ii,iy)=0d0
                 v2p2fz(ii,iy)=0d0
                 s2pfrz(ii,iy)=0d0
                 etpfrz(ii,iy)=0d0
                 xnpfrz(ii,iy)=0d0
csp07/05
                 etpl(ii,iy)=0d0
                 etps(ii,iy)=0d0
                 etplf(ii,iy)=0d0
                 etpsf(ii,iy)=0d0
                 etlfrz(ii,iy)=0d0
                 etsfrz(ii,iy)=0d0
              xnpl(ii,iy)=0d0
              xnps(ii,iy)=0d0
              xnplf(ii,iy)=0d0
              xnpsf(ii,iy)=0d0
              xnlfrz(ii,iy)=0d0
              xnsfrz(ii,iy)=0d0
              v2pl(ii,iy)=0d0
              v2ps(ii,iy)=0d0
              v2plf(ii,iy)=0d0
              v2psf(ii,iy)=0d0
              s2pl(ii,iy)=0d0
              s2ps(ii,iy)=0d0
              s2plf(ii,iy)=0d0
              s2psf(ii,iy)=0d0
 1004      continue
 1005   continue
           do 1006 iy=1,3
              v2pevt(iy)=0d0
              v2pavg(iy)=0d0
              varv2p(iy)=0d0
              v2pp(iy)=0.d0
              xnpp(iy)=0d0
              v2psum(iy)=0.d0
              v2p2sm(iy)=0.d0
 1006      continue
c     idd=1: calculate parton elliptic flow, called by zpc.f:
        else if(idd.eq.1) then        
           t2time = FT5(iscat)
           do 1008 ianp = 1, 30
              if (t2time.lt.tsp(ianp+1).and.t2time.ge.tsp(ianp)) then
c     write flow info only once at each fixed time:
                 xncoll(ianp)=xncoll(ianp)+1d0
c     to prevent an earlier t2time comes later in the same event 
c     and mess up nevt:
                 if(ianp.le.itimep.and.iaevt.eq.iaevtp) goto 101
                 nevt(ianp)=nevt(ianp)+1
                 tscatt(ianp+1)=t2time
                 iscatt(ianp)=1
                 nevtfz(ianp)=nevtfz(ianp)+1
                 do 100 I=1,mul
            ypartn=0.5d0*dlog((E5(i)+PZ5(i))/(E5(i)-PZ5(i)+1.d-8))
                    pt2=PX5(I)**2+PY5(I)**2
ctest off: initial (pt,y) and (x,y) distribution:
c                    idtime=1
c                    if(ianp.eq.idtime) then
c                       iityp=iabs(ITYP5(I))
c                       if(iityp.eq.1.or.iityp.eq.2) then
c                          write(651,*) dsqrt(pt2),ypartn
c                          write(654,*) GX5(I),GY5(I)
c                       elseif(iityp.eq.1103.or.iityp.eq.2101
c     1 .or.iityp.eq.2103.or.iityp.eq.2203.
c     2 .or.iityp.eq.3101.or.iityp.eq.3103.
c     3 .or.iityp.eq.3201.or.iityp.eq.3203.or.iityp.eq.3303) 
c     4 then
c                          write(652,*) dsqrt(pt2),ypartn
c                          write(655,*) GX5(I),GY5(I)
c                       elseif(iityp.eq.21) then
c                          write(653,*) dsqrt(pt2),ypartn
c                          write(656,*) GX5(I),GY5(I)
c                       endif
c                    endif
ctest-end
ctest off density with 2fm radius and z:(-0.1*t,0.1*t):
c                    gx_now=GX5(i)+(t2time-FT5(i))*PX5(i)/E5(i)
c                    gy_now=GY5(i)+(t2time-FT5(i))*PY5(i)/E5(i)
c                    gz_now=GZ5(i)+(t2time-FT5(i))*PZ5(i)/E5(i)
c                    rt_now=dsqrt(gx_now**2+gy_now**2)
c                    zmax=0.1d0*t2time
c                    volume=3.1416d0*(2d0**2)*(2*zmax)
c                    if(rt_now.gt.2d0.or.dabs(gz_now).gt.zmax)
c     1                   goto 100
ctest-end
                    iloop=1
                    if(dabs(ypartn).le.1d0) then
                       iloop=2
                       if(dabs(ypartn).le.0.5d0) then
                          iloop=3
                       endif
                    endif
                    do 50 iy=1,iloop
clin-5/2012:
c                       if(pt2.gt.0.) then
                       if(pt2.gt.0d0) then
                          v2prtn=(PX5(I)**2-PY5(I)**2)/pt2
clin-5/2012:
c                          if(abs(v2prtn).gt.1.) 
                          if(dabs(v2prtn).gt.1d0) 
     1 write(nfile(iy),*) 'v2prtn>1',v2prtn
                          v2p(ianp,iy)=v2p(ianp,iy)+v2prtn
                          v2p2(ianp,iy)=v2p2(ianp,iy)+v2prtn**2
                       endif
                       xperp2=GX5(I)**2+GY5(I)**2
clin-5/2012:
c                       if(xperp2.gt.0.) 
                       if(xperp2.gt.0d0) 
     1        s2p(ianp,iy)=s2p(ianp,iy)+(GX5(I)**2-GY5(I)**2)/xperp2
                       xnpart(ianp,iy)=xnpart(ianp,iy)+1d0
                       etp(ianp,iy)=etp(ianp,iy)+dsqrt(pt2+XMASS5(I)**2)
ctest off density:
c                       etp(ianp,iy)=etp(ianp,iy)
c     1                  +dsqrt(pt2+XMASS5(I)**2+PZ5(i)**2)/volume
clin-2/22/00 to write out parton info only for formed ones:
                       if(FT5(I).le.t2time) then
                          v2pf(ianp,iy)=v2pf(ianp,iy)+v2prtn
                          v2pf2(ianp,iy)=v2pf2(ianp,iy)+v2prtn**2
clin-5/2012:
c                          if(xperp2.gt.0.) 
                          if(xperp2.gt.0d0) 
     1        s2pf(ianp,iy)=s2pf(ianp,iy)+(GX5(I)**2-GY5(I)**2)/xperp2
                          xnpf(ianp,iy)=xnpf(ianp,iy)+1d0
                  etpf(ianp,iy)=etpf(ianp,iy)+dsqrt(pt2+XMASS5(I)**2)
ctest off density:
c                  etpf(ianp,iy)=etpf(ianp,iy)
c     1                   +dsqrt(pt2+XMASS5(I)**2+PZ5(i)**2)/volume
                       endif
 50                    continue
 100                 continue
                 itimep=ianp
                 iaevtp=iaevt
clin-3/30/00 ebe v2 variables:
                 if(ianp.eq.30) then
                    do 1007 iy=1,3
                       npartn=IDINT(xnpart(ianp,iy)-xnpp(iy))
                       if(npartn.ne.0) then
                          v2pevt(iy)=(v2p(ianp,iy)-v2pp(iy))/npartn
                          v2psum(iy)=v2psum(iy)+v2pevt(iy)
                          v2p2sm(iy)=v2p2sm(iy)+v2pevt(iy)**2
                          v2pp(iy)=v2p(ianp,iy)
                          xnpp(iy)=xnpart(ianp,iy)
                       endif
 1007               continue
                    write(49, 160) iaevt,v2pevt
                 endif
                 goto 101
              endif
 1008      continue
clin-11/28/00 for animation:
 101           if(nevent.eq.1) then
              do 110 nt = 1, ntmax
                 time1=dble(nt*dt)
                 time2=dble((nt+1)*dt)
                 if (t2time.lt.time2.and.t2time.ge.time1) then
                    if(nt.le.itanim) return
c                    write(92,*) t2time
                    iform=0
                    do 1009 I=1,mul
c     write out parton info only for formed ones:
                       if(FT5(I).le.t2time) then
                          iform=iform+1
                       endif
 1009               continue
c                    write(92,*) iform
                    do 120 I=1,mul
                       if(FT5(I).le.t2time) then
clin-11/29/00-ctest off calculate parton coordinates after propagation:
c                          gx_now=GX5(i)+(t2time-FT5(i))*PX5(i)/E5(i)
c                          gy_now=GY5(i)+(t2time-FT5(i))*PY5(i)/E5(i)
c                          gz_now=GZ5(i)+(t2time-FT5(i))*PZ5(i)/E5(i)
c          write(92,140) ITYP5(i),GX5(i),GY5(i),GZ5(i),FT5(i)
c          write(92,180) ITYP5(i),GX5(i),GY5(i),GZ5(i),FT5(i),
c     1    PX5(i),PY5(i),PZ5(i),E5(i)
ctest-end
                       endif
 120                    continue
                    itanim=nt
                 endif
 110              continue
           endif
c
 140           format(i10,4(2x,f7.2))
 160           format(i10,3(2x,f9.5))
c 180           format(i6,8(1x,f7.2))
clin-5/17/01 calculate v2 for active partons (which have not frozen out):
c     idd=3, called at end of zpc.f:
        else if(idd.eq.3) then        
           do 1010 ianp=1,30
              if(iscatt(ianp).eq.0) tscatt(ianp+1)=tscatt(ianp)
 1010      continue
           do 350 I=1,mul
              ypartn=0.5d0*dlog((E5(i)+PZ5(i)+1.d-8)
     1 /(E5(i)-PZ5(i)+1.d-8))
              pt2=PX5(I)**2+PY5(I)**2
              iloop=1
              if(dabs(ypartn).le.1d0) then
                 iloop=2
                 if(dabs(ypartn).le.0.5d0) then
                    iloop=3
                 endif
              endif
c
              do 325 ianp=1,30
                 if(iscatt(ianp).ne.0) then
                    if(FT5(I).lt.tscatt(ianp+1)
     1 .and.FT5(I).ge.tscatt(ianp)) then
                       do 1011 iy=1,iloop
clin-5/2012:
c                          if(pt2.gt.0.) then
                          if(pt2.gt.0d0) then
                             v2prtn=(PX5(I)**2-PY5(I)**2)/pt2
                             v2pfrz(ianp,iy)=v2pfrz(ianp,iy)+v2prtn
                     v2p2fz(ianp,iy)=v2p2fz(ianp,iy)+v2prtn**2
                          endif
                          xperp2=GX5(I)**2+GY5(I)**2
clin-5/2012:
c                          if(xperp2.gt.0.) s2pfrz(ianp,iy)=
                          if(xperp2.gt.0d0) s2pfrz(ianp,iy)=
     1 s2pfrz(ianp,iy)+(GX5(I)**2-GY5(I)**2)/xperp2
        etpfrz(ianp,iy)=etpfrz(ianp,iy)+dsqrt(pt2+XMASS5(I)**2)
                          xnpfrz(ianp,iy)=xnpfrz(ianp,iy)+1d0
ctest off density:
c                    etpfrz(ianp,iy)=etpfrz(ianp,iy)
c     1                   +dsqrt(pt2+XMASS5(I)**2+PZ5(i)**2)/volume
csp07/05
            if(ITYP5(I).eq.1.or.ITYP5(I).eq.2)then
              etlfrz(ianp,iy)=etlfrz(ianp,iy)+dsqrt(pt2+XMASS5(I)**2)
              xnlfrz(ianp,iy)=xnlfrz(ianp,iy)+1d0
            elseif(ITYP5(I).eq.3)then
              etsfrz(ianp,iy)=etsfrz(ianp,iy)+dsqrt(pt2+XMASS5(I)**2)
              xnsfrz(ianp,iy)=xnsfrz(ianp,iy)+1d0
            endif
csp07/05 end
 1011    continue
c     parton freezeout info taken, proceed to next parton:
                       goto 350
                    endif
                 endif
 325          continue
 350       continue
c     idd=2: calculate average partonic elliptic flow, called from artdri.f,
        else if(idd.eq.2) then
           do 1012 i=1,3
              write(nfile(i),*) '   tsp,   v2p,     v2p2, '//
     1 '   s2p,  etp,   xmult,    nevt,  xnctot'
              write ((nfile(i)+1),*) '  tsp,   v2pf,   v2pf2, '//
     1 '   s2pf, etpf,  xnform,  xcrate'
              write ((nfile(i)+2),*) '  tsp,   v2pa,   v2pa2, '//
     1 '   s2pa, etpa,  xmult_ap,  xnform,   nevt'
              write ((nfile(i)+3),*) '  tsph,  v2ph,   v2ph2, '//
     1 '   s2ph, etph,  xmult_(ap/2+h),xmult_ap/2,nevt'
csp
           write(nlfile(i),*) '   tsp,    v2,     s2,    etp,    xmul'
           write(nsfile(i),*) '   tsp,    v2,     s2,    etp,    xmul'
           write(nlfile(i)+1,*) '   tsp,    v2,     s2,    etp,    xmul'
           write(nsfile(i)+1,*) '   tsp,    v2,     s2,    etp,    xmul'
c
 1012   continue
clin-3/30/00 ensemble average & variance of v2 (over particles in all events):
           do 150 ii=1, 30
              if(nevt(ii).eq.0) goto 150
              do 1014 iy=1,3
clin-5/2012:
c                 if(xnpart(ii,iy).gt.1) then
                 if(xnpart(ii,iy).gt.1d0) then
                    v2p(ii,iy)=v2p(ii,iy)/xnpart(ii,iy)
                    v2p2(ii,iy)=dsqrt((v2p2(ii,iy)/xnpart(ii,iy)
     1                    -v2p(ii,iy)**2)/(xnpart(ii,iy)-1))
                    s2p(ii,iy)=s2p(ii,iy)/xnpart(ii,iy)
c xmult and etp are multiplicity and et for an averaged event:
                    xmult=dble(xnpart(ii,iy)/dble(nevt(ii)))
                    etp(ii,iy)=etp(ii,iy)/dble(nevt(ii))
csp
                    etpl(ii,iy)=etpl(ii,iy)/dble(nevt(ii))
                    etps(ii,iy)=etps(ii,iy)/dble(nevt(ii))
c
                    xnctot=0d0
                    do 1013 inum=1,ii
                       xnctot=xnctot+xncoll(inum)
 1013               continue
                    if(nevt(1).ne.0) xnctot=xnctot/nevt(1)
                    write (nfile(iy),200) tsp(ii),v2p(ii,iy),
     1      v2p2(ii,iy),s2p(ii,iy),etp(ii,iy),xmult,nevt(ii),xnctot
                 endif
                 if(nevt(ii).ne.0) 
     1                xcrate=xncoll(ii)/(tsp(ii+1)-tsp(ii))/nevt(ii)
c
clin-5/2012:
c                 if(xnpf(ii,iy).gt.1) then
                 if(xnpf(ii,iy).gt.1d0) then
                    v2pf(ii,iy)=v2pf(ii,iy)/xnpf(ii,iy)
                    v2pf2(ii,iy)=dsqrt((v2pf2(ii,iy)/xnpf(ii,iy)
     1                    -v2pf(ii,iy)**2)/(xnpf(ii,iy)-1))
                    s2pf(ii,iy)=s2pf(ii,iy)/xnpf(ii,iy)
                    xnform=dble(xnpf(ii,iy)/dble(nevt(ii)))
                    etpf(ii,iy)=etpf(ii,iy)/dble(nevt(ii))
csp
                    etplf(ii,iy)=etplf(ii,iy)/dble(nevt(ii))
                    etpsf(ii,iy)=etpsf(ii,iy)/dble(nevt(ii))
c
                    write (nfile(iy)+1, 210) tsp(ii),v2pf(ii,iy),
     1      v2pf2(ii,iy),s2pf(ii,iy),etpf(ii,iy),xnform,xcrate
                 endif
csp
clin-5/2012:
c                 if(xnpl(ii,iy).gt.1) then
                 if(xnpl(ii,iy).gt.1d0) then
                    v2pl(ii,iy)=v2pl(ii,iy)/xnpl(ii,iy)
                    s2pl(ii,iy)=s2pl(ii,iy)/xnpl(ii,iy)
                    xmult=dble(xnpl(ii,iy)/dble(nevt(ii)))
                    etpl(ii,iy)=etpl(ii,iy)/dble(nevt(ii))
                    write (nlfile(iy),201) tsp(ii),v2pl(ii,iy),
     1        s2pl(ii,iy),etpl(ii,iy),xmult
                 endif
clin-5/2012:
c                 if(xnps(ii,iy).gt.1) then
                 if(xnps(ii,iy).gt.1d0) then
                    v2ps(ii,iy)=v2ps(ii,iy)/xnps(ii,iy)
                    s2ps(ii,iy)=s2ps(ii,iy)/xnps(ii,iy)
                    xmult=dble(xnps(ii,iy)/dble(nevt(ii)))
                    etps(ii,iy)=etps(ii,iy)/dble(nevt(ii))
                    write (nsfile(iy),201) tsp(ii),v2ps(ii,iy),
     1        s2ps(ii,iy),etps(ii,iy),xmult
                 endif
clin-5/2012:
c                 if(xnplf(ii,iy).gt.1) then
                 if(xnplf(ii,iy).gt.1d0) then
                    v2plf(ii,iy)=v2plf(ii,iy)/xnplf(ii,iy)
                    s2plf(ii,iy)=s2plf(ii,iy)/xnplf(ii,iy)
                    xmult=dble(xnplf(ii,iy)/dble(nevt(ii)))
                    etplf(ii,iy)=etplf(ii,iy)/dble(nevt(ii))
                    write (nlfile(iy)+1,201) tsp(ii),v2plf(ii,iy),
     1        s2plf(ii,iy),etplf(ii,iy),xmult
                 endif
clin-5/2012:
c                 if(xnpsf(ii,iy).gt.1) then
                 if(xnpsf(ii,iy).gt.1d0) then
                    v2psf(ii,iy)=v2psf(ii,iy)/xnpsf(ii,iy)
                    s2psf(ii,iy)=s2psf(ii,iy)/xnpsf(ii,iy)
                    xmult=dble(xnpsf(ii,iy)/dble(nevt(ii)))
                    etpsf(ii,iy)=etpsf(ii,iy)/dble(nevt(ii))
                    write (nsfile(iy)+1,201) tsp(ii),v2psf(ii,iy),
     1        s2psf(ii,iy),etpsf(ii,iy),xmult
                 endif
csp-end
 1014         continue
 150           continue
csp07/05 initial & final mt distrb 
               scalei=0d0
               scalef=0d0               
               if(nevt(1).ne.0) SCALEi = 1d0 / dble(nevt(1)) / BMT
               if(nevt(30).ne.0) SCALEf = 1d0 / dble(nevt(30)) / BMT
         do 1016 iy=2,3
           yra = 1d0
           if(iy .eq. 2)yra = 2d0
         do 1015 i=1,50
           WRITE(nmfile(iy),251) BMT*dble(I - 0.5), 
     &     SCALEi*DMYil(I,iy)/yra, SCALEf*DMYfl(I,iy)/yra,
     &     SCALEi*DMYis(I,iy)/yra, SCALEf*DMYfs(I,iy)/yra
 1015   continue
 1016 continue
csp07/05 end
clin-3/30/00 event-by-event average & variance of v2:
           if(nevt(30).ge.1) then
              do 1017 iy=1,3
                 v2pavg(iy)=v2psum(iy)/nevt(30)
                 v2var0=v2p2sm(iy)/nevt(30)-v2pavg(iy)**2
clin-5/2012:
c                 if(v2var0.gt.0) varv2p(iy)=dsqrt(v2var0)
                 if(v2var0.gt.0d0) varv2p(iy)=dsqrt(v2var0)
 1017 continue
              write(49, 240) 'EBE v2p,v2p(y2),v2p(y1): avg=', v2pavg
              write(49, 240) 'EBE v2p,v2p(y2),v2p(y1): var=', varv2p
           endif
clin-11/28/00 for animation:
            if(nevent.eq.1) then
              do 1018 I=1,mul
                 if(FT5(I).le.t2time) then
         write(93,140) ITYP5(i),GX5(i),GY5(i),GZ5(i),FT5(i)
                 endif
 1018         continue
clin-11/29/00 signal the end of animation file:
c              write(91,*) -10.
c              write(91,*) 0
c              write(92,*) -10.
c              write(92,*) 0
c              close (91)
c              close (92)
              close (93)
           endif
clin-5/18/01 calculate v2 for active partons:
           do 450 ianp=1,30
              do 400 iy=1,3
                 v2pact=0d0
                 v2p2ac=0d0
                 s2pact=0d0
                 etpact=0d0
                 xnacti=0d0
clin-5/2012:
c                 if(xnpf(ianp,iy).gt.1) then
                 if(xnpf(ianp,iy).gt.1d0) then
c     reconstruct the sum of v2p, v2p2, s2p, etp, and xnp for formed partons:
                    v2pact=v2pf(ianp,iy)*xnpf(ianp,iy)
                    v2p2ac=(v2pf2(ianp,iy)**2*(xnpf(ianp,iy)-1)
     1 +v2pf(ianp,iy)**2)*xnpf(ianp,iy)
                    s2pact=s2pf(ianp,iy)*xnpf(ianp,iy)
                    etpact=etpf(ianp,iy)*dble(nevt(ianp))
                    xnpact=xnpf(ianp,iy)
c
                    do 1019 kanp=1,ianp
                       v2pact=v2pact-v2pfrz(kanp,iy)
                       v2p2ac=v2p2ac-v2p2fz(kanp,iy)
                       s2pact=s2pact-s2pfrz(kanp,iy)
                       etpact=etpact-etpfrz(kanp,iy)
                       xnpact=xnpact-xnpfrz(kanp,iy)
 1019               continue
c     save the sum of v2p, v2p2, s2p, etp, and xnp for formed partons:
                    v2ph=v2pact
                    v2ph2=v2p2ac
                    s2ph=s2pact
                    etph=etpact
                    xnp2=xnpact/2d0
c
clin-5/2012:
c                    if(xnpact.gt.1.and.nevt(ianp).ne.0) then
                    if(xnpact.gt.1d0.and.nevt(ianp).ne.0) then
                       v2pact=v2pact/xnpact
                       v2p2ac=dsqrt((v2p2ac/xnpact
     1                    -v2pact**2)/(xnpact-1))
                       s2pact=s2pact/xnpact
                       xnacti=dble(xnpact/dble(nevt(ianp)))
                       etpact=etpact/dble(nevt(ianp))
                       write (nfile(iy)+2, 250) tsp(ianp),v2pact,
     1 v2p2ac,s2pact,etpact,xnacti,
     2 xnpf(ianp,iy)/dble(nevt(ianp)),nevt(ianp)
                    endif
                 endif
c     To calculate combined v2 for active partons plus formed hadrons, 
c     add the sum of v2h, v2h2, s2h, eth, and xnh for formed hadrons:
c     scale the hadron part in case nevt(ianp) != nevent:
                 shadr=dble(nevt(ianp))/dble(nevent)
                 ianh=ianp
                 v2ph=v2ph+v2h(ianh,iy)*xnhadr(ianh,iy)*shadr
                 v2ph2=v2ph2+(v2h2(ianh,iy)**2*(xnhadr(ianh,iy)-1)
     1 +v2h(ianh,iy)**2)*xnhadr(ianh,iy)*shadr
                 s2ph=s2ph+s2h(ianh,iy)*xnhadr(ianh,iy)*shadr
                 etph=etph+eth(ianh,iy)*dble(nevent)*shadr
                 xnph=xnpact+xnhadr(ianh,iy)*shadr
                 xnp2h=xnp2+xnhadr(ianh,iy)*shadr
clin-5/2012:
c                 if(xnph.gt.1) then
                 if(xnph.gt.1d0) then
                    v2ph=v2ph/xnph
                    v2ph2=dsqrt((v2ph2/xnph-v2ph**2)/(xnph-1))
                    s2ph=s2ph/xnph
                    etph=etph/dble(nevt(ianp))
                    xnp2=xnp2/dble(nevt(ianp))
                    xnp2h=xnp2h/dble(nevent)
                    if(tsp(ianp).le.(ntmax*dt)) 
     1                    write (nfile(iy)+3, 250) tsp(ianp),v2ph,
     2 v2ph2,s2ph,etph,xnp2h,xnp2,nevt(ianp)
                 endif
c
 400              continue
 450       continue
           do 550 ianp=1,30
              do 500 iy=1,3
                 v2pact=0d0
                 v2p2ac=0d0
                 s2pact=0d0
                 etpact=0d0
                 xnacti=0d0
c     reconstruct the sum of v2p, v2p2, s2p, etp, and xnp for formed partons:
                    v2pact=v2pf(ianp,iy)*xnpf(ianp,iy)
                    v2p2ac=(v2pf2(ianp,iy)**2*(xnpf(ianp,iy)-1)
     1 +v2pf(ianp,iy)**2)*xnpf(ianp,iy)
                    s2pact=s2pf(ianp,iy)*xnpf(ianp,iy)
                    etpact=etpf(ianp,iy)*dble(nevt(ianp))
                    xnpact=xnpf(ianp,iy)
 500              continue
 550           continue
           close (620)
           close (630)
           do 1021 nf=1,3
              do 1020 ifile=0,3
                 close(nfile(nf)+ifile)
 1020        continue
 1021     continue
           do 1022 nf=1,3
              close(740+nf)
 1022      continue
        endif
 200        format(2x,f5.2,3(2x,f7.4),2(2x,f9.2),i6,2x,f9.2)
 210        format(2x,f5.2,3(2x,f7.4),3(2x,f9.2))
 240        format(a30,3(2x,f9.5))
 250        format(2x,f5.2,3(2x,f7.4),3(2x,f9.2),i6)
csp
 201        format(2x,f5.2,4(2x,f9.2))
 251        format(5e15.5)
c
        return
        end

c=======================================================================
c     Calculate flow from formed hadrons, called by art1e.f:
c     Note: numbers in art not in double precision!
        subroutine flowh(ct)
        PARAMETER (MAXSTR=150001, MAXR=1)
        dimension tsh(31)
        DOUBLE PRECISION  v2h,xnhadr,eth,v2h2,s2h
        DOUBLE PRECISION  v2hp,xnhadp,v2hsum,v2h2sm,v2hevt(3)
        DOUBLE PRECISION  pt2, v2hadr
        COMMON /hflow/ v2h(30,3),xnhadr(30,3),eth(30,3),
     1 v2h2(30,3),s2h(30,3)
cc      SAVE /hflow/
        common/ebe/v2hp(3),xnhadp(3),v2hsum(3),v2h2sm(3)
cc      SAVE /ebe/
        common /lastt/itimeh,bimp
cc      SAVE /lastt/
        COMMON /RUN/ NUM
cc      SAVE /RUN/
        COMMON  /AA/      R(3,MAXSTR)
cc      SAVE /AA/
        COMMON  /BB/      P(3,MAXSTR)
cc      SAVE /BB/
        COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
        COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        COMMON  /RR/      MASSR(0:MAXR)
cc      SAVE /RR/
        common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
        COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
        SAVE   
c
        do 1001 ii = 1, 31
           tsh(ii)=float(ii-1)
 1001   continue
c
        do 1004 ianh = 1, 30
           if ((ct+0.0001).lt.tsh(ianh+1)
     1 .and.(ct+0.0001).ge.tsh(ianh)) then
              if(ianh.eq.itimeh) goto 101
              IA = 0
              DO 1002 J = 1, NUM
                 mult=MASSR(J)
                 IA = IA + MASSR(J - 1)
                 DO 100 IC = 1, mult
                    I = IA + IC
c     5/04/01 exclude leptons and photons:
                    if(iabs(LB(I)-10000).lt.100) goto 100
                    px=p(1,i)
                    py=p(2,i)
                    pt2=dble(px)**2+dble(py)**2
c     2/18/00 Note: e(i) gives the mass in ART:
                    ene=sqrt(e(i)**2+sngl(pt2)+p(3,i)**2)
                    RAP=0.5*alog((ene+p(3,i))/(ene-p(3,i)))
ctest off density with 2fm radius and z:(-0.1*t,0.1*t):
c                rt_now=sqrt(r(1,i)**2+r(2,i)**2)
c                gz_now=r(3,i)
c                zmax=0.1*ct
c                volume=3.1416*(2.**2)*2*zmax
c                if(rt_now.gt.2.or.abs(gz_now).gt.zmax)
c     1               goto 100
                    iloop=1
                    if(abs(rap).le.1) then
                       iloop=2
                       if(abs(rap).le.0.5) then
                          iloop=3
                       endif
                    endif
                    do 50 iy=1,iloop
                       if(pt2.gt.0d0) then
                          v2hadr=(dble(px)**2-dble(py)**2)/pt2
                          v2h(ianh,iy)=v2h(ianh,iy)+v2hadr
                          v2h2(ianh,iy)=v2h2(ianh,iy)+v2hadr**2
                          if(dabs(v2hadr).gt.1d0) 
     1 write(1,*) 'v2hadr>1',v2hadr,px,py
                       endif
                       xperp2=r(1,I)**2+r(2,I)**2
                       if(xperp2.gt.0.) 
     1 s2h(ianh,iy)=s2h(ianh,iy)+dble((r(1,I)**2-r(2,I)**2)/xperp2)
               eth(ianh,iy)=eth(ianh,iy)+dble(SQRT(e(i)**2+sngl(pt2)))
ctest off density:
c               eth(ianh,iy)=eth(ianh,iy)
c     1                  +dble(SQRT(e(i)**2+sngl(pt2)+p(3,i)**2))/volume
                       xnhadr(ianh,iy)=xnhadr(ianh,iy)+1d0
 50                    continue
 100                 continue
 1002         CONTINUE
              itimeh=ianh
clin-5/04/01 ebe v2 variables:
              if(ianh.eq.30) then
                 do 1003 iy=1,3
                    nhadrn=IDINT(xnhadr(ianh,iy)-xnhadp(iy))
                    if(nhadrn.ne.0) then
               v2hevt(iy)=(v2h(ianh,iy)-v2hp(iy))/dble(nhadrn)
                       v2hsum(iy)=v2hsum(iy)+v2hevt(iy)
                       v2h2sm(iy)=v2h2sm(iy)+v2hevt(iy)**2
                       v2hp(iy)=v2h(ianh,iy)
                       xnhadp(iy)=xnhadr(ianh,iy)
                    endif
 1003            continue
                 write(88, 160) iaevt,v2hevt
              endif
              goto 101
           endif
 1004   continue
 160        format(i10,3(2x,f9.5))
clin-11/27/00 for animation:
 101        if(nevent.eq.1) then
           IA = 0
           do 1005 J = 1, NUM
              mult=MASSR(J)
              IA = IA + MASSR(J - 1)
c              write(91,*) ct
c              write(91,*) mult
              DO 150 IC = 1, mult
                 I = IA + IC
c                 write(91,210) LB(i),r(1,i),r(2,i),r(3,i),
c     1              p(1,i),p(2,i),p(3,i),e(i)
 150              continue
 1005      continue
           return
        endif
c 210        format(i6,7(1x,f8.3))
        return
        end

c=======================================================================
        subroutine flowh0(NEVNT,idd)
c
        dimension tsh(31)
        DOUBLE PRECISION  v2h,xnhadr,eth,v2h2,s2h
        DOUBLE PRECISION  v2hp,xnhadp,v2hsum,v2h2sm,
     1 v2havg(3),varv2h(3)
        COMMON /hflow/ v2h(30,3),xnhadr(30,3),eth(30,3),
     1 v2h2(30,3),s2h(30,3)
cc      SAVE /hflow/
        common/ebe/v2hp(3),xnhadp(3),v2hsum(3),v2h2sm(3)
cc      SAVE /ebe/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
        common /lastt/itimeh,bimp
cc      SAVE /lastt/
        SAVE   
      
c     idd=0: initialization for flow analysis, called by artdri.f::
        if(idd.eq.0) then
           itimeh=0
c
           do 1001 ii = 1, 31
              tsh(ii)=float(ii-1)
 1001      continue
c
           do 1003 ii=1,30
              do 1002 iy=1,3
                 v2h(ii,iy)=0d0
                 xnhadr(ii,iy)=0d0
                 eth(ii,iy)=0d0
                 v2h2(ii,iy)=0d0
                 s2h(ii,iy)=0d0
 1002         continue
 1003      continue
           do 1004 iy=1,3
              v2hp(iy)=0d0
              xnhadp(iy)=0d0
              v2hsum(iy)=0d0
              v2h2sm(iy)=0d0
            if(iy.eq.1) then
               nunit=59
            elseif(iy.eq.2) then
               nunit=68
            else
               nunit=69
            endif
              write(nunit,*) '   tsh,   v2h,     v2h2,     s2h, '//
     1 ' eth,   xmulth'
 1004      continue
c     idd=2: calculate average hadronic elliptic flow, called by artdri.f:
        else if(idd.eq.2) then
           do 100 ii=1, 30
              do 1005 iy=1,3
                 if(xnhadr(ii,iy).eq.0) then
                    xmulth=0.
                 elseif(xnhadr(ii,iy).gt.1) then
                    v2h(ii,iy)=v2h(ii,iy)/xnhadr(ii,iy)
                    eth(ii,iy)=eth(ii,iy)/dble(NEVNT)
                    v2h2(ii,iy)=dsqrt((v2h2(ii,iy)/xnhadr(ii,iy)
     1                    -v2h(ii,iy)**2)/(xnhadr(ii,iy)-1))
                    s2h(ii,iy)=s2h(ii,iy)/xnhadr(ii,iy)
                    xmulth=sngl(xnhadr(ii,iy)/NEVNT)
                 endif
             if(iy.eq.1) then
                nunit=59
             elseif(iy.eq.2) then
                nunit=68
             else
                nunit=69
             endif
                 if(tsh(ii).le.(ntmax*dt)) 
     1                    write (nunit,200) tsh(ii),v2h(ii,iy),
     2      v2h2(ii,iy),s2h(ii,iy),eth(ii,iy),xmulth
 1005         continue
 100           continue
c     event-by-event average & variance of v2h:
           do 1006 iy=1,3
              v2havg(iy)=v2hsum(iy)/dble(NEVNT)
      varv2h(iy)=dsqrt(v2h2sm(iy)/dble(NEVNT)-v2havg(iy)**2)
 1006 continue
           write(88, 240) 'EBE v2h,v2h(y2),v2h(y1): avg=', v2havg
           write(88, 240) 'EBE v2h,v2h(y2),v2h(y1): var=', varv2h
        endif
 200        format(2x,f5.2,3(2x,f7.4),2(2x,f9.2))
 240        format(a30,3(2x,f9.5))
        return
        end

c=======================================================================
c     2/23/00 flow from all initial hadrons just before entering ARTMN:
        subroutine iniflw(NEVNT,idd)
        PARAMETER (MAXSTR=150001, MAXR=1)
        DOUBLE PRECISION  v2i,eti,xmulti,v2mi,s2mi,xmmult,
     1       v2bi,s2bi,xbmult
        COMMON /RUN/ NUM
cc      SAVE /RUN/
        COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
        COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cc      SAVE /ARPRC1/
        COMMON/iflow/v2i,eti,xmulti,v2mi,s2mi,xmmult,v2bi,s2bi,xbmult
cc      SAVE /iflow/
        SAVE   
c        
        if(idd.eq.0) then
           v2i=0d0
           eti=0d0
           xmulti=0d0
           v2mi=0d0
           s2mi=0d0
           xmmult=0d0
           v2bi=0d0
           s2bi=0d0
           xbmult=0d0
        else if(idd.eq.1) then
           do 1002 J = 1, NUM
              do 1001 I = 1, MULTI1(J)
                 ITYP = ITYP1(I, J)
c     all hadrons:
                 IF (ITYP .GT. -100 .AND. ITYP .LT. 100) GOTO 100
                 xmulti=xmulti+1.d0
                 PX = PX1(I, J)
                 PY = PY1(I, J)
                 XM = XM1(I, J)
                 pt2=px**2+py**2
                 xh=gx1(I,J)
                 yh=gy1(I,J)
                 xt2=xh**2+yh**2
                 if(pt2.gt.0) v2i=v2i+dble((px**2-py**2)/pt2)
                 eti=eti+dble(SQRT(PX ** 2 + PY ** 2 + XM ** 2))
c     baryons only:
                 IF (ITYP .LT. -1000 .or. ITYP .GT. 1000) then
                    xbmult=xbmult+1.d0
                    if(pt2.gt.0) v2bi=v2bi+dble((px**2-py**2)/pt2)
                    if(xt2.gt.0) s2bi=s2bi+dble((xh**2-yh**2)/xt2)
c     mesons only:
                 else
                    xmmult=xmmult+1.d0
                    if(pt2.gt.0) v2mi=v2mi+dble((px**2-py**2)/pt2)
                    if(xt2.gt.0) s2mi=s2mi+dble((xh**2-yh**2)/xt2)
                 endif
 100                 continue
 1001         continue
 1002      continue
        else if(idd.eq.2) then
           if(xmulti.ne.0) v2i=v2i/xmulti
           eti=eti/dble(NEVNT)
           xmulti=xmulti/dble(NEVNT)
           if(xmmult.ne.0) then
              v2mi=v2mi/xmmult
              s2mi=s2mi/xmmult
           endif
           xmmult=xmmult/dble(NEVNT)
           if(xbmult.ne.0) then
              v2bi=v2bi/xbmult
              s2bi=s2bi/xbmult
           endif
           xbmult=xbmult/dble(NEVNT)
        endif
c
        return
        end

c=======================================================================
c     2/25/00 dN/dt analysis for production (before ZPCMN)  
c     and freezeout (right after ZPCMN) for all partons.
        subroutine frztm(NEVNT,idd)
c
        implicit double precision  (a-h, o-z)
        PARAMETER (MAXPTN=400001)
        dimension tsf(31)
        COMMON /PARA1/ MUL
cc      SAVE /PARA1/
        COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &       PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &       XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
        COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &       PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &       XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
        COMMON /frzout/ xnprod(30),etprod(30),xnfrz(30),etfrz(30),
     & dnprod(30),detpro(30),dnfrz(30),detfrz(30)
cc      SAVE /frzout/ 
        SAVE   
        data tsf/0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
     &       1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
     &       2  , 3,   4,   5,   6,   7,   8,   9,   10,  20,  30/
c
        if(idd.eq.0) then
           do 1001 ii=1,30
              xnprod(ii)=0d0
              etprod(ii)=0d0
              xnfrz(ii)=0d0
              etfrz(ii)=0d0
              dnprod(ii)=0d0
              detpro(ii)=0d0
              dnfrz(ii)=0d0
              detfrz(ii)=0d0
 1001      continue
           OPEN (86, FILE = 'ana1/production.dat', STATUS = 'UNKNOWN')
           OPEN (87, FILE = 'ana1/freezeout.dat', STATUS = 'UNKNOWN')
        else if(idd.eq.1) then
           DO 100 ip = 1, MUL
              do 1002 ii=1,30
                 eth0=dSQRT(PX0(ip)**2+PY0(ip)**2+XMASS0(ip)**2)
                 eth2=dSQRT(PX5(ip)**2+PY5(ip)**2+XMASS5(ip)**2)
c     total number and Et produced by time tsf(ii):
                 if (ft0(ip).lt.tsf(ii+1)) then
                    xnprod(ii)=xnprod(ii)+1d0
                    etprod(ii)=etprod(ii)+eth0
c     number and Et produced from time tsf(ii) to tsf(ii+1):
                    if (ft0(ip).ge.tsf(ii)) then
                       dnprod(ii)=dnprod(ii)+1d0
                       detpro(ii)=detpro(ii)+eth0
                    endif
                 endif
c     total number and Et freezed out by time tsf(ii):
                 if (FT5(ip).lt.tsf(ii+1)) then
                    xnfrz(ii)=xnfrz(ii)+1d0
                    etfrz(ii)=etfrz(ii)+eth2
c     number and Et freezed out from time tsf(ii) to tsf(ii+1):
                    if (FT5(ip).ge.tsf(ii)) then
                       dnfrz(ii)=dnfrz(ii)+1d0
                       detfrz(ii)=detfrz(ii)+eth2
                    endif
                 endif
 1002         continue
 100           continue
        else if(idd.eq.2) then
           write (86,*) '       t,       np,       dnp/dt,      etp '//
     1 ' detp/dt'
           write (87,*) '       t,       nf,       dnf/dt,      etf '//
     1 ' detf/dt'
           do 1003 ii=1,30
              xnp=xnprod(ii)/dble(NEVNT)
              xnf=xnfrz(ii)/dble(NEVNT)
              etp=etprod(ii)/dble(NEVNT)
              etf=etfrz(ii)/dble(NEVNT)
              dxnp=dnprod(ii)/dble(NEVNT)/(tsf(ii+1)-tsf(ii))
              dxnf=dnfrz(ii)/dble(NEVNT)/(tsf(ii+1)-tsf(ii))
              detp=detpro(ii)/dble(NEVNT)/(tsf(ii+1)-tsf(ii))
              detf=detfrz(ii)/dble(NEVNT)/(tsf(ii+1)-tsf(ii))
              write (86, 200) 
     1        tsf(ii+1),xnp,dxnp,etp,detp
              write (87, 200) 
     1        tsf(ii+1),xnf,dxnf,etf,detf
 1003      continue
        endif
 200    format(2x,f9.2,4(2x,f10.2))
c
        return
        end

c=======================================================================
clin-6/2009 write out initial minijet information 
c     before propagating to its formation time:
clin-2/2012:
c        subroutine minijet_out(BB)
        subroutine minijet_out(BB,phiRP)
        PARAMETER (MAXSTR=150001)
        COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
        COMMON/hjcrdn/YP(3,300),YT(3,300)
        COMMON/HJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &                PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &                PJPM(300,500),NTJ(300),KFTJ(300,500),
     &                PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &                PJTE(300,500),PJTM(300,500)
        COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &       K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &       PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
        COMMON /AREVT/ IAEVT, IARUN, MISS
        common /para7/ ioscar,nsmbbbar,nsmmeson
        common/phidcy/iphidcy,pttrig,ntrig,maxmiss,ipi0dcy
        SAVE
        ntrig=0
        do I = 1, IHNT2(1)
           do J = 1, NPJ(I)
              pt=sqrt(PJPX(I,J)**2+PJPY(I,J)**2)
              if(pt.ge.pttrig) ntrig=ntrig+1
           enddo
        enddo
        do I = 1, IHNT2(3)
           do J = 1, NTJ(I)
              pt=sqrt(PJTX(I,J)**2+PJTY(I,J)**2)
              if(pt.ge.pttrig) ntrig=ntrig+1
           enddo
        enddo
        do I = 1, NSG
           do J = 1, NJSG(I)
              pt=sqrt(PXSG(I,J)**2+PYSG(I,J)**2)
              if(pt.ge.pttrig) ntrig=ntrig+1
           enddo
        enddo
c     Require at least 1 initial minijet parton above the trigger Pt value:
        if(ntrig.eq.0) return

c.....transfer data from HIJING to ZPC
        if(ioscar.eq.3) write(96,*) IAEVT,MISS,IHNT2(1),IHNT2(3)
        DO 1008 I = 1, IHNT2(1)
           DO 1007 J = 1, NPJ(I)
              ityp=KFPJ(I,J)
c     write out not only gluons:
c              if(ityp.ne.21) goto 1007
clin-2/2012:
c              gx=YP(1,I)+0.5*BB
c              gy=YP(2,I)
              gx=YP(1,I)+0.5*BB*cos(phiRP)
              gy=YP(2,I)+0.5*BB*sin(phiRP)
              gz=0.
              ft=0.
              px=PJPX(I,J)
              py=PJPY(I,J)
              pz=PJPZ(I,J)
              xmass=PJPM(I,J)
              if(ioscar.eq.3) then
                 if(amax1(abs(gx),abs(gy),
     1                abs(gz),abs(ft)).lt.9999) then
                    write(96,200) ityp,px,py,pz,xmass,gx,gy,gz,ft,1
                 else
                    write(96,201) ityp,px,py,pz,xmass,gx,gy,gz,ft,1
                 endif
              endif
 1007      CONTINUE
 1008   CONTINUE
        DO 1010 I = 1, IHNT2(3)
           DO 1009 J = 1, NTJ(I)
              ityp=KFTJ(I,J)
c              if(ityp.ne.21) goto 1009
clin-2/2012:
c              gx=YT(1,I)-0.5*BB
c              gy=YT(2,I)
              gx=YT(1,I)-0.5*BB*cos(phiRP)
              gy=YT(2,I)-0.5*BB*sin(phiRP)
              gz=0.
              ft=0.
              px=PJTX(I,J)
              py=PJTY(I,J)
              pz=PJTZ(I,J)
              xmass=PJTM(I,J)
              if(ioscar.eq.3) then
                 if(amax1(abs(gx),abs(gy),
     1                abs(gz),abs(ft)).lt.9999) then
                    write(96,200) ityp,px,py,pz,xmass,gx,gy,gz,ft,2
                 else
                    write(96,201) ityp,px,py,pz,xmass,gx,gy,gz,ft,2
                 endif
              endif
 1009      CONTINUE
 1010   CONTINUE
        DO 1012 I = 1, NSG
           DO 1011 J = 1, NJSG(I)
              ityp=K2SG(I,J)
c              if(ityp.ne.21) goto 1011
              gx=0.5*(YP(1,IASG(I,1))+YT(1,IASG(I,2)))
              gy=0.5*(YP(2,IASG(I,1))+YT(2,IASG(I,2)))
              gz=0.
              ft=0.
              px=PXSG(I,J)
              py=PYSG(I,J)
              pz=PZSG(I,J)
              xmass=PMSG(I,J)
              if(ioscar.eq.3) then
                 if(amax1(abs(gx),abs(gy),
     1                abs(gz),abs(ft)).lt.9999) then
                    write(96,200) ityp,px,py,pz,xmass,gx,gy,gz,ft,3
                 else
                    write(96,201) ityp,px,py,pz,xmass,gx,gy,gz,ft,3
                 endif
              endif
 1011      CONTINUE
 1012   CONTINUE
 200  format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,2(1x,f8.2),2(2x,f2.0),2x,I2)
 201  format(I6,2(1x,f8.3),1x,f10.3,1x,f6.3,2(1x,e8.2),2(2x,f2.0),2x,I2)
c
        return
        end

c=======================================================================
clin-6/2009 embed back-to-back high-Pt quark/antiquark pair
c     via embedding back-to-back high-Pt pion pair then melting the pion pair
c     by generating the internal quark and antiquark momentum parallel to 
c      the pion momentum (in order to produce a high-Pt and a low Pt parton):
      subroutine embedHighPt
      PARAMETER (MAXSTR=150001,MAXR=1,pichmass=0.140,pi0mass=0.135,
     1     pi=3.1415926,nxymax=10001)
      common/embed/iembed,nsembd,pxqembd,pyqembd,xembd,yembd,
     1     psembd,tmaxembd,phidecomp
      COMMON/RNDF77/NSEED
      COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
      COMMON/HMAIN2/KATT(MAXSTR,4),PATT(MAXSTR,4)
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
      common/anim/nevent,isoft,isflag,izpc
      COMMON /AREVT/ IAEVT, IARUN, MISS
      common/xyembed/nxyjet,xyjet(nxymax,2)
      SAVE
c
      if(iembed.eq.1.or.iembed.eq.2) then
         xjet=xembd
         yjet=yembd
      elseif(iembed.eq.3.or.iembed.eq.4) then
         if(nevent.le.nxyjet) then
            read(97,*) xjet,yjet
         else
            ixy=mod(IAEVT,nxyjet)
            if(ixy.eq.0) ixy=nxyjet
            xjet=xyjet(ixy,1)
            yjet=xyjet(ixy,2)
         endif
      else
         return
      endif
c
      ptq=sqrt(pxqembd**2+pyqembd**2)
      if(ptq.lt.(pichmass/2.)) then
         print *, 'Embedded quark transverse momentum is too small'
         stop
      endif
c     Randomly embed u/ubar or d/dbar at high Pt:
      idqembd=1+int(2*RANART(NSEED))
c     Flavor content for the charged pion that contains the leading quark:
      if(idqembd.eq.1) then 
         idqsoft=-2
         idpi1=-211
      elseif(idqembd.eq.2) then 
         idqsoft=-1
         idpi1=211
      else
         print *, 'Wrong quark flavor embedded'
         stop
      endif
c     Caculate transverse momentum of the parent charged pion:
      xmq=ulmass(idqembd)
      xmqsoft=ulmass(idqsoft)
      ptpi=((pichmass**2+xmq**2-xmqsoft**2)*ptq
     1     -sqrt((xmq**2+ptq**2)*(pichmass**4
     2     -2.*pichmass**2*(xmq**2+xmqsoft**2)+(xmq**2-xmqsoft**2)**2)))
     3     /(2.*xmq**2)
      if(iembed.eq.1.or.iembed.eq.3) then
         pxpi1=ptpi*pxqembd/ptq
         pypi1=ptpi*pyqembd/ptq
         phidecomp=acos(pxqembd/ptq)
         if(pyqembd.lt.0) phidecomp=2.*pi-phidecomp
      else
         phidecomp=2.*pi*RANART(NSEED)
         pxpi1=ptpi*cos(phidecomp)
         pypi1=ptpi*sin(phidecomp)
      endif
c     Embedded quark/antiquark are assumed to have pz=0:
      pzpi1=0.
c     Insert the two parent charged pions, 
c     ipion=1 for the pion containing the leading quark, 
c     ipion=2 for the pion containing the leading antiquark of the same flavor:
      do ipion=1,2
         if(ipion.eq.1) then
            idpi=idpi1
            pxpi=pxpi1
            pypi=pypi1
            pzpi=pzpi1
         elseif(ipion.eq.2) then
            idpi=-idpi1
            pxpi=-pxpi1
            pypi=-pypi1
            pzpi=-pzpi1
         endif
         NATT=NATT+1
         KATT(NATT,1)=idpi
         KATT(NATT,2)=40
         KATT(NATT,3)=0
         PATT(NATT,1)=pxpi
         PATT(NATT,2)=pypi
         PATT(NATT,3)=pzpi
         PATT(NATT,4)=sqrt(pxpi**2+pypi**2+pzpi**2+pichmass**2)
         EATT=EATT+PATT(NATT,4)
         GXAR(NATT)=xjet
         GYAR(NATT)=yjet
         GZAR(NATT)=0.
         FTAR(NATT)=0.
         ITYPAR(NATT)=KATT(NATT,1) 
         PXAR(NATT)=PATT(NATT,1)
         PYAR(NATT)=PATT(NATT,2)
         PZAR(NATT)=PATT(NATT,3)
         PEAR(NATT)=PATT(NATT,4)
         XMAR(NATT)=pichmass
      enddo
c
clin-8/2009
c     Randomly embed a number of soft pions around each high-Pt quark in pair:
      if(nsembd.gt.0) then
         do ipion=1,2
            do ispion=1,nsembd
               idsart=3+int(3*RANART(NSEED))
               if(idsart.eq.3) then 
                  pimass=pichmass
                  idpis=-211
               elseif(idsart.eq.4) then 
                  pimass=pi0mass
                  idpis=111
               else
                  pimass=pichmass
                  idpis=211
               endif
               NATT=NATT+1
               KATT(NATT,1)=idpis
               KATT(NATT,2)=40
               KATT(NATT,3)=0
c     theta: relative angle between soft pion & associated high-Pt q or qbar,
c     generate theta and phi uniformly:
c     Note: it is not generated uniformly in solid angle because that gives 
c     a valley at theta=0, unlike the jet-like correlation (a peak at theta=0).
               theta=tmaxembd*RANART(NSEED)
               phi=2.*pi*RANART(NSEED)
               pxspi=psembd*sin(theta)*cos(phi)
               pyspi=psembd*sin(theta)*sin(phi)
               pzspi=psembd*cos(theta)
               if(ipion.eq.1) then
                  call rotate(pxpi1,pypi1,pzpi1,pxspi,pyspi,pzspi)
               else
                  call rotate(-pxpi1,-pypi1,-pzpi1,pxspi,pyspi,pzspi)
               endif
ctest off
c               write(99,*) "2  ",pxspi,pyspi,pzspi
               PATT(NATT,1)=pxspi
               PATT(NATT,2)=pyspi
               PATT(NATT,3)=pzspi
               PATT(NATT,4)=sqrt(psembd**2+pimass**2)
               EATT=EATT+PATT(NATT,4)
               GXAR(NATT)=xjet
               GYAR(NATT)=yjet
               GZAR(NATT)=0.
               FTAR(NATT)=0.
               ITYPAR(NATT)=KATT(NATT,1) 
               PXAR(NATT)=PATT(NATT,1)
               PYAR(NATT)=PATT(NATT,2)
               PZAR(NATT)=PATT(NATT,3)
               PEAR(NATT)=PATT(NATT,4)
               XMAR(NATT)=pimass
            enddo
         enddo
      endif
clin-8/2009-end
c
      return
      end
