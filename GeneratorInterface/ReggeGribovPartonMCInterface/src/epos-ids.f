c-----------------------------------------------------------------------
      subroutine iclass(id,icl)
c-----------------------------------------------------------------------
c      determines hadron class
c-----------------------------------------------------------------------
      ida=iabs(id)
      if(ida.eq.0.or.(ida.ge.17.and.ida.le.19))then
       icl=2
      elseif(ida.eq.130.or.ida.eq.230.or.ida.eq.20)then
       icl=3
      elseif(ida.eq.140.or.ida.eq.240.or.ida.eq.340.or.ida/10.eq.44)then
       icl=4
      elseif(ida.ge.100.and.ida.le.999)then
       icl=1
      elseif(ida.ge.1000.and.ida.le.9999)then
       icl=2
      else
       stop'iclass: id not known'
      endif
      end

c-----------------------------------------------------------------------
      subroutine idchrg(id,chrg)
c     computes charge of particle with ident code id
c     ichrg must be dimensioned nqlep+12
c-----------------------------------------------------------------------
      dimension ichrg(53),ifl(3)
      data ichrg/0,2,-1,-1,2,-1,2,-1,2,0,0,0,-3,0,-3,0,-3,1,1,2,2*0
     *,2,-1,-1,2,-1,2,-1,2,0,0,0,-3,0,-3,0,-3,0,-3,3,0
     *,3,0,0,0,3,3,3,6,6,6,0/
      idabs=iabs(id)
      call idflav(id,ifl(1),ifl(2),ifl(3),jspin,ind)
      if(idabs.lt.100) goto 200
      isum=0
      do 100 i=1,3
      if(abs(ifl(i)).gt.52)goto 100
      isum=isum+ichrg(iabs(ifl(i))+1)*isign(1,ifl(i))
  100 continue
      chrg=isum/3.
      return
200   chrg=ichrg(ind+1)*isign(1,id)
      chrg=chrg/3.
      return
      end

c-----------------------------------------------------------------------
      subroutine idspin(id,iso,jspin,istra)
c     computes iso (isospin), jspin and istra (strangeness) of particle
c     with ident code id
c-----------------------------------------------------------------------
      include 'epos.inc'
      dimension ifl(3)
      iso=0
      jspin=0
      istra=0
      idabs=abs(id)
      if(idabs.le.nflav)then
        iso=isospin(idabs)*sign(1,id)
        if(idabs.ge.3)istra=sign(1,id)
        return
      endif
      call idflav(id,ifl(1),ifl(2),ifl(3),jspin,ind)
      iq1=abs(ifl(1))
      iq2=abs(ifl(2))
      iq3=abs(ifl(3))
      if(iq1.ge.3)istra=istra+sign(1,ifl(1))
      if(iq2.ge.3)istra=istra+sign(1,ifl(2))
      if(iq3.ge.3)istra=istra+sign(1,ifl(3))
      if(iq1.ne.0)then         !baryon
        iso=(isospin(iq1)+isospin(iq2)+isospin(iq3))*sign(1,iq1)
      else
        iso=isospin(iq2)*sign(1,ifl(2))
        iso=iso+isospin(iq3)*sign(1,ifl(3))
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine idcomk(ic)
c     compactifies ic
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer ic(2),icx(2),jc(nflav,2)
      call idcomp(ic,icx,jc,1)
      ic(1)=icx(1)
      ic(2)=icx(2)
      return
      end

cc-----------------------------------------------------------------------
c      subroutine idcomi(ic,icx)
cc     compactifies ic
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      integer ic(2),icx(2),jc(nflav,2)
c      call idcomp(ic,icx,jc,1)
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine idcomj(jc)
c     compactifies jc
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer ic(2),icx(2),jc(nflav,2)
      call idcomp(ic,icx,jc,2)
      return
      end

c-----------------------------------------------------------------------
      subroutine idcomp(ic,icx,jc,im)
c-----------------------------------------------------------------------
c     compactifies ic,jc
c     input: im (1 or 2)
c            ic (if im=1)
c            jc (if im=2)
c     output: icx (if im=1)
c             jc
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer ic(2),icx(2),jc(nflav,2)
      if(im.eq.1)call iddeco(ic,jc)
      icx(1)=0
      icx(2)=0
           do n=1,nflav
           do j=1,2
      if(jc(n,j).ne.0)goto1
           enddo
           enddo
      return
1     continue
      nq=0
      na=0
           do n=1,nflav
      nq=nq+jc(n,1)
      na=na+jc(n,2)
           enddo
      l=0
           do n=1,nflav
      k=min0(jc(n,1),jc(n,2))
      if(nq.eq.1.and.na.eq.1)k=0
      jc(n,1)=jc(n,1)-k
      jc(n,2)=jc(n,2)-k
      if(jc(n,1).lt.0.or.jc(n,2).lt.0)
     *call utstop('idcomp: jc negative&',
     +sizeof('idcomp: jc negative&'))
      l=l+jc(n,1)+jc(n,2)
           enddo
           if(l.eq.0)then
      jc(1,1)=1
      jc(1,2)=1
           endif
           if(im.eq.1)then
      call idenco(jc,icx,ireten)
      if(ireten.eq.1)call utstop('idcomp: idenco ret code = 1&',
     +sizeof('idcomp: idenco ret code = 1&'))
           endif
      return
      end

c-----------------------------------------------------------------------
      subroutine iddeco(ic,jc)
c     decode particle id
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2),ic(2)
      ici=ic(1)
      jc(6,1)=mod(ici,10)
      jc(5,1)=mod(ici/10,10)
      jc(4,1)=mod(ici/100,10)
      jc(3,1)=mod(ici/1000,10)
      jc(2,1)=mod(ici/10000,10)
      jc(1,1)=mod(ici/100000,10)
      ici=ic(2)
      jc(6,2)=mod(ici,10)
      jc(5,2)=mod(ici/10,10)
      jc(4,2)=mod(ici/100,10)
      jc(3,2)=mod(ici/1000,10)
      jc(2,2)=mod(ici/10000,10)
      jc(1,2)=mod(ici/100000,10)
      return
      end

c-----------------------------------------------------------------------
      subroutine idenco(jc,ic,ireten)
c     encode particle id
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2),ic(2)
      ireten=0
      ic(1)=0
      do 20 i=1,nflav
      if(jc(i,1).ge.10)goto22
20    ic(1)=ic(1)+jc(i,1)*10**(nflav-i)
      ic(2)=0
      do 21 i=1,nflav
      if(jc(i,2).ge.10)goto22
21    ic(2)=ic(2)+jc(i,2)*10**(nflav-i)
      return
22    ireten=1
      ic(1)=0
      ic(2)=0
      return
      end

c-----------------------------------------------------------------------
      subroutine idenct(jc,id,ib1,ib2,ib3,ib4)
c     encode particle id
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2),ic(2)

      do 40 nf=1,nflav
      do 40 ij=1,2
      if(jc(nf,ij).ge.10)id=7*10**8
40    continue
           if(id/10**8.ne.7)then
      call idenco(jc,ic,ireten)
      if(ireten.eq.1)call utstop('idenct: idenco ret code = 1&',
     +sizeof('idenct: idenco ret code = 1&'))
      if(mod(ic(1),100).ne.0.or.mod(ic(2),100).ne.0)then
      id=9*10**8
      else
      id=8*10**8+ic(1)*100+ic(2)/100
      endif
           else
      call idtrbi(jc,ib1,ib2,ib3,ib4)
      id=id
     *+mod(jc(1,1)+jc(2,1)+jc(3,1)+jc(4,1),10**4)*10**4
     *+mod(jc(1,2)+jc(2,2)+jc(3,2)+jc(4,2),10**4)
           endif
      return
      end

c-----------------------------------------------------------------------
      subroutine idflav(id,ifl1,ifl2,ifl3,jspin,index)
c     unpacks the ident code id=+/-ijkl
c
c          mesons--
c          i=0, j<=k, +/- is sign for j
c          id=110 for pi0, id=220 for eta, etc.
c
c          baryons--
c          i<=j<=k in general
c          j<i<k for second state antisymmetric in (i,j), eg. l = 2130
c
c          other--
c          id=1,...,6 for quarks
c          id=9 for gluon
c          id=10 for photon
c          id=11,...,16 for leptons
c          i=17 for deuteron
c          i=18 for triton
c          i=19 for alpha
c          id=20 for ks, id=-20 for kl
c
c          i=21...26 for scalar quarks
c          i=29 for gluino
c
c          i=30 for h-dibaryon
c
c          i=31...36 for scalar leptons
c          i=39 for wino
c          i=40 for zino
c
c          id=80 for w+
c          id=81,...,83 for higgs mesons (h0, H0, A0, H+)
c          id=84,...,87 for excited bosons (Z'0, Z''0, W'+)
c          id=90 for z0
c
c          diquarks--
c          id=+/-ij00, i<j for diquark composed of i,j.
c
c
c          index is a sequence number used internally
c          (index=-1 if id doesn't exist)
c
c-----------------------------------------------------------------------
      parameter ( nqlep=41,nmes=2)
      ifl1=0
      ifl2=0
      ifl3=0
      jspin=0
      idabs=iabs(id)
      i=idabs/1000
      j=mod(idabs/100,10)
      k=mod(idabs/10,10)
      jspin=mod(idabs,10)
      if(id.ge.10000) goto 400
      if(id.ne.0.and.mod(id,100).eq.0) goto 300
      if(j.eq.0) goto 200
      if(i.eq.0) goto 100
c          baryons
c          only x,y baryons are qqx, qqy, q=u,d,s.
      ifl1=isign(i,id)
      ifl2=isign(j,id)
      ifl3=isign(k,id)
      if(k.le.6) then
        index=max0(i-1,j-1)**2+i+max0(i-j,0)+(k-1)*k*(2*k-1)/6
     1  +109*jspin+36*nmes+nqlep+11
      else
        index=max0(i-1,j-1)**2+i+max0(i-j,0)+9*(k-7)+91
     1  +109*jspin+36*nmes+nqlep+11
      endif
      return
c          mesons
100   continue
      ifl1=0
      ifl2=isign(j,id)
      ifl3=isign(k,-id)
      index=j+k*(k-1)/2+36*jspin+nqlep
      index=index+11
      return
c          quarks, leptons, etc
200   continue
      ifl1=0
      ifl2=0
      ifl3=0
      jspin=0
      index=idabs
      if(idabs.lt.20) return
c          define index=20 for ks, index=21 for kl
      index=idabs+1
      if(id.eq.20) index=20
c          index=nqlep+1,...,nqlep+11 for w+, higgs, z0
      if(idabs.lt.80) return
      index=nqlep+idabs-79
      return
300   ifl1=isign(i,id)
      ifl2=isign(j,id)
      ifl3=0
      jspin=0
      index=0
      return
 400  index=-1
      return
      end

c-----------------------------------------------------------------------
      subroutine idqufl(n,id,nqu,nqd,nqs)
c     unpacks the ident code of particle (n) and give the number of
c     quarks of each flavour(only u,d,s)
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      integer jc(nflav,2),ic(2)

      nqu=0
      nqd=0
      nqs=0
      if(iabs(id).ge.7.and.iabs(id).lt.100.and.iabs(id).ne.20)return
      if(iabs(id)/10.eq.11.or.iabs(id)/10.eq.22)return
      if(iabs(id).eq.20)then
        if(iorptl(n).gt.0)then
          if(idptl(iorptl(n)).gt.0)then
            nqd=1
            nqs=-1
          else
            nqd=-1
            nqs=1
          endif
        else
          if(ish.ge.4)write(ifch,*)'Cannot count the number of quark'
        endif
        return
      endif
      if(id.ne.0.and.mod(id,100).eq.0.and.id.le.10**8) goto 300
      if(id/10**8.ne.7)then
        call idtr4(id,ic)
        call iddeco(ic,jc)
      else
        call idtrb(ibptl(1,n),ibptl(2,n),ibptl(3,n),ibptl(4,n),jc)
      endif
      nqu=jc(1,1)-jc(1,2)
      nqd=jc(2,1)-jc(2,2)
      nqs=jc(3,1)-jc(3,2)
      return
 300  i=iabs(id)/1000
      j=mod(iabs(id)/100,10)
      ifl1=isign(i,id)
      ifl2=isign(j,id)
      if(iabs(ifl1).eq.1)nqu=isign(1,ifl1)
      if(iabs(ifl1).eq.2)nqd=isign(1,ifl1)
      if(iabs(ifl1).eq.3)nqs=isign(1,ifl1)
      if(iabs(ifl2).eq.1)nqu=nqu+isign(1,ifl2)
      if(iabs(ifl2).eq.2)nqd=nqd+isign(1,ifl2)
      if(iabs(ifl2).eq.3)nqs=nqs+isign(1,ifl2)
c      write(ifch,*)'id',id,ifl1,ifl2,nqu,nqd,nqs
      return
      end

c-----------------------------------------------------------------------
      function idlabl(id)
c     returns the character*8 label for the particle id
c-----------------------------------------------------------------------
      parameter ( nqlep=41,nmes=2)
c
      character*8 idlabl
      character*8 llep,lmes0,lmes1,lbar0,labar0,lbar1,labar1
      character*8 lqq,laqq
      dimension llep(104)
      dimension lmes0(64),lmes1(64)
      dimension lbar0(109),labar0(109),lbar1(109),labar1(109)
      dimension lqq(21),laqq(21)
c          diquark labels
      data lqq/
     1'uu0. ','ud0. ','dd0. ','us0. ','ds0. ','ss0. ','uc0. ','dc0. ',
     2'sc0. ','cc0. ','ub0. ','db0. ','sb0. ','cb0. ','bb0. ','ut0. ',
     3'dt0. ','st0. ','ct0. ','bt0. ','tt0. '/
      data laqq/
     1'auu0.','aud0.','add0.','aus0.','ads0.','ass0.','auc0.','adc0.',
     2'asc0.','acc0.','aub0.','adb0.','asb0.','acb0.','abb0.','aut0.',
     3'adt0.','ast0.','act0.','abt0.','att0.'/
c          quark and lepton labels
      data llep/
     *'     ','up   ','ub   ','dn   ','db   ','st   ','sb   ','ch   ',
     *'cb   ','bt   ','bb   ','tp   ','tb   ','y    ','yb   ','x    ',
     *'xb   ','gl   ','err  ','gm   ','err  ','nue  ','anue ','e-   ',
     *'e+   ','num  ','anum ','mu-  ','mu+  ','nut  ','anut ','tau- ',
     *'tau+ ','deut ','adeut','trit ','atrit','alph ','aalph','ks   ',
     *'err  ','err  ','kl   ',
     *'upss ','ubss ','dnss ','dbss ','stss ','sbss ','chss ','cbss ',
     *'btss ','bbss ','tpss ','tbss ','err  ','err  ','err  ','err  ',
     *'glss ','err  ','hdiba','err  ','ness ','aness','e-ss ','e+ss ',
     *'nmss ','anmss','mu-ss','mu+ss','ntss ','antss','t-ss ','t+ss ',
     *'err  ','err  ','err  ','err  ','w+ss ','w-ss ','z0ss ','err  ',
     *'w+   ','w-   ','h0   ','ah0  ','H0   ','aH0  ','A0   ','aA0  ',
     *'H+   ','H-   ','Zp0  ','aZp0 ','Zpp0 ','aZpp0','Wp+  ','Wp-  ',
     *'err  ','err  ','err  ','err  ','z0   '/
c          0- meson labels
      data lmes0/
     1'pi0  ','pi+  ','eta  ','pi-  ','k+   ','k0   ','etap ','ak0  ',
     2'k-   ','ad0  ','d-   ','f-   ','etac ','f+   ','d+   ','d0   ',
     2'ub.  ','db.  ','sb.  ','cb.  ','bb.  ','bc.  ','bs.  ','bd.  ',
     3'bu.  ','ut.  ','dt.  ','st.  ','ct.  ','bt.  ','tt.  ','tb.  ',
     4'tc.  ','ts.  ','td.  ','tu.  ','uy.  ','dy.  ','sy.  ','cy.  ',
     5'by.  ','ty.  ','yy.  ','yt.  ','yb.  ','yc.  ','ys.  ','yd.  ',
     6'yu.  ','ux.  ','dx.  ','sx.  ','cx.  ','bx.  ','tx.  ','yx.  ',
     7'xx.  ','xy.  ','xt.  ','xb.  ','xc.  ','xs.  ','xd.  ','xu.  '/
c          1- meson labels
      data lmes1/
     1'rho0 ','rho+ ','omeg ','rho- ','k*+  ','k*0  ','phi  ','ak*0 ',
     2'k*-  ','ad*0 ','d*-  ','f*-  ','jpsi ','f*+  ','d*+  ','d*0  ',
     3'ub*  ','db*  ','sb*  ','cb*  ','upsl ','bc*  ','bs*  ','bd*  ',
     4'bu*  ','ut*  ','dt*  ','st*  ','ct*  ','bt*  ','tt*  ','tb*  ',
     5'tc*  ','ts*  ','td*  ','tu*  ','uy*  ','dy*  ','sy*  ','cy*  ',
     6'by*  ','ty*  ','yy*  ','yt*  ','yb*  ','yc*  ','ys*  ','yd*  ',
     7'yu*  ','ux*  ','dx*  ','sx*  ','cx*  ','bx*  ','tx*  ','yx*  ',
     8'xx*  ','xy*  ','xt*  ','xb*  ','xc*  ','xs*  ','xd*  ','xu*  '/
c          1/2+ baryon labels
      data lbar0/
     1'err  ','p    ','n    ','err  ','err  ','s+   ','s0   ','s-   ',
     2'l    ','xi0  ','xi-  ','err  ','err  ','err  ','sc++ ','sc+  ',
     3'sc0  ','lc+  ','usc. ','dsc. ','ssc. ','sdc. ','suc. ','ucc. ',
     4'dcc. ','scc. ','err  ','err  ','err  ','err  ','uub. ','udb. ',
     5'ddb. ','dub. ','usb. ','dsb. ','ssb. ','sdb. ','sub. ','ucb. ',
     6'dcb. ','scb. ','ccb. ','csb. ','cdb. ','cub. ','ubb. ','dbb. ',
     7'sbb. ','cbb. ','err  ','err  ','err  ','err  ','err  ','utt. ',
     8'udt. ','ddt. ','dut. ','ust. ','dst. ','sst. ','sdt. ','sut. ',
     9'uct. ','dct. ','sct. ','cct. ','cst. ','cdt. ','cut. ','ubt. ',
     1'dbt. ','sbt. ','cbt. ','bbt. ','bct. ','bst. ','bdt. ','but. ',
     2'utt. ','dtt. ','stt. ','ctt. ','btt. ','err  ','err  ','err  ',
     3'err  ','err  ','err  ','uuy. ','udy. ','ddy. ','duy. ','usy. ',
     4'dsy. ','ssy. ','sdy. ','suy. ','uux. ','udx. ','ddx. ','dux. ',
     5'usx. ','dsx. ','ssx. ','sdx. ','sux. '/
      data labar0/
     1'err  ','ap   ','an   ','err  ','err  ','as-  ','as0  ','as+  ',
     2'al   ','axi0 ','axi+ ','err  ','err  ','err  ','asc--','asc- ',
     3'asc0 ','alc- ','ausc.','adsc.','assc.','asdc.','asuc.','aucc.',
     4'adcc.','ascc.','err  ','err  ','err  ','err  ','auub.','audb.',
     5'addb.','adub.','ausb.','adsb.','assb.','asdb.','asub.','aucb.',
     6'adcb.','ascb.','accb.','acsb.','acdb.','acub.','aubb.','adbb.',
     7'asbb.','acbb.','err  ','err  ','err  ','err  ','err  ','autt.',
     8'audt.','addt.','adut.','aust.','adst.','asst.','asdt.','asut.',
     9'auct.','adct.','asct.','acct.','acst.','acdt.','acut.','aubt.',
     1'adbt.','asbt.','acbt.','abbt.','abct.','abst.','abdt.','abut.',
     2'autt.','adtt.','astt.','actt.','abtt.','err  ','err  ','err  ',
     3'err  ','err  ','err  ','auuy.','audy.','addy.','aduy.','ausy.',
     4'adsy.','assy.','asdy.','asuy.','auux.','audx.','addx.','adux.',
     5'ausx.','adsx.','assx.','asdx.','asux.'/
c          3/2+ baryon labels
      data lbar1/
     1'dl++ ','dl+  ','dl0  ','dl-  ','err  ','s*+  ','s*0  ','s*-  ',
     2'err  ','xi*0 ','xi*- ','om-  ','err  ','err  ','uuc* ','udc* ',
     3'ddc* ','err  ','usc* ','dsc* ','ssc* ','err  ','err  ','ucc* ',
     4'dcc* ','scc* ','ccc* ','err  ','err  ','err  ','uub* ','udb* ',
     5'ddb* ','err  ','usb* ','dsb* ','ssb* ','err  ','err  ','ucb* ',
     6'dcb* ','scb* ','ccb* ','err  ','err  ','err  ','ubb* ','dbb* ',
     7'sbb* ','cbb* ','bbb* ','err  ','err  ','err  ','err  ','utt* ',
     8'udt* ','ddt* ','err  ','ust* ','dst* ','sst* ','err  ','err  ',
     9'uct* ','dct* ','sct* ','cct* ','err  ','err  ','err  ','ubt* ',
     1'dbt* ','sbt* ','cbt* ','bbt* ','err  ','err  ','err  ','err  ',
     2'utt* ','dtt* ','stt* ','ctt* ','btt* ','ttt* ','err  ','err  ',
     3'err  ','err  ','err  ','uuy* ','udy* ','ddy* ','err  ','usy* ',
     4'dsy* ','ssy* ','err  ','err  ','uux* ','udx* ','ddx* ','err  ',
     5'usx* ','dsx* ','ssx* ','err  ','err  '/
      data labar1/
     1'adl--','adl- ','adl0 ','adl+ ','err  ','as*- ','as*0 ','as*+ ',
     2'err  ','axi*0','axi*+','aom+ ','err  ','err  ','auuc*','audc*',
     3'addc*','err  ','ausc*','adsc*','assc*','err  ','err  ','aucc*',
     4'adcc*','ascc*','accc*','err  ','err  ','err  ','auub*','audb*',
     5'addb*','err  ','ausb*','adsb*','assb*','err  ','err  ','aucb*',
     6'adcb*','ascb*','accb*','err  ','err  ','err  ','aubb*','adbb*',
     7'asbb*','acbb*','abbb*','err  ','err  ','err  ','err  ','autt*',
     8'audt*','addt*','err  ','aust*','adst*','asst*','err  ','err  ',
     9'auct*','adct*','asct*','acct*','err  ','err  ','err  ','aubt*',
     1'adbt*','asbt*','acbt*','abbt*','err  ','err  ','err  ','err  ',
     2'autt*','adtt*','astt*','actt*','abtt*','attt*','err  ','err  ',
     3'err  ','err  ','err  ','auuy*','audy*','addy*','err  ','ausy*',
     4'adsy*','assy*','err  ','err  ','auux*','audx*','addx*','err  ',
     5'ausx*','adsx*','assx*','err  ','err  '/
c          entry
      call idflav(id,ifl1,ifl2,ifl3,jspin,ind)
      if(iabs(id).lt.100) goto200
      if(iabs(id).lt.1000) goto100
      if(id.ne.0.and.mod(id,100).eq.0) goto300
c          baryons
      ind=ind-109*jspin-36*nmes-nqlep
      ind=ind-11
      if(jspin.eq.0.and.id.gt.0) idlabl=lbar0(ind)
      if(jspin.eq.0.and.id.lt.0) idlabl=labar0(ind)
      if(jspin.eq.1.and.id.gt.0) idlabl=lbar1(ind)
      if(jspin.eq.1.and.id.lt.0) idlabl=labar1(ind)
      return
c          mesons
100   continue
      i=max0(ifl2,ifl3)
      j=-min0(ifl2,ifl3)
      ind=max0(i-1,j-1)**2+i+max0(i-j,0)
      if(jspin.eq.0) idlabl=lmes0(ind)
      if(jspin.eq.1) idlabl=lmes1(ind)
      return
c          quarks, leptons, etc.
200   continue
      ind=2*ind
      if(id.le.0) ind=ind+1
      idlabl=llep(ind)
      return
300   i=iabs(ifl1)
      j=iabs(ifl2)
      ind=i+j*(j-1)/2
      if(id.gt.0) idlabl=lqq(ind)
      if(id.lt.0) idlabl=laqq(ind)
      return
      end

c-----------------------------------------------------------------------
      subroutine idmass(idi,amass)
c     returns the mass of the particle with ident code id.
c     (deuteron, triton and alpha mass come from Gheisha ???)
c-----------------------------------------------------------------------
      dimension ammes0(15),ammes1(15),ambar0(30),ambar1(30)
      dimension amlep(52)
      parameter ( nqlep=41,nmes=2)
c-c   data amlep/.3,.3,.5,1.6,4.9,30.,-1.,-1.,0.,0.,
      data amlep/.005,.009,.180,1.6,4.9,170.,-1.,-1.,0.,0.,0.
     *     ,.5109989e-3,0.,.105658,0.,1.777,1.87656,2.8167,3.755,.49767
     *     ,.49767,100.3,100.3,100.5,101.6,104.9,130.,2*-1.,100.,0.,
     *     100.,100.005,100.,100.1,100.,101.8,2*-1.,100.,100.,
     *     11*0./
c          0- meson mass table
      data ammes0/.1349766,.13957018,.547853       !pi0,pi+-,eta
     *           ,.493677,.497614,.95778           !K+-, K0,eta'
     *    ,1.86483,1.86960,1.96847,2.9803          !D0,D+-,Ds,etac
     1    ,5.27917,5.27950,5.3663,6.277,9.390/     !B+-,B0,Bs,Bc,etab
c     1- meson mass table
      data ammes1/.77549,.77549,.78265             !rho0,rho+-,omega
     *           ,.889166,.89594,1.019455          !K*+-,K0*,phi
     1     ,2.00693,2.01022,2.1123,3.096916        !D0*,D*+-,D*s,j/psi
     *     ,5.3251,5.3251,5.4154,6.610,9.46030/    !B*+-,B0*,B*s,B*c,upsilon
c     1/2+ baryon mass table
      data ambar0/-1.,.93828,.93957,2*-1.,1.1894,1.1925,1.1974
     1     ,1.1156,1.3149,1.3213,3*-1.
     $     ,2.453               !15          sigma_c++!
     $     ,2.454               !            sigma_c+
     $     ,2.452               !            sigma_c0
     $     ,2.286               !            lambda_c+
     2     ,2.576               !19  1340   !Xi'_c+
     $     ,2.578               !20  2340   !Xi'_c0
     $     ,2.695               !21  3340   !omegac0
     $     ,2.471               !22  3240   !Xi_c0
     $     ,2.468               !23  3140   !Xi_c+
     $     ,3.55                !24  1440
     $     ,3.55                !25  2440
     $     ,3.70                !26  3440
     $     ,4*-1./
c     3/2+ baryon mass table
      data ambar1/1.232,1.232,1.232,1.232,-1.,1.3823,1.3820
     1     ,1.3875,-1.,1.5318,1.5350,1.6722,2*-1.
     2     ,2.519               !15          sigma_c++
     $     ,2.52                !            sigma_c+
     $     ,2.517               !            sigma_c0
     $     ,-1.
     $     ,2.645
     $     ,2.644
     $     ,2.80
     $     ,2*-1.
     $     ,3.75
     $     ,3.75
     3     ,3.90
     $     ,4.80
     $     ,3*-1./
c     entry
      id=idi
      amass=0.
ctp060829      if(iabs(id).eq.30)then
ctp060829        amass=amhdibar
ctp060829        return
ctp060829      endif
      if(idi.eq.0)then
        id=1120                 !for air target
      elseif(abs(idi).ge.1000000000)then
        goto 500                !nucleus
      endif
      if(idi.gt.10000)return
      call idflav(id,ifl1,ifl2,ifl3,jspin,ind)
      if(id.ne.0.and.mod(id,100).eq.0) goto400
      if(iabs(ifl1).ge.5.or.iabs(ifl2).ge.5.or.iabs(ifl3).ge.5)
     1     goto300
      if(ifl2.eq.0) goto200
      if(ifl1.eq.0) goto100
c          baryons
      ind=ind-109*jspin-36*nmes-nqlep
      ind=ind-11
      amass=(1-jspin)*ambar0(ind)+jspin*ambar1(ind)
      return
c          mesons
100   continue
      ind=ind-36*jspin-nqlep
      ind=ind-11
      amass=(1-jspin)*ammes0(ind)+jspin*ammes1(ind)
      return
c          quarks and leptons (+deuteron, triton, alpha, Ks and Kl)
200   continue
      amass=amlep(ind)
      return
c          b and t particles
300   continue
      amass=amlep(iabs(ifl2))+amlep(iabs(ifl3))+1.07+.045*jspin
      if(ifl1.ne.0) amass=amass+amlep(iabs(ifl1))
      return
c          diquarks
400   amass=amlep(iabs(ifl1))+amlep(iabs(ifl2))
      return
c          nuclei
500   nbrpro=mod(abs(id/10000),1000)
      nbrneu=mod(abs(id/10),1000)-nbrpro
      amass=nbrpro*ambar0(2)+nbrneu*ambar0(3)
      return
      end

cc-----------------------------------------------------------------------
c      subroutine idmix(ic,jspin,icm,idm)
cc     accounts for flavour mixing
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      real pmix1(3,2),pmix2(3,2)
c      integer ic(2),icm(2)
c      data pmix1/.25,.25,.5,0.,.5,1./,pmix2/.5,.5,1.,0.,0.,1./
c      icm(1)=0
c      icm(2)=0
c      idm=0
c      i=ic(1)
c      if(i.ne.ic(2))return
c      id=0
c      if(i.eq.100000)id=1
c      if(i.eq. 10000)id=2
c      if(i.eq.  1000)id=3
c      if(id.eq.0)return
c      rnd=rangen()
c      idm=int(pmix1(id,jspin+1)+rnd)+int(pmix2(id,jspin+1)+rnd)+1
c      icm(1)=10**(nflav-idm)
c      icm(2)=ic(1)
c      idm=idm*100+idm*10+jspin
c      return
c      end
c
cc-----------------------------------------------------------------------
c      subroutine idcleanjc(jc)
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      integer jc(nflav,2)
c      ns=0
c      do n=1,nflav
c        jj=min(jc(n,1),jc(n,2))
c        jc(n,1)=jc(n,1)-jj
c        jc(n,2)=jc(n,2)-jj
c        ns=ns+jc(n,1)+jc(n,2)
c      enddo
c      if(ns.eq.0)then
c        jc(1,1)=1
c        jc(1,2)=1
c      endif
c      end
c
c-----------------------------------------------------------------------
      subroutine idquacjc(jc,nqu,naq)
c     returns quark content of jc
c        jc(nflav,2) = jc-type particle identification code.
c        nqu = # quarks
c        naq = # antiquarks
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2)
      nqu=0
      naq=0
      do 53 n=1,nflav
        nqu=nqu+jc(n,1)
53      naq=naq+jc(n,2)
      return
      end

c-----------------------------------------------------------------------
      subroutine idquacic(ic,nqu,naq)
c     returns quark content of ic
c        ic(2) = ic-type particle identification code.
c        nqu = # quarks
c        naq = # antiquarks
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2),ic(2)
      nqu=0
      naq=0
      call iddeco(ic,jc)
      do 53 n=1,nflav
        nqu=nqu+jc(n,1)
53      naq=naq+jc(n,2)
      return
      end

c-----------------------------------------------------------------------
      subroutine idquac(i,nq,ns,na,jc)
c     returns quark content of ptl i from /cptl/ .
c        nq = # quarks - # antiquarks
c        ns = # strange quarks - # strange antiquarks
c        na = # quarks + # antiquarks
c        jc(nflav,2) = jc-type particle identification code.
c-----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incems'
      integer jc(nflav,2),ic(2)


      if(iabs(idptl(i)).eq.20)then
      idptl(i)=230
      if(rangen().lt..5)idptl(i)=-230
      goto9999
      endif

      if(iabs(idptl(i)).lt.100)then
      nq=0
      ns=0
      do 1 n=1,nflav
      jc(n,1)=0
1     jc(n,2)=0
      return
      endif
9999  if(idptl(i)/10**8.ne.7)then
      call idtr4(idptl(i),ic)
      call iddeco(ic,jc)
      else
      call idtrb(ibptl(1,i),ibptl(2,i),ibptl(3,i),ibptl(4,i),jc)
      endif
      na=0
      nq=0
      do 53 n=1,nflav
      na=na+jc(n,1)+jc(n,2)
53    nq=nq+jc(n,1)-jc(n,2)
      ns=   jc(3,1)-jc(3,2)
      return
      end

cc-----------------------------------------------------------------------
c      subroutine idquad(i,nq,na,jc)
cc-----------------------------------------------------------------------
cc     returns quark content of ptl i from /cptl/ .
cc        nq = # quarks - # antiquarks
cc        na = # quarks + # antiquarks
cc        jc(nflav,2) = jc-type particle identification code.
cc-----------------------------------------------------------------------
c      include 'epos.inc'
c      integer jc(nflav,2),ic(2)
c
c      id=idptl(i)
c      if(iabs(id).eq.20)then
c      id=230
c      if(rangen().lt..5)id=-230
c      goto9999
c      endif
c
c      if(iabs(id).lt.100)then
c      nq=0
cc      ns=0
c      do 1 n=1,nflav
c      jc(n,1)=0
c1     jc(n,2)=0
c      return
c      endif
c
c9999  if(id/10**8.ne.7)then
c      call idtr4(id,ic)
c      call iddeco(ic,jc)
c      else
c      call idtrb(ibptl(1,i),ibptl(2,i),ibptl(3,i),ibptl(4,i),jc)
c      endif
c      na=0
c      nq=0
c      do 53 n=1,nflav
c      na=na+jc(n,1)+jc(n,2)
c53    nq=nq+jc(n,1)-jc(n,2)
cc      ns=   jc(3,1)-jc(3,2)
c      return
c      end
c
c-----------------------------------------------------------------------
      integer function idraflx(proba,xxx,qqs,icl,jc,jcval,j,iso,c)
c-----------------------------------------------------------------------
c     returns random flavor, according to jc and GRV structure function
c for x(=xxx) and Q2(=qqs) for valence quark (jcval) and sea quarks
c and update jc with quark-antiquark cancellation
c             jc : quark content of remnant
c     j=1 quark, j=2 antiquark,
c     iso : isospin
c     proba : probability of the selected quark (output)
c     c     : "v" is to choose a valence quark, "s" a sea or valence quark
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),jcval(nflav,2)
      double precision s,puv,pdv,psv,pcv,pus,pds,pss,pcs,piso,proba
      character c*1

       if(ish.ge.8)then
         write(ifch,10)c,j,xxx,qqs,jc,jcval
       endif
 10    format('entry idraflx, j,x,q: ',a1,i2,2g13.5,/
     &  ,15x,'jc :',2(1x,6i2),/,14x,'jcv :',2(1x,6i2))

       puv=dble(jcval(1,j))
       pdv=dble(jcval(2,j))
       psv=dble(jcval(3,j))
       pcv=dble(jcval(4,j))
       if(c.eq."v")then
         pus=0d0
         pds=0d0
         pss=0d0
         pcs=0d0
       else
         pus=dble(jc(1,j))-puv
         pds=dble(jc(2,j))-pdv
         pss=dble(jc(3,j))-psv
         pcs=dble(jc(4,j))-pcv
       endif

      if(ish.ge.8)then
        write(ifch,'(a,4f6.3)')'idraflx valence:',puv,pdv,psv,pcv
        write(ifch,'(a,4f6.3)')'idraflx sea:',pus,pds,pss,pcs
      endif

       qq=0.
       if(iso.gt.0)then
         if(icl.eq.2)puv=puv*0.5d0   !because GRV already take into account the fact that there is 2 u quark in a proton
         puv=puv*dble(psdfh4(xxx,qqs,qq,icl,1))
         pus=pus*dble(psdfh4(xxx,qqs,qq,icl,-1))
         pdv=pdv*dble(psdfh4(xxx,qqs,qq,icl,2))
         pds=pds*dble(psdfh4(xxx,qqs,qq,icl,-2))
       elseif(iso.lt.0)then
         puv=puv*dble(psdfh4(xxx,qqs,qq,icl,2))
         pus=pus*dble(psdfh4(xxx,qqs,qq,icl,-2))
         if(icl.eq.2)pdv=pdv*0.5d0
         pdv=pdv*dble(psdfh4(xxx,qqs,qq,icl,1))
         pds=pds*dble(psdfh4(xxx,qqs,qq,icl,-1))
       else
         piso=(dble(psdfh4(xxx,qqs,qq,icl,1))
     &        +dble(psdfh4(xxx,qqs,qq,icl,2)))
         if(icl.eq.2)then   !3 quarks
           piso=piso/3d0
         else               !2 quarks
           piso=piso/2d0
         endif
         puv=puv*piso
         pdv=pdv*piso
         piso=0.5d0*(dble(psdfh4(xxx,qqs,qq,icl,-1))
     &              +dble(psdfh4(xxx,qqs,qq,icl,-2)))
         pus=pus*piso
         pds=pds*piso
       endif
       psv=psv*dble(psdfh4(xxx,qqs,qq,icl,3))
       pss=pss*dble(psdfh4(xxx,qqs,qq,icl,-3))
       if(nrflav.ge.4)then
         pcv=pcv*dble(psdfh4(xxx,qqs,qq,icl,4))
         pcs=pcs*dble(psdfh4(xxx,qqs,qq,icl,-4))
       else
         pcv=0d0
         pcs=0d0
       endif

      if(ish.ge.8)then
        write(ifch,'(a,4f6.3)')'idraflx P(valence):',puv,pdv,psv,pcv
        write(ifch,'(a,4f6.3)')'idraflx P(sea):',pus,pds,pss,pcs
      endif

      s=puv+pdv+psv+pcv+pus+pds+pss+pcs
      if(s.gt.0.)then
       r=rangen()*s
       if(r.gt.(pdv+pus+pds+pss+psv+pcv+pcs).and.puv.gt.0.)then
        i=1
        jcval(i,j)=jcval(i,j)-1
        proba=puv
       elseif(r.gt.(pus+pds+pss+psv+pcv+pcs).and.pdv.gt.0.)then
        i=2
        jcval(i,j)=jcval(i,j)-1
        proba=pdv
       elseif(r.gt.(pds+pss+psv+pcv+pcs).and.pus.gt.0.)then
        i=1
        proba=pus
       elseif(r.gt.(pss+psv+pcv+pcs).and.pds.gt.0.)then
        i=2
        proba=pds
       elseif(r.gt.(psv+pcv+pcs).and.pss.gt.0.)then
        i=3
        proba=pss
       elseif(r.gt.(pcv+pcs).and.psv.gt.0.)then
        i=3
        jcval(i,j)=jcval(i,j)-1
        proba=psv
       elseif(r.gt.pcs.and.pcv.gt.0.)then
        i=4
        jcval(i,j)=jcval(i,j)-1
        proba=pcv
       elseif(pcs.gt.0.)then
        i=4
        proba=pcs
       else
        call utstop("Problem in idraflx, should not be !&",
     +sizeof("Problem in idraflx, should not be !&"))
       endif
      else
        i=idrafl(icl,jc,j,"v",0,iretso)      !no update of jc here
        if(jc(i,j)-jcval(i,j).lt.1)jcval(i,j)=jcval(i,j)-1
        proba=0d0
      endif
      idraflx=i

      if(ish.ge.8)then
        write(ifch,'(a,2(1x,6i2))')'jc before updating:',jc
        write(ifch,20)i,j,jcval,proba
      endif
 20   format('i,j|jcval|P:',2i3,' |',2(1x,6i2),' |',g15.3)

      call idsufl3(i,j,jc)

      if(ish.ge.8)
     & write(ifch,'(a,2(1x,6i2))')'jc after updating:',jc

      return
      end

c-----------------------------------------------------------------------
      integer function idrafl(icl,jc,j,c,imod,iretso)
c-----------------------------------------------------------------------
c     returns random flavor,
c     if : c='v' : according to jc
c          c='s' : from sea
c          c='r' : from sea (always without c quarks)
c          c='d' : from sea for second quark in diquark
c          c='c' : take out c quark first
c             jc : quark content of remnant
c     j=1 quark, j=2 antiquark,
c     imod=0     : returns random flavor of a quark
c     imod=1     : returns random flavor of a quark and update jc
c                 (with quark-antiquark cancellation
c     imod=2     : returns random flavor of a quark and update jc
c                 (without quark-antiquak cancellation -> accumulate quark)
c     imod=3     : returns random flavor of a quark and update jc with
c                  the corresponding quark-antiquark pair
c                 (without quark-antiquak cancellation -> accumulate quark)
c
c     iretso=0   : ok
c           =1   : more than 9 quarks of same flavor attempted
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2),ic(2)
      character c*1

c       write(ifch,*)'entry idrafl, j,imod,c: ',j,imod,' ',c

      pui=1.
      if(c.eq.'s')then
        pu=pui
        pd=pui*exp(-pi*difud/fkappa)
        ps=pui*exp(-pi*difus/fkappa)
        pc=pui*exp(-pi*difuc/fkappa)
        pu=rstrau(icl)*pu
        pd=rstrad(icl)*pd
        ps=rstras(icl)*ps
        pc=rstrac(icl)*pc
      elseif(c.eq.'d')then
        pu=pui*exp(-pi*difuuu/fkappa)
        pd=pui*exp(-pi*difudd/fkappa)
        ps=pui*exp(-pi*difuss/fkappa)
        pc=pui*exp(-pi*difucc/fkappa)
        pu=pu*rstrau(icl)
        pd=pd*rstrad(icl)
        ps=ps*rstras(icl)
        pc=pc*rstrac(icl)
      elseif(c.eq.'v')then
        pu=float(jc(1,j))
        pd=float(jc(2,j))
        ps=float(jc(3,j))
        pc=float(jc(4,j))
      elseif(c.eq.'r')then
        pu=1.
        pd=1.
        ps=1.
        pc=0.
        pu=rstrau(icl)*pu
        pd=rstrad(icl)*pd
        ps=rstras(icl)*ps
      elseif(c.eq.'c')then
        pu=0.
        pd=0.
        ps=0.
        pc=1.
      else
        stop'idrafl: dunnowhatodo'
      endif

c      write(ifch,*)'idrafl',pu,pd,ps

      s=pu+pd+ps+pc
      if(s.gt.0.)then
       r=rangen()*s
       if(r.gt.(pu+pd+ps).and.pc.gt.0d0)then
        i=4
       elseif(r.gt.(pu+pd).and.ps.gt.0d0)then
        i=3
       elseif(r.gt.pu.and.pd.gt.0d0)then
        i=2
       else
        i=1
       endif
      elseif(iremn.le.1.or.c.ne.'v')then
        i=1+min(2,int((2.+rstras(icl))*rangen()))
      else
        idrafl=0
        return
      endif
      idrafl=i

c      write(ifch,*)'jc before updating',jc
c      write(ifch,*)'i,j,jc',i,j,jc

      if(imod.eq.1)then
        if(iremn.eq.2)then
          call idsufl3(i,j,jc)
c   be sure that jc is not empty
          if(jc(i,j).eq.0)then
            call idenco(jc,ic,iret)
            if(iret.eq.0.and.ic(1).eq.0.and.ic(2).eq.0)then
              jc(i,j)=jc(i,j)+1
              jc(i,3-j)=jc(i,3-j)+1
              iretso=1
            endif
          endif
        elseif(iremn.eq.3)then
          call idsufl3(i,j,jc)
        else
          call idsufl(i,j,jc,iretso)
          if(iretso.ne.0.and.ish.ge.2)then
            call utmsg('idrafl')
            write(ifmt,*)'iret none 0 in idrafl',iretso
            write(ifch,*)'iret none 0 in idrafl',iretso
            call utmsgf
          endif
        endif
      elseif(imod.eq.2)then
        call idsufl2(i,j,jc)    !do not cancel antiquarks with quarks
      elseif(imod.eq.3)then
        call idsufl2(i,1,jc)    !do not cancel antiquarks with quarks
        call idsufl2(i,2,jc)    !do not cancel antiquarks with quarks
      endif


c      write(ifch,*)'jc after updating',jc

      return
      end


c-----------------------------------------------------------------------
      integer function idraflz(jc,j)
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2)

      pu=float(jc(1,j))
      pd=float(jc(2,j))
      ps=float(jc(3,j))
      pc=float(jc(4,j))

      s=pu+pd+ps+pc
      if(s.gt.0.)then
       r=rangen()*s
       if(r.gt.(pu+pd+ps).and.pc.gt.0d0)then
        i=4
       elseif(r.gt.(pu+pd).and.ps.gt.0d0)then
        i=3
       elseif(r.gt.pu.and.pd.gt.0d0)then
        i=2
       else
        i=1
       endif
      else
       stop'in idraflz (1)                      '
      endif
      idraflz=i

      if(jc(i,j).lt.1)stop'in idraflz (2)              '
      jc(i,j)=jc(i,j)-1

      end

c-----------------------------------------------------------------------
      subroutine idsufl(i,j,jc,iretso)
c-----------------------------------------------------------------------
c subtract flavor i, j=1 quark, j=2 antiquark
c add antiflavor if jc(i,j)=0
c iretso=0  ok
c       =1 : more than 9 quarks of same flavor attempted
c-----------------------------------------------------------------------
      integer jc(6,2),ic(2)

      if(jc(i,j).gt.0)then
       jc(i,j)=jc(i,j)-1
       call idenco(jc,ic,iret)
       if(ic(1).eq.0.and.ic(2).eq.0)then
         jc(i,j)=jc(i,j)+1
         if(jc(i,3-j).lt.9.and.iret.eq.0)then
           jc(i,3-j)=jc(i,3-j)+1
         else
           iretso=1
         endif
       endif
      else
        if(j.eq.1)then
          if(jc(i,2).lt.9)then
            jc(i,2)=jc(i,2)+1
          else
            iretso=1
          endif
        else
          if(jc(i,1).lt.9)then
            jc(i,1)=jc(i,1)+1
          else
            iretso=1
          endif
        endif
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine idsufl2(i,j,jc)
c-----------------------------------------------------------------------
c substract flavor i, by adding antiquark i, j=1 quark, j=2 antiquark
c Can replace idsufl if we don't want to cancel quarks and antiquarks
c
c Warning : No protection against jc(i,j)>9 ! should not encode jc without test
c
c-----------------------------------------------------------------------
      parameter(nflav=6)
      integer jc(nflav,2)

      jc(i,3-j)=jc(i,3-j)+1

      return
      end

c-----------------------------------------------------------------------
      subroutine idsufl3(i,j,jc)
c-----------------------------------------------------------------------
c subtract flavor i, j=1 quark, j=2 antiquark
c add antiflavor if jc(i,j)=0
c
c Warning : No protection against jc(i,j)>9 ! should not encode jc without test
c
c-----------------------------------------------------------------------
      parameter(nflav=6)
      integer jc(nflav,2)

      if(jc(i,j).gt.0)then
        jc(i,j)=jc(i,j)-1
      else
        jc(i,3-j)=jc(i,3-j)+1
      endif

      return
      end

cc-----------------------------------------------------------------------
c      subroutine idchfl(jc1,jc2,iret)
cc-----------------------------------------------------------------------
cc checks whether jc1 and jc2 have the same number of quarks and antiquarks
cc if yes: iret=0, if no: iret=1
cc-----------------------------------------------------------------------
c      integer jc1(6,2),jc2(6,2)
c
c      iret=0
c
c      do j=1,2
c       n1=0
c       n2=0
c       do i=1,6
c        n1=n1+jc1(i,j)
c        n2=n2+jc2(i,j)
c       enddo
c       if(n1.ne.n2)then
c        iret=1
c        return
c       endif
c      enddo
c
c      end
c
c
c-----------------------------------------------------------------------
      subroutine idres(idi,am,idr,iadj)
c     returns resonance id idr corresponding to mass am.
c     performs mass adjustment, if necessary (if so iadj=1, 0 else)
c     (only for mesons and baryons, error (stop) otherwise)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxindx=1000,mxre=100,mxma=11,mxmx=6)
      common/crema/indx(mxindx),rema(mxre,mxma),rewi(mxre,mxma)
     *,idmx(mxma,mxmx),icre1(mxre,mxma),icre2(mxre,mxma)
      character cad*10

      write(cad,'(i10)')idi
      iadj=0
      idr=0
      if(idi.eq.10)return
      if(abs(am).lt.1.e-5)am=1e-5
      id=idi
      ami=am
      if(am.lt.0.)then
        call idmass(id,am)
        iadj=1
        if(am.le.0.)then
        write(ifch,*)'*****  warning in idres (0): '
     *,'neg mass returned from idmass'
        write(ifch,*)'id,am(input):',idi,ami
        am=1e-5
        endif
      endif

      if(id.eq.0)goto 9999
      if(abs(id).eq.20)id=sign(230,idi)
      m1=1
      if(iabs(id).ge.1000)m1=3
      m2=2
      if(iabs(id).ge.1000)m2=mxmx
      do 3 k=m1,m2
      do 3 m=2,mxma
        if(iabs(id).eq.idmx(m,k)) then
          id=idmx(1,k)*10*id/iabs(id)
          goto 43
        endif
 3    continue
 43   continue
      ix=iabs(id)/10
      if(ix.lt.1.or.ix.gt.mxindx)then
        call utstop('idres: ix out of range. id='//cad//'&',
     +sizeof('idres: ix out of range. id='//cad//'&'))
      endif
      i=indx(ix)
      if(i.lt.1.or.i.gt.mxre)then
        write(ifch,*)'idres problem',id,am
        call utstop('idres: particle not in table&',
     +sizeof('idres: particle not in table&'))
      endif
      do 1 j=1,mxma-1
      if(am.ge.rema(i,j).and.am.le.rema(i,j+1))then
      if(j-1.gt.9)call utstop('idres: spin > 9&',
     +sizeof('idres: spin > 9&'))
      idr=id/10*10+(j-1)*id/iabs(id)
      goto 2
      endif
1     continue
      goto 9999
2     continue

      do 4 k=1,mxmx
      if(ix.eq.idmx(1,k))then
      if(j.lt.1.or.j.gt.mxma-1)
     *call utstop('idres: index j out of range&',
     +sizeof('idres: index j out of range&'))
      if(idmx(j+1,k).ne.0)idr=idmx(j+1,k)*id/iabs(id)
      endif
4     continue

      iy=mod(iabs(idr),10)
      if(iy.gt.maxres)then
      iadj=0
      idr=0
      goto 9999
      endif

      if(iy.ne.0.and.iy.ne.1)goto 9999

      call idmass(idr,am)
      if(am.lt.0.)then
      write(ifch,*)'*****  error in idres: '
     *,'neg mass returned from idmass'
      write(ifch,*)'id,am(input):',idi,ami
      write(ifch,*)'idr,am:',idr,am
      call utstop('idres: neg mass returned from idmass&',
     +sizeof('idres: neg mass returned from idmass&'))
      endif
      del=max(1.e-3,2.*rewi(i,j))
      if(abs(ami-am).gt.del)iadj=1
c      write(ifch,*)'res:',id,idr,ami,am,rewi(i,j),iadj

9999  if(.not.(ish.ge.8))return
      write(ifch,*)'return from idres. id,ami,am,idr,iadj:'
      write(ifch,*)idi,ami,am,idr,iadj
      return
      end

c-----------------------------------------------------------------------
      subroutine idresi
c-----------------------------------------------------------------------
c  initializes /crema/
c  masses are limit between stable state (so the average between 2 mass states)
c  width=hbar(6.582e-25 GeV.s)/tau for 151, 251, 351, 451 arbitrary
c  (no data found) !!!!!!!!!!!
c-----------------------------------------------------------------------

      parameter (mxindx=1000,mxre=100,mxma=11,mxmx=6)
      common/crema/indx(mxindx),rema(mxre,mxma),rewi(mxre,mxma)
     *,idmx(mxma,mxmx),icre1(mxre,mxma),icre2(mxre,mxma)
      parameter (n=35)
      dimension remai(n,mxma),rewii(n,mxma),idmxi(mxma,mxmx)
     *,icrei(n,2*mxma)

      data (idmxi(j,1),j=1,mxma)/ 11, 110, 111,   0,   0,   0,   0, 4*0/
      data (idmxi(j,2),j=1,mxma)/ 22, 220, 330, 331,   0,   0,   0, 4*0/
      data (idmxi(j,3),j=1,mxma)/123,2130,1230,1231,1233,1234,1235, 4*0/
      data (idmxi(j,4),j=1,mxma)/124,2140,1240,1241,   0,   0,   0, 4*0/
      data (idmxi(j,5),j=1,mxma)/134,3140,1340,1341,   0,   0,   0, 4*0/
      data (idmxi(j,6),j=1,mxma)/234,3240,2340,2341,   0,   0,   0, 4*0/

      data ((icrei(k,m),m=1,2*mxma),k=1,10)/
     *111,000000, 9*300000,    11*0,
     *222,000000, 9*030000,    11*0,
     *112,       10*210000,    11*0,
     *122,       10*120000,    11*0,
     *113,       10*201000,    11*0,
     *223,       10*021000,    11*0,
     *123,       10*111000,    11*0,
     *133,       10*102000,    11*0,
     *233,       10*012000,    11*0,
     *333,000000, 9*003000,    11*0/
      data ((icrei(k,m),m=1,2*mxma),k=11,20)/
     *114,       10*200100,    11*0,
     *124,       10*110100,    11*0,
     *224,       10*020100,    11*0,
     *134,       10*101100,    11*0,
     *234,       10*011100,    11*0,
     *334,       10*002100,    11*0,
     *144,       10*100200,    11*0,
     *244,       10*010200,    11*0,
     *344,       10*001200,    11*0,
     *444,000000, 9*000300,    11*0/
      data ((icrei(k,m),m=1,2*mxma),k=21,29)/
     * 11,  10*100000,    0,   10*100000,
     * 22,  10*001000,    0,   10*001000,
     * 12,  10*100000,    0,   10*010000,
     * 13,  10*100000,    0,   10*001000,
     * 23,  10*010000,    0,   10*001000,
     * 14,  10*100000,    0,   10*000100,
     * 24,  10*010000,    0,   10*000100,
     * 34,  10*001000,    0,   10*000100,
     * 44,  10*000100,    0,   10*000100/
      data ((icrei(k,m),m=1,2*mxma),k=30,35)/
     * 15,  10*100000,    0,   10*000010,
     * 25,  10*010000,    0,   10*000010,
     * 35,  10*001000,    0,   10*000010,
     * 45,  10*000100,    0,   10*000010,
     * 55,  10*000010,    0,   10*000010,
     *  3,  10*222000,    0,   10*000010/

      data ((remai(k,m),m=1,mxma),k=1,10)/
     *111.,0.000,1.425,1.660,1.825,2.000,0.000,0.000,0.000,0.000,0.000,
     *222.,0.000,1.425,1.660,1.825,2.000,0.000,0.000,0.000,0.000,0.000,
     *112.,1.080,1.315,1.485,1.575,1.645,1.685,1.705,1.825,2.000,0.000,
     *122.,1.080,1.315,1.485,1.575,1.645,1.685,1.705,1.825,2.000,0.000,
     *113.,1.300,1.500,1.700,1.850,2.000,0.000,0.000,0.000,0.000,0.000,
     *223.,1.300,1.500,1.700,1.850,2.000,0.000,0.000,0.000,0.000,0.000,
     *123.,1.117,1.300,1.395,1.465,1.540,1.655,1.710,1.800,1.885,2.000,
c     *123.,1.154,1.288,1.395,1.463,1.560,1.630,1.710,1.800,1.885,2.000,
     *133.,1.423,2.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *233.,1.428,2.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
c     *133.,1.423,1.638,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
c     *233.,1.427,1.634,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *333.,0.000,2.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000/
      data ((remai(k,m),m=1,mxma),k=11,20)/
     *114.,2.530,2.730,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *124.,2.345,2.530,2.730,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *224.,2.530,2.730,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *134.,2.450,2.600,2.800,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *234.,2.450,2.600,2.800,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *334.,2.700,2.900,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *144.,3.650,3.850,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *244.,3.650,3.850,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *344.,3.800,4.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *444.,0.000,5.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000/
      data ((remai(k,m),m=1,mxma),k=21,29)/
     * 11.,0.450,0.950,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 22.,0.750,0.965,1.500,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 12.,0.450,0.950,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 13.,0.500,1.075,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 23.,0.500,1.075,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 14.,1.900,2.250,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 24.,1.900,2.250,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 34.,2.050,2.500,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 44.,3.037,3.158,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000/
      data ((remai(k,m),m=1,mxma),k=30,35)/
     * 15.,5.300,5.400,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 25.,5.300,5.400,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 35.,5.396,5.500,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 45.,6.450,7.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     * 55.,9.660,9.999,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,
     *  3.,2.230,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000/

      data ((rewii(k,m),m=1,mxma),k=1,5)/
     *111.,0.000e+00,0.115e+00,0.140e+00,0.250e+00,0.250e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *222.,0.000e+00,0.115e+00,0.140e+00,0.250e+00,0.250e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *112.,0.000e+00,0.115e+00,0.200e+00,0.140e+00,0.140e+00,
     *     0.145e+00,0.250e+00,0.140e+00,0.250e+00,0.000e+00,
     *122.,7.451e-28,0.115e+00,0.200e+00,0.140e+00,0.140e+00,
     *     0.145e+00,0.250e+00,0.140e+00,0.250e+00,0.000e+00,
     *113.,0.824e-14,0.036e+00,0.080e+00,0.100e+00,0.170e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=6,10)/
     *223.,0.445e-14,0.039e+00,0.080e+00,0.100e+00,0.170e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *123.,0.250e-14,0.890e-05,0.036e+00,0.040e+00,0.016e+00,
     *     0.090e+00,0.080e+00,0.100e+00,0.145e+00,0.170e+00,
     *133.,0.227e-14,0.009e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *233.,0.400e-14,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *333.,0.000e+00,0.800e-14,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=11,15)/
     *114.,0.400e-11,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *124.,0.400e-11,0.400e-11,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *224.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *134.,0.150e-11,0.400e-11,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *234.,0.150e-11,0.400e-11,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=16,20)/
     *334.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *144.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *244.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *344.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *444.,0.400e-11,0.010e+00,0.010e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=21,25)/
     * 11.,7.849e-09,0.153e+00,0.057e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 22.,0.130e-05,0.210e-03,0.034e+00,0.004e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 12.,2.524e-17,0.153e+00,0.057e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 13.,5.307e-17,0.051e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 23.,0.197e-02,0.051e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=26,29)/
     * 14.,0.154e-11,0.002e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 24.,0.615e-12,0.002e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 34.,0.133e-11,0.002e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 44.,0.010e+00,0.068e-03,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/
      data ((rewii(k,m),m=1,mxma),k=30,35)/
     * 15.,0.402e-12,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 25.,0.430e-12,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 35.,0.448e-12,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 45.,0.143e-13,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     * 55.,0.525e-04,0.010e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *  3.,0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00,
     *     0.000e+00,0.000e+00,0.000e+00,0.000e+00,0.000e+00/

      do 3 i=1,mxindx
3     indx(i)=0
      do 4 k=1,mxre
      do 4 m=1,mxma
4     rema(k,m)=0

      do 2 j=1,mxma
      do 2 i=1,mxmx
2     idmx(j,i)=idmxi(j,i)

      ntec=n
      if(ntec.gt.mxre)call utstop('idresi: dimension mxre too small&',
     +sizeof('idresi: dimension mxre too small&'))
      do 1 k=1,n
      ix=nint(remai(k,1))
      ix2=nint(rewii(k,1))
      ix3=icrei(k,1)
      if(ix.ne.ix2)call utstop('idresi: ix /= ix2&',
     +sizeof('idresi: ix /= ix2&'))
      if(ix.ne.ix3)call utstop('idresi: ix /= ix3&',
     +sizeof('idresi: ix /= ix3&'))
      if(ix.lt.1.or.ix.gt.mxindx)
     *call utstop('idresi: ix out of range.&',
     +sizeof('idresi: ix out of range.&'))
      indx(ix)=k
      rema(k,1)=0.
      rewi(k,1)=0.
      icre1(k,1)=0
      icre2(k,1)=0
      do 1 m=2,mxma
      rema(k,m)=remai(k,m)
      rewi(k,m)=rewii(k,m)
      icre1(k,m)=icrei(k,m)
1     icre2(k,m)=icrei(k,mxma+m)

      indx(33) =indx(22)
      indx(213)=indx(123)
      indx(214)=indx(124)
      indx(314)=indx(134)
      indx(324)=indx(234)

      return
      end

cc-----------------------------------------------------------------------
c      integer function idsgl(ic,gen,cmp)
cc     returns 1 for singlets (qqq or qqbar) 0 else.
cc-----------------------------------------------------------------------
c      parameter (nflav=6)
c      integer ic(2),jcx(nflav,2),icx(2)
c      character gen*6,cmp*6
c      idsgl=0
c      if(cmp.eq.'cmp-ys')then
c      call idcomi(ic,icx)
c      else
c      icx(1)=ic(1)
c      icx(2)=ic(2)
c      endif
c      call iddeco(icx,jcx)
c      nq=0
c      na=0
c      do 1 i=1,nflav
c      nq=nq+jcx(i,1)
c1     na=na+jcx(i,2)
c      if(nq.eq.0.and.na.eq.0)return
c      if(gen.eq.'gen-no')then
c      if(nq.eq.3.and.na.eq.0.or.nq.eq.1.and.na.eq.1
c     *.or.nq.eq.0.and.na.eq.3)idsgl=1
c      elseif(gen.eq.'gen-ys')then
c      if(mod(nq-na,3).eq.0)idsgl=1
c      endif
c      return
c      end
c
c-----------------------------------------------------------------------
      subroutine idtau(id,p4,p5,taugm)
c     returns lifetime(c*tau(fm))*gamma for id with energy p4, mass p5
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxindx=1000,mxre=100,mxma=11,mxmx=6)
      common/crema/indx(mxindx),rema(mxre,mxma),rewi(mxre,mxma)
     *,idmx(mxma,mxmx),icre1(mxre,mxma),icre2(mxre,mxma)
           if(iabs(id).eq.14)then
      wi=.197/658.654e15
           elseif(iabs(id).eq.16)then
      wi=.197/87.11e9
           elseif(id.eq.-20)then
      wi=.197/15.34e15
           elseif(id.eq.20)then
      wi=.197/2.6842e13
           elseif((iabs(id).lt.100.and.id.ne.20)
     *         .or.id.gt.1e9)then
      wi=0
           elseif(iabs(id).lt.1e8)then
      ix=iabs(id)/10
      if(ix.lt.1.or.ix.gt.mxindx)then
        write(ifch,*)'id:',id
        call utstop('idtau: ix out of range.&',
     +sizeof('idtau: ix out of range.&'))
      endif
      ii=indx(ix)
      jj=mod(iabs(id),10)+2

      m1=1
      if(iabs(id).ge.1000)m1=3
      m2=2
      if(iabs(id).ge.1000)m2=mxmx
      do 75 imx=m1,m2
      do 75 ima=2,mxma
      if(iabs(id).eq.idmx(ima,imx))then
        jj=ima
        goto 75
      endif
75    continue
      if(ii.lt.1.or.ii.gt.mxre.or.jj.lt.1.or.jj.gt.mxma)then
      write(ifch,*)'id,ii,jj:',id,'   ',ii,jj
      call utstop('idtau: ii or jj out of range&',
     +sizeof('idtau: ii or jj out of range&'))
      endif
      wi=rewi(ii,jj)
           else
      tauz=taunll
c-c   tauz=amin1(9./p5**2,tauz)
c-c   tauz=amax1(.2,tauz)
      wi=.197/tauz
           endif
      if(wi.eq.0.)then
      tau=ainfin
      else
      tau=.197/wi
      endif
      if(p5.ne.0.)then
      gm=p4/p5
      else
      gm=ainfin
      endif
      if(tau.ge.ainfin.or.gm.ge.ainfin)then
      taugm=ainfin
      else
      taugm=tau*gm
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine idtr4(id,ic)
c     transforms generalized paige_id -> werner_id  (for < 4 flv)
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (mxindx=1000,mxre=100,mxma=11,mxmx=6)
      common/crema/indx(mxindx),rema(mxre,mxma),rewi(mxre,mxma)
     *     ,idmx(mxma,mxmx),icre1(mxre,mxma),icre2(mxre,mxma)
      integer ic(2)

      ic(1)=000000
      ic(2)=000000
      if(mod(abs(id),100).eq.99)return !not a particle
      if(iabs(id).lt.20)then
        if(id.eq.1)then
          ic(1)=100000
          ic(2)=000000
        elseif(id.eq.-1)then
          ic(1)=000000
          ic(2)=100000
        elseif(id.eq.2)then
          ic(1)=010000
          ic(2)=000000
        elseif(id.eq.-2)then
          ic(1)=000000
          ic(2)=010000
        elseif(id.eq.3)then
          ic(1)=001000
          ic(2)=000000
        elseif(id.eq.-3)then
          ic(1)=000000
          ic(2)=001000
        elseif(id.eq.4)then
          ic(1)=000100
          ic(2)=000000
        elseif(id.eq.-4)then
          ic(1)=000000
          ic(2)=000100
        elseif(id.eq.5)then
          ic(1)=000010
          ic(2)=000000
        elseif(id.eq.-5)then
          ic(1)=000000
          ic(2)=000010
        elseif(id.eq.17)then
          ic(1)=330000
          ic(2)=000000
        elseif(id.eq.-17)then
          ic(1)=000000
          ic(2)=330000
        elseif(id.eq.18)then
          ic(1)=450000
          ic(2)=000000
        elseif(id.eq.-18)then
          ic(1)=000000
          ic(2)=450000
        elseif(id.eq.19)then
          ic(1)=660000
          ic(2)=000000
        elseif(id.eq.-19)then
          ic(1)=000000
          ic(2)=660000
        endif
        return
      endif
      if(id.eq.30)then
         ic(1)=222000
         ic(2)=000000
         return
      endif
      if(iabs(id).lt.1e8)then
        ix=iabs(id)/10
        if(ix.lt.1.or.ix.gt.mxindx)goto9999
        ii=indx(ix)
        if(ii.eq.0)goto9998
        jj=mod(iabs(id),10)+2
        do 27 imx=1,mxmx
          do 27 ima=2,mxma
            if(iabs(id).eq.idmx(ima,imx))jj=ima
 27     continue
        if(id.gt.0)then
          ic(1)=icre1(ii,jj)
          ic(2)=icre2(ii,jj)
        else
          ic(2)=icre1(ii,jj)
          ic(1)=icre2(ii,jj)
        endif
        if(ic(1).eq.100000.and.ic(2).eq.100000.and.rangen().lt.0.5)
     $       then
          ic(1)=010000
          ic(2)=010000
        endif
      elseif(mod(id/10**8,10).eq.8)then
        ic(1)=mod(id,10**8)/10000*100
        ic(2)=mod(id,10**4)*100
      elseif(id/10**9.eq.1.and.mod(id,10).eq.0)then   !nuclei
        nstr=mod(id,10**8)/10000000
        npro=mod(id,10**7)/10000
        nneu=mod(id,10**4)/10
        ic(1)=(2*npro+nneu)*10**5+(2*nneu+npro)*10**4+nstr*10**3
        ic(2)=0
      else
        write(ifch,*)'***** id: ',id
        call utstop('idtr4: unrecognized id&',
     +sizeof('idtr4: unrecognized id&'))
      endif
      return

 9998 continue
      write(ifch,*)'id: ',id
      call utstop('idtr4: indx=0.&',
     +sizeof('idtr4: indx=0.&'))
      
 9999 continue
      write(ifch,*)'id: ',id
      call utstop('idtr4: ix out of range.&',
     +sizeof('idtr4: ix out of range.&'))
      end

c-----------------------------------------------------------------------
      integer function idtra(ic,ier,ires,imix)
c-----------------------------------------------------------------------
c     tranforms from werner-id to paige-id
c         ier .... error message (1) or not (0) in case of problem
c         ires ... dummy variable, not used  any more !!!!
c         imix ... 1 not supported any more
c                  2 010000 010000 -> 110, 001000 000100 -> 110
c                  3 010000 010000 -> 110, 001000 000100 -> 220
c-----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nidt=54)
      integer idt(3,nidt),ic(2)!,icm(2)
      data idt/
     * 100000,100000, 110   ,100000,010000, 120   ,010000,010000, 220
     *,100000,001000, 130   ,010000,001000, 230   ,001000,001000, 330
     *,100000,000100, 140   ,010000,000100, 240   ,001000,000100, 340
     *,000100,000100, 440
     *,100000,000010, 150   ,010000,000010, 250   ,001000,000010, 350
     *,000100,000010, 450   ,000010,000010, 550   ,100000,000000,   1
     *,010000,000000,   2   ,001000,000000,   3   ,000100,000000,   4
     *,000010,000000,   5
     *,200000,000000,1100   ,110000,000000,1200   ,020000,000000,2200
     *,101000,000000,1300   ,011000,000000,2300   ,002000,000000,3300
     *,100100,000000,1400   ,010100,000000,2400   ,001100,000000,3400
     *,000200,000000,4400
     *,330000,000000,  17   ,450000,000000,  18   ,660000,000000,  19
     *,300000,000000,1111   ,210000,000000,1120   ,120000,000000,1220
     *,030000,000000,2221   ,201000,000000,1130   ,111000,000000,1230
     *,000001,000000,   6
     *,021000,000000,2230   ,102000,000000,1330   ,012000,000000,2330
     *,003000,000000,3331   ,200100,000000,1140   ,110100,000000,1240
     *,020100,000000,2240   ,101100,000000,1340   ,011100,000000,2340
     *,002100,000000,3340
     *,100200,000000,1440   ,010200,000000,2440   ,001200,000000,3440
     *,000300,000000,4441/

      idtra=0
      if(ic(1).eq.0.and.ic(2).eq.0)return
      i=1
      do while(i.le.nidt.and.idtra.eq.0)
        if(ic(2).eq.idt(1,i).and.ic(1).eq.idt(2,i))idtra=-idt(3,i)
        if(ic(1).eq.idt(1,i).and.ic(2).eq.idt(2,i))idtra=idt(3,i)
        i=i+1
      enddo
      isi=1
      if(idtra.ne.0)isi=idtra/iabs(idtra)

      jspin=0

      if(imix.eq.1)stop'imix=1 no longer supported'
      if(imix.eq.2)then
      if(idtra.eq.220)idtra=110
      if(idtra.eq.330)idtra=110
      elseif(imix.eq.3)then
      if(idtra.eq.220)idtra=110
      if(idtra.eq.330)idtra=220
      endif

      if(idtra.ne.0)idtra=idtra+jspin*isi

      if(idtra.ne.0)return
      if(ier.ne.1)return
      write(ifch,*)'idtra: ic = ',ic,ires
      call utstop('idtra: unknown code&',
     +sizeof('idtra: unknown code&'))

      entry idtrai(num,id,ier)
      idtrai=0
      if(iabs(id).eq.20)then
        j=5
      elseif(iabs(id).eq.110.or.iabs(id).eq.220)then
        j=1+2*int(2.*rangen())
      else
        j=0
        do i=1,nidt
          if(iabs(id).eq.idt(3,i))then
            j=i
            goto 2
          endif
        enddo
 2      continue
      endif
      if(j.ne.0)then
        if(id.lt.0)then
          idtrai=idt(3-num,j)
        else
          idtrai=idt(num,j)
        endif
        return
      endif
      if(ier.ne.1)return
      write(ifch,*)'idtrai: id = ',id
      call utstop('idtrai: unknown code&',
     +sizeof('idtrai: unknown code&'))
      end

c-----------------------------------------------------------------------
      subroutine idtrb(ib1,ib2,ib3,ib4,jc)
c     id transformation ib -> jc
c-----------------------------------------------------------------------
      parameter (nflav=6)
      integer jc(nflav,2)
      jc(1,1)=ib1/10**4
      jc(2,1)=ib2/10**4
      jc(3,1)=ib3/10**4
      jc(4,1)=ib4/10**4
      jc(5,1)=0
      jc(6,1)=0
      jc(1,2)=mod(ib1,10**4)
      jc(2,2)=mod(ib2,10**4)
      jc(3,2)=mod(ib3,10**4)
      jc(4,2)=mod(ib4,10**4)
      jc(5,2)=0
      jc(6,2)=0
      return
      end

c-----------------------------------------------------------------------
      subroutine idtrbi(jc,ib1,ib2,ib3,ib4)
c     id transformation jc -> ib
c-----------------------------------------------------------------------
      include 'epos.inc'
      integer jc(nflav,2)
      ib1=jc(1,1)*10**4+jc(1,2)
      ib2=jc(2,1)*10**4+jc(2,2)
      ib3=jc(3,1)*10**4+jc(3,2)
      ib4=jc(4,1)*10**4+jc(4,2)
      ib5=jc(5,1)*10**4+jc(5,2)
      ib6=jc(6,1)*10**4+jc(6,2)
      if(ib5.ne.0.or.ib6.ne.0)then
      write(ifch,*)'***** error in idtrbi: bottom or top quarks'
      write(ifch,*)'jc:'
      write(ifch,*)jc
      call utstop('idtrbi: bottom or top quarks&',
     +sizeof('idtrbi: bottom or top quarks&'))
      endif
      return
      end

c------------------------------------------------------------------------------
      integer function idtrafo(code1,code2,idi)
c------------------------------------------------------------------------------
c.....tranforms id of code1 (=idi) into id of code2 (=idtrafo)
c.....supported codes:
c.....'nxs' = epos
c.....'pdg' = PDG 1996
c.....'qgs' = QGSJet
c.....'ghe' = Gheisha
c.....'sib' = Sibyll
c.....'cor' = Corsika (GEANT)
c.....'flk' = Fluka

C --- ighenex(I)=EPOS CODE CORRESPONDING TO GHEISHA CODE I ---

      common /ighnx/ ighenex(35)
      data ighenex/
     $               10,   11,   -12,    12,   -14,    14,   120,   110,
     $             -120,  130,    20,   -20,  -130,  1120, -1120,  1220,
     $            -1220, 2130, -2130,  1130,  1230,  2230, -1130, -1230,
     $            -2230, 1330,  2330, -1330, -2330,    17,    18,    19,
     $            3331, -3331,  30/

C --- DATA STMTS. FOR GEANT/GHEISHA PARTICLE CODE CONVERSIONS ---
C --- KIPART(I)=GHEISHA CODE CORRESPONDING TO GEANT   CODE I ---
C --- IKPART(I)=GEANT   CODE CORRESPONDING TO GHEISHA CODE I ---
      DIMENSION        KIPART(48)!,IKPART(35)
      DATA KIPART/
     $               1,   3,   4,   2,   5,   6,   8,   7,
     $               9,  12,  10,  13,  16,  14,  15,  11,
     $              35,  18,  20,  21,  22,  26,  27,  33,
     $              17,  19,  23,  24,  25,  28,  29,  34,
     $              35,  35,  35,  35,  35,  35,  35,  35,
     $              35,  35,  35,  35,  30,  31,  32,  35/

c      DATA IKPART/
c     $               1,   4,   2,   3,   5,   6,   8,   7,
c     $               9,  11,  16,  10,  12,  14,  15,  13,
c     $              25,  18,  26,  19,  20,  21,  27,  28,
c     $              29,  22,  23,  30,  31,  45,  46,  47,
c     $              24,  32,  48/
      INTEGER          ICFTABL(200),IFCTABL(-6:100)
C  ICTABL CONVERTS CORSIKA PARTICLES INTO FLUKA PARTICLES
C  FIRST TABLE ONLY IF CHARMED PARTICLES CAN BE TREATED
      DATA ICFTABL/
     *   7,   4,   3,   0,  10,  11,  23,  13,  14,  12,  ! 10
     *  15,  16,   8,   1,   2,  19,   0,  17,  21,  22,  ! 20
     *  20,  34,  36,  38,   9,  18,  31,  32,  33,  34,  ! 30
     *  37,  39,  24,  25, 6*0,
     *  0,    0,   0,   0,  -3,  -4,  -6,  -5,   0,   0,  ! 50
     *  10*0,
     *   0,   0,   0,   0,   0,   5,   6,  27,  28,   0,  ! 70
     *  10*0,
     *  10*0,
     *  10*0,                                             !100
     *  10*0,
     *   0,   0,   0,   0,   0,  47,  45,  46,  48,  49,  !120
     *  50,   0,   0,   0,   0,   0,   0,   0,   0,   0,  !130
     *  41,  42,  43,  44,   0,   0,  51,  52,  53,   0,  !140
     *   0,   0,  54,  55,  56,   0,   0,   0,  57,  58,  !150
     *  59,   0,   0,   0,  60,  61,  62,   0,   0,   0,  !160
     *  40*0/
C  IFCTABL CONVERTS FLUKA PARTICLES INTO CORSIKA PARTICLES
      DATA IFCTABL/
     *                402, 302, 301, 201,   0,   0,   0,
     *  14,  15,   3,   2,  66,  67,   1,  13,  25,   5,
     *   6,  10,   8,   9,  11,  12,  18,  26,  16,  21,
     *  19,  20,   7,  33,  34,   0,  68,  69,   0,   0,
     *  27,  28,  29,  22,  30,  23,  31,  24,  32,   0,
     * 131, 132, 133, 134, 117, 118, 116, 119, 120, 121,
     * 137, 138, 139, 143, 144, 145, 149, 150, 151, 155,
     * 156, 157,   0,   0,   36*0/
c-------------------------------------------------------------------------------

      character*3 code1,code2
      parameter (ncode=5,nidt=338)
      integer idt(ncode,nidt)
      double precision drangen,dummy

c            nxs|pdg|qgs|cor|sib
      data ((idt(i,j),i=1,ncode),j= 1,68)/
     *          1,2,99,99,99             !u quark
     *     ,    2,1,99,99,99             !d
     *     ,    3,3,99,99,99             !s
     *     ,    4,4,99,99,99             !c
     *     ,    5,5,99,99,99             !b
     *     ,    6,6,99,99,99             !t
     *     ,   10,22,99,1,1              !gamma
     *     ,   9 ,21,99,99,99            !gluon
     *     ,   12,11,11,3,3              !e-
     *     ,  -12,-11,-11,2,2            !e+
     *     ,   11,12,99,66,15            !nu_e-
     *     ,  -11,-12,99,67,16           !nu_e+
     *     ,   14,13,99,6,5              !mu-
     *     ,  -14,-13,99,5,4             !mu+
     *     ,   13,14,99,68,17            !nu_mu-
     *     ,  -13,-14,99,69,18           !nu_mu+
     *     ,   16,15,99,132,19           !tau-
     *     ,  -16,-15,99,131,-19         !tau+
     *     ,   15,16,99,133,20           !nu_tau-
     *     ,  -15,-16,99,134,-20         !nu_tau+
     *     ,  110,111,0,7,6              !pi0
     *     ,  120,211,1,8,7              !pi+
     *     , -120,-211,-1,9,8            !pi-
     *     ,  220,221,10,17,23           !eta
     *     ,  130,321,4,11,9             !k+
     *     , -130,-321,-4,12,10          !k-
     *     ,  230,311,5,33,21            !k0
     *     , -230,-311,-5,34,22          !k0b
     *     ,   20,310,5,16,12            !kshort
     *     ,  -20,130,-5,10,11           !klong
     *     ,  330,331,99,99,24           !etaprime
     *     ,  111,113,19,51,27           !rho0
     *     ,  121,213,99,52,25           !rho+
     *     , -121,-213,99,53,26          !rho-
     *     ,  221,223,99,50,32           !omega
     *     ,  131,323,99,63,28           !k*+
     *     , -131,-323,99,64,29          !k*-
     *     ,  231,313,99,62,30           !k*0
     *     , -231,-313,99,65,31          !k*0b
     *     ,  331,333,99,99,33           !phi
     *     , -140,421,8,116,99           !D0(1.864)
     *     ,  140,-421,8,119,99          !D0b(1.864)
     *     , -240,411,7,117,99           !D(1.869)+
     *     ,  240,-411,7,118,99          !Db(1.869)-
     *     , 1120,2212,2,14,13           !proton
     *     , 1220,2112,3,13,14           !neutron
     *     , 2130,3122,6,18,39           !lambda
     *     , 1130,3222,99,19,34          !sigma+
     *     , 1230,3212,99,20,35          !sigma0
     *     , 2230,3112,99,21,36          !sigma-
     *     , 1330,3322,99,22,37          !xi0
     *     , 2330,3312,99,23,38          !xi-
     *     , 1111,2224,99,54,40          !delta++
     *     , 1121,2214,99,55,41          !delta+
     *     , 1221,2114,99,56,42          !delta0
     *     , 2221,1114,99,57,43          !delta-
     *     , 1131,3224,99,99,44          !sigma*+
     *     , 1231,3214,99,99,45          !sigma*0
     *     , 2231,3114,99,99,46          !sigma*-
     *     , 1331, 3324,99,99,47         !xi*0
     *     , 2331, 3314,99,99,48         !xi*-
     *     , 3331, 3334,99,24,49         !omega-
     *     , 2140, 4122,9,137,99         !LambdaC(2.285)+
     *     ,17,1000010020,99,201,1002            !  Deuteron
     *     ,18,1000010030,99,301,1003            !  Triton
     *     ,19,1000020040,99,402,1004            !  Alpha
     *     ,0,0,99,0,0                  !  Air
     *     ,99,99,99,99,99 /             !  unknown
      data ((idt(i,j),i=1,ncode),j= 69,89)/
     $      -340,431,99,120,99           !  Ds+
     $     ,340,-431,99,121,99           !  Ds-
     $     ,-241,413,99,124,99           !  D*+
     $     ,241,-413,99,125,99           !  D*-
     $     ,-141,423,99,123,99           !  D*0
     $     ,141,-423,99,126,99           !  D*0b
     $     ,-341,433,99,127,99           !  Ds*+
     $     ,341,-433,99,128,99           !  Ds*-
     $     ,440,441,99,122,99            !  etac
     $     ,441,443,99,130,99            !  J/psi
     $     ,2240,4112,99,142,99          !  sigmac0
     $     ,1240,4212,99,141,99          !  sigmac+
     $     ,1140,4222,99,140,99          !  sigmac++
     $     ,2241,4114,99,163,99          !  sigma*c0
     $     ,1241,4214,99,162,99          !  sigma*c+
     $     ,1141,4224,99,161,99          !  sigma*c++
     $     ,3240,4132,99,139,99          !  Xic0
     $     ,2340,4312,99,144,99          !  Xi'c0
     $     ,3140,4232,99,138,99          !  Xic+
     $     ,1340,4322,99,143,99          !  Xi'c+
     $     ,3340,4332,99,145,99 /        !  omegac0
      data ((idt(i,j),i=1,ncode),j= 90,nidt)/
     $       1112,32224,99,99,99         !  Delta(1600)++
     $     , 1112, 2222,99,99,99         !  Delta(1620)++
     $     , 1113,12224,99,99,99         !  Delta(1700)++
     $     , 1114,12222,99,99,99         !  Delta(1900)++
     $     , 1114, 2226,99,99,99         !  Delta(1905)++
     $     , 1114,22222,99,99,99         !  Delta(1910)++
     $     , 1114,22224,99,99,99         !  Delta(1920)++
     $     , 1114,12226,99,99,99         !  Delta(1930)++
     $     , 1114, 2228,99,99,99         !  Delta(1950)++
     $     , 2222,31114,99,99,99         !  Delta(1600)-
     $     , 2222, 1112,99,99,99         !  Delta(1620)-
     $     , 2223,11114,99,99,99         !  Delta(1700)-
     $     , 2224,11112,99,99,99         !  Delta(1900)-
     $     , 2224, 1116,99,99,99         !  Delta(1905)-
     $     , 2224,21112,99,99,99         !  Delta(1910)-
     $     ,2224,21114,99,99,99          !  Delta(1920)-
     $     ,2224,11116,99,99,99          !  Delta(1930)-
     $     ,2224, 1118,99,99,99          !  Delta(1950)-
     $     ,1122,12212,99,99,99          !  N(1440)+
     $     ,1123, 2124,99,99,99          !  N(1520)+
     $     ,1123,22212,99,99,99          !  N(1535)+
     $     ,1124,32214,99,99,99          !  Delta(1600)+
     $     ,1124, 2122,99,99,99          !  Delta(1620)+
     $     ,1125,32212,99,99,99          !  N(1650)+
     $     ,1125, 2216,99,99,99          !  N(1675)+
     $     ,1125,12216,99,99,99          !  N(1680)+
     $     ,1126,12214,99,99,99          !  Delta(1700)+
     $     ,1127,22124,99,99,99          !  N(1700)+
     $     ,1127,42212,99,99,99          !  N(1710)+
     $     ,1127,32124,99,99,99          !  N(1720)+
     $     ,1128,12122,99,99,99          !  Delta(1900)+
     $     ,1128, 2126,99,99,99          !  Delta(1905)+
     $     ,1128,22122,99,99,99          !  Delta(1910)+
     $     ,1128,22214,99,99,99          !  Delta(1920)+
     $     ,1128,12126,99,99,99          !  Delta(1930)+
     $     ,1128, 2218,99,99,99          !  Delta(1950)+
     $     ,1222,12112,99,99,99          !  N(1440)0
     $     ,1223, 1214,99,99,99          !  N(1520)0
     $     ,1223,22112,99,99,99          !  N(1535)0
     $     ,1224,32114,99,99,99          !  Delta(1600)0
     $     ,1224, 1212,99,99,99          !  Delta(1620)0
     $     ,1225,32112,99,99,99          !  N(1650)0
     $     ,1225, 2116,99,99,99          !  N(1675)0
     $     ,1225,12116,99,99,99          !  N(1680)0
     $     ,1226,12114,99,99,99          !  Delta(1700)0
     $     ,1227,21214,99,99,99          !  N(1700)0
     $     ,1227,42112,99,99,99          !  N(1710)0
     $     ,1227,31214,99,99,99          !  N(1720)0
     $     ,1228,11212,99,99,99          !  Delta(1900)0
     $     ,1228, 1216,99,99,99          !  Delta(1905)0
     $     ,1228,21212,99,99,99          !  Delta(1910)0
     $     ,1228,22114,99,99,99          !  Delta(1920)0
     $     ,1228,11216,99,99,99          !  Delta(1930)0
     $     ,1228, 2118,99,99,99          !  Delta(1950)0
     $     ,1233,13122,99,99,99          !  Lambda(1405)0
     $     ,1234, 3124,99,99,99          !  Lambda(1520)0
     $     ,1235,23122,99,99,99          !  Lambda(1600)0
     $     ,1235,33122,99,99,99          !  Lambda(1670)0
     $     ,1235,13124,99,99,99          !  Lambda(1690)0
     $     ,1236,13212,99,99,99          !  Sigma(1660)0
     $     ,1236,13214,99,99,99          !  Sigma(1670)0
     $     ,1237,23212,99,99,99          !  Sigma(1750)0
     $     ,1237, 3216,99,99,99          !  Sigma(1775)0
     $     ,1238,43122,99,99,99          !  Lambda(1800)0
     $     ,1238,53122,99,99,99          !  Lambda(1810)0
     $     ,1238, 3126,99,99,99          !  Lambda(1820)0
     $     ,1238,13126,99,99,99          !  Lambda(1830)0
     $     ,1238,23124,99,99,99          !  Lambda(1890)0
     $     ,1239,13216,99,99,99          !  Sigma(1915)0
     $     ,1239,23214,99,99,99          !  Sigma(1940)0
     $     ,1132,13222,99,99,99          !  Sigma(1660)+
     $     ,1132,13224,99,99,99          !  Sigma(1670)+
     $     ,1133,23222,99,99,99          !  Sigma(1750)+
     $     ,1133,3226,99,99,99           !  Sigma(1775)+
     $     ,1134,13226,99,99,99          !  Sigma(1915)+
     $     ,1134,23224,99,99,99          !  Sigma(1940)+
     $     ,2232,13112,99,99,99          !  Sigma(1660)-
     $     ,2232,13114,99,99,99          !  Sigma(1670)-
     $     ,2233,23112,99,99,99          !  Sigma(1750)-
     $     ,2233,3116,99,99,99           !  Sigma(1775)-
     $     ,2234,13116,99,99,99          !  Sigma(1915)-
     $     ,2234,23114,99,99,99          !  Sigma(1940)-
     $     ,5,7,99,99,99                 !  quark b'
     $     ,6,8,99,99,99                 !  quark t'
     $     ,16,17,99,99,99               !  lepton tau'
     $     ,15,18,99,99,99               !  lepton nu' tau
     $     ,90,23,99,99,99               !  Z0
     $     ,80,24,99,99,99               !  W+
     $     ,81,25,99,99,99               !  h0
     $     ,85,32,99,99,99               !  Z'0
     $     ,86,33,99,99,99               !  Z''0
     $     ,87,34,99,99,99               !  W'+
     $     ,82,35,99,99,99               !  H0
     $     ,83,36,99,99,99               !  A0
     $     ,84,37,99,99,99               !  H+
     $     ,1200,2101,99,99,99           !  diquark ud_0
     $     ,2300,3101,99,99,99           !  diquark sd_0
     $     ,1300,3201,99,99,99           !  diquark su_0
     $     ,2400,4101,99,99,99           !  diquark cd_0
     $     ,1400,4201,99,99,99           !  diquark cu_0
     $     ,3400,4301,99,99,99           !  diquark cs_0
     $     ,2500,5101,99,99,99           !  diquark bd_0
     $     ,1500,5201,99,99,99           !  diquark bu_0
     $     ,3500,5301,99,99,99           !  diquark bs_0
     $     ,4500,5401,99,99,99           !  diquark bc_0
     $     ,2200,1103,99,99,99           !  diquark dd_1
     $     ,1200,2103,99,99,99           !  diquark ud_1
     $     ,1100,2203,99,99,99           !  diquark uu_1
     $     ,2300,3103,99,99,99           !  diquark sd_1
     $     ,1300,3203,99,99,99           !  diquark su_1
     $     ,3300,3303,99,99,99           !  diquark ss_1
     $     ,2400,4103,99,99,99           !  diquark cd_1
     $     ,1400,4203,99,99,99           !  diquark cu_1
     $     ,3400,4303,99,99,99           !  diquark cs_1
     $     ,4400,4403,99,99,99           !  diquark cc_1
     $     ,2500,5103,99,99,99           !  diquark bd_1
     $     ,1500,5203,99,99,99           !  diquark bu_1
     $     ,3500,5303,99,99,99           !  diquark bs_1
     $     ,4500,5403,99,99,99           !  diquark bc_1
     $     ,5500,5503,99,99,99           !  diquark bb_1
     $     ,800000091,91,99,99,99        !  parton system in cluster fragmentation  (pythia)
     $     ,800000092,92,99,99,99        !  parton system in string fragmentation  (pythia)
     $     ,800000093,93,99,99,99        !  parton system in independent system  (pythia)
     $     ,800000094,94,99,99,99        !  CMshower (pythia)
     $     ,250,511,99,99,99             !  B0
     $     ,150,521,99,99,99             !  B+
     $     ,350,531,99,99,99             !  B0s+
     $     ,450,541,99,99,99             !  Bc+
     $     ,251,513,99,99,99             !  B*0
     $     ,151,523,99,99,99             !  B*+
     $     ,351,533,99,99,99             !  B*0s+
     $     ,451,543,99,99,99             !  B*c+
     $     ,550,551,99,99,99             !  etab
     $     ,551,553,99,99,99             !  Upsilon
     $     ,2341,4314,99,99,99           !  Xi*c0
     $     ,1341,4324,99,99,99           !  Xi*c+
     $     ,3341,4334,99,99,99           !  omega*c0
     $     ,2440,4412,99,99,99           !  dcc
     $     ,2441,4414,99,99,99           !  dcc*
     $     ,1440,4422,99,99,99           !  ucc
     $     ,1441,4424,99,99,99           !  ucc*
     $     ,3440,4432,99,99,99           !  scc
     $     ,3441,4434,99,99,99           !  scc*
     $     ,4441,4444,99,99,99           !  ccc*
     $     ,2250,5112,99,99,99           !  sigmab-
     $     ,2150,5122,99,99,99           !  lambdab0
     $     ,3250,5132,99,99,99           !  sdb
     $     ,4250,5142,99,99,99           !  cdb
     $     ,1250,5212,99,99,99           !  sigmab0
     $     ,1150,5222,99,99,99           !  sigmab+
     $     ,3150,5232,99,99,99           !  sub
     $     ,4150,5242,99,99,99           !  cub
     $     ,2350,5312,99,99,99           !  dsb
     $     ,1350,5322,99,99,99           !  usb
     $     ,3350,5332,99,99,99           !  ssb
     $     ,4350,5342,99,99,99           !  csb
     $     ,2450,5412,99,99,99           !  dcb
     $     ,1450,5422,99,99,99           !  ucb
     $     ,3450,5432,99,99,99           !  scb
     $     ,4450,5442,99,99,99           !  ccb
     $     ,2550,5512,99,99,99           !  dbb
     $     ,1550,5522,99,99,99           !  ubb
     $     ,3550,5532,99,99,99           !  sbb
     $     ,3550,5542,99,99,99           !  scb
     $     ,2251,5114,99,99,99           !  sigma*b-
     $     ,1251,5214,99,99,99           !  sigma*b0
     $     ,1151,5224,99,99,99           !  sigma*b+
     $     ,2351,5314,99,99,99           !  dsb*
     $     ,1351,5324,99,99,99           !  usb*
     $     ,3351,5334,99,99,99           !  ssb*
     $     ,2451,5414,99,99,99           !  dcb*
     $     ,1451,5424,99,99,99           !  ucb*
     $     ,3451,5434,99,99,99           !  scb*
     $     ,4451,5444,99,99,99           !  ccb*
     $     ,2551,5514,99,99,99           !  dbb*
     $     ,1551,5524,99,99,99           !  ubb*
     $     ,3551,5534,99,99,99           !  sbb*
     $     ,4551,5544,99,99,99           !  cbb*
     $     ,5551,5554,99,99,99           !  bbb*
     $     ,123,10213,99,99,99           !  b1
     $     ,122,10211,99,99,99           !  a0+
     $     ,233,10313,99,99,99           !  K0_1
     $     ,232,10311,99,99,99           !  K*0_1
     $     ,133,10323,99,99,99           !  K+_1
     $     ,132,10321,99,99,99           !  K*+_1
     $     ,143,10423,99,99,99           !  D0_1
     $     ,132,10421,99,99,99           !  D*0_1
     $     ,243,10413,99,99,99           !  D+_1
     $     ,242,10411,99,99,99           !  D*+_1
     $     ,343,10433,99,99,99           !  D+s_1
     $     ,342,10431,99,99,99           !  D*0s+_1
     $     ,223,10113,99,99,99           !  b_10
     $     ,222,10111,99,99,99           !  a_00
     $     ,113,10223,99,99,99           !  h_10
     $     ,112,10221,99,99,99           !  f_00
     $     ,333,10333,99,99,99           !  h'_10
     $     ,332,10331,99,99,99           !  f'_00
     $     ,443,10443,99,99,99           !  h_1c0
     $     ,442,10441,99,99,99           !  Xi_0c0
     $     ,444,10443,99,99,99           !  psi'
     $     ,253,10513,99,99,99           !  db_10
     $     ,252,10511,99,99,99           !  db*_00
     $     ,153,10523,99,99,99           !  ub_10
     $     ,152,10521,99,99,99           !  ub*_00
     $     ,353,10533,99,99,99           !  sb_10
     $     ,352,10531,99,99,99           !  sb*_00
     $     ,453,10543,99,99,99           !  cb_10
     $     ,452,10541,99,99,99           !  cb*_00
     $     ,553,10553,99,99,99           !  Upsilon'
     $     ,552,10551,99,99,99           !  Upsilon'*
     $     ,124,20213,99,99,99           !  a_1+
     $     ,125,215,99,99,99             !  a_2+
     $     ,234,20313,99,99,99           !  K*0_1
     $     ,235,315,99,99,99             !  K*0_2
     $     ,134,20323,99,99,99           !  K*+_1
     $     ,135,325,99,99,99             !  K*+_2
     $     ,144,20423,99,99,99           !  D*_10
     $     ,135,425,99,99,99             !  D*_20
     $     ,244,20413,99,99,99           !  D*_1+
     $     ,245,415,99,99,99             !  D*_2+
     $     ,344,20433,99,99,99           !  D*_1s+
     $     ,345,435,99,99,99             !  D*_2s+
     $     ,224,20113,99,99,99           !  a_10
     $     ,225,115,99,99,99             !  a_20
     $     ,114,20223,99,99,99           !  f_10
     $     ,115,225,99,99,99             !  f_20
     $     ,334,20333,99,99,99           !  f'_10
     $     ,335,335,99,99,99             !  f'_20
     $     ,444,20443,99,99,99           !  Xi_1c0
     $     ,445,445,99,99,99             !  Xi_2c0
     $     ,254,20513,99,99,99           !  db*_10
     $     ,255,515,99,99,99             !  db*_20
     $     ,154,20523,99,99,99           !  ub*_10
     $     ,155,525,99,99,99             !  ub*_20
     $     ,354,20533,99,99,99           !  sb*_10
     $     ,355,535,99,99,99             !  sb*_20
     $     ,454,20543,99,99,99           !  cb*_10
     $     ,455,545,99,99,99             !  cb*_20
     $     ,554,20553,99,99,99           !  bb*_10
     $     ,555,555,99,99,99             !  bb*_20
     $     ,11099,9900110,99,99,99       !  diff pi0 state
     $     ,12099,9900210,99,99,99       !  diff pi+ state
     $     ,22099,9900220,99,99,99       !  diff omega state
     $     ,33099,9900330,99,99,99       !  diff phi state
     $     ,44099,9900440,99,99,99       !  diff J/psi state
     $     ,112099,9902210,99,99,99      !  diff proton state
     $     ,122099,9902110,99,99,99      !  diff neutron state
     $     ,800000110,110,99,99,99       !  Reggeon
     $     ,800000990,990,99,99,99 /     !  Pomeron


c      print *,'idtrafo',' ',code1,' ',code2,idi

      nidtmx=68
      id1=idi
      if(code1.eq.'nxs')then
        i=1
      elseif(code1.eq.'pdg')then
        i=2
      elseif(code1.eq.'qgs')then
        i=3
        if(id1.eq.-10)id1=19
      elseif(code1.eq.'cor')then
        i=4
      elseif(code1.eq.'sib')then
        i=5
      elseif(code1.eq.'ghe')then
        id1=ighenex(id1)
        i=1
      elseif(code1.eq.'flk')then
        id1=IFCTABL(id1)          !convert to corsika code
        i=4
      else
        stop "unknown code in idtrafo"
      endif
      if(code2.eq.'nxs')then
        j=1
        ji=j
        if(i.eq.2.and.id1.gt.1000000000)then   !nucleus from PDG
          idtrafo=id1 
          return
        elseif(i.eq.4.and.id1.gt.402)then               !nucleus from Corsika
          idtrafo=1000000000+mod(id1,100)*10000+(id1/100)*10   
          return
        elseif(i.eq.5.and.id1.gt.1004)then               !nucleus from Sibyll
          id1=(id1-1000)
          idtrafo=1000000000+id1/2*10000+id1*10   
          return
        elseif(id1.eq.130.and.i.eq.2)then
          idtrafo=-20
          return
        endif
        if(i.eq.2) nidtmx=nidt
        if(i.eq.4) nidtmx=89
      elseif(code2.eq.'pdg')then
        j=2
        ji=j
        if(i.eq.1.and.id1.gt.1000000000)then !nucleus from NEXUS
          idtrafo=id1 
          return
        elseif(i.eq.4.and.id1.gt.402)then               !nucleus from Corsika
          idtrafo=1000000000+mod(id1,100)*10000+(id1/100)*10   
          return
        elseif(i.eq.5.and.id1.gt.1004)then               !nucleus from Sibyll
          id1=(id1-1000)
          idtrafo=1000000000+id1/2*10000+id1*10   
          return
        elseif(id1.eq.-20.and.i.eq.1)then
          idtrafo=130
          return
        endif
        if(i.eq.1) nidtmx=nidt
        if(i.eq.4) nidtmx=89
       elseif(code2.eq.'qgs')then
        j=3
        ji=j
      elseif(code2.eq.'cor')then
        j=4
        ji=j
      elseif(code2.eq.'sib')then
        j=5
        ji=j
      elseif(code2.eq.'ghe')then
        j=4
        ji=6
      elseif(code2.eq.'flk')then
        j=4
        ji=7
        if(i.le.2) nidtmx=89
       else
        stop "unknown code in idtrafo"
      endif
      if(i.eq.4)then !corsika  id always >0 so convert antiparticles
        iadtr=id1
        if(iadtr.eq.25)then
          id1=-13
        elseif(iadtr.eq.15)then
          id1=-14
        elseif(iadtr.ge.26.and.iadtr.le.32)then
          id1=-iadtr+8
        elseif(iadtr.ge.58.and.iadtr.le.61)then
          id1=-iadtr+4
        elseif(iadtr.ge.149.and.iadtr.le.157)then
          id1=-iadtr+12
        elseif(iadtr.ge.171.and.iadtr.le.173)then
          id1=-iadtr+10
        endif
      endif
      iad1=abs(id1)
      isi=sign(1,id1)

      if(i.ne.j)then
      do n=1,nidtmx
        if(iad1.eq.abs(idt(i,n)))then
          m=1
          if(n+m.lt.nidt)then
            do while(abs(idt(i,n+m)).eq.iad1)
              m=m+1
            enddo
          endif
          mm=0
          if(m.gt.1)then
            if(m.eq.2.and.idt(i,n)*idt(i,n+1).lt.0)then
              if(id1.eq.idt(i,n+1))mm=1
              isi=1
            else
              mm=int(drangen(dummy)*dble(m))
            endif
          endif
          idtrafo=idt(j,n+mm)*isi
          if(abs(idtrafo).eq.99)call utstop('New particle not allowed ',
     +sizeof('New particle not allowed '))
          if(idtrafo.lt.0.and.j.eq.4)then           !corsika  id always >0
            iadtr=abs(idtrafo)
            if(iadtr.eq.13)then
              idtrafo=25
            elseif(iadtr.eq.14)then
              idtrafo=15
            elseif(iadtr.ge.18.and.iadtr.le.24)then
              idtrafo=iadtr+8
            elseif(iadtr.ge.54.and.iadtr.le.57)then
              idtrafo=iadtr+4
            elseif(iadtr.ge.137.and.iadtr.le.145)then
              idtrafo=iadtr+12
            elseif(iadtr.ge.161.and.iadtr.le.163)then
              idtrafo=iadtr+10
            else
              idtrafo=iadtr
            endif
          elseif(idtrafo.eq.19.and.j.eq.3)then
            idtrafo=-10
          endif
          if(j.ne.ji)goto 100
          return
        endif
      enddo
      else
        idtrafo=id1
        if(j.ne.ji)goto 100
        return
      endif

      print *, 'idtrafo: ',code1,' -> ', code2,id1,' not found.   '
      stop
c      idtrafocx=0
c      return

 100  if(j.eq.4)then            !corsika
        if(idtrafo.eq.201)then
          idtrafo=45
        elseif(idtrafo.eq.301)then
          idtrafo=46
        elseif(idtrafo.eq.402)then
          idtrafo=47
        elseif(idtrafo.eq.302)then
          idtrafo=48
        endif
        if(idtrafo.ne.0)then      !air
          if(ji.eq.6)then
            idtrafo=kipart(idtrafo)
          elseif(ji.eq.7)then
            idtrafo=ICFTABL(idtrafo)
          endif
        endif
        return
      else
        call utstop('Should not happen in idtrafo !&',
     +sizeof('Should not happen in idtrafo !&'))
      endif

      end

