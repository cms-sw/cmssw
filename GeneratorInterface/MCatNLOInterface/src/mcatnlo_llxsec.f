      subroutine freal(s,x,yi,tk,uk,q1q,q2q,jproc,vecout,axlout)
c Returns the real matrix elements times xii**2*(1-yi**2)=4*tk*uk/s**2.
c See the subroutine xmewrap for details concerning the normalization
      implicit none
      real * 8 s,x,yi,tk,uk,q1q,q2q,vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      include 'llpcblks.h'
      real * 8 xii,vecin(1:4,1:3),axlin(1:4,1:3)
      integer i,j
c
      xii=1-x
      if(jproc.eq.2)then
        call freal_qq(s,xii,yi,tk,uk,q1q,q2q,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      elseif(jproc.eq.3)then
        call freal_qg(s,xii,yi,tk,uk,q1q,q2q,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      else
        write(*,*)'Unknown process in freal',jproc
        stop
      endif
      return
      end


      subroutine freal_qq(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,vecout,axlout)
c Real matrix elements for qqbar --> l1 l2 g, from eq.(5) of NPB185(81)274
      implicit none
      real * 8 xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      real * 8 s,xii,yi,tk,uk,q1q,q2q,tiny,pi,vcf,s_red,t_red,
     #  x_ap,ap_kern
      real * 8 vecin(1:4,1:3),axlin(1:4,1:3)
      character * 4 stype
      integer i,j,itwo,icode
      parameter (tiny=1.d-6)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (itwo=2)
c
      s=xs
      xii=xxii
      yi=xyi
      tk=xtk
      uk=xuk
      q1q=xq1q
      q2q=xq2q
      if(xii.lt.tiny)then
        s_red=s*(1-xii)
        t_red=q1q
        call fborn(s_red,t_red,itwo,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=16*vcf/s_red*vecin(i,j)
            axlout(i,j)=16*vcf/s_red*axlin(i,j)
          enddo
        enddo
      elseif(yi.gt.1-tiny)then
        icode=4
        x_ap=1-xii
        s_red=s*x_ap
        t_red=q1q*x_ap
        call fborn(s_red,t_red,itwo,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      elseif(yi.lt.-1+tiny)then
        icode=4
        x_ap=1-xii
        s_red=s*x_ap
        t_red=q1q
        call fborn(s_red,t_red,itwo,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qqre'
        call xmewrap(s,tk,uk,q1q,q2q,itwo,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=xii**2*(1-yi**2)*vecin(i,j)
            axlout(i,j)=xii**2*(1-yi**2)*axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine freal_qg(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,vecout,axlout)
c Real matrix elements for qg --> l1 l2 q, from eq.(6) of NPB185(81)274
      implicit none
      real * 8 xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      real * 8 s,xii,yi,tk,uk,q1q,q2q,tiny,pi,vcf,s_red,t_red,
     #  x_ap,ap_kern
      real * 8 vecin(1:4,1:3),axlin(1:4,1:3)
      character * 4 stype
      integer i,j,itwo,ithree,icode
      parameter (tiny=1.d-6)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (itwo=2)
      parameter (ithree=3)
c
      s=xs
      xii=xxii
      yi=xyi
      tk=xtk
      uk=xuk
      q1q=xq1q
      q2q=xq2q
      do i=1,4
        do j=1,3
          vecout(i,j)=0.d0
          axlout(i,j)=0.d0
        enddo
      enddo
      if(xii.lt.tiny)then
        continue
      elseif(yi.gt.1-tiny)then
        icode=2
        x_ap=1-xii
        s_red=s*x_ap
        t_red=q1q*x_ap
        call fborn(s_red,t_red,itwo,vecin,axlin)
        do i=3,4
          do j=1,3
            vecout(i,j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      elseif(yi.lt.-1+tiny)then
        icode=2
        x_ap=1-xii
        s_red=s*x_ap
        t_red=q1q
        call fborn(s_red,t_red,itwo,vecin,axlin)
        do i=1,2
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qgre'
        call xmewrap(s,tk,uk,q1q,q2q,ithree,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=xii**2*(1-yi**2)*vecin(i,j)
            axlout(i,j)=xii**2*(1-yi**2)*axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine f2pr(xs,xt,xx,xxc,xyic,xxlmude,jproc,
     #                vecout,axlout)
c Returns sig_2pr of FKS. It is derived from the function f2b_coll of the 
c single vector boson code
      implicit none
      real * 8 xs,xt,xx,xxc,xyic,xxlmude,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      include 'llpcblks.h'
      real * 8 s,t,xii,xiic,yic,xlmude,x_ap,s_red,one,xicut,
     # delta,tmp,xdfct1,xdfct2,xdfct3p,xdfct3l,xdfct5,
     # xrfct1,xrfct2,xrfct3p,xrfct3l,xrfct5,ap_kern,
     # apprime_kern,xkplus,xklog,xkreg,xkdelta
      real * 8 vecin(1:4,1:3),axlin(1:4,1:3)
      common/parsub/xicut,delta
      parameter (one=1.d0)
      character * 2 scheme
      integer i,j,jproc,icoded,icoder,itwo
      parameter (itwo=2)
c
      s=xs
      t=xt
      xii=1-xx
      xiic=1-xxc
      yic=xyic
      xlmude=xxlmude
c
      x_ap=1-xiic
      s_red=s*x_ap
      if(yic.eq.1.d0)then
        scheme=schhad1
        if(jproc.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc.eq.3)then
          icoded=0
          icoder=2
        else
          write(*,*)'Unknown process in f2pr',jproc
          stop
        endif
      elseif(yic.eq.-1.d0)then
        scheme=schhad2
        if(jproc.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc.eq.3)then
          icoded=2
          icoder=0
        else
          write(*,*)'Unknown process in f2pr',jproc
          stop
        endif
      else
        write(6,*)'Error in f2pr',yic
        stop
      endif
      if(icoded.ne.0.or.icoder.ne.0)then
        call fborn(s_red,t,itwo,vecin,axlin)
      else
        do i=1,4
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
        enddo
      endif
      if(icoded.ne.0)then
        xdfct1=ap_kern(x_ap,abs(icoded))
        xdfct2=apprime_kern(x_ap,abs(icoded))
        xdfct3p=0.d0
        xdfct3l=0.d0
        xdfct5=0.d0
c
        if(scheme.eq.'DI')then
        xdfct3p=xkplus(x_ap,abs(icoded))
          xdfct3l=xklog(x_ap,abs(icoded))
          if(xiic.ne.0.d0)then
            xdfct5=xkreg(x_ap,abs(icoded))
          else
            xdfct5=xkdelta(abs(icoded))
     #           +xkplus(one,abs(icoded))*log(xicut)
     #           +xklog(one,abs(icoded))*log(xicut)**2/2.d0
c This part contributes to sig2pr(soft), which is integrated in xi
c over the range (0,xicut). This implies the presence of a jacobian
c equal to xicut in the soft term, which has to be removed by hand
c in this case
            xdfct5=xdfct5/xicut
          endif
        elseif(scheme.ne.'MS')then
          write(6,*)'Error in f2pr, y=',yic
          write(6,*)'Factorization scheme ',scheme,' not known'
        endif
c
        tmp=xdfct1*(xlmude+2*log(xii))-xdfct2
     #      -xdfct3p-xdfct3l*log(xii) 
     #      -xii*xdfct5
        do i=1,2
          do j=1,3
            vecout(i,j)=tmp*vecin(i,j)
            axlout(i,j)=tmp*axlin(i,j)
          enddo
        enddo
      else
        do i=1,2
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
        enddo
      endif
      if(icoder.ne.0)then
        xrfct1=ap_kern(x_ap,abs(icoder))
        xrfct2=apprime_kern(x_ap,abs(icoder))
        xrfct3p=0.d0
        xrfct3l=0.d0
        xrfct5=0.d0
c
        if(scheme.eq.'DI')then
        xrfct3p=xkplus(x_ap,abs(icoder))
          xrfct3l=xklog(x_ap,abs(icoder))
          if(xiic.ne.0.d0)then
            xrfct5=xkreg(x_ap,abs(icoder))
          else
            xrfct5=xkdelta(abs(icoder))
     #           +xkplus(one,abs(icoder))*log(xicut)
     #           +xklog(one,abs(icoder))*log(xicut)**2/2.d0
c This part contributes to sig2pr(soft), which is integrated in xi
c over the range (0,xicut). This implies the presence of a jacobian
c equal to xicut in the soft term, which has to be removed by hand
c in this case
            xrfct5=xrfct5/xicut
          endif
        elseif(scheme.ne.'MS')then
          write(6,*)'Error in f2pr, y=',yic
          write(6,*)'Factorization scheme ',scheme,' not known'
        endif
c
        tmp=xrfct1*(xlmude+2*log(xii))-xrfct2
     #      -xrfct3p-xrfct3l*log(xii) 
     #      -xii*xrfct5
        do i=3,4
          do j=1,3
            vecout(i,j)=tmp*vecin(i,j)
            axlout(i,j)=tmp*axlin(i,j)
          enddo
        enddo
      else
        do i=3,4
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
        enddo
      endif
      return
      end


      subroutine f2sv(xs,xt,jproc,vecout,axlout)
c Returns sig_2pv of FKS. It is derived from the function f1sv of the 
c single vector boson code
      implicit none
      real * 8 xs,xt,vecout(1:4,1:3),axlout(1:4,1:3)
      include 'llpcblks.h'
      real * 8 tiny,pi,s,t,eikcon,fincon,vcf,xicut,delta,xmat
      real * 8 vecin(1:4,1:3),axlin(1:4,1:3)
      integer i,j,jproc,itwo
      common/parsub/xicut,delta
      parameter (tiny=1.d-8)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (itwo=2)
c
      if(jproc.eq.2)then
        s=xs
        t=xt
        if(abs(xmuf2h1-xmuf2h2).gt.tiny .or.
     #     abs(xmuf2h1-xmur2).gt.tiny)then
          write(*,*)'Error in f2sv: no such scale choice'
          stop
        endif
        eikcon=2*vcf*( 0.5d0*log(xicut**2*s/xmuf2h1)**2-
     #                 pi**2/6.d0 )
        fincon=-vcf*(log(xmuf2h1/xmll2)**2+
     #         3*log(xmuf2h1/xmll2)-pi**2+8)
        xmat=eikcon+fincon
      else
        xmat=0.d0
      endif
      if(xmat.ne.0.d0)call fborn(s,t,itwo,vecin,axlin)
      do i=1,4
        do j=1,3
          vecout(i,j)=xmat*vecin(i,j)
          axlout(i,j)=xmat*axlin(i,j)
        enddo
      enddo
      return
      end


      subroutine fborn(xs,xt,jproc,vecout,axlout)
c Returns the Born matrix elements; see the subroutine xmewrap for 
c details concerning the normalization
      implicit none
      real * 8 xs,xt,vecout(1:4,1:3),axlout(1:4,1:3)
      character * 4 stype
      real * 8 s,t,zero,vecin(1:4,1:3),axlin(1:4,1:3)
      parameter (zero=0.d0)
      integer i,j,jproc
c
      if(jproc.eq.2)then
        s=xs
        t=xt
        stype='born'
        call xmewrap(s,zero,zero,t,t,jproc,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      else
        do i=1,4
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
        enddo
      endif
      return
      end


      subroutine xmewrap(s,tk,uk,q1q,q2q,jproc,
     #                   vecout,axlout,stype)
c Returns the parts of eqs.(7) or (8) of NPB185(81)274 which depend non
c trivially upon the invariants:  the functions T of eqs.(4), (5), and (6), 
c the propagators (see the subroutine propagators), and the flux factor;
c the coefficient C_i (ccoef) is also included. The remaining parts,
c which depends upon the couplings and the parton identities, are defined
c in the routine setpar, and inserted in the main code.
c This subroutine is meant to be called with stype='born', 'qqre', 'qgre',
c which correspond to eqs.(4) [only the O(as^0) part],(5), and (6) respectively
      implicit none
      character * 4 stype
      real * 8 s,tk,uk,q1q,q2q,vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      include 'llpcblks.h'
      real * 8 sal,tal,ual,t1al,t2al,xk2al,ccoef,vectord,axiald,
     #  vectorr,axialr
      real * 8 props(1:3)
      integer i,i1,i3,j,jmin,jmax,iprdct
      common/ciprdct/iprdct
c
      sal=s
      tal=uk
      ual=tk
      t1al=q1q
      t2al=-s-uk-q2q
      xk2al=s+tk+uk
      do i=1,4
        do j=1,3
          vecout(i,j)=0.d0
          axlout(i,j)=0.d0
        enddo
      enddo
      call propagators(xk2al,props)
      if(iprdct.eq.135)then
        jmin=1
        jmax=3
      elseif(iprdct.eq.136)then
        jmin=3
        jmax=3
      elseif(iprdct.eq.137)then
        jmin=1
        jmax=1
      elseif(iprdct.eq.146.or.iprdct.eq.147)then
        jmin=3
        jmax=3
      else
        write(*,*)'Error in xmewrap: unknown IPROC',iprdct
        stop
      endif
      if(stype.eq.'born')then
        i1=2
        i3=4
        ccoef=1/3.d0
        call altdy0(sal,t1al,t2al,vectord,axiald)
        call altdy0(sal,t2al,t1al,vectorr,axialr)
      elseif(stype.eq.'qqre')then
        i1=2
        i3=4
        ccoef=4/9.d0
        call alta(sal,tal,ual,t1al,t2al,xk2al,vectord,axiald)
        call alta(sal,ual,tal,t2al,t1al,xk2al,vectorr,axialr)
      elseif(stype.eq.'qgre')then
        i1=2
        i3=4
        ccoef=1/6.d0
        call altc(sal,tal,ual,t1al,t2al,xk2al,vectord,axiald)
        call altc(sal,ual,tal,t2al,t1al,xk2al,vectorr,axialr)
      else
        write(*,*)'Error in xmewrap: unknown stype',stype
        stop
      endif
      do i=1,i1
        do j=jmin,jmax
          vecout(i,j)=ccoef*props(j)*vectord/(2*s)
          axlout(i,j)=sign(1.d0,1-dfloat(i))*
     #                ccoef*props(j)*axiald/(2*s)
        enddo
      enddo
      do i=3,i3
        do j=jmin,jmax
          vecout(i,j)=ccoef*props(j)*vectorr/(2*s)
          axlout(i,j)=sign(1.d0,3-dfloat(i))*
     #                ccoef*props(j)*axialr/(2*s)
        enddo
      enddo
      return
      end


      subroutine propagators(xs,props)
c Given the invariant mass squared of the lepton pair (xs, denoted as k^2
c in NPB185(81)274), returns the factors due to W, Z and gamma propagators
c which appear in eqs.(7) and (8); the couplings are set in setpar()
      implicit none
      real * 8 xs,props(1:3),s
      include 'llpcblks.h'
c
      s=xs
      props(1)=1/s**2
      props(2)=(s-xm02)/(s*((s-xm02)**2+xm02*gah**2))
      props(3)=1/((s-xm02)**2+xm02*gah**2)
      return
      end

      

      subroutine altdy0(xs,xt1,xt2,vector,axial)
c Returns the O(as^0) parts of eq.(4) in NPB185(81)274, for ep=0;
c the outputs of this routine are vector and axial, the terms which 
c multiply A_l A_q and B_l B_q respectively
      implicit none
      real * 8 xs,xt1,xt2,vector,axial,s,t1,t2
c
      s=xs
      t1=xt1
      t2=xt2
      vector=8*(t1**2+t2**2)
      axial=8*(t1**2-t2**2)
      return
      end


      subroutine alta(xs,xt,xu,xt1,xt2,xxk2,vector,axial)
c Returns eq.(5) in NPB185(81)274, for ep=0 and g^2=1;
c the outputs of this routine are vector and axial, the terms 
c which multiply A_l A_q and B_l B_q respectively
      implicit none
      real * 8 xs,xt,xu,xt1,xt2,xxk2,vector,axial
      real * 8 s,t,u,t1,t2,xk2,altavt,altaax
c
      s=xs
      t=xt
      u=xu
      t1=xt1
      t2=xt2
      xk2=xxk2
      vector=altavt(s,t,u,t1,t2,xk2)+altavt(s,u,t,t2,t1,xk2)
      vector=-16*xk2*vector
      axial=altaax(s,t,u,t1,t2,xk2)-altaax(s,u,t,t2,t1,xk2)
      axial=-16*xk2*axial
      return
      end


      function altavt(xs,xt,xu,xt1,xt2,xxk2)
c Auxiliary function for the vector part of the routine alta
      implicit none
      real * 8 xs,xt,xu,xt1,xt2,xxk2,altavt
      real * 8 s,t,u,t1,t2,xk2
c
      s=xs
      t=xt
      u=xu
      t1=xt1
      t2=xt2
      xk2=xxk2
      altavt=1+(s-2*t1-xk2)/t-(t1**2+t2**2+s*(t1+t2+xk2))/(t*u)
      return
      end


      function altaax(xs,xt,xu,xt1,xt2,xxk2)
c Auxiliary function for the vector part of the routine alta
      implicit none
      real * 8 xs,xt,xu,xt1,xt2,xxk2,altaax
      real * 8 s,t,u,t1,t2,xk2
c
      s=xs
      t=xt
      u=xu
      t1=xt1
      t2=xt2
      xk2=xxk2
      altaax=(s+2*t2+xk2)/t+xk2*(t1-t2)/(t*u)
      return
      end


      subroutine altc(xs,xt,xu,xt1,xt2,xxk2,vector,axial)
c Returns eq.(6) in NPB185(81)274, for ep=0 and g^2=1;
c the outputs of this routine are vector and axial, the terms 
c which multiply A_l A_q and B_l B_q respectively
      implicit none
      real * 8 xs,xt,xu,xt1,xt2,xxk2,vector,axial
      real * 8 s,t,u,t1,t2,xk2
c
      s=xs
      t=xt
      u=xu
      t1=xt1
      t2=xt2
      xk2=xxk2
      vector=(t-2*(t1+xk2))/s+(s+2*(t1+t2))/t+
     #       2*((t1+t2+xk2)**2+t1**2-t2*xk2)/(s*t)
      vector=-16*xk2*vector
      axial=(2*(t1+xk2)-t)/s+(s+2*(t1+t2))/t-
     #      2*xk2*(2*t1+t2+xk2)/(s*t)
      axial=-16*xk2*axial
      return
      end
c
c
c From the jet package, Altarelli-Parisi kernels and change of scheme
c
c
      function ap_kern(x,index)
c This function returns the quantity (1-x)*P_{ab}(x), where
c P_{ab} are the Altarelli-Parisi kernels, and the splitting partons
c {ab} are defined with the following conventions
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
        ap_kern=2*vca*(x+(1-x)**2/x+x*(1-x)**2)
      elseif(index.eq.2)then
        ap_kern=vtf*(1-x)*(x**2+(1-x)**2)
      elseif(index.eq.3)then
        ap_kern=vcf*(1-x)*(1+(1-x)**2)/x
      elseif(index.eq.4)then
        ap_kern=vcf*(1+x**2)
      else
        write(6,*)'Error in ap_kern: wrong index value'
        stop
      endif
      return
      end


      function apprime_kern(x,index)
c This function returns the quantity (1-x)*P_{ab}^{prime}(x), where
c P_{ab}^{prime} is the ep-dependent part of the Altarelli-Parisi kernels, 
c and the codes for the splitting partons {ab} are defined above
      implicit real * 8 (a-h,o-z)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
c
      if(index.eq.1)then
        apprime_kern=0.d0
      elseif(index.eq.2)then
        apprime_kern=-2*vtf*x*(1-x)**2
      elseif(index.eq.3)then
        apprime_kern=-vcf*(1-x)*x
      elseif(index.eq.4)then
        apprime_kern=-vcf*(1-x)**2
      else
        write(6,*)'Error in apprime_kern: wrong index value'
        stop
      endif
      return
      end


      function xkdelta(index)
c This function returns the quantity K^{(d)}_{ab}, relevant for
c the MS --> DIS change in the factorization scheme. 
c The codes for the splitting partons {ab} are defined above
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)
      common/nl/nl
c
      if(index.eq.1)then
        xkdelta=0.d0
      elseif(index.eq.2)then
        xkdelta=0.d0
      elseif(index.eq.3)then
        xkdelta=vcf*(9.d0/2.d0+pi**2/3.d0)
      elseif(index.eq.4)then
        xkdelta=-vcf*(9.d0/2.d0+pi**2/3.d0)
      else
        write(6,*)'Error in xkdelta: wrong index value'
        stop
      endif
      return
      end


      function xkplus(x,index)
c This function returns the quantity K^{(+)}_{ab}(x), relevant for
c the MS --> DIS change in the factorization scheme. Notice that
c there's NO multiplicative (1-x) factor like in the previous functions.
c The codes for the splitting partons {ab} are defined above
      implicit real * 8 (a-h,o-z)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)
      common/nl/nl
c
      if(index.eq.1)then
        xkplus=0.d0
      elseif(index.eq.2)then
        xkplus=0.d0
      elseif(index.eq.3)then
        xkplus=-vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
      elseif(index.eq.4)then
        xkplus=vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
      else
        write(6,*)'Error in xkplus: wrong index value'
        stop
      endif
      return
      end


      function xklog(x,index)
c This function returns the quantity K^{(l)}_{ab}(x), relevant for
c the MS --> DIS change in the factorization scheme. Notice that
c there's NO multiplicative (1-x) factor like in the previous functions.
c The codes for the splitting partons {ab} are defined above
      implicit real * 8 (a-h,o-z)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)
      common/nl/nl
c
      if(index.eq.1)then
        xklog=0.d0
      elseif(index.eq.2)then
        xklog=0.d0
      elseif(index.eq.3)then
        xklog=-vcf*(1+x**2)
      elseif(index.eq.4)then
        xklog=vcf*(1+x**2)
      else
        write(6,*)'Error in xklog: wrong index value'
        stop
      endif
      return
      end


      function xkreg(x,index)
c This function returns the quantity K^{(reg)}_{ab}(x), relevant for
c the MS --> DIS change in the factorization scheme. Notice that
c there's NO multiplicative (1-x) factor like in the previous functions.
c The codes for the splitting partons {ab} are defined above
      implicit real * 8 (a-h,o-z)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)
      common/nl/nl
c
      if(index.eq.1)then
        xkreg=-2*nl*vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
      elseif(index.eq.2)then
        xkreg=vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
      elseif(index.eq.3)then
        xkreg=0.d0
      elseif(index.eq.4)then
        xkreg=0.d0
      else
        write(6,*)'Error in xkreg: wrong index value'
        stop
      endif
      return
      end


