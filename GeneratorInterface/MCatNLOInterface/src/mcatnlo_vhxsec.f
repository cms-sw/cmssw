      subroutine freal_und(s,x,yi,tk,uk,q1q,q2q,jproc,vecout,axlout)
c Returns the real matrix elements times xii**2*(1-yi**2)=4*tk*uk/s**2,
c in the case of undecayed vector boson. See the subroutine xmewrap_und 
c for details concerning the normalization
      implicit none
      real * 8 s,x,yi,tk,uk,q1q,q2q,vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      include 'vhgcblks.h'
      real * 8 xii,vecin(1:4,1:3),axlin(1:4,1:3)
      integer i,j
c
      xii=1-x
      if(jproc.eq.2)then
        call freal_und_qq(s,xii,yi,tk,uk,q1q,q2q,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      elseif(jproc.eq.3)then
        call freal_und_qg(s,xii,yi,tk,uk,q1q,q2q,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      else
        write(*,*)'Unknown process in freal_und',jproc
        stop
      endif
      return
      end


      subroutine freal_dec(s,x,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                     jproc,vecout,axlout)
c Returns the real matrix elements times xii**2*(1-yi**2)=4*tk*uk/s**2,
c in the case of decayed vector boson; it is the analogue of freal_und. 
c See the subroutine xmewrap_dec for details concerning the normalization
      implicit none
      real * 8 s,x,yi,tk,uk,q1q,q2q,vecout(1:4,1:3),axlout(1:4,1:3)
      real * 8 v1a,v1b,v1c
      integer jproc
      include 'vhgcblks.h'
      real * 8 xii,vecin(1:4,1:3),axlin(1:4,1:3)
      integer i,j
c
      xii=1-x
      if(jproc.eq.2)then
        call freal_dec_qq(s,xii,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                    vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      elseif(jproc.eq.3)then
        call freal_dec_qg(s,xii,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                    vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      else
        write(*,*)'Unknown process in freal_dec',jproc
        stop
      endif
      return
      end


      subroutine freal_und_qq(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #                        vecout,axlout)
c Real matrix elements for qqbar --> V H g
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
        call fborn_und(s_red,t_red,itwo,vecin,axlin)
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
        t_red=q2q
        call fborn_und(s_red,t_red,itwo,vecin,axlin)
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
        call fborn_und(s_red,t_red,itwo,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qqre'
        call xmewrap_und(s,tk,uk,q1q,q2q,itwo,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine freal_dec_qq(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #                        xv1a,xv1b,xv1c,vecout,axlout)
c Real matrix elements for qqbar --> V(-->llbar) H g
      implicit none
      real * 8 xs,xxii,xyi,xtk,xuk,xq1q,xq2q,xv1a,xv1b,xv1c,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      real * 8 s,xii,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #  tiny,pi,vcf,s_red,t_red,v1a_red,v1b_red,x_ap,ap_kern
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
      v1a=xv1a
      v1b=xv1b
      v1c=xv1c
      if(xii.lt.tiny)then
        s_red=s*(1-xii)
        t_red=q1q
        v1a_red = v1a
        v1b_red = v1b
        call fborn_dec(s_red,t_red,v1a_red,v1b_red,itwo,vecin,axlin)
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
        t_red=q2q
        v1a_red = v1a*x_ap
        v1b_red = v1b
        call fborn_dec(s_red,t_red,v1a_red,v1b_red,itwo,vecin,axlin)
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
        v1a_red = v1a
        v1b_red = v1b*x_ap
        call fborn_dec(s_red,t_red,v1a_red,v1b_red,itwo,vecin,axlin)
        do i=1,4
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qqre'
        call xmewrap_dec(s,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                   itwo,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine freal_und_qg(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #                        vecout,axlout)
c Real matrix elements for qg --> V H q
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
        t_red=q2q
        call fborn_und(s_red,t_red,itwo,vecin,axlin)
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
        call fborn_und(s_red,t_red,itwo,vecin,axlin)
        do i=1,2
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qgre'
        call xmewrap_und(s,tk,uk,q1q,q2q,ithree,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine freal_dec_qg(xs,xxii,xyi,xtk,xuk,xq1q,xq2q,
     #                        xv1a,xv1b,xv1c,vecout,axlout)
c Real matrix elements for qg --> V(-->llbar) H q
      implicit none
      real * 8 xs,xxii,xyi,xtk,xuk,xq1q,xq2q,xv1a,xv1b,xv1c,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      real * 8 s,xii,yi,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #  tiny,pi,vcf,s_red,t_red,v1a_red,v1b_red,x_ap,ap_kern
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
      v1a=xv1a
      v1b=xv1b
      v1c=xv1c
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
        t_red=q2q
        v1a_red = v1a*x_ap
        v1b_red = v1b
        call fborn_dec(s_red,t_red,v1a_red,v1b_red,itwo,vecin,axlin)
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
        v1a_red = v1a
        v1b_red = v1b*x_ap
        call fborn_dec(s_red,t_red,v1a_red,v1b_red,itwo,vecin,axlin)
        do i=1,2
          do j=1,3
            vecout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*vecin(i,j)
            axlout(i,j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*axlin(i,j)
          enddo
        enddo
      else
        stype='qgre'
        call xmewrap_dec(s,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #                   ithree,vecin,axlin,stype)
        do i=1,4
          do j=1,3
            vecout(i,j)=vecin(i,j)
            axlout(i,j)=axlin(i,j)
          enddo
        enddo
      endif
      return
      end


      subroutine f2pr_und(xs,xt,xx,xxc,xyic,xxlmude,jproc,
     #                    vecout,axlout)
c Returns sig_2pr of FKS. It is derived from the function f2b_coll of the 
c single vector boson code
      implicit none
      real * 8 xs,xt,xx,xxc,xyic,xxlmude,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      include 'vhgcblks.h'
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
          write(*,*)'Unknown process in f2pr_und',jproc
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
          write(*,*)'Unknown process in f2pr_und',jproc
          stop
        endif
      else
        write(6,*)'Error in f2pr_und',yic
        stop
      endif
      if(icoded.ne.0.or.icoder.ne.0)then
        call fborn_und(s_red,t,itwo,vecin,axlin)
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
          write(6,*)'Error in f2pr_und, y=',yic
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
          write(6,*)'Error in f2pr_und, y=',yic
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


      subroutine f2pr_dec(xs,xt,xv1a,xv1b,xv1c,xx,xxc,xyic,xxlmude,
     #                    jproc,vecout,axlout)
c  the same as f2pr_und above, but with decay V --> llbar
      implicit none
      real * 8 xs,xt,xv1a,xv1b,xv1c,xx,xxc,xyic,xxlmude,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      include 'vhgcblks.h'
      real * 8 s,t,v1a,v1b,v1c,xii,xiic,yic,xlmude,x_ap,s_red,
     # one,xicut,delta,tmp,xdfct1,xdfct2,xdfct3p,xdfct3l,xdfct5,
     # xrfct1,xrfct2,xrfct3p,xrfct3l,xrfct5,ap_kern,
     # apprime_kern,xkplus,xklog,xkreg,xkdelta,v1a_red,v1b_red
      real * 8 vecin(1:4,1:3),axlin(1:4,1:3)
      common/parsub/xicut,delta
      parameter (one=1.d0)
      character * 2 scheme
      integer i,j,jproc,icoded,icoder,itwo
      parameter (itwo=2)
c
      s=xs
      t=xt
      v1a=xv1a
      v1b=xv1b
      v1c=xv1c
      xii=1-xx
      xiic=1-xxc
      yic=xyic
      xlmude=xxlmude
c
      x_ap=1-xiic
      s_red=s*x_ap
      if(yic.eq.1.d0)then
        scheme=schhad1
        v1a_red=v1a*x_ap
        v1b_red=v1b
        if(jproc.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc.eq.3)then
          icoded=0
          icoder=2
        else
          write(*,*)'Unknown process in f2pr_dec',jproc
          stop
        endif
      elseif(yic.eq.-1.d0)then
        scheme=schhad2
        v1a_red=v1a
        v1b_red=v1b*x_ap
        if(jproc.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc.eq.3)then
          icoded=2
          icoder=0
        else
          write(*,*)'Unknown process in f2pr_dec',jproc
          stop
        endif
      else
        write(6,*)'Error in f2pr_dec',yic
        stop
      endif
      if(icoded.ne.0.or.icoder.ne.0)then
        call fborn_dec(s_red,t,v1a_red,v1b_red,itwo,vecin,axlin)
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
          write(6,*)'Error in f2pr_dec, y=',yic
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
          write(6,*)'Error in f2pr_dec, y=',yic
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


      subroutine f2sv_und(xs,xt,jproc,vecout,axlout)
c Returns sig_2pv of FKS. It is derived from the function f1sv of the 
c single vector boson code
      implicit none
      real * 8 xs,xt,vecout(1:4,1:3),axlout(1:4,1:3)
      include 'vhgcblks.h'
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
          write(*,*)'Error in f2sv_und: no such scale choice'
          stop
        endif
        eikcon=2*vcf*( 0.5d0*log(xicut**2*s/xmuf2h1)**2-
     #                 pi**2/6.d0 )
        fincon=-vcf*(log(xmuf2h1/xmi2)**2+
     #         3*log(xmuf2h1/xmi2)-pi**2+8)
        xmat=eikcon+fincon
      else
        xmat=0.d0
      endif
      if(xmat.ne.0.d0)call fborn_und(s,t,itwo,vecin,axlin)
      do i=1,4
        do j=1,3
          vecout(i,j)=xmat*vecin(i,j)
          axlout(i,j)=xmat*axlin(i,j)
        enddo
      enddo
      return
      end


      subroutine f2sv_dec(xs,xt,xv1a,xv1b,xv1c,jproc,vecout,axlout)
c  the same as f2sv_und above, but with decay V --> llbar
      implicit none
      real * 8 xs,xt,xv1a,xv1b,xv1c,vecout(1:4,1:3),axlout(1:4,1:3)
      include 'vhgcblks.h'
      real * 8 tiny,pi,s,t,v1a,v1b,v1c
      real * 8 eikcon,fincon,vcf,xicut,delta,xmat
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
        v1a=xv1a
        v1b=xv1b
        v1c=xv1c
        if(abs(xmuf2h1-xmuf2h2).gt.tiny .or.
     #     abs(xmuf2h1-xmur2).gt.tiny)then
          write(*,*)'Error in f2sv_und: no such scale choice'
          stop
        endif
        eikcon=2*vcf*( 0.5d0*log(xicut**2*s/xmuf2h1)**2-
     #                 pi**2/6.d0 )
        fincon=-vcf*(log(xmuf2h1/xmi2)**2+
     #         3*log(xmuf2h1/xmi2)-pi**2+8)
        xmat=eikcon+fincon
      else
        xmat=0.d0
      endif
      if(xmat.ne.0.d0)call fborn_dec(s,t,v1a,v1b,itwo,vecin,axlin)
      do i=1,4
        do j=1,3
          vecout(i,j)=xmat*vecin(i,j)
          axlout(i,j)=xmat*axlin(i,j)
        enddo
      enddo
      return
      end


      subroutine fborn_und(xs,xt,jproc,vecout,axlout)
c Returns the Born matrix elements in the case of undecayed vector boson. 
c See the subroutine xmewrap_und for details concerning the normalization
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
        call xmewrap_und(s,zero,zero,t,t,jproc,vecin,axlin,stype)
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


      subroutine fborn_dec(xs,xt,xv1a,xv1b,jproc,vecout,axlout)
c Returns the Born matrix elements in the case of decayed vector boson;
c it is the analogue of fborn_und. See the subroutine xmewrap_dec for 
c details concerning the normalization
      implicit none
      real * 8 xs,xt,xv1a,xv1b,vecout(1:4,1:3),axlout(1:4,1:3)
      include 'vhgcblks.h'
      character * 4 stype
      real * 8 s,t,v1a,v1b,v1c,zero,vecin(1:4,1:3),axlin(1:4,1:3)
      parameter (zero=0.d0)
      integer i,j,jproc
c
      if(jproc.eq.2)then
        s=xs
        t=xt
        v1a=xv1a
        v1b=xv1b
        v1c=-v1a-v1b-xmv2
        stype='born'
        call xmewrap_dec(s,zero,zero,t,t,v1a,v1b,v1c,jproc,
     #                   vecin,axlin,stype)
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


      subroutine xmewrap_und(s,tk,uk,q1q,q2q,jproc,
     #                       vecout,axlout,stype)
c Returns the non-trivial part of the matrix elements in the case of
c undecayed vector boson. The normalization is
c
c    dsigma = fact * M * dPhi,  with
c
c    M(ab->cVH) = g_VH^2 * g_Vqq^2 * gs^2 * (V_V^2+A_V^2) * vecout
c    vecout = 1/(2*s) * 1/((M_VH^2-M_V^2)^2+(M_V*Gamma_V)^2) * vectord
c
c    fact = 1                for Born contribution
c         = 4*tk*uk/s**2     for real contributions
c
c and vectord can be replaced by vectorr depending on the identities
c of parton a and b. The entries axlout are unused in this routine. The
c coupling factor gs^2 is not present in the case of the Born contribution.
c A factor M_V^2 due to the VVH vertex has been inserted in the routines
c that return vectord. The coupling factors are defined in setpar and 
c inserted in the main code.
c This subroutine is meant to be called with stype='born', 'qqre', 'qgre'.
      implicit none
      character * 4 stype
      real * 8 s,tk,uk,q1q,q2q,vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      include 'vhgcblks.h'
      real * 8 dummy,s2,q1c,q2c,cos,coq2,coppv,copppv,cot,cou,
     #  vectord,axiald,vectorr,axialr,co_born_und,co_realqq_und,
     #  co_realqg_und
      real * 8 props(1:3)
      integer i,i1,i3,j,jmin,jmax
c
      dummy=xm0v2
      s2=s+tk+uk
      q1c=xmv2+xmh2-s-tk-q1q
      q2c=xmv2+xmh2-s-uk-q2q
      cos=s
      coq2=s2
      coppv=-(q1q-xmv2)/2.d0
      copppv=-(q2c-xmv2)/2.d0
      cot=uk
      cou=tk
      do i=1,4
        do j=1,3
          vecout(i,j)=0.d0
          axlout(i,j)=0.d0
        enddo
      enddo
      call propagatovh(coq2,dummy,props)
      jmin=3
      jmax=3
      if(stype.eq.'born')then
        i1=2
        i3=4
        vectord=co_born_und(cos,coppv,copppv,xmv2,xmh2)
        vectorr=co_born_und(cos,copppv,coppv,xmv2,xmh2)
      elseif(stype.eq.'qqre')then
        i1=2
        i3=4
        vectord=co_realqq_und(cos,coppv,copppv,coq2,cot,cou,xmv2,xmh2)
        vectorr=co_realqq_und(cos,copppv,coppv,coq2,cou,cot,xmv2,xmh2)
      elseif(stype.eq.'qgre')then
        i1=2
        i3=4
        vectord=co_realqg_und(cos,coppv,copppv,coq2,cot,cou,xmv2,xmh2)
        vectorr=co_realqg_und(cos,copppv,coppv,coq2,cou,cot,xmv2,xmh2)
      else
        write(*,*)'Error in xmewrap_und: unknown stype',stype
        stop
      endif
      axiald=0.d0
      axialr=0.d0
      do i=1,i1
        do j=jmin,jmax
          vecout(i,j)=props(j)*vectord/(2*s)
          axlout(i,j)=sign(1.d0,1-dfloat(i))*
     #                props(j)*axiald/(2*s)
        enddo
      enddo
      do i=3,i3
        do j=jmin,jmax
          vecout(i,j)=props(j)*vectorr/(2*s)
          axlout(i,j)=sign(1.d0,3-dfloat(i))*
     #                props(j)*axialr/(2*s)
        enddo
      enddo
      return
      end


      subroutine xmewrap_dec(s,tk,uk,q1q,q2q,v1a,v1b,v1c,jproc,
     #                       vecout,axlout,stype)
c Returns the non-trivial part of the matrix elements in the case of
c decayed vector boson; it is the analogue of xmewrap_und. The normalization is
c
c    dsigma = fact * M * dPhi,  with
c
c    M(ab->cV(->ll)H) = g_VH^2 * g_Vqq^2 * g_Vll^2 * gs^2 * 
c           [ (V_Vq^2+A_Vq^2) * (V_Vl^2+A_Vl^2) * vecout +
c             4 * V_Vq * A_Vq * V_Vl * A_Vl * axlout ]
c    vecout = 1/(2*s) * 1/((M_VH^2-M_V^2)^2+(M_V*Gamma_V)^2) * 
c             1/((M_ll^2-M_V^2)^2+(M_V*Gamma_V)^2) * vectord
c    axlout = 1/(2*s) * 1/((M_VH^2-M_V^2)^2+(M_V*Gamma_V)^2) * 
c             1/((M_ll^2-M_V^2)^2+(M_V*Gamma_V)^2) * axiald
c
c    fact = 1                for Born contribution
c         = 4*tk*uk/s**2     for real contributions
c
c and vectord/axiald can be replaced by vectorr/axialr depending on 
c the identities of parton a and b. The coupling factor gs^2 is not 
c present in the case of the Born contribution. A factor M_V^2 due to 
c the VVH vertex has been inserted in the routines that return vectord 
c and axiald. The coupling factors are defined in setpar and inserted in 
c the main code.
c This subroutine is meant to be called with stype='born', 'qqre', 'qgre'.
      implicit none
      character * 4 stype
      real * 8 s,tk,uk,q1q,q2q,v1a,v1b,v1c,
     #  vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      include 'vhgcblks.h'
      real * 8 s2,q1c,q2c,cos,coq2,coppv,copppv,cot,cou,
     #  coppl1,copppl1,coppl2,copppl2,coqpl1,coqpl2,
     #  vectord,axiald,vectorr,axialr
      real * 8 props(1:3)
      integer i,i1,i3,j,jmin,jmax
c
      s2=s+tk+uk
      q1c=xmv2+xmh2-s-tk-q1q
      q2c=xmv2+xmh2-s-uk-q2q
      cos=s
      coq2=s2
      coppv=-(q1q-xmv2)/2.d0
      copppv=-(q2c-xmv2)/2.d0
      cot=uk
      cou=tk
      coppl1 = -v1a/2.d0
      copppl1 = -v1b/2.d0
      coppl2 = (xmv2-q1q+v1a)/2.d0
      copppl2 = (-xmh2+q2q+s+uk+v1b)/2.d0
      coqpl1 = (v1c+xmv2)/2.d0
      coqpl2 = (-xmh2+s+tk+uk-v1c)/2.d0
      do i=1,4
        do j=1,3
          vecout(i,j)=0.d0
          axlout(i,j)=0.d0
        enddo
      enddo
      call propagatovh(coq2,xmv2,props)
      jmin=3
      jmax=3
      if(stype.eq.'born')then
        i1=2
        i3=4
        call co_born_dec(coppl1,copppl1,coppl2,copppl2,xm0v2,xm0h2,
     #                   vectord,axiald)
        call co_born_dec(copppl1,coppl1,copppl2,coppl2,xm0v2,xm0h2,
     #                   vectorr,axialr)
      elseif(stype.eq.'qqre')then
        i1=2
        i3=4
        call co_realqq_dec(cos,cot,cou,coppl1,copppl1,coppl2,
     #        copppl2,coqpl1,coqpl2,xm0v2,xm0h2,vectord,axiald)
        call co_realqq_dec(cos,cou,cot,copppl1,coppl1,copppl2,
     #        coppl2,coqpl1,coqpl2,xm0v2,xm0h2,vectorr,axialr)
      elseif(stype.eq.'qgre')then
        i1=2
        i3=4
        call co_realqg_dec(cos,cot,cou,coppl1,copppl1,coppl2,
     #        copppl2,coqpl1,coqpl2,xm0v2,xm0h2,vectord,axiald)
        call co_realqg_dec(cos,cou,cot,copppl1,coppl1,copppl2,
     #        coppl2,coqpl1,coqpl2,xm0v2,xm0h2,vectorr,axialr)
      else
        write(*,*)'Error in xmewrap_dec: unknown stype',stype
        stop
      endif
      do i=1,i1
        do j=jmin,jmax
          vecout(i,j)=props(j)*vectord/(2*s)
          axlout(i,j)=sign(1.d0,1-dfloat(i))*
     #                props(j)*axiald/(2*s)
        enddo
      enddo
      do i=3,i3
        do j=jmin,jmax
          vecout(i,j)=props(j)*vectorr/(2*s)
          axlout(i,j)=sign(1.d0,3-dfloat(i))*
     #                props(j)*axialr/(2*s)
        enddo
      enddo
      return
      end


      subroutine propagatovh(xsvh,xsll,props)
c Given the invariant mass squared of the VH pair (xsvh) and of the 
c lepton pair (xsll), returns the factors due to the intermediate and 
c final-state V propagatovh. In the case of undecayed vector boson, the
c latter is set to one, and xsll is unused. The couplings are set in setpar()
      implicit none
      include 'vhgcblks.h'
      real * 8 xsvh,xsll,props(1:3),svh,sll
      integer iprdct
      common/ciprdct/iprdct
c
      svh=xsvh
      sll=xsll
      props(1)=0.d0
      props(2)=0.d0
      if(iprdct.eq.248.or.iprdct.eq.268.or.iprdct.eq.278)then
        props(3)=1/((svh-xm0i2)**2+xm0i2*gai**2)
      elseif(iprdct.eq.240.or.iprdct.eq.260.or.iprdct.eq.270)then
        props(3)=1/((svh-xm0i2)**2+xm0i2*gai**2)*
     #           1/((sll-xm0v2)**2+xm0v2*gav**2)
      endif
      return
      end

      
      function co_born_und(xs,xppv,xpppv,xxmv2,xxmh2)
c Born-level qqbar --> V H; see xmewrap_und for the normalization
      implicit none
      real * 8 co_born_und,xs,xppv,xpppv,xxmv2,xxmh2,
     #  q2,ppv,pppv,xmv2,xnc
      parameter (xnc=3.d0)
c
      q2=xs
      ppv=xppv
      pppv=xpppv
      xmv2=xxmv2
      co_born_und=2*q2*xmv2+8*pppv*ppv
      co_born_und=co_born_und/(4*xnc)
      return
      end


      subroutine co_born_dec(xppl1,xpppl1,xppl2,xpppl2,xxmv2,xxmh2,
     #                       vector,axial)
c Born-level qqbar --> V(->ll) H; see xmewrap_dec for the normalization
      implicit none
      real * 8 xppl1,xpppl1,xppl2,xpppl2,xxmv2,xxmh2,ppl1,pppl1,
     #  ppl2,pppl2,xmv2,vector,axial,xnc
      parameter (xnc=3.d0)
c
      ppl1=xppl1
      pppl1=xpppl1
      ppl2=xppl2
      pppl2=xpppl2
      xmv2=xxmv2
c The factor 8/xnc in vector and axial was originally 32/(4*xnc)
      vector=8*xmv2*(ppl2*pppl1+ppl1*pppl2)/xnc
      axial=8*xmv2*(ppl2*pppl1-ppl1*pppl2)/xnc
      return
      end


      function co_realqq_und(xs,xppv,xpppv,xq2,xt,xu,xxmv2,xxmh2)
c Tree-level qqbar --> V H g, times 4*tk*uk/s**2==4*xt*xu/xs**2;
c see xmewrap_und for the normalization
      implicit none
      real * 8 co_realqq_und,xs,xppv,xpppv,xq2,xt,xu,xxmv2,xxmh2
      real * 8 s,ppv,pppv,q2,t,u,xmv2,xmh2,vector,vcf,xnc
      parameter (vcf=4.d0/3.d0)
      parameter (xnc=3.d0)
c
      s=xs
      ppv=xppv
      pppv=xpppv
      q2=xq2
      t=xt
      u=xu
      xmv2=xxmv2
      xmh2=xxmh2
      vector = -4*q2*(ppv**2+pppv**2)
     #         +2*(u+s)*(q2+xmv2-xmh2)*ppv
     #         +2*(t+s)*(q2+xmv2-xmh2)*pppv
     #         +xmv2*(u**2+t**2+2*s*q2)
c This is vector * 4/(xmv2*t*u) * xmv2*vcf/(4*xnc) * 4*t*u/s**2
      co_realqq_und=4*vcf*vector/(xnc*s**2)
      return
      end


      subroutine co_realqq_dec(xs,xt,xu,xppl1,xpppl1,xppl2,xpppl2,
     #                         xqpl1,xqpl2,xxmv2,xxmh2,vector,axial)
c Tree-level qqbar --> V(->ll) H g, times 4*tk*uk/s**2==4*xt*xu/xs**2;
c see xmewrap_dec for the normalization
      implicit none
      real * 8 xs,xt,xu,xppl1,xpppl1,xppl2,xpppl2,xqpl1,xqpl2,
     #  xxmv2,xxmh2,vector,axial
      real * 8 s,t,u,q2,ppl1,pppl1,ppl2,pppl2,qpl1,qpl2,xmv2,xmh2,
     #  veckern,vecfact,axlkern,axlfact,vcf,xnc
      parameter (vcf=4.d0/3.d0)
      parameter (xnc=3.d0)
c
      s=xs
      t=xt
      u=xu
      q2=s+t+u
      ppl1=xppl1
      pppl1=xpppl1
      ppl2=xppl2
      pppl2=xpppl2
      qpl1=xqpl1
      qpl2=xqpl2
      xmv2=xxmv2
      xmh2=xxmh2
c The prefactors are xmv2*64/(t*u) * vcf/(4*xnc) * 4*t*u/s**2
      veckern=(t+s)*(pppl1*qpl2+pppl2*qpl1)
     #       +(u+s)*(ppl2*qpl1+ppl1*qpl2)
     #       -2*q2*(ppl1*ppl2+pppl1*pppl2)
      vecfact=64*vcf*xmv2/(xnc*s**2)
      axlkern=(t+s)*(pppl1*qpl2-pppl2*qpl1)
     #       +(u+s)*(ppl2*qpl1-ppl1*qpl2)
      axlfact=vecfact
      vector=veckern*vecfact
      axial=axlkern*axlfact
      return
      end


      function co_realqg_und(xs,xppv,xpppv,xq2,xt,xu,xxmv2,xxmh2)
c Tree-level qg --> V H q, times 4*tk*uk/s**2==4*xt*xu/xs**2; 
c see xmewrap_und for the normalization
      implicit none
      real * 8 co_realqg_und,xs,xppv,xpppv,xq2,xt,xu,xxmv2,xxmh2
      real * 8 s,ppv,pppv,q2,t,u,xmv2,xmh2,vector,vtf,xnc
      parameter (vtf=1.d0/2.d0)
      parameter (xnc=3.d0)
c
      s=xs
      ppv=xppv
      pppv=xpppv
      q2=xq2
      t=xt
      u=xu
      xmv2=xxmv2
      xmh2=xxmh2
      vector = 4*(2*ppv**2+2*pppv*ppv+pppv**2)*q2
     #       -2*(q2+xmv2-xmh2)*((t+2*u+3*s)*ppv+(t+u+2*s)*pppv)
     #       +(s-2*xmv2)*q2**2
     #       +2*(xmv2*t+2*xmv2*s-xmh2*s)*q2
     #       -(xmv2*t**2+xmv2*s**2-xmv2**2*s+2*xmh2*xmv2*s-xmh2**2*s)
c This is vector * 4/(xmv2*s*t) * xmv2*vtf/(4*xnc) * 4*t*u/s**2
      co_realqg_und=4*vtf*u*vector/(xnc*s**3)
      return
      end


      subroutine co_realqg_dec(xs,xt,xu,xppl1,xpppl1,xppl2,xpppl2,
     #                         xqpl1,xqpl2,xxmv2,xxmh2,vector,axial)
c Tree-level qg --> V(->ll) H q, times 4*tk*uk/s**2==4*xt*xu/xs**2;
c see xmewrap_dec for the normalization
      implicit none
      real * 8 xs,xt,xu,xppl1,xpppl1,xppl2,xpppl2,xqpl1,xqpl2,
     #  xxmv2,xxmh2,vector,axial
      real * 8 s,t,u,q2,ppl1,pppl1,ppl2,pppl2,qpl1,qpl2,xmv2,xmh2,
     #  veckern,vecfact,axlkern,axlfact,vtf,xnc
      parameter (vtf=1.d0/2.d0)
      parameter (xnc=3.d0)
c
      s=xs
      t=xt
      u=xu
      q2=s+t+u
      ppl1=xppl1
      pppl1=xpppl1
      ppl2=xppl2
      pppl2=xpppl2
      qpl1=xqpl1
      qpl2=xqpl2
      xmv2=xxmv2
      xmh2=xxmh2
c The prefactors are 64*xmv2/(s*t) * vtf/(4*xnc) * 4*t*u/s**2
      veckern=2*s*qpl1*qpl2
     #      +2*q2*(ppl1*pppl2+ppl2*pppl1+2*ppl1*ppl2+pppl1*pppl2)
     #      -(s+q2)*(pppl1*qpl2+pppl2*qpl1)
     #      -(2*u+t+3*s)*(ppl1*qpl2+ppl2*qpl1);
      vecfact=64*vtf*xmv2*u/(xnc*s**3)
      axlkern=(t+u)*(pppl1*qpl2-pppl2*qpl1)
     #       +(2*u+t+s)*(ppl1*qpl2-ppl2*qpl1)
      axlfact=vecfact
      vector=veckern*vecfact
      axial=axlkern*axlfact
      return
      end
c
c
c From the jet package, Altarelli-Parisi kernels and change of scheme
c
c
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
C        write(6,*)'Error in ap_kern: wrong index value'
C        stop
C      endif
C      return
C      end


C      function apprime_kern(x,index)
Cc This function returns the quantity (1-x)*P_{ab}^{prime}(x), where
Cc P_{ab}^{prime} is the ep-dependent part of the Altarelli-Parisi kernels, 
Cc and the codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
Cc
C      if(index.eq.1)then
C        apprime_kern=0.d0
C      elseif(index.eq.2)then
C        apprime_kern=-2*vtf*x*(1-x)**2
C      elseif(index.eq.3)then
C        apprime_kern=-vcf*(1-x)*x
C      elseif(index.eq.4)then
C        apprime_kern=-vcf*(1-x)**2
C      else
C        write(6,*)'Error in apprime_kern: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkdelta(index)
Cc This function returns the quantity K^{(d)}_{ab}, relevant for
Cc the MS --> DIS change in the factorization scheme. 
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (pi=3.14159265358979312D0)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkdelta=0.d0
C      elseif(index.eq.2)then
C        xkdelta=0.d0
C      elseif(index.eq.3)then
C        xkdelta=vcf*(9.d0/2.d0+pi**2/3.d0)
C      elseif(index.eq.4)then
C        xkdelta=-vcf*(9.d0/2.d0+pi**2/3.d0)
C      else
C        write(6,*)'Error in xkdelta: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkplus(x,index)
Cc This function returns the quantity K^{(+)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkplus=0.d0
C      elseif(index.eq.2)then
C        xkplus=0.d0
C      elseif(index.eq.3)then
C        xkplus=-vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
C      elseif(index.eq.4)then
C        xkplus=vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
C      else
C        write(6,*)'Error in xkplus: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xklog(x,index)
Cc This function returns the quantity K^{(l)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xklog=0.d0
C      elseif(index.eq.2)then
C        xklog=0.d0
C      elseif(index.eq.3)then
C        xklog=-vcf*(1+x**2)
C      elseif(index.eq.4)then
C        xklog=vcf*(1+x**2)
C      else
C        write(6,*)'Error in xklog: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkreg(x,index)
Cc This function returns the quantity K^{(reg)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkreg=-2*nl*vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
C      elseif(index.eq.2)then
C        xkreg=vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
C      elseif(index.eq.3)then
C        xkreg=0.d0
C      elseif(index.eq.4)then
C        xkreg=0.d0
C      else
C        write(6,*)'Error in xkreg: wrong index value'
C        stop
C      endif
C      return
C      end
