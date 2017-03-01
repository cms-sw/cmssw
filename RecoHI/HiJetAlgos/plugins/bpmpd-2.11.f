C Version for CMSSW, modified to suppress gfortran warnings
C -------------------------------------------------------------------
c Primal-dual method with supernodal cholesky factorization
c               Version 2.11 (1996 December)
c  Written by Cs. Meszaros, MTA SzTAKI, Budapest, Hungary
c        Questions, remarks to the e-mail address:
c               meszaros@lutra.sztaki.hu
c
c  All rights reserved ! Free for academic and research use only !
c  Commercial users are required to purchase a software license.
c
c Related publications:
c
c    Meszaros, Cs.: Fast Cholesky Factorization for Interior Point Methods
c       of Linear Programming. Computers & Mathematics with Applications,
c       Vol. 31. No.4/5 (1996) pp. 49-51.
c
c    Meszaros, Cs.: The "inexact" minimum local fill-in ordering algorithm.
c       Working Paper WP 95-7, Computer and Automation Institute, Hungarian
c       Academy of Sciences
c
c    Maros I., Meszaros Cs.: The Role of the Augmented System in Interior
c       Point Methods.  European Journal of Operations Researches
c       (submitted)
c
c ===========================================================================
c
c  Callable interface
c
c  Standard form: ax-s=b    u>=x,s>=l
c
c  remarks:
c          EQ  rows    0  >= s >=   0
c          GT  rows  +inf >= s >=   0
c          LT  rows    0  >= s >= -inf
c          FR  rows  +inf >= s >= -inf
c
c  input:   obj           objective function (to be minimize)       (n)
c           rhs           right-hand side                           (m)
c           lbound        lower bounds                              (m+n)
c           ubound        upper bounds                              (m+n)
c           colpnt        pointer to the columns                    (n+1)
c           rowidx        row indices                               (nz)
c           nonzeros      nonzero values                            (nz)
c           big           practical +inf
c
c  output: code           termination code
c          xs             primal values
c          dv             dual values
c          dspr           dual resuduals
c
c Input arrays will be destroyed !
c
c ===========================================================================
c
      subroutine solver(
     x obj,rhs,lbound,ubound,diag,odiag,xs,dxs,dxsn,up,dspr,ddspr,
     x ddsprn,dsup,ddsup,ddsupn,dv,ddv,ddvn,prinf,upinf,duinf,scale,
     x nonzeros,
     x vartyp,slktyp,colpnt,ecolpnt,count,vcstat,pivots,invprm,
     x snhead,nodtyp,inta1,prehis,rowidx,rindex,
     x code,opt,iter,corect,fixn,dropn,fnzmax,fnzmin,addobj,
     x bigbou,big,ft)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/initv/ prmin,upmax,dumin,stamet,safmet,premet,regul
      real*8        prmin,upmax,dumin
      integer*4     stamet,safmet,premet,regul
c
      integer*4 fixn,dropn,code,iter,corect,fnzmin,fnzmax,ft
      real*8  addobj,opt,big,
     x obj(n),rhs(m),lbound(mn),ubound(mn),scale(mn),diag(mn),odiag(mn),
     x xs(mn),dxs(mn),dxsn(mn),up(mn),dspr(mn),ddspr(mn),ddsprn(mn),
     x dsup(mn),ddsup(mn),ddsupn(mn),dv(m),ddv(m),ddvn(m),
     x nonzeros(cfree),prinf(m),upinf(mn),duinf(mn),bigbou
      integer*4 vartyp(n),slktyp(m),colpnt(n1),ecolpnt(mn),
     x count(mn),vcstat(mn),pivots(mn),invprm(mn),snhead(mn),
     x nodtyp(mn),inta1(mn),prehis(mn),rowidx(cfree),rindex(rfree)
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
      common/ascal/ objnor,rhsnor,scdiff,scpass,scalmet
      real*8        objnor,rhsnor,scdiff
      integer*4     scpass,scalmet
c ---------------------------------------------------------------------------
      integer*4 i,j,k,active,pnt1,pnt2,prelen,freen
      real*8 scobj,scrhs,sol,lbig
      character*99 buff
C CMSSW: Temporary integer array needed to avoid reusing REAL*8 for
C integer storage
      integer*4 pmbig(m),ppbig(m),dmbig(n),dpbig(n)
      integer*4 iwork1(mn+mn),iwork2(mn+mn),iwork3(mn+mn),iwork4(mn+mn),
     &     iwork5(mn+mn)
c ---------------------------------------------------------------------------
c
c inicializalas
c
      if(cfree.le.(nz+1)*2)then
        write(buff,'(1x,a)')'Not enough memory, realmem < nz !'
        call mprnt(buff)
        code=-2
        goto 50
      endif
      if(rfree.le.nz)then
        write(buff,'(1x,a)')'Not enough memory, intmem < nz !'
        call mprnt(buff)
        code=-2
        goto 50
      endif
      iter=0
      corect=0
      prelen=0
      fnzmin=cfree
      fnzmax=-1
      scobj=1.0d+0
      scrhs=1.0d+0
      code=0
      lbig=0.9d+0*big
      if(bigbou.gt.lbig)then
        lbig=bigbou
        big=lbig/0.9d+0
      endif
      do i=1,mn
        scale(i)=1.0d+0
      enddo
c
c Remove fix variables and free rows
c
      do i=1,n
        vartyp(i)=0
        if(abs(ubound(i)-lbound(i)).le.tplus*(abs(lbound(i)+1.0d0)))then
          vartyp(i)= 1
          vcstat(i)=-2-1
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            rhs(rowidx(j))=rhs(rowidx(j))-ubound(i)*nonzeros(j)
          enddo
          addobj=addobj+obj(i)*lbound(i)
        else
          vcstat(i)=0
        endif
      enddo
      do i=1,m
        slktyp(i)=0
        j=i+n
        if((ubound(j).gt.lbig).and.(lbound(j).lt.-lbig))then
          vcstat(j)=-2-1
        else
          vcstat(j)=0
        endif
      enddo
c
c   p r e s o l v e r
c
      call timer(k)
      if(premet.gt.0)then
        write(buff,'(1x)')
        call mprnt(buff)
        write(buff,'(1x,a)')'Process: presolv'
        call mprnt(buff)
        call presol(colpnt,rowidx,nonzeros,rindex,nonzeros(nz+1),
     x  snhead,snhead(n1),nodtyp,nodtyp(n1),vcstat,vcstat(n1),
     x  ecolpnt,count,ecolpnt(n1),count(n1),
C CMSSW: Prevent REAL*8 reusage warning
C Was:  vartyp,dxsn(n1),dxs(n1),diag(n1),odiag(n1),
     x  vartyp,dxsn(n1),dxs(n1),pmbig,ppbig,
     x  ubound,lbound,ubound(n1),lbound(n1),rhs,obj,prehis,prelen,
C CMSSW: Prevent REAL*8 reusage warning
C Was:  addobj,big,pivots,invprm,dv,ddv,dxsn,dxs,diag,odiag,premet,code)
     x  addobj,big,pivots,invprm,dv,ddv,dxsn,dxs,dmbig,dpbig,premet,
     x  code)
        write(buff,'(1x,a)')'Presolv done...'
        call mprnt(buff)
        if(code.ne.0)goto 45
      endif
c
c Remove lower bounds
c
      call stndrd(ubound,lbound,rhs,obj,nonzeros,
     x vartyp,slktyp,vcstat,colpnt,rowidx,addobj,tplus,tzer,lbig,big)
c
c Scaling before aggregator
c
      i=iand(scalmet,255)
      j=iand(scpass,255)
      if(i.gt.0)call mscale(colpnt,rowidx,nonzeros,obj,rhs,ubound,
     x vcstat,scale,upinf,i,j,scdiff,ddsup,dxsn,dxs,snhead)
c
c Aggregator
c
      if(premet.gt.127)then
        write(buff,'(1x)')
        call mprnt(buff)
        write(buff,'(1x,a)')'Process: aggregator'
        call mprnt(buff)
        call aggreg(colpnt,rowidx,nonzeros,rindex,
     x  vcstat,vcstat(n1),ecolpnt,count,ecolpnt(n1),count(n1),
     x  rhs,obj,prehis,prelen,pivots,vartyp,slktyp,invprm,snhead,
     x  nodtyp,inta1,inta1(n1),dv,addobj,premet,code)
        write(buff,'(1x,a)')'Aggregator done...'
        call mprnt(buff)
        if(code.ne.0)goto 55
      endif
c
c Scaling after aggregator
c
      i=scalmet/256
      j=scpass/256
      if(i.gt.0)call mscale(colpnt,rowidx,nonzeros,obj,rhs,
     x ubound,vcstat,scale,upinf,i,j,scdiff,ddsup,dxsn,dxs,snhead)
c
      call timer(j)
      write(buff,'(1x)')
      call mprnt(buff)
      write(buff,'(1x,a,f8.2,a)')
     x 'Time for presolv, scaling and aggregator: ',0.01*(j-k),' sec.'
      call mprnt(buff)
c
c cleaning
c
      do i=1,mn
        xs(i)=0.0d+0
        dspr(i)=0.0d+0
        dsup(i)=0.0d+0
        up(i)=0.0d+0
      enddo
      do i=1,m
        dv(i)=0.0d+0
      enddo
c
c Is the problem solved ?
c
      fixn=0
      dropn=0
      freen=0
      do i=1,n
        if(vcstat(i).le.-2)then
          fixn=fixn+1
        else if(vartyp(i).eq.0) then
          freen=freen+1
        endif
      enddo
      do i=1,m
        if(vcstat(i+n).le.-2)dropn=dropn+1
      enddo
      active=mn-fixn-dropn
      if(active.eq.0)code=2
      if(code.gt.0)then
        opt=addobj
        write(buff,'(1x,a)')'Problem is solved by the pre-solver'
        call mprnt(buff)
        if(code.gt.0)goto 55
        goto 50
      endif
c
c Presolve statistics
c
      if(premet.gt.0)then
        i=0
        j=0
        do k=1,n
          if(vcstat(k).gt.-2)then
            i=i+count(k)-ecolpnt(k)+1
            if(j.lt.count(k)-ecolpnt(k)+1)j=count(k)-ecolpnt(k)+1
          endif
        enddo
        write(buff,'(1x,a22,i8)')'Number of rows       :',(m-dropn)
        call mprnt(buff)
        write(buff,'(1x,a22,i8)')'Number of columns    :',(n-fixn)
        call mprnt(buff)
        write(buff,'(1x,a22,i8)')'Free variables       :',freen
        call mprnt(buff)
        write(buff,'(1x,a22,i8)')'No. of nonzeros      :',i
        call mprnt(buff)
        write(buff,'(1x,a22,i8)')'Longest column count :',j
        call mprnt(buff)
      endif
c
c Incrase rowidx by n
c
      j=colpnt(1)
      k=colpnt(n+1)-1
      do i=j,k
        rowidx(i)=rowidx(i)+n
      enddo
      active=mn-fixn-dropn
c
c Normalize obj and rhs
c
      if(objnor.gt.tzer)then
        call scalobj(obj,scobj,vcstat,objnor)
      endif
      if(rhsnor.gt.tzer)then
        call scalrhs(rhs,scrhs,vcstat,rhsnor,ubound,xs,up)
      endif
c
c Calling phas12
c
      sol=scobj*scrhs
      i=mn+mn
      call timer(k)
      call phas12(
     x obj,rhs,ubound,diag,odiag,xs,dxs,dxsn,up,dspr,ddspr,
     x ddsprn,dsup,ddsup,ddsupn,dv,ddv,ddvn,nonzeros,prinf,upinf,duinf,
     x vartyp,slktyp,colpnt,ecolpnt,count,vcstat,pivots,invprm,
     x snhead,nodtyp,inta1,rowidx,rindex,
C CMSSW: Prevent REAL*8 reusage warning
C Was: dxs,dxsn,ddspr,ddsprn,ddsup,ddsupn,
     x dxs,iwork1,iwork2,iwork3,iwork4,iwork5,
     x code,opt,iter,corect,fixn,dropn,active,fnzmax,fnzmin,addobj,
     x sol,ft,i)
      call timer(j)
      write(buff,'(1x,a,f11.2,a)')'Solver time ',0.01*(j-k),' sec.'
      call mprnt(buff)
c
c Decrease rowidx by n
c
      j=colpnt(1)
      k=colpnt(n+1)-1
      do i=j,k
        rowidx(i)=rowidx(i)-n
      enddo
c
c Rescaling
c
  55  do i=1,m
        rhs(i)=rhs(i)*scrhs*scale(i+n)
        ubound(i+n)=ubound(i+n)*scrhs*scale(i+n)
        xs(i+n)=xs(i+n)*scrhs*scale(i+n)
        up(i+n)=up(i+n)*scrhs*scale(i+n)
        dv(i)=dv(i)*scobj/scale(i+n)
        dspr(i+n)=dspr(i+n)/scale(i+n)*scobj
        dsup(i+n)=dsup(i+n)/scale(i+n)*scobj
      enddo
c
      do i=1,n
        obj(i)=obj(i)*scobj*scale(i)
        ubound(i)=ubound(i)*scrhs/scale(i)
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do j=pnt1,pnt2
          nonzeros(j)=nonzeros(j)*scale(i)*scale(rowidx(j)+n)
        enddo
c
        xs(i)=xs(i)/scale(i)*scrhs
        up(i)=up(i)/scale(i)*scrhs
        dspr(i)=dspr(i)*scale(i)*scobj
        dsup(i)=dsup(i)*scale(i)*scobj
      enddo
c
c Postprocessing
c
  45  call pstsol(colpnt,rowidx,nonzeros,vcstat,vcstat(n1),
     x vartyp,slktyp,ubound,lbound,ubound(n1),lbound(n1),rhs,obj,xs,
     x inta1,ddvn,prehis,prelen,big)
c
  50  return
      end
c
c ===========================================================================
c
      subroutine stndrd(ubound,lbound,rhs,obj,nonzeros,
     x vartyp,slktyp,vcstat,colpnt,rowidx,addobj,tplus,tzer,lbig,big)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 vartyp(n),slktyp(m),vcstat(mn),colpnt(n1),rowidx(nz)
      real*8 ubound(mn),lbound(mn),rhs(m),obj(n),nonzeros(nz),
     x addobj,tplus,tzer,lbig,big
c
      integer*4 i,j,k,pnt1,pnt2
c
c generate standard form, row modification
c
      k=0
      do 150 i=1,m
        j=i+n
        if(vcstat(j).gt.-2)then
          if(abs(ubound(j)-lbound(j)).le.tplus*(abs(lbound(j))+1d0))then
            slktyp(i)=0
            ubound(j)=0.0d+00
            rhs(i)=rhs(i)+lbound(j)
            goto 150
          endif
ccc          if((ubound(j).gt.lbig).and.(lbound(j).lt.-lbig))then
ccc            vcstat(j)=-2
ccc            slktyp(i)=0
ccc            goto 150
ccc          endif
          if(lbound(j).lt.-lbig)then
            slktyp(i)=2
            lbound(j)=-ubound(j)
            ubound(j)=big
            rhs(i)=-rhs(i)
            k=k+1
          else
            slktyp(i)=1
          endif
          rhs(i)=rhs(i)+lbound(j)
          ubound(j)=ubound(j)-lbound(j)
          if(ubound(j).lt.lbig)slktyp(i)=-slktyp(i)
        else
          slktyp(i)=0
        endif
 150  continue
c
c negate reverse rows
c
      if(k.gt.0)then
        do i=1,n
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if(abs(slktyp(rowidx(j))).ge.2)nonzeros(j)=-nonzeros(j)
          enddo
        enddo
      endif
c
c column modification
c
      do 155 i=1,n
        if(vcstat(i).gt.-2)then
ccc          if(abs(ubound(i)-lbound(i)).le.tplus*(abs(lbound(i))+1d0))then
ccc            vcstat(i)=-2
ccc            vartyp(i)= 1
ccc            do j=colpnt(i),colpnt(i+1)-1
ccc              rhs(rowidx(j))=rhs(rowidx(j))-nonzeros(j)*lbound(i)
ccc            enddo
ccc            addobj=addobj+obj(i)*lbound(i)
ccc            goto 155
ccc          endif
          if((ubound(i).gt.lbig).and.(lbound(i).lt.-lbig))then
            vartyp(i)=0
            goto 155
          endif
          if(lbound(i).lt.-lbig)then
            vartyp(i)=2
            lbound(i)=-ubound(i)
            ubound(i)=big
            obj(i)=-obj(i)
            do j=colpnt(i),colpnt(i+1)-1
              nonzeros(j)=-nonzeros(j)
            enddo
          else
            vartyp(i)=1
          endif
          if(abs(lbound(i)).gt.tzer)then
            if(ubound(i).lt.lbig)ubound(i)=ubound(i)-lbound(i)
            do j=colpnt(i),colpnt(i+1)-1
              rhs(rowidx(j))=rhs(rowidx(j))-nonzeros(j)*lbound(i)
            enddo
            addobj=addobj+obj(i)*lbound(i)
          endif
          if(ubound(i).lt.lbig)vartyp(i)=-vartyp(i)
        endif
 155  continue
      return
      end
c
c ===========================================================================
c Primal-dual method with supernodal cholesky factorization
c               Version 2.11 (1996 December)
c  Written by Cs. Meszaros, MTA SzTAKI, Budapest, Hungary
c           e-mail: meszaros@lutra.sztaki.hu
c                      see "bpmain.f"
c
c code=-2 General memory limit (no solution)
c code=-1 Memory limit during iterations
c code= 0
c code= 1 No optimum
c code= 2 Otimal solution
c code= 3 Primal Infeasible
c code= 4 Dual Infeasible
c
c ===========================================================================
c
      subroutine phas12(
     x obj,rhs,bounds,diag,odiag,xs,dxs,dxsn,up,dspr,ddspr,
     x ddsprn,dsup,ddsup,ddsupn,dv,ddv,ddvn,nonzeros,prinf,upinf,duinf,
     x vartyp,slktyp,colpnt,ecolpnt,count,vcstat,pivots,invprm,
     x snhead,nodtyp,inta1,rowidx,rindex,
     x rwork1,iwork1,iwork2,iwork3,iwork4,iwork5,
     x code,opt,iter,corect,fixn,dropn,active,fnzmax,fnzmin,addobj,
     x scobj,factim,mn2)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/mscal/ varadd,slkadd,scfree
      real*8        varadd,slkadd,scfree
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      common/param/ palpha,dalpha
      real*8        palpha,dalpha
c
      common/factor/ tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8         tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
c
      common/toler/ tsdir,topt1,topt2,tfeas1,tfeas2,feas1,feas2,
     x              pinfs,dinfs,inftol,maxiter
      real*8        tsdir,topt1,topt2,tfeas1,tfeas2,feas1,feas2,
     x              pinfs,dinfs,inftol
      integer*4     maxiter
c
      common/initv/ prmin,upmax,dumin,stamet,safmet,premet,regul
      real*8        prmin,upmax,dumin
      integer*4     stamet,safmet,premet,regul
c
      integer*4 fixn,dropn,active,code,iter,corect,fnzmin,fnzmax,mn2
      real*8  addobj,scobj,opt
c
      common/predp/ ccstop,barset,bargrw,barmin,mincor,maxcor,inibar
      real*8        ccstop,barset,bargrw,barmin
      integer*4     mincor,maxcor,inibar
c
      common/predc/ target,tsmall,tlarge,center,corstp,mincc,maxcc
      real*8        target,tsmall,tlarge,center,corstp
      integer*4     mincc,maxcc

      common/itref/ tresx,tresy,maxref
      real*8        tresx,tresy
      integer*4     maxref
c
      real*8 obj(n),rhs(m),bounds(mn),diag(mn),odiag(mn),xs(mn),
     x dxs(mn),dxsn(mn),up(mn),dspr(mn),ddspr(mn),ddsprn(mn),dsup(mn),
     x ddsup(mn),ddsupn(mn),dv(m),ddv(m),ddvn(m),nonzeros(cfree),
     x prinf(m),upinf(mn),duinf(mn),rwork1(mn)

      integer*4 vartyp(n),slktyp(m),colpnt(n1),ecolpnt(mn),count(mn),
     x vcstat(mn),pivots(mn),invprm(mn),snhead(mn),nodtyp(mn),
     x inta1(mn),rowidx(cfree),rindex(rfree),factim,
     x iwork1(mn2),iwork2(mn2),iwork3(mn2),iwork4(mn2),iwork5(mn2)
c
c ---------------------------------------------------------------------------
c
      integer*4 i,j,err,factyp,pphase,dphase,t1,t2,opphas,odphas
      real*8 pinf,dinf,uinf,prelinf,drelinf,popt,dopt,cgap,
     x prstpl,dustpl,barpar,oper,maxstp,pinfrd,dinfrd,objerr,nonopt,
     x oprelinf,odrelinf,opinf,odinf,ocgap
      integer*4 corr,corrc,barn,fxp,fxd,fxu,nropt
      character*99 buff,sbuff
      character*1 wmark
c
c to save parameters
c
      integer*4 maxcco,mxrefo
      real*8 lamo,spdeno,bargro,topto
C CMSSW: Temporary integer array needed to avoid reusing REAL*8 for
C integer storage
      integer*4 inta12(mn)
c
c --------------------------------------------------------------------------
c
 101  format(1x,' ')
 102  format(1x,'It-PC   P.Inf   D.Inf  U.Inf   Actions           ',
     x 'P.Obj           D.Obj  Barpar')
 103  format(1x,'------------------------------------------------',
     x '------------------------------')
 104  format(1x,I2,a1,I1,I1,' ',1PD7.1,' ',1PD7.1,' ',1PD6.0,
     x ' ',I2,' ',I3,' ',I3,' ',1PD15.8,' ',1PD15.8,' ',1PD6.0)
c
c Saving parameters
c
      maxcco=maxcc
      mxrefo=maxref
      lamo=lam
      spdeno=supdens
      bargro=bargrw
      topto=topt1
c
c Include dummy ranges if requested
c
      if(regul.gt.0)then
        do i=1,m
          if(slktyp(i).eq.0)then
            slktyp(i)=-1
            bounds(i+n)=0.0d+0
          endif
        enddo
      endif
c
c Other initialization
c
      nropt=0
      factim=0
      wmark='-'
      fxp=0
      fxd=0
      fxu=0
c
      call stlamb(colpnt,vcstat,rowidx,inta1,fixn,dropn,factyp)
      call timer(t1)
      j=0
      do i=1,n
        if((vcstat(i).gt.-2).and.(vartyp(i).eq.0))j=j+1
      enddo
      if((j.gt.0).and.(scfree.lt.tzer))factyp=1
c
c Initial scaling matrix (diagonal)
c
      call fscale (vcstat,diag,odiag,vartyp,slktyp)
      do i=1,m
        dv(i)=0.0d+0
      enddo

ccc      i=2*rfree
ccc      j=400
ccc      call paintmat(m,n,nz,i,rowidx,colpnt,rindex,j,'matrix01.pic')

c
c Initial factorization
c
      fnzmax=0
      if(factyp.eq.1)then
        call ffactor(ecolpnt,vcstat,colpnt,rowidx,
     x  iwork4,pivots,count,nonzeros,diag,
     x  iwork1,iwork1(mn+1),iwork2,iwork2(mn+1),inta1,iwork5,
     x  iwork5(mn+1),iwork3,iwork3(mn+1),iwork4(mn+1),rindex,
     x  rwork1,fixn,dropn,fnzmax,fnzmin,active,oper,xs,slktyp,code)
        if(code.ne.0)goto 999
        call supnode(ecolpnt,count,rowidx,vcstat,pivots,snhead,
     x  invprm,nodtyp)
      else
c
c minimum local fill-in ordering
c
        i=int(tfind)
        if(order.lt.1.5)i=0
        if(order.lt.0.5)i=-1
        call symmfo(inta1,pivots,ecolpnt,vcstat,
     x  colpnt,rowidx,nodtyp,rindex,iwork3,invprm,
     x  count,snhead,iwork1,iwork1(mn+1),iwork2,iwork2(mn+1),
     x  iwork4,iwork4(mn+1),iwork3(mn+1),iwork5,iwork5(mn+1),
C CMSSW: Prevent REAL*8 reusage warning
C Was:  nonzeros,fnzmax,oper,i,rwork1,code
     x  nonzeros,fnzmax,oper,i,inta12,code)
        if(code.ne.0)goto 999
        call supnode(ecolpnt,count,rowidx,vcstat,pivots,snhead,
     x  invprm,nodtyp)
        popt=trabs
        trabs=tabs
        call nfactor(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,err,rwork1,iwork2,iwork2(mn+1),dropn,slktyp,
     x  snhead,iwork3,invprm,nodtyp,dv,odiag)
        trabs=popt
      endif
      fnzmin=fnzmax
c
c Compute centrality and iterative refinement power
c
      if(fnzmin.eq.0)fnzmin=1
      cgap=oper/fnzmin/10.0d+0
      j=0
  78  if(cgap.ge.1.0d+0)then
        cgap=cgap/2
        j=j+1
        goto 78
      endif
      if(j.eq.0)j=1
      if(maxcc.le.0d+0)then
        maxcc=-maxcc
      else
        if(j.le.maxcc)maxcc=j
      endif
      if(mincc.gt.maxcc)maxcc=mincc
      cgap=log(1.0d+0+oper/fnzmin/5.0d+0)/log(2.0d+00)
      if(maxref.le.0)then
        maxref=-maxref
      else
        maxref=int(cgap*maxref)
      endif
      if(maxref.le.0)maxref=0
      write(buff,'(1x,a,i2)')'Centrality correction Power:',maxcc
      call mprnt(buff)
      write(buff,'(1x,a,i2)')'Iterative refinement  Power:',maxref
      call mprnt(buff)
c
c Starting point
c
      call initsol(xs,up,dv,dspr,dsup,rhs,obj,bounds,vartyp,slktyp,
     x vcstat,colpnt,ecolpnt,pivots,rowidx,nonzeros,diag,rwork1,
     x count)
      call timer(t2)
c
      write(buff,'(1x,a,f12.2,a)')'FIRSTFACTOR TIME :',
     x (dble(t2-t1)*0.01d+0),' sec'
      call mprnt(buff)
c
      maxstp=1.0d+0
      iter=0
      corect=0
      corr=0
      corrc=0
      barn=0
      cgap=0.0d+0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            cgap=cgap+xs(i)*dspr(i)
            barn=barn+1
          endif
          if(j.lt.0)then
            cgap=cgap+up(i)*dsup(i)
            barn=barn+1
          endif
        endif
      enddo
      if(barn.lt.1)barn=1

ccc      i=2*rfree
ccc      j=350
ccc      call paintaat(mn,nz,pivotn,i,rowidx,ecolpnt,count,rindex,
ccc     x j,pivots,iwork1,iwork1(mn+1),iwork2,iwork2(mn+1),iwork3,
ccc     x iwork3(mn+1),'normal01.pic')

ccc      i=2*rfree
ccc      j=400
ccc      call paintata(mn,nz,pivotn,i,rowidx,ecolpnt,count,rindex,
ccc     x j,pivots,iwork1,iwork1(mn+1),iwork2,iwork2(mn+1),iwork3,
ccc     x 'atapat01.pic')


ccc      i=2*rfree
ccc      j=350
ccc      err=nz
ccc      call paintfct(mn,cfree,pivotn,i,rowidx,ecolpnt,count,rindex,
ccc     x j,pivots,iwork2,err,'factor01.pic')
c
c Initialize for the iteration loop
c
      do i=1,n
        if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
          if(xs(i).gt.dspr(i))then
            vcstat(i)=1
          else
            vcstat(i)=0
          endif
        endif
      enddo
      do i=1,m
        if((vcstat(i+n).gt.-2).and.(slktyp(i).ne.0))then
          if(xs(i+n).gt.dspr(i+n))then
            vcstat(i+n)=1
          else
            vcstat(i+n)=0
          endif
        endif
      enddo
      opphas=0
      odphas=0
      pinfrd=1.0d+0
      dinfrd=1.0d+0
      barpar=0.0d+0
c
c main iteration loop
c
  10  if(mod(iter,20).eq.0)then
        write(buff,101)
        call mprnt(buff)
        write(buff,102)
        call mprnt(buff)
        write(buff,103)
        call mprnt(buff)
      endif
c
c Infeasibilities
c
      call cprinf(xs,prinf,slktyp,colpnt,rowidx,nonzeros,
     x rhs,vcstat,pinf)
      call cduinf(dv,dspr,dsup,duinf,vartyp,slktyp,colpnt,rowidx,
     x nonzeros,obj,vcstat,dinf)
      call cupinf(xs,up,upinf,bounds,vartyp,slktyp,vcstat,
     x uinf)
c
c Objectives
c
      call cpdobj(popt,dopt,obj,rhs,bounds,xs,dv,dsup,
     x vcstat,vartyp,slktyp)
      popt=scobj*popt+addobj
      dopt=scobj*dopt+addobj
c
c Stopping criteria
c
      call stpcrt(prelinf,drelinf,popt,dopt,cgap,iter,
     x code,pphase,dphase,maxstp,pinf,uinf,dinf,
     x prinf,upinf,duinf,nonopt,pinfrd,dinfrd,
     x prstpl,dustpl,obj,rhs,bounds,xs,dxs,dspr,ddspr,dsup,ddsup,dv,ddv,
     x up,addobj,scobj,vcstat,vartyp,slktyp,
     x oprelinf,odrelinf,opinf,odinf,ocgap,opphas,odphas,sbuff)
c
      write(buff,104)iter,wmark,corr,corrc,pinf,dinf,uinf,fxp,fxd,fxu,
     x popt,dopt,barpar
      call mprnt(buff)
      if(code.ne.0)then
        write(buff,'(1x)')
        call mprnt(buff)
        call mprnt(sbuff)
        goto 90
      endif
c
c P-D solution modification
c
      call pdmodi(xs,dspr,vcstat,vartyp,slktyp,cgap,popt,
     x dopt,prinf,duinf,upinf,colpnt,rowidx,nonzeros,pinf,uinf,dinf)
c
c Fixing variables / dropping rows / handling dual slacks
c
      i=fixn
      call varfix(vartyp,slktyp,rhs,colpnt,rowidx,nonzeros,
     x xs,up,dspr,dsup,vcstat,fixn,dropn,addobj,scobj,obj,bounds,
     x duinf,dinf,fxp,fxd,fxu)
      if(fixn.ne.i)then
        call supupd(pivots,invprm,snhead,nodtyp,vcstat,ecolpnt)
        call cprinf(xs,prinf,slktyp,colpnt,rowidx,nonzeros,
     x  rhs,vcstat,pinf)
        call cupinf(xs,up,upinf,bounds,vartyp,slktyp,vcstat,
     x  uinf)
      endif
c
c Compute gap
c
      cgap=0.0d+0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            cgap=cgap+xs(i)*dspr(i)
            if(j.lt.0)then
              cgap=cgap+up(i)*dsup(i)
            endif
          endif
        endif
      enddo
c
c Computation of the scaling matrix
c
      objerr=abs(dopt-popt)/(abs(popt)+1.0d+0)
      call cdiag(xs,up,dspr,dsup,vartyp,slktyp,vcstat,diag,odiag)
      pinfrd=pinf
      dinfrd=dinf
c
c The actual factorization
c
  50  err=0
      call timer(t1)
      if (factyp.eq.1) then
        call mfactor(ecolpnt,vcstat,colpnt,rowidx,pivots,
     x  count,iwork4,nonzeros,diag,err,rwork1,iwork2,iwork2(mn+1),
     x  dropn,slktyp,snhead,iwork3,invprm,nodtyp,dv,odiag)
      else
        call nfactor(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,err,rwork1,iwork2,iwork2(mn+1),dropn,slktyp,
     x  snhead,iwork3,invprm,nodtyp,dv,odiag)
      endif
      call timer(t2)
      if(err.gt.0)then
        do i=1,mn
          diag(i)=odiag(i)
        enddo
        call newsmf(colpnt,pivots,rowidx,nonzeros,ecolpnt,count,
     x  vcstat,invprm,snhead,nodtyp,iwork1,rwork1,iwork2,iwork3,
     x  iwork4,code)
        if(code.lt.0)then
          write(buff,'(1x)')
          call mprnt(buff)
          goto 90
        endif
        goto 50
      endif
      factim=factim+t2-t1
c
c We are in the finish ?
c
      wmark(1:1)='-'
      if(objerr.gt.1.0d+0)objerr=1.0d+0
      if(objerr.lt.topt1)objerr=topt1
      if((objerr.le.topt1*10.0d+0).and.(pphase+dphase.eq.4))then
         if(bargrw.gt.0.1d+0)bargrw=0.1d+0
         nropt=nropt+1
         if(nropt.eq.5)then
           nropt=0
           topt1=topt1*sqrt(10.d+0)
           write(buff,'(1x,a)')'Near otptimal but slow convergence.'
           call mprnt(buff)
         endif
         wmark(1:1)='+'
      endif
c
c primal-dual predictor-corrector direction
c
      call  cpdpcd(xs,up,dspr,dsup,prinf,duinf,upinf,
     x dxsn,ddvn,ddsprn,ddsupn,dxs,ddv,ddspr,ddsup,bounds,
     x ecolpnt,count,pivots,vcstat,diag,odiag,rowidx,nonzeros,
     x colpnt,vartyp,slktyp,barpar,corr,prstpl,dustpl,barn,cgap)
      corect=corect+corr
c
c primal-dual centality-correction
c
      call  cpdccd(xs,up,dspr,dsup,upinf,
     x dxsn,ddvn,ddsprn,ddsupn,dxs,ddv,ddspr,ddsup,bounds,
     x ecolpnt,count,pivots,vcstat,diag,odiag,rowidx,nonzeros,
     x colpnt,vartyp,slktyp,barpar,corrc,prstpl,dustpl)
      corect=corect+corrc
c
c compute steplengths
c
      iter=iter+1
      prstpl=prstpl*palpha
      dustpl=dustpl*dalpha
c
c compute the new primal-dual solution
c
      call cnewpd(prstpl,xs,dxs,up,upinf,dustpl,dv,ddv,dspr,
     x ddspr,dsup,ddsup,vartyp,slktyp,vcstat,maxstp)
c
c End main loop
c
      goto 10
c
 90   opt=(dopt-popt)/(abs(popt)+1.0d+0)
      write(buff,'(1x,a,1PD11.4,a,1PD18.10)')
     x 'ABSOLUTE infeas.   Primal  :',pinf,   '    Dual         :',dinf
      call mprnt(buff)
      write(buff,'(1x,a,1PD11.4,a,1PD18.10)')
     x 'PRIMAL :  Relative infeas. :',prelinf,'    Objective    :',popt
      call mprnt(buff)
      write(buff,'(1x,a,1PD11.4,a,1PD18.10)')
     x 'DUAL   :  Relative infeas. :',drelinf,'    Objective    :',dopt
      call mprnt(buff)
      write(buff,'(1x,a,1PD11.4,a,1PD18.10)')
     x 'Complementarity gap        :',cgap,'    Duality gap  :',opt
      call mprnt(buff)
      opt=popt
c
c Restoring parameters
c
 999  maxcc=maxcco
      maxref=mxrefo
      lam=lamo
      supdens=spdeno
      bargrw=bargro
      topt1=topto
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine mscale(colpnt,rowidx,nonzeros,
     x obj,rhs,ubound,vcstat,scale,scalen,scalmet,scpass,scdiff,
     x ddsup,ddsupn,dxs,snhead)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 colpnt(n1),rowidx(nz),vcstat(mn),
     x scalmet,scpass,snhead(mn)
      real*8    nonzeros(cfree),obj(n),rhs(m),ubound(mn),scale(mn),
     x scalen(mn),scdiff,ddsup(mn),ddsupn(mn),dxs(mn)
c
      integer*4 i
      character*99 buff
c
      write(buff,'(1x)')
      call mprnt(buff)
      write(buff,'(1x,a)')'Process: scaling'
      call mprnt(buff)
c
      do i=1,mn
        scalen(i)=1.0d+0
      enddo
c
      if((scalmet.eq.2).or.(scalmet.eq.4))then
        call scale1(ubound,nonzeros,colpnt,obj,scalen,vcstat,
     x  rowidx,rhs,ddsup,scpass,scdiff,snhead,nonzeros(nz+1))
      endif
      if((scalmet.eq.3).or.(scalmet.eq.5))then
        call scale2(ubound,nonzeros,colpnt,obj,scalen,vcstat,
     x  rowidx,rhs,scpass,scdiff,ddsup,ddsupn,dxs,snhead)
      endif
      if((scalmet.gt.0).and.(scalmet.le.3))then
        call sccol2(ubound,nonzeros,colpnt,obj,scalen,
     x  vcstat,rowidx)
        call scrow2(rhs,ubound,nonzeros,rowidx,colpnt,ddsup,
     x  scalen,vcstat)
      endif
c
      do i=1,mn
        scale(i)=scale(i)*scalen(i)
      enddo
c
      write(buff,'(1x,a)')'Scaling done...'
      call mprnt(buff)
      return
      end
c
c ============================================================================
c
      subroutine scale1(bounds,rownzs,colpnt,obj,scale,
     x vcstat,rowidx,rhs,work1,scpass,scdif,veclen,
     x lognz)

      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/numer/ tplus,tzer
      real*8        tplus,tzer

      real*8 bounds(mn),rownzs(cfree),obj(n),scale(mn),
     x  rhs(m),work1(mn),scdif,lognz(nz)
      integer*4 rowidx(cfree),colpnt(n1),vcstat(mn),scpass,veclen(mn)
c
      real*8 defic,odefic
      integer*4 pass,i,j,pnt1,pnt2,nonz
      character*99 buff
c
      pass=0
      nonz=0
      defic= 1.0d+0
      odefic=0.0d+0
      do i=1,mn
        veclen(i)=0
      enddo
      do i=1,n
        if(vcstat(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if((abs(rownzs(j)).gt.tzer).and.
     x      (vcstat(rowidx(j)+n).gt.-2))then
              lognz(j)=log(abs(rownzs(j)))
              veclen(i)=veclen(i)+1
              veclen(rowidx(j)+n)=veclen(rowidx(j)+n)+1
              nonz=nonz+1
              odefic=odefic+abs(lognz(j))
            else
              lognz(j)=0.0d+0
            endif
          enddo
        endif
      enddo
      do i=1,mn
        if(veclen(i).eq.0)veclen(i)=1
        scale(i)=0.0d+0
      enddo
      if(nonz.eq.0)goto 999
      odefic=exp(odefic/dble(nonz))
      if(odefic.le.scdif)goto 999
  10  write(buff,'(1x,a,i2,a,d12.6)')'Pass',pass,'. Average def.',odefic
      call mprnt(buff)
      call sccol1(colpnt,scale,
     x vcstat,rowidx,veclen,lognz)
      pass=pass+1
      call scrow1(rowidx,colpnt,work1,scale,vcstat,defic,veclen,lognz)
      defic=exp(defic/dble(nonz))
      if(defic.le.scdif)goto 999
      if(pass.ge.scpass)goto 999
      if(odefic.le.defic)goto 999
      odefic=defic
      goto 10
 999  write(buff,'(1x,a,i2,a,d12.6)')'Pass',pass,'. Average def.',defic
      call mprnt(buff)
c
c Scaling
c
      do i=1,mn
        scale(i)=exp(scale(i))
      enddo
      do i=1,n
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do j=pnt1,pnt2
          rownzs(j)=rownzs(j)/scale(i)/scale(rowidx(j)+n)
        enddo
        obj(i)=obj(i)/scale(i)
        bounds(i)=bounds(i)*scale(i)
      enddo
      do i=1,m
        rhs(i)=rhs(i)/scale(i+n)
        bounds(i+n)=bounds(i+n)/scale(i+n)
      enddo
      return
      end
c
c ============================================================================
c
      subroutine scrow1(rowidx,colpnt,
     x maxi,scale,excld,ss,veclen,lognz)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 lognz(nz),maxi(mn),scale(mn),ss
      integer*4 rowidx(cfree),colpnt(n1),excld(mn),veclen(mn)
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c ---------------------------------------------------------------------------
      integer*4 i,j,pnt1,pnt2
      real*8 sol
c ---------------------------------------------------------------------------
      ss=0
      do i=1,m
        maxi(i)=0.0d+0
      enddo
      do i=1,n
        if(excld(i).gt.-2)then
          sol=scale(i)
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if(excld(rowidx(j)+n).gt.-2)then
              maxi(rowidx(j))=maxi(rowidx(j))+lognz(j)-sol
              ss=ss+abs(lognz(j)-sol-scale(rowidx(j)+n))
            endif
          enddo
        endif
      enddo
      do i=1,m
        scale(n+i)=maxi(i)/veclen(i+n)
      enddo
      return
      end
c
c ===========================================================================
c
      subroutine sccol1(colpnt,scale,
     x excld,rowidx,veclen,lognz)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 scale(mn),lognz(nz)
      integer*4 colpnt(n1),excld(mn),rowidx(cfree),veclen(mn)
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c ---------------------------------------------------------------------------
      integer*4 i,j,pnt1,pnt2
      real*8 ma
c ---------------------------------------------------------------------------
      do i=1,n
        ma=0.0d+0
        if(excld(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            ma=ma+lognz(j)-scale(rowidx(j)+n)
          enddo
          scale(i)=ma/veclen(i)
        endif
      enddo
      return
      end
c
c ===========================================================================
c
      subroutine scrow2(rhs,bounds,rownzs,rowidx,
     x colpnt,maxi,scale,excld)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      real*8 rownzs(cfree),bounds(mn),rhs(m),maxi(m),scale(mn)
      integer*4 rowidx(cfree),colpnt(n1),excld(mn)
c ---------------------------------------------------------------------------
      integer*4 i,j,pnt1,pnt2,k
      real*8 sol
c ---------------------------------------------------------------------------
      do i=1,m
        maxi(i)=0
      enddo
      do i=1,n
        if(excld(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            k=rowidx(j)
            sol=abs(rownzs(j))
            if (maxi(k).lt.sol)maxi(k)=sol
          enddo
        endif
      enddo
      do i=1,m
        if(maxi(i).le.tzer)maxi(i)=1.0d+0
        scale(n+i)=maxi(i)*scale(n+i)
        rhs(i)=rhs(i)/maxi(i)
        bounds(i+n)=bounds(i+n)/maxi(i)
      enddo
      do i=1,n
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do j=pnt1,pnt2
          k=rowidx(j)
          rownzs(j)=rownzs(j)/maxi(k)
        enddo
      enddo
      return
      end
c
c ===========================================================================

c
      subroutine sccol2(bounds,rownzs,colpnt,obj,scale,
     x excld,rowidx)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 rownzs(cfree),bounds(mn),obj(n),scale(mn)
      integer*4 colpnt(n1),excld(mn),rowidx(cfree)
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c ---------------------------------------------------------------------------
      integer*4 i,j,pnt1,pnt2
      real*8 sol,ma
c ---------------------------------------------------------------------------
      do i=1,n
        if(excld(i).gt.-2)then
          ma=0
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if(excld(rowidx(j)+n).gt.-2)then
              sol=abs(rownzs(j))
              if (ma.lt.sol)ma=sol
            endif
          enddo
          if (ma.le.tzer)ma=1.0d+0
          scale(i)=ma*scale(i)
          do j=pnt1,pnt2
            rownzs(j)=rownzs(j)/ma
          enddo
          obj(i)=obj(i)/ma
          bounds(i)=bounds(i)*ma
        endif
      enddo
      return
      end
c
c ===========================================================================
c
      subroutine scalobj(obj,scobj,excld,objnor)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 obj(n),scobj,objnor
      integer*4 excld(n),i
      character*99 buff
c ---------------------------------------------------------------------------
      scobj=0.0d+0
      do i=1,n
        if(excld(i).gt.-2)then
          if (abs(obj(i)).gt.scobj)scobj=abs(obj(i))
        endif
      enddo
      scobj=scobj/objnor
      if(scobj.lt.1.0d-08)scobj=1.0d-08
      write(buff,'(1x,a,d8.2)')'Obj. scaled ',scobj
      call mprnt(buff)
      do i=1,n
        obj(i)=obj(i)/scobj
      enddo
      return
      end
c
c ===========================================================================
c
      subroutine scalrhs(rhs,scrhs,excld,rhsnor,bounds,xs,up )
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 rhs(m),scrhs,rhsnor,bounds(mn),xs(mn),up(mn)
      integer*4 excld(mn),i
      character*99 buff
c ---------------------------------------------------------------------------
      scrhs=0.0d+0
      do i=1,m
        if(excld(i+n).gt.-2)then
          if(abs(rhs(i)).gt.scrhs)scrhs=abs(rhs(i))
        endif
      enddo
      scrhs=scrhs/rhsnor
      if(scrhs.lt.1.0d-08)scrhs=1.0d-08
      write(buff,'(1x,a,d8.2)')'Rhs. scaled ',scrhs
      call mprnt(buff)
      do i=1,m
        rhs(i)=rhs(i)/scrhs
      enddo
      do i=1,mn
        bounds(i)=bounds(i)/scrhs
        xs(i)=xs(i)/scrhs
        up(i)=up(i)/scrhs
      enddo
      return
      end
c
c ============================================================================
c Curtis-Reid Scaling algorithm
c ============================================================================
c
      subroutine scale2(bounds,rownzs,colpnt,obj,sc,
     x  vcstat,rowidx,rhs,scpass,scdif,scm1,rk,logsum,count)

      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      real*8 bounds(mn),rownzs(cfree),obj(n),sc(mn),
     x  rhs(m),scdif,scm1(mn),rk(mn),logsum(mn)
      integer*4 rowidx(cfree),colpnt(n1),vcstat(mn),scpass,count(mn)
c
      integer*4 i,j,in,pnt1,pnt2,pass
      real*8 logdef,s,qk,qkm1,ek,ekm1,ekm2,sk,skm1
      character*99 buff
c
      pass=0
      do i=1,mn
       count(i)=0
       logsum(i)=0.0d+0
      enddo
      logdef=0.0d+0
      in=0
      do i=1,n
        if(vcstat(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if(vcstat(rowidx(j)+n).gt.-2)then
              if(abs(rownzs(j)).gt.tzer)then
                s=log(abs(rownzs(j)))
                count(rowidx(j)+n)=count(rowidx(j)+n)+1
                count(i)=count(i)+1
                logsum(i)=logsum(i)+s
                logsum(rowidx(j)+n)=logsum(rowidx(j)+n)+s
                logdef=logdef+s*s
                in=in+1
              endif
            endif
          enddo
        endif
      enddo
      do i=1,mn
       if((vcstat(i).le.-2).or.(count(i).eq.0))count(i)=1
      enddo
      logdef=sqrt(logdef)/dble(in)
      logdef=exp(logdef)
      write(buff,'(1x,a,i2,a,d12.6)')'Pass',pass,'. Average def.',logdef
      call mprnt(buff)
      if(logdef.le.scdif)then
        do i=1,mn
          sc(i)=1.0d+0
        enddo
        goto 999
      endif
c
c Initialize
c
      do i=1,m
        sc(i+n)=logsum(i+n)/count(i+n)
        rk(i+n)=0
      enddo
      sk=0
      do i=1,n
        if(vcstat(i).gt.-2)then
          s=logsum(i)
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            s=s-logsum(rowidx(j)+n)/count(rowidx(j)+n)
          enddo
        else
          s=0
        endif
        rk(i)=s
        sk=sk+s*s/count(i)
        sc(i)=0.0d+0
      enddo
      do i=1,mn
        scm1(i)=sc(i)
      enddo
      ekm1=0
      ek=0
      qk=1.0d+0
c
c Curtis-Reid scaling
c
  10  pass=pass+1
        do i=1,m
          rk(i+n)=ek*rk(i+n)
        enddo
        do i=1,n
          if(vcstat(i).gt.-2)then
            pnt1=colpnt(i)
            pnt2=colpnt(i+1)-1
            s=rk(i)/count(i)
            do j=pnt1,pnt2
              if(vcstat(rowidx(j)+n).gt.-2)
     x        rk(rowidx(j)+n)=rk(rowidx(j)+n)+s
            enddo
          endif
        enddo
        skm1=sk
        sk=0.0d+0
        do i=1,m
          rk(i+n)=-rk(i+n)/qk
          sk=sk+rk(i+n)*rk(i+n)/count(i+n)
        enddo
        ekm2=ekm1
        ekm1=ek
        ek=qk*sk/skm1
        qkm1=qk
        qk=1-ek
        if(pass.gt.scpass)goto 20
c
c Update Column-scale factors
c
        do i=1,n
          if(vcstat(i).gt.-2)then
            s=sc(i)
            sc(i)=s+(rk(i)/count(i)+ekm1*ekm2*(s-scm1(i)))/qk/qkm1
            scm1(i)=s
          endif
        enddo
c
c even pass
c
        do i=1,n
          if(vcstat(i).gt.-2)then
            s=ek*rk(i)
            pnt1=colpnt(i)
            pnt2=colpnt(i+1)-1
            do j=pnt1,pnt2
              if(vcstat(rowidx(j)+n).gt.-2)
     x        s=s+rk(rowidx(j)+n)/count(rowidx(j)+n)
            enddo
            s=-s/qk
          else
            s=0
          endif
          rk(i)=s
        enddo
        skm1=sk
        sk=0.0d+0
        do i=1,n
          sk=sk+rk(i)*rk(i)/count(i)
        enddo
        ekm2=ekm1
        ekm1=ek
        ek=qk*sk/skm1
        qkm1=qk
        qk=1-ek
c
c Update Row-scale factors
c
        do i=1,m
          j=i+n
          if(vcstat(j).gt.-2)then
            s=sc(j)
            sc(j)=s+(rk(j)/count(j)+ekm1*ekm2*(s-scm1(j)))/qk/qkm1
            scm1(j)=s
          endif
        enddo
      goto 10
c
c Syncronize Column factors
c
  20  do i=1,n
        if(vcstat(i).gt.-2)then
          sc(i)=sc(i)+(rk(i)/count(i)+ekm1*ekm2*(sc(i)-scm1(i)))/qkm1
        endif
      enddo
c
c Scaling
c
      logdef=0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          sc(i)=exp(sc(i))
        else
          sc(i)=1.0d+0
        endif
      enddo
      do i=1,n
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do j=pnt1,pnt2
          rownzs(j)=rownzs(j)/sc(i)/sc(rowidx(j)+n)
          if((vcstat(rowidx(j)+n).gt.-2).and.
     x      (abs(rownzs(j)).gt.tzer))then
            s=log(abs(rownzs(j)))
            logdef=logdef+s*s
          endif
        enddo
        obj(i)=obj(i)/sc(i)
        bounds(i)=bounds(i)*sc(i)
      enddo
      do i=1,m
        rhs(i)=rhs(i)/sc(i+n)
        bounds(i+n)=bounds(i+n)/sc(i+n)
      enddo
      logdef=sqrt(logdef)/dble(in)
      logdef=exp(logdef)
      pass=pass-1
      write(buff,'(1x,a,i2,a,d12.6)')'Pass',pass,'. Average def.',logdef
      call mprnt(buff)
 999  return
      end
c
c ============================================================================
c ===========================================================================
c
      subroutine stlamb(colpnt,vcstat,rowidx,cnt,fixn,dropn,p)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 colpnt(n1),vcstat(mn),rowidx(nz),cnt(mn),
     x          fixn,dropn,p
c
      common/factor/ tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8         tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
c
      common/setden/ maxdense,densgap,setlam,denslen
      real*8         maxdense,densgap
      integer*4      setlam,denslen
c
      integer*4 i,j,pnt1,pnt2,cn,lcn,lcd,ndn,z,maxcn
      real*8    la
      character*99 buff
c
c ---------------------------------------------------------------------------
c

C CMSSW: Explicit initialization needed
      ndn=0

      write(buff,'(1X)')
      call mprnt(buff)
      do i=1,m
        cnt(i)=0
      enddo
      if((m-dropn).ge.(n-fixn))then
        cnt(1)=m-dropn-n+fixn
      endif
      maxcn=0
      do i=1,n
        if(vcstat(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          cn=0
          do j=pnt1,pnt2
            if(vcstat(rowidx(j)).gt.-2)cn=cn+1
          enddo
          if(cn.gt.0)cnt(cn)=cnt(cn)+1
          vcstat(i)=cn
          if(maxcn.lt.cn)maxcn=cn
        endif
      enddo
      if(setlam.lt.0)goto 70
c
      cn =maxcn
      lcd=maxcn
      lcn=maxcn
      z=0
C CMSSW: Explicit integer conversion needed
      pnt1=int((n-fixn+m-dropn)*maxdense)
      pnt2=0
      if((m-dropn).ge.1.5*(n-fixn))then
        maxdense=1.0
      endif
      if((m-dropn).ge.2.5*(n-fixn))then
        lcn=1
        lcd=2
        goto 60
      endif
c
      do while ((pnt2.le.pnt1).and.(cn.gt.0))
        if(cnt(cn).eq.0)then
          z=z+1
        else
          if(z.gt.0)then
            if((densgap*cn*cn).le.(cn+z+1)*(cn+z+1))then
              lcd=cn+z+1
              lcn=cn
              ndn=pnt2
            endif
            z=0
          endif
          pnt2=pnt2+cnt(cn)
        endif
        cn=cn-1
      enddo
c
  60  write(buff,'(1X,A,I6)')'Largest sparse column length :',lcn
      call mprnt(buff)
      if((maxcn.le.denslen).or.(lcn.eq.maxcn))then
        write(buff,'(1X,A)')'Problem has no dense columns'
        call mprnt(buff)
        lcn=maxcn
      else
        write(buff,'(1X,A,I6)')'Smallest dense column length :',lcd
        call mprnt(buff)
        write(buff,'(1X,A,I6)')'Number of dense columns      :',ndn
        call mprnt(buff)
      endif
      la=lcn+0.5
      la=la/m
      write(buff,'(1X,A,F7.4)')'Computed density parameter   : ',la
      call mprnt(buff)
      if(la.gt.lam)then
        lam=la
      else
        write(buff,'(1X,A,F7.4)') 'Parameter reset to value    : ',lam
        call mprnt(buff)
      endif
  70  lam=lam*m
      p=1
      if((lam.ge.maxcn).and.(setlam.le.0))p=2
      if(supdens.le.lam)supdens=lam
c
      write(buff,'(1X)')
      call mprnt(buff)
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine symmfo(inta1,pivots,ecolpnt,vcstat,
     x colpnt,rowidx,rowpnt,colindex,perm,invperm,
     x count,inta2,inta3,inta4,inta5,inta6,inta7,inta8,inta9,
     x inta10,inta11,nonzeros,l,oper,tfind,inta12,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 inta1(mn),ecolpnt(mn),pivots(mn),vcstat(mn),
     x colpnt(n1),rowidx(cfree),rowpnt(mn),colindex(rfree),
     x perm(mn),invperm(mn),count(mn),inta2(mn),inta3(mn),inta4(mn),
     x inta5(mn),inta6(mn),inta7(mn),inta8(mn),inta9(mn),inta10(mn),
     x inta11(mn),tfind,inta12(mn),l,code
c
      real*8   nonzeros(cfree),oper
c
      integer*4 i,j,k,t1,tt1,t2,p1,p2,pnt,pnt1,pnt2,aatnz
      character*99 buff
c
c ---------------------------------------------------------------------------
c
    1 format(1x,'Building aat                 time:',f9.2,' sec')
    2 format(1x,'Building ordering    list    time:',f9.2,' sec')
    4 format(1x,'Symbolic factorisation       time:',f9.2,' sec')
    5 format(1x,'Total symbolic phase         time:',f9.2,' sec')
    6 format(1x,'Sub-diagonal nonzeros in aat     :',i9)
    7 format(1x,'Sub-diagonal nonzeros in L       :',i9)
    8 format(1x,'NONZEROS         :',i12)
    9 format(1x,'OPERATIONS       :',f13.0)
   10 format(1x,'Minimum Local Fill-in Ordering with Power:',i3)
   11 format(1x,'Minimum Degree Ordering (Power=0)')
   12 format(1x,'Without Ordering')
c
      call timer (tt1)
      if(tfind.lt.0)then
        write(buff,12)
      else if(tfind.eq.0)then
        write(buff,11)
      else
        write(buff,10)tfind
      endif
      oper=0.0d+0
      call mprnt(buff)
      do i=1,nz
        rowidx(i)=rowidx(i)-n
      enddo
      if(rfree.lt.nz)then
        write(buff,'(1x,a)')'Not enough integer memory'
        call mprnt(buff)
        code=-2
        goto 999
      endif
      if(cfree.lt.2*nz)then
        write(buff,'(1x,a)')'Not enough real memory'
        call mprnt(buff)
        code=-2
        goto 999
      endif
c
c If no ordering...
c
      if(tfind.lt.0)then
        t2=tt1
        do i=1,m
          perm(i)=i
        enddo
        goto 50
      endif
c
c Otherwise...
c
      do i=1,n
        inta2(i)=i
      enddo
      call transps(n,m,nz,colpnt,rowidx,nonzeros,
     x rowpnt,colindex,nonzeros(nz+1),inta2)
      k=1
      l=m
      do i=1,m
        pivots(i)=0
        if(vcstat(i+n).le.-2)then
          invperm(l)=i
          l=l-1
        else
          invperm(k)=i
          k=k+1
        endif
      enddo
      call transps(m,n,nz,rowpnt,colindex,nonzeros(nz+1),
     x colpnt,rowidx,nonzeros,invperm)
      do i=1,n
        p1=colpnt(i)
        if(vcstat(i).le.-2)then
          p2=colpnt(i)-1
        else
          p2=colpnt(i+1)-1
  19      if((p1.le.p2).and.(vcstat(rowidx(p2)+n).le.-2))then
            p2=p2-1
            goto 19
          endif
        endif
        perm(i)=p1
        invperm(i)=p2
      enddo
c
      pnt=nz+1
      do i=1,m
        if(pnt+mn.gt.cfree)then
          write(buff,'(1x,a)')'Not enough real memory'
          call mprnt(buff)
          code=-2
          goto 999
        endif
        pivots(i)=1
        if(vcstat(i+n).gt.-2)then
          ecolpnt(i)=pnt
          pnt1=rowpnt(i)
          pnt2=rowpnt(i+1)-1
          do j=pnt1,pnt2
            k=colindex(j)
            if(vcstat(k).gt.-2)then
              p1=perm(k)
              p2=invperm(k)
              perm(k)=perm(k)+1
              do l=p1,p2
                if(pivots(rowidx(l)).eq.0)then
                  pivots(rowidx(l))=1
                  rowidx(pnt)=rowidx(l)
                  pnt=pnt+1
                endif
              enddo
            endif
          enddo
          count(i)=pnt-ecolpnt(i)
          do j=ecolpnt(i),pnt-1
            pivots(rowidx(j))=0
          enddo
        endif
      enddo
      aatnz=pnt-nz-1
c
c
      call timer (t2)
      write(buff,1)dble(t2-tt1)/100.0d+0
      call mprnt(buff)
c
c call minimum fill-in ordering
c
      call genmfo(m,mn,nz,cfree,rfree,pivotn,
     x ecolpnt,count,perm,rowpnt,vcstat(n+1),rowidx,
     x invperm,inta1,inta2,inta3,inta4,inta5,inta6,inta7,
     x inta8,inta9,inta10,inta11,colindex,tfind,inta12,pivots,code)
      if(code.lt.0)goto 999
c
c
 50   call timer(t1)
      write(buff,2)dble(t1-t2)/100.0d+0
      call mprnt(buff)
c
      pivotn=0
      do 30 i=1,n
        ecolpnt(i)=colpnt(i)
        count(i)=colpnt(i+1)-1
        inta2(i)=i
        if(vcstat(i).le.-2)goto 30
        pivotn=pivotn+1
        pivots(pivotn)=i
  30  continue
c
      call transps(n,m,nz,colpnt,rowidx,nonzeros,
     x rowpnt,colindex,nonzeros(nz+1),inta2)
c
      k=1
      l=m
      do 40 i=1,m
        j=perm(i)
        if(vcstat(j+n).le.-2)then
          invperm(l)=j
          l=l-1
        else
          pivotn=pivotn+1
          pivots(pivotn)=j+n
          invperm(k)=j
          k=k+1
        endif
  40  continue
c
      call transps(m,n,nz,rowpnt,colindex,nonzeros(nz+1),
     x colpnt,rowidx,nonzeros,invperm)
c
      do 20 i=1,nz
        rowidx(i)=rowidx(i)+n
  20  continue
c
      do i=1,n
        if(vcstat(i).gt.-2)then
          k=ecolpnt(i)
          l=count(i)
  35      if((l.ge.k).and.(vcstat(rowidx(l)).le.-2))then
            l=l-1
            goto 35
          endif
          count(i)=l
        endif
      enddo
c
      call symfact(pivots,rowidx,ecolpnt,count,vcstat,
     x perm,invperm,inta2,inta1,l,code)
      if(code.lt.0)goto 999
      call timer(t2)
      write(buff,4)dble(t2-t1)/100.0d+0
      call mprnt(buff)
      if(tfind.ge.0)then
        write(buff,6)aatnz
        call mprnt(buff)
      endif
      write(buff,7)l
      call mprnt(buff)
c
      do 55 i=1,mn
        inta1(i)=0
  55  continue
      l=0
      do 60 i=1,pivotn
        j=pivots(pivotn-i+1)
        k=count(j)-ecolpnt(j)+1
        if(k.eq.0)goto 60
        l=l+k
        inta1(j)=inta1(rowidx(ecolpnt(j)))+k
        oper=oper+(dble(k)*dble(k)+dble(k))/2.0d+0
  60  continue
      call timer(t1)
      write(buff,5)dble(t2-tt1)/100.0d+0
      call mprnt(buff)
      write(buff,8)l
      call mprnt(buff)
      write(buff,9)oper
      call mprnt(buff)
c
 999  return
      end
c
c ===========================================================================
c Minimum local fill-in ordering
c
c ===========================================================================
c
      subroutine genmfo(m,mn,nz,cfree,rfree,pivotn,
     x pntc,ccol,permut,pntr,crow,rowidx,
     x mark,cpermf,cpermb,rpermf,rpermb,cfill,rfill,cpnt,
     x cnext,cprew,suplst,fillin,colidx,tfind,noddeg,supdeg,code)
c
      integer*4 m,mn,nz,cfree,rfree,pivotn,rowidx(cfree),colidx(rfree),
     x permut(m),cpermf(m),cpermb(m),rpermf(m),rpermb(m),
     x ccol(m),crow(m),pntc(m),pntr(m),mark(m),cfill(m),cpnt(m),
     x cnext(m),cprew(m),rfill(m),suplst(m),fillin(m),tfind,
     x noddeg(m),supdeg(m),code
      character*99 buff
c
c ---------------------------------------------------------------------------
c INPUT PARAMETERS
c
c  m       number of rows
c  mn      an number greather than m
c  nz      last used position of the column file
c  cfree   length of the column file (column file is used from nz+1 to cfree)
c  rfree   length of the row file (row file is used from 1 to rfree)
c  rowidx  column file (containing the lower tiriangular part of AAT)
c  colidx  row file
c  pntc    pointer to the columns of the lower diagonal of AAT
c  ccol    column lengths of AAT
c  crow    if crow(i)<-1 row i is removed from the ordering
c  tfind   search loop,  tfind=0 gives the minimum degree ordering
c          suggested value tfind=25
c
c
c OUTPUT PARAMETERS
c permut   the ordering
c pivotn   Number of ordered nodes
c
c
c Others: Integer working arrays of size m
c
c
c --------------------------------------------------------------------------
      integer*4 pnt,pnt1,pnt2,i,j,k,l,o,p,endmem,ccfree,rcfree,pmode,
     x rfirst,rlast,cfirst,clast,pcol,pcnt,ppnt1,ppnt2,fill,prewcol,
     x ii,mm,mfill,supnd,hsupnd,oo,nnz,fnd,oldpcol,q,fl
c---------------------------------------------------------------------------
c
   1  format(' NOT ENOUGH MEMORY IN THE ROW    FILE ')
   2  format(' NOT ENOUGH MEMORY IN THE COLUMN FILE ')
   3  format(' Analyse for supernodes in aat    :',i9,' col')
   4  format(' Final supernodal columns disabled:',i9,' col')
   5  format(' Hidden supernodal columns        :',i9,' col')

C CMSSW: Explicit initialization needed
      clast=0
c
c initialization
c
      code=0
      endmem=cfree
      pivotn=0
      pmode =0
      do i=1,m
        permut(i)=0
        suplst(i)=0
        fillin(i)=-1
        supdeg(i)=1
        if(crow(i).gt.-2)then
          crow(i)=0
        endif
      enddo
c
c Compute crow
c
      do 10 i=1,m
        if(crow(i).le.-2)goto 10
        pnt1=pntc(i)
        pnt2=pnt1+ccol(i)-1
        do j=pnt1,pnt2
          crow(rowidx(j))=crow(rowidx(j))+1
        enddo
        clast=i
  10  continue
      cpermf(clast)=0
      ccfree=cfree-pntc(clast)-ccol(clast)
      if(ccfree.lt.mn)then
        write(buff,2)
        call mprnt(buff)
        code=-2
        goto 999
      endif
c
c create pointers to colidx
c
      do i=1,m
        cprew(i)=0
      enddo
      pnt=1
      do i=1,m
        if(crow(i).ge.0)then
          pntr(i)=pnt
          rfill(i)=pnt
          pnt=pnt+crow(i)
        endif
      enddo
      rcfree=rfree-pnt
      if(rcfree.lt.mn)then
        write(buff,1)
        call mprnt(buff)
        code=-2
        goto 999
      endif
c
c create the row file : symbolical transps the matrix, set up noddeg
c
      do i=1,m
        noddeg(i)=ccol(i)+crow(i)
        if(crow(i).ge.0)then
          pnt1=pntc(i)
          pnt2=pnt1+ccol(i)-1
          do j=pnt1,pnt2
            k=rowidx(j)
            colidx(rfill(k))=i
            rfill(k)=rfill(k)+1
          enddo
        endif
      enddo
c
c Search supernodes
c
      hsupnd=0
      supnd=0
      do i=1,m
        if(crow(i).ge.0)then
          pnt1=pntr(i)
          pnt2=pnt1+crow(i)-1
          do j=pnt1,pnt2
            mark(colidx(j))=i
          enddo
          mark(i)=i
          pnt1=pntc(i)
          pnt2=pnt1+ccol(i)-1
          do j=pnt1,pnt2
            mark(rowidx(j))=i
          enddo
          p=ccol(i)+crow(i)
 118      if (pnt1.le.pnt2)then
            o=rowidx(pnt1)
            call chknod(m,cfree,rfree,i,o,p,ccol,crow,mark,pntc,
     x      pntr,rowidx,colidx,supdeg,suplst,ii)
            supnd=supnd+ii
            pnt1=pnt1-ii
            pnt2=pnt2-ii
            pnt1=pnt1+1
            goto 118
          endif
        endif
      enddo
      write(buff,3)supnd
      call mprnt(buff)
c
c Set up lists
c
      do i=1,m
        mark(i)=0
        cpnt(i)=0
        cnext(i)=0
      enddo
      cfirst=0
      clast=0
      rfirst=0
      rlast=0
      mm=0
      do i=1,m
        if(crow(i).ge.0)then
          mm=mm+1
          if(cfirst.eq.0)then
            cfirst=i
          else
            cpermf(clast)=i
          endif
          cpermb(i)=clast
          clast=i
c
          if(rfirst.eq.0)then
            rfirst=i
          else
            rpermf(rlast)=i
          endif
          rpermb(i)=rlast
          rlast=i
c
          j=noddeg(i)-supdeg(i)+2
          if(j.gt.0)then
            o=cpnt(j)
            cnext(i)=o
            cpnt(j)=i
            if(o.ne.0)cprew(o)=i
          endif
          cprew(i)=0
        endif
      enddo
      cpermf(clast)=0
      rpermf(rlast)=0
      pcol=0
c
c loop for pivots
c
  50  oldpcol=pcol
      pcol=0
      nnz=1
      if(oldpcol.eq.0)goto 9114
c
c Find supernodal pivot
c
      mfill=0
      k=pntc(oldpcol)
      l=k+ccol(oldpcol)-1
      oo=ccol(oldpcol)-1
9125  if(k.gt.l)goto 9114
      j=rowidx(k)
      if(crow(j)+ccol(j).eq.oo)then
        hsupnd=hsupnd+1
        pcol=j
        goto 9200
      endif
      k=k+1
      goto 9125
c
c Find another pivot
c
9114  pmode=0
      fnd=0
      mfill=-1
9110  j=cpnt(nnz)
      if((j.gt.0).and.(pmode.eq.0))then
        pmode=nnz
        if(tfind.eq.0)then
          pcol=j
          mfill=1
          goto 9200
        endif
      endif
9120  if(j.le.0)goto 9150
      if(fillin(j).ge.0)then
        fill=fillin(j)
        goto 9175
      endif
c
c set up mark and cfill
c
      q=0
      fill=0
      k=pntc(j)
      l=k+ccol(j)-1
      p=0
      do o=k,l
        q=q+1
        cfill(q)=rowidx(o)
        mark(rowidx(o))=supdeg(rowidx(o))
        fill=fill-(supdeg(rowidx(o))*(supdeg(rowidx(o))-1))/2
      enddo
      k=pntr(j)
      l=k+crow(j)-1
      do o=k,l
        q=q+1
        cfill(q)=colidx(o)
        mark(colidx(o))=supdeg(colidx(o))
        fill=fill-(supdeg(colidx(o))*(supdeg(colidx(o))-1))/2
      enddo
c
c compute fill-in
c
      fill=fill+((noddeg(j)-supdeg(j))*(noddeg(j)-supdeg(j)+1))/2
      do p=1,q
        fl=0
        o=cfill(p)
        k=pntc(o)
        l=k+ccol(o)-1
        do oo=k,l
          fl=fl+mark(rowidx(oo))
        enddo
        fill=fill-supdeg(o)*fl
      enddo
c
c administration
c
      do o=1,q
        mark(cfill(o))=0
      enddo
c
c Test
c
      fillin(j)=fill
9175  if(mfill.lt.0)mfill=fill+1
      if(fill.lt.mfill)then
        mfill=fill
        pcol=j
      endif
      fnd=fnd+1
      if((fnd.gt.tfind).or.(mfill.eq.0))goto 9200
      j=cnext(j)
      goto 9120
c
c next bunch
c
9150  nnz=nnz+1
      if(nnz.le.m)goto 9110
9200  if (pcol.eq.0)goto 900
      endmem=cfree
      ccfree=cfree-pntc(clast)-ccol(clast)
      rcfree=rfree-pntr(rlast)-crow(rlast)
c
c compress column file
c
      if(ccfree.lt.mn)then
       call mccmpr(mn,cfree,ccfree,endmem,nz,
     x  pntc,ccol,cfirst,cpermf,rowidx,code)
       if(code.lt.0)goto 999
      endif
c
c remove pcol from the cpermf lists
c
      prewcol=cpermb(pcol)
      o=cpermf(pcol)
      if(prewcol.ne.0)then
        cpermf(prewcol)=o
      else
        cfirst=o
      endif
      if(o.eq.0)then
        clast=prewcol
      else
        cpermb(o)=prewcol
      endif
c
c remove pcol from the rpermf lists
c
      prewcol=rpermb(pcol)
      o=rpermf(pcol)
      if(prewcol.ne.0)then
        rpermf(prewcol)=o
      else
        rfirst=o
      endif
      if(o.eq.0)then
        rlast=prewcol
      else
        rpermb(o)=prewcol
      endif
c
c administration
c
      pivotn=pivotn+1
      permut(pivotn)=pcol
      pcnt=ccol(pcol)+crow(pcol)
c
c remove pcol from the counter lists
c
      o=cnext(pcol)
      ii=cprew(pcol)
      if(ii.eq.0)then
        cpnt(noddeg(pcol)-supdeg(pcol)+2)=o
      else
        cnext(ii)=o
      endif
      if(o.ne.0)cprew(o)=ii
c
      ppnt1=endmem-pcnt
      ppnt2=ppnt1+pcnt-1
      endmem=endmem-pcnt
      ccfree=ccfree-pcnt
      pnt=ppnt1
c
c create pivot column from the row file
c
      pnt1=pntr(pcol)
      pnt2=pnt1+crow(pcol)-1
      do 70 i=pnt1,pnt2
        o=colidx(i)
        l=pntc(o)
        p=l+ccol(o)-1
c
c find element and move in the column o
c
        cfill(o)=ccol(o)-1
        rfill(o)= 0
        do 75 k=l,p
          if(rowidx(k).eq.pcol)then
            mark(o)=1
            rowidx(pnt)=o
            pnt=pnt+1
            rowidx(k)=rowidx(p)
            goto 70
          endif
  75    continue
  70  continue
      mm=pnt
c
c extend pivot column from the column file
c
      pnt1=pntc(pcol)
      pnt2=pnt1+ccol(pcol)-1
      do 60 j=pnt1,pnt2
        o=rowidx(j)
        mark(o)=1
        rowidx(pnt)=o
        pnt=pnt+1
c
c remove pcol from the row file
c
        rfill(o)=-1
        cfill(o)=ccol(o)
        l=pntr(o)
        p=l+crow(o)-2
        do 55 k=l,p
          if(colidx(k).eq.pcol)then
            colidx(k)=colidx(p+1)
            goto 60
          endif
  55    continue
  60  continue
      pntc(pcol)=ppnt1
      ccol(pcol)=pcnt
c
c remove columns from the counter lists
c
      do 77 j=ppnt1,ppnt2
        i=rowidx(j)
        o=cnext(i)
        ii=cprew(i)
        if(ii.eq.0)then
          cpnt(noddeg(i)-supdeg(i)+2)=o
        else
          cnext(ii)=o
        endif
        if(o.ne.0)cprew(o)=ii
  77  continue
c
c elimination loop
c
      if(mfill.gt.0)then
c
        if(ppnt1.lt.mm)call hpsort((mm-ppnt1),rowidx(ppnt1))
        if(mm.lt.ppnt2)call hpsort((ppnt2-mm+1),rowidx(mm))
c
        do 80 p=ppnt1,ppnt2
          i=rowidx(p)
c
c delete element from mark
c
          mark(i)=0
          pcnt=pcnt-1
c
c transformation on the column i
c
          fill=pcnt
          pnt1=pntc(i)
          pnt2=pnt1+cfill(i)-1
          do 90 k=pnt1,pnt2
             o=rowidx(k)
             if(mark(o).ne.0)then
               fill=fill-1
               mark(o)=0
             endif
  90      continue
c
c compute the free space
c
          ii=cpermf(i)
          if(ii.eq.0)then
            k=endmem-pnt2-1
          else
            k=pntc(ii)-pnt2-1
          endif
c
c move column to the end of the column file
c
          if(fill.gt.k)then
            if (ccfree.lt.mn)then
              call mccmpr(mn,cfree,ccfree,endmem,nz,
     x        pntc,ccol,cfirst,cpermf,rowidx,code)
              if(code.lt.0)goto 999
              pnt1=pntc(i)
              pnt2=pnt1+cfill(i)-1
            endif
            if(i.ne.clast)then
              l=pntc(clast)+ccol(clast)
              pntc(i)=l
              do 95 k=pnt1,pnt2
                rowidx(l)=rowidx(k)
                l=l+1
  95          continue
              pnt1=pntc(i)
              pnt2=l-1
              prewcol=cpermb(i)
              if(prewcol.eq.0)then
                cfirst=ii
              else
                cpermf(prewcol)=ii
              endif
              cpermb(ii)=prewcol
              cpermf(clast)=i
              cpermb(i)=clast
              clast=i
              cpermf(clast)=0
            endif
          endif
c
c create fill in
c
          do 97 k=p+1,ppnt2
            o=rowidx(k)
            if(mark(o).eq.0)then
              mark(o)=1
            else
              pnt2=pnt2+1
              rowidx(pnt2)=o
              rfill(o)=rfill(o)+1
            endif
   97     continue
          pnt2=pnt2+1
          ccol(i)=pnt2-pnt1
          if(i.eq.clast)then
            ccfree=endmem-pnt2-1
          endif
  80    continue
      else
        do p=ppnt1,ppnt2
          i=rowidx(p)
          ccol(i)=ccol(i)-1-rfill(i)
          mark(i)=0
        enddo
      endif
c
c make space for fills in the row file
c
      do 100 j=ppnt1,ppnt2
        i=rowidx(j)
        if(mfill.eq.0)goto 135
        pnt2=pntr(i)+crow(i)-1
c
c compute the free space
c
        ii=rpermf(i)
        if(ii.eq.0)then
          k=rfree-pnt2-1
        else
          k=pntr(ii)-pnt2-1
        endif
c
c move row to the end of the row file
c
        if(k.lt.rfill(i))then
          if(rcfree.lt.mn)then
            call rcomprs(mn,rfree,
     x      rcfree,pntr,crow,rfirst,rpermf,colidx,code)
            if(code.lt.0)goto 999
          endif
          if(ii.ne.0)then
            pnt1=pntr(i)
            pnt2=pnt1+crow(i)-1
            pnt=pntr(rlast)+crow(rlast)
            pntr(i)=pnt
            do 110 l=pnt1,pnt2
              colidx(pnt)=colidx(l)
              pnt=pnt+1
 110        continue
c
c update the rperm lists
c
            prewcol=rpermb(i)
            if(prewcol.eq.0)then
              rfirst=ii
            else
              rpermf(prewcol)=ii
            endif
            rpermb(ii)=prewcol
            rpermf(rlast)=i
            rpermb(i)=rlast
            rlast=i
            rpermf(rlast)=0
          endif
        endif
 135    crow(i)=crow(i)+rfill(i)
        if(i.eq.rlast)rcfree=rfree-crow(i)-pntr(i)
        noddeg(i)=noddeg(i)-supdeg(pcol)
 100  continue
      if(mfill.eq.0)goto 150
c
c make pointers to the end of the filled rows
c
      do 120 j=ppnt1,ppnt2
        rfill(rowidx(j))=pntr(rowidx(j))+crow(rowidx(j))-1
 120  continue
c
c generate fill-in in the row file, update noddeg
c
      do j=ppnt1,ppnt2
        o=rowidx(j)
        pnt1=pntc(o)+cfill(o)
        pnt2=pntc(o)+ccol(o)-1
        do k=pnt1,pnt2
          colidx(rfill(rowidx(k)))=o
          rfill(rowidx(k))=rfill(rowidx(k))-1
          noddeg(o)=noddeg(o)+supdeg(rowidx(k))
          noddeg(rowidx(k))=noddeg(rowidx(k))+supdeg(o)
        enddo
      enddo
c
c Indicate new fill-in computation
c
      if(tfind.gt.0)then
        do j=ppnt1,ppnt2
          i=rowidx(j)
          fillin(i)=-1
          pnt1=pntc(i)+cfill(i)
          pnt2=pntc(i)+ccol(i)-1
          do pnt=pnt1,pnt2
            ii=rowidx(pnt)
            if(rfill(ii).ge.0)then
              k=pntc(ii)
              l=k+ccol(ii)-1
              do o=k,l
                fillin(rowidx(o))=-1
              enddo
              k=pntr(ii)
              l=k+crow(ii)-1
              do o=k,l
                fillin(colidx(o))=-1
              enddo
              rfill(ii)=-1
            endif
          enddo
        enddo
      endif
c
c Searching for new supernodes
c
 150  l=0
      j=ppnt1
 151  if(j.le.ppnt2)then
        i=rowidx(j)
        p=ccol(i)+crow(i)
c
        pnt1=pntc(i)
        pnt2=pnt1+ccol(i)-1
        do k=pnt1,pnt2
          if(mark(rowidx(k)).eq.0)then
            l=l+1
            cfill(l)=rowidx(k)
          endif
          mark(rowidx(k))=i
        enddo
c
        if(mark(i).eq.0)then
          l=l+1
          cfill(l)=i
        endif
        mark(i)=i
c
        pnt1=pntr(i)
        pnt2=pnt1+crow(i)-1
        do k=pnt1,pnt2
          if(mark(colidx(k)).eq.0)then
            l=l+1
            cfill(l)=colidx(k)
          endif
          mark(colidx(k))=i
        enddo
c
        k=j+1
  152   if(k.le.ppnt2)then
          o=rowidx(k)
          call chknod(m,cfree,rfree,i,o,p,ccol,crow,mark,pntc,
     x    pntr,rowidx,colidx,supdeg,suplst,ii)
          if(ii.gt.0)then
            supnd=supnd+1
c
            prewcol=cpermb(o)
            oo=cpermf(o)
            if(prewcol.ne.0)then
              cpermf(prewcol)=oo
            else
              cfirst=oo
            endif
            if(oo.eq.0)then
              clast=prewcol
            else
              cpermb(oo)=prewcol
            endif
c
            prewcol=rpermb(o)
            oo=rpermf(o)
            if(prewcol.ne.0)then
              rpermf(prewcol)=oo
            else
              rfirst=oo
            endif
            if(oo.eq.0)then
              rlast=prewcol
            else
              rpermb(oo)=prewcol
            endif
c
            rowidx(k)=rowidx(ppnt2)
            k=k-1
            ppnt2=ppnt2-1
            ccol(pcol)=ccol(pcol)-1
          endif
          k=k+1
          goto 152
        endif
        j=j+1
        goto 151
      endif
      do i=1,l
        mark(cfill(i))=0
      enddo
c
c update the counter lists
c
      do j=ppnt1,ppnt2
        i=rowidx(j)
        fill=noddeg(i)-supdeg(i)+2
        o=cpnt(fill)
        cnext(i)=o
        cpnt(fill)=i
        if(o.ne.0)cprew(o)=i
        cprew(i)=0
      enddo
c
c Augment the permutation with the supernodes
c
      i=suplst(pcol)
 155  if(i.gt.0)then
        pivotn=pivotn+1
        permut(pivotn)=i
        i=suplst(i)
        goto 155
      endif
      goto 50
c
c Augment the permutation with the disabled rows
c
 900  do i=1,m
        if(crow(i).le.-2)then
          pivotn=pivotn+1
          permut(pivotn)=i
        endif
      enddo
      write(buff,4)supnd
      call mprnt(buff)
      write(buff,5)hsupnd
      call mprnt(buff)
c
c Ready
c
 999  return
      end
c
c ===========================================================================
c
      subroutine mccmpr(mn,cfree,ccfree,endmem,nz,
     x pnt,count,cfirst,cpermf,rowidx,code)
      integer*4 mn,cfree,ccfree,endmem,nz,pnt(mn),rowidx(cfree),
     x count(mn),cpermf(mn),cfirst,code
c
      integer*4 i,j,pnt1,pnt2,pnt0
      character*99 buff
c ---------------------------------------------------------------------------
   2  format(' NOT ENOUGH MEMORY DETECTED IN SUBROUTINE CCOMPRESS')
      pnt0=nz+1
      i=cfirst
  40  if(i.le.0)goto 30
        pnt1=pnt(i)
        if(pnt1.lt.pnt0)goto 10
        if(pnt1.eq.pnt0)then
          pnt0=pnt0+count(i)
          goto 10
        endif
        pnt(i)=pnt0
        pnt2=pnt1+count(i)-1
        do 20 j=pnt1,pnt2
          rowidx(pnt0)=rowidx(j)
          pnt0=pnt0+1
  20    continue
  10    i=cpermf(i)
      goto 40
  30  ccfree=endmem-pnt0-1
      if(ccfree.lt.mn)then
        write(buff,2)
        call mprnt(buff)
        code=-2
      endif
      return
      end
c
c ===========================================================================
c
      subroutine chknod(m,cfree,rfree,i,o,p,ccol,crow,mark,pntc,
     x pntr,rowidx,colidx,supdeg,suplst,fnd)
c
      integer*4 m,cfree,rfree,i,o,p,ccol(m),crow(m),mark(m),pntc(m),
     x pntr(m),rowidx(cfree),colidx(rfree),supdeg(m),suplst(m),fnd
c
      integer*4 ppnt1,ppnt2,k,l,pnt,ii,pnod
c
      fnd=0
      if(ccol(o)+crow(o).ne.p)goto 120
      ppnt1=pntr(o)
      ppnt2=ppnt1+crow(o)-1
 111  if(ppnt1.le.ppnt2)then
        if(mark(colidx(ppnt1)).ne.i)goto 119
          ppnt1=ppnt1+1
          goto 111
        endif
      ppnt1=pntc(o)
      ppnt2=ppnt1+ccol(o)-1
 112  if(ppnt1.le.ppnt2)then
        if(mark(rowidx(ppnt1)).ne.i)goto 119
        ppnt1=ppnt1+1
        goto 112
      endif
c
c include column o (and its list) in to the list of column i
c
      pnod=o
 211  if(suplst(pnod).ne.0)then
        pnod=suplst(pnod)
        goto 211
      endif
      suplst(pnod)=suplst(i)
      suplst(i)=o
      supdeg(i)=supdeg(i)+supdeg(o)
c
c remove column/row o from the row and column files
c
      ppnt1=pntr(o)
      ppnt2=ppnt1+crow(o)-1
      do 124 k=ppnt1,ppnt2
        l=colidx(k)
        pnt=pntc(l)
        ii=pnt+ccol(l)-1
        ccol(l)=ccol(l)-1
 123    if(pnt.le.ii)then
          if(rowidx(pnt).eq.o)then
            rowidx(pnt)=rowidx(ii)
            goto 124
          endif
          pnt=pnt+1
          goto 123
        endif
 124  continue
      ppnt1=pntc(o)
      ppnt2=ppnt1+ccol(o)-1
      do 127 k=ppnt1,ppnt2
        l=rowidx(k)
        pnt=pntr(l)
        ii=pnt+crow(l)-1
        crow(l)=crow(l)-1
 126    if(pnt.le.ii)then
          if(colidx(pnt).eq.o)then
            colidx(pnt)=colidx(ii)
            goto 127
          endif
          pnt=pnt+1
          goto 126
        endif
 127  continue
      crow(o)=-1
      p=p-1
      fnd=1
      goto 120
 119  fnd=0
 120  return
      end
c
c ===========================================================================
c
      subroutine hpsort(n,iarr)
c
      integer*4 n,iarr(n)
c
      integer*4 i,j,l,ir,rra
c
c ---------------------------------------------------------------------------
c
      l=n/2+1
      ir=n
  10  if(l.gt.1)then
         l=l-1
         rra=iarr(l)
      else
        rra=iarr(ir)
        iarr(ir)=iarr(1)
        ir=ir-1
        if(ir.le.1)then
          iarr(1)=rra
          goto 999
        endif
      endif
      i=l
      j=l+l
  20  if(j-ir)40,50,60
  40  if(iarr(j).lt.iarr(j+1))j=j+1
  50  if(rra.lt.iarr(j))then
        iarr(i)=iarr(j)
        i=j
        j=j+j
      else
        j=ir+1
      endif
      goto 20
  60  iarr(i)=rra
      goto 10
 999  return
      end
c
c ===========================================================================
c ==========================================================================
c
      subroutine symfact (pivots,rowidx,ecolpnt,count,
     x vcstat,list,next,work,mark,fill,code)


      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 pivots(mn),ecolpnt(mn),count(mn),vcstat(mn),
     x list(mn),next(mn),work(mn),mark(mn),rowidx(cfree),code
c
      integer*4 i,ii,j,k,l,pnt1,pnt2,fnz,kprew,fill
      character*99 buff
c
c --------------------------------------------------------------------------
c
      fnz=nz+1
c
      do 10 i=1,mn
        list(i)=0
        next(i)=0
        mark(i)=0
        work(i)=mn+1
  10  continue
      do 15 i=1,pivotn
        work(pivots(i))=i
  15  continue
      do 20 i=1,n
        if(vcstat(i).le.-2)goto 20
        j=rowidx(ecolpnt(i))
        next(i)=list(j)
        list(j)=i
  20  continue
c
      do 50 ii=1,pivotn
        i=pivots(ii)
        mark(i)=1
        if(i.le.n)goto 50
        l=fnz
        ecolpnt(i)=fnz
        kprew=list(i)
  60    if(kprew.eq.0)goto 70
        pnt1=ecolpnt(kprew)
        pnt2=count(kprew)
        if(fnz.ge.cfree-m)then
          write(buff,'(1x,a)')'Not enough memory'
          call mprnt(buff)
          code=-2
          goto 999
        endif
        do j=pnt1,pnt2
          k=rowidx(j)
          if(mark(k).eq.0)then
            mark(k)=1
            rowidx(fnz)=k
            fnz=fnz+1
          endif
        enddo
        kprew=next(kprew)
        goto 60

  70    do j=l,fnz-1
          mark(rowidx(j))=0
        enddo
        count(i)=fnz-1
        k=fnz-l
        if(k.gt.0)then
          call hpsrt(k,mn,rowidx(l),work)
          j=rowidx(l)
          next(i)=list(j)
          list(j)=i
        endif
  50  continue
      fill=fnz-nz-1
 999  return
      end
c
c ===========================================================================
c
      subroutine transps(n,m,nz,colpnt,rowidx,colnz,
     x           rowpnt,colindex,rownz,perm)
c
      integer*4 n,m,nz,colpnt(n+1),rowidx(nz),rowpnt(m+1),
     x          colindex(nz),perm(n)
      real*8    colnz(nz),rownz(nz)

c
      integer*4 i,j,k,pnt1,pnt2,ii
c
c ---------------------------------------------------------------------------
c
      do 10 i=1,m+1
        rowpnt(i)=0
  10  continue
      do 20 i=1,n
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do 30 j=pnt1,pnt2
          k=rowidx(j)          
          rowpnt(k)=rowpnt(k)+1
  30    continue
  20  continue
c
      j=rowpnt(1)
      k=1
      do 40 i=1,m
        rowpnt(i)=k
        k=k+j
        j=rowpnt(i+1)
  40  continue
c
      do 50 ii=1,n
        i=perm(ii)
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do 60 j=pnt1,pnt2
          k=rowidx(j)
          colindex(rowpnt(k))=i
          rownz(rowpnt(k))=colnz(j)
          rowpnt(k)=rowpnt(k)+1
  60    continue
  50  continue
c
      do 70 i=1,m
        rowpnt(m-i+2)=rowpnt(m-i+1)
  70  continue
      rowpnt(1)=1
      return
      end
c
c =========================================================================
c
      subroutine hpsrt(n,mn,iarr,index)
c
      integer*4 n,mn,iarr(n),index(mn)
c
      integer*4 i,j,l,ir,rra     
c
c ---------------------------------------------------------------------------
c
      l=n/2+1
      ir=n
  10  if(l.gt.1)then
         l=l-1
         rra=iarr(l)
      else
        rra=iarr(ir)
        iarr(ir)=iarr(1)
        ir=ir-1
        if(ir.le.1)then
          iarr(1)=rra
          goto 999
        endif
      endif
      i=l
      j=l+l
  20  if(j-ir)40,50,60
  40  if(index(iarr(j)).lt.index(iarr(j+1)))j=j+1
  50  if(index(rra).lt.index(iarr(j)))then
        iarr(i)=iarr(j)
        i=j
        j=j+j
      else
        j=ir+1
      endif
      goto 20
  60  iarr(i)=rra
      goto 10
 999  if(n.gt.0)then
        if(index(iarr(n)).gt.mn)then
          n=n-1
          goto 999
        endif
      endif
      return
      end
c
c ===========================================================================
c ==========================================================================
c
      subroutine newsmf(colpnt,pivots,rowidx,cnonz,ecolpnt,count,
     x vcstat,invprm,snhead,nodtyp,mark,workr,list,prew,next,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/logprt/ loglog,lfile
      integer*4      loglog,lfile
c
      integer*4 pivots(mn),ecolpnt(mn),count(mn),vcstat(mn),
     x invprm(mn),snhead(mn),nodtyp(mn),mark(mn),rowidx(cfree),
     x colpnt(n1),list(mn),prew(mn),next(mn),code
      real*8 cnonz(nz),workr(mn)
c
      integer*4 i,ii,j,k,l,o,pnt1,pnt2,fnz,kprew
      character*99 buff
c
c --------------------------------------------------------------------------
c
      fnz=nz+1
c
c Restructuring the ordering
c
      k=0
      l=mn+1
      do i=1,pivotn
        j=pivots(i)
        if(vcstat(j).gt.-1)then
          k=k+1
          invprm(k)=j          
        else if(vcstat(j).eq.-1)then          
          l=l-1
          invprm(l)=j         
        endif
      enddo
c
      write(buff,'(1x,a,i5,a)')
     x 'Instable pivot(s), correcting',(mn-l+1),' pivot position(s)'
      call mprnt(buff)
c
      do i=1,k
        pivots(i)=invprm(i)
      enddo
      pivotn=k
      do i=l,mn
        pivotn=pivotn+1
        pivots(pivotn)=invprm(i)
      enddo
c
c Reorder the matrix
c
      do 10 i=1,mn
        invprm(i)=0
        snhead(i)=0
        mark(i)=0
        nodtyp(i)=mn+1
        next(i)=0
        prew(i)=0
        list(i)=0
  10  continue
      do i=1,pivotn
        nodtyp(pivots(i))=i
      enddo
      do ii=1,pivotn
        i=pivots(ii)
        if(i.le.n)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          k=pnt2-pnt1+1
          if(k.gt.0)then
            do j=pnt1,pnt2
              workr(rowidx(j))=cnonz(j)
            enddo
            call hpsrt(k,mn,rowidx(pnt1),nodtyp)
            do j=pnt1,pnt2
              cnonz(j)=workr(rowidx(j))
            enddo
          endif
          ecolpnt(i)=pnt1
  15      if((pnt1.le.pnt2).and.(vcstat(rowidx(pnt2)).le.-2))then
            pnt2=pnt2-1
            goto 15
          endif
          count(i)=pnt2
          if(pnt1.le.pnt2)then
            j=rowidx(pnt1)
            o=list(j)
            next(i)=o
            list(j)=i
            if(o.ne.0)prew(o)=i
            prew(i)=0
          endif
        endif
      enddo
c
      do 50 ii=1,pivotn
        i=pivots(ii)
        mark(i)=1
        if(i.le.n)then
c
c Remove i from the secondary list
c
          if(ecolpnt(i).le.count(i))then
            k=next(i)
            l=prew(i)
            if(k.gt.0)then
              prew(k)=l
            endif
            if(l.gt.0)then
              next(l)=k
            else
              list(rowidx(ecolpnt(i)))=k
            endif
          endif
c
c Simple column of A
c
          if(invprm(i).eq.0)then
            l=ecolpnt(i)
            k=count(i)-l+1
            goto 72
          endif
c
c Transformed column of A
c
          pnt1=ecolpnt(i)
          pnt2=count(i)
          l=fnz
          ecolpnt(i)=fnz
          if(fnz.ge.cfree-mn)then
            write(buff,'(1x,a)')'Not enough memory'
            call mprnt(buff)
            code=-1
            goto 999
          endif
          do j=pnt1,pnt2
            mark(rowidx(j))=1
            rowidx(fnz)=rowidx(j)
            fnz=fnz+1
          enddo
          goto 59
        endif
c
c Create nonzero pattern
c        
        l=fnz
        ecolpnt(i)=fnz
        if(fnz.ge.cfree-mn)then
          write(buff,'(1x,a)')'Not enough memory'
          call mprnt(buff)
          code=-2
          goto 999
        endif
        kprew=list(i)
  25    if(kprew.eq.0)goto 59
        k=next(kprew)
        mark(kprew)=1
        rowidx(fnz)=kprew
        fnz=fnz+1
        pnt1=ecolpnt(kprew)+1
        pnt2=count(kprew)
        ecolpnt(kprew)=pnt1
        if(pnt1.le.pnt2)then
          j=rowidx(pnt1)
          o=list(j)
          next(kprew)=o
          list(j)=kprew
          if(o.ne.0)prew(o)=kprew
          prew(kprew)=0
        endif
        kprew=k
        goto 25
c
c Build new column structure
c
  59    kprew=invprm(i)
  60    if(kprew.eq.0)goto 70
        pnt1=ecolpnt(kprew)
        pnt2=count(kprew)
        do j=pnt1,pnt2
          k=rowidx(j)
          if(mark(k).eq.0)then
            mark(k)=1
            rowidx(fnz)=k
            fnz=fnz+1
          endif
        enddo
        kprew=snhead(kprew)
        goto 60
c
c Linking invperms, free working arrays
c
  70    do j=l,fnz-1
          mark(rowidx(j))=0
        enddo
        count(i)=fnz-1
        k=fnz-l
        if(k.gt.1)call hpsrt(k,mn,rowidx(l),nodtyp)
  72    if(k.gt.0)then
          j=rowidx(l)
          snhead(i)=invprm(j)
          invprm(j)=i
        endif
  50  continue
c
c End of creation of nonzero pattern, set up new supernode partitions
c
      k=loglog
      loglog=0
      call supnode(ecolpnt,count,rowidx,vcstat,pivots,snhead,
     x  invprm,nodtyp)
      loglog=k
 999  return
      end
c
c =========================================================================
c  super dense oszlopok 'multiply' kezelessel
c
c ===========================================================================
c
      subroutine ffactor(pntc,crow,colpnt,rowidx,
     x mark,pivcols,ccol,nonz,diag,
     x cpermf,cpermb,rpermf,rpermb,pntr,cfill,rfill,
     x cpnt,cnext,cprew,rindex,workr,
     x fixn,dropn,fnzmax,fnzmin,active,oper,actual,slktyp,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/factor/ tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8         tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      integer*4 rowidx(cfree),rindex(rfree),colpnt(n1),
     x pivcols(mn),cpermf(mn),cpermb(mn),rpermf(mn),rpermb(mn),
     x ccol(mn),crow(mn),pntc(mn),pntr(mn),mark(mn),cfill(mn),
     x cpnt(mn),cnext(mn),cprew(mn),slktyp(m),rfill(mn),fixn,
     x dropn,fnzmax,fnzmin,active,col,dcols,code
      real*8 nonz(cfree),diag(mn),workr(mn),actual(mn),oper
      character*99 buff
c
c ---------------------------------------------------------------------------
c
c     cpermf       oszloplista elore lancolasa, fejmutato cfirst
c     cpermb       oszloplista hatra lancolasa, fejmutato clast
c     rpermf       sorlista    elore lancolase, fejmutato rfirst
c     rpermb       sorlista    hatra lancolasa, fejmutato rlast
c     ccol         oszlopszamlalok
c     crow         sorszamlalok (vcstat)
c     pntc         oszlopmutatok
c     pntr         sormutatok
c     mark         eliminacios integer segedtomb
c     workr        eliminacios real    segedtomb
c     cfill        a sorfolytonos tarolas update-elesehez segedtomb
c     rfill        a sorfolytonos tarolas update-elesehez segedtomb
c     cpnt         szamlalok szerinti listak fejmutatoja
c     cnext        szamlalok szerinti elore-lancolt lista
c     cprew        szamlalok szerinti hatra-lancolt lista
c
c --------------------------------------------------------------------------
      integer*4 pnt,pnt1,pnt2,i,j,k,l,o,p,endmem,ccfree,rcfree,pmode,
     x rfirst,rlast,cfirst,clast,pcol,pcnt,ppnt1,ppnt2,fill,
     x prewcol,ii,pass,minm,w1,wignore,method
      real*8    pivot,ss,tltmp1,tltmp2
C CMSSW: Temporary integer array needed to avoid reusing REAL*8 for
C integer storage
      integer*4 inds(mn)
c---------------------------------------------------------------------------
c
   1  format(' NOT ENOUGH MEMORY IN THE ROW    FILE ')
   2  format(' NOT ENOUGH MEMORY IN THE COLUMN FILE ')
   3  format(' ROW    REALAXED  :',i6,'  DIAG :',d12.6,'  TYPE :',i3)
   4  format(' COLUMN DROPPED   :',i6,'  DIAG :',d12.6)
   6  format(' NONZEROS         :',i12)
   7  format(' OPERATIONS       :',f13.0)
   8  format(' Superdense cols. :',i12)
C CMSSW: Explicit initialization needed
      tltmp1=0
      tltmp2=0
c
c move elements in the dropped rows to the end of the columns
c
      code=0
      if((order.gt.2.5).and.(order.lt.3.5))then
        method=1
        write(buff,'(a)')' Minimum Local Fill-In Heuristic'
      else
        method=0
        write(buff,'(a)')' Minimum Degree Heuristic'
      endif
      call mprnt(buff)
      wignore=10
      w1=0
      pass=2
      minm=-m-1
      if(dropn.gt.0)then
        do 15 i=1,n
          if(crow(i).le.-2)goto 15
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          p=pnt2
          do 16 j=pnt2,pnt1,-1
            if(crow(rowidx(j)).gt.-2)goto 16
            o=rowidx(j)
            pivot=nonz(j)
            rowidx(j)=rowidx(p)
            rowidx(p)=o
            nonz(j)=nonz(p)
            nonz(p)=pivot
            p=p-1
  16      continue
  15    continue
      endif
c
c initialization
c
      endmem=cfree
      pivotn=0
      pnt=nz+1
      cfirst=0
      clast =0
      pmode =0
      do 11 i=1,mn
        pivcols(i)=0
        ccol(i)=0
        if(crow(i).gt.-2)then
          crow(i)=0
        else
          if(minm.ge.crow(i))minm=crow(i)-1
        endif
        mark(i)=0
  11  continue
c
c set up the permut lists and compute crow
c
      dcols=0
      do 10 i=1,mn
        if(crow(i).le.-2)goto 10
        if(i.le.n)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          o=0
          do while((pnt1+o.le.pnt2).and.(crow(rowidx(pnt1+o)).gt.-2))
            o=o+1
          enddo
          if(o.ge.supdens)then
            pass=1
            crow(i)=minm
            dcols=dcols+1
            goto 10
          endif
          pnt2=pnt1+o-1
          do j=pnt1,pnt2
            crow(rowidx(j))=crow(rowidx(j))+1
          enddo
          pntc(i)=pnt1
          ccol(i)=o
        else
          pntc(i)=pnt
          ccol(i)=0
          pnt=pnt+1
        endif
        if(cfirst.eq.0)then
           cfirst=i
        else
           cpermf(clast)=i
        endif
        cpermb(i)=clast
        clast=i
  10  continue
      cpermf(clast)=0
      ccfree=cfree-pnt
      if(ccfree.lt.mn)then
        write(buff,2)
        call mprnt(buff)
        code=-2
        goto 999
      endif
      write(buff,8)dcols
      call mprnt(buff)
      if(pass.eq.1)then
        tltmp1=tpiv2
        tltmp2=tabs
        tabs=tpiv2
      endif
c
c create pointers to rindex
c
 500  do i=1,mn
        cpnt(i) =0
        cnext(i)=0
        cprew(i)=0
        workr(i)=0.0d+0
      enddo
      pnt=1
      i=cfirst
      rfirst=0
      rlast=0
  25  if(i.gt.0)then
        if(rfirst.eq.0)then           
          rfirst=i
        else
          rpermf(rlast)=i
        endif
        rpermb(i)=rlast        
        rlast=i        
        pntr(i)=pnt
        rfill(i)=pnt
        pnt=pnt+crow(i)
c
c initialize the counter lists
c
        j=crow(i)+ccol(i)+1
        if(j.gt.0)then
          o=cpnt(j)
          cnext(i)=o
          cpnt(j)=i
          if(o.ne.0)cprew(o)=i
        endif
        cprew(i)=0
        i=cpermf(i)
        goto 25
      endif
      rcfree=rfree-pnt
      if(rcfree.lt.mn)then
        write(buff,1)
        call mprnt(buff)
        code=-2
        goto 999
      endif
c
c create the row file : symbolical transps the matrix
c      
      i=cfirst
  26  if(i.gt.0)then
        pnt1=pntc(i)
        pnt2=pnt1+ccol(i)-1
        do 27 j=pnt1,pnt2
          k=rowidx(j)
          if(crow(k).le.-2)goto 27
          rindex(rfill(k))=i
          rfill(k)=rfill(k)+1
  27    continue
        i=cpermf(i)
        goto 26
      endif
      rpermf(rlast)=0
      pcol=0
c
c loop for pivots
c
  50  call fndpiv(cpnt,cnext,pntc,ccol,crow,rowidx,nonz,
C CMSSW: Prevent REAL*8 reusage warning
C Was: diag,pcol,pivot,pmode,method,workr,mark,rindex,pntr
     x diag,pcol,pivot,pmode,method,inds,mark,rindex,pntr)
      if (pcol.eq.0)goto 900
      pivot=1.0d+0/pivot
      diag(pcol)=pivot
      ccfree=endmem-pntc(clast)-ccol(clast)
c
c compress column file
c
      if(ccfree.lt.mn)then
        call ccomprs(mn,cfree,ccfree,endmem,nz,
     x  pntc,ccol,cfirst,cpermf,rowidx,nonz,code)
        if(code.lt.0)goto 999 
      endif
c
c remove pcol from the cpermf lists
c
      prewcol=cpermb(pcol)
      o=cpermf(pcol)
      if(prewcol.ne.0)then
        cpermf(prewcol)=o
      else
        cfirst=o
      endif
      if(o.eq.0)then
        clast=prewcol
      else
        cpermb(o)=prewcol
      endif
c
c remove pcol from the rpermf lists
c
      prewcol=rpermb(pcol)
      o=rpermf(pcol)
      if(prewcol.ne.0)then
        rpermf(prewcol)=o
      else
        rfirst=o
      endif
      if(o.eq.0)then
        rlast=prewcol
      else
        rpermb(o)=prewcol
      endif
c
c administration
c
      pivotn=pivotn+1
      pivcols(pivotn)=pcol
      pcnt=ccol(pcol)+crow(pcol)
c
c remove pcol from the counter lists
c
      o=cnext(pcol)
      ii=cprew(pcol)
      if(ii.eq.0)then
        cpnt(pcnt+1)=o
      else
        cnext(ii)=o
      endif
      if(o.ne.0)cprew(o)=ii
      pnt1=pntc(pcol)
      pnt2=pnt1+ccol(pcol)-1
      if(pnt1.gt.nz)then
        ppnt1=endmem-pcnt
        ppnt2=ppnt1+pcnt-1
        endmem=endmem-pcnt
        ccfree=ccfree-pcnt
        pnt=ppnt1
        do 60 j=pnt1,pnt2
          o=rowidx(j)
          mark(o)=1
          workr(o)=nonz(j)          
          rowidx(pnt)=o
          pnt=pnt+1
c
c remove pcol from the row file
c
          rfill(o)=-1
          cfill(o)=ccol(o)
          l=pntr(o)
          p=l+crow(o)-2
          do 55 k=l,p
            if(rindex(k).eq.pcol)then
              rindex(k)=rindex(p+1)
              goto 60
            endif
  55      continue
  60    continue
        pntc(pcol)=ppnt1
c
c create pivot column from the row file
c
        pnt1=pntr(pcol)
        pnt2=pnt1+crow(pcol)-1
        do 70 i=pnt1,pnt2
          o=rindex(i)
          l=pntc(o)
          p=l+ccol(o)-1
c
c move the original column
c
          if(l.le.nz)then
            if(ccfree.lt.mn)then
              call ccomprs(mn,cfree,ccfree,endmem,nz,
     x        pntc,ccol,cfirst,cpermf,rowidx,nonz,code)
              if(code.lt.0)goto 999 
              l=pntc(o)
              p=l+ccol(o)-1
            endif
            ccfree=ccfree-ccol(o)
            j=pntc(clast)+ccol(clast)
            if(j.le.nz)j=nz+1
            pntc(o)=j
            do 72 k=l,p
              nonz(j)=nonz(k)
              rowidx(j)=rowidx(k)
              j=j+1
  72        continue
            l=pntc(o)
            p=j-1
c
c update the cpermf lists
c
            prewcol=cpermb(o)
            k=cpermf(o)            
            if(prewcol.ne.0)then
              cpermf(prewcol)=k
            else
              if(k.ne.0)then
                cfirst=k
              else
                goto 93
              endif
            endif
            if(k.eq.0)then
              clast=prewcol
            else
              cpermb(k)=prewcol
            endif
            cpermf(clast)=o
            cpermb(o)=clast
            cpermf(o)=0
            clast=o
  93      endif
c
c find element and move in the column o
c
          cfill(o)=ccol(o)-1
          rfill(o)= 0
          do 75 k=l,p
            if(rowidx(k).eq.pcol)then
              mark(o)=1              
              rowidx(pnt)=o
              pnt=pnt+1
              workr(o)=nonz(k)
              rowidx(k)=rowidx(p)
              nonz(k)=nonz(p)
              goto 70
            endif
  75      continue
  70    continue
      else
        ppnt1=pnt1
        ppnt2=pnt2
        do 65 j=pnt1,pnt2
          o=rowidx(j)
          mark(o)=1
          workr(o)=nonz(j)
c
c remove pcol from the row file
c
          rfill(o)=-1
          cfill(o)=ccol(o)
          l=pntr(o)
          p=l+crow(o)-2
          do 67 k=l,p
            if(rindex(k).eq.pcol)then
              rindex(k)=rindex(p+1)
              goto 65
            endif
  67      continue
  65    continue
      endif
      ccol(pcol)=pcnt
c
c remove columns from the counter lists
c
      do 77 j=ppnt1,ppnt2
        i=rowidx(j)
        o=cnext(i)
        ii=cprew(i)
        if(ii.eq.0)then
          cpnt(crow(i)+ccol(i)+1)=o
        else
          cnext(ii)=o
        endif
        if(o.ne.0)cprew(o)=ii
  77  continue
c
c Sort pivot column, set-up workr
c
      if(ppnt1.lt.ppnt2)call hpsort((ppnt2-ppnt1+1),rowidx(ppnt1))
      do p=ppnt1,ppnt2
        nonz(p)=workr(rowidx(p))
        workr(rowidx(p))=workr(rowidx(p))*pivot
      enddo
c
c elimination loop
c
      do 80 p=ppnt1,ppnt2
        i=rowidx(p)
        ss=nonz(p)
c
c transforme diag and delete element from mark 
c
        diag(i)=diag(i)-ss*workr(i)        
        mark(i)=0
        pcnt=pcnt-1
c
c transformation on the column i
c
        fill=pcnt
        pnt1=pntc(i)
        pnt2=pnt1+cfill(i)-1
        do 90 k=pnt1,pnt2
           o=rowidx(k)
           if(mark(o).ne.0)then
             nonz(k)=nonz(k)-ss*workr(o)
             fill=fill-1
             mark(o)=0
           endif
  90    continue
c
c compute the free space
c
        ii=cpermf(i)
        if(ii.eq.0)then
          k=endmem-pnt2-1
        else
          k=pntc(ii)-pnt2-1
        endif
c
c move column to the end of the column file
c
        if(fill.gt.k)then
          if (ccfree.lt.mn)then
            call ccomprs(mn,cfree,ccfree,endmem,nz,
     x      pntc,ccol,cfirst,cpermf,rowidx,nonz,code)
            if(code.lt.0)goto 999 
            pnt1=pntc(i)
            pnt2=pnt1+cfill(i)-1
          endif
          if(i.ne.clast)then
            l=pntc(clast)+ccol(clast)
            pntc(i)=l
            do 95 k=pnt1,pnt2
              rowidx(l)=rowidx(k)
              nonz(l)=nonz(k)
              l=l+1
  95        continue
            pnt1=pntc(i)
            pnt2=l-1
            prewcol=cpermb(i)
            if(prewcol.eq.0)then
              cfirst=ii
            else
              cpermf(prewcol)=ii
            endif
            cpermb(ii)=prewcol
            cpermf(clast)=i
            cpermb(i)=clast
            clast=i
            cpermf(clast)=0
          endif
        endif
c
c create fill in
c
        do 97 k=p+1,ppnt2
          o=rowidx(k)
          if(mark(o).eq.0)then
            mark(o)=1
          else
            pnt2=pnt2+1
            nonz(pnt2)=-ss*workr(o)
            rowidx(pnt2)=o
            rfill(o)=rfill(o)+1
          endif
   97   continue
        pnt2=pnt2+1
        ccol(i)=pnt2-pnt1
        if(i.eq.clast)then
          ccfree=endmem-pnt2-1
        endif
  80  continue
c
c make space for fills in the row file
c
      do 100 j=ppnt1,ppnt2
        i=rowidx(j)
c
c update the counter lists
c
        fill=ccol(i)+crow(i)+rfill(i)+1
        o=cpnt(fill)
        cnext(i)=o
        cpnt(fill)=i
        if(o.ne.0)cprew(o)=i
        cprew(i)=0
        pnt2=pntr(i)+crow(i)-1
c
c compute the free space
c
        ii=rpermf(i)
        if(ii.eq.0)then
          k=rfree-pnt2-1
        else
          k=pntr(ii)-pnt2-1
        endif
c
c move row to the end of the row file
c
        if(k.lt.rfill(i))then
          if(rcfree.lt.mn)then
            call rcomprs(mn,rfree,
     x      rcfree,pntr,crow,rfirst,rpermf,rindex,code)
            if(code.lt.0)goto 999
          endif
          if(ii.ne.0)then
            pnt1=pntr(i)
            pnt2=pnt1+crow(i)-1
            pnt=pntr(rlast)+crow(rlast)
            pntr(i)=pnt
            do 110 l=pnt1,pnt2
              rindex(pnt)=rindex(l)
              pnt=pnt+1
 110        continue
c
c update the rperm lists
c
            prewcol=rpermb(i)
            if(prewcol.eq.0)then
              rfirst=ii
            else
              rpermf(prewcol)=ii
            endif
            rpermb(ii)=prewcol
            rpermf(rlast)=i
            rpermb(i)=rlast
            rlast=i
            rpermf(rlast)=0
          endif
        endif
        crow(i)=crow(i)+rfill(i)
        if(i.eq.rlast)rcfree=rfree-crow(i)-pntr(i)
 100  continue
c
c make pointers to the end of the filled rows
c
      do 120 j=ppnt1,ppnt2
        rfill(rowidx(j))=pntr(rowidx(j))+crow(rowidx(j))-1
 120  continue
c
c generate fill in the row file
c
      do 130 j=ppnt1,ppnt2
        o=rowidx(j)
        pnt1=pntc(o)+cfill(o)
        pnt2=pntc(o)+ccol(o)-1
        do 140 k=pnt1,pnt2
          rindex(rfill(rowidx(k)))=o
          rfill(rowidx(k))=rfill(rowidx(k))-1
 140    continue
 130  continue
c
c end of the pivot loop
c
      goto 50
c
c compute the 'superdense' columns, enter in pass=2
c
 900  if(pass.eq.1)then
        pass=pass+1
        tpiv2=tltmp1
        tabs=tltmp2         
        pmode=1
        call ccomprs(mn,cfree,ccfree,endmem,nz,
     x  pntc,ccol,cfirst,cpermf,rowidx,nonz,code)
        if(code.lt.0)goto 999 
        call excols(rowidx,nonz,rpermf,rpermb,crow,
     x  pntc,ccol,pivcols,cpermf,cpermb,workr,colpnt,diag,
     x  cfirst,clast,endmem,ccfree,minm,code)
        if(code.lt.0)goto 999 
        goto 500
      endif
c
c rank check
c
      if(pivotn.lt.mn-fixn-dropn)then
        i=cfirst
 910    if (i.gt.0)then
          crow(i)=-2
          if(i.le.n)then
            w1=w1+1
            if(w1.le.wignore)then
              write(buff,4)i,diag(i)
              call mprnt(buff)
            endif
            fixn=fixn+1
          else
            w1=w1+1
            if(w1.le.wignore)then
              write(buff,3)(i-n),diag(i),slktyp(i-n)
              call mprnt(buff)
            endif
            actual(i)=-1           
            dropn=dropn+1
          endif
          i=cpermf(i)
          goto 910
        endif
        active=mn-pivotn
        w1=w1-wignore
        if(w1.gt.0)then
          write(buff,'(1x,a,i5)')'Warnings ignored:',w1
          call mprnt(buff)
        endif
      endif
c
c repermut
c
      do 955 i=1,mn
        mark(i)=mn+1
        pntr(i)=0
 955  continue
      do 915 i=1,pivotn
        mark(pivcols(i))=i
 915  continue
      fill=0
      oper=0.0d+0
      do 920 i=1,mn
        if(crow(i).le.-2)goto 920
        pnt1=pntc(i)
        if(pnt1.le.nz)goto 920
        if(ccol(i).gt.0)then
          pnt2=pnt1+ccol(i)-1
          do j=pnt1,pnt2
            workr(rowidx(j))=nonz(j)
          enddo
          call hpsrt(ccol(i),mn,rowidx(pnt1),mark)
          do j=pnt1,pnt2
            nonz(j)=workr(rowidx(j))
          enddo
        endif
        fill=fill+ccol(i)
        oper=oper+dble(ccol(i)*ccol(i)+ccol(i))/2.0d+0
 920  continue
      do 950 i=1,n
        if(crow(i).le.-2)goto 950
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        k=pnt2-pnt1+1
        if(k.gt.0)then         
          do j=pnt1,pnt2
            workr(rowidx(j))=nonz(j)
          enddo
          call hpsrt(k,mn,rowidx(pnt1),mark)
          do j=pnt1,pnt2
            nonz(j)=workr(rowidx(j))
          enddo
        endif
        if(pntc(i).lt.nz)then
          ccol(i)=k
          fill=fill+ccol(i)
          oper=oper+dble(ccol(i)*ccol(i)+ccol(i))/2.0d+0
        endif
 950  continue
c
c create the counter inta1 for the minor iterations
c
      do 960 i=1,pivotn
        col=pivcols(pivotn-i+1)
        if(ccol(col).eq.0)goto 960
        pntr(col)=pntr(rowidx(pntc(col)))+ccol(col)
 960  continue
c
c modify ccol   ( counter ->> pointer )
c
      do 970 i=1,pivotn
        col=pivcols(i)
        ccol(col)=pntc(col)+ccol(col)-1
 970  continue
c
c end of ffactor
c
      if(fnzmin.gt.fill)fnzmin=fill
      if(fnzmax.lt.fill)fnzmax=fill
      write(buff,6)fill
      call mprnt(buff)
      write(buff,7)oper
      call mprnt(buff)
      if(method.eq.1)tfind=-tfind
 999  return
      end
c
c ===========================================================================
c
      subroutine ccomprs(mn,cfree,ccfree,endmem,nz,
     x pnt,count,cfirst,cpermf,rowidx,nonz,code)
      integer*4 mn,cfree,ccfree,endmem,nz,pnt(mn),rowidx(cfree),
     x count(mn),cpermf(mn),cfirst,code
      real*8 nonz(cfree)
c
      integer*4 i,j,pnt1,pnt2,pnt0
      character*99 buff
c ---------------------------------------------------------------------------
   2  format(' NOT ENOUGH MEMORY DETECTED IN SUBROUTINE CCOMPRESS')
      pnt0=nz+1
      i=cfirst
  40  if(i.le.0)goto 30
        pnt1=pnt(i)
        if(pnt1.lt.pnt0)goto 10
        if(pnt1.eq.pnt0)then
          pnt0=pnt0+count(i)
          goto 10
        endif
        pnt(i)=pnt0
        pnt2=pnt1+count(i)-1
        do 20 j=pnt1,pnt2
          rowidx(pnt0)=rowidx(j)
          nonz(pnt0)=nonz(j)
          pnt0=pnt0+1
  20    continue
  10    i=cpermf(i)
      goto 40
  30  ccfree=endmem-pnt0-1
      if(ccfree.lt.mn)then
        write(buff,2)
        call mprnt(buff)
        code=-2
      endif
      return
      end
c
c ===========================================================================
c
      subroutine rcomprs(mn,rfree,rcfree,pnt,count,rfirst,
     x rpermf,rindex,code)
      integer*4 mn,rfree,rcfree,pnt(mn),count(mn),rfirst,rpermf(mn),
     x rindex(rfree),code
c
      integer*4 i,j,ppnt,pnt1,pnt2
      character*99 buff
c
c ---------------------------------------------------------------------------
c
   2  format(' NOT ENOUGH MEMORY DETECTED IN SUBROUTINE RCOMPRESS')
      ppnt=1
      i=rfirst
   5  if(i.eq.0)goto 20
      pnt1=pnt(i)
      if(ppnt.eq.pnt1)then
         ppnt=ppnt+count(i)
         goto 15
      endif
      pnt2=pnt1+count(i)-1
      pnt(i)=ppnt
      do 10 j=pnt1,pnt2
        rindex(ppnt)=rindex(j)
        ppnt=ppnt+1
  10  continue
  15  i=rpermf(i)
      goto 5
  20  rcfree=rfree-ppnt
      if(rcfree.lt.mn)then
        write(buff,2)
        call mprnt(buff)
        code=-2
      endif
      return
      end
c
c ==========================================================================
c
      subroutine excols(rowidx,nonz,rpermf,rpermb,crow,
     x pntc,ccol,pivcols,cpermf,cpermb,workr,colpnt,diag,
     x cfirst,clast,endmem,ccfree,minm,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 rowidx(cfree),rpermf(mn),rpermb(mn),crow(mn),
     x pntc(mn),ccol(mn),pivcols(mn),cpermf(mn),cpermb(mn),colpnt(n1),
     x cfirst,clast,endmem,ccfree,minm,code
      real*8 nonz(cfree),workr(mn),diag(mn)
c
      integer*4 i,j,k,l,o,prewcol,pnt1,pnt2,ppnt1,ppnt2
      real*8 ss
      character*99 buff
c
c ----------------------------------------------------------------------------
c
   2  format(' NOT ENOUGH MEMORY IN THE COLUMN FILE ')
c
      do i=1,mn
        rpermf(i)=0
        rpermb(i)=0
        if(crow(i).gt.-2)crow(i)=0
      enddo
      do i=1,pivotn
        crow(pivcols(i))=-1
      enddo
      prewcol=0
      do 200 i=1,n
        if(crow(i).ne.minm)goto 200
        if(prewcol.eq.0)prewcol=i
        ppnt1=colpnt(i)
        ppnt2=colpnt(i+1)-1
c
c update column's permut list
c
        if(clast.ne.0)then
          pnt1=pntc(clast)+ccol(clast)
          cpermf(clast)=i
        else
          cfirst=i
          pnt1=0
        endif
        cpermb(i)=clast
        cpermf(i)=0
        clast=i
        if(pnt1.lt.nz)pnt1=nz+1
        pntc(i)=pnt1
        pnt2=pnt1
c
c repack the original column
c
        do 202 j=ppnt1,ppnt2
          k=rowidx(j)
          if(crow(k).gt.-2)then
            workr(k)=nonz(j)
            rpermf(k)=1
            if(crow(k).eq.-1)then
              rpermb(k)=rpermb(k)+1
            endif
            rowidx(pnt2)=k
            pnt2=pnt2+1
          endif
 202    continue
c
c Ftran on the column
c
        do j=1,pivotn
          o=pivcols(j)
          if(rpermf(o).gt.0)then
            ppnt1=pntc(o)
            ppnt2=ppnt1+ccol(o)-1
            ss=-workr(o)*diag(o)
            diag(i)=diag(i)+ss*workr(o)
            do k=ppnt1,ppnt2
              l=rowidx(k)
              if(rpermf(l).eq.0)then
                workr(l)=nonz(k)*ss
                rowidx(pnt2)=l
                pnt2=pnt2+1
                rpermf(l)=1
                if((crow(l).eq.-1).or.(l.lt.i))then
                  rpermb(l)=rpermb(l)+1
                endif
              else
                workr(l)=workr(l)+nonz(k)*ss
              endif
            enddo
          endif
        enddo
c
c augftr with the prewious columns
c
        j=prewcol
 215    if(j.ne.i)then        
          ppnt1=pntc(j)
          ppnt2=ppnt1+ccol(j)-1
          do k=ppnt1,ppnt2
            l=rowidx(k)
            if((crow(l).eq.-1).and.(rpermf(l).gt.0))then              
              if(rpermf(j).eq.0)then
                workr(j)=-workr(l)*nonz(k)*diag(l)
                rowidx(pnt2)=j
                pnt2=pnt2+1
                rpermf(j)=1                
                rpermb(j)=rpermb(j)+1
              else
                workr(j)=workr(j)-workr(l)*nonz(k)*diag(l)
              endif
            endif
          enddo
          j=cpermf(j)
          goto 215 
        endif
        ccol(i)=pnt2-pnt1
c
c pack the column
c
        pnt2=pnt2-1
        do j=pnt1,pnt2
          o=rowidx(j)
          nonz(j)=workr(o)         
          rpermf(o)=0
        enddo
        ccfree=endmem-pntc(clast)-ccol(clast)
        if(ccfree.lt.mn)then
          write(buff,2)
          call mprnt(buff)
          code=-2
          goto 999
        endif
        crow(i)=0
 200  continue
c
c Make space in the old factors
c
      o=0
      do i=1,pivotn
        j=pivcols(i)
        o=o+rpermb(j)
      enddo
      ppnt1=endmem-o
      if(ccfree.le.o)then
        write(buff,2)
        call mprnt(buff)
        code=-2
        goto 999
      endif
      endmem=ppnt1
      ccfree=ccfree-o
      do i=pivotn,1,-1
        k=pivcols(i)
        pnt1=pntc(k)
        if(pnt1.gt.nz)then
          pnt2=pnt1+ccol(k)-1
          pntc(k)=ppnt1
          do j=pnt1,pnt2
            rowidx(ppnt1)=rowidx(j)
            nonz(ppnt1)=nonz(j)
            ppnt1=ppnt1+1
          enddo          
          ppnt1=ppnt1+rpermb(k)
        endif
      enddo
c
c make space in the active submatrix
c
      o=0
      i=cfirst
 220  if(i.ne.0)then
        o=o+rpermb(i)
        i=cpermf(i)
        goto 220
      endif 
      if(ccfree.le.o)then
        write(buff,2)
        call mprnt(buff)
        code=-2
        goto 999
      endif
      ccfree=ccfree-o
      ppnt1=pntc(clast)+ccol(clast)+o
      i=clast
  230 if(i.ne.0)then
        pnt1=pntc(i)
        if(pnt1.gt.nz)then
          pnt2=pnt1+ccol(i)-1
          ppnt1=ppnt1-rpermb(i)                    
          do j=pnt2,pnt1,-1
            rowidx(ppnt1)=rowidx(j)
            nonz(ppnt1)=nonz(j)
            ppnt1=ppnt1-1
          enddo
          pntc(i)=ppnt1+1
        endif
        i=cpermb(i)
        goto 230
      endif
c
c Store the dense columns in the final positions
c
      i=prewcol
 250  if(i.gt.0)then
        pnt1=pntc(i)
        pnt2=pnt1+ccol(i)-1
        ppnt1=pnt1
        do j=pnt1,pnt2
          o=rowidx(j)
          if((crow(o).eq.-1).or.(o.lt.i))then
            k=pntc(o)+ccol(o)
            nonz(k)=nonz(j)
            rowidx(k)=i
            ccol(o)=ccol(o)+1 
          else
            nonz(ppnt1)=nonz(j)
            rowidx(ppnt1)=rowidx(j)
            ppnt1=ppnt1+1
          endif
        enddo
        ccol(i)=ppnt1-pnt1
        i=cpermf(i)
        goto 250
      endif
c
c compute crow
c
      do i=1,mn
        if(crow(i).gt.-2)crow(i)=0
      enddo
      i=cfirst
  280 if(i.gt.0)then
        pnt1=pntc(i)
        pnt2=pnt1+ccol(i)-1
        do j=pnt1,pnt2
          crow(rowidx(j))=crow(rowidx(j))+1
        enddo
        i=cpermf(i)
        goto 280
      endif
999   return
      end
c
c =============================================================================
c Find pivot in the augmented system
c Prefer the pivot for expanding the supernodes
c Method=0  minimum count
c Method=1  minimum local fill in
c ===========================================================================
c
      subroutine fndpiv(cpnt,cnext,pntc,ccol,crow,rowidx,nonzeros,
     x diag,pivcol,pivot,md,method,inds,mark,rindex,pntr)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/factor/   tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8           tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
c
      integer*4 cpnt(mn),cnext(mn),pntc(mn),ccol(mn),crow(mn),pivcol,
     x rowidx(cfree),md,method,inds(mn),mark(mn),rindex(rfree),
     x pntr(mn)
      real*8    nonzeros(cfree),diag(mn),pivot

c --------------------------------------------------------------------------
      integer*4 j,k,l,o,nnz,ffind,oldpcol,oldlen,p1,p2,srcmod
      integer*4 fill,mfill,q,oo,kk
      real*8    sol,stab,stab1,d,toler,ss
c --------------------------------------------------------------------------
C CMSSW: Explicit initialization needed
      p1=0
      p2=0
      oldlen=0
      fill=0
      stab=0
      stab1=0
c
c find pivot in sparse columns
c
      mfill=-1
      toler=tpiv1
      if(md.gt.0)then
        srcmod=1
        goto 101
      endif
  10  pivcol=pivcol+1
      if (pivcol.ge.n)goto 100
      if(crow(pivcol).ne.0)goto 10
      if(ccol(pivcol).gt.lam)goto 10
      pivot=diag(pivcol)
      if(abs(pivot).lt.tpiv2)goto 10
      goto 200
c
c find pivot in the another columns
c
 100  md=1
      srcmod=0
 101  oldpcol=pivcol      
      pivcol=0
      stab1=0
      pivot=0
      nnz=md-1
      ffind=0
      if(nnz.lt.1)nnz=1
      md=md-1
      if(md.le.1)md=1
c
c Find supernodal pivot (srcmode=1)
c
 115  if(oldpcol.eq.0)goto 112
      p1=pntc(oldpcol)
      p2=p1+ccol(oldpcol)-1
      oldlen=ccol(oldpcol)
 125  if(p1.gt.p2)goto 114
      j=rowidx(p1)
      if((crow(j)+ccol(j)).lt.oldlen)goto 121
 145  p1=p1+1
      goto 125
 114  if(pivcol.gt.0)goto 200
c
c Find another pivot
c
 112  srcmod=0
      md=0
 110  j=cpnt(nnz)
      if((j.gt.0).and.(md.eq.0))md=nnz
 120  if(j.le.0)goto 150
c
c Compute fill in
c
      if(method.ne.0)then
        q=0
        k=pntc(j)
        l=k+ccol(j)-1
        do o=k,l
          q=q+1
          inds(q)=rowidx(o)
          mark(rowidx(o))=1
        enddo
        k=pntr(j)
        l=k+crow(j)-1
        do o=k,l
          q=q+1
          inds(q)=rindex(o)
          mark(rindex(o))=1
        enddo
        fill=(q*(q-1))/2
        do kk=1,q
          o=inds(kk)
          k=pntc(o)
          l=k+ccol(o)-1
          do oo=k,l
            fill=fill-mark(rowidx(oo))
          enddo
        enddo
        do o=1,q
          mark(inds(o))=0
        enddo
      else
        fill=crow(j)
      endif
      ffind=ffind+1
      if((mfill.ge.0).and.(fill.ge.mfill))goto 130
 121  d=diag(j)
      sol=abs(d)
      if(sol.lt.tabs)goto 130
      k=pntc(j)
c
c stability test
c
      stab=sol
      l=k+ccol(j)-1
      do 32 o=k,l
        ss=abs(nonzeros(o))
        if(stab.lt.ss)stab=ss
  32  continue
      stab=sol/stab
      if(stab.lt.toler)goto 130
      if(mfill.lt.0)mfill=fill+1
      if((fill.lt.mfill).or.((fill.eq.mfill).and.(stab.gt.stab1)))then
        pivot=d
        pivcol=j
        stab1=stab
        mfill=fill
        goto 130
      endif
 130  if((srcmod.gt.0).and.(pivcol.ne.0))then
cccc          md=md-1
cccc          if(md.lt.1)md=1
         goto 200
      endif
      if((ffind.gt.tfind).and.(pivcol.ne.0))goto 200
      if(srcmod.gt.0)goto 145
      j=cnext(j)
      goto 120
 150  if((pivcol.eq.0).or.(method.ne.0))then
        nnz=nnz+1
        if(nnz.le.mn)goto 110
        if(pivcol.gt.0)goto 200
        toler=toler/10
        nnz=md
        if((toler.ge.tpiv2).and.(nnz.gt.0))goto 115
        md=1
      endif
 200  return
      end
c
c ==========================================================================
c Supernodal left looking, primer supernode loop (cache),
c Supernode update with indirect addressing
c Relative pivot tolerance
c =============================================================================
c
      subroutine nfactor(ecolpnt,
     x vcstat,rowidx,pivots,count,
     x nonzeros,diag,err,updat,mut,index,dropn,slktyp,
     x snhead,fpnt,invperm,nodtyp,dv,odiag)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 err,mut(mn),dropn,ecolpnt(mn),vcstat(mn),
     x rowidx(cfree),pivots(mn),count(mn),index(mn),slktyp(m)
      integer*4 snhead(mn),fpnt(mn),invperm(mn),nodtyp(mn)
      real*8 nonzeros(cfree),diag(mn),updat(mn),dv(m),odiag(mn)
c
      common/factor/ tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8         tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c -----------------------------------------------------------------------------
      integer*4 i,j,k,o,p,pnt1,pnt2,ppnt1,ppnt2,col,kprew,
     x prewnode,ppnode,rb,w1
      real*8 s,diap,diam
      character*99 buff
c------------------------------------------------------------------------------
C CMSSW: Explicit initialization needed
      ppnt1=0
      ppnt2=0

      err=0
      w1=0
c
c  initialization
c
      do 10 i=1,mn
        mut(i)=0
        index(i)=0
        updat(i)=0.0
        fpnt(i)=ecolpnt(i)
  10  continue
      ppnode=0
      prewnode=0
      i=0
c
c  loop for pivot columns
c
 100  i=i+1
      if(i.gt.pivotn)goto 60
      col=pivots(i)
c
c  step vcstat if relaxed
c
      if(vcstat(col).le.-2)then
        call colremv(i,col,mut,index,fpnt,count,pivots,invperm,
     x  snhead,nodtyp,rowidx,nonzeros,ppnode,prewnode)
        diag(col)=0.0
        i=i-1
        if((ppnode.gt.0).and.(prewnode.eq.i))goto 110        
        goto 100
      endif
c
      ppnt1=ecolpnt(col)
      ppnt2=count(col)
      if(ppnt1.le.nz)then
        diag(col)=1.0d00/diag(col)
        goto 180
      endif
      kprew=index(col)
c
c  compute
c
      diap=diag(col)
      diam=0.0d+0      
 130  if(kprew)129,150,131
c
c Standard transformation
c
 131  k=mut(kprew)
      pnt1=fpnt(kprew)
      pnt2=count(kprew)
      if(pnt1.lt.pnt2)then
        o=rowidx(pnt1+1)
        mut(kprew)=index(o)
        index(o)=kprew
      endif
      pnt1=pnt1+1
      fpnt(kprew)=pnt1
      s=-nonzeros(pnt1-1)*diag(kprew)
      if(kprew.le.n)then
        diap=diap+s*nonzeros(pnt1-1)
      else
        diam=diam+s*nonzeros(pnt1-1)
      endif
      do 170 o=pnt1,pnt2
        updat(rowidx(o))=updat(rowidx(o))+s*nonzeros(o)
 170  continue
      kprew=k
      goto 130
c
c supernodal transformation
c
 129  kprew=-kprew
      k=mut(kprew)
      p=invperm(kprew)
      pnt1=fpnt(kprew)+1      
      if(pnt1.le.count(kprew))then
        o=rowidx(pnt1)
        mut(kprew)=index(o)
        index(o)=-kprew
      endif
      if(kprew.le.n)then
        call cspnd(p,snhead(p),diag,nonzeros,
     x  fpnt,count,pivots,updat,diap,rowidx(pnt1))
      else
        call cspnd(p,snhead(p),diag,nonzeros,
     x  fpnt,count,pivots,updat,diam,rowidx(pnt1))
      endif
      kprew=k
      goto 130
c
c  pack a column, and free the working array
c
 150  do k=ppnt1,ppnt2
        nonzeros(k)=updat(rowidx(k))
        updat(rowidx(k))=0
      enddo
c
c set up diag
c
      if((ppnode.le.0).or.(prewnode.ne.snhead(i)))then
        diap=diap+diam
        diam=max(trabs,abs(diam*trabs))
        if(abs(diap).lt.diam)then
          call rngchk(rowidx,nonzeros,ecolpnt(col),count(col),
     x    vcstat,rb,diag,slktyp,dropn,col,dv,diap,w1,odiag(col))
          if(rb.ne.0)err=1
          diag(col)=diap
          if(vcstat(col).le.-2)goto 100
        else
          diag(col)=1.0d00/diap
        endif
      else
        diag(col)=diam
        updat(col)=diap
      endif
c
c Transformation in (primer) supernode
c
 110  if(prewnode.eq.i)then
        if(ppnode.gt.0)then
          do j=ppnode+1,i
            o=j-1
            p=pivots(j)
            call cspnode(ppnode,o,diag,nonzeros,fpnt,count,pivots,
     x      nonzeros(ecolpnt(p)),diag(p))
            diam=max(trabs,abs(diag(p)*trabs))
            diag(p)=diag(p)+updat(p)
            if(abs(diag(p)).lt.diam)then
              call rngchk(rowidx,nonzeros,ecolpnt(p),count(p),
     x        vcstat,rb,diag,slktyp,dropn,p,dv,diag(p),w1,odiag(p))
              if(rb.ne.0)err=1
            else
              diag(p)=1.0d00/diag(p)
            endif
          enddo
        endif
        ppnode=0
      endif
c
c Update the linked list
c
 180  if(snhead(i).eq.0)then
        ppnode=0
        if(ppnt1.le.ppnt2)then
          j=rowidx(ppnt1)
          mut(col)=index(j)
          index(j)=col
        endif
        prewnode=0
      else
        if(prewnode.ne.snhead(i))then
          prewnode=snhead(i)
          if(nodtyp(i).gt.0)then
            ppnode=i
          else
            ppnode=-i
          endif
          if(ecolpnt(pivots(prewnode)).le.count(pivots(prewnode)))then
            j=rowidx(ecolpnt(pivots(prewnode)))
            mut(col)=index(j)
            index(j)=-col
          endif
        endif
      endif
c
c  end of the main loop
c
      goto 100
c
c  end of mfactor
c
  60  if(w1.gt.0)then
        write(buff,'(1x,a,i6)')'Total warnings of row dependencies:',w1
        call mprnt(buff)
      endif
      return
      end
c
c =============================================================================
c
      subroutine colremv(i,col,mut,index,fpnt,count,pivots,invperm,
     x snhead,nodtyp,rowidx,nonzeros,ppnode,prewnode)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 i,col,mut(mn),index(mn),fpnt(mn),count(mn),pivots(mn),
     x invperm(mn),snhead(mn),nodtyp(mn),rowidx(cfree),ppnode,
     x prewnode
      real*8 nonzeros(cfree)
c
      integer*4 j,jj,k,l,o,p,pnt1
c 
        jj=index(col)
 195    if(jj.eq.0)goto 103
        if(jj.lt.0)then
          j=-jj
        else
          j=jj
        endif
        k=mut(j)
        pnt1=fpnt(j)
        call move(pnt1,count(j),rowidx,nonzeros)
        if(pnt1.le.count(j))then
          o=rowidx(pnt1)
          mut(j)=index(o)
          index(o)=jj
        endif
        if(jj.lt.0)then
          p=invperm(j)
          l=snhead(p)
          do o=p+1,l
          call move(fpnt(pivots(o)),count(pivots(o)),rowidx,nonzeros)
          enddo
        endif
        jj=k
        goto 195
c
c Step in the primer supernode
c
 103  if((ppnode.gt.0).and.(prewnode.eq.snhead(i)))then
        l=i-1
        do o=ppnode,l
          pnt1=fpnt(pivots(o))
 104      if(pnt1.le.count(pivots(o)))then
            if(rowidx(pnt1).eq.col)then
              call move(pnt1,count(pivots(o)),rowidx,nonzeros)
              pnt1=count(pivots(o)) 
            endif
            pnt1=pnt1+1
            goto 104
          endif
        enddo
      endif
c
c Make changes
c
      pivotn=pivotn-1
      do j=i,pivotn
        pivots(j)=pivots(j+1)
        snhead(j)=snhead(j+1)
        nodtyp(j)=nodtyp(j+1)
      enddo
      do j=1,pivotn
        if(snhead(j).ge.i)snhead(j)=snhead(j)-1
        invperm(pivots(j))=j
      enddo
      if(prewnode.ge.i)prewnode=prewnode-1
      return
      end
c
c =============================================================================
c
      subroutine move(pnt1,pnt2,rowidx,nonzeros)
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4 pnt1,pnt2,rowidx(cfree),i,j
      real*8    nonzeros(cfree),s
      if(pnt1.le.pnt2)then
        j=rowidx(pnt1)
        s=nonzeros(pnt1)       
        pnt2=pnt2-1
        do i=pnt1,pnt2
          nonzeros(i)=nonzeros(i+1)
          rowidx(i)=rowidx(i+1)
        enddo
        rowidx(pnt2+1)=j
        nonzeros(pnt2+1)=s
      endif
      return
      end
c
c =============================================================================      
              
c Supernodal left looking, primer supernode loop (cache),
c Supernode update with indirect addressing
c Relative pivot tolerance
c ==========================================================================
c
      subroutine mfactor(ecolpnt,
     x vcstat,colpnt,rowidx,pivots,count,mut,nonzeros,
     x diag,err,updat,list,index,dropn,slktyp,
     x snhead,fpnt,invperm,nodtyp,dv,odiag)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 err,list(mn),mut(mn),dropn
      integer*4 ecolpnt(mn),vcstat(mn),colpnt(n1),rowidx(cfree)
      integer*4 pivots(mn),count(mn),index(mn),slktyp(m)
      integer*4 snhead(mn),fpnt(mn),invperm(mn),nodtyp(mn)
      real*8 nonzeros(cfree),diag(mn),updat(mn),dv(m),odiag(mn)
c
      common/factor/ tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      real*8         tpiv1,tpiv2,tabs,trabs,lam,tfind,order,supdens
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c --------------------------------------------------------------------------
      integer*4 i,j,k,l,o,p,pnt1,pnt2,ppnt1,ppnt2,mk,col,kprew,rb,
     x ppnode,prewnode,w1
      real*8 s,diap,diam
      character*99 buff
c---------------------------------------------------------------------------
C CMSSW: Explicit initialization needed
      o=0

      err=0
      w1=0
c
c  initialization
c
      do 10 i=1,mn
        list(i)=0
        index(i)=0
        updat(i)=0.0d+0
        fpnt(i)=0
  10  continue
c
c initialize dll
c
      do 15 i=1,n
        if(vcstat(i).le.-2)goto 15
        k=ecolpnt(i)
        if(k.le.nz)goto 15
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        if(pnt1.le.pnt2)then
          o=rowidx(pnt1)
          fpnt(i)=index(o)
          index(o)=i
          list(i)=pnt1
        endif
  15  continue
c
c  set the extra part of the matrix using a dll
c
      do 20 col=1,pivotn
        i=pivots(col)
        pnt1=ecolpnt(i)
        if(pnt1.le.nz)goto 20
        pnt2=count(i)
        o=0
        if(i.le.n)then
          if(vcstat(i).le.-2)goto 20
          ppnt1=list(i)
          ppnt2=colpnt(i+1)-1
          do 18 j=ppnt1,ppnt2
            k=rowidx(j)
            updat(k)=nonzeros(j)
            o=o+1
            mut(o)=k
  18      continue
          list(i)=ppnt2+1
        else
          kprew=index(i)
          if(kprew.eq.0)goto 25
          if(vcstat(i).le.-2)then
  21        mk=fpnt(kprew)
            ppnt1=list(kprew)+1
            if(ppnt1.lt.colpnt(kprew+1))then
              list(kprew)=ppnt1
              k=rowidx(ppnt1)
              fpnt(kprew)=index(k)
              index(k)=kprew
            endif
            kprew=mk
            if(kprew.ne.0)goto 21
          else
  22        mk=fpnt(kprew)
            ppnt1=list(kprew)+1
            if(ppnt1-colpnt(kprew+1))11,12,13
  11        updat(kprew)=nonzeros(ppnt1-1)
            list(kprew)=ppnt1
            k=rowidx(ppnt1)
            fpnt(kprew)=index(k)
            index(k)=kprew
            o=o+1
            mut(o)=kprew
            goto 13
  12        updat(kprew)=nonzeros(ppnt1-1)
            list(kprew)=ppnt1
            o=o+1
            mut(o)=kprew
  13        kprew=mk
            if(kprew.ne.0)goto 22
          endif
        endif
c
c set column i and delete updat
c
  25    do 23 j=pnt1,pnt2
           nonzeros(j)=updat(rowidx(j))
  23    continue
        do 26 j=1,o
          updat(mut(j))=0
  26    continue
  20  continue
c
c  initialize for the computation
c
      do 30 i=1,mn
        mut(i)=0
        fpnt(i)=ecolpnt(i)
        list(i)=0
        index(i)=0
        updat(i)=0.0
  30  continue
      ppnode=0
      prewnode=0
      i=0
c
c  loop for pivot columns
c
 100  i=i+1
      if(i.gt.pivotn)goto 60
      col=pivots(i)
      ppnt1=ecolpnt(col)
      ppnt2=count(col)
c
c  step vcstat if relaxed
c
      if(vcstat(col).le.-2)then
        call colremv(i,col,mut,index,fpnt,count,pivots,invperm,
     x  snhead,nodtyp,rowidx,nonzeros,ppnode,prewnode)
        do 75 j=ppnt1,ppnt2
          k=rowidx(j)
          if((k.gt.n).or.(ecolpnt(k).le.nz))goto 75
          l=colpnt(k)
          o=colpnt(k+1)-1
          do p=l,o
            if(rowidx(p).eq.col)then
              call move(p,o,rowidx,nonzeros)
              goto 75
            endif
          enddo
  75    continue
        i=i-1
        if((ppnode.gt.0).and.(prewnode.eq.i))goto 110
        goto 100
      endif
c
      if(ppnt1.le.nz)then
        diag(col)=1.0d00/diag(col)
        goto 180
      endif
      kprew=index(col)
c
c repack a column
c
      do k=ppnt1,ppnt2
        updat(rowidx(k))=nonzeros(k)
      enddo
      if(col.le.n)then
        diam=diag(col)
        diap=0.0d+0
      else
        diap=diag(col)
        diam=0.0d+0
      endif
 130  if(kprew)129,150,131
c
c Standard transformation
c
 131  k=mut(kprew)
      pnt1=fpnt(kprew)
      pnt2=count(kprew)
      if(pnt1.lt.pnt2)then
        o=rowidx(pnt1+1)
        mut(kprew)=index(o)
        index(o)=kprew
      endif
      pnt1=pnt1+1
      fpnt(kprew)=pnt1
      s=-nonzeros(pnt1-1)*diag(kprew)
      if(kprew.le.n)then
        diap=diap+s*nonzeros(pnt1-1)
      else
        diam=diam+s*nonzeros(pnt1-1)
      endif
      do 170 o=pnt1,pnt2
        updat(rowidx(o))=updat(rowidx(o))+s*nonzeros(o)
 170  continue
      kprew=k
      goto 130
c
c supernodal transformation
c
 129  kprew=-kprew
      k=mut(kprew)
      p=invperm(kprew)
      pnt1=fpnt(kprew)+1
      if(pnt1.le.count(kprew))then
        o=rowidx(pnt1)
        mut(kprew)=index(o)
        index(o)=-kprew
      endif
      if(kprew.le.n)then
        call cspnd(p,snhead(p),diag,nonzeros,
     x  fpnt,count,pivots,updat,diap,rowidx(pnt1))
      else
        call cspnd(p,snhead(p),diag,nonzeros,
     x  fpnt,count,pivots,updat,diam,rowidx(pnt1))
      endif
      kprew=k
      goto 130
c
c  pack a column
c
 150  do k=ppnt1,ppnt2
        nonzeros(k)=updat(rowidx(k))
      enddo
c
c  set up diag
c
      if((ppnode.le.0).or.(prewnode.ne.snhead(i)))then
        diap=diap+diam
        diam=max(trabs,abs(diam*trabs))
        if(abs(diap).lt.diam)then
          call rngchk(rowidx,nonzeros,ecolpnt(col),count(col),
     x    vcstat,rb,diag,slktyp,dropn,col,dv,diap,w1,odiag(col))
          if(rb.ne.0)err=1
          diag(col)=diap
          if(vcstat(col).le.-2)goto 100
        else
          diag(col)=1.0d00/diap
        endif
      else
        diag(col)=diam
        updat(col)=diap
      endif
c
c Transformation in (primer) supernode
c
 110  if(prewnode.eq.i)then
        if(ppnode.gt.0)then
          do j=ppnode+1,i
            o=j-1
            p=pivots(j)
            call cspnode(ppnode,o,diag,nonzeros,fpnt,count,pivots,
     x      nonzeros(ecolpnt(p)),diag(p))
            diam=max(trabs,abs(diag(p)*trabs))
            diag(p)=diag(p)+updat(p)
            if(abs(diag(p)).lt.diam)then
              call rngchk(rowidx,nonzeros,ecolpnt(p),count(p),
     x        vcstat,rb,diag,slktyp,dropn,p,dv,diag(p),w1,odiag(p))
              if(rb.ne.0)err=1
            else
              diag(p)=1.0d+0/diag(p)
            endif
          enddo
        endif
        ppnode=0
      endif
c
c Update the linked list
c
 180  if(snhead(i).eq.0)then
        ppnode=0
        if(ppnt1.le.ppnt2)then
          j=rowidx(ppnt1)
          mut(col)=index(j)
          index(j)=col
        endif
        prewnode=0
      else
        if(prewnode.ne.snhead(i))then
          prewnode=snhead(i)
          if(nodtyp(i).gt.0)then
            ppnode=i
          else
            ppnode=-i
          endif
          if(ecolpnt(pivots(prewnode)).le.count(pivots(prewnode)))then
            j=rowidx(ecolpnt(pivots(prewnode)))
            mut(col)=index(j)
            index(j)=-col
          endif
        endif
      endif
c
c  end of the main loop
c
      goto 100
c
c  end of mfactor
c
  60  if(w1.gt.0)then
        write(buff,'(1x,a,i6)')'Total warnings of row dependencies:',w1
        call mprnt(buff)
      endif
      return
      end
c
c =============================================================================
c =============================================================================
c
      subroutine rngchk(rowidx,nonzeros,pnt1,pnt2,
     x  vcstat,rb,diag,slktyp,dropn,col,dv,dia,w1,odia)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      integer*4 pnt1,pnt2,rowidx(cfree),vcstat(mn),rb,
     x slktyp(m),dropn,col
      real*8    nonzeros(cfree),diag(mn),dv(m),dia,odia
c
      integer*4 i,j,w1,wignore
      character*99 buff
c
c --------------------------------------------------------------------------
c      
      wignore=5
      rb=0
      if(col.le.n)then
        if(diag(col).lt.0)then
          dia=-1.0d+12
          odia=odia+1.0d+00/dia
        else
          dia=+1.0d+12
          odia=odia+1.0d+00/dia
        endif
      else
        dia=0.0d+0
c
c Check for modification columns
c
        do 10 i=pnt1,pnt2
          j=rowidx(i)
          if((vcstat(j).le.-2).or.(j.gt.n))goto 10
          if(abs(nonzeros(i)).lt.tzer)goto 10
ccc             dia=+1.0+10
ccc             odia=odia+1.0d+00/dia          
          rb=1
          vcstat(col)=-1
          goto 20
 10     continue
c
c Dependent row, relax only if the dual variable is zero !
c
        if(abs(dv(col-n)).lt.tzer)then
          vcstat(col)=-2
          dropn=dropn+1  
          w1=w1+1
          if(w1.le.wignore)then
            write(buff,'(1x,a,i5,a,i6)')
     x      'WARNING :  Row DROPPED ',col-n,'  Type:',slktyp(col-n)
            call mprnt(buff)
          endif
        endif
      endif
 20   return
      end
c
c ==========================================================================
c     nem relativ nullazassal
c ==========================================================================
c
      subroutine augftr(ecolpnt,
     x vcstat,rowidx,pivots,count,nonzeros,diag,vector)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 ecolpnt(mn),vcstat(mn),rowidx(cfree)
      integer*4 pivots(mn),count(mn)
      real*8 nonzeros(cfree),diag(mn),vector(mn)
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c --------------------------------------------------------------------------
      integer*4 i,j,pnt1,pnt2,col,o
      real*8 val
c---------------------------------------------------------------------------
      do i=1,pivotn
        col=pivots(i)
        if (vcstat(col).gt.-2)then
          val=vector(col)*diag(col)
          if(abs(val).gt.tzer)then
            pnt1=ecolpnt(col)
            pnt2=count(col)
            do j=pnt1,pnt2
              o=rowidx(j)
              vector(o)=vector(o)-val*nonzeros(j)
            enddo
          endif
        endif
      enddo
      do i=1,mn
        if(vcstat(i).le.-2)vector(i)=0
      enddo
      return
      end
c
c ==========================================================================
c
      subroutine augbtr(ecolpnt,
     x vcstat,rowidx,pivots,count,nonzeros,diag,vector)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 ecolpnt(mn),vcstat(mn),rowidx(cfree)
      integer*4 pivots(mn),count(mn)
      real*8 nonzeros(cfree),diag(mn),vector(mn)
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c --------------------------------------------------------------------------
      integer*4 i,j,col,pnt1,pnt2
      real*8 sol
c---------------------------------------------------------------------------
c
      do i=1,pivotn
        col=pivots(pivotn+1-i)
        if(vcstat(col).gt.-2)then
          sol=vector(col)
          pnt1=ecolpnt(col)
          pnt2=count(col)
          do j=pnt1,pnt2
            sol=sol-nonzeros(j)*vector(rowidx(j))
          enddo
          vector(col)=sol*diag(col)
        endif
      enddo
      return
      end
c ==========================================================================
c Multi predictor-corrector direction
c L2 norm
c ===========================================================================
c
      subroutine citref(diag,odiag,pivots,rowidx,nonzeros,colpnt,
     x ecolpnt,count,vcstat,xrhs,rwork1,rwork2,rwork3,
     x bounds,xs,up,vartyp,slktyp)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/numer/ tplus,tzer
      real*8        tplus,tzer
      common/itref/ tresx,tresy,maxref
      real*8        tresx,tresy
      integer*4     maxref
c
      integer*4 ecolpnt(mn),count(mn),rowidx(cfree),
     x pivots(mn),colpnt(n1),vcstat(mn),vartyp(n),slktyp(m)
      real*8 diag(mn),odiag(mn),nonzeros(cfree),xrhs(mn),
     x rwork1(mn),rwork2(mn),rwork3(mn),bounds(mn),xs(mn),up(mn)
c
c ---------------------------------------------------------------------------
c
      integer*4 i,j,pnt1,pnt2,refn
      real*8 maxrx,maxry,sx,sol,l2,ol2
c
c ---------------------------------------------------------------------------
c
c Simple case : No refinement
c
      if(maxref.le.0)then
        call augftr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,xrhs)
        call augbtr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,xrhs)
        goto 999
      endif
      do i=1,mn
        rwork1(i)=xrhs(i)
      enddo
      ol2=1.0d+0/tzer
      do i=1,mn
        rwork3(i)=0.0d+0
      enddo
      refn=-1
c
c Main loop
c
  10  refn=refn+1
      call augftr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x diag,xrhs)
      call augbtr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x diag,xrhs)
      do i=1,mn
        xrhs(i)=xrhs(i)+rwork3(i)
      enddo
c
c Compute the residuals
c
      l2=0.0d+0
      maxrx=0.0d+0
      maxry=0.0d+0
      do i=1,mn
        rwork2(i)=rwork1(i)-odiag(i)*xrhs(i)
      enddo
      do i=1,n
        if(vcstat(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          sx=xrhs(i)
          sol=rwork2(i)
          do j=pnt1,pnt2
            rwork2(rowidx(j))=rwork2(rowidx(j))-nonzeros(j)*sx
            sol=sol-nonzeros(j)*xrhs(rowidx(j))
          enddo
          rwork2(i)=sol
          if(maxry.lt.abs(sol))maxry=abs(sol)
          l2=l2+sol*sol
        endif
      enddo
      do i=1,m
        if(vcstat(i+n).gt.-2)then
           if(maxrx.lt.abs(rwork2(i+n)))maxrx=abs(rwork2(i+n))
           l2=l2+rwork2(i+n)*rwork2(i+n)
        endif
      enddo
      l2=sqrt(l2)
      if(l2.ge.ol2)then
        do i=1,mn
          xrhs(i)=rwork3(i)
        enddo
      else
        if((maxrx.gt.tresx).or.(maxry.gt.tresy))then
          if(refn.lt.maxref)then
            ol2=l2
            do i=1,mn
              rwork3(i)=xrhs(i)
              xrhs(i)=rwork2(i)
            enddo
            goto 10
          endif
        endif
      endif
c
c End of the main loop, reset work3 (upinf)=bounds-xs-up
c
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.lt.0)then
            sol=bounds(i)-xs(i)-up(i)
          else
            sol=0.0d+0
          endif
        else
          sol=0.0d+0
        endif
        rwork3(i)=sol
      enddo
c
c return
c
 999  return
      end
c
c ============================================================================
c 6 way loop unrolling
c
c ============================================================================
c
      subroutine cspnode(firstc,lastc,diag,nonzeros,
     x fpnt,count,pivots,knz,dia)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 firstc,lastc,fpnt(mn),count(mn),pivots(mn)
      real*8 diag(mn),nonzeros(cfree),knz(mn),dia
c
      integer*4 pnt11,pnt12,pnt13,pnt14,pnt15,pnt16,
     x col1,col2,col3,col4,col5,col6,frs,j,pnt2
      real*8    s1,s2,s3,s4,s5,s6
c
c compute
c
      frs=firstc
c
  99  if(lastc-2-frs) 98,30,97
  98  if(lastc-frs) 999,10,20 
  97  if(lastc-4-frs) 40,50,60 
c
c
c
  60  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      col5=pivots(frs+4)
      col6=pivots(frs+5)  
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt15=fpnt(col5)
      pnt16=fpnt(col6)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      fpnt(col5)=pnt15+1
      fpnt(col6)=pnt16+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      s5=-nonzeros(pnt15)*diag(col5)
      s6=-nonzeros(pnt16)*diag(col6)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4+
     x    nonzeros(pnt15)*s5+nonzeros(pnt16)*s6
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4+
     x  nonzeros(pnt15+j)*s5+nonzeros(pnt16+j)*s6
      enddo
      frs=frs+6
      goto 99
c
c
c
  50  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      col5=pivots(frs+4)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt15=fpnt(col5)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      fpnt(col5)=pnt15+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      s5=-nonzeros(pnt15)*diag(col5)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4+
     x    nonzeros(pnt15)*s5  
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4+
     x  nonzeros(pnt15+j)*s5  
      enddo             
      goto 999
c
c
c     
  40  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4
      enddo             
      goto 999
c
c
c
  30  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3
      enddo       
      goto 999
c
c
c
  20  col1=pivots(frs)
      col2=pivots(frs+1)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2     
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2     
      enddo       
      goto 999
c
c
c
  10  col1=pivots(frs)
      pnt11=fpnt(col1)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      s1=-nonzeros(pnt11)*diag(col1)
      dia=dia+
     x    nonzeros(pnt11)*s1    
      do j=1,pnt2
        knz(j)=knz(j)+
     x  nonzeros(pnt11+j)*s1       
      enddo
c
 999  return
      end
c
c ==========================================================================
c 6 way loop unrolling
c
c ============================================================================
c
      subroutine cspnd(firstc,lastc,diag,nonzeros,
     x fpnt,count,pivots,knz,dia,index)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 firstc,lastc,fpnt(mn),count(mn),pivots(mn),index(mn)
      real*8 diag(mn),nonzeros(cfree),knz(mn),dia
c
      integer*4 pnt11,pnt12,pnt13,pnt14,pnt15,pnt16,
     x col1,col2,col3,col4,col5,col6,frs,j,pnt2
      real*8    s1,s2,s3,s4,s5,s6
c
c compute
c
      frs=firstc
c
  99  if(lastc-2-frs) 98,30,97
  98  if(lastc-frs) 999,10,20 
  97  if(lastc-4-frs) 40,50,60 
c
c
c
  60  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      col5=pivots(frs+4)
      col6=pivots(frs+5)  
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt15=fpnt(col5)
      pnt16=fpnt(col6)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      fpnt(col5)=pnt15+1
      fpnt(col6)=pnt16+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      s5=-nonzeros(pnt15)*diag(col5)
      s6=-nonzeros(pnt16)*diag(col6)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4+
     x    nonzeros(pnt15)*s5+nonzeros(pnt16)*s6
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4+
     x  nonzeros(pnt15+j)*s5+nonzeros(pnt16+j)*s6
      enddo
      frs=frs+6
      goto 99
c
c
c
  50  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      col5=pivots(frs+4)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt15=fpnt(col5)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      fpnt(col5)=pnt15+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      s5=-nonzeros(pnt15)*diag(col5)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4+
     x    nonzeros(pnt15)*s5  
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4+
     x  nonzeros(pnt15+j)*s5  
      enddo             
      goto 999
c
c
c     
  40  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      col4=pivots(frs+3)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt14=fpnt(col4)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      fpnt(col4)=pnt14+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      s4=-nonzeros(pnt14)*diag(col4)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3+nonzeros(pnt14)*s4
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3+nonzeros(pnt14+j)*s4
      enddo             
      goto 999
c
c
c
  30  col1=pivots(frs)
      col2=pivots(frs+1)
      col3=pivots(frs+2)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt13=fpnt(col3)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      fpnt(col3)=pnt13+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      s3=-nonzeros(pnt13)*diag(col3)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2+
     x    nonzeros(pnt13)*s3
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2+
     x  nonzeros(pnt13+j)*s3
      enddo       
      goto 999
c
c
c
  20  col1=pivots(frs)
      col2=pivots(frs+1)
      pnt11=fpnt(col1)
      pnt12=fpnt(col2)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      fpnt(col2)=pnt12+1
      s1=-nonzeros(pnt11)*diag(col1)
      s2=-nonzeros(pnt12)*diag(col2)
      dia=dia+
     x    nonzeros(pnt11)*s1+nonzeros(pnt12)*s2     
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1+nonzeros(pnt12+j)*s2     
      enddo       
      goto 999
c
c
c
  10  col1=pivots(frs)
      pnt11=fpnt(col1)
      pnt2=count(col1)-pnt11
      fpnt(col1)=pnt11+1
      s1=-nonzeros(pnt11)*diag(col1)
      dia=dia+
     x    nonzeros(pnt11)*s1    
      do j=1,pnt2
        knz(index(j))=knz(index(j))+
     x  nonzeros(pnt11+j)*s1       
      enddo
c
 999  return
      end
c
c ==========================================================================
c ===========================================================================
c
      subroutine supnode(ecolpnt,count,rowidx,vcstat,pivots,
     x snhead,invperm,nodtyp)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/sprnod/     psupn,ssupn,maxsnz
      integer*4          psupn,ssupn,maxsnz
c
      integer*4 ecolpnt(mn),count(mn),rowidx(cfree),vcstat(mn),
     x pivots(mn),snhead(mn),invperm(mn),nodtyp(mn)
c
      integer*4 i,j,k,l,i1,i2,ppnt1,ppnt2,pnt1,pnt2,pcol,col,snmode
      integer*4 sn1,sn2,ss1,ss2,supnz
      character*99 buff
c
   1  format(1x,'Supernodes       :',i12,'   ',i12)
   2  format(1x,'Supernodal cols. :',i12,'   ',i12) 
   3  format(1x,'Dense window     :',i12) 
c
C CMSSW: Explicit initialization needed
      j=0

      do i=1,mn
        snhead(i)=0
        invperm(i)=0
        nodtyp(mn)=0
      enddo
      do i=1,pivotn
        invperm(pivots(i))=i
      enddo
      sn1=0
      sn2=0
      ss1=0
      ss2=0
      pnt1=1
      pnt2=0
      i=0
  10  i=i+1
      if(i.le.pivotn)then
        pcol=pivots(i)
        if(vcstat(pcol).gt.-2)then
          j=0
          ppnt1=ecolpnt(pcol)
          ppnt2=count(pcol)
          k=i+1
          snmode=1
          supnz=pnt2-pnt1+1
  20      if((k.le.pivotn).and.(ppnt1.le.ppnt2))then
            col=pivots(k)
            pnt1=ecolpnt(col)
            pnt2=count(col)
            supnz=supnz+pnt2-pnt1+1
            if(((ppnt2-ppnt1-pnt2+pnt1).eq.1).and.(supnz.lt.maxsnz))then
              if(col.ne.rowidx(ppnt1))goto 30
              i2=ppnt1+1
              i1=pnt1
  40          if(i1.le.pnt2)then
                if(rowidx(i1).ne.rowidx(i2))goto 30
                i1=i1+1
                i2=i2+1
                goto 40
              endif
              k=k+1
              ppnt1=ppnt1+1
              goto 20
            endif
          endif
  30      if(k.eq.i+1)then
            snmode=-1
            supnz=pnt2-pnt1+1
  25        if((k.le.pivotn).and.(ppnt1.le.ppnt2))then
              col=pivots(k)
              pnt1=ecolpnt(col)
              pnt2=count(col)
              supnz=supnz+pnt2-pnt1+1
              if((ppnt2-ppnt1.eq.pnt2-pnt1).and.(supnz.le.maxsnz))then
                i2=ppnt1
                i1=pnt1
  45            if(i1.le.pnt2)then
                  if(rowidx(i1).ne.rowidx(i2))goto 35
                  i1=i1+1
                  i2=i2+1
                  goto 45
                endif
                k=k+1
                goto 25
              endif
            endif
          endif
  35      if(snmode.eq.1)then
            denwin=k-i
            if((k-i).lt.psupn)goto 10
            sn1=sn1+1
            ss1=ss1+(k-i)
            j=sn1
          else
            if((k-i).lt.ssupn)goto 10
            sn2=sn2+1
            ss2=ss2+(k-i)
            j=-sn2
          endif
          do l=i,k-1
            snhead(l)=j
            nodtyp(l)=j
          enddo
          i=k-1         
        endif
        goto 10
      endif
      write(buff,1)sn1,sn2
      call mprnt(buff)
      write(buff,2)ss1,ss2
      call mprnt(buff)
      write(buff,3)denwin
      call mprnt(buff)
      k=0
      do i=pivotn,1,-1
        if(snhead(i).ne.0)then
          if(k.ne.snhead(i))then
            j=i
            k=snhead(i)
          endif
          snhead(i)=j
        else
          k=0
        endif
      enddo
      return
      end
c
c ============================================================================
c  Update supernode partitions after  column fixing
c (only in the sparse part of the constraint matrix)
c =============================================================================
c
      subroutine supupd(pivots,invperm,snhead,nodtyp,vcstat,
     x ecolpnt)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 pivots(mn),invperm(mn),snhead(mn),nodtyp(mn),
     x ecolpnt(mn),vcstat(mn)
c
      integer*4 i,j,k
c 
c Make changes : Compress pivots,nodetyp,snhead
c      
      i=1
      j=0
  10  if(i.le.pivotn)then
        k=pivots(i)
        if((ecolpnt(k).gt.nz).or.(vcstat(k).gt.-2))then
          j=j+1 
          pivots(j)=pivots(i)
          snhead(j)=snhead(i)
          nodtyp(j)=nodtyp(i)         
        endif
        invperm(i)=j
        i=i+1
        goto 10
      endif
      pivotn=j
c
c Change snhead
c
      do j=1,pivotn
        if(snhead(j).gt.0)snhead(j)=invperm(snhead(j))
      enddo
c
c Create new invperm
c
      do j=1,pivotn
        invperm(pivots(j))=j
      enddo      
      return
      end
c
c =============================================================================
c Computing the starting point  xs,up in the primal space,
c                       dv, dspr,dsup in the dual   space.
c
c ===========================================================================
c
      subroutine initsol(xs,up,dv,dspr,dsup,rhs,obj,bounds,vartyp,
     x slktyp,vcstat,colpnt,ecolpnt,pivots,rowidx,nonzeros,diag,
     x updat1,count)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/initv/ prmin,upmax,dumin,stamet,safmet,premet,regul
      real*8        prmin,upmax,dumin
      integer*4     stamet,safmet,premet,regul
c
      common/mscal/ varadd,slkadd,scfree
      real*8        varadd,slkadd,scfree
c
      common/numer/  tplus,tzer
      real*8         tplus,tzer
c
      integer*4 ecolpnt(mn),vcstat(mn),colpnt(n1),rowidx(cfree),
     x pivots(mn),vartyp(n),slktyp(m),count(mn)
      real*8  xs(mn),up(mn),dv(m),dspr(mn),dsup(mn),rhs(m),obj(n),
     x bounds(mn),diag(mn),updat1(mn),nonzeros(cfree)
c
      integer*4 i,j,pnt1,pnt2
      real*8    sol,sb,spr,sdu,prlo,dulo,ngap
      logical   addall
c
c ---------------------------------------------------------------------------
c
c Reset all values
c
      do i=1,mn
        xs(i)=0.0d+0
        up(i)=0.0d+0
        dspr(i)=0.0d+0
        dsup(i)=0.0d+0
        if(i.le.m)dv(i)=0.0d+0
      enddo
c
c RHS for XS ans UP
c
      do i=1,m
        if(slktyp(i).lt.0)then
          if(bounds(i+n).gt.upmax)then
            sol=upmax/2
          else
            sol=bounds(i+n)/2
          endif
        else
          sol=0.0d+0
        endif
        updat1(i+n)=rhs(i)+sol
      enddo
      do i=1,n
        if(vartyp(i).lt.0)then
          if(bounds(i).gt.upmax)then
            sol=-upmax
          else
            sol=-bounds(i)
          endif
        else
          sol=0.0d+0
        endif
        updat1(i)=sol
      enddo
c
      call augftr(ecolpnt,
     x vcstat,rowidx,pivots,count,nonzeros,diag,updat1)
      call augbtr(ecolpnt,
     x vcstat,rowidx,pivots,count,nonzeros,diag,updat1)
c
c Initial values for xs, up
c
      do i=1,n
        if(vcstat(i).gt.-2)then
          xs(i)=updat1(i)
          if(vartyp(i).lt.0)then
            up(i)=bounds(i)-xs(i)
          endif
        endif
      enddo
      do i=1,m
        j=i+n
        if((vcstat(j).gt.-2).and.(slktyp(i).ne.0))then
          xs(j)=-updat1(j)
          if(slktyp(i).lt.0)then
            xs(j)=(bounds(j)-updat1(j))/2
            up(j)=bounds(j)-xs(j)
          endif
        endif
      enddo
c
c Initial dual variables, stamet=2
c
      if(stamet.eq.1)then
        do i=1,m
          dv(i)=0
          dspr(i+n)=0
          dsup(i+n)=0
        enddo
        do i=1,n
          if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
            if(vartyp(i).lt.0)then
              dspr(i)=obj(i)/2
              dsup(i)=-obj(i)/2
            else
              dspr(i)=obj(i)
            endif
          endif
        enddo
      else if(stamet.eq.2)then
        do i=1,m
          updat1(i+n)=0.0d+0
        enddo
        do i=1,n
          updat1(i)=obj(i)
        enddo
        call augftr(ecolpnt,
     x  vcstat,rowidx,pivots,count,nonzeros,diag,updat1)
        call augbtr(ecolpnt,
     x  vcstat,rowidx,pivots,count,nonzeros,diag,updat1)
        do i=1,m
          if(vcstat(i+n).gt.-2)then
            dv(i)=updat1(i+n)
          else
            dv(i)=0.0d+0
          endif
          if(slktyp(i).ne.0)then
            dspr(i+n)=-dv(i)
            if(slktyp(i).lt.0)then
              dspr(i+n)=-dv(i)/2
              dsup(i+n)=dv(i)/2
            endif
          endif
        enddo
        do i=1,n
          if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
            if(vartyp(i).lt.0)then
              dspr(i)=-updat1(i)
              dsup(i)=updat1(i)
            else
              dspr(i)=-updat1(i)
            endif
          endif
        enddo
      endif
c
c Compute prmin,dumin
c
      if(safmet.lt.0)then
        safmet=-safmet
        addall=.true.
      else
        addall=.false.
      endif
c
c Marsten et al.
c
      if(safmet.eq.2)then
        do i=1,m
          updat1(i)=0
        enddo
        do i=1,n
          if(vcstat(i).gt.-2)then
            pnt1=colpnt(i)
            pnt2=colpnt(i+1)-1
            sol=0.0d+0
            sb=obj(i)
            do j=pnt1,pnt2
              if(vcstat(rowidx(j)).gt.-2)then
                sol=sol+rhs(rowidx(j)-n)*nonzeros(j)
                updat1(rowidx(j)-n)=updat1(rowidx(j)-n)+nonzeros(j)*sb
              endif
            enddo
            if(prmin.lt.sol)prmin=sol
          endif
        enddo
        do i=1,m
          if(dumin.lt.abs(updat1(i)))dumin=abs(updat1(i))
        enddo
      endif
c
c Mehrotra
c
      if(safmet.eq.3)then
        spr=1.0d+0/tzer
        sdu=1.0d+0/tzer
        do i=1,mn
          if(i.le.n)then
             j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if((vcstat(i).gt.-2).and.(j.ne.0))then
            if(spr.gt.xs(i))spr=xs(i)
            if(sdu.gt.dspr(i))sdu=dspr(i)
            if(j.lt.0)then
              if(spr.gt.up(i))spr=up(i)
              if(sdu.gt.dsup(i))sdu=dsup(i)
            endif
          endif
        enddo
        spr=-1.5d+0*spr
        sdu=-1.5d+0*sdu
        if(spr.lt.0.001d+0)spr=0.001d+0
        if(sdu.lt.0.001d+0)sdu=0.001d+0
        prlo=0.0d+0
        dulo=0.0d+0
        ngap=0.0d+0
        do i=1,mn
          if(i.le.n)then
             j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if((vcstat(i).gt.-2).and.(j.ne.0))then
             sol=xs(i)+spr
             sb=dspr(i)+sdu
             ngap=ngap+sol*sb
             prlo=prlo+sol
             dulo=dulo+sb
             if(j.lt.0)then
               sol=up(i)+spr
               sb=dsup(i)+sdu
               ngap=ngap+sol*sb
               prlo=prlo+sol
               dulo=dulo+sb
             endif
          endif
        enddo
        prmin=spr+0.5d+0*ngap/dulo
        dumin=sdu+0.5d+0*ngap/prlo
      endif
      if(addall.and.(safmet.lt.3))then
        sol=1.0d+0/tzer
        sb=1.0d+0/tzer
        do i=1,mn
          if(vcstat(i).gt.-2)then
            if(i.le.n)then
              j=vartyp(i)
            else
              j=slktyp(i-n)
            endif
            if(j.ne.0)then
              if(sol.gt.xs(i))sol=xs(i)
              if(sb.gt.dspr(i))sb=dspr(i)
            endif
            if(j.lt.0)then
              if(sol.gt.up(i))sol=up(i)
              if(sb.gt.dsup(i))sb=dsup(i)
            endif
          endif
        enddo
        if(sol.lt.0)prmin=prmin-sol
        if(sb.lt.0)dumin=dumin-sb
      endif
c
c Correcting
c
      if(addall)then
        spr=1.0d+0/tzer
        sdu=1.0d+0/tzer
        sol=1.0d+0
      else
        spr=prmin
        sdu=dumin
        sol=0.0d+0
      endif
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            if(xs(i).lt.spr)then
              xs(i)=sol*xs(i)+prmin
            endif
            if(dspr(i).lt.sdu)then
              dspr(i)=sol*dspr(i)+dumin
            endif
            if(j.lt.0)then
              if(up(i).lt.spr)then
                up(i)=sol*up(i)+prmin
              endif
              if(dsup(i).lt.sdu)then
                dsup(i)=sol*dsup(i)+dumin
              endif
            endif
          endif
        endif
      enddo
c
      return
      end
c
c ===========================================================================
c
c     Set up the initial scaling matrix
c     (for the computation of the initial solution)
c
      subroutine fscale(vcstat,diag,odiag,vartyp,slktyp)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/mscal/ varadd,slkadd,scfree
      real*8        varadd,slkadd,scfree
c
      integer*4 vcstat(mn),vartyp(n),slktyp(m)
      real*8 diag(mn),odiag(mn)
c
      integer*4 i,j
      real*8 sol
c
      do i=1,mn
        sol=0.0d+0
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
            if(j.gt.0)then
              sol=-1.0d0
            else if(j.lt.0)then
              sol=-2.0d0
            else
              sol=-scfree
            endif
          else
            j=slktyp(i-n)
            if(j.gt.0)then
              sol=1.0d0
            else if(j.lt.0)then
              sol=0.5d+0
            else
              sol=0.0d+0
            endif
          endif
        endif
        diag(i)=sol
        odiag(i)=sol
      enddo
      return
      end
c
c ============================================================================
c Compute primal, upper, dual infeasibilities
c ===========================================================================
c
       subroutine cprinf(xs,prinf,slktyp,colpnt,rowidx,nonzeros,
     x  rhs,vcstat,pinf)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 slktyp(m),colpnt(n1),rowidx(nz),vcstat(mn)
      real*8    xs(mn),prinf(m),rhs(m),nonzeros(nz),pinf
c
      integer*4 i,j,pnt1,pnt2
      real*8 sol
c
c ---------------------------------------------------------------------------
c
      do i=1,m
        prinf(i)=rhs(i)
      enddo
      pinf=0.0D+0     
c
      do i=1,n
        if(vcstat(i).gt.-2)then
          sol=xs(i)
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            prinf(rowidx(j)-n)=prinf(rowidx(j)-n)-sol*nonzeros(j)
          enddo
        endif
      enddo
      do i=1,m
        if(vcstat(i+n).gt.-2)then
          if(slktyp(i).ne.0)then
            sol=prinf(i)+xs(i+n)            
          else
            sol=prinf(i)
          endif
        else
          sol=0.0d+0
        endif
        prinf(i)=sol
        if(pinf.lt.abs(sol))pinf=abs(sol)               
      enddo
      return
      end
c
c ===========================================================================
c
      subroutine cupinf(xs,up,upinf,bounds,vartyp,slktyp,vcstat,
     x uinf)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 vartyp(n),slktyp(m),vcstat(mn)
      real*8 xs(mn),up(mn),upinf(mn),bounds(mn),uinf
c
      integer*4 i
c
      do i=1,mn
        upinf(i)=0.0d+0
      enddo
      uinf=0.0d+0
      do i=1,n
        if((vcstat(i).gt.-2).and.(vartyp(i).lt.0))then         
          upinf(i)=bounds(i)-xs(i)-up(i)
          if(uinf.lt.abs(upinf(i)))uinf=abs(upinf(i))
        endif
      enddo
      do i=1,m
        if((vcstat(i+n).gt.-2).and.(slktyp(i).lt.0))then          
          upinf(i+n)=bounds(i+n)-xs(i+n)-up(i+n)
          if(uinf.lt.abs(upinf(i+n)))uinf=abs(upinf(i+n))
        endif
      enddo
      return
      end
c
c ============================================================================
c
      subroutine cduinf(dv,dspr,dsup,duinf,vartyp,slktyp,colpnt,
     x rowidx,nonzeros,obj,vcstat,dinf)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 vartyp(n),slktyp(m),colpnt(n1),rowidx(nz),
     x vcstat(mn)
      real*8 dv(m),dspr(mn),dsup(mn),duinf(mn),nonzeros(nz),obj(n),
     x dinf
c
      integer*4 i,j,pnt1,pnt2
      real*8 sol
c
c ------------------------------------------------------------------------------
c
      dinf=0.0d+0
c
      do i=1,m
        sol=0.0d+0
        if(vcstat(i+n).gt.-2)then
          if(slktyp(i).gt.0)then
            sol=dv(i)-dspr(i+n)
          else if(slktyp(i).lt.0)then
            sol=dv(i)-dspr(i+n)+dsup(i+n)
          endif
        endif
        duinf(i+n)=sol
      enddo
c
      do i=1,n
        sol=0.0d+0
        if(vcstat(i).gt.-2)then
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            if(vcstat(rowidx(j)).gt.-2)then
              sol=sol+dv(rowidx(j)-n)*nonzeros(j)
            endif
          enddo
          if(vartyp(i))10,11,12
c
c Upper bounded variable
c
  10      sol=obj(i)-sol-dspr(i)+dsup(i)
          goto 15
c
c Free variable
c
  11      sol=obj(i)-sol
          goto 15
c
c Standard variable
c
  12      sol=obj(i)-sol-dspr(i)
        endif
  15    duinf(i)=sol        
      enddo
c
c Compute absolute and relative infeasibility
c
      do i=1,mn
        sol=abs(duinf(i))
        if(dinf.lt.sol)dinf=sol
      enddo
c
      return
      end
c
c ==============================================================================
c
      subroutine cpdobj(popt,dopt,obj,rhs,bounds,xs,dv,
     x dsup,vcstat,vartyp,slktyp)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 vcstat(mn),vartyp(n),slktyp(m)
      real*8  popt,dopt,obj(n),rhs(m),bounds(mn),xs(mn),dv(m),dsup(mn)
c
      integer*4 i
c
      popt=0.0d+0
      dopt=0.0d+0
      do i=1,n
        if(vcstat(i).gt.-2)then
          popt=popt+obj(i)*xs(i)
          if(vartyp(i).lt.0)then
            dopt=dopt-bounds(i)*dsup(i)
          endif
        endif
      enddo
      do i=1,m
        if(vcstat(i+n).gt.-2)then
          dopt=dopt+rhs(i)*dv(i)
          if(slktyp(i).lt.0)then
            dopt=dopt-bounds(i+n)*dsup(i+n)
          endif
        endif
      enddo      
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine stpcrt(prelinf,drelinf,popt,dopt,cgap,
     x iter,code,pphase,dphase,maxstp,pinf,uinf,dinf,
     x prinf,upinf,duinf,oldmp,pb,db,
     x prstpl,dustpl,obj,rhs,bounds,xs,dxs,dspr,ddspr,dsup,
     x ddsup,dv,ddv,up,addobj,scobj,vcstat,vartyp,slktyp,
     x oprelinf,odrelinf,opinf,odinf,ocgap,opphas,odphas,buff)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      real*8 prelinf,drelinf,popt,dopt,cgap,maxstp,
     x pinf,uinf,oldmp,dinf,pb,db,oprelinf,odrelinf,opinf,odinf,ocgap
      integer*4 iter,code,pphase,dphase,opphas,odphas
c
      real*8 prstpl,dustpl,obj(n),rhs(m),bounds(mn),xs(mn),dxs(mn),
     x dspr(mn),ddspr(mn),dsup(mn),ddsup(mn),dv(m),ddv(m),upinf(mn),
     x up(mn),prinf(m),duinf(mn),addobj,scobj
      integer*4 vcstat(mn),vartyp(n),slktyp(m)
      character*99 buff
c
      common/toler/ tsdir,topt1,topt2,tfeas1,tfeas2,feas1,feas2,
     x              pinfs,dinfs,inftol,maxiter
      real*8        tsdir,topt1,topt2,tfeas1,tfeas2,feas1,feas2,
     x              pinfs,dinfs,inftol
      integer*4     maxiter
c
      real*8 oldpopt,olddopt,objnrm,rhsnrm,bndnrm,urelinf,mp
      integer*4 i
c
      prelinf=0.0d+0
      urelinf=0.0d+0
      drelinf=0.0d+0
      objnrm =0.0d+0
      rhsnrm =0.0d+0
      bndnrm =0.0d+0

      do i=1,n
        if(vcstat(i).gt.-2)then
          objnrm=objnrm+obj(i)*obj(i)
          drelinf=drelinf+duinf(i)*duinf(i)
          if(vartyp(i).lt.0)then
            bndnrm=bndnrm+bounds(i)*bounds(i)
            urelinf=urelinf+upinf(i)*upinf(i)
          endif
        endif
      enddo
      do i=1,m
        if(vcstat(i+n).gt.-2)then
          rhsnrm=rhsnrm+rhs(i)*rhs(i)
          prelinf=prelinf+prinf(i)*prinf(i)
          drelinf=drelinf+duinf(i+n)*duinf(i+n)
          if(slktyp(i).lt.0)then
            bndnrm=bndnrm+bounds(i+n)*bounds(i+n)
            urelinf=urelinf+upinf(i+n)*upinf(i+n)
          endif
        endif
      enddo
c
      prelinf=sqrt(prelinf+urelinf)/(1.0d+0+sqrt(bndnrm+rhsnrm))
      drelinf=sqrt(drelinf)/(1.0d+0+sqrt(objnrm))
      if(drelinf.gt.dinf)drelinf=dinf
      if(prelinf.gt.max(pinf,uinf))prelinf=max(pinf,uinf)
c
      mp=prelinf+drelinf+
     x abs(popt-dopt)/scobj/(1.0d+0+sqrt(rhsnrm+bndnrm)+sqrt(objnrm))
      if(iter.le.1)oldmp=mp
c
      code=0
      if((prelinf.lt.tfeas1).and.
     x  (pinf.lt.feas1).and.(uinf.lt.feas1))then
        pphase=2
      else
        pphase=1
        pb=abs(pb-pinf)/(abs(pinf))
      endif
      if((drelinf.lt.tfeas2).and.(dinf.lt.feas2))then
        dphase=2
      else
        dphase=1
        db=abs(db-dinf)/(abs(dinf))
      endif
c
      if((abs(popt-dopt)/(abs(popt)+1.0d+0).le.topt1)
     x. and.(pphase.eq.2).and.(dphase.eq.2))then
        code=2
        write(buff,'(1x,a)')
     x  'Stopping criterion : Small infeasibility and duality gap'
      else if((popt.lt.dopt).and.(pphase.eq.2).and.(dphase.eq.2))then
        code=0
        if(iter.gt.0)then
          call cpdobj(oldpopt,olddopt,obj,rhs,bounds,dxs,ddv,ddsup,
     x    vcstat,vartyp,slktyp)
          oldpopt=popt-oldpopt*scobj*prstpl
          olddopt=dopt-olddopt*scobj*dustpl
          if(oldpopt.ge.olddopt)then
            code=2
            maxstp=1.0d+0-(oldpopt-olddopt)/(dopt-olddopt-popt+oldpopt)
            dustpl=-maxstp*dustpl
            prstpl=-maxstp*prstpl
            call cnewpd(prstpl,xs,dxs,up,upinf,dustpl,dv,ddv,dspr,
     x      ddspr,dsup,ddsup,vartyp,slktyp,vcstat,maxstp)
            call cpdobj(popt,dopt,obj,rhs,bounds,xs,dv,dsup,
     x      vcstat,vartyp,slktyp)
            popt=popt*scobj+addobj
            dopt=dopt*scobj+addobj
          endif
        endif
        if(code.gt.0)then
          write(buff,'(1x,a)')
     x    'Stopping criterion : Small infeasibility and duality gap'
        else
          write(buff,'(1x,a)')
     x    'Stopping criterion : Negative gap (Wrong tolerances ?)'
          code=1
        endif
      else if((mp.gt.topt1).and.(mp.gt.inftol*oldmp))then
        if(pphase+dphase.eq.4)then
          code=1
          write(buff,'(1x,a)')
     x    'Stopping Criterion: Possible numerical problems'
        else if (opphas+odphas.eq.4)then
          code=1
          write(buff,'(1x,a)')
     x    'Stopping Criterion: Instability, Suboptimal solution'
          dustpl=-dustpl
          prstpl=-prstpl
          call cnewpd(prstpl,xs,dxs,up,upinf,dustpl,dv,ddv,dspr,
     x    ddspr,dsup,ddsup,vartyp,slktyp,vcstat,maxstp)
          call cpdobj(popt,dopt,obj,rhs,bounds,xs,dv,dsup,
     x    vcstat,vartyp,slktyp)
          call cpdobj(popt,dopt,obj,rhs,bounds,xs,dv,dsup,
     x    vcstat,vartyp,slktyp)
          popt=popt*scobj+addobj
          dopt=dopt*scobj+addobj
          prelinf=oprelinf
          drelinf=odrelinf
          pinf=opinf
          dinf=odinf
          pphase=opphas
          dphase=odphas
          cgap=ocgap
        else
          write(buff,'(1x,a)')
     x    'Stopping Criterion: Problem infeasibile'
          code=4
          if(pphase.eq.2)code=3
        endif
      else if(abs(cgap).lt.topt2)then
        code=1
        if((pphase.eq.2).and.(dphase.eq.2))code=2
        write(buff,'(1x,a)')
     x  'Stopping Criterion : Small complementarity gap'
      else if(iter.ge.maxiter)then
        code=1
        write(buff,'(1x,a)')
     x  'Stopping Criterion : Iteration limit is exeeded'
      else if(maxstp.lt.tsdir)then
        code=1
        write(buff,'(1x,a)')
     x  'Stopping Criterion : Very small step'
      else if((iter.gt.0).and.(pphase.eq.1).and.(pb.lt.pinfs))then
        code=4
        write(buff,'(1x,a)')
     x  'Stopping Criterion: Pinfs limit. Problem primal infeasibile'
      else if((iter.gt.0).and.(dphase.eq.1).and.(db.lt.dinfs))then
        code=3
        write(buff,'(1x,a)')
     x  'Stopping Criterion: Dinfs limit. Problem dual infeasibile'
      endif
      if(oldmp.gt.mp)oldmp=mp
      oprelinf=prelinf
      odrelinf=drelinf
      opinf=pinf
      odinf=dinf
      opphas=pphase
      odphas=dphase
      ocgap=cgap
      return
      end
c
c ===========================================================================
c Compute the primal and dual steplengts
c
c ===========================================================================
c
      subroutine cstpln(prstpl,xs,dxs,up,upinf,
     x  dustpl,dspr,ddspr,dsup,ddsup,vartyp,slktyp,vcstat)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      common/param/ palpha,dalpha
      real*8        palpha,dalpha
c
      integer*4 vartyp(n),slktyp(m),vcstat(mn)
      real*8 prstpl,xs(mn),dxs(mn),up(mn),upinf(mn),
     x dustpl,dspr(mn),ddspr(mn),dsup(mn),ddsup(mn)
c
      integer*4 i,j
      real*8 sol,dup
c
      prstpl=1.0d0/palpha
      dustpl=1.0d0/dalpha
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            if(dxs(i).lt.0.0d+0)then
              sol=-xs(i)/dxs(i)
              if(sol.lt.prstpl)prstpl=sol
            endif
            if(ddspr(i).lt.0.0d+0)then
              sol=-dspr(i)/ddspr(i)
              if(sol.lt.dustpl)dustpl=sol
            endif
            if (j.lt.0)then
              dup=upinf(i)-dxs(i)
              if(dup.lt.0.0d+0)then
                sol=-up(i)/dup
                if(sol.lt.prstpl)prstpl=sol
              endif
              if(ddsup(i).lt.0.0d+0)then
                sol=-dsup(i)/ddsup(i)
                if(sol.lt.dustpl)dustpl=sol
              endif
            endif
          endif
        endif
      enddo
      return
      end
c
c ===========================================================================
c Compute the new primal and dual solution
c
c ===========================================================================
c
      subroutine cnewpd(prstpl,xs,dxs,up,upinf,dustpl,dv,ddv,
     x dspr,ddspr,dsup,ddsup,vartyp,slktyp,vcstat,maxd)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 vartyp(n),slktyp(m),vcstat(mn)
      real*8 prstpl,xs(mn),dxs(mn),up(mn),upinf(mn),dustpl,dv(m),
     x ddv(m),dspr(mn),ddspr(mn),dsup(mn),ddsup(mn),maxd
c
      integer*4 i,j
      real*8 maxdd,maxdp
c
      maxdp=0.0d+0
      maxdd=0.0d+0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
            dv(i-n)=dv(i-n)+dustpl*ddv(i-n)
            if(maxdd.lt.abs(ddv(i-n)))maxdd=abs(ddv(i-n))
          endif
          if((i.le.n).or.(j.ne.0))then
            xs(i)=xs(i)+prstpl*dxs(i)
            if(maxdp.lt.abs(dxs(i)))maxdp=abs(dxs(i))
            dspr(i)=dspr(i)+dustpl*ddspr(i)
            if(maxdd.lt.abs(ddspr(i)))maxdd=abs(ddspr(i))
          endif
          if (j.lt.0)then
            up(i)=up(i)+prstpl*(upinf(i)-dxs(i))
            if(maxdp.lt.abs(upinf(i)-dxs(i)))maxdp=abs(upinf(i)-dxs(i))
            dsup(i)=dsup(i)+dustpl*ddsup(i)
            if(maxdd.lt.abs(ddsup(i)))maxdd=abs(ddsup(i))
          endif
        endif
      enddo
      maxd=max(maxdp*prstpl,maxdd*dustpl)
      return
      end
c
c ===========================================================================
c Fixing variables and dropping rows
c ===========================================================================
c
      subroutine varfix(vartyp,slktyp,rhs,colpnt,rowidx,nonzeros,
     x xs,up,dspr,dsup,vcstat,fixn,dropn,addobj,scobj,obj,bounds,
     x duinf,dinf,fxp,fxd,fxu)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/drop/  tfixvar,tfixslack,slklim
      real*8        tfixvar,tfixslack,slklim
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      common/mscal/ varadd,slkadd,scfree
      real*8        varadd,slkadd,scfree
c
      integer*4 colpnt(n1),vartyp(n),slktyp(m),rowidx(nz),
     x vcstat(mn),fixn,dropn,fxp,fxd,fxu
      real*8 rhs(m),nonzeros(nz),xs(mn),up(mn),addobj,scobj,obj(n),
     x dspr(mn),dsup(mn),bounds(mn),duinf(mn),dinf
c
      integer*4 i,j,pnt1,pnt2
      real*8    sol
c
c ---------------------------------------------------------------------------
c
      fxp=0
      fxd=0
      fxu=0
      do i=1,n
        if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
          if((xs(i).lt.tfixvar).or.
     x       ((vartyp(i).lt.0).and.(up(i).lt.tfixvar)))then
            fixn=fixn+1
            fxp=fxp+1
            vcstat(i)=-2
            if(xs(i).lt.tfixvar)then
             xs(i)=0.0d+0
             up(i)=bounds(i) 
            else
             xs(i)=bounds(i)
             up(i)=0.0d+0           
            endif  
            sol=xs(i)
            pnt1=colpnt(i)
            pnt2=colpnt(i+1)-1
            do j=pnt1,pnt2
              rhs(rowidx(j)-n)=rhs(rowidx(j)-n)-sol*nonzeros(j)
            enddo
            addobj=addobj+scobj*obj(i)*sol
          endif
          if (dspr(i).lt.tfixslack)then
            fxd=fxd+1
            duinf(i)=duinf(i)-slklim+dspr(i) 
            dspr(i)=slklim
          endif  
        endif
      enddo
c
c Release upper bounds
c
      do i=1,mn
        if(i.le.n)then
          j=vartyp(i)
        else
          j=slktyp(i-n)
        endif
        if((vcstat(i).gt.-2).and.(j.lt.0))then
           if(dsup(i).lt.slklim)then
             fxu=fxu+1
             duinf(i)=duinf(i)-dsup(i)
             dsup(i)=0            
             if(i.le.n)then
               vartyp(i)=-j
             else
               slktyp(i-n)=-j
             endif
           endif
        endif
      enddo
c
c Relax rows
c
      do i=1,m
        j=i+n
        if((vcstat(j).gt.-2).and.(slktyp(i).gt.0))then
          if(dspr(j).lt.tfixslack)then
            fxd=fxd+1
            dropn=dropn+1
            vcstat(j)=-2
          endif
        endif
      enddo
c
c Compute new dual infeasibility
c
      if((fxd.gt.0).or.(fxu.gt.0))then
         dinf=0.0d+0
         do i=1,mn
           if(vcstat(i).gt.-2)then
             if(abs(duinf(i)).gt.dinf)dinf=abs(duinf(i))
           endif
         enddo
      endif
c
      return
      end
c
c ===========================================================================
c Modifying the primal and dual variables
c ===========================================================================
c
      subroutine pdmodi(xs,dspr,vcstat,
     x vartyp,slktyp,gap,pobj,dobj,prinf,duinf,upinf,
     x colpnt,rowidx,rownz,pinf,uinf,dinf)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/compl/ climit,ccorr
      real*8        climit,ccorr
c
      integer*4 vcstat(mn),vartyp(n),slktyp(m),colpnt(n1),rowidx(nz)
      real*8 xs(mn),dspr(mn),gap,pobj,dobj,
     x prinf(m),upinf(mn),duinf(mn),rownz(nz),pinf,uinf,dinf
c
      integer*4 i,j,k,prm,dum,upm,pnt1,pnt2
      real*8 sp,sd,sol,s
c
c --------------------------------------------------------------------------
c
      prm=0
      dum=0
      upm=0
      sd=gap
      sp=abs(pobj-dobj)/(abs(pobj)+1.0d0)
      sd=sd*ccorr
      if(sd.gt.climit)sd=climit
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            sp=xs(i)*dspr(i)
            if(sp.lt.sd)then
              if(xs(i).gt.dspr(i))then
                sol=sd/xs(i)
                duinf(i)=duinf(i)+dspr(i)-sol
                dspr(i)=sol
                dum=dum+1
              else
                sol=sd/dspr(i)
                s=xs(i)-sol
                xs(i)=sol
                if(j.lt.0)then
                  upinf(i)=upinf(i)+s
                  upm=upm+1
                endif
                if(i.le.n)then
                  pnt1=colpnt(i)
                  pnt2=colpnt(i+1)-1
                  do k=pnt1,pnt2
                    prinf(rowidx(k)-n)=prinf(rowidx(k)-n)+s*rownz(k)
                  enddo
                else
                  prinf(i-n)=prinf(i-n)-s
                endif
                prm=prm+1
              endif
            endif
ccc
ccc It's totally wrong! Do not modify upper bounds !
ccc
ccc              if(j.lt.0)then
ccc                sp=up(i)*dsup(i)
ccc                if(sp.lt.sd)then
ccc                  if(up(i).gt.dsup(i))then
ccc                    sol=sd/up(i)
ccc                    duinf(i)=duinf(i)-dsup(i)+sol
ccc                    dsup(i)=sol
ccc                    dum=dum+1
ccc                  else
ccc                    sol=sd/dsup(i)
ccc                    upinf(i)=upinf(i)+up(i)-sol
ccc                    up(i)=sol
ccc                    upm=upm+1
ccc                  endif
ccc                endif
ccc              endif
            endif
        endif
      enddo
c
c Correct infeas. norm
c
      if(prm.gt.0)then
        pinf=0.0d+0
        do i=1,m
          if(vcstat(i+n).gt.-2)then
            if(abs(prinf(i)).gt.pinf)pinf=abs(prinf(i))
          else
            prinf(i)=0.0d+0
         endif
       enddo
      endif
      if(upm.gt.0)then
        uinf=0.0d+0
        do i=1,mn
          if(vcstat(i).gt.-2)then
            if(abs(upinf(i)).gt.uinf)uinf=abs(upinf(i))
          else
            upinf(i)=0.0d+0
          endif
        enddo
      endif
      if(dum.gt.0)then
        dinf=0.0d+0
        do i=1,mn
          if(vcstat(i).gt.-2)then
            if(abs(duinf(i)).gt.dinf)dinf=abs(duinf(i))
          else
            duinf(i)=0.0d+0
          endif
        enddo
      endif
      return
      end
c
c ===========================================================================
c Scaling of free variables : "Average" of basics * scfree
c Correcting : "Average" of basics * varadd
c
c ===========================================================================
c
      subroutine cdiag(xs,up,dspr,dsup,vartyp,slktyp,vcstat,diag,
     x odiag)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/mscal/ varadd,slkadd,scfree
      real*8        varadd,slkadd,scfree
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      integer*4 vartyp(n),slktyp(m),vcstat(mn)
      real*8 xs(mn),up(mn),dspr(mn),dsup(mn),diag(mn),odiag(mn)
c
      integer*4 i,j
      real*8 sol,sn,sm,mins
c
c ---------------------------------------------------------------------------
c
      sn=0.0d+0
      mins=1.0d+0
      j=0
      do i=1,n
        sol=0.0d+0
        if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
          if(vartyp(i).lt.0)then
            sol=dspr(i)/xs(i)+dsup(i)/up(i)
          else
            sol=dspr(i)/xs(i)
          endif
c
c Compute average on "basic" variables
c
          if(mins.gt.sol)mins=sol
          if(vcstat(i).gt.0)then
            j=j+1
            sn=sn+log(sol)
          endif
        endif
        diag(i)=sol
        odiag(i)=sol
      enddo
c
c Compute geometric mean of the "basics"
c
      if(j.eq.0)j=1
      sol=exp(sn/dble(j))
c
c Set scale parameter for free variables
c
      if(abs(scfree).lt.tzer)then
        sn=0.0d+0
      else if(scfree.lt.0.0d+0)then
        sn=-scfree
      else
        sn=max(sol*scfree,mins)
      endif
c
c Set regularization parameter
c
      if(abs(varadd).lt.tzer)then
        sm=0.0d+0
      else if(varadd.lt.0.0d+0)then
        sm=-varadd
      else
        sm=sol*varadd
      endif
c
c Second pass: Set free variables and regularize
c
      do i=1,n
        if(vcstat(i).gt.-2)then
          if(vartyp(i).eq.0)then
            sol=sn
          else
            sol=diag(i)
          endif
ccc          if(sol.lt.sm*sm)sol=sm*sqrt(sol)
          if(sol.lt.sm)sol=sm*sqrt(sol/sm)
          diag(i)=-sol
          odiag(i)=-sol
        endif
      enddo
c
c
c
      j=0
      sn=0.0d+0
      do i=1,m
        sol=0.0d+0
        if(vcstat(i+n).gt.-2)then
          if(slktyp(i).eq.0)then
            sol=0.0d+0
          else
            if(slktyp(i).lt.0)then
              sol=1.0d+0/(dspr(i+n)/xs(i+n)+dsup(i+n)/up(i+n))+0.0d+0
            else
              sol=xs(i+n)/dspr(i+n)
            endif
            if(vcstat(i+n).gt.0)then
              j=j+1
              sn=sn+log(sol)
            endif
          endif
        endif
        diag(i+n)=sol
        odiag(i+n)=sol
      enddo

      if(j.eq.0)j=1
      if(abs(slkadd).lt.tzer)then
        sm=0.0d+0
      else if(slkadd.lt.0.0d+0)then
        sm=-slkadd
      else
        sm=exp(sn/dble(j))*slkadd
      endif

      if(sm.gt.tzer)then
        do i=1,m
          if(vcstat(i+n).gt.-2)then
            sol=diag(i+n)
ccc            if(sol.gt.sm*sm)sol=sm*sqrt(sol)
            if(sol.gt.sm)sol=sm*sqrt(sol/sm)
            diag(i+n)=sol
            odiag(i+n)=sol
          endif
        enddo
      endif
      return
      end
c
c ===========================================================================
c Multi predictor-corrector direction (Merothra)
c
c ===========================================================================
c
      subroutine cpdpcd(xs,up,dspr,dsup,prinf,duinf,upinf,
     x dxsn,ddvn,ddsprn,ddsupn,dxs,ddv,ddspr,ddsup,bounds,
     x ecolpnt,count,pivots,vcstat,diag,odiag,rowidx,nonzeros,
     x colpnt,vartyp,slktyp,barpar,corr,prstpl,dustpl,barn,cgap)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree

      common/numer/ tplus,tzer
      real*8        tplus,tzer

      common/predp/ ccstop,barset,bargrw,barmin,mincor,maxcor,inibar
      real*8        ccstop,barset,bargrw,barmin
      integer*4     mincor,maxcor,inibar
c
      integer*4 ecolpnt(mn),count(mn),vcstat(mn),rowidx(cfree),
     x pivots(mn),colpnt(n1),vartyp(n),slktyp(m),corr,barn
      real*8 xs(mn),up(mn),dspr(mn),dsup(mn),prinf(m),duinf(mn),
     x upinf(mn),dxsn(mn),ddvn(m),ddsprn(mn),ddsupn(mn),
     x dxs(mn),ddv(m),ddspr(mn),ddsup(mn),bounds(mn),
     x diag(mn),odiag(mn),nonzeros(cfree),barpar,prstpl,dustpl,cgap
c
      integer*4 i,j,cr,mxcor
      real*8 sol,sb,ogap,ngap,obpar,ostp,ostd
c
c ---------------------------------------------------------------------------
c
c Compute ogap
c
      ogap=cgap
      if(barpar.lt.tzer)barpar=ogap/dble(barn)*barset
      obpar=barpar
      if(inibar.le.0)then
        barpar=0.0d+0
      else
        barpar=ogap/dble(barn)*barset
        if(barpar.gt.obpar*bargrw)barpar=obpar*bargrw
      endif
c
      cr=0
      mxcor=maxcor
c
c Initialize : Reset
c
      do i=1,m
        ddv(i)=0.0d+0
      enddo
      do i=1,mn
        dxs(i)=0.0d+0
        ddspr(i)=0.0d+0
        ddsup(i)=0.0d+0
      enddo
c
c Affine scaling / primal-dual direction
c
      do i=1,n
        sol=0.0d+0
        if(vcstat(i).gt.-2)then
          if(vartyp(i))10,11,12
  10      sol=duinf(i)+dspr(i)-barpar/xs(i)
     x    -dsup(i)+(barpar-dsup(i)*upinf(i))/up(i)
          goto 15
  11      sol=duinf(i)
          goto 15
  12      sol=duinf(i)+dspr(i)-barpar/xs(i)
        endif
  15    dxsn(i)=sol
      enddo
c
      do i=1,m
       j=i+n
       sol=0.0d+0
       if(vcstat(j).gt.-2)then
         if(slktyp(i))20,21,22
  20     sol=-(duinf(j)+dspr(j)-barpar/xs(j)
     x   -dsup(j)+(barpar-dsup(j)*upinf(j))/up(j))*odiag(j)
         goto 25
  21     sol=0.0d+0
         goto 25
  22     sol=-(duinf(j)+dspr(j)-barpar/xs(j))*odiag(j)
       endif
  25   dxsn(j)=prinf(i)+sol
      enddo
c
c Solve the augmented system
c
      if(cr.lt.mincor)then
        call augftr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,dxsn)
        call augbtr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,dxsn)
      else
        call citref(diag,odiag,pivots,rowidx,nonzeros,colpnt,
     x  ecolpnt,count,vcstat,dxsn,ddsprn,ddsupn,upinf,
     x  bounds,xs,up,vartyp,slktyp)
      endif
c
c Primal and dual variables
c Primal slacks : ds=D_s^{-1}*(b_s+dy)
c
      do i=1,m
        j=i+n
        if(vcstat(j).gt.-2)then
          ddvn(i)=dxsn(j)
          if(slktyp(i).ne.0)then
            if(slktyp(i).gt.0)then
              sb=duinf(j)+dspr(j)-barpar/xs(j)
            else
              sb=duinf(j)+dspr(j)-barpar/xs(j)
     x        -dsup(j)+(barpar-dsup(j)*upinf(j))/up(j)
            endif
            dxsn(j)=-odiag(j)*(ddvn(i)+sb)
          endif
        endif
      enddo
c
c Primal upper bounds, dual slacks
c dz=-Z+X^{-1}(mu -dx*dz -Z*dx)
c
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.lt.0)then
            ddsupn(i)=-dsup(i)+(barpar-dsup(i)*(upinf(i)-dxsn(i)))/up(i)
          endif
          if(j.ne.0)then
            ddsprn(i)=-dspr(i)+(barpar-dspr(i)*dxsn(i))/xs(i)
          else if(i.le.n)then
            ddsprn(i)=-dspr(i)
          endif
        endif
      enddo
c
c Compute primal and dual steplengths
c
      call cstpln(prstpl,xs,dxsn,up,upinf,
     x dustpl,dspr,ddsprn,dsup,ddsupn,vartyp,slktyp,vcstat)
c
c Estimate basic variables vcstat(i)=1 for basic, 0 for nonbasic
c
      do i=1,n
        if((vcstat(i).gt.-2).and.(vartyp(i).ne.0))then
          if(abs(ddsprn(i))*xs(i).gt.abs(dxsn(i))*dspr(i))then
            vcstat(i)=1
          else
            vcstat(i)=0
          endif
        endif
      enddo
      do i=1,m
        if((vcstat(i+n).gt.-2).and.(slktyp(i).ne.0))then
          if(abs(ddsprn(i+n))*xs(i+n).gt.abs(dxsn(i+n))*dspr(i+n))then
            vcstat(i+n)=1
          else
            vcstat(i+n)=0
          endif
        endif
      enddo
c
c Compute ngap
c
      ngap=0.0d+0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            ngap=ngap+(xs(i)+prstpl*dxsn(i))*(dspr(i)+dustpl*ddsprn(i))
            if(j.lt.0)then
              ngap=ngap+(up(i)+prstpl*(upinf(i)-dxsn(i)))*
     x        (dsup(i)+dustpl*ddsupn(i))
            endif
          endif
        endif
      enddo
      cgap=ngap/dble(barn)
      ostp=prstpl
      ostd=dustpl
      do i=1,mn
        dxs(i)=dxsn(i)
        ddspr(i)=ddsprn(i)
        ddsup(i)=ddsupn(i)
      enddo
      do i=1,m
        ddv(i)=ddvn(i)
      enddo
c
c Compute barrier
c
      barpar=ngap*ngap*ngap/(ogap*ogap*dble(barn))
      if(barpar.gt.ogap/dble(barn)*barset)barpar=ogap/dble(barn)*barset
      if(barpar.gt.obpar*bargrw)barpar=obpar*bargrw
      if(barpar.lt.barmin)barpar=0.0d+0
      if(mxcor.le.0)goto 999
c
c Higher order predictor-corrector direction
c
  50  cr=cr+1
      do i=1,n
        sol=0.0d+0
        if(vcstat(i).gt.-2)then
          if(vartyp(i))30,31,32
  30      sol=duinf(i)+dspr(i)+(ddspr(i)*dxs(i)-barpar)/xs(i)
     x    -dsup(i)-(ddsup(i)*(upinf(i)-dxs(i))-barpar+dsup(i)*
     x    upinf(i))/up(i)
          goto 35
  31      sol=duinf(i)
          goto 35
  32      sol=duinf(i)+dspr(i)+(ddspr(i)*dxs(i)-barpar)/xs(i)
        endif
  35    dxsn(i)=sol
      enddo
c
      do i=1,m
       j=i+n
       sol=0.0d+0
       if(vcstat(j).gt.-2)then
         if(slktyp(i))40,41,42
  40     sol=-(duinf(j)+dspr(j)+(ddspr(j)*dxs(j)-barpar)/xs(j)
     x   -dsup(j)-(ddsup(j)*(upinf(j)-dxs(j))-barpar+dsup(j)*
     x   upinf(j))/up(j))*odiag(j)
         goto 45
  41     sol=0.0d+0
         goto 45
  42     sol=-(duinf(j)+dspr(j)+(ddspr(j)*dxs(j)-barpar)/xs(j))*odiag(j)
       endif
  45   dxsn(j)=prinf(i)+sol
      enddo
c
c Solve the augmented system
c
      if(cr.lt.mincor)then
        call augftr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,dxsn)
        call augbtr(ecolpnt,vcstat,rowidx,pivots,count,nonzeros,
     x  diag,dxsn)
      else
        call citref(diag,odiag,pivots,rowidx,nonzeros,colpnt,
     x  ecolpnt,count,vcstat,dxsn,ddsprn,ddsupn,upinf,
     x  bounds,xs,up,vartyp,slktyp)
      endif
c
c Primal and dual variables
c Primal slacks : ds=D_s^{-1}*(b_s+dy)
c
      do i=1,m
        j=i+n
        if(vcstat(j).gt.-2)then
          ddvn(i)=dxsn(j)
          if(slktyp(i).ne.0)then
            if(slktyp(i).gt.0)then
              sb=duinf(j)+dspr(j)+(ddspr(j)*dxs(j)-barpar)/xs(j)
            else
              sb=duinf(j)+dspr(j)+(ddspr(j)*dxs(j)-barpar)/xs(j)
     x        -dsup(j)-(ddsup(j)*(upinf(j)-dxs(j))-barpar+dsup(j)*
     x         upinf(j))/up(j)
            endif
            dxsn(j)=-odiag(j)*(ddvn(i)+sb)
          endif
        endif
      enddo
c
c Primal upper bounds, dual slacks
c dz=-Z+X^{-1}(mu -dx*dz -Z*dx)
c
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.lt.0)then
            ddsupn(i)=
     x       -dsup(i)+(barpar-ddsup(i)*(upinf(i)-dxs(i))
     x       -dsup(i)*(upinf(i)-dxsn(i)))/up(i)
          endif
          if(j.ne.0)then
            ddsprn(i)=
     x       -dspr(i)+(barpar-ddspr(i)*dxs(i)-dspr(i)*dxsn(i))/xs(i)
          else if(i.le.n)then
            ddsprn(i)=-dspr(i)
          endif
        endif
      enddo
c
c Compute primal and dual steplengths
c
      call cstpln(prstpl,xs,dxsn,up,upinf,dustpl,dspr,
     x ddsprn,dsup,ddsupn,vartyp,slktyp,vcstat)
c
c Compute ngap
c
      ngap=0.0d+0
      do i=1,mn
        if(vcstat(i).gt.-2)then
          if(i.le.n)then
            j=vartyp(i)
          else
            j=slktyp(i-n)
          endif
          if(j.ne.0)then
            ngap=ngap+(xs(i)+prstpl*dxsn(i))*(dspr(i)+dustpl*ddsprn(i))
            if(j.lt.0)then
              ngap=ngap+(up(i)+prstpl*(upinf(i)-dxsn(i)))*
     x        (dsup(i)+dustpl*ddsupn(i))
            endif
          endif
        endif
      enddo
c
c Check corrections criteria
c
      if(cr.gt.mincor)then
        if(min(prstpl,dustpl).lt.ccstop*min(ostp,ostd))then
          if(min(prstpl,dustpl).lt.min(ostp,ostd))then
            prstpl=ostp
            dustpl=ostd
            cr=cr-1
            goto 999
          else
            mxcor=cr
          endif
        endif
      endif
c
c Continue correcting, change the actual search direction
c
      cgap=ngap/dble(barn)
      ostp=prstpl
      ostd=dustpl
      do i=1,mn
        dxs(i)=dxsn(i)
        ddspr(i)=ddsprn(i)
        ddsup(i)=ddsupn(i)
      enddo
      do i=1,m
        ddv(i)=ddvn(i)
      enddo
      if(cr.ge.mxcor)goto 999
      goto 50
c
c End of the correction loop, save the number of the corrections
c
 999  corr=cr
      return
      end
c
c ============================================================================
c Multi-centrality corrections
c
c ===========================================================================
c
      subroutine cpdccd(xs,up,dspr,dsup,upinf,
     x dxsn,ddvn,ddsprn,ddsupn,dxs,ddv,ddspr,ddsup,bounds,
     x ecolpnt,count,pivots,vcstat,diag,odiag,rowidx,nonzeros,
     x colpnt,vartyp,slktyp,barpar,corr,prstpl,dustpl)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree

      common/numer/ tplus,tzer
      real*8        tplus,tzer

      common/predc/ target,tsmall,tlarge,center,corstp,mincc,maxcc
      real*8        target,tsmall,tlarge,center,corstp
      integer*4     mincc,maxcc
c
      integer*4 ecolpnt(mn),count(mn),vcstat(mn),rowidx(cfree),
     x pivots(mn),colpnt(n1),vartyp(n),slktyp(m),corr
      real*8 xs(mn),up(mn),dspr(mn),dsup(mn),
     x upinf(mn),dxsn(mn),ddvn(m),ddsprn(mn),ddsupn(mn),
     x dxs(mn),ddv(m),ddspr(mn),ddsup(mn),bounds(mn),
     x diag(mn),odiag(mn),nonzeros(cfree),barpar,prstpl,dustpl
c
      integer*4 i,j,cr,maxccx
      real*8 s,ss,ostp,ostd,prs,dus,dp
c
c ---------------------------------------------------------------------------
      maxccx=maxcc
      cr=0
      ostp=prstpl
      ostd=dustpl
      if(maxcc.le.0)goto 999      
      cr=1
c
c Define Target
c
   1  prs=prstpl*(target+1.0d+0)+target
      dus=dustpl*(target+1.0d+0)+target
      if (prs.ge.1.0d+0)prs=1.0d+0
      if (dus.ge.1.0d+0)dus=1.0d+0

      do 10 j=1,n
        if(vcstat(j).le.-2)then
          dxsn(j)=0.0d+0
          goto 10
        endif
        if(vartyp(j).eq.0)then
          dxsn(j)=0.0d+0
          goto 10
        endif
        dp=(xs(j)+prs*dxs(j))*(dspr(j)+dus*ddspr(j))
        if (dp.le.tsmall*barpar)then
          s=barpar-dp
        else if(dp.ge.tlarge*barpar)then
          s=-center*barpar
        else
          s=0.0d+0
        endif

        if(vartyp(j).gt.0)then
          dxsn(j)=-s/xs(j)
          goto 10
        endif

        dp=(up(j)+prs*(upinf(j)-dxs(j)))*(dsup(j)+dus*ddsup(j))
        if (dp.le.tsmall*barpar)then
          ss=barpar-dp
        else if(dp.ge.tlarge*barpar)then
          ss=-center*barpar
        else
          ss=0.0d+0
        endif
        dxsn(j)=-s/xs(j)+ss/up(j)
  10  continue
c
      do 20 i=1,m
        j=i+n
        if(vcstat(j).le.-2)then
          dxsn(j)=0.0d+0
          goto 20
        endif
        if(slktyp(i).eq.0)then
          dxsn(j)=0.0d+0
          goto 20
        endif
c
c Bounded variable
c
        dp=(xs(j)+prs*dxs(j))*(dspr(j)+dus*ddspr(j))
        if (dp.le.tsmall*barpar)then
          s=barpar-dp
        else if (dp.ge.tlarge*barpar)then
          s=-center*barpar
        else
          s=0.0d+0
        endif
        if(slktyp(i).gt.0)then
          dxsn(j)=s/xs(j)*odiag(j)
          goto 20
        endif
c
c upper bounded variable
c
        dp=(up(j)+prs*(upinf(j)-dxs(j)))*(dsup(j)+dus*ddsup(j))
        if (dp.le.tsmall*barpar)then
          ss=barpar-dp
        else if(dp.ge.tlarge*barpar)then
          ss=-center*barpar
        else
          ss=0.0d+0
        endif
        dxsn(j)=(s/xs(j)-ss/up(j))*odiag(j)
  20  continue
c
c solve the augmented system
c
      call citref(diag,odiag,pivots,rowidx,nonzeros,colpnt,
     x ecolpnt,count,vcstat,dxsn,ddsprn,ddsupn,upinf,
     x bounds,xs,up,vartyp,slktyp)
c
c Primal and dual variables
c
      do 30 i=1,m
        j=i+n
        if(vcstat(j).le.-2)goto 30
        ddvn(i)=ddv(i)+dxsn(j)
        if(slktyp(i).eq.0)goto 30
        dp=(xs(j)+prs*dxs(j))*(dspr(j)+dus*ddspr(j))
        if (dp.le.tsmall*barpar)then
          s=barpar-dp
        else if (dp.ge.tlarge*barpar)then
          s=-center*barpar
        else
          s=0.0d+0
        endif
        if(slktyp(i).gt.0)then
          dxsn(j)=-odiag(j)*(dxsn(j)-s/xs(j))
          goto 30
        endif
        dp=(up(j)+prs*(upinf(j)-dxs(j)))*(dsup(j)+dus*ddsup(j))
        if (dp.le.tsmall*barpar)then
          ss=barpar-dp
        else if(dp.ge.tlarge*barpar)then
          ss=-center*barpar
        else
          ss=0.0d+0
        endif
        dxsn(j)=-odiag(j)*(dxsn(j)-s/xs(j)+ss/up(j))
  30  continue
c
c Primal upper bounds, dual slacks
c
      do 40 i=1,mn
        if(vcstat(i).le.-2)goto 40
        if(i.le.n)then
          j=vartyp(i)
        else
          j=slktyp(i-n)
        endif
        if(j.eq.0)then
          if(i.le.n)then
            ddsprn(i)=ddsprn(i)+ddspr(i)
          endif
          goto 45
        endif
        dp=(xs(i)+prs*dxs(i))*(dspr(i)+dus*ddspr(i))
        if (dp.le.tsmall*barpar)then
          s=barpar-dp
        else if(dp.ge.tlarge*barpar)then
          s=-center*barpar
        else
          s=0.0d+0
        endif
        ddsprn(i)=(s-dspr(i)*dxsn(i))/xs(i)+ddspr(i)
        if(j.lt.0)then
          dp=(up(i)+prs*(upinf(i)-dxs(i)))*(dsup(i)+dus*ddsup(i))
          if (dp.le.tsmall*barpar)then
            ss=barpar-dp
          else if(dp.ge.tlarge*barpar)then
            ss=-center*barpar
          else
            ss=0.0d+0
          endif
          ddsupn(i)=(ss+dsup(i)*dxsn(i))/up(i)+ddsup(i)
        endif
  45    dxsn(i)=dxsn(i)+dxs(i)
  40  continue
c
c Compute primal and dual steplengths
c
      call cstpln(prstpl,xs,dxsn,up,upinf,
     x dustpl,dspr,ddsprn,dsup,ddsupn,vartyp,slktyp,vcstat)
c
c Check corrections criteria
c
      if(cr.gt.mincc)then
        if(min(prstpl,dustpl).lt.corstp*min(ostp,ostd))then
          if(min(prstpl,dustpl).lt.min(ostp,ostd))then
            prstpl=ostp
            dustpl=ostd
            cr=cr-1
            goto 999
          else
            maxccx=cr
          endif
        endif
      endif
c
c Continue correcting, change the actual search direction
c
      ostp=prstpl
      ostd=dustpl
      do i=1,mn
        dxs(i)=dxsn(i)
        ddspr(i)=ddsprn(i)
        ddsup(i)=ddsupn(i)
      enddo
      do i=1,m
        ddv(i)=ddvn(i)
      enddo     
      if(cr.lt.maxccx)then
        cr=cr+1
        goto 1
      endif
c
c End of the correction loop, save the number of the corrections
c
 999  corr=cr
      return
      end
c
c ============================================================================
c
c  Prelev:  1 :  rowsng
c           2 :  colsng
c           4 :  rowact
c           8 :  chepdu
c          16 :  duchek
c          32 :  bndchk
c          64 :  splchk
c         128 :  freagr
c         256 :  sparse
c         512 :  xduchk
c
c ========================================================================
c
      subroutine presol(colpnt,colidx,colnzs,rowidx,rownzs,
     x collst,rowlst,colmrk,rowmrk,colsta,rowsta,
     x colbeg,colend,rowbeg,rowend,
     x vartyp,pmaxr,pminr,pmbig,ppbig,
     x upperb,lowerb,upslck,loslck,rhs,obj,prehis,prelen,
     x addobj,big,list,mrk,
     x dulo,duup,dmaxc,dminc,dmbig,dpbig,prelev,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 colpnt(n1),colidx(nz),rowidx(nz),
     x collst(n),rowlst(m),pmbig(m),ppbig(m),
     x colbeg(n),colend(n),rowbeg(m),rowend(m),
     x colmrk(n),rowmrk(m),colsta(n),rowsta(m),
     x list(mn),mrk(mn),prehis(mn),prelen,prelev,code,
     x dpbig(n),dmbig(n),vartyp(n)
      real*8    colnzs(nz),rownzs(nz),pmaxr(m),pminr(m),addobj,
     x upperb(n),lowerb(n),upslck(m),loslck(m),rhs(m),obj(n),
     x dulo(m),duup(m),dmaxc(n),dminc(n),big
c
      integer*4 i,j,k,p,o,pnt1,pnt2,pass,cnum,procn,rnum,coln,rown
      real*8    sol,up,lo,tfeas,zero,lbig,bigbou,dbigbo
      integer*4 dusrch,bndsrc,bndchg
      character*99 buff
C CMSSW: Explicit initialization needed
      pnt1=0
      pnt2=0
c
c initialize : clean up the matrix and set-up row-wise structure
c
      tfeas  = 1.0d-08
      zero   = 1.0d-15
      dusrch =  10
      bndsrc =   5
      bndchg =   6
      bigbou = 1.0d+5
      dbigbo = 1.0d+5
c
      lbig = big*0.9d+0
      pass=0
      rown=0
      coln=0
      cnum=0
      rnum=0
      do i=1,mn
        mrk(i)=-1
      enddo
      do i=1,m
        pmaxr(i)=0.0d+0
        pminr(i)=0.0d+0
        pmbig(i)=0
        ppbig(i)=0
        rowend(i)=0
        if(rowsta(i).gt.-2)then
          rown=rown+1
          rowlst(rown)=i
          rowmrk(i)=0
        endif
      enddo
      do i=1,n
        dmaxc(i)=0.0d+0
        dminc(i)=0.0d+0
        dmbig(i)=0
        dpbig(i)=0
        if(colsta(i).gt.-2)then
          coln=coln+1
          collst(coln)=i
          colmrk(i)=0
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          p=pnt2
          do j=pnt2,pnt1,-1
           if((rowsta(colidx(j)).le.-2).or.(abs(colnzs(j)).lt.zero))then
             o=colidx(j)
             sol=colnzs(j)
             colidx(j)=colidx(p)
             colnzs(j)=colnzs(p)
             colidx(p)=o
             colnzs(p)=sol
             p=p-1
           else
             rowend(colidx(j))=rowend(colidx(j))+1
           endif
          enddo
          colbeg(i)=pnt1
          colend(i)=p
        endif
      enddo
      pnt1=1
      do j=1,rown
        i=rowlst(j)
        rowbeg(i)=pnt1
        pnt1=pnt1+rowend(i)
        rowend(i)=rowbeg(i)-1
      enddo
      do k=1,coln
        i=collst(k)
        pnt1=colbeg(i)
        pnt2=colend(i)
        do j=pnt1,pnt2
          rowend(colidx(j))=rowend(colidx(j))+1
          rowidx(rowend(colidx(j)))=i
          rownzs(rowend(colidx(j)))=colnzs(j)
        enddo
      enddo
c
c Initialize the minimum and maximum row activity
c
      sol=0.9d+0*big
      o=1
      do j=1,coln
        i=collst(j)
        pnt1=colbeg(i)
        pnt2=colend(i)
        up=upperb(i)
        lo=lowerb(i)
        call chgmxm(pnt1,pnt2,up,lo,colidx,colnzs,
     x  ppbig,pmaxr,pmbig,pminr,sol,o,m)
      enddo
c
c Start Presolve sequence: Step 1 : ROW SINGLETONS
c
  10  procn=1
      call setlst(n,m,nz,rown,rowlst,rowmrk,coln,collst,colmrk,
     x procn,rowsta,colsta,rowbeg,rowend,cnum,list,mrk,pass,
     x colbeg,colend,colidx)
      if(coln+rown.eq.0)goto 50
      if((iand(prelev,1).gt.0).and.(cnum.gt.0))then
        call rowsng(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  coln,collst,colmrk,rown,rowlst,rowmrk,
     x  cnum,list,mrk,procn,
     x  ppbig,pmaxr,pmbig,pminr,
     x  lbig,tfeas,zero,code)
        if(code.gt.0)goto 100
      endif
c
c Step 2 : COLUMN SINGLETONS
c
      procn=2
      call setlst(m,n,nz,coln,collst,colmrk,rown,rowlst,rowmrk,
     x procn,colsta,rowsta,colbeg,colend,cnum,list,mrk,pass,
     x rowbeg,rowend,rowidx)
      if(coln+rown.eq.0)goto 50
      if((iand(prelev,2).gt.0).and.(cnum.gt.0))then
        call colsng(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  coln,collst,colmrk,
     x  cnum,list,mrk,procn,
     x  ppbig,pmaxr,pmbig,pminr,
     x  lbig,tfeas,zero,code)
        if(code.gt.0)goto 100
      endif
c
c Step 3 : ROW ACTIVITY CHECK
c
      procn=3
      call setlst(n,m,nz,rown,rowlst,rowmrk,coln,collst,colmrk,
     x procn,rowsta,colsta,rowbeg,rowend,cnum,list,mrk,pass,
     x colbeg,colend,colidx)
      if(coln+rown.eq.0)goto 50
      if((iand(prelev,4).gt.0).and.(cnum.gt.0))then
        call rowact(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  coln,collst,colmrk,rown,rowlst,rowmrk,
     x  cnum,list,mrk,procn,
     x  ppbig,pmaxr,pmbig,pminr,
     x  lbig,tfeas,code)
        if(code.gt.0)goto 100
      endif
c
c Step 4 : CHEAP DUAL TEST
c
      procn=4
      call setlst(m,n,nz,coln,collst,colmrk,rown,rowlst,rowmrk,
     x procn,colsta,rowsta,colbeg,colend,cnum,list,mrk,pass,
     x rowbeg,rowend,rowidx)
      if(coln+rown.eq.0)goto 50
      if((iand(prelev,8).gt.0).and.(cnum.gt.0))then
        call chepdu(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  coln,collst,colmrk,rown,rowlst,rowmrk,
     x  cnum,list,mrk,procn,
     x  ppbig,pmaxr,pmbig,pminr,
     x  lbig,zero,code)
        if(code.gt.0)goto 100
      endif
c
c Step 5 : USUAL DUAL TEST
c
      procn=5
      call setlst(n,m,nz,rown,rowlst,rowmrk,coln,collst,colmrk,
     x procn,rowsta,colsta,rowbeg,rowend,rnum,list(n+1),mrk(n+1),pass,
     x colbeg,colend,colidx)
c
c Remove zero entries at the first loop from the main list
c
      if (pass.eq.5)then
        k=1
   5    if(k.le.coln)then
          if(colmrk(collst(k)).eq.0)then
            colmrk(collst(k))=-1
            collst(k)=collst(coln)
            coln=coln-1
          else
            k=k+1
          endif
          goto 5
        endif
        k=1
  20    if(k.le.rown)then
          if(rowmrk(rowlst(k)).eq.0)then
            rowmrk(rowlst(k))=-1
            rowlst(k)=rowlst(rown)
            rown=rown-1
          else
            k=k+1
          endif
          goto 20
        endif
      endif
c
      if((iand(prelev,16).gt.0).and.(cnum+rnum.gt.0))then
        call duchek(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  coln,collst,colmrk,rown,rowlst,rowmrk,
     x  cnum,list,mrk,rnum,list(n+1),mrk(n+1),procn,
     x  ppbig,pmaxr,pmbig,pminr,
C CMSSW: Prevent REAL*8 reusage warning (note that this is cured by
C simply using the matching temporary array already available)
C Was:  dulo,duup,dmaxc,dminc,dpbig,dmbig,
     x  dulo,duup,dpbig,dmbig,dmaxc,dminc,
     x  big,lbig,tfeas,zero,dbigbo,dusrch,code,prelev)
        if(code.gt.0)goto 100
      endif
      goto 10
c
c Bound check
c
  50  procn=6
      if(iand(prelev,32).gt.0)then
        call bndchk(n,m,mn,nz,
     x  colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,rownzs,
     x  upperb,lowerb,upslck,loslck,
     x  rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x  cnum,list,mrk,procn,dmaxc,dminc,
     x  ppbig,pmaxr,pmbig,pminr,dpbig,dmbig,
     x  big,lbig,tfeas,bndsrc,bndchg,bigbou,code)
        if(code.gt.0)goto 100
      endif
c
c Finding splitted free variables
c
      procn=7
      if(iand(prelev,64).gt.0)then
       call coldbl(n,m,mn,nz,colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,upperb,lowerb,obj,colsta,
     x prelen,prehis,procn,list,dmaxc,vartyp,big,lbig,tfeas,zero)
        if(code.gt.0)goto 100
      endif
      goto 999
c
c Infeasibility detected
c
 100  if(code.eq.3)then
        write(buff,'(1x,a)')'Dual infeasibility detected in presolve'
      else
        write(buff,'(1x,a)')'Primal infeasibility detected in presolve'
      endif
      call mprnt(buff)
      if (procn.eq.1)then
        write(buff,'(1x,a)')'Presolve process: Row singleton check'
      else if (procn.eq.2)then
        write(buff,'(1x,a)')'Presolve process: Column singleton check'
      else if (procn.eq.3)then
        write(buff,'(1x,a)')'Presolve process: Row activity check'
      else if (procn.eq.4)then
        write(buff,'(1x,a)')'Presolve process: Cheap dual check'
      else if (procn.eq.5)then
        write(buff,'(1x,a)')'Presolve process: Dual check'
      else if (procn.eq.6)then
        write(buff,'(1x,a)')'Presolve process: Bound check'
      else if (procn.eq.7)then
        write(buff,'(1x,a)')'Presolve process: Splitcol check'
      endif
      call mprnt(buff)
c
  999 return
      end
c
c ============================================================================
c
      subroutine chgmxm(pnt1,pnt2,upper,lower,idx,nonzrs,
     x pbig,maxr,mbig,minr,lbig,dir,siz)
C
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
c This subroutine changes/updates the minimum/maximum row activity
c values
c
      integer*4 siz,pnt1,pnt2,idx(nz),pbig(siz),mbig(siz),dir
      real*8    upper,lower,nonzrs(nz),maxr(siz),minr(siz),lbig
c
      integer*4 j,k
      real*8 s
c
      do j=pnt1,pnt2
        k=idx(j)
        s=nonzrs(j)
        if(s.gt.0d+0)then
          if(upper.ge.lbig)then
            pbig(k)=pbig(k)+dir
          else
            maxr(k)=maxr(k)+upper*s*dble(dir)
          endif
          if(lower.le.-lbig)then
            mbig(k)=mbig(k)+dir
          else
            minr(k)=minr(k)+lower*s*dble(dir)
          endif
        else
          if(upper.ge.lbig)then
            mbig(k)=mbig(k)+dir
          else
            minr(k)=minr(k)+upper*s*dble(dir)
          endif
          if(lower.le.-lbig)then
            pbig(k)=pbig(k)+dir
          else
            maxr(k)=maxr(k)+lower*s*dble(dir)
          endif
        endif
      enddo
      return
      end
c
c ============================================================================
c
      subroutine modmxm(nz,pnt1,pnt2,oldb,newb,rowidx,nonzeros,
     x pbig,maxr,mbig,minr,lbig,dir,siz)
c
c This subroutine modifies the row (column) activity values
c from an old bound (oldb) to a new one (newb)
c dir= 1 update on upper bound
c dir=-1 update on lower bound
c
      integer*4 nz,siz,pnt1,pnt2,rowidx(nz),pbig(siz),mbig(siz),dir
      real*8    oldb,newb,nonzeros(nz),maxr(siz),minr(siz),lbig
c
      integer*4 f,j,k
      real*8 s,diff
c
      f=0
      diff=newb-oldb
      if(abs(oldb).gt.lbig)then
        diff=newb
        f=1
        if(oldb.gt.0.0d+0)then
          dir=1
        else
          dir=-1
        endif
        do j=pnt1,pnt2
          k=rowidx(j)
          if((nonzeros(j)*dble(dir)).gt.0.0d+0)then
            pbig(k)=pbig(k)-1
          else
            mbig(k)=mbig(k)-1
          endif
        enddo
      endif
      if(abs(newb).gt.lbig)then
        diff=-oldb
        f=f+2
        if(newb.gt.0)then
          dir=1
        else
          dir=-1
        endif
        do j=pnt1,pnt2
          k=rowidx(j)
          if((nonzeros(j)*dble(dir)).gt.0.0)then
            pbig(k)=pbig(k)+1
          else
            mbig(k)=mbig(k)+1
          endif
        enddo
      endif
      if(f.lt.3)then
        do j=pnt1,pnt2
          k=rowidx(j)
          s=nonzeros(j)
          if(s.gt.0.0d+0)then
            if(dir.eq.1)then
              maxr(k)=maxr(k)+diff*s
            else
              minr(k)=minr(k)+diff*s
            endif
          else
            if(dir.eq.1)then
              minr(k)=minr(k)+diff*s
            else
              maxr(k)=maxr(k)+diff*s
            endif
          endif
        enddo
      endif
      return
      end
c
c ============================================================================
c
      subroutine remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x colbeg,colend,rowbeg,rowend,rhs,pivot,traf)
c
c This subroutine removes a column from the row-wise representation
c and updates the right-hand side, if parameter traf is set
c
      integer*4 m,n,nz,col,colidx(nz),rowidx(nz),
     x colbeg(n),colend(n),rowbeg(m),rowend(m)
      real*8    rhs(m),pivot,colnzs(nz),rownzs(nz)
      logical   traf
c
      integer*4 i,j,k,pnt1,pnt2
      real*8    sol
c
      do i=colbeg(col),colend(col)
        j=colidx(i)
        pnt1=rowbeg(j)
        pnt2=rowend(j)-1
        do k=pnt1,pnt2
          if(rowidx(k).eq.col)then
            sol=rownzs(k)
            rowidx(k)=rowidx(pnt2+1)
            rownzs(k)=rownzs(pnt2+1)
            rowidx(pnt2+1)=col
            rownzs(pnt2+1)=sol
            goto 10
          endif
        enddo
  10    rowend(j)=pnt2
      enddo
      if(traf)then
        do i=colbeg(col),colend(col)
          rhs(colidx(i))=rhs(colidx(i))-pivot*colnzs(i)
        enddo
      endif      
      return
      end
c
c =============================================================================
c
      subroutine setlst(m,n,nz,coln,collst,colmrk,rown,rowlst,rowmrk,
     x procn,colsta,rowsta,colbeg,colend,cnum,list,mrk,pass,
     x rowbeg,rowend,rowidx)
c
c This subroutine deletes entries from the main search list 
c and set-up the local search list for the presolv subprocesses.
c
      integer*4 m,n,nz,coln,collst(n),colmrk(n),procn,colsta(n),
     x cnum,list(n),mrk(n),pass,colbeg(n),colend(n),
     x rown,rowlst(m),rowmrk(m),rowsta(m),rowbeg(m),rowend(m),
     x rowidx(nz)
c
      integer*4 i,j,k,p1,p2
c
      pass=pass+1
      k=1
      cnum=0     
  10  if(k.le.coln)then
        i=collst(k)
        if((colsta(i).le.-2).or.(colmrk(i).eq.procn))then
          collst(k)=collst(coln)
          colmrk(i)=-procn
          coln=coln-1
        else
          k=k+1
          if((procn.le.2).and.(colbeg(i).ne.colend(i)))goto 10
          cnum=cnum+1
          list(cnum)=i
          mrk(i)=pass
        endif
        goto 10
      endif
c
      k=1
  20  if(k.le.rown)then
        i=rowlst(k)
        if((rowsta(i).le.-2).or.(rowmrk(i).eq.procn))then
          rowlst(k)=rowlst(rown)
          rowmrk(i)=-procn
          rown=rown-1
        else
          k=k+1
        endif
        goto 20
      endif
c
c Extend lists
c
      k=1
      do while (k.le.rown)
        p1=rowbeg(rowlst(k))
        p2=rowend(rowlst(k))
        do i=p1,p2
          j=rowidx(i)
          if((mrk(j).lt.0).and.
     x    ((procn.gt.2).or.(colbeg(j).eq.colend(j))))then
            mrk(j)=procn
            cnum=cnum+1
            list(cnum)=j
          endif
        enddo
        k=k+1
      enddo
c
      return
      end
c
c ==========================================================================
c ===========================================================================
c
      subroutine rowsng(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x coln,collst,colmrk,rown,rowlst,rowmrk,
     x cnum,list,mrk,procn,
     x ppbig,pmaxr,pmbig,pminr,
     x lbig,tfeas,tzer,code)
c
c This subroutine removes singleton rows and may fixes variables
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,list(m),mrk(m),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x coln,rown,collst(n),rowlst(n),colmrk(n),rowmrk(m),
     x ppbig(m),pmbig(m),code
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),addobj,lbig,tfeas,tzer
c
      integer*4 i,l,row,col,dir,crem,rrem
      real*8 ub,lb,upper,lower,sol,pivot
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c 
      rrem=0
      crem=0
  10  if(cnum.ge.1)then
        row=list(1)
        mrk(row)=-1
        list(1)=list(cnum)
        cnum=cnum-1
        if(rowbeg(row).eq.rowend(row))then
c
c Remove singleton row
c
          col=rowidx(rowbeg(row))
          pivot=rownzs(rowbeg(row))
          traf=.false.
          call remove(n,m,nz,row,rowidx,rownzs,colidx,colnzs,
     x    rowbeg,rowend,colbeg,colend,obj,lower,traf)
          rrem=rrem+1         
          prelen=prelen+1
          prehis(prelen)=row+n
          rowsta(row)=-2-procn
c
c Calculate new bounds (ub,lb)
c
          if(ups(row).lt.lbig)then
            ub=rhs(row)+ups(row)
          else
            ub=ups(row)
          endif
          if(los(row).gt.-lbig)then
            lb=rhs(row)+los(row)
          else
            lb=los(row)
          endif
          if(pivot.gt.0)then
            if(ub.lt.lbig)ub=ub/pivot
            if(lb.gt.-lbig)lb=lb/pivot
          else
            if(ub.lt.lbig)then
              sol=ub/pivot
            else
              sol=-ub
            endif
            if(lb.gt.-lbig)then
              ub=lb/pivot
            else
              ub=-lb
            endif
            lb=sol
          endif
c
c update
c
          upper=upb(col)
          lower=lob(col)
          dir=-1
          call chgmxm(colbeg(col),colend(col),upper,lower,colidx,
     x    colnzs,ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
          if(lb.gt.lower)lower=lb
          if(ub.lt.upper)upper=ub
c
c Check primal feasibility
c
          if((lower-upper).gt.((abs(lower)+1.0d+0)*tfeas))then
            cnum=-col
            code=4
            goto 100
          endif
c
c Check for fix variable
c
          if((upper-lower).lt.((abs(lower)+1.0d+0)*tzer))then
            prelen=prelen+1
            prehis(prelen)=col
            colsta(col)=-2-procn
            traf=.true.
            call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x      colbeg,colend,rowbeg,rowend,rhs,lower,traf)
            crem=crem+1
            addobj=addobj+obj(col)*lower
            do i=colbeg(col),colend(col)
              l=colidx(i)
              if((mrk(l).lt.0).and.(rowbeg(l).eq.rowend(l)))then
                mrk(l)=procn
                cnum=cnum+1
                list(cnum)=l
              endif
            enddo
          else
c
c Update bounds
c
            dir=1
            call chgmxm(colbeg(col),colend(col),upper,lower,colidx,
     x      colnzs,ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
          endif
          lob(col)=lower
          upb(col)=upper

c
c Update search lists
c
          do i=colbeg(col),colend(col)
            l=colidx(i)
            if(rowmrk(l).lt.0)then
              rown=rown+1
              rowlst(rown)=l
            endif
            rowmrk(l)=procn
          enddo
          if(colsta(col).gt.-2)then
            if (colmrk(col).lt.0)then
              coln=coln+1
              collst(coln)=col
            endif
            colmrk(col)=procn
          endif
        endif
        goto 10
      endif
c
 100  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'ROWSNG:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine colsng(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x coln,collst,colmrk,
     x cnum,list,mrk,procn,
     x ppbig,pmaxr,pmbig,pminr,
     x lbig,tfeas,tzer,code)
c
c This subroutine cheks singleton columns
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,list(n),mrk(n),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x coln,collst(n),colmrk(n),ppbig(m),pmbig(m),code
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),addobj,lbig,tfeas,tzer
c
      integer*4 i,j,k,l,row,col,crem,rrem
      real*8 ub,lb,upper,lower,sol,pivot
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c
      rrem=0
      crem=0
  10  if(cnum.ge.1)then
        col=list(1)
        mrk(col)=-1
        list(1)=list(cnum)
        cnum=cnum-1
        if(colbeg(col).eq.colend(col))then
          row=colidx(colbeg(col))
          pivot=colnzs(colbeg(col))
          if(pivot.gt.0.0d+0)then
            lb=lob(col)
            ub=upb(col)
            sol=obj(col)
          else
            ub=-lob(col)
            lb=-upb(col)
            pivot=-pivot
            sol=-obj(col)
          endif
          if((lb.gt.-lbig).or.(ub.lt.lbig))then
c
c Compute lower bound of the LP constraint
c
            if(lb.le.-lbig)then
              l=pmbig(row)-1
              lower=pminr(row)
            else
              l=pmbig(row)
              lower=pminr(row)-lb*pivot
            endif
            if(ups(row).gt.lbig)then
              l=l+1
            else
              lower=lower-ups(row)
            endif
            if(l.gt.0)lower=-lbig
c
c Compute upper bound of the LP constraint
c
            if(ub.gt.lbig)then
              l=ppbig(row)-1
              upper=pmaxr(row)
            else
              l=ppbig(row)
              upper=pmaxr(row)-ub*pivot
            endif
            if(los(row).lt.-lbig)then
              l=l+1
            else
              upper=upper-los(row)
            endif
            if(l.gt.0)upper=lbig
c
c Check new upper and lower bound
c
            if(lb.gt.-lbig)then
              upper=(rhs(row)-upper)/pivot
              if((lb-upper).gt.(abs(lb)+1.0d+0)*tfeas)goto 10
            endif
            if(ub.lt.lbig)then
              lower=(rhs(row)-lower)/pivot
              if((lower-ub).gt.(abs(ub)+1.0d+0)*tfeas)goto 10
            endif
          endif
c
c ( Hidden ) free singleton column found, check slacks
c
          pivot=sol/pivot
          if(pivot.gt.tzer)then
            if(los(row).lt.-lbig)then
              cnum=-col
              code=3
              goto 999
            endif
            rhs(row)=rhs(row)+los(row)
          else if(pivot.lt.-tzer)then
            if(ups(row).gt.lbig)then
              cnum=-col
              code=3
              goto 999
            endif
            rhs(row)=rhs(row)+ups(row)
          endif
c
c Column administration
c
          prelen=prelen+1
          prehis(prelen)=col
          colsta(col)=-2-procn
          traf=.false.
          call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x    colbeg,colend,rowbeg,rowend,rhs,pivot,traf)
          crem=crem+1
          addobj=addobj+rhs(row)*pivot
c
c Row administration
c
          prelen=prelen+1
          prehis(prelen)=row+n
          rowsta(row)=-2-procn
          traf=.true.
          call remove(n,m,nz,row,rowidx,rownzs,colidx,colnzs,
     x    rowbeg,rowend,colbeg,colend,obj,pivot,traf)
          rrem=rrem+1
          j=rowbeg(row)
          k=rowend(row)
          do i=j,k
            l=rowidx(i)
            if(colmrk(l).lt.0)then
              coln=coln+1
              collst(coln)=l
            endif
            colmrk(l)=procn
            if((mrk(l).lt.0).and.(colbeg(l).eq.colend(l)))then
              mrk(l)=procn
              cnum=cnum+1
              list(cnum)=l
            endif
          enddo
        endif
        goto 10
      endif
 999  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'COLSNG:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine rowact(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x coln,collst,colmrk,rown,rowlst,rowmrk,
     x cnum,list,mrk,procn,
     x ppbig,pmaxr,pmbig,pminr,
     x lbig,tfeas,code)
c
c This subroutine removes singleton rows and may fixes variables
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,list(m),mrk(m),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x coln,rown,collst(n),rowlst(n),colmrk(n),rowmrk(m),
     x ppbig(m),pmbig(m),code
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),addobj,lbig,tfeas
c
      integer*4 i,j,k,l,row,col,dir,setdir,p,p1,p2,red,crem,rrem
      real*8 upper,lower,pivot,eps
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c
      rrem=0
      crem=0
  10  if(cnum.ge.1)then
        row=list(1)
        mrk(row)=-1
        list(1)=list(cnum)
        cnum=cnum-1
c
        if(ppbig(row).le.0)then
          upper=pmaxr(row)-rhs(row)
        else
          upper=lbig
        endif
        if(pmbig(row).le.0)then
          lower=pminr(row)-rhs(row)
        else
          lower=-lbig
        endif
c
c Check feasibility
c
        eps=abs(rhs(row)+1.0d+0)*tfeas
        if((lower-ups(row).gt.eps) .or.
     x     (los(row)-upper.gt.eps))then
          cnum=-row-n
          code=4
          goto 100
        endif
c
c Check redundancy
c
        setdir=0
        red=0
        if((los(row)-lower.lt.eps) .and.
     x     (upper-ups(row).lt.eps))then
          red=1
        endif
        if(ups(row)-lower.lt.eps)then
          red=1
          setdir=-1
        else if(upper-los(row).lt.eps)then
          red=1
          setdir=1
        endif
c
c 
c
        if(red.gt.0)then
          prelen=prelen+1
          prehis(prelen)=row+n
          rowsta(row)=-2-procn
          traf=.false.
          call remove(n,m,nz,row,rowidx,rownzs,colidx,colnzs,
     x    rowbeg,rowend,colbeg,colend,obj,pivot,traf)
          rrem=rrem+1
          if(setdir.eq.0)then
            j=rowbeg(row)
            k=rowend(row)
            do i=j,k
              l=rowidx(i)
              if(colmrk(l).lt.0)then
                coln=coln+1
                collst(coln)=l
              endif
              colmrk(l)=procn
            enddo
          else
            dir=-1
            traf=.true.
            j=rowbeg(row)
            k=rowend(row)
            do i=j,k
              col=rowidx(i)
              if(rownzs(i)*dble(setdir).gt.0.0d+0)then
                pivot=upb(col)
              else
                pivot=lob(col)
              endif
              p1=colbeg(col)
              p2=colend(col)
              call chgmxm(p1,p2,upb(col),lob(col),colidx,colnzs,
     x        ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
              addobj=addobj+pivot*obj(col)
              lob(col)=pivot
              upb(col)=pivot
              prelen=prelen+1
              prehis(prelen)=col
              colsta(col)=-2-procn
              call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x        colbeg,colend,rowbeg,rowend,rhs,pivot,traf)
              crem=crem+1
              p1=colbeg(col)
              p2=colend(col)
              do p=p1,p2
                l=colidx(p)
                if(rowmrk(l).lt.0)then
                  rown=rown+1
                  rowlst(rown)=l
                endif
                rowmrk(l)=procn
                if(mrk(l).lt.0)then
                  mrk(l)=procn
                  cnum=cnum+1
                  list(cnum)=l
                endif
              enddo
            enddo
          endif
        endif
        goto 10
      endif
c
 100  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'ROWACT:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine chepdu(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x coln,collst,colmrk,rown,rowlst,rowmrk,
     x cnum,list,mrk,procn,
     x ppbig,pmaxr,pmbig,pminr,
     x lbig,tzer,code)
c
c This subroutine performs the "cheap" dual tests
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,list(n),mrk(n),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x coln,rown,collst(n),rowlst(n),colmrk(n),rowmrk(m),
     x ppbig(m),pmbig(m),code
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),addobj,lbig,tzer
c
      integer*4 i,j,k,l,row,col,dir,p,p1,p2,mode,crem,rrem
      real*8  pivot,sol
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c
      crem=0
      rrem=0
  10  if(cnum.ge.1)then
        col=list(1)
        mrk(col)=-1
        list(1)=list(cnum)
        cnum=cnum-1
c
        p1=colbeg(col)
        p2=colend(col)
        mode=0
        do i=p1,p2
          if (abs(colnzs(i)).gt.tzer)then
            row=colidx(i)
            if(ups(row).gt.lbig)then
              k=1
            else if(los(row).lt.-lbig)then
              k=-1
            else
              goto 10
            endif
            if(colnzs(i).gt.0.0d+0)then
              j=1
            else
              j=-1
            endif
            if(mode.eq.0)then
              mode=j*k
              if((obj(col)*dble(mode)).gt.0.0d+0)goto 10
            else
              if(j*k*mode.lt.0)goto 10
            endif
          endif
        enddo
c
c Check the column
c
        if(mode.gt.0)then
          sol=upb(col)
        else if(mode.lt.0)then
          sol=lob(col)
        else
          if(obj(col).lt.0.0d+0)then
            sol=upb(col)
          else if(obj(col).gt.0.0) then
            sol=lob(col)
          else
            sol=lob(col)
            if(upb(col).ge.lbig)sol=upb(col)
          endif
        endif
c
c Adminisztracio
c
        dir=-1
        call chgmxm(p1,p2,upb(col),lob(col),colidx,colnzs,
     x  ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
c
        prelen=prelen+1
        prehis(prelen)=col
        colsta(col)=-2-procn
        traf=.true.
        if(abs(sol).gt.lbig)then
          pivot=0.0d+0
        else
          pivot=sol
        endif
        call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x  colbeg,colend,rowbeg,rowend,rhs,pivot,traf)
        crem=crem+1
c
        if(abs(sol).gt.lbig)then
          if(abs(obj(col)).gt.tzer)then
            cnum=-col
            code=3
            goto 999
          endif
c
c Row redundacncy with the column
c
          do i=p1,p2
            row=colidx(i)
            if(abs(colnzs(i)).gt.tzer)then
              prelen=prelen+1
              prehis(prelen)=row+n
              rowsta(row)=-2-procn
              traf=.false.
              call remove(n,m,nz,row,rowidx,rownzs,colidx,colnzs,
     x        rowbeg,rowend,colbeg,colend,obj,sol,traf)
              rrem=rrem+1
              j=rowbeg(row)
              k=rowend(row)
              do p=j,k
                l=rowidx(p)
                if(colmrk(l).lt.0)then
                  coln=coln+1
                  collst(coln)=l
                endif
                colmrk(l)=procn
                if(mrk(l).lt.0)then
                  mrk(l)=procn
                  cnum=cnum+1
                  list(cnum)=l
                endif
              enddo
            endif
          enddo
        else
c
c Column is fixed to one bound
c
          do i=p1,p2
            row=colidx(i)
            if(rowmrk(row).lt.0)then
              rown=rown+1
              rowlst(rown)=row
            endif
            rowmrk(row)=procn
          enddo
          addobj=addobj+obj(col)*sol
          lob(col)=pivot
          upb(col)=pivot
        endif
c
        goto 10
      endif
 999  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'CHEPDU:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine duchek(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x coln,collst,colmrk,rown,rowlst,rowmrk,
     x cnum,clist,cmrk,rnum,rlist,rmrk,procn,
     x ppbig,pmaxr,pmbig,pminr,
     x p,q,pbig,mbig,maxc,minc,
     x big,lbig,tfeas,tzer,bigbou,search,code,prelev)
c
c This subroutine removes singleton rows and may fixes variables
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,clist(n),cmrk(n),
     x rnum,rlist(m),rmrk(m),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x coln,rown,collst(n),rowlst(n),colmrk(n),rowmrk(m),
     x ppbig(m),pmbig(m),pbig(n),mbig(n),search,code,prelev
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),p(m),q(m),maxc(n),minc(n),
     x addobj,big,lbig,tfeas,tzer,bigbou
c
      integer*4 i,j,up,row,col,dir,pnt1,pnt2,p1,p2,crem,rrem,up3,
     x lstcnt
      real*8  sol,toler,up1,up2
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c     
      crem=0
      rrem=0
      do while (rnum.ge.1)
        row=rlist(1)
        if (ups(row).gt.lbig)then
          p(row)=0.0d+0
        else
          p(row)=-big
        endif
        if(los(row).lt.-lbig)then
          q(row)=0.0d+0
        else
          q(row)=big
        endif
        rmrk(row)=-1
        rlist(1)=rlist(rnum)
        rnum=rnum-1
      enddo

      cnum=0
      do i=1,n
        if(upb(i).lt.lbig)then
          mbig(i)=1
        else
          mbig(i)=0
        endif
        if(lob(i).gt.-lbig)then
          pbig(i)=1
        else
          pbig(i)=0
        endif
        maxc(i)=0.0d+0
        minc(i)=0.0d+0
        if((colsta(i).gt.-2).and.(upb(i)-lob(i).gt.lbig))then
          cnum=cnum+1
          cmrk(i)=1
          clist(cnum)=i
         else
          cmrk(i)=-2
        endif
      enddo
      dir=1
      do i=1,m
        if(rowsta(i).gt.-2)then
          call chgmxm(rowbeg(i),rowend(i),q(i),p(i),rowidx,rownzs,
     x    pbig,maxc,mbig,minc,lbig,dir,n)
        endif
      enddo
c
      lstcnt=0
      do while (cnum.ne.lstcnt)
        lstcnt=lstcnt+1
        if(lstcnt.gt.n)then
          lstcnt=1
          search=search-1
          if(search.eq.0)goto 100
        endif
        col=clist(lstcnt)
        cmrk(col)=-1
        pnt1=colbeg(col)
        pnt2=colend(col)
        do i=pnt1,pnt2
          row=colidx(i)
c
c Compute new upper bound: up1+(obj-up2)/nzs
c
          if(colnzs(i).gt.0.0d+0)then
            up2=minc(col)
            up3=mbig(col)
          else
            up2=maxc(col)
            up3=pbig(col)
          endif
          if(p(row).lt.-lbig)then
            up1=0.0d+0
            up=1
          else
            up1=p(row)
            up=0
          endif
          if(up.eq.up3)then
            sol=up1+(obj(col)-up2)/colnzs(i)
            if(abs(sol).lt.bigbou)then
              if(q(row)-sol.gt.(abs(sol)+1.0d+0)*tfeas)then
                p1=rowbeg(row)
                p2=rowend(row)
                dir=1
                call modmxm(nz,p1,p2,q(row),sol,rowidx,rownzs,
     x          pbig,maxc,mbig,minc,lbig,dir,n)
                q(row)=sol
                do j=p1,p2
                  if(cmrk(rowidx(j)).eq.-1)then
                    if(upb(rowidx(j))-lob(rowidx(j)).gt.lbig)then
                      cnum=cnum+1
                      if(cnum.gt.n)cnum=1
                      clist(cnum)=rowidx(j)
                      cmrk(rowidx(j))=1
                    endif
                  endif
                enddo
              endif
            endif
          endif
c
c Compute new lower bound: up1+(obj-up2)/nzs
c
          if(colnzs(i).gt.0.0d+0)then
            up2=maxc(col)
            up3=pbig(col)
          else
            up2=minc(col)
            up3=mbig(col)
          endif
          if(q(row).gt.lbig)then
            up1=0.0d+0
            up=1
          else
            up1=q(row)
            up=0
          endif
          if(up.eq.up3)then
            sol=up1+(obj(col)-up2)/colnzs(i)
            if(abs(sol).lt.bigbou)then
              if(sol-p(row).gt.(abs(sol)+1.0d+0)*tfeas)then
                p1=rowbeg(row)
                p2=rowend(row)
                dir=-1
                call modmxm(nz,p1,p2,p(row),sol,rowidx,rownzs,
     x          pbig,maxc,mbig,minc,lbig,dir,n)
                p(row)=sol
                do j=p1,p2
                  if(cmrk(rowidx(j)).eq.-1)then
                    if(upb(rowidx(j))-lob(rowidx(j)).gt.lbig)then
                      cnum=cnum+1
                      if(cnum.gt.n)cnum=1
                      clist(cnum)=rowidx(j)
                      cmrk(rowidx(j))=1
                    endif
                  endif
                enddo
              endif
            endif  
          endif
        enddo
      enddo
c
c Dual feasibility check
c
  100 do while (cnum.ne.lstcnt)
        lstcnt=lstcnt+1
        if(lstcnt.gt.n)lstcnt=1        
        cmrk(clist(lstcnt))=-1
      enddo
      cnum=0
      do row=1,m
        if(rowsta(row).gt.-2)then
          if((p(row)-q(row)).gt.(abs(p(row))+1.0d+0)*tfeas)then
            code=3
            cnum=-row-n
            goto 999
          else if (iand(prelev,512).gt.0)then
            if(q(row)-p(row).lt.(abs(p(row))+1.0d+0)*tfeas)then
              sol=(p(row)+q(row))/2.0d+0
              prelen=prelen+1
              prehis(prelen)=row
              rowsta(row)=-2-procn
              traf=.true.
              call remove(n,m,nz,row,rowidx,rownzs,colidx,colnzs,
     x        rowbeg,rowend,colbeg,colend,obj,sol,traf)
              addobj=addobj+rhs(row)*sol
              do i=rowbeg(row),rowend(row)
                col=rowidx(i)
                if(colmrk(col).lt.0)then
                  coln=coln+1
                  collst(coln)=col
                endif
                colmrk(col)=procn
              enddo
              rrem=rrem+1
            endif
          endif
        endif
      enddo
c
c Checking variables
c
      do 10 col=1,n
        if(colsta(col).le.-2)goto 10
        toler=(abs(obj(col))+1.0d+0)*tfeas
        if(upb(col).lt.lbig)then
          i=1
        else
          i=0
        endif
        if(lob(col).gt.-lbig)then
          j=1
        else
          j=0
        endif
        if((mbig(col).eq.i).and.(obj(col)-minc(col).lt.-toler))then
          sol=upb(col)
        else if((pbig(col).eq.j).and.(obj(col)-maxc(col).ge.toler))then
          sol=lob(col)
        else
          goto 10
        endif
c
c Variable is set to a bound
c
        if(abs(sol).gt.lbig)then
          if(abs(obj(col)).gt.tzer)then
            cnum=-col
            code=3
            goto 999
          endif
        endif
        prelen=prelen+1
        prehis(prelen)=col
        colsta(col)=-2-procn
        traf=.true.
        call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x  colbeg,colend,rowbeg,rowend,rhs,sol,traf)
        crem=crem+1
        addobj=addobj+obj(col)*sol
        do i=colbeg(col),colend(col)
          j=colidx(i)
          if(rowmrk(j).lt.0)then
            rown=rown+1
            rowlst(rown)=j
          endif
          rowmrk(j)=procn
        enddo
        dir=-1
        call chgmxm(colbeg(col),colend(col),upb(col),lob(col),colidx,
     x  colnzs,ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
        upb(col)=sol
        lob(col)=sol
  10  continue
c
 999  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'DUCHEK:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine bndchk(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,ups,los,
     x rhs,obj,addobj,colsta,rowsta,prelen,prehis,
     x cnum,list,mrk,procn,oldlob,oldupb,
     x ppbig,pmaxr,pmbig,pminr,chglob,chgupb,
     x big,lbig,tfeas,search,chgmax,bigbou,code)
c
c This subroutine checks bounds on variables
c NOTE : this subroutine destroys min and max row activity counters !
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),cnum,list(n),mrk(n),
     x colsta(n),rowsta(m),prehis(mn),procn,prelen,
     x ppbig(m),pmbig(m),chglob(n),chgupb(n),search,chgmax,code
c
      real*8 colnzs(nz),rownzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),pmaxr(m),pminr(m),addobj,big,lbig,bigbou,
     x tfeas,oldlob(n),oldupb(n)
c
      integer*4 i,j,up,row,col,dir,pnt1,pnt2,p1,p2,crem,rrem,up3,lstcnt
      real*8  sol,toler,up1,up2
      logical traf
      character*99 buff
c
c ---------------------------------------------------------------------------
c
      crem=0
      rrem=0
      cnum=0
      do i=1,n
        chglob(i)=0
        chgupb(i)=0
        oldupb(i)=upb(i)
        oldlob(i)=lob(i)
      enddo
      do i=1,m
        if(ups(i).gt.lbig)then
          pmbig(i)=pmbig(i)+1
        else
          pminr(i)=pminr(i)-ups(i)
        endif
        if(los(i).lt.-lbig)then
          ppbig(i)=ppbig(i)+1
        else
          pmaxr(i)=pmaxr(i)-los(i)
        endif
        if(rowsta(i).gt.-2)then
          cnum=cnum+1
          mrk(i)=1
          list(cnum)=i
         else
          mrk(i)=-2
        endif
      enddo
c
      lstcnt=0
      do while (cnum.ne.lstcnt)
        lstcnt=lstcnt+1
        if(lstcnt.gt.m)then
          lstcnt=1
          search=search-1
          if(search.eq.0)goto 100
        endif
        row=list(lstcnt)        
        mrk(row)=-1
        pnt1=rowbeg(row)
        pnt2=rowend(row)
c
        do i=pnt1,pnt2
          col=rowidx(i)
c
c Compute new upper bound: lo1+(rhs-up2)/nzs
c
          if(rownzs(i).gt.0.0d+0)then
            up2=pminr(row)
            up3=pmbig(row)
          else
            up2=pmaxr(row)
            up3=ppbig(row)
          endif
          if(lob(col).lt.-lbig)then
            up1=0.0d+0
            up=1
          else
            up1=lob(col)
            up=0
          endif
          if(up.eq.up3)then
            sol=up1+(rhs(row)-up2)/rownzs(i)
            toler=(abs(sol)+1.0d+0)*tfeas
             if(abs(sol).lt.bigbou)then  
              if(upb(col)-sol.gt.toler)then
                chgupb(col)=chgupb(col)+1
                p1=colbeg(col)
                p2=colend(col)
                dir=1
                if(lob(col)-sol.gt.toler)then
                  cnum=-col
                  code=4
                  goto 999
                endif
                if(sol-lob(col).lt.toler)then
                  sol=lob(col)
                endif
                call modmxm(nz,p1,p2,upb(col),sol,colidx,colnzs,
     x          ppbig,pmaxr,pmbig,pminr,lbig,dir,m)
                upb(col)=sol
                if(chgupb(col).lt.chgmax)then
                  do j=p1,p2
                    if(mrk(colidx(j)).eq.-1)then
                      cnum=cnum+1
                      if(cnum.gt.m)cnum=1
                      list(cnum)=colidx(j)
                      mrk(colidx(j))=1
                    endif
                  enddo
                endif
              endif
            endif
          endif
c
c Compute new lower bound: up1+(rhs-up2)/nzs
c
          if(rownzs(i).gt.0.0d+0)then
            up2=pmaxr(row)
            up3=ppbig(row)
          else
            up2=pminr(row)
            up3=pmbig(row)
          endif
          if(upb(col).gt.lbig)then
            up1=0.0d+0
            up=1
          else
            up1=upb(col)
            up=0
          endif
          if(up.eq.up3)then
            sol=up1+(rhs(row)-up2)/rownzs(i)
            toler=(abs(sol)+1.0d+0)*tfeas
             if(abs(sol).lt.bigbou)then  
              if(sol-lob(col).gt.(abs(sol)+1.0d+0)*tfeas)then
                chglob(col)=chglob(col)+1
                p1=colbeg(col)
                p2=colend(col)
                dir=-1
                if((sol-upb(col)).gt.toler)then
                  cnum=-col
                  code=4
                  goto 999
                endif
                if(upb(col)-sol.lt.toler)then
                  sol=upb(col)
                endif
                call modmxm(nz,p1,p2,lob(col),sol,colidx,colnzs,
     x          ppbig,pmaxr,pmbig,pminr,lbig,dir,m)                
                lob(col)=sol
                if(chglob(col).lt.chgmax)then
                  do j=p1,p2
                    if(mrk(colidx(j)).eq.-1)then
                      cnum=cnum+1
                      if(cnum.gt.m)cnum=1
                      list(cnum)=colidx(j)
                      mrk(colidx(j))=1
                    endif
                  enddo
                endif
              endif
            endif
          endif
        enddo
      enddo
c
c Checking row feasibility
c
  100 cnum=0
      do row=1,m
        if(rowsta(row).gt.-2)then
          sol=(abs(rhs(row))+1.0d+0)*tfeas
          if((ppbig(row).eq.0).and.(pmaxr(row)-rhs(row).lt.-sol))then
            code=3
            cnum=-row-n
            goto 999
          endif
          if((pmbig(row).eq.0).and.(rhs(row)-pminr(row).lt.-sol))then
            code=3
            cnum=-row-n
            goto 999
          endif
        endif
      enddo
c
c Bound check and modification
c
      do col=1,n
        if(colsta(col).gt.-2)then
          if((lob(col)-upb(col)).gt.(abs(lob(col))+1.0d+0)*tfeas)then
            code=4
            cnum=-col
            goto 999
          else if(upb(col)-lob(col).lt.abs(lob(col)+1.0d+0)*tfeas)then
            sol=(upb(col)+lob(col))/2.0d+0
            prelen=prelen+1
            prehis(prelen)=col
            colsta(col)=-2-procn
            traf=.true.
            call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x      colbeg,colend,rowbeg,rowend,rhs,sol,traf)
            crem=crem+1
            addobj=addobj+obj(col)*sol
            upb(col)=sol
            lob(col)=sol
          else
            if(chglob(col).gt.0)then
              if(oldlob(col).gt.-lbig)rrem=rrem+1
              lob(col)=-big
            endif
            if(chgupb(col).gt.0)then
              if(oldupb(col).lt.lbig)rrem=rrem+1
              upb(col)=big
            endif
          endif
        endif
      enddo
c
 999  if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'BNDCHK:',crem,' columns,',rrem,' bounds removed'
        call mprnt(buff)
      endif
      return
      end
c
c ===========================================================================
c ============================================================================
c
      subroutine coldbl(n,m,mn,nz,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x upb,lob,obj,colsta,
     x prelen,prehis,procn,mark,valc,vartyp,
     x big,lbig,tfeas,tzer)
c
      integer*4 n,m,mn,nz,colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),colsta(n),
     x prelen,prehis(mn),procn,mark(m),vartyp(n)
      real*8 obj(n),lob(n),upb(n),colnzs(nz),rownzs(nz),valc(m),
     x big,lbig,tfeas,tzer
c
      integer*4 i,j,k,l,col,row,pcol,pnt1,pnt2,ppnt1,ppnt2,pntt1,pntt2,
     x crem,rrem,collen
      real*8 sd,toler,obj1,obj2,lo1,lo2,up1,up2,sol
      logical traf
      character*99 buff
c
c ============================================================================
c
      crem=0
      rrem=0
      do i=1,m
        mark(i)=0
      enddo
c
c Start search
c
      do 25 col=1,n
        if((colsta(col).gt.-2).and.(colend(col).ge.colbeg(col)))then
          pnt1=colbeg(col)
          pnt2=colend(col)
          collen=pnt2-pnt1
          do i=pnt1,pnt2
            mark(colidx(i))=col
            valc(colidx(i))=colnzs(i)
          enddo
c
c Select row
c
          row=0
          l=n+1
          do j=pnt1,pnt2
            k=colidx(j)
            if(rowend(k)-rowbeg(k).lt.l)then
              l=rowend(k)-rowbeg(k)
              row=k
            endif
          enddo
c
c Start search in the row
c
          if(row.ne.0)then
            pntt1=rowbeg(row)
            pntt2=rowend(row)
            do 15 l=pntt1,pntt2
              pcol=rowidx(l)
              ppnt1=colbeg(pcol)
              ppnt2=colend(pcol)
              if((pcol.le.col).or.(collen.ne.ppnt2-ppnt1))goto 15
              do i=ppnt1,ppnt2
                if(mark(colidx(i)).ne.col)goto 15
              enddo
c
c Nonzero structure is O.K.
c
              sd=valc(colidx(ppnt1))/colnzs(ppnt1)
              toler=(abs(sd)+1.0d+0)*tzer
              do i=ppnt1,ppnt2
                if(abs(valc(colidx(i))/colnzs(i)-sd).gt.toler)goto 15
              enddo
c
c Nonzeros are dependent, factor : sd, columns: col,pcol
c
              obj1=obj(col)
              obj2=obj(pcol)*sd
c
c Identical columns found
c
              if(abs(obj1-obj2).le.(abs(obj1)+1.0d+0)*tfeas)then
                lo1=lob(pcol)
                up1=upb(pcol)
                if(lob(col).lt.-lbig)then
                  if(sd.gt.0.0d+0)then
                    lo2=lob(col)
                  else
                    lo2=-lob(col)
                  endif
                else
                  lo2=lob(col)/sd
                endif
                if(upb(col).gt.lbig)then
                  if(sd.gt.0.0d+0)then
                    up2=upb(col)
                  else
                    up2=-upb(col)
                  endif
                else
                  up2=upb(col)/sd
                endif
                if(sd.lt.0.0d+0)then
                  sol=up2
                  up2=lo2
                  lo2=sol
                endif
c
c Store factor and old bound info
c
                obj(col)=sd
                vartyp(col)=0
                if(lo2.lt.-lbig)then
                  vartyp(col)=4
                  lob(col)=lo1
                else
                  lob(col)=lo2
                endif
                if(up2.gt.lbig)then
                  vartyp(col)=vartyp(col)+8
                  upb(col)=up1
                else
                  upb(col)=up2
                endif
                if((lo1.gt.-lbig).and.(lo2.gt.-lbig))then
                  lob(pcol)=lo1+lo2
                else
                  lob(pcol)=-big
                endif
                if((up1.lt.lbig).and.(up2.lt.lbig))then
                  upb(pcol)=up1+up2
                else
                  upb(pcol)=big
                endif
                prelen=prelen+1
                prehis(prelen)=col
                colsta(col)=-2-procn-pcol-10
                traf=.false.
                call remove(m,n,nz,col,colidx,colnzs,rowidx,rownzs,
     x          colbeg,colend,rowbeg,rowend,obj,sol,traf)
                crem=crem+1
                goto 25
              endif
  15        continue
          endif
        endif
  25  continue
      if(rrem+crem.gt.0)then
        write(buff,'(1x,a,i5,a,i5,a)')
     x  'COLDBL:',crem,' columns,',rrem,' rows removed'
        call mprnt(buff)
      endif
      return
      end
c ============================================================================
c ========================================================================
c
      subroutine aggreg(colpnt,colidx,colnzs,rowidx,
     x colsta,rowsta,colbeg,colend,rowbeg,rowend,
     x rhs,obj,prehis,prelen,mrk,vartyp,slktyp,iwrk1,iwrk2,
     x iwrk3,pivcol,pivrow,rwork,addobj,prelev,code)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      common/numer/ tplus,tzer
      real*8        tplus,tzer
c
      integer*4 colpnt(n1),colidx(cfree),rowidx(rfree),
     x colbeg(n),colend(n),rowbeg(m),rowend(m),slktyp(m),
     x colsta(n),rowsta(m),prehis(mn),prelen,prelev,code,mrk(mn),
     x iwrk1(mn),iwrk2(mn),iwrk3(mn),pivcol(n),pivrow(m),vartyp(n)
      real*8  colnzs(cfree),addobj,rhs(m),obj(n),
     x rwork(m)
c
      real*8 reltol,abstol,redtol,filtol
      integer*4 i,j,k,pnt1,pnt2,nfill,pnt,pnto,procn,fpnt
      character*99 buff
c
      reltol=1.0d-3
      abstol=1.0d-5
      redtol=1.0d-4
      filtol=4.0d+0
c
      if(iand(prelev,128).gt.0)then
        procn=8
        pnto=colpnt(1)
        pnt=nz+1
        do i=1,n
          if(colsta(i).gt.-2)then
            pnt1=colbeg(i)
            pnt2=colend(i)
            colbeg(i)=pnt
            do j=pnt1,pnt2
              colidx(pnt)=colidx(j)
              colnzs(pnt)=colnzs(j)
              pnt=pnt+1
            enddo
            colend(i)=pnt-1
            pnt1=pnt2+1
          else
            pnt1=colpnt(i)
          endif
          pnt2=colpnt(i+1)-1
          colpnt(i)=pnto
          do j=pnt1,pnt2
            colidx(pnto)=colidx(j)
            colnzs(pnto)=colnzs(j)
            pnto=pnto+1
          enddo
        enddo
        colpnt(n+1)=pnto
        call elimin(m,n,nz,cfree,rfree,
     x  colbeg,colend,rowbeg,rowend,colidx,rowidx,colnzs,colsta,rowsta,
     x  obj,rhs,vartyp,slktyp,
     x  iwrk1,iwrk2,iwrk1(n+1),iwrk2(n+1),mrk,mrk(n+1),
     x  iwrk3,iwrk3(n+1),rwork,pivcol,pivrow,abstol,reltol,filtol,
     x  pivotn,nfill,addobj,fpnt,code)
        if(code.ne.0)goto 999
c
c Compute new column lengths
c
        do i=1,n
          iwrk1(i)=colpnt(i+1)-colpnt(i)
          if(colsta(i).gt.-2)iwrk1(i)=iwrk1(i)+colend(i)-colbeg(i)+1
        enddo
        do j=1,pivotn
          i=pivrow(j)
          pnt1=rowbeg(i)
          pnt2=rowend(i)
          do k=pnt1,pnt2
            iwrk1(colidx(k))=iwrk1(colidx(k))+1
          enddo
        enddo
c
c Generate new pointers for columns
c
        pnt=1
        do i=1,n
          iwrk2(i)=pnt
          pnt=pnt+iwrk1(i)
        enddo
        if(pnt.gt.fpnt)then
          code=-2
          write(buff,'(1x,a)')'Ran out of RERAL memory'
          call mprnt(buff)
          goto 999
        endif
c
c Assemble the transformed matrix
c
        do i=n,1,-1
          pnt1=colpnt(i)
          pnt2=colpnt(i+1)-1
          pnt=iwrk2(i)+iwrk1(i)-1
          do j=pnt2,pnt1,-1
            colidx(pnt)=colidx(j)
            colnzs(pnt)=colnzs(j)
            pnt=pnt-1
          enddo
          if(colsta(i).gt.-2)then
            fpnt=iwrk2(i)
            pnt1=colbeg(i)
            pnt2=colend(i)
            do j=pnt1,pnt2
              colidx(fpnt)=colidx(j)
              colnzs(fpnt)=colnzs(j)
              fpnt=fpnt+1
            enddo
            iwrk3(i)=fpnt
          endif
        enddo
        colpnt(1)=1
        do i=1,n
          colbeg(i)=iwrk2(i)
          colend(i)=iwrk3(i)-1
          colpnt(i+1)=colpnt(i)+iwrk1(i)
        enddo
        do j=1,pivotn
          i=pivrow(j)
          pnt1=rowbeg(i)
          pnt2=rowend(i)
          do k=pnt1,pnt2
            pnt=iwrk3(colidx(k))
            colidx(pnt)=i
            colnzs(pnt)=colnzs(k)
            iwrk3(colidx(k))=pnt+1
          enddo
        enddo
c
        do i=1,pivotn
          prelen=prelen+1
          prehis(prelen)=pivcol(i)
          colsta(pivcol(i))=-2-procn
          prelen=prelen+1
          prehis(prelen)=pivrow(i)+n
          rowsta(pivrow(i))=-2-procn
        enddo
        write(buff,'(1x,i5,a,i5,a)')pivotn,' row/cols eliminated, ',
     x  nfill,' fill-in created.'
        call mprnt(buff)
        nz=colpnt(n+1)-1
        if(cfree.lt.nz*2)then
          code=-2
          write(buff,'(1x,a)')'Ran out of RERAL memory'
          call mprnt(buff)
        endif
      endif
c
      if(iand(prelev,256).gt.0)then
        do i=1,m
          rowend(i)=rowbeg(i)-1
        enddo
        do i=1,n
          if(colsta(i).gt.-2)then
            pnt1=colbeg(i)
            pnt2=colend(i)
            do j=pnt1,pnt2
              rowend(colidx(j))=rowend(colidx(j))+1
              rowidx(rowend(colidx(j)))=i
              colnzs(nz+rowend(colidx(j)))=colnzs(j)
            enddo
          endif
        enddo
        call sparser(n,n1,m,nz,colpnt,colbeg,colend,colidx,colnzs,
     x  rowbeg,rowend,rowidx,colnzs(nz+1),colsta,rowsta,rhs,slktyp,
     x  mrk,mrk(n+1),tplus,tzer,redtol,reltol,abstol)
      endif
c
 999  return
      end
c
c ============================================================================
c
c Numerically more stable version
c
c ===========================================================================
c
       subroutine  sparser(n,n1,m,nz,colpnt,
     x colbeg,colend,colidx,colnzs,
     x rowbeg,rowend,rowidx,rownzs,
     x colsta,rowsta,rhs,slktyp,
     x mark,rflag,tplus,tzer,redtol,reltol,abstol)
c
      integer*4 n,n1,m,nz,colpnt(n1),colbeg(n),colend(n),colidx(nz),
     x rowbeg(m),rowend(m),rowidx(nz),colsta(n),rowsta(m),
     x rflag(m),mark(n),slktyp(m)
      real*8 colnzs(nz),rownzs(nz),rhs(m),
     x tplus,tzer,redtol,reltol,abstol
c
      integer*4 i,j,k,pnt1,pnt2,rpnt1,rpnt2,row,col,prow,pcol,
     x pnt,ppnt1,ppnt2,elim,total,totaln,iw,rowlen
      real*8 pivot,nval,tol
      character*99 buff
c
c ---------------------------------------------------------------------------
c
      total=0
      totaln=0
      tol=1.0d+0/reltol
      do i=1,m
        if(rowsta(i).gt.-2)then         
          rflag(i)=0
          totaln=totaln+rowend(i)-rowbeg(i)+1
        else
          rflag(i)=2
        endif
      enddo
      do i=1,n
        mark(i)=0
      enddo
c
 100  elim=0
      do 20 row=1,m
        if((rflag(row).lt.2).and.(slktyp(row).eq.0))then
          iw=rflag(row)
          pnt1=rowbeg(row)
          pnt2=rowend(row)
          rowlen=pnt2-pnt1
c
c Select the shortest column
c
          col=0
          k=m+1
          do j=pnt1,pnt2
            i=rowidx(j)
            mark(i)=j
            if(colend(i)-colbeg(i).lt.k)then
              col=i
              k=colend(i)-colbeg(i)
            endif
          enddo
          if(col.eq.0)then
            rflag(row)=1
            goto 20
          endif
c
c Scan the selected column
c
          ppnt1=colbeg(col)
          ppnt2=colend(col)
          do 30 i=ppnt1,ppnt2
            prow=colidx(i)
            rpnt1=rowbeg(prow)
            rpnt2=rowend(prow)
            if((rowlen.gt.rpnt2-rpnt1).or.(iw+rflag(prow).ge.2).or.
     x      (row.eq.prow))goto 30
            k=-1
            do pnt=rpnt1,rpnt2
              if(mark(rowidx(pnt)).gt.0)k=k+1
            enddo
            if(k.ne.rowlen)goto 30
c
c Select pivot
c
            pcol=0
            pivot=tol
            do pnt=rpnt1,rpnt2
              if(mark(rowidx(pnt)).gt.0)then
                if(abs(rownzs(mark(rowidx(pnt)))).gt.abstol)then
                  nval=-rownzs(pnt)/rownzs(mark(rowidx(pnt)))
                  if(abs(nval).lt.abs(pivot))then
                    pivot=nval
                    pcol=rowidx(pnt)
                  endif
                endif
              endif
            enddo
            if(pcol.eq.0)goto 20
c
c Transformation
c
            rflag(prow)=0
            rhs(prow)=rhs(prow)+pivot*rhs(row)
            do pnt=rpnt1,rpnt2
              if(mark(rowidx(pnt)).gt.0)then
                nval=rownzs(pnt)+pivot*rownzs(mark(rowidx(pnt)))
                if(abs(nval).lt.tplus*(abs(rownzs(pnt))))nval=0.0d+0
                rownzs(pnt)=nval
              endif
            enddo
            do while (rpnt1.le.rpnt2)
              if(abs(rownzs(rpnt1)).lt.tzer)then
                k=rowidx(rpnt1)
                rownzs(rpnt1)=rownzs(rpnt2)
                rowidx(rpnt1)=rowidx(rpnt2)
                rownzs(rpnt2)=0.0d+0
                rowidx(rpnt2)=k
                rpnt2=rpnt2-1
                elim=elim+1
              else
                rpnt1=rpnt1+1
              endif
            enddo
            rowend(prow)=rpnt2
  30      continue
          do j=pnt1,pnt2
            mark(rowidx(j))=0
          enddo
          rflag(row)=1
        endif
  20  continue
      total=total+elim
      totaln=totaln-elim
      if(dble(elim)/(dble(totaln)+1.0d+0).gt.redtol)goto 100
c
c making modification in the column file
c
      if(total.gt.0)then
        do i=1,n         
          mark(i)=colbeg(i)-1
        enddo
        do i=1,m
          if(rowsta(i).gt.-2)then
            pnt1=rowbeg(i)
            pnt2=rowend(i)
            do j=pnt1,pnt2
              col=rowidx(j)
              mark(col)=mark(col)+1
              colidx(mark(col))=i
              colnzs(mark(col))=rownzs(j)
            enddo
          endif
        enddo
        pnt=colpnt(1)
        do i=1,n
          iw=pnt
          if(colsta(i).gt.-2)then
            pnt1=colbeg(i)
            pnt2=mark(i)
            colbeg(i)=pnt
            do j=pnt1,pnt2
              colnzs(pnt)=colnzs(j)
              colidx(pnt)=colidx(j)
              pnt=pnt+1
            enddo
            pnt1=colend(i)+1
            colend(i)=pnt-1
          else
            pnt1=colpnt(i)            
          endif
          pnt2=colpnt(i+1)-1
          do j=pnt1,pnt2
            colnzs(pnt)=colnzs(j)
            colidx(pnt)=colidx(j)
            pnt=pnt+1
          enddo
          colpnt(i)=iw
        enddo
        colpnt(n+1)=pnt
      endif
c
      write(buff,'(1x,i5,a)')total,' nonzeros eliminated'
      call mprnt(buff)
      return
      end
c
c ===========================================================================
c ===========================================================================
c
      subroutine elimin(m,n,nz,cfre,rfre,
     x colbeg,ccol,rowbeg,crow,colidx,rowidx,colnzs,colsta,rowsta,
     x obj,rhs,vartyp,slktyp,cpermf,cpermb,rpermf,rpermb,colcan,
     x mark,cfill,rfill,workr,pivcol,pivrow,abstol,reltol,filtol,
     x pivotn,nfill,addobj,pnt,code)
c
      integer*4 m,n,nz,cfre,rfre,colbeg(n),ccol(n),rowbeg(m),
     x crow(m),colidx(cfre),rowidx(rfre),colsta(n),rowsta(m),
     x cpermf(n),cpermb(n),rpermf(m),rpermb(m),colcan(n),mark(m),
     x cfill(n),rfill(m),pivcol(n),pivrow(m),pivotn,code,vartyp(n),
     x slktyp(m),nfill,pnt
      real*8 workr(m),obj(n),rhs(m),colnzs(cfre),abstol,reltol,
     x filtol,addobj
c
      integer*4 i,j,k,l,p,pnt1,pnt2,ppnt1,ppnt2,pcol,prow,
     x fren,cfirst,rfirst,clast,rlast,endmem,prewcol,mn,fill,
     x ccfre,rcfre,rpnt1,rpnt2,ii
      real*8 pivot,s
c
c ---------------------------------------------------------------------------
c
c     cpermf       oszloplista elore lancolasa, fejmutato cfirst
c     cpermb       oszloplista hatra lancolasa, fejmutato clast
c     rpermf       sorlista    elore lancolase, fejmutato rfirst
c     rpermb       sorlista    hatra lancolasa, fejmutato rlast
c     colcan       lehetseges pivot oszlopok
c     ccol         oszlopszamlalok
c     crow         sorszamlalok (vcstat)
c     colbeg       oszlopmutatok
c     rowbeg       sormutatok
c     mark         eliminacios integer segedtomb
c     workr        eliminacios real    segedtomb
c     cfill        a sorfolytonos tarolas update-elesehez segedtomb
c     rfill        a sorfolytonos tarolas update-elesehez segedtomb
c     pivcol
c     pivrow
c
c --------------------------------------------------------------------------
      pivot=0
      ppnt1=0
c
c initialization
c
      nfill=0
      mn=m+n
      endmem=cfre
      fren =0
      pivotn=0
      cfirst=0
      clast =0
      rfirst=0
      rlast =0
      do i=1,n
        if(colsta(i).gt.-2)then
          if(cfirst.eq.0)then
            cfirst=i
          else
            cpermf(clast)=i
          endif
          cpermb(i)=clast
          clast=i
          ccol(i)=ccol(i)-colbeg(i)+1
          if(vartyp(i).eq.0)then
            fren=fren+1
            colcan(fren)=i
          endif
        endif
      enddo
C CMSSW: Bugfix for an empty matrix, where now clast=0 causes an invalid
C memory access
      if(clast.ne.0)then
      cpermf(clast)=0
      endif
      do i=1,m
        mark(i)=0
        if(rowsta(i).gt.-2)then
          if(rfirst.eq.0)then
            rfirst=i
          else
            rpermf(rlast)=i
          endif
          rpermb(i)=rlast
          rlast=i
          crow(i)=crow(i)-rowbeg(i)+1
        endif
      enddo
      rpermf(rlast)=0
c
c Elimination loop
c
  50  pcol=0
      prow=0
      i=-1
c
c Find pivot
c
      do ii=1,fren
        p=colcan(ii)
        pnt1=colbeg(p)
        pnt2=pnt1+ccol(p)-1
        s=0.0d+0
        do j=pnt1,pnt2
          if(s.lt.abs(colnzs(j)))s=abs(colnzs(j))
        enddo
        s=s*reltol
        do j=pnt1,pnt2
          if(slktyp(colidx(j)).eq.0)then
            if(abs(colnzs(j)).gt.abstol)then
              k=(ccol(p)-1)*(crow(colidx(j))-1)
              if(dble(k).lt.filtol*dble(ccol(p)+crow(colidx(j))-1))then
                if((i.lt.0).or.(k.lt.i))then
                  if(abs(colnzs(j)).gt.s)then
                    i=k
                    pcol=p
                    prow=colidx(j)
                    pivot=colnzs(j)
                    ppnt1=ii
                  endif
                else if((k.eq.i).and.(abs(pivot).lt.abs(colnzs(j))))then
                  pcol=p
                  prow=colidx(j)
                  pivot=colnzs(j)
                  ppnt1=ii
                endif
              endif
            endif
          endif
        enddo
      enddo
      if (pcol.eq.0)goto 900
      colcan(ppnt1)=colcan(fren)
      fren=fren-1
      pivot=1.0d+0/pivot
      rcfre=rfre-rowbeg(rlast)-crow(rlast)
      ccfre=endmem-colbeg(clast)-ccol(clast)
c
c compress column file
c
      if(ccfre.lt.mn)then
        call ccomprs(mn,cfre,ccfre,endmem,nz,
     x  colbeg,ccol,cfirst,cpermf,colidx,colnzs,code)
        if(code.lt.0)goto 999
      endif
c
c remove pcol from the cpermf lists
c
      j=cpermb(pcol)
      i=cpermf(pcol)
      if(j.ne.0)then
        cpermf(j)=i
      else
        cfirst=i
      endif
      if(i.eq.0)then
        clast=j
      else
        cpermb(i)=j
      endif
c
c remove prow from the rpermf lists
c
      j=rpermb(prow)
      i=rpermf(prow)
      if(j.ne.0)then
        rpermf(j)=i
      else
        rfirst=i
      endif
      if(i.eq.0)then
        rlast=j
      else
        rpermb(i)=j
      endif
c
c administration
c
      pivotn=pivotn+1
      pivcol(pivotn)=pcol
      pivrow(pivotn)=prow   
      addobj=addobj+obj(pcol)*rhs(prow)*pivot
c
c Create pivot column
c
      pnt1=colbeg(pcol)
      pnt2=pnt1+ccol(pcol)-1
      ppnt1=endmem-ccol(pcol)
      ppnt2=ppnt1+ccol(pcol)-1
      pnt=ppnt1
      do j=pnt1,pnt2
        k=colidx(j)
        mark(k)=1
        colidx(pnt)=k
        colnzs(pnt)=colnzs(j)
        if(k.eq.prow)then
          p=pnt
          workr(k)=pivot
        else
          workr(k)=-colnzs(j)*pivot
          rhs(k)=rhs(k)+rhs(prow)*workr(k)
        endif
        pnt=pnt+1
c
        i=rowbeg(k)
        do while(rowidx(i).ne.pcol)
          i=i+1
        enddo
        rowidx(i)=rowidx(rowbeg(k)+crow(k)-1)
        rfill(k)=-1
      enddo
c
      colbeg(pcol)=ppnt1
      j=colidx(ppnt1)
      s=colnzs(ppnt1)
      colidx(ppnt1)=colidx(p)
      colnzs(ppnt1)=colnzs(p)
      colidx(p)=j
      colnzs(p)=s
      ppnt1=ppnt1+1
c
c create pivot row
c
      pnt1=rowbeg(prow)
      crow(prow)=crow(prow)-1
      pnt2=pnt1+crow(prow)-1
      rowbeg(prow)=colbeg(pcol)-crow(prow)
      pnt=rowbeg(prow)
      do i=pnt1,pnt2
        k=rowidx(i)
        j=colbeg(k)
        do while(colidx(j).ne.prow)
          j=j+1
        enddo
        colidx(pnt)=k
        colnzs(pnt)=colnzs(j)
        pnt=pnt+1
        colidx(j)=colidx(colbeg(k)+ccol(k)-1)
        colnzs(j)=colnzs(colbeg(k)+ccol(k)-1)
        cfill(k)=ccol(k)-1
      enddo
      endmem=endmem-ccol(pcol)-crow(prow)
      ccfre=ccfre-ccol(pcol)-crow(prow)
c
c elimination loop
c
      rpnt1=rowbeg(prow)
      rpnt2=rpnt1+crow(prow)-1
      do p=rpnt1,rpnt2
        i=colidx(p)
        s=colnzs(p)
        obj(i)=obj(i)-s*obj(pcol)*pivot
        fill=ccol(pcol)-1
        pnt1=colbeg(i)
        pnt2=pnt1+cfill(i)-1
        do j=pnt1,pnt2
          k=colidx(j)
          if(mark(k).ne.0)then
            colnzs(j)=colnzs(j)+s*workr(k)
            fill=fill-1
            mark(k)=0
          endif
        enddo
c
c compute the free space
c
        j=cpermf(i)
        if(j.eq.0)then
          k=endmem-pnt2-1
        else
          k=colbeg(j)-pnt2-1
        endif
c
c move column to the end of the column file
c
        if(fill.gt.k)then
          if (ccfre.lt.m)then
            call ccomprs(mn,cfre,ccfre,endmem,nz,
     x      colbeg,ccol,cfirst,cpermf,colidx,colnzs,code)
            if(code.lt.0)goto 999
            pnt1=colbeg(i)
            pnt2=pnt1+cfill(i)-1
          endif
          if(i.ne.clast)then
            j=colbeg(clast)+ccol(clast)
            colbeg(i)=j
            do k=pnt1,pnt2
              colidx(j)=colidx(k)
              colnzs(j)=colnzs(k)
              j=j+1
            enddo
            pnt1=colbeg(i)
            pnt2=j-1
            k=cpermf(i)
            j=cpermb(i)
            if(j.eq.0)then
              cfirst=k
            else
              cpermf(j)=k
            endif
            cpermb(k)=j
            cpermf(clast)=i
            cpermb(i)=clast
            clast=i
            cpermf(clast)=0
          endif
        endif
c
c create fill-in
c
        do k=ppnt1,ppnt2
          j=colidx(k)
          if(mark(j).eq.0)then
            mark(j)=1
          else
            pnt2=pnt2+1
            colnzs(pnt2)=s*workr(j)
            colidx(pnt2)=j
            rfill(j)=rfill(j)+1
          endif
        enddo
        ccol(i)=pnt2-pnt1+1
        if(i.eq.clast)then
          ccfre=endmem-pnt2
        endif
      enddo
c
c make space for fills in the row file
c
      do j=ppnt1,ppnt2
        i=colidx(j)
        mark(i)=0
c
c compute the free space
c
        pnt2=rowbeg(i)+crow(i)-1
        p=rpermf(i)
        if(p.eq.0)then
          k=rfre-pnt2-1
        else
          k=rowbeg(p)-pnt2-1
        endif
c
c move row to the end of the row file
c
        if(k.lt.rfill(i))then
          if(rcfre.lt.n)then
            call rcomprs(mn,rfre,
     x      rcfre,rowbeg,crow,rfirst,rpermf,rowidx,code)
            if(code.lt.0)goto 999
          endif
          if(p.ne.0)then
            pnt1=rowbeg(i)
            pnt2=pnt1+crow(i)-1
            pnt=rowbeg(rlast)+crow(rlast)
            rowbeg(i)=pnt
            do l=pnt1,pnt2
              rowidx(pnt)=rowidx(l)
              pnt=pnt+1
            enddo
            prewcol=rpermb(i)
            if(prewcol.eq.0)then
              rfirst=p
            else
              rpermf(prewcol)=p
            endif
            rpermb(p)=prewcol
            rpermf(rlast)=i
            rpermb(i)=rlast
            rlast=i
            rpermf(rlast)=0
          endif
        endif
        crow(i)=crow(i)+rfill(i)
        if(i.eq.rlast)rcfre=rfre-crow(i)-rowbeg(i)
        nfill=nfill+rfill(i)+1
      enddo       
c
c make pointers to the end of the filled rows
c
      do j=ppnt1,ppnt2
        rfill(colidx(j))=rowbeg(colidx(j))+crow(colidx(j))-1
      enddo
c
c generate fill-in the row file
c
      do j=rpnt1,rpnt2
        i=colidx(j)
        pnt1=colbeg(i)+cfill(i)
        pnt2=colbeg(i)+ccol(i)-1
        do k=pnt1,pnt2
          rowidx(rfill(colidx(k)))=i
          rfill(colidx(k))=rfill(colidx(k))-1
        enddo
      enddo
      goto 50
c
c End of the elimination, compress arrays
c
 900  call rcomprs(mn,rfre,rcfre,rowbeg,crow,rfirst,rpermf,rowidx,code)
      pnt=endmem
      i=clast
      do while(i.ne.0)
        pnt1=colbeg(i)
        pnt2=pnt1+ccol(i)-1
        do j=pnt2,pnt1,-1
          pnt=pnt-1
          colidx(pnt)=colidx(j)
          colnzs(pnt)=colnzs(j)          
        enddo
        colbeg(i)=pnt
        i=cpermb(i)
      enddo
c
c Make pointers form counters
c
      do i=1,n
        ccol(i)=colbeg(i)+ccol(i)-1
      enddo
      do i=1,m
        crow(i)=rowbeg(i)+crow(i)-1
      enddo
 999  return
      end
c
c ===========================================================================
c This is a POSTSOLV procedure
c
c ========================================================================
c
      subroutine pstsol(colpnt,colidx,colnzs,colsta,rowsta,
     x vartyp,slktyp,upb,lob,ups,los,rhs,obj,xs,
     x status,rowval,prehis,prelen,big)
c
      common/dims/ n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
      integer*4    n,n1,m,mn,nz,cfree,pivotn,denwin,rfree
c
      integer*4 colpnt(n1),colidx(nz),colsta(n),rowsta(m),
     x prehis(mn),prelen,vartyp(n),slktyp(m),status(mn)
      real*8    colnzs(nz),upb(n),lob(n),ups(m),los(m),
     x rhs(m),obj(n),rowval(m),xs(n),big
c
      integer*4 i,j,k,l,p,pnt1,pnt2,row,col
      real*8    sol,lo1,lo2,up1,up2,lbig,sol1,sol2,s
c
C CMSSW: Explicit initialization needed
      sol=0

      lbig=0.9d+0*big
      do i=1,mn
        if(i.le.n)then
          j=colsta(i)
        else
          j=rowsta(i-n)
        endif
        if(j.eq.-3)then
          status(i)=0
        else
          status(i)=prelen+1
        endif
      enddo
c
      do i=1,prelen
        status(prehis(i))=i
      enddo
c
      do i=1,m
        rowval(i)=0.0d+0
        if(abs(slktyp(i)).eq.2)rhs(i)=-rhs(i)
      enddo
c
      do i=1,n
        pnt1=colpnt(i)
        pnt2=colpnt(i+1)-1
        do j=pnt1,pnt2
          if(abs(slktyp(colidx(j))).eq.2)then
            colnzs(j)=-colnzs(j)
          endif
        enddo
        if((status(i).gt.prelen).or.(status(i).eq.0))then
          if(vartyp(i).ne.0)then
            if(upb(i).lt.lbig)upb(i)=upb(i)+lob(i)
            xs(i)=xs(i)+lob(i)
            do j=pnt1,pnt2
              rhs(colidx(j))=rhs(colidx(j))+colnzs(j)*lob(i)
            enddo
          endif
          if(abs(vartyp(i)).eq.2)then
            obj(i)=-obj(i)
            upb(i)=-lob(i)
            lob(i)=-big
            xs(i)=-xs(i)
            do j=pnt1,pnt2
              colnzs(j)=-colnzs(j)
            enddo
          endif
          do j=pnt1,pnt2
            rowval(colidx(j))=rowval(colidx(j))+xs(i)*colnzs(j)
          enddo
        endif
      enddo
c
      i=prelen
      do while(i.ge.1)
        j=prehis(i)
        if(j.le.n)then
           k=-colsta(j)-2
          if((k.eq.1).or.(k.eq.3).or.(k.eq.5).or.(k.eq.6))then
            sol=lob(j)
            xs(j)=sol
          else if((k.eq.2).or.(k.eq.8))then
            row=prehis(i+1)-n         
            l=colpnt(j)
            do while(l.lt.colpnt(j+1))
              if(colidx(l).eq.row)then
                sol=colnzs(l)
                l=colpnt(j+1)
              endif
              l=l+1
            enddo
            sol=(rhs(row)-rowval(row))/sol
            xs(j)=sol
          else if(k.eq.4)then
            k=0
            sol1=lob(j)
            sol2=upb(j)
            p=i+1
            do while ((p.le.prelen).and.(prehis(p).gt.n).and.
     x        (-rowsta(prehis(p)-n)-2.eq.4))
              row=prehis(p)-n
              l=colpnt(j)
              do while(l.lt.colpnt(j+1))
                if(colidx(l).eq.row)then
                  sol=colnzs(l)
                  l=colpnt(j+1)
                endif
                l=l+1
              enddo
              if(los(row).gt.-lbig)then
                s=(rhs(row)-rowval(row)+los(row))/sol
                if((sol.gt.0.0d+0).and.(s.gt.sol1))then
                  k=1
                  sol1=s
                endif
                if((sol.lt.0.0d+0).and.(s.lt.sol2))then
                  k=2
                  sol2=s
                endif
              endif
              if(ups(row).lt.lbig)then
                s=(rhs(row)-rowval(row)+ups(row))/sol
                if((sol.gt.0.0d+0).and.(s.lt.sol2))then
                  k=2
                  sol2=s
                endif
                if((sol.lt.0.0d+0).and.(s.gt.sol1))then
                  k=1
                  sol1=s
                endif
              endif
              p=p+1
            enddo
            if(k.eq.1)sol=sol1
            if(k.eq.2)sol=sol2
            if(k.eq.0)then
              sol=sol1
              if(sol.lt.-lbig)sol=sol2
              if(sol.gt.lbig)sol=0.0d+0
            endif
            xs(j)=sol
          else if(k.gt.17)then
            col=k-17
            if((vartyp(j).eq.4).or.(vartyp(j).eq.12))then
              lo2=-big
              lo1=lob(j)
            else
              lo2=lob(j)
              if((lo2.gt.-lbig).and.(lob(col).gt.-lbig))then
                lo1=lob(col)-lo2
              else
                lo1=-big
              endif
            endif
            if((vartyp(j).eq.8).or.(vartyp(j).eq.12))then
              up2=big
              up1=upb(j)
            else
              up2=upb(j)
              if((up2.lt.lbig).and.(upb(col).lt.lbig))then
                up1=upb(col)-up2
              else
                up1=big
              endif
            endif
            lob(col)=lo1
            upb(col)=up1
            sol=0.0d+0
            if(sol.lt.lo2)sol=lo2
            if(sol.gt.up2)sol=up2 
            if(xs(col)-sol.lt.lo1)sol=xs(col)-lo1
            if(xs(col)-sol.gt.up1)sol=xs(col)-up1
            xs(j)=sol*obj(j)
            xs(col)=xs(col)-sol
            sol=0.0d+0
          endif
          l=colpnt(j)
          do while(l.lt.colpnt(j+1))
            row=colidx(l)
            if(status(row+n).gt.status(j))then
              rhs(row)=rhs(row)+colnzs(l)*sol
            else
              rowval(row)=rowval(row)+colnzs(l)*sol
            endif
            l=l+1
          enddo
        endif
        i=i-1
      enddo
c
      return
      end
c
c ============================================================================
      subroutine mprnt(buff)
      character*99 buff
      common/logprt/ loglog,lfile
      integer*4      loglog,lfile
c
    1 format(a79)
      if((loglog.eq.1).or.(loglog.eq.3))then
        write(*,1)buff
      endif
      if((loglog.eq.2).or.(loglog.eq.3))then
        write(lfile,1)buff
      endif
c      
      return
      end     
c ==========================================================================
c
      subroutine timer(i)
      implicit none
      integer*4 i
      real t
c
c --------------------------------------------------------------------------
c
c Implementation based on the Fortran 95 cpu_time()
      call cpu_time(t)
      i=nint(t*100.0)
      end 
c
c =========================================================================
