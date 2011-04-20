c============================================================================
c                             april 10, 2002, v6.01
c                             february 23, 2003, v6.1
c
c   ref[1]: "new generation of parton distributions with uncertainties from global qcd analysis"
c       by: j. pumplin, d.r. stump, j.huston, h.l. lai, p. nadolsky, w.k. tung
c       jhep 0207:012(2002), hep-ph/0201195
c
c   ref[2]: "inclusive jet production, parton distributions, and the search for new physics"
c       by : d. stump, j. huston, j. pumplin, w.k. tung, h.l. lai, s. kuhlmann, j. owens
c       hep-ph/0303013
c
c   this package contains
c   (1) 4 standard sets of cteq6 pdf's (cteq6m, cteq6d, cteq6l, cteq6l1) ;
c   (2) 40 up/down sets (with respect to cteq6m) for uncertainty studies from ref[1];
c   (3) updated version of the above: cteq6.1m and its 40 up/down eigenvector sets from ref[2].
c
c  the cteq6.1m set provides a global fit that is almost equivalent in every respect
c  to the published cteq6m, ref[1], although some parton distributions (e.g., the gluon)
c  may deviate from cteq6m in some kinematic ranges by amounts that are well within the
c  specified uncertainties.
c  the more significant improvements of the new version are associated with some of the
c  40 eigenvector sets, which are made more symmetrical and reliable in (3), compared to (2).

c  details about calling convention are:
c ---------------------------------------------------------------------------
c  iset   pdf-set     description       alpha_s(mz)**lam4  lam5   table_file
c ===========================================================================
c standard, "best-fit", sets:
c --------------------------
c   1    cteq6m   standard msbar scheme   0.118     326   226    cteq6m.tbl
c   2    cteq6d   standard dis scheme     0.118     326   226    cteq6d.tbl
c   3    cteq6l   leading order           0.118**   326** 226    cteq6l.tbl
c   4    cteq6l1  leading order           0.130**   215** 165    cteq6l1.tbl
c ============================================================================
c for uncertainty calculations using eigenvectors of the hessian:
c ---------------------------------------------------------------
c     central + 40 up/down sets along 20 eigenvector directions
c                             -----------------------------
c                original version, ref[1]:  central fit: cteq6m (=cteq6m.00)
c                             -----------------------
c  1xx  cteq6m.xx  +/- sets               0.118     326   226    cteq6m1xx.tbl
c        where xx = 01-40: 01/02 corresponds to +/- for the 1st eigenvector, ... etc.
c        e.g. 100      is cteq6m.00 (=cteq6m),
c             101/102 are cteq6m.01/02, +/- sets of 1st eigenvector, ... etc.
c                              -----------------------
c                updated version, ref[2]:  central fit: cteq6.1m (=cteq61.00)
c                              -----------------------
c  2xx  cteq61.xx  +/- sets               0.118     326   226    ctq61.xx.tbl
c        where xx = 01-40: 01/02 corresponds to +/- for the 1st eigenvector, ... etc.
c        e.g. 200      is cteq61.00 (=cteq6.1m),
c             201/202 are cteq61.01/02, +/- sets of 1st eigenvector, ... etc.
c ===========================================================================
c   ** all fits are obtained by using the same coupling strength
c   \alpha_s(mz)=0.118 and the nlo running \alpha_s formula, except cteq6l1
c   which uses the lo running \alpha_s and its value determined from the fit.
c   for the lo fits, the evolution of the pdf and the hard cross sections are
c   calculated at lo.  more detailed discussions are given in the references.
c
c   the table grids are generated for 10^-6 < x < 1 and 1.3 < q < 10,000 (gev).
c   pdf values outside of the above range are returned using extrapolation.
c   lam5 (lam4) represents lambda value (in mev) for 5 (4) flavors.
c   the matching alpha_s between 4 and 5 flavors takes place at q=4.5 gev,
c   which is defined as the bottom quark mass, whenever it can be applied.
c
c   the table_files are assumed to be in the working directory.
c
c   before using the pdf, it is necessary to do the initialization by
c       call setctq6(iset)
c   where iset is the desired pdf specified in the above table.
c
c   the function ctq6pdf (iparton, x, q)
c   returns the parton distribution inside the proton for parton [iparton]
c   at [x] bjorken_x and scale [q] (gev) in pdf set [iset].
c   iparton  is the parton label (5, 4, 3, 2, 1, 0, -1, ......, -5)
c                            for (b, c, s, d, u, g, u_bar, ..., b_bar),
c
c   for detailed information on the parameters used, e.q. quark masses,
c   qcd lambda, ... etc.,  see info lines at the beginning of the
c   table_files.
c
c   these programs, as provided, are in double precision.  by removing the
c   "implicit double precision" lines, they can also be run in single
c   precision.
c
c   if you have detailed questions concerning these cteq6 distributions,
c   or if you find problems/bugs using this package, direct inquires to
c   pumplin@pa.msu.edu or tung@pa.msu.edu.
c
c===========================================================================

      double precision function ctq6pdf (iparton, x, q)
      implicit double precision (a-h,o-z)
      logical warn
      common
     > / ctqpar2 / nx, nt, nfmx
     > / qcdtable /  alambda, nfl, iorder

      data warn /.true./
      save warn

      if (x .lt. 0d0 .or. x .gt. 1d0) then
        print *, 'x out of range in ctq6pdf: ', x
        stop
      endif
      if (q .lt. alambda) then
        print *, 'q out of range in ctq6pdf: ', q
        stop
      endif
      if ((iparton .lt. -nfmx .or. iparton .gt. nfmx)) then
         if (warn) then
c        put a warning for calling extra flavor.
             warn = .false.
             print *, 'warning: iparton out of range in ctq6pdf! '
             print *, 'iparton, mxflvn0: ', iparton, nfmx
         endif
         ctq6pdf = 0d0
         return
      endif

      ctq6pdf = partonx6 (iparton, x, q)

      if(ctq6pdf.lt.1.0d-16)  ctq6pdf = 0.0d0

      return

c                             ********************
      end

      subroutine setctq6 (iset)
      implicit double precision (a-h,o-z)
      parameter (isetmax0=5)
      character flnm(isetmax0)*6, nn*3, tablefile*140
      data (flnm(i), i=1,isetmax0)
     > / 'cteq6m', 'cteq6d', 'cteq6l', 'cteq6l','ctq61.'/
      data isetold, isetmin0, isetmin1, isetmax1 /-987,1,100,140/
      data isetmin2,isetmax2 /200,240/
      save
c             if data file not initialized, do so.
      character cmsdir*82
      call getenv('CMSSW_BASE',cmsdir)
      kspace=index(cmsdir,' ')
      if(iset.ne.isetold) then
         iu= nextun()
         if (iset.ge.isetmin0 .and. iset.le.3) then
            tablefile=cmsdir(1:kspace-1)
     $           //'/src/GeneratorInterface/GenExtensions/data/'
     $           //flnm(iset)//'.tbl'
         elseif (iset.eq.4) then
            tablefile=cmsdir(1:kspace-1)
     $           //'/src/GeneratorInterface/GenExtensions/data/'
     $           //flnm(iset)//'1.tbl'
         elseif (iset.ge.isetmin1 .and. iset.le.isetmax1) then
            write(nn,'(i3)') iset
            tablefile=cmsdir(1:kspace-1)
     $           //'/src/GeneratorInterface/GenExtensions/data/'
     $           //flnm(1)//nn//'.tbl'
         elseif (iset.ge.isetmin2 .and. iset.le.isetmax2) then
            write(nn,'(i3)') iset
            tablefile=cmsdir(1:kspace-1)
     $           //'/src/GeneratorInterface/GenExtensions/data/'
     $           //flnm(5)//nn(2:3)//'.tbl'
         else
            print *, 'invalid iset number in setctq6 :', iset
            stop
         endif
c        print *, 'pdf file ', tablefile	 
         open(iu, file=tablefile, status='old', err=100)
 21      call readtbl (iu)
         close (iu)
         isetold=iset
      endif
      return

 100  print *, ' data file ', tablefile, ' cannot be opened '
     >//'in setctq6!!'
      stop
c                             ********************
      end

      subroutine readtbl (nu)
      implicit double precision (a-h,o-z)
      character line*80
      parameter (mxx = 96, mxq = 20, mxf = 5)
      parameter (mxpqx = (mxf + 3) * mxq * mxx)
      common
     > / ctqpar1 / al, xv(0:mxx), tv(0:mxq), upd(mxpqx)
     > / ctqpar2 / nx, nt, nfmx
     > / xqrange / qini, qmax, xmin
     > / qcdtable /  alambda, nfl, iorder
     > / masstbl / amass(6)

      read  (nu, '(a)') line
      read  (nu, '(a)') line
      read  (nu, *) dr, fl, al, (amass(i),i=1,6)
      iorder = nint(dr)
      nfl = nint(fl)
      alambda = al

      read  (nu, '(a)') line
      read  (nu, *) nx,  nt, nfmx

      read  (nu, '(a)') line
      read  (nu, *) qini, qmax, (tv(i), i =0, nt)

      read  (nu, '(a)') line
      read  (nu, *) xmin, (xv(i), i =0, nx)

      do 11 iq = 0, nt
         tv(iq) = log(log (tv(iq) /al))
   11 continue
c
c                  since quark = anti-quark for nfl>2 at this stage,
c                  we read  out only the non-redundent data points
c     no of flavors = nfmx (sea) + 1 (gluon) + 2 (valence)

      nblk = (nx+1) * (nt+1)
      npts =  nblk  * (nfmx+3)
      read  (nu, '(a)') line
      read  (nu, *, iostat=iret) (upd(i), i=1,npts)

      return
c                        ****************************
      end

      function nextun()
c                                 returns an unallocated fortran i/o unit.
      logical ex
c
      do 10 n = 10, 300
         inquire (unit=n, opened=ex)
         if (.not. ex) then
            nextun = n
            return
         endif
 10   continue
      stop ' there is no available i/o unit. '
c               *************************
      end
c

      subroutine polint (xa,ya,n,x,y,dy)

      implicit double precision (a-h, o-z)
c                                        adapted from "numerical recipes"
      parameter (nmax=10)
      dimension xa(n),ya(n),c(nmax),d(nmax)
      ns=1
      dif=abs(x-xa(1))
      do 11 i=1,n
        dift=abs(x-xa(i))
        if (dift.lt.dif) then
          ns=i
          dif=dift
        endif
        c(i)=ya(i)
        d(i)=ya(i)
11    continue
      y=ya(ns)
      ns=ns-1
      do 13 m=1,n-1
        do 12 i=1,n-m
          ho=xa(i)-x
          hp=xa(i+m)-x
          w=c(i+1)-d(i)
          den=ho-hp
          if(den.eq.0.)pause
          den=w/den
          d(i)=hp*den
          c(i)=ho*den
12      continue
        if (2*ns.lt.n-m)then
          dy=c(ns+1)
        else
          dy=d(ns)
          ns=ns-1
        endif
        y=y+dy
13    continue
      return
      end

      double precision function partonx6 (iprtn, xx, qq)

c  given the parton distribution function in the array u in
c  common / pevldt / , this routine interpolates to find
c  the parton distribution at an arbitray point in x and q.
c
      implicit double precision (a-h,o-z)

      parameter (mxx = 96, mxq = 20, mxf = 5)
      parameter (mxqx= mxq * mxx,   mxpqx = mxqx * (mxf+3))

      common
     > / ctqpar1 / al, xv(0:mxx), tv(0:mxq), upd(mxpqx)
     > / ctqpar2 / nx, nt, nfmx
     > / xqrange / qini, qmax, xmin

      dimension fvec(4), fij(4)
      dimension xvpow(0:mxx)
      data onep / 1.00001 /
      data xpow / 0.3d0 /       !**** choice of interpolation variable
      data nqvec / 4 /
      data ientry / 0 /
      save ientry,xvpow

c store the powers used for interpolation on first call...
      if(ientry .eq. 0) then
         ientry = 1

         xvpow(0) = 0d0
         do i = 1, nx
            xvpow(i) = xv(i)**xpow
         enddo
      endif

      x = xx
      q = qq
      tt = log(log(q/al))

c      -------------    find lower end of interval containing x, i.e.,
c                       get jx such that xv(jx) .le. x .le. xv(jx+1)...
      jlx = -1
      ju = nx+1
 11   if (ju-jlx .gt. 1) then
         jm = (ju+jlx) / 2
         if (x .ge. xv(jm)) then
            jlx = jm
         else
            ju = jm
         endif
         goto 11
      endif
c                     ix    0   1   2      jx  jlx         nx-2     nx
c                           |---|---|---|...|---|-x-|---|...|---|---|
c                     x     0  xmin               x                 1
c
      if     (jlx .le. -1) then
        print '(a,1pe12.4)', 'severe error: x <= 0 in partonx6! x = ', x
        stop
      elseif (jlx .eq. 0) then
         jx = 0
      elseif (jlx .le. nx-2) then

c                for interrior points, keep x in the middle, as shown above
         jx = jlx - 1
      elseif (jlx.eq.nx-1 .or. x.lt.onep) then

c                  we tolerate a slight over-shoot of one (onep=1.00001),
c              perhaps due to roundoff or whatever, but not more than that.
c                                      keep at least 4 points >= jx
         jx = jlx - 2
      else
        print '(a,1pe12.4)', 'severe error: x > 1 in partonx6! x = ', x
        stop
      endif
c          ---------- note: jlx uniquely identifies the x-bin; jx does not.

c                       this is the variable to be interpolated in
      ss = x**xpow

      const1 = 0.
      const2 = 0.
      const3 = 0.
      const4 = 0.
      const5 = 0.
      const6 = 0.
      sy2 = 0.
      sy3 = 0.
      s23 = 0.

      if (jlx.ge.2 .and. jlx.le.nx-2) then

c     initiation work for "interior bins": store the lattice points in s...
      svec1 = xvpow(jx)
      svec2 = xvpow(jx+1)
      svec3 = xvpow(jx+2)
      svec4 = xvpow(jx+3)

      s12 = svec1 - svec2
      s13 = svec1 - svec3
      s23 = svec2 - svec3
      s24 = svec2 - svec4
      s34 = svec3 - svec4

      sy2 = ss - svec2
      sy3 = ss - svec3

c constants needed for interpolating in s at fixed t lattice points...
      const1 = s13/s23
      const2 = s12/s23
      const3 = s34/s23
      const4 = s24/s23
      s1213 = s12 + s13
      s2434 = s24 + s34
      sdet = s12*s34 - s1213*s2434
      tmp = sy2*sy3/sdet
      const5 = (s34*sy2-s2434*sy3)*tmp/s12
      const6 = (s1213*sy2-s12*sy3)*tmp/s34

      endif

c         --------------now find lower end of interval containing q, i.e.,
c                          get jq such that qv(jq) .le. q .le. qv(jq+1)...
      jlq = -1
      ju = nt+1
 12   if (ju-jlq .gt. 1) then
         jm = (ju+jlq) / 2
         if (tt .ge. tv(jm)) then
            jlq = jm
         else
            ju = jm
         endif
         goto 12
       endif

      if     (jlq .le. 0) then
         jq = 0
      elseif (jlq .le. nt-2) then
c                                  keep q in the middle, as shown above
         jq = jlq - 1
      else
c                         jlq .ge. nt-1 case:  keep at least 4 points >= jq.
        jq = nt - 3

      endif
c                                   this is the interpolation variable in q

      tdet = 0.
      t12 = 0.
      t13 = 0.
      t23 = 0.
      t24 = 0.
      t34 = 0.
      ty2 = 0.
      ty3 = 0.
      tmp1 = 0.
      tmp2 = 0.

	
      if (jlq.ge.1 .and. jlq.le.nt-2) then
c                                        store the lattice points in t...
      tvec1 = tv(jq)
      tvec2 = tv(jq+1)
      tvec3 = tv(jq+2)
      tvec4 = tv(jq+3)

      t12 = tvec1 - tvec2
      t13 = tvec1 - tvec3
      t23 = tvec2 - tvec3
      t24 = tvec2 - tvec4
      t34 = tvec3 - tvec4

      ty2 = tt - tvec2
      ty3 = tt - tvec3

      tmp1 = t12 + t13
      tmp2 = t24 + t34

      tdet = t12*t34 - tmp1*tmp2

      endif


c get the pdf function values at the lattice points...

      if (iprtn .ge. 3) then
         ip = - iprtn
      else
         ip = iprtn
      endif
      jtmp = ((ip + nfmx)*(nt+1)+(jq-1))*(nx+1)+jx+1

      do it = 1, nqvec

         j1  = jtmp + it*(nx+1)

       if (jx .eq. 0) then
c                          for the first 4 x points, interpolate x^2*f(x,q)
c                           this applies to the two lowest bins jlx = 0, 1
c            we can not put the jlx.eq.1 bin into the "interrior" section
c                           (as we do for q), since upd(j1) is undefined.
         fij(1) = 0
         fij(2) = upd(j1+1) * xv(1)**2
         fij(3) = upd(j1+2) * xv(2)**2
         fij(4) = upd(j1+3) * xv(3)**2
c
c                 use polint which allows x to be anywhere w.r.t. the grid

         call polint (xvpow(0), fij(1), 4, ss, fx, dfx)

         if (x .gt. 0d0)  fvec(it) =  fx / x**2
c                                              pdf is undefined for x.eq.0
       elseif  (jlx .eq. nx-1) then
c                                                this is the highest x bin:

        call polint (xvpow(nx-3), upd(j1), 4, ss, fx, dfx)

        fvec(it) = fx

       else
c                       for all interior points, use jon's in-line function
c                              this applied to (jlx.ge.2 .and. jlx.le.nx-2)
         sf2 = upd(j1+1)
         sf3 = upd(j1+2)

         g1 =  sf2*const1 - sf3*const2
         g4 = -sf2*const3 + sf3*const4

         fvec(it) = (const5*(upd(j1)-g1)
     &               + const6*(upd(j1+3)-g4)
     &               + sf2*sy3 - sf3*sy2) / s23

       endif

      enddo
c                                   we now have the four values fvec(1:4)
c     interpolate in t...

      if (jlq .le. 0) then
c                         1st q-bin, as well as extrapolation to lower q
        call polint (tv(0), fvec(1), 4, tt, ff, dfq)

      elseif (jlq .ge. nt-1) then
c                         last q-bin, as well as extrapolation to higher q
        call polint (tv(nt-3), fvec(1), 4, tt, ff, dfq)
      else
c                         interrior bins : (jlq.ge.1 .and. jlq.le.nt-2)
c       which include jlq.eq.1 and jlq.eq.nt-2, since upd is defined for
c                         the full range qv(0:nt)  (in contrast to xv)
        tf2 = fvec(2)
        tf3 = fvec(3)

        g1 = ( tf2*t13 - tf3*t12) / t23
        g4 = (-tf2*t34 + tf3*t24) / t23

        h00 = ((t34*ty2-tmp2*ty3)*(fvec(1)-g1)/t12
     &    +  (tmp1*ty2-t12*ty3)*(fvec(4)-g4)/t34)

        ff = (h00*ty2*ty3/tdet + tf2*ty3 - tf3*ty2) / t23
      endif

      partonx6 = ff

      return
      end

      double precision function alpcteq(q2, naord)
*********************************************************************
*                                                                   *
*   the alpha_s routine.                                            *
*                                                                   *
*   input :  q2    =  scale in gev**2  (not too low, of course);    *
*            naord =  1 (lo),  2 (nlo).                             *
*                                                                   *
*   output:  alphas_s/(4 pi) for use with the grv(98) partons.      *  
*                                                                   *
*******************************************************i*************
*
      implicit double precision (a - z)
      integer nf, k, i, naord
      dimension lambdal (3:6),  lambdan (3:6), q2thr (3)
*
*...heavy quark thresholds and lambda values :
      data q2thr   /  1.690,  20.25,  32400. /
      data lambdal /  0.247,  0.215,  0.1650, 0.0847  /
      data lambdan /  0.350, 0.326, 0.2260, 0.151 /
*
*...determination of the appropriate number of flavours :
      nf = 3
      do 10 k = 1, 3
      if (q2 .gt. q2thr (k)) then
         nf = nf + 1
      else
          go to 20
       end if
  10   continue
*
*...lo alpha_s and beta function for nlo calculation :
  20   b0 = 11.- 2./3.* nf
       b1 = 102.- 38./3.* nf
       b10 = b1 / (b0*b0)
       if (naord .eq. 1) then
         lam2 = lambdal (nf) * lambdal (nf)
         alp  = 1./(b0 * dlog (q2/lam2))
         go to 1
       else if (naord .eq. 2) then
         lam2 = lambdan (nf) * lambdan (nf)
         b1 = 102.- 38./3.* nf
         b10 = b1 / (b0*b0)
       else
         write (6,91)
  91     format ('invalid choice for order in alpha_s')
         stop
       end if
*
*...start value for nlo iteration :
       lq2 = dlog (q2 / lam2)
       alp = 1./(b0*lq2) * (1.- b10*dlog(lq2)/lq2)
*
*...exact nlo value, found via newton procedure :
       do 2 i = 1, 6
       xl  = dlog (1./(b0*alp) + b10)
       xlp = dlog (1./(b0*alp*1.01) + b10)
       xlm = dlog (1./(b0*alp*0.99) + b10)
       y  = lq2 - 1./ (b0*alp) + b10 * xl
       y1 = (- 1./ (b0*alp*1.01) + b10 * xlp
     1       + 1./ (b0*alp*0.99) - b10 * xlp) / (0.02d0*alp)
       alp = alp - y/y1
  2    continue
*
*...output :
  1    alpcteq = alp
       return
       end
c*******************************************************************
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                                                                   *
*     g r v  -  p r o t o n  - p a r a m e t r i z a t i o n s      *
*                                                                   *
*                          1998 update                              *
*                                                                   *
*                  for a detailed explanation see                   *
*                   m. glueck, e. reya, a. vogt :                   *
*        hep-ph/9806404  =  do-th 98/07  =  wue-itp-98-019          *
*                  (to appear in eur. phys. j. c)                   *
*                                                                   *
*   this package contains subroutines returning the light-parton    *
*   distributions in nlo (for the msbar and dis schemes) and lo;    * 
*   the respective light-parton, charm, and bottom contributions    *
*   to f2(electromagnetic); and the scale dependence of alpha_s.    *
*                                                                   *
*   the parton densities and f2 values are calculated from inter-   *
*   polation grids covering the regions                             *
*         q^2/gev^2  between   0.8   and  1.e6 ( 1.e4 for f2 )      *
*            x       between  1.e-9  and   1.                       *
*   any call outside these regions stops the program execution.     *
*                                                                   *
*   at q^2 = mz^2, alpha_s reads  0.114 (0.125) in nlo (lo); the    *
*   heavy quark thresholds, qh^2 = mh^2, in the beta function are   *
*            mc = 1.4 gev,  mb = 4.5 gev,  mt = 175 gev.            *
*   note that the nlo alpha_s running is different from grv(94).    * 
*                                                                   *
*    questions, comments etc to:  avogt@physik.uni-wuerzburg.de     *
*                                                                   *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*
*
*
*
      subroutine grv98pa (iset, x, q2, uv, dv, us, ds, ss, gl)
*********************************************************************
*                                                                   *
*   the parton routine.                                             *
*                                     __                            *
*   input:   iset =  1 (lo),  2 (nlo, ms), or  3 (nlo, dis)         *
*            x  =  bjorken-x        (between  1.e-9 and 1.)         *
*            q2 =  scale in gev**2  (between  0.8 and 1.e6)         *
*                                                                   *
*   output:  uv = u - u(bar),  dv = d - d(bar),  us = u(bar),       *
*            ds = d(bar),  ss = s = s(bar),  gl = gluon.            *
*            always x times the distribution is returned.           *
*                                                                   *
*   common:  the main program or the calling routine has to have    *
*            a common block  common / intinip / iinip , and the     *
*            integer variable  iinip  has always to be zero when    *
*            grv98pa is called for the first time or when  iset     *
*            has been changed.                                      *
*                                                                   *
*   grids:   1. grv98lo.grid, 2. grv98nlm.grid, 3. grv98nld.grid,   *
*            (1+1809 lines with 6 columns, 4 significant figures)   *
*                                                                   *
*******************************************************i*************
*
      implicit double precision (a-h, o-z)
      parameter (npart=6, nx=68, nq=27, narg=2)
      dimension xuvf(nx,nq), xdvf(nx,nq), xdef(nx,nq), xudf(nx,nq),
     1          xsf(nx,nq), xgf(nx,nq), parton (npart,nq,nx-1), 
     2          qs(nq), xb(nx), xt(narg), na(narg), arrf(nx+nq) 
      character*80 line
      common /intinip/iinip
	save xuvf, xdvf, xdef, xudf, xsf, xgf, na, arrf
*
*...bjorken-x and q**2 values of the grid :
       data qs / 0.8e0, 
     1           1.0e0, 1.3e0, 1.8e0, 2.7e0, 4.0e0, 6.4e0,
     2           1.0e1, 1.6e1, 2.5e1, 4.0e1, 6.4e1,
     3           1.0e2, 1.8e2, 3.2e2, 5.7e2,
     4           1.0e3, 1.8e3, 3.2e3, 5.7e3,
     5           1.0e4, 2.2e4, 4.6e4,
     6           1.0e5, 2.2e5, 4.6e5, 
     7           1.e6 /
       data xb / 1.0e-9, 1.8e-9, 3.2e-9, 5.7e-9, 
     a           1.0e-8, 1.8e-8, 3.2e-8, 5.7e-8, 
     b           1.0e-7, 1.8e-7, 3.2e-7, 5.7e-7, 
     c           1.0e-6, 1.4e-6, 2.0e-6, 3.0e-6, 4.5e-6, 6.7e-6,
     1           1.0e-5, 1.4e-5, 2.0e-5, 3.0e-5, 4.5e-5, 6.7e-5,
     2           1.0e-4, 1.4e-4, 2.0e-4, 3.0e-4, 4.5e-4, 6.7e-4,
     3           1.0e-3, 1.4e-3, 2.0e-3, 3.0e-3, 4.5e-3, 6.7e-3,
     4           1.0e-2, 1.4e-2, 2.0e-2, 3.0e-2, 4.5e-2, 0.06, 0.08,
     5           0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
     6           0.3, 0.325, 0.35, 0.375, 0.4,  0.45, 0.5, 0.55,
     7           0.6, 0.65,  0.7,  0.75,  0.8,  0.85, 0.9, 0.95, 1. /
*
*...check of x and q2 values : 
      if ( (x.lt.0.99d-9) .or. (x.gt.1.d0) ) then
         write(6,91) 
  91     format (2x,'parton interpolation: x out of range')
         stop
      endif
      if ( (q2.lt.0.799) .or. (q2.gt.1.01e6) ) then
c         write(6,92) 
c  92     format (2x,'parton interpolation: q2 out of range')
         q2=1.0d6
c	   stop
      endif
      if (iinip .ne. 0) goto 16
*
*...initialization, if required :
*
*    selection and reading of the grid : 
*    (comment: first number in the first line of the grid)
      if (iset .eq. 1) then
        open (11,file='grv98lo.grid',status='old')   ! 7.332e-05
      else if (iset .eq. 2) then
        open (11,file='grv98nlm.grid',status='old')  ! 1.015e-04
      else if (iset .eq. 3) then
        open (11,file='grv98nld.grid',status='old')  ! 1.238e-04
      else
        write(6,93)
  93    format (2x,'no or invalid parton set choice')
        stop
      end if
      iinip = 1
      read(11,89) line
  89  format(a80)
      do 15 m = 1, nx-1 
      do 15 n = 1, nq
      read(11,90) parton(1,n,m), parton(2,n,m), parton(3,n,m), 
     1            parton(4,n,m), parton(5,n,m), parton(6,n,m) 
  90  format (6(1pe10.3))
  15  continue
      close(11)
*
*....arrays for the interpolation subroutine :
      do 10 iq = 1, nq
      do 20 ix = 1, nx-1
        xb0v = xb(ix)**0.5 
        xb0s = xb(ix)**(-0.2) 
        xb1 = 1.-xb(ix)
        xuvf(ix,iq) = parton(1,iq,ix) / (xb1**3 * xb0v)
        xdvf(ix,iq) = parton(2,iq,ix) / (xb1**4 * xb0v)
        xdef(ix,iq) = parton(3,iq,ix) / (xb1**7 * xb0v) 
        xudf(ix,iq) = parton(4,iq,ix) / (xb1**7 * xb0s)
        xsf(ix,iq)  = parton(5,iq,ix) / (xb1**7 * xb0s)
        xgf(ix,iq)  = parton(6,iq,ix) / (xb1**5 * xb0s)
  20  continue
        xuvf(nx,iq) = 0.e0
        xdvf(nx,iq) = 0.e0
        xdef(nx,iq) = 0.e0
        xudf(nx,iq) = 0.e0
        xsf(nx,iq)  = 0.e0
        xgf(nx,iq)  = 0.e0
  10  continue  
      na(1) = nx
      na(2) = nq
      do 30 ix = 1, nx
        arrf(ix) = dlog(xb(ix))
  30  continue
      do 40 iq = 1, nq
        arrf(nx+iq) = dlog(qs(iq))
  40  continue
*
*...continuation, if initialization was done previously.
*
  16  continue
*
*...interpolation :
      xt(1) = dlog(x)
      xt(2) = dlog(q2)
      x1 = 1.- x
      xv = x**0.5
      xs = x**(-0.2)
      uv = fint(narg,xt,na,arrf,xuvf) * x1**3 * xv
      dv = fint(narg,xt,na,arrf,xdvf) * x1**4 * xv
      de = fint(narg,xt,na,arrf,xdef) * x1**7 * xv
      ud = fint(narg,xt,na,arrf,xudf) * x1**7 * xs
      us = 0.5 * (ud - de)
      ds = 0.5 * (ud + de)
      ss = fint(narg,xt,na,arrf,xsf)  * x1**7 * xs
      gl = fint(narg,xt,na,arrf,xgf)  * x1**5 * xs 
*
 60   return
      end
*
*
*
*
      subroutine grv98f2 (iset, x, q2, f2l, f2c, f2b, f2)
*********************************************************************
*                                                                   *
*   the f2 routine.                                                 *
*                                                                   *
*   input :  iset = 1 (lo),  2 (nlo).                               *
*            x  =  bjorken-x        (between  1.e-9 and 1.)         *
*            q2 =  scale in gev**2  (between  0.8 and 1.e4)         *
*                                                                   *
*   output:  f2l = f2(light), f2c = f2(charm), f2b = f2(bottom,)    *
*            f2  = sum, all given for electromagnetic proton dis.   *
*                                                                   *
*   common:  the main program or the calling routine has to have    *
*            a common block  common / intinif / iinif , and the     *
*            integer variable  iinif  has always to be zero when    *
*            grv98f2 is called for the first time or when  iset     *
*            has been changed.                                      *
*                                                                   *
*   grids:   1. grv98lof.grid, 2. grv98nlf.grid.                    *
*            (1+1407 lines with 3 columns, 4 significant figures)   *
*                                                                   *
*******************************************************i*************
*
      implicit double precision (a-h, o-z)
      parameter (nstrf=3, nx=68, nq=21, narg=2)
      dimension xf2lf(nx,nq), xf2cf(nx,nq), xf2bf(nx,nq), 
     1          strfct (nstrf,nq,nx-1), qs(nq), xb(nx), 
     3          xt(narg), na(narg), arrf(nx+nq) 
      character*80 line
      common / intinif / iinif
      save xf2lf, xf2cf, xf2bf, na, arrf
*
*...bjorken-x and q**2 values of the grid :
       data qs / 0.8e0, 
     1           1.0e0, 1.3e0, 1.8e0, 2.7e0, 4.0e0, 6.4e0,
     2           1.0e1, 1.6e1, 2.5e1, 4.0e1, 6.4e1,
     3           1.0e2, 1.8e2, 3.2e2, 5.7e2,
     4           1.0e3, 1.8e3, 3.2e3, 5.7e3,
     5           1.0e4/ 
       data xb / 1.0e-9, 1.8e-9, 3.2e-9, 5.7e-9, 
     a           1.0e-8, 1.8e-8, 3.2e-8, 5.7e-8, 
     b           1.0e-7, 1.8e-7, 3.2e-7, 5.7e-7, 
     c           1.0e-6, 1.4e-6, 2.0e-6, 3.0e-6, 4.5e-6, 6.7e-6,
     1           1.0e-5, 1.4e-5, 2.0e-5, 3.0e-5, 4.5e-5, 6.7e-5,
     2           1.0e-4, 1.4e-4, 2.0e-4, 3.0e-4, 4.5e-4, 6.7e-4,
     3           1.0e-3, 1.4e-3, 2.0e-3, 3.0e-3, 4.5e-3, 6.7e-3,
     4           1.0e-2, 1.4e-2, 2.0e-2, 3.0e-2, 4.5e-2, 0.06, 0.08,
     5           0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
     6           0.3, 0.325, 0.35, 0.375, 0.4,  0.45, 0.5, 0.55,
     7           0.6, 0.65,  0.7,  0.75,  0.8,  0.85, 0.9, 0.95, 1. /
*
*...check of x and q2 values : 
      if ( (x.lt.0.99d-9) .or. (x.gt.1.d0) ) then
         write(6,91) 
  91     format (2x,'str.fct. interpolation: x out of range')
         stop
      endif
      if ( (q2.lt.0.799) .or. (q2.gt.1.01e4) ) then
         write(6,92) 
  92     format (2x,'str.fct. interpolation: q2 out of range')
         stop
      endif
      if (iinif .ne. 0) goto 16
*
*...initialization, if required :
*
*    selection and reading of the grid : 
*    (comment: first number in the first line of the grid)
      if (iset .eq. 1) then
        open (11,file='grv98lof.grid',status='old')  !  7.907e-01
      else if (iset.eq.2) then
        open (11,file='grv98nlf.grid',status='old')  !  9.368e-01
      else
        write(6,93)
  93    format (2x,'no or invalid str.fct. set choice')
        stop
      end if
      iinif = 1
      read(11,89) line
  89  format(a80)
      do 15 m = 1, nx-1 
      do 15 n = 1, nq
      read(11,90) strfct(1,n,m), strfct(2,n,m), strfct(3,n,m) 
  90  format (3(1pe10.3))
  15  continue
      close(11)
*
*....arrays for the interpolation subroutine :
      do 10 iq = 1, nq
      do 20 ix = 1, nx-1
        xbs = xb(ix)**0.2 
        xb1 = 1.-xb(ix)
        xf2lf(ix,iq) = strfct(1,iq,ix) / xb1**3 * xbs
        xf2cf(ix,iq) = strfct(2,iq,ix) / xb1**7 * xbs
        xf2bf(ix,iq) = strfct(3,iq,ix) / xb1**7 * xbs 
  20  continue
        xf2lf(nx,iq) = 0.e0
        xf2cf(nx,iq) = 0.e0
        xf2bf(nx,iq) = 0.e0
  10  continue  
      na(1) = nx
      na(2) = nq
      do 30 ix = 1, nx
        arrf(ix) = dlog(xb(ix))
  30  continue
      do 40 iq = 1, nq
        arrf(nx+iq) = dlog(qs(iq))
  40  continue
*
*...continuation, if initialization was done previously.
*
  16  continue
*
*...interpolation :
      xt(1) = dlog(x)
      xt(2) = dlog(q2)
      x1 = 1.- x
      xs = x**(-0.2)
      f2l = fint(narg,xt,na,arrf,xf2lf) * x1**3 * xs
      f2c = fint(narg,xt,na,arrf,xf2cf) * x1**7 * xs
      f2b = fint(narg,xt,na,arrf,xf2bf) * x1**7 * xs
      f2  = f2l + f2c + f2b
*
 60   return
      end
*
*
      double precision function fint(narg,arg,nent,ent,table)
*********************************************************************
*                                                                   *
*   the interpolation routine (cern library routine e104)           *
*                                                                   *
*********************************************************************
      implicit double precision (a-h, o-z)
      dimension arg(5),nent(5),ent(1000),table(10000)
      dimension d(5),ncomb(5),ient(5)

      kd=1
      m=1
      ja=1

      do 5 ii=1,narg
        ncomb(ii)=1
        jb=ja-1+nent(ii)
        do 2 j=ja,jb
          if (arg(ii).le.ent(j)) go to 3
    2   continue
        j=jb
    3   if (j.ne.ja) go to 4
        j=j+1
    4   jr=j-1
        d(ii)=(ent(j)-arg(ii))/(ent(j)-ent(jr))
        ient(ii)=j-ja
        kd=kd+ient(ii)*m
        m=m*nent(ii)
        ja=jb+1
    5 continue

      fint=0.
   10 fac=1.
      iadr=kd
      ifadr=1

      do 15 i=1,narg
        if (ncomb(i).eq.0) go to 12
        fac=fac*(1.-d(i))
        go to 15
   12   fac=fac*d(i)
        iadr=iadr-ifadr
        ifadr=ifadr*nent(i)
   15 continue

      fint=fint+fac*table(iadr)
      il=narg
   40 if (ncomb(il).eq.0) go to 80
      ncomb(il)=0
      if (il.eq.narg) go to 10
      il=il+1
      do 50  k=il,narg
   50    ncomb(k)=1
      go to 10
   80 il=il-1
      if(il.ne.0) go to 40
      
	return
      end
*
*
      double precision function alpgrv(q2, naord)
*********************************************************************
*                                                                   *
*   the alpha_s routine.                                            *
*                                                                   *
*   input :  q2    =  scale in gev**2  (not too low, of course);    *
*            naord =  1 (lo),  2 (nlo).                             *
*                                                                   *
*   output:  alphas_s/(4 pi) for use with the grv(98) partons.      *  
*                                                                   *
*******************************************************i*************
*
      implicit double precision (a - z)
      integer nf, k, i, naord
      dimension lambdal (3:6),  lambdan (3:6), q2thr (3)
*
*...heavy quark thresholds and lambda values :
      data q2thr   /  1.960,  20.25,  30625. /
      data lambdal / 0.2041, 0.1750, 0.1320, 0.0665 /
      data lambdan / 0.2994, 0.2460, 0.1677, 0.0678 /
*
*...determination of the appropriate number of flavours :
      nf = 3
      do 10 k = 1, 3
      if (q2 .gt. q2thr (k)) then
         nf = nf + 1
      else
          go to 20
       end if
  10   continue
*
*...lo alpha_s and beta function for nlo calculation :
  20   b0 = 11.- 2./3.* nf
       b1 = 102.- 38./3.* nf
       b10 = b1 / (b0*b0)
       if (naord .eq. 1) then
         lam2 = lambdal (nf) * lambdal (nf)
         alp  = 1./(b0 * dlog (q2/lam2))
         go to 1
       else if (naord .eq. 2) then
         lam2 = lambdan (nf) * lambdan (nf)
         b1 = 102.- 38./3.* nf
         b10 = b1 / (b0*b0)
       else
         write (6,91)
  91     format ('invalid choice for order in alpha_s')
         stop
       end if
*
*...start value for nlo iteration :
       lq2 = dlog (q2 / lam2)
       alp = 1./(b0*lq2) * (1.- b10*dlog(lq2)/lq2)
*
*...exact nlo value, found via newton procedure :
       do 2 i = 1, 6
       xl  = dlog (1./(b0*alp) + b10)
       xlp = dlog (1./(b0*alp*1.01) + b10)
       xlm = dlog (1./(b0*alp*0.99) + b10)
       y  = lq2 - 1./ (b0*alp) + b10 * xl
       y1 = (- 1./ (b0*alp*1.01) + b10 * xlp
     1       + 1./ (b0*alp*0.99) - b10 * xlp) / (0.02d0*alp)
       alp = alp - y/y1
  2    continue
*
*...output :
  1    alpgrv = alp
       return
       end
c*******************************************************************
      subroutine mrstlo(x,q,mode,upv,dnv,usea,dsea,str,chm,bot,glu)
c***************************************************************c
c								c
c  this is a package for the new mrst 2001 lo parton            c
c  distributions.                                               c     
c  reference: a.d. martin, r.g. roberts, w.j. stirling and      c
c  r.s. thorne, hep-ph/0201xxx                                  c
c                                                               c
c  there is 1 pdf set corresponding to mode = 1                 c
c                                                               c
c  mode=1 gives the default set with lambda(msbar,nf=4) = 0.220 c
c  corresponding to alpha_s(m_z) of 0.130                       c
c  this set reads a grid whose first number is 0.02868          c
c                                                               c
c   this subroutine uses an improved interpolation procedure    c 
c   for extracting values of the pdf's from the grid            c
c                                                               c
c         comments to : w.j.stirling@durham.ac.uk               c
c                                                               c
c***************************************************************c
      implicit real*8(a-h,o-z)
      data xmin,xmax,qsqmin,qsqmax/1d-5,1d0,1.25d0,1d7/
      q2=q*q
      if(q2.lt.qsqmin.or.q2.gt.qsqmax) print *, 99,q2
      if(x.lt.xmin.or.x.gt.xmax)       print *, 98,x
          if(mode.eq.1) then
        call mrst1(x,q2,upv,dnv,usea,dsea,str,chm,bot,glu) 
      endif 
  99  format('  warning:  q^2 value is out of range   ','q2= ',e10.5)
  98  format('  warning:   x  value is out of range   ','x= ',e10.5)
      return
      end
c********************************************************
      subroutine mrst1(x,qsq,upv,dnv,usea,dsea,str,chm,bot,glu)
      implicit real*8(a-h,o-z)
      parameter(nx=49,nq=37,np=8,nqc0=2,nqb0=11,nqc=35,nqb=26)
      real*8 f1(nx,nq),f2(nx,nq),f3(nx,nq),f4(nx,nq),f5(nx,nq),
     .f6(nx,nq),f7(nx,nq),f8(nx,nq),fc(nx,nqc),fb(nx,nqb)
      real*8 qq(nq),xx(nx),cc1(nx,nq,4,4),cc2(nx,nq,4,4),
     .cc3(nx,nq,4,4),cc4(nx,nq,4,4),cc6(nx,nq,4,4),cc8(nx,nq,4,4),
     .ccc(nx,nqc,4,4),ccb(nx,nqb,4,4)
      real*8 xxl(nx),qql(nq),qqlc(nqc),qqlb(nqb)
      data xx/1d-5,2d-5,4d-5,6d-5,8d-5,
     .	      1d-4,2d-4,4d-4,6d-4,8d-4,
     .	      1d-3,2d-3,4d-3,6d-3,8d-3,
     .	      1d-2,1.4d-2,2d-2,3d-2,4d-2,6d-2,8d-2,
     .	   .1d0,.125d0,.15d0,.175d0,.2d0,.225d0,.25d0,.275d0,
     .	   .3d0,.325d0,.35d0,.375d0,.4d0,.425d0,.45d0,.475d0,
     .	   .5d0,.525d0,.55d0,.575d0,.6d0,.65d0,.7d0,.75d0,
     .	   .8d0,.9d0,1d0/
      data qq/1.25d0,1.5d0,2d0,2.5d0,3.2d0,4d0,5d0,6.4d0,8d0,1d1,
     .        1.2d1,1.8d1,2.6d1,4d1,6.4d1,1d2,
     .        1.6d2,2.4d2,4d2,6.4d2,1d3,1.8d3,3.2d3,5.6d3,1d4,
     .        1.8d4,3.2d4,5.6d4,1d5,1.8d5,3.2d5,5.6d5,1d6,
     .        1.8d6,3.2d6,5.6d6,1d7/
      data xmin,xmax,qsqmin,qsqmax/1d-5,1d0,1.25d0,1d7/
      data init/0/
      save
      xsave=x
      q2save=qsq
      if(init.ne.0) goto 10
        open(unit=33,file='lo2002.dat',status='old')
        do 20 n=1,nx-1
        do 20 m=1,nq
        read(33,50)f1(n,m),f2(n,m),f3(n,m),f4(n,m),
     .		  f5(n,m),f7(n,m),f6(n,m),f8(n,m)
c notation: 1=uval 2=val 3=glue 4=usea 5=chm 6=str 7=btm 8=dsea
  20  continue
      do 40 m=1,nq
      f1(nx,m)=0.d0
      f2(nx,m)=0.d0
      f3(nx,m)=0.d0
      f4(nx,m)=0.d0
      f5(nx,m)=0.d0
      f6(nx,m)=0.d0
      f7(nx,m)=0.d0
      f8(nx,m)=0.d0
  40  continue
      do n=1,nx
      xxl(n)=dlog(xx(n))
      enddo
      do m=1,nq
      qql(m)=dlog(qq(m))
      enddo

      call jeppe1(nx,nq,xxl,qql,f1,cc1)
      call jeppe1(nx,nq,xxl,qql,f2,cc2)
      call jeppe1(nx,nq,xxl,qql,f3,cc3)
      call jeppe1(nx,nq,xxl,qql,f4,cc4)
      call jeppe1(nx,nq,xxl,qql,f6,cc6)
      call jeppe1(nx,nq,xxl,qql,f8,cc8)

      emc2=2.045
      emb2=18.5

      do 44 m=1,nqc
      qqlc(m)=qql(m+nqc0)
      do 44 n=1,nx
      fc(n,m)=f5(n,m+nqc0)
   44 continue
      qqlc(1)=dlog(emc2)
      call jeppe1(nx,nqc,xxl,qqlc,fc,ccc)

      do 45 m=1,nqb
      qqlb(m)=qql(m+nqb0)
      do 45 n=1,nx
      fb(n,m)=f7(n,m+nqb0)
   45 continue
      qqlb(1)=dlog(emb2)
      call jeppe1(nx,nqb,xxl,qqlb,fb,ccb)


      init=1
   10 continue
      
      xlog=dlog(x)
      qsqlog=dlog(qsq)

      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc1,upv)
      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc2,dnv)
      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc3,glu)
      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc4,usea)
      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc6,str)
      call jeppe2(xlog,qsqlog,nx,nq,xxl,qql,cc8,dsea)

      chm=0.d0
      if(qsq.gt.emc2) then 
      call jeppe2(xlog,qsqlog,nx,nqc,xxl,qqlc,ccc,chm)
      endif

      bot=0.d0
      if(qsq.gt.emb2) then 
      call jeppe2(xlog,qsqlog,nx,nqb,xxl,qqlb,ccb,bot)
      endif

      x=xsave
      qsq=q2save

	close(33)

      return
   50 format(8f10.5)
      end
c************************************************ 
      subroutine jeppe1(nx,my,xx,yy,ff,cc)
      implicit real*8(a-h,o-z)
      parameter(nnx=49,mmy=37)
      dimension xx(nx),yy(my),ff(nx,my),ff1(nnx,mmy),ff2(nnx,mmy),
     xff12(nnx,mmy),yy0(4),yy1(4),yy2(4),yy12(4),z(16),wt(16,16),
     xcl(16),cc(nx,my,4,4),iwt(16,16)

      data iwt/1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     x		  0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
     x		  -3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0,
     x		  2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0,
     x		  0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
     x		  0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
     x		  0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1,
     x		  0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1,
     x		  -3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0,
     x		  0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0,
     x		  9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2,
     x		  -6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2,
     x		  2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
     x		  0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0,
     x		  -6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1,
     x		  4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1/


      do 42 m=1,my
      dx=xx(2)-xx(1)
      ff1(1,m)=(ff(2,m)-ff(1,m))/dx
      dx=xx(nx)-xx(nx-1)
      ff1(nx,m)=(ff(nx,m)-ff(nx-1,m))/dx
      do 41 n=2,nx-1
      ff1(n,m)=polderiv(xx(n-1),xx(n),xx(n+1),ff(n-1,m),ff(n,m),
     xff(n+1,m))
   41 continue
   42 continue

      do 44 n=1,nx
      dy=yy(2)-yy(1)
      ff2(n,1)=(ff(n,2)-ff(n,1))/dy
      dy=yy(my)-yy(my-1)
      ff2(n,my)=(ff(n,my)-ff(n,my-1))/dy
      do 43 m=2,my-1
      ff2(n,m)=polderiv(yy(m-1),yy(m),yy(m+1),ff(n,m-1),ff(n,m),
     xff(n,m+1))
   43 continue
   44 continue

      do 46 m=1,my
      dx=xx(2)-xx(1)
      ff12(1,m)=(ff2(2,m)-ff2(1,m))/dx
      dx=xx(nx)-xx(nx-1)
      ff12(nx,m)=(ff2(nx,m)-ff2(nx-1,m))/dx
      do 45 n=2,nx-1
      ff12(n,m)=polderiv(xx(n-1),xx(n),xx(n+1),ff2(n-1,m),ff2(n,m),
     xff2(n+1,m))
   45 continue
   46 continue

      do 53 n=1,nx-1
      do 52 m=1,my-1
      d1=xx(n+1)-xx(n)
      d2=yy(m+1)-yy(m)
      d1d2=d1*d2

      yy0(1)=ff(n,m)
      yy0(2)=ff(n+1,m)
      yy0(3)=ff(n+1,m+1)
      yy0(4)=ff(n,m+1)

      yy1(1)=ff1(n,m)
      yy1(2)=ff1(n+1,m)
      yy1(3)=ff1(n+1,m+1)
      yy1(4)=ff1(n,m+1)

      yy2(1)=ff2(n,m)
      yy2(2)=ff2(n+1,m)
      yy2(3)=ff2(n+1,m+1)
      yy2(4)=ff2(n,m+1)

      yy12(1)=ff12(n,m)
      yy12(2)=ff12(n+1,m)
      yy12(3)=ff12(n+1,m+1)
      yy12(4)=ff12(n,m+1)

      do 47 k=1,4
      z(k)=yy0(k)
      z(k+4)=yy1(k)*d1
      z(k+8)=yy2(k)*d2
      z(k+12)=yy12(k)*d1d2
   47 continue

      do 49 l=1,16
      xxd=0.
      do 48 k=1,16
      xxd=xxd+iwt(k,l)*z(k)
   48 continue
      cl(l)=xxd
   49 continue
      l=0
      do 51 k=1,4
      do 50 j=1,4
      l=l+1
      cc(n,m,k,j)=cl(l)
   50 continue
   51 continue
   52 continue
   53 continue
      return
      end
c***************************************
      subroutine jeppe2(x,y,nx,my,xx,yy,cc,z)
      implicit real*8(a-h,o-z)
      dimension xx(nx),yy(my),cc(nx,my,4,4)      

      n=locx(xx,nx,x)
      m=locx(yy,my,y)

      t=(x-xx(n))/(xx(n+1)-xx(n))
      u=(y-yy(m))/(yy(m+1)-yy(m))

      z=0.
      do 1 l=4,1,-1
      z=t*z+((cc(n,m,l,4)*u+cc(n,m,l,3))*u
     .       +cc(n,m,l,2))*u+cc(n,m,l,1)
    1 continue
      return
      end
c**************************************
      integer function locx(xx,nx,x)
      implicit real*8(a-h,o-z)
      dimension xx(nx)
      if(x.le.xx(1)) then
      locx=1
      return
      endif
      if(x.ge.xx(nx)) then 
      locx=nx-1  
      return
      endif
      ju=nx+1
      jl=0
    1 if((ju-jl).le.1) go to 2
      jm=(ju+jl)/2
      if(x.ge.xx(jm)) then
      jl=jm
      else
      ju=jm
      endif
      go to 1
    2 locx=jl
      return
      end
c*******************************************
      real*8 function  polderiv(x1,x2,x3,y1,y2,y3)
      implicit real*8(a-h,o-z)
      polderiv=(x3*x3*(y1-y2)-2.0*x2*(x3*(y1-y2)+x1*
     .(y2-y3))+x2*x2*(y1-y3)+x1*x1*(y2-y3))/((x1-x2)*(x1-x3)*(x2-x3))
      return
      end
c**************************************************
      double precision function alpmsrt(scale,lambda,iorder)
c.
c.   the qcd coupling used in the mrs analysis.
c.   automatically corrects for nf=3,4,5 with thresholds at m_q.
c.   the following parameters must be supplied:
c.       scale   =  the qcd scale in gev (real)
c.       lambda  =  the 4 flavour msbar lambda parameter in gev (real)
c.       iorder  =  0 for lo, 1 for nlo and 2 for nnlo
c. 
      implicit real*8 (a-h,o-z)
      real*8 lambda
      data tol,qsct,qsdt/5d-4,74d0,8.18d0/

c-ap initialize to avoid compiler warnings
      alpmsrt =0.

      pi=4d0*datan(1d0)
      iord=iorder
      ith=0
      flav=4d0
      al=lambda
      al2=lambda*lambda
      qs=scale*scale
      t=dlog(qs/al2)
      tt=t
c.
      qsctt=qsct/4.
      qsdtt=qsdt/4.
      if(qs.lt.0.5d0) then   !!  running stops below 0.5
          qs=0.5d0
          t=dlog(qs/al2)
          tt=t
      endif

      alfqc5=0.
      alfqc4=0.
      alfqc3=0.

      if(qs.gt.qsctt) go	to 12  
      if(qs.lt.qsdtt) go	to 312  
   11 continue
      b0=11-2.*flav/3. 
      x1=4.*pi/b0
  5   continue    
      if(iord.eq.0) then
      alpmsrt=x1/t
      elseif(iord.eq.1) then
      b1=102.-38.*flav/3.
      x2=b1/b0**2
      as2=x1/t*(1.-x2*dlog(t)/t)
  95    as=as2
        f=-t+x1/as-x2*dlog(x1/as+x2)
        fp=-x1/as**2*(1.-x2/(x1/as+x2))
        as2=as-f/fp
        del=abs(f/fp/as)
        if((del-tol).gt.0.) go to 95
      alpmsrt=as2
      elseif(iord.eq.2) then
      alpmsrt=qwikalf(t,2,flav)
      endif
      if(ith.eq.0) return
      go to (13,14,15) ith
      print *, 'here'
      go to 5
   12 ith=1
      t=dlog(qsctt/al2)
      go to 11
   13 alfqc4=alpmsrt
      flav=5.   
      ith=2
      go to 11
   14 alfqc5=alpmsrt
      ith=3
      t=tt
      go to 11
   15 alfqs5=alpmsrt
      alfinv=1./alfqs5+1./alfqc4-1./alfqc5
      alpmsrt=1./alfinv
      return

  311 continue
      b0=11-2.*flav/3. 
      x1=4.*pi/b0
      if(iord.eq.0) then
      alpmsrt=x1/t
      elseif(iord.eq.1) then
      b1=102.-38.*flav/3.
      x2=b1/b0**2
      as2=x1/t*(1.-x2*dlog(t)/t)
   35 as=as2
      f=-t+x1/as-x2*dlog(x1/as+x2)
      fp=-x1/as**2*(1.-x2/(x1/as+x2))
      as2=as-f/fp
      del=abs(f/fp/as)
      if((del-tol).gt.0.) go to 35
      elseif(iord.eq.2) then
      alpmsrt=qwikalf(t,2,flav)
      endif
      if(ith.eq.0) return

      go to (313,314,315) ith
  312 ith=1
      t=dlog(qsdtt/al2)
      go to 311
  313 alfqc4=alpmsrt
      flav=3.   
      ith=2
      go to 311
  314 alfqc3=alpmsrt
      ith=3
      t=tt
      go to 311
  315 alfqs3=alpmsrt
      alfinv=1./alfqs3+1./alfqc4-1./alfqc3
      alpmsrt=1./alfinv
      return
      end
c***********************************
      double precision function qwikalf(t,iord,flav)
      implicit real*8(a-h,o-z)
      dimension z3(6),z4(6),z5(6),zz3(6),zz4(6),zz5(6)
      data z3/ -.161667e+01,0.954244e+01,
     .0.768623e+01,0.101523e+00,-.360127e-02,0.457867e-04/
      data z4/ -.172239e+01,0.831185e+01,
     .0.721463e+01,0.835531e-01,-.285436e-02,0.349129e-04/
      data z5/ -.872190e+00,0.572816e+01,
     .0.716119e+01,0.195884e-01,-.300199e-03,0.151741e-05/
      data zz3/-.155611e+02,0.168406e+02,
     .0.603014e+01,0.257682e+00,-.970217e-02,0.127628e-03/
      data zz4/-.106762e+02,0.118497e+02,0.664964e+01,
     .0.112996e+00,-.317551e-02,0.302434e-04/
      data zz5/-.531860e+01,0.708503e+01,0.698352e+01,
     .0.274170e-01,-.426894e-03,0.217591e-05/
      data pi/3.14159/
      nfm2=flav-2.
      x=dsqrt(t)
      x2=x*x
      x3=x*x2
      x4=x*x3
      x5=x*x4
      go to (1,2) iord  !!iord change!!
    1 go to (3,4,5) nfm2
    3 y=z3(1)+z3(2)*x+z3(3)*x2+z3(4)*x3+z3(5)*x4+z3(6)*x5
      go to 10
    4 y=z4(1)+z4(2)*x+z4(3)*x2+z4(4)*x3+z4(5)*x4+z4(6)*x5
      go to 10
    5 y=z5(1)+z5(2)*x+z5(3)*x2+z5(4)*x3+z5(5)*x4+z5(6)*x5
      go to 10
    2 go to (6,7,8) nfm2
    6 y=zz3(1)+zz3(2)*x+zz3(3)*x2+zz3(4)*x3+zz3(5)*x4+zz3(6)*x5
      go to 10
    7 y=zz4(1)+zz4(2)*x+zz4(3)*x2+zz4(4)*x3+zz4(5)*x4+zz4(6)*x5
      go to 10
    8 y=zz5(1)+zz5(2)*x+zz5(3)*x2+zz5(4)*x3+zz5(5)*x4+zz5(6)*x5
      go to 10
   10 qwikalf=4.*pi/y
      return
      end
