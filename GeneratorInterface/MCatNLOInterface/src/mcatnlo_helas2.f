c
c This file collects all the files of the HELAS package as obtained from
c MadEvent, except those listed in packHELAS.com.
c
c The lines relevant to preprocessor instructions, to be found between
c #ifdef and #endif in the original codes, have been eliminated by hand.
c They are affected by trivial programming bugs (as one can see by commenting
c out #ifdef and #endif), and are therefore apparently not used, since the 
c compilation usually runs smoothly, which is equivalent to setting
c HELAS_CHECK=0
c
      subroutine boostx(p,q , pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momentum p is assumed to be given in the rest frame of q.  pboost is
c the momentum p boosted to the frame in which q is given.  q must be a
c timelike momentum.
c
c input:
c       real    p(0:3)         : four-momentum p in the q rest  frame
c       real    q(0:3)         : four-momentum q in the boosted frame
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

c
      qq = q(1)**2+q(2)**2+q(3)**2

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         m = sqrt(q(0)**2-qq)
         lf = ((q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)+pq)/m
         pboost(1) =  p(1)+q(1)*lf
         pboost(2) =  p(2)+q(2)*lf
         pboost(3) =  p(3)+q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end
      subroutine eaixxx(eb,ea,shlf,chlf,phi,nhe,nha , eai)
c
c This subroutine computes an off-shell electron wavefunction after
c emitting a photon from the electron beam, with a special care for the
c small angle region.  The momenta are measured in the laboratory frame,
c where the e- beam is along the positive z axis.
c
c input:
c       real    eb             : energy (GeV)    of beam  e-
c       real    ea             : energy (GeV)    of final photon
c       real    shlf           : sin(theta/2)    of final photon
c       real    chlf           : cos(theta/2)    of final photon
c       real    phi            : azimuthal angle of final photon
c       integer nhe  = -1 or 1 : helicity        of beam  e-
c       integer nha  = -1 or 1 : helicity        of final photon
c
c output:
c       complex eai(6)         : off-shell electron             |e',A,e>
c
      implicit none
      double complex eai(6),phs
      double precision eb,ea,shlf,chlf,phi,alpha,gal,rnhe,x,c,s,d
      double precision coeff,xnnp,xnnm,snp,csp
      integer nhe,nha,nn

      double precision rHalf, rOne, rTwo, rFour, rOte
      double precision rPi, rIalph
      parameter( rHalf = 0.5d0, rOne = 1.0d0, rTwo = 2.0d0 )
      parameter( rFour = 4.0d0, rOte = 128.9d0 )
      parameter( rPi = 3.14159265358979323846d0 )
      parameter( rIalph = 137.0359895d0 )

      double precision me
      parameter( me = 0.510998902d-3 )

c
      alpha = rOne/rOte
      gal = sqrt(alpha*rFour*rPi)

      nn = nha*nhe
      rnhe = nhe
      x = ea/eb
      c = (chlf+shlf)*(chlf-shlf)
      s = rTwo*chlf*shlf
      d = -rOne/(ea*eb*(rFour*shlf**2+(me/eb)**2*c))
      coeff = -nn*gal*sqrt(eb)*d
      xnnp = x*(1+nn)
      xnnm = x*(1-nn)
      snp = sin(phi)
      csp = cos(phi)
      phs = dcmplx( csp, rnhe*snp )

      eai((5-3*nhe)/2) = -rnhe*coeff*me*s*(rOne+xnnp*rHalf)
      eai((5-nhe)/2)   =  xnnp*coeff*me*chlf**2*phs
      eai((5+nhe)/2)   =  rnhe*coeff*eb*s*(-rTwo+xnnm)
      eai((5+3*nhe)/2) =  xnnm*coeff*eb*shlf**2*phs*rTwo

      eai(5) =  eb*dcmplx( rOne-x, rOne-x*c )
      eai(6) = -eb*x*s*dcmplx( csp, snp )
c
      return
      end
      subroutine eaoxxx(eb,ea,shlf,chlf,phi,nhe,nha , eao)
c
c This subroutine computes an off-shell positron wavefunction after
c emitting a photon from the positron beam, with a special care for the
c small angle region.  The momenta are measured in the laboratory frame,
c where the e+ beam is along the negative z axis.
c
c input:
c       real    eb             : energy (GeV)    of beam  e+
c       real    ea             : energy (GeV)    of final photon
c       real    shlf           : sin(theta/2)    of final photon
c       real    chlf           : cos(theta/2)    of final photon
c       real    phi            : azimuthal angle of final photon
c       integer nhe  = -1 or 1 : helicity        of beam  e+
c       integer nha  = -1 or 1 : helicity        of final photon
c
c output:
c       complex eao(6)         : off-shell positron             <e,A,e'|
c
      implicit none
      double complex eao(6),phs
      double precision eb,ea,shlf,chlf,phi,alpha,gal,rnhe,x,c,s,d
      double precision coeff,xnnp,xnnm,snp,csp
      integer nhe,nha,nn

      double precision rHalf, rOne, rTwo, rFour, rOte
      double precision rPi, rIalph
      parameter( rHalf = 0.5d0, rOne = 1.0d0, rTwo = 2.0d0 )
      parameter( rFour = 4.0d0, rOte = 128.9d0 )
      parameter( rPi = 3.14159265358979323846d0 )
      parameter( rIalph = 137.0359895d0 )

      double precision me
      parameter( me = 0.510998902d-3 )

c

      alpha = rOne/rOte
      gal = sqrt(alpha*rFour*rPi)

      nn = nha*nhe
      rnhe = nhe
      x = ea/eb
      c = (chlf+shlf)*(chlf-shlf)
      s = rTwo*chlf*shlf
      d = -rOne/(ea*eb*(rFour*chlf**2-(me/eb)**2*c))
      coeff = nn*gal*sqrt(eb)*d
      xnnp = x*(1+nn)
      xnnm = x*(1-nn)
      snp = sin(phi)
      csp = cos(phi)
      phs = dcmplx( csp, -rnhe*snp )

      eao((5-3*nhe)/2) =               coeff*me*s*(rOne+xnnp*rHalf)
      eao((5-nhe)/2)   = rnhe*xnnp    *coeff*me*shlf**2*phs
      eao((5+nhe)/2)   =               coeff*eb*s*(-rTwo+xnnm)
      eao((5+3*nhe)/2) = real(nha-nhe)*coeff*eb*x*chlf**2*phs*rTwo

      eao(5) = eb*dcmplx( x-rOne, x*c+rOne )
      eao(6) = eb*x*s*dcmplx( csp, snp )
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine ficxxx(fi , fic)
c
c this subroutine charge conjugates a flowing-in fermion wavefunction.  
c                                                                       
c input:                                                                
c       complex fi(6)          : flowing-in fermion                 |fi>
c                                                                       
c output:                                                               
c       complex fic(6)         : charge conjugated fermion         <fic|
c
      implicit none
      double complex fi(6), fic(6)
c
      fic(1) = -fi(2)
      fic(2) =  fi(1)
      fic(3) =  fi(4)
      fic(4) = -fi(3)
      fic(5) = -fi(5)
      fic(6) = -fi(6)
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine focxxx(fo , foc)
c
c this subroutine charge conjugates a flowing-out fermion wavefunction. 
c                                                                       
c input:                                                                
c       complex fo(6)          : flowing-out fermion                <fo|
c                                                                       
c output:                                                               
c       complex foc(6)         : charge conjugated fermion         |foc>
c
      implicit none
      double complex fo(6), foc(6)
c
      foc(1) =  fo(2)
      foc(2) = -fo(1)
      foc(3) = -fo(4)
      foc(4) =  fo(3)
      foc(5) = -fo(5)
      foc(6) = -fo(6)
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fsicxx(fic,sc,gc,fmass,fwidth , fsic)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-in external antifermion and a vector boson.                   
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion           |fic>
c       complex sc(3)          : input    scalar                   s 
c       complex gc(2)          : coupling constants              gchf
c       real    fmass          : mass  of output antifermion fc'     
c       real    fwidth         : width of output antifermion fc'     
c                                                                       
c output:                                                               
c       complex fsic(6)        : off-shell fermion        |fc',s,fic>
c
      implicit none
      double complex fic(6),sc(3),fsic(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
c
      fsic(5) = fic(5)-sc(2)
      fsic(6) = fic(6)-sc(3)

      pf(0) = dble( fsic(5))
      pf(1) = dble( fsic(6))
      pf(2) = dimag(fsic(6))
      pf(3) = dimag(fsic(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(1)*(p0p3*fic(1)+dconjg(fsic(6))*fic(2))
      sl2 = gc(1)*(p0m3*fic(2)       +fsic(6) *fic(1))
      sr1 = gc(2)*(p0m3*fic(3)-dconjg(fsic(6))*fic(4))
      sr2 = gc(2)*(p0p3*fic(4)       -fsic(6) *fic(3))

      fsic(1) = ( gc(1)*fmass*fic(1) + sr1 )*ds
      fsic(2) = ( gc(1)*fmass*fic(2) + sr2 )*ds
      fsic(3) = ( gc(2)*fmass*fic(3) + sl1 )*ds
      fsic(4) = ( gc(2)*fmass*fic(4) + sl2 )*ds
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fsigld(fi,sc,gc,fmass,fwidth,smass,mNLSP,idecay , fsi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-in external fermion and a scalar boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fsi(6)         : off-shell fermion             |f',s,fi>
c
      implicit none
      double complex  fi(6), sc(3), gc(2), fsi(6), s1, s2, s3, s4, ds
      double complex  p14p, p14m, p23p, p23m
      double precision  pf(0:3), fmass, fwidth, mNLSP, smass, pf2
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fsi(5) = fi(5) - sc(2)
      fsi(6) = fi(6) - sc(3)

      pf(0) = dble( fsi(5))
      pf(1) = dble( fsi(6))
      pf(2) = dimag(fsi(6))
      pf(3) = dimag(fsi(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of FSIGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )

      s1 = -idecay*gc(1)*fi(1)*smass**2
      s2 = -idecay*gc(1)*fi(2)*smass**2
      s3 = gc(1)*mNLSP*( fi(1)*p14p + fi(2)*p23m ) 
      s4 = gc(1)*mNLSP*( fi(1)*p23p + fi(2)*p14m )

      if ( gc(2).ne.cZero ) then
         s1 = s1 + gc(2)*mNLSP*( fi(3)*p14m - fi(4)*p23m )
         s2 = s2 + gc(2)*mNLSP*(-fi(3)*p23p + fi(4)*p14p )
         s3 = s3 - gc(2)*idecay*fi(3)*smass**2
         s4 = s4 - gc(2)*idecay*fi(4)*smass**2
      end if

      fsi(1) = ( (pf(0)-pf(3))*s3 - dconjg(fsi(6))*s4 + fmass*s1 )*ds
      fsi(2) = (       -fsi(6)*s3 +  (pf(0)+pf(3))*s4 + fmass*s2 )*ds
      fsi(3) = ( (pf(0)+pf(3))*s1 + dconjg(fsi(6))*s2 + fmass*s3 )*ds
      fsi(4) = (        fsi(6)*s1 +  (pf(0)-pf(3))*s2 + fmass*s4 )*ds
c
      return          
      end
      subroutine fsixxx(fi,sc,gc,fmass,fwidth , fsi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-IN external fermion and a vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                 gchf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fsi(6)         : off-shell fermion             |f',s,fi>
c
      implicit none
      double complex fi(6),sc(3),fsi(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3

c

      fsi(5) = fi(5)-sc(2)
      fsi(6) = fi(6)-sc(3)

      pf(0) = dble( fsi(5))
      pf(1) = dble( fsi(6))
      pf(2) = dimag(fsi(6))
      pf(3) = dimag(fsi(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(1)*(p0p3*fi(1)+dconjg(fsi(6))*fi(2))
      sl2 = gc(1)*(p0m3*fi(2)       +fsi(6) *fi(1))
      sr1 = gc(2)*(p0m3*fi(3)-dconjg(fsi(6))*fi(4))
      sr2 = gc(2)*(p0p3*fi(4)       -fsi(6) *fi(3))

      fsi(1) = ( gc(1)*fmass*fi(1) + sr1 )*ds
      fsi(2) = ( gc(1)*fmass*fi(2) + sr2 )*ds
      fsi(3) = ( gc(2)*fmass*fi(3) + sl1 )*ds
      fsi(4) = ( gc(2)*fmass*fi(4) + sl2 )*ds
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fsocxx(foc,sc,gc,fmass,fwidth , fsoc)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-out external antifermion and a vector boson.                  
c                                                                       
c input:                                                                
c       complex foc(6)         : flow-out fermion               <foc|
c       complex sc(6)          : input    scalar                   s 
c       complex gc(2)          : coupling constants              gchf
c       real     fmass         : mass  of output antifermion fc'     
c       real     fwidth        : width of output antifermion fc'     
c                                                                       
c output:                                                               
c       complex fsoc(6)        : off-shell fermion         <fo,s,fc'|
c
      implicit none
      double complex foc(6),sc(6),fsoc(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
c
      fsoc(5) = foc(5)+sc(2)
      fsoc(6) = foc(6)+sc(3)

      pf(0) = dble( fsoc(5))
      pf(1) = dble( fsoc(6))
      pf(2) = dimag(fsoc(6))
      pf(3) = dimag(fsoc(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(2)*(p0p3*foc(3)       +fsoc(6) *foc(4))
      sl2 = gc(2)*(p0m3*foc(4)+dconjg(fsoc(6))*foc(3))
      sr1 = gc(1)*(p0m3*foc(1)       -fsoc(6) *foc(2))
      sr2 = gc(1)*(p0p3*foc(2)-dconjg(fsoc(6))*foc(1))

      fsoc(1) = ( gc(1)*fmass*foc(1) + sl1 )*ds
      fsoc(2) = ( gc(1)*fmass*foc(2) + sl2 )*ds
      fsoc(3) = ( gc(2)*fmass*foc(3) + sr1 )*ds
      fsoc(4) = ( gc(2)*fmass*foc(4) + sr2 )*ds
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fsogld(fo,sc,gc,fmass,fwidth,smass,mNLSP,idecay , fso)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-out external fermion and a scalar boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fso(6)         : off-shell fermion             <fo,s,f'|
c
      implicit none
      double complex  fo(6), sc(3), gc(2), fso(6), s1, s2, s3, s4, ds
      double precision  pf(0:3), fmass, fwidth, mNLSP, smass, pf2
      double precision  p14p, p14m, p23p, p23m
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fso(5) = fo(5) + sc(2)
      fso(6) = fo(6) + sc(3)

      pf(0) = dble( fso(5))
      pf(1) = dble( fso(6))
      pf(2) = dimag(fso(6))
      pf(3) = dimag(fso(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of FSOGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )

      s1 = -idecay*gc(1)*fo(1)*smass**2
      s2 = -idecay*gc(1)*fo(2)*smass**2
      s3 = gc(1)*mNLSP*( fo(1)*p14m - fo(2)*p23p )
      s4 = gc(1)*mNLSP*(-fo(1)*p23m + fo(2)*p14p )

      if ( gc(2).ne.cZero ) then
         s1 = s1 + gc(2)*mNLSP*( fo(3)*p14p + fo(4)*p23p )
         s2 = s2 + gc(2)*mNLSP*( fo(3)*p23m + fo(4)*p14m )
         s3 = s3 - gc(2)*idecay*fo(3)*smass**2
         s4 = s4 - gc(2)*idecay*fo(4)*smass**2
      end if

      fso(1) = (  (pf(0)+pf(3))*s3 +         fso(6)*s4 + fmass*s1 )*ds
      fso(2) = ( dconjg(fso(6))*s3 +  (pf(0)-pf(3))*s4 + fmass*s2 )*ds
      fso(3) = (  (pf(0)-pf(3))*s1 -         fso(6)*s2 + fmass*s3 )*ds
      fso(4) = (-dconjg(fso(6))*s1 +  (pf(0)+pf(3))*s2 + fmass*s4 )*ds
c
      return          
      end
      subroutine fsoxxx(fo,sc,gc,fmass,fwidth , fso)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-OUT external fermion and a vector boson.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(6)          : input    scalar                      s
c       complex gc(2)          : coupling constants                 gchf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fso(6)         : off-shell fermion             <fo,s,f'|
c
      implicit none
      double complex fo(6),sc(6),fso(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3

c

      fso(5) = fo(5)+sc(2)
      fso(6) = fo(6)+sc(3)

      pf(0) = dble( fso(5))
      pf(1) = dble( fso(6))
      pf(2) = dimag(fso(6))
      pf(3) = dimag(fso(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(2)*(p0p3*fo(3)       +fso(6) *fo(4))
      sl2 = gc(2)*(p0m3*fo(4)+dconjg(fso(6))*fo(3))
      sr1 = gc(1)*(p0m3*fo(1)       -fso(6) *fo(2))
      sr2 = gc(1)*(p0p3*fo(2)-dconjg(fso(6))*fo(1))

      fso(1) = ( gc(1)*fmass*fo(1) + sl1 )*ds
      fso(2) = ( gc(1)*fmass*fo(2) + sl2 )*ds
      fso(3) = ( gc(2)*fmass*fo(3) + sr1 )*ds
      fso(4) = ( gc(2)*fmass*fo(4) + sr2 )*ds
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fvicxx(fic,vc,gc,fmass,fwidth , fvic)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-in external antifermion and a vector boson.                   
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output antifermion f'         
c       real    fwidth         : width of output antifermion f'         
c                                                                       
c output:                                                               
c       complex fvic(6)        : off-shell antifermion       |fc',v,fic>
c
      implicit none
      double complex fic(6),vc(6),gc(2),fvic(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvic(5) = fic(5)-vc(5)
      fvic(6) = fic(6)-vc(6)

      pf(0) = dble( fvic(5))
      pf(1) = dble( fvic(6))
      pf(2) = dimag(fvic(6))
      pf(3) = dimag(fvic(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      d = rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*fic(1)
     &      + (vc(2)-cImag*vc(3))*fic(2)
      sl2 =   (vc(2)+cImag*vc(3))*fic(1)
     &      + (vc(1)-      vc(4))*fic(2)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*fic(3)
     &         - (vc(2)-cImag*vc(3))*fic(4)
         sr2 = - (vc(2)+cImag*vc(3))*fic(3)
     &         + (vc(1)+      vc(4))*fic(4)

         fvic(1) = ( gc(2)*((pf(0)-pf(3))*sl1 - dconjg(fvic(6))*sl2)
     &              +gc(1)*fmass*sr1 )*d
         fvic(2) = ( gc(2)*(     -fvic(6)*sl1 +   (pf(0)+pf(3))*sl2)
     &              +gc(1)*fmass*sr2 )*d
         fvic(3) = ( gc(1)*((pf(0)+pf(3))*sr1 + dconjg(fvic(6))*sr2)
     &              +gc(2)*fmass*sl1 )*d
         fvic(4) = ( gc(1)*(      fvic(6)*sr1 +   (pf(0)-pf(3))*sr2)
     &              +gc(2)*fmass*sl2 )*d
      else
         d = d * gc(1)
         fvic(1) = fmass*sr1*d
         fvic(2) = fmass*sr2*d
         fvic(3) = ((pf(0)+pf(3))*sr1 + dconjg(fvic(6))*sr2)*d
         fvic(4) = (      fvic(6)*sr1 +   (pf(0)-pf(3))*sr2)*d
      end if
c
      return
      end
      subroutine fvidmx(fi,vc,gc,fmass,fwidth , fvi)
c
c This subroutine computes a dipole moment off-shell fermion
c wavefunction from a flowing-IN external fermion and a vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2,2)        : coupling constants                  gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex fi(6), vc(6), fvi(6), sl1, sl2, sr1, sr2, d
      double complex gc(2,2), gL, gR
      double precision pf(0:3), fmass, fwidth, pf2

      double complex kvc21, kvc31, kvc41, kvc32, kvc42, kvc43
      double precision k(1:4)
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      gL = -gc(1,1) + cImag*gc(1,2)
      gR =  gc(2,1) + cImag*gc(2,2)

c k in vertex formula defined as (pi - po)
      k(1) = dble( vc(5))
      k(2) = dble( vc(6))
      k(3) = dimag(vc(6))
      k(4) = dimag(vc(5))

      kvc21 = (k(2)*vc(1) - k(1)*vc(2))*cImag
      kvc31 =  k(3)*vc(1) - k(1)*vc(3)
      kvc41 = (k(4)*vc(1) - k(1)*vc(4))*cImag
      kvc32 =  k(3)*vc(2) - k(2)*vc(3)
      kvc42 = (k(4)*vc(2) - k(2)*vc(4))*cImag
      kvc43 =  k(4)*vc(3) - k(3)*vc(4)

      fvi(5) = fi(5) - vc(5)
      fvi(6) = fi(6) - vc(6)

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )

      sl1 = gL*(  fi(1)*(kvc41 + kvc32)
     &          + fi(2)*(kvc42 + kvc21 + kvc43 + kvc31) )
      sl2 = gL*(- fi(1)*(kvc42 - kvc21 - kvc43 + kvc31)
     &          - fi(2)*(kvc41 + kvc32)                 )

      if ( gc(2,1).ne.cZero .or.
     &     gc(2,2).ne.cZero      ) then
         sr1 = gR*(- fi(3)*(kvc41 - kvc32)
     &             + fi(4)*(kvc42 - kvc21 + kvc43 - kvc31) )
         sr2 = gR*(- fi(3)*(kvc42 + kvc21 - kvc43 - kvc31)
     &             + fi(4)*(kvc41 - kvc32)                 )

         fvi(1) = ( (pf(0)-pf(3))*sr1 - dconjg(fvi(6))*sr2
     &             + fmass*sl1                             )*d
         fvi(2) = (      - fvi(6)*sr1 +  (pf(0)+pf(3))*sr2
     &             + fmass*sl2                             )*d
         fvi(3) = ( (pf(0)+pf(3))*sl1 + dconjg(fvi(6))*sl2
     &             + fmass*sr1                             )*d
         fvi(4) = (        fvi(6)*sl1 +  (pf(0)-pf(3))*sl2
     &             + fmass*sr2                             )*d

      else
         fvi(1) = fmass*sl1*d
         fvi(2) = fmass*sl2*d
         fvi(3) = ( (pf(0)+pf(3))*sl1 + dconjg(fvi(6))*sl2 )*d
         fvi(4) = (        fvi(6)*sl1 +  (pf(0)-pf(3))*sl2 )*d
      end if
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fvigld(fi,vc,gc,fmass,fwidth,idecay , fvi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-in external fermion and a vector boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex  fi(6), vc(6), gc(2), fvi(6), sl1, sl2, sr1, sr2, d
      double complex  p14p, p14m, p23p, p23m, A14p, A14m, A23p, A23m
      double complex  AdotpG
      double precision  fmass, fwidth
      double precision  pf(0:3), pv(4), pf2, pdotpG
      integer idecay

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvi(5) = fi(5) - vc(5)
      fvi(6) = fi(6) - vc(6)

      pv(1) = dble( vc(5))
      pv(2) = dble( vc(6))
      pv(3) = dimag(vc(6))
      pv(4) = dimag(vc(5))

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.eq.1 ) then
         pdotpG = pv(1)*pf(0) - pv(2)*pf(1) - pv(3)*pf(2) - pv(4)*pf(3)
         AdotpG = vc(1)*pf(0) - vc(2)*pf(1) - vc(3)*pf(2) - vc(4)*pf(3)
      else if ( idecay.eq.-1 ) then
         pdotpG =  pv(1)*dble( fi(5)) - pv(2)*dble( fi(6))
     &           - pv(3)*dimag(fi(6)) - pv(4)*dimag(fi(5))
         AdotpG =  vc(1)*dble( fi(5)) - vc(2)*dble( fi(6))
     &           - vc(3)*dimag(fi(6)) - vc(4)*dimag(fi(5))
      else
         write(6,*) 'error in idecay of FVIGLD'
         stop
      end if

      p14p = dble(vc(5)) + dimag(vc(5))
      p14m = dble(vc(5)) - dimag(vc(5))
      p23p = vc(6)
      p23m = dconjg(vc(6))
      A14p = vc(1) + vc(4)
      A14m = vc(1) - vc(4)
      A23p = vc(2) + vc(3)*ci
      A23m = vc(2) - vc(3)*ci

      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      d = d*idecay

      sl1 =  (p14p*AdotpG - A14p*pdotpG)*fi(1)
     &      +(p23m*AdotpG - A23m*pdotpG)*fi(2)
      sl2 =  (p23p*AdotpG - A23p*pdotpG)*fi(1)
     &      +(p14m*AdotpG - A14m*pdotpG)*fi(2)

      if ( gc(2).ne.cZero ) then
         sr1 =  (p14m*AdotpG - A14m*pdotpG)*fi(3)
     &         -(p23m*AdotpG - A23m*pdotpG)*fi(4)
         sr2 = -(p23p*AdotpG - A23p*pdotpG)*fi(3)
     &         +(p14p*AdotpG - A14p*pdotpG)*fi(4)

         fvi(1) = ( gc(1)*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2 )
     &             +gc(2)*fmass*sr1 )*d
         fvi(2) = ( gc(1)*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2 )
     &             +gc(2)*fmass*sr2 )*d
         fvi(3) = ( gc(2)*((pf(0)+pf(3))*sr1 + dconjg(fvi(6))*sr2 )
     &             +gc(1)*fmass*sl1 )*d
         fvi(4) = ( gc(2)*(       fvi(6)*sr1 +  (pf(0)-pf(3))*sr2 )
     &             +gc(1)*fmass*sl2 )*d

      else
         d = d*gc(1)
         fvi(1) = d*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)
         fvi(2) = d*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)
         fvi(3) = d*fmass*sl1
         fvi(4) = d*fmass*sl2
      end if
c
      return          
      end
      subroutine fvixxx(fi,vc,gc,fmass,fwidth , fvi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-IN external fermion and a vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex fi(6),vc(6),gc(2),fvi(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2
      
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      fvi(5) = fi(5)-vc(5)
      fvi(6) = fi(6)-vc(6)

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*fi(1)
     &      + (vc(2)-cImag*vc(3))*fi(2)
      sl2 =   (vc(2)+cImag*vc(3))*fi(1)
     &      + (vc(1)-      vc(4))*fi(2)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*fi(3)
     &         - (vc(2)-cImag*vc(3))*fi(4)
         sr2 = - (vc(2)+cImag*vc(3))*fi(3)
     &         + (vc(1)+      vc(4))*fi(4)

         fvi(1) = ( gc(1)*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)
     &             +gc(2)*fmass*sr1 )*d
         fvi(2) = ( gc(1)*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)
     &             +gc(2)*fmass*sr2 )*d
         fvi(3) = ( gc(2)*((pf(0)+pf(3))*sr1 + dconjg(fvi(6))*sr2)
     &             +gc(1)*fmass*sl1 )*d
         fvi(4) = ( gc(2)*(       fvi(6)*sr1 +  (pf(0)-pf(3))*sr2)
     &             +gc(1)*fmass*sl2 )*d

      else
         d = d * gc(1)
         fvi(1) = ((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)*d
         fvi(2) = (      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)*d
         fvi(3) = fmass*sl1*d
         fvi(4) = fmass*sl2*d
      end if
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fvocxx(foc,vc,gc,fmass,fwidth , fvoc)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-out external antifermion and a vector boson.                  
c                                                                       
c input:                                                                
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output antifermion f'         
c       real    fwidth         : width of output antifermion f'         
c                                                                       
c output:                                                               
c       complex fvoc(6)        : off-shell antifermion       <foc,v,fc'|
c
      implicit none
      double complex foc(6),vc(6),gc(2),fvoc(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvoc(5) = foc(5)+vc(5)
      fvoc(6) = foc(6)+vc(6)

      pf(0) = dble( fvoc(5))
      pf(1) = dble( fvoc(6))
      pf(2) = dimag(fvoc(6))
      pf(3) = dimag(fvoc(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      d = rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*foc(3)
     &      + (vc(2)+cImag*vc(3))*foc(4)
      sl2 =   (vc(2)-cImag*vc(3))*foc(3)
     &      + (vc(1)-      vc(4))*foc(4)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*foc(1)
     &         - (vc(2)+cImag*vc(3))*foc(2)
         sr2 = - (vc(2)-cImag*vc(3))*foc(1)
     &         + (vc(1)+      vc(4))*foc(2)

         fvoc(1) = ( gc(1)*( (pf(0)+pf(3))*sr1          +fvoc(6)*sr2)
     &              +gc(2)*fmass*sl1 )*d
         fvoc(2) = ( gc(1)*( dconjg(fvoc(6))*sr1 + (pf(0)-pf(3))*sr2)
     &              +gc(2)*fmass*sl2 )*d
         fvoc(3) = ( gc(2)*( (pf(0)-pf(3))*sl1          -fvoc(6)*sl2)
     &              +gc(1)*fmass*sr1 )*d
         fvoc(4) = ( gc(2)*(-dconjg(fvoc(6))*sl1 + (pf(0)+pf(3))*sl2)
     &              +gc(1)*fmass*sr2 )*d
      else
         d = d * gc(1)
         fvoc(1) = ((pf(0)+pf(3))*sr1          +fvoc(6)*sr2)*d
         fvoc(2) = (dconjg(fvoc(6))*sr1 + (pf(0)-pf(3))*sr2)*d
         fvoc(3) = fmass*sr1*d
         fvoc(4) = fmass*sr2*d
      end if
c
      return
      end
      subroutine fvodmx(fo,vc,gc,fmass,fwidth , fvo)
c
c This subroutine computes a dipole moment off-shell fermion
c wavefunction from a flowing-OUT external fermion and a vector boson.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2,2)        : coupling constants                  gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,v,f'|
c
      implicit none
      double complex fo(6), vc(6), fvo(6), sl1, sl2, sr1, sr2, d
      double complex gc(2,2), gL, gR
      double precision  pf(0:3), fmass, fwidth, pf2

      double complex kvc21, kvc31, kvc41, kvc32, kvc42, kvc43
      double precision k(1:4)
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      gL = -gc(1,1) + cImag*gc(1,2)
      gR =  gc(2,1) + cImag*gc(2,2)

c k in vertex formula defined as (pi - po)
      k(1) = dble( vc(5))
      k(2) = dble( vc(6))
      k(3) = dimag(vc(6))
      k(4) = dimag(vc(5))

      kvc21 = (k(2)*vc(1) - k(1)*vc(2))*cImag
      kvc31 =  k(3)*vc(1) - k(1)*vc(3)
      kvc41 = (k(4)*vc(1) - k(1)*vc(4))*cImag
      kvc32 =  k(3)*vc(2) - k(2)*vc(3)
      kvc42 = (k(4)*vc(2) - k(2)*vc(4))*cImag
      kvc43 =  k(4)*vc(3) - k(3)*vc(4)

      fvo(5) = fo(5) + vc(5)
      fvo(6) = fo(6) + vc(6)

      pf(0) = dble( fvo(5))
      pf(1) = dble( fvo(6))
      pf(2) = dimag(fvo(6))
      pf(3) = dimag(fvo(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )

      sl1 = gL*(  fo(1)*(kvc41 + kvc32)
     &          - fo(2)*(kvc42 - kvc21 - kvc43 + kvc31) )
      sl2 = gL*(  fo(1)*(kvc42 + kvc21 + kvc43 + kvc31)
     &          - fo(2)*(kvc41 + kvc32)                 )

      if ( gc(2,1).ne.cZero .or.
     &     gc(2,2).ne.cZero      ) then
         sr1 = gR*(- fo(3)*(kvc41 - kvc32)
     &             - fo(4)*(kvc42 + kvc21 - kvc43 - kvc31) )
         sr2 = gR*(  fo(3)*(kvc42 - kvc21 + kvc43 - kvc31)
     &             + fo(4)*(kvc41 - kvc32)                 )

         fvo(1) = (  (pf(0)+pf(3))*sr1       + fvo(6)*sr2
     &             + fmass*sl1                            )*d
         fvo(2) = ( dconjg(fvo(6))*sr1 +(pf(0)-pf(3))*sr2
     &             + fmass*sl2                            )*d
         fvo(3) = (  (pf(0)-pf(3))*sl1       - fvo(6)*sl2
     &             + fmass*sr1                            )*d
         fvo(4) = (-dconjg(fvo(6))*sl1 +(pf(0)+pf(3))*sl2
     &             + fmass*sr2                            )*d

      else
         fvo(1) = fmass*sl1*d
         fvo(2) = fmass*sl2*d
         fvo(3) = (  (pf(0)-pf(3))*sl1        - fvo(6)*sl2)*d
         fvo(4) = (-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)*d
      end if
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine fvogld(fo,vc,gc,fmass,fwidth,idecay , fvo)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-out external fermion and a vector boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,v,f'|
c
      implicit none
      double complex  fo(6), vc(6), gc(2), fvo(6), sl1, sl2, sr1, sr2, d
      double complex  p14p, p14m, p23p, p23m, A14p, A14m, A23p, A23m
      double complex  AdotpG
      double precision  fmass, fwidth
      double precision  pf(0:3), pv(4), pf2, pdotpG
      integer idecay

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvo(5) = fo(5) + vc(5)
      fvo(6) = fo(6) + vc(6)

      pv(1) = dble( vc(5))
      pv(2) = dble( vc(6))
      pv(3) = dimag(vc(6))
      pv(4) = dimag(vc(5))

      pf(0) = dble( fvo(5))
      pf(1) = dble( fvo(6))
      pf(2) = dimag(fvo(6))
      pf(3) = dimag(fvo(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.eq.1 ) then
         pdotpG = pv(1)*pf(0) - pv(2)*pf(1) - pv(3)*pf(2) - pv(4)*pf(3)
         AdotpG = vc(1)*pf(0) - vc(2)*pf(1) - vc(3)*pf(2) - vc(4)*pf(3)
      else if ( idecay.eq.-1 ) then
         pdotpG =  pv(1)*dble( fo(5)) - pv(2)*dble( fo(6))
     &           - pv(3)*dimag(fo(6)) - pv(4)*dimag(fo(5))
         AdotpG =  vc(1)*dble( fo(5)) - vc(2)*dble( fo(6))
     &           - vc(3)*dimag(fo(6)) - vc(4)*dimag(fo(5))
      else
         write(6,*) 'error in idecay of FVOGLD'
         stop
      end if

      p14p = dble(vc(5)) + dimag(vc(5))
      p14m = dble(vc(5)) - dimag(vc(5))
      p23p = vc(6)
      p23m = dconjg(vc(6))
      A14p = vc(1) + vc(4)
      A14m = vc(1) - vc(4)
      A23p = vc(2) + vc(3)*ci
      A23m = vc(2) - vc(3)*ci

      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      d = d*idecay

      sl1 =  (p14p*AdotpG - A14p*pdotpG)*fo(3)
     &      +(p23p*AdotpG - A23p*pdotpG)*fo(4)
      sl2 =  (p23m*AdotpG - A23m*pdotpG)*fo(3)
     &      +(p14m*AdotpG - A14m*pdotpG)*fo(4)

      if ( gc(2).ne.cZero ) then
         sr1 =  (p14m*AdotpG - A14m*pdotpG)*fo(1)
     &         -(p23p*AdotpG - A23p*pdotpG)*fo(2)
         sr2 = -(p23m*AdotpG - A23m*pdotpG)*fo(1)
     &         +(p14p*AdotpG - A14p*pdotpG)*fo(2)

         fvo(1) = ( gc(2)*(  (pf(0)+pf(3))*sr1         +fvo(6)*sr2 )
     &             +gc(1)*fmass*sl1 )*d
         fvo(2) = ( gc(2)*( dconjg(fvo(6))*sr1 + (pf(0)-pf(3))*sr2 )
     &             +gc(1)*fmass*sl2 )*d
         fvo(3) = ( gc(1)*(  (pf(0)-pf(3))*sl1         -fvo(6)*sl2 )
     &             +gc(2)*fmass*sr1 )*d
         fvo(4) = ( gc(1)*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2 )
     &             +gc(2)*fmass*sr2 )*d

      else
         d = d*gc(1)
         fvo(1) = d*fmass*sl1
         fvo(2) = d*fmass*sl2
         fvo(3) = d*(  (pf(0)-pf(3))*sl1         -fvo(6)*sl2)
         fvo(4) = d*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)
      end if
c
      return          
      end
      subroutine fvoxxx(fo,vc,gc,fmass,fwidth , fvo)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-OUT external fermion and a vector boson.
c
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,v,f'|
c
      implicit none
      double complex fo(6),vc(6),gc(2),fvo(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      fvo(5) = fo(5)+vc(5)
      fvo(6) = fo(6)+vc(6)

      pf(0) = dble( fvo(5))
      pf(1) = dble( fvo(6))
      pf(2) = dimag(fvo(6))
      pf(3) = dimag(fvo(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)


      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*fo(3)
     &      + (vc(2)+cImag*vc(3))*fo(4)
      sl2 =   (vc(2)-cImag*vc(3))*fo(3)
     &      + (vc(1)-      vc(4))*fo(4)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*fo(1)
     &         - (vc(2)+cImag*vc(3))*fo(2)
         sr2 = - (vc(2)-cImag*vc(3))*fo(1)
     &         + (vc(1)+      vc(4))*fo(2)

         fvo(1) = ( gc(2)*( (pf(0)+pf(3))*sr1  +        fvo(6)*sr2)
     &             +gc(1)*fmass*sl1 )*d
         fvo(2) = ( gc(2)*( dconjg(fvo(6))*sr1 + (pf(0)-pf(3))*sr2)
     &             +gc(1)*fmass*sl2 )*d
         fvo(3) = ( gc(1)*( (pf(0)-pf(3))*sl1  -        fvo(6)*sl2)
     &             +gc(2)*fmass*sr1 )*d
         fvo(4) = ( gc(1)*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)
     &             +gc(2)*fmass*sr2 )*d

      else
         d = d * gc(1)
         fvo(1) = fmass*sl1*d
         fvo(2) = fmass*sl2*d
         fvo(3) = (  (pf(0)-pf(3))*sl1 -        fvo(6)*sl2)*d
         fvo(4) = (-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)*d
      end if
c
      return
      end
      subroutine ggggxx(ga,gb,gc,gd,g, vertex)
c
c This subroutine computes the portion of the amplitude of the four-point 
c coupling of 4 massless color octet gauge bosons (gluons) corresponding 
c to the color structure f^{a,b,e} f{c,d,e}. 
c To optain the complete amplitude, this coupling must be called three
c times (once for each color structure) with the following permutations:
c	call ggggxx(ga,gb,gc,gd,g,v1)
c       call ggggxx(ga,gc,gd,gb,g,v2)
c       call ggggxx(ga,gd,gb,gc,g,v3)
c
c	f^{a,b,e} f{c,d,e}*v1+
c	f^{a,c,e} f{d,b,e}*v2+
c	f^{a,d,e} f{b,c,e}*v3
c (See 2.9.1 of the manual for more information).
c                                                                       
c input:                                                                
c       complex ga(0:3)        : Boson with adjoint color index a 
c       complex gb(0:3)        : Boson with adjoint color index b
c       complex gc(0:3)        : Boson with adjoint color index c 
c       complex gd(0:3)        : Boson with adjoint color index d
c       real    g              : coupling of w31 with w-/w+             
c
      implicit none
      double complex ga(6),gb(6),gc(6),gd(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),
     &     dvertx,v12,v13,v14,v23,v24,v34
      double precision pga(0:3),pgb(0:3),pgc(0:3),pgd(0:3),g

      save dv1,dv2,dv3, dv4
c      save dv1,dv2,dv3,dv4,dvertx,v12,v13,v14,v23,v24,v34

c

      dv1(0) = dcmplx(ga(1))
      dv1(1) = dcmplx(ga(2))
      dv1(2) = dcmplx(ga(3))
      dv1(3) = dcmplx(ga(4))
      dv2(0) = dcmplx(gb(1))
      dv2(1) = dcmplx(gb(2))
      dv2(2) = dcmplx(gb(3))
      dv2(3) = dcmplx(gb(4))
      dv3(0) = dcmplx(gc(1))
      dv3(1) = dcmplx(gc(2))
      dv3(2) = dcmplx(gc(3))
      dv3(3) = dcmplx(gc(4))
      dv4(0) = dcmplx(gd(1))
      dv4(1) = dcmplx(gd(2))
      dv4(2) = dcmplx(gd(3))
      dv4(3) = dcmplx(gd(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

      dvertx = v14*v23 -v13*v24

      vertex = dcmplx( dvertx ) * (g*g)

c      if (abs(dvertx) .gt. 1d40) then
c         write(*,*) 'Careful',abs(dvertx)
c         write(*,*) v12,v13,v14
c         write(*,*) v23,v24,v34
c      endif
c
      return
      end
      subroutine gggxxx(wm,wp,w3,g , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c the gauge bosons.
c
c input:
c       complex wm(6)          : vector               flow-out W-
c       complex wp(6)          : vector               flow-out W+
c       complex w3(6)          : vector               j3 or A    or Z
c       real    g              : coupling constant    gw or gwwa or gwwz
c
c output:
c       complex vertex         : amplitude               gamma(wm,wp,w3)
c
      implicit none
      double complex wm(6),wp(6),w3(6),vertex
      double complex xv1,xv2,xv3,v12,v23,v31
      double complex p12,p13,p21,p23,p31,p32
      double precision pwm(0:3),pwp(0:3),pw3(0:3),g

      double precision rZero, rTenth
      parameter( rZero = 0.0d0, rTenth = 0.1d0 )

c
      pwm(0) = dble( wm(5))
      pwm(1) = dble( wm(6))
      pwm(2) = dimag(wm(6))
      pwm(3) = dimag(wm(5))
      pwp(0) = dble( wp(5))
      pwp(1) = dble( wp(6))
      pwp(2) = dimag(wp(6))
      pwp(3) = dimag(wp(5))
      pw3(0) = dble( w3(5))
      pw3(1) = dble( w3(6))
      pw3(2) = dimag(w3(6))
      pw3(3) = dimag(w3(5))


      v12 = wm(1)*wp(1)-wm(2)*wp(2)-wm(3)*wp(3)-wm(4)*wp(4)
      v23 = wp(1)*w3(1)-wp(2)*w3(2)-wp(3)*w3(3)-wp(4)*w3(4)
      v31 = w3(1)*wm(1)-w3(2)*wm(2)-w3(3)*wm(3)-w3(4)*wm(4)
      xv1 = rZero
      xv2 = rZero
      xv3 = rZero
      if ( abs(wm(1)).ne.rZero ) then
         if ( abs(wm(1)).ge.max(abs(wm(2)),abs(wm(3)),abs(wm(4)))
     &        *rTenth )
     &      xv1 = pwm(0)/wm(1)
      endif
      if ( abs(wp(1)).ne.rZero ) then
         if ( abs(wp(1)).ge.max(abs(wp(2)),abs(wp(3)),abs(wp(4)))
     &        *rTenth )
     &      xv2 = pwp(0)/wp(1)
      endif
      if ( abs(w3(1)).ne.rZero ) then
         if ( abs(w3(1)).ge.max(abs(w3(2)),abs(w3(3)),abs(w3(4)))
     &        *rTenth )
     &      xv3 = pw3(0)/w3(1)
      endif

      p12 = (pwm(0)-xv1*wm(1))*wp(1)-(pwm(1)-xv1*wm(2))*wp(2)
     &     -(pwm(2)-xv1*wm(3))*wp(3)-(pwm(3)-xv1*wm(4))*wp(4)
      p13 = (pwm(0)-xv1*wm(1))*w3(1)-(pwm(1)-xv1*wm(2))*w3(2)
     &     -(pwm(2)-xv1*wm(3))*w3(3)-(pwm(3)-xv1*wm(4))*w3(4)
      p21 = (pwp(0)-xv2*wp(1))*wm(1)-(pwp(1)-xv2*wp(2))*wm(2)
     &     -(pwp(2)-xv2*wp(3))*wm(3)-(pwp(3)-xv2*wp(4))*wm(4)
      p23 = (pwp(0)-xv2*wp(1))*w3(1)-(pwp(1)-xv2*wp(2))*w3(2)
     &     -(pwp(2)-xv2*wp(3))*w3(3)-(pwp(3)-xv2*wp(4))*w3(4)
      p31 = (pw3(0)-xv3*w3(1))*wm(1)-(pw3(1)-xv3*w3(2))*wm(2)
     &     -(pw3(2)-xv3*w3(3))*wm(3)-(pw3(3)-xv3*w3(4))*wm(4)
      p32 = (pw3(0)-xv3*w3(1))*wp(1)-(pw3(1)-xv3*w3(2))*wp(2)
     &     -(pw3(2)-xv3*w3(3))*wp(3)-(pw3(3)-xv3*w3(4))*wp(4)

      vertex = -(v12*(p13-p23)+v23*(p21-p31)+v31*(p32-p12))*g
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine hiocxx(fic,foc,gc,smass,swidth , hioc)                
c                                                                       
c this subroutine computes an off-shell scalar current from an external 
c antifermion pair.                                                     
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex gc(2)          : coupling constants                 gchf
c       real    smass          : mass  of output scalar s               
c       real    swidth         : width of output scalar s               
c                                                                       
c output:                                                               
c       complex hioc(3)        : scalar current           j(<fic|s|foc>)
c                                                                       
      implicit none
      double complex fic(6),foc(6),hioc(3),gc(2),dn
      double precision q(0:3),smass,swidth,q2
c
      hioc(2) = foc(5)-fic(5)
      hioc(3) = foc(6)-fic(6)

      q(0) = dble( hioc(2))
      q(1) = dble( hioc(3))
      q(2) = dimag(hioc(3))
      q(3) = dimag(hioc(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

      dn = -dcmplx( q2-smass**2, smass*swidth )

      hioc(1) = (  gc(1)*(foc(1)*fic(1)+foc(2)*fic(2))
     &           + gc(2)*(foc(3)*fic(3)+foc(4)*fic(4)) )/dn
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine hiogld(fi,fo,gc,smass,swidth,mNLSP,idecay , hio)
c
c This subroutine computes an off-shell scalar current for the NLSP-
c Goldstino vertex from the external fermion pair. The h.c. of the NLSP
c decay is handled via the input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gsf
c       real    smass          : mass  of output scalar s
c       real    swidth         : width of output scalar s
c       real    mNLSP          : mass of NLSP
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex hio(3)         : scalar current          j^mu(<fo|s|fi>)
c
      implicit none
      double complex fi(6), fo(6), gc(2), hio(3)
      double complex dn, p14p, p14m, p23p, p23m
      double precision q(0:3), smass, swidth, mNLSP, q2
      double precision pG(1:4)
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      hio(2) = -fi(5) + fo(5)
      hio(3) = -fi(6) + fo(6)

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of HIOGLD'
         stop
      end if

      q(0) = dble( hio(2))
      q(1) = dble( hio(3))
      q(2) = dimag(hio(3))
      q(3) = dimag(hio(2))
      q2 = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2

      p14p = q(0) + q(3)
      p14m = q(0) - q(3)
      p23p = q(1) + q(2)*ci
      p23m = q(1) - q(2)*ci

      dn = -dcmplx( q2-smass**2, smass*swidth )

      hio(1) = gc(1)*( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &                  +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*mNLSP
     &                -( fo(1)*fi(1) + fo(2)*fi(2) )*idecay*smass**2 )

      if ( gc(2).ne.cZero ) then
         hio(1) = hio(1) + gc(2) *
     &            ( ( ( fo(1)*p14m - fo(2)*p23p )*fi(3)
     &               -( fo(1)*p23m - fo(2)*p14p )*fi(4) )*mNLSP
     &             -( fo(3)*fi(3) + fo(4)*fi(4) )*idecay*smass**2 )
      end if

      hio(1) = hio(1)/dn
c
      return
      end
      subroutine hioxxx(fi,fo,gc,smass,swidth , hio)
c
c This subroutine computes an off-shell scalar current from an external
c fermion pair.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                 gchf
c       real    smass          : mass  of OUTPUT scalar s
c       real    swidth         : width of OUTPUT scalar s
c
c output:
c       complex hio(3)         : scalar current             j(<fi|s|fo>)
c
      implicit none
      double complex fi(6),fo(6),hio(3),gc(2),dn
      double precision q(0:3),smass,swidth,q2

c

      hio(2) = fo(5)-fi(5)
      hio(3) = fo(6)-fi(6)

      q(0) = dble( hio(2))
      q(1) = dble( hio(3))
      q(2) = dimag(hio(3))
      q(3) = dimag(hio(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dn = -dcmplx( q2-smass**2, smass*swidth )

      hio(1) = ( gc(1)*(fo(1)*fi(1)+fo(2)*fi(2))
     &          +gc(2)*(fo(3)*fi(3)+fo(4)*fi(4)) )/dn
c
      return
      end
      subroutine hsssxx(s1,s2,s3,gc,smass,swidth , hsss)
c
c This subroutine computes an off-shell scalar current from the four-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex s3(3)          : third  scalar                        s3
c       complex gc             : coupling constant                 ghhhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hsss(3)        : scalar current           j(s':s1,s2,s3)
c     
      implicit none
      double complex s1(3),s2(3),s3(3),gc,hsss(3),dg
      double precision q(0:3),smass,swidth,q2

c

      hsss(2) = s1(2)+s2(2)+s3(2)
      hsss(3) = s1(3)+s2(3)+s3(3)

      q(0) = dble( hsss(2))
      q(1) = dble( hsss(3))
      q(2) = dimag(hsss(3))
      q(3) = dimag(hsss(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hsss(1) = dg * s1(1)*s2(1)*s3(1)
c
      return
      end
      subroutine hssxxx(s1,s2,gc,smass,swidth , hss)
c
c This subroutine computes an off-shell scalar current from the three-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                  ghhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hss(3)         : scalar current              j(s':s1,s2)
c     
      implicit none
      double complex s1(3),s2(3),gc,hss(3),dg
      double precision q(0:3),smass,swidth,q2

c

      hss(2) = s1(2)+s2(2)
      hss(3) = s1(3)+s2(3)

      q(0) = dble( hss(2))
      q(1) = dble( hss(3))
      q(2) = dimag(hss(3))
      q(3) = dimag(hss(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hss(1) = dg*s1(1)*s2(1)
c
      return
      end
      subroutine hvsxxx(vc,sc,gc,smass,swidth , hvs)
c
c This subroutine computes an off-shell scalar current from the vector-
c scalar-scalar coupling.  The coupling is absent in the minimal SM in
c unitary gauge.
c
c input:
c       complex vc(6)          : input vector                          v
c       complex sc(3)          : input scalar                          s
c       complex gc             : coupling constant (s charge)
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c examples of the coupling constant gc for susy particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=A   |   v=Z   |   v=W   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the sc charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex hvs(3)         : scalar current                j(s':v,s)
c     
      implicit none
      double complex vc(6),sc(3),hvs(3),dg,qvv,qpv,gc
      double precision qv(0:3),qp(0:3),qa(0:3),smass,swidth,q2

      double precision rTwo
      parameter( rTwo = 2.0d0 )

c

      hvs(2) = vc(5)+sc(2)
      hvs(3) = vc(6)+sc(3)

      qv(0) = dble(  vc(5))
      qv(1) = dble(  vc(6))
      qv(2) = dimag( vc(6))
      qv(3) = dimag( vc(5))
      qp(0) = dble(  sc(2))
      qp(1) = dble(  sc(3))
      qp(2) = dimag( sc(3))
      qp(3) = dimag( sc(2))
      qa(0) = dble( hvs(2))
      qa(1) = dble( hvs(3))
      qa(2) = dimag(hvs(3))
      qa(3) = dimag(hvs(2))
      q2 = qa(0)**2-(qa(1)**2+qa(2)**2+qa(3)**2)


      dg = -gc/dcmplx( q2-smass**2, smass*swidth )
      qvv = qv(0)*vc(1)-qv(1)*vc(2)-qv(2)*vc(3)-qv(3)*vc(4)
      qpv = qp(0)*vc(1)-qp(1)*vc(2)-qp(2)*vc(3)-qp(3)*vc(4)

      hvs(1) = dg*(rTwo*qpv+qvv)*sc(1)
c
      return
      end
      subroutine hvvsxx(v1,v2,sc,gc,smass,swidth , hvvs)
c
c This subroutine computes an off-shell scalar current of the vector-
c vector-scalar-scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex sc(3)          : input  scalar                        s
c       complex gc             : coupling constant                 gvvhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hvvs(3)        : scalar current            j(s':v1,v2,s)
c     
      implicit none
      double complex v1(6),v2(6),sc(3),gc,hvvs(3),dg
      double precision q(0:3),smass,swidth,q2

c

      hvvs(2) = v1(5)+v2(5)+sc(2)
      hvvs(3) = v1(6)+v2(6)+sc(3)

      q(0) = dble( hvvs(2))
      q(1) = dble( hvvs(3))
      q(2) = dimag(hvvs(3))
      q(3) = dimag(hvvs(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hvvs(1) = dg*sc(1)
     &         *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
      subroutine hvvxxx(v1,v2,gc,smass,swidth , hvv)
c
c This subroutine computes an off-shell scalar current from the vector-
c vector-scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex gc             : coupling constant                  gvvh
c       real    smass          : mass  of OUTPUT scalar s
c       real    swidth         : width of OUTPUT scalar s
c
c output:
c       complex hvv(3)         : off-shell scalar current     j(s:v1,v2)
c     
      implicit none
      double complex v1(6),v2(6),gc,hvv(3),dg
      double precision q(0:3),smass,swidth,q2

      double precision rZero
      parameter( rZero = 0.0d0 )

c

      hvv(2) = v1(5)+v2(5)
      hvv(3) = v1(6)+v2(6)

      q(0) = dble( hvv(2))
      q(1) = dble( hvv(3))
      q(2) = dimag(hvv(3))
      q(3) = dimag(hvv(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hvv(1) = dg*(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine ioscxx(fic,foc,sc,gc , vertex)
c
c This subroutine computes an amplitude of the antifermion-antifermion- 
c scalar coupling.                                                      
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex sc(3)          : input    scalar                      s 
c       complex gc(2)          : coupling constants                 gchf
c                                                                       
c output:                                                               
c       complex vertex         : amplitude                   <foc|s|fic>
c
      implicit none
      double complex fic(6),foc(6),sc(3),gc(2),vertex
c
      vertex = sc(1)*( gc(1)*(fic(1)*foc(1)+fic(2)*foc(2))
     &                +gc(2)*(fic(3)*foc(3)+fic(4)*foc(4)) )
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine iosgld(fi,fo,sc,gc,smass,mNLSP,idecay , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-scalar
c SUSY Goldstino coupling. In this routine, the NLSP is decaying to a
c boson and a Goldstino. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    mNLSP          : mass of the NLSP
c       real    smass          : mass of the scalar boson
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex vertex         : amplitude                     <fo|s|fi>
c
      implicit none
      double complex  fi(6), fo(6), gc(2), sc(3), vertex
      double complex  p14p, p14m, p23p, p23m
      double precision  mNLSP, smass
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of IOSGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      vertex = gc(1) *
     &         ( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &            +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*mNLSP
     &          -( fo(1)*fi(1) + fo(2)*fi(2) )*idecay*smass**2 )

      if ( gc(2).ne.cZero ) then
         vertex = vertex + gc(2) *
     &            ( ( ( fo(1)*p14m - fo(2)*p23p )*fi(3)
     &               -( fo(1)*p23m - fo(2)*p14p )*fi(4) )*mNLSP
     &             -( fo(3)*fi(3) + fo(4)*fi(4) )*idecay*smass**2 )
      end if

      vertex = vertex * sc(1)
c
      return
      end
      subroutine iosxxx(fi,fo,sc,gc , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-scalar
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                 gchf
c
c output:
c       complex vertex         : amplitude                     <fo|s|fi>
c     
      implicit none
      double complex fi(6),fo(6),sc(3),gc(2),vertex

c

      vertex = sc(1)*( gc(1)*(fi(1)*fo(1)+fi(2)*fo(2))
     &                +gc(2)*(fi(3)*fo(3)+fi(4)*fo(4)) )
c
      return
      end
      subroutine iotxkk(fi,fo,tc,g,fmass , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex tc(6,4)        : input    tensor                      t
c       real    g              : coupling constant                 -kappa/8
c       real    fmass          : fermion mass                        m_f
c
c output:
c       complex vertex         : amplitude                        <fo|t|fi>
c     
      implicit none
      double complex fi(6), fo(6), tc(6,4), vertex
      double precision g, fmass

      double complex k23, k23s, D1, D2, D3, D4, Tii
      double complex T11, T22, T33, T44, T12, T13, T14, T23, T24, T34
      double complex f13, f14, f23, f24, f31, f32, f41, f42
      double precision k(4), k14p, k14m, m2

      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
      m2 = rTwo*fmass

      k(1) = dreal(fi(5)+fo(5))
      k(2) = dreal(fi(6)+fo(6))
      k(3) = dimag(fi(6)+fo(6))
      k(4) = dimag(fi(5)+fo(5))
      k23  = dcmplx( k(2),k(3) )
      k23s = dconjg( k23 )
      k14p = k(1) + k(4)
      k14m = k(1) - k(4)

      f13 = fo(1)*fi(3)
      f14 = fo(1)*fi(4)
      f23 = fo(2)*fi(3)
      f24 = fo(2)*fi(4)
      f31 = fo(3)*fi(1)
      f32 = fo(3)*fi(2)
      f41 = fo(4)*fi(1)
      f42 = fo(4)*fi(2)

      T11 = rTwo*tc(1,1)
      T22 = rTwo*tc(2,2)
      T33 = rTwo*tc(3,3)
      T44 = rTwo*tc(4,4)
      T12 = tc(1,2) + tc(2,1)
      T13 = tc(1,3) + tc(3,1)
      T14 = tc(1,4) + tc(4,1)
      T23 = tc(2,3) + tc(3,2)
      T24 = tc(2,4) + tc(4,2)
      T34 = tc(3,4) + tc(4,3)

      D1 =   k(1)*(T11-T14) - k(2)*(T12-T24)
     &     - k(3)*(T13-T34) - k(4)*(T14-T44)

      D2 = - k(1)*(T12-ci*T13) + k(2)*(T22-ci*T23)
     &     + k(3)*(T23-ci*T33) + k(4)*(T24-ci*T34)

      D3 = - k(1)*(T12+ci*T13) + k(2)*(T22+ci*T23)
     &     + k(3)*(T23+ci*T33) + k(4)*(T24+ci*T34)

      D4 =   k(1)*(T11+T14) - k(2)*(T12+T24)
     &     - k(3)*(T13+T34) - k(4)*(T14+T44)

      Tii = T11 - T22 - T33 - T44

      vertex = D1*(f13+f42) + D2*(f14-f32) + D3*(f23-f41) + D4*(f24+f31)

      vertex = vertex + Tii*( - k14p*(f24+f31) - k14m*(f13+f42)
     &                        +  k23*(f23-f41) + k23s*(f14-f32) )

      if ( fmass.ne.rZero ) then
         vertex = vertex + m2*Tii*(  fo(1)*fi(1) + fo(2)*fi(2)
     &                             + fo(3)*fi(3) + fo(4)*fi(4) )
      end if

      vertex = vertex * g
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine iovcxx(fic,foc,vc,gc , vertex)
c
c this subroutine computes an amplitude of the antifermion-antifermion- 
c vector coupling.                                                      
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c                                                                       
c output:                                                               
c       complex vertex         : amplitude                   <foc|v|fic>
c
      implicit none
      double complex fic(6),foc(6),vc(6),gc(2),vertex

      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      vertex = - gc(1)*( (foc(1)*fic(3)+foc(2)*fic(4))*vc(1)
     &                  -(foc(1)*fic(4)+foc(2)*fic(3))*vc(2)
     &                  +(foc(1)*fic(4)-foc(2)*fic(3))*vc(3)*cImag
     &                  -(foc(1)*fic(3)-foc(2)*fic(4))*vc(4)       )

      if ( gc(2).ne.cZero ) then
         vertex = vertex
     &            - gc(2)*( (foc(3)*fic(1)+foc(4)*fic(2))*vc(1)
     &                     +(foc(3)*fic(2)+foc(4)*fic(1))*vc(2)
     &                     -(foc(3)*fic(2)-foc(4)*fic(1))*vc(3)*cImag
     &                     +(foc(3)*fic(1)-foc(4)*fic(2))*vc(4)       )
      end if
c
      return
      end
      subroutine iovdmx(fi,fo,vc,gc, vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c dipole moment (non-renormalizable) coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2,2)        : coupling constants                  gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c
c output:
c       complex vertex         : amplitude                     <fo|v|fi>
c
      implicit none
      double complex fi(6), fo(6), vc(6), vertex, gc(2,2)

      double complex q(5:6), dum1, dum2
      double complex f1122, f12, f21, f3344, f34, f43
      double complex f12p21, f12m21, f34p43, f34m43
      double complex kvc21, kvc31, kvc41, kvc32, kvc42, kvc43
      double precision  rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      q(5) = fi(5) - fo(5)
      q(6) = fi(6) - fo(6)

      f1122  = fo(1)*fi(1) - fo(2)*fi(2)
      f12    = fo(1)*fi(2)
      f21    = fo(2)*fi(1)
      f12p21 = f12 + f21
      f12m21 = f12 - f21

      kvc21 = ( dble(q(6))*vc(1) -  dble(q(5))*vc(2))*cImag
      kvc31 =  dimag(q(6))*vc(1) -  dble(q(5))*vc(3)
      kvc41 = (dimag(q(5))*vc(1) -  dble(q(5))*vc(4))*cImag
      kvc32 =  dimag(q(6))*vc(2) -  dble(q(6))*vc(3)
      kvc42 = (dimag(q(5))*vc(2) -  dble(q(6))*vc(4))*cImag
      kvc43 =  dimag(q(5))*vc(3) - dimag(q(6))*vc(4)

      dum1 =   ( kvc31 + kvc42 )*f12m21
     &       + ( kvc32 + kvc41 )*f1122
     &       + ( kvc43 + kvc21 )*f12p21

c     (-) from gamma^5 in EDM only
      vertex = ( -gc(1,1) + cImag*gc(1,2) )*dum1    

      if ( gc(2,1).ne.cZero .or.
     &     gc(2,2).ne.cZero      ) then
         f3344  = fo(3)*fi(3) - fo(4)*fi(4)
         f34    = fo(3)*fi(4)
         f43    = fo(4)*fi(3)
         f34p43 = f34 + f43
         f34m43 = f34 - f43
         dum2 =   (-kvc31 + kvc42 )*f34m43
     &          + ( kvc32 - kvc41 )*f3344
     &          + ( kvc43 - kvc21 )*f34p43
         vertex = vertex + ( gc(2,1) + cImag*gc(2,2) )*dum2
      end if
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine iovgld(fi,fo,vc,gc,idecay , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c SUSY Goldstino coupling. In this routine, the NLSP is decaying to a
c boson and a Goldstino. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex vertex         : amplitude                     <fo|v|fi>
c
      implicit none
      double complex  fi(6), fo(6), gc(2), vc(6), vertex
      double complex  AdotpG, A14p, A14m, A23p, A23m
      double complex  p14p, p14m, p23p, p23m
      double precision  pdotpG
      integer idecay

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.eq.1 ) then
         pdotpG =  dble( vc(5))*dble( fo(5))
     &           - dble( vc(6))*dble( fo(6))
     &           - dimag(vc(6))*dimag(fo(6))
     &           - dimag(vc(5))*dimag(fo(5))
         AdotpG =  vc(1)*dble( fo(5)) - vc(2)*dble( fo(6))
     &           - vc(3)*dimag(fo(6)) - vc(4)*dimag(fo(5))
      else if ( idecay.eq.-1 ) then
         pdotpG =  dble( vc(5))*dble( fi(5))
     &           - dble( vc(6))*dble( fi(6))
     &           - dimag(vc(6))*dimag(fi(6))
     &           - dimag(vc(5))*dimag(fi(5))
         AdotpG =  vc(1)*dble( fi(5)) - vc(2)*dble( fi(6))
     &           - vc(3)*dimag(fi(6)) - vc(4)*dimag(fi(5))
      else
         write(6,*) 'error in idecay of IOVGLD'
         stop
      end if

      p14p = dble(vc(5)) + dimag(vc(5))
      p14m = dble(vc(5)) - dimag(vc(5))
      p23p = vc(6)
      p23m = dconjg(vc(6))
      A14p = vc(1) + vc(4)
      A14m = vc(1) - vc(4)
      A23p = vc(2) + vc(3)*ci
      A23m = vc(2) - vc(3)*ci

      vertex = gc(1)*( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &                  +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*AdotpG
     &                -( ( fo(3)*A14p + fo(4)*A23p )*fi(1)
     &                  +( fo(3)*A23m + fo(4)*A14m )*fi(2) )*pdotpG )

      if ( gc(2).ne.cZero ) then
         vertex = vertex
     &          + gc(2)*( ( (fo(1)*p14m - fo(2)*p23p )*fi(3)
     &                     -(fo(1)*p23m - fo(2)*p14p )*fi(4) )*AdotpG
     &                   -( (fo(1)*A14m - fo(2)*A23p )*fi(3)
     &                     -(fo(1)*A23m - fo(2)*A14p )*fi(4) )*pdotpG )
      end if

      vertex = vertex * idecay
c
      return
      end
      subroutine iovtkk(fi,fo,vc,tc,g , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a Kaluza-Klein tensor boson.
c
c input:
c       complex fi(6)          : flow-in  fermion     SM |fi>
c       complex fo(6)          : flow-out fermion     SM <fo|
c       complex vc(6)          : vector               SM   v
c       complex tc(6,4)        : tensor               KK   t
c       real    g(2)           : coupling constant    -g(L,R)*kappa/4
c
c output:
c       complex vertex         : amplitude            gamma(fi,fo,vc,tc)
c     
      implicit none
      double complex fi(6), fo(6), vc(6), tc(6,4), vertex
      double precision g(2)

      double complex f13, f14, f23, f24, f31, f32, f41, f42
      double complex fs1L, fs1R, fs2L, fs2R, fs3L, fs3R, fs4L, fs4R
      double complex T12, T13, T14, T23, T24, T34

      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
      f31 = fo(3)*fi(1)*g(1)
      f32 = fo(3)*fi(2)*g(1)
      f41 = fo(4)*fi(1)*g(1)
      f42 = fo(4)*fi(2)*g(1)

      fs1L =  f31 + f42
      fs2L = -f32 - f41
      fs3L = (f32 - f41)*ci
      fs4L = -f31 + f42

      if ( g(2).ne.rZero ) then
         f14 = fo(1)*fi(4)*g(2)
         f13 = fo(1)*fi(3)*g(2)
         f23 = fo(2)*fi(3)*g(2)
         f24 = fo(2)*fi(4)*g(2)
         fs1R =  f13 + f24
         fs2R =  f23 + f14
         fs3R = (f23 - f14)*ci
         fs4R =  f13 - f24
      end if

      T12 = tc(1,2) + tc(2,1)
      T13 = tc(1,3) + tc(3,1)
      T14 = tc(1,4) + tc(4,1)
      T23 = tc(2,3) + tc(3,2)
      T24 = tc(2,4) + tc(4,2)
      T34 = tc(3,4) + tc(4,3)

      if ( g(2).ne.rZero ) then

         vertex =  (fs1L + fs1R)*(  vc(1)*rTwo*tc(1,1)
     &                            - vc(2)*T12 - vc(3)*T13 - vc(4)*T14 )

     &           + (fs2L + fs2R)*(  vc(2)*rTwo*tc(2,2)
     &                            - vc(1)*T12 + vc(3)*T23 + vc(4)*T24 )

     &           + (fs3L + fs3R)*(  vc(3)*rTwo*tc(3,3)
     &                            - vc(1)*T13 + vc(2)*T23 + vc(4)*T34 )

     &           + (fs4L + fs4R)*(  vc(4)*rTwo*tc(4,4)
     &                            - vc(1)*T14 + vc(2)*T24 + vc(3)*T34 )

         vertex = vertex - rTwo*( tc(1,1)-tc(2,2)-tc(3,3)-tc(4,4) )
     &                         *(  (vc(1)+      vc(4))*(f31+f24)
     &                           + (vc(1)-      vc(4))*(f13+f42)
     &                           + (vc(2)+ci*vc(3))*(f41-f23)
     &                           + (vc(2)-ci*vc(3))*(f32-f14) )

      else

         vertex =  fs1L*(  vc(1)*rTwo*tc(1,1)
     &                   - vc(2)*T12 - vc(3)*T13 - vc(4)*T14 )

     &           + fs2L*(  vc(2)*rTwo*tc(2,2)
     &                   - vc(1)*T12 + vc(3)*T23 + vc(4)*T24 )

     &           + fs3L*(  vc(3)*rTwo*tc(3,3)
     &                   - vc(1)*T13 + vc(2)*T23 + vc(4)*T34 )

     &           + fs4L*(  vc(4)*rTwo*tc(4,4)
     &                   - vc(1)*T14 + vc(2)*T24 + vc(3)*T34 )

         vertex = vertex - rTwo*( tc(1,1)-tc(2,2)-tc(3,3)-tc(4,4) )
     &                         *(  (vc(1)+      vc(4))*f31
     &                           + (vc(1)-      vc(4))*f42
     &                           + (vc(2)+ci*vc(3))*f41
     &                           + (vc(2)-ci*vc(3))*f32 )

      end if
c
      return
      end
      subroutine iovxxx(fi,fo,vc,gc , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c
c output:
c       complex vertex         : amplitude                     <fo|v|fi>
c     
      implicit none
      double complex fi(6),fo(6),gc(2),vc(6),vertex

      double precision rZero, rOne
      parameter( rZero = 0.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      vertex =  gc(1)*( (fo(3)*fi(1)+fo(4)*fi(2))*vc(1)
     &                 +(fo(3)*fi(2)+fo(4)*fi(1))*vc(2)
     &                 -(fo(3)*fi(2)-fo(4)*fi(1))*vc(3)*cImag
     &                 +(fo(3)*fi(1)-fo(4)*fi(2))*vc(4)        )

      if ( gc(2).ne.cZero ) then
         vertex = vertex
     &          + gc(2)*( (fo(1)*fi(3)+fo(2)*fi(4))*vc(1)
     &                   -(fo(1)*fi(4)+fo(2)*fi(3))*vc(2)
     &                   +(fo(1)*fi(4)-fo(2)*fi(3))*vc(3)*cImag
     &                   -(fo(1)*fi(3)-fo(2)*fi(4))*vc(4)        )
      end if
c
      return
      end
      subroutine ixxxxx(p,fmass,nhel,nsf , fi)
c
c This subroutine computes a fermion wavefunction with the flowing-IN
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(6)          : fermion wavefunction               |fi>
c     
      implicit none
      double complex fi(6),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm
      integer nhel,nsf,ip,im,nh

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      
c

      fi(5) = dcmplx(p(0),p(3))*nsf
      fi(6) = dcmplx(p(1),p(2))*nsf

      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
         
         if ( pp.eq.rZero ) then
            
            sqm = dsqrt(fmass)
            ip = (1+nh)/2
            im = (1-nh)/2
            
            fi(1) = ip     * sqm
            fi(2) = im*nsf * sqm
            fi(3) = ip*nsf * sqm
            fi(4) = im     * sqm            

         else

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , p(2) )/dsqrt(rTwo*pp*pp3)
            endif
            
            fi(1) = sfomeg(1)*chi(im)
            fi(2) = sfomeg(1)*chi(ip)
            fi(3) = sfomeg(2)*chi(im)
            fi(4) = sfomeg(2)*chi(ip)
            
         endif
         
      else
         
         sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fi(1) = dcmplx( rZero )
            fi(2) = dcmplx( rZero )
            fi(3) = chi(1)
            fi(4) = chi(2)
         else
            fi(1) = chi(2)
            fi(2) = chi(1)
            fi(3) = dcmplx( rZero )
            fi(4) = dcmplx( rZero )
         endif
      endif
c
      return
      end
      subroutine j3xxxx(fi,fo,gaf,gzf,zmass,zwidth , j3)
c
c This subroutine computes the sum of photon and Z currents with the
c suitable weights ( j(W3) = cos(theta_W) j(Z) + sin(theta_W) j(A) ).
c The output j3 is useful as an input of vvvxxx, jvvxxx or w3w3xx.
c The photon propagator is given in Feynman gauge, and the Z propagator
c is given in unitary gauge.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gaf(2)         : fi couplings with A                 gaf
c       complex gzf(2)         : fi couplings with Z                 gzf
c       real    zmass          : mass  of Z
c       real    zwidth         : width of Z
c
c output:
c       complex j3(6)          : W3 current             j^mu(<fo|w3|fi>)
c     
      implicit none
      double complex fi(6),fo(6),j3(6),gaf(2),gzf(2)
      double complex c0l,c1l,c2l,c3l,csl,c0r,c1r,c2r,c3r,csr,dz,ddif
      double complex gn,gz3l,ga3l
      double complex cm2  ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),zmass,zwidth,zm2,zmw
      double precision q2,da,ww,cw,sw

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      j3(5) = fo(5)-fi(5)
      j3(6) = fo(6)-fi(6)

      q(0) = -dble( j3(5))
      q(1) = -dble( j3(6))
      q(2) = -dimag(j3(6))
      q(3) = -dimag(j3(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      zm2 = zmass**2
      zmw = zmass*zwidth


      da = rOne/q2
C      ww = max(dsign(zmw,q2), rZero)
      dz = rOne/dcmplx( q2-zm2, zmw )
      ddif = dcmplx( -zm2, zmw )*da*dz

c ddif is the difference : ddif=da-dz
c  For the running width, use below instead of the above ww,dz and ddif.
c      ww = max( zwidth*q2/zmass, rZero )
c      dz = rOne/dcmplx( q2-zm2, zmw )
c      ddif = dcmplx( -zm2, zmw )*da*dz



      cw = rOne/sqrt(rOne+(gzf(2)/gaf(2))**2)
      sw = sqrt((rOne-cw)*(rOne+cw))
      gn = gaf(2)*sw
      gz3l = gzf(1)*cw
      ga3l = gaf(1)*sw
      c0l =   fo(3)*fi(1)+fo(4)*fi(2)
      c0r =   fo(1)*fi(3)+fo(2)*fi(4)
      c1l = -(fo(3)*fi(2)+fo(4)*fi(1))
      c1r =   fo(1)*fi(4)+fo(2)*fi(3)
      c2l =  (fo(3)*fi(2)-fo(4)*fi(1))*cImag
      c2r = (-fo(1)*fi(4)+fo(2)*fi(3))*cImag
      c3l =  -fo(3)*fi(1)+fo(4)*fi(2)
      c3r =   fo(1)*fi(3)-fo(2)*fi(4)

c     Fabio's implementation of the fixed width
      cm2=dcmplx( zm2, -zmw )
c     csl = (q(0)*c0l-q(1)*c1l-q(2)*c2l-q(3)*c3l)/zm2
c     csr = (q(0)*c0r-q(1)*c1r-q(2)*c2r-q(3)*c3r)/zm2
      csl = (q(0)*c0l-q(1)*c1l-q(2)*c2l-q(3)*c3l)/cm2
      csr = (q(0)*c0r-q(1)*c1r-q(2)*c2r-q(3)*c3r)/cm2
      
      j3(1) =  gz3l*dz*(c0l-csl*q(0))+ga3l*c0l*da
     &       + gn*(c0r*ddif+csr*q(0)*dz)
      j3(2) =  gz3l*dz*(c1l-csl*q(1))+ga3l*c1l*da
     &       + gn*(c1r*ddif+csr*q(1)*dz)
      j3(3) =  gz3l*dz*(c2l-csl*q(2))+ga3l*c2l*da
     &       + gn*(c2r*ddif+csr*q(2)*dz)
      j3(4) =  gz3l*dz*(c3l-csl*q(3))+ga3l*c3l*da
     &       + gn*(c3r*ddif+csr*q(3)*dz)
c
      return
      end
      subroutine jeexxx(eb,ef,shlf,chlf,phi,nhb,nhf,nsf , jee)
c
c This subroutine computes an off-shell photon wavefunction emitted from
c the electron or positron beam, with a special care for the small angle
c region.  The momenta are measured in the laboratory frame, where the
c e- (e+) beam is along the positive (negative) z axis.
c
c input:
c       real    eb             : energy (gev)    of beam  e-/e+
c       real    ef             : energy (gev)    of final e-/e+
c       real    shlf           : sin(theta/2)    of final e-/e+
c       real    chlf           : cos(theta/2)    of final e-/e+
c       real    phi            : azimuthal angle of final e-/e+
c       integer nhb  = -1 or 1 : helicity        of beam  e-/e+
c       integer nhf  = -1 or 1 : helicity        of final e-/e+
c       integer nsf  = -1 or 1 : +1 for electron, -1 for positron
c
c output:
c       complex jee(6)         : off-shell photon          j^mu(<e|a|e>)
c     
      implicit none
      double complex jee(6),coeff
      double precision cs(2),eb,ef,shlf,chlf,phi,alpha,gal,hi,sf,sfh
      double precision x,me2,q2,rfp,rfm,snp,csp,rxc,c,s
      integer nhb,nhf,nsf

      double precision rZero, rHalf, rOne, rTwo, rFour, rOte
      double precision rPi, rIalph
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rOne = 1.0d0 )
      parameter( rTwo = 2.0d0, rFour = 4.0d0, rOte = 128.9d0 )
      parameter( rPi = 3.14159265358979323846d0 )
      parameter( rIalph = 137.0359895d0 )

      double precision me
      parameter( me = 0.51099906d-3 )

c

      alpha = rOne/rOte
      gal = sqrt(alpha*rFour*rPi)

      hi = nhb
      sf = nsf
      sfh = nhb*nsf
      cs((3+nsf)/2) = shlf
      cs((3-nsf)/2) = chlf
c cs(1)=chlf and cs(2)=shlf for electron
c cs(1)=shlf and cs(2)=chlf for positron
      x = ef/eb
      me2 = me**2
      q2 = - rFour*cs(2)**2*(ef*eb-me2)
     &     + sf*(rOne-x)**2/x*(shlf+chlf)*(shlf-chlf)*me2
      rfp = (1+nsf)
      rfm = (1-nsf)
      snp = sin(phi)
      csp = cos(phi)

      if ( nhb.eq.nhf ) then
         rxc = rTwo*x/(rOne-x)*cs(1)**2
         coeff = gal*rTwo*eb*sqrt(x)*cs(2)/q2
     &          *(dcmplx( rfp )-rfm*dcmplx( csp, -snp*hi ))*rHalf
         jee(1) = dcmplx( rZero )
         jee(2) = coeff*dcmplx( (rOne+rxc)*csp, -sfh*snp )
         jee(3) = coeff*dcmplx( (rOne+rxc)*snp,  sfh*csp )
         jee(4) = coeff*(-sf*rxc/cs(1)*cs(2))
      else
         coeff = gal*me/q2/sqrt(x)
     &          *(dcmplx( rfp )+rfm*dcmplx( csp, snp*hi ))*rHalf*hi
         jee(1) = -coeff*(rOne+x)*cs(2)*dcmplx( csp , sfh*snp )
         jee(2) =  coeff*(rOne-x)*cs(1)
         jee(3) =  jee(2)*dcmplx( rZero, sfh )
         jee(4) =  jee(1)*sf*(rOne-x)/(rOne+x)
      endif

      c = (chlf+shlf)*(chlf-shlf)
      s = rTwo*chlf*shlf

      jee(5) = -eb*dcmplx( rOne-x, sf-x*c )
      jee(6) =  eb*x*s*dcmplx( csp, snp )
c
      return
      end
      subroutine jgggxx(w1,w2,w3,g, jw3w)
c
c This subroutine computes an off-shell W+, W-, W3, Z or photon current
c from the four-point gauge boson coupling, including the contributions
c of W exchange diagrams.  The vector propagator is given in Feynman
c gauge for a photon and in unitary gauge for W and Z bosons.  If one
c sets wmass=0.0, then the ggg-->g current is given (see sect 2.9.1 of
c the manual).
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       complex w3(6)          : third  vector                        w3
c       real    g             : first  coupling constant
c                                                  (see the table below)
c
c output:
c       complex jw3w(6)        : W current             j^mu(w':w1,w2,w3)
c
      implicit none
      double complex w1(6),w2(6),w3(6),jw3w(6)
      double complex dw1(0:3),dw2(0:3),dw3(0:3)
      double complex jj(0:3),dv,w32,w13
      double precision p1(0:3),p2(0:3),p3(0:3),q(0:3),g,dg2,q2

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

c

      jw3w(5) = w1(5)+w2(5)+w3(5)
      jw3w(6) = w1(6)+w2(6)+w3(6)


      dw1(0) = dcmplx(w1(1))
      dw1(1) = dcmplx(w1(2))
      dw1(2) = dcmplx(w1(3))
      dw1(3) = dcmplx(w1(4))
      dw2(0) = dcmplx(w2(1))
      dw2(1) = dcmplx(w2(2))
      dw2(2) = dcmplx(w2(3))
      dw2(3) = dcmplx(w2(4))
      dw3(0) = dcmplx(w3(1))
      dw3(1) = dcmplx(w3(2))
      dw3(2) = dcmplx(w3(3))
      dw3(3) = dcmplx(w3(4))
      p1(0) = dble(      w1(5))
      p1(1) = dble(      w1(6))
      p1(2) = dble(dimag(w1(6)))
      p1(3) = dble(dimag(w1(5)))
      p2(0) = dble(      w2(5))
      p2(1) = dble(      w2(6))
      p2(2) = dble(dimag(w2(6)))
      p2(3) = dble(dimag(w2(5)))
      p3(0) = dble(      w3(5))
      p3(1) = dble(      w3(6))
      p3(2) = dble(dimag(w3(6)))
      p3(3) = dble(dimag(w3(5)))
      q(0) = -(p1(0)+p2(0)+p3(0))
      q(1) = -(p1(1)+p2(1)+p3(1))
      q(2) = -(p1(2)+p2(2)+p3(2))
      q(3) = -(p1(3)+p2(3)+p3(3))

      q2 = q(0)**2 -(q(1)**2 +q(2)**2 +q(3)**2)

      dg2 = dble(g)*dble(g)

      dv = rOne/dcmplx( q2 )

      w32 = dw3(0)*dw2(0)-dw3(1)*dw2(1)-dw3(2)*dw2(2)-dw3(3)*dw2(3)

      w13 = dw1(0)*dw3(0)-dw1(1)*dw3(1)-dw1(2)*dw3(2)-dw1(3)*dw3(3)

      jj(0) = dg2*( dw1(0)*w32 - dw2(0)*w13 )
      jj(1) = dg2*( dw1(1)*w32 - dw2(1)*w13 )
      jj(2) = dg2*( dw1(2)*w32 - dw2(2)*w13 )
      jj(3) = dg2*( dw1(3)*w32 - dw2(3)*w13 )

      jw3w(1) = dcmplx( jj(0)*dv )
      jw3w(2) = dcmplx( jj(1)*dv )
      jw3w(3) = dcmplx( jj(2)*dv )
      jw3w(4) = dcmplx( jj(3)*dv )
c
      return
      end
      subroutine jggxxx(v1,v2,g, jvv)
c
c This subroutine computes an off-shell vector current from the three-
c point gauge boson coupling.  The vector propagator is given in Feynman
c gauge for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       real    g              : coupling constant (see the table below)
c
c output:
c       complex jvv(6)         : vector current            j^mu(v:v1,v2)
c
      implicit none
      double complex v1(6),v2(6),jvv(6),j12(0:3)
      double complex sv1,sv2,v12
      double precision p1(0:3),p2(0:3),q(0:3),g,gs,s

c

      jvv(5) = v1(5) + v2(5)
      jvv(6) = v1(6) + v2(6)

      p1(0) = dble( v1(5))
      p1(1) = dble( v1(6))
      p1(2) = dimag(v1(6))
      p1(3) = dimag(v1(5))
      p2(0) = dble( v2(5))
      p2(1) = dble( v2(6))
      p2(2) = dimag(v2(6))
      p2(3) = dimag(v2(5))
      q(0) = -dble( jvv(5))
      q(1) = -dble( jvv(6))
      q(2) = -dimag(jvv(6))
      q(3) = -dimag(jvv(5))
      s = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

      v12 = v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4)
      sv1 =   (p2(0)-q(0))*v1(1) -(p2(1)-q(1))*v1(2)
     &      - (p2(2)-q(2))*v1(3) -(p2(3)-q(3))*v1(4)
      sv2 = - (p1(0)-q(0))*v2(1) +(p1(1)-q(1))*v2(2)
     &      + (p1(2)-q(2))*v2(3) +(p1(3)-q(3))*v2(4)
      j12(0) = (p1(0)-p2(0))*v12 +sv1*v2(1) +sv2*v1(1)
      j12(1) = (p1(1)-p2(1))*v12 +sv1*v2(2) +sv2*v1(2)
      j12(2) = (p1(2)-p2(2))*v12 +sv1*v2(3) +sv2*v1(3)
      j12(3) = (p1(3)-p2(3))*v12 +sv1*v2(4) +sv2*v1(4)

      gs = -g/s

      jvv(1) = gs*j12(0)
      jvv(2) = gs*j12(1)
      jvv(3) = gs*j12(2)
      jvv(4) = gs*j12(3)
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine jiocxx(fic,foc,gc,vmass,vwidth , jioc)
c
c This subroutine computes an off-shell vector current from an external 
c antifermion pair. The vector boson propagator is given in Feynman     
c gauge for a massless vector and in unitary gauge for a massive vector.
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of output vector v               
c       real    vwidth         : width of output vector v               
c                                                                       
c output:                                                               
c       complex jioc(6)        : vector current        j^mu(<foc|v|fic>)
c
      implicit none
      double complex fic(6),foc(6),gc(2),jioc(6),c0,c1,c2,c3,cs,d
      double precision q(0:3),vmass,vwidth,q2,vm2,dd

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      jioc(5) = foc(5)-fic(5)
      jioc(6) = foc(6)-fic(6)

      q(0) = dble( jioc(5))
      q(1) = dble( jioc(6))
      q(2) = dimag(jioc(6))
      q(3) = dimag(jioc(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

      if ( vmass.ne.rZero ) then

         d = -rOne/dcmplx( q2-vm2, vmass*vwidth )
c  for the running width, use below instead of the above d.
c         d = -rOne/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

         if ( gc(2).ne.cZero ) then
            c0=  gc(2)*( foc(3)*fic(1)+foc(4)*fic(2))
     &          +gc(1)*( foc(1)*fic(3)+foc(2)*fic(4))
            c1= -gc(2)*( foc(3)*fic(2)+foc(4)*fic(1))
     &          +gc(1)*( foc(1)*fic(4)+foc(2)*fic(3))
            c2=( gc(2)*( foc(3)*fic(2)-foc(4)*fic(1)) 
     &          +gc(1)*(-foc(1)*fic(4)+foc(2)*fic(3)))*cImag
            c3=  gc(2)*(-foc(3)*fic(1)+foc(4)*fic(2))
     &          +gc(1)*( foc(1)*fic(3)-foc(2)*fic(4))
         else
            d = d*gc(1)
            c0 =  foc(1)*fic(3)+foc(2)*fic(4)
            c1 =  foc(1)*fic(4)+foc(2)*fic(3)
            c2 =(-foc(1)*fic(4)+foc(2)*fic(3))*cImag
            c3 =  foc(1)*fic(3)-foc(2)*fic(4)
         end if
         cs = (q(0)*c0-q(1)*c1-q(2)*c2-q(3)*c3)/vm2
         jioc(1) = (c0-cs*q(0))*d
         jioc(2) = (c1-cs*q(1))*d
         jioc(3) = (c2-cs*q(2))*d
         jioc(4) = (c3-cs*q(3))*d

      else

         d = dcmplx( -rOne/q2, rZero )
         if ( gc(2).ne.cZero ) then
            jioc(1) = ( gc(2)*( foc(3)*fic(1)+foc(4)*fic(2))
     &                 +gc(1)*( foc(1)*fic(3)+foc(2)*fic(4)) )*d
            jioc(2) = (-gc(2)*( foc(3)*fic(2)+foc(4)*fic(1))
     &                 +gc(1)*( foc(1)*fic(4)+foc(2)*fic(3)) )*d
            jioc(3) = ( gc(2)*( foc(3)*fic(2)-foc(4)*fic(1))
     &                 +gc(1)*(-foc(1)*fic(4)+foc(2)*fic(3)) )
     &                *d*cImag
            jioc(4) = ( gc(2)*(-foc(3)*fic(1)+foc(4)*fic(2))
     &                 +gc(1)*( foc(1)*fic(3)-foc(2)*fic(4)) )*d
         else
            d = d*gc(1)
            jioc(1) = ( foc(1)*fic(3)+foc(2)*fic(4))*d
            jioc(2) = ( foc(1)*fic(4)+foc(2)*fic(3))*d
            jioc(3) = (-foc(1)*fic(4)+foc(2)*fic(3))*d*cImag
            jioc(4) = ( foc(1)*fic(3)-foc(2)*fic(4))*d
         end if

      end if
c
      return
      end
      subroutine jiodmx(fi,fo,gc,vmass,vwidth , jio)
c
c This subroutine computes an off-shell vector dipole moment
c (non-renormalizable) current from an external
c fermion pair.  The vector boson propagator is given in Feynman gauge
c for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2,2)        : coupling constants                 gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c
      implicit none
      double complex fi(6), fo(6), jio(6), c0, c1, c2, c3, d
      double complex gc(2,2), gL, gR
      double precision  q(0:3), vmass, vwidth, q2, vm2, dd

      double complex f1122, f12, f21, f12p21, f12m21
      double complex f3344, f34, f43, f34p43, f34m43
      double complex dumL1, dumL2, dumL3, dumL4
      double complex dumR1, dumR2, dumR3, dumR4
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      gL = -gc(1,1) + cImag*gc(1,2)
      gR =  gc(2,1) + cImag*gc(2,2)

      jio(5) = fo(5) - fi(5)
      jio(6) = fo(6) - fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      f1122  = fo(1)*fi(1) - fo(2)*fi(2)
      f12    = fo(1)*fi(2)
      f21    = fo(2)*fi(1)
      f12p21 = f12 + f21
      f12m21 = f12 - f21
      if ( gc(2,1).ne.cZero .or. gc(2,2).ne.cZero ) then
         f3344  = fo(3)*fi(3) - fo(4)*fi(4)
         f34    = fo(3)*fi(4)
         f43    = fo(4)*fi(3)
         f34p43 = f34 + f43
         f34m43 = f34 - f43
      end if

c note overall (-), since k in vertex is -q above
      dumL1 = -q(2)*f12m21 - cImag*( q(1)*f12p21 + q(3)*f1122 )
      dumL2 =  q(2)*f1122  - cImag*( q(0)*f12p21 - q(3)*f12m21 )
      dumL3 = -q(0)*f12m21 - q(1)*f1122 - q(3)*f12p21
      dumL4 = -q(2)*f12p21 - cImag*( q(0)*f1122  + q(1)*f12m21 )
      if ( gc(2,1).ne.cZero .or. gc(2,2).ne.cZero ) then
         dumR1 =  q(2)*f34m43 + cImag*( q(1)*f34p43 + q(3)*f3344 )
         dumR2 =  q(2)*f3344  + cImag*( q(0)*f34p43 + q(3)*f34m43 )
         dumR3 =  q(0)*f34m43 - q(1)*f3344 - q(3)*f34p43
         dumR4 = -q(2)*f34p43 + cImag*( q(0)*f3344  - q(1)*f34m43 )
      end if

      if ( vmass.ne.rZero ) then

         d = rOne/dcmplx( q2-vm2, vmass*vwidth )

         c0 = gL*dumL1
         c1 = gL*dumL2
         c2 = gL*dumL3
         c3 = gL*dumL4

         if ( gc(2,1).ne.cZero .or.
     &        gc(2,2).ne.cZero      ) then
            c0 = c0 + gR*dumR1
            c1 = c1 + gR*dumR2
            c2 = c2 + gR*dumR3
            c3 = c3 + gR*dumR4
         end if

         jio(1) = c0*d
         jio(2) = c1*d
         jio(3) = c2*d
         jio(4) = c3*d

      else

         dd = rOne/q2

         jio(1) = gL*dumL1
         jio(2) = gL*dumL2
         jio(3) = gL*dumL3
         jio(4) = gL*dumL4

         if ( gc(2,1).ne.cZero .or.
     &        gc(2,2).ne.cZero      ) then
            jio(1) = jio(1) + gR*dumR1
            jio(2) = jio(2) + gR*dumR2
            jio(3) = jio(3) + gR*dumR3
            jio(4) = jio(4) + gR*dumR4
         end if

         jio(1) = jio(1)*dd
         jio(2) = jio(2)*dd
         jio(3) = jio(3)*dd
         jio(4) = jio(4)*dd

      end if
c
      return
      end
c
c ----------------------------------------------------------------------
c
      subroutine jiogld(fi,fo,gc,vmass,vwidth,idecay , jio)
c
c This subroutine computes an off-shell vector current for the NLSP-
c Goldstino vertex from an external fermion pair. The vector boson 
c propagator is given in feynman gauge for a massless vector and in 
c unitary gauge for a massive vector. The h.c. of the NLSP decay is
c handled via the input parameter idecay (picks out correct
c Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c
      implicit none
      double complex  fi(6), fo(6), gc(2), jio(6), c0, c1, c2, c3, cs
      double complex  d, dum, p14p, p14m, p23p, p23m
      double precision  q(0:3), vmass, vwidth, q2, vm2, dd
      double precision  pG(1:4), pdotpG
      integer idecay

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.eq.1 ) then
         pG(1) =  dble(fo(5))
         pG(2) =  dble(fo(6))
         pG(3) = dimag(fo(6))
         pG(4) = dimag(fo(5))
      else if ( idecay.eq.-1 ) then
         pG(1) =  dble(fi(5))
         pG(2) =  dble(fi(6))
         pG(3) = dimag(fi(6))
         pG(4) = dimag(fi(5))
      else
         write(6,*) 'error in idecay of JIOGLD'
         stop
      end if

      jio(5) = fo(5) - fi(5)
      jio(6) = fo(6) - fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2  = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2
      vm2 = vmass**2

      pdotpG = q(0)*pG(1) - q(1)*pG(2) - q(2)*pG(3) - q(3)*pG(4)

      p14p = q(0) + q(3)
      p14m = q(0) - q(3)
      p23p = jio(6)
      p23m = dconjg(jio(6))

      if ( vmass.ne.rZero ) then

         d = rOne/dcmplx( q2-vm2, vmass*vwidth )
         d = d*idecay
c  for the running width, use below instead of the above d.
c         d = rOne/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

         if ( gc(2).ne.cZero ) then
            dum =  ( (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &              +(fo(3)*p23m + fo(4)*p14m)*fi(2) )*gc(1)
     &           + ( (fo(1)*p14m - fo(2)*p23p)*fi(3)
     &              -(fo(1)*p23m - fo(2)*p14p)*fi(4) )*gc(2)

            c0 =  dum*pG(1)
     &           -pdotpG*( gc(1)*( fo(3)*fi(1) + fo(4)*fi(2) )
     &                    +gc(2)*( fo(1)*fi(3) + fo(2)*fi(4) ) )
            c1 =  dum*pG(2)
     &           -pdotpG*(-gc(1)*( fo(4)*fi(1) + fo(3)*fi(2) )
     &                    +gc(2)*( fo(2)*fi(3) + fo(1)*fi(4) ) )
            c2 =  dum*pG(3)
     &           -pdotpG*( gc(1)*(-fo(4)*fi(1) + fo(3)*fi(2) )
     &                    +gc(2)*( fo(2)*fi(3) - fo(1)*fi(4) ) )*ci
            c3 =  dum*pG(4)
     &           -pdotpG*( gc(1)*(-fo(3)*fi(1) + fo(4)*fi(2) )
     &                    +gc(2)*( fo(1)*fi(3) - fo(2)*fi(4) ) )
         else
            d = d*gc(1)
            dum =  (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &            +(fo(3)*p23m + fo(4)*p14m)*fi(2)

            c0 = dum*pG(1) - ( fo(3)*fi(1) + fo(4)*fi(2) )*pdotpG
            c1 = dum*pG(2) + ( fo(4)*fi(1) + fo(3)*fi(2) )*pdotpG
            c2 = dum*pG(3) + ( fo(4)*fi(1) - fo(3)*fi(2) )*pdotpG*ci
            c3 = dum*pG(4) + ( fo(3)*fi(1) - fo(4)*fi(2) )*pdotpG
         end if

         cs = (q(0)*c0 - q(1)*c1 - q(2)*c2 - q(3)*c3) / vm2

         jio(1) = (c0-cs*q(0))*d
         jio(2) = (c1-cs*q(1))*d
         jio(3) = (c2-cs*q(2))*d
         jio(4) = (c3-cs*q(3))*d

      else
         dd = idecay*rOne/q2

         if ( gc(2).ne.cZero ) then
            dum =  ( (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &              +(fo(3)*p23m + fo(4)*p14m)*fi(2) )*gc(1)
     &           + ( (fo(1)*p14m - fo(2)*p23p)*fi(3)
     &              -(fo(1)*p23m - fo(2)*p14p)*fi(4) )*gc(2)

            jio(1) = ( dum*pG(1) - pdotpG*(
     &                 gc(1)*( fo(3)*fi(1) + fo(4)*fi(2) )
     &                +gc(2)*( fo(1)*fi(3) + fo(2)*fi(4) ) ) )*dd
            jio(2) = ( dum*pG(2) - pdotpG*(
     &                -gc(1)*( fo(4)*fi(1) + fo(3)*fi(2) )
     &                +gc(2)*( fo(2)*fi(3) + fo(1)*fi(4) ) ) )*dd
            jio(3) = ( dum*pG(3) - pdotpG*ci*(
     &                 gc(1)*(-fo(4)*fi(1) + fo(3)*fi(2) )
     &                +gc(2)*( fo(2)*fi(3) - fo(1)*fi(4) ) ) )*dd
            jio(4) = ( dum*pG(4) - pdotpG*(
     &                 gc(1)*(-fo(3)*fi(1) + fo(4)*fi(2) )
     &                +gc(2)*( fo(1)*fi(3) - fo(2)*fi(4) ) ) )*dd

         else
            dd = dd*gc(1)
            dum =  (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &            +(fo(3)*p23m + fo(4)*p14m)*fi(2)

            jio(1)=dd*(dum*pG(1) - pdotpG*(fo(3)*fi(1) + fo(4)*fi(2)))
            jio(2)=dd*(dum*pG(2) + pdotpG*(fo(4)*fi(1) + fo(3)*fi(2)))
            jio(3)=dd*(dum*pG(3) + ci*pdotpG*(fo(4)*fi(1)-fo(3)*fi(2)))
            jio(4)=dd*(dum*pG(4) + pdotpG*(fo(3)*fi(1) - fo(4)*fi(2)))
         end if
      end if
c
      return
      end
      subroutine jioxxx(fi,fo,gc,vmass,vwidth , jio)
c
c This subroutine computes an off-shell vector current from an external
c fermion pair.  The vector boson propagator is given in Feynman gauge
c for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of OUTPUT vector v
c       real    vwidth         : width of OUTPUT vector v
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c     
      implicit none
      double complex fi(6),fo(6),gc(2),jio(6),c0,c1,c2,c3,cs,d
      double precision q(0:3),vmass,vwidth,q2,vm2
      double complex cm2 ! mass**2- I Gamma mass (Fabio)


      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

c

      jio(5) = fo(5)-fi(5)
      jio(6) = fo(6)-fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      if ( vmass.ne.rZero ) then

         d = rOne/dcmplx( q2-vm2, vmass*vwidth )
c     For the running width, use below instead of the above d.
c     d = rOne/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )


         if ( gc(2).ne.cZero ) then
            c0 =  gc(1)*( fo(3)*fi(1)+fo(4)*fi(2))
     &           +gc(2)*( fo(1)*fi(3)+fo(2)*fi(4))
            c1 = -gc(1)*( fo(3)*fi(2)+fo(4)*fi(1))
     &           +gc(2)*( fo(1)*fi(4)+fo(2)*fi(3))
            c2 =( gc(1)*( fo(3)*fi(2)-fo(4)*fi(1))
     &           +gc(2)*(-fo(1)*fi(4)+fo(2)*fi(3)))*cImag
            c3 =  gc(1)*(-fo(3)*fi(1)+fo(4)*fi(2))
     &           +gc(2)*( fo(1)*fi(3)-fo(2)*fi(4))
         else
            d = d*gc(1)
            c0 =   fo(3)*fi(1)+fo(4)*fi(2)
            c1 =  -fo(3)*fi(2)-fo(4)*fi(1)
            c2 = ( fo(3)*fi(2)-fo(4)*fi(1))*cImag
            c3 =  -fo(3)*fi(1)+fo(4)*fi(2)
         end if

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     cs = (q(0)*c0-q(1)*c1-q(2)*c2-q(3)*c3)/vm2
         cs = (q(0)*c0-q(1)*c1-q(2)*c2-q(3)*c3)/cm2
         jio(1) = (c0-cs*q(0))*d
         jio(2) = (c1-cs*q(1))*d
         jio(3) = (c2-cs*q(2))*d
         jio(4) = (c3-cs*q(3))*d

      else

         d = dcmplx( rOne/q2, rZero )
         if ( gc(2).ne.cZero ) then
            jio(1) = ( gc(1)*( fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)+fo(2)*fi(4)) )*d
            jio(2) = (-gc(1)*( fo(3)*fi(2)+fo(4)*fi(1))
     &                +gc(2)*( fo(1)*fi(4)+fo(2)*fi(3)) )*d
            jio(3) = ( gc(1)*( fo(3)*fi(2)-fo(4)*fi(1))
     &                +gc(2)*(-fo(1)*fi(4)+fo(2)*fi(3)))
     &               *d*cImag
            jio(4) = ( gc(1)*(-fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)-fo(2)*fi(4)) )*d
         else
            d = d*gc(1)
            jio(1) =  ( fo(3)*fi(1)+fo(4)*fi(2))*d
            jio(2) = -( fo(3)*fi(2)+fo(4)*fi(1))*d
            jio(3) =  ( fo(3)*fi(2)-fo(4)*fi(1))*d*cImag
            jio(4) =  (-fo(3)*fi(1)+fo(4)*fi(2))*d
         end if

      end if
c
      return
      end
      subroutine jssxxx(s1,s2,gc,vmass,vwidth , jss)
c
c This subroutine computes an off-shell vector current from the vector-
c scalar-scalar coupling.  The coupling is absent in the minimal SM in
c unitary gauge.  The propagator is given in Feynman gauge for a
c massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant (s1 charge)
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c
c examples of the coupling constant g for susy particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=A   |   v=Z   |   v=W   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the s1 charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex jss(6)         : vector current            j^mu(v:s1,s2)
c     
      implicit none
      double complex s1(3),s2(3),gc,jss(6),dg,adg
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision pp(0:3),pa(0:3),q(0:3),vmass,vwidth
      double precision q2,vm2,mp2,ma2,m2d

      double precision rZero
      parameter( rZero = 0.0d0 )

c

      jss(5) = s1(2)+s2(2)
      jss(6) = s1(3)+s2(3)

      q(0) = dble( jss(5))
      q(1) = dble( jss(6))
      q(2) = dimag(jss(6))
      q(3) = dimag(jss(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      if ( vmass.ne.rZero ) then

         dg = gc/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = g/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )


         adg = dg*s1(1)*s2(1)

         pp(0) = dble( s1(2))
         pp(1) = dble( s1(3))
         pp(2) = dimag(s1(3))
         pp(3) = dimag(s1(2))
         pa(0) = dble( s2(2))
         pa(1) = dble( s2(3))
         pa(2) = dimag(s2(3))
         pa(3) = dimag(s2(2))
         mp2 = pp(0)**2-(pp(1)**2+pp(2)**2+pp(3)**2)
         ma2 = pa(0)**2-(pa(1)**2+pa(2)**2+pa(3)**2)
         m2d = mp2-ma2

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     jss(1) = adg*( (pp(0)-pa(0)) - q(0)*m2d/vm2)
c     jss(2) = adg*( (pp(1)-pa(1)) - q(1)*m2d/vm2)
c     jss(3) = adg*( (pp(2)-pa(2)) - q(2)*m2d/vm2)
c     jss(4) = adg*( (pp(3)-pa(3)) - q(3)*m2d/vm2)
         jss(1) = adg*( (pp(0)-pa(0)) - q(0)*m2d/cm2)
         jss(2) = adg*( (pp(1)-pa(1)) - q(1)*m2d/cm2)
         jss(3) = adg*( (pp(2)-pa(2)) - q(2)*m2d/cm2)
         jss(4) = adg*( (pp(3)-pa(3)) - q(3)*m2d/cm2)

      else

         adg = gc*s1(1)*s2(1)/q2

         jss(1) = adg*dble( s1(2)-s2(2))
         jss(2) = adg*dble( s1(3)-s2(3))
         jss(3) = adg*dimag(s1(3)-s2(3))
         jss(4) = adg*dimag(s1(2)-s2(2))

      endif
c
      return
      end
      subroutine jvssxx(vc,s1,s2,gc,vmass,vwidth , jvss)
c
c This subroutine computes an off-shell vector current from the vector-
c vector-scalar-scalar coupling.  The vector propagator is given in
c Feynman gauge for a massless vector and in unitary gauge for a massive
c vector.
c
c input:
c       complex vc(6)          : input  vector                        v
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                 gvvhh
c       real    vmass          : mass  of output vector v'
c       real    vwidth         : width of output vector v'
c
c output:
c       complex jvss(6)        : vector current         j^mu(v':v,s1,s2)
c     
      implicit none
      double complex vc(6),s1(3),s2(3),gc,jvss(6),dg
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),vmass,vwidth,q2,vk,vm2

      double precision rZero
      parameter( rZero = 0.0d0 )

c

      jvss(5) = vc(5)+s1(2)+s2(2)
      jvss(6) = vc(6)+s1(3)+s2(3)

      q(0) = dble( jvss(5))
      q(1) = dble( jvss(6))
      q(2) = dimag(jvss(6))
      q(3) = dimag(jvss(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      if ( vmass.ne.rZero ) then

         dg = gc*s1(1)*s2(1)/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = gc*s1(1)*s2(1)/cmplx( q2-vm2 , max( vwidth*q2/vmass ,rZero))

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     vk = (q(0)*vc(1)-q(1)*vc(2)-q(2)*vc(3)-q(3)*vc(4))/vm2
         vk = (q(0)*vc(1)-q(1)*vc(2)-q(2)*vc(3)-q(3)*vc(4))/cm2

         jvss(1) = dg*(vc(1)-vk*q(0))
         jvss(2) = dg*(vc(2)-vk*q(1))
         jvss(3) = dg*(vc(3)-vk*q(2))
         jvss(4) = dg*(vc(4)-vk*q(3))

      else

         dg = gc*s1(1)*s2(1)/q2

         jvss(1) = dg*vc(1)
         jvss(2) = dg*vc(2)
         jvss(3) = dg*vc(3)
         jvss(4) = dg*vc(4)

      endif
c
      return
      end
      subroutine jvsxxx(vc,sc,gc,vmass,vwidth , jvs)
c
c This subroutine computes an off-shell vector current from the vector-
c vector-scalar coupling.  The vector propagator is given in Feynman
c gauge for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex vc(6)          : input vector                          v
c       complex sc(3)          : input scalar                          s
c       complex gc             : coupling constant                  gvvh
c       real    vmass          : mass  of output vector v'
c       real    vwidth         : width of output vector v'
c
c output:
c       complex jvs(6)         : vector current             j^mu(v':v,s)
c     
      implicit none
      double complex vc(6),sc(3),gc,jvs(6),dg,vk
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),vmass,vwidth,q2,vm2

      double precision rZero
      parameter( rZero = 0.0d0 )

c

      jvs(5) = vc(5)+sc(2)
      jvs(6) = vc(6)+sc(3)

      q(0) = dble( jvs(5))
      q(1) = dble( jvs(6))
      q(2) = dimag(jvs(6))
      q(3) = dimag(jvs(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      if ( vmass.ne.rZero ) then

         dg = gc*sc(1)/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = g*sc(1)/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     vk = (-q(0)*vc(1)+q(1)*vc(2)+q(2)*vc(3)+q(3)*vc(4))/vm2
         vk = (-q(0)*vc(1)+q(1)*vc(2)+q(2)*vc(3)+q(3)*vc(4))/cm2

         jvs(1) = dg*(q(0)*vk+vc(1))
         jvs(2) = dg*(q(1)*vk+vc(2))
         jvs(3) = dg*(q(2)*vk+vc(3))
         jvs(4) = dg*(q(3)*vk+vc(4))

      else

         dg=gc*sc(1)/q2

         jvs(1) = dg*vc(1)
         jvs(2) = dg*vc(2)
         jvs(3) = dg*vc(3)
         jvs(4) = dg*vc(4)

      endif
c
      return
      end
      subroutine jvvxxx(v1,v2,g,vmass,vwidth , jvv)
c
c This subroutine computes an off-shell vector current from the three-
c point gauge boson coupling.  The vector propagator is given in Feynman
c gauge for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       real    g              : coupling constant (see the table below)
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c
c the possible sets of the inputs are as follows:
c    ------------------------------------------------------------------
c    |   v1   |   v2   |  jvv   |      g       |   vmass  |  vwidth   |
c    ------------------------------------------------------------------
c    |   W-   |   W+   |  A/Z   |  gwwa/gwwz   | 0./zmass | 0./zwidth |
c    | W3/A/Z |   W-   |  W+    | gw/gwwa/gwwz |   wmass  |  wwidth   |
c    |   W+   | W3/A/Z |  W-    | gw/gwwa/gwwz |   wmass  |  wwidth   |
c    ------------------------------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex jvv(6)         : vector current            j^mu(v:v1,v2)
c     
      implicit none
      double complex v1(6),v2(6),jvv(6),j12(0:3),js,dg
      double complex sv1,sv2,s11,s12,s21,s22,v12
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision p1(0:3),p2(0:3),q(0:3),g,vmass,vwidth,gs,s
      double precision vm2,m1,m2

      double precision rZero
      parameter( rZero = 0.0d0 )
 
c

      jvv(5) = v1(5)+v2(5)
      jvv(6) = v1(6)+v2(6)

      p1(0) =  dble( v1(5))
      p1(1) =  dble( v1(6))
      p1(2) =  dimag(v1(6))
      p1(3) =  dimag(v1(5))
      p2(0) =  dble( v2(5))
      p2(1) =  dble( v2(6))
      p2(2) =  dimag(v2(6))
      p2(3) =  dimag(v2(5))
      q(0)  = -dble( jvv(5))
      q(1)  = -dble( jvv(6))
      q(2)  = -dimag(jvv(6))
      q(3)  = -dimag(jvv(5))
      s = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2


      v12 = v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4)
      sv1 =   (p2(0)-q(0))*v1(1) -(p2(1)-q(1))*v1(2)
     &      - (p2(2)-q(2))*v1(3) -(p2(3)-q(3))*v1(4)
      sv2 = - (p1(0)-q(0))*v2(1) +(p1(1)-q(1))*v2(2)
     &      + (p1(2)-q(2))*v2(3) +(p1(3)-q(3))*v2(4)
      j12(0) = (p1(0)-p2(0))*v12 +sv1*v2(1) +sv2*v1(1)
      j12(1) = (p1(1)-p2(1))*v12 +sv1*v2(2) +sv2*v1(2)
      j12(2) = (p1(2)-p2(2))*v12 +sv1*v2(3) +sv2*v1(3)
      j12(3) = (p1(3)-p2(3))*v12 +sv1*v2(4) +sv2*v1(4)

      if ( vmass.ne.rZero ) then

         m1 = p1(0)**2-(p1(1)**2+p1(2)**2+p1(3)**2)
         m2 = p2(0)**2-(p2(1)**2+p2(2)**2+p2(3)**2)
         s11 = p1(0)*v1(1)-p1(1)*v1(2)-p1(2)*v1(3)-p1(3)*v1(4)
         s12 = p1(0)*v2(1)-p1(1)*v2(2)-p1(2)*v2(3)-p1(3)*v2(4)
         s21 = p2(0)*v1(1)-p2(1)*v1(2)-p2(2)*v1(3)-p2(3)*v1(4)
         s22 = p2(0)*v2(1)-p2(1)*v2(2)-p2(2)*v2(3)-p2(3)*v2(4)

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     js = (v12*(-m1+m2) +s11*s12 -s21*s22)/vm2
         js = (v12*(-m1+m2) +s11*s12 -s21*s22)/cm2
        
         dg = -g/dcmplx( s-vm2, vmass*vwidth )

c  For the running width, use below instead of the above dg.
c         dg = -g/dcmplx( s-vm2, max(vwidth*s/vmass,rZero) )

         jvv(1) = dg*(j12(0)-q(0)*js)
         jvv(2) = dg*(j12(1)-q(1)*js)
         jvv(3) = dg*(j12(2)-q(2)*js)
         jvv(4) = dg*(j12(3)-q(3)*js)

      else

         gs = -g/s

         jvv(1) = gs*j12(0)
         jvv(2) = gs*j12(1)
         jvv(3) = gs*j12(2)
         jvv(4) = gs*j12(3)

      end if
c
      return
      end
      subroutine jw3wxx(w1,w2,w3,g1,g2,vmass,vwidth , jw3w)
c
c This subroutine computes an off-shell W+, W-, W3, Z or photon current
c from the four-point gauge boson coupling.  The vector propagator is
c given in Feynman gauge for a photon and in unitary gauge for W and
c Z bosons.  If one sets wmass=0.0, then the ggg-->g current is given
c (see sect 2.9.1 of the manual).
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       complex w3(6)          : third  vector                        w3
c       real    g1             : first  coupling constant
c       real    g2             : second coupling constant
c                                                  (see the table below)
c       real    wmass          : mass  of internal W
c       real    wwidth         : width of internal W
c       real    vmass          : mass  of output W'
c       real    vwidth         : width of output W'
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------------------------------
c   |  w1  |  w2  |  w3  | g1 | g2 |wmass|wwidth|vmass|vwidth || jw3w |
c   -------------------------------------------------------------------
c   |  W-  |  W3  |  W+  | gw |gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  W3  |  W+  | gw |gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   |  W-  |  Z   |  W+  |gwwz|gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  Z   |  W+  |gwwz|gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   |  W-  |  A   |  W+  |gwwa|gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  A   |  W+  |gwwa|gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   -------------------------------------------------------------------
c   |  W3  |  W-  |  W3  | gw | gw |wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  W3  | gw | gw |wmass|wwidth|wmass|wwidth ||  W-  |
c   |  W3  |  W-  |  Z   | gw |gwwz|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  Z   | gw |gwwz|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  W3  |  W-  |  A   | gw |gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  A   | gw |gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  Z   |  W-  |  Z   |gwwz|gwwz|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  Z   |  W+  |  Z   |gwwz|gwwz|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  Z   |  W-  |  A   |gwwz|gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  Z   |  W+  |  A   |gwwz|gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  A   |  W-  |  A   |gwwa|gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  A   |  W+  |  A   |gwwa|gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   -------------------------------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex jw3w(6)        : W current             j^mu(w':w1,w2,w3)
c     
      implicit none
      double complex w1(6),w2(6),w3(6),jw3w(6)
      double complex dw1(0:3),dw2(0:3),dw3(0:3)
      double complex jj(0:3),j4(0:3),dv,w12,w32,w13,jq
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision g1,g2,vmass,vwidth
      double precision p1(0:3),p2(0:3),p3(0:3),q(0:3)
      double precision dg2,dmv,dwv,mv2,q2

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

c

      jw3w(5) = w1(5)+w2(5)+w3(5)
      jw3w(6) = w1(6)+w2(6)+w3(6)

      dw1(0) = dcmplx(w1(1))
      dw1(1) = dcmplx(w1(2))
      dw1(2) = dcmplx(w1(3))
      dw1(3) = dcmplx(w1(4))
      dw2(0) = dcmplx(w2(1))
      dw2(1) = dcmplx(w2(2))
      dw2(2) = dcmplx(w2(3))
      dw2(3) = dcmplx(w2(4))
      dw3(0) = dcmplx(w3(1))
      dw3(1) = dcmplx(w3(2))
      dw3(2) = dcmplx(w3(3))
      dw3(3) = dcmplx(w3(4))
      p1(0) = dble(      w1(5))
      p1(1) = dble(      w1(6))
      p1(2) = dble(dimag(w1(6)))
      p1(3) = dble(dimag(w1(5)))
      p2(0) = dble(      w2(5))
      p2(1) = dble(      w2(6))
      p2(2) = dble(dimag(w2(6)))
      p2(3) = dble(dimag(w2(5)))
      p3(0) = dble(      w3(5))
      p3(1) = dble(      w3(6))
      p3(2) = dble(dimag(w3(6)))
      p3(3) = dble(dimag(w3(5)))
      q(0) = -(p1(0)+p2(0)+p3(0))
      q(1) = -(p1(1)+p2(1)+p3(1))
      q(2) = -(p1(2)+p2(2)+p3(2))
      q(3) = -(p1(3)+p2(3)+p3(3))

      q2 = q(0)**2 -(q(1)**2 +q(2)**2 +q(3)**2)
      dg2 = dble(g1)*dble(g2)
      dmv = dble(vmass)
      dwv = dble(vwidth)
      mv2 = dmv**2


      if ( vmass.eq.rZero ) then
         dv = rOne/dcmplx( q2 )
      else
         dv = rOne/dcmplx( q2-mv2, dmv*dwv )
      endif

c  For the running width, use below instead of the above dv.
c      dv = rOne/dcmplx( q2-mv2 , max(dwv*q2/dmv,rZero) )

      w12=dw1(0)*dw2(0)-dw1(1)*dw2(1)-dw1(2)*dw2(2)-dw1(3)*dw2(3)
      w32=dw3(0)*dw2(0)-dw3(1)*dw2(1)-dw3(2)*dw2(2)-dw3(3)*dw2(3)

      w13=dw1(0)*dw3(0)-dw1(1)*dw3(1)-dw1(2)*dw3(2)-dw1(3)*dw3(3)
      
      j4(0) = dg2*( dw1(0)*w32 + dw3(0)*w12 - rTwo*dw2(0)*w13 )
      j4(1) = dg2*( dw1(1)*w32 + dw3(1)*w12 - rTwo*dw2(1)*w13 )
      j4(2) = dg2*( dw1(2)*w32 + dw3(2)*w12 - rTwo*dw2(2)*w13 )
      j4(3) = dg2*( dw1(3)*w32 + dw3(3)*w12 - rTwo*dw2(3)*w13 )

      jj(0) = j4(0)
      jj(1) = j4(1)
      jj(2) = j4(2)
      jj(3) = j4(3)

      if ( vmass.ne.rZero ) then

c     Fabio's implementation of the fixed width
         cm2=dcmplx( mv2, -dmv*dwv)
c     jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/mv2
         jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/cm2
         
         jw3w(1) = dcmplx( (jj(0)-jq*q(0))*dv )
         jw3w(2) = dcmplx( (jj(1)-jq*q(1))*dv )
         jw3w(3) = dcmplx( (jj(2)-jq*q(2))*dv )
         jw3w(4) = dcmplx( (jj(3)-jq*q(3))*dv )

      else

         jw3w(1) = dcmplx( jj(0)*dv )
         jw3w(2) = dcmplx( jj(1)*dv )
         jw3w(3) = dcmplx( jj(2)*dv )
         jw3w(4) = dcmplx( jj(3)*dv )
      end if
c
      return
      end
      subroutine jwwwxx(w1,w2,w3,gwwa,gwwz,wmass,wwidth , jwww)
c
c This subroutine computes an off-shell W+/W- current from the four-
c point gauge boson coupling.  The vector propagators for the output
c W and the internal Z bosons are given in unitary gauge, and that of
c the internal photon is given in Feynman gauge.
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       complex w3(6)          : third  vector                        w3
c       real    gwwa           : coupling constant of W and A       gwwa
c       real    gwwz           : coupling constant of W and Z       gwwz
c       real    zmass          : mass  of internal Z
c       real    zwidth         : width of internal Z
c       real    wmass          : mass  of output W
c       real    wwidth         : width of output W
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------------------------------
c   |  w1  |  w2  |  w3  |gwwa|gwwz|zmass|zwidth|wmass|wwidth || jwww |
c   -------------------------------------------------------------------
c   |  W-  |  W+  |  W-  |gwwa|gwwz|zmass|zwidth|wmass|wwidth ||  W+  |
c   |  W+  |  W-  |  W+  |gwwa|gwwz|zmass|zwidth|wmass|wwidth ||  W-  |
c   -------------------------------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex jwww(6)        : W current             j^mu(w':w1,w2,w3)
c     
      implicit none
      double complex w1(6),w2(6),w3(6),jwww(6)
      double complex dw1(0:3),dw2(0:3),dw3(0:3),jj(0:3)
      double complex dw,w12,w32,w13,jq
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision gwwa,gwwz,wmass,wwidth
      double precision p1(0:3),p2(0:3),p3(0:3),q(0:3)
      double precision dgwwa2,dgwwz2,dgw2,dmw,dww,mw2,q2

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

c

      jwww(5) = w1(5)+w2(5)+w3(5)
      jwww(6) = w1(6)+w2(6)+w3(6)

      dw1(0) = dcmplx(w1(1))
      dw1(1) = dcmplx(w1(2))
      dw1(2) = dcmplx(w1(3))
      dw1(3) = dcmplx(w1(4))
      dw2(0) = dcmplx(w2(1))
      dw2(1) = dcmplx(w2(2))
      dw2(2) = dcmplx(w2(3))
      dw2(3) = dcmplx(w2(4))
      dw3(0) = dcmplx(w3(1))
      dw3(1) = dcmplx(w3(2))
      dw3(2) = dcmplx(w3(3))
      dw3(3) = dcmplx(w3(4))
      p1(0) = dble(      w1(5))
      p1(1) = dble(      w1(6))
      p1(2) = dble(dimag(w1(6)))
      p1(3) = dble(dimag(w1(5)))
      p2(0) = dble(      w2(5))
      p2(1) = dble(      w2(6))
      p2(2) = dble(dimag(w2(6)))
      p2(3) = dble(dimag(w2(5)))
      p3(0) = dble(      w3(5))
      p3(1) = dble(      w3(6))
      p3(2) = dble(dimag(w3(6)))
      p3(3) = dble(dimag(w3(5)))
      q(0) = -(p1(0)+p2(0)+p3(0))
      q(1) = -(p1(1)+p2(1)+p3(1))
      q(2) = -(p1(2)+p2(2)+p3(2))
      q(3) = -(p1(3)+p2(3)+p3(3))
      q2 = q(0)**2 -(q(1)**2 +q(2)**2 +q(3)**2)
      dgwwa2 = dble(gwwa)**2
      dgwwz2 = dble(gwwz)**2
      dgw2 = dgwwa2+dgwwz2
      dmw = dble(wmass)
      dww = dble(wwidth)
      mw2 = dmw**2


      dw = -rOne/dcmplx( q2-mw2, dmw*dww )
c  For the running width, use below instead of the above dw.
c      dw = -rOne/dcmplx( q2-mw2 , max(dww*q2/dmw,rZero) )

      w12=dw1(0)*dw2(0)-dw1(1)*dw2(1)-dw1(2)*dw2(2)-dw1(3)*dw2(3)
      w32=dw3(0)*dw2(0)-dw3(1)*dw2(1)-dw3(2)*dw2(2)-dw3(3)*dw2(3)

      w13=dw1(0)*dw3(0)-dw1(1)*dw3(1)-dw1(2)*dw3(2)-dw1(3)*dw3(3)

      jj(0) = dgw2*( dw1(0)*w32 + dw3(0)*w12 - rTwo*dw2(0)*w13 )
      jj(1) = dgw2*( dw1(1)*w32 + dw3(1)*w12 - rTwo*dw2(1)*w13 )
      jj(2) = dgw2*( dw1(2)*w32 + dw3(2)*w12 - rTwo*dw2(2)*w13 )
      jj(3) = dgw2*( dw1(3)*w32 + dw3(3)*w12 - rTwo*dw2(3)*w13 )

c     Fabio's implementation of the fixed width
      cm2=dcmplx( mw2, -dmw*dww )
c     jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/mw2
      jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/cm2

      jwww(1) = dcmplx( (jj(0)-jq*q(0))*dw )
      jwww(2) = dcmplx( (jj(1)-jq*q(1))*dw )
      jwww(3) = dcmplx( (jj(2)-jq*q(2))*dw )
      jwww(4) = dcmplx( (jj(3)-jq*q(3))*dw )
c
      return
      end
      subroutine mom2cx(esum,mass1,mass2,costh1,phi1 , p1,p2)
c
c This subroutine sets up two four-momenta in the two particle rest
c frame.
c
c input:
c       real    esum           : energy sum of particle 1 and 2
c       real    mass1          : mass            of particle 1
c       real    mass2          : mass            of particle 2
c       real    costh1         : cos(theta)      of particle 1
c       real    phi1           : azimuthal angle of particle 1
c
c output:
c       real    p1(0:3)        : four-momentum of particle 1
c       real    p2(0:3)        : four-momentum of particle 2
c     
      implicit none
      double precision p1(0:3),p2(0:3),
     &     esum,mass1,mass2,costh1,phi1,md2,ed,pp,sinth1

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )

c

      md2 = (mass1-mass2)*(mass1+mass2)
      ed = md2/esum
      if ( mass1*mass2.eq.rZero ) then
         pp = (esum-abs(ed))*rHalf
      else
         pp = sqrt((md2/esum)**2-rTwo*(mass1**2+mass2**2)+esum**2)*rHalf
      endif
      sinth1 = sqrt((rOne-costh1)*(rOne+costh1))

      p1(0) = max((esum+ed)*rHalf,rZero)
      p1(1) = pp*sinth1*cos(phi1)
      p1(2) = pp*sinth1*sin(phi1)
      p1(3) = pp*costh1

      p2(0) = max((esum-ed)*rHalf,rZero)
      p2(1) = -p1(1)
      p2(2) = -p1(2)
      p2(3) = -p1(3)
c
      return
      end
      subroutine momntx(energy,mass,costh,phi , p)
c
c This subroutine sets up a four-momentum from the four inputs.
c
c input:
c       real    energy         : energy
c       real    mass           : mass
c       real    costh          : cos(theta)
c       real    phi            : azimuthal angle
c
c output:
c       real    p(0:3)         : four-momentum
c     
      implicit none
      double precision p(0:3),energy,mass,costh,phi,pp,sinth

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

c

      p(0) = energy

      if ( energy.eq.mass ) then

         p(1) = rZero
         p(2) = rZero
         p(3) = rZero

      else

         pp = sqrt((energy-mass)*(energy+mass))
         sinth = sqrt((rOne-costh)*(rOne+costh))
         p(3) = pp*costh
         if ( phi.eq.rZero ) then
            p(1) = pp*sinth
            p(2) = rZero
         else
            p(1) = pp*sinth*cos(phi)
            p(2) = pp*sinth*sin(phi)
         endif

      endif
c
      return
      end
      subroutine oxxxxx(p,fmass,nhel,nsf , fo)
c
c This subroutine computes a fermion wavefunction with the flowing-OUT
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(6)          : fermion wavefunction               <fo|
c     
      implicit none
      double complex fo(6),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm
      integer nhel,nsf,nh,ip,im

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )

c

      fo(5) = dcmplx(p(0),p(3))*nsf
      fo(6) = dcmplx(p(1),p(2))*nsf

      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then
            
            sqm = dsqrt(fmass)
            ip = -((1+nh)/2)
            im =  (1-nh)/2
            
            fo(1) = im     * sqm
            fo(2) = ip*nsf * sqm
            fo(3) = im*nsf * sqm
            fo(4) = ip     * sqm
            
         else
            
            pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , -p(2) )/dsqrt(rTwo*pp*pp3)
            endif
            
            fo(1) = sfomeg(2)*chi(im)
            fo(2) = sfomeg(2)*chi(ip)
            fo(3) = sfomeg(1)*chi(im)
            fo(4) = sfomeg(1)*chi(ip)

         endif
         
      else
         
         sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), -p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fo(1) = chi(1)
            fo(2) = chi(2)
            fo(3) = dcmplx( rZero )
            fo(4) = dcmplx( rZero )
         else
            fo(1) = dcmplx( rZero )
            fo(2) = dcmplx( rZero )
            fo(3) = chi(2)
            fo(4) = chi(1)
         endif
         
      endif
c
      return
      end
      subroutine rotxxx(p,q , prot)
c
c This subroutine performs the spacial rotation of a four-momentum.
c the momentum p is assumed to be given in the frame where the spacial
c component of q points the positive z-axis.  prot is the momentum p
c rotated to the frame where q is given.
c
c input:
c       real    p(0:3)         : four-momentum p in q(1)=q(2)=0 frame
c       real    q(0:3)         : four-momentum q in the rotated frame
c
c output:
c       real    prot(0:3)      : four-momentum p in the rotated frame
c     
      implicit none
      double precision p(0:3),q(0:3),prot(0:3),qt2,qt,psgn,qq,p1

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

c
      prot(0) = p(0)

      qt2 = q(1)**2 + q(2)**2


      if ( qt2.eq.rZero ) then
          if ( q(3).eq.rZero ) then
             prot(1) = p(1)
             prot(2) = p(2)
             prot(3) = p(3)
          else
             psgn = dsign(rOne,q(3))
             prot(1) = p(1)*psgn
             prot(2) = p(2)*psgn
             prot(3) = p(3)*psgn
          endif
      else
          qq = sqrt(qt2+q(3)**2)
          qt = sqrt(qt2)
          p1 = p(1)
          prot(1) = q(1)*q(3)/qq/qt*p1 -q(2)/qt*p(2) +q(1)/qq*p(3)
          prot(2) = q(2)*q(3)/qq/qt*p1 +q(1)/qt*p(2) +q(2)/qq*p(3)
          prot(3) =          -qt/qq*p1               +q(3)/qq*p(3)
      endif
c
      return
      end
      subroutine ssssxx(s1,s2,s3,s4,gc , vertex)
c
c This subroutine computes an amplitude of the four-scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex s3(3)          : third  scalar                        s3
c       complex s4(3)          : fourth scalar                        s4
c       complex gc             : coupling constant                 ghhhh
c
c output:
c       complex vertex         : amplitude            gamma(s1,s2,s3,s4)
c     
      implicit none
      double complex s1(3),s2(3),s3(3),s4(3),gc,vertex

c

      vertex = gc*s1(1)*s2(1)*s3(1)*s4(1)
c
      return
      end
      subroutine sssxxx(s1,s2,s3,gc , vertex)
c
c This subroutine computes an amplitude of the three-scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex s3(3)          : third  scalar                        s3
c       complex gc             : coupling constant                  ghhh
c
c output:
c       complex vertex         : amplitude               gamma(s1,s2,s3)
c     
      implicit none
      double complex s1(3),s2(3),s3(3),gc,vertex

c

      vertex = gc*s1(1)*s2(1)*s3(1)
c
      return
      end
      subroutine sxxxxx(p,nss , sc)
c
c This subroutine computes a complex SCALAR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc(3)          : scalar wavefunction                   s
c     
      implicit none
      double complex sc(3)
      double precision p(0:3)
      integer nss

      double precision rOne
      parameter( rOne = 1.0d0 )

c

      sc(1) = dcmplx( rOne )
      sc(2) = dcmplx(p(0),p(3))*nss
      sc(3) = dcmplx(p(1),p(2))*nss
c
      return
      end
      subroutine txxxx2(p,tmass,nhel,nst , tc)
c
c This subroutine computes k^mu*e^nu where e is delta(i,nhel).
c It is used to test for gauge invariance of the tensor routines.
c
c input:
c       real    p(0:3)             : four-momentum of tensor boson
c       real    tmass              : mass          of tensor boson
c       integer nhel = 1..4        : construction of e^nu
c       integer nst  = -1 or 1     : +1 for final, -1 for initial
c
c output:
c       complex tc(6,4)            : tensor wavefunction       epsilon^mu^nu(t)
c     
      implicit none
      double complex tc(6,4), ep(4), em(4)
      double precision p(0:3), tmass
      integer nhel, nst, i, j

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
c
      tc(5,1) = dcmplx(p(0),p(3))*nst
      tc(6,1) = dcmplx(p(1),p(2))*nst

      ep(1) = dcmplx( p(0), rZero )
      ep(2) = dcmplx( p(1), rZero )
      ep(3) = dcmplx( p(2), rZero )
      ep(4) = dcmplx( p(3), rZero )

      if ( nhel.eq.1 ) then
         em(1) = dcmplx( rOne , rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.2 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rOne , rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.3 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rOne , rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.4 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rOne , rZero )
      end if

      do j = 1,4
         do i = 1,4
            tc(i,j) = ep(i)*em(j)
         end do
      end do
c
      return
      end
      subroutine txxxxx(p,tmass,nhel,nst , tc)
c
c This subroutine computes a TENSOR wavefunction.
c
c input:
c       real    p(0:3)             : four-momentum of tensor boson
c       real    tmass              : mass          of tensor boson
c       integer nhel = -2,-1,0,1,2 : helicity      of tensor boson
c                                    (0 is forbidden if tmass = 0)
c       integer nst  = -1 or 1     : +1 for final, -1 for initial
c
c output:
c       complex tc(6,4)            : tensor wavefunction   epsilon^mu^nu(t)
c     
      implicit none
      double complex tc(6,4), ep(4), em(4), e0(4)
      double precision p(0:3), tmass, pt, pt2, pp, pzpt, emp, sqh, sqs
      integer nhel, nst, i, j

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )
c
      sqh = sqrt(rHalf)
      sqs = sqrt(rHalf/3.d0)

      pt2 = p(1)**2 + p(2)**2
      pp = min(p(0),sqrt(pt2+p(3)**2))
      pt = min(pp,sqrt(pt2))

      tc(5,1) = dcmplx(p(0),p(3))*nst
      tc(6,1) = dcmplx(p(1),p(2))*nst

      if ( nhel.ge.0 ) then  !construct eps+
         if ( pp.eq.rZero ) then
            ep(1) = dcmplx( rZero )
            ep(2) = dcmplx( -sqh )
            ep(3) = dcmplx( rZero , nst*sqh )
            ep(4) = dcmplx( rZero )
         else
            ep(1) = dcmplx( rZero )
            ep(4) = dcmplx( pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh
               ep(2) = dcmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh )
               ep(3) = dcmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh )
            else
               ep(2) = dcmplx( -sqh )
               ep(3) = dcmplx( rZero , nst*sign(sqh,p(3)) )
            endif
         endif
      end if

      if ( nhel.le.0 ) then  !construct eps-
         if ( pp.eq.rZero ) then
            em(1) = dcmplx( rZero )
            em(2) = dcmplx( sqh )
            em(3) = dcmplx( rZero , nst*sqh )
            em(4) = dcmplx( rZero )
         else
            em(1) = dcmplx( rZero )
            em(4) = dcmplx( -pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = -p(3)/(pp*pt)*sqh
               em(2) = dcmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh )
               em(3) = dcmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh )
            else
               em(2) = dcmplx( sqh )
               em(3) = dcmplx( rZero , nst*sign(sqh,p(3)) )
            endif
         endif
      end if

      if ( abs(nhel).le.1 ) then  !construct eps0
         if ( pp.eq.rZero ) then
            e0(1) = dcmplx( rZero )
            e0(2) = dcmplx( rZero )
            e0(3) = dcmplx( rZero )
            e0(4) = dcmplx( rOne )
         else
            emp = p(0)/(tmass*pp)
            e0(1) = dcmplx( pp/tmass )
            e0(4) = dcmplx( p(3)*emp )
            if ( pt.ne.rZero ) then
               e0(2) = dcmplx( p(1)*emp )
               e0(3) = dcmplx( p(2)*emp )
            else
               e0(2) = dcmplx( rZero )
               e0(3) = dcmplx( rZero )
            endif
         end if
      end if

      if ( nhel.eq.2 ) then
         do j = 1,4
            do i = 1,4
               tc(i,j) = ep(i)*ep(j)
            end do
         end do
      else if ( nhel.eq.1 ) then
         do j = 1,4
            do i = 1,4
               tc(i,j) = sqh*( ep(i)*e0(j) + e0(i)*ep(j) )
            end do
         end do
      else if ( nhel.eq.0 ) then
         do j = 1,4
            do i = 1,4
               tc(i,j) = sqs*( ep(i)*em(j) + em(i)*ep(j)
     &                                + rTwo*e0(i)*e0(j) )
            end do
         end do
      else if ( nhel.eq.-1 ) then
         do j = 1,4
            do i = 1,4
               tc(i,j) = sqh*( em(i)*e0(j) + e0(i)*em(j) )
            end do
         end do
      else if ( nhel.eq.-2 ) then
         do j = 1,4
            do i = 1,4
               tc(i,j) = em(i)*em(j)
            end do
         end do
      else
         write(6,*) 'invalid helicity in TXXXXX'
         stop
      end if
c
      return
      end
      subroutine vssxxx(vc,s1,s2,gc , vertex)
c
c This subroutine computes an amplitude from the vector-scalar-scalar
c coupling.  The coupling is absent in the minimal SM in unitary gauge.
c
c       complex vc(6)          : input  vector                        v
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant (s1 charge)
c
c examples of the coupling constant gc for SUSY particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=a   |   v=z   |   v=w   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the s1 charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex vertex         : amplitude                gamma(v,s1,s2)
c     
      implicit none
      double complex vc(6),s1(3),s2(3),gc,vertex
      double precision p(0:3)

c

      p(0) = dble( s1(2)-s2(2))
      p(1) = dble( s1(3)-s2(3))
      p(2) = dimag(s1(3)-s2(3))
      p(3) = dimag(s1(2)-s2(2))

      vertex = gc*s1(1)*s2(1)
     &        *(vc(1)*p(0)-vc(2)*p(1)-vc(3)*p(2)-vc(4)*p(3))
c
      return
      end
      subroutine vvssxx(v1,v2,s1,s2,gc , vertex)
c
c This subroutine computes an amplitude of the vector-vector-scalar-
c scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                 gvvhh
c
c output:
c       complex vertex         : amplitude            gamma(v1,v2,s1,s2)
c     
      implicit none
      double complex v1(6),v2(6),s1(3),s2(3),gc,vertex

c

      vertex = gc*s1(1)*s2(1)
     &        *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
      subroutine vvsxxx(v1,v2,sc,gc , vertex)
c
c This subroutine computes an amplitude of the vector-vector-scalar
c coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex sc(3)          : input  scalar                        s
c       complex gc             : coupling constant                  gvvh
c
c output:
c       complex vertex         : amplitude                gamma(v1,v2,s)
c     
      implicit none
      double complex v1(6),v2(6),sc(3),gc,vertex

c

      vertex = gc*sc(1)
     &        *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
      subroutine vvtxkk(wm,wp,tc,g,vmass , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a Kaluza-Klein tensor boson.
c
c input:
c       complex wm(6)          : vector               flow-in  V
c       complex wp(6)          : vector               flow-out V~
c       complex tc(6,4)        : tensor               KK mode T
c       real    g              : coupling constant    -kappa/2
c       real    vmass          : V boson mass          m_V
c
c output:
c       complex vertex         : amplitude            gamma(wm,wp,tc)
c     
      implicit none
      double complex wm(6), wp(6), tc(6,4), vertex
      double precision g, vmass

      double complex T12, T13, T14, T23, T24, T34
      double complex V1V2, k1V2, k2V1
      double complex Tkk, TVV, Tk1V2, Tk2V1, dum
      double precision pwm(4), pwp(4), F

      integer i, j

      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
c
      pwm(1) = dreal(wm(5))
      pwm(2) = dreal(wm(6))
      pwm(3) = dimag(wm(6))
      pwm(4) = dimag(wm(5))
      pwp(1) = dreal(wp(5))
      pwp(2) = dreal(wp(6))
      pwp(3) = dimag(wp(6))
      pwp(4) = dimag(wp(5))

      T12 = tc(1,2) + tc(2,1)
      T13 = tc(1,3) + tc(3,1)
      T14 = tc(1,4) + tc(4,1)
      T23 = tc(2,3) + tc(3,2)
      T24 = tc(2,4) + tc(4,2)
      T34 = tc(3,4) + tc(4,3)

      V1V2 =  wm(1)*wp(1) -  wm(2)*wp(2) -  wm(3)*wp(3) -  wm(4)*wp(4)
      k1V2 = pwm(1)*wp(1) - pwm(2)*wp(2) - pwm(3)*wp(3) - pwm(4)*wp(4)
      k2V1 = pwp(1)*wm(1) - pwp(2)*wm(2) - pwp(3)*wm(3) - pwp(4)*wm(4)

      F = pwm(1)*pwp(1) - pwm(2)*pwp(2) - pwm(3)*pwp(3) - pwm(4)*pwp(4)
      if ( vmass.ne.rZero ) then
         F = F + vmass**2
      end if

      Tkk   = cZero
      TVV   = cZero
      Tk1V2 = cZero
      Tk2V1 = cZero

      do i = 1,4
         dum   = tc(i,i)*pwm(i)
         Tkk   = Tkk   + dum*pwp(i)
         Tk1V2 = Tk1V2 + dum*wp(i)
         dum   = tc(i,i)*wm(i)
         TVV   = TVV   + dum*wp(i)
         Tk2V1 = Tk2V1 + dum*pwp(i)
      end do

      Tkk   = rTwo*Tkk
      TVV   = rTwo*TVV
      Tk1V2 = rTwo*Tk1V2
      Tk2V1 = rTwo*Tk2V1

      Tkk = Tkk - T12*(pwm(1)*pwp(2) + pwm(2)*pwp(1))
     &          - T13*(pwm(1)*pwp(3) + pwm(3)*pwp(1))
     &          - T14*(pwm(1)*pwp(4) + pwm(4)*pwp(1))
     &          + T23*(pwm(2)*pwp(3) + pwm(3)*pwp(2))
     &          + T24*(pwm(2)*pwp(4) + pwm(4)*pwp(2))
     &          + T34*(pwm(3)*pwp(4) + pwm(4)*pwp(3))

      Tk1V2 = Tk1V2 - T12*(pwm(1)*wp(2) + pwm(2)*wp(1))
     &              - T13*(pwm(1)*wp(3) + pwm(3)*wp(1))
     &              - T14*(pwm(1)*wp(4) + pwm(4)*wp(1))
     &              + T23*(pwm(2)*wp(3) + pwm(3)*wp(2))
     &              + T24*(pwm(2)*wp(4) + pwm(4)*wp(2))
     &              + T34*(pwm(3)*wp(4) + pwm(4)*wp(3))

      TVV = TVV - T12*(wm(1)*wp(2) + wm(2)*wp(1))
     &          - T13*(wm(1)*wp(3) + wm(3)*wp(1))
     &          - T14*(wm(1)*wp(4) + wm(4)*wp(1))
     &          + T23*(wm(2)*wp(3) + wm(3)*wp(2))
     &          + T24*(wm(2)*wp(4) + wm(4)*wp(2))
     &          + T34*(wm(3)*wp(4) + wm(4)*wp(3))

      Tk2V1 = Tk2V1 - T12*(wm(1)*pwp(2) + wm(2)*pwp(1))
     &              - T13*(wm(1)*pwp(3) + wm(3)*pwp(1))
     &              - T14*(wm(1)*pwp(4) + wm(4)*pwp(1))
     &              + T23*(wm(2)*pwp(3) + wm(3)*pwp(2))
     &              + T24*(wm(2)*pwp(4) + wm(4)*pwp(2))
     &              + T34*(wm(3)*pwp(4) + wm(4)*pwp(3))

      vertex =  (tc(1,1)-tc(2,2)-tc(3,3)-tc(4,4))*( k1V2*k2V1 - V1V2*F )
     &        + F*TVV + V1V2*Tkk - k2V1*Tk1V2 - k1V2*Tk2V1

C      vertex = F*TVV + V1V2*Tkk - k2V1*Tk1V2 - k1V2*Tk2V1

      vertex = vertex * g
c
      return
      end
      subroutine vvvxxx(wm,wp,w3,g , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c the gauge bosons.
c
c input:
c       complex wm(6)          : vector               flow-out W-
c       complex wp(6)          : vector               flow-out W+
c       complex w3(6)          : vector               j3 or A    or Z
c       real    g              : coupling constant    gw or gwwa or gwwz
c
c output:
c       complex vertex         : amplitude               gamma(wm,wp,w3)
c     
      implicit none
      double complex wm(6),wp(6),w3(6),vertex,
     &     xv1,xv2,xv3,v12,v23,v31,p12,p13,p21,p23,p31,p32
      double precision pwm(0:3),pwp(0:3),pw3(0:3),g

      double precision rZero, rTenth
      parameter( rZero = 0.0d0, rTenth = 0.1d0 )

c
      pwm(0) = dble( wm(5))
      pwm(1) = dble( wm(6))
      pwm(2) = dimag(wm(6))
      pwm(3) = dimag(wm(5))
      pwp(0) = dble( wp(5))
      pwp(1) = dble( wp(6))
      pwp(2) = dimag(wp(6))
      pwp(3) = dimag(wp(5))
      pw3(0) = dble( w3(5))
      pw3(1) = dble( w3(6))
      pw3(2) = dimag(w3(6))
      pw3(3) = dimag(w3(5))


      v12 = wm(1)*wp(1)-wm(2)*wp(2)-wm(3)*wp(3)-wm(4)*wp(4)
      v23 = wp(1)*w3(1)-wp(2)*w3(2)-wp(3)*w3(3)-wp(4)*w3(4)
      v31 = w3(1)*wm(1)-w3(2)*wm(2)-w3(3)*wm(3)-w3(4)*wm(4)
      xv1 = rZero
      xv2 = rZero
      xv3 = rZero

      if ( abs(wm(1)).ne.rZero ) then
         if ( abs(wm(1)).ge.max(abs(wm(2)),abs(wm(3)),abs(wm(4)))
     &        *rTenth )
     &      xv1 = pwm(0)/wm(1)
      endif
      if ( abs(wp(1)).ne.rZero) then
         if ( abs(wp(1)).ge.max(abs(wp(2)),abs(wp(3)),abs(wp(4)))
     &        *rTenth )
     &      xv2 = pwp(0)/wp(1)
      endif
      if ( abs(w3(1)).ne.rZero) then
         if ( abs(w3(1)).ge.max(abs(w3(2)),abs(w3(3)),abs(w3(4)))
     &        *rTenth )
     &      xv3 = pw3(0)/w3(1)
      endif

      p12 = (pwm(0)-xv1*wm(1))*wp(1)-(pwm(1)-xv1*wm(2))*wp(2)
     &     -(pwm(2)-xv1*wm(3))*wp(3)-(pwm(3)-xv1*wm(4))*wp(4)
      p13 = (pwm(0)-xv1*wm(1))*w3(1)-(pwm(1)-xv1*wm(2))*w3(2)
     &     -(pwm(2)-xv1*wm(3))*w3(3)-(pwm(3)-xv1*wm(4))*w3(4)
      p21 = (pwp(0)-xv2*wp(1))*wm(1)-(pwp(1)-xv2*wp(2))*wm(2)
     &     -(pwp(2)-xv2*wp(3))*wm(3)-(pwp(3)-xv2*wp(4))*wm(4)
      p23 = (pwp(0)-xv2*wp(1))*w3(1)-(pwp(1)-xv2*wp(2))*w3(2)
     &     -(pwp(2)-xv2*wp(3))*w3(3)-(pwp(3)-xv2*wp(4))*w3(4)
      p31 = (pw3(0)-xv3*w3(1))*wm(1)-(pw3(1)-xv3*w3(2))*wm(2)
     &     -(pw3(2)-xv3*w3(3))*wm(3)-(pw3(3)-xv3*w3(4))*wm(4)
      p32 = (pw3(0)-xv3*w3(1))*wp(1)-(pw3(1)-xv3*w3(2))*wp(2)
     &     -(pw3(2)-xv3*w3(3))*wp(3)-(pw3(3)-xv3*w3(4))*wp(4)

      vertex = -(v12*(p13-p23)+v23*(p21-p31)+v31*(p32-p12))*g
c
      return
      end
      subroutine vxxxxx(p,vmass,nhel,nsv , vc)
c
c This subroutine computes a VECTOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of vector boson
c       real    vmass          : mass          of vector boson
c       integer nhel = -1, 0, 1: helicity      of vector boson
c                                (0 is forbidden if vmass=0.0)
c       integer nsv  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex vc(6)          : vector wavefunction       epsilon^mu(v)
c     
      implicit none
      double complex vc(6)
      double precision p(0:3),vmass,hel,hel0,pt,pt2,pp,pzpt,emp,sqh
      integer nhel,nsv,nsvahl

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )
      
c

      sqh = dsqrt(rHalf)
      hel = dble(nhel)
      nsvahl = nsv*dabs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = min(p(0),dsqrt(pt2+p(3)**2))
      pt = min(pp,dsqrt(pt2))

      vc(5) = dcmplx(p(0),p(3))*nsv
      vc(6) = dcmplx(p(1),p(2))*nsv


      if ( vmass.ne.rZero ) then

         hel0 = rOne-dabs(hel)

         if ( pp.eq.rZero ) then

            vc(1) = dcmplx( rZero )
            vc(2) = dcmplx(-hel*sqh )
            vc(3) = dcmplx( rZero , nsvahl*sqh )
            vc(4) = dcmplx( hel0 )

         else

            emp = p(0)/(vmass*pp)
            vc(1) = dcmplx( hel0*pp/vmass )
            vc(4) = dcmplx( hel0*p(3)*emp+hel*pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc(2) = dcmplx( hel0*p(1)*emp-p(1)*pzpt , 
     &                         -nsvahl*p(2)/pt*sqh       )
               vc(3) = dcmplx( hel0*p(2)*emp-p(2)*pzpt ,  
     &                          nsvahl*p(1)/pt*sqh       )
            else
               vc(2) = dcmplx( -hel*sqh )
               vc(3) = dcmplx( rZero , nsvahl*sign(sqh,p(3)) )
            endif

         endif

      else

         pp = p(0)
         pt = sqrt(p(1)**2+p(2)**2)
         vc(1) = dcmplx( rZero )
         vc(4) = dcmplx( hel*pt/pp*sqh )
         if ( pt.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc(2) = dcmplx( -p(1)*pzpt , -nsv*p(2)/pt*sqh )
            vc(3) = dcmplx( -p(2)*pzpt ,  nsv*p(1)/pt*sqh )
         else
            vc(2) = dcmplx( -hel*sqh )
            vc(3) = dcmplx( rZero , nsv*sign(sqh,p(3)) )
         endif

      endif
c
      return
      end
      subroutine w3w3xx(wm,w31,wp,w32,g31,g32, vertex)
c
c This subroutine computes an amplitude of the four-point coupling of
c the W-, W+ and two W3/Z/A.
c If one sets wmass=0.0, then the gggg vertex is given
c (see sect 2.9.1 of the manual).
c
c input:
c       complex wm(0:3)        : flow-out W-                         wm
c       complex w31(0:3)       : first    W3/Z/A                     w31
c       complex wp(0:3)        : flow-out W+                         wp
c       complex w32(0:3)       : second   W3/Z/A                     w32
c       real    g31            : coupling of w31 with W-/W+
c       real    g32            : coupling of w32 with W-/W+
c                                                  (see the table below)
c       real    wmass          : mass  of W
c       real    wwidth         : width of W
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------
c   |  wm  |  w31 |  wp  |  w32 |  g31 |  g32 |
c   -------------------------------------------
c   |  W-  |  W3  |  W+  |  W3  |  gw  |  gw  |
c   |  W-  |  W3  |  W+  |  Z   |  gw  | gwwz |
c   |  W-  |  W3  |  W+  |  A   |  gw  | gwwa |
c   |  W-  |  Z   |  W+  |  Z   | gwwz | gwwz |
c   |  W-  |  Z   |  W+  |  A   | gwwz | gwwa |
c   |  W-  |  A   |  W+  |  A   | gwwa | gwwa |
c   -------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex vertex         : amplitude          gamma(wm,w31,wp,w32)
c     
      implicit none
      double complex wm(6),w31(6),wp(6),w32(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),dvertx
      double complex v12,v13,v14,v23,v24,v34
      double precision pwm(0:3),pw31(0:3),pwp(0:3),pw32(0:3)
      double precision g31,g32

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )


      pwm(0) = dble( wm(5))
      pwm(1) = dble( wm(6))
      pwm(2) = dimag(wm(6))
      pwm(3) = dimag(wm(5))
      pwp(0) = dble( wp(5))
      pwp(1) = dble( wp(6))
      pwp(2) = dimag(wp(6))
      pwp(3) = dimag(wp(5))
      pw31(0) = dble( w31(5))
      pw31(1) = dble( w31(6))
      pw31(2) = dimag(w31(6))
      pw31(3) = dimag(w31(5))
      pw32(0) = dble( w32(5))
      pw32(1) = dble( w32(6))
      pw32(2) = dimag(w32(6))
      pw32(3) = dimag(w32(5))


      dv1(0) = dcmplx(wm(1))
      dv1(1) = dcmplx(wm(2))
      dv1(2) = dcmplx(wm(3))
      dv1(3) = dcmplx(wm(4))
      dv2(0) = dcmplx(w31(1))
      dv2(1) = dcmplx(w31(2))
      dv2(2) = dcmplx(w31(3))
      dv2(3) = dcmplx(w31(4))
      dv3(0) = dcmplx(wp(1))
      dv3(1) = dcmplx(wp(2))
      dv3(2) = dcmplx(wp(3))
      dv3(3) = dcmplx(wp(4))
      dv4(0) = dcmplx(w32(1))
      dv4(1) = dcmplx(w32(2))
      dv4(2) = dcmplx(w32(3))
      dv4(3) = dcmplx(w32(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

      dvertx = v12*v34 + v14*v23 - rTwo*v13*v24
      
      vertex = dcmplx( dvertx ) * (g31*g32)
c
      return
      end
      subroutine wwwwxx(wm1,wp1,wm2,wp2,gwwa,gwwz , vertex)
c
c This subroutine computes an amplitude of the four-point W-/W+ coupling.
c
c input:
c       complex wm1(0:3)       : first  flow-out W-                  wm1
c       complex wp1(0:3)       : first  flow-out W+                  wp1
c       complex wm2(0:3)       : second flow-out W-                  wm2
c       complex wp2(0:3)       : second flow-out W+                  wp2
c       real    gwwa           : coupling constant of W and A       gwwa
c       real    gwwz           : coupling constant of W and Z       gwwz
c       real    zmass          : mass  of Z
c       real    zwidth         : width of Z
c
c output:
c       complex vertex         : amplitude        gamma(wm1,wp1,wm2,wp2)
c     
      implicit none
      double complex wm1(6),wp1(6),wm2(6),wp2(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),dvertx
      double complex v12,v13,v14,v23,v24,v34
      double precision pwm1(0:3),pwp1(0:3),pwm2(0:3),pwp2(0:3)
      double precision gwwa,gwwz

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

c
      pwm1(0) = dble( wm1(5))
      pwm1(1) = dble( wm1(6))
      pwm1(2) = dimag(wm1(6))
      pwm1(3) = dimag(wm1(5))
      pwp1(0) = dble( wp1(5))
      pwp1(1) = dble( wp1(6))
      pwp1(2) = dimag(wp1(6))
      pwp1(3) = dimag(wp1(5))
      pwm2(0) = dble( wm2(5))
      pwm2(1) = dble( wm2(6))
      pwm2(2) = dimag(wm2(6))
      pwm2(3) = dimag(wm2(5))
      pwp2(0) = dble( wp2(5))
      pwp2(1) = dble( wp2(6))
      pwp2(2) = dimag(wp2(6))
      pwp2(3) = dimag(wp2(5))


      dv1(0) = dcmplx(wm1(1))
      dv1(1) = dcmplx(wm1(2))
      dv1(2) = dcmplx(wm1(3))
      dv1(3) = dcmplx(wm1(4))
      dv2(0) = dcmplx(wp1(1))
      dv2(1) = dcmplx(wp1(2))
      dv2(2) = dcmplx(wp1(3))
      dv2(3) = dcmplx(wp1(4))
      dv3(0) = dcmplx(wm2(1))
      dv3(1) = dcmplx(wm2(2))
      dv3(2) = dcmplx(wm2(3))
      dv3(3) = dcmplx(wm2(4))
      dv4(0) = dcmplx(wp2(1))
      dv4(1) = dcmplx(wp2(2))
      dv4(2) = dcmplx(wp2(3))
      dv4(3) = dcmplx(wp2(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

      dvertx = (v12*v34 + v14*v23 - rTwo*v13*v24)*(gwwa**2+gwwz**2)

      vertex = -dcmplx( dvertx )
c
      return
      end
