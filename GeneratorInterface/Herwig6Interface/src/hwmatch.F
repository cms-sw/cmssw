C MLM-style matching for aMC@NLO showering in Herwig6
C Original code from R.Frederix and S.Frixione
C Adapted for CMSSW interface by J.Bendavid
C----------------------------------------------------------------------
      SUBROUTINE HWDUMMY
C     Force compiler to generate code for hwmatch
      INTEGER DUMINT
      call HWMATCH(DUMINT)
      RETURN
C----------------------------------------------------------------------
      END
C----------------------------------------------------------------------
      SUBROUTINE HWHDECAY
c
      INCLUDE 'HERWIG65.INC'
c
      INTEGER NMAX,IHEP,IST,ID1
      PARAMETER (NMAX=2000)
c
      DO 200 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(IST.GE.120 .or. IST.LE.125) THEN
          IF(ID1.EQ.25)THEN
	    ISTHEP(IHEP) = 1
	    CALL HWDHIG(ZERO)
	  ENDIF
	ENDIF
200   CONTINUE
c
      RETURN
c
      END
C----------------------------------------------------------------------
      SUBROUTINE HWMATCH(PASS)
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
c Variables specific to matching
      INTEGER PASS
      INTEGER NMAX
      PARAMETER (NMAX=2000)
      INTEGER NS,IHEP,IST,ID1,IM1,IM2,I,n_match,max_multiplicity_flag
      DOUBLE PRECISION PS(4,NMAX),matching_scale
      LOGICAL max_multiplicity,matched
c Other variables, used here for checks
      INTEGER IFH,IJ,J
      DOUBLE PRECISION PPH(5),PSUM(2)
c
      common/HWMATCHPRAM/n_match,max_multiplicity_flag,matching_scale
c
      if (MAX_MULTIPLICITY_FLAG.GT.0) then
	MAX_MULTIPLICITY = .true.
      else 
	MAX_MULTIPLICITY = .false.
      endif
c
      IF (IERROR.NE.0) RETURN
c
      NS=0
      IFH=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IM1=JMOHEP(1,IHEP)
        IM2=JMOHEP(2,IHEP)
        IF ((IST.EQ.2) .AND. (ABS(ID1).LE.5 .or.
     &       ID1.EQ.21 .or.ID1.eq.22) .AND. 
     &	     IDHEP(IM1).NE.25 .AND. IDHEP(IM1).NE.23 .AND.
     &       IDHEP(IM1).NE.24 .AND. IDHEP(IM1).NE.-24) THEN
          NS=NS+1
          IF (NS.GT.NMAX) STOP 'Too many particles!'
          DO I=1,4
             PS(I,NS)=PHEP(I,IHEP)
          ENDDO
        ENDIF
        IF(IST.EQ.195)THEN
          IF(ID1.EQ.25)THEN
            IFH=IFH+1
            DO IJ=1,5
	      PPH(IJ)=PHEP(IJ,IHEP)
	    ENDDO
          ENDIF
        ENDIF
  100 CONTINUE
c Specific for Higgs production, safety check: there must be only
c one Higgs. Stops otherwise
      IF(IFH.NE.1.AND.IERROR.EQ.0)THEN
c         CALL HWUEPR
         CALL HWWARN('HWMATCH',501)
      ENDIF
c Must not include in PS anything except what produced by the shower;
c Higgs decay products must be excluded. Loose check on this
      do j=1,2
         psum(j)=0d0
      enddo
      do i=1,ns
         do j=1,2
            psum(j)=psum(j)+ps(j,i)
         enddo
      enddo
c      if (abs(pph(1)+psum(1)).gt.1d0 .or.
c     #    abs(pph(2)+psum(2)).gt.1d0) then
c         call hwuepr
c         write (*,*) pph
c         write (*,*) psum
c         call hwwarn('HWMATCH',110)
c      endif
c
c      WRITE(*,*) NS
c matching_scale: this is the \mu_Q of the paper
c      matching_scale=50d0
c n_match identified the i-parton sample of the paper: that's the number
c of partons AT THE BORN LEVEL in a given multiplicity
c set max_multiplicity=.true. if the current sample is that of largest
c multiplicity, and equal to .false. otherwise
      call MLM_matching(ps,ns,matching_scale,n_match
     &     ,max_multiplicity, matched)
c
c
      if (matched) then
	pass = 1
      else
	pass = 0
      endif
c
      return
c
c From this point onwards, continue with your normal analysis
c
      END


      subroutine MLM_matching(p_shower,n_shower,matching_scale,n_match
     &     ,max_multiplicity, matched)
      implicit none
C Hard event information:
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
      integer n_hard,i,j
      double precision p_hard(4,MAXNUP)
      INTEGER njet_hard,jet_hard(MAXNUP),n_match,n_shower,njet_shower
     &     ,jet_shower(n_shower)
      DOUBLE PRECISION palg,rfj,sycut,pjet_hard(4,MAXNUP),matching_scale
     &     ,p_shower(4,n_shower),pjet_shower(4,n_shower),getdrv
      logical matched,max_multiplicity,jet_shower_matched
      double precision SHSCALE
      common/cSHSCALE/SHSCALE
      matched=.true.
      palg=1d0
      sycut=matching_scale
      rfj=1.0d0
      call fastjetppgenkt(p_shower,n_shower,rfj,sycut,palg,pjet_shower
     &     ,njet_shower,jet_shower)
c
c      double precision firsteta
c      firsteta = getpseudorap(pjet_shower(4,1),pjet_shower(1,1),pjet_shower(2,1),pjet_shower(3,1))
c      WRITE(*,*) njet_shower, pjet_shower(4,1), pjet_shower(1,1)
c      WRITE(*,*) njet_shower, pjet_shower(2,1), pjet_shower(3,1)
c      WRITE(*,*) njet_shower, sycut
c First check that the number of jets after showering is okay if the
c number of jets required in the matching
      If (max_multiplicity) then
         if (njet_shower.lt.n_match) then
            matched=.false.
            return
         endif
      else
         if (njet_shower.ne.n_match) then
            matched=.false.
            return
         endif
      endif
      n_hard=0
      do i=1,nup
         if (ISTUP(i).eq.1.and.(abs(IDUP(i)).lt.6.or.IDUP(i).eq.21))
     &        then
c Final state QCD parton            
            n_hard=n_hard+1
            do j=1,4
               p_hard(j,n_hard)=PUP(j,i)
            enddo
         endif
      enddo
      sycut=0d0
c      WRITE(*,*) n_hard
      call fastjetppgenkt(p_hard,n_hard,rfj,sycut,palg,pjet_hard
     &     ,njet_hard,jet_hard)
c Check that the hardest jet after showering 'match' with the hardest
c jets before showering
      do i=1,n_match
         jet_shower_matched=.false.
         do j=1,njet_hard
            if (getdrv(pjet_shower(1,i),pjet_hard(1,j)).lt.1.5d0*rfj)
     &           then
               jet_shower_matched=.true.
               exit
            endif
         enddo
         if (.not.jet_shower_matched) then
            matched=.false.
            return
            exit
         endif
      enddo
c
      return
      end


      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-8)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end


      function getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      getdelphi=tmp
      return
      end


      function getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getpseudorap,getdelphi
c
      deta=getpseudorap(en1,ptx1,pty1,pl1)-
     #     getpseudorap(en2,ptx2,pty2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdr=sqrt(dphi**2+deta**2)
      return
      end


      function getdrv(p1,p2)
      implicit none
      real*8 getdrv,p1(5),p2(5)
      real*8 getdr
c
      getdrv=getdr(p1(4),p1(1),p1(2),p1(3),
     #             p2(4),p2(1),p2(2),p2(3))
      return
      end


      FUNCTION PTCALC(P)
      IMPLICIT NONE
      DOUBLE PRECISION PTCALC,P(4),PTSQ
      PTSQ=P(1)**2+P(2)**2
      IF (PTSQ.EQ.0D0) THEN
         PTCALC=0D0
      ELSE
         PTCALC=SQRT(PTSQ)
      ENDIF
      END



