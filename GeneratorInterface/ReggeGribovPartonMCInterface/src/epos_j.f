      SUBROUTINE FASTJETPPgenkt(P,NPART,R,PALG,F77JETS,NJETS)
      DOUBLE PRECISION P(4,*), R, PALG, F77JETS(4,*)
      DOUBLE PRECISION dP, dR, dPALG
      INTEGER          NPART, NJETS,Nd

      F77JETS(1,1)=0d0
      NJETS=0
      dP=P(1,1)
      Nd=NPART
      dR=R
      dPALG=PALG
c      write(*,*)"FastJet called with :",F77JETS(1,1),NJETS,R,PALG
c      stop"But FastJet not installed !"

      END
