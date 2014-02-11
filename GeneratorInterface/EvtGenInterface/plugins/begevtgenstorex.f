C--------------------------------------------------------------------------
C $Id: begevtgenstorex.F,v 1.1 2009/02/19 16:21:21 covarell Exp $
C
C Environment:
C      This software is part of the EvtGen package developed jointly
C      for the BaBar and CLEO collaborations.  If you use all or part
C      of it, please give an appropriate acknowledgement.
C
C Copyright Information: See EvtGen/COPYRIGHT
C      Copyright (C) 1998      Caltech, UCSB
C
C Module: begevtgenstorex.F
C
C Description:
C
C Modification history:
C
C    DJL/RYD     August 11, 1998         Module created
C
C------------------------------------------------------------------------
      subroutine begevtgenstorex(entry,daugfirst,dauglast)
      implicit none
*
* routine to fill the stdhep common blocks from
* evtgen (C++). This routine allows the C++ program not to
* have to mess with common blocks.
*
* Anders Ryd,  Dec 96   Created.
*
* 

#include "EvtGenModels/common_hepevt.inc"
      logical qedrad
      integer ph_nmxhep ! this is parameter nmxhep in photos/photos_make
*                     ! Renamed here to avoid name conflict in stdhep.inc
      parameter (ph_nmxhep=10000)
      common / phoqed / qedrad(ph_nmxhep)
      integer entry
c     ,eventnum,numparticle,istat,partnum
c      integer mother,
      integer daugfirst,dauglast

      integer i
      
c      double precision px,py,pz,e,m,x,y,z,t
      
c      stdhepnum=partnum
      
      
c      d_h_nevhep=eventnum
c      d_h_nhep=numparticle
c      d_h_isthep(entry)=istat
c      d_h_idhep(entry)=stdhepnum
c      d_h_jmohep(1,entry)=mother
c      d_h_jmohep(2,entry)=0
c      d_h_jdahep(1,entry)=daugfirst
c      d_h_jdahep(2,entry)=dauglast
c      d_h_phep(1,entry)=px
c      d_h_phep(2,entry)=py
c      d_h_phep(3,entry)=pz
c      d_h_phep(4,entry)=e
c      d_h_phep(5,entry)=m
c      d_h_vhep(1,entry)=x
c      d_h_vhep(2,entry)=y
c      d_h_vhep(3,entry)=z
c      d_h_vhep(4,entry)=t
      
      qedrad(entry)=.true.
      if (daugfirst.gt.0 .and. dauglast.gt.0) THEN
        do i=daugfirst, dauglast
          qedrad(i) = .true.
        end do
      end if
      
      return
      
      end
      

