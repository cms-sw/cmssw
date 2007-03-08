      subroutine mysetPDFpath(pathname)
      implicit real*8 (A-H,O-Z)
      integer noemax,nopmax,npfmax,nofmax,linemax,nmxset
      parameter (noemax=1000,nopmax=40,npfmax=10,nofmax=10,linemax=20)
c nmxset is the max number of sets that can be initialised at one time ---- added V5
      parameter (nmxset=3)
      character*232 pathname
      character*232 lhapath
      character*272 lhaname,lhanames(nmxset)
c      character*132 lhapath
      common/LHAPDFC/lhapath
      character*20 lhaparm(20)
      real*8 lhavalue(20)
      common/LHACONTROL/lhaparm,lhavalue

C      write(*,*) 'PATHNAME: ',pathname

      lhaparm(20) = 'LHAPATH'
      do j=1,lnblnk(lhapath)
        lhapath(j:j)=''
      enddo
      lhapath = pathname

C      lhapath = '/afs/cern.ch/sw/lcg/external/MCGenerators/lhapdf/5.2.3/
C     &share/PDFsets'

      return
      end


      subroutine setherwpdf()
      CHARACTER*20
     &     AUTPDF
      CHARACTER*4
     & BDECAY
      COMMON/HWPRCH/AUTPDF(2),BDECAY

      AUTPDF(1) = 'LHAPDF'
      AUTPDF(2) = 'LHAPDF'

      return
      end
