C
C     Returns the interpolated value of the BFKL 
C     amplitude for hard color singlet exchange.
C     It needs a file in which the exact values  
C     are stored in the chosen form. 
C     
C     Modifications for PYTHIA:
C        -gives result in GeV-2 instead of mb
C        -not function of controll anymore
C        -checks range of (y,q)


C
C    initialisation: reads tabularized values of cross section
C

C    Modified by Sheila Amaral, 16 Oct 2009
C
C    Implemented the file path using environment variable CMSSW_BASE
C    The .dat files should be in data/ directory

      SUBROUTINE read_hcs_file
      IMPLICIT NONE   

C   Pythia variables
      INTEGER MSTP,MSTI
      DOUBLE PRECISION PARP,PARI
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)

C   Local variables      
      REAL*8 hcstab(12,25,3)
      COMMON / hcstab / hcstab
         
      CHARACTER hcs_name*120   !! 
      COMMON/ hcsfile / hcs_name          
      
      REAL*8 x1,x2

      INTEGER iq,iy
     
C  Choose value of s_0
C  The *.dat files must reside in the directory where the program is run.      
C
      character cmsdir*82
      call getenv('CMSSW_BASE',cmsdir)

C
      IF(MSTP(198).EQ.1) THEN
         hcs_name = cmsdir(1:index(cmsdir,' ')-1)
     $            //'/src/GeneratorInterface/GenExtensions/data/'
     $            //'dJYs05rc1.dat'
      ELSEIF(MSTP(198).EQ.2) THEN
         hcs_name = cmsdir(1:index(cmsdir,' ')-1)
     $            //'/src/GeneratorInterface/GenExtensions/data/'
     $            //'dJYs1rc1.dat'
      ELSEIF(MSTP(198).EQ.3) THEN
         hcs_name = cmsdir(1:index(cmsdir,' ')-1)
     $            //'/src/GeneratorInterface/GenExtensions/data/'
     $            //'dJYs2rc1.dat'
      ELSE
         WRITE(*,*) 
         WRITE(*,*) 'ERROR: MSTP(198) must be 1,2 or 3!'
         WRITE(*,*) 'STOPPING EXECUTION...'
         WRITE(*,*) 
         STOP
      ENDIF
      
      WRITE(*,*) 'read_hcs_file: chose ',hcs_name
         
      OPEN( UNIT = 5, FILE = hcs_name, STATUS = 'OLD')
C      REWIND(5) 
      READ(5,*)
     
      DO 10, iq = 1, 12 	
        DO 20, iy = 1,25 		
          READ(5,*) hcstab(iq,iy,2),hcstab(iq,iy,1),x1,x2,
     >              hcstab(iq,iy,3)		
 20     CONTINUE   
 10   CONTINUE               
    
      CLOSE(5)
      RETURN

      END


C
C    returns the  value based on direct quadratic interpolation
C    table of hcs XS values (multiplied by some factors) for 
C    0<Y<11.5, 5<Q<144
C
      FUNCTION HCS_XS(y,q)      
      IMPLICIT NONE

      REAL*8  HCS_XS
      REAL*8  y,q, controll     

      REAL*8 Pi, CONV
     
      PARAMETER ( Pi = 3.141592654D0 )      
      PARAMETER ( CONV = 0.3894D6 )  
 
      REAL*8 hcstab(12,25,3)
      COMMON / hcstab / hcstab
      
      INTEGER NQ,NY
      PARAMETER ( NY = 25 ) 
      PARAMETER ( NQ = 12 ) 
   
      REAL*8 YMAX, YSTEP, QRAT, QMIN     
      PARAMETER ( YMAX  = 12.D0 )
      PARAMETER ( YSTEP = 5.D-1 ) 
      PARAMETER ( QMIN =  5.D0  )
      PARAMETER ( QRAT =  1.4D0 )

      INTEGER iq,iy
       
      REAL*8 q1,q2,q3,y1,y2,y3,z1,zq2,zq3,zy2,zy3,zqy
      REAL*8 aq,ay,aqy,bq,by,c, t2

C      WRITE(*,*)
C      WRITE(*,*) 'y=',y
C      WRITE(*,*) 'q=',q
      
      IF(y.GT.11.5D0.OR.q.LT.5D0.OR.q.GT.144D0) THEN
        HCS_XS=0D0
C         WRITE(*,*)
C         WRITE(*,*) 'WARNING! Outside of kinematical region where'
C         WRITE(*,*) 'calculation of d(sigma-hat)/d(t-hat) is valid!'
C         WRITE(*,*) 'y=',y
C         WRITE(*,*) 'q=',q
C         WRITE(*,*)
        GOTO 30
      ENDIF
  
      iy = IDINT(y/YSTEP)+1
      iq = IDINT(DLOG(q/QMIN) / DLOG(QRAT)) + 1 

      q1 = hcstab(iq,iy,1)
      q2 = hcstab(iq+1,iy,1)     
      q3 = hcstab(iq+2,iy,1)

      y1 = hcstab(iq,iy,2)
      y2 = hcstab(iq,iy+1,2)     
      y3 = hcstab(iq,iy+2,2)

      z1 =  hcstab(iq,iy,3)
      zq2 = hcstab(iq+1,iy,3)
      zq3 = hcstab(iq+2,iy,3)
      zy2 = hcstab(iq,iy+1,3)
      zy3 = hcstab(iq,iy+2,3)
      zqy = hcstab(iq+1,iy+1,3)


C  010619: aq and ay changed according to Leszek
      aq = ((q3-q1)*(zq2-z1)-(q2-q1)*(zq3-z1))/
     >     ((q2-q1)*(q3-q1)*(q2-q3))

      ay =  ((y3-y1)*(zy2-z1)-(y2-y1)*(zy3-z1))/
     >     ((y2-y1)*(y3-y1)*(y2-y3))     

      aqy = (z1 + zqy - zq2 - zy2)/(q2*y2+q1*y1-q1*y2-q2*y1)      

      bq = (zq2 - z1)/(q2-q1) - aq*(q1+q2) - aqy*y1   
      by = (zy2 - z1)/(y2-y1) - ay*(y1+y2) - aqy*q1  

      c = zqy - aq*q2*q2 - bq*q2 - ay*y2*y2 - by*y2 - aqy*y2*q2      
      
      t2 = Pi*32.D0/9.D0 / q**4
      t2 = t2* (12.D0*Pi / 25.D0 / DLOG(q**2/0.04D0))**2 

      controll = aq*q**2+ay*y**2+aqy*q*y+bq*q+by*y+c

      HCS_XS = t2*(aq*q**2+ay*y**2+aqy*q*y+bq*q+by*y+c) 

C      WRITE(6,*), q1, q2, q3
C      WRITE(6,*), y1, y2, y3
C      WRITE(6,*), z1,zq2,zy2,zqy
C      WRITE(6,*), controll  

 30   RETURN
      END



