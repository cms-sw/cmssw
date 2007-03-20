*----------------------------------------------------------------------------*
c----------------------------------------------------------------------------c
C..SLHAGIVE
C read SLHA input spectrum file name and gives it to Pythia
C...Sets values of commonblock variables.
 
      SUBROUTINE SLHAGIVE(CHIN)
      implicit none 
*
      character *80 SLHAFILE
            
      common /SLHAPAR/SLHAFILE
      save   /SLHAPAR/
***
      integer IER
      character *(*) chin
      character *40 inam 
      character *80 STRIN
****

       Inam = 'SLHAFILE ='
        call TXRSTR2(inam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           read(strin(1:80),*) SLHAFILE 
         endif



      return
      end

     
******************************************************************************

      SUBROUTINE SLHA_INIT
           
      implicit none
 
      character *80 SLHAFILE
      common /SLHAPAR/SLHAFILE
      save   /SLHAPAR/ 
    
   
      OPEN(33,FORM='FORMATTED',STATUS='UNKNOWN',FILE=SLHAFILE)
 
 
      END
