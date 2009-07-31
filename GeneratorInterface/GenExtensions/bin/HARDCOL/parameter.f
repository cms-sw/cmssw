cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...this subroutine is used to set the necessary parameters for      c
c...the initialization for hard color singlet exchange.              c
c...to use the program youd need to make a directory: (data) to      c
c...save all the obtained data-files.                                c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   to have a better understanding of setting the parameters         c
c   you may see the README file to get more detailed information.    c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) Rikard Enberg, Gunnar Ingelman, Leszek Motyka        c
c reference: hep-ph/0111090                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	SUBROUTINE SETPARAMETERS
c...preamble: declarations.
        IMPLICIT DOUBLE PRECISION(A-H, O-Z)
	IMPLICIT INTEGER(I-N)

      include "hardcol_set_par.inc"

c...user process event common block.
      
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/HCLPAR/ECM,NEV
      SAVE /PYPARS/,/PYDAT1/,/PYDATR/,/PYSUBS/,/HCLPAR/
      

c	logical wronginput
c
c... Read parameter setting from file hardcol_set_par.nam
c	
        Call Read_parameter_settings
c	
c... change the intitial state of the random number
c        mrpy(1) = 19780503		! default value
	 mrpy(1) = irandom
	 write( 6, * ) ' '
	 write( 6, * ) ' Change default value of random, mrpy(1), to ',
     +	 mrpy(1)	  
	 write( 6, * ) ' '
c... end random change	 	

C	pi = dacos(-1.0d0)

      ecm = ENERGYOFLHC
      nev = NUMOFEVENTS
      MSTP(2) = MSTP2
      CKIN(3) = CKIN3
      MSEL = MSEL0
      MSUB(406) = MSUB406
      MSUB(407) = MSUB407
      MSUB(408) = MSUB408
      MSTP(198) = MSTP198


c...error message.
c      wronginput=.false.
c	CALL uperror(wronginput)
c	if(wronginput) stop '-----input error! stop the program !!!'

c	CALL parameters()
c      CALL dparameters()
c      CALL coupling()
	
	return
	end


c***************************************
c***************************************
	SUBROUTINE Read_parameter_settings
c
c... Get parameters from namelist
        implicit double precision(a-h, o-z)
	implicit integer(i-n)
	
	Namelist / hardcol_set_par / ENERGYOFLHC, NUMOFEVENTS,
     +  MSTP2, CKIN3, MSEL0, MSUB406, MSUB407, MSUB408, MSTP198,
     +  irandom

      include "hardcol_set_par.inc"
c
c-------------------------------------------------------------------------------
c
	open( unit=1, file='hardcol_set_par.nam',Status='Old',Err=99)
        read( 1, nml=hardcol_set_par, err=90)
	write( 6, * ) ' '
	write( 6, * ) ' Contents of namelist *hardcol_set_par*: '
        write( 6, nml=hardcol_set_par)
	write( 6, * ) ' '
        Close( 1 ) 
        Return
c
  90  	Write( 6, * ) ' !!!!! Unable to read namelist hardcol_set_par '  
        Call Exit 
  99	Write( 6, * ) ' !!!!! Unable to open hardcol_set_par.nam'
        Call Exit
 	End

