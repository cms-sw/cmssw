      PROGRAM AllMaterialMixtures
C     ========================

      IMPLICIT NONE

      Integer Narg, Iarg, Istatus


      CALL SYSTEM('rm -f do', Istatus)
      CALL SYSTEM('touch do', Istatus)


      Narg = IARGC()

      if (Narg.eq.0) then
         write (*,*) "No input file(s) given."
         write (*,*) "Usage: mixture FILE"
         write (*,*) "Run the mixture program on the input FILE(s)."
         write (*,*) "File names without the extension .in!"
      endif


      do Iarg=1, Narg

         call MaterialMixtures(Iarg)
         
      enddo


      END


      SUBROUTINE MaterialMixtures(Iarg)
C     ========================

      IMPLICIT NONE

      Integer Iarg

      Integer ISTAT,i,j,k,l

      Character*1 Coding, Code
      Character*20 Filename,OutFile,InFile,tzfile,x0file,l0file
      Character*120 inputstring

      Integer Nmix, Ndiv,LunOut,LunIn,Index, Luntz,Lunx0,Lunl0

C...Common Block .................................................
      Integer MaxDiv
      Parameter (MaxDiv=30)
      Character*40 MixtureName, GMIXName
      Character*80 Command
      Character*60 Comment(MaxDiv),Material(MaxDiv)
      Character*3 Type(MaxDiv)
      Real Volume(MaxDiv), Mult(MaxDiv),
     +     Density(MaxDiv),Radl(MaxDiv),MCVolume,MCArea
      Real Intl(MaxDiv)
      Common /MatMix/ Comment, Material,Volume,Mult,Density,Radl,Intl,
     +                MCVolume,MCArea,MixtureName,GMIXName,Type
C.................................................................


      External Lenocc
      Integer Lenocc      


C... initialization

C---> read in the input file from standard input
C      write(*,*) " Which file do you want me to open ?"
C      read(*,*) Filename

      CALL GETARG(Iarg , Filename) 

      InFile = Filename(1:LENOCC(Filename))//".in"
      OutFile = Filename(1:LENOCC(Filename))//".tex" 
      tzfile  = Filename(1:LENOCC(Filename))//".titles"
      x0file  = Filename(1:LENOCC(Filename))//".x0"
      l0file  = Filename(1:LENOCC(Filename))//".l0"

C      write(*,*) Filename, InFile, OutFile

      LunIn = 23
      LunOut = 24
      Luntz = LunOut + 1
      Lunx0 = LunOut + 2
      Lunl0 = LunOut + 3
      open(unit=LunIn,file=InFile,status="OLD",IOSTAT=istat)
      if(istat.ne.0) then
         write(*,*) "Input file not found. Filename: ", InFile
         return
      endif

      open(unit=LunOut,file=OutFile,status="REPLACE")
      open(unit=Luntz,file=tzfile,status="REPLACE")
      open(unit=Lunx0,file=x0file,status="REPLACE")
      open(unit=Lunl0,file=l0file,status="REPLACE")
      call LatexSetup(LunOut)


C---> Big loop over input file

      Nmix = 0
      Ndiv = 0
      do l = 1, 10000     !! file should not have more lines than that

         read(LunIn,'(A)',END=20) inputstring
C         write(*,*) inputstring
         
C...     first check for start of new mixture

         Coding = inputstring(1:1)
C         write(*,*) "Coding ", Coding
         if ( Coding.eq.'#') then   ! the next mixture starts
            if (Nmix.gt.0) then     ! do everything for the last mixture 
               call MakeMixture(NMix,Ndiv,LunOut)
C               write(*,*) "Nmix ", Nmix
C               write(*,*) "Mixture Name:  ", MixtureName, GMixName
C               do j = 1, Ndiv
C                  write(*,*) Code, Index, Comment(j), Material(j),
C     +                 Volume(j), Mult(j), MCVolume
C               enddo
            endif
C... reset everything
            Call ClearCommon
            Nmix = Nmix + 1
            Ndiv = 0
            read(inputstring,*)  Code,MixtureName,GMIXName,
     +           MCVolume,MCArea

         elseif ( Coding.eq.'*') then        ! components
            Ndiv = Ndiv + 1
            read(inputstring,*) Code, Index, Comment(Ndiv), 
     +           Material(Ndiv),Volume(Ndiv), Mult(Ndiv), Type(Ndiv)
            call MatchMaterial(Ndiv)
            if(Ndiv.ne.Index)  write(*,*) 
     +         "******* Inconsistency reading in ",InFile," ******"
         endif

      enddo
 20   continue


C      write(LunOut,*) "\\end{center}"
      write(LunOut,*) "\\end{landscape}"
      write(LunOut,*) "\\end{document}"

      close(LunIn)
      close(LunOut)
      close(Luntz)
      close(Lunx0)
      close(Lunl0)
       
C... write out little latex/dvips script
C      open(30,file="do",status="OLD")
C      write(30,*) "latex ",Filename(1:LENOCC(Filename))
C      write(30,*) "dvips ",Filename(1:LENOCC(Filename)),
C     +     " -o",Filename(1:LENOCC(Filename)),".ps"
C      write(30,*) "gv ",Filename(1:LENOCC(Filename))," &"
C      close(30)

C      write(*,*) "--> I made ",Filename(1:LENOCC(Filename)),
C     +   "  for you. Type ''do'' to see it " 

      write(command,*) "echo 'latex ",Filename(1:LENOCC(Filename)),
     +     "' >> do"
      CALL SYSTEM(command, istat)

      write(command,*) "echo 'dvips ",Filename(1:LENOCC(Filename)),
     +     " -o",Filename(1:LENOCC(Filename)),".ps' ",
     +     " >> do"
      CALL SYSTEM(command, istat)

      write(command,*) "echo 'gv -landscape ",
     +     Filename(1:LENOCC(Filename))," &' ",
     +     " >> do"
      CALL SYSTEM(command, istat)

      write(command,*) "chmod +x do"
      CALL SYSTEM(command, istat)

      write(*,*) "--> I made ",Filename(1:LENOCC(Filename)),
     +   "  for you. Type ''do'' to see it " 

      return
      end


C----------------------------------------------------------------
      
      Subroutine MatchMaterial(Index)
C     ========================

      Implicit None

      Integer Index, Istat, I,J

      Integer MaxPure
      Parameter (MaxPure=350)
      Integer NPure,match
      CHARACTER*25 PureName(MaxPure)
      REAL Pureweight(MaxPure), Purenumber(MaxPure), Puredens(MaxPure),
     +     PureX0(MaxPure), PureL0(MaxPure)
      SAVE NPure, Pureweight, Purenumber, Puredens, PureX0, PureL0,
     +     Purename

      Character*60 string,teststring

      Logical DEBUG,FIRST
      DATA FIRST /.TRUE./
      DATA DEBUG /.TRUE./

      EXTERNAL LENOCC
      Integer LENOCC


C...Common Block .................................................
      Integer MaxDiv
      Parameter (MaxDiv=30)
      Character*40 MixtureName, GMIXName
      Character*60 Comment(MaxDiv),Material(MaxDiv)
      Character*3 Type(MaxDiv)
      Real Volume(MaxDiv), Mult(MaxDiv),
     +     Density(MaxDiv),Radl(MaxDiv),MCVolume,MCArea
      Real Intl(MaxDiv)
      Common /MatMix/ Comment, Material,Volume,Mult,Density,Radl,Intl,
     +                MCVolume,MCArea,MixtureName,GMIXName,Type
C.................................................................


C... read in pure material file

      if (FIRST) then

         open(unit=22,file="pure_materials.input",status="OLD",
     +        IOSTAT=istat)
         
         if(istat.ne.0) then
            write(*,*) "Pure Materials input file could not be opened",
     +           " - I quit"
            stop
         endif

         Npure = 0
         do i=1, MaxPure
            read(22,*,END=10) PureName(i), Pureweight(i), 
     +           PureNumber(i),PureDens(i), PureX0(i), PureL0(i)
            Npure = Npure + 1
         enddo
 10      continue

         close(22)

C... read in mixed material file

         open(unit=22,file="mixed_materials.input",status="OLD",
     +        IOSTAT=istat)
         
         if(istat.ne.0) then
            write(*,*) "Mixed Materials input file could not be opened",
     +           " - I quit"
            stop
         endif

         do i=Npure+1, MaxPure
            read(22,*,END=20) PureName(i), Pureweight(i), 
     +           PureNumber(i),PureDens(i), PureX0(i), PureL0(i)
            Npure = Npure + 1
         enddo
 20      continue

         close(22)
C
         if (debug) then
            write(*,*) "Number of pure materials:  ", Npure
            write(*,*) "Material name            ", "A        ",
     +           "Z         ",
     +           "dens [g/cm3]", "  X_0 [cm]  ","  l_0 [cm]"
            do j= 1, NPure
               write(*,200) PureName(j), Pureweight(j), 
     +              PureNumber(j),PureDens(j), PureX0(j), PureL0(j)
            enddo
         endif
 200     Format(A30,F10.5,F7.0,3F15.5)
 201     Format(A30,A30,F10.5,F7.0,3F15.5)

         FIRST = .FALSE.
      endif

C---> try to match material here !

      String = Material(Index)

      if (DEBUG) write(*,*) 'Matching now ', String

      match = 0
      Do i = 1,NPure
         teststring = PureName(i)
         if(teststring(1:LENOCC(teststring)).eq.
     +        string(1:LENOCC(string))) then
            if (debug)  write(*,201) string, PureName(i), Pureweight(i), 
     +           PureNumber(i), PureDens(i), PureX0(i), PureL0(i)
            match = 1
C... set density and radiation lenght and nuclear interaction length
            Density(Index) = Puredens(I)
            Radl(Index) = PureX0(I)
            Intl(Index) = PureL0(I)
       endif
      enddo

      if (match.ne.1)then
         write(*,*) "Couldn't find match for material  ",
     +        Index, Material(Index)
         write(*,*) "Exiting !!"
         stop
      else
         if(Radl(Index).le.0.) then
            write(*,*) "Radiation length is zero for material ",
     +           Index, Material(Index)
         endif
         if(Density(Index).le.0) then
             write(*,*) "Density is zero for material ",
     +           Index, Material(Index)
          endif
         if(Intl(Index).le.0.) then
            write(*,*)
     +           "Nuclear Interaction length is zero for material ",
     +           Index, Material(Index)
         endif
      endif

      return
      end

C--------------------------------------------------------------

      Subroutine MakeMixture(Nmix,NMat,LUN)
C     =====================================

      Implicit None

C...Common Block .................................................
      Integer MaxDiv
      Parameter (MaxDiv=30)
      Character*40 MixtureName, GMIXName
      Character*60 Comment(MaxDiv),Material(MaxDiv)
      Character*3 Type(MaxDiv)
      Real Volume(MaxDiv), Mult(MaxDiv),
     +     Density(MaxDiv),Radl(MaxDiv),MCVolume,MCArea
      Real Intl(MaxDiv)
      Common /MatMix/ Comment, Material,Volume,Mult,Density,Radl,Intl,
     +                MCVolume,MCArea,MixtureName,GMIXName,Type
C.................................................................

      Integer NMat, i, j, k,LUN,NMix,LUNTZ,NTZ,Lunx0,Lunl0

      Real TVOL,TDEN,TRAD,Weight,PVOL(MaxDiv),PWeight(MaxDiv)
      Real TINT
      Real ws(MaxDiv),tmp,PRAD(MaxDiv),Norm,Ndens,NRadl,PRadl
      Real ws2(MaxDiv),tmp2,PINT(MaxDiv),NIntl,PIntl

      Real PSUP,PSEN,PCAB,PCOL,PELE
      Real PSUP2,PSEN2,PCAB2,PCOL2,PELE2

      Character*60 string,string1,string2,stringmatname

      Character*30 TZName(MaxDiv)
      Character*32 tzstring
      Real         TZVol(MaxDiv), TZVolTot

      External LENOCC
      Integer LENOCC

      character*100 sformat

C..................................................................

C..initialize
      TVOL = 0.     ! compound volume
      TDEN = 0.     ! compound density
      TRAD = 0.     ! compound radiation length
      Weight = 0.   ! Total weight
      TINT = 0.     ! compound nuclear interaction length
      call VZERO(PVOL,MaxDiv)
      call VZERO(Pweight,MaxDiv)
      call VZERO(ws,MaxDiv)
      call VZERO(ws2,MaxDiv)
      call VZERO(PRAD,MaxDiv)
      call VZERO(TZVol,MaxDiv)
      tmp = 0.
      tmp2 = 0.

* total volume
      do i=1, NMat
         Volume(i) = Mult(i)*Volume(i)
         TVOL = TVOL + Volume(i)
      enddo

      if (tvol.le.0.) return
* percentual volume and total density
      do i=1,NMat
         PVOL(i) = Volume(i)/TVOL
         TDEN = TDEN + PVOL(i)*Density(i)
      enddo

* total weight
      Weight = TDEN * TVOL

      do j = 1,NMat
* percentual weight
         if(Volume(j).gt.0.) then
            PWeight(j) = Density(j)*Volume(j)/Weight
* weight for X0 calculation (pweight/(density*radl))
            ws(j) =  Pweight(j)/(Density(j)*Radl(j))
            tmp = tmp + ws(j)
* weight for Lambda0 calculation (pweight/(density*intl))
            ws2(j) = Pweight(j)/(Density(j)*Intl(j))
            tmp2 = tmp2 + ws2(j)
         endif
      enddo
      
* radiation length of compound
      TRAD = 1/(tmp*TDEN)

* nuclear interaction length of compound
      TINT = 1/(tmp2*TDEN)

* contribution to compound X0
      do k = 1,NMat
         PRAD(k) = ws(k)*TRAD*TDEN
      enddo

* contribution to compound Lambda0
      do k = 1,NMat
         PINT(k) = ws2(k)*TINT*TDEN
      enddo

* Normalization factor Mixture/MC volume
      if (MCVolume.gt.0.) then
         Norm = TVOL/MCVolume
      else
         Norm = 1.
      endif

* Normalized density and radiation length and nuclear interaction length

      ndens = TDEN*Norm
      NRadl = TRAD / norm
      NIntl = TINT / norm

* percentual radiation length of compound (if area is given)
      if (MCArea.gt.0) then
         PRadl = MCVolume/(MCArea*NRadl)
      endif

* percentual nuclear interaction length of compound (if area is given)
      if (MCArea.gt.0) then
         PIntl = MCVolume/(MCArea*NIntl)
      endif

C---> separate contributions to X_0 by type
      PSUP = 0.
      PSEN = 0.
      PCAB = 0.
      PCOL = 0.
      PELE = 0.
      do i = 1, NMat
         if(Type(i).eq."SUP") then
            PSUP = PSUP + PRAD(i)
         elseif (Type(i).eq."SEN") then
            PSEN = PSEN + PRAD(i)
         elseif (Type(i).eq."CAB") then
            PCAB = PCAB + PRAD(i) 
         elseif (Type(i).eq."COL") then
            PCOL = PCOL + PRAD(i) 
         elseif (Type(i).eq."ELE") then
            PELE = PELE + PRAD(i) 
         else
            write(*,*) "No grouping given for material ",
     +           Material(i)
         endif
      enddo
      
C---> separate contributions to Lambda_0 by type
      PSUP2 = 0.
      PSEN2 = 0.
      PCAB2 = 0.
      PCOL2 = 0.
      PELE2 = 0.
      do i = 1, NMat
         if(Type(i).eq."SUP") then
            PSUP2 = PSUP2 + PINT(i)
         elseif (Type(i).eq."SEN") then
            PSEN2 = PSEN2 + PINT(i)
         elseif (Type(i).eq."CAB") then
            PCAB2 = PCAB2 + PINT(i) 
         elseif (Type(i).eq."COL") then
            PCOL2 = PCOL2 + PINT(i) 
         elseif (Type(i).eq."ELE") then
            PELE2 = PELE2 + PINT(i) 
         else
            write(*,*) "No grouping given for material ",
     +           Material(i)
         endif
      enddo

C---> write out the results ..................

c$$$      stringmatname = GMIXName
c$$$      call LatexUnderscore(stringmatname)
c$$$      write(LUN,1000) Nmix,MixtureName,stringmatname
c$$$ 1000 Format('\\subsection*{\\underline{',I3,2X,A40,2X,
c$$$     +     '(Material name: ',A40,')',' }}')
c$$$      
c$$$C      write(LUN,*) "\\begin{table}[ht]"
c$$$      write(LUN,*) "\\begin{tabular}{rlrrr}"
c$$$      write(LUN,*) "\\hline\\hline"
c$$$      write(LUN,*) " & Item & \\% Volume & \\% Weight & ",
c$$$     +     "\\% Total X0  \\","\\"
c$$$      write(LUN,*) "\\hline\\hline"
c$$$      
c$$$      do k=1,NMat
c$$$         string = Material(k)
c$$$         call LatexUnderscore(string)
c$$$         write(LUN,1001) k, string(1:LENOCC(string)),100.*PVOL(k),
c$$$     +        100.*Pweight(k),100.*PRAD(k)
c$$$         write(LUN,*) "\\hline"
c$$$      enddo
c$$$ 1001 Format(1X,I4,2X,' & ',A20,' & ',2(1X,F8.3,' & '),1X,F8.3,
c$$$     +     '\\','\\')
      
C
C--------------------New big table START
C
      stringmatname = GMIXName
      call LatexUnderscore(stringmatname)
      write(LUN,1000) MixtureName,stringmatname
 1000 Format('\\subsection*{\\underline{',2X,A40,2X,
     +     '(Material name: ',A40,')',' }}')
      
C      write(LUN,*) "\\begin{table}[ht]"
      write(LUN,*) "\\begin{tabular}{crlrlrlcrrrr}"
C      write(LUN,*) "\\hline\\hline"
      write(LUN,*) " & Component & Material & ",
     +     " Volume & \\%  & ",
     + " Weight & \\% & Density ",
     +     " & X$_0$ & ",
     +     " \\% ",
     +     " & $\\lambda_0$ & ",
     +     " \\% "
      write(LUN,*) "\\","\\"
      write(LUN,*) " & & & ",
     +     " [cm$^3$] & & ",
     + " [g] & & [g/cm$^3$]",
     +     " & [cm] & ",
     +     " ",
     +     " & [cm] & ",
     +     " "
      write(LUN,*) "\\","\\"
C      write(LUN,*) "\\hline\\hline"
      write(LUN,*) "\\hline"
      
      do k=1,NMat
         string = Material(k)
         string1 = Comment(k)
         call LatexUnderscore(string)
         call LatexUnderscore(string1)
         
         if (Volume(k).ge.0.1) then
            write(LUN,1001) k,string1(1:LENOCC(string1)),
     +           string(1:LENOCC(string)),Volume(k),100.*PVOL(k),
     +           Density(k)*Volume(k),
     +           100.*Pweight(k),Density(k),Radl(k),100.*PRAD(k),
     +           Intl(k),100*PINT(k)
        else
            write(LUN,2001) k,string1(1:LENOCC(string1)),
     +           string(1:LENOCC(string)),Volume(k),100.*PVOL(k),
     +           Density(k)*Volume(k),
     +           100.*Pweight(k),Density(k),Radl(k),100.*PRAD(k),
     +           Intl(k),100*PINT(k)
       endif
         write(LUN,*) "\\hline"
      enddo
 1001 Format(1X,I4,2X,' & ',A60,' & ',A20,' & ',
     +     (1X,F10.4,' & '),(1X,F12.3,' & '),(1X,F10.4,' & '),
     +     5(1X,F12.3,' & '),1X,F12.3,
     +     '\\','\\')
 2001 Format(1X,I4,2X,' & ',A60,' & ',A20,' & ',
     +     (1X,E10.4,' & '),(1X,F12.3,' & '),(1X,E10.4,' & '),
     +     5(1X,F12.3,' & '),1X,F12.3,
     +     '\\','\\')

C
C--------------------New big table END
C
      write(LUN,*) "\\end{tabular}"
C      write(LUN,*) "\\vskip 0.1cm"
      write(LUN,*) " "
      write(LUN,*) "\\begin{tabular}{lrr}"
      write(LUN,*) "\\fbox{\\begin{tabular}{rl}"
      write(LUN,1002) "Mixture density [g/cm$^3$]",TDEN
      write(LUN,1002) "Norm. mixture density [g/cm$^3$]",Ndens
      write(LUN,1002) "Mixture Volume [cm$^3$]",TVOL
      write(LUN,1002) "MC Volume [cm$^3$]",MCVolume
      write(LUN,1002) "MC Area [cm$^2]$",MCArea
      write(LUN,1002) "Normalization factor",Norm
      write(LUN,1002) "Mixture X$_0$ [cm]", TRAD
      write(LUN,1002) "Norm. Mixture X$_0$ [cm]",NRadl
      if (MCArea.gt.0) then
         write(LUN,1002) "Norm. Mixture X$_0$ (\\%)",100*PRadl
      endif
      write(LUN,1002) "Mixture $\\lambda_0$ [cm]", TINT
      write(LUN,1002) "Norm. Mixture $\\lambda_0$ [cm]",NIntl
      if (MCArea.gt.0) then
         write(LUN,1002) "Norm. Mixture $\\lambda_0$ (\\%)",100*PIntl
      endif
      write(LUN,1002) "Total weight (g)",weight
 1002 Format(A40," & ",F15.5," \\","\\")

      write(LUN,*) "\\end{tabular}} & \\fbox{\\begin{tabular}{rl}"
      
      write(LUN,1006) "\\underline{X$_0$ contribution}"
      write(LUN,1005) "Support: ",PSUP
      write(LUN,1005) "Sensitive: ",PSEN
      write(LUN,1005) "Cables: ",PCAB
      write(LUN,1005) "Cooling: ",PCOL
      write(LUN,1005) "Electronics: ", PELE
      
      write(LUN,*) "\\end{tabular}} & \\fbox{\\begin{tabular}{rl}"
      
      write(LUN,1006) "\\underline{$\\lambda_0$ contribution}"
      write(LUN,1005) "Support: ",PSUP2
      write(LUN,1005) "Sensitive: ",PSEN2
      write(LUN,1005) "Cables: ",PCAB2
      write(LUN,1005) "Cooling: ",PCOL2
      write(LUN,1005) "Electronics: ", PELE2
      
 1005 Format(A25," & ",F5.3,"\\","\\")
 1006 Format(A40," & \\","\\")
      
      write(LUN,*) "\\end{tabular}}\\end{tabular}"
      write(LUN,*) "\\clearpage"
      
C----> now write out a pseudo title file

      LUNTZ = LUN+1

C * first add volumes of same material
      
      Ntz = 0

      do 500 i = 1, NMat  
C.. see if there's a match    
         do j = 1, Ntz
            if(Material(i)(1:LENOCC(Material(i))).eq.
     +           TZName(j)(1:LENOCC(TZName(j))) ) then
               TZVol(j) = TZVol(j) + PVol(i)
               go to 500
            endif
         enddo
         Ntz = Ntz + 1
         TZName(Ntz) = material(i)
C         write(*,*) "Ntz increased: ",NTz, TZName(Ntz)
         TZVol(Ntz)  = PVol(i)
 500  continue

      TZVolTot = 0.
      do i = 1, Ntz
         TZVolTot = TZVolTot + TZVol(i)
      enddo
      if( abs(TZVolTot-1.) .gt. 1.E-6) then
         write(*,*) "Percentual Volumes don't add up to 1 !!"
      endif

C      write(*,*) "NTZ: ", Ntz
C      do i =1, Ntz
C         write(*,*) TZName(i)
C      enddo

      tzstring = '"'//GMIXName(1:LENOCC(GMIXName))//'"'
      write(LUNTZ,1010) tzstring,-1*Ntz, ndens
      do j = 1, Ntz
         tzstring = '"'//TZName(j)(1:LENOCC(TZName(j)))//'"'
         write(LUNTZ,1011) tzstring, -100.*TZVol(j)
      enddo

 1010 Format(7X,A20,I3,4X,F12.6)
 1011 Format(10X,A22,F8.3)

C--> and x0 contributions into a separate file

      Lunx0 = LUN+2
      Lunl0 = LUN+3

C     Original
C     tzstring = '"'//GMIXName(1:LENOCC(GMIXName))//'"'
C     if(PSUP.gt.0.) write(Lunx0,1012) tzstring,'" SUP"',PSUP
C     if(PSEN.gt.0.) write(Lunx0,1012) tzstring,'" SEN"',PSEN
C     if(PCAB.gt.0.) write(Lunx0,1012) tzstring,'" CAB"',PCAB
C     if(PCOL.gt.0.) write(Lunx0,1012) tzstring,'" COL"',PCOL
C     if(PELE.gt.0.) write(Lunx0,1012) tzstring,'" ELE"',PELE
C     write(Lunx0,*) "   "
C     1012 Format(1X,A23,A6,1X,F5.3)
C     

C     rr
      tzstring = GMIXName(1:LENOCC(GMIXName))
      write(Lunx0,1012) tzstring,PSUP,PSEN,PCAB,PCOL,PELE
      write(Lunx0,*) "   "
      tzstring = GMIXName(1:LENOCC(GMIXName))
      write(Lunl0,1012) tzstring,PSUP2,PSEN2,PCAB2,PCOL2,PELE2
      write(Lunl0,*) "   "
 1012 Format(A32,1X,F5.3,1X,F5.3,1X,F5.3,1X,F5.3,1X,F5.3)
C     rr
      
      return
      end

C----------------------------------------------------------------
      
      Subroutine ClearCommon
C     ----------------------

      Implicit None

      Integer I

C...Common Block .................................................
      Integer MaxDiv
      Parameter (MaxDiv=30)
      Character*40 MixtureName, GMIXName
      Character*60 Comment(MaxDiv),Material(MaxDiv)
      Character*3 Type(MaxDiv)
      Real Volume(MaxDiv), Mult(MaxDiv),
     +     Density(MaxDiv),Radl(MaxDiv),MCVolume,MCArea
      Real Intl(MaxDiv)
      Common /MatMix/ Comment, Material,Volume,Mult,Density,Radl,Intl,
     +                MCVolume,MCArea,MixtureName,GMIXName,Type
C.................................................................

      do i=1, MaxDiv
         Comment(i) = "  "
         Material(i) = "  "
         Type(i)     = "  "
      enddo

      Call VZERO(Volume,MaxDiv)
      Call VZERO(Mult,MaxDiv)
      Call VZERO(Density,MaxDiv)
      Call VZERO(Radl,MaxDiv)
      Call VZERO(Intl,MaxDiv)
      MCVolume = 0.
      MCArea = 0.
      MixtureName = " "
      GMIXName = " "

      return 
      end

C------------------------------------------------------------------

      Subroutine LatexSetup(LUN)
C     ==========================
      
      Implicit None
      Integer LUN
C--
      write(LUN,*) "\\documentclass[10pt]{article}"
      write(LUN,*) "\\usepackage{lscape}"
      write(LUN,*) "\\usepackage{a4}"
      write(LUN,*) "\\pagestyle{empty}"
      write(LUN,*) "\\renewcommand{\\baselinestretch}{1.1}"
      write(LUN,*) "\\parskip4pt"
      write(LUN,*) "\\setlength{\\textwidth}{18cm}"
      write(LUN,*) "\\setlength{\\textheight}{28cm}"     
      write(LUN,*) "\\addtolength{\\oddsidemargin}{-1.5cm}"
      write(LUN,*) "\\addtolength{\\evensidemargin}{-1.5cm}"
      write(LUN,*) "\\addtolength{\\topmargin}{-5cm}"
      write(LUN,*) "\\begin{document}"
      write(LUN,*) "\\begin{landscape}"
        
      return
      end
      


      Subroutine LatexUnderscore(stringname)
C     =======================================
      Implicit None
      Character*60 stringname,stringtemp
      Integer      k,maxunderscore,findunderscore,findspace
      Integer      underscorefound
      
      stringtemp = stringname
      findunderscore = 0
      k = 0
      maxunderscore = 5  !At most maxunderscore '_' searched
      underscorefound = 0
      
C     Avoid LaTeX errors when compiling names with '_'
c     write(*,*) k,stringname,stringtemp
      do k=1,maxunderscore
         findunderscore = INDEX(stringtemp,'_')
         if(findunderscore.ne.0) then
            underscorefound = underscorefound + 1
            if(k.eq.1) then
               stringname = stringtemp(1:findunderscore-1) // '\\'
     +              // stringtemp(findunderscore:findunderscore)
            else
               findspace = INDEX(stringname,' ')
               stringname = stringname(1:findspace-1)
     +              // stringtemp(1:findunderscore-1) // '\\'
     +              // stringtemp(findunderscore:findunderscore)
            endif
            stringtemp = stringtemp(findunderscore+1:)
         endif
c     write(*,*) k,stringname,stringtemp
      enddo
      if(underscorefound.ne.0) then
         findspace = INDEX(stringname,' ')
         stringname = stringname(1:findspace-1) // stringtemp
      endif
c     write(*,*) k,stringname,stringtemp
      return
      end
C     
      











