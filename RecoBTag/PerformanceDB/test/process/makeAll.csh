#!/bin/tcsh

set inputpoolheader=templates/Pool_header.fragment
set inputbtagheader=templates/Btag_header.fragment
set inputtestheader=templates/Test_header.fragment

set inputpoolheader=templates/Pool_footer.fragment
#set inputbtagheader=templates/Btag_footer.fragment
set inputtestheader=templates/Test_footer.fragment

set oututfragname=Pool_template.py
set oututbtagfragname=Btag_template.py
set outputtestfragname=Test_template.py

rm -f Pool_template.py
rm -f Btag_template.py
rm -f tmp.py

cat templates/Pool_pre.fragment > $oututfragname
cat templates/Btag_pre.fragment > $oututbtagfragname
#MC
./makeSingle.csh MC/MCSSVLb.txt MCSSVLb
./makeSingle.csh MC/MCSSVLc.txt MCSSVLc
./makeSingle.csh MC/MCSSVLl.txt MCSSVLl
./makeSingle.csh MC/MCSSVMb.txt MCSSVMb
./makeSingle.csh MC/MCSSVMc.txt MCSSVMc
./makeSingle.csh MC/MCSSVMl.txt MCSSVMl
./makeSingle.csh MC/MCSSVTb.txt MCSSVTb
./makeSingle.csh MC/MCSSVTc.txt MCSSVTc
./makeSingle.csh MC/MCSSVTl.txt MCSSVTl
./makeSingle.csh MC/MCTCHELb.txt MCTCHELb
./makeSingle.csh MC/MCTCHELc.txt MCTCHELc
./makeSingle.csh MC/MCTCHELl.txt MCTCHELl
./makeSingle.csh MC/MCTCHEMb.txt MCTCHEMb
./makeSingle.csh MC/MCTCHEMc.txt MCTCHEMc
./makeSingle.csh MC/MCTCHEMl.txt MCTCHEMl
./makeSingle.csh MC/MCTCHETb.txt MCTCHETb
./makeSingle.csh MC/MCTCHETc.txt MCTCHETc
./makeSingle.csh MC/MCTCHETl.txt MCTCHETl
./makeSingle.csh MC/MCTCHPLb.txt MCTCHPLb
./makeSingle.csh MC/MCTCHPLc.txt MCTCHPLc
./makeSingle.csh MC/MCTCHPLl.txt MCTCHPLl
./makeSingle.csh MC/MCTCHPMb.txt MCTCHPMb
./makeSingle.csh MC/MCTCHPMc.txt MCTCHPMc
./makeSingle.csh MC/MCTCHPMl.txt MCTCHPMl
./makeSingle.csh MC/MCTCHPTb.txt MCTCHPTb
./makeSingle.csh MC/MCTCHPTc.txt MCTCHPTc
./makeSingle.csh MC/MCTCHPTl.txt MCTCHPTl

./makeSingle.csh SYSTEM8/SYSTEM8SSVM.txt SYSTEM8SSVM
./makeSingle.csh SYSTEM8/SYSTEM8SSVT.txt SYSTEM8SSVT
./makeSingle.csh SYSTEM8/SYSTEM8TCHEL.txt SYSTEM8TCHEL
./makeSingle.csh SYSTEM8/SYSTEM8TCHEM.txt SYSTEM8TCHEM
./makeSingle.csh SYSTEM8/SYSTEM8TCHET.txt SYSTEM8TCHET
./makeSingle.csh SYSTEM8/SYSTEM8TCHPL.txt SYSTEM8TCHPL
./makeSingle.csh SYSTEM8/SYSTEM8TCHPM.txt SYSTEM8TCHPM
./makeSingle.csh SYSTEM8/SYSTEM8TCHPT.txt SYSTEM8TCHPT

./makeSingle.csh MISTAG/MISTAGJPL.txt  MISTAGJPL
./makeSingle.csh MISTAG/MISTAGJPM.txt  MISTAGJPM
./makeSingle.csh MISTAG/MISTAGJPT.txt  MISTAGJPT
./makeSingle.csh MISTAG/MISTAGSSVM.txt MISTAGSSVM
./makeSingle.csh MISTAG/MISTAGTCHEL.txt MISTAGTCHEL
./makeSingle.csh MISTAG/MISTAGTCHEM.txt MISTAGTCHEM
./makeSingle.csh MISTAG/MISTAGTCHPM.txt MISTAGTCHPM
./makeSingle.csh MISTAG/MISTAGTCHPT.txt MISTAGTCHPT

./makeSingle.csh PTREL/PTRELJBPL.txt PTRELJBPL
./makeSingle.csh PTREL/PTRELJBPM.txt PTRELJBPM
./makeSingle.csh PTREL/PTRELJBPT.txt PTRELJBPT
./makeSingle.csh PTREL/PTRELJPL.txt  PTRELJPL
./makeSingle.csh PTREL/PTRELJPM.txt  PTRELJPM
./makeSingle.csh PTREL/PTRELJPT.txt  PTRELJPT
./makeSingle.csh PTREL/PTRELSSVL.txt PTRELSSVL
./makeSingle.csh PTREL/PTRELSSVM.txt PTRELSSVM
./makeSingle.csh PTREL/PTRELSSVT.txt PTRELSSVT
./makeSingle.csh PTREL/PTRELTCHEL.txt PTRELTCHEL
./makeSingle.csh PTREL/PTRELTCHEM.txt PTRELTCHEM
./makeSingle.csh PTREL/PTRELTCHET.txt PTRELTCHET
./makeSingle.csh PTREL/PTRELTCHPL.txt PTRELTCHPL
./makeSingle.csh PTREL/PTRELTCHPM.txt PTRELTCHPM
./makeSingle.csh PTREL/PTRELTCHPT.txt PTRELTCHPT

cat templates/Pool_post.fragment >> $oututfragname
