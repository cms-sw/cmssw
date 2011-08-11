How to generate Reference Manual from zero?
 - ensure that you have rwx permission in current folder
 - run command ./refmanual.sh CMSSW_4_3_0
   - checkouting source
   - building Reference Manual
   - applying non doxygen features



How to generate Reference Manual from your source?
 - provide your own INPUT folder in doxygen/cfgfile file line 462
 - comment or remove 34 line in refmanual.sh file
 - run command ./refmanual.sh CMSSW_4_3_0
   - building Reference Manual on your provided source code
   - applying non doxygen features
