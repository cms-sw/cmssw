How to build your own Reference Manual

 cmsrel CMSSW_4_3_0
 cd CMSSW_4_3_0/src/
 cmsenv
 cvs co Documentation

 cvs co YOUR_PACKAGE(s)

 cd ..

 BUILD ( scramv1 b referencemanual )

 wait...

 firefox doc/html/index.html