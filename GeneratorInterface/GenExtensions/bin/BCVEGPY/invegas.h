
* vegas.h

**************************************************
*** used in VEGAS, the default bin number in original
*** version of VEGAS is 50.
*** a higher bin number might help to derive a more precise
*** grade. subtle point: the value of ncall used in VEGAS
*** should be increased accordingly, otherwise a larger
*** bin number will have no effects or even make the
*** precision lower than before.
                                                                                                                                               
*** [more explaination on this subtle point can be
*** found in CPC174,241(2006)]
*************************************************
                                                                                                                                               
**************************************************
*** it lies in three folders: generate/, phase/ and system/
                                                                                                                                               
*** NVEGBIN is used in files:
*** generate/ evntinit.F genevnt.F initmixgrade.F
***    phase/ vegas.F
***   system/ vegaslogo.F
*** NVEGCALL and NVEGITMX is used in file:  parameter.F
***************************************************
                                                                                                                                               
*** NVEGBIN ---- bin number
*** these values only for reference.                                                                                                                                               
#ifndef NVEGBIN
#define NVEGBIN  300
#endif
