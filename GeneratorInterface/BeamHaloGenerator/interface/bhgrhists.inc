*****************************************************************
*                                                               *
*     DESCRIPTION: this include file contains common blocks     *
*       with histogram information needed for random number     *
*       generation according to distributions in histograms     *
*       - numbering scheme: 1. = 1/2/3/4 = Vx/Vy/dx.dy          *
*                           2. = 1 ... 8 = eight E_K ranges     *
*                           01 = ln(E_K)                        *
*     last update (31/05/2005) by droll                         *
*                                                               *
*****************************************************************
      INTEGER              N_BINS
      PARAMETER           (N_BINS = 100) ! number of bins

* declaration of common block for mu+
      REAL          H01MUP        ,L01MUP,W01MUP
     &             ,H11MUP        ,L11MUP,W11MUP
     &             ,H12MUP        ,L12MUP,W12MUP
     &             ,H13MUP        ,L13MUP,W13MUP
     &             ,H14MUP        ,L14MUP,W14MUP
     &             ,H15MUP        ,L15MUP,W15MUP
     &             ,H16MUP        ,L16MUP,W16MUP
     &             ,H17MUP        ,L17MUP,W17MUP
     &             ,H18MUP        ,L18MUP,W18MUP
     &             ,H21MUP        ,L21MUP,W21MUP
     &             ,H22MUP        ,L22MUP,W22MUP
     &             ,H23MUP        ,L23MUP,W23MUP
     &             ,H24MUP        ,L24MUP,W24MUP
     &             ,H25MUP        ,L25MUP,W25MUP
     &             ,H26MUP        ,L26MUP,W26MUP
     &             ,H27MUP        ,L27MUP,W27MUP
     &             ,H28MUP        ,L28MUP,W28MUP
     &             ,H31MUP        ,L31MUP,W31MUP
     &             ,H32MUP        ,L32MUP,W32MUP
     &             ,H33MUP        ,L33MUP,W33MUP
     &             ,H34MUP        ,L34MUP,W34MUP
     &             ,H35MUP        ,L35MUP,W35MUP
     &             ,H36MUP        ,L36MUP,W36MUP
     &             ,H37MUP        ,L37MUP,W37MUP
     &             ,H38MUP        ,L38MUP,W38MUP
     &             ,H41MUP        ,L41MUP,W41MUP
     &             ,H42MUP        ,L42MUP,W42MUP
     &             ,H43MUP        ,L43MUP,W43MUP
     &             ,H44MUP        ,L44MUP,W44MUP
     &             ,H45MUP        ,L45MUP,W45MUP
     &             ,H46MUP        ,L46MUP,W46MUP
     &             ,H47MUP        ,L47MUP,W47MUP
     &             ,H48MUP        ,L48MUP,W48MUP
      !             histo  # bins  lowlim binwid
      COMMON/RHIMUP/H01MUP(N_BINS),L01MUP,W01MUP
     &             ,H11MUP(N_BINS),L11MUP,W11MUP
     &             ,H12MUP(N_BINS),L12MUP,W12MUP
     &             ,H13MUP(N_BINS),L13MUP,W13MUP
     &             ,H14MUP(N_BINS),L14MUP,W14MUP
     &             ,H15MUP(N_BINS),L15MUP,W15MUP
     &             ,H16MUP(N_BINS),L16MUP,W16MUP
     &             ,H17MUP(N_BINS),L17MUP,W17MUP
     &             ,H18MUP(N_BINS),L18MUP,W18MUP
     &             ,H21MUP(N_BINS),L21MUP,W21MUP
     &             ,H22MUP(N_BINS),L22MUP,W22MUP
     &             ,H23MUP(N_BINS),L23MUP,W23MUP
     &             ,H24MUP(N_BINS),L24MUP,W24MUP
     &             ,H25MUP(N_BINS),L25MUP,W25MUP
     &             ,H26MUP(N_BINS),L26MUP,W26MUP
     &             ,H27MUP(N_BINS),L27MUP,W27MUP
     &             ,H28MUP(N_BINS),L28MUP,W28MUP
     &             ,H31MUP(N_BINS),L31MUP,W31MUP
     &             ,H32MUP(N_BINS),L32MUP,W32MUP
     &             ,H33MUP(N_BINS),L33MUP,W33MUP
     &             ,H34MUP(N_BINS),L34MUP,W34MUP
     &             ,H35MUP(N_BINS),L35MUP,W35MUP
     &             ,H36MUP(N_BINS),L36MUP,W36MUP
     &             ,H37MUP(N_BINS),L37MUP,W37MUP
     &             ,H38MUP(N_BINS),L38MUP,W38MUP
     &             ,H41MUP(N_BINS),L41MUP,W41MUP
     &             ,H42MUP(N_BINS),L42MUP,W42MUP
     &             ,H43MUP(N_BINS),L43MUP,W43MUP
     &             ,H44MUP(N_BINS),L44MUP,W44MUP
     &             ,H45MUP(N_BINS),L45MUP,W45MUP
     &             ,H46MUP(N_BINS),L46MUP,W46MUP
     &             ,H47MUP(N_BINS),L47MUP,W47MUP
     &             ,H48MUP(N_BINS),L48MUP,W48MUP

* declaration of common block for mu-
      REAL          H01MUM        ,L01MUM,W01MUM
     &             ,H11MUM        ,L11MUM,W11MUM
     &             ,H12MUM        ,L12MUM,W12MUM
     &             ,H13MUM        ,L13MUM,W13MUM
     &             ,H14MUM        ,L14MUM,W14MUM
     &             ,H15MUM        ,L15MUM,W15MUM
     &             ,H16MUM        ,L16MUM,W16MUM
     &             ,H17MUM        ,L17MUM,W17MUM
     &             ,H18MUM        ,L18MUM,W18MUM
     &             ,H21MUM        ,L21MUM,W21MUM
     &             ,H22MUM        ,L22MUM,W22MUM
     &             ,H23MUM        ,L23MUM,W23MUM
     &             ,H24MUM        ,L24MUM,W24MUM
     &             ,H25MUM        ,L25MUM,W25MUM
     &             ,H26MUM        ,L26MUM,W26MUM
     &             ,H27MUM        ,L27MUM,W27MUM
     &             ,H28MUM        ,L28MUM,W28MUM
     &             ,H31MUM        ,L31MUM,W31MUM
     &             ,H32MUM        ,L32MUM,W32MUM
     &             ,H33MUM        ,L33MUM,W33MUM
     &             ,H34MUM        ,L34MUM,W34MUM
     &             ,H35MUM        ,L35MUM,W35MUM
     &             ,H36MUM        ,L36MUM,W36MUM
     &             ,H37MUM        ,L37MUM,W37MUM
     &             ,H38MUM        ,L38MUM,W38MUM
     &             ,H41MUM        ,L41MUM,W41MUM
     &             ,H42MUM        ,L42MUM,W42MUM
     &             ,H43MUM        ,L43MUM,W43MUM
     &             ,H44MUM        ,L44MUM,W44MUM
     &             ,H45MUM        ,L45MUM,W45MUM
     &             ,H46MUM        ,L46MUM,W46MUM
     &             ,H47MUM        ,L47MUM,W47MUM
     &             ,H48MUM        ,L48MUM,W48MUM
      !             histo  # bins  lowlim binwid
      COMMON/RHIMUM/H01MUM(N_BINS),L01MUM,W01MUM
     &             ,H11MUM(N_BINS),L11MUM,W11MUM
     &             ,H12MUM(N_BINS),L12MUM,W12MUM
     &             ,H13MUM(N_BINS),L13MUM,W13MUM
     &             ,H14MUM(N_BINS),L14MUM,W14MUM
     &             ,H15MUM(N_BINS),L15MUM,W15MUM
     &             ,H16MUM(N_BINS),L16MUM,W16MUM
     &             ,H17MUM(N_BINS),L17MUM,W17MUM
     &             ,H18MUM(N_BINS),L18MUM,W18MUM
     &             ,H21MUM(N_BINS),L21MUM,W21MUM
     &             ,H22MUM(N_BINS),L22MUM,W22MUM
     &             ,H23MUM(N_BINS),L23MUM,W23MUM
     &             ,H24MUM(N_BINS),L24MUM,W24MUM
     &             ,H25MUM(N_BINS),L25MUM,W25MUM
     &             ,H26MUM(N_BINS),L26MUM,W26MUM
     &             ,H27MUM(N_BINS),L27MUM,W27MUM
     &             ,H28MUM(N_BINS),L28MUM,W28MUM
     &             ,H31MUM(N_BINS),L31MUM,W31MUM
     &             ,H32MUM(N_BINS),L32MUM,W32MUM
     &             ,H33MUM(N_BINS),L33MUM,W33MUM
     &             ,H34MUM(N_BINS),L34MUM,W34MUM
     &             ,H35MUM(N_BINS),L35MUM,W35MUM
     &             ,H36MUM(N_BINS),L36MUM,W36MUM
     &             ,H37MUM(N_BINS),L37MUM,W37MUM
     &             ,H38MUM(N_BINS),L38MUM,W38MUM
     &             ,H41MUM(N_BINS),L41MUM,W41MUM
     &             ,H42MUM(N_BINS),L42MUM,W42MUM
     &             ,H43MUM(N_BINS),L43MUM,W43MUM
     &             ,H44MUM(N_BINS),L44MUM,W44MUM
     &             ,H45MUM(N_BINS),L45MUM,W45MUM
     &             ,H46MUM(N_BINS),L46MUM,W46MUM
     &             ,H47MUM(N_BINS),L47MUM,W47MUM
     &             ,H48MUM(N_BINS),L48MUM,W48MUM

* declaration of common block for pi+
      REAL          H01PIP        ,L01PIP,W01PIP
     &             ,H11PIP        ,L11PIP,W11PIP
     &             ,H12PIP        ,L12PIP,W12PIP
     &             ,H13PIP        ,L13PIP,W13PIP
     &             ,H14PIP        ,L14PIP,W14PIP
     &             ,H15PIP        ,L15PIP,W15PIP
     &             ,H16PIP        ,L16PIP,W16PIP
     &             ,H17PIP        ,L17PIP,W17PIP
     &             ,H18PIP        ,L18PIP,W18PIP
     &             ,H21PIP        ,L21PIP,W21PIP
     &             ,H22PIP        ,L22PIP,W22PIP
     &             ,H23PIP        ,L23PIP,W23PIP
     &             ,H24PIP        ,L24PIP,W24PIP
     &             ,H25PIP        ,L25PIP,W25PIP
     &             ,H26PIP        ,L26PIP,W26PIP
     &             ,H27PIP        ,L27PIP,W27PIP
     &             ,H28PIP        ,L28PIP,W28PIP
     &             ,H31PIP        ,L31PIP,W31PIP
     &             ,H32PIP        ,L32PIP,W32PIP
     &             ,H33PIP        ,L33PIP,W33PIP
     &             ,H34PIP        ,L34PIP,W34PIP
     &             ,H35PIP        ,L35PIP,W35PIP
     &             ,H36PIP        ,L36PIP,W36PIP
     &             ,H37PIP        ,L37PIP,W37PIP
     &             ,H38PIP        ,L38PIP,W38PIP
     &             ,H41PIP        ,L41PIP,W41PIP
     &             ,H42PIP        ,L42PIP,W42PIP
     &             ,H43PIP        ,L43PIP,W43PIP
     &             ,H44PIP        ,L44PIP,W44PIP
     &             ,H45PIP        ,L45PIP,W45PIP
     &             ,H46PIP        ,L46PIP,W46PIP
     &             ,H47PIP        ,L47PIP,W47PIP
     &             ,H48PIP        ,L48PIP,W48PIP
      !             histo  # bins  lowlim binwid
      COMMON/RHIPIP/H01PIP(N_BINS),L01PIP,W01PIP
     &             ,H11PIP(N_BINS),L11PIP,W11PIP
     &             ,H12PIP(N_BINS),L12PIP,W12PIP
     &             ,H13PIP(N_BINS),L13PIP,W13PIP
     &             ,H14PIP(N_BINS),L14PIP,W14PIP
     &             ,H15PIP(N_BINS),L15PIP,W15PIP
     &             ,H16PIP(N_BINS),L16PIP,W16PIP
     &             ,H17PIP(N_BINS),L17PIP,W17PIP
     &             ,H18PIP(N_BINS),L18PIP,W18PIP
     &             ,H21PIP(N_BINS),L21PIP,W21PIP
     &             ,H22PIP(N_BINS),L22PIP,W22PIP
     &             ,H23PIP(N_BINS),L23PIP,W23PIP
     &             ,H24PIP(N_BINS),L24PIP,W24PIP
     &             ,H25PIP(N_BINS),L25PIP,W25PIP
     &             ,H26PIP(N_BINS),L26PIP,W26PIP
     &             ,H27PIP(N_BINS),L27PIP,W27PIP
     &             ,H28PIP(N_BINS),L28PIP,W28PIP
     &             ,H31PIP(N_BINS),L31PIP,W31PIP
     &             ,H32PIP(N_BINS),L32PIP,W32PIP
     &             ,H33PIP(N_BINS),L33PIP,W33PIP
     &             ,H34PIP(N_BINS),L34PIP,W34PIP
     &             ,H35PIP(N_BINS),L35PIP,W35PIP
     &             ,H36PIP(N_BINS),L36PIP,W36PIP
     &             ,H37PIP(N_BINS),L37PIP,W37PIP
     &             ,H38PIP(N_BINS),L38PIP,W38PIP
     &             ,H41PIP(N_BINS),L41PIP,W41PIP
     &             ,H42PIP(N_BINS),L42PIP,W42PIP
     &             ,H43PIP(N_BINS),L43PIP,W43PIP
     &             ,H44PIP(N_BINS),L44PIP,W44PIP
     &             ,H45PIP(N_BINS),L45PIP,W45PIP
     &             ,H46PIP(N_BINS),L46PIP,W46PIP
     &             ,H47PIP(N_BINS),L47PIP,W47PIP
     &             ,H48PIP(N_BINS),L48PIP,W48PIP

* declaration of common block for pi-
      REAL          H01PIM        ,L01PIM,W01PIM
     &             ,H11PIM        ,L11PIM,W11PIM
     &             ,H12PIM        ,L12PIM,W12PIM
     &             ,H13PIM        ,L13PIM,W13PIM
     &             ,H14PIM        ,L14PIM,W14PIM
     &             ,H15PIM        ,L15PIM,W15PIM
     &             ,H16PIM        ,L16PIM,W16PIM
     &             ,H17PIM        ,L17PIM,W17PIM
     &             ,H18PIM        ,L18PIM,W18PIM
     &             ,H21PIM        ,L21PIM,W21PIM
     &             ,H22PIM        ,L22PIM,W22PIM
     &             ,H23PIM        ,L23PIM,W23PIM
     &             ,H24PIM        ,L24PIM,W24PIM
     &             ,H25PIM        ,L25PIM,W25PIM
     &             ,H26PIM        ,L26PIM,W26PIM
     &             ,H27PIM        ,L27PIM,W27PIM
     &             ,H28PIM        ,L28PIM,W28PIM
     &             ,H31PIM        ,L31PIM,W31PIM
     &             ,H32PIM        ,L32PIM,W32PIM
     &             ,H33PIM        ,L33PIM,W33PIM
     &             ,H34PIM        ,L34PIM,W34PIM
     &             ,H35PIM        ,L35PIM,W35PIM
     &             ,H36PIM        ,L36PIM,W36PIM
     &             ,H37PIM        ,L37PIM,W37PIM
     &             ,H38PIM        ,L38PIM,W38PIM
     &             ,H41PIM        ,L41PIM,W41PIM
     &             ,H42PIM        ,L42PIM,W42PIM
     &             ,H43PIM        ,L43PIM,W43PIM
     &             ,H44PIM        ,L44PIM,W44PIM
     &             ,H45PIM        ,L45PIM,W45PIM
     &             ,H46PIM        ,L46PIM,W46PIM
     &             ,H47PIM        ,L47PIM,W47PIM
     &             ,H48PIM        ,L48PIM,W48PIM
      !             histo  # bins  lowlim binwid
      COMMON/RHIPIM/H01PIM(N_BINS),L01PIM,W01PIM
     &             ,H11PIM(N_BINS),L11PIM,W11PIM
     &             ,H12PIM(N_BINS),L12PIM,W12PIM
     &             ,H13PIM(N_BINS),L13PIM,W13PIM
     &             ,H14PIM(N_BINS),L14PIM,W14PIM
     &             ,H15PIM(N_BINS),L15PIM,W15PIM
     &             ,H16PIM(N_BINS),L16PIM,W16PIM
     &             ,H17PIM(N_BINS),L17PIM,W17PIM
     &             ,H18PIM(N_BINS),L18PIM,W18PIM
     &             ,H21PIM(N_BINS),L21PIM,W21PIM
     &             ,H22PIM(N_BINS),L22PIM,W22PIM
     &             ,H23PIM(N_BINS),L23PIM,W23PIM
     &             ,H24PIM(N_BINS),L24PIM,W24PIM
     &             ,H25PIM(N_BINS),L25PIM,W25PIM
     &             ,H26PIM(N_BINS),L26PIM,W26PIM
     &             ,H27PIM(N_BINS),L27PIM,W27PIM
     &             ,H28PIM(N_BINS),L28PIM,W28PIM
     &             ,H31PIM(N_BINS),L31PIM,W31PIM
     &             ,H32PIM(N_BINS),L32PIM,W32PIM
     &             ,H33PIM(N_BINS),L33PIM,W33PIM
     &             ,H34PIM(N_BINS),L34PIM,W34PIM
     &             ,H35PIM(N_BINS),L35PIM,W35PIM
     &             ,H36PIM(N_BINS),L36PIM,W36PIM
     &             ,H37PIM(N_BINS),L37PIM,W37PIM
     &             ,H38PIM(N_BINS),L38PIM,W38PIM
     &             ,H41PIM(N_BINS),L41PIM,W41PIM
     &             ,H42PIM(N_BINS),L42PIM,W42PIM
     &             ,H43PIM(N_BINS),L43PIM,W43PIM
     &             ,H44PIM(N_BINS),L44PIM,W44PIM
     &             ,H45PIM(N_BINS),L45PIM,W45PIM
     &             ,H46PIM(N_BINS),L46PIM,W46PIM
     &             ,H47PIM(N_BINS),L47PIM,W47PIM
     &             ,H48PIM(N_BINS),L48PIM,W48PIM

* declaration of common block for K+
      REAL          H01KAP        ,L01KAP,W01KAP
     &             ,H11KAP        ,L11KAP,W11KAP
     &             ,H12KAP        ,L12KAP,W12KAP
     &             ,H13KAP        ,L13KAP,W13KAP
     &             ,H14KAP        ,L14KAP,W14KAP
     &             ,H15KAP        ,L15KAP,W15KAP
     &             ,H16KAP        ,L16KAP,W16KAP
     &             ,H17KAP        ,L17KAP,W17KAP
     &             ,H18KAP        ,L18KAP,W18KAP
     &             ,H21KAP        ,L21KAP,W21KAP
     &             ,H22KAP        ,L22KAP,W22KAP
     &             ,H23KAP        ,L23KAP,W23KAP
     &             ,H24KAP        ,L24KAP,W24KAP
     &             ,H25KAP        ,L25KAP,W25KAP
     &             ,H26KAP        ,L26KAP,W26KAP
     &             ,H27KAP        ,L27KAP,W27KAP
     &             ,H28KAP        ,L28KAP,W28KAP
     &             ,H31KAP        ,L31KAP,W31KAP
     &             ,H32KAP        ,L32KAP,W32KAP
     &             ,H33KAP        ,L33KAP,W33KAP
     &             ,H34KAP        ,L34KAP,W34KAP
     &             ,H35KAP        ,L35KAP,W35KAP
     &             ,H36KAP        ,L36KAP,W36KAP
     &             ,H37KAP        ,L37KAP,W37KAP
     &             ,H38KAP        ,L38KAP,W38KAP
     &             ,H41KAP        ,L41KAP,W41KAP
     &             ,H42KAP        ,L42KAP,W42KAP
     &             ,H43KAP        ,L43KAP,W43KAP
     &             ,H44KAP        ,L44KAP,W44KAP
     &             ,H45KAP        ,L45KAP,W45KAP
     &             ,H46KAP        ,L46KAP,W46KAP
     &             ,H47KAP        ,L47KAP,W47KAP
     &             ,H48KAP        ,L48KAP,W48KAP
      !             histo  # bins  lowlim binwid
      COMMON/RHIKAP/H01KAP(N_BINS),L01KAP,W01KAP
     &             ,H11KAP(N_BINS),L11KAP,W11KAP
     &             ,H12KAP(N_BINS),L12KAP,W12KAP
     &             ,H13KAP(N_BINS),L13KAP,W13KAP
     &             ,H14KAP(N_BINS),L14KAP,W14KAP
     &             ,H15KAP(N_BINS),L15KAP,W15KAP
     &             ,H16KAP(N_BINS),L16KAP,W16KAP
     &             ,H17KAP(N_BINS),L17KAP,W17KAP
     &             ,H18KAP(N_BINS),L18KAP,W18KAP
     &             ,H21KAP(N_BINS),L21KAP,W21KAP
     &             ,H22KAP(N_BINS),L22KAP,W22KAP
     &             ,H23KAP(N_BINS),L23KAP,W23KAP
     &             ,H24KAP(N_BINS),L24KAP,W24KAP
     &             ,H25KAP(N_BINS),L25KAP,W25KAP
     &             ,H26KAP(N_BINS),L26KAP,W26KAP
     &             ,H27KAP(N_BINS),L27KAP,W27KAP
     &             ,H28KAP(N_BINS),L28KAP,W28KAP
     &             ,H31KAP(N_BINS),L31KAP,W31KAP
     &             ,H32KAP(N_BINS),L32KAP,W32KAP
     &             ,H33KAP(N_BINS),L33KAP,W33KAP
     &             ,H34KAP(N_BINS),L34KAP,W34KAP
     &             ,H35KAP(N_BINS),L35KAP,W35KAP
     &             ,H36KAP(N_BINS),L36KAP,W36KAP
     &             ,H37KAP(N_BINS),L37KAP,W37KAP
     &             ,H38KAP(N_BINS),L38KAP,W38KAP
     &             ,H41KAP(N_BINS),L41KAP,W41KAP
     &             ,H42KAP(N_BINS),L42KAP,W42KAP
     &             ,H43KAP(N_BINS),L43KAP,W43KAP
     &             ,H44KAP(N_BINS),L44KAP,W44KAP
     &             ,H45KAP(N_BINS),L45KAP,W45KAP
     &             ,H46KAP(N_BINS),L46KAP,W46KAP
     &             ,H47KAP(N_BINS),L47KAP,W47KAP
     &             ,H48KAP(N_BINS),L48KAP,W48KAP

* declaration of common block for K-
      REAL          H01KAM        ,L01KAM,W01KAM
     &             ,H11KAM        ,L11KAM,W11KAM
     &             ,H12KAM        ,L12KAM,W12KAM
     &             ,H13KAM        ,L13KAM,W13KAM
     &             ,H14KAM        ,L14KAM,W14KAM
     &             ,H15KAM        ,L15KAM,W15KAM
     &             ,H16KAM        ,L16KAM,W16KAM
     &             ,H17KAM        ,L17KAM,W17KAM
     &             ,H18KAM        ,L18KAM,W18KAM
     &             ,H21KAM        ,L21KAM,W21KAM
     &             ,H22KAM        ,L22KAM,W22KAM
     &             ,H23KAM        ,L23KAM,W23KAM
     &             ,H24KAM        ,L24KAM,W24KAM
     &             ,H25KAM        ,L25KAM,W25KAM
     &             ,H26KAM        ,L26KAM,W26KAM
     &             ,H27KAM        ,L27KAM,W27KAM
     &             ,H28KAM        ,L28KAM,W28KAM
     &             ,H31KAM        ,L31KAM,W31KAM
     &             ,H32KAM        ,L32KAM,W32KAM
     &             ,H33KAM        ,L33KAM,W33KAM
     &             ,H34KAM        ,L34KAM,W34KAM
     &             ,H35KAM        ,L35KAM,W35KAM
     &             ,H36KAM        ,L36KAM,W36KAM
     &             ,H37KAM        ,L37KAM,W37KAM
     &             ,H38KAM        ,L38KAM,W38KAM
     &             ,H41KAM        ,L41KAM,W41KAM
     &             ,H42KAM        ,L42KAM,W42KAM
     &             ,H43KAM        ,L43KAM,W43KAM
     &             ,H44KAM        ,L44KAM,W44KAM
     &             ,H45KAM        ,L45KAM,W45KAM
     &             ,H46KAM        ,L46KAM,W46KAM
     &             ,H47KAM        ,L47KAM,W47KAM
     &             ,H48KAM        ,L48KAM,W48KAM
      !             histo  # bins  lowlim binwid
      COMMON/RHIKAM/H01KAM(N_BINS),L01KAM,W01KAM
     &             ,H11KAM(N_BINS),L11KAM,W11KAM
     &             ,H12KAM(N_BINS),L12KAM,W12KAM
     &             ,H13KAM(N_BINS),L13KAM,W13KAM
     &             ,H14KAM(N_BINS),L14KAM,W14KAM
     &             ,H15KAM(N_BINS),L15KAM,W15KAM
     &             ,H16KAM(N_BINS),L16KAM,W16KAM
     &             ,H17KAM(N_BINS),L17KAM,W17KAM
     &             ,H18KAM(N_BINS),L18KAM,W18KAM
     &             ,H21KAM(N_BINS),L21KAM,W21KAM
     &             ,H22KAM(N_BINS),L22KAM,W22KAM
     &             ,H23KAM(N_BINS),L23KAM,W23KAM
     &             ,H24KAM(N_BINS),L24KAM,W24KAM
     &             ,H25KAM(N_BINS),L25KAM,W25KAM
     &             ,H26KAM(N_BINS),L26KAM,W26KAM
     &             ,H27KAM(N_BINS),L27KAM,W27KAM
     &             ,H28KAM(N_BINS),L28KAM,W28KAM
     &             ,H31KAM(N_BINS),L31KAM,W31KAM
     &             ,H32KAM(N_BINS),L32KAM,W32KAM
     &             ,H33KAM(N_BINS),L33KAM,W33KAM
     &             ,H34KAM(N_BINS),L34KAM,W34KAM
     &             ,H35KAM(N_BINS),L35KAM,W35KAM
     &             ,H36KAM(N_BINS),L36KAM,W36KAM
     &             ,H37KAM(N_BINS),L37KAM,W37KAM
     &             ,H38KAM(N_BINS),L38KAM,W38KAM
     &             ,H41KAM(N_BINS),L41KAM,W41KAM
     &             ,H42KAM(N_BINS),L42KAM,W42KAM
     &             ,H43KAM(N_BINS),L43KAM,W43KAM
     &             ,H44KAM(N_BINS),L44KAM,W44KAM
     &             ,H45KAM(N_BINS),L45KAM,W45KAM
     &             ,H46KAM(N_BINS),L46KAM,W46KAM
     &             ,H47KAM(N_BINS),L47KAM,W47KAM
     &             ,H48KAM(N_BINS),L48KAM,W48KAM

* declaration of common block for p+
      REAL          H01PRO        ,L01PRO,W01PRO
     &             ,H11PRO        ,L11PRO,W11PRO
     &             ,H12PRO        ,L12PRO,W12PRO
     &             ,H13PRO        ,L13PRO,W13PRO
     &             ,H14PRO        ,L14PRO,W14PRO
     &             ,H15PRO        ,L15PRO,W15PRO
     &             ,H16PRO        ,L16PRO,W16PRO
     &             ,H17PRO        ,L17PRO,W17PRO
     &             ,H18PRO        ,L18PRO,W18PRO
     &             ,H21PRO        ,L21PRO,W21PRO
     &             ,H22PRO        ,L22PRO,W22PRO
     &             ,H23PRO        ,L23PRO,W23PRO
     &             ,H24PRO        ,L24PRO,W24PRO
     &             ,H25PRO        ,L25PRO,W25PRO
     &             ,H26PRO        ,L26PRO,W26PRO
     &             ,H27PRO        ,L27PRO,W27PRO
     &             ,H28PRO        ,L28PRO,W28PRO
     &             ,H31PRO        ,L31PRO,W31PRO
     &             ,H32PRO        ,L32PRO,W32PRO
     &             ,H33PRO        ,L33PRO,W33PRO
     &             ,H34PRO        ,L34PRO,W34PRO
     &             ,H35PRO        ,L35PRO,W35PRO
     &             ,H36PRO        ,L36PRO,W36PRO
     &             ,H37PRO        ,L37PRO,W37PRO
     &             ,H38PRO        ,L38PRO,W38PRO
     &             ,H41PRO        ,L41PRO,W41PRO
     &             ,H42PRO        ,L42PRO,W42PRO
     &             ,H43PRO        ,L43PRO,W43PRO
     &             ,H44PRO        ,L44PRO,W44PRO
     &             ,H45PRO        ,L45PRO,W45PRO
     &             ,H46PRO        ,L46PRO,W46PRO
     &             ,H47PRO        ,L47PRO,W47PRO
     &             ,H48PRO        ,L48PRO,W48PRO
      !             histo  # bins  lowlim binwid
      COMMON/RHIPRO/H01PRO(N_BINS),L01PRO,W01PRO
     &             ,H11PRO(N_BINS),L11PRO,W11PRO
     &             ,H12PRO(N_BINS),L12PRO,W12PRO
     &             ,H13PRO(N_BINS),L13PRO,W13PRO
     &             ,H14PRO(N_BINS),L14PRO,W14PRO
     &             ,H15PRO(N_BINS),L15PRO,W15PRO
     &             ,H16PRO(N_BINS),L16PRO,W16PRO
     &             ,H17PRO(N_BINS),L17PRO,W17PRO
     &             ,H18PRO(N_BINS),L18PRO,W18PRO
     &             ,H21PRO(N_BINS),L21PRO,W21PRO
     &             ,H22PRO(N_BINS),L22PRO,W22PRO
     &             ,H23PRO(N_BINS),L23PRO,W23PRO
     &             ,H24PRO(N_BINS),L24PRO,W24PRO
     &             ,H25PRO(N_BINS),L25PRO,W25PRO
     &             ,H26PRO(N_BINS),L26PRO,W26PRO
     &             ,H27PRO(N_BINS),L27PRO,W27PRO
     &             ,H28PRO(N_BINS),L28PRO,W28PRO
     &             ,H31PRO(N_BINS),L31PRO,W31PRO
     &             ,H32PRO(N_BINS),L32PRO,W32PRO
     &             ,H33PRO(N_BINS),L33PRO,W33PRO
     &             ,H34PRO(N_BINS),L34PRO,W34PRO
     &             ,H35PRO(N_BINS),L35PRO,W35PRO
     &             ,H36PRO(N_BINS),L36PRO,W36PRO
     &             ,H37PRO(N_BINS),L37PRO,W37PRO
     &             ,H38PRO(N_BINS),L38PRO,W38PRO
     &             ,H41PRO(N_BINS),L41PRO,W41PRO
     &             ,H42PRO(N_BINS),L42PRO,W42PRO
     &             ,H43PRO(N_BINS),L43PRO,W43PRO
     &             ,H44PRO(N_BINS),L44PRO,W44PRO
     &             ,H45PRO(N_BINS),L45PRO,W45PRO
     &             ,H46PRO(N_BINS),L46PRO,W46PRO
     &             ,H47PRO(N_BINS),L47PRO,W47PRO
     &             ,H48PRO(N_BINS),L48PRO,W48PRO

* declaration of common block for n0
      REAL          H01NEU        ,L01NEU,W01NEU
     &             ,H11NEU        ,L11NEU,W11NEU
     &             ,H12NEU        ,L12NEU,W12NEU
     &             ,H13NEU        ,L13NEU,W13NEU
     &             ,H14NEU        ,L14NEU,W14NEU
     &             ,H15NEU        ,L15NEU,W15NEU
     &             ,H16NEU        ,L16NEU,W16NEU
     &             ,H17NEU        ,L17NEU,W17NEU
     &             ,H18NEU        ,L18NEU,W18NEU
     &             ,H21NEU        ,L21NEU,W21NEU
     &             ,H22NEU        ,L22NEU,W22NEU
     &             ,H23NEU        ,L23NEU,W23NEU
     &             ,H24NEU        ,L24NEU,W24NEU
     &             ,H25NEU        ,L25NEU,W25NEU
     &             ,H26NEU        ,L26NEU,W26NEU
     &             ,H27NEU        ,L27NEU,W27NEU
     &             ,H28NEU        ,L28NEU,W28NEU
     &             ,H31NEU        ,L31NEU,W31NEU
     &             ,H32NEU        ,L32NEU,W32NEU
     &             ,H33NEU        ,L33NEU,W33NEU
     &             ,H34NEU        ,L34NEU,W34NEU
     &             ,H35NEU        ,L35NEU,W35NEU
     &             ,H36NEU        ,L36NEU,W36NEU
     &             ,H37NEU        ,L37NEU,W37NEU
     &             ,H38NEU        ,L38NEU,W38NEU
     &             ,H41NEU        ,L41NEU,W41NEU
     &             ,H42NEU        ,L42NEU,W42NEU
     &             ,H43NEU        ,L43NEU,W43NEU
     &             ,H44NEU        ,L44NEU,W44NEU
     &             ,H45NEU        ,L45NEU,W45NEU
     &             ,H46NEU        ,L46NEU,W46NEU
     &             ,H47NEU        ,L47NEU,W47NEU
     &             ,H48NEU        ,L48NEU,W48NEU
      !             histo  # bins  lowlim binwid
      COMMON/RHINEU/H01NEU(N_BINS),L01NEU,W01NEU
     &             ,H11NEU(N_BINS),L11NEU,W11NEU
     &             ,H12NEU(N_BINS),L12NEU,W12NEU
     &             ,H13NEU(N_BINS),L13NEU,W13NEU
     &             ,H14NEU(N_BINS),L14NEU,W14NEU
     &             ,H15NEU(N_BINS),L15NEU,W15NEU
     &             ,H16NEU(N_BINS),L16NEU,W16NEU
     &             ,H17NEU(N_BINS),L17NEU,W17NEU
     &             ,H18NEU(N_BINS),L18NEU,W18NEU
     &             ,H21NEU(N_BINS),L21NEU,W21NEU
     &             ,H22NEU(N_BINS),L22NEU,W22NEU
     &             ,H23NEU(N_BINS),L23NEU,W23NEU
     &             ,H24NEU(N_BINS),L24NEU,W24NEU
     &             ,H25NEU(N_BINS),L25NEU,W25NEU
     &             ,H26NEU(N_BINS),L26NEU,W26NEU
     &             ,H27NEU(N_BINS),L27NEU,W27NEU
     &             ,H28NEU(N_BINS),L28NEU,W28NEU
     &             ,H31NEU(N_BINS),L31NEU,W31NEU
     &             ,H32NEU(N_BINS),L32NEU,W32NEU
     &             ,H33NEU(N_BINS),L33NEU,W33NEU
     &             ,H34NEU(N_BINS),L34NEU,W34NEU
     &             ,H35NEU(N_BINS),L35NEU,W35NEU
     &             ,H36NEU(N_BINS),L36NEU,W36NEU
     &             ,H37NEU(N_BINS),L37NEU,W37NEU
     &             ,H38NEU(N_BINS),L38NEU,W38NEU
     &             ,H41NEU(N_BINS),L41NEU,W41NEU
     &             ,H42NEU(N_BINS),L42NEU,W42NEU
     &             ,H43NEU(N_BINS),L43NEU,W43NEU
     &             ,H44NEU(N_BINS),L44NEU,W44NEU
     &             ,H45NEU(N_BINS),L45NEU,W45NEU
     &             ,H46NEU(N_BINS),L46NEU,W46NEU
     &             ,H47NEU(N_BINS),L47NEU,W47NEU
     &             ,H48NEU(N_BINS),L48NEU,W48NEU
