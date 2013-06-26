*
*       IRNMCP		run number
*       IEVMCP          event number
*       WGTMCP          event weight
*       XSECN           cross section equivalent
*       IFILTER         filter pattern
*       NVRMCP          number of additional variables
*       VARMCP(NMXMCP)  list of additional variables
*
* Note: VARMCP(1) = PARI(17)   (standard word) 
*       VARMCP(2) = MSTI(1)    (standard word) 
*       VARMCP(3) =            (reserved standard word)  
*       VARMCP(4) =            (reserved standard word)  
*       VARMCP(5) =            (reserved standard word) 
*
*       Words 6 and up are free for users    09.02.05

        INTEGER IVERCMS,IRNMCP,IEVMCP,IFILTER,NVRMCP,NMXMCP,NMCHED
     &,         NMCTOT
        REAL VARMCP,WGTMCP,XSECN
        PARAMETER (NMXMCP=200,NMCHED=7,NMCTOT=NMXMCP+NMCHED)
        COMMON /MC_PARAM/ IVERCMS,IRNMCP,IEVMCP,WGTMCP,XSECN,IFILTER
     &,                   NVRMCP,VARMCP(NMXMCP)

