cc ***************************************************** 
cc  the common blocks pinput and cwdidth are for input parameters 
cc  these parameters needed to be interfaced to other program  
      common/pinput/ hbcvm,hbcm,fmb,fmc,hbcp,wwpi,fbc,fb1,fb2,fb3,
     .  fb4,cbcp0,cbcp1,cbcp2,cbc1p1
   
cc *****************************************************
cc the following common blocks are for deduced parameters
cc these parameters needed not to be interfaced to other program 
      common/cdeduce/ ffmcfmb 
      common/cmass2/fmc2,fmb2,hbcvm2,hbcm2,hbcm5,hbcm4,hbcm3,fmb3,
     &	fmc3 
      common/dmass/ dhbcvm2,dhbcm2
      
