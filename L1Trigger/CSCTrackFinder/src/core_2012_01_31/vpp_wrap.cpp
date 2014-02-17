#include "vpp_generated.h"

void vpp_generated_2012_01_31::wrap
(
	unsigned me1aValidp, unsigned me1aQp, unsigned me1aEtap, unsigned me1aPhip,	unsigned me1aCSCidp, unsigned me1aCLCTp,
	unsigned me1bValidp, unsigned me1bQp, unsigned me1bEtap, unsigned me1bPhip,	unsigned me1bCSCidp, unsigned me1bCLCTp,
	unsigned me1cValidp, unsigned me1cQp, unsigned me1cEtap, unsigned me1cPhip,	unsigned me1cCSCidp, unsigned me1cCLCTp,
	unsigned me1dValidp, unsigned me1dQp, unsigned me1dEtap, unsigned me1dPhip,	unsigned me1dCSCidp, unsigned me1dCLCTp,
	unsigned me1eValidp, unsigned me1eQp, unsigned me1eEtap, unsigned me1ePhip,	unsigned me1eCSCidp, unsigned me1eCLCTp,
	unsigned me1fValidp, unsigned me1fQp, unsigned me1fEtap, unsigned me1fPhip,	unsigned me1fCSCidp, unsigned me1fCLCTp,
			 					   				   				   	  			  
	unsigned me2aValidp, unsigned me2aQp, unsigned me2aEtap, unsigned me2aPhip,
	unsigned me2bValidp, unsigned me2bQp, unsigned me2bEtap, unsigned me2bPhip,
	unsigned me2cValidp, unsigned me2cQp, unsigned me2cEtap, unsigned me2cPhip,
			 					   				   				   	  			  
	unsigned me3aValidp, unsigned me3aQp, unsigned me3aEtap, unsigned me3aPhip,
	unsigned me3bValidp, unsigned me3bQp, unsigned me3bEtap, unsigned me3bPhip,
	unsigned me3cValidp, unsigned me3cQp, unsigned me3cEtap, unsigned me3cPhip,
			 					   				   				   	  			  
	unsigned me4aValidp, unsigned me4aQp, unsigned me4aEtap, unsigned me4aPhip,
	unsigned me4bValidp, unsigned me4bQp, unsigned me4bEtap, unsigned me4bPhip,
	unsigned me4cValidp, unsigned me4cQp, unsigned me4cEtap, unsigned me4cPhip,
			 					   									  									  
	unsigned mb1aValidp, unsigned mb1aQp,                    unsigned mb1aPhip, unsigned mb1aBendp,
	unsigned mb1bValidp, unsigned mb1bQp,					 unsigned mb1bPhip,	unsigned mb1bBendp,
	unsigned mb1cValidp, unsigned mb1cQp,					 unsigned mb1cPhip,	unsigned mb1cBendp,
	unsigned mb1dValidp, unsigned mb1dQp,					 unsigned mb1dPhip,	unsigned mb1dBendp,

	unsigned& ptHp, unsigned& signHp, unsigned& modeMemHp, unsigned& etaPTHp, unsigned& FRHp, unsigned& phiHp, unsigned& phdiff_aHp,  unsigned& phdiff_bHp, 
	unsigned& ptMp, unsigned& signMp, unsigned& modeMemMp, unsigned& etaPTMp, unsigned& FRMp, unsigned& phiMp, unsigned& phdiff_aMp,  unsigned& phdiff_bMp,
	unsigned& ptLp, unsigned& signLp, unsigned& modeMemLp, unsigned& etaPTLp, unsigned& FRLp, unsigned& phiLp, unsigned& phdiff_aLp,  unsigned& phdiff_bLp,

	unsigned& me1idH, unsigned& me2idH, unsigned& me3idH, unsigned& me4idH, unsigned& mb1idH, unsigned& mb2idH,
	unsigned& me1idM, unsigned& me2idM, unsigned& me3idM, unsigned& me4idM, unsigned& mb1idM, unsigned& mb2idM,
	unsigned& me1idL, unsigned& me2idL, unsigned& me3idL, unsigned& me4idL, unsigned& mb1idL, unsigned& mb2idL,

	unsigned mneta12p, unsigned mneta13p, unsigned mneta23p, unsigned mneta24p, unsigned mneta34p, unsigned mneta12dtp, unsigned mneta14p,
	unsigned mxeta12p, unsigned mxeta13p, unsigned mxeta23p, unsigned mxeta24p, unsigned mxeta34p, unsigned mxeta12dtp, unsigned mxeta14p,
	unsigned etawn12p, unsigned etawn13p, unsigned etawn23p, unsigned etawn24p, unsigned etawn34p,				        unsigned etawn14p,
	unsigned mindphip, unsigned mindetap,

	unsigned mindeta_halo12p, unsigned maxdeta_halo12p, unsigned maxdphi_halo12p,
	unsigned mindeta_halo13p, unsigned maxdeta_halo13p, unsigned maxdphi_halo13p,

	unsigned mindeta_halo112p, unsigned maxdeta_halo112p, unsigned maxdphi_halo112p,
	unsigned mindeta_halo113p, unsigned maxdeta_halo113p, unsigned maxdphi_halo113p,
	unsigned mindphi_halop, unsigned mindeta_halop,

	unsigned straightp, unsigned curvedp,
	unsigned mb1a_phi_offp, unsigned mb1b_phi_offp,
	unsigned controlp
)
{

	// delayed CSC track stubs
	static ULLONG me1arr,me1brr,me1crr,me1drr,me1err,me1frr;
	static ULLONG me2arr,me2brr,me2crr,me3arr,me3brr,me3crr;
	static ULLONG me4arr,me4brr,me4crr;

	// delayed DT inputs for serialization
	//	static ULLONG mb1aPhir, mb1aQr;
	//	static ULLONG mb1bPhir, mb1bQr;

	// put delayed inputs into the core inputs (internal variables of vpp_generated)
	me1ar = me1arr;
	me1br = me1brr;
	me1cr = me1crr;
	me1dr = me1drr;
	me1er = me1err;
	me1fr = me1frr;
				 		    
	me2ar = me2arr;
	me2br = me2brr;
	me2cr = me2crr;
				 		    
	me3ar = me3arr;
	me3br = me3brr;
	me3cr = me3crr;
				 		    
	me4ar = me4arr;
	me4br = me4brr;
	me4cr = me4crr;

	// pack CSC inputs into delayed inputs. This delay is necessary to compensate for deserialization of DT inputs in the core
	me1arr = ((me1aValidp&1)<<27) | ((me1aCSCidp&0xf)<<23) | ((me1aQp&0xf)<<19) | ((me1aEtap&0x7f)<<12) |  (me1aPhip&0xfff) | ((me1aCLCTp&0xf)<<28);
	me1brr = ((me1bValidp&1)<<27) | ((me1bCSCidp&0xf)<<23) | ((me1bQp&0xf)<<19) | ((me1bEtap&0x7f)<<12) |  (me1bPhip&0xfff) | ((me1bCLCTp&0xf)<<28);
	me1crr = ((me1cValidp&1)<<27) | ((me1cCSCidp&0xf)<<23) | ((me1cQp&0xf)<<19) | ((me1cEtap&0x7f)<<12) |  (me1cPhip&0xfff) | ((me1cCLCTp&0xf)<<28);
	me1drr = ((me1dValidp&1)<<27) | ((me1dCSCidp&0xf)<<23) | ((me1dQp&0xf)<<19) | ((me1dEtap&0x7f)<<12) |  (me1dPhip&0xfff) | ((me1dCLCTp&0xf)<<28);
	me1err = ((me1eValidp&1)<<27) | ((me1eCSCidp&0xf)<<23) | ((me1eQp&0xf)<<19) | ((me1eEtap&0x7f)<<12) |  (me1ePhip&0xfff) | ((me1eCLCTp&0xf)<<28);
	me1frr = ((me1fValidp&1)<<27) | ((me1fCSCidp&0xf)<<23) | ((me1fQp&0xf)<<19) | ((me1fEtap&0x7f)<<12) |  (me1fPhip&0xfff) | ((me1fCLCTp&0xf)<<28);
		 		  				   	 		 			   		   				  	   				   		   			  				  						  	  	  	  
	me2arr = ((me2aValidp&1)<<23) |                          ((me2aQp&0xf)<<19) | ((me2aEtap&0x7f)<<12) |  (me2aPhip&0xfff);
	me2brr = ((me2bValidp&1)<<23) |                          ((me2bQp&0xf)<<19) | ((me2bEtap&0x7f)<<12) |  (me2bPhip&0xfff);
	me2crr = ((me2cValidp&1)<<23) |                          ((me2cQp&0xf)<<19) | ((me2cEtap&0x7f)<<12) |  (me2cPhip&0xfff);
		 		  				   								   				 	   				   		   			  			 
	me3arr = ((me3aValidp&1)<<23) |                          ((me3aQp&0xf)<<19) | ((me3aEtap&0x7f)<<12) |  (me3aPhip&0xfff);
	me3brr = ((me3bValidp&1)<<23) |                          ((me3bQp&0xf)<<19) | ((me3bEtap&0x7f)<<12) |  (me3bPhip&0xfff);
	me3crr = ((me3cValidp&1)<<23) |                          ((me3cQp&0xf)<<19) | ((me3cEtap&0x7f)<<12) |  (me3cPhip&0xfff);
		 		  				   	 		 				   	   				 	   				   		   			  				 						     	   
	me4arr = ((me4aValidp&1)<<23) |                          ((me4aQp&0xf)<<19) | ((me4aEtap&0x7f)<<12) |  (me4aPhip&0xfff);
	me4brr = ((me4bValidp&1)<<23) |                          ((me4bQp&0xf)<<19) | ((me4bEtap&0x7f)<<12) |  (me4bPhip&0xfff);
	me4crr = ((me4cValidp&1)<<23) |                          ((me4cQp&0xf)<<19) | ((me4cEtap&0x7f)<<12) |  (me4cPhip&0xfff);

			  			
	// serialize 4 DT inputs into 2 inputs for the core
	// if      (mb1aQp != 0) {mb1ap =             ((mb1aQp&0x7)<<12) | (mb1aPhip&0xfff); }
	// else if (mb1aQr != 0) {mb1ap = (1 << 15) | ((mb1aQr&0x7)<<12) | (mb1aPhir&0xfff);}
	// else mb1ap = 0;

	// if      (mb1bQp != 0) {mb1bp =             ((mb1bQp&0x7)<<12) | (mb1bPhip&0xfff); }
	// else if (mb1bQr != 0) {mb1bp = (1 << 15) | ((mb1bQr&0x7)<<12) | (mb1bPhir&0xfff);}
	// else mb1bp = 0;

	mb1ap = ((mb1aValidp&1) << 15) | ((mb1aQp&0x7)<<12) | (mb1aPhip&0xfff) | ((mb1aBendp&0x1f)<<16);
	mb1bp = ((mb1bValidp&1) << 15) | ((mb1bQp&0x7)<<12) | (mb1bPhip&0xfff) | ((mb1bBendp&0x1f)<<16);

	// mb1aPhir = mb1cPhip; mb1aQr = mb1cQp;
	// mb1bPhir = mb1dPhip; mb1bQr = mb1dQp;

	// put all other inputs into core inputs
	mneta12  = mneta12p; 
	mneta13  = mneta13p; 
	mneta23  = mneta23p; 
	mneta24  = mneta24p; 
	mneta34  = mneta34p; 
	mneta12dt  = mneta12dtp; 
	mneta14  = mneta14p; 
	 	   	   	   
	mxeta12  = mxeta12p; 
	mxeta13  = mxeta13p; 
	mxeta23  = mxeta23p; 
	mxeta24  = mxeta24p; 
	mxeta34  = mxeta34p; 
	mxeta12dt  = mxeta12dtp; 
	mxeta14  = mxeta14p; 
	 	   	   	   
	etawn12  = etawn12p; 
	etawn13  = etawn13p; 
	etawn23  = etawn23p; 
	etawn24  = etawn24p; 
	etawn34  = etawn34p; 
	etawn14  = etawn14p; 

	control = controlp;
	mindphi = mindphip;
	mindeta = mindetap;

	mindeta_halo12 = mindeta_halo12p;
	maxdeta_halo12 = maxdeta_halo12p;
	maxdphi_halo12 = maxdphi_halo12p;

	mindeta_halo13 = mindeta_halo13p;
	maxdeta_halo13 = maxdeta_halo13p;
	maxdphi_halo13 = maxdphi_halo13p;

	mindeta_halo112 = mindeta_halo112p;
	maxdeta_halo112 = maxdeta_halo112p;
	maxdphi_halo112 = maxdphi_halo112p;

	mindeta_halo113 = mindeta_halo113p;
	maxdeta_halo113 = maxdeta_halo113p;
	maxdphi_halo113 = maxdphi_halo113p;

	mindphi_halo = mindphi_halop; 
	mindeta_halo = mindeta_halop;

	straight = straightp;
	curved = curvedp;
	mb1a_phi_off = mb1a_phi_offp;
	mb1b_phi_off = mb1b_phi_offp;

	// drive clock up and down
	for (int i = 1; i >= 0; i--)
	{
		clkp = i;
		run();
	}
	
	// unpack track parameters
	int bp = 0;

	etaPTHp = pHp & 0x1f;
	etaPTMp = pMp & 0x1f;
	etaPTLp = pLp & 0x1f;

	bp += 5;

	modeMemHp = (pHp >> bp) & 0xf;
	modeMemMp = (pMp >> bp) & 0xf;
	modeMemLp = (pLp >> bp) & 0xf;

	bp += 4;

	signHp = (pHp >> bp) & 1;
	signMp = (pMp >> bp) & 1;
	signLp = (pLp >> bp) & 1;

	bp += 1;

	ptHp = (pHp >> bp) & 0x1fff;
	ptMp = (pMp >> bp) & 0x1fff;
	ptLp = (pLp >> bp) & 0x1fff;

	bp += 13;

	phiHp = (pHp >> bp) & 0x1f;
	phiMp = (pMp >> bp) & 0x1f;
	phiLp = (pLp >> bp) & 0x1f;

	bp += 5;

	FRHp = (pHp >> bp) & 1;
	FRMp = (pMp >> bp) & 1;
	FRLp = (pLp >> bp) & 1;

	//Retrieve ints from id codes
	bp = 0;

	me1idH = (idHp >> bp) & 0x1f;
	me1idM = (idMp >> bp) & 0x1f;
	me1idL = (idLp >> bp) & 0x1f;

	bp += 5;

	me2idH = (idHp >> bp) & 0xf;	
	me2idM = (idMp >> bp) & 0xf;
	me2idL = (idLp >> bp) & 0xf;

	bp += 4;

	me3idH = (idHp >> bp) & 0xf;	
	me3idM = (idMp >> bp) & 0xf;
	me3idL = (idLp >> bp) & 0xf;

	bp += 4;

	me4idH = (idHp >> bp) & 0xf;	
	me4idM = (idMp >> bp) & 0xf;
	me4idL = (idLp >> bp) & 0xf;

	bp += 4;

	mb1idH = (idHp >> bp) & 0x1f;
	mb1idM = (idMp >> bp) & 0x1f;
	mb1idL = (idLp >> bp) & 0x1f;

	mb2idH = 0;
	mb2idM = 0;
	mb2idL = 0;

	phdiff_aHp = phdiff_aH;
	phdiff_aMp = phdiff_aM;	 
	phdiff_aLp = phdiff_aL;
	phdiff_bHp = phdiff_bH;
	phdiff_bMp = phdiff_bM;
	phdiff_bLp = phdiff_bL;


}
