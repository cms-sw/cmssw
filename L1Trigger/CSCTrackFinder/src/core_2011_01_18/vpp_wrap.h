void wrap
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

	unsigned mb1aValidp, unsigned mb1aQp, unsigned mb1aPhip, unsigned mb1aBendp,
	unsigned mb1bValidp, unsigned mb1bQp, unsigned mb1bPhip, unsigned mb1bBendp,
	unsigned mb1cValidp, unsigned mb1cQp, unsigned mb1cPhip, unsigned mb1cBendp,
	unsigned mb1dValidp, unsigned mb1dQp, unsigned mb1dPhip, unsigned mb1dBendp,

	unsigned& ptHp, unsigned& signHp, unsigned& modeMemHp, unsigned& etaPTHp, unsigned& FRHp, unsigned& phiHp,
	unsigned& ptMp, unsigned& signMp, unsigned& modeMemMp, unsigned& etaPTMp, unsigned& FRMp, unsigned& phiMp,
	unsigned& ptLp, unsigned& signLp, unsigned& modeMemLp, unsigned& etaPTLp, unsigned& FRLp, unsigned& phiLp,

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
);
