enum ParamTypes {
	kInt,
	kDouble,
	kLogical
};

// zero-dimensional (single-value) parameter
#define PARM_N0(type, block, name) \
	{ #name, &block.name, type, { { 0, 0 }, { 0, 0 }, { 0, 0 } } }

// one-dimensional parameter
#define PARM_N1_O0(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 0 }, { 0, 0 }, { 0, 0 } } }
#define PARM_N1_O1(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 1 }, { 0, 0 }, { 0, 0 } } }

// two-dimensional parameter
#define PARM_N2_O0(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 0 }, { sizeof block.name[0] / sizeof block.name[0][0], 0 }, { 0, 0 } } }
#define PARM_N2_O1(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 1 }, { sizeof block.name[0] / sizeof block.name[0][0], 1 }, { 0, 0 } } }

// three-dimensional parameter
#define PARM_N3_O0(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 0 }, { sizeof block.name[0] / sizeof block.name[0][0], 0 }, { sizeof block.name[0][0] / sizeof block.name[0][0][0], 0 } } }
#define PARM_N3_O1(type, block, name) \
	{ #name, &block.name, type, { { sizeof block.name / sizeof block.name[0], 1 }, { sizeof block.name[0] / sizeof block.name[0][0], 1 }, { sizeof block.name[0][0] / sizeof block.name[0][0][0], 1 } } }

// end
#define PARM_LAST { 0, }

struct ConfigParam {
	const char		*name;
	void			*ptr;
	ParamTypes		type;
	struct {
		unsigned short	size;
		unsigned short	offset;
	} dim[3];
} static const configParams[] = {
	PARM_N0   (kDouble,  hw6202, DXRCYL),
	PARM_N0   (kDouble,  hw6202, DXRSPH),
	PARM_N0   (kDouble,  hw6202, DXZMAX),
	PARM_N0   (kInt,     hw6202, LRSUSY),
	PARM_N1_O1(kDouble,  hw6202, VIPWID),
	PARM_N0   (kLogical, hw6202, WZRFR),
	PARM_N0   (kDouble,  hw6203, AVABW),
	PARM_N0   (kLogical, hw6203, NEGWTS),
	PARM_N0   (kInt,     hw6300, IOPSH),
	PARM_N0   (kInt,     hw6300, IOPSTP),
	PARM_N0   (kLogical, hw6300, OPTM),
	PARM_N0   (kLogical, hw6500, PRESPL),
	PARM_N1_O1(kDouble,  hwbosc, ENHANC),
	PARM_N0   (kDouble,  hwbosc, GAMMAX),
	PARM_N0   (kInt,     hwbosc, IOPHIG),
	PARM_N1_O1(kInt,     hwbosc, MODBOS),
	PARM_N0   (kLogical, hwbrch, BREIT),
	PARM_N0   (kLogical, hwbrch, USECMF),
	PARM_N0   (kDouble,  hwdist, EXAG),
	PARM_N0   (kInt,     hwdist, IOPDKL),
	PARM_N0   (kInt,     hwdist, MAXDKL),
	PARM_N0   (kInt,     hwdist, MIXING),
	PARM_N0   (kInt,     hwdist, PIPSMR),
	PARM_N0   (kDouble,  hwdist, PLTCUT),
	PARM_N0   (kDouble,  hwdist, VMIN2),
	PARM_N1_O1(kDouble,  hwdist, XMIX),
	PARM_N1_O1(kDouble,  hwdist, YMIX),
	PARM_N0   (kLogical, hwdspn, FOURB),
	PARM_N0   (kLogical, hwdspn, SYSPIN),
	PARM_N0   (kLogical, hwdspn, THREEB),
	PARM_N0   (kDouble,  hwevnt, AVWGT),
	PARM_N0   (kInt,     hwevnt, LWEVT),
	PARM_N0   (kInt,     hwevnt, MAXER),
	PARM_N0   (kInt,     hwevnt, MAXPR),
	PARM_N0   (kInt,     hwevnt, NOWGT),
	PARM_N0   (kDouble,  hwevnt, WGTMAX),
	PARM_N0   (kLogical, hwgupr, LHGLSF),
	PARM_N0   (kLogical, hwgupr, LHSOFT),
	PARM_N0   (kDouble,  hwhard, ASFIXD),
	PARM_N0   (kInt,     hwhard, BGSHAT),
	PARM_N0   (kInt,     hwhard, DURHAM),
	PARM_N0   (kDouble,  hwhard, EMMAX),
	PARM_N0   (kDouble,  hwhard, EMMIN),
	PARM_N0   (kDouble,  hwhard, EMPOW),
	PARM_N1_O1(kDouble,  hwhard, EPOLN),
	PARM_N0   (kInt,     hwhard, IAPHIG),
	PARM_N1_O1(kInt,     hwhard, IBRN),
	PARM_N0   (kInt,     hwhard, IBSH),
	PARM_N0   (kInt,     hwhard, IHPRO),
	PARM_N0   (kDouble,  hwhard, OMEGA0),
	PARM_N0   (kDouble,  hwhard, PHOMAS),
	PARM_N1_O1(kDouble,  hwhard, PPOLN),
	PARM_N0   (kDouble,  hwhard, PTMAX),
	PARM_N0   (kDouble,  hwhard, PTMIN),
	PARM_N0   (kDouble,  hwhard, PTPOW),
	PARM_N0   (kDouble,  hwhard, Q2MAX),
	PARM_N0   (kDouble,  hwhard, Q2MIN),
	PARM_N0   (kDouble,  hwhard, Q2POW),
	PARM_N0   (kDouble,  hwhard, Q2WWMN),
	PARM_N0   (kDouble,  hwhard, Q2WWMX),
	PARM_N0   (kDouble,  hwhard, QLIM),
	PARM_N0   (kDouble,  hwhard, THMAX),
	PARM_N0   (kDouble,  hwhard, TMNISR),
	PARM_N0   (kDouble,  hwhard, WHMIN),
	PARM_N0   (kDouble,  hwhard, Y4JT),
	PARM_N0   (kDouble,  hwhard, YBMAX),
	PARM_N0   (kDouble,  hwhard, YBMIN),
	PARM_N0   (kDouble,  hwhard, YJMAX),
	PARM_N0   (kDouble,  hwhard, YJMIN),
	PARM_N0   (kDouble,  hwhard, YWWMAX),
	PARM_N0   (kDouble,  hwhard, YWWMIN),
	PARM_N0   (kDouble,  hwhard, ZJMAX),
	PARM_N0   (kDouble,  hwhard, ZMXISR),
	PARM_N0   (kDouble,  hwminb, PMBK1),
	PARM_N0   (kDouble,  hwminb, PMBK2),
	PARM_N0   (kDouble,  hwminb, PMBM1),
	PARM_N0   (kDouble,  hwminb, PMBM2),
	PARM_N0   (kDouble,  hwminb, PMBN1),
	PARM_N0   (kDouble,  hwminb, PMBN2),
	PARM_N0   (kDouble,  hwminb, PMBN3),
	PARM_N0   (kDouble,  hwminb, PMBP1),
	PARM_N0   (kDouble,  hwminb, PMBP2),
	PARM_N0   (kDouble,  hwminb, PMBP3),
	PARM_N2_O1(kDouble,  hwpram, AFCH),
	PARM_N0   (kDouble,  hwpram, ALPHEM),
	PARM_N0   (kLogical, hwpram, AZSOFT),
	PARM_N0   (kLogical, hwpram, AZSPIN),
	PARM_N0   (kDouble,  hwpram, B1LIM),
	PARM_N0   (kDouble,  hwpram, BTCLM),
	PARM_N1_O1(kInt,     hwpram, CLDIR),
	PARM_N0   (kDouble,  hwpram, CLMAX),
	PARM_N0   (kDouble,  hwpram, CLPOW),
	PARM_N1_O1(kDouble,  hwpram, CLSMR),
	PARM_N0   (kDouble,  hwpram, CSPEED),
	PARM_N0   (kDouble,  hwpram, ENSOF),
	PARM_N0   (kDouble,  hwpram, ET2MIX),
	PARM_N0   (kDouble,  hwpram, ETAMIX),
	PARM_N0   (kDouble,  hwpram, F0MIX),
	PARM_N0   (kDouble,  hwpram, F1MIX),
	PARM_N0   (kDouble,  hwpram, F2MIX),
	PARM_N0   (kDouble,  hwpram, GAMW),
	PARM_N0   (kDouble,  hwpram, GAMZ),
	PARM_N0   (kDouble,  hwpram, GAMZP),
	PARM_N0   (kDouble,  hwpram, GCUTME),
	PARM_N0   (kDouble,  hwpram, GEV2NB),
	PARM_N0   (kDouble,  hwpram, H1MIX),
	PARM_N0   (kLogical, hwpram, HARDME),
	PARM_N1_O1(kInt,     hwpram, IOP4JT),
	PARM_N0   (kInt,     hwpram, IOPREM),
	PARM_N0   (kInt,     hwpram, IPRINT),
	PARM_N0   (kInt,     hwpram, ISPAC),
	PARM_N0   (kInt,     hwpram, LRSUD),
	PARM_N0   (kInt,     hwpram, LWSUD),
	PARM_N1_O1(kInt,     hwpram, MODPDF),
	PARM_N0   (kInt,     hwpram, NBTRY),
	PARM_N0   (kInt,     hwpram, NCOLO),
	PARM_N0   (kInt,     hwpram, NCTRY),
	PARM_N0   (kInt,     hwpram, NETRY),
	PARM_N0   (kInt,     hwpram, NFLAV),
	PARM_N0   (kLogical, hwpram, NOSPAC),
	PARM_N0   (kInt,     hwpram, NPRFMT),
	PARM_N0   (kInt,     hwpram, NSTRU),
	PARM_N0   (kInt,     hwpram, NSTRY),
	PARM_N0   (kInt,     hwpram, NZBIN),
	PARM_N0   (kDouble,  hwpram, OMHMIX),
	PARM_N0   (kDouble,  hwpram, PDIQK),
	PARM_N0   (kDouble,  hwpram, PH3MIX),
	PARM_N0   (kDouble,  hwpram, PHIMIX),
	PARM_N0   (kLogical, hwpram, PRNDEC),
	PARM_N0   (kLogical, hwpram, PRNDEF),
	PARM_N0   (kLogical, hwpram, PRNTEX),
	PARM_N0   (kLogical, hwpram, PRNWEB),
	PARM_N0   (kDouble,  hwpram, PRSOF),
	PARM_N0   (kLogical, hwpram, PRVTX),
	PARM_N1_O1(kDouble,  hwpram, PSPLT),
	PARM_N0   (kDouble,  hwpram, PTRMS),
	PARM_N0   (kDouble,  hwpram, QCDLAM),
	PARM_N0   (kDouble,  hwpram, QDIQK),
	PARM_N1_O1(kDouble,  hwpram, QFCH),
	PARM_N0   (kDouble,  hwpram, QSPAC),
	PARM_N0   (kDouble,  hwpram, SCABI),
	PARM_N0   (kLogical, hwpram, SOFTME),
	PARM_N0   (kDouble,  hwpram, SWEIN),
	PARM_N1_O1(kDouble,  hwpram, VCKM),
	PARM_N2_O1(kDouble,  hwpram, VFCH),
	PARM_N0   (kDouble,  hwpram, VGCUT),
	PARM_N0   (kDouble,  hwpram, VPCUT),
	PARM_N0   (kDouble,  hwpram, VQCUT),
	PARM_N0   (kDouble,  hwpram, ZBINM),
	PARM_N0   (kLogical, hwpram, ZPRIME),
	PARM_N0   (kInt,     hwproc, IPROC),
	PARM_N1_O0(kDouble,  hwprop, RLTIM),
	PARM_N1_O0(kDouble,  hwprop, RMASS),
	PARM_N1_O1(kLogical, hwprop, VTOCDK),
	PARM_N1_O1(kLogical, hwprop, VTORDK),
	PARM_N0   (kLogical, hwrpar, RPARTY),
	PARM_N0   (kLogical, hwsusy, SUSYIN),
	PARM_N0   (kInt,     hwuclu, CLRECO),
	PARM_N0   (kDouble,  hwuclu, PRECO),
	PARM_N0   (kDouble,  hwusud, ACCUR),
	PARM_N0   (kInt,     hwusud, INTER),
	PARM_N0   (kInt,     hwusud, NQEV),
	PARM_N0   (kInt,     hwusud, SUDORD),
	PARM_N0   (kDouble,  hwuwts, DECWT),
	PARM_N1_O1(kDouble,  hwuwts, PWT),
	PARM_N3_O0(kDouble,  hwuwts, REPWT),
	PARM_N0   (kDouble,  hwuwts, SNGWT),
	PARM_N1_O1(kDouble,  jmparm, JMRAD),
	PARM_N0   (kInt,     jmparm, JMUEO),
	PARM_N0   (kDouble,  jmparm, PTJIM),
	PARM_N3_O1 (kDouble,  hwrpar, LAMDA1),    // Added for RPV (orig. fomr L. Sonnenschein)
	PARM_N3_O1 (kDouble,  hwrpar, LAMDA2),    //  \"
	PARM_N3_O1 (kDouble,  hwrpar, LAMDA3),    //  \"
	PARM_LAST
};

#undef PARM_N0
#undef PARM_N1_O0
#undef PARM_N1_O1
#undef PARM_N2_O0
#undef PARM_N2_O1
#undef PARM_N3_O0
#undef PARM_N3_O1
#undef PARM_LAST
